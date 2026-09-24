"""V5-lineage-bound streaming training for the Joint V3 learner.

The V5 bundle is the sole row authority.  Safe inputs, private labels and
source lineage are joined by authenticated row ordinals and payload hashes;
manifest list position is never used as identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .benchmark_v3_unlabeled_step_view import (
    BenchmarkV3UnlabeledStepView,
    UnlabeledIntrinsicBRepGraph,
)
from .brep_tensor_cache_v4_unlabeled import (
    AuthenticatedUnlabeledTrainingLabel,
    _TRAINING_LABEL_FACTORY_TOKEN,
    _build_cache_owned_unlabeled_step_view,
    _strict_json,
    _tensor_matches,
)
from .brep_tensor_cache_v5_lineage import load_lineage_index_v1
from .joint_interface_program_learner_v3 import (
    JointInterfaceProgramLearnerConfigV3,
    JointInterfaceProgramLearnerV3,
    load_externally_frozen_catalog_spec_v3,
    load_joint_interface_checkpoint_v3,
    save_joint_interface_checkpoint_v3,
)
from .joint_interface_training_v2 import (
    ARTIFACT_SCHEMA,
    RECEIPT_SCHEMA,
    SUITE_V2_ARTIFACT_PIN,
    BestEpochTrackerV2,
    _NpzLRU,
    _binding,
    _canonical_bytes,
    _clone_state,
    _file_sha,
    _process_memory,
    _run_streaming_epoch,
    _sha,
    group_row_ordinals_by_input,
    model_selection_key,
)


TRAINING_RECEIPT_SCHEMA_V3 = "joint_interface_training_receipt.v3"
TRAINING_ARTIFACT_SCHEMA_V3 = "joint_interface_training_artifact.v3"
TRAINING_CONTRACT_SCHEMA_V1 = "joint_interface_training_contract.v1"
TRAINING_CHECKPOINT_SCHEMA_V1 = "joint_interface_training_resume_checkpoint.v1"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_bound_json(path: Path, binding: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
  raw = path.read_bytes()
  observed = {"name": str(binding.get("name")), "bytes": len(raw),
              "sha256": hashlib.sha256(raw).hexdigest()}
  if observed != dict(binding):
    raise ValueError(f"{label} changed after V5 authority verification")
  value = _strict_json(raw, label=label)
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} is not an object")
  return value


def _ordinal_map(rows: Any, *, label: str) -> dict[int, Mapping[str, Any]]:
  if not isinstance(rows, list):
    raise ValueError(f"V5 {label} rows differ")
  result: dict[int, Mapping[str, Any]] = {}
  for row in rows:
    if not isinstance(row, Mapping) or type(row.get("row_ordinal")) is not int:
      raise ValueError(f"V5 {label} row ordinal differs")
    ordinal = int(row["row_ordinal"])
    if ordinal in result:
      raise ValueError(f"V5 {label} row ordinal is duplicated")
    result[ordinal] = row
  return result


@dataclass(frozen=True, slots=True)
class JointTrainingRowBindingV5:
  row_ordinal: int
  safe_input_sha256: str
  label_commitment_sha256: str
  target_program_index: int
  assignment_split: str
  case_id: str
  family_id: str
  body_sha1s: tuple[str, ...]
  graph_identity_sha256: str
  source_lineages: tuple[str, ...]
  source_row_ordinal: int
  source_row_payload_sha256: str
  lineage_row_payload_sha256: str

  def __post_init__(self) -> None:
    if type(self.row_ordinal) is not int or self.row_ordinal < 0:
      raise ValueError("V5 training row ordinal differs")
    if type(self.source_row_ordinal) is not int or self.source_row_ordinal < 0:
      raise ValueError("V5 training source row ordinal differs")
    if type(self.target_program_index) is not int or not 0 <= self.target_program_index < 19:
      raise ValueError("V5 training program index differs")
    for name in (
        "safe_input_sha256", "label_commitment_sha256",
        "graph_identity_sha256", "source_row_payload_sha256",
        "lineage_row_payload_sha256",
    ):
      if _SHA256.fullmatch(str(getattr(self, name))) is None:
        raise ValueError(f"V5 training {name} differs")
    if self.assignment_split not in {"train", "dev"}:
      raise ValueError("V5 training split differs")
    if not self.case_id.startswith("fusionv2_") or not self.family_id:
      raise ValueError("V5 training case/family differs")
    if not self.body_sha1s or not self.source_lineages:
      raise ValueError("V5 training lineage identity differs")


@dataclass(frozen=True, slots=True)
class LoadedJointTrainingIndexV5:
  rows: tuple[JointTrainingRowBindingV5, ...]
  authority_bindings: Mapping[str, Any]
  artifact_sha256: str
  index_payload_sha256: str


_STEP_IDENTITY_FACTORY_TOKEN = object()


class AuthenticatedJointStepIdentityV5:
  """Identity-only V5 projection for post-model STEP execution.

  This capability deliberately exposes neither training labels nor source
  assembly transforms.  It proves only the ordered full-STEP byte identities
  that produced one authenticated V5 safe view.
  """

  __slots__ = (
      "_row_ordinal", "_safe_input_sha256", "_bundle_artifact_sha256",
      "_safe_row_payload_sha256", "_full_graph_proofs",
  )

  def __init__(
      self, *, row_ordinal: int, safe_input_sha256: str,
      bundle_artifact_sha256: str, safe_row_payload_sha256: str,
      full_graph_proofs: Sequence[Mapping[str, Any]],
      _factory_token: object,
  ) -> None:
    if _factory_token is not _STEP_IDENTITY_FACTORY_TOKEN:
      raise TypeError("V5 STEP identities are dataset-factory-only")
    if (
        type(row_ordinal) is not int or row_ordinal < 0
        or _SHA256.fullmatch(str(safe_input_sha256)) is None
        or _SHA256.fullmatch(str(bundle_artifact_sha256)) is None
        or _SHA256.fullmatch(str(safe_row_payload_sha256)) is None
        or len(full_graph_proofs) != 2
        or any(
            not isinstance(value, Mapping)
            or _SHA256.fullmatch(str(value.get("step_sha256"))) is None
            or _SHA256.fullmatch(str(value.get("full_topology_sha256"))) is None
            for value in full_graph_proofs
        )
    ):
      raise ValueError("V5 STEP identity requires one ordered query pair")
    object.__setattr__(self, "_row_ordinal", int(row_ordinal))
    object.__setattr__(self, "_safe_input_sha256", str(safe_input_sha256))
    object.__setattr__(self, "_bundle_artifact_sha256", str(bundle_artifact_sha256))
    object.__setattr__(self, "_safe_row_payload_sha256", str(safe_row_payload_sha256))
    object.__setattr__(
        self, "_full_graph_proofs",
        tuple(MappingProxyType(dict(value)) for value in full_graph_proofs),
    )

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("authenticated V5 STEP identities are sealed")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated V5 STEP identities are not serializable")

  @property
  def row_ordinal(self) -> int:
    return self._row_ordinal

  @property
  def safe_input_sha256(self) -> str:
    return self._safe_input_sha256

  @property
  def bundle_artifact_sha256(self) -> str:
    return self._bundle_artifact_sha256

  @property
  def step_sha256s(self) -> tuple[str, str]:
    return tuple(str(row["step_sha256"]) for row in self._full_graph_proofs)  # type: ignore[return-value]

  @property
  def identity_payload_sha256(self) -> str:
    return _sha({
        "schema_version": "joint_v5_step_identity.v1",
        "row_ordinal": self._row_ordinal,
        "safe_input_sha256": self._safe_input_sha256,
        "bundle_artifact_sha256": self._bundle_artifact_sha256,
        "safe_row_payload_sha256": self._safe_row_payload_sha256,
        "full_graph_proofs": [dict(value) for value in self._full_graph_proofs],
    })


class StreamingJointDatasetV5:
  """Random-access V5 dataset authenticated by one external bundle pin."""

  def __init__(
      self, *, bundle_root: str | Path, bundle_artifact_sha256: str,
      expected_v5_producer_revision: str, shard_lru_capacity: int = 4,
  ) -> None:
    if _SHA256.fullmatch(bundle_artifact_sha256) is None:
      raise ValueError("V5 training bundle pin differs")
    if _REVISION.fullmatch(expected_v5_producer_revision) is None:
      raise ValueError("V5 training producer revision differs")
    root = Path(bundle_root)
    lineage_manifest = load_lineage_index_v1(
        root, expected_artifact_sha256=bundle_artifact_sha256
    )
    bundle_path = root / "artifact.json"
    if _file_sha(bundle_path) != bundle_artifact_sha256:
      raise ValueError("V5 training bundle changed after authority verification")
    bundle = _strict_json(bundle_path.read_bytes(), label="V5 training bundle")
    if not isinstance(bundle, Mapping):
      raise ValueError("V5 training bundle differs")
    producer = bundle.get("producer_binding")
    if not isinstance(producer, Mapping) or producer.get("revision") != expected_v5_producer_revision:
      raise ValueError("V5 training producer revision differs")
    if bundle.get("final_test_touched") is not False:
      raise ValueError("V5 training bundle touched final test")

    child_values: dict[str, Mapping[str, Any]] = {}
    child_manifests: dict[str, Mapping[str, Any]] = {}
    for key, directory in (("safe", "safe"), ("label", "labels")):
      artifact_binding = bundle.get(f"{key}_artifact")
      if not isinstance(artifact_binding, Mapping):
        raise ValueError(f"V5 training {key} artifact binding differs")
      artifact_path = root / str(artifact_binding["name"])
      child = _read_bound_json(
          artifact_path, artifact_binding, label=f"V5 training {key} artifact"
      )
      manifest_binding = child.get("manifest")
      if not isinstance(manifest_binding, Mapping):
        raise ValueError(f"V5 training {key} manifest binding differs")
      manifest_path = root / directory / str(manifest_binding["name"])
      child_values[key] = child
      child_manifests[key] = _read_bound_json(
          manifest_path, manifest_binding, label=f"V5 training {key} manifest"
      )

    safe_manifest = child_manifests["safe"]
    label_manifest = child_manifests["label"]
    input_domain = str(lineage_manifest.get("input_domain_sha256"))
    if (
        lineage_manifest.get("final_test_touched") is not False
        or lineage_manifest.get("family_partition_suite_v2_artifact_sha256")
        != SUITE_V2_ARTIFACT_PIN
        or safe_manifest.get("input_domain_sha256") != input_domain
        or label_manifest.get("input_domain_sha256") != input_domain
    ):
      raise ValueError("V5 training authority domain differs")

    lineage_by_ordinal = _ordinal_map(lineage_manifest.get("rows"), label="lineage")
    safe_by_ordinal = _ordinal_map(safe_manifest.get("rows"), label="safe")
    label_by_ordinal = _ordinal_map(label_manifest.get("rows"), label="label")
    ordinal_domain = set(lineage_by_ordinal)
    if ordinal_domain != set(safe_by_ordinal) or ordinal_domain != set(label_by_ordinal):
      raise ValueError("V5 training row ordinal domains differ")
    if ordinal_domain != set(range(len(ordinal_domain))):
      raise ValueError("V5 training row ordinal domain is not contiguous")

    rows: list[JointTrainingRowBindingV5] = []
    self._safe_rows: list[Mapping[str, Any]] = []
    self._label_rows: list[Mapping[str, Any]] = []
    for ordinal in range(len(ordinal_domain)):
      lineage = lineage_by_ordinal[ordinal]
      safe = safe_by_ordinal[ordinal]
      label = label_by_ordinal[ordinal]
      if (
          lineage.get("safe_row_payload_sha256") != safe.get("row_payload_sha256")
          or lineage.get("label_row_payload_sha256") != label.get("row_payload_sha256")
          or lineage.get("safe_input_sha256") != safe.get("input_sha256")
          or lineage.get("safe_input_sha256") != label.get("input_sha256")
          or lineage.get("label_commitment_sha256")
          != label.get("label_commitment_sha256")
      ):
        raise ValueError("V5 training explicit lineage/child row binding differs")
      rows.append(JointTrainingRowBindingV5(
          row_ordinal=ordinal,
          safe_input_sha256=str(lineage["safe_input_sha256"]),
          label_commitment_sha256=str(lineage["label_commitment_sha256"]),
          target_program_index=int(lineage["target_program_index"]),
          assignment_split=str(lineage["assignment_split"]),
          case_id=str(lineage["case_id"]), family_id=str(lineage["family_id"]),
          body_sha1s=tuple(str(value) for value in lineage["body_sha1s"]),
          graph_identity_sha256=str(lineage["graph_identity_sha256"]),
          source_lineages=tuple(str(value) for value in lineage["source_lineages"]),
          source_row_ordinal=int(lineage["source_row_ordinal"]),
          source_row_payload_sha256=str(lineage["source_row_payload_sha256"]),
          lineage_row_payload_sha256=str(lineage["row_payload_sha256"]),
      ))
      self._safe_rows.append(safe)
      self._label_rows.append(label)

    self._root = root
    self._safe_root = root / "safe"
    self._label_root = root / "labels"
    self._safe_shards = tuple(safe_manifest["shards"])
    self._label_shards = tuple(label_manifest["shards"])
    self._safe_lru = _NpzLRU(capacity=shard_lru_capacity)
    self._label_lru = _NpzLRU(capacity=shard_lru_capacity)
    self._safe_shard_for_row = self._locator(self._safe_shards, len(rows))
    self._label_shard_for_row = self._locator(self._label_shards, len(rows))
    authority = {
        "bundle_artifact_sha256": bundle_artifact_sha256,
        "safe_artifact_sha256": str(bundle["safe_artifact_sha256"]),
        "label_artifact_sha256": str(bundle["label_artifact_sha256"]),
        "lineage_artifact_sha256": str(bundle["lineage_artifact_sha256"]),
        "lineage_manifest_payload_sha256": str(
            lineage_manifest["manifest_payload_sha256"]
        ),
        "input_domain_sha256": input_domain,
        "family_partition_suite_v2_artifact_sha256": SUITE_V2_ARTIFACT_PIN,
        "v5_producer_revision": expected_v5_producer_revision,
        "join_policy": "explicit_authenticated_ordinal_and_payload_hash.v1",
    }
    self.index = LoadedJointTrainingIndexV5(
        rows=tuple(rows), authority_bindings=authority,
        artifact_sha256=bundle_artifact_sha256,
        index_payload_sha256=str(lineage_manifest["manifest_payload_sha256"]),
    )
    self._groups = group_row_ordinals_by_input(self.index.rows)  # type: ignore[arg-type]

  @staticmethod
  def _locator(
      shards: Sequence[Mapping[str, Any]], row_count: int,
  ) -> tuple[int, ...]:
    values = [-1] * row_count
    for shard_index, shard in enumerate(shards):
      start, count = int(shard["row_start"]), int(shard["row_count"])
      for ordinal in range(start, start + count):
        if ordinal >= row_count or values[ordinal] != -1:
          raise ValueError("V5 training shard row domain differs")
        values[ordinal] = shard_index
    if any(value < 0 for value in values):
      raise ValueError("V5 training shard row domain is incomplete")
    return tuple(values)

  @property
  def input_domain_sha256(self) -> str:
    return str(self.index.authority_bindings["input_domain_sha256"])

  @property
  def row_count(self) -> int:
    return len(self.index.rows)

  def groups(self, split: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if split not in self._groups:
      raise ValueError("V5 training split differs")
    return self._groups[split]

  def close(self) -> None:
    self._safe_lru.close()
    self._label_lru.close()

  def load_view(self, ordinal: int) -> BenchmarkV3UnlabeledStepView:
    row = self._safe_rows[ordinal]
    shard = self._safe_shards[self._safe_shard_for_row[ordinal]]
    archive = self._safe_lru.get(
        self._safe_root / str(shard["name"]), expected_binding=shard
    )
    receipts = {str(item["name"]): item for item in row["tensors"]}
    arrays = {name: archive[name].copy() for name in receipts}
    if any(not _tensor_matches(receipts[name], array) for name, array in arrays.items()):
      raise ValueError("V5 training safe tensor differs from manifest")
    prefix = f"r{ordinal:08d}"
    graphs = tuple(
        UnlabeledIntrinsicBRepGraph(
            node_features=arrays[f"{prefix}_g{part}_nodes"],
            edge_index=arrays[f"{prefix}_g{part}_edge_index"],
            edge_features=arrays[f"{prefix}_g{part}_edge_features"],
        )
        for part in range(int(row["graph_count"]))
    )
    view = _build_cache_owned_unlabeled_step_view(graphs=graphs)
    binding = self.index.rows[ordinal]
    if (
        view.input_sha256 != binding.safe_input_sha256
        or view.receipt_payload() != row["view_receipt"]
    ):
      raise ValueError("V5 training safe view differs from lineage authority")
    return view

  def load_step_identity(self, ordinal: int) -> AuthenticatedJointStepIdentityV5:
    """Return the V5-authenticated ordered STEP identities for one safe row."""

    row = self._safe_rows[ordinal]
    binding = self.index.rows[ordinal]
    unsigned = dict(row)
    observed = unsigned.pop("row_payload_sha256", None)
    proofs = row.get("full_graph_proofs")
    if observed != _sha(unsigned):
      raise ValueError("V5 safe row identity commitment differs")
    if not isinstance(proofs, list) or len(proofs) != 2:
      raise ValueError("V5 safe row full-STEP identity domain differs")
    for proof in proofs:
      if (
          not isinstance(proof, Mapping)
          or proof.get("schema_version")
          != "benchmark_v3_full_step_graph_proof.v1"
          or proof.get("complete_component") is not True
          or _SHA256.fullmatch(str(proof.get("step_sha256"))) is None
          or _SHA256.fullmatch(str(proof.get("full_topology_sha256"))) is None
      ):
        raise ValueError("V5 safe row full-STEP proof differs")
    # load_view reopens the receipt-bound shard and proves the tensors still
    # match this exact safe row before an identity capability is issued.
    view = self.load_view(ordinal)
    if view.input_sha256 != binding.safe_input_sha256:
      raise ValueError("V5 STEP identity/safe view binding differs")
    return AuthenticatedJointStepIdentityV5(
        row_ordinal=ordinal,
        safe_input_sha256=binding.safe_input_sha256,
        bundle_artifact_sha256=self.index.artifact_sha256,
        safe_row_payload_sha256=str(observed),
        full_graph_proofs=proofs,
        _factory_token=_STEP_IDENTITY_FACTORY_TOKEN,
    )

  def load_label(self, ordinal: int) -> AuthenticatedUnlabeledTrainingLabel:
    row = self._label_rows[ordinal]
    shard = self._label_shards[self._label_shard_for_row[ordinal]]
    archive = self._label_lru.get(
        self._label_root / str(shard["name"]), expected_binding=shard
    )
    prefix = f"r{ordinal:08d}"
    names = (
        f"{prefix}_class", f"{prefix}_translation", f"{prefix}_rotation",
        f"{prefix}_mask", f"{prefix}_endpoint_a", f"{prefix}_endpoint_b",
    )
    arrays = {name: archive[name].copy() for name in names}
    receipts = {str(item["name"]): item for item in row["tensors"]}
    if set(receipts) != set(arrays) or any(
        not _tensor_matches(receipts[name], arrays[name]) for name in arrays
    ):
      raise ValueError("V5 training label tensor differs from manifest")
    label = AuthenticatedUnlabeledTrainingLabel(
        target_program_index=int(arrays[f"{prefix}_class"].reshape(-1)[0]),
        residual_translation=arrays[f"{prefix}_translation"],
        residual_rotation_vector=arrays[f"{prefix}_rotation"],
        residual_mask=bool(arrays[f"{prefix}_mask"].reshape(-1)[0]),
        endpoint_graph_node_indices=(
            arrays[f"{prefix}_endpoint_a"].tolist(),
            arrays[f"{prefix}_endpoint_b"].tolist(),
        ),
        _factory_token=_TRAINING_LABEL_FACTORY_TOKEN,
    )
    binding = self.index.rows[ordinal]
    if (
        label.commitment_sha256 != binding.label_commitment_sha256
        or label.target_program_index != binding.target_program_index
    ):
      raise ValueError("V5 training label differs from lineage authority")
    return label


def _write_atomic(path: Path, payload: bytes) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("xb") as stream:
      stream.write(payload)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


def write_recoverable_training_checkpoint_v1(
    directory: str | Path, *, contract_sha256: str, next_epoch: int,
    model_state: Mapping[str, Any], optimizer_state: Mapping[str, Any],
    best_state: Mapping[str, Any], best_epoch: int,
    best_key: Sequence[float], epochs_without_improvement: int,
    history: Sequence[Mapping[str, Any]],
) -> str:
  """Atomically advance a content-bound, development-only resume pointer."""

  if _SHA256.fullmatch(contract_sha256) is None:
    raise ValueError("training checkpoint contract differs")
  if type(next_epoch) is not int or next_epoch < 1:
    raise ValueError("training checkpoint next epoch differs")
  if type(best_epoch) is not int or not 0 <= best_epoch < next_epoch:
    raise ValueError("training checkpoint best epoch differs")
  if len(tuple(best_key)) != 4:
    raise ValueError("training checkpoint selection key differs")
  root = Path(directory)
  root.mkdir(parents=True, exist_ok=True)
  state_payload = {
      "model_state": dict(model_state),
      "optimizer_state": dict(optimizer_state),
      "best_state": dict(best_state),
      "torch_rng_state": torch.get_rng_state(),
  }
  temporary_state = root / f".training-state-{os.getpid()}.tmp"
  try:
    torch.save(state_payload, temporary_state)
    state_sha256 = _file_sha(temporary_state)
    state_name = f"training-state-epoch-{next_epoch:04d}-{state_sha256[:16]}.pt"
    state_path = root / state_name
    if state_path.exists():
      if _file_sha(state_path) != state_sha256:
        raise ValueError("training checkpoint state name collision")
      temporary_state.unlink()
    else:
      os.replace(temporary_state, state_path)
  except BaseException:
    temporary_state.unlink(missing_ok=True)
    raise
  state_binding = _binding(state_path)
  metadata: dict[str, Any] = {
      "schema_version": TRAINING_CHECKPOINT_SCHEMA_V1,
      "scope": "train_dev_only_no_final_access",
      "final_test_touched": False,
      "contract_sha256": contract_sha256,
      "next_epoch": next_epoch,
      "best_epoch": best_epoch,
      "best_key": [float(value) for value in best_key],
      "epochs_without_improvement": int(epochs_without_improvement),
      "history": [dict(row) for row in history],
      "state": state_binding,
  }
  metadata["checkpoint_payload_sha256"] = _sha(metadata)
  pointer = root / "training_checkpoint.json"
  _write_atomic(pointer, _canonical_bytes(metadata))
  for old_state in root.glob("training-state-epoch-*.pt"):
    if old_state != state_path:
      old_state.unlink(missing_ok=True)
  return _file_sha(pointer)


def load_recoverable_training_checkpoint_v1(
    directory: str | Path, *, expected_contract_sha256: str,
) -> Mapping[str, Any]:
  """Load a resume checkpoint only after its contract and bytes agree."""

  if _SHA256.fullmatch(expected_contract_sha256) is None:
    raise ValueError("training checkpoint expected contract differs")
  root = Path(directory)
  pointer = root / "training_checkpoint.json"
  raw = pointer.read_bytes()
  metadata = _strict_json(raw, label="training resume checkpoint")
  if not isinstance(metadata, Mapping):
    raise ValueError("training checkpoint is not an object")
  unsigned = dict(metadata)
  observed = unsigned.pop("checkpoint_payload_sha256", None)
  if (
      metadata.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_V1
      or metadata.get("scope") != "train_dev_only_no_final_access"
      or metadata.get("final_test_touched") is not False
      or observed != _sha(unsigned)
      or metadata.get("contract_sha256") != expected_contract_sha256
  ):
    raise ValueError("training checkpoint authority differs")
  binding = metadata.get("state")
  if not isinstance(binding, Mapping) or set(binding) != {"name", "bytes", "sha256"}:
    raise ValueError("training checkpoint state binding differs")
  state_path = root / str(binding["name"])
  if _binding(state_path) != dict(binding):
    raise ValueError("training checkpoint state bytes changed")
  state = torch.load(state_path, map_location="cpu", weights_only=True)
  if not isinstance(state, Mapping) or set(state) != {
      "model_state", "optimizer_state", "best_state", "torch_rng_state",
  }:
    raise ValueError("training checkpoint state schema differs")
  return {
      "metadata_sha256": hashlib.sha256(raw).hexdigest(),
      "state_binding": dict(binding),
      "next_epoch": int(metadata["next_epoch"]),
      "best_epoch": int(metadata["best_epoch"]),
      "best_key": tuple(float(value) for value in metadata["best_key"]),
      "epochs_without_improvement": int(metadata["epochs_without_improvement"]),
      "history": tuple(dict(row) for row in metadata["history"]),
      **dict(state),
  }


@dataclass(frozen=True, slots=True)
class JointInterfaceTrainingConfigV3:
  seed: int
  max_epochs: int
  learning_rate: float
  weight_decay: float
  gradient_clip_norm: float
  views_per_optimizer_step: int
  early_stop_patience: int
  device_policy: str
  model_config: JointInterfaceProgramLearnerConfigV3
  engineering_smoke: bool = False
  smoke_train_unique_views: int | None = None
  smoke_dev_unique_views: int | None = None

  def __post_init__(self) -> None:
    if type(self.seed) is not int or self.seed < 0:
      raise ValueError("V5 training seed differs")
    for name in ("max_epochs", "views_per_optimizer_step"):
      value = getattr(self, name)
      if type(value) is not int or value < 1:
        raise ValueError(f"V5 training {name} differs")
    if type(self.early_stop_patience) is not int or self.early_stop_patience < 0:
      raise ValueError("V5 training early-stop patience differs")
    for name in ("learning_rate", "gradient_clip_norm"):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0:
        raise ValueError(f"V5 training {name} differs")
    if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
      raise ValueError("V5 training weight decay differs")
    if self.device_policy != "reproducible_cpu.v1":
      raise ValueError("V5 training policy is reproducible CPU only")
    if type(self.model_config) is not JointInterfaceProgramLearnerConfigV3:
      raise TypeError("V5 training requires an explicit Joint V3 model config")
    limits = (self.smoke_train_unique_views, self.smoke_dev_unique_views)
    if self.engineering_smoke:
      if any(type(value) is not int or value < 1 for value in limits):
        raise ValueError("V5 engineering smoke requires positive view limits")
    elif any(value is not None for value in limits):
      raise ValueError("V5 smoke limits are engineering-smoke-only")


def _append_telemetry(path: Path, event: Mapping[str, Any]) -> None:
  line = _canonical_bytes(dict(event)) + b"\n"
  with path.open("ab") as stream:
    stream.write(line)
    stream.flush()
    os.fsync(stream.fileno())


def _checked_revision(root: Path) -> str:
  revision = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=root, text=True
  ).strip()
  dirty = subprocess.check_output(
      ["git", "status", "--porcelain", "--untracked-files=all"],
      cwd=root, text=True,
  ).strip()
  if _REVISION.fullmatch(revision) is None or dirty:
    raise ValueError("V5 training requires a clean exact revision")
  return revision


def _training_contract(
    *, revision: str, output: Path, dataset: StreamingJointDatasetV5,
    config: JointInterfaceTrainingConfigV3, catalog: Any,
) -> dict[str, Any]:
  root = Path(__file__).resolve().parent
  sources = (
      "joint_interface_training_v3.py", "joint_interface_training_v2.py",
      "joint_interface_program_learner_v3.py", "tools/train_joint_interface_v3_v5.py",
  )
  contract: dict[str, Any] = {
      "schema_version": TRAINING_CONTRACT_SCHEMA_V1,
      "scope": "p0_train_dev_single_seed_no_final_access",
      "final_test_touched": False,
      "output_name": output.name,
      "code": {
          "revision": revision,
          "source_sha256s": {name: _file_sha(root / name) for name in sources},
      },
      "data": dict(dataset.index.authority_bindings),
      "catalog": {
          "roster_sha256": catalog.roster.catalog_sha256,
          "spec_file_sha256": catalog.spec_file_sha256,
          "spec_payload_sha256": catalog.spec_payload_sha256,
      },
      "config": {
          **{key: value for key, value in asdict(config).items() if key != "model_config"},
          "model_config": asdict(config.model_config),
          "model_config_sha256": config.model_config.sha256,
      },
  }
  contract["contract_payload_sha256"] = _sha(contract)
  return contract


def train_joint_interface_v3_v5(
    output_directory: str | Path, *, dataset: StreamingJointDatasetV5,
    config: JointInterfaceTrainingConfigV3, catalog_spec_path: str | Path,
    expected_catalog_roster_sha256: str, expected_catalog_spec_file_sha256: str,
    resume: bool = False,
) -> str:
  """Train one p0 seed with best-epoch selection and epoch-safe recovery."""

  output = Path(output_directory)
  if output.exists():
    raise FileExistsError("V5 training output already exists")
  code_root = Path(__file__).resolve().parent
  revision = _checked_revision(code_root)
  catalog = load_externally_frozen_catalog_spec_v3(
      catalog_spec_path,
      expected_catalog_roster_sha256=expected_catalog_roster_sha256,
      expected_spec_file_sha256=expected_catalog_spec_file_sha256,
  )
  if config.model_config.expected_catalog_roster_sha256 != catalog.roster.catalog_sha256:
    raise ValueError("V5 training config/catalog binding differs")
  train_groups = dataset.groups("train")
  dev_groups = dataset.groups("dev")
  if config.engineering_smoke:
    train_groups = train_groups[:int(config.smoke_train_unique_views)]
    dev_groups = dev_groups[:int(config.smoke_dev_unique_views)]
  if not train_groups or not dev_groups:
    raise ValueError("V5 training requires nonempty train and dev groups")

  contract = _training_contract(
      revision=revision, output=output, dataset=dataset, config=config,
      catalog=catalog,
  )
  contract_sha256 = str(contract["contract_payload_sha256"])
  staging = output.parent / f".{output.name}.inprogress"
  contract_path = staging / "training_contract.json"
  telemetry_path = staging / "telemetry.jsonl"
  if resume:
    if not staging.is_dir():
      raise FileNotFoundError("V5 training resume directory does not exist")
    observed_contract = _strict_json(
        contract_path.read_bytes(), label="V5 training contract"
    )
    if observed_contract != contract:
      raise ValueError("V5 training resume contract differs")
  else:
    if staging.exists():
      raise FileExistsError("V5 training staging already exists; use resume")
    staging.mkdir(parents=True)
    _write_atomic(contract_path, _canonical_bytes(contract))

  torch.manual_seed(config.seed)
  np.random.seed(config.seed)
  torch.use_deterministic_algorithms(True, warn_only=False)
  torch.set_num_threads(1)
  cuda_available = bool(torch.cuda.is_available())
  cuda_observation = {
      "available": cuda_available,
      "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
      "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
      if cuda_available else [],
      "selected_device": "cpu", "policy": config.device_policy,
  }
  model = JointInterfaceProgramLearnerV3(config.model_config, seed=config.seed)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
  )
  tracker = BestEpochTrackerV2(patience=config.early_stop_patience)
  best_state: dict[str, torch.Tensor] | None = None
  history: list[dict[str, Any]] = []
  start_epoch = 0
  if resume:
    saved = load_recoverable_training_checkpoint_v1(
        staging, expected_contract_sha256=contract_sha256
    )
    model.load_state_dict(saved["model_state"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    best_state = {name: value.clone() for name, value in saved["best_state"].items()}
    tracker.best_epoch = int(saved["best_epoch"])
    tracker.best_key = tuple(saved["best_key"])
    tracker.epochs_without_improvement = int(saved["epochs_without_improvement"])
    history = [dict(row) for row in saved["history"]]
    start_epoch = int(saved["next_epoch"])
    torch.set_rng_state(saved["torch_rng_state"])
  _append_telemetry(telemetry_path, {
      "event": "training_resumed" if resume else "training_started",
      "revision": revision, "contract_sha256": contract_sha256,
      "seed": config.seed, "start_epoch": start_epoch,
      "train_rows": sum(len(rows) for _, rows in train_groups),
      "dev_rows": sum(len(rows) for _, rows in dev_groups),
      "train_unique_views": len(train_groups), "dev_unique_views": len(dev_groups),
      "device": cuda_observation, "timestamp_unix": time.time(),
  })

  for epoch in range(start_epoch, config.max_epochs):
    train_metrics = _run_streaming_epoch(
        model, dataset=dataset, groups=train_groups, catalog=catalog,
        optimizer=optimizer,
        views_per_optimizer_step=config.views_per_optimizer_step,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    with torch.no_grad():
      dev_metrics = _run_streaming_epoch(
          model, dataset=dataset, groups=dev_groups, catalog=catalog,
          optimizer=None,
          views_per_optimizer_step=config.views_per_optimizer_step,
          gradient_clip_norm=config.gradient_clip_norm,
      )
    improved, stop = tracker.observe(epoch, dev_metrics)
    if improved:
      best_state = _clone_state(model)
    epoch_row = {
        "epoch": epoch, "train": train_metrics, "dev": dev_metrics,
        "selection_key": list(model_selection_key(dev_metrics)),
        "improved": improved,
        "epochs_without_improvement": tracker.epochs_without_improvement,
        "early_stop_triggered": stop,
    }
    history.append(epoch_row)
    if best_state is None or tracker.best_epoch is None or tracker.best_key is None:
      raise RuntimeError("V5 training failed to select a best epoch")
    checkpoint_pin = write_recoverable_training_checkpoint_v1(
        staging, contract_sha256=contract_sha256, next_epoch=epoch + 1,
        model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        best_state=best_state, best_epoch=tracker.best_epoch,
        best_key=tracker.best_key,
        epochs_without_improvement=tracker.epochs_without_improvement,
        history=history,
    )
    _append_telemetry(telemetry_path, {
        "event": "epoch_complete", **epoch_row,
        "resume_checkpoint_sha256": checkpoint_pin,
        "timestamp_unix": time.time(),
    })
    if stop:
      break

  if best_state is None or tracker.best_epoch is None or tracker.best_key is None:
    raise RuntimeError("V5 training has no completed epoch")
  model.load_state_dict(best_state, strict=True)
  checkpoint = staging / "joint_interface_checkpoint_v3.pt"
  checkpoint_sha256 = save_joint_interface_checkpoint_v3(
      model.eval(), checkpoint, catalog_spec=catalog
  )
  loaded = load_joint_interface_checkpoint_v3(
      checkpoint, expected_checkpoint_sha256=checkpoint_sha256,
      catalog_spec_path=catalog_spec_path,
      expected_catalog_roster_sha256=expected_catalog_roster_sha256,
      expected_catalog_spec_file_sha256=expected_catalog_spec_file_sha256,
  )
  if loaded.config.sha256 != config.model_config.sha256:
    raise ValueError("V5 training checkpoint replay config differs")
  peak_working_set, working_set = _process_memory()
  _append_telemetry(telemetry_path, {
      "event": "training_complete", "best_epoch": tracker.best_epoch,
      "epochs_executed": len(history), "checkpoint_sha256": checkpoint_sha256,
      "timestamp_unix": time.time(),
  })
  receipt: dict[str, Any] = {
      "schema_version": TRAINING_RECEIPT_SCHEMA_V3,
      "scope": "p0_train_dev_single_seed_no_final_access",
      "final_test_touched": False,
      "performance_claim_eligible": False,
      "engineering_smoke": config.engineering_smoke,
      "data": {
          **dict(dataset.index.authority_bindings),
          "loaded_rows": dataset.row_count,
          "train_rows": sum(len(rows) for _, rows in train_groups),
          "dev_rows": sum(len(rows) for _, rows in dev_groups),
          "train_unique_views": len(train_groups),
          "dev_unique_views": len(dev_groups),
      },
      "catalog": contract["catalog"],
      "training": contract["config"],
      "device_observation": cuda_observation,
      "model_selection": {
          "rule": [
              "max_dev_joint_r5", "max_dev_program_r5",
              "max_dev_joint_r1", "min_dev_loss",
          ],
          "tie_policy": "earliest_epoch", "patience": config.early_stop_patience,
          "best_epoch": tracker.best_epoch,
          "best_selection_key": list(tracker.best_key),
          "saved_checkpoint_is_best_not_last": True,
          "epochs_executed": len(history),
      },
      "epochs": history,
      "checkpoint": {
          "schema_version": "joint_interface_program_checkpoint.v3",
          "name": checkpoint.name, "sha256": checkpoint_sha256,
          "optimizer_state_included": False,
      },
      "recovery": {
          "policy": "content_bound_epoch_checkpoint.v1",
          "resumed": resume,
          "contract_sha256": contract_sha256,
          "checkpoint_pointer": _binding(staging / "training_checkpoint.json"),
          "telemetry": _binding(telemetry_path),
      },
      "environment": {
          "python": platform.python_version(), "torch": torch.__version__,
          "numpy": np.__version__,
          "windows_peak_working_set_bytes": peak_working_set,
          "windows_working_set_bytes": working_set,
      },
      "code": contract["code"],
  }
  receipt["receipt_payload_sha256"] = _sha(receipt)
  receipt_path = staging / "training_receipt.json"
  _write_atomic(receipt_path, _canonical_bytes(receipt))
  artifact: dict[str, Any] = {
      "schema_version": TRAINING_ARTIFACT_SCHEMA_V3,
      "files": [
          _binding(path) for path in sorted(staging.iterdir(), key=lambda value: value.name)
          if path.is_file() and path.name != "artifact.json"
      ],
  }
  artifact["artifact_payload_sha256"] = _sha(artifact)
  artifact_path = staging / "artifact.json"
  _write_atomic(artifact_path, _canonical_bytes(artifact))
  pin = _file_sha(artifact_path)
  os.replace(staging, output)
  return pin
