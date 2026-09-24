"""Bounded V3 joint face-pair/program learner and formal proposal protocol.

V3 adds four trust boundaries around the V2 neural core:

* training uses one joint contrastive candidate set made from inference hard
  faces union the supervised face on each side, crossed with all 19 programs;
* model config, state and checkpoints bind an externally expected catalog
  roster hash rather than accepting a self-selected runtime roster;
* graph queries bind full graph bytes, per-node visible signatures, STEP and
  instance identity, plus an independently captured full-view frame receipt;
* the formal worker owns inference, sorting and top-K under one preemptive
  process deadline and returns only lightweight rows.

No target, selected-face marker or source endpoint enters ``forward``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .joint_interface_program_learner_v2 import (
    CATALOG_SCHEMA_VERSION,
    FORMAL_CATALOG_SIZE,
    PAIR_FRAME_POLICY_SHA256,
    FiniteProgramCatalogEntryV2,
    FiniteProgramCatalogRosterV2,
    FiniteProgramDescriptorV2,
    GraphProgramInputsV2,
    JointInterfaceProgramLearnerConfigV2,
    JointInterfaceProgramLearnerV2,
    JointInterfaceProgramOutputV2,
    finite_program_catalog_sha256_v2,
)


SCHEMA_VERSION = "joint_interface_program_learner.v3"
CATALOG_SPEC_SCHEMA_VERSION = "externally_frozen_program_catalog_spec.v3"
CHECKPOINT_SCHEMA_VERSION = "joint_interface_program_checkpoint.v3"
QUERY_BINDING_SCHEMA_VERSION = "full_graph_query_binding.v3"
FACE_SIGNATURE_SCHEMA_VERSION = "model_visible_face_signature.v3"
PART_BINDING_SCHEMA_VERSION = "full_part_graph_binding.v3"
FRAME_RECEIPT_SCHEMA_VERSION = "full_view_face_frame_receipt.v3"
EXECUTABLE_SCHEMA_VERSION = "executable_predicted_program.v3"
TRACE_SCHEMA_VERSION = "preemptive_topk_trace.v3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUERY_FACTORY_TOKEN = object()
_FRAME_FACTORY_TOKEN = object()
_TRACE_FACTORY_TOKEN = object()
_CATALOG_SPEC_FACTORY_TOKEN = object()


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
  value = tensor.detach().cpu().contiguous()
  digest = hashlib.sha256()
  digest.update(str(value.dtype).encode("ascii"))
  digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
  digest.update(value.numpy().tobytes(order="C"))
  return digest.hexdigest()


def _state_dict_sha256(model: nn.Module) -> str:
  digest = hashlib.sha256()
  for name, tensor in sorted(model.state_dict().items()):
    digest.update(name.encode("utf-8"))
    digest.update(bytes.fromhex(_tensor_sha256(tensor)))
  return digest.hexdigest()


def _verify_self_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
  claimed = payload.get(field)
  if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
    raise ValueError(f"{label} self hash is malformed")
  unsigned = dict(payload)
  unsigned.pop(field, None)
  if _canonical_sha256(unsigned) != claimed:
    raise ValueError(f"{label} self hash differs")


@dataclass(frozen=True, slots=True)
class ExternallyFrozenCatalogSpecV3:
  roster: FiniteProgramCatalogRosterV2
  source_artifact_sha256: str
  producer_code_sha256: str
  spec_payload_sha256: str
  spec_file_sha256: str
  _factory_token: object = field(repr=False, compare=False)
  schema_version: str = CATALOG_SPEC_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CATALOG_SPEC_SCHEMA_VERSION:
      raise ValueError("V3 catalog spec schema differs")
    if self._factory_token is not _CATALOG_SPEC_FACTORY_TOKEN:
      raise TypeError("V3 external catalog specs are loader-only")
    if type(self.roster) is not FiniteProgramCatalogRosterV2:
      raise TypeError("V3 catalog spec requires the exact V2 roster")
    for value in (
        self.source_artifact_sha256,
        self.producer_code_sha256,
        self.spec_payload_sha256,
        self.spec_file_sha256,
    ):
      if _SHA256.fullmatch(value) is None:
        raise ValueError("V3 catalog spec hash is malformed")


def frozen_catalog_spec_payload_v3(
    roster: FiniteProgramCatalogRosterV2,
    *,
    source_artifact_sha256: str,
    producer_code_sha256: str,
) -> dict[str, Any]:
  if type(roster) is not FiniteProgramCatalogRosterV2:
    raise TypeError("V3 frozen spec requires a validated roster")
  for value in (source_artifact_sha256, producer_code_sha256):
    if _SHA256.fullmatch(value) is None:
      raise ValueError("V3 frozen spec source hashes must be SHA-256")
  payload = {
      "schema_version": CATALOG_SPEC_SCHEMA_VERSION,
      "source_artifact_sha256": source_artifact_sha256,
      "producer_code_sha256": producer_code_sha256,
      "roster": {
          "schema_version": CATALOG_SCHEMA_VERSION,
          "entry_count": len(roster.entries),
          "entries": [entry.payload() for entry in roster.entries],
          "catalog_sha256": roster.catalog_sha256,
      },
  }
  payload["spec_payload_sha256"] = _canonical_sha256(payload)
  return payload


def load_externally_frozen_catalog_spec_v3(
    path: str | Path,
    *,
    expected_catalog_roster_sha256: str,
    expected_spec_file_sha256: str,
) -> ExternallyFrozenCatalogSpecV3:
  if _SHA256.fullmatch(expected_catalog_roster_sha256) is None or _SHA256.fullmatch(
      expected_spec_file_sha256
  ) is None:
    raise ValueError("V3 catalog loader requires external expected SHA-256 values")
  source = Path(path)
  if _file_sha256(source) != expected_spec_file_sha256:
    raise ValueError("V3 catalog spec file differs from external commitment")
  try:
    payload = json.loads(source.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("V3 catalog spec is unreadable strict JSON") from error
  if not isinstance(payload, Mapping):
    raise ValueError("V3 catalog spec must be an object")
  _verify_self_hash(payload, field="spec_payload_sha256", label="V3 catalog spec")
  if set(payload) != {
      "schema_version",
      "source_artifact_sha256",
      "producer_code_sha256",
      "roster",
      "spec_payload_sha256",
  } or payload.get("schema_version") != CATALOG_SPEC_SCHEMA_VERSION:
    raise ValueError("V3 catalog spec fields differ")
  raw_roster = payload.get("roster")
  if not isinstance(raw_roster, Mapping) or set(raw_roster) != {
      "schema_version",
      "entry_count",
      "entries",
      "catalog_sha256",
  }:
    raise ValueError("V3 catalog roster payload differs")
  raw_entries = raw_roster.get("entries")
  if not isinstance(raw_entries, list):
    raise ValueError("V3 catalog roster entries differ")
  entries: list[FiniteProgramCatalogEntryV2] = []
  for index, row in enumerate(raw_entries):
    if not isinstance(row, Mapping) or set(row) != {
        "program_index",
        "program_id",
        "descriptor",
        "descriptor_sha256",
    }:
      raise ValueError("V3 catalog entry fields differ")
    descriptor_raw = row.get("descriptor")
    if not isinstance(descriptor_raw, Mapping) or set(descriptor_raw) != {
        "relation_hint",
        "contact_type",
        "surface_type_a",
        "surface_type_b",
    }:
      raise ValueError("V3 catalog descriptor fields differ")
    descriptor = FiniteProgramDescriptorV2(**dict(descriptor_raw))
    entries.append(
        FiniteProgramCatalogEntryV2(
            program_index=int(row["program_index"]),
            program_id=str(row["program_id"]),
            descriptor=descriptor,
            descriptor_sha256=str(row["descriptor_sha256"]),
        )
    )
    if entries[-1].program_index != index:
      raise ValueError("V3 catalog entry order differs")
  roster = FiniteProgramCatalogRosterV2(
      entries=tuple(entries),
      catalog_sha256=str(raw_roster.get("catalog_sha256")),
  )
  if roster.catalog_sha256 != expected_catalog_roster_sha256:
    raise ValueError("V3 alternate self-signed catalog is forbidden")
  if raw_roster.get("entry_count") != FORMAL_CATALOG_SIZE:
    raise ValueError("V3 catalog entry count differs")
  return ExternallyFrozenCatalogSpecV3(
      roster=roster,
      source_artifact_sha256=str(payload["source_artifact_sha256"]),
      producer_code_sha256=str(payload["producer_code_sha256"]),
      spec_payload_sha256=str(payload["spec_payload_sha256"]),
      spec_file_sha256=expected_spec_file_sha256,
      _factory_token=_CATALOG_SPEC_FACTORY_TOKEN,
  )


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramLearnerConfigV3:
  expected_catalog_roster_sha256: str
  hidden_dim: int = 96
  layers: int = 4
  top_l: int = 8
  dropout: float = 0.0
  face_salience_loss_weight: float = 0.5
  translation_loss_weight: float = 0.25
  rotation_loss_weight: float = 0.25
  pair_frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256
  catalog_size: int = FORMAL_CATALOG_SIZE

  def __post_init__(self) -> None:
    if _SHA256.fullmatch(self.expected_catalog_roster_sha256) is None:
      raise ValueError("V3 config requires an externally expected catalog hash")
    JointInterfaceProgramLearnerConfigV2(
        hidden_dim=self.hidden_dim,
        layers=self.layers,
        top_l=self.top_l,
        dropout=self.dropout,
        face_salience_loss_weight=self.face_salience_loss_weight,
        translation_loss_weight=self.translation_loss_weight,
        rotation_loss_weight=self.rotation_loss_weight,
        pair_frame_policy_sha256=self.pair_frame_policy_sha256,
        catalog_size=self.catalog_size,
    )

  @property
  def sha256(self) -> str:
    return _canonical_sha256(asdict(self))

  def v2_config(self) -> JointInterfaceProgramLearnerConfigV2:
    return JointInterfaceProgramLearnerConfigV2(
        hidden_dim=self.hidden_dim,
        layers=self.layers,
        top_l=self.top_l,
        dropout=self.dropout,
        face_salience_loss_weight=self.face_salience_loss_weight,
        translation_loss_weight=self.translation_loss_weight,
        rotation_loss_weight=self.rotation_loss_weight,
        pair_frame_policy_sha256=self.pair_frame_policy_sha256,
        catalog_size=self.catalog_size,
    )


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramOutputV3:
  core: JointInterfaceProgramOutputV2
  v3_config_sha256: str
  expected_catalog_roster_sha256: str
  model_state_sha256: str
  learner_source_sha256: str
  schema_version: str = SCHEMA_VERSION


class JointInterfaceProgramLearnerV3(nn.Module):
  """V2 neural architecture with externally rooted catalog state bindings."""

  def __init__(
      self, config: JointInterfaceProgramLearnerConfigV3, *, seed: int
  ) -> None:
    super().__init__()
    if type(config) is not JointInterfaceProgramLearnerConfigV3:
      raise TypeError("V3 learner config type differs")
    self.config = config
    self.seed = seed
    self.core = JointInterfaceProgramLearnerV2(config.v2_config(), seed=seed)
    self.register_buffer(
        "expected_catalog_roster_sha256_bytes",
        torch.tensor(
            list(bytes.fromhex(config.expected_catalog_roster_sha256)),
            dtype=torch.uint8,
        ),
        persistent=True,
    )
    self.register_buffer(
        "v3_config_sha256_bytes",
        torch.tensor(list(bytes.fromhex(config.sha256)), dtype=torch.uint8),
        persistent=True,
    )

  @property
  def parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters())

  def forward(self, inputs: GraphProgramInputsV2) -> JointInterfaceProgramOutputV3:
    if type(inputs) is not GraphProgramInputsV2:
      raise TypeError("V3 forward accepts GraphProgramInputsV2 only")
    if inputs.catalog_sha256 != self.config.expected_catalog_roster_sha256:
      raise ValueError("V3 forward catalog differs from externally frozen config")
    core = self.core(inputs)
    return JointInterfaceProgramOutputV3(
        core=core,
        v3_config_sha256=self.config.sha256,
        expected_catalog_roster_sha256=self.config.expected_catalog_roster_sha256,
        model_state_sha256=_state_dict_sha256(self),
        learner_source_sha256=_file_sha256(Path(__file__).resolve()),
    )


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramTargetsV3:
  face_index_a: Tensor
  face_index_b: Tensor
  program_index: Tensor
  residual_translation_pair_local: Tensor
  residual_rotation_vector_pair_local: Tensor
  residual_mask: Tensor
  expected_catalog_roster_sha256: str
  pair_frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256


@dataclass(frozen=True, slots=True)
class JointContrastiveRowV3:
  face_indices_a: Tensor
  face_indices_b: Tensor
  logits: Tensor
  target_flat_position: int


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramLossV3:
  total: Tensor
  mandatory_salience_ce: Tensor
  joint_face_pair_program_ce: Tensor
  residual_translation_pair_local: Tensor
  residual_rotation_pair_local: Tensor
  contrastive_rows: tuple[JointContrastiveRowV3, ...]


def _hard_union(selected: Tensor, gold: int) -> Tensor:
  valid = [int(value) for value in selected.detach().cpu().tolist() if int(value) >= 0]
  if gold not in valid:
    valid.append(gold)
  return torch.tensor(valid, dtype=torch.long, device=selected.device)


def joint_interface_program_loss_v3(
    model: JointInterfaceProgramLearnerV3,
    output: JointInterfaceProgramOutputV3,
    targets: JointInterfaceProgramTargetsV3,
) -> JointInterfaceProgramLossV3:
  """One CE over (hard faces union gold)^2 x all 19 programs."""

  if type(model) is not JointInterfaceProgramLearnerV3:
    raise TypeError("V3 loss requires JointInterfaceProgramLearnerV3")
  if type(output) is not JointInterfaceProgramOutputV3:
    raise TypeError("V3 loss requires JointInterfaceProgramOutputV3")
  if type(targets) is not JointInterfaceProgramTargetsV3:
    raise TypeError("V3 loss requires separate JointInterfaceProgramTargetsV3")
  core = output.core
  config = model.config
  if (
      output.v3_config_sha256 != config.sha256
      or output.expected_catalog_roster_sha256
      != config.expected_catalog_roster_sha256
      or output.model_state_sha256 != _state_dict_sha256(model)
      or output.learner_source_sha256 != _file_sha256(Path(__file__).resolve())
      or core.top_l != config.top_l
      or core.catalog_size != FORMAL_CATALOG_SIZE
  ):
    raise ValueError("V3 loss output/config/state/code binding differs")
  if (
      targets.expected_catalog_roster_sha256
      != config.expected_catalog_roster_sha256
      or core.catalog_sha256 != config.expected_catalog_roster_sha256
      or targets.pair_frame_policy_sha256 != config.pair_frame_policy_sha256
  ):
    raise ValueError("V3 loss catalog/frame binding differs")
  batch = core.all_face_embeddings.shape[0]
  for value in (
      targets.face_index_a,
      targets.face_index_b,
      targets.program_index,
      targets.residual_mask,
  ):
    if value.shape != (batch,):
      raise ValueError("V3 scalar target shape differs")
  if targets.residual_translation_pair_local.shape != (batch, 3) or (
      targets.residual_rotation_vector_pair_local.shape != (batch, 3)
  ):
    raise ValueError("V3 residual target shape differs")
  salience = 0.5 * (
      F.cross_entropy(core.face_salience_logits[:, 0], targets.face_index_a)
      + F.cross_entropy(core.face_salience_logits[:, 1], targets.face_index_b)
  )
  row_losses: list[Tensor] = []
  translations: list[Tensor] = []
  rotations: list[Tensor] = []
  rows: list[JointContrastiveRowV3] = []
  for row_index in range(batch):
    gold_a = int(targets.face_index_a[row_index])
    gold_b = int(targets.face_index_b[row_index])
    if (
        gold_a < 0
        or gold_b < 0
        or gold_a >= core.all_face_mask.shape[2]
        or gold_b >= core.all_face_mask.shape[2]
        or not bool(core.all_face_mask[row_index, 0, gold_a])
        or not bool(core.all_face_mask[row_index, 1, gold_b])
    ):
      raise ValueError("V3 gold face is outside the complete graph")
    set_a = _hard_union(core.selected_face_indices[row_index, 0], gold_a)
    set_b = _hard_union(core.selected_face_indices[row_index, 1], gold_b)
    embedding_a = core.all_face_embeddings[row_index, 0, set_a]
    embedding_b = core.all_face_embeddings[row_index, 1, set_b]
    count_a, count_b = set_a.shape[0], set_b.shape[0]
    left = embedding_a[:, None, :].expand(-1, count_b, -1)
    right = embedding_b[None, :, :].expand(count_a, -1, -1)
    candidates = core.candidate_program_indices[row_index][None, None, :].expand(
        count_a, count_b, -1
    )
    logits, residual = model.core._pair_program_scores(left, right, candidates)
    position_a = int(torch.nonzero(set_a == gold_a).flatten()[0])
    position_b = int(torch.nonzero(set_b == gold_b).flatten()[0])
    program_matches = core.candidate_program_indices[row_index] == targets.program_index[
        row_index
    ]
    if int(program_matches.sum()) != 1:
      raise ValueError("V3 gold program differs from exact catalog")
    program_position = int(torch.nonzero(program_matches).flatten()[0])
    flat_target = (
        (position_a * count_b + position_b) * FORMAL_CATALOG_SIZE
        + program_position
    )
    flat_logits = logits.reshape(1, -1)
    row_losses.append(
        F.cross_entropy(
            flat_logits,
            torch.tensor([flat_target], device=flat_logits.device),
        )
    )
    predicted = residual[position_a, position_b, program_position]
    translations.append(
        F.smooth_l1_loss(
            predicted[:3],
            targets.residual_translation_pair_local[row_index],
            reduction="mean",
        )
    )
    rotations.append(
        F.smooth_l1_loss(
            predicted[3:],
            targets.residual_rotation_vector_pair_local[row_index],
            reduction="mean",
        )
    )
    rows.append(
        JointContrastiveRowV3(
            face_indices_a=set_a,
            face_indices_b=set_b,
            logits=logits,
            target_flat_position=flat_target,
        )
    )
  joint_ce = torch.stack(row_losses).mean()
  mask = targets.residual_mask.to(joint_ce.dtype)
  denominator = mask.sum().clamp_min(1.0)
  translation = (torch.stack(translations) * mask).sum() / denominator
  rotation = (torch.stack(rotations) * mask).sum() / denominator
  total = (
      config.face_salience_loss_weight * salience
      + joint_ce
      + config.translation_loss_weight * translation
      + config.rotation_loss_weight * rotation
  )
  return JointInterfaceProgramLossV3(
      total=total,
      mandatory_salience_ce=salience,
      joint_face_pair_program_ce=joint_ce,
      residual_translation_pair_local=translation,
      residual_rotation_pair_local=rotation,
      contrastive_rows=tuple(rows),
  )


@dataclass(frozen=True, slots=True)
class ModelVisibleFaceSignatureV3:
  graph_local_face_index: int
  signature_sha256: str
  schema_version: str = FACE_SIGNATURE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FullPartGraphBindingV3:
  part_instance_id: str
  body_id: str
  step_sha256: str
  full_graph_sha256: str
  face_signatures: tuple[ModelVisibleFaceSignatureV3, ...]
  schema_version: str = PART_BINDING_SCHEMA_VERSION


def _part_graph_binding(
    inputs: GraphProgramInputsV2,
    *,
    part_index: int,
    part_instance_id: str,
    body_id: str,
    step_sha256: str,
) -> FullPartGraphBindingV3:
  if not part_instance_id or not body_id or _SHA256.fullmatch(step_sha256) is None:
    raise ValueError("V3 part identity/STEP binding is malformed")
  tensors = inputs.graph_tensors
  if tensors.node_features.shape[0] != 1:
    raise ValueError("V3 query binding accepts one graph pair")
  node_count = int(tensors.node_mask[0, part_index].sum())
  edge_mask = tensors.edge_mask[0, part_index]
  nodes = tensors.node_features[0, part_index, :node_count]
  edges = tensors.edge_index[0, part_index, edge_mask]
  edge_features = tensors.edge_features[0, part_index, edge_mask]
  graph_payload = {
      "node_tensor_sha256": _tensor_sha256(nodes),
      "edge_index_sha256": _tensor_sha256(edges),
      "edge_feature_sha256": _tensor_sha256(edge_features),
      "node_count": node_count,
      "edge_count": int(edges.shape[0]),
  }
  signatures: list[ModelVisibleFaceSignatureV3] = []
  for face_index in range(node_count):
    incident = edges[(edges[:, 0] == face_index) | (edges[:, 1] == face_index)]
    signature = _canonical_sha256(
        {
            "schema_version": FACE_SIGNATURE_SCHEMA_VERSION,
            "graph_local_face_index": face_index,
            "node_feature_sha256": _tensor_sha256(nodes[face_index]),
            "incident_edge_index_sha256": _tensor_sha256(incident),
        }
    )
    signatures.append(ModelVisibleFaceSignatureV3(face_index, signature))
  return FullPartGraphBindingV3(
      part_instance_id=part_instance_id,
      body_id=body_id,
      step_sha256=step_sha256,
      full_graph_sha256=_canonical_sha256(graph_payload),
      face_signatures=tuple(signatures),
  )


def _query_core_payload(
    *,
    case_id: str,
    row_id: str,
    part_a: FullPartGraphBindingV3,
    part_b: FullPartGraphBindingV3,
    frame_policy_sha256: str,
    frame_source_sha256: str,
) -> dict[str, Any]:
  def part_payload(part: FullPartGraphBindingV3) -> dict[str, Any]:
    return {
        "part_instance_id": part.part_instance_id,
        "body_id": part.body_id,
        "step_sha256": part.step_sha256,
        "full_graph_sha256": part.full_graph_sha256,
        "face_signatures": [asdict(signature) for signature in part.face_signatures],
    }

  return {
      "schema_version": "graph_query_core.v3",
      "case_id": case_id,
      "row_id": row_id,
      "part_a": part_payload(part_a),
      "part_b": part_payload(part_b),
      "frame_policy_sha256": frame_policy_sha256,
      "frame_source_sha256": frame_source_sha256,
  }


def graph_query_core_commitment_v3(
    inputs: GraphProgramInputsV2,
    *,
    case_id: str,
    row_id: str,
    part_instance_id_a: str,
    body_id_a: str,
    step_sha256_a: str,
    part_instance_id_b: str,
    body_id_b: str,
    step_sha256_b: str,
    frame_policy_sha256: str,
    frame_source_sha256: str,
) -> tuple[str, FullPartGraphBindingV3, FullPartGraphBindingV3]:
  """Public pre-receipt commitment for an external full-view frame producer."""

  if not case_id or not row_id:
    raise ValueError("V3 query case/row identity must be non-empty")
  part_a = _part_graph_binding(
      inputs,
      part_index=0,
      part_instance_id=part_instance_id_a,
      body_id=body_id_a,
      step_sha256=step_sha256_a,
  )
  part_b = _part_graph_binding(
      inputs,
      part_index=1,
      part_instance_id=part_instance_id_b,
      body_id=body_id_b,
      step_sha256=step_sha256_b,
  )
  payload = _query_core_payload(
      case_id=case_id,
      row_id=row_id,
      part_a=part_a,
      part_b=part_b,
      frame_policy_sha256=frame_policy_sha256,
      frame_source_sha256=frame_source_sha256,
  )
  return _canonical_sha256(payload), part_a, part_b


def full_view_frame_receipt_payload_v3(
    *,
    query_core_sha256: str,
    frame_policy_sha256: str,
    frame_source_sha256: str,
    frames_a: Sequence[Mapping[str, Any]],
    frames_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  payload = {
      "schema_version": FRAME_RECEIPT_SCHEMA_VERSION,
      "query_core_sha256": query_core_sha256,
      "frame_policy_sha256": frame_policy_sha256,
      "frame_source_sha256": frame_source_sha256,
      "frames": {"a": list(frames_a), "b": list(frames_b)},
  }
  payload["receipt_payload_sha256"] = _canonical_sha256(payload)
  return payload


def _load_frame_receipt(path: Path) -> tuple[dict[str, Any], str]:
  file_sha = _file_sha256(path)
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("V3 frame receipt is unreadable strict JSON") from error
  if not isinstance(payload, Mapping):
    raise ValueError("V3 frame receipt must be an object")
  _verify_self_hash(payload, field="receipt_payload_sha256", label="V3 frame receipt")
  if set(payload) != {
      "schema_version",
      "query_core_sha256",
      "frame_policy_sha256",
      "frame_source_sha256",
      "frames",
      "receipt_payload_sha256",
  } or payload.get("schema_version") != FRAME_RECEIPT_SCHEMA_VERSION:
    raise ValueError("V3 frame receipt fields differ")
  return dict(payload), file_sha


class GraphQueryBindingV3:
  __slots__ = (
      "_case_id",
      "_row_id",
      "_part_a",
      "_part_b",
      "_frame_receipt_sha256",
      "_frame_policy_sha256",
      "_frame_source_sha256",
      "_query_core_sha256",
      "_binding_sha256",
  )

  def __init__(
      self,
      *,
      case_id: str,
      row_id: str,
      part_a: FullPartGraphBindingV3,
      part_b: FullPartGraphBindingV3,
      frame_receipt_sha256: str,
      frame_policy_sha256: str,
      frame_source_sha256: str,
      query_core_sha256: str,
      binding_sha256: str,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _QUERY_FACTORY_TOKEN:
      raise TypeError("V3 graph query bindings are factory-only")
    self._case_id = case_id
    self._row_id = row_id
    self._part_a = part_a
    self._part_b = part_b
    self._frame_receipt_sha256 = frame_receipt_sha256
    self._frame_policy_sha256 = frame_policy_sha256
    self._frame_source_sha256 = frame_source_sha256
    self._query_core_sha256 = query_core_sha256
    self._binding_sha256 = binding_sha256

  def __setattr__(self, name: str, value: Any) -> None:
    if hasattr(self, name):
      raise AttributeError("V3 graph query bindings are immutable")
    object.__setattr__(self, name, value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("V3 graph query bindings are not serializable")

  @property
  def case_id(self) -> str: return self._case_id
  @property
  def row_id(self) -> str: return self._row_id
  @property
  def part_a(self) -> FullPartGraphBindingV3: return self._part_a
  @property
  def part_b(self) -> FullPartGraphBindingV3: return self._part_b
  @property
  def frame_receipt_sha256(self) -> str: return self._frame_receipt_sha256
  @property
  def frame_policy_sha256(self) -> str: return self._frame_policy_sha256
  @property
  def frame_source_sha256(self) -> str: return self._frame_source_sha256
  @property
  def query_core_sha256(self) -> str: return self._query_core_sha256
  @property
  def binding_sha256(self) -> str: return self._binding_sha256

  @classmethod
  def from_full_graph_and_frame_receipt(
      cls,
      inputs: GraphProgramInputsV2,
      *,
      case_id: str,
      row_id: str,
      part_instance_id_a: str,
      body_id_a: str,
      step_sha256_a: str,
      part_instance_id_b: str,
      body_id_b: str,
      step_sha256_b: str,
      frame_receipt_path: str | Path,
      frame_policy_sha256: str,
      frame_source_sha256: str,
  ) -> "GraphQueryBindingV3":
    if not case_id or not row_id:
      raise ValueError("V3 query case/row identity must be non-empty")
    for value in (frame_policy_sha256, frame_source_sha256):
      if _SHA256.fullmatch(value) is None:
        raise ValueError("V3 query frame source/policy hash is malformed")
    part_a = _part_graph_binding(
        inputs,
        part_index=0,
        part_instance_id=part_instance_id_a,
        body_id=body_id_a,
        step_sha256=step_sha256_a,
    )
    part_b = _part_graph_binding(
        inputs,
        part_index=1,
        part_instance_id=part_instance_id_b,
        body_id=body_id_b,
        step_sha256=step_sha256_b,
    )
    core_payload = _query_core_payload(
        case_id=case_id,
        row_id=row_id,
        part_a=part_a,
        part_b=part_b,
        frame_policy_sha256=frame_policy_sha256,
        frame_source_sha256=frame_source_sha256,
    )
    query_core_sha = _canonical_sha256(core_payload)
    receipt, receipt_file_sha = _load_frame_receipt(Path(frame_receipt_path))
    if (
        receipt.get("query_core_sha256") != query_core_sha
        or receipt.get("frame_policy_sha256") != frame_policy_sha256
        or receipt.get("frame_source_sha256") != frame_source_sha256
    ):
      raise ValueError("V3 frame receipt differs from full graph query core")
    _validate_receipt_frames(receipt, part_a=part_a, part_b=part_b)
    binding_payload = {
        "schema_version": QUERY_BINDING_SCHEMA_VERSION,
        "query_core_sha256": query_core_sha,
        "frame_receipt_sha256": receipt_file_sha,
        "frame_receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }
    return cls(
        case_id=case_id,
        row_id=row_id,
        part_a=part_a,
        part_b=part_b,
        frame_receipt_sha256=receipt_file_sha,
        frame_policy_sha256=frame_policy_sha256,
        frame_source_sha256=frame_source_sha256,
        query_core_sha256=query_core_sha,
        binding_sha256=_canonical_sha256(binding_payload),
        _factory_token=_QUERY_FACTORY_TOKEN,
    )

  def revalidate(self, inputs: GraphProgramInputsV2) -> None:
    for part_index, expected in ((0, self.part_a), (1, self.part_b)):
      actual = _part_graph_binding(
          inputs,
          part_index=part_index,
          part_instance_id=expected.part_instance_id,
          body_id=expected.body_id,
          step_sha256=expected.step_sha256,
      )
      if actual != expected:
        raise ValueError("V3 graph query replay differs from bound full graph")


def _validate_receipt_frames(
    receipt: Mapping[str, Any],
    *,
    part_a: FullPartGraphBindingV3,
    part_b: FullPartGraphBindingV3,
) -> None:
  frames = receipt.get("frames")
  if not isinstance(frames, Mapping) or set(frames) != {"a", "b"}:
    raise ValueError("V3 frame receipt part domain differs")
  for slot, part in (("a", part_a), ("b", part_b)):
    rows = frames.get(slot)
    if not isinstance(rows, list) or len(rows) != len(part.face_signatures):
      raise ValueError("V3 frame receipt face domain is incomplete")
    for index, (row, signature) in enumerate(zip(rows, part.face_signatures, strict=True)):
      if not isinstance(row, Mapping) or set(row) != {
          "graph_local_face_index",
          "face_signature_sha256",
          "origin_world",
          "rotation_local_to_world",
      }:
        raise ValueError("V3 frame receipt row fields differ")
      if (
          row.get("graph_local_face_index") != index
          or row.get("face_signature_sha256") != signature.signature_sha256
      ):
        raise ValueError("V3 frame receipt face signature replay differs")
      origin = np.asarray(row.get("origin_world"), dtype=float)
      rotation = np.asarray(row.get("rotation_local_to_world"), dtype=float)
      if origin.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(
          origin
      ).all() or not np.isfinite(rotation).all():
        raise ValueError("V3 frame receipt transform is malformed")
      orthogonal = all(
          math.isclose(
              sum(
                  float(rotation[axis, first]) * float(rotation[axis, second])
                  for axis in range(3)
              ),
              1.0 if first == second else 0.0,
              abs_tol=1e-7,
          )
          for first in range(3)
          for second in range(3)
      )
      determinant = float(
          rotation[0, 0]
          * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
          - rotation[0, 1]
          * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
          + rotation[0, 2]
          * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
      )
      if not orthogonal or not math.isclose(determinant, 1.0, abs_tol=1e-7):
        raise ValueError("V3 frame receipt rotation must be proper orthogonal")


class AuthenticatedPredictedFaceFrameV3:
  __slots__ = (
      "_query_binding_sha256",
      "_part_slot",
      "_face_index",
      "_face_signature_sha256",
      "_origin_world",
      "_rotation_local_to_world",
      "_receipt_sha256",
  )

  def __init__(
      self,
      *,
      query_binding_sha256: str,
      part_slot: str,
      face_index: int,
      face_signature_sha256: str,
      origin_world: Sequence[float],
      rotation_local_to_world: Sequence[Sequence[float]],
      receipt_sha256: str,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _FRAME_FACTORY_TOKEN:
      raise TypeError("V3 predicted face frames are receipt-factory-only")
    self._query_binding_sha256 = query_binding_sha256
    self._part_slot = part_slot
    self._face_index = face_index
    self._face_signature_sha256 = face_signature_sha256
    self._origin_world = tuple(float(value) for value in origin_world)
    self._rotation_local_to_world = tuple(
        tuple(float(value) for value in row) for row in rotation_local_to_world
    )
    self._receipt_sha256 = receipt_sha256

  def __setattr__(self, name: str, value: Any) -> None:
    if hasattr(self, name):
      raise AttributeError("V3 authenticated face frames are immutable")
    object.__setattr__(self, name, value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("V3 authenticated face frames are not serializable")

  @property
  def query_binding_sha256(self) -> str: return self._query_binding_sha256
  @property
  def part_slot(self) -> str: return self._part_slot
  @property
  def graph_local_face_index(self) -> int: return self._face_index
  @property
  def face_signature_sha256(self) -> str: return self._face_signature_sha256
  @property
  def origin_world(self) -> tuple[float, float, float]: return self._origin_world
  @property
  def rotation_local_to_world(self) -> tuple[tuple[float, ...], ...]: return self._rotation_local_to_world
  @property
  def receipt_sha256(self) -> str: return self._receipt_sha256


class FullViewFrameFactoryV3:
  def __init__(
      self,
      query: GraphQueryBindingV3,
      *,
      frame_receipt_path: str | Path,
  ) -> None:
    if type(query) is not GraphQueryBindingV3:
      raise TypeError("V3 frame factory requires a graph query binding")
    payload, file_sha = _load_frame_receipt(Path(frame_receipt_path))
    if (
        file_sha != query.frame_receipt_sha256
        or payload.get("query_core_sha256") != query.query_core_sha256
        or payload.get("frame_policy_sha256") != query.frame_policy_sha256
        or payload.get("frame_source_sha256") != query.frame_source_sha256
    ):
      raise ValueError("V3 frame factory receipt differs from query binding")
    _validate_receipt_frames(payload, part_a=query.part_a, part_b=query.part_b)
    self._query = query
    self._payload = payload

  def frame_for(
      self, *, part_slot: str, graph_local_face_index: int
  ) -> AuthenticatedPredictedFaceFrameV3:
    part = self._query.part_a if part_slot == "a" else self._query.part_b if part_slot == "b" else None
    if part is None or not 0 <= graph_local_face_index < len(part.face_signatures):
      raise ValueError("V3 requested frame is outside the bound face domain")
    row = self._payload["frames"][part_slot][graph_local_face_index]
    signature = part.face_signatures[graph_local_face_index]
    return AuthenticatedPredictedFaceFrameV3(
        query_binding_sha256=self._query.binding_sha256,
        part_slot=part_slot,
        face_index=graph_local_face_index,
        face_signature_sha256=signature.signature_sha256,
        origin_world=row["origin_world"],
        rotation_local_to_world=row["rotation_local_to_world"],
        receipt_sha256=self._query.frame_receipt_sha256,
        _factory_token=_FRAME_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class HardGraphProposalBudgetV3:
  top_k: int
  expanded_candidate_pool: int
  max_occ_calls: int
  wall_time_seconds: float

  def __post_init__(self) -> None:
    if type(self.top_k) is not int or self.top_k < 1:
      raise ValueError("V3 proposal top_k must be positive")
    if type(self.expanded_candidate_pool) is not int or self.expanded_candidate_pool < self.top_k:
      raise ValueError("V3 expanded candidate pool must cover top_k")
    if self.max_occ_calls != 0:
      raise ValueError("V3 proposal requires max_occ_calls=0")
    if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds <= 0.0:
      raise ValueError("V3 proposal wall time must be finite and positive")


class PreemptiveTopKTraceV3:
  __slots__ = (
      "_elapsed_seconds",
      "_top_k",
      "_expanded_candidate_pool",
      "_face_pair_pool",
      "_program_pool",
      "_worker_pid",
      "_worker_exit_code",
  )

  def __init__(
      self,
      *,
      elapsed_seconds: float,
      top_k: int,
      expanded_candidate_pool: int,
      face_pair_pool: int,
      program_pool: int,
      worker_pid: int,
      worker_exit_code: int,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _TRACE_FACTORY_TOKEN:
      raise TypeError("V3 execution traces are executor-only")
    self._elapsed_seconds = elapsed_seconds
    self._top_k = top_k
    self._expanded_candidate_pool = expanded_candidate_pool
    self._face_pair_pool = face_pair_pool
    self._program_pool = program_pool
    self._worker_pid = worker_pid
    self._worker_exit_code = worker_exit_code

  def __setattr__(self, name: str, value: Any) -> None:
    if hasattr(self, name):
      raise AttributeError("V3 execution traces are immutable")
    object.__setattr__(self, name, value)

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "elapsed_seconds": self._elapsed_seconds,
        "top_k": self._top_k,
        "expanded_candidate_pool": self._expanded_candidate_pool,
        "face_pair_pool": self._face_pair_pool,
        "program_pool": self._program_pool,
        "worker_pid": self._worker_pid,
        "worker_exit_code": self._worker_exit_code,
        "deadline_scope": "pre_process_start_through_parent_topk_accept.v3",
        "worker_payload": "lightweight_topk_only.v3",
    }


@dataclass(frozen=True, slots=True)
class _LightweightTopKRowV3:
  score: float
  face_index_a: int
  face_index_b: int
  program_index: int
  residual_translation_pair_local: tuple[float, float, float]
  residual_rotation_vector_pair_local: tuple[float, float, float]


def _topk_worker_v3(
    connection: Connection,
    model: JointInterfaceProgramLearnerV3,
    inputs: GraphProgramInputsV2,
    top_k: int,
) -> None:
  try:
    model.eval()
    with torch.inference_mode():
      output = model(inputs).core
    indices = output.selected_face_indices[0].cpu()
    logits = output.inference_joint_logits[0].cpu()
    scored: list[tuple[float, int, int, int, int, int]] = []
    for rank_a in range(model.config.top_l):
      face_a = int(indices[0, rank_a])
      if face_a < 0:
        continue
      for rank_b in range(model.config.top_l):
        face_b = int(indices[1, rank_b])
        if face_b < 0:
          continue
        for program_position in range(FORMAL_CATALOG_SIZE):
          program_index = int(output.candidate_program_indices[0, program_position])
          scored.append(
              (
                  float(logits[rank_a, rank_b, program_position]),
                  face_a,
                  face_b,
                  program_index,
                  rank_a,
                  rank_b,
              )
          )
    scored.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    rows: list[_LightweightTopKRowV3] = []
    for score, face_a, face_b, program_index, rank_a, rank_b in scored[:top_k]:
      rows.append(
          _LightweightTopKRowV3(
              score=score,
              face_index_a=face_a,
              face_index_b=face_b,
              program_index=program_index,
              residual_translation_pair_local=tuple(
                  float(value)
                  for value in output.inference_residual_translation_local[
                      0, rank_a, rank_b, program_index
                  ].cpu()
              ),
              residual_rotation_vector_pair_local=tuple(
                  float(value)
                  for value in output.inference_residual_rotation_vector_local[
                      0, rank_a, rank_b, program_index
                  ].cpu()
              ),
          )
      )
    connection.send(("ok", tuple(rows)))
  except BaseException as error:
    connection.send(("error", type(error).__name__, str(error)))
  finally:
    connection.close()


class PreemptiveTopKExecutorV3:
  def execute(
      self,
      model: JointInterfaceProgramLearnerV3,
      inputs: GraphProgramInputsV2,
      *,
      budget: HardGraphProposalBudgetV3,
  ) -> tuple[tuple[_LightweightTopKRowV3, ...], PreemptiveTopKTraceV3]:
    started = time.monotonic()
    deadline = started + budget.wall_time_seconds
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_topk_worker_v3,
        args=(child, model, inputs, budget.top_k),
        daemon=True,
    )
    process.start()
    worker_pid = int(process.pid or -1)
    child.close()
    try:
      remaining = deadline - time.monotonic()
      if remaining <= 0.0 or not parent.poll(remaining):
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
          process.kill()
          process.join(timeout=1.0)
        raise TimeoutError("V3 worker exceeded total proposal deadline")
      message = parent.recv()
      remaining = deadline - time.monotonic()
      process.join(timeout=max(0.0, remaining))
      if process.is_alive() or time.monotonic() > deadline:
        process.kill()
        process.join(timeout=1.0)
        raise TimeoutError("V3 worker/topK completion exceeded total deadline")
      if not isinstance(message, tuple) or not message or message[0] != "ok":
        detail = message[1:] if isinstance(message, tuple) else message
        raise RuntimeError(f"V3 topK worker failed: {detail}")
      rows = message[1]
      if not isinstance(rows, tuple) or len(rows) != budget.top_k or any(
          type(row) is not _LightweightTopKRowV3 for row in rows
      ):
        raise RuntimeError("V3 worker returned a non-lightweight topK payload")
      completed = time.monotonic()
      if completed > deadline:
        raise TimeoutError("V3 parent accepted topK after total deadline")
      trace = PreemptiveTopKTraceV3(
          elapsed_seconds=completed - started,
          top_k=budget.top_k,
          expanded_candidate_pool=budget.expanded_candidate_pool,
          face_pair_pool=model.config.top_l**2,
          program_pool=FORMAL_CATALOG_SIZE,
          worker_pid=worker_pid,
          worker_exit_code=int(process.exitcode or 0),
          _factory_token=_TRACE_FACTORY_TOKEN,
      )
      return rows, trace
    finally:
      parent.close()
      if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


@dataclass(frozen=True, slots=True)
class ExecutablePredictedProgramV3:
  query_binding_sha256: str
  case_id: str
  row_id: str
  face_index_a: int
  face_signature_sha256_a: str
  face_index_b: int
  face_signature_sha256_b: str
  program_index: int
  program_id: str
  descriptor: FiniteProgramDescriptorV2
  descriptor_sha256: str
  catalog_roster_sha256: str
  learner_source_sha256: str
  model_config_sha256: str
  model_state_sha256: str
  residual_translation_pair_local: tuple[float, float, float]
  residual_rotation_vector_pair_local: tuple[float, float, float]
  pair_frame_policy_sha256: str
  score: float
  trace: PreemptiveTopKTraceV3
  schema_version: str = EXECUTABLE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != EXECUTABLE_SCHEMA_VERSION:
      raise ValueError("V3 executable schema differs")
    for value in (
        self.query_binding_sha256,
        self.face_signature_sha256_a,
        self.face_signature_sha256_b,
        self.descriptor_sha256,
        self.catalog_roster_sha256,
        self.learner_source_sha256,
        self.model_config_sha256,
        self.model_state_sha256,
        self.pair_frame_policy_sha256,
    ):
      if _SHA256.fullmatch(value) is None:
        raise ValueError("V3 executable hash binding is malformed")
    if self.descriptor_sha256 != self.descriptor.sha256 or self.program_id != f"catalog_{self.descriptor_sha256[:16]}":
      raise ValueError("V3 executable program/descriptor binding differs")
    if type(self.trace) is not PreemptiveTopKTraceV3:
      raise TypeError("V3 executable requires executor-produced trace")


class BoundedGraphProgramProposerV3:
  def __init__(
      self,
      model: JointInterfaceProgramLearnerV3,
      catalog_spec: ExternallyFrozenCatalogSpecV3,
  ) -> None:
    if type(model) is not JointInterfaceProgramLearnerV3 or type(
        catalog_spec
    ) is not ExternallyFrozenCatalogSpecV3:
      raise TypeError("V3 proposer requires bound model and external catalog spec")
    if model.config.expected_catalog_roster_sha256 != catalog_spec.roster.catalog_sha256:
      raise ValueError("V3 proposer model/spec catalog binding differs")
    if model.config.top_l != 8:
      raise ValueError("V3 formal proposer requires top_l=8")
    self._model = model.eval()
    self._catalog_spec = catalog_spec
    self._executor = PreemptiveTopKExecutorV3()
    self._state_sha256 = _state_dict_sha256(model)
    self._source_sha256 = _file_sha256(Path(__file__).resolve())

  def propose(
      self,
      query: GraphQueryBindingV3,
      inputs: GraphProgramInputsV2,
      *,
      budget: HardGraphProposalBudgetV3,
  ) -> tuple[ExecutablePredictedProgramV3, ...]:
    if type(query) is not GraphQueryBindingV3:
      raise TypeError("V3 proposer requires a full graph query binding")
    query.revalidate(inputs)
    if query.frame_policy_sha256 != self._model.config.pair_frame_policy_sha256:
      raise ValueError("V3 query/model frame policy binding differs")
    if inputs.catalog_sha256 != self._catalog_spec.roster.catalog_sha256:
      raise ValueError("V3 proposer input/catalog binding differs")
    expanded = self._model.config.top_l**2 * FORMAL_CATALOG_SIZE
    if budget.expanded_candidate_pool != expanded:
      raise ValueError("V3 hard expanded candidate pool must equal topL^2 x 19")
    if _state_dict_sha256(self._model) != self._state_sha256:
      raise ValueError("V3 proposer model state changed after binding")
    if _file_sha256(Path(__file__).resolve()) != self._source_sha256:
      raise ValueError("V3 proposer source changed after binding")
    rows, trace = self._executor.execute(self._model, inputs, budget=budget)
    result: list[ExecutablePredictedProgramV3] = []
    for row in rows:
      entry = self._catalog_spec.roster.entry(row.program_index)
      signature_a = query.part_a.face_signatures[row.face_index_a]
      signature_b = query.part_b.face_signatures[row.face_index_b]
      result.append(
          ExecutablePredictedProgramV3(
              query_binding_sha256=query.binding_sha256,
              case_id=query.case_id,
              row_id=query.row_id,
              face_index_a=row.face_index_a,
              face_signature_sha256_a=signature_a.signature_sha256,
              face_index_b=row.face_index_b,
              face_signature_sha256_b=signature_b.signature_sha256,
              program_index=row.program_index,
              program_id=entry.program_id,
              descriptor=entry.descriptor,
              descriptor_sha256=entry.descriptor_sha256,
              catalog_roster_sha256=self._catalog_spec.roster.catalog_sha256,
              learner_source_sha256=self._source_sha256,
              model_config_sha256=self._model.config.sha256,
              model_state_sha256=self._state_sha256,
              residual_translation_pair_local=row.residual_translation_pair_local,
              residual_rotation_vector_pair_local=row.residual_rotation_vector_pair_local,
              pair_frame_policy_sha256=self._model.config.pair_frame_policy_sha256,
              score=row.score,
              trace=trace,
          )
      )
    return tuple(result)


def validate_executable_query_replay_v3(
    executable: ExecutablePredictedProgramV3,
    query: GraphQueryBindingV3,
) -> None:
  if type(executable) is not ExecutablePredictedProgramV3 or type(
      query
  ) is not GraphQueryBindingV3:
    raise TypeError("V3 replay requires executable and graph query binding")
  if (
      executable.query_binding_sha256 != query.binding_sha256
      or executable.case_id != query.case_id
      or executable.row_id != query.row_id
      or executable.face_signature_sha256_a
      != query.part_a.face_signatures[executable.face_index_a].signature_sha256
      or executable.face_signature_sha256_b
      != query.part_b.face_signatures[executable.face_index_b].signature_sha256
  ):
    raise ValueError("V3 executable cross-query replay is forbidden")


def frames_for_executable_v3(
    factory: FullViewFrameFactoryV3,
    executable: ExecutablePredictedProgramV3,
    query: GraphQueryBindingV3,
) -> tuple[AuthenticatedPredictedFaceFrameV3, AuthenticatedPredictedFaceFrameV3]:
  validate_executable_query_replay_v3(executable, query)
  if executable.pair_frame_policy_sha256 != query.frame_policy_sha256:
    raise ValueError("V3 executable/query frame policy binding differs")
  first = factory.frame_for(
      part_slot="a", graph_local_face_index=executable.face_index_a
  )
  second = factory.frame_for(
      part_slot="b", graph_local_face_index=executable.face_index_b
  )
  if (
      first.query_binding_sha256 != executable.query_binding_sha256
      or second.query_binding_sha256 != executable.query_binding_sha256
      or first.face_signature_sha256 != executable.face_signature_sha256_a
      or second.face_signature_sha256 != executable.face_signature_sha256_b
  ):
    raise ValueError("V3 executable/frame signature binding differs")
  return first, second


def save_joint_interface_checkpoint_v3(
    model: JointInterfaceProgramLearnerV3,
    path: str | Path,
    *,
    catalog_spec: ExternallyFrozenCatalogSpecV3,
) -> str:
  if model.config.expected_catalog_roster_sha256 != catalog_spec.roster.catalog_sha256:
    raise ValueError("V3 checkpoint model/catalog binding differs")
  metadata = {
      "schema_version": CHECKPOINT_SCHEMA_VERSION,
      "model_config": asdict(model.config),
      "model_config_sha256": model.config.sha256,
      "catalog_roster_sha256": catalog_spec.roster.catalog_sha256,
      "catalog_spec_payload_sha256": catalog_spec.spec_payload_sha256,
      "catalog_spec_file_sha256": catalog_spec.spec_file_sha256,
      "learner_source_sha256": _file_sha256(Path(__file__).resolve()),
      "model_state_sha256": _state_dict_sha256(model),
      "seed": model.seed,
  }
  target = Path(path)
  torch.save(
      {
          "metadata": metadata,
          "state_dict": {
              key: value.detach().cpu() for key, value in model.state_dict().items()
          },
      },
      target,
  )
  return _file_sha256(target)


def load_joint_interface_checkpoint_v3(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    catalog_spec_path: str | Path,
    expected_catalog_roster_sha256: str,
    expected_catalog_spec_file_sha256: str,
) -> JointInterfaceProgramLearnerV3:
  checkpoint_path = Path(path)
  if _file_sha256(checkpoint_path) != expected_checkpoint_sha256:
    raise ValueError("V3 checkpoint file differs from external commitment")
  spec = load_externally_frozen_catalog_spec_v3(
      catalog_spec_path,
      expected_catalog_roster_sha256=expected_catalog_roster_sha256,
      expected_spec_file_sha256=expected_catalog_spec_file_sha256,
  )
  payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state_dict"}:
    raise ValueError("V3 checkpoint payload fields differ")
  metadata = payload["metadata"]
  state_dict = payload["state_dict"]
  if not isinstance(metadata, Mapping) or metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
    raise ValueError("V3 checkpoint metadata differs")
  if (
      metadata.get("catalog_roster_sha256") != expected_catalog_roster_sha256
      or metadata.get("catalog_spec_payload_sha256") != spec.spec_payload_sha256
      or metadata.get("catalog_spec_file_sha256") != spec.spec_file_sha256
      or metadata.get("learner_source_sha256")
      != _file_sha256(Path(__file__).resolve())
  ):
    raise ValueError("V3 checkpoint external trust bindings differ")
  config = JointInterfaceProgramLearnerConfigV3(**dict(metadata["model_config"]))
  if config.sha256 != metadata.get("model_config_sha256"):
    raise ValueError("V3 checkpoint config hash differs")
  model = JointInterfaceProgramLearnerV3(config, seed=int(metadata["seed"]))
  model.load_state_dict(state_dict, strict=True)
  if _state_dict_sha256(model) != metadata.get("model_state_sha256"):
    raise ValueError("V3 checkpoint model state hash differs")
  return model.eval()
