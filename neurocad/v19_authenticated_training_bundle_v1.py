"""Torch-first loader, collator, and loss bridge for the V19 training bundle.

This module deliberately has no dependency on semantic authority, adapter
builders/verifiers, OCP, or the graph-cache producer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .v19_brep_program_model_v1 import (
    BRepGraphExample,
    BRepLearnerExample,
    BRepProgramInputs,
    BRepProgramLearnerV1,
    LearnerBatchBudget,
    collate_nonformal_brep_learner_examples,
)
from .v19_set_valued_program_learning_v1 import (
    BundleSetValuedTrainingStateV1,
    CatalogTargetAlternativesV1,
    JointSetProgramResidualLossV1,
    OOVProgramAlternativeV1,
    ProgramResidualAlternativeV1,
    SetValuedProgramSampleV1,
    SetValuedProgramTargets,
    build_bundle_set_valued_training_state_v1,
    rotation_matrix_to_vector_v1,
    set_valued_program_training_loss_v1,
)
from .v19_training_bundle_protocol_v1 import (
    BUNDLE_SCHEMA_VERSION,
    FIXED_INDEX_BYTES,
    FIXED_INDEX_RELATIVE_PATH,
    FIXED_INDEX_SHA256,
    FIXED_COUNTS,
    GRAPH_CACHE_ARTIFACT_SHA256,
    GRAPH_CACHE_RELATIVE_PATH,
    IMPLEMENTATION_SOURCE_ROSTER,
    MANIFEST_SCHEMA_VERSION,
    SHARD_SCHEMA_VERSION,
    TASK_SCOPE,
    TENSOR_FIELDS,
    PlainFileCaptureV1,
    actual_int,
    canonical_sha256,
    capture_plain_file,
    has_reparse_attribute,
    plain_child,
    plain_directory,
    read_validated_tensor_npz,
    require_sha256,
    strict_json,
    validate_graph_tensor_arrays,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts" / "development"
_LOAD_TOKEN = object()


def _capture_bound_file(
    root: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> PlainFileCaptureV1:
  if set(binding) != {"name", "bytes", "sha256"}:
    raise ValueError(f"{label} binding differs")
  path = plain_child(root, binding.get("name"), label=label, kind="file")
  captured = capture_plain_file(path, label=label)
  if (
      captured.byte_count != binding.get("bytes")
      or captured.sha256 != binding.get("sha256")
  ):
    raise ValueError(f"{label} bytes differ")
  return captured


def _declared_shard_arrays_v1(
    tensor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
  """Validate canonical shard-local member names before dictionary insertion."""

  declared: dict[str, Mapping[str, Any]] = {}
  for local_index, tensor_row in enumerate(tensor_rows):
    arrays = tensor_row.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(TENSOR_FIELDS):
      raise ValueError("training bundle tensor array field roster differs")
    prefix = f"t{local_index:04d}"
    for field in TENSOR_FIELDS:
      receipt = arrays[field]
      if (
          not isinstance(receipt, Mapping)
          or set(receipt) != {
              "archive_name", "dtype", "shape", "sha256",
          }
      ):
        raise ValueError("training bundle tensor archive receipt differs")
      archive_name = receipt.get("archive_name")
      if archive_name != f"{prefix}_{field}":
        raise ValueError("training bundle tensor archive name/prefix differs")
      if archive_name in declared:
        raise ValueError("training bundle tensor archive member is reused")
      declared[archive_name] = {
          key: receipt[key] for key in ("dtype", "shape", "sha256")
      }
  return declared


def _plain_project_source(relative_value: Any, *, label: str) -> Path:
  relative = Path(str(relative_value))
  if (
      relative.is_absolute()
      or not relative.parts
      or any(part in {"", ".", ".."} for part in relative.parts)
  ):
    raise ValueError(f"{label} relative path differs")
  root = plain_directory(PROJECT_ROOT, label="project root")
  current = root
  for part in relative.parts:
    current = current / part
    if (
        current.is_symlink()
        or not current.exists()
        or has_reparse_attribute(current)
    ):
      raise ValueError(f"{label} path is missing or redirected")
  if not current.is_file():
    raise ValueError(f"{label} source is not a file")
  try:
    current.resolve(strict=True).relative_to(root)
  except ValueError as error:
    raise ValueError(f"{label} source escapes project") from error
  return current


def _verify_self_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
  unsigned = dict(payload)
  observed = unsigned.pop(field, None)
  if observed != canonical_sha256(unsigned):
    raise ValueError(f"{label} self-hash differs")


def _quick_revalidate(captured: PlainFileCaptureV1, *, label: str) -> None:
  path = captured.path
  try:
    current = path.stat(follow_symlinks=False)
  except OSError as error:
    raise ValueError(f"{label} disappeared") from error
  if (
      path.is_symlink()
      or path.resolve(strict=True) != captured.resolved_path
      or int(current.st_dev) != captured.device
      or int(current.st_ino) != captured.inode
      or int(current.st_size) != captured.byte_count
      or int(current.st_nlink) != 1
  ):
    raise ValueError(f"{label} identity changed")


def _deep_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({
        str(key): _deep_freeze(item) for key, item in value.items()
    })
  if isinstance(value, list | tuple):
    return tuple(_deep_freeze(item) for item in value)
  return value


@dataclass(frozen=True, slots=True)
class BundleGeometryTensorV1:
  tensor_key_sha256: str
  node_features: np.ndarray
  edge_index: np.ndarray
  edge_features: np.ndarray
  center_node_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class FixedCatalogDescriptorV1:
  program_index: int
  program_id: str
  program_type: str
  descriptor: Mapping[str, str]
  descriptor_sha256: str

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("fixed catalog descriptors are not serializable")


@dataclass(frozen=True, slots=True)
class FixedTrainingSourceProvenanceV1:
  fixed_index_relative_path: str
  fixed_index_sha256: str
  fixed_index_bytes: int
  fixed_index_payload_sha256: str
  exact_graph_cache_relative_path: str
  exact_graph_cache_artifact_sha256: str
  exact_graph_cache_artifact_bytes: int
  semantic_lineage: Mapping[str, Any]

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("fixed source provenance is not serializable")


@dataclass(frozen=True, slots=True)
class V19AuthenticatedTrainingBatchV1:
  inputs: BRepProgramInputs
  targets: SetValuedProgramTargets
  global_state: BundleSetValuedTrainingStateV1
  development_split: str
  sample_indices: tuple[int, ...]

  def to(self, device: torch.device | str) -> "V19AuthenticatedTrainingBatchV1":
    return V19AuthenticatedTrainingBatchV1(
        inputs=self.inputs.to(device),
        targets=self.targets,
        global_state=self.global_state,
        development_split=self.development_split,
        sample_indices=self.sample_indices,
    )

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated training batches are not serializable")


class LoadedV19AuthenticatedTrainingBundleV1:
  """Immutable, non-serializable Torch-side capability."""

  __slots__ = (
      "_root",
      "_artifact_sha256",
      "_captures",
      "_capture_times",
      "_samples",
      "_work",
      "_tensors",
      "_state",
      "_train_indices",
      "_dev_indices",
      "_catalog",
      "_catalog_descriptors",
      "_source_provenance",
      "_sealed",
  )

  def __init__(
      self,
      *,
      root: Path,
      artifact_sha256: str,
      captures: Sequence[PlainFileCaptureV1],
      samples: Sequence[SetValuedProgramSampleV1],
      work: Mapping[str, tuple[str, str]],
      tensors: Mapping[str, BundleGeometryTensorV1],
      state: BundleSetValuedTrainingStateV1,
      catalog: Mapping[str, Any],
      source_provenance: FixedTrainingSourceProvenanceV1,
      _token: object,
  ) -> None:
    if _token is not _LOAD_TOKEN:
      raise TypeError("training bundle handles are loader-only")
    object.__setattr__(self, "_root", root)
    object.__setattr__(self, "_artifact_sha256", artifact_sha256)
    object.__setattr__(self, "_captures", tuple(captures))
    object.__setattr__(
        self,
        "_capture_times",
        tuple(
            (
                int(capture.path.stat(follow_symlinks=False).st_mtime_ns),
                int(capture.path.stat(follow_symlinks=False).st_ctime_ns),
            )
            for capture in captures
        ),
    )
    object.__setattr__(self, "_samples", tuple(samples))
    object.__setattr__(self, "_work", MappingProxyType(dict(work)))
    object.__setattr__(self, "_tensors", MappingProxyType(dict(tensors)))
    object.__setattr__(self, "_state", state)
    object.__setattr__(self, "_catalog", _deep_freeze(catalog))
    object.__setattr__(
        self,
        "_catalog_descriptors",
        tuple(
            FixedCatalogDescriptorV1(
                program_index=int(row["program_index"]),
                program_id=str(row["program_id"]),
                program_type=str(row["program_type"]),
                descriptor=_deep_freeze(row["descriptor"]),
                descriptor_sha256=str(row["descriptor_sha256"]),
            )
            for row in catalog["entries"]
        ),
    )
    if type(source_provenance) is not FixedTrainingSourceProvenanceV1:
      raise TypeError("fixed source provenance type differs")
    object.__setattr__(self, "_source_provenance", source_provenance)
    object.__setattr__(
        self,
        "_train_indices",
        tuple(
            index for index, sample in enumerate(samples)
            if sample.development_split == "train"
            and sample.classification_evaluable
        ),
    )
    object.__setattr__(
        self,
        "_dev_indices",
        tuple(
            index for index, sample in enumerate(samples)
            if sample.development_split == "dev"
        ),
    )
    object.__setattr__(self, "_sealed", True)

  def __setattr__(self, _name: str, _value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("training bundle handle is immutable")
    object.__setattr__(self, _name, _value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("training bundle handles are not serializable")

  @property
  def artifact_sha256(self) -> str:
    return self._artifact_sha256

  @property
  def counts(self) -> Mapping[str, int]:
    return MappingProxyType({
        **FIXED_COUNTS,
        "train_usable_sample_count": len(self._train_indices),
        "dev_evaluation_sample_count": len(self._dev_indices),
    })

  @property
  def catalog_descriptors(self) -> tuple[FixedCatalogDescriptorV1, ...]:
    """Fixed candidate encoder metadata; contains no split/cluster identity."""

    return self._catalog_descriptors

  @property
  def source_provenance(self) -> FixedTrainingSourceProvenanceV1:
    """Audit-only fixed index/cache lineage, never a model feature channel."""

    return self._source_provenance

  def revalidate(self) -> None:
    for capture, expected_times in zip(
        self._captures, self._capture_times, strict=True
    ):
      _quick_revalidate(capture, label="training bundle file")
      current = capture.path.stat(follow_symlinks=False)
      if (
          int(current.st_mtime_ns),
          int(current.st_ctime_ns),
      ) != expected_times:
        raise ValueError("training bundle file timestamps changed")

  def training_indices(self) -> tuple[int, ...]:
    return self._train_indices

  def development_evaluation_indices(self) -> tuple[int, ...]:
    return self._dev_indices

  def _graph(self, tensor: BundleGeometryTensorV1, *, endpoint: int) -> BRepGraphExample:
    marker = np.zeros((tensor.node_features.shape[0], 2), dtype=np.float32)
    marker[tensor.center_node_indices, endpoint] = 1.0
    nodes = np.concatenate(
        (tensor.node_features.astype(np.float32, copy=False), marker), axis=1
    )
    if nodes.shape[1] != 21:
      raise ValueError("bundle node19+endpoint2 feature contract differs")
    return BRepGraphExample(
        node_features=torch.from_numpy(nodes.copy()),
        edge_index=torch.from_numpy(tensor.edge_index.astype(np.int64, copy=True)),
        edge_features=torch.from_numpy(
            tensor.edge_features.astype(np.float32, copy=True)
        ),
        endpoint_markers=torch.from_numpy(marker.copy()),
    )

  def collate(
      self,
      sample_indices: Sequence[int],
  ) -> V19AuthenticatedTrainingBatchV1:
    self.revalidate()
    indices = tuple(sample_indices)
    if (
        not indices
        or len(indices) > 32
        or any(type(value) is not int for value in indices)
        or len(indices) != len(set(indices))
        or any(value < 0 or value >= len(self._samples) for value in indices)
    ):
      raise ValueError("training bundle batch indices differ")
    selected = tuple(self._samples[index] for index in indices)
    splits = {sample.development_split for sample in selected}
    if len(splits) != 1:
      raise ValueError("training bundle batch cannot mix train and dev")
    split = next(iter(splits))
    if split == "train" and any(
        not sample.classification_evaluable for sample in selected
    ):
      raise ValueError("train OOV-only rows are excluded from batching")
    examples: list[BRepLearnerExample] = []
    for sample in selected:
      endpoint_keys = self._work.get(sample.graph_work_id)
      if endpoint_keys is None:
        raise ValueError("bundle sample lost its graph-work join")
      known = sample.known_catalog_targets
      target_index = known[0].catalog_index if known else 0
      twist = (
          known[0].alternatives[0].residual_twist_pair_local
          if known else (0.0,) * 6
      )
      examples.append(BRepLearnerExample(
          graphs=(
              self._graph(self._tensors[endpoint_keys[0]], endpoint=0),
              self._graph(self._tensors[endpoint_keys[1]], endpoint=1),
          ),
          candidate_program_indices=torch.arange(19, dtype=torch.long),
          target_program_index=target_index,
          residual_translation=torch.tensor(twist[:3], dtype=torch.float32),
          residual_rotation_vector=torch.tensor(twist[3:], dtype=torch.float32),
          residual_mask=bool(known),
          model_view_sha256=sample.graph_work_id,
      ))
    learner_batch = collate_nonformal_brep_learner_examples(
        examples,
        budget=LearnerBatchBudget(max_batch_size=32, max_candidates=19),
    )
    return V19AuthenticatedTrainingBatchV1(
        inputs=learner_batch.inputs,
        targets=SetValuedProgramTargets(samples=selected, catalog_size=19),
        global_state=self._state,
        development_split=split,
        sample_indices=indices,
    )


def _sample_from_payload(row: Mapping[str, Any]) -> SetValuedProgramSampleV1:
  groups: dict[int, tuple[str, list[ProgramResidualAlternativeV1]]] = {}
  oov: list[OOVProgramAlternativeV1] = []
  for target in row["targets"]:
    twist = tuple(float(value) for value in (
        *target["residual_translation"],
        *rotation_matrix_to_vector_v1(target["residual_rotation_row_major"]),
    ))
    if target["oov"] is True:
      oov.append(OOVProgramAlternativeV1(
          semantic_program_id=str(target["program_id"]),
          residual_twist_pair_local=twist,
      ))
    else:
      index = actual_int(target["catalog_index"], label="catalog target index")
      program_id = str(target["catalog_program_id"])
      existing = groups.setdefault(index, (program_id, []))
      if existing[0] != program_id:
        raise ValueError("catalog target program ID differs")
      existing[1].append(ProgramResidualAlternativeV1(
          semantic_program_id=str(target["program_id"]),
          residual_twist_pair_local=twist,
      ))
  sample = SetValuedProgramSampleV1(
      opaque_query_identity=require_sha256(
          row["opaque_query_identity"], label="opaque query identity"
      ),
      direction=str(row["direction"]),
      graph_work_id=require_sha256(row["graph_work_id"], label="graph work ID"),
      development_split=str(row["development_split"]),
      family_cluster_id=require_sha256(
          row["family_cluster_id"], label="family cluster ID"
      ),
      assembly_cluster_id=require_sha256(
          row["assembly_cluster_id"], label="assembly cluster ID"
      ),
      known_catalog_targets=tuple(
          CatalogTargetAlternativesV1(
              catalog_index=index,
              catalog_program_id=value[0],
              alternatives=tuple(value[1]),
          )
          for index, value in groups.items()
      ),
      oov_targets=tuple(oov),
  )
  if (
      row["classification_evaluable"] is not sample.classification_evaluable
      or row["target_count"]
      != sum(len(group.alternatives) for group in sample.known_catalog_targets)
      + len(sample.oov_targets)
  ):
    raise ValueError("bundle sample target accounting differs")
  return sample


def load_v19_authenticated_training_bundle_v1(
    directory: str | Path,
    *,
    expected_artifact_sha256: str,
) -> LoadedV19AuthenticatedTrainingBundleV1:
  """Load only a pinned development bundle; no authority code is imported."""

  require_sha256(expected_artifact_sha256, label="training bundle artifact")
  development_root = plain_directory(DEVELOPMENT_ROOT, label="development root")
  root = plain_directory(directory, label="training bundle")
  if root.parent != development_root:
    raise ValueError("training bundle must be a direct development child")
  if any(
      token in root.name.casefold() for token in ("formal", "final", "withheld")
  ):
    raise ValueError("training bundle path crosses development boundary")
  artifact_capture = capture_plain_file(
      plain_child(root, "artifact.json", label="bundle artifact", kind="file"),
      label="bundle artifact",
  )
  if artifact_capture.sha256 != expected_artifact_sha256:
    raise ValueError("training bundle artifact pin differs")
  artifact = strict_json(artifact_capture.raw_bytes, label="bundle artifact")
  if (
      not isinstance(artifact, Mapping)
      or artifact.get("schema_version") != BUNDLE_SCHEMA_VERSION
      or artifact.get("task_scope") != TASK_SCOPE
      or artifact.get("development") is not True
      or artifact.get("formal") is not False
      or artifact.get("final_test_touched") is not False
      or artifact.get("withheld_test_touched") is not False
  ):
    raise ValueError("training bundle artifact boundary differs")
  _verify_self_hash(artifact, field="artifact_payload_sha256", label="artifact")
  manifest_capture = _capture_bound_file(
      root, artifact["manifest"], label="bundle manifest"
  )
  manifest = strict_json(manifest_capture.raw_bytes, label="bundle manifest")
  if (
      not isinstance(manifest, Mapping)
      or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
      or manifest.get("task_scope") != TASK_SCOPE
      or manifest.get("development") is not True
      or manifest.get("formal") is not False
      or manifest.get("publication_eligible") is not False
      or manifest.get("final_test_touched") is not False
      or manifest.get("withheld_test_touched") is not False
      or manifest.get("fixed_counts") != FIXED_COUNTS
  ):
    raise ValueError("training bundle manifest boundary differs")
  _verify_self_hash(manifest, field="manifest_payload_sha256", label="manifest")

  captures: list[PlainFileCaptureV1] = [artifact_capture, manifest_capture]
  sources = manifest["source_bindings"]
  fixed_index = sources["fixed_index"]
  graph_cache = sources["exact_graph_cache_artifact"]
  for binding, expected, label in (
      (fixed_index, FIXED_INDEX_SHA256, "fixed index"),
      (graph_cache, GRAPH_CACHE_ARTIFACT_SHA256, "exact graph cache"),
  ):
    candidate = _plain_project_source(
        binding["relative_path"], label=label
    )
    captured = capture_plain_file(candidate, label=label)
    if (
        captured.sha256 != binding["sha256"]
        or captured.byte_count != binding["bytes"]
        or (expected is not None and captured.sha256 != expected)
        or (
            label == "fixed index"
            and (
                binding["relative_path"] != FIXED_INDEX_RELATIVE_PATH
                or captured.byte_count != FIXED_INDEX_BYTES
            )
        )
        or (
            label == "exact graph cache"
            and binding["relative_path"] != GRAPH_CACHE_RELATIVE_PATH
        )
    ):
      raise ValueError(f"{label} source binding differs")
    captures.append(captured)
    if label == "fixed index":
      fixed_index_payload = strict_json(
          captured.raw_bytes, label="fixed index source"
      )
  implementation = sources["implementation"]
  if (
      not isinstance(implementation, list)
      or [row.get("relative_path") for row in implementation]
      != list(IMPLEMENTATION_SOURCE_ROSTER)
  ):
    raise ValueError("implementation source roster/order differs")
  for binding in implementation:
    captured = capture_plain_file(
        _plain_project_source(
            binding["relative_path"], label="training implementation source"
        ),
        label="training implementation source",
    )
    if (
        captured.sha256 != binding["sha256"]
        or captured.byte_count != binding["bytes"]
    ):
      raise ValueError("training implementation source binding differs")
    captures.append(captured)
  if (
      not isinstance(fixed_index_payload, Mapping)
      or manifest["samples"] != fixed_index_payload.get("samples")
      or manifest["catalog"] != fixed_index_payload.get("program_catalog")
      or manifest["lineage"] != fixed_index_payload.get("source_authority")
  ):
    raise ValueError("bundle supervision/catalog differs from fixed index bytes")

  tensors: dict[str, BundleGeometryTensorV1] = {}
  roster = manifest["tensor_key_roster"]
  next_start = 0
  declared_root = {"artifact.json", "manifest.json"}
  for expected_shard_index, shard_row in enumerate(manifest["shards"]):
    if (
        shard_row["shard_index"] != expected_shard_index
        or shard_row["tensor_start"] != next_start
    ):
      raise ValueError("training bundle shard partition differs")
    shard = plain_child(
        root, shard_row["directory_name"], label="bundle shard", kind="directory"
    )
    declared_root.add(shard.name)
    commit_capture = _capture_bound_file(
        shard, shard_row["commit"], label="bundle shard commit"
    )
    commit = strict_json(commit_capture.raw_bytes, label="bundle shard commit")
    _verify_self_hash(commit, field="commit_payload_sha256", label="shard commit")
    metadata_capture = _capture_bound_file(
        shard, commit["metadata"], label="bundle shard metadata"
    )
    tensor_capture = _capture_bound_file(
        shard, commit["tensors"], label="bundle shard tensors"
    )
    metadata = strict_json(metadata_capture.raw_bytes, label="bundle shard metadata")
    _verify_self_hash(metadata, field="metadata_payload_sha256", label="shard metadata")
    count = shard_row["tensor_count"]
    if (
        metadata["schema_version"] != SHARD_SCHEMA_VERSION
        or metadata["shard_index"] != expected_shard_index
        or metadata["tensor_start"] != next_start
        or metadata["tensor_count"] != count
        or len(metadata["tensors"]) != count
    ):
      raise ValueError("training bundle shard metadata differs")
    declared_arrays = _declared_shard_arrays_v1(metadata["tensors"])
    arrays = read_validated_tensor_npz(
        tensor_capture.raw_bytes, declared=declared_arrays
    )
    for local_index, tensor_row in enumerate(metadata["tensors"]):
      key = require_sha256(
          tensor_row["tensor_key_sha256"], label="bundle tensor key"
      )
      expected_key = roster[next_start + local_index]
      if key != expected_key or key in tensors:
        raise ValueError("training bundle tensor roster differs")
      fields = dict(validate_graph_tensor_arrays({
          field: arrays[receipt["archive_name"]]
          for field, receipt in tensor_row["arrays"].items()
      }))
      node = fields["node_features"]
      edge_index = fields["edge_index"]
      edge = fields["edge_features"]
      centers = fields["center_node_indices"]
      tensors[key] = BundleGeometryTensorV1(
          tensor_key_sha256=key,
          node_features=node,
          edge_index=edge_index,
          edge_features=edge,
          center_node_indices=centers,
      )
    next_start += count
    captures.extend((commit_capture, metadata_capture, tensor_capture))
  if (
      next_start != FIXED_COUNTS["tensor_count"]
      or list(tensors) != roster
      or {path.name for path in root.iterdir()} != declared_root
  ):
    raise ValueError("training bundle tensor/root coverage differs")

  work: dict[str, tuple[str, str]] = {}
  for row in manifest["graph_work"]:
    work_id = require_sha256(row["graph_work_id"], label="bundle graph work")
    endpoints = (
        require_sha256(
            row["endpoint_a_tensor_key_sha256"], label="endpoint A tensor"
        ),
        require_sha256(
            row["endpoint_b_tensor_key_sha256"], label="endpoint B tensor"
        ),
    )
    if work_id in work or any(key not in tensors for key in endpoints):
      raise ValueError("training bundle graph-work join differs")
    work[work_id] = endpoints
  samples = tuple(_sample_from_payload(row) for row in manifest["samples"])
  if (
      len(work) != FIXED_COUNTS["graph_work_count"]
      or len(samples) != FIXED_COUNTS["graph_work_count"]
      or {sample.graph_work_id for sample in samples} != set(work)
      or sum(
          len(group.alternatives)
          for sample in samples
          for group in sample.known_catalog_targets
      ) + sum(len(sample.oov_targets) for sample in samples)
      != FIXED_COUNTS["target_count"]
  ):
    raise ValueError("training bundle sample/work coverage differs")
  catalog = manifest["catalog"]
  statistics = catalog["training_statistics"]
  state = build_bundle_set_valued_training_state_v1(
      samples=samples,
      class_weights=statistics["class_weight_by_program_index"],
      catalog_payload_sha256=catalog["catalog_payload_sha256"],
      source_commitment_sha256=statistics["source_commitment_sha256"],
  )
  return LoadedV19AuthenticatedTrainingBundleV1(
      root=root,
      artifact_sha256=expected_artifact_sha256,
      captures=captures,
      samples=samples,
      work=work,
      tensors=tensors,
      state=state,
      catalog=catalog,
      source_provenance=FixedTrainingSourceProvenanceV1(
          fixed_index_relative_path=str(fixed_index["relative_path"]),
          fixed_index_sha256=str(fixed_index["sha256"]),
          fixed_index_bytes=int(fixed_index["bytes"]),
          fixed_index_payload_sha256=str(
              fixed_index["index_payload_sha256"]
          ),
          exact_graph_cache_relative_path=str(
              graph_cache["relative_path"]
          ),
          exact_graph_cache_artifact_sha256=str(graph_cache["sha256"]),
          exact_graph_cache_artifact_bytes=int(graph_cache["bytes"]),
          semantic_lineage=_deep_freeze(manifest["lineage"]),
      ),
      _token=_LOAD_TOKEN,
  )


def v19_brep_set_valued_training_loss_v1(
    model: BRepProgramLearnerV1,
    batch: V19AuthenticatedTrainingBatchV1,
    *,
    residual_lambda: float = 1.0,
) -> JointSetProgramResidualLossV1:
  """Forward the fixed19 learner and apply the existing set-valued objective."""

  if type(model) is not BRepProgramLearnerV1:
    raise TypeError("V19 training wrapper requires BRepProgramLearnerV1")
  if type(batch) is not V19AuthenticatedTrainingBatchV1:
    raise TypeError("V19 training wrapper requires an authenticated bundle batch")
  output = model(batch.inputs)
  residual = torch.cat(
      (output.residual_translation, output.residual_rotation_vector), dim=-1
  )
  expected = (len(batch.targets.samples), 19)
  if output.logits.shape != expected or residual.shape != (*expected, 6):
    raise ValueError("V19 learner output shape differs from fixed19 contract")
  return set_valued_program_training_loss_v1(
      output.logits,
      residual,
      batch.targets,
      global_cluster_weights=batch.global_state.global_cluster_weights,
      catalog_class_weights=(
          batch.global_state.catalog_class_weights
          if batch.development_split == "train"
          else None
      ),
      residual_lambda=residual_lambda,
  )
