"""Single benchmark-v2 B-Rep learner for finite mate-program prediction.

The module has one deliberately narrow data boundary: real examples can only
be tensorized from ``AuthenticatedBenchmarkV2ModelViewV2`` capabilities.  The
network sees intrinsic face/topology features, endpoint marks and opaque
train-catalog indices; targets are held in a separate object and are never
passed to ``forward``.

The dataset is lazy by design.  It materializes one authority-backed case and
one semantic row at a time.  A future optimized all-row materializer may cache
these already-sanitized tensors, but must preserve the same capability and
checkpoint bindings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence
import weakref

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .benchmark_v2_model_view_v2 import (
    EDGE_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
    SCHEMA_VERSION as MODEL_VIEW_SCHEMA_VERSION,
    AuthenticatedBenchmarkV2DevelopmentIndex,
    AuthenticatedBenchmarkV2ModelViewV2,
    GraphSizeBudget,
)
from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    capture_file_artifact,
    reverify_captured_file_artifact,
)
from .mate_pose_retriever import MateProgramRetrieval
from .mate_programs import MateProgram
from .program_class_reranker import mate_program_class_from_program
from .program_proposer import (
    PROGRAM_PROPOSER_PROTOCOL,
    ProgramHypothesis,
    ProposalBudget,
)


LEARNER_SCHEMA_VERSION = "brep_program_learner.v1"
CHECKPOINT_SCHEMA_VERSION = "brep_program_learner_inference_checkpoint.v2"
CHECKPOINT_METADATA_SCHEMA_VERSION = "brep_program_learner_inference_metadata.v2"
CHECKPOINT_ARTIFACT_SCHEMA_VERSION = "brep_program_learner_inference_artifact.v1"
FORMAL_CATALOG_SIZE = 19
SURFACE_TYPES = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "spline",
    "other",
)
NODE_INPUT_DIM = len(NODE_FEATURE_NAMES) * 2 + len(SURFACE_TYPES) + 2
EDGE_INPUT_DIM = len(EDGE_FEATURE_NAMES)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_AUTHENTICATED_EXAMPLE_FACTORY_TOKEN = object()
_LOADED_CHECKPOINT_FACTORY_TOKEN = object()


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


def _reject_reparse_path_chain(path: Path, *, label: str) -> None:
  """Reject symlinks/junctions/reparse points anywhere in an existing path."""

  absolute = path.absolute()
  candidates = [absolute, *absolute.parents]
  for candidate in candidates:
    if not candidate.exists():
      continue
    try:
      stat_result = os.lstat(candidate)
    except OSError as error:
      raise ValueError(f"{label} path is unreadable") from error
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if candidate.is_symlink() or bool(attributes & reparse_flag):
      raise ValueError(f"{label} path must not contain symlink/reparse components")


def _captured_binding(captured: CapturedFileArtifact, *, name: str) -> dict[str, Any]:
  return {
      "name": name,
      "bytes": captured.byte_count,
      "sha256": captured.sha256,
  }


def seed_brep_learner(seed: int) -> None:
  """Set all learner RNGs and request deterministic kernels where available."""

  if type(seed) is not int or seed < 0:
    raise ValueError("learner seed must be a non-negative actual integer")
  random.seed(seed)
  np.random.seed(seed % (2**32))
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True, slots=True)
class BRepProgramLearnerConfig:
  catalog_size: int = FORMAL_CATALOG_SIZE
  hidden_dim: int = 96
  layers: int = 4
  dropout: float = 0.0
  translation_loss_weight: float = 0.25
  rotation_loss_weight: float = 0.25

  def __post_init__(self) -> None:
    if type(self.catalog_size) is not int or self.catalog_size < 2:
      raise ValueError("catalog_size must be an actual integer of at least two")
    if type(self.hidden_dim) is not int or self.hidden_dim < 8:
      raise ValueError("hidden_dim must be an actual integer of at least eight")
    if type(self.layers) is not int or self.layers < 1:
      raise ValueError("layers must be a positive actual integer")
    if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
      raise ValueError("dropout must be finite in [0, 1)")
    for label, value in (
        ("translation_loss_weight", self.translation_loss_weight),
        ("rotation_loss_weight", self.rotation_loss_weight),
    ):
      if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LearnerBatchBudget:
  max_batch_size: int = 32
  max_graphs: int = 2
  max_faces_per_graph: int = 1024
  max_edges_per_graph: int = 4096
  max_candidates: int = FORMAL_CATALOG_SIZE

  def __post_init__(self) -> None:
    values = asdict(self)
    if any(type(value) is not int or value < 1 for value in values.values()):
      raise ValueError("learner batch budgets must be positive actual integers")
    if self.max_graphs > 2 or self.max_faces_per_graph > 1024:
      raise ValueError("learner batch graph budget exceeds model-view policy")
    if self.max_edges_per_graph > 4096 or self.max_candidates > 512:
      raise ValueError("learner batch edge/candidate budget exceeds model-view policy")


@dataclass(frozen=True, slots=True)
class BRepGraphExample:
  node_features: Tensor
  edge_index: Tensor
  edge_features: Tensor
  endpoint_markers: Tensor


@dataclass(frozen=True, slots=True)
class BRepLearnerExample:
  graphs: tuple[BRepGraphExample, ...]
  candidate_program_indices: Tensor
  target_program_index: int
  residual_translation: Tensor
  residual_rotation_vector: Tensor
  residual_mask: bool = True
  model_view_sha256: str = "synthetic"


class AuthenticatedBRepLearnerExample:
  """Non-serializable lineage wrapper produced only from a real capability."""

  __slots__ = ("_capability", "_example", "_lineage", "__weakref__")

  def __init__(
      self,
      *,
      capability: AuthenticatedBenchmarkV2ModelViewV2,
      example: BRepLearnerExample,
      lineage: Mapping[str, Any],
      _factory_token: object,
  ) -> None:
    if _factory_token is not _AUTHENTICATED_EXAMPLE_FACTORY_TOKEN:
      raise TypeError("authenticated learner examples are factory-only")
    self._capability = capability
    self._example = example
    self._lineage = MappingProxyType(dict(lineage))

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated learner examples are not serializable")

  @property
  def example(self) -> BRepLearnerExample:
    self.revalidate()
    return self._example

  @property
  def lineage(self) -> Mapping[str, Any]:
    self.revalidate()
    return self._lineage

  def revalidate(self) -> None:
    if type(self._capability) is not AuthenticatedBenchmarkV2ModelViewV2:
      raise TypeError("learner example lost its authenticated capability")
    self._capability.revalidate()
    if self._capability.sha256 != self._example.model_view_sha256:
      raise ValueError("learner example differs from capability lineage")


@dataclass(frozen=True, slots=True)
class BRepProgramInputs:
  node_features: Tensor
  node_mask: Tensor
  edge_index: Tensor
  edge_features: Tensor
  edge_mask: Tensor
  endpoint_markers: Tensor
  graph_mask: Tensor
  candidate_program_indices: Tensor
  candidate_mask: Tensor

  def to(self, device: torch.device | str) -> "BRepProgramInputs":
    return BRepProgramInputs(
        node_features=self.node_features.to(device),
        node_mask=self.node_mask.to(device),
        edge_index=self.edge_index.to(device),
        edge_features=self.edge_features.to(device),
        edge_mask=self.edge_mask.to(device),
        endpoint_markers=self.endpoint_markers.to(device),
        graph_mask=self.graph_mask.to(device),
        candidate_program_indices=self.candidate_program_indices.to(device),
        candidate_mask=self.candidate_mask.to(device),
    )


@dataclass(frozen=True, slots=True)
class BRepProgramTargets:
  target_positions: Tensor
  residual_translation: Tensor
  residual_rotation_vector: Tensor
  residual_mask: Tensor

  def to(self, device: torch.device | str) -> "BRepProgramTargets":
    return BRepProgramTargets(
        target_positions=self.target_positions.to(device),
        residual_translation=self.residual_translation.to(device),
        residual_rotation_vector=self.residual_rotation_vector.to(device),
        residual_mask=self.residual_mask.to(device),
    )


@dataclass(frozen=True, slots=True)
class BRepProgramBatch:
  inputs: BRepProgramInputs
  targets: BRepProgramTargets
  model_view_sha256s: tuple[str, ...]

  def to(self, device: torch.device | str) -> "BRepProgramBatch":
    return BRepProgramBatch(
        inputs=self.inputs.to(device),
        targets=self.targets.to(device),
        model_view_sha256s=self.model_view_sha256s,
    )


@dataclass(frozen=True, slots=True)
class BRepProgramOutput:
  logits: Tensor
  residual_translation: Tensor
  residual_rotation_vector: Tensor


@dataclass(frozen=True, slots=True)
class BRepProgramLoss:
  total: Tensor
  classification: Tensor
  residual_translation: Tensor
  residual_rotation: Tensor


def _rotation_matrix_to_vector(matrix: Sequence[float]) -> Tensor:
  rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
  cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
  angle = math.acos(cosine)
  if angle < 1e-8:
    vector = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / 2.0
  elif math.pi - angle < 1e-5:
    diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
    axis = np.sqrt(diagonal)
    largest = int(np.argmax(axis))
    if axis[largest] < 1e-8:
      axis = np.asarray([1.0, 0.0, 0.0])
    else:
      for index in range(3):
        if index != largest:
          axis[index] = (
              rotation[largest, index] + rotation[index, largest]
          ) / (4.0 * axis[largest])
      axis /= np.linalg.norm(axis)
    vector = angle * axis
  else:
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    vector = angle * axis
  return torch.tensor(vector, dtype=torch.float32)


def rotation_vector_to_matrix(vector: Tensor) -> Tensor:
  """Differentiable SO(3) exponential map for program-local residuals."""

  if vector.shape[-1] != 3:
    raise ValueError("rotation vectors must have a final dimension of three")
  theta2 = torch.sum(vector * vector, dim=-1, keepdim=True)
  theta = torch.sqrt(torch.clamp(theta2, min=1e-16))
  a = torch.where(
      theta2 < 1e-8,
      1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
      torch.sin(theta) / theta,
  )
  b = torch.where(
      theta2 < 1e-8,
      0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
      (1.0 - torch.cos(theta)) / theta2,
  )
  x, y, z = vector.unbind(dim=-1)
  zero = torch.zeros_like(x)
  skew = torch.stack(
      (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
  ).reshape(*vector.shape[:-1], 3, 3)
  identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
  identity = identity.expand(*vector.shape[:-1], 3, 3)
  return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def _graph_from_payload(
    graph: Mapping[str, Any],
    *,
    endpoint_a: Mapping[str, Any],
    endpoint_b: Mapping[str, Any],
    graph_index: int,
) -> BRepGraphExample:
  nodes = graph["nodes"]
  node_rows: list[list[float]] = []
  endpoint_a_faces = (
      set(endpoint_a["face_indices"])
      if endpoint_a["graph_index"] == graph_index
      else set()
  )
  endpoint_b_faces = (
      set(endpoint_b["face_indices"])
      if endpoint_b["graph_index"] == graph_index
      else set()
  )
  markers: list[list[float]] = []
  for node_index, node in enumerate(nodes):
    surface = [float(node["surface_type"] == item) for item in SURFACE_TYPES]
    node_rows.append(
        [float(value) for value in node["features"]]
        + [float(value) for value in node["feature_mask"]]
        + surface
        + [
            float(node_index in endpoint_a_faces),
            float(node_index in endpoint_b_faces),
        ]
    )
    markers.append(node_rows[-1][-2:])
  edges = graph["edges"]
  edge_index = torch.tensor(
      [[row["source"], row["target"]] for row in edges], dtype=torch.long
  )
  if not edges:
    edge_index = torch.zeros((0, 2), dtype=torch.long)
  edge_features = torch.tensor(
      [row["features"] for row in edges], dtype=torch.float32
  )
  if not edges:
    edge_features = torch.zeros((0, EDGE_INPUT_DIM), dtype=torch.float32)
  return BRepGraphExample(
      node_features=torch.tensor(node_rows, dtype=torch.float32),
      edge_index=edge_index,
      edge_features=edge_features,
      endpoint_markers=torch.tensor(markers, dtype=torch.float32),
  )


def tensorize_authenticated_model_view(
    capability: AuthenticatedBenchmarkV2ModelViewV2,
) -> AuthenticatedBRepLearnerExample:
  """Convert one authenticated v2 capability; bare/fake payloads fail closed."""

  if type(capability) is not AuthenticatedBenchmarkV2ModelViewV2:
    raise TypeError(
        "learner examples require an authenticated benchmark-v2 model-view "
        "capability; bare or fake model views are forbidden"
    )
  capability.revalidate()
  payload = capability.model_view
  if payload.get("schema_version") != MODEL_VIEW_SCHEMA_VERSION:
    raise ValueError("authenticated learner input uses another model-view schema")
  examples = payload.get("examples")
  if not isinstance(examples, list) or len(examples) != 1:
    raise ValueError("lazy learner input must contain exactly one semantic row")
  example = examples[0]
  endpoint_a = example["query"]["endpoint_a"]
  endpoint_b = example["query"]["endpoint_b"]
  graphs = tuple(
      _graph_from_payload(
          graph,
          endpoint_a=endpoint_a,
          endpoint_b=endpoint_b,
          graph_index=index,
      )
      for index, graph in enumerate(payload["graphs"])
  )
  candidates = example["program_candidates"]
  target = example["target"]
  tensor_example = BRepLearnerExample(
      graphs=graphs,
      candidate_program_indices=torch.tensor(
          [row["program_index"] for row in candidates], dtype=torch.long
      ),
      target_program_index=int(target["program_index"]),
      residual_translation=torch.tensor(
          target["residual_translation"], dtype=torch.float32
      ),
      residual_rotation_vector=_rotation_matrix_to_vector(
          target["residual_rotation_row_major"]
      ),
      model_view_sha256=capability.sha256,
  )
  lineage = {
      "schema_version": "authenticated_brep_learner_example_lineage.v1",
      "model_view_sha256": capability.sha256,
      "model_view_schema_version": payload["schema_version"],
      "catalog_descriptor_manifest_sha256": _canonical_sha256(
          _catalog_descriptor_manifest(payload)
      ),
      "local_subgraph_policy_sha256": _canonical_sha256(
          payload["normalization"]["local_interface_subgraph_policy"]
      ),
  }
  lineage["lineage_payload_sha256"] = _canonical_sha256(lineage)
  return AuthenticatedBRepLearnerExample(
      capability=capability,
      example=tensor_example,
      lineage=lineage,
      _factory_token=_AUTHENTICATED_EXAMPLE_FACTORY_TOKEN,
  )


def _validate_graph_example(graph: BRepGraphExample, *, label: str) -> None:
  if graph.node_features.ndim != 2 or graph.node_features.shape[1] != NODE_INPUT_DIM:
    raise ValueError(f"{label} node feature shape differs")
  node_count = graph.node_features.shape[0]
  if node_count < 1 or not torch.isfinite(graph.node_features).all():
    raise ValueError(f"{label} nodes must be non-empty and finite")
  if graph.edge_index.ndim != 2 or graph.edge_index.shape[1] != 2:
    raise ValueError(f"{label} edge index shape differs")
  if graph.edge_features.shape != (graph.edge_index.shape[0], EDGE_INPUT_DIM):
    raise ValueError(f"{label} edge feature shape differs")
  if graph.endpoint_markers.shape != (node_count, 2):
    raise ValueError(f"{label} endpoint marker shape differs")
  if graph.edge_index.numel() and (
      int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= node_count
  ):
    raise ValueError(f"{label} edge index is out of range")


def collate_nonformal_brep_learner_examples(
    examples: Sequence[BRepLearnerExample],
    *,
    budget: LearnerBatchBudget = LearnerBatchBudget(),
) -> BRepProgramBatch:
  """Pad explicit synthetic/nonformal rows; formal training must not call this."""

  if not examples or len(examples) > budget.max_batch_size:
    raise ValueError("learner batch is empty or exceeds its hard batch budget")
  for example_index, example in enumerate(examples):
    if not isinstance(example, BRepLearnerExample):
      raise TypeError("learner collator accepts BRepLearnerExample rows only")
    if not 1 <= len(example.graphs) <= budget.max_graphs:
      raise ValueError("learner graph count exceeds its hard budget")
    for graph_index, graph in enumerate(example.graphs):
      _validate_graph_example(graph, label=f"example[{example_index}].graph[{graph_index}]")
      if graph.node_features.shape[0] > budget.max_faces_per_graph:
        raise ValueError("learner face count exceeds its hard budget")
      if graph.edge_index.shape[0] > budget.max_edges_per_graph:
        raise ValueError("learner edge count exceeds its hard budget")
    candidates = example.candidate_program_indices
    if (
        candidates.ndim != 1
        or not 2 <= candidates.shape[0] <= budget.max_candidates
        or candidates.dtype != torch.long
        or len(set(int(value) for value in candidates.tolist())) != candidates.shape[0]
    ):
      raise ValueError("learner candidates are malformed or exceed their hard budget")
    if example.target_program_index not in candidates.tolist():
      raise ValueError("learner target program is absent from candidate IDs")
    if example.residual_translation.shape != (3,) or example.residual_rotation_vector.shape != (3,):
      raise ValueError("learner residual target shape differs")

  batch_size = len(examples)
  max_graphs = max(len(row.graphs) for row in examples)
  max_nodes = max(graph.node_features.shape[0] for row in examples for graph in row.graphs)
  max_edges = max(graph.edge_index.shape[0] for row in examples for graph in row.graphs)
  max_candidates = max(row.candidate_program_indices.shape[0] for row in examples)
  nodes = torch.zeros((batch_size, max_graphs, max_nodes, NODE_INPUT_DIM))
  node_mask = torch.zeros((batch_size, max_graphs, max_nodes), dtype=torch.bool)
  edges = torch.zeros((batch_size, max_graphs, max_edges, 2), dtype=torch.long)
  edge_features = torch.zeros((batch_size, max_graphs, max_edges, EDGE_INPUT_DIM))
  edge_mask = torch.zeros((batch_size, max_graphs, max_edges), dtype=torch.bool)
  markers = torch.zeros((batch_size, max_graphs, max_nodes, 2))
  graph_mask = torch.zeros((batch_size, max_graphs), dtype=torch.bool)
  candidate_ids = torch.zeros((batch_size, max_candidates), dtype=torch.long)
  candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
  target_positions = torch.zeros((batch_size,), dtype=torch.long)
  translations = torch.zeros((batch_size, 3))
  rotations = torch.zeros((batch_size, 3))
  residual_mask = torch.zeros((batch_size,), dtype=torch.bool)
  hashes: list[str] = []
  for batch_index, example in enumerate(examples):
    for graph_index, graph in enumerate(example.graphs):
      node_count = graph.node_features.shape[0]
      edge_count = graph.edge_index.shape[0]
      nodes[batch_index, graph_index, :node_count] = graph.node_features
      node_mask[batch_index, graph_index, :node_count] = True
      markers[batch_index, graph_index, :node_count] = graph.endpoint_markers
      graph_mask[batch_index, graph_index] = True
      if edge_count:
        edges[batch_index, graph_index, :edge_count] = graph.edge_index
        edge_features[batch_index, graph_index, :edge_count] = graph.edge_features
        edge_mask[batch_index, graph_index, :edge_count] = True
    count = example.candidate_program_indices.shape[0]
    candidate_ids[batch_index, :count] = example.candidate_program_indices
    candidate_mask[batch_index, :count] = True
    target_positions[batch_index] = example.candidate_program_indices.tolist().index(
        example.target_program_index
    )
    translations[batch_index] = example.residual_translation
    rotations[batch_index] = example.residual_rotation_vector
    residual_mask[batch_index] = example.residual_mask
    hashes.append(str(example.model_view_sha256))
  return BRepProgramBatch(
      inputs=BRepProgramInputs(
          node_features=nodes,
          node_mask=node_mask,
          edge_index=edges,
          edge_features=edge_features,
          edge_mask=edge_mask,
          endpoint_markers=markers,
          graph_mask=graph_mask,
          candidate_program_indices=candidate_ids,
          candidate_mask=candidate_mask,
      ),
      targets=BRepProgramTargets(
          target_positions=target_positions,
          residual_translation=translations,
          residual_rotation_vector=rotations,
          residual_mask=residual_mask,
      ),
      model_view_sha256s=tuple(hashes),
  )


def collate_authenticated_brep_learner_examples(
    examples: Sequence[
        AuthenticatedBRepLearnerExample | "AuthenticatedCachedBRepLearnerExample"
    ],
    *,
    budget: LearnerBatchBudget = LearnerBatchBudget(),
) -> BRepProgramBatch:
  """The only direct formal collator; every row retains capability lineage."""

  allowed_types = (
      AuthenticatedBRepLearnerExample,
      AuthenticatedCachedBRepLearnerExample,
  )
  if not examples or any(type(row) not in allowed_types for row in examples):
    raise TypeError("formal learner batches require authenticated example lineage")
  for row in examples:
    row.revalidate()
    if (
        type(row) is AuthenticatedCachedBRepLearnerExample
        and not row.formal_training_eligible
    ):
      raise ValueError("development-subset cache cannot unlock formal training")
  return collate_nonformal_brep_learner_examples(
      [row.example for row in examples], budget=budget
  )


class LazyAuthenticatedBRepDataset:
  """Lazy train/dev adapter; it has no final-test or merged-pool API."""

  def __init__(
      self,
      index: AuthenticatedBenchmarkV2DevelopmentIndex,
      *,
      projection_artifact_sha256: str,
      graph_budget: GraphSizeBudget = GraphSizeBudget(),
  ) -> None:
    if type(index) is not AuthenticatedBenchmarkV2DevelopmentIndex:
      raise TypeError("lazy learner dataset requires an authenticated development index")
    if index.catalog_summary["entry_count"] != FORMAL_CATALOG_SIZE:
      raise ValueError("formal learner requires the frozen 19-entry train catalog")
    if _SHA256.fullmatch(projection_artifact_sha256) is None:
      raise ValueError("lazy learner dataset requires a pinned projection artifact")
    self._index = index
    self._projection_artifact_sha256 = projection_artifact_sha256
    self._graph_budget = graph_budget

  @property
  def split_counts(self) -> Mapping[str, int]:
    return self._index.split_counts

  @property
  def semantic_row_counts(self) -> Mapping[str, tuple[int, ...]]:
    return self._index.semantic_row_counts

  @property
  def semantic_row_totals(self) -> Mapping[str, int]:
    return self._index.semantic_row_totals

  @property
  def catalog_summary(self) -> Mapping[str, Any]:
    return self._index.catalog_summary

  @property
  def projection_artifact_sha256(self) -> str:
    return self._projection_artifact_sha256

  @property
  def graph_budget_summary(self) -> Mapping[str, int]:
    """Exact model-view graph budget bound into every cache checkpoint."""

    return MappingProxyType(dict(self._graph_budget.to_dict()))

  @property
  def graph_budget_payload_sha256(self) -> str:
    return _canonical_sha256(dict(self.graph_budget_summary))

  def load_example(
      self, *, split: str, case_index: int, program_index: int = 0
  ) -> AuthenticatedBRepLearnerExample:
    normalized = str(split).strip().lower()
    if normalized not in {"train", "dev"}:
      raise ValueError("learner dataset exposes train/dev only")
    graph = self._index.materialize_graph_capability(
        split=normalized,
        case_index=case_index,
        program_index=program_index,
        budget=self._graph_budget,
    )
    return self._example_from_graph_capability(
        graph,
        split=normalized,
        case_index=case_index,
        program_index=program_index,
        catalog_payload_sha256=self.catalog_summary["catalog_payload_sha256"],
    )

  def _example_from_graph_capability(
      self,
      graph: Any,
      *,
      split: str,
      case_index: int,
      program_index: int,
      catalog_payload_sha256: str,
  ) -> AuthenticatedBRepLearnerExample:
    view = self._index.load_model_view(
        graph, split=split, case_index=case_index
    )
    authenticated = tensorize_authenticated_model_view(view)
    lineage = dict(authenticated.lineage)
    lineage.pop("lineage_payload_sha256")
    lineage.update(
        {
            "projection_artifact_sha256": self._projection_artifact_sha256,
            "catalog_payload_sha256": catalog_payload_sha256,
            "split": split,
            "case_index": case_index,
            "program_index": program_index,
        }
    )
    lineage["lineage_payload_sha256"] = _canonical_sha256(lineage)
    return AuthenticatedBRepLearnerExample(
        capability=authenticated._capability,
        example=authenticated.example,
        lineage=lineage,
        _factory_token=_AUTHENTICATED_EXAMPLE_FACTORY_TOKEN,
    )

  def load_case_examples(
      self,
      *,
      split: str,
      case_index: int,
      program_indices: Sequence[int],
  ) -> tuple[AuthenticatedBRepLearnerExample, ...]:
    """Load an ordered program subset while parsing each case STEP once."""

    normalized = str(split).strip().lower()
    if normalized not in {"train", "dev"}:
      raise ValueError("learner dataset exposes train/dev only")
    normalized_indices = tuple(program_indices)
    graphs = self._index.materialize_case_graph_capabilities(
        split=normalized,
        case_index=case_index,
        program_indices=normalized_indices,
        budget=self._graph_budget,
    )
    if len(graphs) != len(normalized_indices):
      raise ValueError("case graph batch count differs from requested programs")
    catalog_payload_sha256 = self.catalog_summary["catalog_payload_sha256"]
    return tuple(
        self._example_from_graph_capability(
            graph,
            split=normalized,
            case_index=case_index,
            program_index=program_index,
            catalog_payload_sha256=catalog_payload_sha256,
        )
        for graph, program_index in zip(
            graphs, normalized_indices, strict=True
        )
    )

  def iter_examples(
      self, *, split: str, program_index: int = 0
  ) -> Iterable[AuthenticatedBRepLearnerExample]:
    """Lazy one-pass iterator; multi-epoch training should publish a cache."""

    normalized = str(split).strip().lower()
    if normalized not in {"train", "dev"}:
      raise ValueError("learner dataset exposes train/dev only")
    return (
        self.load_example(
            split=normalized, case_index=case_index, program_index=program_index
        )
        for case_index in range(self.split_counts[normalized])
    )

  def iter_all_examples(
      self, *, split: str
  ) -> Iterable[AuthenticatedBRepLearnerExample]:
    """Yield every semantic row in deterministic case/program order."""

    normalized = str(split).strip().lower()
    if normalized not in {"train", "dev"}:
      raise ValueError("learner dataset exposes train/dev only")
    counts = tuple(self.semantic_row_counts[normalized])
    expected_total = sum(counts)

    def iterator() -> Iterable[AuthenticatedBRepLearnerExample]:
      emitted = 0
      for case_index, program_count in enumerate(counts):
        for program_index in range(program_count):
          yield self.load_example(
              split=normalized,
              case_index=case_index,
              program_index=program_index,
          )
          emitted += 1
      if emitted != expected_total:
        raise ValueError("semantic-row iterator count differs from authority")

    return iterator()


_CACHE_FACTORY_TOKEN = object()


class AuthenticatedCachedBRepLearnerExample:
  """One sanitized tensor row authenticated by a captured cache artifact."""

  __slots__ = ("_cache", "_example", "_lineage", "_validation_token")

  def __init__(
      self,
      *,
      cache: "AuthenticatedSanitizedBRepTensorCache",
      example: BRepLearnerExample,
      lineage: Mapping[str, Any],
      _validation_token: object | None = None,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _CACHE_FACTORY_TOKEN:
      raise TypeError("cached authenticated examples are factory-only")
    self._cache = cache
    self._example = example
    self._lineage = MappingProxyType(dict(lineage))
    self._validation_token = _validation_token

  @property
  def example(self) -> BRepLearnerExample:
    self.revalidate()
    return self._example

  @property
  def lineage(self) -> Mapping[str, Any]:
    self.revalidate()
    return self._lineage

  @property
  def formal_training_eligible(self) -> bool:
    return self._cache.formal_training_eligible

  def revalidate(self) -> None:
    fast_validator = getattr(self._cache, "_revalidate_cached_example", None)
    if fast_validator is None:
      self._cache.revalidate()
      return
    fast_validator(self._validation_token)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("cached authenticated examples are not serializable")


def _tensor_receipt(name: str, array: np.ndarray) -> dict[str, Any]:
  contiguous = np.ascontiguousarray(array)
  return {
      "name": name,
      "dtype": str(contiguous.dtype),
      "shape": list(contiguous.shape),
      "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
  }


def _cache_arrays_for_example(
    example: BRepLearnerExample, *, prefix: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
  arrays: dict[str, np.ndarray] = {}
  for graph_index, graph in enumerate(example.graphs):
    for suffix, tensor in (
        ("nodes", graph.node_features),
        ("edge_index", graph.edge_index),
        ("edge_features", graph.edge_features),
        ("endpoint_markers", graph.endpoint_markers),
    ):
      arrays[f"{prefix}_g{graph_index}_{suffix}"] = tensor.detach().cpu().numpy()
  arrays[f"{prefix}_candidate_ids"] = (
      example.candidate_program_indices.detach().cpu().numpy()
  )
  arrays[f"{prefix}_translation"] = example.residual_translation.detach().cpu().numpy()
  arrays[f"{prefix}_rotation"] = example.residual_rotation_vector.detach().cpu().numpy()
  receipts = [_tensor_receipt(name, arrays[name]) for name in sorted(arrays)]
  return arrays, receipts


class AuthenticatedSanitizedBRepTensorCache:
  """Captured all-row cache used for repeat epochs without reopening STEP/OCC."""

  __slots__ = ("_captures", "_rows", "_manifest", "artifact_sha256")

  def __init__(
      self,
      *,
      captures: Sequence[CapturedFileArtifact],
      rows: Mapping[str, Sequence[tuple[BRepLearnerExample, Mapping[str, Any]]]],
      manifest: Mapping[str, Any],
      artifact_sha256: str,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _CACHE_FACTORY_TOKEN:
      raise TypeError("sanitized tensor caches are verifier-only")
    self._captures = tuple(captures)
    self._rows = {key: tuple(value) for key, value in rows.items()}
    self._manifest = MappingProxyType(dict(manifest))
    self.artifact_sha256 = artifact_sha256

  @property
  def formal_training_eligible(self) -> bool:
    return bool(self._manifest["formal_training_eligible"])

  @property
  def training_example_manifest_sha256(self) -> str:
    return str(self._manifest["training_example_manifest_sha256"])

  @property
  def split_counts(self) -> Mapping[str, int]:
    return MappingProxyType({key: len(rows) for key, rows in self._rows.items()})

  def revalidate(self) -> None:
    for captured in self._captures:
      reverify_captured_file_artifact(captured, label="sanitized B-Rep tensor cache")

  def iter_examples(
      self, *, split: str
  ) -> Iterable[AuthenticatedCachedBRepLearnerExample]:
    normalized = str(split).strip().lower()
    if normalized not in self._rows:
      raise ValueError("tensor cache exposes train/dev only")
    self.revalidate()
    for example, lineage in self._rows[normalized]:
      yield AuthenticatedCachedBRepLearnerExample(
          cache=self,
          example=example,
          lineage=lineage,
          _factory_token=_CACHE_FACTORY_TOKEN,
      )

  def iter_formal_batches(
      self,
      *,
      split: str,
      batch_size: int,
      budget: LearnerBatchBudget = LearnerBatchBudget(),
  ) -> Iterable[BRepProgramBatch]:
    del split, batch_size, budget
    raise ValueError(
        "formal cached training is unavailable until the complete 20,696-row "
        "authority API is implemented"
    )
    yield  # pragma: no cover - keeps this method an iterator API


def publish_sanitized_brep_tensor_cache(
    output_directory: str | Path,
    *,
    dataset: LazyAuthenticatedBRepDataset,
    selection: Mapping[str, Sequence[int]] | None = None,
    require_complete: bool = False,
) -> str:
  """Materialize an authenticated development cache without formal claims.

  The current public index exposes case counts but not independently verified
  per-case semantic-row counts.  Therefore even selecting all 500/100 cases at
  one program index is a subset of the 17,182/3,514 semantic rows and must not
  be labelled a complete formal training cache.
  """

  if type(dataset) is not LazyAuthenticatedBRepDataset:
    raise TypeError("tensor cache publisher requires the lazy authenticated dataset")
  selected = {
      split: list(range(dataset.split_counts[split]))
      for split in ("train", "dev")
  } if selection is None else {
      split: [int(value) for value in selection.get(split, ())]
      for split in ("train", "dev")
  }
  if require_complete:
    raise ValueError(
        "formal tensor cache is unavailable until authenticated per-case "
        "semantic-row counts are exposed and all rows are materialized"
    )
  complete = False
  root = Path(output_directory)
  if root.exists():
    raise FileExistsError(f"tensor cache output already exists: {root}")
  root.parent.mkdir(parents=True, exist_ok=True)
  _reject_reparse_path_chain(root.parent, label="tensor cache output")
  temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
  arrays: dict[str, np.ndarray] = {}
  row_receipts: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
  try:
    row_ordinal = 0
    for split in ("train", "dev"):
      for case_index in selected[split]:
        authenticated = dataset.load_example(split=split, case_index=case_index)
        prefix = f"r{row_ordinal:06d}"
        row_arrays, tensor_receipts = _cache_arrays_for_example(
            authenticated.example, prefix=prefix
        )
        arrays.update(row_arrays)
        row = {
            "split": split,
            "case_index": case_index,
            "program_index": 0,
            "graph_count": len(authenticated.example.graphs),
            "target_program_index": authenticated.example.target_program_index,
            "residual_mask": authenticated.example.residual_mask,
            "model_view_sha256": authenticated.example.model_view_sha256,
            "lineage": dict(authenticated.lineage),
            "tensors": tensor_receipts,
        }
        row["row_payload_sha256"] = _canonical_sha256(row)
        row_receipts[split].append(row)
        row_ordinal += 1
    tensor_path = temporary / "tensors.npz"
    np.savez_compressed(tensor_path, **arrays)
    tensor_bytes = tensor_path.read_bytes()
    example_manifest = [
        {
            "split": split,
            "case_index": row["case_index"],
            "row_payload_sha256": row["row_payload_sha256"],
        }
        for split in ("train", "dev")
        for row in row_receipts[split]
    ]
    manifest = {
        "schema_version": "authenticated_sanitized_brep_tensor_cache.v1",
        "formal_training_eligible": complete,
        "scope": "authenticated_development_semantic_row_subset",
        "projection_artifact_sha256": dataset.projection_artifact_sha256,
        "catalog_payload_sha256": dataset.catalog_summary["catalog_payload_sha256"],
        "counts": {split: len(row_receipts[split]) for split in ("train", "dev")},
        "training_example_manifest_sha256": _canonical_sha256(example_manifest),
        "rows": row_receipts,
        "tensors": {
            "name": "tensors.npz",
            "bytes": len(tensor_bytes),
            "sha256": hashlib.sha256(tensor_bytes).hexdigest(),
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    manifest_bytes = _canonical_bytes(manifest)
    (temporary / "cache_manifest.json").write_bytes(manifest_bytes)
    artifact = {
        "schema_version": "authenticated_sanitized_brep_tensor_cache_artifact.v1",
        "files": [
            {
                "name": "cache_manifest.json",
                "bytes": len(manifest_bytes),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            manifest["tensors"],
        ],
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    artifact_bytes = _canonical_bytes(artifact)
    (temporary / "artifact.json").write_bytes(artifact_bytes)
    os.replace(temporary, root)
    return hashlib.sha256(artifact_bytes).hexdigest()
  except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise


def load_sanitized_brep_tensor_cache(
    cache_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_projection_artifact_sha256: str,
    expected_catalog_payload_sha256: str,
) -> AuthenticatedSanitizedBRepTensorCache:
  root = Path(cache_directory)
  _reject_reparse_path_chain(root, label="tensor cache input")
  artifact_capture = capture_file_artifact(root / "artifact.json", label="tensor cache artifact")
  if artifact_capture.sha256 != expected_artifact_sha256:
    raise ValueError("tensor cache artifact differs from external pin")
  artifact = json.loads(artifact_capture.raw_bytes.decode("utf-8"))
  artifact_hash = artifact.pop("artifact_payload_sha256", None)
  if artifact_hash != _canonical_sha256(artifact):
    raise ValueError("tensor cache artifact self-hash differs")
  files = artifact.get("files")
  if (
      artifact.get("schema_version")
      != "authenticated_sanitized_brep_tensor_cache_artifact.v1"
      or not isinstance(files, list)
      or [row.get("name") for row in files] != [
      "cache_manifest.json", "tensors.npz"
      ]
  ):
    raise ValueError("tensor cache artifact policy differs")
  captures = tuple(
      capture_file_artifact(root / row["name"], label="tensor cache member")
      for row in files
  )
  if [
      _captured_binding(captured, name=row["name"])
      for captured, row in zip(captures, files, strict=True)
  ] != files:
    raise ValueError("tensor cache member binding differs")
  manifest_capture, tensor_capture = captures
  manifest = json.loads(manifest_capture.raw_bytes.decode("utf-8"))
  manifest_hash = manifest.pop("manifest_payload_sha256", None)
  if manifest_hash != _canonical_sha256(manifest):
    raise ValueError("tensor cache manifest self-hash differs")
  manifest["manifest_payload_sha256"] = manifest_hash
  if (
      manifest.get("schema_version") != "authenticated_sanitized_brep_tensor_cache.v1"
      or manifest.get("formal_training_eligible") is not False
      or manifest.get("scope")
      != "authenticated_development_semantic_row_subset"
      or manifest.get("projection_artifact_sha256")
      != expected_projection_artifact_sha256
      or manifest.get("catalog_payload_sha256") != expected_catalog_payload_sha256
      or manifest.get("tensors") != files[1]
  ):
    raise ValueError("tensor cache authority binding differs")
  reconstructed: dict[str, list[tuple[BRepLearnerExample, Mapping[str, Any]]]] = {
      "train": [], "dev": []
  }
  example_manifest: list[dict[str, Any]] = []
  with np.load(io.BytesIO(tensor_capture.raw_bytes), allow_pickle=False) as archive:
    for split in ("train", "dev"):
      for row in manifest["rows"][split]:
        row_copy = dict(row)
        row_hash = row_copy.pop("row_payload_sha256", None)
        if row_hash != _canonical_sha256(row_copy):
          raise ValueError("tensor cache row self-hash differs")
        tensor_receipts = row["tensors"]
        for receipt in tensor_receipts:
          if receipt["name"] not in archive.files:
            raise ValueError("tensor cache array is missing")
          if _tensor_receipt(receipt["name"], archive[receipt["name"]]) != receipt:
            raise ValueError("tensor cache array binding differs")
        prefix = tensor_receipts[0]["name"].split("_", 1)[0]
        graph_count = int(row["graph_count"])
        graphs = tuple(
            BRepGraphExample(
                node_features=torch.from_numpy(archive[f"{prefix}_g{index}_nodes"].copy()),
                edge_index=torch.from_numpy(archive[f"{prefix}_g{index}_edge_index"].copy()),
                edge_features=torch.from_numpy(archive[f"{prefix}_g{index}_edge_features"].copy()),
                endpoint_markers=torch.from_numpy(archive[f"{prefix}_g{index}_endpoint_markers"].copy()),
            )
            for index in range(graph_count)
        )
        example = BRepLearnerExample(
            graphs=graphs,
            candidate_program_indices=torch.from_numpy(archive[f"{prefix}_candidate_ids"].copy()),
            target_program_index=int(row["target_program_index"]),
            residual_translation=torch.from_numpy(archive[f"{prefix}_translation"].copy()),
            residual_rotation_vector=torch.from_numpy(archive[f"{prefix}_rotation"].copy()),
            residual_mask=bool(row["residual_mask"]),
            model_view_sha256=str(row["model_view_sha256"]),
        )
        reconstructed[split].append((example, row["lineage"]))
        example_manifest.append(
            {"split": split, "case_index": row["case_index"], "row_payload_sha256": row_hash}
        )
  if manifest["training_example_manifest_sha256"] != _canonical_sha256(example_manifest):
    raise ValueError("tensor cache training-example manifest differs")
  all_captures = (artifact_capture, *captures)
  for captured in all_captures:
    reverify_captured_file_artifact(captured, label="sanitized B-Rep tensor cache")
  return AuthenticatedSanitizedBRepTensorCache(
      captures=all_captures,
      rows=reconstructed,
      manifest=manifest,
      artifact_sha256=expected_artifact_sha256,
      _factory_token=_CACHE_FACTORY_TOKEN,
  )


_SHARDED_CACHE_SCHEMA = "authenticated_sanitized_brep_tensor_cache.v3"
_SHARDED_CACHE_ARTIFACT_SCHEMA = (
    "authenticated_sanitized_brep_tensor_cache_artifact.v3"
)


def _semantic_row_ledger(
    dataset: LazyAuthenticatedBRepDataset,
) -> tuple[dict[str, list[int]], dict[str, int]]:
  raw_counts = dataset.semantic_row_counts
  if not isinstance(raw_counts, Mapping) or set(raw_counts) != {"train", "dev"}:
    raise ValueError("semantic-row count ledger must contain train/dev only")
  for split in ("train", "dev"):
    if not isinstance(raw_counts[split], (tuple, list)) or any(
        type(value) is not int or value < 0 for value in raw_counts[split]
    ):
      raise ValueError("semantic-row per-case counts are malformed")
  counts = {
      split: list(raw_counts[split])
      for split in ("train", "dev")
  }
  if any(
      len(counts[split]) != dataset.split_counts[split]
      or any(type(value) is not int or value < 0 for value in counts[split])
      for split in ("train", "dev")
  ):
    raise ValueError("semantic-row count ledger differs from case slots")
  totals = {split: sum(counts[split]) for split in ("train", "dev")}
  if any(totals[split] < 1 for split in ("train", "dev")):
    raise ValueError("semantic-row train/dev domains must both be non-empty")
  return counts, totals


def _semantic_row_positions(
    counts: Mapping[str, Sequence[int]],
) -> Iterable[tuple[str, int, int]]:
  for split in ("train", "dev"):
    for case_index, program_count in enumerate(counts[split]):
      for program_index in range(int(program_count)):
        yield split, case_index, program_index


def _atomic_immutable_bytes(root: Path, name: str, payload: bytes) -> None:
  target = root / name
  if target.exists():
    if target.read_bytes() != payload:
      raise ValueError("immutable sharded-cache member already differs")
    return
  descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=root)
  try:
    with os.fdopen(descriptor, "wb") as stream:
      stream.write(payload)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary_name, target)
  except BaseException:
    try:
      os.unlink(temporary_name)
    except FileNotFoundError:
      pass
    raise


def _validated_v2_lineage(
    lineage: Any,
    *,
    split: str,
    case_index: int,
    program_index: int,
    model_view_sha256: str,
    projection_artifact_sha256: str,
    catalog_payload_sha256: str,
) -> dict[str, Any]:
  if not isinstance(lineage, Mapping):
    raise ValueError("sharded-cache row lineage must be an object")
  normalized = dict(lineage)
  if set(normalized) != {
      "schema_version",
      "model_view_sha256",
      "model_view_schema_version",
      "catalog_descriptor_manifest_sha256",
      "local_subgraph_policy_sha256",
      "projection_artifact_sha256",
      "catalog_payload_sha256",
      "split",
      "case_index",
      "program_index",
      "lineage_payload_sha256",
  }:
    raise ValueError("sharded-cache row lineage schema differs")
  payload_hash = normalized.pop("lineage_payload_sha256", None)
  if payload_hash != _canonical_sha256(normalized):
    raise ValueError("sharded-cache row lineage self-hash differs")
  normalized["lineage_payload_sha256"] = payload_hash
  if (
      normalized.get("schema_version")
      != "authenticated_brep_learner_example_lineage.v1"
      or normalized.get("model_view_sha256") != model_view_sha256
      or normalized.get("projection_artifact_sha256")
      != projection_artifact_sha256
      or normalized.get("catalog_payload_sha256") != catalog_payload_sha256
      or normalized.get("split") != split
      or normalized.get("case_index") != case_index
      or normalized.get("program_index") != program_index
  ):
    raise ValueError("sharded-cache row lineage binding differs")
  return normalized


def _v2_row_from_example(
    authenticated: AuthenticatedBRepLearnerExample,
    *,
    row_ordinal: int,
    split: str,
    case_index: int,
    program_index: int,
    shard_index: int,
    projection_artifact_sha256: str,
    catalog_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  authenticated.revalidate()
  example = authenticated.example
  prefix = f"r{row_ordinal:08d}"
  arrays, receipts = _cache_arrays_for_example(example, prefix=prefix)
  lineage = _validated_v2_lineage(
      authenticated.lineage,
      split=split,
      case_index=case_index,
      program_index=program_index,
      model_view_sha256=example.model_view_sha256,
      projection_artifact_sha256=projection_artifact_sha256,
      catalog_payload_sha256=catalog_payload_sha256,
  )
  row = {
      "row_ordinal": row_ordinal,
      "split": split,
      "case_index": case_index,
      "program_index": program_index,
      "shard_index": shard_index,
      "graph_count": len(example.graphs),
      "target_program_index": example.target_program_index,
      "residual_mask": example.residual_mask,
      "model_view_sha256": example.model_view_sha256,
      "lineage": lineage,
      "tensors": receipts,
  }
  row["row_payload_sha256"] = _canonical_sha256(row)
  return row, arrays


def _publish_v2_checkpoint(
    root: Path,
    *,
    projection_artifact_sha256: str,
    catalog_payload_sha256: str,
    graph_budget: Mapping[str, int],
    semantic_row_counts: Mapping[str, Sequence[int]],
    semantic_row_totals: Mapping[str, int],
    rows: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    previous_checkpoint_artifact_sha256: str | None,
) -> str:
  total = sum(int(semantic_row_totals[split]) for split in ("train", "dev"))
  complete = len(rows) == total
  materialized_counts = {
      split: sum(1 for row in rows if row.get("split") == split)
      for split in ("train", "dev")
  }
  example_manifest = [
      {
          "row_ordinal": row["row_ordinal"],
          "split": row["split"],
          "case_index": row["case_index"],
          "program_index": row["program_index"],
          "row_payload_sha256": row["row_payload_sha256"],
      }
      for row in rows
  ]
  manifest = {
      "schema_version": _SHARDED_CACHE_SCHEMA,
      "scope": "authenticated_train_dev_semantic_rows",
      "formal_training_eligible": complete,
      "projection_artifact_sha256": projection_artifact_sha256,
      "catalog_payload_sha256": catalog_payload_sha256,
      "graph_budget": dict(graph_budget),
      "graph_budget_payload_sha256": _canonical_sha256(dict(graph_budget)),
      "semantic_row_counts": {
          split: list(semantic_row_counts[split]) for split in ("train", "dev")
      },
      "semantic_row_totals": {
          split: int(semantic_row_totals[split]) for split in ("train", "dev")
      },
      "materialized_row_counts": materialized_counts,
      "row_count": len(rows),
      "training_example_manifest_sha256": _canonical_sha256(example_manifest),
      "previous_checkpoint_artifact_sha256": previous_checkpoint_artifact_sha256,
      "rows": [dict(row) for row in rows],
      "shards": [dict(shard) for shard in shards],
  }
  manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
  manifest_bytes = _canonical_bytes(manifest)
  manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
  manifest_name = f"manifest-{manifest_sha256}.json"
  _atomic_immutable_bytes(root, manifest_name, manifest_bytes)
  artifact = {
      "schema_version": _SHARDED_CACHE_ARTIFACT_SCHEMA,
      "manifest": {
          "name": manifest_name,
          "bytes": len(manifest_bytes),
          "sha256": manifest_sha256,
      },
      "shards": [
          {key: shard[key] for key in ("name", "bytes", "sha256")}
          for shard in shards
      ],
  }
  artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
  artifact_bytes = _canonical_bytes(artifact)
  artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
  artifact_name = f"artifact-{artifact_sha256}.json"
  _atomic_immutable_bytes(root, artifact_name, artifact_bytes)
  pointer = {
      "schema_version": "authenticated_sanitized_brep_tensor_cache_pointer.v3",
      "artifact_sha256": artifact_sha256,
      "artifact_name": artifact_name,
  }
  pointer["pointer_payload_sha256"] = _canonical_sha256(pointer)
  pointer_bytes = _canonical_bytes(pointer)
  descriptor, temporary_name = tempfile.mkstemp(prefix=".latest.", dir=root)
  try:
    with os.fdopen(descriptor, "wb") as stream:
      stream.write(pointer_bytes)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary_name, root / "latest_checkpoint.json")
  except BaseException:
    try:
      os.unlink(temporary_name)
    except FileNotFoundError:
      pass
    raise
  return artifact_sha256


def _example_from_v2_archive(
    archive: Any, row: Mapping[str, Any]
) -> BRepLearnerExample:
  prefix = f"r{int(row['row_ordinal']):08d}"
  graph_count = int(row["graph_count"])
  return BRepLearnerExample(
      graphs=tuple(
          BRepGraphExample(
              node_features=torch.from_numpy(
                  archive[f"{prefix}_g{index}_nodes"].copy()
              ),
              edge_index=torch.from_numpy(
                  archive[f"{prefix}_g{index}_edge_index"].copy()
              ),
              edge_features=torch.from_numpy(
                  archive[f"{prefix}_g{index}_edge_features"].copy()
              ),
              endpoint_markers=torch.from_numpy(
                  archive[f"{prefix}_g{index}_endpoint_markers"].copy()
              ),
          )
          for index in range(graph_count)
      ),
      candidate_program_indices=torch.from_numpy(
          archive[f"{prefix}_candidate_ids"].copy()
      ),
      target_program_index=int(row["target_program_index"]),
      residual_translation=torch.from_numpy(
          archive[f"{prefix}_translation"].copy()
      ),
      residual_rotation_vector=torch.from_numpy(
          archive[f"{prefix}_rotation"].copy()
      ),
      residual_mask=bool(row["residual_mask"]),
      model_view_sha256=str(row["model_view_sha256"]),
  )


def _validate_v2_shard(
    root: Path,
    shard: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
  capture = capture_file_artifact(
      root / str(shard["name"]), label="sharded tensor-cache member"
  )
  if _captured_binding(capture, name=str(shard["name"])) != {
      key: shard[key] for key in ("name", "bytes", "sha256")
  }:
    raise ValueError("sharded tensor-cache member binding differs")
  expected_names = {
      receipt["name"] for row in rows for receipt in row["tensors"]
  }
  with np.load(io.BytesIO(capture.raw_bytes), allow_pickle=False) as archive:
    if set(archive.files) != expected_names:
      raise ValueError("sharded tensor-cache array domain differs")
    for row in rows:
      for receipt in row["tensors"]:
        if _tensor_receipt(receipt["name"], archive[receipt["name"]]) != receipt:
          raise ValueError("sharded tensor-cache array binding differs")
  reverify_captured_file_artifact(capture, label="sharded tensor-cache member")


def _load_v2_ancestor_manifest(root: Path, artifact_sha256: str) -> dict[str, Any]:
  """Verify one immutable checkpoint link for strict-prefix ancestry."""

  if _SHA256.fullmatch(artifact_sha256) is None:
    raise ValueError("sharded tensor-cache ancestor pin is malformed")
  artifact_capture = capture_file_artifact(
      root / f"artifact-{artifact_sha256}.json",
      label="sharded tensor-cache ancestor artifact",
  )
  if artifact_capture.sha256 != artifact_sha256:
    raise ValueError("sharded tensor-cache ancestor artifact differs")
  artifact = json.loads(artifact_capture.raw_bytes.decode("utf-8"))
  artifact_hash = artifact.pop("artifact_payload_sha256", None)
  if artifact_hash != _canonical_sha256(artifact):
    raise ValueError("sharded tensor-cache ancestor artifact self-hash differs")
  manifest_binding = artifact.get("manifest")
  artifact_shards = artifact.get("shards")
  if (
      artifact.get("schema_version") != _SHARDED_CACHE_ARTIFACT_SCHEMA
      or not isinstance(manifest_binding, Mapping)
      or set(manifest_binding) != {"name", "bytes", "sha256"}
      or not isinstance(artifact_shards, list)
  ):
    raise ValueError("sharded tensor-cache ancestor artifact policy differs")
  manifest_capture = capture_file_artifact(
      root / str(manifest_binding.get("name") or ""),
      label="sharded tensor-cache ancestor manifest",
  )
  if _captured_binding(
      manifest_capture, name=str(manifest_binding.get("name") or "")
  ) != dict(manifest_binding):
    raise ValueError("sharded tensor-cache ancestor manifest binding differs")
  manifest = json.loads(manifest_capture.raw_bytes.decode("utf-8"))
  manifest_hash = manifest.pop("manifest_payload_sha256", None)
  if manifest_hash != _canonical_sha256(manifest):
    raise ValueError("sharded tensor-cache ancestor manifest self-hash differs")
  manifest["manifest_payload_sha256"] = manifest_hash
  ancestor_shards = manifest.get("shards")
  if (
      manifest.get("schema_version") != _SHARDED_CACHE_SCHEMA
      or not isinstance(manifest.get("rows"), list)
      or not isinstance(ancestor_shards, list)
      or artifact_shards
      != [
          {key: row[key] for key in ("name", "bytes", "sha256")}
          for row in ancestor_shards
      ]
  ):
    raise ValueError("sharded tensor-cache ancestor manifest policy differs")
  reverify_captured_file_artifact(
      artifact_capture, label="sharded tensor-cache ancestor artifact"
  )
  reverify_captured_file_artifact(
      manifest_capture, label="sharded tensor-cache ancestor manifest"
  )
  return manifest


def _load_v2_checkpoint_manifest(
    root: Path,
    *,
    dataset: LazyAuthenticatedBRepDataset,
    expected_artifact_sha256: str,
) -> tuple[dict[str, Any], CapturedFileArtifact, CapturedFileArtifact]:
  if _SHA256.fullmatch(expected_artifact_sha256) is None:
    raise ValueError("sharded tensor cache requires an external artifact pin")
  _reject_reparse_path_chain(root, label="sharded tensor cache input")
  artifact_capture = capture_file_artifact(
      root / f"artifact-{expected_artifact_sha256}.json",
      label="sharded tensor-cache artifact",
  )
  if artifact_capture.sha256 != expected_artifact_sha256:
    raise ValueError("sharded tensor-cache artifact differs from external pin")
  artifact = json.loads(artifact_capture.raw_bytes.decode("utf-8"))
  artifact_hash = artifact.pop("artifact_payload_sha256", None)
  if artifact_hash != _canonical_sha256(artifact):
    raise ValueError("sharded tensor-cache artifact self-hash differs")
  manifest_binding = artifact.get("manifest")
  artifact_shards = artifact.get("shards")
  if (
      artifact.get("schema_version") != _SHARDED_CACHE_ARTIFACT_SCHEMA
      or not isinstance(manifest_binding, Mapping)
      or not isinstance(artifact_shards, list)
      or set(manifest_binding) != {"name", "bytes", "sha256"}
  ):
    raise ValueError("sharded tensor-cache artifact policy differs")
  manifest_name = manifest_binding.get("name")
  manifest_sha256 = manifest_binding.get("sha256")
  if (
      not isinstance(manifest_name, str)
      or not isinstance(manifest_sha256, str)
      or _SHA256.fullmatch(manifest_sha256) is None
      or manifest_name != f"manifest-{manifest_sha256}.json"
      or any(
          not isinstance(row, Mapping)
          or set(row) != {"name", "bytes", "sha256"}
          or not isinstance(row.get("name"), str)
          or re.fullmatch(r"shard-[0-9]{6}-[0-9a-f]{16}\.npz", row["name"])
          is None
          for row in artifact_shards
      )
      or len({row["name"] for row in artifact_shards}) != len(artifact_shards)
  ):
    raise ValueError("sharded tensor-cache artifact member policy differs")
  manifest_capture = capture_file_artifact(
      root / manifest_name,
      label="sharded tensor-cache manifest",
  )
  if _captured_binding(
      manifest_capture, name=manifest_name
  ) != dict(manifest_binding):
    raise ValueError("sharded tensor-cache manifest binding differs")
  manifest = json.loads(manifest_capture.raw_bytes.decode("utf-8"))
  manifest_hash = manifest.pop("manifest_payload_sha256", None)
  if manifest_hash != _canonical_sha256(manifest):
    raise ValueError("sharded tensor-cache manifest self-hash differs")
  manifest["manifest_payload_sha256"] = manifest_hash
  counts, totals = _semantic_row_ledger(dataset)
  rows = manifest.get("rows")
  shards = manifest.get("shards")
  if (
      manifest.get("schema_version") != _SHARDED_CACHE_SCHEMA
      or manifest.get("scope") != "authenticated_train_dev_semantic_rows"
      or manifest.get("projection_artifact_sha256")
      != dataset.projection_artifact_sha256
      or manifest.get("catalog_payload_sha256")
      != dataset.catalog_summary["catalog_payload_sha256"]
      or manifest.get("graph_budget") != dict(dataset.graph_budget_summary)
      or manifest.get("graph_budget_payload_sha256")
      != dataset.graph_budget_payload_sha256
      or manifest.get("semantic_row_counts") != counts
      or manifest.get("semantic_row_totals") != totals
      or not isinstance(rows, list)
      or not isinstance(shards, list)
      or any(
          not isinstance(row, Mapping)
          or set(row) != {
              "shard_index", "name", "bytes", "sha256", "row_start",
              "row_count", "rows_payload_sha256",
          }
          for row in shards
      )
      or artifact_shards
      != [{key: row[key] for key in ("name", "bytes", "sha256")} for row in shards]
  ):
    raise ValueError("sharded tensor-cache authority binding differs")
  expected_positions = list(_semantic_row_positions(counts))
  if len(rows) > len(expected_positions):
    raise ValueError("sharded tensor-cache has excess semantic rows")
  seen_row_hashes: set[str] = set()
  split_counts = {"train": 0, "dev": 0}
  for ordinal, row in enumerate(rows):
    if not isinstance(row, Mapping):
      raise ValueError("sharded tensor-cache row must be an object")
    if set(row) != {
        "row_ordinal", "split", "case_index", "program_index", "shard_index",
        "graph_count", "target_program_index", "residual_mask",
        "model_view_sha256", "lineage", "tensors", "row_payload_sha256",
    }:
      raise ValueError("sharded tensor-cache row schema differs")
    row_copy = dict(row)
    row_hash = row_copy.pop("row_payload_sha256", None)
    if row_hash != _canonical_sha256(row_copy) or row_hash in seen_row_hashes:
      raise ValueError("sharded tensor-cache row hash or uniqueness differs")
    seen_row_hashes.add(str(row_hash))
    split, case_index, program_index = expected_positions[ordinal]
    if (
        row.get("row_ordinal") != ordinal
        or row.get("split") != split
        or row.get("case_index") != case_index
        or row.get("program_index") != program_index
        or type(row.get("shard_index")) is not int
    ):
      raise ValueError("sharded tensor-cache row order differs")
    _validated_v2_lineage(
        row.get("lineage"),
        split=split,
        case_index=case_index,
        program_index=program_index,
        model_view_sha256=str(row.get("model_view_sha256") or ""),
        projection_artifact_sha256=dataset.projection_artifact_sha256,
        catalog_payload_sha256=dataset.catalog_summary["catalog_payload_sha256"],
    )
    split_counts[split] += 1
  if manifest.get("row_count") != len(rows):
    raise ValueError("sharded tensor-cache row count differs")
  if manifest.get("materialized_row_counts") != split_counts:
    raise ValueError("sharded tensor-cache split counts differ")
  previous_pin = manifest.get("previous_checkpoint_artifact_sha256")
  if (
      (not rows and (previous_pin is not None or shards))
      or (rows and (
          not isinstance(previous_pin, str)
          or _SHA256.fullmatch(previous_pin) is None
      ))
  ):
    raise ValueError("sharded tensor-cache checkpoint ancestry differs")
  if rows:
    ancestor = _load_v2_ancestor_manifest(root, str(previous_pin))
    ancestor_rows = ancestor["rows"]
    ancestor_shards = ancestor["shards"]
    if (
        ancestor.get("projection_artifact_sha256")
        != manifest.get("projection_artifact_sha256")
        or ancestor.get("catalog_payload_sha256")
        != manifest.get("catalog_payload_sha256")
        or ancestor.get("graph_budget") != manifest.get("graph_budget")
        or ancestor.get("graph_budget_payload_sha256")
        != manifest.get("graph_budget_payload_sha256")
        or ancestor.get("semantic_row_counts")
        != manifest.get("semantic_row_counts")
        or ancestor.get("semantic_row_totals")
        != manifest.get("semantic_row_totals")
        or len(ancestor_rows) >= len(rows)
        or rows[:len(ancestor_rows)] != ancestor_rows
        or len(ancestor_shards) >= len(shards)
        or shards[:len(ancestor_shards)] != ancestor_shards
    ):
      raise ValueError(
          "sharded tensor-cache checkpoint is not a strict ancestor extension"
      )
  complete = len(rows) == len(expected_positions)
  if manifest.get("formal_training_eligible") is not complete:
    raise ValueError("sharded tensor-cache completeness flag differs")
  example_manifest = [
      {
          "row_ordinal": row["row_ordinal"],
          "split": row["split"],
          "case_index": row["case_index"],
          "program_index": row["program_index"],
          "row_payload_sha256": row["row_payload_sha256"],
      }
      for row in rows
  ]
  if manifest.get("training_example_manifest_sha256") != _canonical_sha256(
      example_manifest
  ):
    raise ValueError("sharded tensor-cache example manifest differs")
  shard_row_total = 0
  for shard_index, shard in enumerate(shards):
    if (
        not isinstance(shard, Mapping)
        or shard.get("shard_index") != shard_index
        or shard.get("row_start") != shard_row_total
        or type(shard.get("row_count")) is not int
        or shard.get("row_count") < 1
    ):
      raise ValueError("sharded tensor-cache shard ledger differs")
    shard_rows = rows[shard_row_total: shard_row_total + shard["row_count"]]
    if (
        len(shard_rows) != shard["row_count"]
        or any(row.get("shard_index") != shard_index for row in shard_rows)
        or shard.get("rows_payload_sha256")
        != _canonical_sha256([row["row_payload_sha256"] for row in shard_rows])
    ):
      raise ValueError("sharded tensor-cache shard row domain differs")
    _validate_v2_shard(root, shard, shard_rows)
    shard_row_total += shard["row_count"]
  if shard_row_total != len(rows):
    raise ValueError("sharded tensor-cache rows are missing from shards")
  reverify_captured_file_artifact(
      artifact_capture, label="sharded tensor-cache artifact"
  )
  reverify_captured_file_artifact(
      manifest_capture, label="sharded tensor-cache manifest"
  )
  return manifest, artifact_capture, manifest_capture


class AuthenticatedShardedBRepTensorCacheV2:
  """Verifier-owned, shard-lazy cache for the complete semantic-row domain."""

  __slots__ = (
      "_root",
      "_dataset",
      "_manifest",
      "_root_captures",
      "_row_validation_token",
      "artifact_sha256",
  )

  def __init__(
      self,
      *,
      root: Path,
      dataset: LazyAuthenticatedBRepDataset,
      manifest: Mapping[str, Any],
      root_captures: Sequence[CapturedFileArtifact],
      artifact_sha256: str,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _CACHE_FACTORY_TOKEN:
      raise TypeError("sharded tensor caches are verifier-only")
    self._root = root
    self._dataset = dataset
    self._manifest = MappingProxyType(dict(manifest))
    self._root_captures = tuple(root_captures)
    # Rows are reconstructed only from shard bytes captured and checked by
    # this verifier-owned cache.  The unforgeable in-process token lets the
    # collator authenticate those immutable in-memory rows in O(1), instead
    # of rehashing the 20,696-row manifest twice per training example.
    self._row_validation_token = object()
    self.artifact_sha256 = artifact_sha256

  @property
  def formal_training_eligible(self) -> bool:
    return bool(self._manifest["formal_training_eligible"])

  @property
  def training_example_manifest_sha256(self) -> str:
    return str(self._manifest["training_example_manifest_sha256"])

  @property
  def split_counts(self) -> Mapping[str, int]:
    return MappingProxyType(dict(self._manifest["materialized_row_counts"]))

  def revalidate(self) -> None:
    """Revalidate the small checkpoint root; shard/ledger checks are per pass."""

    for captured in self._root_captures:
      reverify_captured_file_artifact(captured, label="sharded B-Rep tensor cache")

  def _revalidate_cached_example(self, validation_token: object | None) -> None:
    if validation_token is not self._row_validation_token:
      raise ValueError("cached B-Rep example lacks an authenticated shard lease")

  def _revalidate_authority_ledger(self) -> None:
    counts, totals = _semantic_row_ledger(self._dataset)
    if (
        self._manifest["semantic_row_counts"] != counts
        or self._manifest["semantic_row_totals"] != totals
    ):
      raise ValueError("sharded tensor-cache authority changed")

  def iter_examples(
      self, *, split: str
  ) -> Iterable[AuthenticatedCachedBRepLearnerExample]:
    normalized = str(split).strip().lower()
    if normalized not in {"train", "dev"}:
      raise ValueError("sharded tensor cache exposes train/dev only")
    self.revalidate()
    self._revalidate_authority_ledger()
    rows = list(self._manifest["rows"])
    for shard in self._manifest["shards"]:
      start = int(shard["row_start"])
      shard_rows = rows[start: start + int(shard["row_count"])]
      selected = [row for row in shard_rows if row["split"] == normalized]
      if not selected:
        continue
      capture = capture_file_artifact(
          self._root / shard["name"], label="sharded tensor-cache iteration"
      )
      if _captured_binding(capture, name=shard["name"]) != {
          key: shard[key] for key in ("name", "bytes", "sha256")
      }:
        raise ValueError("sharded tensor-cache changed before iteration")
      with np.load(io.BytesIO(capture.raw_bytes), allow_pickle=False) as archive:
        for row in selected:
          yield AuthenticatedCachedBRepLearnerExample(
              cache=self,
              example=_example_from_v2_archive(archive, row),
              lineage=row["lineage"],
              _validation_token=self._row_validation_token,
              _factory_token=_CACHE_FACTORY_TOKEN,
          )
      reverify_captured_file_artifact(
          capture, label="sharded tensor-cache iteration"
      )

  def iter_examples_by_row_ordinals(
      self, *, row_ordinals: Sequence[int]
  ) -> Iterable[AuthenticatedCachedBRepLearnerExample]:
    """Load a sparse, ordered source-row set without scanning prior shards."""

    requested = tuple(row_ordinals)
    if (
        not requested
        or any(type(value) is not int or value < 0 for value in requested)
        or tuple(sorted(set(requested))) != requested
    ):
      raise ValueError("sparse sharded-cache row ordinals differ")
    self.revalidate()
    self._revalidate_authority_ledger()
    rows = list(self._manifest["rows"])
    if requested[-1] >= len(rows):
      raise ValueError("sparse sharded-cache row ordinal exceeds the manifest")
    requested_set = set(requested)
    yielded: list[int] = []
    for shard in self._manifest["shards"]:
      start = int(shard["row_start"])
      stop = start + int(shard["row_count"])
      selected_ordinals = [value for value in requested if start <= value < stop]
      if not selected_ordinals:
        continue
      capture = capture_file_artifact(
          self._root / shard["name"], label="sparse sharded tensor-cache iteration"
      )
      if _captured_binding(capture, name=shard["name"]) != {
          key: shard[key] for key in ("name", "bytes", "sha256")
      }:
        raise ValueError("sparse sharded tensor-cache changed before iteration")
      with np.load(io.BytesIO(capture.raw_bytes), allow_pickle=False) as archive:
        for ordinal in selected_ordinals:
          row = rows[ordinal]
          if row["row_ordinal"] != ordinal or int(row["shard_index"]) != int(
              shard["shard_index"]
          ):
            raise ValueError("sparse sharded-cache row identity differs")
          yielded.append(ordinal)
          yield AuthenticatedCachedBRepLearnerExample(
              cache=self,
              example=_example_from_v2_archive(archive, row),
              lineage=row["lineage"],
              _validation_token=self._row_validation_token,
              _factory_token=_CACHE_FACTORY_TOKEN,
          )
      reverify_captured_file_artifact(
          capture, label="sparse sharded tensor-cache iteration"
      )
    if tuple(yielded) != requested or set(yielded) != requested_set:
      raise ValueError("sparse sharded-cache row domain is incomplete")

  def iter_formal_batches(
      self,
      *,
      split: str,
      batch_size: int,
      budget: LearnerBatchBudget = LearnerBatchBudget(),
  ) -> Iterable[BRepProgramBatch]:
    if not self.formal_training_eligible:
      raise ValueError("incomplete sharded cache cannot unlock formal training")
    if type(batch_size) is not int or batch_size < 1:
      raise ValueError("formal cache batch_size must be positive")
    pending: list[AuthenticatedCachedBRepLearnerExample] = []
    for row in self.iter_examples(split=split):
      pending.append(row)
      if len(pending) == batch_size:
        yield collate_authenticated_brep_learner_examples(pending, budget=budget)
        pending = []
    if pending:
      yield collate_authenticated_brep_learner_examples(pending, budget=budget)


def publish_sharded_brep_tensor_cache_v2(
    output_directory: str | Path,
    *,
    dataset: LazyAuthenticatedBRepDataset,
    max_new_rows: int,
    shard_row_limit: int = 128,
    resume_artifact_sha256: str | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> str:
  """Advance an immutable, resumable cache checkpoint by a bounded row count."""

  if type(dataset) is not LazyAuthenticatedBRepDataset:
    raise TypeError("sharded tensor cache requires the authenticated lazy dataset")
  if type(max_new_rows) is not int or max_new_rows < 0:
    raise ValueError("max_new_rows must be a nonnegative actual integer")
  if type(shard_row_limit) is not int or shard_row_limit < 1:
    raise ValueError("shard_row_limit must be a positive actual integer")
  root = Path(output_directory)
  if resume_artifact_sha256 is None:
    if root.exists():
      raise FileExistsError("new sharded tensor-cache output already exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_path_chain(root.parent, label="sharded tensor-cache output")
    root.mkdir()
    counts, totals = _semantic_row_ledger(dataset)
    rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    current_pin = _publish_v2_checkpoint(
        root,
        projection_artifact_sha256=dataset.projection_artifact_sha256,
        catalog_payload_sha256=dataset.catalog_summary["catalog_payload_sha256"],
        graph_budget=dataset.graph_budget_summary,
        semantic_row_counts=counts,
        semantic_row_totals=totals,
        rows=rows,
        shards=shards,
        previous_checkpoint_artifact_sha256=None,
    )
    if progress_callback is not None:
      progress_callback(
          MappingProxyType(
              {
                  "artifact_sha256": current_pin,
                  "materialized_rows": 0,
                  "total_rows": sum(totals.values()),
                  "shard_count": 0,
                  "complete": False,
              }
          )
      )
  else:
    if not root.is_dir():
      raise ValueError("resumed sharded tensor-cache directory is missing")
    manifest, _, _ = _load_v2_checkpoint_manifest(
        root,
        dataset=dataset,
        expected_artifact_sha256=resume_artifact_sha256,
    )
    counts = {
        split: list(manifest["semantic_row_counts"][split])
        for split in ("train", "dev")
    }
    totals = dict(manifest["semantic_row_totals"])
    rows = [dict(row) for row in manifest["rows"]]
    shards = [dict(shard) for shard in manifest["shards"]]
    current_pin = resume_artifact_sha256
    if progress_callback is not None:
      progress_callback(
          MappingProxyType(
              {
                  "artifact_sha256": current_pin,
                  "materialized_rows": len(rows),
                  "total_rows": sum(totals.values()),
                  "shard_count": len(shards),
                  "complete": len(rows) == sum(totals.values()),
              }
          )
      )
  positions = list(_semantic_row_positions(counts))
  remaining = min(max_new_rows, len(positions) - len(rows))
  catalog_payload_sha256 = dataset.catalog_summary["catalog_payload_sha256"]
  while remaining:
    shard_count = min(shard_row_limit, remaining)
    shard_index = len(shards)
    row_start = len(rows)
    shard_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    shard_positions = positions[row_start:row_start + shard_count]
    group_start = 0
    while group_start < len(shard_positions):
      split, case_index, _ = shard_positions[group_start]
      group_end = group_start + 1
      while group_end < len(shard_positions) and shard_positions[group_end][
          :2
      ] == (split, case_index):
        group_end += 1
      program_indices = tuple(
          position[2] for position in shard_positions[group_start:group_end]
      )
      authenticated_rows = dataset.load_case_examples(
          split=split,
          case_index=case_index,
          program_indices=program_indices,
      )
      if len(authenticated_rows) != len(program_indices):
        raise ValueError("case materialization count differs from shard ledger")
      for offset, (program_index, authenticated) in enumerate(
          zip(program_indices, authenticated_rows, strict=True)
      ):
        ordinal = row_start + group_start + offset
        row, row_arrays = _v2_row_from_example(
            authenticated,
            row_ordinal=ordinal,
            split=split,
            case_index=case_index,
            program_index=program_index,
            shard_index=shard_index,
            projection_artifact_sha256=dataset.projection_artifact_sha256,
            catalog_payload_sha256=catalog_payload_sha256,
        )
        shard_rows.append(row)
        arrays.update(row_arrays)
      group_start = group_end
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    shard_bytes = stream.getvalue()
    shard_sha256 = hashlib.sha256(shard_bytes).hexdigest()
    shard_name = f"shard-{shard_index:06d}-{shard_sha256[:16]}.npz"
    _atomic_immutable_bytes(root, shard_name, shard_bytes)
    shard = {
        "shard_index": shard_index,
        "name": shard_name,
        "bytes": len(shard_bytes),
        "sha256": shard_sha256,
        "row_start": row_start,
        "row_count": len(shard_rows),
        "rows_payload_sha256": _canonical_sha256(
            [row["row_payload_sha256"] for row in shard_rows]
        ),
    }
    rows.extend(shard_rows)
    shards.append(shard)
    previous_pin = current_pin
    current_pin = _publish_v2_checkpoint(
        root,
        projection_artifact_sha256=dataset.projection_artifact_sha256,
        catalog_payload_sha256=dataset.catalog_summary["catalog_payload_sha256"],
        graph_budget=dataset.graph_budget_summary,
        semantic_row_counts=counts,
        semantic_row_totals=totals,
        rows=rows,
        shards=shards,
        previous_checkpoint_artifact_sha256=previous_pin,
    )
    if progress_callback is not None:
      progress_callback(
          MappingProxyType(
              {
                  "artifact_sha256": current_pin,
                  "materialized_rows": len(rows),
                  "total_rows": len(positions),
                  "shard_count": len(shards),
                  "complete": len(rows) == len(positions),
              }
          )
      )
    remaining -= shard_count
  return current_pin


def load_sharded_brep_tensor_cache_v2(
    cache_directory: str | Path,
    *,
    dataset: LazyAuthenticatedBRepDataset,
    expected_artifact_sha256: str,
    require_formal: bool = False,
) -> AuthenticatedShardedBRepTensorCacheV2:
  """Load one externally pinned v2 checkpoint and verify every shard/row."""

  if type(dataset) is not LazyAuthenticatedBRepDataset:
    raise TypeError("sharded tensor cache requires the authenticated lazy dataset")
  manifest, artifact_capture, manifest_capture = _load_v2_checkpoint_manifest(
      Path(cache_directory),
      dataset=dataset,
      expected_artifact_sha256=expected_artifact_sha256,
  )
  if require_formal and manifest["formal_training_eligible"] is not True:
    raise ValueError("incomplete sharded cache is not formal-training eligible")
  return AuthenticatedShardedBRepTensorCacheV2(
      root=Path(cache_directory),
      dataset=dataset,
      manifest=manifest,
      root_captures=(artifact_capture, manifest_capture),
      artifact_sha256=expected_artifact_sha256,
      _factory_token=_CACHE_FACTORY_TOKEN,
  )


class _MessagePassingLayer(nn.Module):
  def __init__(self, hidden_dim: int, dropout: float) -> None:
    super().__init__()
    self.message = nn.Sequential(
        nn.Linear(2 * hidden_dim + EDGE_INPUT_DIM, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.update = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(
      self,
      hidden: Tensor,
      edge_index: Tensor,
      edge_features: Tensor,
  ) -> Tensor:
    if edge_index.numel() == 0:
      aggregate = torch.zeros_like(hidden)
    else:
      source, target = edge_index.unbind(dim=-1)
      forward = self.message(
          torch.cat((hidden[source], hidden[target], edge_features), dim=-1)
      )
      reverse = self.message(
          torch.cat((hidden[target], hidden[source], edge_features), dim=-1)
      )
      aggregate = torch.zeros_like(hidden)
      degree = torch.zeros((hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device)
      aggregate.index_add_(0, target, forward)
      aggregate.index_add_(0, source, reverse)
      ones = torch.ones((source.shape[0], 1), dtype=hidden.dtype, device=hidden.device)
      degree.index_add_(0, target, ones)
      degree.index_add_(0, source, ones)
      aggregate = aggregate / degree.clamp_min(1.0)
    return self.norm(hidden + self.update(torch.cat((hidden, aggregate), dim=-1)))


class BRepProgramLearnerV1(nn.Module):
  """Permutation-invariant endpoint-centred B-Rep catalog scorer."""

  def __init__(self, config: BRepProgramLearnerConfig, *, seed: int) -> None:
    super().__init__()
    if not isinstance(config, BRepProgramLearnerConfig):
      raise TypeError("learner config must be BRepProgramLearnerConfig")
    self.config = config
    self.seed = int(seed)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(self.seed)
      self.node_encoder = nn.Sequential(
          nn.Linear(NODE_INPUT_DIM, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.layers = nn.ModuleList(
          _MessagePassingLayer(config.hidden_dim, config.dropout)
          for _ in range(config.layers)
      )
      self.graph_pool = nn.Sequential(
          nn.Linear(3 * config.hidden_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.query_encoder = nn.Sequential(
          nn.Linear(4 * config.hidden_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.catalog_embedding = nn.Embedding(config.catalog_size, config.hidden_dim)
      cross_dim = 4 * config.hidden_dim
      self.cross_scorer = nn.Sequential(
          nn.Linear(cross_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Dropout(config.dropout),
          nn.Linear(config.hidden_dim, 1),
      )
      self.residual_head = nn.Sequential(
          nn.Linear(cross_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Dropout(config.dropout),
          nn.Linear(config.hidden_dim, 6),
      )

  @property
  def parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters())

  def _encode_graphs(self, inputs: BRepProgramInputs) -> Tensor:
    batch_size, graph_count, _, _ = inputs.node_features.shape
    pooled = torch.zeros(
        (batch_size, graph_count, self.config.hidden_dim),
        dtype=inputs.node_features.dtype,
        device=inputs.node_features.device,
    )
    for batch_index in range(batch_size):
      for graph_index in range(graph_count):
        if not bool(inputs.graph_mask[batch_index, graph_index]):
          continue
        mask = inputs.node_mask[batch_index, graph_index]
        hidden = self.node_encoder(inputs.node_features[batch_index, graph_index, mask])
        edge_mask = inputs.edge_mask[batch_index, graph_index]
        edge_index = inputs.edge_index[batch_index, graph_index, edge_mask]
        edge_features = inputs.edge_features[batch_index, graph_index, edge_mask]
        for layer in self.layers:
          hidden = layer(hidden, edge_index, edge_features)
        markers = inputs.endpoint_markers[batch_index, graph_index, mask]
        whole = hidden.mean(dim=0)
        marked: list[Tensor] = []
        for endpoint in range(2):
          weights = markers[:, endpoint: endpoint + 1]
          marked.append((hidden * weights).sum(dim=0) / weights.sum().clamp_min(1.0))
        pooled[batch_index, graph_index] = self.graph_pool(
            torch.cat((whole, marked[0], marked[1]), dim=-1)
        )
    return pooled

  def forward(self, inputs: BRepProgramInputs) -> BRepProgramOutput:
    if not isinstance(inputs, BRepProgramInputs):
      raise TypeError("learner forward accepts BRepProgramInputs only")
    if inputs.candidate_program_indices.numel() and (
        int(inputs.candidate_program_indices.min()) < 0
        or int(inputs.candidate_program_indices.max()) >= self.config.catalog_size
    ):
      raise ValueError("candidate program index is outside the frozen catalog")
    graph_embeddings = self._encode_graphs(inputs)
    if graph_embeddings.shape[1] == 1:
      graph_embeddings = torch.cat((graph_embeddings, torch.zeros_like(graph_embeddings)), dim=1)
    left, right = graph_embeddings[:, 0], graph_embeddings[:, 1]
    query = self.query_encoder(
        torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
    )
    candidates = self.catalog_embedding(inputs.candidate_program_indices)
    expanded_query = query.unsqueeze(1).expand_as(candidates)
    cross = torch.cat(
        (
            expanded_query,
            candidates,
            expanded_query * candidates,
            torch.abs(expanded_query - candidates),
        ),
        dim=-1,
    )
    logits = self.cross_scorer(cross).squeeze(-1)
    logits = logits.masked_fill(~inputs.candidate_mask, -torch.inf)
    residual = self.residual_head(cross)
    return BRepProgramOutput(
        logits=logits,
        residual_translation=residual[..., :3],
        residual_rotation_vector=residual[..., 3:],
    )


def brep_program_loss(
    output: BRepProgramOutput,
    targets: BRepProgramTargets,
    *,
    config: BRepProgramLearnerConfig,
) -> BRepProgramLoss:
  """Cross entropy plus residual loss gathered only at the target program."""

  classification = F.cross_entropy(output.logits, targets.target_positions)
  batch = torch.arange(output.logits.shape[0], device=output.logits.device)
  predicted_translation = output.residual_translation[batch, targets.target_positions]
  predicted_rotation = output.residual_rotation_vector[batch, targets.target_positions]
  mask = targets.residual_mask.to(output.logits.dtype)
  denominator = mask.sum().clamp_min(1.0)
  translation_rows = F.smooth_l1_loss(
      predicted_translation, targets.residual_translation, reduction="none"
  ).mean(dim=-1)
  rotation_rows = F.smooth_l1_loss(
      predicted_rotation, targets.residual_rotation_vector, reduction="none"
  ).mean(dim=-1)
  translation = (translation_rows * mask).sum() / denominator
  rotation = (rotation_rows * mask).sum() / denominator
  total = (
      classification
      + config.translation_loss_weight * translation
      + config.rotation_loss_weight * rotation
  )
  return BRepProgramLoss(
      total=total,
      classification=classification,
      residual_translation=translation,
      residual_rotation=rotation,
  )


@dataclass(frozen=True, slots=True)
class BRepLearnerCheckpointMetadata:
  model_view_projection_sha256: str
  catalog_payload_sha256: str
  local_subgraph_policy_sha256: str
  code_revision: str
  learner_source_sha256: str
  catalog_descriptor_manifest_sha256: str
  training_example_manifest_sha256: str
  seed: int
  model_config: Mapping[str, Any]
  torch_version: str
  catalog_entry_count: int = FORMAL_CATALOG_SIZE
  schema_version: str = CHECKPOINT_METADATA_SCHEMA_VERSION

  def __post_init__(self) -> None:
    for label, value in (
        ("model_view_projection_sha256", self.model_view_projection_sha256),
        ("catalog_payload_sha256", self.catalog_payload_sha256),
        ("local_subgraph_policy_sha256", self.local_subgraph_policy_sha256),
        ("learner_source_sha256", self.learner_source_sha256),
        (
            "catalog_descriptor_manifest_sha256",
            self.catalog_descriptor_manifest_sha256,
        ),
        ("training_example_manifest_sha256", self.training_example_manifest_sha256),
    ):
      if _SHA256.fullmatch(str(value)) is None:
        raise ValueError(f"{label} must be SHA-256")
    if _GIT_REVISION.fullmatch(self.code_revision) is None:
      raise ValueError("checkpoint code_revision must be a Git object identifier")
    if type(self.seed) is not int or self.seed < 0:
      raise ValueError("checkpoint seed must be a non-negative actual integer")
    if self.catalog_entry_count != FORMAL_CATALOG_SIZE:
      raise ValueError("formal checkpoint must bind the fixed 19-entry catalog")
    config = BRepProgramLearnerConfig(**dict(self.model_config))
    if config.catalog_size != self.catalog_entry_count:
      raise ValueError("checkpoint config and catalog entry count differ")

  def payload(self) -> dict[str, Any]:
    payload = {
        "schema_version": self.schema_version,
        "learner_schema_version": LEARNER_SCHEMA_VERSION,
        "model_view_schema_version": MODEL_VIEW_SCHEMA_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "model_view_projection_sha256": self.model_view_projection_sha256,
        "catalog_payload_sha256": self.catalog_payload_sha256,
        "catalog_entry_count": self.catalog_entry_count,
        "local_subgraph_policy_sha256": self.local_subgraph_policy_sha256,
        "code_revision": self.code_revision,
        "learner_source_sha256": self.learner_source_sha256,
        "catalog_descriptor_manifest_sha256": (
            self.catalog_descriptor_manifest_sha256
        ),
        "training_example_manifest_sha256": self.training_example_manifest_sha256,
        "checkpoint_kind": "inference_only",
        "optimizer_state_included": False,
        "seed": self.seed,
        "model_config": dict(self.model_config),
        "torch_version": self.torch_version,
    }
    payload["metadata_payload_sha256"] = _canonical_sha256(payload)
    return payload


def build_checkpoint_metadata(
    *,
    config: BRepProgramLearnerConfig,
    seed: int,
    model_view_projection_sha256: str,
    catalog_payload_sha256: str,
    local_subgraph_policy: Mapping[str, Any],
    code_revision: str,
    catalog_descriptor_manifest_sha256: str,
    training_example_manifest_sha256: str,
) -> BRepLearnerCheckpointMetadata:
  return BRepLearnerCheckpointMetadata(
      model_view_projection_sha256=model_view_projection_sha256,
      catalog_payload_sha256=catalog_payload_sha256,
      local_subgraph_policy_sha256=_canonical_sha256(dict(local_subgraph_policy)),
      code_revision=code_revision,
      learner_source_sha256=_file_sha256(Path(__file__).resolve()),
      catalog_descriptor_manifest_sha256=catalog_descriptor_manifest_sha256,
      training_example_manifest_sha256=training_example_manifest_sha256,
      seed=seed,
      model_config=MappingProxyType(asdict(config)),
      torch_version=torch.__version__,
  )


@dataclass(frozen=True, slots=True)
class LoadedBRepLearnerInferenceCheckpoint:
  model: BRepProgramLearnerV1
  metadata: Mapping[str, Any]
  artifact_sha256: str
  captures: tuple[CapturedFileArtifact, ...]
  model_state_sha256: str
  _factory_token: object = field(repr=False)

  def __post_init__(self) -> None:
    if self._factory_token is not _LOADED_CHECKPOINT_FACTORY_TOKEN:
      raise TypeError("loaded inference checkpoints are verifier-only")
    if len(self.captures) != 4 or self.captures[0].sha256 != self.artifact_sha256:
      raise ValueError("loaded inference checkpoint capture set differs")
    if _SHA256.fullmatch(self.model_state_sha256) is None:
      raise ValueError("loaded inference checkpoint state digest differs")

  def revalidate(self) -> None:
    for captured in self.captures:
      reverify_captured_file_artifact(captured, label="B-Rep inference checkpoint")
    if _model_state_sha256(self.model) != self.model_state_sha256:
      raise ValueError("loaded inference checkpoint model state was mutated")


def _model_state_sha256(model: BRepProgramLearnerV1) -> str:
  rows = []
  for name, tensor in sorted(model.state_dict().items()):
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    rows.append(
        {
            "name": name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )
  return _canonical_sha256(rows)


def save_brep_learner_inference_checkpoint(
    output_directory: str | Path,
    *,
    model: BRepProgramLearnerV1,
    metadata: BRepLearnerCheckpointMetadata,
) -> str:
  """Publish a pickle-free inference artifact; no optimizer state is stored.

  The returned pin is SHA-256 of the exact ``artifact.json`` bytes.  That
  artifact binds the exact manifest, metadata and weights bytes by name, byte
  count and SHA-256.  It is therefore an external immutable trust root without
  a self-referential file hash.
  """

  root = Path(output_directory)
  if model.config != BRepProgramLearnerConfig(**dict(metadata.model_config)):
    raise ValueError("checkpoint model config differs from metadata")
  if model.seed != metadata.seed:
    raise ValueError("checkpoint model seed differs from metadata")
  if root.exists():
    raise FileExistsError(f"checkpoint output already exists: {root}")
  root.parent.mkdir(parents=True, exist_ok=True)
  _reject_reparse_path_chain(root.parent, label="checkpoint output")
  temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
  try:
    state = model.state_dict()
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in sorted(state.items())
    }
    weights_path = temporary / "weights.npz"
    np.savez_compressed(weights_path, **arrays)
    weights_bytes = weights_path.read_bytes()
    weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()
    metadata_payload = metadata.payload()
    metadata_payload["weights"] = {
        "file": "weights.npz",
        "sha256": weights_sha256,
        "tensor_count": len(arrays),
        "tensor_schema_sha256": _canonical_sha256(
            [
                {
                    "name": name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
                for name, array in arrays.items()
            ]
        ),
    }
    metadata_bytes = _canonical_bytes(metadata_payload)
    (temporary / "metadata.json").write_bytes(metadata_bytes)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "inference_only",
        "optimizer_state_included": False,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "weights_sha256": weights_sha256,
    }
    manifest["checkpoint_payload_sha256"] = _canonical_sha256(manifest)
    manifest_bytes = _canonical_bytes(manifest)
    (temporary / "checkpoint.json").write_bytes(manifest_bytes)
    artifact = {
        "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
        "definition": (
            "sha256_of_exact_artifact_json_binding_exact_manifest_metadata_weights.v1"
        ),
        "files": [
            {
                "name": "checkpoint.json",
                "bytes": len(manifest_bytes),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            {
                "name": "metadata.json",
                "bytes": len(metadata_bytes),
                "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            },
            {
                "name": "weights.npz",
                "bytes": len(weights_bytes),
                "sha256": weights_sha256,
            },
        ],
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    artifact_bytes = _canonical_bytes(artifact)
    (temporary / "artifact.json").write_bytes(artifact_bytes)
    os.replace(temporary, root)
    return hashlib.sha256(artifact_bytes).hexdigest()
  except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise


def load_brep_learner_inference_checkpoint(
    checkpoint_directory: str | Path,
    *,
    expected_projection_sha256: str,
    expected_catalog_sha256: str,
    expected_policy_sha256: str,
    expected_code_revision: str,
    expected_learner_source_sha256: str,
    expected_training_example_manifest_sha256: str,
    expected_artifact_sha256: str,
) -> LoadedBRepLearnerInferenceCheckpoint:
  root = Path(checkpoint_directory)
  if _SHA256.fullmatch(expected_artifact_sha256) is None:
    raise ValueError("checkpoint requires an externally pinned artifact SHA-256")
  _reject_reparse_path_chain(root, label="checkpoint input")
  artifact_capture = capture_file_artifact(
      root / "artifact.json", label="B-Rep checkpoint artifact pin"
  )
  if artifact_capture.sha256 != expected_artifact_sha256:
    raise ValueError("checkpoint artifact differs from external pin")
  try:
    artifact = json.loads(artifact_capture.raw_bytes.decode("utf-8"))
  except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("checkpoint artifact is malformed") from error
  artifact_self_hash = artifact.pop("artifact_payload_sha256", None)
  if artifact_self_hash != _canonical_sha256(artifact):
    raise ValueError("checkpoint artifact self-hash differs")
  artifact["artifact_payload_sha256"] = artifact_self_hash
  expected_files = artifact.get("files")
  if (
      artifact.get("schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION
      or artifact.get("definition")
      != "sha256_of_exact_artifact_json_binding_exact_manifest_metadata_weights.v1"
      or not isinstance(expected_files, list)
      or [row.get("name") for row in expected_files if isinstance(row, Mapping)]
      != ["checkpoint.json", "metadata.json", "weights.npz"]
  ):
    raise ValueError("checkpoint artifact policy differs")
  captures = tuple(
      capture_file_artifact(root / row["name"], label="B-Rep checkpoint member")
      for row in expected_files
  )
  if [
      _captured_binding(captured, name=row["name"])
      for captured, row in zip(captures, expected_files, strict=True)
  ] != expected_files:
    raise ValueError("checkpoint artifact member binding differs")
  manifest_capture, metadata_capture, weights_capture = captures
  manifest_bytes = manifest_capture.raw_bytes
  metadata_bytes = metadata_capture.raw_bytes
  manifest = json.loads(manifest_bytes.decode("utf-8"))
  metadata = json.loads(metadata_bytes.decode("utf-8"))
  expected_manifest_hash = manifest.pop("checkpoint_payload_sha256", None)
  if expected_manifest_hash != _canonical_sha256(manifest):
    raise ValueError("checkpoint manifest self-hash differs")
  manifest["checkpoint_payload_sha256"] = expected_manifest_hash
  if (
      manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
      or manifest.get("checkpoint_kind") != "inference_only"
      or manifest.get("optimizer_state_included") is not False
      or manifest.get("metadata_sha256") != hashlib.sha256(metadata_bytes).hexdigest()
      or manifest.get("weights_sha256") != weights_capture.sha256
  ):
    raise ValueError("checkpoint file binding differs")
  metadata_self_hash = metadata.pop("metadata_payload_sha256", None)
  weights_receipt = metadata.pop("weights", None)
  if metadata_self_hash != _canonical_sha256(metadata):
    raise ValueError("checkpoint metadata self-hash differs")
  metadata["metadata_payload_sha256"] = metadata_self_hash
  metadata["weights"] = weights_receipt
  if (
      metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION
      or metadata.get("model_view_projection_sha256") != expected_projection_sha256
      or metadata.get("catalog_payload_sha256") != expected_catalog_sha256
      or metadata.get("local_subgraph_policy_sha256") != expected_policy_sha256
      or metadata.get("code_revision") != expected_code_revision
      or metadata.get("learner_source_sha256")
      != expected_learner_source_sha256
      or metadata.get("training_example_manifest_sha256")
      != expected_training_example_manifest_sha256
      or metadata.get("checkpoint_kind") != "inference_only"
      or metadata.get("optimizer_state_included") is not False
      or not isinstance(weights_receipt, Mapping)
      or weights_receipt.get("sha256") != manifest["weights_sha256"]
  ):
    raise ValueError("checkpoint authority binding differs")
  config = BRepProgramLearnerConfig(**metadata["model_config"])
  model = BRepProgramLearnerV1(config, seed=int(metadata["seed"]))
  expected_state = model.state_dict()
  with np.load(io.BytesIO(weights_capture.raw_bytes), allow_pickle=False) as archive:
    if set(archive.files) != set(expected_state):
      raise ValueError("checkpoint tensor names differ")
    loaded: dict[str, Tensor] = {}
    tensor_schema: list[dict[str, Any]] = []
    for name in sorted(archive.files):
      array = archive[name]
      tensor_schema.append(
          {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}
      )
      if tuple(array.shape) != tuple(expected_state[name].shape):
        raise ValueError("checkpoint tensor shape differs")
      loaded[name] = torch.from_numpy(array.copy()).to(expected_state[name].dtype)
  if weights_receipt.get("tensor_schema_sha256") != _canonical_sha256(tensor_schema):
    raise ValueError("checkpoint tensor schema differs")
  model.load_state_dict(loaded, strict=True)
  all_captures = (artifact_capture, *captures)
  for captured in all_captures:
    reverify_captured_file_artifact(captured, label="B-Rep inference checkpoint")
  return LoadedBRepLearnerInferenceCheckpoint(
      model=model,
      metadata=MappingProxyType(metadata),
      artifact_sha256=expected_artifact_sha256,
      captures=all_captures,
      model_state_sha256=_model_state_sha256(model),
      _factory_token=_LOADED_CHECKPOINT_FACTORY_TOKEN,
  )


def _program_descriptor(program: MateProgram) -> tuple[str, str, str, str]:
  bound = program.metadata.get("brep_catalog_descriptor")
  if isinstance(bound, Mapping):
    return tuple(
        str(bound.get(key) or "").strip().lower()
        for key in (
            "relation_hint",
            "contact_type",
            "surface_type_a",
            "surface_type_b",
        )
    )
  return (
      str(program.relation_hint).strip().lower(),
      str(program.contact_type).strip().lower(),
      str(program.parent_surface).strip().lower(),
      str(program.child_surface).strip().lower(),
  )


def _catalog_descriptor_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
  return [
      {
          "program_index": int(row["program_index"]),
          "program_id": str(row["program_id"]),
          "program_type": str(row["program_type"]),
          "descriptor": dict(row["descriptor"]),
      }
      for row in payload["examples"][0]["program_candidates"]
  ]


def catalog_descriptor_manifest_sha256(
    model_view: AuthenticatedBenchmarkV2ModelViewV2,
) -> str:
  if type(model_view) is not AuthenticatedBenchmarkV2ModelViewV2:
    raise TypeError("catalog descriptor binding requires an authenticated model view")
  return _canonical_sha256(_catalog_descriptor_manifest(model_view.model_view))


def _endpoint_fingerprint(
    payload: Mapping[str, Any], *, endpoint_key: str
) -> str:
  endpoint = payload["examples"][0]["query"][endpoint_key]
  graph_index = int(endpoint["graph_index"])
  face_indices = [int(value) for value in endpoint["face_indices"]]
  graph = payload["graphs"][graph_index]
  selected = set(face_indices)
  projection = {
      "schema_version": "brep_runtime_endpoint_fingerprint.v1",
      # Deliberately exclude the target program/residual from the runtime
      # query identity.  A label-bearing model-view hash must never become a
      # feature or query identifier, even indirectly.
      "label_free_model_input_sha256": _canonical_sha256(
          {
              "schema_version": payload["schema_version"],
              "normalization": payload["normalization"],
              "graphs": payload["graphs"],
              "query": payload["examples"][0]["query"],
              "program_candidates": payload["examples"][0][
                  "program_candidates"
              ],
          }
      ),
      "graph_index": graph_index,
      "face_indices": face_indices,
      "face_rows": [graph["nodes"][index] for index in face_indices],
      "incident_edges": [
          edge
          for edge in graph["edges"]
          if edge["source"] in selected or edge["target"] in selected
      ],
  }
  return _canonical_sha256(projection)


@dataclass(frozen=True, slots=True)
class _BoundRuntimeQueryState:
  proposer_nonce: object
  query_ref: Any
  query_payload_sha256: str
  capability: AuthenticatedBenchmarkV2ModelViewV2


@dataclass(frozen=True, slots=True)
class _BoundCatalogCandidateState:
  proposer_nonce: object
  candidate_ref: Any
  bound_query_ref: Any
  program_payload_sha256: str
  catalog_position: int


_BOUND_QUERY_REGISTRY: dict[int, _BoundRuntimeQueryState] = {}
_BOUND_CANDIDATE_REGISTRY: dict[int, _BoundCatalogCandidateState] = {}
_BOUND_REGISTRY_LOCK = threading.RLock()


def _drop_bound_query(object_id: int, dead_ref: Any) -> None:
  with _BOUND_REGISTRY_LOCK:
    state = _BOUND_QUERY_REGISTRY.get(object_id)
    if state is not None and state.query_ref is dead_ref:
      _BOUND_QUERY_REGISTRY.pop(object_id, None)


def _drop_bound_candidate(object_id: int, dead_ref: Any) -> None:
  with _BOUND_REGISTRY_LOCK:
    state = _BOUND_CANDIDATE_REGISTRY.get(object_id)
    if state is not None and state.candidate_ref is dead_ref:
      _BOUND_CANDIDATE_REGISTRY.pop(object_id, None)


class BRepLearnerProgramProposer:
  """``ProgramProposer`` adapter for one authenticated query capability."""

  mode = "brep_program_learner_v1"

  def __init__(
      self,
      checkpoint: LoadedBRepLearnerInferenceCheckpoint,
      model_view: AuthenticatedBenchmarkV2ModelViewV2,
      *,
      device: torch.device | str = "cpu",
  ) -> None:
    if type(checkpoint) is not LoadedBRepLearnerInferenceCheckpoint:
      raise TypeError("B-Rep proposer requires a loaded pinned inference checkpoint")
    if type(model_view) is not AuthenticatedBenchmarkV2ModelViewV2:
      raise TypeError("B-Rep proposer requires an authenticated model-view capability")
    checkpoint.revalidate()
    metadata = checkpoint.metadata
    payload = model_view.model_view
    policy = payload["normalization"]["local_interface_subgraph_policy"]
    if _canonical_sha256(policy) != metadata["local_subgraph_policy_sha256"]:
      raise ValueError("runtime model view differs from checkpoint graph policy")
    descriptor_manifest_sha256 = _canonical_sha256(
        _catalog_descriptor_manifest(payload)
    )
    if (
        descriptor_manifest_sha256
        != metadata["catalog_descriptor_manifest_sha256"]
    ):
      raise ValueError("runtime model view differs from checkpoint catalog descriptors")
    if _file_sha256(Path(__file__).resolve()) != metadata["learner_source_sha256"]:
      raise ValueError("runtime learner source differs from checkpoint provenance")
    self._checkpoint = checkpoint
    self._model = checkpoint.model.to(device)
    self._model.eval()
    checkpoint.revalidate()
    self.model_sha256 = checkpoint.artifact_sha256
    self.formal_validated = True
    self.provenance = MappingProxyType(
        {
            key: metadata[key]
            for key in (
                "model_view_projection_sha256",
                "catalog_payload_sha256",
                "local_subgraph_policy_sha256",
                "code_revision",
                "learner_source_sha256",
                "training_example_manifest_sha256",
            )
        }
    )
    self._device = torch.device(device)
    self._model_view = model_view
    self._proposer_nonce = object()
    self._example = tensorize_authenticated_model_view(model_view)
    example_payload = payload["examples"][0]
    self._descriptor_to_position = {
        (
            row["descriptor"]["relation_hint"],
            row["descriptor"]["contact_type"],
            row["descriptor"]["surface_type_a"],
            row["descriptor"]["surface_type_b"],
        ): position
        for position, row in enumerate(example_payload["program_candidates"])
    }
    self._catalog_rows = tuple(
        dict(row) for row in example_payload["program_candidates"]
    )

  def bind_single_evaluation_query(self) -> MateProgram:
    """Bind a whole query payload to this authenticated single-query adapter.

    This is intentionally not a general multi-pair assembly provider.  A
    future grounder integration must supply an authenticated model-view per
    pair; copying strings or this wrapper cannot manufacture that capability.
    """

    self._checkpoint.revalidate()
    self._model_view.revalidate()
    # Construct the query exclusively from the authenticated, label-free
    # model input.  Accepting an arbitrary caller MateProgram here would only
    # authenticate the caller's assertion, not its relationship to this
    # model-view capability.
    payload = self._model_view.model_view
    endpoint_a = _endpoint_fingerprint(payload, endpoint_key="endpoint_a")
    endpoint_b = _endpoint_fingerprint(payload, endpoint_key="endpoint_b")
    label_free_binding = _canonical_sha256(
        {
            "schema_version": "brep_single_query_binding.v1",
            "endpoint_a": endpoint_a,
            "endpoint_b": endpoint_b,
            "catalog_payload_sha256": self.provenance[
                "catalog_payload_sha256"
            ],
            "local_subgraph_policy_sha256": self.provenance[
                "local_subgraph_policy_sha256"
            ],
        }
    )
    bound = MateProgram(
        program_id=f"brep_query_{label_free_binding[:16]}",
        case_id="",
        assembly_dir="",
        part_a="opaque_query_part_a",
        part_b="opaque_query_part_b",
        body_uuid_a="",
        body_uuid_b="",
        interface_a={
            "model_input_protocol": "benchmark_v2_model_view_v2",
            "endpoint_fingerprint": endpoint_a,
        },
        interface_b={
            "model_input_protocol": "benchmark_v2_model_view_v2",
            "endpoint_fingerprint": endpoint_b,
        },
        relation_hint="",
        contact_type="",
        residual_rotation=np.eye(3, dtype=float).tolist(),
        residual_translation=[0.0, 0.0, 0.0],
        source_contact={},
        features={},
        metadata={
            "model_input_protocol": "benchmark_v2_model_view_v2",
            "label_free_runtime_binding_sha256": label_free_binding,
        },
    )
    object_id = id(bound)
    query_ref = weakref.ref(
        bound, lambda dead_ref: _drop_bound_query(object_id, dead_ref)
    )
    state = _BoundRuntimeQueryState(
        proposer_nonce=self._proposer_nonce,
        query_ref=query_ref,
        query_payload_sha256=_canonical_sha256(bound.to_dict()),
        capability=self._model_view,
    )
    with _BOUND_REGISTRY_LOCK:
      _BOUND_QUERY_REGISTRY[object_id] = state
    return bound

  def _verify_bound_query(self, bound: MateProgram) -> MateProgram:
    if type(bound) is not MateProgram:
      raise TypeError("B-Rep proposer requires its factory-bound MateProgram query")
    with _BOUND_REGISTRY_LOCK:
      state = _BOUND_QUERY_REGISTRY.get(id(bound))
    if (
        state is None
        or state.query_ref() is not bound
        or state.proposer_nonce is not self._proposer_nonce
        or state.capability is not self._model_view
    ):
      raise ValueError("runtime query is not bound to this authenticated proposer")
    state.capability.revalidate()
    if _canonical_sha256(bound.to_dict()) != state.query_payload_sha256:
      raise ValueError("bound runtime query was mutated after authentication")
    return bound

  def catalog_candidate_pool(
      self, bound_query: MateProgram
  ) -> list[MateProgramRetrieval]:
    """Create 19 private catalog candidates for one exact bound query."""

    self._checkpoint.revalidate()
    query = self._verify_bound_query(bound_query)
    result: list[MateProgramRetrieval] = []
    for row in self._catalog_rows:
      program = replace(
          query,
          program_id=str(row["program_id"]),
          relation_hint=str(row["descriptor"]["relation_hint"]),
          contact_type=str(row["descriptor"]["contact_type"]),
          residual_rotation=np.eye(3, dtype=float).tolist(),
          residual_translation=[0.0, 0.0, 0.0],
          metadata={
              **dict(query.metadata),
              "brep_catalog_program_index": int(row["program_index"]),
              "brep_catalog_descriptor": dict(row["descriptor"]),
              "brep_catalog_payload_sha256": self.provenance[
                  "catalog_payload_sha256"
              ],
          },
      )
      candidate = MateProgramRetrieval(
          program=program,
          score=0.0,
          reasons=["authenticated_train_catalog_candidate.v1"],
      )
      object_id = id(candidate)
      candidate_ref = weakref.ref(
          candidate,
          lambda dead_ref, object_id=object_id: _drop_bound_candidate(
              object_id, dead_ref
          ),
      )
      state = _BoundCatalogCandidateState(
          proposer_nonce=self._proposer_nonce,
          candidate_ref=candidate_ref,
          bound_query_ref=weakref.ref(bound_query),
          program_payload_sha256=_canonical_sha256(program.to_dict()),
          catalog_position=int(row["program_index"]),
      )
      with _BOUND_REGISTRY_LOCK:
        _BOUND_CANDIDATE_REGISTRY[object_id] = state
      result.append(candidate)
    return result

  def _verify_bound_candidate(
      self,
      candidate: MateProgramRetrieval,
      *,
      bound_query: MateProgram,
  ) -> tuple[MateProgram, int]:
    if type(candidate) is not MateProgramRetrieval:
      raise TypeError("B-Rep proposer accepts only its private catalog candidates")
    with _BOUND_REGISTRY_LOCK:
      state = _BOUND_CANDIDATE_REGISTRY.get(id(candidate))
    if (
        state is None
        or state.candidate_ref() is not candidate
        or state.proposer_nonce is not self._proposer_nonce
        or state.bound_query_ref() is not bound_query
    ):
      raise ValueError("catalog candidate is not bound to this query/proposer")
    if _canonical_sha256(candidate.program.to_dict()) != state.program_payload_sha256:
      raise ValueError("bound catalog candidate was mutated")
    row = self._catalog_rows[state.catalog_position]
    if (
        candidate.program.program_id != row["program_id"]
        or candidate.program.metadata.get("brep_catalog_program_index")
        != state.catalog_position
        or candidate.program.metadata.get("brep_catalog_descriptor")
        != row["descriptor"]
        or candidate.program.metadata.get("brep_catalog_payload_sha256")
        != self.provenance["catalog_payload_sha256"]
        or not np.array_equal(
            np.asarray(candidate.program.residual_rotation, dtype=float), np.eye(3)
        )
        or not np.array_equal(
            np.asarray(candidate.program.residual_translation, dtype=float),
            np.zeros(3),
        )
    ):
      raise ValueError("private catalog candidate policy differs")
    return candidate.program, state.catalog_position

  def propose(
      self,
      query: MateProgram,
      candidates: Sequence[MateProgramRetrieval],
      *,
      budget: ProposalBudget,
  ) -> list[ProgramHypothesis]:
    self._checkpoint.revalidate()
    runtime_query = self._verify_bound_query(query)
    if budget.max_occ_calls is None or budget.max_occ_calls != 0:
      raise ValueError("B-Rep scorer requires explicit max_occ_calls=0")
    if budget.wall_time_seconds is None:
      raise ValueError("B-Rep scorer requires an explicit cooperative wall-time budget")
    if budget.wall_time_seconds < 0.01:
      raise ValueError("B-Rep cooperative wall-time budget is unrealistically small")
    started = time.monotonic()
    deadline = started + budget.wall_time_seconds
    if len(candidates) != budget.candidate_pool_size:
      raise ValueError("B-Rep candidate pool must exactly match its budget")
    pool = list(candidates)
    batch = collate_authenticated_brep_learner_examples(
        [self._example]
    ).to(self._device)
    with torch.inference_mode():
      output = self._model(batch.inputs)
    if time.monotonic() > deadline:
      raise TimeoutError(
          "B-Rep program proposal exceeded its cooperative wall-time budget"
      )
    scored: list[tuple[float, int, MateProgramRetrieval, MateProgram, int]] = []
    seen_positions: set[int] = set()
    for retrieval_rank, row in enumerate(pool):
      program, position = self._verify_bound_candidate(row, bound_query=query)
      if position in seen_positions:
        raise ValueError("B-Rep candidate pool repeats a catalog program index")
      seen_positions.add(position)
      score = float(output.logits[0, position].item())
      if not math.isfinite(score):
        raise ValueError("B-Rep program proposer produced a non-finite score")
      scored.append((score, retrieval_rank, row, program, position))
      if time.monotonic() > deadline:
        raise TimeoutError(
            "B-Rep program proposal exceeded its cooperative wall-time budget"
        )
    if len(scored) < budget.top_k:
      raise ValueError("B-Rep candidate pool cannot satisfy top_k")
    scored.sort(key=lambda item: (-item[0], item[3].program_id))
    hypotheses: list[ProgramHypothesis] = []
    for score, retrieval_rank, _row, program, position in scored[: budget.top_k]:
      translation = output.residual_translation[0, position].detach().cpu()
      rotation = rotation_vector_to_matrix(
          output.residual_rotation_vector[0, position].detach().cpu()
      )
      hypotheses.append(
          ProgramHypothesis(
              program=program,
              score=score,
              program_id=str(program.program_id),
              program_class=mate_program_class_from_program(program),
              query_parent_interface_id=_canonical_sha256(runtime_query.interface_a)[:24],
              query_child_interface_id=_canonical_sha256(runtime_query.interface_b)[:24],
              residual_rotation=tuple(
                  tuple(float(value) for value in rotation_row)
                  for rotation_row in rotation.tolist()
              ),
              residual_translation=tuple(float(value) for value in translation.tolist()),
              score_components={"brep_catalog_logit": score},
              model_sha256=self.model_sha256,
              retrieval_rank=retrieval_rank,
              proposer_mode=self.mode,
              proposal_budget={
                  **budget.trace_dict(),
                  "wall_time_enforcement": "cooperative_post_forward.v1",
                  "hard_preemption": False,
              },
              proposer_protocol=PROGRAM_PROPOSER_PROTOCOL,
          )
      )
    return hypotheses


# The training-only model surface is defined in a dependency-minimal module so
# Torch-first V19 workers never import CAD or semantic-authority code.  The
# historical module re-exports those exact types for API compatibility.
from .v19_brep_program_model_v1 import (  # noqa: E402
    BRepGraphExample as BRepGraphExample,
    BRepLearnerExample as BRepLearnerExample,
    BRepProgramBatch as BRepProgramBatch,
    BRepProgramInputs as BRepProgramInputs,
    BRepProgramLearnerConfig as BRepProgramLearnerConfig,
    BRepProgramOutput as BRepProgramOutput,
    BRepProgramTargets as BRepProgramTargets,
    BRepProgramLearnerV1 as BRepProgramLearnerV1,
    LearnerBatchBudget as LearnerBatchBudget,
    collate_nonformal_brep_learner_examples as collate_nonformal_brep_learner_examples,
)
