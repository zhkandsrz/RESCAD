"""Immutable full-STEP, label-isolated B-Rep cache.

This cache deliberately does *not* convert the graph arrays in the v3 cache:
those arrays were cropped around authority-mapped endpoint faces.  Instead, a
v3 row is first authenticated, then both receipt-bound STEP streams are opened
again and every OCC face/adjacency is encoded.  The directional query A/B order
is preserved while each graph uses intrinsic OCC identity ordering.  Only after
that input hash is frozen is a factory-owned training label issued.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .benchmark_v2_model_view_v2 import (
    GraphSizeBudget,
    _GraphWitnessState,
    _CapturedStepStreamCache,
    _intrinsic_graph_from_face_subset,
    _get_private_witness,
)
from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    reverify_captured_file_artifact,
)
from .benchmark_v3_unlabeled_step_view import (
    EDGE_INPUT_DIM,
    EDGE_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    INTRINSIC_NODE_INPUT_DIM,
    INTRINSIC_NODE_FEATURE_NAMES,
    SCHEMA_VERSION as UNLABELED_VIEW_SCHEMA,
    SURFACE_TYPES,
    BenchmarkV3UnlabeledStepView,
    UnlabeledIntrinsicBRepGraph,
    assert_no_forbidden_model_visible_fields,
    _build_cache_owned_unlabeled_step_view,
    canonical_bytes,
    canonical_sha256,
    tensor_receipt,
)
from .brep_program_learner_v1 import (
    AuthenticatedCachedBRepLearnerExample,
    AuthenticatedShardedBRepTensorCacheV2,
)


CACHE_SCHEMA_VERSION = "brep_tensor_cache_v4_unlabeled.v1"
CACHE_ARTIFACT_SCHEMA_VERSION = "brep_tensor_cache_v4_unlabeled_artifact.v1"
LABEL_SCHEMA_VERSION = "brep_tensor_cache_v4_training_labels.v1"
LABEL_ARTIFACT_SCHEMA_VERSION = "brep_tensor_cache_v4_training_labels_artifact.v1"
AUDIT_SCHEMA_VERSION = "brep_tensor_cache_v4_unlabeled_audit.v1"
SOURCE_CACHE_SCHEMA_VERSION = "authenticated_sanitized_brep_tensor_cache.v3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_LABEL_FACTORY_TOKEN = object()
_TRAINING_ROW_FACTORY_TOKEN = object()
_LOADED_CACHE_FACTORY_TOKEN = object()
_LOADED_LABEL_FACTORY_TOKEN = object()
_PRODUCER_SOURCE_NAMES = (
    "brep_tensor_cache_v4_unlabeled.py",
    "tools/materialize_unlabeled_brep_tensor_cache_v4.py",
    "benchmark_v3_unlabeled_step_view.py",
    "benchmark_v2_model_view_v2.py",
    "benchmark_v2_training_provenance.py",
    "brep_full_graph_budget_preflight.py",
    "tools/preflight_unlabeled_full_graph_budget_v1.py",
    "brep_graph_budget_decision_v2.py",
    "tools/publish_brep_graph_budget_decision_v2.py",
)


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _binding(path: Path, *, name: str) -> dict[str, Any]:
  return {"name": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _write_exclusive(path: Path, raw: bytes) -> None:
  descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
  try:
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
      descriptor = -1
      stream.write(raw)
      stream.flush()
      os.fsync(stream.fileno())
  finally:
    if descriptor >= 0:
      os.close(descriptor)


def _strict_json(raw: bytes, *, label: str) -> Any:
  def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
      if key in result:
        raise ValueError(f"{label} contains duplicate JSON keys")
      result[key] = value
    return result

  try:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != expected:
    raise ValueError(f"{label} keys differ")
  return value


def _actual_int(value: Any, *, label: str, minimum: int = 0) -> int:
  if type(value) is not int or value < minimum:
    raise ValueError(f"{label} is not a valid integer")
  return value


def _sha256_value(value: Any, *, label: str) -> str:
  if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
    raise ValueError(f"{label} is not a SHA-256 value")
  return value


def _validate_binding(value: Any, *, label: str) -> Mapping[str, Any]:
  binding = _exact_keys(value, {"name", "bytes", "sha256"}, label=label)
  name = binding["name"]
  if not isinstance(name, str) or not name or Path(name).name != name:
    raise ValueError(f"{label} name differs")
  _actual_int(binding["bytes"], label=f"{label} bytes")
  _sha256_value(binding["sha256"], label=f"{label} sha256")
  return binding


def _producer_source_paths() -> Mapping[str, Path]:
  root = Path(__file__).resolve().parent
  return MappingProxyType(
      {
          "brep_tensor_cache_v4_unlabeled.py": root / "brep_tensor_cache_v4_unlabeled.py",
          "tools/materialize_unlabeled_brep_tensor_cache_v4.py": (
              root / "tools" / "materialize_unlabeled_brep_tensor_cache_v4.py"
          ),
          "benchmark_v3_unlabeled_step_view.py": root / "benchmark_v3_unlabeled_step_view.py",
          "benchmark_v2_model_view_v2.py": root / "benchmark_v2_model_view_v2.py",
          "benchmark_v2_training_provenance.py": (
              root / "benchmark_v2_training_provenance.py"
          ),
          "brep_full_graph_budget_preflight.py": root / "brep_full_graph_budget_preflight.py",
          "tools/preflight_unlabeled_full_graph_budget_v1.py": (
              root / "tools" / "preflight_unlabeled_full_graph_budget_v1.py"
          ),
          "brep_graph_budget_decision_v2.py": root / "brep_graph_budget_decision_v2.py",
          "tools/publish_brep_graph_budget_decision_v2.py": (
              root / "tools" / "publish_brep_graph_budget_decision_v2.py"
          ),
      }
  )


def producer_binding(*, expected_revision: str) -> dict[str, Any]:
  """Bind a cache to the exact checkout and all model-input producing code."""

  if not isinstance(expected_revision, str) or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
    raise ValueError("expected revision is not a full Git object ID")
  root = Path(__file__).resolve().parent
  observed = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=root, text=True
  ).strip()
  if observed != expected_revision:
    raise ValueError("unlabeled cache producer revision differs")
  dirty = subprocess.check_output(
      ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
  )
  if dirty.strip():
    raise ValueError("unlabeled cache producer tracked worktree is not clean")
  sources = {
      name: _file_sha256(path) for name, path in _producer_source_paths().items()
  }
  return {"revision": expected_revision, "source_sha256s": sources}


def _validate_producer_binding(
    value: Any,
    *,
    label: str,
    expected_historical_revision: str | None = None,
) -> Mapping[str, Any]:
  binding = _exact_keys(value, {"revision", "source_sha256s"}, label=label)
  revision = binding["revision"]
  if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError(f"{label} revision differs")
  sources = _exact_keys(
      binding["source_sha256s"], set(_PRODUCER_SOURCE_NAMES), label=f"{label} sources"
  )
  for name in _PRODUCER_SOURCE_NAMES:
    _sha256_value(sources[name], label=f"{label} source {name}")
  if expected_historical_revision is None:
    if dict(binding) != producer_binding(expected_revision=revision):
      raise ValueError(f"{label} differs from the checked-out producer")
  else:
    if revision != expected_historical_revision:
      raise ValueError(f"{label} differs from the externally pinned producer revision")
    root = Path(__file__).resolve().parent
    for name in _PRODUCER_SOURCE_NAMES:
      try:
        raw = subprocess.check_output(
            ["git", "show", f"{revision}:{name}"], cwd=root
        )
      except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} historical producer source is unavailable") from error
      if hashlib.sha256(raw).hexdigest() != sources[name]:
        raise ValueError(f"{label} historical producer source differs")
  return binding


def _tensor_matches(receipt: Mapping[str, Any], array: np.ndarray) -> bool:
  observed = tensor_receipt(str(receipt.get("name")), array)
  return all(observed[key] == receipt.get(key) for key in ("name", "dtype", "shape", "sha256"))


def _validate_preflight_binding(value: Any) -> Mapping[str, Any]:
  binding = _exact_keys(
      value,
      {
          "artifact_sha256", "ledger_sha256", "ledger_payload_sha256",
          "case_domain_sha256", "case_exclusion_set_sha256",
      },
      label="full-graph preflight binding",
  )
  for key, digest in binding.items():
    _sha256_value(digest, label=f"full-graph preflight {key}")
  return binding


def _validate_budget_decision_binding(value: Any) -> Mapping[str, Any]:
  binding = _exact_keys(
      value,
      {
          "artifact_sha256", "decision_sha256", "decision_payload_sha256",
          "case_domain_sha256", "decision_exclusion_set_sha256",
      },
      label="graph-budget decision binding",
  )
  for key, digest in binding.items():
    _sha256_value(digest, label=f"graph-budget decision {key}")
  return binding


def _input_domain_sha256(
    input_hashes: Sequence[str],
    *,
    preflight_binding: Mapping[str, Any],
    decision_binding: Mapping[str, Any],
    case_selection_policy: str,
) -> str:
  return canonical_sha256(
      {
          "full_graph_budget_preflight": dict(preflight_binding),
          "graph_budget_decision": dict(decision_binding),
          "case_selection_policy": case_selection_policy,
          "input_sha256s": list(input_hashes),
      }
  )


def _case_selected_by_budget_authorities(
    *,
    preflight: Any,
    budget_decision: Any,
    case_selection_policy: str,
    split: str,
    case_index: int,
) -> bool:
  if not budget_decision.allows(split=split, case_index=case_index):
    return False
  if case_selection_policy == "all_decision_admitted_cases":
    return True
  if case_selection_policy == "preflight_excluded_decision_admitted_cases":
    return not preflight.allows(split=split, case_index=case_index)
  raise ValueError("case selection policy differs")


def _graph_payload_to_unlabeled(
    graph: Mapping[str, Any],
    *,
    raw_face_count: int,
    raw_adjacency_count: int,
    complete_component: bool,
    budget: GraphSizeBudget,
) -> UnlabeledIntrinsicBRepGraph:
  """Convert a proven whole STEP graph, never an endpoint-local v2 graph."""

  if complete_component is not True:
    raise ValueError("unlabeled cache refuses a graph without complete-component proof")
  nodes = graph.get("nodes")
  edges = graph.get("edges")
  if not isinstance(nodes, list) or not isinstance(edges, list):
    raise ValueError("full graph payload lacks nodes/edges")
  if len(nodes) != raw_face_count or len(edges) != raw_adjacency_count:
    raise ValueError("full graph does not cover the complete STEP topology")
  if len(nodes) > budget.max_faces_per_graph or len(edges) > budget.max_edges_per_graph:
    raise ValueError("full graph exceeds the frozen 1024/4096 budget")
  rows: list[list[float]] = []
  for node in nodes:
    if not isinstance(node, Mapping):
      raise ValueError("full graph node is not an object")
    features = node.get("features")
    mask = node.get("feature_mask")
    surface = node.get("surface_type")
    if (
        not isinstance(features, list)
        or len(features) != 6
        or not isinstance(mask, list)
        or len(mask) != 6
        or surface not in SURFACE_TYPES
    ):
      raise ValueError("full graph node differs from the intrinsic feature schema")
    rows.append(
        [float(value) for value in features]
        + [float(bool(value)) for value in mask]
        + [float(surface == item) for item in SURFACE_TYPES]
    )
  edge_index = np.asarray(
      [[int(row["source"]), int(row["target"])] for row in edges], dtype=np.int64
  ).reshape(-1, 2)
  edge_features = np.asarray(
      [[float(value) for value in row["features"]] for row in edges],
      dtype=np.float32,
  ).reshape(-1, EDGE_INPUT_DIM)
  return UnlabeledIntrinsicBRepGraph(
      node_features=np.asarray(rows, dtype=np.float32).reshape(-1, INTRINSIC_NODE_INPUT_DIM),
      edge_index=edge_index,
      edge_features=edge_features,
  )


@dataclass(frozen=True, slots=True)
class FullGraphProof:
  step_sha256: str
  face_count: int
  adjacency_count: int
  full_topology_sha256: str
  complete_component: bool = True

  def __post_init__(self) -> None:
    if _SHA256.fullmatch(self.step_sha256) is None or _SHA256.fullmatch(
        self.full_topology_sha256
    ) is None:
      raise ValueError("full graph proof requires SHA-256 commitments")
    if self.complete_component is not True or self.face_count < 1 or self.adjacency_count < 0:
      raise ValueError("full graph proof is incomplete")

  def public_receipt(self) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema_version": "benchmark_v3_full_step_graph_proof.v1",
            "step_sha256": self.step_sha256,
            "face_count": self.face_count,
            "adjacency_count": self.adjacency_count,
            "full_topology_sha256": self.full_topology_sha256,
            "complete_component": True,
        }
    )


class AuthenticatedUnlabeledTrainingLabel:
  """Verifier-issued gold.  A model input has no link back to this object."""

  __slots__ = (
      "_target_program_index",
      "_residual_translation",
      "_residual_rotation_vector",
      "_residual_mask",
      "_endpoint_graph_node_indices",
      "_commitment_sha256",
  )

  def __init__(
      self,
      *,
      target_program_index: int,
      residual_translation: np.ndarray,
      residual_rotation_vector: np.ndarray,
      residual_mask: bool,
      endpoint_graph_node_indices: Sequence[Sequence[int]],
      _factory_token: object,
  ) -> None:
    if _factory_token is not _TRAINING_LABEL_FACTORY_TOKEN:
      raise TypeError("unlabeled training labels are verifier-factory-only")
    translation = np.asarray(residual_translation, dtype=np.float32).reshape(3).copy()
    rotation = np.asarray(residual_rotation_vector, dtype=np.float32).reshape(3).copy()
    if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
      raise ValueError("unlabeled training label residual is non-finite")
    translation.setflags(write=False)
    rotation.setflags(write=False)
    self._target_program_index = int(target_program_index)
    self._residual_translation = translation
    self._residual_rotation_vector = rotation
    self._residual_mask = bool(residual_mask)
    endpoints = tuple(tuple(int(index) for index in values) for values in endpoint_graph_node_indices)
    if len(endpoints) != 2 or any(not values or min(values) < 0 for values in endpoints):
      raise ValueError("training label requires two nonempty canonical endpoint sets")
    self._endpoint_graph_node_indices = endpoints
    self._commitment_sha256 = canonical_sha256(
        {
            "schema_version": LABEL_SCHEMA_VERSION,
            "target_program_index": self._target_program_index,
            "residual_translation": translation.tolist(),
            "residual_rotation_vector": rotation.tolist(),
            "residual_mask": self._residual_mask,
            "endpoint_graph_node_indices": [list(values) for values in endpoints],
        }
    )

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated training labels are not serializable")

  @property
  def target_program_index(self) -> int:
    return self._target_program_index

  @property
  def residual_translation(self) -> np.ndarray:
    return self._residual_translation

  @property
  def residual_rotation_vector(self) -> np.ndarray:
    return self._residual_rotation_vector

  @property
  def residual_mask(self) -> bool:
    return self._residual_mask

  @property
  def endpoint_graph_node_indices(self) -> tuple[tuple[int, ...], ...]:
    return self._endpoint_graph_node_indices

  @property
  def commitment_sha256(self) -> str:
    return self._commitment_sha256


class AuthenticatedUnlabeledTrainingRow:
  __slots__ = ("_model_input", "_training_label", "_proofs")

  def __init__(
      self,
      *,
      model_input: BenchmarkV3UnlabeledStepView,
      training_label: AuthenticatedUnlabeledTrainingLabel,
      proofs: Sequence[FullGraphProof],
      _factory_token: object,
  ) -> None:
    if _factory_token is not _TRAINING_ROW_FACTORY_TOKEN:
      raise TypeError("unlabeled training rows are verifier-factory-only")
    if type(model_input) is not BenchmarkV3UnlabeledStepView:
      raise TypeError("training row model input differs")
    if type(training_label) is not AuthenticatedUnlabeledTrainingLabel:
      raise TypeError("training row label differs")
    if len(proofs) != len(model_input.graphs) or any(type(row) is not FullGraphProof for row in proofs):
      raise ValueError("training row full-graph proof ledger differs")
    self._model_input = model_input
    self._training_label = training_label
    self._proofs = tuple(proofs)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated unlabeled training rows are not serializable")

  @property
  def model_input(self) -> BenchmarkV3UnlabeledStepView:
    return self._model_input

  @property
  def training_label(self) -> AuthenticatedUnlabeledTrainingLabel:
    return self._training_label

  @property
  def full_graph_proofs(self) -> tuple[FullGraphProof, ...]:
    return self._proofs


def _issue_label(
    example: Any, *, endpoint_graph_node_indices: Sequence[Sequence[int]]
) -> AuthenticatedUnlabeledTrainingLabel:
  return AuthenticatedUnlabeledTrainingLabel(
      target_program_index=example.target_program_index,
      residual_translation=example.residual_translation.detach().cpu().numpy(),
      residual_rotation_vector=example.residual_rotation_vector.detach().cpu().numpy(),
      residual_mask=example.residual_mask,
      endpoint_graph_node_indices=endpoint_graph_node_indices,
      _factory_token=_TRAINING_LABEL_FACTORY_TOKEN,
  )


class _FullStepGraphCache:
  def __init__(self, *, budget_decision: Any) -> None:
    from .brep_graph_budget_decision_v2 import AuthenticatedGraphBudgetDecisionV2
    if type(budget_decision) is not AuthenticatedGraphBudgetDecisionV2:
      raise TypeError("full STEP graphs require an authenticated budget decision")
    self._budget = GraphSizeBudget(
        max_graphs=2,
        max_faces_per_graph=budget_decision.max_faces_per_graph,
        max_edges_per_graph=budget_decision.max_edges_per_graph,
        max_examples=1,
        max_program_candidates_per_example=19,
    )
    self._streams = _CapturedStepStreamCache()
    self._values: dict[
        str, tuple[UnlabeledIntrinsicBRepGraph, FullGraphProof, Mapping[int, int], tuple[str, ...]]
    ] = {}

  def load(
      self, captured: CapturedFileArtifact
  ) -> tuple[
      UnlabeledIntrinsicBRepGraph, FullGraphProof, Mapping[int, int], tuple[str, ...]
  ]:
    existing = self._values.get(captured.sha256)
    if existing is not None:
      reverify_captured_file_artifact(captured, label="unlabeled full STEP reuse")
      return existing
    reverify_captured_file_artifact(captured, label="unlabeled full STEP input")
    faces, face_edges, edge_groups, signatures = (
        self._streams.load_step_topology(captured)
    )
    adjacency = sum(
        1
        for group in edge_groups
        if len(group) == 2 and group[0][0] != group[1][0]
    )
    if (
        len(faces) > self._budget.max_faces_per_graph
        or adjacency > self._budget.max_edges_per_graph
    ):
      raise ValueError("full STEP topology exceeds its authenticated decision budget")
    graph, remap = _intrinsic_graph_from_face_subset(
        faces,
        face_edges,
        edge_groups,
        range(len(faces)),
        budget=self._budget,
    )
    if set(remap) != set(range(len(faces))):
      raise ValueError("full STEP graph omitted or reordered an OCC face identity")
    safe_graph = _graph_payload_to_unlabeled(
        graph,
        raw_face_count=len(faces),
        raw_adjacency_count=adjacency,
        complete_component=True,
        budget=self._budget,
    )
    topology_payload = {
        "face_signature_multiset": sorted(signatures),
        "adjacency_signature_multiset": sorted(
            sorted((signatures[group[0][0]], signatures[group[1][0]]))
            for group in edge_groups
            if len(group) == 2 and group[0][0] != group[1][0]
        ),
    }
    proof = FullGraphProof(
        step_sha256=captured.sha256,
        face_count=len(faces),
        adjacency_count=adjacency,
        full_topology_sha256=canonical_sha256(topology_payload),
    )
    result = (safe_graph, proof, MappingProxyType(dict(remap)), tuple(signatures))
    self._values[captured.sha256] = result
    return result


def materialize_unlabeled_training_row(
    *,
    index: Any,
    cached_row: AuthenticatedCachedBRepLearnerExample,
    full_graph_cache: _FullStepGraphCache,
) -> AuthenticatedUnlabeledTrainingRow:
  """Rebuild one full graph pair before issuing its separate gold label."""

  if type(cached_row) is not AuthenticatedCachedBRepLearnerExample:
    raise TypeError("unlabeled conversion requires an authenticated v3 cache row")
  cached_row.revalidate()
  lineage = cached_row.lineage
  split = lineage.get("split")
  case_index = lineage.get("case_index")
  program_index = lineage.get("program_index")
  if split not in {"train", "dev"} or type(case_index) is not int or type(program_index) is not int:
    raise ValueError("authenticated v3 row lacks split/case/program lineage")
  capability = index.materialize_graph_capability(
      split=split,
      case_index=case_index,
      program_index=program_index,
      # Replay the already-authenticated v3 local-capability contract at its
      # original 1024/4096 budget.  The independent decision applies only to
      # the endpoint-free complete graph extracted below.
      budget=GraphSizeBudget(
          max_graphs=2,
          max_faces_per_graph=1024,
          max_edges_per_graph=4096,
          max_examples=1,
          max_program_candidates_per_example=19,
      ),
  )
  return _materialize_unlabeled_training_row_from_capability(
      capability=capability, cached_row=cached_row,
      full_graph_cache=full_graph_cache,
  )


def _materialize_unlabeled_training_row_from_capability(
    *, capability: Any, cached_row: AuthenticatedCachedBRepLearnerExample,
    full_graph_cache: _FullStepGraphCache,
) -> AuthenticatedUnlabeledTrainingRow:
  if type(cached_row) is not AuthenticatedCachedBRepLearnerExample:
    raise TypeError("unlabeled capability replay requires an authenticated cache row")
  cached_row.revalidate()
  capability.revalidate()
  if capability.sha256 != cached_row.example.model_view_sha256:
    raise ValueError("v3 cache row differs from replayed semantic authority")
  witness = _get_private_witness(capability, _GraphWitnessState)
  if len(witness.audit_inputs) != 2:
    raise ValueError("semantic row does not bind exactly two receipt-bound STEP inputs")
  # Freeze the complete, endpoint-free input before reading the target tensors.
  # Preserve query A/B roles.  Sorting the pair would invalidate directional
  # program/residual supervision even though each graph is intrinsically ordered.
  graph_proof_pairs = [full_graph_cache.load(audit.step_capture) for audit in witness.audit_inputs]
  graphs = tuple(value[0] for value in graph_proof_pairs)
  proofs = tuple(value[1] for value in graph_proof_pairs)
  model_input = _build_cache_owned_unlabeled_step_view(graphs=graphs)
  assert_no_forbidden_model_visible_fields(model_input.receipt_payload())
  endpoint_graph_node_indices: list[tuple[int, ...]] = []
  for audit, value in zip(witness.audit_inputs, graph_proof_pairs, strict=True):
    remap = value[2]
    signatures = value[3]
    observed = tuple(signatures[index] for index in audit.endpoint_raw_indices)
    if observed != audit.endpoint_face_signatures:
      raise ValueError("full graph endpoint identity differs from authenticated witness")
    endpoint_graph_node_indices.append(
        tuple(sorted(remap[index] for index in audit.endpoint_raw_indices))
    )
  label = _issue_label(
      cached_row.example,
      endpoint_graph_node_indices=endpoint_graph_node_indices,
  )
  return AuthenticatedUnlabeledTrainingRow(
      model_input=model_input,
      training_label=label,
      proofs=proofs,
      _factory_token=_TRAINING_ROW_FACTORY_TOKEN,
  )


def materialize_unlabeled_training_rows_by_case(
    *, index: Any,
    cached_rows: Sequence[AuthenticatedCachedBRepLearnerExample],
    full_graph_cache: _FullStepGraphCache,
) -> tuple[AuthenticatedUnlabeledTrainingRow, ...]:
  """Batch one case while preserving exact per-program endpoint authority."""

  rows = tuple(cached_rows)
  if not rows or any(type(row) is not AuthenticatedCachedBRepLearnerExample for row in rows):
    raise TypeError("batched unlabeled conversion requires authenticated v3 cache rows")
  lineages = [row.lineage for row in rows]
  split = lineages[0].get("split")
  case_index = lineages[0].get("case_index")
  program_indices = tuple(lineage.get("program_index") for lineage in lineages)
  if (
      split not in {"train", "dev"}
      or type(case_index) is not int
      or any(
          lineage.get("split") != split or lineage.get("case_index") != case_index
          for lineage in lineages
      )
      or any(type(value) is not int for value in program_indices)
      or tuple(sorted(set(program_indices))) != program_indices
  ):
    raise ValueError("batched unlabeled rows differ in case/program order")
  capabilities = index.materialize_case_graph_capabilities(
      split=split, case_index=case_index, program_indices=program_indices,
      budget=GraphSizeBudget(
          max_graphs=2, max_faces_per_graph=1024, max_edges_per_graph=4096,
          max_examples=1, max_program_candidates_per_example=19,
      ),
  )
  return tuple(
      _materialize_unlabeled_training_row_from_capability(
          capability=capability, cached_row=row,
          full_graph_cache=full_graph_cache,
      )
      for capability, row in zip(capabilities, rows, strict=True)
  )


def materialize_unlabeled_training_rows_with_authority_by_case(
    *, index: Any,
    cached_rows: Sequence[AuthenticatedCachedBRepLearnerExample],
    full_graph_cache: _FullStepGraphCache,
) -> tuple[tuple[AuthenticatedUnlabeledTrainingRow, Mapping[str, Any]], ...]:
  """Batch a case and retain a private producer-only authority projection."""

  rows = tuple(cached_rows)
  if not rows or any(type(row) is not AuthenticatedCachedBRepLearnerExample for row in rows):
    raise TypeError("authority batch requires authenticated v3 cache rows")
  lineages = [row.lineage for row in rows]
  split = lineages[0].get("split")
  case_index = lineages[0].get("case_index")
  program_indices = tuple(lineage.get("program_index") for lineage in lineages)
  if (
      split not in {"train", "dev"} or type(case_index) is not int
      or any(
          lineage.get("split") != split or lineage.get("case_index") != case_index
          for lineage in lineages
      )
      or any(type(value) is not int for value in program_indices)
      or tuple(sorted(set(program_indices))) != program_indices
  ):
    raise ValueError("authority batch case/program order differs")
  capabilities = index.materialize_case_graph_capabilities(
      split=split, case_index=case_index, program_indices=program_indices,
      budget=GraphSizeBudget(
          max_graphs=2, max_faces_per_graph=1024, max_edges_per_graph=4096,
          max_examples=1, max_program_candidates_per_example=19,
      ),
  )
  result = []
  for capability, row in zip(capabilities, rows, strict=True):
    result.append((
        _materialize_unlabeled_training_row_from_capability(
            capability=capability, cached_row=row, full_graph_cache=full_graph_cache,
        ),
        capability.authority_projection(),
    ))
  return tuple(result)


def _row_arrays(
    row: AuthenticatedUnlabeledTrainingRow, *, ordinal: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
  prefix = f"r{ordinal:08d}"
  safe: dict[str, np.ndarray] = {}
  for index, graph in enumerate(row.model_input.graphs):
    safe[f"{prefix}_g{index}_nodes"] = graph.node_features
    safe[f"{prefix}_g{index}_edge_index"] = graph.edge_index
    safe[f"{prefix}_g{index}_edge_features"] = graph.edge_features
  label = row.training_label
  gold = {
      f"{prefix}_class": np.asarray([label.target_program_index], dtype=np.int64),
      f"{prefix}_translation": label.residual_translation,
      f"{prefix}_rotation": label.residual_rotation_vector,
      f"{prefix}_mask": np.asarray([label.residual_mask], dtype=np.bool_),
      f"{prefix}_endpoint_a": np.asarray(
          label.endpoint_graph_node_indices[0], dtype=np.int64
      ),
      f"{prefix}_endpoint_b": np.asarray(
          label.endpoint_graph_node_indices[1], dtype=np.int64
      ),
  }
  safe_row = {
      "row_ordinal": ordinal,
      "schema_version": UNLABELED_VIEW_SCHEMA,
      "input_sha256": row.model_input.input_sha256,
      "graph_count": len(row.model_input.graphs),
      "view_receipt": row.model_input.receipt_payload(),
      "full_graph_proofs": [dict(proof.public_receipt()) for proof in row.full_graph_proofs],
      "tensors": [tensor_receipt(name, value) for name, value in sorted(safe.items())],
  }
  safe_row["row_payload_sha256"] = canonical_sha256(safe_row)
  gold_row = {
      "row_ordinal": ordinal,
      "input_sha256": row.model_input.input_sha256,
      "label_commitment_sha256": label.commitment_sha256,
      "tensors": [tensor_receipt(name, value) for name, value in sorted(gold.items())],
  }
  gold_row["row_payload_sha256"] = canonical_sha256(gold_row)
  return safe, gold, safe_row, gold_row


def publish_unlabeled_tensor_cache_v4(
    output_directory: str | Path,
    *,
    training_label_output_directory: str | Path,
    index: Any,
    source_cache: AuthenticatedShardedBRepTensorCacheV2,
    preflight: Any,
    budget_decision: Any,
    expected_revision: str,
    row_limit: int,
    shard_row_limit: int = 25,
    split_order: Sequence[str] = ("train", "dev"),
    case_selection_policy: str = "all_decision_admitted_cases",
) -> tuple[str, str]:
  """Publish an immutable pilot/full cache from authenticated p0 rows."""

  if type(source_cache) is not AuthenticatedShardedBRepTensorCacheV2:
    raise TypeError("v4 converter requires the authenticated v3 cache capability")
  from .brep_full_graph_budget_preflight import AuthenticatedFullGraphBudgetPreflight
  from .brep_graph_budget_decision_v2 import AuthenticatedGraphBudgetDecisionV2
  if type(preflight) is not AuthenticatedFullGraphBudgetPreflight:
    raise TypeError("v4 converter requires an authenticated full-graph preflight")
  preflight_binding = _validate_preflight_binding(preflight.binding)
  if type(budget_decision) is not AuthenticatedGraphBudgetDecisionV2:
    raise TypeError("v4 converter requires an authenticated graph-budget decision")
  decision_binding = _validate_budget_decision_binding(budget_decision.binding)
  if (
      dict(budget_decision.preflight_binding) != dict(preflight_binding)
      or decision_binding["case_domain_sha256"] != preflight_binding["case_domain_sha256"]
  ):
    raise ValueError("graph-budget decision domain differs from preflight")
  if case_selection_policy not in {
      "all_decision_admitted_cases",
      "preflight_excluded_decision_admitted_cases",
  }:
    raise ValueError("case selection policy differs")
  if type(row_limit) is not int or row_limit < 1:
    raise ValueError("row_limit must be a positive actual integer")
  if type(shard_row_limit) is not int or shard_row_limit < 1:
    raise ValueError("shard_row_limit must be positive")
  build_binding = producer_binding(expected_revision=expected_revision)
  output = Path(output_directory)
  label_output = Path(training_label_output_directory)
  if output.exists() or label_output.exists():
    raise FileExistsError("unlabeled input or training-label output already exists")
  if output.resolve() == label_output.resolve():
    raise ValueError("model input and training labels require separate sibling roots")
  output.parent.mkdir(parents=True, exist_ok=True)
  label_output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
  label_root = Path(
      tempfile.mkdtemp(prefix=f".{label_output.name}.staging-", dir=label_output.parent)
  )
  graph_cache = _FullStepGraphCache(budget_decision=budget_decision)
  safe_rows: list[dict[str, Any]] = []
  label_rows: list[dict[str, Any]] = []
  safe_shards: list[dict[str, Any]] = []
  label_shards: list[dict[str, Any]] = []
  pending: list[AuthenticatedUnlabeledTrainingRow] = []
  selected_cases: set[tuple[str, int]] = set()

  def flush() -> None:
    if not pending:
      return
    shard_index = len(safe_shards)
    safe_arrays: dict[str, np.ndarray] = {}
    label_arrays: dict[str, np.ndarray] = {}
    shard_safe_rows: list[dict[str, Any]] = []
    shard_label_rows: list[dict[str, Any]] = []
    start = len(safe_rows)
    for offset, value in enumerate(pending):
      safe, gold, safe_row, gold_row = _row_arrays(value, ordinal=start + offset)
      safe_arrays.update(safe)
      label_arrays.update(gold)
      shard_safe_rows.append(safe_row)
      shard_label_rows.append(gold_row)
    for arrays, root, prefix, rows, ledger in (
        (safe_arrays, staging, "input-shard", shard_safe_rows, safe_shards),
        (label_arrays, label_root, "label-shard", shard_label_rows, label_shards),
    ):
      stream = io.BytesIO()
      np.savez_compressed(stream, **arrays)
      raw = stream.getvalue()
      digest = hashlib.sha256(raw).hexdigest()
      name = f"{prefix}-{shard_index:06d}-{digest[:16]}.npz"
      _write_exclusive(root / name, raw)
      ledger.append(
          {
              "shard_index": shard_index,
              "name": name,
              "bytes": len(raw),
              "sha256": digest,
              "row_start": start,
              "row_count": len(rows),
              "rows_payload_sha256": canonical_sha256(
                  [item["row_payload_sha256"] for item in rows]
              ),
          }
      )
    safe_rows.extend(shard_safe_rows)
    label_rows.extend(shard_label_rows)
    pending.clear()

  try:
    for split in split_order:
      for cached in source_cache.iter_examples(split=split):
        lineage = cached.lineage
        row_split = str(lineage.get("split"))
        row_case_index = int(lineage.get("case_index"))
        if not _case_selected_by_budget_authorities(
            preflight=preflight,
            budget_decision=budget_decision,
            case_selection_policy=case_selection_policy,
            split=row_split,
            case_index=row_case_index,
        ):
          continue
        selected_cases.add((row_split, row_case_index))
        pending.append(
            materialize_unlabeled_training_row(
                index=index, cached_row=cached, full_graph_cache=graph_cache
            )
        )
        if len(pending) == shard_row_limit:
          flush()
        if len(safe_rows) + len(pending) >= row_limit:
          break
      if len(safe_rows) + len(pending) >= row_limit:
        break
    flush()
    if len(safe_rows) != row_limit:
      raise ValueError("authenticated source cache has fewer rows than requested")
    input_hashes = [row["input_sha256"] for row in safe_rows]
    label_commitments = [row["label_commitment_sha256"] for row in label_rows]
    label_manifest = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "scope": "factory_only_training_supervision",
        "source_cache_artifact_sha256": source_cache.artifact_sha256,
        "full_graph_budget_preflight": dict(preflight_binding),
        "graph_budget_decision": dict(decision_binding),
        "case_selection_policy": case_selection_policy,
        "case_count": len(selected_cases),
        "producer_binding": build_binding,
        "row_count": len(label_rows),
        "input_domain_sha256": _input_domain_sha256(
            input_hashes,
            preflight_binding=preflight_binding,
            decision_binding=decision_binding,
            case_selection_policy=case_selection_policy,
        ),
        "label_commitment_ledger_sha256": canonical_sha256(label_commitments),
        "rows": label_rows,
        "shards": label_shards,
    }
    label_manifest["manifest_payload_sha256"] = canonical_sha256(label_manifest)
    label_manifest_path = label_root / "manifest.json"
    _write_exclusive(label_manifest_path, canonical_bytes(label_manifest))
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scope": "p0_train_dev_full_step_inputs_only",
        "model_view_schema_version": UNLABELED_VIEW_SCHEMA,
        "source_cache_schema_version": SOURCE_CACHE_SCHEMA_VERSION,
        "source_cache_artifact_sha256": source_cache.artifact_sha256,
        "full_graph_budget_preflight": dict(preflight_binding),
        "graph_budget_decision": dict(decision_binding),
        "case_selection_policy": case_selection_policy,
        "case_count": len(selected_cases),
        "producer_binding": build_binding,
        "graph_policy": {
            "selection": "all_occ_faces_and_all_two_owner_adjacencies",
            "pair_order": "query_role_a_then_query_role_b",
            "max_faces_per_graph": budget_decision.max_faces_per_graph,
            "max_edges_per_graph": budget_decision.max_edges_per_graph,
            "endpoint_annotations": "forbidden",
        },
        "row_count": len(safe_rows),
        "input_domain_sha256": _input_domain_sha256(
            input_hashes,
            preflight_binding=preflight_binding,
            decision_binding=decision_binding,
            case_selection_policy=case_selection_policy,
        ),
        "rows": safe_rows,
        "shards": safe_shards,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    manifest_path = staging / "manifest.json"
    _write_exclusive(manifest_path, canonical_bytes(manifest))
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "producer_binding": build_binding,
        "row_count": len(safe_rows),
        "source_cache_artifact_sha256": source_cache.artifact_sha256,
        "full_graph_budget_preflight": dict(preflight_binding),
        "graph_budget_decision": dict(decision_binding),
        "case_selection_policy": case_selection_policy,
        "case_count": len(selected_cases),
        "input_manifest": _binding(manifest_path, name="manifest.json"),
        "training_label_manifest": _binding(
            label_manifest_path, name="manifest.json"
        ),
        "input_domain_sha256": manifest["input_domain_sha256"],
        "all_graphs_complete_component": all(
            proof["complete_component"]
            for row in safe_rows for proof in row["full_graph_proofs"]
        ),
        "max_face_count": max(
            proof["face_count"] for row in safe_rows for proof in row["full_graph_proofs"]
        ),
        "max_adjacency_count": max(
            proof["adjacency_count"] for row in safe_rows for proof in row["full_graph_proofs"]
        ),
        "forbidden_model_fields": [],
        "final_test_touched": False,
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    audit_path = label_root / "audit.json"
    _write_exclusive(audit_path, canonical_bytes(audit))
    label_artifact = {
        "schema_version": LABEL_ARTIFACT_SCHEMA_VERSION,
        "producer_binding": build_binding,
        "manifest": _binding(label_manifest_path, name="manifest.json"),
        "audit": _binding(audit_path, name="audit.json"),
        "shards": [_binding(label_root / row["name"], name=row["name"]) for row in label_shards],
    }
    label_artifact["artifact_payload_sha256"] = canonical_sha256(label_artifact)
    label_artifact_path = label_root / "artifact.json"
    _write_exclusive(label_artifact_path, canonical_bytes(label_artifact))
    label_pin = _file_sha256(label_artifact_path)
    artifact = {
        "schema_version": CACHE_ARTIFACT_SCHEMA_VERSION,
        "producer_binding": build_binding,
        "manifest": _binding(manifest_path, name="manifest.json"),
        "shards": [_binding(staging / row["name"], name=row["name"]) for row in safe_shards],
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    artifact_path = staging / "artifact.json"
    _write_exclusive(artifact_path, canonical_bytes(artifact))
    cache_pin = _file_sha256(artifact_path)
    os.replace(staging, output)
    try:
      os.replace(label_root, label_output)
    except BaseException:
      shutil.rmtree(output, ignore_errors=True)
      raise
    return cache_pin, label_pin
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(label_root, ignore_errors=True)
    raise


def _validate_tensor_receipt_payload(value: Any, *, label: str) -> Mapping[str, Any]:
  receipt = _exact_keys(value, {"name", "dtype", "shape", "sha256"}, label=label)
  if not isinstance(receipt["name"], str) or not receipt["name"]:
    raise ValueError(f"{label} name differs")
  if receipt["dtype"] not in {"float32", "int64", "bool"}:
    raise ValueError(f"{label} dtype differs")
  shape = receipt["shape"]
  if not isinstance(shape, list) or not shape or any(
      type(item) is not int or item < 0 for item in shape
  ):
    raise ValueError(f"{label} shape differs")
  _sha256_value(receipt["sha256"], label=f"{label} sha256")
  return receipt


def _validate_view_receipt(value: Any, *, label: str) -> Mapping[str, Any]:
  view = _exact_keys(
      value, {"schema_version", "graphs", "feature_contract", "input_sha256"}, label=label
  )
  if view["schema_version"] != UNLABELED_VIEW_SCHEMA:
    raise ValueError(f"{label} schema differs")
  _sha256_value(view["input_sha256"], label=f"{label} input sha256")
  graphs = view["graphs"]
  if not isinstance(graphs, list) or len(graphs) != 2:
    raise ValueError(f"{label} graph count differs")
  for graph_index, graph_value in enumerate(graphs):
    graph = _exact_keys(
        graph_value, {"schema_version", "tensors"}, label=f"{label} graph"
    )
    if graph["schema_version"] != GRAPH_SCHEMA_VERSION:
      raise ValueError(f"{label} graph schema differs")
    tensors = graph["tensors"]
    if not isinstance(tensors, list) or len(tensors) != 3:
      raise ValueError(f"{label} graph tensor count differs")
    for tensor_index, tensor in enumerate(tensors):
      _validate_tensor_receipt_payload(
          tensor, label=f"{label} graph {graph_index} tensor {tensor_index}"
      )
  contract = _exact_keys(
      view["feature_contract"],
      {"intrinsic_node_feature_names", "edge_feature_names", "endpoint_annotation_columns"},
      label=f"{label} feature contract",
  )
  if contract["intrinsic_node_feature_names"] != list(INTRINSIC_NODE_FEATURE_NAMES):
    raise ValueError(f"{label} node feature contract differs")
  if contract["edge_feature_names"] != list(EDGE_FEATURE_NAMES):
    raise ValueError(f"{label} edge feature contract differs")
  if contract["endpoint_annotation_columns"] != 0:
    raise ValueError(f"{label} endpoint contract differs")
  expected_input = dict(view)
  observed_input = expected_input.pop("input_sha256")
  if observed_input != canonical_sha256(expected_input):
    raise ValueError(f"{label} input self hash differs")
  assert_no_forbidden_model_visible_fields(view)
  return view


def _validate_full_graph_proof(value: Any, *, label: str) -> Mapping[str, Any]:
  proof = _exact_keys(
      value,
      {
          "schema_version", "step_sha256", "face_count", "adjacency_count",
          "full_topology_sha256", "complete_component",
      },
      label=label,
  )
  if proof["schema_version"] != "benchmark_v3_full_step_graph_proof.v1":
    raise ValueError(f"{label} schema differs")
  _sha256_value(proof["step_sha256"], label=f"{label} STEP sha256")
  _sha256_value(proof["full_topology_sha256"], label=f"{label} topology sha256")
  _actual_int(proof["face_count"], label=f"{label} face count", minimum=1)
  _actual_int(proof["adjacency_count"], label=f"{label} adjacency count")
  if proof["complete_component"] is not True:
    raise ValueError(f"{label} is not complete")
  return proof


def _validate_safe_row(value: Any, *, expected_ordinal: int) -> Mapping[str, Any]:
  row = _exact_keys(
      value,
      {
          "row_ordinal", "schema_version", "input_sha256", "graph_count",
          "view_receipt", "full_graph_proofs", "tensors", "row_payload_sha256",
      },
      label="unlabeled cache row",
  )
  if row["row_ordinal"] != expected_ordinal or type(row["row_ordinal"]) is not int:
    raise ValueError("unlabeled cache row ordinal differs")
  if row["schema_version"] != UNLABELED_VIEW_SCHEMA or row["graph_count"] != 2:
    raise ValueError("unlabeled cache row graph schema differs")
  _sha256_value(row["input_sha256"], label="unlabeled cache row input sha256")
  view = _validate_view_receipt(row["view_receipt"], label="unlabeled cache row view")
  if view["input_sha256"] != row["input_sha256"]:
    raise ValueError("unlabeled cache row input binding differs")
  proofs = row["full_graph_proofs"]
  if not isinstance(proofs, list) or len(proofs) != 2:
    raise ValueError("unlabeled cache row proof count differs")
  for proof_index, proof in enumerate(proofs):
    _validate_full_graph_proof(proof, label=f"unlabeled cache proof {proof_index}")
  tensors = row["tensors"]
  if not isinstance(tensors, list) or len(tensors) != 6:
    raise ValueError("unlabeled cache row tensor count differs")
  prefix = f"r{expected_ordinal:08d}"
  contracts = {
      f"{prefix}_g{graph_index}_nodes": (
          "float32", [int(proof["face_count"]), INTRINSIC_NODE_INPUT_DIM]
      )
      for graph_index, proof in enumerate(proofs)
  }
  contracts.update(
      {
          f"{prefix}_g{graph_index}_edge_index": (
              "int64", [int(proof["adjacency_count"]), 2]
          )
          for graph_index, proof in enumerate(proofs)
      }
  )
  contracts.update(
      {
          f"{prefix}_g{graph_index}_edge_features": (
              "float32", [int(proof["adjacency_count"]), EDGE_INPUT_DIM]
          )
          for graph_index, proof in enumerate(proofs)
      }
  )
  receipts: dict[str, Mapping[str, Any]] = {}
  for tensor in tensors:
    receipt = _validate_tensor_receipt_payload(tensor, label="unlabeled cache tensor")
    name = str(receipt["name"])
    if name in receipts:
      raise ValueError("unlabeled cache row tensor names differ")
    receipts[name] = receipt
  if set(receipts) != set(contracts):
    raise ValueError("unlabeled cache row tensor names differ")
  for name, (dtype, shape) in contracts.items():
    if receipts[name]["dtype"] != dtype or receipts[name]["shape"] != shape:
      raise ValueError("unlabeled cache row tensor contract differs")
  expected = dict(row)
  observed = expected.pop("row_payload_sha256")
  _sha256_value(observed, label="unlabeled cache row payload sha256")
  if observed != canonical_sha256(expected):
    raise ValueError("unlabeled cache row self hash differs")
  return row


def _validate_label_row(value: Any, *, expected_ordinal: int) -> Mapping[str, Any]:
  row = _exact_keys(
      value,
      {
          "row_ordinal", "input_sha256", "label_commitment_sha256", "tensors",
          "row_payload_sha256",
      },
      label="unlabeled label row",
  )
  if row["row_ordinal"] != expected_ordinal or type(row["row_ordinal"]) is not int:
    raise ValueError("unlabeled label row ordinal differs")
  _sha256_value(row["input_sha256"], label="unlabeled label row input sha256")
  _sha256_value(row["label_commitment_sha256"], label="unlabeled label commitment")
  tensors = row["tensors"]
  if not isinstance(tensors, list) or len(tensors) != 6:
    raise ValueError("unlabeled label row tensor count differs")
  names: set[str] = set()
  for tensor in tensors:
    receipt = _validate_tensor_receipt_payload(tensor, label="unlabeled label tensor")
    if receipt["name"] in names:
      raise ValueError("unlabeled label row tensor names differ")
    names.add(str(receipt["name"]))
  expected = dict(row)
  observed = expected.pop("row_payload_sha256")
  _sha256_value(observed, label="unlabeled label row payload sha256")
  if observed != canonical_sha256(expected):
    raise ValueError("unlabeled label row self hash differs")
  return row


def _validate_shards(
    values: Any, *, rows: Sequence[Mapping[str, Any]], label: str
) -> list[Mapping[str, Any]]:
  if not isinstance(values, list) or not values:
    raise ValueError(f"{label} list differs")
  observed_start = 0
  validated: list[Mapping[str, Any]] = []
  for shard_index, value in enumerate(values):
    shard = _exact_keys(
        value,
        {
            "shard_index", "name", "bytes", "sha256", "row_start", "row_count",
            "rows_payload_sha256",
        },
        label=f"{label} entry",
    )
    if shard["shard_index"] != shard_index or type(shard["shard_index"]) is not int:
      raise ValueError(f"{label} index differs")
    _validate_binding(
        {key: shard[key] for key in ("name", "bytes", "sha256")}, label=f"{label} binding"
    )
    row_start = _actual_int(shard["row_start"], label=f"{label} row start")
    row_count = _actual_int(shard["row_count"], label=f"{label} row count", minimum=1)
    if row_start != observed_start or row_start + row_count > len(rows):
      raise ValueError(f"{label} row coverage differs")
    expected_rows_hash = canonical_sha256(
        [row["row_payload_sha256"] for row in rows[row_start:row_start + row_count]]
    )
    if shard["rows_payload_sha256"] != expected_rows_hash:
      raise ValueError(f"{label} row commitment differs")
    observed_start += row_count
    validated.append(shard)
  if observed_start != len(rows):
    raise ValueError(f"{label} does not cover all rows")
  return validated


def _validate_graph_policy(value: Any) -> Mapping[str, Any]:
  policy = _exact_keys(
      value,
      {
          "selection", "pair_order", "max_faces_per_graph", "max_edges_per_graph",
          "endpoint_annotations",
      },
      label="unlabeled cache graph policy",
  )
  expected = {
      "selection": "all_occ_faces_and_all_two_owner_adjacencies",
      "pair_order": "query_role_a_then_query_role_b",
      "max_faces_per_graph": 2048,
      "max_edges_per_graph": 4096,
      "endpoint_annotations": "forbidden",
  }
  if dict(policy) != expected:
    raise ValueError("unlabeled cache graph policy differs")
  return policy


class LoadedUnlabeledTensorCacheV4:
  """Factory-only loader; model iteration never opens the label sidecar."""

  __slots__ = ("_root", "_manifest", "artifact_sha256")

  def __init__(self, *, root: Path, manifest: Mapping[str, Any], pin: str, _factory_token: object) -> None:
    if _factory_token is not _LOADED_CACHE_FACTORY_TOKEN:
      raise TypeError("loaded unlabeled caches are verifier-factory-only")
    self._root = root
    self._manifest = MappingProxyType(dict(manifest))
    self.artifact_sha256 = pin

  @property
  def row_count(self) -> int:
    return int(self._manifest["row_count"])

  @property
  def input_domain_sha256(self) -> str:
    return str(self._manifest.get("input_domain_sha256") or "")

  def iter_model_inputs(self) -> Iterable[BenchmarkV3UnlabeledStepView]:
    rows = list(self._manifest["rows"])
    for shard in self._manifest["shards"]:
      path = self._root / shard["name"]
      if _binding(path, name=shard["name"]) != {
          key: shard[key] for key in ("name", "bytes", "sha256")
      }:
        raise ValueError("unlabeled input shard changed")
      with np.load(path, allow_pickle=False) as archive:
        selected = rows[shard["row_start"]: shard["row_start"] + shard["row_count"]]
        expected_names = {
            str(receipt["name"])
            for row in selected
            for receipt in row["tensors"]
        }
        if set(archive.files) != expected_names or len(archive.files) != len(expected_names):
          raise ValueError("unlabeled input shard tensor keys differ")
        for row in selected:
          prefix = f"r{row['row_ordinal']:08d}"
          receipts = {str(item["name"]): item for item in row["tensors"]}
          arrays = {name: archive[name].copy() for name in receipts}
          if any(
              not _tensor_matches(receipt, arrays[name])
              for name, receipt in receipts.items()
          ):
            raise ValueError("unlabeled input tensor differs from manifest")
          graphs = []
          for index in range(row["graph_count"]):
            graphs.append(
                UnlabeledIntrinsicBRepGraph(
                    node_features=arrays[f"{prefix}_g{index}_nodes"],
                    edge_index=arrays[f"{prefix}_g{index}_edge_index"],
                    edge_features=arrays[f"{prefix}_g{index}_edge_features"],
                )
            )
          view = _build_cache_owned_unlabeled_step_view(graphs=graphs)
          if view.input_sha256 != row["input_sha256"] or view.receipt_payload() != row["view_receipt"]:
            raise ValueError("unlabeled input row differs from its manifest")
          yield view


def load_unlabeled_tensor_cache_v4(
    cache_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_historical_producer_revision: str | None = None,
) -> LoadedUnlabeledTensorCacheV4:
  root = Path(cache_directory)
  artifact_path = root / "artifact.json"
  if _file_sha256(artifact_path) != expected_artifact_sha256:
    raise ValueError("unlabeled cache artifact pin differs")
  artifact = _strict_json(artifact_path.read_bytes(), label="unlabeled cache artifact")
  _exact_keys(
      artifact,
      {"schema_version", "producer_binding", "manifest", "shards", "artifact_payload_sha256"},
      label="unlabeled cache artifact",
  )
  if artifact.get("schema_version") != CACHE_ARTIFACT_SCHEMA_VERSION:
    raise ValueError("unlabeled cache artifact schema differs")
  artifact_producer = _validate_producer_binding(
      artifact["producer_binding"],
      label="unlabeled cache artifact producer",
      expected_historical_revision=expected_historical_producer_revision,
  )
  expected_self = dict(artifact)
  observed_self = expected_self.pop("artifact_payload_sha256", None)
  if observed_self != canonical_sha256(expected_self):
    raise ValueError("unlabeled cache artifact self hash differs")
  manifest_binding = _validate_binding(
      artifact["manifest"], label="unlabeled cache manifest binding"
  )
  manifest_path = root / str(manifest_binding.get("name"))
  if _binding(manifest_path, name=str(manifest_binding.get("name"))) != dict(manifest_binding):
    raise ValueError("unlabeled cache manifest changed")
  manifest = _strict_json(manifest_path.read_bytes(), label="unlabeled cache manifest")
  _exact_keys(
      manifest,
      {
          "schema_version", "scope", "model_view_schema_version",
          "source_cache_schema_version", "source_cache_artifact_sha256", "producer_binding",
          "full_graph_budget_preflight", "graph_budget_decision",
          "case_selection_policy", "case_count", "graph_policy", "row_count",
          "input_domain_sha256", "rows", "shards",
          "manifest_payload_sha256",
      },
      label="unlabeled cache manifest",
  )
  if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
    raise ValueError("unlabeled cache manifest schema differs")
  if (
      manifest["scope"] != "p0_train_dev_full_step_inputs_only"
      or manifest["model_view_schema_version"] != UNLABELED_VIEW_SCHEMA
      or manifest["source_cache_schema_version"] != SOURCE_CACHE_SCHEMA_VERSION
  ):
    raise ValueError("unlabeled cache manifest contract differs")
  _sha256_value(
      manifest["source_cache_artifact_sha256"], label="unlabeled source cache pin"
  )
  preflight_binding = _validate_preflight_binding(
      manifest["full_graph_budget_preflight"]
  )
  decision_binding = _validate_budget_decision_binding(
      manifest["graph_budget_decision"]
  )
  if decision_binding["case_domain_sha256"] != preflight_binding["case_domain_sha256"]:
    raise ValueError("unlabeled cache decision domain differs")
  selection_policy = manifest["case_selection_policy"]
  if selection_policy not in {
      "all_decision_admitted_cases",
      "preflight_excluded_decision_admitted_cases",
  }:
    raise ValueError("unlabeled cache case selection differs")
  _actual_int(manifest["case_count"], label="unlabeled cache case count", minimum=1)
  manifest_producer = _validate_producer_binding(
      manifest["producer_binding"],
      label="unlabeled cache manifest producer",
      expected_historical_revision=expected_historical_producer_revision,
  )
  if dict(manifest_producer) != dict(artifact_producer):
    raise ValueError("unlabeled cache producer binding differs")
  _validate_graph_policy(manifest["graph_policy"])
  expected_manifest = dict(manifest)
  observed_manifest = expected_manifest.pop("manifest_payload_sha256", None)
  if observed_manifest != canonical_sha256(expected_manifest):
    raise ValueError("unlabeled cache manifest self hash differs")
  rows_value = manifest["rows"]
  if not isinstance(rows_value, list):
    raise ValueError("unlabeled cache rows differ")
  row_count = _actual_int(manifest["row_count"], label="unlabeled cache row count", minimum=1)
  if len(rows_value) != row_count:
    raise ValueError("unlabeled cache row count differs")
  rows = [_validate_safe_row(row, expected_ordinal=index) for index, row in enumerate(rows_value)]
  _sha256_value(manifest["input_domain_sha256"], label="unlabeled input domain sha256")
  if manifest["input_domain_sha256"] != _input_domain_sha256(
      [row["input_sha256"] for row in rows],
      preflight_binding=preflight_binding,
      decision_binding=decision_binding,
      case_selection_policy=selection_policy,
  ):
    raise ValueError("unlabeled cache input domain differs")
  shards = _validate_shards(manifest["shards"], rows=rows, label="unlabeled cache shards")
  artifact_shards = artifact["shards"]
  if not isinstance(artifact_shards, list) or len(artifact_shards) != len(shards):
    raise ValueError("unlabeled cache artifact shard count differs")
  for artifact_binding, shard in zip(artifact_shards, shards, strict=True):
    validated_binding = _validate_binding(
        artifact_binding, label="unlabeled cache artifact shard binding"
    )
    expected_binding = {key: shard[key] for key in ("name", "bytes", "sha256")}
    if dict(validated_binding) != expected_binding:
      raise ValueError("unlabeled cache artifact shard binding differs")
  return LoadedUnlabeledTensorCacheV4(
      root=root,
      manifest=manifest,
      pin=expected_artifact_sha256,
      _factory_token=_LOADED_CACHE_FACTORY_TOKEN,
  )


class LoadedUnlabeledTrainingLabelsV4:
  """Independent verifier for the sibling gold artifact."""

  __slots__ = ("_root", "_manifest", "artifact_sha256")

  def __init__(self, *, root: Path, manifest: Mapping[str, Any], pin: str, _factory_token: object) -> None:
    if _factory_token is not _LOADED_LABEL_FACTORY_TOKEN:
      raise TypeError("loaded unlabeled labels are verifier-factory-only")
    self._root = root
    self._manifest = MappingProxyType(dict(manifest))
    self.artifact_sha256 = pin

  @property
  def row_count(self) -> int:
    return int(self._manifest["row_count"])

  @property
  def input_domain_sha256(self) -> str:
    return str(self._manifest["input_domain_sha256"])

  def iter_bound_labels(
      self,
  ) -> Iterable[tuple[str, AuthenticatedUnlabeledTrainingLabel]]:
    rows = list(self._manifest["rows"])
    for shard in self._manifest["shards"]:
      path = self._root / shard["name"]
      if _binding(path, name=shard["name"]) != {
          key: shard[key] for key in ("name", "bytes", "sha256")
      }:
        raise ValueError("unlabeled training-label shard changed")
      with np.load(path, allow_pickle=False) as archive:
        selected = rows[shard["row_start"]: shard["row_start"] + shard["row_count"]]
        for row in selected:
          prefix = f"r{row['row_ordinal']:08d}"
          arrays = {
              f"{prefix}_class": archive[f"{prefix}_class"].copy(),
              f"{prefix}_translation": archive[f"{prefix}_translation"].copy(),
              f"{prefix}_rotation": archive[f"{prefix}_rotation"].copy(),
              f"{prefix}_mask": archive[f"{prefix}_mask"].copy(),
              f"{prefix}_endpoint_a": archive[f"{prefix}_endpoint_a"].copy(),
              f"{prefix}_endpoint_b": archive[f"{prefix}_endpoint_b"].copy(),
          }
          receipts = {item["name"]: item for item in row["tensors"]}
          if set(receipts) != set(arrays) or any(
              not _tensor_matches(receipts[name], array) for name, array in arrays.items()
          ):
            raise ValueError("unlabeled training-label tensor differs from manifest")
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
          if label.commitment_sha256 != row["label_commitment_sha256"]:
            raise ValueError("unlabeled training label differs from commitment")
          yield str(row["input_sha256"]), label


def load_unlabeled_training_labels_v4(
    label_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_input_domain_sha256: str,
    expected_historical_producer_revision: str | None = None,
) -> LoadedUnlabeledTrainingLabelsV4:
  root = Path(label_directory)
  artifact_path = root / "artifact.json"
  if _file_sha256(artifact_path) != expected_artifact_sha256:
    raise ValueError("unlabeled label artifact pin differs")
  artifact = _strict_json(artifact_path.read_bytes(), label="unlabeled label artifact")
  _exact_keys(
      artifact,
      {
          "schema_version", "producer_binding", "manifest", "audit", "shards",
          "artifact_payload_sha256",
      },
      label="unlabeled label artifact",
  )
  if artifact.get("schema_version") != LABEL_ARTIFACT_SCHEMA_VERSION:
    raise ValueError("unlabeled label artifact schema differs")
  artifact_producer = _validate_producer_binding(
      artifact["producer_binding"],
      label="unlabeled label artifact producer",
      expected_historical_revision=expected_historical_producer_revision,
  )
  expected_self = dict(artifact)
  observed_self = expected_self.pop("artifact_payload_sha256", None)
  if observed_self != canonical_sha256(expected_self):
    raise ValueError("unlabeled label artifact self hash differs")
  manifest_binding = _validate_binding(
      artifact["manifest"], label="unlabeled label manifest binding"
  )
  manifest_path = root / str(manifest_binding.get("name"))
  if _binding(manifest_path, name=str(manifest_binding.get("name"))) != dict(manifest_binding):
    raise ValueError("unlabeled label manifest changed")
  manifest = _strict_json(manifest_path.read_bytes(), label="unlabeled label manifest")
  _exact_keys(
      manifest,
      {
          "schema_version", "scope", "source_cache_artifact_sha256", "producer_binding",
          "full_graph_budget_preflight", "graph_budget_decision",
          "case_selection_policy", "case_count", "row_count", "input_domain_sha256",
          "label_commitment_ledger_sha256",
          "rows", "shards", "manifest_payload_sha256",
      },
      label="unlabeled label manifest",
  )
  if manifest.get("schema_version") != LABEL_SCHEMA_VERSION:
    raise ValueError("unlabeled label manifest schema differs")
  if manifest["scope"] != "factory_only_training_supervision":
    raise ValueError("unlabeled label manifest scope differs")
  _sha256_value(
      manifest["source_cache_artifact_sha256"], label="unlabeled label source cache pin"
  )
  preflight_binding = _validate_preflight_binding(
      manifest["full_graph_budget_preflight"]
  )
  decision_binding = _validate_budget_decision_binding(
      manifest["graph_budget_decision"]
  )
  if decision_binding["case_domain_sha256"] != preflight_binding["case_domain_sha256"]:
    raise ValueError("unlabeled label decision domain differs")
  selection_policy = manifest["case_selection_policy"]
  if selection_policy not in {
      "all_decision_admitted_cases",
      "preflight_excluded_decision_admitted_cases",
  }:
    raise ValueError("unlabeled label case selection differs")
  _actual_int(manifest["case_count"], label="unlabeled label case count", minimum=1)
  manifest_producer = _validate_producer_binding(
      manifest["producer_binding"],
      label="unlabeled label manifest producer",
      expected_historical_revision=expected_historical_producer_revision,
  )
  if dict(manifest_producer) != dict(artifact_producer):
    raise ValueError("unlabeled label producer binding differs")
  expected_manifest = dict(manifest)
  observed_manifest = expected_manifest.pop("manifest_payload_sha256", None)
  if observed_manifest != canonical_sha256(expected_manifest):
    raise ValueError("unlabeled label manifest self hash differs")
  if manifest.get("input_domain_sha256") != expected_input_domain_sha256:
    raise ValueError("unlabeled label input-domain commitment differs")
  rows_value = manifest["rows"]
  if not isinstance(rows_value, list):
    raise ValueError("unlabeled label rows differ")
  row_count = _actual_int(manifest["row_count"], label="unlabeled label row count", minimum=1)
  if len(rows_value) != row_count:
    raise ValueError("unlabeled label row count differs")
  rows = [_validate_label_row(row, expected_ordinal=index) for index, row in enumerate(rows_value)]
  _sha256_value(manifest["input_domain_sha256"], label="unlabeled label input domain sha256")
  if manifest["input_domain_sha256"] != _input_domain_sha256(
      [row["input_sha256"] for row in rows],
      preflight_binding=preflight_binding,
      decision_binding=decision_binding,
      case_selection_policy=selection_policy,
  ):
    raise ValueError("unlabeled label input domain differs")
  _sha256_value(
      manifest["label_commitment_ledger_sha256"], label="unlabeled label ledger sha256"
  )
  if manifest["label_commitment_ledger_sha256"] != canonical_sha256(
      [row["label_commitment_sha256"] for row in rows]
  ):
    raise ValueError("unlabeled label commitment ledger differs")
  shards = _validate_shards(manifest["shards"], rows=rows, label="unlabeled label shards")
  artifact_shards = artifact["shards"]
  if not isinstance(artifact_shards, list) or len(artifact_shards) != len(shards):
    raise ValueError("unlabeled label artifact shard count differs")
  for artifact_binding, shard in zip(artifact_shards, shards, strict=True):
    validated_binding = _validate_binding(
        artifact_binding, label="unlabeled label artifact shard binding"
    )
    expected_binding = {key: shard[key] for key in ("name", "bytes", "sha256")}
    if dict(validated_binding) != expected_binding:
      raise ValueError("unlabeled label artifact shard binding differs")
  audit_binding = _validate_binding(
      artifact["audit"], label="unlabeled label audit binding"
  )
  audit_path = root / str(audit_binding["name"])
  if _binding(audit_path, name=str(audit_binding["name"])) != dict(audit_binding):
    raise ValueError("unlabeled label audit changed")
  audit = _strict_json(audit_path.read_bytes(), label="unlabeled label audit")
  _exact_keys(
      audit,
      {
          "schema_version", "producer_binding", "row_count", "source_cache_artifact_sha256",
          "full_graph_budget_preflight", "graph_budget_decision",
          "case_selection_policy", "case_count", "input_manifest",
          "training_label_manifest", "input_domain_sha256",
          "all_graphs_complete_component", "max_face_count", "max_adjacency_count",
          "forbidden_model_fields", "final_test_touched", "audit_payload_sha256",
      },
      label="unlabeled label audit",
  )
  if audit["schema_version"] != AUDIT_SCHEMA_VERSION:
    raise ValueError("unlabeled label audit schema differs")
  expected_audit = dict(audit)
  observed_audit = expected_audit.pop("audit_payload_sha256")
  _sha256_value(observed_audit, label="unlabeled label audit payload sha256")
  if observed_audit != canonical_sha256(expected_audit):
    raise ValueError("unlabeled label audit self hash differs")
  audit_producer = _validate_producer_binding(
      audit["producer_binding"],
      label="unlabeled label audit producer",
      expected_historical_revision=expected_historical_producer_revision,
  )
  if dict(audit_producer) != dict(artifact_producer):
    raise ValueError("unlabeled label audit producer binding differs")
  if (
      audit["row_count"] != row_count
      or type(audit["row_count"]) is not int
      or audit["source_cache_artifact_sha256"] != manifest["source_cache_artifact_sha256"]
      or audit["full_graph_budget_preflight"] != manifest["full_graph_budget_preflight"]
      or audit["graph_budget_decision"] != manifest["graph_budget_decision"]
      or audit["case_selection_policy"] != manifest["case_selection_policy"]
      or audit["case_count"] != manifest["case_count"]
      or audit["input_domain_sha256"] != manifest["input_domain_sha256"]
      or audit["all_graphs_complete_component"] is not True
      or type(audit["max_face_count"]) is not int
      or audit["max_face_count"] < 1
      or type(audit["max_adjacency_count"]) is not int
      or audit["max_adjacency_count"] < 0
      or audit["forbidden_model_fields"] != []
      or audit["final_test_touched"] is not False
  ):
    raise ValueError("unlabeled label audit verdict differs")
  input_manifest_binding = _validate_binding(
      audit["input_manifest"], label="unlabeled audit input manifest binding"
  )
  label_manifest_binding = _validate_binding(
      audit["training_label_manifest"], label="unlabeled audit label manifest binding"
  )
  if dict(label_manifest_binding) != dict(manifest_binding):
    raise ValueError("unlabeled audit label manifest binding differs")
  if input_manifest_binding["name"] != "manifest.json":
    raise ValueError("unlabeled audit input manifest name differs")
  return LoadedUnlabeledTrainingLabelsV4(
      root=root,
      manifest=manifest,
      pin=expected_artifact_sha256,
      _factory_token=_LOADED_LABEL_FACTORY_TOKEN,
  )
