"""Authenticated, leakage-safe B-Rep graph views for benchmark-v2.

Version 2 is deliberately a separate schema and API from
``benchmark_v2_model_view`` v1.  Model-visible values are restricted to
intrinsic B-Rep geometry, local topology, local integer references and
program-local supervision.  Provenance identities and filesystem bindings are
kept in an ephemeral authenticated wrapper implemented below.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import weakref

from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    capture_file_artifact,
    reverify_captured_file_artifact,
)


SCHEMA_VERSION = "benchmark_v2_model_view.v2"
GRAPH_SCHEMA_VERSION = "benchmark_v2_intrinsic_brep_graph.v2"

NODE_FEATURE_NAMES = (
    "area_fraction",
    "perimeter_over_sqrt_total_area",
    "mean_curvature_times_sqrt_total_area",
    "gaussian_curvature_times_total_area",
    "boundary_loop_count",
    "boundary_edge_count",
)
EDGE_FEATURE_NAMES = (
    "shared_edge_length_over_sqrt_total_area",
    "dihedral_cos",
    "dihedral_sin_abs",
)
_SURFACE_TYPES = frozenset(
    {"plane", "cylinder", "cone", "sphere", "torus", "spline", "other"}
)
_PROGRAM_TYPES = frozenset(
    {"coaxial", "insert", "link", "planar", "support"}
)
_ASSIGNMENT_SCHEMA = "neurocad_formal_model_development_assignment.v3"
_FREEZE_MANIFEST_SCHEMA = "neurocad_formal_model_development_freeze.v3"
_LEAKAGE_AUDIT_SCHEMA = "neurocad_formal_model_development_leakage_audit.v3"
_AUTHORITY_POOL_SCHEMA = "neurocad_authority_ready_pool_700_merged.v1"
_STAGE_RECEIPT_SCHEMA = "neurocad_formal_authority_stage_receipt.v1"
_PROJECTION_SCHEMA = "benchmark_v2_development_authority_projection.v2"
_PROJECTION_RECEIPT_SCHEMA = (
    "benchmark_v2_development_authority_projection_receipt.v2"
)
_PROGRAM_CATALOG_SCHEMA = "benchmark_v2_train_program_catalog.v1"
_PROJECTION_PRODUCER_IDENTITY = (
    "24655dfca896c90e14ab3150eac173db66ffc20b2f84540b065266f18f812c95"
)
_DEVELOPMENT_TRUST_ROOT = MappingProxyType(
    {
        "schema_version": "benchmark_v2_development_trust_root.v1",
        "freeze_manifest_schema": _FREEZE_MANIFEST_SCHEMA,
        "freeze_manifest_file_sha256": (
            "b95c9a62c6c609faf8884b4b46fac769830c8aea3f7428ed4ba984ababdcd933"
        ),
        "freeze_manifest_payload_sha256": (
            "a6c1c5f0978dc66a714a3d2891c5be9b676145faf9f093f740d5d30116001144"
        ),
        "policy_payload_sha256": (
            "04fd0ab9c0315b74020321911c89da78cc8f282676512d44ad4f41b54b4ba99c"
        ),
        "authority_pool_file_sha256": (
            "a7d5dea57d7ae6bfb0776f13eeb6d2771191aa2376e8461b1526072df00f4d79"
        ),
        "authority_pool_merge_receipt_file_sha256": (
            "f2067534262d4422dd4bf0ea3f26f9dca2758f7aea4de172e8ec458f3c38f769"
        ),
        "audit_file_sha256": (
            "31814f40359a23f1656f5c967cf27da410a045dfcaadf340225ddd08583e31c9"
        ),
        "train_assignment_file_sha256": (
            "f82d6f90c76b17c4b34294f8a0eb07fb4bffd34300eccda0a1837b0fa28deb82"
        ),
        "dev_assignment_file_sha256": (
            "ba630f62e59c9af5a818e96fabc3f36b98614d221d17c881414b20cca944bef5"
        ),
        "freeze_producer_code_sha256": (
            "4872d2446500f8cc3d4687f95ba3e85bdb5c7e524a3d1fcd8649c89ff982cd39"
        ),
        "family_split_builder_code_sha256": (
            "9f4151a5624d426516cd5d479ac54da5723413203afaa57cf0810c63f26d088c"
        ),
        "immutable_writer_code_sha256": (
            "fb17660d291165ccf2b667e20cea5da7db7573128424472dc6be9e03dd27f89d"
        ),
        "projection_schema": _PROJECTION_SCHEMA,
        "projection_producer_identity_sha256": _PROJECTION_PRODUCER_IDENTITY,
        "status": "development_unsealed_nonpublication",
    }
)
# Filled only for the independently produced 600-row projection.  Exact file
# roots make a self-consistent counterfeit projection insufficient.
_TRUSTED_PROJECTION_FILE_SHA256 = (
    "2ca8cb6e13e2b176267ac1f81e13fcbc474f791e89dadb282e1ff6f4a6370966"
)
_TRUSTED_PROJECTION_RECEIPT_FILE_SHA256 = (
    "0055f20076cf6cef09b4c6e7472b9afec1e0a079470e33622264b560ea098c35"
)
_FACTORY_TOKEN = object()
# Frozen before model training.  Every authority-backed part graph uses the
# same endpoint-centred policy; no case identity, model output or OCC face
# enumeration order participates in selecting a shell.
LOCAL_INTERFACE_SUBGRAPH_POLICY = MappingProxyType(
    {
        "schema_version": "benchmark_v2_local_interface_subgraph_policy.v1",
        "center": "authority_mapped_endpoint_faces",
        "distance": "unweighted_brep_face_adjacency_hops",
        "max_hop_radius": 4,
        "shell_admission": "complete_shell_or_stop",
        "partial_shell_tie_break": "forbidden",
        "one_hop_over_budget": "fail_closed",
        "component_exhaustion": "complete_component",
        "formal_graph_budget": {
            "max_graphs": 2,
            "max_faces_per_graph": 1024,
            "max_edges_per_graph": 4096,
        },
    }
)
# OCCT's STEP configuration and reader/transfer machinery share process-global
# Interface_Static state.  Keep the complete reader transaction serialized;
# a per-cache lock would not protect concurrent caches or direct loader calls.
_OCCT_STEP_READ_LOCK = threading.RLock()
_NORMALIZATION_METADATA = {
    "schema_version": "benchmark_v2_graph_normalization.v2",
    "rounding_decimal_places": 12,
    "length_scale": "sqrt(sum_face_area)",
    "area_scale": "sum_face_area",
    "node_feature_names": list(NODE_FEATURE_NAMES),
    "node_feature_mask": "true_means_authority_replay_computed_value.v1",
    "edge_feature_names": list(EDGE_FEATURE_NAMES),
    "curvature_sample": "deterministic_trimmed_uv_midpoint.v1",
    "dihedral_orientation": "cos_and_absolute_sine.v1",
    "residual_translation": "program_local_millimetres.v1",
    "residual_rotation": "program_local_so3_row_major.v1",
    "node_canonicalization": "intrinsic_signature_wl_adjacency_refinement.v1",
    "node_residual_tie_policy": (
        "occ_enumeration_only_for_intrinsically_indistinguishable_automorphisms.v1"
    ),
    "graph_scope": "authority_endpoint_local_complete_bfs_shells.v1",
    "local_interface_subgraph_policy": dict(LOCAL_INTERFACE_SUBGRAPH_POLICY),
    "program_catalog": "authenticated_train_authority_discrete_templates.v1",
    "residual_target_tie_policy": (
        "exact_selected_authority_row_with_shared_catalog_template.v1"
    ),
}

_SAFE_CATALOG_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CATALOG_DESCRIPTOR_KEYS = (
    "relation_hint",
    "contact_type",
    "surface_type_a",
    "surface_type_b",
)


def _is_sha256(value: Any) -> bool:
  return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class ModelViewV2LeakageError(ValueError):
  """A purported v2 view contains a channel outside its exact allowlist."""


@dataclass(frozen=True, slots=True)
class ModelViewV2Sanitization:
  model_view: dict[str, Any]
  canonical_json: str
  sha256: str


@dataclass(frozen=True, slots=True)
class _CapturedJson:
  file: CapturedFileArtifact
  payload: Any


@dataclass(frozen=True, slots=True)
class _CaseAuthorityBinding:
  authority_production_split: str
  model_assignment_split: str
  source_ordinal: int
  case_id: str
  binding_sha256: str
  face_payload: Mapping[str, Any]
  contact_payload: Mapping[str, Any]
  semantic_payload: Mapping[str, Any]
  private_gold_payload: Mapping[str, Any]
  library_capture: CapturedFileArtifact
  captures: tuple[CapturedFileArtifact, ...]


@dataclass(frozen=True, slots=True)
class _GraphCapabilityState:
  model_view_bytes: bytes
  receipt_bytes: bytes
  binding_sha256: str
  catalog_payload_bytes: bytes
  captures: tuple[CapturedFileArtifact, ...]


@dataclass(frozen=True, slots=True)
class _GraphWitnessAuditInput:
  graph_index: int
  step_capture: CapturedFileArtifact
  endpoint_raw_indices: tuple[int, ...]
  endpoint_face_signatures: tuple[str, ...]
  budget: "GraphSizeBudget"


@dataclass(frozen=True, slots=True)
class _GraphWitnessState:
  """Factory-only topology fact, deliberately separate from public receipts."""

  witness_payload_bytes: bytes
  witness_payload_sha256: str
  audit_inputs: tuple[_GraphWitnessAuditInput, ...]


@dataclass(frozen=True, slots=True)
class _DevelopmentIndexState:
  bindings: Mapping[str, tuple[_CaseAuthorityBinding, ...]]
  catalog_payload_bytes: bytes
  root_captures: tuple[CapturedFileArtifact, ...]


@dataclass(frozen=True, slots=True)
class _ModelViewState:
  model_view_bytes: bytes
  graph_capability: Any
  authority_captures: tuple[CapturedFileArtifact, ...]


def _captured_step_key(
    captured: CapturedFileArtifact,
) -> tuple[str, int, str]:
  suffix = captured.resolved_path.suffix.lower()
  if suffix not in {".step", ".stp"}:
    raise ValueError("receipt-bound STEP must use a STEP/STP suffix")
  if hashlib.sha256(captured.raw_bytes).hexdigest() != captured.sha256:
    raise ValueError("receipt-bound STEP raw bytes differ from captured hash")
  if len(captured.raw_bytes) != captured.byte_count:
    raise ValueError("receipt-bound STEP raw byte count differs from capture")
  return captured.sha256, captured.byte_count, suffix


def _load_captured_step_shape(captured: CapturedFileArtifact) -> Any:
  """Parse the exact captured STEP bytes without exposing a filesystem path."""

  key = _captured_step_key(captured)
  import cadquery as cq  # type: ignore
  from OCP.IFSelect import IFSelect_RetDone  # type: ignore
  from OCP.Interface import Interface_Static  # type: ignore
  from OCP.STEPControl import STEPControl_Reader  # type: ignore

  # Match CadQuery importStep's target-unit and root-transfer behavior.  Reader
  # construction initializes the Interface_Static registry, so it must precede
  # the checked unit assignment while remaining inside the same global lock.
  # The stream name is diagnostic only: it is deliberately not a usable path.
  with _OCCT_STEP_READ_LOCK:
    reader = STEPControl_Reader()
    if Interface_Static.SetCVal_s("xstep.cascade.unit", "MM") is not True:
      raise ValueError("receipt-bound STEP target unit could not be set")
    stream_name = f"captured-{key[0]}{key[2]}"
    with io.BytesIO(captured.raw_bytes) as step_stream:
      status = reader.ReadStream(stream_name, step_stream)
      if status != IFSelect_RetDone:
        raise ValueError("receipt-bound STEP stream could not be loaded")
      root_count = int(reader.NbRootsForTransfer())
      if root_count <= 0:
        raise ValueError("receipt-bound STEP stream has no transferable roots")
      for root_index in range(root_count):
        if reader.TransferRoot(root_index + 1) is not True:
          raise ValueError(
              "receipt-bound STEP root "
              f"{root_index + 1} could not be transferred"
          )
      shape_count = int(reader.NbShapes())
      if shape_count <= 0:
        raise ValueError("receipt-bound STEP stream produced no bodies")
      bodies: list[Any] = []
      for shape_index in range(shape_count):
        wrapped = reader.Shape(shape_index + 1)
        if wrapped.IsNull():
          raise ValueError("receipt-bound STEP stream produced a null body")
        bodies.append(cq.Shape.cast(wrapped))
    if len(bodies) != shape_count:
      raise ValueError("receipt-bound STEP body count changed during wrapping")
    shape = cq.Workplane("XY").newObject(bodies).val()
    if shape is None:
      raise ValueError("receipt-bound STEP stream produced no CadQuery shape")
  return shape


class _CapturedStepStreamCache:
  """Per-materialization cache of captured geometry and endpoint graphs."""

  __slots__ = ("_endpoint_graphs", "_shapes", "_topologies")

  def __init__(self) -> None:
    self._shapes: dict[tuple[str, int, str], Any] = {}
    self._topologies: dict[tuple[str, int, str], Any] = {}
    self._endpoint_graphs: dict[tuple[Any, ...], Any] = {}

  def load_step_shape(self, captured: CapturedFileArtifact) -> Any:
    key = _captured_step_key(captured)
    existing = self._shapes.get(key)
    if existing is not None:
      return existing
    shape = _load_captured_step_shape(captured)
    self._shapes[key] = shape
    return shape

  def load_step_topology(self, captured: CapturedFileArtifact) -> Any:
    """Capture OCC topology once for every receipt-bound STEP byte stream."""

    key = _captured_step_key(captured)
    existing = self._topologies.get(key)
    if existing is not None:
      return existing
    topology = _captured_brep_topology(self.load_step_shape(captured))
    self._topologies[key] = topology
    return topology

  def derive_private_graph(
      self,
      captured: CapturedFileArtifact,
      *,
      graph_index: int,
      endpoint_raw_indices: Sequence[int],
      endpoint_face_signatures: Sequence[str],
      budget: "GraphSizeBudget",
  ) -> tuple[dict[str, Any], dict[int, int], tuple[str, ...], dict[str, Any]]:
    """Reuse an immutable endpoint graph while rebuilding its indexed witness."""

    budget = _validated_budget(budget)
    raw_indices = tuple(endpoint_raw_indices)
    expected_signatures = tuple(endpoint_face_signatures)
    topology = self.load_step_topology(captured)
    faces, face_edges, edge_groups, imported_signatures = topology
    cache_key = (
        _captured_step_key(captured),
        raw_indices,
        expected_signatures,
        budget.max_faces_per_graph,
        budget.max_edges_per_graph,
        _canonical_sha256(dict(LOCAL_INTERFACE_SUBGRAPH_POLICY)),
    )
    cached = self._endpoint_graphs.get(cache_key)
    if cached is None:
      if any(
          type(index) is not int
          or index < 0
          or index >= len(imported_signatures)
          for index in raw_indices
      ):
        raise ValueError("mapped OCC face index is outside re-opened STEP")
      observed = tuple(imported_signatures[index] for index in raw_indices)
      if observed != expected_signatures:
        raise ValueError(
            "re-opened STEP face signatures differ from face authority"
        )
      selected, coverage = _endpoint_local_face_subset(
          face_count=len(faces),
          edge_groups=edge_groups,
          endpoint_raw_indices=raw_indices,
          budget=budget,
      )
      graph, remap = _intrinsic_graph_from_face_subset(
          faces,
          face_edges,
          edge_groups,
          selected,
          budget=budget,
      )
      if any(index not in remap for index in raw_indices):
        raise ValueError("endpoint face was not preserved by local graph policy")
      cached = (
          copy.deepcopy(graph),
          dict(remap),
          tuple(imported_signatures),
          tuple(selected),
          copy.deepcopy(coverage),
      )
      self._endpoint_graphs[cache_key] = cached
    graph = copy.deepcopy(cached[0])
    remap = dict(cached[1])
    signatures = tuple(cached[2])
    selected = tuple(cached[3])
    coverage = copy.deepcopy(cached[4])
    witness = _private_graph_witness(
        graph_index=graph_index,
        step_capture=captured,
        endpoint_raw_indices=raw_indices,
        endpoint_face_signatures=expected_signatures,
        imported_signatures=signatures,
        edge_groups=edge_groups,
        selected_raw_indices=selected,
        graph=graph,
        coverage=coverage,
        budget=budget,
    )
    return graph, remap, signatures, witness


def _private_state_registry() -> tuple[Any, Any]:
  """Return same-process, thread-safe capability state accessors.

  The registry lock protects every WeakKeyDictionary operation.  It does not
  turn a capability into a cross-process token: capabilities remain
  deliberately unpicklable and must be authenticated again in another
  process.
  """

  states: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
  lock = threading.RLock()

  def register(instance: Any, state: Any) -> None:
    with lock:
      if instance in states:
        raise TypeError("authenticated capability state is already registered")
      states[instance] = state

  def get(instance: Any, expected_type: type[Any]) -> Any:
    with lock:
      try:
        state = states[instance]
      except KeyError as error:
        raise TypeError(
            "authenticated capability lacks factory-owned state"
        ) from error
    if not isinstance(state, expected_type):
      raise TypeError("authenticated capability state type differs")
    return state

  return register, get


_register_private_state, _get_private_state = _private_state_registry()
_register_private_witness, _get_private_witness = _private_state_registry()


@dataclass(frozen=True, slots=True)
class GraphSizeBudget:
  """Hard limits applied before any graph is admitted to a model view."""

  max_graphs: int = 2
  max_faces_per_graph: int = 1024
  max_edges_per_graph: int = 4096
  max_examples: int = 4096
  max_program_candidates_per_example: int = 512

  def to_dict(self) -> dict[str, int]:
    return {
        "max_graphs": self.max_graphs,
        "max_faces_per_graph": self.max_faces_per_graph,
        "max_edges_per_graph": self.max_edges_per_graph,
        "max_examples": self.max_examples,
        "max_program_candidates_per_example": (
            self.max_program_candidates_per_example
        ),
    }


DEFAULT_GRAPH_SIZE_BUDGET = GraphSizeBudget()
HARD_MAX_GRAPH_SIZE_BUDGET = GraphSizeBudget(
    max_graphs=2,
    max_faces_per_graph=1024,
    max_edges_per_graph=4096,
    max_examples=4096,
    max_program_candidates_per_example=512,
)


def _validated_budget(budget: Any) -> GraphSizeBudget:
  """Accept caller limits only when every field is at or below policy."""

  if not isinstance(budget, GraphSizeBudget):
    raise TypeError("budget must be a GraphSizeBudget")
  values = budget.to_dict()
  hard = HARD_MAX_GRAPH_SIZE_BUDGET.to_dict()
  if any(type(value) is not int or value < 0 for value in values.values()):
    raise ValueError("graph budget fields must be non-negative actual integers")
  if any(values[key] > hard[key] for key in hard):
    raise ValueError("graph budget may only lower module hard limits")
  return budget


def _validated_formal_structural_budget(budget: Any) -> GraphSizeBudget:
  """Require one structural graph protocol for authority-backed model views."""

  budget = _validated_budget(budget)
  expected = LOCAL_INTERFACE_SUBGRAPH_POLICY["formal_graph_budget"]
  observed = {
      "max_graphs": budget.max_graphs,
      "max_faces_per_graph": budget.max_faces_per_graph,
      "max_edges_per_graph": budget.max_edges_per_graph,
  }
  if observed != expected:
    raise ValueError("authority-backed graph structural budget differs from policy")
  return budget


def _finite(value: Any, *, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{label} must be numeric")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{label} must be finite")
  result = round(result, 12)
  return 0.0 if result == 0.0 else result


def _surface_bucket(face: Any) -> str:
  try:
    name = str(face.geomType()).strip().lower()
  except Exception:
    return "other"
  for token, bucket in (
      ("plane", "plane"),
      ("cylinder", "cylinder"),
      ("cone", "cone"),
      ("sphere", "sphere"),
      ("torus", "torus"),
      ("spline", "spline"),
      ("bezier", "spline"),
  ):
    if token in name:
      return bucket
  return "other"


def _vec3(value: Any) -> tuple[float, float, float]:
  if hasattr(value, "toTuple"):
    raw = value.toTuple()
  else:
    raw = (value.x, value.y, value.z)
  return tuple(float(component) for component in raw)


def _unit(value: Any) -> tuple[float, float, float]:
  x, y, z = _vec3(value)
  length = math.sqrt(x * x + y * y + z * z)
  if not math.isfinite(length) or length <= 1e-15:
    raise ValueError("normal is degenerate")
  return x / length, y / length, z / length


def _face_curvatures(face: Any) -> tuple[float, float, bool]:
  """Sample intrinsic mean/Gaussian curvature at a deterministic UV point."""

  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomLProp import GeomLProp_SLProps  # type: ignore

    adaptor = BRepAdaptor_Surface(face.wrapped)
    bounds = (
        float(adaptor.FirstUParameter()),
        float(adaptor.LastUParameter()),
        float(adaptor.FirstVParameter()),
        float(adaptor.LastVParameter()),
    )
    if not all(math.isfinite(value) for value in bounds):
      u, v = face.paramAt(face.Center())
      u, v = float(u), float(v)
    else:
      u = 0.5 * (bounds[0] + bounds[1])
      v = 0.5 * (bounds[2] + bounds[3])
    properties = GeomLProp_SLProps(
        adaptor.Surface().Surface(), u, v, 2, 1e-9
    )
    if not properties.IsCurvatureDefined():
      return 0.0, 0.0, False
    return float(properties.MeanCurvature()), float(
        properties.GaussianCurvature()
    ), True
  except Exception:
    # Zero is only a storage placeholder; the explicit mask prevents it from
    # being mistaken for a measured planar curvature.
    return 0.0, 0.0, False


def _face_normal_at_edge(face: Any, edge: Any) -> tuple[float, float, float]:
  point = edge.positionAt(0.5)
  try:
    return _unit(face.normalAt(point))
  except Exception:
    return _unit(face.normalAt())


def _occ_shapes_same(left: Any, right: Any) -> bool:
  """Compare OCC topology identity without treating hashCode as identity.

  OCC explicitly documents hash codes as bucket keys.  ``IsSame`` and
  ``IsEqual`` are the topology predicates; a collision must therefore remain
  two independent edge groups.
  """

  compared = False
  for left_shape, right_shape in (
      (getattr(left, "wrapped", left), getattr(right, "wrapped", right)),
      (left, right),
  ):
    for method_name in ("IsSame", "IsEqual", "isSame", "isEqual"):
      method = getattr(left_shape, method_name, None)
      if not callable(method):
        continue
      compared = True
      try:
        if bool(method(right_shape)):
          return True
      except (TypeError, RuntimeError):
        continue
  if not compared:
    raise TypeError("OCC edge lacks IsSame/IsEqual topology identity")
  return False


def _group_occ_edge_owners(
    face_edges: Sequence[Sequence[Any]],
) -> list[list[tuple[int, Any]]]:
  """Group shared edges using hash buckets followed by exact OCC identity."""

  buckets: dict[int, list[list[tuple[int, Any]]]] = {}
  for face_index, edges in enumerate(face_edges):
    for edge in edges:
      bucket = buckets.setdefault(int(edge.hashCode()), [])
      matching: list[tuple[int, Any]] | None = None
      for identity_group in bucket:
        if _occ_shapes_same(edge, identity_group[0][1]):
          matching = identity_group
          break
      if matching is None:
        matching = []
        bucket.append(matching)
      if all(owner_index != face_index for owner_index, _ in matching):
        matching.append((face_index, edge))
  return [group for bucket in buckets.values() for group in bucket]


def _refined_intrinsic_face_order(
    nodes: Sequence[Mapping[str, Any]],
    edge_groups: Sequence[Sequence[tuple[int, Any]]],
    *,
    length_scale: float,
) -> list[int]:
  """Refine intrinsic node signatures with local B-Rep adjacency.

  Remaining exact structural automorphisms intentionally fall back to OCC
  enumeration.  Such nodes are indistinguishable in the model-visible graph;
  the residual tie policy is recorded in normalization metadata.
  """

  base = [
      (
          str(node["surface_type"]),
          tuple(node["features"]),
          tuple(node["feature_mask"]),
      )
      for node in nodes
  ]
  adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
  for group in edge_groups:
    if len(group) != 2:
      continue
    left, right = group[0][0], group[1][0]
    if left == right:
      continue
    normalized_length = _finite(
        float(group[0][1].Length()) / length_scale,
        label="canonical shared edge length",
    )
    adjacency[left].append((right, normalized_length))
    adjacency[right].append((left, normalized_length))

  def ranks(keys: Sequence[Any]) -> list[int]:
    ordered = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    return [ordered[key] for key in keys]

  colors = ranks(base)
  refined_keys: list[Any] = list(base)
  for _ in range(max(1, len(nodes))):
    refined_keys = [
        (
            base[index],
            tuple(
                sorted(
                    (edge_length, colors[neighbor])
                    for neighbor, edge_length in adjacency[index]
                )
            ),
        )
        for index in range(len(nodes))
    ]
    updated = ranks(refined_keys)
    if updated == colors:
      break
    colors = updated
  return sorted(
      range(len(nodes)),
      key=lambda index: (colors[index], refined_keys[index], index),
  )


def _captured_brep_topology(
    shape: Any,
) -> tuple[list[Any], list[list[Any]], list[list[tuple[int, Any]]], tuple[str, ...]]:
  """Capture full face topology before any model-size policy is applied."""

  if not hasattr(shape, "Faces"):
    raise TypeError("shape must expose CadQuery Faces()")
  faces = list(shape.Faces())
  if not faces:
    raise ValueError("B-Rep must contain at least one face")
  face_edges = [list(face.Edges()) for face in faces]
  edge_groups = _group_occ_edge_owners(face_edges)
  from .cadquery_backend import source_face_signature_sha256

  signatures = tuple(source_face_signature_sha256(face) for face in faces)
  return faces, face_edges, edge_groups, signatures


def _intrinsic_graph_from_face_subset(
    faces: Sequence[Any],
    face_edges: Sequence[Sequence[Any]],
    edge_groups: Sequence[Sequence[tuple[int, Any]]],
    included_raw_indices: Sequence[int],
    *,
    budget: GraphSizeBudget,
) -> tuple[dict[str, Any], dict[int, int]]:
  """Build and re-canonicalize the induced graph for a complete face subset."""

  selected = sorted(set(included_raw_indices))
  if (
      not selected
      or len(selected) > budget.max_faces_per_graph
      or any(type(index) is not int or index < 0 or index >= len(faces) for index in selected)
  ):
    raise ValueError("B-Rep face subset is outside the graph-size budget")
  areas = [_finite(faces[index].Area(), label="face area") for index in selected]
  total_area = sum(areas)
  if not math.isfinite(total_area) or total_area <= 0.0:
    raise ValueError("B-Rep total face area must be positive")
  length_scale = math.sqrt(total_area)

  unsorted_nodes: list[dict[str, Any]] = []
  for raw_index, area in zip(selected, areas, strict=True):
    face = faces[raw_index]
    edges = list(face_edges[raw_index])
    wires = list(face.Wires())
    perimeter = sum(float(edge.Length()) for edge in edges)
    mean_curvature, gaussian_curvature, curvature_known = _face_curvatures(face)
    unsorted_nodes.append(
        {
            "surface_type": _surface_bucket(face),
            "features": [
                _finite(area / total_area, label="area_fraction"),
                _finite(
                    perimeter / length_scale,
                    label="perimeter_over_sqrt_total_area",
                ),
                _finite(
                    mean_curvature * length_scale,
                    label="mean_curvature_times_sqrt_total_area",
                ),
                _finite(
                    gaussian_curvature * total_area,
                    label="gaussian_curvature_times_total_area",
                ),
                float(len(wires)),
                float(len(edges)),
            ],
            "feature_mask": [
                True,
                True,
                curvature_known,
                curvature_known,
                True,
                True,
            ],
        }
    )
  subset_index = {raw_index: index for index, raw_index in enumerate(selected)}
  induced_edge_groups: list[list[tuple[int, Any]]] = []
  induced_raw_edge_groups: list[list[tuple[int, Any]]] = []
  for group in edge_groups:
    owners = [
        (subset_index[raw_index], edge)
        for raw_index, edge in group
        if raw_index in subset_index
    ]
    if len(group) == 2 and len(owners) == 2:
      induced_edge_groups.append(owners)
      induced_raw_edge_groups.append(list(group))
  order = _refined_intrinsic_face_order(
      unsorted_nodes,
      induced_edge_groups,
      length_scale=length_scale,
  )
  local_remap = {old: new for new, old in enumerate(order)}
  remap = {
      raw_index: local_remap[local_index]
      for raw_index, local_index in subset_index.items()
  }
  nodes = [unsorted_nodes[index] for index in order]

  graph_edges: list[dict[str, Any]] = []
  for distinct in induced_raw_edge_groups:
    (left_index, shared), (right_index, _) = distinct
    left_normal = _face_normal_at_edge(faces[left_index], shared)
    right_normal = _face_normal_at_edge(faces[right_index], shared)
    cosine = max(
        -1.0,
        min(1.0, sum(a * b for a, b in zip(left_normal, right_normal, strict=True))),
    )
    source, target = sorted((remap[left_index], remap[right_index]))
    graph_edges.append(
        {
            "source": source,
            "target": target,
            "features": [
                _finite(
                    float(shared.Length()) / length_scale,
                    label="shared_edge_length_over_sqrt_total_area",
                ),
                _finite(cosine, label="dihedral_cos"),
                _finite(
                    math.sqrt(max(0.0, 1.0 - cosine * cosine)),
                    label="dihedral_sin_abs",
                ),
            ],
        }
    )
  graph_edges.sort(
      key=lambda row: (row["source"], row["target"], tuple(row["features"]))
  )
  if len(graph_edges) > budget.max_edges_per_graph:
    raise ValueError("B-Rep adjacency count exceeds the graph-size budget")
  graph = {
      "schema_version": GRAPH_SCHEMA_VERSION,
      "nodes": nodes,
      "edges": graph_edges,
  }
  return graph, remap


def _full_topology_payload(
    signatures: Sequence[str],
    edge_groups: Sequence[Sequence[tuple[int, Any]]],
) -> dict[str, Any]:
  """Permutation-invariant full topology retained only in the private witness."""

  adjacency = sorted(
      tuple(sorted((signatures[group[0][0]], signatures[group[1][0]])))
      for group in edge_groups
      if len(group) == 2 and group[0][0] != group[1][0]
  )
  payload = {
      "schema_version": "benchmark_v2_full_brep_topology_authority.v1",
      "face_signature_multiset": sorted(signatures),
      "adjacency_signature_multiset": [list(pair) for pair in adjacency],
  }
  payload["payload_sha256"] = _canonical_sha256(payload)
  return payload


def _full_topology_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Public commitment projection of the private full-topology witness."""

  return {
      "schema_version": payload["schema_version"],
      "face_count": len(payload["face_signature_multiset"]),
      "adjacency_count": len(payload["adjacency_signature_multiset"]),
      "payload_sha256": payload["payload_sha256"],
  }


def _selected_topology_payload(
    signatures: Sequence[str],
    edge_groups: Sequence[Sequence[tuple[int, Any]]],
    selected_raw_indices: Sequence[int],
    *,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
  """Private exact selected-face/edge fact, independent of receipt assembly."""

  selected = set(selected_raw_indices)
  face_multiset = sorted(signatures[index] for index in selected)
  adjacency_multiset = sorted(
      tuple(sorted((signatures[group[0][0]], signatures[group[1][0]])))
      for group in edge_groups
      if len(group) == 2
      and group[0][0] != group[1][0]
      and group[0][0] in selected
      and group[1][0] in selected
  )
  graph_bytes = _canonical_bytes(graph)
  return {
      "schema_version": "benchmark_v2_selected_brep_subgraph_witness.v1",
      "selected_raw_occ_face_indices": sorted(selected),
      "face_signature_multiset": face_multiset,
      "adjacency_signature_multiset": [list(pair) for pair in adjacency_multiset],
      "face_signature_multiset_sha256": _canonical_sha256(face_multiset),
      "adjacency_signature_multiset_sha256": _canonical_sha256(
          [list(pair) for pair in adjacency_multiset]
      ),
      "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
      "graph_bytes": len(graph_bytes),
  }


def _selected_topology_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Public commitments for selected topology; raw identities remain private."""

  return {
      "schema_version": payload["schema_version"],
      "face_count": len(payload["face_signature_multiset"]),
      "edge_count": len(payload["adjacency_signature_multiset"]),
      "face_signature_multiset_sha256": payload[
          "face_signature_multiset_sha256"
      ],
      "adjacency_signature_multiset_sha256": payload[
          "adjacency_signature_multiset_sha256"
      ],
  }


def _private_graph_witness(
    *,
    graph_index: int,
    step_capture: CapturedFileArtifact,
    endpoint_raw_indices: Sequence[int],
    endpoint_face_signatures: Sequence[str],
    imported_signatures: Sequence[str],
    edge_groups: Sequence[Sequence[tuple[int, Any]]],
    selected_raw_indices: Sequence[int],
    graph: Mapping[str, Any],
    coverage: Mapping[str, Any],
    budget: "GraphSizeBudget",
) -> dict[str, Any]:
  """Canonical fact derived from captured topology before any public receipt."""

  full_topology = _full_topology_payload(imported_signatures, edge_groups)
  selected_topology = _selected_topology_payload(
      imported_signatures,
      edge_groups,
      selected_raw_indices,
      graph=graph,
  )
  return {
      "schema_version": "benchmark_v2_full_topology_and_subgraph_witness.v1",
      "graph_index": graph_index,
      "captured_step": {
          "sha256": step_capture.sha256,
          "bytes": step_capture.byte_count,
      },
      "authority_endpoint_centers": {
          "raw_occ_face_indices": list(endpoint_raw_indices),
          "source_face_signature_sha256s": list(endpoint_face_signatures),
      },
      "policy": {
          "payload": dict(LOCAL_INTERFACE_SUBGRAPH_POLICY),
          "payload_sha256": _canonical_sha256(
              dict(LOCAL_INTERFACE_SUBGRAPH_POLICY)
          ),
      },
      "budget": {
          "max_faces_per_graph": budget.max_faces_per_graph,
          "max_edges_per_graph": budget.max_edges_per_graph,
      },
      "full_topology": full_topology,
      "selected_subgraph": selected_topology,
      "coverage": dict(coverage),
  }


def _public_graph_witness_projection(witness: Mapping[str, Any]) -> dict[str, Any]:
  """Exact receipt fields that must remain bound to the private witness."""

  return {
      "full_graph_authority": _full_topology_authority(
          witness["full_topology"]
      ),
      "selected_subgraph_authority": _selected_topology_authority(
          witness["selected_subgraph"]
      ),
      "local_interface_subgraph": dict(witness["coverage"]),
      "full_topology_and_subgraph_witness_sha256": _canonical_sha256(witness),
  }


def _validated_private_graph_witness_projection(
    witness: Any,
    *,
    graph_index: int,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
  """Validate immutable private facts before projecting receipt commitments."""

  if (
      not isinstance(witness, Mapping)
      or witness.get("schema_version")
      != "benchmark_v2_full_topology_and_subgraph_witness.v1"
      or witness.get("graph_index") != graph_index
  ):
    raise ValueError("authenticated private graph witness schema differs")
  policy = witness.get("policy")
  full = witness.get("full_topology")
  selected = witness.get("selected_subgraph")
  coverage = witness.get("coverage")
  budget = witness.get("budget")
  centers = witness.get("authority_endpoint_centers")
  captured_step = witness.get("captured_step")
  if (
      not isinstance(policy, Mapping)
      or policy.get("payload") != dict(LOCAL_INTERFACE_SUBGRAPH_POLICY)
      or policy.get("payload_sha256")
      != _canonical_sha256(dict(LOCAL_INTERFACE_SUBGRAPH_POLICY))
      or not isinstance(full, Mapping)
      or not isinstance(selected, Mapping)
      or not isinstance(coverage, Mapping)
      or not isinstance(budget, Mapping)
      or not isinstance(centers, Mapping)
      or not isinstance(captured_step, Mapping)
  ):
    raise ValueError("authenticated private graph witness policy differs")
  full_unsigned = dict(full)
  full_hash = full_unsigned.pop("payload_sha256", None)
  selected_faces = selected.get("face_signature_multiset")
  selected_edges = selected.get("adjacency_signature_multiset")
  graph_bytes = _canonical_bytes(graph)
  if (
      full_hash != _canonical_sha256(full_unsigned)
      or not isinstance(selected_faces, list)
      or not isinstance(selected_edges, list)
      or selected.get("face_signature_multiset_sha256")
      != _canonical_sha256(selected_faces)
      or selected.get("adjacency_signature_multiset_sha256")
      != _canonical_sha256(selected_edges)
      or selected.get("graph_sha256")
      != hashlib.sha256(graph_bytes).hexdigest()
      or selected.get("graph_bytes") != len(graph_bytes)
      or budget.get("max_faces_per_graph")
      != coverage.get("effective_max_faces")
      or budget.get("max_edges_per_graph")
      != coverage.get("effective_max_edges")
      or coverage.get("policy_payload_sha256")
      != policy.get("payload_sha256")
      or coverage.get("included_face_count") != len(selected_faces)
      or coverage.get("included_edge_count") != len(selected_edges)
      or not _is_sha256(captured_step.get("sha256"))
      or type(captured_step.get("bytes")) is not int
      or captured_step.get("bytes") <= 0
      or not isinstance(centers.get("raw_occ_face_indices"), list)
      or not isinstance(centers.get("source_face_signature_sha256s"), list)
      or len(centers["raw_occ_face_indices"])
      != len(centers["source_face_signature_sha256s"])
      or not centers["raw_occ_face_indices"]
  ):
    raise ValueError("authenticated private graph witness topology differs")
  return _public_graph_witness_projection(witness)


def _endpoint_local_face_subset(
    *,
    face_count: int,
    edge_groups: Sequence[Sequence[tuple[int, Any]]],
    endpoint_raw_indices: Sequence[int],
    budget: GraphSizeBudget,
) -> tuple[list[int], dict[str, Any]]:
  """Select complete endpoint-centred BFS shells under the frozen policy."""

  centers = sorted(set(endpoint_raw_indices))
  if (
      not centers
      or any(type(index) is not int or index < 0 or index >= face_count for index in centers)
  ):
    raise ValueError("endpoint face identity is outside the full B-Rep graph")
  adjacency: list[set[int]] = [set() for _ in range(face_count)]
  adjacency_edges: list[tuple[int, int]] = []
  for group in edge_groups:
    if len(group) != 2:
      continue
    left, right = group[0][0], group[1][0]
    if left == right:
      continue
    adjacency[left].add(right)
    adjacency[right].add(left)
    adjacency_edges.append((left, right))

  distances = {index: 0 for index in centers}
  frontier = list(centers)
  while frontier:
    next_frontier: list[int] = []
    for index in frontier:
      for neighbor in adjacency[index]:
        if neighbor not in distances:
          distances[neighbor] = distances[index] + 1
          next_frontier.append(neighbor)
    frontier = next_frontier
  max_component_radius = max(distances.values())
  shells = [
      sorted(index for index, distance in distances.items() if distance == radius)
      for radius in range(max_component_radius + 1)
  ]

  def induced_edge_count(indices: set[int]) -> int:
    return sum(left in indices and right in indices for left, right in adjacency_edges)

  selected: set[int] = set()
  admitted_shell_counts: list[int] = []
  achieved_radius = -1
  stop_reason = "none"
  max_radius = int(LOCAL_INTERFACE_SUBGRAPH_POLICY["max_hop_radius"])
  first_omitted_shell: list[int] = []
  for radius, shell in enumerate(shells):
    if radius > max_radius:
      stop_reason = "max_hop_radius"
      first_omitted_shell = shell
      break
    candidate = selected.union(shell)
    fits = (
        len(candidate) <= budget.max_faces_per_graph
        and induced_edge_count(candidate) <= budget.max_edges_per_graph
    )
    if not fits:
      if radius <= 1:
        raise ValueError(
            "complete endpoint one-hop shell exceeds local graph-size budget"
        )
      stop_reason = "face_or_edge_budget"
      first_omitted_shell = shell
      break
    selected = candidate
    admitted_shell_counts.append(len(shell))
    achieved_radius = radius

  complete_component = len(selected) == len(distances)
  if complete_component:
    stop_reason = "none"
    first_omitted_shell = []
  ledger = {
      "schema_version": LOCAL_INTERFACE_SUBGRAPH_POLICY["schema_version"],
      "policy_payload_sha256": _canonical_sha256(
          dict(LOCAL_INTERFACE_SUBGRAPH_POLICY)
      ),
      "effective_max_faces": budget.max_faces_per_graph,
      "effective_max_edges": budget.max_edges_per_graph,
      "max_hop_radius": max_radius,
      "endpoint_face_count": len(centers),
      "endpoint_component_face_count": len(distances),
      "endpoint_component_edge_count": induced_edge_count(set(distances)),
      "included_face_count": len(selected),
      "included_edge_count": induced_edge_count(selected),
      "admitted_shell_face_counts": admitted_shell_counts,
      "achieved_hop_radius": achieved_radius,
      "complete_component": complete_component,
      "truncated": not complete_component,
      "truncation_reason": stop_reason,
      "omitted_frontier_face_count": len(first_omitted_shell),
      "omitted_component_face_count": len(distances) - len(selected),
  }
  return sorted(selected), ledger


def _extract_local_intrinsic_brep_graph_with_identity(
    shape: Any,
    *,
    endpoint_raw_indices: Sequence[int],
    budget: GraphSizeBudget = DEFAULT_GRAPH_SIZE_BUDGET,
) -> tuple[
    dict[str, Any], dict[int, int], tuple[str, ...], dict[str, Any], dict[str, Any]
]:
  """Extract a canonical endpoint-local graph and receipt-only coverage ledger."""

  budget = _validated_budget(budget)
  faces, face_edges, edge_groups, signatures = _captured_brep_topology(shape)
  selected, ledger = _endpoint_local_face_subset(
      face_count=len(faces),
      edge_groups=edge_groups,
      endpoint_raw_indices=endpoint_raw_indices,
      budget=budget,
  )
  graph, remap = _intrinsic_graph_from_face_subset(
      faces, face_edges, edge_groups, selected, budget=budget
  )
  if any(index not in remap for index in endpoint_raw_indices):
    raise ValueError("endpoint face was not preserved by local graph policy")
  return graph, remap, signatures, ledger, _full_topology_authority(
      _full_topology_payload(signatures, edge_groups)
  )


def _derive_private_graph_from_captured_shape(
    shape: Any,
    *,
    graph_index: int,
    step_capture: CapturedFileArtifact,
    endpoint_raw_indices: Sequence[int],
    endpoint_face_signatures: Sequence[str],
    budget: GraphSizeBudget,
) -> tuple[dict[str, Any], dict[int, int], tuple[str, ...], dict[str, Any]]:
  """Derive graph and private witness in one pass over captured STEP topology."""

  budget = _validated_budget(budget)
  faces, face_edges, edge_groups, signatures = _captured_brep_topology(shape)
  if any(
      type(index) is not int or index < 0 or index >= len(signatures)
      for index in endpoint_raw_indices
  ):
    raise ValueError("mapped OCC face index is outside re-opened STEP")
  observed = [signatures[index] for index in endpoint_raw_indices]
  if observed != list(endpoint_face_signatures):
    raise ValueError("re-opened STEP face signatures differ from face authority")
  selected, coverage = _endpoint_local_face_subset(
      face_count=len(faces),
      edge_groups=edge_groups,
      endpoint_raw_indices=endpoint_raw_indices,
      budget=budget,
  )
  graph, remap = _intrinsic_graph_from_face_subset(
      faces, face_edges, edge_groups, selected, budget=budget
  )
  if any(index not in remap for index in endpoint_raw_indices):
    raise ValueError("endpoint face was not preserved by local graph policy")
  witness = _private_graph_witness(
      graph_index=graph_index,
      step_capture=step_capture,
      endpoint_raw_indices=endpoint_raw_indices,
      endpoint_face_signatures=endpoint_face_signatures,
      imported_signatures=signatures,
      edge_groups=edge_groups,
      selected_raw_indices=selected,
      graph=graph,
      coverage=coverage,
      budget=budget,
  )
  return graph, remap, signatures, witness


def _extract_intrinsic_brep_graph_with_identity(
    shape: Any,
    *,
    budget: GraphSizeBudget = DEFAULT_GRAPH_SIZE_BUDGET,
) -> tuple[dict[str, Any], dict[int, int], tuple[str, ...]]:
  """Extract a whole graph plus the receipt-bound OCC-index identity map."""

  budget = _validated_budget(budget)
  faces, face_edges, edge_groups, signatures = _captured_brep_topology(shape)
  if len(faces) > budget.max_faces_per_graph:
    raise ValueError("B-Rep face count is outside the graph-size budget")
  graph, remap = _intrinsic_graph_from_face_subset(
      faces, face_edges, edge_groups, range(len(faces)), budget=budget
  )
  return graph, remap, signatures


def extract_intrinsic_brep_graph(
    shape: Any,
    *,
    budget: GraphSizeBudget = DEFAULT_GRAPH_SIZE_BUDGET,
) -> dict[str, Any]:
  """Extract a rigid-motion-invariant face-adjacency graph from a CadQuery shape."""

  graph, _, _ = _extract_intrinsic_brep_graph_with_identity(shape, budget=budget)
  return graph


def _exact_keys(
    value: Any, expected: set[str], *, label: str
) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ModelViewV2LeakageError(f"{label} must be an object")
  if set(value) != expected:
    raise ModelViewV2LeakageError(
        f"{label} has unknown or missing keys"
    )
  return value


def _normalized_vector(
    value: Any, *, length: int, label: str
) -> list[float]:
  if not isinstance(value, (list, tuple)) or len(value) != length:
    raise ValueError(f"{label} must contain exactly {length} values")
  return [_finite(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _validated_mask(value: Any, *, length: int, label: str) -> list[bool]:
  if not isinstance(value, (list, tuple)) or len(value) != length:
    raise ValueError(f"{label} must contain exactly {length} values")
  if any(type(item) is not bool for item in value):
    raise ValueError(f"{label} must contain actual booleans")
  return list(value)


def _validated_graph(
    raw: Any, *, graph_index: int, budget: GraphSizeBudget
) -> dict[str, Any]:
  graph = _exact_keys(
      raw,
      {"schema_version", "nodes", "edges"},
      label=f"graph[{graph_index}]",
  )
  if graph.get("schema_version") != GRAPH_SCHEMA_VERSION:
    raise ValueError("graph schema_version mismatch")
  raw_nodes = graph.get("nodes")
  raw_edges = graph.get("edges")
  if not isinstance(raw_nodes, list) or not raw_nodes:
    raise ValueError("graph nodes must be a non-empty list")
  if len(raw_nodes) > budget.max_faces_per_graph:
    raise ValueError("graph face count exceeds budget")
  if not isinstance(raw_edges, list):
    raise ValueError("graph edges must be a list")
  if len(raw_edges) > budget.max_edges_per_graph:
    raise ValueError("graph edge count exceeds budget")
  nodes: list[dict[str, Any]] = []
  for node_index, raw_node in enumerate(raw_nodes):
    node = _exact_keys(
        raw_node,
        {"surface_type", "features", "feature_mask"},
        label=f"graph[{graph_index}].node[{node_index}]",
    )
    surface_type = str(node.get("surface_type") or "").strip().lower()
    if surface_type not in _SURFACE_TYPES:
      raise ValueError("graph node surface_type is not allowlisted")
    features = _normalized_vector(
        node.get("features"),
        length=len(NODE_FEATURE_NAMES),
        label=f"graph[{graph_index}].node[{node_index}].features",
    )
    feature_mask = _validated_mask(
        node.get("feature_mask"),
        length=len(NODE_FEATURE_NAMES),
        label=f"graph[{graph_index}].node[{node_index}].feature_mask",
    )
    if any(not feature_mask[index] for index in (0, 1, 4, 5)):
      raise ValueError("only curvature feature dimensions may be unknown")
    if feature_mask[2] != feature_mask[3]:
      raise ValueError("mean and Gaussian curvature masks must agree")
    if any(
        not known and features[index] != 0.0
        for index, known in enumerate(feature_mask)
    ):
      raise ValueError("masked feature values must be canonical zero")
    nodes.append(
        {
            "surface_type": surface_type,
            "features": features,
            "feature_mask": feature_mask,
        }
    )
  edges: list[dict[str, Any]] = []
  for edge_index, raw_edge in enumerate(raw_edges):
    edge = _exact_keys(
        raw_edge,
        {"source", "target", "features"},
        label=f"graph[{graph_index}].edge[{edge_index}]",
    )
    source = edge.get("source")
    target = edge.get("target")
    if (
        type(source) is not int
        or type(target) is not int
        or source < 0
        or target < 0
        or source >= len(nodes)
        or target >= len(nodes)
        or source >= target
    ):
      raise ValueError("graph edge endpoints must be ordered valid node indices")
    edges.append(
        {
            "source": source,
            "target": target,
            "features": _normalized_vector(
                edge.get("features"),
                length=len(EDGE_FEATURE_NAMES),
                label=f"graph[{graph_index}].edge[{edge_index}].features",
            ),
        }
    )
  # A trimmed B-Rep may contain multiple topological edges with identical
  # geometry between the same two faces.  Preserve that multiplicity instead
  # of inventing an aggregation rule; canonical sorting remains deterministic.
  return {
      "schema_version": GRAPH_SCHEMA_VERSION,
      "nodes": nodes,
      "edges": sorted(
          edges,
          key=lambda row: (
              row["source"], row["target"], tuple(row["features"])
          ),
      ),
  }


def _validated_endpoint(
    raw: Any, *, graphs: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Any]:
  endpoint = _exact_keys(
      raw, {"graph_index", "face_indices"}, label=label
  )
  graph_index = endpoint.get("graph_index")
  face_indices = endpoint.get("face_indices")
  if type(graph_index) is not int or graph_index < 0 or graph_index >= len(graphs):
    raise ValueError(f"{label}.graph_index is outside the graph list")
  if not isinstance(face_indices, list) or not face_indices:
    raise ValueError(f"{label}.face_indices must be a non-empty list")
  if any(type(value) is not int for value in face_indices):
    raise ValueError(f"{label}.face_indices must contain actual integers")
  normalized = sorted(set(face_indices))
  if len(normalized) != len(face_indices) or normalized[-1] >= len(
      graphs[graph_index]["nodes"]
  ) or normalized[0] < 0:
    raise ValueError(f"{label}.face_indices are duplicate or out of range")
  return {"graph_index": graph_index, "face_indices": normalized}


def _validated_rotation(raw: Any, *, label: str) -> list[float]:
  values = _normalized_vector(raw, length=9, label=label)
  rows = [values[0:3], values[3:6], values[6:9]]
  for left in range(3):
    for right in range(3):
      dot = sum(rows[left][axis] * rows[right][axis] for axis in range(3))
      expected = 1.0 if left == right else 0.0
      if abs(dot - expected) > 1e-6:
        raise ValueError(f"{label} must be an orthonormal rotation")
  determinant = (
      values[0] * (values[4] * values[8] - values[5] * values[7])
      - values[1] * (values[3] * values[8] - values[5] * values[6])
      + values[2] * (values[3] * values[7] - values[4] * values[6])
  )
  if abs(determinant - 1.0) > 1e-6:
    raise ValueError(f"{label} must have determinant +1")
  return values


def _validated_catalog_descriptor(raw: Any, *, label: str) -> dict[str, str]:
  descriptor = _exact_keys(raw, set(_CATALOG_DESCRIPTOR_KEYS), label=label)
  normalized: dict[str, str] = {}
  for key in _CATALOG_DESCRIPTOR_KEYS:
    value = str(descriptor.get(key) or "").strip().lower()
    if _SAFE_CATALOG_TOKEN.fullmatch(value) is None:
      raise ValueError(f"{label}.{key} is not a safe catalog token")
    normalized[key] = value
  if normalized["relation_hint"] not in _PROGRAM_TYPES:
    raise ValueError(f"{label}.relation_hint is not allowlisted")
  return normalized


def _validated_example(
    raw: Any,
    *,
    example_index: int,
    graphs: Sequence[Mapping[str, Any]],
    budget: GraphSizeBudget,
) -> dict[str, Any]:
  example = _exact_keys(
      raw,
      {"query", "program_candidates", "target"},
      label=f"example[{example_index}]",
  )
  query_raw = _exact_keys(
      example.get("query"),
      {"endpoint_a", "endpoint_b"},
      label=f"example[{example_index}].query",
  )
  query = {
      key: _validated_endpoint(
          query_raw[key], graphs=graphs, label=f"example[{example_index}].query.{key}"
      )
      for key in ("endpoint_a", "endpoint_b")
  }
  raw_candidates = example.get("program_candidates")
  if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
    raise ValueError("program candidates must contain at least two programs")
  if len(raw_candidates) > budget.max_program_candidates_per_example:
    raise ValueError("program candidate count exceeds budget")
  candidates: list[dict[str, Any]] = []
  seen_program_indices: set[int] = set()
  seen_program_ids: set[str] = set()
  seen_descriptors: set[str] = set()
  for candidate_index, raw_candidate in enumerate(raw_candidates):
    candidate = _exact_keys(
        raw_candidate,
        {
            "program_index",
            "program_id",
            "program_type",
            "descriptor",
            "endpoint_a",
            "endpoint_b",
        },
        label=f"example[{example_index}].candidate[{candidate_index}]",
    )
    program_index = candidate.get("program_index")
    if type(program_index) is not int or program_index < 0:
      raise ValueError("program_index must be a non-negative actual integer")
    if program_index in seen_program_indices:
      raise ValueError("program candidates contain duplicate program_index")
    seen_program_indices.add(program_index)
    program_type = str(candidate.get("program_type") or "").strip().lower()
    if program_type not in _PROGRAM_TYPES:
      raise ValueError("program_type is not a finite allowlisted type")
    program_id = str(candidate.get("program_id") or "").strip().lower()
    if re.fullmatch(r"catalog_[0-9a-f]{16}", program_id) is None:
      raise ValueError("program_id is not a fixed opaque catalog identifier")
    descriptor = _validated_catalog_descriptor(
        candidate.get("descriptor"),
        label=f"example[{example_index}].candidate[{candidate_index}].descriptor",
    )
    if descriptor["relation_hint"] != program_type:
      raise ValueError("candidate program_type differs from descriptor")
    descriptor_sha256 = _canonical_sha256(descriptor)
    if program_id != f"catalog_{descriptor_sha256[:16]}":
      raise ValueError("program_id differs from its fixed catalog descriptor")
    if program_id in seen_program_ids or descriptor_sha256 in seen_descriptors:
      raise ValueError("program candidates must be genuinely distinct")
    seen_program_ids.add(program_id)
    seen_descriptors.add(descriptor_sha256)
    candidates.append(
        {
            "program_index": program_index,
            "program_id": program_id,
            "program_type": program_type,
            "descriptor": descriptor,
            "endpoint_a": _validated_endpoint(
                candidate.get("endpoint_a"),
                graphs=graphs,
                label=f"example[{example_index}].candidate[{candidate_index}].endpoint_a",
            ),
            "endpoint_b": _validated_endpoint(
                candidate.get("endpoint_b"),
                graphs=graphs,
                label=f"example[{example_index}].candidate[{candidate_index}].endpoint_b",
            ),
        }
    )
  target = _exact_keys(
      example.get("target"),
      {"program_index", "residual_translation", "residual_rotation_row_major"},
      label=f"example[{example_index}].target",
  )
  target_program = target.get("program_index")
  if type(target_program) is not int or target_program not in seen_program_indices:
    raise ValueError("target program_index is not in the finite candidate set")
  return {
      "query": query,
      "program_candidates": sorted(candidates, key=lambda row: row["program_index"]),
      "target": {
          "program_index": target_program,
          "residual_translation": _normalized_vector(
              target.get("residual_translation"),
              length=3,
              label=f"example[{example_index}].target.residual_translation",
          ),
          "residual_rotation_row_major": _validated_rotation(
              target.get("residual_rotation_row_major"),
              label=f"example[{example_index}].target.residual_rotation_row_major",
          ),
      },
  }


def build_benchmark_v2_model_view_v2(
    *,
    graphs: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    budget: GraphSizeBudget = DEFAULT_GRAPH_SIZE_BUDGET,
) -> ModelViewV2Sanitization:
  """Build the exact model-visible v2 payload; no provenance enters this API."""

  budget = _validated_budget(budget)
  if not isinstance(graphs, (list, tuple)) or not graphs:
    raise ValueError("graphs must be a non-empty sequence")
  if len(graphs) > budget.max_graphs:
    raise ValueError("graph count exceeds budget")
  if not isinstance(examples, (list, tuple)):
    raise ValueError("examples must be a sequence")
  if len(examples) > budget.max_examples:
    raise ValueError("example count exceeds budget")
  normalized_graphs = [
      _validated_graph(raw, graph_index=index, budget=budget)
      for index, raw in enumerate(graphs)
  ]
  normalized_examples = [
      _validated_example(
          raw,
          example_index=index,
          graphs=normalized_graphs,
          budget=budget,
      )
      for index, raw in enumerate(examples)
  ]
  model_view = {
      "schema_version": SCHEMA_VERSION,
      "normalization": copy.deepcopy(_NORMALIZATION_METADATA),
      "budget": budget.to_dict(),
      "graphs": normalized_graphs,
      "examples": normalized_examples,
  }
  canonical_json = json.dumps(
      model_view,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  )
  return ModelViewV2Sanitization(
      model_view=model_view,
      canonical_json=canonical_json,
      sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
  )


def validate_benchmark_v2_model_view_v2(
    model_view: Mapping[str, Any], *, expected_sha256: str | None = None
) -> str:
  """Fail closed on any v2 schema drift or non-canonical value."""

  top = _exact_keys(
      model_view,
      {"schema_version", "normalization", "budget", "graphs", "examples"},
      label="benchmark_v2_model_view.v2",
  )
  if top.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("benchmark-v2 model-view v2 schema_version mismatch")
  if top.get("normalization") != _NORMALIZATION_METADATA:
    raise ModelViewV2LeakageError("normalization metadata differs from v2 policy")
  raw_budget = _exact_keys(
      top.get("budget"), set(DEFAULT_GRAPH_SIZE_BUDGET.to_dict()), label="budget"
  )
  try:
    budget = GraphSizeBudget(**dict(raw_budget))
  except TypeError as error:
    raise ValueError("v2 graph budget is malformed") from error
  budget = _validated_budget(budget)
  rebuilt = build_benchmark_v2_model_view_v2(
      graphs=top.get("graphs"), examples=top.get("examples"), budget=budget
  )
  if rebuilt.model_view != model_view:
    raise ValueError("benchmark-v2 model-view v2 is not canonical")
  if expected_sha256 is not None and rebuilt.sha256 != str(expected_sha256):
    raise ValueError("benchmark-v2 model-view v2 hash mismatch")
  return rebuilt.sha256


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


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key is forbidden: {key}")
    result[key] = value
  return result


def _strict_json_bytes(raw_bytes: bytes, *, label: str) -> Any:
  try:
    return json.loads(
        raw_bytes.decode("utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant is forbidden: {value}")
        ),
    )
  except (UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _capture_json(path: str | Path, *, label: str) -> _CapturedJson:
  captured = capture_file_artifact(path, label=label)
  payload = _strict_json_bytes(captured.raw_bytes, label=label)
  return _CapturedJson(file=captured, payload=payload)


def _verify_binding(
    binding: Any, captured: CapturedFileArtifact, *, label: str
) -> None:
  if not isinstance(binding, Mapping):
    raise ValueError(f"{label} binding must be an object")
  if (
      binding.get("sha256") != captured.sha256
      or binding.get("bytes") != captured.byte_count
      or Path(str(binding.get("path") or "")).resolve(strict=False)
      != captured.resolved_path
  ):
    raise ValueError(f"{label} binding differs from captured file")


def _verify_self_hash(payload: Any, *, field: str, label: str) -> None:
  if not isinstance(payload, Mapping):
    raise ValueError(f"{label} must be an object")
  unsigned = dict(payload)
  observed = unsigned.pop(field, None)
  if observed != _canonical_sha256(unsigned):
    raise ValueError(f"{label} self-hash differs")


def _require_development_flags(payload: Mapping[str, Any], *, label: str) -> None:
  if (
      payload.get("development") is not True
      or payload.get("publication_eligible") is not False
      or payload.get("final_test_touched") is not False
  ):
    raise ValueError(f"{label} is not train/dev-only and final-test-untouched")


def _stage_output(
    receipt: Mapping[str, Any], *, role: str, schema_version: str
) -> Mapping[str, Any]:
  outputs = receipt.get("outputs")
  if not isinstance(outputs, list):
    raise ValueError("formal stage receipt outputs must be a list")
  matches = [
      row for row in outputs
      if isinstance(row, Mapping) and row.get("role") == role
  ]
  if len(matches) != 1 or matches[0].get("schema_version") != schema_version:
    raise ValueError(f"formal stage receipt lacks unique {role} output")
  return matches[0]


def _captured_stage_authorities(
    record: Mapping[str, Any], *, case_id: str
) -> _CaseAuthorityBinding:
  source_ordinal = record.get("source_ordinal")
  if type(source_ordinal) is not int or source_ordinal < 0:
    raise ValueError("authority record source ordinal is malformed")
  raw_receipts = record.get("stage_receipts")
  if not isinstance(raw_receipts, list):
    raise ValueError("authority record stage_receipts must be a list")
  by_stage: dict[str, tuple[_CapturedJson, Mapping[str, Any]]] = {}
  captures: list[CapturedFileArtifact] = []
  for row in raw_receipts:
    if not isinstance(row, Mapping):
      raise ValueError("authority stage receipt binding is malformed")
    captured = _capture_json(
        str(row.get("path") or ""), label="formal stage receipt"
    )
    _verify_binding(row, captured.file, label="formal stage receipt")
    payload = captured.payload
    if not isinstance(payload, Mapping):
      raise ValueError("formal stage receipt payload must be an object")
    _verify_self_hash(
        payload,
        field="receipt_payload_sha256",
        label="formal stage receipt",
    )
    stage = str(payload.get("stage") or "")
    if (
        payload.get("schema_version") != _STAGE_RECEIPT_SCHEMA
        or payload.get("case_id") != case_id
        or payload.get("status") != "verified"
        or payload.get("formal") is not True
        or payload.get("development") is not False
        or payload.get("final_test_touched") is not False
        or payload.get("source_ordinal") != source_ordinal
        or row.get("status") != "verified"
        or row.get("schema_version") != _STAGE_RECEIPT_SCHEMA
        or row.get("receipt_payload_sha256")
        != payload.get("receipt_payload_sha256")
    ):
      raise ValueError("formal stage receipt is not verified and final-test-safe")
    if stage in by_stage:
      raise ValueError("authority record contains duplicate stage receipt")
    by_stage[stage] = (captured, payload)
    captures.append(captured.file)
  required = {"face_v4_builder", "contact_authority", "semantic_v2"}
  if not required <= set(by_stage):
    raise ValueError("authority record lacks face/contact/mate stages")

  output_specs = (
      (
          "face_v4_builder",
          "obj_extension",
          "fusion_content_receipt_obj_extension.v2",
      ),
      (
          "face_v4_builder", "face_map", "fusion_step_face_map_audit.v4"
      ),
      (
          "contact_authority",
          "contact_census",
          "fusion_contact_census_authority.v1",
      ),
      (
          "semantic_v2", "mate_semantic", "mate_semantic_authority.v2"
      ),
  )
  output_payloads: dict[str, Mapping[str, Any]] = {}
  for stage, role, schema in output_specs:
    output = _stage_output(by_stage[stage][1], role=role, schema_version=schema)
    captured_output = _capture_json(
        str(output.get("path") or ""), label=f"formal {role} authority"
    )
    _verify_binding(output, captured_output.file, label=f"formal {role} authority")
    payload = captured_output.payload
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != schema
        or payload.get("formal") is not True
    ):
      raise ValueError(f"formal {role} authority payload is invalid")
    _verify_self_hash(
        payload,
        field="receipt_payload_sha256",
        label=f"formal {role} authority",
    )
    output_payloads[role] = payload
    captures.append(captured_output.file)

  semantic_stage = by_stage["semantic_v2"][1]
  library_output = _stage_output(
      semantic_stage,
      role="mate_library",
      schema_version="mate_program_library.v2",
  )
  library_capture = capture_file_artifact(
      str(library_output.get("path") or ""), label="formal mate library"
  )
  _verify_binding(library_output, library_capture, label="formal mate library")
  captures.append(library_capture)

  semantic_receipt_capture, semantic_stage_payload = by_stage["semantic_v2"]
  request_capture = _capture_json(
      semantic_receipt_capture.file.resolved_path.with_name(
          "semantic_v2.request.json"
      ),
      label="formal semantic stage request",
  )
  request_payload = request_capture.payload
  if not isinstance(request_payload, Mapping):
    raise ValueError("formal semantic stage request must be an object")
  _verify_self_hash(
      request_payload,
      field="request_payload_sha256",
      label="formal semantic stage request",
  )
  if (
      request_payload.get("schema_version")
      != "neurocad_formal_authority_stage_request.v1"
      or request_payload.get("stage") != "semantic_v2"
      or request_payload.get("case_id") != case_id
      or request_payload.get("source_ordinal") != source_ordinal
      or request_payload.get("formal") is not True
      or request_payload.get("development") is not False
      or request_payload.get("final_test_touched") is not False
      or request_capture.file.sha256
      != semantic_stage_payload.get("request_sha256")
  ):
    raise ValueError("semantic stage request differs from verified receipt")
  bound_inputs = request_payload.get("bound_inputs")
  stage_manifest = (
      bound_inputs.get("stage_input_manifest")
      if isinstance(bound_inputs, Mapping)
      else None
  )
  manifest_cases = (
      stage_manifest.get("cases") if isinstance(stage_manifest, Mapping) else None
  )
  authority_production_split = (
      stage_manifest.get("source_split")
      if isinstance(stage_manifest, Mapping)
      else None
  )
  candidate_row = request_payload.get("candidate_row")
  if (
      authority_production_split not in {"train", "dev"}
      or not isinstance(candidate_row, Mapping)
      or candidate_row.get("case_id") != case_id
      or candidate_row.get("source_ordinal") != source_ordinal
  ):
    raise ValueError("semantic request case/source provenance differs")
  matching_cases = [
      row for row in manifest_cases or []
      if isinstance(row, Mapping) and row.get("case_id") == case_id
  ]
  if len(matching_cases) != 1:
    raise ValueError("semantic request lacks a unique case input binding")
  case_inputs = matching_cases[0].get("inputs")
  private_gold_binding = (
      case_inputs.get("private_gold") if isinstance(case_inputs, Mapping) else None
  )
  if not isinstance(private_gold_binding, Mapping):
    raise ValueError("semantic request lacks private-gold endpoint authority")
  private_gold_capture = _capture_json(
      str(private_gold_binding.get("path") or ""),
      label="formal private-gold endpoint authority",
  )
  _verify_binding(
      private_gold_binding,
      private_gold_capture.file,
      label="formal private-gold endpoint authority",
  )
  if not isinstance(private_gold_capture.payload, Mapping):
    raise ValueError("formal private-gold endpoint authority must be an object")
  _verify_private_gold_production_contract(
      private_gold_capture.payload,
      authority_production_split=str(authority_production_split),
      case_id=case_id,
  )
  captures.extend((request_capture.file, private_gold_capture.file))

  binding_sha256 = _canonical_sha256(
      {
          "schema_version": "benchmark_v2_case_graph_authority_binding.v1",
          "case_id": case_id,
          "source_ordinal": source_ordinal,
          "authority_production_split": authority_production_split,
          "face_authority_sha256": _canonical_sha256(output_payloads["face_map"]),
          "contact_authority_sha256": _canonical_sha256(
              output_payloads["contact_census"]
          ),
          "semantic_authority_sha256": _canonical_sha256(
              output_payloads["mate_semantic"]
          ),
          "private_gold_file_sha256": private_gold_capture.file.sha256,
          "mate_library_file_sha256": library_capture.sha256,
      }
  )
  return _CaseAuthorityBinding(
      authority_production_split=str(authority_production_split),
      model_assignment_split="",
      source_ordinal=source_ordinal,
      case_id=case_id,
      binding_sha256=binding_sha256,
      face_payload=MappingProxyType(dict(output_payloads["face_map"])),
      contact_payload=MappingProxyType(dict(output_payloads["contact_census"])),
      semantic_payload=MappingProxyType(dict(output_payloads["mate_semantic"])),
      private_gold_payload=MappingProxyType(
          dict(private_gold_capture.payload)
      ),
      library_capture=library_capture,
      captures=tuple(captures),
  )


def _pool_records(pool: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
  if pool.get("schema_version") != _AUTHORITY_POOL_SCHEMA:
    raise ValueError("unsupported authority-ready pool schema")
  _verify_self_hash(pool, field="pool_payload_sha256", label="authority pool")
  records = pool.get("authority_ready_records")
  if not isinstance(records, list) or len(records) != 700:
    raise ValueError("authority-ready pool must contain exactly 700 records")
  result: dict[str, Mapping[str, Any]] = {}
  for record in records:
    if not isinstance(record, Mapping):
      raise ValueError("authority-ready pool record is malformed")
    unsigned = dict(record)
    observed = unsigned.pop("merged_record_payload_sha256", None)
    if observed != _canonical_sha256(unsigned):
      raise ValueError("authority-ready record self-hash differs")
    case_id = str(record.get("case_id") or "")
    if (
        not case_id
        or case_id in result
        or record.get("authority_ready") is not True
        or record.get("status") != "authority_ready"
    ):
      raise ValueError("authority-ready pool contains invalid or duplicate case")
    result[case_id] = record
  return result


def _assignment_rows(
    captured: _CapturedJson,
    *,
    split: str,
    manifest: Mapping[str, Any],
    pool_capture: _CapturedJson,
    pool_records: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[_CaseAuthorityBinding]]:
  assignment = captured.payload
  if not isinstance(assignment, Mapping):
    raise ValueError("formal assignment must be an object")
  _verify_self_hash(
      assignment,
      field="assignment_payload_sha256",
      label=f"{split} assignment",
  )
  _require_development_flags(assignment, label=f"{split} assignment")
  if (
      assignment.get("schema_version") != _ASSIGNMENT_SCHEMA
      or assignment.get("formal_authority_backed") is not True
      or assignment.get("split") != split
  ):
    raise ValueError(f"{split} assignment schema or split differs")
  _verify_binding(
      assignment.get("authority_pool"),
      pool_capture.file,
      label=f"{split} assignment authority pool",
  )
  manifest_binding = manifest.get("output_bindings", {}).get(split)
  _verify_binding(manifest_binding, captured.file, label=f"manifest {split}")
  rows = assignment.get("records")
  expected = 500 if split == "train" else 100
  if (
      not isinstance(rows, list)
      or assignment.get("case_count") != expected
      or len(rows) != expected
  ):
    raise ValueError(f"{split} assignment count differs from frozen policy")
  bindings: list[_CaseAuthorityBinding] = []
  seen: set[str] = set()
  for row in rows:
    if not isinstance(row, Mapping) or set(row) != {
        "case_id",
        "source_ordinal",
        "leakage_component_id",
        "authority_record_payload_sha256",
    }:
      raise ValueError(f"{split} assignment row schema differs")
    case_id = str(row.get("case_id") or "")
    if not case_id or case_id in seen or case_id not in pool_records:
      raise ValueError(f"{split} assignment case join is invalid")
    seen.add(case_id)
    record = pool_records[case_id]
    if (
        row.get("authority_record_payload_sha256")
        != record.get("merged_record_payload_sha256")
        or row.get("source_ordinal") != record.get("source_ordinal")
    ):
      raise ValueError(f"{split} assignment differs from authority record")
    authority = _captured_stage_authorities(record, case_id=case_id)
    bindings.append(
        _CaseAuthorityBinding(
            authority_production_split=authority.authority_production_split,
            model_assignment_split=split,
            source_ordinal=authority.source_ordinal,
            case_id=authority.case_id,
            binding_sha256=authority.binding_sha256,
            face_payload=authority.face_payload,
            contact_payload=authority.contact_payload,
            semantic_payload=authority.semantic_payload,
            private_gold_payload=authority.private_gold_payload,
            library_capture=authority.library_capture,
            captures=authority.captures,
        )
    )
  return rows, bindings


_GRAPH_AUTHORITY_RECEIPT_SCHEMA = (
    "benchmark_v2_authenticated_brep_graph_receipt.v2"
)


def _authority_case(
    payload: Mapping[str, Any], *, case_id: str, label: str
) -> Mapping[str, Any]:
  cases = payload.get("cases")
  matches = [
      row for row in cases or []
      if isinstance(row, Mapping) and row.get("case_id") == case_id
  ]
  if len(matches) != 1:
    raise ValueError(f"{label} lacks one receipt-bound case")
  return matches[0]


def _verify_private_gold_production_contract(
    payload: Mapping[str, Any],
    *,
    authority_production_split: str,
    case_id: str,
) -> Mapping[str, Any]:
  """Bind private gold to authority-production provenance, not model assignment.

  Formal authority for the current pool was produced from the original family
  ``train`` split.  A later v3 development assignment may independently place
  the same authenticated case in model ``train`` or ``dev``.  Conflating those
  two meanings would either reject every reassigned dev case or weaken the
  original private-gold provenance check.
  """

  if (
      authority_production_split not in {"train", "dev"}
      or payload.get("schema_version")
      != "benchmark_v2_private_evaluation_gold_split.v1"
      or payload.get("split") != authority_production_split
      or payload.get("visibility") != "private_evaluator_only"
  ):
    raise ValueError("private-gold authority-production contract differs")
  return _authority_case(payload, case_id=case_id, label="private gold")


def _captured_bound_file(
    binding: Any, *, label: str
) -> CapturedFileArtifact:
  if not isinstance(binding, Mapping):
    raise ValueError(f"{label} binding must be an object")
  captured = capture_file_artifact(str(binding.get("path") or ""), label=label)
  _verify_binding(binding, captured, label=label)
  return captured


def _private_gold_contact(
    binding: _CaseAuthorityBinding, *, source_contact_ordinal: int
) -> Mapping[str, Any]:
  payload = binding.private_gold_payload
  case = _verify_private_gold_production_contract(
      payload,
      authority_production_split=binding.authority_production_split,
      case_id=binding.case_id,
  )
  authority = case.get("evaluation_gold_contacts")
  contacts = authority.get("contacts") if isinstance(authority, Mapping) else None
  matches = [
      row for row in contacts or []
      if isinstance(row, Mapping)
      and row.get("source_contact_ordinal") == source_contact_ordinal
  ]
  if len(matches) != 1:
    raise ValueError("semantic row lacks one private-gold endpoint pair")
  return matches[0]


def _face_endpoint_binding(
    binding: _CaseAuthorityBinding,
    *,
    endpoint: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
  if not isinstance(endpoint, Mapping):
    raise ValueError("private-gold endpoint is malformed")
  part = endpoint.get("part")
  fusion_face_index = endpoint.get("fusion_face_index")
  if (
      not isinstance(part, str)
      or not part
      or type(fusion_face_index) is not int
      or fusion_face_index < 0
      or endpoint.get("entity_type") != "BRepFace"
  ):
    raise ValueError("private-gold endpoint identity is malformed")
  face_case = _authority_case(
      binding.face_payload, case_id=binding.case_id, label="face authority"
  )
  mappings = face_case.get("face_mappings")
  matches = [
      row for row in mappings or []
      if isinstance(row, Mapping)
      and row.get("part") == part
      and row.get("fusion_face_index") == fusion_face_index
  ]
  if len(matches) != 1 or matches[0].get("status") != "mapped_unique":
    raise ValueError("private-gold endpoint lacks unique mapped OCC identity")
  mapping = matches[0]
  raw_indices = mapping.get("raw_occ_face_indices")
  signatures = mapping.get("source_face_signature_sha256s")
  if (
      not isinstance(raw_indices, list)
      or not raw_indices
      or any(type(value) is not int or value < 0 for value in raw_indices)
      or raw_indices != sorted(set(raw_indices))
      or not isinstance(signatures, list)
      or len(signatures) != len(raw_indices)
  ):
    raise ValueError("mapped OCC endpoint identity is malformed")
  inputs = face_case.get("inputs")
  if not isinstance(inputs, Mapping):
    raise ValueError("face authority lacks receipt-bound geometry inputs")
  steps = [
      row for row in inputs.get("steps") or []
      if isinstance(row, Mapping) and row.get("part") == part
  ]
  instances = [
      row for row in inputs.get("instances") or []
      if isinstance(row, Mapping) and row.get("part") == part
  ]
  if len(steps) != 1 or len(instances) != 1:
    raise ValueError("face authority part has ambiguous STEP or instance identity")
  instance = instances[0]
  if (
      mapping.get("world_transform_chain_sha256")
      != instance.get("world_transform_chain_sha256")
      or mapping.get("world_transform_mm") != instance.get("world_transform_mm")
      or mapping.get("source_instance_key") != instance.get("source_instance_key")
  ):
    raise ValueError("mapped endpoint differs from instance transform authority")
  return mapping, steps[0], instance


def _captured_obj_for_part(
    binding: _CaseAuthorityBinding, *, part: str
) -> CapturedFileArtifact:
  obj_payload = binding.face_payload
  # The face authority binds the OBJ extension by hash; recover the actual
  # extension payload from the already-captured case authorities.
  face_case = _authority_case(
      obj_payload, case_id=binding.case_id, label="face authority"
  )
  expected_geometry_assets = {
      row.get("geometry_asset")
      for row in face_case.get("face_mappings") or []
      if isinstance(row, Mapping) and row.get("part") == part
  }
  if len(expected_geometry_assets) != 1:
    raise ValueError("part lacks a unique OBJ geometry asset")
  # Locate the captured obj-extension payload by its formal schema.
  extension_payload: Mapping[str, Any] | None = None
  for captured in binding.captures:
    if not captured.resolved_path.name.endswith("obj_extension.v2.json"):
      continue
    parsed = _capture_json(captured.resolved_path, label="formal OBJ extension")
    if (
        isinstance(parsed.payload, Mapping)
        and parsed.payload.get("schema_version")
        == "fusion_content_receipt_obj_extension.v2"
    ):
      extension_payload = parsed.payload
      break
  if extension_payload is None:
    raise ValueError("case authority lacks captured OBJ extension")
  members = extension_payload.get("members")
  matches: list[Mapping[str, Any]] = []
  for member in members or []:
    if not isinstance(member, Mapping):
      continue
    references = member.get("references")
    if any(
        isinstance(reference, Mapping)
        and reference.get("case_id") == binding.case_id
        and reference.get("part") == part
        and reference.get("geometry_asset") in expected_geometry_assets
        for reference in references or []
    ):
      matches.append(member)
  if len(matches) != 1:
    raise ValueError("part lacks one receipt-bound authoritative OBJ member")
  member = matches[0]
  captured_obj = capture_file_artifact(
      str(member.get("restored_path") or ""), label="receipt-bound OBJ"
  )
  if (
      captured_obj.sha256 != member.get("sha256")
      or captured_obj.byte_count != member.get("bytes")
  ):
    raise ValueError("restored OBJ differs from formal extension authority")
  return captured_obj


def _verified_contact_pair(
    binding: _CaseAuthorityBinding, *, part_a: str, part_b: str
) -> Mapping[str, Any]:
  contact_case = _authority_case(
      binding.contact_payload,
      case_id=binding.case_id,
      label="contact authority",
  )
  expected = sorted((part_a, part_b))
  matches = [
      row for row in contact_case.get("pair_domain") or []
      if isinstance(row, Mapping)
      and sorted((str(row.get("part_a") or ""), str(row.get("part_b") or "")))
      == expected
  ]
  if (
      len(matches) != 1
      or matches[0].get("classification") != "verified_positive"
      or matches[0].get("terminal_status") is not True
      or type(matches[0].get("source_contact_count")) is not int
      or matches[0].get("source_contact_count") < 1
  ):
    raise ValueError("semantic endpoint pair is not contact-authority positive")
  return matches[0]


@dataclass(frozen=True)
class _ValidatedCaseGraphInputs:
  """Case-local authority inputs validated once for one materialization batch."""

  binding_sha256: str
  rows: tuple[Mapping[str, Any], ...]
  commitments: tuple[Any, ...]
  catalog: Mapping[str, Any]
  budget: GraphSizeBudget


def _validated_case_graph_inputs(
    binding: _CaseAuthorityBinding,
    *,
    catalog: Mapping[str, Any],
    budget: GraphSizeBudget,
) -> _ValidatedCaseGraphInputs:
  for captured in binding.captures:
    reverify_captured_file_artifact(captured, label="case graph authority")
  rows = tuple(_library_rows(binding.library_capture))
  commitments = binding.semantic_payload.get("row_commitments")
  if (
      not isinstance(commitments, list)
      or len(commitments) != len(rows)
  ):
    raise ValueError("case semantic rows differ from authority commitments")
  validated_catalog = _validated_program_catalog(catalog)
  if (
      len(validated_catalog["entries"])
      > budget.max_program_candidates_per_example
  ):
    raise ValueError("graph budget is smaller than the frozen program catalog")
  return _ValidatedCaseGraphInputs(
      binding_sha256=binding.binding_sha256,
      rows=rows,
      commitments=tuple(commitments),
      catalog=validated_catalog,
      budget=budget,
  )


def _materialize_graph_capability_from_case_inputs(
    binding: _CaseAuthorityBinding,
    *,
    program_index: int,
    case_inputs: _ValidatedCaseGraphInputs,
    geometry_streams: _CapturedStepStreamCache,
) -> "AuthenticatedBenchmarkV2GraphCapability":
  if case_inputs.binding_sha256 != binding.binding_sha256:
    raise ValueError("case graph inputs differ from authority binding")
  rows = case_inputs.rows
  commitments = case_inputs.commitments
  catalog = case_inputs.catalog
  budget = case_inputs.budget
  if (
      type(program_index) is not int
      or program_index < 0
      or program_index >= len(rows)
  ):
    raise ValueError("program_index is outside semantic authority")
  row = rows[program_index]
  target_entry = _catalog_entry_for_row(catalog, row)
  commitment = commitments[program_index]
  if (
      not isinstance(commitment, Mapping)
      or commitment.get("row_sha256") != _canonical_sha256(row)
      or commitment.get("program_id") != row.get("program_id")
  ):
    raise ValueError("semantic program row differs from authority commitment")
  ordinal = commitment.get("source_contact_ordinal")
  source_contact = row.get("source_contact")
  if (
      type(ordinal) is not int
      or not isinstance(source_contact, Mapping)
      or source_contact.get("contact_index") != ordinal
      or source_contact.get("direction") != commitment.get("direction")
  ):
    raise ValueError("semantic row source ordinal/direction differs")
  gold_contact = _private_gold_contact(
      binding, source_contact_ordinal=ordinal
  )
  endpoint_rows = [gold_contact.get("endpoint_a"), gold_contact.get("endpoint_b")]
  if commitment.get("direction") == "entity_two_to_entity_one":
    endpoint_rows.reverse()
  elif commitment.get("direction") != "entity_one_to_entity_two":
    raise ValueError("semantic row direction is not allowlisted")
  derived: list[dict[str, Any]] = []
  graphs: list[dict[str, Any]] = []
  private_graph_witnesses: list[dict[str, Any]] = []
  witness_audit_inputs: list[_GraphWitnessAuditInput] = []
  geometry_captures: list[CapturedFileArtifact] = []
  for graph_index, endpoint in enumerate(endpoint_rows):
    mapping, step, instance = _face_endpoint_binding(binding, endpoint=endpoint)
    step_capture = _captured_bound_file(
        step.get("file") if isinstance(step, Mapping) else None,
        label="receipt-bound STEP",
    )
    obj_capture = _captured_obj_for_part(binding, part=str(mapping["part"]))

    raw_indices = list(mapping["raw_occ_face_indices"])
    expected_signatures = list(mapping["source_face_signature_sha256s"])
    graph, raw_to_node, imported_signatures, private_witness = (
        geometry_streams.derive_private_graph(
            step_capture,
            graph_index=graph_index,
            endpoint_raw_indices=raw_indices,
            endpoint_face_signatures=expected_signatures,
            budget=budget,
        )
    )
    node_indices = sorted(raw_to_node[index] for index in raw_indices)
    graph_bytes = _canonical_bytes(graph)
    public_witness = _public_graph_witness_projection(private_witness)
    graphs.append(graph)
    private_graph_witnesses.append(private_witness)
    witness_audit_inputs.append(
        _GraphWitnessAuditInput(
            graph_index=graph_index,
            step_capture=step_capture,
            endpoint_raw_indices=tuple(raw_indices),
            endpoint_face_signatures=tuple(expected_signatures),
            budget=budget,
        )
    )
    derived.append(
        {
            "graph_index": graph_index,
            "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "graph_bytes": len(graph_bytes),
            # These bind the exact captured bytes consumed directly by OCC,
            # never a later read of the source path.
            "step_sha256": step_capture.sha256,
            "step_bytes": step_capture.byte_count,
            "obj_sha256": obj_capture.sha256,
            "obj_bytes": obj_capture.byte_count,
            "instance_transform_chain_sha256": mapping[
                "world_transform_chain_sha256"
            ],
            "instance_transform_sha256": _canonical_sha256(
                instance["world_transform_mm"]
            ),
            "fusion_face_index": mapping["fusion_face_index"],
            "mapped_occ_face_indices": raw_indices,
            "mapped_face_signature_sha256s": expected_signatures,
            "mapped_graph_node_indices": node_indices,
            **public_witness,
            "mapper_mapping_payload_sha256": mapping[
                "mapper_mapping_payload_sha256"
            ],
        }
    )
    geometry_captures.extend((step_capture, obj_capture))
  private_witness_payload = {
      "schema_version": "benchmark_v2_graph_capability_private_witness.v1",
      "case_authority_binding_sha256": binding.binding_sha256,
      "graphs": private_graph_witnesses,
  }
  contact_pair = _verified_contact_pair(
      binding,
      part_a=str(endpoint_rows[0]["part"]),
      part_b=str(endpoint_rows[1]["part"]),
  )
  endpoint_a = {"graph_index": 0, "face_indices": derived[0]["mapped_graph_node_indices"]}
  endpoint_b = {"graph_index": 1, "face_indices": derived[1]["mapped_graph_node_indices"]}
  example = {
      "query": {"endpoint_a": endpoint_a, "endpoint_b": endpoint_b},
      "program_candidates": _catalog_candidates(
          catalog, endpoint_a=endpoint_a, endpoint_b=endpoint_b
      ),
      "target": {
          "program_index": target_entry["program_index"],
          "residual_translation": row.get("residual_translation"),
          "residual_rotation_row_major": [
              value for rotation_row in row.get("residual_rotation") or []
              for value in rotation_row
          ],
      },
  }
  built = build_benchmark_v2_model_view_v2(
      graphs=graphs, examples=[example], budget=budget
  )
  receipt = {
      "schema_version": _GRAPH_AUTHORITY_RECEIPT_SCHEMA,
      "case_authority_binding_sha256": binding.binding_sha256,
      "normalization_sha256": _canonical_sha256(_NORMALIZATION_METADATA),
      "model_view_sha256": built.sha256,
      "private_witness_payload_sha256": _canonical_sha256(
          private_witness_payload
      ),
      "graphs": derived,
      "contact_authority": {
          "authority_payload_sha256": _canonical_sha256(
              dict(binding.contact_payload)
          ),
          "pair_payload_sha256": _canonical_sha256(contact_pair),
          "classification": contact_pair["classification"],
          "source_contact_ordinal": ordinal,
      },
      "semantic_authority": {
          "authority_payload_sha256": _canonical_sha256(
              dict(binding.semantic_payload)
          ),
          "program_index": program_index,
          "program_id": commitment["program_id"],
          "row_sha256": commitment["row_sha256"],
          "residual_sha256": commitment["residual_sha256"],
          "endpoint_a_sha256": commitment["endpoint_a_sha256"],
          "endpoint_b_sha256": commitment["endpoint_b_sha256"],
      },
      "program_catalog": {
          "schema_version": catalog["schema_version"],
          "catalog_payload_sha256": catalog["catalog_payload_sha256"],
          "entry_count": catalog["entry_count"],
          "target_program_index": target_entry["program_index"],
          "target_program_id": target_entry["program_id"],
          "target_descriptor_sha256": target_entry["descriptor_sha256"],
      },
      "derived_example_sha256": _canonical_sha256(
          built.model_view["examples"][0]
      ),
      "candidate_policy": "train_authority_fixed_catalog.v1",
  }
  receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
  return AuthenticatedBenchmarkV2GraphCapability(
      model_view=built.model_view,
      sha256=built.sha256,
      receipt=receipt,
      binding_sha256=binding.binding_sha256,
      catalog=catalog,
      captures=(*binding.captures, *geometry_captures),
      _factory_token=_FACTORY_TOKEN,
      witness_payload=private_witness_payload,
      witness_audit_inputs=tuple(witness_audit_inputs),
  )


def _materialize_graph_capability(
    binding: _CaseAuthorityBinding,
    *,
    program_index: int,
    catalog: Mapping[str, Any],
    budget: GraphSizeBudget,
    geometry_streams: _CapturedStepStreamCache | None = None,
) -> "AuthenticatedBenchmarkV2GraphCapability":
  """Materialize one row, optionally sharing parsed STEP geometry per case."""

  if geometry_streams is None:
    geometry_streams = _CapturedStepStreamCache()
  elif not isinstance(geometry_streams, _CapturedStepStreamCache):
    raise TypeError("shared geometry streams must be a captured STEP cache")
  case_inputs = _validated_case_graph_inputs(
      binding, catalog=catalog, budget=budget
  )
  return _materialize_graph_capability_from_case_inputs(
      binding,
      program_index=program_index,
      case_inputs=case_inputs,
      geometry_streams=geometry_streams,
  )


class AuthenticatedBenchmarkV2GraphCapability:
  """Ephemeral proof that graphs/endpoints were derived from case authority."""

  __slots__ = ("__weakref__",)

  def __init__(
      self,
      *,
      model_view: Mapping[str, Any],
      sha256: str,
      receipt: Mapping[str, Any],
      binding_sha256: str,
      catalog: Mapping[str, Any],
      captures: Sequence[CapturedFileArtifact],
      _factory_token: object,
      witness_payload: Mapping[str, Any] | None = None,
      witness_audit_inputs: Sequence[_GraphWitnessAuditInput] = (),
  ) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError("authenticated graph capability must be authority-produced")
    model_view_bytes = _canonical_bytes(model_view)
    if hashlib.sha256(model_view_bytes).hexdigest() != str(sha256):
      raise ValueError("authenticated graph capability model-view hash differs")
    receipt_bytes = _canonical_bytes(receipt)
    catalog_payload = _validated_program_catalog(catalog)
    if not isinstance(witness_payload, Mapping):
      raise ValueError("authenticated graph capability lacks private witness")
    witness_payload_bytes = _canonical_bytes(witness_payload)
    witness_payload_sha256 = hashlib.sha256(witness_payload_bytes).hexdigest()
    if (
        len(witness_audit_inputs) != 2
        or any(
            not isinstance(value, _GraphWitnessAuditInput)
            for value in witness_audit_inputs
        )
    ):
      raise ValueError("authenticated graph capability witness inputs differ")
    _register_private_state(
        self,
        _GraphCapabilityState(
            model_view_bytes=model_view_bytes,
            receipt_bytes=receipt_bytes,
            binding_sha256=str(binding_sha256),
            catalog_payload_bytes=_canonical_bytes(catalog_payload),
            captures=tuple(captures),
        ),
    )
    _register_private_witness(
        self,
        _GraphWitnessState(
            witness_payload_bytes=witness_payload_bytes,
            witness_payload_sha256=witness_payload_sha256,
            audit_inputs=tuple(witness_audit_inputs),
        ),
    )
    self.revalidate()

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated graph capability is not serializable")

  @property
  def sha256(self) -> str:
    state = _get_private_state(self, _GraphCapabilityState)
    return hashlib.sha256(state.model_view_bytes).hexdigest()

  @property
  def receipt_sha256(self) -> str:
    state = _get_private_state(self, _GraphCapabilityState)
    receipt = _strict_json_bytes(state.receipt_bytes, label="graph receipt")
    return str(receipt.get("receipt_payload_sha256") or "")

  def authority_projection(self) -> Mapping[str, Any]:
    """Return the revalidated non-model authority needed by cache producers."""

    self.revalidate()
    state = _get_private_state(self, _GraphCapabilityState)
    receipt = _strict_json_bytes(state.receipt_bytes, label="graph receipt")
    projection = {
        "case_authority_binding_sha256": receipt["case_authority_binding_sha256"],
        "contact_authority": dict(receipt["contact_authority"]),
        "semantic_authority": dict(receipt["semantic_authority"]),
        "program_catalog": dict(receipt["program_catalog"]),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }
    return MappingProxyType(projection)

  def revalidate(self) -> None:
    state = _get_private_state(self, _GraphCapabilityState)
    witness_state = _get_private_witness(self, _GraphWitnessState)
    for captured in state.captures:
      reverify_captured_file_artifact(captured, label="graph authority input")
    if (
        hashlib.sha256(witness_state.witness_payload_bytes).hexdigest()
        != witness_state.witness_payload_sha256
    ):
      raise ValueError("authenticated private graph witness hash differs")
    private_witness = _strict_json_bytes(
        witness_state.witness_payload_bytes,
        label="private full-topology/subgraph witness",
    )
    if (
        not isinstance(private_witness, Mapping)
        or private_witness.get("schema_version")
        != "benchmark_v2_graph_capability_private_witness.v1"
        or private_witness.get("case_authority_binding_sha256")
        != state.binding_sha256
        or not isinstance(private_witness.get("graphs"), list)
    ):
      raise ValueError("authenticated private graph witness binding differs")
    model_view = _strict_json_bytes(
        state.model_view_bytes, label="authenticated model view"
    )
    if not isinstance(model_view, Mapping):
      raise ValueError("authenticated model view must be an object")
    validate_benchmark_v2_model_view_v2(
        model_view, expected_sha256=self.sha256
    )
    receipt = _strict_json_bytes(state.receipt_bytes, label="graph receipt")
    if not isinstance(receipt, Mapping):
      raise ValueError("authenticated graph receipt must be an object")
    if receipt.get("schema_version") != _GRAPH_AUTHORITY_RECEIPT_SCHEMA:
      raise ValueError("authenticated graph receipt schema differs")
    unsigned = dict(receipt)
    observed = unsigned.pop("receipt_payload_sha256", None)
    if observed != self.receipt_sha256 or _canonical_sha256(unsigned) != observed:
      raise ValueError("authenticated graph receipt hash differs")
    if (
        receipt.get("private_witness_payload_sha256")
        != witness_state.witness_payload_sha256
    ):
      raise ValueError(
          "authenticated graph receipt private witness commitment differs"
      )
    if (
        receipt.get("case_authority_binding_sha256") != state.binding_sha256
        or receipt.get("model_view_sha256") != self.sha256
        or receipt.get("normalization_sha256")
        != _canonical_sha256(_NORMALIZATION_METADATA)
        or receipt.get("derived_example_sha256")
        != _canonical_sha256(model_view["examples"][0])
    ):
      raise ValueError("authenticated graph receipt binding differs")
    catalog = _strict_json_bytes(
        state.catalog_payload_bytes, label="authenticated train program catalog"
    )
    catalog = _validated_program_catalog(catalog)
    catalog_receipt = receipt.get("program_catalog")
    if (
        not isinstance(catalog_receipt, Mapping)
        or receipt.get("candidate_policy")
        != "train_authority_fixed_catalog.v1"
        or catalog_receipt.get("schema_version") != catalog["schema_version"]
        or catalog_receipt.get("catalog_payload_sha256")
        != catalog["catalog_payload_sha256"]
        or catalog_receipt.get("entry_count") != catalog["entry_count"]
    ):
      raise ValueError("authenticated graph catalog receipt differs")
    graph_rows = receipt.get("graphs")
    if not isinstance(graph_rows, list) or len(graph_rows) != len(
        model_view["graphs"]
    ):
      raise ValueError("authenticated graph receipt graph ledger differs")
    private_graphs = private_witness["graphs"]
    if len(private_graphs) != len(graph_rows):
      raise ValueError("authenticated private graph witness ledger differs")
    for index, (row, graph, private_graph_witness) in enumerate(
        zip(graph_rows, model_view["graphs"], private_graphs, strict=True)
    ):
      graph_bytes = _canonical_bytes(graph)
      local = row.get("local_interface_subgraph") if isinstance(row, Mapping) else None
      full = row.get("full_graph_authority") if isinstance(row, Mapping) else None
      shell_counts = (
          local.get("admitted_shell_face_counts")
          if isinstance(local, Mapping)
          else None
      )
      if (
          not isinstance(row, Mapping)
          or not isinstance(private_graph_witness, Mapping)
          or row.get("graph_index") != index
          or row.get("graph_sha256")
          != hashlib.sha256(graph_bytes).hexdigest()
          or row.get("graph_bytes") != len(graph_bytes)
      ):
        raise ValueError("authenticated graph bytes differ from receipt")
      expected_witness_projection = _validated_private_graph_witness_projection(
          private_graph_witness,
          graph_index=index,
          graph=graph,
      )
      if any(
          row.get(key) != value
          for key, value in expected_witness_projection.items()
      ):
        raise ValueError(
            "authenticated local-subgraph receipt differs from private witness"
        )
      private_centers = private_graph_witness.get(
          "authority_endpoint_centers"
      )
      private_step = private_graph_witness.get("captured_step")
      if (
          not isinstance(private_centers, Mapping)
          or not isinstance(private_step, Mapping)
          or row.get("mapped_occ_face_indices")
          != private_centers.get("raw_occ_face_indices")
          or row.get("mapped_face_signature_sha256s")
          != private_centers.get("source_face_signature_sha256s")
          or row.get("step_sha256") != private_step.get("sha256")
          or row.get("step_bytes") != private_step.get("bytes")
      ):
        raise ValueError("authenticated graph centers differ from private witness")
      if (
          not isinstance(local, Mapping)
          or set(local) != {
              "schema_version",
              "policy_payload_sha256",
              "effective_max_faces",
              "effective_max_edges",
              "max_hop_radius",
              "endpoint_face_count",
              "endpoint_component_face_count",
              "endpoint_component_edge_count",
              "included_face_count",
              "included_edge_count",
              "admitted_shell_face_counts",
              "achieved_hop_radius",
              "complete_component",
              "truncated",
              "truncation_reason",
              "omitted_frontier_face_count",
              "omitted_component_face_count",
          }
          or local.get("schema_version")
          != LOCAL_INTERFACE_SUBGRAPH_POLICY["schema_version"]
          or local.get("policy_payload_sha256")
          != _canonical_sha256(dict(LOCAL_INTERFACE_SUBGRAPH_POLICY))
          or local.get("max_hop_radius")
          != LOCAL_INTERFACE_SUBGRAPH_POLICY["max_hop_radius"]
          or local.get("included_face_count") != len(graph["nodes"])
          or local.get("included_edge_count") != len(graph["edges"])
          or type(local.get("endpoint_face_count")) is not int
          or local.get("endpoint_face_count") <= 0
          or type(local.get("endpoint_component_face_count")) is not int
          or local.get("endpoint_component_face_count") <= 0
          or type(local.get("endpoint_component_edge_count")) is not int
          or local.get("endpoint_component_edge_count") < 0
          or local.get("effective_max_faces")
          != model_view["budget"]["max_faces_per_graph"]
          or local.get("effective_max_edges")
          != model_view["budget"]["max_edges_per_graph"]
          or not isinstance(shell_counts, list)
          or not shell_counts
          or any(type(count) is not int or count <= 0 for count in shell_counts)
          or sum(shell_counts) != local.get("included_face_count")
          or shell_counts[0] != local.get("endpoint_face_count")
          or type(local.get("complete_component")) is not bool
          or type(local.get("truncated")) is not bool
          or local.get("complete_component") == local.get("truncated")
          or type(local.get("achieved_hop_radius")) is not int
          or local.get("achieved_hop_radius") < 0
          or local.get("achieved_hop_radius")
          > LOCAL_INTERFACE_SUBGRAPH_POLICY["max_hop_radius"]
          or len(shell_counts) != local.get("achieved_hop_radius") + 1
          or type(local.get("omitted_frontier_face_count")) is not int
          or local.get("omitted_frontier_face_count") < 0
          or type(local.get("omitted_component_face_count")) is not int
          or local.get("omitted_component_face_count") < 0
          or local.get("endpoint_component_face_count")
          != local.get("included_face_count")
          + local.get("omitted_component_face_count")
          or (
              local.get("complete_component")
              and (
                  local.get("truncation_reason") != "none"
                  or local.get("omitted_frontier_face_count") != 0
                  or local.get("omitted_component_face_count") != 0
              )
          )
          or (
              local.get("truncated")
              and (
                  local.get("truncation_reason")
                  not in {"max_hop_radius", "face_or_edge_budget"}
                  or local.get("omitted_frontier_face_count") <= 0
                  or local.get("omitted_frontier_face_count")
                  > local.get("omitted_component_face_count")
              )
          )
          or not isinstance(full, Mapping)
          or set(full)
          != {"schema_version", "face_count", "adjacency_count", "payload_sha256"}
          or full.get("schema_version")
          != "benchmark_v2_full_brep_topology_authority.v1"
          or type(full.get("face_count")) is not int
          or full.get("face_count") < len(graph["nodes"])
          or type(full.get("adjacency_count")) is not int
          or full.get("adjacency_count") < len(graph["edges"])
          or not _is_sha256(full.get("payload_sha256"))
      ):
        raise ValueError("authenticated local-subgraph receipt differs")
      mapped_nodes = row.get("mapped_graph_node_indices")
      if (
          not isinstance(mapped_nodes, list)
          or not mapped_nodes
          or any(
              type(node_index) is not int
              or node_index < 0
              or node_index >= len(graph["nodes"])
              for node_index in mapped_nodes
          )
      ):
        raise ValueError("authenticated endpoint was not preserved")
    if len(model_view["examples"]) != 1:
      raise ValueError("authenticated graph capability must contain one example")
    example = model_view["examples"][0]
    expected_endpoints = {
        "endpoint_a": {
            "graph_index": 0,
            "face_indices": graph_rows[0].get("mapped_graph_node_indices"),
        },
        "endpoint_b": {
            "graph_index": 1,
            "face_indices": graph_rows[1].get("mapped_graph_node_indices"),
        },
    }
    if example.get("query") != expected_endpoints or any(
        candidate.get("endpoint_a") != expected_endpoints["endpoint_a"]
        or candidate.get("endpoint_b") != expected_endpoints["endpoint_b"]
        for candidate in example.get("program_candidates", [])
    ):
      raise ValueError("model endpoints differ from graph producer receipt")
    expected_candidates = _catalog_candidates(
        catalog,
        endpoint_a=expected_endpoints["endpoint_a"],
        endpoint_b=expected_endpoints["endpoint_b"],
    )
    if example.get("program_candidates") != expected_candidates:
      raise ValueError("model candidates differ from authenticated train catalog")


def deep_audit_authenticated_benchmark_v2_graph_capability(
    capability: AuthenticatedBenchmarkV2GraphCapability,
) -> Mapping[str, Any]:
  """Reopen captured STEP bytes and independently recompute the private witness.

  Normal training loads use the immutable same-process witness to avoid a
  second OCC traversal.  This explicit offline audit pays that cost and proves
  the cached fact is reproducible from the receipt-bound raw geometry.
  """

  if not isinstance(capability, AuthenticatedBenchmarkV2GraphCapability):
    raise TypeError("deep graph audit requires an authenticated capability")
  capability.revalidate()
  state = _get_private_state(capability, _GraphCapabilityState)
  witness_state = _get_private_witness(capability, _GraphWitnessState)
  geometry_streams = _CapturedStepStreamCache()
  recomputed_graphs: list[dict[str, Any]] = []
  for audit_input in witness_state.audit_inputs:
    reverify_captured_file_artifact(
        audit_input.step_capture, label="deep-audit captured STEP"
    )
    shape = geometry_streams.load_step_shape(audit_input.step_capture)
    _, _, _, witness = _derive_private_graph_from_captured_shape(
        shape,
        graph_index=audit_input.graph_index,
        step_capture=audit_input.step_capture,
        endpoint_raw_indices=audit_input.endpoint_raw_indices,
        endpoint_face_signatures=audit_input.endpoint_face_signatures,
        budget=audit_input.budget,
    )
    recomputed_graphs.append(witness)
  recomputed = {
      "schema_version": "benchmark_v2_graph_capability_private_witness.v1",
      "case_authority_binding_sha256": state.binding_sha256,
      "graphs": recomputed_graphs,
  }
  recomputed_bytes = _canonical_bytes(recomputed)
  recomputed_sha256 = hashlib.sha256(recomputed_bytes).hexdigest()
  if (
      recomputed_bytes != witness_state.witness_payload_bytes
      or recomputed_sha256 != witness_state.witness_payload_sha256
  ):
    raise ValueError("deep-audit topology/subgraph witness differs")
  return MappingProxyType(
      {
          "schema_version": "benchmark_v2_graph_capability_deep_audit.v1",
          "status": "verified",
          "graph_count": len(recomputed_graphs),
          "witness_payload_sha256": recomputed_sha256,
      }
  )


class AuthenticatedBenchmarkV2DevelopmentIndex:
  """Non-serializable train/dev index with no public case or path identities."""

  __slots__ = ("__weakref__",)

  def __init__(
      self,
      *,
      bindings: Mapping[str, Sequence[_CaseAuthorityBinding]],
      catalog: Mapping[str, Any],
      root_captures: Sequence[CapturedFileArtifact],
      _factory_token: object,
  ) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError(
          "AuthenticatedBenchmarkV2DevelopmentIndex must be created by its verifier"
      )
    normalized = {
        key: tuple(value) for key, value in bindings.items()
    }
    if set(normalized) != {"train", "dev"}:
      raise ValueError("authenticated development index requires train/dev")
    catalog_payload = _validated_program_catalog(catalog)
    _register_private_state(
        self,
        _DevelopmentIndexState(
            bindings=MappingProxyType(normalized),
            catalog_payload_bytes=_canonical_bytes(catalog_payload),
            root_captures=tuple(root_captures),
        ),
    )

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated development index is not serializable")

  @property
  def split_counts(self) -> Mapping[str, int]:
    state = _get_private_state(self, _DevelopmentIndexState)
    return MappingProxyType(
        {split: len(rows) for split, rows in state.bindings.items()}
    )

  def _verified_semantic_row_counts(self) -> dict[str, tuple[int, ...]]:
    """Return only per-slot row counts after replaying the captured ledgers.

    Counts deliberately reveal neither case identity, source path, program ID,
    nor target.  They are recomputed from captured train/dev authorities on
    every call so callers cannot use a stale count ledger after file drift.
    """

    self.revalidate()
    state = _get_private_state(self, _DevelopmentIndexState)
    result: dict[str, tuple[int, ...]] = {}
    for split in ("train", "dev"):
      counts: list[int] = []
      for binding in state.bindings[split]:
        for captured in binding.captures:
          reverify_captured_file_artifact(
              captured, label="semantic-row count authority binding"
          )
        rows = _library_rows(binding.library_capture)
        commitments = binding.semantic_payload.get("row_commitments")
        if (
            not isinstance(commitments, list)
            or binding.semantic_payload.get("row_count") != len(commitments)
            or len(commitments) != len(rows)
        ):
          raise ValueError(
              "mate library and semantic authority row counts differ"
          )
        for row, commitment in zip(rows, commitments, strict=True):
          if (
              not isinstance(commitment, Mapping)
              or commitment.get("row_sha256") != _canonical_sha256(row)
              or commitment.get("program_id") != row.get("program_id")
          ):
            raise ValueError(
                "mate library row differs from semantic authority commitment"
            )
        counts.append(len(rows))
      result[split] = tuple(counts)
    return result

  @property
  def semantic_row_counts(self) -> Mapping[str, tuple[int, ...]]:
    """Read-only per-case semantic-row counts in authenticated slot order."""

    return MappingProxyType(self._verified_semantic_row_counts())

  @property
  def semantic_row_totals(self) -> Mapping[str, int]:
    """Read-only train/dev semantic-row totals derived from current evidence."""

    counts = self._verified_semantic_row_counts()
    return MappingProxyType(
        {split: sum(counts[split]) for split in ("train", "dev")}
    )

  @property
  def catalog_summary(self) -> Mapping[str, Any]:
    state = _get_private_state(self, _DevelopmentIndexState)
    catalog = _validated_program_catalog(
        _strict_json_bytes(
            state.catalog_payload_bytes, label="authenticated train program catalog"
        )
    )
    return MappingProxyType(
        {
            "entry_count": catalog["entry_count"],
            "catalog_payload_sha256": catalog["catalog_payload_sha256"],
            "source_binding_count": catalog["source_binding_count"],
            "source_row_count": catalog["source_row_count"],
            "source_split": "train",
        }
    )

  def revalidate(self) -> None:
    state = _get_private_state(self, _DevelopmentIndexState)
    for captured in state.root_captures:
      reverify_captured_file_artifact(captured, label="model-development root")

  def materialize_graph_capability(
      self,
      *,
      split: str,
      case_index: int,
      program_index: int = 0,
      budget: GraphSizeBudget = DEFAULT_GRAPH_SIZE_BUDGET,
  ) -> AuthenticatedBenchmarkV2GraphCapability:
    """Derive graphs and endpoints from authority; accepts no caller graph."""

    normalized_split = str(split).strip().lower()
    if normalized_split not in {"train", "dev"}:
      raise ValueError("v2 graph materialization is train/dev-only")
    state = _get_private_state(self, _DevelopmentIndexState)
    if type(case_index) is not int or not 0 <= case_index < len(
        state.bindings[normalized_split]
    ):
      raise ValueError("case_index is outside the authenticated split")
    budget = _validated_formal_structural_budget(budget)
    self.revalidate()
    binding = state.bindings[normalized_split][case_index]
    if binding.model_assignment_split != normalized_split:
      raise ValueError("case binding differs from authenticated assignment split")
    catalog = _validated_program_catalog(
        _strict_json_bytes(
            state.catalog_payload_bytes, label="authenticated train program catalog"
        )
    )
    return _materialize_graph_capability(
        binding,
        program_index=program_index,
        catalog=catalog,
        budget=budget,
    )

  def materialize_case_graph_capabilities(
      self,
      *,
      split: str,
      case_index: int,
      program_indices: Sequence[int],
      budget: GraphSizeBudget = DEFAULT_GRAPH_SIZE_BUDGET,
  ) -> tuple[AuthenticatedBenchmarkV2GraphCapability, ...]:
    """Materialize an ordered semantic-row subset with one case STEP cache.

    The indices must be strictly increasing.  This makes the batch API a
    deterministic optimization of the existing single-row path, rather than
    a second row-selection policy.
    """

    normalized_split = str(split).strip().lower()
    if normalized_split not in {"train", "dev"}:
      raise ValueError("v2 graph materialization is train/dev-only")
    if isinstance(program_indices, (str, bytes)) or not isinstance(
        program_indices, Sequence
    ):
      raise TypeError("program_indices must be an ordered integer sequence")
    normalized_indices = tuple(program_indices)
    if not normalized_indices:
      raise ValueError("program_indices must not be empty")
    if any(type(value) is not int for value in normalized_indices):
      raise TypeError("program_indices must contain actual integers")
    if any(
        left >= right
        for left, right in zip(
            normalized_indices, normalized_indices[1:], strict=False
        )
    ):
      raise ValueError("program_indices must be strictly increasing and unique")
    state = _get_private_state(self, _DevelopmentIndexState)
    if type(case_index) is not int or not 0 <= case_index < len(
        state.bindings[normalized_split]
    ):
      raise ValueError("case_index is outside the authenticated split")
    budget = _validated_formal_structural_budget(budget)
    self.revalidate()
    binding = state.bindings[normalized_split][case_index]
    if binding.model_assignment_split != normalized_split:
      raise ValueError("case binding differs from authenticated assignment split")
    catalog = _validated_program_catalog(
        _strict_json_bytes(
            state.catalog_payload_bytes, label="authenticated train program catalog"
        )
    )
    case_inputs = _validated_case_graph_inputs(
        binding, catalog=catalog, budget=budget
    )
    if normalized_indices[-1] >= len(case_inputs.rows):
      raise ValueError("program_index is outside semantic authority")
    geometry_streams = _CapturedStepStreamCache()
    return tuple(
        _materialize_graph_capability_from_case_inputs(
            binding,
            program_index=program_index,
            case_inputs=case_inputs,
            geometry_streams=geometry_streams,
        )
        for program_index in normalized_indices
    )

  def load_model_view(
      self,
      graph_capability: AuthenticatedBenchmarkV2GraphCapability,
      *,
      split: str,
      case_index: int,
  ) -> "AuthenticatedBenchmarkV2ModelViewV2":
    normalized_split = str(split).strip().lower()
    if normalized_split not in {"train", "dev"}:
      raise ValueError("v2 model views are available only for train or dev")
    state = _get_private_state(self, _DevelopmentIndexState)
    if type(case_index) is not int or not 0 <= case_index < len(
        state.bindings[normalized_split]
    ):
      raise ValueError("case_index is outside the authenticated split")
    if not isinstance(
        graph_capability, AuthenticatedBenchmarkV2GraphCapability
    ):
      raise TypeError(
          "real-case model views require an authenticated graph capability; "
          "bare graphs, endpoints, and JSON paths are forbidden"
      )
    self.revalidate()
    binding = state.bindings[normalized_split][case_index]
    if binding.model_assignment_split != normalized_split:
      raise ValueError("case binding differs from authenticated assignment split")
    for captured in binding.captures:
      reverify_captured_file_artifact(captured, label="case authority binding")
    graph_capability.revalidate()
    capability_state = _get_private_state(
        graph_capability, _GraphCapabilityState
    )
    if capability_state.binding_sha256 != binding.binding_sha256:
      raise ValueError("authenticated graph capability belongs to another case")
    if capability_state.catalog_payload_bytes != state.catalog_payload_bytes:
      raise ValueError("authenticated graph capability uses another program catalog")
    receipt = _strict_json_bytes(
        capability_state.receipt_bytes, label="authenticated graph receipt"
    )
    semantic_receipt = (
        receipt.get("semantic_authority") if isinstance(receipt, Mapping) else None
    )
    source_program_index = (
        semantic_receipt.get("program_index")
        if isinstance(semantic_receipt, Mapping)
        else None
    )
    if type(source_program_index) is not int:
      raise ValueError("authenticated graph receipt lacks source program index")
    payload = _strict_json_bytes(
        capability_state.model_view_bytes,
        label="authenticated graph model view",
    )
    if not isinstance(payload, Mapping):
      raise ValueError("authenticated graph model view must be an object")
    view_sha256 = validate_benchmark_v2_model_view_v2(
        payload, expected_sha256=graph_capability.sha256
    )
    catalog = _validated_program_catalog(
        _strict_json_bytes(
            state.catalog_payload_bytes, label="authenticated train program catalog"
        )
    )
    _verify_model_rows_against_semantic_authority(
        payload,
        binding=binding,
        source_program_index=source_program_index,
        catalog=catalog,
    )
    return AuthenticatedBenchmarkV2ModelViewV2(
        model_view=payload,
        sha256=view_sha256,
        graph_capability=graph_capability,
        authority_captures=(*state.root_captures, *binding.captures),
        _factory_token=_FACTORY_TOKEN,
    )


class AuthenticatedBenchmarkV2ModelViewV2:
  """Factory-token protected, immutable capability exposing only model-safe data."""

  __slots__ = ("__weakref__",)

  def __init__(
      self,
      *,
      model_view: Mapping[str, Any],
      sha256: str,
      graph_capability: AuthenticatedBenchmarkV2GraphCapability,
      authority_captures: Sequence[CapturedFileArtifact],
      _factory_token: object,
  ) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError(
          "AuthenticatedBenchmarkV2ModelViewV2 must be created by its loader"
      )
    model_view_bytes = _canonical_bytes(model_view)
    if hashlib.sha256(model_view_bytes).hexdigest() != str(sha256):
      raise ValueError("authenticated model view hash differs")
    _register_private_state(
        self,
        _ModelViewState(
            model_view_bytes=model_view_bytes,
            graph_capability=graph_capability,
            authority_captures=tuple(authority_captures),
        ),
    )

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated benchmark-v2 model view is not serializable")

  @property
  def model_view(self) -> dict[str, Any]:
    self.revalidate()
    state = _get_private_state(self, _ModelViewState)
    payload = _strict_json_bytes(state.model_view_bytes, label="model view")
    if not isinstance(payload, dict):
      raise ValueError("model view must be an object")
    return payload

  @property
  def sha256(self) -> str:
    state = _get_private_state(self, _ModelViewState)
    return hashlib.sha256(state.model_view_bytes).hexdigest()

  def revalidate(self) -> None:
    state = _get_private_state(self, _ModelViewState)
    if not isinstance(
        state.graph_capability, AuthenticatedBenchmarkV2GraphCapability
    ):
      raise TypeError("model view lacks authenticated graph capability")
    state.graph_capability.revalidate()
    graph_state = _get_private_state(
        state.graph_capability, _GraphCapabilityState
    )
    if state.model_view_bytes != graph_state.model_view_bytes:
      raise ValueError("model view differs from authenticated graph capability")
    for captured in state.authority_captures:
      reverify_captured_file_artifact(captured, label="case authority binding")
    payload = _strict_json_bytes(state.model_view_bytes, label="model view")
    if not isinstance(payload, Mapping):
      raise ValueError("model view must be an object")
    validate_benchmark_v2_model_view_v2(
        payload, expected_sha256=self.sha256
    )


def _library_rows(captured: CapturedFileArtifact) -> list[Mapping[str, Any]]:
  rows: list[Mapping[str, Any]] = []
  for line_index, raw_line in enumerate(captured.raw_bytes.splitlines()):
    if not raw_line.strip():
      raise ValueError("formal mate library contains a blank JSONL row")
    try:
      row = json.loads(
          raw_line.decode("utf-8"), object_pairs_hook=_strict_json_object
      )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
      raise ValueError("formal mate library contains invalid strict JSONL") from error
    if not isinstance(row, Mapping):
      raise ValueError(f"formal mate library row {line_index} is not an object")
    rows.append(row)
  return rows


def _catalog_descriptor_from_authority_row(
    row: Mapping[str, Any],
) -> dict[str, str]:
  descriptor: dict[str, str] = {
      "relation_hint": str(row.get("relation_hint") or "").strip().lower(),
      "contact_type": str(row.get("contact_type") or "").strip().lower(),
  }
  for endpoint_key, descriptor_key in (
      ("interface_a", "surface_type_a"),
      ("interface_b", "surface_type_b"),
  ):
    endpoint = row.get(endpoint_key)
    model_view = (
        endpoint.get("benchmark_v2_model_view")
        if isinstance(endpoint, Mapping)
        else None
    )
    descriptor[descriptor_key] = str(
        model_view.get("surface_type") if isinstance(model_view, Mapping) else ""
    ).strip().lower()
  return _validated_catalog_descriptor(
      descriptor, label="authority program catalog descriptor"
  )


def _build_train_program_catalog(
    train_bindings: Sequence[_CaseAuthorityBinding],
) -> dict[str, Any]:
  """Freeze a deterministic candidate catalog from authenticated train rows.

  No dev binding, target index, family identity, path, or residual participates
  in candidate construction or ordering.  Continuous residuals remain
  program-local supervision selected after this catalog is frozen.
  """

  descriptors: dict[str, dict[str, str]] = {}
  source_rows: list[dict[str, Any]] = []
  for binding in train_bindings:
    if binding.model_assignment_split != "train":
      raise ValueError("program catalog accepts authenticated train bindings only")
    rows = _library_rows(binding.library_capture)
    commitments = binding.semantic_payload.get("row_commitments")
    if not isinstance(commitments, list) or len(commitments) != len(rows):
      raise ValueError("train semantic row ledger is malformed")
    for source_program_index, (row, commitment) in enumerate(
        zip(rows, commitments, strict=True)
    ):
      if (
          not isinstance(commitment, Mapping)
          or commitment.get("row_sha256") != _canonical_sha256(row)
          or commitment.get("program_id") != row.get("program_id")
      ):
        raise ValueError("train program row differs from semantic commitment")
      descriptor = _catalog_descriptor_from_authority_row(row)
      descriptor_sha256 = _canonical_sha256(descriptor)
      descriptors.setdefault(descriptor_sha256, descriptor)
      source_rows.append(
          {
              "case_authority_binding_sha256": binding.binding_sha256,
              "source_program_index": source_program_index,
              "semantic_row_sha256": commitment["row_sha256"],
              "descriptor_sha256": descriptor_sha256,
          }
      )
  entries: list[dict[str, Any]] = []
  seen_ids: set[str] = set()
  for descriptor_sha256, descriptor in sorted(descriptors.items()):
    program_id = f"catalog_{descriptor_sha256[:16]}"
    if program_id in seen_ids:
      raise ValueError("program catalog identifier collision")
    seen_ids.add(program_id)
    entries.append(
        {
            "program_index": len(entries),
            "program_id": program_id,
            "program_type": descriptor["relation_hint"],
            "descriptor": descriptor,
            "descriptor_sha256": descriptor_sha256,
        }
    )
  if len(entries) < 2:
    raise ValueError("train authority program catalog is degenerate")
  if len(entries) > HARD_MAX_GRAPH_SIZE_BUDGET.max_program_candidates_per_example:
    raise ValueError("train authority program catalog exceeds candidate hard limit")
  payload = {
      "schema_version": _PROGRAM_CATALOG_SCHEMA,
      "source_split": "train",
      "candidate_policy": "train_authority_fixed_catalog.v1",
      "candidate_order": "descriptor_sha256_ascending.v1",
      "target_policy": "catalog_index_or_explicit_oov_fail_closed.v1",
      "entry_count": len(entries),
      "source_binding_count": len(train_bindings),
      "source_row_count": len(source_rows),
      "source_commitment_sha256": _canonical_sha256(source_rows),
      "entries": entries,
  }
  payload["catalog_payload_sha256"] = _canonical_sha256(payload)
  return payload


def _validated_program_catalog(payload: Any) -> dict[str, Any]:
  if not isinstance(payload, Mapping):
    raise ValueError("train program catalog must be an object")
  _verify_self_hash(
      payload, field="catalog_payload_sha256", label="train program catalog"
  )
  if set(payload) != {
      "schema_version",
      "source_split",
      "candidate_policy",
      "candidate_order",
      "target_policy",
      "entry_count",
      "source_binding_count",
      "source_row_count",
      "source_commitment_sha256",
      "entries",
      "catalog_payload_sha256",
  }:
    raise ValueError("train program catalog schema differs")
  entries = payload.get("entries")
  if (
      payload.get("schema_version") != _PROGRAM_CATALOG_SCHEMA
      or payload.get("source_split") != "train"
      or payload.get("candidate_policy") != "train_authority_fixed_catalog.v1"
      or payload.get("candidate_order") != "descriptor_sha256_ascending.v1"
      or payload.get("target_policy")
      != "catalog_index_or_explicit_oov_fail_closed.v1"
      or not isinstance(entries, list)
      or payload.get("entry_count") != len(entries)
      or type(payload.get("source_binding_count")) is not int
      or payload.get("source_binding_count") < 1
      or type(payload.get("source_row_count")) is not int
      or payload.get("source_row_count") < len(entries)
      or re.fullmatch(
          r"[0-9a-f]{64}", str(payload.get("source_commitment_sha256") or "")
      ) is None
      or len(entries) < 2
      or len(entries)
      > HARD_MAX_GRAPH_SIZE_BUDGET.max_program_candidates_per_example
  ):
    raise ValueError("train program catalog policy differs")
  validated_entries: list[dict[str, Any]] = []
  descriptor_hashes: list[str] = []
  program_ids: set[str] = set()
  for index, raw_entry in enumerate(entries):
    entry = _exact_keys(
        raw_entry,
        {
            "program_index",
            "program_id",
            "program_type",
            "descriptor",
            "descriptor_sha256",
        },
        label=f"program_catalog.entry[{index}]",
    )
    descriptor = _validated_catalog_descriptor(
        entry.get("descriptor"), label=f"program_catalog.entry[{index}].descriptor"
    )
    descriptor_sha256 = _canonical_sha256(descriptor)
    program_id = str(entry.get("program_id") or "")
    if (
        entry.get("program_index") != index
        or entry.get("program_type") != descriptor["relation_hint"]
        or entry.get("descriptor_sha256") != descriptor_sha256
        or program_id != f"catalog_{descriptor_sha256[:16]}"
        or program_id in program_ids
    ):
      raise ValueError("train program catalog entry differs")
    descriptor_hashes.append(descriptor_sha256)
    program_ids.add(program_id)
    validated_entries.append(
        {
            "program_index": index,
            "program_id": program_id,
            "program_type": descriptor["relation_hint"],
            "descriptor": descriptor,
            "descriptor_sha256": descriptor_sha256,
        }
    )
  if descriptor_hashes != sorted(set(descriptor_hashes)):
    raise ValueError("train program catalog order or uniqueness differs")
  normalized = dict(payload)
  normalized["entries"] = validated_entries
  return normalized


def _catalog_entry_for_row(
    catalog: Mapping[str, Any], row: Mapping[str, Any]
) -> Mapping[str, Any]:
  descriptor_sha256 = _canonical_sha256(
      _catalog_descriptor_from_authority_row(row)
  )
  matches = [
      entry for entry in catalog.get("entries") or []
      if isinstance(entry, Mapping)
      and entry.get("descriptor_sha256") == descriptor_sha256
  ]
  if len(matches) != 1:
    raise ValueError(
        "authority program is OOV in the frozen train-only catalog; gold "
        "injection is forbidden"
    )
  return matches[0]


def _catalog_candidates(
    catalog: Mapping[str, Any],
    *,
    endpoint_a: Mapping[str, Any],
    endpoint_b: Mapping[str, Any],
) -> list[dict[str, Any]]:
  return [
      {
          "program_index": entry["program_index"],
          "program_id": entry["program_id"],
          "program_type": entry["program_type"],
          "descriptor": copy.deepcopy(entry["descriptor"]),
          "endpoint_a": dict(endpoint_a),
          "endpoint_b": dict(endpoint_b),
      }
      for entry in catalog["entries"]
  ]


def _verify_model_rows_against_semantic_authority(
    model_view: Mapping[str, Any],
    *,
    binding: _CaseAuthorityBinding,
    source_program_index: int,
    catalog: Mapping[str, Any],
) -> None:
  semantic = binding.semantic_payload
  commitments = semantic.get("row_commitments")
  if (
      not isinstance(commitments, list)
      or semantic.get("row_count") != len(commitments)
  ):
    raise ValueError("mate semantic authority row ledger is malformed")
  library_rows = _library_rows(binding.library_capture)
  if len(library_rows) != len(commitments):
    raise ValueError("mate library and semantic authority row counts differ")
  if not 0 <= source_program_index < len(library_rows):
    raise ValueError("source program index is outside semantic authority")
  source_row = library_rows[source_program_index]
  if commitments[source_program_index].get("row_sha256") != _canonical_sha256(
      source_row
  ):
    raise ValueError("mate library source row differs from semantic authority")
  target_entry = _catalog_entry_for_row(catalog, source_row)
  for example in model_view["examples"]:
    expected_candidates = _catalog_candidates(
        catalog,
        endpoint_a=example["query"]["endpoint_a"],
        endpoint_b=example["query"]["endpoint_b"],
    )
    if example["program_candidates"] != expected_candidates:
      raise ValueError("program candidates differ from frozen train-only catalog")
    target = example["target"]
    if target.get("program_index") != target_entry["program_index"]:
      raise ValueError("target index differs from frozen train-only catalog")
    expected_translation = _normalized_vector(
        source_row.get("residual_translation"),
        length=3,
        label="semantic residual_translation",
    )
    raw_rotation = source_row.get("residual_rotation")
    if (
        not isinstance(raw_rotation, list)
        or len(raw_rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in raw_rotation)
    ):
      raise ValueError("semantic residual_rotation is malformed")
    expected_rotation = _validated_rotation(
        [value for row in raw_rotation for value in row],
        label="semantic residual_rotation",
    )
    if (
        target["residual_translation"] != expected_translation
        or target["residual_rotation_row_major"] != expected_rotation
    ):
      raise ValueError("program-local residual target differs from semantic row")


def _verify_trusted_freeze_for_projection(
    freeze_directory: str | Path,
) -> tuple[
    dict[str, list[Mapping[str, Any]]],
    dict[str, list[_CaseAuthorityBinding]],
    dict[str, Mapping[str, Any]],
]:
  """Offline-only verification of the fixed v3 freeze and merged pool.

  This is the only path that opens the 700-row pool.  It is used to emit the
  fixed 600-row train/dev projection, never by the training-side loader.
  """

  root = Path(freeze_directory)
  manifest_capture = _capture_json(
      root / "model_development_freeze_manifest.json",
      label="model-development freeze manifest",
  )
  manifest = manifest_capture.payload
  if not isinstance(manifest, Mapping):
    raise ValueError("model-development freeze manifest must be an object")
  _verify_self_hash(
      manifest,
      field="manifest_payload_sha256",
      label="model-development freeze manifest",
  )
  _require_development_flags(manifest, label="model-development freeze manifest")
  final_test = manifest.get("final_test")
  input_bindings = manifest.get("input_bindings")
  output_bindings = manifest.get("output_bindings")
  policy_identity = manifest.get("policy_identity")
  code_identity = manifest.get("code_identity")
  if any(
      not isinstance(value, Mapping)
      for value in (
          final_test,
          input_bindings,
          output_bindings,
          policy_identity,
          code_identity,
      )
  ):
    raise ValueError("freeze manifest trust-root objects are malformed")
  if (
      manifest.get("schema_version") != _FREEZE_MANIFEST_SCHEMA
      or manifest.get("formal_authority_backed") is not True
      or manifest.get("formal_split_created") is not False
      or manifest.get("cryptographically_sealed") is not False
      or manifest.get("scope") != "model_development_train_dev_assignment_only"
      or final_test.get("generated") is not False
      or final_test.get("touched") is not False
  ):
    raise ValueError("freeze manifest is not train/dev-only and unsealed")
  dependencies = code_identity.get("dependencies")
  if not isinstance(dependencies, Mapping):
    raise ValueError("freeze manifest code dependencies are malformed")
  observed_root = {
      "schema_version": "benchmark_v2_development_trust_root.v1",
      "freeze_manifest_schema": manifest.get("schema_version"),
      "freeze_manifest_file_sha256": manifest_capture.file.sha256,
      "freeze_manifest_payload_sha256": manifest.get(
          "manifest_payload_sha256"
      ),
      "policy_payload_sha256": policy_identity.get("payload_sha256"),
      "authority_pool_file_sha256": input_bindings.get(
          "authority_ready_pool", {}
      ).get("sha256"),
      "authority_pool_merge_receipt_file_sha256": input_bindings.get(
          "authority_ready_pool_merge_receipt", {}
      ).get("sha256"),
      "audit_file_sha256": output_bindings.get(
          "audit", {}
      ).get("sha256"),
      "train_assignment_file_sha256": output_bindings.get(
          "train", {}
      ).get("sha256"),
      "dev_assignment_file_sha256": output_bindings.get(
          "dev", {}
      ).get("sha256"),
      "freeze_producer_code_sha256": code_identity.get(
          "producer", {}
      ).get("sha256"),
      "family_split_builder_code_sha256": dependencies.get(
          "family_split_builder", {}
      ).get("sha256"),
      "immutable_writer_code_sha256": dependencies.get(
          "immutable_writer", {}
      ).get("sha256"),
      "projection_schema": _PROJECTION_SCHEMA,
      "projection_producer_identity_sha256": _PROJECTION_PRODUCER_IDENTITY,
      "status": "development_unsealed_nonpublication",
  }
  if observed_root != dict(_DEVELOPMENT_TRUST_ROOT):
    raise ValueError("freeze root is not in the development trust-root allowlist")
  pool_capture = _capture_json(
      str(input_bindings.get("authority_ready_pool", {}).get("path") or ""),
      label="authority-ready pool",
  )
  _verify_binding(
      input_bindings.get("authority_ready_pool"),
      pool_capture.file,
      label="freeze authority-ready pool",
  )
  if not isinstance(pool_capture.payload, Mapping):
    raise ValueError("authority-ready pool must be an object")
  if pool_capture.file.sha256 != _DEVELOPMENT_TRUST_ROOT[
      "authority_pool_file_sha256"
  ]:
    raise ValueError("merged authority pool is outside the trust-root allowlist")
  records = _pool_records(pool_capture.payload)

  audit_capture = _capture_json(
      root / "leakage_assignment_audit.json", label="leakage assignment audit"
  )
  audit = audit_capture.payload
  if not isinstance(audit, Mapping):
    raise ValueError("leakage assignment audit must be an object")
  _verify_binding(
      manifest.get("output_bindings", {}).get("audit"),
      audit_capture.file,
      label="manifest leakage audit",
  )
  _verify_self_hash(audit, field="audit_payload_sha256", label="leakage audit")
  if audit_capture.file.sha256 != _DEVELOPMENT_TRUST_ROOT[
      "audit_file_sha256"
  ]:
    raise ValueError("leakage audit is outside the trust-root allowlist")
  _require_development_flags(audit, label="leakage audit")
  if (
      audit.get("schema_version") != _LEAKAGE_AUDIT_SCHEMA
      or audit.get("zero_cross_split_connected_component_leakage") is not True
      or any(audit.get("pairwise_case_overlap_counts", {}).values())
      or any(audit.get("pairwise_component_overlap_counts", {}).values())
      or audit.get("withheld_disclosure", {}).get(
          "plaintext_ids_or_rows_emitted"
      ) is not False
      or audit.get("withheld_disclosure", {}).get("not_a_final_test") is not True
  ):
    raise ValueError("leakage audit does not authorize isolated train/dev use")

  assignments: dict[str, list[Mapping[str, Any]]] = {}
  bindings: dict[str, list[_CaseAuthorityBinding]] = {}
  for split in ("train", "dev"):
    assignment_capture = _capture_json(
        root / f"{split}_assignment.json", label=f"{split} assignment"
    )
    if assignment_capture.file.sha256 != _DEVELOPMENT_TRUST_ROOT[
        f"{split}_assignment_file_sha256"
    ]:
      raise ValueError(f"{split} assignment is outside the trust-root allowlist")
    split_rows, split_bindings = _assignment_rows(
        assignment_capture,
        split=split,
        manifest=manifest,
        pool_capture=pool_capture,
        pool_records=records,
    )
    assignments[split] = split_rows
    bindings[split] = split_bindings
  train_cases = {str(row["case_id"]) for row in assignments["train"]}
  dev_cases = {str(row["case_id"]) for row in assignments["dev"]}
  train_components = {
      str(row["leakage_component_id"]) for row in assignments["train"]
  }
  dev_components = {
      str(row["leakage_component_id"]) for row in assignments["dev"]
  }
  if train_cases & dev_cases or train_components & dev_components:
    raise ValueError("train/dev assignments overlap after authenticated loading")
  return assignments, bindings, records


def _write_exclusive(path: Path, raw_bytes: bytes) -> None:
  descriptor = os.open(
      path,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
      0o444,
  )
  try:
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
      stream.write(raw_bytes)
      stream.flush()
      os.fsync(stream.fileno())
  except BaseException:
    try:
      path.unlink()
    except OSError:
      pass
    raise


def _projection_content_binding(
    raw_bytes: bytes, *, name: str, schema_version: str
) -> dict[str, Any]:
  return {
      "name": name,
      "sha256": hashlib.sha256(raw_bytes).hexdigest(),
      "bytes": len(raw_bytes),
      "schema_version": schema_version,
  }


def _verify_projection_content_binding(
    binding: Any,
    captured: CapturedFileArtifact,
    *,
    expected_name: str,
    expected_schema_version: str,
    label: str,
) -> None:
  if (
      not isinstance(binding, Mapping)
      or set(binding) != {"name", "sha256", "bytes", "schema_version"}
      or binding.get("name") != expected_name
      or binding.get("schema_version") != expected_schema_version
      or binding.get("sha256") != captured.sha256
      or binding.get("bytes") != captured.byte_count
  ):
    raise ValueError(f"{label} content binding differs")


def _validate_projection_pair_bytes(
    projection_bytes: bytes, receipt_bytes: bytes
) -> None:
  projection = _strict_json_bytes(
      projection_bytes, label="model-development authority projection"
  )
  receipt = _strict_json_bytes(
      receipt_bytes, label="model-development authority projection receipt"
  )
  if not isinstance(projection, Mapping) or not isinstance(receipt, Mapping):
    raise ValueError("development projection artifacts must be objects")
  _verify_self_hash(
      projection,
      field="projection_payload_sha256",
      label="model-development authority projection",
  )
  _verify_self_hash(
      receipt,
      field="receipt_payload_sha256",
      label="model-development authority projection receipt",
  )
  _require_development_flags(projection, label="development projection")
  _require_development_flags(receipt, label="development projection receipt")
  if set(projection) != {
      "schema_version",
      "development",
      "publication_eligible",
      "cryptographically_sealed",
      "final_test_touched",
      "scope",
      "trust_root",
      "counts",
      "projection_producer_identity_sha256",
      "graphs_materialized",
      "program_catalog",
      "records",
      "projection_payload_sha256",
  } or set(receipt) != {
      "schema_version",
      "development",
      "publication_eligible",
      "cryptographically_sealed",
      "final_test_touched",
      "trust_root",
      "projection",
      "program_catalog",
      "counts",
      "graphs_materialized",
      "receipt_payload_sha256",
  }:
    raise ValueError("development projection pair schema differs")
  if (
      projection.get("schema_version") != _PROJECTION_SCHEMA
      or receipt.get("schema_version") != _PROJECTION_RECEIPT_SCHEMA
      or projection.get("cryptographically_sealed") is not False
      or receipt.get("cryptographically_sealed") is not False
      or projection.get("graphs_materialized") is not False
      or receipt.get("graphs_materialized") is not False
      or projection.get("scope")
      != "model_development_train_dev_authority_projection_only"
      or projection.get("trust_root") != dict(_DEVELOPMENT_TRUST_ROOT)
      or receipt.get("trust_root") != dict(_DEVELOPMENT_TRUST_ROOT)
      or projection.get("projection_producer_identity_sha256")
      != _PROJECTION_PRODUCER_IDENTITY
      or projection.get("counts") != {"train": 500, "dev": 100}
      or receipt.get("counts") != {"train": 500, "dev": 100}
  ):
    raise ValueError("development projection pair policy differs")
  catalog = _validated_program_catalog(projection.get("program_catalog"))
  records = projection.get("records")
  if not isinstance(records, Mapping) or set(records) != {"train", "dev"}:
    raise ValueError("development projection records are malformed")
  for split, expected_count in (("train", 500), ("dev", 100)):
    rows = records.get(split)
    if not isinstance(rows, list) or len(rows) != expected_count:
      raise ValueError("development projection record count differs")
    for slot_index, row in enumerate(rows):
      if (
          not isinstance(row, Mapping)
          or set(row)
          != {
              "slot_index",
              "authority_binding_sha256",
              "authority_record",
              "projection_record_payload_sha256",
          }
          or row.get("slot_index") != slot_index
      ):
        raise ValueError("development projection record schema differs")
      _verify_self_hash(
          row,
          field="projection_record_payload_sha256",
          label="development projection record",
      )
  expected_binding = _projection_content_binding(
      projection_bytes,
      name="development_authority_projection.json",
      schema_version=_PROJECTION_SCHEMA,
  )
  if (
      receipt.get("projection") != expected_binding
      or receipt.get("program_catalog")
      != {
          "schema_version": _PROGRAM_CATALOG_SCHEMA,
          "catalog_payload_sha256": catalog["catalog_payload_sha256"],
          "entry_count": catalog["entry_count"],
          "source_split": "train",
      }
  ):
    raise ValueError("development projection receipt binding differs")


def _fsync_directory(path: Path) -> None:
  """Durably publish a directory entry where the platform permits it."""

  try:
    descriptor = os.open(path, os.O_RDONLY)
  except OSError:
    # Windows does not expose directory fsync through ordinary file handles.
    return
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _publish_projection_directory(
    output: Path, projection_bytes: bytes, receipt_bytes: bytes
) -> tuple[Path, Path]:
  """Publish the validated pair by one sibling-directory rename."""

  parent = output.parent
  parent.mkdir(parents=True, exist_ok=True)
  if output.exists():
    raise FileExistsError(f"projection output already exists: {output}")
  temporary = Path(
      tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(parent))
  )
  try:
    projection_path = temporary / "development_authority_projection.json"
    receipt_path = temporary / "development_authority_projection.receipt.json"
    _write_exclusive(projection_path, projection_bytes)
    _write_exclusive(receipt_path, receipt_bytes)
    _validate_projection_pair_bytes(
        projection_path.read_bytes(), receipt_path.read_bytes()
    )
    _fsync_directory(temporary)
    if output.exists():
      raise FileExistsError(f"projection output already exists: {output}")
    os.rename(temporary, output)
    _fsync_directory(parent)
  except BaseException:
    def clear_readonly_and_retry(
        operation: Any, failed_path: str, _error: Any
    ) -> None:
      try:
        os.chmod(failed_path, 0o700)
        operation(failed_path)
      except OSError:
        pass

    shutil.rmtree(temporary, onerror=clear_readonly_and_retry)
    raise
  return (
      output / "development_authority_projection.json",
      output / "development_authority_projection.receipt.json",
  )


def produce_benchmark_v2_development_authority_projection(
    freeze_directory: str | Path,
    output_directory: str | Path,
) -> tuple[Path, Path]:
  """Offline-produce the fixed 600-row authority projection, without graphs.

  This producer is development-only and unsealed.  It verifies the exact v3
  freeze and 700-row merged pool, then emits only the 500 train and 100 dev
  authority bindings.  The held-out complement is inspected only inside this
  offline verification step and is never emitted into the projection.
  """

  assignments, bindings, pool_records = _verify_trusted_freeze_for_projection(
      freeze_directory
  )
  catalog = _build_train_program_catalog(bindings["train"])
  projected: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
  for split in ("train", "dev"):
    for slot_index, (assignment, authority) in enumerate(
        zip(assignments[split], bindings[split], strict=True)
    ):
      case_id = str(assignment.get("case_id") or "")
      source = pool_records.get(case_id)
      if (
          not isinstance(source, Mapping)
          or authority.case_id != case_id
          or authority.model_assignment_split != split
      ):
        raise ValueError("projection join differs from verified assignment")
      authority_record = {
          "case_id": case_id,
          "source_ordinal": authority.source_ordinal,
          "stage_receipts": copy.deepcopy(source.get("stage_receipts")),
      }
      row = {
          "slot_index": slot_index,
          "authority_binding_sha256": authority.binding_sha256,
          "authority_record": authority_record,
      }
      row["projection_record_payload_sha256"] = _canonical_sha256(row)
      projected[split].append(row)
  projection = {
      "schema_version": _PROJECTION_SCHEMA,
      "development": True,
      "publication_eligible": False,
      "cryptographically_sealed": False,
      "final_test_touched": False,
      "scope": "model_development_train_dev_authority_projection_only",
      "trust_root": dict(_DEVELOPMENT_TRUST_ROOT),
      "counts": {"train": 500, "dev": 100},
      "projection_producer_identity_sha256": _PROJECTION_PRODUCER_IDENTITY,
      "graphs_materialized": False,
      "program_catalog": catalog,
      "records": projected,
  }
  projection["projection_payload_sha256"] = _canonical_sha256(projection)
  projection_bytes = _canonical_bytes(projection)
  receipt = {
      "schema_version": _PROJECTION_RECEIPT_SCHEMA,
      "development": True,
      "publication_eligible": False,
      "cryptographically_sealed": False,
      "final_test_touched": False,
      "trust_root": dict(_DEVELOPMENT_TRUST_ROOT),
      "projection": _projection_content_binding(
          projection_bytes,
          name="development_authority_projection.json",
          schema_version=_PROJECTION_SCHEMA,
      ),
      "program_catalog": {
          "schema_version": _PROGRAM_CATALOG_SCHEMA,
          "catalog_payload_sha256": catalog["catalog_payload_sha256"],
          "entry_count": catalog["entry_count"],
          "source_split": "train",
      },
      "counts": {"train": 500, "dev": 100},
      "graphs_materialized": False,
  }
  receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
  return _publish_projection_directory(
      Path(output_directory), projection_bytes, _canonical_bytes(receipt)
  )


def load_authenticated_benchmark_v2_development_index(
    projection_directory: str | Path,
) -> AuthenticatedBenchmarkV2DevelopmentIndex:
  """Load the fixed 600-row projection without opening the 700-row pool."""

  root = Path(projection_directory)
  projection_capture = _capture_json(
      root / "development_authority_projection.json",
      label="model-development authority projection",
  )
  receipt_capture = _capture_json(
      root / "development_authority_projection.receipt.json",
      label="model-development authority projection receipt",
  )
  if (
      projection_capture.file.sha256 != _TRUSTED_PROJECTION_FILE_SHA256
      or receipt_capture.file.sha256
      != _TRUSTED_PROJECTION_RECEIPT_FILE_SHA256
  ):
    raise ValueError("projection root is outside the development trust allowlist")
  projection = projection_capture.payload
  receipt = receipt_capture.payload
  if not isinstance(projection, Mapping) or not isinstance(receipt, Mapping):
    raise ValueError("development projection artifacts must be objects")
  _validate_projection_pair_bytes(
      projection_capture.file.raw_bytes, receipt_capture.file.raw_bytes
  )
  _verify_self_hash(
      projection,
      field="projection_payload_sha256",
      label="model-development authority projection",
  )
  _verify_self_hash(
      receipt,
      field="receipt_payload_sha256",
      label="model-development authority projection receipt",
  )
  _require_development_flags(projection, label="development projection")
  _require_development_flags(receipt, label="development projection receipt")
  if (
      projection.get("schema_version") != _PROJECTION_SCHEMA
      or receipt.get("schema_version") != _PROJECTION_RECEIPT_SCHEMA
      or projection.get("cryptographically_sealed") is not False
      or receipt.get("cryptographically_sealed") is not False
      or projection.get("graphs_materialized") is not False
      or receipt.get("graphs_materialized") is not False
      or projection.get("scope")
      != "model_development_train_dev_authority_projection_only"
      or projection.get("trust_root") != dict(_DEVELOPMENT_TRUST_ROOT)
      or receipt.get("trust_root") != dict(_DEVELOPMENT_TRUST_ROOT)
      or projection.get("projection_producer_identity_sha256")
      != _PROJECTION_PRODUCER_IDENTITY
      or projection.get("counts") != {"train": 500, "dev": 100}
      or receipt.get("counts") != {"train": 500, "dev": 100}
  ):
    raise ValueError("development projection policy or trust root differs")
  _verify_projection_content_binding(
      receipt.get("projection"),
      projection_capture.file,
      expected_name="development_authority_projection.json",
      expected_schema_version=_PROJECTION_SCHEMA,
      label="development authority projection receipt",
  )
  catalog = _validated_program_catalog(projection.get("program_catalog"))
  records = projection.get("records")
  if not isinstance(records, Mapping) or set(records) != {"train", "dev"}:
    raise ValueError("development projection must contain only train/dev records")
  bindings: dict[str, list[_CaseAuthorityBinding]] = {"train": [], "dev": []}
  seen_case_ids: set[str] = set()
  for split, expected_count in (("train", 500), ("dev", 100)):
    rows = records.get(split)
    if not isinstance(rows, list) or len(rows) != expected_count:
      raise ValueError(f"{split} projection count differs")
    for slot_index, row in enumerate(rows):
      if not isinstance(row, Mapping) or set(row) != {
          "slot_index",
          "authority_binding_sha256",
          "authority_record",
          "projection_record_payload_sha256",
      }:
        raise ValueError("projection record schema differs")
      _verify_self_hash(
          row,
          field="projection_record_payload_sha256",
          label="development projection record",
      )
      authority_record = row.get("authority_record")
      if (
          row.get("slot_index") != slot_index
          or not isinstance(authority_record, Mapping)
          or set(authority_record) != {
              "case_id", "source_ordinal", "stage_receipts"
          }
      ):
        raise ValueError("development projection record ordering differs")
      case_id = str(authority_record.get("case_id") or "")
      if not case_id or case_id in seen_case_ids:
        raise ValueError("development projection contains duplicate case authority")
      seen_case_ids.add(case_id)
      authority = _captured_stage_authorities(
          authority_record, case_id=case_id
      )
      if authority.binding_sha256 != row.get("authority_binding_sha256"):
        raise ValueError("projected case authority binding differs")
      bindings[split].append(
          _CaseAuthorityBinding(
              authority_production_split=authority.authority_production_split,
              model_assignment_split=split,
              source_ordinal=authority.source_ordinal,
              case_id=authority.case_id,
              binding_sha256=authority.binding_sha256,
              face_payload=authority.face_payload,
              contact_payload=authority.contact_payload,
              semantic_payload=authority.semantic_payload,
              private_gold_payload=authority.private_gold_payload,
              library_capture=authority.library_capture,
              captures=authority.captures,
          )
      )
  recomputed_catalog = _build_train_program_catalog(bindings["train"])
  if recomputed_catalog != catalog:
    raise ValueError(
        "projected program catalog differs from authenticated train authority"
    )
  return AuthenticatedBenchmarkV2DevelopmentIndex(
      bindings=bindings,
      catalog=catalog,
      root_captures=(projection_capture.file, receipt_capture.file),
      _factory_token=_FACTORY_TOKEN,
  )
