"""Supervised, killable OCC preprocessing for constraint-manifold V4.

The parent never imports a STEP file and never receives filesystem paths or
source/gold fields from the child.  A one-shot child owns STEP replay, absolute
meshing, selected-face signature replay, trim/topology extraction, landmark
construction, and the complete cheap candidate domain.  The parent applies a
single wall-clock deadline to spawn, communication, kill/wait, and parsing and
turns every started subprocess path into a nonce-bound terminal receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
from weakref import WeakKeyDictionary

import numpy as np
from numba import njit

from . import constraint_manifold_solver_v1 as _v1
from .cadquery_backend import load_step_shape, shape_bbox, source_face_signature_sha256
from .constraint_manifold_occ_adapter_v3 import extract_occ_manifold_face_v3
from .constraint_manifold_factorized_p11_v2 import (
    P11_FACTORIZED_DOMAIN_SCHEMA_V2, P11_FACTORIZED_STRUCTURAL_TOPK_SCHEMA_V2,
    canonical_yaw_v2,
    commit_p11_factorized_domain_v2, select_p11_factorized_topk_v2,
    multiply_se3_row_major_v2,
    verify_p11_factorized_domain_v2, verify_p11_factorized_selection_v2,
)
from .constraint_manifold_p9_collision_topk_v1 import (
    P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1,
    body_frame_aabb_collision_metrics_p9_v1,
    p9_collision_policy_v1,
    select_p9_collision_aware_topk_v1,
    verify_p9_collision_selection_v1,
)
from .constraint_manifold_solver_v3 import (
    CylinderTrimV3, ManifoldFaceGeometryV3, PlaneTrimV3,
    canonical_sha256, file_sha256,
)
from .constraint_manifold_solver_v4 import (
    P11_BODY_AABB_RANK_SCHEMA_V1, candidate_rank_payload_p11_body_aabb_v1,
    P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2,
    P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1,
    candidate_rank_payload_v4, canonical_candidate_payload_v4,
)
from .domain_types import Transform
from .supervised_worker_attestation_v1 import NATIVE_THREAD_ENV


PREPROCESS_REQUEST_SCHEMA_VERSION = "constraint_manifold_preprocess_request.v4"
PREPROCESS_REQUEST_SCHEMA_VERSION_V5 = "constraint_manifold_preprocess_request.v5"
PREPROCESS_WORKER_TERMINAL_SCHEMA_VERSION = (
    "constraint_manifold_preprocess_worker_terminal.v4"
)
PREPROCESS_RECEIPT_SCHEMA_VERSION = "constraint_manifold_preprocess_receipt.v4"
SANITIZED_GEOMETRY_SCHEMA_VERSION = "preprocess_compact_topk.v1"
SANITIZED_GEOMETRY_P9_COLLISION_TOPK_SCHEMA_VERSION_V1 = (
    "preprocess_compact_topk_p9_collision_structural.v1"
)
SANITIZED_GEOMETRY_P11_BODY_AABB_SCHEMA_VERSION_V1 = (
    "preprocess_compact_topk_p11_body_aabb.v1"
)
SANITIZED_GEOMETRY_P11_STRATIFIED_DIVERSITY_SCHEMA_VERSION_V1 = (
    "preprocess_compact_topk_p11_stratified_diversity.v1"
)
SANITIZED_GEOMETRY_P11_FACTORIZED_SCHEMA_VERSION_V2 = (
    "preprocess_compact_topk_p11_factorized.v2"
)
SANITIZED_AUTHORITY_SCHEMA_VERSION = "constraint_manifold_preprocess_authority.v4"
PREPROCESS_POLICY_SCHEMA_VERSION = "constraint_manifold_preprocess_policy.v4"
PREPROCESS_POLICY_P9_COLLISION_TOPK_SCHEMA_VERSION_V1 = (
    "constraint_manifold_preprocess_policy.p9_collision_topk.v1"
)
PREPROCESS_POLICY_P11_BODY_AABB_SCHEMA_VERSION_V1 = (
    "constraint_manifold_preprocess_policy.p11_body_aabb.v1"
)
PREPROCESS_POLICY_P11_STRATIFIED_DIVERSITY_SCHEMA_VERSION_V1 = (
    "constraint_manifold_preprocess_policy.p11_stratified_diversity.v1"
)
PREPROCESS_POLICY_P11_FACTORIZED_SCHEMA_VERSION_V2 = (
    "constraint_manifold_preprocess_policy.p11_factorized.v2"
)
PREPROCESS_OPERATION_V4 = "constraint_manifold_preprocess_bundle"
PREPROCESS_STAGE_TELEMETRY_SCHEMA_VERSION_V7 = (
    "constraint_manifold_preprocess_stage_telemetry.v7"
)
PREPROCESS_STAGE_TELEMETRY_PREFIX_V7 = b"NEUROCAD_STAGE_V7 "

LANDMARK_CAP_PER_ENDPOINT_PER_TYPE_V4 = 6
LANDMARK_PAIR_POLICY_V4 = "all_cross_pairs_after_lexicographic_even_span_cap"
MAX_WORKER_STDOUT_BYTES_V4 = 64 * 1024 * 1024
_ALIGNMENT_TYPES_V4 = (
    "trim_vertices_local_2d", "mesh_vertices_local_2d",
    "trim_edge_midpoints_local_2d", "mesh_edge_midpoints_local_2d",
    "triangle_centroids_local_2d", "loop_centroids_local_2d",
    "adjacent_shared_edge_vertices_local_2d",
    "adjacent_shared_edge_midpoints_local_2d",
)
_CYLINDER_YAW_TYPES_V4 = (
    "trim_vertices_local_2d", "trim_edge_midpoints_local_2d",
    "loop_centroids_local_2d", "adjacent_shared_edge_vertices_local_2d",
    "adjacent_shared_edge_midpoints_local_2d",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _SanitizedAuthorityExpectedAttestationV19:
  canonical_state_sha256: str
  canonical_state_byte_count: int
  authority_payload_sha256: str
  sanitized_geometry_sha256: str


_AUTHORITY_STATES: WeakKeyDictionary[Any, bytes] = WeakKeyDictionary()
_AUTHORITY_EXPECTED_ATTESTATIONS: WeakKeyDictionary[
    Any, _SanitizedAuthorityExpectedAttestationV19
] = WeakKeyDictionary()
_FORBIDDEN_RESULT_KEYS = (
    "ground_truth", "gold", "source_contact", "source_label", "source_pose",
    "residual", "mate_program", "mate_pose", "mate_type", "mate_id",
    "target_transform", "step_path", "row_ordinal",
)
_STAGE_ALLOWLIST_V7 = frozenset((
    "worker_import", "request_validation", "endpoint_step_load",
    "endpoint_face_replay", "endpoint_geometry_mesh",
    "endpoint_landmarks_topology", "endpoint_bbox", "candidate_domain_prepare",
    "candidate_domain_ready", "candidate_overlap_progress", "candidate_domain_complete",
    "candidate_hash_dedupe", "sanitized_validation", "terminal_serialization",
))
_COUNTER_ALLOWLIST_V7 = frozenset((
    "program_index", "endpoint_index", "step_bytes", "face_count",
    "selected_edge_count", "selected_wire_count", "adjacent_face_count",
    "shared_edge_count", "mesh_node_count", "mesh_triangle_count",
    "retained_triangle_count", "alignment_event_count", "yaw_event_count_raw",
    "yaw_state_count", "offset_event_count_raw", "offset_state_count",
    "candidate_count_progress", "candidate_count_raw", "candidate_count_unique",
    "overlap_call_count", "overlap_dense_aabb_pair_count",
    "overlap_index_build_count",
    "overlap_periodic_copy_count", "overlap_index_candidate_pair_count",
    "overlap_exact_clip_pair_count", "overlap_numba_batch_pair_count",
    "overlap_numba_batch_call_count", "overlap_index_elapsed_seconds_raw",
    "overlap_exact_elapsed_seconds_raw", "candidate_payload_hash_elapsed_seconds_raw",
    "candidate_dedupe_sort_elapsed_seconds_raw", "sanitized_hash_elapsed_seconds_raw",
    "serialized_bytes", "sign_index",
    "factor_sign_count", "factor_yaw_count", "factor_axial_offset_count",
    "factorized_candidate_count", "materialized_candidate_count",
))
_MAX_STAGE_EVENTS_V7 = 8192
COMPACT_TOP_K_V1 = 32


def issue_preprocess_stage_event_v7(
    *, request_payload_sha256: str, call_nonce: str, sequence: int, stage: str,
    elapsed_seconds_raw: float, counters: Mapping[str, int | float],
) -> dict[str, Any]:
  if (_SHA256_RE.fullmatch(request_payload_sha256) is None
      or _SHA256_RE.fullmatch(call_nonce) is None):
    raise ValueError("preprocess telemetry binding differs")
  if type(sequence) is not int or sequence < 0 or stage not in _STAGE_ALLOWLIST_V7:
    raise ValueError("preprocess telemetry identity differs")
  elapsed = float(elapsed_seconds_raw)
  if not math.isfinite(elapsed) or elapsed < 0.0:
    raise ValueError("preprocess telemetry elapsed differs")
  if not isinstance(counters, Mapping) or set(counters) - _COUNTER_ALLOWLIST_V7:
    raise ValueError("preprocess telemetry counter allowlist differs")
  normalized = {}
  for key, value in counters.items():
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
      raise ValueError("preprocess telemetry counter differs")
    normalized[str(key)] = value
  unsigned = {
      "schema_version": PREPROCESS_STAGE_TELEMETRY_SCHEMA_VERSION_V7,
      "request_payload_sha256": request_payload_sha256, "call_nonce": call_nonce,
      "sequence": sequence, "stage": stage,
      "elapsed_seconds_raw": elapsed, "counters": normalized,
  }
  return {**unsigned, "event_payload_sha256": canonical_sha256(unsigned)}


def parse_preprocess_stage_telemetry_v7(
    stderr: bytes, *, request_payload_sha256: str, call_nonce: str,
) -> list[dict[str, Any]]:
  events = []
  for raw_line in bytes(stderr).splitlines():
    if not raw_line.startswith(PREPROCESS_STAGE_TELEMETRY_PREFIX_V7):
      continue
    if len(events) >= _MAX_STAGE_EVENTS_V7:
      raise ValueError("preprocess telemetry event budget differs")
    raw_event = raw_line[len(PREPROCESS_STAGE_TELEMETRY_PREFIX_V7):]
    event = _strict_json_bytes(raw_event, label="preprocess stage telemetry")
    if set(event) != {
        "schema_version", "request_payload_sha256", "call_nonce", "sequence",
        "stage", "elapsed_seconds_raw", "counters", "event_payload_sha256",
    }:
      raise ValueError("preprocess telemetry schema differs")
    observed = dict(event)
    commitment = observed.pop("event_payload_sha256")
    if commitment != canonical_sha256(observed):
      raise ValueError("preprocess telemetry commitment differs")
    replayed = issue_preprocess_stage_event_v7(
        request_payload_sha256=str(event["request_payload_sha256"]),
        call_nonce=str(event["call_nonce"]), sequence=int(event["sequence"]),
        stage=str(event["stage"]),
        elapsed_seconds_raw=float(event["elapsed_seconds_raw"]),
        counters=event["counters"],
    )
    if (replayed != dict(event)
        or event["request_payload_sha256"] != request_payload_sha256
        or event["call_nonce"] != call_nonce
        or int(event["sequence"]) != len(events)):
      raise ValueError("preprocess telemetry receipt binding differs")
    events.append(replayed)
  return events


def _clean_float(value: float) -> float:
  result = round(float(value), 12)
  if not math.isfinite(result):
    raise ValueError("preprocess geometry contains non-finite value")
  return 0.0 if result == 0.0 else result


def _group_canonical_yaw_events_v2(
    raw_events: Sequence[Mapping[str, Any]],
) -> dict[float, list[dict[str, Any]]]:
  """Merge periodic P11 events under their cleaned canonical yaw before grouping."""

  groups: dict[float, dict[str, dict[str, Any]]] = {}
  for raw_event in raw_events:
    raw_yaw = float(raw_event["yaw"])
    event = {str(key): value for key, value in raw_event.items() if key != "yaw"}
    event_sha256 = canonical_sha256(event)
    for value in (raw_yaw, raw_yaw + math.pi):
      cleaned = _clean_float(value % (2.0 * math.pi))
      yaw = canonical_yaw_v2(cleaned)
      prior = groups.setdefault(yaw, {}).setdefault(event_sha256, event)
      if prior != event:
        raise ValueError("P11 canonical yaw event commitment collision")
  return {
      yaw: sorted(events.values(), key=canonical_sha256)
      for yaw, events in groups.items()
  }


def _point2(value: Sequence[float]) -> tuple[float, float]:
  row = np.asarray(value, dtype=float)
  if row.shape != (2,) or not np.isfinite(row).all():
    raise ValueError("preprocess 2D landmark differs")
  return _clean_float(row[0]), _clean_float(row[1])


def canonicalize_landmark_points_v4(
    rows: Sequence[Sequence[float]], *, maximum: int,
) -> tuple[tuple[float, float], ...]:
  """Input-order-independent, geometry-spanning frozen landmark selection."""

  if type(maximum) is not int or maximum < 1:
    raise ValueError("landmark maximum differs")
  unique = tuple(sorted({_point2(row) for row in rows}))
  if len(unique) <= maximum:
    return unique
  if maximum == 1:
    return (unique[len(unique) // 2],)
  indices = tuple(round(index * (len(unique) - 1) / (maximum - 1))
                  for index in range(maximum))
  return tuple(unique[index] for index in indices)


def _deep_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
  if isinstance(value, (list, tuple)):
    return tuple(_deep_freeze(item) for item in value)
  return value


def _deep_thaw(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(key): _deep_thaw(item) for key, item in value.items()}
  if isinstance(value, tuple):
    return [_deep_thaw(item) for item in value]
  return value


def _strict_json_bytes(raw: bytes, *, label: str) -> Mapping[str, Any]:
  def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
      if key in result:
        raise ValueError(f"{label} contains duplicate key")
      result[key] = value
    return result

  try:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
  except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict JSON") from error
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be an object")
  return value


def _require_matrix(value: Sequence[float], *, label: str) -> tuple[float, ...]:
  matrix = np.asarray(value, dtype=float)
  if matrix.shape != (16,) or not np.isfinite(matrix).all():
    raise ValueError(f"{label} differs")
  checked = _v1._matrix_tuple(matrix.reshape(4, 4))
  return tuple(float(item) for item in checked)


def _transform(value: Sequence[float]) -> Transform:
  matrix = np.asarray(value, dtype=float).reshape(4, 4)
  return Transform(rotation=matrix[:3, :3], translation=matrix[:3, 3])


@dataclass(frozen=True, slots=True)
class ConstraintManifoldPreprocessEndpointV4:
  part_slot: str
  step_path: str
  step_sha256: str
  graph_face_index: int
  raw_occ_face_index: int
  face_signature_sha256: str
  current_world_row_major: tuple[float, ...]

  def __post_init__(self) -> None:
    if self.part_slot not in {"a", "b"}:
      raise ValueError("preprocess endpoint slot differs")
    if not self.step_path:
      raise ValueError("preprocess endpoint STEP path is absent")
    if _SHA256_RE.fullmatch(self.step_sha256) is None:
      raise ValueError("preprocess endpoint STEP hash differs")
    if _SHA256_RE.fullmatch(self.face_signature_sha256) is None:
      raise ValueError("preprocess endpoint face signature differs")
    if min(self.graph_face_index, self.raw_occ_face_index) < 0:
      raise ValueError("preprocess endpoint face index differs")
    _require_matrix(self.current_world_row_major, label="endpoint world")

  def worker_payload(self) -> dict[str, Any]:
    return {
        "part_slot": self.part_slot,
        "step_path": str(Path(self.step_path).resolve()),
        "step_sha256": self.step_sha256,
        "graph_face_index": self.graph_face_index,
        "raw_occ_face_index": self.raw_occ_face_index,
        "face_signature_sha256": self.face_signature_sha256,
        "current_world_row_major": list(self.current_world_row_major),
    }


@dataclass(frozen=True, slots=True)
class ConstraintManifoldPreprocessRequestV4:
  query_id: str
  program_index: int
  endpoints: tuple[ConstraintManifoldPreprocessEndpointV4, ...]
  child_world_row_major: tuple[float, ...]
  schema_version: str = PREPROCESS_REQUEST_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != PREPROCESS_REQUEST_SCHEMA_VERSION or not self.query_id:
      raise ValueError("preprocess request identity differs")
    if self.program_index not in {9, 11}:
      raise ValueError("preprocess V4 authorizes only programs 9/11")
    if len(self.endpoints) != 2 or tuple(row.part_slot for row in self.endpoints) != ("a", "b"):
      raise ValueError("preprocess endpoint order differs")
    child = _require_matrix(self.child_world_row_major, label="child world")
    if not np.allclose(
        np.asarray(child), np.asarray(self.endpoints[1].current_world_row_major),
        atol=1e-12, rtol=0.0,
    ):
      raise ValueError("child world differs from endpoint b current world")

  def worker_payload(self, *, call_nonce: str) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(call_nonce) is None:
      raise ValueError("preprocess nonce differs")
    return {
        "schema_version": self.schema_version,
        "operation": PREPROCESS_OPERATION_V4,
        "call_nonce": call_nonce,
        "query_id_sha256": hashlib.sha256(self.query_id.encode("utf-8")).hexdigest(),
        "program_index": self.program_index,
        "endpoints": [row.worker_payload() for row in self.endpoints],
        "child_world_row_major": list(self.child_world_row_major),
        "policy": preprocess_policy_for_program_v1(self.program_index),
        "producer_source_sha256s": preprocess_source_sha256s_v4(),
    }


@dataclass(frozen=True, slots=True)
class ConstraintManifoldPreprocessRequestV5:
  """P9-collision-only request; never accepted as a historical V4 request."""

  query_id: str
  program_index: int
  endpoints: tuple[ConstraintManifoldPreprocessEndpointV4, ...]
  child_world_row_major: tuple[float, ...]
  ranking_schema: str
  schema_version: str = PREPROCESS_REQUEST_SCHEMA_VERSION_V5

  def __post_init__(self) -> None:
    if self.schema_version != PREPROCESS_REQUEST_SCHEMA_VERSION_V5 or not self.query_id:
      raise ValueError("preprocess V5 request identity differs")
    if self.program_index != 9:
      raise ValueError("preprocess V5 authorizes only program 9")
    if self.ranking_schema != P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1:
      raise ValueError("preprocess V5 ranking schema differs")
    if len(self.endpoints) != 2 or tuple(
        row.part_slot for row in self.endpoints
    ) != ("a", "b"):
      raise ValueError("preprocess V5 endpoint order differs")
    child = _require_matrix(self.child_world_row_major, label="V5 child world")
    if not np.allclose(
        np.asarray(child), np.asarray(self.endpoints[1].current_world_row_major),
        atol=1e-12, rtol=0.0,
    ):
      raise ValueError("V5 child world differs from endpoint b current world")

  def worker_payload(self, *, call_nonce: str) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(call_nonce) is None:
      raise ValueError("preprocess V5 nonce differs")
    return {
        "schema_version": self.schema_version,
        "operation": PREPROCESS_OPERATION_V4,
        "call_nonce": call_nonce,
        "query_id_sha256": hashlib.sha256(self.query_id.encode("utf-8")).hexdigest(),
        "program_index": self.program_index,
        "endpoints": [row.worker_payload() for row in self.endpoints],
        "child_world_row_major": list(self.child_world_row_major),
        "ranking_schema": self.ranking_schema,
        "policy": preprocess_policy_for_program_v1(
            self.program_index, ranking_schema=self.ranking_schema,
        ),
        "producer_source_sha256s": preprocess_source_sha256s_v4(),
    }


def preprocess_source_sha256s_v4() -> dict[str, str]:
  root = Path(__file__).resolve().parent
  paths = (
      root / "constraint_manifold_preprocess_v4.py",
      root / "tools" / "constraint_manifold_preprocess_worker_v4.py",
      root / "constraint_manifold_occ_adapter_v3.py",
      root / "constraint_manifold_solver_v4.py",
      root / "constraint_manifold_factorized_p11_v2.py",
      root / "constraint_manifold_p9_collision_topk_v1.py",
      root / "constraint_manifold_solver_v3.py",
      root / "constraint_manifold_solver_v2.py",
      root / "constraint_manifold_solver_v1.py",
      root / "cadquery_backend.py",
  )
  return {path.name: file_sha256(path) for path in paths}


def preprocess_policy_v4() -> dict[str, Any]:
  unsigned = {
      "schema_version": PREPROCESS_POLICY_SCHEMA_VERSION,
      "absolute_mesh_policy": "constraint_manifold_absolute_mesh_proof.v3",
      "landmark_cap_per_endpoint_per_type": LANDMARK_CAP_PER_ENDPOINT_PER_TYPE_V4,
      "landmark_selection": "sorted_unique_lexicographic_even_span.v4",
      "landmark_pair_policy": LANDMARK_PAIR_POLICY_V4,
      "translation_alignment_landmark_types": list(_ALIGNMENT_TYPES_V4),
      "plane_yaw_policy": "all_boundary_and_shared_edge_direction_pairs_mod_pi.v4",
      "worker_top_k": COMPACT_TOP_K_V1,
      "candidate_canonical_order": "candidate_key_ascending",
      "compact_transport_order": "frozen_solver_cheap_rank_then_candidate_key",
      "parent_ranking": "worker_committed_rank_no_parent_resort",
      "native_thread_environment": dict(sorted(NATIVE_THREAD_ENV.items())),
  }
  return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


def preprocess_policy_for_program_v1(
    program_index: int, *, ranking_schema: str | None = None,
) -> dict[str, Any]:
  if program_index == 9:
    if ranking_schema is None:
      return preprocess_policy_v4()
    if ranking_schema != P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1:
      raise ValueError("preprocess P9 ranking schema differs")
    return p9_collision_policy_v1()
  if ranking_schema is not None:
    raise ValueError("preprocess ranking schema/program differs")
  if program_index != 11:
    raise ValueError("preprocess ranking program differs")
  unsigned = dict(preprocess_policy_v4())
  unsigned.pop("policy_payload_sha256")
  unsigned["schema_version"] = PREPROCESS_POLICY_P11_FACTORIZED_SCHEMA_VERSION_V2
  unsigned["ranking_schema"] = P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2
  unsigned["candidate_domain"] = (
      "normal_sign_join_canonical_yaw_x_axial_offset_factor_commitment.v2"
  )
  unsigned["compact_transport_order"] = (
      "p11_factor_anchor_then_sign_offset_family_yaw_round_robin.v2"
  )
  unsigned["stratum_inputs"] = [
      "factor_normal_sign", "factor_canonical_yaw",
      "factor_offset_event_family",
  ]
  unsigned["stage_a"] = "per_sign_nearest_zero_pi_yaw_baseline_factor_anchors"
  unsigned["stage_b"] = (
      "minimum_sign_offset_family_coverage_then_factor_yaw_round_robin"
  )
  unsigned["placement_materialization_budget"] = COMPACT_TOP_K_V1
  return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


def preprocess_policy_p11_stratified_diversity_v1() -> dict[str, Any]:
  """Historical exhaustive P11 diversity policy; never the worker default."""

  unsigned = dict(preprocess_policy_v4())
  unsigned.pop("policy_payload_sha256")
  unsigned["schema_version"] = (
      PREPROCESS_POLICY_P11_STRATIFIED_DIVERSITY_SCHEMA_VERSION_V1
  )
  unsigned["ranking_schema"] = P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1
  unsigned["compact_transport_order"] = (
      "p11_sign_offset_family_coverage_then_yaw_round_robin.v1"
  )
  unsigned["stratum_inputs"] = [
      "candidate_normal_sign", "candidate_canonical_yaw_bucket",
      "candidate_offset_event_family",
  ]
  unsigned["stage_a"] = "per_sign_nearest_zero_pi_yaw_baseline_offset_anchors"
  unsigned["stage_b"] = (
      "minimum_sign_offset_family_coverage_then_yaw_round_robin_legacy_rank"
  )
  return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


def preprocess_policy_p11_body_aabb_v1() -> dict[str, Any]:
  """Historical development baseline retained under its original schema."""

  unsigned = dict(preprocess_policy_v4())
  unsigned.pop("policy_payload_sha256")
  unsigned["schema_version"] = PREPROCESS_POLICY_P11_BODY_AABB_SCHEMA_VERSION_V1
  unsigned["ranking_schema"] = P11_BODY_AABB_RANK_SCHEMA_V1
  unsigned["compact_transport_order"] = (
      "p11_body_frame_aabb_overlap_upper_mm3_ascending_before_trim_ratio.v1"
  )
  unsigned["body_aabb_overlap_claim"] = (
      "ranking_heuristic_only_never_negative_or_certificate_evidence"
  )
  return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


def compact_candidate_domain_v1(
    candidate_rows: Sequence[Mapping[str, Any]], *, program_index: int,
) -> dict[str, Any]:
  """Commit the complete canonical domain and retain only frozen top-32 IPC rows."""

  unique: dict[str, dict[str, Any]] = {}
  for value in candidate_rows:
    row = json.loads(json.dumps(value, allow_nan=False))
    key = str(row.get("candidate_key", ""))
    if _SHA256_RE.fullmatch(key) is None:
      raise ValueError("compact candidate key differs")
    prior = unique.setdefault(key, row)
    if prior != row:
      raise ValueError("compact duplicate candidate evidence differs")
  canonical_domain = [unique[key] for key in sorted(unique)]
  if not canonical_domain:
    raise ValueError("compact candidate domain is empty")
  ranked = sorted(
      canonical_domain,
      key=lambda row: candidate_rank_payload_v4(program_index, row),
  )
  compact = ranked[:COMPACT_TOP_K_V1]
  return {
      "candidate_domain_count": len(canonical_domain),
      "candidate_domain_sha256": canonical_sha256([
          canonical_candidate_payload_v4(program_index, row)
          for row in canonical_domain
      ]),
      "compact_top_k": COMPACT_TOP_K_V1,
      "compact_topk_payload_sha256": canonical_sha256(compact),
      "cheap_geometry_candidates": compact,
  }


def compact_candidate_domain_p11_body_aabb_v1(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Commit p11 heuristic evidence and rank it under an explicit new schema."""

  unique: dict[str, dict[str, Any]] = {}
  for value in candidate_rows:
    row = json.loads(json.dumps(value, allow_nan=False))
    key = str(row.get("candidate_key", ""))
    if _SHA256_RE.fullmatch(key) is None:
      raise ValueError("p11 compact candidate key differs")
    prior = unique.setdefault(key, row)
    if prior != row:
      raise ValueError("p11 compact duplicate candidate evidence differs")
  canonical_domain = [unique[key] for key in sorted(unique)]
  if not canonical_domain:
    raise ValueError("p11 compact candidate domain is empty")
  ranked = sorted(
      canonical_domain, key=candidate_rank_payload_p11_body_aabb_v1,
  )
  compact = ranked[:COMPACT_TOP_K_V1]
  return {
      "ranking_schema": P11_BODY_AABB_RANK_SCHEMA_V1,
      "candidate_domain_count": len(canonical_domain),
      "candidate_domain_sha256": canonical_sha256([
          canonical_candidate_payload_v4(11, row) for row in canonical_domain
      ]),
      "compact_top_k": COMPACT_TOP_K_V1,
      "compact_topk_payload_sha256": canonical_sha256(compact),
      "cheap_geometry_candidates": compact,
  }


_P11_BASELINE_OFFSET_FAMILIES_V1 = (
    "interval_center", "lower_endpoint", "upper_endpoint", "opposed_endpoint",
)
_P11_OFFSET_FAMILIES_V1 = frozenset(
    (*_P11_BASELINE_OFFSET_FAMILIES_V1, *_ALIGNMENT_TYPES_V4)
)


def _p11_yaw_state_v1(candidate: Mapping[str, Any]) -> tuple[float, str]:
  raw = float(candidate.get("yaw_radians", math.nan))
  if not math.isfinite(raw):
    raise ValueError("p11 diversity yaw differs")
  value = _clean_float(raw % (2.0 * math.pi))
  if math.isclose(value, 2.0 * math.pi, abs_tol=5e-13, rel_tol=0.0):
    value = 0.0
  bucket = canonical_sha256({
      "schema_version": "constraint_manifold_p11_canonical_yaw_bucket.v1",
      "yaw_radians_mod_2pi": value,
  })
  return value, bucket


def _p11_offset_families_v1(candidate: Mapping[str, Any]) -> tuple[str, ...]:
  events = candidate.get("alignment_events")
  if not isinstance(events, (list, tuple)):
    raise ValueError("p11 diversity alignment events differ")
  families = tuple(sorted({
      str(event.get("kind")) for event in events if isinstance(event, Mapping)
      and str(event.get("kind")) in _P11_OFFSET_FAMILIES_V1
  }))
  if not families:
    raise ValueError("p11 diversity offset family is absent")
  return families


def candidate_strata_p11_diversity_v1(candidate: Mapping[str, Any]) -> tuple[str, ...]:
  """Return source-free quota strata from candidate-owned generation events."""

  sign = int(candidate.get("normal_sign", 0))
  if sign not in {-1, 1}:
    raise ValueError("p11 diversity normal sign differs")
  _p11_yaw_state_v1(candidate)
  return tuple(
      f"sign={sign}|offset_family={family}"
      for family in _p11_offset_families_v1(candidate)
  )


def _p11_diversity_descriptor_v1(candidate: Mapping[str, Any]) -> dict[str, Any]:
  sign = int(candidate.get("normal_sign", 0))
  if sign not in {-1, 1}:
    raise ValueError("p11 diversity normal sign differs")
  yaw, bucket = _p11_yaw_state_v1(candidate)
  families = _p11_offset_families_v1(candidate)
  return {
      "normal_sign": sign, "yaw_radians": yaw, "yaw_bucket_sha256": bucket,
      "offset_families": families,
  }


def _p11_circular_yaw_distance_v1(value: float, target: float) -> float:
  difference = abs(value - target) % (2.0 * math.pi)
  return min(difference, 2.0 * math.pi - difference)


def compact_candidate_domain_p11_stratified_diversity_v1(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Select top32 by non-oracle sign/family coverage and yaw round-robin."""

  unique: dict[str, dict[str, Any]] = {}
  for value in candidate_rows:
    row = json.loads(json.dumps(value, allow_nan=False))
    key = str(row.get("candidate_key", ""))
    if _SHA256_RE.fullmatch(key) is None:
      raise ValueError("p11 diversity candidate key differs")
    prior = unique.setdefault(key, row)
    if prior != row:
      raise ValueError("p11 diversity duplicate candidate evidence differs")
  canonical_domain = [unique[key] for key in sorted(unique)]
  if not canonical_domain:
    raise ValueError("p11 diversity candidate domain is empty")
  descriptors = {
      row["candidate_key"]: _p11_diversity_descriptor_v1(row)
      for row in canonical_domain
  }
  strata_by_key = {
      row["candidate_key"]: tuple(
          f"sign={descriptors[row['candidate_key']]['normal_sign']}|offset_family={family}"
          for family in descriptors[row["candidate_key"]]["offset_families"]
      )
      for row in canonical_domain
  }
  legacy_ranked = sorted(
      canonical_domain, key=lambda row: candidate_rank_payload_v4(11, row)
  )
  legacy_position = {
      row["candidate_key"]: index for index, row in enumerate(legacy_ranked)
  }
  def row_order(row: Mapping[str, Any]) -> tuple[int, str]:
    return legacy_position[row["candidate_key"]], str(row["candidate_key"])

  sign_rows: dict[int, list[dict[str, Any]]] = {-1: [], 1: []}
  sign_yaw_family_rows: dict[tuple[int, float, str], list[dict[str, Any]]] = {}
  stratum_rows: dict[str, list[dict[str, Any]]] = {}
  stratum_yaw_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
  for row in canonical_domain:
    key = str(row["candidate_key"])
    descriptor = descriptors[key]
    sign = int(descriptor["normal_sign"])
    yaw = float(descriptor["yaw_radians"])
    bucket = str(descriptor["yaw_bucket_sha256"])
    sign_rows[sign].append(row)
    for family, stratum in zip(
        descriptor["offset_families"], strata_by_key[key], strict=True,
    ):
      sign_yaw_family_rows.setdefault((sign, yaw, str(family)), []).append(row)
      stratum_rows.setdefault(stratum, []).append(row)
      stratum_yaw_rows.setdefault(stratum, {}).setdefault(bucket, []).append(row)
  sign_rows = {
      sign: sorted(rows, key=row_order) for sign, rows in sign_rows.items()
  }
  sign_yaw_family_rows = {
      key: sorted(rows, key=row_order) for key, rows in sign_yaw_family_rows.items()
  }
  stratum_rows = {
      key: sorted(rows, key=row_order) for key, rows in stratum_rows.items()
  }
  stratum_yaw_rows = {
      stratum: {
          bucket: sorted(rows, key=row_order) for bucket, rows in by_yaw.items()
      }
      for stratum, by_yaw in stratum_yaw_rows.items()
  }
  yaw_buckets_by_stratum = {
      stratum: tuple(sorted(by_yaw, key=lambda bucket: (
          legacy_position[by_yaw[bucket][0]["candidate_key"]], bucket,
      )))
      for stratum, by_yaw in stratum_yaw_rows.items()
  }
  selected: list[dict[str, Any]] = []
  selected_keys: set[str] = set()
  covered: set[str] = set()

  def add_first(rows: Sequence[dict[str, Any]]) -> bool:
    for row in rows:
      key = str(row["candidate_key"])
      if key not in selected_keys:
        selected.append(row); selected_keys.add(key)
        covered.update(strata_by_key[key])
        return True
    return False

  # Stage A: bounded analytic anchors.  The family/yaw specification is fixed
  # independently of any source pose, label, residual, or exact observation.
  anchor_specs = (
      (0.0, ("lower_endpoint", "upper_endpoint")),
      (math.pi, ("interval_center", "opposed_endpoint")),
  )
  for sign in (-1, 1):
    rows_for_sign = sign_rows[sign]
    yaw_values = sorted({
        float(descriptors[row["candidate_key"]]["yaw_radians"])
        for row in rows_for_sign
    })
    for target_yaw, families in anchor_specs:
      if not yaw_values:
        continue
      anchor_yaw = min(yaw_values, key=lambda value: (
          _p11_circular_yaw_distance_v1(value, target_yaw), value,
      ))
      for family in families:
        add_first(sign_yaw_family_rows.get((sign, anchor_yaw, family), ()))
  stage_a_count = len(selected)

  quota_strata = tuple(sorted(stratum_rows))

  # Stage B1: every available sign x offset-family receives minimum coverage.
  for stratum in quota_strata:
    if stratum in covered or len(selected) >= COMPACT_TOP_K_V1:
      continue
    by_yaw = stratum_yaw_rows[stratum]
    for bucket in yaw_buckets_by_stratum[stratum]:
      if add_first(by_yaw[bucket]):
        break
  if len(quota_strata) <= COMPACT_TOP_K_V1 and set(quota_strata) - covered:
    raise ValueError("p11 diversity minimum stratum coverage failed")
  stage_b_quota_count = len(selected) - stage_a_count

  # Stage B2: rotate through canonical yaw buckets inside each quota stratum;
  # candidates inside a bucket retain the legacy cheap order.
  queues: dict[str, list[dict[str, Any]]] = {}
  for stratum in quota_strata:
    by_yaw = stratum_yaw_rows[stratum]
    ordered_buckets = [
        by_yaw[bucket] for bucket in yaw_buckets_by_stratum[stratum]
    ]
    queue = []
    depth = 0
    while any(depth < len(rows) for rows in ordered_buckets):
      for rows in ordered_buckets:
        if depth < len(rows):
          queue.append(rows[depth])
      depth += 1
    queues[stratum] = queue
  cursors = {stratum: 0 for stratum in quota_strata}
  while len(selected) < min(COMPACT_TOP_K_V1, len(canonical_domain)):
    progressed = False
    for stratum in quota_strata:
      queue = queues[stratum]
      while (cursors[stratum] < len(queue)
             and queue[cursors[stratum]]["candidate_key"] in selected_keys):
        cursors[stratum] += 1
      if cursors[stratum] < len(queue):
        row = queue[cursors[stratum]]; cursors[stratum] += 1
        key = str(row["candidate_key"])
        selected.append(row); selected_keys.add(key); covered.update(strata_by_key[key])
        progressed = True
        if len(selected) == min(COMPACT_TOP_K_V1, len(canonical_domain)):
          break
    if not progressed:
      break
  for row in legacy_ranked:
    if len(selected) == min(COMPACT_TOP_K_V1, len(canonical_domain)):
      break
    if row["candidate_key"] not in selected_keys:
      key = str(row["candidate_key"])
      selected.append(row); selected_keys.add(key); covered.update(strata_by_key[key])

  domain_sha = canonical_sha256([
      canonical_candidate_payload_v4(11, row) for row in canonical_domain
  ])
  coverage = {
      "schema_version": "constraint_manifold_p11_diversity_coverage.v1",
      "stratum_definition": (
          "candidate_normal_sign_x_candidate_offset_event_family"
      ),
      "yaw_bucket_definition": "canonical_yaw_radians_mod_2pi_sha256",
      "quota_strata": list(quota_strata),
      "covered_quota_strata": sorted(covered),
      "stage_a_candidate_count": stage_a_count,
      "stage_b_quota_candidate_count": stage_b_quota_count,
      "selected_candidate_count": len(selected),
      "selected_yaw_bucket_count": len({
          descriptors[row["candidate_key"]]["yaw_bucket_sha256"] for row in selected
      }),
  }
  coverage["coverage_payload_sha256"] = canonical_sha256(coverage)
  return {
      "ranking_schema": P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1,
      "candidate_domain_count": len(canonical_domain),
      "candidate_domain_sha256": domain_sha,
      "compact_top_k": COMPACT_TOP_K_V1,
      "compact_topk_payload_sha256": canonical_sha256(selected),
      "diversity_coverage": coverage,
      "cheap_geometry_candidates": selected,
  }


def validate_compact_candidate_domain_v1(payload: Mapping[str, Any]) -> None:
  candidates = payload.get("cheap_geometry_candidates")
  domain_count = payload.get("candidate_domain_count")
  if (
      not isinstance(candidates, list) or not candidates
      or type(domain_count) is not int or domain_count < len(candidates)
      or domain_count > 250_000
      or payload.get("compact_top_k") != COMPACT_TOP_K_V1
      or len(candidates) != min(COMPACT_TOP_K_V1, domain_count)
      or _SHA256_RE.fullmatch(str(payload.get("candidate_domain_sha256"))) is None
      or payload.get("compact_topk_payload_sha256") != canonical_sha256(candidates)
  ):
    raise ValueError("sanitized compact candidate commitment differs")


@dataclass(frozen=True, slots=True)
class ConstraintManifoldPreprocessReceiptV4:
  call_nonce: str
  request_payload_sha256: str
  worker_code_bundle_sha256: str
  terminal_status: str
  issuer: str
  child_started: bool
  elapsed_seconds: float
  result_items: tuple[tuple[str, Any], ...]
  receipt_payload_sha256: str
  schema_version: str = PREPROCESS_RECEIPT_SCHEMA_VERSION

  def __post_init__(self) -> None:
    object.__setattr__(self, "result_items", tuple(
        (str(key), _deep_freeze(value)) for key, value in self.result_items
    ))
    for label, value in (
        ("call nonce", self.call_nonce),
        ("request payload", self.request_payload_sha256),
        ("worker code bundle", self.worker_code_bundle_sha256),
        ("receipt payload", self.receipt_payload_sha256),
    ):
      if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"preprocess {label} differs")
    if self.terminal_status not in {"ok", "timeout", "kernel_error"}:
      raise ValueError("preprocess receipt terminal status differs")
    if self.issuer not in {"child_process_echo", "parent_observer"}:
      raise ValueError("preprocess receipt issuer differs")
    if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
      raise ValueError("preprocess receipt elapsed time differs")
    if self.receipt_payload_sha256 != canonical_sha256(self.unsigned_payload()):
      raise ValueError("preprocess receipt commitment differs")

  @property
  def result(self) -> Mapping[str, Any]:
    return MappingProxyType({key: _deep_freeze(value) for key, value in self.result_items})

  def unsigned_payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version,
        "operation": PREPROCESS_OPERATION_V4,
        "call_nonce": self.call_nonce,
        "request_payload_sha256": self.request_payload_sha256,
        "worker_code_bundle_sha256": self.worker_code_bundle_sha256,
        "terminal_status": self.terminal_status,
        "issuer": self.issuer,
        "child_started": self.child_started,
        "elapsed_seconds": self.elapsed_seconds,
        "result": {key: _deep_thaw(value) for key, value in self.result_items},
    }

  def payload(self) -> dict[str, Any]:
    return {**self.unsigned_payload(), "receipt_payload_sha256": self.receipt_payload_sha256}


def _issue_receipt_v4(
    *, call_nonce: str, request_payload_sha256: str,
    worker_code_bundle_sha256: str, terminal_status: str, issuer: str,
    child_started: bool, elapsed_seconds: float, result: Mapping[str, Any],
) -> ConstraintManifoldPreprocessReceiptV4:
  fields = {
      "call_nonce": call_nonce,
      "request_payload_sha256": request_payload_sha256,
      "worker_code_bundle_sha256": worker_code_bundle_sha256,
      "terminal_status": terminal_status,
      "issuer": issuer,
      "child_started": child_started,
      "elapsed_seconds": max(0.0, _clean_float(elapsed_seconds)),
      "result_items": tuple(sorted(
          (str(key), _deep_freeze(value)) for key, value in result.items()
      )),
  }
  unsigned = {
      "schema_version": PREPROCESS_RECEIPT_SCHEMA_VERSION,
      "operation": PREPROCESS_OPERATION_V4,
      "call_nonce": call_nonce,
      "request_payload_sha256": request_payload_sha256,
      "worker_code_bundle_sha256": worker_code_bundle_sha256,
      "terminal_status": terminal_status,
      "issuer": issuer,
      "child_started": child_started,
      "elapsed_seconds": fields["elapsed_seconds"],
      "result": {key: _deep_thaw(value) for key, value in fields["result_items"]},
  }
  return ConstraintManifoldPreprocessReceiptV4(
      **fields, receipt_payload_sha256=canonical_sha256(unsigned),
  )


def _assert_sanitized_keys(value: Any, *, path: str = "sanitized_geometry") -> None:
  if isinstance(value, Mapping):
    for key, item in value.items():
      normalized = str(key).lower()
      if any(token in normalized for token in _FORBIDDEN_RESULT_KEYS):
        raise ValueError(f"preprocess child returned forbidden field at {path}.{key}")
      _assert_sanitized_keys(item, path=f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _assert_sanitized_keys(item, path=f"{path}[{index}]")


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != expected:
    raise ValueError(f"{label} exact schema differs")
  return value


def validate_sanitized_geometry_v4(payload: Mapping[str, Any]) -> None:
  program_index = int(payload.get("program_index", -1))
  if program_index not in {9, 11}:
    raise ValueError("sanitized geometry program differs")
  expected_top_level_keys = {
      "schema_version", "query_id_sha256", "program_index",
      "input_step_sha256s", "producer_source_sha256s", "endpoints",
      "cheap_geometry_candidates", "candidate_policy", "complete_cheap_domain",
      "candidate_domain_count", "candidate_domain_sha256", "compact_top_k",
      "compact_topk_payload_sha256",
      "oracle_inputs_absent", "final_test_touched", "sanitized_geometry_sha256",
  }
  if program_index == 11:
    expected_top_level_keys.add("ranking_schema")
    if payload.get("ranking_schema") in {
        P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1,
        P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2,
    }:
      expected_top_level_keys.add("diversity_coverage")
    if payload.get("ranking_schema") == P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2:
      expected_top_level_keys.update({"factorized_domain", "factorized_selection"})
  elif payload.get("ranking_schema") == (
      P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
  ):
    expected_top_level_keys.update({"ranking_schema", "collision_selection"})
  _require_exact_keys(payload, expected_top_level_keys, label="sanitized geometry")
  ranking_schema = payload.get("ranking_schema")
  expected_schema = SANITIZED_GEOMETRY_SCHEMA_VERSION
  if program_index == 9 and ranking_schema == (
      P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
  ):
    expected_schema = SANITIZED_GEOMETRY_P9_COLLISION_TOPK_SCHEMA_VERSION_V1
  elif program_index == 11:
    expected_schema = ({
        P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2:
        SANITIZED_GEOMETRY_P11_FACTORIZED_SCHEMA_VERSION_V2,
        P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1:
        SANITIZED_GEOMETRY_P11_STRATIFIED_DIVERSITY_SCHEMA_VERSION_V1,
        P11_BODY_AABB_RANK_SCHEMA_V1:
        SANITIZED_GEOMETRY_P11_BODY_AABB_SCHEMA_VERSION_V1,
    }).get(ranking_schema)
  if payload.get("schema_version") != expected_schema:
    raise ValueError("sanitized geometry schema differs")
  if (program_index == 11
      and ranking_schema not in {
          P11_BODY_AABB_RANK_SCHEMA_V1, P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1,
          P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2,
      }):
    raise ValueError("sanitized geometry ranking schema differs")
  if program_index == 9 and ranking_schema not in {
      None, P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1,
  }:
    raise ValueError("sanitized P9 ranking schema differs")
  if (
      payload.get("oracle_inputs_absent") is not True
      or payload.get("complete_cheap_domain") is not True
      or payload.get("final_test_touched") is not False
  ):
    raise ValueError("sanitized geometry oracle exclusion differs")
  _assert_sanitized_keys(payload)
  if _SHA256_RE.fullmatch(str(payload.get("query_id_sha256"))) is None:
    raise ValueError("sanitized query identity differs")
  input_hashes = payload.get("input_step_sha256s")
  if (
      not isinstance(input_hashes, list) or len(input_hashes) != 2
      or any(_SHA256_RE.fullmatch(str(value)) is None for value in input_hashes)
  ):
    raise ValueError("sanitized STEP commitments differ")
  if payload.get("producer_source_sha256s") != preprocess_source_sha256s_v4():
    raise ValueError("sanitized producer source binding differs")
  policy = payload.get("candidate_policy")
  expected_policy = ({
      P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1:
      p9_collision_policy_v1,
      P11_BODY_AABB_RANK_SCHEMA_V1: preprocess_policy_p11_body_aabb_v1,
      P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1:
      preprocess_policy_p11_stratified_diversity_v1,
  }).get(ranking_schema, lambda: preprocess_policy_for_program_v1(program_index))()
  if not isinstance(policy, Mapping) or policy != expected_policy:
    raise ValueError("sanitized geometry policy differs")
  endpoints = payload.get("endpoints")
  candidates = payload.get("cheap_geometry_candidates")
  if not isinstance(endpoints, list) or len(endpoints) != 2:
    raise ValueError("sanitized geometry endpoints differ")
  if not isinstance(candidates, list) or not candidates:
    raise ValueError("sanitized geometry candidate domain is empty")
  validate_compact_candidate_domain_v1(payload)
  if ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1:
    verify_p9_collision_selection_v1(
        candidate_domain_count=payload["candidate_domain_count"],
        candidate_domain_sha256=payload["candidate_domain_sha256"],
        compact_candidates=candidates,
        selection=payload["collision_selection"],
    )
  elif ranking_schema == P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2:
    factor_domain = payload["factorized_domain"]
    factor_selection = payload["factorized_selection"]
    verify_p11_factorized_domain_v2(factor_domain)
    verify_p11_factorized_selection_v2(factor_domain, factor_selection)
    if (
        factor_domain.get("schema_version") != P11_FACTORIZED_DOMAIN_SCHEMA_V2
        or factor_selection.get("ranking_schema")
        != P11_FACTORIZED_STRUCTURAL_TOPK_SCHEMA_V2
        or factor_domain["candidate_domain_count"] != payload["candidate_domain_count"]
        or factor_domain["candidate_domain_sha256"] != payload["candidate_domain_sha256"]
        or factor_selection["diversity_coverage"] != payload["diversity_coverage"]
    ):
      raise ValueError("p11 factorized authority binding differs")
  endpoint_keys = {
      "part_slot", "graph_face_index", "selected_raw_occ_face_index",
      "face_signature_sha256", "surface_type", "origin_world_mm",
      "rotation_local_to_world", "trim", "topology", "landmarks",
      "body_projected_aabb_local_2d",
  }
  landmark_keys = {
      "mesh_vertices_local_2d", "mesh_edge_midpoints_local_2d",
      "triangle_centroids_local_2d", "trim_vertices_local_2d",
      "trim_edge_midpoints_local_2d", "loop_centroids_local_2d",
      "boundary_edge_directions_local_2d", "adjacent_shared_edges",
  }
  mesh_keys = {
      "requested_linear_deflection_mm", "actual_triangulation_deflection_mm",
      "deflection_upper_bound_mm", "relative", "angular_deflection_radians",
      "parallel", "node_count", "triangle_count", "occ_version", "schema_version",
  }
  for index, endpoint in enumerate(endpoints):
    row = _require_exact_keys(endpoint, endpoint_keys, label=f"sanitized endpoint {index}")
    if row["part_slot"] != ("a", "b")[index]:
      raise ValueError("sanitized endpoint order differs")
    if _SHA256_RE.fullmatch(str(row["face_signature_sha256"])) is None:
      raise ValueError("sanitized endpoint face signature differs")
    rotation = np.asarray(row["rotation_local_to_world"], dtype=float)
    if rotation.shape != (3, 3):
      raise ValueError("sanitized endpoint frame differs")
    _v1._proper_rotation(rotation, label="sanitized endpoint frame")
    origin = np.asarray(row["origin_world_mm"], dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
      raise ValueError("sanitized endpoint origin differs")
    trim = row["trim"]
    if row["surface_type"] == "plane":
      trim_row = _require_exact_keys(trim, {
          "triangles_xy", "exact_surface_area_mm2", "boundary_length_mm",
          "wire_count", "hole_count", "overlap_area_error_upper_mm2", "mesh_proof",
      }, label="sanitized plane trim")
    elif row["surface_type"] == "cylinder":
      trim_row = _require_exact_keys(trim, {
          "triangles_uz", "radius_mm", "surface_side", "exact_surface_area_mm2",
          "boundary_length_mm", "wire_count", "hole_count",
          "seam_crossing_triangle_count", "overlap_area_error_upper_mm2", "mesh_proof",
      }, label="sanitized cylinder trim")
    else:
      raise ValueError("sanitized endpoint surface differs")
    _require_exact_keys(trim_row["mesh_proof"], mesh_keys, label="sanitized mesh proof")
    topology = _require_exact_keys(row["topology"], {
        "selected_edge_count", "selected_wire_count", "adjacent_faces",
        "step_topology_replayed",
    }, label="sanitized topology")
    for adjacent in topology["adjacent_faces"]:
      _require_exact_keys(adjacent, {
          "adjacent_face_signature_sha256", "surface_type", "shared_edge_count",
      }, label="sanitized adjacent face")
    landmark_row = _require_exact_keys(
        row["landmarks"], landmark_keys, label="sanitized landmarks",
    )
    for edge in landmark_row["adjacent_shared_edges"]:
      _require_exact_keys(edge, {
          "adjacent_face_signature_sha256", "vertices_local_2d",
          "midpoint_local_2d", "direction_local_2d", "length_mm",
      }, label="sanitized adjacent shared edge")
    _require_exact_keys(row["body_projected_aabb_local_2d"], {
        "minimum", "maximum",
    }, label="sanitized body projected AABB")
  keys = []
  common_metrics = {
      "trim_overlap_lower_bound", "trim_overlap_ratio_lower_bound",
      "body_projected_aabb_separation_mm",
      "body_projected_aabb_overlap_upper_mm2", "fixed_displacement_mm",
  }
  program_metrics = (
      {"normal_error_degrees", "normal_gap_mm"} if program_index == 9 else
      {"axis_error_degrees", "radial_axis_distance_mm", "axial_overlap_mm",
       "angular_overlap_radians", "body_frame_aabb_overlap_upper_mm3"}
  )
  for row in candidates:
    if not isinstance(row, Mapping):
      raise ValueError("sanitized candidate differs")
    _require_exact_keys(row, {
        "candidate_key", "candidate_evidence_sha256", "kind", "program_index",
        "yaw_radians", "normal_sign", "offset_local_2d", "a_landmark",
        "b_landmark", "alignment_events", "yaw_events",
        "base_world_delta_row_major", "world_delta_row_major",
        "child_world_row_major", "cheap_metrics",
    }, label="sanitized candidate")
    for event in (*row["alignment_events"], *row["yaw_events"]):
      if not isinstance(event, Mapping) or set(event) not in (
          {"kind", "a", "b"}, {"kind", "a_angle", "b_angle"},
      ):
        raise ValueError("candidate event exact schema differs")
    unsigned = dict(row)
    observed = unsigned.pop("candidate_key", None)
    evidence_sha = unsigned.pop("candidate_evidence_sha256", None)
    if evidence_sha != canonical_sha256(unsigned):
      raise ValueError("sanitized candidate evidence commitment differs")
    identity = {
        "schema_version": "constraint_manifold_candidate_identity.v4",
        "program_index": row["program_index"],
        "world_delta_row_major": row["world_delta_row_major"],
        "child_world_row_major": row["child_world_row_major"],
    }
    if observed != canonical_sha256(identity):
      raise ValueError("sanitized candidate commitment differs")
    if row.get("program_index") != program_index:
      raise ValueError("sanitized candidate program differs")
    metrics = row.get("cheap_metrics")
    expected_metrics = common_metrics | program_metrics | {
        "trim_overlap_estimate_mm2", "trim_overlap_error_upper_mm2",
    }
    if ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1:
      expected_metrics |= {
          "body_frame_aabb_intersection_volume_mm3",
          "body_frame_aabb_normalized_overlap",
      }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_metrics:
      raise ValueError("sanitized candidate cheap metrics differ")
    for matrix_key in (
        "base_world_delta_row_major", "world_delta_row_major",
        "child_world_row_major",
    ):
      _require_matrix(row[matrix_key], label=matrix_key)
    keys.append(observed)
  if ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1:
    ranked_keys = list(
        payload["collision_selection"]["compact_top32_candidate_keys"]
    )
  elif ranking_schema == P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2:
    coverage = payload["diversity_coverage"]
    factor_selection = payload["factorized_selection"]
    selected_keys = [
        row["candidate_key"]
        for row in factor_selection["selected_factor_memberships"]
    ]
    if (
        coverage != factor_selection["diversity_coverage"]
        or keys != selected_keys
    ):
      raise ValueError("p11 factorized selected transport differs")
    ranked_keys = keys
  elif ranking_schema == P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1:
    coverage = _require_exact_keys(payload.get("diversity_coverage"), {
        "schema_version", "stratum_definition", "yaw_bucket_definition",
        "quota_strata", "covered_quota_strata", "stage_a_candidate_count",
        "stage_b_quota_candidate_count", "selected_candidate_count",
        "selected_yaw_bucket_count", "coverage_payload_sha256",
    }, label="p11 diversity coverage")
    unsigned_coverage = dict(coverage)
    observed_coverage = unsigned_coverage.pop("coverage_payload_sha256")
    selected_coverage = sorted({
        stratum for row in candidates
        for stratum in candidate_strata_p11_diversity_v1(row)
    })
    selected_yaws = {
        _p11_yaw_state_v1(row)[1] for row in candidates
    }
    if (
        coverage["schema_version"] != "constraint_manifold_p11_diversity_coverage.v1"
        or observed_coverage != canonical_sha256(unsigned_coverage)
        or coverage["covered_quota_strata"] != selected_coverage
        or int(coverage["selected_candidate_count"]) != len(candidates)
        or int(coverage["selected_yaw_bucket_count"]) != len(selected_yaws)
        or not set(coverage["quota_strata"]).issubset(selected_coverage)
        or int(coverage["stage_a_candidate_count"]) < 0
        or int(coverage["stage_b_quota_candidate_count"]) < 0
        or int(coverage["stage_a_candidate_count"])
        + int(coverage["stage_b_quota_candidate_count"]) > len(candidates)
    ):
      raise ValueError("p11 diversity coverage commitment differs")
    ranked_keys = keys
  else:
    ranked_keys = [
        row["candidate_key"] for row in sorted(
            candidates,
            key=(
                (lambda row: candidate_rank_payload_v4(program_index, row))
                if program_index == 9 else candidate_rank_payload_p11_body_aabb_v1
            ),
        )
    ]
  if keys != ranked_keys or len(keys) != len(set(keys)):
    raise ValueError("sanitized candidate order/domain differs")
  unsigned = dict(payload)
  observed = unsigned.pop("sanitized_geometry_sha256", None)
  if observed != canonical_sha256(unsigned):
    raise ValueError("sanitized geometry commitment differs")


class SanitizedGeometryAuthorityV4:
  __slots__ = ("__weakref__",)

  def __init__(self, payload: Mapping[str, Any], *, _factory_token: object) -> None:
    if _factory_token is not _AUTHORITY_FACTORY_TOKEN:
      raise TypeError("sanitized geometry authority is supervisor-factory-only")
    copied = json.loads(json.dumps(payload, allow_nan=False))
    unsigned = dict(copied)
    observed = unsigned.pop("authority_payload_sha256", None)
    if (
        unsigned.get("schema_version") != SANITIZED_AUTHORITY_SCHEMA_VERSION
        or observed != canonical_sha256(unsigned)
    ):
      raise ValueError("sanitized geometry authority commitment differs")
    validate_sanitized_geometry_v4(unsigned["sanitized_geometry"])
    canonical_state = json.dumps(
        copied, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    geometry_sha256 = copied["sanitized_geometry"].get("sanitized_geometry_sha256")
    if _SHA256_RE.fullmatch(str(geometry_sha256)) is None:
      raise ValueError("sanitized geometry authority embedded commitment differs")
    _AUTHORITY_STATES[self] = canonical_state
    _AUTHORITY_EXPECTED_ATTESTATIONS[self] = (
        _SanitizedAuthorityExpectedAttestationV19(
            canonical_state_sha256=hashlib.sha256(canonical_state).hexdigest(),
            canonical_state_byte_count=len(canonical_state),
            authority_payload_sha256=str(observed),
            sanitized_geometry_sha256=str(geometry_sha256),
        )
    )

  def _validated_binding_payload(self) -> dict[str, Any]:
    try:
      canonical_state = _AUTHORITY_STATES[self]
      expected = _AUTHORITY_EXPECTED_ATTESTATIONS[self]
    except KeyError as error:
      raise ValueError("sanitized geometry authority private state absent") from error
    if type(canonical_state) is not bytes or (
        len(canonical_state) != expected.canonical_state_byte_count
        or hashlib.sha256(canonical_state).hexdigest()
        != expected.canonical_state_sha256
    ):
      raise ValueError("sanitized geometry authority tamper differs")
    try:
      payload = json.loads(canonical_state)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
      raise ValueError("sanitized geometry authority private bytes differ") from error
    if not isinstance(payload, dict):
      raise ValueError("sanitized geometry authority private payload differs")
    geometry = payload.get("sanitized_geometry")
    if (
        payload.get("schema_version") != SANITIZED_AUTHORITY_SCHEMA_VERSION
        or payload.get("authority_payload_sha256")
        != expected.authority_payload_sha256
        or not isinstance(geometry, dict)
        or geometry.get("sanitized_geometry_sha256")
        != expected.sanitized_geometry_sha256
    ):
      raise ValueError("sanitized geometry authority embedded commitment differs")
    return payload

  def revalidate(self) -> None:
    self._validated_binding_payload()

  def authenticated_snapshot(self) -> tuple[str, Mapping[str, Any]]:
    payload = self._validated_binding_payload()
    geometry = payload["sanitized_geometry"]
    return str(geometry["sanitized_geometry_sha256"]), geometry

  @property
  def sha256(self) -> str:
    # Stable across nonce-distinct replays of identical bytes/geometry.  The
    # nonce-specific launch/terminal binding remains available separately via
    # ``binding_payload`` and the terminal receipt.
    authority_sha256, _payload = self.authenticated_snapshot()
    return authority_sha256

  def payload(self) -> dict[str, Any]:
    _authority_sha256, payload = self.authenticated_snapshot()
    return payload

  def binding_payload(self) -> dict[str, Any]:
    return self._validated_binding_payload()

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("sanitized geometry authority is immutable")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("sanitized geometry authority is not serializable")


@dataclass(frozen=True, slots=True)
class ConstraintManifoldPreprocessRunV4:
  receipt: ConstraintManifoldPreprocessReceiptV4
  geometry_authority: SanitizedGeometryAuthorityV4 | None

  def __post_init__(self) -> None:
    if (self.receipt.terminal_status == "ok") != (self.geometry_authority is not None):
      raise ValueError("preprocess run authority/terminal status differs")


def _surface_type(face: Any) -> str:
  from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
  from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane  # type: ignore

  kind = BRepAdaptor_Surface(face.wrapped).GetType()
  if kind == GeomAbs_Plane:
    return "plane"
  if kind == GeomAbs_Cylinder:
    return "cylinder"
  return "other"


def _world_point(point: Any, world: Transform) -> np.ndarray:
  if hasattr(point, "toTuple"):
    point = point.toTuple()
  local = np.asarray(point, dtype=float).reshape(3)
  return np.asarray(world.rotation) @ local + np.asarray(world.translation)


def _face_local_2d(
    point: Any, *, world: Transform, geometry: ManifoldFaceGeometryV3,
) -> tuple[float, float]:
  value = _world_point(point, world)
  frame = np.asarray(geometry.rotation_local_to_world, dtype=float)
  delta = value - np.asarray(geometry.origin_world_mm, dtype=float)
  if geometry.surface_type == "plane":
    return _point2((np.dot(delta, frame[:, 0]), np.dot(delta, frame[:, 1])))
  return _point2((
      math.atan2(float(np.dot(delta, frame[:, 1])), float(np.dot(delta, frame[:, 0]))),
      np.dot(delta, frame[:, 2]),
  ))


def _periodic_delta(first: float, second: float) -> float:
  return math.atan2(math.sin(second - first), math.cos(second - first))


def _canonical_direction(value: Sequence[float]) -> tuple[float, float] | None:
  row = np.asarray(value, dtype=float).reshape(2)
  norm = float(np.linalg.norm(row))
  if not np.isfinite(row).all() or norm <= 1e-10:
    return None
  row /= norm
  if row[0] < -1e-12 or (abs(row[0]) <= 1e-12 and row[1] < 0.0):
    row = -row
  return _point2(row)


def _edge_landmark(
    edge: Any, *, world: Transform, geometry: ManifoldFaceGeometryV3,
) -> dict[str, Any]:
  samples = [_face_local_2d(edge.positionAt(value), world=world, geometry=geometry)
             for value in (0.0, 0.25, 0.5, 0.75, 1.0)]
  vertices = [_face_local_2d(vertex.Center(), world=world, geometry=geometry)
              for vertex in edge.Vertices()]
  vertices = list(canonicalize_landmark_points_v4(vertices or (samples[0], samples[-1]), maximum=6))
  start, end = samples[1], samples[3]
  delta = (
      _periodic_delta(start[0], end[0]) if geometry.surface_type == "cylinder"
      else end[0] - start[0],
      end[1] - start[1],
  )
  direction = _canonical_direction(delta)
  if direction is None:
    tangent = edge.tangentAt(0.5)
    centre = np.asarray(edge.positionAt(0.5).toTuple(), dtype=float)
    other = centre + np.asarray(tangent.toTuple(), dtype=float)
    first = _face_local_2d(centre, world=world, geometry=geometry)
    second = _face_local_2d(other, world=world, geometry=geometry)
    direction = _canonical_direction((
        _periodic_delta(first[0], second[0]) if geometry.surface_type == "cylinder"
        else second[0] - first[0],
        second[1] - first[1],
    ))
  return {
      "vertices_local_2d": [list(row) for row in vertices],
      "midpoint_local_2d": list(samples[2]),
      "direction_local_2d": None if direction is None else list(direction),
      "length_mm": _clean_float(edge.Length()),
  }


def _mean_points(rows: Sequence[Sequence[float]], *, periodic_first: bool) -> tuple[float, float]:
  values = np.asarray(rows, dtype=float)
  if periodic_first:
    first = math.atan2(float(np.mean(np.sin(values[:, 0]))),
                       float(np.mean(np.cos(values[:, 0]))))
  else:
    first = float(np.mean(values[:, 0]))
  return _point2((first, float(np.mean(values[:, 1]))))


def _mesh_landmarks(geometry: ManifoldFaceGeometryV3) -> dict[str, Any]:
  triangles = (
      geometry.plane_trim.triangles_xy if geometry.plane_trim is not None
      else geometry.cylinder_trim.triangles_uz  # type: ignore[union-attr]
  )
  vertices = [point for triangle in triangles for point in triangle]
  edge_midpoints = []
  centroids = []
  for triangle in triangles:
    values = np.asarray(triangle, dtype=float)
    centroids.append(np.mean(values, axis=0))
    for index in range(3):
      first, second = values[index], values[(index + 1) % 3]
      if geometry.surface_type == "cylinder":
        midpoint = (first[0] + 0.5 * _periodic_delta(first[0], second[0]),
                    0.5 * (first[1] + second[1]))
      else:
        midpoint = 0.5 * (first + second)
      edge_midpoints.append(midpoint)
  return {
      "mesh_vertices_local_2d": [list(row) for row in canonicalize_landmark_points_v4(
          vertices, maximum=max(1, len(vertices)),
      )],
      "mesh_edge_midpoints_local_2d": [list(row) for row in canonicalize_landmark_points_v4(
          edge_midpoints, maximum=max(1, len(edge_midpoints)),
      )],
      "triangle_centroids_local_2d": [list(row) for row in canonicalize_landmark_points_v4(
          centroids, maximum=max(1, len(centroids)),
      )],
  }


def _canonical_triangle(
    row: Sequence[Sequence[float]], *, periodic_first: bool,
) -> tuple[tuple[float, float], ...]:
  points = [_point2(point) for point in row]
  if periodic_first:
    centre = sum(point[0] for point in points) / 3.0
    shift = 2.0 * math.pi * math.floor((centre + math.pi) / (2.0 * math.pi))
    points = [(_clean_float(point[0] - shift), point[1]) for point in points]
  permutations = []
  for values in (points, list(reversed(points))):
    for index in range(3):
      permutations.append(tuple(values[index:] + values[:index]))
  return min(permutations)


def _canonical_geometry(geometry: ManifoldFaceGeometryV3) -> ManifoldFaceGeometryV3:
  common = {
      "part_slot": geometry.part_slot,
      "graph_face_index": geometry.graph_face_index,
      "raw_occ_face_index": geometry.raw_occ_face_index,
      "face_signature_sha256": geometry.face_signature_sha256,
      "surface_type": geometry.surface_type,
      "origin_world_mm": geometry.origin_world_mm,
      "rotation_local_to_world": geometry.rotation_local_to_world,
  }
  if geometry.plane_trim is not None:
    trim = geometry.plane_trim
    triangles = tuple(sorted({
        _canonical_triangle(row, periodic_first=False) for row in trim.triangles_xy
    }))
    return ManifoldFaceGeometryV3(
        **common,
        plane_trim=PlaneTrimV3(
            triangles_xy=triangles,
            exact_surface_area_mm2=trim.exact_surface_area_mm2,
            boundary_length_mm=trim.boundary_length_mm,
            wire_count=trim.wire_count, hole_count=trim.hole_count,
            mesh_proof=trim.mesh_proof,
        ),
    )
  trim = geometry.cylinder_trim
  assert trim is not None
  triangles = tuple(sorted({
      _canonical_triangle(row, periodic_first=True) for row in trim.triangles_uz
  }))
  return ManifoldFaceGeometryV3(
      **common,
      cylinder_trim=CylinderTrimV3(
          triangles_uz=triangles, radius_mm=trim.radius_mm,
          surface_side=trim.surface_side,
          exact_surface_area_mm2=trim.exact_surface_area_mm2,
          boundary_length_mm=trim.boundary_length_mm,
          wire_count=trim.wire_count, hole_count=trim.hole_count,
          seam_crossing_triangle_count=trim.seam_crossing_triangle_count,
          mesh_proof=trim.mesh_proof,
      ),
  )


def _shared_selected_edge_indices_v2(
    selected_edges: Sequence[Any], candidate_edges: Sequence[Any],
) -> tuple[int, ...]:
  """Match OCC edges by hash bucket, then confirm exact IsSame identity."""

  selected_by_hash: dict[int, list[int]] = {}
  for index, edge in enumerate(selected_edges):
    selected_by_hash.setdefault(int(edge.hashCode()), []).append(index)
  matched: set[int] = set()
  for candidate in candidate_edges:
    for index in selected_by_hash.get(int(candidate.hashCode()), ()):
      if selected_edges[index].wrapped.IsSame(candidate.wrapped):
        matched.add(index)
  return tuple(sorted(matched))


def _endpoint_landmarks(
    *, shape: Any, selected_index: int, world: Transform,
    geometry: ManifoldFaceGeometryV3,
) -> tuple[dict[str, Any], dict[str, Any]]:
  faces = tuple(shape.Faces())
  selected = faces[selected_index]
  selected_edges = tuple(selected.Edges())
  edge_landmarks_by_index = [
      _edge_landmark(edge, world=world, geometry=geometry)
      for edge in selected_edges
  ]
  edge_rows = list(edge_landmarks_by_index)
  edge_rows.sort(key=canonical_sha256)
  trim_vertices = [row for edge in edge_rows for row in edge["vertices_local_2d"]]
  trim_midpoints = [row["midpoint_local_2d"] for row in edge_rows]
  directions = [row["direction_local_2d"] for row in edge_rows
                if row["direction_local_2d"] is not None]
  loops = []
  for wire in selected.Wires():
    points = [
        _face_local_2d(edge.positionAt(0.5), world=world, geometry=geometry)
        for edge in wire.Edges()
    ]
    if points:
      loops.append(_mean_points(points, periodic_first=geometry.surface_type == "cylinder"))
  adjacent_shared = []
  topology_faces = []
  for other_index, other in enumerate(faces):
    if other_index == selected_index:
      continue
    shared = [
        edge_landmarks_by_index[index]
        for index in _shared_selected_edge_indices_v2(
            selected_edges, tuple(other.Edges()),
        )
    ]
    if not shared:
      continue
    signature = source_face_signature_sha256(other)
    shared.sort(key=canonical_sha256)
    topology_faces.append({
        "adjacent_face_signature_sha256": signature,
        "surface_type": _surface_type(other),
        "shared_edge_count": len(shared),
    })
    for row in shared:
      adjacent_shared.append({
          "adjacent_face_signature_sha256": signature,
          **row,
      })
  adjacent_shared.sort(key=canonical_sha256)
  topology_faces.sort(key=canonical_sha256)
  mesh = _mesh_landmarks(geometry)
  landmarks = {
      **mesh,
      "trim_vertices_local_2d": [list(row) for row in canonicalize_landmark_points_v4(
          trim_vertices, maximum=max(1, len(trim_vertices)),
      )],
      "trim_edge_midpoints_local_2d": [list(row) for row in canonicalize_landmark_points_v4(
          trim_midpoints, maximum=max(1, len(trim_midpoints)),
      )],
      "loop_centroids_local_2d": [list(row) for row in canonicalize_landmark_points_v4(
          loops, maximum=max(1, len(loops)),
      )],
      "boundary_edge_directions_local_2d": [list(row) for row in canonicalize_landmark_points_v4(
          directions, maximum=max(1, len(directions)),
      )],
      "adjacent_shared_edges": adjacent_shared,
  }
  topology = {
      "selected_edge_count": len(selected_edges),
      "selected_wire_count": len(tuple(selected.Wires())),
      "adjacent_faces": topology_faces,
      "step_topology_replayed": True,
  }
  return landmarks, topology


def _body_world_corners(shape: Any, world: Transform) -> np.ndarray:
  low, high = shape_bbox(shape)
  local = np.asarray([
      (x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1])
      for z in (low[2], high[2])
  ], dtype=float)
  return local @ np.asarray(world.rotation).T + np.asarray(world.translation)


def _load_captured_step_shape_v4(
    source: Path, *, expected_sha256: str,
) -> Any:
  """Load the exact byte capture that was hashed, closing the STEP TOCTOU seam."""

  captured = source.read_bytes()
  if hashlib.sha256(captured).hexdigest() != expected_sha256:
    raise ValueError("step_bytes")
  temporary_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".step", prefix="neurocad-preprocess-v4-", delete=False,
    ) as stream:
      stream.write(captured)
      stream.flush()
      os.fsync(stream.fileno())
      temporary_path = Path(stream.name)
    if file_sha256(temporary_path) != expected_sha256:
      raise ValueError("captured_step_bytes")
    shape = load_step_shape(temporary_path)
    if file_sha256(temporary_path) != expected_sha256:
      raise ValueError("captured_step_changed_during_load")
    return shape
  finally:
    if temporary_path is not None:
      try:
        temporary_path.unlink(missing_ok=True)
      except OSError:
        pass


def _project_aabb(corners: np.ndarray, geometry: ManifoldFaceGeometryV3) -> tuple[np.ndarray, np.ndarray]:
  frame = np.asarray(geometry.rotation_local_to_world, dtype=float)
  origin = np.asarray(geometry.origin_world_mm, dtype=float)
  local = (np.asarray(corners, dtype=float) - origin) @ frame
  return np.min(local[:, :2], axis=0), np.max(local[:, :2], axis=0)


def _aabb_payload(bounds: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
  return {"minimum": list(_point2(bounds[0])), "maximum": list(_point2(bounds[1]))}


def _capped_points(landmarks: Mapping[str, Any], key: str) -> tuple[tuple[float, float], ...]:
  if key == "adjacent_shared_edge_vertices_local_2d":
    rows = [point for edge in landmarks["adjacent_shared_edges"]
            for point in edge["vertices_local_2d"]]
  elif key == "adjacent_shared_edge_midpoints_local_2d":
    rows = [edge["midpoint_local_2d"] for edge in landmarks["adjacent_shared_edges"]]
  else:
    rows = landmarks[key]
  return canonicalize_landmark_points_v4(
      rows, maximum=LANDMARK_CAP_PER_ENDPOINT_PER_TYPE_V4,
  ) if rows else ()


def _alignment_pairs(
    first: Mapping[str, Any], second: Mapping[str, Any], *, coordinate: int | None,
) -> list[dict[str, Any]]:
  rows = []
  for kind in _ALIGNMENT_TYPES_V4:
    left, right = _capped_points(first, kind), _capped_points(second, kind)
    for a in left:
      for b in right:
        rows.append({
            "kind": kind,
            "a": list(a), "b": list(b),
            "difference": (
                list(_point2(np.asarray(a) - np.asarray(b))) if coordinate is None
                else _clean_float(a[coordinate] - b[coordinate])
            ),
        })
  rows.sort(key=canonical_sha256)
  return rows


def _candidate_payload(unsigned: Mapping[str, Any]) -> dict[str, Any]:
  normalized = json.loads(json.dumps(unsigned, allow_nan=False))
  identity = {
      "schema_version": "constraint_manifold_candidate_identity.v4",
      "program_index": normalized["program_index"],
      "world_delta_row_major": normalized["world_delta_row_major"],
      "child_world_row_major": normalized["child_world_row_major"],
  }
  return {
      "candidate_key": canonical_sha256(identity),
      "candidate_evidence_sha256": canonical_sha256(normalized),
      **normalized,
  }


def _body_proxy_metrics(
    corners_a: np.ndarray, corners_b_current: np.ndarray,
    delta: np.ndarray, reference: ManifoldFaceGeometryV3,
) -> tuple[float, float]:
  homogeneous = np.concatenate((corners_b_current, np.ones((len(corners_b_current), 1))), axis=1)
  moved_b = (homogeneous @ delta.T)[:, :3]
  low_a, high_a = _project_aabb(corners_a, reference)
  low_b, high_b = _project_aabb(moved_b, reference)
  gaps = np.maximum(0.0, np.maximum(low_a - high_b, low_b - high_a))
  overlap = np.maximum(0.0, np.minimum(high_a, high_b) - np.maximum(low_a, low_b))
  return _clean_float(np.linalg.norm(gaps)), _clean_float(np.prod(overlap))


def _body_frame_aabb_overlap_upper_mm3_v1(
    corners_a: np.ndarray, corners_b_current: np.ndarray,
    delta: np.ndarray, reference: ManifoldFaceGeometryV3,
) -> float:
  """Conservative full-3D body AABB overlap in the selected cylinder frame."""

  homogeneous = np.concatenate((
      np.asarray(corners_b_current, dtype=float),
      np.ones((len(corners_b_current), 1)),
  ), axis=1)
  moved_b = (homogeneous @ np.asarray(delta, dtype=float).T)[:, :3]
  frame = np.asarray(reference.rotation_local_to_world, dtype=float)
  origin = np.asarray(reference.origin_world_mm, dtype=float)
  local_a = (np.asarray(corners_a, dtype=float) - origin) @ frame
  local_b = (moved_b - origin) @ frame
  low_a, high_a = np.min(local_a, axis=0), np.max(local_a, axis=0)
  low_b, high_b = np.min(local_b, axis=0), np.max(local_b, axis=0)
  overlap = np.maximum(0.0, np.minimum(high_a, high_b) - np.maximum(low_a, low_b))
  return _clean_float(np.prod(overlap))


@dataclass(frozen=True, slots=True)
class PlaneYawStateOverlapIndexV1:
  first: np.ndarray
  second_base: np.ndarray
  first_low: np.ndarray
  first_high: np.ndarray
  second_low: np.ndarray
  second_high: np.ndarray
  x_global_low: float
  x_global_high: float
  x_bin_count: int
  x_bin_width: float
  x_bins: Mapping[int, tuple[int, ...]]
  x_bin_offsets: np.ndarray
  x_bin_right_indices: np.ndarray


@njit(cache=True, fastmath=False)
def _cross2_numba_v1(ax: float, ay: float, bx: float, by: float) -> float:
  return ax * by - ay * bx


@njit(cache=True, fastmath=False)
def _triangle_clip_area_numba_v1(subject: np.ndarray, raw_clip: np.ndarray) -> float:
  """Sutherland-Hodgman triangle clip with the frozen V1 tolerances."""

  clip = raw_clip.copy()
  signed = 0.5 * _cross2_numba_v1(
      clip[1, 0] - clip[0, 0], clip[1, 1] - clip[0, 1],
      clip[2, 0] - clip[0, 0], clip[2, 1] - clip[0, 1],
  )
  if signed < 0.0:
    x, y = clip[0, 0], clip[0, 1]
    clip[0, 0], clip[0, 1] = clip[2, 0], clip[2, 1]
    clip[2, 0], clip[2, 1] = x, y
  output = np.empty((8, 2), dtype=np.float64)
  previous = np.empty((8, 2), dtype=np.float64)
  for index in range(3):
    output[index, 0], output[index, 1] = subject[index, 0], subject[index, 1]
  output_count = 3
  for edge_index in range(3):
    if output_count == 0:
      break
    previous_count = output_count
    for index in range(previous_count):
      previous[index, 0], previous[index, 1] = output[index, 0], output[index, 1]
    output_count = 0
    edge_a = clip[edge_index]
    edge_b = clip[(edge_index + 1) % 3]
    edge_x, edge_y = edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]
    start_x, start_y = previous[previous_count - 1]
    start_inside = _cross2_numba_v1(
        edge_x, edge_y, start_x - edge_a[0], start_y - edge_a[1],
    ) >= -1e-10
    for index in range(previous_count):
      end_x, end_y = previous[index]
      end_inside = _cross2_numba_v1(
          edge_x, edge_y, end_x - edge_a[0], end_y - edge_a[1],
      ) >= -1e-10
      if end_inside:
        if not start_inside:
          direction_x, direction_y = end_x - start_x, end_y - start_y
          denominator = _cross2_numba_v1(
              direction_x, direction_y, edge_x, edge_y,
          )
          if abs(denominator) <= 1e-15:
            ix, iy = (start_x + end_x) * 0.5, (start_y + end_y) * 0.5
          else:
            parameter = _cross2_numba_v1(
                edge_a[0] - start_x, edge_a[1] - start_y, edge_x, edge_y,
            ) / denominator
            ix = start_x + parameter * direction_x
            iy = start_y + parameter * direction_y
          output[output_count, 0], output[output_count, 1] = ix, iy
          output_count += 1
        output[output_count, 0], output[output_count, 1] = end_x, end_y
        output_count += 1
      elif start_inside:
        direction_x, direction_y = end_x - start_x, end_y - start_y
        denominator = _cross2_numba_v1(direction_x, direction_y, edge_x, edge_y)
        if abs(denominator) <= 1e-15:
          ix, iy = (start_x + end_x) * 0.5, (start_y + end_y) * 0.5
        else:
          parameter = _cross2_numba_v1(
              edge_a[0] - start_x, edge_a[1] - start_y, edge_x, edge_y,
          ) / denominator
          ix = start_x + parameter * direction_x
          iy = start_y + parameter * direction_y
        output[output_count, 0], output[output_count, 1] = ix, iy
        output_count += 1
      start_x, start_y, start_inside = end_x, end_y, end_inside
  if output_count < 3:
    return 0.0
  doubled = 0.0
  for index in range(output_count):
    following = (index + 1) % output_count
    doubled += _cross2_numba_v1(
        output[following, 0], output[following, 1],
        output[index, 0], output[index, 1],
    )
  return 0.5 * abs(doubled)


@njit(cache=True, fastmath=False)
def _triangle_pair_areas_numba_v1(
    left: np.ndarray, right: np.ndarray,
    left_indices: np.ndarray, right_indices: np.ndarray,
) -> np.ndarray:
  areas = np.empty(len(left_indices), dtype=np.float64)
  for index in range(len(left_indices)):
    areas[index] = _triangle_clip_area_numba_v1(
        left[left_indices[index]], right[right_indices[index]],
    )
  return areas


@njit(cache=True, fastmath=False)
def _indexed_aabb_overlap_pairs_numba_v2(
    left_low: np.ndarray, left_high: np.ndarray,
    right_low: np.ndarray, right_high: np.ndarray,
    bin_offsets: np.ndarray, bin_right_indices: np.ndarray,
    global_low: float, global_high: float, bin_width: float,
    delta_low_x: float, delta_high_x: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Return exact AABB survivors in legacy left/right row-major order."""

  right_count = len(right_low)
  bin_count = len(bin_offsets) - 1

  def survivor_count() -> int:
    marks = np.zeros(right_count, dtype=np.int64)
    scratch = np.empty(right_count, dtype=np.int64)
    epoch = 0
    total = 0
    for left_index in range(len(left_low)):
      query_low = np.nextafter(
          left_low[left_index, 0] - delta_high_x, -np.inf,
      )
      query_high = np.nextafter(
          left_high[left_index, 0] - delta_low_x, np.inf,
      )
      if query_high < global_low or query_low > global_high:
        continue
      start_bin = int(math.floor((query_low - global_low) / bin_width))
      stop_bin = int(math.floor((query_high - global_low) / bin_width))
      start_bin = max(0, min(bin_count - 1, start_bin))
      stop_bin = max(0, min(bin_count - 1, stop_bin))
      epoch += 1
      touched = 0
      for bin_index in range(start_bin, stop_bin + 1):
        for position in range(
            bin_offsets[bin_index], bin_offsets[bin_index + 1],
        ):
          right_index = bin_right_indices[position]
          if marks[right_index] != epoch:
            marks[right_index] = epoch
            scratch[touched] = right_index
            touched += 1
      ordered = np.sort(scratch[:touched])
      for position in range(touched):
        right_index = ordered[position]
        if (
            right_high[right_index, 0] >= left_low[left_index, 0]
            and right_low[right_index, 0] <= left_high[left_index, 0]
            and right_high[right_index, 1] >= left_low[left_index, 1]
            and right_low[right_index, 1] <= left_high[left_index, 1]
        ):
          total += 1
    return total

  total = survivor_count()
  left_output = np.empty(total, dtype=np.int64)
  right_output = np.empty(total, dtype=np.int64)
  marks = np.zeros(right_count, dtype=np.int64)
  scratch = np.empty(right_count, dtype=np.int64)
  epoch = 0
  output_index = 0
  for left_index in range(len(left_low)):
    query_low = np.nextafter(
        left_low[left_index, 0] - delta_high_x, -np.inf,
    )
    query_high = np.nextafter(
        left_high[left_index, 0] - delta_low_x, np.inf,
    )
    if query_high < global_low or query_low > global_high:
      continue
    start_bin = int(math.floor((query_low - global_low) / bin_width))
    stop_bin = int(math.floor((query_high - global_low) / bin_width))
    start_bin = max(0, min(bin_count - 1, start_bin))
    stop_bin = max(0, min(bin_count - 1, stop_bin))
    epoch += 1
    touched = 0
    for bin_index in range(start_bin, stop_bin + 1):
      for position in range(
          bin_offsets[bin_index], bin_offsets[bin_index + 1],
      ):
        right_index = bin_right_indices[position]
        if marks[right_index] != epoch:
          marks[right_index] = epoch
          scratch[touched] = right_index
          touched += 1
    ordered = np.sort(scratch[:touched])
    for position in range(touched):
      right_index = ordered[position]
      if (
          right_high[right_index, 0] >= left_low[left_index, 0]
          and right_low[right_index, 0] <= left_high[left_index, 0]
          and right_high[right_index, 1] >= left_low[left_index, 1]
          and right_low[right_index, 1] <= left_high[left_index, 1]
      ):
        left_output[output_index] = left_index
        right_output[output_index] = right_index
        output_index += 1
  return left_output, right_output


@njit(cache=True, fastmath=False)
def _indexed_overlap_areas_for_offsets_numba_v2(
    left: np.ndarray, left_low: np.ndarray, left_high: np.ndarray,
    right_batch: np.ndarray, second_low: np.ndarray, second_high: np.ndarray,
    bin_offsets: np.ndarray, bin_right_indices: np.ndarray,
    global_low: float, global_high: float, bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Evaluate a complete yaw's offsets without per-offset pair allocations."""

  offset_count = len(right_batch)
  right_count = len(second_low)
  bin_count = len(bin_offsets) - 1
  totals = np.zeros(offset_count, dtype=np.float64)
  exact_counts = np.zeros(offset_count, dtype=np.int64)
  right_low = np.empty((right_count, 2), dtype=np.float64)
  right_high = np.empty((right_count, 2), dtype=np.float64)
  positions = np.empty(bin_count, dtype=np.int64)
  ends = np.empty(bin_count, dtype=np.int64)

  for offset_index in range(offset_count):
    right = right_batch[offset_index]
    delta_low_x = np.inf
    delta_high_x = -np.inf
    for right_index in range(right_count):
      low_x = min(
          right[right_index, 0, 0],
          right[right_index, 1, 0],
          right[right_index, 2, 0],
      )
      high_x = max(
          right[right_index, 0, 0],
          right[right_index, 1, 0],
          right[right_index, 2, 0],
      )
      low_y = min(
          right[right_index, 0, 1],
          right[right_index, 1, 1],
          right[right_index, 2, 1],
      )
      high_y = max(
          right[right_index, 0, 1],
          right[right_index, 1, 1],
          right[right_index, 2, 1],
      )
      right_low[right_index, 0] = np.nextafter(low_x, -np.inf)
      right_low[right_index, 1] = np.nextafter(low_y, -np.inf)
      right_high[right_index, 0] = np.nextafter(high_x, np.inf)
      right_high[right_index, 1] = np.nextafter(high_y, np.inf)
      delta_low_x = min(
          delta_low_x,
          right_low[right_index, 0] - second_low[right_index, 0],
      )
      delta_high_x = max(
          delta_high_x,
          right_high[right_index, 0] - second_high[right_index, 0],
      )

    total = 0.0
    exact_count = 0
    for left_index in range(len(left)):
      query_low = np.nextafter(
          left_low[left_index, 0] - delta_high_x, -np.inf,
      )
      query_high = np.nextafter(
          left_high[left_index, 0] - delta_low_x, np.inf,
      )
      if query_high < global_low or query_low > global_high:
        continue
      start_bin = int(math.floor((query_low - global_low) / bin_width))
      stop_bin = int(math.floor((query_high - global_low) / bin_width))
      start_bin = max(0, min(bin_count - 1, start_bin))
      stop_bin = max(0, min(bin_count - 1, stop_bin))
      for bin_index in range(start_bin, stop_bin + 1):
        positions[bin_index] = bin_offsets[bin_index]
        ends[bin_index] = bin_offsets[bin_index + 1]

      # Each bin list is already in ascending right-index order. Merge the
      # relevant lists and advance duplicates together to preserve the frozen
      # dense-mask row-major clip order without sorting or pair arrays.
      while True:
        right_index = right_count
        for bin_index in range(start_bin, stop_bin + 1):
          position = positions[bin_index]
          if (
              position < ends[bin_index]
              and bin_right_indices[position] < right_index
          ):
            right_index = bin_right_indices[position]
        if right_index == right_count:
          break
        for bin_index in range(start_bin, stop_bin + 1):
          position = positions[bin_index]
          while (
              position < ends[bin_index]
              and bin_right_indices[position] == right_index
          ):
            position += 1
          positions[bin_index] = position
        if (
            right_high[right_index, 0] >= left_low[left_index, 0]
            and right_low[right_index, 0] <= left_high[left_index, 0]
            and right_high[right_index, 1] >= left_low[left_index, 1]
            and right_low[right_index, 1] <= left_high[left_index, 1]
        ):
          exact_count += 1
          total += _triangle_clip_area_numba_v1(
              left[left_index], right[right_index],
          )
    totals[offset_index] = total
    exact_counts[offset_index] = exact_count
  return totals, exact_counts


def build_plane_yaw_state_overlap_index_v1(
    first: Sequence[Sequence[Sequence[float]] | np.ndarray],
    second_base: Sequence[Sequence[Sequence[float]] | np.ndarray],
    *, _profile: dict[str, int | float] | None = None,
) -> PlaneYawStateOverlapIndexV1:
  """Build the conservative AABB index shared by every offset of one yaw."""

  started = time.perf_counter()
  left = np.asarray(first, dtype=float)
  right = np.asarray(second_base, dtype=float)
  if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != (3, 2) or right.shape[1:] != (3, 2):
    raise ValueError("plane overlap triangle domain differs")
  if not np.isfinite(left).all() or not np.isfinite(right).all():
    raise ValueError("plane overlap triangle domain is non-finite")
  left_low = np.nextafter(np.min(left, axis=1), -np.inf)
  left_high = np.nextafter(np.max(left, axis=1), np.inf)
  right_low = np.nextafter(np.min(right, axis=1), -np.inf)
  right_high = np.nextafter(np.max(right, axis=1), np.inf)
  global_low = float(np.min(right_low[:, 0]))
  global_high = float(np.max(right_high[:, 0]))
  bin_count = max(1, min(4096, int(math.ceil(math.sqrt(len(right))))))
  width = max(global_high - global_low, np.finfo(float).tiny) / bin_count

  def cell(value: float) -> int:
    return min(bin_count - 1, max(0, int(math.floor((value - global_low) / width))))

  bins: dict[int, list[int]] = {}
  for index, bounds in enumerate(zip(right_low, right_high, strict=True)):
    for bin_index in range(cell(float(bounds[0][0])), cell(float(bounds[1][0])) + 1):
      bins.setdefault(bin_index, []).append(index)
  bin_offsets = [0]
  bin_right_indices = []
  for bin_index in range(bin_count):
    bin_right_indices.extend(bins.get(bin_index, ()))
    bin_offsets.append(len(bin_right_indices))
  if _profile is not None:
    _profile["overlap_index_build_count"] = int(
        _profile.get("overlap_index_build_count", 0)
    ) + 1
    _profile["overlap_index_elapsed_seconds_raw"] = float(
        _profile.get("overlap_index_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - started)
  return PlaneYawStateOverlapIndexV1(
      first=left, second_base=right, first_low=left_low, first_high=left_high,
      second_low=right_low, second_high=right_high,
      x_global_low=global_low, x_global_high=global_high,
      x_bin_count=bin_count, x_bin_width=width,
      x_bins=MappingProxyType({key: tuple(value) for key, value in bins.items()}),
      x_bin_offsets=np.asarray(bin_offsets, dtype=np.int64),
      x_bin_right_indices=np.asarray(bin_right_indices, dtype=np.int64),
  )


def plane_overlap_area_from_yaw_state_index_v1(
    index: PlaneYawStateOverlapIndexV1, *, offset_xy: Sequence[float],
    translated_second: Sequence[Sequence[Sequence[float]] | np.ndarray] | None = None,
    _profile: dict[str, int | float] | None = None,
) -> float:
  """Evaluate one translated offset without rebuilding its yaw-state index."""

  offset = np.asarray(offset_xy, dtype=float)
  if offset.shape != (2,) or not np.isfinite(offset).all():
    raise ValueError("plane overlap offset differs")
  right = np.asarray(
      index.second_base + offset if translated_second is None else translated_second,
      dtype=float,
  )
  if right.shape != index.second_base.shape or not np.isfinite(right).all():
    raise ValueError("plane translated triangle domain differs")
  right_low = np.nextafter(np.min(right, axis=1), -np.inf)
  right_high = np.nextafter(np.max(right, axis=1), np.inf)

  exact_started = time.perf_counter()
  dense_pair_count = len(index.first) * len(right)
  delta_low_x = float(np.min(
      right_low[:, 0] - index.second_low[:, 0]
  ))
  delta_high_x = float(np.max(
      right_high[:, 0] - index.second_high[:, 0]
  ))
  left_indices, right_indices = _indexed_aabb_overlap_pairs_numba_v2(
      index.first_low, index.first_high, right_low, right_high,
      index.x_bin_offsets, index.x_bin_right_indices,
      index.x_global_low, index.x_global_high, index.x_bin_width,
      delta_low_x, delta_high_x,
  )
  areas = _triangle_pair_areas_numba_v1(
      index.first, right, left_indices, right_indices,
  )
  # Accumulate in the exact legacy (left_index, right_index) row-major order.
  total = 0.0
  for area in areas:
    total += float(area)
  exact_count = len(left_indices)
  if _profile is not None:
    _profile["overlap_dense_aabb_pair_count"] = int(
        _profile.get("overlap_dense_aabb_pair_count", 0)
    ) + dense_pair_count
    _profile["overlap_index_candidate_pair_count"] = int(
        _profile.get("overlap_index_candidate_pair_count", 0)
    ) + exact_count
    _profile["overlap_exact_clip_pair_count"] = int(
        _profile.get("overlap_exact_clip_pair_count", 0)
    ) + exact_count
    _profile["overlap_numba_batch_pair_count"] = int(
        _profile.get("overlap_numba_batch_pair_count", 0)
    ) + exact_count
    _profile["overlap_numba_batch_call_count"] = int(
        _profile.get("overlap_numba_batch_call_count", 0)
    ) + 1
    _profile["overlap_exact_elapsed_seconds_raw"] = float(
        _profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - exact_started)
  return float(total)


def plane_overlap_areas_from_yaw_state_index_v2(
    index: PlaneYawStateOverlapIndexV1, *,
    offsets_xy: Sequence[Sequence[float]],
    translated_seconds: Sequence[
        Sequence[Sequence[Sequence[float]] | np.ndarray]
    ],
    _profile: dict[str, int | float] | None = None,
) -> np.ndarray:
  """Evaluate every offset of one yaw in a single deterministic Numba call."""

  offsets = np.asarray(offsets_xy, dtype=float)
  rights = np.asarray(translated_seconds, dtype=float)
  if offsets.ndim != 2 or offsets.shape[1:] != (2,) or not np.isfinite(offsets).all():
    raise ValueError("plane overlap offset batch differs")
  expected_shape = (len(offsets), *index.second_base.shape)
  if rights.shape != expected_shape or not np.isfinite(rights).all():
    raise ValueError("plane translated triangle batch differs")
  exact_started = time.perf_counter()
  totals, exact_counts = _indexed_overlap_areas_for_offsets_numba_v2(
      index.first, index.first_low, index.first_high,
      rights, index.second_low, index.second_high,
      index.x_bin_offsets, index.x_bin_right_indices,
      index.x_global_low, index.x_global_high, index.x_bin_width,
  )
  if _profile is not None:
    dense_pair_count = len(index.first) * len(index.second_base) * len(offsets)
    exact_count = int(np.sum(exact_counts))
    _profile["overlap_dense_aabb_pair_count"] = int(
        _profile.get("overlap_dense_aabb_pair_count", 0)
    ) + dense_pair_count
    _profile["overlap_index_candidate_pair_count"] = int(
        _profile.get("overlap_index_candidate_pair_count", 0)
    ) + exact_count
    _profile["overlap_exact_clip_pair_count"] = int(
        _profile.get("overlap_exact_clip_pair_count", 0)
    ) + exact_count
    _profile["overlap_numba_batch_pair_count"] = int(
        _profile.get("overlap_numba_batch_pair_count", 0)
    ) + exact_count
    # Preserve the established per-offset accounting while exposing the
    # physical number of compiled calls separately.
    _profile["overlap_numba_batch_call_count"] = int(
        _profile.get("overlap_numba_batch_call_count", 0)
    ) + len(offsets)
    _profile["overlap_yaw_batch_call_count"] = int(
        _profile.get("overlap_yaw_batch_call_count", 0)
    ) + 1
    _profile["overlap_exact_elapsed_seconds_raw"] = float(
        _profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - exact_started)
  return totals


def plane_overlap_area_pruned_v4(
    first: Sequence[Sequence[Sequence[float]] | np.ndarray],
    second: Sequence[Sequence[Sequence[float]] | np.ndarray],
    *, _profile: dict[str, int | float] | None = None,
) -> float:
  """Brute-force-equivalent overlap with conservative triangle-AABB pruning."""

  index_started = time.perf_counter()
  left = np.asarray(first, dtype=float)
  right = np.asarray(second, dtype=float)
  if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != (3, 2) or right.shape[1:] != (3, 2):
    raise ValueError("plane overlap triangle domain differs")
  if not np.isfinite(left).all() or not np.isfinite(right).all():
    raise ValueError("plane overlap triangle domain is non-finite")
  left_low = np.nextafter(np.min(left, axis=1), -np.inf)
  left_high = np.nextafter(np.max(left, axis=1), np.inf)
  right_low = np.nextafter(np.min(right, axis=1), -np.inf)
  right_high = np.nextafter(np.max(right, axis=1), np.inf)
  if _profile is not None:
    _profile["overlap_index_elapsed_seconds_raw"] = float(
        _profile.get("overlap_index_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - index_started)
  total = 0.0
  # Chunking bounds the boolean broadphase without changing pair order or sum order.
  for start in range(0, len(left), 128):
    stop = min(len(left), start + 128)
    index_started = time.perf_counter()
    overlaps = np.all(
        np.maximum(left_low[start:stop, None, :], right_low[None, :, :])
        <= np.minimum(left_high[start:stop, None, :], right_high[None, :, :]),
        axis=2,
    )
    if _profile is not None:
      _profile["overlap_dense_aabb_pair_count"] = int(
          _profile.get("overlap_dense_aabb_pair_count", 0)
      ) + (stop - start) * len(right)
      _profile["overlap_index_elapsed_seconds_raw"] = float(
          _profile.get("overlap_index_elapsed_seconds_raw", 0.0)
      ) + (time.perf_counter() - index_started)
    exact_started = time.perf_counter()
    exact_count = 0
    for local_index, right_indices in enumerate(overlaps):
      left_index = start + local_index
      for right_index in np.flatnonzero(right_indices):
        exact_count += 1
        total += _v1._polygon_area(_v1._convex_clip(
            left[left_index], right[int(right_index)],
        ))
    if _profile is not None:
      _profile["overlap_exact_clip_pair_count"] = int(
          _profile.get("overlap_exact_clip_pair_count", 0)
      ) + exact_count
      _profile["overlap_exact_elapsed_seconds_raw"] = float(
          _profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
      ) + (time.perf_counter() - exact_started)
  return float(total)


def _plane_yaws(first: Mapping[str, Any], second: Mapping[str, Any],
                 geometry_a: ManifoldFaceGeometryV3,
                 geometry_b: ManifoldFaceGeometryV3) -> list[dict[str, Any]]:
  assert geometry_a.plane_trim is not None and geometry_b.plane_trim is not None
  tri_a = tuple(np.asarray(row, dtype=float) for row in geometry_a.plane_trim.triangles_xy)
  tri_b = tuple(np.asarray(row, dtype=float) for row in geometry_b.plane_trim.triangles_xy)
  alpha, beta = _v1._dominant_trim_angle(tri_a), _v1._dominant_trim_angle(tri_b)
  events = [
      {"kind": "dominant_trim", "a_angle": _clean_float(alpha),
       "b_angle": _clean_float(beta), "yaw": alpha + beta},
      {"kind": "dominant_trim_perpendicular", "a_angle": _clean_float(alpha),
       "b_angle": _clean_float(beta), "yaw": alpha + beta + math.pi / 2.0},
  ]
  for kind in ("boundary_edge_directions_local_2d", "adjacent_shared_edge_directions_local_2d"):
    if kind.startswith("adjacent"):
      left_raw = [edge["direction_local_2d"] for edge in first["adjacent_shared_edges"]
                  if edge["direction_local_2d"] is not None]
      right_raw = [edge["direction_local_2d"] for edge in second["adjacent_shared_edges"]
                   if edge["direction_local_2d"] is not None]
    else:
      left_raw, right_raw = first[kind], second[kind]
    left = canonicalize_landmark_points_v4(
        left_raw, maximum=LANDMARK_CAP_PER_ENDPOINT_PER_TYPE_V4,
    ) if left_raw else ()
    right = canonicalize_landmark_points_v4(
        right_raw, maximum=LANDMARK_CAP_PER_ENDPOINT_PER_TYPE_V4,
    ) if right_raw else ()
    for a in left:
      for b in right:
        aa, bb = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
        events.append({"kind": kind, "a_angle": _clean_float(aa),
                       "b_angle": _clean_float(bb), "yaw": aa + bb})
  grouped: dict[float, list[dict[str, Any]]] = {}
  for event in events:
    yaw = _clean_float(float(event.pop("yaw")) % math.pi)
    grouped.setdefault(yaw, []).append(event)
  return [
      {"yaw_radians": yaw, "events": sorted(rows, key=canonical_sha256)}
      for yaw, rows in sorted(grouped.items())
  ]


def _plane_candidates(
    *, geometry_a: ManifoldFaceGeometryV3, geometry_b: ManifoldFaceGeometryV3,
    landmarks_a: Mapping[str, Any], landmarks_b: Mapping[str, Any],
    child_world: np.ndarray, corners_a: np.ndarray, corners_b: np.ndarray,
    _profile: dict[str, int | float] | None = None,
    _telemetry: Callable[[str, float, Mapping[str, int | float]], None] | None = None,
    include_collision_metrics_v1: bool = False,
) -> list[dict[str, Any]]:
  assert geometry_a.plane_trim is not None and geometry_b.plane_trim is not None
  ra, rb = np.asarray(geometry_a.rotation_local_to_world), np.asarray(geometry_b.rotation_local_to_world)
  oa, ob = np.asarray(geometry_a.origin_world_mm), np.asarray(geometry_b.origin_world_mm)
  tri_a = tuple(np.asarray(row, dtype=float) for row in geometry_a.plane_trim.triangles_xy)
  tri_b_local = tuple(np.asarray(row, dtype=float) for row in geometry_b.plane_trim.triangles_xy)
  area_error = (geometry_a.plane_trim.overlap_area_error_upper_mm2
                + geometry_b.plane_trim.overlap_area_error_upper_mm2)
  pair_events = _alignment_pairs(landmarks_a, landmarks_b, coordinate=None)
  yaw_rows = _plane_yaws(landmarks_a, landmarks_b, geometry_a, geometry_b)
  candidate_started = time.perf_counter()
  profile = {} if _profile is None else _profile
  profile["alignment_event_count"] = len(pair_events)
  profile["yaw_event_count_raw"] = sum(len(row["events"]) for row in yaw_rows)
  profile["yaw_state_count"] = len(yaw_rows)
  profile["offset_event_count_raw"] = 0
  profile["offset_state_count"] = 0
  profile["overlap_call_count"] = 0
  profile["candidate_payload_hash_elapsed_seconds_raw"] = 0.0
  candidates = []
  next_progress_offset = 64
  for yaw_index, yaw_row in enumerate(yaw_rows):
    yaw = float(yaw_row["yaw_radians"])
    x = math.cos(yaw) * ra[:, 0] + math.sin(yaw) * ra[:, 1]
    y = -math.sin(yaw) * ra[:, 0] + math.cos(yaw) * ra[:, 1]
    target = np.stack((x, -y, -ra[:, 2]), axis=1)
    rotation = target @ rb.T
    base = np.eye(4)
    base[:3, :3] = rotation
    base[:3, 3] = oa - rotation @ ob
    transformed_b = []
    for triangle in tri_b_local:
      world = ob + triangle[:, 0, None] * rb[:, 0] + triangle[:, 1, None] * rb[:, 1]
      moved = world @ rotation.T + base[:3, 3]
      transformed_b.append(np.stack(((moved - oa) @ ra[:, 0], (moved - oa) @ ra[:, 1]), axis=1))
    points_a = np.concatenate(tri_a, axis=0)
    points_b = np.concatenate(transformed_b, axis=0)
    baseline = [
        {"kind": "triangle_centroid", "a": list(_point2(_v1._triangle_centroid(tri_a))),
         "b": list(_point2(_v1._triangle_centroid(transformed_b)))},
        {"kind": "lower_extrema", "a": list(_point2(np.min(points_a, axis=0))),
         "b": list(_point2(np.min(points_b, axis=0)))},
        {"kind": "upper_extrema", "a": list(_point2(np.max(points_a, axis=0))),
         "b": list(_point2(np.max(points_b, axis=0)))},
        {"kind": "bbox_center", "a": list(_point2(0.5 * (np.min(points_a, axis=0) + np.max(points_a, axis=0)))),
         "b": list(_point2(0.5 * (np.min(points_b, axis=0) + np.max(points_b, axis=0))))},
    ]
    offsets: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for event in (*baseline, *pair_events):
      a_point = np.asarray(event["a"], dtype=float)
      b_local = np.asarray(event["b"], dtype=float)
      if event in pair_events:
        b_world = ob + b_local[0] * rb[:, 0] + b_local[1] * rb[:, 1]
        b_moved = rotation @ b_world + base[:3, 3]
        b_point = np.asarray((np.dot(b_moved - oa, ra[:, 0]), np.dot(b_moved - oa, ra[:, 1])))
      else:
        b_point = b_local
      offset = _point2(a_point - b_point)
      clean_event = {key: value for key, value in event.items() if key != "difference"}
      offsets.setdefault(offset, []).append(clean_event)
    profile["offset_event_count_raw"] = int(profile["offset_event_count_raw"]) + (
        len(baseline) + len(pair_events)
    )
    profile["offset_state_count"] = int(profile["offset_state_count"]) + len(offsets)
    overlap_index = build_plane_yaw_state_overlap_index_v1(
        tri_a, transformed_b, _profile=profile,
    )
    offset_rows = sorted(offsets.items())
    offset_array = np.asarray([row[0] for row in offset_rows], dtype=float)
    translated_batch = (
        np.asarray(transformed_b, dtype=float)[None, :, :, :]
        + offset_array[:, None, None, :]
    )
    estimates = plane_overlap_areas_from_yaw_state_index_v2(
        overlap_index, offsets_xy=offset_array,
        translated_seconds=translated_batch, _profile=profile,
    )
    for offset_row_index, (offset, events) in enumerate(offset_rows):
      delta = base.copy()
      delta[:3, 3] += ra[:, 0] * offset[0] + ra[:, 1] * offset[1]
      profile["overlap_call_count"] = int(profile["overlap_call_count"]) + 1
      estimate = float(estimates[offset_row_index])
      lower = max(0.0, estimate - area_error)
      ratio = lower / min(geometry_a.plane_trim.exact_surface_area_mm2,
                          geometry_b.plane_trim.exact_surface_area_mm2)
      separation, aabb_overlap = _body_proxy_metrics(corners_a, corners_b, delta, geometry_a)
      collision_metrics = (
          body_frame_aabb_collision_metrics_p9_v1(
              corners_a, corners_b, list(_v1._matrix_tuple(delta)),
              reference_origin_world_mm=geometry_a.origin_world_mm,
              reference_rotation_local_to_world=geometry_a.rotation_local_to_world,
          ) if include_collision_metrics_v1 else None
      )
      moved_origin = rotation @ ob + delta[:3, 3]
      moved_normal = rotation @ rb[:, 2]
      unsigned = {
          "kind": "plane_contact_seed", "program_index": 9,
          "yaw_radians": _clean_float(yaw), "normal_sign": -1,
          "offset_local_2d": list(offset),
          "a_landmark": sorted(events, key=canonical_sha256)[0]["a"],
          "b_landmark": sorted(events, key=canonical_sha256)[0]["b"],
          "alignment_events": sorted(events, key=canonical_sha256),
          "yaw_events": yaw_row["events"],
          "base_world_delta_row_major": list(_v1._matrix_tuple(base)),
          "world_delta_row_major": list(_v1._matrix_tuple(delta)),
          "child_world_row_major": list(_v1._matrix_tuple(delta @ child_world)),
          "cheap_metrics": {
              "trim_overlap_estimate_mm2": _clean_float(estimate),
              "trim_overlap_error_upper_mm2": _clean_float(area_error),
              "trim_overlap_lower_bound": _clean_float(lower),
              "trim_overlap_ratio_lower_bound": _clean_float(ratio),
              "body_projected_aabb_separation_mm": separation,
              "body_projected_aabb_overlap_upper_mm2": aabb_overlap,
              "fixed_displacement_mm": _clean_float(np.linalg.norm(moved_origin - ob)),
              "normal_error_degrees": _clean_float(
                  _v1._angle_degrees(moved_normal, -ra[:, 2], unoriented=False)
              ),
              "normal_gap_mm": _clean_float(abs(np.dot(moved_origin - oa, ra[:, 2]))),
              **({
                  "body_frame_aabb_intersection_volume_mm3": collision_metrics[
                      "body_frame_aabb_intersection_volume_mm3"
                  ],
                  "body_frame_aabb_normalized_overlap": collision_metrics[
                      "body_frame_aabb_normalized_overlap"
                  ],
              } if collision_metrics is not None else {}),
          },
      }
      hash_started = time.perf_counter()
      candidates.append(_candidate_payload(unsigned))
      profile["candidate_payload_hash_elapsed_seconds_raw"] = float(
          profile["candidate_payload_hash_elapsed_seconds_raw"]
      ) + (time.perf_counter() - hash_started)
      if _telemetry is not None and int(profile["overlap_call_count"]) >= next_progress_offset:
        _telemetry("candidate_overlap_progress", time.perf_counter() - candidate_started, {
            "program_index": 9, "yaw_state_count": yaw_index + 1,
            "offset_state_count": int(profile["overlap_call_count"]),
            "candidate_count_progress": len(candidates),
            "overlap_call_count": int(profile["overlap_call_count"]),
            "overlap_index_build_count": int(
                profile.get("overlap_index_build_count", 0)
            ),
            "overlap_dense_aabb_pair_count": int(
                profile.get("overlap_dense_aabb_pair_count", 0)
            ),
            "overlap_exact_clip_pair_count": int(
                profile.get("overlap_exact_clip_pair_count", 0)
            ),
            "overlap_numba_batch_pair_count": int(
                profile.get("overlap_numba_batch_pair_count", 0)
            ),
            "overlap_numba_batch_call_count": int(
                profile.get("overlap_numba_batch_call_count", 0)
            ),
            "overlap_index_elapsed_seconds_raw": float(
                profile.get("overlap_index_elapsed_seconds_raw", 0.0)
            ),
            "overlap_exact_elapsed_seconds_raw": float(
                profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
            ),
            "candidate_payload_hash_elapsed_seconds_raw": float(
                profile.get("candidate_payload_hash_elapsed_seconds_raw", 0.0)
            ),
        })
        next_progress_offset += 64
  if _telemetry is not None and (
      not candidates or int(profile["overlap_call_count"]) % 64 != 0
  ):
    _telemetry("candidate_overlap_progress", time.perf_counter() - candidate_started, {
        "program_index": 9, "yaw_state_count": len(yaw_rows),
        "offset_state_count": int(profile["overlap_call_count"]),
        "candidate_count_progress": len(candidates),
        "overlap_call_count": int(profile["overlap_call_count"]),
        "overlap_index_build_count": int(profile.get("overlap_index_build_count", 0)),
        "overlap_dense_aabb_pair_count": int(
            profile.get("overlap_dense_aabb_pair_count", 0)
        ),
        "overlap_exact_clip_pair_count": int(profile.get("overlap_exact_clip_pair_count", 0)),
        "overlap_numba_batch_pair_count": int(
            profile.get("overlap_numba_batch_pair_count", 0)
        ),
        "overlap_numba_batch_call_count": int(
            profile.get("overlap_numba_batch_call_count", 0)
        ),
        "overlap_index_elapsed_seconds_raw": float(
            profile.get("overlap_index_elapsed_seconds_raw", 0.0)
        ),
        "overlap_exact_elapsed_seconds_raw": float(
            profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
        ),
        "candidate_payload_hash_elapsed_seconds_raw": float(
            profile.get("candidate_payload_hash_elapsed_seconds_raw", 0.0)
        ),
    })
  return candidates


def _cylinder_candidates(
    *, geometry_a: ManifoldFaceGeometryV3, geometry_b: ManifoldFaceGeometryV3,
    landmarks_a: Mapping[str, Any], landmarks_b: Mapping[str, Any],
    child_world: np.ndarray, corners_a: np.ndarray, corners_b: np.ndarray,
    _profile: dict[str, int | float] | None = None,
    _telemetry: Callable[[str, float, Mapping[str, int | float]], None] | None = None,
) -> list[dict[str, Any]]:
  ta, tb = geometry_a.cylinder_trim, geometry_b.cylinder_trim
  assert ta is not None and tb is not None
  ra, rb = np.asarray(geometry_a.rotation_local_to_world), np.asarray(geometry_b.rotation_local_to_world)
  oa, ob = np.asarray(geometry_a.origin_world_mm), np.asarray(geometry_b.origin_world_mm)
  area_error = ta.overlap_area_error_upper_mm2 + tb.overlap_area_error_upper_mm2
  amid, bmid = ta.angular_midpoint_radians, tb.angular_midpoint_radians
  alo, ahi = ta.axial_interval_mm
  blo, bhi = tb.axial_interval_mm
  pair_events = _alignment_pairs(landmarks_a, landmarks_b, coordinate=1)
  candidate_started = time.perf_counter()
  profile = {} if _profile is None else _profile
  profile["alignment_event_count"] = len(pair_events)
  profile["yaw_event_count_raw"] = 0
  profile["yaw_state_count"] = 0
  profile["offset_event_count_raw"] = 0
  profile["offset_state_count"] = 0
  profile["overlap_call_count"] = 0
  profile["candidate_payload_hash_elapsed_seconds_raw"] = 0.0
  candidates = []
  yaw_progress = 0
  for sign_index, sign in enumerate((1.0, -1.0)):
    yaw_events = [
        {"kind": "trim_angular_midpoint", "a": _clean_float(amid),
         "b": _clean_float(bmid), "yaw": amid - sign * bmid},
    ]
    for kind in _CYLINDER_YAW_TYPES_V4:
      for a in _capped_points(landmarks_a, kind):
        for b in _capped_points(landmarks_b, kind):
          yaw_events.append({"kind": kind, "a": list(a), "b": list(b),
                             "yaw": a[0] - sign * b[0]})
    yaw_groups: dict[float, list[dict[str, Any]]] = {}
    for event in yaw_events:
      raw = float(event.pop("yaw"))
      for yaw in (raw % (2.0 * math.pi), (raw + math.pi) % (2.0 * math.pi)):
        yaw_groups.setdefault(_clean_float(yaw), []).append(event)
    profile["yaw_event_count_raw"] = int(profile["yaw_event_count_raw"]) + 2 * len(yaw_events)
    profile["yaw_state_count"] = int(profile["yaw_state_count"]) + len(yaw_groups)
    transformed_interval = (blo, bhi) if sign > 0 else (-bhi, -blo)
    baseline_offsets = [
        {"kind": "interval_center", "a": [0.0, _clean_float((alo + ahi) * 0.5)],
         "b": [0.0, _clean_float(sum(transformed_interval) * 0.5)]},
        {"kind": "lower_endpoint", "a": [0.0, _clean_float(alo)],
         "b": [0.0, _clean_float(transformed_interval[0])]},
        {"kind": "upper_endpoint", "a": [0.0, _clean_float(ahi)],
         "b": [0.0, _clean_float(transformed_interval[1])]},
        {"kind": "opposed_endpoint", "a": [0.0, _clean_float(alo)],
         "b": [0.0, _clean_float(transformed_interval[1])]},
    ]
    offsets: dict[float, list[dict[str, Any]]] = {}
    for event in (*baseline_offsets, *pair_events):
      a_z, b_z = float(event["a"][1]), float(event["b"][1])
      offset = _clean_float(a_z - (sign * b_z if event in pair_events else b_z))
      offsets.setdefault(offset, []).append({key: value for key, value in event.items()
                                             if key != "difference"})
    profile["offset_event_count_raw"] = int(profile["offset_event_count_raw"]) + (
        len(baseline_offsets) + len(pair_events)
    )
    profile["offset_state_count"] = int(profile["offset_state_count"]) + len(offsets)
    for yaw, yaw_rows in sorted(yaw_groups.items()):
      yaw_progress += 1
      x = math.cos(yaw) * ra[:, 0] + math.sin(yaw) * ra[:, 1]
      y = -math.sin(yaw) * ra[:, 0] + math.cos(yaw) * ra[:, 1]
      target = np.stack((x, y if sign > 0 else -y, sign * ra[:, 2]), axis=1)
      rotation = target @ rb.T
      base = np.eye(4)
      base[:3, :3] = rotation
      base[:3, 3] = oa - rotation @ ob
      base_triangles_b = []
      for triangle in tb.triangles_uz:
        values = np.asarray(triangle, dtype=float)
        u = yaw + values[:, 0] if sign > 0 else yaw - values[:, 0]
        z = values[:, 1] if sign > 0 else -values[:, 1]
        base_triangles_b.append(np.stack((u, z), axis=1))
      periodic_index = __import__(
          "neurocad.constraint_manifold_solver_v2", fromlist=[
              "build_periodic_yaw_state_overlap_index_v1",
          ],
      ).build_periodic_yaw_state_overlap_index_v1(
          tuple(np.asarray(row, dtype=float) for row in ta.triangles_uz),
          base_triangles_b, _profile=profile,
      )
      for offset, events in sorted(offsets.items()):
        delta = base.copy()
        delta[:3, 3] += offset * ra[:, 2]
        triangles_b = []
        for triangle in tb.triangles_uz:
          values = np.asarray(triangle, dtype=float)
          u = yaw + values[:, 0] if sign > 0 else yaw - values[:, 0]
          z = offset + values[:, 1] if sign > 0 else offset - values[:, 1]
          triangles_b.append(np.stack((u, z), axis=1))
        profile["overlap_call_count"] = int(profile["overlap_call_count"]) + 1
        parameter_overlap = __import__(
            "neurocad.constraint_manifold_solver_v2", fromlist=[
                "periodic_overlap_from_yaw_state_index_v1",
            ],
        ).periodic_overlap_from_yaw_state_index_v1(
            periodic_index, offset_uz=(0.0, offset),
            translated_second=triangles_b, _profile=profile,
        )
        estimate = min(ta.radius_mm, tb.radius_mm) * parameter_overlap
        lower = max(0.0, estimate - area_error)
        ratio = lower / min(ta.exact_surface_area_mm2, tb.exact_surface_area_mm2)
        separation, aabb_overlap = _body_proxy_metrics(corners_a, corners_b, delta, geometry_a)
        body_aabb_overlap_3d = _body_frame_aabb_overlap_upper_mm3_v1(
            corners_a, corners_b, delta, geometry_a,
        )
        moved_origin = rotation @ ob + delta[:3, 3]
        moved_axis = rotation @ rb[:, 2]
        radial = moved_origin - oa - np.dot(moved_origin - oa, ra[:, 2]) * ra[:, 2]
        moved_interval = (
            transformed_interval[0] + offset, transformed_interval[1] + offset,
        )
        axial_overlap = max(
            0.0, min(ahi, moved_interval[1]) - max(alo, moved_interval[0]),
        )
        angular_overlap = parameter_overlap / max(axial_overlap, 1e-12)
        representative = sorted(events, key=canonical_sha256)[0]
        unsigned = {
            "kind": "cylinder_contact_seed", "program_index": 11,
            "yaw_radians": _clean_float(yaw), "normal_sign": int(sign),
            "offset_local_2d": [0.0, _clean_float(offset)],
            "a_landmark": representative["a"], "b_landmark": representative["b"],
            "alignment_events": sorted(events, key=canonical_sha256),
            "yaw_events": sorted(yaw_rows, key=canonical_sha256),
            "base_world_delta_row_major": list(_v1._matrix_tuple(base)),
            "world_delta_row_major": list(_v1._matrix_tuple(delta)),
            "child_world_row_major": list(_v1._matrix_tuple(delta @ child_world)),
            "cheap_metrics": {
                "trim_overlap_estimate_mm2": _clean_float(estimate),
                "trim_overlap_error_upper_mm2": _clean_float(area_error),
                "trim_overlap_lower_bound": _clean_float(lower),
                "trim_overlap_ratio_lower_bound": _clean_float(ratio),
                "body_projected_aabb_separation_mm": separation,
                "body_projected_aabb_overlap_upper_mm2": aabb_overlap,
                "body_frame_aabb_overlap_upper_mm3": body_aabb_overlap_3d,
                "fixed_displacement_mm": _clean_float(np.linalg.norm(moved_origin - ob)),
                "axis_error_degrees": _clean_float(
                    _v1._angle_degrees(moved_axis, ra[:, 2], unoriented=True)
                ),
                "radial_axis_distance_mm": _clean_float(np.linalg.norm(radial)),
                "axial_overlap_mm": _clean_float(axial_overlap),
                "angular_overlap_radians": _clean_float(angular_overlap),
            },
        }
        hash_started = time.perf_counter()
        candidates.append(_candidate_payload(unsigned))
        profile["candidate_payload_hash_elapsed_seconds_raw"] = float(
            profile["candidate_payload_hash_elapsed_seconds_raw"]
        ) + (time.perf_counter() - hash_started)
      if _telemetry is not None and (
          yaw_progress % 32 == 0 or (
              sign_index == 1 and yaw_progress == int(profile["yaw_state_count"])
          )
      ):
        _telemetry("candidate_overlap_progress", time.perf_counter() - candidate_started, {
            "program_index": 11, "sign_index": sign_index,
            "yaw_state_count": yaw_progress,
            "offset_state_count": int(profile["offset_state_count"]),
            "candidate_count_progress": len(candidates),
            "overlap_call_count": int(profile["overlap_call_count"]),
            "overlap_index_build_count": int(
                profile.get("overlap_index_build_count", 0)
            ),
            "overlap_periodic_copy_count": int(
                profile.get("overlap_periodic_copy_count", 0)
            ),
            "overlap_index_candidate_pair_count": int(
                profile.get("overlap_index_candidate_pair_count", 0)
            ),
            "overlap_exact_clip_pair_count": int(
                profile.get("overlap_exact_clip_pair_count", 0)
            ),
            "overlap_index_elapsed_seconds_raw": float(
                profile.get("overlap_index_elapsed_seconds_raw", 0.0)
            ),
            "overlap_exact_elapsed_seconds_raw": float(
                profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
            ),
            "candidate_payload_hash_elapsed_seconds_raw": float(
                profile.get("candidate_payload_hash_elapsed_seconds_raw", 0.0)
            ),
        })
  return candidates


def _cylinder_candidates_factorized_v2(
    *, geometry_a: ManifoldFaceGeometryV3, geometry_b: ManifoldFaceGeometryV3,
    landmarks_a: Mapping[str, Any], landmarks_b: Mapping[str, Any],
    child_world: np.ndarray, corners_a: np.ndarray, corners_b: np.ndarray,
    _profile: dict[str, int | float] | None = None,
    _telemetry: Callable[[str, float, Mapping[str, int | float]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
  """Build all P11 factors, then evaluate geometry for only selected top-32."""

  ta, tb = geometry_a.cylinder_trim, geometry_b.cylinder_trim
  assert ta is not None and tb is not None
  ra = np.asarray(geometry_a.rotation_local_to_world)
  rb = np.asarray(geometry_b.rotation_local_to_world)
  oa = np.asarray(geometry_a.origin_world_mm)
  ob = np.asarray(geometry_b.origin_world_mm)
  amid, bmid = ta.angular_midpoint_radians, tb.angular_midpoint_radians
  alo, ahi = ta.axial_interval_mm
  blo, bhi = tb.axial_interval_mm
  pair_events = _alignment_pairs(landmarks_a, landmarks_b, coordinate=1)
  profile = {} if _profile is None else _profile
  profile.update({
      "alignment_event_count": len(pair_events), "yaw_event_count_raw": 0,
      "yaw_state_count": 0, "offset_event_count_raw": 0,
      "offset_state_count": 0, "overlap_call_count": 0,
      "candidate_payload_hash_elapsed_seconds_raw": 0.0,
  })
  yaw_payloads: list[dict[str, Any]] = []
  offset_payloads: list[dict[str, Any]] = []
  for sign_value in (1.0, -1.0):
    sign = int(sign_value)
    raw_yaw_events = [{
        "kind": "trim_angular_midpoint", "a": _clean_float(amid),
        "b": _clean_float(bmid), "yaw": amid - sign_value * bmid,
    }]
    for kind in _CYLINDER_YAW_TYPES_V4:
      for a in _capped_points(landmarks_a, kind):
        for b in _capped_points(landmarks_b, kind):
          raw_yaw_events.append({
              "kind": kind, "a": list(a), "b": list(b),
              "yaw": a[0] - sign_value * b[0],
          })
    yaw_groups = _group_canonical_yaw_events_v2(raw_yaw_events)
    profile["yaw_event_count_raw"] = int(profile["yaw_event_count_raw"]) + (
        2 * len(raw_yaw_events)
    )
    profile["yaw_state_count"] = int(profile["yaw_state_count"]) + len(yaw_groups)
    transformed_interval = (blo, bhi) if sign > 0 else (-bhi, -blo)
    baseline_offsets = [
        {"kind": "interval_center", "a": [0.0, _clean_float((alo + ahi) * 0.5)],
         "b": [0.0, _clean_float(sum(transformed_interval) * 0.5)]},
        {"kind": "lower_endpoint", "a": [0.0, _clean_float(alo)],
         "b": [0.0, _clean_float(transformed_interval[0])]},
        {"kind": "upper_endpoint", "a": [0.0, _clean_float(ahi)],
         "b": [0.0, _clean_float(transformed_interval[1])]},
        {"kind": "opposed_endpoint", "a": [0.0, _clean_float(alo)],
         "b": [0.0, _clean_float(transformed_interval[1])]},
    ]
    offsets: dict[float, list[dict[str, Any]]] = {}
    for event in (*baseline_offsets, *pair_events):
      a_z, b_z = float(event["a"][1]), float(event["b"][1])
      offset = _clean_float(
          a_z - (sign_value * b_z if event in pair_events else b_z)
      )
      offsets.setdefault(offset, []).append({
          key: value for key, value in event.items() if key != "difference"
      })
    profile["offset_event_count_raw"] = int(profile["offset_event_count_raw"]) + (
        len(baseline_offsets) + len(pair_events)
    )
    profile["offset_state_count"] = int(profile["offset_state_count"]) + len(offsets)
    for yaw, yaw_rows in sorted(yaw_groups.items()):
      x = math.cos(yaw) * ra[:, 0] + math.sin(yaw) * ra[:, 1]
      y = -math.sin(yaw) * ra[:, 0] + math.cos(yaw) * ra[:, 1]
      target = np.stack((x, y if sign > 0 else -y, sign_value * ra[:, 2]), axis=1)
      rotation = target @ rb.T
      base = np.eye(4)
      base[:3, :3] = rotation
      base[:3, 3] = oa - rotation @ ob
      yaw_payloads.append({
          "normal_sign": sign, "yaw_radians": _clean_float(yaw),
          "yaw_events": sorted(yaw_rows, key=canonical_sha256),
          "base_world_delta_row_major": list(_v1._matrix_tuple(base)),
          "child_world_input_row_major": list(_v1._matrix_tuple(child_world)),
      })
    for offset, events in sorted(offsets.items()):
      representative = sorted(events, key=canonical_sha256)[0]
      offset_payloads.append({
          "normal_sign": sign, "axial_offset_mm": _clean_float(offset),
          "translation_delta_world_mm": [
              float(value) for value in offset * ra[:, 2]
          ],
          "a_landmark": representative["a"], "b_landmark": representative["b"],
          "alignment_events": sorted(events, key=canonical_sha256),
          "offset_families": sorted({str(event["kind"]) for event in events}),
          "fixed_displacement_mm": _clean_float(np.linalg.norm(
              oa + offset * ra[:, 2] - ob
          )),
      })
  factor_domain = commit_p11_factorized_domain_v2(
      yaw_factor_payloads=yaw_payloads,
      axial_offset_factor_payloads=offset_payloads,
  )
  selection = select_p11_factorized_topk_v2(factor_domain)
  profile.update({
      "factor_sign_count": len(factor_domain["sign_factors"]),
      "factor_yaw_count": len(factor_domain["yaw_factors"]),
      "factor_axial_offset_count": len(factor_domain["axial_offset_factors"]),
      "factorized_candidate_count": int(factor_domain["candidate_domain_count"]),
      "materialized_candidate_count": int(selection["materialized_candidate_count"]),
  })
  yaws = {
      row["factor_payload_sha256"]: row for row in factor_domain["yaw_factors"]
  }
  offsets = {
      row["factor_payload_sha256"]: row
      for row in factor_domain["axial_offset_factors"]
  }
  area_error = ta.overlap_area_error_upper_mm2 + tb.overlap_area_error_upper_mm2
  periodic_indices: dict[str, Any] = {}
  candidates: list[dict[str, Any]] = []
  started = time.perf_counter()
  for membership in selection["selected_factor_memberships"]:
    yaw_factor = yaws[membership["yaw_factor_sha256"]]
    offset_factor = offsets[membership["axial_offset_factor_sha256"]]
    yaw = float(yaw_factor["yaw_radians"])
    sign = int(yaw_factor["normal_sign"])
    offset = float(offset_factor["axial_offset_mm"])
    base = np.asarray(yaw_factor["base_world_delta_row_major"], dtype=float).reshape(4, 4)
    delta = base.copy()
    delta[:3, 3] += np.asarray(offset_factor["translation_delta_world_mm"], dtype=float)
    rotation = base[:3, :3]
    periodic_key = str(yaw_factor["factor_payload_sha256"])
    if periodic_key not in periodic_indices:
      base_triangles_b = []
      for triangle in tb.triangles_uz:
        values = np.asarray(triangle, dtype=float)
        u = yaw + values[:, 0] if sign > 0 else yaw - values[:, 0]
        z = values[:, 1] if sign > 0 else -values[:, 1]
        base_triangles_b.append(np.stack((u, z), axis=1))
      periodic_indices[periodic_key] = __import__(
          "neurocad.constraint_manifold_solver_v2", fromlist=[
              "build_periodic_yaw_state_overlap_index_v1",
          ],
      ).build_periodic_yaw_state_overlap_index_v1(
          tuple(np.asarray(row, dtype=float) for row in ta.triangles_uz),
          base_triangles_b, _profile=profile,
      )
    triangles_b = []
    for triangle in tb.triangles_uz:
      values = np.asarray(triangle, dtype=float)
      u = yaw + values[:, 0] if sign > 0 else yaw - values[:, 0]
      z = offset + values[:, 1] if sign > 0 else offset - values[:, 1]
      triangles_b.append(np.stack((u, z), axis=1))
    profile["overlap_call_count"] = int(profile["overlap_call_count"]) + 1
    parameter_overlap = __import__(
        "neurocad.constraint_manifold_solver_v2", fromlist=[
            "periodic_overlap_from_yaw_state_index_v1",
        ],
    ).periodic_overlap_from_yaw_state_index_v1(
        periodic_indices[periodic_key], offset_uz=(0.0, offset),
        translated_second=triangles_b, _profile=profile,
    )
    estimate = min(ta.radius_mm, tb.radius_mm) * parameter_overlap
    lower = max(0.0, estimate - area_error)
    ratio = lower / min(ta.exact_surface_area_mm2, tb.exact_surface_area_mm2)
    separation, aabb_overlap = _body_proxy_metrics(
        corners_a, corners_b, delta, geometry_a,
    )
    body_aabb_overlap_3d = _body_frame_aabb_overlap_upper_mm3_v1(
        corners_a, corners_b, delta, geometry_a,
    )
    moved_origin = rotation @ ob + delta[:3, 3]
    moved_axis = rotation @ rb[:, 2]
    radial = moved_origin - oa - np.dot(moved_origin - oa, ra[:, 2]) * ra[:, 2]
    transformed_interval = (blo, bhi) if sign > 0 else (-bhi, -blo)
    moved_interval = (
        transformed_interval[0] + offset, transformed_interval[1] + offset,
    )
    axial_overlap = max(
        0.0, min(ahi, moved_interval[1]) - max(alo, moved_interval[0]),
    )
    angular_overlap = parameter_overlap / max(axial_overlap, 1e-12)
    unsigned = {
        "kind": "cylinder_contact_seed", "program_index": 11,
        "yaw_radians": _clean_float(yaw), "normal_sign": sign,
        "offset_local_2d": [0.0, _clean_float(offset)],
        "a_landmark": offset_factor["a_landmark"],
        "b_landmark": offset_factor["b_landmark"],
        "alignment_events": offset_factor["alignment_events"],
        "yaw_events": yaw_factor["yaw_events"],
        "base_world_delta_row_major": list(_v1._matrix_tuple(base)),
        "world_delta_row_major": list(_v1._matrix_tuple(delta)),
        "child_world_row_major": list(_v1._matrix_tuple(
            multiply_se3_row_major_v2(delta, child_world)
        )),
        "cheap_metrics": {
            "trim_overlap_estimate_mm2": _clean_float(estimate),
            "trim_overlap_error_upper_mm2": _clean_float(area_error),
            "trim_overlap_lower_bound": _clean_float(lower),
            "trim_overlap_ratio_lower_bound": _clean_float(ratio),
            "body_projected_aabb_separation_mm": separation,
            "body_projected_aabb_overlap_upper_mm2": aabb_overlap,
            "body_frame_aabb_overlap_upper_mm3": body_aabb_overlap_3d,
            "fixed_displacement_mm": offset_factor["fixed_displacement_mm"],
            "axis_error_degrees": _clean_float(
                _v1._angle_degrees(moved_axis, ra[:, 2], unoriented=True)
            ),
            "radial_axis_distance_mm": _clean_float(np.linalg.norm(radial)),
            "axial_overlap_mm": _clean_float(axial_overlap),
            "angular_overlap_radians": _clean_float(angular_overlap),
        },
    }
    hash_started = time.perf_counter()
    candidate = _candidate_payload(unsigned)
    profile["candidate_payload_hash_elapsed_seconds_raw"] = float(
        profile["candidate_payload_hash_elapsed_seconds_raw"]
    ) + (time.perf_counter() - hash_started)
    if candidate["candidate_key"] != membership["candidate_key"]:
      raise ValueError("p11_factorized_selected_identity")
    candidates.append(candidate)
  if _telemetry is not None:
    _telemetry("candidate_overlap_progress", time.perf_counter() - started, {
        "program_index": 11, "candidate_count_progress": len(candidates),
        "overlap_call_count": int(profile["overlap_call_count"]),
        "yaw_state_count": int(profile["yaw_state_count"]),
        "offset_state_count": int(profile["offset_state_count"]),
        "factor_sign_count": int(profile["factor_sign_count"]),
        "factor_yaw_count": int(profile["factor_yaw_count"]),
        "factor_axial_offset_count": int(profile["factor_axial_offset_count"]),
        "factorized_candidate_count": int(profile["factorized_candidate_count"]),
        "materialized_candidate_count": int(profile["materialized_candidate_count"]),
    })
  return candidates, factor_domain, selection


def _preprocess_child_v4(
    request: Mapping[str, Any], *,
    telemetry: Callable[[str, float, Mapping[str, int | float]], None] | None = None,
    diagnostic_full_candidate_observer: (
        Callable[[Sequence[Mapping[str, Any]]], None] | None
    ) = None,
) -> dict[str, Any]:
  """Run only inside the one-shot worker process."""

  try:
    validation_started = time.perf_counter()
    schema_version = request.get("schema_version")
    v4_request_keys = {
        "schema_version", "operation", "call_nonce", "query_id_sha256",
        "program_index", "endpoints", "child_world_row_major", "policy",
        "producer_source_sha256s",
    }
    v5_request_keys = v4_request_keys | {"ranking_schema"}
    if schema_version not in {
        PREPROCESS_REQUEST_SCHEMA_VERSION, PREPROCESS_REQUEST_SCHEMA_VERSION_V5,
    }:
      raise ValueError("request_schema")
    if set(request) != (
        v4_request_keys
        if schema_version == PREPROCESS_REQUEST_SCHEMA_VERSION else v5_request_keys
    ):
      raise ValueError("request_exact_schema")
    if request.get("operation") != PREPROCESS_OPERATION_V4:
      raise ValueError("request_operation")
    program_index = int(request["program_index"])
    if program_index not in {9, 11}:
      raise ValueError("request_program")
    ranking_schema = (
        None if schema_version == PREPROCESS_REQUEST_SCHEMA_VERSION
        else request["ranking_schema"]
    )
    if schema_version == PREPROCESS_REQUEST_SCHEMA_VERSION_V5 and (
        program_index != 9
        or ranking_schema != P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
    ):
      raise ValueError("request_ranking_schema")
    if request.get("policy") != preprocess_policy_for_program_v1(
        program_index, ranking_schema=ranking_schema,
    ):
      raise ValueError("request_policy")
    if request.get("producer_source_sha256s") != preprocess_source_sha256s_v4():
      raise ValueError("request_source_binding")
    nonce = str(request["call_nonce"])
    if _SHA256_RE.fullmatch(nonce) is None:
      raise ValueError("request_nonce")
    endpoint_rows = request["endpoints"]
    if len(endpoint_rows) != 2 or tuple(row["part_slot"] for row in endpoint_rows) != ("a", "b"):
      raise ValueError("request_endpoints")
    endpoint_keys = {
        "part_slot", "step_path", "step_sha256", "graph_face_index",
        "raw_occ_face_index", "face_signature_sha256", "current_world_row_major",
    }
    if any(not isinstance(row, Mapping) or set(row) != endpoint_keys for row in endpoint_rows):
      raise ValueError("request_endpoint_exact_schema")
    if telemetry is not None:
      telemetry("request_validation", time.perf_counter() - validation_started, {
          "program_index": program_index,
      })
    shapes, geometries, landmarks, topologies, worlds, corners = [], [], [], [], [], []
    sanitized_endpoints = []
    expected_surface = "plane" if program_index == 9 else "cylinder"
    for endpoint_index, endpoint in enumerate(endpoint_rows):
      step_path = Path(endpoint["step_path"]).resolve(strict=True)
      stage_started = time.perf_counter()
      shape = _load_captured_step_shape_v4(
          step_path, expected_sha256=endpoint["step_sha256"],
      )
      if telemetry is not None:
        telemetry("endpoint_step_load", time.perf_counter() - stage_started, {
            "endpoint_index": endpoint_index, "step_bytes": step_path.stat().st_size,
        })
      stage_started = time.perf_counter()
      faces = tuple(shape.Faces())
      raw_index = int(endpoint["raw_occ_face_index"])
      if not 0 <= raw_index < len(faces):
        raise ValueError("selected_face_index")
      if source_face_signature_sha256(faces[raw_index]) != endpoint["face_signature_sha256"]:
        raise ValueError("selected_face_signature")
      world = _transform(endpoint["current_world_row_major"])
      if telemetry is not None:
        telemetry("endpoint_face_replay", time.perf_counter() - stage_started, {
            "endpoint_index": endpoint_index, "face_count": len(faces),
        })
      stage_started = time.perf_counter()
      geometry = _canonical_geometry(extract_occ_manifold_face_v3(
          shape, part_slot=endpoint["part_slot"],
          graph_face_index=int(endpoint["graph_face_index"]),
          raw_occ_face_index=raw_index,
          face_signature_sha256=endpoint["face_signature_sha256"],
          current_world=world,
      ))
      if geometry.surface_type != expected_surface:
        raise ValueError("selected_surface_program")
      trim_row = geometry.plane_trim if geometry.plane_trim is not None else geometry.cylinder_trim
      assert trim_row is not None
      if telemetry is not None:
        telemetry("endpoint_geometry_mesh", time.perf_counter() - stage_started, {
            "endpoint_index": endpoint_index,
            "mesh_node_count": trim_row.mesh_proof.node_count,
            "mesh_triangle_count": trim_row.mesh_proof.triangle_count,
            "retained_triangle_count": len(
                trim_row.triangles_xy if geometry.plane_trim is not None
                else trim_row.triangles_uz  # type: ignore[union-attr]
            ),
        })
      stage_started = time.perf_counter()
      endpoint_landmarks, topology = _endpoint_landmarks(
          shape=shape, selected_index=raw_index, world=world, geometry=geometry,
      )
      if telemetry is not None:
        telemetry("endpoint_landmarks_topology", time.perf_counter() - stage_started, {
            "endpoint_index": endpoint_index,
            "selected_edge_count": int(topology["selected_edge_count"]),
            "selected_wire_count": int(topology["selected_wire_count"]),
            "adjacent_face_count": len(topology["adjacent_faces"]),
            "shared_edge_count": sum(
                int(row["shared_edge_count"]) for row in topology["adjacent_faces"]
            ),
        })
      stage_started = time.perf_counter()
      body_corners = _body_world_corners(shape, world)
      if telemetry is not None:
        telemetry("endpoint_bbox", time.perf_counter() - stage_started, {
            "endpoint_index": endpoint_index,
        })
      shapes.append(shape); geometries.append(geometry); landmarks.append(endpoint_landmarks)
      topologies.append(topology); worlds.append(world); corners.append(body_corners)
      trim = geometry.plane_trim.payload() if geometry.plane_trim is not None else (
          geometry.cylinder_trim.payload()  # type: ignore[union-attr]
      )
      sanitized_endpoints.append({
          "part_slot": endpoint["part_slot"],
          "graph_face_index": int(endpoint["graph_face_index"]),
          "selected_raw_occ_face_index": raw_index,
          "face_signature_sha256": endpoint["face_signature_sha256"],
          "surface_type": geometry.surface_type,
          "origin_world_mm": list(geometry.origin_world_mm),
          "rotation_local_to_world": [list(row) for row in geometry.rotation_local_to_world],
          "trim": trim, "topology": topology, "landmarks": endpoint_landmarks,
          "body_projected_aabb_local_2d": _aabb_payload(_project_aabb(body_corners, geometry)),
      })
    child_world = np.asarray(request["child_world_row_major"], dtype=float).reshape(4, 4)
    candidate_profile: dict[str, int | float] = {}
    candidate_started = time.perf_counter()
    if telemetry is not None:
      telemetry("candidate_domain_prepare", 0.0, {"program_index": program_index})
    factor_domain: dict[str, Any] | None = None
    factor_selection: dict[str, Any] | None = None
    if program_index == 9:
      candidate_rows = _plane_candidates(
            geometry_a=geometries[0], geometry_b=geometries[1],
            landmarks_a=landmarks[0], landmarks_b=landmarks[1],
            child_world=child_world, corners_a=corners[0], corners_b=corners[1],
            _profile=candidate_profile, _telemetry=telemetry,
            include_collision_metrics_v1=(
                ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
            ),
      )
    else:
      candidate_rows, factor_domain, factor_selection = (
          _cylinder_candidates_factorized_v2(
            geometry_a=geometries[0], geometry_b=geometries[1],
            landmarks_a=landmarks[0], landmarks_b=landmarks[1],
            child_world=child_world, corners_a=corners[0], corners_b=corners[1],
            _profile=candidate_profile, _telemetry=telemetry,
          )
      )
    candidate_domain_count = (
        len(candidate_rows) if factor_domain is None
        else int(factor_domain["candidate_domain_count"])
    )
    if telemetry is not None:
      telemetry("candidate_domain_ready", time.perf_counter() - candidate_started, {
          "program_index": program_index,
          "yaw_state_count": int(candidate_profile.get("yaw_state_count", 0)),
          "offset_state_count": int(candidate_profile.get("offset_state_count", 0)),
      })
      telemetry("candidate_domain_complete", time.perf_counter() - candidate_started, {
          "program_index": program_index,
          "candidate_count_raw": candidate_domain_count,
          **{key: value for key, value in candidate_profile.items()
             if key in _COUNTER_ALLOWLIST_V7},
      })
    if diagnostic_full_candidate_observer is not None:
      # Deliberately unavailable through the worker request schema.  This
      # read-only hook supports train-only rank diagnostics without expanding
      # production IPC or changing the committed compact domain.
      diagnostic_full_candidate_observer(tuple(candidate_rows))
    dedupe_started = time.perf_counter()
    if program_index == 9:
      compact_domain = (
          select_p9_collision_aware_topk_v1(candidate_rows)
          if ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
          else compact_candidate_domain_v1(candidate_rows, program_index=program_index)
      )
    else:
      assert factor_domain is not None and factor_selection is not None
      compact_domain = {
          "ranking_schema": P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2,
          "candidate_domain_count": factor_domain["candidate_domain_count"],
          "candidate_domain_sha256": factor_domain["candidate_domain_sha256"],
          "compact_top_k": COMPACT_TOP_K_V1,
          "compact_topk_payload_sha256": canonical_sha256(candidate_rows),
          "diversity_coverage": factor_selection["diversity_coverage"],
          "cheap_geometry_candidates": candidate_rows,
          "factorized_domain": factor_domain,
          "factorized_selection": factor_selection,
      }
    candidates = compact_domain["cheap_geometry_candidates"]
    dedupe_elapsed = time.perf_counter() - dedupe_started
    if telemetry is not None:
      telemetry("candidate_hash_dedupe", dedupe_elapsed, {
          "candidate_count_raw": len(candidate_rows),
          "candidate_count_unique": int(compact_domain["candidate_domain_count"]),
          "candidate_dedupe_sort_elapsed_seconds_raw": dedupe_elapsed,
          "candidate_payload_hash_elapsed_seconds_raw": float(
              candidate_profile.get("candidate_payload_hash_elapsed_seconds_raw", 0.0)
          ),
      })
    unsigned = {
        "schema_version": (
            SANITIZED_GEOMETRY_P9_COLLISION_TOPK_SCHEMA_VERSION_V1
            if ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
            else SANITIZED_GEOMETRY_SCHEMA_VERSION if program_index == 9
            else SANITIZED_GEOMETRY_P11_FACTORIZED_SCHEMA_VERSION_V2
        ),
        "query_id_sha256": request["query_id_sha256"],
        "program_index": program_index,
        "input_step_sha256s": [row["step_sha256"] for row in endpoint_rows],
        "producer_source_sha256s": preprocess_source_sha256s_v4(),
        "endpoints": sanitized_endpoints,
        "cheap_geometry_candidates": candidates,
        "candidate_domain_count": compact_domain["candidate_domain_count"],
        "candidate_domain_sha256": compact_domain["candidate_domain_sha256"],
        "compact_top_k": compact_domain["compact_top_k"],
        "compact_topk_payload_sha256": compact_domain["compact_topk_payload_sha256"],
        "candidate_policy": preprocess_policy_for_program_v1(
            program_index, ranking_schema=ranking_schema,
        ),
        "complete_cheap_domain": True,
        "oracle_inputs_absent": True,
        "final_test_touched": False,
    }
    if program_index == 11:
      unsigned["ranking_schema"] = compact_domain["ranking_schema"]
      unsigned["diversity_coverage"] = compact_domain["diversity_coverage"]
      unsigned["factorized_domain"] = compact_domain["factorized_domain"]
      unsigned["factorized_selection"] = compact_domain["factorized_selection"]
    elif ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1:
      unsigned["ranking_schema"] = ranking_schema
      unsigned["collision_selection"] = compact_domain["collision_selection"]
    hash_started = time.perf_counter()
    sanitized = {**unsigned, "sanitized_geometry_sha256": canonical_sha256(unsigned)}
    hash_elapsed = time.perf_counter() - hash_started
    validation_started = time.perf_counter()
    validate_sanitized_geometry_v4(sanitized)
    if telemetry is not None:
      telemetry("sanitized_validation", time.perf_counter() - validation_started, {
          "candidate_count_unique": int(compact_domain["candidate_domain_count"]),
          "sanitized_hash_elapsed_seconds_raw": hash_elapsed,
      })
    return {
        "status": "ok", "operation": PREPROCESS_OPERATION_V4,
        "call_nonce": nonce, "sanitized_geometry": sanitized,
        "child_pid": os.getpid(),
    }
  except BaseException as error:
    message = str(error)
    safe_code = message if re.fullmatch(r"[a-z0-9_]+", message) else "unspecified"
    return {
        "status": "kernel_error", "operation": PREPROCESS_OPERATION_V4,
        "call_nonce": str(request.get("call_nonce", "absent")),
        "error": f"preprocess_failed:{type(error).__name__}:{safe_code}",
        "child_pid": os.getpid(),
    }


class ConstraintManifoldPreprocessSupervisorV4:
  __slots__ = ("worker_script", "_seen_nonces")

  def __init__(self, *, worker_script: str | Path | None = None) -> None:
    default = Path(__file__).resolve().parent / "tools" / "constraint_manifold_preprocess_worker_v4.py"
    self.worker_script = str((default if worker_script is None else Path(worker_script)).resolve())
    self._seen_nonces: set[str] = set()

  def run(
      self, request: ConstraintManifoldPreprocessRequestV4 | ConstraintManifoldPreprocessRequestV5, *,
      wall_time_seconds: float, call_nonce: str | None = None,
  ) -> ConstraintManifoldPreprocessRunV4:
    if type(request) not in {
        ConstraintManifoldPreprocessRequestV4,
        ConstraintManifoldPreprocessRequestV5,
    }:
      raise TypeError("preprocess supervisor requires a typed V4/V5 request")
    if not math.isfinite(wall_time_seconds) or wall_time_seconds <= 0.0:
      raise ValueError("preprocess total wall time differs")
    nonce = secrets.token_hex(32) if call_nonce is None else call_nonce
    if _SHA256_RE.fullmatch(nonce) is None:
      raise ValueError("preprocess nonce differs")
    if nonce in self._seen_nonces:
      raise ValueError("preprocess nonce was reused")
    self._seen_nonces.add(nonce)
    start = time.monotonic()
    deadline = start + wall_time_seconds
    request_unsigned = request.worker_payload(call_nonce=nonce)
    request_payload = {
        **request_unsigned, "request_payload_sha256": canonical_sha256(request_unsigned),
    }
    request_sha = request_payload["request_payload_sha256"]
    source_hashes = preprocess_source_sha256s_v4()
    code_bundle_sha = canonical_sha256({
        "producer_source_sha256s": source_hashes,
        "launched_worker_script_sha256": file_sha256(self.worker_script),
    })

    process = None
    stage_evidence: dict[str, Any] = {
        "stage_telemetry": [], "stage_telemetry_event_count": 0,
        "stage_telemetry_payload_sha256": canonical_sha256([]),
    }

    def capture_stage_evidence(stderr: bytes) -> None:
      nonlocal stage_evidence
      events = parse_preprocess_stage_telemetry_v7(
          stderr, request_payload_sha256=request_sha, call_nonce=nonce,
      )
      stage_evidence = {
          "stage_telemetry": events, "stage_telemetry_event_count": len(events),
          "stage_telemetry_payload_sha256": canonical_sha256(events),
      }

    def terminal(
        status: str, *, issuer: str, child_started: bool,
        result: Mapping[str, Any], authority: SanitizedGeometryAuthorityV4 | None = None,
    ) -> ConstraintManifoldPreprocessRunV4:
      normalized_result = dict(result)
      normalized_result.update(stage_evidence)
      normalized_result.setdefault(
          "child_reaped",
          not child_started or (process is not None and process.poll() is not None),
      )
      normalized_result.setdefault("cleanup_window_seconds", 0.0)
      normalized_result.setdefault("cleanup_elapsed_seconds", 0.0)
      receipt = _issue_receipt_v4(
          call_nonce=nonce, request_payload_sha256=request_sha,
          worker_code_bundle_sha256=code_bundle_sha,
          terminal_status=status, issuer=issuer, child_started=child_started,
          elapsed_seconds=time.monotonic() - start, result=normalized_result,
      )
      return ConstraintManifoldPreprocessRunV4(receipt=receipt, geometry_authority=authority)

    def expired_after(stage: str) -> ConstraintManifoldPreprocessRunV4 | None:
      if time.monotonic() < deadline:
        return None
      return terminal(
          "timeout", issuer="parent_observer", child_started=process is not None,
          result={"error": f"deadline_expired_after_{stage}",
                  "child_pid": None if process is None else process.pid},
      )

    if time.monotonic() >= deadline:
      return terminal(
          "timeout", issuer="parent_observer", child_started=False,
          result={"error": "deadline_expired_before_spawn"},
      )
    environment = os.environ.copy()
    environment.update(NATIVE_THREAD_ENV)
    package_parent = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = package_parent + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    library_bin = str(Path(sys.prefix).resolve() / "Library" / "bin")
    environment["PATH"] = library_bin + os.pathsep + environment.get("PATH", "")
    try:
      try:
        process = subprocess.Popen(
            [sys.executable, self.worker_script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
            env=environment,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                if os.name == "nt" else 0
            ),
        )
      except Exception as error:
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=False,
            result={"error": f"spawn_failed:{type(error).__name__}"},
        )
      raw_request = json.dumps(
          request_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
      ).encode("utf-8")
      try:
        stdout, stderr = process.communicate(
            raw_request, timeout=max(0.0, deadline - time.monotonic()),
        )
      except Exception as communicate_error:
        kill_error = None
        try:
          process.kill()
        except Exception as error:
          kill_error = type(error).__name__
        stdout = stderr = b""
        cleanup_window_seconds = 0.5
        cleanup_started = time.monotonic()
        wait_error = None
        try:
          stdout, stderr = process.communicate(timeout=cleanup_window_seconds)
        except Exception as error:
          wait_error = type(error).__name__
        cleanup_elapsed_seconds = time.monotonic() - cleanup_started
        child_reaped = process.poll() is not None
        status = "timeout" if isinstance(communicate_error, subprocess.TimeoutExpired) else "kernel_error"
        telemetry_error = None
        try:
          capture_stage_evidence(stderr)
        except Exception as error:
          telemetry_error = type(error).__name__
          status = "kernel_error"
        return terminal(
            status, issuer="parent_observer", child_started=True,
            result={
                "error": f"communicate_failed:{type(communicate_error).__name__}",
                "kill_error": kill_error, "wait_error": wait_error,
                "cleanup_window_seconds": cleanup_window_seconds,
                "cleanup_elapsed_seconds": cleanup_elapsed_seconds,
                "child_reaped": child_reaped,
                "child_pid": process.pid,
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "telemetry_error_type": telemetry_error,
            },
        )
      try:
        capture_stage_evidence(stderr)
      except Exception as error:
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={
                "error": f"stage_telemetry_invalid:{type(error).__name__}",
                "child_pid": process.pid,
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            },
        )
      if time.monotonic() >= deadline:
        return terminal(
            "timeout", issuer="parent_observer", child_started=True,
            result={"error": "deadline_expired_before_parse", "child_pid": process.pid,
                    "stderr_sha256": hashlib.sha256(stderr).hexdigest()},
        )
      if len(stdout) > MAX_WORKER_STDOUT_BYTES_V4:
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={"error": "worker_stdout_budget_exceeded", "child_pid": process.pid,
                    "stdout_bytes": len(stdout),
                    "stderr_sha256": hashlib.sha256(stderr).hexdigest()},
        )
      try:
        worker_terminal = _strict_json_bytes(stdout, label="preprocess worker terminal")
        unsigned_terminal = dict(worker_terminal)
        observed_terminal_sha = unsigned_terminal.pop("terminal_payload_sha256", None)
        valid = (
            unsigned_terminal.get("schema_version") == PREPROCESS_WORKER_TERMINAL_SCHEMA_VERSION
            and unsigned_terminal.get("request_payload_sha256") == request_sha
            and observed_terminal_sha == canonical_sha256(unsigned_terminal)
            and process.returncode == 0
        )
      except Exception:
        worker_terminal, observed_terminal_sha, valid = {}, None, False
      if time.monotonic() >= deadline:
        return terminal(
            "timeout", issuer="parent_observer", child_started=True,
            result={"error": "deadline_expired_during_parse", "child_pid": process.pid,
                    "stderr_sha256": hashlib.sha256(stderr).hexdigest()},
        )
      if not valid:
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={"error": "worker_terminal_invalid", "child_pid": process.pid,
                    "stderr_sha256": hashlib.sha256(stderr).hexdigest()},
        )
      child_result = worker_terminal.get("result")
      if not isinstance(child_result, Mapping):
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={"error": "worker_result_invalid", "child_pid": process.pid},
        )
      status = str(child_result.get("status", "kernel_error"))
      if (
          child_result.get("call_nonce") != nonce
          or child_result.get("operation") != PREPROCESS_OPERATION_V4
      ):
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={"error": "worker_nonce_or_operation_echo_mismatch",
                    "child_pid": process.pid},
        )
      timeout_run = expired_after("worker_result_schema")
      if timeout_run is not None:
        return timeout_run
      if status != "ok":
        return terminal(
            "kernel_error", issuer="child_process_echo", child_started=True,
            result={
                "error": str(child_result.get("error", "preprocess_failed")),
                "child_pid": int(child_result.get("child_pid", process.pid)),
                "worker_terminal_payload_sha256": observed_terminal_sha,
            },
        )
      sanitized = child_result.get("sanitized_geometry")
      if not isinstance(sanitized, Mapping):
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={"error": "worker_sanitized_geometry_absent", "child_pid": process.pid},
        )
      try:
        validate_sanitized_geometry_v4(sanitized)
      except Exception as error:
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={"error": f"sanitized_geometry_invalid:{type(error).__name__}",
                    "child_pid": process.pid},
        )
      timeout_run = expired_after("compact_schema_validation")
      if timeout_run is not None:
        return timeout_run
      authority_unsigned = {
          "schema_version": SANITIZED_AUTHORITY_SCHEMA_VERSION,
          "preprocess_request_payload_sha256": request_sha,
          "worker_terminal_payload_sha256": observed_terminal_sha,
          "worker_code_bundle_sha256": code_bundle_sha,
          "sanitized_geometry": sanitized,
      }
      authority_payload_sha256 = canonical_sha256(authority_unsigned)
      timeout_run = expired_after("authority_canonical_hash")
      if timeout_run is not None:
        return timeout_run
      try:
        authority = SanitizedGeometryAuthorityV4(
            {**authority_unsigned,
             "authority_payload_sha256": authority_payload_sha256},
            _factory_token=_AUTHORITY_FACTORY_TOKEN,
        )
      except Exception as error:
        return terminal(
            "kernel_error", issuer="parent_observer", child_started=True,
            result={"error": f"sanitized_authority_invalid:{type(error).__name__}",
                    "child_pid": process.pid},
        )
      timeout_run = expired_after("authority_deep_freeze")
      if timeout_run is not None:
        return timeout_run
      authority_sha = authority.sha256
      timeout_run = expired_after("authority_revalidation")
      if timeout_run is not None:
        return timeout_run
      success_run = terminal(
          "ok", issuer="child_process_echo", child_started=True,
          authority=authority,
          result={
              "sanitized_geometry_sha256": sanitized["sanitized_geometry_sha256"],
              "sanitized_authority_sha256": authority_sha,
              "candidate_count": int(sanitized["candidate_domain_count"]),
              "compact_candidate_count": len(sanitized["cheap_geometry_candidates"]),
              "endpoint_count": len(sanitized["endpoints"]),
              "child_pid": int(child_result.get("child_pid", process.pid)),
              "worker_terminal_payload_sha256": observed_terminal_sha,
          },
      )
      timeout_run = expired_after("success_receipt_construction")
      return success_run if timeout_run is None else timeout_run
    finally:
      # No fixed grace interval is allowed outside the total wall deadline.
      if process is not None and process.poll() is None:
        try:
          process.kill()
        except Exception:
          pass
        remaining = deadline - time.monotonic()
        try:
          process.wait(timeout=max(0.0, remaining))
        except Exception:
          pass


__all__ = [
    "ConstraintManifoldPreprocessEndpointV4",
    "ConstraintManifoldPreprocessReceiptV4",
    "ConstraintManifoldPreprocessRequestV4",
    "ConstraintManifoldPreprocessRequestV5",
    "ConstraintManifoldPreprocessRunV4",
    "ConstraintManifoldPreprocessSupervisorV4",
    "LANDMARK_CAP_PER_ENDPOINT_PER_TYPE_V4",
    "PREPROCESS_OPERATION_V4", "PREPROCESS_RECEIPT_SCHEMA_VERSION",
    "PREPROCESS_REQUEST_SCHEMA_VERSION", "PREPROCESS_REQUEST_SCHEMA_VERSION_V5",
    "PREPROCESS_WORKER_TERMINAL_SCHEMA_VERSION",
    "PREPROCESS_STAGE_TELEMETRY_PREFIX_V7",
    "PREPROCESS_STAGE_TELEMETRY_SCHEMA_VERSION_V7",
    "SANITIZED_AUTHORITY_SCHEMA_VERSION", "SANITIZED_GEOMETRY_SCHEMA_VERSION",
    "SANITIZED_GEOMETRY_P9_COLLISION_TOPK_SCHEMA_VERSION_V1",
    "SANITIZED_GEOMETRY_P11_BODY_AABB_SCHEMA_VERSION_V1",
    "SANITIZED_GEOMETRY_P11_FACTORIZED_SCHEMA_VERSION_V2",
    "SANITIZED_GEOMETRY_P11_STRATIFIED_DIVERSITY_SCHEMA_VERSION_V1",
    "SanitizedGeometryAuthorityV4", "canonicalize_landmark_points_v4",
    "candidate_strata_p11_diversity_v1",
    "compact_candidate_domain_p11_body_aabb_v1", "compact_candidate_domain_v1",
    "compact_candidate_domain_p11_stratified_diversity_v1",
    "validate_compact_candidate_domain_v1", "plane_overlap_area_pruned_v4",
    "preprocess_policy_for_program_v1", "preprocess_policy_p11_body_aabb_v1",
    "preprocess_policy_p11_stratified_diversity_v1",
    "preprocess_policy_v4",
    "preprocess_source_sha256s_v4", "validate_sanitized_geometry_v4",
    "issue_preprocess_stage_event_v7", "parse_preprocess_stage_telemetry_v7",
]
