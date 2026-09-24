"""Fail-closed V2 constraint-manifold solver for frozen programs 9 and 11.

V2 is deliberately a new protocol.  It does not reinterpret V1 receipts.  In
particular, thresholds and adjacency requirements come from one factory-owned
policy/topology authority, every exact call has a reserved budget slot,
deadline and nonce-bound terminal receipt, and any started-call failure makes
the complete query unknown even when an earlier candidate was feasible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import secrets
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence
from weakref import WeakKeyDictionary

import numpy as np

from . import constraint_manifold_solver_v1 as _v1


SCHEMA_VERSION = "constraint_manifold_solver.v2"
CERTIFICATE_SCHEMA_VERSION = "constraint_manifold_certificate.v2"
POLICY_SCHEMA_VERSION = "constraint_manifold_frozen_policy.v2"
TOPOLOGY_SCHEMA_VERSION = "constraint_manifold_step_topology_authority.v2"
CHILD_RECEIPT_SCHEMA_VERSION = "constraint_manifold_exact_child_receipt.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TAU = 2.0 * math.pi
FROZEN_TESSELLATION_CHORD_TOLERANCE_MM_V2 = 1e-2
_POLICY_TOKEN = object()
_TOPOLOGY_TOKEN = object()
_TOPOLOGY_FACTORY_CAPABILITY_V2 = object()
_CERTIFICATE_TOKEN = object()


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
  if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{label} must match ^[0-9a-f]{{64}}$")
  return value


@dataclass(frozen=True, slots=True)
class ConstraintManifoldThresholdsV2:
  plane_normal_error_degrees: float = 1e-5
  plane_normal_gap_mm: float = 1e-6
  cylinder_axis_error_degrees: float = 1e-5
  cylinder_axis_distance_mm: float = 1e-6
  intended_clearance_mm: float = 0.1
  whole_common_volume_mm3: float = 1e-7
  minimum_plane_overlap_mm2: float = 1e-4
  minimum_cylinder_axial_overlap_mm: float = 1e-3
  minimum_cylinder_angular_overlap_radians: float = 1e-3
  minimum_trim_overlap_ratio: float = 0.05
  maximum_radial_clearance_mm: float = 0.1
  adjacent_gap_mm: float = 0.1


@dataclass(frozen=True, slots=True)
class ConstraintManifoldBudgetV2:
  max_candidates: int = 16
  max_occ_calls: int = 32
  wall_time_seconds: float = 60.0
  per_occ_child_timeout_seconds: float = 12.0


_FROZEN_POLICY_BODY_V2 = {
    "schema_version": POLICY_SCHEMA_VERSION,
    "authorized_programs": {
        "9": {
            "descriptor": ["support", "seat_plane", "plane", "plane"],
            "topology_requirement": "adjacent_shoulder_optional",
            "candidate_policy": "plane_primary_then_aabb_separation_secondary.v2",
        },
        "11": {
            "descriptor": ["insert", "shaft_in_bore", "cylinder", "cylinder"],
            "topology_requirement": "adjacent_shoulder_optional",
            "candidate_policy": "periodic_tessellated_trim_axis_manifold.v2",
        },
    },
    "thresholds": asdict(ConstraintManifoldThresholdsV2()),
    "budget": asdict(ConstraintManifoldBudgetV2()),
    "geometry_approximation": {
        "cylinder_tessellation_chord_tolerance_mm": (
            FROZEN_TESSELLATION_CHORD_TOLERANCE_MM_V2
        ),
        "overlap_error_upper_bound": (
            "sum_per_face(4*wire_length_mm*chord_tolerance_mm+"
            "4*pi*wire_count*chord_tolerance_mm^2)"
        ),
        "overlap_acceptance_uses_lower_bound": True,
    },
    "failure_rule": (
        "preprocessing_wall_timeout_or_any_started_child_timeout_or_kernel_error_"
        "makes_query_unknown"
    ),
}
_FROZEN_POLICY_CODE_V2 = (
    "constraint_manifold_solver.v2|frozen-thresholds|official-topology|"
    "reserve-before-child|total-query-deadline|nonce-terminal-receipt|"
    "fail-query-on-child-failure|"
    "p9-aabb-secondary|p11-periodic-triangle-trim|chord-0.01|"
    "conservative-overlap-error"
)


class FrozenConstraintManifoldPolicyV2:
  __slots__ = ("_token", "_payload", "_sha256", "_code_sha256")

  def __init__(self, *, _factory_token: object) -> None:
    if _factory_token is not _POLICY_TOKEN:
      raise TypeError("constraint-manifold policy is factory-owned")
    object.__setattr__(self, "_token", _factory_token)
    object.__setattr__(self, "_payload", MappingProxyType(dict(_FROZEN_POLICY_BODY_V2)))
    object.__setattr__(self, "_sha256", _canonical_sha256(_FROZEN_POLICY_BODY_V2))
    object.__setattr__(self, "_code_sha256", hashlib.sha256(
        _FROZEN_POLICY_CODE_V2.encode("utf-8")
    ).hexdigest())

  @property
  def thresholds(self) -> ConstraintManifoldThresholdsV2:
    return ConstraintManifoldThresholdsV2()

  @property
  def budget(self) -> ConstraintManifoldBudgetV2:
    return ConstraintManifoldBudgetV2()

  @property
  def sha256(self) -> str:
    return self._sha256

  @property
  def code_sha256(self) -> str:
    return self._code_sha256

  def topology_requirement(self, program_index: int) -> str:
    try:
      return str(_FROZEN_POLICY_BODY_V2["authorized_programs"][str(program_index)][
          "topology_requirement"
      ])
    except KeyError as error:
      raise ValueError("program is absent from frozen manifold policy") from error

  def payload(self) -> dict[str, Any]:
    return {
        **json.loads(json.dumps(_FROZEN_POLICY_BODY_V2)),
        "policy_payload_sha256": self.sha256,
        "policy_code_sha256": self.code_sha256,
    }

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("constraint-manifold policy is immutable")


OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V2 = FrozenConstraintManifoldPolicyV2(
    _factory_token=_POLICY_TOKEN
)


@dataclass(frozen=True, slots=True)
class CylinderTrimV2:
  """Actual OCC face tessellation in unwrapped analytic-cylinder (u,z)."""

  triangles_uz: tuple[tuple[tuple[float, float], ...], ...]
  radius_mm: float
  surface_side: str
  wire_count: int
  hole_count: int
  seam_crossing_triangle_count: int
  exact_surface_area_mm2: float | None = None
  boundary_length_mm: float | None = None
  tessellation_chord_tolerance_mm: float = FROZEN_TESSELLATION_CHORD_TOLERANCE_MM_V2
  source: str = "occ_face_wires_and_tessellated_triangles.v2"

  def __post_init__(self) -> None:
    if not self.triangles_uz:
      raise ValueError("cylinder trim has no actual tessellated triangles")
    if not math.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
      raise ValueError("cylinder radius differs")
    if self.surface_side not in {"interior", "exterior"}:
      raise ValueError("cylinder surface side differs")
    if self.wire_count < 1 or not 0 <= self.hole_count < self.wire_count:
      raise ValueError("cylinder wire/hole topology differs")
    if self.seam_crossing_triangle_count < 0:
      raise ValueError("cylinder seam count differs")
    if (
        not math.isfinite(self.tessellation_chord_tolerance_mm)
        or self.tessellation_chord_tolerance_mm
        != FROZEN_TESSELLATION_CHORD_TOLERANCE_MM_V2
    ):
      raise ValueError("cylinder tessellation chord tolerance differs")
    for triangle in self.triangles_uz:
      values = np.asarray(triangle, dtype=float)
      if values.shape != (3, 2) or not np.isfinite(values).all():
        raise ValueError("cylinder trim triangle differs")
      if _v1._triangle_area(values) <= 1e-12:
        raise ValueError("cylinder trim contains a degenerate triangle")
    if self.exact_surface_area_mm2 is None:
      object.__setattr__(self, "exact_surface_area_mm2", self.parametric_area_mm2)
    if self.boundary_length_mm is None:
      # Unit/synthetic trims may omit OCC wire lengths. Retaining all triangle
      # edges overestimates the true boundary and therefore stays conservative.
      edge_sum = 0.0
      for triangle in self.triangles_uz:
        values = np.asarray(triangle, dtype=float)
        metric = values.copy()
        metric[:, 0] *= self.radius_mm
        edge_sum += sum(
            float(np.linalg.norm(metric[(index + 1) % 3] - metric[index]))
            for index in range(3)
        )
      object.__setattr__(self, "boundary_length_mm", edge_sum)
    if (
        not math.isfinite(float(self.exact_surface_area_mm2))
        or float(self.exact_surface_area_mm2) <= 0.0
        or not math.isfinite(float(self.boundary_length_mm))
        or float(self.boundary_length_mm) <= 0.0
    ):
      raise ValueError("cylinder exact area/boundary evidence differs")

  @property
  def parametric_area_mm2(self) -> float:
    return self.radius_mm * sum(
        _v1._triangle_area(np.asarray(row, dtype=float)) for row in self.triangles_uz
    )

  @property
  def axial_interval_mm(self) -> tuple[float, float]:
    values = [point[1] for triangle in self.triangles_uz for point in triangle]
    return min(values), max(values)

  @property
  def angular_midpoint_radians(self) -> float:
    values = [point[0] for triangle in self.triangles_uz for point in triangle]
    return sum(values) / len(values)

  @property
  def overlap_area_error_upper_mm2(self) -> float:
    delta = self.tessellation_chord_tolerance_mm
    boundary_band = (
        4.0 * float(self.boundary_length_mm) * delta
        + 4.0 * math.pi * self.wire_count * delta * delta
    )
    measured_area_delta = abs(
        self.parametric_area_mm2 - float(self.exact_surface_area_mm2)
    )
    return max(boundary_band, measured_area_delta)

  def payload(self) -> dict[str, Any]:
    return {
        "triangles_uz": [[list(point) for point in row] for row in self.triangles_uz],
        "radius_mm": self.radius_mm, "surface_side": self.surface_side,
        "wire_count": self.wire_count, "hole_count": self.hole_count,
        "seam_crossing_triangle_count": self.seam_crossing_triangle_count,
        "exact_surface_area_mm2": self.exact_surface_area_mm2,
        "boundary_length_mm": self.boundary_length_mm,
        "tessellation_chord_tolerance_mm": self.tessellation_chord_tolerance_mm,
        "overlap_area_error_upper_mm2": self.overlap_area_error_upper_mm2,
        "source": self.source,
    }


@dataclass(frozen=True, slots=True)
class ManifoldFaceGeometryV2:
  part_slot: str
  graph_face_index: int
  raw_occ_face_index: int
  face_signature_sha256: str
  surface_type: str
  origin_world_mm: tuple[float, float, float]
  rotation_local_to_world: tuple[tuple[float, float, float], ...]
  plane_trim: _v1.PlaneTrimV1 | None = None
  cylinder_trim: CylinderTrimV2 | None = None

  def __post_init__(self) -> None:
    if self.part_slot not in {"a", "b"} or min(
        self.graph_face_index, self.raw_occ_face_index
    ) < 0:
      raise ValueError("manifold face identity differs")
    _require_sha256(self.face_signature_sha256, label="face signature")
    origin = np.asarray(self.origin_world_mm, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
      raise ValueError("manifold face origin differs")
    _v1._proper_rotation(self.rotation_local_to_world, label="manifold face frame")
    if self.surface_type == "plane":
      valid = self.plane_trim is not None and self.cylinder_trim is None
    elif self.surface_type == "cylinder":
      valid = self.cylinder_trim is not None and self.plane_trim is None
    else:
      valid = False
    if not valid:
      raise ValueError("manifold face surface/trim schema differs")

  def payload(self) -> dict[str, Any]:
    return {
        "part_slot": self.part_slot, "graph_face_index": self.graph_face_index,
        "raw_occ_face_index": self.raw_occ_face_index,
        "face_signature_sha256": self.face_signature_sha256,
        "surface_type": self.surface_type,
        "origin_world_mm": list(self.origin_world_mm),
        "rotation_local_to_world": [list(row) for row in self.rotation_local_to_world],
        "plane_trim": None if self.plane_trim is None else self.plane_trim.payload(),
        "cylinder_trim": None if self.cylinder_trim is None else self.cylinder_trim.payload(),
    }


_TOPOLOGY_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


class StepTopologyAuthorityV2:
  __slots__ = ("__weakref__",)

  def __init__(self, payload: Mapping[str, Any], replay: Callable[[], Mapping[str, Any]],
               *, _factory_token: object) -> None:
    if _factory_token is not _TOPOLOGY_TOKEN or not callable(replay):
      raise TypeError("STEP topology authority is factory-owned")
    frozen = json.loads(json.dumps(payload))
    if frozen.get("schema_version") != TOPOLOGY_SCHEMA_VERSION:
      raise ValueError("STEP topology authority schema differs")
    observed = frozen.pop("authority_payload_sha256", None)
    if observed != _canonical_sha256(frozen):
      raise ValueError("STEP topology authority commitment differs")
    frozen["authority_payload_sha256"] = observed
    _TOPOLOGY_STATES[self] = MappingProxyType({"payload": frozen, "replay": replay})
    self.revalidate()

  def revalidate(self) -> None:
    state = _TOPOLOGY_STATES[self]
    payload = dict(state["payload"])
    expected = payload.pop("authority_payload_sha256")
    replayed = json.loads(json.dumps(state["replay"]()))
    if replayed != payload or _canonical_sha256(replayed) != expected:
      raise ValueError("STEP topology authority replay differs")

  @property
  def sha256(self) -> str:
    self.revalidate()
    return str(_TOPOLOGY_STATES[self]["payload"]["authority_payload_sha256"])

  def evidence(self, program_index: int) -> Mapping[str, Any]:
    self.revalidate()
    payload = _TOPOLOGY_STATES[self]["payload"]
    if int(payload["program_index"]) != program_index:
      raise ValueError("STEP topology authority program differs")
    return MappingProxyType(dict(payload["topology_evidence"]))

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("STEP topology authority is immutable")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("STEP topology authority is not serializable")


def _bind_step_topology_authority_v2(
    *, program_index: int, geometry_witness_sha256: str,
    topology_evidence: Mapping[str, Any], replay: Callable[[], Mapping[str, Any]],
    _factory_capability: object,
) -> StepTopologyAuthorityV2:
  if _factory_capability is not _TOPOLOGY_FACTORY_CAPABILITY_V2:
    raise TypeError("STEP topology authority requires the loader-owned factory capability")
  _require_sha256(geometry_witness_sha256, label="geometry witness")
  requirement = OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V2.topology_requirement(program_index)
  unsigned = {
      "schema_version": TOPOLOGY_SCHEMA_VERSION,
      "program_index": program_index,
      "geometry_witness_sha256": geometry_witness_sha256,
      "official_policy_sha256": OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V2.sha256,
      "official_policy_code_sha256": OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V2.code_sha256,
      "official_requirement": requirement,
      "topology_evidence": json.loads(json.dumps(topology_evidence)),
  }
  payload = {**unsigned, "authority_payload_sha256": _canonical_sha256(unsigned)}
  return StepTopologyAuthorityV2(payload, replay, _factory_token=_TOPOLOGY_TOKEN)


@dataclass(frozen=True, slots=True)
class ConstraintManifoldRequestV2:
  query_id: str
  program_index: int
  face_a: ManifoldFaceGeometryV2
  face_b: ManifoldFaceGeometryV2
  child_world_row_major: tuple[float, ...]
  geometry_witness_sha256: str
  topology_authority: StepTopologyAuthorityV2
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != SCHEMA_VERSION or not self.query_id:
      raise ValueError("constraint-manifold V2 request identity differs")
    if self.program_index not in {9, 11}:
      raise ValueError("constraint-manifold V2 authorizes only programs 9/11")
    _require_sha256(self.geometry_witness_sha256, label="geometry witness")
    if type(self.topology_authority) is not StepTopologyAuthorityV2:
      raise TypeError("request requires factory-owned STEP topology authority")
    if self.topology_authority.sha256 == "":
      raise ValueError("topology authority differs")
    topology = self.topology_authority.evidence(self.program_index)
    if topology.get("geometry_witness_sha256") != self.geometry_witness_sha256:
      raise ValueError("topology/geometry witness binding differs")
    expected = "plane" if self.program_index == 9 else "cylinder"
    if (self.face_a.part_slot, self.face_b.part_slot) != ("a", "b") or (
        self.face_a.surface_type, self.face_b.surface_type
    ) != (expected, expected):
      raise ValueError("selected faces differ from official program policy")
    _v1._matrix_tuple(np.asarray(self.child_world_row_major).reshape(4, 4))
    _v1.assert_no_source_pose_oracle_v1(self.payload())

  @property
  def policy(self) -> FrozenConstraintManifoldPolicyV2:
    return OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V2

  @property
  def thresholds(self) -> ConstraintManifoldThresholdsV2:
    return self.policy.thresholds

  @property
  def budget(self) -> ConstraintManifoldBudgetV2:
    return self.policy.budget

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version, "query_id": self.query_id,
        "program_index": self.program_index, "face_a": self.face_a.payload(),
        "face_b": self.face_b.payload(),
        "child_world_row_major": list(self.child_world_row_major),
        "geometry_witness_sha256": self.geometry_witness_sha256,
        "topology_authority_sha256": self.topology_authority.sha256,
        "frozen_policy": self.policy.payload(),
    }


@dataclass(frozen=True, slots=True)
class ExactChildReceiptV2:
  call_nonce: str
  operation: str
  terminal_status: str
  issuer: str
  result_items: tuple[tuple[str, Any], ...]
  receipt_payload_sha256: str
  schema_version: str = CHILD_RECEIPT_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CHILD_RECEIPT_SCHEMA_VERSION:
      raise ValueError("exact child receipt schema differs")
    if not self.call_nonce or self.operation != "candidate_exact_bundle":
      raise ValueError("exact child receipt identity differs")
    if self.terminal_status not in {"ok", "timeout", "kernel_error"}:
      raise ValueError("exact child receipt terminal status differs")
    if self.issuer not in {"child_process_echo", "parent_observer"}:
      raise ValueError("exact child receipt issuer differs")
    _require_sha256(self.receipt_payload_sha256, label="child receipt commitment")
    if self.receipt_payload_sha256 != _canonical_sha256(self.unsigned_payload()):
      raise ValueError("exact child receipt commitment differs")

  @property
  def result(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.result_items))

  def unsigned_payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version, "call_nonce": self.call_nonce,
        "operation": self.operation, "terminal_status": self.terminal_status,
        "issuer": self.issuer,
        "result": dict(self.result_items),
    }

  def payload(self) -> dict[str, Any]:
    return {**self.unsigned_payload(), "receipt_payload_sha256": self.receipt_payload_sha256}


def issue_exact_child_receipt_v2(
    *, call_nonce: str, operation: str, terminal_status: str,
    result: Mapping[str, Any] | None = None, issuer: str = "child_process_echo",
) -> ExactChildReceiptV2:
  items = tuple(sorted((str(k), v) for k, v in (result or {}).items()))
  unsigned = {
      "schema_version": CHILD_RECEIPT_SCHEMA_VERSION,
      "call_nonce": call_nonce, "operation": operation,
      "terminal_status": terminal_status, "issuer": issuer, "result": dict(items),
  }
  return ExactChildReceiptV2(
      call_nonce=call_nonce, operation=operation, terminal_status=terminal_status,
      issuer=issuer, result_items=items, receipt_payload_sha256=_canonical_sha256(unsigned),
  )


class ExactManifoldExecutorV2(Protocol):
  def run_child(
      self, candidate: _v1.ConstraintManifoldCandidateV1, *, operation: str,
      call_nonce: str, absolute_deadline: float,
  ) -> ExactChildReceiptV2: ...


def _periodic_triangle_overlap(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray],
    *, _profile: dict[str, int | float] | None = None,
) -> float:
  if not first or not second:
    return 0.0

  def bounds(triangle: np.ndarray) -> tuple[float, float, float, float]:
    # Outward nextafter makes every broad-phase box a superset of the exact
    # floating-point triangle box, including equality at cell boundaries.
    low = np.nextafter(np.min(triangle, axis=0), -math.inf)
    high = np.nextafter(np.max(triangle, axis=0), math.inf)
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])

  index_started = time.perf_counter()
  first_rows = tuple((triangle, bounds(triangle)) for triangle in first)
  global_low = min(row[1][0] for row in first_rows)
  global_high = max(row[1][1] for row in first_rows)
  copies: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
  for triangle in second:
    b_low, b_high, _z_low, _z_high = bounds(triangle)
    # The extra periodic copy on either side is deliberately conservative.
    shift_low = math.floor((global_low - b_high) / _TAU) - 1
    shift_high = math.ceil((global_high - b_low) / _TAU) + 1
    for shift_index in range(shift_low, shift_high + 1):
      moved = triangle.copy()
      moved[:, 0] += shift_index * _TAU
      moved_bounds = bounds(moved)
      if moved_bounds[1] < global_low or moved_bounds[0] > global_high:
        continue
      copies.append((moved, moved_bounds))
  if not copies:
    if _profile is not None:
      _profile["overlap_index_elapsed_seconds_raw"] = float(
          _profile.get("overlap_index_elapsed_seconds_raw", 0.0)
      ) + (time.perf_counter() - index_started)
    return 0.0

  bin_count = max(1, min(4096, int(math.ceil(math.sqrt(len(copies))))))
  span = max(global_high - global_low, np.finfo(float).tiny)
  width = span / bin_count

  def bin_index(value: float) -> int:
    return min(bin_count - 1, max(0, int(math.floor((value - global_low) / width))))

  bins: dict[int, list[int]] = {}
  for copy_index, (_triangle, row_bounds) in enumerate(copies):
    for cell in range(bin_index(row_bounds[0]), bin_index(row_bounds[1]) + 1):
      bins.setdefault(cell, []).append(copy_index)
  if _profile is not None:
    _profile["overlap_periodic_copy_count"] = int(
        _profile.get("overlap_periodic_copy_count", 0)
    ) + len(copies)
    _profile["overlap_index_elapsed_seconds_raw"] = float(
        _profile.get("overlap_index_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - index_started)

  total = 0.0
  exact_started = time.perf_counter()
  index_candidate_count = exact_count = 0
  for tri_a, a_bounds in first_rows:
    possible: set[int] = set()
    for cell in range(bin_index(a_bounds[0]), bin_index(a_bounds[1]) + 1):
      possible.update(bins.get(cell, ()))
    index_candidate_count += len(possible)
    for copy_index in sorted(possible):
      moved, b_bounds = copies[copy_index]
      if (
          b_bounds[1] < a_bounds[0] or b_bounds[0] > a_bounds[1]
          or b_bounds[3] < a_bounds[2] or b_bounds[2] > a_bounds[3]
      ):
        continue
      exact_count += 1
      total += _v1._polygon_area(_v1._convex_clip(tri_a, moved))
  if _profile is not None:
    _profile["overlap_index_candidate_pair_count"] = int(
        _profile.get("overlap_index_candidate_pair_count", 0)
    ) + index_candidate_count
    _profile["overlap_exact_clip_pair_count"] = int(
        _profile.get("overlap_exact_clip_pair_count", 0)
    ) + exact_count
    _profile["overlap_exact_elapsed_seconds_raw"] = float(
        _profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - exact_started)
  return total


@dataclass(frozen=True, slots=True)
class PeriodicYawStateOverlapIndexV1:
  first_rows: tuple[tuple[np.ndarray, tuple[float, float, float, float]], ...]
  second_base: tuple[np.ndarray, ...]
  copy_descriptors: tuple[tuple[int, int], ...]
  global_low: float
  global_high: float
  bin_count: int
  bin_width: float
  bins: Mapping[int, tuple[int, ...]]


def _triangle_bounds_v1(triangle: np.ndarray) -> tuple[float, float, float, float]:
  low = np.nextafter(np.min(triangle, axis=0), -math.inf)
  high = np.nextafter(np.max(triangle, axis=0), math.inf)
  return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def build_periodic_yaw_state_overlap_index_v1(
    first: Sequence[np.ndarray], second_base: Sequence[np.ndarray],
    *, _profile: dict[str, int | float] | None = None,
) -> PeriodicYawStateOverlapIndexV1:
  """Build periodic copies and their angular bins once for one (sign, yaw)."""

  if not first or not second_base:
    raise ValueError("periodic yaw-state index requires non-empty triangles")
  started = time.perf_counter()
  first_rows = tuple((np.asarray(row, dtype=float), _triangle_bounds_v1(
      np.asarray(row, dtype=float),
  )) for row in first)
  second = tuple(np.asarray(row, dtype=float) for row in second_base)
  if any(row.shape != (3, 2) or not np.isfinite(row).all()
         for row in (*tuple(row[0] for row in first_rows), *second)):
    raise ValueError("periodic yaw-state triangle domain differs")
  global_low = min(row[1][0] for row in first_rows)
  global_high = max(row[1][1] for row in first_rows)
  descriptors: list[tuple[int, int]] = []
  u_bounds: list[tuple[float, float]] = []
  for source_index, triangle in enumerate(second):
    b_low, b_high, _z_low, _z_high = _triangle_bounds_v1(triangle)
    shift_low = math.floor((global_low - b_high) / _TAU) - 1
    shift_high = math.ceil((global_high - b_low) / _TAU) + 1
    for shift_index in range(shift_low, shift_high + 1):
      moved = triangle.copy()
      moved[:, 0] += shift_index * _TAU
      moved_bounds = _triangle_bounds_v1(moved)
      if moved_bounds[1] < global_low or moved_bounds[0] > global_high:
        continue
      descriptors.append((source_index, shift_index))
      u_bounds.append((moved_bounds[0], moved_bounds[1]))
  bin_count = max(1, min(4096, int(math.ceil(math.sqrt(len(descriptors))))))
  span = max(global_high - global_low, np.finfo(float).tiny)
  width = span / bin_count

  def cell(value: float) -> int:
    return min(bin_count - 1, max(0, int(math.floor((value - global_low) / width))))

  bins: dict[int, list[int]] = {}
  for copy_index, row_bounds in enumerate(u_bounds):
    for bin_index in range(cell(row_bounds[0]), cell(row_bounds[1]) + 1):
      bins.setdefault(bin_index, []).append(copy_index)
  if _profile is not None:
    _profile["overlap_index_build_count"] = int(
        _profile.get("overlap_index_build_count", 0)
    ) + 1
    _profile["overlap_periodic_copy_count"] = int(
        _profile.get("overlap_periodic_copy_count", 0)
    ) + len(descriptors)
    _profile["overlap_index_elapsed_seconds_raw"] = float(
        _profile.get("overlap_index_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - started)
  return PeriodicYawStateOverlapIndexV1(
      first_rows=first_rows, second_base=second,
      copy_descriptors=tuple(descriptors),
      global_low=global_low, global_high=global_high,
      bin_count=bin_count, bin_width=width,
      bins=MappingProxyType({key: tuple(value) for key, value in bins.items()}),
  )


def periodic_overlap_from_yaw_state_index_v1(
    index: PeriodicYawStateOverlapIndexV1, *, offset_uz: Sequence[float],
    translated_second: Sequence[np.ndarray] | None = None,
    _profile: dict[str, int | float] | None = None,
) -> float:
  """Evaluate one axial offset while reusing the periodic angular index."""

  offset = np.asarray(offset_uz, dtype=float)
  if (offset.shape != (2,) or not np.isfinite(offset).all()
      or float(offset[0]) != 0.0):
    raise ValueError("periodic yaw-state offset differs")
  second = tuple(np.asarray(row, dtype=float) for row in (
      tuple(row + offset for row in index.second_base)
      if translated_second is None else translated_second
  ))
  if (len(second) != len(index.second_base)
      or any(row.shape != (3, 2) or not np.isfinite(row).all() for row in second)):
    raise ValueError("periodic translated triangle domain differs")

  copies: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
  for source_index, shift_index in index.copy_descriptors:
    moved = second[source_index].copy()
    moved[:, 0] += shift_index * _TAU
    copies.append((moved, _triangle_bounds_v1(moved)))

  def cell(value: float) -> int:
    return min(index.bin_count - 1, max(
        0, int(math.floor((value - index.global_low) / index.bin_width)),
    ))

  total = 0.0
  exact_started = time.perf_counter()
  index_candidate_count = exact_count = 0
  for tri_a, a_bounds in index.first_rows:
    possible: set[int] = set()
    for bin_index in range(cell(a_bounds[0]), cell(a_bounds[1]) + 1):
      possible.update(index.bins.get(bin_index, ()))
    index_candidate_count += len(possible)
    for copy_index in sorted(possible):
      moved, b_bounds = copies[copy_index]
      if (
          b_bounds[1] < a_bounds[0] or b_bounds[0] > a_bounds[1]
          or b_bounds[3] < a_bounds[2] or b_bounds[2] > a_bounds[3]
      ):
        continue
      exact_count += 1
      total += _v1._polygon_area(_v1._convex_clip(tri_a, moved))
  if _profile is not None:
    _profile["overlap_index_candidate_pair_count"] = int(
        _profile.get("overlap_index_candidate_pair_count", 0)
    ) + index_candidate_count
    _profile["overlap_exact_clip_pair_count"] = int(
        _profile.get("overlap_exact_clip_pair_count", 0)
    ) + exact_count
    _profile["overlap_exact_elapsed_seconds_raw"] = float(
        _profile.get("overlap_exact_elapsed_seconds_raw", 0.0)
    ) + (time.perf_counter() - exact_started)
  return total


def _cylinder_candidates_v2(
    request: ConstraintManifoldRequestV2,
) -> tuple[_v1.ConstraintManifoldCandidateV1, ...]:
  a, b = request.face_a, request.face_b
  assert a.cylinder_trim is not None and b.cylinder_trim is not None
  ta, tb = a.cylinder_trim, b.cylinder_trim
  ra, rb = np.asarray(a.rotation_local_to_world), np.asarray(b.rotation_local_to_world)
  oa, ob = np.asarray(a.origin_world_mm), np.asarray(b.origin_world_mm)
  child = np.asarray(request.child_world_row_major, dtype=float).reshape(4, 4)
  triangles_a = tuple(np.asarray(row, dtype=float) for row in ta.triangles_uz)
  amid, bmid = ta.angular_midpoint_radians, tb.angular_midpoint_radians
  alo, ahi = ta.axial_interval_mm
  blo, bhi = tb.axial_interval_mm
  rows: list[_v1.ConstraintManifoldCandidateV1] = []
  for sign in (1.0, -1.0):
    base_yaw = amid - bmid if sign > 0 else amid + bmid
    for yaw_index, yaw in enumerate(_v1._unique_angles((base_yaw, base_yaw + math.pi))):
      x = math.cos(yaw) * ra[:, 0] + math.sin(yaw) * ra[:, 1]
      y = -math.sin(yaw) * ra[:, 0] + math.cos(yaw) * ra[:, 1]
      target_rotation = np.stack((x, y if sign > 0 else -y, sign * ra[:, 2]), axis=1)
      rotation = _v1._matmul(target_rotation, rb.T)
      transformed_interval = (blo, bhi) if sign > 0 else (-bhi, -blo)
      centre = sum(transformed_interval) * 0.5
      offsets = (
          ("interval_center", (alo + ahi) * 0.5 - centre),
          ("lower_endpoint", alo - transformed_interval[0]),
          ("upper_endpoint", ahi - transformed_interval[1]),
          ("opposed_endpoint", alo - transformed_interval[1]),
      )
      for axial_index, (tag, offset) in enumerate(offsets):
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = oa + offset * ra[:, 2] - _v1._matvec(rotation, ob)
        triangles_b = []
        for triangle in tb.triangles_uz:
          values = np.asarray(triangle, dtype=float)
          u = yaw + values[:, 0] if sign > 0 else yaw - values[:, 0]
          z = offset + values[:, 1] if sign > 0 else offset - values[:, 1]
          triangles_b.append(np.stack((u, z), axis=1))
        overlap_parametric = _periodic_triangle_overlap(triangles_a, triangles_b)
        overlap_estimate = min(ta.radius_mm, tb.radius_mm) * overlap_parametric
        overlap_error = (
            ta.overlap_area_error_upper_mm2 + tb.overlap_area_error_upper_mm2
        )
        overlap = max(0.0, overlap_estimate - overlap_error)
        ratio = overlap / min(
            float(ta.exact_surface_area_mm2), float(tb.exact_surface_area_mm2)
        )
        moved_origin = _v1._matvec(rotation, ob) + delta[:3, 3]
        moved_axis = _v1._matvec(rotation, rb[:, 2])
        radial = moved_origin - oa - np.dot(moved_origin - oa, ra[:, 2]) * ra[:, 2]
        axial = max(0.0, min(ahi, transformed_interval[1] + offset)
                    - max(alo, transformed_interval[0] + offset))
        rows.append(_v1._candidate(
            candidate_id=f"v2-p11-s{int(sign):+d}-y{yaw_index:02d}-z{axial_index:02d}-{tag}",
            program_index=11, delta=delta, child_world=child,
            overlap_measure=overlap, overlap_ratio=ratio,
            axial_overlap=axial,
            angular_overlap=(overlap_parametric / max(axial, 1e-12)),
            metrics={
                "axis_error_degrees": _v1._angle_degrees(
                    moved_axis, ra[:, 2], unoriented=True
                ),
                "radial_axis_distance_mm": _v1._norm(radial),
                "radius_difference_mm": abs(ta.radius_mm - tb.radius_mm),
                "trim_overlap_estimate_mm2": overlap_estimate,
                "trim_overlap_error_upper_mm2": overlap_error,
                "trim_overlap_lower_bound_mm2": overlap,
                "trim_overlap_ratio_lower_bound": ratio,
            },
            grammar_tags=("actual_periodic_trim", f"axis_sign_{int(sign):+d}", tag),
            displacement_mm=_v1._norm(moved_origin - ob),
        ))
  return tuple(rows[:request.budget.max_candidates])


def generate_constraint_manifold_candidates_v2(
    request: ConstraintManifoldRequestV2,
) -> tuple[_v1.ConstraintManifoldCandidateV1, ...]:
  if type(request) is not ConstraintManifoldRequestV2:
    raise TypeError("constraint-manifold V2 requires a typed request")
  if request.program_index == 9:
    proxy = SimpleNamespace(
        face_a=request.face_a, face_b=request.face_b,
        child_world_row_major=request.child_world_row_major,
        budget=_v1.ConstraintManifoldBudgetV1(
            max_candidates=4, max_occ_calls=request.budget.max_occ_calls,
            wall_time_seconds=request.budget.wall_time_seconds,
        ),
    )
    rows = _v1._plane_candidates(proxy)
  else:
    rows = _cylinder_candidates_v2(request)
  unique: dict[str, _v1.ConstraintManifoldCandidateV1] = {}
  for row in rows:
    unique.setdefault(_canonical_sha256({
        "program_index": row.program_index,
        "world_delta_row_major": row.world_delta_row_major,
    }), row)
  if not unique or len(unique) > request.budget.max_candidates:
    raise ValueError("constraint-manifold V2 candidate domain differs")
  return tuple(unique.values())


def _analytic_reasons_v2(
    request: ConstraintManifoldRequestV2,
    candidate: _v1.ConstraintManifoldCandidateV1,
) -> tuple[str, ...]:
  thresholds = request.thresholds
  reasons: list[str] = []
  if request.program_index == 9:
    if candidate.overlap_measure < thresholds.minimum_plane_overlap_mm2:
      reasons.append("plane_zero_or_tiny_trim_overlap")
  else:
    assert request.face_a.cylinder_trim is not None
    assert request.face_b.cylinder_trim is not None
    first, second = request.face_a.cylinder_trim, request.face_b.cylinder_trim
    if {first.surface_side, second.surface_side} != {"exterior", "interior"}:
      reasons.append("shaft_bore_side_relation")
    else:
      shaft = first if first.surface_side == "exterior" else second
      bore = first if first.surface_side == "interior" else second
      clearance = bore.radius_mm - shaft.radius_mm
      if clearance < -1e-9 or clearance > thresholds.maximum_radial_clearance_mm:
        reasons.append("shaft_bore_radius_relation")
    if (candidate.axial_overlap_mm or 0.0) < thresholds.minimum_cylinder_axial_overlap_mm:
      reasons.append("cylinder_zero_axial_overlap")
    if (candidate.angular_overlap_radians or 0.0) < thresholds.minimum_cylinder_angular_overlap_radians:
      reasons.append("cylinder_zero_angular_overlap")
  if candidate.overlap_ratio < thresholds.minimum_trim_overlap_ratio:
    reasons.append("trim_overlap_ratio")
  return tuple(sorted(set(reasons)))


def _shift_plane_candidate_v2(
    request: ConstraintManifoldRequestV2,
    candidate: _v1.ConstraintManifoldCandidateV1,
    offset_xy: Sequence[float], *, tag: str,
) -> _v1.ConstraintManifoldCandidateV1 | None:
  a, b = request.face_a, request.face_b
  assert a.plane_trim is not None and b.plane_trim is not None
  delta = candidate.world_delta
  ra = np.asarray(a.rotation_local_to_world)
  offset = np.asarray(offset_xy, dtype=float).reshape(2)
  delta[:3, 3] += ra[:, 0] * offset[0] + ra[:, 1] * offset[1]
  tri_a = _v1._reference_plane_triangles(a)
  moved = _v1._transformed_plane_triangles(b, delta, a)
  overlap = _v1._plane_overlap_area(tri_a, moved)
  ratio = overlap / min(a.plane_trim.area_mm2, b.plane_trim.area_mm2)
  if (
      overlap < request.thresholds.minimum_plane_overlap_mm2
      or ratio < request.thresholds.minimum_trim_overlap_ratio
  ):
    return None
  child = np.asarray(request.child_world_row_major, dtype=float).reshape(4, 4)
  metrics = dict(candidate.constraint_metrics)
  return _v1._candidate(
      candidate_id=f"{candidate.candidate_id}-secondary-{tag}",
      program_index=9, delta=delta, child_world=child,
      overlap_measure=overlap, overlap_ratio=ratio,
      axial_overlap=None, angular_overlap=None, metrics=metrics,
      grammar_tags=(*candidate.grammar_tags, "collision_free_secondary", tag),
      displacement_mm=candidate.displacement_mm + _v1._norm(offset),
  )


@dataclass(frozen=True, slots=True)
class ConstraintManifoldSolveResultV2:
  selected: _v1.ConstraintManifoldCandidateV1 | None
  certificate: Mapping[str, Any] | None
  rejection_certificate: Mapping[str, Any] | None
  candidate_count: int
  ledger_items: tuple[tuple[str, Any], ...]
  terminal_status: str

  @property
  def ledger(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.ledger_items))


def _certificate_v2(
    request: ConstraintManifoldRequestV2,
    candidate: _v1.ConstraintManifoldCandidateV1,
    *, accepted: bool, reason_codes: Sequence[str], observation: Mapping[str, Any] | None,
    ledger: Mapping[str, Any], child_receipts: Sequence[ExactChildReceiptV2],
) -> Mapping[str, Any]:
  request.topology_authority.revalidate()
  payload = {
      "schema_version": CERTIFICATE_SCHEMA_VERSION,
      "query_id": request.query_id, "program_index": request.program_index,
      "request_sha256": _canonical_sha256(request.payload()),
      "geometry_witness_sha256": request.geometry_witness_sha256,
      "frozen_policy": request.policy.payload(),
      "topology_authority_sha256": request.topology_authority.sha256,
      "topology_evidence": dict(request.topology_authority.evidence(request.program_index)),
      "candidate": candidate.payload(), "observation": dict(observation or {}),
      "child_receipts": [row.payload() for row in child_receipts],
      "ledger": dict(ledger), "accepted": bool(accepted),
      "reason_codes": sorted(set(str(row) for row in reason_codes)),
      "acceptance_semantics": "constraint_manifold_feasible_not_source_pose.v2",
      "oracle_inputs_absent": True,
  }
  _v1.assert_no_source_pose_oracle_v1(payload)
  payload["certificate_payload_sha256"] = _canonical_sha256(payload)
  return MappingProxyType(payload)


def solve_constraint_manifold_v2(
    request: ConstraintManifoldRequestV2, *, exact_executor: ExactManifoldExecutorV2,
    clock: Callable[[], float] = time.monotonic,
    query_started_at: float | None = None,
) -> ConstraintManifoldSolveResultV2:
  if type(request) is not ConstraintManifoldRequestV2:
    raise TypeError("constraint-manifold V2 requires a typed request")
  request.topology_authority.revalidate()
  start = clock() if query_started_at is None else float(query_started_at)
  deadline = start + request.budget.wall_time_seconds

  def preprocessing_timeout(
      *, candidate_count: int, elapsed: float,
  ) -> ConstraintManifoldSolveResultV2:
    ledger = {
        "generated_candidates": candidate_count,
        "candidate_budget": request.budget.max_candidates,
        "exact_evaluated_candidates": 0, "analytic_rejected_candidates": 0,
        "secondary_candidates_generated": 0,
        "reserved_occ_calls": 0, "started_occ_calls": 0,
        "started_occ_children": 0, "occ_call_budget": request.budget.max_occ_calls,
        "timeout_count": 1, "preprocessing_timeout_count": 1,
        "kernel_error_count": 0, "unreached_candidates": candidate_count,
        "elapsed_seconds": elapsed,
        "query_failure_rule": _FROZEN_POLICY_BODY_V2["failure_rule"],
        "terminal_child_receipts": (),
    }
    return ConstraintManifoldSolveResultV2(
        selected=None, certificate=None, rejection_certificate=None,
        candidate_count=candidate_count, ledger_items=tuple(ledger.items()),
        terminal_status="unknown_preprocessing_timeout",
    )

  if query_started_at is not None:
    now = clock()
    if not math.isfinite(start) or start > now:
      raise ValueError("query preprocessing start time differs")
    if now >= deadline:
      return preprocessing_timeout(candidate_count=0, elapsed=now - start)
  queue = list(sorted(generate_constraint_manifold_candidates_v2(request), key=lambda row: (
      -round(row.overlap_ratio, 12), -round(row.overlap_measure, 9),
      row.displacement_mm, row.rotation_degrees, row.candidate_id,
  )))
  if query_started_at is not None:
    now = clock()
    if now >= deadline:
      return preprocessing_timeout(candidate_count=len(queue), elapsed=now - start)
  generated_ids = {row.candidate_id for row in queue}
  evaluated: list[tuple[Any, Mapping[str, Any], tuple[str, ...], tuple[ExactChildReceiptV2, ...]]] = []
  reserved_calls = 0
  started_calls = 0
  started_children = 0
  analytic_rejects = 0
  secondary_generated = 0
  timeout_count = 0
  kernel_error_count = 0
  terminal_failure: str | None = None
  all_child_receipts: list[ExactChildReceiptV2] = []

  def run_child(candidate: Any, operation: str) -> ExactChildReceiptV2:
    nonlocal reserved_calls, started_calls, started_children, timeout_count, kernel_error_count
    now = clock()
    occ_cost = 2
    if reserved_calls + occ_cost > request.budget.max_occ_calls:
      raise RuntimeError("occ_call_budget_exhausted_before_child")
    if now >= deadline:
      raise TimeoutError("wall_deadline_exhausted_before_child")
    reserved_calls += occ_cost
    nonce = secrets.token_hex(16)
    child_deadline = min(deadline, now + request.budget.per_occ_child_timeout_seconds)
    if child_deadline <= now:
      raise TimeoutError("child_deadline_exhausted_before_start")
    started_calls += occ_cost
    started_children += 1
    receipt = exact_executor.run_child(
        candidate, operation=operation, call_nonce=nonce,
        absolute_deadline=child_deadline,
    )
    if type(receipt) is not ExactChildReceiptV2:
      raise TypeError("exact executor returned an untyped child receipt")
    if receipt.call_nonce != nonce or receipt.operation != operation:
      raise ValueError("exact child receipt nonce/operation differs")
    if receipt.terminal_status == "timeout":
      timeout_count += 1
    elif receipt.terminal_status == "kernel_error":
      kernel_error_count += 1
    return receipt

  cursor = 0
  while cursor < len(queue) and len(queue) <= request.budget.max_candidates:
    candidate = queue[cursor]
    cursor += 1
    analytic = _analytic_reasons_v2(request, candidate)
    if analytic:
      analytic_rejects += 1
      evaluated.append((candidate, {}, analytic, ()))
      continue
    receipts: list[ExactChildReceiptV2] = []
    try:
      exact = run_child(candidate, "candidate_exact_bundle")
      receipts.append(exact)
      all_child_receipts.append(exact)
      if exact.terminal_status != "ok":
        terminal_failure = f"exact_{exact.terminal_status}"
        break
    except TimeoutError:
      timeout_count += 1
      terminal_failure = "exact_timeout"
      break
    except (RuntimeError, ValueError, TypeError):
      kernel_error_count += 1
      terminal_failure = "exact_kernel_error"
      break
    observation = {
        "intended_face_clearance_mm": float(exact.result.get("distance_mm", math.inf)),
        "whole_solid_common_volume_mm3": float(
            exact.result.get("common_volume_mm3", math.inf)
        ),
        "aabb_intersection_upper_mm3": float(
            exact.result.get("aabb_intersection_upper_mm3", math.inf)
        ),
    }
    reasons: list[str] = []
    if observation["intended_face_clearance_mm"] > request.thresholds.intended_clearance_mm:
      reasons.append("intended_face_clearance")
    if observation["whole_solid_common_volume_mm3"] > request.thresholds.whole_common_volume_mm3:
      reasons.append("whole_solid_interference")
    topology = request.topology_authority.evidence(request.program_index)
    required = bool(topology.get("required_adjacent_co_constraint"))
    status = str(topology.get("adjacent_co_constraint_status"))
    if required and status != "proven":
      reasons.append("required_adjacent_co_constraint_unproven")
    if status == "contradicted":
      reasons.append("adjacent_co_constraint_contradicted")
    evaluated.append((candidate, observation, tuple(sorted(set(reasons))), tuple(receipts)))

    if (
        request.program_index == 9
        and "whole_solid_interference" in reasons
        and len(queue) < request.budget.max_candidates
    ):
      offsets = exact.result.get("projected_aabb_separation_offsets_xy_mm", ())
      if isinstance(offsets, (list, tuple)):
        rows = []
        for index, value in enumerate(offsets):
          try:
            shifted = _shift_plane_candidate_v2(
                request, candidate, value, tag=f"aabb{index:02d}"
            )
          except (TypeError, ValueError):
            shifted = None
          if shifted is not None and shifted.candidate_id not in generated_ids:
            rows.append(shifted)
        rows.sort(key=lambda row: (
            -round(row.overlap_ratio, 12), row.displacement_mm, row.candidate_id
        ))
        capacity = request.budget.max_candidates - len(queue)
        for row in rows[:capacity]:
          generated_ids.add(row.candidate_id)
          queue.append(row)
          secondary_generated += 1

  ledger = {
      "generated_candidates": len(queue), "candidate_budget": request.budget.max_candidates,
      "exact_evaluated_candidates": sum(bool(row[3]) for row in evaluated),
      "analytic_rejected_candidates": analytic_rejects,
      "secondary_candidates_generated": secondary_generated,
      "reserved_occ_calls": reserved_calls, "started_occ_calls": started_calls,
      "started_occ_children": started_children,
      "occ_call_budget": request.budget.max_occ_calls,
      "timeout_count": timeout_count, "kernel_error_count": kernel_error_count,
      "unreached_candidates": len(queue) - len(evaluated),
      "elapsed_seconds": clock() - start,
      "query_failure_rule": _FROZEN_POLICY_BODY_V2["failure_rule"],
      "terminal_child_receipts": tuple(row.payload() for row in all_child_receipts),
  }
  if terminal_failure is not None:
    rejection = None
    if evaluated:
      row = evaluated[-1]
      rejection = _certificate_v2(
          request, row[0], accepted=False,
          reason_codes=(*row[2], terminal_failure), observation=row[1],
          ledger=ledger, child_receipts=row[3],
      )
    return ConstraintManifoldSolveResultV2(
        selected=None, certificate=None, rejection_certificate=rejection,
        candidate_count=len(queue), ledger_items=tuple(ledger.items()),
        terminal_status="unknown_exact_failure",
    )

  accepted = [row for row in evaluated if not row[2] and len(row[3]) == 1]
  if not accepted:
    rejection = None
    if evaluated:
      row = min(evaluated, key=lambda item: (len(item[2]), item[0].candidate_id))
      rejection = _certificate_v2(
          request, row[0], accepted=False, reason_codes=row[2], observation=row[1],
          ledger=ledger, child_receipts=row[3],
      )
    return ConstraintManifoldSolveResultV2(
        selected=None, certificate=None, rejection_certificate=rejection,
        candidate_count=len(queue), ledger_items=tuple(ledger.items()),
        terminal_status="no_feasible_representative",
    )
  selected = min(accepted, key=lambda row: (
      -round(row[0].overlap_measure, 9),
      row[1]["aabb_intersection_upper_mm3"],
      row[1]["whole_solid_common_volume_mm3"],
      row[1]["intended_face_clearance_mm"], row[0].displacement_mm,
      row[0].candidate_id,
  ))
  certificate = _certificate_v2(
      request, selected[0], accepted=True, reason_codes=(), observation=selected[1],
      ledger=ledger, child_receipts=selected[3],
  )
  return ConstraintManifoldSolveResultV2(
      selected=selected[0], certificate=certificate, rejection_certificate=None,
      candidate_count=len(queue), ledger_items=tuple(ledger.items()),
      terminal_status="accepted",
  )


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION", "CHILD_RECEIPT_SCHEMA_VERSION",
    "OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V2", "POLICY_SCHEMA_VERSION",
    "SCHEMA_VERSION", "TOPOLOGY_SCHEMA_VERSION", "ConstraintManifoldBudgetV2",
    "ConstraintManifoldRequestV2", "ConstraintManifoldSolveResultV2",
    "ConstraintManifoldThresholdsV2", "CylinderTrimV2", "ExactChildReceiptV2",
    "ExactManifoldExecutorV2", "FrozenConstraintManifoldPolicyV2",
    "ManifoldFaceGeometryV2", "StepTopologyAuthorityV2",
    "generate_constraint_manifold_candidates_v2",
    "issue_exact_child_receipt_v2", "solve_constraint_manifold_v2",
]
