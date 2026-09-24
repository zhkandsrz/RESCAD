"""Evidence-bearing constraint-manifold solver V3 for programs 9 and 11.

V3 is a new protocol: it neither accepts V1/V2 receipts nor upgrades them.
Geometry is supplied by a replayable, loader-owned authority; policy and
budgets are fixed in code; and every started exact call has a parent-observed
terminal receipt.  The solver searches only the free degrees of freedom of a
plane or cylinder contact manifold.  It never tries to reproduce a source
assembly pose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence
from weakref import WeakKeyDictionary

import numpy as np

from . import constraint_manifold_solver_v1 as _v1


SCHEMA_VERSION = "constraint_manifold_solver.v3"
CERTIFICATE_SCHEMA_VERSION = "constraint_manifold_certificate.v3"
POLICY_SCHEMA_VERSION = "constraint_manifold_frozen_policy.v3"
GEOMETRY_AUTHORITY_SCHEMA_VERSION = "constraint_manifold_geometry_authority.v3"
CHILD_RECEIPT_SCHEMA_VERSION = "constraint_manifold_exact_child_receipt.v3"
MESH_PROOF_SCHEMA_VERSION = "constraint_manifold_absolute_mesh_proof.v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TAU = 2.0 * math.pi
_GEOMETRY_TOKEN = object()
_GEOMETRY_FACTORY_CAPABILITY_V3 = object()
_CERTIFICATE_TOKEN = object()


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
      allow_nan=False,
  ).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
  if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{label} must be lowercase hexadecimal SHA-256")
  return value


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


def producer_source_sha256s_v3() -> dict[str, str]:
  root = Path(__file__).resolve().parent
  paths = (
      root / "constraint_manifold_solver_v3.py",
      root / "constraint_manifold_occ_adapter_v3.py",
      root / "tools" / "constraint_manifold_occ_worker_v3.py",
  )
  return {path.name: file_sha256(path) for path in paths}


@dataclass(frozen=True, slots=True)
class ConstraintManifoldThresholdsV3:
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


@dataclass(frozen=True, slots=True)
class ConstraintManifoldBudgetV3:
  max_candidates: int = 16
  max_occ_calls: int = 32
  wall_time_seconds: float = 60.0
  per_occ_child_timeout_seconds: float = 12.0


ABSOLUTE_CHORD_DEFLECTION_MM_V3 = 0.01
ANGULAR_DEFLECTION_RADIANS_V3 = 0.2


class FrozenConstraintManifoldPolicyV3:
  __slots__ = ()

  @property
  def thresholds(self) -> ConstraintManifoldThresholdsV3:
    return ConstraintManifoldThresholdsV3()

  @property
  def budget(self) -> ConstraintManifoldBudgetV3:
    return ConstraintManifoldBudgetV3()

  def topology_requirement(self, program_index: int) -> str:
    if program_index not in {9, 11}:
      raise ValueError("program is absent from frozen manifold policy")
    return "adjacent_shoulder_optional_unless_cross_body_gap_is_proven"

  def payload(self) -> dict[str, Any]:
    source_hashes = producer_source_sha256s_v3()
    unsigned = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "authorized_programs": {
            "9": {
                "descriptor": ["support", "seat_plane", "plane", "plane"],
                "candidate_policy": "bounded_plane_quotient_overlap_preserving.v3",
                "topology_requirement": self.topology_requirement(9),
            },
            "11": {
                "descriptor": ["insert", "shaft_in_bore", "cylinder", "cylinder"],
                "candidate_policy": "absolute_mesh_periodic_cylinder_quotient.v3",
                "topology_requirement": self.topology_requirement(11),
            },
        },
        "thresholds": asdict(self.thresholds),
        "budget": asdict(self.budget),
        "mesh_policy": {
            "schema_version": MESH_PROOF_SCHEMA_VERSION,
            "linear_deflection_mm": ABSOLUTE_CHORD_DEFLECTION_MM_V3,
            "relative": False,
            "angular_deflection_radians": ANGULAR_DEFLECTION_RADIANS_V3,
            "parallel": False,
            "error_bound": (
                "sum_per_face(4*wire_length_mm*deflection_upper_bound_mm+"
                "4*pi*wire_count*deflection_upper_bound_mm^2)"
            ),
        },
        "producer_source_sha256s": source_hashes,
        "failure_rule": (
            "preprocess_or_child_unknown_makes_whole_query_unknown_even_after_accept"
        ),
    }
    return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V3 = FrozenConstraintManifoldPolicyV3()


@dataclass(frozen=True, slots=True)
class AbsoluteMeshProofV3:
  requested_linear_deflection_mm: float
  actual_triangulation_deflection_mm: float
  deflection_upper_bound_mm: float
  relative: bool
  angular_deflection_radians: float
  parallel: bool
  node_count: int
  triangle_count: int
  occ_version: str
  schema_version: str = MESH_PROOF_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != MESH_PROOF_SCHEMA_VERSION:
      raise ValueError("absolute mesh proof schema differs")
    if self.relative or self.parallel:
      raise ValueError("mesh proof must use deterministic absolute serial meshing")
    if self.requested_linear_deflection_mm != ABSOLUTE_CHORD_DEFLECTION_MM_V3:
      raise ValueError("mesh proof linear deflection differs")
    if self.angular_deflection_radians != ANGULAR_DEFLECTION_RADIANS_V3:
      raise ValueError("mesh proof angular deflection differs")
    values = (
        self.actual_triangulation_deflection_mm, self.deflection_upper_bound_mm,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
      raise ValueError("mesh proof deflection differs")
    if self.deflection_upper_bound_mm < max(
        self.requested_linear_deflection_mm,
        self.actual_triangulation_deflection_mm,
    ):
      raise ValueError("mesh proof upper bound is not conservative")
    if self.node_count < 3 or self.triangle_count < 1 or not self.occ_version:
      raise ValueError("mesh proof triangulation identity differs")

  def payload(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class PlaneTrimV3:
  triangles_xy: tuple[tuple[tuple[float, float], ...], ...]
  exact_surface_area_mm2: float
  boundary_length_mm: float
  wire_count: int
  hole_count: int
  mesh_proof: AbsoluteMeshProofV3

  def __post_init__(self) -> None:
    if not self.triangles_xy or self.exact_surface_area_mm2 <= 0.0:
      raise ValueError("plane trim is empty")
    if self.boundary_length_mm <= 0.0 or self.wire_count < 1:
      raise ValueError("plane trim wire evidence differs")
    if not 0 <= self.hole_count < self.wire_count:
      raise ValueError("plane trim hole evidence differs")
    for row in self.triangles_xy:
      values = np.asarray(row, dtype=float)
      if values.shape != (3, 2) or not np.isfinite(values).all():
        raise ValueError("plane trim triangle differs")
      if _v1._triangle_area(values) <= 1e-12:
        raise ValueError("plane trim contains a degenerate triangle")

  @property
  def area_mm2(self) -> float:
    return float(sum(_v1._triangle_area(np.asarray(row)) for row in self.triangles_xy))

  @property
  def overlap_area_error_upper_mm2(self) -> float:
    delta = self.mesh_proof.deflection_upper_bound_mm
    band = 4.0 * self.boundary_length_mm * delta + 4.0 * math.pi * self.wire_count * delta * delta
    return max(band, abs(self.area_mm2 - self.exact_surface_area_mm2))

  def payload(self) -> dict[str, Any]:
    return {
        "triangles_xy": [[list(point) for point in row] for row in self.triangles_xy],
        "exact_surface_area_mm2": self.exact_surface_area_mm2,
        "boundary_length_mm": self.boundary_length_mm,
        "wire_count": self.wire_count, "hole_count": self.hole_count,
        "overlap_area_error_upper_mm2": self.overlap_area_error_upper_mm2,
        "mesh_proof": self.mesh_proof.payload(),
    }


@dataclass(frozen=True, slots=True)
class CylinderTrimV3:
  triangles_uz: tuple[tuple[tuple[float, float], ...], ...]
  radius_mm: float
  surface_side: str
  exact_surface_area_mm2: float
  boundary_length_mm: float
  wire_count: int
  hole_count: int
  seam_crossing_triangle_count: int
  mesh_proof: AbsoluteMeshProofV3

  def __post_init__(self) -> None:
    if not self.triangles_uz or self.radius_mm <= 0.0:
      raise ValueError("cylinder trim is empty")
    if self.surface_side not in {"interior", "exterior"}:
      raise ValueError("cylinder surface side differs")
    if self.exact_surface_area_mm2 <= 0.0 or self.boundary_length_mm <= 0.0:
      raise ValueError("cylinder exact trim evidence differs")
    if self.wire_count < 1 or not 0 <= self.hole_count < self.wire_count:
      raise ValueError("cylinder wire/hole topology differs")
    if self.seam_crossing_triangle_count < 0:
      raise ValueError("cylinder seam evidence differs")
    for row in self.triangles_uz:
      values = np.asarray(row, dtype=float)
      if values.shape != (3, 2) or not np.isfinite(values).all():
        raise ValueError("cylinder trim triangle differs")
      if _v1._triangle_area(values) <= 1e-12:
        raise ValueError("cylinder trim contains a degenerate triangle")

  @property
  def parametric_area_mm2(self) -> float:
    return self.radius_mm * sum(
        _v1._triangle_area(np.asarray(row)) for row in self.triangles_uz
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
    delta = self.mesh_proof.deflection_upper_bound_mm
    band = 4.0 * self.boundary_length_mm * delta + 4.0 * math.pi * self.wire_count * delta * delta
    return max(band, abs(self.parametric_area_mm2 - self.exact_surface_area_mm2))

  def payload(self) -> dict[str, Any]:
    return {
        "triangles_uz": [[list(point) for point in row] for row in self.triangles_uz],
        "radius_mm": self.radius_mm, "surface_side": self.surface_side,
        "exact_surface_area_mm2": self.exact_surface_area_mm2,
        "boundary_length_mm": self.boundary_length_mm,
        "wire_count": self.wire_count, "hole_count": self.hole_count,
        "seam_crossing_triangle_count": self.seam_crossing_triangle_count,
        "overlap_area_error_upper_mm2": self.overlap_area_error_upper_mm2,
        "mesh_proof": self.mesh_proof.payload(),
    }


@dataclass(frozen=True, slots=True)
class ManifoldFaceGeometryV3:
  part_slot: str
  graph_face_index: int
  raw_occ_face_index: int
  face_signature_sha256: str
  surface_type: str
  origin_world_mm: tuple[float, float, float]
  rotation_local_to_world: tuple[tuple[float, float, float], ...]
  plane_trim: PlaneTrimV3 | None = None
  cylinder_trim: CylinderTrimV3 | None = None

  def __post_init__(self) -> None:
    if self.part_slot not in {"a", "b"} or min(self.graph_face_index, self.raw_occ_face_index) < 0:
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


_GEOMETRY_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


class GeometryWitnessAuthorityV3:
  __slots__ = ("__weakref__",)

  def __init__(self, payload: Mapping[str, Any], replay: Callable[[], Mapping[str, Any]],
               *, _factory_token: object) -> None:
    if _factory_token is not _GEOMETRY_TOKEN or not callable(replay):
      raise TypeError("geometry authority is loader-factory-only")
    frozen = json.loads(json.dumps(payload))
    observed = frozen.pop("authority_payload_sha256", None)
    if (
        frozen.get("schema_version") != GEOMETRY_AUTHORITY_SCHEMA_VERSION
        or observed != canonical_sha256(frozen)
    ):
      raise ValueError("geometry authority commitment differs")
    frozen["authority_payload_sha256"] = observed
    _GEOMETRY_STATES[self] = MappingProxyType({
        "payload": _deep_freeze(frozen), "replay": replay,
    })
    self.revalidate()

  def revalidate(self) -> None:
    state = _GEOMETRY_STATES[self]
    payload = _deep_thaw(state["payload"])
    observed = payload.pop("authority_payload_sha256")
    replayed = json.loads(json.dumps(state["replay"]()))
    if replayed != payload or canonical_sha256(replayed) != observed:
      raise ValueError("geometry authority replay/tamper differs")

  @property
  def sha256(self) -> str:
    self.revalidate()
    return str(_GEOMETRY_STATES[self]["payload"]["authority_payload_sha256"])

  def payload(self) -> dict[str, Any]:
    self.revalidate()
    return _deep_thaw(_GEOMETRY_STATES[self]["payload"])

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("geometry authority is immutable")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("geometry authority is not serializable")


def _bind_geometry_authority_v3(
    payload: Mapping[str, Any], replay: Callable[[], Mapping[str, Any]],
    *, _factory_capability: object,
) -> GeometryWitnessAuthorityV3:
  if _factory_capability is not _GEOMETRY_FACTORY_CAPABILITY_V3:
    raise TypeError("geometry authority requires loader-owned capability")
  return GeometryWitnessAuthorityV3(payload, replay, _factory_token=_GEOMETRY_TOKEN)


@dataclass(frozen=True, slots=True)
class ConstraintManifoldRequestV3:
  query_id: str
  program_index: int
  face_a: ManifoldFaceGeometryV3
  face_b: ManifoldFaceGeometryV3
  child_world_row_major: tuple[float, ...]
  geometry_authority: GeometryWitnessAuthorityV3
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != SCHEMA_VERSION or not self.query_id:
      raise ValueError("constraint-manifold V3 request identity differs")
    if self.program_index not in {9, 11}:
      raise ValueError("constraint-manifold V3 authorizes only programs 9/11")
    if type(self.geometry_authority) is not GeometryWitnessAuthorityV3:
      raise TypeError("request requires loader-owned geometry authority")
    authority = self.geometry_authority.payload()
    if int(authority["program_index"]) != self.program_index:
      raise ValueError("geometry authority program differs")
    expected = "plane" if self.program_index == 9 else "cylinder"
    if (self.face_a.part_slot, self.face_b.part_slot) != ("a", "b"):
      raise ValueError("request face order differs")
    if (self.face_a.surface_type, self.face_b.surface_type) != (expected, expected):
      raise ValueError("request surfaces differ from program policy")
    if authority["face_geometry"] != [self.face_a.payload(), self.face_b.payload()]:
      raise ValueError("request geometry differs from authority replay")
    _v1._matrix_tuple(np.asarray(self.child_world_row_major).reshape(4, 4))
    _v1.assert_no_source_pose_oracle_v1(self.payload())

  @property
  def policy(self) -> FrozenConstraintManifoldPolicyV3:
    return OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V3

  @property
  def thresholds(self) -> ConstraintManifoldThresholdsV3:
    return self.policy.thresholds

  @property
  def budget(self) -> ConstraintManifoldBudgetV3:
    return self.policy.budget

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version, "query_id": self.query_id,
        "program_index": self.program_index,
        "face_a": self.face_a.payload(), "face_b": self.face_b.payload(),
        "child_world_row_major": list(self.child_world_row_major),
        "geometry_authority_sha256": self.geometry_authority.sha256,
        "frozen_policy": self.policy.payload(),
    }


@dataclass(frozen=True, slots=True)
class ExactChildReceiptV3:
  call_nonce: str
  operation: str
  terminal_status: str
  issuer: str
  child_started: bool
  result_items: tuple[tuple[str, Any], ...]
  receipt_payload_sha256: str
  schema_version: str = CHILD_RECEIPT_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CHILD_RECEIPT_SCHEMA_VERSION:
      raise ValueError("exact child receipt schema differs")
    if not self.call_nonce or self.operation != "candidate_exact_bundle":
      raise ValueError("exact child receipt identity differs")
    if self.terminal_status not in {"ok", "timeout", "kernel_error"}:
      raise ValueError("exact child terminal status differs")
    if self.issuer not in {"child_process_echo", "parent_observer"}:
      raise ValueError("exact child receipt issuer differs")
    _require_sha256(self.receipt_payload_sha256, label="child receipt")
    if self.receipt_payload_sha256 != canonical_sha256(self.unsigned_payload()):
      raise ValueError("exact child receipt commitment differs")

  @property
  def result(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.result_items))

  def unsigned_payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version, "call_nonce": self.call_nonce,
        "operation": self.operation, "terminal_status": self.terminal_status,
        "issuer": self.issuer, "child_started": self.child_started,
        "result": dict(self.result_items),
    }

  def payload(self) -> dict[str, Any]:
    return {**self.unsigned_payload(), "receipt_payload_sha256": self.receipt_payload_sha256}


def issue_exact_child_receipt_v3(
    *, call_nonce: str, operation: str, terminal_status: str,
    child_started: bool, result: Mapping[str, Any] | None = None,
    issuer: str = "child_process_echo",
) -> ExactChildReceiptV3:
  items = tuple(sorted((str(key), value) for key, value in (result or {}).items()))
  unsigned = {
      "schema_version": CHILD_RECEIPT_SCHEMA_VERSION, "call_nonce": call_nonce,
      "operation": operation, "terminal_status": terminal_status,
      "issuer": issuer, "child_started": child_started, "result": dict(items),
  }
  return ExactChildReceiptV3(
      call_nonce=call_nonce, operation=operation, terminal_status=terminal_status,
      issuer=issuer, child_started=child_started, result_items=items,
      receipt_payload_sha256=canonical_sha256(unsigned),
  )


class ExactManifoldExecutorV3(Protocol):
  def run_child(
      self, candidate: _v1.ConstraintManifoldCandidateV1, *, operation: str,
      call_nonce: str, absolute_deadline: float,
  ) -> ExactChildReceiptV3: ...


def _periodic_triangle_overlap_v3(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray],
) -> float:
  # V2's implementation was already brute-force equivalent and nextafter-safe.
  from .constraint_manifold_solver_v2 import _periodic_triangle_overlap
  return _periodic_triangle_overlap(first, second)


def _plane_overlap_v3(
    first: ManifoldFaceGeometryV3, second: ManifoldFaceGeometryV3,
    delta: np.ndarray,
) -> tuple[float, float, float]:
  assert first.plane_trim is not None and second.plane_trim is not None
  proxy_a = SimpleNamespace(
      origin_world_mm=first.origin_world_mm,
      rotation_local_to_world=first.rotation_local_to_world,
      plane_trim=first.plane_trim,
  )
  proxy_b = SimpleNamespace(
      origin_world_mm=second.origin_world_mm,
      rotation_local_to_world=second.rotation_local_to_world,
      plane_trim=second.plane_trim,
  )
  tri_a = _v1._reference_plane_triangles(proxy_a)
  tri_b = _v1._transformed_plane_triangles(proxy_b, delta, proxy_a)
  estimate = _v1._plane_overlap_area(tri_a, tri_b)
  error = (
      first.plane_trim.overlap_area_error_upper_mm2
      + second.plane_trim.overlap_area_error_upper_mm2
  )
  lower = max(0.0, estimate - error)
  ratio = lower / min(
      first.plane_trim.exact_surface_area_mm2,
      second.plane_trim.exact_surface_area_mm2,
  )
  return estimate, lower, ratio


def _plane_candidates_v3(
    request: ConstraintManifoldRequestV3,
) -> tuple[_v1.ConstraintManifoldCandidateV1, ...]:
  a, b = request.face_a, request.face_b
  assert a.plane_trim is not None and b.plane_trim is not None
  proxy = SimpleNamespace(
      face_a=a, face_b=b, child_world_row_major=request.child_world_row_major,
      budget=_v1.ConstraintManifoldBudgetV1(max_candidates=4, max_occ_calls=32,
                                            wall_time_seconds=60.0),
  )
  rows = []
  for row in _v1._plane_candidates(proxy):
    estimate, lower, ratio = _plane_overlap_v3(a, b, row.world_delta)
    metrics = dict(row.constraint_metrics)
    metrics.update({
        "trim_overlap_estimate_mm2": estimate,
        "trim_overlap_error_upper_mm2": estimate - lower,
        "trim_overlap_lower_bound_mm2": lower,
        "trim_overlap_ratio_lower_bound": ratio,
    })
    rows.append(_v1._candidate(
        candidate_id=f"v3-{row.candidate_id}", program_index=9,
        delta=row.world_delta,
        child_world=np.asarray(request.child_world_row_major).reshape(4, 4),
        overlap_measure=lower, overlap_ratio=ratio, axial_overlap=None,
        angular_overlap=None, metrics=metrics,
        grammar_tags=("absolute_mesh_trim", *row.grammar_tags),
        displacement_mm=row.displacement_mm,
    ))
  return tuple(rows)


def _cylinder_candidates_v3(
    request: ConstraintManifoldRequestV3,
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
  rows = []
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
        overlap_parameter = _periodic_triangle_overlap_v3(triangles_a, triangles_b)
        estimate = min(ta.radius_mm, tb.radius_mm) * overlap_parameter
        error = ta.overlap_area_error_upper_mm2 + tb.overlap_area_error_upper_mm2
        lower = max(0.0, estimate - error)
        ratio = lower / min(ta.exact_surface_area_mm2, tb.exact_surface_area_mm2)
        moved_origin = _v1._matvec(rotation, ob) + delta[:3, 3]
        moved_axis = _v1._matvec(rotation, rb[:, 2])
        radial = moved_origin - oa - np.dot(moved_origin - oa, ra[:, 2]) * ra[:, 2]
        axial = max(0.0, min(ahi, transformed_interval[1] + offset)
                    - max(alo, transformed_interval[0] + offset))
        rows.append(_v1._candidate(
            candidate_id=f"v3-p11-s{int(sign):+d}-y{yaw_index:02d}-z{axial_index:02d}-{tag}",
            program_index=11, delta=delta, child_world=child,
            overlap_measure=lower, overlap_ratio=ratio,
            axial_overlap=axial,
            angular_overlap=overlap_parameter / max(axial, 1e-12),
            metrics={
                "axis_error_degrees": _v1._angle_degrees(moved_axis, ra[:, 2], unoriented=True),
                "radial_axis_distance_mm": _v1._norm(radial),
                "radius_difference_mm": abs(ta.radius_mm - tb.radius_mm),
                "trim_overlap_estimate_mm2": estimate,
                "trim_overlap_error_upper_mm2": error,
                "trim_overlap_lower_bound_mm2": lower,
                "trim_overlap_ratio_lower_bound": ratio,
            },
            grammar_tags=("absolute_mesh_trim", f"axis_sign_{int(sign):+d}", tag),
            displacement_mm=_v1._norm(moved_origin - ob),
        ))
  return tuple(rows[:request.budget.max_candidates])


def generate_constraint_manifold_candidates_v3(
    request: ConstraintManifoldRequestV3,
) -> tuple[_v1.ConstraintManifoldCandidateV1, ...]:
  if type(request) is not ConstraintManifoldRequestV3:
    raise TypeError("constraint-manifold V3 requires a typed request")
  rows = _plane_candidates_v3(request) if request.program_index == 9 else _cylinder_candidates_v3(request)
  unique: dict[str, _v1.ConstraintManifoldCandidateV1] = {}
  for row in rows:
    unique.setdefault(canonical_sha256({
        "program_index": row.program_index,
        "world_delta_row_major": row.world_delta_row_major,
    }), row)
  if not unique or len(unique) > request.budget.max_candidates:
    raise ValueError("constraint-manifold V3 candidate domain differs")
  return tuple(unique.values())


def _analytic_reasons_v3(
    request: ConstraintManifoldRequestV3,
    candidate: _v1.ConstraintManifoldCandidateV1,
) -> tuple[str, ...]:
  metrics = dict(candidate.constraint_metrics)
  thresholds = request.thresholds
  reasons = []
  if request.program_index == 9:
    if metrics["normal_error_degrees"] > thresholds.plane_normal_error_degrees:
      reasons.append("plane_normal_error")
    if metrics["normal_gap_mm"] > thresholds.plane_normal_gap_mm:
      reasons.append("plane_normal_gap")
    if candidate.overlap_measure < thresholds.minimum_plane_overlap_mm2:
      reasons.append("plane_trim_overlap_lower_bound")
  else:
    assert request.face_a.cylinder_trim is not None
    assert request.face_b.cylinder_trim is not None
    first, second = request.face_a.cylinder_trim, request.face_b.cylinder_trim
    if metrics["axis_error_degrees"] > thresholds.cylinder_axis_error_degrees:
      reasons.append("cylinder_axis_error")
    if metrics["radial_axis_distance_mm"] > thresholds.cylinder_axis_distance_mm:
      reasons.append("cylinder_axis_distance")
    if {first.surface_side, second.surface_side} != {"interior", "exterior"}:
      reasons.append("shaft_bore_side_relation")
    else:
      shaft = first if first.surface_side == "exterior" else second
      bore = first if first.surface_side == "interior" else second
      clearance = bore.radius_mm - shaft.radius_mm
      if clearance < -1e-9 or clearance > thresholds.maximum_radial_clearance_mm:
        reasons.append("shaft_bore_radius_relation")
    if (candidate.axial_overlap_mm or 0.0) < thresholds.minimum_cylinder_axial_overlap_mm:
      reasons.append("cylinder_axial_overlap")
    if (candidate.angular_overlap_radians or 0.0) < thresholds.minimum_cylinder_angular_overlap_radians:
      reasons.append("cylinder_angular_overlap")
  if candidate.overlap_ratio < thresholds.minimum_trim_overlap_ratio:
    reasons.append("trim_overlap_ratio_lower_bound")
  return tuple(sorted(set(reasons)))


def _shift_plane_candidate_v3(
    request: ConstraintManifoldRequestV3,
    candidate: _v1.ConstraintManifoldCandidateV1,
    offset_xy: Sequence[float], *, tag: str,
) -> _v1.ConstraintManifoldCandidateV1 | None:
  delta = candidate.world_delta
  frame = np.asarray(request.face_a.rotation_local_to_world)
  offset = np.asarray(offset_xy, dtype=float).reshape(2)
  delta[:3, 3] += frame[:, 0] * offset[0] + frame[:, 1] * offset[1]
  estimate, lower, ratio = _plane_overlap_v3(request.face_a, request.face_b, delta)
  if (
      lower < request.thresholds.minimum_plane_overlap_mm2
      or ratio < request.thresholds.minimum_trim_overlap_ratio
  ):
    return None
  metrics = dict(candidate.constraint_metrics)
  metrics.update({
      "trim_overlap_estimate_mm2": estimate,
      "trim_overlap_error_upper_mm2": estimate - lower,
      "trim_overlap_lower_bound_mm2": lower,
      "trim_overlap_ratio_lower_bound": ratio,
  })
  return _v1._candidate(
      candidate_id=f"{candidate.candidate_id}-quotient-{tag}", program_index=9,
      delta=delta,
      child_world=np.asarray(request.child_world_row_major).reshape(4, 4),
      overlap_measure=lower, overlap_ratio=ratio,
      axial_overlap=None, angular_overlap=None, metrics=metrics,
      grammar_tags=(*candidate.grammar_tags, "overlap_preserving_collision_search", tag),
      displacement_mm=candidate.displacement_mm + _v1._norm(offset),
  )


@dataclass(frozen=True, slots=True)
class ConstraintManifoldSolveResultV3:
  selected: _v1.ConstraintManifoldCandidateV1 | None
  certificate: "ConstraintManifoldCertificateV3 | None"
  rejection_certificate: "ConstraintManifoldCertificateV3 | None"
  candidate_count: int
  ledger_items: tuple[tuple[str, Any], ...]
  terminal_status: str

  @property
  def ledger(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.ledger_items))


_CERTIFICATE_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


class ConstraintManifoldCertificateV3:
  __slots__ = ("__weakref__",)

  def __init__(self, payload: Mapping[str, Any], authority: GeometryWitnessAuthorityV3,
               *, _factory_token: object) -> None:
    if _factory_token is not _CERTIFICATE_TOKEN or type(authority) is not GeometryWitnessAuthorityV3:
      raise TypeError("constraint-manifold V3 certificate is factory-only")
    frozen = json.loads(json.dumps(payload))
    observed = frozen.pop("certificate_payload_sha256", None)
    if (
        frozen.get("schema_version") != CERTIFICATE_SCHEMA_VERSION
        or observed != canonical_sha256(frozen)
    ):
      raise ValueError("constraint-manifold V3 certificate commitment differs")
    frozen["certificate_payload_sha256"] = observed
    _CERTIFICATE_STATES[self] = MappingProxyType({
        "payload": _deep_freeze(frozen), "authority": authority,
    })
    self.revalidate()

  def revalidate(self) -> None:
    state = _CERTIFICATE_STATES[self]
    authority = state["authority"]
    authority.revalidate()
    payload = _deep_thaw(state["payload"])
    observed = payload.pop("certificate_payload_sha256")
    if payload["geometry_authority_sha256"] != authority.sha256:
      raise ValueError("certificate geometry authority binding differs")
    if payload["frozen_policy"] != OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V3.payload():
      raise ValueError("certificate executable policy bytes changed")
    if observed != canonical_sha256(payload):
      raise ValueError("constraint-manifold V3 certificate changed")

  @property
  def accepted(self) -> bool:
    self.revalidate()
    return bool(_CERTIFICATE_STATES[self]["payload"]["accepted"])

  def payload(self) -> dict[str, Any]:
    self.revalidate()
    return _deep_thaw(_CERTIFICATE_STATES[self]["payload"])

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("constraint-manifold V3 certificate is immutable")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("constraint-manifold V3 certificate is not serializable")


def _certificate_v3(
    request: ConstraintManifoldRequestV3,
    candidate: _v1.ConstraintManifoldCandidateV1,
    *, accepted: bool, reason_codes: Sequence[str], observation: Mapping[str, Any],
    ledger: Mapping[str, Any], child_receipts: Sequence[ExactChildReceiptV3],
) -> ConstraintManifoldCertificateV3:
  request.geometry_authority.revalidate()
  unsigned = {
      "schema_version": CERTIFICATE_SCHEMA_VERSION,
      "query_id": request.query_id, "program_index": request.program_index,
      "request_sha256": canonical_sha256(request.payload()),
      "geometry_authority_sha256": request.geometry_authority.sha256,
      "frozen_policy": request.policy.payload(),
      "candidate": candidate.payload(), "observation": dict(observation),
      "child_receipts": [row.payload() for row in child_receipts],
      "ledger": dict(ledger), "accepted": accepted,
      "reason_codes": sorted(set(str(row) for row in reason_codes)),
      "acceptance_semantics": "constraint_manifold_feasible_not_source_pose.v3",
      "oracle_inputs_absent": True,
  }
  _v1.assert_no_source_pose_oracle_v1(unsigned)
  payload = {**unsigned, "certificate_payload_sha256": canonical_sha256(unsigned)}
  return ConstraintManifoldCertificateV3(
      payload, request.geometry_authority, _factory_token=_CERTIFICATE_TOKEN,
  )


def load_constraint_manifold_certificate_v3(
    certificate_path: str | Path, *, geometry_authority: GeometryWitnessAuthorityV3,
) -> ConstraintManifoldCertificateV3:
  payload = json.loads(Path(certificate_path).read_text(encoding="utf-8"))
  return ConstraintManifoldCertificateV3(
      payload, geometry_authority, _factory_token=_CERTIFICATE_TOKEN,
  )


def solve_constraint_manifold_v3(
    request: ConstraintManifoldRequestV3, *, exact_executor: ExactManifoldExecutorV3,
    clock: Callable[[], float] = time.monotonic,
    query_started_at: float | None = None,
) -> ConstraintManifoldSolveResultV3:
  if type(request) is not ConstraintManifoldRequestV3:
    raise TypeError("constraint-manifold V3 requires a typed request")
  request.geometry_authority.revalidate()
  start = clock() if query_started_at is None else float(query_started_at)
  now = clock()
  if not math.isfinite(start) or start > now:
    raise ValueError("query preprocessing start time differs")
  deadline = start + request.budget.wall_time_seconds

  def terminal(
      status: str, *, queue: Sequence[Any], evaluated: Sequence[Any],
      analytic_rejects: int, secondary_generated: int, reserved_calls: int,
      started_calls: int, receipts: Sequence[ExactChildReceiptV3],
      timeout_count: int, kernel_error_count: int,
  ) -> ConstraintManifoldSolveResultV3:
    ledger = {
        "generated_candidates": len(queue), "candidate_budget": request.budget.max_candidates,
        "exact_evaluated_candidates": sum(bool(row[3]) for row in evaluated),
        "analytic_rejected_candidates": analytic_rejects,
        "secondary_candidates_generated": secondary_generated,
        "reserved_occ_calls": reserved_calls, "started_occ_calls": started_calls,
        "started_occ_children": sum(row.child_started for row in receipts),
        "terminal_receipt_count": len(receipts),
        "unreceipted_started_children": max(0, sum(row.child_started for row in receipts) - len([
            row for row in receipts if row.child_started
        ])),
        "occ_call_budget": request.budget.max_occ_calls,
        "timeout_count": timeout_count, "kernel_error_count": kernel_error_count,
        "unreached_candidates": len(queue) - len(evaluated),
        "elapsed_seconds": clock() - start,
        "terminal_child_receipts": tuple(row.payload() for row in receipts),
        "query_failure_rule": request.policy.payload()["failure_rule"],
    }
    if status.startswith("unknown"):
      rejection = None
      if evaluated:
        row = evaluated[-1]
        rejection = _certificate_v3(
            request, row[0], accepted=False,
            reason_codes=(*row[2], status), observation=row[1], ledger=ledger,
            child_receipts=row[3],
        )
      return ConstraintManifoldSolveResultV3(
          None, None, rejection, len(queue), tuple(ledger.items()), status,
      )
    accepted_rows = [row for row in evaluated if not row[2] and len(row[3]) == 1]
    if not accepted_rows:
      rejection = None
      if evaluated:
        row = min(evaluated, key=lambda value: (len(value[2]), value[0].candidate_id))
        rejection = _certificate_v3(
            request, row[0], accepted=False, reason_codes=row[2],
            observation=row[1], ledger=ledger, child_receipts=row[3],
        )
      return ConstraintManifoldSolveResultV3(
          None, None, rejection, len(queue), tuple(ledger.items()),
          "no_feasible_representative",
      )
    selected = min(accepted_rows, key=lambda row: (
        -round(row[0].overlap_measure, 9),
        row[1]["whole_solid_common_volume_mm3"],
        row[1]["intended_face_clearance_mm"],
        row[0].displacement_mm, row[0].candidate_id,
    ))
    certificate = _certificate_v3(
        request, selected[0], accepted=True, reason_codes=(),
        observation=selected[1], ledger=ledger, child_receipts=selected[3],
    )
    return ConstraintManifoldSolveResultV3(
        selected[0], certificate, None, len(queue), tuple(ledger.items()), "accepted",
    )

  if now >= deadline:
    return terminal(
        "unknown_preprocessing_timeout", queue=(), evaluated=(), analytic_rejects=0,
        secondary_generated=0, reserved_calls=0, started_calls=0, receipts=(),
        timeout_count=1, kernel_error_count=0,
    )
  queue = list(sorted(generate_constraint_manifold_candidates_v3(request), key=lambda row: (
      -round(row.overlap_ratio, 12), -round(row.overlap_measure, 9),
      row.displacement_mm, row.rotation_degrees, row.candidate_id,
  )))
  if clock() >= deadline:
    return terminal(
        "unknown_preprocessing_timeout", queue=queue, evaluated=(), analytic_rejects=0,
        secondary_generated=0, reserved_calls=0, started_calls=0, receipts=(),
        timeout_count=1, kernel_error_count=0,
    )
  generated_ids = {row.candidate_id for row in queue}
  evaluated: list[tuple[Any, Mapping[str, Any], tuple[str, ...], tuple[ExactChildReceiptV3, ...]]] = []
  receipts: list[ExactChildReceiptV3] = []
  reserved_calls = started_calls = analytic_rejects = secondary_generated = 0
  timeout_count = kernel_error_count = 0
  failure: str | None = None
  cursor = 0
  while cursor < len(queue) and len(queue) <= request.budget.max_candidates:
    if clock() >= deadline:
      timeout_count += 1
      failure = "unknown_wall_timeout"
      break
    candidate = queue[cursor]
    cursor += 1
    reasons = _analytic_reasons_v3(request, candidate)
    if reasons:
      analytic_rejects += 1
      evaluated.append((candidate, {}, reasons, ()))
      continue
    if reserved_calls + 2 > request.budget.max_occ_calls:
      kernel_error_count += 1
      failure = "unknown_occ_budget_exhausted"
      break
    reserved_calls += 2
    nonce = secrets.token_hex(16)
    child_deadline = min(deadline, clock() + request.budget.per_occ_child_timeout_seconds)
    if child_deadline <= clock():
      timeout_count += 1
      failure = "unknown_wall_timeout"
      break
    try:
      receipt = exact_executor.run_child(
          candidate, operation="candidate_exact_bundle", call_nonce=nonce,
          absolute_deadline=child_deadline,
      )
    except BaseException as error:
      # Even a broken executor implementation is represented by a parent receipt.
      receipt = issue_exact_child_receipt_v3(
          call_nonce=nonce, operation="candidate_exact_bundle",
          terminal_status="kernel_error", child_started=False,
          result={"error": f"executor_escape:{type(error).__name__}:{error}"},
          issuer="parent_observer",
      )
    if type(receipt) is not ExactChildReceiptV3:
      receipt = issue_exact_child_receipt_v3(
          call_nonce=nonce, operation="candidate_exact_bundle",
          terminal_status="kernel_error", child_started=False,
          result={"error": "executor_returned_untyped_receipt"},
          issuer="parent_observer",
      )
    receipts.append(receipt)
    if receipt.call_nonce != nonce or receipt.operation != "candidate_exact_bundle":
      kernel_error_count += 1
      failure = "unknown_child_receipt_binding"
      break
    started_calls += 2 if receipt.child_started else 0
    if receipt.terminal_status != "ok":
      if receipt.terminal_status == "timeout":
        timeout_count += 1
      else:
        kernel_error_count += 1
      failure = f"unknown_exact_{receipt.terminal_status}"
      evaluated.append((candidate, {}, (failure,), (receipt,)))
      break
    observation = {
        "intended_face_clearance_mm": float(receipt.result.get("distance_mm", math.inf)),
        "whole_solid_common_volume_mm3": float(receipt.result.get("common_volume_mm3", math.inf)),
        "aabb_intersection_upper_mm3": float(receipt.result.get("aabb_intersection_upper_mm3", math.inf)),
    }
    post_reasons = []
    if observation["intended_face_clearance_mm"] > request.thresholds.intended_clearance_mm:
      post_reasons.append("intended_face_clearance")
    if observation["whole_solid_common_volume_mm3"] > request.thresholds.whole_common_volume_mm3:
      post_reasons.append("whole_solid_interference")
    topology = request.geometry_authority.payload()["topology_evidence"]
    if topology["requirement"] != request.policy.topology_requirement(request.program_index):
      post_reasons.append("topology_policy_binding")
    if bool(topology.get("required")) and topology.get("status") != "proven":
      post_reasons.append("required_adjacency_unproven")
    evaluated.append((candidate, observation, tuple(sorted(set(post_reasons))), (receipt,)))

    if (
        request.program_index == 9
        and "whole_solid_interference" in post_reasons
        and len(queue) < request.budget.max_candidates
    ):
      raw_offsets = receipt.result.get("projected_aabb_separation_offsets_xy_mm", ())
      proposals = []
      if isinstance(raw_offsets, (tuple, list)):
        for offset_index, raw in enumerate(raw_offsets):
          try:
            vector = np.asarray(raw, dtype=float).reshape(2)
          except (TypeError, ValueError):
            continue
          # Fixed fractions remain on the plane quotient manifold.  Every
          # proposal is independently trim-replayed below; no ordinal rules.
          for fraction in (0.25, 0.5, 0.75, 1.0):
            shifted = _shift_plane_candidate_v3(
                request, candidate, vector * fraction,
                tag=f"aabb{offset_index:02d}-f{int(fraction * 100):03d}",
            )
            if shifted is not None and shifted.candidate_id not in generated_ids:
              proposals.append(shifted)
      proposals.sort(key=lambda row: (
          -round(row.overlap_ratio, 12), row.displacement_mm, row.candidate_id,
      ))
      capacity = request.budget.max_candidates - len(queue)
      for row in proposals[:capacity]:
        generated_ids.add(row.candidate_id)
        queue.append(row)
        secondary_generated += 1

  return terminal(
      failure or "complete", queue=queue, evaluated=evaluated,
      analytic_rejects=analytic_rejects, secondary_generated=secondary_generated,
      reserved_calls=reserved_calls, started_calls=started_calls, receipts=receipts,
      timeout_count=timeout_count, kernel_error_count=kernel_error_count,
  )


__all__ = [
    "ABSOLUTE_CHORD_DEFLECTION_MM_V3", "ANGULAR_DEFLECTION_RADIANS_V3",
    "CERTIFICATE_SCHEMA_VERSION", "CHILD_RECEIPT_SCHEMA_VERSION",
    "GEOMETRY_AUTHORITY_SCHEMA_VERSION", "MESH_PROOF_SCHEMA_VERSION",
    "OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V3", "POLICY_SCHEMA_VERSION",
    "SCHEMA_VERSION", "AbsoluteMeshProofV3", "ConstraintManifoldBudgetV3",
    "ConstraintManifoldCertificateV3", "ConstraintManifoldRequestV3",
    "ConstraintManifoldSolveResultV3", "ConstraintManifoldThresholdsV3",
    "CylinderTrimV3", "ExactChildReceiptV3", "ExactManifoldExecutorV3",
    "FrozenConstraintManifoldPolicyV3", "GeometryWitnessAuthorityV3",
    "ManifoldFaceGeometryV3", "PlaneTrimV3", "canonical_sha256",
    "file_sha256", "generate_constraint_manifold_candidates_v3",
    "issue_exact_child_receipt_v3", "load_constraint_manifold_certificate_v3",
    "producer_source_sha256s_v3", "solve_constraint_manifold_v3",
]
