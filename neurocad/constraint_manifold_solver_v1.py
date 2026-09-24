"""Deterministic constraint-manifold representatives for programs 9 and 11.

This module deliberately solves an equivalence class, not a source pose.  Its
public request grammar has no source/gold/target transform field.  Plane and
cylinder trim geometry is intrinsic to the selected, receipt-bound faces; an
exact evaluator supplies only post-placement physical observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
from weakref import WeakKeyDictionary

import numpy as np


SCHEMA_VERSION = "constraint_manifold_solver.v1"
CERTIFICATE_SCHEMA_VERSION = "constraint_manifold_certificate.v1"
_TAU = 2.0 * math.pi
_WITNESS_TOKEN = object()
_CERTIFICATE_TOKEN = object()
_FORBIDDEN_ORACLE_TOKENS = (
    "source_pose", "source_transform", "gold", "target_transform",
    "mate_residual", "intended_transform", "pose_proximity",
)


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")).hexdigest()


def _unit(value: Sequence[float], *, label: str) -> np.ndarray:
  vector = np.asarray(value, dtype=float).reshape(3)
  length = _norm(vector)
  if not np.isfinite(vector).all() or length <= 1e-12:
    raise ValueError(f"{label} is degenerate")
  return vector / length


def _proper_rotation(value: Sequence[Sequence[float]], *, label: str) -> np.ndarray:
  rotation = np.asarray(value, dtype=float).reshape(3, 3)
  gram = _matmul(rotation.T, rotation)
  determinant = float(
      rotation[0, 0] * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
      - rotation[0, 1] * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
      + rotation[0, 2] * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
  )
  if (
      not np.isfinite(rotation).all()
      or not np.allclose(gram, np.eye(3), atol=1e-8, rtol=0.0)
      or not math.isclose(determinant, 1.0, abs_tol=1e-8)
  ):
    raise ValueError(f"{label} is not proper SO(3)")
  return rotation


def _norm(value: Sequence[float] | np.ndarray) -> float:
  vector = np.asarray(value, dtype=float).reshape(-1)
  return math.sqrt(sum(float(item) * float(item) for item in vector))


def _matmul(first: Any, second: Any) -> np.ndarray:
  left = np.asarray(first, dtype=float)
  right = np.asarray(second, dtype=float)
  if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
    raise ValueError("matrix dimensions differ")
  return np.asarray([
      [sum(float(left[row, inner]) * float(right[inner, column])
           for inner in range(left.shape[1]))
       for column in range(right.shape[1])]
      for row in range(left.shape[0])
  ], dtype=float)


def _matvec(matrix: Any, vector: Any) -> np.ndarray:
  value = np.asarray(vector, dtype=float).reshape(-1)
  rows = np.asarray(matrix, dtype=float)
  return np.asarray([
      sum(float(rows[row, column]) * float(value[column])
          for column in range(rows.shape[1]))
      for row in range(rows.shape[0])
  ], dtype=float)


def _matrix_tuple(value: np.ndarray) -> tuple[float, ...]:
  matrix = np.asarray(value, dtype=float).reshape(4, 4)
  if not np.isfinite(matrix).all() or not np.allclose(
      matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10, rtol=0.0
  ):
    raise ValueError("SE(3) matrix differs")
  _proper_rotation(matrix[:3, :3], label="SE(3) rotation")
  return tuple(float(value) for value in matrix.reshape(-1))


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
  cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
  return math.degrees(math.acos(cosine))


def _angle_degrees(first: np.ndarray, second: np.ndarray, *, unoriented: bool) -> float:
  cosine = float(np.dot(_unit(first, label="first axis"), _unit(second, label="second axis")))
  if unoriented:
    cosine = abs(cosine)
  return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def assert_no_source_pose_oracle_v1(value: Any, *, path: str = "request") -> None:
  """Fail closed if an extension tries to smuggle a pose oracle into the solver."""

  if isinstance(value, Mapping):
    for key, item in value.items():
      normalized = str(key).strip().lower()
      if any(token in normalized for token in _FORBIDDEN_ORACLE_TOKENS):
        raise ValueError(f"source-pose oracle field forbidden at {path}.{key}")
      assert_no_source_pose_oracle_v1(item, path=f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      assert_no_source_pose_oracle_v1(item, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ConstraintManifoldThresholdsV1:
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

  def __post_init__(self) -> None:
    values = tuple(asdict(self).values())
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
      raise ValueError("constraint-manifold thresholds differ")
    if not 0.0 <= self.minimum_trim_overlap_ratio <= 1.0:
      raise ValueError("trim overlap ratio differs")


@dataclass(frozen=True, slots=True)
class ConstraintManifoldBudgetV1:
  max_candidates: int = 16
  max_occ_calls: int = 32
  wall_time_seconds: float = 60.0

  def __post_init__(self) -> None:
    if (
        type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 16
        or type(self.max_occ_calls) is not int or self.max_occ_calls < 1
        or not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds <= 0.0
    ):
      raise ValueError("constraint-manifold budget differs")

  def payload(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class PlaneTrimV1:
  """Non-overlapping OCC planar tessellation triangles in face-local XY."""

  triangles_xy: tuple[tuple[tuple[float, float], ...], ...]

  def __post_init__(self) -> None:
    if not self.triangles_xy:
      raise ValueError("plane trim requires at least one triangle")
    for triangle in self.triangles_xy:
      values = np.asarray(triangle, dtype=float)
      if values.shape != (3, 2) or not np.isfinite(values).all():
        raise ValueError("plane trim triangle differs")
      if _triangle_area(values) <= 1e-12:
        raise ValueError("plane trim contains a degenerate triangle")

  @property
  def area_mm2(self) -> float:
    return float(sum(_triangle_area(np.asarray(row)) for row in self.triangles_xy))

  def payload(self) -> dict[str, Any]:
    return {"triangles_xy": [[list(point) for point in row] for row in self.triangles_xy]}


@dataclass(frozen=True, slots=True)
class CylinderTrimV1:
  axial_interval_mm: tuple[float, float]
  angular_start_radians: float
  angular_span_radians: float
  radius_mm: float
  surface_side: str
  adjacent_axial_landmarks_mm: tuple[float, ...] = ()

  def __post_init__(self) -> None:
    low, high = self.axial_interval_mm
    values = (low, high, self.angular_start_radians, self.angular_span_radians,
              self.radius_mm, *self.adjacent_axial_landmarks_mm)
    if any(not math.isfinite(float(value)) for value in values):
      raise ValueError("cylinder trim is non-finite")
    if high <= low or not 0.0 < self.angular_span_radians <= _TAU + 1e-8:
      raise ValueError("cylinder trim interval differs")
    if self.radius_mm <= 0.0 or self.surface_side not in {"exterior", "interior"}:
      raise ValueError("cylinder radius/side differs")
    if tuple(sorted(set(self.adjacent_axial_landmarks_mm))) != self.adjacent_axial_landmarks_mm:
      raise ValueError("cylinder adjacent landmarks must be sorted and unique")

  @property
  def parametric_area_mm2(self) -> float:
    return (
        (self.axial_interval_mm[1] - self.axial_interval_mm[0])
        * self.angular_span_radians * self.radius_mm
    )

  def payload(self) -> dict[str, Any]:
    return {
        "axial_interval_mm": list(self.axial_interval_mm),
        "angular_start_radians": self.angular_start_radians,
        "angular_span_radians": self.angular_span_radians,
        "radius_mm": self.radius_mm,
        "surface_side": self.surface_side,
        "adjacent_axial_landmarks_mm": list(self.adjacent_axial_landmarks_mm),
    }


@dataclass(frozen=True, slots=True)
class ManifoldFaceGeometryV1:
  part_slot: str
  graph_face_index: int
  raw_occ_face_index: int
  face_signature_sha256: str
  surface_type: str
  origin_world_mm: tuple[float, float, float]
  rotation_local_to_world: tuple[tuple[float, float, float], ...]
  plane_trim: PlaneTrimV1 | None = None
  cylinder_trim: CylinderTrimV1 | None = None

  def __post_init__(self) -> None:
    if self.part_slot not in {"a", "b"} or min(self.graph_face_index, self.raw_occ_face_index) < 0:
      raise ValueError("manifold face identity differs")
    if len(self.face_signature_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in self.face_signature_sha256
    ):
      raise ValueError("manifold face signature differs")
    origin = np.asarray(self.origin_world_mm, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
      raise ValueError("manifold face origin differs")
    _proper_rotation(self.rotation_local_to_world, label="manifold face frame")
    expected = (
        self.surface_type == "plane" and self.plane_trim is not None and self.cylinder_trim is None
    ) or (
        self.surface_type == "cylinder" and self.cylinder_trim is not None and self.plane_trim is None
    )
    if not expected:
      raise ValueError("manifold face surface/trim schema differs")

  def payload(self) -> dict[str, Any]:
    return {
        "part_slot": self.part_slot,
        "graph_face_index": self.graph_face_index,
        "raw_occ_face_index": self.raw_occ_face_index,
        "face_signature_sha256": self.face_signature_sha256,
        "surface_type": self.surface_type,
        "origin_world_mm": list(self.origin_world_mm),
        "rotation_local_to_world": [list(row) for row in self.rotation_local_to_world],
        "plane_trim": None if self.plane_trim is None else self.plane_trim.payload(),
        "cylinder_trim": None if self.cylinder_trim is None else self.cylinder_trim.payload(),
    }


@dataclass(frozen=True, slots=True)
class ConstraintManifoldRequestV1:
  query_id: str
  program_index: int
  face_a: ManifoldFaceGeometryV1
  face_b: ManifoldFaceGeometryV1
  child_world_row_major: tuple[float, ...]
  geometry_witness_sha256: str
  thresholds: ConstraintManifoldThresholdsV1 = ConstraintManifoldThresholdsV1()
  budget: ConstraintManifoldBudgetV1 = ConstraintManifoldBudgetV1()
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != SCHEMA_VERSION or not self.query_id:
      raise ValueError("constraint-manifold request identity differs")
    if self.program_index not in {9, 11}:
      raise ValueError("constraint-manifold solver authorizes only programs 9/11")
    if self.face_a.part_slot != "a" or self.face_b.part_slot != "b":
      raise ValueError("constraint-manifold face order differs")
    expected_surface = "plane" if self.program_index == 9 else "cylinder"
    if (self.face_a.surface_type, self.face_b.surface_type) != (expected_surface,) * 2:
      raise ValueError("selected face types do not match the program")
    if len(self.geometry_witness_sha256) != 64:
      raise ValueError("constraint-manifold geometry witness differs")
    _matrix_tuple(np.asarray(self.child_world_row_major).reshape(4, 4))
    assert_no_source_pose_oracle_v1(self.payload())

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version,
        "query_id": self.query_id,
        "program_index": self.program_index,
        "face_a": self.face_a.payload(),
        "face_b": self.face_b.payload(),
        "child_world_row_major": list(self.child_world_row_major),
        "geometry_witness_sha256": self.geometry_witness_sha256,
        "thresholds": asdict(self.thresholds),
        "budget": self.budget.payload(),
    }


@dataclass(frozen=True, slots=True)
class ExactManifoldObservationV1:
  intended_face_clearance_mm: float
  whole_solid_common_volume_mm3: float
  aabb_intersection_upper_mm3: float
  adjacent_co_constraint: str = "not_proven_optional"
  adjacent_gap_mm: float | None = None
  required_adjacent_co_constraint: bool = False
  occ_calls: int = 2
  terminal_status: str = "ok"

  def __post_init__(self) -> None:
    numeric = (
        self.intended_face_clearance_mm, self.whole_solid_common_volume_mm3,
        self.aabb_intersection_upper_mm3,
    )
    if any(not math.isfinite(float(value)) or value < 0.0 for value in numeric):
      raise ValueError("exact manifold observation differs")
    if self.adjacent_gap_mm is not None and (
        not math.isfinite(self.adjacent_gap_mm) or self.adjacent_gap_mm < 0.0
    ):
      raise ValueError("adjacent gap differs")
    if self.adjacent_co_constraint not in {
        "proven", "not_proven_optional", "contradicted"
    }:
      raise ValueError("adjacent co-constraint status differs")
    if type(self.occ_calls) is not int or self.occ_calls < 0:
      raise ValueError("exact observation OCC count differs")
    if self.terminal_status not in {"ok", "timeout", "kernel_error"}:
      raise ValueError("exact observation terminal status differs")

  def payload(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class ConstraintManifoldCandidateV1:
  candidate_id: str
  program_index: int
  world_delta_row_major: tuple[float, ...]
  child_world_row_major: tuple[float, ...]
  overlap_measure: float
  overlap_ratio: float
  axial_overlap_mm: float | None
  angular_overlap_radians: float | None
  constraint_metric_items: tuple[tuple[str, float], ...]
  displacement_mm: float
  rotation_degrees: float
  grammar_tags: tuple[str, ...]

  @property
  def world_delta(self) -> np.ndarray:
    return np.asarray(self.world_delta_row_major, dtype=float).reshape(4, 4).copy()

  @property
  def child_world(self) -> np.ndarray:
    return np.asarray(self.child_world_row_major, dtype=float).reshape(4, 4).copy()

  @property
  def constraint_metrics(self) -> Mapping[str, float]:
    return MappingProxyType(dict(self.constraint_metric_items))

  def payload(self) -> dict[str, Any]:
    return {
        "candidate_id": self.candidate_id, "program_index": self.program_index,
        "world_delta_row_major": list(self.world_delta_row_major),
        "child_world_row_major": list(self.child_world_row_major),
        "overlap_measure": self.overlap_measure, "overlap_ratio": self.overlap_ratio,
        "axial_overlap_mm": self.axial_overlap_mm,
        "angular_overlap_radians": self.angular_overlap_radians,
        "constraint_metrics": dict(self.constraint_metric_items),
        "displacement_mm": self.displacement_mm,
        "rotation_degrees": self.rotation_degrees,
        "grammar_tags": list(self.grammar_tags),
    }


class GeometryWitnessAuthorityV1:
  __slots__ = ("_sha256", "_revalidate", "_factory_token")

  def __init__(self, sha256: str, revalidate: Callable[[], str], *, _factory_token: object) -> None:
    if _factory_token is not _WITNESS_TOKEN or len(sha256) != 64 or not callable(revalidate):
      raise TypeError("geometry witness authorities are binder-only")
    object.__setattr__(self, "_sha256", sha256)
    object.__setattr__(self, "_revalidate", revalidate)
    object.__setattr__(self, "_factory_token", _factory_token)
    self.revalidate()

  @property
  def sha256(self) -> str:
    return self._sha256

  def revalidate(self) -> None:
    if self._revalidate() != self._sha256:
      raise ValueError("constraint-manifold geometry witness changed after authentication")

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("geometry witness authority is immutable")


def bind_geometry_witness_v1(
    sha256: str, *, revalidate: Callable[[], str],
) -> GeometryWitnessAuthorityV1:
  return GeometryWitnessAuthorityV1(sha256, revalidate, _factory_token=_WITNESS_TOKEN)


_CERTIFICATE_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


class ConstraintManifoldCertificateV1:
  __slots__ = ("__weakref__",)

  def __init__(self, *, payload: Mapping[str, Any], authority: GeometryWitnessAuthorityV1,
               _factory_token: object) -> None:
    if _factory_token is not _CERTIFICATE_TOKEN or type(authority) is not GeometryWitnessAuthorityV1:
      raise TypeError("constraint-manifold certificates are solver-factory-only")
    if payload.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
      raise ValueError("constraint-manifold certificate schema differs")
    unsigned = dict(payload)
    observed = unsigned.pop("certificate_payload_sha256", None)
    if observed != _canonical_sha256(unsigned):
      raise ValueError("constraint-manifold certificate commitment differs")
    _CERTIFICATE_STATES[self] = MappingProxyType({
        "payload": _deep_freeze(dict(payload)), "authority": authority,
    })
    self.revalidate()

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("constraint-manifold certificate is immutable")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("constraint-manifold certificate is not serializable")

  def revalidate(self) -> None:
    state = _CERTIFICATE_STATES[self]
    authority = state["authority"]
    authority.revalidate()
    payload = _deep_thaw(state["payload"])
    if payload["geometry_witness_sha256"] != authority.sha256:
      raise ValueError("constraint-manifold certificate/witness binding differs")
    unsigned = dict(payload)
    observed = unsigned.pop("certificate_payload_sha256")
    if observed != _canonical_sha256(unsigned):
      raise ValueError("constraint-manifold certificate changed")

  @property
  def accepted(self) -> bool:
    self.revalidate()
    return bool(_CERTIFICATE_STATES[self]["payload"]["accepted"])

  @property
  def reason_codes(self) -> tuple[str, ...]:
    self.revalidate()
    return tuple(_CERTIFICATE_STATES[self]["payload"]["reason_codes"])

  def payload(self) -> dict[str, Any]:
    self.revalidate()
    return _deep_thaw(_CERTIFICATE_STATES[self]["payload"])


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


@dataclass(frozen=True, slots=True)
class ConstraintManifoldSolveResultV1:
  selected: ConstraintManifoldCandidateV1 | None
  certificate: ConstraintManifoldCertificateV1 | None
  candidate_count: int
  ledger_items: tuple[tuple[str, Any], ...]
  terminal_status: str
  rejection_certificate: ConstraintManifoldCertificateV1 | None = None

  @property
  def ledger(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.ledger_items))


def _triangle_signed_area(triangle: np.ndarray) -> float:
  return 0.5 * _cross2(triangle[1] - triangle[0], triangle[2] - triangle[0])


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
  return float(first[0]) * float(second[1]) - float(first[1]) * float(second[0])


def _triangle_area(triangle: np.ndarray) -> float:
  return abs(_triangle_signed_area(triangle))


def _line_intersection(first: np.ndarray, second: np.ndarray,
                       edge_a: np.ndarray, edge_b: np.ndarray) -> np.ndarray:
  direction = second - first
  edge = edge_b - edge_a
  denominator = _cross2(direction, edge)
  if abs(denominator) <= 1e-15:
    return (first + second) * 0.5
  parameter = _cross2(edge_a - first, edge) / denominator
  return first + parameter * direction


def _convex_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
  if _triangle_signed_area(clip) < 0.0:
    clip = clip[::-1]
  output = [np.asarray(point, dtype=float) for point in subject]
  for index in range(len(clip)):
    edge_a, edge_b = clip[index], clip[(index + 1) % len(clip)]
    previous = output
    output = []
    if not previous:
      break
    def inside(point: np.ndarray) -> bool:
      return _cross2(edge_b - edge_a, point - edge_a) >= -1e-10
    start = previous[-1]
    for end in previous:
      if inside(end):
        if not inside(start):
          output.append(_line_intersection(start, end, edge_a, edge_b))
        output.append(end)
      elif inside(start):
        output.append(_line_intersection(start, end, edge_a, edge_b))
      start = end
  return np.asarray(output, dtype=float)


def _polygon_area(polygon: np.ndarray) -> float:
  if polygon.shape[0] < 3:
    return 0.0
  return 0.5 * abs(sum(
      _cross2(polygon[(index + 1) % len(polygon)], polygon[index])
      for index in range(len(polygon))
  ))


def _plane_overlap_area(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray],
) -> float:
  return float(sum(
      _polygon_area(_convex_clip(np.asarray(left), np.asarray(right)))
      for left in first for right in second
  ))


def _triangle_centroid(triangles: Sequence[np.ndarray]) -> np.ndarray:
  weighted = np.zeros(2)
  area = 0.0
  for triangle in triangles:
    value = _triangle_area(triangle)
    weighted += value * np.mean(triangle, axis=0)
    area += value
  if area <= 1e-12:
    raise ValueError("trim area is degenerate")
  return weighted / area


def _dominant_trim_angle(triangles: Sequence[np.ndarray]) -> float:
  rows: list[tuple[float, float, float]] = []
  for triangle in triangles:
    for index in range(3):
      delta = triangle[(index + 1) % 3] - triangle[index]
      length = _norm(delta)
      if length > 1e-10:
        angle = math.atan2(float(delta[1]), float(delta[0])) % math.pi
        rows.append((-length, round(angle, 12), angle))
  if not rows:
    raise ValueError("trim has no intrinsic direction")
  return min(rows)[2]


def _transformed_plane_triangles(
    face: ManifoldFaceGeometryV1, world_delta: np.ndarray,
    reference: ManifoldFaceGeometryV1,
) -> tuple[np.ndarray, ...]:
  assert face.plane_trim is not None
  origin = np.asarray(face.origin_world_mm)
  rotation = np.asarray(face.rotation_local_to_world)
  ref_origin = np.asarray(reference.origin_world_mm)
  ref_rotation = np.asarray(reference.rotation_local_to_world)
  result = []
  for triangle in face.plane_trim.triangles_xy:
    points = []
    for xy in triangle:
      world = origin + rotation[:, 0] * xy[0] + rotation[:, 1] * xy[1]
      moved = _matvec(world_delta[:3, :3], world) + world_delta[:3, 3]
      local = _matvec(ref_rotation.T, moved - ref_origin)
      points.append(local[:2])
    result.append(np.asarray(points))
  return tuple(result)


def _reference_plane_triangles(face: ManifoldFaceGeometryV1) -> tuple[np.ndarray, ...]:
  assert face.plane_trim is not None
  return tuple(np.asarray(row, dtype=float) for row in face.plane_trim.triangles_xy)


def _circular_segments(start: float, span: float) -> tuple[tuple[float, float], ...]:
  if span >= _TAU - 1e-10:
    return ((0.0, _TAU),)
  normalized = start % _TAU
  end = normalized + span
  if end <= _TAU:
    return ((normalized, end),)
  return ((normalized, _TAU), (0.0, end - _TAU))


def _circular_overlap(first_start: float, first_span: float,
                      second_start: float, second_span: float) -> float:
  return float(sum(
      max(0.0, min(a1, b1) - max(a0, b0))
      for a0, a1 in _circular_segments(first_start, first_span)
      for b0, b1 in _circular_segments(second_start, second_span)
  ))


def _unique_angles(values: Sequence[float]) -> tuple[float, ...]:
  result = []
  for value in values:
    normalized = value % _TAU
    if not any(abs(math.atan2(math.sin(normalized - old), math.cos(normalized - old))) <= 1e-9
               for old in result):
      result.append(normalized)
  return tuple(result)


def _candidate(
    *, candidate_id: str, program_index: int, delta: np.ndarray,
    child_world: np.ndarray, overlap_measure: float, overlap_ratio: float,
    axial_overlap: float | None, angular_overlap: float | None,
    metrics: Mapping[str, float], grammar_tags: Sequence[str],
    displacement_mm: float,
) -> ConstraintManifoldCandidateV1:
  return ConstraintManifoldCandidateV1(
      candidate_id=candidate_id, program_index=program_index,
      world_delta_row_major=_matrix_tuple(delta),
      child_world_row_major=_matrix_tuple(_matmul(delta, child_world)),
      overlap_measure=float(overlap_measure), overlap_ratio=float(overlap_ratio),
      axial_overlap_mm=None if axial_overlap is None else float(axial_overlap),
      angular_overlap_radians=(None if angular_overlap is None else float(angular_overlap)),
      constraint_metric_items=tuple(sorted((key, float(value)) for key, value in metrics.items())),
      displacement_mm=float(displacement_mm),
      rotation_degrees=_rotation_angle_degrees(delta[:3, :3]),
      grammar_tags=tuple(str(value) for value in grammar_tags),
  )


def _plane_candidates(request: ConstraintManifoldRequestV1) -> tuple[ConstraintManifoldCandidateV1, ...]:
  a, b = request.face_a, request.face_b
  assert a.plane_trim is not None and b.plane_trim is not None
  ra, rb = np.asarray(a.rotation_local_to_world), np.asarray(b.rotation_local_to_world)
  oa, ob = np.asarray(a.origin_world_mm), np.asarray(b.origin_world_mm)
  tri_a = _reference_plane_triangles(a)
  area_a, area_b = a.plane_trim.area_mm2, b.plane_trim.area_mm2
  alpha, beta = _dominant_trim_angle(tri_a), _dominant_trim_angle(_reference_plane_triangles(b))
  yaw_modes = _unique_angles((alpha + beta, alpha + beta + math.pi,
                              alpha + beta + math.pi / 2.0,
                              alpha + beta - math.pi / 2.0))
  child = np.asarray(request.child_world_row_major, dtype=float).reshape(4, 4)
  rows: list[ConstraintManifoldCandidateV1] = []
  for yaw_index, yaw in enumerate(yaw_modes):
    x = math.cos(yaw) * ra[:, 0] + math.sin(yaw) * ra[:, 1]
    y = -math.sin(yaw) * ra[:, 0] + math.cos(yaw) * ra[:, 1]
    target_rotation = np.stack((x, -y, -ra[:, 2]), axis=1)
    rotation = _matmul(target_rotation, rb.T)
    base = np.eye(4)
    base[:3, :3] = rotation
    base[:3, 3] = oa - _matvec(rotation, ob)
    tri_b_base = _transformed_plane_triangles(b, base, a)
    points_a = np.concatenate(tri_a, axis=0)
    points_b = np.concatenate(tri_b_base, axis=0)
    centroid_a, centroid_b = _triangle_centroid(tri_a), _triangle_centroid(tri_b_base)
    shifts = (
        ("centroid", centroid_a - centroid_b),
        ("lower_extrema", np.min(points_a, axis=0) - np.min(points_b, axis=0)),
        ("upper_extrema", np.max(points_a, axis=0) - np.max(points_b, axis=0)),
        ("bbox_center", (np.min(points_a, axis=0) + np.max(points_a, axis=0)
                         - np.min(points_b, axis=0) - np.max(points_b, axis=0)) * 0.5),
    )
    unique_shifts: list[tuple[str, np.ndarray]] = []
    for tag, shift in shifts:
      if not any(_norm(shift - old) <= 1e-9 for _, old in unique_shifts):
        unique_shifts.append((tag, shift))
    for shift_index, (tag, shift) in enumerate(unique_shifts):
      delta = base.copy()
      delta[:3, 3] += ra[:, 0] * shift[0] + ra[:, 1] * shift[1]
      moved = _transformed_plane_triangles(b, delta, a)
      overlap = _plane_overlap_area(tri_a, moved)
      ratio = overlap / min(area_a, area_b)
      moved_origin = _matvec(rotation, ob) + delta[:3, 3]
      moved_normal = _matvec(rotation, rb[:, 2])
      metrics = {
          "normal_error_degrees": _angle_degrees(moved_normal, -ra[:, 2], unoriented=False),
          "normal_gap_mm": abs(float(np.dot(moved_origin - oa, ra[:, 2]))),
      }
      rows.append(_candidate(
          candidate_id=f"p9-y{yaw_index:02d}-t{shift_index:02d}-{tag}",
          program_index=9, delta=delta, child_world=child,
          overlap_measure=overlap, overlap_ratio=ratio,
          axial_overlap=None, angular_overlap=None, metrics=metrics,
          grammar_tags=(f"intrinsic_yaw_{yaw_index}", tag),
          displacement_mm=_norm(moved_origin - ob),
      ))
  return tuple(rows[:request.budget.max_candidates])


def _cylinder_candidates(request: ConstraintManifoldRequestV1) -> tuple[ConstraintManifoldCandidateV1, ...]:
  a, b = request.face_a, request.face_b
  assert a.cylinder_trim is not None and b.cylinder_trim is not None
  ta, tb = a.cylinder_trim, b.cylinder_trim
  ra, rb = np.asarray(a.rotation_local_to_world), np.asarray(b.rotation_local_to_world)
  oa, ob = np.asarray(a.origin_world_mm), np.asarray(b.origin_world_mm)
  child = np.asarray(request.child_world_row_major, dtype=float).reshape(4, 4)
  amid = (ta.angular_start_radians + ta.angular_span_radians * 0.5) % _TAU
  bmid = (tb.angular_start_radians + tb.angular_span_radians * 0.5) % _TAU
  rows: list[ConstraintManifoldCandidateV1] = []
  for sign in (1.0, -1.0):
    base_yaw = amid - bmid if sign > 0 else amid + bmid
    for yaw_index, yaw in enumerate(_unique_angles((base_yaw, base_yaw + math.pi))):
      x = math.cos(yaw) * ra[:, 0] + math.sin(yaw) * ra[:, 1]
      y = -math.sin(yaw) * ra[:, 0] + math.cos(yaw) * ra[:, 1]
      target_rotation = np.stack((x, y if sign > 0 else -y, sign * ra[:, 2]), axis=1)
      rotation = _matmul(target_rotation, rb.T)
      alo, ahi = ta.axial_interval_mm
      blo, bhi = tb.axial_interval_mm
      transformed_b = (blo, bhi) if sign > 0 else (-bhi, -blo)
      bcenter = sum(transformed_b) * 0.5
      offsets = (
          ("interval_center", (alo + ahi) * 0.5 - bcenter),
          ("lower_endpoint", alo - transformed_b[0]),
          ("upper_endpoint", ahi - transformed_b[1]),
          ("opposed_endpoint", alo - transformed_b[1]),
      )
      for axial_index, (tag, offset) in enumerate(offsets):
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = oa + offset * ra[:, 2] - _matvec(rotation, ob)
        moved_interval = (transformed_b[0] + offset, transformed_b[1] + offset)
        axial = max(0.0, min(ahi, moved_interval[1]) - max(alo, moved_interval[0]))
        moved_start = (
            yaw + tb.angular_start_radians if sign > 0
            else yaw - (tb.angular_start_radians + tb.angular_span_radians)
        )
        angular = _circular_overlap(
            ta.angular_start_radians, ta.angular_span_radians,
            moved_start, tb.angular_span_radians,
        )
        overlap = min(ta.radius_mm, tb.radius_mm) * axial * angular
        ratio = overlap / min(ta.parametric_area_mm2, tb.parametric_area_mm2)
        moved_origin = _matvec(rotation, ob) + delta[:3, 3]
        moved_axis = _matvec(rotation, rb[:, 2])
        radial = moved_origin - oa - np.dot(moved_origin - oa, ra[:, 2]) * ra[:, 2]
        metrics = {
            "axis_error_degrees": _angle_degrees(moved_axis, ra[:, 2], unoriented=True),
            "radial_axis_distance_mm": _norm(radial),
            "radius_difference_mm": abs(ta.radius_mm - tb.radius_mm),
        }
        rows.append(_candidate(
            candidate_id=f"p11-s{int(sign):+d}-y{yaw_index:02d}-z{axial_index:02d}-{tag}",
            program_index=11, delta=delta, child_world=child,
            overlap_measure=overlap, overlap_ratio=ratio,
            axial_overlap=axial, angular_overlap=angular, metrics=metrics,
            grammar_tags=(f"axis_sign_{int(sign):+d}", f"intrinsic_yaw_{yaw_index}", tag),
            displacement_mm=_norm(moved_origin - ob),
        ))
  return tuple(rows[:request.budget.max_candidates])


def generate_constraint_manifold_candidates_v1(
    request: ConstraintManifoldRequestV1,
) -> tuple[ConstraintManifoldCandidateV1, ...]:
  if type(request) is not ConstraintManifoldRequestV1:
    raise TypeError("constraint-manifold solver requires a typed request")
  rows = _plane_candidates(request) if request.program_index == 9 else _cylinder_candidates(request)
  unique: dict[str, ConstraintManifoldCandidateV1] = {}
  for row in rows:
    commitment = _canonical_sha256({
        "program_index": row.program_index,
        "world_delta_row_major": row.world_delta_row_major,
    })
    unique.setdefault(commitment, row)
  result = tuple(unique.values())
  if not result or len(result) > request.budget.max_candidates:
    raise ValueError("constraint-manifold candidate domain differs")
  return result


def _certificate_reasons(
    request: ConstraintManifoldRequestV1, candidate: ConstraintManifoldCandidateV1,
    observation: ExactManifoldObservationV1,
) -> tuple[str, ...]:
  thresholds = request.thresholds
  metrics = candidate.constraint_metrics
  reasons: list[str] = []
  if observation.terminal_status != "ok":
    reasons.append(f"exact_{observation.terminal_status}")
  if request.program_index == 9:
    if metrics["normal_error_degrees"] > thresholds.plane_normal_error_degrees:
      reasons.append("plane_normal_error")
    if metrics["normal_gap_mm"] > thresholds.plane_normal_gap_mm:
      reasons.append("plane_normal_gap")
    if candidate.overlap_measure < thresholds.minimum_plane_overlap_mm2:
      reasons.append("plane_zero_or_tiny_trim_overlap")
  else:
    assert request.face_a.cylinder_trim is not None and request.face_b.cylinder_trim is not None
    first, second = request.face_a.cylinder_trim, request.face_b.cylinder_trim
    sides = {first.surface_side, second.surface_side}
    if sides != {"exterior", "interior"}:
      reasons.append("shaft_bore_side_relation")
    else:
      shaft = first if first.surface_side == "exterior" else second
      bore = first if first.surface_side == "interior" else second
      clearance = bore.radius_mm - shaft.radius_mm
      if clearance < -1e-9 or clearance > thresholds.maximum_radial_clearance_mm:
        reasons.append("shaft_bore_radius_relation")
    if metrics["axis_error_degrees"] > thresholds.cylinder_axis_error_degrees:
      reasons.append("cylinder_axis_error")
    if metrics["radial_axis_distance_mm"] > thresholds.cylinder_axis_distance_mm:
      reasons.append("cylinder_axis_distance")
    if (candidate.axial_overlap_mm or 0.0) < thresholds.minimum_cylinder_axial_overlap_mm:
      reasons.append("cylinder_zero_axial_overlap")
    if (candidate.angular_overlap_radians or 0.0) < thresholds.minimum_cylinder_angular_overlap_radians:
      reasons.append("cylinder_zero_angular_overlap")
  if candidate.overlap_ratio < thresholds.minimum_trim_overlap_ratio:
    reasons.append("trim_overlap_ratio")
  if observation.intended_face_clearance_mm > thresholds.intended_clearance_mm:
    reasons.append("intended_face_clearance")
  if observation.whole_solid_common_volume_mm3 > thresholds.whole_common_volume_mm3:
    reasons.append("whole_solid_interference")
  if observation.adjacent_co_constraint == "contradicted":
    reasons.append("adjacent_co_constraint_contradicted")
  if observation.required_adjacent_co_constraint and observation.adjacent_co_constraint != "proven":
    reasons.append("required_adjacent_co_constraint_unproven")
  if (
      observation.adjacent_co_constraint == "proven"
      and observation.adjacent_gap_mm is not None
      and observation.adjacent_gap_mm > thresholds.adjacent_gap_mm
  ):
    reasons.append("adjacent_co_constraint_gap")
  return tuple(sorted(set(reasons)))


def _issue_certificate(
    request: ConstraintManifoldRequestV1, candidate: ConstraintManifoldCandidateV1,
    observation: ExactManifoldObservationV1, authority: GeometryWitnessAuthorityV1,
    *, ledger: Mapping[str, Any],
) -> ConstraintManifoldCertificateV1:
  authority.revalidate()
  if authority.sha256 != request.geometry_witness_sha256:
    raise ValueError("solver request geometry witness differs")
  reasons = _certificate_reasons(request, candidate, observation)
  payload = {
      "schema_version": CERTIFICATE_SCHEMA_VERSION,
      "query_id": request.query_id, "program_index": request.program_index,
      "request_sha256": _canonical_sha256(request.payload()),
      "geometry_witness_sha256": request.geometry_witness_sha256,
      "face_a": request.face_a.payload(), "face_b": request.face_b.payload(),
      "candidate": candidate.payload(), "observation": observation.payload(),
      "thresholds": asdict(request.thresholds), "budget": request.budget.payload(),
      "ledger": dict(ledger), "accepted": not reasons,
      "reason_codes": list(reasons),
      "acceptance_semantics": "constraint_manifold_feasible_representative_not_source_pose",
      "oracle_inputs_absent": True,
  }
  assert_no_source_pose_oracle_v1(payload)
  payload["certificate_payload_sha256"] = _canonical_sha256(payload)
  return ConstraintManifoldCertificateV1(
      payload=payload, authority=authority, _factory_token=_CERTIFICATE_TOKEN
  )


def solve_constraint_manifold_v1(
    request: ConstraintManifoldRequestV1,
    *, exact_evaluator: Callable[[ConstraintManifoldCandidateV1], ExactManifoldObservationV1],
    geometry_authority: GeometryWitnessAuthorityV1,
) -> ConstraintManifoldSolveResultV1:
  """Enumerate and physically certify a bounded set of feasible representatives."""

  if type(request) is not ConstraintManifoldRequestV1:
    raise TypeError("constraint-manifold solver requires a typed request")
  if type(geometry_authority) is not GeometryWitnessAuthorityV1:
    raise TypeError("constraint-manifold solver requires an authenticated geometry witness")
  geometry_authority.revalidate()
  start = time.monotonic()
  candidates = generate_constraint_manifold_candidates_v1(request)
  ordered = sorted(candidates, key=lambda row: (
      -round(row.overlap_ratio, 12), -round(row.overlap_measure, 9), row.displacement_mm,
      row.rotation_degrees, row.candidate_id,
  ))
  evaluated: list[tuple[ConstraintManifoldCandidateV1, ExactManifoldObservationV1,
                        ConstraintManifoldCertificateV1]] = []
  occ_calls = 0
  timeout_count = 0
  kernel_error_count = 0
  rejection_reason_counts: dict[str, int] = {}
  for candidate in ordered:
    if time.monotonic() - start >= request.budget.wall_time_seconds:
      timeout_count += len(ordered) - len(evaluated)
      break
    observation = exact_evaluator(candidate)
    if type(observation) is not ExactManifoldObservationV1:
      raise TypeError("exact manifold evaluator returned an untyped observation")
    if occ_calls + observation.occ_calls > request.budget.max_occ_calls:
      break
    occ_calls += observation.occ_calls
    timeout_count += observation.terminal_status == "timeout"
    kernel_error_count += observation.terminal_status == "kernel_error"
    provisional_ledger = {
        "generated_candidates": len(candidates),
        "exact_evaluated_candidates": len(evaluated) + 1,
        "occ_calls": occ_calls,
        "timeout_count": timeout_count,
        "kernel_error_count": kernel_error_count,
    }
    certificate = _issue_certificate(
        request, candidate, observation, geometry_authority,
        ledger=provisional_ledger,
    )
    evaluated.append((candidate, observation, certificate))
    for reason in certificate.reason_codes:
      rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
  ledger = {
      "generated_candidates": len(candidates),
      "candidate_budget": request.budget.max_candidates,
      "exact_evaluated_candidates": len(evaluated),
      "exact_pruned_or_unreached_candidates": len(candidates) - len(evaluated),
      "occ_calls": occ_calls, "occ_call_budget": request.budget.max_occ_calls,
      "timeout_count": timeout_count, "kernel_error_count": kernel_error_count,
      "rejection_reason_counts": tuple(sorted(rejection_reason_counts.items())),
      "elapsed_seconds": time.monotonic() - start,
      "oracle_inputs_absent": True,
  }
  accepted = [row for row in evaluated if row[2].accepted]
  if not accepted:
    status = "timeout" if timeout_count else "no_feasible_representative"
    rejection_certificate = None
    if evaluated:
      rejection_certificate = min(evaluated, key=lambda row: (
          len(row[2].reason_codes), row[1].intended_face_clearance_mm,
          row[1].whole_solid_common_volume_mm3, row[0].candidate_id,
      ))[2]
    return ConstraintManifoldSolveResultV1(
        selected=None, certificate=None, candidate_count=len(candidates),
        ledger_items=tuple(ledger.items()), terminal_status=status,
        rejection_certificate=rejection_certificate,
    )
  selected_candidate, selected_observation, _ = min(accepted, key=lambda row: (
      -round(row[0].overlap_measure, 9),
      row[1].aabb_intersection_upper_mm3,
      row[1].whole_solid_common_volume_mm3,
      row[1].intended_face_clearance_mm,
      row[0].displacement_mm,
      row[0].candidate_id,
  ))
  final_certificate = _issue_certificate(
      request, selected_candidate, selected_observation, geometry_authority,
      ledger=ledger,
  )
  return ConstraintManifoldSolveResultV1(
      selected=selected_candidate, certificate=final_certificate,
      candidate_count=len(candidates), ledger_items=tuple(ledger.items()),
      terminal_status="accepted", rejection_certificate=None,
  )


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION", "ConstraintManifoldBudgetV1",
    "ConstraintManifoldCandidateV1", "ConstraintManifoldCertificateV1",
    "ConstraintManifoldRequestV1", "ConstraintManifoldSolveResultV1",
    "ConstraintManifoldThresholdsV1", "CylinderTrimV1",
    "ExactManifoldObservationV1", "GeometryWitnessAuthorityV1",
    "ManifoldFaceGeometryV1", "PlaneTrimV1",
    "assert_no_source_pose_oracle_v1", "bind_geometry_witness_v1",
    "generate_constraint_manifold_candidates_v1", "solve_constraint_manifold_v1",
]
