"""Fail-closed Fusion-native face to STEP/OCC development audit.

This module deliberately does not produce or promote formal gold.  It consumes
the private evaluation sidecar embedded in an authoritative benchmark-v2
family source and emits a private, receipt-bound development audit.  Fusion
face collection indices are treated only as keys into authoritative OBJ
``g face N`` groups; they are never used as OpenCASCADE face indices.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import heapq
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .cadquery_backend import load_step_shape, source_face_signature_sha256
from .tools.restore_fusion_assembly_dataset import (
    ARCHIVE_NAMES,
    CONTENT_RECEIPT_NAME,
    DATASET_VERSION,
    _is_reparse_point,
    find_7z,
)


LEGACY_FACE_MAP_AUDIT_SCHEMA_VERSION = "fusion_step_face_map_audit.v2"
FACE_MAP_AUDIT_SCHEMA_VERSION = "fusion_step_face_map_audit.v2_6"
ENDPOINT_LOOKUP_SCHEMA_VERSION = "fusion_step_face_endpoint_lookup.v2"
MAPPER_VERSION = "fusion_face_mapper.v2.6_analytic_trim"
ANALYTIC_TRIM_ACCEPTANCE_BASIS = "analytic_trim_equivalence_v1"
PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS = (
    "planar_chordal_trim_equivalence_v1"
)
DISTANCE_ACCEPTANCE_BASIS = "distance_equivalence_v1"
ANALYTIC_TRIM_ALGORITHM_REVISION = (
    "cylinder_rectangular_analytic_trim_mapper_integration.v1"
)
PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION = (
    "planar_closed_circle_chordal_trim_mapper_integration.v1"
)
FAMILY_SOURCE_BUILDER_VERSION = "fusion360assembly_family_source.v3"
FAMILY_SOURCE_BUILDER_PATH = "tools/build_benchmark_v2_family_source.py"
FAMILY_SOURCE_SCHEMA_VERSION = 2
GOLD_CONTACTS_SCHEMA_VERSION = "benchmark_v2_evaluation_gold_contacts.v2"
PRIVATE_SOURCE_BINDINGS_SCHEMA_VERSION = "benchmark_v2_private_source_bindings.v1"
INSTANCE_IDENTITY_SCHEMA_VERSION = "fusion_assembly_instance_identity.v1"
PART_INSTANCE_SEMANTICS = "assembly_body_instance"
_PUBLIC_INSTANCE_IDENTITY_CONTRACT = {
    "schema_version": INSTANCE_IDENTITY_SCHEMA_VERSION,
    "part_semantics": "one_anonymous_part_per_assembly_body_instance",
    "native_body_semantics": "reusable_body_local_geometry_asset",
    "occurrence_instance_key_fields": ["occurrence_uuid", "body_uuid"],
    "root_instance_key_fields": ["root_component_uuid", "body_uuid"],
    "face_map_lookup_key_fields": ["case_id", "part", "fusion_face_index"],
    "repeated_body_uuid_allowed": True,
}
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_TRIANGLE_BVH_LEAF_SIZE = 8
_FACE_AABB_BVH_LEAF_SIZE = 8
_TRIANGLE_BVH_WORK_SCHEMA_VERSION = "exact_triangle_aabb_bvh_work.v1"
_TRIANGLE_BVH_ALGORITHM = "exact_triangle_aabb_bvh_lower_bound.v1"
_FORWARD_DISTANCE_WORK_SCHEMA_VERSION = "occ_point_face_distance_call_work.v4"
_FORWARD_DISTANCE_ALGORITHM = (
    "interval_analytic_support_trim_bndbox_or_persistent_occ_bvh_forward_"
    "ranking.v4"
)
_FACE_MAP_PROGRESS_SCHEMA_VERSION = "fusion_step_face_map_progress.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FACE_GROUP_RE = re.compile(r"^face\s+([0-9]+)$")
_MODULE_ROOT = Path(__file__).resolve().parent
_EXECUTED_SOURCE_PATHS = {
    "neurocad.fusion_face_mapper": Path(__file__).resolve(),
    "neurocad.cadquery_backend": _MODULE_ROOT / "cadquery_backend.py",
    "neurocad.tools.restore_fusion_assembly_dataset": (
        _MODULE_ROOT / "tools" / "restore_fusion_assembly_dataset.py"
    ),
    "neurocad.source_obj_analytic_support_v1": (
        _MODULE_ROOT / "source_obj_analytic_support_v1.py"
    ),
    "neurocad.analytic_trim_domain_v1": (
        _MODULE_ROOT / "analytic_trim_domain_v1.py"
    ),
    "neurocad.planar_chordal_trim_domain_v1": (
        _MODULE_ROOT / "planar_chordal_trim_domain_v1.py"
    ),
}
_IMPORT_TIME_EXECUTED_SOURCE_SHA256S = {
    name: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, path in _EXECUTED_SOURCE_PATHS.items()
}


class FaceMapAuditError(RuntimeError):
  """Raised when the authoritative source contract cannot be established."""


class _CaseFailure(RuntimeError):
  def __init__(self, status: str, message: str):
    super().__init__(message)
    self.status = status


@dataclass(frozen=True)
class MappingTolerances:
  """Locked development tolerances, all geometric distances in OCC millimetres."""

  unit_relative_error: float = 0.02
  sample_distance_mm: float = 0.20
  minimum_sample_margin_mm: float = 0.05
  point_distance_mm: float = 0.20
  bbox_absolute_mm: float = 0.50
  bbox_relative_error: float = 0.005
  mesh_area_relative_error: float = 0.08
  occ_tessellation_mm: float = 0.05
  minimum_reverse_face_coverage: float = 0.99
  analytic_support_distance_mm: float = 0.01
  boundary_quantization_mm: float = 1e-5
  boundary_boolean_tolerance_mm: float = 1e-7
  boundary_union_area_relative_error: float = 1e-7
  boundary_distance_mm: float = 0.20
  boundary_sampling_step_mm: float = 0.10
  boundary_length_relative_error: float = 0.08
  maximum_boundary_edge_uses: int = 16384
  maximum_boundary_samples: int = 65536
  maximum_boundary_distance_pairs: int = 50_000_000
  # Forward/reverse proof samples are evaluated completely in deterministic
  # chunks.  The maxima are fail-closed total resource caps, not downsampling
  # targets; no sample below the cap may be skipped.
  forward_sample_chunk_size: int = 1024
  reverse_sample_chunk_size: int = 1024
  maximum_obj_samples: int = 65536
  maximum_occ_reverse_samples_per_face: int = 65536
  # OCC point-to-face calls are complete proof work, not sampling targets.
  # These fail-closed caps bound black-box forward distance work without
  # changing which samples/candidates are checked or any geometric tolerance.
  maximum_forward_distance_calls_per_endpoint: int = 100_000
  maximum_forward_distance_calls_per_case: int = 250_000
  maximum_forward_distance_calls_per_run: int = 1_000_000
  # Exact triangle evaluations are shared, fail-closed budgets.  They bound
  # pathological overlapping AABBs without changing any geometric tolerance
  # or skipping a reverse sample that can be completed within the budget.
  maximum_reverse_triangle_evaluations_per_endpoint: int = 250_000
  maximum_reverse_triangle_evaluations_per_case: int = 1_000_000
  maximum_reverse_triangle_evaluations_per_run: int = 4_000_000
  # Development proof feature, deliberately default-off so adaptive boundary /
  # curved-chord differentials remain attributable before phase-4 activation.
  # This never changes a geometric threshold or skips a face without a receipt-
  # bound zero-area analytic-support proof.
  development_enable_reverse_analytic_preexclusion: bool = False

  def validate(self) -> None:
    if not isinstance(
        self.development_enable_reverse_analytic_preexclusion,
        bool,
    ):
      raise ValueError(
          "development reverse analytic preexclusion flag must be boolean"
      )
    numeric = (
        self.unit_relative_error,
        self.sample_distance_mm,
        self.minimum_sample_margin_mm,
        self.point_distance_mm,
        self.bbox_absolute_mm,
        self.bbox_relative_error,
        self.mesh_area_relative_error,
        self.occ_tessellation_mm,
        self.minimum_reverse_face_coverage,
        self.analytic_support_distance_mm,
        self.boundary_quantization_mm,
        self.boundary_boolean_tolerance_mm,
        self.boundary_union_area_relative_error,
        self.boundary_distance_mm,
        self.boundary_sampling_step_mm,
        self.boundary_length_relative_error,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
      raise ValueError("all face-mapping tolerances must be finite and positive")
    if self.maximum_obj_samples < 8:
      raise ValueError("maximum_obj_samples must be at least 8")
    if self.maximum_occ_reverse_samples_per_face < 1:
      raise ValueError("maximum_occ_reverse_samples_per_face must be positive")
    forward_work_caps = (
        self.maximum_forward_distance_calls_per_endpoint,
        self.maximum_forward_distance_calls_per_case,
        self.maximum_forward_distance_calls_per_run,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in forward_work_caps
    ):
      raise ValueError("forward distance-call caps must be positive integers")
    if not forward_work_caps[0] <= forward_work_caps[1] <= forward_work_caps[2]:
      raise ValueError(
          "forward distance-call caps must satisfy endpoint <= case <= run"
      )
    work_caps = (
        self.maximum_reverse_triangle_evaluations_per_endpoint,
        self.maximum_reverse_triangle_evaluations_per_case,
        self.maximum_reverse_triangle_evaluations_per_run,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in work_caps
    ):
      raise ValueError("reverse triangle-evaluation caps must be positive integers")
    if not work_caps[0] <= work_caps[1] <= work_caps[2]:
      raise ValueError(
          "reverse triangle-evaluation caps must satisfy endpoint <= case <= run"
      )
    if self.forward_sample_chunk_size < 4:
      raise ValueError("forward_sample_chunk_size must be at least 4")
    if self.reverse_sample_chunk_size < 1:
      raise ValueError("reverse_sample_chunk_size must be positive")
    if self.forward_sample_chunk_size > self.maximum_obj_samples:
      raise ValueError("forward sample chunk cannot exceed the total cap")
    if (
        self.reverse_sample_chunk_size
        > self.maximum_occ_reverse_samples_per_face
    ):
      raise ValueError("reverse sample chunk cannot exceed the per-face cap")
    if self.maximum_boundary_edge_uses < 3:
      raise ValueError("maximum_boundary_edge_uses must be at least 3")
    if self.maximum_boundary_samples < 3:
      raise ValueError("maximum_boundary_samples must be at least 3")
    if self.maximum_boundary_distance_pairs < 9:
      raise ValueError("maximum_boundary_distance_pairs must be at least 9")
    if self.minimum_reverse_face_coverage > 1.0:
      raise ValueError("minimum_reverse_face_coverage cannot exceed one")


@dataclass(frozen=True)
class AnalyticTrimReplayInputs:
  """Byte-bound inputs for narrow development trim replays."""

  obj_bytes: bytes
  assembly_json_bytes: bytes
  step_bytes: bytes
  body_uuid: str
  source_face_index: int
  source_occurrence_path: tuple[str, ...]
  source_path_identity: Mapping[str, str]
  world_transform_row_major: tuple[float, ...]
  source_smt_bytes: bytes | None = None
  source_smt_archive_member: str | None = None

  def __post_init__(self) -> None:
    if any(
        not isinstance(value, bytes) or not value
        for value in (
            self.obj_bytes,
            self.assembly_json_bytes,
            self.step_bytes,
        )
    ):
      raise TypeError("analytic trim replay bytes must be non-empty bytes")
    if not isinstance(self.body_uuid, str) or not self.body_uuid:
      raise ValueError("analytic trim replay body UUID is invalid")
    if type(self.source_face_index) is not int or self.source_face_index < 0:
      raise ValueError("analytic trim replay source face index is invalid")
    if (
        not isinstance(self.source_occurrence_path, tuple)
        or any(
            not isinstance(value, str) or not value
            for value in self.source_occurrence_path
        )
    ):
      raise ValueError("analytic trim replay occurrence path is invalid")
    expected_paths = {
        "obj_archive_member",
        "step_archive_member",
        "assembly_archive_member",
    }
    if (
        not isinstance(self.source_path_identity, Mapping)
        or set(self.source_path_identity) != expected_paths
        or any(
            not isinstance(value, str) or not value
            for value in self.source_path_identity.values()
        )
    ):
      raise ValueError("analytic trim replay path identity is invalid")
    if (
        not isinstance(self.world_transform_row_major, tuple)
        or len(self.world_transform_row_major) != 16
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.world_transform_row_major
        )
    ):
      raise ValueError("analytic trim replay world transform is invalid")
    if (self.source_smt_bytes is None) != (
        self.source_smt_archive_member is None
    ):
      raise ValueError("analytic trim replay SMT bytes/path are incomplete")
    if self.source_smt_bytes is not None and (
        not isinstance(self.source_smt_bytes, bytes)
        or not self.source_smt_bytes
        or not isinstance(self.source_smt_archive_member, str)
        or not self.source_smt_archive_member
        or "\\" in self.source_smt_archive_member
    ):
      raise ValueError("analytic trim replay SMT binding is invalid")


@dataclass(frozen=True)
class _Digest:
  bytes: int
  sha1: str
  sha256: str
  captured: bytes | None = None


@dataclass(frozen=True)
class _ReceiptMember:
  path: str
  bytes: int
  sha256: str


@dataclass(frozen=True)
class _VolumeReceipt:
  archive: str
  root: Path
  receipt_sha256: str
  members: Mapping[str, _ReceiptMember]
  archive_path: Path | None
  archive_bytes: int | None
  archive_sha256: str | None


@dataclass(frozen=True)
class _ObjFaceGroup:
  fusion_face_index: int
  triangles_mm: np.ndarray
  vertices_mm: np.ndarray
  mesh_area_mm2: float
  bbox_min_mm: np.ndarray
  bbox_max_mm: np.ndarray
  signature_sha256: str


@dataclass(frozen=True)
class _OccFace:
  index: int
  face: Any
  surface_type: str
  bbox_min_mm: np.ndarray
  bbox_max_mm: np.ndarray
  area_mm2: float
  signature_sha256: str


@dataclass(frozen=True)
class _AnalyticFaceSupport:
  kind: str
  center_mm: np.ndarray
  axis: np.ndarray
  radius_mm: float
  u_min: float
  u_max: float
  v_min: float
  v_max: float
  u_period: float | None
  primitive: Any


_ANALYTIC_TRIM_GUARD_MIN_MM = 0.001
_ANALYTIC_UV_GUARD = 1e-9


def _analytic_face_support(face: _OccFace) -> _AnalyticFaceSupport | None:
  if face.surface_type not in {"PLANE", "SPHERE", "CYLINDER"}:
    return None
  try:
    from OCP.BRepTools import BRepTools

    surface = face.face._geomAdaptor()
    if face.surface_type == "SPHERE":
      primitive = surface.Sphere()
      location = primitive.Location()
      direction = primitive.Position().Direction()
      radius = float(primitive.Radius())
    elif face.surface_type == "CYLINDER":
      primitive = surface.Cylinder()
      location = primitive.Location()
      direction = primitive.Position().Direction()
      radius = float(primitive.Radius())
    else:
      primitive = surface.Pln()
      location = primitive.Location()
      direction = primitive.Axis().Direction()
      radius = 0.0
    center = np.asarray(
        [location.X(), location.Y(), location.Z()], dtype=float
    )
    axis = np.asarray(
        [direction.X(), direction.Y(), direction.Z()], dtype=float
    )
    u_min, u_max, v_min, v_max = (
        float(value) for value in BRepTools.UVBounds_s(face.face.wrapped)
    )
    u_period = float(surface.UPeriod()) if surface.IsUPeriodic() else None
  except Exception:
    return None
  values = np.asarray(
      [*center, *axis, radius, u_min, u_max, v_min, v_max], dtype=float
  )
  if (
      not np.all(np.isfinite(values))
      or (face.surface_type != "PLANE" and radius <= 0.0)
      or abs(float(np.sum(axis * axis)) - 1.0) > 1e-10
      or u_max <= u_min
      or v_max <= v_min
      or (
          u_period is not None
          and (not math.isfinite(u_period) or u_period <= 0.0)
      )
  ):
    return None
  return _AnalyticFaceSupport(
      kind=face.surface_type,
      center_mm=center,
      axis=axis,
      radius_mm=radius,
      u_min=u_min,
      u_max=u_max,
      v_min=v_min,
      v_max=v_max,
      u_period=u_period,
      primitive=primitive,
  )


def _analytic_roundoff_guard_mm(
    points_mm: np.ndarray,
    support: _AnalyticFaceSupport,
) -> float:
  scale = max(
      1.0,
      support.radius_mm,
      *(abs(float(value)) for value in support.center_mm),
      *(abs(float(value)) for value in np.asarray(points_mm).reshape(-1)),
  )
  return max(math.ulp(scale) * 64.0, math.ulp(1.0))


def _interval_subtract(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  with np.errstate(over="ignore", invalid="ignore"):
    lower = np.nextafter(left_lower - right_upper, -math.inf)
    upper = np.nextafter(left_upper - right_lower, math.inf)
  return lower, upper


def _interval_multiply(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  with np.errstate(over="ignore", invalid="ignore"):
    products = np.stack(
        (
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        ),
        axis=0,
    )
  lower = np.nextafter(np.min(products, axis=0), -math.inf)
  upper = np.nextafter(np.max(products, axis=0), math.inf)
  return lower, upper


def _interval_sum_last_axis(
    lower_terms: np.ndarray,
    upper_terms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  lower = np.zeros(lower_terms.shape[:-1], dtype=float)
  upper = np.zeros(upper_terms.shape[:-1], dtype=float)
  with np.errstate(over="ignore", invalid="ignore"):
    for position in range(lower_terms.shape[-1]):
      lower = np.nextafter(lower + lower_terms[..., position], -math.inf)
      upper = np.nextafter(upper + upper_terms[..., position], math.inf)
  return lower, upper


def _interval_norm(
    vector_lower: np.ndarray,
    vector_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  crosses_zero = (vector_lower <= 0.0) & (vector_upper >= 0.0)
  with np.errstate(over="ignore", invalid="ignore"):
    endpoint_lower_squared = np.nextafter(
        vector_lower * vector_lower,
        -math.inf,
    )
    endpoint_upper_squared = np.nextafter(
        vector_upper * vector_upper,
        -math.inf,
    )
  squared_lower = np.where(
      crosses_zero,
      0.0,
      np.minimum(endpoint_lower_squared, endpoint_upper_squared),
  )
  with np.errstate(over="ignore", invalid="ignore"):
    squared_upper = np.nextafter(
        np.maximum(vector_lower * vector_lower, vector_upper * vector_upper),
        math.inf,
    )
  norm_squared_lower, norm_squared_upper = _interval_sum_last_axis(
      np.maximum(0.0, squared_lower),
      squared_upper,
  )
  with np.errstate(over="ignore", invalid="ignore"):
    norm_lower = np.maximum(
        0.0,
        np.nextafter(np.sqrt(np.maximum(0.0, norm_squared_lower)), -math.inf),
    )
    norm_upper = np.nextafter(np.sqrt(norm_squared_upper), math.inf)
  return norm_lower, norm_upper


def _point_difference_intervals(
    points_mm: np.ndarray,
    center_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  with np.errstate(over="ignore", invalid="ignore"):
    rounded = np.asarray(points_mm, dtype=float) - center_mm
  return (
      np.nextafter(rounded, -math.inf),
      np.nextafter(rounded, math.inf),
  )


def _interval_dot_with_constant(
    vector_lower: np.ndarray,
    vector_upper: np.ndarray,
    constant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  constant_lower = np.broadcast_to(constant, vector_lower.shape)
  product_lower, product_upper = _interval_multiply(
      vector_lower,
      vector_upper,
      constant_lower,
      constant_lower,
  )
  return _interval_sum_last_axis(product_lower, product_upper)


def _interval_positive_divide(
    numerator_lower: np.ndarray,
    numerator_upper: np.ndarray,
    denominator_lower: float,
    denominator_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
  if denominator_lower <= 0.0 or not math.isfinite(denominator_upper):
    return (
        np.zeros_like(numerator_lower, dtype=float),
        np.full_like(numerator_upper, math.inf, dtype=float),
    )
  with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    lower = np.nextafter(numerator_lower / denominator_upper, -math.inf)
    upper = np.nextafter(numerator_upper / denominator_lower, math.inf)
  return np.maximum(0.0, lower), upper


def _interval_distance_to_radius_lower_bound(
    radial_lower: np.ndarray,
    radial_upper: np.ndarray,
    radius: float,
) -> np.ndarray:
  with np.errstate(over="ignore", invalid="ignore"):
    below = np.nextafter(radius - radial_upper, -math.inf)
    above = np.nextafter(radial_lower - radius, -math.inf)
  lower = np.where(
      radial_upper < radius,
      below,
      np.where(radial_lower > radius, above, 0.0),
  )
  return np.maximum(0.0, lower)


def _finite_certified_lower_bound(values: np.ndarray) -> np.ndarray:
  array = np.asarray(values, dtype=float)
  return np.where(np.isfinite(array) & (array >= 0.0), array, 0.0)


def _axis_norm_interval(axis: np.ndarray) -> tuple[float, float]:
  exact = np.asarray(axis, dtype=float).reshape(1, 3)
  lower, upper = _interval_norm(exact, exact)
  return float(lower[0]), float(upper[0])


def _analytic_support_lower_bounds(
    points_mm: np.ndarray,
    support: _AnalyticFaceSupport,
) -> np.ndarray:
  """Return certified lower bounds using outward-rounded intervals.

  Every floating-point operation that can affect the bound is enclosed on both
  sides.  This is deliberately more conservative than subtracting an empirical
  ULP guard: the result may become zero at extreme scales, but cannot be used to
  prune a face unless the interval proof itself remains separated.
  """

  points = np.asarray(points_mm, dtype=float)
  if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
    return np.zeros(len(points), dtype=float)
  difference_lower, difference_upper = _point_difference_intervals(
      points,
      support.center_mm,
  )
  if support.kind == "SPHERE":
    radial_lower, radial_upper = _interval_norm(
        difference_lower,
        difference_upper,
    )
    return _finite_certified_lower_bound(
        _interval_distance_to_radius_lower_bound(
            radial_lower,
            radial_upper,
            support.radius_mm,
        )
    )
  axis_norm_lower, axis_norm_upper = _axis_norm_interval(support.axis)
  if support.kind == "PLANE":
    signed_lower, signed_upper = _interval_dot_with_constant(
        difference_lower,
        difference_upper,
        support.axis,
    )
    absolute_lower = np.where(
        signed_lower > 0.0,
        signed_lower,
        np.where(signed_upper < 0.0, -signed_upper, 0.0),
    )
    lower, _ = _interval_positive_divide(
        np.maximum(0.0, absolute_lower),
        np.maximum(np.abs(signed_lower), np.abs(signed_upper)),
        axis_norm_lower,
        axis_norm_upper,
    )
    return _finite_certified_lower_bound(lower)
  # The radial distance to a cylinder axis is ||d x axis|| / ||axis||.
  # Computing the cross product as intervals avoids the unsafe cancellation in
  # d - dot(d, axis) * axis when coordinates are large.
  axis = support.axis
  cross_lower_terms: list[np.ndarray] = []
  cross_upper_terms: list[np.ndarray] = []
  for left_position, right_position in ((1, 2), (2, 0), (0, 1)):
    left_lower, left_upper = _interval_multiply(
        difference_lower[:, left_position],
        difference_upper[:, left_position],
        np.full(len(points), axis[right_position], dtype=float),
        np.full(len(points), axis[right_position], dtype=float),
    )
    right_lower, right_upper = _interval_multiply(
        difference_lower[:, right_position],
        difference_upper[:, right_position],
        np.full(len(points), axis[left_position], dtype=float),
        np.full(len(points), axis[left_position], dtype=float),
    )
    component_lower, component_upper = _interval_subtract(
        left_lower,
        left_upper,
        right_lower,
        right_upper,
    )
    cross_lower_terms.append(component_lower)
    cross_upper_terms.append(component_upper)
  cross_lower = np.stack(cross_lower_terms, axis=1)
  cross_upper = np.stack(cross_upper_terms, axis=1)
  cross_norm_lower, cross_norm_upper = _interval_norm(
      cross_lower,
      cross_upper,
  )
  radial_lower, radial_upper = _interval_positive_divide(
      cross_norm_lower,
      cross_norm_upper,
      axis_norm_lower,
      axis_norm_upper,
  )
  return _finite_certified_lower_bound(
      _interval_distance_to_radius_lower_bound(
          radial_lower,
          radial_upper,
          support.radius_mm,
      )
  )


def _normalize_periodic_parameter(
    value: float,
    minimum: float,
    maximum: float,
    period: float | None,
) -> float | None:
  result = float(value)
  if period is not None:
    while result < minimum:
      result += period
    while result > maximum:
      result -= period
  if result < minimum or result > maximum:
    return None
  return result


def _trimmed_analytic_distance_upper_bound(
    point_mm: np.ndarray,
    face: _OccFace,
    support: _AnalyticFaceSupport,
    classifier: Any | None = None,
) -> tuple[float, float] | None:
  try:
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.ElSLib import ElSLib
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Pnt, gp_Pnt2d

    point = np.asarray(point_mm, dtype=float)
    difference = point - support.center_mm
    radial = float(np.sqrt(np.sum(difference**2)))
    guard = max(
        _ANALYTIC_TRIM_GUARD_MIN_MM,
        _analytic_roundoff_guard_mm(point, support),
    )
    if not math.isfinite(radial) or (
        support.kind == "SPHERE" and radial <= guard
    ):
      return None
    if support.kind == "CYLINDER":
      axial = float(np.sum(difference * support.axis))
      radial_vector = difference - axial * support.axis
      cylinder_radial = float(np.sqrt(np.sum(radial_vector**2)))
      if not math.isfinite(cylinder_radial) or cylinder_radial <= guard:
        return None
      projected = (
          support.center_mm
          + axial * support.axis
          + support.radius_mm * radial_vector / cylinder_radial
      )
      raw_distance = abs(cylinder_radial - support.radius_mm)
    elif support.kind == "SPHERE":
      projected = (
          support.center_mm + support.radius_mm * difference / radial
      )
      raw_distance = abs(radial - support.radius_mm)
    else:
      axial = float(np.sum(difference * support.axis))
      projected = point - axial * support.axis
      raw_distance = abs(axial)
    projected_point = gp_Pnt(*(float(value) for value in projected))
    u_raw, v_raw = (
        float(value)
        for value in ElSLib.Parameters_s(support.primitive, projected_point)
    )
    u_value = _normalize_periodic_parameter(
        u_raw,
        support.u_min,
        support.u_max,
        support.u_period,
    )
    if u_value is None or not support.v_min <= v_raw <= support.v_max:
      return None
    if (
        min(u_value - support.u_min, support.u_max - u_value)
        <= _ANALYTIC_UV_GUARD
        or min(v_raw - support.v_min, support.v_max - v_raw)
        <= _ANALYTIC_UV_GUARD
        or (
            support.kind == "SPHERE"
            and abs(math.cos(v_raw)) * support.radius_mm <= guard
        )
    ):
      return None
    active_classifier = classifier or BRepClass_FaceClassifier()
    active_classifier.Perform(
        face.face.wrapped,
        gp_Pnt2d(u_value, v_raw),
        _ANALYTIC_UV_GUARD,
        True,
    )
    if active_classifier.State() != TopAbs_IN:
      return None
    if not math.isfinite(raw_distance) or raw_distance < 0.0:
      return None
    upper = math.nextafter(raw_distance + guard, math.inf)
    return upper, guard
  except Exception:
    return None


def _curved_triangle_chord_error_proof(
    triangle_mm: np.ndarray,
    sample_mm: np.ndarray,
    face: _OccFace,
    *,
    locked_distance_mm: float,
) -> dict[str, Any] | None:
  """Fail closed until a <= locked-distance curved-chord proof exists.

  The retired development proof bounded support distance by a triangle edge
  length.  That upper bound could be hundreds of millimetres and therefore did
  not establish the locked 0.20 mm coverage predicate.  Keeping this hook
  explicitly disabled prevents an exact OCC miss from being overridden while
  preserving the receipt shape and a narrow place for a future tight proof.
  """

  del triangle_mm, sample_mm, face, locked_distance_mm
  return None


class _PersistentPointFaceDistanceKernel:
  def __init__(self, face: _OccFace) -> None:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    self._extrema = BRepExtrema_DistShapeShape()
    self._extrema.LoadS1(face.face.wrapped)

  def distance(self, point_mm: np.ndarray) -> float:
    try:
      from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
      from OCP.gp import gp_Pnt

      maker = BRepBuilderAPI_MakeVertex(
          gp_Pnt(*(float(value) for value in point_mm))
      )
      vertex = maker.Vertex()
      self._extrema.LoadS2(vertex)
      self._extrema.Perform()
      if not self._extrema.IsDone():
        raise RuntimeError("persistent point-face extrema did not complete")
      distance = float(self._extrema.Value())
    except Exception as error:
      raise _CaseFailure(
          "occ_distance_failed",
          "persistent OCC point-face distance failed",
      ) from error
    if not math.isfinite(distance) or distance < 0.0:
      raise _CaseFailure(
          "occ_distance_failed",
          "persistent OCC returned an invalid point-face distance",
      )
    return distance


@dataclass(frozen=True)
class _BoundaryProfile:
  edge_count: int
  loop_count: int
  length_mm: float
  polylines_mm: tuple[np.ndarray, ...]
  samples_mm: np.ndarray
  boundary_shape: Any
  maximum_sample_gap_mm: float
  sample_step_mm: float
  refine_samples: Callable[[float], tuple[np.ndarray, float]]
  normalization_checks: Mapping[str, Any]


@dataclass(frozen=True)
class _PartInstanceBinding:
  part: str
  geometry_asset: str
  body_uuid: str
  source_instance_key: Mapping[str, str]
  occurrence_path: tuple[str, ...]
  is_visible: bool
  is_grounded: bool
  step_binding: Mapping[str, Any]


def _canonical_json(value: Any) -> str:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  )


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _emit_face_map_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    event: str,
    case_id: str,
    endpoint_face_index: int | None = None,
    endpoint_part: str | None = None,
    processed_sample_count: int = 0,
    total_sample_count: int = 0,
    exact_occ_distance_call_count: int = 0,
    pruned_occ_face_candidate_count: int = 0,
    elapsed_seconds: float = 0.0,
    bvh_max_depth: int = 0,
    bvh_max_traversal_depth_visited: int = 0,
    status: str | None = None,
    selected_case_ordinal: int | None = None,
    selected_case_count: int | None = None,
) -> None:
  """Emit one path-free, instance-private progress event when requested."""

  if callback is None:
    return
  payload: dict[str, Any] = {
      "schema_version": _FACE_MAP_PROGRESS_SCHEMA_VERSION,
      "event": str(event),
      "case_id": str(case_id),
      "endpoint_face_index": endpoint_face_index,
      "endpoint_part": endpoint_part,
      "processed_sample_count": int(processed_sample_count),
      "total_sample_count": int(total_sample_count),
      "exact_occ_distance_call_count": int(exact_occ_distance_call_count),
      "pruned_occ_face_candidate_count": int(pruned_occ_face_candidate_count),
      "elapsed_seconds": max(0.0, float(elapsed_seconds)),
      "bvh_max_depth": int(bvh_max_depth),
      "bvh_max_traversal_depth_visited": int(
          bvh_max_traversal_depth_visited
      ),
  }
  if status is not None:
    payload["status"] = str(status)
  if selected_case_ordinal is not None:
    payload["selected_case_ordinal"] = int(selected_case_ordinal)
  if selected_case_count is not None:
    payload["selected_case_count"] = int(selected_case_count)
  try:
    callback(payload)
  except Exception as error:
    raise FaceMapAuditError(
        f"face-map progress callback failed: {type(error).__name__}:{error}"
    ) from error


def _plain_file_digest(path: Path, *, capture: bool = False) -> _Digest:
  if _is_reparse_point(path) or not path.is_file():
    raise _CaseFailure("unsafe_input_path", f"input is not a plain file: {path}")
  before = path.stat(follow_symlinks=False)
  sha1 = hashlib.sha1()
  sha256 = hashlib.sha256()
  captured = bytearray() if capture else None
  with path.open("rb") as stream:
    while chunk := stream.read(_HASH_CHUNK_BYTES):
      sha1.update(chunk)
      sha256.update(chunk)
      if captured is not None:
        captured.extend(chunk)
  after = path.stat(follow_symlinks=False)
  stable_fields = ("st_size", "st_mtime_ns", "st_mode", "st_ino", "st_dev")
  if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
    raise _CaseFailure("input_changed_while_hashing", f"input changed: {path}")
  return _Digest(
      bytes=int(after.st_size),
      sha1=sha1.hexdigest(),
      sha256=sha256.hexdigest(),
      captured=(bytes(captured) if captured is not None else None),
  )


def _stage_verified_digest(
    staging_root: Path,
    filename: str,
    digest: _Digest,
    *,
    status: str,
) -> Path:
  """Materialize and re-verify an immutable-by-convention parser snapshot."""

  if digest.captured is None:
    raise _CaseFailure(status, "verified parser input bytes were not captured")
  staged_path = staging_root / filename
  try:
    with staged_path.open("xb") as stream:
      stream.write(digest.captured)
      stream.flush()
      os.fsync(stream.fileno())
  except OSError as error:
    raise _CaseFailure(status, "failed to isolate verified parser input") from error
  staged_digest = _plain_file_digest(staged_path)
  _verify_digest(
      staged_digest,
      expected_bytes=digest.bytes,
      expected_sha256=digest.sha256,
      status=status,
  )
  if staged_digest.sha1 != digest.sha1:
    raise _CaseFailure(status, "isolated parser input SHA1 changed")
  return staged_path


def _implementation_source_sha256s() -> dict[str, str]:
  paths = {
      **_EXECUTED_SOURCE_PATHS,
      "neurocad.tools.audit_fusion_step_face_map": (
          _MODULE_ROOT / "tools" / "audit_fusion_step_face_map.py"
      ),
  }
  return {
      name: _plain_file_digest(path).sha256
      for name, path in sorted(paths.items())
  }


def _safe_relative_path(raw: Any, *, suffix: str | None = None) -> str:
  if not isinstance(raw, str) or not raw or "\\" in raw:
    raise _CaseFailure("unsafe_input_path", "source contains an unsafe member path")
  parsed = PurePosixPath(raw)
  if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
    raise _CaseFailure("unsafe_input_path", "source contains an unsafe member path")
  if parsed.as_posix() != raw:
    raise _CaseFailure("unsafe_input_path", "source member path is not canonical")
  if suffix is not None and parsed.suffix.lower() != suffix.lower():
    raise _CaseFailure("unsafe_input_path", f"source member is not {suffix}")
  return raw


def _resolve_member(root: Path, relative: str) -> Path:
  candidate = root.joinpath(*PurePosixPath(relative).parts)
  try:
    resolved = candidate.resolve(strict=True)
  except OSError as error:
    raise _CaseFailure("missing_input_member", f"missing dataset member: {relative}") from error
  try:
    resolved.relative_to(root)
  except ValueError as error:
    raise _CaseFailure("unsafe_input_path", "dataset member escapes volume root") from error
  return resolved


def _verify_digest(
    digest: _Digest,
    *,
    expected_bytes: Any,
    expected_sha256: Any,
    status: str,
) -> None:
  if (
      not isinstance(expected_bytes, int)
      or isinstance(expected_bytes, bool)
      or expected_bytes < 0
      or not isinstance(expected_sha256, str)
      or _SHA256_RE.fullmatch(expected_sha256) is None
      or digest.bytes != expected_bytes
      or digest.sha256 != expected_sha256
  ):
    raise _CaseFailure(status, f"receipt-bound bytes failed verification: {status}")


def _load_json_bytes(raw: bytes, *, status: str) -> Mapping[str, Any]:
  try:
    value = json.loads(raw.decode("utf-8"))
  except (UnicodeError, json.JSONDecodeError) as error:
    raise _CaseFailure(status, f"invalid JSON: {status}") from error
  if not isinstance(value, Mapping):
    raise _CaseFailure(status, f"JSON root is not an object: {status}")
  return value


def _load_family_source(path: Path) -> tuple[Mapping[str, Any], _Digest]:
  try:
    resolved = path.expanduser().resolve(strict=True)
  except OSError as error:
    raise FaceMapAuditError(f"family source is missing: {path}") from error
  try:
    digest = _plain_file_digest(resolved, capture=True)
  except _CaseFailure as error:
    raise FaceMapAuditError(str(error)) from error
  try:
    payload = _load_json_bytes(digest.captured or b"", status="family_source_invalid")
  except _CaseFailure as error:
    raise FaceMapAuditError(str(error)) from error
  return payload, digest


def _validate_builder_identity(source: Mapping[str, Any]) -> None:
  provenance = source.get("builder_provenance")
  if not isinstance(provenance, Mapping):
    raise FaceMapAuditError("family source lacks builder provenance")
  if provenance.get("producer") != "neurocad.tools.build_benchmark_v2_family_source":
    raise FaceMapAuditError("family source builder identity is not authoritative")
  if provenance.get("version") != FAMILY_SOURCE_BUILDER_VERSION:
    raise FaceMapAuditError("family source builder version is not authoritative")
  raw_path = provenance.get("path")
  if raw_path != FAMILY_SOURCE_BUILDER_PATH:
    raise FaceMapAuditError("family source builder path is not authoritative")
  try:
    relative = _safe_relative_path(raw_path, suffix=".py")
  except _CaseFailure as error:
    raise FaceMapAuditError(str(error)) from error
  project_root = Path(__file__).resolve().parent
  try:
    builder_path = _resolve_member(project_root, relative)
    digest = _plain_file_digest(builder_path)
  except _CaseFailure as error:
    raise FaceMapAuditError(f"family source builder cannot be replay-bound: {error}") from error
  expected = provenance.get("sha256")
  if expected != "sha256:" + digest.sha256:
    raise FaceMapAuditError("family source builder SHA256 does not match this checkout")


def _validate_family_source(source: Mapping[str, Any]) -> None:
  if source.get("schema_version") != FAMILY_SOURCE_SCHEMA_VERSION:
    raise FaceMapAuditError("unsupported authoritative family-source schema")
  if source.get("formal") is not False:
    raise FaceMapAuditError(
        "this mapper is development-only and refuses a formal family source"
    )
  development = source.get("development")
  if not isinstance(development, Mapping) or development.get("enabled") is not True:
    raise FaceMapAuditError("family source is not an explicit development pilot")
  selection = development.get("selection")
  if selection not in {
      "deterministic_top_after_fixed_filters",
      "target_positive_confirmation_projection",
  }:
    raise FaceMapAuditError("family source development selection is not deterministic")
  confirmation_projection = selection == "target_positive_confirmation_projection"
  if confirmation_projection:
    projection = source.get("confirmation_projection")
    if not isinstance(projection, Mapping):
      raise FaceMapAuditError("confirmation projection contract is missing")
    unsigned_projection = dict(projection)
    projection_hash = unsigned_projection.pop("projection_payload_sha256", None)
    program_counts = projection.get("program_counts")
    if (
        projection.get("schema_version")
        != "automatic_proposal_confirmation_target_projection.v1"
        or _SHA256_RE.fullmatch(
            str(projection.get("private_targets_payload_sha256") or "")
        )
        is None
        or type(projection.get("case_count")) is not int
        or int(projection["case_count"]) < 1
        or not isinstance(program_counts, Mapping)
        or set(program_counts) - {"9", "11"}
        or any(type(value) is not int or value < 0 for value in program_counts.values())
        or sum(program_counts.values()) != projection["case_count"]
        or projection.get("selection_precedes_mapper_outcomes") is not True
        or projection.get("replacement_after_outcomes_allowed") is not False
        or projection.get("training_use_allowed") is not False
        or projection.get("negative_generation_allowed") is not False
        or projection.get("source_absence_is_negative") is not False
        or projection_hash != _canonical_sha256(unsigned_projection)
    ):
      raise FaceMapAuditError("confirmation projection contract differs")
  gold = source.get("evaluation_gold_contacts")
  if not isinstance(gold, Mapping):
    raise FaceMapAuditError("family source lacks the private evaluation sidecar")
  if (
      gold.get("schema_version") != GOLD_CONTACTS_SCHEMA_VERSION
      or gold.get("visibility") != "private_evaluation_only"
      or gold.get("storage_boundary")
      != "family_source_top_level_sidecar_not_model_case"
      or gold.get("multiplicity_semantics")
      != (
          "one_precommitted_target_contact_per_confirmation_case"
          if confirmation_projection
          else "one_record_per_fusion_source_contact_instance_pair"
      )
      or gold.get("gold_interface_certificate_ready") is not False
  ):
    raise FaceMapAuditError("private evaluation sidecar has an unsafe readiness state")
  private = source.get("private_source_bindings")
  if (
      not isinstance(private, Mapping)
      or private.get("schema_version") != PRIVATE_SOURCE_BINDINGS_SCHEMA_VERSION
      or private.get("visibility") != "private_custodian_only"
      or private.get("storage_boundary")
      != "family_source_top_level_sidecar_not_model_case"
      or private.get("instance_identity_schema_version")
      != INSTANCE_IDENTITY_SCHEMA_VERSION
  ):
    raise FaceMapAuditError("family source lacks authoritative private source bindings")
  dataset = source.get("dataset_provenance")
  if not isinstance(dataset, Mapping):
    raise FaceMapAuditError("family source lacks dataset provenance")
  receipts = dataset.get("receipts")
  if (
      dataset.get("dataset_version") != DATASET_VERSION
      or dataset.get("official") is not True
      or dataset.get("archive_count") != len(ARCHIVE_NAMES)
      or dataset.get("official_receipt_count") != len(ARCHIVE_NAMES)
      or not isinstance(receipts, list)
      or len(receipts) != len(ARCHIVE_NAMES)
  ):
    raise FaceMapAuditError("family source lacks all official volume receipt bindings")
  receipt_archives = [
      row.get("archive") if isinstance(row, Mapping) else None for row in receipts
  ]
  if sorted(str(value) for value in receipt_archives) != sorted(ARCHIVE_NAMES):
    raise FaceMapAuditError("family source receipt archives are incomplete or duplicated")
  _validate_builder_identity(source)


def _volume_summary_by_archive(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
  dataset = source["dataset_provenance"]
  result: dict[str, Mapping[str, Any]] = {}
  for raw in dataset["receipts"]:
    if not isinstance(raw, Mapping):
      raise FaceMapAuditError("family source receipt summary is malformed")
    archive = str(raw.get("archive") or "")
    if archive in result:
      raise FaceMapAuditError("family source contains duplicate volume receipts")
    result[archive] = raw
  return result


def _load_volume_receipt(
    archive_root: Path,
    archive: str,
    summary: Mapping[str, Any],
) -> _VolumeReceipt:
  if archive not in ARCHIVE_NAMES:
    raise _CaseFailure("archive_binding_mismatch", "case archive is not official")
  root = (archive_root / Path(archive).stem).resolve(strict=False)
  try:
    root.relative_to(archive_root)
  except ValueError as error:
    raise _CaseFailure("unsafe_input_path", "volume root escapes archive root") from error
  if _is_reparse_point(root) or not root.is_dir():
    raise _CaseFailure("archive_receipt_mismatch", "restored volume root is missing")
  receipt_path = root / CONTENT_RECEIPT_NAME
  digest = _plain_file_digest(receipt_path, capture=True)
  expected_sha = summary.get("receipt_sha256")
  if (
      not isinstance(expected_sha, str)
      or _SHA256_RE.fullmatch(expected_sha) is None
      or digest.sha256 != expected_sha
  ):
    raise _CaseFailure(
        "archive_receipt_mismatch", "live content receipt differs from family source"
    )
  payload = _load_json_bytes(
      digest.captured or b"",
      status="archive_receipt_mismatch",
  )
  archive_payload = payload.get("archive")
  members_raw = payload.get("members")
  if (
      payload.get("dataset_version") != DATASET_VERSION
      or not isinstance(archive_payload, Mapping)
      or archive_payload.get("name") != archive
      or not isinstance(members_raw, list)
  ):
    raise _CaseFailure("archive_receipt_mismatch", "content receipt identity is invalid")
  expected_count = summary.get("member_count")
  if not isinstance(expected_count, int) or expected_count != len(members_raw):
    raise _CaseFailure("archive_receipt_mismatch", "content receipt member count changed")
  members: dict[str, _ReceiptMember] = {}
  for raw in members_raw:
    if not isinstance(raw, Mapping):
      raise _CaseFailure("archive_receipt_mismatch", "receipt member is malformed")
    relative = _safe_relative_path(raw.get("path"))
    size = raw.get("bytes")
    sha256 = raw.get("sha256")
    if (
        relative in members
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
    ):
      raise _CaseFailure("archive_receipt_mismatch", "receipt member metadata is invalid")
    members[relative] = _ReceiptMember(relative, size, sha256)
  archive_record = payload.get("archive")
  archive_path: Path | None = None
  archive_bytes: int | None = None
  archive_sha256: str | None = None
  if isinstance(archive_record, Mapping) and (
      "bytes" in archive_record or "sha256" in archive_record
  ):
    expected_archive_bytes = archive_record.get("bytes")
    expected_archive_sha256 = archive_record.get("sha256")
    if (
        archive_record.get("name") != archive
        or not isinstance(expected_archive_bytes, int)
        or isinstance(expected_archive_bytes, bool)
        or expected_archive_bytes <= 0
        or not isinstance(expected_archive_sha256, str)
        or _SHA256_RE.fullmatch(expected_archive_sha256) is None
    ):
      raise _CaseFailure("archive_receipt_mismatch", "receipt archive record is invalid")
    archive_path = archive_root / archive
    try:
      archive_digest = _plain_file_digest(archive_path)
    except _CaseFailure as error:
      raise _CaseFailure(
          "archive_receipt_mismatch", "official archive is unavailable for OBJ binding"
      ) from error
    _verify_digest(
        archive_digest,
        expected_bytes=expected_archive_bytes,
        expected_sha256=expected_archive_sha256,
        status="archive_receipt_mismatch",
    )
    archive_bytes = archive_digest.bytes
    archive_sha256 = archive_digest.sha256
  elif isinstance(archive_record, Mapping) and archive_record.get("name") != archive:
    raise _CaseFailure("archive_receipt_mismatch", "receipt archive name is invalid")
  return _VolumeReceipt(
      archive,
      root,
      digest.sha256,
      members,
      archive_path,
      archive_bytes,
      archive_sha256,
  )


def _verify_receipt_member(
    volume: _VolumeReceipt,
    relative: str,
    *,
    status: str,
    capture: bool = False,
) -> tuple[Path, _Digest]:
  member = volume.members.get(relative)
  if member is None:
    raise _CaseFailure(status, f"member is absent from receipt: {relative}")
  path = _resolve_member(volume.root, relative)
  digest = _plain_file_digest(path, capture=capture)
  _verify_digest(
      digest,
      expected_bytes=member.bytes,
      expected_sha256=member.sha256,
      status=status,
  )
  return path, digest


def _verify_obj_member(
    volume: _VolumeReceipt,
    relative: str,
) -> tuple[Path, _Digest, str]:
  """Bind OBJ bytes either directly or through the authenticated official 7z.

  The current extraction receipt intentionally hashes STEP/JSON only.  Its
  own receipt-bound archive record authenticates the complete official 7z, so
  an OBJ absent from the content-member list is verified byte-for-byte against
  the matching member streamed from that already SHA-256-authenticated archive.
  """

  if relative in volume.members:
    path, digest = _verify_receipt_member(
        volume,
        relative,
        status="obj_receipt_mismatch",
        capture=True,
    )
    return path, digest, "content_receipt_member"
  if volume.archive_path is None or volume.archive_sha256 is None:
    raise _CaseFailure(
        "obj_receipt_mismatch",
        f"OBJ lacks both a content member and authenticated archive binding: {relative}",
    )
  path = _resolve_member(volume.root, relative)
  live_digest = _plain_file_digest(path, capture=True)
  try:
    seven_zip = find_7z(None)
    completed = subprocess.run(
        [
            str(seven_zip),
            "x",
            "-so",
            "-bd",
            "-y",
            str(volume.archive_path),
            relative,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
  except (OSError, subprocess.SubprocessError) as error:
    raise _CaseFailure(
        "obj_receipt_mismatch",
        f"failed to stream OBJ from authenticated archive: {relative}",
    ) from error
  if completed.returncode != 0:
    raise _CaseFailure(
        "obj_receipt_mismatch",
        f"authenticated archive does not yield the requested OBJ: {relative}",
    )
  archived_bytes = completed.stdout
  archived_sha256 = hashlib.sha256(archived_bytes).hexdigest()
  if (
      len(archived_bytes) != live_digest.bytes
      or archived_sha256 != live_digest.sha256
  ):
    raise _CaseFailure(
        "obj_receipt_mismatch",
        f"live OBJ differs from authenticated archive member: {relative}",
    )
  return path, live_digest, "authenticated_archive_member_bytes"


def _point3(raw: Any, *, status: str) -> np.ndarray:
  if isinstance(raw, Mapping):
    values = (raw.get("x"), raw.get("y"), raw.get("z"))
  elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
    values = tuple(raw)
  else:
    raise _CaseFailure(status, "expected a three-dimensional point")
  if len(values) != 3:
    raise _CaseFailure(status, "expected a three-dimensional point")
  try:
    point = np.asarray([float(value) for value in values], dtype=float)
  except (TypeError, ValueError) as error:
    raise _CaseFailure(status, "point contains a non-numeric value") from error
  if not np.all(np.isfinite(point)):
    raise _CaseFailure(status, "point contains a non-finite value")
  return point


def _entity_bbox_mm(entity: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
  raw = entity.get("bounding_box")
  if not isinstance(raw, Mapping):
    raise _CaseFailure("source_geometry_invalid", "contact endpoint lacks bounding_box")
  minimum = _point3(raw.get("min_point"), status="source_geometry_invalid") * 10.0
  maximum = _point3(raw.get("max_point"), status="source_geometry_invalid") * 10.0
  if np.any(maximum < minimum):
    raise _CaseFailure("source_geometry_invalid", "contact endpoint bbox is inverted")
  return minimum, maximum


def _entity_point_mm(entity: Mapping[str, Any]) -> np.ndarray:
  return _point3(
      entity.get("point_on_entity"),
      status="source_geometry_invalid",
  ) * 10.0


def _surface_type(raw: Any) -> str:
  text = str(raw or "").strip().upper().replace("_", "")
  if text.endswith("SURFACETYPE"):
    text = text[: -len("SURFACETYPE")]
  aliases = {
      "NURBS": "BSPLINE",
      "BSPLINESURFACE": "BSPLINE",
      "BEZIERSURFACE": "BEZIER",
      "PLANAR": "PLANE",
      "CYLINDRICAL": "CYLINDER",
      "CONICAL": "CONE",
      "SPHERICAL": "SPHERE",
      "TOROIDAL": "TORUS",
  }
  return aliases.get(text, text)


def _parse_obj_face_groups(path: Path) -> dict[int, _ObjFaceGroup]:
  vertices_cm: list[tuple[float, float, float]] = []
  triangles_by_group: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
  current: int | None = None
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except (OSError, UnicodeError) as error:
    raise _CaseFailure("obj_parse_failed", "authoritative OBJ is unreadable") from error
  for line_number, raw_line in enumerate(lines, start=1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    fields = line.split()
    if fields[0] == "v":
      if len(fields) < 4:
        raise _CaseFailure("obj_parse_failed", f"invalid OBJ vertex at line {line_number}")
      try:
        point = tuple(float(value) for value in fields[1:4])
      except ValueError as error:
        raise _CaseFailure("obj_parse_failed", "OBJ vertex is non-numeric") from error
      if not all(math.isfinite(value) for value in point):
        raise _CaseFailure("obj_parse_failed", "OBJ vertex is non-finite")
      vertices_cm.append(point)
      continue
    if fields[0] == "g":
      name = " ".join(fields[1:])
      match = _FACE_GROUP_RE.fullmatch(name)
      current = int(match.group(1)) if match is not None else None
      continue
    if fields[0] != "f" or current is None:
      continue
    if len(fields) < 4:
      raise _CaseFailure("obj_parse_failed", "OBJ face has fewer than three vertices")
    indices: list[int] = []
    for token in fields[1:]:
      raw_index = token.split("/", 1)[0]
      try:
        index = int(raw_index)
      except ValueError as error:
        raise _CaseFailure("obj_parse_failed", "OBJ face index is invalid") from error
      if index == 0:
        raise _CaseFailure("obj_parse_failed", "OBJ uses forbidden zero vertex index")
      resolved = index - 1 if index > 0 else len(vertices_cm) + index
      if resolved < 0 or resolved >= len(vertices_cm):
        raise _CaseFailure("obj_parse_failed", "OBJ face index is out of range")
      indices.append(resolved)
    for offset in range(1, len(indices) - 1):
      triangles_by_group[current].append(
          (indices[0], indices[offset], indices[offset + 1])
      )
  vertices = np.asarray(vertices_cm, dtype=float) * 10.0
  groups: dict[int, _ObjFaceGroup] = {}
  for fusion_index, raw_triangles in triangles_by_group.items():
    if not raw_triangles:
      continue
    triangle_indices = np.asarray(raw_triangles, dtype=int)
    triangle_points = vertices[triangle_indices]
    cross = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    triangle_areas = np.linalg.norm(cross, axis=1) * 0.5
    if (
        np.any(~np.isfinite(triangle_areas))
        or np.any(triangle_areas <= 1e-12)
        or float(np.sum(triangle_areas)) <= 0.0
    ):
      raise _CaseFailure("obj_parse_failed", "OBJ face group has zero/invalid mesh area")
    referenced = vertices[np.unique(triangle_indices.reshape(-1))]
    signature_payload = {
        "schema": "fusion_obj_face_group.v1",
        "fusion_face_index": fusion_index,
        "triangles_mm": [
            [
                [format(float(value), ".12g") for value in point]
                for point in triangle
            ]
            for triangle in triangle_points
        ],
    }
    groups[fusion_index] = _ObjFaceGroup(
        fusion_face_index=fusion_index,
        triangles_mm=triangle_points,
        vertices_mm=referenced,
        mesh_area_mm2=float(np.sum(triangle_areas)),
        bbox_min_mm=np.min(referenced, axis=0),
        bbox_max_mm=np.max(referenced, axis=0),
        signature_sha256=_canonical_sha256(signature_payload),
    )
  return groups


def _occ_faces(shape: Any) -> list[_OccFace]:
  value = shape.val() if hasattr(shape, "val") else shape
  result: list[_OccFace] = []
  for index, face in enumerate(value.Faces()):
    bbox = face.BoundingBox()
    minimum = np.asarray([bbox.xmin, bbox.ymin, bbox.zmin], dtype=float)
    maximum = np.asarray([bbox.xmax, bbox.ymax, bbox.zmax], dtype=float)
    area = float(face.Area())
    if (
        not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or not math.isfinite(area)
        or area <= 0.0
    ):
      raise _CaseFailure("occ_geometry_invalid", "OCC produced invalid face geometry")
    result.append(
        _OccFace(
            index=index,
            face=face,
            surface_type=_surface_type(face.geomType()),
            bbox_min_mm=minimum,
            bbox_max_mm=maximum,
            area_mm2=area,
            signature_sha256=source_face_signature_sha256(face),
        )
    )
  if not result:
    raise _CaseFailure("occ_geometry_invalid", "STEP import produced no faces")
  return result


def _obj_sample_count(group: _ObjFaceGroup) -> int:
  return int(len(group.triangles_mm) * 4)


def _obj_sample_chunks(
    group: _ObjFaceGroup,
    *,
    maximum: int,
    chunk_size: int,
) -> Iterator[np.ndarray]:
  """Yield every locked OBJ sample in stable triangle-major order.

  Chunking bounds transient arrays only.  It never selects, subsamples, or
  changes the four barycentric proof points generated for each triangle.
  """

  sample_count = _obj_sample_count(group)
  if sample_count > maximum:
    raise _CaseFailure(
        "sampling_budget_exceeded",
        "authoritative OBJ group exceeds the locked total sampling cap",
    )
  barycentric = np.asarray(
      [
          (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
          (0.60, 0.20, 0.20),
          (0.20, 0.60, 0.20),
          (0.20, 0.20, 0.60),
      ],
      dtype=float,
  )
  triangles_per_chunk = max(1, chunk_size // len(barycentric))
  for start in range(0, len(group.triangles_mm), triangles_per_chunk):
    triangles = group.triangles_mm[start : start + triangles_per_chunk]
    yield np.einsum("bi,tij->tbj", barycentric, triangles).reshape(-1, 3)


def _array_chunks(values: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
  for start in range(0, len(values), chunk_size):
    yield values[start : start + chunk_size]


def _point_triangle_distance(
    point: np.ndarray,
    triangle: np.ndarray,
) -> float:
  """Exact Euclidean point-to-triangle distance (Ericson region tests)."""

  a, b, c = triangle
  ab = b - a
  ac = c - a
  ap = point - a
  d1 = float(np.dot(ab, ap))
  d2 = float(np.dot(ac, ap))
  if d1 <= 0.0 and d2 <= 0.0:
    return float(np.linalg.norm(ap))
  bp = point - b
  d3 = float(np.dot(ab, bp))
  d4 = float(np.dot(ac, bp))
  if d3 >= 0.0 and d4 <= d3:
    return float(np.linalg.norm(bp))
  vc = d1 * d4 - d3 * d2
  if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
    fraction = d1 / (d1 - d3)
    return float(np.linalg.norm(point - (a + fraction * ab)))
  cp = point - c
  d5 = float(np.dot(ab, cp))
  d6 = float(np.dot(ac, cp))
  if d6 >= 0.0 and d5 <= d6:
    return float(np.linalg.norm(cp))
  vb = d5 * d2 - d1 * d6
  if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
    fraction = d2 / (d2 - d6)
    return float(np.linalg.norm(point - (a + fraction * ac)))
  va = d3 * d6 - d5 * d4
  if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
    edge = c - b
    fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    return float(np.linalg.norm(point - (b + fraction * edge)))
  denominator = va + vb + vc
  if abs(denominator) <= 1e-18:
    raise _CaseFailure("obj_parse_failed", "OBJ contains a degenerate triangle")
  inverse = 1.0 / denominator
  v = vb * inverse
  w = vc * inverse
  projection = a + ab * v + ac * w
  return float(np.linalg.norm(point - projection))


@dataclass
class _ForwardDistanceCallBudget:
  """One mutable OCC point-to-face call counter at one proof scope."""

  scope: str
  maximum: int
  required_distance_calls: int = 0
  attempted_distance_calls: int = 0
  completed_distance_calls: int = 0
  work_cap_exhaustion_count: int = 0

  def __post_init__(self) -> None:
    if self.scope not in {"endpoint", "case", "run"}:
      raise ValueError("forward distance-call budget scope is invalid")
    if (
        isinstance(self.maximum, bool)
        or not isinstance(self.maximum, int)
        or self.maximum < 1
    ):
      raise ValueError("forward distance-call budget maximum must be positive")

  @property
  def remaining(self) -> int:
    return max(0, self.maximum - self.attempted_distance_calls)

  def receipt(self) -> dict[str, int]:
    return {
        "maximum_distance_call_count": self.maximum,
        "required_distance_call_count": self.required_distance_calls,
        "attempted_distance_call_count": self.attempted_distance_calls,
        "completed_distance_call_count": self.completed_distance_calls,
        "remaining_distance_call_count": self.remaining,
        "work_cap_exhaustion_count": self.work_cap_exhaustion_count,
    }


class _ForwardWorkCapExceeded(_CaseFailure):
  def __init__(
      self,
      *,
      scope: str,
      required_total_calls: int,
      required_new_calls: int,
  ) -> None:
    super().__init__(
        "forward_distance_work_budget_exceeded",
        "complete forward OCC point-to-face proof exceeds the locked "
        f"{scope} distance-call cap",
    )
    self.scope = scope
    self.required_total_calls = int(required_total_calls)
    self.required_new_calls = int(required_new_calls)


@dataclass(frozen=True)
class _ForwardSampleRanking:
  best_distance_mm: float
  best_index: int
  second_best_distance_lower_bound_mm: float | None
  second_best_index: int | None
  exact_candidate_count: int
  pruned_candidate_count: int


_FaceAabbStableKey = tuple[float | int, ...]


@dataclass(frozen=True)
class _FaceAabbBvhNode:
  bbox_min_mm: np.ndarray
  bbox_max_mm: np.ndarray
  faces: tuple[_OccFace, ...]
  left: _FaceAabbBvhNode | None
  right: _FaceAabbBvhNode | None
  face_count: int
  minimum_face_index: int
  stable_key: _FaceAabbStableKey
  ordinal: int
  depth: int


@dataclass(frozen=True)
class _FaceAabbBvhIndex:
  root: _FaceAabbBvhNode
  nodes: tuple[_FaceAabbBvhNode, ...]
  faces: tuple[_OccFace, ...]
  face_position_by_index: Mapping[int, int]
  candidate_keys: tuple[_FaceAabbStableKey, ...]
  node_count: int
  leaf_count: int
  maximum_depth: int
  leaf_size: int


def _outward_down_nonnegative(value: float) -> float:
  numeric = float(value)
  if not math.isfinite(numeric) or numeric < 0.0:
    raise ValueError("outward-rounded distance input must be finite and nonnegative")
  if numeric == 0.0:
    return 0.0
  return max(0.0, math.nextafter(numeric, -math.inf))


def _points_aabb_distance_lower_bounds(
    points_mm: np.ndarray,
    bbox_min_mm: np.ndarray,
    bbox_max_mm: np.ndarray,
) -> np.ndarray:
  """Conservative lower bounds for a true points-by-AABBs NumPy block.

  Every floating subtraction, square, axis accumulation, and square root is
  rounded one representable value toward negative infinity.  This mirrors the
  locked scalar v2.4 proof while evaluating one leaf-sized AABB block at once.
  """

  points = np.asarray(points_mm, dtype=float)
  minimum = np.asarray(bbox_min_mm, dtype=float)
  maximum = np.asarray(bbox_max_mm, dtype=float)
  if (
      points.ndim != 2
      or points.shape[1:] != (3,)
      or not np.all(np.isfinite(points))
  ):
    raise _CaseFailure("occ_geometry_invalid", "forward sample block is invalid")
  if (
      minimum.ndim != 2
      or minimum.shape[1:] != (3,)
      or maximum.shape != minimum.shape
      or not np.all(np.isfinite(minimum))
      or not np.all(np.isfinite(maximum))
  ):
    raise _CaseFailure("occ_geometry_invalid", "OCC face AABB block is invalid")
  if np.any(minimum > maximum):
    raise _CaseFailure("occ_geometry_invalid", "OCC face AABB is inverted")
  if len(points) == 0 or len(minimum) == 0:
    return np.empty((len(points), len(minimum)), dtype=float)
  with np.errstate(over="ignore", invalid="ignore"):
    below = np.subtract(
        minimum[np.newaxis, :, :],
        points[:, np.newaxis, :],
    )
    above = np.subtract(
        points[:, np.newaxis, :],
        maximum[np.newaxis, :, :],
    )
  if not np.all(np.isfinite(below)) or not np.all(np.isfinite(above)):
    raise _CaseFailure("occ_geometry_invalid", "OCC face AABB distance overflowed")
  raw_delta = np.maximum(np.maximum(below, above), 0.0)
  delta = np.where(
      raw_delta > 0.0,
      np.nextafter(raw_delta, -np.inf),
      0.0,
  )
  with np.errstate(over="ignore", invalid="ignore"):
    raw_squared = np.multiply(delta, delta)
  if not np.all(np.isfinite(raw_squared)):
    raise _CaseFailure("occ_geometry_invalid", "OCC face AABB square overflowed")
  squared = np.where(
      raw_squared > 0.0,
      np.maximum(0.0, np.nextafter(raw_squared, -np.inf)),
      0.0,
  )
  total = np.zeros((len(points), len(minimum)), dtype=float)
  for axis in range(3):
    raw_total = np.add(total, squared[:, :, axis])
    if not np.all(np.isfinite(raw_total)):
      raise _CaseFailure("occ_geometry_invalid", "OCC face AABB sum overflowed")
    total = np.where(
        raw_total > 0.0,
        np.maximum(0.0, np.nextafter(raw_total, -np.inf)),
        0.0,
    )
  root = np.sqrt(total)
  if not np.all(np.isfinite(root)):
    raise _CaseFailure("occ_geometry_invalid", "OCC face AABB root is invalid")
  return np.where(
      root > 0.0,
      np.maximum(0.0, np.nextafter(root, -np.inf)),
      0.0,
  )


def _point_aabb_distance_lower_bounds(
    point_mm: np.ndarray,
    bbox_min_mm: np.ndarray,
    bbox_max_mm: np.ndarray,
) -> np.ndarray:
  point = np.asarray(point_mm, dtype=float)
  if point.shape != (3,):
    raise _CaseFailure("occ_geometry_invalid", "forward sample point is invalid")
  return _points_aabb_distance_lower_bounds(
      point.reshape(1, 3),
      bbox_min_mm,
      bbox_max_mm,
  )[0]


def _point_face_aabb_distance_lower_bound(
    point_mm: np.ndarray,
    face: _OccFace,
) -> float:
  """Return an IEEE-754 outward-down point-to-face-AABB lower bound."""

  return float(
      _point_aabb_distance_lower_bounds(
          point_mm,
          np.asarray(face.bbox_min_mm, dtype=float).reshape(1, 3),
          np.asarray(face.bbox_max_mm, dtype=float).reshape(1, 3),
      )[0]
  )


def _occ_face_aabb_stable_key(face: _OccFace) -> _FaceAabbStableKey:
  minimum = np.asarray(face.bbox_min_mm, dtype=float)
  maximum = np.asarray(face.bbox_max_mm, dtype=float)
  if (
      minimum.shape != (3,)
      or maximum.shape != (3,)
      or not np.all(np.isfinite(minimum))
      or not np.all(np.isfinite(maximum))
  ):
    raise _CaseFailure("occ_geometry_invalid", "OCC face AABB is invalid")
  if np.any(minimum > maximum):
    raise _CaseFailure("occ_geometry_invalid", "OCC face AABB is inverted")
  if isinstance(face.index, bool) or not isinstance(face.index, int) or face.index < 0:
    raise _CaseFailure("occ_geometry_invalid", "OCC face index is invalid")
  coordinates = tuple(
      0.0 if float(value) == 0.0 else float(value)
      for value in np.concatenate((minimum, maximum))
  )
  return (*coordinates, face.index)


def _build_occ_face_aabb_bvh(
    candidates: Sequence[_OccFace],
) -> _FaceAabbBvhIndex:
  """Build a balanced OCC-face AABB BVH independent of input ordering."""

  if not candidates:
    raise ValueError("forward ranking requires at least one OCC face")
  keyed = [(_occ_face_aabb_stable_key(face), face) for face in candidates]
  indices = [face.index for _, face in keyed]
  if len(set(indices)) != len(indices):
    raise _CaseFailure("occ_geometry_invalid", "OCC face indices are duplicated")
  keyed.sort(key=lambda row: row[0])
  candidate_keys = tuple(key for key, _ in keyed)
  key_by_index = {face.index: key for key, face in keyed}
  ordinal_counter = 0
  node_count = 0
  leaf_count = 0
  maximum_depth = 0
  nodes_by_ordinal: dict[int, _FaceAabbBvhNode] = {}

  def build(faces: tuple[_OccFace, ...], depth: int) -> _FaceAabbBvhNode:
    nonlocal ordinal_counter, node_count, leaf_count, maximum_depth
    ordinal = ordinal_counter
    ordinal_counter += 1
    node_count += 1
    maximum_depth = max(maximum_depth, depth)
    minimum_face_index = min(face.index for face in faces)
    stable_key = min(key_by_index[face.index] for face in faces)
    if len(faces) <= _FACE_AABB_BVH_LEAF_SIZE:
      leaf_count += 1
      ordered_faces = tuple(
          sorted(faces, key=lambda face: key_by_index[face.index])
      )
      minimum = np.min(
          np.asarray([face.bbox_min_mm for face in ordered_faces], dtype=float),
          axis=0,
      )
      maximum = np.max(
          np.asarray([face.bbox_max_mm for face in ordered_faces], dtype=float),
          axis=0,
      )
      node = _FaceAabbBvhNode(
          bbox_min_mm=minimum,
          bbox_max_mm=maximum,
          faces=ordered_faces,
          left=None,
          right=None,
          face_count=len(ordered_faces),
          minimum_face_index=minimum_face_index,
          stable_key=stable_key,
          ordinal=ordinal,
          depth=depth,
      )
      nodes_by_ordinal[ordinal] = node
      return node
    centroids = {
        face.index: (
            np.asarray(face.bbox_min_mm, dtype=float) * 0.5
            + np.asarray(face.bbox_max_mm, dtype=float) * 0.5
        )
        for face in faces
    }
    centroid_values = np.asarray(list(centroids.values()), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
      spans = np.max(centroid_values, axis=0) - np.min(
          centroid_values, axis=0
      )
    axis = int(np.argmax(spans))
    ordered_faces = tuple(
        sorted(
            faces,
            key=lambda face: (
                float(centroids[face.index][axis]),
                key_by_index[face.index],
            ),
        )
    )
    midpoint = len(ordered_faces) // 2
    left = build(ordered_faces[:midpoint], depth + 1)
    right = build(ordered_faces[midpoint:], depth + 1)
    minimum = np.minimum(left.bbox_min_mm, right.bbox_min_mm)
    maximum = np.maximum(left.bbox_max_mm, right.bbox_max_mm)
    if (
        np.any(minimum > left.bbox_min_mm)
        or np.any(minimum > right.bbox_min_mm)
        or np.any(maximum < left.bbox_max_mm)
        or np.any(maximum < right.bbox_max_mm)
    ):
      raise _CaseFailure("occ_geometry_invalid", "face BVH node is not conservative")
    node = _FaceAabbBvhNode(
        bbox_min_mm=minimum,
        bbox_max_mm=maximum,
        faces=(),
        left=left,
        right=right,
        face_count=len(ordered_faces),
        minimum_face_index=minimum_face_index,
        stable_key=stable_key,
        ordinal=ordinal,
        depth=depth,
    )
    nodes_by_ordinal[ordinal] = node
    return node

  canonical_faces = tuple(face for _, face in keyed)
  root = build(canonical_faces, 0)
  return _FaceAabbBvhIndex(
      root=root,
      nodes=tuple(nodes_by_ordinal[index] for index in range(node_count)),
      faces=canonical_faces,
      face_position_by_index={
          face.index: position for position, face in enumerate(canonical_faces)
      },
      candidate_keys=candidate_keys,
      node_count=node_count,
      leaf_count=leaf_count,
      maximum_depth=maximum_depth,
      leaf_size=_FACE_AABB_BVH_LEAF_SIZE,
  )


def _outward_up_sum(left: float, right: float) -> float:
  total = float(left) + float(right)
  if not math.isfinite(total) or total < 0.0:
    raise ValueError("margin proof inputs must be finite and nonnegative")
  return math.nextafter(total, math.inf)


def _outward_down_difference(upper: float, lower: float) -> float:
  difference = float(upper) - float(lower)
  return _outward_down_nonnegative(max(0.0, difference))


def _outward_up_nonnegative_difference(upper: float, lower: float) -> float:
  difference = float(upper) - float(lower)
  if not math.isfinite(difference):
    raise ValueError("ambiguity proof inputs must be finite")
  if difference <= 0.0:
    return 0.0
  return math.nextafter(difference, math.inf)


class _ForwardDistanceCallEngine:
  """Audit and enforce complete forward OCC point-to-face call work."""

  def __init__(
      self,
      *,
      maximum_endpoint_calls: int,
      case_budget: _ForwardDistanceCallBudget,
      run_budget: _ForwardDistanceCallBudget,
      analytic_coverage_threshold_mm: float | None = None,
  ) -> None:
    self.endpoint_budget = _ForwardDistanceCallBudget(
        scope="endpoint",
        maximum=maximum_endpoint_calls,
    )
    if case_budget.scope != "case" or run_budget.scope != "run":
      raise ValueError("shared forward distance-call budget scopes are invalid")
    self.case_budget = case_budget
    self.run_budget = run_budget
    self.forward_sample_count = 0
    self.candidate_occ_face_count = 0
    self.sample_candidate_call_upper_bound = 0
    self.source_point_candidate_call_upper_bound = 0
    self._sample_proof_configured = False
    self._source_point_configured = False
    self._cap_failure: _ForwardWorkCapExceeded | None = None
    self.aabb_lower_bound_evaluations = 0
    self.bvh_node_lower_bound_evaluations = 0
    self.bvh_leaf_face_lower_bound_evaluations = 0
    self.pruned_occ_face_candidates = 0
    self.forward_sample_queries = 0
    self.completed_forward_sample_queries = 0
    self.pruned_sample_queries = 0
    self.fully_exact_sample_queries = 0
    self.bvh_max_traversal_depth_visited = 0
    self.analytic_support_lb_queries = 0
    self.analytic_trimmed_proof_count = 0
    self.analytic_guard_fallback_count = 0
    self.persistent_kernel_performs = 0
    self.ambiguous_proof_stop_count = 0
    self.ambiguous_proof_pruned_occ_face_candidate_count = 0
    self._locked_minimum_margin_mm: float | None = None
    self._minimum_proven_margin_lower_bound_mm: float | None = None
    self._face_bvh_index: _FaceAabbBvhIndex | None = None
    self._analytic_coverage_threshold_mm = analytic_coverage_threshold_mm
    self._analytic_supports: dict[int, _AnalyticFaceSupport] = {}
    self._analytic_trim_classifiers: dict[int, Any] = {}
    self._persistent_kernels: dict[int, _PersistentPointFaceDistanceKernel] = {}

  @property
  def budgets(self) -> tuple[_ForwardDistanceCallBudget, ...]:
    return (self.endpoint_budget, self.case_budget, self.run_budget)

  def configure_forward_proof(
      self,
      *,
      sample_count: int,
      candidate_count: int | None = None,
      candidates: Sequence[_OccFace] | None = None,
  ) -> None:
    samples = int(sample_count)
    if (candidate_count is None) == (candidates is None):
      raise ValueError("provide exactly one forward candidate configuration")
    if candidates is not None:
      self._face_bvh_index = _build_occ_face_aabb_bvh(candidates)
      configured_candidate_count = len(self._face_bvh_index.faces)
      for face in self._face_bvh_index.faces:
        support = _analytic_face_support(face)
        if support is not None:
          self._analytic_supports[face.index] = support
          try:
            from OCP.BRepClass import BRepClass_FaceClassifier

            self._analytic_trim_classifiers[face.index] = (
                BRepClass_FaceClassifier()
            )
          except Exception:
            self._analytic_supports.pop(face.index, None)
        if hasattr(face.face, "wrapped"):
          try:
            self._persistent_kernels[face.index] = (
                _PersistentPointFaceDistanceKernel(face)
            )
          except Exception as error:
            raise _CaseFailure(
                "occ_distance_failed",
                "persistent OCC point-face kernel initialization failed",
            ) from error
    else:
      assert candidate_count is not None
      configured_candidate_count = int(candidate_count)
    if samples < 0 or configured_candidate_count < 0:
      raise ValueError("forward proof dimensions cannot be negative")
    if self._sample_proof_configured:
      raise ValueError("forward proof dimensions were already configured")
    self.forward_sample_count = samples
    self.candidate_occ_face_count = configured_candidate_count
    self.sample_candidate_call_upper_bound = samples * configured_candidate_count
    self._sample_proof_configured = True

  def configure_source_point_candidates(self, candidate_count: int) -> None:
    candidates = int(candidate_count)
    if candidates < 0:
      raise ValueError("source-point candidate count cannot be negative")
    if self._source_point_configured:
      raise ValueError("source-point distance work was already configured")
    self.source_point_candidate_call_upper_bound = candidates
    self._source_point_configured = True

  def _raise_cap(
      self,
      budget: _ForwardDistanceCallBudget,
      *,
      required_total_calls: int,
      required_new_calls: int,
  ) -> None:
    budget.work_cap_exhaustion_count += 1
    failure = _ForwardWorkCapExceeded(
        scope=budget.scope,
        required_total_calls=required_total_calls,
        required_new_calls=required_new_calls,
    )
    self._cap_failure = failure
    raise failure

  def distance(self, point_mm: np.ndarray, face: _OccFace, cq: Any) -> float:
    for budget in self.budgets:
      budget.required_distance_calls += 1
    for budget in self.budgets:
      if budget.attempted_distance_calls >= budget.maximum:
        self._raise_cap(
            budget,
            required_total_calls=budget.attempted_distance_calls + 1,
            required_new_calls=1,
        )
    for budget in self.budgets:
      budget.attempted_distance_calls += 1
    kernel = self._persistent_kernels.get(getattr(face, "index", None))
    if kernel is None:
      distance = _point_face_distance(point_mm, face, cq)
    else:
      self.persistent_kernel_performs += 1
      distance = kernel.distance(point_mm)
    for budget in self.budgets:
      budget.completed_distance_calls += 1
    return distance

  def _rank_sample_with_precomputed_bounds(
      self,
      point_mm: np.ndarray,
      node_lower_bounds: np.ndarray,
      face_lower_bounds: np.ndarray,
      cq: Any,
      *,
      margin: float,
  ) -> _ForwardSampleRanking:
    index = self._face_bvh_index
    if index is None:
      raise ValueError("forward ranking candidates were not frozen at configure time")
    self.forward_sample_queries += 1
    best: tuple[float, int] | None = None
    second: tuple[float, int] | None = None
    exact_count = 0
    pruned_count = 0
    pruned_lower_bounds: list[tuple[float, int | None]] = []
    root = index.root
    queue: list[
        tuple[float, int, int, int, _FaceAabbBvhNode | _OccFace, int]
    ] = [
        (
            float(node_lower_bounds[root.ordinal]),
            root.minimum_face_index,
            0,
            root.ordinal,
            root,
            root.depth,
        )
    ]
    maximum_depth_visited = 0
    ordered_face_positions = sorted(
        range(len(index.faces)),
        key=lambda position: (
            float(face_lower_bounds[position]),
            index.faces[position].index,
        ),
    )
    if (
        self._analytic_coverage_threshold_mm is not None
        and len(ordered_face_positions) > 1
    ):
      best_position = ordered_face_positions[0]
      best_face = index.faces[best_position]
      support = self._analytic_supports.get(best_face.index)
      if support is not None:
        estimate = _trimmed_analytic_distance_upper_bound(
            point_mm,
            best_face,
            support,
            self._analytic_trim_classifiers.get(best_face.index),
        )
        second_lower_bound = float(
            face_lower_bounds[ordered_face_positions[1]]
        )
        if estimate is not None:
          estimate_upper, guard = estimate
          unique_safe = (
              _outward_up_sum(estimate_upper, margin)
              <= second_lower_bound
          )
          coverage_safe = (
              estimate_upper
              <= self._analytic_coverage_threshold_mm - guard
          )
          if unique_safe and coverage_safe:
            pruned_count = len(index.faces) - 1
            self.pruned_occ_face_candidates += pruned_count
            self.pruned_sample_queries += 1
            self.analytic_trimmed_proof_count += 1
            self.completed_forward_sample_queries += 1
            proven_margin = _outward_down_difference(
                second_lower_bound,
                estimate_upper,
            )
            self._minimum_proven_margin_lower_bound_mm = (
                proven_margin
                if self._minimum_proven_margin_lower_bound_mm is None
                else min(
                    self._minimum_proven_margin_lower_bound_mm,
                    proven_margin,
                )
            )
            return _ForwardSampleRanking(
                best_distance_mm=estimate_upper,
                best_index=best_face.index,
                second_best_distance_lower_bound_mm=second_lower_bound,
                second_best_index=index.faces[
                    ordered_face_positions[1]
                ].index,
                exact_candidate_count=0,
                pruned_candidate_count=pruned_count,
            )
        self.analytic_guard_fallback_count += 1
    while queue:
      lower_bound, _, kind, _, item, depth = heapq.heappop(queue)
      maximum_depth_visited = max(maximum_depth_visited, depth)
      if best is not None:
        proof_threshold = _outward_up_sum(best[0], margin)
        evaluated_second_ambiguous = (
            second is not None and second[0] < proof_threshold
        )
        ambiguity_threshold = _outward_up_nonnegative_difference(
            best[0], margin
        )
        if evaluated_second_ambiguous and lower_bound >= ambiguity_threshold:
          frontier = [(kind, item), *((row[2], row[4]) for row in queue)]
          ambiguity_pruned = sum(
              candidate.face_count if candidate_kind == 0 else 1
              for candidate_kind, candidate in frontier
          )
          pruned_count += ambiguity_pruned
          self.ambiguous_proof_stop_count += 1
          self.ambiguous_proof_pruned_occ_face_candidate_count += (
              ambiguity_pruned
          )
          pruned_lower_bounds.append(
              (
                  lower_bound,
                  item.index
                  if kind == 1 and isinstance(item, _OccFace)
                  else None,
              )
          )
          queue.clear()
          break
        evaluated_second_safe = (
            second is None or second[0] >= proof_threshold
        )
        if evaluated_second_safe and lower_bound >= proof_threshold:
          frontier = [(kind, item), *((row[2], row[4]) for row in queue)]
          pruned_count += sum(
              candidate.face_count if candidate_kind == 0 else 1
              for candidate_kind, candidate in frontier
          )
          pruned_lower_bounds.append(
              (
                  lower_bound,
                  item.index if kind == 1 and isinstance(item, _OccFace) else None,
              )
          )
          queue.clear()
          break
      if kind == 1:
        assert isinstance(item, _OccFace)
        candidate = (self.distance(point_mm, item, cq), item.index)
        exact_count += 1
        if best is None or candidate < best:
          second = best
          best = candidate
        elif second is None or candidate < second:
          second = candidate
        continue
      assert isinstance(item, _FaceAabbBvhNode)
      node = item
      if node.left is not None and node.right is not None:
        for child in (node.left, node.right):
          heapq.heappush(
              queue,
              (
                  float(node_lower_bounds[child.ordinal]),
                  child.minimum_face_index,
                  0,
                  child.ordinal,
                  child,
                  child.depth,
              ),
          )
        continue
      for face in node.faces:
        position = index.face_position_by_index[face.index]
        heapq.heappush(
            queue,
            (
                float(face_lower_bounds[position]),
                face.index,
                1,
                face.index,
                face,
                node.depth,
            ),
        )

    if pruned_count:
      self.pruned_occ_face_candidates += pruned_count
      self.pruned_sample_queries += 1
    else:
      self.fully_exact_sample_queries += 1
    if best is None:
      raise _CaseFailure("occ_mapping_failed", "forward BVH produced no candidate")
    second_candidates: list[tuple[float, int | None]] = []
    if second is not None:
      second_candidates.append((second[0], second[1]))
    second_candidates.extend(
        (lower_bound, face_index)
        for lower_bound, face_index in pruned_lower_bounds
    )
    second_lower_bound: float | None = None
    second_index: int | None = None
    if second_candidates:
      second_lower_bound, second_index = min(
          second_candidates,
          key=lambda row: (
              row[0],
              row[1] is None,
              -1 if row[1] is None else row[1],
          ),
      )
    if pruned_count and second_lower_bound is not None:
      proven_margin = _outward_down_difference(
          second_lower_bound,
          best[0],
      )
      self._minimum_proven_margin_lower_bound_mm = (
          proven_margin
          if self._minimum_proven_margin_lower_bound_mm is None
          else min(
              self._minimum_proven_margin_lower_bound_mm,
              proven_margin,
          )
      )
    self.completed_forward_sample_queries += 1
    self.bvh_max_traversal_depth_visited = max(
        self.bvh_max_traversal_depth_visited,
        maximum_depth_visited,
    )
    return _ForwardSampleRanking(
        best_distance_mm=best[0],
        best_index=best[1],
        second_best_distance_lower_bound_mm=second_lower_bound,
        second_best_index=second_index,
        exact_candidate_count=exact_count,
        pruned_candidate_count=pruned_count,
    )

  def rank_sample_chunk(
      self,
      points_mm: np.ndarray,
      cq: Any,
      *,
      minimum_margin_mm: float,
  ) -> Iterator[_ForwardSampleRanking]:
    """Rank one locked sample chunk using 2-D vectorized AABB bounds."""

    margin = float(minimum_margin_mm)
    if not math.isfinite(margin) or margin <= 0.0:
      raise ValueError("forward ranking margin must be finite and positive")
    if self._locked_minimum_margin_mm is None:
      self._locked_minimum_margin_mm = margin
    elif self._locked_minimum_margin_mm != margin:
      raise ValueError("forward ranking margin changed within one proof")
    index = self._face_bvh_index
    if index is None:
      raise ValueError("forward ranking candidates were not frozen at configure time")
    points = np.asarray(points_mm, dtype=float)
    node_lower_bounds = _points_aabb_distance_lower_bounds(
        points,
        np.asarray([node.bbox_min_mm for node in index.nodes], dtype=float),
        np.asarray([node.bbox_max_mm for node in index.nodes], dtype=float),
    )
    face_lower_bounds = _points_aabb_distance_lower_bounds(
        points,
        np.asarray([face.bbox_min_mm for face in index.faces], dtype=float),
        np.asarray([face.bbox_max_mm for face in index.faces], dtype=float),
    )
    for position, face in enumerate(index.faces):
      support = self._analytic_supports.get(face.index)
      if support is None:
        continue
      support_bounds = _analytic_support_lower_bounds(points, support)
      face_lower_bounds[:, position] = np.maximum(
          face_lower_bounds[:, position],
          support_bounds,
      )
      self.analytic_support_lb_queries += len(points)
    self.bvh_node_lower_bound_evaluations += int(node_lower_bounds.size)
    self.bvh_leaf_face_lower_bound_evaluations += int(face_lower_bounds.size)
    self.aabb_lower_bound_evaluations += int(face_lower_bounds.size)
    def rankings() -> Iterator[_ForwardSampleRanking]:
      for offset, point in enumerate(points):
        yield self._rank_sample_with_precomputed_bounds(
            point,
            node_lower_bounds[offset],
            face_lower_bounds[offset],
            cq,
            margin=margin,
        )

    return rankings()

  def rank_sample_candidates(
      self,
      point_mm: np.ndarray,
      cq: Any,
      *,
      minimum_margin_mm: float,
  ) -> _ForwardSampleRanking:
    """Compatibility wrapper for one preconfigured sample query."""

    return next(
        self.rank_sample_chunk(
            np.asarray(point_mm, dtype=float).reshape(1, 3),
            cq,
            minimum_margin_mm=minimum_margin_mm,
        )
    )

  def receipt(
      self,
      *,
      all_forward_samples_checked: bool,
      source_point_checked: bool,
  ) -> dict[str, Any]:
    failure = self._cap_failure
    endpoint = self.endpoint_budget
    return {
        "schema_version": _FORWARD_DISTANCE_WORK_SCHEMA_VERSION,
        "algorithm": _FORWARD_DISTANCE_ALGORITHM,
        "forward_sample_count": self.forward_sample_count,
        "candidate_occ_face_count": self.candidate_occ_face_count,
        "sample_candidate_distance_call_upper_bound": (
            self.sample_candidate_call_upper_bound
        ),
        "source_point_candidate_distance_call_upper_bound": (
            self.source_point_candidate_call_upper_bound
        ),
        "total_proof_distance_call_upper_bound": (
            self.sample_candidate_call_upper_bound
            + self.source_point_candidate_call_upper_bound
        ),
        "required_distance_call_count": endpoint.required_distance_calls,
        "attempted_distance_call_count": endpoint.attempted_distance_calls,
        "completed_distance_call_count": endpoint.completed_distance_calls,
        "aabb_lower_bound_evaluation_count": (
            self.aabb_lower_bound_evaluations
        ),
        "bvh_node_lower_bound_evaluation_count": (
            self.bvh_node_lower_bound_evaluations
        ),
        "bvh_leaf_face_lower_bound_evaluation_count": (
            self.bvh_leaf_face_lower_bound_evaluations
        ),
        "bvh_pruned_occ_face_candidate_count": (
            self.pruned_occ_face_candidates
        ),
        "bvh_node_count": (
            0 if self._face_bvh_index is None else self._face_bvh_index.node_count
        ),
        "bvh_leaf_count": (
            0 if self._face_bvh_index is None else self._face_bvh_index.leaf_count
        ),
        "bvh_max_depth": (
            0
            if self._face_bvh_index is None
            else self._face_bvh_index.maximum_depth
        ),
      "bvh_max_traversal_depth_visited": (
          self.bvh_max_traversal_depth_visited
      ),
      "analytic_support_lb_queries": self.analytic_support_lb_queries,
      "analytic_trimmed_proof_count": self.analytic_trimmed_proof_count,
      "analytic_guard_fallback_count": self.analytic_guard_fallback_count,
      "persistent_kernel_performs": self.persistent_kernel_performs,
      "analytic_trim_guard_mm": _ANALYTIC_TRIM_GUARD_MIN_MM,
      "ambiguous_proof_stop_count": self.ambiguous_proof_stop_count,
      "ambiguous_proof_pruned_occ_face_candidate_count": (
          self.ambiguous_proof_pruned_occ_face_candidate_count
      ),
        "bvh_leaf_size": _FACE_AABB_BVH_LEAF_SIZE,
        "pruned_occ_face_candidate_count": self.pruned_occ_face_candidates,
        "exact_occ_distance_call_count": endpoint.completed_distance_calls,
        "conservative_margin_proof": {
            "schema_version": "occ_face_aabb_conservative_margin_proof.v1",
            "aabb_lower_bound_rounding": (
                "ieee754_outward_down_nextafter.v1"
            ),
            "prune_comparison": (
                "remaining_lower_bound_ge_nextafter_best_plus_margin_"
                "toward_posinf.v1"
            ),
            "locked_minimum_margin_mm": self._locked_minimum_margin_mm,
            "sample_query_count": self.forward_sample_queries,
            "completed_sample_query_count": (
                self.completed_forward_sample_queries
            ),
            "pruned_sample_query_count": self.pruned_sample_queries,
            "fully_exact_sample_query_count": self.fully_exact_sample_queries,
            "minimum_proven_second_best_margin_lower_bound_mm": (
                self._minimum_proven_margin_lower_bound_mm
            ),
            "all_prunes_margin_safe": True,
        },
        "caps": {
            "endpoint": endpoint.maximum,
            "case": self.case_budget.maximum,
            "run": self.run_budget.maximum,
        },
        "scope_counts": {
            budget.scope: budget.receipt() for budget in self.budgets
        },
        "work_cap_exhausted": failure is not None,
        "exhausted_scope": None if failure is None else failure.scope,
        "required_total_distance_call_count_at_failure": (
            None if failure is None else failure.required_total_calls
        ),
        "required_new_distance_call_count_at_failure": (
            None if failure is None else failure.required_new_calls
        ),
        "all_forward_samples_checked": bool(all_forward_samples_checked),
        "source_point_checked": bool(source_point_checked),
    }


@dataclass
class _TriangleEvaluationBudget:
  """One mutable exact-work counter shared at endpoint/case/run scope."""

  scope: str
  maximum: int
  exact_triangle_evaluations: int = 0
  attempted_triangle_evaluations: int = 0
  work_cap_exhaustion_count: int = 0

  def __post_init__(self) -> None:
    if self.scope not in {"endpoint", "case", "run"}:
      raise ValueError("triangle-evaluation budget scope is invalid")
    if (
        isinstance(self.maximum, bool)
        or not isinstance(self.maximum, int)
        or self.maximum < 1
    ):
      raise ValueError("triangle-evaluation budget maximum must be positive")

  @property
  def remaining(self) -> int:
    return max(0, self.maximum - self.exact_triangle_evaluations)

  def receipt(self) -> dict[str, int]:
    return {
        "maximum_triangle_evaluation_count": self.maximum,
        "exact_triangle_evaluation_count": self.exact_triangle_evaluations,
        "attempted_triangle_evaluation_count": (
            self.attempted_triangle_evaluations
        ),
        "remaining_triangle_evaluation_count": self.remaining,
        "work_cap_exhaustion_count": self.work_cap_exhaustion_count,
    }


class _TriangleWorkCapExceeded(_CaseFailure):
  def __init__(
      self,
      *,
      scope: str,
      required_lower_bound: int,
      minimum_new_evaluations: int,
  ) -> None:
    super().__init__(
        "reverse_distance_work_budget_exceeded",
        "exact reverse point-to-OBJ distance exceeds the locked "
        f"{scope} triangle-evaluation cap",
    )
    self.scope = scope
    self.required_lower_bound = int(required_lower_bound)
    self.minimum_new_evaluations = int(minimum_new_evaluations)


_TriangleGeometryKey = tuple[str, ...]


@dataclass(frozen=True)
class _TriangleBvhNode:
  bbox_min_mm: np.ndarray
  bbox_max_mm: np.ndarray
  triangle_indices: tuple[int, ...]
  left: _TriangleBvhNode | None
  right: _TriangleBvhNode | None
  triangle_count: int
  stable_key: _TriangleGeometryKey
  ordinal: int


@dataclass(frozen=True)
class _TriangleBvhIndex:
  triangles_mm: np.ndarray
  triangle_bbox_min_mm: np.ndarray
  triangle_bbox_max_mm: np.ndarray
  triangle_keys: tuple[_TriangleGeometryKey, ...]
  root: _TriangleBvhNode
  node_count: int
  leaf_count: int
  maximum_depth: int
  leaf_size: int


def _stable_float_key(value: float) -> str:
  normalized = 0.0 if value == 0.0 else float(value)
  return normalized.hex()


def _triangle_geometry_key(triangle: np.ndarray) -> _TriangleGeometryKey:
  vertices = sorted(
      tuple(_stable_float_key(float(value)) for value in vertex)
      for vertex in triangle
  )
  return tuple(value for vertex in vertices for value in vertex)


def _build_triangle_bvh(triangles_mm: np.ndarray) -> _TriangleBvhIndex:
  """Build a balanced, input-order-invariant triangle-AABB BVH."""

  triangles = np.asarray(triangles_mm, dtype=float)
  if (
      triangles.ndim != 3
      or triangles.shape[1:] != (3, 3)
      or len(triangles) == 0
      or not np.all(np.isfinite(triangles))
  ):
    raise _CaseFailure("obj_parse_failed", "OBJ triangle array is invalid")
  cross = np.cross(
      triangles[:, 1] - triangles[:, 0],
      triangles[:, 2] - triangles[:, 0],
  )
  if np.any(~np.isfinite(cross)) or np.any(np.linalg.norm(cross, axis=1) <= 1e-12):
    raise _CaseFailure("obj_parse_failed", "OBJ contains a degenerate triangle")
  triangle_minimum = np.min(triangles, axis=1)
  triangle_maximum = np.max(triangles, axis=1)
  centroids = np.mean(triangles, axis=1)
  if not np.all(np.isfinite(centroids)):
    raise _CaseFailure("obj_parse_failed", "OBJ triangle centroids are invalid")
  keys = tuple(_triangle_geometry_key(triangle) for triangle in triangles)
  ordinal_counter = 0
  node_count = 0
  leaf_count = 0
  maximum_depth = 0

  def build(indices: tuple[int, ...], depth: int) -> _TriangleBvhNode:
    nonlocal ordinal_counter, node_count, leaf_count, maximum_depth
    ordinal = ordinal_counter
    ordinal_counter += 1
    node_count += 1
    maximum_depth = max(maximum_depth, depth)
    index_array = np.asarray(indices, dtype=int)
    bbox_minimum = np.min(triangle_minimum[index_array], axis=0)
    bbox_maximum = np.max(triangle_maximum[index_array], axis=0)
    stable_key = min(keys[index] for index in indices)
    if len(indices) <= _TRIANGLE_BVH_LEAF_SIZE:
      leaf_count += 1
      ordered = tuple(sorted(indices, key=lambda index: (keys[index], index)))
      return _TriangleBvhNode(
          bbox_min_mm=bbox_minimum,
          bbox_max_mm=bbox_maximum,
          triangle_indices=ordered,
          left=None,
          right=None,
          triangle_count=len(indices),
          stable_key=stable_key,
          ordinal=ordinal,
      )
    centroid_minimum = np.min(centroids[index_array], axis=0)
    centroid_maximum = np.max(centroids[index_array], axis=0)
    axis = int(np.argmax(centroid_maximum - centroid_minimum))
    ordered = tuple(
        sorted(
            indices,
            key=lambda index: (
                float(centroids[index, axis]),
                keys[index],
                index,
            ),
        )
    )
    midpoint = len(ordered) // 2
    left = build(ordered[:midpoint], depth + 1)
    right = build(ordered[midpoint:], depth + 1)
    return _TriangleBvhNode(
        bbox_min_mm=bbox_minimum,
        bbox_max_mm=bbox_maximum,
        triangle_indices=(),
        left=left,
        right=right,
        triangle_count=len(indices),
        stable_key=stable_key,
        ordinal=ordinal,
    )

  root = build(tuple(range(len(triangles))), 0)
  return _TriangleBvhIndex(
      triangles_mm=triangles,
      triangle_bbox_min_mm=triangle_minimum,
      triangle_bbox_max_mm=triangle_maximum,
      triangle_keys=keys,
      root=root,
      node_count=node_count,
      leaf_count=leaf_count,
      maximum_depth=maximum_depth,
      leaf_size=_TRIANGLE_BVH_LEAF_SIZE,
  )


def _aabb_distance_lower_bound_squared(
    point: np.ndarray,
    bbox_minimum: np.ndarray,
    bbox_maximum: np.ndarray,
) -> float:
  """Return a conservatively rounded squared point-to-AABB lower bound."""

  total = 0.0
  for coordinate, minimum, maximum in zip(
      point,
      bbox_minimum,
      bbox_maximum,
  ):
    if coordinate < minimum:
      raw_delta = float(minimum - coordinate)
    elif coordinate > maximum:
      raw_delta = float(coordinate - maximum)
    else:
      raw_delta = 0.0
    if raw_delta <= 0.0:
      continue
    delta = math.nextafter(raw_delta, -math.inf)
    squared = math.nextafter(delta * delta, -math.inf)
    total = math.nextafter(total + max(0.0, squared), -math.inf)
  return max(0.0, total)


class _TriangleMeshDistanceEngine:
  """Exact point-to-mesh distance with deterministic BVH pruning and work caps."""

  def __init__(
      self,
      triangles_mm: np.ndarray,
      *,
      maximum_endpoint_triangle_evaluations: int,
      case_budget: _TriangleEvaluationBudget,
      run_budget: _TriangleEvaluationBudget,
  ) -> None:
    self.index = _build_triangle_bvh(triangles_mm)
    self.endpoint_budget = _TriangleEvaluationBudget(
        scope="endpoint",
        maximum=maximum_endpoint_triangle_evaluations,
    )
    if case_budget.scope != "case" or run_budget.scope != "run":
      raise ValueError("shared triangle-evaluation budget scopes are invalid")
    self.case_budget = case_budget
    self.run_budget = run_budget
    self.started_queries = 0
    self.completed_queries = 0
    self.aabb_evaluations = 0
    self.pruned_triangle_candidates = 0
    self.maximum_exact_evaluations_per_query = 0
    self._cap_failure: _TriangleWorkCapExceeded | None = None

  @property
  def budgets(self) -> tuple[_TriangleEvaluationBudget, ...]:
    return (self.endpoint_budget, self.case_budget, self.run_budget)

  def _raise_cap(
      self,
      budget: _TriangleEvaluationBudget,
      *,
      required_lower_bound: int,
      minimum_new_evaluations: int,
  ) -> None:
    budget.work_cap_exhaustion_count += 1
    error = _TriangleWorkCapExceeded(
        scope=budget.scope,
        required_lower_bound=required_lower_bound,
        minimum_new_evaluations=minimum_new_evaluations,
    )
    self._cap_failure = error
    raise error

  def require_minimum_evaluations(self, count: int) -> None:
    """Preflight work known to require at least one exact test per sample."""

    minimum = int(count)
    if minimum < 0:
      raise ValueError("minimum triangle-evaluation requirement is negative")
    for budget in self.budgets:
      required = budget.exact_triangle_evaluations + minimum
      if required > budget.maximum:
        self._raise_cap(
            budget,
            required_lower_bound=required,
            minimum_new_evaluations=minimum,
        )

  def _consume_exact_evaluation(self) -> None:
    for budget in self.budgets:
      budget.attempted_triangle_evaluations += 1
    for budget in self.budgets:
      if budget.exact_triangle_evaluations >= budget.maximum:
        self._raise_cap(
            budget,
            required_lower_bound=budget.attempted_triangle_evaluations,
            minimum_new_evaluations=1,
        )
    for budget in self.budgets:
      budget.exact_triangle_evaluations += 1

  def distance(self, point_mm: np.ndarray) -> float:
    point = np.asarray(point_mm, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
      raise _CaseFailure("reverse_coverage_failed", "reverse sample is invalid")
    self.started_queries += 1
    exact_before = self.endpoint_budget.exact_triangle_evaluations
    best_distance = math.inf
    best_squared = math.inf
    best_key: tuple[float, _TriangleGeometryKey, int] | None = None
    root = self.index.root
    root_lower = _aabb_distance_lower_bound_squared(
        point,
        root.bbox_min_mm,
        root.bbox_max_mm,
    )
    self.aabb_evaluations += 1
    queue: list[
        tuple[float, _TriangleGeometryKey, int, _TriangleBvhNode]
    ] = [(root_lower, root.stable_key, root.ordinal, root)]
    try:
      while queue:
        lower_bound, _, _, node = heapq.heappop(queue)
        if lower_bound > best_squared:
          self.pruned_triangle_candidates += node.triangle_count
          continue
        if node.left is not None and node.right is not None:
          for child in (node.left, node.right):
            child_lower = _aabb_distance_lower_bound_squared(
                point,
                child.bbox_min_mm,
                child.bbox_max_mm,
            )
            self.aabb_evaluations += 1
            if child_lower > best_squared:
              self.pruned_triangle_candidates += child.triangle_count
            else:
              heapq.heappush(
                  queue,
                  (
                      child_lower,
                      child.stable_key,
                      child.ordinal,
                      child,
                  ),
              )
          continue
        for triangle_index in node.triangle_indices:
          triangle_lower = _aabb_distance_lower_bound_squared(
              point,
              self.index.triangle_bbox_min_mm[triangle_index],
              self.index.triangle_bbox_max_mm[triangle_index],
          )
          self.aabb_evaluations += 1
          if triangle_lower > best_squared:
            self.pruned_triangle_candidates += 1
            continue
          self._consume_exact_evaluation()
          distance = _point_triangle_distance(
              point,
              self.index.triangles_mm[triangle_index],
          )
          if not math.isfinite(distance) or distance < 0.0:
            raise _CaseFailure("obj_parse_failed", "OBJ mesh distance is invalid")
          candidate = (
              distance,
              self.index.triangle_keys[triangle_index],
              triangle_index,
          )
          if best_key is None or candidate < best_key:
            best_key = candidate
            best_distance = distance
            distance_upper = math.nextafter(distance, math.inf)
            squared_upper = distance_upper * distance_upper
            best_squared = (
                squared_upper
                if not math.isfinite(squared_upper)
                else math.nextafter(squared_upper, math.inf)
            )
    finally:
      exact_this_query = (
          self.endpoint_budget.exact_triangle_evaluations - exact_before
      )
      self.maximum_exact_evaluations_per_query = max(
          self.maximum_exact_evaluations_per_query,
          exact_this_query,
      )
    if best_key is None or not math.isfinite(best_distance):
      raise _CaseFailure("obj_parse_failed", "OBJ mesh distance is invalid")
    self.completed_queries += 1
    return best_distance

  def receipt(self, *, all_reverse_samples_checked: bool) -> dict[str, Any]:
    failure = self._cap_failure
    required_lower_bound = (
        failure.required_lower_bound
        if failure is not None
        else self.endpoint_budget.exact_triangle_evaluations
    )
    return {
        "schema_version": _TRIANGLE_BVH_WORK_SCHEMA_VERSION,
        "algorithm": _TRIANGLE_BVH_ALGORITHM,
        "triangle_count": len(self.index.triangles_mm),
        "bvh_node_count": self.index.node_count,
        "bvh_leaf_count": self.index.leaf_count,
        "bvh_max_depth": self.index.maximum_depth,
        "bvh_leaf_size": self.index.leaf_size,
        "started_reverse_sample_query_count": self.started_queries,
        "completed_reverse_sample_query_count": self.completed_queries,
        "exact_triangle_evaluation_count": (
            self.endpoint_budget.exact_triangle_evaluations
        ),
        "attempted_triangle_evaluation_count": (
            self.endpoint_budget.attempted_triangle_evaluations
        ),
        "required_triangle_evaluation_count_lower_bound": (
            required_lower_bound
        ),
        "brute_force_triangle_evaluation_upper_bound": (
            self.started_queries * len(self.index.triangles_mm)
        ),
        "aabb_lower_bound_evaluation_count": self.aabb_evaluations,
        "pruned_triangle_candidate_count": self.pruned_triangle_candidates,
        "maximum_exact_triangle_evaluations_per_query": (
            self.maximum_exact_evaluations_per_query
        ),
        "caps": {
            "endpoint": self.endpoint_budget.maximum,
            "case": self.case_budget.maximum,
            "run": self.run_budget.maximum,
        },
        "scope_counts": {
            budget.scope: budget.receipt() for budget in self.budgets
        },
        "work_cap_exhausted": failure is not None,
        "exhausted_scope": failure.scope if failure is not None else None,
        "minimum_new_triangle_evaluations_at_failure": (
            failure.minimum_new_evaluations if failure is not None else None
        ),
        "all_reverse_samples_checked": bool(all_reverse_samples_checked),
    }


def _distance_to_obj_mesh(
    point: np.ndarray,
    engine: _TriangleMeshDistanceEngine,
) -> float:
  return engine.distance(point)


def _boundary_point_key(
    point: np.ndarray,
    quantization_mm: float,
) -> tuple[int, int, int]:
  values = np.asarray(point, dtype=float)
  if values.shape != (3,) or not np.all(np.isfinite(values)):
    raise ValueError("boundary point is not a finite 3-vector")
  return tuple(int(value) for value in np.rint(values / quantization_mm))


def _boundary_loop_count(
    edge_endpoints: Sequence[
        tuple[tuple[int, int, int], tuple[int, int, int]]
    ],
    *,
    invalid_status: str,
) -> int:
  if not edge_endpoints:
    raise _CaseFailure(invalid_status, "boundary contains no edges")
  degrees: Counter[tuple[int, int, int]] = Counter()
  neighbours: dict[
      tuple[int, int, int], set[tuple[int, int, int]]
  ] = defaultdict(set)
  for start, end in edge_endpoints:
    if start == end:
      degrees[start] += 2
      neighbours[start].add(start)
    else:
      degrees[start] += 1
      degrees[end] += 1
      neighbours[start].add(end)
      neighbours[end].add(start)
  if any(degree != 2 for degree in degrees.values()):
    raise _CaseFailure(
        invalid_status,
        "boundary edge graph is open or non-manifold",
    )
  remaining = set(degrees)
  components = 0
  while remaining:
    components += 1
    stack = [remaining.pop()]
    while stack:
      current = stack.pop()
      for neighbour in neighbours[current]:
        if neighbour in remaining:
          remaining.remove(neighbour)
          stack.append(neighbour)
  return components


def _linear_boundary_samples(
    start: np.ndarray,
    end: np.ndarray,
    step_mm: float,
) -> np.ndarray:
  length = float(np.linalg.norm(end - start))
  count = max(2, int(math.ceil(length / step_mm)) + 1)
  fractions = np.linspace(0.0, 1.0, count, dtype=float)[:, None]
  return start[None, :] + fractions * (end - start)[None, :]


def _obj_boundary_profile(
    group: _ObjFaceGroup,
    tolerances: MappingTolerances,
    cq: Any,
) -> _BoundaryProfile:
  uses: dict[
      tuple[tuple[int, int, int], tuple[int, int, int]],
      list[tuple[np.ndarray, np.ndarray]],
  ] = defaultdict(list)
  edge_use_count = 0
  for triangle in group.triangles_mm:
    for first, second in ((0, 1), (1, 2), (2, 0)):
      edge_use_count += 1
      if edge_use_count > tolerances.maximum_boundary_edge_uses:
        raise _CaseFailure(
            "boundary_sampling_budget_exceeded",
            "OBJ boundary edge-use budget was exceeded",
        )
      start = np.asarray(triangle[first], dtype=float)
      end = np.asarray(triangle[second], dtype=float)
      start_key = _boundary_point_key(start, tolerances.boundary_quantization_mm)
      end_key = _boundary_point_key(end, tolerances.boundary_quantization_mm)
      if start_key == end_key:
        raise _CaseFailure(
            "obj_boundary_invalid",
            "OBJ boundary analysis found a zero-length triangle edge",
        )
      signature = tuple(sorted((start_key, end_key)))
      uses[signature].append((start, end))
  boundary_edges: list[tuple[np.ndarray, np.ndarray]] = []
  for rows in uses.values():
    if len(rows) == 1:
      boundary_edges.append(rows[0])
    elif len(rows) == 2:
      first_start = _boundary_point_key(
          rows[0][0], tolerances.boundary_quantization_mm
      )
      first_end = _boundary_point_key(
          rows[0][1], tolerances.boundary_quantization_mm
      )
      second_start = _boundary_point_key(
          rows[1][0], tolerances.boundary_quantization_mm
      )
      second_end = _boundary_point_key(
          rows[1][1], tolerances.boundary_quantization_mm
      )
      if first_start != second_end or first_end != second_start:
        raise _CaseFailure(
            "obj_boundary_invalid",
            "OBJ internal mesh edge orientations are inconsistent",
        )
    else:
      raise _CaseFailure(
          "obj_boundary_invalid",
          "OBJ boundary analysis found a non-manifold mesh edge",
      )
  edge_endpoints = [
      (
          _boundary_point_key(start, tolerances.boundary_quantization_mm),
          _boundary_point_key(end, tolerances.boundary_quantization_mm),
      )
      for start, end in boundary_edges
  ]
  loop_count = _boundary_loop_count(
      edge_endpoints,
      invalid_status="obj_boundary_invalid",
  )
  polylines = tuple(
      np.asarray([start, end], dtype=float) for start, end in boundary_edges
  )
  def refine_samples(step_mm: float) -> tuple[np.ndarray, float]:
    if not math.isfinite(step_mm) or step_mm <= 0.0:
      raise _CaseFailure(
          "obj_boundary_invalid",
          "OBJ boundary refinement step is invalid",
      )
    refined = tuple(
        _linear_boundary_samples(start, end, step_mm)
        for start, end in boundary_edges
    )
    sample_count = sum(len(points) for points in refined)
    if sample_count > tolerances.maximum_boundary_samples:
      raise _CaseFailure(
          "boundary_sampling_budget_exceeded",
          "OBJ boundary sample budget was exceeded",
      )
    maximum_gap = max(
        float(np.linalg.norm(end - start)) / (len(points) - 1)
        for (start, end), points in zip(boundary_edges, refined)
    )
    return np.concatenate(refined, axis=0), maximum_gap

  samples_mm, maximum_sample_gap = refine_samples(
      tolerances.boundary_sampling_step_mm
  )
  length = float(
      sum(np.linalg.norm(end - start) for start, end in boundary_edges)
  )
  if not math.isfinite(length) or length <= 0.0:
    raise _CaseFailure("obj_boundary_invalid", "OBJ boundary length is invalid")
  try:
    boundary_shape = cq.Compound.makeCompound(
        [
            cq.Edge.makeLine(
                cq.Vector(*(float(value) for value in start)),
                cq.Vector(*(float(value) for value in end)),
            )
            for start, end in boundary_edges
        ]
    )
  except Exception as error:
    raise _CaseFailure(
        "obj_boundary_invalid",
        "OBJ boundary line compound construction failed",
    ) from error
  return _BoundaryProfile(
      edge_count=len(boundary_edges),
      loop_count=loop_count,
      length_mm=length,
      polylines_mm=polylines,
      samples_mm=samples_mm,
      boundary_shape=boundary_shape,
      maximum_sample_gap_mm=maximum_sample_gap,
      sample_step_mm=tolerances.boundary_sampling_step_mm,
      refine_samples=refine_samples,
      normalization_checks={},
  )


def _occ_wire_edge_uses(face: Any, cq: Any) -> list[Any]:
  try:
    from OCP.BRepTools import BRepTools_WireExplorer  # type: ignore
    from OCP.TopoDS import TopoDS  # type: ignore

    edge_cast = getattr(TopoDS, "Edge", None) or getattr(TopoDS, "Edge_s")
    result: list[Any] = []
    for wire in face.Wires():
      explorer = BRepTools_WireExplorer(wire.wrapped, face.wrapped)
      while explorer.More():
        result.append(cq.Edge(edge_cast(explorer.Current())))
        explorer.Next()
    return result
  except Exception as error:
    raise _CaseFailure(
        "occ_boundary_invalid",
        "OCC boundary edge-use traversal failed",
    ) from error


def _edge_length_samples(
    edge: Any,
    *,
    length_mm: float,
    closed: bool,
    step_mm: float,
) -> tuple[np.ndarray, float]:
  if closed:
    requested_count = max(3, int(math.ceil(length_mm / step_mm)))
  else:
    segment_count = max(1, int(math.ceil(length_mm / step_mm)))
    requested_count = segment_count + 1
  try:
    from OCP.BRepAdaptor import BRepAdaptor_Curve  # type: ignore
    from OCP.GCPnts import GCPnts_AbscissaPoint  # type: ignore

    curve = BRepAdaptor_Curve(edge.wrapped)
    first_parameter = float(curve.FirstParameter())
    last_parameter = float(curve.LastParameter())
    _, raw_parameters = edge.sample(requested_count)
    parameter_scale = max(1.0, abs(first_parameter), abs(last_parameter))
    parameter_tolerance = 1e-12 * parameter_scale
    parameters = sorted(float(value) for value in raw_parameters)
    if not parameters or abs(parameters[0] - first_parameter) > parameter_tolerance:
      parameters.insert(0, first_parameter)
    else:
      parameters[0] = first_parameter
    if closed:
      parameters = [
          value
          for value in parameters
          if abs(value - last_parameter) > parameter_tolerance
      ]
    elif abs(parameters[-1] - last_parameter) > parameter_tolerance:
      parameters.append(last_parameter)
    else:
      parameters[-1] = last_parameter
    deduplicated: list[float] = []
    for parameter in parameters:
      if (
          not deduplicated
          or abs(parameter - deduplicated[-1]) > parameter_tolerance
      ):
        deduplicated.append(parameter)
    parameters = deduplicated
    if len(parameters) < (3 if closed else 2):
      raise ValueError("edge sampler returned too few unique parameters")
    points = np.asarray(
        [
            [
                float(curve.Value(parameter).X()),
                float(curve.Value(parameter).Y()),
                float(curve.Value(parameter).Z()),
            ]
            for parameter in parameters
        ],
        dtype=float,
    )
    intervals = list(zip(parameters, parameters[1:]))
    if closed:
      intervals.append((parameters[-1], last_parameter))
    arc_gaps = [
        float(GCPnts_AbscissaPoint.Length_s(curve, start, end))
        for start, end in intervals
    ]
    maximum_gap = max(arc_gaps)
  except Exception as error:
    raise _CaseFailure(
        "occ_boundary_invalid",
        "OCC boundary edge arc-length sampling failed",
    ) from error
  if points.shape != (len(parameters), 3) or not np.all(np.isfinite(points)):
    raise _CaseFailure(
        "occ_boundary_invalid",
        "OCC boundary edge produced malformed arc-length samples",
    )
  if not closed:
    vertex_points = np.asarray(
        [
            [float(value) for value in vertex.toTuple()]
            for vertex in edge.Vertices()
        ],
        dtype=float,
    )
    direct = float(
        np.linalg.norm(points[0] - vertex_points[0])
        + np.linalg.norm(points[-1] - vertex_points[1])
    )
    reverse = float(
        np.linalg.norm(points[0] - vertex_points[1])
        + np.linalg.norm(points[-1] - vertex_points[0])
    )
    endpoints = vertex_points if direct <= reverse else vertex_points[::-1]
    points[0] = endpoints[0]
    points[-1] = endpoints[1]
  if (
      not math.isfinite(maximum_gap)
      or maximum_gap <= 0.0
      or maximum_gap > length_mm
  ):
    raise _CaseFailure(
        "occ_boundary_invalid",
        "OCC boundary sampler produced an invalid exact arc gap",
    )
  return points, maximum_gap


def _occ_boundary_profile(
    mapped_faces: Sequence[_OccFace],
    tolerances: MappingTolerances,
    cq: Any,
) -> _BoundaryProfile:
  cad_faces = [face.face for face in mapped_faces]
  if not cad_faces:
    raise _CaseFailure("occ_boundary_invalid", "mapped OCC face union is empty")
  try:
    normalized = (
        cad_faces[0]
        if len(cad_faces) == 1
        else cad_faces[0].fuse(
            *cad_faces[1:],
            tol=tolerances.boundary_boolean_tolerance_mm,
        )
    )
    input_area = float(sum(face.Area() for face in cad_faces))
    normalized_area = float(normalized.Area())
  except Exception as error:
    raise _CaseFailure(
        "occ_boundary_invalid",
        "mapped OCC faces could not be boolean-normalized for union boundary analysis",
    ) from error
  if (
      not math.isfinite(input_area)
      or not math.isfinite(normalized_area)
      or min(input_area, normalized_area) <= 0.0
  ):
    raise _CaseFailure(
        "occ_boundary_invalid",
        "mapped OCC face-union area is invalid",
    )
  union_area_relative_error = abs(input_area - normalized_area) / max(
      input_area,
      normalized_area,
  )
  if union_area_relative_error > tolerances.boundary_union_area_relative_error:
    raise _CaseFailure(
        "occ_boundary_invalid",
        "mapped OCC faces overlap or lose area during boundary normalization",
    )
  try:
    from OCP.BRep import BRep_Tool  # type: ignore
  except Exception as error:
    raise _CaseFailure(
        "occ_boundary_invalid",
        "OCC degenerate-edge classifier is unavailable",
    ) from error
  use_buckets: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
  edge_use_count = 0
  for face in normalized.Faces():
    for edge in _occ_wire_edge_uses(face, cq):
      edge_use_count += 1
      if edge_use_count > tolerances.maximum_boundary_edge_uses:
        raise _CaseFailure(
            "boundary_sampling_budget_exceeded",
            "OCC boundary edge-use budget was exceeded",
        )
      try:
        length = float(edge.Length())
        vertices = edge.Vertices()
        if not math.isfinite(length):
          raise ValueError("edge length is invalid")
        if length <= tolerances.boundary_quantization_mm:
          if BRep_Tool.Degenerated_s(edge.wrapped):
            continue
          raise ValueError("non-degenerate edge length is invalid")
        closed = bool(edge.IsClosed())
        if (closed and len(vertices) not in (1, 2)) or (
            not closed and len(vertices) != 2
        ):
          raise ValueError("edge has invalid closure/vertex cardinality")
        orientation = edge.wrapped.Orientation()
        hash_code = int(edge.hashCode())
      except Exception as error:
        raise _CaseFailure(
            "occ_boundary_invalid",
            "OCC boundary edge geometry could not be sampled",
        ) from error
      row = {
          "edge": edge,
          "closed": closed,
          "length": length,
          "orientation": orientation,
          "vertices": vertices,
      }
      for rows in use_buckets[hash_code]:
        if edge.isSame(rows[0]["edge"]):
          rows.append(row)
          break
      else:
        use_buckets[hash_code].append([row])
  boundary_edges: list[dict[str, Any]] = []
  for bucket in use_buckets.values():
    for rows in bucket:
      if len(rows) == 1:
        boundary_edges.append(rows[0])
      elif len(rows) == 2:
        if rows[0]["orientation"] == rows[1]["orientation"]:
          raise _CaseFailure(
              "occ_boundary_invalid",
              "mapped OCC face union repeats a same-oriented internal edge",
          )
      else:
        raise _CaseFailure(
            "occ_boundary_invalid",
            "mapped OCC face union has a non-manifold topological edge",
        )
  polylines: list[np.ndarray] = []
  samples: list[np.ndarray] = []
  edge_endpoints: list[
      tuple[tuple[int, int, int], tuple[int, int, int]]
  ] = []
  maximum_sample_gap = 0.0
  for row in boundary_edges:
    closed = bool(row["closed"])
    length = float(row["length"])
    points, sample_gap = _edge_length_samples(
        row["edge"],
        length_mm=length,
        closed=closed,
        step_mm=tolerances.boundary_sampling_step_mm,
    )
    maximum_sample_gap = max(maximum_sample_gap, sample_gap)
    samples.append(points)
    if closed:
      polylines.append(np.concatenate((points, points[:1]), axis=0))
      endpoint = _boundary_point_key(
          points[0],
          tolerances.boundary_quantization_mm,
      )
      edge_endpoints.append((endpoint, endpoint))
    else:
      vertex_points = np.asarray(
          [
              [float(value) for value in vertex.toTuple()]
              for vertex in row["vertices"]
          ],
          dtype=float,
      )
      polylines.append(points)
      edge_endpoints.append(
          (
              _boundary_point_key(
                  vertex_points[0], tolerances.boundary_quantization_mm
              ),
              _boundary_point_key(
                  vertex_points[1], tolerances.boundary_quantization_mm
              ),
          )
      )
  sample_count = sum(len(points) for points in samples)
  if sample_count > tolerances.maximum_boundary_samples:
    raise _CaseFailure(
        "boundary_sampling_budget_exceeded",
        "OCC boundary sample budget was exceeded",
    )
  def refine_samples(step_mm: float) -> tuple[np.ndarray, float]:
    if not math.isfinite(step_mm) or step_mm <= 0.0:
      raise _CaseFailure(
          "occ_boundary_invalid",
          "OCC boundary refinement step is invalid",
      )
    refined: list[np.ndarray] = []
    maximum_gap = 0.0
    for boundary_row in boundary_edges:
      points, sample_gap = _edge_length_samples(
          boundary_row["edge"],
          length_mm=float(boundary_row["length"]),
          closed=bool(boundary_row["closed"]),
          step_mm=step_mm,
      )
      refined.append(points)
      maximum_gap = max(maximum_gap, sample_gap)
    refined_count = sum(len(points) for points in refined)
    if refined_count > tolerances.maximum_boundary_samples:
      raise _CaseFailure(
          "boundary_sampling_budget_exceeded",
          "OCC boundary sample budget was exceeded",
      )
    return np.concatenate(refined, axis=0), maximum_gap
  loop_count = _boundary_loop_count(
      edge_endpoints,
      invalid_status="occ_boundary_invalid",
  )
  length = float(sum(float(row["length"]) for row in boundary_edges))
  if not math.isfinite(length) or length <= 0.0:
    raise _CaseFailure("occ_boundary_invalid", "OCC boundary length is invalid")
  try:
    boundary_shape = cq.Compound.makeCompound(
        [row["edge"] for row in boundary_edges]
    )
  except Exception as error:
    raise _CaseFailure(
        "occ_boundary_invalid",
        "OCC union-boundary compound construction failed",
    ) from error
  return _BoundaryProfile(
      edge_count=len(boundary_edges),
      loop_count=loop_count,
      length_mm=length,
      polylines_mm=tuple(polylines),
      samples_mm=np.concatenate(samples, axis=0),
      boundary_shape=boundary_shape,
      maximum_sample_gap_mm=maximum_sample_gap,
      sample_step_mm=tolerances.boundary_sampling_step_mm,
      refine_samples=refine_samples,
      normalization_checks={
          "occ_boundary_union_mode": "boolean_normalize_then_topological_edge_use",
          "occ_boundary_input_face_area_mm2": input_area,
          "occ_boundary_normalized_union_area_mm2": normalized_area,
          "occ_boundary_union_area_relative_error": union_area_relative_error,
          "occ_boundary_union_area_relative_tolerance": (
              tolerances.boundary_union_area_relative_error
          ),
          "occ_boundary_boolean_tolerance_mm": (
              tolerances.boundary_boolean_tolerance_mm
          ),
      },
  )


def _maximum_boundary_distance(
    source: _BoundaryProfile,
    target: _BoundaryProfile,
    tolerances: MappingTolerances,
    cq: Any,
) -> tuple[float, float]:
  samples = source.samples_mm
  maximum_gap = source.maximum_sample_gap_mm
  step_mm = source.sample_step_mm
  refinement_round_count = 0
  cumulative_pair_evaluations = 0
  rounds: list[dict[str, Any]] = []
  while True:
    round_pair_evaluations = len(samples) * target.edge_count
    cumulative_pair_evaluations += round_pair_evaluations
    if cumulative_pair_evaluations > tolerances.maximum_boundary_distance_pairs:
      raise _CaseFailure(
          "boundary_sampling_budget_exceeded",
          "adaptive boundary distance-pair budget was exceeded",
      )
    sampled_maximum = 0.0
    try:
      for point in samples:
        vertex = cq.Vertex.makeVertex(*(float(value) for value in point))
        distance = float(vertex.distance(target.boundary_shape))
        if not math.isfinite(distance) or distance < 0.0:
          raise ValueError("exact boundary distance is invalid")
        sampled_maximum = max(sampled_maximum, distance)
    except Exception as error:
      raise _CaseFailure(
          "boundary_geometry_mismatch",
          "exact boundary distance computation failed",
      ) from error
    half_gap_upper = math.nextafter(maximum_gap / 2.0, math.inf)
    certified_upper_bound = _outward_up_sum(
        sampled_maximum,
        half_gap_upper,
    )
    if not math.isfinite(certified_upper_bound):
      raise _CaseFailure(
          "boundary_geometry_mismatch",
          "boundary distance computation was non-finite",
      )
    round_receipt = {
        "round": refinement_round_count,
        "sample_step_mm": step_mm,
        "sample_count": len(samples),
        "maximum_exact_sample_distance_mm": sampled_maximum,
        "maximum_sample_gap_mm": maximum_gap,
        "certified_distance_upper_bound_mm": certified_upper_bound,
        "exact_sample_to_boundary_query_count": len(samples),
        "conservative_sample_edge_pair_count": round_pair_evaluations,
    }
    rounds.append(round_receipt)
    if sampled_maximum > tolerances.boundary_distance_mm:
      decision = "certified_fail_sampled_lower_bound"
      break
    if certified_upper_bound <= tolerances.boundary_distance_mm:
      decision = "certified_pass_upper_bound"
      break
    next_step = step_mm / 2.0
    refined_samples, refined_gap = source.refine_samples(next_step)
    if (
        refined_samples.ndim != 2
        or refined_samples.shape[1:] != (3,)
        or len(refined_samples) <= len(samples)
        or not np.all(np.isfinite(refined_samples))
        or not math.isfinite(refined_gap)
        or refined_gap <= 0.0
        or refined_gap >= maximum_gap
    ):
      raise _CaseFailure(
          "boundary_geometry_mismatch",
          "adaptive boundary refinement did not tighten the proof interval",
      )
    samples = refined_samples
    maximum_gap = refined_gap
    step_mm = next_step
    refinement_round_count += 1
  return sampled_maximum, certified_upper_bound, {
      "schema_version": "adaptive_exact_boundary_distance_proof.v1",
      "decision": decision,
      "locked_boundary_distance_tolerance_mm": (
          tolerances.boundary_distance_mm
      ),
      "refinement_round_count": refinement_round_count,
      "initial_sample_step_mm": source.sample_step_mm,
      "final_sample_step_mm": step_mm,
      "initial_sample_count": len(source.samples_mm),
      "final_sample_count": len(samples),
      "cumulative_conservative_sample_edge_pair_count": (
          cumulative_pair_evaluations
      ),
      "rounds": rounds,
  }


def _boundary_checks(
    group: _ObjFaceGroup,
    mapped_faces: Sequence[_OccFace],
    tolerances: MappingTolerances,
    cq: Any,
) -> tuple[str | None, dict[str, Any]]:
  obj = _obj_boundary_profile(group, tolerances, cq)
  occ = _occ_boundary_profile(mapped_faces, tolerances, cq)
  checks: dict[str, Any] = {
      "obj_boundary_edge_count": obj.edge_count,
      "mapped_occ_boundary_edge_count": occ.edge_count,
      "obj_boundary_loop_count": obj.loop_count,
      "mapped_occ_boundary_loop_count": occ.loop_count,
      "obj_boundary_length_mm": obj.length_mm,
      "mapped_occ_boundary_length_mm": occ.length_mm,
      "obj_boundary_sample_count": len(obj.samples_mm),
      "mapped_occ_boundary_sample_count": len(occ.samples_mm),
      **occ.normalization_checks,
  }
  if obj.loop_count != occ.loop_count:
    return "boundary_topology_mismatch", checks
  obj_to_occ_sampled, obj_to_occ, obj_to_occ_proof = _maximum_boundary_distance(
      obj, occ, tolerances, cq
  )
  occ_to_obj_sampled, occ_to_obj, occ_to_obj_proof = _maximum_boundary_distance(
      occ, obj, tolerances, cq
  )
  length_relative_error = abs(obj.length_mm - occ.length_mm) / max(
      obj.length_mm,
      occ.length_mm,
  )
  checks.update(
      {
          "sampled_max_obj_to_occ_boundary_distance_mm": obj_to_occ_sampled,
          "sampled_max_occ_to_obj_boundary_distance_mm": occ_to_obj_sampled,
          "maximum_obj_to_occ_boundary_distance_mm": obj_to_occ,
          "maximum_occ_to_obj_boundary_distance_mm": occ_to_obj,
          "boundary_distance_mode": (
              "adaptive_exact_sample_to_edge_plus_half_arc_gap_upper_bound"
          ),
          "obj_to_occ_boundary_distance_proof": obj_to_occ_proof,
          "occ_to_obj_boundary_distance_proof": occ_to_obj_proof,
          "boundary_distance_tolerance_mm": tolerances.boundary_distance_mm,
          "boundary_length_relative_error": length_relative_error,
          "boundary_length_relative_tolerance": (
              tolerances.boundary_length_relative_error
          ),
      }
  )
  if (
      max(obj_to_occ, occ_to_obj) > tolerances.boundary_distance_mm
      or length_relative_error > tolerances.boundary_length_relative_error
  ):
    return "boundary_geometry_mismatch", checks
  return None, checks


def _occ_face_reverse_samples(
    face: _OccFace,
    tolerances: MappingTolerances,
    *,
    distance_engine: _TriangleMeshDistanceEngine | None = None,
) -> np.ndarray:
  try:
    vertices, triangles = face.face.tessellate(tolerances.occ_tessellation_mm)
  except Exception as error:
    raise _CaseFailure(
        "reverse_coverage_failed",
        "OCC face tessellation failed during reverse coverage",
    ) from error
  try:
    triangle_count = len(triangles)
  except Exception as error:
    raise _CaseFailure(
        "reverse_coverage_failed",
        "OCC face tessellation has no finite triangle count",
    ) from error
  if triangle_count < 1:
    raise _CaseFailure(
        "reverse_coverage_failed",
        "OCC face produced no reverse-coverage triangles",
    )
  if triangle_count > tolerances.maximum_occ_reverse_samples_per_face:
    raise _CaseFailure(
        "reverse_sampling_budget_exceeded",
        "OCC face exceeds the locked complete reverse-sampling budget",
    )
  if distance_engine is not None:
    # Every reverse sample needs at least one exact triangle evaluation.  Check
    # the hierarchical work caps before allocating points/indices/samples.
    distance_engine.require_minimum_evaluations(triangle_count)
  try:
    points = np.asarray(
        [
            [float(value) for value in vertex.toTuple()]
            for vertex in vertices
        ],
        dtype=float,
    )
    indices = np.asarray(triangles, dtype=int)
    samples = np.mean(points[indices], axis=1)
  except Exception as error:
    raise _CaseFailure(
        "reverse_coverage_failed",
        "OCC face tessellation failed during reverse coverage",
    ) from error
  if (
      samples.ndim != 2
      or samples.shape[1] != 3
      or len(samples) == 0
      or not np.all(np.isfinite(samples))
  ):
    raise _CaseFailure(
        "reverse_coverage_failed",
        "OCC face produced no finite reverse-coverage samples",
    )
  if len(samples) != triangle_count:
    raise _CaseFailure(
        "reverse_coverage_failed",
        "OCC reverse sample count changed during materialization",
    )
  return samples


def _plane_support(face: _OccFace) -> tuple[np.ndarray, np.ndarray] | None:
  if face.surface_type != "PLANE":
    return None
  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomAbs import GeomAbs_Plane  # type: ignore

    adaptor = BRepAdaptor_Surface(face.face.wrapped)
    if adaptor.GetType() != GeomAbs_Plane:
      return None
    plane = adaptor.Plane()
    direction = plane.Axis().Direction()
    location = plane.Location()
    normal = np.asarray(
        [direction.X(), direction.Y(), direction.Z()],
        dtype=float,
    )
    point = np.asarray([location.X(), location.Y(), location.Z()], dtype=float)
    normal /= np.linalg.norm(normal)
    return normal, point
  except Exception:
    return None


def _cylinder_support(
    face: _OccFace,
) -> tuple[np.ndarray, np.ndarray, float] | None:
  if face.surface_type != "CYLINDER":
    return None
  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomAbs import GeomAbs_Cylinder  # type: ignore

    adaptor = BRepAdaptor_Surface(face.face.wrapped)
    if adaptor.GetType() != GeomAbs_Cylinder:
      return None
    cylinder = adaptor.Cylinder()
    direction = cylinder.Axis().Direction()
    location = cylinder.Axis().Location()
    axis = np.asarray([direction.X(), direction.Y(), direction.Z()], dtype=float)
    origin = np.asarray([location.X(), location.Y(), location.Z()], dtype=float)
    axis /= np.linalg.norm(axis)
    return axis, origin, float(cylinder.Radius())
  except Exception:
    return None


def _sphere_support(
    face: _OccFace,
) -> tuple[np.ndarray, float] | None:
  if face.surface_type != "SPHERE":
    return None
  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomAbs import GeomAbs_Sphere  # type: ignore

    adaptor = BRepAdaptor_Surface(face.face.wrapped)
    if adaptor.GetType() != GeomAbs_Sphere:
      return None
    sphere = adaptor.Sphere()
    location = sphere.Location()
    center = np.asarray(
        [location.X(), location.Y(), location.Z()],
        dtype=float,
    )
    return center, float(sphere.Radius())
  except Exception:
    return None


def _same_analytic_support(
    reference: _OccFace,
    candidate: _OccFace,
    tolerance_mm: float,
) -> bool | None:
  """Return same/different support, or None when support is not auditable."""

  if reference.surface_type != candidate.surface_type:
    return False
  reference_plane = _plane_support(reference)
  candidate_plane = _plane_support(candidate)
  if reference_plane is not None and candidate_plane is not None:
    normal_a, point_a = reference_plane
    normal_b, point_b = candidate_plane
    if abs(float(np.dot(normal_a, normal_b))) < 1.0 - 1e-8:
      return False
    return abs(float(np.dot(normal_a, point_b - point_a))) <= tolerance_mm
  reference_cylinder = _cylinder_support(reference)
  candidate_cylinder = _cylinder_support(candidate)
  if reference_cylinder is not None and candidate_cylinder is not None:
    axis_a, origin_a, radius_a = reference_cylinder
    axis_b, origin_b, radius_b = candidate_cylinder
    if abs(float(np.dot(axis_a, axis_b))) < 1.0 - 1e-8:
      return False
    axis_distance = float(np.linalg.norm(np.cross(origin_b - origin_a, axis_a)))
    return (
        axis_distance <= tolerance_mm
        and abs(radius_a - radius_b) <= tolerance_mm
    )
  reference_sphere = _sphere_support(reference)
  candidate_sphere = _sphere_support(candidate)
  if reference_sphere is not None and candidate_sphere is not None:
    center_a, radius_a = reference_sphere
    center_b, radius_b = candidate_sphere
    return (
        float(np.linalg.norm(center_b - center_a)) <= tolerance_mm
        and abs(radius_a - radius_b) <= tolerance_mm
    )
  return None


def _analytic_zero_area_exclusion_proof(
    reference_faces: Sequence[_OccFace],
    candidate: _OccFace,
    tolerance_mm: float,
) -> dict[str, Any] | None:
  """Prove an extra same-type regular analytic face has zero area overlap."""

  if (
      not reference_faces
      or candidate.surface_type not in {"PLANE", "CYLINDER", "SPHERE"}
      or any(face.surface_type != candidate.surface_type for face in reference_faces)
  ):
    return None
  relations = [
      _same_analytic_support(reference, candidate, tolerance_mm)
      for reference in reference_faces
  ]
  if not relations or any(relation is not False for relation in relations):
    return None
  return {
      "candidate_occ_face_index": candidate.index,
      "reference_occ_face_indices": sorted(face.index for face in reference_faces),
      "surface_type": candidate.surface_type,
      "support_relation": "provably_distinct",
      "analytic_support_distance_tolerance_mm": tolerance_mm,
      "proof": (
          "distinct_same_type_regular_analytic_supports_have_zero_area_intersection"
      ),
  }


def _bbox_tolerance(
    minimum: np.ndarray,
    maximum: np.ndarray,
    tolerances: MappingTolerances,
) -> float:
  diagonal = float(np.linalg.norm(maximum - minimum))
  return tolerances.bbox_absolute_mm + tolerances.bbox_relative_error * diagonal


def _bbox_residual(
    minimum_a: np.ndarray,
    maximum_a: np.ndarray,
    minimum_b: np.ndarray,
    maximum_b: np.ndarray,
) -> float:
  return float(
      max(
          np.max(np.abs(minimum_a - minimum_b)),
          np.max(np.abs(maximum_a - maximum_b)),
      )
  )


def _point_face_distance(point_mm: np.ndarray, face: _OccFace, cq: Any) -> float:
  vertex = cq.Vertex.makeVertex(*(float(value) for value in point_mm))
  distance = float(face.face.distance(vertex))
  if not math.isfinite(distance) or distance < 0.0:
    raise _CaseFailure("occ_distance_failed", "OCC returned an invalid point-face distance")
  return distance


def _failed_mapping(
    *,
    status: str,
    fusion_index: int,
    message: str,
    checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  return {
      "status": status,
      "mapping_mode": "authoritative_obj_face_group",
      "acceptance_basis": DISTANCE_ACCEPTANCE_BASIS,
      "fusion_face_index": fusion_index,
      "raw_occ_face_indices": [],
      "source_face_signature_sha256s": [],
      "checks": dict(checks or {}),
      "reason": message,
  }


def _forward_work_receipts(
    mappings: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
  rows: list[Mapping[str, Any]] = []
  for mapping in mappings:
    checks = mapping.get("checks")
    if not isinstance(checks, Mapping):
      continue
    work = checks.get("forward_distance_work")
    if isinstance(work, Mapping):
      rows.append(work)
  return rows


def _forward_work_aggregates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
  return {
      "mapping_count_with_work_receipt": len(rows),
      "work_cap_exhausted_mapping_count": sum(
          work.get("work_cap_exhausted") is True for work in rows
      ),
      "sample_candidate_distance_call_upper_bound": sum(
          int(work.get("sample_candidate_distance_call_upper_bound", 0))
          for work in rows
      ),
      "source_point_candidate_distance_call_upper_bound": sum(
          int(
              work.get(
                  "source_point_candidate_distance_call_upper_bound",
                  0,
              )
          )
          for work in rows
      ),
      "total_proof_distance_call_upper_bound": sum(
          int(work.get("total_proof_distance_call_upper_bound", 0))
          for work in rows
      ),
      "attempted_distance_call_count_in_mapping_receipts": sum(
          int(work.get("attempted_distance_call_count", 0)) for work in rows
      ),
      "completed_distance_call_count_in_mapping_receipts": sum(
          int(work.get("completed_distance_call_count", 0)) for work in rows
      ),
      "aabb_lower_bound_evaluation_count": sum(
          int(work.get("aabb_lower_bound_evaluation_count", 0))
          for work in rows
      ),
      "bvh_node_lower_bound_evaluation_count": sum(
          int(work.get("bvh_node_lower_bound_evaluation_count", 0))
          for work in rows
      ),
      "bvh_leaf_face_lower_bound_evaluation_count": sum(
          int(work.get("bvh_leaf_face_lower_bound_evaluation_count", 0))
          for work in rows
      ),
      "bvh_pruned_occ_face_candidate_count": sum(
          int(work.get("bvh_pruned_occ_face_candidate_count", 0))
          for work in rows
      ),
      "bvh_max_depth": max(
          (int(work.get("bvh_max_depth", 0)) for work in rows),
          default=0,
      ),
      "bvh_max_traversal_depth_visited": max(
          (
              int(work.get("bvh_max_traversal_depth_visited", 0))
              for work in rows
          ),
          default=0,
      ),
      "analytic_support_lb_queries": sum(
          int(work.get("analytic_support_lb_queries", 0)) for work in rows
      ),
      "analytic_trimmed_proof_count": sum(
          int(work.get("analytic_trimmed_proof_count", 0)) for work in rows
      ),
      "analytic_guard_fallback_count": sum(
          int(work.get("analytic_guard_fallback_count", 0)) for work in rows
      ),
      "persistent_kernel_performs": sum(
          int(work.get("persistent_kernel_performs", 0)) for work in rows
      ),
      "ambiguous_proof_stop_count": sum(
          int(work.get("ambiguous_proof_stop_count", 0)) for work in rows
      ),
      "ambiguous_proof_pruned_occ_face_candidate_count": sum(
          int(
              work.get(
                  "ambiguous_proof_pruned_occ_face_candidate_count",
                  0,
              )
          )
          for work in rows
      ),
      "pruned_occ_face_candidate_count": sum(
          int(work.get("pruned_occ_face_candidate_count", 0))
          for work in rows
      ),
      "exact_occ_distance_call_count_in_mapping_receipts": sum(
          int(work.get("exact_occ_distance_call_count", 0)) for work in rows
      ),
  }


def _map_obj_group_to_occ_distance(
    *,
    group: _ObjFaceGroup,
    entity: Mapping[str, Any],
    occ_faces: Sequence[_OccFace],
    tolerances: MappingTolerances,
    cq: Any,
    case_forward_budget: _ForwardDistanceCallBudget | None = None,
    run_forward_budget: _ForwardDistanceCallBudget | None = None,
    case_triangle_budget: _TriangleEvaluationBudget | None = None,
    run_triangle_budget: _TriangleEvaluationBudget | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_case_id: str = "",
    progress_endpoint_part: str | None = None,
) -> dict[str, Any]:
  endpoint_started_at = time.perf_counter()
  fusion_index = group.fusion_face_index
  expected_surface = _surface_type(entity.get("surface_type"))
  if not expected_surface:
    return _failed_mapping(
        status="surface_type_missing",
        fusion_index=fusion_index,
        message="Fusion source endpoint lacks a surface type",
    )
  source_bbox_min, source_bbox_max = _entity_bbox_mm(entity)
  source_point = _entity_point_mm(entity)
  obj_source_bbox_residual = _bbox_residual(
      group.bbox_min_mm,
      group.bbox_max_mm,
      source_bbox_min,
      source_bbox_max,
  )
  bbox_tolerance = max(
      _bbox_tolerance(group.bbox_min_mm, group.bbox_max_mm, tolerances),
      _bbox_tolerance(source_bbox_min, source_bbox_max, tolerances),
  )
  if obj_source_bbox_residual > bbox_tolerance:
    return _failed_mapping(
        status="bbox_mismatch",
        fusion_index=fusion_index,
        message="authoritative OBJ group disagrees with Fusion source bbox",
        checks={
            "obj_source_bbox_max_abs_error_mm": obj_source_bbox_residual,
            "bbox_tolerance_mm": bbox_tolerance,
        },
    )
  candidates = [
      face
      for face in occ_faces
      if face.surface_type == expected_surface
      and np.all(face.bbox_max_mm >= group.bbox_min_mm - bbox_tolerance)
      and np.all(face.bbox_min_mm <= group.bbox_max_mm + bbox_tolerance)
  ]
  if not candidates:
    return _failed_mapping(
        status="surface_type_mismatch",
        fusion_index=fusion_index,
        message="no STEP/OCC face matches the Fusion surface type and bbox",
    )
  sample_count = _obj_sample_count(group)
  active_case_forward_budget = case_forward_budget or _ForwardDistanceCallBudget(
      scope="case",
      maximum=tolerances.maximum_forward_distance_calls_per_case,
  )
  active_run_forward_budget = run_forward_budget or _ForwardDistanceCallBudget(
      scope="run",
      maximum=tolerances.maximum_forward_distance_calls_per_run,
  )
  forward_engine = _ForwardDistanceCallEngine(
      maximum_endpoint_calls=(
          tolerances.maximum_forward_distance_calls_per_endpoint
      ),
      case_budget=active_case_forward_budget,
      run_budget=active_run_forward_budget,
      analytic_coverage_threshold_mm=tolerances.sample_distance_mm,
  )
  forward_engine.configure_forward_proof(
      sample_count=sample_count,
      candidates=candidates,
  )
  all_forward_samples_checked = False
  source_point_checked = False
  analytic_margin_disambiguated_sample_count = 0
  analytic_margin_disambiguation_proofs: list[dict[str, Any]] = []
  curved_chord_proofs: list[dict[str, Any]] = []

  def forward_geometry_proof_checks() -> dict[str, Any]:
    return {
        "analytic_margin_disambiguated_sample_count": (
            analytic_margin_disambiguated_sample_count
        ),
        "analytic_margin_disambiguation_proofs": (
            analytic_margin_disambiguation_proofs
        ),
        "curved_chord_proof_sample_count": len(curved_chord_proofs),
        "curved_chord_proof_triangle_indices": sorted(
            {int(proof["triangle_index"]) for proof in curved_chord_proofs}
        ),
        "maximum_curved_chord_error_upper_bound_mm": max(
            (
                float(proof["triangle_chord_error_upper_bound_mm"])
                for proof in curved_chord_proofs
            ),
            default=0.0,
        ),
        "curved_chord_proofs": curved_chord_proofs,
    }

  def failed_after_forward(
      *,
      status: str,
      message: str,
      checks: Mapping[str, Any] | None = None,
  ) -> dict[str, Any]:
    return _failed_mapping(
        status=status,
        fusion_index=fusion_index,
        message=message,
        checks={
            **forward_geometry_proof_checks(),
            **dict(checks or {}),
            "forward_distance_work": forward_engine.receipt(
                all_forward_samples_checked=all_forward_samples_checked,
                source_point_checked=source_point_checked,
            ),
        },
    )

  if sample_count > tolerances.maximum_obj_samples:
    return failed_after_forward(
        status="sampling_budget_exceeded",
        message="authoritative OBJ group exceeds the locked total sampling cap",
    )
  assignment_support: Counter[int] = Counter()
  minimum_margin: float | None = None
  maximum_best_distance = 0.0
  processed_samples = 0
  by_index = {face.index: face for face in candidates}

  def emit_forward_chunk_completed() -> None:
    bvh = forward_engine._face_bvh_index
    _emit_face_map_progress(
        progress_callback,
        event="face_map_forward_sample_chunk_completed",
        case_id=progress_case_id,
        endpoint_face_index=fusion_index,
        endpoint_part=progress_endpoint_part,
        processed_sample_count=(
            forward_engine.completed_forward_sample_queries
        ),
        total_sample_count=sample_count,
        exact_occ_distance_call_count=(
            forward_engine.endpoint_budget.completed_distance_calls
        ),
        pruned_occ_face_candidate_count=(
            forward_engine.pruned_occ_face_candidates
        ),
        elapsed_seconds=time.perf_counter() - endpoint_started_at,
        bvh_max_depth=0 if bvh is None else bvh.maximum_depth,
        bvh_max_traversal_depth_visited=(
            forward_engine.bvh_max_traversal_depth_visited
        ),
    )

  for sample_chunk in _obj_sample_chunks(
      group,
      maximum=tolerances.maximum_obj_samples,
      chunk_size=tolerances.forward_sample_chunk_size,
  ):
    rankings = forward_engine.rank_sample_chunk(
        sample_chunk,
        cq,
        minimum_margin_mm=tolerances.minimum_sample_margin_mm,
    )
    try:
      for sample, ranking in zip(sample_chunk, rankings, strict=True):
        processed_samples = forward_engine.completed_forward_sample_queries
        best_distance = ranking.best_distance_mm
        best_index = ranking.best_index
        maximum_best_distance = max(maximum_best_distance, best_distance)
        if best_distance > tolerances.sample_distance_mm:
          triangle_index = (processed_samples - 1) // 4
          chord_proof = _curved_triangle_chord_error_proof(
              group.triangles_mm[triangle_index],
              sample,
              by_index[best_index],
              locked_distance_mm=tolerances.sample_distance_mm,
          )
          if chord_proof is None:
            second_distance = (
                ranking.second_best_distance_lower_bound_mm
            )
            ranking_margin = (
                None
                if second_distance is None
                else _outward_down_difference(
                    second_distance,
                    best_distance,
                )
            )
            all_forward_samples_checked = processed_samples == sample_count
            emit_forward_chunk_completed()
            return failed_after_forward(
                status="sample_uncovered",
                message="an authoritative OBJ interior sample is not covered by STEP",
                checks={
                    "sample_count": sample_count,
                    "processed_sample_count": processed_samples,
                    "maximum_best_distance_mm": maximum_best_distance,
                    "sample_distance_tolerance_mm": tolerances.sample_distance_mm,
                    "best_occ_face_index": best_index,
                    "second_best_distance_lower_bound_mm": second_distance,
                    "sample_margin_lower_bound_mm": ranking_margin,
                    "candidate_ranking_unique_margin_safe": (
                        ranking_margin is None
                        or ranking_margin
                        >= tolerances.minimum_sample_margin_mm
                    ),
                    "candidate_occ_face_indices": sorted(by_index),
                },
            )
          curved_chord_proofs.append(
              {
                  **chord_proof,
                  "triangle_index": triangle_index,
                  "sample_ordinal": processed_samples - 1,
                  "exact_occ_distance_mm": best_distance,
              }
          )
        if ranking.second_best_distance_lower_bound_mm is not None:
          second_best_distance = ranking.second_best_distance_lower_bound_mm
          margin = _outward_down_difference(
              second_best_distance,
              best_distance,
          )
          if margin < tolerances.minimum_sample_margin_mm:
            second_index = ranking.second_best_index
            analytic_proof = (
                None
                if second_index is None
                else _analytic_zero_area_exclusion_proof(
                    [by_index[best_index]],
                    by_index[second_index],
                    tolerances.analytic_support_distance_mm,
                )
            )
            if analytic_proof is None:
              all_forward_samples_checked = processed_samples == sample_count
              emit_forward_chunk_completed()
              return failed_after_forward(
                  status="ambiguous_mapping",
                  message="OBJ sample has two indistinguishable OCC face candidates",
                  checks={
                      "sample_count": sample_count,
                      "processed_sample_count": processed_samples,
                      "best_distance_mm": best_distance,
                      "second_best_distance_mm": second_best_distance,
                      "sample_margin_mm": margin,
                      "minimum_sample_margin_mm": (
                          tolerances.minimum_sample_margin_mm
                      ),
                      "candidate_occ_face_indices": sorted(
                          face.index for face in candidates
                      ),
                  },
              )
            analytic_margin_disambiguated_sample_count += 1
            analytic_margin_disambiguation_proofs.append(
                {
                    **analytic_proof,
                    "best_occ_face_index": best_index,
                    "best_distance_mm": best_distance,
                    "second_best_distance_lower_bound_mm": second_best_distance,
                    "sample_margin_lower_bound_mm": margin,
                    "locked_minimum_sample_margin_mm": (
                        tolerances.minimum_sample_margin_mm
                    ),
                }
            )
          minimum_margin = (
              margin if minimum_margin is None else min(minimum_margin, margin)
          )
        assignment_support[best_index] += 1
    except _CaseFailure as error:
      processed_samples = forward_engine.completed_forward_sample_queries
      emit_forward_chunk_completed()
      return failed_after_forward(status=error.status, message=str(error))
    emit_forward_chunk_completed()
  if processed_samples != sample_count:
    raise _CaseFailure(
        "occ_mapping_failed",
        "chunked forward sampling did not consume every locked OBJ sample",
    )
  all_forward_samples_checked = True
  forward_indices = sorted(assignment_support)
  active_case_budget = case_triangle_budget or _TriangleEvaluationBudget(
      scope="case",
      maximum=tolerances.maximum_reverse_triangle_evaluations_per_case,
  )
  active_run_budget = run_triangle_budget or _TriangleEvaluationBudget(
      scope="run",
      maximum=tolerances.maximum_reverse_triangle_evaluations_per_run,
  )
  distance_engine = _TriangleMeshDistanceEngine(
      group.triangles_mm,
      maximum_endpoint_triangle_evaluations=(
          tolerances.maximum_reverse_triangle_evaluations_per_endpoint
      ),
      case_budget=active_case_budget,
      run_budget=active_run_budget,
  )
  reference_faces = [by_index[index] for index in forward_indices]
  analytic_zero_area_exclusion_proofs: list[dict[str, Any]] = []
  analytic_zero_area_preexcluded_indices: list[int] = []
  reverse_candidates: list[_OccFace] = []
  forward_index_set = set(forward_indices)
  for face in candidates:
    proof = (
        None
        if (
            not tolerances.development_enable_reverse_analytic_preexclusion
            or face.index in forward_index_set
        )
        else _analytic_zero_area_exclusion_proof(
            reference_faces,
            face,
            tolerances.analytic_support_distance_mm,
        )
    )
    if proof is None:
      reverse_candidates.append(face)
      continue
    analytic_zero_area_exclusion_proofs.append(proof)
    analytic_zero_area_preexcluded_indices.append(face.index)

  def reverse_analytic_proof_checks() -> dict[str, Any]:
    return {
        "analytic_zero_area_preexcluded_occ_face_indices": sorted(
            analytic_zero_area_preexcluded_indices
        ),
        "development_reverse_analytic_preexclusion_enabled": (
            tolerances.development_enable_reverse_analytic_preexclusion
        ),
        "analytic_zero_area_excluded_occ_face_indices": sorted(
            {
                int(proof["candidate_occ_face_index"])
                for proof in analytic_zero_area_exclusion_proofs
            }
        ),
        "analytic_zero_area_exclusion_proofs": (
            analytic_zero_area_exclusion_proofs
        ),
    }

  reverse_checks: dict[str, dict[str, Any]] = {}
  reverse_covered_indices: list[int] = []
  partial_reverse_indices: list[int] = []
  for face in reverse_candidates:
    exact_before = distance_engine.endpoint_budget.exact_triangle_evaluations
    aabb_before = distance_engine.aabb_evaluations
    pruned_before = distance_engine.pruned_triangle_candidates
    try:
      reverse_samples = _occ_face_reverse_samples(
          face,
          tolerances,
          distance_engine=distance_engine,
      )
    except _TriangleWorkCapExceeded as error:
      return failed_after_forward(
          status=error.status,
          message=str(error),
          checks={
              **reverse_analytic_proof_checks(),
              "reverse_face_checks": reverse_checks,
              "reverse_distance_work": distance_engine.receipt(
                  all_reverse_samples_checked=False
              ),
          },
      )
    covered = 0
    processed_reverse = 0
    maximum_reverse_distance = 0.0
    try:
      for reverse_chunk in _array_chunks(
          reverse_samples,
          tolerances.reverse_sample_chunk_size,
      ):
        for sample in reverse_chunk:
          distance = _distance_to_obj_mesh(sample, distance_engine)
          processed_reverse += 1
          maximum_reverse_distance = max(maximum_reverse_distance, distance)
          if distance <= tolerances.sample_distance_mm:
            covered += 1
    except _TriangleWorkCapExceeded as error:
      reverse_checks[str(face.index)] = {
          "sample_count": len(reverse_samples),
          "processed_sample_count": processed_reverse,
          "distance_work": {
              "exact_triangle_evaluation_count": (
                  distance_engine.endpoint_budget.exact_triangle_evaluations
                  - exact_before
              ),
              "aabb_lower_bound_evaluation_count": (
                  distance_engine.aabb_evaluations - aabb_before
              ),
              "pruned_triangle_candidate_count": (
                  distance_engine.pruned_triangle_candidates - pruned_before
              ),
          },
      }
      return failed_after_forward(
          status=error.status,
          message=str(error),
          checks={
              **reverse_analytic_proof_checks(),
              "reverse_face_checks": reverse_checks,
              "reverse_distance_work": distance_engine.receipt(
                  all_reverse_samples_checked=False
              ),
          },
      )
    if processed_reverse != len(reverse_samples):
      raise _CaseFailure(
          "reverse_coverage_failed",
          "chunked reverse coverage did not consume every OCC sample",
      )
    reverse_coverage = covered / processed_reverse
    reverse_checks[str(face.index)] = {
        "sample_count": processed_reverse,
        "coverage": reverse_coverage,
        "maximum_distance_mm": maximum_reverse_distance,
        "distance_work": {
            "query_count": processed_reverse,
            "exact_triangle_evaluation_count": (
                distance_engine.endpoint_budget.exact_triangle_evaluations
                - exact_before
            ),
            "brute_force_triangle_evaluation_upper_bound": (
                processed_reverse * len(group.triangles_mm)
            ),
            "aabb_lower_bound_evaluation_count": (
                distance_engine.aabb_evaluations - aabb_before
            ),
            "pruned_triangle_candidate_count": (
                distance_engine.pruned_triangle_candidates - pruned_before
            ),
        },
    }
    if reverse_coverage >= tolerances.minimum_reverse_face_coverage:
      reverse_covered_indices.append(face.index)
    elif reverse_coverage > 0.0:
      partial_reverse_indices.append(face.index)
  reverse_distance_work = distance_engine.receipt(
      all_reverse_samples_checked=True
  )
  unresolved_partial_indices: list[int] = []
  for partial_index in sorted(partial_reverse_indices):
    proof = (
        None
        if partial_index in set(forward_indices)
        else _analytic_zero_area_exclusion_proof(
            reference_faces,
            by_index[partial_index],
            tolerances.analytic_support_distance_mm,
        )
    )
    if proof is None:
      unresolved_partial_indices.append(partial_index)
    else:
      analytic_zero_area_exclusion_proofs.append(proof)
  if unresolved_partial_indices:
    return failed_after_forward(
        status="partial_occ_face_overlap",
        message="an OCC face is only partially covered by the Fusion OBJ group",
        checks={
            "forward_assigned_occ_face_indices": forward_indices,
            "partial_reverse_occ_face_indices": sorted(partial_reverse_indices),
            "unresolved_partial_occ_face_indices": unresolved_partial_indices,
            **reverse_analytic_proof_checks(),
            "reverse_face_checks": reverse_checks,
            "reverse_distance_work": reverse_distance_work,
        },
    )
  reverse_set = set(reverse_covered_indices)
  if not set(forward_indices).issubset(reverse_set):
    return failed_after_forward(
        status="reverse_coverage_failed",
        message="forward-mapped OCC face failed reverse OBJ coverage",
        checks={
            "forward_assigned_occ_face_indices": forward_indices,
            "reverse_covered_occ_face_indices": sorted(reverse_set),
            **reverse_analytic_proof_checks(),
            "reverse_face_checks": reverse_checks,
          "reverse_distance_work": reverse_distance_work,
        },
    )
  analytic_same_support_union_indices: list[int] = []
  for new_index in sorted(reverse_set - set(forward_indices)):
    support_results = [
        _same_analytic_support(
            reference,
            by_index[new_index],
            tolerances.analytic_support_distance_mm,
        )
        for reference in reference_faces
    ]
    if True in support_results:
      analytic_same_support_union_indices.append(new_index)
      continue
    proof = _analytic_zero_area_exclusion_proof(
        reference_faces,
        by_index[new_index],
        tolerances.analytic_support_distance_mm,
    )
    if proof is not None:
      analytic_zero_area_exclusion_proofs.append(proof)
      reverse_set.remove(new_index)
      continue
    return failed_after_forward(
        status="reverse_coverage_unresolved",
        message=(
            "reverse coverage found an additional OCC face without a "
            "provably identical or provably zero-area analytic support"
        ),
        checks={
            "forward_assigned_occ_face_indices": forward_indices,
            "reverse_covered_occ_face_indices": sorted(reverse_set),
            "unresolved_occ_face_index": new_index,
            **reverse_analytic_proof_checks(),
            "reverse_face_checks": reverse_checks,
            "reverse_distance_work": reverse_distance_work,
        },
    )
  mapped_indices = sorted(reverse_set)
  mapped = [by_index[index] for index in mapped_indices]
  boundary_status, boundary_checks = _boundary_checks(
      group,
      mapped,
      tolerances,
      cq,
  )
  if boundary_status is not None:
    return failed_after_forward(
        status=boundary_status,
        message=(
            "mapped OCC face-union boundary topology disagrees with the "
            "authoritative OBJ face"
            if boundary_status == "boundary_topology_mismatch"
            else (
                "mapped OCC face-union boundary geometry disagrees with the "
                "authoritative OBJ face"
            )
        ),
        checks={
            **boundary_checks,
            "candidate_occ_face_indices": mapped_indices,
            "reverse_distance_work": reverse_distance_work,
        },
    )
  mapped_bbox_min = np.min([face.bbox_min_mm for face in mapped], axis=0)
  mapped_bbox_max = np.max([face.bbox_max_mm for face in mapped], axis=0)
  mapped_obj_bbox_residual = _bbox_residual(
      mapped_bbox_min,
      mapped_bbox_max,
      group.bbox_min_mm,
      group.bbox_max_mm,
  )
  mapped_source_bbox_residual = _bbox_residual(
      mapped_bbox_min,
      mapped_bbox_max,
      source_bbox_min,
      source_bbox_max,
  )
  if max(mapped_obj_bbox_residual, mapped_source_bbox_residual) > bbox_tolerance:
    return failed_after_forward(
        status="bbox_mismatch",
        message="mapped OCC face union disagrees with source/OBJ bbox",
        checks={
            "mapped_obj_bbox_max_abs_error_mm": mapped_obj_bbox_residual,
            "mapped_source_bbox_max_abs_error_mm": mapped_source_bbox_residual,
            "bbox_tolerance_mm": bbox_tolerance,
            "candidate_occ_face_indices": mapped_indices,
            "reverse_distance_work": reverse_distance_work,
        },
    )
  forward_engine.configure_source_point_candidates(len(mapped))
  try:
    point_distance = min(
        forward_engine.distance(source_point, face, cq) for face in mapped
    )
  except _CaseFailure as error:
    return failed_after_forward(status=error.status, message=str(error))
  source_point_checked = True
  if point_distance > tolerances.point_distance_mm:
    return failed_after_forward(
        status="point_mismatch",
        message="Fusion point_on_entity is not on the mapped OCC face union",
        checks={
            "point_distance_mm": point_distance,
            "point_distance_tolerance_mm": tolerances.point_distance_mm,
            "candidate_occ_face_indices": mapped_indices,
            "reverse_distance_work": reverse_distance_work,
        },
    )
  mapped_area = float(sum(face.area_mm2 for face in mapped))
  area_relative_error = abs(mapped_area - group.mesh_area_mm2) / max(
      mapped_area,
      group.mesh_area_mm2,
  )
  if area_relative_error > tolerances.mesh_area_relative_error:
    return failed_after_forward(
        status="area_mismatch",
        message="mapped OCC face area disagrees with authoritative OBJ mesh area",
        checks={
            "obj_mesh_area_mm2": group.mesh_area_mm2,
            "mapped_occ_area_mm2": mapped_area,
            "area_relative_error": area_relative_error,
            "area_relative_tolerance": tolerances.mesh_area_relative_error,
            "candidate_occ_face_indices": mapped_indices,
            "reverse_distance_work": reverse_distance_work,
        },
    )
  checks = {
      "expected_surface_type": expected_surface,
      "mapped_surface_types": sorted({face.surface_type for face in mapped}),
      "obj_group_signature_sha256": group.signature_sha256,
      "obj_sample_count": sample_count,
      "obj_sample_coverage": processed_samples / sample_count,
      "coverage_mode": (
          "complete_chunked_forward_plus_bidirectional_occ_tessellation"
      ),
      "forward_sample_chunk_size": tolerances.forward_sample_chunk_size,
      "reverse_sample_chunk_size": tolerances.reverse_sample_chunk_size,
      "forward_assigned_occ_face_indices": forward_indices,
      "reverse_covered_occ_face_indices": mapped_indices,
      "reverse_face_checks": reverse_checks,
      "forward_distance_work": forward_engine.receipt(
          all_forward_samples_checked=all_forward_samples_checked,
          source_point_checked=source_point_checked,
      ),
      "reverse_distance_work": reverse_distance_work,
      "maximum_best_distance_mm": maximum_best_distance,
      "minimum_second_best_margin_mm": minimum_margin,
      **forward_geometry_proof_checks(),
      "analytic_same_support_union_occ_face_indices": (
          analytic_same_support_union_indices
      ),
      **reverse_analytic_proof_checks(),
      "minimum_forward_samples_per_forward_face": min(
          assignment_support.values()
      ),
      "reverse_only_mapped_face_count": len(
          set(mapped_indices) - set(forward_indices)
      ),
      "obj_source_bbox_max_abs_error_mm": obj_source_bbox_residual,
      "mapped_obj_bbox_max_abs_error_mm": mapped_obj_bbox_residual,
      "mapped_source_bbox_max_abs_error_mm": mapped_source_bbox_residual,
      "bbox_tolerance_mm": bbox_tolerance,
      "point_distance_mm": point_distance,
      "obj_mesh_area_mm2": group.mesh_area_mm2,
      "mapped_occ_area_mm2": mapped_area,
      "area_relative_error": area_relative_error,
      **boundary_checks,
  }
  return {
      "status": "mapped_unique",
      "mapping_mode": "authoritative_obj_face_group",
      "acceptance_basis": DISTANCE_ACCEPTANCE_BASIS,
      "fusion_face_index": fusion_index,
      "raw_occ_face_indices": mapped_indices,
      "source_face_signature_sha256s": [
          by_index[index].signature_sha256 for index in mapped_indices
      ],
      "checks": checks,
      "reason": None,
  }


_IDENTITY_WORLD_TRANSFORM = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
_ANALYTIC_TRIM_ELIGIBLE_FAILURES = {
    "sample_uncovered",
    "reverse_coverage_failed",
    "boundary_geometry_mismatch",
    "boundary_topology_mismatch",
}


def _analytic_trim_input_binding(
    inputs: AnalyticTrimReplayInputs,
    *,
    raw_occ_face_index: int,
    algorithm_revision: str = ANALYTIC_TRIM_ALGORITHM_REVISION,
) -> dict[str, Any]:
  unsigned = {
      "algorithm_revision": algorithm_revision,
      "obj_bytes_sha256": hashlib.sha256(inputs.obj_bytes).hexdigest(),
      "assembly_json_bytes_sha256": hashlib.sha256(
          inputs.assembly_json_bytes
      ).hexdigest(),
      "step_bytes_sha256": hashlib.sha256(inputs.step_bytes).hexdigest(),
      "body_uuid": inputs.body_uuid,
      "source_face_index": inputs.source_face_index,
      "raw_occ_face_index": raw_occ_face_index,
      "source_occurrence_path": list(inputs.source_occurrence_path),
      "source_path_identity": dict(inputs.source_path_identity),
      "world_transform_row_major": [
          float(value) for value in inputs.world_transform_row_major
      ],
  }
  if algorithm_revision == PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION:
    if (
        inputs.source_smt_bytes is None
        or inputs.source_smt_archive_member is None
    ):
      raise ValueError("planar chordal trim requires authenticated SMT bytes")
    unsigned.update({
        "source_smt_bytes_sha256": hashlib.sha256(
            inputs.source_smt_bytes
        ).hexdigest(),
        "source_smt_archive_member": inputs.source_smt_archive_member,
    })
  return {**unsigned, "input_binding_sha256": _canonical_sha256(unsigned)}


def _analytic_trim_cache_key(
    inputs: AnalyticTrimReplayInputs,
    *,
    raw_occ_face_index: int,
    algorithm_revision: str = ANALYTIC_TRIM_ALGORITHM_REVISION,
) -> tuple[str, ...]:
  binding = _analytic_trim_input_binding(
      inputs,
      raw_occ_face_index=raw_occ_face_index,
      algorithm_revision=algorithm_revision,
  )
  return (
      algorithm_revision,
      str(binding["obj_bytes_sha256"]),
      str(binding["assembly_json_bytes_sha256"]),
      str(binding["step_bytes_sha256"]),
      inputs.body_uuid,
      str(inputs.source_face_index),
      str(raw_occ_face_index),
      _canonical_json(binding["source_occurrence_path"]),
      _canonical_json(binding["source_path_identity"]),
      _canonical_json(binding["world_transform_row_major"]),
      str(binding["input_binding_sha256"]),
  )


def _build_analytic_trim_evidence(
    inputs: AnalyticTrimReplayInputs,
    *,
    raw_occ_face_index: int,
) -> dict[str, Any]:
  # These imports are intentionally lazy: analytic_trim_domain_v1 reuses
  # mapper boundary helpers and therefore cannot be imported while this module
  # is still being initialized.
  from .analytic_trim_domain_v1 import (
      SCHEMA_VERSION as trim_schema_version,
      capture_occ_cylinder_trim_domain_v1,
      certify_cylinder_rectangular_analytic_trim_v1,
  )
  from .source_obj_analytic_support_v1 import (
      capture_source_obj_face_v1,
      recover_source_obj_analytic_support_v1,
  )

  source = capture_source_obj_face_v1(
      obj_bytes=inputs.obj_bytes,
      assembly_json_bytes=inputs.assembly_json_bytes,
      body_uuid=inputs.body_uuid,
      face_index=inputs.source_face_index,
      unit_binding={
          "source_length_unit": "centimetre",
          "millimeters_per_source_unit": 10.0,
      },
      path_metadata={
          "obj_archive_member": inputs.source_path_identity[
              "obj_archive_member"
          ],
          "assembly_archive_member": inputs.source_path_identity[
              "assembly_archive_member"
          ],
      },
  )
  support = recover_source_obj_analytic_support_v1(source)
  if support.surface_type != "cylinder":
    raise ValueError("source analytic support is not Cylinder")
  occ = capture_occ_cylinder_trim_domain_v1(
      step_bytes=inputs.step_bytes,
      source_assembly_json_bytes=inputs.assembly_json_bytes,
      source_body_uuid=inputs.body_uuid,
      raw_occ_face_index=raw_occ_face_index,
      source_occurrence_path=inputs.source_occurrence_path,
      source_path_identity=inputs.source_path_identity,
  )
  certificate = certify_cylinder_rectangular_analytic_trim_v1(
      source,
      support,
      occ,
  )
  receipt = json.loads(certificate.canonical_json)
  return {
      "schema_version": trim_schema_version,
      "algorithm_revision": ANALYTIC_TRIM_ALGORITHM_REVISION,
      "receipt_sha256": certificate.sha256,
      "receipt": receipt,
      "input_binding": _analytic_trim_input_binding(
          inputs,
          raw_occ_face_index=raw_occ_face_index,
      ),
  }


def _build_planar_chordal_trim_evidence(
    inputs: AnalyticTrimReplayInputs,
    *,
    raw_occ_face_index: int,
) -> dict[str, Any]:
  from .planar_chordal_trim_domain_v1 import (
      SCHEMA_VERSION as trim_schema_version,
      build_planar_closed_circle_chordal_trim_receipt_v1,
  )

  receipt = build_planar_closed_circle_chordal_trim_receipt_v1(
      obj_bytes=inputs.obj_bytes,
      assembly_json_bytes=inputs.assembly_json_bytes,
      source_smt_bytes=inputs.source_smt_bytes or b"",
      source_smt_archive_member=inputs.source_smt_archive_member or "",
      step_bytes=inputs.step_bytes,
      body_uuid=inputs.body_uuid,
      source_face_index=inputs.source_face_index,
      raw_occ_face_index=raw_occ_face_index,
      source_occurrence_path=inputs.source_occurrence_path,
      source_path_identity=inputs.source_path_identity,
  )
  return {
      "schema_version": trim_schema_version,
      "algorithm_revision": PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION,
      "receipt_sha256": _canonical_sha256(receipt),
      "receipt": receipt,
      "input_binding": _analytic_trim_input_binding(
          inputs,
          raw_occ_face_index=raw_occ_face_index,
          algorithm_revision=PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION,
      ),
  }


def verify_analytic_trim_mapping_full_replay(
    mapping: Mapping[str, Any],
    *,
    inputs: AnalyticTrimReplayInputs,
) -> dict[str, Any]:
  """Replay a v2.6 analytic-trim mapping from the bound source/STEP bytes."""

  if (
      not isinstance(mapping, Mapping)
      or mapping.get("status") != "mapped_unique"
      or mapping.get("acceptance_basis")
      not in {
          ANALYTIC_TRIM_ACCEPTANCE_BASIS,
          PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS,
      }
  ):
    raise FaceMapAuditError("analytic trim mapping contract is invalid")
  indices = mapping.get("raw_occ_face_indices")
  signatures = mapping.get("source_face_signature_sha256s")
  evidence = mapping.get("analytic_trim_certificate")
  if (
      not isinstance(indices, list)
      or len(indices) != 1
      or type(indices[0]) is not int
      or not isinstance(signatures, list)
      or len(signatures) != 1
      or not isinstance(evidence, Mapping)
  ):
    raise FaceMapAuditError("analytic trim mapping identity is malformed")
  raw_occ_face_index = indices[0]
  planar = (
      mapping.get("acceptance_basis")
      == PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS
  )
  algorithm_revision = (
      PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION
      if planar
      else ANALYTIC_TRIM_ALGORITHM_REVISION
  )
  expected_binding = _analytic_trim_input_binding(
      inputs,
      raw_occ_face_index=raw_occ_face_index,
      algorithm_revision=algorithm_revision,
  )
  if evidence.get("input_binding") != expected_binding:
    raise FaceMapAuditError("analytic trim mapping input binding differs")

  receipt = evidence.get("receipt")
  if not isinstance(receipt, Mapping):
    raise FaceMapAuditError("analytic trim certificate receipt is malformed")
  receipt_sha256 = hashlib.sha256(
      _canonical_json(receipt).encode("utf-8")
  ).hexdigest()
  if evidence.get("receipt_sha256") != receipt_sha256:
    raise FaceMapAuditError("analytic trim certificate hash differs")
  if planar:
    from .planar_chordal_trim_domain_v1 import (
        SCHEMA_VERSION as planar_schema_version,
        verify_planar_closed_circle_chordal_trim_receipt_v1,
    )

    if (
        evidence.get("schema_version") != planar_schema_version
        or evidence.get("algorithm_revision")
        != PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION
    ):
      raise FaceMapAuditError("planar chordal trim certificate contract differs")
    try:
      verified = verify_planar_closed_circle_chordal_trim_receipt_v1(
          receipt,
          obj_bytes=inputs.obj_bytes,
          assembly_json_bytes=inputs.assembly_json_bytes,
          source_smt_bytes=inputs.source_smt_bytes or b"",
          source_smt_archive_member=(
              inputs.source_smt_archive_member or ""
          ),
          step_bytes=inputs.step_bytes,
          body_uuid=inputs.body_uuid,
          source_face_index=inputs.source_face_index,
          raw_occ_face_index=raw_occ_face_index,
          source_occurrence_path=inputs.source_occurrence_path,
          source_path_identity=inputs.source_path_identity,
      )
    except (TypeError, ValueError) as error:
      raise FaceMapAuditError(
          f"planar chordal trim certificate full replay failed: {error}"
      ) from error
    if _canonical_sha256(verified) != receipt_sha256:
      raise FaceMapAuditError("planar chordal trim replay hash differs")
    occ_binding = receipt.get("occ_binding")
    if (
        not isinstance(occ_binding, Mapping)
        or occ_binding.get("raw_occ_face_index") != raw_occ_face_index
        or signatures
        != [occ_binding.get("deterministic_face_signature_sha256")]
    ):
      raise FaceMapAuditError(
          "planar chordal trim face identity differs from STEP replay"
      )
    return dict(mapping)

  from .analytic_trim_domain_v1 import (
      SCHEMA_VERSION as trim_schema_version,
      capture_occ_cylinder_trim_domain_v1,
      verify_cylinder_rectangular_analytic_trim_v1,
  )
  from .source_obj_analytic_support_v1 import (
      capture_source_obj_face_v1,
      recover_source_obj_analytic_support_v1,
  )

  if (
      evidence.get("schema_version") != trim_schema_version
      or evidence.get("algorithm_revision") != ANALYTIC_TRIM_ALGORITHM_REVISION
  ):
    raise FaceMapAuditError("analytic trim certificate contract differs")
  try:
    source = capture_source_obj_face_v1(
        obj_bytes=inputs.obj_bytes,
        assembly_json_bytes=inputs.assembly_json_bytes,
        body_uuid=inputs.body_uuid,
        face_index=inputs.source_face_index,
        unit_binding={
            "source_length_unit": "centimetre",
            "millimeters_per_source_unit": 10.0,
        },
        path_metadata={
            "obj_archive_member": inputs.source_path_identity[
                "obj_archive_member"
            ],
            "assembly_archive_member": inputs.source_path_identity[
                "assembly_archive_member"
            ],
        },
    )
    support = recover_source_obj_analytic_support_v1(source)
    occ = capture_occ_cylinder_trim_domain_v1(
        step_bytes=inputs.step_bytes,
        source_assembly_json_bytes=inputs.assembly_json_bytes,
        source_body_uuid=inputs.body_uuid,
        raw_occ_face_index=raw_occ_face_index,
        source_occurrence_path=inputs.source_occurrence_path,
        source_path_identity=inputs.source_path_identity,
    )
    verified = verify_cylinder_rectangular_analytic_trim_v1(
        receipt,
        source,
        support,
        occ,
    )
  except (TypeError, ValueError) as error:
    raise FaceMapAuditError(
        f"analytic trim certificate full replay failed: {error}"
    ) from error
  if verified.sha256 != receipt_sha256:
    raise FaceMapAuditError("analytic trim certificate replay hash differs")
  occ_binding = receipt.get("occ_binding")
  if (
      not isinstance(occ_binding, Mapping)
      or occ_binding.get("raw_occ_face_index") != raw_occ_face_index
      or signatures
      != [occ_binding.get("deterministic_face_signature_sha256")]
  ):
    raise FaceMapAuditError(
        "analytic trim mapping face identity differs from STEP replay"
    )
  return dict(mapping)


def _map_obj_group_to_occ(
    *,
    group: _ObjFaceGroup,
    entity: Mapping[str, Any],
    occ_faces: Sequence[_OccFace],
    tolerances: MappingTolerances,
    cq: Any,
    case_forward_budget: _ForwardDistanceCallBudget | None = None,
    run_forward_budget: _ForwardDistanceCallBudget | None = None,
    case_triangle_budget: _TriangleEvaluationBudget | None = None,
    run_triangle_budget: _TriangleEvaluationBudget | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_case_id: str = "",
    progress_endpoint_part: str | None = None,
    analytic_trim_inputs: AnalyticTrimReplayInputs | None = None,
    analytic_trim_cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
  legacy = _map_obj_group_to_occ_distance(
      group=group,
      entity=entity,
      occ_faces=occ_faces,
      tolerances=tolerances,
      cq=cq,
      case_forward_budget=case_forward_budget,
      run_forward_budget=run_forward_budget,
      case_triangle_budget=case_triangle_budget,
      run_triangle_budget=run_triangle_budget,
      progress_callback=progress_callback,
      progress_case_id=progress_case_id,
      progress_endpoint_part=progress_endpoint_part,
  )
  if analytic_trim_inputs is None:
    return legacy
  expected_surface = _surface_type(entity.get("surface_type"))
  if (
      legacy.get("status") not in _ANALYTIC_TRIM_ELIGIBLE_FAILURES
      or expected_surface not in {"CYLINDER", "PLANE"}
      or analytic_trim_inputs.source_face_index != group.fusion_face_index
      or analytic_trim_inputs.source_occurrence_path
      or tuple(
          float(value)
          for value in analytic_trim_inputs.world_transform_row_major
      ) != _IDENTITY_WORLD_TRANSFORM
  ):
    return legacy

  source_bbox_min, source_bbox_max = _entity_bbox_mm(entity)
  bbox_tolerance = max(
      _bbox_tolerance(group.bbox_min_mm, group.bbox_max_mm, tolerances),
      _bbox_tolerance(source_bbox_min, source_bbox_max, tolerances),
  )
  candidates = sorted(
      (
          face
          for face in occ_faces
          if face.surface_type == expected_surface
          and np.all(face.bbox_max_mm >= group.bbox_min_mm - bbox_tolerance)
          and np.all(face.bbox_min_mm <= group.bbox_max_mm + bbox_tolerance)
      ),
      key=lambda face: face.index,
  )
  legacy_checks = legacy.get("checks")
  if not isinstance(legacy_checks, Mapping):
    return legacy
  if legacy.get("status") == "sample_uncovered":
    candidate_index = legacy_checks.get("best_occ_face_index")
    ranking_safe = (
        legacy_checks.get("candidate_ranking_unique_margin_safe") is True
    )
  else:
    ranked_indices = legacy_checks.get("forward_assigned_occ_face_indices")
    if not isinstance(ranked_indices, list):
      ranked_indices = legacy_checks.get("candidate_occ_face_indices")
    candidate_index = (
        ranked_indices[0]
        if isinstance(ranked_indices, list) and len(ranked_indices) == 1
        else None
    )
    # Reaching reverse/boundary gates means every forward ranking already
    # passed the unchanged ambiguity/margin gate.
    ranking_safe = candidate_index is not None
  candidate = next(
      (face for face in candidates if face.index == candidate_index),
      None,
  )
  if candidate is None or not ranking_safe:
    return legacy
  if (
      expected_surface == "PLANE"
      and legacy.get("status") != "boundary_geometry_mismatch"
  ):
    return legacy
  if (
      expected_surface == "PLANE"
      and (
          analytic_trim_inputs.source_smt_bytes is None
          or analytic_trim_inputs.source_smt_archive_member is None
      )
  ):
    return legacy
  algorithm_revision = (
      PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION
      if expected_surface == "PLANE"
      else ANALYTIC_TRIM_ALGORITHM_REVISION
  )
  cache = analytic_trim_cache if analytic_trim_cache is not None else {}
  key = _analytic_trim_cache_key(
      analytic_trim_inputs,
      raw_occ_face_index=candidate.index,
      algorithm_revision=algorithm_revision,
  )
  cached = cache.get(key)
  if cached is None:
    try:
      cached = {
          "status": "certified",
          "evidence": (
              _build_planar_chordal_trim_evidence(
                  analytic_trim_inputs,
                  raw_occ_face_index=candidate.index,
              )
              if expected_surface == "PLANE"
              else _build_analytic_trim_evidence(
                  analytic_trim_inputs,
                  raw_occ_face_index=candidate.index,
              )
          ),
      }
    except Exception as error:
      cached = {
          "status": "failed",
          "error_type": type(error).__name__,
          "reason": str(error),
      }
    cache[key] = cached
  if cached.get("status") != "certified":
    return {
        **legacy,
        "checks": {
            **dict(legacy.get("checks") or {}),
            "analytic_trim_certificate_attempt": dict(cached),
        },
    }
  evidence = cached.get("evidence")
  if not isinstance(evidence, Mapping):
    return legacy
  return {
      "status": "mapped_unique",
      "mapping_mode": "authoritative_obj_face_group",
      "acceptance_basis": (
          PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS
          if expected_surface == "PLANE"
          else ANALYTIC_TRIM_ACCEPTANCE_BASIS
      ),
      "fusion_face_index": group.fusion_face_index,
      "raw_occ_face_indices": [candidate.index],
      "source_face_signature_sha256s": [candidate.signature_sha256],
      "analytic_trim_certificate": dict(evidence),
      "checks": {
          "expected_surface_type": expected_surface,
          "mapped_surface_types": [expected_surface],
          "candidate_occ_face_indices": [candidate.index],
          "candidate_ranking_unique_margin_safe": True,
          "legacy_failure_status": legacy["status"],
          "legacy_failure_reason": legacy["reason"],
          "legacy_distance_checks": dict(legacy.get("checks") or {}),
          "locked_sample_distance_tolerance_mm": (
              tolerances.sample_distance_mm
          ),
          "locked_minimum_sample_margin_mm": (
              tolerances.minimum_sample_margin_mm
          ),
          "analytic_trim_introduced_new_distance_threshold": False,
          "locked_boundary_distance_tolerance_mm": (
              tolerances.boundary_distance_mm
          ),
      },
      "reason": None,
  }


def _step_length_unit_sidecar(step_path: Path) -> dict[str, Any]:
  """Replay the receipt-bound STEP file-unit declaration without guessing."""

  try:
    from OCP.IFSelect import IFSelect_RetDone  # type: ignore
    from OCP.STEPControl import STEPControl_Reader  # type: ignore
    from OCP.TColStd import TColStd_SequenceOfAsciiString  # type: ignore

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
      raise ValueError("STEP reader did not accept the staged file")
    length_units = TColStd_SequenceOfAsciiString()
    angle_units = TColStd_SequenceOfAsciiString()
    solid_angle_units = TColStd_SequenceOfAsciiString()
    reader.FileUnits(length_units, angle_units, solid_angle_units)
    names = [
        str(length_units.Value(index).ToCString()).strip().lower()
        for index in range(1, length_units.Length() + 1)
    ]
  except Exception as error:
    raise _CaseFailure(
        "unit_contract_failed",
        "receipt-bound STEP length-unit replay failed",
    ) from error
  file_to_occ_scale_by_unit = {
      "millimetre": 1.0,
      "centimetre": 10.0,
  }
  if len(names) != 1 or names[0] not in file_to_occ_scale_by_unit:
    raise _CaseFailure(
        "unit_contract_failed",
        "receipt-bound STEP length unit is unsupported or ambiguous",
    )
  return {
      "schema_version": "receipt_bound_step_length_unit_sidecar.v1",
      "status": "verified_step_declared_length_unit_to_occ_millimetre",
      "file_length_unit_names": names,
      "fusion_source_length_unit": "centimetre",
      "occ_runtime_length_unit": "millimetre",
      "expected_source_to_occ_length_scale": 10.0,
      "expected_file_to_occ_length_scale": file_to_occ_scale_by_unit[names[0]],
  }


def _unit_contract(
    *,
    shape: Any,
    physical: Mapping[str, Any],
    step_unit_sidecar: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
  try:
    json_area_cm2 = float(physical.get("area"))
    json_volume_cm3 = float(physical.get("volume"))
    value = shape.val() if hasattr(shape, "val") else shape
    occ_area_mm2 = float(value.Area())
    occ_volume_mm3 = float(value.Volume())
    solid_count = len(value.Solids())
  except (TypeError, ValueError, AttributeError) as error:
    raise _CaseFailure("unit_contract_failed", "body physical properties are invalid") from error
  file_units = step_unit_sidecar.get("file_length_unit_names")
  expected_file_scale = {
      "millimetre": 1.0,
      "centimetre": 10.0,
  }
  expected_unit_sidecar = {
      "schema_version": "receipt_bound_step_length_unit_sidecar.v1",
      "status": "verified_step_declared_length_unit_to_occ_millimetre",
      "file_length_unit_names": file_units,
      "fusion_source_length_unit": "centimetre",
      "occ_runtime_length_unit": "millimetre",
      "expected_source_to_occ_length_scale": 10.0,
      "expected_file_to_occ_length_scale": (
          expected_file_scale.get(file_units[0])
          if isinstance(file_units, list) and len(file_units) == 1
          else None
      ),
  }
  if (
      not isinstance(file_units, list)
      or len(file_units) != 1
      or file_units[0] not in expected_file_scale
      or dict(step_unit_sidecar) != expected_unit_sidecar
  ):
    raise _CaseFailure(
        "unit_contract_failed",
        "STEP length-unit sidecar does not prove the locked cm-to-mm contract",
    )
  values = (json_area_cm2, json_volume_cm3, occ_area_mm2, occ_volume_mm3)
  if any(not math.isfinite(item) for item in values):
    raise _CaseFailure("unit_contract_failed", "body physical properties are non-finite")
  if json_area_cm2 <= 0.0 or occ_area_mm2 <= 0.0 or json_volume_cm3 < 0.0:
    raise _CaseFailure("unit_contract_failed", "body physical properties are invalid")
  area_scale = occ_area_mm2 / json_area_cm2
  area_error = abs(area_scale - 100.0) / 100.0
  topology_dimension = "solid" if solid_count > 0 else "surface"
  if topology_dimension == "solid":
    if json_volume_cm3 <= 0.0 or occ_volume_mm3 <= 0.0:
      physical_status = "solid_volume_invalid"
      volume_scale: float | None = None
      volume_error: float | None = None
    else:
      volume_scale = occ_volume_mm3 / json_volume_cm3
      volume_error = abs(volume_scale - 1000.0) / 1000.0
      area_ok = area_error <= tolerance
      volume_ok = volume_error <= tolerance
      if area_ok and volume_ok:
        physical_status = "verified"
      elif not area_ok and not volume_ok:
        physical_status = "area_and_volume_scale_mismatch"
      elif not area_ok:
        physical_status = "area_scale_mismatch"
      else:
        physical_status = "volume_scale_mismatch"
    volume_check_mode = "positive_volume_scale"
  else:
    volume_scale = None
    volume_error = None
    volume_check_mode = "source_zero_volume_surface_not_applicable"
    if json_volume_cm3 != 0.0:
      physical_status = "surface_source_volume_nonzero"
    elif area_error <= tolerance:
      physical_status = "verified"
    else:
      physical_status = "area_scale_mismatch"
  status = (
      "verified_cm_to_occ_mm"
      if physical_status == "verified"
      else "physical_property_replay_failed"
  )
  physical_property_replay = {
      "schema_version": "fusion_source_physical_property_replay.v1",
      "status": physical_status,
      "topology_dimension": topology_dimension,
      "solid_count": solid_count,
      "volume_check_mode": volume_check_mode,
      "json_area_cm2": json_area_cm2,
      "json_volume_cm3": json_volume_cm3,
      "occ_area_mm2": occ_area_mm2,
      "occ_volume_mm3": occ_volume_mm3,
      "observed_area_scale": area_scale,
      "observed_volume_scale": volume_scale,
      "expected_area_scale": 100.0,
      "expected_volume_scale": 1000.0,
      "area_scale_relative_error": area_error,
      "volume_scale_relative_error": volume_error,
      "relative_tolerance": tolerance,
  }
  return {
      "status": status,
      "schema_version": "step_unit_and_source_physical_property_replay.v1",
      "step_unit_sidecar": dict(step_unit_sidecar),
      "physical_property_replay": physical_property_replay,
      "json_area_cm2": json_area_cm2,
      "json_volume_cm3": json_volume_cm3,
      "occ_area_mm2": occ_area_mm2,
      "occ_volume_mm3": occ_volume_mm3,
      "observed_area_scale": area_scale,
      "observed_volume_scale": volume_scale,
      "expected_area_scale": 100.0,
      "expected_volume_scale": 1000.0,
      "area_scale_relative_error": area_error,
      "volume_scale_relative_error": volume_error,
      "relative_tolerance": tolerance,
  }


def _gold_cases(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
  gold = source["evaluation_gold_contacts"]
  raw_cases = gold.get("cases")
  if not isinstance(raw_cases, list):
    raise FaceMapAuditError("private evaluation sidecar cases are malformed")
  result: dict[str, Mapping[str, Any]] = {}
  for row in raw_cases:
    if not isinstance(row, Mapping):
      raise FaceMapAuditError("private evaluation case is malformed")
    case_id = str(row.get("case_id") or "")
    if not case_id or case_id in result:
      raise FaceMapAuditError("private evaluation case IDs are invalid")
    result[case_id] = row
  return result


def _gold_endpoint_rows(gold_case: Mapping[str, Any]) -> list[dict[str, Any]]:
  contacts = gold_case.get("contacts")
  if not isinstance(contacts, list):
    raise _CaseFailure("gold_binding_mismatch", "gold case contacts are malformed")
  rows: list[dict[str, Any]] = []
  for contact in contacts:
    if not isinstance(contact, Mapping):
      raise _CaseFailure("gold_binding_mismatch", "gold contact is malformed")
    ordinal = contact.get("source_contact_ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
      raise _CaseFailure("gold_binding_mismatch", "gold contact ordinal is invalid")
    contact_id = str(contact.get("contact_id") or "")
    for role in ("a", "b"):
      endpoint = contact.get(f"endpoint_{role}")
      if not isinstance(endpoint, Mapping):
        raise _CaseFailure("gold_binding_mismatch", "gold endpoint is malformed")
      fusion_index = endpoint.get("fusion_face_index")
      part = endpoint.get("part")
      geometry_asset = endpoint.get("geometry_asset")
      if (
          not isinstance(part, str)
          or not part
          or not isinstance(geometry_asset, str)
          or not geometry_asset
          or endpoint.get("entity_type") != "BRepFace"
          or not isinstance(fusion_index, int)
          or isinstance(fusion_index, bool)
          or fusion_index < 0
      ):
        raise _CaseFailure("gold_binding_mismatch", "gold endpoint identity is invalid")
      rows.append(
          {
              "contact_id": contact_id,
              "source_contact_ordinal": ordinal,
              "endpoint_role": role,
              "part": part,
              "geometry_asset": geometry_asset,
              "fusion_face_index": fusion_index,
              "surface_type": str(endpoint.get("surface_type") or ""),
          }
      )
  return rows


def _private_source_cases(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
  private = source["private_source_bindings"]
  raw_cases = private.get("cases")
  if not isinstance(raw_cases, list):
    raise FaceMapAuditError("private source-binding cases are malformed")
  result: dict[str, Mapping[str, Any]] = {}
  for row in raw_cases:
    if not isinstance(row, Mapping):
      raise FaceMapAuditError("private source-binding case is malformed")
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id or case_id in result:
      raise FaceMapAuditError("private source-binding case IDs are invalid")
    result[case_id] = row
  return result


def _source_instance_key(raw: Any) -> dict[str, str]:
  if not isinstance(raw, Mapping):
    raise _CaseFailure("source_binding_invalid", "source instance key is malformed")
  kind = raw.get("kind")
  body_uuid = raw.get("body_uuid")
  if not isinstance(body_uuid, str) or not body_uuid:
    raise _CaseFailure("source_binding_invalid", "source instance body is invalid")
  if kind == "occurrence":
    expected = {"kind", "body_uuid", "occurrence_uuid"}
    occurrence_uuid = raw.get("occurrence_uuid")
    if set(raw) != expected or not isinstance(occurrence_uuid, str) or not occurrence_uuid:
      raise _CaseFailure(
          "source_binding_invalid",
          "occurrence instance key is malformed",
      )
    return {
        "kind": "occurrence",
        "body_uuid": body_uuid,
        "occurrence_uuid": occurrence_uuid,
    }
  if kind == "root":
    expected = {"kind", "body_uuid", "root_component_uuid"}
    root_component_uuid = raw.get("root_component_uuid")
    if (
        set(raw) != expected
        or not isinstance(root_component_uuid, str)
        or not root_component_uuid
    ):
      raise _CaseFailure("source_binding_invalid", "root instance key is malformed")
    return {
        "kind": "root",
        "body_uuid": body_uuid,
        "root_component_uuid": root_component_uuid,
    }
  raise _CaseFailure("source_binding_invalid", "source instance kind is invalid")


def _source_instance_preimage_key(source_key: Mapping[str, str]) -> list[str]:
  if source_key["kind"] == "occurrence":
    return [
        "occurrence",
        source_key["occurrence_uuid"],
        source_key["body_uuid"],
    ]
  return ["root", source_key["root_component_uuid"], source_key["body_uuid"]]


def _case_instance_bindings(
    public_case: Mapping[str, Any],
    private_case: Mapping[str, Any],
) -> tuple[list[str], dict[str, _PartInstanceBinding]]:
  aliases_raw = public_case.get("selected_part_names")
  assets_raw = public_case.get("selected_geometry_assets")
  if (
      not isinstance(aliases_raw, list)
      or not isinstance(assets_raw, list)
      or not aliases_raw
      or len(aliases_raw) != len(assets_raw)
      or public_case.get("instance_identity_contract")
      != _PUBLIC_INSTANCE_IDENTITY_CONTRACT
  ):
    raise _CaseFailure("source_binding_invalid", "public instance aliases are invalid")
  aliases = list(aliases_raw)
  assets = list(assets_raw)
  if (
      any(not isinstance(value, str) or not value for value in aliases)
      or any(not isinstance(value, str) or not value for value in assets)
      or len(set(aliases)) != len(aliases)
  ):
    raise _CaseFailure("source_binding_invalid", "public instance aliases are invalid")
  body_ids_raw = private_case.get("selected_body_uuids")
  parts_raw = private_case.get("parts")
  if (
      private_case.get("case_id") != public_case.get("id")
      or private_case.get("selected_part_names") != aliases
      or private_case.get("selected_geometry_assets") != assets
      or not isinstance(body_ids_raw, list)
      or not isinstance(parts_raw, list)
      or len(body_ids_raw) != len(aliases)
      or len(parts_raw) != len(aliases)
      or any(not isinstance(value, str) or not value for value in body_ids_raw)
  ):
    raise _CaseFailure(
        "source_binding_invalid",
        "public/private instance identity lists differ",
    )
  binding = private_case.get("source_receipt_binding")
  if not isinstance(binding, Mapping):
    raise _CaseFailure("source_binding_invalid", "case lacks source receipt binding")
  raw_steps = binding.get("body_steps")
  raw_instances = binding.get("instances")
  if not isinstance(raw_steps, list) or not isinstance(raw_instances, list):
    raise _CaseFailure("source_binding_invalid", "instance receipt rows are malformed")

  def rows_by_part(rows: Sequence[Any], label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
      if not isinstance(raw, Mapping):
        raise _CaseFailure("source_binding_invalid", f"{label} row is malformed")
      part = raw.get("part")
      if not isinstance(part, str) or not part or part in indexed:
        raise _CaseFailure("source_binding_invalid", f"{label} part is duplicated")
      indexed[part] = raw
    if set(indexed) != set(aliases):
      raise _CaseFailure("source_binding_invalid", f"{label} rows do not cover parts")
    return indexed

  step_by_part = rows_by_part(raw_steps, "STEP binding")
  instance_by_part = rows_by_part(raw_instances, "instance binding")
  result: dict[str, _PartInstanceBinding] = {}
  seen_instance_keys: set[str] = set()
  asset_receipts: dict[str, str] = {}
  body_assets: dict[str, str] = {}
  for index, part in enumerate(aliases):
    geometry_asset = assets[index]
    body_uuid = body_ids_raw[index]
    raw_step = step_by_part[part]
    raw_instance = instance_by_part[part]
    if (
        raw_step.get("geometry_asset") != geometry_asset
        or raw_step.get("body_uuid") != body_uuid
        or raw_instance.get("geometry_asset") != geometry_asset
    ):
      raise _CaseFailure(
          "source_binding_invalid",
          "part, geometry asset, and native body bindings disagree",
      )
    step_relative = _safe_relative_path(raw_step.get("path"), suffix=".step")
    selected_step = _safe_relative_path(parts_raw[index], suffix=".step")
    if len(PurePosixPath(selected_step).parts) != 1 or (
        PurePosixPath(step_relative).name != selected_step
    ):
      raise _CaseFailure("source_binding_invalid", "selected STEP binding differs")
    source_key = _source_instance_key(raw_instance.get("source_instance_key"))
    if source_key["body_uuid"] != body_uuid:
      raise _CaseFailure("source_binding_invalid", "instance/native body binding differs")
    occurrence_path = raw_instance.get("occurrence_path")
    if (
        not isinstance(occurrence_path, list)
        or any(not isinstance(value, str) or not value for value in occurrence_path)
        or not isinstance(raw_instance.get("is_visible"), bool)
        or not isinstance(raw_instance.get("is_grounded"), bool)
    ):
      raise _CaseFailure("source_binding_invalid", "instance metadata is malformed")
    if source_key["kind"] == "occurrence":
      if not occurrence_path or occurrence_path[-1] != source_key["occurrence_uuid"]:
        raise _CaseFailure("source_binding_invalid", "occurrence path/key differ")
    elif occurrence_path:
      raise _CaseFailure("source_binding_invalid", "root instance path is not empty")
    canonical_key = _canonical_json(source_key)
    if canonical_key in seen_instance_keys:
      raise _CaseFailure("source_binding_invalid", "assembly instance key is duplicated")
    seen_instance_keys.add(canonical_key)

    asset_receipt = _canonical_json(
        {key: value for key, value in raw_step.items() if key != "part"}
    )
    previous_asset_receipt = asset_receipts.setdefault(geometry_asset, asset_receipt)
    if previous_asset_receipt != asset_receipt:
      raise _CaseFailure("source_binding_invalid", "geometry asset receipts conflict")
    previous_asset = body_assets.setdefault(body_uuid, geometry_asset)
    if previous_asset != geometry_asset:
      raise _CaseFailure("source_binding_invalid", "native body has multiple assets")
    result[part] = _PartInstanceBinding(
        part=part,
        geometry_asset=geometry_asset,
        body_uuid=body_uuid,
        source_instance_key=source_key,
        occurrence_path=tuple(occurrence_path),
        is_visible=bool(raw_instance["is_visible"]),
        is_grounded=bool(raw_instance["is_grounded"]),
        step_binding=raw_step,
    )
  return aliases, result


def _bind_endpoint_instances(
    endpoint_rows: Sequence[Mapping[str, Any]],
    instances_by_part: Mapping[str, _PartInstanceBinding],
) -> list[dict[str, Any]]:
  bound: list[dict[str, Any]] = []
  for endpoint in endpoint_rows:
    part = str(endpoint.get("part") or "")
    instance = instances_by_part.get(part)
    if instance is None:
      raise _CaseFailure(
          "gold_binding_mismatch",
          "gold endpoint part has no selected instance binding",
      )
    if endpoint.get("geometry_asset") != instance.geometry_asset:
      raise _CaseFailure(
          "gold_binding_mismatch",
          "gold endpoint geometry asset differs from its part instance",
      )
    bound.append(
        {
            **dict(endpoint),
            "body_uuid": instance.body_uuid,
            "source_instance_key": dict(instance.source_instance_key),
        }
    )
  return bound


def _failure_endpoints(
    endpoint_rows: Sequence[Mapping[str, Any]],
    *,
    status: str,
    reason: str,
) -> list[dict[str, Any]]:
  return [
      {
          **dict(row),
          **_failed_mapping(
              status=status,
              fusion_index=int(row["fusion_face_index"]),
              message=reason,
          ),
      }
      for row in endpoint_rows
  ]


def _invalidate_case_receipt(
    case_receipt: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> None:
  """Erase executable face identities when an after-use integrity gate fails."""

  endpoints = _failure_endpoints(
      case_receipt.get("endpoints", []),
      status=status,
      reason=reason,
  )
  previous_summary = case_receipt.get("summary")
  previous_forward_work = (
      previous_summary.get("forward_distance_work")
      if isinstance(previous_summary, Mapping)
      else None
  )
  previous_reverse_work = (
      previous_summary.get("reverse_distance_work")
      if isinstance(previous_summary, Mapping)
      else None
  )
  if not isinstance(previous_forward_work, Mapping):
    raise FaceMapAuditError(
        "case receipt lacks forward work summary during after-use invalidation"
    )
  if not isinstance(previous_reverse_work, Mapping):
    raise FaceMapAuditError(
        "case receipt lacks reverse work summary during after-use invalidation"
    )
  case_receipt.update(
      {
          "status": "input_integrity_failed",
          "failure_status": status,
          "reason": reason,
          "endpoints": endpoints,
          "face_mappings": _face_mappings_from_endpoints(endpoints),
          "summary": {
              "endpoint_count": len(endpoints),
              "mapped_endpoint_count": 0,
              "status_counts": {status: len(endpoints)},
              "all_endpoints_mapped": False,
              "body_status_counts": {"input_integrity_failed": len(case_receipt.get("bodies", {}))},
              "all_bodies_ready": False,
              "case_complete": False,
              "forward_distance_work": dict(previous_forward_work),
              "reverse_distance_work": dict(previous_reverse_work),
          },
      }
  )
  for body in case_receipt.get("bodies", {}).values():
    if isinstance(body, dict):
      body.update(
          {
              "status": "input_integrity_failed",
              "failure_status": status,
              "reason": reason,
          }
      )


def _face_mappings_from_endpoints(
    endpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
  """Deduplicate contact endpoints into the receipt's consumption boundary."""

  rows: dict[tuple[str, int], dict[str, Any]] = {}
  copied_fields = (
      "part",
      "geometry_asset",
      "body_uuid",
      "source_instance_key",
      "fusion_face_index",
      "surface_type",
      "status",
      "mapping_mode",
      "acceptance_basis",
      "raw_occ_face_indices",
      "source_face_signature_sha256s",
      "analytic_trim_certificate",
      "checks",
      "reason",
      "rejected_occ_face_indices",
      "rejected_source_face_signature_sha256s",
  )
  for endpoint in endpoints:
    key = (str(endpoint.get("part") or ""), int(endpoint["fusion_face_index"]))
    row = {
        field: endpoint[field]
        for field in copied_fields
        if field in endpoint
    }
    previous = rows.get(key)
    if previous is not None and _canonical_json(previous) != _canonical_json(row):
      raise _CaseFailure(
          "source_entity_conflict",
          "duplicate endpoint identities produced different face mappings",
      )
    rows[key] = row
  return [rows[key] for key in sorted(rows)]


def _bind_source_entity(
    *,
    assembly_contacts: Sequence[Any],
    endpoint: Mapping[str, Any],
    instances_by_part: Mapping[str, _PartInstanceBinding],
) -> Mapping[str, Any]:
  ordinal = int(endpoint["source_contact_ordinal"])
  if ordinal >= len(assembly_contacts):
    raise _CaseFailure("gold_binding_mismatch", "gold ordinal exceeds source contacts")
  contact = assembly_contacts[ordinal]
  if not isinstance(contact, Mapping):
    raise _CaseFailure("gold_binding_mismatch", "source contact is malformed")
  part = str(endpoint["part"])
  instance = instances_by_part.get(part)
  if instance is None:
    raise _CaseFailure("gold_binding_mismatch", "gold endpoint part is unknown")
  body_id = instance.body_uuid
  root_body_is_unique = sum(
      candidate.body_uuid == body_id
      for candidate in instances_by_part.values()
  ) == 1
  fusion_index = int(endpoint["fusion_face_index"])
  matches: list[Mapping[str, Any]] = []
  for key in ("entity_one", "entity_two"):
    entity = contact.get(key)
    raw_occurrence = entity.get("occurrence") if isinstance(entity, Mapping) else None
    if instance.source_instance_key["kind"] == "occurrence":
      instance_matches = (
          raw_occurrence == instance.source_instance_key["occurrence_uuid"]
      )
    else:
      instance_matches = (
          raw_occurrence in (None, "") and root_body_is_unique
      )
    if (
        isinstance(entity, Mapping)
        and str(entity.get("body") or "") == body_id
        and instance_matches
        and entity.get("index") == fusion_index
        and entity.get("type") == "BRepFace"
    ):
      matches.append(entity)
  if len(matches) != 1:
    raise _CaseFailure(
        "gold_binding_mismatch",
        "gold endpoint does not uniquely bind to its source contact entity",
    )
  if _surface_type(matches[0].get("surface_type")) != _surface_type(
      endpoint.get("surface_type")
  ):
    raise _CaseFailure("gold_binding_mismatch", "gold/source surface types disagree")
  return matches[0]


def _body_local_entity_payload(entity: Mapping[str, Any]) -> dict[str, Any]:
  """Drop assembly-instance identity while retaining all face-local evidence."""

  payload = dict(entity)
  payload.pop("occurrence", None)
  return payload


def _case_preimage_id(
    *,
    archive: str,
    assembly_relative: str,
    assembly_sha256: str,
    step_sha256s: Iterable[str],
    source_instance_keys: Iterable[Mapping[str, str]],
) -> str:
  preimage = {
      "dataset_version": DATASET_VERSION,
      "archive": archive,
      "assembly_member_path": assembly_relative,
      "assembly_sha256": assembly_sha256,
      "body_sha256s": sorted(set(step_sha256s)),
      "instance_source_keys": sorted(
          _canonical_json(_source_instance_preimage_key(key))
          for key in source_instance_keys
      ),
  }
  return "fusionv2_" + _canonical_sha256(preimage)[:24]


def _audit_case(
    *,
    public_case: Mapping[str, Any],
    private_case: Mapping[str, Any],
    gold_case: Mapping[str, Any],
    volume: _VolumeReceipt,
    tolerances: MappingTolerances,
    run_forward_budget: _ForwardDistanceCallBudget,
    run_triangle_budget: _TriangleEvaluationBudget,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
  case_id = str(public_case.get("id") or "")
  aliases, instances_by_part = _case_instance_bindings(public_case, private_case)
  endpoint_rows = _bind_endpoint_instances(
      _gold_endpoint_rows(gold_case),
      instances_by_part,
  )
  case_forward_budget = _ForwardDistanceCallBudget(
      scope="case",
      maximum=tolerances.maximum_forward_distance_calls_per_case,
  )
  case_triangle_budget = _TriangleEvaluationBudget(
      scope="case",
      maximum=tolerances.maximum_reverse_triangle_evaluations_per_case,
  )
  result: dict[str, Any] = {
      "case_id": case_id,
      "status": "input_integrity_failed",
      "archive": volume.archive,
      "inputs": {},
      "bodies": {},
      "endpoints": [],
  }
  progressed_identities: set[tuple[str, int]] = set()
  endpoint_progress_context: dict[tuple[str, int], tuple[float, int]] = {}
  try:
    binding = private_case.get("source_receipt_binding")
    if not isinstance(binding, Mapping):
      raise _CaseFailure("source_binding_invalid", "case lacks source receipt binding")
    if (
        binding.get("archive") != volume.archive
        or binding.get("receipt_sha256") != volume.receipt_sha256
    ):
      raise _CaseFailure("archive_binding_mismatch", "case/volume receipt binding differs")
    assembly_binding = binding.get("assembly_json")
    if not isinstance(assembly_binding, Mapping):
      raise _CaseFailure("source_binding_invalid", "case lacks assembly binding")
    assembly_relative = _safe_relative_path(
        assembly_binding.get("path"), suffix=".json"
    )
    assembly_path, assembly_digest = _verify_receipt_member(
        volume,
        assembly_relative,
        status="assembly_receipt_mismatch",
        capture=True,
    )
    _verify_digest(
        assembly_digest,
        expected_bytes=assembly_binding.get("bytes"),
        expected_sha256=assembly_binding.get("sha256"),
        status="assembly_receipt_mismatch",
    )
    assembly = _load_json_bytes(
        assembly_digest.captured or b"",
        status="assembly_parse_failed",
    )
    result["inputs"]["assembly_json"] = {
        "path": assembly_relative,
        "bytes": assembly_digest.bytes,
        "sha256": assembly_digest.sha256,
    }

    raw_bodies = assembly.get("bodies")
    raw_contacts = assembly.get("contacts")
    if not isinstance(raw_bodies, Mapping) or not isinstance(raw_contacts, list):
      raise _CaseFailure("assembly_parse_failed", "assembly bodies/contacts are malformed")

    cq = importlib.import_module("cadquery")
    body_runtime: dict[str, dict[str, Any]] = {}
    geometry_cache: dict[str, dict[str, Any]] = {}
    geometry_failures: dict[str, tuple[str, str, str]] = {}
    step_hashes_by_asset: dict[str, str] = {}
    analytic_trim_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for part in (str(alias) for alias in aliases):
      instance = instances_by_part[part]
      body_id = instance.body_uuid
      geometry_asset = instance.geometry_asset
      raw_body = raw_bodies.get(body_id)
      body_result: dict[str, Any] = {
          "geometry_asset": geometry_asset,
          "body_uuid": body_id,
          "source_instance_key": dict(instance.source_instance_key),
          "occurrence_path": list(instance.occurrence_path),
          "is_visible": instance.is_visible,
          "is_grounded": instance.is_grounded,
          "status": "input_integrity_failed",
      }
      result["bodies"][part] = body_result
      cached = geometry_cache.get(geometry_asset)
      if cached is not None:
        body_result.update(
            json.loads(_canonical_json(cached["body_local_receipt"]))
        )
        body_result["geometry_cache_reused"] = True
        body_runtime[part] = {
            "body_id": body_id,
            "shape": cached["shape"],
            "occ_faces": cached["occ_faces"],
            "obj_groups": cached["obj_groups"],
            "analytic_trim_bytes": cached["analytic_trim_bytes"],
            "body_result": body_result,
        }
        continue
      cached_failure = geometry_failures.get(geometry_asset)
      if cached_failure is not None:
        failure_status, status, reason = cached_failure
        body_result.update(
            {
                "status": status,
                "failure_status": failure_status,
                "reason": reason,
                "geometry_cache_reused": True,
            }
        )
        continue
      if not isinstance(raw_body, Mapping):
        reason = "native body is missing from the assembly JSON"
        body_result.update(
            {"failure_status": "source_binding_invalid", "reason": reason}
        )
        geometry_failures[geometry_asset] = (
            "source_binding_invalid",
            "input_integrity_failed",
            reason,
        )
        continue
      step_relative = _safe_relative_path(
          instance.step_binding.get("path"), suffix=".step"
      )
      expected_step_name = _safe_relative_path(raw_body.get("step"), suffix=".step")
      if PurePosixPath(step_relative).name != expected_step_name:
        reason = "receipt-bound STEP does not match the assembly body"
        body_result.update(
            {"failure_status": "source_binding_invalid", "reason": reason}
        )
        geometry_failures[geometry_asset] = (
            "source_binding_invalid",
            "input_integrity_failed",
            reason,
        )
        continue
      try:
        step_path, step_digest = _verify_receipt_member(
            volume,
            step_relative,
            status="step_receipt_mismatch",
            capture=True,
        )
        step_binding = instance.step_binding
        _verify_digest(
            step_digest,
            expected_bytes=step_binding.get("bytes"),
            expected_sha256=step_binding.get("sha256"),
            status="step_receipt_mismatch",
        )
        if step_binding.get("sha1") != step_digest.sha1:
          raise _CaseFailure("step_receipt_mismatch", "STEP SHA1 binding changed")
        assembly_parent = PurePosixPath(assembly_relative).parent
        obj_name = _safe_relative_path(raw_body.get("obj"), suffix=".obj")
        if len(PurePosixPath(obj_name).parts) != 1:
          raise _CaseFailure("unsafe_input_path", "body OBJ path must be assembly-local")
        obj_relative = (assembly_parent / obj_name).as_posix()
        obj_path, obj_digest, obj_binding_mode = _verify_obj_member(
            volume,
            obj_relative,
        )
        smt_relative: str | None = None
        smt_path: Path | None = None
        smt_digest: _Digest | None = None
        smt_binding_mode: str | None = None
        raw_smt_name = raw_body.get("smt")
        if raw_smt_name is not None and not (
            isinstance(raw_smt_name, str) and not raw_smt_name.strip()
        ):
          smt_name = _safe_relative_path(raw_smt_name, suffix=".smt")
          if len(PurePosixPath(smt_name).parts) != 1:
            raise _CaseFailure(
                "unsafe_input_path", "body SMT path must be assembly-local"
            )
          smt_relative = (assembly_parent / smt_name).as_posix()
          smt_path, smt_digest, smt_binding_mode = _verify_obj_member(
              volume,
              smt_relative,
          )
        with tempfile.TemporaryDirectory(
            prefix="neurocad-face-map-parser-"
        ) as staging_directory:
          staging_root = Path(staging_directory)
          staged_step_path = _stage_verified_digest(
              staging_root,
              "verified.step",
              step_digest,
              status="step_receipt_mismatch",
          )
          staged_obj_path = _stage_verified_digest(
              staging_root,
              "verified.obj",
              obj_digest,
              status="obj_receipt_mismatch",
          )
          shape = load_step_shape(staged_step_path)
          step_unit_sidecar = _step_length_unit_sidecar(staged_step_path)
          unit = _unit_contract(
              shape=shape,
              physical=(
                  raw_body.get("physical_properties")
                  if isinstance(raw_body.get("physical_properties"), Mapping)
                  else {}
              ),
              step_unit_sidecar=step_unit_sidecar,
              tolerance=tolerances.unit_relative_error,
          )
          occ_faces = _occ_faces(shape)
          obj_groups = _parse_obj_face_groups(staged_obj_path)
          staged_step_after_use = _plain_file_digest(staged_step_path)
          _verify_digest(
              staged_step_after_use,
              expected_bytes=step_digest.bytes,
              expected_sha256=step_digest.sha256,
              status="step_changed_after_use",
          )
          if staged_step_after_use.sha1 != step_digest.sha1:
            raise _CaseFailure(
                "step_changed_after_use",
                "isolated STEP SHA1 changed while CAD geometry was parsed",
            )
          staged_obj_after_use = _plain_file_digest(staged_obj_path)
          _verify_digest(
              staged_obj_after_use,
              expected_bytes=obj_digest.bytes,
              expected_sha256=obj_digest.sha256,
              status="obj_changed_after_use",
          )
        step_after_use = _plain_file_digest(step_path)
        _verify_digest(
            step_after_use,
            expected_bytes=step_digest.bytes,
            expected_sha256=step_digest.sha256,
            status="step_changed_after_use",
        )
        if step_after_use.sha1 != step_digest.sha1:
          raise _CaseFailure(
              "step_changed_after_use",
              "STEP SHA1 changed while CAD geometry was being parsed",
          )
        obj_after_use = _plain_file_digest(obj_path)
        _verify_digest(
            obj_after_use,
            expected_bytes=obj_digest.bytes,
            expected_sha256=obj_digest.sha256,
            status="obj_changed_after_use",
        )
        if smt_path is not None and smt_digest is not None:
          smt_after_capture = _plain_file_digest(smt_path)
          _verify_digest(
              smt_after_capture,
              expected_bytes=smt_digest.bytes,
              expected_sha256=smt_digest.sha256,
              status="smt_changed_after_capture",
          )
        body_local_receipt = {
            "status": (
                "ready_for_mapping"
                if unit["status"] == "verified_cm_to_occ_mm"
                else str(unit["status"])
            ),
            "step": {
                "path": step_relative,
                "bytes": step_digest.bytes,
                "sha1": step_digest.sha1,
                "sha256": step_digest.sha256,
            },
            "obj": {
                "path": obj_relative,
                "bytes": obj_digest.bytes,
                "sha256": obj_digest.sha256,
                "receipt_binding_mode": obj_binding_mode,
                "authenticated_archive_sha256": (
                    volume.archive_sha256
                    if obj_binding_mode == "authenticated_archive_member_bytes"
                    else None
                ),
            },
            "unit_contract": unit,
            "occ_face_count": len(occ_faces),
        }
        if smt_digest is not None:
          body_local_receipt["smt"] = {
              "path": smt_relative,
              "bytes": smt_digest.bytes,
              "sha256": smt_digest.sha256,
              "receipt_binding_mode": smt_binding_mode,
              "authenticated_archive_sha256": (
                  volume.archive_sha256
                  if smt_binding_mode == "authenticated_archive_member_bytes"
                  else None
              ),
          }
        body_result.update(body_local_receipt)
        body_result["geometry_cache_reused"] = False
        step_hashes_by_asset[geometry_asset] = step_digest.sha256
        geometry_cache[geometry_asset] = {
            "shape": shape,
            "occ_faces": occ_faces,
            "obj_groups": obj_groups,
            "body_local_receipt": body_local_receipt,
            "analytic_trim_bytes": {
                "obj": obj_digest.captured,
                "step": step_digest.captured,
                "smt": (
                    smt_digest.captured if smt_digest is not None else None
                ),
            },
        }
        body_runtime[part] = {
            "body_id": body_id,
            "shape": shape,
            "occ_faces": occ_faces,
            "obj_groups": obj_groups,
            "analytic_trim_bytes": geometry_cache[geometry_asset][
                "analytic_trim_bytes"
            ],
            "body_result": body_result,
        }
      except _CaseFailure as error:
        status = (
            error.status
            if error.status in {
                "unit_contract_failed",
                "physical_property_replay_failed",
            }
            else "input_integrity_failed"
        )
        body_result.update(
            {"status": status, "failure_status": error.status, "reason": str(error)}
        )
        geometry_failures[geometry_asset] = (error.status, status, str(error))
      except Exception as error:  # CAD import failures remain data, not crashes.
        reason = f"{type(error).__name__}:{error}"
        body_result.update(
            {
                "status": "input_integrity_failed",
                "failure_status": "cad_import_failed",
                "reason": reason,
            }
        )
        geometry_failures[geometry_asset] = (
            "cad_import_failed",
            "input_integrity_failed",
            reason,
        )

    expected_assets = {instance.geometry_asset for instance in instances_by_part.values()}
    if set(step_hashes_by_asset) == expected_assets:
      expected_case_id = _case_preimage_id(
          archive=volume.archive,
          assembly_relative=assembly_relative,
          assembly_sha256=assembly_digest.sha256,
          step_sha256s=step_hashes_by_asset.values(),
          source_instance_keys=(
              instance.source_instance_key
              for instance in instances_by_part.values()
          ),
      )
      if expected_case_id != case_id:
        raise _CaseFailure("source_binding_invalid", "case ID preimage does not replay")

    entity_by_identity: dict[tuple[str, int], Mapping[str, Any]] = {}
    endpoint_binding_failures: dict[tuple[str, int], tuple[str, str]] = {}
    for endpoint in endpoint_rows:
      identity = (str(endpoint["part"]), int(endpoint["fusion_face_index"]))
      try:
        entity = _bind_source_entity(
            assembly_contacts=raw_contacts,
            endpoint=endpoint,
            instances_by_part=instances_by_part,
        )
        previous = entity_by_identity.get(identity)
        if previous is not None and _canonical_json(
            _body_local_entity_payload(previous)
        ) != _canonical_json(_body_local_entity_payload(entity)):
          raise _CaseFailure(
              "source_entity_conflict",
              "one Fusion face identity has conflicting source geometry",
          )
        entity_by_identity[identity] = entity
      except _CaseFailure as error:
        endpoint_binding_failures[identity] = (error.status, str(error))

    mapped_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for identity in sorted(set((str(row["part"]), int(row["fusion_face_index"])) for row in endpoint_rows)):
      part, fusion_index = identity
      endpoint_started_at = time.perf_counter()
      runtime = body_runtime.get(part)
      body_result = result["bodies"].get(part, {})
      group = (
          runtime["obj_groups"].get(fusion_index)
          if runtime is not None
          else None
      )
      total_sample_count = 0 if group is None else _obj_sample_count(group)
      _emit_face_map_progress(
          progress_callback,
          event="face_map_endpoint_started",
          case_id=case_id,
          endpoint_face_index=fusion_index,
          endpoint_part=part,
          total_sample_count=total_sample_count,
      )
      progressed_identities.add(identity)
      endpoint_progress_context[identity] = (
          endpoint_started_at,
          total_sample_count,
      )

      if identity in endpoint_binding_failures:
        status, reason = endpoint_binding_failures[identity]
        mapped_by_identity[identity] = _failed_mapping(
            status=status,
            fusion_index=fusion_index,
            message=reason,
        )
        continue
      if runtime is None:
        status = str(body_result.get("failure_status") or "body_input_failed")
        mapped_by_identity[identity] = _failed_mapping(
            status=status,
            fusion_index=fusion_index,
            message=str(
                body_result.get("reason") or "body is not available for mapping"
            ),
        )
        continue
      if body_result.get("status") in {
          "unit_contract_failed",
          "physical_property_replay_failed",
      }:
        body_status = str(body_result["status"])
        mapped_by_identity[identity] = _failed_mapping(
            status=body_status,
            fusion_index=fusion_index,
            message=(
                "body failed the fixed cm-to-OCC-mm contract"
                if body_status == "unit_contract_failed"
                else "source physical properties do not replay against STEP/OCC"
            ),
        )
        continue
      if group is None:
        mapped_by_identity[identity] = _failed_mapping(
            status="obj_face_group_missing",
            fusion_index=fusion_index,
            message="authoritative OBJ lacks g face N for the Fusion endpoint",
        )
        continue
      try:
        instance = instances_by_part[part]
        trim_bytes = runtime["analytic_trim_bytes"]
        analytic_trim_inputs = None
        if (
            instance.source_instance_key["kind"] == "root"
            and not instance.occurrence_path
            and isinstance(trim_bytes.get("obj"), bytes)
            and isinstance(trim_bytes.get("step"), bytes)
        ):
          smt_bytes = trim_bytes.get("smt")
          smt_binding = body_result.get("smt")
          analytic_trim_inputs = AnalyticTrimReplayInputs(
              obj_bytes=trim_bytes["obj"],
              assembly_json_bytes=assembly_digest.captured or b"",
              step_bytes=trim_bytes["step"],
              body_uuid=instance.body_uuid,
              source_face_index=fusion_index,
              source_occurrence_path=(),
              source_path_identity={
                  "obj_archive_member": str(body_result["obj"]["path"]),
                  "step_archive_member": str(body_result["step"]["path"]),
                  "assembly_archive_member": assembly_relative,
              },
              world_transform_row_major=_IDENTITY_WORLD_TRANSFORM,
              source_smt_bytes=(
                  smt_bytes if isinstance(smt_bytes, bytes) else None
              ),
              source_smt_archive_member=(
                  str(smt_binding["path"])
                  if isinstance(smt_binding, Mapping)
                  else None
              ),
          )
        mapped_by_identity[identity] = _map_obj_group_to_occ(
            group=group,
            entity=entity_by_identity[identity],
            occ_faces=runtime["occ_faces"],
            tolerances=tolerances,
            cq=cq,
            case_forward_budget=case_forward_budget,
            run_forward_budget=run_forward_budget,
            case_triangle_budget=case_triangle_budget,
            run_triangle_budget=run_triangle_budget,
            progress_callback=progress_callback,
            progress_case_id=case_id,
            progress_endpoint_part=part,
            analytic_trim_inputs=analytic_trim_inputs,
            analytic_trim_cache=analytic_trim_cache,
        )
      except FaceMapAuditError:
        raise
      except _CaseFailure as error:
        mapped_by_identity[identity] = _failed_mapping(
            status=error.status,
            fusion_index=fusion_index,
            message=str(error),
        )
      except Exception as error:
        mapped_by_identity[identity] = _failed_mapping(
            status="occ_mapping_failed",
            fusion_index=fusion_index,
            message=f"{type(error).__name__}:{error}",
        )

    progress_mappings_by_identity = dict(mapped_by_identity)
    identities_by_face: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for identity, mapping in mapped_by_identity.items():
      if mapping["status"] != "mapped_unique":
        continue
      for occ_index in mapping["raw_occ_face_indices"]:
        identities_by_face[(identity[0], int(occ_index))].append(identity)
    overlapping: set[tuple[str, int]] = set()
    for identities in identities_by_face.values():
      unique = sorted(set(identities))
      if len(unique) > 1:
        overlapping.update(unique)
    for identity in overlapping:
      previous = mapped_by_identity[identity]
      mapped_by_identity[identity] = {
          **_failed_mapping(
              status="overlapping_mapping",
              fusion_index=identity[1],
              message="distinct Fusion faces map to an overlapping OCC face set",
          ),
          "rejected_occ_face_indices": previous["raw_occ_face_indices"],
          "rejected_source_face_signature_sha256s": previous[
              "source_face_signature_sha256s"
          ],
      }

    progress_endpoint_rows: list[dict[str, Any]] = []
    for identity, mapping in sorted(mapped_by_identity.items()):
      part, fusion_index = identity
      endpoint_started_at, total_sample_count = endpoint_progress_context[
          identity
      ]
      progress_mapping = progress_mappings_by_identity[identity]
      checks = progress_mapping.get("checks")
      work = (
          checks.get("forward_distance_work")
          if isinstance(checks, Mapping)
          else None
      )
      proof = (
          work.get("conservative_margin_proof")
          if isinstance(work, Mapping)
          else None
      )
      progress_endpoint_rows.append(
        {
          "endpoint_face_index": fusion_index,
          "endpoint_part": part,
          "processed_sample_count": (
              int(proof.get("completed_sample_query_count", 0))
              if isinstance(proof, Mapping)
              else 0
          ),
          "total_sample_count": total_sample_count,
          "exact_occ_distance_call_count": (
              int(work.get("exact_occ_distance_call_count", 0))
              if isinstance(work, Mapping)
              else 0
          ),
          "pruned_occ_face_candidate_count": (
              int(work.get("pruned_occ_face_candidate_count", 0))
              if isinstance(work, Mapping)
              else 0
          ),
          "started_at": endpoint_started_at,
          "bvh_max_depth": (
              int(work.get("bvh_max_depth", 0))
              if isinstance(work, Mapping)
              else 0
          ),
          "bvh_max_traversal_depth_visited": (
              int(work.get("bvh_max_traversal_depth_visited", 0))
              if isinstance(work, Mapping)
              else 0
          ),
          "status_before_after_use": str(mapping.get("status") or "unknown"),
        }
      )
    result["_progress_endpoints"] = progress_endpoint_rows

    endpoints: list[dict[str, Any]] = []
    for endpoint in endpoint_rows:
      identity = (str(endpoint["part"]), int(endpoint["fusion_face_index"]))
      endpoints.append(
          {
              **dict(endpoint),
              **mapped_by_identity[identity],
          }
      )
    result["endpoints"] = endpoints
    result["face_mappings"] = _face_mappings_from_endpoints(endpoints)
    statuses = Counter(str(row["status"]) for row in endpoints)
    input_failure_statuses = {
        "archive_binding_mismatch",
        "assembly_receipt_mismatch",
        "step_receipt_mismatch",
        "obj_receipt_mismatch",
        "unsafe_input_path",
        "missing_input_member",
        "input_changed_while_hashing",
        "cad_import_failed",
        "source_binding_invalid",
        "assembly_parse_failed",
    }
    body_statuses = Counter(
        str(body.get("status") or "missing")
        for body in result["bodies"].values()
        if isinstance(body, Mapping)
    )
    all_bodies_ready = (
        set(result["bodies"]) == set(instances_by_part)
        and
        sum(body_statuses.values()) == len(instances_by_part)
        and body_statuses.get("ready_for_mapping", 0) == len(instances_by_part)
    )
    if (
        body_statuses.get("input_integrity_failed", 0) > 0
        or any(status in input_failure_statuses for status in statuses)
    ):
      result["status"] = "input_integrity_failed"
    elif all_bodies_ready and statuses.get("mapped_unique", 0) == len(endpoints):
      result["status"] = "mapped_complete"
    else:
      result["status"] = "mapping_incomplete"
    case_forward_work_rows = _forward_work_receipts(
        result["face_mappings"]
    )
    result["summary"] = {
        "endpoint_count": len(endpoints),
        "mapped_endpoint_count": statuses.get("mapped_unique", 0),
        "status_counts": dict(sorted(statuses.items())),
        "all_endpoints_mapped": statuses.get("mapped_unique", 0) == len(endpoints),
        "body_status_counts": dict(sorted(body_statuses.items())),
        "all_bodies_ready": all_bodies_ready,
        "case_complete": result["status"] == "mapped_complete",
        "forward_distance_work": {
            "schema_version": _FORWARD_DISTANCE_WORK_SCHEMA_VERSION,
            "scope": "case",
            **case_forward_budget.receipt(),
            "work_cap_exhausted": (
                case_forward_budget.work_cap_exhaustion_count > 0
            ),
            **_forward_work_aggregates(case_forward_work_rows),
        },
        "reverse_distance_work": {
            "schema_version": _TRIANGLE_BVH_WORK_SCHEMA_VERSION,
            "scope": "case",
            **case_triangle_budget.receipt(),
            "work_cap_exhausted": (
                case_triangle_budget.work_cap_exhaustion_count > 0
            ),
        },
    }
    return result
  except _CaseFailure as error:
    failure_progress_rows: list[dict[str, Any]] = []
    for part, fusion_index in sorted(
        set(
            (str(row["part"]), int(row["fusion_face_index"]))
            for row in endpoint_rows
        )
    ):
      identity = (part, fusion_index)
      if identity not in progressed_identities:
        endpoint_progress_context[identity] = (time.perf_counter(), 0)
        _emit_face_map_progress(
            progress_callback,
            event="face_map_endpoint_started",
            case_id=case_id,
            endpoint_face_index=fusion_index,
            endpoint_part=part,
        )
      started_at, total_sample_count = endpoint_progress_context[identity]
      failure_progress_rows.append(
          {
              "endpoint_face_index": fusion_index,
              "endpoint_part": part,
              "processed_sample_count": 0,
              "total_sample_count": total_sample_count,
              "exact_occ_distance_call_count": 0,
              "pruned_occ_face_candidate_count": 0,
              "started_at": started_at,
              "bvh_max_depth": 0,
              "bvh_max_traversal_depth_visited": 0,
              "status_before_after_use": error.status,
          }
      )
    result["status"] = "input_integrity_failed"
    result["failure_status"] = error.status
    result["reason"] = str(error)
    result["endpoints"] = _failure_endpoints(
        endpoint_rows,
        status=error.status,
        reason=str(error),
    )
    result["face_mappings"] = _face_mappings_from_endpoints(result["endpoints"])
    result["_progress_endpoints"] = failure_progress_rows
    case_forward_work_rows = _forward_work_receipts(
        result["face_mappings"]
    )
    result["summary"] = {
        "endpoint_count": len(endpoint_rows),
        "mapped_endpoint_count": 0,
        "status_counts": {error.status: len(endpoint_rows)},
        "all_endpoints_mapped": False,
        "body_status_counts": {},
        "all_bodies_ready": False,
        "case_complete": False,
        "forward_distance_work": {
            "schema_version": _FORWARD_DISTANCE_WORK_SCHEMA_VERSION,
            "scope": "case",
            **case_forward_budget.receipt(),
            "work_cap_exhausted": (
                case_forward_budget.work_cap_exhaustion_count > 0
            ),
            **_forward_work_aggregates(case_forward_work_rows),
        },
        "reverse_distance_work": {
            "schema_version": _TRIANGLE_BVH_WORK_SCHEMA_VERSION,
            "scope": "case",
            **case_triangle_budget.receipt(),
            "work_cap_exhausted": (
                case_triangle_budget.work_cap_exhaustion_count > 0
            ),
        },
    }
    return result


def _environment_receipt(
    implementation_source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
  cadquery = importlib.import_module("cadquery")
  ocp = importlib.import_module("OCP")
  return {
      "python": platform.python_version(),
      "python_implementation": platform.python_implementation(),
      "cadquery": str(getattr(cadquery, "__version__", "unknown")),
      "ocp": str(getattr(ocp, "__version__", "unknown")),
      "platform": sys.platform,
      "implementation": {
          "module": "neurocad.fusion_face_mapper",
          "version": MAPPER_VERSION,
          "source_sha256s": dict(implementation_source_sha256s),
      },
  }


def select_development_case_ids(
    source_case_ids: Sequence[str],
    *,
    development_max_cases: int | None = None,
    development_case_ids: Sequence[str] | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
  """Select a deterministic, development-only subset of source case IDs."""

  source_ids = list(source_case_ids)
  if (
      not source_ids
      or any(not isinstance(case_id, str) or not case_id for case_id in source_ids)
      or len(set(source_ids)) != len(source_ids)
  ):
    raise ValueError("source case IDs must be nonempty and unique")
  if source_ids != sorted(source_ids):
    raise ValueError("source case IDs must be in deterministic source order")
  has_prefix = development_max_cases is not None
  has_explicit = development_case_ids is not None
  has_any_shard = shard_index is not None or shard_count is not None
  if (shard_index is None) != (shard_count is None):
    raise ValueError("shard_index and shard_count must be provided together")
  if sum((has_prefix, has_explicit, has_any_shard)) > 1:
    raise ValueError(
        "development prefix, explicit case IDs, and shard selection are mutually exclusive"
    )
  source_count = len(source_ids)
  if has_explicit:
    assert development_case_ids is not None
    requested = list(development_case_ids)
    if not requested or any(
        not isinstance(case_id, str) or not case_id for case_id in requested
    ):
      raise ValueError("development_case_ids must contain nonempty strings")
    if len(set(requested)) != len(requested):
      raise ValueError("development_case_ids contains duplicated case IDs")
    missing = sorted(set(requested) - set(source_ids))
    if missing:
      raise ValueError(
          "development case IDs are not present in the family source: "
          + ", ".join(missing)
      )
    requested_set = set(requested)
    selected = [case_id for case_id in source_ids if case_id in requested_set]
    return selected, {
        "selection": "explicit_case_ids_in_source_order",
        "requested_case_ids": requested,
        "selected_case_ids": selected,
        "source_case_count": source_count,
    }
  if has_any_shard:
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
    ):
      raise ValueError("shard_index and shard_count must be integers")
    assert shard_index is not None and shard_count is not None
    if shard_count <= 0:
      raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
      raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    selected = [
        case_id
        for ordinal, case_id in enumerate(source_ids)
        if ordinal % shard_count == shard_index
    ]
    if not selected:
      raise ValueError("selected development shard contains no source cases")
    return selected, {
        "selection": "deterministic_source_order_modulo_shard",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_case_ids": selected,
        "source_case_count": source_count,
    }
  maximum = 1 if development_max_cases is None else development_max_cases
  if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
    raise ValueError("development_max_cases must be a positive integer")
  if maximum > source_count:
    raise ValueError("development_max_cases exceeds the family source case count")
  selected = source_ids[:maximum]
  return selected, {
      "selection": "deterministic_case_id_prefix",
      "requested_max_cases": maximum,
      "selected_case_ids": selected,
      "source_case_count": source_count,
  }


def audit_family_source(
    family_source_path: str | Path,
    archive_root: str | Path,
    *,
    development_max_cases: int | None = None,
    development_case_ids: Sequence[str] | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    tolerances: MappingTolerances | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
  """Audit a deterministic development selection and return a private receipt.

  This function never changes ``gold_interface_certificate_ready`` and never
  emits a formal-ready artifact, even when every development endpoint maps.
  """

  active_tolerances = tolerances or MappingTolerances()
  active_tolerances.validate()
  run_forward_budget = _ForwardDistanceCallBudget(
      scope="run",
      maximum=active_tolerances.maximum_forward_distance_calls_per_run,
  )
  run_triangle_budget = _TriangleEvaluationBudget(
      scope="run",
      maximum=(
          active_tolerances.maximum_reverse_triangle_evaluations_per_run
      ),
  )
  try:
    implementation_hashes_start = _implementation_source_sha256s()
  except _CaseFailure as error:
    raise FaceMapAuditError(f"implementation source is not stable: {error}") from error
  for name, imported_sha256 in _IMPORT_TIME_EXECUTED_SOURCE_SHA256S.items():
    if implementation_hashes_start.get(name) != imported_sha256:
      raise FaceMapAuditError(
          f"executed implementation source changed after import: {name}"
      )
  source_path = Path(family_source_path)
  source, source_digest = _load_family_source(source_path)
  _validate_family_source(source)
  raw_cases = source.get("cases")
  if not isinstance(raw_cases, list) or not raw_cases:
    raise FaceMapAuditError("family source cases are missing")
  case_ids = [str(case.get("id") or "") if isinstance(case, Mapping) else "" for case in raw_cases]
  if not all(case_ids) or len(set(case_ids)) != len(case_ids):
    raise FaceMapAuditError("family source case IDs are invalid")
  if case_ids != sorted(case_ids):
    raise FaceMapAuditError("family source cases are not in deterministic case-id order")
  gold_by_case = _gold_cases(source)
  if set(gold_by_case) != set(case_ids):
    raise FaceMapAuditError("family source cases and private gold cases differ")
  private_by_case = _private_source_cases(source)
  if set(private_by_case) != set(case_ids):
    raise FaceMapAuditError("public and private source-binding cases differ")
  volume_summaries = _volume_summary_by_archive(source)
  selected_case_ids, selection_rule = select_development_case_ids(
      case_ids,
      development_max_cases=development_max_cases,
      development_case_ids=development_case_ids,
      shard_index=shard_index,
      shard_count=shard_count,
  )
  try:
    root = Path(archive_root).expanduser().resolve(strict=True)
  except OSError as error:
    raise FaceMapAuditError(f"archive root is missing: {archive_root}") from error
  if _is_reparse_point(root) or not root.is_dir():
    raise FaceMapAuditError("archive root is not a plain directory")
  selected_id_set = set(selected_case_ids)
  selected_cases = [
      case for case in raw_cases if str(case.get("id")) in selected_id_set
  ]

  loaded_volumes: dict[str, _VolumeReceipt | _CaseFailure] = {}
  case_receipts: list[dict[str, Any]] = []
  case_progress_context: dict[str, tuple[float, int, int]] = {}
  for selected_ordinal, raw_case in enumerate(selected_cases, start=1):
    if not isinstance(raw_case, Mapping):
      raise FaceMapAuditError("family source case is malformed")
    case_id = str(raw_case["id"])
    case_started_at = time.perf_counter()
    case_progress_context[case_id] = (
        case_started_at,
        selected_ordinal,
        len(selected_cases),
    )
    _emit_face_map_progress(
        progress_callback,
        event="face_map_case_started",
        case_id=case_id,
        selected_case_ordinal=selected_ordinal,
        selected_case_count=len(selected_cases),
    )
    private_case = private_by_case[case_id]
    binding = private_case.get("source_receipt_binding")
    archive = str(binding.get("archive") or "") if isinstance(binding, Mapping) else ""
    try:
      _, instances_by_part = _case_instance_bindings(raw_case, private_case)
      endpoint_rows = _bind_endpoint_instances(
          _gold_endpoint_rows(gold_by_case[case_id]),
          instances_by_part,
      )
    except _CaseFailure as error:
      raise FaceMapAuditError(
          f"family source case identity binding is invalid: {case_id}: {error}"
      ) from error
    if archive not in loaded_volumes:
      try:
        summary = volume_summaries.get(archive)
        if summary is None:
          raise _CaseFailure("archive_binding_mismatch", "case archive has no source receipt")
        loaded_volumes[archive] = _load_volume_receipt(root, archive, summary)
      except _CaseFailure as error:
        loaded_volumes[archive] = error
    volume_or_error = loaded_volumes[archive]
    if isinstance(volume_or_error, _CaseFailure):
      empty_case_forward_budget = _ForwardDistanceCallBudget(
          scope="case",
          maximum=active_tolerances.maximum_forward_distance_calls_per_case,
      )
      endpoints = _failure_endpoints(
          endpoint_rows,
          status=volume_or_error.status,
          reason=str(volume_or_error),
      )
      failed_progress_rows: list[dict[str, Any]] = []
      for part, fusion_index in sorted(
          set(
              (str(row["part"]), int(row["fusion_face_index"]))
              for row in endpoint_rows
          )
      ):
        endpoint_started_at = time.perf_counter()
        _emit_face_map_progress(
            progress_callback,
            event="face_map_endpoint_started",
            case_id=case_id,
            endpoint_face_index=fusion_index,
            endpoint_part=part,
        )
        failed_progress_rows.append(
            {
                "endpoint_face_index": fusion_index,
                "endpoint_part": part,
                "processed_sample_count": 0,
                "total_sample_count": 0,
                "exact_occ_distance_call_count": 0,
                "pruned_occ_face_candidate_count": 0,
                "started_at": endpoint_started_at,
                "bvh_max_depth": 0,
                "bvh_max_traversal_depth_visited": 0,
                "status_before_after_use": volume_or_error.status,
            }
        )
      case_receipts.append(
          {
              "case_id": case_id,
              "archive": archive,
              "status": "input_integrity_failed",
              "failure_status": volume_or_error.status,
              "reason": str(volume_or_error),
              "inputs": {},
              "bodies": {},
              "endpoints": endpoints,
              "face_mappings": _face_mappings_from_endpoints(endpoints),
              "_progress_endpoints": failed_progress_rows,
              "summary": {
                  "endpoint_count": len(endpoints),
                  "mapped_endpoint_count": 0,
                  "status_counts": {volume_or_error.status: len(endpoints)},
                  "all_endpoints_mapped": False,
                  "body_status_counts": {},
                  "all_bodies_ready": False,
                  "case_complete": False,
                  "forward_distance_work": {
                      "schema_version": _FORWARD_DISTANCE_WORK_SCHEMA_VERSION,
                      "scope": "case",
                      **empty_case_forward_budget.receipt(),
                      "work_cap_exhausted": False,
                      **_forward_work_aggregates([]),
                  },
              },
          }
      )
      continue
    case_receipt = _audit_case(
        public_case=raw_case,
        private_case=private_case,
        gold_case=gold_by_case[case_id],
        volume=volume_or_error,
        tolerances=active_tolerances,
        run_forward_budget=run_forward_budget,
        run_triangle_budget=run_triangle_budget,
        progress_callback=progress_callback,
    )
    case_receipts.append(case_receipt)
  for archive, volume_or_error in loaded_volumes.items():
    if (
        isinstance(volume_or_error, _CaseFailure)
        or volume_or_error.archive_path is None
        or volume_or_error.archive_bytes is None
        or volume_or_error.archive_sha256 is None
    ):
      continue
    try:
      archive_after_use = _plain_file_digest(volume_or_error.archive_path)
      _verify_digest(
          archive_after_use,
          expected_bytes=volume_or_error.archive_bytes,
          expected_sha256=volume_or_error.archive_sha256,
          status="archive_changed_after_use",
      )
    except _CaseFailure as error:
      reason = (
          "authenticated archive changed while OBJ members were being consumed: "
          f"{error}"
      )
      for case_receipt in case_receipts:
        if case_receipt.get("archive") == archive:
          _invalidate_case_receipt(
              case_receipt,
              status="archive_changed_after_use",
              reason=reason,
          )
  endpoint_statuses = Counter(
      str(endpoint["status"])
      for case in case_receipts
      for endpoint in case["endpoints"]
  )
  total_endpoints = sum(endpoint_statuses.values())
  mapped_endpoints = endpoint_statuses.get("mapped_unique", 0)
  all_cases_complete = all(
      case.get("status") == "mapped_complete" for case in case_receipts
  )
  reverse_work_rows = [
      work
      for case in case_receipts
      for mapping in case.get("face_mappings", [])
      if isinstance(mapping, Mapping)
      for checks in [mapping.get("checks")]
      if isinstance(checks, Mapping)
      for work in [checks.get("reverse_distance_work")]
      if isinstance(work, Mapping)
  ]
  forward_work_rows = [
      work
      for case in case_receipts
      for mapping in case.get("face_mappings", [])
      if isinstance(mapping, Mapping)
      for checks in [mapping.get("checks")]
      if isinstance(checks, Mapping)
      for work in [checks.get("forward_distance_work")]
      if isinstance(work, Mapping)
  ]
  try:
    implementation_hashes_end = _implementation_source_sha256s()
  except _CaseFailure as error:
    raise FaceMapAuditError(f"implementation source is not stable: {error}") from error
  if implementation_hashes_end != implementation_hashes_start:
    raise FaceMapAuditError("implementation source changed during face-map audit")
  for case_receipt in case_receipts:
    case_id = str(case_receipt.get("case_id") or "")
    progress_rows = case_receipt.pop("_progress_endpoints", [])
    final_mappings = {
        (str(mapping.get("part") or ""), int(mapping["fusion_face_index"])): mapping
        for mapping in case_receipt.get("face_mappings", [])
        if isinstance(mapping, Mapping)
    }
    invalidated = case_receipt.get("failure_status") in {
        "archive_changed_after_use",
        "step_changed_after_use",
        "obj_changed_after_use",
    }
    for raw_progress in progress_rows:
      part = str(raw_progress["endpoint_part"])
      fusion_index = int(raw_progress["endpoint_face_index"])
      final_mapping = final_mappings.get((part, fusion_index), {})
      status = str(final_mapping.get("status") or "unknown")
      event = (
          "face_map_endpoint_invalidated"
          if invalidated
          else (
              "face_map_endpoint_completed"
              if status == "mapped_unique"
              else "face_map_endpoint_failed"
          )
      )
      _emit_face_map_progress(
          progress_callback,
          event=event,
          case_id=case_id,
          endpoint_face_index=fusion_index,
          endpoint_part=part,
          processed_sample_count=int(raw_progress["processed_sample_count"]),
          total_sample_count=int(raw_progress["total_sample_count"]),
          exact_occ_distance_call_count=int(
              raw_progress["exact_occ_distance_call_count"]
          ),
          pruned_occ_face_candidate_count=int(
              raw_progress["pruned_occ_face_candidate_count"]
          ),
          elapsed_seconds=(
              time.perf_counter() - float(raw_progress["started_at"])
          ),
          bvh_max_depth=int(raw_progress["bvh_max_depth"]),
          bvh_max_traversal_depth_visited=int(
              raw_progress["bvh_max_traversal_depth_visited"]
          ),
          status=status,
      )
    case_forward_work = case_receipt.get("summary", {}).get(
        "forward_distance_work", {}
    )
    case_started_at, selected_ordinal, selected_count = case_progress_context[
        case_id
    ]
    case_status = str(case_receipt.get("status") or "unknown")
    case_event = (
        "face_map_case_invalidated"
        if invalidated
        else (
            "face_map_case_completed"
            if case_status == "mapped_complete"
            else "face_map_case_failed"
        )
    )
    _emit_face_map_progress(
        progress_callback,
        event=case_event,
        case_id=case_id,
        exact_occ_distance_call_count=int(
            case_forward_work.get("completed_distance_call_count", 0)
        ),
        pruned_occ_face_candidate_count=int(
            case_forward_work.get("pruned_occ_face_candidate_count", 0)
        ),
        elapsed_seconds=time.perf_counter() - case_started_at,
        bvh_max_depth=int(case_forward_work.get("bvh_max_depth", 0)),
        bvh_max_traversal_depth_visited=int(
            case_forward_work.get("bvh_max_traversal_depth_visited", 0)
        ),
        status=case_status,
        selected_case_ordinal=selected_ordinal,
        selected_case_count=selected_count,
    )
  receipt = {
      "schema_version": FACE_MAP_AUDIT_SCHEMA_VERSION,
      "visibility": "private_evaluation_only",
      "formal_gold_eligible": False,
      "gold_interface_certificate_ready": False,
      "blocking_reason": "development_face_map_audit_requires_separate_formal_gold_gate",
      "mapper_version": MAPPER_VERSION,
      "endpoint_lookup_contract": {
          "schema_version": ENDPOINT_LOOKUP_SCHEMA_VERSION,
          "key_fields": ["case_id", "part", "fusion_face_index"],
          "part_semantics": PART_INSTANCE_SEMANTICS,
          "success_status": "mapped_unique",
          "fusion_face_index_semantics": "lookup_key_only_never_occ_index",
      },
      "source": {
          "family_source_path": str(Path(family_source_path).expanduser().resolve()),
          "family_source_bytes": source_digest.bytes,
          "family_source_sha256": source_digest.sha256,
          "family_source_formal": False,
          "source_gold_interface_certificate_ready": False,
          "family_source_schema_version": FAMILY_SOURCE_SCHEMA_VERSION,
          "private_source_bindings_schema_version": (
              PRIVATE_SOURCE_BINDINGS_SCHEMA_VERSION
          ),
          "instance_identity_schema_version": INSTANCE_IDENTITY_SCHEMA_VERSION,
          "dataset_version": DATASET_VERSION,
          "receipt_set_sha256": source["dataset_provenance"].get("receipt_set_sha256"),
      },
      "development": {
          "enabled": True,
          **selection_rule,
          "audited_case_ids": [str(case["case_id"]) for case in case_receipts],
          "warning": "development-only receipt; forbidden as final gold",
      },
      "unit_contract": {
          "fusion_json_and_obj_length_unit": "cm",
          "occ_runtime_length_unit": "mm",
          "length_scale": 10.0,
          "area_scale": 100.0,
          "volume_scale": 1000.0,
      },
      "tolerances": asdict(active_tolerances),
      "environment": _environment_receipt(implementation_hashes_start),
      "cases": case_receipts,
      "summary": {
          "case_count": len(case_receipts),
          "endpoint_count": total_endpoints,
          "mapped_endpoint_count": mapped_endpoints,
          "status_counts": dict(sorted(endpoint_statuses.items())),
          "all_cases_complete": all_cases_complete,
          "all_endpoints_mapped": (
              mapped_endpoints == total_endpoints
              and total_endpoints > 0
              and all_cases_complete
          ),
          "forward_distance_work": {
              "schema_version": _FORWARD_DISTANCE_WORK_SCHEMA_VERSION,
              "scope": "run",
              **run_forward_budget.receipt(),
              "work_cap_exhausted": (
                  run_forward_budget.work_cap_exhaustion_count > 0
              ),
              **_forward_work_aggregates(forward_work_rows),
          },
          "reverse_distance_work": {
              "schema_version": _TRIANGLE_BVH_WORK_SCHEMA_VERSION,
              "scope": "run",
              **run_triangle_budget.receipt(),
              "work_cap_exhausted": (
                  run_triangle_budget.work_cap_exhaustion_count > 0
              ),
              "mapping_count_with_work_receipt": len(reverse_work_rows),
              "work_cap_exhausted_mapping_count": sum(
                  work.get("work_cap_exhausted") is True
                  for work in reverse_work_rows
              ),
              "completed_reverse_sample_query_count": sum(
                  int(work.get("completed_reverse_sample_query_count", 0))
                  for work in reverse_work_rows
              ),
              "aabb_lower_bound_evaluation_count": sum(
                  int(work.get("aabb_lower_bound_evaluation_count", 0))
                  for work in reverse_work_rows
              ),
              "pruned_triangle_candidate_count": sum(
                  int(work.get("pruned_triangle_candidate_count", 0))
                  for work in reverse_work_rows
              ),
          },
      },
  }
  receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
  return receipt


def _validate_analytic_trim_evidence_structure(
    mapping: Mapping[str, Any],
    evidence: Any,
    *,
    fusion_index: int,
    body_uuid: str,
    occ_indices: Sequence[int],
) -> None:
  planar = (
      mapping.get("acceptance_basis")
      == PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS
  )
  expected_schema = (
      "planar_closed_circle_chordal_trim_certificate.v1"
      if planar
      else "cylinder_rectangular_analytic_trim_certificate.v1"
  )
  expected_algorithm = (
      PLANAR_CHORDAL_TRIM_ALGORITHM_REVISION
      if planar
      else ANALYTIC_TRIM_ALGORITHM_REVISION
  )
  if not isinstance(evidence, Mapping) or set(evidence) != {
      "schema_version",
      "algorithm_revision",
      "receipt_sha256",
      "receipt",
      "input_binding",
  }:
    raise FaceMapAuditError("analytic trim certificate fields differ")
  if (
      evidence.get("schema_version") != expected_schema
      or evidence.get("algorithm_revision") != expected_algorithm
      or len(occ_indices) != 1
  ):
    raise FaceMapAuditError("analytic trim certificate contract differs")
  receipt = evidence.get("receipt")
  binding = evidence.get("input_binding")
  if not isinstance(receipt, Mapping) or not isinstance(binding, Mapping):
    raise FaceMapAuditError("analytic trim certificate payload is malformed")
  receipt_sha256 = hashlib.sha256(
      _canonical_json(receipt).encode("utf-8")
  ).hexdigest()
  if evidence.get("receipt_sha256") != receipt_sha256:
    raise FaceMapAuditError("analytic trim certificate hash differs")
  unsigned_receipt = dict(receipt)
  receipt_self_hash = unsigned_receipt.pop("receipt_payload_sha256", None)
  if (
      receipt.get("schema_version") != expected_schema
      or receipt.get("decision") != "certified_equivalent"
      or receipt.get("formal_authorized") is not False
      or receipt_self_hash != _canonical_sha256(unsigned_receipt)
  ):
    raise FaceMapAuditError("analytic trim certificate self-hash differs")
  expected_binding_fields = {
      "algorithm_revision",
      "obj_bytes_sha256",
      "assembly_json_bytes_sha256",
      "step_bytes_sha256",
      "body_uuid",
      "source_face_index",
      "raw_occ_face_index",
      "source_occurrence_path",
      "source_path_identity",
      "world_transform_row_major",
      "input_binding_sha256",
  }
  if planar:
    expected_binding_fields.update({
        "source_smt_bytes_sha256",
        "source_smt_archive_member",
    })
  if set(binding) != expected_binding_fields:
    raise FaceMapAuditError("analytic trim input binding fields differ")
  unsigned_binding = dict(binding)
  input_hash = unsigned_binding.pop("input_binding_sha256", None)
  if input_hash != _canonical_sha256(unsigned_binding):
    raise FaceMapAuditError("analytic trim input binding self-hash differs")
  if (
      binding.get("algorithm_revision") != expected_algorithm
      or binding.get("body_uuid") != body_uuid
      or binding.get("source_face_index") != fusion_index
      or binding.get("raw_occ_face_index") != occ_indices[0]
      or binding.get("source_occurrence_path") != []
      or tuple(binding.get("world_transform_row_major") or ())
      != _IDENTITY_WORLD_TRANSFORM
      or any(
          not isinstance(binding.get(field), str)
          or _SHA256_RE.fullmatch(str(binding.get(field))) is None
          for field in (
              "obj_bytes_sha256",
              "assembly_json_bytes_sha256",
              "step_bytes_sha256",
          )
      )
  ):
    raise FaceMapAuditError("analytic trim input binding identity differs")
  source_binding = receipt.get("source_binding")
  source_smt_binding = receipt.get("source_smt_binding")
  occ_binding = receipt.get("occ_binding")
  if (
      not isinstance(source_binding, Mapping)
      or not isinstance(occ_binding, Mapping)
      or source_binding.get("body_uuid") != body_uuid
      or source_binding.get("face_index") != fusion_index
      or source_binding.get("obj_bytes_sha256")
      != binding.get("obj_bytes_sha256")
      or source_binding.get("assembly_json_bytes_sha256")
      != binding.get("assembly_json_bytes_sha256")
      or occ_binding.get("step_bytes_sha256")
      != binding.get("step_bytes_sha256")
      or occ_binding.get("raw_occ_face_index") != occ_indices[0]
      or mapping.get("source_face_signature_sha256s")
      != [occ_binding.get("deterministic_face_signature_sha256")]
      or (
          planar
          and (
              not isinstance(source_smt_binding, Mapping)
              or source_smt_binding.get("source_smt_bytes_sha256")
              != binding.get("source_smt_bytes_sha256")
              or source_smt_binding.get("source_smt_archive_member")
              != binding.get("source_smt_archive_member")
              or source_smt_binding.get("body_uuid") != body_uuid
              or source_smt_binding.get("source_face_index")
              != fusion_index
          )
      )
  ):
    raise FaceMapAuditError("analytic trim certificate/input identity differs")
  checks = mapping.get("checks")
  if (
      not isinstance(checks, Mapping)
      or checks.get("candidate_ranking_unique_margin_safe") is not True
      or checks.get("candidate_occ_face_indices") != list(occ_indices)
      or checks.get("analytic_trim_introduced_new_distance_threshold") is not False
      or (
          planar
          and checks.get("locked_boundary_distance_tolerance_mm") != 0.2
      )
  ):
    raise FaceMapAuditError("analytic trim mapper checks differ")


def build_mapped_endpoint_lookup(
    receipt: Mapping[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
  """Validate a private receipt and expose only uniquely mapped face identities.

  Consumers must use the tuple key ``(case_id, part, fusion_face_index)``.
  The Fusion index is never an OCC index; the only executable identities are
  the receipt's ``raw_occ_face_indices`` paired with their source-face hashes.
  Failed or ambiguous mappings are omitted, so a missing key is a hard gate.
  """

  if not isinstance(receipt, Mapping):
    raise FaceMapAuditError("face-map receipt is not an object")
  expected_payload_sha256 = receipt.get("receipt_payload_sha256")
  unsigned = dict(receipt)
  unsigned.pop("receipt_payload_sha256", None)
  if (
      not isinstance(expected_payload_sha256, str)
      or _SHA256_RE.fullmatch(expected_payload_sha256) is None
      or _canonical_sha256(unsigned) != expected_payload_sha256
  ):
    raise FaceMapAuditError("face-map receipt payload SHA256 is invalid")
  expected_contract = {
      "schema_version": ENDPOINT_LOOKUP_SCHEMA_VERSION,
      "key_fields": ["case_id", "part", "fusion_face_index"],
      "part_semantics": PART_INSTANCE_SEMANTICS,
      "success_status": "mapped_unique",
      "fusion_face_index_semantics": "lookup_key_only_never_occ_index",
  }
  schema_version = receipt.get("schema_version")
  if (
      schema_version
      not in {
          LEGACY_FACE_MAP_AUDIT_SCHEMA_VERSION,
          FACE_MAP_AUDIT_SCHEMA_VERSION,
      }
      or receipt.get("visibility") != "private_evaluation_only"
      or receipt.get("formal_gold_eligible") is not False
      or receipt.get("gold_interface_certificate_ready") is not False
      or receipt.get("endpoint_lookup_contract") != expected_contract
  ):
    raise FaceMapAuditError("face-map receipt contract is invalid")
  source_binding = receipt.get("source")
  if (
      not isinstance(source_binding, Mapping)
      or source_binding.get("family_source_schema_version")
      != FAMILY_SOURCE_SCHEMA_VERSION
      or source_binding.get("private_source_bindings_schema_version")
      != PRIVATE_SOURCE_BINDINGS_SCHEMA_VERSION
      or source_binding.get("instance_identity_schema_version")
      != INSTANCE_IDENTITY_SCHEMA_VERSION
  ):
    raise FaceMapAuditError("face-map receipt source contract is invalid")
  raw_cases = receipt.get("cases")
  if not isinstance(raw_cases, list):
    raise FaceMapAuditError("face-map receipt cases are malformed")
  lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
  seen_keys: set[tuple[str, str, int]] = set()
  for case in raw_cases:
    if not isinstance(case, Mapping):
      raise FaceMapAuditError("face-map receipt case is malformed")
    raw_case_id = case.get("case_id")
    case_id = raw_case_id if isinstance(raw_case_id, str) else ""
    mappings = case.get("face_mappings")
    if not case_id or not isinstance(mappings, list):
      raise FaceMapAuditError("face-map receipt case lacks face_mappings")
    identity_by_part: dict[str, tuple[str, str, str]] = {}
    part_by_instance: dict[str, str] = {}
    for mapping in mappings:
      if not isinstance(mapping, Mapping):
        raise FaceMapAuditError("face-map receipt mapping row is malformed")
      raw_part = mapping.get("part")
      raw_geometry_asset = mapping.get("geometry_asset")
      raw_body_uuid = mapping.get("body_uuid")
      part = raw_part if isinstance(raw_part, str) else ""
      geometry_asset = (
          raw_geometry_asset if isinstance(raw_geometry_asset, str) else ""
      )
      body_uuid = raw_body_uuid if isinstance(raw_body_uuid, str) else ""
      fusion_index = mapping.get("fusion_face_index")
      occ_indices = mapping.get("raw_occ_face_indices")
      signatures = mapping.get("source_face_signature_sha256s")
      if (
          not part
          or not geometry_asset
          or not body_uuid
          or not isinstance(fusion_index, int)
          or isinstance(fusion_index, bool)
          or fusion_index < 0
          or not isinstance(occ_indices, list)
          or not isinstance(signatures, list)
      ):
        raise FaceMapAuditError("endpoint lookup identity is malformed")
      try:
        source_instance_key = _source_instance_key(
            mapping.get("source_instance_key")
        )
      except _CaseFailure as error:
        raise FaceMapAuditError(
            f"endpoint lookup instance identity is malformed: {error}"
        ) from error
      if source_instance_key["body_uuid"] != body_uuid:
        raise FaceMapAuditError("endpoint lookup instance identity/body differs")
      canonical_instance = _canonical_json(source_instance_key)
      part_identity = (geometry_asset, body_uuid, canonical_instance)
      previous_part_identity = identity_by_part.setdefault(part, part_identity)
      if previous_part_identity != part_identity:
        raise FaceMapAuditError("endpoint lookup part instance identity conflicts")
      previous_part = part_by_instance.setdefault(canonical_instance, part)
      if previous_part != part:
        raise FaceMapAuditError("endpoint lookup assembly instance is duplicated")
      key = (case_id, part, fusion_index)
      if key in seen_keys:
        raise FaceMapAuditError("endpoint lookup identity is duplicated")
      seen_keys.add(key)
      if mapping.get("mapping_mode") != "authoritative_obj_face_group":
        raise FaceMapAuditError("endpoint lookup mapping mode is not authoritative")
      acceptance_basis = mapping.get("acceptance_basis")
      analytic_evidence = mapping.get("analytic_trim_certificate")
      if schema_version == LEGACY_FACE_MAP_AUDIT_SCHEMA_VERSION:
        if acceptance_basis is not None or analytic_evidence is not None:
          raise FaceMapAuditError(
              "legacy face-map receipt carries v2.6 acceptance semantics"
          )
      elif acceptance_basis not in {
          DISTANCE_ACCEPTANCE_BASIS,
          ANALYTIC_TRIM_ACCEPTANCE_BASIS,
          PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS,
      }:
        raise FaceMapAuditError("v2.6 face-map acceptance basis is invalid")
      if mapping.get("status") != "mapped_unique":
        if occ_indices or signatures:
          raise FaceMapAuditError("failed endpoint unexpectedly carries OCC identity")
        if (
            schema_version == FACE_MAP_AUDIT_SCHEMA_VERSION
            and (
                acceptance_basis != DISTANCE_ACCEPTANCE_BASIS
                or analytic_evidence is not None
            )
        ):
          raise FaceMapAuditError(
              "failed v2.6 endpoint carries certificate acceptance"
          )
        continue
      if (
          not occ_indices
          or any(
              not isinstance(index, int) or isinstance(index, bool) or index < 0
              for index in occ_indices
          )
          or occ_indices != sorted(set(occ_indices))
          or len(signatures) != len(occ_indices)
          or any(
              not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
              for value in signatures
          )
      ):
        raise FaceMapAuditError("mapped endpoint OCC identity is malformed")
      if schema_version == FACE_MAP_AUDIT_SCHEMA_VERSION:
        if acceptance_basis in {
            ANALYTIC_TRIM_ACCEPTANCE_BASIS,
            PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS,
        }:
          _validate_analytic_trim_evidence_structure(
              mapping,
              analytic_evidence,
              fusion_index=fusion_index,
              body_uuid=body_uuid,
              occ_indices=occ_indices,
          )
        elif analytic_evidence is not None:
          raise FaceMapAuditError(
              "distance-accepted endpoint carries analytic certificate"
          )
      lookup[key] = {"case_id": case_id, **dict(mapping)}
  return lookup


def verify_face_map_receipt_analytic_trim_full_replay(
    receipt: Mapping[str, Any],
    *,
    inputs_by_endpoint: Mapping[
        tuple[str, str, int], AnalyticTrimReplayInputs
    ],
) -> dict[tuple[str, str, int], dict[str, Any]]:
  """Validate a receipt, then byte-replay every v2.6 analytic acceptance."""

  lookup = build_mapped_endpoint_lookup(receipt)
  analytic_keys = {
      key
      for key, mapping in lookup.items()
      if mapping.get("acceptance_basis")
      in {
          ANALYTIC_TRIM_ACCEPTANCE_BASIS,
          PLANAR_CHORDAL_TRIM_ACCEPTANCE_BASIS,
      }
  }
  if set(inputs_by_endpoint) != analytic_keys:
    raise FaceMapAuditError(
        "analytic trim full-replay input endpoint domain differs"
    )
  for key in sorted(analytic_keys):
    verify_analytic_trim_mapping_full_replay(
        lookup[key],
        inputs=inputs_by_endpoint[key],
    )
  return lookup


__all__ = [
    "ANALYTIC_TRIM_ACCEPTANCE_BASIS",
    "ANALYTIC_TRIM_ALGORITHM_REVISION",
    "AnalyticTrimReplayInputs",
    "FACE_MAP_AUDIT_SCHEMA_VERSION",
    "LEGACY_FACE_MAP_AUDIT_SCHEMA_VERSION",
    "MAPPER_VERSION",
    "ENDPOINT_LOOKUP_SCHEMA_VERSION",
    "FaceMapAuditError",
    "MappingTolerances",
    "audit_family_source",
    "build_mapped_endpoint_lookup",
    "verify_analytic_trim_mapping_full_replay",
    "verify_face_map_receipt_analytic_trim_full_replay",
    "select_development_case_ids",
]
