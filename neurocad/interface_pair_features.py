"""Pair-level features for learned interface compatibility.

The single-interface scorer answers whether one face is likely useful. This
module answers the next question: whether two independently extracted local
interfaces form a physically plausible mate. Features are local/interface-only
and do not encode ground-truth relative poses.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .math3d import normalize
from .domain_types import Socket
from .benchmark_v2_model_view import validate_benchmark_v2_model_view


ROLES = (
    "center_bore",
    "threaded_hole",
    "obround_slot",
    "pin_boss",
    "hole_entry",
    "shaft_axis",
    "cylindrical_interface",
    "planar_seat",
    "shoulder_stop",
    "generic_interface",
)

SURFACES = ("plane", "cylinder", "cone", "sphere", "torus", "spline", "other")

PAIR_FEATURE_NAMES = [
    "bias",
    "relation_coaxial",
    "relation_planar",
    "relation_insert",
    "relation_support",
    "relation_link",
    *[f"role_a_{role}" for role in ROLES],
    *[f"role_b_{role}" for role in ROLES],
    *[f"surface_a_{surface}" for surface in SURFACES],
    *[f"surface_b_{surface}" for surface in SURFACES],
    "same_role",
    "same_surface",
    "pair_hole_shaft",
    "pair_center_bore_pin",
    "pair_threaded_fastener",
    "pair_pin_slot",
    "pair_boss_slot",
    "pair_planar_planar",
    "pair_cylindrical_cylindrical",
    "illegal_planar_cylindrical",
    "illegal_shaft_shaft",
    "illegal_hole_hole",
    "score_a",
    "score_b",
    "score_min",
    "score_product",
    "radius_present_both",
    "radius_abs_diff",
    "radius_rel_diff",
    "radius_fit_hole_minus_pin",
    "radius_fit_positive",
    "radius_fit_negative",
    "radius_fit_too_loose",
    "axis_dot_signed",
    "axis_dot_abs",
    "normal_dot_signed",
    "normal_dot_abs",
    "z_dot_signed",
    "z_dot_abs",
    "facing_normals",
    "parallel_axes",
    "area_log_abs_diff",
    "area_min_over_max",
    "edge_count_log_abs_diff",
    "concavity_product",
    "concavity_opposed",
    "boundary_both",
    "boundary_either",
    "composite_a",
    "composite_b",
    "composite_either",
    "heuristic_prior",
]

BENCHMARK_V2_PAIR_FEATURE_NAMES = (
    "bias",
    "relation_coaxial",
    "relation_planar",
    "relation_insert",
    "relation_support",
    "relation_link",
    *[f"role_a_{role}" for role in ROLES],
    *[f"role_b_{role}" for role in ROLES],
    *[f"surface_a_{surface}" for surface in SURFACES],
    *[f"surface_b_{surface}" for surface in SURFACES],
    "same_role",
    "same_surface",
    "pair_hole_shaft",
    "pair_center_bore_pin",
    "pair_threaded_fastener",
    "pair_pin_slot",
    "pair_boss_slot",
    "pair_planar_planar",
    "pair_cylindrical_cylindrical",
    "illegal_planar_cylindrical",
    "illegal_shaft_shaft",
    "illegal_hole_hole",
    "score_a",
    "score_b",
    "score_min",
    "score_product",
    "radius_ratio_present_both",
    "radius_ratio_abs_diff",
    "radius_ratio_rel_diff",
    "radius_ratio_min_over_max",
    "area_ratio_log_abs_diff",
    "area_ratio_min_over_max",
    "composite_a",
    "composite_b",
    "composite_either",
    "heuristic_prior",
)


def pair_feature_vector(
    features: dict[str, float],
    *,
    protocol: str = "legacy",
) -> list[float]:
  names = (
      BENCHMARK_V2_PAIR_FEATURE_NAMES
      if _normalized_pair_protocol(protocol) == "benchmark_v2"
      else PAIR_FEATURE_NAMES
  )
  return [float(features.get(name, 0.0)) for name in names]


def pair_features_from_rows(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    *,
    relation_hint: str = "link",
    protocol: str = "legacy",
) -> dict[str, float]:
  """Build compatibility features from two candidate-interface JSON rows."""

  protocol = _normalized_pair_protocol(protocol)
  if protocol == "benchmark_v2":
    return _benchmark_v2_pair_features(
        view_a=_benchmark_view_from_mapping(row_a),
        view_b=_benchmark_view_from_mapping(row_b),
        score_a=_score(row_a),
        score_b=_score(row_b),
        relation_hint=relation_hint,
    )
  role_a = _bucket_role(row_a.get("role_hint"))
  role_b = _bucket_role(row_b.get("role_hint"))
  surface_a = _bucket_surface(row_a.get("surface_type"))
  surface_b = _bucket_surface(row_b.get("surface_type"))
  meta_a = _metadata(row_a)
  meta_b = _metadata(row_b)
  features_a = _features(row_a)
  features_b = _features(row_b)
  frame_a = _frame(row_a)
  frame_b = _frame(row_b)
  return _pair_features(
      role_a=role_a,
      role_b=role_b,
      surface_a=surface_a,
      surface_b=surface_b,
      score_a=_score(row_a),
      score_b=_score(row_b),
      radius_a=_float_or_none(meta_a.get("radius")),
      radius_b=_float_or_none(meta_b.get("radius")),
      area_a=_float_or_zero(meta_a.get("area")),
      area_b=_float_or_zero(meta_b.get("area")),
      edge_a=_float_or_zero(meta_a.get("edge_count")),
      edge_b=_float_or_zero(meta_b.get("edge_count")),
      concavity_a=_float_or_zero(meta_a.get("concavity")),
      concavity_b=_float_or_zero(meta_b.get("concavity")),
      boundary_a=_float_or_zero(features_a.get("near_bbox_boundary")),
      boundary_b=_float_or_zero(features_b.get("near_bbox_boundary")),
      composite_a=float(bool(meta_a.get("composite_interface", False))),
      composite_b=float(bool(meta_b.get("composite_interface", False))),
      axis_a=_axis(frame_a),
      axis_b=_axis(frame_b),
      normal_a=_normal(frame_a),
      normal_b=_normal(frame_b),
      z_a=_z_axis(frame_a),
      z_b=_z_axis(frame_b),
      relation_hint=relation_hint,
  )


def pair_features_from_sockets(
    socket_a: Socket,
    socket_b: Socket,
    *,
    relation_hint: str = "link",
    protocol: str = "legacy",
) -> dict[str, float]:
  """Build compatibility features from runtime learned-interface sockets."""

  protocol = _normalized_pair_protocol(protocol)
  meta_a = socket_a.metadata if isinstance(socket_a.metadata, dict) else {}
  meta_b = socket_b.metadata if isinstance(socket_b.metadata, dict) else {}
  if protocol == "benchmark_v2":
    return _benchmark_v2_pair_features(
        view_a=_benchmark_view_from_mapping(meta_a),
        view_b=_benchmark_view_from_mapping(meta_b),
        score_a=_socket_score(socket_a),
        score_b=_socket_score(socket_b),
        relation_hint=relation_hint,
    )
  features_a = meta_a.get("interface_features")
  features_b = meta_b.get("interface_features")
  if not isinstance(features_a, dict):
    features_a = {}
  if not isinstance(features_b, dict):
    features_b = {}
  return _pair_features(
      role_a=_bucket_role(meta_a.get("interface_role")),
      role_b=_bucket_role(meta_b.get("interface_role")),
      surface_a=_bucket_surface(meta_a.get("surface_type")),
      surface_b=_bucket_surface(meta_b.get("surface_type")),
      score_a=_socket_score(socket_a),
      score_b=_socket_score(socket_b),
      radius_a=socket_a.radius,
      radius_b=socket_b.radius,
      area_a=_float_or_zero(meta_a.get("area")),
      area_b=_float_or_zero(meta_b.get("area")),
      edge_a=_float_or_zero(meta_a.get("edge_count")),
      edge_b=_float_or_zero(meta_b.get("edge_count")),
      concavity_a=_float_or_zero(meta_a.get("concavity")),
      concavity_b=_float_or_zero(meta_b.get("concavity")),
      boundary_a=_float_or_zero(features_a.get("near_bbox_boundary")),
      boundary_b=_float_or_zero(features_b.get("near_bbox_boundary")),
      composite_a=float(bool(meta_a.get("composite_interface", False))),
      composite_b=float(bool(meta_b.get("composite_interface", False))),
      axis_a=None if socket_a.axis is None else np.asarray(socket_a.axis, dtype=float),
      axis_b=None if socket_b.axis is None else np.asarray(socket_b.axis, dtype=float),
      normal_a=None if socket_a.normal is None else np.asarray(socket_a.normal, dtype=float),
      normal_b=None if socket_b.normal is None else np.asarray(socket_b.normal, dtype=float),
      z_a=None if socket_a.z_axis is None else np.asarray(socket_a.z_axis, dtype=float),
      z_b=None if socket_b.z_axis is None else np.asarray(socket_b.z_axis, dtype=float),
      relation_hint=relation_hint,
  )


def _normalized_pair_protocol(protocol: str) -> str:
  value = str(protocol or "legacy").strip().lower()
  if value not in {"legacy", "benchmark_v2"}:
    raise ValueError("pair feature protocol must be 'legacy' or 'benchmark_v2'")
  return value


def _benchmark_view_from_mapping(container: dict[str, Any]) -> dict[str, Any]:
  if str(container.get("model_input_protocol") or "").strip().lower() != "benchmark_v2":
    raise ValueError("benchmark_v2 pair input requires benchmark_v2 socket metadata")
  view = container.get("benchmark_v2_model_view")
  if not isinstance(view, dict):
    raise ValueError("benchmark_v2 pair input is missing sanitized model view")
  expected_hash = container.get("benchmark_v2_model_view_sha256")
  if not isinstance(expected_hash, str) or not expected_hash:
    raise ValueError("benchmark_v2 pair input is missing model view hash")
  validate_benchmark_v2_model_view(view, expected_sha256=expected_hash)
  return view


def _benchmark_role(view: dict[str, Any]) -> str:
  topology = view.get("topology", {})
  role = str(topology.get("composite_role") or "").strip().lower()
  if role:
    return _bucket_role(role)
  surface = _bucket_surface(view.get("surface_type"))
  if surface == "plane":
    return "planar_seat"
  if surface == "cylinder":
    return "cylindrical_interface"
  if surface in {"cone", "torus"}:
    return "shoulder_stop"
  return "generic_interface"


def _benchmark_v2_pair_features(
    *,
    view_a: dict[str, Any],
    view_b: dict[str, Any],
    score_a: float,
    score_b: float,
    relation_hint: str,
) -> dict[str, float]:
  """Pair descriptor with no cross-world coordinates or directions."""

  result = {name: 0.0 for name in BENCHMARK_V2_PAIR_FEATURE_NAMES}
  result["bias"] = 1.0
  relation = _bucket_relation(relation_hint)
  result[f"relation_{relation}"] = 1.0
  role_a = _benchmark_role(view_a)
  role_b = _benchmark_role(view_b)
  surface_a = _bucket_surface(view_a.get("surface_type"))
  surface_b = _bucket_surface(view_b.get("surface_type"))
  result[f"role_a_{role_a}"] = 1.0
  result[f"role_b_{role_b}"] = 1.0
  result[f"surface_a_{surface_a}"] = 1.0
  result[f"surface_b_{surface_b}"] = 1.0

  roles = {role_a, role_b}
  hole_roles = {"hole_entry", "center_bore", "threaded_hole"}
  pin_roles = {"shaft_axis", "cylindrical_interface", "pin_boss"}
  slot_roles = {"obround_slot"}
  cylindrical_roles = hole_roles | pin_roles | slot_roles
  planar_roles = {"planar_seat", "shoulder_stop"}
  result["same_role"] = float(role_a == role_b)
  result["same_surface"] = float(surface_a == surface_b)
  result["pair_hole_shaft"] = float(
      bool(roles & hole_roles) and bool(roles & pin_roles)
  )
  result["pair_center_bore_pin"] = float(
      "center_bore" in roles and bool(roles & pin_roles)
  )
  result["pair_threaded_fastener"] = float(
      "threaded_hole" in roles and bool(roles & pin_roles)
  )
  result["pair_pin_slot"] = float("obround_slot" in roles and "shaft_axis" in roles)
  result["pair_boss_slot"] = float("obround_slot" in roles and "pin_boss" in roles)
  result["pair_planar_planar"] = float(
      role_a in planar_roles and role_b in planar_roles
  )
  result["pair_cylindrical_cylindrical"] = float(
      role_a in cylindrical_roles and role_b in cylindrical_roles
  )
  result["illegal_planar_cylindrical"] = float(
      (role_a in planar_roles and role_b in cylindrical_roles)
      or (role_b in planar_roles and role_a in cylindrical_roles)
  )
  result["illegal_shaft_shaft"] = float(role_a == role_b == "shaft_axis")
  result["illegal_hole_hole"] = float(role_a == role_b and role_a in hole_roles)

  score_a = _finite_float(score_a, name="score_a")
  score_b = _finite_float(score_b, name="score_b")
  result["score_a"] = score_a
  result["score_b"] = score_b
  result["score_min"] = min(score_a, score_b)
  result["score_product"] = score_a * score_b

  geometry_a = view_a.get("geometry", {})
  geometry_b = view_b.get("geometry", {})
  radius_a = _float_or_none(geometry_a.get("radius_ratio"))
  radius_b = _float_or_none(geometry_b.get("radius_ratio"))
  if radius_a is not None and radius_b is not None:
    radius_a = abs(radius_a)
    radius_b = abs(radius_b)
    maximum = max(radius_a, radius_b, 1e-12)
    difference = abs(radius_a - radius_b)
    result["radius_ratio_present_both"] = 1.0
    result["radius_ratio_abs_diff"] = difference
    result["radius_ratio_rel_diff"] = difference / maximum
    result["radius_ratio_min_over_max"] = min(radius_a, radius_b) / maximum
  area_a = _float_or_zero(geometry_a.get("area_ratio"))
  area_b = _float_or_zero(geometry_b.get("area_ratio"))
  result["area_ratio_log_abs_diff"] = abs(
      math.log1p(max(0.0, area_a)) - math.log1p(max(0.0, area_b))
  )
  result["area_ratio_min_over_max"] = min(area_a, area_b) / max(
      1e-12, area_a, area_b
  )
  topology_a = view_a.get("topology", {})
  topology_b = view_b.get("topology", {})
  composite_a = float(bool(topology_a.get("composite_interface", False)))
  composite_b = float(bool(topology_b.get("composite_interface", False)))
  result["composite_a"] = composite_a
  result["composite_b"] = composite_b
  result["composite_either"] = float(bool(composite_a or composite_b))
  result["heuristic_prior"] = heuristic_pair_prior(result)
  return result


def _finite_float(value: Any, *, name: str) -> float:
  number = float(value)
  if not math.isfinite(number):
    raise ValueError(f"benchmark_v2 pair feature {name} must be finite")
  return number


def heuristic_pair_prior(features: dict[str, float]) -> float:
  """A deterministic prior used for hard-negative mining and fallback sorting."""

  score = 0.0
  score += 2.5 * float(features.get("pair_hole_shaft", 0.0))
  score += 3.2 * float(features.get("pair_center_bore_pin", 0.0))
  score += 3.0 * float(features.get("pair_pin_slot", 0.0))
  score += 3.0 * float(features.get("pair_boss_slot", 0.0))
  score += 2.7 * float(features.get("pair_threaded_fastener", 0.0))
  score += 1.6 * float(features.get("pair_planar_planar", 0.0))
  score += 0.8 * float(features.get("pair_cylindrical_cylindrical", 0.0))
  score += 1.2 * float(features.get("radius_fit_positive", 0.0))
  score -= 1.8 * float(features.get("radius_fit_negative", 0.0))
  score -= 0.6 * float(features.get("radius_fit_too_loose", 0.0))
  score += 0.9 * float(features.get("facing_normals", 0.0))
  score += 0.8 * float(features.get("parallel_axes", 0.0))
  score -= 2.5 * float(features.get("illegal_planar_cylindrical", 0.0))
  score -= 1.6 * float(features.get("illegal_shaft_shaft", 0.0))
  score -= 1.4 * float(features.get("illegal_hole_hole", 0.0))
  score += 0.75 * float(features.get("composite_either", 0.0))
  score += 0.5 * float(features.get("score_min", 0.0))
  return float(score)


def _pair_features(
    *,
    role_a: str,
    role_b: str,
    surface_a: str,
    surface_b: str,
    score_a: float,
    score_b: float,
    radius_a: float | None,
    radius_b: float | None,
    area_a: float,
    area_b: float,
    edge_a: float,
    edge_b: float,
    concavity_a: float,
    concavity_b: float,
    boundary_a: float,
    boundary_b: float,
    composite_a: float,
    composite_b: float,
    axis_a: np.ndarray | None,
    axis_b: np.ndarray | None,
    normal_a: np.ndarray | None,
    normal_b: np.ndarray | None,
    z_a: np.ndarray | None,
    z_b: np.ndarray | None,
    relation_hint: str,
) -> dict[str, float]:
  result = {name: 0.0 for name in PAIR_FEATURE_NAMES}
  result["bias"] = 1.0
  relation = _bucket_relation(relation_hint)
  result[f"relation_{relation}"] = 1.0
  result[f"role_a_{role_a}"] = 1.0
  result[f"role_b_{role_b}"] = 1.0
  result[f"surface_a_{surface_a}"] = 1.0
  result[f"surface_b_{surface_b}"] = 1.0

  roles = {role_a, role_b}
  hole_roles = {"hole_entry", "center_bore", "threaded_hole"}
  pin_roles = {"shaft_axis", "cylindrical_interface", "pin_boss"}
  slot_roles = {"obround_slot"}
  cylindrical_roles = hole_roles | pin_roles | slot_roles
  planar_roles = {"planar_seat", "shoulder_stop"}
  role_a_cyl = role_a in cylindrical_roles
  role_b_cyl = role_b in cylindrical_roles
  role_a_planar = role_a in planar_roles
  role_b_planar = role_b in planar_roles
  result["same_role"] = float(role_a == role_b)
  result["same_surface"] = float(surface_a == surface_b)
  result["pair_hole_shaft"] = float(
      bool(roles & hole_roles) and bool(roles & pin_roles)
  )
  result["pair_center_bore_pin"] = float(
      "center_bore" in roles and bool(roles & pin_roles)
  )
  result["pair_threaded_fastener"] = float(
      "threaded_hole" in roles and bool(roles & pin_roles)
  )
  result["pair_pin_slot"] = float("obround_slot" in roles and "shaft_axis" in roles)
  result["pair_boss_slot"] = float("obround_slot" in roles and "pin_boss" in roles)
  result["pair_planar_planar"] = float(role_a_planar and role_b_planar)
  result["pair_cylindrical_cylindrical"] = float(role_a_cyl and role_b_cyl)
  result["illegal_planar_cylindrical"] = float(
      (role_a_planar and role_b_cyl) or (role_b_planar and role_a_cyl)
  )
  result["illegal_shaft_shaft"] = float(role_a == role_b == "shaft_axis")
  result["illegal_hole_hole"] = float(role_a == role_b and role_a in hole_roles)

  result["score_a"] = float(score_a)
  result["score_b"] = float(score_b)
  result["score_min"] = min(float(score_a), float(score_b))
  result["score_product"] = float(score_a) * float(score_b)

  if radius_a is not None and radius_b is not None:
    ra = abs(float(radius_a))
    rb = abs(float(radius_b))
    diff = abs(ra - rb)
    max_r = max(ra, rb, 1e-6)
    result["radius_present_both"] = 1.0
    result["radius_abs_diff"] = float(diff)
    result["radius_rel_diff"] = float(diff / max_r)
    fit = 0.0
    if role_a in hole_roles and role_b in pin_roles:
      fit = ra - rb
    elif role_b in hole_roles and role_a in pin_roles:
      fit = rb - ra
    result["radius_fit_hole_minus_pin"] = float(fit)
    result["radius_fit_positive"] = float(fit >= -0.05 and result["pair_hole_shaft"] > 0.0)
    result["radius_fit_negative"] = float(fit < -0.05 and result["pair_hole_shaft"] > 0.0)
    result["radius_fit_too_loose"] = float(fit > max(2.0, 0.35 * max_r) and result["pair_hole_shaft"] > 0.0)

  result["axis_dot_signed"] = _dot(axis_a, axis_b)
  result["axis_dot_abs"] = abs(result["axis_dot_signed"])
  result["normal_dot_signed"] = _dot(normal_a, normal_b)
  result["normal_dot_abs"] = abs(result["normal_dot_signed"])
  result["z_dot_signed"] = _dot(z_a, z_b)
  result["z_dot_abs"] = abs(result["z_dot_signed"])
  result["facing_normals"] = float(result["normal_dot_signed"] < -0.75)
  result["parallel_axes"] = float(
      max(result["axis_dot_abs"], result["z_dot_abs"]) > 0.82
  )

  result["area_log_abs_diff"] = abs(math.log1p(max(0.0, area_a)) - math.log1p(max(0.0, area_b)))
  result["area_min_over_max"] = min(max(0.0, area_a), max(0.0, area_b)) / max(
      1e-9,
      max(max(0.0, area_a), max(0.0, area_b)),
  )
  result["edge_count_log_abs_diff"] = abs(math.log1p(max(0.0, edge_a)) - math.log1p(max(0.0, edge_b)))
  result["concavity_product"] = float(concavity_a) * float(concavity_b)
  result["concavity_opposed"] = float(float(concavity_a) * float(concavity_b) < 0.0)
  result["boundary_both"] = float(boundary_a > 0.5 and boundary_b > 0.5)
  result["boundary_either"] = float(boundary_a > 0.5 or boundary_b > 0.5)
  result["composite_a"] = float(composite_a)
  result["composite_b"] = float(composite_b)
  result["composite_either"] = float(composite_a > 0.5 or composite_b > 0.5)
  result["heuristic_prior"] = heuristic_pair_prior(result)
  return result


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
  value = row.get("metadata")
  return value if isinstance(value, dict) else {}


def _features(row: dict[str, Any]) -> dict[str, Any]:
  value = row.get("features")
  return value if isinstance(value, dict) else {}


def _frame(row: dict[str, Any]) -> dict[str, Any]:
  value = row.get("local_frame")
  return value if isinstance(value, dict) else {}


def _score(row: dict[str, Any]) -> float:
  raw = row.get("score")
  if isinstance(raw, (int, float)):
    return float(raw)
  return float(row.get("label", 0) or 0) * 0.5


def _socket_score(socket: Socket) -> float:
  raw = socket.metadata.get("interface_score", socket.metadata.get("quality", 0.0))
  return float(raw) if isinstance(raw, (int, float)) else 0.0


def _bucket_role(value: Any) -> str:
  text = str(value or "").strip().lower()
  return text if text in ROLES else "generic_interface"


def _bucket_surface(value: Any) -> str:
  text = str(value or "").strip().lower()
  for surface in SURFACES:
    if surface in text:
      return surface
  return "other"


def _bucket_relation(value: Any) -> str:
  text = str(value or "").strip().lower()
  if text in {"coaxial", "fasten"}:
    return "coaxial"
  if text in {"planar", "coincident"}:
    return "planar"
  if text in {
      "insert",
      "interface_insert",
      "shaft_in_bore",
      "screw_in_hole",
      "pin_in_slot",
      "boss_in_slot",
  }:
    return "insert"
  if text in {"support", "seat", "through_stop"}:
    return "support"
  return "link"


def _axis(frame: dict[str, Any]) -> np.ndarray | None:
  for key in ("axis", "z_axis", "normal"):
    value = _vector_or_none(frame.get(key))
    if value is not None:
      return value
  return None


def _normal(frame: dict[str, Any]) -> np.ndarray | None:
  for key in ("normal", "z_axis", "axis"):
    value = _vector_or_none(frame.get(key))
    if value is not None:
      return value
  return None


def _z_axis(frame: dict[str, Any]) -> np.ndarray | None:
  for key in ("z_axis", "axis", "normal"):
    value = _vector_or_none(frame.get(key))
    if value is not None:
      return value
  return None


def _vector_or_none(value: Any) -> np.ndarray | None:
  if not isinstance(value, (list, tuple, np.ndarray)):
    return None
  try:
    return normalize(np.asarray(value, dtype=float).reshape(3))
  except Exception:
    return None


def _dot(a: np.ndarray | None, b: np.ndarray | None) -> float:
  if a is None or b is None:
    return 0.0
  try:
    return float(np.dot(normalize(a), normalize(b)))
  except Exception:
    return 0.0


def _float_or_none(value: Any) -> float | None:
  if isinstance(value, (int, float)):
    return float(value)
  return None


def _float_or_zero(value: Any) -> float:
  if isinstance(value, (int, float)):
    return float(value)
  return 0.0
