"""Leakage-safe model views for the pose-scrambled benchmark-v2 protocol.

This module is intentionally separate from legacy feature extraction.  It
accepts an already extracted candidate dictionary and publishes only an
explicit allowlist of rigid-invariant geometry and local interface topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = 1

_SAFE_FEATURE_KEYS = frozenset(
    {
        "bias",
        "surface_plane",
        "surface_cylinder",
        "surface_cone",
        "surface_sphere",
        "surface_torus",
        "surface_spline",
        "surface_other",
        "log_area_ratio",
        "sqrt_area_ratio",
        "has_helical_edge",
    }
)

_FORBIDDEN_FEATURE_REASONS = {
    "center_x_norm": "absolute_or_axis_aligned_center",
    "center_y_norm": "absolute_or_axis_aligned_center",
    "center_z_norm": "absolute_or_axis_aligned_center",
    "bbox_x_ratio": "axis_aligned_world_bbox",
    "bbox_y_ratio": "axis_aligned_world_bbox",
    "bbox_z_ratio": "axis_aligned_world_bbox",
    "normal_abs_x": "absolute_world_direction",
    "normal_abs_y": "absolute_world_direction",
    "normal_abs_z": "absolute_world_direction",
    "axis_abs_x": "absolute_world_direction",
    "axis_abs_y": "absolute_world_direction",
    "axis_abs_z": "absolute_world_direction",
    "near_bbox_boundary": "axis_aligned_world_bbox",
    "radius_ratio": "legacy_axis_aligned_bbox_scale",
    "concavity": "legacy_concavity_not_part_of_formal_model_view",
    "edge_count_log": "occ_seam_segmentation_not_gauge_stable",
}

_FORBIDDEN_TOP_LEVEL_REASONS = {
    "interface_id": "identifier",
    "role_hint": "legacy_pose_sensitive_derived_role",
    "part_name": "identifier",
    "body_uuid": "identifier",
    "step_path": "filesystem_path",
    "assembly_id": "identifier",
    "assembly_dir": "identifier",
    "case_id": "identifier",
    "program_id": "identifier",
    "face_index": "file_order_index",
    "file_order_index": "file_order_index",
    "local_frame": "absolute_frame",
    "world_frame": "absolute_frame",
    "origin": "absolute_world_coordinate",
    "center": "absolute_world_coordinate",
    "axis": "absolute_world_direction",
    "normal": "absolute_world_direction",
    "bbox_min": "axis_aligned_world_bbox",
    "bbox_max": "axis_aligned_world_bbox",
    "aabb_min": "axis_aligned_world_bbox",
    "aabb_max": "axis_aligned_world_bbox",
    "source_transform": "source_transform",
    "transform": "source_transform",
    "score": "legacy_score_may_encode_forbidden_features",
    "label": "supervision_only",
    "label_source": "supervision_only",
}

_SAFE_METADATA_KEYS = frozenset(
    {
        "area",
        "radius",
        "has_helical_edge",
        "composite_interface",
        "composite_role",
        "compatible_relations",
        "slot_width",
        "slot_length",
        "slot_center_distance",
    }
)

_FORBIDDEN_METADATA_REASONS = {
    "source": "source_provenance_not_model_input",
    "member_face_indices": "file_order_index",
    "source_transform": "source_transform",
    "assembly_id": "identifier",
    "part_name": "identifier",
    "body_uuid": "identifier",
    "step_path": "filesystem_path",
    "concavity": "legacy_concavity_not_part_of_formal_model_view",
    "centeredness": "legacy_axis_aligned_bbox_derived_role_evidence",
    "boundary_score": "legacy_axis_aligned_bbox_derived_role_evidence",
    "radius_ratio": "legacy_axis_aligned_bbox_scale",
    "edge_count": "occ_seam_segmentation_not_gauge_stable",
}


BENCHMARK_V2_FEATURE_NAMES = (
    "bias",
    "surface_plane",
    "surface_cylinder",
    "surface_cone",
    "surface_sphere",
    "surface_torus",
    "surface_spline",
    "surface_other",
    "log_area_ratio",
    "sqrt_area_ratio",
    "has_helical_edge",
    "area_ratio",
    "radius_ratio",
    "slot_width_ratio",
    "slot_length_ratio",
    "slot_center_distance_ratio",
    "composite_interface",
    "role_center_bore",
    "role_threaded_hole",
    "role_obround_slot",
    "role_pin_boss",
    "role_shaft_axis",
    "role_planar_seat",
    "role_shoulder_stop",
)

_MODEL_VIEW_GEOMETRY_KEYS = frozenset(
    {
        "area_ratio",
        "radius_ratio",
        "slot_width_ratio",
        "slot_length_ratio",
        "slot_center_distance_ratio",
    }
)
_MODEL_VIEW_TOPOLOGY_KEYS = frozenset(
    {
        "has_helical_edge",
        "composite_interface",
        "composite_role",
        "compatible_relations",
    }
)
_MODEL_VIEW_SURFACES = frozenset(
    {"plane", "cylinder", "cone", "sphere", "torus", "spline", "other"}
)
_MODEL_VIEW_COMPOSITE_ROLES = frozenset(
    {
        "center_bore",
        "threaded_hole",
        "obround_slot",
        "pin_boss",
        "shaft_axis",
        "planar_seat",
        "shoulder_stop",
    }
)
_MODEL_VIEW_RELATIONS = frozenset(
    {
        "shaft_in_bore",
        "insert_axis",
        "screw_in_hole",
        "threaded_interference",
        "pin_in_slot",
        "boss_in_slot",
        "planar_seat",
        "seat_plane",
        "generic_contact",
    }
)


@dataclass(frozen=True)
class ModelViewSanitization:
  """Sanitized benchmark-v2 input plus deterministic leakage evidence."""

  model_view: dict[str, Any]
  audit: dict[str, Any]
  canonical_json: str
  sha256: str

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_view": self.model_view,
        "audit": self.audit,
        "canonical_json": self.canonical_json,
        "sha256": self.sha256,
    }


class ModelViewLeakageError(ValueError):
  """Formal benchmark-v2 input contained a key outside the allowlist."""

  def __init__(self, audit: dict[str, Any]):
    unknown_paths = [str(row.get("path")) for row in audit.get("unknown", [])]
    super().__init__(
        "Unknown benchmark-v2 model-visible keys: " + ", ".join(unknown_paths)
    )
    self.audit = audit


def _canonical_float(value: Any, *, path: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"Benchmark-v2 numeric feature is not numeric: {path}")
  number = float(value)
  if not math.isfinite(number):
    raise ValueError(f"Benchmark-v2 numeric feature is not finite: {path}")
  number = round(number, 12)
  return 0.0 if number == 0.0 else number


def _positive_scale(value: Any, *, name: str) -> float:
  scale = _canonical_float(value, path=name)
  if scale <= 0.0:
    raise ValueError(f"{name} must be finite and positive")
  return scale


def _audit_row(path: str, reason: str) -> dict[str, str]:
  return {"path": path, "reason": reason}


def sanitize_benchmark_v2_model_view(
    candidate: Mapping[str, Any],
    *,
    invariant_part_scale: float,
    part_surface_area: float,
    formal: bool = True,
) -> ModelViewSanitization:
  """Build a deterministic, rigid-invariant model view for benchmark-v2."""

  if not isinstance(candidate, Mapping):
    raise TypeError("candidate must be a mapping")
  length_scale = _positive_scale(
      invariant_part_scale,
      name="invariant_part_scale",
  )
  area_scale = _positive_scale(part_surface_area, name="part_surface_area")
  forbidden: list[dict[str, str]] = []
  unknown: list[dict[str, str]] = []
  for key, reason in _FORBIDDEN_TOP_LEVEL_REASONS.items():
    if key in candidate:
      forbidden.append(_audit_row(key, reason))
  known_top_level = {
      "surface_type",
      "features",
      "metadata",
      *_FORBIDDEN_TOP_LEVEL_REASONS,
  }
  for key in candidate:
    text_key = str(key)
    if text_key not in known_top_level:
      unknown.append(_audit_row(text_key, "unknown_not_allowlisted"))

  raw_features = candidate.get("features")
  if raw_features is None:
    raw_features = {}
  if not isinstance(raw_features, Mapping):
    raise ValueError("candidate.features must be a mapping")
  features: dict[str, float] = {}
  for key in sorted(raw_features):
    text_key = str(key)
    if text_key in _SAFE_FEATURE_KEYS:
      features[text_key] = _canonical_float(
          raw_features[key],
          path=f"features.{text_key}",
      )
    elif text_key in _FORBIDDEN_FEATURE_REASONS:
      forbidden.append(
          _audit_row(
              f"features.{text_key}",
              _FORBIDDEN_FEATURE_REASONS[text_key],
          )
      )
    else:
      unknown.append(
          _audit_row(f"features.{text_key}", "unknown_not_allowlisted")
      )

  raw_metadata = candidate.get("metadata")
  if raw_metadata is None:
    raw_metadata = {}
  if not isinstance(raw_metadata, Mapping):
    raise ValueError("candidate.metadata must be a mapping")
  for key, reason in _FORBIDDEN_METADATA_REASONS.items():
    if key in raw_metadata:
      forbidden.append(_audit_row(f"metadata.{key}", reason))
  known_metadata = _SAFE_METADATA_KEYS | frozenset(_FORBIDDEN_METADATA_REASONS)
  for key in raw_metadata:
    text_key = str(key)
    if text_key not in known_metadata:
      unknown.append(
          _audit_row(f"metadata.{text_key}", "unknown_not_allowlisted")
      )

  geometry: dict[str, float] = {}
  normalization: list[dict[str, str]] = []
  if raw_metadata.get("area") is not None:
    geometry["area_ratio"] = _canonical_float(
        _canonical_float(raw_metadata["area"], path="metadata.area")
        / area_scale,
        path="geometry.area_ratio",
    )
    normalization.append(
        {
            "source_path": "metadata.area",
            "output_path": "geometry.area_ratio",
            "divisor": "part_surface_area",
        }
    )
  for source_key, output_key in (
      ("radius", "radius_ratio"),
      ("slot_width", "slot_width_ratio"),
      ("slot_length", "slot_length_ratio"),
      ("slot_center_distance", "slot_center_distance_ratio"),
  ):
    if raw_metadata.get(source_key) is None:
      continue
    geometry[output_key] = _canonical_float(
        _canonical_float(
            raw_metadata[source_key],
            path=f"metadata.{source_key}",
        )
        / length_scale,
        path=f"geometry.{output_key}",
    )
    normalization.append(
        {
            "source_path": f"metadata.{source_key}",
            "output_path": f"geometry.{output_key}",
            "divisor": "invariant_part_scale",
        }
    )

  topology: dict[str, Any] = {}
  for key in ("has_helical_edge",):
    if raw_metadata.get(key) is not None:
      topology[key] = _canonical_float(
          raw_metadata[key],
          path=f"metadata.{key}",
      )
  if raw_metadata.get("composite_interface") is not None:
    topology["composite_interface"] = bool(
        raw_metadata["composite_interface"]
    )
  if raw_metadata.get("composite_role") is not None:
    composite_role = str(raw_metadata["composite_role"]).strip().lower()
    if composite_role not in _MODEL_VIEW_COMPOSITE_ROLES:
      raise ValueError("metadata.composite_role is not benchmark-v2 allowlisted")
    topology["composite_role"] = composite_role
  if raw_metadata.get("compatible_relations") is not None:
    raw_relations = raw_metadata["compatible_relations"]
    if not isinstance(raw_relations, (list, tuple, set)):
      raise ValueError("metadata.compatible_relations must be a sequence")
    compatible_relations = {
        str(value).strip().lower()
        for value in raw_relations
        if str(value).strip()
    }
    unknown_relations = compatible_relations - _MODEL_VIEW_RELATIONS
    if unknown_relations:
      raise ValueError(
          "metadata.compatible_relations contains non-allowlisted values"
      )
    topology["compatible_relations"] = sorted(compatible_relations)

  surface_type = str(candidate.get("surface_type") or "other").strip().lower()
  if surface_type not in _MODEL_VIEW_SURFACES:
    raise ValueError("candidate.surface_type is not benchmark-v2 allowlisted")
  model_view = {
      "schema_version": SCHEMA_VERSION,
      "surface_type": surface_type,
      "features": dict(sorted(features.items())),
      "geometry": dict(sorted(geometry.items())),
      "topology": dict(sorted(topology.items())),
  }
  canonical_json = json.dumps(
      model_view,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  )
  forbidden = sorted(forbidden, key=lambda row: row["path"])
  unknown = sorted(unknown, key=lambda row: row["path"])
  audit = {
      "schema_version": SCHEMA_VERSION,
      "protocol": "benchmark_v2",
      "formal": bool(formal),
      "forbidden": forbidden,
      "dropped": list(forbidden),
      "normalization": sorted(
          normalization,
          key=lambda row: row["output_path"],
      ),
      "unknown": unknown,
  }
  if formal and unknown:
    raise ModelViewLeakageError(audit)
  return ModelViewSanitization(
      model_view=model_view,
      audit=audit,
      canonical_json=canonical_json,
      sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
  )


def benchmark_v2_numeric_feature_dict(
    sanitized: ModelViewSanitization,
) -> dict[str, float]:
  """Convert a formally sanitized view into the scorer's fixed numeric schema."""

  if not isinstance(sanitized, ModelViewSanitization):
    raise TypeError(
        "benchmark_v2 scorer input must be a ModelViewSanitization; "
        "call sanitize_benchmark_v2_model_view first"
    )
  view = sanitized.model_view
  features = view.get("features")
  geometry = view.get("geometry")
  topology = view.get("topology")
  if not isinstance(features, Mapping):
    raise ValueError("sanitized model_view.features must be a mapping")
  if not isinstance(geometry, Mapping):
    raise ValueError("sanitized model_view.geometry must be a mapping")
  if not isinstance(topology, Mapping):
    raise ValueError("sanitized model_view.topology must be a mapping")

  result = {name: 0.0 for name in BENCHMARK_V2_FEATURE_NAMES}
  for name in (
      "bias",
      "surface_plane",
      "surface_cylinder",
      "surface_cone",
      "surface_sphere",
      "surface_torus",
      "surface_spline",
      "surface_other",
      "log_area_ratio",
      "sqrt_area_ratio",
      "has_helical_edge",
  ):
    if name in features:
      result[name] = _canonical_float(
          features[name],
          path=f"model_view.features.{name}",
      )
  for name in (
      "area_ratio",
      "radius_ratio",
      "slot_width_ratio",
      "slot_length_ratio",
      "slot_center_distance_ratio",
  ):
    if name in geometry:
      result[name] = _canonical_float(
          geometry[name],
          path=f"model_view.geometry.{name}",
      )
  result["composite_interface"] = float(
      bool(topology.get("composite_interface", False))
  )
  role = str(topology.get("composite_role") or "").strip().lower()
  role_key = f"role_{role}"
  if role_key in result:
    result[role_key] = 1.0
  return result


def benchmark_v2_numeric_feature_vector(
    sanitized: ModelViewSanitization,
) -> list[float]:
  values = benchmark_v2_numeric_feature_dict(sanitized)
  return [float(values[name]) for name in BENCHMARK_V2_FEATURE_NAMES]


def validate_benchmark_v2_model_view(
    model_view: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> str:
  """Fail closed if a purported sanitized view contains any extra channel."""

  if not isinstance(model_view, Mapping):
    raise ValueError("benchmark_v2 model view must be a mapping")
  expected_top_level = {
      "schema_version",
      "surface_type",
      "features",
      "geometry",
      "topology",
  }
  if set(model_view) != expected_top_level:
    raise ValueError("benchmark_v2 model view has unknown or missing top-level keys")
  if model_view.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("benchmark_v2 model view schema_version mismatch")
  surface = str(model_view.get("surface_type") or "").strip().lower()
  if surface not in _MODEL_VIEW_SURFACES:
    raise ValueError("benchmark_v2 model view has unknown surface_type")

  features = model_view.get("features")
  geometry = model_view.get("geometry")
  topology = model_view.get("topology")
  if not isinstance(features, Mapping) or not set(features) <= _SAFE_FEATURE_KEYS:
    raise ValueError("benchmark_v2 model view has unknown feature keys")
  if not isinstance(geometry, Mapping) or not set(geometry) <= _MODEL_VIEW_GEOMETRY_KEYS:
    raise ValueError("benchmark_v2 model view has unknown geometry keys")
  if not isinstance(topology, Mapping) or not set(topology) <= _MODEL_VIEW_TOPOLOGY_KEYS:
    raise ValueError("benchmark_v2 model view has unknown topology keys")
  for key, value in features.items():
    _canonical_float(value, path=f"model_view.features.{key}")
  for key, value in geometry.items():
    _canonical_float(value, path=f"model_view.geometry.{key}")
  if "has_helical_edge" in topology:
    _canonical_float(
        topology["has_helical_edge"],
        path="model_view.topology.has_helical_edge",
    )
  if "composite_interface" in topology and not isinstance(
      topology["composite_interface"], bool
  ):
    raise ValueError("topology.composite_interface must be boolean")
  if "composite_role" in topology:
    role = str(topology["composite_role"]).strip().lower()
    if role not in _MODEL_VIEW_COMPOSITE_ROLES:
      raise ValueError("benchmark_v2 model view has unknown composite_role")
  if "compatible_relations" in topology:
    relations = topology["compatible_relations"]
    if not isinstance(relations, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in relations
    ):
      raise ValueError("topology.compatible_relations must be a string list")
    if set(relations) - _MODEL_VIEW_RELATIONS:
      raise ValueError("topology.compatible_relations has unknown values")

  canonical_json = json.dumps(
      model_view,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  )
  actual_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
  if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
    raise ValueError("benchmark_v2 model view hash mismatch")
  return actual_sha256
