"""Single-part interface candidates and learned scorer features.

This module is deliberately test-time safe: candidate interfaces are extracted
from one STEP file at a time. Assembly JSON may be used by dataset builders to
assign training labels, but pairwise relative poses are never stored here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .cadquery_backend import (
    extract_brep_faces_from_shape,
    load_step_shape,
    shape_bbox,
)
from .math3d import normalize, orthonormal_basis_from_z
from .offline import BRepFace


SURFACE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus", "spline", "other")

COMPOSITE_INTERFACE_ROLES = (
    "center_bore",
    "threaded_hole",
    "obround_slot",
    "pin_boss",
    "shaft_axis",
    "planar_seat",
    "shoulder_stop",
)

FEATURE_NAMES = [
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
    "center_x_norm",
    "center_y_norm",
    "center_z_norm",
    "bbox_x_ratio",
    "bbox_y_ratio",
    "bbox_z_ratio",
    "normal_abs_x",
    "normal_abs_y",
    "normal_abs_z",
    "axis_abs_x",
    "axis_abs_y",
    "axis_abs_z",
    "radius_ratio",
    "edge_count_log",
    "concavity",
    "has_helical_edge",
    "near_bbox_boundary",
]


def _normalized_interface_protocol(protocol: str) -> str:
  value = str(protocol or "legacy").strip().lower()
  if value not in {"legacy", "benchmark_v2"}:
    raise ValueError(
        "interface protocol must be 'legacy' or 'benchmark_v2', "
        f"got {protocol!r}"
    )
  return value


@dataclass
class CandidateInterface:
  """A local single-part mating interface candidate."""

  interface_id: str
  part_name: str
  body_uuid: str
  step_path: str
  face_index: Optional[int]
  surface_type: str
  role_hint: str
  local_frame: dict[str, Any]
  features: dict[str, float]
  score: Optional[float] = None
  label: Optional[int] = None
  label_source: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def extract_candidate_interfaces_from_step(
    step_path: str | Path,
    *,
    part_name: str = "",
    body_uuid: str = "",
    max_candidates: int = 0,
    protocol: str = "legacy",
) -> list[CandidateInterface]:
  """Legacy-compatible load wrapper around shape-boundary extraction."""

  protocol = _normalized_interface_protocol(protocol)
  if protocol != "legacy":
    raise ValueError(
        "benchmark_v2 candidates require a caller-supplied, already gauged "
        "shape; STEP reload is not a valid benchmark boundary"
    )
  path = Path(step_path)
  shape = load_step_shape(path)
  return extract_candidate_interfaces_from_shape(
      shape,
      part_name=part_name or path.stem,
      body_uuid=body_uuid or path.stem,
      source_step_path=path,
      max_candidates=max_candidates,
      equivariant_frames=False,
      protocol="legacy",
  )


def extract_candidate_interfaces_from_shape(
    shape: Any,
    *,
    part_name: str,
    body_uuid: str = "",
    source_step_path: str | Path | None = None,
    max_candidates: int = 0,
    equivariant_frames: bool = True,
    protocol: str = "legacy",
) -> list[CandidateInterface]:
  """Extract interfaces from a caller-supplied B-Rep without reloading STEP.

  Benchmark-v2 applies its per-part gauge before calling this function.  The
  legacy STEP wrapper above remains behavior-compatible for existing evidence.
  """

  protocol = _normalized_interface_protocol(protocol)
  if not str(part_name).strip():
    raise ValueError("part_name is required for shape-boundary extraction")
  source_path = (
      ""
      if source_step_path is None
      else str(Path(source_step_path).resolve(strict=False))
  )
  faces = extract_brep_faces_from_shape(shape, protocol=protocol)
  if not equivariant_frames:
    # Keep the historical world-axis fallback bit-for-bit on the legacy STEP
    # path.  Benchmark-v2's shape boundary opts into intrinsic B-Rep frames.
    for face in faces:
      face.metadata.pop("reference_direction", None)
  bbox_min, bbox_max = shape_bbox(shape)
  model_diag = float(np.linalg.norm(np.asarray(bbox_max) - np.asarray(bbox_min)))
  model_area = sum(max(0.0, float(face.area)) for face in faces) or 1.0

  candidates: list[CandidateInterface] = []
  safe_part_name = str(part_name).strip()
  safe_body_uuid = str(body_uuid).strip()
  for ordinal, face in enumerate(faces):
    frame = _local_frame_from_face(face)
    if frame is None:
      continue
    feature_values = interface_feature_dict(
        face=face,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        model_area=model_area,
        model_diag=model_diag,
    )
    face_index = _face_index(face)
    role_hint = infer_interface_role(
        face,
        feature_values=feature_values,
        protocol=protocol,
    )
    candidates.append(
        CandidateInterface(
            interface_id=f"iface_{ordinal:04d}_{role_hint}",
            part_name=safe_part_name,
            body_uuid=safe_body_uuid,
            step_path=source_path,
            face_index=face_index,
            surface_type=_surface_bucket(face.surface_type),
            role_hint=role_hint,
            local_frame=frame,
            features=feature_values,
            metadata={
                "area": float(face.area),
                "radius": None if face.radius is None else float(face.radius),
                "edge_count": int(face.edge_count),
                "concavity": int(face.concavity),
                "source": "single_step_geometry",
            },
        )
    )

  candidates.extend(
      _extract_composite_interfaces(
          faces=faces,
          bbox_min=bbox_min,
          bbox_max=bbox_max,
          model_area=model_area,
          model_diag=model_diag,
          part_name=safe_part_name,
          body_uuid=safe_body_uuid,
          step_path=source_path,
          protocol=protocol,
      )
  )

  candidates.sort(
      key=lambda item: (
          -float(bool(item.metadata.get("composite_interface", False))),
          -float(item.features.get("sqrt_area_ratio", 0.0)),
          item.interface_id,
      )
  )
  if max_candidates > 0:
    candidates = candidates[: int(max_candidates)]
  return candidates


def interface_feature_vector(candidate: CandidateInterface) -> list[float]:
  return [float(candidate.features.get(name, 0.0)) for name in FEATURE_NAMES]


def interface_feature_dict(
    *,
    face: BRepFace,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    model_area: float,
    model_diag: float,
) -> dict[str, float]:
  bbox_min = np.asarray(bbox_min, dtype=float).reshape(3)
  bbox_max = np.asarray(bbox_max, dtype=float).reshape(3)
  dims = np.maximum(bbox_max - bbox_min, 1e-9)
  center_norm = (np.asarray(face.center, dtype=float).reshape(3) - bbox_min) / dims
  bbox_extents = (
      np.zeros(3, dtype=float)
      if face.bbox_extents is None
      else np.asarray(face.bbox_extents, dtype=float).reshape(3)
  )
  bbox_ratio = bbox_extents / dims
  normal = (
      np.zeros(3, dtype=float)
      if face.normal is None
      else np.abs(np.asarray(face.normal, dtype=float).reshape(3))
  )
  axis = (
      np.zeros(3, dtype=float)
      if face.axis is None
      else np.abs(np.asarray(face.axis, dtype=float).reshape(3))
  )
  area_ratio = max(0.0, float(face.area)) / max(1e-9, float(model_area))
  radius_ratio = (
      0.0
      if face.radius is None
      else float(face.radius) / max(1e-9, float(model_diag))
  )
  near_boundary = float(
      np.max(np.maximum(1.0 - center_norm, center_norm)) >= 0.88
  )
  surface = _surface_bucket(face.surface_type)
  result = {name: 0.0 for name in FEATURE_NAMES}
  result["bias"] = 1.0
  result[f"surface_{surface}"] = 1.0
  result["log_area_ratio"] = math.log1p(1000.0 * area_ratio)
  result["sqrt_area_ratio"] = math.sqrt(area_ratio)
  result["center_x_norm"] = float(center_norm[0])
  result["center_y_norm"] = float(center_norm[1])
  result["center_z_norm"] = float(center_norm[2])
  result["bbox_x_ratio"] = float(bbox_ratio[0])
  result["bbox_y_ratio"] = float(bbox_ratio[1])
  result["bbox_z_ratio"] = float(bbox_ratio[2])
  result["normal_abs_x"] = float(normal[0])
  result["normal_abs_y"] = float(normal[1])
  result["normal_abs_z"] = float(normal[2])
  result["axis_abs_x"] = float(axis[0])
  result["axis_abs_y"] = float(axis[1])
  result["axis_abs_z"] = float(axis[2])
  result["radius_ratio"] = float(radius_ratio)
  result["edge_count_log"] = math.log1p(max(0, int(face.edge_count)))
  result["concavity"] = float(face.concavity)
  result["has_helical_edge"] = float(bool(face.has_helical_edge))
  result["near_bbox_boundary"] = near_boundary
  return result


def infer_interface_role(
    face: BRepFace,
    *,
    feature_values: Optional[dict[str, float]] = None,
    protocol: str = "legacy",
) -> str:
  protocol = _normalized_interface_protocol(protocol)
  surface = _surface_bucket(face.surface_type)
  if surface == "cylinder":
    features = feature_values or {}
    if protocol == "benchmark_v2":
      if face.has_helical_edge and face.concavity <= 0:
        return "threaded_hole"
      if face.concavity < 0:
        return "center_bore"
      if face.concavity > 0:
        if _cylinder_is_short_boss_invariant(face):
          return "pin_boss"
        return "shaft_axis"
      return "cylindrical_interface"
    centered = _axis_aware_centeredness_from_features(features, face.axis)
    boundary = float(features.get("near_bbox_boundary", 0.0) or 0.0)
    radius_ratio = float(features.get("radius_ratio", 0.0) or 0.0)
    if face.has_helical_edge and face.concavity <= 0:
      return "threaded_hole"
    if face.concavity < 0:
      if centered >= 0.58 and (boundary < 0.75 or radius_ratio <= 0.1):
        return "center_bore"
      return "slot_arc"
    if face.concavity > 0:
      if _cylinder_is_short_boss(face, feature_values=features):
        return "pin_boss"
      return "shaft_axis"
    if centered >= 0.68 and (boundary < 0.5 or radius_ratio <= 0.1):
      return "center_bore"
    return "cylindrical_interface"
  if surface == "plane":
    return "planar_seat"
  if surface in {"cone", "torus"}:
    return "shoulder_stop"
  return "generic_interface"


def _extract_composite_interfaces(
    *,
    faces: list[BRepFace],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    model_area: float,
    model_diag: float,
    part_name: str,
    body_uuid: str,
    step_path: str,
    protocol: str = "legacy",
) -> list[CandidateInterface]:
  """Build leakage-free mechanical interfaces from groups of single-part faces.

  Face-level cylinders are often too weak: an outer rim arc and a true bore are
  both just cylindrical surfaces. These composite interfaces encode the
  mechanical affordance we can infer from one part alone, without any assembly
  pair pose.
  """

  protocol = _normalized_interface_protocol(protocol)
  result: list[CandidateInterface] = []
  cylinders = [
      face
      for face in faces
      if _surface_bucket(face.surface_type) == "cylinder"
      and face.axis is not None
      and face.radius is not None
  ]
  planes = [
      face
      for face in faces
      if _surface_bucket(face.surface_type) == "plane"
      and face.normal is not None
  ]
  seen_faces: set[int] = set()

  for face in cylinders:
    feature_values = interface_feature_dict(
        face=face,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        model_area=model_area,
        model_diag=model_diag,
    )
    centered = _axis_aware_centeredness_from_features(feature_values, face.axis)
    boundary = float(feature_values.get("near_bbox_boundary", 0.0) or 0.0)
    radius_ratio = float(feature_values.get("radius_ratio", 0.0) or 0.0)
    face_index = _face_index(face)
    if face_index is None:
      continue
    role = ""
    if protocol == "benchmark_v2":
      if face.has_helical_edge and face.concavity <= 0:
        role = "threaded_hole"
      elif face.concavity < 0:
        role = "center_bore"
      elif face.concavity > 0:
        role = (
            "pin_boss"
            if _cylinder_is_short_boss_invariant(face)
            else "shaft_axis"
        )
    else:
      if face.has_helical_edge and face.concavity <= 0:
        role = "threaded_hole"
      elif face.concavity < 0 and centered >= 0.58 and (
          boundary < 0.8 or radius_ratio <= 0.1
      ):
        role = "center_bore"
      elif face.concavity == 0 and centered >= 0.68 and (
          boundary < 0.45 or radius_ratio <= 0.1
      ):
        role = "center_bore"
      elif face.concavity > 0 and _cylinder_is_short_boss(
          face, feature_values=feature_values
      ):
        role = "pin_boss"
      elif face.concavity > 0:
        role = "shaft_axis"
    if not role:
      continue
    frame = _local_frame_from_face(face)
    if frame is None:
      continue
    seen_faces.add(face_index)
    metadata = {
        "area": float(face.area),
        "radius": None if face.radius is None else float(face.radius),
        "edge_count": int(face.edge_count),
        "concavity": int(face.concavity),
        "source": "composite_single_part_geometry",
        "composite_interface": True,
        "composite_role": role,
        "centeredness": float(centered),
        "boundary_score": float(boundary),
        "radius_ratio": float(radius_ratio),
        "compatible_relations": _compatible_relations_for_role(role),
        "member_face_indices": [face_index],
    }
    result.append(
        CandidateInterface(
            interface_id=f"composite_{role}_{face_index:04d}",
            part_name=part_name,
            body_uuid=body_uuid,
            step_path=step_path,
            face_index=face_index,
            surface_type="cylinder",
            role_hint=role,
            local_frame=frame,
            features=feature_values,
            metadata=metadata,
        )
    )

  result.extend(
      _extract_obround_slots(
          cylinders=cylinders,
          bbox_min=bbox_min,
          bbox_max=bbox_max,
          model_area=model_area,
          model_diag=model_diag,
          part_name=part_name,
          body_uuid=body_uuid,
          step_path=step_path,
          protocol=protocol,
      )
  )

  # Add a few large, clean support/seat planes as composite stops. These are
  # critical for "insert until shoulder seats" relations.
  plane_candidates: list[tuple[float, BRepFace, dict[str, float]]] = []
  for face in planes:
    features = interface_feature_dict(
        face=face,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        model_area=model_area,
        model_diag=model_diag,
    )
    score = float(features.get("sqrt_area_ratio", 0.0))
    if protocol == "legacy":
      score -= 0.25 * float(features.get("near_bbox_boundary", 0.0))
    plane_candidates.append((score, face, features))
  for ordinal, (_, face, features) in enumerate(
      sorted(plane_candidates, key=lambda item: -item[0])[:6]
  ):
    frame = _local_frame_from_face(face)
    if frame is None:
      continue
    face_index = _face_index(face)
    metadata = {
        "area": float(face.area),
        "radius": None,
        "edge_count": int(face.edge_count),
        "concavity": int(face.concavity),
        "source": "composite_single_part_geometry",
        "composite_interface": True,
        "composite_role": "planar_seat",
        "compatible_relations": ["planar_seat", "seat_plane"],
        "member_face_indices": [] if face_index is None else [face_index],
    }
    result.append(
        CandidateInterface(
            interface_id=f"composite_planar_seat_{ordinal:04d}",
            part_name=part_name,
            body_uuid=body_uuid,
            step_path=step_path,
            face_index=face_index,
            surface_type="plane",
            role_hint="planar_seat",
            local_frame=frame,
            features=features,
            metadata=metadata,
        )
    )
  return result


def _extract_obround_slots(
    *,
    cylinders: list[BRepFace],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    model_area: float,
    model_diag: float,
    part_name: str,
    body_uuid: str,
    step_path: str,
    protocol: str = "legacy",
) -> list[CandidateInterface]:
  protocol = _normalized_interface_protocol(protocol)
  result: list[CandidateInterface] = []
  concave = [
      face
      for face in cylinders
      if face.concavity < 0 and face.axis is not None and face.radius is not None
  ]
  used: set[tuple[int, int]] = set()
  for i, face_a in enumerate(concave):
    idx_a = _face_index(face_a)
    if idx_a is None:
      continue
    for face_b in concave[i + 1 :]:
      idx_b = _face_index(face_b)
      if idx_b is None:
        continue
      key = tuple(sorted((idx_a, idx_b)))
      if key in used:
        continue
      if not _parallel(face_a.axis, face_b.axis, threshold=0.94):
        continue
      radius_a = float(face_a.radius or 0.0)
      radius_b = float(face_b.radius or 0.0)
      if radius_a <= 0.0 or radius_b <= 0.0:
        continue
      rel = abs(radius_a - radius_b) / max(radius_a, radius_b, 1e-6)
      if rel > 0.18:
        continue
      axis = normalize(np.asarray(face_a.axis, dtype=float))
      center_a = np.asarray(face_a.center, dtype=float).reshape(3)
      center_b = np.asarray(face_b.center, dtype=float).reshape(3)
      delta = center_b - center_a
      along_axis = abs(float(np.dot(delta, axis)))
      transverse = delta - float(np.dot(delta, axis)) * axis
      slot_len_between_arcs = float(np.linalg.norm(transverse))
      radius = 0.5 * (radius_a + radius_b)
      threshold_scale = (
          float(model_diag)
          if protocol == "legacy"
          else float(math.sqrt(max(1e-12, model_area)))
      )
      if along_axis > max(1.5 * radius, 0.08 * threshold_scale):
        continue
      if slot_len_between_arcs < max(1.2 * radius, 0.01 * threshold_scale):
        continue
      if slot_len_between_arcs > max(40.0 * radius, 0.9 * threshold_scale):
        continue
      try:
        x_axis = normalize(transverse)
        z_axis = axis
        y_axis = normalize(np.cross(z_axis, x_axis))
        x_axis = normalize(np.cross(y_axis, z_axis))
      except Exception:
        continue
      origin = 0.5 * (center_a + center_b)
      bbox_dims = np.maximum(np.asarray(bbox_max) - np.asarray(bbox_min), 1e-9)
      center_norm = (origin - np.asarray(bbox_min)) / bbox_dims
      features = {name: 0.0 for name in FEATURE_NAMES}
      features["bias"] = 1.0
      features["surface_cylinder"] = 1.0
      combined_area = max(0.0, float(face_a.area) + float(face_b.area))
      area_ratio = combined_area / max(1e-9, float(model_area))
      features["log_area_ratio"] = math.log1p(1000.0 * area_ratio)
      features["sqrt_area_ratio"] = math.sqrt(area_ratio)
      features["center_x_norm"] = float(center_norm[0])
      features["center_y_norm"] = float(center_norm[1])
      features["center_z_norm"] = float(center_norm[2])
      features["axis_abs_x"] = abs(float(z_axis[0]))
      features["axis_abs_y"] = abs(float(z_axis[1]))
      features["axis_abs_z"] = abs(float(z_axis[2]))
      features["radius_ratio"] = float(radius / max(1e-9, model_diag))
      features["edge_count_log"] = math.log1p(max(0, int(face_a.edge_count + face_b.edge_count)))
      features["concavity"] = -1.0
      features["near_bbox_boundary"] = float(
          np.max(np.maximum(1.0 - center_norm, center_norm)) >= 0.88
      )
      used.add(key)
      frame = {
          "origin": origin.tolist(),
          "x_axis": x_axis.tolist(),
          "y_axis": y_axis.tolist(),
          "z_axis": z_axis.tolist(),
          "normal": z_axis.tolist(),
          "axis": z_axis.tolist(),
      }
      result.append(
          CandidateInterface(
              interface_id=f"composite_obround_slot_{idx_a:04d}_{idx_b:04d}",
              part_name=part_name,
              body_uuid=body_uuid,
              step_path=step_path,
              face_index=None,
              surface_type="cylinder",
              role_hint="obround_slot",
              local_frame=frame,
              features=features,
              metadata={
                  "area": combined_area,
                  "radius": radius,
                  "edge_count": int(face_a.edge_count + face_b.edge_count),
                  "concavity": -1,
                  "source": "composite_single_part_geometry",
                  "composite_interface": True,
                  "composite_role": "obround_slot",
                  "slot_width": 2.0 * radius,
                  "slot_length": slot_len_between_arcs + 2.0 * radius,
                  "slot_center_distance": slot_len_between_arcs,
                  "compatible_relations": ["pin_in_slot", "boss_in_slot"],
                  "member_face_indices": [idx_a, idx_b],
              },
          )
      )
  result.sort(
      key=lambda item: (
          -float(item.metadata.get("slot_center_distance", 0.0)),
          item.interface_id,
      )
  )
  return result[:8]


def _axis_aware_centeredness_from_features(
    features: dict[str, float],
    axis: Optional[np.ndarray],
) -> float:
  coords = []
  for key in ("center_x_norm", "center_y_norm", "center_z_norm"):
    raw = features.get(key)
    if not isinstance(raw, (int, float)):
      return 0.5
    coords.append(float(raw))
  used = [0, 1, 2]
  if axis is not None:
    try:
      dominant = int(np.argmax(np.abs(np.asarray(axis, dtype=float).reshape(3))))
      used = [idx for idx in used if idx != dominant]
    except Exception:
      used = [0, 1, 2]
  arr = np.asarray([coords[idx] for idx in used], dtype=float)
  distance = float(np.linalg.norm(arr - 0.5))
  divisor = 0.5 if len(used) == 2 else 0.72
  return max(0.0, min(1.0, 1.0 - distance / divisor))


def _cylinder_is_short_boss(
    face: BRepFace,
    *,
    feature_values: Optional[dict[str, float]] = None,
) -> bool:
  if face.axis is None or face.bbox_extents is None or face.radius is None:
    return False
  try:
    features = feature_values or {}
    radius_ratio = float(features.get("radius_ratio", 0.0) or 0.0)
    centered = _axis_aware_centeredness_from_features(features, face.axis)
    # Large centered cylindrical arcs are often outer rims or scalloped grips,
    # not protruding assembly bosses. Treat them as non-mating surface evidence
    # so they do not compete with a true central bore.
    if radius_ratio > 0.12 and centered >= 0.8:
      return False
    axis = normalize(np.asarray(face.axis, dtype=float))
    extents = np.asarray(face.bbox_extents, dtype=float).reshape(3)
    axial_extent = float(np.dot(np.abs(axis), extents))
    radial_extent = max(1e-6, 2.0 * float(face.radius))
    return axial_extent <= 2.5 * radial_extent
  except Exception:
    return False


def _cylinder_is_short_boss_invariant(face: BRepFace) -> bool:
  """Classify boss aspect ratio without world axes, AABBs, or centres."""

  if face.radius is None:
    return False
  raw_extent = face.metadata.get("axial_extent")
  if not isinstance(raw_extent, (int, float)):
    return False
  axial_extent = max(0.0, float(raw_extent))
  radial_extent = max(1e-9, 2.0 * float(face.radius))
  return axial_extent <= 2.5 * radial_extent


def _parallel(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    *,
    threshold: float,
) -> bool:
  if a is None or b is None:
    return False
  try:
    return abs(float(np.dot(normalize(a), normalize(b)))) >= float(threshold)
  except Exception:
    return False


def _compatible_relations_for_role(role: str) -> list[str]:
  if role == "center_bore":
    return ["shaft_in_bore", "insert_axis"]
  if role == "threaded_hole":
    return ["screw_in_hole", "threaded_interference"]
  if role == "obround_slot":
    return ["pin_in_slot", "boss_in_slot"]
  if role == "pin_boss":
    return ["boss_in_slot", "pin_in_slot", "shaft_in_bore"]
  if role == "shaft_axis":
    return ["shaft_in_bore", "screw_in_hole", "insert_axis"]
  if role == "planar_seat":
    return ["planar_seat", "seat_plane"]
  if role == "shoulder_stop":
    return ["planar_seat", "seat_plane"]
  return ["generic_contact"]


def label_interfaces_from_assembly_contacts(
    candidates: list[CandidateInterface],
    assembly_json_path: str | Path,
    *,
    positive_distance_ratio: float = 0.08,
) -> list[CandidateInterface]:
  """Assign train labels from Fusion contacts without storing pair transforms."""

  assembly_path = Path(assembly_json_path)
  data = json.loads(assembly_path.read_text(encoding="utf-8"))
  contacts_by_body = _contact_face_descriptors_by_body(data)
  if not contacts_by_body:
    return candidates

  # Work per body so scale estimation is local and never pairwise.
  by_body: dict[str, list[CandidateInterface]] = {}
  for candidate in candidates:
    by_body.setdefault(candidate.body_uuid, []).append(candidate)

  for body_uuid, body_candidates in by_body.items():
    descriptors = contacts_by_body.get(body_uuid, [])
    positive_indices = {
        int(item["face_index"])
        for item in descriptors
        if isinstance(item.get("face_index"), int)
    }
    descriptor_points = [
        item for item in descriptors if isinstance(item.get("point"), list)
    ]
    scale = _estimate_contact_scale(body_candidates, descriptor_points)
    body_diag = _candidate_body_diag(body_candidates)
    threshold = max(1e-6, float(body_diag) * float(positive_distance_ratio))
    for candidate in body_candidates:
      label = 0
      source = "negative_uncontacted_face"
      if candidate.face_index is not None and candidate.face_index in positive_indices:
        label = 1
        source = "positive_contact_face_index"
      elif _matches_contact_descriptor(
          candidate=candidate,
          descriptors=descriptor_points,
          scale=scale,
          threshold=threshold,
      ):
        label = 1
        source = "positive_contact_point_match"
      candidate.label = label
      candidate.label_source = source
  return candidates


def _local_frame_from_face(face: BRepFace) -> Optional[dict[str, Any]]:
  # Cylindrical interfaces mate along their cylinder axis.  OCC face normals on
  # cylinders are radial and can be perpendicular to the actual insertion axis;
  # using them as the MCF z axis makes learned insert programs rotate shafts
  # sideways.  Planar interfaces still fall back to the face normal.
  z_axis = face.axis if face.axis is not None else face.normal
  if z_axis is None:
    return None
  try:
    z_axis = normalize(z_axis)
    reference = face.metadata.get("reference_direction")
    if isinstance(reference, (list, tuple, np.ndarray)):
      tangent = np.asarray(reference, dtype=float).reshape(3)
      tangent = tangent - float(np.dot(tangent, z_axis)) * z_axis
      if float(np.linalg.norm(tangent)) > 1e-10:
        x_axis = normalize(tangent)
        y_axis = normalize(np.cross(z_axis, x_axis))
        x_axis = normalize(np.cross(y_axis, z_axis))
      else:
        x_axis, y_axis, z_axis = orthonormal_basis_from_z(z_axis)
    else:
      x_axis, y_axis, z_axis = orthonormal_basis_from_z(z_axis)
  except Exception:
    return None
  return {
      "origin": np.asarray(face.center, dtype=float).reshape(3).tolist(),
      "x_axis": x_axis.tolist(),
      "y_axis": y_axis.tolist(),
      "z_axis": z_axis.tolist(),
      "normal": None if face.normal is None else normalize(face.normal).tolist(),
      "axis": None if face.axis is None else normalize(face.axis).tolist(),
  }


def _face_index(face: BRepFace) -> Optional[int]:
  raw = face.metadata.get("face_index")
  if isinstance(raw, int):
    return int(raw)
  if isinstance(raw, float):
    return int(raw)
  return None


def _surface_bucket(surface_type: str) -> str:
  text = str(surface_type or "").lower()
  if "plane" in text:
    return "plane"
  if "cylinder" in text:
    return "cylinder"
  if "cone" in text:
    return "cone"
  if "sphere" in text:
    return "sphere"
  if "torus" in text:
    return "torus"
  if any(token in text for token in ("spline", "bspline", "bezier", "nurbs")):
    return "spline"
  return "other"


def _contact_face_descriptors_by_body(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
  result: dict[str, list[dict[str, Any]]] = {}
  contacts = data.get("contacts")
  if not isinstance(contacts, list):
    return result
  for contact in contacts:
    if not isinstance(contact, dict):
      continue
    for key in ("entity_one", "entity_two"):
      entity = contact.get(key)
      if not isinstance(entity, dict):
        continue
      body = entity.get("body")
      if not isinstance(body, str) or not body.strip():
        continue
      face_index = entity.get("index")
      point = _point3(entity.get("point_on_entity"))
      result.setdefault(body, []).append(
          {
              "face_index": int(face_index) if isinstance(face_index, int) else None,
              "surface_type": _surface_bucket(str(entity.get("surface_type") or "")),
              "point": point,
          }
      )
  return result


def _point3(value: Any) -> Optional[list[float]]:
  if not isinstance(value, dict):
    return None
  coords = []
  for key in ("x", "y", "z"):
    raw = value.get(key)
    if not isinstance(raw, (int, float)):
      return None
    coords.append(float(raw))
  return coords


def _candidate_body_diag(candidates: Iterable[CandidateInterface]) -> float:
  origins = [
      np.asarray(candidate.local_frame.get("origin"), dtype=float).reshape(3)
      for candidate in candidates
      if isinstance(candidate.local_frame.get("origin"), list)
  ]
  if len(origins) < 2:
    return 1.0
  stacked = np.stack(origins, axis=0)
  return float(np.linalg.norm(np.max(stacked, axis=0) - np.min(stacked, axis=0)))


def _estimate_contact_scale(
    candidates: list[CandidateInterface],
    descriptors: list[dict[str, Any]],
) -> float:
  if not candidates or not descriptors:
    return 1.0
  candidate_origins = [
      np.asarray(candidate.local_frame["origin"], dtype=float).reshape(3)
      for candidate in candidates
  ]
  points = [
      np.asarray(item["point"], dtype=float).reshape(3)
      for item in descriptors
      if isinstance(item.get("point"), list)
  ]
  if not points:
    return 1.0
  best_scale = 1.0
  best_error = float("inf")
  for scale in (1.0, 10.0, 0.1, 25.4, 2.54):
    errors = []
    for point in points[:16]:
      scaled = point * scale
      errors.append(min(float(np.linalg.norm(origin - scaled)) for origin in candidate_origins))
    error = float(np.median(errors)) if errors else float("inf")
    if error < best_error:
      best_error = error
      best_scale = scale
  return best_scale


def _matches_contact_descriptor(
    *,
    candidate: CandidateInterface,
    descriptors: list[dict[str, Any]],
    scale: float,
    threshold: float,
) -> bool:
  origin = np.asarray(candidate.local_frame["origin"], dtype=float).reshape(3)
  for descriptor in descriptors:
    if descriptor.get("surface_type") != candidate.surface_type:
      continue
    point = descriptor.get("point")
    if not isinstance(point, list):
      continue
    scaled = np.asarray(point, dtype=float).reshape(3) * float(scale)
    if float(np.linalg.norm(origin - scaled)) <= float(threshold):
      return True
  return False
