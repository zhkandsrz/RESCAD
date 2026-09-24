"""Mobility-quotiented source-pose equivalence for LinkCAD programs."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


TRANSLATION_TOLERANCE_MM = 0.1
ANGULAR_TOLERANCE_DEG = 1.0


def _vector(value: Any, *, scale: float = 1.0) -> np.ndarray:
  raw = (
      [value[axis] for axis in ("x", "y", "z")]
      if isinstance(value, Mapping)
      else value
  )
  result = scale * np.asarray(raw, dtype=float)
  if result.shape != (3,) or not np.isfinite(result).all():
    raise ValueError("Fusion vector differs")
  return result


def _unit(value: np.ndarray) -> np.ndarray:
  norm = float(np.linalg.norm(value))
  if not math.isfinite(norm) or norm <= 1e-12:
    raise ValueError("frame direction is degenerate")
  return value / norm


def _transform_matrix(raw: Mapping[str, Any]) -> np.ndarray:
  matrix = np.eye(4, dtype=float)
  matrix[:3, :3] = np.column_stack([
      _unit(_vector(raw[name])) for name in ("x_axis", "y_axis", "z_axis")
  ])
  matrix[:3, 3] = _vector(raw["origin"], scale=10.0)
  if not np.allclose(
      matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6
  ) or np.linalg.det(matrix[:3, :3]) < 0.999999:
    raise ValueError("Fusion occurrence rotation differs")
  return matrix


def source_occurrence_world_matrices_v1(
    assembly: Mapping[str, Any],
) -> dict[str, np.ndarray]:
  occurrences = assembly.get("occurrences")
  tree = assembly.get("tree")
  if not isinstance(occurrences, Mapping) or not isinstance(tree, Mapping):
    raise ValueError("Fusion assembly occurrence tree differs")
  root_tree = tree.get("root")
  if not isinstance(root_tree, Mapping):
    raise ValueError("Fusion assembly root tree differs")
  result: dict[str, np.ndarray] = {}

  def walk(subtree: Mapping[str, Any], parent: np.ndarray) -> None:
    for occurrence_id, children in subtree.items():
      if occurrence_id in result or occurrence_id not in occurrences:
        raise ValueError("Fusion occurrence tree identity differs")
      occurrence = occurrences[occurrence_id]
      if not isinstance(occurrence, Mapping) or not isinstance(children, Mapping):
        raise ValueError("Fusion occurrence row differs")
      world = parent @ _transform_matrix(occurrence["transform"])
      result[str(occurrence_id)] = world
      walk(children, world)

  walk(root_tree, np.eye(4, dtype=float))
  if set(result) != {str(value) for value in occurrences}:
    raise ValueError("Fusion occurrence table has unreachable rows")
  return result


def _endpoint_world_frame(endpoint: Mapping[str, Any]) -> np.ndarray:
  origin = _vector(endpoint["origin"], scale=10.0)
  primary = _unit(_vector(endpoint["primary_axis"]))
  secondary = _vector(endpoint["secondary_axis"])
  secondary = secondary - primary * float(np.dot(primary, secondary))
  if float(np.linalg.norm(secondary)) <= 1e-9:
    basis = min(np.eye(3), key=lambda row: abs(float(np.dot(row, primary))))
    secondary = basis - primary * float(np.dot(basis, primary))
  x_axis = _unit(secondary)
  y_axis = _unit(np.cross(primary, x_axis))
  matrix = np.eye(4, dtype=float)
  matrix[:3, :3] = np.column_stack((x_axis, y_axis, primary))
  matrix[:3, 3] = origin
  return matrix


def _rotation_angle_deg(rotation: np.ndarray) -> float:
  cosine = min(1.0, max(-1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
  return math.degrees(math.acos(cosine))


def mobility_quotient_errors_v1(
    *,
    source_child_world: np.ndarray,
    predicted_child_world: np.ndarray,
    child_endpoint_world: Mapping[str, Any],
    mobility: str,
) -> dict[str, float | bool | str]:
  """Compare a predicted child pose to the source pose modulo joint mobility."""

  if mobility not in {"fixed", "revolute", "prismatic", "cylindrical"}:
    raise ValueError("LinkCAD mobility quotient differs")
  source_child_world = np.asarray(source_child_world, dtype=float).reshape(4, 4)
  predicted_child_world = np.asarray(predicted_child_world, dtype=float).reshape(4, 4)
  child_frame_world = _endpoint_world_frame(child_endpoint_world)
  child_frame_local = np.linalg.inv(source_child_world) @ child_frame_world
  predicted_frame_world = predicted_child_world @ child_frame_local
  delta = np.linalg.inv(child_frame_world) @ predicted_frame_world
  translation = delta[:3, 3]
  rotation = delta[:3, :3]
  axial_translation = abs(float(translation[2]))
  perpendicular_translation = float(np.linalg.norm(translation[:2]))
  full_translation = float(np.linalg.norm(translation))
  full_rotation = _rotation_angle_deg(rotation)
  axis_rotation = math.degrees(math.acos(min(
      1.0, max(-1.0, float(rotation[2, 2]))
  )))
  if mobility == "fixed":
    translation_error = full_translation
    rotation_error = full_rotation
  elif mobility == "revolute":
    translation_error = full_translation
    rotation_error = axis_rotation
  elif mobility == "prismatic":
    translation_error = perpendicular_translation
    rotation_error = full_rotation
  else:
    translation_error = perpendicular_translation
    rotation_error = axis_rotation
  equivalent = (
      translation_error <= TRANSLATION_TOLERANCE_MM
      and rotation_error <= ANGULAR_TOLERANCE_DEG
  )
  return {
      "mobility": mobility,
      "translation_error_mm": translation_error,
      "rotation_error_deg": rotation_error,
      "full_translation_mm": full_translation,
      "axial_translation_mm": axial_translation,
      "perpendicular_translation_mm": perpendicular_translation,
      "full_rotation_deg": full_rotation,
      "axis_rotation_deg": axis_rotation,
      "translation_tolerance_mm": TRANSLATION_TOLERANCE_MM,
      "angular_tolerance_deg": ANGULAR_TOLERANCE_DEG,
      "equivalent": equivalent,
  }


__all__ = [
    "ANGULAR_TOLERANCE_DEG",
    "TRANSLATION_TOLERANCE_MM",
    "mobility_quotient_errors_v1",
    "source_occurrence_world_matrices_v1",
]
