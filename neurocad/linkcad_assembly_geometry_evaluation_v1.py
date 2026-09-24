"""Failure-aware geometric evaluation for complete LinkCAD assemblies.

The metrics complement exact interface and kernel verdicts; they do not replace
them.  Assemblies are aligned by the declared anchor occurrence, scored per
role so that large base parts cannot dominate, and aggregated with missing or
invalid outputs retained in the denominator.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark_v2_training_provenance import capture_file_artifact
from .cadquery_backend import load_step_shape


ROW_SCHEMA_VERSION = "linkcad_assembly_geometry_row.v1"
SUMMARY_SCHEMA_VERSION = "linkcad_assembly_geometry_summary.v1"


def _matrix(row_major: Sequence[float]) -> np.ndarray:
  if len(row_major) != 16:
    raise ValueError("LinkCAD geometry pose must contain sixteen values")
  matrix = np.asarray(row_major, dtype=float).reshape(4, 4)
  if not np.isfinite(matrix).all() or not np.allclose(
      matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8,
  ):
    raise ValueError("LinkCAD geometry pose is not homogeneous")
  rotation = matrix[:3, :3]
  if (
      not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
      or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
  ):
    raise ValueError("LinkCAD geometry pose is not rigid")
  return matrix


def _components(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
  rows = manifest.get("components")
  if not isinstance(rows, list) or not rows:
    raise ValueError("LinkCAD geometry manifest lacks components")
  result = {str(row["role_id"]): row for row in rows}
  if len(result) != len(rows):
    raise ValueError("LinkCAD geometry component role duplicates")
  return result


def _resolved_step(
    component: Mapping[str, Any], root: Path,
) -> Path:
  path = (root / str(component["source_step_path"])).resolve()
  try:
    path.relative_to(root)
  except ValueError as error:
    raise ValueError("LinkCAD geometry STEP escapes dataset root") from error
  capture = capture_file_artifact(path, label="LinkCAD geometry source STEP")
  if capture.sha256 != str(component["source_step_sha256"]):
    raise ValueError("LinkCAD geometry source STEP bytes differ")
  return path


def _mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
  shape = load_step_shape(path)
  bounds = shape.BoundingBox()
  diagonal = float(np.linalg.norm(np.asarray((
      bounds.xmax - bounds.xmin,
      bounds.ymax - bounds.ymin,
      bounds.zmax - bounds.zmin,
  ))))
  vertices, triangles = shape.tessellate(max(diagonal * 0.002, 1e-4))
  vertex_array = np.asarray([row.toTuple() for row in vertices], dtype=float)
  triangle_array = np.asarray(triangles, dtype=int)
  if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or not len(vertices):
    raise ValueError("LinkCAD geometry tessellation has no vertices")
  if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
    raise ValueError("LinkCAD geometry tessellation has no triangles")
  return vertex_array, triangle_array


def _surface_samples(
    vertices: np.ndarray, triangles: np.ndarray, *, count: int, seed: int,
) -> np.ndarray:
  if count < 32:
    raise ValueError("LinkCAD geometry surface sample count is too small")
  triplets = vertices[triangles]
  areas = 0.5 * np.linalg.norm(
      np.cross(triplets[:, 1] - triplets[:, 0],
               triplets[:, 2] - triplets[:, 0]), axis=1,
  )
  total_area = float(np.sum(areas))
  if not math.isfinite(total_area) or total_area <= 0.0:
    raise ValueError("LinkCAD geometry tessellation has zero area")
  generator = np.random.default_rng(seed)
  chosen = generator.choice(
      len(triplets), size=count, replace=True, p=areas / total_area,
  )
  u = np.sqrt(generator.random(count))
  v = generator.random(count)
  selected = triplets[chosen]
  return (
      (1.0 - u)[:, None] * selected[:, 0]
      + (u * (1.0 - v))[:, None] * selected[:, 1]
      + (u * v)[:, None] * selected[:, 2]
  )


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
  return points @ pose[:3, :3].T + pose[:3, 3]


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
  rows = []
  for start in range(0, len(source), 256):
    chunk = source[start:start + 256]
    squared = np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2)
    rows.append(np.sqrt(np.min(squared, axis=1)))
  return np.concatenate(rows)


def _rotation_error_deg(
    predicted: np.ndarray, reference: np.ndarray,
    symmetries: Sequence[Sequence[float]] | None,
) -> float:
  alternatives = [np.eye(3)]
  for raw in symmetries or ():
    values = np.asarray(raw, dtype=float)
    if values.size == 16:
      values = values.reshape(4, 4)[:3, :3]
    elif values.size == 9:
      values = values.reshape(3, 3)
    else:
      raise ValueError("LinkCAD geometry symmetry rotation differs")
    alternatives.append(values)
  best = 180.0
  for symmetry in alternatives:
    delta = reference.T @ predicted @ symmetry
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    best = min(best, math.degrees(math.acos(cosine)))
  return best


def _failure_row(
    *, query_id: str, status: str, intended_exact: bool | None,
) -> dict[str, Any]:
  return {
      "schema_version": ROW_SCHEMA_VERSION,
      "query_id": query_id,
      "status": status,
      "scoreable": False,
      "all_part_assignment_correct": False,
      "collision_free": False,
      "intended_exact": intended_exact is True,
      "part_aware_chamfer_pct_bbox": None,
      "part_accuracy_at_1pct": None,
      "surface_fscore_at_1pct": None,
      "mean_translation_error_pct_bbox": None,
      "mean_rotation_error_deg": None,
  }


def evaluate_assembly_geometry_v1(
    *, prediction_manifest: Mapping[str, Any] | None,
    reference_manifest: Mapping[str, Any], dataset_root: str | Path,
    surface_sample_count_per_component: int = 1024,
    failure_reason: str | None = None,
    intended_exact: bool | None = None,
) -> dict[str, Any]:
  """Score one complete assembly after anchor-gauge alignment."""

  query_id = str(reference_manifest["query_id"])
  if prediction_manifest is None:
    return _failure_row(
        query_id=query_id, status=failure_reason or "missing_output",
        intended_exact=intended_exact,
    )
  if str(prediction_manifest.get("query_id")) != query_id:
    raise ValueError("LinkCAD geometry query identity differs")
  reference = _components(reference_manifest)
  predicted = _components(prediction_manifest)
  if set(reference) != set(predicted):
    return _failure_row(
        query_id=query_id, status="incomplete_component_domain",
        intended_exact=intended_exact,
    )
  anchor = str(reference_manifest.get("anchor_role") or sorted(reference)[0])
  if anchor not in reference or anchor not in predicted:
    raise ValueError("LinkCAD geometry anchor role differs")

  reference_pose = {
      role: _matrix(row["world_pose_row_major"])
      for role, row in reference.items()
  }
  predicted_pose = {
      role: _matrix(row["world_pose_row_major"])
      for role, row in predicted.items()
  }
  gauge = reference_pose[anchor] @ np.linalg.inv(predicted_pose[anchor])
  predicted_pose = {
      role: gauge @ pose for role, pose in predicted_pose.items()
  }

  root = Path(dataset_root).resolve()
  mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  reference_geometry = {}
  predicted_geometry = {}
  all_reference_vertices = []
  for role in sorted(reference):
    for source, target, pose in (
        (reference[role], reference_geometry, reference_pose[role]),
        (predicted[role], predicted_geometry, predicted_pose[role]),
    ):
      path = _resolved_step(source, root)
      sha256 = str(source["source_step_sha256"])
      if sha256 not in mesh_cache:
        mesh_cache[sha256] = _mesh(path)
      vertices, triangles = mesh_cache[sha256]
      seed_material = f"{sha256}:{role}".encode("utf-8")
      seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
      samples = _surface_samples(
          vertices, triangles,
          count=surface_sample_count_per_component, seed=seed,
      )
      world_vertices = _transform(vertices, pose)
      target[role] = {
          "samples": _transform(samples, pose),
          "vertices": world_vertices,
      }
      if source is reference[role]:
        all_reference_vertices.append(world_vertices)

  combined_reference = np.concatenate(all_reference_vertices, axis=0)
  diagonal = float(np.linalg.norm(
      np.max(combined_reference, axis=0) - np.min(combined_reference, axis=0)
  ))
  if not math.isfinite(diagonal) or diagonal <= 1e-9:
    raise ValueError("LinkCAD geometry reference assembly is degenerate")
  threshold = 0.01 * diagonal

  chamfer_rows = []
  fscore_rows = []
  for role in sorted(reference):
    pred_points = predicted_geometry[role]["samples"]
    ref_points = reference_geometry[role]["samples"]
    pred_to_ref = _nearest_distances(pred_points, ref_points)
    ref_to_pred = _nearest_distances(ref_points, pred_points)
    chamfer_rows.append(0.5 * (
        float(np.mean(pred_to_ref)) + float(np.mean(ref_to_pred))
    ))
    precision = float(np.mean(pred_to_ref <= threshold))
    recall = float(np.mean(ref_to_pred <= threshold))
    fscore_rows.append(
        0.0 if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )

  matching_roles = [
      role for role in reference
      if str(reference[role]["source_step_sha256"])
      == str(predicted[role]["source_step_sha256"])
  ]
  translation_errors = []
  rotation_errors = []
  for role in matching_roles:
    translation_errors.append(float(np.linalg.norm(
        predicted_pose[role][:3, 3] - reference_pose[role][:3, 3]
    )))
    rotation_errors.append(_rotation_error_deg(
        predicted_pose[role][:3, :3], reference_pose[role][:3, :3],
        reference[role].get("symmetry_rotations_row_major"),
    ))

  verification = prediction_manifest.get("verification")
  collision_free = (
      isinstance(verification, Mapping)
      and verification.get("global_assembly_conditionally_feasible") is True
  )
  result = {
      "schema_version": ROW_SCHEMA_VERSION,
      "query_id": query_id,
      "status": "scored",
      "scoreable": True,
      "component_count": len(reference),
      "pose_metric_component_count": len(matching_roles),
      "reference_bbox_diagonal_mm": diagonal,
      "surface_sample_count_per_component": surface_sample_count_per_component,
      "all_part_assignment_correct": len(matching_roles) == len(reference),
      "collision_free": collision_free,
      "intended_exact": intended_exact is True,
      "part_aware_chamfer_pct_bbox": (
          100.0 * float(np.mean(chamfer_rows)) / diagonal
      ),
      "part_accuracy_at_1pct": float(np.mean(
          np.asarray(chamfer_rows, dtype=float) <= threshold
      )),
      "surface_fscore_at_1pct": float(np.mean(fscore_rows)),
      "mean_translation_error_pct_bbox": (
          None if not translation_errors
          else 100.0 * float(np.mean(translation_errors)) / diagonal
      ),
      "mean_rotation_error_deg": (
          None if not rotation_errors else float(np.mean(rotation_errors))
      ),
  }
  return result


def _mean(values: Sequence[float]) -> float | None:
  return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def summarize_assembly_geometry_v1(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Aggregate complete-query rows without dropping terminal failures."""

  if not rows:
    raise ValueError("LinkCAD geometry summary is empty")
  if len({str(row["query_id"]) for row in rows}) != len(rows):
    raise ValueError("LinkCAD geometry summary query identity duplicates")
  count = len(rows)
  scored = [row for row in rows if row.get("scoreable") is True]
  status_counts: dict[str, int] = {}
  for row in rows:
    status = str(row["status"])
    status_counts[status] = status_counts.get(status, 0) + 1

  conditional_chamfer = [
      float(row["part_aware_chamfer_pct_bbox"]) for row in scored
  ]
  conditional_fscore = [
      float(row["surface_fscore_at_1pct"]) for row in scored
  ]
  conditional_part_accuracy = [
      float(row["part_accuracy_at_1pct"]) for row in scored
  ]
  failure_chamfer = [
      min(float(row["part_aware_chamfer_pct_bbox"]), 100.0)
      if row.get("scoreable") is True else 100.0
      for row in rows
  ]
  failure_fscore = [
      float(row["surface_fscore_at_1pct"])
      if row.get("scoreable") is True else 0.0
      for row in rows
  ]
  failure_part_accuracy = [
      float(row["part_accuracy_at_1pct"])
      if row.get("scoreable") is True else 0.0
      for row in rows
  ]
  return {
      "schema_version": SUMMARY_SCHEMA_VERSION,
      "query_count": count,
      "scoreable_count": len(scored),
      "scoreable_rate": len(scored) / count,
      "all_part_assignment_rate": sum(
          row.get("all_part_assignment_correct") is True for row in rows
      ) / count,
      "collision_free_rate": sum(
          row.get("collision_free") is True for row in rows
      ) / count,
      "intended_exact_rate": sum(
          row.get("intended_exact") is True for row in rows
      ) / count,
      "conditional_part_aware_chamfer_pct_bbox_mean": _mean(
          conditional_chamfer
      ),
      "conditional_surface_fscore_at_1pct_mean": _mean(conditional_fscore),
      "conditional_part_accuracy_at_1pct_mean": _mean(
          conditional_part_accuracy
      ),
      "failure_aware_part_aware_chamfer_pct_bbox_mean": _mean(
          failure_chamfer
      ),
      "failure_aware_surface_fscore_at_1pct_mean": _mean(failure_fscore),
      "failure_aware_part_accuracy_at_1pct_mean": _mean(
          failure_part_accuracy
      ),
      "status_counts": dict(sorted(status_counts.items())),
  }


__all__ = [
    "evaluate_assembly_geometry_v1",
    "summarize_assembly_geometry_v1",
]
