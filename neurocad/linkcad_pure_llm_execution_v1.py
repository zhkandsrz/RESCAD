"""One-shot CAD-kernel scoring and export for pure-LLM assemblies."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from .benchmark_v2_training_provenance import capture_file_artifact
from .cadquery_backend import (
    boolean_intersection,
    load_step_shape,
    shape_volume,
    transform_shape,
)
from .domain_types import Transform
from .linkcad_assembly_export_v1 import _location, inspect_assembly_step_v1
from .linkcad_kinematic_execution_v2 import (
    _primitive_role,
    resolve_primitive_v2,
)


SCHEMA_VERSION = "linkcad_pure_llm_kernel_audit.v1"
COLLISION_VOLUME_TOLERANCE_MM3 = 1e-7
INTERFACE_DISTANCE_TOLERANCE_MM = 0.1
INTERFACE_ANGLE_TOLERANCE_DEG = 1.0


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")).hexdigest()


def _matrix(row_major: Sequence[float]) -> np.ndarray:
  matrix = np.asarray(row_major, dtype=float).reshape(4, 4)
  if not np.isfinite(matrix).all() or not np.allclose(
      matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5,
  ):
    raise ValueError("Pure-LLM execution pose is not homogeneous")
  rotation = matrix[:3, :3]
  if (
      not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
      or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3)
  ):
    raise ValueError("Pure-LLM execution pose is not rigid")
  return matrix


def _bbox_overlap_upper(left: Any, right: Any) -> float:
  a, b = left.BoundingBox(), right.BoundingBox()
  overlap = np.asarray((
      min(a.xmax, b.xmax) - max(a.xmin, b.xmin),
      min(a.ymax, b.ymax) - max(a.ymin, b.ymin),
      min(a.zmax, b.zmax) - max(a.zmin, b.zmin),
  ))
  return 0.0 if np.any(overlap <= 0.0) else float(np.prod(overlap))


def _support_family(left: Any, right: Any) -> str:
  roles = (_primitive_role(left), _primitive_role(right))
  if roles == ("axis", "axis"):
    return "axial_support"
  if roles == ("plane", "plane"):
    return "planar_support"
  if sorted(roles) == ["axis", "plane"]:
    return "axis_plane_support"
  return "unsupported_support"


def _world_frame(frame: Any, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  origin = pose[:3, :3] @ np.asarray(frame.origin_local_mm) + pose[:3, 3]
  local_rotation = np.asarray(frame.rotation_local).reshape(3, 3)
  primary = pose[:3, :3] @ local_rotation[:, 2]
  primary /= np.linalg.norm(primary)
  return origin, primary


def _axis_interval_gap(left: Any, right: Any, axis: np.ndarray) -> float:
  def interval(shape: Any) -> tuple[float, float]:
    box = shape.BoundingBox()
    corners = np.asarray([
        (x, y, z)
        for x in (box.xmin, box.xmax)
        for y in (box.ymin, box.ymax)
        for z in (box.zmin, box.zmax)
    ])
    values = corners @ axis
    return float(values.min()), float(values.max())

  a0, a1 = interval(left)
  b0, b1 = interval(right)
  return max(0.0, a0 - b1, b0 - a1)


def _interface_observation(
    *, left: Any, right: Any, pose_a: np.ndarray, pose_b: np.ndarray,
    world_shape_a: Any, world_shape_b: Any,
) -> dict[str, Any]:
  best = None
  for frame_a in left.frames:
    for frame_b in right.frames:
      family = _support_family(frame_a, frame_b)
      if family == "unsupported_support":
        continue
      origin_a, primary_a = _world_frame(frame_a, pose_a)
      origin_b, primary_b = _world_frame(frame_b, pose_b)
      cosine = float(np.clip(abs(np.dot(primary_a, primary_b)), -1.0, 1.0))
      angle = math.degrees(math.acos(cosine))
      delta = origin_b - origin_a
      if family == "axial_support":
        distance = float(np.linalg.norm(delta - np.dot(delta, primary_a) * primary_a))
        interval_gap = _axis_interval_gap(world_shape_a, world_shape_b, primary_a)
      else:
        distance = float(np.linalg.norm(delta))
        interval_gap = 0.0
      accepted = (
          angle <= INTERFACE_ANGLE_TOLERANCE_DEG
          and distance <= INTERFACE_DISTANCE_TOLERANCE_MM
          and interval_gap <= INTERFACE_DISTANCE_TOLERANCE_MM
      )
      row = {
          "support_family": family,
          "raw_primitive_a": int(frame_a.raw_index),
          "raw_primitive_b": int(frame_b.raw_index),
          "axis_angle_error_deg": angle,
          "interface_distance_error_mm": distance,
          "axial_interval_gap_mm": interval_gap,
          "interface_satisfied": accepted,
      }
      score = (
          0 if accepted else 1,
          angle / INTERFACE_ANGLE_TOLERANCE_DEG
          + distance / INTERFACE_DISTANCE_TOLERANCE_MM
          + interval_gap / INTERFACE_DISTANCE_TOLERANCE_MM,
      )
      if best is None or score < best[0]:
        best = (score, row)
  return (
      best[1] if best is not None else {
          "support_family": "unsupported_support",
          "interface_satisfied": False,
          "failure_reason": "selected primitive pair has no executable support",
      }
  )


def execute_and_export_pure_llm_v1(
    *, query: Mapping[str, Any], prediction: Mapping[str, Any],
    dataset_root: str | Path, output_step: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
  """Execute the LLM matrices exactly once and audit without pose changes."""

  import cadquery as cq

  query_id = str(query["query_id"])
  if str(prediction.get("query_id")) != query_id:
    raise ValueError("Pure-LLM execution query differs")
  selected_ids = {
      str(k): str(v) for k, v in prediction["candidate_by_role"].items()
  }
  candidates = {}
  for role in query["roles"]:
    role_id = str(role["role_id"])
    by_id = {
        str(row["candidate_id"]): row
        for row in query["candidate_sets"][role_id]
    }
    candidates[role_id] = by_id[selected_ids[role_id]]
  if set(candidates) != set(selected_ids):
    raise ValueError("Pure-LLM execution role domain differs")
  poses = {
      str(role): _matrix(values)
      for role, values in prediction["role_pose_row_major"].items()
  }
  if set(poses) != set(candidates):
    raise ValueError("Pure-LLM execution pose domain differs")

  root = Path(dataset_root).resolve()
  components, local_shapes, world_shapes = [], {}, {}
  assembly = cq.Assembly(name=f"pure_llm_{query_id}")
  for role_id in sorted(candidates):
    candidate = candidates[role_id]
    source = (root / str(candidate["step_path"])).resolve()
    try:
      source.relative_to(root)
    except ValueError as error:
      raise ValueError("Pure-LLM execution STEP escapes dataset root") from error
    capture = capture_file_artifact(source, label="Pure-LLM source STEP")
    if capture.sha256 != str(candidate["step_sha256"]):
      raise ValueError("Pure-LLM source STEP bytes differ")
    local = load_step_shape(source)
    pose = poses[role_id]
    transform = Transform(rotation=pose[:3, :3], translation=pose[:3, 3])
    world = transform_shape(local, transform)
    local_shapes[role_id], world_shapes[role_id] = local, world
    assembly.add(local, name=role_id, loc=_location(pose))
    components.append({
        "role_id": role_id,
        "candidate_id": str(candidate["candidate_id"]),
        "source_step_path": str(candidate["step_path"]),
        "source_step_sha256": capture.sha256,
        "world_pose_row_major": [float(v) for v in pose.reshape(-1)],
    })

  collision_rows = []
  roles = sorted(world_shapes)
  for index, role_a in enumerate(roles):
    for role_b in roles[index + 1:]:
      aabb_upper = _bbox_overlap_upper(world_shapes[role_a], world_shapes[role_b])
      common_volume = 0.0 if aabb_upper <= COLLISION_VOLUME_TOLERANCE_MM3 else max(
          0.0, float(shape_volume(boolean_intersection(
              world_shapes[role_a], world_shapes[role_b],
          )))
      )
      collision_rows.append({
          "role_a": role_a, "role_b": role_b,
          "aabb_intersection_upper_mm3": aabb_upper,
          "whole_solid_common_volume_mm3": common_volume,
          "positive_volume_collision": (
              common_volume > COLLISION_VOLUME_TOLERANCE_MM3
          ),
      })
  raw_collision_free = not any(
      row["positive_volume_collision"] for row in collision_rows
  )

  programs = {str(row["edge_id"]): row for row in prediction["edge_programs"]}
  connections = []
  for edge in query["functional_edges"]:
    edge_id = str(edge["edge_id"])
    program = programs[edge_id]
    role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
    primitive_a, primitive_b = (
        int(program["interface_primitive_a"]),
        int(program["interface_primitive_b"]),
    )
    left = resolve_primitive_v2(
        candidate=candidates[role_a], primitive_ordinal=primitive_a,
        dataset_root=root,
    )
    right = resolve_primitive_v2(
        candidate=candidates[role_b], primitive_ordinal=primitive_b,
        dataset_root=root,
    )
    observation = _interface_observation(
        left=left, right=right, pose_a=poses[role_a], pose_b=poses[role_b],
        world_shape_a=world_shapes[role_a], world_shape_b=world_shapes[role_b],
    )
    mobility_correct = str(program["mobility"]) == str(edge["requested_mobility"])
    connections.append({
        "edge_id": edge_id, "role_a": role_a, "role_b": role_b,
        "instruction": edge.get("instruction"),
        "requested_mobility": edge.get("requested_mobility"),
        "predicted_mobility": str(program["mobility"]),
        "support_family": observation["support_family"],
        "interface_primitive_a": primitive_a,
        "interface_primitive_b": primitive_b,
        "selected_observation": observation,
        "mobility_correct": mobility_correct,
        "connection_satisfied": (
            observation.get("interface_satisfied") is True and mobility_correct
        ),
    })
  all_connections_satisfied = all(
      row["connection_satisfied"] for row in connections
  )
  kernel_feasible = raw_collision_free and all_connections_satisfied

  output_step_path = Path(output_step).resolve()
  output_manifest_path = Path(output_manifest).resolve()
  output_step_path.parent.mkdir(parents=True, exist_ok=True)
  output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
  token = uuid.uuid4().hex
  temporary_step = output_step_path.with_name(
      f".{output_step_path.stem}.{token}.tmp{output_step_path.suffix}"
  )
  try:
    assembly.export(
        str(temporary_step), exportType="STEP", mode="default", unit="MM",
        outputUnit="MM",
    )
    inspection = inspect_assembly_step_v1(temporary_step)
    if (
        inspection["root_count"] != 1
        or inspection["root_is_assembly"] is not True
        or inspection["component_names"] != roles
    ):
      raise ValueError("Pure-LLM exported STEP product structure differs")
    manifest = {
        "schema_version": "linkcad_assembly_manifest.v1",
        "artifact_role": "prediction",
        "method_id": "pure_llm_direct_pose",
        "query_id": query_id,
        "root_assembly_name": f"pure_llm_{query_id}",
        "unit": "mm", "anchor_role": min(roles),
        "components": components, "connections": connections,
        "verification": {
            "global_execution_status": (
                "accepted_exactly_as_predicted" if kernel_feasible
                else "rejected_exactly_as_predicted"
            ),
            "raw_collision_free": raw_collision_free,
            "all_connections_satisfied": all_connections_satisfied,
            "global_assembly_conditionally_feasible": kernel_feasible,
            "collision_pairs": collision_rows,
            "step_reimport": inspection,
            "pose_search_or_repair_used": False,
        },
        "assembly_step_sha256": inspection["step_sha256"],
      }
    manifest["manifest_payload_sha256"] = _json_sha256(manifest)
    temporary_manifest = output_manifest_path.with_suffix(
        output_manifest_path.suffix + ".tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary_step.replace(output_step_path)
    temporary_manifest.replace(output_manifest_path)
  finally:
    temporary_step.unlink(missing_ok=True)
  return {
      "schema_version": SCHEMA_VERSION,
      "query_id": query_id,
      "status": "exported_and_scored_without_repair",
      "raw_collision_free": raw_collision_free,
      "all_connections_satisfied": all_connections_satisfied,
      "kernel_feasible": kernel_feasible,
      "positive_collision_pair_count": sum(
          row["positive_volume_collision"] for row in collision_rows
      ),
      "output_step": str(output_step_path),
      "output_step_sha256": _sha256(output_step_path),
      "output_manifest": str(output_manifest_path),
      "output_manifest_sha256": _sha256(output_manifest_path),
  }


__all__ = [
    "COLLISION_VOLUME_TOLERANCE_MM3", "INTERFACE_ANGLE_TOLERANCE_DEG",
    "INTERFACE_DISTANCE_TOLERANCE_MM", "SCHEMA_VERSION",
    "execute_and_export_pure_llm_v1",
]
