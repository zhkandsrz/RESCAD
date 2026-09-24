"""Contact-complete target-free execution for LinkCAD interface programs.

V2 certified primitive alignment and absence of positive-volume collision.  It
could therefore accept an axially aligned part that remained several
millimetres away from its intended neighbour.  V3 keeps the same public
program inputs, but requires every accepted pose to be both collision-free and
in whole-solid contact.  For axial programs it finds the first collision-free
pose on each seating direction by deterministic bisection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .cadquery_backend import (
    boolean_intersection,
    shape_minimum_distance,
    shape_volume,
    transform_shape,
)
from .domain_types import Transform
from .linkcad_kinematic_execution_v2 import (
    PREDICTION_SCHEMA_VERSION,
    PrimitiveFrameV2,
    ResolvedPrimitiveV2,
    _axial_clearance_poses,
    _bbox_overlap_upper,
    _pose_candidates,
    _support_compatible,
    resolve_primitive_v2,
)


EXECUTION_SCHEMA_VERSION = "linkcad_kinematic_pair_execution.v3"
SUBSET_SCHEMA_VERSION = "linkcad_kinematic_execution_subset.v3"


def _sha(value: Any) -> str:
  raw = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(raw).hexdigest()


def _pose_matrix(pose: Transform) -> list[float]:
  return [
      float(value)
      for value in np.block([
          [pose.rotation, pose.translation[:, None]],
          [np.asarray([[0.0, 0.0, 0.0, 1.0]])],
      ]).reshape(-1)
  ]


def _resolve_primitive_contact_frame_v3(
    *, candidate: Mapping[str, Any], primitive_ordinal: int,
    dataset_root: str | Path,
) -> ResolvedPrimitiveV2:
  """Use trimmed-face centres instead of arbitrary support-plane origins."""

  resolved = resolve_primitive_v2(
      candidate=candidate, primitive_ordinal=primitive_ordinal,
      dataset_root=dataset_root,
  )
  if resolved.primitive_kind != "face":
    return resolved
  raw_faces = list(resolved.shape.Faces())
  frames = []
  for frame in resolved.frames:
    if frame.geometry_type != "PLANE":
      frames.append(frame)
      continue
    center = np.asarray(
        raw_faces[frame.raw_index].Center().toTuple(), dtype=float
    )
    frames.append(PrimitiveFrameV2(
        primitive_kind=frame.primitive_kind,
        geometry_type=frame.geometry_type,
        raw_index=frame.raw_index,
        origin_local_mm=tuple(float(value) for value in center),
        rotation_local=frame.rotation_local,
    ))
  return ResolvedPrimitiveV2(
      capture=resolved.capture, shape=resolved.shape,
      primitive_ordinal=resolved.primitive_ordinal,
      primitive_kind=resolved.primitive_kind,
      member_raw_indices=resolved.member_raw_indices,
      frames=tuple(frames),
  )


def _measure_pose(
    left_shape: Any, right_shape: Any, pose: Transform,
) -> tuple[Any, float, float, str, float]:
  moved = transform_shape(right_shape, pose)
  aabb_upper = _bbox_overlap_upper(left_shape, moved)
  if aabb_upper <= 1e-7:
    common_volume = 0.0
    volume_status = "certified_by_aabb_upper_bound"
  else:
    common_volume = max(0.0, float(shape_volume(
        boolean_intersection(left_shape, moved)
    )))
    volume_status = "exact_occ_boolean"
  distance = shape_minimum_distance(left_shape, moved)
  return moved, aabb_upper, common_volume, volume_status, distance


def _interpolate_pose(start: Transform, end: Transform, alpha: float) -> Transform:
  if not np.allclose(start.rotation, end.rotation, atol=1e-10):
    raise ValueError("LinkCAD axial seating rotation differs")
  return Transform(
      rotation=start.rotation,
      translation=(
          start.translation
          + float(alpha) * (end.translation - start.translation)
      ),
  )


def _contact_pose_candidates(
    *, left_shape: Any, right_shape: Any, base_pose: Transform,
    support_family: str, target_axis: np.ndarray,
    collision_tolerance_mm3: float, contact_tolerance_mm: float,
) -> tuple[tuple[Transform, float], ...]:
  """Return collision-free poses that remain in contact with the parent."""

  _, _, base_common, _, base_distance = _measure_pose(
      left_shape, right_shape, base_pose
  )
  if (
      base_common <= collision_tolerance_mm3
      and base_distance <= contact_tolerance_mm
  ):
    return ((base_pose, 0.0),)
  if support_family != "axial_support" or base_common <= collision_tolerance_mm3:
    return ()

  seated = []
  endpoints = _axial_clearance_poses(
      left_shape=left_shape, right_shape=right_shape,
      base_pose=base_pose, target_axis=target_axis,
  )
  for endpoint, endpoint_offset in endpoints:
    if abs(endpoint_offset) <= 1e-12:
      continue
    _, _, endpoint_common, _, _ = _measure_pose(
        left_shape, right_shape, endpoint
    )
    if endpoint_common > collision_tolerance_mm3:
      continue
    colliding, clear = 0.0, 1.0
    # Twenty-four deterministic steps localize the seating boundary to well
    # below one micrometre for the candidate extents used by LinkCAD.
    for _ in range(24):
      middle = 0.5 * (colliding + clear)
      pose = _interpolate_pose(base_pose, endpoint, middle)
      _, _, common, _, _ = _measure_pose(left_shape, right_shape, pose)
      if common > collision_tolerance_mm3:
        colliding = middle
      else:
        clear = middle
    pose = _interpolate_pose(base_pose, endpoint, clear)
    _, _, common, _, distance = _measure_pose(left_shape, right_shape, pose)
    if (
        common <= collision_tolerance_mm3
        and distance <= contact_tolerance_mm
    ):
      seated.append((pose, float(endpoint_offset * clear)))
  return tuple(seated)


def execute_kinematic_pair_v3(
    *, query_id: str, edge_id: str, hypothesis_rank: int,
    candidate_a: Mapping[str, Any], candidate_b: Mapping[str, Any],
    primitive_a: int, primitive_b: int,
    support_family: str, mobility: str, dataset_root: str | Path,
    maximum_pose_candidates: int = 32,
    enable_axial_seating: bool = True,
    maximum_accepted_poses: int = 1,
    collision_tolerance_mm3: float = 1e-7,
    contact_tolerance_mm: float = 0.1,
) -> dict[str, Any]:
  """Execute one interface program and require physical contact."""

  if min(maximum_pose_candidates, maximum_accepted_poses) < 1:
    raise ValueError("LinkCAD V3 pose budget must be positive")
  if collision_tolerance_mm3 < 0.0 or contact_tolerance_mm < 0.0:
    raise ValueError("LinkCAD V3 geometry tolerance differs")
  started = time.monotonic()
  base = {
      "schema_version": EXECUTION_SCHEMA_VERSION,
      "scope": "pairwise_kinematic_contact_and_collision_feasibility",
      "private_targets_opened": False,
      "query_id": query_id, "edge_id": edge_id,
      "hypothesis_rank": hypothesis_rank,
      "candidate_id_a": candidate_a["candidate_id"],
      "candidate_id_b": candidate_b["candidate_id"],
      "primitive_a": primitive_a, "primitive_b": primitive_b,
      "support_family": support_family, "mobility": mobility,
      "axial_seating_enabled": bool(enable_axial_seating),
      "maximum_accepted_poses": int(maximum_accepted_poses),
      "collision_tolerance_mm3": float(collision_tolerance_mm3),
      "contact_tolerance_mm": float(contact_tolerance_mm),
  }
  observations = []
  try:
    left = _resolve_primitive_contact_frame_v3(
        candidate=candidate_a, primitive_ordinal=primitive_a,
        dataset_root=dataset_root,
    )
    right = _resolve_primitive_contact_frame_v3(
        candidate=candidate_b, primitive_ordinal=primitive_b,
        dataset_root=dataset_root,
    )
    compatible_pairs = [
        (left_frame, right_frame)
        for left_frame in left.frames
        for right_frame in right.frames
        if _support_compatible(support_family, left_frame, right_frame)
    ]
    compatible_count = len(compatible_pairs)
    accepted_poses = []
    pose_count = 0
    base_poses_by_pair = [
        _pose_candidates(support_family, left_frame, right_frame)
        for left_frame, right_frame in compatible_pairs
    ]
    pose_schedule = [
        (left_frame, right_frame, poses[pose_ordinal])
        for pose_ordinal in range(max(
            (len(poses) for poses in base_poses_by_pair), default=0
        ))
        for (left_frame, right_frame), poses in zip(
            compatible_pairs, base_poses_by_pair, strict=True
        )
        if pose_ordinal < len(poses)
    ]
    # Interleave equivalent primitive representatives before spending the
    # budget on additional roll choices of the first representative.  This is
    # crucial when a symmetry orbit contains several physically distinct
    # ports on the same part.
    for left_frame, right_frame, base_pose in pose_schedule:
          pose_rows = (
              _contact_pose_candidates(
                  left_shape=left.shape, right_shape=right.shape,
                  base_pose=base_pose, support_family=support_family,
                  target_axis=left_frame.primary,
                  collision_tolerance_mm3=collision_tolerance_mm3,
                  contact_tolerance_mm=contact_tolerance_mm,
              )
              if enable_axial_seating or support_family != "axial_support"
              else ((base_pose, 0.0),)
          )
          for pose, axial_offset in pose_rows:
            if pose_count >= maximum_pose_candidates:
              break
            _, aabb_upper, common, volume_status, distance = _measure_pose(
                left.shape, right.shape, pose
            )
            observation = {
                "pose_rank": pose_count,
                "raw_primitive_a": left_frame.raw_index,
                "raw_primitive_b": right_frame.raw_index,
                "child_world_row_major": _pose_matrix(pose),
                "aabb_intersection_upper_mm3": aabb_upper,
                "whole_solid_common_volume_mm3": common,
                "common_volume_status": volume_status,
                "whole_solid_minimum_distance_mm": distance,
                "contact_complete": distance <= contact_tolerance_mm,
                "axial_offset_mm": axial_offset,
            }
            observations.append(observation)
            pose_count += 1
            if (
                common <= collision_tolerance_mm3
                and distance <= contact_tolerance_mm
            ):
              accepted_poses.append(observation)
              if len(accepted_poses) >= maximum_accepted_poses:
                break
          if (
              len(accepted_poses) >= maximum_accepted_poses
              or pose_count >= maximum_pose_candidates
          ):
            break
    if compatible_count == 0:
      status = "unsupported_support_primitive_pair"
    elif accepted_poses:
      status = "accepted_contact_complete"
    else:
      status = "rejected_no_collision_free_contact_pose"
    result = {
        **base, "status": status,
        "conditional_kinematic_feasibility_accepted": bool(accepted_poses),
        "contact_complete": bool(accepted_poses),
        "compatible_representative_pair_count": compatible_count,
        "evaluated_pose_count": len(observations),
        "maximum_pose_candidates": maximum_pose_candidates,
        "selected_observation": accepted_poses[0] if accepted_poses else None,
        "accepted_pose_count": len(accepted_poses),
        "accepted_observations": accepted_poses,
        "observations": observations,
        "manifold_fallback": None,
        "elapsed_seconds": time.monotonic() - started,
    }
  except Exception as error:
    result = {
        **base, "status": "pre_execution_error",
        "conditional_kinematic_feasibility_accepted": False,
        "contact_complete": False,
        "compatible_representative_pair_count": 0,
        "evaluated_pose_count": len(observations),
        "maximum_pose_candidates": maximum_pose_candidates,
        "selected_observation": None,
        "accepted_pose_count": 0, "accepted_observations": [],
        "observations": observations,
        "failure_reason": f"{type(error).__name__}:{error}",
        "elapsed_seconds": time.monotonic() - started,
    }
  result["result_payload_sha256"] = _sha(result)
  return result


def execute_public_prediction_subset_v3(
    *, public: Mapping[str, Any], predictions: Mapping[str, Any],
    dataset_root: str | Path, query_limit: int,
    hypotheses_per_query: int = 1, alternatives_per_edge: int = 3,
    maximum_attempts_per_query: int = 25,
    maximum_accepted_poses_per_edge: int = 8,
    contact_tolerance_mm: float = 0.1,
) -> dict[str, Any]:
  """Run the contact-complete executor on a public prediction subset."""

  if (
      predictions.get("schema_version") not in {
          PREDICTION_SCHEMA_VERSION, "linkcad_symbolic_predictions.v1",
          "linkcad_public_primitive_predictions.v2",
      }
      or predictions.get("private_targets_opened") is not False
      or min(query_limit, hypotheses_per_query, alternatives_per_edge,
             maximum_attempts_per_query,
             maximum_accepted_poses_per_edge) < 1
  ):
    raise ValueError("LinkCAD V3 exact subset scope differs")
  query_by_id = {row["query_id"]: row for row in public["queries"]}
  results = []
  for prediction_row in sorted(
      predictions["rows"], key=lambda row: row["query_id"]
  )[:query_limit]:
    query = query_by_id[prediction_row["query_id"]]
    edge_by_id = {row["edge_id"]: row for row in query["functional_edges"]}
    candidates_by_role = {
        role_id: {row["candidate_id"]: row for row in rows}
        for role_id, rows in query["candidate_sets"].items()
    }
    attempts = 0
    for hypothesis in prediction_row["hypotheses"][:hypotheses_per_query]:
      for edge_program in hypothesis["edge_programs"]:
        if attempts >= maximum_attempts_per_query:
          break
        edge = edge_by_id[edge_program["edge_id"]]
        candidate_a = candidates_by_role[edge["role_a"]][
            hypothesis["candidate_by_role"][edge["role_a"]]
        ]
        candidate_b = candidates_by_role[edge["role_b"]][
            hypothesis["candidate_by_role"][edge["role_b"]]
        ]
        alternatives = edge_program.get("primitive_alternatives") or ({
            "rank": 0,
            "interface_primitive_a": edge_program["interface_primitive_a"],
            "interface_primitive_b": edge_program["interface_primitive_b"],
            "model_score": None,
        },)
        for alternative in alternatives[:alternatives_per_edge]:
          if attempts >= maximum_attempts_per_query:
            break
          row = execute_kinematic_pair_v3(
              query_id=prediction_row["query_id"],
              edge_id=edge_program["edge_id"],
              hypothesis_rank=int(hypothesis["rank"]),
              candidate_a=candidate_a, candidate_b=candidate_b,
              primitive_a=int(alternative["interface_primitive_a"]),
              primitive_b=int(alternative["interface_primitive_b"]),
              support_family=str(edge_program["support_family"]),
              mobility=str(edge_program["mobility"]),
              dataset_root=dataset_root,
              maximum_accepted_poses=maximum_accepted_poses_per_edge,
              contact_tolerance_mm=contact_tolerance_mm,
          )
          row["primitive_alternative_rank"] = int(alternative["rank"])
          row["primitive_model_score"] = alternative["model_score"]
          row.pop("result_payload_sha256", None)
          row["result_payload_sha256"] = _sha(row)
          results.append(row)
          attempts += 1
          if row["conditional_kinematic_feasibility_accepted"] is True:
            break
  accepted_keys = {
      (row["query_id"], row["hypothesis_rank"], row["edge_id"])
      for row in results
      if row["conditional_kinematic_feasibility_accepted"] is True
  }
  status_counts: dict[str, int] = {}
  for row in results:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
  payload = {
      "schema_version": SUBSET_SCHEMA_VERSION,
      "scope": "development_public_prediction_contact_complete_subset",
      "private_targets_opened": False,
      "prediction_payload_sha256": predictions["prediction_payload_sha256"],
      "query_limit": query_limit,
      "hypotheses_per_query": hypotheses_per_query,
      "alternatives_per_edge": alternatives_per_edge,
      "maximum_attempts_per_query": maximum_attempts_per_query,
      "maximum_accepted_poses_per_edge": maximum_accepted_poses_per_edge,
      "contact_tolerance_mm": float(contact_tolerance_mm),
      "accepted_edge_count": len(accepted_keys),
      "execution_attempt_count": len(results),
      "status_counts": dict(sorted(status_counts.items())),
      "rows": results,
  }
  payload["subset_payload_sha256"] = _sha(payload)
  return payload


__all__ = [
    "EXECUTION_SCHEMA_VERSION", "SUBSET_SCHEMA_VERSION",
    "execute_kinematic_pair_v3", "execute_public_prediction_subset_v3",
]
