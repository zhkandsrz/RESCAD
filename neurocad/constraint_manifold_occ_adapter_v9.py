"""Decisive-interference short-circuit for exact OCC candidate batches."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from .cadquery_backend import (
    boolean_intersection, load_step_shape, shape_volume,
    source_face_signature_sha256, transform_shape,
)
from .constraint_manifold_occ_adapter_v3 import _transform, _transformed_bbox
from .constraint_manifold_occ_adapter_v8 import OccExactManifoldExecutorV8
from .constraint_manifold_solver_v3 import file_sha256
from .constraint_manifold_solver_v4 import (
    ExactBatchCandidateReceiptV8, OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4,
    issue_exact_batch_candidate_receipt_v8,
)


WORKER_REQUEST_SCHEMA_VERSION_V9 = "constraint_manifold_occ_batch_worker_request.v9"
WORKER_TERMINAL_SCHEMA_VERSION_V9 = "constraint_manifold_occ_batch_worker_terminal.v9"
EXACT_EVALUATION_SCHEMA_VERSION_V9 = (
    "constraint_manifold_decisive_interference_short_circuit.v9"
)


def _exact_metrics_from_prepared_v9(
    *, moved_source_a: Any, source_b: Any, moved_face_a: Any, face_b: Any,
    low_a: np.ndarray, high_a: np.ndarray, corners_a: np.ndarray,
    child_world_row_major: list[float] | tuple[float, ...],
    plane_frame_a: list[list[float]] | tuple[tuple[float, ...], ...],
    plane_origin_a: list[float] | tuple[float, ...],
) -> dict[str, Any]:
  """Evaluate decisive interference first; face distance remains mandatory to accept."""

  world_b = _transform(child_world_row_major)
  low_b, high_b, corners_b = _transformed_bbox(source_b, world_b)
  overlap = np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
  aabb_upper = 0.0 if np.any(overlap <= 0.0) else float(np.prod(overlap))
  threshold = OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.thresholds.whole_common_volume_mm3
  if aabb_upper <= threshold:
    volume = 0.0
    volume_status = "certified_by_aabb_upper_bound"
  else:
    moved_source_b = transform_shape(source_b, world_b)
    volume = max(0.0, float(shape_volume(boolean_intersection(
        moved_source_a, moved_source_b,
    ))))
    volume_status = "exact_occ_boolean"

  result: dict[str, Any] = {
      "common_volume_mm3": volume,
      "aabb_intersection_upper_mm3": aabb_upper,
      "common_volume_status": volume_status,
      "exact_evaluation_schema": EXACT_EVALUATION_SCHEMA_VERSION_V9,
  }
  if volume > threshold:
    result["selected_face_distance_status"] = (
        "not_evaluated_decisive_whole_solid_interference"
    )
  else:
    moved_face_b = transform_shape(face_b, world_b)
    result["distance_mm"] = float(moved_face_a.distance(moved_face_b))
    result["selected_face_distance_status"] = "exact_occ_distance"

  frame = np.asarray(plane_frame_a, dtype=float).reshape(3, 3)
  origin = np.asarray(plane_origin_a, dtype=float).reshape(3)
  local_a = (corners_a - origin) @ frame
  local_b = (corners_b - origin) @ frame
  proj_a = np.min(local_a[:, :2], axis=0), np.max(local_a[:, :2], axis=0)
  proj_b = np.min(local_b[:, :2], axis=0), np.max(local_b[:, :2], axis=0)
  epsilon = 1e-4
  result["projected_aabb_separation_offsets_xy_mm"] = (
      (float(proj_a[1][0] - proj_b[0][0] + epsilon), 0.0),
      (float(proj_a[0][0] - proj_b[1][0] - epsilon), 0.0),
      (0.0, float(proj_a[1][1] - proj_b[0][1] + epsilon)),
      (0.0, float(proj_a[0][1] - proj_b[1][1] - epsilon)),
  )
  return result


def execute_occ_exact_batch_v9(
    arguments: Mapping[str, Any], *, request_payload_sha256: str,
    emit: Callable[[ExactBatchCandidateReceiptV8], None],
) -> dict[str, Any]:
  """Load once, cache endpoint A, and short-circuit decisively interfering rows."""

  if arguments.get("operation") != "candidate_exact_batch":
    raise ValueError("V9 OCC batch operation differs")
  if file_sha256(arguments["step_path_a"]) != arguments["step_sha256_a"] or (
      file_sha256(arguments["step_path_b"]) != arguments["step_sha256_b"]
  ):
    raise ValueError("V9 batch receipt-bound STEP bytes changed")
  source_a = load_step_shape(arguments["step_path_a"])
  source_b = load_step_shape(arguments["step_path_b"])
  faces_a, faces_b = tuple(source_a.Faces()), tuple(source_b.Faces())
  face_a = faces_a[int(arguments["raw_face_index_a"])]
  face_b = faces_b[int(arguments["raw_face_index_b"])]
  if source_face_signature_sha256(face_a) != arguments["face_signature_a"] or (
      source_face_signature_sha256(face_b) != arguments["face_signature_b"]
  ):
    raise ValueError("V9 batch reloaded STEP face signature differs")
  candidates = arguments.get("candidates")
  if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
    raise ValueError("V9 OCC batch candidate cardinality differs")
  batch_nonce = str(arguments.get("batch_nonce", ""))
  world_a = _transform(arguments["world_a_row_major"])
  moved_source_a = transform_shape(source_a, world_a)
  moved_face_a = transform_shape(face_a, world_a)
  low_a, high_a, corners_a = _transformed_bbox(source_a, world_a)
  thresholds = OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.thresholds
  evaluated = 0
  stop_reason = "exhausted"
  for sequence, row in enumerate(candidates):
    if not isinstance(row, Mapping) or int(row.get("rank_zero_based", -1)) != sequence:
      raise ValueError("V9 OCC batch candidate order differs")
    candidate_key = str(row.get("candidate_key", ""))
    call_nonce = str(row.get("call_nonce", ""))
    try:
      result = _exact_metrics_from_prepared_v9(
          moved_source_a=moved_source_a, source_b=source_b,
          moved_face_a=moved_face_a, face_b=face_b,
          low_a=low_a, high_a=high_a, corners_a=corners_a,
          child_world_row_major=row["child_world_row_major"],
          plane_frame_a=arguments["plane_frame_a"],
          plane_origin_a=arguments["plane_origin_a"],
      )
      status = "ok"
    except BaseException as error:
      status = "kernel_error"
      result = {"error_sha256": hashlib.sha256(
          f"{type(error).__name__}:{error}".encode("utf-8")
      ).hexdigest()}
    receipt = issue_exact_batch_candidate_receipt_v8(
        request_payload_sha256=request_payload_sha256, batch_nonce=batch_nonce,
        rank_zero_based=sequence, candidate_key=candidate_key,
        call_nonce=call_nonce, terminal_status=status, result=result,
    )
    emit(receipt)
    evaluated += 1
    if status != "ok":
      stop_reason = "candidate_kernel_error"
      break
    if (
        float(result.get("distance_mm", np.inf))
        <= thresholds.selected_face_clearance_mm
        and float(result["common_volume_mm3"])
        <= thresholds.whole_common_volume_mm3
    ):
      stop_reason = "first_feasible"
      break
  return {
      "stop_reason": stop_reason, "evaluated_candidate_count": evaluated,
      "unassessed_candidate_count": len(candidates) - evaluated,
      # Keep the frozen conservative two-call reservation per candidate even
      # when decisive interference makes the distance observation unnecessary.
      "occ_calls_reserved": 2 * evaluated, "step_load_count": 2,
      "process_launch_count": 1,
  }


class OccExactManifoldExecutorV9(OccExactManifoldExecutorV8):
  """V8 transport with V9 decisive-interference exact evaluation."""

  __slots__ = ()
  WORKER_REQUEST_SCHEMA = WORKER_REQUEST_SCHEMA_VERSION_V9
  WORKER_TERMINAL_SCHEMA = WORKER_TERMINAL_SCHEMA_VERSION_V9
  WORKER_FILENAME = "constraint_manifold_occ_batch_worker_v9.py"


__all__ = [
    "EXACT_EVALUATION_SCHEMA_VERSION_V9",
    "WORKER_REQUEST_SCHEMA_VERSION_V9", "WORKER_TERMINAL_SCHEMA_VERSION_V9",
    "OccExactManifoldExecutorV9", "execute_occ_exact_batch_v9",
]
