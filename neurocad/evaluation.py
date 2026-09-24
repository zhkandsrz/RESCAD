"""Quality evaluation helpers for neuro-symbolic CAD generation runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
import itertools
import math
from typing import Any


def evaluate_generation_quality(result: dict[str, Any] | Any) -> dict[str, Any]:
  """Compute quality metrics from PipelineResult dict/object."""
  data = _to_result_dict(result)

  success = bool(data.get("success", False))
  attempts = data.get("attempts", []) or []
  final_graph = data.get("final_graph") or {}
  final_assembly = data.get("final_assembly") or {}

  graph_instances = final_graph.get("instances") or {}
  graph_constraints = final_graph.get("constraints") or []
  anchor = final_graph.get("anchor")

  part_names = set(graph_instances.keys())
  part_count = len(part_names)

  covered_parts = set()
  if isinstance(anchor, str) and anchor in part_names:
    covered_parts.add(anchor)
  constraint_type_counter: Counter[str] = Counter()
  for constraint in graph_constraints:
    if not isinstance(constraint, dict):
      continue
    part_a = constraint.get("part_a")
    part_b = constraint.get("part_b")
    if part_a in part_names:
      covered_parts.add(part_a)
    if part_b in part_names:
      covered_parts.add(part_b)
    ctype = str(constraint.get("type", "")).strip().lower()
    if ctype:
      constraint_type_counter[ctype] += 1

  coverage_ratio = (
      1.0 if part_count == 0 else float(len(covered_parts)) / float(part_count)
  )
  constraint_count = len(graph_constraints)
  constraint_density = (
      0.0 if part_count == 0 else float(constraint_count) / float(part_count)
  )

  placed_ratio = _placed_ratio(final_assembly)
  feedback_stats = _feedback_stats(data)
  connectivity_stats = _connectivity_stats(part_names, anchor, graph_constraints)
  frame_stats = _frame_grounding_stats(data)

  score = _score_quality(
      success=success,
      coverage_ratio=coverage_ratio,
      placed_ratio=placed_ratio,
      collision_count=feedback_stats["collision_count"],
      solve_error_count=feedback_stats["solve_error_count"],
      unplaced_count=feedback_stats["unplaced_count"],
      planner_error_count=feedback_stats["planner_error_count"],
      disconnected_count=connectivity_stats["disconnected_count"],
  )
  tolerated_collision_count = _tolerated_collision_count(data)
  tolerated_collision_volume_sum = _tolerated_collision_volume_sum(data)
  strict_success = bool(success) and tolerated_collision_count == 0

  return {
      "score": round(score, 2),
      "success": success,
      "strict_success": strict_success,
      "tolerant_success": success,
      "part_count": part_count,
      "constraint_count": constraint_count,
      "constraint_density": round(constraint_density, 4),
      "coverage_ratio": round(coverage_ratio, 4),
      "placed_ratio": round(placed_ratio, 4),
      "connected_to_anchor_ratio": round(
          connectivity_stats["connected_ratio"], 4
      ),
      "connected_components": connectivity_stats["connected_components"],
      "disconnected_count": connectivity_stats["disconnected_count"],
      "feedback": feedback_stats,
      "tolerated_collision_count": tolerated_collision_count,
      "tolerated_collision_volume_sum": round(
          tolerated_collision_volume_sum, 6
      ),
      "constraint_types": dict(constraint_type_counter),
      "attempt_count": len(attempts),
      "frame_grounding_coverage": round(
          float(frame_stats["frame_grounding_coverage"]),
          4,
      ),
      "direct_frame_solve_rate": round(
          float(frame_stats["direct_frame_solve_rate"]),
          4,
      ),
      "axial_fallback_rate": round(
          float(frame_stats["axial_fallback_rate"]),
          4,
      ),
      "frame_grounding": frame_stats,
  }


def _to_result_dict(result: dict[str, Any] | Any) -> dict[str, Any]:
  if isinstance(result, dict):
    return result
  if hasattr(result, "to_dict") and callable(result.to_dict):
    return result.to_dict()
  if is_dataclass(result):
    return asdict(result)
  raise TypeError("Unsupported result type for evaluation.")


def _benchmark_task(data: dict[str, Any]) -> str:
  task = str(data.get("benchmark_task") or "").strip().lower()
  if task:
    return task
  case = data.get("case")
  if isinstance(case, dict):
    meta = case.get("benchmark_metadata")
    if isinstance(meta, dict):
      task = str(meta.get("case_mode") or "").strip().lower()
      if task:
        return task
    if isinstance(case.get("candidate_parts"), list):
      return "candidate_tray"
    if isinstance(case.get("parts"), list) and case.get("parts"):
      return "oracle"
  return ""


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
  value = metrics.get(key)
  if isinstance(value, (int, float)):
    return float(value)
  return None


def _metric_int(metrics: dict[str, Any], key: str) -> int | None:
  value = metrics.get(key)
  if isinstance(value, int):
    return int(value)
  if isinstance(value, float):
    return int(value)
  return None


def assess_retrieval_confidence(result: dict[str, Any] | Any) -> dict[str, Any]:
  data = _to_result_dict(result)
  retrieval = data.get("retrieval")
  if not isinstance(retrieval, dict):
    return {
        "available": False,
        "low_confidence": False,
        "reasons": [],
    }
  notes = [str(item) for item in (retrieval.get("notes") or []) if str(item).strip()]
  retrieval_metrics = data.get("retrieval_metrics") or {}
  prompt_concepts = [
      str(item).strip().lower()
      for item in (retrieval.get("prompt_concepts") or [])
      if str(item).strip()
  ]
  semantic_score = retrieval.get("semantic_score")
  semantic_score_value = (
      float(semantic_score) if isinstance(semantic_score, (int, float)) else None
  )
  fallback = any("retrieval_assignment_fallback=true" in note for note in notes)
  cross_assembly = any("retrieval_selected_cross_assembly=true" in note for note in notes)
  missing_concepts: list[str] = []
  anchor_matches: list[str] = []
  anchor_missing: list[str] = []
  for note in notes:
    if note.startswith("retrieval_missing_concepts="):
      missing_concepts = [
          item.strip()
          for item in note.split("=", 1)[1].split(",")
          if item.strip()
      ]
    elif note.startswith("retrieval_anchor_match="):
      anchor_matches = [
          item.strip()
          for item in note.split("=", 1)[1].split(",")
          if item.strip()
      ]
    elif note.startswith("retrieval_anchor_missing="):
      anchor_missing = [
          item.strip()
          for item in note.split("=", 1)[1].split(",")
          if item.strip()
      ]
  retrieval_recall = retrieval_metrics.get("retrieval_recall")
  retrieval_precision = retrieval_metrics.get("retrieval_precision")
  retrieval_recall_value = _metric_float(retrieval_metrics, "retrieval_recall")
  retrieval_precision_value = _metric_float(
      retrieval_metrics, "retrieval_precision"
  )
  reference_part_count = _metric_int(retrieval_metrics, "reference_part_count")
  retrieved_part_count = _metric_int(retrieval_metrics, "retrieved_part_count")
  task_mode = _benchmark_task(data)
  reasons: list[str] = []
  if fallback:
    reasons.append("fallback_assignment")
  if prompt_concepts and len(missing_concepts) >= len(prompt_concepts):
    reasons.append("all_prompt_concepts_missing")
  if cross_assembly and not anchor_matches:
    reasons.append("cross_assembly_without_anchor_match")
  if (
      semantic_score_value is not None
      and semantic_score_value < max(10.0, 5.0 * max(1, len(prompt_concepts)))
  ):
    reasons.append("low_semantic_score")
  if (
      isinstance(retrieval_recall, (int, float))
      and isinstance(retrieval_precision, (int, float))
      and float(retrieval_recall) == 0.0
      and float(retrieval_precision) == 0.0
      and fallback
  ):
    reasons.append("benchmark_zero_match_under_fallback")
  if task_mode == "candidate_tray" and reference_part_count is not None:
    if (
        retrieval_recall_value is not None
        and reference_part_count >= 2
        and retrieval_recall_value < 0.999
    ):
      reasons.append("candidate_tray_missing_reference_parts")
    if (
        retrieved_part_count is not None
        and reference_part_count >= 2
        and retrieved_part_count < reference_part_count
    ):
      reasons.append("candidate_tray_incomplete_part_set")
  elif task_mode == "retrieval" and retrieval_recall_value is not None:
    if retrieval_recall_value < 0.5:
      reasons.append("low_retrieval_recall")

  low_confidence = False
  if task_mode == "candidate_tray" and any(
      reason.startswith("candidate_tray_") for reason in reasons
  ):
    low_confidence = True
  elif task_mode == "retrieval" and "low_retrieval_recall" in reasons:
    low_confidence = True
  if "fallback_assignment" in reasons:
    weak_numeric_match = (
        retrieval_recall_value is not None
        and retrieval_precision_value is not None
        and float(retrieval_recall_value) <= 0.25
        and float(retrieval_precision_value) <= 0.5
    )
    low_confidence = bool(low_confidence or
        "benchmark_zero_match_under_fallback" in reasons
        or "cross_assembly_without_anchor_match" in reasons
        or (
            "all_prompt_concepts_missing" in reasons
            and not anchor_matches
            and weak_numeric_match
        )
    )
  return {
      "available": True,
      "low_confidence": bool(low_confidence),
      "reasons": reasons,
      "fallback_assignment": fallback,
      "cross_assembly": cross_assembly,
      "missing_concepts": missing_concepts,
      "anchor_matches": anchor_matches,
      "anchor_missing": anchor_missing,
      "semantic_score": semantic_score_value,
      "reference_part_count": reference_part_count,
      "retrieved_part_count": retrieved_part_count,
      "retrieval_recall": retrieval_recall_value,
      "retrieval_precision": retrieval_precision_value,
  }


def assess_contact_coverage(result: dict[str, Any] | Any) -> dict[str, Any]:
  data = _to_result_dict(result)
  metrics = data.get("contact_metrics")
  if not isinstance(metrics, dict):
    return {
        "available": False,
        "valid": True,
        "reasons": [],
    }
  task_mode = _benchmark_task(data)
  expected = _metric_int(metrics, "expected_contact_pairs") or 0
  recall = _metric_float(metrics, "contact_recall")
  precision = _metric_float(metrics, "contact_precision")
  reasons: list[str] = []
  if task_mode in {"candidate_tray", "oracle"} and expected >= 2:
    if recall is not None and recall < 0.5:
      reasons.append("low_reference_contact_recall")
    predicted = _metric_int(metrics, "predicted_pairs")
    if predicted is not None and predicted <= 0:
      reasons.append("no_predicted_contact_edges")
  return {
      "available": True,
      "valid": not reasons,
      "reasons": reasons,
      "expected_contact_pairs": expected,
      "contact_recall": recall,
      "contact_precision": precision,
  }


def assess_exact_contact_validation(result: dict[str, Any] | Any) -> dict[str, Any]:
  data = _to_result_dict(result)
  exact = data.get("exact_contact_validation")
  if not isinstance(exact, dict):
    return {
        "available": False,
        "valid": True,
        "reasons": [],
    }
  reasons: list[str] = []
  if not bool(exact.get("valid", True)):
    reasons.append("geometric_contact_distance_failure")
  recall = _metric_float(exact, "exact_contact_recall")
  min_recall = _metric_float(exact, "min_contact_recall")
  if recall is not None and min_recall is not None and recall < min_recall:
    reasons.append("low_exact_contact_recall")
  if exact.get("errors"):
    reasons.append("exact_contact_distance_errors")
  return {
      "available": True,
      "valid": not reasons,
      "reasons": reasons,
      "checked_pairs": _metric_int(exact, "checked_pairs"),
      "expected_pairs": _metric_int(exact, "expected_pairs"),
      "exact_contact_recall": recall,
      "max_contact_distance": _metric_float(exact, "max_contact_distance"),
      "mean_contact_distance": _metric_float(exact, "mean_contact_distance"),
      "violations": exact.get("violations", []),
      "errors": exact.get("errors", []),
  }


def assess_axial_mate_contacts(
    result: dict[str, Any] | Any,
    gap_tolerance: float = 0.75,
) -> dict[str, Any]:
  data = _to_result_dict(result)
  final_graph = data.get("final_graph") or {}
  final_assembly = data.get("final_assembly") or {}
  constraints = final_graph.get("constraints")
  instances = final_assembly.get("instances")
  if not isinstance(constraints, list) or not isinstance(instances, dict):
    return {"available": False, "valid": True, "checked_pairs": 0, "violations": []}

  pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for constraint in constraints:
    if not isinstance(constraint, dict):
      continue
    part_a = str(constraint.get("part_a") or "").strip()
    part_b = str(constraint.get("part_b") or "").strip()
    if not part_a or not part_b or part_a == part_b:
      continue
    key = tuple(sorted((part_a, part_b)))
    pair_groups.setdefault(key, []).append(constraint)

  checked_pairs = 0
  violations: list[dict[str, Any]] = []
  for pair_key, group in pair_groups.items():
    has_secondary_seat = any(
        bool((item.get("metadata") or {}).get("secondary_seat", False))
        for item in group
        if isinstance(item, dict)
    )
    if has_secondary_seat:
      continue
    primary = None
    for item in group:
      if not isinstance(item, dict):
        continue
      if str(item.get("type", "")).strip().lower() != "concentric":
        continue
      meta = item.get("metadata") or {}
      hint = str(meta.get("relation_hint", "")).strip().lower()
      mode = str(meta.get("stack_mode", "")).strip().lower()
      if hint in {"coaxial", "fasten"} or mode in {"hole_on_pin", "pin_into_hole"}:
        primary = item
        break
    if primary is None:
      continue
    part_a = instances.get(primary.get("part_a"))
    part_b = instances.get(primary.get("part_b"))
    if not isinstance(part_a, dict) or not isinstance(part_b, dict):
      continue
    axis = _world_socket_direction(part_a, str(primary.get("socket_a") or ""))
    if axis is None:
      continue
    interval_a = _projected_bbox_interval(part_a, axis)
    interval_b = _projected_bbox_interval(part_b, axis)
    if interval_a is None or interval_b is None:
      continue
    checked_pairs += 1
    gap = max(
        float(interval_a[0] - interval_b[1]),
        float(interval_b[0] - interval_a[1]),
        0.0,
    )
    overlap_ok = _perpendicular_bbox_overlap(part_a, part_b, axis)
    if gap > float(gap_tolerance) or not overlap_ok:
      violations.append(
          {
              "part_a": str(primary.get("part_a")),
              "part_b": str(primary.get("part_b")),
              "label": str(primary.get("label") or ""),
              "gap": round(float(gap), 6),
              "perpendicular_overlap": bool(overlap_ok),
          }
      )
  return {
      "available": True,
      "valid": len(violations) == 0,
      "checked_pairs": checked_pairs,
      "violations": violations,
  }


def derive_validated_success(
    result: dict[str, Any] | Any,
    tolerated_collision_count: int,
) -> dict[str, Any]:
  data = _to_result_dict(result)
  raw_success = bool(data.get("success", False))
  retrieval_confidence = assess_retrieval_confidence(data)
  mate_contact_validation = assess_axial_mate_contacts(data)
  contact_coverage = assess_contact_coverage(data)
  exact_contact_validation = assess_exact_contact_validation(data)
  validated_success = bool(raw_success)
  rejection_reasons: list[str] = []
  if retrieval_confidence.get("low_confidence"):
    validated_success = False
    rejection_reasons.extend(
        "retrieval:" + str(item)
        for item in (retrieval_confidence.get("reasons") or [])
    )
  if not bool(mate_contact_validation.get("valid", True)):
    validated_success = False
    rejection_reasons.append("mate_contact:underconstrained_stack")
  if not bool(contact_coverage.get("valid", True)):
    validated_success = False
    rejection_reasons.extend(
        "contact_coverage:" + str(item)
        for item in (contact_coverage.get("reasons") or [])
    )
  if not bool(exact_contact_validation.get("valid", True)):
    validated_success = False
    rejection_reasons.extend(
        "exact_contact:" + str(item)
        for item in (exact_contact_validation.get("reasons") or [])
    )
  return {
      "raw_success": raw_success,
      "success": validated_success,
      "tolerant_success": validated_success,
      "strict_success": validated_success and int(tolerated_collision_count) == 0,
      "retrieval_confidence": retrieval_confidence,
      "mate_contact_validation": mate_contact_validation,
      "contact_coverage": contact_coverage,
      "exact_contact_validation": exact_contact_validation,
      "rejection_reasons": rejection_reasons,
  }


def _frame_grounding_stats(result: dict[str, Any]) -> dict[str, Any]:
  final_graph = result.get("final_graph") or {}
  attempts = result.get("attempts") or []
  constraints = final_graph.get("constraints")
  instances = final_graph.get("instances")
  if not isinstance(constraints, list) or not isinstance(instances, dict):
    return {
        "eligible_constraint_count": 0,
        "hinted_constraint_count": 0,
        "frame_grounding_coverage": 0.0,
        "direct_frame_solve_count": 0,
        "direct_frame_solve_rate": 0.0,
        "axial_fallback_count": 0,
        "axial_fallback_rate": 0.0,
    }

  eligible = 0
  hinted = 0
  for constraint in constraints:
    if not isinstance(constraint, dict):
      continue
    ctype = str(constraint.get("type") or "").strip().lower()
    if ctype not in {"concentric", "coincident"}:
      continue
    part_a = str(constraint.get("part_a") or "").strip()
    part_b = str(constraint.get("part_b") or "").strip()
    socket_a = str(constraint.get("socket_a") or "").strip()
    socket_b = str(constraint.get("socket_b") or "").strip()
    if not part_a or not part_b or not socket_a or not socket_b:
      continue
    inst_a = instances.get(part_a)
    inst_b = instances.get(part_b)
    if not isinstance(inst_a, dict) or not isinstance(inst_b, dict):
      continue
    sock_a = (inst_a.get("sockets") or {}).get(socket_a)
    sock_b = (inst_b.get("sockets") or {}).get(socket_b)
    if not isinstance(sock_a, dict) or not isinstance(sock_b, dict):
      continue
    if not (_socket_has_frame_data(sock_a) and _socket_has_frame_data(sock_b)):
      continue
    eligible += 1
    meta = constraint.get("metadata") or {}
    if (
        isinstance(meta, dict)
        and (
            str(meta.get("source_frame_hint") or "").strip()
            or str(meta.get("target_frame_hint") or "").strip()
            or str(meta.get("alignment_mode") or "").strip()
        )
    ):
      hinted += 1

  direct_frame_solves = 0
  axial_fallbacks = 0
  for attempt in attempts:
    if not isinstance(attempt, dict):
      continue
    for log in attempt.get("solver_logs", []) or []:
      text = str(log).strip()
      if text.startswith("frame_align ") or text.startswith("interface_frame_align "):
        direct_frame_solves += 1
      elif text.startswith("axial_fallback "):
        axial_fallbacks += 1

  denom = max(1, eligible)
  return {
      "eligible_constraint_count": int(eligible),
      "hinted_constraint_count": int(hinted),
      "frame_grounding_coverage": min(1.0, float(hinted) / float(denom)),
      "direct_frame_solve_count": int(direct_frame_solves),
      "direct_frame_solve_rate": min(
          1.0,
          float(direct_frame_solves) / float(denom),
      ),
      "axial_fallback_count": int(axial_fallbacks),
      "axial_fallback_rate": min(
          1.0,
          float(axial_fallbacks) / float(denom),
      ),
  }


def _socket_has_frame_data(socket: dict[str, Any]) -> bool:
  if any(
      isinstance(socket.get(key), list) and len(socket.get(key) or []) == 3
      for key in ("x_axis", "y_axis", "z_axis")
  ):
    return True
  metadata = socket.get("metadata") or {}
  frame_variants = metadata.get("frame_variants") if isinstance(metadata, dict) else None
  return isinstance(frame_variants, dict) and len(frame_variants) > 0


def _placed_ratio(final_assembly: dict[str, Any]) -> float:
  instances = final_assembly.get("instances")
  if not isinstance(instances, dict) or not instances:
    return 0.0
  placed = 0
  for instance in instances.values():
    if not isinstance(instance, dict):
      continue
    if instance.get("transform") is not None:
      placed += 1
  return float(placed) / float(max(1, len(instances)))


def _normalize(vec: list[float]) -> list[float]:
  length = math.sqrt(sum(float(item) * float(item) for item in vec))
  if length <= 1e-12:
    return [0.0, 0.0, 1.0]
  return [float(item) / length for item in vec]


def _apply_rotation(rotation: list[list[float]], vec: list[float]) -> list[float]:
  return [
      float(rotation[0][0]) * vec[0]
      + float(rotation[0][1]) * vec[1]
      + float(rotation[0][2]) * vec[2],
      float(rotation[1][0]) * vec[0]
      + float(rotation[1][1]) * vec[1]
      + float(rotation[1][2]) * vec[2],
      float(rotation[2][0]) * vec[0]
      + float(rotation[2][1]) * vec[1]
      + float(rotation[2][2]) * vec[2],
  ]


def _apply_point(
    rotation: list[list[float]],
    translation: list[float],
    point: list[float],
) -> list[float]:
  rotated = _apply_rotation(rotation, point)
  return [
      rotated[0] + translation[0],
      rotated[1] + translation[1],
      rotated[2] + translation[2],
  ]


def _instance_transform(instance: dict[str, Any]) -> tuple[list[list[float]], list[float]] | None:
  transform = instance.get("transform")
  if not isinstance(transform, dict):
    return None
  rotation = transform.get("rotation")
  translation = transform.get("translation")
  if (
      not isinstance(rotation, list)
      or len(rotation) != 3
      or not isinstance(translation, list)
      or len(translation) != 3
  ):
    return None
  return rotation, [float(item) for item in translation]


def _world_socket_direction(
    instance: dict[str, Any],
    socket_name: str,
) -> list[float] | None:
  sockets = instance.get("sockets")
  transform = _instance_transform(instance)
  if not isinstance(sockets, dict) or transform is None:
    return None
  socket = sockets.get(socket_name)
  if not isinstance(socket, dict):
    return None
  direction = socket.get("axis")
  if direction is None:
    direction = socket.get("normal")
  if not isinstance(direction, list) or len(direction) != 3:
    return None
  rotation, _ = transform
  return _normalize(_apply_rotation(rotation, [float(x) for x in direction]))


def _world_bbox_corners(instance: dict[str, Any]) -> list[list[float]] | None:
  transform = _instance_transform(instance)
  bbox_min = instance.get("local_bbox_min")
  bbox_max = instance.get("local_bbox_max")
  if (
      transform is None
      or not isinstance(bbox_min, list)
      or not isinstance(bbox_max, list)
      or len(bbox_min) != 3
      or len(bbox_max) != 3
  ):
    return None
  rotation, translation = transform
  mins = [float(x) for x in bbox_min]
  maxs = [float(x) for x in bbox_max]
  corners: list[list[float]] = []
  for xyz in itertools.product((mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])):
    corners.append(_apply_point(rotation, translation, [float(x) for x in xyz]))
  return corners


def _projected_bbox_interval(
    instance: dict[str, Any],
    axis: list[float],
) -> tuple[float, float] | None:
  corners = _world_bbox_corners(instance)
  if not corners:
    return None
  values = [
      float(corner[0] * axis[0] + corner[1] * axis[1] + corner[2] * axis[2])
      for corner in corners
  ]
  return min(values), max(values)


def _orthogonal_basis(axis: list[float]) -> tuple[list[float], list[float]]:
  axis_n = _normalize(axis)
  trial = [1.0, 0.0, 0.0]
  dot = sum(axis_n[idx] * trial[idx] for idx in range(3))
  if abs(dot) > 0.9:
    trial = [0.0, 1.0, 0.0]
  basis_u = _normalize(
      [
          axis_n[1] * trial[2] - axis_n[2] * trial[1],
          axis_n[2] * trial[0] - axis_n[0] * trial[2],
          axis_n[0] * trial[1] - axis_n[1] * trial[0],
      ]
  )
  basis_v = _normalize(
      [
          axis_n[1] * basis_u[2] - axis_n[2] * basis_u[1],
          axis_n[2] * basis_u[0] - axis_n[0] * basis_u[2],
          axis_n[0] * basis_u[1] - axis_n[1] * basis_u[0],
      ]
  )
  return basis_u, basis_v


def _perpendicular_bbox_overlap(
    part_a: dict[str, Any],
    part_b: dict[str, Any],
    axis: list[float],
) -> bool:
  basis_u, basis_v = _orthogonal_basis(axis)
  intervals = []
  for basis in (basis_u, basis_v):
    interval_a = _projected_bbox_interval(part_a, basis)
    interval_b = _projected_bbox_interval(part_b, basis)
    if interval_a is None or interval_b is None:
      return False
    intervals.append((interval_a, interval_b))
  for interval_a, interval_b in intervals:
    overlap = min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0])
    if overlap < -1e-3:
      return False
  return True


def _feedback_stats(result_data: dict[str, Any]) -> dict[str, Any]:
  attempts = result_data.get("attempts", []) or []
  messages: list[str] = []
  for attempt in attempts:
    if not isinstance(attempt, dict):
      continue
    for message in attempt.get("feedback_messages", []) or []:
      messages.append(str(message))
  for message in result_data.get("final_feedback", []) or []:
    messages.append(str(message))

  collision_count = sum(1 for msg in messages if msg.startswith("collision("))
  solve_error_count = sum(1 for msg in messages if "solve_error=" in msg)
  unplaced_count = sum(1 for msg in messages if "unplaced_part=" in msg)
  planner_error_count = sum(1 for msg in messages if "planner_error=" in msg)
  validation_error_count = sum(1 for msg in messages if "validation_error=" in msg)
  unique_messages = len(set(messages))
  repeat_ratio = 0.0
  if messages:
    repeat_ratio = 1.0 - float(unique_messages) / float(len(messages))

  return {
      "message_count": len(messages),
      "unique_message_count": unique_messages,
      "repeat_ratio": round(repeat_ratio, 4),
      "collision_count": collision_count,
      "solve_error_count": solve_error_count,
      "unplaced_count": unplaced_count,
      "planner_error_count": planner_error_count,
      "validation_error_count": validation_error_count,
  }


def _tolerated_collision_count(result_data: dict[str, Any]) -> int:
  records = result_data.get("final_tolerated_collisions")
  if isinstance(records, list):
    return len(
        [
            item
            for item in records
            if isinstance(item, dict)
            and not _is_intentional_interference_collision(item)
        ]
    )
  total = 0
  for attempt in result_data.get("attempts", []) or []:
    if not isinstance(attempt, dict):
      continue
    total += len(attempt.get("tolerated_collision_messages", []) or [])
  return total


def _is_intentional_interference_collision(item: dict[str, Any]) -> bool:
  metadata = item.get("metadata")
  if not isinstance(metadata, dict):
    return False
  return bool(metadata.get("intentional_interference")) or str(
      metadata.get("predicted_contact_type") or ""
  ) in {"threaded_interference", "screw_in_hole"}


def _tolerated_collision_volume_sum(result_data: dict[str, Any]) -> float:
  total = 0.0
  records = result_data.get("final_tolerated_collisions")
  if isinstance(records, list):
    for item in records:
      if not isinstance(item, dict):
        continue
      volume = item.get("volume")
      if isinstance(volume, (int, float)):
        total += float(volume)
  if total > 0.0:
    return float(total)
  for attempt in result_data.get("attempts", []) or []:
    if not isinstance(attempt, dict):
      continue
    for text in attempt.get("tolerated_collision_messages", []) or []:
      message = str(text)
      if "volume=" not in message:
        continue
      token = message.split("volume=", 1)[1].split(" ", 1)[0].replace(",", "")
      try:
        total += float(token)
      except ValueError:
        continue
  return float(total)


def _connectivity_stats(
    part_names: set[str],
    anchor: Any,
    constraints: list[Any],
) -> dict[str, Any]:
  if not part_names:
    return {
        "connected_components": 0,
        "connected_ratio": 1.0,
        "disconnected_count": 0,
    }

  adjacency = {name: set() for name in part_names}
  for constraint in constraints:
    if not isinstance(constraint, dict):
      continue
    part_a = constraint.get("part_a")
    part_b = constraint.get("part_b")
    if part_a in part_names and part_b in part_names:
      adjacency[part_a].add(part_b)
      adjacency[part_b].add(part_a)

  visited: set[str] = set()
  components = 0
  for start in part_names:
    if start in visited:
      continue
    components += 1
    stack = [start]
    while stack:
      node = stack.pop()
      if node in visited:
        continue
      visited.add(node)
      for nxt in adjacency[node]:
        if nxt not in visited:
          stack.append(nxt)

  if isinstance(anchor, str) and anchor in part_names:
    connected_to_anchor = set()
    stack = [anchor]
    while stack:
      node = stack.pop()
      if node in connected_to_anchor:
        continue
      connected_to_anchor.add(node)
      for nxt in adjacency[node]:
        if nxt not in connected_to_anchor:
          stack.append(nxt)
  else:
    connected_to_anchor = set()

  connected_ratio = float(len(connected_to_anchor)) / float(max(1, len(part_names)))
  disconnected_count = max(0, len(part_names) - len(connected_to_anchor))
  return {
      "connected_components": components,
      "connected_ratio": connected_ratio,
      "disconnected_count": disconnected_count,
  }


def _score_quality(
    *,
    success: bool,
    coverage_ratio: float,
    placed_ratio: float,
    collision_count: int,
    solve_error_count: int,
    unplaced_count: int,
    planner_error_count: int,
    disconnected_count: int,
) -> float:
  score = 20.0
  if success:
    score += 45.0
  score += 18.0 * max(0.0, min(1.0, coverage_ratio))
  score += 15.0 * max(0.0, min(1.0, placed_ratio))

  score -= min(20.0, 2.0 * float(collision_count))
  score -= min(16.0, 3.5 * float(solve_error_count))
  score -= min(14.0, 4.0 * float(unplaced_count))
  score -= min(10.0, 2.5 * float(planner_error_count))
  score -= min(12.0, 3.0 * float(disconnected_count))

  return max(0.0, min(100.0, score))
