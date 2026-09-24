"""Failure taxonomy for neuro-symbolic assembly experiments."""

from __future__ import annotations

from typing import Any


def _as_float(value: Any) -> float | None:
  if isinstance(value, (int, float)):
    return float(value)
  return None


def _as_int(value: Any) -> int | None:
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(value)
  return None


def _contact_graph_metadata(payload: dict[str, Any]) -> dict[str, Any]:
  case = payload.get("case") or {}
  if not isinstance(case, dict):
    return {}
  meta = case.get("benchmark_metadata") or {}
  if isinstance(meta, dict):
    return meta
  return {}


def classify_failure(payload: dict[str, Any]) -> dict[str, Any]:
  """Return a paper-friendly failure category with lightweight evidence."""

  success = bool(payload.get("success", False))
  tolerant_success = bool(payload.get("tolerant_success", success))
  strict_success = bool(payload.get("strict_success", tolerant_success))
  retrieval_metrics = payload.get("retrieval_metrics") or {}
  retrieval_confidence = payload.get("retrieval_confidence") or {}
  mate_contact_validation = payload.get("mate_contact_validation") or {}
  exact_contact_validation = payload.get("exact_contact_validation") or {}
  evaluation = payload.get("evaluation") or {}
  feedback = evaluation.get("feedback") or {}
  meta = _contact_graph_metadata(payload)

  retrieval_recall = _as_float(retrieval_metrics.get("retrieval_recall"))
  retrieval_precision = _as_float(retrieval_metrics.get("retrieval_precision"))
  exact_match = retrieval_metrics.get("exact_part_set_match")
  collision_volume = _as_float(payload.get("collision_volume_sum")) or 0.0
  tolerated_collision_volume = (
      _as_float(payload.get("tolerated_collision_volume_sum")) or 0.0
  )
  planner_error_count = _as_int(feedback.get("planner_error_count")) or 0
  solve_error_count = _as_int(feedback.get("solve_error_count")) or 0
  unplaced_count = _as_int(feedback.get("unplaced_count")) or 0
  disconnected_count = _as_int(evaluation.get("disconnected_count")) or 0
  validation_error_count = _as_int(feedback.get("validation_error_count")) or 0
  output_step = payload.get("output_step")
  output_step_error = payload.get("output_step_error")
  final_feedback = payload.get("final_feedback") or []

  timeout_evidence = [
      str(item) for item in final_feedback if "case_timeout_seconds=" in str(item)
  ]
  if output_step_error == "case_timeout" or timeout_evidence:
    return {
        "category": "case_timeout",
        "stage": "runtime",
        "evidence": timeout_evidence or [str(output_step_error)],
    }
  if output_step_error == "case_process_error":
    return {
        "category": "case_process_error",
        "stage": "runtime",
        "evidence": [str(item) for item in final_feedback[:4]],
    }

  reference_contact_connected = meta.get("reference_contact_graph_connected")
  reference_contact_component_count = meta.get("reference_contact_component_count")
  possibly_missing_bridge = meta.get("possibly_missing_bridge")
  reference_oracle_preflight_success = meta.get("reference_oracle_preflight_success")

  if strict_success:
    return {
        "category": "success_strict",
        "stage": "complete",
        "evidence": [],
    }
  if bool(retrieval_confidence.get("low_confidence", False)):
    return {
        "category": "retrieval_low_confidence",
        "stage": "retrieval",
        "evidence": [
            str(item)
            for item in (retrieval_confidence.get("reasons") or [])[:5]
        ],
    }
  if not bool(mate_contact_validation.get("valid", True)):
    violations = mate_contact_validation.get("violations") or []
    return {
        "category": "underconstrained_stack",
        "stage": "symbolic_grounding",
        "evidence": [
            f"checked_pairs={mate_contact_validation.get('checked_pairs', 0)}",
            *(str(item) for item in violations[:3]),
        ],
    }
  if not bool(exact_contact_validation.get("valid", True)):
    violations = exact_contact_validation.get("violations") or []
    return {
        "category": "geometric_scatter",
        "stage": "symbolic_grounding",
        "evidence": [
            f"exact_contact_recall={exact_contact_validation.get('exact_contact_recall')}",
            f"max_contact_distance={exact_contact_validation.get('max_contact_distance')}",
            *(str(item) for item in violations[:3]),
        ],
    }
  if tolerant_success and not strict_success:
    return {
        "category": "legal_interference_fit",
        "stage": "physical_validation",
        "evidence": [
            f"tolerated_collision_volume={tolerated_collision_volume:.6f}",
            f"blocking_collision_volume={collision_volume:.6f}",
        ],
    }
  if success and output_step is None and output_step_error:
    return {
        "category": "export_failure",
        "stage": "export",
        "evidence": [str(output_step_error)],
    }
  if planner_error_count > 0 or validation_error_count > 0:
    return {
        "category": "topology_planning_failure",
        "stage": "planning",
        "evidence": [
            f"planner_error_count={planner_error_count}",
            f"validation_error_count={validation_error_count}",
        ],
    }
  if solve_error_count > 0 or unplaced_count > 0 or disconnected_count > 0:
    return {
        "category": "grounding_failure",
        "stage": "symbolic_grounding",
        "evidence": [
            f"solve_error_count={solve_error_count}",
            f"unplaced_count={unplaced_count}",
            f"disconnected_count={disconnected_count}",
        ],
    }
  if possibly_missing_bridge or (
      reference_contact_connected is False
      or (
          isinstance(reference_contact_component_count, int)
          and reference_contact_component_count > 1
      )
  ):
    return {
        "category": "missing_bridge_part",
        "stage": "benchmark_ground_truth",
        "evidence": [
            f"reference_contact_connected={reference_contact_connected}",
            f"reference_contact_component_count={reference_contact_component_count}",
        ],
    }
  if reference_oracle_preflight_success is False:
    return {
        "category": "unsatisfiable_ground_truth",
        "stage": "benchmark_ground_truth",
        "evidence": ["reference_oracle_preflight_success=false"],
    }
  if (
      retrieval_recall is not None
      and retrieval_precision is not None
      and (retrieval_recall < 0.5 or retrieval_precision < 0.5)
      and not bool(exact_match)
  ):
    return {
        "category": "retrieval_failure",
        "stage": "retrieval",
        "evidence": [
            f"retrieval_recall={retrieval_recall:.4f}",
            f"retrieval_precision={retrieval_precision:.4f}",
            f"retrieval_exact_match={exact_match}",
        ],
    }
  if collision_volume > 0.0 or tolerated_collision_volume > 0.0:
    return {
        "category": "hard_collision",
        "stage": "physical_validation",
        "evidence": [
            f"collision_volume={collision_volume:.6f}",
            f"tolerated_collision_volume={tolerated_collision_volume:.6f}",
        ],
    }
  if final_feedback:
    return {
        "category": "unknown_failure",
        "stage": "unknown",
        "evidence": [str(item) for item in final_feedback[:4]],
    }
  return {
      "category": "unknown_failure",
      "stage": "unknown",
      "evidence": [],
  }


def summarize_failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for row in rows:
    category = row.get("failure_category")
    if not isinstance(category, str) or not category:
      continue
    counts[category] = counts.get(category, 0) + 1
  return counts
