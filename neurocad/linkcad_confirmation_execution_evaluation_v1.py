"""Private join for frozen LinkCAD confirmation execution receipts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .linkcad_pose_equivalence_v1 import mobility_quotient_errors_v1


SCHEMA_VERSION = "linkcad_confirmation_execution_private_evaluation.v1"


def _primitive_targets(
    supervision: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[int, int]]:
  by_endpoint = {
      (str(row["query_id"]), str(row["edge_id"]), str(row["side"])): int(
          row["primitive_orbit_ordinal"]
      )
      for row in supervision["rows"]
      if row.get("status") == "mapped_type_consistent"
  }
  result = {}
  for query_id, edge_id, _ in sorted(by_endpoint):
    left = by_endpoint.get((query_id, edge_id, "a"))
    right = by_endpoint.get((query_id, edge_id, "b"))
    if left is not None and right is not None:
      result[(query_id, edge_id)] = (left, right)
  return result


def evaluate_confirmation_execution_v1(
    *,
    public: Mapping[str, Any],
    private_targets: Mapping[str, Any],
    primitive_supervision: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    source_worlds_by_query: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
  """Measure intended-interface recovery after target-free exact execution."""

  if (
      execution_receipt.get("schema_version")
      != "linkcad_confirmation_execution_runtime_receipt.v1"
      or execution_receipt.get("private_targets_read_by_executor") is not False
      or execution_receipt.get("query_selection_uses_targets") is not False
  ):
    raise ValueError("LinkCAD confirmation execution scope differs")
  public_by_id = {str(row["query_id"]): row for row in public["queries"]}
  private_by_id = {
      str(row["query_id"]): row for row in private_targets["targets"]
  }
  execution = execution_receipt["execution"]
  query_limit = int(execution["query_limit"])
  selected_query_ids = tuple(sorted(public_by_id)[:query_limit])
  attempts_by_edge: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
  for row in execution["rows"]:
    key = (str(row["query_id"]), str(row["edge_id"]))
    attempts_by_edge.setdefault(key, []).append(row)
  primitive_targets = _primitive_targets(primitive_supervision)
  query_rows = []
  counts = {
      "selected_query_count": len(selected_query_ids),
      "selected_edge_count": 0,
      "primitive_supervised_edge_count": 0,
      "predicted_edge_conditionally_feasible": 0,
      "intended_part_edge_top1": 0,
      "intended_support_mobility_top1": 0,
      "intended_primitive_attempted_top3": 0,
      "intended_primitive_conditionally_feasible_top3": 0,
      "source_pose_authority_edge_count": 0,
      "source_pose_input_edge_count": 0,
      "unsupported_pose_equivalence_edge_count": 0,
      "intended_pose_equivalent_edge_top3": 0,
      "primitive_complete_query_count": 0,
      "intended_full_query_conditionally_feasible_top3": 0,
      "source_pose_complete_query_count": 0,
      "intended_full_query_pose_equivalent_top3": 0,
  }
  for query_id in selected_query_ids:
    query = public_by_id[query_id]
    target = private_by_id[query_id]
    target_edges = {
        str(row["edge_id"]): row
        for row in target["functional_edge_targets"]
    }
    edge_rows = []
    for edge in query["functional_edges"]:
      edge_id = str(edge["edge_id"])
      attempts = attempts_by_edge.get((query_id, edge_id), [])
      target_edge = target_edges[edge_id]
      target_a = str(target["target_candidate_by_role"][edge["role_a"]])
      target_b = str(target["target_candidate_by_role"][edge["role_b"]])
      intended_part = any(
          str(row["candidate_id_a"]) == target_a
          and str(row["candidate_id_b"]) == target_b
          for row in attempts
      )
      intended_semantic = any(
          str(row["candidate_id_a"]) == target_a
          and str(row["candidate_id_b"]) == target_b
          and str(row["support_family"]) == target_edge["support_family"]
          and str(row["mobility"]) == target_edge["target_mobility"]
          for row in attempts
      )
      target_primitive = primitive_targets.get((query_id, edge_id))
      supervised = target_primitive is not None
      intended_attempts = [
          row for row in attempts
          if str(row["candidate_id_a"]) == target_a
          and str(row["candidate_id_b"]) == target_b
          and str(row["support_family"]) == target_edge["support_family"]
          and str(row["mobility"]) == target_edge["target_mobility"]
          and (int(row["primitive_a"]), int(row["primitive_b"]))
          == target_primitive
      ] if supervised else []
      intended_executable = any(
          row["conditional_kinematic_feasibility_accepted"] is True
          for row in intended_attempts
      )
      predicted_feasible = any(
          row["conditional_kinematic_feasibility_accepted"] is True
          for row in attempts
      )
      source_worlds = (
          None if source_worlds_by_query is None
          else source_worlds_by_query.get(query_id)
      )
      occurrence_a = str(target_edge.get("endpoint_a", {}).get(
          "occurrence_id", ""
      ))
      occurrence_b = str(target_edge.get("endpoint_b", {}).get(
          "occurrence_id", ""
      ))
      source_pose_input = (
          source_worlds is not None
          and occurrence_a in source_worlds
          and occurrence_b in source_worlds
          and isinstance(target_edge.get("endpoint_b"), Mapping)
      )
      pose_equivalence_supported = str(
          target_edge["target_mobility"]
      ) in {"fixed", "revolute", "prismatic", "cylindrical"}
      source_pose_authority = source_pose_input and pose_equivalence_supported
      pose_equivalent = False
      best_pose_errors = None
      if source_pose_authority:
        for row in attempts:
          selected = row.get("selected_observation")
          if not (
              str(row["candidate_id_a"]) == target_a
              and str(row["candidate_id_b"]) == target_b
              and str(row["support_family"]) == target_edge["support_family"]
              and str(row["mobility"]) == target_edge["target_mobility"]
              and row["conditional_kinematic_feasibility_accepted"] is True
              and isinstance(selected, Mapping)
              and isinstance(selected.get("child_world_row_major"), list)
          ):
            continue
          relative = np.asarray(
              selected["child_world_row_major"], dtype=float
          ).reshape(4, 4)
          errors = mobility_quotient_errors_v1(
              source_child_world=source_worlds[occurrence_b],
              predicted_child_world=source_worlds[occurrence_a] @ relative,
              child_endpoint_world=target_edge["endpoint_b"],
              mobility=str(target_edge["target_mobility"]),
          )
          if (
              best_pose_errors is None
              or (
                  float(errors["translation_error_mm"]),
                  float(errors["rotation_error_deg"]),
              ) < (
                  float(best_pose_errors["translation_error_mm"]),
                  float(best_pose_errors["rotation_error_deg"]),
              )
          ):
            best_pose_errors = errors
          pose_equivalent = pose_equivalent or bool(errors["equivalent"])
      counts["selected_edge_count"] += 1
      counts["primitive_supervised_edge_count"] += int(supervised)
      counts["predicted_edge_conditionally_feasible"] += int(predicted_feasible)
      counts["intended_part_edge_top1"] += int(intended_part)
      counts["intended_support_mobility_top1"] += int(intended_semantic)
      counts["intended_primitive_attempted_top3"] += int(bool(intended_attempts))
      counts["intended_primitive_conditionally_feasible_top3"] += int(
          intended_executable
      )
      counts["source_pose_authority_edge_count"] += int(source_pose_authority)
      counts["source_pose_input_edge_count"] += int(source_pose_input)
      counts["unsupported_pose_equivalence_edge_count"] += int(
          source_pose_input and not pose_equivalence_supported
      )
      counts["intended_pose_equivalent_edge_top3"] += int(pose_equivalent)
      edge_rows.append({
          "edge_id": edge_id,
          "attempt_count": len(attempts),
          "primitive_supervised": supervised,
          "predicted_edge_conditionally_feasible": predicted_feasible,
          "intended_part_edge_top1": intended_part,
          "intended_support_mobility_top1": intended_semantic,
          "intended_primitive_attempted_top3": bool(intended_attempts),
          "intended_primitive_conditionally_feasible_top3": intended_executable,
          "source_pose_authority": source_pose_authority,
          "source_pose_input_available": source_pose_input,
          "pose_equivalence_supported": pose_equivalence_supported,
          "intended_pose_equivalent_top3": pose_equivalent,
          "best_pose_errors": best_pose_errors,
      })
    primitive_complete = all(
        row["primitive_supervised"] for row in edge_rows
    )
    full_intended = primitive_complete and all(
        row["intended_primitive_conditionally_feasible_top3"]
        for row in edge_rows
    )
    source_pose_complete = all(
        row["source_pose_authority"] for row in edge_rows
    )
    full_pose_equivalent = source_pose_complete and all(
        row["intended_pose_equivalent_top3"] for row in edge_rows
    )
    counts["primitive_complete_query_count"] += int(primitive_complete)
    counts["intended_full_query_conditionally_feasible_top3"] += int(
        full_intended
    )
    counts["source_pose_complete_query_count"] += int(source_pose_complete)
    counts["intended_full_query_pose_equivalent_top3"] += int(
        full_pose_equivalent
    )
    query_rows.append({
        "query_id": query_id,
        "primitive_complete": primitive_complete,
        "all_predicted_edges_conditionally_feasible": all(
            row["predicted_edge_conditionally_feasible"] for row in edge_rows
        ),
        "intended_full_query_conditionally_feasible_top3": full_intended,
        "source_pose_complete": source_pose_complete,
        "intended_full_query_pose_equivalent_top3": full_pose_equivalent,
        "edges": edge_rows,
    })

  edge_count = counts["selected_edge_count"]
  supervised_count = counts["primitive_supervised_edge_count"]
  complete_count = counts["primitive_complete_query_count"]
  pose_edge_count = counts["source_pose_authority_edge_count"]
  pose_query_count = counts["source_pose_complete_query_count"]
  metrics = {
      "predicted_edge_conditional_feasibility_rate": (
          counts["predicted_edge_conditionally_feasible"] / edge_count
          if edge_count else 0.0
      ),
      "intended_part_edge_top1_rate": (
          counts["intended_part_edge_top1"] / edge_count if edge_count else 0.0
      ),
      "intended_support_mobility_top1_rate": (
          counts["intended_support_mobility_top1"] / edge_count
          if edge_count else 0.0
      ),
      "intended_primitive_attempted_top3_rate": (
          counts["intended_primitive_attempted_top3"] / supervised_count
          if supervised_count else 0.0
      ),
      "intended_primitive_conditional_feasibility_top3_rate": (
          counts["intended_primitive_conditionally_feasible_top3"]
          / supervised_count if supervised_count else 0.0
      ),
      "intended_full_query_conditional_feasibility_top3_rate": (
          counts["intended_full_query_conditionally_feasible_top3"]
          / complete_count if complete_count else 0.0
      ),
      "intended_pose_equivalent_edge_top3_rate": (
          counts["intended_pose_equivalent_edge_top3"] / pose_edge_count
          if pose_edge_count else 0.0
      ),
      "intended_full_query_pose_equivalent_top3_rate": (
          counts["intended_full_query_pose_equivalent_top3"] / pose_query_count
          if pose_query_count else 0.0
      ),
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "scope": (
          "private_join_after_target_free_execution_of_preunseal_frozen_"
          "predictions"
      ),
      "pairwise_feasibility_not_global_pose_graph_consistency": True,
      "counts": counts,
      "metrics": metrics,
      "per_query": query_rows,
  }


__all__ = ["SCHEMA_VERSION", "evaluate_confirmation_execution_v1"]
