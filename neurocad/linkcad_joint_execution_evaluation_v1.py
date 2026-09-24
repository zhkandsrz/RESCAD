"""Private evaluation of the preunseal LinkCAD joint-execution top choice."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .linkcad_confirmation_execution_evaluation_v1 import (
    evaluate_confirmation_execution_v1,
)


def evaluate_joint_execution_selection_v1(
    *,
    public: Mapping[str, Any],
    private_targets: Mapping[str, Any],
    primitive_supervision: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    reranked_predictions: Mapping[str, Any],
) -> dict[str, Any]:
  """Evaluate only the original hypothesis selected by the joint reranker."""

  if (
      execution_receipt.get("private_targets_read_by_executor") is not False
      or execution_receipt.get("query_selection_uses_targets") is not False
      or reranked_predictions.get("private_targets_opened") is not False
      or reranked_predictions.get("source_prediction_payload_sha256")
      != execution_receipt.get("prediction_payload_sha256")
  ):
    raise ValueError("LinkCAD joint execution selection scope differs")
  execution = execution_receipt["execution"]
  query_by_id = {str(row["query_id"]): row for row in public["queries"]}
  reranked_by_id = {
      str(row["query_id"]): row for row in reranked_predictions["rows"]
  }
  if set(query_by_id) != set(reranked_by_id):
    raise ValueError("LinkCAD joint execution query roster differs")
  selected_rank = {}
  for query_id, prediction_row in reranked_by_id.items():
    hypotheses = prediction_row["hypotheses"]
    if not hypotheses:
      raise ValueError("LinkCAD joint execution selection is missing")
    top = hypotheses[0]
    joint = top.get("joint_execution")
    if not isinstance(joint, Mapping) or joint.get("complete_assessment") is not True:
      raise ValueError("LinkCAD joint execution top choice is not fully assessed")
    selected_rank[query_id] = int(top["original_rank"])
  selected_rows = [
      copy.deepcopy(dict(row)) for row in execution["rows"]
      if int(row["hypothesis_rank"]) == selected_rank[str(row["query_id"])]
  ]
  edge_keys = {
      (str(row["query_id"]), str(row["edge_id"])) for row in selected_rows
  }
  expected_edge_keys = {
      (query_id, str(edge["edge_id"]))
      for query_id, query in query_by_id.items()
      for edge in query["functional_edges"]
  }
  if edge_keys != expected_edge_keys or len(selected_rows) != len(edge_keys):
    raise ValueError("LinkCAD joint execution selected edge coverage differs")
  filtered_receipt = {
      "schema_version": "linkcad_confirmation_execution_runtime_receipt.v1",
      "private_targets_read_by_executor": False,
      "query_selection_uses_targets": False,
      "execution": {
          "query_limit": len(query_by_id),
          "rows": selected_rows,
      },
  }
  base = evaluate_confirmation_execution_v1(
      public=public,
      private_targets=private_targets,
      primitive_supervision=primitive_supervision,
      execution_receipt=filtered_receipt,
      source_worlds_by_query=None,
  )
  metrics = base["metrics"]
  renamed_metrics = {
      "predicted_edge_exact_feasibility_rate": metrics[
          "predicted_edge_conditional_feasibility_rate"
      ],
      "intended_part_edge_top1_rate": metrics["intended_part_edge_top1_rate"],
      "intended_support_mobility_top1_rate": metrics[
          "intended_support_mobility_top1_rate"
      ],
      "intended_primitive_exact_top1_rate": metrics[
          "intended_primitive_conditional_feasibility_top3_rate"
      ],
      "intended_full_query_exact_top1_rate": metrics[
          "intended_full_query_conditional_feasibility_top3_rate"
      ],
  }
  return {
      "schema_version": "linkcad_joint_execution_private_evaluation.v1",
      "scope": "private_join_of_preunseal_public_joint_execution_top_choice",
      "source_pose_equivalence_not_defined": True,
      "selected_original_rank_by_query": dict(sorted(selected_rank.items())),
      "counts": base["counts"],
      "metrics": renamed_metrics,
      "per_query": base["per_query"],
  }


__all__ = ["evaluate_joint_execution_selection_v1"]
