"""Target-free joint reranking from bounded LinkCAD execution receipts.

The pair executor emits one terminal row for each attempted functional edge.
This module groups those public-only rows by query and assignment hypothesis,
then promotes hypotheses for which every functional edge was both attempted and
geometrically accepted.  It never consumes intended candidate or primitive
targets and does not claim simultaneous pose consistency.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


RERANK_SCHEMA_VERSION = "linkcad_joint_execution_rerank.v1"


def _canonical_sha256(value: Any) -> str:
  raw = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(raw).hexdigest()


def _validate_public_inputs(
    public: Mapping[str, Any],
    predictions: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
  if public.get("contains_private_targets") is not False:
    raise ValueError("LinkCAD reranker public query scope differs")
  if predictions.get("private_targets_opened") is not False:
    raise ValueError("LinkCAD reranker prediction scope differs")
  if execution.get("private_targets_opened") is not False:
    raise ValueError("LinkCAD reranker execution scope differs")
  if execution.get("prediction_payload_sha256") != predictions.get(
      "prediction_payload_sha256"
  ):
    raise ValueError("LinkCAD reranker prediction binding differs")
  forbidden = (
      "target_candidate_by_role", "target_candidate_id", "target_mobility",
      "source_joint_id", "source_joint_type",
  )
  if any(key in repr((public, predictions, execution)) for key in forbidden):
    raise ValueError("LinkCAD reranker input contains private target material")


def _execution_payload(execution: Mapping[str, Any]) -> Mapping[str, Any]:
  payload = execution.get("execution")
  if payload is None:
    return execution
  if (
      not isinstance(payload, Mapping)
      or execution.get("private_targets_read_by_executor") is not False
  ):
    raise ValueError("LinkCAD reranker execution wrapper differs")
  return payload


def rerank_by_joint_execution_v1(
    *,
    public: Mapping[str, Any],
    predictions: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
  """Rerank fully assessed hypotheses by joint edge feasibility.

  Hypotheses with complete execution coverage are ordered by: all edges
  accepted, accepted-edge count, original learned score, and original rank.
  Unassessed or partially assessed hypotheses retain their relative order after
  the completely assessed prefix.  This makes the exact-call budget explicit
  and prevents missing execution rows from being interpreted as negatives.
  """

  execution_payload = _execution_payload(execution)
  _validate_public_inputs(public, predictions, execution_payload)
  public_by_id = {row["query_id"]: row for row in public["queries"]}
  if len(public_by_id) != len(public["queries"]):
    raise ValueError("LinkCAD reranker query identities duplicate")
  prediction_by_id = {row["query_id"]: row for row in predictions["rows"]}
  if set(prediction_by_id) != set(public_by_id):
    raise ValueError("LinkCAD reranker prediction query domain differs")

  valid_execution_keys = {}
  for query_id, query in public_by_id.items():
    edges = {row["edge_id"]: row for row in query["functional_edges"]}
    hypotheses = {
        int(row["rank"]): row
        for row in prediction_by_id[query_id]["hypotheses"]
    }
    if len(hypotheses) != len(prediction_by_id[query_id]["hypotheses"]):
      raise ValueError("LinkCAD reranker hypothesis ranks duplicate")
    for rank, hypothesis in hypotheses.items():
      for edge_id, edge in edges.items():
        valid_execution_keys[(query_id, rank, str(edge_id))] = (
            hypothesis["candidate_by_role"][edge["role_a"]],
            hypothesis["candidate_by_role"][edge["role_b"]],
        )

  execution_by_key: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
  for row in execution_payload["rows"]:
    key = (
        str(row["query_id"]), int(row["hypothesis_rank"]),
        str(row["edge_id"]),
    )
    expected_candidates = valid_execution_keys.get(key)
    if expected_candidates is None or (
        str(row["candidate_id_a"]), str(row["candidate_id_b"])
    ) != expected_candidates:
      raise ValueError("LinkCAD reranker execution row identity differs")
    execution_by_key.setdefault(key, []).append(row)

  output_rows = []
  complete_assessed_hypotheses = 0
  fully_feasible_hypotheses = 0
  reranked_query_count = 0
  for query_id in sorted(public_by_id):
    edge_ids = tuple(
        str(row["edge_id"])
        for row in public_by_id[query_id]["functional_edges"]
    )
    if not edge_ids or len(set(edge_ids)) != len(edge_ids):
      raise ValueError("LinkCAD reranker functional edge domain differs")
    source_row = prediction_by_id[query_id]
    assessed = []
    unassessed = []
    for hypothesis in source_row["hypotheses"]:
      original_rank = int(hypothesis["rank"])
      edge_rows = [
          execution_by_key.get((query_id, original_rank, edge_id), ())
          for edge_id in edge_ids
      ]
      covered = sum(bool(rows) for rows in edge_rows)
      accepted = sum(
          any(
              row.get("conditional_kinematic_feasibility_accepted") is True
              for row in rows
          )
          for rows in edge_rows
      )
      decorated = copy.deepcopy(dict(hypothesis))
      decorated["original_rank"] = original_rank
      decorated["joint_execution"] = {
          "edge_count": len(edge_ids),
          "assessed_edge_count": covered,
          "accepted_edge_count": accepted,
          "complete_assessment": covered == len(edge_ids),
          "all_edges_conditionally_feasible": accepted == len(edge_ids),
      }
      decorated.pop("hypothesis_payload_sha256", None)
      decorated["hypothesis_payload_sha256"] = _canonical_sha256(decorated)
      if covered == len(edge_ids):
        complete_assessed_hypotheses += 1
        fully_feasible_hypotheses += int(accepted == len(edge_ids))
        assessed.append(decorated)
      else:
        unassessed.append(decorated)
    assessed.sort(key=lambda row: (
        -int(row["joint_execution"]["all_edges_conditionally_feasible"]),
        -int(row["joint_execution"]["accepted_edge_count"]),
        -float(row["score"]),
        int(row["original_rank"]),
    ))
    unassessed.sort(key=lambda row: int(row["original_rank"]))
    ordered = assessed + unassessed
    reranked_query_count += int(any(
        int(row["original_rank"]) != rank for rank, row in enumerate(ordered)
    ))
    for rank, row in enumerate(ordered):
      row["rank"] = rank
      row.pop("hypothesis_payload_sha256", None)
      row["hypothesis_payload_sha256"] = _canonical_sha256(row)
    output_row = {
        "query_id": query_id,
        "terminal_status": source_row["terminal_status"],
        "hypotheses": ordered,
    }
    output_row["row_payload_sha256"] = _canonical_sha256(output_row)
    output_rows.append(output_row)

  payload = {
      "schema_version": predictions["schema_version"],
      "scope": "public_joint_execution_reranked_predictions",
      "private_targets_opened": False,
      "decoder_schema_version": RERANK_SCHEMA_VERSION,
      "source_prediction_payload_sha256": predictions["prediction_payload_sha256"],
      "execution_subset_payload_sha256": execution_payload[
          "subset_payload_sha256"
      ],
      "checkpoint_sha256": predictions["checkpoint_sha256"],
      "model_kind": predictions["model_kind"],
      "beam_size": predictions["beam_size"],
      "interface_alternatives_per_edge": predictions[
          "interface_alternatives_per_edge"
      ],
      "query_count": len(output_rows),
      "complete_assessed_hypothesis_count": complete_assessed_hypotheses,
      "fully_feasible_hypothesis_count": fully_feasible_hypotheses,
      "reranked_query_count": reranked_query_count,
      "rows": output_rows,
  }
  payload["prediction_payload_sha256"] = _canonical_sha256(payload)
  return payload


__all__ = ["RERANK_SCHEMA_VERSION", "rerank_by_joint_execution_v1"]
