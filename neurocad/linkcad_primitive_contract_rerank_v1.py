"""Target-free primitive reranking from exact public port contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from .linkcad_port_contract_v1 import describe_ports_v1


SCHEMA_VERSION = "linkcad_primitive_contract_rerank.v1"


def _canonical_sha256(value: Any) -> str:
  raw = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(raw).hexdigest()


def _contract_key(value: Mapping[str, Any]) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def rerank_primitive_alternatives_v1(
    alternatives: list[Mapping[str, Any]],
    *,
    target_a: Mapping[str, Any],
    target_b: Mapping[str, Any],
    available_a: tuple[Mapping[str, Any], ...],
    available_b: tuple[Mapping[str, Any], ...],
) -> list[dict[str, Any]]:
  """Promote alternatives matching both endpoint contracts exactly."""

  target_keys = (_contract_key(target_a), _contract_key(target_b))
  decorated = []
  for original_rank, source in enumerate(alternatives):
    row = copy.deepcopy(dict(source))
    ordinal_a = int(row["interface_primitive_a"])
    ordinal_b = int(row["interface_primitive_b"])
    if not (
        0 <= ordinal_a < len(available_a)
        and 0 <= ordinal_b < len(available_b)
    ):
      raise ValueError("LinkCAD primitive alternative ordinal differs")
    exact_a = _contract_key(available_a[ordinal_a]) == target_keys[0]
    exact_b = _contract_key(available_b[ordinal_b]) == target_keys[1]
    row["original_rank"] = original_rank
    row["contract_exact_a"] = exact_a
    row["contract_exact_b"] = exact_b
    row["contract_exact_pair"] = exact_a and exact_b
    decorated.append(row)
  decorated.sort(key=lambda row: (
      -int(row["contract_exact_pair"]),
      -int(row["contract_exact_a"]) - int(row["contract_exact_b"]),
      -float(row.get("model_score") or 0.0),
      int(row["original_rank"]),
  ))
  for rank, row in enumerate(decorated):
    row["rank"] = rank
  return decorated


def rerank_predictions_by_primitive_contract_v1(
    *,
    public: Mapping[str, Any],
    predictions: Mapping[str, Any],
    cache: Any,
) -> dict[str, Any]:
  """Rerank each public primitive pool without candidate or target labels."""

  if (
      public.get("contains_private_targets") is not False
      or predictions.get("private_targets_opened") is not False
  ):
    raise ValueError("LinkCAD primitive contract reranker scope differs")
  forbidden = (
      "target_candidate_by_role", "target_candidate_id", "target_mobility",
      "source_joint_id", "source_joint_type",
  )
  if any(key in repr((public, predictions)) for key in forbidden):
    raise ValueError("LinkCAD primitive contract reranker contains private target")
  public_by_id = {str(row["query_id"]): row for row in public["queries"]}
  prediction_by_id = {
      str(row["query_id"]): row for row in predictions["rows"]
  }
  if set(public_by_id) != set(prediction_by_id):
    raise ValueError("LinkCAD primitive contract reranker query domain differs")

  graph_contracts: dict[str, tuple[Mapping[str, Any], ...]] = {}

  def contracts(step_sha256: str):
    if step_sha256 not in graph_contracts:
      graph = cache.graph(step_sha256)
      if graph is None:
        raise ValueError("LinkCAD primitive contract graph is unavailable")
      graph_contracts[step_sha256] = describe_ports_v1(graph)
    return graph_contracts[step_sha256]

  output_rows = []
  alternative_count = 0
  exact_pair_count = 0
  promoted_edge_program_count = 0
  for query_id in sorted(public_by_id):
    query = public_by_id[query_id]
    edge_by_id = {
        str(row["edge_id"]): row for row in query["functional_edges"]
    }
    candidates_by_role = {
        role: {str(row["candidate_id"]): row for row in rows}
        for role, rows in query["candidate_sets"].items()
    }
    source_row = prediction_by_id[query_id]
    hypotheses = []
    for source_hypothesis in source_row["hypotheses"]:
      hypothesis = copy.deepcopy(dict(source_hypothesis))
      edge_programs = []
      for source_program in hypothesis["edge_programs"]:
        program = copy.deepcopy(dict(source_program))
        edge = edge_by_id[str(program["edge_id"])]
        specified = (
            edge.get("port_contract_status") == "specified"
            and isinstance(edge.get("port_contract_a"), Mapping)
            and isinstance(edge.get("port_contract_b"), Mapping)
        )
        alternatives = program.get("primitive_alternatives", [])
        if specified and alternatives:
          role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
          candidate_a = candidates_by_role[role_a][
              hypothesis["candidate_by_role"][role_a]
          ]
          candidate_b = candidates_by_role[role_b][
              hypothesis["candidate_by_role"][role_b]
          ]
          original = (
              int(alternatives[0]["interface_primitive_a"]),
              int(alternatives[0]["interface_primitive_b"]),
          )
          alternatives = rerank_primitive_alternatives_v1(
              list(alternatives),
              target_a=edge["port_contract_a"],
              target_b=edge["port_contract_b"],
              available_a=contracts(str(candidate_a["step_sha256"])),
              available_b=contracts(str(candidate_b["step_sha256"])),
          )
          promoted = (
              int(alternatives[0]["interface_primitive_a"]),
              int(alternatives[0]["interface_primitive_b"]),
          ) != original
          promoted_edge_program_count += int(promoted)
          alternative_count += len(alternatives)
          exact_pair_count += sum(
              bool(row["contract_exact_pair"]) for row in alternatives
          )
          program["primitive_alternatives"] = alternatives
          program["interface_primitive_a"] = int(
              alternatives[0]["interface_primitive_a"]
          )
          program["interface_primitive_b"] = int(
              alternatives[0]["interface_primitive_b"]
          )
          program["primitive_contract_reranked"] = True
        else:
          program["primitive_contract_reranked"] = False
        edge_programs.append(program)
      hypothesis["edge_programs"] = edge_programs
      hypothesis.pop("hypothesis_payload_sha256", None)
      hypothesis["hypothesis_payload_sha256"] = _canonical_sha256(hypothesis)
      hypotheses.append(hypothesis)
    row = {
        "query_id": query_id,
        "terminal_status": source_row["terminal_status"],
        "hypotheses": hypotheses,
    }
    row["row_payload_sha256"] = _canonical_sha256(row)
    output_rows.append(row)
  payload = {
      **{
          key: value for key, value in predictions.items()
          if key not in {"rows", "prediction_payload_sha256", "scope"}
      },
      "scope": "public_exact_port_contract_primitive_reranked_predictions",
      "primitive_contract_rerank_schema_version": SCHEMA_VERSION,
      "source_prediction_payload_sha256": predictions[
          "prediction_payload_sha256"
      ],
      "alternative_count": alternative_count,
      "exact_contract_pair_alternative_count": exact_pair_count,
      "promoted_edge_program_count": promoted_edge_program_count,
      "rows": output_rows,
  }
  payload["prediction_payload_sha256"] = _canonical_sha256(payload)
  return payload


__all__ = [
    "SCHEMA_VERSION", "rerank_primitive_alternatives_v1",
    "rerank_predictions_by_primitive_contract_v1",
]
