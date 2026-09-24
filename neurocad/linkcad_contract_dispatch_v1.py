"""Dispatch frozen LinkCAD part rankings to exact public interface contracts."""

from __future__ import annotations

from typing import Any, Mapping

from .linkcad_generalist_agent_baseline_v1 import canonical_sha256
from .linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2
from .linkcad_symbolic_baseline_v1 import (
    materialize_external_assignment_predictions_v1,
)


def role_rankings_from_prediction_beam_v1(
    *, public: Mapping[str, Any], predictions: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
  """Recover target-free per-role rankings while preserving each frozen Top-1."""

  public_queries = {
      str(row["query_id"]): row for row in public["queries"]
  }
  prediction_rows = {
      str(row["query_id"]): row for row in predictions["rows"]
  }
  if set(public_queries) != set(prediction_rows):
    raise ValueError("Contract dispatch query domain differs")
  rankings: dict[str, dict[str, list[str]]] = {}
  for query_id, query in public_queries.items():
    row = prediction_rows[query_id]
    hypotheses = row.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
      raise ValueError("Contract dispatch requires a frozen hypothesis")
    candidate_sets = query.get("candidate_sets")
    if isinstance(candidate_sets, Mapping):
      candidates_by_role = {
          str(role["role_id"]): [
              str(candidate["candidate_id"])
              for candidate in candidate_sets[str(role["role_id"])]
          ]
          for role in query["roles"]
      }
    else:
      candidates_by_role = {
          str(role["role_id"]): [
              str(candidate["candidate_id"]) for candidate in role["candidates"]
          ]
          for role in query["roles"]
      }
    ordered: dict[str, list[str]] = {role: [] for role in candidates_by_role}
    for hypothesis in hypotheses:
      assignment = {
          str(role): str(candidate)
          for role, candidate in hypothesis["candidate_by_role"].items()
      }
      if set(assignment) != set(candidates_by_role):
        raise ValueError("Contract dispatch role domain differs")
      for role, candidate in assignment.items():
        if candidate not in candidates_by_role[role]:
          raise ValueError("Contract dispatch candidate domain differs")
        if candidate not in ordered[role]:
          ordered[role].append(candidate)
    for role, candidates in candidates_by_role.items():
      ordered[role].extend(
          candidate for candidate in candidates if candidate not in ordered[role]
      )
    rankings[query_id] = ordered
  return rankings


def materialize_contract_dispatch_predictions_v1(
    *, public: Mapping[str, Any], predictions: Mapping[str, Any],
    cache: LinkCADPrimitiveGraphCacheV2, beam_size: int = 25,
    interface_alternatives_per_edge: int = 8,
) -> dict[str, Any]:
  """Keep learned part ordering and use exact interface matches when unique."""

  output = materialize_external_assignment_predictions_v1(
      public=public, cache=cache,
      role_rankings_by_query=role_rankings_from_prediction_beam_v1(
          public=public, predictions=predictions,
      ),
      method_id="linkcad_contract_dispatch",
      model_id=str(predictions.get("model_kind", "linkcad")),
      beam_size=beam_size,
      interface_alternatives_per_edge=interface_alternatives_per_edge,
  )
  output["scope"] = (
      "frozen_linkcad_part_order_with_unique_public_interface_contract_dispatch"
  )
  output["source_prediction_payload_sha256"] = predictions.get(
      "prediction_payload_sha256"
  )
  output["checkpoint_sha256"] = predictions.get("checkpoint_sha256")
  output["model_kind"] = "linkcad_contract_dispatch"
  output.pop("prediction_payload_sha256", None)
  output["prediction_payload_sha256"] = canonical_sha256(output)
  return output


__all__ = [
    "materialize_contract_dispatch_predictions_v1",
    "role_rankings_from_prediction_beam_v1",
]
