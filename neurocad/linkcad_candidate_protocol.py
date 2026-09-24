"""Public/private protocol for language-conditioned LinkCAD candidate sets."""

from __future__ import annotations

import copy
from typing import Any, Mapping


PUBLIC_SCHEMA_VERSION = "linkcad_candidate_set_public.v1"
PRIVATE_SCHEMA_VERSION = "linkcad_candidate_set_private_targets.v1"
COUNTERFACTUAL_SCHEMA_VERSION = "linkcad_counterfactual_intent.v1"


def validate_candidate_set_artifacts(
    public: Mapping[str, Any],
    private: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
) -> None:
  """Fail closed on target leakage or an incomplete candidate oracle."""

  if public.get("schema_version") != PUBLIC_SCHEMA_VERSION:
    raise ValueError("public candidate-set schema differs")
  if private.get("schema_version") != PRIVATE_SCHEMA_VERSION:
    raise ValueError("private candidate-set schema differs")
  if counterfactual.get("schema_version") != COUNTERFACTUAL_SCHEMA_VERSION:
    raise ValueError("counterfactual schema differs")
  if public.get("contains_private_targets") is not False:
    raise ValueError("public artifact target scope differs")
  public_rows = public.get("queries")
  private_rows = private.get("targets")
  if not isinstance(public_rows, list) or not isinstance(private_rows, list):
    raise ValueError("candidate-set rows are missing")
  public_by_id = {
      row["query_id"]: row for row in public_rows if isinstance(row, Mapping)
  }
  private_by_id = {
      row["query_id"]: row for row in private_rows if isinstance(row, Mapping)
  }
  if len(public_by_id) != len(public_rows) or set(public_by_id) != set(private_by_id):
    raise ValueError("public/private query domains differ")

  forbidden_keys = {
      "target_candidate_id",
      "target_candidate_by_role",
      "source_joint_id",
      "source_joint_type",
      "target_mobility",
  }
  for query_id, query in public_by_id.items():
    serialized = repr(query)
    if any(key in serialized for key in forbidden_keys):
      raise ValueError(f"public query {query_id} contains private target fields")
    candidate_sets = query.get("candidate_sets")
    target_by_role = private_by_id[query_id].get("target_candidate_by_role")
    if not isinstance(candidate_sets, Mapping) or not isinstance(
        target_by_role, Mapping
    ):
      raise ValueError("candidate-set target rows differ")
    if set(candidate_sets) != set(target_by_role):
      raise ValueError("candidate-set role domain differs")
    for role_id, target_id in target_by_role.items():
      candidate_ids = {
          row.get("candidate_id")
          for row in candidate_sets[role_id]
          if isinstance(row, Mapping)
      }
      if target_id not in candidate_ids:
        raise ValueError("candidate oracle does not contain the target part")

  pairs = counterfactual.get("pairs")
  if not isinstance(pairs, list):
    raise ValueError("counterfactual pairs are missing")
  for pair in pairs:
    if not isinstance(pair, Mapping):
      raise ValueError("counterfactual row differs")
    query_id = pair.get("query_id")
    query = public_by_id.get(query_id)
    if query is None:
      raise ValueError("counterfactual query domain differs")
    if pair.get("candidate_sets") != query.get("candidate_sets"):
      raise ValueError("counterfactual candidate geometry differs")
    original = pair.get("original_edge")
    changed = pair.get("counterfactual_edge")
    if not isinstance(original, Mapping) or not isinstance(changed, Mapping):
      raise ValueError("counterfactual functional edge differs")
    if original.get("edge_id") != changed.get("edge_id"):
      raise ValueError("counterfactual edge identity differs")
    if original.get("instruction") == changed.get("instruction"):
      raise ValueError("counterfactual instruction did not change")


def public_copy_without_targets(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Return an isolated copy suitable for prediction-time transport."""

  result = copy.deepcopy(dict(payload))
  validate_keys = {
      "targets",
      "target_candidate_by_role",
      "source_joint_id",
      "source_joint_type",
  }
  if any(key in repr(result) for key in validate_keys):
    raise ValueError("prediction-time payload contains target material")
  return result
