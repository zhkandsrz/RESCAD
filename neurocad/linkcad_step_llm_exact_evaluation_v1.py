"""Private evaluation for target-free STEP-LLM transfer exact execution."""

from __future__ import annotations

from collections import Counter
import math
import random
from statistics import median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "linkcad_step_llm_exact_evaluation.v1"
EXECUTION_RECEIPT_SCHEMA_VERSION = (
    "linkcad_step_llm_exact_execution_receipt.v1"
)


def _primitive_targets(
    supervision: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[int, int]]:
  if (
      supervision.get("schema_version")
      != "linkcad_direct_primitive_orbit_supervision.v2"
  ):
    raise ValueError("STEP-LLM primitive supervision differs")
  endpoints = {
      (str(row["query_id"]), str(row["edge_id"]), str(row["side"])): int(
          row["primitive_orbit_ordinal"]
      )
      for row in supervision["rows"]
      if row.get("status") == "mapped_type_consistent"
  }
  result = {}
  for query_id, edge_id, _ in sorted(endpoints):
    left = endpoints.get((query_id, edge_id, "a"))
    right = endpoints.get((query_id, edge_id, "b"))
    if left is not None and right is not None:
      result[(query_id, edge_id)] = (left, right)
  return result


def evaluate_step_llm_exact_execution_v1(
    *, public: Mapping[str, Any], private_targets: Mapping[str, Any],
    primitive_supervision: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
  """Measure intended execution at a frozen 25-attempt global budget."""

  if (
      public.get("contains_private_targets") is not False
      or private_targets.get("schema_version")
      != "linkcad_candidate_set_private_targets.v1"
      or execution_receipt.get("schema_version")
      != EXECUTION_RECEIPT_SCHEMA_VERSION
      or execution_receipt.get("private_targets_read_by_executor") is not False
      or execution_receipt.get("query_selection_uses_targets") is not False
  ):
    raise ValueError("STEP-LLM exact execution scope differs")
  execution = execution_receipt["execution"]
  if int(execution.get("maximum_attempts_per_query", 0)) != 25:
    raise ValueError("STEP-LLM exact execution budget differs")

  public_by_id = {str(row["query_id"]): row for row in public["queries"]}
  private_by_id = {
      str(row["query_id"]): row for row in private_targets["targets"]
  }
  if set(public_by_id) != set(private_by_id):
    raise ValueError("STEP-LLM exact private query domain differs")
  query_limit = int(execution["query_limit"])
  selected_ids = tuple(sorted(public_by_id)[:query_limit])
  attempts_by_query: dict[str, list[Mapping[str, Any]]] = {
      query_id: [] for query_id in selected_ids
  }
  for row in execution["rows"]:
    query_id = str(row["query_id"])
    if query_id not in attempts_by_query:
      raise ValueError("STEP-LLM exact execution query differs")
    attempts_by_query[query_id].append(row)
  primitive_targets = _primitive_targets(primitive_supervision)

  rows = []
  for query_id in selected_ids:
    query = public_by_id[query_id]
    target = private_by_id[query_id]
    if len(query["functional_edges"]) != 1:
      raise ValueError("STEP-LLM exact edge domain differs")
    edge = query["functional_edges"][0]
    edge_id = str(edge["edge_id"])
    target_edge = next(
        row for row in target["functional_edge_targets"]
        if str(row["edge_id"]) == edge_id
    )
    primitive_pair = primitive_targets.get((query_id, edge_id))
    if primitive_pair is None:
      raise ValueError("STEP-LLM intended primitive target is absent")
    target_a = str(target["target_candidate_by_role"][edge["role_a"]])
    target_b = str(target["target_candidate_by_role"][edge["role_b"]])
    attempts = attempts_by_query[query_id]

    def assignment_match(row: Mapping[str, Any]) -> bool:
      return (
          str(row["candidate_id_a"]) == target_a
          and str(row["candidate_id_b"]) == target_b
      )

    def interface_match(row: Mapping[str, Any]) -> bool:
      return (
          assignment_match(row)
          and str(row["support_family"]) == str(
              target_edge["support_family"]
          )
          and str(row["mobility"]) == str(target_edge["target_mobility"])
          and (int(row["primitive_a"]), int(row["primitive_b"]))
          == primitive_pair
      )

    any_accepted = any(
        row["conditional_kinematic_feasibility_accepted"] is True
        for row in attempts
    )
    assignment_attempted = any(assignment_match(row) for row in attempts)
    interface_attempted = any(interface_match(row) for row in attempts)
    intended_exact = any(
        interface_match(row)
        and row["conditional_kinematic_feasibility_accepted"] is True
        for row in attempts
    )
    first_accepted = next((
        row for row in attempts
        if row["conditional_kinematic_feasibility_accepted"] is True
    ), None)
    first_accepted_intended = (
        first_accepted is not None and interface_match(first_accepted)
    )
    if not attempts:
      failure_stage = "no_hypothesis_executed"
    elif not assignment_attempted:
      failure_stage = "intended_assignment_not_reached"
    elif not interface_attempted:
      failure_stage = "intended_interface_not_reached"
    elif not intended_exact:
      failure_stage = "intended_interface_not_executable"
    elif not first_accepted_intended:
      failure_stage = "first_accept_not_intended"
    else:
      failure_stage = "intended_first_accept"
    rows.append({
        "query_id": query_id,
        "attempt_count": len(attempts),
        "any_exact_accepted_at_25": any_accepted,
        "intended_assignment_attempted_at_25": assignment_attempted,
        "intended_interface_attempted_at_25": interface_attempted,
        "intended_interface_exact_at_25": intended_exact,
        "first_accepted_intended_at_25": first_accepted_intended,
        "failure_stage": failure_stage,
    })

  metric_names = (
      "any_exact_accepted_at_25",
      "intended_assignment_attempted_at_25",
      "intended_interface_attempted_at_25",
      "intended_interface_exact_at_25",
      "first_accepted_intended_at_25",
  )
  count = len(rows)
  return {
      "schema_version": SCHEMA_VERSION,
      "scope": "private_join_after_frozen_target_free_exact_execution",
      "query_count": count,
      "metrics": {
          name: (
              sum(bool(row[name]) for row in rows) / count if count else 0.0
          )
          for name in metric_names
      },
      "failure_stage_counts": dict(sorted(Counter(
          str(row["failure_stage"]) for row in rows
      ).items())),
      "per_query": rows,
  }


def _percentile(values: Sequence[float], probability: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(float(value) for value in values)
  index = min(
      len(ordered) - 1, max(0, math.floor(probability * len(ordered)))
  )
  return ordered[index]


def _metric_vector(
    evaluation: Mapping[str, Any], metric: str,
) -> tuple[tuple[str, ...], list[float]]:
  rows = sorted(evaluation["per_query"], key=lambda row: str(row["query_id"]))
  if any(metric not in row for row in rows):
    raise ValueError("STEP-LLM exact bootstrap metric differs")
  return (
      tuple(str(row["query_id"]) for row in rows),
      [float(bool(row[metric])) for row in rows],
  )


def paired_exact_seed_median_bootstrap_v1(
    *, learned: Sequence[Mapping[str, Any]], symbolic: Mapping[str, Any],
    metric: str, samples: int = 10_000, seed: int = 20260823,
) -> dict[str, Any]:
  """Query-paired CI for median learned exact success versus symbolic."""

  if not learned or samples < 1:
    raise ValueError("STEP-LLM exact bootstrap scope differs")
  symbolic_ids, symbolic_vector = _metric_vector(symbolic, metric)
  learned_vectors = []
  for evaluation in learned:
    query_ids, vector = _metric_vector(evaluation, metric)
    if query_ids != symbolic_ids:
      raise ValueError("STEP-LLM exact bootstrap query domain differs")
    learned_vectors.append(vector)
  count = len(symbolic_vector)
  if count == 0:
    raise ValueError("STEP-LLM exact bootstrap domain is empty")

  def learned_value(indices: Sequence[int]) -> float:
    return median(
        sum(vector[index] for index in indices) / len(indices)
        for vector in learned_vectors
    )

  complete = tuple(range(count))
  point = learned_value(complete) - sum(symbolic_vector) / count
  rng = random.Random(seed)
  deltas = []
  for _ in range(samples):
    indices = tuple(rng.randrange(count) for _ in range(count))
    deltas.append(
        learned_value(indices)
        - sum(symbolic_vector[index] for index in indices) / count
    )
  return {
      "metric": metric,
      "query_count": count,
      "learned_seed_count": len(learned_vectors),
      "bootstrap_samples": samples,
      "point_difference": point,
      "ci95": [_percentile(deltas, 0.025), _percentile(deltas, 0.975)],
  }


def paired_exact_seed_median_comparison_bootstrap_v1(
    *, treatment: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]], metric: str,
    samples: int = 10_000, seed: int = 20260823,
) -> dict[str, Any]:
  """Query-paired CI for two equally replicated exact-execution methods."""

  if not treatment or len(treatment) != len(control) or samples < 1:
    raise ValueError("STEP-LLM exact treatment/control scope differs")
  treatment_vectors = []
  control_vectors = []
  query_ids = None
  for evaluation in treatment:
    current_ids, vector = _metric_vector(evaluation, metric)
    if query_ids is None:
      query_ids = current_ids
    elif current_ids != query_ids:
      raise ValueError("STEP-LLM exact treatment query domain differs")
    treatment_vectors.append(vector)
  for evaluation in control:
    current_ids, vector = _metric_vector(evaluation, metric)
    if current_ids != query_ids:
      raise ValueError("STEP-LLM exact control query domain differs")
    control_vectors.append(vector)
  count = len(query_ids or ())
  if count == 0:
    raise ValueError("STEP-LLM exact comparison domain is empty")

  def method_value(
      vectors: Sequence[Sequence[float]], indices: Sequence[int],
  ) -> float:
    return median(
        sum(vector[index] for index in indices) / len(indices)
        for vector in vectors
    )

  complete = tuple(range(count))
  point = method_value(treatment_vectors, complete) - method_value(
      control_vectors, complete,
  )
  rng = random.Random(seed)
  deltas = []
  for _ in range(samples):
    indices = tuple(rng.randrange(count) for _ in range(count))
    deltas.append(
        method_value(treatment_vectors, indices)
        - method_value(control_vectors, indices)
    )
  return {
      "metric": metric,
      "query_count": count,
      "seed_count_per_method": len(treatment_vectors),
      "bootstrap_samples": samples,
      "point_difference": point,
      "ci95": [_percentile(deltas, 0.025), _percentile(deltas, 0.975)],
  }


__all__ = [
    "EXECUTION_RECEIPT_SCHEMA_VERSION", "SCHEMA_VERSION",
    "evaluate_step_llm_exact_execution_v1",
    "paired_exact_seed_median_comparison_bootstrap_v1",
    "paired_exact_seed_median_bootstrap_v1",
]
