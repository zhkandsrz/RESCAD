"""Evaluation and paired statistics for LinkCAD contract-strength curves."""

from __future__ import annotations

import math
import random
from statistics import median
from typing import Any, Mapping, Sequence

from .linkcad_contract_strength_v1 import port_contract_matches_v1
from .linkcad_port_contract_v1 import describe_port_v1
from .linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2


SCHEMA_VERSION = "linkcad_contract_strength_evaluation.v1"


def target_retention_audit_v1(
    *, public: Mapping[str, Any], private: Mapping[str, Any],
    primitive_targets: Mapping[tuple[str, str], tuple[int, int]],
    cache: LinkCADPrimitiveGraphCacheV2,
) -> dict[str, Any]:
  private_by_id = {str(row["query_id"]): row for row in private["targets"]}
  endpoint_count = endpoint_candidate_hits = primitive_hits = 0
  query_hits = 0
  rows = []
  for query in public["queries"]:
    query_id = str(query["query_id"])
    target = private_by_id[query_id]
    query_ok = True
    edge_rows = []
    for edge in query["functional_edges"]:
      edge_id = str(edge["edge_id"])
      target_primitives = primitive_targets.get((query_id, edge_id))
      endpoints = []
      for side_index, (side, role_key) in enumerate(
          (("a", "role_a"), ("b", "role_b"))
      ):
        endpoint_count += 1
        role_id = str(edge[role_key])
        candidate_id = str(target["target_candidate_by_role"][role_id])
        candidate = next(
            row for row in query["candidate_sets"][role_id]
            if str(row["candidate_id"]) == candidate_id
        )
        graph = cache.graph(str(candidate["step_sha256"]))
        contract = edge[f"port_contract_{side}"]
        matching = () if graph is None else tuple(
            ordinal
            for ordinal in range(int(graph.primitive_features.shape[0]))
            if port_contract_matches_v1(describe_port_v1(graph, ordinal), contract)
        )
        candidate_hit = bool(matching)
        endpoint_candidate_hits += int(candidate_hit)
        primitive_ordinal = (
            None if target_primitives is None else int(target_primitives[side_index])
        )
        primitive_hit = (
            primitive_ordinal is not None and primitive_ordinal in matching
        )
        primitive_hits += int(primitive_hit)
        query_ok = query_ok and candidate_hit and primitive_hit
        endpoints.append({
            "side": side,
            "target_candidate_id": candidate_id,
            "matching_primitive_count": len(matching),
            "target_primitive_ordinal": primitive_ordinal,
            "candidate_retained": candidate_hit,
            "target_primitive_retained": primitive_hit,
        })
      edge_rows.append({"edge_id": edge_id, "endpoints": endpoints})
    query_hits += int(query_ok)
    rows.append({
        "query_id": query_id,
        "target_assignment_and_primitives_retained": query_ok,
        "edges": edge_rows,
    })
  query_count = len(rows)
  return {
      "query_count": query_count,
      "endpoint_count": endpoint_count,
      "target_endpoint_candidate_retention": (
          endpoint_candidate_hits / endpoint_count if endpoint_count else 0.0
      ),
      "target_primitive_retention": (
          primitive_hits / endpoint_count if endpoint_count else 0.0
      ),
      "target_query_retention": query_hits / query_count if query_count else 0.0,
      "rows": rows,
  }


def ambiguity_summary_v1(audit: Mapping[str, Any]) -> dict[str, float | int]:
  raw = [int(row["raw_assignment_count"]) for row in audit["rows"]]
  feasible = [int(row["feasible_assignment_count"]) for row in audit["rows"]]
  pruning = [
      1.0 - kept / total if total else 0.0
      for kept, total in zip(feasible, raw, strict=True)
  ]
  return {
      "query_count": len(raw),
      "median_raw_assignment_count": median(raw) if raw else 0,
      "median_feasible_assignment_count": median(feasible) if feasible else 0,
      "median_pruning_ratio": median(pruning) if pruning else 0.0,
      "zero_feasible_query_count": sum(value == 0 for value in feasible),
  }


def _metric_vector(
    evaluation: Mapping[str, Any], metric: str,
) -> list[float]:
  if metric == "edge_part_and_primitive_top8_recall_at_25":
    values = []
    for row in evaluation["per_query"]:
      edges = [
          edge for edge in row["edges"]
          if edge.get("primitive_supervised") is True
      ]
      values.append(
          sum(bool(edge["edge_part_and_primitive_top8_hit"]) for edge in edges)
          / len(edges) if edges else 0.0
      )
    return values
  key = {
      "top1_strict_part_accuracy": "top1_strict_part",
      "assignment_recall_at_25": "assignment_hit",
  }.get(metric)
  if key is None:
    raise ValueError("LinkCAD contract-strength bootstrap metric differs")
  return [float(bool(row[key])) for row in evaluation["per_query"]]


def _percentile(values: Sequence[float], probability: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  index = min(len(ordered) - 1, max(0, math.floor(probability * len(ordered))))
  return float(ordered[index])


def paired_seed_median_bootstrap_v1(
    *, learned: Sequence[Mapping[str, Any]], symbolic: Mapping[str, Any],
    metric: str, samples: int = 10_000, seed: int = 20260823,
) -> dict[str, Any]:
  """Query-paired CI for the median of frozen learned seeds vs symbolic."""

  learned_vectors = [_metric_vector(row, metric) for row in learned]
  symbolic_vector = _metric_vector(symbolic, metric)
  count = len(symbolic_vector)
  if count == 0 or any(len(row) != count for row in learned_vectors):
    raise ValueError("LinkCAD contract-strength paired domain differs")
  point = median(
      sum(vector) / count for vector in learned_vectors
  ) - sum(symbolic_vector) / count
  rng = random.Random(seed)
  deltas = []
  for _ in range(samples):
    indices = [rng.randrange(count) for _ in range(count)]
    learned_value = median(
        sum(vector[index] for index in indices) / count
        for vector in learned_vectors
    )
    symbolic_value = sum(symbolic_vector[index] for index in indices) / count
    deltas.append(learned_value - symbolic_value)
  return {
      "metric": metric,
      "query_count": count,
      "learned_seed_count": len(learned_vectors),
      "bootstrap_samples": samples,
      "point_difference": point,
      "ci95": [_percentile(deltas, 0.025), _percentile(deltas, 0.975)],
  }


def paired_seed_median_comparison_bootstrap_v1(
    *, treatment: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]], metric: str,
    samples: int = 10_000, seed: int = 20260823,
) -> dict[str, Any]:
  """Query-paired CI for two equally replicated learned training regimes."""

  treatment_vectors = [_metric_vector(row, metric) for row in treatment]
  control_vectors = [_metric_vector(row, metric) for row in control]
  if len(treatment_vectors) != len(control_vectors) or not treatment_vectors:
    raise ValueError("LinkCAD contract-strength treatment/control seeds differ")
  count = len(treatment_vectors[0])
  if count == 0 or any(
      len(row) != count for row in treatment_vectors + control_vectors
  ):
    raise ValueError("LinkCAD contract-strength learned paired domain differs")

  def method_value(vectors: Sequence[Sequence[float]], indices: Sequence[int]):
    return median(
        sum(vector[index] for index in indices) / len(indices)
        for vector in vectors
    )

  complete = list(range(count))
  point = method_value(treatment_vectors, complete) - method_value(
      control_vectors, complete,
  )
  rng = random.Random(seed)
  deltas = []
  for _ in range(samples):
    indices = [rng.randrange(count) for _ in range(count)]
    deltas.append(
        method_value(treatment_vectors, indices)
        - method_value(control_vectors, indices)
    )
  return {
      "metric": metric, "query_count": count,
      "seed_count_per_method": len(treatment_vectors),
      "bootstrap_samples": samples, "point_difference": point,
      "ci95": [_percentile(deltas, 0.025), _percentile(deltas, 0.975)],
  }


__all__ = [
    "SCHEMA_VERSION", "ambiguity_summary_v1",
    "paired_seed_median_comparison_bootstrap_v1",
    "paired_seed_median_bootstrap_v1", "target_retention_audit_v1",
]
