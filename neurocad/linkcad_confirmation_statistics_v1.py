"""Paired query-cluster statistics for LinkCAD confirmation evidence."""

from __future__ import annotations

import math
import random
from statistics import median
from typing import Any, Mapping, Sequence


METRICS = (
    "assignment_recall_at_25",
    "joint_mobility_recall_at_25",
    "support_mobility_program_recall_at_25",
    "top1_strict_part_accuracy",
    "top1_strict_joint_mobility_accuracy",
    "top1_strict_support_mobility_program_accuracy",
    "edge_part_pair_recall_at_25",
    "edge_part_and_primitive_top1_recall_at_25",
    "edge_part_and_primitive_top8_recall_at_25",
    "primitive_top1_given_edge_part_hit",
    "primitive_top8_given_edge_part_hit",
    "full_primitive_top1_program_recall_at_25",
    "full_primitive_top8_program_recall_at_25",
)


def _percentile(values: Sequence[float], probability: float) -> float:
  if not values or not 0.0 <= probability <= 1.0:
    raise ValueError("percentile arguments differ")
  ordered = sorted(float(value) for value in values)
  position = (len(ordered) - 1) * probability
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _ratio(numerator: int | float, denominator: int | float) -> float:
  return float(numerator) / float(denominator) if denominator else 0.0


def _query_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
  result: dict[str, Mapping[str, Any]] = {}
  for row in rows:
    query_id = str(row.get("query_id", ""))
    if not query_id or query_id in result:
      raise ValueError("query identities must be nonempty and unique")
    result[query_id] = row
  if not result:
    raise ValueError("confirmation statistics require query rows")
  return result


def metric_counts_v1(
    rows: Sequence[Mapping[str, Any]], metric: str,
) -> tuple[float, float]:
  """Return the additive numerator and denominator for a frozen metric."""

  if metric not in METRICS:
    raise ValueError(f"unsupported LinkCAD confirmation metric: {metric}")
  if metric == "assignment_recall_at_25":
    return sum(bool(row["assignment_hit"]) for row in rows), len(rows)
  if metric == "joint_mobility_recall_at_25":
    return sum(bool(row["joint_mobility_hit"]) for row in rows), len(rows)
  if metric == "support_mobility_program_recall_at_25":
    return sum(
        bool(row["support_mobility_program_hit"]) for row in rows
    ), len(rows)
  if metric == "top1_strict_part_accuracy":
    return sum(bool(row["top1_strict_part"]) for row in rows), len(rows)
  if metric == "top1_strict_joint_mobility_accuracy":
    return sum(
        bool(row["top1_strict_joint_mobility"]) for row in rows
    ), len(rows)
  if metric == "top1_strict_support_mobility_program_accuracy":
    return sum(
        bool(row["top1_strict_support_mobility_program"]) for row in rows
    ), len(rows)
  if metric in {
      "full_primitive_top1_program_recall_at_25",
      "full_primitive_top8_program_recall_at_25",
  }:
    eligible = [row for row in rows if bool(row["primitive_complete"])]
    hit_field = (
        "full_primitive_top1_program_hit"
        if metric == "full_primitive_top1_program_recall_at_25"
        else "full_primitive_top8_program_hit"
    )
    return sum(
        bool(row[hit_field]) for row in eligible
    ), len(eligible)

  edges = [edge for row in rows for edge in row["edges"]]
  if metric == "edge_part_pair_recall_at_25":
    return sum(bool(edge["edge_part_hit"]) for edge in edges), len(edges)
  supervised = [edge for edge in edges if bool(edge["primitive_supervised"])]
  if metric == "edge_part_and_primitive_top1_recall_at_25":
    return sum(
        bool(edge["edge_part_and_primitive_top1_hit"])
        for edge in supervised
    ), len(supervised)
  if metric == "edge_part_and_primitive_top8_recall_at_25":
    return sum(
        bool(edge["edge_part_and_primitive_top8_hit"])
        for edge in supervised
    ), len(supervised)
  supervised_part_hits = [
      edge for edge in supervised if bool(edge["edge_part_hit"])
  ]
  hit_field = (
      "edge_part_and_primitive_top1_hit"
      if metric == "primitive_top1_given_edge_part_hit"
      else "edge_part_and_primitive_top8_hit"
  )
  return sum(bool(edge[hit_field]) for edge in supervised_part_hits), len(
      supervised_part_hits
  )


def metric_value_v1(
    rows: Sequence[Mapping[str, Any]], metric: str,
) -> float:
  numerator, denominator = metric_counts_v1(rows, metric)
  return _ratio(numerator, denominator)


def paired_query_cluster_bootstrap_v1(
    learned_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    resamples: int = 10_000,
    seed: int = 1701,
) -> dict[str, Any]:
  """Resample paired assemblies, represented by one query per assembly."""

  if type(resamples) is not int or resamples < 1 or type(seed) is not int:
    raise ValueError("bootstrap controls differ")
  learned = _query_index(learned_rows)
  baseline = _query_index(baseline_rows)
  if set(learned) != set(baseline):
    raise ValueError("paired query identities differ")
  query_ids = tuple(sorted(learned))
  point = metric_value_v1(list(learned.values()), metric) - metric_value_v1(
      list(baseline.values()), metric
  )
  rng = random.Random(seed)
  deltas: list[float] = []
  for _ in range(resamples):
    sample = [query_ids[rng.randrange(len(query_ids))] for _ in query_ids]
    learned_sample = [learned[query_id] for query_id in sample]
    baseline_sample = [baseline[query_id] for query_id in sample]
    deltas.append(
        metric_value_v1(learned_sample, metric)
        - metric_value_v1(baseline_sample, metric)
    )
  return {
      "metric": metric,
      "cluster_unit": "source_assembly_one_query_per_assembly",
      "query_cluster_count": len(query_ids),
      "resamples": resamples,
      "seed": seed,
      "point_estimate_delta": point,
      "percentile_ci_95": [
          _percentile(deltas, 0.025),
          _percentile(deltas, 0.975),
      ],
      "p_delta_gt_0": sum(delta > 0.0 for delta in deltas) / resamples,
  }


def paired_multiseed_query_cluster_bootstrap_v1(
    learned_seed_rows: Sequence[Sequence[Mapping[str, Any]]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    resamples: int = 10_000,
    seed: int = 1701,
) -> dict[str, Any]:
  """Bootstrap the median-across-seeds estimator against one baseline."""

  if not learned_seed_rows:
    raise ValueError("multi-seed bootstrap requires learned runs")
  baseline = _query_index(baseline_rows)
  learned = [_query_index(rows) for rows in learned_seed_rows]
  query_ids = tuple(sorted(baseline))
  if any(set(rows) != set(baseline) for rows in learned):
    raise ValueError("paired query identities differ")
  point = median(
      metric_value_v1(list(rows.values()), metric) for rows in learned
  ) - metric_value_v1(list(baseline.values()), metric)
  rng = random.Random(seed)
  deltas: list[float] = []
  for _ in range(resamples):
    sample = [query_ids[rng.randrange(len(query_ids))] for _ in query_ids]
    learned_values = [
        metric_value_v1([rows[query_id] for query_id in sample], metric)
        for rows in learned
    ]
    baseline_value = metric_value_v1(
        [baseline[query_id] for query_id in sample], metric
    )
    deltas.append(median(learned_values) - baseline_value)
  return {
      "metric": metric,
      "estimator": "median_across_optimizer_seeds",
      "optimizer_seed_count": len(learned),
      "cluster_unit": "source_assembly_one_query_per_assembly",
      "query_cluster_count": len(query_ids),
      "resamples": resamples,
      "seed": seed,
      "point_estimate_delta": point,
      "percentile_ci_95": [
          _percentile(deltas, 0.025),
          _percentile(deltas, 0.975),
      ],
      "p_delta_gt_0": sum(delta > 0.0 for delta in deltas) / resamples,
  }


def paired_two_multiseed_query_cluster_bootstrap_v1(
    learned_seed_rows: Sequence[Sequence[Mapping[str, Any]]],
    baseline_seed_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    metric: str,
    resamples: int = 10_000,
    seed: int = 1701,
) -> dict[str, Any]:
  """Bootstrap a median-across-seeds difference for two frozen methods."""

  if not learned_seed_rows or not baseline_seed_rows:
    raise ValueError("two-method multi-seed bootstrap requires both rosters")
  learned = [_query_index(rows) for rows in learned_seed_rows]
  baseline = [_query_index(rows) for rows in baseline_seed_rows]
  query_ids = tuple(sorted(learned[0]))
  expected = set(query_ids)
  if any(set(rows) != expected for rows in (*learned, *baseline)):
    raise ValueError("paired query identities differ")
  learned_point = median(
      metric_value_v1(list(rows.values()), metric) for rows in learned
  )
  baseline_point = median(
      metric_value_v1(list(rows.values()), metric) for rows in baseline
  )
  rng = random.Random(seed)
  deltas: list[float] = []
  for _ in range(resamples):
    sample = [query_ids[rng.randrange(len(query_ids))] for _ in query_ids]
    learned_value = median(
        metric_value_v1([rows[query_id] for query_id in sample], metric)
        for rows in learned
    )
    baseline_value = median(
        metric_value_v1([rows[query_id] for query_id in sample], metric)
        for rows in baseline
    )
    deltas.append(learned_value - baseline_value)
  return {
      "metric": metric,
      "estimator": "difference_of_medians_across_optimizer_seeds",
      "learned_optimizer_seed_count": len(learned),
      "baseline_optimizer_seed_count": len(baseline),
      "cluster_unit": "source_assembly_one_query_per_assembly",
      "query_cluster_count": len(query_ids),
      "resamples": resamples,
      "seed": seed,
      "learned_point_estimate": learned_point,
      "baseline_point_estimate": baseline_point,
      "point_estimate_delta": learned_point - baseline_point,
      "percentile_ci_95": [
          _percentile(deltas, 0.025), _percentile(deltas, 0.975),
      ],
      "p_delta_gt_0": sum(delta > 0.0 for delta in deltas) / resamples,
  }


__all__ = [
    "METRICS",
    "metric_counts_v1",
    "metric_value_v1",
    "paired_multiseed_query_cluster_bootstrap_v1",
    "paired_two_multiseed_query_cluster_bootstrap_v1",
    "paired_query_cluster_bootstrap_v1",
]
