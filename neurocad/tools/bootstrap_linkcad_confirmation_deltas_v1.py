"""Compute paired assembly-clustered CIs for LinkCAD confirmation deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from neurocad.linkcad_confirmation_statistics_v1 import (
    METRICS,
    metric_value_v1,
    paired_multiseed_query_cluster_bootstrap_v1,
    paired_query_cluster_bootstrap_v1,
)


SCHEMA_VERSION = "linkcad_confirmation_paired_bootstrap.v1"


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_confirmation_bootstrap_v1(
    linkcad: dict[str, Any],
    baselines: dict[str, Any],
    *,
    resamples: int = 10_000,
    seed: int = 1701,
) -> dict[str, Any]:
  learned_runs = linkcad["per_seed"]
  baseline_runs = baselines["rows"]
  if len(learned_runs) < 2 or not baseline_runs:
    raise ValueError("paired confirmation bootstrap roster differs")
  results = {}
  for metric in METRICS:
    baseline_points = {
        row["name"]: metric_value_v1(row["per_query"], metric)
        for row in baseline_runs
    }
    strongest_name = max(
        sorted(baseline_points), key=lambda name: baseline_points[name]
    )
    strongest = next(
        row for row in baseline_runs if row["name"] == strongest_name
    )
    per_seed = [
        {
            "checkpoint_sha256": row["checkpoint_sha256"],
            **paired_query_cluster_bootstrap_v1(
                row["per_query"], strongest["per_query"], metric=metric,
                resamples=resamples, seed=seed,
            ),
        }
        for row in learned_runs
    ]
    aggregate = paired_multiseed_query_cluster_bootstrap_v1(
        [row["per_query"] for row in learned_runs],
        strongest["per_query"], metric=metric,
        resamples=resamples, seed=seed,
    )
    results[metric] = {
        "strongest_baseline": strongest_name,
        "strongest_baseline_point_estimate": baseline_points[strongest_name],
        "all_baseline_point_estimates": baseline_points,
        "per_optimizer_seed": per_seed,
        "three_seed_median": aggregate,
        "three_seed_ci_lower_gt_zero": (
            aggregate["percentile_ci_95"][0] > 0.0
        ),
        "all_optimizer_seed_ci_lower_gt_zero": all(
            row["percentile_ci_95"][0] > 0.0 for row in per_seed
        ),
    }
  return {
      "schema_version": SCHEMA_VERSION,
      "scope": (
          "untouched_private_join_linkcad_vs_preunseal_frozen_baselines_"
          "with_postunseal_target_free_predictions"
      ),
      "cluster_unit": "source_assembly_one_query_per_assembly",
      "family_resampling": False,
      "family_resampling_reason": (
          "the confirmation protocol defines one independent source assembly "
          "per query and does not publish a separate family identity"
      ),
      "optimizer_seed_count": len(learned_runs),
      "baseline_count": len(baseline_runs),
      "resamples": resamples,
      "seed": seed,
      "metrics": results,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--linkcad-evaluation", type=Path, required=True)
  parser.add_argument("--baseline-evaluation", type=Path, required=True)
  parser.add_argument("--resamples", type=int, default=10_000)
  parser.add_argument("--seed", type=int, default=1701)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  linkcad = _read(args.linkcad_evaluation)
  baselines = _read(args.baseline_evaluation)
  payload = compute_confirmation_bootstrap_v1(
      linkcad, baselines, resamples=args.resamples, seed=args.seed
  )
  payload["linkcad_evaluation_sha256"] = _sha(args.linkcad_evaluation)
  payload["baseline_evaluation_sha256"] = _sha(args.baseline_evaluation)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
      metric: {
          "baseline": result["strongest_baseline"],
          "delta": result["three_seed_median"]["point_estimate_delta"],
          "ci95": result["three_seed_median"]["percentile_ci_95"],
      }
      for metric, result in payload["metrics"].items()
  }, sort_keys=True))


if __name__ == "__main__":
  main()
