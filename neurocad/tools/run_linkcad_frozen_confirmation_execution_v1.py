"""Execute a deterministic subset of frozen LinkCAD confirmation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from neurocad.linkcad_kinematic_execution_v2 import (
    execute_public_prediction_subset_v2,
)


SCHEMA_VERSION = "linkcad_confirmation_execution_runtime_receipt.v1"


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  position = (len(ordered) - 1) * probability
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--predictions", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--query-limit", type=int, default=25)
  parser.add_argument("--hypotheses-per-query", type=int, default=1)
  parser.add_argument("--alternatives-per-edge", type=int, default=3)
  parser.add_argument("--maximum-attempts-per-query", type=int, default=25)
  parser.add_argument("--enable-axial-seating", action="store_true")
  parser.add_argument("--maximum-accepted-poses-per-edge", type=int, default=1)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  public = _read(args.public)
  predictions = _read(args.predictions)
  started = time.monotonic()
  execution = execute_public_prediction_subset_v2(
      public=public,
      predictions=predictions,
      dataset_root=args.dataset_root,
      query_limit=args.query_limit,
      hypotheses_per_query=args.hypotheses_per_query,
      alternatives_per_edge=args.alternatives_per_edge,
      maximum_attempts_per_query=args.maximum_attempts_per_query,
      enable_axial_seating=args.enable_axial_seating,
      maximum_accepted_poses_per_edge=args.maximum_accepted_poses_per_edge,
  )
  wall_seconds = time.monotonic() - started
  query_runtime: dict[str, float] = {}
  for row in execution["rows"]:
    query_id = str(row["query_id"])
    query_runtime[query_id] = query_runtime.get(query_id, 0.0) + float(
        row["elapsed_seconds"]
    )
  runtimes = list(query_runtime.values())
  payload = {
      "schema_version": SCHEMA_VERSION,
      "scope": (
          "deterministic_query_id_ordered_subset_of_preunseal_frozen_"
          "predictions_executed_without_reading_private_targets"
      ),
      "public_sha256": _sha(args.public),
      "prediction_file_sha256": _sha(args.predictions),
      "prediction_payload_sha256": predictions["prediction_payload_sha256"],
      "checkpoint_sha256": predictions["checkpoint_sha256"],
      "private_targets_read_by_executor": False,
      "query_selection_uses_targets": False,
      "query_selection_policy": "query_id_ascending_first_n",
      "selected_query_count": len(query_runtime),
      "runtime": {
          "wall_seconds": wall_seconds,
          "query_seconds_sum_of_pair_attempts": {
              "p50": _percentile(runtimes, 0.50),
              "p95": _percentile(runtimes, 0.95),
              "mean": sum(runtimes) / len(runtimes) if runtimes else 0.0,
              "maximum": max(runtimes) if runtimes else 0.0,
          },
      },
      "execution": execution,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
      "selected_query_count": payload["selected_query_count"],
      "edge_target_count": execution["edge_target_count"],
      "accepted_edge_count": execution["accepted_edge_count"],
      "execution_attempt_count": execution["execution_attempt_count"],
      "wall_seconds": wall_seconds,
      "query_runtime_p50": payload["runtime"][
          "query_seconds_sum_of_pair_attempts"
      ]["p50"],
      "query_runtime_p95": payload["runtime"][
          "query_seconds_sum_of_pair_attempts"
      ]["p95"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
