"""Compose LinkCAD Top-1 pair programs and audit the complete assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from neurocad.linkcad_global_assembly_execution_v1 import (
    execute_global_assembly_subset_v1,
)


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  position = (len(ordered) - 1) * probability
  lower, upper = math.floor(position), math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--predictions", type=Path, required=True)
  parser.add_argument("--pair-execution", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--query-limit", type=int)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  public = _read(args.public)
  predictions = _read(args.predictions)
  pair_execution = _read(args.pair_execution)
  started = time.monotonic()
  execution = execute_global_assembly_subset_v1(
      public=public, predictions=predictions,
      pair_execution=pair_execution, dataset_root=args.dataset_root,
      query_limit=args.query_limit,
  )
  wall_seconds = time.monotonic() - started
  runtimes = [float(row["elapsed_seconds"]) for row in execution["rows"]]
  payload = {
      "schema_version": "linkcad_global_assembly_runtime_receipt.v1",
      "scope": "target_free_top1_global_pose_and_all_pair_collision_audit",
      "public_sha256": _sha(args.public),
      "prediction_file_sha256": _sha(args.predictions),
      "pair_execution_file_sha256": _sha(args.pair_execution),
      "private_targets_read_by_executor": False,
      "runtime": {
          "wall_seconds": wall_seconds,
          "query_seconds": {
              "p50": _percentile(runtimes, 0.5),
              "p95": _percentile(runtimes, 0.95),
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
      "query_count": execution["query_count"],
      "accepted_query_count": execution["accepted_query_count"],
      "status_counts": execution["status_counts"],
      "wall_seconds": wall_seconds,
      "query_p50": payload["runtime"]["query_seconds"]["p50"],
      "query_p95": payload["runtime"]["query_seconds"]["p95"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
