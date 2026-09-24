"""Join frozen global LinkCAD execution with staged private evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def _read(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--staged-evaluation", type=Path, required=True)
  parser.add_argument("--global-preunseal-receipt", type=Path, required=True)
  parser.add_argument("--global-execution", type=Path, nargs=3, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  staged = _read(args.staged_evaluation)
  preunseal = _read(args.global_preunseal_receipt)
  if (
      preunseal.get("private_targets_opened") is not False
      or staged.get("staged_preunseal_receipt_sha256")
      != _sha(args.global_preunseal_receipt)
  ):
    raise ValueError("LinkCAD global private chronology differs")
  expected = {
      int(row["seed"]): row for row in preunseal["rows"]
  }
  staged_by_seed = {int(row["seed"]): row for row in staged["per_seed"]}
  if set(expected) != set(staged_by_seed) or len(args.global_execution) != 3:
    raise ValueError("LinkCAD global seed roster differs")
  observed = {}
  for path in args.global_execution:
    digest = _sha(path)
    seeds = [
        seed for seed, row in expected.items()
        if row["global_execution_file_sha256"] == digest
    ]
    if len(seeds) != 1:
      raise ValueError("LinkCAD global receipt file binding differs")
    observed[seeds[0]] = (path, _read(path))
  if set(observed) != set(expected):
    raise ValueError("LinkCAD global execution domain differs")

  per_seed = []
  for seed in sorted(expected):
    _, wrapper = observed[seed]
    execution = wrapper["execution"]
    stage = staged_by_seed[seed]
    if (
        wrapper.get("private_targets_read_by_executor") is not False
        or execution.get("private_targets_opened") is not False
        or execution.get("global_execution_payload_sha256")
        != expected[seed]["global_execution_payload_sha256"]
        or execution.get("query_count") != staged.get("query_count")
    ):
      raise ValueError("LinkCAD global execution payload differs")
    global_by_id = {
        str(row["query_id"]): row for row in execution["rows"]
    }
    exact_by_id = {
        str(row["query_id"]): row for row in stage["exact_per_query"]
    }
    prediction_by_id = {
        str(row["query_id"]): row for row in stage["prediction_per_query"]
    }
    if set(global_by_id) != set(exact_by_id) or set(global_by_id) != set(
        prediction_by_id
    ):
      raise ValueError("LinkCAD global per-query domain differs")
    per_query = []
    for query_id in sorted(global_by_id):
      global_row = global_by_id[query_id]
      exact_row = exact_by_id[query_id]
      prediction_row = prediction_by_id[query_id]
      globally_feasible = (
          global_row.get("global_assembly_conditionally_feasible") is True
      )
      intended_part = all(
          edge.get("intended_part_edge_top1") is True
          for edge in exact_row["edges"]
      )
      intended_program = all(
          edge.get("intended_support_mobility_top1") is True
          for edge in exact_row["edges"]
      )
      intended_primitive = (
          exact_row.get("intended_full_query_conditionally_feasible_top3")
          is True
      )
      per_query.append({
          "query_id": query_id,
          "global_assembly_conditionally_feasible": globally_feasible,
          "global_intended_part_success": globally_feasible and intended_part,
          "global_intended_support_mobility_success": (
              globally_feasible and intended_part and intended_program
          ),
          "global_intended_exact_success": (
              globally_feasible and intended_primitive
          ),
          "strict_part_and_global_success": (
              globally_feasible
              and prediction_row.get("top1_strict_part") is True
          ),
          "strict_program_and_global_success": (
              globally_feasible
              and prediction_row.get(
                  "top1_strict_support_mobility_program"
              ) is True
          ),
          "global_status": global_row["status"],
      })
    metric_names = (
        "global_assembly_conditionally_feasible",
        "global_intended_part_success",
        "global_intended_support_mobility_success",
        "global_intended_exact_success",
        "strict_part_and_global_success",
        "strict_program_and_global_success",
    )
    metrics = {
        name: sum(row[name] is True for row in per_query) / len(per_query)
        for name in metric_names
    }
    per_seed.append({
        "seed": seed,
        "checkpoint_sha256": stage["checkpoint_sha256"],
        "status_counts": execution["status_counts"],
        "metrics": metrics,
        "per_query": per_query,
    })
  metric_names = tuple(per_seed[0]["metrics"])
  aggregate = {
      name: {
          "median": median(row["metrics"][name] for row in per_seed),
          "minimum": min(row["metrics"][name] for row in per_seed),
          "maximum": max(row["metrics"][name] for row in per_seed),
      }
      for name in metric_names
  }
  payload = {
      "schema_version": "linkcad_global_decoder_private_evaluation.v1",
      "scope": "private_join_after_global_decoder_preunseal_closure",
      "query_count": staged["query_count"],
      "seed_count": len(per_seed),
      "staged_evaluation_sha256": _sha(args.staged_evaluation),
      "global_preunseal_receipt_sha256": _sha(
          args.global_preunseal_receipt
      ),
      "per_seed": per_seed,
      "aggregate": aggregate,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
      name: row["median"] for name, row in aggregate.items()
  }, sort_keys=True))


if __name__ == "__main__":
  main()
