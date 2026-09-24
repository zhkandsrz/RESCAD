"""Run a fixed-budget SPADA-style compile--test--repair comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from neurocad.linkcad_pure_llm_baseline_v1 import (
    build_direct_pose_request_v1,
    canonical_sha256,
    frozen_selection_v1,
)
from neurocad.linkcad_pure_llm_execution_v1 import (
    execute_and_export_pure_llm_v1,
)
from neurocad.linkcad_spada_style_baseline_v1 import (
    DEFAULT_MODEL,
    FrozenSPADAStyleRepairAgentV1,
    prompt_sha256_v1,
    public_test_report_v1,
)


def _read(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  temporary.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--frozen-rankings", type=Path, required=True)
  parser.add_argument("--initial-predictions", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--max-repair-rounds", type=int, default=2)
  parser.add_argument("--timeout-seconds", type=int, default=240)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--limit", type=int)
  args = parser.parse_args()
  if args.max_repair_rounds < 0:
    raise ValueError("SPADA-style repair budget must be non-negative")

  public = _read(args.public)
  rankings = _read(args.frozen_rankings)["rankings"]
  initial = _read(args.initial_predictions)
  if public.get("contains_private_targets") is not False:
    raise ValueError("SPADA-style public scope differs")
  queries = list(public["queries"])
  if args.limit is not None:
    queries = queries[:args.limit]
  args.output_root.mkdir(parents=True, exist_ok=True)
  progress_path = args.output_root / "progress.json"
  progress = {
      "schema_version": "linkcad_spada_style_progress.v1",
      "prompt_sha256": prompt_sha256_v1(),
      "max_repair_rounds": args.max_repair_rounds,
      "predictions": {}, "rows": {}, "receipts": {}, "failures": {},
      "terminal": False,
  }
  if progress_path.exists():
    if not args.resume:
      raise ValueError("SPADA-style progress exists; pass --resume")
    progress = _read(progress_path)
    if (
        progress.get("prompt_sha256") != prompt_sha256_v1()
        or int(progress.get("max_repair_rounds", -1)) != args.max_repair_rounds
    ):
      raise ValueError("SPADA-style resume contract differs")

  agent = FrozenSPADAStyleRepairAgentV1(
      model=DEFAULT_MODEL, timeout_seconds=args.timeout_seconds,
  )
  started = time.perf_counter()
  for index, query in enumerate(queries):
    query_id = str(query["query_id"])
    if query_id in progress["rows"] or query_id in progress["failures"]:
      continue
    if query_id not in initial.get("predictions", {}):
      progress["failures"][query_id] = {
          "status": "no_valid_initial_prediction",
          "source_failure": initial.get("failures", {}).get(query_id),
      }
      _write(progress_path, progress)
      continue
    selection = frozen_selection_v1(query, rankings[query_id])
    request = build_direct_pose_request_v1(
        query=query, candidate_by_role=selection,
        dataset_root=args.dataset_root.resolve(),
    )
    prediction = initial["predictions"][query_id]
    round_receipts = []
    final_audit = None
    try:
      for repair_round in range(args.max_repair_rounds + 1):
        output = args.output_root / "exports" / query_id
        final_audit = execute_and_export_pure_llm_v1(
            query=query, prediction=prediction,
            dataset_root=args.dataset_root.resolve(),
            output_step=output / "assembly.step",
            output_manifest=output / "assembly.manifest.json",
        )
        manifest = _read(output / "assembly.manifest.json")
        report = public_test_report_v1(manifest)
        round_receipts.append({
            "round": repair_round,
            "prediction_sha256": canonical_sha256(prediction),
            "test_report": report,
            "kernel_feasible": final_audit["kernel_feasible"],
        })
        if final_audit["kernel_feasible"] or repair_round == args.max_repair_rounds:
          break
        repaired = agent.repair(
            request=request, previous_prediction=prediction,
            test_report=report, repair_round=repair_round + 1,
        )
        prediction = repaired.prediction
        round_receipts[-1]["repair_call_receipt"] = repaired.receipt
    except Exception as error:
      progress["failures"][query_id] = {
          "status": "repair_or_execution_failure",
          "error_type": type(error).__name__, "message": str(error),
          "completed_rounds": round_receipts,
      }
    else:
      progress["predictions"][query_id] = prediction
      progress["rows"][query_id] = final_audit
      progress["receipts"][query_id] = round_receipts
    _write(progress_path, progress)
    completed = len(progress["rows"]) + len(progress["failures"])
    print(json.dumps({
        "completed": completed, "total": len(queries),
        "kernel_feasible": sum(
            row.get("kernel_feasible") is True
            for row in progress["rows"].values()
        ),
        "failures": len(progress["failures"]), "query_index": index,
    }, sort_keys=True), flush=True)

  domain = {str(row["query_id"]) for row in queries}
  if set(progress["rows"]) | set(progress["failures"]) != domain:
    raise RuntimeError("SPADA-style run did not reach every query")
  rows = []
  for query_id in sorted(domain):
    if query_id in progress["rows"]:
      rows.append(progress["rows"][query_id])
    else:
      rows.append({
          "query_id": query_id, "status": "no_valid_final_prediction",
          "raw_collision_free": False, "all_connections_satisfied": False,
          "kernel_feasible": False,
          "failure": progress["failures"][query_id],
      })
  result = {
      "schema_version": "linkcad_spada_style_execution_batch.v1",
      "method_id": "spada_style_compile_test_repair",
      "query_count": len(rows),
      "max_repair_rounds": args.max_repair_rounds,
      "raw_collision_free_rate": sum(
          row.get("raw_collision_free") is True for row in rows
      ) / len(rows),
      "all_connections_satisfied_rate": sum(
          row.get("all_connections_satisfied") is True for row in rows
      ) / len(rows),
      "kernel_feasible_rate": sum(
          row.get("kernel_feasible") is True for row in rows
      ) / len(rows),
      "private_targets_opened": False,
      "rows": rows,
  }
  result["payload_sha256"] = canonical_sha256(result)
  _write(args.output_root / "execution_batch.json", result)
  _write(args.output_root / "final_predictions.json", {
      "schema_version": "linkcad_spada_style_final_predictions.v1",
      "method_id": result["method_id"], "query_count": len(rows),
      "predictions": progress["predictions"],
      "failures": progress["failures"], "private_targets_opened": False,
  })
  progress["terminal"] = True
  _write(progress_path, progress)
  _write(args.output_root / "run_receipt.json", {
      "schema_version": "linkcad_spada_style_run_receipt.v1",
      "query_count": len(rows), "max_repair_rounds": args.max_repair_rounds,
      "kernel_feasible_count": sum(
          row.get("kernel_feasible") is True for row in rows
      ),
      "failed_query_count": len(progress["failures"]),
      "wall_time_seconds": time.perf_counter() - started,
      "credential_material_included": False,
      "private_targets_opened": False,
  })
  print(json.dumps({
      "query_count": len(rows),
      "kernel_feasible_rate": result["kernel_feasible_rate"],
      "failed_query_count": len(progress["failures"]),
  }, sort_keys=True))


if __name__ == "__main__":
  main()
