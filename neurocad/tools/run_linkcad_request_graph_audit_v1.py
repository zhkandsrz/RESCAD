"""Run the frozen whole-request LinkCAD graph compiler and its component audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from neurocad.linkcad_language_planner_v1 import DEFAULT_MODEL, canonical_sha256
from neurocad.linkcad_request_graph_compiler_v1 import (
    FrozenLinkCADRequestGraphCompilerV1,
    PREDICTION_SCHEMA_VERSION,
    build_request_graph_benchmark_v1,
    evaluate_request_graph_compiler_v1,
    materialize_compiler_public_v1,
    prompt_sha256_v1,
)


def _write(path: Path, payload) -> None:
  staging = path.with_suffix(path.suffix + ".tmp")
  staging.write_text(
      json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  staging.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--split-name", required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--timeout-seconds", type=int, default=240)
  parser.add_argument("--max-queries", type=int)
  args = parser.parse_args()
  if args.batch_size < 1:
    raise ValueError("LinkCAD request-graph batch size differs")

  source_bytes = args.public.read_bytes()
  public = json.loads(source_bytes.decode("utf-8"))
  inputs, targets, summary = build_request_graph_benchmark_v1(
      public, split_name=args.split_name,
  )
  items = inputs["items"]
  if args.max_queries is not None:
    items = items[:args.max_queries]
    selected = {row["query_id"] for row in items}
    targets = {
        **targets,
        "item_count": len(items),
        "items": [row for row in targets["items"] if row["query_id"] in selected],
    }

  args.output_root.mkdir(parents=True, exist_ok=True)
  precommit = {
      "schema_version": "linkcad_request_graph_precommit.v1",
      "scope": "frozen_whole_request_frontend_audit",
      "source_public_sha256": hashlib.sha256(source_bytes).hexdigest(),
      "input_sha256": canonical_sha256({**inputs, "items": items}),
      "reference_target_sha256": canonical_sha256(targets),
      "prompt_sha256": prompt_sha256_v1(),
      "provider": "deepseek", "model": DEFAULT_MODEL, "temperature": 0.0,
      "reference_not_used_by_prediction_runner": True,
      "candidate_or_reference_graph_identity_exposed": False,
      "failed_queries_remain_in_evaluation_denominator": True,
  }
  _write(args.output_root / "inputs.json", {**inputs, "items": items,
                                             "item_count": len(items)})
  _write(args.output_root / "targets.reference.json", targets)
  _write(args.output_root / "benchmark_summary.json", summary)
  _write(args.output_root / "precommit.json", precommit)

  compiler = FrozenLinkCADRequestGraphCompilerV1(
      model=DEFAULT_MODEL, timeout_seconds=args.timeout_seconds,
  )
  predictions = []
  receipts = []
  errors = []
  started = time.perf_counter()

  def compile_batch(batch, batch_start):
    """Keep valid requests when one response in a batch fails validation."""
    try:
      result = compiler.compile(batch)
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
      if len(batch) > 1:
        for offset, item in enumerate(batch):
          compile_batch([item], batch_start + offset)
        return
      errors.append({
          "batch_start": batch_start,
          "query_ids": [row["query_id"] for row in batch],
          "terminal_status": "compiler_error",
          "error_type": type(error).__name__,
          "message": str(error),
      })
      return
    predictions.extend(result.predictions)
    receipts.append({"batch_start": batch_start, **result.receipt})

  for start in range(0, len(items), args.batch_size):
    batch = items[start:start + args.batch_size]
    compile_batch(batch, start)
    progress = {
        "schema_version": "linkcad_request_graph_progress.v1",
        "requested_query_count": len(items),
        "processed_query_count": min(start + len(batch), len(items)),
        "predictions": predictions,
        "errors": errors,
        "terminal": False,
    }
    _write(args.output_root / "progress.json", progress)
    print(json.dumps({
        "completed": progress["processed_query_count"],
        "total": len(items), "errors": len(errors),
    }, sort_keys=True), flush=True)

  prediction_payload = {
      "schema_version": PREDICTION_SCHEMA_VERSION,
      "scope": "frozen_whole_request_graph_predictions",
      "provider": "deepseek", "model": DEFAULT_MODEL, "temperature": 0.0,
      "prompt_sha256": prompt_sha256_v1(),
      "requested_query_count": len(items),
      "prediction_count": len(predictions),
      "items": predictions,
  }
  evaluation = evaluate_request_graph_compiler_v1(targets, prediction_payload)
  conditioned_public = materialize_compiler_public_v1(public, prediction_payload)
  receipt = {
      "schema_version": "linkcad_request_graph_run_receipt.v1",
      "prediction_sha256": canonical_sha256(prediction_payload),
      "successful_batch_count": len(receipts),
      "failed_batch_count": len(errors),
      "wall_time_seconds": time.perf_counter() - started,
      "batch_receipts": receipts,
      "errors": errors,
      "credential_material_included": False,
  }
  _write(args.output_root / "predictions.json", prediction_payload)
  _write(args.output_root / "evaluation.json", evaluation)
  _write(args.output_root / "public.compiler_conditioned.json", conditioned_public)
  _write(args.output_root / "run_receipt.json", receipt)
  _write(args.output_root / "progress.json", {
      "schema_version": "linkcad_request_graph_progress.v1",
      "requested_query_count": len(items),
      "processed_query_count": len(items),
      "predictions": predictions, "errors": errors, "terminal": True,
  })
  print(json.dumps({
      key: evaluation[key] for key in (
          "query_count", "prediction_count", "compiler_failure_count",
          "role_set_exact_rate", "graph_exact_rate", "edge_f1",
          "full_request_exact_rate",
      )
  }, sort_keys=True))


if __name__ == "__main__":
  main()
