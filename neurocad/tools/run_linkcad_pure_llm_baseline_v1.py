"""Run the pure-LLM direct assembly baseline with resumable per-query calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from neurocad.linkcad_pure_llm_baseline_v1 import (
    DEFAULT_MODEL,
    FrozenPureLLMAssemblerV1,
    build_direct_pose_request_v1,
    canonical_sha256,
    frozen_selection_v1,
    prompt_sha256_v1,
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
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--timeout-seconds", type=int, default=240)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--limit", type=int)
  args = parser.parse_args()

  public = _read(args.public)
  ranking_payload = _read(args.frozen_rankings)
  if public.get("contains_private_targets") is not False:
    raise ValueError("Pure-LLM public scope differs")
  rankings = ranking_payload["rankings"]
  queries = list(public["queries"])
  if args.limit is not None:
    queries = queries[:args.limit]
  args.output_root.mkdir(parents=True, exist_ok=True)
  progress_path = args.output_root / "progress.json"
  progress = {
      "schema_version": "linkcad_pure_llm_progress.v1",
      "prompt_sha256": prompt_sha256_v1(),
      "predictions": {}, "receipts": {}, "failures": {}, "terminal": False,
  }
  if progress_path.exists():
    if not args.resume:
      raise ValueError("Pure-LLM progress exists; pass --resume")
    progress = _read(progress_path)
    if progress.get("prompt_sha256") != prompt_sha256_v1():
      raise ValueError("Pure-LLM resume prompt differs")

  agent = FrozenPureLLMAssemblerV1(
      model=DEFAULT_MODEL, timeout_seconds=args.timeout_seconds,
  )
  started = time.perf_counter()
  domain = {str(row["query_id"]) for row in queries}
  for index, query in enumerate(queries):
    query_id = str(query["query_id"])
    if query_id in progress["predictions"] or query_id in progress["failures"]:
      continue
    try:
      selection = frozen_selection_v1(query, rankings[query_id])
      request = build_direct_pose_request_v1(
          query=query, candidate_by_role=selection,
          dataset_root=args.dataset_root.resolve(),
      )
      result = agent.assemble(request)
    except Exception as error:
      progress["failures"][query_id] = {
          "error_type": type(error).__name__, "message": str(error),
      }
    else:
      progress["predictions"][query_id] = result.prediction
      progress["receipts"][query_id] = result.receipt
    _write(progress_path, progress)
    print(json.dumps({
        "completed": len(progress["predictions"]) + len(progress["failures"]),
        "total": len(queries), "valid": len(progress["predictions"]),
        "failures": len(progress["failures"]), "query_index": index,
    }, sort_keys=True), flush=True)

  if set(progress["predictions"]) | set(progress["failures"]) != domain:
    raise RuntimeError("Pure-LLM run did not reach every query")
  payload = {
      "schema_version": "linkcad_pure_llm_batch.v1",
      "method_id": "pure_llm_direct_pose",
      "provider": "deepseek", "model": DEFAULT_MODEL,
      "temperature": 0.0, "prompt_sha256": prompt_sha256_v1(),
      "query_count": len(queries),
      "valid_prediction_count": len(progress["predictions"]),
      "failed_prediction_count": len(progress["failures"]),
      "private_targets_opened": False,
      "predictions": progress["predictions"],
      "failures": progress["failures"],
      "receipts": progress["receipts"],
  }
  payload["prediction_payload_sha256"] = canonical_sha256(payload)
  _write(args.output_root / "pure_llm_predictions.json", payload)
  progress["terminal"] = True
  _write(progress_path, progress)
  _write(args.output_root / "run_receipt.json", {
      "schema_version": "linkcad_pure_llm_run_receipt.v1",
      "prediction_payload_sha256": payload["prediction_payload_sha256"],
      "query_count": len(queries),
      "valid_prediction_count": len(progress["predictions"]),
      "failed_prediction_count": len(progress["failures"]),
      "wall_time_seconds": time.perf_counter() - started,
      "credential_material_included": False,
      "private_targets_opened": False,
  })
  print(json.dumps({
      "query_count": len(queries), "valid": len(progress["predictions"]),
      "failures": len(progress["failures"]),
  }, sort_keys=True))


if __name__ == "__main__":
  main()
