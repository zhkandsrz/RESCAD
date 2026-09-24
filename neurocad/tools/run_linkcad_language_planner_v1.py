"""Run the frozen LinkCAD language planner without loading reference targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from neurocad.linkcad_language_planner_v1 import (
    DEFAULT_MODEL,
    FrozenLinkCADLanguagePlannerV1,
    PREDICTION_SCHEMA_VERSION,
    canonical_sha256,
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
  parser.add_argument("--inputs", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=12)
  parser.add_argument("--timeout-seconds", type=int, default=180)
  parser.add_argument("--max-items", type=int)
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  if args.batch_size < 1:
    raise ValueError("LinkCAD language-planner batch size differs")
  source = json.loads(args.inputs.read_text(encoding="utf-8"))
  items = source["items"]
  if args.max_items is not None:
    items = items[:args.max_items]
  planner = FrozenLinkCADLanguagePlannerV1(
      model=DEFAULT_MODEL, timeout_seconds=args.timeout_seconds,
  )
  args.output_root.mkdir(parents=True, exist_ok=True)
  progress_path = args.output_root / "progress.json"
  source_sha256 = canonical_sha256(source)
  predictions = []
  receipts = []
  errors = []
  processed_count = 0
  if progress_path.exists():
    if not args.resume:
      raise ValueError("LinkCAD language-planner progress exists; pass --resume")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if (
        progress.get("input_sha256") != source_sha256
        or progress.get("requested_item_count") != len(items)
        or progress.get("batch_size") != args.batch_size
    ):
      raise ValueError("LinkCAD language-planner resume binding differs")
    predictions = list(progress["predictions"])
    receipts = list(progress["batch_receipts"])
    errors = list(progress["errors"])
    processed_count = int(progress["processed_item_count"])
  started = time.perf_counter()
  for start in range(processed_count, len(items), args.batch_size):
    batch = items[start:start + args.batch_size]
    try:
      result = planner.plan(batch)
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
      errors.append({
          "batch_start": start,
          "batch_item_count": len(batch),
          "terminal_status": "planner_error",
          "error_type": type(error).__name__,
          "message": str(error),
      })
    else:
      predictions.extend(result.predictions)
      receipts.append({"batch_start": start, **result.receipt})
    processed_count = min(start + len(batch), len(items))
    _write(progress_path, {
        "schema_version": "linkcad_language_planner_progress.v1",
        "input_sha256": source_sha256,
        "requested_item_count": len(items),
        "processed_item_count": processed_count,
        "batch_size": args.batch_size,
        "predictions": predictions,
        "batch_receipts": receipts,
        "errors": errors,
        "terminal": False,
    })
    print(json.dumps({
        "completed": min(start + len(batch), len(items)),
        "total": len(items), "errors": len(errors),
    }, sort_keys=True), flush=True)
  prediction_payload = {
      "schema_version": PREDICTION_SCHEMA_VERSION,
      "scope": "frozen_target_id_free_language_planner_predictions",
      "provider": "deepseek",
      "model": DEFAULT_MODEL,
      "temperature": 0.0,
      "prompt_sha256": prompt_sha256_v1(),
      "input_sha256": source_sha256,
      "requested_item_count": len(items),
      "prediction_count": len(predictions),
      "items": predictions,
  }
  receipt = {
      "schema_version": "linkcad_language_planner_run_receipt.v1",
      "prediction_sha256": canonical_sha256(prediction_payload),
      "batch_count": (len(items) + args.batch_size - 1) // args.batch_size,
      "successful_batch_count": len(receipts),
      "failed_batch_count": len(errors),
      "wall_time_seconds": time.perf_counter() - started,
      "batch_receipts": receipts,
      "errors": errors,
      "credential_material_included": False,
  }
  _write(args.output_root / "predictions.json", prediction_payload)
  _write(args.output_root / "run_receipt.json", receipt)
  _write(progress_path, {
      "schema_version": "linkcad_language_planner_progress.v1",
      "input_sha256": source_sha256,
      "requested_item_count": len(items),
      "processed_item_count": len(items),
      "batch_size": args.batch_size,
      "predictions": predictions,
      "batch_receipts": receipts,
      "errors": errors,
      "terminal": True,
  })
  print(json.dumps({
      "requested_item_count": len(items),
      "prediction_count": len(predictions),
      "failed_batch_count": len(errors),
      "wall_time_seconds": receipt["wall_time_seconds"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
