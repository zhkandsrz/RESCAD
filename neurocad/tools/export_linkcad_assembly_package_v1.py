"""Export accepted LinkCAD predictions as product-structured STEP packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from neurocad.linkcad_assembly_export_v1 import export_assembly_package_v1


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
  nested = value.get("execution")
  return nested if isinstance(nested, Mapping) else value


def export_prediction_packages_v1(
    *, public: Mapping[str, Any], predictions: Mapping[str, Any],
    global_execution: Mapping[str, Any], pair_execution: Mapping[str, Any],
    dataset_root: Path, output_dir: Path, query_id: str | None = None,
) -> dict[str, Any]:
  queries = {str(row["query_id"]): row for row in public["queries"]}
  prediction_rows = {
      str(row["query_id"]): row for row in predictions["rows"]
  }
  global_rows = {
      str(row["query_id"]): row
      for row in _payload(global_execution)["rows"]
  }
  pair_rows = list(_payload(pair_execution)["rows"])
  requested = [query_id] if query_id is not None else sorted(global_rows)
  receipts = []
  status_counts: dict[str, int] = {}
  for current_id in requested:
    if current_id not in queries or current_id not in prediction_rows:
      raise ValueError(f"LinkCAD export query is unavailable: {current_id}")
    global_row = global_rows.get(current_id)
    if (
        not isinstance(global_row, Mapping)
        or global_row.get("status") != "accepted_global_assembly"
    ):
      status = "skipped_not_accepted"
      status_counts[status] = status_counts.get(status, 0) + 1
      continue
    hypotheses = prediction_rows[current_id].get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
      raise ValueError("LinkCAD export prediction lacks hypotheses")
    hypothesis = hypotheses[0]
    output = output_dir / current_id
    receipt = export_assembly_package_v1(
        query=queries[current_id], hypothesis=hypothesis,
        global_execution=global_row,
        pair_execution_rows=[
            row for row in pair_rows if str(row.get("query_id")) == current_id
        ],
        dataset_root=dataset_root,
        output_step=output / "assembly.step",
        output_manifest=output / "assembly.manifest.json",
    )
    receipt_path = output / "assembly.export_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8",
    )
    receipts.append(receipt)
    status = str(receipt["status"])
    status_counts[status] = status_counts.get(status, 0) + 1
  return {
      "schema_version": "linkcad_assembly_export_batch.v1",
      "requested_query_count": len(requested),
      "exported_query_count": len(receipts),
      "status_counts": dict(sorted(status_counts.items())),
      "receipts": receipts,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--predictions", type=Path, required=True)
  parser.add_argument("--global-execution", type=Path, required=True)
  parser.add_argument("--pair-execution", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--query-id")
  args = parser.parse_args()
  result = export_prediction_packages_v1(
      public=_read(args.public), predictions=_read(args.predictions),
      global_execution=_read(args.global_execution),
      pair_execution=_read(args.pair_execution),
      dataset_root=args.dataset_root.resolve(),
      output_dir=args.output_dir.resolve(), query_id=args.query_id,
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  output = args.output_dir / "assembly_export_batch.json"
  output.write_text(
      json.dumps(result, indent=2, sort_keys=True), encoding="utf-8",
  )
  print(output)


if __name__ == "__main__":
  main()

