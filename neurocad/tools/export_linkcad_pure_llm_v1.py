"""Export and score every pure-LLM prediction without pose search or repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurocad.linkcad_pure_llm_execution_v1 import (
    execute_and_export_pure_llm_v1,
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
  parser.add_argument("--predictions", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  args = parser.parse_args()
  public = _read(args.public)
  batch = _read(args.predictions)
  query_by_id = {str(row["query_id"]): row for row in public["queries"]}
  failures = dict(batch.get("failures", {}))
  rows = []
  for query_id in sorted(query_by_id):
    if query_id not in batch["predictions"]:
      rows.append({
          "query_id": query_id, "status": "no_valid_llm_prediction",
          "raw_collision_free": False, "all_connections_satisfied": False,
          "kernel_feasible": False,
          "failure": failures.get(query_id, {"message": "missing prediction"}),
      })
      continue
    output = args.output_root / query_id
    try:
      row = execute_and_export_pure_llm_v1(
          query=query_by_id[query_id],
          prediction=batch["predictions"][query_id],
          dataset_root=args.dataset_root.resolve(),
          output_step=output / "assembly.step",
          output_manifest=output / "assembly.manifest.json",
      )
    except Exception as error:
      row = {
          "query_id": query_id, "status": "kernel_or_export_failure",
          "raw_collision_free": False, "all_connections_satisfied": False,
          "kernel_feasible": False,
          "failure": {"error_type": type(error).__name__, "message": str(error)},
      }
    rows.append(row)
    print(json.dumps({
        "completed": len(rows), "total": len(query_by_id),
        "collision_free": sum(r["raw_collision_free"] for r in rows),
        "kernel_feasible": sum(r["kernel_feasible"] for r in rows),
    }, sort_keys=True), flush=True)
  count = len(rows)
  result = {
      "schema_version": "linkcad_pure_llm_execution_batch.v1",
      "method_id": "pure_llm_direct_pose",
      "query_count": count,
      "raw_collision_free_rate": sum(
          row["raw_collision_free"] for row in rows
      ) / count,
      "all_connections_satisfied_rate": sum(
          row["all_connections_satisfied"] for row in rows
      ) / count,
      "kernel_feasible_rate": sum(row["kernel_feasible"] for row in rows) / count,
      "pose_search_or_repair_used": False,
      "rows": rows,
  }
  _write(args.output_root / "execution_batch.json", result)
  print(json.dumps({k: result[k] for k in (
      "query_count", "raw_collision_free_rate",
      "all_connections_satisfied_rate", "kernel_feasible_rate",
  )}, sort_keys=True))


if __name__ == "__main__":
  main()
