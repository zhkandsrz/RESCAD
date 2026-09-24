"""Materialize the LinkCAD confirmation exact-execution private join."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from neurocad.linkcad_confirmation_execution_evaluation_v1 import (
    evaluate_confirmation_execution_v1,
)
from neurocad.linkcad_pose_equivalence_v1 import (
    source_occurrence_world_matrices_v1,
)


def _read(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--private-targets", type=Path, required=True)
  parser.add_argument("--primitive-supervision", type=Path, required=True)
  parser.add_argument("--execution-receipt", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  private = _read(args.private_targets)
  source_worlds = {}
  source_bindings = []
  for target in private["targets"]:
    path = (args.dataset_root / target["source_assembly_json"]).resolve()
    if _sha(path) != target["source_assembly_sha256"]:
      raise ValueError("LinkCAD source assembly bytes differ")
    source_worlds[target["query_id"]] = source_occurrence_world_matrices_v1(
        _read(path)
    )
    source_bindings.append({
        "query_id": target["query_id"],
        "source_assembly_sha256": target["source_assembly_sha256"],
    })
  payload = evaluate_confirmation_execution_v1(
      public=_read(args.public),
      private_targets=private,
      primitive_supervision=_read(args.primitive_supervision),
      execution_receipt=_read(args.execution_receipt),
      source_worlds_by_query=source_worlds,
  )
  payload["execution_receipt_sha256"] = _sha(args.execution_receipt)
  payload["source_assembly_bindings"] = source_bindings
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
  main()
