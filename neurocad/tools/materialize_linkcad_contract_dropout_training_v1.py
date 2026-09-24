"""Create a development-only training view spanning fixed contract strengths."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from neurocad.linkcad_contract_strength_v1 import (
    LEVEL_FIELDS,
    project_public_contract_strength_v1,
)


SCHEMA_VERSION = "linkcad_contract_dropout_training_view.v1"


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_contract_dropout_training_v1(
    *, public: dict[str, Any], private: dict[str, Any],
    supervision: dict[str, Any], levels: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  if not levels or any(level not in LEVEL_FIELDS for level in levels):
    raise ValueError("LinkCAD contract-dropout level domain differs")
  private_by_id = {str(row["query_id"]): row for row in private["targets"]}
  supervision_by_id: dict[str, list[dict[str, Any]]] = {}
  for row in supervision["rows"]:
    supervision_by_id.setdefault(str(row["query_id"]), []).append(row)
  public_rows = []
  private_rows = []
  supervision_rows = []
  for level in levels:
    projected = project_public_contract_strength_v1(public, level=level)
    for query in projected["queries"]:
      source_id = str(query["query_id"])
      augmented_id = f"{source_id}__contract_{level}"
      query["query_id"] = augmented_id
      query["contract_dropout_source_query_id"] = source_id
      target = deepcopy(private_by_id[source_id])
      target["query_id"] = augmented_id
      target["contract_dropout_source_query_id"] = source_id
      public_rows.append(query)
      private_rows.append(target)
      for source_row in supervision_by_id.get(source_id, []):
        row = deepcopy(source_row)
        row["query_id"] = augmented_id
        row["contract_dropout_source_query_id"] = source_id
        supervision_rows.append(row)
  public_view = {
      key: deepcopy(value)
      for key, value in public.items() if key not in {"queries", "query_count"}
  }
  public_view.update({
      "schema_version": SCHEMA_VERSION,
      "scope": "development_only_contract_dropout_training_view",
      "query_count": len(public_rows),
      "source_query_count": len(public["queries"]),
      "contract_levels": list(levels),
      "queries": public_rows,
  })
  private_view = {
      "schema_version": SCHEMA_VERSION,
      "scope": "development_only_contract_dropout_training_targets",
      "query_count": len(private_rows),
      "targets": private_rows,
  }
  supervision_view = {
      "schema_version": SCHEMA_VERSION,
      "scope": "development_only_contract_dropout_primitive_supervision",
      "query_count": len(public_rows),
      "endpoint_count": len(supervision_rows),
      "rows": supervision_rows,
  }
  return public_view, private_view, supervision_view


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument(
      "--levels", nargs="+", choices=tuple(LEVEL_FIELDS),
      default=list(LEVEL_FIELDS),
  )
  args = parser.parse_args()
  public_path = args.source_root / "public.json"
  private_path = args.source_root / "private_targets.sealed.json"
  supervision_path = args.source_root / "primitive_supervision.json"
  public, private, supervision = materialize_contract_dropout_training_v1(
      public=_read(public_path), private=_read(private_path),
      supervision=_read(supervision_path), levels=tuple(args.levels),
  )
  _write(args.output_root / "public.json", public)
  _write(args.output_root / "private_targets.sealed.json", private)
  _write(args.output_root / "primitive_supervision.json", supervision)
  receipt = {
      "schema_version": SCHEMA_VERSION,
      "scope": "development_only_contract_dropout_training_view",
      "source_files": {
          "public_sha256": _sha(public_path),
          "private_targets_sha256": _sha(private_path),
          "primitive_supervision_sha256": _sha(supervision_path),
      },
      "source_query_count": public["source_query_count"],
      "augmented_query_count": public["query_count"],
      "contract_levels": list(args.levels),
      "candidate_geometry_unchanged": True,
      "target_assignments_unchanged": True,
  }
  _write(args.output_root / "receipt.json", receipt)
  print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
  main()
