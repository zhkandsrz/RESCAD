"""Materialize LinkCAD input from a generated graph and frozen edge parsing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurocad.linkcad_request_graph_compiler_v1 import (
    materialize_two_pass_compiler_public_v1,
)


def _load(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--graph-inputs", type=Path, required=True)
  parser.add_argument("--graph-predictions", type=Path, required=True)
  parser.add_argument("--edge-predictions", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = materialize_two_pass_compiler_public_v1(
      _load(args.public), _load(args.graph_inputs),
      _load(args.graph_predictions), _load(args.edge_predictions),
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "query_count": result["query_count"],
      "compiler_error_count": result["compiler_error_count"],
      "missing_edge_parse_count": result["missing_edge_parse_count"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
