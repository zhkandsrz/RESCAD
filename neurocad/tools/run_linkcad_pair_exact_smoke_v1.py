"""Materialize public LinkCAD predictions and a bounded pairwise OCC smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurocad.linkcad_brep_cache_v1 import LinkCADBRepGraphCacheV1
from neurocad.linkcad_exact_execution_v1 import (
    execute_prediction_subset_v1,
    materialize_public_predictions_v1,
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--brep-cache", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--query-limit", type=int, default=3)
  parser.add_argument("--hypotheses-per-query", type=int, default=1)
  parser.add_argument(
      "--execution-mode", choices=("manifold", "zero_pose"), default="manifold"
  )
  parser.add_argument("--interface-alternatives-per-edge", type=int, default=3)
  parser.add_argument("--maximum-attempts-per-query", type=int, default=25)
  return parser.parse_args()


def _write(path: Path, payload: dict) -> None:
  path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def main() -> None:
  args = _parse_args()
  public = json.loads(args.public.read_text(encoding="utf-8"))
  cache = LinkCADBRepGraphCacheV1(args.brep_cache)
  cache.load_all()
  predictions = materialize_public_predictions_v1(
      public=public,
      cache=cache,
      checkpoint_path=args.checkpoint,
      beam_size=25,
  )
  exact = execute_prediction_subset_v1(
      public=public,
      predictions=predictions,
      dataset_root=args.dataset_root,
      query_limit=args.query_limit,
      hypotheses_per_query=args.hypotheses_per_query,
      execution_mode=args.execution_mode,
      interface_alternatives_per_edge=args.interface_alternatives_per_edge,
      maximum_attempts_per_query=args.maximum_attempts_per_query,
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _write(args.output_dir / "public_predictions.json", predictions)
  _write(args.output_dir / "pair_exact_subset.json", exact)
  print(json.dumps({
      "query_count": predictions["query_count"],
      "executed_edge_count": exact["edge_execution_count"],
      "manifold_attempt_count": exact["manifold_attempt_count"],
      "accepted_count": exact["accepted_count"],
      "status_counts": exact["status_counts"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
