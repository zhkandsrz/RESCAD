"""Run bounded, target-free V3 pair execution on public ranked predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurocad.linkcad_kinematic_execution_v3 import execute_public_prediction_subset_v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--hypotheses-per-query", type=int, default=1)
    parser.add_argument("--alternatives-per-edge", type=int, default=3)
    parser.add_argument("--maximum-attempts-per-query", type=int, default=25)
    args = parser.parse_args()
    public = json.loads(args.public.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    limit = len(public["queries"]) if args.query_limit is None else args.query_limit
    if min(limit, args.hypotheses_per_query, args.alternatives_per_edge,
           args.maximum_attempts_per_query) < 1:
        parser.error("Query and search budgets must be positive.")
    result = execute_public_prediction_subset_v3(
        public=public, predictions=predictions, dataset_root=args.dataset_root,
        query_limit=limit, hypotheses_per_query=args.hypotheses_per_query,
        alternatives_per_edge=args.alternatives_per_edge,
        maximum_attempts_per_query=args.maximum_attempts_per_query,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Pair execution saved.")


if __name__ == "__main__":
    main()

