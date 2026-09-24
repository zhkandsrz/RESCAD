"""Materialize target-free symbolic port feasibility for LinkCAD candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from neurocad.linkcad_factorized_model_v1 import tensorize_linkcad_query
from neurocad.linkcad_primitive_cache_v2 import (
    LinkCADPrimitiveGraphCacheV2,
    attach_primitive_graphs_v2,
)


SCHEMA_VERSION = "linkcad_port_feasibility_sidecar.v1"


def materialize_port_feasibility_v1(
    *, public: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
) -> dict[str, Any]:
  rows = []
  raw_pair_count = 0
  feasible_pair_count = 0
  zero_feasible_edge_count = 0
  for public_query in public["queries"]:
    query = attach_primitive_graphs_v2(
        tensorize_linkcad_query(public_query), public_query, cache,
    )
    if query.port_candidate_mask is None:
      raise ValueError("LinkCAD port feasibility mask is missing")
    edge_rows = []
    for edge_ordinal, edge in enumerate(public_query["functional_edges"]):
      left = query.port_candidate_mask[edge_ordinal, 0]
      right = query.port_candidate_mask[edge_ordinal, 1]
      left_count = int(left.sum())
      right_count = int(right.sum())
      pair_count = left_count * right_count
      raw = query.candidate_count ** 2
      raw_pair_count += raw
      feasible_pair_count += pair_count
      zero_feasible_edge_count += int(pair_count == 0)
      edge_rows.append({
          "edge_id": edge["edge_id"],
          "port_contract_status": edge.get("port_contract_status", "unknown"),
          "compatible_candidate_a": left.tolist(),
          "compatible_candidate_b": right.tolist(),
          "raw_candidate_pair_count": raw,
          "feasible_candidate_pair_count": pair_count,
      })
    rows.append({"query_id": public_query["query_id"], "edges": edge_rows})
  return {
      "schema_version": SCHEMA_VERSION,
      "scope": "public_target_id_free_symbolic_port_feasibility",
      "selection_uses_private_targets_or_model_outputs": False,
      "query_count": len(rows),
      "edge_count": sum(len(row["edges"]) for row in rows),
      "raw_candidate_pair_count": raw_pair_count,
      "feasible_candidate_pair_count": feasible_pair_count,
      "pruned_candidate_pair_count": raw_pair_count - feasible_pair_count,
      "zero_feasible_edge_count": zero_feasible_edge_count,
      "rows": rows,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  public = json.loads(args.public.read_text(encoding="utf-8"))
  cache = LinkCADPrimitiveGraphCacheV2(args.primitive_cache)
  cache.load_all()
  payload = materialize_port_feasibility_v1(public=public, cache=cache)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
      key: payload[key] for key in (
          "query_count", "edge_count", "raw_candidate_pair_count",
          "feasible_candidate_pair_count", "zero_feasible_edge_count",
      )
  }, sort_keys=True))


if __name__ == "__main__":
  main()
