"""Materialize resumable intrinsic B-Rep graph shards for LinkCAD candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "linkcad_brep_graph_cache.v1"


def _candidate_steps(public_paths: list[Path], dataset_root: Path) -> dict[str, Path]:
  result: dict[str, Path] = {}
  for public_path in public_paths:
    payload = json.loads(public_path.read_text(encoding="utf-8"))
    for query in payload["queries"]:
      for candidates in query["candidate_sets"].values():
        for row in candidates:
          sha = str(row["step_sha256"])
          path = (dataset_root / row["step_path"]).resolve()
          previous = result.get(sha)
          if previous is None or path.as_posix() < previous.as_posix():
            result[sha] = path
  return result


def _worker(item: tuple[str, str]) -> dict[str, Any]:
  expected_sha256, raw_path = item
  try:
    path = Path(raw_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
      raise ValueError("STEP bytes differ from the public candidate binding")
    import cadquery as cq
    from neurocad.benchmark_v2_model_view_v2 import (
        GraphSizeBudget,
        extract_intrinsic_brep_graph,
    )
    from neurocad.linkcad_brep_encoder_v1 import graph_payload_to_tensors

    shape = cq.importers.importStep(str(path)).val()
    graph_payload = extract_intrinsic_brep_graph(
        shape,
        budget=GraphSizeBudget(
            max_graphs=1,
            max_faces_per_graph=1024,
            max_edges_per_graph=4096,
            max_examples=1,
            max_program_candidates_per_example=1,
        ),
    )
    graph = graph_payload_to_tensors(graph_payload)
    return {
        "status": "ok",
        "step_sha256": expected_sha256,
        "node_features": graph.node_features,
        "edge_index": graph.edge_index,
        "edge_features": graph.edge_features,
        "orbit_members": graph.orbit_members,
        "face_count": int(graph.node_features.shape[0]),
        "edge_count": int(graph.edge_index.shape[0]),
        "orbit_count": len(graph.orbit_members),
    }
  except Exception as error:  # worker boundary retains every failure in the funnel
    return {
        "status": "excluded",
        "step_sha256": expected_sha256,
        "reason": f"{type(error).__name__}: {error}",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
  path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def materialize_cache(
    *,
    public_paths: list[Path],
    dataset_root: Path,
    output_dir: Path,
    workers: int,
    shard_size: int,
    max_steps: int = 0,
) -> dict[str, Any]:
  output_dir.mkdir(parents=True, exist_ok=True)
  partial_path = output_dir / "partial_manifest.json"
  if partial_path.exists():
    manifest = json.loads(partial_path.read_text(encoding="utf-8"))
  else:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "entries": {},
        "exclusions": {},
        "shards": [],
    }
  candidates = _candidate_steps(public_paths, dataset_root.resolve())
  pending = [
      (sha, path.as_posix()) for sha, path in sorted(candidates.items())
      if sha not in manifest["entries"] and sha not in manifest["exclusions"]
  ]
  if max_steps > 0:
    pending = pending[:max_steps]
  shard_rows: dict[str, dict[str, Any]] = {}
  shard_index = len(manifest["shards"])

  def flush() -> None:
    nonlocal shard_index, shard_rows
    if not shard_rows:
      return
    shard_name = f"shard_{shard_index:05d}.pt"
    torch.save(shard_rows, output_dir / shard_name)
    manifest["shards"].append({"path": shard_name, "row_count": len(shard_rows)})
    for row_index, (sha, row) in enumerate(shard_rows.items()):
      manifest["entries"][sha] = {
          "shard": shard_name,
          "row_index": row_index,
          "face_count": row["face_count"],
          "edge_count": row["edge_count"],
          "orbit_count": row["orbit_count"],
      }
    shard_index += 1
    shard_rows = {}
    _write_json(partial_path, manifest)

  with ProcessPoolExecutor(max_workers=workers) as executor:
    for result in executor.map(_worker, pending, chunksize=1):
      sha = result["step_sha256"]
      if result["status"] == "ok":
        shard_rows[sha] = {
            key: value for key, value in result.items()
            if key not in {"status", "step_sha256"}
        }
        if len(shard_rows) >= shard_size:
          flush()
      else:
        manifest["exclusions"][sha] = result["reason"]
  flush()
  accounted = len(manifest["entries"]) + len(manifest["exclusions"])
  manifest.update({
      "candidate_step_count": len(candidates),
      "accounted_step_count": accounted,
      "complete": accounted == len(candidates),
      "worker_count": workers,
      "shard_size": shard_size,
  })
  _write_json(partial_path, manifest)
  if manifest["complete"]:
    _write_json(output_dir / "manifest.json", manifest)
  return manifest


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, nargs="+", required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
  parser.add_argument("--shard-size", type=int, default=128)
  parser.add_argument("--max-steps", type=int, default=0)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  result = materialize_cache(
      public_paths=args.public,
      dataset_root=args.dataset_root,
      output_dir=args.output_dir,
      workers=args.workers,
      shard_size=args.shard_size,
      max_steps=args.max_steps,
  )
  print(json.dumps({
      "candidate_step_count": result["candidate_step_count"],
      "accounted_step_count": result["accounted_step_count"],
      "success_count": len(result["entries"]),
      "excluded_count": len(result["exclusions"]),
      "complete": result["complete"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
