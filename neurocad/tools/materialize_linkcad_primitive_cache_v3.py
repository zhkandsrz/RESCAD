"""Materialize analytic-radius LinkCAD primitive graph shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from neurocad.tools.materialize_linkcad_brep_cache_v1 import _candidate_steps


SCHEMA_VERSION = "linkcad_primitive_graph_cache.v3"


def _worker(item: tuple[str, str]) -> dict[str, Any]:
  expected_sha256, raw_path = item
  try:
    path = Path(raw_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
      raise ValueError("STEP bytes differ from public candidate binding")
    import cadquery as cq
    from neurocad.linkcad_primitive_graph_v3 import extract_primitive_graph_v3

    graph = extract_primitive_graph_v3(cq.importers.importStep(str(path)).val())
    return {
        "status": "ok", "step_sha256": expected_sha256,
        "node_features": graph.face_graph.node_features,
        "edge_index": graph.face_graph.edge_index,
        "edge_features": graph.face_graph.edge_features,
        "face_orbit_members": graph.face_graph.orbit_members,
        "primitive_features": graph.primitive_features,
        "primitive_kinds": graph.primitive_kinds,
        "primitive_members": graph.primitive_members,
        "face_orbit_count": graph.face_orbit_count,
        "face_count": int(graph.face_graph.node_features.shape[0]),
        "primitive_count": int(graph.primitive_features.shape[0]),
    }
  except Exception as error:
    return {
        "status": "excluded", "step_sha256": expected_sha256,
        "reason": f"{type(error).__name__}: {error}",
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
  path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def materialize_cache_v3(
    *, public_paths: list[Path], dataset_root: Path, output_dir: Path,
    workers: int, shard_size: int,
) -> dict[str, Any]:
  output_dir.mkdir(parents=True, exist_ok=True)
  partial = output_dir / "partial_manifest.json"
  manifest = (
      json.loads(partial.read_text(encoding="utf-8"))
      if partial.exists() else {
          "schema_version": SCHEMA_VERSION, "complete": False,
          "entries": {}, "exclusions": {}, "shards": [],
      }
  )
  if manifest.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("LinkCAD V3 partial cache schema differs")
  candidates = _candidate_steps(public_paths, dataset_root.resolve())
  pending = [
      (sha, path.as_posix()) for sha, path in sorted(candidates.items())
      if sha not in manifest["entries"] and sha not in manifest["exclusions"]
  ]
  shard_rows = {}
  shard_index = len(manifest["shards"])

  def flush() -> None:
    nonlocal shard_rows, shard_index
    if not shard_rows:
      return
    name = f"shard_{shard_index:05d}.pt"
    torch.save(shard_rows, output_dir / name)
    manifest["shards"].append({"path": name, "row_count": len(shard_rows)})
    for row_index, (sha, row) in enumerate(shard_rows.items()):
      manifest["entries"][sha] = {
          "shard": name, "row_index": row_index,
          "face_count": row["face_count"],
          "primitive_count": row["primitive_count"],
      }
    shard_rows = {}
    shard_index += 1
    _write(partial, manifest)

  with ProcessPoolExecutor(max_workers=workers) as executor:
    for result in executor.map(_worker, pending, chunksize=1):
      sha = result.pop("step_sha256")
      if result.pop("status") == "ok":
        shard_rows[sha] = result
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
      "primitive_feature_schema": "linkcad_primitive_features.v3",
  })
  _write(partial, manifest)
  if manifest["complete"]:
    _write(output_dir / "manifest.json", manifest)
  return manifest


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, nargs="+", required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
  parser.add_argument("--shard-size", type=int, default=128)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  result = materialize_cache_v3(
      public_paths=args.public, dataset_root=args.dataset_root,
      output_dir=args.output_dir, workers=args.workers,
      shard_size=args.shard_size,
  )
  print(json.dumps({
      "candidate_step_count": result["candidate_step_count"],
      "success_count": len(result["entries"]),
      "excluded_count": len(result["exclusions"]),
      "complete": result["complete"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
