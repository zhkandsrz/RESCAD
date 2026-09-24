"""Deterministically merge disjoint LinkCAD primitive graph caches."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .linkcad_primitive_cache_v2 import SCHEMA_VERSION


def merge_primitive_caches_v2(
    *, input_roots: list[Path], output_root: Path, shard_size: int = 128,
) -> dict:
  if len(input_roots) < 2 or shard_size < 1:
    raise ValueError("LinkCAD primitive cache merge configuration differs")
  if output_root.exists() and any(output_root.iterdir()):
    raise ValueError("LinkCAD primitive cache merge output exists")
  rows = {}
  source_counts = {}
  for raw_root in input_roots:
    root = raw_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("exclusions")
    ):
      raise ValueError("LinkCAD primitive cache merge input differs")
    source_rows = {}
    for shard in manifest["shards"]:
      payload = torch.load(
          root / shard["path"], map_location="cpu", weights_only=False,
      )
      if not isinstance(payload, dict):
        raise ValueError("LinkCAD primitive cache merge shard differs")
      source_rows.update(payload)
    if set(source_rows) != set(manifest["entries"]):
      raise ValueError("LinkCAD primitive cache merge accounting differs")
    overlap = set(rows) & set(source_rows)
    if overlap:
      raise ValueError("LinkCAD primitive cache merge identity overlaps")
    rows.update(source_rows)
    source_counts[root.as_posix()] = len(source_rows)
  output_root.mkdir(parents=True, exist_ok=True)
  entries = {}
  shards = []
  ordered = sorted(rows.items())
  for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
    block = dict(ordered[start:start + shard_size])
    name = f"shard_{shard_index:05d}.pt"
    torch.save(block, output_root / name)
    shards.append({"path": name, "row_count": len(block)})
    for row_index, (step_sha256, row) in enumerate(block.items()):
      entries[step_sha256] = {
          "shard": name,
          "row_index": row_index,
          "face_count": int(row["node_features"].shape[0]),
          "primitive_count": int(row["primitive_features"].shape[0]),
      }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "complete": True,
      "candidate_step_count": len(rows),
      "accounted_step_count": len(rows),
      "entries": entries,
      "exclusions": {},
      "shards": shards,
      "worker_count": 0,
      "shard_size": shard_size,
      "merge_source_count": len(input_roots),
      "merge_source_candidate_counts": source_counts,
      "identity_overlap_count": 0,
  }
  (output_root / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return manifest


__all__ = ["merge_primitive_caches_v2"]
