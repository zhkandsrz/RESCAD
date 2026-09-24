"""Materialize target-free LinkCAD V2 predictions from a frozen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurocad.linkcad_kinematic_execution_v2 import (
    materialize_public_primitive_predictions_v2,
)
from neurocad.linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--beam-size", type=int, default=25)
  parser.add_argument("--interface-alternatives-per-edge", type=int, default=8)
  parser.add_argument(
      "--model-kind", choices=("primitive_v2", "port_v6", "port_v9"),
      default="primitive_v2",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  public = json.loads(args.public.read_text(encoding="utf-8"))
  cache = LinkCADPrimitiveGraphCacheV2(args.primitive_cache)
  cache.load_all()
  payload = materialize_public_primitive_predictions_v2(
      public=public, cache=cache, checkpoint_path=args.checkpoint,
      beam_size=args.beam_size,
      interface_alternatives_per_edge=args.interface_alternatives_per_edge,
      seed=args.seed,
      model_kind=args.model_kind,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps({
      "query_count": payload["query_count"],
      "checkpoint_sha256": payload["checkpoint_sha256"],
      "prediction_payload_sha256": payload["prediction_payload_sha256"],
      "private_targets_opened": payload["private_targets_opened"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
