"""Keep frozen LinkCAD part rankings and dispatch unique interface contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurocad.linkcad_contract_dispatch_v1 import (
    materialize_contract_dispatch_predictions_v1,
)
from neurocad.linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2


def _read(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--predictions", type=Path, required=True)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  cache = LinkCADPrimitiveGraphCacheV2(args.primitive_cache)
  cache.load_all()
  output = materialize_contract_dispatch_predictions_v1(
      public=_read(args.public), predictions=_read(args.predictions), cache=cache,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  print(args.output)


if __name__ == "__main__":
  main()
