"""Loader for analytic-radius LinkCAD primitive graph shards."""

from __future__ import annotations

import json
from pathlib import Path

from .linkcad_primitive_cache_v2 import (
    LinkCADPrimitiveGraphCacheV2,
    attach_primitive_graphs_v2,
)


SCHEMA_VERSION = "linkcad_primitive_graph_cache.v3"


class LinkCADPrimitiveGraphCacheV3(LinkCADPrimitiveGraphCacheV2):
  def __init__(self, root: str | Path) -> None:
    self.root = Path(root).resolve()
    self.manifest = json.loads(
        (self.root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        self.manifest.get("schema_version") != SCHEMA_VERSION
        or self.manifest.get("complete") is not True
    ):
      raise ValueError("LinkCAD V3 primitive cache manifest differs")
    self._rows = {}
    self._graphs = {}
    self._port_contract_keys = {}


attach_primitive_graphs_v3 = attach_primitive_graphs_v2


__all__ = [
    "LinkCADPrimitiveGraphCacheV3", "attach_primitive_graphs_v3",
]
