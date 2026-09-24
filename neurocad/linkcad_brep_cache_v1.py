"""Loader and query attachment for LinkCAD B-Rep graph shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .linkcad_brep_encoder_v1 import LinkCADPartGraph
from .linkcad_factorized_model_v1 import LinkCADQueryTensors


SCHEMA_VERSION = "linkcad_brep_graph_cache.v1"


class LinkCADBRepGraphCacheV1:
  def __init__(self, root: str | Path) -> None:
    self.root = Path(root).resolve()
    self.manifest = json.loads(
        (self.root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        self.manifest.get("schema_version") != SCHEMA_VERSION
        or self.manifest.get("complete") is not True
    ):
      raise ValueError("LinkCAD B-Rep cache manifest differs")
    self._rows: dict[str, Mapping[str, Any]] = {}

  @property
  def excluded_step_sha256s(self) -> frozenset[str]:
    return frozenset(self.manifest["exclusions"])

  def load_all(self) -> None:
    if self._rows:
      return
    for shard in self.manifest["shards"]:
      payload = torch.load(
          self.root / shard["path"], map_location="cpu", weights_only=False
      )
      if not isinstance(payload, Mapping):
        raise ValueError("LinkCAD B-Rep graph shard differs")
      self._rows.update(payload)
    if len(self._rows) != len(self.manifest["entries"]):
      raise ValueError("LinkCAD B-Rep cache accounting differs")

  def graph(self, step_sha256: str) -> LinkCADPartGraph | None:
    self.load_all()
    row = self._rows.get(step_sha256)
    if row is None:
      return None
    return LinkCADPartGraph(
        node_features=row["node_features"],
        edge_index=row["edge_index"],
        edge_features=row["edge_features"],
        orbit_members=tuple(tuple(members) for members in row["orbit_members"]),
    )


def attach_brep_graphs(
    query: LinkCADQueryTensors,
    public_query: Mapping[str, Any],
    cache: LinkCADBRepGraphCacheV1,
) -> LinkCADQueryTensors:
  role_ids = [str(row["role_id"]) for row in public_query["roles"]]
  graphs = []
  masks = []
  for role_id in role_ids:
    role_graphs = []
    role_masks = []
    for candidate in public_query["candidate_sets"][role_id]:
      graph = cache.graph(str(candidate["step_sha256"]))
      role_graphs.append(graph)
      role_masks.append(graph is not None)
    graphs.append(tuple(role_graphs))
    masks.append(role_masks)
  return LinkCADQueryTensors(
      query_id=query.query_id,
      candidate_features=query.candidate_features,
      role_text_features=query.role_text_features,
      edge_index=query.edge_index,
      edge_text_features=query.edge_text_features,
      candidate_mask=torch.tensor(masks, dtype=torch.bool),
      target_assignment=query.target_assignment,
      target_mobility=query.target_mobility,
      target_support=query.target_support,
      candidate_ids=query.candidate_ids,
      edge_ids=query.edge_ids,
      candidate_graphs=tuple(graphs),
      requested_mobility=query.requested_mobility,
      port_candidate_mask=query.port_candidate_mask,
  )
