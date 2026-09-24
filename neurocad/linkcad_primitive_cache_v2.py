"""Loader and query attachment for LinkCAD face-and-edge primitive graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .linkcad_brep_encoder_v1 import LinkCADPartGraph
from .linkcad_contract_strength_v1 import (
    SCHEMA_VERSION as PARTIAL_CONTRACT_SCHEMA_VERSION,
    port_contract_matches_v1,
)
from .linkcad_factorized_model_v1 import LinkCADQueryTensors
from .linkcad_primitive_graph_v2 import LinkCADPrimitiveGraphV2
from .linkcad_port_contract_v1 import describe_port_v1, describe_ports_v1


SCHEMA_VERSION = "linkcad_primitive_graph_cache.v2"


def _port_contract_key(value: Mapping[str, Any]) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_port_candidate_mask_v1(
    public_query: Mapping[str, Any],
    graphs: tuple[tuple[LinkCADPrimitiveGraphV2 | None, ...], ...],
    contract_keys: tuple[tuple[frozenset[str] | None, ...], ...] | None = None,
) -> torch.Tensor:
  role_ids = [str(row["role_id"]) for row in public_query["roles"]]
  role_index = {role_id: index for index, role_id in enumerate(role_ids)}
  candidate_count = len(graphs[0])
  edge_masks = []
  for edge in public_query["functional_edges"]:
    side_masks = []
    for side, role_field in (("a", "role_a"), ("b", "role_b")):
      contract = edge.get(f"port_contract_{side}")
      specified = (
          edge.get("port_contract_status") == "specified"
          and isinstance(contract, Mapping)
      )
      if not specified:
        side_masks.append([True] * candidate_count)
        continue
      target = _port_contract_key(contract)
      target_role = role_index[str(edge[role_field])]
      if (
          contract_keys is not None
          and contract.get("schema_version") != PARTIAL_CONTRACT_SCHEMA_VERSION
      ):
        side_masks.append([
            keys is not None and target in keys
            for keys in contract_keys[target_role]
        ])
      else:
        role_graphs = graphs[target_role]
        side_masks.append([
            graph is not None and any(
                port_contract_matches_v1(
                    describe_port_v1(graph, ordinal), contract,
                )
                for ordinal in range(int(graph.primitive_features.shape[0]))
            )
            for graph in role_graphs
        ])
    edge_masks.append(side_masks)
  return torch.tensor(edge_masks, dtype=torch.bool)


class LinkCADPrimitiveGraphCacheV2:
  def __init__(self, root: str | Path) -> None:
    self.root = Path(root).resolve()
    self.manifest = json.loads(
        (self.root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        self.manifest.get("schema_version") != SCHEMA_VERSION
        or self.manifest.get("complete") is not True
    ):
      raise ValueError("LinkCAD primitive cache manifest differs")
    self._rows: dict[str, Mapping[str, Any]] = {}
    self._graphs: dict[str, LinkCADPrimitiveGraphV2 | None] = {}
    self._port_contract_keys: dict[str, frozenset[str] | None] = {}

  def load_all(self) -> None:
    if self._rows:
      return
    for shard in self.manifest["shards"]:
      payload = torch.load(
          self.root / shard["path"], map_location="cpu", weights_only=False
      )
      if not isinstance(payload, Mapping):
        raise ValueError("LinkCAD primitive cache shard differs")
      self._rows.update(payload)
    if len(self._rows) != len(self.manifest["entries"]):
      raise ValueError("LinkCAD primitive cache accounting differs")

  def graph(self, step_sha256: str) -> LinkCADPrimitiveGraphV2 | None:
    self.load_all()
    if step_sha256 in self._graphs:
      return self._graphs[step_sha256]
    row = self._rows.get(step_sha256)
    if row is None:
      self._graphs[step_sha256] = None
      return None
    graph = LinkCADPrimitiveGraphV2(
        face_graph=LinkCADPartGraph(
            node_features=row["node_features"],
            edge_index=row["edge_index"],
            edge_features=row["edge_features"],
            orbit_members=tuple(tuple(value) for value in row["face_orbit_members"]),
        ),
        primitive_features=row["primitive_features"],
        primitive_kinds=tuple(row["primitive_kinds"]),
        primitive_members=tuple(tuple(value) for value in row["primitive_members"]),
        face_orbit_count=int(row["face_orbit_count"]),
    )
    self._graphs[step_sha256] = graph
    return graph

  def port_contract_keys(self, step_sha256: str) -> frozenset[str] | None:
    if step_sha256 in self._port_contract_keys:
      return self._port_contract_keys[step_sha256]
    graph = self.graph(step_sha256)
    keys = (
        None
        if graph is None
        else frozenset(
            _port_contract_key(contract)
            for contract in describe_ports_v1(graph)
        )
    )
    self._port_contract_keys[step_sha256] = keys
    return keys


def attach_primitive_graphs_v2(
    query: LinkCADQueryTensors,
    public_query: Mapping[str, Any],
    cache: LinkCADPrimitiveGraphCacheV2,
) -> LinkCADQueryTensors:
  role_ids = [str(row["role_id"]) for row in public_query["roles"]]
  graphs = []
  masks = []
  contract_keys = []
  for role_id in role_ids:
    role_graphs = []
    role_masks = []
    role_contract_keys = []
    for candidate in public_query["candidate_sets"][role_id]:
      step_sha256 = str(candidate["step_sha256"])
      graph = cache.graph(step_sha256)
      role_graphs.append(graph)
      role_masks.append(graph is not None)
      role_contract_keys.append(cache.port_contract_keys(step_sha256))
    padding = query.candidate_count - len(role_graphs)
    if padding < 0:
      raise ValueError("LinkCAD primitive cache candidates differ")
    role_graphs.extend([None] * padding)
    role_masks.extend([False] * padding)
    role_contract_keys.extend([None] * padding)
    graphs.append(tuple(role_graphs))
    masks.append(role_masks)
    contract_keys.append(tuple(role_contract_keys))
  graph_mask = torch.tensor(masks, dtype=torch.bool)
  if query.candidate_mask is not None:
    graph_mask &= query.candidate_mask
  return LinkCADQueryTensors(
      query_id=query.query_id,
      candidate_features=query.candidate_features,
      role_text_features=query.role_text_features,
      edge_index=query.edge_index,
      edge_text_features=query.edge_text_features,
      candidate_mask=graph_mask,
      target_assignment=query.target_assignment,
      target_mobility=query.target_mobility,
      target_support=query.target_support,
      candidate_ids=query.candidate_ids,
      edge_ids=query.edge_ids,
      candidate_graphs=tuple(graphs),
      requested_mobility=query.requested_mobility,
      port_candidate_mask=build_port_candidate_mask_v1(
          public_query, tuple(graphs), tuple(contract_keys)
      ),
  )


__all__ = [
    "LinkCADPrimitiveGraphCacheV2", "attach_primitive_graphs_v2",
    "build_port_candidate_mask_v1",
]
