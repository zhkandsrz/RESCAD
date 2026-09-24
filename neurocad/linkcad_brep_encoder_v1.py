"""Intrinsic B-Rep tensorization and symmetry-aware LinkCAD part encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


SURFACE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus", "spline", "other")
NODE_DIM = 19
EDGE_DIM = 3


def normalize_node_features(node_features: Tensor) -> Tensor:
  """Bound curvature/count long tails while preserving signs and invariance."""

  if node_features.ndim != 2 or node_features.shape[1] != NODE_DIM:
    raise ValueError("LinkCAD B-Rep node tensor differs")
  result = node_features.clone()
  result[:, 1] = torch.log1p(result[:, 1].clamp_min(0.0))
  for index in (2, 3):
    value = result[:, index]
    result[:, index] = torch.sign(value) * torch.log1p(torch.abs(value))
  for index in (4, 5):
    result[:, index] = torch.log1p(result[:, index].clamp_min(0.0))
  return torch.nan_to_num(result, nan=0.0, posinf=30.0, neginf=-30.0)


@dataclass(frozen=True, slots=True)
class LinkCADPartGraph:
  node_features: Tensor
  edge_index: Tensor
  edge_features: Tensor
  orbit_members: tuple[tuple[int, ...], ...]

  def to(self, device: str | torch.device) -> "LinkCADPartGraph":
    return LinkCADPartGraph(
        node_features=self.node_features.to(device),
        edge_index=self.edge_index.to(device),
        edge_features=self.edge_features.to(device),
        orbit_members=self.orbit_members,
    )


def _structural_orbits(
    node_features: Tensor,
    edge_index: Tensor,
    edge_features: Tensor,
) -> tuple[tuple[int, ...], ...]:
  node_count = int(node_features.shape[0])
  adjacency: list[list[tuple[int, tuple[float, ...]]]] = [
      [] for _ in range(node_count)
  ]
  for ordinal, (source, target) in enumerate(edge_index.tolist()):
    edge_key = tuple(round(float(value), 6) for value in edge_features[ordinal])
    adjacency[source].append((target, edge_key))
    adjacency[target].append((source, edge_key))
  keys: list[Any] = [
      tuple(round(float(value), 6) for value in node_features[index])
      for index in range(node_count)
  ]
  for _ in range(3):
    refined = [
        (keys[index], tuple(sorted((edge, keys[neighbor]) for neighbor, edge in adjacency[index])))
        for index in range(node_count)
    ]
    vocabulary = {key: ordinal for ordinal, key in enumerate(sorted(set(refined), key=repr))}
    updated = [vocabulary[key] for key in refined]
    if updated == keys:
      break
    keys = updated
  groups: dict[Any, list[int]] = {}
  for index, key in enumerate(keys):
    groups.setdefault(key, []).append(index)
  return tuple(
      tuple(indices) for _, indices in sorted(groups.items(), key=lambda row: row[1][0])
  )


def graph_payload_to_tensors(graph: Mapping[str, Any]) -> LinkCADPartGraph:
  nodes = graph.get("nodes")
  edges = graph.get("edges")
  if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
    raise ValueError("LinkCAD B-Rep graph payload differs")
  node_rows = []
  for node in nodes:
    surface = str(node["surface_type"])
    if surface not in SURFACE_TYPES:
      raise ValueError("LinkCAD B-Rep surface type differs")
    features = [float(value) for value in node["features"]]
    masks = [float(value) for value in node["feature_mask"]]
    row = features + masks + [float(surface == item) for item in SURFACE_TYPES]
    if len(row) != NODE_DIM:
      raise ValueError("LinkCAD B-Rep node dimension differs")
    node_rows.append(row)
  edge_index = torch.tensor(
      [[int(row["source"]), int(row["target"])] for row in edges],
      dtype=torch.long,
  )
  edge_features = torch.tensor(
      [[float(value) for value in row["features"]] for row in edges],
      dtype=torch.float32,
  )
  if not edges:
    edge_index = torch.zeros((0, 2), dtype=torch.long)
    edge_features = torch.zeros((0, EDGE_DIM), dtype=torch.float32)
  result = LinkCADPartGraph(
      node_features=torch.tensor(node_rows, dtype=torch.float32),
      edge_index=edge_index,
      edge_features=edge_features,
      orbit_members=(),
  )
  return LinkCADPartGraph(
      node_features=result.node_features,
      edge_index=result.edge_index,
      edge_features=result.edge_features,
      orbit_members=_structural_orbits(
          result.node_features, result.edge_index, result.edge_features
      ),
  )


class _MessagePassing(nn.Module):
  def __init__(self, hidden_dim: int) -> None:
    super().__init__()
    self.message = nn.Sequential(
        nn.Linear(2 * hidden_dim + EDGE_DIM, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.update = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(self, hidden: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
    aggregate = torch.zeros_like(hidden)
    degree = torch.zeros((hidden.shape[0], 1), device=hidden.device)
    if edge_index.numel():
      source, target = edge_index.unbind(dim=-1)
      forward = self.message(torch.cat((hidden[source], hidden[target], edge_features), dim=-1))
      reverse = self.message(torch.cat((hidden[target], hidden[source], edge_features), dim=-1))
      aggregate.index_add_(0, target, forward)
      aggregate.index_add_(0, source, reverse)
      ones = torch.ones((source.shape[0], 1), device=hidden.device)
      degree.index_add_(0, target, ones)
      degree.index_add_(0, source, ones)
    aggregate = aggregate / degree.clamp_min(1.0)
    return self.norm(hidden + self.update(torch.cat((hidden, aggregate), dim=-1)))


class LinkCADBRepEncoderV1(nn.Module):
  """SE(3)-invariant face graph encoder with explicit orbit pooling."""

  def __init__(self, *, hidden_dim: int = 64, layers: int = 3) -> None:
    super().__init__()
    self.node_encoder = nn.Sequential(
        nn.Linear(NODE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
    )
    self.layers = nn.ModuleList(_MessagePassing(hidden_dim) for _ in range(layers))
    self.orbit_attention = nn.Linear(hidden_dim, 1, bias=False)
    self.output = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
    )

  def forward(self, graph: LinkCADPartGraph, *, orbit_mode: str = "attention") -> Tensor:
    return self.forward_many((graph,), orbit_mode=orbit_mode)[0]

  def forward_many(
      self,
      graphs: tuple[LinkCADPartGraph, ...] | list[LinkCADPartGraph],
      *,
      orbit_mode: str = "attention",
  ) -> Tensor:
    """Encode a disjoint graph batch without padding faces or topology."""

    parts, _orbits = self.forward_many_with_orbits(
        graphs, orbit_mode=orbit_mode
    )
    return parts

  def forward_many_with_orbits(
      self,
      graphs: tuple[LinkCADPartGraph, ...] | list[LinkCADPartGraph],
      *,
      orbit_mode: str = "attention",
  ) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Return one part embedding and the retained local orbit set per graph."""

    if orbit_mode not in {"attention", "canonical"}:
      raise ValueError("LinkCAD orbit mode differs")
    if not graphs:
      raise ValueError("LinkCAD B-Rep graph batch is empty")
    device = next(self.parameters()).device
    node_rows = []
    edge_rows = []
    edge_feature_rows = []
    node_to_orbit = []
    orbit_to_graph = []
    canonical_nodes = []
    node_offset = 0
    orbit_offset = 0
    for graph_index, graph in enumerate(graphs):
      graph = graph.to(device)
      node_rows.append(graph.node_features)
      if graph.edge_index.numel():
        edge_rows.append(graph.edge_index + node_offset)
        edge_feature_rows.append(graph.edge_features)
      local_node_to_orbit = [-1] * int(graph.node_features.shape[0])
      for local_orbit, members in enumerate(graph.orbit_members):
        global_orbit = orbit_offset + local_orbit
        for member in members:
          local_node_to_orbit[member] = global_orbit
        canonical_nodes.append(node_offset + members[0])
        orbit_to_graph.append(graph_index)
      if any(value < 0 for value in local_node_to_orbit):
        raise ValueError("LinkCAD orbit partition is incomplete")
      node_to_orbit.extend(local_node_to_orbit)
      node_offset += int(graph.node_features.shape[0])
      orbit_offset += len(graph.orbit_members)
    nodes = normalize_node_features(torch.cat(node_rows))
    edge_index = (
        torch.cat(edge_rows)
        if edge_rows else torch.zeros((0, 2), dtype=torch.long, device=device)
    )
    edge_features = (
        torch.cat(edge_feature_rows)
        if edge_feature_rows else torch.zeros((0, EDGE_DIM), device=device)
    )
    hidden = self.node_encoder(nodes)
    for layer in self.layers:
      hidden = layer(hidden, edge_index, edge_features)
    orbit_ids = torch.tensor(node_to_orbit, dtype=torch.long, device=device)
    orbit_graph_ids = torch.tensor(orbit_to_graph, dtype=torch.long, device=device)
    orbit_count = len(orbit_to_graph)
    if orbit_mode == "canonical":
      orbit_tensor = hidden[
          torch.tensor(canonical_nodes, dtype=torch.long, device=device)
      ]
    else:
      logits = self.orbit_attention(hidden).squeeze(-1)
      maximum = torch.full((orbit_count,), -torch.inf, device=device)
      maximum.scatter_reduce_(0, orbit_ids, logits, reduce="amax", include_self=True)
      weights = torch.exp(logits - maximum[orbit_ids])
      denominator = torch.zeros((orbit_count,), device=device)
      denominator.index_add_(0, orbit_ids, weights)
      weighted = hidden * (weights / denominator[orbit_ids].clamp_min(1e-12))[:, None]
      orbit_tensor = torch.zeros(
          (orbit_count, hidden.shape[1]), device=device, dtype=hidden.dtype
      )
      orbit_tensor.index_add_(0, orbit_ids, weighted)
    graph_count = len(graphs)
    graph_sum = torch.zeros(
        (graph_count, hidden.shape[1]), device=device, dtype=hidden.dtype
    )
    graph_sum.index_add_(0, orbit_graph_ids, orbit_tensor)
    graph_degree = torch.zeros((graph_count, 1), device=device)
    graph_degree.index_add_(
        0, orbit_graph_ids, torch.ones((orbit_count, 1), device=device)
    )
    graph_mean = graph_sum / graph_degree.clamp_min(1.0)
    graph_max = torch.full_like(graph_mean, -torch.inf)
    expanded_graph_ids = orbit_graph_ids[:, None].expand_as(orbit_tensor)
    graph_max.scatter_reduce_(
        0, expanded_graph_ids, orbit_tensor, reduce="amax", include_self=True
    )
    part_embeddings = self.output(torch.cat((graph_mean, graph_max), dim=-1))
    orbit_sets = tuple(
        orbit_tensor[orbit_graph_ids == graph_index]
        for graph_index in range(graph_count)
    )
    return part_embeddings, orbit_sets
