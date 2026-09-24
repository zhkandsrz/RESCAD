"""JoinABLe-style B-Rep link predictor adapted to LinkCAD candidate sets.

This module intentionally uses the name ``style``: it transfers the published
independent B-Rep graph encoding and cross-part entity-link prediction pattern,
but trains on LinkCAD's public primitive graphs and task splits.  It is not a
drop-in reproduction of the official JoinABLe preprocessing or checkpoints.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .linkcad_brep_encoder_v1 import NODE_DIM, normalize_node_features
from .linkcad_factorized_model_v1 import FactorizedOutput, LinkCADQueryTensors
from .linkcad_primitive_factorized_model_v2 import PrimitiveFactorizedLinkCADV2
from .linkcad_primitive_graph_v2 import (
    LinkCADPrimitiveGraphV2,
    PRIMITIVE_FEATURE_DIM,
)


PREDICTION_SCHEMA_VERSION = "linkcad_joinable_style_predictions.v2"


class _GATv2LayerV1(nn.Module):
  """Small dependency-free GATv2 layer with directed attention and self loops."""

  def __init__(self, hidden_dim: int, *, heads: int = 8) -> None:
    super().__init__()
    if hidden_dim % heads:
      raise ValueError("JoinABLe-style hidden dimension must divide attention heads")
    self.hidden_dim = hidden_dim
    self.heads = heads
    self.head_dim = hidden_dim // heads
    self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.target = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.attention = nn.Parameter(torch.empty(heads, self.head_dim))
    self.bias = nn.Parameter(torch.zeros(hidden_dim))
    nn.init.xavier_uniform_(self.attention)

  def forward(self, hidden: Tensor, edge_index: Tensor) -> Tensor:
    node_count = int(hidden.shape[0])
    device = hidden.device
    if edge_index.numel():
      source, target = edge_index.unbind(dim=-1)
      source = torch.cat((source, target))
      target = torch.cat((target, source[: len(target)]))
    else:
      source = torch.zeros((0,), dtype=torch.long, device=device)
      target = torch.zeros((0,), dtype=torch.long, device=device)
    self_nodes = torch.arange(node_count, dtype=torch.long, device=device)
    source = torch.cat((source, self_nodes))
    target = torch.cat((target, self_nodes))

    source_features = self.source(hidden).view(node_count, self.heads, self.head_dim)
    target_features = self.target(hidden).view(node_count, self.heads, self.head_dim)
    logits = (
        F.leaky_relu(
            source_features[source] + target_features[target],
            negative_slope=0.2,
        )
        * self.attention[None, :, :]
    ).sum(dim=-1)

    weights = torch.empty_like(logits)
    for head in range(self.heads):
      maximum = torch.full((node_count,), -torch.inf, device=device)
      maximum.scatter_reduce_(
          0, target, logits[:, head], reduce="amax", include_self=True
      )
      exponent = torch.exp(logits[:, head] - maximum[target])
      denominator = torch.zeros((node_count,), device=device)
      denominator.index_add_(0, target, exponent)
      weights[:, head] = exponent / denominator[target].clamp_min(1e-12)

    messages = source_features[source] * weights[:, :, None]
    aggregate = torch.zeros(
        (node_count, self.heads, self.head_dim),
        dtype=hidden.dtype,
        device=device,
    )
    aggregate.index_add_(0, target, messages)
    return aggregate.reshape(node_count, self.hidden_dim) + self.bias


class JoinABLeStylePrimitiveEncoderV1(nn.Module):
  """Two-layer graph-attention encoder followed by entity-level embeddings."""

  def __init__(self, *, hidden_dim: int = 64, heads: int = 8) -> None:
    super().__init__()
    self.hidden_dim = hidden_dim
    self.face_input = nn.Sequential(
        nn.Linear(NODE_DIM, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.gat1 = _GATv2LayerV1(hidden_dim, heads=heads)
    self.gat2 = _GATv2LayerV1(hidden_dim, heads=heads)
    self.primitive_input = nn.Sequential(
        nn.Linear(PRIMITIVE_FEATURE_DIM, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.face_fusion = nn.LayerNorm(hidden_dim)
    self.part_output = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )

  @staticmethod
  def _normalize_primitives(features: Tensor) -> Tensor:
    result = torch.nan_to_num(
        features, nan=0.0, posinf=30.0, neginf=-30.0
    ).clone()
    continuous = result[:, :8]
    result[:, :8] = torch.sign(continuous) * torch.log1p(torch.abs(continuous))
    return result

  def forward_many_with_primitives(
      self,
      graphs: list[LinkCADPrimitiveGraphV2] | tuple[LinkCADPrimitiveGraphV2, ...],
  ) -> tuple[Tensor, tuple[Tensor, ...]]:
    if not graphs:
      raise ValueError("JoinABLe-style graph batch is empty")
    device = next(self.parameters()).device
    node_rows = []
    edge_rows = []
    node_offsets = []
    offset = 0
    for graph in graphs:
      face_graph = graph.face_graph.to(device)
      node_offsets.append(offset)
      node_rows.append(face_graph.node_features)
      if face_graph.edge_index.numel():
        edge_rows.append(face_graph.edge_index + offset)
      offset += int(face_graph.node_features.shape[0])
    nodes = normalize_node_features(torch.cat(node_rows))
    edges = (
        torch.cat(edge_rows)
        if edge_rows else torch.zeros((0, 2), dtype=torch.long, device=device)
    )
    hidden = self.face_input(nodes)
    hidden = F.elu(self.gat1(hidden, edges))
    hidden = self.gat2(hidden, edges)

    primitive_sets = []
    part_rows = []
    for graph, node_offset in zip(graphs, node_offsets, strict=True):
      face_graph = graph.face_graph
      primitive = self.primitive_input(
          self._normalize_primitives(graph.primitive_features.to(device))
      )
      face_rows = []
      for members in face_graph.orbit_members:
        indices = torch.tensor(
            [node_offset + int(value) for value in members],
            dtype=torch.long,
            device=device,
        )
        face_rows.append(hidden[indices].mean(dim=0))
      if len(face_rows) != graph.face_orbit_count:
        raise ValueError("JoinABLe-style face/primitive identity differs")
      if face_rows:
        face_tensor = torch.stack(face_rows)
        primitive = primitive.clone()
        primitive[: graph.face_orbit_count] = self.face_fusion(
            primitive[: graph.face_orbit_count] + face_tensor
        )
      primitive_sets.append(primitive)
      part_rows.append(self.part_output(torch.cat((
          primitive.mean(dim=0), primitive.max(dim=0).values,
      ))))
    return torch.stack(part_rows), tuple(primitive_sets)


class JoinABLeStyleLinkCADV1(PrimitiveFactorizedLinkCADV2):
  """Language-free entity link prediction under LinkCAD's matched budgets."""

  def __init__(
      self,
      *,
      hidden_dim: int = 64,
      use_language: bool = False,
      use_global: bool = False,
      orbit_mode: str = "attention",
      max_orbits_per_candidate: int = 16,
      seed: int = 1701,
  ) -> None:
    if use_language or use_global:
      raise ValueError("JoinABLe-style baseline must remain language/global free")
    super().__init__(
        hidden_dim=hidden_dim,
        use_language=False,
        use_global=False,
        orbit_mode=orbit_mode,
        max_orbits_per_candidate=max_orbits_per_candidate,
        seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(seed + 211)
      self.joinable_encoder = JoinABLeStylePrimitiveEncoderV1(
          hidden_dim=hidden_dim, heads=8
      )
      self.link_hidden = nn.Sequential(
          nn.Linear(2 * hidden_dim, hidden_dim), nn.ELU(),
          nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
      )
      self.link_score = nn.Linear(hidden_dim, 1)

  def _encode_graphs(self, available_graphs):
    return self.joinable_encoder.forward_many_with_primitives(available_graphs)

  def _interface_hidden(self, inputs):
    left, right = torch.chunk(inputs[..., : 2 * self.hidden_dim], 2, dim=-1)
    forward = self.link_hidden(torch.cat((left, right), dim=-1))
    reverse = self.link_hidden(torch.cat((right, left), dim=-1))
    return 0.5 * (forward + reverse)

  def _interface_scores(self, hidden):
    return self.link_score(hidden)

  def _assignment_pair_scores(self, _part_scores, interface_scores):
    return interface_scores

  def forward(self, query: LinkCADQueryTensors) -> FactorizedOutput:
    output = super().forward(query)
    unary = torch.zeros_like(output.unary_logits)
    if query.candidate_mask is not None:
      unary = unary.masked_fill(~query.candidate_mask, -torch.inf)
    pair_logits = output.edge_pair_logits
    if query.port_candidate_mask is not None:
      expected = (int(query.edge_index.shape[0]), 2, query.candidate_count)
      if tuple(query.port_candidate_mask.shape) != expected:
        raise ValueError("JoinABLe-style port mask dimensions differ")
      pair_mask = (
          query.port_candidate_mask[:, 0, :, None]
          & query.port_candidate_mask[:, 1, None, :]
      )
      pair_logits = pair_logits.masked_fill(~pair_mask, -torch.inf)
    return replace(
        output,
        unary_logits=unary,
        edge_pair_logits=pair_logits,
    )


__all__ = [
    "JoinABLeStyleLinkCADV1",
    "JoinABLeStylePrimitiveEncoderV1",
    "PREDICTION_SCHEMA_VERSION",
]
