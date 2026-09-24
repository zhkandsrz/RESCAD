"""External B-Rep architectures adapted to LinkCAD candidate-set linking.

The implementations transfer the central modeling pattern of each cited work
while keeping LinkCAD's inputs, supervision, feasibility mask, and decoder.
They are architectural adaptations, not reproductions of the native training
pipelines or published checkpoints.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .linkcad_brep_encoder_v1 import EDGE_DIM, NODE_DIM, normalize_node_features
from .linkcad_factorized_model_v1 import FactorizedOutput
from .linkcad_primitive_factorized_model_v2 import PrimitiveFactorizedLinkCADV2
from .linkcad_primitive_graph_v2 import (
    LinkCADPrimitiveGraphV2,
    PRIMITIVE_FEATURE_DIM,
)


PREDICTION_SCHEMA_VERSION = "linkcad_external_brep_predictions.v1"


def _apply_public_port_mask(
    query, output: FactorizedOutput, *, zero_unary: bool,
) -> FactorizedOutput:
  unary = (
      torch.zeros_like(output.unary_logits)
      if zero_unary else output.unary_logits
  )
  if query.candidate_mask is not None:
    unary = unary.masked_fill(~query.candidate_mask, -torch.inf)
  pair_logits = output.edge_pair_logits
  if query.port_candidate_mask is not None:
    expected = (int(query.edge_index.shape[0]), 2, query.candidate_count)
    if tuple(query.port_candidate_mask.shape) != expected:
      raise ValueError("External B-Rep port mask dimensions differ")
    pair_mask = (
        query.port_candidate_mask[:, 0, :, None]
        & query.port_candidate_mask[:, 1, None, :]
    )
    pair_logits = pair_logits.masked_fill(~pair_mask, -torch.inf)
  return replace(
      output, unary_logits=unary, edge_pair_logits=pair_logits,
  )


class _StructuredBRepConv(nn.Module):
  """Face-topology convolution used by the AutoMate-style encoder."""

  def __init__(self, hidden_dim: int) -> None:
    super().__init__()
    self.message = nn.Sequential(
        nn.Linear(2 * hidden_dim + EDGE_DIM, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.update = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(
      self, hidden: Tensor, edge_index: Tensor, edge_features: Tensor,
  ) -> Tensor:
    aggregate = torch.zeros_like(hidden)
    degree = torch.zeros((hidden.shape[0], 1), device=hidden.device)
    if edge_index.numel():
      source, target = edge_index.unbind(dim=-1)
      forward = self.message(torch.cat((
          hidden[source], hidden[target], edge_features,
      ), dim=-1))
      reverse = self.message(torch.cat((
          hidden[target], hidden[source], edge_features,
      ), dim=-1))
      aggregate.index_add_(0, target, forward)
      aggregate.index_add_(0, source, reverse)
      ones = torch.ones((source.shape[0], 1), device=hidden.device)
      degree.index_add_(0, target, ones)
      degree.index_add_(0, source, ones)
    aggregate = aggregate / degree.clamp_min(1.0)
    return self.norm(hidden + self.update(torch.cat((hidden, aggregate), dim=-1)))


class AutoMateStylePrimitiveEncoderV1(nn.Module):
  """Hierarchical face-topology and B-Rep-entity encoder."""

  def __init__(self, *, hidden_dim: int = 64, layers: int = 3) -> None:
    super().__init__()
    self.face_input = nn.Sequential(
        nn.Linear(NODE_DIM, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.layers = nn.ModuleList(
        _StructuredBRepConv(hidden_dim) for _ in range(layers)
    )
    self.primitive_input = nn.Sequential(
        nn.Linear(PRIMITIVE_FEATURE_DIM, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.face_primitive_fusion = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.part_output = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )

  @staticmethod
  def _normalize_primitives(features: Tensor) -> Tensor:
    result = torch.nan_to_num(
        features, nan=0.0, posinf=30.0, neginf=-30.0,
    ).clone()
    continuous = result[:, :8]
    result[:, :8] = torch.sign(continuous) * torch.log1p(
        torch.abs(continuous)
    )
    return result

  def forward_many_with_primitives(
      self,
      graphs: list[LinkCADPrimitiveGraphV2] | tuple[LinkCADPrimitiveGraphV2, ...],
  ) -> tuple[Tensor, tuple[Tensor, ...]]:
    if not graphs:
      raise ValueError("AutoMate-style graph batch is empty")
    device = next(self.parameters()).device
    node_rows = []
    edge_rows = []
    edge_feature_rows = []
    offsets = []
    offset = 0
    for graph in graphs:
      face_graph = graph.face_graph.to(device)
      offsets.append(offset)
      node_rows.append(face_graph.node_features)
      if face_graph.edge_index.numel():
        edge_rows.append(face_graph.edge_index + offset)
        edge_feature_rows.append(face_graph.edge_features)
      offset += int(face_graph.node_features.shape[0])
    nodes = normalize_node_features(torch.cat(node_rows))
    edge_index = (
        torch.cat(edge_rows)
        if edge_rows else torch.zeros((0, 2), dtype=torch.long, device=device)
    )
    edge_features = (
        torch.cat(edge_feature_rows)
        if edge_feature_rows else torch.zeros((0, EDGE_DIM), device=device)
    )
    hidden = self.face_input(nodes)
    for layer in self.layers:
      hidden = layer(hidden, edge_index, edge_features)

    primitive_sets = []
    part_rows = []
    for graph, node_offset in zip(graphs, offsets, strict=True):
      primitive = self.primitive_input(
          self._normalize_primitives(graph.primitive_features.to(device))
      )
      face_rows = []
      for members in graph.face_graph.orbit_members:
        indices = torch.tensor(
            [node_offset + int(value) for value in members],
            dtype=torch.long, device=device,
        )
        face_rows.append(hidden[indices].mean(dim=0))
      if len(face_rows) != graph.face_orbit_count:
        raise ValueError("AutoMate-style face/primitive identity differs")
      if face_rows:
        face_tensor = torch.stack(face_rows)
        primitive = primitive.clone()
        primitive[: graph.face_orbit_count] = self.face_primitive_fusion(
            torch.cat((
                primitive[: graph.face_orbit_count], face_tensor,
            ), dim=-1)
        )
      primitive_sets.append(primitive)
      part_rows.append(self.part_output(torch.cat((
          primitive.mean(dim=0), primitive.max(dim=0).values,
      ))))
    return torch.stack(part_rows), tuple(primitive_sets)


class AutoMateStyleLinkCADV1(PrimitiveFactorizedLinkCADV2):
  """Language-free SB-GCN-style mate scorer under LinkCAD's decoder."""

  def __init__(
      self, *, hidden_dim: int = 64, use_language: bool = False,
      use_global: bool = False, orbit_mode: str = "attention",
      max_orbits_per_candidate: int = 16, seed: int = 1701,
  ) -> None:
    if use_language or use_global:
      raise ValueError("AutoMate-style baseline must remain language/global free")
    super().__init__(
        hidden_dim=hidden_dim, use_language=False, use_global=False,
        orbit_mode=orbit_mode,
        max_orbits_per_candidate=max_orbits_per_candidate, seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(seed + 307)
      self.automate_encoder = AutoMateStylePrimitiveEncoderV1(
          hidden_dim=hidden_dim,
      )
      self.mate_hidden = nn.Sequential(
          nn.Linear(4 * hidden_dim, 2 * hidden_dim), nn.ReLU(),
          nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
      )
      self.mate_score = nn.Linear(hidden_dim, 1)

  def _encode_graphs(self, available_graphs):
    return self.automate_encoder.forward_many_with_primitives(available_graphs)

  def _interface_hidden(self, inputs):
    left, right = torch.chunk(inputs[..., : 2 * self.hidden_dim], 2, dim=-1)
    symmetric = torch.cat((
        left + right, torch.abs(left - right), left * right,
        0.5 * (left.square() + right.square()),
    ), dim=-1)
    return self.mate_hidden(symmetric)

  def _interface_scores(self, hidden):
    return self.mate_score(hidden)

  def _assignment_pair_scores(self, _part_scores, interface_scores):
    return interface_scores

  def forward(self, query) -> FactorizedOutput:
    return _apply_public_port_mask(
        query, super().forward(query), zero_unary=True,
    )


class _CosinePointerProposal(nn.Module):
  def __init__(self, hidden_dim: int) -> None:
    super().__init__()
    self.hidden_dim = hidden_dim
    self.entity = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.log_scale = nn.Parameter(torch.tensor(2.0))

  def forward(self, inputs: Tensor) -> Tensor:
    entity = inputs[..., : self.hidden_dim]
    query = inputs[..., self.hidden_dim : 2 * self.hidden_dim]
    score = (
        F.normalize(self.entity(entity), dim=-1)
        * F.normalize(self.query(query), dim=-1)
    ).sum(dim=-1, keepdim=True)
    return score * self.log_scale.exp().clamp(max=100.0)


class _PointerPairInteraction(nn.Module):
  def __init__(self, hidden_dim: int) -> None:
    super().__init__()
    self.hidden_dim = hidden_dim
    self.entity_pair = nn.Sequential(
        nn.Linear(4 * hidden_dim, 2 * hidden_dim), nn.GELU(),
        nn.Linear(2 * hidden_dim, hidden_dim),
    )
    self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)

  def forward(self, inputs: Tensor) -> Tensor:
    left, right, language = torch.chunk(inputs, 7, dim=-1)[:3]
    pair = self.entity_pair(torch.cat((
        left + right, torch.abs(left - right), left * right,
        0.5 * (left.square() + right.square()),
    ), dim=-1))
    return F.normalize(pair, dim=-1) * F.normalize(
        self.query(language), dim=-1,
    )


class _PointerPairScore(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.log_scale = nn.Parameter(torch.tensor(2.0))

  def forward(self, hidden: Tensor) -> Tensor:
    return hidden.sum(dim=-1, keepdim=True) * self.log_scale.exp().clamp(
        max=100.0,
    )


class PointerCADStyleLinkCADV1(PrimitiveFactorizedLinkCADV2):
  """Text-conditioned face/edge pointer adapted to assembly interfaces."""

  def __init__(
      self, *, hidden_dim: int = 128, use_language: bool = True,
      use_global: bool = True, orbit_mode: str = "attention",
      max_orbits_per_candidate: int = 16, seed: int = 1701,
  ) -> None:
    if not use_language:
      raise ValueError("Pointer-CAD-style baseline requires language")
    super().__init__(
        hidden_dim=hidden_dim, use_language=True, use_global=use_global,
        orbit_mode=orbit_mode,
        max_orbits_per_candidate=max_orbits_per_candidate, seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(seed + 401)
      self.orbit_proposal = _CosinePointerProposal(hidden_dim)
      self.interface_edge = _PointerPairInteraction(hidden_dim)
      self.interface_score = _PointerPairScore()

  def _assignment_pair_scores(self, part_scores, interface_scores):
    return part_scores + interface_scores

  def forward(self, query) -> FactorizedOutput:
    return _apply_public_port_mask(
        query, super().forward(query), zero_unary=False,
    )


MODEL_CLASSES = {
    "automate_style": AutoMateStyleLinkCADV1,
    "pointercad_style": PointerCADStyleLinkCADV1,
}


__all__ = [
    "AutoMateStyleLinkCADV1",
    "AutoMateStylePrimitiveEncoderV1",
    "MODEL_CLASSES",
    "PREDICTION_SCHEMA_VERSION",
    "PointerCADStyleLinkCADV1",
]
