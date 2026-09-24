"""Torch-only B-Rep program model with no CAD/authority dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn


FORMAL_CATALOG_SIZE = 19
NODE_INPUT_DIM = 21
EDGE_INPUT_DIM = 3


@dataclass(frozen=True, slots=True)
class BRepProgramLearnerConfig:
  catalog_size: int = FORMAL_CATALOG_SIZE
  hidden_dim: int = 96
  layers: int = 4
  dropout: float = 0.0
  translation_loss_weight: float = 0.25
  rotation_loss_weight: float = 0.25

  def __post_init__(self) -> None:
    if type(self.catalog_size) is not int or self.catalog_size < 2:
      raise ValueError("catalog_size must be an actual integer of at least two")
    if type(self.hidden_dim) is not int or self.hidden_dim < 8:
      raise ValueError("hidden_dim must be an actual integer of at least eight")
    if type(self.layers) is not int or self.layers < 1:
      raise ValueError("layers must be a positive actual integer")
    if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
      raise ValueError("dropout must be finite in [0, 1)")
    for label, value in (
        ("translation_loss_weight", self.translation_loss_weight),
        ("rotation_loss_weight", self.rotation_loss_weight),
    ):
      if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LearnerBatchBudget:
  max_batch_size: int = 32
  max_graphs: int = 2
  max_faces_per_graph: int = 1024
  max_edges_per_graph: int = 4096
  max_candidates: int = FORMAL_CATALOG_SIZE

  def __post_init__(self) -> None:
    if any(
        type(value) is not int or value < 1 for value in asdict(self).values()
    ):
      raise ValueError("learner batch budgets must be positive actual integers")
    if (
        self.max_graphs > 2
        or self.max_faces_per_graph > 1024
        or self.max_edges_per_graph > 4096
        or self.max_candidates > 512
    ):
      raise ValueError("learner batch budget exceeds the frozen graph policy")


@dataclass(frozen=True, slots=True)
class BRepGraphExample:
  node_features: Tensor
  edge_index: Tensor
  edge_features: Tensor
  endpoint_markers: Tensor


@dataclass(frozen=True, slots=True)
class BRepLearnerExample:
  graphs: tuple[BRepGraphExample, ...]
  candidate_program_indices: Tensor
  target_program_index: int
  residual_translation: Tensor
  residual_rotation_vector: Tensor
  residual_mask: bool = True
  model_view_sha256: str = "synthetic"


@dataclass(frozen=True, slots=True)
class BRepProgramInputs:
  node_features: Tensor
  node_mask: Tensor
  edge_index: Tensor
  edge_features: Tensor
  edge_mask: Tensor
  endpoint_markers: Tensor
  graph_mask: Tensor
  candidate_program_indices: Tensor
  candidate_mask: Tensor

  def to(self, device: torch.device | str) -> "BRepProgramInputs":
    return BRepProgramInputs(**{
        field: getattr(self, field).to(device)
        for field in self.__dataclass_fields__
    })


@dataclass(frozen=True, slots=True)
class BRepProgramTargets:
  target_positions: Tensor
  residual_translation: Tensor
  residual_rotation_vector: Tensor
  residual_mask: Tensor

  def to(self, device: torch.device | str) -> "BRepProgramTargets":
    return BRepProgramTargets(
        target_positions=self.target_positions.to(device),
        residual_translation=self.residual_translation.to(device),
        residual_rotation_vector=self.residual_rotation_vector.to(device),
        residual_mask=self.residual_mask.to(device),
    )


@dataclass(frozen=True, slots=True)
class BRepProgramBatch:
  inputs: BRepProgramInputs
  targets: BRepProgramTargets
  model_view_sha256s: tuple[str, ...]

  def to(self, device: torch.device | str) -> "BRepProgramBatch":
    return BRepProgramBatch(
        inputs=self.inputs.to(device),
        targets=self.targets.to(device),
        model_view_sha256s=self.model_view_sha256s,
    )


@dataclass(frozen=True, slots=True)
class BRepProgramOutput:
  logits: Tensor
  residual_translation: Tensor
  residual_rotation_vector: Tensor


def _validate_graph(graph: BRepGraphExample) -> None:
  if (
      type(graph) is not BRepGraphExample
      or graph.node_features.ndim != 2
      or graph.node_features.shape[1] != NODE_INPUT_DIM
      or graph.node_features.shape[0] < 1
      or not torch.isfinite(graph.node_features).all()
      or graph.edge_index.ndim != 2
      or graph.edge_index.shape[1] != 2
      or graph.edge_index.dtype != torch.long
      or graph.edge_features.shape != (graph.edge_index.shape[0], EDGE_INPUT_DIM)
      or not torch.isfinite(graph.edge_features).all()
      or graph.endpoint_markers.shape != (graph.node_features.shape[0], 2)
      or not torch.isfinite(graph.endpoint_markers).all()
  ):
    raise ValueError("B-Rep graph tensor contract differs")
  if graph.edge_index.numel() and (
      int(graph.edge_index.min()) < 0
      or int(graph.edge_index.max()) >= graph.node_features.shape[0]
  ):
    raise ValueError("B-Rep edge index is out of range")


def collate_nonformal_brep_learner_examples(
    examples: Sequence[BRepLearnerExample],
    *,
    budget: LearnerBatchBudget = LearnerBatchBudget(),
) -> BRepProgramBatch:
  if not examples or len(examples) > budget.max_batch_size:
    raise ValueError("learner batch is empty or exceeds its budget")
  for example in examples:
    if type(example) is not BRepLearnerExample:
      raise TypeError("learner collator accepts typed examples only")
    if not 1 <= len(example.graphs) <= budget.max_graphs:
      raise ValueError("learner graph count differs")
    for graph in example.graphs:
      _validate_graph(graph)
      if graph.node_features.shape[0] > budget.max_faces_per_graph:
        raise ValueError("learner face count exceeds its budget")
      if graph.edge_index.shape[0] > budget.max_edges_per_graph:
        raise ValueError("learner edge count exceeds its budget")
    candidates = example.candidate_program_indices
    if (
        candidates.ndim != 1
        or candidates.dtype != torch.long
        or not 2 <= candidates.shape[0] <= budget.max_candidates
        or len(set(int(value) for value in candidates.tolist()))
        != candidates.shape[0]
        or example.target_program_index not in candidates.tolist()
    ):
      raise ValueError("learner candidate contract differs")
  batch_size = len(examples)
  max_graphs = max(len(row.graphs) for row in examples)
  max_nodes = max(
      graph.node_features.shape[0] for row in examples for graph in row.graphs
  )
  max_edges = max(
      graph.edge_index.shape[0] for row in examples for graph in row.graphs
  )
  max_candidates = max(
      row.candidate_program_indices.shape[0] for row in examples
  )
  nodes = torch.zeros((batch_size, max_graphs, max_nodes, NODE_INPUT_DIM))
  node_mask = torch.zeros((batch_size, max_graphs, max_nodes), dtype=torch.bool)
  edges = torch.zeros(
      (batch_size, max_graphs, max_edges, 2), dtype=torch.long
  )
  edge_features = torch.zeros(
      (batch_size, max_graphs, max_edges, EDGE_INPUT_DIM)
  )
  edge_mask = torch.zeros((batch_size, max_graphs, max_edges), dtype=torch.bool)
  markers = torch.zeros((batch_size, max_graphs, max_nodes, 2))
  graph_mask = torch.zeros((batch_size, max_graphs), dtype=torch.bool)
  candidate_ids = torch.zeros(
      (batch_size, max_candidates), dtype=torch.long
  )
  candidate_mask = torch.zeros(
      (batch_size, max_candidates), dtype=torch.bool
  )
  target_positions = torch.zeros(batch_size, dtype=torch.long)
  translations = torch.zeros((batch_size, 3))
  rotations = torch.zeros((batch_size, 3))
  residual_mask = torch.zeros(batch_size, dtype=torch.bool)
  hashes: list[str] = []
  for batch_index, example in enumerate(examples):
    for graph_index, graph in enumerate(example.graphs):
      node_count = graph.node_features.shape[0]
      edge_count = graph.edge_index.shape[0]
      nodes[batch_index, graph_index, :node_count] = graph.node_features
      node_mask[batch_index, graph_index, :node_count] = True
      markers[batch_index, graph_index, :node_count] = graph.endpoint_markers
      graph_mask[batch_index, graph_index] = True
      if edge_count:
        edges[batch_index, graph_index, :edge_count] = graph.edge_index
        edge_features[batch_index, graph_index, :edge_count] = graph.edge_features
        edge_mask[batch_index, graph_index, :edge_count] = True
    count = example.candidate_program_indices.shape[0]
    candidate_ids[batch_index, :count] = example.candidate_program_indices
    candidate_mask[batch_index, :count] = True
    target_positions[batch_index] = (
        example.candidate_program_indices.tolist().index(
            example.target_program_index
        )
    )
    translations[batch_index] = example.residual_translation
    rotations[batch_index] = example.residual_rotation_vector
    residual_mask[batch_index] = example.residual_mask
    hashes.append(example.model_view_sha256)
  return BRepProgramBatch(
      inputs=BRepProgramInputs(
          node_features=nodes,
          node_mask=node_mask,
          edge_index=edges,
          edge_features=edge_features,
          edge_mask=edge_mask,
          endpoint_markers=markers,
          graph_mask=graph_mask,
          candidate_program_indices=candidate_ids,
          candidate_mask=candidate_mask,
      ),
      targets=BRepProgramTargets(
          target_positions=target_positions,
          residual_translation=translations,
          residual_rotation_vector=rotations,
          residual_mask=residual_mask,
      ),
      model_view_sha256s=tuple(hashes),
  )


class _MessagePassingLayer(nn.Module):
  def __init__(self, hidden_dim: int, dropout: float) -> None:
    super().__init__()
    self.message = nn.Sequential(
        nn.Linear(2 * hidden_dim + EDGE_INPUT_DIM, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.update = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(
      self, hidden: Tensor, edge_index: Tensor, edge_features: Tensor
  ) -> Tensor:
    if edge_index.numel() == 0:
      aggregate = torch.zeros_like(hidden)
    else:
      source, target = edge_index.unbind(dim=-1)
      forward = self.message(
          torch.cat((hidden[source], hidden[target], edge_features), dim=-1)
      )
      reverse = self.message(
          torch.cat((hidden[target], hidden[source], edge_features), dim=-1)
      )
      aggregate = torch.zeros_like(hidden)
      degree = torch.zeros(
          (hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device
      )
      aggregate.index_add_(0, target, forward)
      aggregate.index_add_(0, source, reverse)
      ones = torch.ones(
          (source.shape[0], 1), dtype=hidden.dtype, device=hidden.device
      )
      degree.index_add_(0, target, ones)
      degree.index_add_(0, source, ones)
      aggregate = aggregate / degree.clamp_min(1.0)
    return self.norm(
        hidden + self.update(torch.cat((hidden, aggregate), dim=-1))
    )


class BRepProgramLearnerV1(nn.Module):
  """Permutation-invariant endpoint-centred fixed-catalog scorer."""

  def __init__(self, config: BRepProgramLearnerConfig, *, seed: int) -> None:
    super().__init__()
    if type(config) is not BRepProgramLearnerConfig:
      raise TypeError("learner config type differs")
    self.config = config
    self.seed = int(seed)
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(self.seed)
      self.node_encoder = nn.Sequential(
          nn.Linear(NODE_INPUT_DIM, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.layers = nn.ModuleList(
          _MessagePassingLayer(config.hidden_dim, config.dropout)
          for _ in range(config.layers)
      )
      self.graph_pool = nn.Sequential(
          nn.Linear(3 * config.hidden_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.query_encoder = nn.Sequential(
          nn.Linear(4 * config.hidden_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.catalog_embedding = nn.Embedding(
          config.catalog_size, config.hidden_dim
      )
      cross_dim = 4 * config.hidden_dim
      self.cross_scorer = nn.Sequential(
          nn.Linear(cross_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Dropout(config.dropout),
          nn.Linear(config.hidden_dim, 1),
      )
      self.residual_head = nn.Sequential(
          nn.Linear(cross_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Dropout(config.dropout),
          nn.Linear(config.hidden_dim, 6),
      )

  @property
  def parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters())

  def _encode_graphs(self, inputs: BRepProgramInputs) -> Tensor:
    batch_size, graph_count, _, _ = inputs.node_features.shape
    pooled = torch.zeros(
        (batch_size, graph_count, self.config.hidden_dim),
        dtype=inputs.node_features.dtype,
        device=inputs.node_features.device,
    )
    for batch_index in range(batch_size):
      for graph_index in range(graph_count):
        if not bool(inputs.graph_mask[batch_index, graph_index]):
          continue
        mask = inputs.node_mask[batch_index, graph_index]
        hidden = self.node_encoder(
            inputs.node_features[batch_index, graph_index, mask]
        )
        edge_mask = inputs.edge_mask[batch_index, graph_index]
        edge_index = inputs.edge_index[batch_index, graph_index, edge_mask]
        edge_features = inputs.edge_features[
            batch_index, graph_index, edge_mask
        ]
        for layer in self.layers:
          hidden = layer(hidden, edge_index, edge_features)
        endpoint_markers = inputs.endpoint_markers[
            batch_index, graph_index, mask
        ]
        whole = hidden.mean(dim=0)
        marked = []
        for endpoint in range(2):
          weights = endpoint_markers[:, endpoint:endpoint + 1]
          marked.append(
              (hidden * weights).sum(dim=0) / weights.sum().clamp_min(1.0)
          )
        pooled[batch_index, graph_index] = self.graph_pool(
            torch.cat((whole, marked[0], marked[1]), dim=-1)
        )
    return pooled

  def forward(self, inputs: BRepProgramInputs) -> BRepProgramOutput:
    if type(inputs) is not BRepProgramInputs:
      raise TypeError("learner forward input type differs")
    if inputs.candidate_program_indices.numel() and (
        int(inputs.candidate_program_indices.min()) < 0
        or int(inputs.candidate_program_indices.max()) >= self.config.catalog_size
    ):
      raise ValueError("candidate index lies outside the fixed catalog")
    graph_embeddings = self._encode_graphs(inputs)
    if graph_embeddings.shape[1] == 1:
      graph_embeddings = torch.cat(
          (graph_embeddings, torch.zeros_like(graph_embeddings)), dim=1
      )
    left, right = graph_embeddings[:, 0], graph_embeddings[:, 1]
    query = self.query_encoder(
        torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
    )
    candidates = self.catalog_embedding(inputs.candidate_program_indices)
    expanded = query.unsqueeze(1).expand_as(candidates)
    cross = torch.cat(
        (expanded, candidates, expanded * candidates,
         torch.abs(expanded - candidates)),
        dim=-1,
    )
    logits = self.cross_scorer(cross).squeeze(-1)
    logits = logits.masked_fill(~inputs.candidate_mask, -torch.inf)
    residual = self.residual_head(cross)
    return BRepProgramOutput(
        logits=logits,
        residual_translation=residual[..., :3],
        residual_rotation_vector=residual[..., 3:],
    )
