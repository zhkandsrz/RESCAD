"""Label-free joint interface and finite-program learner.

This module deliberately does not reuse ``BRepProgramInputs``: that legacy
schema contains authority-selected face markers.  The only model input here is
two complete, intrinsic B-Rep graphs plus the fixed train-only program catalog.
Discrete face/program supervision and continuous graph-local residuals live in
``JointInterfaceTargetsV1`` and are accepted only by the loss function.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .mate_pose_retriever import MateProgramRetrieval
from .program_class_reranker import mate_program_class_from_program
from .program_proposer import (
    PROGRAM_PROPOSER_PROTOCOL,
    ProgramHypothesis,
    ProposalBudget,
)


SCHEMA_VERSION = "joint_interface_program_learner.v1"
INPUT_SCHEMA_VERSION = "unannotated_two_part_brep_inputs.v1"
GRAPH_SCHEMA_VERSION = "unannotated_intrinsic_brep_graph.v1"
TARGET_SCHEMA_VERSION = "joint_interface_program_targets.v1"
FORMAL_CATALOG_SIZE = 19
SURFACE_TYPES = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "spline",
    "other",
)
INTRINSIC_NODE_FEATURES = (
    "area_fraction",
    "perimeter_over_sqrt_total_area",
    "mean_curvature_times_sqrt_total_area",
    "gaussian_curvature_times_total_area",
    "boundary_loop_count",
    "boundary_edge_count",
)
LABEL_FREE_NODE_FEATURE_NAMES = (
    tuple(f"{name}_value" for name in INTRINSIC_NODE_FEATURES)
    + tuple(f"{name}_present" for name in INTRINSIC_NODE_FEATURES)
    + tuple(f"surface_is_{name}" for name in SURFACE_TYPES)
)
LABEL_FREE_EDGE_FEATURE_NAMES = (
    "shared_edge_length_over_sqrt_total_area",
    "dihedral_cos",
    "dihedral_sin_abs",
)
LABEL_FREE_NODE_INPUT_DIM = len(LABEL_FREE_NODE_FEATURE_NAMES)
LABEL_FREE_EDGE_INPUT_DIM = len(LABEL_FREE_EDGE_FEATURE_NAMES)
_FORBIDDEN_MODEL_INPUT_TOKENS = (
    "endpoint_marker",
    "gold",
    "label",
    "target",
    "supervision",
    "capability",
)


def _contains_forbidden_input_key(value: Any) -> str | None:
  if isinstance(value, Mapping):
    for key, child in value.items():
      normalized = str(key).strip().lower()
      for token in _FORBIDDEN_MODEL_INPUT_TOKENS:
        if token in normalized:
          return normalized
      nested = _contains_forbidden_input_key(child)
      if nested is not None:
        return nested
  elif isinstance(value, (list, tuple)):
    for child in value:
      nested = _contains_forbidden_input_key(child)
      if nested is not None:
        return nested
  return None


@dataclass(frozen=True, slots=True)
class LabelFreeBRepGraphV1:
  """One full part graph with no selected-face or authority marker channel."""

  node_features: Tensor
  edge_index: Tensor
  edge_features: Tensor
  node_feature_names: tuple[str, ...] = LABEL_FREE_NODE_FEATURE_NAMES
  edge_feature_names: tuple[str, ...] = LABEL_FREE_EDGE_FEATURE_NAMES
  schema_version: str = GRAPH_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != GRAPH_SCHEMA_VERSION:
      raise ValueError("label-free graph schema version differs")
    if tuple(self.node_feature_names) != LABEL_FREE_NODE_FEATURE_NAMES:
      raise ValueError("node feature schema is not the frozen label-free schema")
    if tuple(self.edge_feature_names) != LABEL_FREE_EDGE_FEATURE_NAMES:
      raise ValueError("edge feature schema is not the frozen intrinsic schema")
    if (
        self.node_features.ndim != 2
        or self.node_features.shape[0] < 1
        or self.node_features.shape[1] != LABEL_FREE_NODE_INPUT_DIM
        or not torch.is_floating_point(self.node_features)
        or not bool(torch.isfinite(self.node_features).all())
    ):
      raise ValueError("label-free graph node tensor is malformed")
    if (
        self.edge_index.ndim != 2
        or self.edge_index.shape[1] != 2
        or self.edge_index.dtype != torch.long
    ):
      raise ValueError("label-free graph edge index is malformed")
    if self.edge_features.shape != (
        self.edge_index.shape[0],
        LABEL_FREE_EDGE_INPUT_DIM,
    ) or not bool(torch.isfinite(self.edge_features).all()):
      raise ValueError("label-free graph edge features are malformed")
    if self.edge_index.numel() and (
        int(self.edge_index.min()) < 0
        or int(self.edge_index.max()) >= self.node_features.shape[0]
    ):
      raise ValueError("label-free graph edge index is out of range")

  @classmethod
  def from_mapping(cls, payload: Mapping[str, Any]) -> "LabelFreeBRepGraphV1":
    """Parse a strict public tensor schema and fail closed on marker fields."""

    if not isinstance(payload, Mapping):
      raise TypeError("label-free graph payload must be an object")
    forbidden = _contains_forbidden_input_key(payload)
    if forbidden is not None:
      raise ValueError(f"model input contains forbidden field: {forbidden}")
    if set(payload) != {
        "schema_version",
        "node_feature_names",
        "edge_feature_names",
        "node_features",
        "edge_index",
        "edge_features",
    }:
      raise ValueError("label-free graph payload fields differ")
    edge_index = torch.as_tensor(payload["edge_index"], dtype=torch.long).clone()
    edge_features = torch.as_tensor(
        payload["edge_features"], dtype=torch.float32
    ).clone()
    if edge_index.numel() == 0:
      edge_index = edge_index.reshape(0, 2)
    if edge_features.numel() == 0:
      edge_features = edge_features.reshape(0, LABEL_FREE_EDGE_INPUT_DIM)
    return cls(
        node_features=torch.as_tensor(
            payload["node_features"], dtype=torch.float32
        ).clone(),
        edge_index=edge_index,
        edge_features=edge_features,
        node_feature_names=tuple(str(v) for v in payload["node_feature_names"]),
        edge_feature_names=tuple(str(v) for v in payload["edge_feature_names"]),
        schema_version=str(payload["schema_version"]),
    )


@dataclass(frozen=True, slots=True)
class LabelFreeBRepPairV1:
  part_a: LabelFreeBRepGraphV1
  part_b: LabelFreeBRepGraphV1

  def __post_init__(self) -> None:
    if type(self.part_a) is not LabelFreeBRepGraphV1:
      raise TypeError("part_a must be a label-free B-Rep graph")
    if type(self.part_b) is not LabelFreeBRepGraphV1:
      raise TypeError("part_b must be a label-free B-Rep graph")


@dataclass(frozen=True, slots=True)
class JointInterfaceBatchBudgetV1:
  max_batch_size: int = 32
  max_faces_per_part: int = 2048
  max_edges_per_part: int = 4096

  def __post_init__(self) -> None:
    if any(
        type(value) is not int or value < 1
        for value in (
            self.max_batch_size,
            self.max_faces_per_part,
            self.max_edges_per_part,
        )
    ):
      raise ValueError("joint learner budgets must be positive actual integers")
    if self.max_faces_per_part > 2048 or self.max_edges_per_part > 4096:
      raise ValueError("joint learner budget exceeds the frozen graph policy")


@dataclass(frozen=True, slots=True)
class JointInterfaceInputsV1:
  """The complete forward boundary: two graphs and 19 opaque catalog IDs."""

  node_features: Tensor
  node_mask: Tensor
  edge_index: Tensor
  edge_features: Tensor
  edge_mask: Tensor
  candidate_program_indices: Tensor
  node_feature_names: tuple[str, ...] = LABEL_FREE_NODE_FEATURE_NAMES
  edge_feature_names: tuple[str, ...] = LABEL_FREE_EDGE_FEATURE_NAMES
  schema_version: str = INPUT_SCHEMA_VERSION

  def to(self, device: torch.device | str) -> "JointInterfaceInputsV1":
    return JointInterfaceInputsV1(
        node_features=self.node_features.to(device),
        node_mask=self.node_mask.to(device),
        edge_index=self.edge_index.to(device),
        edge_features=self.edge_features.to(device),
        edge_mask=self.edge_mask.to(device),
        candidate_program_indices=self.candidate_program_indices.to(device),
        node_feature_names=self.node_feature_names,
        edge_feature_names=self.edge_feature_names,
        schema_version=self.schema_version,
    )


def _validate_inputs(inputs: JointInterfaceInputsV1, *, catalog_size: int) -> None:
  if type(inputs) is not JointInterfaceInputsV1:
    raise TypeError("joint learner accepts JointInterfaceInputsV1 only")
  if inputs.schema_version != INPUT_SCHEMA_VERSION:
    raise ValueError("joint learner input schema differs")
  if tuple(inputs.node_feature_names) != LABEL_FREE_NODE_FEATURE_NAMES:
    raise ValueError("joint learner refuses non-label-free node features")
  if tuple(inputs.edge_feature_names) != LABEL_FREE_EDGE_FEATURE_NAMES:
    raise ValueError("joint learner refuses non-intrinsic edge features")
  if inputs.node_features.ndim != 4 or inputs.node_features.shape[1] != 2:
    raise ValueError("joint learner requires exactly two padded part graphs")
  batch, _, max_nodes, node_dim = inputs.node_features.shape
  if node_dim != LABEL_FREE_NODE_INPUT_DIM or batch < 1 or max_nodes < 1:
    raise ValueError("joint learner node tensor shape differs")
  if inputs.node_mask.shape != (batch, 2, max_nodes):
    raise ValueError("joint learner node mask shape differs")
  if inputs.edge_index.ndim != 4 or inputs.edge_index.shape[:2] != (batch, 2):
    raise ValueError("joint learner edge index shape differs")
  max_edges = inputs.edge_index.shape[2]
  if inputs.edge_index.shape[3] != 2:
    raise ValueError("joint learner edge index final dimension differs")
  if inputs.edge_features.shape != (
      batch,
      2,
      max_edges,
      LABEL_FREE_EDGE_INPUT_DIM,
  ) or inputs.edge_mask.shape != (batch, 2, max_edges):
    raise ValueError("joint learner edge tensor shape differs")
  if not bool(torch.isfinite(inputs.node_features).all()) or not bool(
      torch.isfinite(inputs.edge_features).all()
  ):
    raise ValueError("joint learner inputs must be finite")
  if not bool(inputs.node_mask.any(dim=-1).all()):
    raise ValueError("each joint learner part must contain at least one face")
  candidates = inputs.candidate_program_indices
  if candidates.shape != (batch, catalog_size) or candidates.dtype != torch.long:
    raise ValueError("joint learner requires the complete fixed catalog")
  expected = set(range(catalog_size))
  for row in candidates.detach().cpu().tolist():
    if set(int(value) for value in row) != expected:
      raise ValueError("joint learner candidate catalog is incomplete or repeated")
  for batch_index in range(batch):
    for part_index in range(2):
      node_count = int(inputs.node_mask[batch_index, part_index].sum())
      edge_mask = inputs.edge_mask[batch_index, part_index]
      local_edges = inputs.edge_index[batch_index, part_index, edge_mask]
      if local_edges.numel() and (
          int(local_edges.min()) < 0 or int(local_edges.max()) >= node_count
      ):
        raise ValueError("joint learner padded edge index is out of range")


def collate_label_free_brep_pairs_v1(
    pairs: Sequence[LabelFreeBRepPairV1],
    *,
    candidate_program_indices: Sequence[int] = tuple(range(FORMAL_CATALOG_SIZE)),
    budget: JointInterfaceBatchBudgetV1 = JointInterfaceBatchBudgetV1(),
) -> JointInterfaceInputsV1:
  if not pairs or len(pairs) > budget.max_batch_size:
    raise ValueError("joint learner batch is empty or exceeds its budget")
  if (
      len(candidate_program_indices) != FORMAL_CATALOG_SIZE
      or set(int(value) for value in candidate_program_indices)
      != set(range(FORMAL_CATALOG_SIZE))
  ):
    raise ValueError("joint learner requires all 19 frozen catalog positions")
  graphs: list[LabelFreeBRepGraphV1] = []
  for pair in pairs:
    if type(pair) is not LabelFreeBRepPairV1:
      raise TypeError("joint learner collator accepts label-free pairs only")
    for graph in (pair.part_a, pair.part_b):
      if graph.node_features.shape[0] > budget.max_faces_per_part:
        raise ValueError("joint learner face budget exceeded")
      if graph.edge_index.shape[0] > budget.max_edges_per_part:
        raise ValueError("joint learner edge budget exceeded")
      graphs.append(graph)
  batch = len(pairs)
  max_nodes = max(graph.node_features.shape[0] for graph in graphs)
  max_edges = max(graph.edge_index.shape[0] for graph in graphs)
  nodes = torch.zeros((batch, 2, max_nodes, LABEL_FREE_NODE_INPUT_DIM))
  node_mask = torch.zeros((batch, 2, max_nodes), dtype=torch.bool)
  edges = torch.zeros((batch, 2, max_edges, 2), dtype=torch.long)
  edge_features = torch.zeros(
      (batch, 2, max_edges, LABEL_FREE_EDGE_INPUT_DIM)
  )
  edge_mask = torch.zeros((batch, 2, max_edges), dtype=torch.bool)
  for batch_index, pair in enumerate(pairs):
    for part_index, graph in enumerate((pair.part_a, pair.part_b)):
      node_count = graph.node_features.shape[0]
      edge_count = graph.edge_index.shape[0]
      nodes[batch_index, part_index, :node_count] = graph.node_features
      node_mask[batch_index, part_index, :node_count] = True
      if edge_count:
        edges[batch_index, part_index, :edge_count] = graph.edge_index
        edge_features[batch_index, part_index, :edge_count] = graph.edge_features
        edge_mask[batch_index, part_index, :edge_count] = True
  candidate_row = torch.tensor(candidate_program_indices, dtype=torch.long)
  result = JointInterfaceInputsV1(
      node_features=nodes,
      node_mask=node_mask,
      edge_index=edges,
      edge_features=edge_features,
      edge_mask=edge_mask,
      candidate_program_indices=candidate_row.unsqueeze(0).repeat(batch, 1),
  )
  _validate_inputs(result, catalog_size=FORMAL_CATALOG_SIZE)
  return result


@dataclass(frozen=True, slots=True)
class JointInterfaceTargetsV1:
  """Training-only joint labels; no method in the inference path accepts it."""

  face_index_a: Tensor
  face_index_b: Tensor
  program_index: Tensor
  residual_translation: Tensor
  residual_rotation_vector: Tensor
  residual_mask: Tensor
  schema_version: str = TARGET_SCHEMA_VERSION

  def to(self, device: torch.device | str) -> "JointInterfaceTargetsV1":
    return JointInterfaceTargetsV1(
        face_index_a=self.face_index_a.to(device),
        face_index_b=self.face_index_b.to(device),
        program_index=self.program_index.to(device),
        residual_translation=self.residual_translation.to(device),
        residual_rotation_vector=self.residual_rotation_vector.to(device),
        residual_mask=self.residual_mask.to(device),
        schema_version=self.schema_version,
    )


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramLearnerConfigV1:
  catalog_size: int = FORMAL_CATALOG_SIZE
  hidden_dim: int = 96
  layers: int = 4
  top_l: int = 8
  dropout: float = 0.0
  face_salience_loss_weight: float = 0.5
  translation_loss_weight: float = 0.25
  rotation_loss_weight: float = 0.25

  def __post_init__(self) -> None:
    if self.catalog_size != FORMAL_CATALOG_SIZE:
      raise ValueError("joint learner catalog is frozen to 19 programs")
    if type(self.hidden_dim) is not int or self.hidden_dim < 8:
      raise ValueError("joint learner hidden_dim must be at least eight")
    if type(self.layers) is not int or self.layers < 1:
      raise ValueError("joint learner layers must be positive")
    if type(self.top_l) is not int or self.top_l < 1 or self.top_l > 64:
      raise ValueError("joint learner top_l must be in [1, 64]")
    if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
      raise ValueError("joint learner dropout must be finite in [0, 1)")
    for value in (
        self.face_salience_loss_weight,
        self.translation_loss_weight,
        self.rotation_loss_weight,
    ):
      if not math.isfinite(value) or value < 0.0:
        raise ValueError("joint learner loss weights must be finite and non-negative")


class _IntrinsicMessagePassingV1(nn.Module):
  def __init__(self, hidden_dim: int, dropout: float) -> None:
    super().__init__()
    self.message = nn.Sequential(
        nn.Linear(2 * hidden_dim + LABEL_FREE_EDGE_INPUT_DIM, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.update = nn.Sequential(
        nn.Linear(2 * hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
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
      ones = torch.ones(
          (source.shape[0], 1), dtype=hidden.dtype, device=hidden.device
      )
      aggregate.index_add_(0, target, forward)
      aggregate.index_add_(0, source, reverse)
      degree.index_add_(0, target, ones)
      degree.index_add_(0, source, ones)
      aggregate = aggregate / degree.clamp_min(1.0)
    return self.norm(hidden + self.update(torch.cat((hidden, aggregate), dim=-1)))


@dataclass(frozen=True, slots=True)
class JointInterfaceOutputV1:
  joint_logits: Tensor
  face_salience_logits: Tensor
  selected_face_indices: Tensor
  selected_face_mask: Tensor
  candidate_program_indices: Tensor
  residual_translation: Tensor
  residual_rotation_vector: Tensor
  predicted_face_index_a: Tensor
  predicted_face_index_b: Tensor
  predicted_program_position: Tensor
  predicted_program_index: Tensor
  predicted_residual_translation: Tensor
  predicted_residual_rotation_vector: Tensor


class JointInterfaceProgramLearnerV1(nn.Module):
  """SE(3)-invariant face-pair and finite-program cross scorer."""

  def __init__(
      self, config: JointInterfaceProgramLearnerConfigV1, *, seed: int
  ) -> None:
    super().__init__()
    if type(config) is not JointInterfaceProgramLearnerConfigV1:
      raise TypeError("joint learner config type differs")
    if type(seed) is not int or seed < 0:
      raise ValueError("joint learner seed must be a non-negative integer")
    self.config = config
    self.seed = seed
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(seed)
      self.node_encoder = nn.Sequential(
          nn.Linear(LABEL_FREE_NODE_INPUT_DIM, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.layers = nn.ModuleList(
          _IntrinsicMessagePassingV1(config.hidden_dim, config.dropout)
          for _ in range(config.layers)
      )
      # Part ordering is model-visible and semantic (parent/child query side),
      # so each side has its own salience readout over a shared graph encoder.
      # This remains rigid-transform invariant and carries no selected-face bit.
      self.face_salience_heads = nn.ModuleList(
          (nn.Linear(config.hidden_dim, 1), nn.Linear(config.hidden_dim, 1))
      )
      self.face_pair_encoder = nn.Sequential(
          nn.Linear(4 * config.hidden_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )
      self.catalog_embedding = nn.Embedding(config.catalog_size, config.hidden_dim)
      self.joint_scorer = nn.Sequential(
          nn.Linear(4 * config.hidden_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Dropout(config.dropout),
          nn.Linear(config.hidden_dim, 1),
      )
      self.program_local_residual = nn.Sequential(
          nn.Linear(4 * config.hidden_dim, config.hidden_dim),
          nn.SiLU(),
          nn.Dropout(config.dropout),
          nn.Linear(config.hidden_dim, 6),
      )

  @property
  def parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters())

  def _encode_and_select(
      self, inputs: JointInterfaceInputsV1
  ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, _, max_nodes, _ = inputs.node_features.shape
    top_l = self.config.top_l
    selected_hidden = torch.zeros(
        (batch, 2, top_l, self.config.hidden_dim),
        dtype=inputs.node_features.dtype,
        device=inputs.node_features.device,
    )
    selected_indices = torch.full(
        (batch, 2, top_l), -1, dtype=torch.long, device=inputs.node_features.device
    )
    selected_mask = torch.zeros(
        (batch, 2, top_l), dtype=torch.bool, device=inputs.node_features.device
    )
    salience_logits = torch.full(
        (batch, 2, max_nodes),
        -torch.inf,
        dtype=inputs.node_features.dtype,
        device=inputs.node_features.device,
    )
    for batch_index in range(batch):
      for part_index in range(2):
        node_count = int(inputs.node_mask[batch_index, part_index].sum())
        hidden = self.node_encoder(
            inputs.node_features[batch_index, part_index, :node_count]
        )
        active_edges = inputs.edge_mask[batch_index, part_index]
        edge_index = inputs.edge_index[batch_index, part_index, active_edges]
        edge_features = inputs.edge_features[batch_index, part_index, active_edges]
        for layer in self.layers:
          hidden = layer(hidden, edge_index, edge_features)
        salience = self.face_salience_heads[part_index](hidden).squeeze(-1)
        salience_logits[batch_index, part_index, :node_count] = salience
        # Stable descending sort gives the frozen tie rule: lower graph-local
        # face index wins when learned salience values are exactly equal.
        order = torch.argsort(salience, descending=True, stable=True)
        count = min(top_l, node_count)
        chosen = order[:count]
        selected_hidden[batch_index, part_index, :count] = hidden[chosen]
        selected_indices[batch_index, part_index, :count] = chosen
        selected_mask[batch_index, part_index, :count] = True
    return selected_hidden, selected_indices, selected_mask, salience_logits

  def forward(self, inputs: JointInterfaceInputsV1) -> JointInterfaceOutputV1:
    _validate_inputs(inputs, catalog_size=self.config.catalog_size)
    selected, indices, face_mask, salience = self._encode_and_select(inputs)
    left = selected[:, 0, :, None, :]
    right = selected[:, 1, None, :, :]
    left = left.expand(-1, -1, self.config.top_l, -1)
    right = right.expand(-1, self.config.top_l, -1, -1)
    pair = self.face_pair_encoder(
        torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
    )
    candidates = self.catalog_embedding(inputs.candidate_program_indices)
    pair_expanded = pair[:, :, :, None, :].expand(
        -1, -1, -1, self.config.catalog_size, -1
    )
    candidates_expanded = candidates[:, None, None, :, :].expand(
        -1, self.config.top_l, self.config.top_l, -1, -1
    )
    cross = torch.cat(
        (
            pair_expanded,
            candidates_expanded,
            pair_expanded * candidates_expanded,
            torch.abs(pair_expanded - candidates_expanded),
        ),
        dim=-1,
    )
    logits = self.joint_scorer(cross).squeeze(-1)
    pair_mask = face_mask[:, 0, :, None] & face_mask[:, 1, None, :]
    logits = logits.masked_fill(~pair_mask[:, :, :, None], -torch.inf)
    residual = self.program_local_residual(cross)
    flat = logits.flatten(start_dim=1)
    best_flat = torch.argmax(flat, dim=1)
    program_position = best_flat.remainder(self.config.catalog_size)
    pair_flat = torch.div(
        best_flat, self.config.catalog_size, rounding_mode="floor"
    )
    rank_a = torch.div(pair_flat, self.config.top_l, rounding_mode="floor")
    rank_b = pair_flat.remainder(self.config.top_l)
    batch_index = torch.arange(flat.shape[0], device=flat.device)
    predicted_a = indices[batch_index, 0, rank_a]
    predicted_b = indices[batch_index, 1, rank_b]
    predicted_program = inputs.candidate_program_indices[
        batch_index, program_position
    ]
    predicted_residual = residual[
        batch_index, rank_a, rank_b, program_position
    ]
    return JointInterfaceOutputV1(
        joint_logits=logits,
        face_salience_logits=salience,
        selected_face_indices=indices,
        selected_face_mask=face_mask,
        candidate_program_indices=inputs.candidate_program_indices,
        residual_translation=residual[..., :3],
        residual_rotation_vector=residual[..., 3:],
        predicted_face_index_a=predicted_a,
        predicted_face_index_b=predicted_b,
        predicted_program_position=program_position,
        predicted_program_index=predicted_program,
        predicted_residual_translation=predicted_residual[..., :3],
        predicted_residual_rotation_vector=predicted_residual[..., 3:],
    )


@dataclass(frozen=True, slots=True)
class JointInterfaceLossV1:
  total: Tensor
  face_salience: Tensor
  joint_classification: Tensor
  residual_translation: Tensor
  residual_rotation: Tensor
  selected_target_rows: int


def joint_interface_program_loss_v1(
    output: JointInterfaceOutputV1,
    targets: JointInterfaceTargetsV1,
    *,
    config: JointInterfaceProgramLearnerConfigV1,
) -> JointInterfaceLossV1:
  """Apply supervision after forward; missing top-L targets train salience only."""

  if type(output) is not JointInterfaceOutputV1:
    raise TypeError("joint loss requires JointInterfaceOutputV1")
  if type(targets) is not JointInterfaceTargetsV1:
    raise TypeError("joint loss requires separate JointInterfaceTargetsV1")
  if targets.schema_version != TARGET_SCHEMA_VERSION:
    raise ValueError("joint target schema differs")
  batch = output.joint_logits.shape[0]
  one_dimensional = (
      targets.face_index_a,
      targets.face_index_b,
      targets.program_index,
      targets.residual_mask,
  )
  if any(value.shape != (batch,) for value in one_dimensional):
    raise ValueError("joint target scalar tensor shapes differ")
  if targets.residual_translation.shape != (batch, 3) or (
      targets.residual_rotation_vector.shape != (batch, 3)
  ):
    raise ValueError("joint target residual tensor shapes differ")
  face_a = F.cross_entropy(
      output.face_salience_logits[:, 0], targets.face_index_a
  )
  face_b = F.cross_entropy(
      output.face_salience_logits[:, 1], targets.face_index_b
  )
  face_salience = 0.5 * (face_a + face_b)
  selected_rows: list[int] = []
  flat_targets: list[Tensor] = []
  positions_a: list[Tensor] = []
  positions_b: list[Tensor] = []
  program_positions: list[Tensor] = []
  for row in range(batch):
    match_a = torch.nonzero(
        output.selected_face_indices[row, 0] == targets.face_index_a[row]
    ).flatten()
    match_b = torch.nonzero(
        output.selected_face_indices[row, 1] == targets.face_index_b[row]
    ).flatten()
    match_program = torch.nonzero(
        output.candidate_program_indices[row] == targets.program_index[row]
    ).flatten()
    if match_program.numel() != 1:
      raise ValueError("joint target program is outside the frozen catalog")
    if match_a.numel() == 1 and match_b.numel() == 1:
      pos_a, pos_b, pos_program = match_a[0], match_b[0], match_program[0]
      selected_rows.append(row)
      positions_a.append(pos_a)
      positions_b.append(pos_b)
      program_positions.append(pos_program)
      flat_targets.append(
          (pos_a * config.top_l + pos_b) * config.catalog_size + pos_program
      )
  if selected_rows:
    row_tensor = torch.tensor(
        selected_rows, dtype=torch.long, device=output.joint_logits.device
    )
    pos_a_tensor = torch.stack(positions_a)
    pos_b_tensor = torch.stack(positions_b)
    pos_program_tensor = torch.stack(program_positions)
    flat_target_tensor = torch.stack(flat_targets).long()
    selected_logits = output.joint_logits[row_tensor].flatten(start_dim=1)
    classification = F.cross_entropy(selected_logits, flat_target_tensor)
    predicted_translation = output.residual_translation[
        row_tensor, pos_a_tensor, pos_b_tensor, pos_program_tensor
    ]
    predicted_rotation = output.residual_rotation_vector[
        row_tensor, pos_a_tensor, pos_b_tensor, pos_program_tensor
    ]
    residual_mask = targets.residual_mask[row_tensor].to(
        output.joint_logits.dtype
    )
    denominator = residual_mask.sum().clamp_min(1.0)
    translation_rows = F.smooth_l1_loss(
        predicted_translation,
        targets.residual_translation[row_tensor],
        reduction="none",
    ).mean(dim=-1)
    rotation_rows = F.smooth_l1_loss(
        predicted_rotation,
        targets.residual_rotation_vector[row_tensor],
        reduction="none",
    ).mean(dim=-1)
    translation = (translation_rows * residual_mask).sum() / denominator
    rotation = (rotation_rows * residual_mask).sum() / denominator
  else:
    differentiable_zero = output.joint_logits[torch.isfinite(output.joint_logits)].sum() * 0.0
    classification = differentiable_zero
    translation = differentiable_zero
    rotation = differentiable_zero
  total = (
      config.face_salience_loss_weight * face_salience
      + classification
      + config.translation_loss_weight * translation
      + config.rotation_loss_weight * rotation
  )
  return JointInterfaceLossV1(
      total=total,
      face_salience=face_salience,
      joint_classification=classification,
      residual_translation=translation,
      residual_rotation=rotation,
      selected_target_rows=len(selected_rows),
  )


def _rotation_vector_to_matrix(vector: Tensor) -> Tensor:
  theta2 = torch.sum(vector * vector)
  theta = torch.sqrt(torch.clamp(theta2, min=1e-16))
  a = torch.where(theta2 < 1e-8, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
  b = torch.where(
      theta2 < 1e-8,
      0.5 - theta2 / 24.0,
      (1.0 - torch.cos(theta)) / theta2,
  )
  x, y, z = vector
  zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
  skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
  return torch.eye(3, dtype=vector.dtype, device=vector.device) + a * skew + b * (skew @ skew)


def _state_dict_sha256(model: nn.Module) -> str:
  digest = hashlib.sha256()
  for name, tensor in sorted(model.state_dict().items()):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes(order="C"))
  return digest.hexdigest()


@dataclass(frozen=True)
class JointInterfaceProgramHypothesisV1(ProgramHypothesis):
  predicted_graph_local_face_index_a: int = -1
  predicted_graph_local_face_index_b: int = -1
  predicted_program_index: int = -1

  def trace_dict(self) -> dict[str, Any]:
    trace = super().trace_dict()
    trace.update(
        {
            "predicted_graph_local_face_index_a": int(
                self.predicted_graph_local_face_index_a
            ),
            "predicted_graph_local_face_index_b": int(
                self.predicted_graph_local_face_index_b
            ),
            "predicted_program_index": int(self.predicted_program_index),
            "input_schema_version": INPUT_SCHEMA_VERSION,
            "learner_schema_version": SCHEMA_VERSION,
        }
    )
    return trace


class JointInterfaceProgramProposerV1:
  """Fixed-budget ProgramProposer adapter whose query is the two graph input."""

  mode = "joint_interface_program_learner_v1"

  def __init__(
      self,
      model: JointInterfaceProgramLearnerV1,
      *,
      device: torch.device | str = "cpu",
  ) -> None:
    if type(model) is not JointInterfaceProgramLearnerV1:
      raise TypeError("joint proposer requires JointInterfaceProgramLearnerV1")
    self._model = model.to(device).eval()
    self._device = torch.device(device)
    self.model_sha256 = _state_dict_sha256(model)

  def propose(
      self,
      query: JointInterfaceInputsV1,
      candidates: Sequence[MateProgramRetrieval],
      *,
      budget: ProposalBudget,
  ) -> list[JointInterfaceProgramHypothesisV1]:
    """Rank joint face-pair/program hypotheses without a MateProgram query."""

    if type(query) is not JointInterfaceInputsV1:
      raise TypeError("joint proposer query must be label-free two-part graphs")
    _validate_inputs(query, catalog_size=self._model.config.catalog_size)
    if query.node_features.shape[0] != 1:
      raise ValueError("joint proposer accepts exactly one two-part query")
    if budget.max_occ_calls != 0:
      raise ValueError("joint proposer requires explicit max_occ_calls=0")
    if budget.wall_time_seconds is None or budget.wall_time_seconds < 0.01:
      raise ValueError("joint proposer requires an explicit wall-time budget")
    if budget.candidate_pool_size != self._model.config.catalog_size:
      raise ValueError("joint proposer pool budget must equal the 19-program catalog")
    if len(candidates) != budget.candidate_pool_size:
      raise ValueError("joint proposer candidate pool must exactly match its budget")
    if budget.top_k > budget.candidate_pool_size:
      raise ValueError("joint proposer top_k exceeds its fixed candidate pool")
    started = time.monotonic()
    with torch.inference_mode():
      output = self._model(query.to(self._device))
    if time.monotonic() - started > budget.wall_time_seconds:
      raise TimeoutError("joint proposal exceeded its cooperative wall-time budget")
    candidate_by_index: dict[int, tuple[int, MateProgramRetrieval]] = {}
    expected_indices = query.candidate_program_indices[0].detach().cpu().tolist()
    for retrieval_rank, (expected_index, row) in enumerate(
        zip(expected_indices, candidates, strict=True)
    ):
      if type(row) is not MateProgramRetrieval:
        raise TypeError("joint proposer candidates must be MateProgramRetrieval rows")
      declared = row.program.metadata.get("brep_catalog_program_index", expected_index)
      if type(declared) is not int or declared != expected_index:
        raise ValueError("joint proposer candidate order differs from catalog indices")
      candidate_by_index[expected_index] = (retrieval_rank, row)
    scored: list[tuple[float, int, int, int, int, MateProgramRetrieval]] = []
    indices = output.selected_face_indices[0].detach().cpu()
    logits = output.joint_logits[0].detach().cpu()
    for rank_a in range(self._model.config.top_l):
      face_a = int(indices[0, rank_a])
      if face_a < 0:
        continue
      for rank_b in range(self._model.config.top_l):
        face_b = int(indices[1, rank_b])
        if face_b < 0:
          continue
        for program_position, program_index in enumerate(expected_indices):
          score = float(logits[rank_a, rank_b, program_position])
          if not math.isfinite(score):
            raise ValueError("joint proposer produced a non-finite valid score")
          retrieval_rank, row = candidate_by_index[program_index]
          scored.append(
              (score, face_a, face_b, program_index, retrieval_rank, row)
          )
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
            item[5].program.program_id,
        )
    )
    if len(scored) < budget.top_k:
      raise ValueError("joint proposer cannot satisfy top_k")
    hypotheses: list[JointInterfaceProgramHypothesisV1] = []
    for score, face_a, face_b, program_index, retrieval_rank, row in scored[
        : budget.top_k
    ]:
      program_position = expected_indices.index(program_index)
      rank_a = int(torch.nonzero(indices[0] == face_a).flatten()[0])
      rank_b = int(torch.nonzero(indices[1] == face_b).flatten()[0])
      translation = output.residual_translation[
          0, rank_a, rank_b, program_position
      ].detach().cpu()
      rotation = _rotation_vector_to_matrix(
          output.residual_rotation_vector[
              0, rank_a, rank_b, program_position
          ].detach().cpu()
      )
      program = row.program
      hypotheses.append(
          JointInterfaceProgramHypothesisV1(
              program=program,
              score=score,
              program_id=str(program.program_id),
              program_class=mate_program_class_from_program(program),
              query_parent_interface_id=f"part_a_graph_face_{face_a}",
              query_child_interface_id=f"part_b_graph_face_{face_b}",
              residual_rotation=tuple(
                  tuple(float(value) for value in matrix_row)
                  for matrix_row in rotation.tolist()
              ),
              residual_translation=tuple(float(value) for value in translation),
              score_components={"joint_face_pair_program_logit": score},
              model_sha256=self.model_sha256,
              retrieval_rank=retrieval_rank,
              proposer_mode=self.mode,
              proposal_budget={
                  **budget.trace_dict(),
                  "wall_time_enforcement": "cooperative_post_forward.v1",
                  "hard_preemption": False,
              },
              proposer_protocol=PROGRAM_PROPOSER_PROTOCOL,
              predicted_graph_local_face_index_a=face_a,
              predicted_graph_local_face_index_b=face_b,
              predicted_program_index=program_index,
          )
      )
      if time.monotonic() - started > budget.wall_time_seconds:
        raise TimeoutError("joint proposal exceeded its cooperative wall-time budget")
    return hypotheses
