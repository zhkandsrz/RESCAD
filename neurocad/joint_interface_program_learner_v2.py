"""Versioned joint interface/program learner with dense-supervision training.

V2 keeps inference sparse (stable top-8 faces per part) but never routes
training supervision through that hard selection.  ``forward`` exposes every
face embedding.  The loss selects the two supervised embeddings and applies
the same face-pair/program heads directly, so salience mistakes cannot starve
the joint scorer of gradients.

The inference boundary emits immutable graph-local execution records.  It
does not retain source ``MateProgram`` objects or their asserted interfaces.
Conversion of graph-local residuals to a world-frame delta is deliberately a
separate, non-learned frame-adapter operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
from multiprocessing.connection import Connection
import re
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .joint_interface_program_learner_v1 import (
    FORMAL_CATALOG_SIZE,
    INPUT_SCHEMA_VERSION as GRAPH_TENSOR_SCHEMA_VERSION,
    LABEL_FREE_EDGE_INPUT_DIM,
    LABEL_FREE_NODE_INPUT_DIM,
    JointInterfaceInputsV1,
    LabelFreeBRepPairV1,
    collate_label_free_brep_pairs_v1,
)
from .program_proposer import ProgramProposer, ProposalBudget


SCHEMA_VERSION = "joint_interface_program_learner.v2"
INPUT_SCHEMA_VERSION = "graph_program_inputs.v2"
OUTPUT_SCHEMA_VERSION = "joint_interface_program_output.v2"
TARGET_SCHEMA_VERSION = "joint_interface_program_targets.v2"
CATALOG_SCHEMA_VERSION = "finite_program_catalog_roster.v2"
EXECUTABLE_SCHEMA_VERSION = "executable_predicted_interface_program.v2"
FRAME_SCHEMA_VERSION = "deterministic_predicted_face_frame.v2"
WORLD_RESIDUAL_SCHEMA_VERSION = "world_residual_transform.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


PAIR_FRAME_POLICY = {
    "schema_version": "ordered_predicted_face_pair_frame.v2",
    "origin": "midpoint_of_deterministic_face_frame_origins",
    "z_seed": "normalized_a_normal_minus_b_normal_else_a_normal",
    "x_seed_order": [
        "a_x_projected_to_z_perpendicular",
        "b_x_projected_to_z_perpendicular",
        "a_y_projected_to_z_perpendicular",
    ],
    "handedness": "y_equals_z_cross_x_then_x_equals_y_cross_z",
    "residual_coordinates": "conjugated_pair_local_se3_delta",
}
PAIR_FRAME_POLICY_SHA256 = _canonical_sha256(PAIR_FRAME_POLICY)


@dataclass(frozen=True, slots=True)
class FiniteProgramDescriptorV2:
  relation_hint: str
  contact_type: str
  surface_type_a: str
  surface_type_b: str

  def __post_init__(self) -> None:
    for value in asdict(self).values():
      if not isinstance(value, str) or not value.strip() or value != value.strip().lower():
        raise ValueError("finite-program descriptor fields must be normalized strings")

  def payload(self) -> dict[str, str]:
    return asdict(self)

  @property
  def sha256(self) -> str:
    return _canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class FiniteProgramCatalogEntryV2:
  program_index: int
  program_id: str
  descriptor: FiniteProgramDescriptorV2
  descriptor_sha256: str

  def __post_init__(self) -> None:
    if type(self.program_index) is not int or not 0 <= self.program_index < FORMAL_CATALOG_SIZE:
      raise ValueError("catalog program index is outside [0, 19)")
    if type(self.descriptor) is not FiniteProgramDescriptorV2:
      raise TypeError("catalog descriptor type differs")
    if self.descriptor_sha256 != self.descriptor.sha256:
      raise ValueError("catalog descriptor hash differs from exact descriptor")
    if self.program_id != f"catalog_{self.descriptor_sha256[:16]}":
      raise ValueError("catalog program_id differs from its exact descriptor")

  def payload(self) -> dict[str, Any]:
    return {
        "program_index": self.program_index,
        "program_id": self.program_id,
        "descriptor": self.descriptor.payload(),
        "descriptor_sha256": self.descriptor_sha256,
    }


def finite_program_catalog_sha256_v2(
    entries: Sequence[FiniteProgramCatalogEntryV2],
) -> str:
  payload = {
      "schema_version": CATALOG_SCHEMA_VERSION,
      "entry_count": len(entries),
      "entries": [entry.payload() for entry in entries],
  }
  return _canonical_sha256(payload)


@dataclass(frozen=True, slots=True)
class FiniteProgramCatalogRosterV2:
  entries: tuple[FiniteProgramCatalogEntryV2, ...]
  catalog_sha256: str
  schema_version: str = CATALOG_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CATALOG_SCHEMA_VERSION:
      raise ValueError("finite-program catalog schema differs")
    if len(self.entries) != FORMAL_CATALOG_SIZE:
      raise ValueError("finite-program catalog must contain exactly 19 entries")
    if any(type(entry) is not FiniteProgramCatalogEntryV2 for entry in self.entries):
      raise TypeError("finite-program catalog contains another entry type")
    if tuple(entry.program_index for entry in self.entries) != tuple(
        range(FORMAL_CATALOG_SIZE)
    ):
      raise ValueError("finite-program catalog roster order differs")
    program_ids = tuple(entry.program_id for entry in self.entries)
    descriptor_hashes = tuple(entry.descriptor_sha256 for entry in self.entries)
    if len(set(program_ids)) != FORMAL_CATALOG_SIZE:
      raise ValueError("finite-program catalog repeats a program_id")
    if len(set(descriptor_hashes)) != FORMAL_CATALOG_SIZE:
      raise ValueError("finite-program catalog repeats a descriptor hash")
    if _SHA256.fullmatch(self.catalog_sha256) is None:
      raise ValueError("finite-program catalog requires an exact SHA-256")
    if self.catalog_sha256 != finite_program_catalog_sha256_v2(self.entries):
      raise ValueError("finite-program catalog hash differs from exact roster")

  def entry(self, program_index: int) -> FiniteProgramCatalogEntryV2:
    if type(program_index) is not int or not 0 <= program_index < len(self.entries):
      raise ValueError("program index is outside the exact roster")
    entry = self.entries[program_index]
    if entry.program_index != program_index:
      raise ValueError("program roster position differs")
    return entry


@dataclass(frozen=True, slots=True)
class GraphProgramInputsV2:
  graph_tensors: JointInterfaceInputsV1
  catalog_sha256: str
  schema_version: str = INPUT_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if type(self.graph_tensors) is not JointInterfaceInputsV1:
      raise TypeError("V2 graph-program inputs require V1 unannotated graph tensors")
    if self.graph_tensors.schema_version != GRAPH_TENSOR_SCHEMA_VERSION:
      raise ValueError("V2 graph tensor schema differs")
    if _SHA256.fullmatch(self.catalog_sha256) is None:
      raise ValueError("V2 graph-program inputs require a catalog SHA-256")

  def to(self, device: torch.device | str) -> "GraphProgramInputsV2":
    return GraphProgramInputsV2(
        graph_tensors=self.graph_tensors.to(device),
        catalog_sha256=self.catalog_sha256,
        schema_version=self.schema_version,
    )


def collate_graph_program_pairs_v2(
    pairs: Sequence[LabelFreeBRepPairV1],
    *,
    catalog: FiniteProgramCatalogRosterV2,
) -> GraphProgramInputsV2:
  if type(catalog) is not FiniteProgramCatalogRosterV2:
    raise TypeError("V2 collator requires the exact finite-program roster")
  return GraphProgramInputsV2(
      graph_tensors=collate_label_free_brep_pairs_v1(
          pairs,
          candidate_program_indices=tuple(
              entry.program_index for entry in catalog.entries
          ),
      ),
      catalog_sha256=catalog.catalog_sha256,
  )


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramLearnerConfigV2:
  hidden_dim: int = 96
  layers: int = 4
  top_l: int = 8
  dropout: float = 0.0
  face_salience_loss_weight: float = 0.5
  translation_loss_weight: float = 0.25
  rotation_loss_weight: float = 0.25
  catalog_size: int = FORMAL_CATALOG_SIZE
  pair_frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256

  def __post_init__(self) -> None:
    if type(self.hidden_dim) is not int or self.hidden_dim < 8:
      raise ValueError("V2 hidden_dim must be at least eight")
    if type(self.layers) is not int or self.layers < 1:
      raise ValueError("V2 layers must be positive")
    if type(self.top_l) is not int or not 1 <= self.top_l <= 64:
      raise ValueError("V2 top_l must be in [1, 64]")
    if self.catalog_size != FORMAL_CATALOG_SIZE:
      raise ValueError("V2 catalog is frozen to 19 programs")
    if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
      raise ValueError("V2 dropout must be finite in [0, 1)")
    if (
        not math.isfinite(self.face_salience_loss_weight)
        or self.face_salience_loss_weight <= 0.0
    ):
      raise ValueError("V2 face salience loss weight must be strictly positive")
    for value in (self.translation_loss_weight, self.rotation_loss_weight):
      if not math.isfinite(value) or value < 0.0:
        raise ValueError("V2 residual loss weights must be finite and non-negative")
    if self.pair_frame_policy_sha256 != PAIR_FRAME_POLICY_SHA256:
      raise ValueError("V2 pair-frame policy differs from the frozen policy")

  @property
  def sha256(self) -> str:
    return _canonical_sha256(asdict(self))


class _MessagePassingV2(nn.Module):
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
class JointInterfaceProgramOutputV2:
  all_face_embeddings: Tensor
  all_face_mask: Tensor
  face_salience_logits: Tensor
  selected_face_indices: Tensor
  selected_face_mask: Tensor
  inference_joint_logits: Tensor
  inference_residual_translation_local: Tensor
  inference_residual_rotation_vector_local: Tensor
  candidate_program_indices: Tensor
  predicted_face_index_a: Tensor
  predicted_face_index_b: Tensor
  predicted_program_index: Tensor
  predicted_residual_translation_local: Tensor
  predicted_residual_rotation_vector_local: Tensor
  model_config_sha256: str
  catalog_sha256: str
  top_l: int
  catalog_size: int
  pair_frame_policy_sha256: str
  schema_version: str = OUTPUT_SCHEMA_VERSION


class JointInterfaceProgramLearnerV2(nn.Module):
  """Full-face encoder with sparse inference and dense gold-pair training."""

  def __init__(
      self, config: JointInterfaceProgramLearnerConfigV2, *, seed: int
  ) -> None:
    super().__init__()
    if type(config) is not JointInterfaceProgramLearnerConfigV2:
      raise TypeError("V2 learner config type differs")
    if type(seed) is not int or seed < 0:
      raise ValueError("V2 learner seed must be a non-negative integer")
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
          _MessagePassingV2(config.hidden_dim, config.dropout)
          for _ in range(config.layers)
      )
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

  def _validate_inputs(self, inputs: GraphProgramInputsV2) -> None:
    if type(inputs) is not GraphProgramInputsV2:
      raise TypeError("V2 forward accepts GraphProgramInputsV2 only")
    tensors = inputs.graph_tensors
    if tensors.node_features.ndim != 4 or tensors.node_features.shape[1] != 2:
      raise ValueError("V2 forward requires exactly two B-Rep graphs")
    batch, _, max_nodes, node_dim = tensors.node_features.shape
    if batch < 1 or max_nodes < 1 or node_dim != LABEL_FREE_NODE_INPUT_DIM:
      raise ValueError("V2 node tensor shape differs")
    if tensors.node_mask.shape != (batch, 2, max_nodes):
      raise ValueError("V2 node mask shape differs")
    if not bool(tensors.node_mask.any(dim=-1).all()):
      raise ValueError("V2 requires at least one face per part")
    if tensors.candidate_program_indices.shape != (
        batch,
        self.config.catalog_size,
    ):
      raise ValueError("V2 requires the complete 19-program catalog")
    for row in tensors.candidate_program_indices.detach().cpu().tolist():
      if tuple(int(value) for value in row) != tuple(range(self.config.catalog_size)):
        raise ValueError("V2 catalog indices differ from the exact roster order")

  def _encode_all_faces(
      self, inputs: GraphProgramInputsV2
  ) -> tuple[Tensor, Tensor]:
    tensors = inputs.graph_tensors
    batch, _, max_nodes, _ = tensors.node_features.shape
    embeddings = torch.zeros(
        (batch, 2, max_nodes, self.config.hidden_dim),
        dtype=tensors.node_features.dtype,
        device=tensors.node_features.device,
    )
    salience = torch.full(
        (batch, 2, max_nodes),
        -torch.inf,
        dtype=tensors.node_features.dtype,
        device=tensors.node_features.device,
    )
    for batch_index in range(batch):
      for part_index in range(2):
        node_count = int(tensors.node_mask[batch_index, part_index].sum())
        hidden = self.node_encoder(
            tensors.node_features[batch_index, part_index, :node_count]
        )
        active_edges = tensors.edge_mask[batch_index, part_index]
        edge_index = tensors.edge_index[batch_index, part_index, active_edges]
        edge_features = tensors.edge_features[
            batch_index, part_index, active_edges
        ]
        for layer in self.layers:
          hidden = layer(hidden, edge_index, edge_features)
        embeddings[batch_index, part_index, :node_count] = hidden
        salience[batch_index, part_index, :node_count] = self.face_salience_heads[
            part_index
        ](hidden).squeeze(-1)
    return embeddings, salience

  def _pair_program_scores(
      self,
      face_a_embeddings: Tensor,
      face_b_embeddings: Tensor,
      candidate_program_indices: Tensor,
  ) -> tuple[Tensor, Tensor]:
    """Score aligned face pairs against a complete catalog.

    The leading dimensions of the two face tensors must match.  A catalog
    axis is appended by this method; callers may supply B gold pairs or BxLxL
    inference pairs without changing the learned heads.
    """

    if face_a_embeddings.shape != face_b_embeddings.shape:
      raise ValueError("V2 aligned face embedding shapes differ")
    if face_a_embeddings.shape[-1] != self.config.hidden_dim:
      raise ValueError("V2 face embedding dimension differs")
    if candidate_program_indices.shape[:-1] != face_a_embeddings.shape[:-1]:
      raise ValueError("V2 candidate catalog leading dimensions differ")
    if candidate_program_indices.shape[-1] != self.config.catalog_size:
      raise ValueError("V2 candidate catalog size differs")
    pair = self.face_pair_encoder(
        torch.cat(
            (
                face_a_embeddings,
                face_b_embeddings,
                torch.abs(face_a_embeddings - face_b_embeddings),
                face_a_embeddings * face_b_embeddings,
            ),
            dim=-1,
        )
    )
    candidates = self.catalog_embedding(candidate_program_indices)
    pair_expanded = pair.unsqueeze(-2).expand_as(candidates)
    cross = torch.cat(
        (
            pair_expanded,
            candidates,
            pair_expanded * candidates,
            torch.abs(pair_expanded - candidates),
        ),
        dim=-1,
    )
    return self.joint_scorer(cross).squeeze(-1), self.program_local_residual(cross)

  def forward(self, inputs: GraphProgramInputsV2) -> JointInterfaceProgramOutputV2:
    self._validate_inputs(inputs)
    tensors = inputs.graph_tensors
    embeddings, salience = self._encode_all_faces(inputs)
    batch = embeddings.shape[0]
    top_l = self.config.top_l
    selected = torch.zeros(
        (batch, 2, top_l, self.config.hidden_dim),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    indices = torch.full(
        (batch, 2, top_l), -1, dtype=torch.long, device=embeddings.device
    )
    selected_mask = torch.zeros(
        (batch, 2, top_l), dtype=torch.bool, device=embeddings.device
    )
    for batch_index in range(batch):
      for part_index in range(2):
        node_count = int(tensors.node_mask[batch_index, part_index].sum())
        order = torch.argsort(
            salience[batch_index, part_index, :node_count],
            descending=True,
            stable=True,
        )
        count = min(top_l, node_count)
        chosen = order[:count]
        selected[batch_index, part_index, :count] = embeddings[
            batch_index, part_index, chosen
        ]
        indices[batch_index, part_index, :count] = chosen
        selected_mask[batch_index, part_index, :count] = True
    left = selected[:, 0, :, None, :].expand(-1, -1, top_l, -1)
    right = selected[:, 1, None, :, :].expand(-1, top_l, -1, -1)
    candidate_indices = tensors.candidate_program_indices[:, None, None, :].expand(
        -1, top_l, top_l, -1
    )
    logits, residual = self._pair_program_scores(
        left, right, candidate_indices
    )
    pair_mask = selected_mask[:, 0, :, None] & selected_mask[:, 1, None, :]
    logits = logits.masked_fill(~pair_mask[:, :, :, None], -torch.inf)
    flat = logits.flatten(start_dim=1)
    best = torch.argmax(flat, dim=1)
    program_position = best.remainder(self.config.catalog_size)
    pair_flat = torch.div(best, self.config.catalog_size, rounding_mode="floor")
    rank_a = torch.div(pair_flat, top_l, rounding_mode="floor")
    rank_b = pair_flat.remainder(top_l)
    batch_index = torch.arange(batch, device=embeddings.device)
    predicted_residual = residual[
        batch_index, rank_a, rank_b, program_position
    ]
    return JointInterfaceProgramOutputV2(
        all_face_embeddings=embeddings,
        all_face_mask=tensors.node_mask,
        face_salience_logits=salience,
        selected_face_indices=indices,
        selected_face_mask=selected_mask,
        inference_joint_logits=logits,
        inference_residual_translation_local=residual[..., :3],
        inference_residual_rotation_vector_local=residual[..., 3:],
        candidate_program_indices=tensors.candidate_program_indices,
        predicted_face_index_a=indices[batch_index, 0, rank_a],
        predicted_face_index_b=indices[batch_index, 1, rank_b],
        predicted_program_index=tensors.candidate_program_indices[
            batch_index, program_position
        ],
        predicted_residual_translation_local=predicted_residual[..., :3],
        predicted_residual_rotation_vector_local=predicted_residual[..., 3:],
        model_config_sha256=self.config.sha256,
        catalog_sha256=inputs.catalog_sha256,
        top_l=self.config.top_l,
        catalog_size=self.config.catalog_size,
        pair_frame_policy_sha256=self.config.pair_frame_policy_sha256,
    )


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramTargetsV2:
  face_index_a: Tensor
  face_index_b: Tensor
  program_index: Tensor
  residual_translation_local: Tensor
  residual_rotation_vector_local: Tensor
  residual_mask: Tensor
  catalog_sha256: str
  pair_frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256
  schema_version: str = TARGET_SCHEMA_VERSION

  def to(self, device: torch.device | str) -> "JointInterfaceProgramTargetsV2":
    return JointInterfaceProgramTargetsV2(
        face_index_a=self.face_index_a.to(device),
        face_index_b=self.face_index_b.to(device),
        program_index=self.program_index.to(device),
        residual_translation_local=self.residual_translation_local.to(device),
        residual_rotation_vector_local=self.residual_rotation_vector_local.to(device),
        residual_mask=self.residual_mask.to(device),
        catalog_sha256=self.catalog_sha256,
        pair_frame_policy_sha256=self.pair_frame_policy_sha256,
        schema_version=self.schema_version,
    )


@dataclass(frozen=True, slots=True)
class JointInterfaceProgramLossV2:
  total: Tensor
  face_salience: Tensor
  gold_pair_program_classification: Tensor
  residual_translation_local: Tensor
  residual_rotation_local: Tensor


def joint_interface_program_loss_v2(
    model: JointInterfaceProgramLearnerV2,
    output: JointInterfaceProgramOutputV2,
    targets: JointInterfaceProgramTargetsV2,
) -> JointInterfaceProgramLossV2:
  """Train on the gold pair directly, irrespective of inference top-L."""

  if type(model) is not JointInterfaceProgramLearnerV2:
    raise TypeError("V2 loss requires JointInterfaceProgramLearnerV2")
  if type(output) is not JointInterfaceProgramOutputV2:
    raise TypeError("V2 loss requires JointInterfaceProgramOutputV2")
  if type(targets) is not JointInterfaceProgramTargetsV2:
    raise TypeError("V2 loss requires separate JointInterfaceProgramTargetsV2")
  config = model.config
  if (
      output.schema_version != OUTPUT_SCHEMA_VERSION
      or targets.schema_version != TARGET_SCHEMA_VERSION
      or output.model_config_sha256 != config.sha256
      or output.top_l != config.top_l
      or output.catalog_size != config.catalog_size
  ):
    raise ValueError("V2 loss output/config binding differs")
  if (
      output.catalog_sha256 != targets.catalog_sha256
      or output.pair_frame_policy_sha256 != targets.pair_frame_policy_sha256
      or targets.pair_frame_policy_sha256 != config.pair_frame_policy_sha256
  ):
    raise ValueError("V2 loss catalog or pair-frame binding differs")
  batch = output.all_face_embeddings.shape[0]
  for value in (
      targets.face_index_a,
      targets.face_index_b,
      targets.program_index,
      targets.residual_mask,
  ):
    if value.shape != (batch,):
      raise ValueError("V2 scalar target shape differs")
  if targets.residual_translation_local.shape != (batch, 3) or (
      targets.residual_rotation_vector_local.shape != (batch, 3)
  ):
    raise ValueError("V2 local residual target shape differs")
  batch_index = torch.arange(batch, device=output.all_face_embeddings.device)
  if not bool(
      output.all_face_mask[batch_index, 0, targets.face_index_a].all()
  ) or not bool(output.all_face_mask[batch_index, 1, targets.face_index_b].all()):
    raise ValueError("V2 gold face index is outside its full part graph")
  face_salience = 0.5 * (
      F.cross_entropy(output.face_salience_logits[:, 0], targets.face_index_a)
      + F.cross_entropy(output.face_salience_logits[:, 1], targets.face_index_b)
  )
  gold_a = output.all_face_embeddings[batch_index, 0, targets.face_index_a]
  gold_b = output.all_face_embeddings[batch_index, 1, targets.face_index_b]
  gold_logits, gold_residual = model._pair_program_scores(
      gold_a,
      gold_b,
      output.candidate_program_indices,
  )
  matches = output.candidate_program_indices == targets.program_index[:, None]
  if not bool((matches.sum(dim=1) == 1).all()):
    raise ValueError("V2 gold program differs from the exact catalog")
  target_positions = torch.argmax(matches.to(torch.long), dim=1)
  classification = F.cross_entropy(gold_logits, target_positions)
  predicted_residual = gold_residual[batch_index, target_positions]
  residual_mask = targets.residual_mask.to(gold_logits.dtype)
  denominator = residual_mask.sum().clamp_min(1.0)
  translation_rows = F.smooth_l1_loss(
      predicted_residual[..., :3],
      targets.residual_translation_local,
      reduction="none",
  ).mean(dim=-1)
  rotation_rows = F.smooth_l1_loss(
      predicted_residual[..., 3:],
      targets.residual_rotation_vector_local,
      reduction="none",
  ).mean(dim=-1)
  translation = (translation_rows * residual_mask).sum() / denominator
  rotation = (rotation_rows * residual_mask).sum() / denominator
  total = (
      config.face_salience_loss_weight * face_salience
      + classification
      + config.translation_loss_weight * translation
      + config.rotation_loss_weight * rotation
  )
  return JointInterfaceProgramLossV2(
      total=total,
      face_salience=face_salience,
      gold_pair_program_classification=classification,
      residual_translation_local=translation,
      residual_rotation_local=rotation,
  )


def _dot3(first: Sequence[float], second: Sequence[float]) -> float:
  return sum(float(first[index]) * float(second[index]) for index in range(3))


def _norm3(value: Sequence[float]) -> float:
  return math.sqrt(_dot3(value, value))


def _cross3(first: Sequence[float], second: Sequence[float]) -> tuple[float, float, float]:
  return (
      float(first[1]) * float(second[2]) - float(first[2]) * float(second[1]),
      float(first[2]) * float(second[0]) - float(first[0]) * float(second[2]),
      float(first[0]) * float(second[1]) - float(first[1]) * float(second[0]),
  )


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
  return tuple(tuple(float(matrix[row][column]) for row in range(3)) for column in range(3))


def _matmul3(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
  return tuple(
      tuple(
          sum(float(first[row][inner]) * float(second[inner][column]) for inner in range(3))
          for column in range(3)
      )
      for row in range(3)
  )


def _matvec3(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
  return tuple(
      sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
      for row in range(3)
  )


def _rotation_vector_to_matrix(
    vector: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
  value = tuple(float(item) for item in vector)
  angle = _norm3(value)
  if angle < 1e-12:
    return (
        (1.0, -value[2], value[1]),
        (value[2], 1.0, -value[0]),
        (-value[1], value[0], 1.0),
    )
  axis = tuple(item / angle for item in value)
  x, y, z = axis
  cosine = math.cos(angle)
  sine = math.sin(angle)
  one_minus_cosine = 1.0 - cosine
  return (
      (
          cosine + x * x * one_minus_cosine,
          x * y * one_minus_cosine - z * sine,
          x * z * one_minus_cosine + y * sine,
      ),
      (
          y * x * one_minus_cosine + z * sine,
          cosine + y * y * one_minus_cosine,
          y * z * one_minus_cosine - x * sine,
      ),
      (
          z * x * one_minus_cosine - y * sine,
          z * y * one_minus_cosine + x * sine,
          cosine + z * z * one_minus_cosine,
      ),
  )


@dataclass(frozen=True, slots=True)
class GraphProgramBudgetTraceV2:
  top_k: int
  candidate_pool_size: int
  max_occ_calls: int
  wall_time_seconds: float
  face_pair_pool: int
  program_pool: int
  expanded_joint_hypotheses: int
  preemptive_executor: bool
  formal_eligible: bool

  def payload(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutablePredictedInterfaceProgramV2:
  predicted_graph_local_face_index_a: int
  predicted_graph_local_face_index_b: int
  program_index: int
  program_id: str
  descriptor: FiniteProgramDescriptorV2
  descriptor_sha256: str
  catalog_sha256: str
  residual_translation_pair_local: tuple[float, float, float]
  residual_rotation_vector_pair_local: tuple[float, float, float]
  pair_frame_policy_sha256: str
  model_sha256: str
  score: float
  budget: GraphProgramBudgetTraceV2
  schema_version: str = EXECUTABLE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != EXECUTABLE_SCHEMA_VERSION:
      raise ValueError("V2 executable schema differs")
    if self.predicted_graph_local_face_index_a < 0 or self.predicted_graph_local_face_index_b < 0:
      raise ValueError("V2 executable face indices must be graph-local non-negative")
    if type(self.descriptor) is not FiniteProgramDescriptorV2:
      raise TypeError("V2 executable descriptor type differs")
    if self.descriptor_sha256 != self.descriptor.sha256:
      raise ValueError("V2 executable descriptor binding differs")
    if self.program_id != f"catalog_{self.descriptor_sha256[:16]}":
      raise ValueError("V2 executable program_id differs from descriptor binding")
    for value in (self.catalog_sha256, self.pair_frame_policy_sha256, self.model_sha256):
      if _SHA256.fullmatch(value) is None:
        raise ValueError("V2 executable hash binding is malformed")
    if self.pair_frame_policy_sha256 != PAIR_FRAME_POLICY_SHA256:
      raise ValueError("V2 executable pair-frame policy differs")
    residual = self.residual_translation_pair_local + self.residual_rotation_vector_pair_local
    if len(residual) != 6 or not all(math.isfinite(value) for value in residual):
      raise ValueError("V2 executable local residual is malformed")
    if not math.isfinite(self.score):
      raise ValueError("V2 executable score must be finite")

  def trace_dict(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version,
        "predicted_graph_local_face_index_a": self.predicted_graph_local_face_index_a,
        "predicted_graph_local_face_index_b": self.predicted_graph_local_face_index_b,
        "program_index": self.program_index,
        "program_id": self.program_id,
        "descriptor": self.descriptor.payload(),
        "descriptor_sha256": self.descriptor_sha256,
        "catalog_sha256": self.catalog_sha256,
        "residual_translation_pair_local": list(self.residual_translation_pair_local),
        "residual_rotation_vector_pair_local": list(
            self.residual_rotation_vector_pair_local
        ),
        "pair_frame_policy_sha256": self.pair_frame_policy_sha256,
        "model_sha256": self.model_sha256,
        "score": self.score,
        "budget": self.budget.payload(),
    }


@dataclass(frozen=True, slots=True)
class DeterministicPredictedFaceFrameV2:
  part_slot: str
  graph_local_face_index: int
  origin_world: tuple[float, float, float]
  rotation_local_to_world: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
  ]
  frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256
  schema_version: str = FRAME_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != FRAME_SCHEMA_VERSION or self.part_slot not in {"a", "b"}:
      raise ValueError("V2 deterministic face frame identity differs")
    if self.graph_local_face_index < 0:
      raise ValueError("V2 deterministic face frame index must be non-negative")
    if self.frame_policy_sha256 != PAIR_FRAME_POLICY_SHA256:
      raise ValueError("V2 deterministic face frame policy differs")
    origin = np.asarray(self.origin_world, dtype=float)
    rotation = np.asarray(self.rotation_local_to_world, dtype=float)
    if origin.shape != (3,) or rotation.shape != (3, 3):
      raise ValueError("V2 deterministic face frame shape differs")
    if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
      raise ValueError("V2 deterministic face frame must be finite")
    determinant = float(
        rotation[0, 0]
        * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
        - rotation[0, 1]
        * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
        + rotation[0, 2]
        * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
    )
    orthogonal = all(
        math.isclose(
            sum(rotation[row, first] * rotation[row, second] for row in range(3)),
            1.0 if first == second else 0.0,
            abs_tol=1e-7,
        )
        for first in range(3)
        for second in range(3)
    )
    if not orthogonal or not math.isclose(determinant, 1.0, abs_tol=1e-7):
      raise ValueError("V2 deterministic face frame rotation must be proper orthogonal")


@dataclass(frozen=True, slots=True)
class WorldResidualTransformV2:
  rotation: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
  ]
  translation: tuple[float, float, float]
  pair_origin_world: tuple[float, float, float]
  schema_version: str = WORLD_RESIDUAL_SCHEMA_VERSION

  def homogeneous_matrix(self) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(self.rotation, dtype=float)
    result[:3, 3] = np.asarray(self.translation, dtype=float)
    return result


class NonLearnedPairFrameAdapterV2:
  """Deterministic gauge-equivariant local-to-world residual adapter."""

  @staticmethod
  def _pair_frame(
      frame_a: DeterministicPredictedFaceFrameV2,
      frame_b: DeterministicPredictedFaceFrameV2,
  ) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    rotation_a = frame_a.rotation_local_to_world
    rotation_b = frame_b.rotation_local_to_world
    origin = tuple(
        0.5 * (frame_a.origin_world[index] + frame_b.origin_world[index])
        for index in range(3)
    )
    normal_a = tuple(rotation_a[row][2] for row in range(3))
    normal_b = tuple(rotation_b[row][2] for row in range(3))
    z_seed = tuple(normal_a[index] - normal_b[index] for index in range(3))
    if _norm3(z_seed) < 1e-10:
      z_seed = normal_a
    z_norm = _norm3(z_seed)
    z_axis = tuple(value / z_norm for value in z_seed)
    x_axis: tuple[float, float, float] | None = None
    seeds = (
        tuple(rotation_a[row][0] for row in range(3)),
        tuple(rotation_b[row][0] for row in range(3)),
        tuple(rotation_a[row][1] for row in range(3)),
    )
    for seed in seeds:
      projection = _dot3(seed, z_axis)
      projected = tuple(seed[index] - z_axis[index] * projection for index in range(3))
      norm = _norm3(projected)
      if norm >= 1e-10:
        x_axis = tuple(value / norm for value in projected)
        break
    if x_axis is None:
      raise ValueError("V2 deterministic pair frame is geometrically degenerate")
    y_axis = _cross3(z_axis, x_axis)
    y_norm = _norm3(y_axis)
    y_axis = tuple(value / y_norm for value in y_axis)
    x_axis = _cross3(y_axis, z_axis)
    pair_rotation = tuple(
        (x_axis[row], y_axis[row], z_axis[row]) for row in range(3)
    )
    return origin, pair_rotation

  def to_world_residual(
      self,
      executable: ExecutablePredictedInterfaceProgramV2,
      *,
      frame_a: DeterministicPredictedFaceFrameV2,
      frame_b: DeterministicPredictedFaceFrameV2,
  ) -> WorldResidualTransformV2:
    if type(executable) is not ExecutablePredictedInterfaceProgramV2:
      raise TypeError("V2 frame adapter requires the immutable executable")
    if type(frame_a) is not DeterministicPredictedFaceFrameV2 or type(
        frame_b
    ) is not DeterministicPredictedFaceFrameV2:
      raise TypeError("V2 frame adapter requires deterministic face frames")
    if (
        frame_a.part_slot != "a"
        or frame_b.part_slot != "b"
        or frame_a.graph_local_face_index
        != executable.predicted_graph_local_face_index_a
        or frame_b.graph_local_face_index
        != executable.predicted_graph_local_face_index_b
        or frame_a.frame_policy_sha256 != executable.pair_frame_policy_sha256
        or frame_b.frame_policy_sha256 != executable.pair_frame_policy_sha256
    ):
      raise ValueError("V2 face frames differ from predicted interface binding")
    origin, pair_rotation = self._pair_frame(frame_a, frame_b)
    local_rotation = _rotation_vector_to_matrix(
        executable.residual_rotation_vector_pair_local
    )
    local_translation = executable.residual_translation_pair_local
    world_rotation = _matmul3(
        _matmul3(pair_rotation, local_rotation), _transpose3(pair_rotation)
    )
    # Full conjugation T_pair @ T_local @ inverse(T_pair), not a learned
    # coordinate conversion.  This makes a gauge change G obey T' = G T G^-1.
    rotated_local_translation = _matvec3(pair_rotation, local_translation)
    rotated_origin = _matvec3(world_rotation, origin)
    world_translation = tuple(
        origin[index] + rotated_local_translation[index] - rotated_origin[index]
        for index in range(3)
    )
    return WorldResidualTransformV2(
        rotation=tuple(
            tuple(float(value) for value in row) for row in world_rotation
        ),
        translation=tuple(float(value) for value in world_translation),
        pair_origin_world=tuple(float(value) for value in origin),
    )


def _state_dict_sha256(model: nn.Module) -> str:
  digest = hashlib.sha256()
  for name, tensor in sorted(model.state_dict().items()):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes(order="C"))
  return digest.hexdigest()


class InProcessGraphInferenceExecutorV2:
  """Development-only executor; deliberately not formal-eligible."""

  preemptive = False

  def execute(
      self,
      model: JointInterfaceProgramLearnerV2,
      inputs: GraphProgramInputsV2,
      *,
      timeout_seconds: float,
  ) -> JointInterfaceProgramOutputV2:
    started = time.monotonic()
    with torch.inference_mode():
      output = model(inputs)
    if time.monotonic() - started > timeout_seconds:
      raise TimeoutError("development in-process V2 inference exceeded budget")
    return output


def _killable_inference_worker_v2(
    connection: Connection,
    model: JointInterfaceProgramLearnerV2,
    inputs: GraphProgramInputsV2,
) -> None:
  try:
    model.eval()
    with torch.inference_mode():
      output = model(inputs)
    connection.send(("ok", output))
  except BaseException as error:
    connection.send(("error", type(error).__name__, str(error)))
  finally:
    connection.close()


class KillableGraphInferenceExecutorV2:
  """Spawned CPU worker with terminate/kill enforcement at the deadline."""

  preemptive = True

  def execute(
      self,
      model: JointInterfaceProgramLearnerV2,
      inputs: GraphProgramInputsV2,
      *,
      timeout_seconds: float,
  ) -> JointInterfaceProgramOutputV2:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
      raise ValueError("V2 killable inference timeout must be finite and positive")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
      raise ValueError("V2 killable executor requires a CPU model")
    if inputs.graph_tensors.node_features.device.type != "cpu":
      raise ValueError("V2 killable executor requires CPU inputs")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_killable_inference_worker_v2,
        args=(child, model, inputs),
        daemon=True,
    )
    process.start()
    child.close()
    try:
      if not parent.poll(timeout_seconds):
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive():
          process.kill()
          process.join(timeout=2.0)
        raise TimeoutError("V2 preemptive inference worker exceeded wall time")
      message = parent.recv()
      process.join(timeout=2.0)
      if process.is_alive():
        process.kill()
        process.join(timeout=2.0)
        raise RuntimeError("V2 inference worker did not exit after publishing output")
      if not isinstance(message, tuple) or not message:
        raise RuntimeError("V2 inference worker returned a malformed message")
      if message[0] != "ok":
        raise RuntimeError(
            f"V2 inference worker failed: {message[1]}: {message[2]}"
        )
      output = message[1]
      if type(output) is not JointInterfaceProgramOutputV2:
        raise RuntimeError("V2 inference worker output type differs")
      return output
    finally:
      parent.close()
      if process.is_alive():
        process.kill()
        process.join(timeout=2.0)


@runtime_checkable
class GraphProgramProposerV2(Protocol):
  proposer_kind: str
  model_sha256: str
  formal_eligible: bool

  def propose(
      self,
      query: GraphProgramInputsV2,
      catalog: FiniteProgramCatalogRosterV2,
      *,
      budget: ProposalBudget,
  ) -> tuple[ExecutablePredictedInterfaceProgramV2, ...]: ...


class JointGraphProgramProposerV2:
  """New graph-query protocol; it is intentionally not ProgramProposer v1."""

  proposer_kind = "joint_graph_program_proposer.v2"

  def __init__(
      self,
      model: JointInterfaceProgramLearnerV2,
      *,
      executor: InProcessGraphInferenceExecutorV2 | KillableGraphInferenceExecutorV2 | None,
      require_formal: bool = True,
  ) -> None:
    if type(model) is not JointInterfaceProgramLearnerV2:
      raise TypeError("V2 proposer requires JointInterfaceProgramLearnerV2")
    if executor is None:
      raise ValueError("V2 proposer refuses a missing preemptive executor")
    if type(executor) not in {
        InProcessGraphInferenceExecutorV2,
        KillableGraphInferenceExecutorV2,
    }:
      raise TypeError("V2 proposer executor type is not auditable")
    self.formal_eligible = (
        type(executor) is KillableGraphInferenceExecutorV2
        and model.config.top_l == 8
    )
    if require_formal and not self.formal_eligible:
      raise ValueError("V2 formal proposer requires killable worker and top_l=8")
    self._model = model.eval()
    self._executor = executor
    self.model_sha256 = _state_dict_sha256(model)

  def propose(
      self,
      query: GraphProgramInputsV2,
      catalog: FiniteProgramCatalogRosterV2,
      *,
      budget: ProposalBudget,
  ) -> tuple[ExecutablePredictedInterfaceProgramV2, ...]:
    if _state_dict_sha256(self._model) != self.model_sha256:
      raise ValueError("V2 proposer model state changed after binding")
    if type(query) is not GraphProgramInputsV2:
      raise TypeError("V2 proposer query must be unannotated two-part graphs")
    if type(catalog) is not FiniteProgramCatalogRosterV2:
      raise TypeError("V2 proposer requires the exact finite-program roster")
    if query.catalog_sha256 != catalog.catalog_sha256:
      raise ValueError("V2 proposer query/catalog hash binding differs")
    if query.graph_tensors.node_features.shape[0] != 1:
      raise ValueError("V2 proposer accepts one graph pair at a time")
    if budget.candidate_pool_size != FORMAL_CATALOG_SIZE:
      raise ValueError("V2 proposer program pool must be exactly 19")
    if budget.max_occ_calls != 0:
      raise ValueError("V2 proposer requires explicit max_occ_calls=0")
    if budget.wall_time_seconds is None:
      raise ValueError("V2 proposer requires an explicit wall-time budget")
    output = self._executor.execute(
        self._model,
        query,
        timeout_seconds=budget.wall_time_seconds,
    )
    if (
        output.catalog_sha256 != catalog.catalog_sha256
        or output.model_config_sha256 != self._model.config.sha256
        or output.top_l != self._model.config.top_l
        or output.catalog_size != FORMAL_CATALOG_SIZE
    ):
      raise ValueError("V2 proposer inference output binding differs")
    face_pair_pool = self._model.config.top_l**2
    program_pool = len(catalog.entries)
    trace = GraphProgramBudgetTraceV2(
        top_k=budget.top_k,
        candidate_pool_size=budget.candidate_pool_size,
        max_occ_calls=0,
        wall_time_seconds=float(budget.wall_time_seconds),
        face_pair_pool=face_pair_pool,
        program_pool=program_pool,
        expanded_joint_hypotheses=face_pair_pool * program_pool,
        preemptive_executor=bool(self._executor.preemptive),
        formal_eligible=self.formal_eligible,
    )
    indices = output.selected_face_indices[0].detach().cpu()
    logits = output.inference_joint_logits[0].detach().cpu()
    scored: list[tuple[float, int, int, int, int, int]] = []
    for rank_a in range(self._model.config.top_l):
      face_a = int(indices[0, rank_a])
      if face_a < 0:
        continue
      for rank_b in range(self._model.config.top_l):
        face_b = int(indices[1, rank_b])
        if face_b < 0:
          continue
        for program_position in range(FORMAL_CATALOG_SIZE):
          program_index = int(output.candidate_program_indices[0, program_position])
          score = float(logits[rank_a, rank_b, program_position])
          if not math.isfinite(score):
            raise ValueError("V2 proposer produced a non-finite valid score")
          scored.append(
              (score, face_a, face_b, program_index, rank_a, rank_b)
          )
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
            catalog.entry(item[3]).program_id,
        )
    )
    if len(scored) < budget.top_k:
      raise ValueError("V2 proposer cannot satisfy top_k")
    result: list[ExecutablePredictedInterfaceProgramV2] = []
    for score, face_a, face_b, program_index, rank_a, rank_b in scored[: budget.top_k]:
      entry = catalog.entry(program_index)
      translation = output.inference_residual_translation_local[
          0, rank_a, rank_b, program_index
      ].detach().cpu().tolist()
      rotation = output.inference_residual_rotation_vector_local[
          0, rank_a, rank_b, program_index
      ].detach().cpu().tolist()
      result.append(
          ExecutablePredictedInterfaceProgramV2(
              predicted_graph_local_face_index_a=face_a,
              predicted_graph_local_face_index_b=face_b,
              program_index=program_index,
              program_id=entry.program_id,
              descriptor=entry.descriptor,
              descriptor_sha256=entry.descriptor_sha256,
              catalog_sha256=catalog.catalog_sha256,
              residual_translation_pair_local=tuple(float(v) for v in translation),
              residual_rotation_vector_pair_local=tuple(float(v) for v in rotation),
              pair_frame_policy_sha256=PAIR_FRAME_POLICY_SHA256,
              model_sha256=self.model_sha256,
              score=score,
              budget=trace,
          )
      )
    return tuple(result)


def old_program_proposer_protocol_matches_v2(value: Any) -> bool:
  """Audit helper: V2 must not structurally masquerade as ProgramProposer."""

  return isinstance(value, ProgramProposer)
