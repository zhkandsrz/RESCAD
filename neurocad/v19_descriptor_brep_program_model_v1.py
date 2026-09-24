"""Descriptor-conditioned fixed-catalog B-Rep learner for the V19 pilot.

This module is deliberately Torch-first.  It consumes only the immutable
catalog descriptor capability exposed by the authenticated training bundle.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .v19_authenticated_training_bundle_v1 import FixedCatalogDescriptorV1
from .v19_brep_program_model_v1 import (
    BRepProgramInputs,
    BRepProgramLearnerConfig,
    BRepProgramLearnerV1,
    BRepProgramOutput,
)
from .v19_training_bundle_protocol_v1 import canonical_sha256


PROGRAM_TYPES = ("insert", "link", "support")
CONTACT_TYPES = (
    "boss_in_slot",
    "generic_contact",
    "pin_in_slot",
    "seat_plane",
    "shaft_in_bore",
)
SURFACE_TYPES = ("cone", "cylinder", "plane", "sphere", "spline", "torus")
DESCRIPTOR_FEATURE_DIM = (
    len(PROGRAM_TYPES)
    + len(CONTACT_TYPES)
    + 2 * len(SURFACE_TYPES)
)
FIXED_CATALOG_SIZE = 19
_DESCRIPTOR_FIELDS = {
    "contact_type",
    "relation_hint",
    "surface_type_a",
    "surface_type_b",
}


def fixed_catalog_descriptor_features_v1(
    descriptors: Sequence[FixedCatalogDescriptorV1],
) -> Tensor:
  """Encode the exact pre-V19 catalog as four deterministic one-hot blocks."""

  rows = tuple(descriptors)
  if (
      len(rows) != FIXED_CATALOG_SIZE
      or any(type(row) is not FixedCatalogDescriptorV1 for row in rows)
      or tuple(row.program_index for row in rows)
      != tuple(range(FIXED_CATALOG_SIZE))
  ):
    raise ValueError("fixed catalog descriptor roster/order differs")
  vocabularies = (
      PROGRAM_TYPES,
      CONTACT_TYPES,
      SURFACE_TYPES,
      SURFACE_TYPES,
  )
  features = torch.zeros(
      (FIXED_CATALOG_SIZE, DESCRIPTOR_FEATURE_DIM), dtype=torch.float32
  )
  for row_index, row in enumerate(rows):
    descriptor = row.descriptor
    if (
        set(descriptor) != _DESCRIPTOR_FIELDS
        or any(type(value) is not str for value in descriptor.values())
        or row.program_type != descriptor["relation_hint"]
        or row.descriptor_sha256 != canonical_sha256(dict(descriptor))
        or row.program_id != f"catalog_{row.descriptor_sha256[:16]}"
    ):
      raise ValueError("fixed catalog descriptor binding differs")
    values = (
        row.program_type,
        descriptor["contact_type"],
        descriptor["surface_type_a"],
        descriptor["surface_type_b"],
    )
    offset = 0
    for value, vocabulary in zip(values, vocabularies, strict=True):
      try:
        category_index = vocabulary.index(value)
      except ValueError as error:
        raise ValueError("fixed catalog descriptor category differs") from error
      features[row_index, offset + category_index] = 1.0
      offset += len(vocabulary)
  return features


class DescriptorConditionedBRepProgramLearnerV1(BRepProgramLearnerV1):
  """B-Rep scorer whose candidate representations share descriptor semantics."""

  def __init__(
      self,
      config: BRepProgramLearnerConfig,
      *,
      seed: int,
      catalog_descriptors: Sequence[FixedCatalogDescriptorV1],
  ) -> None:
    if type(config) is not BRepProgramLearnerConfig:
      raise TypeError("descriptor learner config type differs")
    if config.catalog_size != FIXED_CATALOG_SIZE:
      raise ValueError("descriptor learner requires the fixed19 catalog")
    if type(seed) is not int:
      raise TypeError("descriptor learner seed must be an actual integer")
    features = fixed_catalog_descriptor_features_v1(catalog_descriptors)
    super().__init__(config, seed=seed)
    del self.catalog_embedding
    self.register_buffer(
        "catalog_descriptor_features",
        features.detach().clone(),
        persistent=True,
    )
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices):
      torch.manual_seed(seed ^ 0xD35C)
      self.descriptor_encoder = nn.Sequential(
          nn.Linear(DESCRIPTOR_FEATURE_DIM, config.hidden_dim),
          nn.SiLU(),
          nn.Linear(config.hidden_dim, config.hidden_dim),
      )

  def encode_all_catalog_descriptors(self) -> Tensor:
    """Apply one shared encoder to all fixed semantic descriptor rows."""

    return self.descriptor_encoder(self.catalog_descriptor_features)

  def forward(self, inputs: BRepProgramInputs) -> BRepProgramOutput:
    if type(inputs) is not BRepProgramInputs:
      raise TypeError("descriptor learner forward input type differs")
    candidate_indices = inputs.candidate_program_indices
    if candidate_indices.numel() and (
        int(candidate_indices.min()) < 0
        or int(candidate_indices.max()) >= FIXED_CATALOG_SIZE
    ):
      raise ValueError("candidate index lies outside the fixed19 catalog")
    graph_embeddings = self._encode_graphs(inputs)
    if graph_embeddings.shape[1] == 1:
      graph_embeddings = torch.cat(
          (graph_embeddings, torch.zeros_like(graph_embeddings)), dim=1
      )
    left, right = graph_embeddings[:, 0], graph_embeddings[:, 1]
    query = self.query_encoder(
        torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
    )
    descriptor_embeddings = self.encode_all_catalog_descriptors()
    candidates = descriptor_embeddings[candidate_indices]
    expanded = query.unsqueeze(1).expand_as(candidates)
    cross = torch.cat(
        (
            expanded,
            candidates,
            expanded * candidates,
            torch.abs(expanded - candidates),
        ),
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


__all__ = [
    "CONTACT_TYPES",
    "DESCRIPTOR_FEATURE_DIM",
    "DescriptorConditionedBRepProgramLearnerV1",
    "FIXED_CATALOG_SIZE",
    "PROGRAM_TYPES",
    "SURFACE_TYPES",
    "fixed_catalog_descriptor_features_v1",
]
