"""Dual-branch LinkCAD model for port-conditioned executable linking."""

from __future__ import annotations

import torch

from .linkcad_brep_encoder_v1 import LinkCADBRepEncoderV1
from .linkcad_factorized_model_v1 import TEXT_DIM
from .linkcad_primitive_factorized_model_v2 import PrimitiveFactorizedLinkCADV2


class PortConditionedLinkCADV6(PrimitiveFactorizedLinkCADV2):
  """Separate part retrieval from language-conditioned port grounding."""

  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    hidden_dim = self.hidden_dim
    seed = int(kwargs.get("seed", 1701))
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(seed + 61)
      self.part_brep_encoder = LinkCADBRepEncoderV1(
          hidden_dim=hidden_dim, layers=3
      )
      self.interface_text_encoder = torch.nn.Sequential(
          torch.nn.Linear(TEXT_DIM, hidden_dim),
          torch.nn.SiLU(),
          torch.nn.Linear(hidden_dim, hidden_dim),
      )

  def _encode_graphs(self, available_graphs):
    face_graphs = [graph.face_graph for graph in available_graphs]
    part_embeddings = self.part_brep_encoder.forward_many(
        face_graphs, orbit_mode=self.orbit_mode
    )
    _interface_parts, face_orbit_sets = (
        self.brep_encoder.forward_many_with_orbits(
            face_graphs, orbit_mode=self.orbit_mode
        )
    )
    primitive_sets = []
    for index, graph in enumerate(available_graphs):
      primitive = self.primitive_encoder(
          graph.primitive_features.to(part_embeddings.device)
      )
      face_orbits = face_orbit_sets[index]
      if graph.face_orbit_count != int(face_orbits.shape[0]):
        raise ValueError("LinkCAD V6 primitive/face orbit identity differs")
      primitive = primitive.clone()
      primitive[:graph.face_orbit_count] = self.primitive_norm(
          primitive[:graph.face_orbit_count] + face_orbits
      )
      primitive_sets.append(primitive)
    return part_embeddings, tuple(primitive_sets)

  def _encode_interface_text(self, features):
    return self.interface_text_encoder(features)


__all__ = ["PortConditionedLinkCADV6"]
