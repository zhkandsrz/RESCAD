"""Face-and-edge primitive LinkCAD model with factorized support and mobility."""

from __future__ import annotations

import torch

from .linkcad_brep_factorized_model_v1 import BRepFactorizedLinkCADV1
from .linkcad_factorized_model_v1 import SUPPORT_NAMES
from .linkcad_primitive_graph_v2 import PRIMITIVE_FEATURE_DIM


class PrimitiveFactorizedLinkCADV2(BRepFactorizedLinkCADV1):
  """Encode B-Rep topology for parts and face/edge orbits for interfaces."""

  def __init__(
      self,
      *,
      hidden_dim: int = 64,
      use_language: bool = True,
      use_global: bool = False,
      orbit_mode: str = "attention",
      max_orbits_per_candidate: int = 16,
      seed: int = 1701,
  ) -> None:
    super().__init__(
        hidden_dim=hidden_dim,
        use_language=use_language,
        use_global=use_global,
        orbit_mode=orbit_mode,
        max_orbits_per_candidate=max_orbits_per_candidate,
        seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(seed + 29)
      self.primitive_encoder = torch.nn.Sequential(
          torch.nn.Linear(PRIMITIVE_FEATURE_DIM, hidden_dim),
          torch.nn.SiLU(),
          torch.nn.Linear(hidden_dim, hidden_dim),
      )
      self.primitive_norm = torch.nn.LayerNorm(hidden_dim)
      self.part_primitive_fusion = torch.nn.Sequential(
          torch.nn.Linear(3 * hidden_dim, hidden_dim),
          torch.nn.SiLU(),
          torch.nn.Linear(hidden_dim, hidden_dim),
      )
      self.interface_edge = torch.nn.Sequential(
          torch.nn.Linear(7 * hidden_dim, 2 * hidden_dim),
          torch.nn.SiLU(),
          torch.nn.Linear(2 * hidden_dim, hidden_dim),
      )
      self.interface_score = torch.nn.Linear(hidden_dim, 1)
      self.interface_mobility = torch.nn.Linear(hidden_dim, 4)
      self.support_head = torch.nn.Linear(hidden_dim, len(SUPPORT_NAMES))

  def _encode_graphs(self, available_graphs):
    face_graphs = [graph.face_graph for graph in available_graphs]
    face_parts, face_orbit_sets = self.brep_encoder.forward_many_with_orbits(
        face_graphs, orbit_mode=self.orbit_mode
    )
    primitive_sets = []
    for index, graph in enumerate(available_graphs):
      features = graph.primitive_features.to(face_parts.device)
      primitive = self.primitive_encoder(features)
      face_orbits = face_orbit_sets[index]
      if graph.face_orbit_count != int(face_orbits.shape[0]):
        raise ValueError("LinkCAD primitive/face orbit identity differs")
      primitive = primitive.clone()
      primitive[:graph.face_orbit_count] = self.primitive_norm(
          primitive[:graph.face_orbit_count] + face_orbits
      )
      primitive_sets.append(primitive)
    return face_parts, tuple(primitive_sets)

  def _interface_hidden(self, inputs):
    return self.interface_edge(inputs)

  def _interface_scores(self, hidden):
    return self.interface_score(hidden)

  def _interface_mobility(self, hidden):
    return self.interface_mobility(hidden)

  def _assignment_pair_scores(self, part_scores, _interface_scores):
    return part_scores

  def _decode_support(
      self, query, edge_ordinal, candidate_a, candidate_b,
      primitive_pair, learned_support,
  ):
    role_a, role_b = query.edge_index[edge_ordinal].tolist()
    graph_a = query.candidate_graphs[role_a][candidate_a]
    graph_b = query.candidate_graphs[role_b][candidate_b]
    if graph_a is None or graph_b is None or min(primitive_pair) < 0:
      return learned_support
    type_a = graph_a.primitive_type(primitive_pair[0])
    type_b = graph_b.primitive_type(primitive_pair[1])
    axis_types = {"circle", "cylinder", "cone", "line"}
    if type_a == type_b == "line":
      return "linear_support"
    if type_a in axis_types and type_b in axis_types:
      return "axial_support"
    if type_a == type_b == "plane":
      return "planar_support"
    if (
        (type_a == "plane" and type_b in axis_types)
        or (type_b == "plane" and type_a in axis_types)
    ):
      return "axis_plane_support"
    return learned_support


__all__ = ["PrimitiveFactorizedLinkCADV2"]
