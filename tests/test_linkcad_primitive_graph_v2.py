from __future__ import annotations

import cadquery as cq
import torch

from neurocad.linkcad_factorized_model_v1 import (
    SUPPORT_NAMES,
    LinkCADQueryTensors,
    linkcad_training_loss,
    tensorize_linkcad_query,
)
from neurocad.linkcad_primitive_factorized_model_v2 import (
    PrimitiveFactorizedLinkCADV2,
)
from neurocad.linkcad_primitive_factorized_model_v3 import (
    PrimitiveFactorizedLinkCADV3,
)
from neurocad.linkcad_primitive_graph_v2 import (
    PRIMITIVE_FEATURE_DIM,
    extract_primitive_graph_v2,
)
from neurocad.linkcad_primitive_graph_v3 import extract_primitive_graph_v3
from neurocad.linkcad_port_conditioned_model_v6 import PortConditionedLinkCADV6


def _query():
  candidates = [
      {
          "candidate_id": f"c{i}", "step_path": f"{i}.step",
          "step_sha256": str(i + 1) * 64,
          "volume": float(i + 1), "area": float(i + 2),
      }
      for i in range(2)
  ]
  public = {
      "query_id": "primitive_q",
      "roles": [
          {"role_id": "r0", "description": "shaft"},
          {"role_id": "r1", "description": "housing"},
          {"role_id": "r2", "description": "cap"},
      ],
      "candidate_sets": {key: candidates for key in ("r0", "r1", "r2")},
      "functional_edges": [{
          "edge_id": "e0", "role_a": "r0", "role_b": "r1",
          "instruction": "allow rotation around their shared axis",
      }],
  }
  private = {
      "target_candidate_by_role": {"r0": "c0", "r1": "c1", "r2": "c0"},
      "functional_edge_targets": [{
          "edge_id": "e0", "target_mobility": "revolute",
          "support_family": "axial_support",
      }],
  }
  return tensorize_linkcad_query(public, private)


def test_primitive_graph_contains_face_and_edge_orbits():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  assert graph.primitive_features.shape[1] == PRIMITIVE_FEATURE_DIM
  assert graph.face_orbit_count > 0
  assert set(graph.primitive_kinds) == {"face", "edge"}
  assert len(graph.primitive_members) == graph.primitive_features.shape[0]


def test_primitive_model_predicts_support_mobility_and_interface():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = _query()
  query = LinkCADQueryTensors(
      query_id=query.query_id,
      candidate_features=query.candidate_features,
      role_text_features=query.role_text_features,
      edge_index=query.edge_index,
      edge_text_features=query.edge_text_features,
      candidate_mask=query.candidate_mask,
      target_assignment=query.target_assignment,
      target_mobility=query.target_mobility,
      target_support=query.target_support,
      candidate_ids=query.candidate_ids,
      edge_ids=query.edge_ids,
      candidate_graphs=tuple(tuple(graph for _ in range(2)) for _ in range(3)),
  )
  model = PrimitiveFactorizedLinkCADV2(
      hidden_dim=16, max_orbits_per_candidate=4, seed=13,
  )
  output = model(query)
  assert output.support_logits.shape == (1, 2, 2, len(SUPPORT_NAMES))
  assert output.interface_score_blocks[0][0][1] is not None
  assert torch.isfinite(linkcad_training_loss(model, query))
  prediction = model.decode(query, beam_size=3)
  assert len(prediction.support) == 1
  assert prediction.interface_orbits[0] != (-1, -1)


def test_primitive_interface_training_uses_part_context_fusion():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = _query()
  query = LinkCADQueryTensors(
      query_id=query.query_id,
      candidate_features=query.candidate_features,
      role_text_features=query.role_text_features,
      edge_index=query.edge_index,
      edge_text_features=query.edge_text_features,
      candidate_mask=query.candidate_mask,
      target_assignment=query.target_assignment,
      target_mobility=query.target_mobility,
      target_support=query.target_support,
      candidate_ids=query.candidate_ids,
      edge_ids=query.edge_ids,
      candidate_graphs=tuple(tuple(graph for _ in range(2)) for _ in range(3)),
  )
  model = PrimitiveFactorizedLinkCADV3(
      hidden_dim=16, max_orbits_per_candidate=4, seed=13,
  )
  interface = {"e0": (0, 0)}
  linkcad_training_loss(
      model, query, interface_target_orbits=interface,
  ).backward()
  gradients = [
      parameter.grad for parameter in model.part_primitive_fusion.parameters()
  ]
  assert all(gradient is not None for gradient in gradients)
  assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_v3_primitive_graph_retains_normalized_cylindrical_radius():
  small = extract_primitive_graph_v3(
      cq.Workplane("XY").cylinder(10, 3).val()
  )
  large = extract_primitive_graph_v3(
      cq.Workplane("XY").cylinder(20, 6).val()
  )
  small_index = next(
      index for index in range(small.face_orbit_count)
      if small.primitive_type(index) == "cylinder"
  )
  large_index = next(
      index for index in range(large.face_orbit_count)
      if large.primitive_type(index) == "cylinder"
  )
  assert float(small.primitive_features[small_index, 1]) > 0.0
  assert torch.isclose(
      small.primitive_features[small_index, 1],
      large.primitive_features[large_index, 1],
      atol=1e-6,
  )


def test_port_v6_separates_part_and_interface_encoders():
  model = PortConditionedLinkCADV6(
      hidden_dim=16, max_orbits_per_candidate=4, seed=13,
  )
  assert model.part_brep_encoder is not model.brep_encoder
  assert model.interface_text_encoder is not model.text_encoder
