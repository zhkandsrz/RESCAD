from __future__ import annotations

import cadquery as cq
import torch

from neurocad.linkcad_factorized_model_v1 import (
    PART_FEATURE_DIM,
    TEXT_DIM,
    LinkCADQueryTensors,
)
from neurocad.linkcad_port_constraint_model_v9 import PortConstraintLinkCADV9
from neurocad.linkcad_port_contract_v1 import describe_port_v1
from neurocad.linkcad_primitive_cache_v2 import build_port_candidate_mask_v1
from neurocad.linkcad_primitive_graph_v2 import extract_primitive_graph_v2


def test_public_port_contract_masks_geometrically_incompatible_candidates() -> None:
  cylinder = extract_primitive_graph_v2(cq.Workplane("XY").cylinder(4, 2).val())
  box = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  cylinder_ordinal = next(
      ordinal for ordinal in range(cylinder.face_orbit_count)
      if cylinder.primitive_type(ordinal) == "cylinder"
  )
  contract = describe_port_v1(cylinder, cylinder_ordinal)
  public = {
      "roles": [
          {"role_id": "r0"}, {"role_id": "r1"}, {"role_id": "r2"},
      ],
      "functional_edges": [{
          "role_a": "r0", "role_b": "r1",
          "port_contract_status": "specified",
          "port_contract_a": contract, "port_contract_b": contract,
      }],
  }
  graphs = (
      (cylinder, box), (cylinder, box), (box, cylinder),
  )

  mask = build_port_candidate_mask_v1(public, graphs)

  assert mask.tolist() == [[[True, False], [True, False]]]


def test_v9_masks_only_pair_logits_and_keeps_valid_scores_finite() -> None:
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = LinkCADQueryTensors(
      query_id="q",
      candidate_features=torch.zeros((3, 2, PART_FEATURE_DIM)),
      role_text_features=torch.zeros((3, TEXT_DIM)),
      edge_index=torch.tensor([[0, 1]]),
      edge_text_features=torch.zeros((1, TEXT_DIM)),
      candidate_mask=torch.ones((3, 2), dtype=torch.bool),
      candidate_ids=(("a0", "a1"), ("b0", "b1"), ("c0", "c1")),
      edge_ids=("e0",),
      candidate_graphs=tuple(tuple(graph for _ in range(2)) for _ in range(3)),
      port_candidate_mask=torch.tensor([[[True, False], [False, True]]]),
  )
  model = PortConstraintLinkCADV9(hidden_dim=16, seed=5)
  observed = model(query).edge_pair_logits

  assert torch.isfinite(observed[0, 0, 1])
  assert torch.isneginf(observed[0, 0, 0])
  assert torch.isneginf(observed[0, 1, 1])


def test_v9_returns_no_candidate_for_globally_infeasible_ports() -> None:
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = LinkCADQueryTensors(
      query_id="infeasible",
      candidate_features=torch.zeros((3, 2, PART_FEATURE_DIM)),
      role_text_features=torch.zeros((3, TEXT_DIM)),
      edge_index=torch.tensor([[0, 1]]),
      edge_text_features=torch.zeros((1, TEXT_DIM)),
      candidate_mask=torch.ones((3, 2), dtype=torch.bool),
      candidate_ids=(("a0", "a1"), ("b0", "b1"), ("c0", "c1")),
      edge_ids=("e0",),
      candidate_graphs=tuple(tuple(graph for _ in range(2)) for _ in range(3)),
      port_candidate_mask=torch.zeros((1, 2, 2), dtype=torch.bool),
  )

  assert PortConstraintLinkCADV9(
      hidden_dim=16, seed=5
  ).decode_beam(query, beam_size=3) == []
