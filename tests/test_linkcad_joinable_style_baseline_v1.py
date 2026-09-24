from __future__ import annotations

from dataclasses import replace

import cadquery as cq
import torch

from neurocad.linkcad_factorized_model_v1 import (
    LinkCADPrediction,
    LinkCADQueryTensors,
    linkcad_training_loss,
    tensorize_linkcad_query,
)
from neurocad.linkcad_joinable_style_baseline_v1 import (
    JoinABLeStyleLinkCADV1,
)
from neurocad.linkcad_primitive_graph_v2 import extract_primitive_graph_v2
from neurocad.tools.evaluate_linkcad_joinable_style_v1 import (
    joinable_style_prediction_row_v2,
)


def _query(graph):
  candidates = [
      {
          "candidate_id": f"c{index}",
          "step_path": f"{index}.step",
          "step_sha256": str(index + 1) * 64,
          "volume": float(index + 1),
          "area": float(index + 2),
      }
      for index in range(2)
  ]
  public = {
      "query_id": "joinable_style_q",
      "roles": [
          {"role_id": "r0", "description": "shaft"},
          {"role_id": "r1", "description": "bearing"},
          {"role_id": "r2", "description": "gear"},
      ],
      "candidate_sets": {role: candidates for role in ("r0", "r1", "r2")},
      "functional_edges": [{
          "edge_id": "e0",
          "role_a": "r0",
          "role_b": "r1",
          "instruction": "put the shaft through the bearing",
          "mobility": "revolute",
      }],
  }
  private = {
      "target_candidate_by_role": {"r0": "c0", "r1": "c1", "r2": "c0"},
      "functional_edge_targets": [{
          "edge_id": "e0",
          "target_mobility": "revolute",
          "support_family": "axial_support",
      }],
  }
  query = tensorize_linkcad_query(public, private)
  return LinkCADQueryTensors(
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
      requested_mobility=query.requested_mobility,
  )


def test_joinable_style_scores_interfaces_and_complete_assignments():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = _query(graph)
  model = JoinABLeStyleLinkCADV1(
      hidden_dim=16, max_orbits_per_candidate=4, seed=23
  )
  output = model(query)
  assert torch.equal(output.unary_logits, torch.zeros_like(output.unary_logits))
  assert torch.isfinite(output.edge_pair_logits).all()
  assert output.interface_score_blocks[0][0][1] is not None
  prediction = model.decode(query, beam_size=3)
  assert len(prediction.assignment) == 3
  assert prediction.interface_orbits[0] != (-1, -1)


def test_joinable_style_is_invariant_to_connection_text():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = _query(graph)
  model = JoinABLeStyleLinkCADV1(
      hidden_dim=16, max_orbits_per_candidate=4, seed=23
  ).eval()
  changed = replace(
      query,
      edge_text_features=torch.randn_like(query.edge_text_features),
  )
  with torch.no_grad():
    original = model(query)
    counterfactual = model(changed)
  assert torch.equal(original.edge_pair_logits, counterfactual.edge_pair_logits)
  for original_row, changed_row in zip(
      original.interface_score_blocks[0],
      counterfactual.interface_score_blocks[0],
      strict=True,
  ):
    for original_block, changed_block in zip(
        original_row, changed_row, strict=True
    ):
      assert torch.equal(original_block, changed_block)


def test_joinable_style_uses_the_same_public_port_filter():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = _query(graph)
  port_mask = torch.ones((1, 2, 2), dtype=torch.bool)
  port_mask[0, 0, 1] = False
  constrained = replace(query, port_candidate_mask=port_mask)
  model = JoinABLeStyleLinkCADV1(
      hidden_dim=16, max_orbits_per_candidate=4, seed=23
  )
  output = model(constrained)
  assert torch.isneginf(output.edge_pair_logits[0, 1]).all()
  assert torch.isfinite(output.edge_pair_logits[0, 0]).all()


def test_joinable_style_pair_head_is_symmetric():
  model = JoinABLeStyleLinkCADV1(
      hidden_dim=16, max_orbits_per_candidate=4, seed=23
  )
  left = torch.randn(5, 7, 16)
  right = torch.randn(5, 7, 16)
  zeros = torch.zeros_like(left)
  first = torch.cat((left, right, zeros, left * right, abs(left - right), zeros, zeros), dim=-1)
  second = torch.cat((right, left, zeros, left * right, abs(left - right), zeros, zeros), dim=-1)
  score_first = model._interface_scores(model._interface_hidden(first))
  score_second = model._interface_scores(model._interface_hidden(second))
  assert torch.allclose(score_first, score_second, atol=1e-7)


def test_joinable_style_training_reaches_graph_and_link_parameters():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = _query(graph)
  model = JoinABLeStyleLinkCADV1(
      hidden_dim=16, max_orbits_per_candidate=4, seed=23
  )
  loss = linkcad_training_loss(
      model,
      query,
      interface_target_orbits={"e0": (0, 0)},
  )
  assert torch.isfinite(loss)
  loss.backward()
  assert model.link_score.weight.grad is not None
  assert float(model.link_score.weight.grad.abs().sum()) > 0.0
  graph_gradients = [
      parameter.grad
      for parameter in model.joinable_encoder.parameters()
      if parameter.requires_grad
  ]
  assert any(
      gradient is not None and float(gradient.abs().sum()) > 0.0
      for gradient in graph_gradients
  )


def test_joinable_style_prediction_row_matches_confirmation_evaluator_schema():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  query = _query(graph)
  prediction = LinkCADPrediction(
      assignment=(1, 0, 1),
      candidate_ids=("c1", "c0", "c1"),
      mobility=("revolute",),
      support=("axial_support",),
      interface_orbits=((3, 7),),
      interface_alternatives=(((3, 7, 0.9), (4, 8, 0.4)),),
      score=1.25,
  )

  row = joinable_style_prediction_row_v2(
      query, [prediction], role_ids=("r0", "r1", "r2")
  )

  assert row["query_id"] == query.query_id
  assert row["terminal_status"] == "predicted"
  hypothesis = row["hypotheses"][0]
  assert hypothesis["rank"] == 0
  assert hypothesis["candidate_by_role"] == {
      "r0": "c1", "r1": "c0", "r2": "c1",
  }
  assert hypothesis["edge_programs"] == [{
      "edge_id": "e0",
      "mobility": "revolute",
      "support_family": "axial_support",
      "interface_primitive_a": 3,
      "interface_primitive_b": 7,
      "primitive_alternatives": [
          {
              "rank": 1,
              "interface_primitive_a": 3,
              "interface_primitive_b": 7,
              "model_score": 0.9,
          },
          {
              "rank": 2,
              "interface_primitive_a": 4,
              "interface_primitive_b": 8,
              "model_score": 0.4,
          },
      ],
  }]
