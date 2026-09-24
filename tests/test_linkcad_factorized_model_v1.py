from __future__ import annotations

import torch

from neurocad.linkcad_factorized_model_v1 import (
    FactorizedLinkCADV1,
    MOBILITY_NAMES,
    frozen_text_features,
    linkcad_training_loss,
    tensorize_linkcad_query,
)


def _query_and_target():
  candidates = [
      {
          "candidate_id": f"c{i}",
          "step_path": f"{i}.step",
          "step_sha256": str(i) * 64,
          "volume": float(i + 1),
          "area": float(i + 2),
          "anchor_shape_types": ["CylinderSurfaceType"],
      }
      for i in range(3)
  ]
  public = {
      "query_id": "q",
      "roles": [
          {"role_id": "r0", "description": "shaft"},
          {"role_id": "r1", "description": "housing"},
          {"role_id": "r2", "description": "cap"},
      ],
      "candidate_sets": {"r0": candidates, "r1": candidates, "r2": candidates},
      "functional_edges": [
          {
              "edge_id": "e0", "role_a": "r0", "role_b": "r1",
              "instruction": "allow rotation without sliding",
          },
          {
              "edge_id": "e1", "role_a": "r1", "role_b": "r2",
              "instruction": "make a rigid connection",
          },
      ],
  }
  private = {
      "target_candidate_by_role": {"r0": "c0", "r1": "c1", "r2": "c2"},
      "functional_edge_targets": [
          {"edge_id": "e0", "target_mobility": "revolute"},
          {"edge_id": "e1", "target_mobility": "fixed"},
      ],
  }
  return public, private


def test_text_projection_is_frozen_and_deterministic():
  first = frozen_text_features("Allow rotation without sliding")
  second = frozen_text_features("Allow rotation without sliding")
  assert torch.equal(first, second)
  assert float(first.norm()) > 0.99


def test_tensorizer_and_factorized_decoder_shapes():
  public, private = _query_and_target()
  query = tensorize_linkcad_query(public, private)
  model = FactorizedLinkCADV1(hidden_dim=16, seed=7)
  output = model(query)
  assert output.unary_logits.shape == (3, 3)
  assert output.edge_pair_logits.shape == (2, 3, 3)
  assert output.mobility_logits.shape == (2, 3, 3, len(MOBILITY_NAMES))
  assert torch.isfinite(linkcad_training_loss(model, query))
  prediction = model.decode(query, beam_size=5)
  beam = model.decode_beam(query, beam_size=5)
  assert len(prediction.assignment) == 3
  assert len(prediction.mobility) == 2
  assert prediction.interface_orbits == ((-1, -1), (-1, -1))
  assert len(beam) == 5
  assert beam[0] == prediction


def test_requested_mobility_is_an_executable_symbolic_contract():
  public, private = _query_and_target()
  public["functional_edges"][0]["requested_mobility"] = "cylindrical"
  public["functional_edges"][1]["requested_mobility"] = "prismatic"
  query = tensorize_linkcad_query(public, private)
  model = FactorizedLinkCADV1(hidden_dim=16, seed=7)
  prediction = model.decode(query, beam_size=5)
  assert query.requested_mobility == ("cylindrical", "prismatic")
  assert prediction.mobility == ("cylindrical", "prismatic")


def test_symbolic_mobility_contract_removes_unused_learned_head_loss():
  public, private = _query_and_target()
  public["functional_edges"][0]["requested_mobility"] = "revolute"
  public["functional_edges"][1]["requested_mobility"] = "fixed"
  query = tensorize_linkcad_query(public, private)
  model = FactorizedLinkCADV1(hidden_dim=16, seed=7)
  linkcad_training_loss(model, query).backward()
  assert all(parameter.grad is None for parameter in model.mobility.parameters())


def test_symbolic_contract_can_execute_rare_mobility_without_resizing_head():
  public, _private = _query_and_target()
  public["functional_edges"][0]["requested_mobility"] = "ball"
  public["functional_edges"][1]["requested_mobility"] = "pin_slot"
  query = tensorize_linkcad_query(public, None)
  model = FactorizedLinkCADV1(hidden_dim=16, seed=7)
  assert model.decode(query, beam_size=5).mobility == ("ball", "pin_slot")


def test_two_role_variable_candidate_sets_are_padded_and_masked():
  public, _private = _query_and_target()
  public["roles"] = public["roles"][:2]
  public["candidate_sets"] = {
      "r0": public["candidate_sets"]["r0"],
      "r1": public["candidate_sets"]["r1"][:2],
  }
  public["functional_edges"] = public["functional_edges"][:1]
  query = tensorize_linkcad_query(public)
  assert query.candidate_features.shape == (2, 3, query.candidate_features.shape[-1])
  assert query.candidate_mask.tolist() == [[True, True, True], [True, True, False]]
  assert query.candidate_ids[1][2].startswith("__linkcad_padding__:")

  model = FactorizedLinkCADV1(hidden_dim=16, seed=7)
  predictions = model.decode_beam(query, beam_size=9)
  assert predictions
  assert all(prediction.assignment[1] < 2 for prediction in predictions)
  assert all(
      not candidate_id.startswith("__linkcad_padding__:")
      for prediction in predictions
      for candidate_id in prediction.candidate_ids
  )
