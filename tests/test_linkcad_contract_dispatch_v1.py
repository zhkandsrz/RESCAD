from __future__ import annotations

from neurocad.linkcad_contract_dispatch_v1 import (
    role_rankings_from_prediction_beam_v1,
)


def test_contract_dispatch_preserves_frozen_top_assignment():
  public = {"queries": [{
      "query_id": "q0",
      "roles": [
          {"role_id": "shaft", "candidates": [
              {"candidate_id": "s0"}, {"candidate_id": "s1"},
          ]},
          {"role_id": "wheel", "candidates": [
              {"candidate_id": "w0"}, {"candidate_id": "w1"},
          ]},
      ],
  }]}
  predictions = {"rows": [{
      "query_id": "q0",
      "hypotheses": [
          {"candidate_by_role": {"shaft": "s1", "wheel": "w0"}},
          {"candidate_by_role": {"shaft": "s0", "wheel": "w1"}},
      ],
  }]}

  rankings = role_rankings_from_prediction_beam_v1(
      public=public, predictions=predictions,
  )

  assert rankings["q0"]["shaft"] == ["s1", "s0"]
  assert rankings["q0"]["wheel"] == ["w0", "w1"]
