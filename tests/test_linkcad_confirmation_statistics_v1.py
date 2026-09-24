from __future__ import annotations

import pytest

from neurocad.linkcad_confirmation_statistics_v1 import (
    metric_value_v1,
    paired_query_cluster_bootstrap_v1,
    paired_two_multiseed_query_cluster_bootstrap_v1,
)
from neurocad.tools.evaluate_linkcad_confirmation_v2 import (
    evaluate_confirmation_predictions_v3,
)


def _edge(
    *,
    supervised: bool,
    part_hit: bool,
    primitive_top1_hit: bool = False,
    primitive_top8_hit: bool = False,
) -> dict:
  return {
      "edge_id": "edge",
      "edge_part_hit": part_hit,
      "primitive_supervised": supervised,
      "edge_part_and_primitive_top1_hit": primitive_top1_hit,
      "edge_part_and_primitive_top8_hit": primitive_top8_hit,
  }


def _query(query_id: str, assignment_hit: bool, edge: dict) -> dict:
  return {
      "query_id": query_id,
      "role_correct": int(assignment_hit),
      "role_count": 1,
      "assignment_hit": assignment_hit,
      "joint_mobility_hit": assignment_hit,
      "support_mobility_program_hit": assignment_hit,
      "top1_strict_part": assignment_hit,
      "top1_strict_joint_mobility": assignment_hit,
      "top1_strict_support_mobility_program": assignment_hit,
      "primitive_complete": True,
      "full_primitive_top1_program_hit": assignment_hit,
      "full_primitive_top8_program_hit": assignment_hit,
      "edges": [edge],
  }


def test_conditional_primitive_metric_uses_only_supervised_part_hits() -> None:
  rows = [
      _query(
          "q0", True,
          _edge(
              supervised=True, part_hit=True,
              primitive_top8_hit=True,
          ),
      ),
      _query(
          "q1", True,
          _edge(supervised=False, part_hit=True),
      ),
  ]
  assert metric_value_v1(
      rows, "primitive_top8_given_edge_part_hit"
  ) == 1.0


def test_program_metrics_include_support_and_full_top1() -> None:
  rows = [
      _query(
          "q0", True,
          _edge(
              supervised=True, part_hit=True,
              primitive_top1_hit=True, primitive_top8_hit=True,
          ),
      ),
      _query(
          "q1", False,
          _edge(supervised=True, part_hit=False),
      ),
  ]
  assert metric_value_v1(
      rows, "support_mobility_program_recall_at_25"
  ) == 0.5
  assert metric_value_v1(
      rows, "full_primitive_top1_program_recall_at_25"
  ) == 0.5


def test_paired_query_cluster_bootstrap_is_deterministic_and_paired() -> None:
  learned = [
      _query(
          f"q{i}", True,
          _edge(supervised=True, part_hit=True, primitive_top8_hit=True),
      )
      for i in range(12)
  ]
  baseline = [
      _query(
          f"q{i}", False,
          _edge(supervised=True, part_hit=False),
      )
      for i in range(12)
  ]
  first = paired_query_cluster_bootstrap_v1(
      learned, baseline, metric="assignment_recall_at_25",
      resamples=100, seed=17,
  )
  second = paired_query_cluster_bootstrap_v1(
      learned, baseline, metric="assignment_recall_at_25",
      resamples=100, seed=17,
  )
  assert first == second
  assert first["point_estimate_delta"] == 1.0
  assert first["percentile_ci_95"] == [1.0, 1.0]


def test_paired_query_cluster_bootstrap_rejects_unpaired_queries() -> None:
  left = [_query("q0", True, _edge(supervised=True, part_hit=True))]
  right = [_query("other", False, _edge(supervised=True, part_hit=False))]
  with pytest.raises(ValueError, match="query identities"):
    paired_query_cluster_bootstrap_v1(
        left, right, metric="assignment_recall_at_25",
        resamples=10, seed=1,
    )


def test_two_multiseed_bootstrap_compares_method_medians() -> None:
  learned = [[
      _query(f"q{i}", True, _edge(supervised=True, part_hit=True))
      for i in range(8)
  ] for _ in range(3)]
  baseline = [[
      _query(f"q{i}", False, _edge(supervised=True, part_hit=False))
      for i in range(8)
  ] for _ in range(3)]

  result = paired_two_multiseed_query_cluster_bootstrap_v1(
      learned, baseline, metric="assignment_recall_at_25",
      resamples=50, seed=9,
  )

  assert result["point_estimate_delta"] == 1.0
  assert result["percentile_ci_95"] == [1.0, 1.0]


def test_confirmation_evaluator_tracks_supervised_part_hit_denominator() -> None:
  public = {
      "queries": [{
          "query_id": "q0",
          "roles": [
              {"role_id": "a"},
              {"role_id": "b"},
              {"role_id": "c"},
          ],
          "functional_edges": [
              {"edge_id": "e0", "role_a": "a", "role_b": "b"},
              {"edge_id": "e1", "role_a": "a", "role_b": "c"},
          ],
      }],
  }
  private = {
      "targets": [{
          "query_id": "q0",
          "target_candidate_by_role": {"a": "ca", "b": "cb", "c": "cc"},
          "functional_edge_targets": [
              {
                  "edge_id": "e0", "target_mobility": "fixed",
                  "support_family": "planar_support",
              },
              {
                  "edge_id": "e1", "target_mobility": "fixed",
                  "support_family": "planar_support",
              },
          ],
      }],
  }
  program = lambda edge_id: {
      "edge_id": edge_id,
      "mobility": "fixed",
      "support_family": "planar_support",
      "interface_primitive_a": 1,
      "interface_primitive_b": 2,
      "primitive_alternatives": [{
          "interface_primitive_a": 1,
          "interface_primitive_b": 2,
      }],
  }
  predictions = {
      "rows": [{
          "query_id": "q0",
          "hypotheses": [{
              "candidate_by_role": {"a": "ca", "b": "cb", "c": "cc"},
              "edge_programs": [program("e0"), program("e1")],
          }],
      }],
  }
  result = evaluate_confirmation_predictions_v3(
      public=public,
      private=private,
      primitive_targets={("q0", "e0"): (1, 2)},
      predictions=predictions,
  )
  assert result["counts"]["edge_part_pair_recall_at_25"] == 2
  assert result["counts"]["primitive_supervised_edge_part_hit_count"] == 1
  assert result["metrics"]["primitive_top8_given_edge_part_hit"] == 1.0


def test_no_prediction_remains_in_complete_program_denominator() -> None:
  public = {
      "queries": [{
          "query_id": "q0",
          "roles": [{"role_id": "a"}, {"role_id": "b"}, {"role_id": "c"}],
          "functional_edges": [{
              "edge_id": "e0", "role_a": "a", "role_b": "b",
          }],
      }],
  }
  private = {
      "targets": [{
          "query_id": "q0",
          "target_candidate_by_role": {"a": "ca", "b": "cb", "c": "cc"},
          "functional_edge_targets": [{
              "edge_id": "e0", "target_mobility": "fixed",
              "support_family": "planar_support",
          }],
      }],
  }
  result = evaluate_confirmation_predictions_v3(
      public=public, private=private,
      primitive_targets={("q0", "e0"): (1, 2)},
      predictions={"rows": [{"query_id": "q0", "hypotheses": []}]},
  )
  assert result["counts"]["primitive_complete_query_count"] == 1
  assert result["metrics"]["full_primitive_top8_program_recall_at_25"] == 0.0
