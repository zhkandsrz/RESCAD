from __future__ import annotations

import hashlib

import cadquery as cq

from neurocad.linkcad_global_assembly_execution_v1 import (
    compose_role_poses_v1,
    execute_global_assembly_subset_v1,
)


def _pose(x: float) -> list[float]:
  return [
      1.0, 0.0, 0.0, x,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
  ]


def _fixture(tmp_path, child_two_x: float):
  path = tmp_path / "box.step"
  cq.exporters.export(cq.Workplane("XY").box(10.0, 10.0, 10.0), str(path))
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  roles = ["root", "child_1", "child_2"]
  candidate_sets = {
      role: [{
          "candidate_id": f"part_{role}",
          "step_path": path.name,
          "step_sha256": digest,
      }]
      for role in roles
  }
  query = {
      "query_id": "global_fixture",
      "roles": [{"role_id": role} for role in roles],
      "functional_edges": [
          {"edge_id": "edge_1", "role_a": "root", "role_b": "child_1"},
          {"edge_id": "edge_2", "role_a": "root", "role_b": "child_2"},
      ],
      "candidate_sets": candidate_sets,
  }
  hypothesis = {
      "rank": 0,
      "candidate_by_role": {
          role: f"part_{role}" for role in roles
      },
      "edge_programs": [],
  }
  execution_rows = [
      {
          "query_id": "global_fixture", "edge_id": "edge_1",
          "hypothesis_rank": 0,
          "candidate_id_a": "part_root",
          "candidate_id_b": "part_child_1",
          "conditional_kinematic_feasibility_accepted": True,
          "selected_observation": {"child_world_row_major": _pose(10.0)},
      },
      {
          "query_id": "global_fixture", "edge_id": "edge_2",
          "hypothesis_rank": 0,
          "candidate_id_a": "part_root",
          "candidate_id_b": "part_child_2",
          "conditional_kinematic_feasibility_accepted": True,
          "selected_observation": {"child_world_row_major": _pose(child_two_x)},
      },
  ]
  public = {"contains_private_targets": False, "queries": [query]}
  predictions = {
      "private_targets_opened": False,
      "prediction_payload_sha256": "prediction",
      "rows": [{"query_id": "global_fixture", "hypotheses": [hypothesis]}],
  }
  execution = {
      "private_targets_opened": False,
      "prediction_payload_sha256": "prediction",
      "subset_payload_sha256": "pair-execution",
      "rows": execution_rows,
  }
  return query, hypothesis, execution_rows, public, predictions, execution


def test_global_pose_audit_accepts_collision_free_children(tmp_path):
  _, _, _, public, predictions, execution = _fixture(tmp_path, -10.0)
  result = execute_global_assembly_subset_v1(
      public=public, predictions=predictions, pair_execution=execution,
      dataset_root=tmp_path,
  )
  assert result["accepted_query_count"] == 1
  row = result["rows"][0]
  assert row["status"] == "accepted_global_assembly"
  assert row["pair_count"] == 3
  assert row["non_edge_collision_pair_count"] == 0


def test_global_pose_audit_rejects_child_child_collision(tmp_path):
  _, _, _, public, predictions, execution = _fixture(tmp_path, 10.0)
  result = execute_global_assembly_subset_v1(
      public=public, predictions=predictions, pair_execution=execution,
      dataset_root=tmp_path,
  )
  row = result["rows"][0]
  assert row["status"] == "rejected_no_collision_free_global_pose"
  assert row["collision_pruned_state_count"] >= 1


def test_global_pose_search_uses_later_public_pose_to_avoid_collision(tmp_path):
  _, _, rows, public, predictions, execution = _fixture(tmp_path, 10.0)
  rows[1]["accepted_observations"] = [
      {"pose_rank": 0, "child_world_row_major": _pose(10.0)},
      {"pose_rank": 1, "child_world_row_major": _pose(-10.0)},
  ]
  rows[1]["accepted_pose_count"] = 2
  result = execute_global_assembly_subset_v1(
      public=public, predictions=predictions, pair_execution=execution,
      dataset_root=tmp_path,
  )
  row = result["rows"][0]
  assert row["status"] == "accepted_global_assembly"
  assert row["selected_pose_rank_by_edge"]["edge_2"] == 1


def test_pose_composition_fails_closed_when_edge_evidence_is_missing(tmp_path):
  query, hypothesis, rows, _, _, _ = _fixture(tmp_path, -10.0)
  result = compose_role_poses_v1(
      query=query, hypothesis=hypothesis, execution_rows=rows[:1],
  )
  assert result["status"] == "unknown_incomplete_pair_execution"
  assert result["accepted_edge_count"] == 1


def _add_cycle(query, rows, *, closure_x: float) -> None:
  query["functional_edges"].append({
      "edge_id": "edge_3", "role_a": "child_1", "role_b": "child_2",
  })
  rows.append({
      "query_id": "global_fixture", "edge_id": "edge_3",
      "hypothesis_rank": 0,
      "candidate_id_a": "part_child_1",
      "candidate_id_b": "part_child_2",
      "conditional_kinematic_feasibility_accepted": True,
      "selected_observation": {"child_world_row_major": _pose(closure_x)},
  })


def test_global_pose_search_accepts_consistent_cycle(tmp_path):
  query, _, rows, public, predictions, execution = _fixture(tmp_path, -10.0)
  _add_cycle(query, rows, closure_x=-20.0)
  result = execute_global_assembly_subset_v1(
      public=public, predictions=predictions, pair_execution=execution,
      dataset_root=tmp_path,
  )
  row = result["rows"][0]
  assert row["status"] == "accepted_global_assembly"
  assert row["closure_edge_count"] == 1
  assert row["composition"]["closure_checks"][-1]["within_tolerance"] is True


def test_global_pose_search_rejects_inconsistent_cycle(tmp_path):
  query, _, rows, public, predictions, execution = _fixture(tmp_path, -10.0)
  _add_cycle(query, rows, closure_x=-30.0)
  result = execute_global_assembly_subset_v1(
      public=public, predictions=predictions, pair_execution=execution,
      dataset_root=tmp_path,
  )
  row = result["rows"][0]
  assert row["status"] == "rejected_no_cycle_consistent_global_pose"
  assert row["closure_pruned_state_count"] == 1
