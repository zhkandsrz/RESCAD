from __future__ import annotations

import hashlib
import json

import cadquery as cq

from neurocad.linkcad_assembly_export_v1 import (
    export_assembly_package_v1,
    inspect_assembly_step_v1,
)


def _pose(x: float) -> list[float]:
  return [
      1.0, 0.0, 0.0, x,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
  ]


def _fixture(tmp_path):
  source = tmp_path / "source.step"
  cq.exporters.export(cq.Workplane("XY").box(4.0, 4.0, 4.0), str(source))
  digest = hashlib.sha256(source.read_bytes()).hexdigest()
  query = {
      "query_id": "export_fixture",
      "roles": [{"role_id": "base"}, {"role_id": "shaft"}],
      "functional_edges": [{
          "edge_id": "edge_0",
          "role_a": "base",
          "role_b": "shaft",
          "instruction": "put the shaft through the base",
          "requested_mobility": "revolute",
      }],
      "candidate_sets": {
          role: [{
              "candidate_id": f"candidate_{role}",
              "step_path": source.name,
              "step_sha256": digest,
          }]
          for role in ("base", "shaft")
      },
  }
  hypothesis = {
      "rank": 0,
      "candidate_by_role": {
          "base": "candidate_base", "shaft": "candidate_shaft",
      },
      "edge_programs": [{
          "edge_id": "edge_0",
          "mobility": "revolute",
          "support_family": "axial_support",
          "interface_primitive_a": 2,
          "interface_primitive_b": 5,
      }],
  }
  global_execution = {
      "query_id": "export_fixture",
      "private_targets_opened": False,
      "status": "accepted_global_assembly",
      "global_assembly_conditionally_feasible": True,
      "composition": {
          "status": "composed",
          "anchor_role": "base",
          "role_pose_row_major": {
              "base": _pose(0.0), "shaft": _pose(12.0),
          },
      },
      "selected_pose_rank_by_edge": {"edge_0": 1},
  }
  first_observation = {
      "pose_rank": 0,
      "child_world_row_major": _pose(8.0),
      "representative_raw_face_index_a": 2,
      "representative_raw_face_index_b": 5,
      "face_signature_sha256_a": "a" * 64,
      "face_signature_sha256_b": "b" * 64,
  }
  selected_observation = {
      **first_observation,
      "pose_rank": 1,
      "child_world_row_major": _pose(12.0),
  }
  pair_rows = [{
      "query_id": "export_fixture",
      "edge_id": "edge_0",
      "hypothesis_rank": 0,
      "candidate_id_a": "candidate_base",
      "candidate_id_b": "candidate_shaft",
      "conditional_kinematic_feasibility_accepted": True,
      "selected_observation": first_observation,
      "accepted_observations": [first_observation, selected_observation],
  }]
  return query, hypothesis, global_execution, pair_rows


def test_exported_package_preserves_hierarchy_poses_and_connections(tmp_path):
  query, hypothesis, global_execution, pair_rows = _fixture(tmp_path)
  output_step = tmp_path / "assembled.step"
  output_manifest = tmp_path / "assembled.manifest.json"

  receipt = export_assembly_package_v1(
      query=query,
      hypothesis=hypothesis,
      global_execution=global_execution,
      pair_execution_rows=pair_rows,
      dataset_root=tmp_path,
      output_step=output_step,
      output_manifest=output_manifest,
  )

  assert receipt["status"] == "exported_and_reimport_verified"
  assert output_step.is_file()
  assert output_manifest.is_file()
  manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
  assert manifest["schema_version"] == "linkcad_assembly_manifest.v1"
  assert manifest["root_assembly_name"] == "linkcad_export_fixture"
  assert [row["role_id"] for row in manifest["components"]] == [
      "base", "shaft",
  ]
  assert manifest["components"][1]["world_pose_row_major"] == _pose(12.0)
  assert manifest["connections"] == [{
      "edge_id": "edge_0",
      "role_a": "base",
      "role_b": "shaft",
      "instruction": "put the shaft through the base",
      "requested_mobility": "revolute",
      "predicted_mobility": "revolute",
      "support_family": "axial_support",
      "interface_primitive_a": 2,
      "interface_primitive_b": 5,
      "selected_observation": pair_rows[0]["accepted_observations"][1],
  }]

  inspection = inspect_assembly_step_v1(output_step)
  assert inspection["root_count"] == 1
  assert inspection["root_is_assembly"] is True
  assert inspection["component_count"] == 2
  assert inspection["component_names"] == ["base", "shaft"]
  assert inspection["translation_by_component_mm"] == {
      "base": [0.0, 0.0, 0.0], "shaft": [12.0, 0.0, 0.0],
  }
