from __future__ import annotations

import hashlib

import cadquery as cq

from neurocad.linkcad_assembly_geometry_evaluation_v1 import (
    evaluate_assembly_geometry_v1,
    summarize_assembly_geometry_v1,
)


def _pose(x: float) -> list[float]:
  return [
      1.0, 0.0, 0.0, x,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
  ]


def _manifest(path, digest, *, child_x: float, accepted: bool = True):
  return {
      "schema_version": "linkcad_assembly_manifest.v1",
      "query_id": "geometry_fixture",
      "anchor_role": "base",
      "components": [
          {
              "role_id": "base", "candidate_id": "base_candidate",
              "source_step_path": path.name,
              "source_step_sha256": digest,
              "world_pose_row_major": _pose(0.0),
          },
          {
              "role_id": "child", "candidate_id": "child_candidate",
              "source_step_path": path.name,
              "source_step_sha256": digest,
              "world_pose_row_major": _pose(child_x),
          },
      ],
      "verification": {
          "global_assembly_conditionally_feasible": accepted,
      },
  }


def test_geometry_metrics_are_zero_for_the_same_assembly_up_to_global_gauge(
    tmp_path,
):
  source = tmp_path / "box.step"
  cq.exporters.export(cq.Workplane("XY").box(4.0, 4.0, 4.0), str(source))
  digest = hashlib.sha256(source.read_bytes()).hexdigest()
  reference = _manifest(source, digest, child_x=10.0)
  prediction = _manifest(source, digest, child_x=10.0)
  prediction["components"][0]["world_pose_row_major"] = _pose(25.0)
  prediction["components"][1]["world_pose_row_major"] = _pose(35.0)

  row = evaluate_assembly_geometry_v1(
      prediction_manifest=prediction,
      reference_manifest=reference,
      dataset_root=tmp_path,
      surface_sample_count_per_component=256,
      intended_exact=True,
  )

  assert row["status"] == "scored"
  assert row["all_part_assignment_correct"] is True
  assert row["part_aware_chamfer_pct_bbox"] < 1e-8
  assert row["part_accuracy_at_1pct"] == 1.0
  assert row["surface_fscore_at_1pct"] == 1.0
  assert row["mean_translation_error_pct_bbox"] < 1e-8
  assert row["mean_rotation_error_deg"] < 1e-8
  assert row["collision_free"] is True
  assert row["intended_exact"] is True


def test_summary_keeps_missing_outputs_in_the_denominator(tmp_path):
  source = tmp_path / "box.step"
  cq.exporters.export(cq.Workplane("XY").box(4.0, 4.0, 4.0), str(source))
  digest = hashlib.sha256(source.read_bytes()).hexdigest()
  reference = _manifest(source, digest, child_x=10.0)
  scored = evaluate_assembly_geometry_v1(
      prediction_manifest=_manifest(source, digest, child_x=10.0),
      reference_manifest=reference,
      dataset_root=tmp_path,
      surface_sample_count_per_component=128,
      intended_exact=True,
  )
  missing_reference = dict(reference)
  missing_reference["query_id"] = "geometry_fixture_missing"
  missing = evaluate_assembly_geometry_v1(
      prediction_manifest=None,
      reference_manifest=missing_reference,
      dataset_root=tmp_path,
      failure_reason="no_prediction",
      intended_exact=False,
  )

  summary = summarize_assembly_geometry_v1([scored, missing])

  assert summary["query_count"] == 2
  assert summary["scoreable_count"] == 1
  assert summary["scoreable_rate"] == 0.5
  assert summary["intended_exact_rate"] == 0.5
  assert summary["failure_aware_surface_fscore_at_1pct_mean"] == 0.5
  assert summary["failure_aware_part_accuracy_at_1pct_mean"] == 0.5
  assert summary["failure_aware_part_aware_chamfer_pct_bbox_mean"] == 50.0
  assert summary["status_counts"] == {"no_prediction": 1, "scored": 1}
