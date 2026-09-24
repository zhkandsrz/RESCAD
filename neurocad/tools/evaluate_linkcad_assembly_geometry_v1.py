"""Score exported assembly packages against post-unseal reference packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from neurocad.linkcad_assembly_geometry_evaluation_v1 import (
    evaluate_assembly_geometry_v1,
    summarize_assembly_geometry_v1,
)


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _prediction_verdict(
    manifest: Mapping[str, Any] | None, target: Mapping[str, Any],
) -> tuple[bool, bool, bool, bool]:
  if manifest is None:
    return False, False, False, False
  selected = {
      str(row["role_id"]): str(row["candidate_id"])
      for row in manifest["components"]
  }
  expected = {
      str(role): str(candidate)
      for role, candidate in target["target_candidate_by_role"].items()
  }
  part_correct = selected == expected
  predicted_edges = {
      str(row["edge_id"]): row for row in manifest["connections"]
  }
  target_edges = {
      str(row["edge_id"]): row
      for row in target["functional_edge_targets"]
  }
  program_correct = set(predicted_edges) == set(target_edges)
  if program_correct:
    for edge_id, expected_edge in target_edges.items():
      predicted = predicted_edges[edge_id]
      if (
          str(predicted.get("predicted_mobility"))
          != str(expected_edge["target_mobility"])
          or str(predicted.get("support_family"))
          != str(expected_edge["support_family"])
          or int(predicted.get("interface_primitive_a", -1))
          != int(expected_edge["endpoint_a"]["primitive_orbit_ordinal"])
          or int(predicted.get("interface_primitive_b", -1))
          != int(expected_edge["endpoint_b"]["primitive_orbit_ordinal"])
      ):
        program_correct = False
        break
  verification = manifest.get("verification")
  if isinstance(verification, Mapping) and "raw_collision_free" in verification:
    collision_free = verification.get("raw_collision_free") is True
  else:
    collision_free = (
        isinstance(verification, Mapping)
        and verification.get("global_assembly_conditionally_feasible") is True
    )
  executable = (
      isinstance(verification, Mapping)
      and verification.get("global_assembly_conditionally_feasible") is True
  )
  return part_correct, program_correct, collision_free, executable


def evaluate_exported_method_v1(
    *, method_id: str, prediction_root: Path, reference_root: Path,
    private_targets: Mapping[str, Any], dataset_root: Path,
    surface_sample_count_per_component: int,
) -> dict[str, Any]:
  targets = {
      str(row["query_id"]): row for row in private_targets["targets"]
  }
  reference_batch = _read(reference_root / "reference_assembly_batch.json")
  reference_status = {
      str(row["query_id"]): str(row["status"])
      for row in reference_batch["rows"]
  }
  rows = []
  for query_id in sorted(targets):
    prediction_path = prediction_root / query_id / "assembly.manifest.json"
    reference_path = reference_root / query_id / "reference.manifest.json"
    prediction = _read(prediction_path) if prediction_path.is_file() else None
    reference = _read(reference_path) if reference_path.is_file() else None
    part_correct, program_correct, collision_free, executable = _prediction_verdict(
        prediction, targets[query_id]
    )
    intended_exact = part_correct and program_correct and executable
    if reference is None:
      row = {
          "schema_version": "linkcad_assembly_geometry_row.v1",
          "query_id": query_id,
          "status": reference_status.get(query_id, "reference_unavailable"),
          "scoreable": False,
          "all_part_assignment_correct": part_correct,
          "collision_free": collision_free,
          "intended_exact": intended_exact,
          "part_aware_chamfer_pct_bbox": None,
          "part_accuracy_at_1pct": None,
          "surface_fscore_at_1pct": None,
          "mean_translation_error_pct_bbox": None,
          "mean_rotation_error_deg": None,
      }
    else:
      row = evaluate_assembly_geometry_v1(
          prediction_manifest=prediction, reference_manifest=reference,
          dataset_root=dataset_root,
          surface_sample_count_per_component=surface_sample_count_per_component,
          failure_reason="method_output_unavailable",
          intended_exact=intended_exact,
      )
      row["all_part_assignment_correct"] = part_correct
      row["collision_free"] = collision_free
      row["intended_exact"] = intended_exact
    row["method_id"] = method_id
    row["complete_program_correct"] = program_correct
    rows.append(row)
  summary = summarize_assembly_geometry_v1(rows)
  summary["method_id"] = method_id
  summary["reference_evaluable_count"] = sum(
      reference_status.get(query_id)
      == "reference_exported_and_reimport_verified"
      for query_id in targets
  )
  summary["complete_program_accuracy"] = sum(
      row["complete_program_correct"] is True for row in rows
  ) / len(rows)
  return {
      "schema_version": "linkcad_assembly_geometry_evaluation.v1",
      "method_id": method_id,
      "scope": "all_queries_with_failures_retained",
      "surface_sample_count_per_component": surface_sample_count_per_component,
      "summary": summary,
      "rows": rows,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--method-id", required=True)
  parser.add_argument("--prediction-root", type=Path, required=True)
  parser.add_argument("--reference-root", type=Path, required=True)
  parser.add_argument("--private-targets", type=Path, required=True)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--surface-samples", type=int, default=1024)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = evaluate_exported_method_v1(
      method_id=args.method_id,
      prediction_root=args.prediction_root.resolve(),
      reference_root=args.reference_root.resolve(),
      private_targets=_read(args.private_targets),
      dataset_root=args.dataset_root.resolve(),
      surface_sample_count_per_component=args.surface_samples,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True), encoding="utf-8",
  )
  print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
  main()
