"""Build post-unseal reference assemblies for geometric evaluation only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .linkcad_assembly_export_v1 import export_assembly_package_v1
from .linkcad_global_assembly_execution_v1 import search_global_assembly_query_v1
from .linkcad_kinematic_execution_v2 import execute_kinematic_pair_v2


def target_hypothesis_v1(
    *, query: Mapping[str, Any], target: Mapping[str, Any],
) -> dict[str, Any]:
  """Convert a private target row into an evaluation-only executable program."""

  if str(query["query_id"]) != str(target["query_id"]):
    raise ValueError("LinkCAD reference query identity differs")
  roles = {str(row["role_id"]) for row in query["roles"]}
  candidate_by_role = {
      str(role): str(candidate)
      for role, candidate in target["target_candidate_by_role"].items()
  }
  if set(candidate_by_role) != roles:
    raise ValueError("LinkCAD reference role domain differs")
  query_edges = {
      str(row["edge_id"]): row for row in query["functional_edges"]
  }
  target_edges = {
      str(row["edge_id"]): row for row in target["functional_edge_targets"]
  }
  if set(query_edges) != set(target_edges):
    raise ValueError("LinkCAD reference edge domain differs")
  programs = []
  for edge_id in query_edges:
    edge = query_edges[edge_id]
    row = target_edges[edge_id]
    endpoint_a, endpoint_b = row["endpoint_a"], row["endpoint_b"]
    if (
        str(endpoint_a["candidate_id"])
        != candidate_by_role[str(edge["role_a"])]
        or str(endpoint_b["candidate_id"])
        != candidate_by_role[str(edge["role_b"])]
    ):
      raise ValueError("LinkCAD reference endpoint candidate differs")
    programs.append({
        "edge_id": edge_id,
        "mobility": str(row["target_mobility"]),
        "support_family": str(row["support_family"]),
        "interface_primitive_a": int(endpoint_a["primitive_orbit_ordinal"]),
        "interface_primitive_b": int(endpoint_b["primitive_orbit_ordinal"]),
    })
  return {
      "rank": 0,
      "candidate_by_role": candidate_by_role,
      "edge_programs": programs,
      "reference_only_private_target_replay": True,
  }


def materialize_reference_assembly_v1(
    *, query: Mapping[str, Any], target: Mapping[str, Any],
    dataset_root: str | Path, output_step: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
  """Execute and export one target program after evaluation has been unsealed."""

  hypothesis = target_hypothesis_v1(query=query, target=target)
  candidates = {
      role_id: {str(row["candidate_id"]): row for row in rows}
      for role_id, rows in query["candidate_sets"].items()
  }
  edge_by_id = {
      str(row["edge_id"]): row for row in query["functional_edges"]
  }
  pair_rows = []
  for program in hypothesis["edge_programs"]:
    edge = edge_by_id[program["edge_id"]]
    role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
    row = execute_kinematic_pair_v2(
        query_id=str(query["query_id"]), edge_id=program["edge_id"],
        hypothesis_rank=0,
        candidate_a=candidates[role_a][hypothesis["candidate_by_role"][role_a]],
        candidate_b=candidates[role_b][hypothesis["candidate_by_role"][role_b]],
        primitive_a=program["interface_primitive_a"],
        primitive_b=program["interface_primitive_b"],
        support_family=program["support_family"],
        mobility=program["mobility"], dataset_root=dataset_root,
        enable_axial_seating=True, maximum_accepted_poses=8,
    )
    pair_rows.append(row)
  if any(
      row.get("conditional_kinematic_feasibility_accepted") is not True
      for row in pair_rows
  ):
    return {
        "schema_version": "linkcad_reference_assembly_materialization.v1",
        "query_id": str(query["query_id"]),
        "status": "reference_pair_execution_incomplete",
        "private_targets_opened_by_reference_builder": True,
        "pair_rows": pair_rows,
    }

  global_row = search_global_assembly_query_v1(
      query=query, hypothesis=hypothesis, execution_rows=pair_rows,
      dataset_root=dataset_root,
  )
  if global_row.get("status") != "accepted_global_assembly":
    return {
        "schema_version": "linkcad_reference_assembly_materialization.v1",
        "query_id": str(query["query_id"]),
        "status": "reference_global_execution_incomplete",
        "private_targets_opened_by_reference_builder": True,
        "global_execution": global_row,
        "pair_rows": pair_rows,
    }
  export_receipt = export_assembly_package_v1(
      query=query, hypothesis=hypothesis, global_execution=global_row,
      pair_execution_rows=pair_rows, dataset_root=dataset_root,
      output_step=output_step, output_manifest=output_manifest,
      artifact_role="evaluation_reference",
  )
  return {
      "schema_version": "linkcad_reference_assembly_materialization.v1",
      "query_id": str(query["query_id"]),
      "status": "reference_exported_and_reimport_verified",
      "private_targets_opened_by_reference_builder": True,
      "export_receipt": export_receipt,
      "global_execution": global_row,
      "pair_rows": pair_rows,
  }


__all__ = ["materialize_reference_assembly_v1", "target_hypothesis_v1"]

