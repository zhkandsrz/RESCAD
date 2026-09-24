from __future__ import annotations

import hashlib

import cadquery as cq

from neurocad.linkcad_kinematic_execution_v2 import execute_kinematic_pair_v2
from neurocad.linkcad_port_contract_v1 import describe_ports_v1
from neurocad.linkcad_primitive_graph_v2 import extract_primitive_graph_v2


def test_planar_primitive_pair_finds_collision_free_contact_pose(tmp_path):
  paths = (tmp_path / "a.step", tmp_path / "b.step")
  for path in paths:
    cq.exporters.export(cq.Workplane("XY").box(2.0, 3.0, 4.0), str(path))
  shape = cq.importers.importStep(str(paths[0])).val()
  graph = extract_primitive_graph_v2(shape)
  plane_primitive = next(
      ordinal
      for ordinal, (kind, members) in enumerate(zip(
          graph.primitive_kinds, graph.primitive_members, strict=True,
      ))
      if kind == "face"
      and str(list(shape.Faces())[members[0]].geomType()).upper() == "PLANE"
  )
  candidates = [
      {
          "candidate_id": f"part_{index}",
          "step_path": path.name,
          "step_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      }
      for index, path in enumerate(paths)
  ]
  result = execute_kinematic_pair_v2(
      query_id="fixture", edge_id="edge", hypothesis_rank=0,
      candidate_a=candidates[0], candidate_b=candidates[1],
      primitive_a=plane_primitive, primitive_b=plane_primitive,
      support_family="planar_support", mobility="fixed",
      dataset_root=tmp_path,
  )
  assert result["status"] == "accepted"
  assert result["conditional_kinematic_feasibility_accepted"] is True
  assert result["selected_observation"]["whole_solid_common_volume_mm3"] <= 1e-7


def test_axial_support_searches_finite_endpoint_seating_offsets(tmp_path):
  ring = cq.Workplane("XY").circle(7.0).circle(3.2).extrude(6.0)
  shaft = (
      cq.Workplane("XY").circle(3.0).extrude(16.0)
      .faces(">Z").workplane().circle(5.0).extrude(2.0)
  )
  paths = (tmp_path / "ring.step", tmp_path / "shaft.step")
  for path, workplane in zip(paths, (ring, shaft), strict=True):
    cq.exporters.export(workplane, str(path))
  shapes = [cq.importers.importStep(str(path)).val() for path in paths]
  graphs = [extract_primitive_graph_v2(shape) for shape in shapes]

  def bore_or_shaft(graph):
    return next(
        ordinal for ordinal, contract in enumerate(describe_ports_v1(graph))
        if contract["primitive_kind"] == "face"
        and contract["geometry_type"] == "cylinder"
        and contract["size_class"] == "smallest"
    )

  primitives = [bore_or_shaft(graph) for graph in graphs]
  candidates = [{
      "candidate_id": f"part_{index}",
      "step_path": path.name,
      "step_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  } for index, path in enumerate(paths)]
  result = execute_kinematic_pair_v2(
      query_id="axial_fixture", edge_id="edge", hypothesis_rank=0,
      candidate_a=candidates[0], candidate_b=candidates[1],
      primitive_a=primitives[0], primitive_b=primitives[1],
      support_family="axial_support", mobility="revolute",
      dataset_root=tmp_path,
      enable_axial_seating=True,
  )
  assert result["conditional_kinematic_feasibility_accepted"] is True
  assert abs(result["selected_observation"]["axial_offset_mm"]) > 1e-6
  assert result["selected_observation"]["whole_solid_common_volume_mm3"] <= 1e-7


def test_pair_executor_can_retain_multiple_public_feasible_poses(tmp_path):
  paths = (tmp_path / "a.step", tmp_path / "b.step")
  for path in paths:
    cq.exporters.export(cq.Workplane("XY").box(2.0, 3.0, 4.0), str(path))
  shape = cq.importers.importStep(str(paths[0])).val()
  graph = extract_primitive_graph_v2(shape)
  plane_primitive = next(
      ordinal
      for ordinal, (kind, members) in enumerate(zip(
          graph.primitive_kinds, graph.primitive_members, strict=True,
      ))
      if kind == "face"
      and str(list(shape.Faces())[members[0]].geomType()).upper() == "PLANE"
  )
  candidates = [{
      "candidate_id": f"part_{index}",
      "step_path": path.name,
      "step_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  } for index, path in enumerate(paths)]
  result = execute_kinematic_pair_v2(
      query_id="multi_pose_fixture", edge_id="edge", hypothesis_rank=0,
      candidate_a=candidates[0], candidate_b=candidates[1],
      primitive_a=plane_primitive, primitive_b=plane_primitive,
      support_family="planar_support", mobility="fixed",
      dataset_root=tmp_path, maximum_accepted_poses=3,
  )
  assert result["accepted_pose_count"] == 3
  assert len(result["accepted_observations"]) == 3
