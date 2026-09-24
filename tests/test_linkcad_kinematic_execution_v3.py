from __future__ import annotations

import hashlib

import cadquery as cq
import numpy as np

from neurocad.linkcad_kinematic_execution_v3 import execute_kinematic_pair_v3
from neurocad.linkcad_port_contract_v1 import describe_ports_v1
from neurocad.linkcad_primitive_graph_v2 import extract_primitive_graph_v2


def _candidate(path, ordinal):
  return {
      "candidate_id": f"part_{ordinal}",
      "step_path": path.name,
      "step_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  }


def _port(shape, *, geometry, size):
  graph = extract_primitive_graph_v2(shape)
  contracts = describe_ports_v1(graph)
  preferred = [
      ordinal for ordinal, contract in enumerate(describe_ports_v1(graph))
      if contract["primitive_kind"] == "face"
      and contract["geometry_type"] == geometry
      and contract["size_class"] == size
  ]
  if preferred:
    return preferred[0]
  return next(
      ordinal for ordinal, contract in enumerate(contracts)
      if contract["primitive_kind"] == "face"
      and contract["geometry_type"] == geometry
  )


def test_axial_executor_returns_contact_not_a_detached_clearance_pose(tmp_path):
  ring = cq.Workplane("XY").circle(7.0).circle(3.2).extrude(6.0)
  shaft = (
      cq.Workplane("XY").circle(3.0).extrude(16.0)
      .faces(">Z").workplane().circle(5.0).extrude(2.0)
  )
  paths = (tmp_path / "ring.step", tmp_path / "shaft.step")
  for path, workplane in zip(paths, (ring, shaft), strict=True):
    cq.exporters.export(workplane, str(path))
  shapes = [cq.importers.importStep(str(path)).val() for path in paths]
  result = execute_kinematic_pair_v3(
      query_id="axial_contact", edge_id="edge", hypothesis_rank=0,
      candidate_a=_candidate(paths[0], 0),
      candidate_b=_candidate(paths[1], 1),
      primitive_a=_port(shapes[0], geometry="cylinder", size="smallest"),
      primitive_b=_port(shapes[1], geometry="cylinder", size="smallest"),
      support_family="axial_support", mobility="revolute",
      dataset_root=tmp_path,
  )
  observation = result["selected_observation"]
  assert result["status"] == "accepted_contact_complete"
  assert observation["whole_solid_minimum_distance_mm"] <= 0.1
  assert observation["whole_solid_common_volume_mm3"] <= 1e-7
  assert observation["contact_complete"] is True


def test_planar_executor_aligns_trimmed_face_centres(tmp_path):
  plate = cq.Workplane("XY").box(24.0, 18.0, 3.0).faces(">Z").workplane().hole(6.0)
  ring = cq.Workplane("XY").circle(8.0).circle(3.2).extrude(5.0)
  paths = (tmp_path / "plate.step", tmp_path / "ring.step")
  for path, workplane in zip(paths, (plate, ring), strict=True):
    cq.exporters.export(workplane, str(path))
  shapes = [cq.importers.importStep(str(path)).val() for path in paths]
  primitives = [
      _port(shape, geometry="plane", size="largest") for shape in shapes
  ]
  result = execute_kinematic_pair_v3(
      query_id="planar_center", edge_id="edge", hypothesis_rank=0,
      candidate_a=_candidate(paths[0], 0),
      candidate_b=_candidate(paths[1], 1),
      primitive_a=primitives[0], primitive_b=primitives[1],
      support_family="planar_support", mobility="fixed",
      dataset_root=tmp_path,
  )
  observation = result["selected_observation"]
  matrix = np.asarray(observation["child_world_row_major"]).reshape(4, 4)
  left_face = list(shapes[0].Faces())[observation["raw_primitive_a"]]
  right_face = list(shapes[1].Faces())[observation["raw_primitive_b"]]
  left_center = np.asarray(left_face.Center().toTuple())
  right_center = np.asarray(right_face.Center().toTuple())
  moved_right_center = matrix[:3, :3] @ right_center + matrix[:3, 3]
  assert np.linalg.norm(left_center - moved_right_center) <= 1e-6
  assert observation["whole_solid_minimum_distance_mm"] <= 0.1
  assert observation["whole_solid_common_volume_mm3"] <= 1e-7
