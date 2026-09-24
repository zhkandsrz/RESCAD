"""OCP replay authority and supervised exact executor for solver V3."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark_v3_unlabeled_step_view import canonical_sha256 as _safe_sha
from .cadquery_backend import (
    boolean_intersection, load_step_shape, shape_bbox, shape_volume,
    source_face_signature_sha256, transform_shape,
)
from .constraint_manifold_solver_v3 import (
    ABSOLUTE_CHORD_DEFLECTION_MM_V3, ANGULAR_DEFLECTION_RADIANS_V3,
    GEOMETRY_AUTHORITY_SCHEMA_VERSION, OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V3,
    AbsoluteMeshProofV3, CylinderTrimV3, ExactChildReceiptV3,
    GeometryWitnessAuthorityV3, ManifoldFaceGeometryV3, PlaneTrimV3,
    _GEOMETRY_FACTORY_CAPABILITY_V3, _bind_geometry_authority_v3,
    canonical_sha256, file_sha256, issue_exact_child_receipt_v3,
    producer_source_sha256s_v3,
)
from .domain_types import Transform
from .supervised_worker_attestation_v1 import NATIVE_THREAD_ENV


GEOMETRY_WITNESS_SCHEMA_VERSION = "constraint_manifold_geometry_witness.v3"
WORKER_REQUEST_SCHEMA_VERSION = "constraint_manifold_occ_worker_request.v3"
WORKER_TERMINAL_SCHEMA_VERSION = "constraint_manifold_occ_worker_terminal.v3"
IDENTITY_SCHEMA_VERSION = "constraint_manifold_train_identity_authority.v3"
_IDENTITY_TOKEN = object()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _vec(value: Any) -> np.ndarray:
  if hasattr(value, "toTuple"):
    value = value.toTuple()
  elif all(hasattr(value, key) for key in ("X", "Y", "Z")):
    value = (value.X(), value.Y(), value.Z())
  return np.asarray(value, dtype=float).reshape(3)


def _unit(value: Any, *, label: str) -> np.ndarray:
  result = _vec(value)
  norm = float(np.linalg.norm(result))
  if not np.isfinite(result).all() or norm <= 1e-12:
    raise ValueError(f"{label} is degenerate")
  return result / norm


def _world_point(local: np.ndarray, world: Transform) -> np.ndarray:
  return np.asarray(world.rotation) @ local + np.asarray(world.translation)


def _frame_tuple(value: np.ndarray) -> tuple[tuple[float, float, float], ...]:
  return tuple(tuple(float(item) for item in row) for row in value)


def _matrix_tuple(transform: Transform) -> tuple[float, ...]:
  matrix = np.eye(4)
  matrix[:3, :3] = np.asarray(transform.rotation)
  matrix[:3, 3] = np.asarray(transform.translation)
  return tuple(float(value) for value in matrix.reshape(-1))


def _transform(value: Sequence[float]) -> Transform:
  matrix = np.asarray(value, dtype=float).reshape(4, 4)
  return Transform(rotation=matrix[:3, :3], translation=matrix[:3, 3])


def _normal_at_sample(face: Any) -> tuple[np.ndarray, np.ndarray]:
  try:
    u0, u1, v0, v1 = face.uvBounds()
    point = face.positionAt(0.5 * (float(u0) + float(u1)), 0.5 * (float(v0) + float(v1)))
    return _vec(point), _unit(face.normalAt(point), label="oriented face normal")
  except Exception as error:
    raise ValueError("selected face has no stable oriented normal sample") from error


def _absolute_mesh(face: Any) -> tuple[list[np.ndarray], list[tuple[int, int, int]], AbsoluteMeshProofV3]:
  from OCP.BRep import BRep_Tool  # type: ignore
  from OCP.BRepMesh import BRepMesh_IncrementalMesh  # type: ignore
  from OCP.BRepTools import BRepTools  # type: ignore
  import OCP  # type: ignore
  from OCP.TopLoc import TopLoc_Location  # type: ignore

  # Remove any cache created by CadQuery's relative tessellator before the
  # evidence-bearing mesh is built.
  BRepTools.Clean_s(face.wrapped, True)
  mesher = BRepMesh_IncrementalMesh(
      face.wrapped, ABSOLUTE_CHORD_DEFLECTION_MM_V3, False,
      ANGULAR_DEFLECTION_RADIANS_V3, False,
  )
  if not mesher.IsDone():
    raise ValueError("absolute OCC meshing failed")
  parameters = mesher.Parameters()
  if (
      bool(parameters.Relative)
      or abs(float(parameters.Deflection) - ABSOLUTE_CHORD_DEFLECTION_MM_V3) > 1e-15
      or abs(float(parameters.Angle) - ANGULAR_DEFLECTION_RADIANS_V3) > 1e-15
  ):
    raise ValueError("OCC mesher did not honor absolute frozen parameters")
  location = TopLoc_Location()
  triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
  if triangulation is None or triangulation.NbNodes() < 3 or triangulation.NbTriangles() < 1:
    raise ValueError("OCC absolute triangulation is empty")
  transform = location.Transformation()
  vertices = []
  for index in range(1, triangulation.NbNodes() + 1):
    point = triangulation.Node(index).Transformed(transform)
    vertices.append(np.asarray((point.X(), point.Y(), point.Z()), dtype=float))
  indices = []
  for index in range(1, triangulation.NbTriangles() + 1):
    row = triangulation.Triangle(index).Get()
    indices.append(tuple(int(value) - 1 for value in row))
  actual = max(0.0, float(triangulation.Deflection()))
  proof = AbsoluteMeshProofV3(
      requested_linear_deflection_mm=ABSOLUTE_CHORD_DEFLECTION_MM_V3,
      actual_triangulation_deflection_mm=actual,
      deflection_upper_bound_mm=max(ABSOLUTE_CHORD_DEFLECTION_MM_V3, actual),
      relative=False,
      angular_deflection_radians=ANGULAR_DEFLECTION_RADIANS_V3,
      parallel=False, node_count=len(vertices), triangle_count=len(indices),
      occ_version=str(OCP.__version__),
  )
  return vertices, indices, proof


def _wire_evidence(face: Any) -> tuple[tuple[Any, ...], float]:
  wires = tuple(face.Wires())
  if not wires:
    raise ValueError("selected face has no OCC wires")
  length = sum(float(edge.Length()) for wire in wires for edge in wire.Edges())
  if not math.isfinite(length) or length <= 0.0:
    raise ValueError("selected face wire length differs")
  return wires, length


def _plane_face_geometry_v3(face: Any, **common: Any) -> ManifoldFaceGeometryV3:
  from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore

  current_world: Transform = common.pop("current_world")
  plane = BRepAdaptor_Surface(face.wrapped).Plane()
  origin_local = _vec(face.Center())
  _sample, z_local = _normal_at_sample(face)
  x_seed = _vec(plane.XAxis().Direction())
  x_local = _unit(x_seed - float(np.dot(x_seed, z_local)) * z_local, label="plane x")
  y_local = _unit(np.cross(z_local, x_local), label="plane y")
  x_local = _unit(np.cross(y_local, z_local), label="plane reorthogonalized x")
  world_rotation = np.asarray(current_world.rotation)
  frame = np.stack(tuple(_unit(world_rotation @ axis, label="world plane axis")
                         for axis in (x_local, y_local, z_local)), axis=1)
  origin_world = _world_point(origin_local, current_world)
  vertices, indices, mesh = _absolute_mesh(face)
  triangles = []
  for index_row in indices:
    row = []
    for index in index_row:
      point = _world_point(vertices[index], current_world)
      delta = point - origin_world
      row.append((float(np.dot(delta, frame[:, 0])), float(np.dot(delta, frame[:, 1]))))
    values = np.asarray(row, dtype=float)
    if _triangle_area(values) > 1e-12:
      triangles.append(tuple(row))
  wires, boundary = _wire_evidence(face)
  return ManifoldFaceGeometryV3(
      **common, surface_type="plane",
      origin_world_mm=tuple(float(value) for value in origin_world),
      rotation_local_to_world=_frame_tuple(frame),
      plane_trim=PlaneTrimV3(
          triangles_xy=tuple(triangles), exact_surface_area_mm2=float(face.Area()),
          boundary_length_mm=boundary, wire_count=len(wires),
          hole_count=max(0, len(wires) - 1), mesh_proof=mesh,
      ),
  )


def _triangle_area(values: np.ndarray) -> float:
  first, second = values[1] - values[0], values[2] - values[0]
  return 0.5 * abs(float(first[0] * second[1] - first[1] * second[0]))


def _unwrap_triangle(raw_u: Sequence[float]) -> tuple[tuple[float, ...], bool]:
  values = [float(raw_u[0])]
  for raw in raw_u[1:]:
    value = float(raw)
    while value - values[0] > math.pi:
      value -= 2.0 * math.pi
    while value - values[0] < -math.pi:
      value += 2.0 * math.pi
    values.append(value)
  return tuple(values), max(raw_u) - min(raw_u) > math.pi


def _cylinder_face_geometry_v3(face: Any, **common: Any) -> ManifoldFaceGeometryV3:
  from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore

  current_world: Transform = common.pop("current_world")
  cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
  origin_local = _vec(cylinder.Axis().Location())
  z_local = _unit(cylinder.Axis().Direction(), label="cylinder axis")
  x_local = _unit(cylinder.XAxis().Direction(), label="cylinder x")
  y_local = _unit(np.cross(z_local, x_local), label="cylinder y")
  x_local = _unit(np.cross(y_local, z_local), label="cylinder reorthogonalized x")
  sample, normal = _normal_at_sample(face)
  radial = sample - origin_local - float(np.dot(sample - origin_local, z_local)) * z_local
  side_measure = float(np.dot(radial, normal))
  radius = float(cylinder.Radius())
  if abs(side_measure) <= 1e-8 * radius:
    raise ValueError("cylinder side cannot be proved from OCC orientation")
  side = "exterior" if side_measure > 0.0 else "interior"
  vertices, indices, mesh = _absolute_mesh(face)
  triangles = []
  seam_count = 0
  for index_row in indices:
    points = [vertices[index] for index in index_row]
    raw_u = [math.atan2(float(np.dot(point - origin_local, y_local)),
                        float(np.dot(point - origin_local, x_local))) for point in points]
    unwrapped, seam = _unwrap_triangle(raw_u)
    row = tuple((u, float(np.dot(point - origin_local, z_local)))
                for u, point in zip(unwrapped, points, strict=True))
    if _triangle_area(np.asarray(row, dtype=float)) > 1e-12:
      triangles.append(row)
      seam_count += int(seam)
  wires, boundary = _wire_evidence(face)
  world_rotation = np.asarray(current_world.rotation)
  frame = np.stack(tuple(_unit(world_rotation @ axis, label="world cylinder axis")
                         for axis in (x_local, y_local, z_local)), axis=1)
  origin_world = _world_point(origin_local, current_world)
  return ManifoldFaceGeometryV3(
      **common, surface_type="cylinder",
      origin_world_mm=tuple(float(value) for value in origin_world),
      rotation_local_to_world=_frame_tuple(frame),
      cylinder_trim=CylinderTrimV3(
          triangles_uz=tuple(triangles), radius_mm=radius, surface_side=side,
          exact_surface_area_mm2=float(face.Area()), boundary_length_mm=boundary,
          wire_count=len(wires), hole_count=max(0, len(wires) - 1),
          seam_crossing_triangle_count=seam_count, mesh_proof=mesh,
      ),
  )


def extract_occ_manifold_face_v3(
    shape: Any, *, part_slot: str, graph_face_index: int, raw_occ_face_index: int,
    face_signature_sha256: str, current_world: Transform,
) -> ManifoldFaceGeometryV3:
  from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
  from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane  # type: ignore

  if _SHA256_RE.fullmatch(str(face_signature_sha256)) is None:
    raise ValueError("face signature must be lowercase hexadecimal SHA-256")
  faces = tuple(shape.Faces())
  if not 0 <= raw_occ_face_index < len(faces):
    raise ValueError("raw OCC face index is outside shape")
  face = faces[raw_occ_face_index]
  if source_face_signature_sha256(face) != face_signature_sha256:
    raise ValueError("raw OCC face signature replay differs")
  common = {
      "part_slot": part_slot, "graph_face_index": graph_face_index,
      "raw_occ_face_index": raw_occ_face_index,
      "face_signature_sha256": face_signature_sha256,
      "current_world": current_world,
  }
  surface = BRepAdaptor_Surface(face.wrapped).GetType()
  if surface == GeomAbs_Plane:
    return _plane_face_geometry_v3(face, **common)
  if surface == GeomAbs_Cylinder:
    return _cylinder_face_geometry_v3(face, **common)
  raise ValueError("selected OCC face is not plane/cylinder")


def _edges_same(first: Any, second: Any) -> bool:
  try:
    return bool(first.wrapped.IsSame(second.wrapped))
  except Exception:
    return False


def _endpoint_topology(shape: Any, index: int) -> dict[str, Any]:
  from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
  from OCP.GeomAbs import GeomAbs_Plane  # type: ignore

  faces = tuple(shape.Faces())
  selected = faces[index]
  edges = tuple(selected.Edges())
  adjacent = []
  for other_index, face in enumerate(faces):
    if other_index == index:
      continue
    if any(_edges_same(a, b) for a in edges for b in face.Edges()):
      adjacent.append({
          "raw_occ_face_index": other_index,
          "surface_type": (
              "plane" if BRepAdaptor_Surface(face.wrapped).GetType() == GeomAbs_Plane
              else "other"
          ),
          "face_signature_sha256": source_face_signature_sha256(face),
      })
  adjacent.sort(key=lambda row: (row["raw_occ_face_index"], row["face_signature_sha256"]))
  return {
      "raw_occ_face_index": index, "selected_edge_count": len(edges),
      "adjacent_faces": adjacent,
  }


def topology_evidence_v3(
    *, shape_a: Any, shape_b: Any, raw_face_index_a: int,
    raw_face_index_b: int, program_index: int,
) -> dict[str, Any]:
  endpoints = (
      _endpoint_topology(shape_a, raw_face_index_a),
      _endpoint_topology(shape_b, raw_face_index_b),
  )
  # Endpoint-local adjacent planes are useful diagnostics, but without a
  # cross-body correspondence and gap proof they cannot be called proven.
  return {
      "requirement": OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V3.topology_requirement(program_index),
      "required": False,
      "status": "optional_endpoint_local_evidence_only",
      "cross_body_correspondence_proven": False,
      "adjacent_gap_proven": False,
      "endpoint_topology": list(endpoints),
      "step_topology_replayed": True,
  }


def _geometry_from_payload(payload: Mapping[str, Any]) -> ManifoldFaceGeometryV3:
  mesh_payload = payload["plane_trim"]["mesh_proof"] if payload.get("plane_trim") else (
      payload["cylinder_trim"]["mesh_proof"]
  )
  mesh = AbsoluteMeshProofV3(**mesh_payload)
  plane = None
  cylinder = None
  if payload.get("plane_trim") is not None:
    row = payload["plane_trim"]
    plane = PlaneTrimV3(
        triangles_xy=tuple(tuple(tuple(point) for point in triangle)
                           for triangle in row["triangles_xy"]),
        exact_surface_area_mm2=float(row["exact_surface_area_mm2"]),
        boundary_length_mm=float(row["boundary_length_mm"]),
        wire_count=int(row["wire_count"]), hole_count=int(row["hole_count"]),
        mesh_proof=mesh,
    )
  else:
    row = payload["cylinder_trim"]
    cylinder = CylinderTrimV3(
        triangles_uz=tuple(tuple(tuple(point) for point in triangle)
                           for triangle in row["triangles_uz"]),
        radius_mm=float(row["radius_mm"]), surface_side=str(row["surface_side"]),
        exact_surface_area_mm2=float(row["exact_surface_area_mm2"]),
        boundary_length_mm=float(row["boundary_length_mm"]),
        wire_count=int(row["wire_count"]), hole_count=int(row["hole_count"]),
        seam_crossing_triangle_count=int(row["seam_crossing_triangle_count"]),
        mesh_proof=mesh,
    )
  return ManifoldFaceGeometryV3(
      part_slot=str(payload["part_slot"]),
      graph_face_index=int(payload["graph_face_index"]),
      raw_occ_face_index=int(payload["raw_occ_face_index"]),
      face_signature_sha256=str(payload["face_signature_sha256"]),
      surface_type=str(payload["surface_type"]),
      origin_world_mm=tuple(float(value) for value in payload["origin_world_mm"]),
      rotation_local_to_world=tuple(tuple(float(value) for value in row)
                                    for row in payload["rotation_local_to_world"]),
      plane_trim=plane, cylinder_trim=cylinder,
  )


def materialize_geometry_witness_v3(
    path: str | Path, *, program_index: int,
    step_paths: Sequence[str | Path], step_sha256s: Sequence[str],
    graph_face_indices: Sequence[int], raw_face_indices: Sequence[int],
    face_signatures: Sequence[str], current_worlds: Sequence[Transform],
) -> tuple[ManifoldFaceGeometryV3, ManifoldFaceGeometryV3]:
  if not all(len(rows) == 2 for rows in (
      step_paths, step_sha256s, graph_face_indices, raw_face_indices,
      face_signatures, current_worlds,
  )):
    raise ValueError("geometry witness endpoint cardinality differs")
  shapes = []
  files = []
  geometries = []
  for role, step_path, step_sha, graph_index, raw_index, signature, world in zip(
      ("a", "b"), step_paths, step_sha256s, graph_face_indices, raw_face_indices,
      face_signatures, current_worlds, strict=True,
  ):
    resolved = Path(step_path).resolve(strict=True)
    if _SHA256_RE.fullmatch(str(step_sha)) is None or file_sha256(resolved) != step_sha:
      raise ValueError("geometry witness STEP bytes differ")
    shape = load_step_shape(resolved)
    shapes.append(shape)
    files.append({"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": step_sha})
    geometries.append(extract_occ_manifold_face_v3(
        shape, part_slot=role, graph_face_index=int(graph_index),
        raw_occ_face_index=int(raw_index), face_signature_sha256=str(signature),
        current_world=world,
    ))
  unsigned = {
      "schema_version": GEOMETRY_WITNESS_SCHEMA_VERSION,
      "program_index": int(program_index), "step_files": files,
      "graph_face_indices": [int(value) for value in graph_face_indices],
      "raw_occ_face_indices": [int(value) for value in raw_face_indices],
      "face_signatures": list(face_signatures),
      "current_world_row_major": [_matrix_tuple(value) for value in current_worlds],
      "face_geometry": [row.payload() for row in geometries],
      "topology_evidence": topology_evidence_v3(
          shape_a=shapes[0], shape_b=shapes[1],
          raw_face_index_a=int(raw_face_indices[0]), raw_face_index_b=int(raw_face_indices[1]),
          program_index=int(program_index),
      ),
      "producer_source_sha256s": producer_source_sha256s_v3(),
      "source_pose_or_mate_fields_absent": True,
      "final_test_touched": False,
  }
  payload = {**unsigned, "witness_payload_sha256": canonical_sha256(unsigned)}
  target = Path(path)
  if target.exists():
    raise ValueError("geometry witness path already exists")
  target.parent.mkdir(parents=True, exist_ok=True)
  temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
  temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
  os.replace(temporary, target)
  return tuple(geometries)  # type: ignore[return-value]


def _replay_geometry_witness(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  unsigned = dict(payload)
  observed = unsigned.pop("witness_payload_sha256", None)
  if (
      unsigned.get("schema_version") != GEOMETRY_WITNESS_SCHEMA_VERSION
      or observed != canonical_sha256(unsigned)
      or unsigned.get("final_test_touched") is not False
      or unsigned.get("source_pose_or_mate_fields_absent") is not True
  ):
    raise ValueError("geometry witness commitment differs")
  if unsigned["producer_source_sha256s"] != producer_source_sha256s_v3():
    raise ValueError("geometry witness producer source bytes changed")
  shapes = []
  geometries = []
  for role, file_row, graph_index, raw_index, signature, world_row in zip(
      ("a", "b"), unsigned["step_files"], unsigned["graph_face_indices"],
      unsigned["raw_occ_face_indices"], unsigned["face_signatures"],
      unsigned["current_world_row_major"], strict=True,
  ):
    step_path = Path(file_row["path"]).resolve(strict=True)
    if (
        step_path.stat().st_size != int(file_row["bytes"])
        or file_sha256(step_path) != file_row["sha256"]
    ):
      raise ValueError("geometry witness STEP replay differs")
    shape = load_step_shape(step_path)
    shapes.append(shape)
    geometries.append(extract_occ_manifold_face_v3(
        shape, part_slot=role, graph_face_index=int(graph_index),
        raw_occ_face_index=int(raw_index), face_signature_sha256=str(signature),
        current_world=_transform(world_row),
    ))
  geometry_payloads = [row.payload() for row in geometries]
  if geometry_payloads != unsigned["face_geometry"]:
    raise ValueError("geometry witness trim/mesh replay differs")
  topology = topology_evidence_v3(
      shape_a=shapes[0], shape_b=shapes[1],
      raw_face_index_a=int(unsigned["raw_occ_face_indices"][0]),
      raw_face_index_b=int(unsigned["raw_occ_face_indices"][1]),
      program_index=int(unsigned["program_index"]),
  )
  if topology != unsigned["topology_evidence"]:
    raise ValueError("geometry witness topology replay differs")
  witness_binding = {
      "path": str(path.resolve()), "bytes": path.stat().st_size,
      "sha256": file_sha256(path),
  }
  return {
      "schema_version": GEOMETRY_AUTHORITY_SCHEMA_VERSION,
      "program_index": int(unsigned["program_index"]),
      "witness_file": witness_binding,
      "step_files": unsigned["step_files"],
      "face_geometry": geometry_payloads,
      "topology_evidence": topology,
      "producer_source_sha256s": unsigned["producer_source_sha256s"],
      "oracle_inputs_absent": True,
  }


def load_geometry_authority_v3(path: str | Path) -> GeometryWitnessAuthorityV3:
  witness_path = Path(path).resolve(strict=True)
  unsigned = _replay_geometry_witness(witness_path)
  payload = {**unsigned, "authority_payload_sha256": canonical_sha256(unsigned)}
  return _bind_geometry_authority_v3(
      payload, replay=lambda: _replay_geometry_witness(witness_path),
      _factory_capability=_GEOMETRY_FACTORY_CAPABILITY_V3,
  )


def geometry_pair_from_authority_v3(
    authority: GeometryWitnessAuthorityV3,
) -> tuple[ManifoldFaceGeometryV3, ManifoldFaceGeometryV3]:
  rows = authority.payload()["face_geometry"]
  if len(rows) != 2:
    raise ValueError("geometry authority endpoint cardinality differs")
  return _geometry_from_payload(rows[0]), _geometry_from_payload(rows[1])


def _transformed_bbox(shape: Any, transform: Transform) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  low, high = shape_bbox(shape)
  corners = np.asarray([
      (x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1])
      for z in (low[2], high[2])
  ], dtype=float)
  world = corners @ np.asarray(transform.rotation).T + np.asarray(transform.translation)
  return np.min(world, axis=0), np.max(world, axis=0), world


def _occ_manifold_child_v3(
    *, operation: str, step_path_a: str, step_path_b: str,
    step_sha256_a: str, step_sha256_b: str,
    raw_face_index_a: int, raw_face_index_b: int,
    face_signature_a: str, face_signature_b: str,
    world_a_row_major: Sequence[float], child_world_row_major: Sequence[float],
    plane_frame_a: Sequence[Sequence[float]], plane_origin_a: Sequence[float],
    call_nonce: str,
) -> dict[str, Any]:
  try:
    if file_sha256(step_path_a) != step_sha256_a or file_sha256(step_path_b) != step_sha256_b:
      raise ValueError("receipt-bound STEP bytes changed")
    source_a, source_b = load_step_shape(step_path_a), load_step_shape(step_path_b)
    faces_a, faces_b = tuple(source_a.Faces()), tuple(source_b.Faces())
    face_a, face_b = faces_a[raw_face_index_a], faces_b[raw_face_index_b]
    if source_face_signature_sha256(face_a) != face_signature_a or (
        source_face_signature_sha256(face_b) != face_signature_b
    ):
      raise ValueError("reloaded STEP face signature differs")
    if operation != "candidate_exact_bundle":
      raise ValueError("unknown OCC child operation")
    world_a, world_b = _transform(world_a_row_major), _transform(child_world_row_major)
    moved_face_a = transform_shape(face_a, world_a)
    moved_face_b = transform_shape(face_b, world_b)
    distance = float(moved_face_a.distance(moved_face_b))
    low_a, high_a, corners_a = _transformed_bbox(source_a, world_a)
    low_b, high_b, corners_b = _transformed_bbox(source_b, world_b)
    overlap = np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
    aabb_upper = 0.0 if np.any(overlap <= 0.0) else float(np.prod(overlap))
    if aabb_upper <= OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V3.thresholds.whole_common_volume_mm3:
      volume = 0.0
    else:
      volume = max(0.0, float(shape_volume(boolean_intersection(
          transform_shape(source_a, world_a), transform_shape(source_b, world_b),
      ))))
    frame = np.asarray(plane_frame_a, dtype=float).reshape(3, 3)
    origin = np.asarray(plane_origin_a, dtype=float).reshape(3)
    local_a = (corners_a - origin) @ frame
    local_b = (corners_b - origin) @ frame
    proj_a = np.min(local_a[:, :2], axis=0), np.max(local_a[:, :2], axis=0)
    proj_b = np.min(local_b[:, :2], axis=0), np.max(local_b[:, :2], axis=0)
    epsilon = 1e-4
    offsets = (
        (float(proj_a[1][0] - proj_b[0][0] + epsilon), 0.0),
        (float(proj_a[0][0] - proj_b[1][0] - epsilon), 0.0),
        (0.0, float(proj_a[1][1] - proj_b[0][1] + epsilon)),
        (0.0, float(proj_a[0][1] - proj_b[1][1] - epsilon)),
    )
    return {
        "status": "ok", "distance_mm": distance, "common_volume_mm3": volume,
        "aabb_intersection_upper_mm3": aabb_upper,
        "projected_aabb_separation_offsets_xy_mm": offsets,
        "call_nonce": call_nonce, "operation": operation, "child_pid": os.getpid(),
    }
  except BaseException as error:
    return {
        "status": "kernel_error", "error": f"{type(error).__name__}:{error}",
        "call_nonce": call_nonce, "operation": operation, "child_pid": os.getpid(),
    }


class OccExactManifoldExecutorV3:
  __slots__ = (
      "step_path_a", "step_path_b", "step_sha256_a", "step_sha256_b",
      "raw_face_index_a", "raw_face_index_b", "face_signature_a", "face_signature_b",
      "world_a_row_major", "plane_frame_a", "plane_origin_a", "_seen_nonces",
  )

  def __init__(
      self, *, step_path_a: str, step_path_b: str,
      raw_face_index_a: int, raw_face_index_b: int,
      face_signature_a: str, face_signature_b: str,
      step_sha256_a: str, step_sha256_b: str,
      world_a: Transform, face_geometry_a: ManifoldFaceGeometryV3,
  ) -> None:
    self.step_path_a = str(Path(step_path_a).resolve(strict=True))
    self.step_path_b = str(Path(step_path_b).resolve(strict=True))
    self.step_sha256_a, self.step_sha256_b = step_sha256_a, step_sha256_b
    self.raw_face_index_a, self.raw_face_index_b = raw_face_index_a, raw_face_index_b
    self.face_signature_a, self.face_signature_b = face_signature_a, face_signature_b
    for value in (step_sha256_a, step_sha256_b, face_signature_a, face_signature_b):
      if _SHA256_RE.fullmatch(value) is None:
        raise ValueError("OCC exact executor hash binding differs")
    if file_sha256(self.step_path_a) != step_sha256_a or file_sha256(self.step_path_b) != step_sha256_b:
      raise ValueError("OCC exact executor STEP bytes differ")
    self.world_a_row_major = _matrix_tuple(world_a)
    self.plane_frame_a = face_geometry_a.rotation_local_to_world
    self.plane_origin_a = face_geometry_a.origin_world_mm
    self._seen_nonces: set[str] = set()

  def run_child(
      self, candidate: Any, *, operation: str, call_nonce: str,
      absolute_deadline: float,
  ) -> ExactChildReceiptV3:
    if call_nonce in self._seen_nonces:
      raise ValueError("exact child nonce was reused")
    self._seen_nonces.add(call_nonce)
    arguments = {
        "operation": operation, "step_path_a": self.step_path_a,
        "step_path_b": self.step_path_b, "step_sha256_a": self.step_sha256_a,
        "step_sha256_b": self.step_sha256_b,
        "raw_face_index_a": self.raw_face_index_a,
        "raw_face_index_b": self.raw_face_index_b,
        "face_signature_a": self.face_signature_a,
        "face_signature_b": self.face_signature_b,
        "world_a_row_major": self.world_a_row_major,
        "child_world_row_major": candidate.child_world_row_major,
        "plane_frame_a": self.plane_frame_a, "plane_origin_a": self.plane_origin_a,
        "call_nonce": call_nonce,
    }
    unsigned = {"schema_version": WORKER_REQUEST_SCHEMA_VERSION, "arguments": arguments}
    request = {**unsigned, "request_payload_sha256": canonical_sha256(unsigned)}
    worker = Path(__file__).resolve().parent / "tools" / "constraint_manifold_occ_worker_v3.py"
    environment = os.environ.copy()
    environment.update(NATIVE_THREAD_ENV)
    parent = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = parent + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["PATH"] = str(Path(sys.prefix).resolve() / "Library" / "bin") + os.pathsep + environment.get("PATH", "")
    process = None
    stdout = stderr = ""
    try:
      remaining = absolute_deadline - time.monotonic()
      if remaining <= 0.0:
        return issue_exact_child_receipt_v3(
            call_nonce=call_nonce, operation=operation, terminal_status="timeout",
            child_started=False, result={"error": "deadline_expired_before_spawn"},
            issuer="parent_observer",
        )
      try:
        process = subprocess.Popen(
            [sys.executable, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=environment,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                if os.name == "nt" else 0
            ),
        )
      except BaseException as error:
        return issue_exact_child_receipt_v3(
            call_nonce=call_nonce, operation=operation, terminal_status="kernel_error",
            child_started=False,
            result={"error": f"spawn_failed:{type(error).__name__}:{error}"},
            issuer="parent_observer",
        )
      try:
        stdout, stderr = process.communicate(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            timeout=max(0.0, absolute_deadline - time.monotonic()),
        )
      except BaseException as communicate_error:
        kill_error = wait_error = None
        try:
          process.kill()
        except BaseException as error:
          kill_error = f"{type(error).__name__}:{error}"
        try:
          stdout, stderr = process.communicate(timeout=2.0)
        except BaseException as error:
          wait_error = f"{type(error).__name__}:{error}"
        status = "timeout" if isinstance(communicate_error, subprocess.TimeoutExpired) else "kernel_error"
        return issue_exact_child_receipt_v3(
            call_nonce=call_nonce, operation=operation, terminal_status=status,
            child_started=True, issuer="parent_observer",
            result={
                "error": f"communicate_failed:{type(communicate_error).__name__}:{communicate_error}",
                "kill_error": kill_error, "wait_error": wait_error,
                "child_pid": process.pid,
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            },
        )
      try:
        terminal = json.loads(stdout.strip().splitlines()[-1])
        terminal_unsigned = dict(terminal)
        terminal_sha = terminal_unsigned.pop("terminal_payload_sha256", None)
        valid = (
            terminal.get("schema_version") == WORKER_TERMINAL_SCHEMA_VERSION
            and terminal.get("request_payload_sha256") == request["request_payload_sha256"]
            and terminal_sha == canonical_sha256(terminal_unsigned)
        )
      except BaseException:
        terminal, valid, terminal_sha = {}, False, None
      if not valid:
        result = {
            "error": f"worker_terminal_invalid:exit={process.returncode}",
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        }
        status = "kernel_error"
      else:
        result = dict(terminal["result"])
        status = str(result.pop("status", "kernel_error"))
        result["worker_terminal_payload_sha256"] = terminal_sha
      echoed_nonce = result.pop("call_nonce", None)
      echoed_operation = result.pop("operation", None)
      if echoed_nonce != call_nonce or echoed_operation != operation:
        status = "kernel_error"
        result = {"error": "child_nonce_or_operation_echo_mismatch", "child_pid": process.pid}
      if status not in {"ok", "timeout", "kernel_error"}:
        status = "kernel_error"
      return issue_exact_child_receipt_v3(
          call_nonce=call_nonce, operation=operation, terminal_status=status,
          child_started=True, result=result, issuer="child_process_echo",
      )
    finally:
      # Cleanup is best-effort and never allowed to erase a terminal receipt.
      if process is not None and process.poll() is None:
        try:
          process.kill()
        except BaseException:
          pass
        try:
          process.wait(timeout=2.0)
        except BaseException:
          pass


__all__ = [
    "GEOMETRY_WITNESS_SCHEMA_VERSION", "IDENTITY_SCHEMA_VERSION",
    "OccExactManifoldExecutorV3", "extract_occ_manifold_face_v3",
    "geometry_pair_from_authority_v3", "load_geometry_authority_v3",
    "materialize_geometry_witness_v3", "topology_evidence_v3",
]
