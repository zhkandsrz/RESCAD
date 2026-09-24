"""Development-only proof for planar closed-circle chordal trim domains.

The supported domain is deliberately narrow: one root-occurrence planar OBJ
triangle disk whose every boundary loop is a monotone, one-turn chordal
tessellation of one unique closed OCC circle, and one STEP face whose every
wire is exactly one such circle.  No sampling density or chord-sag threshold
authorizes the result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neurocad.cadquery_backend import source_face_signature_sha256
from neurocad.source_obj_analytic_support_v1 import (
    capture_source_obj_face_v1,
    replay_source_obj_face_topology_observation_v1,
)


SCHEMA_VERSION = "planar_closed_circle_chordal_trim_certificate.v1"
ALGORITHM_REVISION = "planar_closed_circle_chordal_trim.v1"
LOCKED_BOUNDARY_DISTANCE_MM = 0.2
LOCKED_AREA_RELATIVE_ERROR = 0.08
_NUMERIC_GUARD_MM = LOCKED_BOUNDARY_DISTANCE_MM / 2.0
_DIRECTION_GUARD = 1.0e-8
_ANGLE_GUARD = 1.0e-8


def _json_data(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(key): _json_data(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_data(item) for item in value]
  return value


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      _json_data(value),
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")


def _sha(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _vec(value: Any) -> np.ndarray:
  if hasattr(value, "toTuple"):
    value = value.toTuple()
  elif hasattr(value, "X") and callable(value.X):
    value = (value.X(), value.Y(), value.Z())
  result = np.asarray(tuple(float(item) for item in value), dtype=float)
  if result.shape != (3,) or not np.all(np.isfinite(result)):
    raise ValueError("geometry vector differs")
  return result


def _unit(value: np.ndarray, *, label: str) -> np.ndarray:
  length = float(np.linalg.norm(value))
  if not math.isfinite(length) or length <= 0.0:
    raise ValueError(f"{label} is degenerate")
  return value / length


def _source_loops(view: Mapping[str, Any]) -> tuple[list[np.ndarray], float, np.ndarray]:
  if view.get("surface_type") != "PlaneSurfaceType":
    raise ValueError("source face is not Plane")
  vertices = np.asarray(view.get("vertices_mm"), dtype=float)
  raw_triangles = view.get("triangles")
  if (
      vertices.ndim != 2
      or vertices.shape[1:] != (3,)
      or not np.all(np.isfinite(vertices))
      or not isinstance(raw_triangles, Sequence)
      or not raw_triangles
  ):
    raise ValueError("source topology differs")
  triangles = []
  for raw in raw_triangles:
    if not isinstance(raw, Sequence) or len(raw) != 3:
      raise ValueError("source triangle topology differs")
    indices = tuple(int(use["vertex_index"]) for use in raw)
    if len(set(indices)) != 3 or any(
        index < 0 or index >= len(vertices) for index in indices
    ):
      raise ValueError("source triangle topology differs")
    triangles.append(indices)
  if len(set(triangles)) != len(triangles):
    raise ValueError("source triangle topology is duplicated")

  crosses = [
      np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
      for a, b, c in triangles
  ]
  areas2 = np.asarray([np.linalg.norm(value) for value in crosses], dtype=float)
  if np.any(~np.isfinite(areas2)) or np.any(areas2 <= 0.0):
    raise ValueError("source triangle area differs")
  normal = _unit(crosses[0], label="source plane normal")
  if any(float(np.dot(value, normal)) <= 0.0 for value in crosses):
    raise ValueError("source triangle orientations disagree")
  origin = vertices[triangles[0][0]]
  plane_residual = np.abs((vertices - origin) @ normal)
  if float(np.max(plane_residual)) > _NUMERIC_GUARD_MM:
    raise ValueError("source vertices are not on one guarded plane")

  edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
  triangle_neighbours: dict[int, set[int]] = defaultdict(set)
  edge_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
  for triangle_index, triangle in enumerate(triangles):
    for start, end in zip(triangle, triangle[1:] + triangle[:1]):
      key = tuple(sorted((start, end)))
      edge_uses[key].append((start, end))
      edge_triangles[key].append(triangle_index)
  boundary: list[tuple[int, int]] = []
  for key, uses in edge_uses.items():
    if len(uses) == 1:
      boundary.append(uses[0])
    elif len(uses) == 2:
      if uses[0] != (uses[1][1], uses[1][0]):
        raise ValueError("source internal triangle orientations disagree")
      left, right = edge_triangles[key]
      triangle_neighbours[left].add(right)
      triangle_neighbours[right].add(left)
    else:
      raise ValueError("source triangle boundary is non-manifold")
  visited = {0}
  stack = [0]
  while stack:
    current = stack.pop()
    for neighbour in triangle_neighbours[current] - visited:
      visited.add(neighbour)
      stack.append(neighbour)
  if len(visited) != len(triangles):
    raise ValueError("source triangle disk is disconnected")

  outgoing: dict[int, int] = {}
  incoming: Counter[int] = Counter()
  for start, end in boundary:
    if start in outgoing:
      raise ValueError("source boundary has multiple outgoing segments")
    outgoing[start] = end
    incoming[end] += 1
  vertices_on_boundary = set(outgoing)
  if (
      vertices_on_boundary != set(incoming)
      or any(incoming[index] != 1 for index in vertices_on_boundary)
  ):
    raise ValueError("source boundary is open or non-manifold")
  loops: list[np.ndarray] = []
  remaining = set(vertices_on_boundary)
  while remaining:
    start = min(remaining)
    indices = [start]
    current = outgoing[start]
    while current != start:
      if current not in remaining or len(indices) > len(vertices_on_boundary):
        raise ValueError("source boundary cycle repeats or escapes")
      indices.append(current)
      current = outgoing[current]
    remaining.difference_update(indices)
    if len(indices) < 3:
      raise ValueError("source boundary loop is degenerate")
    loops.append(vertices[indices])
  mesh_area = float(np.sum(areas2) / 2.0)
  return loops, mesh_area, normal


def _strict_json(raw: bytes, *, label: str) -> Any:
  def pairs(rows: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
      if key in result:
        raise ValueError(f"{label} contains a duplicate JSON key")
      result[key] = value
    return result

  try:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite JSON number {value}")
        ),
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _source_smt_domain(
    *,
    assembly_json_bytes: bytes,
    source_smt_bytes: bytes,
    source_smt_archive_member: str,
    body_uuid: str,
    source_face_index: int,
) -> dict[str, Any]:
  """Parse only the bounded ASM entities needed for one closed-circle face."""

  if (
      not isinstance(source_smt_bytes, bytes)
      or not source_smt_bytes
      or not isinstance(source_smt_archive_member, str)
      or not source_smt_archive_member.endswith(".smt")
      or "\\" in source_smt_archive_member
  ):
    raise ValueError("source SMT bytes/path binding differs")
  assembly = _strict_json(assembly_json_bytes, label="source assembly")
  bodies = assembly.get("bodies") if isinstance(assembly, Mapping) else None
  body = bodies.get(body_uuid) if isinstance(bodies, Mapping) else None
  declared_smt = body.get("smt") if isinstance(body, Mapping) else None
  if (
      not isinstance(declared_smt, str)
      or not declared_smt.endswith(".smt")
      or source_smt_archive_member.split("/")[-1] != declared_smt
  ):
    raise ValueError("assembly body does not bind the source SMT member")
  contacts = assembly.get("contacts")
  if not isinstance(contacts, list):
    raise ValueError("source assembly contact domain differs")
  entities = []
  for contact in contacts:
    if not isinstance(contact, Mapping):
      raise ValueError("source assembly contact schema differs")
    for role in ("entity_one", "entity_two"):
      entity = contact.get(role)
      if (
          isinstance(entity, Mapping)
          and entity.get("body") == body_uuid
          and entity.get("index") == source_face_index
      ):
        entities.append(_json_data(entity))
  if not entities or any(entity != entities[0] for entity in entities[1:]):
    raise ValueError("source face metadata is absent or conflicting")
  entity = entities[0]
  persistent_face_id = entity.get("id")
  if (
      entity.get("type") != "BRepFace"
      or entity.get("surface_type") != "PlaneSurfaceType"
      or type(persistent_face_id) is not int
      or persistent_face_id < 0
  ):
    raise ValueError("source face lacks one planar persistent identity")

  try:
    text = source_smt_bytes.decode("utf-8")
  except UnicodeDecodeError as error:
    raise ValueError("source SMT is not UTF-8 text") from error
  if "\x00" in text:
    raise ValueError("source SMT contains NUL")
  lines = text.splitlines()
  if len(lines) < 4 or not lines[0].split() or lines[0].split()[0] != "22700":
    raise ValueError("source SMT header differs")
  payload = "\n".join(lines[3:])
  records = [
      chunk.split()
      for chunk in payload.split("#")
      if chunk.strip() and not chunk.lstrip().startswith("End-of-")
  ]
  if not records:
    raise ValueError("source SMT entity domain is empty")

  def pointer(value: str, *, label: str) -> tuple[int, list[str]]:
    if (
        not isinstance(value, str)
        or not value.startswith("$")
        or value == "$-1"
    ):
      raise ValueError(f"{label} pointer differs")
    try:
      index = int(value[1:])
    except ValueError as error:
      raise ValueError(f"{label} pointer differs") from error
    if index < 0 or index >= len(records):
      raise ValueError(f"{label} pointer is outside the SMT entity domain")
    return index, records[index]

  face_matches = [
      (index, record)
      for index, record in enumerate(records)
      if len(record) >= 9
      and record[0] == "face"
      and record[2] == str(persistent_face_id)
  ]
  if len(face_matches) != 1:
    raise ValueError("persistent source face identity is absent or duplicated")
  face_record_index, face = face_matches[0]
  _surface_index, surface = pointer(face[8], label="source face surface")
  if len(surface) < 13 or surface[0] != "plane-surface":
    raise ValueError("persistent source face support is not Plane")
  try:
    plane_origin = np.asarray(
        [10.0 * float(value) for value in surface[4:7]], dtype=float
    )
    plane_normal = _unit(
        np.asarray([float(value) for value in surface[7:10]], dtype=float),
        label="source SMT plane normal",
    )
  except (TypeError, ValueError) as error:
    raise ValueError("source SMT plane parameters differ") from error

  loop_rows: list[tuple[int, list[str]]] = []
  loop_pointer = face[5]
  seen_loops: set[int] = set()
  while loop_pointer != "$-1":
    loop_index, loop = pointer(loop_pointer, label="source face loop")
    if (
        loop_index in seen_loops
        or len(loop) < 7
        or loop[0] != "loop"
        or loop[6] != f"${face_record_index}"
    ):
      raise ValueError("source SMT loop chain differs")
    seen_loops.add(loop_index)
    loop_rows.append((loop_index, loop))
    loop_pointer = loop[4]
  if not loop_rows:
    raise ValueError("source SMT face contains no loops")

  circles = []
  for loop_index, loop in loop_rows:
    coedge_index, coedge = pointer(loop[5], label="source loop coedge")
    if (
        len(coedge) < 10
        or coedge[0] != "coedge"
        or coedge[4] != f"${coedge_index}"
        or coedge[5] != f"${coedge_index}"
        or coedge[9] != f"${loop_index}"
    ):
      raise ValueError("source SMT loop is not exactly one closed coedge")
    _edge_index, edge = pointer(coedge[7], label="source loop edge")
    if (
        len(edge) < 10
        or edge[0] != "edge"
        or edge[4] != edge[6]
        or edge[4] == "$-1"
    ):
      raise ValueError("source SMT edge is not topologically closed")
    try:
      first_parameter = float(edge[5])
      last_parameter = float(edge[7])
    except ValueError as error:
      raise ValueError("source SMT edge parameters differ") from error
    if (
        not math.isfinite(first_parameter)
        or not math.isfinite(last_parameter)
        or abs(
            abs(last_parameter - first_parameter) - 2.0 * math.pi
        ) > 1.0e-12
    ):
      raise ValueError("source SMT edge does not span one full period")
    _curve_index, curve = pointer(edge[9], label="source loop curve")
    if len(curve) < 16 or curve[0] != "ellipse-curve":
      raise ValueError("source SMT boundary curve is not analytic ellipse")
    try:
      center = np.asarray(
          [10.0 * float(value) for value in curve[4:7]], dtype=float
      )
      axis = _unit(
          np.asarray([float(value) for value in curve[7:10]], dtype=float),
          label="source SMT circle axis",
      )
      major = np.asarray(
          [10.0 * float(value) for value in curve[10:13]], dtype=float
      )
      ratio = float(curve[13])
    except (TypeError, ValueError) as error:
      raise ValueError("source SMT ellipse parameters differ") from error
    radius = float(np.linalg.norm(major))
    if (
        not np.all(np.isfinite(center))
        or not np.all(np.isfinite(major))
        or not math.isfinite(radius)
        or radius <= 0.0
        or ratio != 1.0
        or abs(float(np.dot(axis, major))) > 1.0e-8
    ):
      raise ValueError("source SMT ellipse is not one guarded circle")
    x_axis = major / radius
    y_axis = _unit(
        np.cross(axis, x_axis), label="source SMT circle y axis"
    )
    circles.append({
        "center": center,
        "axis": axis,
        "radius": radius,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "persistent_edge_id": edge[2],
        "first_parameter": first_parameter,
        "last_parameter": last_parameter,
    })
  return {
      "plane_origin": plane_origin,
      "plane_normal": plane_normal,
      "circles": circles,
      "binding": {
          "source_smt_bytes_sha256": hashlib.sha256(
              source_smt_bytes
          ).hexdigest(),
          "source_smt_byte_count": len(source_smt_bytes),
          "source_smt_archive_member": source_smt_archive_member,
          "body_uuid": body_uuid,
          "source_face_index": source_face_index,
          "persistent_face_id": persistent_face_id,
          "persistent_face_record_index": face_record_index,
          "persistent_face_match_count": len(face_matches),
          "surface_entity_type": surface[0],
          "loop_count": len(loop_rows),
          "full_circle_edge_count": len(circles),
      },
  }


def _occ_domain(step_bytes: bytes, raw_occ_face_index: int) -> dict[str, Any]:
  from neurocad.analytic_trim_domain_v1 import _read_step_bytes
  from neurocad.fusion_face_mapper import _occ_wire_edge_uses
  from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
  from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Plane

  shape, units = _read_step_bytes(step_bytes)
  faces = tuple(shape.Faces())
  if (
      type(raw_occ_face_index) is not int
      or raw_occ_face_index < 0
      or raw_occ_face_index >= len(faces)
  ):
    raise ValueError("STEP face index is outside the body")
  face = faces[raw_occ_face_index]
  surface = BRepAdaptor_Surface(face.wrapped, True)
  if surface.GetType() != GeomAbs_Plane:
    raise ValueError("candidate OCC face is not Plane")
  plane = surface.Plane()
  plane_origin = _vec(plane.Location())
  plane_normal = _unit(_vec(plane.Axis().Direction()), label="OCC plane normal")

  import cadquery as cq
  edges = _occ_wire_edge_uses(face, cq)
  wires = tuple(face.Wires())
  if not wires or len(edges) != len(wires):
    raise ValueError("each OCC wire must contain exactly one edge")
  circles = []
  for edge in edges:
    curve = BRepAdaptor_Curve(edge.wrapped)
    if (
        curve.GetType() != GeomAbs_Circle
        or not edge.IsClosed()
        or len(edge.Vertices()) not in {1, 2}
    ):
      raise ValueError("OCC boundary is not closed-circle-only")
    circle = curve.Circle()
    center = _vec(circle.Location())
    axis = _unit(_vec(circle.Axis().Direction()), label="OCC circle axis")
    radius = float(circle.Radius())
    x_axis = _unit(_vec(circle.XAxis().Direction()), label="OCC circle x axis")
    y_axis = _unit(_vec(circle.YAxis().Direction()), label="OCC circle y axis")
    if not math.isfinite(radius) or radius <= 0.0:
      raise ValueError("OCC circle radius differs")
    if abs(abs(float(np.dot(axis, plane_normal))) - 1.0) > _DIRECTION_GUARD:
      raise ValueError("OCC circle axis differs from its face plane")
    circles.append({
        "center": center,
        "axis": axis,
        "radius": radius,
        "x_axis": x_axis,
        "y_axis": y_axis,
    })
  return {
      "face": face,
      "face_count": len(faces),
      "units": units,
      "plane_origin": plane_origin,
      "plane_normal": plane_normal,
      "circles": circles,
      "area": float(face.Area()),
  }


def _loop_circle_match(
    loop: np.ndarray,
    circle: Mapping[str, Any],
) -> dict[str, Any] | None:
  center = circle["center"]
  axis = circle["axis"]
  radius = float(circle["radius"])
  displacement = loop - center
  axial = displacement @ axis
  radial = displacement - axial[:, None] * axis[None, :]
  radial_length = np.linalg.norm(radial, axis=1)
  maximum_residual = max(
      float(np.max(np.abs(axial))),
      float(np.max(np.abs(radial_length - radius))),
  )
  if maximum_residual > _NUMERIC_GUARD_MM:
    return None
  x = radial @ circle["x_axis"]
  y = radial @ circle["y_axis"]
  angles = np.arctan2(y, x)
  deltas = []
  for first, second in zip(angles, np.roll(angles, -1)):
    delta = math.atan2(math.sin(float(second - first)),
                       math.cos(float(second - first)))
    if abs(delta) <= _ANGLE_GUARD or abs(delta) >= math.pi - _ANGLE_GUARD:
      return None
    deltas.append(delta)
  signs = {1 if value > 0.0 else -1 for value in deltas}
  winding = math.fsum(deltas) / (2.0 * math.pi)
  if len(signs) != 1 or abs(abs(winding) - 1.0) > _ANGLE_GUARD:
    return None
  return {
      "source_segment_count": len(loop),
      "maximum_vertex_to_circle_residual_mm": maximum_residual,
      "winding_sign": next(iter(signs)),
      "winding_turns": winding,
      "maximum_parameter_increment_rad": max(abs(value) for value in deltas),
      "strictly_monotone_full_turn": True,
  }


def _circle_circle_match(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any] | None:
  center_residual = float(np.linalg.norm(source["center"] - target["center"]))
  radius_residual = abs(float(source["radius"]) - float(target["radius"]))
  axis_alignment_error = abs(
      abs(float(np.dot(source["axis"], target["axis"]))) - 1.0
  )
  if (
      max(center_residual, radius_residual) > _NUMERIC_GUARD_MM
      or axis_alignment_error > _DIRECTION_GUARD
  ):
    return None
  return {
      "maximum_center_or_radius_residual_mm": max(
          center_residual, radius_residual
      ),
      "axis_alignment_error": axis_alignment_error,
  }


def build_planar_closed_circle_chordal_trim_receipt_v1(
    *,
    obj_bytes: bytes,
    assembly_json_bytes: bytes,
    source_smt_bytes: bytes,
    source_smt_archive_member: str,
    step_bytes: bytes,
    body_uuid: str,
    source_face_index: int,
    raw_occ_face_index: int,
    source_occurrence_path: Sequence[str],
    source_path_identity: Mapping[str, Any],
) -> dict[str, Any]:
  from neurocad.analytic_trim_domain_v1 import (
      _source_path_identity,
      _source_root_transform_binding,
  )

  if source_occurrence_path:
    raise ValueError("planar chordal trim v1 requires a root occurrence")
  transform = _source_root_transform_binding(
      assembly_json_bytes,
      body_uuid=body_uuid,
      occurrence_path=source_occurrence_path,
  )
  paths = _source_path_identity(source_path_identity, body_uuid=body_uuid)
  source = capture_source_obj_face_v1(
      obj_bytes=obj_bytes,
      assembly_json_bytes=assembly_json_bytes,
      body_uuid=body_uuid,
      face_index=source_face_index,
      unit_binding={
          "source_length_unit": "centimetre",
          "millimeters_per_source_unit": 10.0,
      },
      path_metadata={
          "obj_archive_member": paths["obj_archive_member"],
          "assembly_archive_member": paths["assembly_archive_member"],
      },
  )
  source_view = replay_source_obj_face_topology_observation_v1(source)
  loops, source_area, source_normal = _source_loops(source_view)
  source_smt = _source_smt_domain(
      assembly_json_bytes=assembly_json_bytes,
      source_smt_bytes=source_smt_bytes,
      source_smt_archive_member=source_smt_archive_member,
      body_uuid=body_uuid,
      source_face_index=source_face_index,
  )
  occ = _occ_domain(step_bytes, raw_occ_face_index)
  if (
      abs(abs(float(np.dot(source_normal, source_smt["plane_normal"]))) - 1.0)
      > _DIRECTION_GUARD
      or abs(
          abs(float(np.dot(
              source_smt["plane_normal"], occ["plane_normal"]
          ))) - 1.0
      ) > _DIRECTION_GUARD
  ):
    raise ValueError("OBJ, source SMT, and OCC plane normals differ")
  source_vertices = np.concatenate(loops, axis=0)
  source_smt_plane_residual = np.abs(
      (source_vertices - source_smt["plane_origin"])
      @ source_smt["plane_normal"]
  )
  if float(np.max(source_smt_plane_residual)) > _NUMERIC_GUARD_MM:
    raise ValueError("OBJ boundary differs from the source SMT plane")
  area_relative_error = abs(source_area - occ["area"]) / max(
      source_area, occ["area"]
  )
  if area_relative_error > LOCKED_AREA_RELATIVE_ERROR:
    raise ValueError("source and OCC face areas differ")
  source_circles = source_smt["circles"]
  occ_circles = occ["circles"]
  if not len(loops) == len(source_circles) == len(occ_circles):
    raise ValueError("OBJ/source-SMT/OCC boundary loop counts differ")
  source_matches: list[tuple[int, dict[str, Any]]] = []
  used_source_circles: set[int] = set()
  for loop in loops:
    candidates = [
        (index, proof)
        for index, circle in enumerate(source_circles)
        if (proof := _loop_circle_match(loop, circle)) is not None
    ]
    if len(candidates) != 1:
      raise ValueError("OBJ loop does not uniquely match one source SMT circle")
    index, proof = candidates[0]
    if index in used_source_circles:
      raise ValueError("multiple OBJ loops match one source SMT circle")
    used_source_circles.add(index)
    source_matches.append((index, proof))
  if used_source_circles != set(range(len(source_circles))):
    raise ValueError("source SMT circle coverage is incomplete")
  occ_matches: list[tuple[int, dict[str, Any]]] = []
  used_occ_circles: set[int] = set()
  for source_circle in source_circles:
    candidates = [
        (index, proof)
        for index, circle in enumerate(occ_circles)
        if (
            proof := _circle_circle_match(source_circle, circle)
        ) is not None
    ]
    if len(candidates) != 1:
      raise ValueError(
          "source SMT circle does not uniquely match one OCC circle"
      )
    index, proof = candidates[0]
    if index in used_occ_circles:
      raise ValueError("multiple source SMT circles match one OCC circle")
    used_occ_circles.add(index)
    occ_matches.append((index, proof))
  if used_occ_circles != set(range(len(occ_circles))):
    raise ValueError("OCC circle coverage is incomplete")

  source_binding = _json_data(source_view["source_binding"])
  occ_binding = {
      "step_bytes_sha256": hashlib.sha256(step_bytes).hexdigest(),
      "step_byte_count": len(step_bytes),
      "source_assembly_bytes_sha256": hashlib.sha256(
          assembly_json_bytes
      ).hexdigest(),
      "source_transform": transform,
      "source_body_occurrence_identity": {
          "body_uuid": body_uuid,
          "occurrence_path": list(source_occurrence_path),
          "root_component_uuid": transform["root_component_uuid"],
      },
      "source_path_identity": paths,
      "raw_occ_face_index": raw_occ_face_index,
      "deterministic_face_signature_sha256": (
          source_face_signature_sha256(occ["face"])
      ),
      "imported_face_count": occ["face_count"],
      "step_length_unit_names": list(occ["units"]),
      "occ_runtime_length_unit": "millimetre",
  }
  unsigned = {
      "schema_version": SCHEMA_VERSION,
      "algorithm_revision": ALGORITHM_REVISION,
      "formal_authorized": False,
      "decision": "certified_equivalent",
      "evidence_scope": (
          "development_planar_closed_circle_chordal_trim_only"
      ),
      "source_binding": source_binding,
      "source_smt_binding": source_smt["binding"],
      "occ_binding": occ_binding,
      "source_topology_payload_sha256": source_view[
          "topology_payload_sha256"
      ],
      "domain": {
          "source_loop_count": len(loops),
          "source_smt_closed_circle_count": len(source_circles),
          "occ_closed_circle_count": len(occ_circles),
          "source_boundary_segment_count": sum(len(loop) for loop in loops),
          "unique_obj_to_source_smt_loop_bijection": True,
          "unique_source_smt_to_occ_loop_bijection": True,
          "all_source_loops_strictly_monotone_full_turn": True,
          "source_mesh_area_mm2": source_area,
          "occ_face_area_mm2": occ["area"],
          "area_relative_error": area_relative_error,
          "obj_to_source_smt_loop_matches": [
              {"source_smt_circle_ordinal": index, **proof}
              for index, proof in source_matches
          ],
          "source_smt_to_occ_loop_matches": [
              {"occ_circle_ordinal": index, **proof}
              for index, proof in occ_matches
          ],
      },
      "thresholds": {
          "locked_boundary_distance_mm": LOCKED_BOUNDARY_DISTANCE_MM,
          "conservative_numeric_guard_mm": _NUMERIC_GUARD_MM,
          "locked_area_relative_error": LOCKED_AREA_RELATIVE_ERROR,
          "new_chord_sag_threshold_introduced": False,
          "sampling_density_authorizes": False,
      },
  }
  return {**unsigned, "receipt_payload_sha256": _sha(unsigned)}


def verify_planar_closed_circle_chordal_trim_receipt_v1(
    receipt: Mapping[str, Any],
    **inputs: Any,
) -> dict[str, Any]:
  expected = build_planar_closed_circle_chordal_trim_receipt_v1(**inputs)
  if dict(receipt) != expected:
    raise ValueError("planar chordal trim receipt differs from full replay")
  return expected


__all__ = [
    "ALGORITHM_REVISION",
    "LOCKED_BOUNDARY_DISTANCE_MM",
    "SCHEMA_VERSION",
    "build_planar_closed_circle_chordal_trim_receipt_v1",
    "verify_planar_closed_circle_chordal_trim_receipt_v1",
]
