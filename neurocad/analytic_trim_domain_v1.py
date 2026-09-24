"""Development-only certificate for one rectangular cylindrical trim domain.

This module intentionally supports one narrow case: a source OBJ triangle disk
whose analytic boundary is two cylinder generators plus two circular arcs, and
one receipt-bound STEP face whose single wire has the corresponding four
linear pcurves.  The development v2.6 mapper may consume this certificate
under its narrower root/single-candidate gate; it grants no formal authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

from neurocad.cadquery_backend import source_face_signature_sha256
from neurocad.fusion_face_mapper import (
    _boundary_loop_count,
    _occ_wire_edge_uses,
)
from neurocad.source_obj_analytic_support_v1 import (
    replay_source_obj_face_topology_v1,
)


SCHEMA_VERSION = "cylinder_rectangular_analytic_trim_certificate.v1"
EVIDENCE_SCOPE = "development_cylinder_rectangular_trim_only_no_mapper_authority"
LOCKED_BOUNDARY_DISTANCE_MM = 0.2


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


def _freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({
        str(key): _freeze(item) for key, item in value.items()
    })
  if isinstance(value, list):
    return tuple(_freeze(item) for item in value)
  return value


def _q(value: float) -> Fraction:
  if not math.isfinite(value):
    raise ValueError("non-finite numeric replay")
  return Fraction(*float(value).as_integer_ratio())


def _qdot(a: Sequence[float], b: Sequence[float]) -> Fraction:
  return sum((_q(x) * _q(y) for x, y in zip(a, b, strict=True)), Fraction())


def _qsub(a: Sequence[float], b: Sequence[float]) -> tuple[Fraction, ...]:
  return tuple(_q(x) - _q(y) for x, y in zip(a, b, strict=True))


def _qcross(
    a: Sequence[float] | Sequence[Fraction],
    b: Sequence[float] | Sequence[Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
  x = tuple(item if isinstance(item, Fraction) else _q(item) for item in a)
  y = tuple(item if isinstance(item, Fraction) else _q(item) for item in b)
  return (
      x[1] * y[2] - x[2] * y[1],
      x[2] * y[0] - x[0] * y[2],
      x[0] * y[1] - x[1] * y[0],
  )


def _qnorm_sq(value: Sequence[float] | Sequence[Fraction]) -> Fraction:
  return sum((
      item * item if isinstance(item, Fraction) else _q(item) ** 2
      for item in value
  ), Fraction())


def _fraction_payload(value: Fraction) -> dict[str, int]:
  return {
      "numerator": value.numerator,
      "denominator": value.denominator,
  }


def _fraction_from_payload(value: Mapping[str, Any]) -> Fraction:
  if (
      not isinstance(value, Mapping)
      or set(value) != {"numerator", "denominator"}
      or type(value["numerator"]) is not int
      or type(value["denominator"]) is not int
      or value["denominator"] <= 0
  ):
    raise ValueError("fraction payload differs")
  return Fraction(value["numerator"], value["denominator"])


def _point_distance_sq(
    first: Sequence[float], second: Sequence[float],
) -> Fraction:
  return _qnorm_sq(_qsub(first, second))


def _within_point_tolerance(
    first: Sequence[float],
    second: Sequence[float],
    tolerance: float,
) -> bool:
  return _point_distance_sq(first, second) <= _q(tolerance) ** 2


def _vec(value: Any) -> tuple[float, float, float]:
  if hasattr(value, "toTuple"):
    value = value.toTuple()
  elif hasattr(value, "X") and callable(value.X):
    value = (value.X(), value.Y(), value.Z())
  result = tuple(float(item) for item in value)
  if len(result) != 3 or not all(math.isfinite(item) for item in result):
    raise ValueError("geometry vector differs")
  return result  # type: ignore[return-value]


def _unit(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
  length = math.sqrt(math.fsum(item * item for item in value))
  if not math.isfinite(length) or length <= 0.0:
    raise ValueError(f"{label} is degenerate")
  return tuple(item / length for item in value)  # type: ignore[return-value]


def _source_root_transform_binding(
    assembly_bytes: bytes, *, body_uuid: str, occurrence_path: Sequence[str],
) -> dict[str, Any]:
  assembly = _strict_json(assembly_bytes, label="source assembly")
  if not isinstance(assembly, Mapping):
    raise ValueError("source assembly schema differs")
  if occurrence_path:
    raise ValueError(
        "cylinder trim v1 fails closed on non-root occurrence transforms"
    )
  if not isinstance(body_uuid, str) or not body_uuid:
    raise ValueError("source body UUID differs")
  bodies = assembly.get("bodies")
  root = assembly.get("root")
  components = assembly.get("components")
  if (
      not isinstance(bodies, Mapping)
      or body_uuid not in bodies
      or not isinstance(root, Mapping)
      or not isinstance(root.get("bodies"), Mapping)
      or body_uuid not in root["bodies"]
      or not isinstance(components, Mapping)
  ):
    raise ValueError("source root body binding differs")
  root_component = root.get("component")
  component = components.get(root_component)
  if (
      not isinstance(root_component, str)
      or not isinstance(component, Mapping)
      or not isinstance(component.get("bodies"), list)
      or body_uuid not in component["bodies"]
  ):
    raise ValueError("source root component/body binding differs")
  return {
      "kind": "root_identity",
      "root_component_uuid": root_component,
      "body_uuid": body_uuid,
      "occurrence_path": [],
      "world_transform_row_major": [
          1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0,
      ],
  }


def _source_path_identity(
    value: Mapping[str, Any], *, body_uuid: str,
) -> dict[str, str]:
  required = {
      "obj_archive_member",
      "step_archive_member",
      "assembly_archive_member",
  }
  if not isinstance(value, Mapping) or set(value) != required:
    raise ValueError("OCC source path identity schema differs")
  result = {}
  for key in sorted(required):
    item = value[key]
    if not isinstance(item, str) or not item or "\\" in item:
      raise ValueError("OCC source path identity differs")
    result[key] = item
  obj_member = result["obj_archive_member"]
  step_member = result["step_archive_member"]
  if not obj_member.endswith(f"/{body_uuid}.obj"):
    raise ValueError("OCC OBJ path/body identity differs")
  if not step_member.endswith(f"/{body_uuid}.step"):
    raise ValueError("OCC STEP path/body identity differs")
  if obj_member[:-4] != step_member[:-5]:
    raise ValueError("OCC OBJ/STEP path identity differs")
  parent = obj_member.rsplit("/", 1)[0]
  if result["assembly_archive_member"] != f"{parent}/assembly.json":
    raise ValueError("OCC assembly path identity differs")
  return result


def _read_step_bytes(step_bytes: bytes) -> tuple[Any, list[str]]:
  if not isinstance(step_bytes, bytes) or not step_bytes:
    raise TypeError("step_bytes must be non-empty bytes")
  import cadquery as cq  # type: ignore
  from OCP.IFSelect import IFSelect_RetDone  # type: ignore
  from OCP.Interface import Interface_Static  # type: ignore
  from OCP.STEPControl import STEPControl_Reader  # type: ignore
  from OCP.TColStd import TColStd_SequenceOfAsciiString  # type: ignore

  reader = STEPControl_Reader()
  if Interface_Static.SetCVal_s("xstep.cascade.unit", "MM") is not True:
    raise ValueError("STEP target unit could not be locked to millimetres")
  step_sha = hashlib.sha256(step_bytes).hexdigest()
  with io.BytesIO(step_bytes) as stream:
    if reader.ReadStream(f"captured-{step_sha}.step", stream) != IFSelect_RetDone:
      raise ValueError("captured STEP byte stream could not be read")
  length_units = TColStd_SequenceOfAsciiString()
  angle_units = TColStd_SequenceOfAsciiString()
  solid_angle_units = TColStd_SequenceOfAsciiString()
  reader.FileUnits(length_units, angle_units, solid_angle_units)
  units = [
      str(length_units.Value(index).ToCString()).strip().lower()
      for index in range(1, length_units.Length() + 1)
  ]
  if len(units) != 1 or units[0] not in {"millimetre", "centimetre"}:
    raise ValueError("STEP length-unit declaration is unsupported or ambiguous")
  roots = int(reader.NbRootsForTransfer())
  if roots <= 0:
    raise ValueError("captured STEP has no transferable roots")
  for index in range(1, roots + 1):
    if reader.TransferRoot(index) is not True:
      raise ValueError("captured STEP root transfer failed")
  shapes = []
  for index in range(1, int(reader.NbShapes()) + 1):
    wrapped = reader.Shape(index)
    if wrapped.IsNull():
      raise ValueError("captured STEP produced a null shape")
    shapes.append(cq.Shape.cast(wrapped))
  if not shapes:
    raise ValueError("captured STEP produced no shape")
  shape = cq.Workplane("XY").newObject(shapes).val()
  if shape is None:
    raise ValueError("captured STEP produced no CadQuery shape")
  return shape, units


def _replay_occ_capture(captured: Mapping[str, Any]) -> dict[str, Any]:
  transform = _source_root_transform_binding(
      captured["assembly_bytes"],
      body_uuid=captured["body_uuid"],
      occurrence_path=captured["occurrence_path"],
  )
  shape, units = _read_step_bytes(captured["step_bytes"])
  faces = tuple(shape.Faces())
  index = captured["face_index"]
  if type(index) is not int or index < 0 or index >= len(faces):
    raise ValueError("deterministic STEP face index is outside the body")
  face = faces[index]
  signature = source_face_signature_sha256(face)
  path_identity = _source_path_identity(
      captured["path_identity"], body_uuid=captured["body_uuid"]
  )
  step_sha = hashlib.sha256(captured["step_bytes"]).hexdigest()
  return {
      "shape": shape,
      "face": face,
      "binding": {
          "step_bytes_sha256": step_sha,
          "step_byte_count": len(captured["step_bytes"]),
          "source_assembly_bytes_sha256": hashlib.sha256(
              captured["assembly_bytes"]
          ).hexdigest(),
          "source_assembly_byte_count": len(captured["assembly_bytes"]),
          "source_transform": transform,
          "source_body_occurrence_identity": {
              "body_uuid": captured["body_uuid"],
              "occurrence_path": list(captured["occurrence_path"]),
              "root_component_uuid": transform["root_component_uuid"],
          },
          "source_path_identity": path_identity,
          "step_body_identity": {
              "body_uuid": captured["body_uuid"],
              "step_archive_member": path_identity["step_archive_member"],
              "step_bytes_sha256": step_sha,
          },
          "raw_occ_face_index": index,
          "deterministic_face_signature_sha256": signature,
          "imported_face_count": len(faces),
          "step_length_unit_names": units,
          "occ_runtime_length_unit": "millimetre",
      },
  }


def _sealed_occ_boundary():
  states: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
  seal = object()

  class OccCylinderTrimDomainCapabilityV1:
    __slots__ = ("__weakref__",)

    def __new__(cls, *, _seal: object | None = None):
      if _seal is not seal:
        raise TypeError("OCC trim-domain capabilities are capture-factory-only")
      return super().__new__(cls)

    def __setattr__(self, _name: str, _value: Any) -> None:
      raise TypeError("OCC trim-domain capabilities are sealed")

    def __copy__(self):
      raise TypeError("OCC trim-domain capabilities cannot be copied")

    def __deepcopy__(self, _memo: Any):
      raise TypeError("OCC trim-domain capabilities cannot be deep-copied")

    def __reduce_ex__(self, _protocol: int):
      raise TypeError("OCC trim-domain capabilities are not serializable")

  def capture_occ_cylinder_trim_domain_v1(
      *,
      step_bytes: bytes,
      source_assembly_json_bytes: bytes,
      source_body_uuid: str,
      raw_occ_face_index: int,
      source_occurrence_path: Sequence[str],
      source_path_identity: Mapping[str, Any],
  ):
    captured = {
        "step_bytes": step_bytes,
        "assembly_bytes": source_assembly_json_bytes,
        "body_uuid": source_body_uuid,
        "face_index": raw_occ_face_index,
        "occurrence_path": tuple(source_occurrence_path),
        "path_identity": dict(source_path_identity),
    }
    replayed = _replay_occ_capture(captured)
    value = OccCylinderTrimDomainCapabilityV1(_seal=seal)
    states[value] = MappingProxyType({
        **captured,
        "binding": _freeze(replayed["binding"]),
    })
    return value

  def replay(value: Any) -> dict[str, Any]:
    if type(value) is not OccCylinderTrimDomainCapabilityV1:
      raise TypeError("expected an exact OCC trim-domain capability")
    try:
      captured = states[value]
    except KeyError as error:
      raise TypeError("OCC trim-domain capability lacks capture state") from error
    replayed = _replay_occ_capture(captured)
    if _freeze(replayed["binding"]) != captured["binding"]:
      raise ValueError("OCC trim-domain capture replay differs")
    return replayed

  return (
      OccCylinderTrimDomainCapabilityV1,
      capture_occ_cylinder_trim_domain_v1,
      replay,
  )


(
    OccCylinderTrimDomainCapabilityV1,
    capture_occ_cylinder_trim_domain_v1,
    _replay_occ_capability,
) = _sealed_occ_boundary()
del _sealed_occ_boundary


def _source_parameter(
    point: Sequence[float],
    *,
    axis_point: Sequence[float],
    axis: Sequence[float],
) -> tuple[Fraction, tuple[Fraction, Fraction, Fraction]]:
  displacement = _qsub(point, axis_point)
  axis_q = tuple(_q(value) for value in axis)
  axis_sq = _qnorm_sq(axis_q)
  axial = sum((
      value * direction
      for value, direction in zip(displacement, axis_q, strict=True)
  ), Fraction()) / axis_sq
  radial = tuple(
      value - axial * direction
      for value, direction in zip(displacement, axis_q, strict=True)
  )
  return axial, radial  # type: ignore[return-value]


def _radial_signature(
    radial: Sequence[Fraction],
) -> tuple[tuple[int, int], ...]:
  return tuple((value.numerator, value.denominator) for value in radial)


def _validate_source_segment_cycle(
    loop: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
  if len(loop) < 4 or len(segments) != len(loop):
    raise ValueError("source full boundary segment coverage differs")
  endpoint_uses: Counter[int] = Counter()
  for index, segment in enumerate(segments):
    expected = (loop[index], loop[(index + 1) % len(loop)])
    observed = (segment.get("start_vertex"), segment.get("end_vertex"))
    if observed != expected:
      raise ValueError("source segment cycle has a gap, overlap, or reversal")
    endpoint_uses.update(expected)
  if set(endpoint_uses) != set(loop) or any(
      count != 2 for count in endpoint_uses.values()
  ):
    raise ValueError("source boundary vertex coverage is not exactly two")

  groups: list[dict[str, Any]] = []
  for index, segment in enumerate(segments):
    kind = segment.get("kind")
    if kind not in {"axial", "circular"}:
      raise ValueError("source segment kind differs")
    if not groups or groups[-1]["kind"] != kind:
      groups.append({"kind": kind, "segment_indices": [index]})
    else:
      groups[-1]["segment_indices"].append(index)
  if len(groups) > 1 and groups[0]["kind"] == groups[-1]["kind"]:
    groups[0]["segment_indices"] = (
        groups[-1]["segment_indices"] + groups[0]["segment_indices"]
    )
    groups.pop()
  if (
      len(groups) != 4
      or Counter(group["kind"] for group in groups)
      != {"axial": 2, "circular": 2}
  ):
    raise ValueError("source boundary does not merge to a four-chain rectangle")

  directions: dict[str, list[int]] = {"axial": [], "circular": []}
  for group in groups:
    values = [
        int(segments[index]["direction_sign"])
        for index in group["segment_indices"]
    ]
    if set(values) not in ({1}, {-1}):
      raise ValueError("source chain interval reverses or is non-monotone")
    group["direction_sign"] = values[0]
    directions[group["kind"]].append(values[0])
    if group["kind"] == "circular":
      radial_path = [
          segments[group["segment_indices"][0]]["start_radial_signature"],
          *[
              segments[index]["end_radial_signature"]
              for index in group["segment_indices"]
          ],
      ]
      if len(set(radial_path)) != len(radial_path):
        raise ValueError("source circular chain repeats a period branch")
  if sorted(directions["axial"]) != [-1, 1] or sorted(
      directions["circular"]
  ) != [-1, 1]:
    raise ValueError("source opposite chains do not have opposite orientations")
  return groups


def _source_boundary(source_view: Mapping[str, Any]) -> dict[str, Any]:
  if source_view["surface_type"] != "cylinder":
    raise ValueError("source support is not Cylinder")
  vertices = tuple(tuple(float(x) for x in row)
                   for row in source_view["vertices_mm"])
  triangles = tuple(
      tuple(int(use["vertex_index"]) for use in triangle)
      for triangle in source_view["triangles"]
  )
  if not triangles or len(set(triangles)) != len(triangles):
    raise ValueError("source triangle topology is empty or duplicated")
  edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
  triangle_neighbours: dict[int, set[int]] = defaultdict(set)
  edge_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
  for triangle_index, triangle in enumerate(triangles):
    if len(triangle) != 3 or len(set(triangle)) != 3:
      raise ValueError("source triangle topology differs")
    for start, end in zip(triangle, triangle[1:] + triangle[:1]):
      key = tuple(sorted((start, end)))
      edge_uses[key].append((start, end))
      edge_triangles[key].append(triangle_index)
  for key, uses in edge_uses.items():
    if len(uses) not in {1, 2}:
      raise ValueError("source triangle edge is non-manifold")
    if len(uses) == 2:
      if uses[0] != (uses[1][1], uses[1][0]):
        raise ValueError("source triangle orientations disagree")
      left, right = edge_triangles[key]
      triangle_neighbours[left].add(right)
      triangle_neighbours[right].add(left)
  visited = {0}
  stack = [0]
  while stack:
    stack.extend(triangle_neighbours[stack.pop()] - visited)
    visited.update(stack)
  if len(visited) != len(triangles):
    raise ValueError("source triangle topology is disconnected")
  boundary = [uses[0] for uses in edge_uses.values() if len(uses) == 1]
  try:
    if _boundary_loop_count(
        [((start, 0, 0), (end, 0, 0)) for start, end in boundary],
        invalid_status="source_trim_boundary_invalid",
    ) != 1:
      raise ValueError("source boundary has holes or multiple outer loops")
  except Exception as error:
    raise ValueError("source boundary is open, branched, or disconnected") from error
  outgoing: dict[int, int] = {}
  incoming: Counter[int] = Counter()
  for start, end in boundary:
    if start in outgoing:
      raise ValueError("source boundary is branched")
    outgoing[start] = end
    incoming[end] += 1
  if set(outgoing) != set(incoming) or any(count != 1 for count in incoming.values()):
    raise ValueError("source boundary is open or non-manifold")
  start = min(outgoing)
  loop = [start]
  while True:
    following = outgoing[loop[-1]]
    if following == start:
      break
    if following in loop or len(loop) > len(boundary):
      raise ValueError("source boundary contains multiple/self-repeating loops")
    loop.append(following)
  if len(loop) != len(boundary):
    raise ValueError("source boundary has holes or multiple outer loops")
  used_vertex_count = len({index for triangle in triangles for index in triangle})
  if used_vertex_count - len(edge_uses) + len(triangles) != 1:
    raise ValueError("source triangle disk Euler characteristic differs")

  import cadquery as cq  # type: ignore

  support = source_view["support"]
  axis = tuple(float(x) for x in support["axis_direction"])
  axis_point = tuple(float(x) for x in support["axis_point_mm"])
  radius = float(support["radius_mm"])
  axis_sq = _qnorm_sq(axis)
  tolerance = float(source_view["thresholds"]["support_tolerance_mm"])
  tolerance_sq = _q(tolerance) ** 2
  radius_q = _q(radius)
  parameters = [
      _source_parameter(point, axis_point=axis_point, axis=axis)
      for point in vertices
  ]
  segments: list[dict[str, Any]] = []
  for edge_index, (first_index, second_index) in enumerate(
      zip(loop, loop[1:] + loop[:1])
  ):
    first_axial, first_radial = parameters[first_index]
    second_axial, second_radial = parameters[second_index]
    axial_delta = second_axial - first_axial
    radial_delta = tuple(
        right - left
        for left, right in zip(first_radial, second_radial, strict=True)
    )
    radial_delta_sq = _qnorm_sq(radial_delta)
    axial = (
        radial_delta_sq <= tolerance_sq
        and axial_delta * axial_delta > tolerance_sq
    )
    circular = (
        axial_delta * axial_delta <= tolerance_sq
        and radial_delta_sq > tolerance_sq
    )
    if axial == circular:
      raise ValueError("source segment is neither axial nor angular")
    start_point = vertices[first_index]
    end_point = vertices[second_index]
    if axial:
      if (
          sum((
              left * right
              for left, right in zip(
                  first_radial, second_radial, strict=True
              )
          ), Fraction()) <= 0
      ):
        raise ValueError("source axial segment changes radial branch")
      direction = 1 if axial_delta > 0 else -1
      analytic_edge = cq.Edge.makeLine(
          cq.Vector(*start_point), cq.Vector(*end_point)
      )
      interval = {
          "coordinate": "axial_mm",
          "start": _fraction_payload(first_axial),
          "end": _fraction_payload(second_axial),
      }
      short_arc_proof = None
    else:
      radial_cross = _qcross(first_radial, second_radial)
      signed_cross = sum((
          value * _q(direction)
          for value, direction in zip(radial_cross, axis, strict=True)
      ), Fraction())
      radial_dot = sum((
          left * right
          for left, right in zip(
              first_radial, second_radial, strict=True
          )
      ), Fraction())
      cross_floor_sq = (_q(tolerance) * radius_q) ** 2 * axis_sq
      if (
          signed_cross * signed_cross <= cross_floor_sq
          or radial_dot <= -(radius_q ** 2) + _q(tolerance) * radius_q
      ):
        raise ValueError(
            "source angular segment has half-turn/seam ambiguity"
        )
      direction = 1 if signed_cross > 0 else -1
      radial_sum = tuple(
          float(left + right)
          for left, right in zip(
              first_radial, second_radial, strict=True
          )
      )
      mid_radial = _unit(radial_sum, label="source short-arc midpoint")
      mid_axial = float((first_axial + second_axial) / 2)
      midpoint = tuple(
          axis_point[index]
          + axis[index] * mid_axial
          + radius * mid_radial[index]
          for index in range(3)
      )
      analytic_edge = cq.Edge.makeThreePointArc(
          cq.Vector(*start_point),
          cq.Vector(*midpoint),
          cq.Vector(*end_point),
      )
      interval = {
          "coordinate": "unique_short_arc",
          "start_radial": [
              _fraction_payload(value) for value in first_radial
          ],
          "end_radial": [
              _fraction_payload(value) for value in second_radial
          ],
          "orientation_sign": direction,
      }
      short_arc_proof = {
          "radial_cross_axis_fraction": _fraction_payload(signed_cross),
          "radial_dot_fraction": _fraction_payload(radial_dot),
          "unique_short_arc_strictly_below_pi": True,
          "atan2_authorizing": False,
      }
    interval_length = _q(float(analytic_edge.Length()))
    if interval_length <= 0:
      raise ValueError("source analytic segment interval is empty")
    public = {
        "edge_index": edge_index,
        "start_vertex": first_index,
        "end_vertex": second_index,
        "kind": "axial" if axial else "circular",
        "direction_sign": direction,
        "interval": interval,
        "interval_length_mm_fraction": _fraction_payload(interval_length),
        "short_arc_proof": short_arc_proof,
    }
    public["segment_commitment_sha256"] = _sha(public)
    segments.append({
        **public,
        "start_radial_signature": _radial_signature(first_radial),
        "end_radial_signature": _radial_signature(second_radial),
        "start_axial": first_axial,
        "end_axial": second_axial,
        "start_point": start_point,
        "end_point": end_point,
        "analytic_edge": analytic_edge,
        "interval_length_fraction": interval_length,
    })

  groups = _validate_source_segment_cycle(loop, segments)
  chains = []
  for ordinal, group in enumerate(groups):
    segment_indices = group["segment_indices"]
    chain_segments = [segments[index] for index in segment_indices]
    vertex_indices = [
        chain_segments[0]["start_vertex"],
        *[segment["end_vertex"] for segment in chain_segments],
    ]
    chain_length = sum((
        segment["interval_length_fraction"] for segment in chain_segments
    ), Fraction())
    chains.append({
        "ordinal": ordinal,
        "kind": group["kind"],
        "direction_sign": group["direction_sign"],
        "segment_indices": segment_indices,
        "segments": chain_segments,
        "vertex_indices": vertex_indices,
        "start_point": chain_segments[0]["start_point"],
        "end_point": chain_segments[-1]["end_point"],
        "interval_length_fraction": chain_length,
        "analytic_boundary": cq.Compound.makeCompound([
            segment["analytic_edge"] for segment in chain_segments
        ]),
    })

  coverage_payload = {
      "boundary_cycle_vertex_indices": loop,
      "segment_commitments": [
          segment["segment_commitment_sha256"] for segment in segments
      ],
      "vertex_endpoint_use_counts": {
          str(index): 2 for index in loop
      },
  }
  chain_receipts = []
  for chain in chains:
    chain_receipts.append({
        "ordinal": chain["ordinal"],
        "kind": chain["kind"],
        "direction_sign": chain["direction_sign"],
        "vertex_indices": chain["vertex_indices"],
        "segment_indices": chain["segment_indices"],
        "segment_intervals": [
            {
                key: value
                for key, value in segment.items()
                if key in {
                    "edge_index", "start_vertex", "end_vertex", "kind",
                    "direction_sign", "interval",
                    "interval_length_mm_fraction",
                    "short_arc_proof", "segment_commitment_sha256",
                }
            }
            for segment in chain["segments"]
        ],
        "chain_interval_length_mm_fraction": _fraction_payload(
            chain["interval_length_fraction"]
        ),
    })
  return {
      "vertices": vertices,
      "triangles": triangles,
      "loop": loop,
      "segments": segments,
      "chains": chains,
      "receipt": {
          "topology_payload_sha256": source_view["topology_payload_sha256"],
          "triangle_count": len(triangles),
          "unique_vertex_count": used_vertex_count,
          "boundary_edge_use_count": len(boundary),
          "loop_count": 1,
          "hole_count": 0,
          "orientation_consistent": True,
          "self_crossing_rejected_by": (
              "oriented_manifold_disk_full_segment_monotone_cylinder_rectangle"
          ),
          "axial_chain_count": 2,
          "circular_arc_chain_count": 2,
          "full_boundary_segment_count": len(segments),
          "all_vertices_have_two_adjacent_segment_uses": True,
          "intervals_contiguous_without_gap_overlap_or_reversal": True,
          "period_branch_unique_subject_to_occ_full_interval_match": True,
          "chains": chain_receipts,
          "coverage_commitment_sha256": _sha(coverage_payload),
      },
  }


def _source_trim_bbox_proof(
    source_view: Mapping[str, Any],
    source_domain: Mapping[str, Any],
    *,
    bbox_tolerance: Fraction,
    coordinate_guard: Any,
) -> dict[str, Any]:
  declared = source_view.get("declared_face_bbox_mm")
  if (
      not isinstance(declared, Mapping)
      or set(declared) != {"min", "max"}
      or any(len(declared[key]) != 3 for key in ("min", "max"))
  ):
    raise ValueError("source declared trim bbox differs")

  covered = []
  per_segment = []
  for segment in source_domain["segments"]:
    bbox = segment["analytic_edge"].BoundingBox()
    minimum = (float(bbox.xmin), float(bbox.ymin), float(bbox.zmin))
    maximum = (float(bbox.xmax), float(bbox.ymax), float(bbox.zmax))
    guard, guard_receipt = coordinate_guard((minimum, maximum))
    per_segment.append({
        "segment": segment,
        "guard": guard,
        "minimum": tuple(_q(value) for value in minimum),
        "maximum": tuple(_q(value) for value in maximum),
    })
    covered.append({
        "source_segment_edge_index": segment["edge_index"],
        "segment_commitment_sha256": segment["segment_commitment_sha256"],
        "complete_analytic_edge_bbox_min_mm_fraction": [
            _fraction_payload(_q(value)) for value in minimum
        ],
        "complete_analytic_edge_bbox_max_mm_fraction": [
            _fraction_payload(_q(value)) for value in maximum
        ],
        "coordinate_numeric_guard_v1": guard_receipt,
    })
  if len(per_segment) != len(source_domain["segments"]):
    raise ValueError("source analytic trim bbox segment coverage differs")

  axes = []
  for axis in range(3):
    analytic_min_lower = min(
        row["minimum"][axis] - row["guard"]
        for row in per_segment
    )
    analytic_min_upper = min(
        row["minimum"][axis] + row["guard"]
        for row in per_segment
    )
    analytic_max_lower = max(
        row["maximum"][axis] - row["guard"]
        for row in per_segment
    )
    analytic_max_upper = max(
        row["maximum"][axis] + row["guard"]
        for row in per_segment
    )
    declared_min = _q(float(declared["min"][axis]))
    declared_max = _q(float(declared["max"][axis]))
    if not (
        declared_min - bbox_tolerance <= analytic_min_lower
        and analytic_min_upper <= declared_min + bbox_tolerance
        and declared_max - bbox_tolerance <= analytic_max_lower
        and analytic_max_upper <= declared_max + bbox_tolerance
    ):
      raise ValueError(
          "declared source face bbox differs from complete analytic trim domain"
      )
    axes.append({
        "axis_ordinal": axis,
        "declared_min_mm_fraction": _fraction_payload(declared_min),
        "declared_max_mm_fraction": _fraction_payload(declared_max),
        "analytic_min_interval_mm_fraction": {
            "lower": _fraction_payload(analytic_min_lower),
            "upper": _fraction_payload(analytic_min_upper),
        },
        "analytic_max_interval_mm_fraction": {
            "lower": _fraction_payload(analytic_max_lower),
            "upper": _fraction_payload(analytic_max_upper),
        },
        "bidirectional_locked_tolerance_predicate": True,
    })
  coverage = {
      "segment_commitments": [
          row["segment_commitment_sha256"] for row in covered
      ],
      "segment_count": len(covered),
  }
  return {
      "method": (
          "complete_source_analytic_boundary_edge_bbox_intervals.v1"
      ),
      "source_support_bbox_authorizes_trim": False,
      "complete_source_boundary_segment_count": len(covered),
      "all_source_boundary_segments_covered": True,
      "locked_bbox_tolerance_mm_fraction": _fraction_payload(bbox_tolerance),
      "axes": axes,
      "segment_sampling_receipts": covered,
      "coverage_commitment_sha256": _sha(coverage),
      "declared_bbox_bidirectionally_matches_analytic_trim_domain": True,
  }


def _uv_close(
    first: Sequence[float], second: Sequence[float], tolerance: float,
) -> bool:
  qtolerance = _q(tolerance)
  return all(
      abs(_q(left) - _q(right)) <= qtolerance
      for left, right in zip(first, second, strict=True)
  )


def _interval_close(
    first: Sequence[Fraction],
    second: Sequence[Fraction],
    tolerance: Fraction,
) -> bool:
  return all(
      abs(left - right) <= tolerance
      for left, right in zip(first, second, strict=True)
  )


def _validate_occ_rectangle_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parameter_confusion: float,
    u_period: float,
) -> dict[str, Any]:
  if len(rows) != 4:
    raise ValueError("OCC rectangle is missing a boundary edge")
  tolerance = _q(parameter_confusion)
  if Counter(str(row.get("kind")) for row in rows) != {
      "axial": 2, "circular": 2
  }:
    raise ValueError("OCC rectangle lacks two constant-u/two constant-v edges")
  for index, row in enumerate(rows):
    if not _uv_close(
        row["uv_end"],
        rows[(index + 1) % len(rows)]["uv_start"],
        parameter_confusion,
    ):
      raise ValueError("OCC oriented pcurve endpoints do not close cyclically")

  axial = [row for row in rows if row["kind"] == "axial"]
  circular = [row for row in rows if row["kind"] == "circular"]
  axial_u = [_q(row["uv_start"][0]) for row in axial]
  circular_v = [_q(row["uv_start"][1]) for row in circular]
  if abs(axial_u[0] - axial_u[1]) <= tolerance or abs(
      circular_v[0] - circular_v[1]
  ) <= tolerance:
    raise ValueError("OCC rectangle parameter bounds are not distinct")
  axial_v_intervals = [
      sorted((_q(row["uv_start"][1]), _q(row["uv_end"][1])))
      for row in axial
  ]
  circular_u_intervals = [
      sorted((_q(row["uv_start"][0]), _q(row["uv_end"][0])))
      for row in circular
  ]
  period = _q(u_period)
  if period <= 0:
    raise ValueError("OCC cylinder U period differs")
  for interval in circular_u_intervals:
    span = interval[1] - interval[0]
    if (
        span <= tolerance
        or span >= period - tolerance
        or abs(2 * span - period) <= tolerance
    ):
      raise ValueError("OCC cylinder period branch is ambiguous")
  if not _interval_close(
      axial_v_intervals[0], axial_v_intervals[1], tolerance
  ):
    raise ValueError("OCC opposite axial intervals do not correspond")
  if not _interval_close(
      circular_u_intervals[0], circular_u_intervals[1], tolerance
  ):
    raise ValueError("OCC opposite angular intervals do not correspond")

  signed_twice_area = sum((
      _q(row["uv_start"][0]) * _q(row["uv_end"][1])
      - _q(row["uv_end"][0]) * _q(row["uv_start"][1])
      for row in rows
  ), Fraction())
  if signed_twice_area <= tolerance * tolerance:
    raise ValueError("OCC pcurve rectangle is not a positive outer cycle")
  public_rows = [
      {
          key: row[key]
          for key in (
              "edge_use_ordinal", "topological_orientation", "kind",
              "pcurve_first_parameter", "pcurve_last_parameter",
              "uv_start", "uv_end", "interval_length_mm_fraction",
          )
      }
      for row in rows
  ]
  return {
      "positive_outer_cycle": True,
      "cyclic_endpoint_closure": True,
      "opposite_intervals_correspond": True,
      "u_bounds_distinct": True,
      "v_bounds_distinct": True,
      "u_period_fraction": _fraction_payload(period),
      "period_branch_unique_and_not_half_turn": True,
      "signed_twice_uv_area_fraction": _fraction_payload(
          signed_twice_area
      ),
      "oriented_edge_intervals": public_rows,
      "endpoint_interval_commitment_sha256": _sha(public_rows),
  }


def _occ_domain(occ_replay: Mapping[str, Any]) -> dict[str, Any]:
  import cadquery as cq  # type: ignore
  from OCP.BRepAdaptor import (  # type: ignore
      BRepAdaptor_Curve2d,
      BRepAdaptor_Surface,
  )
  from OCP.GeomAbs import (  # type: ignore
      GeomAbs_Cylinder,
      GeomAbs_Line,
  )
  from OCP.Precision import Precision  # type: ignore
  from OCP.TopAbs import TopAbs_REVERSED  # type: ignore

  face = occ_replay["face"]
  wires = tuple(face.Wires())
  if len(wires) != 1:
    raise ValueError("OCC cylinder trim must contain exactly one wire and no holes")
  surface = BRepAdaptor_Surface(face.wrapped)
  if surface.GetType() != GeomAbs_Cylinder:
    raise ValueError("selected OCC face is not a cylinder")
  edge_uses = _occ_wire_edge_uses(face, cq)
  wrapped_uses = [edge.wrapped for edge in edge_uses]
  if len(edge_uses) != 4:
    raise ValueError("OCC cylinder trim must contain four edge uses")
  for left in range(len(wrapped_uses)):
    for right in range(left + 1, len(wrapped_uses)):
      if wrapped_uses[left].IsSame(wrapped_uses[right]):
        raise ValueError("OCC cylinder trim contains a repeated seam/edge use")

  pconfusion = float(
      (getattr(Precision, "PConfusion", None)
       or getattr(Precision, "PConfusion_s"))()
  )
  cylinder = surface.Cylinder()
  radius = float(cylinder.Radius())
  if not math.isfinite(radius) or radius <= 0.0:
    raise ValueError("OCC cylinder radius differs")
  rows = []
  for edge_ordinal, (edge, wrapped) in enumerate(
      zip(edge_uses, wrapped_uses, strict=True)
  ):
    curve = BRepAdaptor_Curve2d(wrapped, face.wrapped)
    if curve.GetType() != GeomAbs_Line:
      raise ValueError("OCC cylinder trim contains a non-linear pcurve")
    start_parameter = float(curve.FirstParameter())
    end_parameter = float(curve.LastParameter())
    start = curve.Value(start_parameter)
    end = curve.Value(end_parameter)
    raw_uv0 = (float(start.X()), float(start.Y()))
    raw_uv1 = (float(end.X()), float(end.Y()))
    reversed_use = wrapped.Orientation() == TopAbs_REVERSED
    uv0, uv1 = (
        (raw_uv1, raw_uv0) if reversed_use else (raw_uv0, raw_uv1)
    )
    du = _q(uv1[0]) - _q(uv0[0])
    dv = _q(uv1[1]) - _q(uv0[1])
    tolerance = _q(pconfusion)
    axial = du * du <= tolerance * tolerance and (
        dv * dv > tolerance * tolerance
    )
    angular = dv * dv <= tolerance * tolerance and (
        du * du > tolerance * tolerance
    )
    if axial == angular:
      raise ValueError("OCC pcurve is not an axial/angular rectangle boundary")
    expected_geom = "LINE" if axial else "CIRCLE"
    if str(edge.geomType()).upper() != expected_geom:
      raise ValueError("OCC pcurve/3D boundary geometry types disagree")
    interval_length = (
        abs(dv) if axial else _q(radius) * abs(du)
    )
    if interval_length <= 0:
      raise ValueError("OCC pcurve interval is empty")
    rows.append({
        "edge_use_ordinal": edge_ordinal,
        "topological_orientation": "REVERSED" if reversed_use else "FORWARD",
        "kind": "axial" if axial else "circular",
        "edge": edge,
        "wrapped": wrapped,
        "uv_start": uv0,
        "uv_end": uv1,
        "pcurve_first_parameter": start_parameter,
        "pcurve_last_parameter": end_parameter,
        "interval_length_fraction": interval_length,
        "interval_length_mm_fraction": _fraction_payload(interval_length),
        "length_mm": float(edge.Length()),
    })
  if Counter(row["kind"] for row in rows) != {
      "axial": 2, "circular": 2
  }:
    raise ValueError("OCC pcurve rectangle lacks 2 axial and 2 angular sides")
  rectangle = _validate_occ_rectangle_rows(
      rows,
      parameter_confusion=pconfusion,
      u_period=float(surface.UPeriod()),
  )
  axis = cylinder.Axis()
  axis_point = _vec(axis.Location())
  axis_direction = _unit(_vec(axis.Direction()), label="OCC cylinder axis")
  return {
      "rows": rows,
      "surface": surface,
      "support": {
          "axis_point_mm": axis_point,
          "axis_direction": axis_direction,
          "radius_mm": radius,
      },
      "receipt": {
          "wire_count": 1,
          "hole_count": 0,
          "edge_use_count": 4,
          "unique_edge_use_count": 4,
          "repeated_seam_count": 0,
          "pcurve_type_counts": {"line": 4},
          "pcurve_boundary_counts": {"axial": 2, "angular": 2},
          "pcurve_parameter_confusion": pconfusion,
          "face_area_mm2_exact_occ_observation": float(face.Area()),
          "edge_lengths_mm_exact_occ_observations": [
              row["length_mm"] for row in rows
          ],
          "rectangle_proof": rectangle,
      },
  }


def _support_equivalence(
    source: Mapping[str, Any], occ: Mapping[str, Any],
    *, tolerance: float, angular_tolerance: float,
) -> dict[str, Any]:
  source_axis = tuple(float(x) for x in source["axis_direction"])
  occ_axis = tuple(float(x) for x in occ["axis_direction"])
  source_axis_sq = _qnorm_sq(source_axis)
  occ_axis_sq = _qnorm_sq(occ_axis)
  axis_cross_sq = _qnorm_sq(_qcross(source_axis, occ_axis))
  angular_tolerance_q = _q(angular_tolerance)
  if axis_cross_sq > (
      angular_tolerance_q ** 2 * source_axis_sq * occ_axis_sq
  ):
    raise ValueError("source/OCC cylinder axes differ")
  radius_delta = _q(float(source["radius_mm"])) - _q(float(occ["radius_mm"]))
  if radius_delta * radius_delta > _q(tolerance) ** 2:
    raise ValueError("source/OCC cylinder radii differ")
  displacement = _qsub(
      tuple(float(x) for x in occ["axis_point_mm"]),
      tuple(float(x) for x in source["axis_point_mm"]),
  )
  line_distance_cross = _qcross(displacement, source_axis)
  if _qnorm_sq(line_distance_cross) > (
      _q(tolerance) ** 2 * source_axis_sq
  ):
    raise ValueError("source/OCC cylinder axis lines differ")
  return {
      "axis_parallel_squared_fraction_predicate": True,
      "axis_line_distance_squared_fraction_predicate": True,
      "radius_interval_fraction_predicate": True,
  }


def _distance_point_edge(point: Sequence[float], edge: Any) -> float:
  import cadquery as cq  # type: ignore
  value = float(cq.Vertex.makeVertex(*point).distance(edge))
  if not math.isfinite(value) or value < 0.0:
    raise ValueError("OCC exact point/edge distance differs")
  return value


def _segment_count_for_gap(length: Fraction, *, locked: Fraction) -> int:
  ratio = length / locked
  return max(1, -(-ratio.numerator // ratio.denominator))


def _source_segment_samples(
    segment: Mapping[str, Any],
    *,
    locked: Fraction,
) -> tuple[list[tuple[float, float, float]], Fraction, dict[str, Any]]:
  from OCP.BRepAdaptor import BRepAdaptor_Curve  # type: ignore
  from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line  # type: ignore

  edge = segment["analytic_edge"]
  curve = BRepAdaptor_Curve(edge.wrapped)
  if curve.GetType() not in {GeomAbs_Line, GeomAbs_Circle}:
    raise ValueError("source analytic segment curve type differs")
  first = float(curve.FirstParameter())
  last = float(curve.LastParameter())
  length = segment["interval_length_fraction"]
  count = _segment_count_for_gap(length, locked=locked)
  points = [
      _vec(curve.Value(first + (last - first) * index / count))
      for index in range(count + 1)
  ]
  gap = length / count
  return points, gap, {
      "sampling_parameter_first_observation": first,
      "sampling_parameter_last_observation": last,
      "sampling_interval_authority": (
          "OCC_exact_line_or_circle_length_divided_by_segment_count"
      ),
  }


def _occ_edge_samples(
    row: Mapping[str, Any], surface: Any, *, locked: Fraction,
) -> tuple[list[tuple[float, float, float]], Fraction, dict[str, Any]]:
  start = row["uv_start"]
  end = row["uv_end"]
  length = row["interval_length_fraction"]
  count = _segment_count_for_gap(length, locked=locked)
  points = []
  for index in range(count + 1):
    fraction = index / count
    u = float(start[0] + (end[0] - start[0]) * fraction)
    v = float(start[1] + (end[1] - start[1]) * fraction)
    points.append(_vec(surface.Value(u, v)))
  gap = length / count
  return points, gap, {
      "sampling_uv_start": list(start),
      "sampling_uv_end": list(end),
      "sampling_interval_authority": (
          "linear_pcurve_line_length_or_cylinder_radius_times_delta_u_fraction"
      ),
  }


def _lipschitz_distance_proof(
    *,
    points: Sequence[Sequence[float]],
    exact_gap: Fraction,
    target: Any,
    interval: Mapping[str, Any],
    locked: Fraction,
    numeric_guard: Any,
) -> dict[str, Any]:
  if len(points) < 2 or exact_gap <= 0:
    raise ValueError("analytic interval sampling coverage differs")
  sampled = max(_distance_point_edge(point, target) for point in points)
  outward_sample, guard, guard_receipt = numeric_guard(sampled, points)
  upper = outward_sample + guard + exact_gap / 2
  if upper > locked:
    raise ValueError("analytic interval Lipschitz upper bound exceeds 0.2 mm")
  unsigned = _json_data({
      "sample_count": len(points),
      "covered_subinterval_count": len(points) - 1,
      "complete_parameter_interval": _json_data(interval),
      "exact_maximum_uncovered_arc_length_gap_mm_fraction": (
          _fraction_payload(exact_gap)
      ),
      "maximum_occ_exact_sample_distance_mm_nextafter_ratio": (
          _fraction_payload(outward_sample)
      ),
      "kernel_numeric_guard_v1": guard_receipt,
      "exact_kernel_numeric_guard_mm_fraction": _fraction_payload(guard),
      "certified_upper_mm_fraction": _fraction_payload(upper),
      "locked_boundary_distance_mm_fraction": _fraction_payload(locked),
      "fraction_comparison_accepted": True,
      "distance_theorem_id": (
          "distance_to_nonempty_closed_set_is_1_lipschitz.v1"
      ),
      "float_addition_or_nextafter_authorizes": False,
      "raw_kernel_float_authorizes_without_guard": False,
  })
  return {**unsigned, "proof_commitment_sha256": _sha(unsigned)}


def _occ_endpoint_points(
    row: Mapping[str, Any], surface: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
  return (
      _vec(surface.Value(*row["uv_start"])),
      _vec(surface.Value(*row["uv_end"])),
  )


def _endpoint_relation(
    chain: Mapping[str, Any],
    row: Mapping[str, Any],
    surface: Any,
) -> tuple[Fraction, str]:
  occ_start, occ_end = _occ_endpoint_points(row, surface)
  direct = max(
      _point_distance_sq(chain["start_point"], occ_start),
      _point_distance_sq(chain["end_point"], occ_end),
  )
  reverse = max(
      _point_distance_sq(chain["start_point"], occ_end),
      _point_distance_sq(chain["end_point"], occ_start),
  )
  if direct == reverse:
    raise ValueError("source/OCC endpoint orientation is ambiguous")
  return (
      (direct, "same") if direct < reverse else (reverse, "reversed")
  )


def _boundary_correspondence(
    source: Mapping[str, Any],
    occ: Mapping[str, Any],
    *,
    interval_tolerance_mm: float,
    locked: Fraction,
    numeric_guard: Any,
) -> dict[str, Any]:
  def evaluate(
      axial_permutation: tuple[int, int],
      circular_permutation: tuple[int, int],
  ) -> dict[str, Any]:
    selected = []
    endpoint_costs = []
    for kind, permutation in (
        ("axial", axial_permutation),
        ("circular", circular_permutation),
    ):
      source_rows = [
          row for row in source["chains"] if row["kind"] == kind
      ]
      occ_rows = [row for row in occ["rows"] if row["kind"] == kind]
      if len(source_rows) != 2 or len(occ_rows) != 2:
        raise ValueError("source/OCC typed boundary cardinality differs")
      for source_index, occ_index in enumerate(permutation):
        endpoint_cost, orientation = _endpoint_relation(
            source_rows[source_index], occ_rows[occ_index], occ["surface"]
        )
        if endpoint_cost > locked ** 2:
          raise ValueError("source/OCC chain endpoints differ")
        endpoint_costs.append(endpoint_cost)
        selected.append((
            source_rows[source_index],
            occ_rows[occ_index],
            orientation,
            endpoint_cost,
        ))
    if {row[2] for row in selected} not in ({"same"}, {"reversed"}):
      raise ValueError("source/OCC chain orientations disagree globally")

    source_to_occ: list[dict[str, Any]] = []
    occ_to_source: list[dict[str, Any]] = []
    matches = []
    interval_tolerance = _q(interval_tolerance_mm)
    for chain, row, orientation, endpoint_cost in selected:
      length_delta = abs(
          chain["interval_length_fraction"]
          - row["interval_length_fraction"]
      )
      if length_delta > interval_tolerance:
        raise ValueError("source/OCC full chain intervals differ")
      chain_source_proofs = []
      for segment in chain["segments"]:
        points, exact_gap, sampling = _source_segment_samples(
            segment, locked=locked
        )
        proof = _lipschitz_distance_proof(
            points=points,
            exact_gap=exact_gap,
            target=row["edge"],
            interval={
                "source_chain_ordinal": chain["ordinal"],
                "source_segment_edge_index": segment["edge_index"],
                "segment_interval": segment["interval"],
                **sampling,
            },
            locked=locked,
            numeric_guard=numeric_guard,
        )
        chain_source_proofs.append(proof)
        source_to_occ.append(proof)
      points, exact_gap, sampling = _occ_edge_samples(
          row, occ["surface"], locked=locked
      )
      reverse_proof = _lipschitz_distance_proof(
          points=points,
          exact_gap=exact_gap,
          target=chain["analytic_boundary"],
          interval={
              "occ_edge_use_ordinal": row["edge_use_ordinal"],
              "uv_start": list(row["uv_start"]),
              "uv_end": list(row["uv_end"]),
              **sampling,
          },
          locked=locked,
          numeric_guard=numeric_guard,
      )
      occ_to_source.append(reverse_proof)
      matches.append({
          "source_chain_kind": chain["kind"],
          "source_chain_ordinal": chain["ordinal"],
          "occ_edge_use_ordinal": row["edge_use_ordinal"],
          "orientation_relation": orientation,
          "endpoint_maximum_squared_distance_fraction": _fraction_payload(
              endpoint_cost
          ),
          "source_chain_interval_length_mm_fraction": _fraction_payload(
              chain["interval_length_fraction"]
          ),
          "occ_edge_interval_length_mm_fraction": _fraction_payload(
              row["interval_length_fraction"]
          ),
          "interval_length_delta_mm_fraction": _fraction_payload(length_delta),
          "source_segment_proof_commitments": [
              proof["proof_commitment_sha256"]
              for proof in chain_source_proofs
          ],
          "occ_edge_reverse_proof_commitment": (
              reverse_proof["proof_commitment_sha256"]
          ),
      })
    if (
        len(source_to_occ) != len(source["segments"])
        or len(occ_to_source) != 4
    ):
      raise ValueError("bidirectional analytic interval proof coverage differs")
    coverage = {
        "source_segment_proof_commitments": [
            proof["proof_commitment_sha256"] for proof in source_to_occ
        ],
        "occ_edge_proof_commitments": [
            proof["proof_commitment_sha256"] for proof in occ_to_source
        ],
    }
    maximum_upper = max(
        _fraction_from_payload(proof["certified_upper_mm_fraction"])
        for proof in (*source_to_occ, *occ_to_source)
    )
    return {
        "matching": matches,
        "global_orientation_relation": selected[0][2],
        "source_to_occ_full_segment_proofs": source_to_occ,
        "occ_to_source_full_edge_proofs": occ_to_source,
        "source_segment_proof_count": len(source_to_occ),
        "occ_edge_proof_count": len(occ_to_source),
        "maximum_certified_upper_mm_fraction": _fraction_payload(maximum_upper),
        "bidirectional_coverage_commitment_sha256": _sha(coverage),
        "assignment_endpoint_cost_telemetry_fraction": _fraction_payload(
            max(endpoint_costs)
        ),
    }

  admissible = []
  rejected = []
  for axial in ((0, 1), (1, 0)):
    for circular in ((0, 1), (1, 0)):
      identity = {
          "axial_permutation": list(axial),
          "circular_permutation": list(circular),
      }
      try:
        result = evaluate(axial, circular)
      except ValueError as error:
        rejected.append({**identity, "reason": str(error)})
      else:
        admissible.append({**identity, "result": result})
  selected = _require_exactly_one_admissible_assignment(admissible)
  result = selected["result"]
  return {
      **result,
      "unique_one_to_one_matching": True,
      "enumerated_typed_bijection_count": 4,
      "admissible_typed_bijection_count": 1,
      "selected_assignment": {
          "axial_permutation": selected["axial_permutation"],
          "circular_permutation": selected["circular_permutation"],
      },
      "rejected_assignment_telemetry": rejected,
      "cost_used_for_authorization": False,
      "locked_boundary_distance_mm_fraction": _fraction_payload(locked),
      "one_to_one_bidirectional_correspondence": True,
      "all_source_segments_and_occ_edges_covered": True,
  }


def _require_exactly_one_admissible_assignment(
    admissible: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
  if len(admissible) != 1:
    raise ValueError(
        "source/OCC boundary correspondence requires exactly one fully "
        "admissible typed bijection"
    )
  return admissible[0]


def _lineage_equivalence(
    source_binding: Mapping[str, Any],
    occ_binding: Mapping[str, Any],
) -> dict[str, Any]:
  source_paths = source_binding.get("path_metadata")
  occ_paths = occ_binding.get("source_path_identity")
  transform = occ_binding.get("source_transform")
  occurrence = occ_binding.get("source_body_occurrence_identity")
  step_identity = occ_binding.get("step_body_identity")
  if not all(isinstance(value, Mapping) for value in (
      source_paths, occ_paths, transform, occurrence, step_identity
  )):
    raise ValueError("source/OCC lineage binding schema differs")
  body_uuid = source_binding.get("body_uuid")
  if (
      source_binding.get("assembly_json_bytes_sha256")
      != occ_binding.get("source_assembly_bytes_sha256")
  ):
    raise ValueError("source/OCC assembly JSON lineage differs")
  if (
      not isinstance(body_uuid, str)
      or body_uuid != transform.get("body_uuid")
      or body_uuid != occurrence.get("body_uuid")
      or body_uuid != step_identity.get("body_uuid")
  ):
    raise ValueError("source/OCC/STEP body UUID lineage differs")
  if (
      list(transform.get("occurrence_path", ())) != []
      or list(occurrence.get("occurrence_path", ())) != []
      or transform.get("kind") != "root_identity"
      or occurrence.get("root_component_uuid")
      != transform.get("root_component_uuid")
  ):
    raise ValueError("source/OCC body occurrence identity differs")
  if (
      source_paths.get("obj_archive_member")
      != occ_paths.get("obj_archive_member")
      or source_paths.get("assembly_archive_member")
      != occ_paths.get("assembly_archive_member")
  ):
    raise ValueError("source/OCC archive path identity differs")
  expected_step = str(source_paths["obj_archive_member"])[:-4] + ".step"
  if (
      occ_paths.get("step_archive_member") != expected_step
      or step_identity.get("step_archive_member") != expected_step
      or step_identity.get("step_bytes_sha256")
      != occ_binding.get("step_bytes_sha256")
  ):
    raise ValueError("STEP bytes/body path identity differs")
  return {
      "source_assembly_json_sha256_matches_occ_capability": True,
      "source_body_uuid_matches_transform_occurrence_and_step": True,
      "source_obj_assembly_and_step_archive_path_identity_matches": True,
      "root_body_occurrence_identity_matches": True,
      "step_bytes_sha256_bound_to_body_identity": True,
      "body_uuid": body_uuid,
      "root_component_uuid": transform["root_component_uuid"],
      "occurrence_path": [],
      "path_identity": dict(occ_paths),
  }


def _derive_receipt(
    source: Any,
    support_certificate: Any,
    occ: Any,
    *,
    locked: Fraction,
    bbox_tolerance: Fraction,
    numeric_guard: Any,
    coordinate_guard: Any,
) -> dict[str, Any]:
  source_view = replay_source_obj_face_topology_v1(
      source, support_certificate
  )
  occ_replay = _replay_occ_capability(occ)
  lineage = _lineage_equivalence(
      source_view["source_binding"], occ_replay["binding"]
  )
  source_domain = _source_boundary(source_view)
  trim_bbox = _source_trim_bbox_proof(
      source_view,
      source_domain,
      bbox_tolerance=bbox_tolerance,
      coordinate_guard=coordinate_guard,
  )
  occ_domain = _occ_domain(occ_replay)
  support_tolerance = float(
      support_certificate.receipt["thresholds"]["support_tolerance_mm"]
  )
  angular_tolerance = float(
      support_certificate.receipt["thresholds"]["normal_alignment_tolerance"]
  )
  support_checks = _support_equivalence(
      source_view["support"],
      occ_domain["support"],
      tolerance=support_tolerance,
      angular_tolerance=angular_tolerance,
  )
  correspondence = _boundary_correspondence(
      source_domain,
      occ_domain,
      interval_tolerance_mm=support_tolerance,
      locked=locked,
      numeric_guard=numeric_guard,
  )
  unsigned = _json_data({
      "schema_version": SCHEMA_VERSION,
      "evidence_scope": EVIDENCE_SCOPE,
      "formal_authorized": False,
      "decision": "certified_equivalent",
      "source_binding": dict(source_view["source_binding"]),
      "occ_binding": occ_replay["binding"],
      "lineage_equivalence": lineage,
      "source_support_certificate_sha256": support_certificate.sha256,
      "source_domain": source_domain["receipt"],
      "source_trim_bbox_proof": trim_bbox,
      "occ_domain": occ_domain["receipt"],
      "support_equivalence": support_checks,
      "boundary_correspondence": correspondence,
      "decision_evidence": {
          "authorizing": [
              "sealed_source_bytes_topology_replay",
              "STEP_bytes_internal_sha256_ReadStream_reimport",
              "deterministic_face_index_and_internal_signature",
              "Fraction_squared_support_comparisons",
              "OCC_point_to_analytic_edge_with_locked_kernel_numeric_guard_v1",
              "single_wire_four_unique_linear_pcurve_topology",
          ],
          "non_authorizing_observations": [
              "atan2_canonical_period_observation",
              "source_analytic_area_from_angular_float",
          ],
          "area_and_length_authority": (
              "OCC_exact_face_and_edge_properties_after_domain_identity_proof"
          ),
          "source_absence_is_negative": False,
      },
      "thresholds": {
          "locked_boundary_distance_mm_fraction": _fraction_payload(locked),
          "source_support_tolerance_mm": support_tolerance,
          "new_chord_tolerance_introduced": False,
      },
  })
  return {**unsigned, "receipt_payload_sha256": _sha(unsigned)}


_RECEIPT_FIELDS = {
    "schema_version", "evidence_scope", "formal_authorized", "decision",
    "source_binding", "occ_binding", "lineage_equivalence",
    "source_support_certificate_sha256",
    "source_domain", "source_trim_bbox_proof", "occ_domain", "support_equivalence",
    "boundary_correspondence", "decision_evidence", "thresholds",
    "receipt_payload_sha256",
}


def _validated_receipt(receipt: Any) -> dict[str, Any]:
  if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
    raise ValueError("analytic trim receipt schema differs")
  try:
    result = _strict_json(
        _canonical_bytes(receipt), label="analytic trim receipt"
    )
  except (TypeError, ValueError) as error:
    raise ValueError("analytic trim receipt is not canonical JSON data") from error
  observed = result.pop("receipt_payload_sha256")
  if observed != _sha(result):
    raise ValueError("analytic trim receipt self-hash differs")
  result["receipt_payload_sha256"] = observed
  return result


def _sealed_authority_api():
  states: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
  seal = object()
  locked = Fraction(1, 5)
  bbox_tolerance = Fraction(1, 100_000)
  from OCP.Precision import Precision  # type: ignore
  precision_confusion = _q(float(
      (getattr(Precision, "Confusion", None)
       or getattr(Precision, "Confusion_s"))()
  ))

  def coordinate_guard(
      points: Sequence[Sequence[float]],
  ) -> tuple[Fraction, dict[str, Any]]:
    coordinate_ulp = max(
        (_q(math.ulp(float(value))) for point in points for value in point),
        default=Fraction(),
    )
    guard = 32 * precision_confusion + 128 * coordinate_ulp
    return guard, {
        "schema_version": "coordinate_numeric_guard.v1",
        "precision_confusion_mm_fraction": _fraction_payload(
            precision_confusion
        ),
        "maximum_coordinate_ulp_mm_fraction": _fraction_payload(
            coordinate_ulp
        ),
        "precision_confusion_multiplier": 32,
        "coordinate_ulp_multiplier": 128,
        "guard_mm_fraction": _fraction_payload(guard),
        "overrideable_by_caller": False,
    }

  def numeric_guard(
      sampled: float,
      points: Sequence[Sequence[float]],
  ) -> tuple[Fraction, Fraction, dict[str, Any]]:
    outward_float = math.nextafter(sampled, math.inf)
    if not math.isfinite(outward_float):
      raise ValueError("OCC sample distance outward rounding differs")
    outward = _q(outward_float)
    coordinate_ulp = max(
        (_q(math.ulp(float(value))) for point in points for value in point),
        default=Fraction(),
    )
    distance_ulp = _q(math.ulp(sampled))
    guard = (
        32 * precision_confusion
        + 128 * coordinate_ulp
        + 16 * distance_ulp
    )
    return outward, guard, {
        "schema_version": "kernel_numeric_guard.v1",
        "outward_rounding": "math.nextafter_toward_positive_infinity",
        "precision_confusion_mm_fraction": _fraction_payload(
            precision_confusion
        ),
        "maximum_coordinate_ulp_mm_fraction": _fraction_payload(
            coordinate_ulp
        ),
        "sample_distance_ulp_mm_fraction": _fraction_payload(distance_ulp),
        "precision_confusion_multiplier": 32,
        "coordinate_ulp_multiplier": 128,
        "sample_distance_ulp_multiplier": 16,
        "guard_mm_fraction": _fraction_payload(guard),
        "overrideable_by_caller": False,
    }

  class CylinderRectangularAnalyticTrimCertificateV1:
    __slots__ = ("__weakref__",)

    def __new__(cls, *, _seal: object | None = None):
      if _seal is not seal:
        raise TypeError("analytic trim certificates are verifier-factory-only")
      return super().__new__(cls)

    def __setattr__(self, _name: str, _value: Any) -> None:
      raise TypeError("analytic trim certificates are sealed")

    def __copy__(self):
      raise TypeError("analytic trim certificates cannot be copied")

    def __deepcopy__(self, _memo: Any):
      raise TypeError("analytic trim certificates cannot be deep-copied")

    def __reduce_ex__(self, _protocol: int):
      raise TypeError("analytic trim certificates are not serializable")

    @property
    def receipt(self) -> Mapping[str, Any]:
      return states[self]["receipt"]

    @property
    def canonical_json(self) -> str:
      return str(states[self]["canonical_json"])

    @property
    def sha256(self) -> str:
      return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

  def mint(receipt: Mapping[str, Any]):
    value = CylinderRectangularAnalyticTrimCertificateV1(_seal=seal)
    states[value] = MappingProxyType({
        "receipt": _freeze(receipt),
        "canonical_json": _canonical_bytes(receipt).decode("utf-8"),
    })
    return value

  def certify(source: Any, support_certificate: Any, occ: Any):
    expected = _derive_receipt(
        source,
        support_certificate,
        occ,
        locked=locked,
        bbox_tolerance=bbox_tolerance,
        numeric_guard=numeric_guard,
        coordinate_guard=coordinate_guard,
    )
    return mint(expected)

  def verify(
      receipt: Mapping[str, Any],
      source: Any,
      support_certificate: Any,
      occ: Any,
  ):
    observed = _validated_receipt(receipt)
    expected = _derive_receipt(
        source,
        support_certificate,
        occ,
        locked=locked,
        bbox_tolerance=bbox_tolerance,
        numeric_guard=numeric_guard,
        coordinate_guard=coordinate_guard,
    )
    if observed != expected:
      raise ValueError("analytic trim receipt differs from complete byte replay")
    return mint(expected)

  return CylinderRectangularAnalyticTrimCertificateV1, certify, verify


(
    CylinderRectangularAnalyticTrimCertificateV1,
    certify_cylinder_rectangular_analytic_trim_v1,
    verify_cylinder_rectangular_analytic_trim_v1,
) = _sealed_authority_api()
del _sealed_authority_api


__all__ = [
    "CylinderRectangularAnalyticTrimCertificateV1",
    "EVIDENCE_SCOPE",
    "LOCKED_BOUNDARY_DISTANCE_MM",
    "OccCylinderTrimDomainCapabilityV1",
    "SCHEMA_VERSION",
    "capture_occ_cylinder_trim_domain_v1",
    "certify_cylinder_rectangular_analytic_trim_v1",
    "verify_cylinder_rectangular_analytic_trim_v1",
]
