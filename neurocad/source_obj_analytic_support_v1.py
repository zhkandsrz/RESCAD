"""Source-bound analytic-support recovery from OBJ and assembly bytes.

The evidence produced here is development-only and is not a face-map
authority.  Source geometry can enter only through ``capture_source_obj_face_v1``;
the capture hashes and parses the OBJ and assembly bytes itself.  Acceptance
decisions replay finite Python floats as exact integer-ratio rationals and use
squared inequalities.  Decimal is used only for stable human-readable
observations, never to turn an inexact float observation into a bound.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary


SCHEMA_VERSION = "source_obj_analytic_support_certificate.v1"
EVIDENCE_SCOPE = "development_source_obj_support_only_no_mapper_authority"
NUMERIC_METHOD = "decimal_replay_v1"
DECIMAL_PRECISION = 120
DECISIVE_ENGINE = "exact_float_integer_ratio_squared_inequalities.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NORMAL_UNIT_TOLERANCE = 1.0e-9
_NORMAL_ALIGNMENT_TOLERANCE = 1.0e-8
_AXIS_NORMAL_TOLERANCE = 1.0e-8
_MIN_AREA_RATIO_SQUARED = 1.0e-24
_MIN_INDEPENDENT_NORMAL_CROSS_SQUARED = 1.0e-6
_MIN_SOLVE_PIVOT_RATIO = 1.0e-10
_MIN_NORMAL_COVERAGE_DETERMINANT_RATIO = 5.0e-2


def canonical_sha256_v1(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")


def _json_without_duplicates(raw: bytes, *, label: str) -> Any:
  def pairs(values: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
      if key in result:
        raise ValueError(f"{label} contains a duplicate JSON key")
      result[key] = value
    return result

  try:
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite JSON number {value}")
        ),
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _finite_float(value: Any, *, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{label} must be a finite real number")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{label} must be finite")
  return 0.0 if result == 0.0 else result


def _parse_float(value: str, *, label: str) -> float:
  try:
    result = float(value)
  except ValueError as error:
    raise ValueError(f"{label} is not a decimal float") from error
  if not math.isfinite(result):
    raise ValueError(f"{label} must be finite")
  return 0.0 if result == 0.0 else result


def _point(value: Any, *, label: str) -> tuple[float, float, float]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} schema differs")
  fields = set(value)
  if fields == {"type", "x", "y", "z"} and value.get("type") == "Point3D":
    pass
  elif fields == {"x", "y", "z"}:
    pass
  else:
    raise ValueError(f"{label} schema differs")
  return tuple(
      _finite_float(value[key], label=f"{label}.{key}") for key in ("x", "y", "z")
  )  # type: ignore[return-value]


def _bbox(value: Any, *, label: str) -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
  if (
      not isinstance(value, Mapping)
      or set(value) != {"type", "min_point", "max_point"}
      or value.get("type") != "BoundingBox3D"
  ):
    raise ValueError(f"{label} schema differs")
  minimum = _point(value["min_point"], label=f"{label}.min_point")
  maximum = _point(value["max_point"], label=f"{label}.max_point")
  if any(low > high for low, high in zip(minimum, maximum, strict=True)):
    raise ValueError(f"{label} has an inverted extent")
  return minimum, maximum


def _unit_binding(value: Any) -> dict[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {
      "source_length_unit", "millimeters_per_source_unit",
  }:
    raise ValueError("unit binding schema differs")
  name = value.get("source_length_unit")
  scale = _finite_float(
      value.get("millimeters_per_source_unit"),
      label="millimeters_per_source_unit",
  )
  if not isinstance(name, str) or not name or scale <= 0.0:
    raise ValueError("unit binding semantics differ")
  return {
      "source_length_unit": name,
      "millimeters_per_source_unit": scale,
  }


def _path_metadata(value: Any) -> dict[str, str]:
  if not isinstance(value, Mapping) or set(value) != {
      "obj_archive_member", "assembly_archive_member",
  }:
    raise ValueError("path metadata schema differs")
  result = {}
  for key in ("obj_archive_member", "assembly_archive_member"):
    member = value.get(key)
    if not isinstance(member, str) or not member or "\\" in member:
      raise ValueError("path metadata must use non-empty POSIX members")
    parsed = PurePosixPath(member)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
      raise ValueError("path metadata escapes its archive")
    result[key] = parsed.as_posix()
  if not result["obj_archive_member"].lower().endswith(".obj"):
    raise ValueError("OBJ path metadata suffix differs")
  if not result["assembly_archive_member"].lower().endswith(".json"):
    raise ValueError("assembly path metadata suffix differs")
  return result


def _canonical_sign(value: Sequence[float]) -> tuple[float, float, float]:
  cleaned = tuple(
      0.0 if abs(component) <= 1.0e-15 else float(component)
      for component in value
  )
  for component in cleaned:
    if component:
      if component < 0.0:
        return tuple(
            0.0 if item == 0.0 else -item for item in cleaned
        )  # type: ignore[return-value]
      break
  return cleaned  # type: ignore[return-value]


def _parse_obj_face(
    raw: bytes, *, face_index: int,
) -> tuple[tuple[tuple[
    tuple[float, float, float], tuple[float, float, float]
], ...], ...]:
  try:
    text = raw.decode("utf-8")
  except UnicodeDecodeError as error:
    raise ValueError("OBJ bytes are not strict UTF-8") from error
  if "\x00" in text:
    raise ValueError("OBJ bytes contain NUL")
  vertices: list[tuple[float, float, float]] = []
  normals: list[tuple[float, float, float]] = []
  triangles = []
  current_group: tuple[str, ...] = ()
  target_seen = False
  for line_number, line in enumerate(text.splitlines(), start=1):
    fields = line.split()
    if not fields or fields[0].startswith("#"):
      continue
    code = fields[0]
    if code in {"v", "vn"}:
      if len(fields) != 4:
        raise ValueError(f"OBJ {code} arity differs at line {line_number}")
      parsed = tuple(
          _parse_float(value, label=f"OBJ {code} line {line_number}")
          for value in fields[1:]
      )
      (vertices if code == "v" else normals).append(parsed)  # type: ignore[arg-type]
    elif code == "g":
      current_group = tuple(fields[1:])
      if current_group == ("face", str(face_index)):
        target_seen = True
    elif code == "f" and current_group == ("face", str(face_index)):
      if len(fields) != 4:
        raise ValueError("target OBJ face group contains a non-triangle")
      corners = []
      for token in fields[1:]:
        indices = token.split("/")
        if len(indices) != 3 or not indices[0] or not indices[2]:
          raise ValueError("target OBJ face corner lacks vertex/normal index")
        try:
          vertex_index = int(indices[0])
          normal_index = int(indices[2])
        except ValueError as error:
          raise ValueError("target OBJ face index is not an integer") from error
        if (
            vertex_index <= 0 or normal_index <= 0
            or vertex_index > len(vertices) or normal_index > len(normals)
        ):
          raise ValueError("target OBJ face index is outside prior definitions")
        corners.append((vertices[vertex_index - 1], normals[normal_index - 1]))
      corners.sort(key=lambda item: (item[0], _canonical_sign(item[1])))
      triangles.append(tuple(corners))
  if not target_seen or not triangles:
    raise ValueError("target OBJ face group is absent or empty")
  triangles.sort(key=lambda triangle: tuple(
      (vertex, _canonical_sign(normal)) for vertex, normal in triangle
  ))
  return tuple(triangles)


def _parse_obj_face_topology(
    raw: bytes, *, face_index: int, scale_mm: float,
) -> dict[str, Any]:
  """Replay the target OBJ triangle indices without weakening the sealed input.

  The support recovery above deliberately canonicalizes triangle corners.  A
  trim-domain proof additionally needs the original directed triangle uses, so
  this second parser reopens the captured bytes inside the source capability
  closure.  Its result is observation-only until paired with a verified support
  certificate by ``replay_source_obj_face_topology_v1``.
  """

  try:
    text = raw.decode("utf-8")
  except UnicodeDecodeError as error:
    raise ValueError("OBJ bytes are not strict UTF-8") from error
  if "\x00" in text:
    raise ValueError("OBJ bytes contain NUL")
  vertices: list[tuple[float, float, float]] = []
  normals: list[tuple[float, float, float]] = []
  triangles: list[tuple[tuple[int, int], ...]] = []
  current_group: tuple[str, ...] = ()
  target_seen = False
  for line_number, line in enumerate(text.splitlines(), start=1):
    fields = line.split()
    if not fields or fields[0].startswith("#"):
      continue
    code = fields[0]
    if code in {"v", "vn"}:
      if len(fields) != 4:
        raise ValueError(f"OBJ {code} arity differs at line {line_number}")
      parsed = tuple(
          _parse_float(value, label=f"OBJ {code} line {line_number}")
          for value in fields[1:]
      )
      (vertices if code == "v" else normals).append(parsed)  # type: ignore[arg-type]
    elif code == "g":
      current_group = tuple(fields[1:])
      if current_group == ("face", str(face_index)):
        target_seen = True
    elif code == "f" and current_group == ("face", str(face_index)):
      if len(fields) != 4:
        raise ValueError("target OBJ face group contains a non-triangle")
      uses: list[tuple[int, int]] = []
      for token in fields[1:]:
        indices = token.split("/")
        if len(indices) != 3 or not indices[0] or not indices[2]:
          raise ValueError("target OBJ face corner lacks vertex/normal index")
        try:
          vertex_index = int(indices[0])
          normal_index = int(indices[2])
        except ValueError as error:
          raise ValueError("target OBJ face index is not an integer") from error
        if (
            vertex_index <= 0 or normal_index <= 0
            or vertex_index > len(vertices) or normal_index > len(normals)
        ):
          raise ValueError("target OBJ face index is outside prior definitions")
        uses.append((vertex_index - 1, normal_index - 1))
      if len({vertex for vertex, _normal in uses}) != 3:
        raise ValueError("target OBJ face contains a repeated vertex")
      triangles.append(tuple(uses))
  if not target_seen or not triangles:
    raise ValueError("target OBJ face group is absent or empty")

  used_vertices = sorted({
      vertex_index
      for triangle in triangles
      for vertex_index, _normal_index in triangle
  })
  used_normals = sorted({
      normal_index
      for triangle in triangles
      for _vertex_index, normal_index in triangle
  })
  vertex_remap = {
      original: local for local, original in enumerate(used_vertices)
  }
  normal_remap = {
      original: local for local, original in enumerate(used_normals)
  }
  result = {
      "vertices_mm": [
          [scale_mm * component for component in vertices[index]]
          for index in used_vertices
      ],
      "normals": [list(normals[index]) for index in used_normals],
      "triangles": [
          [
              {
                  "vertex_index": vertex_remap[vertex_index],
                  "normal_index": normal_remap[normal_index],
              }
              for vertex_index, normal_index in triangle
          ]
          for triangle in triangles
      ],
  }
  result["topology_payload_sha256"] = canonical_sha256_v1(result)
  return result


def _assembly_face_metadata(
    raw: bytes, *, body_uuid: str, face_index: int,
) -> dict[str, Any]:
  payload = _json_without_duplicates(raw, label="assembly JSON")
  if not isinstance(payload, Mapping) or not isinstance(payload.get("contacts"), list):
    raise ValueError("assembly JSON contact domain differs")
  matches = []
  for contact in payload["contacts"]:
    if not isinstance(contact, Mapping):
      raise ValueError("assembly contact schema differs")
    for role in ("entity_one", "entity_two"):
      entity = contact.get(role)
      if (
          isinstance(entity, Mapping)
          and entity.get("body") == body_uuid
          and type(entity.get("index")) is int
          and entity.get("index") == face_index
      ):
        required_fields = {
            "type", "body", "surface_type", "point_on_entity", "index",
            "bounding_box",
        }
        if (
            not required_fields <= set(entity)
            or not set(entity) <= required_fields | {"id"}
            or entity.get("type") != "BRepFace"
            or ("id" in entity and type(entity.get("id")) is not int)
        ):
          raise ValueError("assembly face metadata schema differs")
        surface_type = entity.get("surface_type")
        if surface_type not in {"PlaneSurfaceType", "CylinderSurfaceType"}:
          raise ValueError("assembly face surface type is unsupported")
        minimum, maximum = _bbox(
            entity["bounding_box"], label="assembly face bounding_box"
        )
        matches.append({
            "body_uuid": body_uuid,
            "face_index": face_index,
            "surface_type": surface_type,
            "point_on_entity": list(_point(
                entity["point_on_entity"], label="assembly face point_on_entity"
            )),
            "bounding_box": {
                "min": list(minimum),
                "max": list(maximum),
            },
        })
  if not matches:
    raise ValueError("assembly face metadata is absent")
  first = matches[0]
  if any(value != first for value in matches[1:]):
    raise ValueError("duplicate assembly face metadata disagrees")
  return first


def _capture_state(
    *,
    obj_bytes: bytes,
    assembly_json_bytes: bytes,
    body_uuid: str,
    face_index: int,
    unit_binding: Mapping[str, Any],
    path_metadata: Mapping[str, Any],
) -> dict[str, Any]:
  if not isinstance(obj_bytes, bytes) or not obj_bytes:
    raise TypeError("obj_bytes must be non-empty bytes")
  if not isinstance(assembly_json_bytes, bytes) or not assembly_json_bytes:
    raise TypeError("assembly_json_bytes must be non-empty bytes")
  if not isinstance(body_uuid, str) or not body_uuid:
    raise ValueError("body_uuid must be non-empty")
  if type(face_index) is not int or face_index < 0:
    raise ValueError("face_index must be a non-negative integer")
  unit = _unit_binding(unit_binding)
  paths = _path_metadata(path_metadata)
  triangles = _parse_obj_face(obj_bytes, face_index=face_index)
  metadata = _assembly_face_metadata(
      assembly_json_bytes, body_uuid=body_uuid, face_index=face_index
  )
  scale = unit["millimeters_per_source_unit"]
  triangles_mm = tuple(
      tuple((
          tuple(scale * component for component in vertex),
          tuple(normal),
      ) for vertex, normal in triangle)
      for triangle in triangles
  )
  point_mm = tuple(scale * value for value in metadata["point_on_entity"])
  bbox_mm = (
      tuple(scale * value for value in metadata["bounding_box"]["min"]),
      tuple(scale * value for value in metadata["bounding_box"]["max"]),
  )
  face_payload = {
      "body_uuid": body_uuid,
      "face_index": face_index,
      "surface_type": metadata["surface_type"],
      "triangles_mm": [[
          {
              "vertex": list(vertex),
              "normal_axis": list(_canonical_sign(normal)),
          }
          for vertex, normal in triangle
      ] for triangle in triangles_mm],
      "point_on_entity_mm": list(point_mm),
      "bounding_box_mm": {"min": list(bbox_mm[0]), "max": list(bbox_mm[1])},
      "unit_binding": unit,
      "path_metadata": paths,
  }
  return {
      "triangles": triangles_mm,
      "point": point_mm,
      "bbox": bbox_mm,
      "surface_type": metadata["surface_type"],
      "source_binding": {
          "obj_bytes_sha256": hashlib.sha256(obj_bytes).hexdigest(),
          "assembly_json_bytes_sha256": hashlib.sha256(
              assembly_json_bytes
          ).hexdigest(),
          "face_payload_sha256": canonical_sha256_v1(face_payload),
          "body_uuid": body_uuid,
          "face_index": face_index,
          "unit_binding": unit,
          "path_metadata": paths,
      },
  }


def _q(value: float) -> Fraction:
  if not math.isfinite(value):
    raise ValueError("exact replay received non-finite float")
  numerator, denominator = value.as_integer_ratio()
  return Fraction(numerator, denominator)


def _qdot(a: Sequence[float], b: Sequence[float]) -> Fraction:
  return sum((_q(x) * _q(y) for x, y in zip(a, b, strict=True)), Fraction())


def _qsub(a: Sequence[float], b: Sequence[float]) -> tuple[Fraction, ...]:
  return tuple(_q(x) - _q(y) for x, y in zip(a, b, strict=True))


def _qnorm_sq(value: Sequence[float] | Sequence[Fraction]) -> Fraction:
  return sum((
      item * item if isinstance(item, Fraction) else _q(item) * _q(item)
      for item in value
  ), Fraction())


def _qcross(
    a: Sequence[float], b: Sequence[float],
) -> tuple[Fraction, Fraction, Fraction]:
  x = tuple(_q(value) for value in a)
  y = tuple(_q(value) for value in b)
  return (
      x[1] * y[2] - x[2] * y[1],
      x[2] * y[0] - x[0] * y[2],
      x[0] * y[1] - x[1] * y[0],
  )


def _require_scalar_abs_le(
    value: Fraction, tolerance: float, *, label: str,
) -> None:
  if value * value > _q(tolerance) * _q(tolerance):
    raise ValueError(f"{label} exceeds its exact replay tolerance")


def _require_vector_sq_le(
    value: Sequence[Fraction], tolerance: float, *, label: str,
) -> None:
  if _qnorm_sq(value) > _q(tolerance) * _q(tolerance):
    raise ValueError(f"{label} exceeds its exact squared replay tolerance")


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
  return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
  return tuple(x - y for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
  return tuple(x + y for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
  return tuple(factor * component for component in value)  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
  return (
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  )


def _norm(value: Sequence[float]) -> float:
  return math.sqrt(_dot(value, value))


def _unit(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
  length = _norm(value)
  if not math.isfinite(length) or length <= 0.0:
    raise ValueError(f"{label} is degenerate")
  return _scale(value, 1.0 / length)


def _clean_unit(value: Sequence[float]) -> list[float]:
  return [0.0 if abs(component) <= 1.0e-15 else float(component)
          for component in value]


def _decimal_text(value: float) -> str:
  with localcontext() as context:
    context.prec = DECIMAL_PRECISION
    return format(+Decimal.from_float(float(value)), "g")


def _ratio_text(value: Fraction) -> str:
  with localcontext() as context:
    context.prec = DECIMAL_PRECISION
    return format(
        +(Decimal(value.numerator) / Decimal(value.denominator)), "g"
    )


def _common_geometry(state: Mapping[str, Any]) -> dict[str, Any]:
  triangles = state["triangles"]
  corners = [corner for triangle in triangles for corner in triangle]
  vertices = [corner[0] for corner in corners]
  raw_normals = [corner[1] for corner in corners]
  tolerance = _q(_NORMAL_UNIT_TOLERANCE)
  lower_unit_sq = (Fraction(1) - tolerance) ** 2
  upper_unit_sq = (Fraction(1) + tolerance) ** 2
  normal_squares = [_qnorm_sq(normal) for normal in raw_normals]
  if any(not lower_unit_sq <= value <= upper_unit_sq for value in normal_squares):
    raise ValueError("source normal violates exact squared unit interval")
  normals = [_unit(normal, label="source normal") for normal in raw_normals]

  observed_min = tuple(min(vertex[index] for vertex in vertices)
                       for index in range(3))
  observed_max = tuple(max(vertex[index] for vertex in vertices)
                       for index in range(3))
  scale = max(
      1.0,
      *(abs(value) for vertex in vertices for value in vertex),
      *(abs(high - low) for low, high in zip(
          observed_min, observed_max, strict=True
      )),
  )
  coordinate_tolerance = max(1.0e-8, scale * 1.0e-10)
  support_tolerance = max(1.0e-8, scale * 2.0e-10)
  declared_min, declared_max = state["bbox"]
  qtolerance = _q(coordinate_tolerance)
  for vertex in vertices:
    for value, low, high in zip(
        vertex, declared_min, declared_max, strict=True
    ):
      if _q(value) < _q(low) - qtolerance or _q(value) > _q(high) + qtolerance:
        raise ValueError("source vertex lies outside declared bounding box")
  for value, low, high in zip(
      state["point"], declared_min, declared_max, strict=True
  ):
    if _q(value) < _q(low) - qtolerance or _q(value) > _q(high) + qtolerance:
      raise ValueError("source point lies outside declared bounding box")

  area_squares = []
  for triangle in triangles:
    edge_a = _sub(triangle[1][0], triangle[0][0])
    edge_b = _sub(triangle[2][0], triangle[0][0])
    area_squares.append(_qnorm_sq(_qcross(edge_a, edge_b)))
  max_area_squared = max(area_squares)
  qscale = _q(scale)
  if max_area_squared < _q(_MIN_AREA_RATIO_SQUARED) * qscale ** 4:
    raise ValueError("source triangle area is exactly under-conditioned")
  return {
      "triangles": triangles,
      "vertices": vertices,
      "normals": normals,
      "raw_normals": raw_normals,
      "point": state["point"],
      "bbox": state["bbox"],
      "scale": scale,
      "coordinate_tolerance": coordinate_tolerance,
      "support_tolerance": support_tolerance,
      "area_ratio_squared": max_area_squared / (qscale ** 4),
      "counts": {
          "triangle_count": len(triangles),
          "corner_count": len(corners),
          "unique_vertex_count": len({tuple(value) for value in vertices}),
          "unique_normal_count": len({
              _canonical_sign(value) for value in normals
          }),
      },
  }


def _bbox_replay(
    common: Mapping[str, Any], *, tolerance: float,
) -> float:
  observed_min = tuple(min(vertex[index] for vertex in common["vertices"])
                       for index in range(3))
  observed_max = tuple(max(vertex[index] for vertex in common["vertices"])
                       for index in range(3))
  declared_min, declared_max = common["bbox"]
  errors = [
      abs(_q(observed) - _q(declared))
      for observed, declared in zip(
          (*observed_min, *observed_max),
          (*declared_min, *declared_max),
          strict=True,
      )
  ]
  if max(errors) > _q(tolerance):
    raise ValueError("source bounding box differs under exact replay")
  return float(max(errors))


def _plane(common: Mapping[str, Any]) -> dict[str, Any]:
  axes = [_canonical_sign(normal) for normal in common["normals"]]
  summed = tuple(math.fsum(axis[index] for axis in axes) for index in range(3))
  normal = _canonical_sign(_unit(summed, label="plane normal mean"))
  normal_sq = _qnorm_sq(normal)
  alignment_tolerance_sq = _q(_NORMAL_ALIGNMENT_TOLERANCE) ** 2
  for axis in axes:
    cross_sq = _qnorm_sq(_qcross(normal, axis))
    if cross_sq > alignment_tolerance_sq * normal_sq * _qnorm_sq(axis):
      raise ValueError("plane normals disagree under exact cross replay")
  projections = [_dot(normal, vertex) for vertex in common["vertices"]]
  offset = math.fsum(sorted(projections)) / len(projections)
  for vertex in common["vertices"]:
    _require_scalar_abs_le(
        _qdot(normal, vertex) - _q(offset),
        common["support_tolerance"],
        label="plane vertex support residual",
    )
  _require_scalar_abs_le(
      _qdot(normal, common["point"]) - _q(offset),
      common["support_tolerance"],
      label="plane source-point support residual",
  )
  bbox_error = _bbox_replay(
      common, tolerance=common["coordinate_tolerance"]
  )
  residuals = [abs(_dot(normal, vertex) - offset)
               for vertex in common["vertices"]]
  return {
      "support": {
          "normal": _clean_unit(normal),
          "origin_mm": list(_scale(normal, offset)),
          "offset_mm": offset,
      },
      "decision_evidence": {
          "exact_checks": [
              "normal_unit_squared_interval",
              "triangle_cross_squared_minimum",
              "normal_cross_squared_alignment",
              "vertex_plane_residual_squared",
              "point_plane_residual_squared",
              "bbox_coordinate_interval",
          ],
          "triangle_area_ratio_squared_decimal": _ratio_text(
              common["area_ratio_squared"]
          ),
      },
      "float_observations": {
          "authorizing": False,
          "maximum_vertex_support_residual_mm": _decimal_text(max(residuals)),
          "point_support_residual_mm": _decimal_text(
              abs(_dot(normal, common["point"]) - offset)
          ),
          "bbox_replay_error_mm": _decimal_text(bbox_error),
      },
      "spans": {
          "axial_span_mm_observation": None,
          "angular_span_radians_observation": None,
          "maximum_angular_gap_radians_observation": None,
      },
  }


def _cylinder_axis(
    normals: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], Fraction]:
  axes = sorted({_canonical_sign(normal) for normal in normals})
  candidates = []
  for first_index, first in enumerate(axes):
    for second in axes[first_index + 1:]:
      cross = _qcross(first, second)
      cross_squared = _qnorm_sq(cross)
      denominator = _qnorm_sq(first) * _qnorm_sq(second)
      if cross_squared:
        candidates.append((
            cross_squared / denominator, first, second,
            _canonical_sign(_unit(_cross(first, second), label="normal cross")),
        ))
  if not candidates:
    raise ValueError("cylinder has no independent source-normal pair")
  maximum = max(value[0] for value in candidates)
  if maximum < _q(_MIN_INDEPENDENT_NORMAL_CROSS_SQUARED):
    raise ValueError("cylinder normal independence is insufficient")
  selected = min(
      (value for value in candidates if value[0] == maximum),
      key=lambda value: (value[1], value[2], value[3]),
  )
  return selected[3], maximum


def _axis_basis(axis: Sequence[float]) -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
  standards = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
  reference = min(
      enumerate(standards),
      key=lambda item: (abs(_dot(axis, item[1])), item[0]),
  )[1]
  first = _canonical_sign(_unit(_cross(axis, reference), label="axis basis"))
  second = _unit(_cross(axis, first), label="axis basis")
  return first, second


def _exact_elimination(
    matrix: Sequence[Sequence[float]], target: Sequence[float],
) -> tuple[tuple[float, float, float], Fraction, Fraction]:
  qmatrix = [[_q(value) for value in row] for row in matrix]
  qtarget = [_q(value) for value in target]
  scale = max(abs(value) for row in qmatrix for value in row)
  if not scale:
    raise ValueError("cylinder linear system is degenerate")
  augmented = [
      qmatrix[row] + [qtarget[row]] for row in range(3)
  ]
  pivots = []
  swaps = 0
  for column in range(3):
    pivot_row = max(
        range(column, 3),
        key=lambda row: (abs(augmented[row][column]), -row),
    )
    pivot = augmented[pivot_row][column]
    if not pivot:
      raise ValueError("cylinder linear system is singular")
    if pivot_row != column:
      swaps += 1
      augmented[column], augmented[pivot_row] = (
          augmented[pivot_row], augmented[column]
      )
    pivots.append(abs(pivot))
    for row in range(column + 1, 3):
      factor = augmented[row][column] / augmented[column][column]
      for index in range(column, 4):
        augmented[row][index] -= factor * augmented[column][index]
  pivot_ratio = min(pivots) / scale
  if pivot_ratio < _q(_MIN_SOLVE_PIVOT_RATIO):
    raise ValueError("cylinder linear system is poorly conditioned")
  determinant = Fraction(-1 if swaps % 2 else 1)
  for pivot in pivots:
    determinant *= pivot
  qresult = [Fraction(), Fraction(), Fraction()]
  for row in range(2, -1, -1):
    qresult[row] = (
        augmented[row][3]
        - sum((augmented[row][column] * qresult[column]
               for column in range(row + 1, 3)), Fraction())
    ) / augmented[row][row]
  result = tuple(float(value) for value in qresult)
  if not all(math.isfinite(value) for value in result):
    raise ValueError("cylinder linear solution is non-finite")
  return result, pivot_ratio, abs(determinant)


def _angular_observations(
    radials: Sequence[Sequence[float]],
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float]:
  angles = sorted({
      math.atan2(_dot(radial, second), _dot(radial, first)) % (2.0 * math.pi)
      for radial in radials
  })
  if len(angles) < 2:
    return 0.0, 2.0 * math.pi
  gaps = [angles[index + 1] - angles[index]
          for index in range(len(angles) - 1)]
  gaps.append(angles[0] + 2.0 * math.pi - angles[-1])
  maximum_gap = max(gaps)
  return 2.0 * math.pi - maximum_gap, maximum_gap


def _cylinder(common: Mapping[str, Any]) -> dict[str, Any]:
  ordered = sorted(
      zip(common["vertices"], common["normals"], strict=True),
      key=lambda item: (item[0], _canonical_sign(item[1])),
  )
  reference = ordered[0][1]
  global_sign = 1.0 if tuple(reference) == _canonical_sign(reference) else -1.0
  normals = [_scale(normal, global_sign) for normal in common["normals"]]
  axis, independent_ratio = _cylinder_axis(normals)
  axis_sq = _qnorm_sq(axis)
  for normal in normals:
    dot = _qdot(axis, normal)
    if dot * dot > (
        _q(_AXIS_NORMAL_TOLERANCE) ** 2
        * axis_sq * _qnorm_sq(normal)
    ):
      raise ValueError("cylinder normal is not exactly orthogonal to axis")
  first, second = _axis_basis(axis)

  rows = []
  for vertex, normal in sorted(
      zip(common["vertices"], normals, strict=True),
      key=lambda item: (item[0], item[1]),
  ):
    rows.append((
        (_dot(normal, first), _dot(normal, second), 1.0),
        _dot(vertex, normal),
        vertex,
        normal,
    ))
  matrix = [[0.0] * 3 for _ in range(3)]
  target = [0.0] * 3
  for row, observed, _vertex, _normal in rows:
    for left in range(3):
      target[left] = math.fsum((target[left], row[left] * observed))
      for right in range(3):
        matrix[left][right] = math.fsum((
            matrix[left][right], row[left] * row[right]
        ))
  solved, pivot_ratio, determinant = _exact_elimination(matrix, target)
  axis_point = _add(_scale(first, solved[0]), _scale(second, solved[1]))
  signed_radius = solved[2]
  radius = abs(signed_radius)
  if _q(radius) ** 2 <= _q(common["coordinate_tolerance"]) ** 2:
    raise ValueError("cylinder radius is degenerate under squared replay")
  orientation = 1.0 if signed_radius >= 0.0 else -1.0

  projections = [
      (_dot(normal, first), _dot(normal, second)) for normal in normals
  ]
  g11 = sum((_q(x) ** 2 for x, _y in projections), Fraction())
  g12 = sum((_q(x) * _q(y) for x, y in projections), Fraction())
  g22 = sum((_q(y) ** 2 for _x, y in projections), Fraction())
  coverage_det = g11 * g22 - g12 * g12
  coverage_trace = g11 + g22
  coverage_ratio = coverage_det / (coverage_trace * coverage_trace)
  if coverage_ratio < _q(_MIN_NORMAL_COVERAGE_DETERMINANT_RATIO):
    raise ValueError("cylinder normal coverage determinant is insufficient")

  axial_values: list[Fraction] = []
  radials_float = []
  model_residual_observations = []
  radial_residual_observations = []
  normal_cross_observations = []
  for _row, _observed, vertex, normal in rows:
    displacement = _sub(vertex, axis_point)
    axial = _qdot(displacement, axis)
    axial_values.append(axial)
    model = _add(
        axis_point,
        _add(_scale(axis, float(axial)), _scale(normal, signed_radius)),
    )
    model_delta = _qsub(vertex, model)
    _require_vector_sq_le(
        model_delta, common["support_tolerance"],
        label="cylinder vertex point-normal model",
    )
    model_residual_observations.append(_norm(
        tuple(float(value) for value in model_delta)
    ))
    radial = _sub(displacement, _scale(axis, float(axial)))
    radial_sq = _qnorm_sq(radial)
    lower_radius = _q(max(0.0, radius - common["support_tolerance"]))
    upper_radius = _q(radius + common["support_tolerance"])
    if not lower_radius ** 2 <= radial_sq <= upper_radius ** 2:
      raise ValueError("cylinder radial squared interval differs")
    radial_residual_observations.append(abs(_norm(radial) - radius))
    expected_normal = _scale(normal, orientation)
    if _qdot(radial, expected_normal) <= 0:
      raise ValueError("cylinder radial/normal orientation differs")
    normal_cross_sq = _qnorm_sq(_qcross(radial, expected_normal))
    if normal_cross_sq > (
        _q(_NORMAL_ALIGNMENT_TOLERANCE) ** 2
        * radial_sq * _qnorm_sq(expected_normal)
    ):
      raise ValueError("cylinder radial/normal cross differs")
    normal_cross_observations.append(math.sqrt(float(normal_cross_sq)))
    radials_float.append(radial)

  axial_span = max(axial_values) - min(axial_values)
  minimum_axial_span = _q(max(
      1.0e-8, common["support_tolerance"] * 4.0
  ))
  if axial_span * axial_span < minimum_axial_span * minimum_axial_span:
    raise ValueError("cylinder requires two separated axial levels")

  point_displacement = _sub(common["point"], axis_point)
  point_axial = _qdot(point_displacement, axis)
  point_radial = _sub(
      point_displacement, _scale(axis, float(point_axial))
  )
  point_radial_sq = _qnorm_sq(point_radial)
  lower_point_radius = _q(max(
      0.0, radius - common["support_tolerance"]
  ))
  upper_point_radius = _q(radius + common["support_tolerance"])
  if not (
      lower_point_radius ** 2
      <= point_radial_sq
      <= upper_point_radius ** 2
  ):
    raise ValueError("source point/cylinder squared support interval differs")

  bbox_tolerance = max(
      common["coordinate_tolerance"], common["scale"] * 1.0e-3
  )
  bbox_error = _bbox_replay(common, tolerance=bbox_tolerance)
  angular_span, maximum_gap = _angular_observations(
      radials_float, first, second
  )
  return {
      "support": {
          "axis_direction": _clean_unit(axis),
          "axis_point_mm": [
              0.0 if abs(value) <= 1.0e-12 else value for value in axis_point
          ],
          "radius_mm": radius,
      },
      "decision_evidence": {
          "exact_checks": [
              "normal_unit_squared_interval",
              "triangle_cross_squared_minimum",
              "independent_normal_cross_squared",
              "normal_axis_dot_squared",
              "normal_coverage_gram_determinant",
              "linear_system_exact_fraction_pivots_and_determinant",
              "vertex_model_residual_squared",
              "radial_squared_interval",
              "radial_normal_cross_squared_and_positive_dot",
              "axial_span_squared",
              "point_radial_squared_interval",
              "bbox_coordinate_interval",
          ],
          "triangle_area_ratio_squared_decimal": _ratio_text(
              common["area_ratio_squared"]
          ),
          "independent_normal_cross_squared_ratio_decimal": _ratio_text(
              independent_ratio
          ),
          "normal_coverage_determinant_ratio_decimal": _ratio_text(
              coverage_ratio
          ),
          "linear_solve_pivot_ratio_decimal": _ratio_text(pivot_ratio),
          "linear_solve_abs_determinant_decimal": _ratio_text(determinant),
      },
      "float_observations": {
          "authorizing": False,
          "maximum_vertex_model_residual_mm": _decimal_text(
              max(model_residual_observations)
          ),
          "maximum_radial_residual_mm": _decimal_text(
              max(radial_residual_observations)
          ),
          "maximum_radial_normal_cross": _decimal_text(
              max(normal_cross_observations)
          ),
          "point_radial_residual_mm": _decimal_text(
              abs(math.sqrt(float(point_radial_sq)) - radius)
          ),
          "bbox_replay_error_mm": _decimal_text(bbox_error),
      },
      "spans": {
          "axial_span_mm_exact_decimal": _ratio_text(axial_span),
          "angular_span_radians_observation": _decimal_text(angular_span),
          "maximum_angular_gap_radians_observation": _decimal_text(maximum_gap),
      },
  }


def _derive_receipt(state: Mapping[str, Any]) -> dict[str, Any]:
  common = _common_geometry(state)
  surface_type = "plane" if state["surface_type"] == "PlaneSurfaceType" else "cylinder"
  analytic = _plane(common) if surface_type == "plane" else _cylinder(common)
  decisive_expressions = [
      "normal_norm_squared_in_locked_interval",
      "triangle_cross_norm_squared_ge_scale_fourth_minimum",
      "bbox_and_point_coordinate_fraction_intervals",
      "support_residual_vector_squared_le_tolerance_squared",
  ]
  if surface_type == "cylinder":
    decisive_expressions.extend([
        "normal_cross_squared_ge_independence_minimum",
        "axis_normal_dot_squared_le_scaled_tolerance_squared",
        "normal_coverage_gram_determinant_ge_ratio_times_trace_squared",
        "linear_system_fraction_pivot_ratio_ge_locked_minimum",
        "radial_squared_between_radius_tolerance_squares",
        "axial_span_squared_ge_locked_minimum_squared",
    ])
  unsigned = {
      "schema_version": SCHEMA_VERSION,
      "evidence_scope": EVIDENCE_SCOPE,
      "formal_authorized": False,
      "surface_type": surface_type,
      "declared_surface_type": state["surface_type"],
      "source_binding": state["source_binding"],
      "support": analytic["support"],
      "counts": common["counts"],
      "decision_evidence": analytic["decision_evidence"],
      "float_observations": analytic["float_observations"],
      "spans": analytic["spans"],
      "numeric_replay": {
          "method": NUMERIC_METHOD,
          "precision_digits": DECIMAL_PRECISION,
          "input_conversion": "Decimal.from_float_exact_then_display_only",
          "decisive_engine": DECISIVE_ENGINE,
          "decisive_expressions": decisive_expressions,
          "float_observations_authorize": False,
          "sqrt_and_angular_observations_authorize": False,
      },
      "thresholds": {
          "coordinate_tolerance_mm": common["coordinate_tolerance"],
          "support_tolerance_mm": common["support_tolerance"],
          "normal_unit_tolerance": _NORMAL_UNIT_TOLERANCE,
          "normal_alignment_tolerance": _NORMAL_ALIGNMENT_TOLERANCE,
          "axis_normal_tolerance": _AXIS_NORMAL_TOLERANCE,
          "minimum_triangle_area_ratio_squared": _MIN_AREA_RATIO_SQUARED,
          "minimum_independent_normal_cross_squared": (
              _MIN_INDEPENDENT_NORMAL_CROSS_SQUARED
          ),
          "minimum_linear_solve_pivot_ratio": _MIN_SOLVE_PIVOT_RATIO,
          "minimum_normal_coverage_determinant_ratio": (
              _MIN_NORMAL_COVERAGE_DETERMINANT_RATIO
          ),
      },
      "checks": {
          "all_source_corners_replayed": True,
          "source_point_checked": True,
          "source_bounding_box_checked": True,
          "source_surface_type_checked": True,
          "occ_geometry_used": False,
          "external_numeric_backend_used": False,
      },
  }
  return {
      **unsigned,
      "receipt_payload_sha256": canonical_sha256_v1(unsigned),
  }


_RECEIPT_FIELDS = {
    "schema_version", "evidence_scope", "formal_authorized", "surface_type",
    "declared_surface_type", "source_binding", "support", "counts",
    "decision_evidence", "float_observations", "spans", "numeric_replay",
    "thresholds", "checks", "receipt_payload_sha256",
}


def _validate_receipt(value: Any) -> dict[str, Any]:
  if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
    raise ValueError("source support receipt schema differs")
  try:
    raw = _canonical_bytes(value)
    result = _json_without_duplicates(raw, label="source support receipt")
  except (TypeError, ValueError) as error:
    raise ValueError("source support receipt is not canonical JSON data") from error
  observed = result.pop("receipt_payload_sha256", None)
  if observed != canonical_sha256_v1(result):
    raise ValueError("source support receipt self-hash differs")
  result["receipt_payload_sha256"] = observed
  return result


def _deep_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({
        str(key): _deep_freeze(item) for key, item in value.items()
    })
  if isinstance(value, list):
    return tuple(_deep_freeze(item) for item in value)
  return value


def _sealed_boundaries():
  source_states: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
  certificate_states: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
  source_seal = object()
  certificate_seal = object()

  class SourceObjFaceCapabilityV1:
    __slots__ = ("__weakref__",)

    def __new__(cls, *, _seal: object | None = None):
      if _seal is not source_seal:
        raise TypeError("source OBJ face capabilities are capture-factory-only")
      return super().__new__(cls)

    def __setattr__(self, _name: str, _value: Any) -> None:
      raise TypeError("source OBJ face capabilities are sealed")

    def __copy__(self):
      raise TypeError("source OBJ face capabilities cannot be copied")

    def __deepcopy__(self, _memo: Any):
      raise TypeError("source OBJ face capabilities cannot be deep-copied")

    def __reduce_ex__(self, _protocol: int):
      raise TypeError("source OBJ face capabilities are not serializable")

  class SourceObjAnalyticSupportCertificateV1:
    __slots__ = ("__weakref__",)

    def __new__(cls, *, _seal: object | None = None):
      if _seal is not certificate_seal:
        raise TypeError("source support certificates are verifier-factory-only")
      return super().__new__(cls)

    def __setattr__(self, _name: str, _value: Any) -> None:
      raise TypeError("source support certificates are sealed")

    def __copy__(self):
      raise TypeError("source support certificates cannot be copied")

    def __deepcopy__(self, _memo: Any):
      raise TypeError("source support certificates cannot be deep-copied")

    def __reduce_ex__(self, _protocol: int):
      raise TypeError("source support certificates are not serializable")

    @property
    def surface_type(self) -> str:
      return str(certificate_state(self)["receipt"]["surface_type"])

    @property
    def receipt(self) -> Mapping[str, Any]:
      return certificate_state(self)["receipt"]

    @property
    def canonical_json(self) -> str:
      return str(certificate_state(self)["canonical_json"])

    @property
    def sha256(self) -> str:
      return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

  SourceObjFaceCapabilityV1.__name__ = "SourceObjFaceCapabilityV1"
  SourceObjFaceCapabilityV1.__qualname__ = "SourceObjFaceCapabilityV1"
  SourceObjAnalyticSupportCertificateV1.__name__ = (
      "SourceObjAnalyticSupportCertificateV1"
  )
  SourceObjAnalyticSupportCertificateV1.__qualname__ = (
      "SourceObjAnalyticSupportCertificateV1"
  )

  def source_state(value: Any) -> Mapping[str, Any]:
    if type(value) is not SourceObjFaceCapabilityV1:
      raise TypeError("expected an exact source OBJ face capability")
    try:
      captured = source_states[value]
    except KeyError as error:
      raise TypeError("source OBJ face capability lacks capture state") from error
    replayed = _capture_state(
        obj_bytes=captured["obj_bytes"],
        assembly_json_bytes=captured["assembly_json_bytes"],
        body_uuid=captured["body_uuid"],
        face_index=captured["face_index"],
        unit_binding=captured["unit_binding"],
        path_metadata=captured["path_metadata"],
    )
    if replayed != captured["parsed_state"]:
      raise TypeError("source OBJ face capability replay differs")
    return replayed

  def certificate_state(value: Any) -> Mapping[str, Any]:
    if type(value) is not SourceObjAnalyticSupportCertificateV1:
      raise TypeError("expected an exact source support certificate")
    try:
      return certificate_states[value]
    except KeyError as error:
      raise TypeError("source support certificate lacks verifier state") from error

  def make_certificate(receipt: Mapping[str, Any]):
    value = SourceObjAnalyticSupportCertificateV1(_seal=certificate_seal)
    canonical_json = _canonical_bytes(receipt).decode("utf-8")
    certificate_states[value] = MappingProxyType({
        "receipt": _deep_freeze(receipt),
        "canonical_json": canonical_json,
    })
    return value

  def capture_source_obj_face_v1(
      *,
      obj_bytes: bytes,
      assembly_json_bytes: bytes,
      body_uuid: str,
      face_index: int,
      unit_binding: Mapping[str, Any],
      path_metadata: Mapping[str, Any],
  ):
    state = _capture_state(
        obj_bytes=obj_bytes,
        assembly_json_bytes=assembly_json_bytes,
        body_uuid=body_uuid,
        face_index=face_index,
        unit_binding=unit_binding,
        path_metadata=path_metadata,
    )
    value = SourceObjFaceCapabilityV1(_seal=source_seal)
    source_states[value] = MappingProxyType({
        "obj_bytes": obj_bytes,
        "assembly_json_bytes": assembly_json_bytes,
        "body_uuid": body_uuid,
        "face_index": face_index,
        "unit_binding": MappingProxyType(dict(unit_binding)),
        "path_metadata": MappingProxyType(dict(path_metadata)),
        "parsed_state": state,
    })
    return value

  def recover_source_obj_analytic_support_v1(source):
    receipt = _derive_receipt(source_state(source))
    return make_certificate(receipt)

  def replay_source_obj_face_topology_v1(source, support_certificate):
    state = source_state(source)
    verified = certificate_state(support_certificate)
    expected_receipt = _derive_receipt(state)
    if verified["receipt"] != _deep_freeze(expected_receipt):
      raise ValueError(
          "source topology replay requires its verified support certificate"
      )
    captured = source_states[source]
    topology = _parse_obj_face_topology(
        captured["obj_bytes"],
        face_index=captured["face_index"],
        scale_mm=float(
            captured["unit_binding"]["millimeters_per_source_unit"]
        ),
    )
    return _deep_freeze({
        "source_binding": state["source_binding"],
        "surface_type": expected_receipt["surface_type"],
        "support": expected_receipt["support"],
        "thresholds": expected_receipt["thresholds"],
        "declared_face_bbox_mm": {
            "min": list(state["bbox"][0]),
            "max": list(state["bbox"][1]),
        },
        **topology,
    })

  def replay_source_obj_face_topology_observation_v1(source):
    """Replay bound OBJ topology without claiming an analytic support.

    This observation is intentionally weaker than
    ``replay_source_obj_face_topology_v1``.  A downstream certificate must
    independently prove its support and trim semantics; this function only
    exposes topology already sealed by ``capture_source_obj_face_v1``.
    """

    state = source_state(source)
    captured = source_states[source]
    topology = _parse_obj_face_topology(
        captured["obj_bytes"],
        face_index=captured["face_index"],
        scale_mm=float(
            captured["unit_binding"]["millimeters_per_source_unit"]
        ),
    )
    return _deep_freeze({
        "source_binding": state["source_binding"],
        "surface_type": state["surface_type"],
        "declared_face_bbox_mm": {
            "min": list(state["bbox"][0]),
            "max": list(state["bbox"][1]),
        },
        "point_on_entity_mm": list(state["point"]),
        **topology,
    })

  def verify_source_analytic_support_v1(receipt, source):
    observed = _validate_receipt(receipt)
    expected = _derive_receipt(source_state(source))
    if observed != expected:
      raise ValueError("source support certificate differs from source replay")
    return make_certificate(expected)

  return (
      SourceObjFaceCapabilityV1,
      SourceObjAnalyticSupportCertificateV1,
      capture_source_obj_face_v1,
      recover_source_obj_analytic_support_v1,
      replay_source_obj_face_topology_v1,
      replay_source_obj_face_topology_observation_v1,
      verify_source_analytic_support_v1,
  )


(
    SourceObjFaceCapabilityV1,
    SourceObjAnalyticSupportCertificateV1,
    capture_source_obj_face_v1,
    recover_source_obj_analytic_support_v1,
    replay_source_obj_face_topology_v1,
    replay_source_obj_face_topology_observation_v1,
    verify_source_analytic_support_v1,
) = _sealed_boundaries()
del _sealed_boundaries


__all__ = [
    "DECIMAL_PRECISION",
    "EVIDENCE_SCOPE",
    "NUMERIC_METHOD",
    "SCHEMA_VERSION",
    "SourceObjAnalyticSupportCertificateV1",
    "SourceObjFaceCapabilityV1",
    "canonical_sha256_v1",
    "capture_source_obj_face_v1",
    "recover_source_obj_analytic_support_v1",
    "replay_source_obj_face_topology_v1",
    "replay_source_obj_face_topology_observation_v1",
    "verify_source_analytic_support_v1",
]
