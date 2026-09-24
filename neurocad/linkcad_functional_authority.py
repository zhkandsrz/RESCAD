"""Source-replayed support and mobility supervision for LinkCAD.

Mobility comes only from Fusion joint metadata.  Support is derived only from
the two bound B-Rep anchors and therefore cannot leak the joint type.  This
separation is the basis of LinkCAD's language counterfactual evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "linkcad_functional_mobility_authority.v1"


@dataclass(frozen=True, slots=True)
class MobilityContract:
  name: str
  rotational_dof: int
  translational_dof: int


SOURCE_MOBILITY_CONTRACTS: Mapping[str, MobilityContract] = {
    "RigidJointType": MobilityContract("fixed", 0, 0),
    "RevoluteJointType": MobilityContract("revolute", 1, 0),
    "SliderJointType": MobilityContract("prismatic", 0, 1),
    "CylindricalJointType": MobilityContract("cylindrical", 1, 1),
    "PlanarJointType": MobilityContract("planar", 1, 2),
    "BallJointType": MobilityContract("ball", 3, 0),
    "PinSlotJointType": MobilityContract("pin_slot", 1, 1),
}


@dataclass(frozen=True, slots=True)
class BRepJointAnchor:
  occurrence_id: str
  body_id: str
  entity_type: str
  entity_index: int
  shape_type: str
  geometry_type: str
  origin: tuple[float, float, float]
  primary_axis: tuple[float, float, float]
  secondary_axis: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FunctionalJointRow:
  assembly_id: str
  joint_id: str
  joint_name: str
  source_joint_type: str
  mobility: MobilityContract
  support_family: str
  endpoint_a: BRepJointAnchor
  endpoint_b: BRepJointAnchor
  offset: float
  angle: float
  is_flipped: bool


class FunctionalJointExclusion(ValueError):
  """A source joint cannot be admitted to functional supervision."""


_AXIAL_SHAPES = frozenset({
    "Arc3DCurveType",
    "Circle3DCurveType",
    "ConeSurfaceType",
    "CylinderSurfaceType",
    "Ellipse3DCurveType",
    "EllipticalConeSurfaceType",
    "EllipticalCylinderSurfaceType",
    "TorusSurfaceType",
})
_PLANAR_SHAPES = frozenset({"PlaneSurfaceType"})
_SPHERICAL_SHAPES = frozenset({"SphereSurfaceType"})
_LINEAR_SHAPES = frozenset({"Line3DCurveType"})


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise FunctionalJointExclusion(f"{label} is missing or not an object")
  return value


def _text(value: Any, *, label: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise FunctionalJointExclusion(f"{label} is missing")
  return value.strip()


def _vector(value: Any, *, label: str) -> tuple[float, float, float]:
  row = _mapping(value, label=label)
  result: list[float] = []
  for axis in ("x", "y", "z"):
    coordinate = row.get(axis)
    if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
      raise FunctionalJointExclusion(f"{label}.{axis} is missing")
    result.append(float(coordinate))
  return (result[0], result[1], result[2])


def _parameter_value(value: Any, *, label: str) -> float:
  row = _mapping(value, label=label)
  scalar = row.get("value")
  if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
    raise FunctionalJointExclusion(f"{label}.value is missing")
  return float(scalar)


def _anchor(value: Any, *, occurrence_id: str, label: str) -> BRepJointAnchor:
  geometry = _mapping(value, label=label)
  entity = _mapping(geometry.get("entity_one"), label=f"{label}.entity_one")
  entity_occurrence = entity.get("occurrence")
  if entity_occurrence is not None and entity_occurrence != occurrence_id:
    raise FunctionalJointExclusion(
        f"{label} occurrence does not match the joint occurrence"
    )
  entity_index = entity.get("index")
  if isinstance(entity_index, bool) or not isinstance(entity_index, int):
    raise FunctionalJointExclusion(f"{label} entity index is missing")
  shape_type = entity.get("surface_type") or entity.get("curve_type")
  return BRepJointAnchor(
      occurrence_id=occurrence_id,
      body_id=_text(entity.get("body"), label=f"{label}.body"),
      entity_type=_text(entity.get("type"), label=f"{label}.entity_type"),
      entity_index=entity_index,
      shape_type=_text(shape_type, label=f"{label}.shape_type"),
      geometry_type=_text(
          geometry.get("geometry_type"), label=f"{label}.geometry_type"
      ),
      origin=_vector(geometry.get("origin"), label=f"{label}.origin"),
      primary_axis=_vector(
          geometry.get("primary_axis_vector"), label=f"{label}.primary_axis"
      ),
      secondary_axis=_vector(
          geometry.get("secondary_axis_vector"),
          label=f"{label}.secondary_axis",
      ),
  )


def derive_support_family(shape_a: str, shape_b: str) -> str:
  """Map B-Rep anchor types to a joint-type-independent support family."""

  shapes = frozenset((shape_a, shape_b))
  if shapes <= _PLANAR_SHAPES:
    return "planar_support"
  if shapes <= _AXIAL_SHAPES:
    return "axial_support"
  if shapes <= _SPHERICAL_SHAPES:
    return "spherical_support"
  if shapes <= _LINEAR_SHAPES:
    return "linear_support"
  if shapes & _SPHERICAL_SHAPES:
    return "point_frame_support"
  if shapes & _AXIAL_SHAPES and shapes & _PLANAR_SHAPES:
    return "axis_plane_support"
  return "general_frame_support"


def replay_functional_joint(
    *,
    assembly_id: str,
    joint_id: str,
    source: Mapping[str, Any],
) -> FunctionalJointRow:
  """Replay one functional joint without consulting geometry/model outcomes."""

  motion = _mapping(source.get("joint_motion"), label="joint_motion")
  source_joint_type = _text(
      motion.get("joint_type"), label="joint_motion.joint_type"
  )
  mobility = SOURCE_MOBILITY_CONTRACTS.get(source_joint_type)
  if mobility is None:
    raise FunctionalJointExclusion(
        f"unsupported source joint type: {source_joint_type}"
    )
  occurrence_a = _text(source.get("occurrence_one"), label="occurrence_one")
  occurrence_b = _text(source.get("occurrence_two"), label="occurrence_two")
  if occurrence_a == occurrence_b:
    raise FunctionalJointExclusion("joint endpoints use the same occurrence")
  endpoint_a = _anchor(
      source.get("geometry_or_origin_one"),
      occurrence_id=occurrence_a,
      label="geometry_or_origin_one",
  )
  endpoint_b = _anchor(
      source.get("geometry_or_origin_two"),
      occurrence_id=occurrence_b,
      label="geometry_or_origin_two",
  )
  return FunctionalJointRow(
      assembly_id=_text(assembly_id, label="assembly_id"),
      joint_id=_text(joint_id, label="joint_id"),
      joint_name=str(source.get("name") or ""),
      source_joint_type=source_joint_type,
      mobility=mobility,
      support_family=derive_support_family(
          endpoint_a.shape_type, endpoint_b.shape_type
      ),
      endpoint_a=endpoint_a,
      endpoint_b=endpoint_b,
      offset=_parameter_value(source.get("offset"), label="offset"),
      angle=_parameter_value(source.get("angle"), label="angle"),
      is_flipped=bool(source.get("is_flipped", False)),
  )


def _source_joint_rows(value: Any) -> Sequence[tuple[str, Mapping[str, Any]]]:
  if not isinstance(value, Mapping):
    return ()
  return tuple(
      (joint_id, row)
      for joint_id, row in sorted(value.items())
      if isinstance(joint_id, str) and isinstance(row, Mapping)
  )


def build_functional_mobility_authority(
    assembly_json: str | Path,
) -> dict[str, Any]:
  """Build a fail-closed source-bound supervision manifest for one assembly."""

  path = Path(assembly_json).resolve()
  raw = path.read_bytes()
  payload = json.loads(raw.decode("utf-8"))
  assembly_id = path.parent.name
  accepted: list[dict[str, Any]] = []
  excluded: list[dict[str, str]] = []
  for joint_id, source in _source_joint_rows(_mapping(payload, label="assembly" ).get("joints")):
    try:
      accepted.append(asdict(replay_functional_joint(
          assembly_id=assembly_id,
          joint_id=joint_id,
          source=source,
      )))
    except FunctionalJointExclusion as error:
      excluded.append({"joint_id": joint_id, "reason": str(error)})
  return {
      "schema_version": SCHEMA_VERSION,
      "scope": "source_joint_functional_supervision_only",
      "assembly_id": assembly_id,
      "source_assembly_json": path.as_posix(),
      "source_assembly_sha256": hashlib.sha256(raw).hexdigest(),
      "accepted_joint_count": len(accepted),
      "excluded_joint_count": len(excluded),
      "rows": accepted,
      "exclusions": excluded,
  }
