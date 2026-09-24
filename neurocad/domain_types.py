"""Core data types for neuro-symbolic CAD planning and solving.

The module deliberately avoids the name ``types`` so running Python from the
repository root cannot shadow the standard-library :mod:`types` module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from .math3d import as_vector, normalize, orthonormal_basis_from_z


_RIGID_ROTATION_ATOL = 1e-6


def _copy_vec(vec: np.ndarray) -> np.ndarray:
  return np.array(vec, dtype=float).copy()


def _validated_rigid_components(
    rotation: Any,
    translation: Any,
    *,
    copy: bool,
) -> tuple[np.ndarray, np.ndarray]:
  try:
    if copy:
      rotation_array = np.array(rotation, dtype=float, copy=True).reshape(3, 3)
      translation_array = np.array(
          translation, dtype=float, copy=True
      ).reshape(3)
    else:
      rotation_array = np.asarray(rotation, dtype=float).reshape(3, 3)
      translation_array = np.asarray(translation, dtype=float).reshape(3)
  except (TypeError, ValueError) as error:
    raise ValueError(
        "Invalid rigid transform: expected a 3x3 rotation and 3-vector "
        "translation."
    ) from error
  if (
      not np.isfinite(rotation_array).all()
      or not np.isfinite(translation_array).all()
  ):
    raise ValueError("Invalid rigid transform: all values must be finite.")
  for column_a in range(3):
    for column_b in range(3):
      dot = sum(
          float(rotation_array[row, column_a])
          * float(rotation_array[row, column_b])
          for row in range(3)
      )
      expected = 1.0 if column_a == column_b else 0.0
      if abs(dot - expected) > _RIGID_ROTATION_ATOL:
        raise ValueError(
            "Invalid rigid transform: rotation must be orthonormal "
            "(R^T R = I)."
        )
  determinant = float(
      rotation_array[0, 0]
      * (
          rotation_array[1, 1] * rotation_array[2, 2]
          - rotation_array[1, 2] * rotation_array[2, 1]
      )
      - rotation_array[0, 1]
      * (
          rotation_array[1, 0] * rotation_array[2, 2]
          - rotation_array[1, 2] * rotation_array[2, 0]
      )
      + rotation_array[0, 2]
      * (
          rotation_array[1, 0] * rotation_array[2, 1]
          - rotation_array[1, 1] * rotation_array[2, 0]
      )
  )
  if not np.isclose(
      determinant,
      1.0,
      atol=_RIGID_ROTATION_ATOL,
      rtol=0.0,
  ):
    raise ValueError(
        "Invalid rigid transform: rotation determinant must be +1, "
        f"got {determinant:.12g}."
    )
  return rotation_array, translation_array


def _readonly_array_copy(array: np.ndarray) -> np.ndarray:
  result = np.array(array, dtype=float, copy=True)
  result.setflags(write=False)
  return result


@dataclass(init=False, slots=True, unsafe_hash=True)
class Transform:
  """Rigid transform with rotation matrix and translation vector."""

  __rotation_values: tuple[float, ...] = field(repr=False)
  __translation_values: tuple[float, ...] = field(repr=False)

  def __setattr__(self, name: str, value: Any) -> None:
    raise AttributeError(f"Transform is immutable; cannot replace {name!r}.")

  def __init__(
      self,
      rotation: Any = None,
      translation: Any = None,
  ) -> None:
    if rotation is None:
      rotation = np.eye(3, dtype=float)
    if translation is None:
      translation = np.zeros(3, dtype=float)
    rotation_array, translation_array = _validated_rigid_components(
        rotation, translation, copy=True
    )
    rotation_values = tuple(float(value) for value in rotation_array.reshape(-1))
    translation_values = tuple(float(value) for value in translation_array.reshape(-1))
    object.__setattr__(self, "_Transform__rotation_values", rotation_values)
    object.__setattr__(self, "_Transform__translation_values", translation_values)

  def _validated_components(self) -> tuple[np.ndarray, np.ndarray]:
    return _validated_rigid_components(
        np.asarray(self.__rotation_values, dtype=float).reshape(3, 3),
        np.asarray(self.__translation_values, dtype=float).reshape(3),
        copy=False,
    )

  @property
  def rotation(self) -> np.ndarray:
    rotation, _ = self._validated_components()
    return _readonly_array_copy(rotation)

  @property
  def translation(self) -> np.ndarray:
    _, translation = self._validated_components()
    return _readonly_array_copy(translation)

  @staticmethod
  def identity() -> "Transform":
    return Transform()

  def copy(self) -> "Transform":
    rotation, translation = self._validated_components()
    return Transform(rotation, translation)

  def with_translation(self, translation: Any) -> "Transform":
    rotation, _ = self._validated_components()
    return Transform(rotation=rotation, translation=translation)

  def translated(self, delta: Any) -> "Transform":
    rotation, translation = self._validated_components()
    return Transform(
        rotation=rotation,
        translation=translation + as_vector(delta),
    )

  def apply_point(self, point: np.ndarray) -> np.ndarray:
    p = as_vector(point)
    rotation, translation = self._validated_components()
    return rotation @ p + translation

  def apply_direction(self, direction: np.ndarray) -> np.ndarray:
    d = as_vector(direction)
    rotation, _ = self._validated_components()
    return rotation @ d

  def to_dict(self) -> dict[str, Any]:
    rotation, translation = self._validated_components()
    return {
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }


@dataclass
class SocketFrame:
  """Socket data in world coordinates."""

  origin: np.ndarray
  axis: Optional[np.ndarray] = None
  normal: Optional[np.ndarray] = None
  x_axis: Optional[np.ndarray] = None
  y_axis: Optional[np.ndarray] = None
  z_axis: Optional[np.ndarray] = None
  radius: Optional[float] = None

  def __post_init__(self) -> None:
    self.origin = as_vector(self.origin)
    if self.axis is not None:
      self.axis = normalize(self.axis)
    if self.normal is not None:
      self.normal = normalize(self.normal)
    if self.x_axis is not None:
      self.x_axis = normalize(self.x_axis)
    if self.y_axis is not None:
      self.y_axis = normalize(self.y_axis)
    if self.z_axis is not None:
      self.z_axis = normalize(self.z_axis)
    if self.radius is not None:
      self.radius = float(self.radius)
    self._ensure_basis()

  def _ensure_basis(self) -> None:
    preferred_z = self.z_axis
    if preferred_z is None:
      preferred_z = self.axis if self.axis is not None else self.normal
    if preferred_z is None:
      return
    basis_x = self.x_axis
    basis_y = self.y_axis
    if basis_x is not None and basis_y is None:
      tangent_x = basis_x - float(np.dot(basis_x, preferred_z)) * preferred_z
      if float(np.linalg.norm(tangent_x)) > 1e-10:
        basis_x = normalize(tangent_x)
        basis_y = normalize(np.cross(preferred_z, basis_x))
      else:
        basis_x = None
    elif basis_y is not None and basis_x is None:
      tangent_y = basis_y - float(np.dot(basis_y, preferred_z)) * preferred_z
      if float(np.linalg.norm(tangent_y)) > 1e-10:
        basis_y = normalize(tangent_y)
        basis_x = normalize(np.cross(basis_y, preferred_z))
      else:
        basis_y = None
    if basis_x is None or basis_y is None:
      basis_x, basis_y, preferred_z = orthonormal_basis_from_z(preferred_z)
    self.x_axis = basis_x
    self.y_axis = basis_y
    self.z_axis = preferred_z

  def basis_matrix(self) -> Optional[np.ndarray]:
    if self.x_axis is None or self.y_axis is None or self.z_axis is None:
      return None
    return np.stack([self.x_axis, self.y_axis, self.z_axis], axis=1)

  def to_dict(self) -> dict[str, Any]:
    return {
        "origin": self.origin.tolist(),
        "axis": None if self.axis is None else self.axis.tolist(),
        "normal": None if self.normal is None else self.normal.tolist(),
        "x_axis": None if self.x_axis is None else self.x_axis.tolist(),
        "y_axis": None if self.y_axis is None else self.y_axis.tolist(),
        "z_axis": None if self.z_axis is None else self.z_axis.tolist(),
        "radius": self.radius,
    }


@dataclass
class Socket:
  """Semantic socket that can be used in topological constraints."""

  name: str
  kind: str
  origin: np.ndarray
  axis: Optional[np.ndarray] = None
  normal: Optional[np.ndarray] = None
  x_axis: Optional[np.ndarray] = None
  y_axis: Optional[np.ndarray] = None
  z_axis: Optional[np.ndarray] = None
  radius: Optional[float] = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.origin = as_vector(self.origin)
    if self.axis is not None:
      self.axis = normalize(self.axis)
    if self.normal is not None:
      self.normal = normalize(self.normal)
    if self.x_axis is not None:
      self.x_axis = normalize(self.x_axis)
    if self.y_axis is not None:
      self.y_axis = normalize(self.y_axis)
    if self.z_axis is not None:
      self.z_axis = normalize(self.z_axis)
    if self.radius is not None:
      self.radius = float(self.radius)
    self._ensure_basis()

  def _ensure_basis(self) -> None:
    preferred_z = self.z_axis
    if preferred_z is None:
      preferred_z = self.axis if self.axis is not None else self.normal
    if preferred_z is None:
      return
    basis_x = self.x_axis
    basis_y = self.y_axis
    if basis_x is not None and basis_y is None:
      tangent_x = basis_x - float(np.dot(basis_x, preferred_z)) * preferred_z
      if float(np.linalg.norm(tangent_x)) > 1e-10:
        basis_x = normalize(tangent_x)
        basis_y = normalize(np.cross(preferred_z, basis_x))
      else:
        basis_x = None
    elif basis_y is not None and basis_x is None:
      tangent_y = basis_y - float(np.dot(basis_y, preferred_z)) * preferred_z
      if float(np.linalg.norm(tangent_y)) > 1e-10:
        basis_y = normalize(tangent_y)
        basis_x = normalize(np.cross(basis_y, preferred_z))
      else:
        basis_y = None
    if basis_x is None or basis_y is None:
      basis_x, basis_y, preferred_z = orthonormal_basis_from_z(preferred_z)
    self.x_axis = basis_x
    self.y_axis = basis_y
    self.z_axis = preferred_z

  def copy(self) -> "Socket":
    return Socket(
        name=self.name,
        kind=self.kind,
        origin=_copy_vec(self.origin),
        axis=None if self.axis is None else _copy_vec(self.axis),
        normal=None if self.normal is None else _copy_vec(self.normal),
        x_axis=None if self.x_axis is None else _copy_vec(self.x_axis),
        y_axis=None if self.y_axis is None else _copy_vec(self.y_axis),
        z_axis=None if self.z_axis is None else _copy_vec(self.z_axis),
        radius=self.radius,
        metadata=dict(self.metadata),
    )

  def transformed(self, transform: Transform) -> SocketFrame:
    axis = None
    if self.axis is not None:
      axis = normalize(transform.apply_direction(self.axis))
    normal = None
    if self.normal is not None:
      normal = normalize(transform.apply_direction(self.normal))
    x_axis = None
    if self.x_axis is not None:
      x_axis = normalize(transform.apply_direction(self.x_axis))
    y_axis = None
    if self.y_axis is not None:
      y_axis = normalize(transform.apply_direction(self.y_axis))
    z_axis = None
    if self.z_axis is not None:
      z_axis = normalize(transform.apply_direction(self.z_axis))
    return SocketFrame(
        origin=transform.apply_point(self.origin),
        axis=axis,
        normal=normal,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        radius=self.radius,
    )

  def local_frame(self) -> SocketFrame:
    return SocketFrame(
        origin=_copy_vec(self.origin),
        axis=None if self.axis is None else _copy_vec(self.axis),
        normal=None if self.normal is None else _copy_vec(self.normal),
        x_axis=None if self.x_axis is None else _copy_vec(self.x_axis),
        y_axis=None if self.y_axis is None else _copy_vec(self.y_axis),
        z_axis=None if self.z_axis is None else _copy_vec(self.z_axis),
        radius=self.radius,
    )

  def local_frame_variants(self) -> dict[str, SocketFrame]:
    result = {"primary": self.local_frame()}
    raw = self.metadata.get("frame_variants")
    if not isinstance(raw, dict):
      return result
    for name, item in raw.items():
      if not isinstance(name, str) or not isinstance(item, dict):
        continue
      origin = item.get("origin")
      if not isinstance(origin, (list, tuple, np.ndarray)):
        continue
      try:
        result[name] = SocketFrame(
            origin=np.asarray(origin, dtype=float),
            axis=(
                None
                if item.get("axis") is None
                else np.asarray(item.get("axis"), dtype=float)
            ),
            normal=(
                None
                if item.get("normal") is None
                else np.asarray(item.get("normal"), dtype=float)
            ),
            x_axis=(
                None
                if item.get("x_axis") is None
                else np.asarray(item.get("x_axis"), dtype=float)
            ),
            y_axis=(
                None
                if item.get("y_axis") is None
                else np.asarray(item.get("y_axis"), dtype=float)
            ),
            z_axis=(
                None
                if item.get("z_axis") is None
                else np.asarray(item.get("z_axis"), dtype=float)
            ),
            radius=(
                None
                if item.get("radius") is None
                else float(item.get("radius"))
            ),
        )
      except Exception:
        continue
    return result

  def world_frame_variants(
      self,
      transform: Transform,
  ) -> dict[str, SocketFrame]:
    result: dict[str, SocketFrame] = {}
    for name, frame in self.local_frame_variants().items():
      axis = None if frame.axis is None else normalize(transform.apply_direction(frame.axis))
      normal = None if frame.normal is None else normalize(transform.apply_direction(frame.normal))
      x_axis = None if frame.x_axis is None else normalize(transform.apply_direction(frame.x_axis))
      y_axis = None if frame.y_axis is None else normalize(transform.apply_direction(frame.y_axis))
      z_axis = None if frame.z_axis is None else normalize(transform.apply_direction(frame.z_axis))
      result[name] = SocketFrame(
          origin=transform.apply_point(frame.origin),
          axis=axis,
          normal=normal,
          x_axis=x_axis,
          y_axis=y_axis,
          z_axis=z_axis,
          radius=frame.radius,
      )
    return result

  def to_dict(self) -> dict[str, Any]:
    return {
        "name": self.name,
        "kind": self.kind,
        "origin": self.origin.tolist(),
        "axis": None if self.axis is None else self.axis.tolist(),
        "normal": None if self.normal is None else self.normal.tolist(),
        "x_axis": None if self.x_axis is None else self.x_axis.tolist(),
        "y_axis": None if self.y_axis is None else self.y_axis.tolist(),
        "z_axis": None if self.z_axis is None else self.z_axis.tolist(),
        "radius": self.radius,
        "metadata": dict(self.metadata),
    }


@dataclass
class PartTemplate:
  """Reusable part archetype."""

  name: str
  sockets: dict[str, Socket]
  local_bbox_min: np.ndarray
  local_bbox_max: np.ndarray
  default_params: dict[str, float] = field(default_factory=dict)
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.local_bbox_min = as_vector(self.local_bbox_min)
    self.local_bbox_max = as_vector(self.local_bbox_max)

  def instantiate(
      self, instance_id: str, params: Optional[dict[str, float]] = None
  ) -> "PartInstance":
    merged = dict(self.default_params)
    if params:
      merged.update(params)
    instance = PartInstance(
        instance_id=instance_id,
        template_name=self.name,
        sockets={name: socket.copy() for name, socket in self.sockets.items()},
        params=merged,
        local_bbox_min=_copy_vec(self.local_bbox_min),
        local_bbox_max=_copy_vec(self.local_bbox_max),
        metadata=dict(self.metadata),
    )
    instance.apply_param_bindings()
    return instance


@dataclass
class PartInstance:
  """Part instance that can be transformed and morphed."""

  instance_id: str
  template_name: str
  sockets: dict[str, Socket]
  params: dict[str, float]
  local_bbox_min: np.ndarray
  local_bbox_max: np.ndarray
  transform: Optional[Transform] = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def copy(self) -> "PartInstance":
    return PartInstance(
        instance_id=self.instance_id,
        template_name=self.template_name,
        sockets={name: socket.copy() for name, socket in self.sockets.items()},
        params=dict(self.params),
        local_bbox_min=_copy_vec(self.local_bbox_min),
        local_bbox_max=_copy_vec(self.local_bbox_max),
        transform=None if self.transform is None else self.transform.copy(),
        metadata=dict(self.metadata),
    )

  def apply_param_bindings(self) -> None:
    """Update socket radii from bound parameters."""
    for socket in self.sockets.values():
      radius_param = socket.metadata.get("radius_param")
      if not radius_param:
        continue
      if radius_param not in self.params:
        continue
      scale = float(socket.metadata.get("radius_scale", 1.0))
      socket.radius = float(self.params[radius_param]) * scale

  def try_set_socket_radius(self, socket_name: str, new_radius: float) -> bool:
    if socket_name not in self.sockets:
      return False
    socket = self.sockets[socket_name]
    if new_radius <= 0:
      return False
    socket.radius = float(new_radius)
    radius_param = socket.metadata.get("radius_param")
    if radius_param:
      scale = float(socket.metadata.get("radius_scale", 1.0))
      if scale == 0:
        return False
      self.params[radius_param] = float(new_radius) / scale
    return True

  def world_socket(self, socket_name: str) -> SocketFrame:
    if socket_name not in self.sockets:
      raise KeyError(
          f"Socket '{socket_name}' not found in part '{self.instance_id}'."
      )
    if self.transform is None:
      raise ValueError(f"Part '{self.instance_id}' has no solved transform.")
    return self.sockets[socket_name].transformed(self.transform)

  def to_dict(self) -> dict[str, Any]:
    return {
        "instance_id": self.instance_id,
        "template_name": self.template_name,
        "params": dict(self.params),
        "sockets": {
            name: socket.to_dict() for name, socket in self.sockets.items()
        },
        "local_bbox_min": self.local_bbox_min.tolist(),
        "local_bbox_max": self.local_bbox_max.tolist(),
        "transform": None if self.transform is None else self.transform.to_dict(),
        "metadata": dict(self.metadata),
    }


class ConstraintType(str, Enum):
  CONCENTRIC = "concentric"
  COINCIDENT = "coincident"
  DISTANCE = "distance"


@dataclass
class CanonicalMateEdge:
  """High-level assembly relation before socket grounding."""

  part_a: str
  part_b: str
  relation_type: str
  label: str = ""
  stack_group: Optional[str] = None
  order_hint: Optional[int] = None
  source_frame_hint: Optional[str] = None
  target_frame_hint: Optional[str] = None
  source_stop_hint: Optional[str] = None
  target_stop_hint: Optional[str] = None
  stack_phase: Optional[str] = None
  alignment_mode: Optional[str] = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
        "part_a": self.part_a,
        "part_b": self.part_b,
        "relation_type": self.relation_type,
        "label": self.label,
        "stack_group": self.stack_group,
        "order_hint": self.order_hint,
        "source_frame_hint": self.source_frame_hint,
        "target_frame_hint": self.target_frame_hint,
        "source_stop_hint": self.source_stop_hint,
        "target_stop_hint": self.target_stop_hint,
        "stack_phase": self.stack_phase,
        "alignment_mode": self.alignment_mode,
        "metadata": dict(self.metadata),
    }


@dataclass
class CanonicalStackGroup:
  """Ordered coaxial assembly chain used by symbolic axial docking."""

  group_id: str
  parts: list[str]
  anchor_part: Optional[str] = None
  relation_type: str = "coaxial"
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
        "group_id": self.group_id,
        "parts": list(self.parts),
        "anchor_part": self.anchor_part,
        "relation_type": self.relation_type,
        "metadata": dict(self.metadata),
    }


@dataclass
class CanonicalMateGraph:
  """Planner-side canonical graph between language and geometric grounding."""

  anchor: str
  edges: list[CanonicalMateEdge]
  role_assignments: dict[str, str] = field(default_factory=dict)
  stack_groups: list[CanonicalStackGroup] = field(default_factory=list)
  plan_summary: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
        "anchor": self.anchor,
        "edges": [edge.to_dict() for edge in self.edges],
        "role_assignments": dict(self.role_assignments),
        "stack_groups": [group.to_dict() for group in self.stack_groups],
        "plan_summary": self.plan_summary,
        "metadata": dict(self.metadata),
    }


@dataclass
class Constraint:
  """Directed geometric relation from socket A to socket B."""

  ctype: ConstraintType
  part_a: str
  socket_a: str
  part_b: str
  socket_b: str
  value: Optional[float] = None
  label: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)

  def signature(self) -> tuple[Any, ...]:
    return (
        self.ctype.value,
        self.part_a,
        self.socket_a,
        self.part_b,
        self.socket_b,
        None if self.value is None else float(self.value),
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "type": self.ctype.value,
        "part_a": self.part_a,
        "socket_a": self.socket_a,
        "part_b": self.part_b,
        "socket_b": self.socket_b,
        "value": self.value,
        "label": self.label,
        "metadata": dict(self.metadata),
    }


@dataclass
class ConstraintGraph:
  """Neural layer output: lightweight, typed topology constraint graph."""

  instances: dict[str, PartInstance]
  constraints: list[Constraint]
  anchor: str
  metadata: dict[str, Any] = field(default_factory=dict)

  def copy(self) -> "ConstraintGraph":
    return ConstraintGraph(
        instances={name: inst.copy() for name, inst in self.instances.items()},
        constraints=[Constraint(**c.__dict__) for c in self.constraints],
        anchor=self.anchor,
        metadata=dict(self.metadata),
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "anchor": self.anchor,
        "instances": {
            name: instance.to_dict() for name, instance in self.instances.items()
        },
        "constraints": [constraint.to_dict() for constraint in self.constraints],
        "metadata": dict(self.metadata),
    }


@dataclass
class CollisionRecord:
  """Collision information from deterministic physical validation."""

  part_a: str
  part_b: str
  volume: float
  overlap_min: np.ndarray
  overlap_max: np.ndarray
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.overlap_min = as_vector(self.overlap_min)
    self.overlap_max = as_vector(self.overlap_max)
    self.volume = float(self.volume)

  def message(self) -> str:
    return (
        f"collision({self.part_a},{self.part_b}) "
        f"volume={self.volume:.6f}"
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "part_a": self.part_a,
        "part_b": self.part_b,
        "volume": self.volume,
        "overlap_min": self.overlap_min.tolist(),
        "overlap_max": self.overlap_max.tolist(),
        "metadata": dict(self.metadata),
    }


@dataclass
class PlannerFeedback:
  """Error and verification feedback sent from symbolic layer to planner."""

  messages: list[str] = field(default_factory=list)
  collisions: list[CollisionRecord] = field(default_factory=list)


@dataclass
class AssemblyState:
  """Result of symbolic solving for one constraint graph."""

  instances: dict[str, PartInstance]
  constraint_graph: ConstraintGraph
  logs: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
        "instances": {
            name: instance.to_dict() for name, instance in self.instances.items()
        },
        "constraint_graph": self.constraint_graph.to_dict(),
        "logs": list(self.logs),
    }
