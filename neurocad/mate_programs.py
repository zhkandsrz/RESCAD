"""Reusable mate-program data structures and SE(3) helpers.

Mate programs are small, learned/retrieved residual transforms between two
semantic mating coordinate frames (MCFs).  They are intentionally independent
of a particular solver: a search module can propose interface pairs, retrieve a
few programs, and then use these helpers to instantiate candidate child poses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Optional

import numpy as np

from .math3d import normalize, orthonormal_basis_from_z
from .domain_types import Socket, SocketFrame, Transform


@dataclass
class MateProgram:
  """A directed residual transform from parent MCF to child MCF."""

  program_id: str
  case_id: str
  assembly_dir: str
  part_a: str
  part_b: str
  body_uuid_a: str
  body_uuid_b: str
  interface_a: dict[str, Any]
  interface_b: dict[str, Any]
  relation_hint: str
  contact_type: str
  residual_rotation: list[list[float]]
  residual_translation: list[float]
  source_contact: dict[str, Any]
  features: dict[str, float]
  metadata: dict[str, Any]

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "MateProgram":
    return cls(
        program_id=str(data.get("program_id") or ""),
        case_id=str(data.get("case_id") or ""),
        assembly_dir=str(data.get("assembly_dir") or ""),
        part_a=str(data.get("part_a") or ""),
        part_b=str(data.get("part_b") or ""),
        body_uuid_a=str(data.get("body_uuid_a") or ""),
        body_uuid_b=str(data.get("body_uuid_b") or ""),
        interface_a=dict(data.get("interface_a") or {}),
        interface_b=dict(data.get("interface_b") or {}),
        relation_hint=str(data.get("relation_hint") or ""),
        contact_type=str(data.get("contact_type") or ""),
        residual_rotation=[
            [float(x) for x in row]
            for row in (data.get("residual_rotation") or np.eye(3).tolist())
        ],
        residual_translation=[
            float(x) for x in (data.get("residual_translation") or [0.0, 0.0, 0.0])
        ],
        source_contact=dict(data.get("source_contact") or {}),
        features={
            str(k): float(v)
            for k, v in (data.get("features") or {}).items()
            if isinstance(v, (int, float))
        },
        metadata=dict(data.get("metadata") or {}),
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  def residual_transform(self) -> Transform:
    return Transform(
        rotation=np.asarray(self.residual_rotation, dtype=float).reshape(3, 3),
        translation=np.asarray(self.residual_translation, dtype=float).reshape(3),
    )

  @property
  def parent_role(self) -> str:
    return _interface_role(self.interface_a)

  @property
  def child_role(self) -> str:
    return _interface_role(self.interface_b)

  @property
  def parent_surface(self) -> str:
    return _interface_surface(self.interface_a)

  @property
  def child_surface(self) -> str:
    return _interface_surface(self.interface_b)

  @property
  def parent_radius(self) -> Optional[float]:
    return _metadata_radius(self.interface_a)

  @property
  def child_radius(self) -> Optional[float]:
    return _metadata_radius(self.interface_b)


def _metadata_radius(interface: dict[str, Any]) -> Optional[float]:
  model_view = interface.get("benchmark_v2_model_view")
  if isinstance(model_view, dict):
    geometry = model_view.get("geometry")
    if isinstance(geometry, dict):
      ratio = geometry.get("radius_ratio")
      if isinstance(ratio, (int, float)):
        return float(ratio)
  metadata = interface.get("metadata")
  if not isinstance(metadata, dict):
    return None
  raw = metadata.get("radius")
  if not isinstance(raw, (int, float)):
    return None
  return float(raw)


def _interface_surface(interface: dict[str, Any]) -> str:
  model_view = interface.get("benchmark_v2_model_view")
  if isinstance(model_view, dict):
    return str(model_view.get("surface_type") or "").lower()
  return str(interface.get("surface_type") or "").lower()


def _interface_role(interface: dict[str, Any]) -> str:
  model_view = interface.get("benchmark_v2_model_view")
  if not isinstance(model_view, dict):
    return str(interface.get("role_hint") or "").lower()
  topology = model_view.get("topology")
  if isinstance(topology, dict):
    role = str(topology.get("composite_role") or "").strip().lower()
    if role:
      return role
  return {
      "plane": "planar_seat",
      "cylinder": "cylindrical_interface",
      "cone": "shoulder_stop",
      "torus": "shoulder_stop",
  }.get(_interface_surface(interface), "generic_interface")


def matrix_from_transform(transform: Transform) -> np.ndarray:
  mat = np.eye(4, dtype=float)
  mat[:3, :3] = np.asarray(transform.rotation, dtype=float).reshape(3, 3)
  mat[:3, 3] = np.asarray(transform.translation, dtype=float).reshape(3)
  return mat


def transform_from_matrix(matrix: np.ndarray) -> Transform:
  mat = np.asarray(matrix, dtype=float).reshape(4, 4)
  return Transform(rotation=mat[:3, :3], translation=mat[:3, 3])


def invert_transform_matrix(matrix: np.ndarray) -> np.ndarray:
  mat = np.asarray(matrix, dtype=float).reshape(4, 4)
  rot = mat[:3, :3]
  trans = mat[:3, 3]
  inv = np.eye(4, dtype=float)
  inv[:3, :3] = rot.T
  inv[:3, 3] = -rot.T @ trans
  return inv


def frame_matrix_from_dict(frame: dict[str, Any]) -> np.ndarray:
  origin = np.asarray(frame.get("origin"), dtype=float).reshape(3)
  x_axis = _optional_vec(frame.get("x_axis"))
  y_axis = _optional_vec(frame.get("y_axis"))
  z_axis = _optional_vec(frame.get("z_axis"))
  if x_axis is None or y_axis is None or z_axis is None:
    preferred_z = _optional_vec(frame.get("axis"))
    if preferred_z is None:
      preferred_z = _optional_vec(frame.get("normal"))
    if preferred_z is None:
      raise ValueError("Frame is missing z/axis/normal direction.")
    x_axis, y_axis, z_axis = orthonormal_basis_from_z(preferred_z)
  return _frame_matrix(
      origin=origin,
      x_axis=x_axis,
      y_axis=y_axis,
      z_axis=z_axis,
  )


def frame_matrix_from_socket_frame(frame: SocketFrame) -> np.ndarray:
  basis = frame.basis_matrix()
  if basis is None:
    preferred_z = frame.z_axis
    if preferred_z is None:
      preferred_z = frame.axis if frame.axis is not None else frame.normal
    if preferred_z is None:
      raise ValueError("SocketFrame is missing basis directions.")
    x_axis, y_axis, z_axis = orthonormal_basis_from_z(preferred_z)
    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
  mat = np.eye(4, dtype=float)
  mat[:3, :3] = np.asarray(basis, dtype=float).reshape(3, 3)
  mat[:3, 3] = np.asarray(frame.origin, dtype=float).reshape(3)
  return mat


def frame_matrix_from_socket(socket: Socket, variant: str = "primary") -> np.ndarray:
  variants = socket.local_frame_variants()
  frame = variants.get(variant) or variants.get("primary")
  if frame is None:
    raise ValueError(f"Socket '{socket.name}' has no frame variant '{variant}'.")
  return frame_matrix_from_socket_frame(frame)


def residual_between_frames(
    parent_frame: dict[str, Any],
    child_frame: dict[str, Any],
) -> Transform:
  """Return M where parent_frame @ M == child_frame."""

  parent = frame_matrix_from_dict(parent_frame)
  child = frame_matrix_from_dict(child_frame)
  return transform_from_matrix(invert_transform_matrix(parent) @ child)


def instantiate_child_transform(
    *,
    parent_transform: Transform,
    parent_socket: Socket,
    child_socket: Socket,
    program: MateProgram,
    parent_variant: str = "primary",
    child_variant: str = "primary",
) -> Transform:
  """Apply a directed mate program to place the child part.

  The returned transform maps the child part's local coordinates into the same
  world frame as ``parent_transform``.
  """

  parent_world = matrix_from_transform(parent_transform)
  parent_frame = frame_matrix_from_socket(parent_socket, variant=parent_variant)
  child_frame = frame_matrix_from_socket(child_socket, variant=child_variant)
  residual = matrix_from_transform(program.residual_transform())
  child_world = parent_world @ parent_frame @ residual @ invert_transform_matrix(child_frame)
  return transform_from_matrix(child_world)


def rotation_angle_degrees(rotation: np.ndarray) -> float:
  rot = np.asarray(rotation, dtype=float).reshape(3, 3)
  trace = float(np.trace(rot))
  value = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
  return float(math.degrees(math.acos(value)))


def _frame_matrix(
    *,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
  x = normalize(np.asarray(x_axis, dtype=float).reshape(3))
  y = normalize(np.asarray(y_axis, dtype=float).reshape(3))
  z = normalize(np.asarray(z_axis, dtype=float).reshape(3))
  # Re-orthogonalize lightly to avoid noisy CAD face frames polluting residuals.
  x = normalize(x - float(np.dot(x, z)) * z)
  y = normalize(np.cross(z, x))
  x = normalize(np.cross(y, z))
  mat = np.eye(4, dtype=float)
  mat[:3, :3] = np.stack([x, y, z], axis=1)
  mat[:3, 3] = np.asarray(origin, dtype=float).reshape(3)
  return mat


def _optional_vec(value: Any) -> Optional[np.ndarray]:
  if value is None:
    return None
  try:
    return normalize(np.asarray(value, dtype=float).reshape(3))
  except Exception:
    return None
