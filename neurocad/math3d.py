"""3D linear algebra helpers for rigid assembly constraints."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-9


def as_vector(data: Iterable[float] | np.ndarray) -> np.ndarray:
  vec = np.asarray(data, dtype=float).reshape(3)
  return vec


def norm(vec: Iterable[float] | np.ndarray) -> float:
  return float(np.linalg.norm(as_vector(vec)))


def normalize(vec: Iterable[float] | np.ndarray) -> np.ndarray:
  v = as_vector(vec)
  n = np.linalg.norm(v)
  if n < EPS:
    raise ValueError("Cannot normalize near-zero vector.")
  return v / n


def orthonormal_basis_from_z(
    z_axis: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Build a deterministic right-handed basis from a preferred z axis."""
  z_u = normalize(z_axis)
  pivot = np.array([1.0, 0.0, 0.0], dtype=float)
  if abs(float(np.dot(z_u, pivot))) > 0.92:
    pivot = np.array([0.0, 1.0, 0.0], dtype=float)
  x_u = normalize(np.cross(pivot, z_u))
  y_u = normalize(np.cross(z_u, x_u))
  return x_u, y_u, z_u


def skew(vec: Iterable[float] | np.ndarray) -> np.ndarray:
  x, y, z = as_vector(vec)
  return np.array([
      [0.0, -z, y],
      [z, 0.0, -x],
      [-y, x, 0.0],
  ])


def rotation_about_axis(
    axis: Iterable[float] | np.ndarray, angle_radians: float
) -> np.ndarray:
  axis_u = normalize(axis)
  x, y, z = axis_u
  c = math.cos(angle_radians)
  s = math.sin(angle_radians)
  one_minus_c = 1.0 - c

  return np.array([
      [c + x * x * one_minus_c, x * y * one_minus_c - z * s,
       x * z * one_minus_c + y * s],
      [y * x * one_minus_c + z * s, c + y * y * one_minus_c,
       y * z * one_minus_c - x * s],
      [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s,
       c + z * z * one_minus_c],
  ])


def rotation_from_to(
    source: Iterable[float] | np.ndarray, target: Iterable[float] | np.ndarray
) -> np.ndarray:
  """Compute rotation matrix R where R @ source ~= target."""
  a = normalize(source)
  b = normalize(target)
  dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

  if dot > 1.0 - EPS:
    return np.eye(3)

  if dot < -1.0 + EPS:
    # 180 degree rotation around any axis orthogonal to source.
    pivot = np.array([1.0, 0.0, 0.0])
    if abs(a[0]) > 0.9:
      pivot = np.array([0.0, 1.0, 0.0])
    axis = normalize(np.cross(a, pivot))
    return rotation_about_axis(axis, math.pi)

  axis = normalize(np.cross(a, b))
  angle = math.acos(dot)
  return rotation_about_axis(axis, angle)


def almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
  return abs(a - b) <= tol
