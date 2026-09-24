"""Independent rigid-frame randomization for the benchmark-v2 protocol.

The functions in this module use the active-transform convention: ``T.apply_point``
maps a point into the transformed frame, and ``compose_se3(A, B)`` means ``A * B``
(apply ``B`` first, then ``A``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .domain_types import Transform


def compose_se3(left: Transform, right: Transform) -> Transform:
  """Return the rigid composition ``left * right``."""

  rotation = left.rotation @ right.rotation
  translation = left.rotation @ right.translation + left.translation
  return Transform(rotation=rotation, translation=translation)


def inverse_se3(transform: Transform) -> Transform:
  """Return the exact rigid inverse of ``transform``."""

  rotation = transform.rotation.T
  translation = -(rotation @ transform.translation)
  return Transform(rotation=rotation, translation=translation)


def transform_relative_label(
    *,
    source_gauge: Transform,
    target_gauge: Transform,
    ground_truth: Transform,
) -> Transform:
  """Apply the required label rule ``T_j * T_gt * T_i^-1`` exactly."""

  return compose_se3(
      target_gauge,
      compose_se3(ground_truth, inverse_se3(source_gauge)),
  )


def transform_interface_frame(
    frame: Mapping[str, Any],
    gauge: Transform,
) -> dict[str, Any]:
  """Transform a serialized interface frame without translating directions."""

  result = dict(frame)
  origin = frame.get("origin")
  if origin is None:
    raise ValueError("interface frame requires an origin")
  result["origin"] = gauge.apply_point(np.asarray(origin, dtype=float)).tolist()
  for key in ("axis", "normal", "x_axis", "y_axis", "z_axis"):
    value = frame.get(key)
    if value is not None:
      result[key] = gauge.apply_direction(np.asarray(value, dtype=float)).tolist()
  return result


def apply_gauge_to_shape(shape: Any, gauge: Transform) -> Any:
  """Apply the full rigid gauge at the B-Rep boundary, failing closed."""

  # Lazy import keeps the numeric SE(3) utilities usable without loading a CAD
  # kernel while still sharing the project's audited rigid-transform boundary.
  from .cadquery_backend import transform_shape

  return transform_shape(shape, gauge)


def sample_uniform_so3(rng: np.random.Generator) -> np.ndarray:
  """Sample Haar-uniform SO(3) using Shoemake's unit-quaternion construction."""

  u1, u2, u3 = (float(value) for value in rng.random(3))
  x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
  y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
  z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
  w = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
  return np.array(
      [
          [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
          [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
          [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
      ],
      dtype=float,
  )


def sample_se3(
    rng: np.random.Generator,
    *,
    assembly_scale: float,
    translation_box_fraction: float = 1.0,
) -> Transform:
  """Sample rotation uniformly and translation in a scale-normalized box."""

  scale = float(assembly_scale)
  fraction = float(translation_box_fraction)
  if not np.isfinite(scale) or scale <= 0.0:
    raise ValueError("assembly_scale must be finite and positive")
  if not np.isfinite(fraction) or fraction < 0.0:
    raise ValueError("translation_box_fraction must be finite and nonnegative")
  half_extent = scale * fraction
  translation = rng.uniform(-half_extent, half_extent, size=3)
  return Transform(
      rotation=sample_uniform_so3(rng),
      translation=translation,
  )


def assembly_scale_from_bounds(
    bounds: Iterable[tuple[Sequence[float], Sequence[float]]],
) -> float:
  """Return the diagonal of the union AABB used to normalize translations."""

  minima: list[np.ndarray] = []
  maxima: list[np.ndarray] = []
  for raw_minimum, raw_maximum in bounds:
    minimum = np.asarray(raw_minimum, dtype=float).reshape(3)
    maximum = np.asarray(raw_maximum, dtype=float).reshape(3)
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
      raise ValueError("assembly bounds must be finite")
    if np.any(maximum < minimum):
      raise ValueError("assembly bound maxima must not be below minima")
    minima.append(minimum)
    maxima.append(maximum)
  if not minima:
    raise ValueError("at least one part bound is required")
  global_minimum = np.min(np.stack(minima, axis=0), axis=0)
  global_maximum = np.max(np.stack(maxima, axis=0), axis=0)
  scale = float(np.linalg.norm(global_maximum - global_minimum))
  if not np.isfinite(scale) or scale <= 0.0:
    raise ValueError("assembly bounds must have positive diagonal scale")
  return scale


@dataclass(frozen=True, slots=True)
class PartGaugeSet:
  """Audit record for one independently randomized assembly."""

  seed: int
  assembly_scale: float
  translation_box_fraction: float
  transforms: Mapping[str, Transform]
  part_seeds: Mapping[str, int]
  assembly_nonce: str = ""
  protocol: str = "benchmark_v2_independent_se3"

  def transform_for(self, part_key: str) -> Transform:
    try:
      return self.transforms[str(part_key)]
    except KeyError as error:
      raise KeyError(f"Unknown randomized part key: {part_key}") from error

  def to_provenance(self) -> dict[str, object]:
    """Return audit-only provenance; this payload must not be model input."""

    return {
        "protocol": self.protocol,
        "seed": int(self.seed),
        "assembly_nonce": self.assembly_nonce,
        "assembly_scale": float(self.assembly_scale),
        "translation_box_fraction": float(self.translation_box_fraction),
        "parts": {
            key: {
                "seed": int(self.part_seeds[key]),
                "transform": transform.to_dict(),
            }
            for key, transform in self.transforms.items()
        },
    }


@dataclass(frozen=True, slots=True)
class RandomizedPartShapes:
  """Shapes and audit gauges emitted by the benchmark-v2 load boundary."""

  shapes: Mapping[str, Any]
  gauges: PartGaugeSet
  raw_transforms: Mapping[str, Transform]
  intrinsic_geometry: Mapping[str, "IntrinsicShapeGeometry"]

  def shape_for(self, part_key: str) -> Any:
    try:
      return self.shapes[str(part_key)]
    except KeyError as error:
      raise KeyError(f"Unknown randomized part key: {part_key}") from error

  def raw_transform_for(self, part_key: str) -> Transform:
    """Return the complete raw-shape to randomized-shape transform."""

    try:
      return self.raw_transforms[str(part_key)]
    except KeyError as error:
      raise KeyError(f"Unknown randomized part key: {part_key}") from error


@dataclass(frozen=True, slots=True)
class IntrinsicShapeGeometry:
  """Rigid-invariant size and an equivariant intrinsic origin for one B-Rep."""

  origin: tuple[float, float, float]
  surface_area: float
  volume: float
  characteristic_scale: float


def intrinsic_shape_geometry(shape: Any) -> IntrinsicShapeGeometry:
  """Measure a B-Rep without using its source placement or world-axis AABB."""

  value = shape.val() if hasattr(shape, "val") else shape
  try:
    center = value.Center()
    if hasattr(center, "toTuple"):
      raw_origin = center.toTuple()
    else:
      raw_origin = center
    origin_array = np.asarray(raw_origin, dtype=float).reshape(3)
    area_attr = getattr(value, "Area")
    volume_attr = getattr(value, "Volume", None)
    surface_area = float(area_attr() if callable(area_attr) else area_attr)
    volume = float(
        volume_attr() if callable(volume_attr) else (volume_attr or 0.0)
    )
  except Exception as error:
    raise ValueError("Cannot measure intrinsic B-Rep geometry") from error
  if not np.all(np.isfinite(origin_array)):
    raise ValueError("Intrinsic B-Rep origin must be finite")
  if np.isfinite(surface_area) and surface_area > 0.0:
    scale = float(np.sqrt(surface_area))
  elif np.isfinite(volume) and volume > 0.0:
    scale = float(np.cbrt(volume))
  else:
    raise ValueError("Intrinsic B-Rep geometry must have positive area or volume")
  return IntrinsicShapeGeometry(
      origin=tuple(float(value) for value in origin_array),
      surface_area=surface_area,
      volume=volume,
      characteristic_scale=scale,
  )


def intrinsic_assembly_scale(
    geometry: Iterable[IntrinsicShapeGeometry],
) -> float:
  """Aggregate per-part intrinsic lengths without using inter-part placement."""

  squared_scales = [float(item.characteristic_scale) ** 2 for item in geometry]
  if not squared_scales:
    raise ValueError("at least one intrinsic part geometry record is required")
  scale = float(np.sqrt(sum(squared_scales)))
  if not np.isfinite(scale) or scale <= 0.0:
    raise ValueError("intrinsic assembly scale must be finite and positive")
  return scale


def _part_seed(master_seed: int, part_key: str, assembly_nonce: str = "") -> int:
  if assembly_nonce:
    text = (
        "benchmark_v2_independent_se3_v2\0"
        f"{int(master_seed)}\0{assembly_nonce}\0{part_key}"
    )
  else:
    # Preserve the original standalone sampler stream for callers outside the
    # benchmark-v2 case protocol. Formal protocol entrypoints require a nonce.
    text = f"benchmark_v2_independent_se3\0{int(master_seed)}\0{part_key}"
  payload = text.encode("utf-8")
  return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sample_independent_part_gauges(
    part_keys: Iterable[str],
    *,
    seed: int,
    assembly_nonce: str = "",
    assembly_scale: float,
    translation_box_fraction: float = 1.0,
) -> PartGaugeSet:
  """Sample an order-independent deterministic ``T_i`` for every part."""

  keys = [str(key) for key in part_keys]
  if not keys or any(not key for key in keys):
    raise ValueError("part_keys must contain nonempty strings")
  if len(set(keys)) != len(keys):
    raise ValueError("part_keys must be unique")
  ordered_keys = sorted(keys)
  nonce = str(assembly_nonce)
  seeds = {
      key: _part_seed(int(seed), key, nonce) for key in ordered_keys
  }
  transforms = {
      key: sample_se3(
          np.random.default_rng(seeds[key]),
          assembly_scale=assembly_scale,
          translation_box_fraction=translation_box_fraction,
      )
      for key in ordered_keys
  }
  return PartGaugeSet(
      seed=int(seed),
      assembly_scale=float(assembly_scale),
      translation_box_fraction=float(translation_box_fraction),
      transforms=MappingProxyType(transforms),
      part_seeds=MappingProxyType(seeds),
      assembly_nonce=nonce,
  )


def randomize_loaded_part_shapes(
    part_shapes: Mapping[str, Any],
    *,
    seed: int,
    assembly_nonce: str = "",
    translation_box_fraction: float = 1.0,
) -> RandomizedPartShapes:
  """Randomize loaded B-Reps before any socket or learned-feature extraction."""

  if not part_shapes:
    raise ValueError("part_shapes must not be empty")
  keys = [str(key) for key in part_shapes]
  if len(set(keys)) != len(keys):
    raise ValueError("part shape keys must be unique after string normalization")
  keyed_shapes = {str(key): shape for key, shape in part_shapes.items()}
  geometry = {
      key: intrinsic_shape_geometry(keyed_shapes[key])
      for key in sorted(keyed_shapes)
  }
  scale = intrinsic_assembly_scale(geometry.values())
  gauges = sample_independent_part_gauges(
      keyed_shapes,
      seed=seed,
      assembly_nonce=assembly_nonce,
      assembly_scale=scale,
      translation_box_fraction=translation_box_fraction,
  )
  raw_transforms: dict[str, Transform] = {}
  randomized: dict[str, Any] = {}
  for key, shape in keyed_shapes.items():
    center_at_origin = Transform(
        translation=-np.asarray(geometry[key].origin, dtype=float)
    )
    raw_transform = compose_se3(gauges.transform_for(key), center_at_origin)
    raw_transforms[key] = raw_transform
    randomized[key] = apply_gauge_to_shape(shape, raw_transform)
  return RandomizedPartShapes(
      shapes=MappingProxyType(randomized),
      gauges=gauges,
      raw_transforms=MappingProxyType(raw_transforms),
      intrinsic_geometry=MappingProxyType(geometry),
  )
