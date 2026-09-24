"""CadQuery/OpenCASCADE backend for STEP loading and exact collision checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue as py_queue
from typing import Any, Optional

import numpy as np

from .benchmark_v2_constants import (
    EXACT_FACE_IDENTITY_PROOFS,
    exact_face_binding_sha256,
)
from .collision import CollisionChecker
from .offline import BRepFace, SocketExtractionConfig, build_template_from_faces
from .domain_types import AssemblyState, CollisionRecord, PartInstance, PartTemplate, Transform


class CadKernelError(RuntimeError):
  """Raised when CadQuery/OCC operations fail."""


class CadKernelChildProcessError(CadKernelError):
  """A started isolated OCC child reached a non-success terminal state."""

  def __init__(
      self, *, operation: str, status: str, error: str, exit_code: int,
  ) -> None:
    if status not in {"error", "timeout", "native_exit"}:
      raise ValueError("OCC child terminal status differs")
    if type(exit_code) is not int:
      raise ValueError("OCC child exit code differs")
    self.operation = operation
    self.status = status
    self.error = error
    self.exit_code = exit_code
    super().__init__(f"{operation}:{status}:{error}")


def _raise_exact_child_failure(operation: str, result: dict[str, Any]) -> None:
  """Preserve whether a worker was started when surfacing an OCC failure."""

  error = str(result.get("error") or "unknown_exact_worker_failure")
  if "process_spawn_failed=" in error or error.startswith("missing_"):
    raise CadKernelError(f"{operation}:pre_spawn:{error}")
  status = "timeout" if result.get("status") == "timeout" else "error"
  exit_code = -9 if status == "timeout" else 0
  marker = "process_exit="
  if marker in error:
    status = "native_exit"
    try:
      exit_code = int(error.split(marker, 1)[1].split(":", 1)[0])
    except ValueError:
      exit_code = -1
  raise CadKernelChildProcessError(
      operation=operation, status=status, error=error, exit_code=exit_code
  )


def _import_cadquery():
  try:
    import cadquery as cq  # type: ignore
  except ImportError as exc:
    raise CadKernelError(
        "CadQuery backend requested but `cadquery` is not installed."
    ) from exc
  return cq


def _shape_from_any(obj: Any) -> Any:
  if hasattr(obj, "val"):
    return obj.val()
  return obj


def _vec_to_np(vec: Any) -> np.ndarray:
  if vec is None:
    raise CadKernelError("Cannot convert None vector.")
  if hasattr(vec, "toTuple"):
    tup = vec.toTuple()
    return np.array([float(tup[0]), float(tup[1]), float(tup[2])], dtype=float)
  if hasattr(vec, "X") and callable(vec.X):
    return np.array([float(vec.X()), float(vec.Y()), float(vec.Z())], dtype=float)
  if hasattr(vec, "x") and hasattr(vec, "y") and hasattr(vec, "z"):
    return np.array([float(vec.x), float(vec.y), float(vec.z)], dtype=float)
  arr = np.asarray(vec, dtype=float).reshape(3)
  return arr


def load_step_shape(step_path: str | Path) -> Any:
  """Load STEP file into a CadQuery shape."""
  cq = _import_cadquery()
  step_path = str(step_path)
  if not Path(step_path).exists():
    raise CadKernelError(f"STEP file not found: {step_path}")
  loaded = cq.importers.importStep(step_path)
  shape = _shape_from_any(loaded)
  if shape is None:
    raise CadKernelError(f"Failed to parse STEP file: {step_path}")
  return shape


def shape_bbox(shape: Any) -> tuple[np.ndarray, np.ndarray]:
  shape = _shape_from_any(shape)
  bb = shape.BoundingBox()
  return (
      np.array([float(bb.xmin), float(bb.ymin), float(bb.zmin)], dtype=float),
      np.array([float(bb.xmax), float(bb.ymax), float(bb.zmax)], dtype=float),
  )


def source_face_signature_sha256(face: Any) -> str:
  """Fingerprint one imported STEP face to detect re-import re-enumeration.

  The signature is certificate-only and contains source-frame geometry only
  inside its SHA-256 preimage.  Its position-sensitive centre and bounds make
  equal-looking symmetric faces distinguishable in all non-coincident cases.
  """

  value = _shape_from_any(face)
  try:
    bounds = value.BoundingBox()
    center = _vec_to_np(value.Center())
    payload = {
        "schema": "step_face_reimport_signature.v1",
        "surface_type": str(value.geomType()).strip().upper(),
        "area": format(float(value.Area()), ".12g"),
        "center": [format(float(item), ".12g") for item in center],
        "bounds": [
            format(float(item), ".12g")
            for item in (
                bounds.xmin,
                bounds.ymin,
                bounds.zmin,
                bounds.xmax,
                bounds.ymax,
                bounds.zmax,
            )
        ],
        "edge_count": len(list(value.Edges())),
    }
  except Exception as exc:
    raise CadKernelError(
        "Failed to fingerprint an imported STEP face: "
        f"{type(exc).__name__}:{exc}"
    ) from exc
  encoded = json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def shape_volume(shape: Any) -> float:
  shape = _shape_from_any(shape)
  if shape is None:
    return 0.0
  if hasattr(shape, "Volume"):
    vol_attr = shape.Volume
    volume = vol_attr() if callable(vol_attr) else vol_attr
    return float(volume)
  if hasattr(shape, "val"):
    return shape_volume(shape.val())
  raise CadKernelError("Cannot read volume from shape object.")


def shape_minimum_distance(shape_a: Any, shape_b: Any) -> float:
  """Return the exact OCCT minimum distance between two shapes in millimetres."""

  shape_a = _shape_from_any(shape_a)
  shape_b = _shape_from_any(shape_b)
  try:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape  # type: ignore

    extrema = BRepExtrema_DistShapeShape(shape_a.wrapped, shape_b.wrapped)
    extrema.Perform()
    if not extrema.IsDone():
      raise CadKernelError("OCCT two-shape distance did not complete.")
    distance = float(extrema.Value())
    if not math.isfinite(distance) or distance < 0.0:
      raise CadKernelError("OCCT returned an invalid two-shape distance.")
    return distance
  except CadKernelError:
    raise
  except Exception as exc:
    raise CadKernelError(
        f"Two-shape minimum-distance query failed: {type(exc).__name__}: {exc}"
    ) from exc


def transform_shape(shape: Any, transform: Transform) -> Any:
  """Apply a complete rigid transform to a CadQuery shape.

  This function deliberately fails closed.  A historical translation-only
  fallback could silently discard rotation when CadQuery rejected the matrix,
  invalidating collision and exact-evaluation results.
  """
  try:
    # Snapshot and revalidate at the CAD boundary as defense in depth, even
    # though Transform's public arrays are immutable copies.
    validated_transform = Transform(
        rotation=np.array(transform.rotation, dtype=float, copy=True),
        translation=np.array(transform.translation, dtype=float, copy=True),
    )
    rot = validated_transform.rotation
    tr = validated_transform.translation
    cq = _import_cadquery()
    shape = _shape_from_any(shape)

    from OCP.gp import gp_Trsf  # type: ignore

    # Construct a rigid gp_Trsf directly. CadQuery's nested-list ``Matrix``
    # builds a gp_GTrsf which OCCT 7.9 may reject when converted back to a
    # rigid transform. gp_Trsf.SetValues orthogonalizes the 3x3 block and keeps
    # the complete translation. ``transformShape`` then preserves analytic
    # geometry types (unlike ``transformGeometry``).
    occ_transform = gp_Trsf()
    occ_transform.SetValues(
        float(rot[0, 0]),
        float(rot[0, 1]),
        float(rot[0, 2]),
        float(tr[0]),
        float(rot[1, 0]),
        float(rot[1, 1]),
        float(rot[1, 2]),
        float(tr[1]),
        float(rot[2, 0]),
        float(rot[2, 1]),
        float(rot[2, 2]),
        float(tr[2]),
    )
    transformed = shape.transformShape(cq.Matrix(occ_transform))
    if transformed is None:
      raise CadKernelError("CadQuery returned no shape for the rigid transform.")
    return transformed
  except CadKernelError:
    raise
  except Exception as exc:
    raise CadKernelError(
        "Failed to apply the full rigid transform; rotation was not discarded: "
        f"{type(exc).__name__}: {exc}"
    ) from exc


def boolean_intersection(shape_a: Any, shape_b: Any) -> Any:
  shape_a = _shape_from_any(shape_a)
  shape_b = _shape_from_any(shape_b)
  try:
    return shape_a.intersect(shape_b)
  except Exception:
    cq = _import_cadquery()
    try:
      wp_a = cq.Workplane(obj=shape_a)
      wp_b = cq.Workplane(obj=shape_b)
      return wp_a.intersect(wp_b).val()
    except Exception as exc:
      raise CadKernelError(f"Boolean intersection failed: {exc}") from exc


def _resolve_shape_for_instance(
    shape_library: dict[str, Any],
    instance: PartInstance,
) -> Any:
  base_shape = shape_library.get(instance.template_name)
  if base_shape is None:
    base_shape = shape_library.get(instance.instance_id)
  if base_shape is None:
    raise CadKernelError(
        f"Missing base shape for template '{instance.template_name}'."
    )
  return base_shape


def export_assembly_step(
    assembly: AssemblyState,
    shape_library: dict[str, Any],
    output_step_path: str | Path,
) -> Path:
  """Export solved assembly to STEP using CadQuery Assembly."""
  cq = _import_cadquery()
  output_path = Path(output_step_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  assy = cq.Assembly(name="neurocad_assembly")
  for part_id, instance in assembly.instances.items():
    if instance.transform is None:
      raise CadKernelError(
          f"Cannot export assembly: part '{part_id}' has no solved transform."
      )
    base_shape = _shape_from_any(
        _resolve_shape_for_instance(shape_library=shape_library, instance=instance)
    )
    try:
      world_shape = transform_shape(base_shape, instance.transform)
    except Exception as exc:
      raise CadKernelError(
          f"Failed to transform '{part_id}' for export: {exc}"
      ) from exc
    assy.add(_shape_from_any(world_shape), name=part_id)

  try:
    assy.save(str(output_path), exportType="STEP")
  except Exception:
    try:
      assy.save(str(output_path))
    except Exception as exc:
      raise CadKernelError(
          f"Failed to export assembly STEP '{output_path}': {exc}"
      ) from exc
  return output_path


def _face_axis_radius_origin(
    face: Any,
) -> tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
  """Read cylinder axis/radius/axis point via OCC adaptor when available."""
  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomAbs import GeomAbs_Cylinder  # type: ignore
  except ImportError:
    return None, None, None

  try:
    adaptor = BRepAdaptor_Surface(face.wrapped)
    if adaptor.GetType() != GeomAbs_Cylinder:
      return None, None, None
    cyl = adaptor.Cylinder()
    direction = cyl.Axis().Direction()
    location = cyl.Axis().Location()
    axis = np.array(
        [float(direction.X()), float(direction.Y()), float(direction.Z())],
        dtype=float,
    )
    radius = float(cyl.Radius())
    origin = np.array(
        [float(location.X()), float(location.Y()), float(location.Z())],
        dtype=float,
    )
    return axis, radius, origin
  except Exception:
    return None, None, None


def _face_normal(face: Any) -> Optional[np.ndarray]:
  try:
    normal = face.normalAt()
    return _vec_to_np(normal)
  except Exception:
    return None


def _face_probe_point_and_normal(
    face: Any,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """Return a point on the face and its oriented solid-face normal."""

  try:
    u_min, u_max, v_min, v_max = face.uvBounds()
    point = face.positionAt(
        0.5 * (float(u_min) + float(u_max)),
        0.5 * (float(v_min) + float(v_max)),
    )
    return _vec_to_np(point), _vec_to_np(face.normalAt(point))
  except Exception:
    return None, None


def _shape_complexity_metadata(faces: list[BRepFace]) -> dict[str, Any]:
  face_count = len(faces)
  helical_face_count = sum(1 for face in faces if face.has_helical_edge)
  spline_face_count = sum(
      1
      for face in faces
      if any(
          token in face.surface_type
          for token in ("spline", "bspline", "bezier", "nurbs")
      )
  )
  max_edge_count = max((face.edge_count for face in faces), default=0)
  complexity_score = (
      float(face_count)
      + 10.0 * float(helical_face_count)
      + 4.0 * float(spline_face_count)
      + 0.15 * float(max_edge_count)
  )
  return {
      "face_count": int(face_count),
      "helical_face_count": int(helical_face_count),
      "spline_face_count": int(spline_face_count),
      "max_edge_count": int(max_edge_count),
      "complexity_score": float(complexity_score),
  }


def _normalized_geometry_protocol(protocol: str) -> str:
  value = str(protocol or "legacy").strip().lower()
  if value not in {"legacy", "benchmark_v2"}:
    raise ValueError(
        "geometry protocol must be 'legacy' or 'benchmark_v2', "
        f"got {protocol!r}"
    )
  return value


def extract_brep_faces_from_shape(
    shape: Any,
    *,
    protocol: str = "legacy",
) -> list[BRepFace]:
  """Extract BRepFace descriptors from a CadQuery shape.

  ``legacy`` preserves the historical AABB-centre concavity heuristic.
  ``benchmark_v2`` uses only a cylinder's oriented radial geometry, which is
  invariant to a rigid change of gauge.  Non-cylindrical concavity is omitted
  from the benchmark descriptor because no equally reliable invariant
  predicate is available at this boundary.
  """
  protocol = _normalized_geometry_protocol(protocol)
  shape = _shape_from_any(shape)
  faces_raw = list(shape.Faces())
  bbox_min, bbox_max = shape_bbox(shape)
  center_model = (bbox_min + bbox_max) / 2.0

  result: list[BRepFace] = []
  for face_index, face in enumerate(faces_raw):
    try:
      geom_type = str(face.geomType()).lower()
    except Exception:
      geom_type = "unknown"

    try:
      area = float(face.Area())
    except Exception:
      continue
    if area <= 0:
      continue

    try:
      raw_center = _vec_to_np(face.Center())
    except Exception:
      continue

    normal = _face_normal(face)
    axis, radius, axis_origin = _face_axis_radius_origin(face)
    center = raw_center
    axis_center = None
    if axis is not None and axis_origin is not None:
      try:
        axis_unit = axis / max(1e-12, float(np.linalg.norm(axis)))
        axis_center = axis_origin + float(np.dot(raw_center - axis_origin, axis_unit)) * axis_unit
        center = axis_center
      except Exception:
        axis_center = None
    edge_count = 0
    has_helical_edge = False
    try:
      edges = list(face.Edges())
      edge_count = len(edges)
      for edge in edges:
        et = str(edge.geomType()).lower()
        if "helix" in et or "spiral" in et:
          has_helical_edge = True
          break
    except Exception:
      edge_count = 0

    bbox_extents = None
    try:
      bb = face.BoundingBox()
      bbox_extents = np.array(
          [
              float(bb.xmax - bb.xmin),
              float(bb.ymax - bb.ymin),
              float(bb.zmax - bb.zmin),
          ],
          dtype=float,
      )
    except Exception:
      bbox_extents = None

    # Use a stable boundary vertex to define an intrinsic in-face direction.
    # Unlike a world-axis fallback, this direction rigidly follows the B-Rep
    # under benchmark-v2 gauges.  Ties keep topology enumeration order.
    reference_direction = None
    axial_extent = None
    preferred_normal = axis if axis is not None else normal
    if preferred_normal is not None:
      try:
        direction_axis = preferred_normal / max(
            1e-12, float(np.linalg.norm(preferred_normal))
        )
        projected_vertices: list[tuple[float, np.ndarray]] = []
        for vertex in list(face.Vertices()):
          delta = _vec_to_np(vertex.Center()) - center
          tangent = delta - float(np.dot(delta, direction_axis)) * direction_axis
          tangent_norm = float(np.linalg.norm(tangent))
          if tangent_norm > 1e-10:
            projected_vertices.append((tangent_norm, tangent))
        if projected_vertices:
          maximum_norm = max(item[0] for item in projected_vertices)
          for tangent_norm, tangent in projected_vertices:
            if tangent_norm >= maximum_norm * (1.0 - 1e-9):
              reference_direction = tangent / tangent_norm
              break
        if axis is not None:
          axial_coordinates = [
              float(np.dot(_vec_to_np(vertex.Center()) - axis_center, direction_axis))
              for vertex in list(face.Vertices())
          ]
          if axial_coordinates:
            axial_extent = max(axial_coordinates) - min(axial_coordinates)
      except Exception:
        reference_direction = None
        axial_extent = None

    concavity = 0
    if protocol == "benchmark_v2":
      # For an analytic cylinder the signed radial normal distinguishes the
      # material-facing outer wall (+) from a void-facing bore wall (-).  Both
      # vectors receive the same rotation and translation, so their dot product
      # is genuinely SE(3)-invariant.  AABB centres are deliberately excluded.
      probe_point, probe_normal = _face_probe_point_and_normal(face)
      if (
          probe_point is not None
          and probe_normal is not None
          and axis is not None
          and axis_origin is not None
      ):
        axis_unit = axis / max(1e-12, float(np.linalg.norm(axis)))
        probe_axis_center = axis_origin + float(
            np.dot(probe_point - axis_origin, axis_unit)
        ) * axis_unit
        radial = probe_point - probe_axis_center
        score = float(np.dot(probe_normal, radial))
        if not math.isclose(score, 0.0, abs_tol=1e-8):
          concavity = -1 if score < 0 else 1
    elif normal is not None:
      radial = raw_center - center_model
      score = float(np.dot(normal, radial))
      if not math.isclose(score, 0.0, abs_tol=1e-8):
        concavity = -1 if score < 0 else 1

    result.append(
        BRepFace(
            surface_type=geom_type,
            area=area,
            center=center,
            normal=normal,
            axis=axis,
            radius=radius,
            concavity=concavity,
            edge_count=edge_count,
            has_helical_edge=has_helical_edge,
            bbox_extents=bbox_extents,
            metadata={
                "source": "cadquery_step",
                "face_index": int(face_index),
                "raw_face_center": raw_center.tolist(),
                "axis_center": None if axis_center is None else axis_center.tolist(),
                "axis_origin": None if axis_origin is None else axis_origin.tolist(),
                "axial_extent": (
                    None if axial_extent is None else float(axial_extent)
                ),
                "reference_direction": (
                    None
                    if reference_direction is None
                    else reference_direction.tolist()
                ),
            },
        )
    )

  return result


@dataclass
class StepPartAsset:
  template: PartTemplate
  shape: Any


def build_template_from_step(
    name: str,
    step_path: str | Path,
    default_params: Optional[dict[str, float]] = None,
    metadata: Optional[dict[str, object]] = None,
    extraction_config: Optional[SocketExtractionConfig] = None,
) -> StepPartAsset:
  """Build PartTemplate from real STEP geometry."""
  shape = load_step_shape(step_path)
  return build_template_from_shape(
      name=name,
      shape=shape,
      default_params=default_params,
      metadata=metadata,
      source_step_path=step_path,
      extraction_config=extraction_config,
      include_equivariant_frame_metadata=False,
      protocol="legacy",
  )


def build_template_from_shape(
    name: str,
    shape: Any,
    default_params: Optional[dict[str, float]] = None,
    metadata: Optional[dict[str, object]] = None,
    source_step_path: str | Path | None = None,
    extraction_config: Optional[SocketExtractionConfig] = None,
    include_equivariant_frame_metadata: bool = True,
    protocol: str = "legacy",
) -> StepPartAsset:
  """Build a template from an already loaded (and possibly gauged) B-Rep."""

  protocol = _normalized_geometry_protocol(protocol)
  faces = extract_brep_faces_from_shape(shape, protocol=protocol)
  if not include_equivariant_frame_metadata:
    # The existing STEP wrapper must not perturb legacy socket metadata or
    # evidence hashes.  The benchmark-v2 shape boundary retains this metadata.
    for face in faces:
      face.metadata.pop("reference_direction", None)
  bbox_min, bbox_max = shape_bbox(shape)
  complexity = _shape_complexity_metadata(faces)
  merged_metadata = {
      **complexity,
      **({} if metadata is None else dict(metadata)),
  }
  if protocol == "benchmark_v2":
    part_surface_area = sum(max(0.0, float(face.area)) for face in faces)
    if part_surface_area <= 0.0:
      raise CadKernelError("benchmark_v2 requires positive part surface area")
    merged_metadata.update(
        {
            "model_input_protocol": "benchmark_v2",
            "benchmark_v2_part_surface_area": float(part_surface_area),
            "benchmark_v2_invariant_part_scale": float(math.sqrt(part_surface_area)),
        }
    )
  if source_step_path is not None:
    merged_metadata["step_path"] = str(source_step_path)
  template = build_template_from_faces(
      name=name,
      faces=faces,
      local_bbox_min=bbox_min,
      local_bbox_max=bbox_max,
      default_params=default_params,
      metadata=merged_metadata,
      extraction_config=extraction_config,
  )
  return StepPartAsset(template=template, shape=shape)


def _safe_exact_boolean_worker(
    step_path_a: str,
    rotation_a: list[list[float]],
    translation_a: list[float],
    local_rotation_a: list[list[float]],
    local_translation_a: list[float],
    step_path_b: str,
    rotation_b: list[list[float]],
    translation_b: list[float],
    local_rotation_b: list[list[float]],
    local_translation_b: list[float],
    result_queue,
) -> None:
  try:
    transform_a = Transform(
        rotation=np.asarray(rotation_a, dtype=float),
        translation=np.asarray(translation_a, dtype=float),
    )
    transform_b = Transform(
        rotation=np.asarray(rotation_b, dtype=float),
        translation=np.asarray(translation_b, dtype=float),
    )
    local_a = transform_shape(
        load_step_shape(step_path_a),
        Transform(
            rotation=np.asarray(local_rotation_a, dtype=float),
            translation=np.asarray(local_translation_a, dtype=float),
        ),
    )
    local_b = transform_shape(
        load_step_shape(step_path_b),
        Transform(
            rotation=np.asarray(local_rotation_b, dtype=float),
            translation=np.asarray(local_translation_b, dtype=float),
        ),
    )
    shape_a = transform_shape(local_a, transform_a)
    shape_b = transform_shape(local_b, transform_b)
    inter = boolean_intersection(shape_a, shape_b)
    volume = shape_volume(inter)
    overlap_min = None
    overlap_max = None
    if volume > 0.0:
      try:
        ov_min, ov_max = shape_bbox(inter)
        overlap_min = ov_min.tolist()
        overlap_max = ov_max.tolist()
      except Exception:
        overlap_min = [0.0, 0.0, 0.0]
        overlap_max = [0.0, 0.0, 0.0]
    result_queue.put(
        {
            "status": "ok",
            "volume": float(volume),
            "overlap_min": overlap_min,
            "overlap_max": overlap_max,
        }
    )
  except Exception as exc:  # pylint: disable=broad-except
    result_queue.put(
        {
            "status": "error",
            "error": f"{type(exc).__name__}:{exc}",
        }
    )


def _safe_exact_distance_worker(
    step_path_a: str,
    rotation_a: list[list[float]],
    translation_a: list[float],
    local_rotation_a: list[list[float]],
    local_translation_a: list[float],
    step_path_b: str,
    rotation_b: list[list[float]],
    translation_b: list[float],
    local_rotation_b: list[list[float]],
    local_translation_b: list[float],
    result_queue,
) -> None:
  try:
    transform_a = Transform(
        rotation=np.asarray(rotation_a, dtype=float),
        translation=np.asarray(translation_a, dtype=float),
    )
    transform_b = Transform(
        rotation=np.asarray(rotation_b, dtype=float),
        translation=np.asarray(translation_b, dtype=float),
    )
    local_a = transform_shape(
        load_step_shape(step_path_a),
        Transform(
            rotation=np.asarray(local_rotation_a, dtype=float),
            translation=np.asarray(local_translation_a, dtype=float),
        ),
    )
    local_b = transform_shape(
        load_step_shape(step_path_b),
        Transform(
            rotation=np.asarray(local_rotation_b, dtype=float),
            translation=np.asarray(local_translation_b, dtype=float),
        ),
    )
    shape_a = transform_shape(local_a, transform_a)
    shape_b = transform_shape(local_b, transform_b)
    result = {
        "status": "ok",
        "distance": float(shape_a.distance(shape_b)),
    }
    try:
      from OCP.BRepExtrema import BRepExtrema_DistShapeShape  # type: ignore

      extrema = BRepExtrema_DistShapeShape(shape_a.wrapped, shape_b.wrapped)
      extrema.Perform()
      if extrema.IsDone():
        point_a = extrema.PointOnShape1(1)
        point_b = extrema.PointOnShape2(1)
        p_a = [float(point_a.X()), float(point_a.Y()), float(point_a.Z())]
        p_b = [float(point_b.X()), float(point_b.Y()), float(point_b.Z())]
        vector = [p_b[i] - p_a[i] for i in range(3)]
        result.update(
            {
                "distance": float(extrema.Value()),
                "point_on_shape_a": p_a,
                "point_on_shape_b": p_b,
                "vector_a_to_b": vector,
            }
        )
    except Exception:
      pass
    result_queue.put(result)
  except Exception as exc:  # pylint: disable=broad-except
    result_queue.put(
      {
          "status": "error",
          "error": f"{type(exc).__name__}:{exc}",
      }
    )


def _safe_exact_interface_distance_worker(
    step_path_a: str,
    rotation_a: list[list[float]],
    translation_a: list[float],
    local_rotation_a: list[list[float]],
    local_translation_a: list[float],
    raw_face_indices_a: list[int],
    raw_face_signature_sha256s_a: list[str],
    step_path_b: str,
    rotation_b: list[list[float]],
    translation_b: list[float],
    local_rotation_b: list[list[float]],
    local_translation_b: list[float],
    raw_face_indices_b: list[int],
    raw_face_signature_sha256s_b: list[str],
    result_queue,
) -> None:
  """Measure only between the selected source B-Rep faces."""

  try:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape  # type: ignore

    source_a = _shape_from_any(load_step_shape(step_path_a))
    source_b = _shape_from_any(load_step_shape(step_path_b))
    faces_a = list(source_a.Faces())
    faces_b = list(source_b.Faces())
    clean_a = sorted({int(index) for index in raw_face_indices_a})
    clean_b = sorted({int(index) for index in raw_face_indices_b})
    if not clean_a or not clean_b:
      raise CadKernelError("intended interface has no source face identity")
    if clean_a[0] < 0 or clean_a[-1] >= len(faces_a):
      raise CadKernelError("intended interface A face index is out of range")
    if clean_b[0] < 0 or clean_b[-1] >= len(faces_b):
      raise CadKernelError("intended interface B face index is out of range")
    expected_signatures_a = [str(value) for value in raw_face_signature_sha256s_a]
    expected_signatures_b = [str(value) for value in raw_face_signature_sha256s_b]
    if len(expected_signatures_a) != len(clean_a) or len(
        expected_signatures_b
    ) != len(clean_b):
      raise CadKernelError(
          "intended interface STEP face signature cardinality is inconsistent"
      )
    actual_signatures_a = [
        source_face_signature_sha256(faces_a[index]) for index in clean_a
    ]
    actual_signatures_b = [
        source_face_signature_sha256(faces_b[index]) for index in clean_b
    ]
    if actual_signatures_a != expected_signatures_a:
      raise CadKernelError(
          "re-imported STEP face enumeration changed for intended interface A"
      )
    if actual_signatures_b != expected_signatures_b:
      raise CadKernelError(
          "re-imported STEP face enumeration changed for intended interface B"
      )

    local_a = Transform(
        rotation=np.asarray(local_rotation_a, dtype=float),
        translation=np.asarray(local_translation_a, dtype=float),
    )
    local_b = Transform(
        rotation=np.asarray(local_rotation_b, dtype=float),
        translation=np.asarray(local_translation_b, dtype=float),
    )
    world_a = Transform(
        rotation=np.asarray(rotation_a, dtype=float),
        translation=np.asarray(translation_a, dtype=float),
    )
    world_b = Transform(
        rotation=np.asarray(rotation_b, dtype=float),
        translation=np.asarray(translation_b, dtype=float),
    )
    selected_a = [
        transform_shape(transform_shape(faces_a[index], local_a), world_a)
        for index in clean_a
    ]
    selected_b = [
        transform_shape(transform_shape(faces_b[index], local_b), world_b)
        for index in clean_b
    ]

    best: dict[str, Any] | None = None
    for offset_a, face_a in enumerate(selected_a):
      for offset_b, face_b in enumerate(selected_b):
        extrema = BRepExtrema_DistShapeShape(face_a.wrapped, face_b.wrapped)
        extrema.Perform()
        if not extrema.IsDone() or extrema.NbSolution() < 1:
          continue
        distance = float(extrema.Value())
        if not math.isfinite(distance) or distance < 0.0:
          raise CadKernelError(
              "OCC returned a non-finite or negative intended-interface distance"
          )
        point_a = extrema.PointOnShape1(1)
        point_b = extrema.PointOnShape2(1)
        point_a_values = np.asarray(
            [float(point_a.X()), float(point_a.Y()), float(point_a.Z())],
            dtype=float,
        )
        point_b_values = np.asarray(
            [float(point_b.X()), float(point_b.Y()), float(point_b.Z())],
            dtype=float,
        )
        if not np.all(np.isfinite(point_a_values)) or not np.all(
            np.isfinite(point_b_values)
        ):
          raise CadKernelError("OCC returned non-finite intended-interface points")
        point_distance = float(np.linalg.norm(point_b_values - point_a_values))
        if not math.isclose(
            point_distance,
            distance,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
          raise CadKernelError(
              "OCC intended-interface distance disagrees with closest points"
          )
        payload = {
            "status": "ok",
            "distance": distance,
            "raw_face_index_a": clean_a[offset_a],
            "raw_face_index_b": clean_b[offset_b],
            "point_on_shape_a": point_a_values.tolist(),
            "point_on_shape_b": point_b_values.tolist(),
        }
        payload["vector_a_to_b"] = [
            payload["point_on_shape_b"][axis]
            - payload["point_on_shape_a"][axis]
            for axis in range(3)
        ]
        if best is None or distance < float(best["distance"]):
          best = payload
    if best is None:
      raise CadKernelError("OCC returned no intended-interface distance solution")
    best["raw_face_indices_a"] = clean_a
    best["raw_face_indices_b"] = clean_b
    result_queue.put(best)
  except Exception as exc:  # pylint: disable=broad-except
    result_queue.put(
        {
            "status": "error",
            "error": f"{type(exc).__name__}:{exc}",
        }
    )


class CadQueryCollisionChecker(CollisionChecker):
  """Exact collision checker using OCC boolean intersection volume."""

  def __init__(
      self,
      shape_library: dict[str, Any],
      allow_aabb_fallback: bool = True,
      exact_timeout_seconds: float = 8.0,
      skip_exact_for_complex_pairs: bool = True,
      complexity_face_threshold: int = 180,
      complexity_score_threshold: float = 240.0,
      complex_pair_aabb_ratio_threshold: float = 0.02,
      exact_worker_only: bool = False,
  ):
    self.shape_library = dict(shape_library)
    self.allow_aabb_fallback = bool(allow_aabb_fallback)
    self.exact_timeout_seconds = max(0.5, float(exact_timeout_seconds))
    self.skip_exact_for_complex_pairs = bool(skip_exact_for_complex_pairs)
    self.complexity_face_threshold = max(1, int(complexity_face_threshold))
    self.complexity_score_threshold = float(complexity_score_threshold)
    self.complex_pair_aabb_ratio_threshold = max(
        0.0, float(complex_pair_aabb_ratio_threshold)
    )
    self.exact_worker_only = bool(exact_worker_only)

  def _world_shape(self, instance) -> Any:
    if instance.transform is None:
      raise CadKernelError(
          f"Instance '{instance.instance_id}' has no solved transform."
      )
    base_shape = self.shape_library.get(instance.template_name)
    if base_shape is None:
      base_shape = self.shape_library.get(instance.instance_id)
    if base_shape is None:
      raise CadKernelError(
          f"Missing base shape for template '{instance.template_name}'."
      )
    return transform_shape(base_shape, instance.transform)

  @staticmethod
  def _world_aabb_without_native_matmul(instance: PartInstance) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact-worker broad phase without loading a BLAS backend."""

    if instance.transform is None:
      raise CadKernelError(
          f"Instance '{instance.instance_id}' has no solved transform."
      )
    rotation = instance.transform.rotation.tolist()
    translation = instance.transform.translation.tolist()
    local_min = instance.local_bbox_min.tolist()
    local_max = instance.local_bbox_max.tolist()
    world_points = []
    for x in (local_min[0], local_max[0]):
      for y in (local_min[1], local_max[1]):
        for z in (local_min[2], local_max[2]):
          local = (x, y, z)
          world_points.append(tuple(
              translation[axis] + sum(
                  rotation[axis][column] * local[column] for column in range(3)
              )
              for axis in range(3)
          ))
    return (
        np.asarray([min(point[axis] for point in world_points) for axis in range(3)]),
        np.asarray([max(point[axis] for point in world_points) for axis in range(3)]),
    )

  def check(
      self, assembly: AssemblyState, epsilon: float = 1e-6
  ) -> list[CollisionRecord]:
    ids = list(assembly.instances.keys())
    world_shapes: dict[str, Any] = {}
    aabb_bounds = {}
    unplaced: list[str] = []
    for part_id in ids:
      instance = assembly.instances[part_id]
      if instance.transform is None:
        unplaced.append(part_id)
        continue
      aabb_bounds[part_id] = (
          self._world_aabb_without_native_matmul(instance)
          if self.exact_worker_only
          else super().world_aabb(instance)
      )

    if unplaced:
      raise CadKernelError(
          "Cannot run collision validation with unplaced parts: "
          + ", ".join(sorted(unplaced))
      )

    if not self.exact_worker_only:
      for part_id in ids:
        try:
          world_shapes[part_id] = self._world_shape(assembly.instances[part_id])
        except CadKernelError:
          continue

    records: list[CollisionRecord] = []
    for i in range(len(ids)):
      for j in range(i + 1, len(ids)):
        part_a = ids[i]
        part_b = ids[j]
        min_a, max_a = aabb_bounds[part_a]
        min_b, max_b = aabb_bounds[part_b]
        overlap_min = np.maximum(min_a, min_b)
        overlap_max = np.minimum(max_a, max_b)
        overlap = overlap_max - overlap_min
        if np.any(overlap <= 0.0):
          continue
        aabb_volume = float(np.prod(overlap))
        if aabb_volume <= epsilon:
          continue

        instance_a = assembly.instances[part_a]
        instance_b = assembly.instances[part_b]
        try_exact = True
        if self.skip_exact_for_complex_pairs and self._is_complex_pair(
            instance_a, instance_b
        ):
          overlap_ratio = self._aabb_overlap_ratio(
              min_a=min_a,
              max_a=max_a,
              min_b=min_b,
              max_b=max_b,
              overlap_volume=aabb_volume,
          )
          if overlap_ratio <= self.complex_pair_aabb_ratio_threshold:
            try_exact = False

        if self.exact_worker_only or (
            part_a in world_shapes and part_b in world_shapes
        ):
          if try_exact:
            exact = self._safe_exact_intersection(
                instance_a=instance_a,
                instance_b=instance_b,
            )
            if exact["status"] == "ok":
              volume = float(exact.get("volume", 0.0))
              if volume > epsilon:
                ov_min = np.asarray(
                    exact.get("overlap_min") or [0.0, 0.0, 0.0],
                    dtype=float,
                )
                ov_max = np.asarray(
                    exact.get("overlap_max") or [0.0, 0.0, 0.0],
                    dtype=float,
                )
                records.append(
                    CollisionRecord(
                        part_a=part_a,
                        part_b=part_b,
                        volume=volume,
                        overlap_min=ov_min,
                        overlap_max=ov_max,
                    )
                )
              continue
            if not self.allow_aabb_fallback:
              _raise_exact_child_failure("whole_solid_interference", exact)

        if not self.allow_aabb_fallback:
          raise CadKernelError(
              f"Missing or invalid CadQuery shape for pair: {part_a}, {part_b}"
          )
        records.append(
            CollisionRecord(
                part_a=part_a,
                part_b=part_b,
                volume=aabb_volume,
                overlap_min=overlap_min,
                overlap_max=overlap_max,
            )
        )

    return records

  def distance_between(
      self,
      assembly: AssemblyState,
      part_a: str,
      part_b: str,
  ) -> float:
    if part_a not in assembly.instances:
      raise CadKernelError(f"Unknown part in distance query: {part_a}")
    if part_b not in assembly.instances:
      raise CadKernelError(f"Unknown part in distance query: {part_b}")
    instance_a = assembly.instances[part_a]
    instance_b = assembly.instances[part_b]
    if instance_a.transform is None or instance_b.transform is None:
      raise CadKernelError(
          f"Cannot run distance query with unplaced pair: {part_a}, {part_b}"
      )
    exact = self._safe_exact_distance(instance_a=instance_a, instance_b=instance_b)
    if exact["status"] == "ok":
      return float(exact.get("distance", 0.0))
    _raise_exact_child_failure("whole_solid_distance", exact)

  def distance_details_between(
      self,
      assembly: AssemblyState,
      part_a: str,
      part_b: str,
  ) -> dict[str, Any]:
    if part_a not in assembly.instances:
      raise CadKernelError(f"Unknown part in distance query: {part_a}")
    if part_b not in assembly.instances:
      raise CadKernelError(f"Unknown part in distance query: {part_b}")
    instance_a = assembly.instances[part_a]
    instance_b = assembly.instances[part_b]
    if instance_a.transform is None or instance_b.transform is None:
      raise CadKernelError(
          f"Cannot run distance query with unplaced pair: {part_a}, {part_b}"
      )
    exact = self._safe_exact_distance(instance_a=instance_a, instance_b=instance_b)
    if exact["status"] == "ok":
      return dict(exact)
    _raise_exact_child_failure("whole_solid_distance", exact)

  def interface_distance_details_between(
      self,
      assembly: AssemblyState,
      part_a: str,
      socket_a: str,
      part_b: str,
      socket_b: str,
  ) -> dict[str, Any]:
    """Measure exact distance between the two selected B-Rep interfaces.

    Unlike :meth:`distance_details_between`, this cannot be satisfied by a
    nearby, unintended face elsewhere on either part.
    """

    if part_a not in assembly.instances or part_b not in assembly.instances:
      raise CadKernelError("Unknown part in intended-interface distance query")
    instance_a = assembly.instances[part_a]
    instance_b = assembly.instances[part_b]
    if socket_a not in instance_a.sockets or socket_b not in instance_b.sockets:
      raise CadKernelError("Unknown socket in intended-interface distance query")
    if instance_a.transform is None or instance_b.transform is None:
      raise CadKernelError("Cannot measure intended interfaces on unplaced parts")
    indices_a = self._raw_face_indices_for_socket(
        instance_a,
        instance_a.sockets[socket_a],
    )
    indices_b = self._raw_face_indices_for_socket(
        instance_b,
        instance_b.sockets[socket_b],
    )
    signatures_a = list(
        instance_a.sockets[socket_a].metadata.get(
            "private_exact_raw_face_signature_sha256s"
        )
        or []
    )
    signatures_b = list(
        instance_b.sockets[socket_b].metadata.get(
            "private_exact_raw_face_signature_sha256s"
        )
        or []
    )
    exact = self._safe_exact_interface_distance(
        instance_a=instance_a,
        raw_face_indices_a=indices_a,
        raw_face_signature_sha256s_a=signatures_a,
        instance_b=instance_b,
        raw_face_indices_b=indices_b,
        raw_face_signature_sha256s_b=signatures_b,
    )
    if exact.get("status") == "ok":
      result = dict(exact)
      result.update(
          {
              "part_a": str(part_a),
              "socket_a": str(socket_a),
              "part_b": str(part_b),
              "socket_b": str(socket_b),
              "interface_id_a": str(
                  instance_a.sockets[socket_a].metadata.get(
                      "private_exact_interface_id"
                  )
                  or ""
              ),
              "interface_id_b": str(
                  instance_b.sockets[socket_b].metadata.get(
                      "private_exact_interface_id"
                  )
                  or ""
              ),
          }
      )
      return result
    _raise_exact_child_failure("intended_interface_distance", exact)

  def _safe_exact_intersection(
      self,
      instance_a: PartInstance,
      instance_b: PartInstance,
  ) -> dict[str, Any]:
    step_path_a = instance_a.metadata.get("step_path")
    step_path_b = instance_b.metadata.get("step_path")
    if not isinstance(step_path_a, str) or not isinstance(step_path_b, str):
      return {"status": "error", "error": "missing_step_path"}
    if instance_a.transform is None or instance_b.transform is None:
      return {"status": "error", "error": "missing_transform"}

    ctx = mp.get_context("spawn")
    result_queue = None
    process = None
    try:
      local_a = self._local_geometry_transform(instance_a)
      local_b = self._local_geometry_transform(instance_b)
      result_queue = ctx.Queue(maxsize=1)
      process = ctx.Process(
          target=_safe_exact_boolean_worker,
          args=(
              step_path_a,
              instance_a.transform.rotation.tolist(),
              instance_a.transform.translation.tolist(),
              local_a.rotation.tolist(),
              local_a.translation.tolist(),
              step_path_b,
              instance_b.transform.rotation.tolist(),
              instance_b.transform.translation.tolist(),
              local_b.rotation.tolist(),
              local_b.translation.tolist(),
              result_queue,
          ),
          daemon=True,
      )
      process.start()
      process.join(self.exact_timeout_seconds)
      if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
        return {"status": "timeout", "error": "boolean_timeout"}

      try:
        result = result_queue.get_nowait()
      except py_queue.Empty:
        return {
            "status": "error",
            "error": f"boolean_process_exit={process.exitcode}",
        }
      if not isinstance(result, dict):
        return {"status": "error", "error": "invalid_boolean_result"}
      return result
    except (PermissionError, OSError) as exc:
      return {
          "status": "error",
          "error": f"boolean_process_spawn_failed={type(exc).__name__}:{exc}",
      }
    finally:
      if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
      if result_queue is not None:
        try:
          result_queue.close()
        except Exception:
          pass

  def _safe_exact_distance(
      self,
      instance_a: PartInstance,
      instance_b: PartInstance,
  ) -> dict[str, Any]:
    step_path_a = instance_a.metadata.get("step_path")
    step_path_b = instance_b.metadata.get("step_path")
    if not isinstance(step_path_a, str) or not isinstance(step_path_b, str):
      return {"status": "error", "error": "missing_step_path"}
    if instance_a.transform is None or instance_b.transform is None:
      return {"status": "error", "error": "missing_transform"}

    ctx = mp.get_context("spawn")
    result_queue = None
    process = None
    try:
      local_a = self._local_geometry_transform(instance_a)
      local_b = self._local_geometry_transform(instance_b)
      result_queue = ctx.Queue(maxsize=1)
      process = ctx.Process(
          target=_safe_exact_distance_worker,
          args=(
              step_path_a,
              instance_a.transform.rotation.tolist(),
              instance_a.transform.translation.tolist(),
              local_a.rotation.tolist(),
              local_a.translation.tolist(),
              step_path_b,
              instance_b.transform.rotation.tolist(),
              instance_b.transform.translation.tolist(),
              local_b.rotation.tolist(),
              local_b.translation.tolist(),
              result_queue,
          ),
          daemon=True,
      )
      process.start()
      process.join(self.exact_timeout_seconds)
      if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
        return {"status": "timeout", "error": "distance_timeout"}

      try:
        result = result_queue.get_nowait()
      except py_queue.Empty:
        return {
            "status": "error",
            "error": f"distance_process_exit={process.exitcode}",
        }
      if not isinstance(result, dict):
        return {"status": "error", "error": "invalid_distance_result"}
      return result
    except (PermissionError, OSError) as exc:
      return {
          "status": "error",
          "error": f"distance_process_spawn_failed={type(exc).__name__}:{exc}",
      }
    finally:
      if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
      if result_queue is not None:
        try:
          result_queue.close()
        except Exception:
          pass

  def _safe_exact_interface_distance(
      self,
      *,
      instance_a: PartInstance,
      raw_face_indices_a: list[int],
      raw_face_signature_sha256s_a: list[str],
      instance_b: PartInstance,
      raw_face_indices_b: list[int],
      raw_face_signature_sha256s_b: list[str],
  ) -> dict[str, Any]:
    step_path_a = instance_a.metadata.get("step_path")
    step_path_b = instance_b.metadata.get("step_path")
    if not isinstance(step_path_a, str) or not isinstance(step_path_b, str):
      return {"status": "error", "error": "missing_step_path"}
    if instance_a.transform is None or instance_b.transform is None:
      return {"status": "error", "error": "missing_transform"}

    ctx = mp.get_context("spawn")
    result_queue = None
    process = None
    try:
      local_a = self._local_geometry_transform(instance_a)
      local_b = self._local_geometry_transform(instance_b)
      result_queue = ctx.Queue(maxsize=1)
      process = ctx.Process(
          target=_safe_exact_interface_distance_worker,
          args=(
              step_path_a,
              instance_a.transform.rotation.tolist(),
              instance_a.transform.translation.tolist(),
              local_a.rotation.tolist(),
              local_a.translation.tolist(),
              list(raw_face_indices_a),
              list(raw_face_signature_sha256s_a),
              step_path_b,
              instance_b.transform.rotation.tolist(),
              instance_b.transform.translation.tolist(),
              local_b.rotation.tolist(),
              local_b.translation.tolist(),
              list(raw_face_indices_b),
              list(raw_face_signature_sha256s_b),
              result_queue,
          ),
          daemon=True,
      )
      process.start()
      process.join(self.exact_timeout_seconds)
      if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
        return {"status": "timeout", "error": "interface_distance_timeout"}
      try:
        result = result_queue.get_nowait()
      except py_queue.Empty:
        return {
            "status": "error",
            "error": f"interface_distance_process_exit={process.exitcode}",
        }
      if not isinstance(result, dict):
        return {"status": "error", "error": "invalid_interface_distance_result"}
      return result
    except (PermissionError, OSError) as exc:
      return {
          "status": "error",
          "error": (
              "interface_distance_process_spawn_failed="
              f"{type(exc).__name__}:{exc}"
          ),
      }
    finally:
      if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
      if result_queue is not None:
        try:
          result_queue.close()
        except Exception:
          pass

  def _raw_face_indices_for_socket(
      self,
      instance: PartInstance,
      socket: Any,
  ) -> list[int]:
    metadata = socket.metadata if isinstance(socket.metadata, dict) else {}
    raw = metadata.get("private_exact_raw_face_indices")
    is_benchmark_v2 = str(instance.metadata.get("frame_protocol") or "").startswith(
        "benchmark_v2"
    )
    if is_benchmark_v2:
      proof = metadata.get("private_exact_face_identity_proof")
      if proof not in EXACT_FACE_IDENTITY_PROOFS:
        raise CadKernelError(
            "benchmark_v2 intended interface lacks a protocol-approved "
            "face-identity proof"
        )
      if not isinstance(raw, list) or not raw:
        raise CadKernelError(
            "benchmark_v2 intended interface lacks source face identities"
        )
      if any(isinstance(index, bool) or not isinstance(index, int) for index in raw):
        raise CadKernelError(
            "benchmark_v2 intended interface face identities must be integers"
        )
    if not isinstance(raw, list):
      raw = list(metadata.get("member_face_indices") or [])
      if not raw and isinstance(metadata.get("face_index"), int):
        raw = [int(metadata["face_index"])]
    try:
      clean = sorted({int(index) for index in raw})
    except (TypeError, ValueError) as exc:
      raise CadKernelError("intended interface face identities are malformed") from exc
    if not clean or clean[0] < 0:
      raise CadKernelError("intended interface has no valid source face identity")
    if is_benchmark_v2:
      namespace = instance.metadata.get("benchmark_v2_exact_identity_namespace")
      raw_gauge = instance.metadata.get(
          "benchmark_v2_raw_to_randomized_transform"
      )
      interface_id = metadata.get("private_exact_interface_id")
      raw_signatures = metadata.get(
          "private_exact_raw_face_signature_sha256s"
      )
      supplied_binding = metadata.get("private_exact_binding_sha256")
      if not isinstance(namespace, str) or not namespace:
        raise CadKernelError(
            "benchmark_v2 intended interface lacks its private identity namespace"
        )
      if not isinstance(raw_gauge, dict):
        raise CadKernelError(
            "benchmark_v2 intended interface lacks its private geometry gauge"
        )
      if not isinstance(interface_id, str) or not interface_id:
        raise CadKernelError(
            "benchmark_v2 intended interface lacks its private interface identity"
        )
      if (
          not isinstance(raw_signatures, list)
          or len(raw_signatures) != len(clean)
          or any(
              not isinstance(value, str)
              or len(value) != 64
              or any(character not in "0123456789abcdef" for character in value)
              for value in raw_signatures
          )
      ):
        raise CadKernelError(
            "benchmark_v2 intended interface lacks valid STEP face signatures"
        )
      if (
          not isinstance(supplied_binding, str)
          or len(supplied_binding) != 64
          or any(character not in "0123456789abcdef" for character in supplied_binding)
      ):
        raise CadKernelError(
            "benchmark_v2 intended interface lacks a valid face binding"
        )
      try:
        expected_binding = exact_face_binding_sha256(
            namespace=namespace,
            raw_to_randomized_transform=raw_gauge,
            interface_id=interface_id,
            raw_face_indices=clean,
            raw_face_signature_sha256s=raw_signatures,
            face_identity_proof=str(proof),
        )
      except (TypeError, ValueError) as exc:
        raise CadKernelError(
            "benchmark_v2 intended-interface binding payload is malformed"
        ) from exc
      if supplied_binding != expected_binding:
        raise CadKernelError(
            "benchmark_v2 intended-interface face binding does not match its "
            "identity, proof, indices, and gauge"
        )
    return clean

  def _local_geometry_transform(self, instance: PartInstance) -> Transform:
    """Return the private transform from source STEP to evaluator-local shape.

    Legacy assets use only their historical canonicalization translation.
    Benchmark-v2 assets instead carry the complete per-part randomized gauge.
    A present but malformed gauge is rejected rather than silently replaced
    with identity, because that would make exact checks use a different frame.
    """

    is_benchmark_v2 = str(instance.metadata.get("frame_protocol") or "").startswith(
        "benchmark_v2"
    )
    raw_gauge = instance.metadata.get(
        "benchmark_v2_raw_to_randomized_transform"
    )
    if raw_gauge is not None:
      if not isinstance(raw_gauge, dict):
        raise CadKernelError("benchmark_v2 exact gauge must be a transform mapping")
      rotation = raw_gauge.get("rotation")
      translation = raw_gauge.get("translation")
      if rotation is None or translation is None:
        raise CadKernelError("benchmark_v2 exact gauge is incomplete")
      try:
        return Transform(
            rotation=np.asarray(rotation, dtype=float),
            translation=np.asarray(translation, dtype=float),
        )
      except Exception as exc:
        raise CadKernelError(
            "benchmark_v2 exact gauge is not a valid rigid transform: "
            f"{type(exc).__name__}:{exc}"
        ) from exc
    if is_benchmark_v2:
      raise CadKernelError(
          "benchmark_v2 exact geometry is missing its raw-to-randomized gauge"
      )

    raw = instance.metadata.get("canonicalization_offset")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
      try:
        offset = np.asarray([float(raw[0]), float(raw[1]), float(raw[2])])
        return Transform(rotation=np.eye(3, dtype=float), translation=offset)
      except (TypeError, ValueError):
        raise CadKernelError("canonicalization_offset is malformed")
    return Transform.identity()

  def _is_complex_pair(
      self,
      instance_a: PartInstance,
      instance_b: PartInstance,
  ) -> bool:
    for instance in (instance_a, instance_b):
      face_count = int(instance.metadata.get("face_count", 0) or 0)
      complexity_score = float(
          instance.metadata.get("complexity_score", 0.0) or 0.0
      )
      if face_count >= self.complexity_face_threshold:
        return True
      if complexity_score >= self.complexity_score_threshold:
        return True
    return False

  def _aabb_overlap_ratio(
      self,
      min_a: np.ndarray,
      max_a: np.ndarray,
      min_b: np.ndarray,
      max_b: np.ndarray,
      overlap_volume: float,
  ) -> float:
    vol_a = float(np.prod(np.maximum(max_a - min_a, 1e-9)))
    vol_b = float(np.prod(np.maximum(max_b - min_b, 1e-9)))
    denom = max(min(vol_a, vol_b), 1e-9)
    return float(overlap_volume / denom)
