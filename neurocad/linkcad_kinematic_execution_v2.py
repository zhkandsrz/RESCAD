"""Shared primitive frames, pose proposals, and frozen-model inference.

The current pair executor is linkcad_kinematic_execution_v3. Obsolete V2
execution/oracle routines are not part of this core release; the shared
geometry and inference functions retain their original implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .benchmark_v2_training_provenance import capture_file_artifact
from .cadquery_backend import (
    load_step_shape,
    transform_shape,
)
from .domain_types import Transform
from .linkcad_factorized_model_v1 import tensorize_linkcad_query
from .linkcad_primitive_cache_v2 import (
    LinkCADPrimitiveGraphCacheV2,
    attach_primitive_graphs_v2,
)
from .linkcad_primitive_factorized_model_v2 import PrimitiveFactorizedLinkCADV2
from .linkcad_primitive_graph_v2 import extract_primitive_graph_v2


PREDICTION_SCHEMA_VERSION = "linkcad_public_primitive_predictions.v2"


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")


def _sha(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def materialize_public_primitive_predictions_v2(
    *, public: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
    checkpoint_path: str | Path, beam_size: int = 25,
    interface_alternatives_per_edge: int = 8, seed: int = 1701,
    model_kind: str = "primitive_v2",
) -> dict[str, Any]:
  if public.get("contains_private_targets") is not False:
    raise ValueError("LinkCAD V2 public prediction scope differs")
  checkpoint = Path(checkpoint_path).resolve()
  if model_kind == "primitive_v2":
    model_class = PrimitiveFactorizedLinkCADV2
  elif model_kind == "port_v6":
    from .linkcad_port_conditioned_model_v6 import PortConditionedLinkCADV6
    model_class = PortConditionedLinkCADV6
  elif model_kind == "port_v9":
    from .linkcad_port_constraint_model_v9 import PortConstraintLinkCADV9
    model_class = PortConstraintLinkCADV9
  else:
    raise ValueError("LinkCAD public prediction model kind differs")
  model = model_class(
      hidden_dim=64, use_language=True, use_global=False,
      orbit_mode="attention", max_orbits_per_candidate=16, seed=seed,
  )
  model.load_state_dict(
      torch.load(checkpoint, map_location="cpu", weights_only=True)
  )
  model.eval()
  rows = []
  with torch.no_grad():
    for public_query in public["queries"]:
      query = attach_primitive_graphs_v2(
          tensorize_linkcad_query(public_query, None), public_query, cache,
      )
      predictions = model.decode_beam(
          query, beam_size=beam_size,
          interface_alternatives_per_edge=interface_alternatives_per_edge,
      )
      role_ids = tuple(str(row["role_id"]) for row in public_query["roles"])
      edge_ids = tuple(str(row["edge_id"]) for row in public_query["functional_edges"])
      hypotheses = []
      for rank, prediction in enumerate(predictions):
        hypothesis = {
            "rank": rank,
            "score": float(prediction.score),
            "candidate_by_role": dict(zip(
                role_ids, prediction.candidate_ids, strict=True,
            )),
            "edge_programs": [
                {
                    "edge_id": edge_id,
                    "support_family": support,
                    "mobility": mobility,
                    "interface_primitive_a": int(primitives[0]),
                    "interface_primitive_b": int(primitives[1]),
                    "primitive_alternatives": [
                        {
                            "rank": alternative_rank,
                            "interface_primitive_a": int(alternative[0]),
                            "interface_primitive_b": int(alternative[1]),
                            "model_score": float(alternative[2]),
                        }
                        for alternative_rank, alternative in enumerate(alternatives)
                    ],
                }
                for edge_id, support, mobility, primitives, alternatives in zip(
                    edge_ids, prediction.support, prediction.mobility,
                    prediction.interface_orbits,
                    prediction.interface_alternatives, strict=True,
                )
            ],
        }
        hypothesis["hypothesis_payload_sha256"] = _sha(hypothesis)
        hypotheses.append(hypothesis)
      row = {
          "query_id": public_query["query_id"],
          "terminal_status": "prediction_ready" if hypotheses else "no_candidate",
          "hypotheses": hypotheses,
      }
      row["row_payload_sha256"] = _sha(row)
      rows.append(row)
  payload = {
      "schema_version": PREDICTION_SCHEMA_VERSION,
      "scope": "public_part_support_mobility_primitive_predictions",
      "private_targets_opened": False,
      "beam_size": beam_size,
      "interface_alternatives_per_edge": interface_alternatives_per_edge,
      "checkpoint_sha256": _file_sha(checkpoint),
      "model_kind": model_kind,
      "query_count": len(rows),
      "rows": rows,
  }
  payload["prediction_payload_sha256"] = _sha(payload)
  return payload


def _unit(value: Any) -> np.ndarray:
  vector = np.asarray(value, dtype=float).reshape(3)
  norm = float(np.linalg.norm(vector))
  if not math.isfinite(norm) or norm <= 1e-12:
    raise ValueError("primitive frame direction is degenerate")
  return vector / norm


def _point(value: Any) -> np.ndarray:
  return np.asarray((value.X(), value.Y(), value.Z()), dtype=float)


def _direction(value: Any) -> np.ndarray:
  return _unit((value.X(), value.Y(), value.Z()))


def _orthogonal(primary: np.ndarray, preferred: np.ndarray | None = None) -> np.ndarray:
  if preferred is not None:
    candidate = np.asarray(preferred, dtype=float) - primary * float(
        np.dot(preferred, primary)
    )
    if float(np.linalg.norm(candidate)) > 1e-9:
      return _unit(candidate)
  basis = min(np.eye(3), key=lambda row: abs(float(np.dot(row, primary))))
  return _unit(basis - primary * float(np.dot(basis, primary)))


@dataclass(frozen=True, slots=True)
class PrimitiveFrameV2:
  primitive_kind: str
  geometry_type: str
  raw_index: int
  origin_local_mm: tuple[float, ...]
  rotation_local: tuple[float, ...]

  @property
  def origin(self) -> np.ndarray:
    return np.asarray(self.origin_local_mm, dtype=float)

  @property
  def rotation(self) -> np.ndarray:
    return np.asarray(self.rotation_local, dtype=float).reshape(3, 3)

  @property
  def primary(self) -> np.ndarray:
    return self.rotation[:, 2]


@dataclass(frozen=True, slots=True)
class ResolvedPrimitiveV2:
  capture: Any
  shape: Any
  primitive_ordinal: int
  primitive_kind: str
  member_raw_indices: tuple[int, ...]
  frames: tuple[PrimitiveFrameV2, ...]


def _face_frame(face: Any, raw_index: int) -> PrimitiveFrameV2:
  from OCP.BRepAdaptor import BRepAdaptor_Surface

  geometry = str(face.geomType()).upper()
  adaptor = BRepAdaptor_Surface(face.wrapped)
  center = np.asarray(face.Center().toTuple(), dtype=float)
  if geometry == "PLANE":
    plane = adaptor.Plane()
    origin = _point(plane.Location())
    primary = _direction(plane.Axis().Direction())
    preferred = None
    for edge in sorted(face.Edges(), key=lambda row: -float(row.Length())):
      vertices = list(edge.Vertices())
      if len(vertices) >= 2:
        preferred = np.asarray(vertices[-1].Center().toTuple()) - np.asarray(
            vertices[0].Center().toTuple()
        )
        if np.linalg.norm(preferred) > 1e-9:
          break
  elif geometry == "CYLINDER":
    cylinder = adaptor.Cylinder()
    axis_origin = _point(cylinder.Location())
    primary = _direction(cylinder.Axis().Direction())
    origin = axis_origin + primary * float(np.dot(center - axis_origin, primary))
    preferred = center - origin
  else:
    raise ValueError(f"unsupported face primitive geometry {geometry}")
  x_axis = _orthogonal(primary, preferred)
  y_axis = _unit(np.cross(primary, x_axis))
  rotation = np.column_stack((x_axis, y_axis, primary))
  return PrimitiveFrameV2(
      primitive_kind="face", geometry_type=geometry, raw_index=raw_index,
      origin_local_mm=tuple(float(value) for value in origin),
      rotation_local=tuple(float(value) for value in rotation.reshape(-1)),
  )


def _edge_frame(edge: Any, raw_index: int) -> PrimitiveFrameV2:
  from OCP.BRepAdaptor import BRepAdaptor_Curve

  geometry = str(edge.geomType()).upper()
  adaptor = BRepAdaptor_Curve(edge.wrapped)
  if geometry == "CIRCLE":
    circle = adaptor.Circle()
    origin = _point(circle.Location())
    primary = _direction(circle.Axis().Direction())
    preferred = np.asarray(edge.startPoint().toTuple(), dtype=float) - origin
  elif geometry == "LINE":
    line = adaptor.Line()
    origin = np.asarray(edge.Center().toTuple(), dtype=float)
    primary = _direction(line.Direction())
    preferred = None
  else:
    raise ValueError(f"unsupported edge primitive geometry {geometry}")
  x_axis = _orthogonal(primary, preferred)
  y_axis = _unit(np.cross(primary, x_axis))
  rotation = np.column_stack((x_axis, y_axis, primary))
  return PrimitiveFrameV2(
      primitive_kind="edge", geometry_type=geometry, raw_index=raw_index,
      origin_local_mm=tuple(float(value) for value in origin),
      rotation_local=tuple(float(value) for value in rotation.reshape(-1)),
  )


def resolve_primitive_v2(
    *, candidate: Mapping[str, Any], primitive_ordinal: int,
    dataset_root: str | Path,
) -> ResolvedPrimitiveV2:
  if primitive_ordinal < 0:
    raise ValueError("LinkCAD V2 primitive is absent")
  path = (Path(dataset_root).resolve() / candidate["step_path"]).resolve()
  capture = capture_file_artifact(path, label="LinkCAD V2 exact STEP")
  if capture.sha256 != candidate["step_sha256"]:
    raise ValueError("LinkCAD V2 exact STEP bytes differ")
  shape = load_step_shape(path)
  graph = extract_primitive_graph_v2(shape)
  if not 0 <= primitive_ordinal < len(graph.primitive_members):
    raise ValueError("LinkCAD V2 primitive ordinal is outside replayed graph")
  kind = graph.primitive_kinds[primitive_ordinal]
  members = graph.primitive_members[primitive_ordinal]
  raw_shapes = list(shape.Faces()) if kind == "face" else list(shape.Edges())
  frames = []
  for raw_index in members:
    try:
      frame = (
          _face_frame(raw_shapes[raw_index], raw_index)
          if kind == "face" else _edge_frame(raw_shapes[raw_index], raw_index)
      )
      frames.append(frame)
    except ValueError:
      continue
  if not frames:
    raise ValueError("LinkCAD V2 primitive has no executable representative")
  return ResolvedPrimitiveV2(
      capture=capture, shape=shape, primitive_ordinal=primitive_ordinal,
      primitive_kind=kind, member_raw_indices=members, frames=tuple(frames),
  )


def _primitive_role(frame: PrimitiveFrameV2) -> str:
  if frame.primitive_kind == "face" and frame.geometry_type == "PLANE":
    return "plane"
  if (
      frame.geometry_type in {"CIRCLE", "CYLINDER", "LINE"}
  ):
    return "axis"
  return "other"


def _support_compatible(
    support_family: str, left: PrimitiveFrameV2, right: PrimitiveFrameV2,
) -> bool:
  roles = (_primitive_role(left), _primitive_role(right))
  if support_family == "axial_support":
    return roles == ("axis", "axis")
  if support_family == "planar_support":
    return roles == ("plane", "plane")
  if support_family == "axis_plane_support":
    return sorted(roles) == ["axis", "plane"]
  if support_family == "linear_support":
    return roles == ("axis", "axis") and (
        left.geometry_type == "LINE" and right.geometry_type == "LINE"
    )
  return False


def _pose_candidates(
    support_family: str, left: PrimitiveFrameV2, right: PrimitiveFrameV2,
) -> list[Transform]:
  target_frame = left.rotation
  if support_family == "planar_support":
    flips = (
        np.diag((1.0, -1.0, -1.0)),
        np.diag((-1.0, 1.0, -1.0)),
    )
  else:
    flips = (np.eye(3), np.diag((1.0, -1.0, -1.0)))
  poses = []
  for flip in flips:
    for yaw in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
      cosine, sine = math.cos(yaw), math.sin(yaw)
      yaw_matrix = np.asarray((
          (cosine, -sine, 0.0),
          (sine, cosine, 0.0),
          (0.0, 0.0, 1.0),
      ))
      rotation = target_frame @ yaw_matrix @ flip @ right.rotation.T
      translation = left.origin - rotation @ right.origin
      poses.append(Transform(rotation=rotation, translation=translation))
  return poses


def _bbox_overlap_upper(left: Any, right: Any) -> float:
  a, b = left.BoundingBox(), right.BoundingBox()
  overlap = np.asarray((
      min(a.xmax, b.xmax) - max(a.xmin, b.xmin),
      min(a.ymax, b.ymax) - max(a.ymin, b.ymin),
      min(a.zmax, b.zmax) - max(a.zmin, b.zmin),
  ))
  return 0.0 if np.any(overlap <= 0.0) else float(np.prod(overlap))


def _bbox_projection_interval(shape: Any, axis: np.ndarray) -> tuple[float, float]:
  box = shape.BoundingBox()
  corners = np.asarray([
      (x, y, z)
      for x in (box.xmin, box.xmax)
      for y in (box.ymin, box.ymax)
      for z in (box.zmin, box.zmax)
  ], dtype=float)
  values = corners @ _unit(axis)
  return float(values.min()), float(values.max())


def _axial_clearance_poses(
    *, left_shape: Any, right_shape: Any,
    base_pose: Transform, target_axis: np.ndarray,
) -> tuple[tuple[Transform, float], ...]:
  """Add finite endpoint-seating choices along an already aligned axis."""

  axis = _unit(target_axis)
  moved = transform_shape(right_shape, base_pose)
  left_low, left_high = _bbox_projection_interval(left_shape, axis)
  right_low, right_high = _bbox_projection_interval(moved, axis)
  offsets = (
      0.0,
      left_low - right_high - 1e-6,
      left_high - right_low + 1e-6,
  )
  unique = []
  for offset in offsets:
    if any(abs(offset - observed) <= 1e-9 for observed in unique):
      continue
    unique.append(offset)
  return tuple((
      Transform(
          rotation=base_pose.rotation,
          translation=base_pose.translation + axis * offset,
      ),
      float(offset),
  ) for offset in unique)


__all__ = [
    "PREDICTION_SCHEMA_VERSION", "PrimitiveFrameV2", "ResolvedPrimitiveV2",
    "materialize_public_primitive_predictions_v2", "resolve_primitive_v2",
]
