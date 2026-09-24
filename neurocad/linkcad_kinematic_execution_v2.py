"""Target-free kinematic execution for LinkCAD face-and-edge primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .benchmark_v2_training_provenance import capture_file_artifact
from .cadquery_backend import (
    boolean_intersection,
    load_step_shape,
    shape_volume,
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
from .linkcad_symbolic_baseline_v1 import EXTERNAL_PREDICTION_SCHEMA_VERSION
from .linkcad_joinable_style_baseline_v1 import (
    PREDICTION_SCHEMA_VERSION as JOINABLE_PREDICTION_SCHEMA_VERSION,
)


PREDICTION_SCHEMA_VERSION = "linkcad_public_primitive_predictions.v2"
EXECUTION_SCHEMA_VERSION = "linkcad_kinematic_pair_execution.v2"


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


def execute_kinematic_pair_v2(
    *, query_id: str, edge_id: str, hypothesis_rank: int,
    candidate_a: Mapping[str, Any], candidate_b: Mapping[str, Any],
    primitive_a: int, primitive_b: int,
    support_family: str, mobility: str, dataset_root: str | Path,
    maximum_pose_candidates: int = 32,
    enable_axial_seating: bool = False,
    maximum_accepted_poses: int = 1,
) -> dict[str, Any]:
  if maximum_accepted_poses < 1:
    raise ValueError("LinkCAD accepted-pose budget must be positive")
  started = time.monotonic()
  base = {
      "schema_version": EXECUTION_SCHEMA_VERSION,
      "scope": "pairwise_kinematic_constraint_and_collision_feasibility",
      "private_targets_opened": False,
      "query_id": query_id, "edge_id": edge_id,
      "hypothesis_rank": hypothesis_rank,
      "candidate_id_a": candidate_a["candidate_id"],
      "candidate_id_b": candidate_b["candidate_id"],
      "primitive_a": primitive_a, "primitive_b": primitive_b,
      "support_family": support_family, "mobility": mobility,
      "axial_seating_enabled": bool(enable_axial_seating),
      "maximum_accepted_poses": int(maximum_accepted_poses),
  }
  observations = []
  try:
    left = resolve_primitive_v2(
        candidate=candidate_a, primitive_ordinal=primitive_a,
        dataset_root=dataset_root,
    )
    right = resolve_primitive_v2(
        candidate=candidate_b, primitive_ordinal=primitive_b,
        dataset_root=dataset_root,
    )
    pose_count = 0
    accepted_pose = None
    accepted_poses = []
    for left_frame in left.frames:
      for right_frame in right.frames:
        if not _support_compatible(support_family, left_frame, right_frame):
          continue
        pose_rows = []
        for base_pose in _pose_candidates(
            support_family, left_frame, right_frame,
        ):
          pose_rows.extend(
              _axial_clearance_poses(
                  left_shape=left.shape,
                  right_shape=right.shape,
                  base_pose=base_pose,
                  target_axis=left_frame.primary,
              )
              if support_family == "axial_support" and enable_axial_seating
              else ((base_pose, 0.0),)
          )
        for pose, axial_offset in pose_rows:
          if pose_count >= maximum_pose_candidates:
            break
          moved = transform_shape(right.shape, pose)
          aabb_upper = _bbox_overlap_upper(left.shape, moved)
          if aabb_upper <= 1e-7:
            common_volume = 0.0
            volume_status = "certified_by_aabb_upper_bound"
          else:
            common_volume = max(0.0, float(shape_volume(boolean_intersection(
                left.shape, moved,
            ))))
            volume_status = "exact_occ_boolean"
          observation = {
              "pose_rank": pose_count,
              "raw_primitive_a": left_frame.raw_index,
              "raw_primitive_b": right_frame.raw_index,
              "child_world_row_major": [
                  float(value)
                  for value in np.block([
                      [pose.rotation, pose.translation[:, None]],
                      [np.asarray([[0.0, 0.0, 0.0, 1.0]])],
                  ]).reshape(-1)
              ],
              "aabb_intersection_upper_mm3": aabb_upper,
              "whole_solid_common_volume_mm3": common_volume,
              "common_volume_status": volume_status,
              "axial_offset_mm": axial_offset,
          }
          observations.append(observation)
          pose_count += 1
          if common_volume <= 1e-7:
            accepted_poses.append(observation)
            if accepted_pose is None:
              accepted_pose = observation
            if len(accepted_poses) >= maximum_accepted_poses:
              break
        if (
            len(accepted_poses) >= maximum_accepted_poses
            or pose_count >= maximum_pose_candidates
        ):
          break
      if (
          len(accepted_poses) >= maximum_accepted_poses
          or pose_count >= maximum_pose_candidates
      ):
        break
    compatible_count = sum(
        _support_compatible(support_family, a, b)
        for a in left.frames for b in right.frames
    )
    if compatible_count == 0:
      status = "unsupported_support_primitive_pair"
    elif accepted_pose is not None:
      status = "accepted"
    else:
      status = "unknown_bounded_no_collision_free_pose"
    manifold_fallback = None
    if (
        accepted_pose is None
        and support_family == "planar_support"
        and left.primitive_kind == right.primitive_kind == "face"
        and mobility in {"fixed", "revolute"}
    ):
      from .linkcad_exact_execution_v1 import execute_pair_edge_manifold_v1

      manifold_fallback = execute_pair_edge_manifold_v1(
          query_id=query_id,
          hypothesis_rank=hypothesis_rank,
          edge_id=edge_id,
          candidate_a=candidate_a,
          candidate_b=candidate_b,
          orbit_a=primitive_a,
          orbit_b=primitive_b,
          mobility=mobility,
          dataset_root=dataset_root,
      )
      if manifold_fallback.get(
          "conditional_physical_feasibility_accepted"
      ) is True:
        status = "accepted_manifold_fallback"
        accepted_pose = {
            "pose_rank": "constraint_manifold_v4",
            "child_world_row_major": manifold_fallback[
                "selected_pose_row_major"
            ],
            "whole_solid_common_volume_mm3": manifold_fallback[
                "certificate"
            ]["replay_core"]["observation"][
                "whole_solid_common_volume_mm3"
            ],
        }
        accepted_poses = [accepted_pose]
    result = {
        **base,
        "status": status,
        "conditional_kinematic_feasibility_accepted": accepted_pose is not None,
        "compatible_representative_pair_count": compatible_count,
        "evaluated_pose_count": len(observations),
        "maximum_pose_candidates": maximum_pose_candidates,
        "selected_observation": accepted_pose,
        "accepted_pose_count": len(accepted_poses),
        "accepted_observations": accepted_poses,
        "observations": observations,
        "manifold_fallback": manifold_fallback,
        "elapsed_seconds": time.monotonic() - started,
    }
  except Exception as error:
    result = {
        **base,
        "status": "pre_execution_error",
        "conditional_kinematic_feasibility_accepted": False,
        "compatible_representative_pair_count": 0,
        "evaluated_pose_count": len(observations),
        "maximum_pose_candidates": maximum_pose_candidates,
        "selected_observation": None,
        "accepted_pose_count": 0,
        "accepted_observations": [],
        "observations": observations,
        "failure_reason": f"{type(error).__name__}:{error}",
        "elapsed_seconds": time.monotonic() - started,
    }
  result["result_payload_sha256"] = _sha(result)
  return result


def execute_public_prediction_subset_v2(
    *, public: Mapping[str, Any], predictions: Mapping[str, Any],
    dataset_root: str | Path, query_limit: int,
    hypotheses_per_query: int = 1, alternatives_per_edge: int = 3,
    maximum_attempts_per_query: int = 25,
    enable_axial_seating: bool = False,
    maximum_accepted_poses_per_edge: int = 1,
) -> dict[str, Any]:
  if (
      predictions.get("schema_version") not in {
          PREDICTION_SCHEMA_VERSION, "linkcad_symbolic_predictions.v1",
          EXTERNAL_PREDICTION_SCHEMA_VERSION,
          JOINABLE_PREDICTION_SCHEMA_VERSION,
      }
      or predictions.get("private_targets_opened") is not False
      or min(query_limit, hypotheses_per_query, alternatives_per_edge,
             maximum_attempts_per_query,
             maximum_accepted_poses_per_edge) < 1
  ):
    raise ValueError("LinkCAD V2 exact subset scope differs")
  query_by_id = {row["query_id"]: row for row in public["queries"]}
  results = []
  for prediction_row in sorted(
      predictions["rows"], key=lambda row: row["query_id"]
  )[:query_limit]:
    query = query_by_id[prediction_row["query_id"]]
    edge_by_id = {row["edge_id"]: row for row in query["functional_edges"]}
    candidates_by_role = {
        role_id: {row["candidate_id"]: row for row in rows}
        for role_id, rows in query["candidate_sets"].items()
    }
    attempts = 0
    for hypothesis in prediction_row["hypotheses"][:hypotheses_per_query]:
      for edge_program in hypothesis["edge_programs"]:
        if attempts >= maximum_attempts_per_query:
          break
        edge = edge_by_id[edge_program["edge_id"]]
        candidate_a = candidates_by_role[edge["role_a"]][
            hypothesis["candidate_by_role"][edge["role_a"]]
        ]
        candidate_b = candidates_by_role[edge["role_b"]][
            hypothesis["candidate_by_role"][edge["role_b"]]
        ]
        alternatives = edge_program.get("primitive_alternatives") or ({
            "rank": 0,
            "interface_primitive_a": edge_program["interface_primitive_a"],
            "interface_primitive_b": edge_program["interface_primitive_b"],
            "model_score": None,
        },)
        for alternative in alternatives[:alternatives_per_edge]:
          if attempts >= maximum_attempts_per_query:
            break
          row = execute_kinematic_pair_v2(
              query_id=prediction_row["query_id"],
              edge_id=edge_program["edge_id"],
              hypothesis_rank=int(hypothesis["rank"]),
              candidate_a=candidate_a, candidate_b=candidate_b,
              primitive_a=int(alternative["interface_primitive_a"]),
              primitive_b=int(alternative["interface_primitive_b"]),
              support_family=str(edge_program["support_family"]),
              mobility=str(edge_program["mobility"]),
              dataset_root=dataset_root,
              enable_axial_seating=enable_axial_seating,
              maximum_accepted_poses=maximum_accepted_poses_per_edge,
          )
          row["primitive_alternative_rank"] = int(alternative["rank"])
          row["primitive_model_score"] = alternative["model_score"]
          row.pop("result_payload_sha256", None)
          row["result_payload_sha256"] = _sha(row)
          results.append(row)
          attempts += 1
          if row["conditional_kinematic_feasibility_accepted"] is True:
            break
  edge_keys = {
      (row["query_id"], row["hypothesis_rank"], row["edge_id"])
      for row in results
  }
  accepted_keys = {
      (row["query_id"], row["hypothesis_rank"], row["edge_id"])
      for row in results
      if row["conditional_kinematic_feasibility_accepted"] is True
  }
  status_counts: dict[str, int] = {}
  for row in results:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
  payload = {
      "schema_version": "linkcad_kinematic_execution_subset.v2",
      "scope": "development_public_prediction_target_free_kinematic_subset",
      "private_targets_opened": False,
      "prediction_payload_sha256": predictions["prediction_payload_sha256"],
      "query_limit": query_limit,
      "hypotheses_per_query": hypotheses_per_query,
      "alternatives_per_edge": alternatives_per_edge,
      "maximum_attempts_per_query": maximum_attempts_per_query,
      "axial_seating_enabled": bool(enable_axial_seating),
      "maximum_accepted_poses_per_edge": int(
          maximum_accepted_poses_per_edge
      ),
      "edge_target_count": len(edge_keys),
      "execution_attempt_count": len(results),
      "accepted_edge_count": len(accepted_keys),
      "status_counts": dict(sorted(status_counts.items())),
      "rows": results,
  }
  payload["subset_payload_sha256"] = _sha(payload)
  return payload


def execute_development_primitive_oracle_subset_v2(
    *, public: Mapping[str, Any], private_targets: Mapping[str, Any],
    primitive_supervision: Mapping[str, Any], dataset_root: str | Path,
    query_limit: int,
) -> dict[str, Any]:
  if (
      public.get("contains_private_targets") is not False
      or private_targets.get("schema_version")
      != "linkcad_candidate_set_private_targets.v1"
      or primitive_supervision.get("schema_version")
      != "linkcad_direct_primitive_orbit_supervision.v2"
      or query_limit < 1
  ):
    raise ValueError("LinkCAD V2 development oracle scope differs")
  public_by_id = {row["query_id"]: row for row in public["queries"]}
  private_by_id = {row["query_id"]: row for row in private_targets["targets"]}
  primitive_by_key = {
      (row["query_id"], row["edge_id"], row["side"]): row
      for row in primitive_supervision["rows"]
      if row.get("status") == "mapped_type_consistent"
  }
  selected = []
  for query_id in sorted(public_by_id):
    query = public_by_id[query_id]
    if all(
        (query_id, edge["edge_id"], side) in primitive_by_key
        for edge in query["functional_edges"] for side in ("a", "b")
    ):
      selected.append(query_id)
      if len(selected) == query_limit:
        break
  results = []
  for query_id in selected:
    query = public_by_id[query_id]
    target = private_by_id[query_id]
    target_edges = {
        row["edge_id"]: row for row in target["functional_edge_targets"]
    }
    candidate_by_role = {
        role_id: {row["candidate_id"]: row for row in rows}
        for role_id, rows in query["candidate_sets"].items()
    }
    for edge in query["functional_edges"]:
      edge_id = edge["edge_id"]
      target_edge = target_edges[edge_id]
      candidate_a = candidate_by_role[edge["role_a"]][
          target["target_candidate_by_role"][edge["role_a"]]
      ]
      candidate_b = candidate_by_role[edge["role_b"]][
          target["target_candidate_by_role"][edge["role_b"]]
      ]
      row = execute_kinematic_pair_v2(
          query_id=query_id, edge_id=edge_id, hypothesis_rank=0,
          candidate_a=candidate_a, candidate_b=candidate_b,
          primitive_a=int(primitive_by_key[(query_id, edge_id, "a")][
              "primitive_orbit_ordinal"
          ]),
          primitive_b=int(primitive_by_key[(query_id, edge_id, "b")][
              "primitive_orbit_ordinal"
          ]),
          support_family=str(target_edge["support_family"]),
          mobility=str(target_edge["target_mobility"]),
          dataset_root=dataset_root,
      )
      row["private_targets_opened"] = True
      row["evidence_role"] = "development_oracle_executor_ceiling"
      row.pop("result_payload_sha256", None)
      row["result_payload_sha256"] = _sha(row)
      results.append(row)
  status_counts: dict[str, int] = {}
  for row in results:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
  payload = {
      "schema_version": "linkcad_kinematic_oracle_subset.v2",
      "scope": "development_private_opened_executor_ceiling_not_blind_evidence",
      "private_targets_opened": True,
      "selection_policy": (
          "query_id_ascending_among_queries_with_all_primitive_endpoints_"
          "mapped_before_execution"
      ),
      "selected_query_ids": selected,
      "selected_query_count": len(selected),
      "edge_execution_count": len(results),
      "accepted_count": sum(
          row["conditional_kinematic_feasibility_accepted"] is True
          for row in results
      ),
      "status_counts": dict(sorted(status_counts.items())),
      "rows": results,
  }
  payload["subset_payload_sha256"] = _sha(payload)
  return payload


__all__ = [
    "EXECUTION_SCHEMA_VERSION", "PREDICTION_SCHEMA_VERSION",
    "execute_kinematic_pair_v2", "materialize_public_primitive_predictions_v2",
    "execute_public_prediction_subset_v2",
    "execute_development_primitive_oracle_subset_v2", "resolve_primitive_v2",
]
