"""Receipt-bound full-graph face frames and executable V3 SE(3) adapter.

This module closes the gap between the label-free full-STEP cache and Joint
V3.  Face frames are replayed from the exact captured STEP bytes and formal
instance transforms; callers can select a receipt-bound visible part but can
never provide an origin, rotation, unit conversion, or face signature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .benchmark_v2_model_view_v2 import (
    AuthenticatedBenchmarkV2GraphCapability,
    AuthenticatedBenchmarkV2DevelopmentIndex,
    GraphSizeBudget,
    _DevelopmentIndexState,
    _GraphCapabilityState,
    _authority_case,
    _captured_bound_file,
    _captured_brep_topology,
    _get_private_state,
    _intrinsic_graph_from_face_subset,
    _load_captured_step_shape,
    _strict_json_bytes,
)
from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    reverify_captured_file_artifact,
)
from .benchmark_v3_unlabeled_step_view import (
    BenchmarkV3UnlabeledStepView,
    assert_no_forbidden_model_visible_fields,
    canonical_sha256,
)
from .brep_tensor_cache_v4_unlabeled import _graph_payload_to_unlabeled
from .domain_types import Transform
from .execution_input_authority_v1 import AuthenticatedExecutionInputRowV1
from .view_execution_pose_authority_v1 import AuthenticatedViewExecutionPoseRowV1
from .joint_interface_program_learner_v1 import JointInterfaceInputsV1
from .joint_interface_program_learner_v2 import (
    FORMAL_CATALOG_SIZE,
    PAIR_FRAME_POLICY_SHA256,
    FiniteProgramCatalogRosterV2,
    FiniteProgramDescriptorV2,
    GraphProgramInputsV2,
)
from .joint_interface_program_learner_v3 import (
    ExternallyFrozenCatalogSpecV3,
    ExecutablePredictedProgramV3,
    GraphQueryBindingV3,
    full_view_frame_receipt_payload_v3,
    graph_query_core_commitment_v3,
    validate_executable_query_replay_v3,
)


AUTHORITY_SCHEMA_VERSION = "full_graph_face_frame_authority.v1"
ARTIFACT_SCHEMA_VERSION = "full_graph_face_frame_authority_artifact.v1"
FRAME_SCHEMA_VERSION = "authenticated_full_graph_face_frame.v1"
EXECUTION_SCHEMA_VERSION = "executable_predicted_program_se3.v1"
PROPOSAL_RUN_SCHEMA_VERSION = "joint_v3_proposal_run.v1"
PROPOSAL_RUN_ARTIFACT_SCHEMA_VERSION = "joint_v3_proposal_run_artifact.v1"
CAPABILITY_SCHEMA_VERSION = "authenticated_predicted_execution_capability.v1"


class FiniteProgramBaseSE3SchemaMissing(ValueError):
  """A categorical program plus residual cannot authorize arbitrary SE(3)."""
FRAME_POLICY_SCHEMA_VERSION = "occ_full_graph_face_frame_policy.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FACTORY_TOKEN = object()
_AUTHORITY_FACTORY_TOKEN = object()
_FRAME_FACTORY_TOKEN = object()
_EXECUTION_FACTORY_TOKEN = object()
_PART_FACTORY_TOKEN = object()
_CAPABILITY_FACTORY_TOKEN = object()
_FULL_GRAPH_BUDGET = GraphSizeBudget(
    max_graphs=2,
    max_faces_per_graph=2048,
    max_edges_per_graph=4096,
    max_examples=1,
    max_program_candidates_per_example=19,
)
_UNIT_POLICY = {
    "schema_version": "receipt_bound_step_world_unit.v1",
    "step_reader_target_unit": "mm",
    "world_transform_translation_unit": "mm",
    "step_local_to_mm_scale": 1.0,
}
_FRAME_POLICY = {
    "schema_version": FRAME_POLICY_SCHEMA_VERSION,
    "origin": "occ_face_center_in_step_mm_then_receipt_world_transform",
    "z_axis": "analytic_axis_for_cylinder_cone_torus_else_oriented_face_normal",
    "x_axis": "analytic_surface_x_axis_else_first_max_projected_boundary_vertex",
    "handedness": "y_equals_z_cross_x_then_x_equals_y_cross_z",
    "graph_identity": "inverse_of_full_intrinsic_graph_raw_occ_to_node_remap",
    "unit_policy_sha256": canonical_sha256(_UNIT_POLICY),
}
FRAME_POLICY_SHA256 = canonical_sha256(_FRAME_POLICY)


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode("utf-8")


def _deep_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
  if isinstance(value, (list, tuple)):
    return tuple(_deep_freeze(item) for item in value)
  if value is None or type(value) in (str, int, float, bool):
    return value
  raise TypeError(f"authority payload contains unsupported type {type(value).__name__}")


def _deep_thaw(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(key): _deep_thaw(item) for key, item in value.items()}
  if isinstance(value, tuple):
    return [_deep_thaw(item) for item in value]
  return value


class _ImmutableFactoryObject:
  __slots__ = ()

  def __setattr__(self, name: str, value: Any) -> None:
    if hasattr(self, name):
      raise AttributeError(f"{type(self).__name__} is immutable")
    object.__setattr__(self, name, value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError(f"{type(self).__name__} is not serializable")

  def __copy__(self) -> Any:
    raise TypeError(f"{type(self).__name__} cannot be copied")

  def __deepcopy__(self, _memo: Any) -> Any:
    raise TypeError(f"{type(self).__name__} cannot be copied")


class _CapturedPathGuardV1(_ImmutableFactoryObject):
  __slots__ = ("path", "resolved_path", "device", "inode", "link_count", "byte_count", "mtime_ns", "ctime_ns", "sha256")

  def __init__(self, path: Path) -> None:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    object.__setattr__(self, "path", path)
    object.__setattr__(self, "resolved_path", resolved)
    object.__setattr__(self, "device", int(stat.st_dev))
    object.__setattr__(self, "inode", int(stat.st_ino))
    object.__setattr__(self, "link_count", int(stat.st_nlink))
    object.__setattr__(self, "byte_count", int(stat.st_size))
    object.__setattr__(self, "mtime_ns", int(stat.st_mtime_ns))
    object.__setattr__(self, "ctime_ns", int(stat.st_ctime_ns))
    object.__setattr__(self, "sha256", _file_sha256(resolved))

  def revalidate(self, *, label: str) -> None:
    try:
      resolved = self.path.resolve(strict=True)
      stat = resolved.stat()
    except OSError as error:
      raise ValueError(f"{label} disappeared after authentication") from error
    identity = (
        resolved, int(stat.st_dev), int(stat.st_ino), int(stat.st_nlink), int(stat.st_size),
        int(stat.st_mtime_ns), int(stat.st_ctime_ns), _file_sha256(resolved),
    )
    expected = (
        self.resolved_path, self.device, self.inode, self.link_count, self.byte_count,
        self.mtime_ns, self.ctime_ns, self.sha256,
    )
    if identity != expected:
      raise ValueError(f"{label} changed after authentication")


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _binding(path: Path, *, name: str) -> dict[str, Any]:
  return {"name": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
  def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
      if key in result:
        raise ValueError(f"{label} contains duplicate JSON keys")
      result[key] = value
    return result

  try:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
  except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be an object")
  return value


def _source_sha256() -> str:
  return _file_sha256(Path(__file__).resolve())


def _proper_rotation(value: Any, *, label: str) -> np.ndarray:
  rotation = np.asarray(value, dtype=float).reshape(3, 3)
  if not np.isfinite(rotation).all():
    raise ValueError(f"{label} is non-finite")
  first = _unit(rotation[:, 0], label=f"{label} x")
  second_seed = rotation[:, 1] - _dot(first, rotation[:, 1]) * first
  second = _unit(second_seed, label=f"{label} y")
  third = _unit(np.cross(first, second), label=f"{label} z")
  if _dot(third, rotation[:, 2]) < 0.0:
    second = -second
    third = -third
  return np.stack((first, second, third), axis=1)


def _dot(left: Any, right: Any) -> float:
  first = np.asarray(left, dtype=float).reshape(-1)
  second = np.asarray(right, dtype=float).reshape(-1)
  if first.shape != second.shape:
    raise ValueError("dot-product dimensions differ")
  return sum(float(a) * float(b) for a, b in zip(first, second, strict=True))


def _matmul(left: Any, right: Any) -> np.ndarray:
  first = np.asarray(left, dtype=float)
  second = np.asarray(right, dtype=float)
  if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[0]:
    raise ValueError("matrix product dimensions differ")
  return np.asarray(
      [
          [sum(float(first[row, inner]) * float(second[inner, column]) for inner in range(first.shape[1]))
           for column in range(second.shape[1])]
          for row in range(first.shape[0])
      ],
      dtype=float,
  )


def _matvec(matrix: Any, vector: Any) -> np.ndarray:
  return _matmul(np.asarray(matrix, dtype=float), np.asarray(vector, dtype=float).reshape(-1, 1)).reshape(-1)


def _unit(value: Any, *, label: str) -> np.ndarray:
  vector = _vec(value)
  length = float(np.linalg.norm(vector))
  if not np.isfinite(vector).all() or not math.isfinite(length) or length <= 1e-12:
    raise ValueError(f"{label} is degenerate")
  return vector / length


def _vec(value: Any) -> np.ndarray:
  if hasattr(value, "toTuple"):
    value = value.toTuple()
  elif all(hasattr(value, name) for name in ("X", "Y", "Z")):
    value = (value.X(), value.Y(), value.Z())
  return np.asarray(value, dtype=float).reshape(3)


def _analytic_axes(face: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
  """Return OCC-parametric x and analytic mating z when available."""

  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomAbs import (  # type: ignore
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Torus,
    )

    adaptor = BRepAdaptor_Surface(face.wrapped)
    surface_type = adaptor.GetType()
    surface = None
    z_axis = None
    if surface_type == GeomAbs_Plane:
      surface = adaptor.Plane()
    elif surface_type == GeomAbs_Cylinder:
      surface = adaptor.Cylinder()
      z_axis = _vec(surface.Axis().Direction())
    elif surface_type == GeomAbs_Cone:
      surface = adaptor.Cone()
      z_axis = _vec(surface.Axis().Direction())
    elif surface_type == GeomAbs_Torus:
      surface = adaptor.Torus()
      z_axis = _vec(surface.Axis().Direction())
    if surface is None:
      return None, None
    return _vec(surface.XAxis().Direction()), z_axis
  except Exception:
    return None, None


def _constraint_surface_type(face: Any) -> str:
  """Resolve the OCC analytic class used by executable constraint programs."""

  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomAbs import (  # type: ignore
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
    )

    surface_type = BRepAdaptor_Surface(face.wrapped).GetType()
    if surface_type == GeomAbs_Plane:
      return "plane"
    if surface_type == GeomAbs_Cylinder:
      return "cylinder"
    if surface_type == GeomAbs_Sphere:
      return "sphere"
  except Exception:
    pass
  return "other"


def _constraint_surface_bucket(model_visible_surface_type: str) -> str:
  normalized = str(model_visible_surface_type).strip().lower()
  return normalized if normalized in {"plane", "cylinder", "sphere"} else "other"


def _local_face_frame(face: Any) -> tuple[np.ndarray, np.ndarray]:
  origin = _vec(face.Center())
  x_seed, analytic_z = _analytic_axes(face)
  if analytic_z is None:
    try:
      z_axis = _unit(face.normalAt(face.Center()), label="OCC face normal")
    except Exception:
      z_axis = _unit(face.normalAt(), label="OCC face normal")
  else:
    z_axis = _unit(analytic_z, label="OCC analytic face axis")
  if x_seed is None:
    candidates: list[tuple[float, np.ndarray]] = []
    for vertex in list(face.Vertices()):
      delta = _vec(vertex.Center()) - origin
      tangent = delta - _dot(delta, z_axis) * z_axis
      norm = float(np.linalg.norm(tangent))
      if norm > 1e-10:
        candidates.append((norm, tangent))
    if not candidates:
      raise ValueError("OCC face has no deterministic tangential seed")
    maximum = max(value[0] for value in candidates)
    x_seed = next(tangent for norm, tangent in candidates if norm >= maximum * (1.0 - 1e-10))
  x_axis = np.asarray(x_seed, dtype=float) - _dot(x_seed, z_axis) * z_axis
  x_axis = _unit(x_axis, label="OCC face x axis")
  y_axis = _unit(np.cross(z_axis, x_axis), label="OCC face y axis")
  x_axis = _unit(np.cross(y_axis, z_axis), label="OCC face reorthogonalized x axis")
  return origin, np.stack((x_axis, y_axis, z_axis), axis=1)


class _FramePartSourceV1(_ImmutableFactoryObject):
  __slots__ = (
      "part", "part_instance_id", "body_id", "step", "world_rotation",
      "world_translation_mm", "world_transform_chain_sha256", "source_instance_key_sha256",
  )

  def __init__(
      self, *, part: str, part_instance_id: str, body_id: str, step: CapturedFileArtifact,
      world_rotation: Sequence[Sequence[float]], world_translation_mm: Sequence[float],
      world_transform_chain_sha256: str, source_instance_key_sha256: str,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _PART_FACTORY_TOKEN:
      raise TypeError("frame part sources are verifier-factory-only")
    if type(step) is not CapturedFileArtifact:
      raise TypeError("frame part STEP capture differs")
    rotation = _proper_rotation(world_rotation, label="frame part world rotation")
    translation = np.asarray(world_translation_mm, dtype=float).reshape(3)
    if not np.isfinite(translation).all():
      raise ValueError("frame part translation is non-finite")
    for value in (part_instance_id, world_transform_chain_sha256, source_instance_key_sha256):
      if _SHA256.fullmatch(value) is None:
        raise ValueError("frame part hash binding differs")
    object.__setattr__(self, "part", str(part))
    object.__setattr__(self, "part_instance_id", part_instance_id)
    object.__setattr__(self, "body_id", str(body_id))
    object.__setattr__(self, "step", step)
    object.__setattr__(self, "world_rotation", tuple(tuple(float(value) for value in row) for row in rotation))
    object.__setattr__(self, "world_translation_mm", tuple(float(value) for value in translation))
    object.__setattr__(self, "world_transform_chain_sha256", world_transform_chain_sha256)
    object.__setattr__(self, "source_instance_key_sha256", source_instance_key_sha256)


class AuthenticatedFrameSourceContextV1(_ImmutableFactoryObject):
  """Factory-only receipt-bound source identities; contains no endpoint face."""

  __slots__ = ("case_id", "row_id", "parts", "source_bindings", "binding_sha256")

  def __init__(
      self,
      *,
      case_id: str,
      row_id: str,
      parts: Sequence[_FramePartSourceV1],
      source_bindings: Mapping[str, Any],
      binding_sha256: str,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _SOURCE_FACTORY_TOKEN:
      raise TypeError("frame source contexts are verifier-factory-only")
    if len(parts) != 2 or any(type(value) is not _FramePartSourceV1 for value in parts):
      raise ValueError("frame source context requires exactly two receipt-bound parts")
    object.__setattr__(self, "case_id", case_id)
    object.__setattr__(self, "row_id", row_id)
    object.__setattr__(self, "parts", tuple(parts))
    object.__setattr__(self, "source_bindings", _deep_freeze(source_bindings))
    object.__setattr__(self, "binding_sha256", binding_sha256)


def load_execution_frame_source_context_v2(
    execution_row: AuthenticatedExecutionInputRowV1,
    *,
    case_id: str,
    row_id: str,
) -> AuthenticatedFrameSourceContextV1:
  """Adapt the executor-only sidecar into the existing frame replay boundary.

  Part identity, STEP bytes and transforms come exclusively from the sealed
  sidecar row.  ``case_id`` and ``row_id`` are presentation/query bindings;
  they cannot influence source-part selection or geometry.
  """

  if type(execution_row) is not AuthenticatedExecutionInputRowV1:
    raise TypeError("execution frame sources require an authenticated sidecar row")
  if not case_id or not row_id:
    raise ValueError("execution frame query identity differs")
  parts: list[_FramePartSourceV1] = []
  part_payloads: list[dict[str, Any]] = []
  for expected_role, part in zip(("a", "b"), execution_row.parts, strict=True):
    if part.query_role != expected_role:
      raise ValueError("execution input part order differs")
    transform = part.world_transform_mm
    rotation = np.asarray(transform["rotation_row_major"], dtype=float).reshape(3, 3)
    translation = np.asarray(transform["translation_mm"], dtype=float).reshape(3)
    captured = part.step
    source = _FramePartSourceV1(
        part=part.source_part,
        part_instance_id=part.source_instance_key_sha256,
        body_id=part.body_uuid,
        step=captured,
        world_rotation=rotation,
        world_translation_mm=translation,
        world_transform_chain_sha256=part.world_transform_chain_sha256,
        source_instance_key_sha256=part.source_instance_key_sha256,
        _factory_token=_PART_FACTORY_TOKEN,
    )
    parts.append(source)
    part_payloads.append({
        "query_role": expected_role,
        "part": source.part,
        "part_instance_id": source.part_instance_id,
        "body_id": source.body_id,
        "step_sha256": source.step.sha256,
        "world_transform_chain_sha256": source.world_transform_chain_sha256,
        "source_instance_key_sha256": source.source_instance_key_sha256,
        "instance_transform_sha256": part.instance_transform_sha256,
    })
  source_bindings = {
      "execution_input_row_receipt_sha256": execution_row.binding_sha256,
      "safe_input_sha256": execution_row.safe_input_sha256,
      "instance_pair_domain_sha256": canonical_sha256(part_payloads),
      "unit_policy_sha256": canonical_sha256(_UNIT_POLICY),
  }
  payload = {
      "case_id": case_id,
      "row_id": row_id,
      "row_ordinal": execution_row.row_ordinal,
      "parts": part_payloads,
      "source_bindings": source_bindings,
  }
  return AuthenticatedFrameSourceContextV1(
      case_id=case_id,
      row_id=row_id,
      parts=parts,
      source_bindings=source_bindings,
      binding_sha256=canonical_sha256(payload),
      _factory_token=_SOURCE_FACTORY_TOKEN,
  )


def load_view_execution_frame_source_context_v1(
    pose_row: AuthenticatedViewExecutionPoseRowV1,
    *,
    case_id: str,
    row_id: str,
) -> AuthenticatedFrameSourceContextV1:
  """Adapt only the safe-view-bound current query pose into frame replay.

  Unlike :func:`load_execution_frame_source_context_v2`, this seam has no
  source-assembly transform available.  The two world transforms are the
  independently scrambled placements authenticated by the V5 safe view.
  """

  if type(pose_row) is not AuthenticatedViewExecutionPoseRowV1:
    raise TypeError("frame execution requires authenticated scrambled current poses")
  if not case_id or not row_id:
    raise ValueError("view execution frame query identity differs")
  parts: list[_FramePartSourceV1] = []
  part_payloads: list[dict[str, Any]] = []
  for expected_role, part in zip(("a", "b"), pose_row.parts, strict=True):
    if part.query_role != expected_role:
      raise ValueError("view execution pose part order differs")
    transform = part.current_world_transform_mm
    rotation = np.asarray(transform["rotation_row_major"], dtype=float).reshape(3, 3)
    translation = np.asarray(transform["translation_mm"], dtype=float).reshape(3)
    source = _FramePartSourceV1(
        part=part.query_part_id,
        part_instance_id=part.query_instance_identity_sha256,
        body_id=part.body_uuid,
        step=part.step,
        world_rotation=rotation,
        world_translation_mm=translation,
        # The legacy frame container calls this a chain, but it is now the
        # commitment of the current scrambled pose, never a source chain.
        world_transform_chain_sha256=part.current_pose_sha256,
        source_instance_key_sha256=part.query_instance_identity_sha256,
        _factory_token=_PART_FACTORY_TOKEN,
    )
    parts.append(source)
    part_payloads.append({
        "query_role": expected_role,
        "query_part_id": source.part,
        "query_instance_identity_sha256": source.source_instance_key_sha256,
        "body_id": source.body_id,
        "step_sha256": source.step.sha256,
        "safe_graph_receipt_sha256": part.safe_graph_receipt_sha256,
        "current_pose_sha256": part.current_pose_sha256,
    })
  source_bindings = {
      "view_execution_pose_row_receipt_sha256": pose_row.binding_sha256,
      "safe_input_sha256": pose_row.safe_input_sha256,
      "current_pose_pair_domain_sha256": canonical_sha256(part_payloads),
      "unit_policy_sha256": canonical_sha256(_UNIT_POLICY),
      "source_pose_bound": False,
  }
  payload = {
      "case_id": case_id,
      "row_id": row_id,
      "row_ordinal": pose_row.row_ordinal,
      "parts": part_payloads,
      "source_bindings": source_bindings,
  }
  return AuthenticatedFrameSourceContextV1(
      case_id=case_id,
      row_id=row_id,
      parts=parts,
      source_bindings=source_bindings,
      binding_sha256=canonical_sha256(payload),
      _factory_token=_SOURCE_FACTORY_TOKEN,
  )


def load_development_frame_source_context_v1(
    index: AuthenticatedBenchmarkV2DevelopmentIndex,
    *,
    split: str,
    case_index: int,
    part_a: str,
    part_b: str,
    row_id: str,
) -> AuthenticatedFrameSourceContextV1:
  """Select two visible parts from an authenticated case without reading gold."""

  if type(index) is not AuthenticatedBenchmarkV2DevelopmentIndex:
    raise TypeError("frame sources require the authenticated development index")
  if split not in {"train", "dev"} or type(case_index) is not int or case_index < 0:
    raise ValueError("frame source case slot differs")
  if not row_id or not part_a or not part_b or part_a == part_b:
    raise ValueError("frame source query identity differs")
  index.revalidate()
  state = _get_private_state(index, _DevelopmentIndexState)
  if case_index >= len(state.bindings[split]):
    raise ValueError("frame source case slot is outside authority")
  binding = state.bindings[split][case_index]
  for captured in binding.captures:
    reverify_captured_file_artifact(captured, label="frame source authority")
  case = _authority_case(binding.face_payload, case_id=binding.case_id, label="face authority")
  inputs = case.get("inputs")
  if not isinstance(inputs, Mapping):
    raise ValueError("frame source authority lacks receipt-bound inputs")
  steps = inputs.get("steps")
  instances = inputs.get("instances")
  if not isinstance(steps, list) or not isinstance(instances, list):
    raise ValueError("frame source STEP/instance domain differs")
  result: list[_FramePartSourceV1] = []
  for part in (part_a, part_b):
    step_rows = [row for row in steps if isinstance(row, Mapping) and row.get("part") == part]
    instance_rows = [
        row for row in instances
        if isinstance(row, Mapping) and row.get("part") == part and row.get("visible") is True
    ]
    if len(step_rows) != 1 or len(instance_rows) != 1:
      raise ValueError("selected part lacks unique visible STEP/instance authority")
    step_row, instance = step_rows[0], instance_rows[0]
    if any(
        step_row.get(key) != instance.get(key)
        for key in ("body_uuid", "geometry_asset", "part")
    ):
      raise ValueError("selected STEP/body/instance identity differs")
    transform = instance.get("world_transform_mm")
    if not isinstance(transform, Mapping):
      raise ValueError("selected instance transform authority differs")
    rotation = _proper_rotation(
        np.asarray(transform.get("rotation_row_major"), dtype=float).reshape(3, 3),
        label="receipt-bound world rotation",
    )
    translation = np.asarray(transform.get("translation_mm"), dtype=float).reshape(3)
    if not np.isfinite(translation).all():
      raise ValueError("receipt-bound world translation is non-finite")
    step_capture = _captured_bound_file(step_row.get("file"), label="frame source STEP")
    instance_key_sha = canonical_sha256(instance.get("source_instance_key"))
    result.append(
        _FramePartSourceV1(
            part=part,
            part_instance_id=instance_key_sha,
            body_id=str(instance.get("body_uuid") or ""),
            step=step_capture,
            world_rotation=tuple(tuple(float(value) for value in row) for row in rotation),
            world_translation_mm=tuple(float(value) for value in translation),
            world_transform_chain_sha256=str(instance.get("world_transform_chain_sha256") or ""),
            source_instance_key_sha256=instance_key_sha,
            _factory_token=_PART_FACTORY_TOKEN,
        )
    )
  source_bindings = {
      "case_authority_binding_sha256": binding.binding_sha256,
      "face_authority_payload_sha256": canonical_sha256(dict(binding.face_payload)),
      "archive_sha256": str((inputs.get("archive") or {}).get("sha256") or ""),
      "assembly_sha256": str((inputs.get("assembly_json") or {}).get("sha256") or ""),
      "content_receipt_sha256": str((inputs.get("content_receipt") or {}).get("sha256") or ""),
      "unit_policy_sha256": canonical_sha256(_UNIT_POLICY),
  }
  if any(_SHA256.fullmatch(value) is None for value in source_bindings.values()):
    raise ValueError("frame source authority hash binding differs")
  payload = {
      "case_id": binding.case_id,
      "row_id": row_id,
      "parts": [
          {
              "part": part.part,
              "part_instance_id": part.part_instance_id,
              "body_id": part.body_id,
              "step_sha256": part.step.sha256,
              "world_transform_chain_sha256": part.world_transform_chain_sha256,
              "source_instance_key_sha256": part.source_instance_key_sha256,
          }
          for part in result
      ],
      "source_bindings": source_bindings,
  }
  return AuthenticatedFrameSourceContextV1(
      case_id=binding.case_id,
      row_id=row_id,
      parts=result,
      source_bindings=source_bindings,
      binding_sha256=canonical_sha256(payload),
      _factory_token=_SOURCE_FACTORY_TOKEN,
  )


def load_development_frame_source_context_for_graph_capability_v2(
    index: AuthenticatedBenchmarkV2DevelopmentIndex,
    graph_capability: AuthenticatedBenchmarkV2GraphCapability,
    *,
    split: str,
    case_index: int,
    source_program_index: int,
    row_id: str,
) -> AuthenticatedFrameSourceContextV1:
  """Recover the ordered query-part pair without accepting caller part names.

  Only the two STEP byte hashes from the authenticated graph receipt are used.
  Endpoint face indices and target program values are never projected through
  this boundary.
  """

  if type(index) is not AuthenticatedBenchmarkV2DevelopmentIndex:
    raise TypeError("graph-bound frame sources require the development index")
  if type(graph_capability) is not AuthenticatedBenchmarkV2GraphCapability:
    raise TypeError("graph-bound frame sources require an authenticated graph capability")
  normalized_split = str(split).strip().lower()
  if normalized_split not in {"train", "dev"}:
    raise ValueError("graph-bound frame source split differs")
  if type(case_index) is not int or type(source_program_index) is not int:
    raise TypeError("graph-bound frame source ordinals must be actual integers")
  index.revalidate()
  graph_capability.revalidate()
  state = _get_private_state(index, _DevelopmentIndexState)
  if not 0 <= case_index < len(state.bindings[normalized_split]):
    raise ValueError("graph-bound frame source case slot is outside authority")
  binding = state.bindings[normalized_split][case_index]
  capability_state = _get_private_state(graph_capability, _GraphCapabilityState)
  receipt = _strict_json_bytes(
      capability_state.receipt_bytes, label="graph-bound frame source receipt"
  )
  semantic = receipt.get("semantic_authority") if isinstance(receipt, Mapping) else None
  graph_rows = receipt.get("graphs") if isinstance(receipt, Mapping) else None
  if (
      not isinstance(semantic, Mapping)
      or semantic.get("program_index") != source_program_index
      or receipt.get("case_authority_binding_sha256") != binding.binding_sha256
      or not isinstance(graph_rows, list)
      or len(graph_rows) != 2
  ):
    raise ValueError("graph capability/program/case frame binding differs")
  step_hashes = tuple(str(row.get("step_sha256") or "") for row in graph_rows)
  transform_chains = tuple(
      str(row.get("instance_transform_chain_sha256") or "") for row in graph_rows
  )
  transform_hashes = tuple(
      str(row.get("instance_transform_sha256") or "") for row in graph_rows
  )
  if any(
      _SHA256.fullmatch(value) is None
      for value in (*step_hashes, *transform_chains, *transform_hashes)
  ):
    raise ValueError("graph-bound frame source STEP hashes differ")

  case = _authority_case(binding.face_payload, case_id=binding.case_id, label="face authority")
  inputs = case.get("inputs")
  steps = inputs.get("steps") if isinstance(inputs, Mapping) else None
  instances = inputs.get("instances") if isinstance(inputs, Mapping) else None
  if not isinstance(steps, list) or not isinstance(instances, list):
    raise ValueError("graph-bound frame source authority lacks STEP/instance inputs")
  ordered_parts: list[str] = []
  for expected_sha256, expected_chain, expected_transform in zip(
      step_hashes, transform_chains, transform_hashes, strict=True
  ):
    candidate_instances = [
        row
        for row in instances
        if isinstance(row, Mapping)
        and row.get("visible") is True
        and row.get("world_transform_chain_sha256") == expected_chain
        and canonical_sha256(row.get("world_transform_mm")) == expected_transform
    ]
    matches = [
        str(row.get("part") or "")
        for row in steps
        if isinstance(row, Mapping)
        and isinstance(row.get("file"), Mapping)
        and row["file"].get("sha256") == expected_sha256
        and any(instance.get("part") == row.get("part") for instance in candidate_instances)
    ]
    if len(matches) != 1 or not matches[0]:
      raise ValueError("graph receipt STEP does not select one authority part")
    ordered_parts.append(matches[0])
  if ordered_parts[0] == ordered_parts[1]:
    raise ValueError("graph receipt selected the same part twice")
  return load_development_frame_source_context_v1(
      index,
      split=normalized_split,
      case_index=case_index,
      part_a=ordered_parts[0],
      part_b=ordered_parts[1],
      row_id=row_id,
  )


def graph_program_inputs_from_safe_view_v1(
    safe_view: BenchmarkV3UnlabeledStepView,
    *,
    catalog: FiniteProgramCatalogRosterV2,
) -> GraphProgramInputsV2:
  if type(safe_view) is not BenchmarkV3UnlabeledStepView:
    raise TypeError("frame authority requires a factory-loaded safe full-BRep view")
  if type(catalog) is not FiniteProgramCatalogRosterV2:
    raise TypeError("frame authority requires the frozen finite catalog")
  assert_no_forbidden_model_visible_fields(safe_view.receipt_payload())
  graphs = safe_view.graphs
  if len(graphs) != 2:
    raise ValueError("frame authority requires exactly two full graphs")
  max_nodes = max(graph.node_features.shape[0] for graph in graphs)
  max_edges = max(graph.edge_index.shape[0] for graph in graphs)
  if max_nodes > 2048 or max_edges > 4096:
    raise ValueError("safe full graph exceeds the authenticated 2048/4096 decision")
  nodes = torch.zeros((1, 2, max_nodes, graphs[0].node_features.shape[1]), dtype=torch.float32)
  node_mask = torch.zeros((1, 2, max_nodes), dtype=torch.bool)
  edges = torch.zeros((1, 2, max_edges, 2), dtype=torch.long)
  edge_features = torch.zeros((1, 2, max_edges, graphs[0].edge_features.shape[1]), dtype=torch.float32)
  edge_mask = torch.zeros((1, 2, max_edges), dtype=torch.bool)
  for slot, graph in enumerate(graphs):
    node_count, edge_count = graph.node_features.shape[0], graph.edge_index.shape[0]
    nodes[0, slot, :node_count] = torch.from_numpy(np.asarray(graph.node_features).copy())
    node_mask[0, slot, :node_count] = True
    if edge_count:
      edges[0, slot, :edge_count] = torch.from_numpy(np.asarray(graph.edge_index).copy())
      edge_features[0, slot, :edge_count] = torch.from_numpy(np.asarray(graph.edge_features).copy())
      edge_mask[0, slot, :edge_count] = True
  return GraphProgramInputsV2(
      graph_tensors=JointInterfaceInputsV1(
          node_features=nodes,
          node_mask=node_mask,
          edge_index=edges,
          edge_features=edge_features,
          edge_mask=edge_mask,
          candidate_program_indices=torch.arange(FORMAL_CATALOG_SIZE, dtype=torch.long).reshape(1, -1),
      ),
      catalog_sha256=catalog.catalog_sha256,
  )


def _replay_frames(
    safe_view: BenchmarkV3UnlabeledStepView,
    context: AuthenticatedFrameSourceContextV1,
    *,
    part_bindings: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  result: list[list[dict[str, Any]]] = []
  for slot, (source, safe_graph, part_binding) in enumerate(
      zip(context.parts, safe_view.graphs, part_bindings, strict=True)
  ):
    reverify_captured_file_artifact(source.step, label="face-frame STEP replay")
    shape = _load_captured_step_shape(source.step)
    faces, face_edges, edge_groups, signatures = _captured_brep_topology(shape)
    graph, raw_to_node = _intrinsic_graph_from_face_subset(
        faces, face_edges, edge_groups, range(len(faces)), budget=_FULL_GRAPH_BUDGET
    )
    adjacency = sum(1 for group in edge_groups if len(group) == 2 and group[0][0] != group[1][0])
    replayed = _graph_payload_to_unlabeled(
        graph,
        raw_face_count=len(faces),
        raw_adjacency_count=adjacency,
        complete_component=True,
        budget=_FULL_GRAPH_BUDGET,
    )
    if any(
        not np.array_equal(left, right)
        for left, right in (
            (replayed.node_features, safe_graph.node_features),
            (replayed.edge_index, safe_graph.edge_index),
            (replayed.edge_features, safe_graph.edge_features),
        )
    ):
      raise ValueError("receipt-bound STEP full graph differs from safe model input")
    node_to_raw = {node: raw for raw, node in raw_to_node.items()}
    if set(node_to_raw) != set(range(len(part_binding.face_signatures))):
      raise ValueError("OCC face identity does not cover the V3 graph node domain")
    world_rotation = _proper_rotation(source.world_rotation, label="receipt-bound instance rotation")
    world_translation = np.asarray(source.world_translation_mm, dtype=float)
    rows: list[dict[str, Any]] = []
    for node_index, model_signature in enumerate(part_binding.face_signatures):
      raw_index = node_to_raw[node_index]
      model_surface_type = _constraint_surface_bucket(
          str(graph["nodes"][node_index]["surface_type"])
      )
      occ_surface_type = _constraint_surface_type(faces[raw_index])
      if model_surface_type != occ_surface_type:
        raise ValueError("model-visible/OCC face surface type replay differs")
      local_origin, local_rotation = _local_face_frame(faces[raw_index])
      rotation = _proper_rotation(_matmul(world_rotation, local_rotation), label="world face frame")
      origin = _matvec(world_rotation, local_origin) + world_translation
      rows.append(
          {
              "graph_local_face_index": node_index,
              "model_visible_face_signature_sha256": model_signature.signature_sha256,
              "occ_face_signature_sha256": signatures[raw_index],
              "raw_occ_face_index": raw_index,
              "surface_type": model_surface_type,
              "origin_world_mm": [float(value) for value in origin],
              "rotation_local_to_world": [[float(value) for value in row] for row in rotation],
              "frame_payload_sha256": canonical_sha256(
                  {
                      "slot": "a" if slot == 0 else "b",
                      "graph_local_face_index": node_index,
                      "model_visible_face_signature_sha256": model_signature.signature_sha256,
                      "occ_face_signature_sha256": signatures[raw_index],
                      "raw_occ_face_index": raw_index,
                      "surface_type": model_surface_type,
                      "origin_world_mm": [float(value) for value in origin],
                      "rotation_local_to_world": [[float(value) for value in row] for row in rotation],
                  }
              ),
          }
      )
    result.append(rows)
  return result[0], result[1]


def _build_authority_payloads(
    safe_view: BenchmarkV3UnlabeledStepView,
    context: AuthenticatedFrameSourceContextV1,
    *,
    catalog: FiniteProgramCatalogRosterV2,
) -> tuple[dict[str, Any], dict[str, Any], GraphProgramInputsV2]:
  if type(context) is not AuthenticatedFrameSourceContextV1:
    raise TypeError("frame authority requires a verifier-issued source context")
  inputs = graph_program_inputs_from_safe_view_v1(safe_view, catalog=catalog)
  source_hash = _source_sha256()
  query_core_sha, part_a, part_b = graph_query_core_commitment_v3(
      inputs,
      case_id=context.case_id,
      row_id=context.row_id,
      part_instance_id_a=context.parts[0].part_instance_id,
      body_id_a=context.parts[0].body_id,
      step_sha256_a=context.parts[0].step.sha256,
      part_instance_id_b=context.parts[1].part_instance_id,
      body_id_b=context.parts[1].body_id,
      step_sha256_b=context.parts[1].step.sha256,
      frame_policy_sha256=PAIR_FRAME_POLICY_SHA256,
      frame_source_sha256=source_hash,
  )
  frames_a, frames_b = _replay_frames(safe_view, context, part_bindings=(part_a, part_b))
  v3_receipt = full_view_frame_receipt_payload_v3(
      query_core_sha256=query_core_sha,
      frame_policy_sha256=PAIR_FRAME_POLICY_SHA256,
      frame_source_sha256=source_hash,
      frames_a=[
          {
              "graph_local_face_index": row["graph_local_face_index"],
              "face_signature_sha256": row["model_visible_face_signature_sha256"],
              "origin_world": row["origin_world_mm"],
              "rotation_local_to_world": row["rotation_local_to_world"],
          }
          for row in frames_a
      ],
      frames_b=[
          {
              "graph_local_face_index": row["graph_local_face_index"],
              "face_signature_sha256": row["model_visible_face_signature_sha256"],
              "origin_world": row["origin_world_mm"],
              "rotation_local_to_world": row["rotation_local_to_world"],
          }
          for row in frames_b
      ],
  )
  authority = {
      "schema_version": AUTHORITY_SCHEMA_VERSION,
      "case_id": context.case_id,
      "row_id": context.row_id,
      "safe_input_sha256": safe_view.input_sha256,
      "query_core_sha256": query_core_sha,
      "source_context_binding_sha256": context.binding_sha256,
      "source_bindings": dict(context.source_bindings),
      "unit_policy": dict(_UNIT_POLICY),
      "frame_policy": dict(_FRAME_POLICY),
      "pair_frame_policy_sha256": PAIR_FRAME_POLICY_SHA256,
      "producer_source_sha256": source_hash,
      "parts": [
          {
              "slot": "a" if index == 0 else "b",
              "part": part.part,
              "part_instance_id": part.part_instance_id,
              "body_id": part.body_id,
              "step": {
                  "path": str(part.step.resolved_path),
                  "bytes": part.step.byte_count,
                  "sha256": part.step.sha256,
              },
              "world_transform_chain_sha256": part.world_transform_chain_sha256,
              "source_instance_key_sha256": part.source_instance_key_sha256,
              "world_transform_mm": {
                  "rotation_row_major": [value for row in part.world_rotation for value in row],
                  "translation_mm": list(part.world_translation_mm),
              },
              "full_graph_sha256": binding.full_graph_sha256,
              "frames": frames_a if index == 0 else frames_b,
          }
          for index, (part, binding) in enumerate(zip(context.parts, (part_a, part_b), strict=True))
      ],
      "v3_receipt_payload_sha256": v3_receipt["receipt_payload_sha256"],
      "final_test_touched": False,
  }
  authority["authority_payload_sha256"] = canonical_sha256(authority)
  return authority, v3_receipt, inputs


def publish_full_graph_face_frame_authority_v1(
    safe_view: BenchmarkV3UnlabeledStepView,
    context: AuthenticatedFrameSourceContextV1,
    *,
    catalog: FiniteProgramCatalogRosterV2,
    output_directory: str | Path,
) -> str:
  authority, v3_receipt, _ = _build_authority_payloads(safe_view, context, catalog=catalog)
  output = Path(output_directory)
  if output.exists():
    raise FileExistsError(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
  try:
    authority_path = staging / "authority.json"
    v3_path = staging / "full_view_face_frame_receipt.v3.json"
    authority_path.write_bytes(_canonical_bytes(authority))
    v3_path.write_bytes(_canonical_bytes(v3_receipt))
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "authority": _binding(authority_path, name=authority_path.name),
        "v3_frame_receipt": _binding(v3_path, name=v3_path.name),
        "producer_source_sha256": _source_sha256(),
        "safe_input_sha256": safe_view.input_sha256,
        "query_core_sha256": authority["query_core_sha256"],
        "final_test_touched": False,
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    artifact_path = staging / "artifact.json"
    artifact_path.write_bytes(_canonical_bytes(artifact))
    pin = _file_sha256(artifact_path)
    os.replace(staging, output)
    return pin
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


def _frame_row_unsigned(row: Mapping[str, Any], *, slot: str) -> dict[str, Any]:
  return {
      "slot": slot,
      "graph_local_face_index": int(row["graph_local_face_index"]),
      "model_visible_face_signature_sha256": str(row["model_visible_face_signature_sha256"]),
      "occ_face_signature_sha256": str(row["occ_face_signature_sha256"]),
      "raw_occ_face_index": int(row["raw_occ_face_index"]),
      "surface_type": str(row["surface_type"]),
      "origin_world_mm": [float(value) for value in row["origin_world_mm"]],
      "rotation_local_to_world": [
          [float(value) for value in values] for values in row["rotation_local_to_world"]
      ],
  }


def _validate_frame_row(row: Mapping[str, Any], *, slot: str) -> None:
  unsigned = _frame_row_unsigned(row, slot=slot)
  if any(
      _SHA256.fullmatch(unsigned[key]) is None
      for key in ("model_visible_face_signature_sha256", "occ_face_signature_sha256")
  ):
    raise ValueError("authority frame signature binding differs")
  if unsigned["surface_type"] not in {"plane", "cylinder", "sphere", "other"}:
    raise ValueError("authority frame surface type differs")
  origin = np.asarray(unsigned["origin_world_mm"], dtype=float)
  rotation = np.asarray(unsigned["rotation_local_to_world"], dtype=float)
  if not np.isfinite(origin).all() or not np.allclose(
      _proper_rotation(rotation, label="authority frame rotation"), rotation, atol=1e-9, rtol=0.0
  ):
    raise ValueError("authority frame payload is not a proper finite frame")
  if row.get("frame_payload_sha256") != canonical_sha256(unsigned):
    raise ValueError("authority frame payload hash differs")


class AuthenticatedFullGraphFaceFrameV1(_ImmutableFactoryObject):
  __slots__ = (
      "query_core_sha256", "part_slot", "graph_local_face_index",
      "model_visible_face_signature_sha256", "occ_face_signature_sha256",
      "raw_occ_face_index", "surface_type", "origin_world_mm", "rotation_local_to_world", "frame_payload_sha256",
      "authority_artifact_sha256",
  )

  def __init__(self, *, row: Mapping[str, Any], slot: str, query_core_sha256: str, pin: str, _factory_token: object) -> None:
    if _factory_token is not _FRAME_FACTORY_TOKEN:
      raise TypeError("full-graph face frames are authority-loader-only")
    _validate_frame_row(row, slot=slot)
    object.__setattr__(self, "query_core_sha256", query_core_sha256)
    object.__setattr__(self, "part_slot", slot)
    object.__setattr__(self, "graph_local_face_index", int(row["graph_local_face_index"]))
    object.__setattr__(self, "model_visible_face_signature_sha256", str(row["model_visible_face_signature_sha256"]))
    object.__setattr__(self, "occ_face_signature_sha256", str(row["occ_face_signature_sha256"]))
    object.__setattr__(self, "raw_occ_face_index", int(row["raw_occ_face_index"]))
    object.__setattr__(self, "surface_type", str(row["surface_type"]))
    object.__setattr__(self, "origin_world_mm", tuple(float(value) for value in row["origin_world_mm"]))
    object.__setattr__(self, "rotation_local_to_world", tuple(
        tuple(float(value) for value in values) for values in row["rotation_local_to_world"]
    ))
    object.__setattr__(self, "frame_payload_sha256", str(row["frame_payload_sha256"]))
    object.__setattr__(self, "authority_artifact_sha256", pin)


class FullGraphFaceFrameAuthorityV1(_ImmutableFactoryObject):
  __slots__ = (
      "_root", "_payload", "_artifact_payload", "_v3_payload", "_file_guards", "_step_guards",
      "_step_captures", "artifact_sha256", "query_core_sha256", "v3_frame_receipt_path",
  )

  def __init__(
      self, *, root: Path, payload: Mapping[str, Any], artifact_payload: Mapping[str, Any],
      v3_payload: Mapping[str, Any], file_guards: Sequence[_CapturedPathGuardV1],
      step_guards: Sequence[_CapturedPathGuardV1], step_captures: Sequence[CapturedFileArtifact],
      pin: str, _factory_token: object,
  ) -> None:
    if _factory_token is not _AUTHORITY_FACTORY_TOKEN:
      raise TypeError("full-graph frame authorities are loader-only")
    object.__setattr__(self, "_root", root.resolve(strict=True))
    object.__setattr__(self, "_payload", _deep_freeze(payload))
    object.__setattr__(self, "_artifact_payload", _deep_freeze(artifact_payload))
    object.__setattr__(self, "_v3_payload", _deep_freeze(v3_payload))
    object.__setattr__(self, "_file_guards", tuple(file_guards))
    object.__setattr__(self, "_step_guards", tuple(step_guards))
    object.__setattr__(self, "_step_captures", tuple(step_captures))
    object.__setattr__(self, "artifact_sha256", pin)
    object.__setattr__(self, "query_core_sha256", str(payload["query_core_sha256"]))
    object.__setattr__(self, "v3_frame_receipt_path", self._root / "full_view_face_frame_receipt.v3.json")
    self._validate_frozen_payload()

  def _validate_frozen_payload(self) -> None:
    payload = _deep_thaw(self._payload)
    observed = payload.pop("authority_payload_sha256", None)
    if observed != canonical_sha256(payload):
      raise ValueError("face-frame authority immutable payload hash differs")
    if payload.get("query_core_sha256") != self.query_core_sha256:
      raise ValueError("face-frame authority immutable query binding differs")
    parts = payload.get("parts")
    if not isinstance(parts, list) or len(parts) != 2:
      raise ValueError("face-frame authority immutable part domain differs")
    for index, part in enumerate(parts):
      slot = "a" if index == 0 else "b"
      if part.get("slot") != slot or not isinstance(part.get("frames"), list):
        raise ValueError("face-frame authority immutable part order differs")
      for frame_index, row in enumerate(part["frames"]):
        if row.get("graph_local_face_index") != frame_index:
          raise ValueError("authority frame ledger ordering differs")
        _validate_frame_row(row, slot=slot)

  def revalidate(self) -> None:
    self._validate_frozen_payload()
    for guard, label in zip(
        self._file_guards,
        ("face-frame artifact", "face-frame authority", "V3 frame receipt"),
        strict=True,
    ):
      guard.revalidate(label=label)
    for guard, captured in zip(self._step_guards, self._step_captures, strict=True):
      guard.revalidate(label="face-frame bound STEP")
      reverify_captured_file_artifact(captured, label="face-frame bound STEP")
    artifact = _deep_thaw(self._artifact_payload)
    authority = _deep_thaw(self._payload)
    v3_payload = _deep_thaw(self._v3_payload)
    if _file_sha256(self._root / "artifact.json") != self.artifact_sha256:
      raise ValueError("face-frame artifact bytes changed after authentication")
    if artifact.get("authority") != _binding(self._root / "authority.json", name="authority.json"):
      raise ValueError("face-frame authority binding changed after authentication")
    if artifact.get("v3_frame_receipt") != _binding(
        self.v3_frame_receipt_path, name=self.v3_frame_receipt_path.name
    ):
      raise ValueError("V3 frame receipt binding changed after authentication")
    if dict(_strict_json(self._root / "authority.json", label="face-frame authority")) != authority:
      raise ValueError("face-frame authority bytes changed after authentication")
    if dict(_strict_json(self.v3_frame_receipt_path, label="V3 frame receipt")) != v3_payload:
      raise ValueError("V3 frame receipt bytes changed after authentication")

  def frame_for(self, *, part_slot: str, graph_local_face_index: int) -> AuthenticatedFullGraphFaceFrameV1:
    self._validate_frozen_payload()
    slot_index = 0 if part_slot == "a" else 1 if part_slot == "b" else -1
    if slot_index < 0 or type(graph_local_face_index) is not int:
      raise ValueError("requested authority frame identity differs")
    frames = self._payload["parts"][slot_index]["frames"]
    if not 0 <= graph_local_face_index < len(frames):
      raise ValueError("requested authority frame is outside the full graph")
    row = frames[graph_local_face_index]
    if row["graph_local_face_index"] != graph_local_face_index:
      raise ValueError("authority frame ledger ordering differs")
    return AuthenticatedFullGraphFaceFrameV1(
        row=row,
        slot=part_slot,
        query_core_sha256=self.query_core_sha256,
        pin=self.artifact_sha256,
        _factory_token=_FRAME_FACTORY_TOKEN,
    )

  @property
  def row_id(self) -> str:
    self._validate_frozen_payload()
    return str(self._payload["row_id"])

  @property
  def safe_input_sha256(self) -> str:
    self._validate_frozen_payload()
    return str(self._payload["safe_input_sha256"])

  @property
  def source_context_binding_sha256(self) -> str:
    self._validate_frozen_payload()
    return str(self._payload["source_context_binding_sha256"])

  @property
  def part_world_transforms(self) -> tuple[Transform, Transform]:
    self._validate_frozen_payload()
    result = []
    for part in self._payload["parts"]:
      transform = part["world_transform_mm"]
      result.append(
          Transform(
              rotation=np.asarray(transform["rotation_row_major"], dtype=float).reshape(3, 3),
              translation=np.asarray(transform["translation_mm"], dtype=float),
          )
      )
    return result[0], result[1]


def load_full_graph_face_frame_authority_v1(
    authority_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    safe_view: BenchmarkV3UnlabeledStepView,
    source_context: AuthenticatedFrameSourceContextV1,
    catalog: FiniteProgramCatalogRosterV2,
) -> FullGraphFaceFrameAuthorityV1:
  root = Path(authority_directory)
  artifact_path = root / "artifact.json"
  if _file_sha256(artifact_path) != expected_artifact_sha256:
    raise ValueError("face-frame authority artifact pin differs")
  artifact = _strict_json(artifact_path, label="face-frame authority artifact")
  expected_artifact = dict(artifact)
  observed_artifact_hash = expected_artifact.pop("artifact_payload_sha256", None)
  if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION or observed_artifact_hash != canonical_sha256(expected_artifact):
    raise ValueError("face-frame authority artifact self hash differs")
  if artifact.get("producer_source_sha256") != _source_sha256():
    raise ValueError("face-frame authority producer source differs")
  authority_path = root / "authority.json"
  v3_path = root / "full_view_face_frame_receipt.v3.json"
  if artifact.get("authority") != _binding(authority_path, name=authority_path.name) or artifact.get(
      "v3_frame_receipt"
  ) != _binding(v3_path, name=v3_path.name):
    raise ValueError("face-frame authority bound files changed")
  authority = _strict_json(authority_path, label="face-frame authority")
  observed_authority_hash = dict(authority).pop("authority_payload_sha256", None)
  unsigned = dict(authority)
  unsigned.pop("authority_payload_sha256", None)
  if observed_authority_hash != canonical_sha256(unsigned):
    raise ValueError("face-frame authority self hash differs")
  expected_authority, expected_v3, _ = _build_authority_payloads(
      safe_view, source_context, catalog=catalog
  )
  v3_payload = _strict_json(v3_path, label="V3 frame receipt")
  if dict(authority) != expected_authority or dict(v3_payload) != expected_v3:
    raise ValueError("face-frame authority OCC/source replay differs")
  file_guards = tuple(_CapturedPathGuardV1(path) for path in (artifact_path, authority_path, v3_path))
  step_captures = tuple(part.step for part in source_context.parts)
  step_guards = tuple(_CapturedPathGuardV1(captured.resolved_path) for captured in step_captures)
  return FullGraphFaceFrameAuthorityV1(
      root=root,
      payload=authority,
      artifact_payload=artifact,
      v3_payload=v3_payload,
      file_guards=file_guards,
      step_guards=step_guards,
      step_captures=step_captures,
      pin=expected_artifact_sha256,
      _factory_token=_AUTHORITY_FACTORY_TOKEN,
  )


def _proposal_output_payload(
    executable: ExecutablePredictedProgramV3, *, topk_ordinal: int,
) -> dict[str, Any]:
  residual = (
      *executable.residual_translation_pair_local,
      *executable.residual_rotation_vector_pair_local,
      executable.score,
  )
  if any(not math.isfinite(float(value)) for value in residual):
    raise ValueError("proposal run contains non-finite model output")
  payload = {
      "topk_ordinal": topk_ordinal,
      "query_binding_sha256": executable.query_binding_sha256,
      "case_id": executable.case_id,
      "row_id": executable.row_id,
      "face_index_a": executable.face_index_a,
      "face_signature_sha256_a": executable.face_signature_sha256_a,
      "face_index_b": executable.face_index_b,
      "face_signature_sha256_b": executable.face_signature_sha256_b,
      "program_index": executable.program_index,
      "program_id": executable.program_id,
      "descriptor": executable.descriptor.payload(),
      "descriptor_sha256": executable.descriptor_sha256,
      "catalog_roster_sha256": executable.catalog_roster_sha256,
      "learner_source_sha256": executable.learner_source_sha256,
      "model_config_sha256": executable.model_config_sha256,
      "model_state_sha256": executable.model_state_sha256,
      "residual_translation_pair_local": list(executable.residual_translation_pair_local),
      "residual_rotation_vector_pair_local": list(executable.residual_rotation_vector_pair_local),
      "pair_frame_policy_sha256": executable.pair_frame_policy_sha256,
      "score": executable.score,
      "budget_trace": executable.trace.payload(),
  }
  payload["output_payload_sha256"] = canonical_sha256(payload)
  return payload


def publish_executor_proposal_run_v1(
    executables: Sequence[ExecutablePredictedProgramV3],
    query: GraphQueryBindingV3,
    *,
    catalog_spec: ExternallyFrozenCatalogSpecV3,
    output_directory: str | Path,
) -> str:
  """Persist executor-produced top-K outputs; this does not issue capabilities."""

  if type(query) is not GraphQueryBindingV3 or type(catalog_spec) is not ExternallyFrozenCatalogSpecV3:
    raise TypeError("proposal run requires a V3 query and frozen catalog")
  rows = tuple(executables)
  if not rows or any(type(row) is not ExecutablePredictedProgramV3 for row in rows):
    raise TypeError("proposal run requires exact executor-produced V3 outputs")
  trace_payload = rows[0].trace.payload()
  if any(row.trace.payload() != trace_payload for row in rows):
    raise ValueError("proposal run top-K outputs do not share one executor trace")
  if (
      trace_payload.get("schema_version") != "preemptive_topk_trace.v3"
      or trace_payload.get("worker_exit_code") != 0
      or trace_payload.get("top_k") != len(rows)
      or trace_payload.get("program_pool") != FORMAL_CATALOG_SIZE
      or trace_payload.get("deadline_scope") != "pre_process_start_through_parent_topk_accept.v3"
      or trace_payload.get("worker_payload") != "lightweight_topk_only.v3"
  ):
    raise ValueError("proposal run executor budget trace differs")
  outputs = []
  for ordinal, executable in enumerate(rows):
    validate_executable_query_replay_v3(executable, query)
    entry = catalog_spec.roster.entry(executable.program_index)
    if (
        executable.catalog_roster_sha256 != catalog_spec.roster.catalog_sha256
        or executable.program_id != entry.program_id
        or executable.descriptor_sha256 != entry.descriptor_sha256
        or executable.descriptor != entry.descriptor
    ):
      raise ValueError("proposal run output differs from frozen catalog")
    outputs.append(_proposal_output_payload(executable, topk_ordinal=ordinal))
  run_payload = {
      "schema_version": PROPOSAL_RUN_SCHEMA_VERSION,
      "query_binding_sha256": query.binding_sha256,
      "query_core_sha256": query.query_core_sha256,
      "case_id": query.case_id,
      "row_id": query.row_id,
      "catalog_roster_sha256": catalog_spec.roster.catalog_sha256,
      "catalog_spec_file_sha256": catalog_spec.spec_file_sha256,
      "budget_trace_sha256": canonical_sha256(trace_payload),
      "topk_count": len(outputs),
      "outputs": outputs,
      "complete_output_commitment_sha256": canonical_sha256(
          [row["output_payload_sha256"] for row in outputs]
      ),
      "producer_source_sha256": _source_sha256(),
      "final_test_touched": False,
  }
  run_payload["run_payload_sha256"] = canonical_sha256(run_payload)
  output = Path(output_directory)
  if output.exists():
    raise FileExistsError(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
  try:
    run_path = staging / "proposal_run.json"
    run_path.write_bytes(_canonical_bytes(run_payload))
    artifact = {
        "schema_version": PROPOSAL_RUN_ARTIFACT_SCHEMA_VERSION,
        "proposal_run": _binding(run_path, name=run_path.name),
        "query_binding_sha256": query.binding_sha256,
        "catalog_spec_file_sha256": catalog_spec.spec_file_sha256,
        "producer_source_sha256": _source_sha256(),
        "final_test_touched": False,
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    artifact_path = staging / "artifact.json"
    artifact_path.write_bytes(_canonical_bytes(artifact))
    pin = _file_sha256(artifact_path)
    os.replace(staging, output)
    return pin
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


class AuthenticatedPredictedExecutionCapabilityV1(_ImmutableFactoryObject):
  __slots__ = (
      "_root", "_artifact", "_run", "_output", "_descriptor", "_guards",
      "artifact_sha256", "complete_output_commitment_sha256", "schema_version",
  )

  def __init__(
      self, *, root: Path, artifact: Mapping[str, Any], run: Mapping[str, Any],
      output: Mapping[str, Any], descriptor: FiniteProgramDescriptorV2,
      guards: Sequence[_CapturedPathGuardV1], pin: str, _factory_token: object,
  ) -> None:
    if _factory_token is not _CAPABILITY_FACTORY_TOKEN:
      raise TypeError("predicted execution capabilities are proposal-run-loader-only")
    object.__setattr__(self, "_root", root.resolve(strict=True))
    object.__setattr__(self, "_artifact", _deep_freeze(artifact))
    object.__setattr__(self, "_run", _deep_freeze(run))
    object.__setattr__(self, "_output", _deep_freeze(output))
    object.__setattr__(self, "_descriptor", descriptor)
    object.__setattr__(self, "_guards", tuple(guards))
    object.__setattr__(self, "artifact_sha256", pin)
    object.__setattr__(self, "complete_output_commitment_sha256", str(run["complete_output_commitment_sha256"]))
    object.__setattr__(self, "schema_version", CAPABILITY_SCHEMA_VERSION)

  def revalidate(self) -> None:
    for guard, label in zip(self._guards, ("proposal artifact", "proposal run"), strict=True):
      guard.revalidate(label=label)
    if _file_sha256(self._root / "artifact.json") != self.artifact_sha256:
      raise ValueError("proposal artifact bytes changed after authentication")
    if dict(_strict_json(self._root / "artifact.json", label="proposal artifact")) != _deep_thaw(self._artifact):
      raise ValueError("proposal artifact payload changed after authentication")
    if dict(_strict_json(self._root / "proposal_run.json", label="proposal run")) != _deep_thaw(self._run):
      raise ValueError("proposal run payload changed after authentication")
    output = _deep_thaw(self._output)
    claimed = output.pop("output_payload_sha256", None)
    if claimed != canonical_sha256(output):
      raise ValueError("proposal output immutable commitment differs")

  def _value(self, key: str) -> Any:
    return self._output[key]

  @property
  def descriptor(self) -> FiniteProgramDescriptorV2: return self._descriptor
  @property
  def topk_ordinal(self) -> int: return int(self._value("topk_ordinal"))
  @property
  def query_binding_sha256(self) -> str: return str(self._value("query_binding_sha256"))
  @property
  def face_index_a(self) -> int: return int(self._value("face_index_a"))
  @property
  def face_index_b(self) -> int: return int(self._value("face_index_b"))
  @property
  def face_signature_sha256_a(self) -> str: return str(self._value("face_signature_sha256_a"))
  @property
  def face_signature_sha256_b(self) -> str: return str(self._value("face_signature_sha256_b"))
  @property
  def program_index(self) -> int: return int(self._value("program_index"))
  @property
  def program_id(self) -> str: return str(self._value("program_id"))
  @property
  def descriptor_sha256(self) -> str: return str(self._value("descriptor_sha256"))
  @property
  def catalog_roster_sha256(self) -> str: return str(self._value("catalog_roster_sha256"))
  @property
  def learner_source_sha256(self) -> str: return str(self._value("learner_source_sha256"))
  @property
  def model_config_sha256(self) -> str: return str(self._value("model_config_sha256"))
  @property
  def model_state_sha256(self) -> str: return str(self._value("model_state_sha256"))
  @property
  def pair_frame_policy_sha256(self) -> str: return str(self._value("pair_frame_policy_sha256"))
  @property
  def residual_translation_pair_local(self) -> tuple[float, ...]:
    return tuple(float(value) for value in self._value("residual_translation_pair_local"))
  @property
  def residual_rotation_vector_pair_local(self) -> tuple[float, ...]:
    return tuple(float(value) for value in self._value("residual_rotation_vector_pair_local"))


def load_authenticated_predicted_execution_capability_v1(
    proposal_run_directory: str | Path,
    *, expected_artifact_sha256: str, query: GraphQueryBindingV3,
    catalog_spec: ExternallyFrozenCatalogSpecV3, topk_ordinal: int,
) -> AuthenticatedPredictedExecutionCapabilityV1:
  if type(query) is not GraphQueryBindingV3 or type(catalog_spec) is not ExternallyFrozenCatalogSpecV3:
    raise TypeError("proposal capability loader requires V3 query and frozen catalog")
  if type(topk_ordinal) is not int or topk_ordinal < 0:
    raise ValueError("proposal capability ordinal differs")
  root = Path(proposal_run_directory)
  artifact_path, run_path = root / "artifact.json", root / "proposal_run.json"
  if _file_sha256(artifact_path) != expected_artifact_sha256:
    raise ValueError("proposal artifact pin differs")
  artifact = _strict_json(artifact_path, label="proposal artifact")
  artifact_unsigned = dict(artifact)
  artifact_hash = artifact_unsigned.pop("artifact_payload_sha256", None)
  if (
      artifact.get("schema_version") != PROPOSAL_RUN_ARTIFACT_SCHEMA_VERSION
      or artifact_hash != canonical_sha256(artifact_unsigned)
      or artifact.get("producer_source_sha256") != _source_sha256()
      or artifact.get("proposal_run") != _binding(run_path, name=run_path.name)
  ):
    raise ValueError("proposal artifact authentication differs")
  run = _strict_json(run_path, label="proposal run")
  run_unsigned = dict(run)
  run_hash = run_unsigned.pop("run_payload_sha256", None)
  outputs = run.get("outputs")
  if (
      run.get("schema_version") != PROPOSAL_RUN_SCHEMA_VERSION
      or run_hash != canonical_sha256(run_unsigned)
      or run.get("producer_source_sha256") != _source_sha256()
      or run.get("query_binding_sha256") != query.binding_sha256
      or run.get("query_core_sha256") != query.query_core_sha256
      or run.get("catalog_roster_sha256") != catalog_spec.roster.catalog_sha256
      or run.get("catalog_spec_file_sha256") != catalog_spec.spec_file_sha256
      or not isinstance(outputs, list)
      or run.get("topk_count") != len(outputs)
      or not 0 <= topk_ordinal < len(outputs)
      or run.get("complete_output_commitment_sha256")
      != canonical_sha256([row.get("output_payload_sha256") for row in outputs if isinstance(row, Mapping)])
  ):
    raise ValueError("proposal run authentication differs")
  for ordinal, row in enumerate(outputs):
    if not isinstance(row, Mapping) or row.get("topk_ordinal") != ordinal:
      raise ValueError("proposal run top-K ordinal differs")
    unsigned = dict(row)
    observed = unsigned.pop("output_payload_sha256", None)
    if observed != canonical_sha256(unsigned):
      raise ValueError("proposal run output commitment differs")
    descriptor_payload = row.get("descriptor")
    if not isinstance(descriptor_payload, Mapping):
      raise ValueError("proposal run descriptor payload differs")
    descriptor = FiniteProgramDescriptorV2(**dict(descriptor_payload))
    entry = catalog_spec.roster.entry(int(row.get("program_index", -1)))
    if (
        row.get("query_binding_sha256") != query.binding_sha256
        or row.get("case_id") != query.case_id or row.get("row_id") != query.row_id
        or row.get("face_signature_sha256_a")
        != query.part_a.face_signatures[int(row.get("face_index_a", -1))].signature_sha256
        or row.get("face_signature_sha256_b")
        != query.part_b.face_signatures[int(row.get("face_index_b", -1))].signature_sha256
        or descriptor != entry.descriptor or row.get("descriptor_sha256") != entry.descriptor_sha256
        or row.get("program_id") != entry.program_id
    ):
      raise ValueError("proposal run output/query/catalog replay differs")
    residual = (*row.get("residual_translation_pair_local", ()), *row.get("residual_rotation_vector_pair_local", ()), row.get("score"))
    if len(residual) != 7 or any(not math.isfinite(float(value)) for value in residual):
      raise ValueError("proposal run output is non-finite")
  selected = outputs[topk_ordinal]
  selected_descriptor = FiniteProgramDescriptorV2(**dict(selected["descriptor"]))
  return AuthenticatedPredictedExecutionCapabilityV1(
      root=root, artifact=artifact, run=run, output=selected, descriptor=selected_descriptor,
      guards=(_CapturedPathGuardV1(artifact_path), _CapturedPathGuardV1(run_path)),
      pin=expected_artifact_sha256, _factory_token=_CAPABILITY_FACTORY_TOKEN,
  )


@dataclass(frozen=True, slots=True)
class ExecutableSE3ResultV1:
  world_delta: Transform
  child_world_transform: Transform
  binding_sha256: str
  authority_artifact_sha256: str
  query_binding_sha256: str
  program_id: str
  descriptor_sha256: str
  catalog_spec_file_sha256: str
  proposal_run_artifact_sha256: str
  complete_output_commitment_sha256: str
  _factory_token: object
  schema_version: str = EXECUTION_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self._factory_token is not _EXECUTION_FACTORY_TOKEN:
      raise TypeError("executable SE(3) results are adapter-only")
    if any(_SHA256.fullmatch(value) is None for value in (
        self.binding_sha256,
        self.authority_artifact_sha256,
        self.query_binding_sha256,
        self.descriptor_sha256,
        self.catalog_spec_file_sha256,
        self.proposal_run_artifact_sha256,
        self.complete_output_commitment_sha256,
    )):
      raise ValueError("executable SE(3) result binding differs")


class ExecutablePredictedProgramAdapterV1:
  """Retired residual-only adapter.

  Categorical program identity plus an unconstrained six-vector is not an
  executable mate.  The only supported execution seam is now
  ``authenticated_exact_execution_v2.world_placement_for_blind_proposal_v2``
  with a loader-authenticated constraint catalog.
  """

  def execute(
      self,
      executable: AuthenticatedPredictedExecutionCapabilityV1,
      query: GraphQueryBindingV3,
      authority: FullGraphFaceFrameAuthorityV1,
      *,
      catalog_spec: ExternallyFrozenCatalogSpecV3,
  ) -> ExecutableSE3ResultV1:
    del executable, query, authority, catalog_spec
    raise FiniteProgramBaseSE3SchemaMissing(
        "residual-only full-SE(3) execution is forbidden; use the "
        "loader-authenticated constraint execution API"
    )
