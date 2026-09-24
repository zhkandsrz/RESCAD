"""V5-safe-view-bound, source-pose-independent exact-execution placements.

The V5 model view is intrinsically SE(3)-invariant and therefore carries no
world placement.  This authority deterministically supplies the missing
*current query pose* from only the safe-view commitment and receipt-bound STEP
instance identity.  Source assembly transforms and intended mate transforms
are neither read while deriving the pose nor serialized into the artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
from weakref import WeakKeyDictionary

import numpy as np

from .benchmark_v2_model_view_v2 import (
    GraphSizeBudget,
    _load_captured_step_shape,
    extract_intrinsic_brep_graph,
)
from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    capture_file_artifact,
    reverify_captured_file_artifact,
)
from .benchmark_v3_unlabeled_step_view import (
    BenchmarkV3UnlabeledStepView,
    canonical_bytes,
    canonical_sha256,
)
from .domain_types import Transform
from .brep_tensor_cache_v4_unlabeled import _graph_payload_to_unlabeled
from .frame_randomization import (
    intrinsic_assembly_scale,
    intrinsic_shape_geometry,
    sample_independent_part_gauges,
)
from .joint_interface_training_v3 import AuthenticatedJointStepIdentityV5


ROW_SCHEMA = "view_execution_pose_row.v1"
MANIFEST_SCHEMA = "view_execution_pose_authority.v1"
ARTIFACT_SCHEMA = "view_execution_pose_authority_artifact.v1"
CONSUMER_SCOPE = "exact_executor_current_pose_pre_physical_execution"
POSE_POLICY = "safe_view_bound_independent_se3_centered_step.v1"
PAIR_ORDER = "query_role_a_then_query_role_b"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FACTORY_TOKEN = object()


def _strict_json(raw: bytes, *, label: str) -> Any:
  def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
      if key in result:
        raise ValueError(f"{label} contains duplicate key {key!r}")
      result[key] = value
    return result
  try:
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite constant {value}")
        ),
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _file_binding(path: Path, *, name: str | None = None) -> dict[str, Any]:
  raw = path.read_bytes()
  return {
      "name": name or path.name,
      "bytes": len(raw),
      "sha256": hashlib.sha256(raw).hexdigest(),
  }


def _external_binding(captured: CapturedFileArtifact) -> dict[str, Any]:
  return {
      "path": str(captured.resolved_path),
      "bytes": captured.byte_count,
      "sha256": captured.sha256,
  }


def _capture_bound_external(value: Any, *, label: str) -> CapturedFileArtifact:
  if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
    raise ValueError(f"{label} binding schema differs")
  path = value.get("path")
  if (
      not isinstance(path, str) or not path or not Path(path).is_absolute()
      or type(value.get("bytes")) is not int or int(value["bytes"]) < 1
      or _SHA256.fullmatch(str(value.get("sha256"))) is None
  ):
    raise ValueError(f"{label} binding differs")
  captured = capture_file_artifact(path, label=label)
  if captured.byte_count != value["bytes"] or captured.sha256 != value["sha256"]:
    raise ValueError(f"{label} bytes changed")
  return captured


def _capture_local_binding(root: Path, value: Any, *, label: str) -> CapturedFileArtifact:
  if not isinstance(value, Mapping) or set(value) != {"name", "bytes", "sha256"}:
    raise ValueError(f"{label} binding schema differs")
  name = value.get("name")
  if (
      not isinstance(name, str) or not name or Path(name).is_absolute()
      or ".." in Path(name).parts
  ):
    raise ValueError(f"{label} path differs")
  captured = capture_file_artifact(root / name, label=label)
  if captured.byte_count != value["bytes"] or captured.sha256 != value["sha256"]:
    raise ValueError(f"{label} bytes changed")
  return captured


def _deep_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
  if isinstance(value, (list, tuple)):
    return tuple(_deep_freeze(item) for item in value)
  return value


def _transform_payload(transform: Transform) -> dict[str, Any]:
  rotation = np.asarray(transform.rotation, dtype=float).reshape(3, 3)
  translation = np.asarray(transform.translation, dtype=float).reshape(3)
  return {
      "rotation_row_major": [float(value) for value in rotation.reshape(-1)],
      "translation_mm": [float(value) for value in translation],
  }


def _validated_transform(value: Any, *, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {
      "rotation_row_major", "translation_mm"
  }:
    raise ValueError(f"{label} schema differs")
  rotation = np.asarray(value["rotation_row_major"], dtype=float)
  translation = np.asarray(value["translation_mm"], dtype=float)
  if rotation.shape != (9,) or translation.shape != (3,) or not (
      np.isfinite(rotation).all() and np.isfinite(translation).all()
  ):
    raise ValueError(f"{label} dimensions differ")
  matrix = rotation.reshape(3, 3)
  orthonormal = all(
      abs(sum(
          float(matrix[row, column_a]) * float(matrix[row, column_b])
          for row in range(3)
      ) - (1.0 if column_a == column_b else 0.0)) <= 1e-8
      for column_a in range(3) for column_b in range(3)
  )
  determinant = float(
      matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
      - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
      + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
  )
  if not orthonormal or not math.isclose(determinant, 1.0, abs_tol=1e-8):
    raise ValueError(f"{label} rotation is not proper SO(3)")
  return value


def _pose_master_seed(safe_input_sha256: str) -> int:
  payload = f"{POSE_POLICY}\0{safe_input_sha256}".encode("ascii")
  return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _pose_part_key(
    *, query_role: str, query_instance_identity_sha256: str, step_sha256: str,
) -> str:
  return canonical_sha256({
      "schema_version": "view_execution_pose_part_key.v1",
      "query_role": query_role,
      "query_instance_identity_sha256": query_instance_identity_sha256,
      "step_sha256": step_sha256,
  })


def _derive_pose_parts(
    *, safe_input_sha256: str, identities: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
  if len(identities) != 2:
    raise ValueError("view execution pose requires exactly two query parts")
  measured = []
  keys = []
  for index, identity in enumerate(identities):
    expected_role = ("a", "b")[index]
    if identity.get("query_role") != expected_role:
      raise ValueError("view execution pose pair order differs")
    captured = identity.get("step_capture")
    if not isinstance(captured, CapturedFileArtifact):
      raise TypeError("view execution pose requires captured STEP bytes")
    reverify_captured_file_artifact(captured, label="view execution pose STEP")
    geometry = intrinsic_shape_geometry(_load_captured_step_shape(captured))
    measured.append(geometry)
    keys.append(_pose_part_key(
        query_role=expected_role,
        query_instance_identity_sha256=str(identity["query_instance_identity_sha256"]),
        step_sha256=captured.sha256,
    ))
  scale = intrinsic_assembly_scale(measured)
  gauges = sample_independent_part_gauges(
      keys,
      seed=_pose_master_seed(safe_input_sha256),
      assembly_nonce=safe_input_sha256,
      assembly_scale=scale,
      translation_box_fraction=1.0,
  )
  result = []
  for index, (identity, geometry, key) in enumerate(
      zip(identities, measured, keys, strict=True)
  ):
    captured = identity["step_capture"]
    gauge = gauges.transform_for(key)
    rotation = np.asarray(gauge.rotation, dtype=float).reshape(3, 3)
    origin = tuple(float(value) for value in geometry.origin)
    translation = np.asarray([
        float(gauge.translation[row]) - sum(
            float(rotation[row, column]) * origin[column] for column in range(3)
        )
        for row in range(3)
    ], dtype=float)
    current = Transform(rotation=rotation, translation=translation)
    transform = _transform_payload(current)
    result.append({
        "graph_index": index,
        "query_role": ("a", "b")[index],
        "query_instance_identity_sha256": str(
            identity["query_instance_identity_sha256"]
        ),
        "step_file": _external_binding(captured),
        "step_sha256": captured.sha256,
        "safe_graph_receipt_sha256": str(identity["safe_graph_receipt_sha256"]),
        "current_world_transform_mm": transform,
        "current_pose_sha256": canonical_sha256(transform),
    })
  return tuple(result)


def _step_graph_receipt(
    captured: CapturedFileArtifact, *, graph_index: int,
) -> Mapping[str, Any]:
  """Replay the exact V5 full-graph encoder from captured STEP bytes."""

  reverify_captured_file_artifact(captured, label="view execution pose STEP graph")
  graph = extract_intrinsic_brep_graph(
      _load_captured_step_shape(captured), budget=GraphSizeBudget()
  )
  nodes = graph.get("nodes")
  edges = graph.get("edges")
  if not isinstance(nodes, list) or not isinstance(edges, list):
    raise ValueError("view execution pose STEP graph differs")
  replayed = _graph_payload_to_unlabeled(
      graph,
      raw_face_count=len(nodes),
      raw_adjacency_count=len(edges),
      complete_component=True,
      budget=GraphSizeBudget(),
  )
  return {
      "schema_version": "benchmark_v3_unlabeled_intrinsic_brep_graph.v1",
      "tensors": [dict(value) for value in replayed.tensor_receipts(
          prefix=f"g{graph_index}"
      )],
  }


def build_view_execution_pose_row_v1(
    *, step_identity: AuthenticatedJointStepIdentityV5,
    step_captures: Sequence[CapturedFileArtifact],
    safe_view: BenchmarkV3UnlabeledStepView,
    v5_bundle_artifact_sha256: str, producer_revision: str,
) -> dict[str, Any]:
  """Derive one current-pose receipt without reading source world placement."""

  if type(safe_view) is not BenchmarkV3UnlabeledStepView:
    raise TypeError("view execution poses require a factory-owned V5 safe view")
  if type(step_identity) is not AuthenticatedJointStepIdentityV5:
    raise TypeError("view execution poses require a factory-owned V5 STEP identity")
  row_ordinal = step_identity.row_ordinal
  safe_input = step_identity.safe_input_sha256
  if (
      type(row_ordinal) is not int or row_ordinal < 0
      or safe_input != safe_view.input_sha256
      or step_identity.bundle_artifact_sha256 != v5_bundle_artifact_sha256
      or not isinstance(step_captures, (tuple, list)) or len(step_captures) != 2
      or _SHA256.fullmatch(str(v5_bundle_artifact_sha256)) is None
      or _REVISION.fullmatch(str(producer_revision)) is None
  ):
    raise ValueError("view execution pose upstream binding differs")
  identities = []
  receipt_graphs = safe_view.receipt_payload().get("graphs")
  if not isinstance(receipt_graphs, list) or len(receipt_graphs) != 2:
    raise ValueError("V5 safe view graph domain differs")
  for index, captured in enumerate(step_captures):
    role = ("a", "b")[index]
    expected_step_sha256 = step_identity.step_sha256s[index]
    instance_identity = canonical_sha256({
        "schema_version": "v5_query_slot_identity.v1",
        "row_ordinal": row_ordinal,
        "safe_input_sha256": safe_input,
        "query_role": role,
        "step_sha256": expected_step_sha256,
    })
    if (
        not isinstance(captured, CapturedFileArtifact)
        or captured.sha256 != expected_step_sha256
        or _SHA256.fullmatch(str(instance_identity)) is None
    ):
      raise ValueError("view execution pose query identity differs")
    actual_graph = _step_graph_receipt(captured, graph_index=index)
    if actual_graph != receipt_graphs[index]:
      raise ValueError("V5 STEP identity does not reproduce the safe graph")
    identities.append({
        "query_role": role,
        "query_instance_identity_sha256": str(instance_identity),
        "step_capture": captured,
        "safe_graph_receipt_sha256": canonical_sha256(receipt_graphs[index]),
    })
  derived = _derive_pose_parts(
      safe_input_sha256=str(safe_input), identities=identities
  )
  receipt = {
      "schema_version": ROW_SCHEMA,
      "consumer_scope": CONSUMER_SCOPE,
      "pose_policy": POSE_POLICY,
      "pair_order": PAIR_ORDER,
      "row_ordinal": row_ordinal,
      "safe_input_sha256": str(safe_input),
      "safe_view_receipt_sha256": canonical_sha256(safe_view.receipt_payload()),
      "v5_step_identity_sha256": step_identity.identity_payload_sha256,
      "v5_bundle_artifact_sha256": str(v5_bundle_artifact_sha256),
      "producer_revision": str(producer_revision),
      "source_pose_read": False,
      "parts": list(derived),
      "final_test_touched": False,
  }
  receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
  return receipt


def _validate_row(value: Any) -> Mapping[str, Any]:
  expected = {
      "schema_version", "consumer_scope", "pose_policy", "pair_order",
      "row_ordinal", "safe_input_sha256", "safe_view_receipt_sha256",
      "v5_step_identity_sha256", "v5_bundle_artifact_sha256",
      "producer_revision", "source_pose_read", "parts",
      "final_test_touched", "receipt_payload_sha256",
  }
  if not isinstance(value, Mapping) or set(value) != expected:
    raise ValueError("view execution pose row schema differs")
  unsigned = dict(value)
  observed = unsigned.pop("receipt_payload_sha256")
  if observed != canonical_sha256(unsigned):
    raise ValueError("view execution pose row hash differs")
  if (
      value["schema_version"] != ROW_SCHEMA
      or value["consumer_scope"] != CONSUMER_SCOPE
      or value["pose_policy"] != POSE_POLICY
      or value["pair_order"] != PAIR_ORDER
      or value["source_pose_read"] is not False
      or value["final_test_touched"] is not False
      or type(value["row_ordinal"]) is not int or value["row_ordinal"] < 0
      or _REVISION.fullmatch(str(value["producer_revision"])) is None
      or any(_SHA256.fullmatch(str(value[field])) is None for field in (
          "safe_input_sha256", "safe_view_receipt_sha256",
          "v5_step_identity_sha256", "v5_bundle_artifact_sha256",
          "receipt_payload_sha256",
      ))
  ):
    raise ValueError("view execution pose row policy differs")
  parts = value["parts"]
  part_keys = {
      "graph_index", "query_role", "query_instance_identity_sha256",
      "step_file", "step_sha256", "safe_graph_receipt_sha256",
      "current_world_transform_mm", "current_pose_sha256",
  }
  if not isinstance(parts, list) or len(parts) != 2:
    raise ValueError("view execution pose part domain differs")
  for index, part in enumerate(parts):
    if not isinstance(part, Mapping) or set(part) != part_keys:
      raise ValueError("view execution pose part schema differs")
    transform = _validated_transform(
        part["current_world_transform_mm"], label="view current transform"
    )
    if (
        part["graph_index"] != index
        or part["query_role"] != ("a", "b")[index]
        or any(_SHA256.fullmatch(str(part[field])) is None for field in (
            "query_instance_identity_sha256", "step_sha256",
            "safe_graph_receipt_sha256", "current_pose_sha256",
        ))
        or part["current_pose_sha256"] != canonical_sha256(transform)
    ):
      raise ValueError("view execution pose part identity differs")
    binding = part["step_file"]
    if not isinstance(binding, Mapping) or binding.get("sha256") != part["step_sha256"]:
      raise ValueError("view execution pose STEP binding differs")
  # An explicit negative canary is allowed; all positive oracle-bearing fields are not.
  lowered_keys = "\n".join(
      str(key).lower()
      for part in parts
      for key in part
  )
  if any(token in lowered_keys for token in (
      "target", "gold", "label", "program", "residual", "contact", "mate",
      "source_world", "source_transform", "world_transform_chain",
  )):
    raise ValueError("view execution pose payload exposes oracle fields")
  return value


def _replay_row_pose(
    row: Mapping[str, Any], *, step_captures: Sequence[CapturedFileArtifact],
) -> None:
  identities = []
  for part, captured in zip(row["parts"], step_captures, strict=True):
    identities.append({
        "query_role": part["query_role"],
        "query_instance_identity_sha256": part[
            "query_instance_identity_sha256"
        ],
        "step_capture": captured,
        "safe_graph_receipt_sha256": part["safe_graph_receipt_sha256"],
    })
  replayed = _derive_pose_parts(
      safe_input_sha256=str(row["safe_input_sha256"]), identities=identities
  )
  if list(replayed) != row["parts"]:
    raise ValueError("view execution pose differs from source-independent replay")


def publish_view_execution_pose_authority_v1(
    output_directory: str | Path, *, row_receipts: Sequence[Mapping[str, Any]],
    v5_bundle_artifact_path: str | Path, v5_bundle_artifact_sha256: str,
    producer_revision: str,
) -> str:
  """Atomically publish a development pose authority without a source-pose pin."""

  output = Path(output_directory)
  if output.exists():
    raise FileExistsError(output)
  rows = [dict(_validate_row(row)) for row in row_receipts]
  rows.sort(key=lambda row: int(row["row_ordinal"]))
  if (
      not rows or len({int(row["row_ordinal"]) for row in rows}) != len(rows)
      or _REVISION.fullmatch(producer_revision) is None
      or any(row["producer_revision"] != producer_revision for row in rows)
      or any(row["v5_bundle_artifact_sha256"] != v5_bundle_artifact_sha256 for row in rows)
  ):
    raise ValueError("view execution pose publication domain differs")
  v5_capture = capture_file_artifact(
      v5_bundle_artifact_path, label="view execution pose V5 artifact"
  )
  if v5_capture.sha256 != v5_bundle_artifact_sha256:
    raise ValueError("view execution pose V5 artifact differs")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
  try:
    receipts_root = staging / "receipts"
    receipts_root.mkdir()
    ledger = []
    receipt_bindings = []
    for row in rows:
      path = receipts_root / f"row-{int(row['row_ordinal']):08d}.json"
      path.write_bytes(canonical_bytes(row))
      binding = _file_binding(path, name=f"receipts/{path.name}")
      receipt_bindings.append(binding)
      ledger.append({
          "row_ordinal": int(row["row_ordinal"]),
          "safe_input_sha256": row["safe_input_sha256"],
          "safe_view_receipt_sha256": row["safe_view_receipt_sha256"],
          "receipt": binding,
          "receipt_payload_sha256": row["receipt_payload_sha256"],
      })
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "consumer_scope": CONSUMER_SCOPE,
        "pose_policy": POSE_POLICY,
        "scope": "p0_train_dev_view_execution_pose_no_final",
        "producer_revision": producer_revision,
        "v5_bundle_artifact_sha256": v5_bundle_artifact_sha256,
        "v5_bundle_artifact": _external_binding(v5_capture),
        "row_count": len(rows),
        "row_domain_sha256": canonical_sha256([
            {"row_ordinal": row["row_ordinal"], "safe_input_sha256": row["safe_input_sha256"]}
            for row in rows
        ]),
        "rows": ledger,
        "source_pose_bound": False,
        "final_test_touched": False,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    manifest_path = staging / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "consumer_scope": CONSUMER_SCOPE,
        "pose_policy": POSE_POLICY,
        "producer_revision": producer_revision,
        "manifest": _file_binding(manifest_path),
        "row_receipts": receipt_bindings,
        "source_pose_bound": False,
        "final_test_touched": False,
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    artifact_path = staging / "artifact.json"
    artifact_path.write_bytes(canonical_bytes(artifact))
    pin = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    os.replace(staging, output)
    return pin
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


_PART_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
_ROW_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
_AUTHORITY_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


class ViewExecutionPosePartV1:
  __slots__ = ("__weakref__",)

  def __init__(self, *, payload: Mapping[str, Any], step: CapturedFileArtifact,
               _factory_token: object) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError("view execution pose parts are loader-factory-only")
    _PART_STATES[self] = MappingProxyType({
        "payload": _deep_freeze(payload), "step": step,
    })

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("authenticated view execution pose parts are sealed")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated view execution pose parts are not serializable")

  def _state(self) -> Mapping[str, Any]:
    try:
      return _PART_STATES[self]
    except KeyError as error:
      raise TypeError("view execution pose part lacks factory state") from error

  @property
  def query_role(self) -> str:
    return str(self._state()["payload"]["query_role"])

  @property
  def query_part_id(self) -> str:
    return f"v5-query-slot-{self.query_role}"

  @property
  def body_uuid(self) -> str:
    return self.query_instance_identity_sha256

  @property
  def query_instance_identity_sha256(self) -> str:
    return str(self._state()["payload"]["query_instance_identity_sha256"])

  @property
  def step(self) -> CapturedFileArtifact:
    captured = self._state()["step"]
    return CapturedFileArtifact(
        path=captured.path, resolved_path=captured.resolved_path,
        raw_bytes=captured.raw_bytes, sha256=captured.sha256,
        byte_count=captured.byte_count, device=captured.device,
        inode=captured.inode, link_count=captured.link_count,
    )

  @property
  def current_world_transform_mm(self) -> Mapping[str, Any]:
    return self._state()["payload"]["current_world_transform_mm"]

  @property
  def current_pose_sha256(self) -> str:
    return str(self._state()["payload"]["current_pose_sha256"])

  @property
  def safe_graph_receipt_sha256(self) -> str:
    return str(self._state()["payload"]["safe_graph_receipt_sha256"])


class AuthenticatedViewExecutionPoseRowV1:
  __slots__ = ("__weakref__",)

  def __init__(self, *, payload: Mapping[str, Any], parts: Sequence[ViewExecutionPosePartV1],
               _factory_token: object) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError("view execution pose rows are loader-factory-only")
    _ROW_STATES[self] = MappingProxyType({
        "payload": _deep_freeze(payload), "parts": tuple(parts),
    })

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("authenticated view execution pose rows are sealed")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated view execution pose rows are not serializable")

  @property
  def row_ordinal(self) -> int:
    return int(self._state()["payload"]["row_ordinal"])

  @property
  def safe_input_sha256(self) -> str:
    return str(self._state()["payload"]["safe_input_sha256"])

  @property
  def binding_sha256(self) -> str:
    return str(self._state()["payload"]["receipt_payload_sha256"])

  @property
  def parts(self) -> tuple[ViewExecutionPosePartV1, ViewExecutionPosePartV1]:
    return self._state()["parts"]  # type: ignore[return-value]

  def _state(self) -> Mapping[str, Any]:
    try:
      return _ROW_STATES[self]
    except KeyError as error:
      raise TypeError("view execution pose row lacks factory state") from error


class AuthenticatedViewExecutionPoseAuthorityV1:
  __slots__ = ("__weakref__",)

  def __init__(self, *, rows: Mapping[int, AuthenticatedViewExecutionPoseRowV1],
               captures: Sequence[CapturedFileArtifact], artifact_sha256: str,
               _factory_token: object) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError("view execution pose authorities are loader-factory-only")
    _AUTHORITY_STATES[self] = MappingProxyType({
        "rows": MappingProxyType(dict(rows)), "captures": tuple(captures),
        "artifact_sha256": artifact_sha256,
    })

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("authenticated view execution pose authorities are sealed")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated view execution pose authorities are not serializable")

  def row(self, row_ordinal: int) -> AuthenticatedViewExecutionPoseRowV1:
    self.revalidate()
    try:
      return self._state()["rows"][int(row_ordinal)]
    except KeyError as error:
      raise ValueError("row ordinal is outside view execution pose authority") from error

  @property
  def row_ordinals(self) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in self._state()["rows"]))

  @property
  def artifact_sha256(self) -> str:
    return str(self._state()["artifact_sha256"])

  def revalidate(self) -> None:
    for captured in self._state()["captures"]:
      try:
        reverify_captured_file_artifact(captured, label="view execution pose authority")
      except ValueError as error:
        raise ValueError("view execution pose authority TOCTOU bytes changed") from error

  def _state(self) -> Mapping[str, Any]:
    try:
      return _AUTHORITY_STATES[self]
    except KeyError as error:
      raise TypeError("view execution pose authority lacks factory state") from error


def load_view_execution_pose_authority_v1(
    root: str | Path, *, expected_artifact_sha256: str,
    expected_v5_bundle_artifact_sha256: str, expected_producer_revision: str,
    safe_view_loader: Callable[[int], BenchmarkV3UnlabeledStepView],
    step_identity_loader: Callable[[int], AuthenticatedJointStepIdentityV5],
) -> AuthenticatedViewExecutionPoseAuthorityV1:
  """Load and independently replay every current pose from safe view + STEP."""

  directory = Path(root)
  artifact_capture = capture_file_artifact(
      directory / "artifact.json", label="view execution pose artifact"
  )
  if artifact_capture.sha256 != expected_artifact_sha256:
    raise ValueError("view execution pose artifact differs from external pin")
  artifact = _strict_json(artifact_capture.raw_bytes, label="view execution pose artifact")
  if not isinstance(artifact, Mapping) or set(artifact) != {
      "schema_version", "consumer_scope", "pose_policy", "producer_revision",
      "manifest", "row_receipts", "source_pose_bound", "final_test_touched",
      "artifact_payload_sha256",
  }:
    raise ValueError("view execution pose artifact schema differs")
  unsigned = dict(artifact)
  observed = unsigned.pop("artifact_payload_sha256")
  if (
      observed != canonical_sha256(unsigned)
      or artifact["schema_version"] != ARTIFACT_SCHEMA
      or artifact["consumer_scope"] != CONSUMER_SCOPE
      or artifact["pose_policy"] != POSE_POLICY
      or artifact["producer_revision"] != expected_producer_revision
      or artifact["source_pose_bound"] is not False
      or artifact["final_test_touched"] is not False
  ):
    raise ValueError("view execution pose artifact policy differs")
  manifest_capture = _capture_local_binding(
      directory, artifact["manifest"], label="view execution pose manifest"
  )
  manifest = _strict_json(manifest_capture.raw_bytes, label="view execution pose manifest")
  if not isinstance(manifest, Mapping) or set(manifest) != {
      "schema_version", "consumer_scope", "pose_policy", "scope",
      "producer_revision", "v5_bundle_artifact_sha256", "v5_bundle_artifact",
      "row_count", "row_domain_sha256", "rows", "source_pose_bound",
      "final_test_touched", "manifest_payload_sha256",
  }:
    raise ValueError("view execution pose manifest schema differs")
  unsigned = dict(manifest)
  observed = unsigned.pop("manifest_payload_sha256")
  if (
      observed != canonical_sha256(unsigned)
      or manifest["schema_version"] != MANIFEST_SCHEMA
      or manifest["consumer_scope"] != CONSUMER_SCOPE
      or manifest["pose_policy"] != POSE_POLICY
      or manifest["scope"] != "p0_train_dev_view_execution_pose_no_final"
      or manifest["producer_revision"] != expected_producer_revision
      or manifest["v5_bundle_artifact_sha256"] != expected_v5_bundle_artifact_sha256
      or manifest["source_pose_bound"] is not False
      or manifest["final_test_touched"] is not False
  ):
    raise ValueError("view execution pose manifest policy differs")
  v5_capture = _capture_bound_external(
      manifest["v5_bundle_artifact"], label="view execution pose V5 artifact"
  )
  if v5_capture.sha256 != expected_v5_bundle_artifact_sha256:
    raise ValueError("view execution pose V5 artifact bytes changed")
  ledger = manifest["rows"]
  artifact_receipts = artifact["row_receipts"]
  if (
      not isinstance(ledger, list) or not isinstance(artifact_receipts, list)
      or len(ledger) != manifest["row_count"]
      or len(ledger) != len(artifact_receipts)
  ):
    raise ValueError("view execution pose row ledger differs")
  by_name = {
      row.get("name"): row for row in artifact_receipts if isinstance(row, Mapping)
  }
  if len(by_name) != len(artifact_receipts):
    raise ValueError("view execution pose receipt binding duplicates")
  captures = [artifact_capture, manifest_capture, v5_capture]
  rows: dict[int, AuthenticatedViewExecutionPoseRowV1] = {}
  row_domain = []
  for ledger_row in ledger:
    if not isinstance(ledger_row, Mapping) or set(ledger_row) != {
        "row_ordinal", "safe_input_sha256", "safe_view_receipt_sha256",
        "receipt", "receipt_payload_sha256",
    }:
      raise ValueError("view execution pose ledger row differs")
    binding = ledger_row["receipt"]
    if by_name.get(binding.get("name")) != binding:
      raise ValueError("view execution pose artifact/manifest binding differs")
    receipt_capture = _capture_local_binding(
        directory, binding, label="view execution pose row receipt"
    )
    receipt = dict(_validate_row(_strict_json(
        receipt_capture.raw_bytes, label="view execution pose row receipt"
    )))
    ordinal = int(receipt["row_ordinal"])
    if (
        ordinal in rows or ledger_row["row_ordinal"] != ordinal
        or ledger_row["safe_input_sha256"] != receipt["safe_input_sha256"]
        or ledger_row["safe_view_receipt_sha256"] != receipt["safe_view_receipt_sha256"]
        or ledger_row["receipt_payload_sha256"] != receipt["receipt_payload_sha256"]
        or receipt["v5_bundle_artifact_sha256"] != expected_v5_bundle_artifact_sha256
        or receipt["producer_revision"] != expected_producer_revision
    ):
      raise ValueError("view execution pose row identity differs")
    safe_view = safe_view_loader(ordinal)
    step_identity = step_identity_loader(ordinal)
    if (
        type(safe_view) is not BenchmarkV3UnlabeledStepView
        or type(step_identity) is not AuthenticatedJointStepIdentityV5
        or safe_view.input_sha256 != receipt["safe_input_sha256"]
        or step_identity.row_ordinal != ordinal
        or step_identity.safe_input_sha256 != receipt["safe_input_sha256"]
        or step_identity.bundle_artifact_sha256
        != expected_v5_bundle_artifact_sha256
        or step_identity.identity_payload_sha256
        != receipt["v5_step_identity_sha256"]
        or canonical_sha256(safe_view.receipt_payload())
        != receipt["safe_view_receipt_sha256"]
    ):
      raise ValueError("view execution pose safe-view replay differs")
    step_captures = tuple(
        _capture_bound_external(part["step_file"], label="view execution pose STEP")
        for part in receipt["parts"]
    )
    receipt_graphs = safe_view.receipt_payload().get("graphs")
    if not isinstance(receipt_graphs, list) or len(receipt_graphs) != 2:
      raise ValueError("view execution pose safe graph domain differs")
    for index, (part, captured) in enumerate(
        zip(receipt["parts"], step_captures, strict=True)
    ):
      if (
          captured.sha256 != step_identity.step_sha256s[index]
          or part["safe_graph_receipt_sha256"]
          != canonical_sha256(receipt_graphs[index])
          or _step_graph_receipt(captured, graph_index=index)
          != receipt_graphs[index]
      ):
        raise ValueError("view execution pose STEP/safe graph identity differs")
    _replay_row_pose(receipt, step_captures=step_captures)
    parts = tuple(
        ViewExecutionPosePartV1(
            payload=part, step=captured, _factory_token=_FACTORY_TOKEN
        )
        for part, captured in zip(receipt["parts"], step_captures, strict=True)
    )
    rows[ordinal] = AuthenticatedViewExecutionPoseRowV1(
        payload=receipt, parts=parts, _factory_token=_FACTORY_TOKEN
    )
    captures.extend((receipt_capture, *step_captures))
    row_domain.append({
        "row_ordinal": ordinal, "safe_input_sha256": receipt["safe_input_sha256"]
    })
  if manifest["row_domain_sha256"] != canonical_sha256(row_domain):
    raise ValueError("view execution pose row domain differs")
  authority = AuthenticatedViewExecutionPoseAuthorityV1(
      rows=rows, captures=captures, artifact_sha256=expected_artifact_sha256,
      _factory_token=_FACTORY_TOKEN,
  )
  authority.revalidate()
  return authority


__all__ = [
    "AuthenticatedViewExecutionPoseAuthorityV1",
    "AuthenticatedViewExecutionPoseRowV1",
    "ViewExecutionPosePartV1",
    "build_view_execution_pose_row_v1",
    "load_view_execution_pose_authority_v1",
    "publish_view_execution_pose_authority_v1",
]
