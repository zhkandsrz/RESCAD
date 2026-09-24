"""Executor-only, post-proposal execution inputs for full-graph V5 rows.

The sidecar does not infer an occurrence from STEP/topology equality.  It is
issued after the producer reads the source mate row and asserts endpoint
supervision consistency.  Its payload excludes face, finite-program, residual
and label fields while preserving repeated STEP instances exactly.
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

from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    capture_file_artifact,
    reverify_captured_file_artifact,
)
from .benchmark_v3_unlabeled_step_view import canonical_bytes, canonical_sha256


MANIFEST_SCHEMA = "benchmark_v6_executor_input_authority.v1"
ROW_RECEIPT_SCHEMA = "benchmark_v6_executor_input_row_receipt.v1"
ARTIFACT_SCHEMA = "benchmark_v6_executor_input_authority_artifact.v1"
CHECKPOINT_SCHEMA = "benchmark_v6_executor_input_authority_checkpoint.v1"
CONSUMER_SCOPE = "executor_only_post_proposal"
PAYLOAD_POLICY = "payload_excludes_face_program_residual_labels"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FACTORY_TOKEN = object()
_FORBIDDEN_TOKENS = (
    "endpoint", "face_index", "program", "residual", "label", "target",
    "contact", "gold", "mate",
)


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
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite constant {value}")
        ),
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _binding(path: Path, *, name: str | None = None) -> dict[str, Any]:
  raw = path.read_bytes()
  return {
      "name": name or path.name,
      "bytes": len(raw),
      "sha256": hashlib.sha256(raw).hexdigest(),
  }


def _capture_binding(root: Path, value: Any, *, label: str) -> CapturedFileArtifact:
  if not isinstance(value, Mapping) or set(value) != {"name", "bytes", "sha256"}:
    raise ValueError(f"{label} binding schema differs")
  name = value.get("name")
  if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
    raise ValueError(f"{label} relative path differs")
  captured = capture_file_artifact(root / name, label=label)
  if (
      captured.byte_count != value.get("bytes")
      or captured.sha256 != value.get("sha256")
  ):
    raise ValueError(f"{label} bytes differ")
  return captured


def _deep_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
  if isinstance(value, list | tuple):
    return tuple(_deep_freeze(item) for item in value)
  return value


def _scan_no_gold(value: Any, *, path: str = "root") -> None:
  if isinstance(value, Mapping):
    for key, item in value.items():
      normalized = str(key).lower()
      if any(token in normalized for token in _FORBIDDEN_TOKENS):
        raise ValueError(f"forbidden execution-input field at {path}.{key}")
      _scan_no_gold(item, path=f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _scan_no_gold(item, path=f"{path}[{index}]")


def _validated_transform(value: Any, *, label: str) -> dict[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {
      "rotation_row_major", "translation_mm"
  }:
    raise ValueError(f"{label} schema differs")
  rotation = value["rotation_row_major"]
  translation = value["translation_mm"]
  if (
      not isinstance(rotation, list) or len(rotation) != 9
      or not isinstance(translation, list) or len(translation) != 3
  ):
    raise ValueError(f"{label} dimensions differ")
  values = [float(item) for item in rotation] + [
      float(item) for item in translation
  ]
  if any(not math.isfinite(item) for item in values):
    raise ValueError(f"{label} is non-finite")
  return {
      "rotation_row_major": [float(item) for item in rotation],
      "translation_mm": [float(item) for item in translation],
  }


def _validated_external_file(value: Any, *, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
    raise ValueError(f"{label} binding schema differs")
  path = value.get("path")
  if (
      not isinstance(path, str)
      or not path
      or not Path(path).is_absolute()
      or type(value.get("bytes")) is not int
      or int(value["bytes"]) < 1
      or _SHA256.fullmatch(str(value.get("sha256"))) is None
  ):
    raise ValueError(f"{label} binding differs")
  return value


def _capture_external_file(value: Any, *, label: str) -> CapturedFileArtifact:
  binding = _validated_external_file(value, label=label)
  captured = capture_file_artifact(binding["path"], label=label)
  if (
      captured.byte_count != binding["bytes"]
      or captured.sha256 != binding["sha256"]
  ):
    raise ValueError(f"{label} bytes changed")
  return captured


def _validate_execution_projection(value: Any) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {
      "schema_version", "consumer_scope", "payload_policy",
      "case_authority_binding_sha256", "private_source_file",
      "stage_manifest_file", "pair_order", "graphs", "projection_payload_sha256",
  }:
    raise ValueError("execution query projection schema differs")
  unsigned = dict(value)
  observed = unsigned.pop("projection_payload_sha256")
  if observed != canonical_sha256(unsigned):
    raise ValueError("execution query projection hash differs")
  if (
      value["schema_version"] != "benchmark_v6_executor_query_inputs.v1"
      or value["consumer_scope"] != CONSUMER_SCOPE
      or value["payload_policy"] != PAYLOAD_POLICY
      or value["pair_order"] != "query_role_a_then_query_role_b"
      or _SHA256.fullmatch(str(value["case_authority_binding_sha256"])) is None
  ):
    raise ValueError("execution query projection policy differs")
  _validated_external_file(
      value["private_source_file"], label="execution private source"
  )
  _validated_external_file(
      value["stage_manifest_file"], label="execution stage manifest"
  )
  graphs = value["graphs"]
  if not isinstance(graphs, list) or len(graphs) != 2:
    raise ValueError("execution query projection requires A/B roles")
  expected_keys = {
      "graph_index", "query_role", "source_part", "body_uuid", "step_file",
      "step_sha256", "source_instance_key_sha256",
      "world_transform_chain_sha256", "instance_transform_sha256",
      "world_transform_mm",
  }
  for index, graph in enumerate(graphs):
    if not isinstance(graph, Mapping) or set(graph) != expected_keys:
      raise ValueError("execution query graph schema differs")
    step = graph["step_file"]
    if (
        graph["graph_index"] != index
        or graph["query_role"] != ("a", "b")[index]
        or not isinstance(graph["source_part"], str) or not graph["source_part"]
        or not isinstance(graph["body_uuid"], str) or not graph["body_uuid"]
        or not isinstance(step, Mapping)
        or set(step) != {"path", "bytes", "sha256"}
        or not isinstance(step.get("path"), str)
        or not step["path"]
        or not Path(step["path"]).is_absolute()
        or step["sha256"] != graph["step_sha256"]
        or type(step["bytes"]) is not int or step["bytes"] < 1
        or any(_SHA256.fullmatch(str(graph[key])) is None for key in (
            "step_sha256", "source_instance_key_sha256",
            "world_transform_chain_sha256", "instance_transform_sha256",
        ))
    ):
      raise ValueError("execution query graph identity differs")
    transform = _validated_transform(
        graph["world_transform_mm"], label="execution query world transform"
    )
    if canonical_sha256(transform) != graph["instance_transform_sha256"]:
      raise ValueError("execution query transform hash differs")
  _scan_no_gold(value)
  return value


def build_execution_input_row_receipt_v1(
    *,
    row_ordinal: int,
    source_row_ordinal: int,
    source_row_payload_sha256: str,
    safe_row: Mapping[str, Any],
    execution_projection: Mapping[str, Any],
    v5_bundle_artifact_sha256: str,
    source_cache_artifact_sha256: str,
    source_authority_sha256: str,
    producer_revision: str,
) -> dict[str, Any]:
  """Build one row at the producer seam after complete-graph materialization."""

  projection = _validate_execution_projection(execution_projection)
  if (
      type(row_ordinal) is not int
      or row_ordinal < 0
      or type(source_row_ordinal) is not int
      or source_row_ordinal < 0
  ):
    raise ValueError("execution input row ordinal differs")
  required_safe = {
      "row_ordinal", "input_sha256", "row_payload_sha256", "view_receipt",
      "full_graph_proofs",
  }
  if not required_safe.issubset(safe_row) or safe_row["row_ordinal"] != row_ordinal:
    raise ValueError("execution input safe row binding differs")
  proofs = safe_row["full_graph_proofs"]
  graphs = projection["graphs"]
  view_graphs = safe_row["view_receipt"].get("graphs") if isinstance(
      safe_row["view_receipt"], Mapping
  ) else None
  if not isinstance(proofs, list) or len(proofs) != 2 or not isinstance(view_graphs, list) or len(view_graphs) != 2:
    raise ValueError("execution input complete-graph receipt differs")
  full_graphs = []
  for index, (query, proof, graph_receipt) in enumerate(
      zip(graphs, proofs, view_graphs, strict=True)
  ):
    if (
        not isinstance(proof, Mapping)
        or proof.get("step_sha256") != query["step_sha256"]
        or _SHA256.fullmatch(str(proof.get("full_topology_sha256"))) is None
    ):
      raise ValueError("execution input STEP/full-topology binding differs")
    full_graphs.append({
        "graph_index": index,
        "query_role": ("a", "b")[index],
        "query_instance": dict(query),
        "full_graph_proof": dict(proof),
        "full_graph_tensor_receipt": dict(graph_receipt),
    })
  for value in (
      str(safe_row["input_sha256"]), str(safe_row["row_payload_sha256"]),
      source_row_payload_sha256, v5_bundle_artifact_sha256,
      source_cache_artifact_sha256, source_authority_sha256,
  ):
    if _SHA256.fullmatch(value) is None:
      raise ValueError("execution input upstream pin differs")
  if _REVISION.fullmatch(producer_revision) is None:
    raise ValueError("execution input producer revision differs")
  if projection["case_authority_binding_sha256"] != source_authority_sha256:
    raise ValueError("execution projection/source authority differs")
  receipt = {
      "schema_version": ROW_RECEIPT_SCHEMA,
      "consumer_scope": CONSUMER_SCOPE,
      "payload_policy": PAYLOAD_POLICY,
      "row_ordinal": row_ordinal,
      "source_row_ordinal": source_row_ordinal,
      "source_row_payload_sha256": source_row_payload_sha256,
      "safe_input_sha256": str(safe_row["input_sha256"]),
      "safe_row_payload_sha256": str(safe_row["row_payload_sha256"]),
      "v5_bundle_artifact_sha256": v5_bundle_artifact_sha256,
      "source_cache_artifact_sha256": source_cache_artifact_sha256,
      "source_authority_sha256": source_authority_sha256,
      "private_source_file": dict(projection["private_source_file"]),
      "stage_manifest_file": dict(projection["stage_manifest_file"]),
      "producer_revision": producer_revision,
      "pair_order": "query_role_a_then_query_role_b",
      "graphs": full_graphs,
      "final_test_touched": False,
  }
  _scan_no_gold(receipt)
  receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
  return receipt


def _validate_row_receipt(value: Any) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError("execution input row receipt is not an object")
  expected = {
      "schema_version", "consumer_scope", "payload_policy",
      "row_ordinal", "source_row_ordinal",
      "source_row_payload_sha256", "safe_input_sha256", "safe_row_payload_sha256",
      "v5_bundle_artifact_sha256", "source_cache_artifact_sha256",
      "source_authority_sha256", "private_source_file", "stage_manifest_file",
      "producer_revision", "pair_order", "graphs",
      "final_test_touched", "receipt_payload_sha256",
  }
  if (
      set(value) != expected
      or value.get("schema_version") != ROW_RECEIPT_SCHEMA
      or value.get("consumer_scope") != CONSUMER_SCOPE
      or value.get("payload_policy") != PAYLOAD_POLICY
  ):
    raise ValueError("execution input row receipt schema differs")
  unsigned = dict(value)
  observed = unsigned.pop("receipt_payload_sha256")
  if observed != canonical_sha256(unsigned):
    raise ValueError("execution input row receipt hash differs")
  if value.get("final_test_touched") is not False or value.get("pair_order") != "query_role_a_then_query_role_b":
    raise ValueError("execution input row receipt policy differs")
  graphs = value.get("graphs")
  if not isinstance(graphs, list) or len(graphs) != 2:
    raise ValueError("execution input row receipt graph domain differs")
  for index, graph in enumerate(graphs):
    if (
        not isinstance(graph, Mapping)
        or set(graph) != {
            "full_graph_proof", "full_graph_tensor_receipt", "graph_index",
            "query_instance", "query_role",
        }
        or graph.get("graph_index") != index
        or graph.get("query_role") != ("a", "b")[index]
    ):
      raise ValueError("execution input row receipt A/B order differs")
    _validate_execution_projection({
        "schema_version": "benchmark_v6_executor_query_inputs.v1",
        "consumer_scope": value["consumer_scope"],
        "payload_policy": value["payload_policy"],
        "case_authority_binding_sha256": value["source_authority_sha256"],
        "private_source_file": dict(value["private_source_file"]),
        "stage_manifest_file": dict(value["stage_manifest_file"]),
        "pair_order": value["pair_order"],
        "graphs": [
            dict(graphs[0]["query_instance"]), dict(graphs[1]["query_instance"])
        ],
        "projection_payload_sha256": canonical_sha256({
            "schema_version": "benchmark_v6_executor_query_inputs.v1",
            "consumer_scope": value["consumer_scope"],
            "payload_policy": value["payload_policy"],
            "case_authority_binding_sha256": value["source_authority_sha256"],
            "private_source_file": dict(value["private_source_file"]),
            "stage_manifest_file": dict(value["stage_manifest_file"]),
            "pair_order": value["pair_order"],
            "graphs": [
                dict(graphs[0]["query_instance"]), dict(graphs[1]["query_instance"])
            ],
        }),
    })
    proof = graph.get("full_graph_proof")
    if not isinstance(proof, Mapping) or proof.get("step_sha256") != graph["query_instance"]["step_sha256"]:
      raise ValueError("execution input row proof differs")
  for field in (
      "source_row_payload_sha256", "safe_input_sha256",
      "safe_row_payload_sha256", "v5_bundle_artifact_sha256",
      "source_cache_artifact_sha256", "source_authority_sha256",
  ):
    if _SHA256.fullmatch(str(value.get(field))) is None:
      raise ValueError("execution input row upstream digest differs")
  if (
      type(value.get("row_ordinal")) is not int
      or value["row_ordinal"] < 0
      or type(value.get("source_row_ordinal")) is not int
      or value["source_row_ordinal"] < 0
      or _REVISION.fullmatch(str(value.get("producer_revision"))) is None
  ):
    raise ValueError("execution input row ordinal/revision differs")
  _scan_no_gold(value)
  return value


def execution_input_upstream_pin_domains(
    row_receipts: Sequence[Mapping[str, Any]],
) -> Mapping[str, str]:
  """Commit exact per-row upstream authority pins for an executor domain."""

  rows = sorted(
      (dict(_validate_row_receipt(row)) for row in row_receipts),
      key=lambda row: int(row["row_ordinal"]),
  )
  if not rows or len({int(row["row_ordinal"]) for row in rows}) != len(rows):
    raise ValueError("execution input upstream pin domain differs")
  return MappingProxyType({
      "source_authority_domain_sha256": canonical_sha256([
          {
              "row_ordinal": row["row_ordinal"],
              "source_authority_sha256": row["source_authority_sha256"],
          }
          for row in rows
      ]),
      "private_source_domain_sha256": canonical_sha256([
          {
              "row_ordinal": row["row_ordinal"],
              "private_source_file_sha256": row["private_source_file"]["sha256"],
          }
          for row in rows
      ]),
      "stage_manifest_domain_sha256": canonical_sha256([
          {
              "row_ordinal": row["row_ordinal"],
              "stage_manifest_file_sha256": row["stage_manifest_file"]["sha256"],
          }
          for row in rows
      ]),
  })


def reconcile_fully_replayed_checkpoint_row_v1(
    target: str | Path,
    *,
    freshly_replayed_receipt: Mapping[str, Any],
) -> str:
  """Persist a new row or require an existing row to equal a full replay."""

  path = Path(target)
  replayed = dict(_validate_row_receipt(freshly_replayed_receipt))
  if path.exists():
    existing = dict(_validate_row_receipt(
        _strict_json(path.read_bytes(), label="execution input checkpoint row")
    ))
    if existing != replayed:
      raise ValueError(
          "execution input resumed row differs from full current replay"
      )
    return "row_revalidated"
  temporary = path.with_suffix(".tmp")
  temporary.write_bytes(canonical_bytes(replayed))
  os.replace(temporary, path)
  return "row_replayed"


def publish_execution_input_authority_v1(
    output_directory: str | Path,
    *,
    row_receipts: Sequence[Mapping[str, Any]],
    v5_bundle_root: str | Path,
    v5_bundle_artifact_sha256: str,
    source_cache_artifact_path: str | Path,
    source_cache_artifact_sha256: str,
    source_authority_projection_path: str | Path,
    source_authority_projection_sha256: str,
    expected_source_authority_domain_sha256: str,
    expected_private_source_domain_sha256: str,
    expected_stage_manifest_domain_sha256: str,
    producer_revision: str,
    selection_policy: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> str:
  """Atomically publish prebuilt producer-time row receipts."""

  output = Path(output_directory)
  if output.exists():
    raise FileExistsError(output)
  if _REVISION.fullmatch(producer_revision) is None:
    raise ValueError("execution input producer revision differs")
  v5_root = Path(v5_bundle_root)
  v5_capture = capture_file_artifact(
      v5_root / "artifact.json", label="execution input V5 bundle"
  )
  if v5_capture.sha256 != v5_bundle_artifact_sha256:
    raise ValueError("execution input V5 bundle differs from external pin")
  source_cache_capture = capture_file_artifact(
      source_cache_artifact_path,
      label="execution input source cache artifact",
  )
  if source_cache_capture.sha256 != source_cache_artifact_sha256:
    raise ValueError("execution input source cache differs from external pin")
  source_authority_capture = capture_file_artifact(
      source_authority_projection_path,
      label="execution input source authority projection",
  )
  if source_authority_capture.sha256 != source_authority_projection_sha256:
    raise ValueError("execution input source authority differs from external pin")
  checked = [dict(_validate_row_receipt(value)) for value in row_receipts]
  if not checked or len({int(row["row_ordinal"]) for row in checked}) != len(checked):
    raise ValueError("execution input row domain differs")
  if any(row["v5_bundle_artifact_sha256"] != v5_bundle_artifact_sha256 for row in checked):
    raise ValueError("execution input rows differ from V5 pin")
  if any(
      row["source_cache_artifact_sha256"] != source_cache_artifact_sha256
      for row in checked
  ):
    raise ValueError("execution input rows differ from source cache pin")
  domains = dict(execution_input_upstream_pin_domains(checked))
  expected_domains = {
      "source_authority_domain_sha256": expected_source_authority_domain_sha256,
      "private_source_domain_sha256": expected_private_source_domain_sha256,
      "stage_manifest_domain_sha256": expected_stage_manifest_domain_sha256,
  }
  if domains != expected_domains or any(
      _SHA256.fullmatch(value) is None for value in expected_domains.values()
  ):
    raise ValueError("execution input external upstream domains differ")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
  try:
    receipt_root = staging / "receipts"
    receipt_root.mkdir()
    ledger = []
    for sequence, row in enumerate(sorted(checked, key=lambda value: int(value["row_ordinal"]))):
      name = f"receipts/row-{int(row['row_ordinal']):08d}.json"
      path = staging / name
      path.write_bytes(canonical_bytes(row))
      ledger.append({
          "row_ordinal": int(row["row_ordinal"]),
          "safe_input_sha256": row["safe_input_sha256"],
          "safe_row_payload_sha256": row["safe_row_payload_sha256"],
          "full_topology_sha256s": [
              graph["full_graph_proof"]["full_topology_sha256"]
              for graph in row["graphs"]
          ],
          "receipt": _binding(path, name=name),
          "receipt_payload_sha256": row["receipt_payload_sha256"],
      })
      if progress_callback is not None:
        progress_callback({"stage": "row_written", "completed_rows": sequence + 1, "total_rows": len(checked)})
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "consumer_scope": CONSUMER_SCOPE,
        "payload_policy": PAYLOAD_POLICY,
        "scope": "p0_train_dev_executor_only_post_proposal_no_final",
        "producer_revision": producer_revision,
        "v5_bundle_artifact_sha256": v5_bundle_artifact_sha256,
        "v5_bundle_artifact": {
            "path": str(v5_capture.resolved_path), "bytes": v5_capture.byte_count,
            "sha256": v5_capture.sha256,
        },
        "source_cache_artifact_sha256": source_cache_artifact_sha256,
        "source_cache_artifact": {
            "path": str(source_cache_capture.resolved_path),
            "bytes": source_cache_capture.byte_count,
            "sha256": source_cache_capture.sha256,
        },
        "source_authority_projection_sha256": source_authority_projection_sha256,
        "source_authority_projection": {
            "path": str(source_authority_capture.resolved_path),
            "bytes": source_authority_capture.byte_count,
            "sha256": source_authority_capture.sha256,
        },
        "upstream_pin_domains": domains,
        "selection_policy": dict(selection_policy),
        "row_count": len(ledger),
        "row_domain_sha256": canonical_sha256([row["row_ordinal"] for row in ledger]),
        "rows": ledger,
        "final_test_touched": False,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    manifest_path = staging / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "consumer_scope": CONSUMER_SCOPE,
        "payload_policy": PAYLOAD_POLICY,
        "producer_revision": producer_revision,
        "manifest": _binding(manifest_path, name="manifest.json"),
        "row_receipts": [_binding(staging / row["receipt"]["name"], name=row["receipt"]["name"]) for row in ledger],
        "final_test_touched": False,
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    artifact_path = staging / "artifact.json"
    artifact_path.write_bytes(canonical_bytes(artifact))
    pin = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    reverify_captured_file_artifact(v5_capture, label="execution input V5 publish")
    reverify_captured_file_artifact(
        source_cache_capture, label="execution input source cache publish"
    )
    reverify_captured_file_artifact(
        source_authority_capture,
        label="execution input source authority publish",
    )
    os.replace(staging, output)
    if progress_callback is not None:
      progress_callback({"stage": "published", "row_count": len(ledger), "artifact_sha256": pin})
    return pin
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


_PART_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
_ROW_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()
_AUTHORITY_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


def _copy_capture(captured: CapturedFileArtifact) -> CapturedFileArtifact:
  """Return a disposable view without exposing the factory-sealed capture."""

  return CapturedFileArtifact(
      path=captured.path,
      resolved_path=captured.resolved_path,
      raw_bytes=captured.raw_bytes,
      sha256=captured.sha256,
      byte_count=captured.byte_count,
      device=captured.device,
      inode=captured.inode,
      link_count=captured.link_count,
  )


class ExecutionInputPartV1:
  """A sealed executor part whose authority capture remains private."""

  __slots__ = ("__weakref__",)

  def __init__(
      self,
      *,
      query_role: str,
      source_part: str,
      body_uuid: str,
      step: CapturedFileArtifact,
      source_instance_key_sha256: str,
      world_transform_chain_sha256: str,
      instance_transform_sha256: str,
      world_transform_mm: Mapping[str, Any],
      _factory_token: object,
  ) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError("execution input parts are loader-factory-only")
    _PART_STATES[self] = MappingProxyType({
        "query_role": query_role,
        "source_part": source_part,
        "body_uuid": body_uuid,
        "step": step,
        "source_instance_key_sha256": source_instance_key_sha256,
        "world_transform_chain_sha256": world_transform_chain_sha256,
        "instance_transform_sha256": instance_transform_sha256,
        "world_transform_mm": _deep_freeze(world_transform_mm),
    })

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("authenticated execution input parts are sealed")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated execution input parts are not serializable")

  def _state(self) -> Mapping[str, Any]:
    try:
      return _PART_STATES[self]
    except KeyError as error:
      raise TypeError("execution input part lacks factory-sealed state") from error

  @property
  def query_role(self) -> str:
    return str(self._state()["query_role"])

  @property
  def source_part(self) -> str:
    return str(self._state()["source_part"])

  @property
  def body_uuid(self) -> str:
    return str(self._state()["body_uuid"])

  @property
  def step(self) -> CapturedFileArtifact:
    return _copy_capture(self._state()["step"])

  @property
  def source_instance_key_sha256(self) -> str:
    return str(self._state()["source_instance_key_sha256"])

  @property
  def world_transform_chain_sha256(self) -> str:
    return str(self._state()["world_transform_chain_sha256"])

  @property
  def instance_transform_sha256(self) -> str:
    return str(self._state()["instance_transform_sha256"])

  @property
  def world_transform_mm(self) -> Mapping[str, Any]:
    return self._state()["world_transform_mm"]


class AuthenticatedExecutionInputRowV1:
  __slots__ = ("__weakref__",)

  def __init__(self, *, payload: Mapping[str, Any], parts: Sequence[ExecutionInputPartV1], _factory_token: object) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError("execution input rows are loader-factory-only")
    if len(parts) != 2:
      raise ValueError("execution input row requires two parts")
    _ROW_STATES[self] = MappingProxyType({
        "payload": _deep_freeze(payload),
        "parts": tuple(parts),
    })

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("authenticated execution input rows are sealed")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated execution input rows are not serializable")

  def _state(self) -> Mapping[str, Any]:
    try:
      return _ROW_STATES[self]
    except KeyError as error:
      raise TypeError("execution input row lacks factory-sealed state") from error

  @property
  def row_ordinal(self) -> int:
    return int(self._state()["payload"]["row_ordinal"])

  @property
  def safe_input_sha256(self) -> str:
    return str(self._state()["payload"]["safe_input_sha256"])

  @property
  def parts(self) -> tuple[ExecutionInputPartV1, ExecutionInputPartV1]:
    return self._state()["parts"]  # type: ignore[return-value]

  @property
  def binding_sha256(self) -> str:
    return str(self._state()["payload"]["receipt_payload_sha256"])


class AuthenticatedExecutionInputAuthorityV1:
  __slots__ = ("__weakref__",)

  def __init__(self, *, rows: Mapping[int, AuthenticatedExecutionInputRowV1], captures: Sequence[CapturedFileArtifact], artifact_sha256: str, v5_pin: str, _factory_token: object) -> None:
    if _factory_token is not _FACTORY_TOKEN:
      raise TypeError("execution input authorities are loader-factory-only")
    _AUTHORITY_STATES[self] = MappingProxyType({
        "rows": MappingProxyType(dict(rows)),
        "captures": tuple(captures),
        "artifact_sha256": artifact_sha256,
        "v5_bundle_artifact_sha256": v5_pin,
    })

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("authenticated execution input authorities are sealed")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated execution input authorities are not serializable")

  def _state(self) -> Mapping[str, Any]:
    try:
      return _AUTHORITY_STATES[self]
    except KeyError as error:
      raise TypeError("execution input authority lacks factory-sealed state") from error

  def row(self, row_ordinal: int) -> AuthenticatedExecutionInputRowV1:
    self.revalidate()
    try:
      return self._state()["rows"][int(row_ordinal)]
    except KeyError as error:
      raise ValueError("row ordinal is outside execution input authority") from error

  @property
  def row_count(self) -> int:
    return len(self._state()["rows"])

  @property
  def row_ordinals(self) -> tuple[int, ...]:
    """The immutable manifest order used for gold-independent prefix pilots."""

    return tuple(sorted(int(value) for value in self._state()["rows"]))

  @property
  def artifact_sha256(self) -> str:
    return str(self._state()["artifact_sha256"])

  @property
  def v5_bundle_artifact_sha256(self) -> str:
    return str(self._state()["v5_bundle_artifact_sha256"])

  def revalidate(self) -> None:
    for captured in self._state()["captures"]:
      reverify_captured_file_artifact(captured, label="execution input authority")


def load_executor_only_execution_input_authority_v1(
    root: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_v5_bundle_artifact_sha256: str,
    expected_source_cache_artifact_sha256: str,
    expected_source_authority_projection_sha256: str,
    expected_source_authority_domain_sha256: str,
    expected_private_source_domain_sha256: str,
    expected_stage_manifest_domain_sha256: str,
    expected_producer_revision: str,
) -> AuthenticatedExecutionInputAuthorityV1:
  """Load a pinned capability only for post-proposal exact execution."""

  directory = Path(root)
  artifact_capture = capture_file_artifact(
      directory / "artifact.json", label="execution input artifact"
  )
  if artifact_capture.sha256 != expected_artifact_sha256:
    raise ValueError("execution input artifact differs from external pin")
  artifact = _strict_json(artifact_capture.raw_bytes, label="execution input artifact")
  if (
      not isinstance(artifact, Mapping)
      or set(artifact) != {
          "schema_version", "consumer_scope", "payload_policy",
          "producer_revision", "manifest", "row_receipts",
          "final_test_touched", "artifact_payload_sha256",
      }
      or artifact.get("schema_version") != ARTIFACT_SCHEMA
      or artifact.get("consumer_scope") != CONSUMER_SCOPE
      or artifact.get("payload_policy") != PAYLOAD_POLICY
      or artifact.get("producer_revision") != expected_producer_revision
      or artifact.get("final_test_touched") is not False
  ):
    raise ValueError("execution input artifact schema/policy differs")
  unsigned_artifact = dict(artifact)
  observed = unsigned_artifact.pop("artifact_payload_sha256", None)
  if observed != canonical_sha256(unsigned_artifact):
    raise ValueError("execution input artifact hash differs")
  manifest_capture = _capture_binding(directory, artifact.get("manifest"), label="execution input manifest")
  manifest = _strict_json(manifest_capture.raw_bytes, label="execution input manifest")
  if (
      not isinstance(manifest, Mapping)
      or set(manifest) != {
          "schema_version", "consumer_scope", "payload_policy", "scope",
          "producer_revision", "v5_bundle_artifact_sha256",
          "v5_bundle_artifact", "source_cache_artifact_sha256",
          "source_cache_artifact", "source_authority_projection_sha256",
          "source_authority_projection", "upstream_pin_domains",
          "selection_policy", "row_count", "row_domain_sha256", "rows",
          "final_test_touched", "manifest_payload_sha256",
      }
      or manifest.get("schema_version") != MANIFEST_SCHEMA
      or manifest.get("consumer_scope") != CONSUMER_SCOPE
      or manifest.get("payload_policy") != PAYLOAD_POLICY
      or manifest.get("scope")
      != "p0_train_dev_executor_only_post_proposal_no_final"
  ):
    raise ValueError("execution input manifest schema differs")
  unsigned_manifest = dict(manifest)
  observed = unsigned_manifest.pop("manifest_payload_sha256", None)
  if observed != canonical_sha256(unsigned_manifest):
    raise ValueError("execution input manifest hash differs")
  if (
      manifest.get("v5_bundle_artifact_sha256") != expected_v5_bundle_artifact_sha256
      or manifest.get("source_cache_artifact_sha256")
      != expected_source_cache_artifact_sha256
      or manifest.get("source_authority_projection_sha256")
      != expected_source_authority_projection_sha256
      or manifest.get("producer_revision") != expected_producer_revision
      or manifest.get("final_test_touched") is not False
  ):
    raise ValueError("execution input manifest upstream differs")
  v5_capture = _capture_external_file(
      manifest.get("v5_bundle_artifact"), label="execution input V5 bundle"
  )
  if v5_capture.sha256 != expected_v5_bundle_artifact_sha256:
    raise ValueError("execution input V5 bytes changed")
  source_cache_capture = _capture_external_file(
      manifest.get("source_cache_artifact"),
      label="execution input source cache artifact",
  )
  if source_cache_capture.sha256 != expected_source_cache_artifact_sha256:
    raise ValueError("execution input source cache bytes changed")
  source_authority_capture = _capture_external_file(
      manifest.get("source_authority_projection"),
      label="execution input source authority projection",
  )
  if source_authority_capture.sha256 != expected_source_authority_projection_sha256:
    raise ValueError("execution input source authority bytes changed")
  projection = _strict_json(
      source_authority_capture.raw_bytes,
      label="execution input source authority projection",
  )
  projection_records = (
      projection.get("records") if isinstance(projection, Mapping) else None
  )
  if not isinstance(projection_records, Mapping):
    raise ValueError("execution input source authority projection schema differs")
  projection_authority_pins = {
      str(row.get("authority_binding_sha256"))
      for split_rows in projection_records.values()
      for row in (split_rows if isinstance(split_rows, list) else [])
      if isinstance(row, Mapping)
      and _SHA256.fullmatch(str(row.get("authority_binding_sha256"))) is not None
  }
  ledger = manifest.get("rows")
  artifact_receipts = artifact.get("row_receipts")
  if not isinstance(ledger, list) or not isinstance(artifact_receipts, list) or len(ledger) != manifest.get("row_count") or len(ledger) != len(artifact_receipts):
    raise ValueError("execution input row ledger differs")
  rows: dict[int, AuthenticatedExecutionInputRowV1] = {}
  captures = [
      artifact_capture, manifest_capture, v5_capture, source_cache_capture,
      source_authority_capture,
  ]
  artifact_by_name = {row.get("name"): row for row in artifact_receipts if isinstance(row, Mapping)}
  if len(artifact_by_name) != len(artifact_receipts):
    raise ValueError("execution input artifact contains duplicate receipt bindings")
  loaded_receipts: list[Mapping[str, Any]] = []
  for ledger_row in ledger:
    if (
        not isinstance(ledger_row, Mapping)
        or set(ledger_row) != {
            "row_ordinal", "safe_input_sha256", "safe_row_payload_sha256",
            "full_topology_sha256s", "receipt", "receipt_payload_sha256",
        }
    ):
      raise ValueError("execution input ledger row differs")
    receipt_binding = ledger_row.get("receipt")
    receipt_capture = _capture_binding(directory, receipt_binding, label="execution input row receipt")
    if artifact_by_name.get(receipt_binding["name"]) != dict(receipt_binding):
      raise ValueError("execution input artifact/manifest receipt binding differs")
    receipt = _validate_row_receipt(_strict_json(receipt_capture.raw_bytes, label="execution input row receipt"))
    loaded_receipts.append(receipt)
    ordinal = int(receipt["row_ordinal"])
    if (
        ordinal in rows or ledger_row.get("row_ordinal") != ordinal
        or ledger_row.get("safe_input_sha256") != receipt["safe_input_sha256"]
        or ledger_row.get("safe_row_payload_sha256") != receipt["safe_row_payload_sha256"]
        or ledger_row.get("receipt_payload_sha256") != receipt["receipt_payload_sha256"]
        or ledger_row.get("full_topology_sha256s") != [
            graph["full_graph_proof"]["full_topology_sha256"]
            for graph in receipt["graphs"]
        ]
        or receipt["v5_bundle_artifact_sha256"]
        != expected_v5_bundle_artifact_sha256
        or receipt["source_cache_artifact_sha256"]
        != expected_source_cache_artifact_sha256
        or receipt["producer_revision"] != expected_producer_revision
        or receipt["source_authority_sha256"] not in projection_authority_pins
    ):
      raise ValueError("execution input row identity differs")
    private_source_capture = _capture_external_file(
        receipt["private_source_file"], label="execution input private source"
    )
    stage_manifest_capture = _capture_external_file(
        receipt["stage_manifest_file"], label="execution input stage manifest"
    )
    captures.extend((private_source_capture, stage_manifest_capture))
    parts = []
    for graph in receipt["graphs"]:
      query = graph["query_instance"]
      step_binding = query["step_file"]
      step_capture = capture_file_artifact(
          Path(str(step_binding["path"])), label="execution input STEP"
      )
      if step_capture.byte_count != step_binding["bytes"] or step_capture.sha256 != step_binding["sha256"]:
        raise ValueError("execution input STEP bytes changed")
      parts.append(ExecutionInputPartV1(
          query_role=str(query["query_role"]), source_part=str(query["source_part"]),
          body_uuid=str(query["body_uuid"]), step=step_capture,
          source_instance_key_sha256=str(query["source_instance_key_sha256"]),
          world_transform_chain_sha256=str(query["world_transform_chain_sha256"]),
          instance_transform_sha256=str(query["instance_transform_sha256"]),
          world_transform_mm=_deep_freeze(query["world_transform_mm"]),
          _factory_token=_FACTORY_TOKEN,
      ))
      captures.append(step_capture)
    rows[ordinal] = AuthenticatedExecutionInputRowV1(
        payload=receipt, parts=parts, _factory_token=_FACTORY_TOKEN
    )
    captures.append(receipt_capture)
  domains = dict(execution_input_upstream_pin_domains(loaded_receipts))
  expected_domains = {
      "source_authority_domain_sha256": expected_source_authority_domain_sha256,
      "private_source_domain_sha256": expected_private_source_domain_sha256,
      "stage_manifest_domain_sha256": expected_stage_manifest_domain_sha256,
  }
  if (
      manifest.get("upstream_pin_domains") != expected_domains
      or domains != expected_domains
      or manifest.get("row_domain_sha256")
      != canonical_sha256(sorted(rows))
  ):
    raise ValueError("execution input external upstream domains changed")
  authority = AuthenticatedExecutionInputAuthorityV1(
      rows=rows, captures=captures, artifact_sha256=expected_artifact_sha256,
      v5_pin=expected_v5_bundle_artifact_sha256, _factory_token=_FACTORY_TOKEN,
  )
  authority.revalidate()
  return authority
