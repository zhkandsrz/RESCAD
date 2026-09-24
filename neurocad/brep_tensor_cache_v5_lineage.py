"""Lineage-augmented full-STEP cache producer.

V5 emits three sibling artifacts in one rollback-safe publication: the
identity-free V4 model-input cache, its isolated label cache, and a new
source-authority lineage index.  Identity is attached while each case is
replayed, never inferred later from manifest position or duplicate content.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .benchmark_v3_unlabeled_step_view import canonical_bytes, canonical_sha256
from .brep_program_learner_v1 import AuthenticatedShardedBRepTensorCacheV2
from .brep_tensor_cache_v4_unlabeled import (
    SOURCE_CACHE_SCHEMA_VERSION,
    UNLABELED_VIEW_SCHEMA,
    _FullStepGraphCache,
    _binding,
    _case_selected_by_budget_authorities,
    _file_sha256,
    _input_domain_sha256,
    _row_arrays,
    _strict_json,
    _validate_budget_decision_binding,
    _validate_graph_policy,
    _validate_label_row,
    _validate_preflight_binding,
    _validate_safe_row,
    _validate_shards,
    _write_exclusive,
    materialize_unlabeled_training_rows_with_authority_by_case,
)
from .joint_interface_training_v2 import (
    SUITE_V2_ARTIFACT_PIN,
    _load_case_semantic_commitments,
    _strings,
)


LINEAGE_SCHEMA = "brep_tensor_cache_v5_lineage_index.v1"
LINEAGE_ARTIFACT_SCHEMA = "brep_tensor_cache_v5_lineage_artifact.v1"
SAFE_SCHEMA = "brep_tensor_cache_v5_unlabeled.v1"
SAFE_ARTIFACT_SCHEMA = "brep_tensor_cache_v5_unlabeled_artifact.v1"
LABEL_SCHEMA = "brep_tensor_cache_v5_training_labels.v1"
LABEL_ARTIFACT_SCHEMA = "brep_tensor_cache_v5_training_labels_artifact.v1"
AUDIT_SCHEMA = "brep_tensor_cache_v5_unlabeled_audit.v1"
BUNDLE_ARTIFACT_SCHEMA = "brep_tensor_cache_v5_lineage_bundle_artifact.v1"
CHECKPOINT_SCHEMA = "brep_tensor_cache_v5_lineage_checkpoint.v1"
CHECKPOINT_CASE_SCHEMA = "brep_tensor_cache_v5_lineage_checkpoint_case.v1"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_V5_SOURCES = (
    "brep_tensor_cache_v5_lineage.py",
    "brep_tensor_cache_v4_unlabeled.py",
    "benchmark_v2_model_view_v2.py",
    "tools/materialize_lineage_augmented_brep_tensor_cache_v5.py",
)


def _v5_producer_binding(expected_revision: str) -> dict[str, Any]:
  if _REVISION.fullmatch(expected_revision) is None:
    raise ValueError("V5 expected revision is not a full Git object ID")
  root = Path(__file__).resolve().parent
  revision = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=root, text=True
  ).strip()
  if revision != expected_revision:
    raise ValueError("V5 producer revision differs")
  dirty = subprocess.check_output(
      ["git", "status", "--porcelain", "--untracked-files=all"],
      cwd=root, text=True,
  )
  if dirty.strip():
    raise ValueError("V5 producer worktree is not clean")
  sources = {
      name: _file_sha256(root / name) for name in _V5_SOURCES
  }
  return {"revision": revision, "source_sha256s": sources}


def _validate_v5_producer_binding(value: Any) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {"revision", "source_sha256s"}:
    raise ValueError("V5 producer binding schema differs")
  revision = value.get("revision")
  sources = value.get("source_sha256s")
  if _REVISION.fullmatch(str(revision)) is None or not isinstance(sources, Mapping):
    raise ValueError("V5 producer binding differs")
  if set(sources) != set(_V5_SOURCES):
    raise ValueError("V5 producer source domain differs")
  if any(_SHA256.fullmatch(str(sources[name])) is None for name in _V5_SOURCES):
    raise ValueError("V5 producer source hash differs")
  return value


def _capture_pinned_json(
    path: Path, pin: str, *, label: str,
) -> tuple[Mapping[str, Any], bytes]:
  if _SHA256.fullmatch(pin) is None:
    raise ValueError(f"{label} pin differs")
  raw = path.read_bytes()
  if hashlib.sha256(raw).hexdigest() != pin:
    raise ValueError(f"{label} differs from external pin")
  value = _strict_json(raw, label=label)
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} is not an object")
  return value, raw


def _capture_source_manifest(
    source_cache_root: Path, *, source_cache_artifact_filename: str,
    source_cache_artifact_sha256: str,
) -> tuple[Mapping[str, Any], str]:
  artifact_path = source_cache_root / source_cache_artifact_filename
  artifact, artifact_raw = _capture_pinned_json(
      artifact_path, source_cache_artifact_sha256, label="V5 source-cache artifact"
  )
  binding = artifact.get("manifest")
  if not isinstance(binding, Mapping) or set(binding) != {"name", "bytes", "sha256"}:
    raise ValueError("V5 source-cache manifest binding differs")
  manifest_path = source_cache_root / str(binding["name"])
  manifest_raw = manifest_path.read_bytes()
  observed = {
      "name": str(binding["name"]), "bytes": len(manifest_raw),
      "sha256": hashlib.sha256(manifest_raw).hexdigest(),
  }
  if observed != dict(binding):
    raise ValueError("V5 source-cache manifest changed")
  manifest = _strict_json(manifest_raw, label="V5 source-cache manifest")
  if not isinstance(manifest, Mapping) or not isinstance(manifest.get("rows"), list):
    raise ValueError("V5 source-cache manifest differs")
  if artifact_path.read_bytes() != artifact_raw or manifest_path.read_bytes() != manifest_raw:
    raise ValueError("V5 source-cache authority changed during verification")
  return manifest, str(binding["sha256"])


def _write_checkpoint_json(path: Path, payload: Mapping[str, Any]) -> None:
  raw = canonical_bytes(payload)
  temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
  _write_exclusive(temporary, raw)
  try:
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _restore_v5_checkpoint(
    staging: Path, *, expected_metadata: Mapping[str, Any],
    groups: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int,
]:
  metadata_path = staging / "checkpoint.json"
  raw = metadata_path.read_bytes()
  metadata = _strict_json(raw, label="V5 checkpoint metadata")
  if not isinstance(metadata, Mapping) or dict(metadata) != dict(expected_metadata):
    raise ValueError("V5 checkpoint configuration differs")
  case_root = staging / "checkpoint-cases"
  paths = sorted(case_root.glob("case-*.json"))
  expected_names = [f"case-{index:06d}.json" for index in range(len(paths))]
  if [path.name for path in paths] != expected_names:
    raise ValueError("V5 checkpoint case sequence differs")
  safe_rows: list[dict[str, Any]] = []
  label_rows: list[dict[str, Any]] = []
  lineage_rows: list[dict[str, Any]] = []
  safe_shards: list[dict[str, Any]] = []
  label_shards: list[dict[str, Any]] = []
  semantic_ledger: list[dict[str, Any]] = []
  captured: list[tuple[Path, bytes]] = [(metadata_path, raw)]
  expected_case_keys = {
      "schema_version", "case_sequence", "source_split", "source_case_index",
      "source_row_ordinals", "safe_rows", "label_rows", "lineage_rows",
      "safe_shards", "label_shards", "semantic_receipt",
      "case_payload_sha256",
  }
  for sequence, path in enumerate(paths):
    case_raw = path.read_bytes()
    case = _strict_json(case_raw, label=f"V5 checkpoint case {sequence}")
    if not isinstance(case, Mapping) or set(case) != expected_case_keys:
      raise ValueError("V5 checkpoint case schema differs")
    unsigned = dict(case)
    observed = unsigned.pop("case_payload_sha256")
    expected_group = groups[sequence]
    if (
        case.get("schema_version") != CHECKPOINT_CASE_SCHEMA
        or observed != canonical_sha256(unsigned)
        or case.get("case_sequence") != sequence
        or case.get("source_split") != expected_group[0]["split"]
        or case.get("source_case_index") != expected_group[0]["case_index"]
        or case.get("source_row_ordinals")
        != [int(row["row_ordinal"]) for row in expected_group]
    ):
      raise ValueError("V5 checkpoint case authority differs")
    for key, target in (
        ("safe_rows", safe_rows), ("label_rows", label_rows),
        ("lineage_rows", lineage_rows), ("safe_shards", safe_shards),
        ("label_shards", label_shards),
    ):
      values = case.get(key)
      if not isinstance(values, list):
        raise ValueError("V5 checkpoint case rows differ")
      target.extend(dict(value) for value in values)
    semantic = case.get("semantic_receipt")
    if not isinstance(semantic, Mapping):
      raise ValueError("V5 checkpoint semantic receipt differs")
    semantic_ledger.append(dict(semantic))
    captured.append((path, case_raw))
  checked_safe = [
      dict(_validate_safe_row(row, expected_ordinal=index))
      for index, row in enumerate(safe_rows)
  ]
  checked_labels = [
      dict(_validate_label_row(row, expected_ordinal=index))
      for index, row in enumerate(label_rows)
  ]
  checked_lineage = [
      dict(_validate_lineage_row(row, index)) for index, row in enumerate(lineage_rows)
  ]
  if not (len(checked_safe) == len(checked_labels) == len(checked_lineage)):
    raise ValueError("V5 checkpoint row domains differ")
  for lineage, safe, label in zip(
      checked_lineage, checked_safe, checked_labels, strict=True
  ):
    if (
        lineage["safe_input_sha256"] != safe["input_sha256"]
        or lineage["safe_row_payload_sha256"] != safe["row_payload_sha256"]
        or lineage["label_commitment_sha256"] != label["label_commitment_sha256"]
        or lineage["label_row_payload_sha256"] != label["row_payload_sha256"]
    ):
      raise ValueError("V5 checkpoint lineage/child binding differs")
  if checked_safe:
    _validate_shards(safe_shards, rows=checked_safe, label="V5 checkpoint safe shards")
    _validate_shards(
        label_shards, rows=checked_labels, label="V5 checkpoint label shards"
    )
  referenced = {"safe": set(), "labels": set()}
  for directory, shards in (("safe", safe_shards), ("labels", label_shards)):
    for shard in shards:
      name = str(shard["name"])
      path = staging / directory / name
      if _binding(path, name=name) != {
          key: shard[key] for key in ("name", "bytes", "sha256")
      }:
        raise ValueError("V5 checkpoint shard bytes changed")
      referenced[directory].add(name)
  for directory, prefix in (("safe", "input-shard-"), ("labels", "label-shard-")):
    for path in (staging / directory).glob(f"{prefix}*.npz"):
      if path.name not in referenced[directory]:
        path.unlink()
  for path, expected_raw in captured:
    if path.read_bytes() != expected_raw:
      raise ValueError("V5 checkpoint changed during verification")
  return (
      checked_safe, checked_labels, checked_lineage, safe_shards, label_shards,
      semantic_ledger, len(paths),
  )


def _identity_hashes(
    *, case_id: str, family_id: str, body_sha1s: Sequence[str],
    graph_identity_sha256: str, source_lineages: Sequence[str],
) -> dict[str, str]:
  return {
      "case_id_sha256": canonical_sha256(case_id),
      "family_id_sha256": canonical_sha256(family_id),
      "body_sha1_set_sha256": canonical_sha256(sorted(body_sha1s)),
      "graph_identity_sha256": graph_identity_sha256,
      "source_lineage_set_sha256": canonical_sha256(sorted(source_lineages)),
  }


def build_lineage_row_v1(
    *, row_ordinal: int, source_row: Mapping[str, Any],
    assignment_split: str, projection_record: Mapping[str, Any],
    assignment_record: Mapping[str, Any], pool_record: Mapping[str, Any],
    semantic_commitment: Mapping[str, Any], capability_authority: Mapping[str, Any],
    safe_row: Mapping[str, Any], label_row: Mapping[str, Any],
    label_target_program_index: int, label_residual_sha256: str,
) -> dict[str, Any]:
  """Build one strict producer-time identity binding."""

  authority_record = projection_record.get("authority_record")
  family_identity = pool_record.get("source_family_identity")
  provenance = family_identity.get("family_provenance") if isinstance(
      family_identity, Mapping
  ) else None
  if not isinstance(authority_record, Mapping) or not isinstance(provenance, Mapping):
    raise ValueError("V5 lineage authority provenance differs")
  semantic = capability_authority.get("semantic_authority")
  contact = capability_authority.get("contact_authority")
  catalog = capability_authority.get("program_catalog")
  if not all(isinstance(value, Mapping) for value in (semantic, contact, catalog)):
    raise ValueError("V5 graph capability authority differs")
  program_index = int(source_row.get("program_index"))
  if (
      semantic.get("program_index") != program_index
      or semantic.get("program_id") != semantic_commitment.get("program_id")
      or semantic.get("row_sha256") != semantic_commitment.get("row_sha256")
      or semantic.get("residual_sha256") != semantic_commitment.get("residual_sha256")
      or contact.get("source_contact_ordinal")
      != semantic_commitment.get("source_contact_ordinal")
      or catalog.get("target_program_index") != label_target_program_index
  ):
    raise ValueError("V5 semantic/contact/catalog commitment differs")
  target_catalog_program_id = str(catalog.get("target_program_id") or "")
  if not target_catalog_program_id.startswith("catalog_"):
    raise ValueError("V5 target catalog program identity differs")
  case_id = str(authority_record.get("case_id"))
  source_ordinal = int(authority_record.get("source_ordinal"))
  if assignment_record.get("authority_record_payload_sha256") != pool_record.get(
      "merged_record_payload_sha256"
  ):
    raise ValueError("V5 p0/pool authority commitment differs")
  body_sha1s = _strings(
      provenance.get("body_sha1_set"), label="V5 body SHA1", sha1=True
  )
  source_lineages = _strings(
      provenance.get("source_lineage"), label="V5 source lineage"
  )
  graph_identity_sha256 = canonical_sha256(provenance.get("assembly_graph"))
  family_id = str(assignment_record.get("leakage_component_id"))
  identity_hashes = _identity_hashes(
      case_id=case_id, family_id=family_id, body_sha1s=body_sha1s,
      graph_identity_sha256=graph_identity_sha256,
      source_lineages=source_lineages,
  )
  if _SHA256.fullmatch(label_residual_sha256) is None:
    raise ValueError("V5 label residual commitment differs")
  row = {
      "row_ordinal": row_ordinal,
      "source_row_ordinal": int(source_row["row_ordinal"]),
      "source_row_payload_sha256": str(source_row["row_payload_sha256"]),
      "source_lineage_payload_sha256": str(
          source_row["lineage"]["lineage_payload_sha256"]
      ),
      "source_model_view_sha256": str(source_row["model_view_sha256"]),
      "source_split": str(source_row["split"]),
      "source_case_index": int(source_row["case_index"]),
      "source_program_index": program_index,
      "assignment_split": assignment_split,
      "case_id": case_id,
      "source_ordinal": source_ordinal,
      "source_contact_ordinal": int(semantic_commitment["source_contact_ordinal"]),
      "authority_record_payload_sha256": str(
          assignment_record["authority_record_payload_sha256"]
      ),
      "projection_record_payload_sha256": str(
          projection_record["projection_record_payload_sha256"]
      ),
      "case_authority_binding_sha256": str(
          capability_authority["case_authority_binding_sha256"]
      ),
      "graph_capability_receipt_payload_sha256": str(
          capability_authority["receipt_payload_sha256"]
      ),
      "semantic_program_id": str(semantic_commitment["program_id"]),
      "semantic_row_sha256": str(semantic_commitment["row_sha256"]),
      "semantic_residual_sha256": str(semantic_commitment["residual_sha256"]),
      "label_residual_sha256": label_residual_sha256,
      "residual_authority_replay_sha256": canonical_sha256({
          "semantic_residual_sha256": semantic_commitment["residual_sha256"],
          "label_residual_sha256": label_residual_sha256,
          "source_model_view_sha256": source_row["model_view_sha256"],
          "graph_capability_receipt_payload_sha256": capability_authority[
              "receipt_payload_sha256"
          ],
      }),
      "target_program_index": label_target_program_index,
      "target_catalog_program_id": target_catalog_program_id,
      "program_catalog_payload_sha256": str(catalog["catalog_payload_sha256"]),
      "program_mapping_sha256": canonical_sha256({
          "semantic_program_id": semantic_commitment["program_id"],
          "target_program_index": label_target_program_index,
          "target_catalog_program_id": target_catalog_program_id,
          "program_catalog_payload_sha256": catalog["catalog_payload_sha256"],
      }),
      "safe_input_sha256": str(safe_row["input_sha256"]),
      "safe_row_payload_sha256": str(safe_row["row_payload_sha256"]),
      "label_commitment_sha256": str(label_row["label_commitment_sha256"]),
      "label_row_payload_sha256": str(label_row["row_payload_sha256"]),
      "family_id": family_id,
      "family_authority_sha256": str(
          family_identity["family_provenance_payload_sha256"]
      ),
      "body_sha1s": list(body_sha1s),
      "graph_identity_sha256": graph_identity_sha256,
      "source_lineages": list(source_lineages),
      "identity_hashes": identity_hashes,
  }
  row["row_payload_sha256"] = canonical_sha256(row)
  return row


def _validate_lineage_row(value: Any, expected_ordinal: int) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or value.get("row_ordinal") != expected_ordinal:
    raise ValueError("V5 lineage row ordinal differs")
  expected_keys = {
      "row_ordinal", "source_row_ordinal", "source_row_payload_sha256",
      "source_lineage_payload_sha256", "source_model_view_sha256", "source_split",
      "source_case_index", "source_program_index", "assignment_split", "case_id",
      "source_ordinal", "source_contact_ordinal", "authority_record_payload_sha256",
      "projection_record_payload_sha256", "case_authority_binding_sha256",
      "graph_capability_receipt_payload_sha256", "semantic_program_id",
      "semantic_row_sha256", "semantic_residual_sha256", "label_residual_sha256",
      "residual_authority_replay_sha256", "target_program_index",
      "target_catalog_program_id", "program_catalog_payload_sha256",
      "program_mapping_sha256", "safe_input_sha256", "safe_row_payload_sha256",
      "label_commitment_sha256", "label_row_payload_sha256", "family_id",
      "family_authority_sha256", "body_sha1s", "graph_identity_sha256",
      "source_lineages", "identity_hashes", "row_payload_sha256",
  }
  if set(value) != expected_keys:
    raise ValueError("V5 lineage row schema differs")
  unsigned = dict(value)
  observed = unsigned.pop("row_payload_sha256")
  if observed != canonical_sha256(unsigned):
    raise ValueError("V5 lineage row self hash differs")
  if value["source_split"] not in {"train", "dev"} or value["assignment_split"] not in {
      "train", "dev"
  }:
    raise ValueError("V5 lineage split differs")
  hashes = value["identity_hashes"]
  if not isinstance(hashes, Mapping) or hashes != _identity_hashes(
      case_id=str(value["case_id"]), family_id=str(value["family_id"]),
      body_sha1s=value["body_sha1s"],
      graph_identity_sha256=str(value["graph_identity_sha256"]),
      source_lineages=value["source_lineages"],
  ):
    raise ValueError("V5 five-dimensional identity hash differs")
  for key, item in value.items():
    if key.endswith("sha256") and _SHA256.fullmatch(str(item)) is None:
      raise ValueError(f"V5 lineage {key} differs")
  if any(_SHA1.fullmatch(str(item)) is None for item in value["body_sha1s"]):
    raise ValueError("V5 lineage body SHA1 differs")
  return value


def load_lineage_index_v1(
    bundle_root: str | Path, *, expected_artifact_sha256: str,
) -> Mapping[str, Any]:
  root = Path(bundle_root)
  artifact_path = root / "artifact.json"
  artifact_raw = artifact_path.read_bytes()
  if hashlib.sha256(artifact_raw).hexdigest() != expected_artifact_sha256:
    raise ValueError("V5 bundle artifact pin differs")
  artifact = _strict_json(artifact_raw, label="V5 bundle artifact")
  unsigned_bundle = dict(artifact)
  observed_bundle = unsigned_bundle.pop("artifact_payload_sha256", None)
  if (
      artifact.get("schema_version") != BUNDLE_ARTIFACT_SCHEMA
      or observed_bundle != canonical_sha256(unsigned_bundle)
  ):
    raise ValueError("V5 bundle artifact authority differs")
  bundle_producer = _validate_v5_producer_binding(artifact.get("producer_binding"))

  def capture(
      binding: Any, label: str, *, expected_name: str,
  ) -> tuple[Path, bytes, Mapping[str, Any]]:
    if not isinstance(binding, Mapping) or set(binding) != {"name", "bytes", "sha256"}:
      raise ValueError(f"{label} binding differs")
    if binding.get("name") != expected_name:
      raise ValueError(f"{label} path differs")
    path = root / expected_name
    raw = path.read_bytes()
    observed = {
        "name": str(binding["name"]), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if observed != dict(binding):
      raise ValueError(f"{label} changed")
    return path, raw, _strict_json(raw, label=label)

  safe_artifact_path, safe_artifact_raw, safe_artifact = capture(
      artifact.get("safe_artifact"), "V5 safe artifact",
      expected_name="safe/artifact.json",
  )
  label_artifact_path, label_artifact_raw, label_artifact = capture(
      artifact.get("label_artifact"), "V5 label artifact",
      expected_name="labels/artifact.json",
  )
  lineage_artifact_path, lineage_artifact_raw, lineage_artifact = capture(
      artifact.get("lineage_artifact"), "V5 lineage artifact",
      expected_name="lineage/artifact.json",
  )
  for child, schema, label in (
      (safe_artifact, SAFE_ARTIFACT_SCHEMA, "safe"),
      (label_artifact, LABEL_ARTIFACT_SCHEMA, "label"),
      (lineage_artifact, LINEAGE_ARTIFACT_SCHEMA, "lineage"),
  ):
    unsigned_child = dict(child)
    observed_child = unsigned_child.pop("artifact_payload_sha256", None)
    if child.get("schema_version") != schema or observed_child != canonical_sha256(unsigned_child):
      raise ValueError(f"V5 {label} artifact authority differs")
    if dict(_validate_v5_producer_binding(child.get("producer_binding"))) != dict(
        bundle_producer
    ):
      raise ValueError(f"V5 {label} producer binding differs")

  def child_manifest(
      child: Mapping[str, Any], child_path: Path, label: str,
  ) -> tuple[Path, bytes, Mapping[str, Any]]:
    binding = child.get("manifest")
    if not isinstance(binding, Mapping) or binding.get("name") != "manifest.json":
      raise ValueError(f"V5 {label} manifest binding differs")
    relative = (child_path.parent / "manifest.json").relative_to(root)
    root_binding = {**dict(binding), "name": relative.as_posix()}
    return capture(
        root_binding, f"V5 {label} manifest", expected_name=relative.as_posix()
    )

  safe_manifest_path, safe_manifest_raw, safe_manifest = child_manifest(
      safe_artifact, safe_artifact_path, "safe"
  )
  label_manifest_path, label_manifest_raw, label_manifest = child_manifest(
      label_artifact, label_artifact_path, "label"
  )
  manifest_path, manifest_raw, manifest = child_manifest(
      lineage_artifact, lineage_artifact_path, "lineage"
  )
  unsigned = dict(manifest)
  observed = unsigned.pop("manifest_payload_sha256", None)
  if manifest.get("schema_version") != LINEAGE_SCHEMA or observed != canonical_sha256(unsigned):
    raise ValueError("V5 lineage manifest self hash differs")
  if dict(_validate_v5_producer_binding(manifest.get("producer_binding"))) != dict(
      bundle_producer
  ):
    raise ValueError("V5 lineage manifest producer binding differs")
  for child_manifest_value, schema, label in (
      (safe_manifest, SAFE_SCHEMA, "safe"),
      (label_manifest, LABEL_SCHEMA, "label"),
  ):
    unsigned_child_manifest = dict(child_manifest_value)
    observed_child_manifest = unsigned_child_manifest.pop("manifest_payload_sha256", None)
    if (
        child_manifest_value.get("schema_version") != schema
        or observed_child_manifest != canonical_sha256(unsigned_child_manifest)
    ):
      raise ValueError(f"V5 {label} manifest authority differs")
    if dict(_validate_v5_producer_binding(
        child_manifest_value.get("producer_binding")
    )) != dict(bundle_producer):
      raise ValueError(f"V5 {label} manifest producer binding differs")
  safe_rows = safe_manifest.get("rows")
  label_rows = label_manifest.get("rows")
  if not isinstance(safe_rows, list) or not isinstance(label_rows, list):
    raise ValueError("V5 child rows differ")
  checked_safe = tuple(_validate_safe_row(row, expected_ordinal=index) for index, row in enumerate(safe_rows))
  checked_labels = tuple(
      _validate_label_row(row, expected_ordinal=index) for index, row in enumerate(label_rows)
  )
  _validate_graph_policy(safe_manifest.get("graph_policy"))
  _validate_shards(safe_manifest.get("shards"), rows=checked_safe, label="V5 safe shards")
  _validate_shards(label_manifest.get("shards"), rows=checked_labels, label="V5 label shards")
  safe_pin = hashlib.sha256(safe_artifact_raw).hexdigest()
  label_pin = hashlib.sha256(label_artifact_raw).hexdigest()
  lineage_pin = hashlib.sha256(lineage_artifact_raw).hexdigest()
  if (
      artifact.get("safe_artifact_sha256") != safe_pin
      or artifact.get("label_artifact_sha256") != label_pin
      or artifact.get("lineage_artifact_sha256") != lineage_pin
      or manifest.get("safe_artifact_sha256") != safe_pin
      or manifest.get("label_artifact_sha256") != label_pin
      or lineage_artifact.get("safe_artifact_sha256") != safe_pin
      or lineage_artifact.get("label_artifact_sha256") != label_pin
      or safe_manifest.get("input_domain_sha256") != label_manifest.get("input_domain_sha256")
      or manifest.get("input_domain_sha256") != safe_manifest.get("input_domain_sha256")
  ):
    raise ValueError("V5 lineage child artifact binding differs")
  rows = manifest.get("rows")
  if not isinstance(rows, list) or manifest.get("row_count") != len(rows):
    raise ValueError("V5 lineage row count differs")
  checked = tuple(_validate_lineage_row(row, index) for index, row in enumerate(rows))
  if len(checked) != len(checked_safe) or len(checked) != len(checked_labels):
    raise ValueError("V5 lineage/child row domain differs")
  for lineage, safe, label in zip(checked, checked_safe, checked_labels, strict=True):
    if (
        lineage["safe_input_sha256"] != safe["input_sha256"]
        or lineage["safe_row_payload_sha256"] != safe["row_payload_sha256"]
        or lineage["label_commitment_sha256"] != label["label_commitment_sha256"]
        or lineage["label_row_payload_sha256"] != label["row_payload_sha256"]
        or safe["input_sha256"] != label["input_sha256"]
    ):
      raise ValueError("V5 lineage row differs from child manifests")
  if len({int(row["source_row_ordinal"]) for row in checked}) != len(checked):
    raise ValueError("V5 source row identity is duplicated")
  split_rows = {
      split: tuple(row for row in checked if row["assignment_split"] == split)
      for split in ("train", "dev")
  }
  dimensions = {
      "case": lambda row: row["identity_hashes"]["case_id_sha256"],
      "family": lambda row: row["identity_hashes"]["family_id_sha256"],
      "body_sha1": lambda row: tuple(row["body_sha1s"]),
      "graph": lambda row: row["identity_hashes"]["graph_identity_sha256"],
      "source_lineage": lambda row: tuple(row["source_lineages"]),
      "safe_input": lambda row: row["safe_input_sha256"],
  }
  for label, getter in dimensions.items():
    train_values = {item for row in split_rows["train"] for item in (
        getter(row) if label in {"body_sha1", "source_lineage"} else (getter(row),)
    )}
    dev_values = {item for row in split_rows["dev"] for item in (
        getter(row) if label in {"body_sha1", "source_lineage"} else (getter(row),)
    )}
    if train_values & dev_values:
      raise ValueError(f"V5 train/dev {label} overlap")
  for path, raw in (
      (artifact_path, artifact_raw), (safe_artifact_path, safe_artifact_raw),
      (label_artifact_path, label_artifact_raw),
      (lineage_artifact_path, lineage_artifact_raw),
      (safe_manifest_path, safe_manifest_raw), (label_manifest_path, label_manifest_raw),
      (manifest_path, manifest_raw),
  ):
    if path.read_bytes() != raw:
      raise ValueError("V5 captured authority changed during verification")
  return manifest


def _load_authority_tables(
    *, projection_path: Path, projection_artifact_sha256: str,
    suite_v2_root: Path, suite_v2_artifact_sha256: str,
    authority_pool_path: Path, authority_pool_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], dict[Any, Any], dict[Any, Any]]:
  projection, projection_raw = _capture_pinned_json(
      projection_path, projection_artifact_sha256, label="V5 development projection"
  )
  if projection.get("development") is not True:
    raise ValueError("V5 projection is not development-only")
  suite_artifact_path = suite_v2_root / "artifact.json"
  artifact, artifact_raw = _capture_pinned_json(
      suite_artifact_path, suite_v2_artifact_sha256,
      label="V5 family partition suite artifact",
  )
  binding = next((
      item for item in artifact.get("files", [])
      if isinstance(item, Mapping) and item.get("name") == "family_partition_suite.json"
  ), None)
  suite_path = suite_v2_root / "family_partition_suite.json"
  suite_raw = suite_path.read_bytes()
  observed_suite_binding = {
      "name": suite_path.name, "bytes": len(suite_raw),
      "sha256": hashlib.sha256(suite_raw).hexdigest(),
  }
  if binding is None or observed_suite_binding != dict(binding):
    raise ValueError("V5 family partition suite changed")
  suite = _strict_json(suite_raw, label="V5 family partition suite")
  if not isinstance(suite, Mapping):
    raise ValueError("V5 family partition suite differs")
  p0 = next((
      part for part in suite.get("partitions", [])
      if isinstance(part, Mapping) and part.get("partition_index") == 0
  ), None)
  if p0 is None:
    raise ValueError("V5 p0 family partition is missing")
  assignments: dict[tuple[str, int], tuple[str, Mapping[str, Any]]] = {}
  for split in ("train", "dev"):
    for record in p0.get(f"{split}_records", []):
      key = (str(record.get("case_id")), int(record.get("source_ordinal")))
      if key in assignments:
        raise ValueError("V5 p0 assignment is duplicated")
      assignments[key] = (split, record)
  pool, pool_raw = _capture_pinned_json(
      authority_pool_path, authority_pool_sha256, label="V5 authority-ready pool"
  )
  pool_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
  for record in pool.get("authority_ready_records", []):
    key = (str(record.get("case_id")), int(record.get("source_ordinal")))
    if key in pool_by_key:
      raise ValueError("V5 authority-ready identity is duplicated")
    pool_by_key[key] = record
  for path, raw in (
      (projection_path, projection_raw), (suite_artifact_path, artifact_raw),
      (suite_path, suite_raw), (authority_pool_path, pool_raw),
  ):
    if path.read_bytes() != raw:
      raise ValueError("V5 authority changed during verification")
  return projection, suite, p0, assignments, pool_by_key


def publish_lineage_augmented_tensor_cache_v5(
    output_directory: str | Path, *, index: Any,
    source_cache: AuthenticatedShardedBRepTensorCacheV2,
    source_cache_root: str | Path, source_cache_artifact_filename: str,
    source_cache_artifact_sha256: str,
    preflight: Any, budget_decision: Any, expected_revision: str,
    projection_path: str | Path, projection_artifact_sha256: str,
    suite_v2_root: str | Path,
    suite_v2_artifact_sha256: str = SUITE_V2_ARTIFACT_PIN,
    authority_pool_path: str | Path = "", authority_pool_sha256: str = "",
    case_limit: int | None = None, shard_row_limit: int = 25,
    case_selection_policy: str = "all_decision_admitted_cases",
    resume: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[str, str, str, str]:
  """Produce V5 identity at replay time; never consume an old V4 manifest."""

  if type(source_cache) is not AuthenticatedShardedBRepTensorCacheV2:
    raise TypeError("V5 requires an authenticated v3 source cache")
  if case_limit is not None and (type(case_limit) is not int or case_limit < 1):
    raise ValueError("V5 case_limit must be positive or omitted")
  if type(shard_row_limit) is not int or shard_row_limit < 1:
    raise ValueError("V5 shard_row_limit must be positive")
  from .brep_full_graph_budget_preflight import AuthenticatedFullGraphBudgetPreflight
  from .brep_graph_budget_decision_v2 import AuthenticatedGraphBudgetDecisionV2
  if type(preflight) is not AuthenticatedFullGraphBudgetPreflight:
    raise TypeError("V5 requires authenticated preflight")
  if type(budget_decision) is not AuthenticatedGraphBudgetDecisionV2:
    raise TypeError("V5 requires authenticated graph decision")
  preflight_binding = _validate_preflight_binding(preflight.binding)
  decision_binding = _validate_budget_decision_binding(budget_decision.binding)
  if dict(budget_decision.preflight_binding) != dict(preflight_binding):
    raise ValueError("V5 graph decision/preflight domain differs")
  # Clean revision is checked before any output directory or STEP is opened.
  v5_binding = _v5_producer_binding(expected_revision)
  projection, suite, p0, assignments, pool_by_key = _load_authority_tables(
      projection_path=Path(projection_path),
      projection_artifact_sha256=projection_artifact_sha256,
      suite_v2_root=Path(suite_v2_root),
      suite_v2_artifact_sha256=suite_v2_artifact_sha256,
      authority_pool_path=Path(authority_pool_path),
      authority_pool_sha256=authority_pool_sha256,
  )
  development = projection.get("records")
  source_manifest, source_manifest_sha256 = _capture_source_manifest(
      Path(source_cache_root),
      source_cache_artifact_filename=source_cache_artifact_filename,
      source_cache_artifact_sha256=source_cache_artifact_sha256,
  )
  selected: list[Mapping[str, Any]] = []
  for source_row in source_manifest["rows"]:
    split = str(source_row["split"])
    case_index = int(source_row["case_index"])
    if _case_selected_by_budget_authorities(
        preflight=preflight, budget_decision=budget_decision,
        case_selection_policy=case_selection_policy,
        split=split, case_index=case_index,
    ):
      selected.append(source_row)
  groups: list[list[Mapping[str, Any]]] = []
  for source_row in selected:
    key = (source_row["split"], source_row["case_index"])
    if not groups or (groups[-1][0]["split"], groups[-1][0]["case_index"]) != key:
      if case_limit is not None and len(groups) == case_limit:
        break
      groups.append([])
    groups[-1].append(source_row)
  if not groups:
    raise ValueError("V5 selected case domain is empty")

  output = Path(output_directory)
  if output.exists():
    raise FileExistsError("V5 bundle output already exists")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = output.parent / f".{output.name}.checkpoint"
  checkpoint_metadata: dict[str, Any] = {
      "schema_version": CHECKPOINT_SCHEMA,
      "producer_binding": v5_binding,
      "output_name": output.name,
      "source_cache_artifact_sha256": source_cache_artifact_sha256,
      "source_cache_manifest_sha256": source_manifest_sha256,
      "development_projection_artifact_sha256": projection_artifact_sha256,
      "family_partition_suite_v2_artifact_sha256": suite_v2_artifact_sha256,
      "authority_ready_pool_sha256": authority_pool_sha256,
      "full_graph_budget_preflight": dict(preflight_binding),
      "graph_budget_decision": dict(decision_binding),
      "case_selection_policy": case_selection_policy,
      "case_limit": case_limit,
      "shard_row_limit": shard_row_limit,
      "selected_case_domain_sha256": canonical_sha256([
          {
              "split": group[0]["split"], "case_index": group[0]["case_index"],
              "source_row_ordinals": [int(row["row_ordinal"]) for row in group],
          }
          for group in groups
      ]),
      "case_count": len(groups),
  }
  checkpoint_metadata["checkpoint_payload_sha256"] = canonical_sha256(
      checkpoint_metadata
  )
  if staging.exists():
    if not resume:
      raise FileExistsError(
          "V5 checkpoint already exists; pass resume=True after verifying its inputs"
      )
  elif resume:
    raise FileNotFoundError("V5 checkpoint does not exist")
  else:
    staging.mkdir()
  safe_root = staging / "safe"
  label_root = staging / "labels"
  lineage_root = staging / "lineage"
  checkpoint_case_root = staging / "checkpoint-cases"
  if not resume:
    safe_root.mkdir()
    label_root.mkdir()
    lineage_root.mkdir()
    checkpoint_case_root.mkdir()
    _write_checkpoint_json(staging / "checkpoint.json", checkpoint_metadata)
    safe_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    safe_shards: list[dict[str, Any]] = []
    label_shards: list[dict[str, Any]] = []
    semantic_receipt_ledger: list[dict[str, Any]] = []
    start_sequence = 0
  else:
    (
        safe_rows, label_rows, lineage_rows, safe_shards, label_shards,
        semantic_receipt_ledger, start_sequence,
    ) = _restore_v5_checkpoint(
        staging, expected_metadata=checkpoint_metadata, groups=groups
    )
    for stale in (
        safe_root / "manifest.json", safe_root / "artifact.json",
        label_root / "manifest.json", label_root / "audit.json",
        label_root / "artifact.json", lineage_root / "manifest.json",
        lineage_root / "artifact.json", staging / "artifact.json",
    ):
      stale.unlink(missing_ok=True)
    progress_callback and progress_callback({
        "stage": "checkpoint_restored", "completed_cases": start_sequence,
        "total_cases": len(groups), "completed_rows": len(safe_rows),
    })
  pending: list[tuple[Any, dict[str, Any]]] = []

  def progress(stage: str, **values: Any) -> None:
    if progress_callback is not None:
      progress_callback({"stage": stage, **values})

  def flush() -> None:
    if not pending:
      return
    shard_index = len(safe_shards)
    start = len(safe_rows)
    safe_arrays: dict[str, np.ndarray] = {}
    label_arrays: dict[str, np.ndarray] = {}
    shard_safe: list[dict[str, Any]] = []
    shard_label: list[dict[str, Any]] = []
    shard_lineage: list[dict[str, Any]] = []
    for offset, (training_row, lineage_base) in enumerate(pending):
      ordinal = start + offset
      safe, gold, safe_row, label_row = _row_arrays(training_row, ordinal=ordinal)
      safe_arrays.update(safe)
      label_arrays.update(gold)
      shard_safe.append(safe_row)
      shard_label.append(label_row)
      shard_lineage.append(build_lineage_row_v1(
          row_ordinal=ordinal, safe_row=safe_row, label_row=label_row,
          label_target_program_index=training_row.training_label.target_program_index,
          label_residual_sha256=canonical_sha256({
              "residual_translation": training_row.training_label.residual_translation.tolist(),
              "residual_rotation_vector": (
                  training_row.training_label.residual_rotation_vector.tolist()
              ),
              "residual_mask": training_row.training_label.residual_mask,
          }),
          **lineage_base,
      ))
    for arrays, root, prefix, rows, ledger in (
        (safe_arrays, safe_root, "input-shard", shard_safe, safe_shards),
        (label_arrays, label_root, "label-shard", shard_label, label_shards),
    ):
      stream = io.BytesIO()
      np.savez_compressed(stream, **arrays)
      raw = stream.getvalue()
      digest = hashlib.sha256(raw).hexdigest()
      name = f"{prefix}-{shard_index:06d}-{digest[:16]}.npz"
      _write_exclusive(root / name, raw)
      ledger.append({
          "shard_index": shard_index, "name": name, "bytes": len(raw),
          "sha256": digest, "row_start": start, "row_count": len(rows),
          "rows_payload_sha256": canonical_sha256(
              [item["row_payload_sha256"] for item in rows]
          ),
      })
    safe_rows.extend(shard_safe)
    label_rows.extend(shard_label)
    lineage_rows.extend(shard_lineage)
    pending.clear()

  try:
    sparse_loader = getattr(source_cache, "iter_examples_by_row_ordinals", None)
    if not callable(sparse_loader):
      raise ValueError("V5 source cache lacks sparse authenticated loading")
    started = time.perf_counter()
    for sequence, group in enumerate(groups[start_sequence:], start=start_sequence):
      case_row_start = len(safe_rows)
      case_safe_shard_start = len(safe_shards)
      case_label_shard_start = len(label_shards)
      split = str(group[0]["split"])
      case_index = int(group[0]["case_index"])
      projection_record = development[split][case_index]
      authority_record = projection_record["authority_record"]
      case_id = str(authority_record["case_id"])
      source_ordinal = int(authority_record["source_ordinal"])
      assignment = assignments.get((case_id, source_ordinal))
      pool_record = pool_by_key.get((case_id, source_ordinal))
      if assignment is None or pool_record is None:
        raise ValueError("V5 case lacks p0/pool authority")
      assignment_split, assignment_record = assignment
      cached_rows = tuple(sparse_loader(
          row_ordinals=tuple(int(row["row_ordinal"]) for row in group)
      ))
      if len(cached_rows) != len(group):
        raise ValueError("V5 source cache returned incomplete case")
      for cached, source_row in zip(cached_rows, group, strict=True):
        if dict(cached.lineage) != source_row["lineage"]:
          raise ValueError("V5 source lineage replay differs")
      commitments, semantic_binding = _load_case_semantic_commitments(
          projection_record, expected_case_id=case_id,
          expected_source_ordinal=source_ordinal,
      )
      graph_cache = _FullStepGraphCache(budget_decision=budget_decision)
      replayed = materialize_unlabeled_training_rows_with_authority_by_case(
          index=index, cached_rows=cached_rows, full_graph_cache=graph_cache,
      )
      for cached, source_row, (training_row, capability_authority) in zip(
          cached_rows, group, replayed, strict=True
      ):
        program_index = int(source_row["program_index"])
        if program_index >= len(commitments):
          raise ValueError("V5 semantic program index differs")
        label = training_row.training_label
        if (
            cached.example.target_program_index != label.target_program_index
            or cached.example.residual_mask != label.residual_mask
            or not np.array_equal(
                cached.example.residual_translation.detach().cpu().numpy(),
                label.residual_translation,
            )
            or not np.array_equal(
                cached.example.residual_rotation_vector.detach().cpu().numpy(),
                label.residual_rotation_vector,
            )
        ):
          raise ValueError("V5 label differs from authenticated source example")
        pending.append((training_row, {
            "source_row": source_row,
            "assignment_split": assignment_split,
            "projection_record": projection_record,
            "assignment_record": assignment_record,
            "pool_record": pool_record,
            "semantic_commitment": commitments[program_index],
            "capability_authority": capability_authority,
        }))
        if len(pending) == shard_row_limit:
          flush()
      flush()
      semantic_receipt = {
          "case_id": case_id, "source_ordinal": source_ordinal,
          **dict(semantic_binding),
      }
      semantic_receipt_ledger.append(semantic_receipt)
      case_checkpoint = {
          "schema_version": CHECKPOINT_CASE_SCHEMA,
          "case_sequence": sequence,
          "source_split": split, "source_case_index": case_index,
          "source_row_ordinals": [int(row["row_ordinal"]) for row in group],
          "safe_rows": safe_rows[case_row_start:],
          "label_rows": label_rows[case_row_start:],
          "lineage_rows": lineage_rows[case_row_start:],
          "safe_shards": safe_shards[case_safe_shard_start:],
          "label_shards": label_shards[case_label_shard_start:],
          "semantic_receipt": semantic_receipt,
      }
      case_checkpoint["case_payload_sha256"] = canonical_sha256(case_checkpoint)
      _write_checkpoint_json(
          checkpoint_case_root / f"case-{sequence:06d}.json", case_checkpoint
      )
      elapsed = time.perf_counter() - started
      progress(
          "case_replayed", completed_cases=sequence + 1, total_cases=len(groups),
          completed_rows=len(safe_rows) + len(pending), case_rows=len(group),
          elapsed_seconds=elapsed,
          estimated_total_seconds=elapsed / (sequence + 1) * len(groups),
      )
    flush()
    input_hashes = [row["input_sha256"] for row in safe_rows]
    input_domain = _input_domain_sha256(
        input_hashes, preflight_binding=preflight_binding,
        decision_binding=decision_binding,
        case_selection_policy=case_selection_policy,
    )
    label_manifest = {
        "schema_version": LABEL_SCHEMA,
        "scope": "factory_only_training_supervision",
        "source_cache_artifact_sha256": source_cache.artifact_sha256,
        "full_graph_budget_preflight": dict(preflight_binding),
        "graph_budget_decision": dict(decision_binding),
        "case_selection_policy": case_selection_policy,
        "case_count": len(groups), "producer_binding": v5_binding,
        "row_count": len(label_rows), "input_domain_sha256": input_domain,
        "label_commitment_ledger_sha256": canonical_sha256(
            [row["label_commitment_sha256"] for row in label_rows]
        ),
        "rows": label_rows, "shards": label_shards,
    }
    label_manifest["manifest_payload_sha256"] = canonical_sha256(label_manifest)
    label_manifest_path = label_root / "manifest.json"
    _write_exclusive(label_manifest_path, canonical_bytes(label_manifest))
    safe_manifest = {
        "schema_version": SAFE_SCHEMA,
        "scope": "p0_train_dev_full_step_inputs_only",
        "model_view_schema_version": UNLABELED_VIEW_SCHEMA,
        "source_cache_schema_version": SOURCE_CACHE_SCHEMA_VERSION,
        "source_cache_artifact_sha256": source_cache.artifact_sha256,
        "full_graph_budget_preflight": dict(preflight_binding),
        "graph_budget_decision": dict(decision_binding),
        "case_selection_policy": case_selection_policy,
        "case_count": len(groups), "producer_binding": v5_binding,
        "graph_policy": {
            "selection": "all_occ_faces_and_all_two_owner_adjacencies",
            "pair_order": "query_role_a_then_query_role_b",
            "max_faces_per_graph": budget_decision.max_faces_per_graph,
            "max_edges_per_graph": budget_decision.max_edges_per_graph,
            "endpoint_annotations": "forbidden",
        },
        "row_count": len(safe_rows), "input_domain_sha256": input_domain,
        "rows": safe_rows, "shards": safe_shards,
    }
    safe_manifest["manifest_payload_sha256"] = canonical_sha256(safe_manifest)
    safe_manifest_path = safe_root / "manifest.json"
    _write_exclusive(safe_manifest_path, canonical_bytes(safe_manifest))
    audit = {
        "schema_version": AUDIT_SCHEMA, "producer_binding": v5_binding,
        "row_count": len(safe_rows),
        "source_cache_artifact_sha256": source_cache.artifact_sha256,
        "full_graph_budget_preflight": dict(preflight_binding),
        "graph_budget_decision": dict(decision_binding),
        "case_selection_policy": case_selection_policy, "case_count": len(groups),
        "input_manifest": _binding(safe_manifest_path, name="manifest.json"),
        "training_label_manifest": _binding(label_manifest_path, name="manifest.json"),
        "input_domain_sha256": input_domain,
        "all_graphs_complete_component": True,
        "max_face_count": max(
            proof["face_count"] for row in safe_rows for proof in row["full_graph_proofs"]
        ),
        "max_adjacency_count": max(
            proof["adjacency_count"] for row in safe_rows for proof in row["full_graph_proofs"]
        ),
        "forbidden_model_fields": [], "final_test_touched": False,
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    audit_path = label_root / "audit.json"
    _write_exclusive(audit_path, canonical_bytes(audit))
    label_artifact = {
        "schema_version": LABEL_ARTIFACT_SCHEMA,
        "producer_binding": v5_binding,
        "manifest": _binding(label_manifest_path, name="manifest.json"),
        "audit": _binding(audit_path, name="audit.json"),
        "shards": [_binding(label_root / row["name"], name=row["name"]) for row in label_shards],
    }
    label_artifact["artifact_payload_sha256"] = canonical_sha256(label_artifact)
    label_artifact_path = label_root / "artifact.json"
    _write_exclusive(label_artifact_path, canonical_bytes(label_artifact))
    label_pin = _file_sha256(label_artifact_path)
    safe_artifact = {
        "schema_version": SAFE_ARTIFACT_SCHEMA,
        "producer_binding": v5_binding,
        "manifest": _binding(safe_manifest_path, name="manifest.json"),
        "shards": [_binding(safe_root / row["name"], name=row["name"]) for row in safe_shards],
    }
    safe_artifact["artifact_payload_sha256"] = canonical_sha256(safe_artifact)
    safe_artifact_path = safe_root / "artifact.json"
    _write_exclusive(safe_artifact_path, canonical_bytes(safe_artifact))
    safe_pin = _file_sha256(safe_artifact_path)
    lineage_manifest = {
        "schema_version": LINEAGE_SCHEMA,
        "scope": "p0_train_dev_authority_only_no_model_access_no_final",
        "producer_binding": v5_binding,
        "safe_artifact_sha256": safe_pin,
        "label_artifact_sha256": label_pin,
        "source_cache_artifact_sha256": source_cache_artifact_sha256,
        "source_cache_manifest_sha256": source_manifest_sha256,
        "development_projection_artifact_sha256": projection_artifact_sha256,
        "family_partition_suite_v2_artifact_sha256": suite_v2_artifact_sha256,
        "family_partition_suite_v2_p0_payload_sha256": p0["partition_payload_sha256"],
        "authority_ready_pool_sha256": authority_pool_sha256,
        "full_graph_budget_preflight": dict(preflight_binding),
        "graph_budget_decision": dict(decision_binding),
        "case_selection_policy": case_selection_policy,
        "case_count": len(groups), "row_count": len(lineage_rows),
        "input_domain_sha256": input_domain,
        "semantic_receipt_ledger": semantic_receipt_ledger,
        "identity_production_policy": "same_transaction_case_authority_replay.v1",
        "five_dimension_policy": (
            "case_family_body_sha1_graph_source_lineage_zero_overlap.v1"
        ),
        "rows": lineage_rows, "final_test_touched": False,
    }
    lineage_manifest["manifest_payload_sha256"] = canonical_sha256(lineage_manifest)
    lineage_manifest_path = lineage_root / "manifest.json"
    _write_exclusive(lineage_manifest_path, canonical_bytes(lineage_manifest))
    lineage_artifact = {
        "schema_version": LINEAGE_ARTIFACT_SCHEMA,
        "producer_binding": v5_binding,
        "safe_artifact_sha256": safe_pin,
        "label_artifact_sha256": label_pin,
        "manifest": _binding(lineage_manifest_path, name="manifest.json"),
    }
    lineage_artifact["artifact_payload_sha256"] = canonical_sha256(lineage_artifact)
    lineage_artifact_path = lineage_root / "artifact.json"
    _write_exclusive(lineage_artifact_path, canonical_bytes(lineage_artifact))
    lineage_pin = _file_sha256(lineage_artifact_path)
    bundle_artifact = {
        "schema_version": BUNDLE_ARTIFACT_SCHEMA,
        "producer_binding": v5_binding,
        "safe_artifact": _binding(safe_artifact_path, name="safe/artifact.json"),
        "label_artifact": _binding(label_artifact_path, name="labels/artifact.json"),
        "lineage_artifact": _binding(
            lineage_artifact_path, name="lineage/artifact.json"
        ),
        "safe_artifact_sha256": safe_pin,
        "label_artifact_sha256": label_pin,
        "lineage_artifact_sha256": lineage_pin,
        "final_test_touched": False,
    }
    bundle_artifact["artifact_payload_sha256"] = canonical_sha256(bundle_artifact)
    bundle_artifact_path = staging / "artifact.json"
    _write_exclusive(bundle_artifact_path, canonical_bytes(bundle_artifact))
    bundle_pin = _file_sha256(bundle_artifact_path)
    shutil.rmtree(checkpoint_case_root)
    (staging / "checkpoint.json").unlink()
    os.replace(staging, output)
    progress(
        "v5_published", rows=len(lineage_rows), safe_artifact_sha256=safe_pin,
        label_artifact_sha256=label_pin, lineage_artifact_sha256=lineage_pin,
    )
    return bundle_pin, safe_pin, label_pin, lineage_pin
  except BaseException:
    if staging.exists():
      progress("checkpoint_preserved", checkpoint_directory=str(staging))
    raise
