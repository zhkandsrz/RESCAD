"""Versioned streaming train/dev runner for the label-free Joint V3 learner.

V2 deliberately does not reinterpret the V1 row index.  Its index binds every
row to the source semantic cache, the p0 family partition and the independent
safe/label artifacts.  Training groups supervision rows by immutable input
hash so a full STEP view is forwarded once per epoch, even when it has many
finite-program targets.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from .benchmark_v3_unlabeled_step_view import (
    BenchmarkV3UnlabeledStepView,
    UnlabeledIntrinsicBRepGraph,
)
from .brep_tensor_cache_v4_unlabeled import (
    AuthenticatedUnlabeledTrainingLabel,
    _FullStepGraphCache,
    _TRAINING_LABEL_FACTORY_TOKEN,
    _build_cache_owned_unlabeled_step_view,
    _strict_json,
    _tensor_matches,
    load_unlabeled_tensor_cache_v4,
    load_unlabeled_training_labels_v4,
    materialize_unlabeled_training_rows_by_case,
)
from .joint_interface_program_learner_v1 import LabelFreeBRepGraphV1, LabelFreeBRepPairV1
from .joint_interface_program_learner_v2 import collate_graph_program_pairs_v2
from .joint_interface_program_learner_v3 import (
    ExternallyFrozenCatalogSpecV3,
    JointInterfaceProgramLearnerConfigV3,
    JointInterfaceProgramLearnerV3,
    JointInterfaceProgramTargetsV3,
    joint_interface_program_loss_v3,
    load_externally_frozen_catalog_spec_v3,
    load_joint_interface_checkpoint_v3,
    save_joint_interface_checkpoint_v3,
)


INDEX_SCHEMA = "joint_interface_training_row_index.v2"
RECEIPT_SCHEMA = "joint_interface_training_receipt.v2"
ARTIFACT_SCHEMA = "joint_interface_training_artifact.v2"
SUITE_V2_ARTIFACT_PIN = "c6c4b740ed2a00d3ef537391e470f0bf7eca4371cfa7c752307bf8d6060b6ca6"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
      allow_nan=False,
  ).encode("utf-8")


def _sha(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _binding(path: Path, *, name: str | None = None) -> dict[str, Any]:
  raw = path.read_bytes()
  return {
      "name": path.name if name is None else name,
      "bytes": len(raw),
      "sha256": hashlib.sha256(raw).hexdigest(),
  }


def _digest(value: str, *, label: str, sha1: bool = False) -> str:
  matcher = _SHA1 if sha1 else _SHA256
  if type(value) is not str or matcher.fullmatch(value) is None:
    raise ValueError(f"{label} differs")
  return value


def _strings(value: Any, *, label: str, sha1: bool = False) -> tuple[str, ...]:
  if not isinstance(value, list) or not value:
    raise ValueError(f"{label} differs")
  result = tuple(str(item) for item in value)
  if len(result) != len(set(result)) or any(not item for item in result):
    raise ValueError(f"{label} differs")
  if sha1:
    for item in result:
      _digest(item, label=label, sha1=True)
  return result


@dataclass(frozen=True, slots=True)
class JointTrainingRowBindingV2:
  row_ordinal: int
  safe_input_sha256: str
  label_input_sha256: str
  label_commitment_sha256: str
  source_row_ordinal: int
  source_row_payload_sha256: str
  source_lineage_payload_sha256: str
  source_model_view_sha256: str
  source_split: str
  source_case_index: int
  source_program_index: int
  target_program_index: int
  residual_mask: bool
  assignment_split: str
  case_id: str
  source_ordinal: int
  authority_record_payload_sha256: str
  projection_record_payload_sha256: str
  family_id: str
  family_authority_sha256: str
  body_sha1s: tuple[str, ...]
  graph_identity_sha256: str
  source_lineages: tuple[str, ...]
  semantic_program_id: str
  semantic_direction: str
  semantic_row_sha256: str
  semantic_residual_sha256: str
  semantic_endpoint_a_sha256: str
  semantic_endpoint_b_sha256: str
  source_label_replay_sha256: str
  face_label_replay_sha256: str

  def __post_init__(self) -> None:
    for name in ("row_ordinal", "source_row_ordinal", "source_case_index",
                 "source_program_index", "target_program_index", "source_ordinal"):
      value = getattr(self, name)
      if type(value) is not int or value < 0:
        raise ValueError(f"joint v2 {name} differs")
    for name in (
        "safe_input_sha256", "label_input_sha256", "label_commitment_sha256",
        "source_row_payload_sha256", "source_lineage_payload_sha256",
        "source_model_view_sha256", "authority_record_payload_sha256",
        "projection_record_payload_sha256", "family_authority_sha256",
        "graph_identity_sha256", "semantic_row_sha256",
        "semantic_residual_sha256", "semantic_endpoint_a_sha256",
        "semantic_endpoint_b_sha256",
        "source_label_replay_sha256", "face_label_replay_sha256",
    ):
      _digest(getattr(self, name), label=f"joint v2 {name}")
    if self.safe_input_sha256 != self.label_input_sha256:
      raise ValueError("joint v2 safe/label input hash differs")
    if self.source_split not in {"train", "dev"} or self.assignment_split not in {
        "train", "dev"
    }:
      raise ValueError("joint v2 split differs")
    if type(self.residual_mask) is not bool:
      raise ValueError("joint v2 residual mask differs")
    if not self.case_id.startswith("fusionv2_") or not self.family_id.startswith("leak_"):
      raise ValueError("joint v2 case/family identity differs")
    if not self.semantic_program_id or self.semantic_direction not in {
        "entity_one_to_entity_two", "entity_two_to_entity_one"
    }:
      raise ValueError("joint v2 semantic program/direction differs")
    _strings(list(self.body_sha1s), label="joint v2 body SHA1s", sha1=True)
    _strings(list(self.source_lineages), label="joint v2 source lineages")


@dataclass(frozen=True, slots=True)
class LoadedJointTrainingIndexV2:
  rows: tuple[JointTrainingRowBindingV2, ...]
  authority_bindings: Mapping[str, Any]
  artifact_sha256: str
  index_payload_sha256: str


def _row_from_payload(value: Any) -> JointTrainingRowBindingV2:
  if not isinstance(value, Mapping):
    raise ValueError("joint v2 row is not an object")
  expected = set(JointTrainingRowBindingV2.__dataclass_fields__)
  if set(value) != expected:
    raise ValueError("joint v2 row schema differs")
  normalized = dict(value)
  normalized["body_sha1s"] = tuple(normalized["body_sha1s"])
  normalized["source_lineages"] = tuple(normalized["source_lineages"])
  return JointTrainingRowBindingV2(**normalized)


def _assert_zero_overlap(rows: Sequence[JointTrainingRowBindingV2]) -> None:
  split_rows = {
      split: tuple(row for row in rows if row.assignment_split == split)
      for split in ("train", "dev")
  }
  dimensions: dict[str, tuple[set[str], set[str]]] = {
      "case_id": tuple({row.case_id for row in split_rows[split]} for split in ("train", "dev")),
      "family_id": tuple({row.family_id for row in split_rows[split]} for split in ("train", "dev")),
      "body_sha1": tuple(
          {value for row in split_rows[split] for value in row.body_sha1s}
          for split in ("train", "dev")
      ),
      "graph_identity": tuple(
          {row.graph_identity_sha256 for row in split_rows[split]}
          for split in ("train", "dev")
      ),
      "safe_input_sha256": tuple(
          {row.safe_input_sha256 for row in split_rows[split]}
          for split in ("train", "dev")
      ),
      "source_lineage": tuple(
          {value for row in split_rows[split] for value in row.source_lineages}
          for split in ("train", "dev")
      ),
  }
  for label, (train_values, dev_values) in dimensions.items():
    overlap = train_values & dev_values
    if overlap:
      raise ValueError(f"joint v2 train/dev {label} overlap")


def load_joint_training_row_index_v2(
    path: str | Path, *, expected_artifact_sha256: str,
    expected_safe_artifact_sha256: str, expected_label_artifact_sha256: str,
    expected_input_domain_sha256: str,
    expected_suite_v2_artifact_sha256: str = SUITE_V2_ARTIFACT_PIN,
) -> LoadedJointTrainingIndexV2:
  source = Path(path)
  if _file_sha(source) != expected_artifact_sha256:
    raise ValueError("joint v2 row index differs from external pin")
  payload = json.loads(source.read_text(encoding="utf-8"))
  if not isinstance(payload, Mapping) or set(payload) != {
      "schema_version", "scope", "authority_bindings", "rows",
      "index_payload_sha256",
  }:
    raise ValueError("joint v2 row index schema differs")
  unsigned = dict(payload)
  observed = unsigned.pop("index_payload_sha256")
  if payload["schema_version"] != INDEX_SCHEMA or observed != _sha(unsigned):
    raise ValueError("joint v2 row index self hash differs")
  if payload["scope"] != "p0_train_dev_only_no_final_access":
    raise ValueError("joint v2 row index scope differs")
  authority = payload["authority_bindings"]
  if not isinstance(authority, Mapping):
    raise ValueError("joint v2 authority bindings differ")
  for name, expected in (
      ("safe_artifact_sha256", expected_safe_artifact_sha256),
      ("label_artifact_sha256", expected_label_artifact_sha256),
      ("input_domain_sha256", expected_input_domain_sha256),
      ("family_partition_suite_v2_artifact_sha256", expected_suite_v2_artifact_sha256),
  ):
    if authority.get(name) != expected:
      raise ValueError(f"joint v2 {name} authority differs")
  if authority.get("face_label_verification") != (
      "semantic_receipt_plus_case_batch_materializer_replay.v1"
  ):
    raise ValueError("joint v2 face label authority differs")
  raw_rows = payload["rows"]
  if not isinstance(raw_rows, list) or not raw_rows:
    raise ValueError("joint v2 row index is empty")
  rows = tuple(_row_from_payload(row) for row in raw_rows)
  ordinals = [row.row_ordinal for row in rows]
  if ordinals != list(range(len(rows))):
    raise ValueError("joint v2 row ordinal/reorder differs")
  if len({row.source_row_ordinal for row in rows}) != len(rows):
    raise ValueError("joint v2 source row identity is duplicated")
  _assert_zero_overlap(rows)
  return LoadedJointTrainingIndexV2(
      rows=rows, authority_bindings=dict(authority),
      artifact_sha256=expected_artifact_sha256,
      index_payload_sha256=str(observed),
  )


def _load_pinned_json(path: Path, pin: str, *, label: str) -> Mapping[str, Any]:
  _digest(pin, label=f"{label} pin")
  if _file_sha(path) != pin:
    raise ValueError(f"{label} differs from external pin")
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} is not an object")
  return value


def _load_source_manifest(
    source_cache_root: Path, *, source_cache_artifact_filename: str,
    source_cache_artifact_sha256: str,
) -> tuple[Mapping[str, Any], str]:
  artifact_path = source_cache_root / source_cache_artifact_filename
  artifact = _load_pinned_json(
      artifact_path, source_cache_artifact_sha256, label="joint v2 source-cache artifact"
  )
  binding = artifact.get("manifest")
  if not isinstance(binding, Mapping) or set(binding) != {"name", "bytes", "sha256"}:
    raise ValueError("joint v2 source-cache manifest binding differs")
  manifest_path = source_cache_root / str(binding["name"])
  if _binding(manifest_path, name=str(binding["name"])) != dict(binding):
    raise ValueError("joint v2 source-cache manifest changed")
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if not isinstance(manifest, Mapping) or not isinstance(manifest.get("rows"), list):
    raise ValueError("joint v2 source-cache manifest differs")
  return manifest, str(binding["sha256"])


def _load_case_semantic_commitments(
    projection_record: Mapping[str, Any], *, expected_case_id: str,
    expected_source_ordinal: int,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
  authority_record = projection_record.get("authority_record")
  receipts = authority_record.get("stage_receipts") if isinstance(authority_record, Mapping) else None
  candidates = [
      row for row in receipts
      if isinstance(row, Mapping)
      and Path(str(row.get("path"))).name == "semantic_v2.receipt.json"
      and row.get("status") == "verified"
  ] if isinstance(receipts, list) else []
  if len(candidates) != 1:
    raise ValueError("joint v2 semantic stage receipt differs")
  stage_binding = candidates[0]
  stage_path = Path(str(stage_binding["path"]))
  raw = stage_path.read_bytes()
  if (
      len(raw) != stage_binding.get("bytes")
      or hashlib.sha256(raw).hexdigest() != stage_binding.get("sha256")
  ):
    raise ValueError("joint v2 semantic stage receipt bytes changed")
  receipt = _strict_json(raw, label="joint v2 semantic stage receipt")
  if not isinstance(receipt, Mapping):
    raise ValueError("joint v2 semantic stage receipt differs")
  unsigned_receipt = dict(receipt)
  receipt_hash = unsigned_receipt.pop("receipt_payload_sha256", None)
  if (
      receipt_hash != _sha(unsigned_receipt)
      or receipt_hash != stage_binding.get("receipt_payload_sha256")
      or receipt.get("case_id") != expected_case_id
      or receipt.get("source_ordinal") != expected_source_ordinal
      or receipt.get("stage") != "semantic_v2"
      or receipt.get("status") != "verified"
      or receipt.get("final_test_touched") is not False
  ):
    raise ValueError("joint v2 semantic stage receipt authority differs")
  outputs = receipt.get("outputs")
  semantic_outputs = [
      row for row in outputs
      if isinstance(row, Mapping) and row.get("role") == "mate_semantic"
  ] if isinstance(outputs, list) else []
  if len(semantic_outputs) != 1:
    raise ValueError("joint v2 semantic authority output differs")
  output = semantic_outputs[0]
  semantic_path = Path(str(output["path"]))
  semantic_raw = semantic_path.read_bytes()
  if (
      len(semantic_raw) != output.get("bytes")
      or hashlib.sha256(semantic_raw).hexdigest() != output.get("sha256")
      or output.get("schema_version") != "mate_semantic_authority.v2"
  ):
    raise ValueError("joint v2 semantic authority bytes changed")
  semantic = _strict_json(semantic_raw, label="joint v2 semantic authority")
  if not isinstance(semantic, Mapping):
    raise ValueError("joint v2 semantic authority differs")
  unsigned_semantic = dict(semantic)
  semantic_hash = unsigned_semantic.pop("receipt_payload_sha256", None)
  commitments = semantic.get("row_commitments")
  if (
      semantic.get("schema_version") != "mate_semantic_authority.v2"
      or semantic.get("formal") is not True
      or semantic_hash != _sha(unsigned_semantic)
      or not isinstance(commitments, list)
      or semantic.get("row_count") != len(commitments)
      or semantic.get("replay_rows_sha256") != _sha(commitments)
  ):
    raise ValueError("joint v2 semantic authority self binding differs")
  expected_keys = {
      "program_id", "source_case_token_sha256", "source_contact_ordinal",
      "direction", "endpoint_a_sha256", "endpoint_b_sha256",
      "residual_sha256", "row_sha256",
  }
  for commitment in commitments:
    if not isinstance(commitment, Mapping) or set(commitment) != expected_keys:
      raise ValueError("joint v2 semantic row commitment schema differs")
    for name in (
        "source_case_token_sha256", "endpoint_a_sha256", "endpoint_b_sha256",
        "residual_sha256", "row_sha256",
    ):
      _digest(str(commitment[name]), label=f"joint v2 semantic {name}")
    if (
        not str(commitment["program_id"])
        or commitment["direction"] not in {
            "entity_one_to_entity_two", "entity_two_to_entity_one"
        }
    ):
      raise ValueError("joint v2 semantic row commitment differs")
  return tuple(dict(row) for row in commitments), {
      "stage_receipt_sha256": str(stage_binding["sha256"]),
      "stage_receipt_payload_sha256": str(receipt_hash),
      "semantic_authority_sha256": str(output["sha256"]),
      "semantic_authority_payload_sha256": str(semantic_hash),
  }


def endpoint_backfill_conflict_count(
    observations: Sequence[tuple[tuple[str, str], tuple[int, ...]]],
) -> int:
  """Count semantic endpoint tokens that cannot safely share a face label."""

  values: dict[tuple[str, str], set[tuple[int, ...]]] = defaultdict(set)
  for token, node_indices in observations:
    values[token].add(tuple(node_indices))
  return sum(len(node_sets) > 1 for node_sets in values.values())


def publish_joint_training_row_index_v2(
    output_path: str | Path, *,
    safe_root: str | Path, safe_artifact_sha256: str,
    label_root: str | Path, label_artifact_sha256: str,
    historical_producer_revision: str,
    source_cache: Any, source_cache_root: str | Path,
    source_cache_artifact_filename: str, source_cache_artifact_sha256: str,
    semantic_index: Any, preflight: Any, budget_decision: Any,
    case_selection_policy: str,
    projection_path: str | Path, projection_artifact_sha256: str,
    suite_v2_root: str | Path,
    suite_v2_artifact_sha256: str = SUITE_V2_ARTIFACT_PIN,
    authority_pool_path: str | Path = "",
    authority_pool_sha256: str = "",
    max_independent_replay_rows: int = 1000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> str:
  """Replay and publish a small/frozen V2 row index.

  Current V4 manifests intentionally contain no source ordinal.  Consequently
  full-scale index creation is refused instead of silently zipping 20,696
  positions or repeating the 12-hour STEP materialization.  The existing 396
  row pilot is below this gate and receives an independent geometry replay of
  every safe input and every face/program/residual label.
  """
  target = Path(output_path)
  def progress(stage: str, **values: Any) -> None:
    if progress_callback is not None:
      progress_callback({"stage": stage, **values})

  if target.exists():
    raise FileExistsError("joint v2 row-index output already exists")
  safe = load_unlabeled_tensor_cache_v4(
      safe_root, expected_artifact_sha256=safe_artifact_sha256,
      expected_historical_producer_revision=historical_producer_revision,
  )
  labels = load_unlabeled_training_labels_v4(
      label_root, expected_artifact_sha256=label_artifact_sha256,
      expected_input_domain_sha256=safe.input_domain_sha256,
      expected_historical_producer_revision=historical_producer_revision,
  )
  if safe.row_count != labels.row_count:
    raise ValueError("joint v2 replay safe/label row counts differ")
  if safe.row_count > max_independent_replay_rows:
    raise ValueError(
        "full V4 rows lack source ordinals; publish a lineage-augmented versioned "
        "cache instead of position-zipping or re-running full STEP materialization"
    )
  safe_views = tuple(safe.iter_model_inputs())
  bound_labels = tuple(labels.iter_bound_labels())
  source_manifest, source_manifest_sha256 = _load_source_manifest(
      Path(source_cache_root),
      source_cache_artifact_filename=source_cache_artifact_filename,
      source_cache_artifact_sha256=source_cache_artifact_sha256,
  )
  source_rows_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
  for row in source_manifest["rows"]:
    if not isinstance(row, Mapping):
      raise ValueError("joint v2 source row differs")
    key = (str(row.get("split")), int(row.get("case_index")), int(row.get("program_index")))
    if key in source_rows_by_key:
      raise ValueError("joint v2 source semantic identity is duplicated")
    source_rows_by_key[key] = row
  progress("source_manifest_validated", source_rows=len(source_rows_by_key))

  projection = _load_pinned_json(
      Path(projection_path), projection_artifact_sha256,
      label="joint v2 development projection",
  )
  if projection.get("development") is not True:
    raise ValueError("joint v2 projection is not development-only")
  development = projection.get("records")
  if not isinstance(development, Mapping):
    raise ValueError("joint v2 development projection differs")

  suite_root = Path(suite_v2_root)
  suite_artifact = _load_pinned_json(
      suite_root / "artifact.json", suite_v2_artifact_sha256,
      label="joint v2 family partition suite artifact",
  )
  suite_binding = next(
      (
          item for item in suite_artifact.get("files", [])
          if isinstance(item, Mapping) and item.get("name") == "family_partition_suite.json"
      ),
      None,
  )
  if suite_binding is None or _binding(
      suite_root / "family_partition_suite.json", name="family_partition_suite.json"
  ) != dict(suite_binding):
    raise ValueError("joint v2 family partition suite payload changed")
  suite = json.loads((suite_root / "family_partition_suite.json").read_text(encoding="utf-8"))
  if not isinstance(suite, Mapping) or not isinstance(suite.get("partitions"), list):
    raise ValueError("joint v2 family partition suite differs")
  p0 = next(
      (part for part in suite["partitions"] if isinstance(part, Mapping) and part.get("partition_index") == 0),
      None,
  )
  if p0 is None:
    raise ValueError("joint v2 p0 family partition is missing")
  assignments: dict[tuple[str, int], tuple[str, Mapping[str, Any]]] = {}
  for split in ("train", "dev"):
    records = p0.get(f"{split}_records")
    if not isinstance(records, list):
      raise ValueError("joint v2 p0 assignment differs")
    for record in records:
      key = (str(record.get("case_id")), int(record.get("source_ordinal")))
      if key in assignments:
        raise ValueError("joint v2 p0 assignment is duplicated")
      assignments[key] = (split, record)

  pool = _load_pinned_json(
      Path(authority_pool_path), authority_pool_sha256,
      label="joint v2 authority-ready pool",
  )
  pool_records = pool.get("authority_ready_records")
  if not isinstance(pool_records, list):
    raise ValueError("joint v2 authority-ready pool differs")
  pool_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
  for record in pool_records:
    key = (str(record.get("case_id")), int(record.get("source_ordinal")))
    if key in pool_by_key:
      raise ValueError("joint v2 authority-ready identity is duplicated")
    pool_by_key[key] = record

  selected_source_rows: list[Mapping[str, Any]] = []
  for source_row in source_manifest["rows"]:
    row_split = str(source_row["split"])
    case_index = int(source_row["case_index"])
    if not budget_decision.allows(split=row_split, case_index=case_index):
      continue
    if case_selection_policy == "preflight_excluded_decision_admitted_cases":
      if preflight.allows(split=row_split, case_index=case_index):
        continue
    elif case_selection_policy != "all_decision_admitted_cases":
      raise ValueError("joint v2 case selection policy differs")
    selected_source_rows.append(source_row)
    if len(selected_source_rows) == safe.row_count:
      break
  if len(selected_source_rows) != safe.row_count:
    raise ValueError("joint v2 selected semantic-row domain differs")
  selected_ordinals = tuple(int(row["row_ordinal"]) for row in selected_source_rows)
  if tuple(sorted(set(selected_ordinals))) != selected_ordinals:
    raise ValueError("joint v2 sparse source-row order differs")
  progress(
      "source_ordinals_selected", selected_rows=len(selected_ordinals),
      first_source_ordinal=selected_ordinals[0],
      last_source_ordinal=selected_ordinals[-1],
  )
  sparse_loader = getattr(source_cache, "iter_examples_by_row_ordinals", None)
  if not callable(sparse_loader):
    raise ValueError("joint v2 source cache lacks sparse authenticated row loading")
  cached_rows = tuple(sparse_loader(row_ordinals=selected_ordinals))
  if len(cached_rows) != safe.row_count:
    raise ValueError("joint v2 sparse source cache returned an incomplete domain")
  selected: list[tuple[Any, Mapping[str, Any]]] = []
  for cached, source_row in zip(cached_rows, selected_source_rows, strict=True):
    if source_row.get("lineage") != dict(cached.lineage):
      raise ValueError("joint v2 source row independent identity differs")
    selected.append((cached, source_row))
  progress("source_rows_loaded", selected_rows=len(selected))

  semantic_cache: dict[tuple[str, int], tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]] = {}
  semantic_commitments: list[Mapping[str, Any]] = []
  semantic_receipt_ledger: list[dict[str, Any]] = []
  for ordinal, source_row in enumerate(selected_source_rows):
    split = str(source_row["split"])
    case_index = int(source_row["case_index"])
    projection_records = development.get(split)
    if not isinstance(projection_records, list) or case_index >= len(projection_records):
      raise ValueError("joint v2 source projection slot differs")
    projection_record = projection_records[case_index]
    authority_record = projection_record.get("authority_record")
    if not isinstance(authority_record, Mapping):
      raise ValueError("joint v2 projection authority record differs")
    cache_key = (split, case_index)
    loaded_semantic = semantic_cache.get(cache_key)
    if loaded_semantic is None:
      loaded_semantic = _load_case_semantic_commitments(
          projection_record,
          expected_case_id=str(authority_record["case_id"]),
          expected_source_ordinal=int(authority_record["source_ordinal"]),
      )
      semantic_cache[cache_key] = loaded_semantic
      semantic_receipt_ledger.append({
          "case_id": str(authority_record["case_id"]),
          "source_ordinal": int(authority_record["source_ordinal"]),
          **dict(loaded_semantic[1]),
      })
    commitments = loaded_semantic[0]
    program_index = int(source_row["program_index"])
    if program_index >= len(commitments):
      raise ValueError("joint v2 semantic program ordinal differs")
    commitment = commitments[program_index]
    semantic_commitments.append(commitment)
  progress(
      "semantic_receipts_validated", cases=len(semantic_cache),
      semantic_rows=len(semantic_commitments),
  )
  backfill_observations: list[tuple[tuple[str, str], tuple[int, ...]]] = []
  for ordinal, commitment in enumerate(semantic_commitments):
    proofs = safe._manifest["rows"][ordinal]["full_graph_proofs"]
    label = bound_labels[ordinal][1]
    backfill_observations.extend((
        (
            (str(commitment["endpoint_a_sha256"]), str(proofs[0]["step_sha256"])),
            tuple(label.endpoint_graph_node_indices[0]),
        ),
        (
            (str(commitment["endpoint_b_sha256"]), str(proofs[1]["step_sha256"])),
            tuple(label.endpoint_graph_node_indices[1]),
        ),
    ))
  rejected_backfill_conflicts = endpoint_backfill_conflict_count(backfill_observations)
  progress(
      "endpoint_backfill_rejected", conflicting_tokens=rejected_backfill_conflicts,
      policy="always_use_exact_case_batch_replay",
  )

  graph_cache = _FullStepGraphCache(budget_decision=budget_decision)
  replayed_rows: list[Any] = [None] * safe.row_count
  case_groups: list[tuple[int, int]] = []
  start = 0
  while start < len(selected):
    first_lineage = selected[start][0].lineage
    stop = start + 1
    while stop < len(selected):
      lineage = selected[stop][0].lineage
      if (
          lineage.get("split") != first_lineage.get("split")
          or lineage.get("case_index") != first_lineage.get("case_index")
      ):
        break
      stop += 1
    case_groups.append((start, stop))
    start = stop
  for case_position, (start, stop) in enumerate(case_groups):
    batch = materialize_unlabeled_training_rows_by_case(
        index=semantic_index,
        cached_rows=tuple(selected[index][0] for index in range(start, stop)),
        full_graph_cache=graph_cache,
    )
    replayed_rows[start:stop] = batch
    progress(
        "case_batch_replayed", completed_cases=case_position + 1,
        total_cases=len(case_groups), completed_rows=stop, total_rows=safe.row_count,
        unique_full_step_graphs=len(graph_cache._values),
    )
  if any(row is None for row in replayed_rows):
    raise RuntimeError("joint v2 case-batch replay domain is incomplete")

  rows: list[dict[str, Any]] = []
  for ordinal, ((cached, source_row), safe_view, bound_label) in enumerate(zip(
      selected, safe_views, bound_labels, strict=True
  )):
    label_input_sha256, label = bound_label
    commitment = semantic_commitments[ordinal]
    replayed = replayed_rows[ordinal]
    replay_label = replayed.training_label
    if (
        replayed.model_input.input_sha256 != safe_view.input_sha256
        or replayed.model_input.receipt_payload() != safe_view.receipt_payload()
        or label_input_sha256 != safe_view.input_sha256
        or replay_label.commitment_sha256 != label.commitment_sha256
        or replay_label.target_program_index != label.target_program_index
        or replay_label.residual_mask != label.residual_mask
        or not np.array_equal(replay_label.residual_translation, label.residual_translation)
        or not np.array_equal(
            replay_label.residual_rotation_vector, label.residual_rotation_vector
        )
        or replay_label.endpoint_graph_node_indices != label.endpoint_graph_node_indices
        or cached.example.target_program_index != label.target_program_index
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
      raise ValueError("joint v2 case-batch safe/program/residual/face replay differs")
    split = str(source_row["split"])
    case_index = int(source_row["case_index"])
    projection_records = development.get(split)
    if not isinstance(projection_records, list) or case_index >= len(projection_records):
      raise ValueError("joint v2 source projection slot differs")
    projection_record = projection_records[case_index]
    authority_record = projection_record.get("authority_record")
    if not isinstance(authority_record, Mapping):
      raise ValueError("joint v2 projection authority record differs")
    case_id = str(authority_record.get("case_id"))
    source_ordinal = int(authority_record.get("source_ordinal"))
    assignment = assignments.get((case_id, source_ordinal))
    pool_record = pool_by_key.get((case_id, source_ordinal))
    if assignment is None or pool_record is None:
      raise ValueError("joint v2 source row lacks p0/pool authority")
    assignment_split, assignment_record = assignment
    if assignment_record.get("authority_record_payload_sha256") != pool_record.get(
        "merged_record_payload_sha256"
    ):
      raise ValueError("joint v2 p0/pool record binding differs")
    family_identity = pool_record.get("source_family_identity")
    provenance = family_identity.get("family_provenance") if isinstance(family_identity, Mapping) else None
    if not isinstance(provenance, Mapping):
      raise ValueError("joint v2 family provenance differs")
    source_lineages = _strings(
        provenance.get("source_lineage"), label="joint v2 authority source lineage"
    )
    body_sha1s = _strings(
        provenance.get("body_sha1_set"), label="joint v2 authority body SHA1", sha1=True
    )
    graph_identity = _sha(provenance.get("assembly_graph"))
    replay_sha256 = _sha({
        "source_row_payload_sha256": source_row["row_payload_sha256"],
        "safe_input_sha256": safe_view.input_sha256,
        "label_commitment_sha256": label.commitment_sha256,
        "semantic_endpoint_a_sha256": commitment["endpoint_a_sha256"],
        "semantic_endpoint_b_sha256": commitment["endpoint_b_sha256"],
        "endpoint_graph_node_indices": [list(value) for value in label.endpoint_graph_node_indices],
        "full_graph_proofs": safe._manifest["rows"][ordinal]["full_graph_proofs"],
        "case_batch_replay_policy": "all_programs_shared_case_inputs.v1",
    })
    source_label_replay_sha256 = _sha({
        "source_row_payload_sha256": source_row["row_payload_sha256"],
        "semantic_row_commitment": dict(commitment),
        "target_program_index": label.target_program_index,
        "residual_mask": label.residual_mask,
        "residual_translation": label.residual_translation.tolist(),
        "residual_rotation_vector": label.residual_rotation_vector.tolist(),
        "label_commitment_sha256": label.commitment_sha256,
    })
    binding = JointTrainingRowBindingV2(
        row_ordinal=ordinal,
        safe_input_sha256=safe_view.input_sha256,
        label_input_sha256=label_input_sha256,
        label_commitment_sha256=label.commitment_sha256,
        source_row_ordinal=int(source_row["row_ordinal"]),
        source_row_payload_sha256=str(source_row["row_payload_sha256"]),
        source_lineage_payload_sha256=str(source_row["lineage"]["lineage_payload_sha256"]),
        source_model_view_sha256=str(source_row["model_view_sha256"]),
        source_split=split, source_case_index=case_index,
        source_program_index=int(source_row["program_index"]),
        target_program_index=label.target_program_index,
        residual_mask=label.residual_mask,
        assignment_split=assignment_split, case_id=case_id,
        source_ordinal=source_ordinal,
        authority_record_payload_sha256=str(
            assignment_record["authority_record_payload_sha256"]
        ),
        projection_record_payload_sha256=str(
            projection_record["projection_record_payload_sha256"]
        ),
        family_id=str(assignment_record["leakage_component_id"]),
        family_authority_sha256=str(
            family_identity["family_provenance_payload_sha256"]
        ),
        body_sha1s=body_sha1s, graph_identity_sha256=graph_identity,
        source_lineages=source_lineages,
        semantic_program_id=str(commitment["program_id"]),
        semantic_direction=str(commitment["direction"]),
        semantic_row_sha256=str(commitment["row_sha256"]),
        semantic_residual_sha256=str(commitment["residual_sha256"]),
        semantic_endpoint_a_sha256=str(commitment["endpoint_a_sha256"]),
        semantic_endpoint_b_sha256=str(commitment["endpoint_b_sha256"]),
        source_label_replay_sha256=source_label_replay_sha256,
        face_label_replay_sha256=replay_sha256,
    )
    rows.append({
        field: list(getattr(binding, field)) if field in {"body_sha1s", "source_lineages"}
        else getattr(binding, field)
        for field in JointTrainingRowBindingV2.__dataclass_fields__
    })
    if ordinal == 0 or (ordinal + 1) % 100 == 0 or ordinal + 1 == safe.row_count:
      progress(
          "semantic_rows_bound", completed=ordinal + 1, total=safe.row_count,
      )
  checked = tuple(_row_from_payload(row) for row in rows)
  _assert_zero_overlap(checked)
  authority_bindings = {
      "safe_artifact_sha256": safe_artifact_sha256,
      "label_artifact_sha256": label_artifact_sha256,
      "input_domain_sha256": safe.input_domain_sha256,
      "source_cache_artifact_sha256": source_cache_artifact_sha256,
      "source_cache_manifest_sha256": source_manifest_sha256,
      "development_projection_artifact_sha256": projection_artifact_sha256,
      "development_projection_payload_sha256": projection["projection_payload_sha256"],
      "family_partition_suite_v2_artifact_sha256": suite_v2_artifact_sha256,
      "family_partition_suite_v2_payload_sha256": suite["suite_payload_sha256"],
      "family_partition_suite_v2_p0_payload_sha256": p0["partition_payload_sha256"],
      "authority_ready_pool_sha256": authority_pool_sha256,
      "full_graph_budget_preflight": dict(preflight.binding),
      "graph_budget_decision": dict(budget_decision.binding),
      "case_selection_policy": case_selection_policy,
      "materializer_traversal": "sparse_source_ordinals_plus_case_batch_replay.v2",
      "semantic_receipt_ledger": semantic_receipt_ledger,
      "case_batch_count": len(case_groups),
      "case_batch_replayed_row_count": len(replayed_rows),
      "rejected_endpoint_backfill_conflict_count": rejected_backfill_conflicts,
      "face_label_verification": (
          "semantic_receipt_plus_case_batch_materializer_replay.v1"
      ),
      "final_test_touched": False,
  }
  payload: dict[str, Any] = {
      "schema_version": INDEX_SCHEMA,
      "scope": "p0_train_dev_only_no_final_access",
      "authority_bindings": authority_bindings,
      "rows": rows,
  }
  payload["index_payload_sha256"] = _sha(payload)
  target.parent.mkdir(parents=True, exist_ok=True)
  temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("xb") as stream:
      stream.write(_canonical_bytes(payload))
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, target)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
  pin = _file_sha(target)
  progress("index_published", rows=len(rows), artifact_sha256=pin)
  return pin


class _NpzLRU:
  def __init__(self, *, capacity: int) -> None:
    if type(capacity) is not int or capacity < 1:
      raise ValueError("NPZ LRU capacity must be positive")
    self._capacity = capacity
    self._values: OrderedDict[Path, Any] = OrderedDict()

  def get(self, path: Path, *, expected_binding: Mapping[str, Any]) -> Any:
    existing = self._values.pop(path, None)
    if existing is not None:
      self._values[path] = existing
      return existing
    if _binding(path, name=str(expected_binding["name"])) != {
        key: expected_binding[key] for key in ("name", "bytes", "sha256")
    }:
      raise ValueError("joint v2 streaming shard changed")
    archive = np.load(path, allow_pickle=False)
    self._values[path] = archive
    while len(self._values) > self._capacity:
      _old_path, old = self._values.popitem(last=False)
      old.close()
    return archive

  def close(self) -> None:
    while self._values:
      _path, archive = self._values.popitem(last=False)
      archive.close()


class StreamingJointDatasetV2:
  """Validated random-access dataset with lazy shards and immutable groups."""

  def __init__(
      self, *, safe_root: str | Path, safe_artifact_sha256: str,
      label_root: str | Path, label_artifact_sha256: str,
      historical_producer_revision: str, row_index_path: str | Path,
      row_index_artifact_sha256: str, shard_lru_capacity: int = 4,
  ) -> None:
    if Path(safe_root).resolve() == Path(label_root).resolve():
      raise ValueError("joint v2 safe inputs and labels must be physically separate")
    if _REVISION.fullmatch(historical_producer_revision) is None:
      raise ValueError("joint v2 historical producer revision differs")
    self._safe = load_unlabeled_tensor_cache_v4(
        safe_root, expected_artifact_sha256=safe_artifact_sha256,
        expected_historical_producer_revision=historical_producer_revision,
    )
    self._labels = load_unlabeled_training_labels_v4(
        label_root, expected_artifact_sha256=label_artifact_sha256,
        expected_input_domain_sha256=self._safe.input_domain_sha256,
        expected_historical_producer_revision=historical_producer_revision,
    )
    self.index = load_joint_training_row_index_v2(
        row_index_path, expected_artifact_sha256=row_index_artifact_sha256,
        expected_safe_artifact_sha256=safe_artifact_sha256,
        expected_label_artifact_sha256=label_artifact_sha256,
        expected_input_domain_sha256=self._safe.input_domain_sha256,
    )
    if self._safe.row_count != self._labels.row_count or len(self.index.rows) != self._safe.row_count:
      raise ValueError("joint v2 safe/label/index row counts differ")
    self._safe_rows = tuple(self._safe._manifest["rows"])
    self._label_rows = tuple(self._labels._manifest["rows"])
    self._safe_shards = tuple(self._safe._manifest["shards"])
    self._label_shards = tuple(self._labels._manifest["shards"])
    self._safe_lru = _NpzLRU(capacity=shard_lru_capacity)
    self._label_lru = _NpzLRU(capacity=shard_lru_capacity)
    self._safe_shard_for_row = self._locator(self._safe_shards, len(self.index.rows))
    self._label_shard_for_row = self._locator(self._label_shards, len(self.index.rows))
    for ordinal, binding in enumerate(self.index.rows):
      safe_row = self._safe_rows[ordinal]
      label_row = self._label_rows[ordinal]
      if (
          safe_row["row_ordinal"] != ordinal or label_row["row_ordinal"] != ordinal
          or safe_row["input_sha256"] != binding.safe_input_sha256
          or label_row["input_sha256"] != binding.label_input_sha256
          or label_row["label_commitment_sha256"] != binding.label_commitment_sha256
      ):
        raise ValueError("joint v2 safe/label/index ordinal binding differs")
    self._groups = group_row_ordinals_by_input(self.index.rows)

  @staticmethod
  def _locator(shards: Sequence[Mapping[str, Any]], row_count: int) -> tuple[int, ...]:
    values = [-1] * row_count
    for shard_index, shard in enumerate(shards):
      start = int(shard["row_start"])
      count = int(shard["row_count"])
      for ordinal in range(start, start + count):
        if ordinal >= row_count or values[ordinal] != -1:
          raise ValueError("joint v2 shard row domain differs")
        values[ordinal] = shard_index
    if any(value < 0 for value in values):
      raise ValueError("joint v2 shard row domain is incomplete")
    return tuple(values)

  @property
  def input_domain_sha256(self) -> str:
    return self._safe.input_domain_sha256

  @property
  def row_count(self) -> int:
    return len(self.index.rows)

  def groups(self, split: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if split not in self._groups:
      raise ValueError("joint v2 split differs")
    return self._groups[split]

  def close(self) -> None:
    self._safe_lru.close()
    self._label_lru.close()

  def load_view(self, ordinal: int) -> BenchmarkV3UnlabeledStepView:
    row = self._safe_rows[ordinal]
    shard = self._safe_shards[self._safe_shard_for_row[ordinal]]
    archive = self._safe_lru.get(self._safe._root / shard["name"], expected_binding=shard)
    receipts = {str(item["name"]): item for item in row["tensors"]}
    arrays = {name: archive[name].copy() for name in receipts}
    if any(not _tensor_matches(receipt, arrays[name]) for name, receipt in receipts.items()):
      raise ValueError("joint v2 safe tensor differs from manifest")
    prefix = f"r{ordinal:08d}"
    graphs = tuple(
        UnlabeledIntrinsicBRepGraph(
            node_features=arrays[f"{prefix}_g{part}_nodes"],
            edge_index=arrays[f"{prefix}_g{part}_edge_index"],
            edge_features=arrays[f"{prefix}_g{part}_edge_features"],
        )
        for part in range(int(row["graph_count"]))
    )
    view = _build_cache_owned_unlabeled_step_view(graphs=graphs)
    if view.input_sha256 != row["input_sha256"] or view.receipt_payload() != row["view_receipt"]:
      raise ValueError("joint v2 safe view differs from manifest")
    return view

  def load_label(self, ordinal: int) -> AuthenticatedUnlabeledTrainingLabel:
    row = self._label_rows[ordinal]
    shard = self._label_shards[self._label_shard_for_row[ordinal]]
    archive = self._label_lru.get(self._labels._root / shard["name"], expected_binding=shard)
    prefix = f"r{ordinal:08d}"
    names = (
        f"{prefix}_class", f"{prefix}_translation", f"{prefix}_rotation",
        f"{prefix}_mask", f"{prefix}_endpoint_a", f"{prefix}_endpoint_b",
    )
    arrays = {name: archive[name].copy() for name in names}
    receipts = {str(item["name"]): item for item in row["tensors"]}
    if set(receipts) != set(arrays) or any(
        not _tensor_matches(receipts[name], arrays[name]) for name in arrays
    ):
      raise ValueError("joint v2 label tensor differs from manifest")
    label = AuthenticatedUnlabeledTrainingLabel(
        target_program_index=int(arrays[f"{prefix}_class"].reshape(-1)[0]),
        residual_translation=arrays[f"{prefix}_translation"],
        residual_rotation_vector=arrays[f"{prefix}_rotation"],
        residual_mask=bool(arrays[f"{prefix}_mask"].reshape(-1)[0]),
        endpoint_graph_node_indices=(
            arrays[f"{prefix}_endpoint_a"].tolist(),
            arrays[f"{prefix}_endpoint_b"].tolist(),
        ),
        _factory_token=_TRAINING_LABEL_FACTORY_TOKEN,
    )
    binding = self.index.rows[ordinal]
    if (
        label.commitment_sha256 != row["label_commitment_sha256"]
        or label.commitment_sha256 != binding.label_commitment_sha256
        or label.target_program_index != binding.target_program_index
        or label.residual_mask != binding.residual_mask
    ):
      raise ValueError("joint v2 label differs from row authority")
    return label


def group_row_ordinals_by_input(
    rows: Sequence[JointTrainingRowBindingV2],
) -> dict[str, tuple[tuple[str, tuple[int, ...]], ...]]:
  grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
  for row in rows:
    grouped[(row.assignment_split, row.safe_input_sha256)].append(row.row_ordinal)
  return {
      split: tuple(
          (input_hash, tuple(grouped[(split, input_hash)]))
          for input_hash in sorted(
              {key[1] for key in grouped if key[0] == split},
              key=lambda value: min(grouped[(split, value)]),
          )
      )
      for split in ("train", "dev")
  }


@dataclass(frozen=True, slots=True)
class JointInterfaceTrainingConfigV2:
  seed: int
  max_epochs: int
  learning_rate: float
  weight_decay: float
  gradient_clip_norm: float
  views_per_optimizer_step: int
  early_stop_patience: int
  device_policy: str
  model_config: JointInterfaceProgramLearnerConfigV3
  engineering_smoke: bool = False
  smoke_dev_family_holdout: bool = False

  def __post_init__(self) -> None:
    if type(self.seed) is not int or self.seed < 0:
      raise ValueError("joint v2 seed differs")
    for name in ("max_epochs", "views_per_optimizer_step", "early_stop_patience"):
      value = getattr(self, name)
      minimum = 0 if name == "early_stop_patience" else 1
      if type(value) is not int or value < minimum:
        raise ValueError(f"joint v2 {name} differs")
    for name in ("learning_rate", "gradient_clip_norm"):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"joint v2 {name} differs")
    if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
      raise ValueError("joint v2 weight decay differs")
    if self.device_policy != "reproducible_cpu.v1":
      raise ValueError("joint v2 default policy is reproducible CPU only")
    if type(self.model_config) is not JointInterfaceProgramLearnerConfigV3:
      raise TypeError("joint v2 requires an explicit V3 model config")
    if self.smoke_dev_family_holdout and not self.engineering_smoke:
      raise ValueError("joint v2 dev remap is engineering-smoke-only")


def _pair_from_view(view: BenchmarkV3UnlabeledStepView) -> LabelFreeBRepPairV1:
  graphs = tuple(
      LabelFreeBRepGraphV1(
          # Cache-owned arrays are immutable.  Copy once per unique view so
          # PyTorch never receives a non-writable NumPy buffer.
          node_features=torch.from_numpy(np.array(graph.node_features, copy=True)),
          edge_index=torch.from_numpy(np.array(graph.edge_index, copy=True)),
          edge_features=torch.from_numpy(np.array(graph.edge_features, copy=True)),
      )
      for graph in view.graphs
  )
  return LabelFreeBRepPairV1(part_a=graphs[0], part_b=graphs[1])


def _forward_view_only(
    model: JointInterfaceProgramLearnerV3, view: BenchmarkV3UnlabeledStepView,
    catalog: ExternallyFrozenCatalogSpecV3,
) -> Any:
  inputs = collate_graph_program_pairs_v2((_pair_from_view(view),), catalog=catalog.roster)
  return model(inputs)


def _target_from_label(
    label: AuthenticatedUnlabeledTrainingLabel, *, catalog_sha256: str,
) -> JointInterfaceProgramTargetsV3:
  endpoints = label.endpoint_graph_node_indices
  if len(endpoints[0]) != 1 or len(endpoints[1]) != 1:
    raise ValueError("Joint V3 requires unique canonical endpoint faces")
  return JointInterfaceProgramTargetsV3(
      face_index_a=torch.tensor([endpoints[0][0]], dtype=torch.long),
      face_index_b=torch.tensor([endpoints[1][0]], dtype=torch.long),
      program_index=torch.tensor([label.target_program_index], dtype=torch.long),
      residual_translation_pair_local=torch.tensor(
          label.residual_translation[None, :], dtype=torch.float32
      ),
      residual_rotation_vector_pair_local=torch.tensor(
          label.residual_rotation_vector[None, :], dtype=torch.float32
      ),
      residual_mask=torch.tensor([label.residual_mask], dtype=torch.bool),
      expected_catalog_roster_sha256=catalog_sha256,
  )


def _joint_hit(output: Any, target: JointInterfaceProgramTargetsV3, k: int) -> int:
  core = output.core
  flat = core.inference_joint_logits[0].reshape(-1)
  top = torch.topk(flat, k=min(k, flat.numel()), sorted=False).indices
  face_b_count = core.inference_joint_logits.shape[2]
  program_count = core.inference_joint_logits.shape[3]
  for value in top.tolist():
    program_pos = value % program_count
    face_b_pos = (value // program_count) % face_b_count
    face_a_pos = value // (program_count * face_b_count)
    if (
        int(core.selected_face_indices[0, 0, face_a_pos]) == int(target.face_index_a[0])
        and int(core.selected_face_indices[0, 1, face_b_pos]) == int(target.face_index_b[0])
        and int(core.candidate_program_indices[0, program_pos]) == int(target.program_index[0])
    ):
      return 1
  return 0


def _program_hit(output: Any, target_program_index: int, k: int) -> int:
  core = output.core
  # Program-only retrieval marginalizes the two selected-face axes with max;
  # it does not receive either gold face identity.
  scores = core.inference_joint_logits[0].amax(dim=(0, 1))
  top_positions = torch.topk(scores, k=min(k, scores.numel()), sorted=False).indices
  predicted = core.candidate_program_indices[0, top_positions]
  return int((predicted == int(target_program_index)).any())


def _face_hit(output: Any, target: JointInterfaceProgramTargetsV3, part: int, k: int) -> int:
  core = output.core
  logits = core.face_salience_logits[0, part].masked_fill(
      ~core.all_face_mask[0, part], float("-inf")
  )
  positions = torch.topk(logits, k=min(k, logits.numel()), sorted=False).indices
  wanted = target.face_index_a[0] if part == 0 else target.face_index_b[0]
  return int((positions == wanted).any())


def _empty_epoch_totals() -> dict[str, float]:
  return {
      "loss": 0.0, "salience": 0.0, "joint_ce": 0.0,
      "translation": 0.0, "rotation": 0.0,
      "face_a_r1": 0.0, "face_a_r8": 0.0,
      "face_b_r1": 0.0, "face_b_r8": 0.0,
      "joint_r1": 0.0, "joint_r5": 0.0,
      "program_r1": 0.0, "program_r5": 0.0,
  }


def _normalize_tail_gradients(
    model: torch.nn.Module, *, actual_views: int, configured_views: int,
) -> float:
  if not 1 <= actual_views <= configured_views:
    raise ValueError("joint v2 gradient accumulation view count differs")
  correction = configured_views / actual_views
  if correction != 1.0:
    for parameter in model.parameters():
      if parameter.grad is not None:
        parameter.grad.mul_(correction)
  return correction


def _run_streaming_epoch(
    model: JointInterfaceProgramLearnerV3, *, dataset: StreamingJointDatasetV2,
    groups: Sequence[tuple[str, tuple[int, ...]]],
    catalog: ExternallyFrozenCatalogSpecV3,
    optimizer: torch.optim.Optimizer | None, views_per_optimizer_step: int,
    gradient_clip_norm: float,
) -> dict[str, float | int]:
  if not groups:
    raise ValueError("joint v2 epoch split is empty")
  training = optimizer is not None
  model.train(training)
  totals = _empty_epoch_totals()
  examples = 0
  forward_calls = 0
  optimizer_steps = 0
  views_since_step = 0
  started = time.perf_counter()
  if optimizer is not None:
    optimizer.zero_grad(set_to_none=True)
  for group_index, (_input_hash, ordinals) in enumerate(groups):
    view = dataset.load_view(ordinals[0])
    with torch.set_grad_enabled(training):
      output = _forward_view_only(model, view, catalog)
      forward_calls += 1
      group_losses: list[torch.Tensor] = []
      for ordinal in ordinals:
        label = dataset.load_label(ordinal)
        target = _target_from_label(label, catalog_sha256=catalog.roster.catalog_sha256)
        loss = joint_interface_program_loss_v3(model, output, target)
        group_losses.append(loss.total)
        examples += 1
        for name, tensor in (
            ("loss", loss.total), ("salience", loss.mandatory_salience_ce),
            ("joint_ce", loss.joint_face_pair_program_ce),
            ("translation", loss.residual_translation_pair_local),
            ("rotation", loss.residual_rotation_pair_local),
        ):
          totals[name] += float(tensor.detach())
        totals["face_a_r1"] += _face_hit(output, target, 0, 1)
        totals["face_a_r8"] += _face_hit(output, target, 0, 8)
        totals["face_b_r1"] += _face_hit(output, target, 1, 1)
        totals["face_b_r8"] += _face_hit(output, target, 1, 8)
        totals["joint_r1"] += _joint_hit(output, target, 1)
        totals["joint_r5"] += _joint_hit(output, target, 5)
        totals["program_r1"] += _program_hit(output, label.target_program_index, 1)
        totals["program_r5"] += _program_hit(output, label.target_program_index, 5)
      if optimizer is not None:
        (torch.stack(group_losses).mean() / views_per_optimizer_step).backward()
        views_since_step += 1
        boundary = (
            (group_index + 1) % views_per_optimizer_step == 0
            or group_index + 1 == len(groups)
        )
        if boundary:
          if views_since_step < views_per_optimizer_step:
            _normalize_tail_gradients(
                model, actual_views=views_since_step,
                configured_views=views_per_optimizer_step,
            )
          torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
          optimizer.step()
          optimizer.zero_grad(set_to_none=True)
          optimizer_steps += 1
          views_since_step = 0
  elapsed = time.perf_counter() - started
  if forward_calls != len(groups):
    raise RuntimeError("joint v2 forwarded a view more than once")
  return {
      **{name: value / examples for name, value in totals.items()},
      "examples": examples, "unique_views": len(groups),
      "forward_calls": forward_calls, "optimizer_steps": optimizer_steps,
      "elapsed_seconds": elapsed,
      "examples_per_second": examples / max(elapsed, 1e-12),
      "views_per_second": forward_calls / max(elapsed, 1e-12),
  }


def model_selection_key(dev: Mapping[str, Any]) -> tuple[float, float, float, float]:
  """Frozen lexicographic rule; greater is better in every position."""
  return (
      float(dev["joint_r5"]), float(dev["program_r5"]),
      float(dev["joint_r1"]), -float(dev["loss"]),
  )


@dataclass(slots=True)
class BestEpochTrackerV2:
  patience: int
  best_epoch: int | None = None
  best_key: tuple[float, float, float, float] | None = None
  epochs_without_improvement: int = 0

  def observe(self, epoch: int, dev: Mapping[str, Any]) -> tuple[bool, bool]:
    key = model_selection_key(dev)
    improved = self.best_key is None or key > self.best_key
    if improved:
      self.best_epoch = epoch
      self.best_key = key
      self.epochs_without_improvement = 0
    else:
      self.epochs_without_improvement += 1
    stop = not improved and self.epochs_without_improvement >= self.patience
    return improved, stop


def _split_groups_for_smoke(
    dataset: StreamingJointDatasetV2, *, enable_holdout: bool,
) -> tuple[
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[str, tuple[int, ...]], ...], bool,
  ]:
  train = dataset.groups("train")
  dev = dataset.groups("dev")
  if dev or not enable_holdout:
    return train, dev, False
  families = sorted({dataset.index.rows[ordinals[0]].family_id for _, ordinals in train})
  if len(families) < 2:
    raise ValueError("joint v2 engineering holdout requires two real families")
  held_out = families[-1]
  train_result = tuple(
      group for group in train
      if dataset.index.rows[group[1][0]].family_id != held_out
  )
  dev_result = tuple(
      group for group in train
      if dataset.index.rows[group[1][0]].family_id == held_out
  )
  if not train_result or not dev_result:
    raise ValueError("joint v2 engineering family holdout differs")
  return train_result, dev_result, True


def _process_memory() -> tuple[int, int]:
  if platform.system() != "Windows":
    return 0, 0
  size_t = ctypes.c_size_t
  class Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", size_t), ("WorkingSetSize", size_t),
        ("QuotaPeakPagedPoolUsage", size_t), ("QuotaPagedPoolUsage", size_t),
        ("QuotaPeakNonPagedPoolUsage", size_t), ("QuotaNonPagedPoolUsage", size_t),
        ("PagefileUsage", size_t), ("PeakPagefileUsage", size_t),
    ]
  counters = Counters()
  counters.cb = ctypes.sizeof(counters)
  kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
  psapi = ctypes.WinDLL("psapi", use_last_error=True)
  kernel32.GetCurrentProcess.restype = wintypes.HANDLE
  psapi.GetProcessMemoryInfo.argtypes = (
      wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
  )
  psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
  if not psapi.GetProcessMemoryInfo(
      kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
  ):
    raise ctypes.WinError(ctypes.get_last_error())
  return int(counters.PeakWorkingSetSize), int(counters.WorkingSetSize)


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
  return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_joint_interface_v2(
    output_directory: str | Path, *, dataset: StreamingJointDatasetV2,
    config: JointInterfaceTrainingConfigV2, catalog_spec_path: str | Path,
    expected_catalog_roster_sha256: str, expected_catalog_spec_file_sha256: str,
    historical_producer_revision: str,
) -> str:
  output = Path(output_directory)
  if output.exists():
    raise FileExistsError("joint v2 training output already exists")
  root = Path(__file__).resolve().parent
  revision = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=root, text=True
  ).strip()
  dirty = subprocess.check_output(
      ["git", "status", "--porcelain", "--untracked-files=all"],
      cwd=root, text=True,
  ).strip()
  if _REVISION.fullmatch(revision) is None or dirty:
    raise ValueError("joint v2 training requires a clean exact revision before training")
  catalog = load_externally_frozen_catalog_spec_v3(
      catalog_spec_path, expected_catalog_roster_sha256=expected_catalog_roster_sha256,
      expected_spec_file_sha256=expected_catalog_spec_file_sha256,
  )
  if config.model_config.expected_catalog_roster_sha256 != catalog.roster.catalog_sha256:
    raise ValueError("joint v2 config/catalog binding differs")
  train_groups, dev_groups, smoke_remap = _split_groups_for_smoke(
      dataset, enable_holdout=config.smoke_dev_family_holdout
  )
  if not train_groups or not dev_groups:
    raise ValueError("joint v2 requires nonempty train and dev groups")
  torch.manual_seed(config.seed)
  np.random.seed(config.seed)
  torch.use_deterministic_algorithms(True, warn_only=False)
  torch.set_num_threads(1)
  cuda_observation = {
      "available": bool(torch.cuda.is_available()),
      "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
      "names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
      if torch.cuda.is_available() else [],
      "selected_device": "cpu",
      "policy": config.device_policy,
  }
  model = JointInterfaceProgramLearnerV3(config.model_config, seed=config.seed)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
  )
  tracker = BestEpochTrackerV2(patience=config.early_stop_patience)
  best_state: dict[str, torch.Tensor] | None = None
  history: list[dict[str, Any]] = []
  for epoch in range(config.max_epochs):
    train_metrics = _run_streaming_epoch(
        model, dataset=dataset, groups=train_groups, catalog=catalog,
        optimizer=optimizer, views_per_optimizer_step=config.views_per_optimizer_step,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    with torch.no_grad():
      dev_metrics = _run_streaming_epoch(
          model, dataset=dataset, groups=dev_groups, catalog=catalog,
          optimizer=None, views_per_optimizer_step=config.views_per_optimizer_step,
          gradient_clip_norm=config.gradient_clip_norm,
      )
    improved, stop = tracker.observe(epoch, dev_metrics)
    if improved:
      best_state = _clone_state(model)
    history.append({
        "epoch": epoch, "train": train_metrics, "dev": dev_metrics,
        "selection_key": list(model_selection_key(dev_metrics)),
        "improved": improved,
        "epochs_without_improvement": tracker.epochs_without_improvement,
        "early_stop_triggered": stop,
    })
    if stop:
      break
  if best_state is None or tracker.best_epoch is None or tracker.best_key is None:
    raise RuntimeError("joint v2 failed to select a best epoch")
  model.load_state_dict(best_state, strict=True)
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  try:
    checkpoint = temporary / "joint_interface_checkpoint_v3.pt"
    checkpoint_sha256 = save_joint_interface_checkpoint_v3(model.eval(), checkpoint, catalog_spec=catalog)
    loaded = load_joint_interface_checkpoint_v3(
        checkpoint, expected_checkpoint_sha256=checkpoint_sha256,
        catalog_spec_path=catalog_spec_path,
        expected_catalog_roster_sha256=expected_catalog_roster_sha256,
        expected_catalog_spec_file_sha256=expected_catalog_spec_file_sha256,
    )
    if loaded.config.sha256 != config.model_config.sha256:
      raise ValueError("joint v2 checkpoint replay config differs")
    peak_working_set, working_set = _process_memory()
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "scope": "train_dev_only_no_final_access",
        "final_test_touched": False,
        # Only the matched frozen evaluator may issue performance claims.
        "performance_claim_eligible": False,
        "engineering_smoke": config.engineering_smoke,
        "smoke_dev_family_remapped": smoke_remap,
        "data": {
            "safe_artifact_sha256": dataset.index.authority_bindings["safe_artifact_sha256"],
            "label_artifact_sha256": dataset.index.authority_bindings["label_artifact_sha256"],
            "input_domain_sha256": dataset.input_domain_sha256,
            "row_index_artifact_sha256": dataset.index.artifact_sha256,
            "row_index_payload_sha256": dataset.index.index_payload_sha256,
            "family_partition_suite_v2_artifact_sha256": dataset.index.authority_bindings[
                "family_partition_suite_v2_artifact_sha256"
            ],
            "historical_producer_revision": historical_producer_revision,
            "loaded_rows": dataset.row_count,
            "train_rows": sum(len(rows) for _, rows in train_groups),
            "dev_rows": sum(len(rows) for _, rows in dev_groups),
            "train_unique_views": len(train_groups), "dev_unique_views": len(dev_groups),
        },
        "catalog": {
            "roster_sha256": catalog.roster.catalog_sha256,
            "spec_file_sha256": catalog.spec_file_sha256,
            "spec_payload_sha256": catalog.spec_payload_sha256,
        },
        "training": {
            **{key: value for key, value in asdict(config).items() if key != "model_config"},
            "model_config": asdict(config.model_config),
            "model_config_sha256": config.model_config.sha256,
            "optimizer": "AdamW", "device_observation": cuda_observation,
            "streaming": "input_sha256_grouped_shard_local_lru.v2",
            "tail_gradient_normalization": "actual_views_per_optimizer_step.v1",
        },
        "model_selection": {
            "rule": ["max_dev_joint_r5", "max_dev_program_r5", "max_dev_joint_r1", "min_dev_loss"],
            "tie_policy": "earliest_epoch",
            "patience": config.early_stop_patience,
            "best_epoch": tracker.best_epoch,
            "best_selection_key": list(tracker.best_key),
            "saved_checkpoint_is_best_not_last": True,
            "epochs_executed": len(history),
        },
        "epochs": history,
        "checkpoint": {
            "schema_version": "joint_interface_program_checkpoint.v3",
            "name": checkpoint.name, "sha256": checkpoint_sha256,
            "optimizer_state_included": False,
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "windows_peak_working_set_bytes": peak_working_set,
            "windows_working_set_bytes": working_set,
        },
        "code": {"revision": revision, "source_sha256": _file_sha(__file__)},
    }
    receipt["receipt_payload_sha256"] = _sha(receipt)
    receipt_path = temporary / "training_receipt.json"
    receipt_path.write_bytes(_canonical_bytes(receipt))
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "files": [_binding(receipt_path), _binding(checkpoint)],
    }
    artifact["artifact_payload_sha256"] = _sha(artifact)
    artifact_path = temporary / "artifact.json"
    artifact_path.write_bytes(_canonical_bytes(artifact))
    pin = _file_sha(artifact_path)
    os.replace(temporary, output)
  except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
  return pin
