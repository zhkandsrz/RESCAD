"""Strict provenance helpers for benchmark-v2 training artifacts.

The benchmark-v2 model boundary is intentionally narrower than the legacy
JSONL format.  A formal dataset is a JSONL file plus an adjacent authenticated
manifest.  The manifest binds the frozen family split, producer code/config,
row schema, protocol, and exact JSONL bytes.  Model files separately bind the
actual numeric parameters, so editing weights cannot be hidden by recomputing
only the outer training-manifest hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import uuid

from .benchmark_v2_constants import PROTOCOL_VERSION


DATASET_MANIFEST_SCHEMA = "benchmark_v2_training_dataset_manifest.v1"
MODEL_TRAINING_MANIFEST_SCHEMA = "model_training_manifest.v2"
INTERFACE_TRAINING_ROW_SCHEMA = "benchmark_v2_interface_training_row.v1"
INTERFACE_TRAINING_ROW_KEYS = frozenset(
    {
        "schema_version",
        "model_input_protocol",
        "protocol_version",
        "source_split",
        "family_split_manifest_sha256",
        "feature_names_sha256",
        "producer_config_sha256",
        "case_token_sha256",
        "opaque_part",
        "opaque_interface",
        "benchmark_v2_model_view",
        "benchmark_v2_model_view_sha256",
        "label",
    }
)
INTERFACE_PAIR_TRAINING_ROW_SCHEMA = (
    "benchmark_v2_interface_pair_training_row.v1"
)
INTERFACE_PAIR_TRAINING_ROW_KEYS = frozenset(
    {
        "schema_version",
        "model_input_protocol",
        "protocol_version",
        "source_split",
        "family_split_manifest_sha256",
        "feature_names_sha256",
        "producer_config_sha256",
        "case_token_sha256",
        "opaque_pair",
        "opaque_interface_a",
        "opaque_interface_b",
        "interface_a",
        "interface_b",
        "relation_hint",
        "contact_type",
        "label",
    }
)
DATASET_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "row_schema",
        "model_input_protocol",
        "protocol_version",
        "source_split",
        "family_split_manifest_sha256",
        "source_split_artifact_sha256",
        "feature_names_sha256",
        "seed",
        "producer_kind",
        "producer_config",
        "producer_config_sha256",
        "producer_code_sha256",
        "data_sha256",
        "row_count",
        "case_count",
        "positive_rows",
        "negative_rows",
        "case_token_sha256",
        "manifest_sha256",
    }
)
FORMAL_MODEL_TRAINING_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "model_schema",
        "model_input_protocol",
        "protocol_version",
        "feature_names_sha256",
        "train_jsonl_sha256",
        "dev_jsonl_sha256",
        "train_rows",
        "dev_rows",
        "train_case_count",
        "dev_case_count",
        "train_positive_rows",
        "dev_positive_rows",
        "optimizer",
        "training_config_sha256",
        "model_parameters_sha256",
        "frozen_family_train_split_sha256",
        "frozen_family_dev_split_sha256",
        "family_split_manifest_sha256",
        "train_dataset_manifest_sha256",
        "dev_dataset_manifest_sha256",
        "train_source_split",
        "dev_source_split",
        "trainer_code_sha256",
        "manifest_sha256",
    }
)
FORMAL_CONTACT_TYPES = frozenset(
    {
        "none",
        "seat_plane",
        "insert_axis",
        "shaft_in_bore",
        "screw_in_hole",
        "pin_in_slot",
        "boss_in_slot",
        "threaded_interference",
        "generic_contact",
    }
)
FUSION_STEP_FACE_MAP_BINDING_SCHEMA = (
    "benchmark_v2_fusion_step_face_map_training_binding.v2"
)
FUSION_STEP_FACE_MAP_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "binding_mode",
        "receipt_sha256",
        "receipt_payload_sha256",
        "endpoint_lookup_schema_version",
        "part_semantics",
        "family_source_sha256",
        "mapped_case_set_sha256",
        "formal_gold_eligible",
        "gold_interface_certificate_ready",
    }
)
PRIVATE_SUPERVISION_BINDING_SCHEMA = (
    "benchmark_v2_private_supervision_training_binding.v1"
)
PRIVATE_SUPERVISION_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "source_schema_version",
        "gold_split_schema_version",
        "gold_contact_schema_version",
        "split",
        "case_count",
        "case_set_sha256",
        "source_payload_sha256",
        "source_file_sha256",
        "gold_payload_sha256",
        "gold_file_sha256",
        "family_split_manifest_sha256",
        "gold_label_semantics",
        "gold_interface_certificate_ready",
        "certified_negative_complete",
    }
)
FORMAL_BINARY_BLOCKING_REASON = (
    "no_authenticated_allowlisted_contact_census_schema"
)
FORMAL_CONTACT_CENSUS_SCHEMA_ALLOWLIST: frozenset[str] = frozenset()
_AUTHENTICATED_RECEIPT_FACTORY_TOKEN = object()


def require_formal_binary_training_available() -> None:
  """Hard gate formal binary training until a real census verifier exists."""

  raise ValueError(
      "formal binary training is unavailable: no authenticated, allowlisted "
      "contact census schema is implemented; receipt object types are not authority"
  )


@dataclass(frozen=True, slots=True)
class TrainingDatasetBinding:
  """Validated identity of one immutable benchmark-v2 JSONL artifact."""

  data_sha256: str
  manifest_sha256: str
  family_split_manifest_sha256: str
  source_split_artifact_sha256: str
  source_split: str
  row_schema: str
  feature_names_sha256: str
  row_count: int
  case_token_sha256: tuple[str, ...]
  producer_config_sha256: str
  producer_code_sha256: str
  positive_rows: int
  negative_rows: int
  case_count: int


@dataclass(frozen=True, slots=True)
class CapturedFileArtifact:
  """Immutable bytes captured from one stable, plain filesystem path."""

  path: Path
  resolved_path: Path
  raw_bytes: bytes = field(repr=False)
  sha256: str
  byte_count: int
  device: int
  inode: int
  link_count: int


@dataclass(frozen=True, slots=True)
class CapturedJsonArtifact:
  """A JSON payload parsed from exactly the bytes represented by ``sha256``."""

  path: Path
  resolved_path: Path
  raw_bytes: bytes = field(repr=False)
  sha256: str
  byte_count: int
  device: int
  inode: int
  link_count: int
  payload: Any


class BenchmarkV2AuthenticatedTrainingReceipts:
  """Ephemeral proof that training inputs were loaded from captured files.

  This object is intentionally non-serializable. A JSON Mapping containing the
  same strings is provenance metadata, not authentication, and cannot unlock a
  formal writer.
  """

  __slots__ = (
      "source_split",
      "case_ids",
      "case_token_sha256",
      "family_split_manifest_sha256",
      "public_cases_file_sha256",
      "private_supervision_binding",
      "face_map_binding",
      "fully_mapped_case_ids",
      "contact_census_binding",
      "formal_binary_ready",
      "blocking_reason",
  )

  def __init__(
      self,
      *,
      source_split: str,
      case_ids: Sequence[str],
      family_split_manifest_sha256: str,
      public_cases_file_sha256: str,
      private_supervision_binding: Mapping[str, Any],
      face_map_binding: Mapping[str, Any],
      fully_mapped_case_ids: Sequence[str],
      contact_census_binding: Mapping[str, Any] | None,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _AUTHENTICATED_RECEIPT_FACTORY_TOKEN:
      raise TypeError(
          "BenchmarkV2AuthenticatedTrainingReceipts must be created by the "
          "path-authenticated receipt loader"
      )
    split = _normalized_split(source_split)
    normalized_case_ids = tuple(str(value).strip() for value in case_ids)
    if (
        not normalized_case_ids
        or any(not value for value in normalized_case_ids)
        or len(set(normalized_case_ids)) != len(normalized_case_ids)
    ):
      raise ValueError("authenticated training receipt case IDs are invalid")
    family_sha = _require_sha256(
        family_split_manifest_sha256,
        name="family_split_manifest_sha256",
    )
    self.source_split = split
    self.case_ids = normalized_case_ids
    self.case_token_sha256 = tuple(
        sorted(
            case_token_sha256(
                family_manifest_sha256=family_sha,
                case_id=case_id,
            )
            for case_id in normalized_case_ids
        )
    )
    self.family_split_manifest_sha256 = family_sha
    self.public_cases_file_sha256 = _require_sha256(
        public_cases_file_sha256,
        name="public_cases_file_sha256",
    )
    self.private_supervision_binding = MappingProxyType(
        validate_private_supervision_binding(
            private_supervision_binding,
            require_certified_binary=False,
        )
    )
    self.face_map_binding = MappingProxyType(
        validate_face_map_binding(
            face_map_binding,
            require_private_receipt=False,
        )
    )
    normalized_fully_mapped = tuple(
        sorted(str(value).strip() for value in fully_mapped_case_ids)
    )
    if (
        any(not value for value in normalized_fully_mapped)
        or len(set(normalized_fully_mapped)) != len(normalized_fully_mapped)
        or not set(normalized_fully_mapped).issubset(normalized_case_ids)
    ):
      raise ValueError("authenticated fully mapped case IDs are invalid")
    self.fully_mapped_case_ids = normalized_fully_mapped
    expected_case_set_sha = canonical_sha256(sorted(normalized_case_ids))
    private_binding = self.private_supervision_binding
    if (
        private_binding["split"] != split
        or private_binding["case_count"] != len(normalized_case_ids)
        or private_binding["case_set_sha256"] != expected_case_set_sha
        or private_binding["family_split_manifest_sha256"] != family_sha
    ):
      raise ValueError(
          "authenticated private supervision does not close over the public case set"
      )
    if self.face_map_binding["mapped_case_set_sha256"] != canonical_sha256(
        list(normalized_fully_mapped)
    ):
      raise ValueError(
          "authenticated face-map binding differs from its fully mapped case set"
      )
    self.contact_census_binding = (
        None
        if contact_census_binding is None
        else MappingProxyType(dict(contact_census_binding))
    )
    self.formal_binary_ready = False
    self.blocking_reason = FORMAL_BINARY_BLOCKING_REASON

  def __repr__(self) -> str:
    return (
        "<BenchmarkV2AuthenticatedTrainingReceipts "
        f"split={self.source_split!r} cases={len(self.case_ids)} "
        f"formal_binary_ready={self.formal_binary_ready}>"
    )

  def __reduce_ex__(self, _protocol):
    raise TypeError("authenticated training receipts are not serializable")

  def require_formal_binary_rows(
      self,
      rows: Sequence[Mapping[str, Any]],
      *,
      source_split: str,
      family_split_manifest_sha256: str,
      producer_config: Mapping[str, Any],
      public_cases_file_sha256: str,
  ) -> None:
    """Validate all closed-set bindings, then report the unavailable census."""

    require_formal_binary_training_available()
    if _normalized_split(source_split) != self.source_split:
      raise ValueError("authenticated receipt split differs from dataset rows")
    family_sha = _require_sha256(
        family_split_manifest_sha256,
        name="family_split_manifest_sha256",
    )
    if family_sha != self.family_split_manifest_sha256:
      raise ValueError("authenticated family manifest binding differs")
    if _require_sha256(
        public_cases_file_sha256,
        name="family split public cases sha256",
    ) != self.public_cases_file_sha256:
      raise ValueError("authenticated public cases artifact binding differs")
    if any(not isinstance(row, Mapping) for row in rows):
      raise ValueError("formal dataset rows must be JSON objects")
    row_tokens = tuple(
        sorted(
            {
                _require_sha256(row.get("case_token_sha256"), name="case token")
                for row in rows
            }
        )
    )
    if row_tokens != self.case_token_sha256:
      raise ValueError(
          "dataset row case-token set does not equal the authenticated case set"
      )
    if set(self.fully_mapped_case_ids) != set(self.case_ids):
      raise ValueError(
          "face-map fully mapped case set does not equal the authenticated case set"
      )
    if not isinstance(producer_config, Mapping):
      raise ValueError("formal producer config must be an object")
    interface_config = producer_config
    if "interface_producer_config" in producer_config:
      raw_interface_config = producer_config.get("interface_producer_config")
      if not isinstance(raw_interface_config, Mapping):
        raise ValueError("pair producer lacks an interface producer config")
      interface_config = raw_interface_config
    raw_private_binding = interface_config.get("private_supervision_binding")
    if not isinstance(raw_private_binding, Mapping) or dict(
        raw_private_binding
    ) != dict(self.private_supervision_binding):
      raise ValueError("producer private supervision binding is not authenticated")
    raw_face_binding = interface_config.get("fusion_step_face_map_binding")
    if not isinstance(raw_face_binding, Mapping) or dict(raw_face_binding) != dict(
        self.face_map_binding
    ):
      raise ValueError("producer face-map binding is not authenticated")
    raise ValueError(
        "formal binary training is explicitly unavailable: no authenticated, "
        "allowlisted contact census receipt schema is implemented"
    )


def _new_authenticated_training_receipts(
    *,
    source_split: str,
    case_ids: Sequence[str],
    family_split_manifest_sha256: str,
    public_cases_file_sha256: str,
    private_supervision_binding: Mapping[str, Any],
    face_map_binding: Mapping[str, Any],
    fully_mapped_case_ids: Sequence[str],
    contact_census_binding: Mapping[str, Any] | None = None,
) -> BenchmarkV2AuthenticatedTrainingReceipts:
  return BenchmarkV2AuthenticatedTrainingReceipts(
      source_split=source_split,
      case_ids=case_ids,
      family_split_manifest_sha256=family_split_manifest_sha256,
      public_cases_file_sha256=public_cases_file_sha256,
      private_supervision_binding=private_supervision_binding,
      face_map_binding=face_map_binding,
      fully_mapped_case_ids=fully_mapped_case_ids,
      contact_census_binding=contact_census_binding,
      _factory_token=_AUTHENTICATED_RECEIPT_FACTORY_TOKEN,
  )


def canonical_sha256(payload: Any) -> str:
  canonical = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  )
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _stat_identity(value: Any) -> tuple[int, int, int, int, int, int]:
  return (
      int(value.st_dev),
      int(value.st_ino),
      int(value.st_size),
      int(value.st_mtime_ns),
      int(value.st_ctime_ns),
      int(value.st_nlink),
  )


def _file_object_identity(value: Any) -> tuple[int, int]:
  return (int(value.st_dev), int(value.st_ino))


def capture_file_artifact(
    path: str | Path,
    *,
    label: str,
) -> CapturedFileArtifact:
  """Capture one file once and reject path replacement during the capture."""

  candidate = Path(path)
  try:
    if candidate.is_symlink() or not candidate.is_file():
      raise ValueError(f"{label} must be an existing plain file")
    resolved = candidate.resolve(strict=True)
    before = candidate.stat()
    if int(before.st_nlink) != 1:
      raise ValueError(f"{label} must not be hard-linked")
    with candidate.open("rb") as stream:
      opened_before = os.fstat(stream.fileno())
      raw_bytes = stream.read()
      opened_after = os.fstat(stream.fileno())
    after = candidate.stat()
    if (
        candidate.is_symlink()
        or candidate.resolve(strict=True) != resolved
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(opened_before) != _stat_identity(opened_after)
        or _file_object_identity(before) != _file_object_identity(opened_before)
        or _file_object_identity(opened_after) != _file_object_identity(after)
        or len(raw_bytes) != int(opened_after.st_size)
        or any(
            int(value.st_nlink) != 1
            for value in (opened_before, opened_after, after)
        )
    ):
      raise ValueError(f"{label} changed while being captured")
  except ValueError:
    raise
  except OSError as error:
    raise ValueError(f"{label} is missing or unreadable") from error
  return CapturedFileArtifact(
      path=candidate,
      resolved_path=resolved,
      raw_bytes=raw_bytes,
      sha256=hashlib.sha256(raw_bytes).hexdigest(),
      byte_count=len(raw_bytes),
      device=int(after.st_dev),
      inode=int(after.st_ino),
      link_count=int(after.st_nlink),
  )


def reverify_captured_file_artifact(
    captured: CapturedFileArtifact,
    *,
    label: str,
) -> None:
  """Require a captured path to still denote the same one-link file and bytes."""

  if not isinstance(captured, CapturedFileArtifact):
    raise TypeError("reverification requires a captured file artifact")
  try:
    current = capture_file_artifact(captured.path, label=label)
  except (OSError, ValueError) as error:
    raise ValueError(f"{label} changed since capture") from error
  if (
      current.resolved_path != captured.resolved_path
      or current.device != captured.device
      or current.inode != captured.inode
      or current.link_count != captured.link_count
      or current.byte_count != captured.byte_count
      or current.sha256 != captured.sha256
  ):
    raise ValueError(f"{label} changed since capture")


def capture_json_artifact(
    path: str | Path,
    *,
    label: str,
) -> CapturedJsonArtifact:
  """Parse JSON from a single immutable byte capture."""

  captured = capture_file_artifact(path, label=label)
  try:
    payload = json.loads(captured.raw_bytes.decode("utf-8"))
  except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not valid UTF-8 JSON") from error
  return CapturedJsonArtifact(
      path=captured.path,
      resolved_path=captured.resolved_path,
      raw_bytes=captured.raw_bytes,
      sha256=captured.sha256,
      byte_count=captured.byte_count,
      device=captured.device,
      inode=captured.inode,
      link_count=captured.link_count,
      payload=payload,
  )


def dataset_manifest_path(data_path: str | Path) -> Path:
  return Path(data_path).with_suffix(".manifest.json")


def _code_bundle_sha256(paths: Iterable[str | Path]) -> str:
  rows = []
  for path in sorted({Path(value).resolve() for value in paths}, key=str):
    if not path.is_file():
      raise ValueError(f"benchmark_v2 code identity source is missing: {path.name}")
    rows.append({"name": path.name, "sha256": file_sha256(path)})
  return canonical_sha256(rows)


def producer_code_sha256(kind: str) -> str:
  root = Path(__file__).resolve().parent
  common = (
      root / "benchmark_v2_training_provenance.py",
      root / "benchmark_v2_protocol.py",
      root / "benchmark_v2_private_supervision.py",
      root / "benchmark_v2_model_view.py",
      root / "build_interface_dataset.py",
  )
  normalized = str(kind).strip().lower()
  if normalized == "interface":
    return _code_bundle_sha256(common)
  if normalized == "pair":
    return _code_bundle_sha256((*common, root / "build_interface_pair_dataset.py"))
  raise ValueError(f"unknown benchmark_v2 producer kind: {kind!r}")


def trainer_code_sha256(kind: str) -> str:
  root = Path(__file__).resolve().parent
  common = (
      root / "benchmark_v2_training_provenance.py",
      root / "benchmark_v2_model_view.py",
  )
  normalized = str(kind).strip().lower()
  if normalized == "interface":
    return _code_bundle_sha256((*common, root / "interface_scorer.py"))
  if normalized == "pair":
    return _code_bundle_sha256(
        (*common, root / "interface_pair_scorer.py", root / "interface_pair_features.py")
    )
  raise ValueError(f"unknown benchmark_v2 trainer kind: {kind!r}")


def _require_sha256(value: Any, *, name: str) -> str:
  text = str(value or "").strip().lower()
  if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
    raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
  return text


def _normalized_split(value: Any, *, allow_dev: bool = True) -> str:
  split = str(value or "").strip().lower()
  allowed = {"train", "dev"} if allow_dev else {"train"}
  if split not in allowed:
    raise ValueError(
        "benchmark_v2 training source_split must be " + " or ".join(sorted(allowed))
    )
  return split


def validate_face_map_binding(
    value: Mapping[str, Any] | None,
    *,
    require_private_receipt: bool,
) -> dict[str, Any]:
  """Validate the redacted public binding for a private face-map receipt."""

  if not isinstance(value, Mapping) or set(value) != set(
      FUSION_STEP_FACE_MAP_BINDING_KEYS
  ):
    raise ValueError("benchmark_v2 requires a fixed Fusion STEP face-map binding")
  binding = dict(value)
  if binding.get("schema_version") != FUSION_STEP_FACE_MAP_BINDING_SCHEMA:
    raise ValueError("Fusion STEP face-map binding schema is invalid")
  mode = str(binding.get("binding_mode") or "")
  if mode not in {
      "private_development_receipt",
      "private_formal_receipt",
      "development_explicit_mapping",
  }:
    raise ValueError("Fusion STEP face-map binding mode is invalid")
  for key in (
      "receipt_sha256",
      "receipt_payload_sha256",
      "family_source_sha256",
      "mapped_case_set_sha256",
  ):
    binding[key] = _require_sha256(binding.get(key), name=key)
  if binding.get("endpoint_lookup_schema_version") != (
      "fusion_step_face_endpoint_lookup.v2"
  ) or binding.get("part_semantics") != "assembly_body_instance":
    raise ValueError("Fusion STEP face-map lookup contract is invalid")
  for key in ("formal_gold_eligible", "gold_interface_certificate_ready"):
    if not isinstance(binding.get(key), bool):
      raise ValueError(f"Fusion STEP face-map binding {key} must be boolean")
  formal_ready = bool(
      binding["formal_gold_eligible"]
      and binding["gold_interface_certificate_ready"]
  )
  if formal_ready and mode != "private_formal_receipt":
    raise ValueError("formal-ready face-map binding has the wrong mode")
  if require_private_receipt and (
      mode != "private_formal_receipt" or not formal_ready
  ):
    raise ValueError(
        "formal benchmark_v2 requires a formal-ready private face-map receipt"
    )
  return binding


def validate_private_supervision_binding(
    value: Mapping[str, Any] | None,
    *,
    require_certified_binary: bool,
) -> dict[str, Any]:
  """Validate the redacted binding to custodian source/gold split sidecars."""

  if require_certified_binary:
    require_formal_binary_training_available()
  if not isinstance(value, Mapping) or set(value) != set(
      PRIVATE_SUPERVISION_BINDING_KEYS
  ):
    raise ValueError("benchmark_v2 requires a fixed private supervision binding")
  binding = dict(value)
  expected = {
      "schema_version": PRIVATE_SUPERVISION_BINDING_SCHEMA,
      "source_schema_version": "benchmark_v2_private_source_split.v1",
      "gold_split_schema_version": (
          "benchmark_v2_private_evaluation_gold_split.v1"
      ),
  }
  if any(binding.get(key) != expected_value for key, expected_value in expected.items()):
    raise ValueError("benchmark_v2 private supervision binding schema is invalid")
  split = str(binding.get("split") or "")
  if split not in {"train", "dev"}:
    raise ValueError("benchmark_v2 private supervision split is invalid")
  case_count = binding.get("case_count")
  if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
    raise ValueError("benchmark_v2 private supervision case_count is invalid")
  for key in (
      "case_set_sha256",
      "source_payload_sha256",
      "source_file_sha256",
      "gold_payload_sha256",
      "gold_file_sha256",
      "family_split_manifest_sha256",
  ):
    binding[key] = _require_sha256(binding.get(key), name=key)
  if not isinstance(binding.get("gold_contact_schema_version"), str) or not str(
      binding["gold_contact_schema_version"]
  ).strip():
    raise ValueError("benchmark_v2 private gold contact schema is invalid")
  for key in ("gold_interface_certificate_ready", "certified_negative_complete"):
    if not isinstance(binding.get(key), bool):
      raise ValueError(f"benchmark_v2 private supervision {key} must be boolean")
  semantics = str(binding.get("gold_label_semantics") or "")
  if semantics not in {
      "source_positive_assertions_only",
      "closed_world_certified_binary",
  }:
    raise ValueError("benchmark_v2 private gold label semantics are invalid")
  return binding


def _validate_formal_face_map_producer_config(
    config: Any,
    *,
    producer_kind: str,
) -> dict[str, Any]:
  if not isinstance(config, Mapping):
    raise ValueError("benchmark_v2 producer_config must be an object")
  normalized_kind = str(producer_kind).strip().lower()
  if normalized_kind == "interface":
    binding = config.get("fusion_step_face_map_binding")
    supervision_binding = config.get("private_supervision_binding")
  elif normalized_kind == "pair":
    interface_config = config.get("interface_producer_config")
    binding = (
        interface_config.get("fusion_step_face_map_binding")
        if isinstance(interface_config, Mapping)
        else None
    )
    supervision_binding = (
        interface_config.get("private_supervision_binding")
        if isinstance(interface_config, Mapping)
        else None
    )
  else:
    raise ValueError("benchmark_v2 producer kind is invalid")
  validate_face_map_binding(binding, require_private_receipt=True)
  private_binding = validate_private_supervision_binding(
      supervision_binding,
      require_certified_binary=True,
  )
  return private_binding


def validate_family_split_binding(
    *,
    cases_path: str | Path,
    manifest_path: str | Path,
    source_split: str,
    case_ids: Sequence[str],
    _cases_capture: CapturedFileArtifact | None = None,
    _manifest_capture: CapturedJsonArtifact | None = None,
    pilot_only: bool = False,
) -> str:
  """Validate that cases are the exact train/dev artifact in a clean family split."""

  split = _normalized_split(source_split)
  manifest_capture = _manifest_capture or capture_json_artifact(
      manifest_path,
      label="family split manifest",
  )
  cases_capture = _cases_capture or capture_file_artifact(
      cases_path,
      label=f"family split {split} cases",
  )
  data = manifest_capture.payload
  if not isinstance(data, Mapping):
    raise ValueError("family split manifest must contain a JSON object")
  expected_identity = (
      data.get("formal") is False
      and data.get("pilot_only") is True
      and data.get("development") is True
      if pilot_only
      else data.get("formal") is True and data.get("pilot_only") is not True
  )
  if data.get("benchmark") != "benchmark_v2" or not expected_identity:
    qualifier = "development pilot" if pilot_only else "formal"
    raise ValueError(
        f"benchmark_v2 requires an explicitly identified {qualifier} family split"
    )
  if data.get("strict_hygiene_clean") is not True:
    raise ValueError("benchmark_v2 family split is not strict_hygiene_clean")
  artifact_hashes = data.get("artifact_hashes")
  artifact = artifact_hashes.get(split) if isinstance(artifact_hashes, Mapping) else None
  if not isinstance(artifact, Mapping):
    raise ValueError(f"family split manifest does not bind the {split} artifact")
  expected_data_sha = str(artifact.get("sha256") or "").removeprefix("sha256:")
  expected_data_sha = _require_sha256(
      expected_data_sha,
      name=f"family split {split} artifact sha256",
  )
  if cases_capture.sha256 != expected_data_sha:
    raise ValueError(f"cases JSON does not match frozen family {split} artifact")

  records = data.get("family_case_records")
  if not isinstance(records, list):
    raise ValueError("formal family split manifest lacks family_case_records")
  normalized_case_ids = [str(case_id).strip() for case_id in case_ids]
  if (
      not normalized_case_ids
      or any(not case_id for case_id in normalized_case_ids)
      or len(set(normalized_case_ids)) != len(normalized_case_ids)
  ):
    raise ValueError("frozen family split cases must be nonempty and unique")
  counts = data.get("split_case_counts")
  split_count = counts.get(split) if isinstance(counts, Mapping) else None
  if (
      isinstance(split_count, bool)
      or not isinstance(split_count, int)
      or split_count != len(normalized_case_ids)
  ):
    raise ValueError("cases JSON does not contain the complete frozen family split")
  manifest_split_case_ids = [
      str(row.get("case_id") or "").strip()
      for row in records
      if isinstance(row, Mapping) and str(row.get("split") or "") == split
  ]
  if (
      any(not case_id for case_id in manifest_split_case_ids)
      or len(set(manifest_split_case_ids)) != len(manifest_split_case_ids)
      or set(manifest_split_case_ids) != set(normalized_case_ids)
  ):
    raise ValueError(
        f"cases JSON case set differs from the frozen family {split} split"
    )
  return manifest_capture.sha256


def case_token_sha256(*, family_manifest_sha256: str, case_id: str) -> str:
  family_sha = _require_sha256(
      family_manifest_sha256,
      name="family_split_manifest_sha256",
  )
  if not str(case_id).strip():
    raise ValueError("benchmark_v2 producer requires a nonempty case id")
  return hashlib.sha256(
      f"benchmark_v2_case\0{family_sha}\0{case_id}".encode("utf-8")
  ).hexdigest()


def write_dataset_manifest(
    data_path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    row_schema: str,
    feature_names: Sequence[str],
    source_split: str,
    family_split_manifest: str | Path,
    seed: int,
    producer_config: Mapping[str, Any],
    producer_kind: str,
    authenticated_receipts: BenchmarkV2AuthenticatedTrainingReceipts | None = None,
    _family_manifest_capture: CapturedJsonArtifact | None = None,
) -> dict[str, Any]:
  """Write and return the authenticated sidecar for an already-written JSONL."""

  require_formal_binary_training_available()
  path = Path(data_path)
  split = _normalized_split(source_split)
  family_capture = _family_manifest_capture or capture_json_artifact(
      family_split_manifest,
      label="family split manifest",
  )
  data_capture = capture_file_artifact(path, label="benchmark_v2 dataset")
  family_sha = family_capture.sha256
  family_data = family_capture.payload
  artifact_hashes = (
      family_data.get("artifact_hashes")
      if isinstance(family_data, Mapping)
      else None
  )
  split_artifact = (
      artifact_hashes.get(split) if isinstance(artifact_hashes, Mapping) else None
  )
  if not isinstance(split_artifact, Mapping):
    raise ValueError(f"family split manifest does not bind the {split} artifact")
  source_artifact_sha = _require_sha256(
      str(split_artifact.get("sha256") or "").removeprefix("sha256:"),
      name=f"family split {split} artifact sha256",
  )
  if not isinstance(
      authenticated_receipts,
      BenchmarkV2AuthenticatedTrainingReceipts,
  ):
    raise ValueError(
        "formal benchmark_v2 writers require a path-authenticated receipt set; "
        "caller-supplied receipt Mappings are not accepted"
    )
  authenticated_receipts.require_formal_binary_rows(
      rows,
      source_split=split,
      family_split_manifest_sha256=family_sha,
      producer_config=producer_config,
      public_cases_file_sha256=source_artifact_sha,
  )
  feature_hash = canonical_sha256(list(feature_names))
  config = json.loads(
      json.dumps(producer_config, sort_keys=True, allow_nan=False)
  )
  private_binding = _validate_formal_face_map_producer_config(
      config,
      producer_kind=producer_kind,
  )
  if private_binding["family_split_manifest_sha256"] != family_sha:
    raise ValueError(
        "private supervision binding differs from the frozen family manifest"
    )
  config_sha = canonical_sha256(config)
  code_sha = producer_code_sha256(producer_kind)
  try:
    labels = [int(row.get("label")) for row in rows]
  except (TypeError, ValueError) as error:
    raise ValueError("benchmark_v2 dataset rows require binary labels") from error
  tokens = sorted({str(row.get("case_token_sha256") or "") for row in rows})
  if not rows or any(label not in {0, 1} for label in labels):
    raise ValueError("benchmark_v2 dataset requires nonempty binary-labeled rows")
  if set(labels) != {0, 1}:
    raise ValueError("benchmark_v2 dataset requires both positive and negative classes")
  for token in tokens:
    _require_sha256(token, name="case_token_sha256")
  for index, row in enumerate(rows):
    expected_keys = (
        INTERFACE_TRAINING_ROW_KEYS
        if row_schema == INTERFACE_TRAINING_ROW_SCHEMA
        else INTERFACE_PAIR_TRAINING_ROW_KEYS
        if row_schema == INTERFACE_PAIR_TRAINING_ROW_SCHEMA
        else None
    )
    if expected_keys is None or set(row) != set(expected_keys):
      raise ValueError(f"row {index} has the wrong benchmark_v2 fixed schema")
    if row.get("schema_version") != row_schema:
      raise ValueError(f"row {index} has the wrong benchmark_v2 row schema")
    if row.get("source_split") != split:
      raise ValueError(f"row {index} has the wrong benchmark_v2 source split")
    if row.get("family_split_manifest_sha256") != family_sha:
      raise ValueError(f"row {index} has the wrong frozen family split binding")
    if row.get("feature_names_sha256") != feature_hash:
      raise ValueError(f"row {index} has the wrong fixed feature schema binding")
    if row.get("producer_config_sha256") != config_sha:
      raise ValueError(f"row {index} has the wrong producer config binding")

  payload: dict[str, Any] = {
      "schema_version": DATASET_MANIFEST_SCHEMA,
      "row_schema": row_schema,
      "model_input_protocol": "benchmark_v2",
      "protocol_version": PROTOCOL_VERSION,
      "source_split": split,
      "family_split_manifest_sha256": family_sha,
      "source_split_artifact_sha256": source_artifact_sha,
      "feature_names_sha256": feature_hash,
      "seed": int(seed),
      "producer_kind": str(producer_kind),
      "producer_config": config,
      "producer_config_sha256": config_sha,
      "producer_code_sha256": code_sha,
      "data_sha256": data_capture.sha256,
      "row_count": len(rows),
      "case_count": len(tokens),
      "positive_rows": sum(label == 1 for label in labels),
      "negative_rows": sum(label == 0 for label in labels),
      "case_token_sha256": tokens,
  }
  payload["manifest_sha256"] = canonical_sha256(payload)
  manifest_path = dataset_manifest_path(path)
  manifest_path.write_text(
      json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
      encoding="utf-8",
  )
  return payload


def write_dataset_artifact(
    data_path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    row_schema: str,
    feature_names: Sequence[str],
    source_split: str,
    family_split_manifest: str | Path,
    seed: int,
    producer_config: Mapping[str, Any],
    producer_kind: str,
    authenticated_receipts: BenchmarkV2AuthenticatedTrainingReceipts | None = None,
) -> dict[str, Any]:
  """Stage JSONL+manifest, publishing the manifest only after data replacement."""

  require_formal_binary_training_available()
  path = Path(data_path)
  split = _normalized_split(source_split)
  family_capture = capture_json_artifact(
      family_split_manifest,
      label="family split manifest",
  )
  family_data = family_capture.payload
  artifact_hashes = (
      family_data.get("artifact_hashes")
      if isinstance(family_data, Mapping)
      else None
  )
  split_artifact = (
      artifact_hashes.get(split) if isinstance(artifact_hashes, Mapping) else None
  )
  if not isinstance(split_artifact, Mapping):
    raise ValueError(f"family split manifest does not bind the {split} artifact")
  source_artifact_sha = _require_sha256(
      str(split_artifact.get("sha256") or "").removeprefix("sha256:"),
      name=f"family split {split} artifact sha256",
  )
  if not isinstance(
      authenticated_receipts,
      BenchmarkV2AuthenticatedTrainingReceipts,
  ):
    raise ValueError(
        "formal benchmark_v2 writers require a path-authenticated receipt set; "
        "caller-supplied receipt Mappings are not accepted"
    )
  authenticated_receipts.require_formal_binary_rows(
      rows,
      source_split=split,
      family_split_manifest_sha256=family_capture.sha256,
      producer_config=producer_config,
      public_cases_file_sha256=source_artifact_sha,
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
  staging_manifest = dataset_manifest_path(staging)
  final_manifest = dataset_manifest_path(path)
  for candidate in (staging, staging_manifest):
    if candidate.exists():
      candidate.unlink()
  try:
    staging.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    payload = write_dataset_manifest(
        staging,
        rows=rows,
        row_schema=row_schema,
        feature_names=feature_names,
        source_split=source_split,
        family_split_manifest=family_split_manifest,
        seed=seed,
        producer_config=producer_config,
        producer_kind=producer_kind,
        authenticated_receipts=authenticated_receipts,
        _family_manifest_capture=family_capture,
    )
    staging.replace(path)
    staging_manifest.replace(final_manifest)
    return payload
  finally:
    for candidate in (staging, staging_manifest):
      if candidate.exists():
        candidate.unlink()


def validate_dataset_manifest(
    data_path: str | Path,
    *,
    expected_row_schema: str,
    expected_feature_names: Sequence[str],
    required_split: str,
    producer_kind: str,
) -> TrainingDatasetBinding:
  """Fail closed unless JSONL and sidecar form one current formal artifact."""

  require_formal_binary_training_available()
  path = Path(data_path)
  manifest_path = dataset_manifest_path(path)
  manifest_capture = capture_json_artifact(
      manifest_path,
      label="benchmark_v2 dataset manifest",
  )
  data_capture = capture_file_artifact(
      path,
      label="benchmark_v2 dataset",
  )
  data = manifest_capture.payload
  if not isinstance(data, Mapping):
    raise ValueError("benchmark_v2 dataset manifest must be a JSON object")
  if set(data) != set(DATASET_MANIFEST_KEYS):
    raise ValueError("benchmark_v2 dataset manifest schema mismatch")
  unsigned = dict(data)
  stored_hash = unsigned.pop("manifest_sha256", None)
  if stored_hash != canonical_sha256(unsigned):
    raise ValueError("benchmark_v2 dataset manifest hash mismatch")
  expected_feature_hash = canonical_sha256(list(expected_feature_names))
  required = {
      "schema_version": DATASET_MANIFEST_SCHEMA,
      "row_schema": expected_row_schema,
      "model_input_protocol": "benchmark_v2",
      "protocol_version": PROTOCOL_VERSION,
      "source_split": _normalized_split(required_split),
      "feature_names_sha256": expected_feature_hash,
      "producer_kind": producer_kind,
      "producer_code_sha256": producer_code_sha256(producer_kind),
  }
  for key, value in required.items():
    if data.get(key) != value:
      raise ValueError(f"benchmark_v2 dataset manifest {key} mismatch")
  if data.get("data_sha256") != data_capture.sha256:
    raise ValueError("benchmark_v2 dataset bytes do not match its manifest")
  family_sha = _require_sha256(
      data.get("family_split_manifest_sha256"),
      name="family_split_manifest_sha256",
  )
  _require_sha256(
      data.get("source_split_artifact_sha256"),
      name="source_split_artifact_sha256",
  )
  config_sha = _require_sha256(
      data.get("producer_config_sha256"),
      name="producer_config_sha256",
  )
  _require_sha256(
      data.get("producer_code_sha256"),
      name="producer_code_sha256",
  )
  if canonical_sha256(data.get("producer_config")) != config_sha:
    raise ValueError("benchmark_v2 producer config hash mismatch")
  private_binding = _validate_formal_face_map_producer_config(
      data.get("producer_config"),
      producer_kind=producer_kind,
  )
  if private_binding["family_split_manifest_sha256"] != family_sha:
    raise ValueError(
        "benchmark_v2 private supervision/family manifest binding mismatch"
    )
  row_count = data.get("row_count")
  if not isinstance(row_count, int) or row_count <= 0:
    raise ValueError("benchmark_v2 dataset manifest has no rows")
  tokens = data.get("case_token_sha256")
  if not isinstance(tokens, list) or not tokens:
    raise ValueError("benchmark_v2 dataset manifest has no case tokens")
  normalized_tokens = tuple(sorted(_require_sha256(value, name="case token") for value in tokens))
  case_count = data.get("case_count")
  if not isinstance(case_count, int) or case_count != len(normalized_tokens):
    raise ValueError("benchmark_v2 dataset manifest case count is invalid")
  positive_rows = data.get("positive_rows")
  negative_rows = data.get("negative_rows")
  if (
      not isinstance(positive_rows, int)
      or not isinstance(negative_rows, int)
      or positive_rows <= 0
      or negative_rows <= 0
      or positive_rows + negative_rows != row_count
  ):
    raise ValueError("benchmark_v2 dataset manifest class counts are invalid")
  raise ValueError(
      "formal benchmark_v2 dataset validation is explicitly unavailable until "
      "the manifest can be verified against actual authenticated source, gold, "
      "face-map, and allowlisted contact-census receipt files"
  )


def validate_rows_against_binding(
    rows: Sequence[Mapping[str, Any]],
    binding: TrainingDatasetBinding,
    *,
    allowed_keys: set[str],
) -> None:
  if len(rows) != binding.row_count:
    raise ValueError("benchmark_v2 row count does not match dataset manifest")
  labels: set[int] = set()
  positive_rows = 0
  negative_rows = 0
  tokens: set[str] = set()
  for index, row in enumerate(rows):
    if set(row) != allowed_keys:
      unknown = sorted(set(row) - allowed_keys)
      missing = sorted(allowed_keys - set(row))
      raise ValueError(
          f"benchmark_v2 row {index} schema mismatch; unknown={unknown}, missing={missing}"
      )
    if row.get("schema_version") != binding.row_schema:
      raise ValueError(f"benchmark_v2 row {index} schema binding mismatch")
    if row.get("model_input_protocol") != "benchmark_v2":
      raise ValueError(f"benchmark_v2 row {index} protocol mismatch")
    if row.get("protocol_version") != PROTOCOL_VERSION:
      raise ValueError(f"benchmark_v2 row {index} protocol version mismatch")
    if row.get("source_split") != binding.source_split:
      raise ValueError(f"benchmark_v2 row {index} source split mismatch")
    if row.get("family_split_manifest_sha256") != binding.family_split_manifest_sha256:
      raise ValueError(f"benchmark_v2 row {index} family split mismatch")
    if row.get("feature_names_sha256") != binding.feature_names_sha256:
      raise ValueError(f"benchmark_v2 row {index} feature schema mismatch")
    if row.get("producer_config_sha256") != binding.producer_config_sha256:
      raise ValueError(f"benchmark_v2 row {index} producer config mismatch")
    label = row.get("label")
    if label not in {0, 1}:
      raise ValueError(f"benchmark_v2 row {index} does not have a binary label")
    labels.add(int(label))
    positive_rows += int(label == 1)
    negative_rows += int(label == 0)
    tokens.add(_require_sha256(row.get("case_token_sha256"), name="case token"))
    if binding.row_schema == INTERFACE_TRAINING_ROW_SCHEMA:
      if not re.fullmatch(r"opaque_part_[0-9]{3,}", str(row.get("opaque_part") or "")):
        raise ValueError(f"benchmark_v2 row {index} has a non-opaque part identity")
      if not re.fullmatch(
          r"opaque_iface_[0-9a-f]{24}",
          str(row.get("opaque_interface") or ""),
      ):
        raise ValueError(f"benchmark_v2 row {index} has a non-opaque interface identity")
    if binding.row_schema == INTERFACE_PAIR_TRAINING_ROW_SCHEMA:
      if not re.fullmatch(
          r"opaque_pair_[0-9a-f]{24}", str(row.get("opaque_pair") or "")
      ):
        raise ValueError(f"benchmark_v2 pair row {index} has a non-opaque pair identity")
      for key in ("opaque_interface_a", "opaque_interface_b"):
        if not re.fullmatch(
            r"opaque_iface_[0-9a-f]{24}", str(row.get(key) or "")
        ):
          raise ValueError(
              f"benchmark_v2 pair row {index} has a non-opaque interface identity"
          )
      if row.get("relation_hint") != "link":
        raise ValueError(
            f"benchmark_v2 pair row {index} contains a role/edge-derived relation hint"
        )
      contact_type = str(row.get("contact_type") or "")
      if contact_type not in FORMAL_CONTACT_TYPES:
        raise ValueError(
            f"benchmark_v2 pair row {index} has an unknown contact type"
        )
      if (int(label) == 0 and contact_type != "none") or (
          int(label) == 1 and contact_type == "none"
      ):
        raise ValueError(
            f"benchmark_v2 pair row {index} has inconsistent contact supervision"
        )
  if labels != {0, 1}:
    raise ValueError("benchmark_v2 training requires both binary classes")
  if tuple(sorted(tokens)) != binding.case_token_sha256:
    raise ValueError("benchmark_v2 row case tokens do not match dataset manifest")
  if (
      positive_rows != binding.positive_rows
      or negative_rows != binding.negative_rows
  ):
    raise ValueError("benchmark_v2 row class counts do not match dataset manifest")


def model_parameter_sha256(
    payload: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> str:
  return canonical_sha256({key: payload.get(key) for key in keys})


def validate_training_hyperparameters(
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    positive_weight: float,
) -> None:
  if isinstance(epochs, bool) or int(epochs) <= 0:
    raise ValueError("benchmark_v2 training epochs must be a positive integer")
  values = {
      "learning_rate": float(learning_rate),
      "l2": float(l2),
      "positive_weight": float(positive_weight),
  }
  if any(not math.isfinite(value) for value in values.values()):
    raise ValueError("benchmark_v2 training hyperparameters must be finite")
  if values["learning_rate"] <= 0.0:
    raise ValueError("benchmark_v2 learning_rate must be positive")
  if values["l2"] < 0.0 or values["positive_weight"] < 0.0:
    raise ValueError("benchmark_v2 l2/positive_weight must be nonnegative")


def validate_formal_training_manifest_counts(manifest: Mapping[str, Any]) -> None:
  """Validate train/dev coverage and class-count invariants in a model receipt."""

  integer_keys = (
      "train_rows",
      "dev_rows",
      "train_case_count",
      "dev_case_count",
      "train_positive_rows",
      "dev_positive_rows",
  )
  if any(
      isinstance(manifest.get(key), bool)
      or not isinstance(manifest.get(key), int)
      for key in integer_keys
  ):
    raise ValueError("benchmark_v2 model training counts must be integers")
  train_rows = int(manifest["train_rows"])
  train_positive = int(manifest["train_positive_rows"])
  if (
      train_rows < 2
      or int(manifest["train_case_count"]) < 1
      or not 0 < train_positive < train_rows
  ):
    raise ValueError("benchmark_v2 model has insufficient train coverage/classes")
  _require_sha256(manifest.get("train_jsonl_sha256"), name="train_jsonl_sha256")
  optimizer = manifest.get("optimizer")
  if not isinstance(optimizer, Mapping) or set(optimizer) != {
      "epochs",
      "learning_rate",
      "l2",
      "positive_weight",
  }:
    raise ValueError("benchmark_v2 model optimizer schema mismatch")
  validate_training_hyperparameters(
      epochs=optimizer["epochs"],
      learning_rate=optimizer["learning_rate"],
      l2=optimizer["l2"],
      positive_weight=optimizer["positive_weight"],
  )
  if float(optimizer["positive_weight"]) <= 0.0:
    raise ValueError("benchmark_v2 stored positive_weight must be positive")

  dev_rows = int(manifest["dev_rows"])
  dev_case_count = int(manifest["dev_case_count"])
  dev_positive = int(manifest["dev_positive_rows"])
  if dev_rows == 0:
    if any(
        manifest.get(key) is not None
        for key in (
            "dev_jsonl_sha256",
            "dev_dataset_manifest_sha256",
            "dev_source_split",
            "frozen_family_dev_split_sha256",
        )
    ) or dev_case_count != 0 or dev_positive != 0:
      raise ValueError("benchmark_v2 model has inconsistent empty-dev provenance")
  else:
    if dev_case_count < 1 or not 0 <= dev_positive <= dev_rows:
      raise ValueError("benchmark_v2 model has invalid dev coverage")
    _require_sha256(manifest.get("dev_jsonl_sha256"), name="dev_jsonl_sha256")
    _require_sha256(
        manifest.get("dev_dataset_manifest_sha256"),
        name="dev_dataset_manifest_sha256",
    )
    if manifest.get("dev_source_split") != "dev":
      raise ValueError("benchmark_v2 model dev source split mismatch")
    _require_sha256(
        manifest.get("frozen_family_dev_split_sha256"),
        name="frozen_family_dev_split_sha256",
    )


def validate_finite_numeric_tree(value: Any, *, name: str) -> None:
  """Reject NaN/Inf and booleans anywhere in numeric model parameters."""

  if isinstance(value, bool):
    raise ValueError(f"{name} contains a boolean instead of a finite number")
  if isinstance(value, (int, float)):
    if not math.isfinite(float(value)):
      raise ValueError(f"{name} contains a non-finite value")
    return
  if isinstance(value, (list, tuple)):
    if not value:
      raise ValueError(f"{name} must not be empty")
    for item in value:
      validate_finite_numeric_tree(item, name=name)
    return
  raise ValueError(f"{name} contains a non-numeric value")
