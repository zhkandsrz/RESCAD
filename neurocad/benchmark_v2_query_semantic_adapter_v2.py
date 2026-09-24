"""Development-only V19 semantic labels -> benchmark-v2 model-view adapter v2.

The adapter deliberately consumes only the non-serializable handle issued by
``verify_query_mate_semantic_full_domain_v1``.  It does not accept JSON rows,
paths, or caller-provided labels.  Geometry is not materialized while the
index is built: each query/direction instead receives one immutable graph-work
binding to the exact STEP bytes, mapped OCC faces, and mapper receipt already
cross-checked by the semantic authority.  A later graph worker must emit
``benchmark_v2_model_view.v2`` / ``benchmark_v2_intrinsic_brep_graph.v2``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .benchmark_v2_model_view_v2 import (
    GRAPH_SCHEMA_VERSION,
    SCHEMA_VERSION as MODEL_VIEW_SCHEMA_VERSION,
    HARD_MAX_GRAPH_SIZE_BUDGET,
    _catalog_descriptor_from_authority_row,
)
from .benchmark_v2_training_provenance import canonical_sha256
from .fixed_program_catalog_protocol_v1 import (
    OFFICIAL_FROZEN_CATALOG_COMMITMENTS_V1,
    load_official_frozen_program_catalog_v1,
)
from .query_mate_semantic_full_domain_v1 import (
    AuthenticatedQueryMateSemanticFullDomain,
    derive_v19_full_pending_semantic_domain,
)

SCHEMA_VERSION = "benchmark_v2_authenticated_query_graph_work_index.v2"
PROGRAM_CATALOG_SCHEMA = "benchmark_v2_pre_v19_fixed_program_catalog.v2"
GRAPH_WORK_SCHEMA = "benchmark_v2_query_graph_work.v2"
SAMPLE_SCHEMA = "benchmark_v2_query_program_sample.v2"
REJECTION_SCHEMA = "benchmark_v2_query_semantic_rejection.v1"
DEVELOPMENT_SPLITS = ("train", "dev")

PROJECT_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark_v2_query_semantic_adapter_v2.schema.json"
)
FIXED_LINEAGE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark_v2_query_semantic_adapter_v1.fixed.json"
)
FIXED_CATALOG_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark_v2_query_semantic_adapter_v2.fixed19_catalog.json"
)
FIXED_V19_MANIFEST = PROJECT_ROOT / (
    "artifacts/development/debug/"
    "v19_query_certification_manifest_window0_c5_blockerfix_20260724.json"
)

_FIXED_CATALOG_POLICY = MappingProxyType({
    "schema_version": (
        "benchmark_v2_pre_v19_fixed_program_catalog_policy.v2"
    ),
    "candidate_policy": (
        "pre_v19_fixed19_catalog_with_explicit_oov_failure.v1"
    ),
    "candidate_order": "pre_v19_roster_program_index.v1",
    "target_policy": "all_targets_retained_catalog_index_or_null_oov.v1",
    "weight_policy": (
        "balanced_inverse_frequency_positive_classes_zero_for_unseen.v1"
    ),
    "entry_count": 19,
    "policy_source_revision": (
        "ee922bbbbb28753b6123bd5e951afeb0b5b5481b"
    ),
    "source_spec_logical_id": (
        "artifacts/formal/"
        "externally_frozen_program_catalog_spec_v3_p0_20260719.json"
    ),
    "source_spec_file_sha256": (
        "e97aed9f936284429698d783ae3eb56ae6d42f8ec72d94da5c4eae26bdc760db"
    ),
    "source_spec_payload_sha256": (
        "71200b9b894419f8519ea1dabd5809207702903d67b44820a2c59b54faaceb3d"
    ),
    "catalog_roster_sha256": (
        "7e97af33edd35186499f6e187a85a0b07a0a0f31afe105b36d46754b49755724"
    ),
    "source_artifact_sha256": (
        "2ca8cb6e13e2b176267ac1f81e13fcbc474f791e89dadb282e1ff6f4a6370966"
    ),
    "producer_code_sha256": (
        "9f4151a5624d426516cd5d479ac54da5723413203afaa57cf0810c63f26d088c"
    ),
})


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def _plain_json(value: Any) -> Any:
  """Thaw verifier-owned mapping proxies/tuples into strict JSON values."""

  if isinstance(value, Mapping):
    return {str(key): _plain_json(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain_json(item) for item in value]
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  raise TypeError(f"authority value is not JSON-compatible: {type(value).__name__}")


def _sha256_bytes(raw: bytes) -> str:
  return hashlib.sha256(raw).hexdigest()


def _opaque_case_id(case: Mapping[str, Any]) -> str:
  """Torch-free byte-equivalent replay of the historical opaque case ID."""

  payload = {
      "id": str(case.get("id") or ""),
      "assembly_dir": str(case.get("assembly_dir") or ""),
  }
  return "opaque_case_" + canonical_sha256(payload)[:20]


def _logical_project_id(path: str | Path) -> str:
  resolved = Path(path).resolve(strict=False)
  try:
    relative = resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
  except ValueError as error:
    raise ValueError("adapter file lies outside the project root") from error
  return relative.as_posix()


def _is_reparse(stat_result: os.stat_result) -> bool:
  attributes = int(getattr(stat_result, "st_file_attributes", 0))
  reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
  return stat.S_ISLNK(stat_result.st_mode) or bool(attributes & reparse_flag)


def _reject_reparse_chain(path: Path, *, allow_missing_leaf: bool = False) -> None:
  absolute = path.absolute()
  project = PROJECT_ROOT.resolve(strict=True)
  try:
    relative = absolute.relative_to(project)
  except ValueError as error:
    raise ValueError("adapter path lies outside the project root") from error
  current = project
  project_stat = os.lstat(project)
  if _is_reparse(project_stat):
    raise ValueError("project root is a link or reparse point")
  for index, component in enumerate(relative.parts):
    current = current / component
    try:
      observed = os.lstat(current)
    except FileNotFoundError:
      if allow_missing_leaf:
        return
      raise ValueError("adapter path component is missing") from None
    if _is_reparse(observed):
      raise ValueError("adapter path contains a link or reparse point")
    if index < len(relative.parts) - 1 and not stat.S_ISDIR(
        observed.st_mode
    ):
      raise ValueError("adapter path parent is not a directory")


@dataclass(frozen=True, slots=True)
class _CapturedProjectFile:
  logical_id: str
  resolved_path: Path
  raw_bytes: bytes
  sha256: str
  byte_count: int
  device: int
  inode: int

  def content_binding(self) -> dict[str, Any]:
    return {
        "logical_id": self.logical_id,
        "sha256": self.sha256,
        "bytes": self.byte_count,
    }


def _capture_project_file(path: str | Path, *, label: str) -> _CapturedProjectFile:
  candidate = Path(path).absolute()
  _reject_reparse_chain(candidate)
  resolved = candidate.resolve(strict=True)
  before = os.stat(resolved, follow_symlinks=False)
  if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
    raise ValueError(f"{label} is not a plain project file")
  with resolved.open("rb") as stream:
    opened = os.fstat(stream.fileno())
    if (
        opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or opened.st_size != before.st_size
    ):
      raise ValueError(f"{label} identity changed before capture")
    raw = stream.read()
    after_open = os.fstat(stream.fileno())
  after_path = os.stat(resolved, follow_symlinks=False)
  if (
      after_open.st_dev != opened.st_dev
      or after_open.st_ino != opened.st_ino
      or after_open.st_size != len(raw)
      or after_path.st_dev != opened.st_dev
      or after_path.st_ino != opened.st_ino
      or after_path.st_size != len(raw)
      or _is_reparse(after_path)
  ):
    raise ValueError(f"{label} changed during capture")
  return _CapturedProjectFile(
      logical_id=_logical_project_id(resolved),
      resolved_path=resolved,
      raw_bytes=raw,
      sha256=_sha256_bytes(raw),
      byte_count=len(raw),
      device=int(opened.st_dev),
      inode=int(opened.st_ino),
  )


def _reverify_capture(captured: _CapturedProjectFile, *, label: str) -> None:
  _reject_reparse_chain(captured.resolved_path)
  observed = os.stat(captured.resolved_path, follow_symlinks=False)
  if (
      not stat.S_ISREG(observed.st_mode)
      or _is_reparse(observed)
      or observed.st_dev != captured.device
      or observed.st_ino != captured.inode
      or observed.st_size != captured.byte_count
  ):
    raise ValueError(f"{label} identity changed after capture")
  with captured.resolved_path.open("rb") as stream:
    raw = stream.read()
  final = os.stat(captured.resolved_path, follow_symlinks=False)
  if (
      final.st_dev != captured.device
      or final.st_ino != captured.inode
      or final.st_size != captured.byte_count
      or _sha256_bytes(raw) != captured.sha256
  ):
    raise ValueError(f"{label} bytes changed after capture")


def _json_from_capture(captured: _CapturedProjectFile, *, label: str) -> Any:
  def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"{label} contains duplicate JSON key: {key}")
      result[key] = value
    return result

  try:
    return json.loads(
        captured.raw_bytes.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite JSON: {value}")
        ),
    )
  except (UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise ValueError(f"{label} is not readable UTF-8 JSON") from error


def _git_revision() -> str:
  try:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
  except (OSError, subprocess.SubprocessError) as error:
    raise ValueError("adapter source revision is unavailable") from error
  revision = result.stdout.strip().lower()
  if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError("adapter source revision differs")
  return revision


@dataclass(frozen=True, slots=True)
class _FixedContext:
  manifest_binding: Mapping[str, Any]
  family_source_binding: Mapping[str, Any]
  authority_receipt_binding: Mapping[str, Any]
  authority_receipt_payload_sha256: str
  authority_domain_sha256: str
  authority_commitments_by_key: Mapping[
      tuple[str, int], Mapping[str, Any]
  ]
  domain_rows: tuple[Mapping[str, Any], ...]
  source_by_case: Mapping[str, Mapping[str, Any]]
  public_by_case: Mapping[str, Mapping[str, Any]]
  mapper_binding_by_case: Mapping[str, Mapping[str, Any]]
  captures: tuple[_CapturedProjectFile, ...]


@dataclass(frozen=True, slots=True)
class _FixedCatalogSource:
  entries: tuple[Mapping[str, Any], ...]
  source_binding: Mapping[str, Any]
  policy_binding: Mapping[str, Any]
  spec_payload_sha256: str
  roster_sha256: str
  source_artifact_sha256: str
  producer_code_sha256: str
  policy_source_revision: str
  trust_root_id: str
  captures: tuple[_CapturedProjectFile, ...]


def _load_fixed_program_catalog() -> _FixedCatalogSource:
  """Load the exact pre-V19 19-class roster from its frozen source bytes."""

  policy_capture = _capture_project_file(
      FIXED_CATALOG_POLICY_PATH, label="fixed19 adapter catalog policy"
  )
  policy = _json_from_capture(
      policy_capture, label="fixed19 adapter catalog policy"
  )
  if (
      not isinstance(policy, Mapping)
      or dict(policy) != dict(_FIXED_CATALOG_POLICY)
  ):
    raise ValueError("fixed19 adapter catalog policy differs")
  source_id = str(policy["source_spec_logical_id"])
  source_capture = _capture_project_file(
      PROJECT_ROOT / source_id, label="pre-V19 fixed19 catalog source"
  )
  if (
      source_capture.logical_id != source_id
      or source_capture.sha256 != policy["source_spec_file_sha256"]
  ):
    raise ValueError("pre-V19 fixed19 catalog source bytes differ")
  # Capture with the adapter's strict duplicate-key parser before delegating
  # typed descriptor/roster verification to the pre-existing V3 loader.
  source_payload = _json_from_capture(
      source_capture, label="pre-V19 fixed19 catalog source"
  )
  if not isinstance(source_payload, Mapping):
    raise ValueError("pre-V19 fixed19 catalog source root differs")
  spec = load_official_frozen_program_catalog_v1(
      source_capture.resolved_path
  )
  if (
      spec.spec_payload_sha256 != policy["source_spec_payload_sha256"]
      or spec.source_artifact_sha256
      != policy["source_artifact_sha256"]
      or spec.producer_code_sha256 != policy["producer_code_sha256"]
      or spec.roster.catalog_sha256 != policy["catalog_roster_sha256"]
      or len(spec.roster.entries) != policy["entry_count"]
  ):
    raise ValueError("pre-V19 fixed19 catalog commitments differ")
  entries = tuple(
      MappingProxyType({
          **entry.payload(),
          "program_type": entry.descriptor.relation_hint,
      })
      for entry in spec.roster.entries
  )
  captures = (policy_capture, source_capture)
  for captured in captures:
    _reverify_capture(captured, label=captured.logical_id)
  return _FixedCatalogSource(
      entries=entries,
      source_binding=MappingProxyType(source_capture.content_binding()),
      policy_binding=MappingProxyType(policy_capture.content_binding()),
      spec_payload_sha256=spec.spec_payload_sha256,
      roster_sha256=spec.roster.catalog_sha256,
      source_artifact_sha256=spec.source_artifact_sha256,
      producer_code_sha256=spec.producer_code_sha256,
      policy_source_revision=str(policy["policy_source_revision"]),
      trust_root_id=str(
          OFFICIAL_FROZEN_CATALOG_COMMITMENTS_V1["trust_root_id"]
      ),
      captures=captures,
  )


def _load_fixed_context() -> _FixedContext:
  lineage_capture = _capture_project_file(
      FIXED_LINEAGE_PATH, label="adapter fixed lineage"
  )
  lineage = _json_from_capture(
      lineage_capture, label="adapter fixed lineage"
  )
  expected_lineage_keys = {
      "schema_version",
      "development",
      "formal",
      "publication_eligible",
      "final_test_touched",
      "withheld_test_touched",
      "authority_receipt_logical_id",
      "authority_receipt_file_sha256",
      "authority_receipt_payload_sha256",
      "authority_domain_sha256",
      "authority_source_revision",
      "expected_selected_query_count",
      "expected_program_row_count",
      "expected_rejected_query_count",
  }
  if (
      not isinstance(lineage, Mapping)
      or set(lineage) != expected_lineage_keys
      or lineage.get("schema_version")
      != "benchmark_v2_query_semantic_adapter_fixed_lineage.v1"
      or lineage.get("development") is not True
      or lineage.get("formal") is not False
      or lineage.get("publication_eligible") is not False
      or lineage.get("final_test_touched") is not False
      or lineage.get("withheld_test_touched") is not False
      or lineage.get("authority_source_revision")
      != "774853033ca097a0c2ebaf27de0320f86f202768"
      or lineage.get("expected_selected_query_count") != 938
      or lineage.get("expected_program_row_count") != 5106
      or lineage.get("expected_rejected_query_count") != 3
  ):
    raise ValueError("adapter fixed lineage contract differs")
  receipt_id = str(lineage["authority_receipt_logical_id"])
  if (
      Path(receipt_id).is_absolute()
      or ".." in Path(receipt_id).parts
      or not receipt_id.startswith("artifacts/development/")
  ):
    raise ValueError("fixed authority receipt logical id differs")
  receipt_capture = _capture_project_file(
      PROJECT_ROOT / Path(receipt_id),
      label="fixed full-domain authority receipt",
  )
  if (
      receipt_capture.logical_id != receipt_id
      or receipt_capture.sha256
      != lineage["authority_receipt_file_sha256"]
  ):
    raise ValueError("fixed full-domain authority receipt bytes differ")
  receipt = _json_from_capture(
      receipt_capture, label="fixed full-domain authority receipt"
  )
  if (
      not isinstance(receipt, Mapping)
      or receipt.get("schema_version")
      != "neurocad_query_mate_semantic_full_domain_authority.v1"
      or receipt.get("development") is not True
      or receipt.get("formal") is not False
      or receipt.get("publication_eligible") is not False
      or receipt.get("formal_promotion_blocked") is not True
      or receipt.get("final_test_touched") is not False
      or receipt.get("withheld_test_touched") is not False
      or receipt.get("receipt_payload_sha256")
      != lineage["authority_receipt_payload_sha256"]
      or receipt.get("input_domain", {}).get("commitment_sha256")
      != lineage["authority_domain_sha256"]
  ):
    raise ValueError("fixed full-domain authority receipt lineage differs")
  unsigned_receipt = dict(receipt)
  receipt_self_hash = unsigned_receipt.pop("receipt_payload_sha256", None)
  if canonical_sha256(unsigned_receipt) != receipt_self_hash:
    raise ValueError("fixed full-domain authority receipt self-hash differs")
  receipt_coverage = receipt.get("coverage")
  if (
      not isinstance(receipt_coverage, Mapping)
      or receipt_coverage.get("selected_query_count") != 938
      or receipt_coverage.get("program_row_count") != 5106
      or receipt_coverage.get("rejected_query_count") != 3
  ):
    raise ValueError("fixed full-domain authority coverage differs")

  manifest_receipt_binding = receipt.get("inputs", {}).get(
      "v19_query_manifest"
  )
  if not isinstance(manifest_receipt_binding, Mapping):
    raise ValueError("fixed authority lacks its V19 manifest binding")
  manifest_capture = _capture_project_file(
      str(manifest_receipt_binding.get("path") or ""),
      label="fixed V19 manifest",
  )
  if (
      manifest_capture.sha256 != manifest_receipt_binding.get("sha256")
      or manifest_capture.byte_count != manifest_receipt_binding.get("bytes")
      or manifest_capture.resolved_path
      != FIXED_V19_MANIFEST.resolve(strict=True)
  ):
    raise ValueError("fixed V19 manifest bytes differ")
  manifest = _json_from_capture(manifest_capture, label="fixed V19 manifest")
  if not isinstance(manifest, Mapping):
    raise ValueError("fixed V19 manifest root differs")
  domain_rows, domain_sha256, domain_summary = (
      derive_v19_full_pending_semantic_domain(manifest)
  )
  if (
      domain_sha256 != lineage["authority_domain_sha256"]
      or domain_summary.get("selected_query_count") != 938
      or domain_summary.get("case_overlap_count") != 0
      or domain_summary.get("family_overlap_count") != 0
      or domain_summary.get("final_test_touched") is not False
      or domain_summary.get("withheld_test_touched") is not False
  ):
    raise ValueError("fixed V19 adapter domain differs")
  inputs = manifest.get("inputs")
  if not isinstance(inputs, Mapping):
    raise ValueError("fixed V19 manifest input bindings differ")
  family_binding = inputs.get("family_source")
  if not isinstance(family_binding, Mapping):
    raise ValueError("fixed V19 family source binding differs")
  family_capture = _capture_project_file(
      str(family_binding.get("path") or ""),
      label="fixed V19 family source",
  )
  if (
      family_capture.sha256 != family_binding.get("sha256")
      or family_capture.byte_count != family_binding.get("bytes")
  ):
    raise ValueError("fixed V19 family source bytes differ")
  family_source = _json_from_capture(
      family_capture, label="fixed V19 family source"
  )
  if not isinstance(family_source, Mapping):
    raise ValueError("fixed V19 family source root differs")
  source_rows = (
      family_source.get("private_source_bindings", {}).get("cases")
      if isinstance(family_source.get("private_source_bindings"), Mapping)
      else None
  )
  public_rows = family_source.get("cases")
  if not isinstance(source_rows, list) or not isinstance(public_rows, list):
    raise ValueError("fixed V19 family source case domains differ")
  source_by_case = {
      str(row.get("case_id") or ""): row
      for row in source_rows
      if isinstance(row, Mapping)
  }
  public_by_case = {
      str(row.get("case_id") or row.get("id") or ""): row
      for row in public_rows
      if isinstance(row, Mapping)
  }
  selected_cases = {
      str(row["key"]["case_id"]) for row in domain_rows
  }
  if (
      not selected_cases
      or not selected_cases <= set(source_by_case)
      or not selected_cases <= set(public_by_case)
  ):
    raise ValueError("fixed V19 family source lacks adapter cases")

  mapper_bindings: dict[str, Mapping[str, Any]] = {}
  for raw in inputs.get("mapper_receipts") or []:
    if isinstance(raw, Mapping):
      mapper_bindings[str(raw.get("case_id") or "")] = raw
  for raw in inputs.get("overlay_receipts") or []:
    if (
        isinstance(raw, Mapping)
        and raw.get("overlay_role") == "mapper"
    ):
      mapper_bindings[str(raw.get("case_id") or "")] = raw
  if not selected_cases <= set(mapper_bindings):
    raise ValueError("fixed V19 manifest lacks adapter mapper receipts")
  mapper_captures: list[_CapturedProjectFile] = []
  normalized_mapper_bindings: dict[str, Mapping[str, Any]] = {}
  for case_id in sorted(selected_cases):
    raw_binding = mapper_bindings[case_id]
    captured = _capture_project_file(
        str(raw_binding.get("path") or ""),
        label=f"fixed V19 mapper receipt {case_id}",
    )
    if (
        captured.sha256 != raw_binding.get("sha256")
        or captured.byte_count != raw_binding.get("bytes")
    ):
      raise ValueError("fixed V19 mapper receipt bytes differ")
    mapper_captures.append(captured)
    normalized_mapper_bindings[case_id] = MappingProxyType(
        captured.content_binding()
    )

  raw_commitments = receipt.get("query_commitments")
  if not isinstance(raw_commitments, list) or len(raw_commitments) != 938:
    raise ValueError("fixed authority query commitments differ")
  commitments: dict[tuple[str, int], Mapping[str, Any]] = {}
  for raw in raw_commitments:
    if not isinstance(raw, Mapping):
      raise ValueError("fixed authority query commitment differs")
    ordinal = raw.get("source_contact_ordinal")
    key = (str(raw.get("case_id") or ""), ordinal)
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or key in commitments
    ):
      raise ValueError("fixed authority query identity differs")
    commitments[key] = MappingProxyType(copy.deepcopy(dict(raw)))
  domain_keys = {
      (
          str(row["key"]["case_id"]),
          int(row["key"]["source_contact_ordinal"]),
      )
      for row in domain_rows
  }
  if set(commitments) != domain_keys:
    raise ValueError("fixed authority commitments do not close V19 domain")

  captures = (
      lineage_capture,
      receipt_capture,
      manifest_capture,
      family_capture,
      *mapper_captures,
  )
  for captured in captures:
    _reverify_capture(captured, label=captured.logical_id)

  return _FixedContext(
      manifest_binding=MappingProxyType(
          manifest_capture.content_binding()
      ),
      family_source_binding=MappingProxyType(
          family_capture.content_binding()
      ),
      authority_receipt_binding=MappingProxyType(
          receipt_capture.content_binding()
      ),
      authority_receipt_payload_sha256=str(
          lineage["authority_receipt_payload_sha256"]
      ),
      authority_domain_sha256=str(lineage["authority_domain_sha256"]),
      authority_commitments_by_key=MappingProxyType(commitments),
      domain_rows=tuple(copy.deepcopy(dict(row)) for row in domain_rows),
      source_by_case=MappingProxyType(dict(source_by_case)),
      public_by_case=MappingProxyType(dict(public_by_case)),
      mapper_binding_by_case=MappingProxyType(
          normalized_mapper_bindings
      ),
      captures=tuple(captures),
  )


def _program_catalog(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    query_id_by_row_key: Mapping[tuple[str, int], str],
    fixed: _FixedCatalogSource,
) -> dict[str, Any]:
  by_descriptor = {
      str(entry["descriptor_sha256"]): entry for entry in fixed.entries
  }
  if len(by_descriptor) != 19:
    raise ValueError("pre-V19 fixed19 catalog descriptor domain differs")
  frequencies = [0 for _ in fixed.entries]
  oov_target_count = 0
  sources: list[dict[str, Any]] = []
  for row in train_rows:
    source_contact = row.get("source_contact")
    if not isinstance(source_contact, Mapping):
      raise ValueError("train program row lacks source contact")
    ordinal = source_contact.get("contact_index")
    opaque_case = str(row.get("case_id") or "")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or (opaque_case, ordinal) not in query_id_by_row_key
    ):
      raise ValueError("train program row query identity differs")
    descriptor = _catalog_descriptor_from_authority_row(row)
    descriptor_sha256 = canonical_sha256(descriptor)
    entry = by_descriptor.get(descriptor_sha256)
    if entry is None:
      oov_target_count += 1
    else:
      frequencies[int(entry["program_index"])] += 1
    sources.append(
        {
            "opaque_query_identity": query_id_by_row_key[
                (opaque_case, ordinal)
            ],
            "source_program_id": str(row.get("program_id") or ""),
            "semantic_row_sha256": canonical_sha256(row),
            "descriptor_sha256": descriptor_sha256,
        }
    )
  if len(fixed.entries) > (
      HARD_MAX_GRAPH_SIZE_BUDGET.max_program_candidates_per_example
  ):
    raise ValueError("pre-V19 fixed19 catalog exceeds the graph budget")
  sources.sort(
      key=lambda row: (
          row["opaque_query_identity"],
          row["source_program_id"],
      )
  )
  known_target_count = sum(frequencies)
  positive_class_count = sum(count > 0 for count in frequencies)
  class_weights = [
      (
          known_target_count / (positive_class_count * count)
          if count > 0 and positive_class_count > 0
          else 0.0
      )
      for count in frequencies
  ]
  payload = {
      "schema_version": PROGRAM_CATALOG_SCHEMA,
      "candidate_policy": _FIXED_CATALOG_POLICY["candidate_policy"],
      "candidate_order": _FIXED_CATALOG_POLICY["candidate_order"],
      "target_policy": _FIXED_CATALOG_POLICY["target_policy"],
      "entry_count": len(fixed.entries),
      "fixed_source": {
          "source_spec": dict(fixed.source_binding),
          "policy": dict(fixed.policy_binding),
          "policy_source_revision": fixed.policy_source_revision,
          "source_spec_payload_sha256": fixed.spec_payload_sha256,
          "catalog_roster_sha256": fixed.roster_sha256,
          "source_artifact_sha256": fixed.source_artifact_sha256,
          "producer_code_sha256": fixed.producer_code_sha256,
          "trust_root_id": fixed.trust_root_id,
      },
      "training_statistics": {
          "schema_version": "v19_train_only_class_statistics.v1",
          "source_split": "train",
          "frequency_policy": (
              "exact_authority_target_count_by_fixed_catalog_index.v1"
          ),
          "weight_policy": _FIXED_CATALOG_POLICY["weight_policy"],
          "source_binding_count": len({
              row["opaque_query_identity"] for row in sources
          }),
          "source_row_count": len(sources),
          "source_commitment_sha256": canonical_sha256(sources),
          "known_target_count": known_target_count,
          "oov_target_count": oov_target_count,
          "positive_class_count": positive_class_count,
          "target_frequency_by_program_index": frequencies,
          "class_weight_by_program_index": class_weights,
      },
      "entries": [copy.deepcopy(dict(entry)) for entry in fixed.entries],
  }
  payload["catalog_payload_sha256"] = canonical_sha256(payload)
  return _validated_fixed_program_catalog(payload)


def _validated_fixed_program_catalog(payload: Any) -> dict[str, Any]:
  if not isinstance(payload, Mapping):
    raise ValueError("fixed19 program catalog must be an object")
  expected_keys = {
      "schema_version",
      "candidate_policy",
      "candidate_order",
      "target_policy",
      "entry_count",
      "fixed_source",
      "training_statistics",
      "entries",
      "catalog_payload_sha256",
  }
  if (
      set(payload) != expected_keys
      or payload.get("schema_version") != PROGRAM_CATALOG_SCHEMA
      or payload.get("candidate_policy")
      != _FIXED_CATALOG_POLICY["candidate_policy"]
      or payload.get("candidate_order")
      != _FIXED_CATALOG_POLICY["candidate_order"]
      or payload.get("target_policy")
      != _FIXED_CATALOG_POLICY["target_policy"]
      or payload.get("entry_count") != 19
  ):
    raise ValueError("fixed19 program catalog contract differs")
  unsigned = dict(payload)
  observed_hash = unsigned.pop("catalog_payload_sha256", None)
  if observed_hash != canonical_sha256(unsigned):
    raise ValueError("fixed19 program catalog self-hash differs")
  fixed_source = payload.get("fixed_source")
  if not isinstance(fixed_source, Mapping) or set(fixed_source) != {
      "source_spec",
      "policy",
      "policy_source_revision",
      "source_spec_payload_sha256",
      "catalog_roster_sha256",
      "source_artifact_sha256",
      "producer_code_sha256",
      "trust_root_id",
  }:
    raise ValueError("fixed19 program catalog source binding differs")
  expected_source_values = {
      "policy_source_revision": _FIXED_CATALOG_POLICY[
          "policy_source_revision"
      ],
      "source_spec_payload_sha256": _FIXED_CATALOG_POLICY[
          "source_spec_payload_sha256"
      ],
      "catalog_roster_sha256": _FIXED_CATALOG_POLICY[
          "catalog_roster_sha256"
      ],
      "source_artifact_sha256": _FIXED_CATALOG_POLICY[
          "source_artifact_sha256"
      ],
      "producer_code_sha256": _FIXED_CATALOG_POLICY[
          "producer_code_sha256"
      ],
      "trust_root_id": "p0_frozen_program_roster_20260719.v1",
  }
  if any(
      fixed_source.get(key) != value
      for key, value in expected_source_values.items()
  ):
    raise ValueError("fixed19 program catalog source commitment differs")
  entries = payload.get("entries")
  if not isinstance(entries, list) or len(entries) != 19:
    raise ValueError("fixed19 program catalog roster differs")
  descriptor_hashes: set[str] = set()
  for index, entry in enumerate(entries):
    if not isinstance(entry, Mapping) or set(entry) != {
        "program_index",
        "program_id",
        "program_type",
        "descriptor",
        "descriptor_sha256",
    }:
      raise ValueError("fixed19 program catalog entry differs")
    descriptor = entry.get("descriptor")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "relation_hint",
        "contact_type",
        "surface_type_a",
        "surface_type_b",
    }:
      raise ValueError("fixed19 program descriptor differs")
    descriptor_sha256 = canonical_sha256(descriptor)
    if (
        entry.get("program_index") != index
        or entry.get("descriptor_sha256") != descriptor_sha256
        or entry.get("program_id")
        != f"catalog_{descriptor_sha256[:16]}"
        or entry.get("program_type") != descriptor.get("relation_hint")
        or descriptor_sha256 in descriptor_hashes
    ):
      raise ValueError("fixed19 program catalog roster identity differs")
    descriptor_hashes.add(descriptor_sha256)
  roster_payload = {
      "schema_version": "finite_program_catalog_roster.v2",
      "entry_count": 19,
      "entries": [
          {
              key: entry[key]
              for key in (
                  "program_index",
                  "program_id",
                  "descriptor",
                  "descriptor_sha256",
              )
          }
          for entry in entries
      ],
  }
  if canonical_sha256(roster_payload) != fixed_source.get(
      "catalog_roster_sha256"
  ):
    raise ValueError("fixed19 program catalog roster hash differs")
  statistics = payload.get("training_statistics")
  if not isinstance(statistics, Mapping) or set(statistics) != {
      "schema_version",
      "source_split",
      "frequency_policy",
      "weight_policy",
      "source_binding_count",
      "source_row_count",
      "source_commitment_sha256",
      "known_target_count",
      "oov_target_count",
      "positive_class_count",
      "target_frequency_by_program_index",
      "class_weight_by_program_index",
  }:
    raise ValueError("fixed19 train-only statistics differ")
  frequencies = statistics.get("target_frequency_by_program_index")
  weights = statistics.get("class_weight_by_program_index")
  if (
      statistics.get("schema_version")
      != "v19_train_only_class_statistics.v1"
      or statistics.get("source_split") != "train"
      or statistics.get("frequency_policy")
      != "exact_authority_target_count_by_fixed_catalog_index.v1"
      or statistics.get("weight_policy")
      != _FIXED_CATALOG_POLICY["weight_policy"]
      or not isinstance(frequencies, list)
      or len(frequencies) != 19
      or any(
          isinstance(value, bool) or not isinstance(value, int) or value < 0
          for value in frequencies
      )
      or not isinstance(weights, list)
      or len(weights) != 19
      or any(
          isinstance(value, bool)
          or not isinstance(value, (int, float))
          or value < 0
          for value in weights
      )
  ):
    raise ValueError("fixed19 train-only frequency/weight domain differs")
  known_count = sum(frequencies)
  positive_count = sum(value > 0 for value in frequencies)
  expected_weights = [
      (
          known_count / (positive_count * count)
          if count > 0 and positive_count > 0
          else 0.0
      )
      for count in frequencies
  ]
  if (
      statistics.get("known_target_count") != known_count
      or statistics.get("positive_class_count") != positive_count
      or statistics.get("source_row_count")
      != known_count + statistics.get("oov_target_count", -1)
      or weights != expected_weights
  ):
    raise ValueError("fixed19 train-only statistics do not close")
  return copy.deepcopy(dict(payload))


def _body_step(
    source_case: Mapping[str, Any],
    *,
    endpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
  receipt = source_case.get("source_receipt_binding")
  steps = (
      receipt.get("body_steps") if isinstance(receipt, Mapping) else None
  )
  matches = [
      row
      for row in steps or []
      if isinstance(row, Mapping)
      and row.get("part") == endpoint.get("part")
      and row.get("body_uuid") == endpoint.get("body_uuid")
      and row.get("geometry_asset") == endpoint.get("geometry_asset")
  ]
  if len(matches) != 1:
    raise ValueError("endpoint lacks one receipt-bound STEP")
  step = matches[0]
  if (
      re.fullmatch(r"[0-9a-f]{64}", str(step.get("sha256") or "")) is None
      or type(step.get("bytes")) is not int
      or step.get("bytes") <= 0
  ):
    raise ValueError("endpoint STEP byte binding differs")
  return step


def _endpoint_graph_binding(
    *,
    endpoint: Mapping[str, Any],
    source_case: Mapping[str, Any],
    mapper_binding: Mapping[str, Any],
) -> dict[str, Any]:
  indices = endpoint.get("raw_occ_face_indices")
  signatures = endpoint.get("source_face_signature_sha256s")
  if (
      endpoint.get("status") != "mapped_unique"
      or not isinstance(indices, list)
      or not indices
      or any(type(value) is not int or value < 0 for value in indices)
      or indices != sorted(set(indices))
      or not isinstance(signatures, list)
      or len(signatures) != len(indices)
      or any(
          re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is None
          for value in signatures
      )
  ):
    raise ValueError("mapped endpoint OCC identity differs")
  step = _body_step(source_case, endpoint=endpoint)
  payload = {
      "schema_version": "benchmark_v2_endpoint_graph_binding.v1",
      "step": {
          "sha256": step["sha256"],
          "bytes": step["bytes"],
          "source_path_sha256": canonical_sha256(str(step.get("path") or "")),
      },
      "face_map_receipt": {
          "sha256": mapper_binding["sha256"],
          "bytes": mapper_binding["bytes"],
      },
      "mapper_endpoint_identity_sha256": canonical_sha256(endpoint),
      "fusion_face_index": endpoint.get("fusion_face_index"),
      "raw_occ_face_indices": list(indices),
      "source_face_signature_sha256s": list(signatures),
  }
  payload["endpoint_graph_cache_key_sha256"] = canonical_sha256(payload)
  return payload


def _target_catalog_entry(
    catalog: Mapping[str, Any],
    row: Mapping[str, Any],
) -> Mapping[str, Any] | None:
  descriptor_sha256 = canonical_sha256(
      _catalog_descriptor_from_authority_row(row)
  )
  matches = [
      entry
      for entry in catalog["entries"]
      if entry["descriptor_sha256"] == descriptor_sha256
  ]
  if len(matches) > 1:
    raise ValueError("program descriptor appears twice in train catalog")
  return matches[0] if matches else None


def _producer_binding(
) -> tuple[dict[str, Any], tuple[_CapturedProjectFile, ...]]:
  files = {
      "adapter": Path(__file__),
      "model_view_v2": PROJECT_ROOT / "benchmark_v2_model_view_v2.py",
      "full_domain_authority": (
          PROJECT_ROOT / "query_mate_semantic_full_domain_v1.py"
      ),
      "publisher_tool": (
          PROJECT_ROOT
          / "tools"
          / "build_benchmark_v2_query_semantic_adapter_v2.py"
      ),
      "serialized_verifier": (
          PROJECT_ROOT
          / "benchmark_v2_query_semantic_adapter_verifier_v2.py"
      ),
      "schema": SCHEMA_PATH,
      "fixed_lineage": FIXED_LINEAGE_PATH,
      "fixed19_catalog_policy": FIXED_CATALOG_POLICY_PATH,
      "historical_catalog_definition": (
          PROJECT_ROOT / "joint_interface_program_learner_v2.py"
      ),
      "historical_catalog_spec_loader_definition": (
          PROJECT_ROOT / "joint_interface_program_learner_v3.py"
      ),
      "historical_catalog_trust_snapshot": (
          PROJECT_ROOT / "constraint_catalog_trust_root_v1.py"
      ),
      "catalog_runtime_loader": (
          PROJECT_ROOT / "fixed_program_catalog_protocol_v1.py"
      ),
      "training_provenance": (
          PROJECT_ROOT / "benchmark_v2_training_provenance.py"
      ),
      "opaque_case_identity": PROJECT_ROOT / "mate_pose_miner.py",
  }
  captures = tuple(
      _capture_project_file(path, label=name)
      for name, path in sorted(files.items())
  )
  return {
      "git_revision": _git_revision(),
      "implementation": {
          name: captured.content_binding()
          for (name, _path), captured in zip(
              sorted(files.items()), captures, strict=True
          )
      },
  }, captures


def _validate_authority_type(
    authority: Any,
    *,
    context: _FixedContext,
) -> AuthenticatedQueryMateSemanticFullDomain:
  if type(authority) is not AuthenticatedQueryMateSemanticFullDomain:
    raise TypeError(
        "V19 model adapter requires the exact verified full-domain handle"
    )
  if (
      re.fullmatch(r"[0-9a-f]{64}", authority.receipt_sha256) is None
      or authority.receipt_sha256
      != context.authority_receipt_binding["sha256"]
  ):
    raise ValueError("verified full-domain handle lineage differs")
  return authority


def _supervision_cluster_ids(
    *,
    context: _FixedContext,
    raw_case_id: str,
    domain: Mapping[str, Any],
) -> dict[str, str]:
  """Derive opaque grouping IDs from the fixed manifest lineage only."""

  family_sha256 = str(domain.get("family_sha256") or "")
  if (
      not raw_case_id
      or re.fullmatch(r"[0-9a-f]{64}", family_sha256) is None
  ):
    raise ValueError("supervision cluster lineage differs")
  lineage = {
      "fixed_v19_manifest_sha256": context.manifest_binding["sha256"],
      "authority_domain_sha256": context.authority_domain_sha256,
  }
  return {
      "family_cluster_id": canonical_sha256({
          "schema_version": "benchmark_v2_family_cluster_id.v1",
          **lineage,
          "family_sha256": family_sha256,
      }),
      "assembly_cluster_id": canonical_sha256({
          "schema_version": "benchmark_v2_assembly_cluster_id.v1",
          **lineage,
          "case_id": raw_case_id,
      }),
  }


def _build_projection(
    authority: AuthenticatedQueryMateSemanticFullDomain,
    *,
    context: _FixedContext,
) -> dict[str, Any]:
  domain_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
  for row in context.domain_rows:
    key = (
        str(row["key"]["case_id"]),
        int(row["key"]["source_contact_ordinal"]),
    )
    if key in domain_by_key:
      raise ValueError("fixed adapter domain contains a duplicate query")
    domain_by_key[key] = row

  ledger_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
  case_sets: dict[str, set[str]] = {"train": set(), "dev": set()}
  family_sets: dict[str, set[str]] = {"train": set(), "dev": set()}
  for split in DEVELOPMENT_SPLITS:
    rows = authority.query_ledger_by_split[split]
    for ledger in rows:
      if (
          not isinstance(ledger, Mapping)
          or ledger.get("development_split") != split
      ):
        raise ValueError("semantic authority ledger split differs")
      key = (
          str(ledger.get("case_id") or ""),
          int(ledger.get("source_contact_ordinal")),
      )
      domain = domain_by_key.get(key)
      if (
          domain is None
          or domain.get("split") != split
          or key in ledger_by_key
          or ledger.get("status") not in {"authorized", "rejected"}
      ):
        raise ValueError("semantic authority ledger domain differs")
      ledger_by_key[key] = ledger
      case_sets[split].add(key[0])
      family_sets[split].add(str(domain.get("family_sha256") or ""))
  if set(ledger_by_key) != set(domain_by_key):
    raise ValueError("semantic authority ledger does not close fixed domain")
  if case_sets["train"] & case_sets["dev"]:
    raise ValueError("train/dev case split contamination")
  if family_sets["train"] & family_sets["dev"]:
    raise ValueError("train/dev family split contamination")
  if "" in family_sets["train"] | family_sets["dev"]:
    raise ValueError("adapter domain lacks family identity")

  opaque_by_raw: dict[str, str] = {}
  raw_by_opaque: dict[str, str] = {}
  for case_id in sorted(case_sets["train"] | case_sets["dev"]):
    public = context.public_by_case[case_id]
    public_case = {
        key: copy.deepcopy(value)
        for key, value in public.items()
        if key != "family_provenance"
    }
    opaque = _opaque_case_id(dict(public_case))
    if opaque in raw_by_opaque and raw_by_opaque[opaque] != case_id:
      raise ValueError("opaque case identity collision")
    opaque_by_raw[case_id] = opaque
    raw_by_opaque[opaque] = case_id

  query_ids: dict[tuple[str, int], str] = {}
  for case_key in sorted(domain_by_key):
    opaque = opaque_by_raw[case_key[0]]
    query_ids[(opaque, case_key[1])] = canonical_sha256(
        {
            "schema_version": "benchmark_v2_opaque_query_identity.v1",
            "authority_receipt_sha256": authority.receipt_sha256,
            "opaque_program_case_id": opaque,
            "source_contact_ordinal": case_key[1],
        }
    )

  program_rows: dict[str, list[Mapping[str, Any]]] = {}
  rows_by_query: dict[
      tuple[str, int], list[Mapping[str, Any]]
  ] = {}
  seen_program_keys: set[tuple[str, int, str, str]] = set()
  for split in DEVELOPMENT_SPLITS:
    program_rows[split] = []
    for raw_row in authority.program_rows_by_split[split]:
      if (
          not isinstance(raw_row, Mapping)
          or raw_row.get("development_split") != split
      ):
        raise ValueError("semantic program row split differs")
      row = _plain_json(raw_row)
      source = row.get("source_contact")
      if not isinstance(source, Mapping):
        raise ValueError("semantic program source contact differs")
      opaque = str(row.get("case_id") or "")
      ordinal = source.get("contact_index")
      direction = str(source.get("direction") or "")
      program_id = str(row.get("program_id") or "")
      if (
          opaque not in raw_by_opaque
          or isinstance(ordinal, bool)
          or not isinstance(ordinal, int)
          or direction
          not in {
              "entity_one_to_entity_two",
              "entity_two_to_entity_one",
          }
          or not program_id
      ):
        raise ValueError("semantic program identity differs")
      raw_case = raw_by_opaque[opaque]
      domain = domain_by_key.get((raw_case, ordinal))
      if domain is None or domain.get("split") != split:
        raise ValueError("semantic program lies outside fixed split domain")
      key = (opaque, ordinal, direction, program_id)
      if key in seen_program_keys:
        raise ValueError("semantic program identity is duplicated")
      seen_program_keys.add(key)
      rows_by_query.setdefault((opaque, ordinal), []).append(row)
      program_rows[split].append(row)

  for (raw_case, ordinal), ledger in ledger_by_key.items():
    opaque_key = (opaque_by_raw[raw_case], ordinal)
    observed = rows_by_query.get(opaque_key, [])
    expected = ledger.get("directed_program_count")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected != len(observed)
        or (ledger["status"] == "authorized") != bool(observed)
    ):
      raise ValueError("semantic query/program coverage differs")
    commitment = context.authority_commitments_by_key[(raw_case, ordinal)]
    fixed_programs = commitment.get("programs")
    if not isinstance(fixed_programs, (list, tuple)):
      raise ValueError("fixed authority program commitment differs")
    expected_rows: list[dict[str, Any]] = []
    for program in fixed_programs:
      if not isinstance(program, Mapping):
        raise ValueError("fixed authority program commitment differs")
      program_row = program.get("program_row")
      if not isinstance(program_row, Mapping):
        raise ValueError("fixed authority program row commitment differs")
      plain_row = _plain_json(program_row)
      if (
          program.get("program_id") != plain_row.get("program_id")
          or program.get("direction")
          != plain_row.get("source_contact", {}).get("direction")
          or program.get("program_row_sha256")
          != canonical_sha256(plain_row)
      ):
        raise ValueError("fixed authority program row hash differs")
      plain_row["development_split"] = commitment.get(
          "development_split"
      )
      expected_rows.append(plain_row)
    expected_rows.sort(
        key=lambda row: (
            str(row["source_contact"]["direction"]),
            str(row["program_id"]),
        )
    )
    observed_rows = sorted(
        (_plain_json(row) for row in observed),
        key=lambda row: (
            str(row["source_contact"]["direction"]),
            str(row["program_id"]),
        ),
    )
    if (
        commitment.get("development_split")
        != domain_by_key[(raw_case, ordinal)]["split"]
        or commitment.get("status") != ledger["status"]
        or commitment.get("rejection_reason") != ledger.get("reason")
        or expected_rows != observed_rows
    ):
      raise ValueError("verified handle differs from fixed receipt lineage")

  fixed_catalog = _load_fixed_program_catalog()
  catalog = _program_catalog(
      program_rows["train"],
      query_id_by_row_key=query_ids,
      fixed=fixed_catalog,
  )
  graph_work: list[dict[str, Any]] = []
  graph_work_id_by_key: dict[tuple[str, int, str], str] = {}
  endpoint_cache_keys: set[str] = set()
  for (raw_case, ordinal), domain in sorted(
      domain_by_key.items(),
      key=lambda item: (
          item[1]["split"],
          item[1]["schedule"]["global_schedule_ordinal"],
      ),
  ):
    ledger = ledger_by_key[(raw_case, ordinal)]
    if ledger["status"] != "authorized":
      continue
    opaque = opaque_by_raw[raw_case]
    endpoint_identity = domain.get("endpoint_identity")
    if not isinstance(endpoint_identity, Mapping):
      raise ValueError("authorized query lacks mapped endpoint identity")
    endpoint_a = endpoint_identity.get("endpoint_a")
    endpoint_b = endpoint_identity.get("endpoint_b")
    if not isinstance(endpoint_a, Mapping) or not isinstance(
        endpoint_b, Mapping
    ):
      raise ValueError("authorized query endpoint identity differs")
    mapper = context.mapper_binding_by_case[raw_case]
    evidence_mapper = (
        domain.get("evidence", {}).get("mapper_receipt")
        if isinstance(domain.get("evidence"), Mapping)
        else None
    )
    if (
        not isinstance(evidence_mapper, Mapping)
        or evidence_mapper.get("sha256") != mapper.get("sha256")
    ):
      raise ValueError("query mapper evidence differs from bound receipt")
    source_case = context.source_by_case[raw_case]
    by_role = {
        "a": _endpoint_graph_binding(
            endpoint=endpoint_a,
            source_case=source_case,
            mapper_binding=mapper,
        ),
        "b": _endpoint_graph_binding(
            endpoint=endpoint_b,
            source_case=source_case,
            mapper_binding=mapper,
        ),
    }
    directions = sorted({
        str(row["source_contact"]["direction"])
        for row in rows_by_query[(opaque, ordinal)]
    })
    for direction in directions:
      roles = (
          ("a", "b")
          if direction == "entity_one_to_entity_two"
          else ("b", "a")
      )
      endpoints = {
          "endpoint_a": copy.deepcopy(by_role[roles[0]]),
          "endpoint_b": copy.deepcopy(by_role[roles[1]]),
      }
      for endpoint in endpoints.values():
        endpoint_cache_keys.add(
            endpoint["endpoint_graph_cache_key_sha256"]
        )
      query_id = query_ids[(opaque, ordinal)]
      work_id = canonical_sha256(
          {
              "schema_version": GRAPH_WORK_SCHEMA,
              "opaque_query_identity": query_id,
              "development_split": domain["split"],
              "direction": direction,
              "endpoints": endpoints,
              "model_view_schema_version": MODEL_VIEW_SCHEMA_VERSION,
              "graph_schema_version": GRAPH_SCHEMA_VERSION,
          }
      )
      graph_work_id_by_key[(opaque, ordinal, direction)] = work_id
      work = {
          "schema_version": GRAPH_WORK_SCHEMA,
          "graph_work_id": work_id,
          "opaque_query_identity": query_id,
          "development_split": domain["split"],
          "source_contact_ordinal": ordinal,
          "direction": direction,
          "model_view_schema_version": MODEL_VIEW_SCHEMA_VERSION,
          "graph_schema_version": GRAPH_SCHEMA_VERSION,
          "graph_materialized": False,
          "endpoints": endpoints,
      }
      work["graph_work_payload_sha256"] = canonical_sha256(work)
      graph_work.append(work)

  rows_by_input: dict[
      tuple[str, str, int, str], list[Mapping[str, Any]]
  ] = {}
  for split in DEVELOPMENT_SPLITS:
    for row in program_rows[split]:
      source = row["source_contact"]
      opaque = str(row["case_id"])
      ordinal = int(source["contact_index"])
      direction = str(source["direction"])
      rows_by_input.setdefault(
          (split, opaque, ordinal, direction), []
      ).append(row)

  samples: list[dict[str, Any]] = []
  for (split, opaque, ordinal, direction), input_rows in sorted(
      rows_by_input.items()
  ):
      query_id = query_ids[(opaque, ordinal)]
      raw_case = raw_by_opaque[opaque]
      cluster_ids = _supervision_cluster_ids(
          context=context,
          raw_case_id=raw_case,
          domain=domain_by_key[(raw_case, ordinal)],
      )
      identity = {
          "opaque_case_identity": canonical_sha256(
              {
                  "schema_version": "benchmark_v2_opaque_case_identity.v1",
                  "authority_receipt_sha256": authority.receipt_sha256,
                  "opaque_program_case_id": opaque,
              }
          ),
          "opaque_query_identity": query_id,
          "source_contact_ordinal": ordinal,
          "development_split": split,
          "direction": direction,
          **cluster_ids,
      }
      sample_id = canonical_sha256(
          {
              "schema_version": SAMPLE_SCHEMA,
              **identity,
          }
      )
      targets: list[dict[str, Any]] = []
      target_program_ids: set[str] = set()
      for row in sorted(
          input_rows, key=lambda value: str(value["program_id"])
      ):
        program_id = str(row["program_id"])
        if program_id in target_program_ids:
          raise ValueError(
              "one query/direction contains duplicate target program IDs"
          )
        target_program_ids.add(program_id)
        catalog_entry = _target_catalog_entry(catalog, row)
        residual_rotation = row.get("residual_rotation")
        residual_translation = row.get("residual_translation")
        if (
            not isinstance(residual_rotation, list)
            or len(residual_rotation) != 3
            or any(
                not isinstance(rotation_row, list)
                or len(rotation_row) != 3
                for rotation_row in residual_rotation
            )
            or not isinstance(residual_translation, list)
            or len(residual_translation) != 3
        ):
          raise ValueError("semantic program residual differs")
        descriptor = _catalog_descriptor_from_authority_row(row)
        descriptor_sha256 = canonical_sha256(descriptor)
        is_oov = catalog_entry is None
        target = {
            "program_id": program_id,
            "source_program_row_sha256": canonical_sha256(row),
            "descriptor": descriptor,
            "descriptor_sha256": descriptor_sha256,
            "catalog_index": (
                None if is_oov else catalog_entry["program_index"]
            ),
            "catalog_program_id": (
                None if is_oov else catalog_entry["program_id"]
            ),
            "oov": is_oov,
            "residual_translation": copy.deepcopy(residual_translation),
            "residual_rotation_row_major": [
                value
                for rotation_row in residual_rotation
                for value in rotation_row
            ],
        }
        target["target_payload_sha256"] = canonical_sha256(target)
        targets.append(target)
      if not targets or len(targets) != len(input_rows):
        raise ValueError("query/direction target coverage differs")
      known_target_count = sum(not target["oov"] for target in targets)
      oov_target_count = len(targets) - known_target_count
      query_id = query_ids[(opaque, ordinal)]
      sample = {
          "schema_version": SAMPLE_SCHEMA,
          "sample_identity_sha256": sample_id,
          **identity,
          "graph_work_id": graph_work_id_by_key[
              (opaque, ordinal, direction)
          ],
          "target_count": len(targets),
          "known_target_count": known_target_count,
          "oov_target_count": oov_target_count,
          "classification_evaluable": known_target_count > 0,
          "targets": targets,
      }
      sample["sample_payload_sha256"] = canonical_sha256(sample)
      samples.append(sample)
  samples.sort(
      key=lambda row: (
          row["development_split"],
          row["opaque_query_identity"],
          row["source_contact_ordinal"],
          row["direction"],
      )
  )
  graph_work.sort(
      key=lambda row: (
          row["development_split"],
          row["opaque_query_identity"],
          row["direction"],
      )
  )
  rejections: list[dict[str, Any]] = []
  for (raw_case, ordinal), ledger in sorted(ledger_by_key.items()):
    if ledger["status"] != "rejected":
      continue
    opaque = opaque_by_raw[raw_case]
    row = {
        "schema_version": REJECTION_SCHEMA,
        "opaque_query_identity": query_ids[(opaque, ordinal)],
        "development_split": domain_by_key[(raw_case, ordinal)]["split"],
        "source_contact_ordinal": ordinal,
        "reason": str(ledger.get("reason") or ""),
        "directed_program_count": ledger["directed_program_count"],
    }
    if not row["reason"] or row["directed_program_count"] != 0:
      raise ValueError("rejected semantic query ledger differs")
    row["rejection_payload_sha256"] = canonical_sha256(row)
    rejections.append(row)
  rejections.sort(
      key=lambda row: (
          row["development_split"], row["opaque_query_identity"]
      )
  )

  sample_counts = {
      split: sum(
          row["development_split"] == split for row in samples
      )
      for split in DEVELOPMENT_SPLITS
  }
  target_counts = {
      split: sum(
          sample["target_count"]
          for sample in samples
          if sample["development_split"] == split
      )
      for split in DEVELOPMENT_SPLITS
  }
  known_target_counts = {
      split: sum(
          sample["known_target_count"]
          for sample in samples
          if sample["development_split"] == split
      )
      for split in DEVELOPMENT_SPLITS
  }
  oov_target_counts = {
      split: sum(
          sample["oov_target_count"]
          for sample in samples
          if sample["development_split"] == split
      )
      for split in DEVELOPMENT_SPLITS
  }
  evaluable_sample_counts = {
      split: sum(
          sample["development_split"] == split
          and sample["classification_evaluable"]
          for sample in samples
      )
      for split in DEVELOPMENT_SPLITS
  }
  authorized_query_counts = {
      split: sum(
          row["development_split"] == split
          and row["status"] == "authorized"
          for (case_id, ordinal), row in (
              (
                  key,
                  {
                      **ledger,
                      "development_split": domain_by_key[key]["split"],
                  },
              )
              for key, ledger in ledger_by_key.items()
          )
      )
      for split in DEVELOPMENT_SPLITS
  }
  coverage = {
      "selected_query_count": len(domain_by_key),
      "authorized_query_count": sum(authorized_query_counts.values()),
      "rejected_query_count": len(rejections),
      "target_count": sum(target_counts.values()),
      "known_target_count": sum(known_target_counts.values()),
      "oov_target_count": sum(oov_target_counts.values()),
      "sample_count": len(samples),
      "classification_evaluable_sample_count": sum(
          evaluable_sample_counts.values()
      ),
      "classification_failure_sample_count": (
          len(samples) - sum(evaluable_sample_counts.values())
      ),
      "unique_case_count": len(opaque_by_raw),
      "unique_query_count": len(domain_by_key),
      "unique_authorized_query_count": sum(
          authorized_query_counts.values()
      ),
      "unique_graph_work_count": len(graph_work),
      "unique_endpoint_graph_count": len(endpoint_cache_keys),
      "by_split": {
          split: {
              "selected_query_count": sum(
                  domain["split"] == split
                  for domain in domain_by_key.values()
              ),
              "authorized_query_count": authorized_query_counts[split],
              "rejected_query_count": sum(
                  row["development_split"] == split
                  for row in rejections
              ),
              "target_count": target_counts[split],
              "known_target_count": known_target_counts[split],
              "oov_target_count": oov_target_counts[split],
              "sample_count": sample_counts[split],
              "classification_evaluable_sample_count": (
                  evaluable_sample_counts[split]
              ),
              "classification_failure_sample_count": (
                  sample_counts[split] - evaluable_sample_counts[split]
              ),
              "unique_case_count": len(case_sets[split]),
              "unique_family_count": len(family_sets[split]),
              "unique_graph_work_count": sum(
                  row["development_split"] == split
                  for row in graph_work
              ),
          }
          for split in DEVELOPMENT_SPLITS
      },
  }
  if (
      coverage["target_count"] != sum(map(len, program_rows.values()))
      or coverage["known_target_count"] + coverage["oov_target_count"]
      != coverage["target_count"]
      or coverage["sample_count"] != coverage["unique_graph_work_count"]
      or any(
          sample["target_count"] != len(sample["targets"])
          or sample["known_target_count"] + sample["oov_target_count"]
          != sample["target_count"]
          or sample["classification_evaluable"]
          != (sample["known_target_count"] > 0)
          for sample in samples
      )
  ):
    raise ValueError("adapter sample/target coverage does not close")

  producer, producer_captures = _producer_binding()
  payload = {
      "schema_version": SCHEMA_VERSION,
      "development": True,
      "formal": False,
      "publication_eligible": False,
      "formal_promotion_blocked": True,
      "final_test_touched": False,
      "withheld_test_touched": False,
      "scope": "v19_full_domain_authenticated_graph_work_index_v2_only",
      "source_authority": {
          "schema_version": (
              "neurocad_query_mate_semantic_full_domain_authority.v1"
          ),
          "authenticated_receipt_sha256": authority.receipt_sha256,
          "receipt_payload_sha256": (
              context.authority_receipt_payload_sha256
          ),
          "domain_sha256": context.authority_domain_sha256,
          "source_revision": (
              "774853033ca097a0c2ebaf27de0320f86f202768"
          ),
          "receipt": dict(context.authority_receipt_binding),
          "fixed_v19_manifest": {
              "logical_id": context.manifest_binding["logical_id"],
              "sha256": context.manifest_binding["sha256"],
              "bytes": context.manifest_binding["bytes"],
          },
          "fixed_family_source": {
              "logical_id": context.family_source_binding["logical_id"],
              "sha256": context.family_source_binding["sha256"],
              "bytes": context.family_source_binding["bytes"],
          },
      },
      "producer": producer,
      "model_protocol": {
          "model_view_schema_version": MODEL_VIEW_SCHEMA_VERSION,
          "graph_schema_version": GRAPH_SCHEMA_VERSION,
          "graph_policy": (
              "query_direction_cached_endpoint_local_brep_graphs.v1"
          ),
          "graphs_materialized": False,
          "supervision_group_contract": {
              "schema_version": (
                  "benchmark_v2_supervision_group_contract.v1"
              ),
              "fields": [
                  "assembly_cluster_id",
                  "family_cluster_id",
              ],
              "derivation": (
                  "fixed_v19_manifest_and_domain_lineage_hash.v1"
              ),
              "allowed_uses": [
                  "clustered_resampling",
                  "training_weighting",
              ],
              "model_feature_access": "forbidden",
          },
          "evaluation_contract": {
              "schema_version": (
                  "benchmark_v2_explicit_oov_evaluation_contract.v1"
              ),
              "classification_evaluable_rule": (
                  "sample_has_at_least_one_in_catalog_gold.v1"
              ),
              "oov_only_metric_policy": (
                  "recall_at_k_and_exact_count_as_failure.v1"
              ),
              "mixed_target_policy": (
                  "known_targets_remain_matchable_oov_targets_disclosed.v1"
              ),
              "target_retention": "all_authorized_targets_preserved.v1",
          },
      },
      "program_catalog": catalog,
      "graph_work": graph_work,
      "samples": samples,
      "rejected_queries": rejections,
      "coverage": coverage,
  }
  payload["index_payload_sha256"] = canonical_sha256(payload)
  for captured in producer_captures:
    _reverify_capture(captured, label=captured.logical_id)
  for captured in fixed_catalog.captures:
    _reverify_capture(captured, label=captured.logical_id)
  return payload


class AuthenticatedBenchmarkV2QuerySemanticIndexV2:
  """In-process capability retaining the verified full-domain authority."""

  __slots__ = ("_payload", "_authority", "_sealed")

  def __init__(
      self,
      *,
      payload: Mapping[str, Any],
      authority: AuthenticatedQueryMateSemanticFullDomain,
      _token: object,
  ) -> None:
    if _token is not _INDEX_FACTORY_TOKEN:
      raise TypeError("query semantic index is builder-only")
    object.__setattr__(self, "_payload", _canonical_bytes(payload))
    object.__setattr__(self, "_authority", authority)
    object.__setattr__(self, "_sealed", True)

  def __setattr__(self, _name: str, _value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("authenticated query semantic index is immutable")
    object.__setattr__(self, _name, _value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated query semantic index is not serializable")

  @property
  def payload_sha256(self) -> str:
    return _sha256_bytes(self._payload)

  @property
  def coverage(self) -> Mapping[str, Any]:
    payload = json.loads(self._payload)
    return MappingProxyType(copy.deepcopy(payload["coverage"]))

  @property
  def catalog_summary(self) -> Mapping[str, Any]:
    payload = json.loads(self._payload)
    catalog = payload["program_catalog"]
    return MappingProxyType(
        {
            "entry_count": catalog["entry_count"],
            "candidate_policy": catalog["candidate_policy"],
            "catalog_roster_sha256": catalog["fixed_source"][
                "catalog_roster_sha256"
            ],
            "source_split": catalog["training_statistics"][
                "source_split"
            ],
            "source_row_count": catalog["training_statistics"][
                "source_row_count"
            ],
            "known_target_count": catalog["training_statistics"][
                "known_target_count"
            ],
            "oov_target_count": catalog["training_statistics"][
                "oov_target_count"
            ],
            "catalog_payload_sha256": catalog[
                "catalog_payload_sha256"
            ],
        }
    )

  def projection(self) -> Mapping[str, Any]:
    if type(self._authority) is not AuthenticatedQueryMateSemanticFullDomain:
      raise TypeError("query semantic index lost its authority handle")
    return MappingProxyType(json.loads(self._payload))


_INDEX_FACTORY_TOKEN = object()


def build_benchmark_v2_query_semantic_index_v2(
    authority: AuthenticatedQueryMateSemanticFullDomain,
) -> AuthenticatedBenchmarkV2QuerySemanticIndexV2:
  """Build a deterministic index from one verified V19 full-domain handle."""

  if type(authority) is not AuthenticatedQueryMateSemanticFullDomain:
    raise TypeError(
        "V19 model adapter requires the exact verified full-domain handle"
    )
  context = _load_fixed_context()
  verified = _validate_authority_type(authority, context=context)
  payload = _build_projection(
      verified,
      context=context,
  )
  for captured in context.captures:
    _reverify_capture(captured, label=captured.logical_id)
  return AuthenticatedBenchmarkV2QuerySemanticIndexV2(
      payload=payload,
      authority=verified,
      _token=_INDEX_FACTORY_TOKEN,
  )


def write_benchmark_v2_query_semantic_index_v2(
    index: AuthenticatedBenchmarkV2QuerySemanticIndexV2,
    *,
    output_path: str | Path,
) -> Path:
  """Atomically publish a new development index without overwrite."""

  if type(index) is not AuthenticatedBenchmarkV2QuerySemanticIndexV2:
    raise TypeError("index writer requires an authenticated adapter index")
  destination = Path(output_path).absolute()
  allowed_root = (
      PROJECT_ROOT / "artifacts" / "development"
  ).resolve(strict=True)
  try:
    relative = destination.relative_to(allowed_root)
  except ValueError as error:
    raise ValueError(
        "graph-work index output must remain under artifacts/development"
    ) from error
  if (
      destination.name != "authenticated_query_graph_work_index.v2.json"
      or any(
          re.search(r"(^|[_\-.])(formal|final|withheld)($|[_\-.])", part.lower())
          for part in relative.parts
      )
  ):
    raise ValueError("graph-work index output name or development scope differs")
  current = allowed_root
  _reject_reparse_chain(current)
  for component in relative.parts[:-1]:
    current = current / component
    try:
      current.mkdir()
    except FileExistsError:
      pass
    _reject_reparse_chain(current)
    if not current.is_dir():
      raise ValueError("graph-work index output parent differs")
  _reject_reparse_chain(destination, allow_missing_leaf=True)
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  if destination.exists() or temporary.exists():
    raise FileExistsError("adapter index destination already exists")
  payload = dict(index.projection())
  if canonical_sha256(
      {key: value for key, value in payload.items()
       if key != "index_payload_sha256"}
  ) != payload.get("index_payload_sha256"):
    raise ValueError("adapter index self-hash differs")
  serialized = _canonical_bytes(payload)
  for value in _walk_strings(payload):
    if Path(value).is_absolute() or value.startswith(("/", "\\")):
      raise ValueError("serialized graph-work index contains an absolute path")
  descriptor = os.open(
      temporary,
      os.O_CREAT | os.O_EXCL | os.O_WRONLY,
      0o600,
  )
  linked = False
  completed = False
  temporary_identity: tuple[int, int] | None = None
  try:
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
      stream.write(serialized)
      stream.flush()
      os.fsync(stream.fileno())
    _reject_reparse_chain(temporary)
    temporary_identity = _plain_file_identity(
        temporary, label="graph-work index temporary"
    )
    try:
      os.link(temporary, destination)
    except FileExistsError as error:
      raise FileExistsError(
          "adapter index destination already exists"
      ) from error
    linked = True
    if _plain_file_identity(
        destination, label="published graph-work index"
    ) != temporary_identity:
      raise ValueError(
          "published graph-work index is not the temporary file inode"
      )
    with destination.open("rb") as stream:
      opened_identity = _stat_identity(os.fstat(stream.fileno()))
      observed = stream.read()
    if (
        opened_identity != temporary_identity
        or observed != serialized
        or _plain_file_identity(
            destination, label="published graph-work index"
        ) != temporary_identity
        or _plain_file_identity(
            temporary, label="graph-work index temporary"
        ) != temporary_identity
    ):
      raise ValueError("published graph-work index bytes differ")
    try:
      directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    except OSError:
      directory_descriptor = None
    if directory_descriptor is not None:
      try:
        os.fsync(directory_descriptor)
      finally:
        os.close(directory_descriptor)
    if (
        _plain_file_identity(
            destination, label="published graph-work index"
        ) != temporary_identity
        or _plain_file_identity(
            temporary, label="graph-work index temporary"
        ) != temporary_identity
    ):
      raise ValueError("published graph-work index identity changed")
    completed = True
  finally:
    if temporary_identity is not None:
      _unlink_only_if_identity(temporary, temporary_identity)
    if linked and not completed and temporary_identity is not None:
      _unlink_only_if_identity(destination, temporary_identity)
  return destination


def _walk_strings(value: Any):
  if isinstance(value, str):
    yield value
  elif isinstance(value, Mapping):
    for item in value.values():
      yield from _walk_strings(item)
  elif isinstance(value, (list, tuple)):
    for item in value:
      yield from _walk_strings(item)


def _stat_identity(observed: os.stat_result) -> tuple[int, int]:
  return int(observed.st_dev), int(observed.st_ino)


def _plain_file_identity(
    path: Path, *, label: str
) -> tuple[int, int]:
  try:
    observed = os.lstat(path)
  except FileNotFoundError as error:
    raise ValueError(f"{label} disappeared") from error
  if not stat.S_ISREG(observed.st_mode) or _is_reparse(observed):
    raise ValueError(f"{label} is not a plain file")
  return _stat_identity(observed)


def _unlink_only_if_identity(
    path: Path, expected: tuple[int, int]
) -> bool:
  """Remove only this call's inode; preserve any later path replacement."""

  try:
    observed = os.lstat(path)
  except FileNotFoundError:
    return False
  if (
      not stat.S_ISREG(observed.st_mode)
      or _is_reparse(observed)
      or _stat_identity(observed) != expected
  ):
    return False
  try:
    path.unlink()
  except FileNotFoundError:
    return False
  return True
