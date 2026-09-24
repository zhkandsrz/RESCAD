"""Build face/interface training data for the learned interface scorer.

This script uses Fusion assembly metadata only to create train labels. The
serialized samples contain single-part local interface frames and local face
features, never pairwise relative poses.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence

from .benchmark_v2_model_view import (
    BENCHMARK_V2_FEATURE_NAMES,
    ModelViewSanitization,
)
from .benchmark_v2_protocol import (
    BenchmarkV2PartSpec,
    PreparedBenchmarkV2Assembly,
    prepare_benchmark_v2_assembly,
)
from .benchmark_v2_private_supervision import (
    BenchmarkV2PrivateSupervision,
    load_benchmark_v2_private_supervision,
    require_certified_binary_training_gold,
)
from .benchmark_v2_constants import PROTOCOL_VERSION
from .benchmark_v2_training_provenance import (
    BenchmarkV2AuthenticatedTrainingReceipts,
    CapturedJsonArtifact,
    FORMAL_CONTACT_CENSUS_SCHEMA_ALLOWLIST,
    INTERFACE_TRAINING_ROW_KEYS,
    INTERFACE_TRAINING_ROW_SCHEMA,
    FUSION_STEP_FACE_MAP_BINDING_SCHEMA,
    canonical_sha256,
    capture_json_artifact,
    case_token_sha256,
    _new_authenticated_training_receipts,
    validate_face_map_binding as _validated_face_map_binding,
    validate_family_split_binding,
    validate_private_supervision_binding,
    write_dataset_artifact,
)
from .fusion_face_mapper import build_mapped_endpoint_lookup
from .batch_infer_assemble import (
    _index_step_files,
    _load_cases,
    _parse_dataset_roots,
    _resolve_part_path,
)
from .interface_features import (
    CandidateInterface,
    extract_candidate_interfaces_from_step,
    label_interfaces_from_assembly_contacts,
)
from .paths import resolve_dataset_roots, resolve_path


@dataclass(frozen=True, slots=True)
class BenchmarkV2InterfaceRecord:
  """Private producer record; face identities never enter serialized rows."""

  opaque_part: str
  opaque_body: str
  candidate: CandidateInterface
  model_view: ModelViewSanitization
  row: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkV2ProducedInterfaceCase:
  """One prepared Fusion case and its leakage-safe single-interface rows."""

  rows: tuple[dict[str, Any], ...]
  records_by_body: Mapping[str, tuple[BenchmarkV2InterfaceRecord, ...]]
  contact_face_group_pairs: frozenset[
      tuple[str, frozenset[int], str, frozenset[int]]
  ]
  prepared: PreparedBenchmarkV2Assembly
  seed: int
  producer_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkV2AuthenticatedTrainingInputs:
  """One path-authenticated, closed-set in-memory producer input bundle."""

  public_cases: tuple[Mapping[str, Any], ...]
  case_ids: tuple[str, ...]
  private_supervision: BenchmarkV2PrivateSupervision
  face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]]
  face_map_binding: Mapping[str, Any]
  receipts: BenchmarkV2AuthenticatedTrainingReceipts


@dataclass(frozen=True, slots=True)
class PrivateMappedFusionEndpoint:
  """Receipt-verified endpoint kept entirely inside producer code."""

  raw_part: str
  geometry_asset: str
  raw_body: str
  opaque_body: str
  raw_occ_face_indices: tuple[int, ...]
  source_face_signature_sha256s: tuple[str, ...]
  randomized_face_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedLiveInput:
  """One private live file pinned to an immutable family-source receipt."""

  path: Path
  label: str
  expected_bytes: int
  expected_sha256: str
  expected_sha1: str | None = None


@dataclass(frozen=True, slots=True)
class _VerifiedBenchmarkV2CaseInputs:
  """Private verification handle used for the post-load TOCTOU check."""

  case_id: str
  files: tuple[_VerifiedLiveInput, ...]
  staging_directory: Any = field(default=None, repr=False, compare=False)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--cases_json",
      default="neurocad/paper_splits_candidate_main_deepseek_v1/train_oracle.json",
      help="Train split JSON. Use train/dev only, never the held-out test split.",
  )
  parser.add_argument("--dataset_root", default=None)
  parser.add_argument("--dataset_roots", nargs="*", default=[])
  parser.add_argument("--output_jsonl", default="neurocad/interface_train.jsonl")
  parser.add_argument("--summary_json", default=None)
  parser.add_argument("--max_cases", type=int, default=0)
  parser.add_argument("--max_candidates_per_part", type=int, default=0)
  parser.add_argument(
      "--input_protocol",
      "--protocol",
      dest="input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )
  parser.add_argument("--frame_randomization_seed", type=int, default=17)
  parser.add_argument(
      "--benchmark_v2_translation_box_fraction",
      type=float,
      default=1.0,
  )
  parser.add_argument("--source_split", choices=["train", "dev"], default="train")
  parser.add_argument("--family_split_manifest", default=None)
  parser.add_argument(
      "--private-source-split",
      dest="private_source_split",
      default=None,
      help="Custodian-only source/instance sidecar for the selected train/dev split.",
  )
  parser.add_argument(
      "--private-evaluation-gold-split",
      dest="private_evaluation_gold_split",
      default=None,
      help="Evaluator-only contact-gold sidecar for the selected train/dev split.",
  )
  parser.add_argument(
      "--fusion-step-face-map-receipt",
      dest="fusion_step_face_map_receipt",
      default=None,
      help=(
          "Private receipt mapping Fusion contact face keys to verified STEP/OCC "
          "faces. Required for formal benchmark_v2 label production."
      ),
  )
  return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  """Anchor all CLI paths at the checkout, independent of process CWD."""
  roots = resolve_dataset_roots(args.dataset_root, args.dataset_roots)
  args.dataset_root = str(roots[0])
  args.dataset_roots = [str(path) for path in roots[1:]]
  args.cases_json = str(resolve_path(args.cases_json))
  args.output_jsonl = str(resolve_path(args.output_jsonl))
  if args.summary_json:
    args.summary_json = str(resolve_path(args.summary_json))
  if args.family_split_manifest:
    args.family_split_manifest = str(resolve_path(args.family_split_manifest))
  if args.private_source_split:
    args.private_source_split = str(resolve_path(args.private_source_split))
  if args.private_evaluation_gold_split:
    args.private_evaluation_gold_split = str(
        resolve_path(args.private_evaluation_gold_split)
    )
  if args.fusion_step_face_map_receipt:
    args.fusion_step_face_map_receipt = str(
        resolve_path(args.fusion_step_face_map_receipt)
    )
  return args


def main() -> None:
  args = normalize_paths(build_parser().parse_args())
  if str(args.input_protocol) == "benchmark_v2":
    _main_benchmark_v2(args)
    return
  _main_legacy(args)


def _sha256_hex(value: Any, *, name: str) -> str:
  text = str(value or "").strip().lower().removeprefix("sha256:")
  if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
  return text


def _fully_mapped_face_receipt_case_ids(
    receipt_cases: Sequence[Any],
) -> tuple[str, ...]:
  complete: list[str] = []
  for raw_case in receipt_cases:
    if not isinstance(raw_case, Mapping):
      continue
    mappings = raw_case.get("face_mappings")
    if (
        isinstance(mappings, list)
        and mappings
        and all(
            isinstance(mapping, Mapping)
            and mapping.get("status") == "mapped_unique"
            for mapping in mappings
        )
    ):
      complete.append(str(raw_case.get("case_id") or ""))
  return tuple(sorted(complete))


def _load_fusion_step_face_map_receipt(
    receipt_path: str | Path,
    *,
    family_split_manifest: str | Path,
    case_ids: Sequence[str],
    _receipt_capture: CapturedJsonArtifact | None = None,
    _family_manifest_capture: CapturedJsonArtifact | None = None,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
  """Load a private map receipt and bind it to the authoritative family source."""

  receipt_capture = _receipt_capture or capture_json_artifact(
      receipt_path,
      label="Fusion STEP face-map receipt",
  )
  payload = receipt_capture.payload
  if not isinstance(payload, Mapping):
    raise ValueError("Fusion STEP face-map receipt must contain a JSON object")
  try:
    lookup = build_mapped_endpoint_lookup(payload)
  except Exception as error:
    raise ValueError(f"Fusion STEP face-map receipt contract failed: {error}") from error

  normalized_case_ids = [str(case_id).strip() for case_id in case_ids]
  if (
      not normalized_case_ids
      or not all(normalized_case_ids)
      or len(set(normalized_case_ids)) != len(normalized_case_ids)
  ):
    raise ValueError("Fusion STEP face-map binding requires unique nonempty case IDs")
  receipt_cases = payload.get("cases")
  if not isinstance(receipt_cases, list):
    raise ValueError("Fusion STEP face-map receipt cases are malformed")
  receipt_case_ids = [
      str(case.get("case_id") or "").strip()
      for case in receipt_cases
      if isinstance(case, Mapping)
  ]
  if (
      len(receipt_case_ids) != len(receipt_cases)
      or any(not case_id for case_id in receipt_case_ids)
      or len(set(receipt_case_ids)) != len(receipt_case_ids)
      or set(receipt_case_ids) != set(normalized_case_ids)
  ):
    raise ValueError(
        "Fusion STEP face-map receipt case set differs from the frozen split"
    )
  lookup_case_ids = {key[0] for key in lookup}
  missing_mapped_cases = sorted(set(normalized_case_ids) - lookup_case_ids)
  if missing_mapped_cases:
    raise ValueError(
        "Fusion STEP face-map receipt has no mapped endpoints for frozen cases: "
        + ", ".join(missing_mapped_cases[:10])
    )

  family_manifest_capture = _family_manifest_capture or capture_json_artifact(
      family_split_manifest,
      label="family split manifest for face-map binding",
  )
  manifest = family_manifest_capture.payload
  artifact_hashes = manifest.get("artifact_hashes") if isinstance(manifest, Mapping) else None
  source_artifact = (
      artifact_hashes.get("input_cases")
      if isinstance(artifact_hashes, Mapping)
      else None
  )
  if not isinstance(source_artifact, Mapping):
    raise ValueError("family split manifest lacks authoritative input_cases hash")
  expected_family_source_sha = _sha256_hex(
      source_artifact.get("sha256"),
      name="family split input_cases sha256",
  )
  source = payload.get("source")
  if not isinstance(source, Mapping):
    raise ValueError("Fusion STEP face-map receipt lacks family source binding")
  receipt_family_source_sha = _sha256_hex(
      source.get("family_source_sha256"),
      name="face-map family_source_sha256",
  )
  if receipt_family_source_sha != expected_family_source_sha:
    raise ValueError(
        "Fusion STEP face-map receipt does not match the family split source"
    )

  contract = payload.get("endpoint_lookup_contract")
  binding = _validated_face_map_binding(
      {
          "schema_version": FUSION_STEP_FACE_MAP_BINDING_SCHEMA,
          "binding_mode": (
              "private_formal_receipt"
              if payload.get("formal_gold_eligible") is True
              and payload.get("gold_interface_certificate_ready") is True
              else "private_development_receipt"
          ),
          "receipt_sha256": receipt_capture.sha256,
          "receipt_payload_sha256": payload.get("receipt_payload_sha256"),
          "endpoint_lookup_schema_version": (
              contract.get("schema_version")
              if isinstance(contract, Mapping)
              else None
          ),
          "part_semantics": (
              contract.get("part_semantics")
              if isinstance(contract, Mapping)
              else None
          ),
          "family_source_sha256": receipt_family_source_sha,
          "mapped_case_set_sha256": canonical_sha256(
              list(_fully_mapped_face_receipt_case_ids(receipt_cases))
          ),
          "formal_gold_eligible": payload.get("formal_gold_eligible"),
          "gold_interface_certificate_ready": payload.get(
              "gold_interface_certificate_ready"
          ),
      },
      require_private_receipt=False,
  )
  return lookup, binding


def _fully_mapped_private_gold_case_ids(
    private_supervision: BenchmarkV2PrivateSupervision,
    *,
    case_ids: Sequence[str],
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[str, ...]:
  complete: list[str] = []
  for case_id in case_ids:
    source_case = private_supervision.source_case(case_id)
    identity_by_part = {
        str(part): (str(asset), str(body))
        for part, asset, body in zip(
            source_case["selected_part_names"],
            source_case["selected_geometry_assets"],
            source_case["selected_body_uuids"],
        )
    }
    gold_case = private_supervision.gold_case(case_id)
    contacts = gold_case.get("contacts")
    if not isinstance(contacts, list) or not contacts:
      continue
    expected_endpoint_count = 0
    case_is_complete = True
    for contact in contacts:
      if not isinstance(contact, Mapping):
        case_is_complete = False
        break
      for endpoint_name in ("endpoint_a", "endpoint_b"):
        endpoint = contact.get(endpoint_name)
        if not isinstance(endpoint, Mapping):
          case_is_complete = False
          break
        part = str(endpoint.get("part") or "")
        fusion_index = endpoint.get("fusion_face_index")
        expected_endpoint_count += 1
        mapping = (
            face_map_lookup.get((case_id, part, fusion_index))
            if isinstance(fusion_index, int) and not isinstance(fusion_index, bool)
            else None
        )
        expected_identity = identity_by_part.get(part)
        if (
            not isinstance(mapping, Mapping)
            or expected_identity is None
            or mapping.get("status") != "mapped_unique"
            or str(mapping.get("geometry_asset") or "") != expected_identity[0]
            or str(mapping.get("body_uuid") or "") != expected_identity[1]
        ):
          case_is_complete = False
          break
      if not case_is_complete:
        break
    if case_is_complete and expected_endpoint_count > 0:
      complete.append(case_id)
  return tuple(sorted(complete))


def load_benchmark_v2_authenticated_training_inputs(
    *,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    family_split_manifest_path: str | Path,
    face_map_receipt_path: str | Path,
    source_split: str,
    contact_census_receipt_path: str | Path | None = None,
) -> BenchmarkV2AuthenticatedTrainingInputs:
  """Capture and close every currently implemented benchmark-v2 input binding.

  The current repository has no allowlisted contact-census schema. Consequently
  the returned receipt set remains useful for source-positive development, but
  can never authorize a formal binary writer.
  """

  public_capture = capture_json_artifact(
      public_cases_path,
      label="public frozen split cases",
  )
  family_capture = capture_json_artifact(
      family_split_manifest_path,
      label="family split manifest",
  )
  face_capture = capture_json_artifact(
      face_map_receipt_path,
      label="Fusion STEP face-map receipt",
  )
  public_cases = public_capture.payload
  if not isinstance(public_cases, list) or not public_cases or any(
      not isinstance(case, Mapping) for case in public_cases
  ):
    raise ValueError("public frozen split cases must be a nonempty JSON list")
  case_ids = tuple(str(case.get("id") or "").strip() for case in public_cases)
  if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
    raise ValueError("public frozen split case IDs must be nonempty and unique")
  family_sha = validate_family_split_binding(
      cases_path=public_cases_path,
      manifest_path=family_split_manifest_path,
      source_split=source_split,
      case_ids=case_ids,
      _cases_capture=public_capture,
      _manifest_capture=family_capture,
  )
  private_supervision = load_benchmark_v2_private_supervision(
      source_path=private_source_path,
      gold_path=private_gold_path,
      family_split_manifest_path=family_split_manifest_path,
      source_split=source_split,
      public_case_ids=case_ids,
      _family_manifest_capture=family_capture,
  )
  if private_supervision.producer_binding["family_split_manifest_sha256"] != family_sha:
    raise ValueError("private supervision and public split use different manifests")
  face_map_lookup, face_map_binding = _load_fusion_step_face_map_receipt(
      face_map_receipt_path,
      family_split_manifest=family_split_manifest_path,
      case_ids=case_ids,
      _receipt_capture=face_capture,
      _family_manifest_capture=family_capture,
  )
  fully_mapped_case_ids = _fully_mapped_private_gold_case_ids(
      private_supervision,
      case_ids=case_ids,
      face_map_lookup=face_map_lookup,
  )
  if contact_census_receipt_path is not None:
    census_capture = capture_json_artifact(
        contact_census_receipt_path,
        label="private contact census receipt",
    )
    census_payload = census_capture.payload
    census_schema = (
        str(census_payload.get("schema_version") or "")
        if isinstance(census_payload, Mapping)
        else ""
    )
    if census_schema not in FORMAL_CONTACT_CENSUS_SCHEMA_ALLOWLIST:
      raise ValueError(
          "formal binary training has no authenticated, allowlisted contact "
          "census schema; caller-asserted future schemas are rejected"
      )
    raise AssertionError("contact census allowlist unexpectedly became nonempty")
  receipts = _new_authenticated_training_receipts(
      source_split=source_split,
      case_ids=case_ids,
      family_split_manifest_sha256=family_sha,
      public_cases_file_sha256=public_capture.sha256,
      private_supervision_binding=private_supervision.producer_binding,
      face_map_binding=face_map_binding,
      fully_mapped_case_ids=fully_mapped_case_ids,
      contact_census_binding=None,
  )
  return BenchmarkV2AuthenticatedTrainingInputs(
      public_cases=tuple(dict(case) for case in public_cases),
      case_ids=case_ids,
      private_supervision=private_supervision,
      face_map_lookup=face_map_lookup,
      face_map_binding=face_map_binding,
      receipts=receipts,
  )


def _safe_receipt_relative_path(value: Any, *, suffix: str) -> PurePosixPath:
  text = str(value or "").strip().replace("\\", "/")
  relative = PurePosixPath(text)
  if (
      not text
      or relative.is_absolute()
      or any(part in {"", ".", ".."} for part in relative.parts)
      or relative.suffix.lower() != suffix
  ):
    raise ValueError("family source receipt contains an unsafe member path")
  return relative


def _plain_file_at_receipt_path(
    path: Path,
    *,
    dataset_roots: Sequence[Path],
    receipt_relative: PurePosixPath,
    label: str,
) -> Path:
  """Resolve one bound input without accepting symlinks or path aliasing."""

  candidate = Path(path)
  if not candidate.is_file() or candidate.is_symlink():
    raise ValueError(f"{label} must be an existing plain file")
  resolved = candidate.resolve(strict=True)
  expected_text = receipt_relative.as_posix()
  matched = False
  for raw_root in dataset_roots:
    root = Path(raw_root)
    if not root.is_dir() or root.is_symlink():
      continue
    resolved_root = root.resolve(strict=True)
    expected = root.joinpath(*receipt_relative.parts)
    try:
      relative = resolved.relative_to(resolved_root)
    except ValueError:
      continue
    if relative.as_posix() != expected_text:
      continue
    expected_absolute = os.path.normcase(os.path.abspath(expected))
    if expected_absolute != os.path.normcase(str(resolved)):
      raise ValueError(f"{label} receipt path traverses a symlink or junction")
    cursor = root
    if any(
        (cursor := cursor / part).is_symlink()
        for part in receipt_relative.parts
    ):
      raise ValueError(f"{label} receipt path traverses a symlink")
    matched = True
    break
  if not matched:
    raise ValueError(f"{label} path does not match its family source receipt")
  return resolved


def _validated_live_input(
    path: Path,
    *,
    record: Mapping[str, Any],
    dataset_roots: Sequence[Path],
    suffix: str,
    label: str,
    require_sha1: bool,
) -> tuple[_VerifiedLiveInput, bytes]:
  expected_keys = {"path", "bytes", "sha256"}
  if require_sha1:
    expected_keys.update({"part", "geometry_asset", "body_uuid", "sha1"})
  if set(record) != expected_keys:
    raise ValueError(f"{label} family source receipt schema is invalid")
  expected_bytes = record.get("bytes")
  if (
      not isinstance(expected_bytes, int)
      or isinstance(expected_bytes, bool)
      or expected_bytes < 0
  ):
    raise ValueError(f"{label} receipt byte count is invalid")
  expected_sha256 = _sha256_hex(record.get("sha256"), name=f"{label} sha256")
  expected_sha1: str | None = None
  if require_sha1:
    expected_sha1 = str(record.get("sha1") or "").strip().lower()
    if len(expected_sha1) != 40 or any(
        character not in "0123456789abcdef" for character in expected_sha1
    ):
      raise ValueError(f"{label} sha1 must be a lowercase SHA-1 digest")
  resolved = _plain_file_at_receipt_path(
      path,
      dataset_roots=dataset_roots,
      receipt_relative=_safe_receipt_relative_path(
          record.get("path"),
          suffix=suffix,
      ),
      label=label,
  )
  payload = resolved.read_bytes()
  if (
      len(payload) != expected_bytes
      or hashlib.sha256(payload).hexdigest() != expected_sha256
      or (
          expected_sha1 is not None
          and hashlib.sha1(payload).hexdigest() != expected_sha1
      )
  ):
    raise ValueError(f"{label} does not match its family source receipt")
  return (
      _VerifiedLiveInput(
          path=resolved,
          label=label,
          expected_bytes=expected_bytes,
          expected_sha256=expected_sha256,
          expected_sha1=expected_sha1,
      ),
      payload,
  )


def _reverify_benchmark_v2_case_inputs(
    verification: _VerifiedBenchmarkV2CaseInputs,
) -> None:
  """Fail if a receipt-bound live input changed while producer code used it."""

  for item in verification.files:
    if not item.path.is_file() or item.path.is_symlink():
      raise ValueError(f"{item.label} changed after verified use")
    payload = item.path.read_bytes()
    if (
        len(payload) != item.expected_bytes
        or hashlib.sha256(payload).hexdigest() != item.expected_sha256
        or (
            item.expected_sha1 is not None
            and hashlib.sha1(payload).hexdigest() != item.expected_sha1
        )
    ):
      raise ValueError(f"{item.label} changed after verified use")


def _load_verified_benchmark_v2_case_inputs(
    case: Mapping[str, Any],
    *,
    dataset_roots: Sequence[Path],
    file_index: Mapping[str, list[Path]],
) -> tuple[
    dict[str, Any],
    tuple[BenchmarkV2PartSpec, ...],
    _VerifiedBenchmarkV2CaseInputs,
]:
  """Hash, parse, and bind every live case input to its family receipt."""

  case_id = str(case.get("case_id") or case.get("id") or "").strip()
  if not case_id:
    raise ValueError("benchmark_v2 live input verification requires a case id")
  binding = case.get("source_receipt_binding")
  if not isinstance(binding, Mapping) or set(binding) != {
      "archive",
      "receipt_sha256",
      "assembly_json",
      "body_steps",
      "instances",
  }:
    raise ValueError(f"benchmark_v2 case {case_id!r} lacks a source receipt binding")
  if not str(binding.get("archive") or "").strip():
    raise ValueError("family source receipt archive identity is missing")
  _sha256_hex(binding.get("receipt_sha256"), name="archive receipt_sha256")
  assembly_record = binding.get("assembly_json")
  if not isinstance(assembly_record, Mapping):
    raise ValueError("family source receipt lacks assembly.json binding")
  assembly_path = _assembly_json_for_case(dict(case), list(dataset_roots))
  if assembly_path is None:
    raise ValueError(f"benchmark_v2 case {case_id!r} lacks assembly.json")
  assembly_verified, assembly_bytes = _validated_live_input(
      assembly_path,
      record=assembly_record,
      dataset_roots=dataset_roots,
      suffix=".json",
      label=f"case {case_id} assembly.json",
      require_sha1=False,
  )
  try:
    assembly_data = json.loads(assembly_bytes.decode("utf-8"))
  except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("receipt-bound assembly.json is not valid UTF-8 JSON") from error
  if not isinstance(assembly_data, dict):
    raise ValueError("receipt-bound assembly.json must contain an object")
  # Parse only captured, already-hashed bytes, then ensure the live path did not
  # change during parsing before resolving or loading any STEP member.
  _reverify_benchmark_v2_case_inputs(
      _VerifiedBenchmarkV2CaseInputs(case_id, (assembly_verified,))
  )

  specs = _case_part_specs(
      dict(case),
      dataset_roots=list(dataset_roots),
      file_index=dict(file_index),
  )
  names = case.get("selected_part_names")
  body_steps = binding.get("body_steps")
  if (
      not isinstance(names, list)
      or len(set(str(value) for value in names)) != len(names)
      or not isinstance(body_steps, list)
  ):
    raise ValueError("family source STEP receipt identities are invalid")
  step_by_part: dict[str, Mapping[str, Any]] = {}
  for row in body_steps:
    if not isinstance(row, Mapping):
      raise ValueError("family source STEP receipt row is invalid")
    part = str(row.get("part") or "")
    if not part or part in step_by_part:
      raise ValueError("family source STEP receipt part identity is ambiguous")
    step_by_part[part] = row
  clean_names = [str(value) for value in names]
  if set(step_by_part) != set(clean_names) or len(specs) != len(clean_names):
    raise ValueError("family source STEP receipts do not cover the selected parts")
  verified_files = [assembly_verified]
  staged_specs: list[BenchmarkV2PartSpec] = []
  staging_directory = tempfile.TemporaryDirectory(
      prefix=f"neurocad_v2_{hashlib.sha256(case_id.encode()).hexdigest()[:12]}_"
  )
  try:
    for index, (part, spec) in enumerate(zip(clean_names, specs)):
      step_record = step_by_part[part]
      if (
          step_record.get("body_uuid") != spec.body_uuid
          or step_record.get("geometry_asset") != spec.geometry_asset
      ):
        raise ValueError(
            "family source STEP receipt differs from its assembly instance"
        )
      step_verified, step_payload = _validated_live_input(
          spec.step_path,
          record=step_record,
          dataset_roots=dataset_roots,
          suffix=".step",
          label=f"case {case_id} STEP part {part}",
          require_sha1=True,
      )
      staged_path = Path(staging_directory.name) / f"part_{index:04d}.step"
      staged_path.write_bytes(step_payload)
      staged_verified = _VerifiedLiveInput(
          path=staged_path,
          label=f"case {case_id} verified STEP staging part {part}",
          expected_bytes=step_verified.expected_bytes,
          expected_sha256=step_verified.expected_sha256,
          expected_sha1=step_verified.expected_sha1,
      )
      staged_specs.append(
          BenchmarkV2PartSpec(
              part_key=spec.part_key,
              part_name=spec.part_name,
              body_uuid=spec.body_uuid,
              step_path=staged_path,
              geometry_asset=spec.geometry_asset,
          )
      )
      verified_files.extend((step_verified, staged_verified))
    verification = _VerifiedBenchmarkV2CaseInputs(
        case_id,
        tuple(verified_files),
        staging_directory,
    )
    _reverify_benchmark_v2_case_inputs(verification)
  except Exception:
    staging_directory.cleanup()
    raise
  return (
      assembly_data,
      tuple(staged_specs),
      verification,
  )


def _release_verified_benchmark_v2_case_inputs(
    verification: _VerifiedBenchmarkV2CaseInputs,
) -> None:
  staging = verification.staging_directory
  if staging is not None:
    staging.cleanup()


@contextmanager
def _open_verified_benchmark_v2_case_inputs(
    case: Mapping[str, Any],
    *,
    dataset_roots: Sequence[Path],
    file_index: Mapping[str, list[Path]],
):
  verified = _load_verified_benchmark_v2_case_inputs(
      case,
      dataset_roots=dataset_roots,
      file_index=file_index,
  )
  try:
    yield verified
  finally:
    _release_verified_benchmark_v2_case_inputs(verified[2])


def _main_legacy(args: argparse.Namespace) -> None:
  dataset_roots = _parse_dataset_roots(
      dataset_root=str(args.dataset_root),
      dataset_roots=list(args.dataset_roots or []),
  )
  if not dataset_roots:
    raise SystemExit("No dataset roots provided.")
  cases = _load_cases(Path(args.cases_json))
  if int(args.max_cases) > 0:
    cases = cases[: int(args.max_cases)]
  file_index = _index_step_files(dataset_roots)
  output_path = Path(args.output_jsonl)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  rows_written = 0
  positive_count = 0
  negative_count = 0
  skipped_cases = 0
  skipped_parts = 0
  case_summaries: list[dict[str, Any]] = []
  with output_path.open("w", encoding="utf-8") as f:
    for case in cases:
      case_id = str(case.get("id") or "")
      assembly_json = _assembly_json_for_case(case, dataset_roots)
      if assembly_json is None or not assembly_json.exists():
        skipped_cases += 1
        continue
      parts = case.get("parts")
      if not isinstance(parts, list) or not parts:
        skipped_cases += 1
        continue
      part_names = case.get("selected_part_names")
      if not isinstance(part_names, list):
        part_names = []
      body_uuids = case.get("selected_body_uuids")
      if not isinstance(body_uuids, list):
        body_uuids = []
      assembly_dir = case.get("assembly_dir")
      if not isinstance(assembly_dir, str):
        assembly_dir = None
      dataset_root_hint = case.get("dataset_root_hint")
      if not isinstance(dataset_root_hint, str):
        dataset_root_hint = None

      case_rows = 0
      case_pos = 0
      for idx, token in enumerate(parts):
        try:
          step_path = _resolve_part_path(
              part_token=str(token),
              dataset_roots=dataset_roots,
              assembly_dir=assembly_dir,
              dataset_root_hint=dataset_root_hint,
              file_index=file_index,
          )
          part_name = (
              str(part_names[idx]).strip()
              if idx < len(part_names) and str(part_names[idx]).strip()
              else step_path.stem
          )
          body_uuid = (
              str(body_uuids[idx]).strip()
              if idx < len(body_uuids) and str(body_uuids[idx]).strip()
              else step_path.stem
          )
          candidates = extract_candidate_interfaces_from_step(
              step_path,
              part_name=part_name,
              body_uuid=body_uuid,
              max_candidates=int(args.max_candidates_per_part),
          )
          candidates = label_interfaces_from_assembly_contacts(
              candidates,
              assembly_json,
          )
        except Exception:
          skipped_parts += 1
          continue
        for candidate in candidates:
          payload = candidate.to_dict()
          payload["case_id"] = case_id
          payload["assembly_dir"] = assembly_dir
          f.write(json.dumps(payload, ensure_ascii=False) + "\n")
          rows_written += 1
          case_rows += 1
          if candidate.label == 1:
            positive_count += 1
            case_pos += 1
          else:
            negative_count += 1
      case_summaries.append(
          {
              "case_id": case_id,
              "assembly_dir": assembly_dir,
              "rows": case_rows,
              "positives": case_pos,
          }
      )

  summary = {
      "cases_seen": len(cases),
      "cases_with_rows": len([item for item in case_summaries if item["rows"] > 0]),
      "skipped_cases": skipped_cases,
      "skipped_parts": skipped_parts,
      "rows_written": rows_written,
      "positive_count": positive_count,
      "negative_count": negative_count,
      "positive_rate": round(positive_count / max(1, rows_written), 4),
      "output_jsonl": str(output_path.resolve()),
  }
  summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
  summary_path.write_text(
      json.dumps({"summary": summary, "cases": case_summaries}, indent=2),
      encoding="utf-8",
  )
  print(json.dumps(summary, indent=2))


def _case_part_specs(
    case: Mapping[str, Any],
    *,
    dataset_roots: list[Path],
    file_index: dict[str, Path],
) -> tuple[BenchmarkV2PartSpec, ...]:
  parts = case.get("parts")
  names = case.get("selected_part_names")
  bodies = case.get("selected_body_uuids")
  assets = case.get("selected_geometry_assets")
  if not isinstance(parts, list) or not parts:
    raise ValueError("benchmark_v2 case requires a nonempty parts list")
  if not isinstance(names, list) or len(names) != len(parts):
    raise ValueError("benchmark_v2 case requires one selected_part_name per part")
  if not isinstance(bodies, list) or len(bodies) != len(parts):
    raise ValueError("benchmark_v2 case requires one selected_body_uuid per part")
  if not isinstance(assets, list) or len(assets) != len(parts):
    raise ValueError("benchmark_v2 case requires one geometry asset per part")
  body_values = [str(value).strip() for value in bodies]
  if not all(body_values):
    raise ValueError("benchmark_v2 selected_body_uuids must be nonempty")
  part_aliases = [str(value).strip() for value in names]
  if (
      not all(part_aliases)
      or len(set(part_aliases)) != len(part_aliases)
  ):
    raise ValueError("benchmark_v2 part instance aliases must be nonempty and unique")
  geometry_assets = [str(value).strip() for value in assets]
  if not all(geometry_assets):
    raise ValueError("benchmark_v2 geometry assets must be nonempty")
  assembly_dir = (
      str(case.get("assembly_dir"))
      if isinstance(case.get("assembly_dir"), str)
      else None
  )
  dataset_root_hint = (
      str(case.get("dataset_root_hint"))
      if isinstance(case.get("dataset_root_hint"), str)
      else None
  )
  specs: list[BenchmarkV2PartSpec] = []
  for index, token in enumerate(parts):
    step_path = _resolve_part_path(
        part_token=str(token),
        dataset_roots=dataset_roots,
        assembly_dir=assembly_dir,
        dataset_root_hint=dataset_root_hint,
        file_index=file_index,
    )
    specs.append(
        BenchmarkV2PartSpec(
            part_key=part_aliases[index],
            part_name=part_aliases[index],
            body_uuid=body_values[index],
            step_path=step_path,
            geometry_asset=geometry_assets[index],
        )
    )
  return tuple(specs)


def _candidate_face_indices(candidate: CandidateInterface) -> frozenset[int]:
  values: set[int] = set()
  if candidate.face_index is not None:
    values.add(int(candidate.face_index))
  members = candidate.metadata.get("member_face_indices")
  if isinstance(members, list):
    for value in members:
      if isinstance(value, int):
        values.add(int(value))
  return frozenset(values)


def _mapped_fusion_contact_endpoint(
    prepared: PreparedBenchmarkV2Assembly,
    endpoint: Mapping[str, Any],
    *,
    case_id: str,
    source_identity_by_part: Mapping[str, tuple[str, str]],
    fusion_step_face_lookup: Mapping[
        tuple[str, str, int], Mapping[str, Any]
    ],
) -> PrivateMappedFusionEndpoint:
  """Resolve one Fusion lookup key to verified live STEP/OCC face identities."""

  clean_case_id = str(case_id).strip()
  if not clean_case_id:
    raise ValueError("benchmark_v2 contact mapping requires a nonempty case id")
  raw_part = str(endpoint.get("part") or "").strip()
  geometry_asset = str(endpoint.get("geometry_asset") or "").strip()
  if not raw_part or raw_part not in source_identity_by_part:
    raise ValueError("Fusion gold endpoint is missing an assembly instance alias")
  expected_asset, raw_body = source_identity_by_part[raw_part]
  if geometry_asset != expected_asset:
    raise ValueError("Fusion gold endpoint geometry asset differs from its instance")
  fusion_index = endpoint.get("fusion_face_index")
  if not isinstance(fusion_index, int) or isinstance(fusion_index, bool):
    raise ValueError("Fusion gold endpoint is missing an integer lookup index")
  if fusion_index < 0:
    raise ValueError("Fusion gold endpoint has a negative lookup index")
  if prepared.private_source_part_for(raw_part) != raw_part:
    raise ValueError("prepared part identity differs from private source instance")
  key = (clean_case_id, raw_part, fusion_index)
  mapping = fusion_step_face_lookup.get(key)
  if not isinstance(mapping, Mapping):
    raise ValueError(
        "Fusion gold endpoint lacks one unambiguous STEP/OCC face mapping: "
        f"case={clean_case_id}, part={raw_part}, fusion_face_index={fusion_index}"
    )
  # v3 authority receipts store case_id on the enclosing case row; the
  # authenticated lookup key already carries it.  Older flat receipts repeat
  # case_id in every endpoint row.  If present it must still agree.
  mapped_case_id = mapping.get("case_id", clean_case_id)
  if (
      mapping.get("status") != "mapped_unique"
      or mapping.get("mapping_mode") != "authoritative_obj_face_group"
      or str(mapped_case_id or "") != clean_case_id
      or str(mapping.get("part") or "") != raw_part
      or str(mapping.get("geometry_asset") or "") != geometry_asset
      or str(mapping.get("body_uuid") or "") != raw_body
      or mapping.get("fusion_face_index") != fusion_index
  ):
    raise ValueError("Fusion STEP/OCC endpoint mapping identity is inconsistent")
  raw_occ_indices = mapping.get("raw_occ_face_indices")
  signatures = mapping.get("source_face_signature_sha256s")
  if (
      not isinstance(raw_occ_indices, list)
      or not isinstance(signatures, list)
      or len(signatures) != len(raw_occ_indices)
      or any(
          not isinstance(signature, str)
          or len(signature) != 64
          or any(character not in "0123456789abcdef" for character in signature)
          for signature in signatures
      )
  ):
    raise ValueError("Fusion STEP/OCC endpoint mapping lacks raw OCC faces")
  randomized = prepared.randomized_face_indices_for_mapped_occ_faces(
      raw_part,
      raw_occ_indices,
      signatures,
  )
  return PrivateMappedFusionEndpoint(
      raw_part=raw_part,
      geometry_asset=geometry_asset,
      raw_body=raw_body,
      opaque_body=prepared.private_identity_context.opaque_body_for(raw_part),
      raw_occ_face_indices=tuple(raw_occ_indices),
      source_face_signature_sha256s=tuple(signatures),
      randomized_face_indices=randomized,
  )


def _remapped_contact_pairs(
    prepared: PreparedBenchmarkV2Assembly,
    evaluation_gold_contacts: Mapping[str, Any],
    *,
    case_id: str,
    source_identity_by_part: Mapping[str, tuple[str, str]],
    fusion_step_face_lookup: Mapping[
        tuple[str, str, int], Mapping[str, Any]
    ],
) -> frozenset[tuple[str, frozenset[int], str, frozenset[int]]]:
  """Map Fusion gold keys through a receipt; never execute a Fusion index."""

  clean_case_id = str(case_id).strip()
  if not clean_case_id:
    raise ValueError("benchmark_v2 contact mapping requires a nonempty case id")
  if not isinstance(fusion_step_face_lookup, Mapping):
    raise ValueError("benchmark_v2 contact mapping requires an endpoint lookup")

  if evaluation_gold_contacts.get("schema_version") != (
      "benchmark_v2_evaluation_gold_contacts.v2"
  ):
    raise ValueError("benchmark_v2 requires explicit private contact gold v2")
  contacts = evaluation_gold_contacts.get("contacts")
  result: set[tuple[str, frozenset[int], str, frozenset[int]]] = set()
  if not isinstance(contacts, list):
    return frozenset()
  for contact in contacts:
    if not isinstance(contact, Mapping):
      continue
    first = contact.get("endpoint_a")
    second = contact.get("endpoint_b")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
      continue
    raw_part_a = str(first.get("part") or "").strip()
    raw_part_b = str(second.get("part") or "").strip()
    if raw_part_a == raw_part_b:
      continue
    if (
        first.get("entity_type") != "BRepFace"
        or second.get("entity_type") != "BRepFace"
    ):
      # Only face/face contacts are part of the private gold sidecar.
      continue
    mapped_a = _mapped_fusion_contact_endpoint(
        prepared,
        first,
        case_id=clean_case_id,
        source_identity_by_part=source_identity_by_part,
        fusion_step_face_lookup=fusion_step_face_lookup,
    )
    mapped_b = _mapped_fusion_contact_endpoint(
        prepared,
        second,
        case_id=clean_case_id,
        source_identity_by_part=source_identity_by_part,
        fusion_step_face_lookup=fusion_step_face_lookup,
    )
    group_a = frozenset(mapped_a.randomized_face_indices)
    group_b = frozenset(mapped_b.randomized_face_indices)
    result.add((mapped_a.opaque_body, group_a, mapped_b.opaque_body, group_b))
    result.add((mapped_b.opaque_body, group_b, mapped_a.opaque_body, group_a))
  return frozenset(result)


def _flatten_contact_face_group_pairs(
    group_pairs: frozenset[
        tuple[str, frozenset[int], str, frozenset[int]]
    ],
) -> frozenset[tuple[str, int, str, int]]:
  return frozenset(
      (body_a, index_a, body_b, index_b)
      for body_a, group_a, body_b, group_b in group_pairs
      for index_a in group_a
      for index_b in group_b
  )


def _candidate_covers_contact_face_group(
    *,
    body: str,
    candidate_face_indices: frozenset[int],
    group_pairs: frozenset[
        tuple[str, frozenset[int], str, frozenset[int]]
    ],
) -> bool:
  """Require complete coverage of at least one Fusion-face OCC group."""

  return any(
      contact_body == body and required_faces <= candidate_face_indices
      for contact_body, required_faces, _other_body, _other_faces in group_pairs
  )


def produce_benchmark_v2_interface_case(
    *,
    part_specs: Sequence[BenchmarkV2PartSpec],
    evaluation_gold_contacts: Mapping[str, Any],
    case_id: str,
    seed: int,
    source_split: str,
    family_split_manifest_sha256: str,
    max_candidates_per_part: int = 0,
    translation_box_fraction: float = 1.0,
    interface_scorer: Any = None,
    shape_loader: Callable[[Path], Any] | None = None,
    fusion_step_face_lookup: Mapping[
        tuple[str, str, int], Mapping[str, Any]
    ] | None = None,
    fusion_step_face_map_binding: Mapping[str, Any] | None = None,
    private_supervision_binding: Mapping[str, Any] | None = None,
    allow_source_positive_only: bool = False,
) -> BenchmarkV2ProducedInterfaceCase:
  """Prepare one case from explicit instance gold; never infer label absence.

  ``allow_source_positive_only`` is a development/audit seam. Under the v2
  source-positive contract it emits matched positives and excludes every
  unlisted candidate as unknown. Formal binary writers reject that policy.
  """

  split = str(source_split).strip().lower()
  if split not in {"train", "dev"}:
    raise ValueError("benchmark_v2 producer source_split must be train or dev")
  if int(max_candidates_per_part) < 0:
    raise ValueError("max_candidates_per_part must be nonnegative")
  if (
      not math.isfinite(float(translation_box_fraction))
      or float(translation_box_fraction) < 0.0
  ):
    raise ValueError("translation_box_fraction must be finite and nonnegative")
  face_map_binding = _validated_face_map_binding(
      fusion_step_face_map_binding,
      require_private_receipt=False,
  )
  if fusion_step_face_lookup is None:
    raise ValueError(
        "benchmark_v2 Fusion contact labels require an explicit STEP face map"
    )
  supervision_binding = validate_private_supervision_binding(
      private_supervision_binding,
      require_certified_binary=not bool(allow_source_positive_only),
  )
  if evaluation_gold_contacts.get("schema_version") == (
      "benchmark_v2_evaluation_gold_contacts.v2"
  ):
    if not allow_source_positive_only:
      require_certified_binary_training_gold(evaluation_gold_contacts)
    label_policy = "source_positive_only_unknown_excluded"
  else:
    require_certified_binary_training_gold(evaluation_gold_contacts)
    label_policy = "closed_world_certified_binary"
  config = {
      "max_candidates_per_part": int(max_candidates_per_part),
      "translation_box_fraction": float(translation_box_fraction),
      "fusion_step_face_map_binding": face_map_binding,
      "private_supervision_binding": supervision_binding,
      "label_policy": label_policy,
  }
  config_sha = canonical_sha256(config)
  feature_sha = canonical_sha256(list(BENCHMARK_V2_FEATURE_NAMES))
  token = case_token_sha256(
      family_manifest_sha256=family_split_manifest_sha256,
      case_id=case_id,
  )
  prepared = prepare_benchmark_v2_assembly(
      part_specs,
      assembly_nonce=str(case_id),
      seed=int(seed),
      translation_box_fraction=float(translation_box_fraction),
      max_candidates_per_part=int(max_candidates_per_part),
      interface_scorer=interface_scorer,
      shape_loader=shape_loader,
  )
  source_identity_by_part = {
      str(spec.part_key): (
          str(spec.geometry_asset or ""),
          str(spec.body_uuid),
      )
      for spec in part_specs
  }
  if (
      len(source_identity_by_part) != len(part_specs)
      or any(not asset or not body for asset, body in source_identity_by_part.values())
  ):
    raise ValueError("benchmark_v2 part specs lack private instance bindings")
  contact_group_pairs = _remapped_contact_pairs(
      prepared,
      evaluation_gold_contacts,
      case_id=case_id,
      source_identity_by_part=source_identity_by_part,
      fusion_step_face_lookup=fusion_step_face_lookup,
  )
  rows: list[dict[str, Any]] = []
  records_by_body: dict[str, list[BenchmarkV2InterfaceRecord]] = {}
  for opaque_part in prepared.model_part_names:
    part = prepared.parts[opaque_part]
    for candidate, model_view in zip(part.candidates, part.model_views):
      face_indices = _candidate_face_indices(candidate)
      label = int(
          _candidate_covers_contact_face_group(
              body=candidate.body_uuid,
              candidate_face_indices=face_indices,
              group_pairs=contact_group_pairs,
          )
      )
      if label == 0 and label_policy == "source_positive_only_unknown_excluded":
        continue
      row = {
          "schema_version": INTERFACE_TRAINING_ROW_SCHEMA,
          "model_input_protocol": "benchmark_v2",
          "protocol_version": PROTOCOL_VERSION,
          "source_split": split,
          "family_split_manifest_sha256": family_split_manifest_sha256,
          "feature_names_sha256": feature_sha,
          "producer_config_sha256": config_sha,
          "case_token_sha256": token,
          "opaque_part": opaque_part,
          "opaque_interface": candidate.interface_id,
          "benchmark_v2_model_view": model_view.model_view,
          "benchmark_v2_model_view_sha256": model_view.sha256,
          "label": label,
      }
      if set(row) != INTERFACE_TRAINING_ROW_KEYS:
        raise AssertionError("internal benchmark_v2 interface row schema drift")
      rows.append(row)
      records_by_body.setdefault(candidate.body_uuid, []).append(
          BenchmarkV2InterfaceRecord(
              opaque_part=opaque_part,
              opaque_body=candidate.body_uuid,
              candidate=candidate,
              model_view=model_view,
              row=row,
          )
      )
  rows.sort(key=lambda row: (str(row["opaque_part"]), str(row["opaque_interface"])))
  return BenchmarkV2ProducedInterfaceCase(
      rows=tuple(rows),
      records_by_body={
          key: tuple(sorted(values, key=lambda item: item.candidate.interface_id))
          for key, values in sorted(records_by_body.items())
      },
      contact_face_group_pairs=contact_group_pairs,
      prepared=prepared,
      seed=int(seed),
      producer_config=dict(config),
  )


def write_benchmark_v2_interface_dataset(
    output_path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    source_split: str,
    family_split_manifest: str | Path,
    seed: int,
    producer_config: Mapping[str, Any],
    authenticated_receipts: BenchmarkV2AuthenticatedTrainingReceipts | None = None,
) -> dict[str, Any]:
  if not isinstance(
      authenticated_receipts,
      BenchmarkV2AuthenticatedTrainingReceipts,
  ):
    raise ValueError(
        "formal benchmark_v2 interface writing requires a path-authenticated "
        "receipt set; receipt Mappings are provenance only"
    )
  _validated_face_map_binding(
      (
          producer_config.get("fusion_step_face_map_binding")
          if isinstance(producer_config, Mapping)
          else None
      ),
      require_private_receipt=True,
  )
  validate_private_supervision_binding(
      (
          producer_config.get("private_supervision_binding")
          if isinstance(producer_config, Mapping)
          else None
      ),
      require_certified_binary=True,
  )
  path = Path(output_path)
  return write_dataset_artifact(
      path,
      rows=rows,
      row_schema=INTERFACE_TRAINING_ROW_SCHEMA,
      feature_names=BENCHMARK_V2_FEATURE_NAMES,
      source_split=source_split,
      family_split_manifest=family_split_manifest,
      seed=seed,
      producer_config=producer_config,
      producer_kind="interface",
      authenticated_receipts=authenticated_receipts,
  )


def _main_benchmark_v2(args: argparse.Namespace) -> None:
  if not args.family_split_manifest:
    raise SystemExit("benchmark_v2 requires --family_split_manifest")
  if not getattr(args, "private_source_split", None):
    raise SystemExit("benchmark_v2 requires --private-source-split")
  if not getattr(args, "private_evaluation_gold_split", None):
    raise SystemExit("benchmark_v2 requires --private-evaluation-gold-split")
  if not args.fusion_step_face_map_receipt:
    raise SystemExit("benchmark_v2 requires --fusion-step-face-map-receipt")
  if int(args.max_cases) > 0:
    raise ValueError(
        "formal benchmark_v2 production must consume the complete frozen split; "
        "--max_cases is legacy smoke-test only"
    )
  if int(args.max_candidates_per_part) > 0:
    raise ValueError(
        "formal benchmark_v2 interface labels require the complete candidate set; "
        "--max_candidates_per_part must be 0"
    )
  dataset_roots = _parse_dataset_roots(
      dataset_root=str(args.dataset_root),
      dataset_roots=list(args.dataset_roots or []),
  )
  authenticated_inputs = load_benchmark_v2_authenticated_training_inputs(
      public_cases_path=Path(args.cases_json),
      private_source_path=Path(args.private_source_split),
      private_gold_path=Path(args.private_evaluation_gold_split),
      family_split_manifest_path=Path(args.family_split_manifest),
      face_map_receipt_path=Path(args.fusion_step_face_map_receipt),
      source_split=str(args.source_split),
  )
  cases = [dict(case) for case in authenticated_inputs.public_cases]
  case_ids = list(authenticated_inputs.case_ids)
  family_sha = authenticated_inputs.receipts.family_split_manifest_sha256
  private_supervision = authenticated_inputs.private_supervision
  face_map_lookup = authenticated_inputs.face_map_lookup
  face_map_binding = authenticated_inputs.face_map_binding
  for case_id in case_ids:
    require_certified_binary_training_gold(
        private_supervision.gold_case(case_id)
    )
  file_index = _index_step_files(dataset_roots)
  config = {
      "max_candidates_per_part": int(args.max_candidates_per_part),
      "translation_box_fraction": float(
          args.benchmark_v2_translation_box_fraction
      ),
      "fusion_step_face_map_binding": face_map_binding,
      "private_supervision_binding": dict(
          private_supervision.producer_binding
      ),
      "label_policy": "closed_world_certified_binary",
  }
  rows: list[dict[str, Any]] = []
  for case in cases:
    case_id = str(case.get("id") or "")
    private_source_case = private_supervision.source_case(case_id)
    private_gold_case = private_supervision.gold_case(case_id)
    with _open_verified_benchmark_v2_case_inputs(
        private_source_case,
        dataset_roots=dataset_roots,
        file_index=file_index,
    ) as (_assembly_data, specs, live_inputs):
      produced = produce_benchmark_v2_interface_case(
          part_specs=specs,
          evaluation_gold_contacts=private_gold_case,
          case_id=case_id,
          seed=int(args.frame_randomization_seed),
          source_split=str(args.source_split),
          family_split_manifest_sha256=family_sha,
          max_candidates_per_part=int(args.max_candidates_per_part),
          translation_box_fraction=float(
              args.benchmark_v2_translation_box_fraction
          ),
          fusion_step_face_lookup=face_map_lookup,
          fusion_step_face_map_binding=face_map_binding,
          private_supervision_binding=private_supervision.producer_binding,
      )
      _reverify_benchmark_v2_case_inputs(live_inputs)
      if not produced.rows or not any(
          int(row["label"]) == 1 for row in produced.rows
      ):
        raise ValueError(
            f"benchmark_v2 case {case.get('id')!r} has no contact-grounded interface rows"
        )
      rows.extend(produced.rows)
  manifest = write_benchmark_v2_interface_dataset(
      Path(args.output_jsonl),
      rows=rows,
      source_split=str(args.source_split),
      family_split_manifest=Path(args.family_split_manifest),
      seed=int(args.frame_randomization_seed),
      producer_config=config,
      authenticated_receipts=authenticated_inputs.receipts,
  )
  summary = {
      "model_input_protocol": "benchmark_v2",
      "protocol_version": PROTOCOL_VERSION,
      "source_split": str(args.source_split),
      "cases_seen": len(cases),
      "rows_written": len(rows),
      "positive_count": sum(int(row["label"] == 1) for row in rows),
      "negative_count": sum(int(row["label"] == 0) for row in rows),
      "dataset_manifest_sha256": manifest["manifest_sha256"],
  }
  summary_path = (
      Path(args.summary_json)
      if args.summary_json
      else Path(args.output_jsonl).with_suffix(".summary.json")
  )
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


def _assembly_json_for_case(
    case: dict[str, Any],
    dataset_roots: list[Path],
) -> Optional[Path]:
  assembly_dir = case.get("assembly_dir")
  if not isinstance(assembly_dir, str) or not assembly_dir.strip():
    return None
  dataset_root_hint = case.get("dataset_root_hint")
  if isinstance(dataset_root_hint, str) and dataset_root_hint.strip():
    candidate = Path(dataset_root_hint) / assembly_dir / "assembly.json"
    if candidate.exists():
      return candidate
  for root in dataset_roots:
    candidate = root / assembly_dir / "assembly.json"
    if candidate.exists():
      return candidate
  return None


if __name__ == "__main__":
  main()
