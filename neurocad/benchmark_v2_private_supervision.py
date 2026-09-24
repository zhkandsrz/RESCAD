"""Strict private source/gold boundary for benchmark-v2 training consumers.

Public family-split cases intentionally contain no raw paths, body UUIDs, or
Fusion labels. Formal producers must join those cases to the two custodian
sidecars explicitly and verify the commitments recorded in the frozen split
manifest before touching any source file.

The current ``evaluation_gold_contacts.v2`` contract contains source-positive
assertions only. It does not certify absence, so this module never converts an
unlisted candidate into a negative training label.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .benchmark_v2_training_provenance import (
    CapturedJsonArtifact,
    canonical_sha256,
    capture_json_artifact,
    require_formal_binary_training_available,
)


PRIVATE_SOURCE_SPLIT_SCHEMA = "benchmark_v2_private_source_split.v1"
PRIVATE_GOLD_SPLIT_SCHEMA = "benchmark_v2_private_evaluation_gold_split.v1"
PRIVATE_GOLD_CONTACT_SCHEMA = "benchmark_v2_evaluation_gold_contacts.v2"
PRIVATE_SUPERVISION_BINDING_SCHEMA = (
    "benchmark_v2_private_supervision_training_binding.v1"
)

_PART_RE = re.compile(r"part_[0-9]{3,}")
_GEOMETRY_RE = re.compile(r"geometry_[0-9]{3,}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTACT_RE = re.compile(r"gold_contact_[0-9a-f]{24}")


def _exact_mapping(
    value: Any,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != keys:
    raise ValueError(f"{label} has an invalid fixed field set")
  return value


def _nonempty_string(value: Any, *, label: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{label} must be a nonempty string")
  return value


def _digest(value: Any, *, label: str, sha1: bool = False) -> str:
  pattern = _SHA1_RE if sha1 else _SHA256_RE
  if not isinstance(value, str) or pattern.fullmatch(value) is None:
    raise ValueError(f"{label} is not a lowercase digest")
  return value


def _positive_bytes(value: Any, *, label: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f"{label} must be a positive byte count")
  return value


def _safe_relative_path(value: Any, *, label: str, suffix: str | None = None) -> str:
  text = _nonempty_string(value, label=label)
  if "\\" in text:
    raise ValueError(f"{label} must use a safe POSIX-relative path")
  path = PurePosixPath(text)
  if (
      path.is_absolute()
      or any(part in {"", ".", ".."} for part in path.parts)
      or path.as_posix() != text
      or (suffix is not None and path.suffix.lower() != suffix)
  ):
    raise ValueError(f"{label} must use a safe POSIX-relative path")
  return text


def _validate_source_instance_key(value: Any) -> dict[str, str]:
  if not isinstance(value, Mapping):
    raise ValueError("private source instance key is malformed")
  kind = value.get("kind")
  expected = (
      {"kind", "body_uuid", "occurrence_uuid"}
      if kind == "occurrence"
      else {"kind", "body_uuid", "root_component_uuid"}
      if kind == "root"
      else set()
  )
  if not expected or set(value) != expected:
    raise ValueError("private source instance key is malformed")
  result = {key: _nonempty_string(value[key], label=key) for key in expected}
  result["kind"] = str(kind)
  return result


def _validate_private_source_payload(payload: Any, *, split: str) -> dict[str, dict[str, Any]]:
  top = _exact_mapping(
      payload,
      {"schema_version", "visibility", "split", "case_count", "cases"},
      label="private source split",
  )
  if (
      top.get("schema_version") != PRIVATE_SOURCE_SPLIT_SCHEMA
      or top.get("visibility") != "private_custodian_only"
      or top.get("split") != split
  ):
    raise ValueError("private source split contract is invalid")
  rows = top.get("cases")
  if not isinstance(rows, list) or not rows or top.get("case_count") != len(rows):
    raise ValueError("private source split case_count is invalid")

  indexed: dict[str, dict[str, Any]] = {}
  seen_instances: set[str] = set()
  for raw_row in rows:
    row = _exact_mapping(
        raw_row,
        {
            "case_id",
            "assembly_dir",
            "parts",
            "selected_part_names",
            "selected_body_uuids",
            "selected_geometry_assets",
            "source_receipt_binding",
        },
        label="private source case",
    )
    case_id = _nonempty_string(row.get("case_id"), label="private source case_id")
    if case_id in indexed:
      raise ValueError("private source case IDs are duplicated")
    _safe_relative_path(row.get("assembly_dir"), label="assembly_dir")
    arrays = [
        row.get("parts"),
        row.get("selected_part_names"),
        row.get("selected_body_uuids"),
        row.get("selected_geometry_assets"),
    ]
    if not all(isinstance(value, list) for value in arrays):
      raise ValueError("private source instance arrays are malformed")
    parts, names, bodies, assets = arrays
    count = len(names)
    if not 2 <= count <= 12 or any(len(value) != count for value in arrays):
      raise ValueError("private source instance arrays disagree")
    if (
        any(not isinstance(name, str) or _PART_RE.fullmatch(name) is None for name in names)
        or len(set(names)) != count
        or any(
            not isinstance(asset, str) or _GEOMETRY_RE.fullmatch(asset) is None
            for asset in assets
        )
        or any(not isinstance(body, str) or not body.strip() for body in bodies)
    ):
      raise ValueError("private source instance aliases are malformed")
    for value in parts:
      _safe_relative_path(value, label="private source STEP", suffix=".step")

    receipt = _exact_mapping(
        row.get("source_receipt_binding"),
        {"archive", "receipt_sha256", "assembly_json", "body_steps", "instances"},
        label="private source receipt",
    )
    archive = _nonempty_string(receipt.get("archive"), label="private source archive")
    if re.fullmatch(r"a1\.0\.0_[0-9]{2}\.7z", archive) is None:
      raise ValueError("private source archive identity is malformed")
    _digest(receipt.get("receipt_sha256"), label="private source receipt_sha256")
    assembly = _exact_mapping(
        receipt.get("assembly_json"),
        {"path", "bytes", "sha256"},
        label="private source assembly receipt",
    )
    _safe_relative_path(assembly.get("path"), label="assembly JSON", suffix=".json")
    _positive_bytes(assembly.get("bytes"), label="assembly JSON bytes")
    _digest(assembly.get("sha256"), label="assembly JSON sha256")

    step_rows = receipt.get("body_steps")
    instance_rows = receipt.get("instances")
    if (
        not isinstance(step_rows, list)
        or not isinstance(instance_rows, list)
        or len(step_rows) != count
        or len(instance_rows) != count
    ):
      raise ValueError("private source receipt does not cover every instance")
    steps_by_part: dict[str, Mapping[str, Any]] = {}
    for raw_step in step_rows:
      step = _exact_mapping(
          raw_step,
          {"part", "geometry_asset", "body_uuid", "path", "bytes", "sha1", "sha256"},
          label="private source STEP receipt",
      )
      part = _nonempty_string(step.get("part"), label="STEP receipt part")
      if part in steps_by_part:
        raise ValueError("private source STEP receipt part is duplicated")
      _safe_relative_path(step.get("path"), label="STEP receipt path", suffix=".step")
      _positive_bytes(step.get("bytes"), label="STEP receipt bytes")
      _digest(step.get("sha1"), label="STEP receipt sha1", sha1=True)
      _digest(step.get("sha256"), label="STEP receipt sha256")
      steps_by_part[part] = step

    instances_by_part: dict[str, Mapping[str, Any]] = {}
    for raw_instance in instance_rows:
      instance = _exact_mapping(
          raw_instance,
          {
              "part",
              "geometry_asset",
              "source_instance_key",
              "occurrence_path",
              "is_visible",
              "is_grounded",
          },
          label="private source instance receipt",
      )
      part = _nonempty_string(instance.get("part"), label="instance part")
      if part in instances_by_part:
        raise ValueError("private source instance part is duplicated")
      source_key = _validate_source_instance_key(instance.get("source_instance_key"))
      canonical_instance = json.dumps(source_key, sort_keys=True, separators=(",", ":"))
      if canonical_instance in seen_instances:
        raise ValueError("private source assembly instance identity is duplicated")
      seen_instances.add(canonical_instance)
      occurrence_path = instance.get("occurrence_path")
      if not isinstance(occurrence_path, list) or any(
          not isinstance(value, str) or not value for value in occurrence_path
      ):
        raise ValueError("private source occurrence_path is malformed")
      if (
          (source_key["kind"] == "root" and occurrence_path)
          or (
              source_key["kind"] == "occurrence"
              and (
                  not occurrence_path
                  or occurrence_path[-1] != source_key["occurrence_uuid"]
              )
          )
          or not isinstance(instance.get("is_visible"), bool)
          or not isinstance(instance.get("is_grounded"), bool)
      ):
        raise ValueError("private source instance placement is inconsistent")
      instances_by_part[part] = instance

    if set(steps_by_part) != set(names) or set(instances_by_part) != set(names):
      raise ValueError("private source receipt part coverage is incomplete")
    for index, part in enumerate(names):
      step = steps_by_part[part]
      instance = instances_by_part[part]
      source_key = _validate_source_instance_key(instance["source_instance_key"])
      if (
          step.get("geometry_asset") != assets[index]
          or instance.get("geometry_asset") != assets[index]
          or step.get("body_uuid") != bodies[index]
          or source_key.get("body_uuid") != bodies[index]
          or PurePosixPath(str(step.get("path"))).name
          != PurePosixPath(str(parts[index])).name
      ):
        raise ValueError("private source per-instance binding is inconsistent")
    indexed[case_id] = copy.deepcopy(dict(row))
  return indexed


def _validate_gold_endpoint(
    value: Any,
    *,
    source_identity_by_part: Mapping[str, tuple[str, str]],
) -> tuple[str, str, int, str, str]:
  endpoint = _exact_mapping(
      value,
      {"part", "geometry_asset", "fusion_face_index", "entity_type", "surface_type"},
      label="private gold endpoint",
  )
  part = endpoint.get("part")
  asset = endpoint.get("geometry_asset")
  if (
      not isinstance(part, str)
      or _PART_RE.fullmatch(part) is None
      or source_identity_by_part.get(part, (None, None))[0] != asset
  ):
    raise ValueError("private gold endpoint is not bound to a source instance")
  face_index = endpoint.get("fusion_face_index")
  if isinstance(face_index, bool) or not isinstance(face_index, int) or face_index < 0:
    raise ValueError("private gold endpoint fusion_face_index is invalid")
  if endpoint.get("entity_type") != "BRepFace" or not isinstance(
      endpoint.get("surface_type"), str
  ) or not str(endpoint.get("surface_type")).strip():
    raise ValueError("private gold endpoint semantics are invalid")
  return (
      str(part),
      str(asset),
      int(face_index),
      str(endpoint["entity_type"]),
      str(endpoint["surface_type"]),
  )


def _validate_source_statistics(
    value: Any,
    *,
    contact_count: int,
) -> int:
  statistics = _exact_mapping(
      value,
      {
          "audit_performed",
          "source_contact_count",
          "gold_contact_count",
          "excluded_contact_count",
          "exclusions",
      },
      label="private evaluation gold source_statistics",
  )
  source_count = statistics.get("source_contact_count")
  gold_count = statistics.get("gold_contact_count")
  excluded_count = statistics.get("excluded_contact_count")
  counts = (source_count, gold_count, excluded_count)
  exclusions = statistics.get("exclusions")
  if (
      statistics.get("audit_performed") is not True
      or any(isinstance(count, bool) or not isinstance(count, int) for count in counts)
      or int(source_count) < 1
      or int(gold_count) < 1
      or int(excluded_count) < 0
      or not isinstance(exclusions, Mapping)
      or any(
          not isinstance(reason, str)
          or not reason
          or isinstance(count, bool)
          or not isinstance(count, int)
          or count < 1
          for reason, count in (
              exclusions.items() if isinstance(exclusions, Mapping) else ()
          )
      )
  ):
    raise ValueError("private evaluation gold source_statistics are invalid")
  if (
      int(gold_count) != contact_count
      or int(source_count) != int(gold_count) + int(excluded_count)
      or sum(exclusions.values()) != int(excluded_count)
  ):
    raise ValueError("private evaluation gold source_statistics accounting does not close")
  return int(source_count)


def _validate_private_gold_payload(
    payload: Any,
    *,
    split: str,
    source_cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
  top = _exact_mapping(
      payload,
      {"schema_version", "visibility", "split", "case_count", "cases"},
      label="private evaluation gold split",
  )
  if (
      top.get("schema_version") != PRIVATE_GOLD_SPLIT_SCHEMA
      or top.get("visibility") != "private_evaluator_only"
      or top.get("split") != split
  ):
    raise ValueError("private evaluation gold split contract is invalid")
  rows = top.get("cases")
  if not isinstance(rows, list) or not rows or top.get("case_count") != len(rows):
    raise ValueError("private evaluation gold case_count is invalid")

  indexed: dict[str, dict[str, Any]] = {}
  contact_ids: set[str] = set()
  for raw_row in rows:
    row = _exact_mapping(
        raw_row,
        {"case_id", "evaluation_gold_contacts"},
        label="private evaluation gold case",
    )
    case_id = _nonempty_string(row.get("case_id"), label="private gold case_id")
    source_case = source_cases.get(case_id)
    if source_case is None or case_id in indexed:
      raise ValueError("private evaluation gold case set is invalid")
    gold = _exact_mapping(
        row.get("evaluation_gold_contacts"),
        {
            "schema_version",
            "visibility",
            "multiplicity_semantics",
            "face_identity_status",
            "gold_interface_certificate_ready",
            "blocking_reason",
            "source_statistics",
            "contacts",
        },
        label="private evaluation contact gold",
    )
    expected = {
        "schema_version": PRIVATE_GOLD_CONTACT_SCHEMA,
        "visibility": "private_evaluation_only",
        "multiplicity_semantics": (
            "one_record_per_fusion_source_contact_instance_pair"
        ),
        "face_identity_status": "fusion_source_index_unverified_step_bijection",
        "gold_interface_certificate_ready": False,
        "blocking_reason": "fusion_to_step_face_identity_bijection_not_verified",
    }
    if any(gold.get(key) != value for key, value in expected.items()):
      raise ValueError("private evaluation contact gold v2 contract is invalid")
    contacts = gold.get("contacts")
    if not isinstance(contacts, list) or not contacts:
      raise ValueError("private evaluation gold contains no source-positive contacts")
    source_contact_count = _validate_source_statistics(
        gold.get("source_statistics"),
        contact_count=len(contacts),
    )
    source_identity_by_part = {
        str(part): (str(asset), str(body))
        for part, asset, body in zip(
            source_case["selected_part_names"],
            source_case["selected_geometry_assets"],
            source_case["selected_body_uuids"],
        )
    }
    seen_ordinals: set[int] = set()
    previous_ordinal = -1
    for raw_contact in contacts:
      contact = _exact_mapping(
          raw_contact,
          {"contact_id", "source_contact_ordinal", "endpoint_a", "endpoint_b"},
          label="private gold contact",
      )
      contact_id = contact.get("contact_id")
      if (
          not isinstance(contact_id, str)
          or _CONTACT_RE.fullmatch(contact_id) is None
          or contact_id in contact_ids
      ):
        raise ValueError("private gold contact_id is invalid or duplicated")
      contact_ids.add(contact_id)
      ordinal = contact.get("source_contact_ordinal")
      if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise ValueError(
            "private gold source_contact_ordinal must be an integer"
        )
      if ordinal < 0 or ordinal >= source_contact_count:
        raise ValueError(
            "private gold source_contact_ordinal is outside source_contact_count"
        )
      if ordinal in seen_ordinals:
        raise ValueError("private gold source_contact_ordinal must be unique per case")
      if ordinal <= previous_ordinal:
        raise ValueError(
            "private gold source_contact_ordinal must be strictly increasing"
        )
      seen_ordinals.add(ordinal)
      previous_ordinal = ordinal
      endpoint_a = _validate_gold_endpoint(
          contact.get("endpoint_a"),
          source_identity_by_part=source_identity_by_part,
      )
      endpoint_b = _validate_gold_endpoint(
          contact.get("endpoint_b"),
          source_identity_by_part=source_identity_by_part,
      )
      if endpoint_a[0] == endpoint_b[0]:
        raise ValueError(
            "private gold contact endpoints reference the same physical instance"
        )
      if endpoint_a[0] > endpoint_b[0]:
        raise ValueError("private gold contact endpoint order is not canonical")
      expected_contact_id = "gold_contact_" + canonical_sha256(
          {
              "case_id": case_id,
              "source_contact_ordinal": ordinal,
              "endpoint_a": dict(contact["endpoint_a"]),
              "endpoint_b": dict(contact["endpoint_b"]),
          }
      )[:24]
      if contact_id != expected_contact_id:
        raise ValueError("private gold contact_id does not replay producer preimage")
    indexed[case_id] = copy.deepcopy(dict(gold))
  if set(indexed) != set(source_cases):
    raise ValueError("private evaluation gold case set differs from private source")
  return indexed


def _manifest_commitment(
    manifest: Mapping[str, Any],
    *,
    section_name: str,
    schema_version: str,
    split: str,
    payload: Any,
) -> str:
  section = manifest.get(section_name)
  if not isinstance(section, Mapping) or (
      section.get("structurally_isolated_from_model_cases") is not True
      or section.get("schema_version") != schema_version
  ):
    raise ValueError(f"family manifest {section_name} binding is invalid")
  splits = section.get("splits")
  commitment = splits.get(split) if isinstance(splits, Mapping) else None
  if not isinstance(commitment, Mapping) or set(commitment) != {"case_count", "sha256"}:
    raise ValueError(f"family manifest {section_name} split binding is invalid")
  expected_count = payload.get("case_count") if isinstance(payload, Mapping) else None
  if commitment.get("case_count") != expected_count:
    raise ValueError(f"family manifest {section_name} case_count differs")
  actual = canonical_sha256(payload)
  expected = str(commitment.get("sha256") or "")
  if expected != "sha256:" + actual:
    label = "private source" if section_name == "private_source_bindings" else "private gold"
    raise ValueError(f"{label} payload does not match the family manifest commitment")
  return actual


@dataclass(frozen=True, slots=True)
class BenchmarkV2PrivateSupervision:
  """Verified in-memory join of one public split and its private sidecars."""

  split: str
  _source_cases: Mapping[str, Mapping[str, Any]]
  _gold_cases: Mapping[str, Mapping[str, Any]]
  producer_binding: Mapping[str, Any]

  def source_case(self, case_id: str) -> dict[str, Any]:
    try:
      return copy.deepcopy(dict(self._source_cases[str(case_id)]))
    except KeyError as error:
      raise KeyError(f"Unknown private source case: {case_id!r}") from error

  def gold_case(self, case_id: str) -> dict[str, Any]:
    try:
      return copy.deepcopy(dict(self._gold_cases[str(case_id)]))
    except KeyError as error:
      raise KeyError(f"Unknown private gold case: {case_id!r}") from error


def load_benchmark_v2_private_supervision(
    *,
    source_path: str | Path,
    gold_path: str | Path,
    family_split_manifest_path: str | Path,
    source_split: str,
    public_case_ids: Sequence[str],
    _source_capture: CapturedJsonArtifact | None = None,
    _gold_capture: CapturedJsonArtifact | None = None,
    _family_manifest_capture: CapturedJsonArtifact | None = None,
    pilot_only: bool = False,
) -> BenchmarkV2PrivateSupervision:
  """Load and authenticate the exact private sidecars for one public split."""

  split = str(source_split).strip().lower()
  if split not in {"train", "dev"}:
    raise ValueError("private supervision source_split must be train or dev")
  normalized_public_ids = [str(value).strip() for value in public_case_ids]
  if (
      not normalized_public_ids
      or any(not value for value in normalized_public_ids)
      or len(set(normalized_public_ids)) != len(normalized_public_ids)
  ):
    raise ValueError("public case set must be nonempty and unique")
  source_capture = _source_capture or capture_json_artifact(
      source_path,
      label="private source split",
  )
  gold_capture = _gold_capture or capture_json_artifact(
      gold_path,
      label="private evaluation gold split",
  )
  manifest_capture = _family_manifest_capture or capture_json_artifact(
      family_split_manifest_path,
      label="family split manifest",
  )
  source_payload = source_capture.payload
  gold_payload = gold_capture.payload
  manifest = manifest_capture.payload
  expected_identity = (
      isinstance(manifest, Mapping)
      and manifest.get("benchmark") == "benchmark_v2"
      and manifest.get("strict_hygiene_clean") is True
      and (
          manifest.get("formal") is False
          and manifest.get("pilot_only") is True
          and manifest.get("development") is True
          if pilot_only
          else manifest.get("formal") is True
          and manifest.get("pilot_only") is not True
      )
  )
  if not expected_identity:
    qualifier = "development-pilot" if pilot_only else "formal"
    raise ValueError(
        f"private supervision requires a clean {qualifier} family manifest"
    )
  source_cases = _validate_private_source_payload(source_payload, split=split)
  gold_cases = _validate_private_gold_payload(
      gold_payload,
      split=split,
      source_cases=source_cases,
  )
  case_set = set(normalized_public_ids)
  if set(source_cases) != case_set or set(gold_cases) != case_set:
    raise ValueError("private supervision does not exactly match the public case set")
  source_payload_sha = _manifest_commitment(
      manifest,
      section_name="private_source_bindings",
      schema_version=PRIVATE_SOURCE_SPLIT_SCHEMA,
      split=split,
      payload=source_payload,
  )
  gold_payload_sha = _manifest_commitment(
      manifest,
      section_name="private_evaluation_gold",
      schema_version=PRIVATE_GOLD_SPLIT_SCHEMA,
      split=split,
      payload=gold_payload,
  )
  binding = {
      "schema_version": PRIVATE_SUPERVISION_BINDING_SCHEMA,
      "source_schema_version": PRIVATE_SOURCE_SPLIT_SCHEMA,
      "gold_split_schema_version": PRIVATE_GOLD_SPLIT_SCHEMA,
      "gold_contact_schema_version": PRIVATE_GOLD_CONTACT_SCHEMA,
      "split": split,
      "case_count": len(case_set),
      "case_set_sha256": canonical_sha256(sorted(case_set)),
      "source_payload_sha256": source_payload_sha,
      "source_file_sha256": source_capture.sha256,
      "gold_payload_sha256": gold_payload_sha,
      "gold_file_sha256": gold_capture.sha256,
      "family_split_manifest_sha256": manifest_capture.sha256,
      "gold_label_semantics": "source_positive_assertions_only",
      "gold_interface_certificate_ready": False,
      "certified_negative_complete": False,
  }
  return BenchmarkV2PrivateSupervision(
      split=split,
      _source_cases=MappingProxyType(source_cases),
      _gold_cases=MappingProxyType(gold_cases),
      producer_binding=MappingProxyType(binding),
  )


def require_certified_binary_training_gold(gold_case: Mapping[str, Any]) -> None:
  """Reject v2 source-positive assertions as a source of binary negatives."""

  del gold_case
  require_formal_binary_training_available()


__all__ = [
    "BenchmarkV2PrivateSupervision",
    "PRIVATE_GOLD_CONTACT_SCHEMA",
    "PRIVATE_GOLD_SPLIT_SCHEMA",
    "PRIVATE_SOURCE_SPLIT_SCHEMA",
    "PRIVATE_SUPERVISION_BINDING_SCHEMA",
    "load_benchmark_v2_private_supervision",
    "require_certified_binary_training_gold",
]
