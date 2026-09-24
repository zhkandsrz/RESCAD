"""Torch-free protocol and loader for the frozen 19-program catalog.

This module intentionally uses only the Python standard library.  It mirrors
the immutable commitments of the historical neural catalog implementation
without importing that implementation, NumPy, Torch, CadQuery, or OCC.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


CATALOG_SCHEMA_VERSION = "finite_program_catalog_roster.v2"
CATALOG_SPEC_SCHEMA_VERSION = "externally_frozen_program_catalog_spec.v3"
FORMAL_CATALOG_SIZE = 19
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

OFFICIAL_FROZEN_CATALOG_COMMITMENTS_V1: Mapping[str, str] = MappingProxyType({
    "trust_root_id": "p0_frozen_program_roster_20260719.v1",
    "spec_filename": (
        "externally_frozen_program_catalog_spec_v3_p0_20260719.json"
    ),
    "catalog_roster_sha256": (
        "7e97af33edd35186499f6e187a85a0b07a0a0f31afe105b36d46754b49755724"
    ),
    "spec_file_sha256": (
        "e97aed9f936284429698d783ae3eb56ae6d42f8ec72d94da5c4eae26bdc760db"
    ),
    "spec_payload_sha256": (
        "71200b9b894419f8519ea1dabd5809207702903d67b44820a2c59b54faaceb3d"
    ),
    "source_artifact_sha256": (
        "2ca8cb6e13e2b176267ac1f81e13fcbc474f791e89dadb282e1ff6f4a6370966"
    ),
    "producer_code_sha256": (
        "9f4151a5624d426516cd5d479ac54da5723413203afaa57cf0810c63f26d088c"
    ),
})


def _canonical_sha256(value: Any) -> str:
  raw = json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes) -> Any:
  def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"frozen catalog contains duplicate key: {key}")
      result[key] = value
    return result

  try:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"frozen catalog contains non-finite JSON: {value}")
        ),
    )
  except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError("frozen catalog is not strict UTF-8 JSON") from error


@dataclass(frozen=True, slots=True)
class FixedProgramDescriptorV1:
  relation_hint: str
  contact_type: str
  surface_type_a: str
  surface_type_b: str

  def __post_init__(self) -> None:
    for value in (
        self.relation_hint,
        self.contact_type,
        self.surface_type_a,
        self.surface_type_b,
    ):
      if (
          type(value) is not str
          or not value
          or value != value.strip().lower()
          or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None
      ):
        raise ValueError("frozen catalog descriptor token differs")

  def payload(self) -> dict[str, str]:
    return {
        "relation_hint": self.relation_hint,
        "contact_type": self.contact_type,
        "surface_type_a": self.surface_type_a,
        "surface_type_b": self.surface_type_b,
    }

  @property
  def sha256(self) -> str:
    return _canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class FixedProgramEntryV1:
  program_index: int
  program_id: str
  descriptor: FixedProgramDescriptorV1
  descriptor_sha256: str

  def __post_init__(self) -> None:
    if (
        type(self.program_index) is not int
        or not 0 <= self.program_index < FORMAL_CATALOG_SIZE
        or type(self.descriptor) is not FixedProgramDescriptorV1
        or self.descriptor_sha256 != self.descriptor.sha256
        or self.program_id != f"catalog_{self.descriptor_sha256[:16]}"
    ):
      raise ValueError("frozen catalog entry identity differs")

  def payload(self) -> dict[str, Any]:
    return {
        "program_index": self.program_index,
        "program_id": self.program_id,
        "descriptor": self.descriptor.payload(),
        "descriptor_sha256": self.descriptor_sha256,
    }


@dataclass(frozen=True, slots=True)
class FixedProgramRosterV1:
  entries: tuple[FixedProgramEntryV1, ...]
  catalog_sha256: str
  schema_version: str = CATALOG_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if (
        self.schema_version != CATALOG_SCHEMA_VERSION
        or len(self.entries) != FORMAL_CATALOG_SIZE
        or tuple(row.program_index for row in self.entries)
        != tuple(range(FORMAL_CATALOG_SIZE))
        or len({row.program_id for row in self.entries})
        != FORMAL_CATALOG_SIZE
        or len({row.descriptor_sha256 for row in self.entries})
        != FORMAL_CATALOG_SIZE
    ):
      raise ValueError("frozen catalog roster structure differs")
    payload = {
        "schema_version": self.schema_version,
        "entry_count": FORMAL_CATALOG_SIZE,
        "entries": [row.payload() for row in self.entries],
    }
    if self.catalog_sha256 != _canonical_sha256(payload):
      raise ValueError("frozen catalog roster hash differs")


@dataclass(frozen=True, slots=True)
class FrozenProgramCatalogSpecV1:
  roster: FixedProgramRosterV1
  source_artifact_sha256: str
  producer_code_sha256: str
  spec_payload_sha256: str
  spec_file_sha256: str


def load_official_frozen_program_catalog_v1(
    path: str | Path,
) -> FrozenProgramCatalogSpecV1:
  """Load the single official spec under fixed, code-owned commitments."""

  source = Path(path).resolve(strict=True)
  trust = OFFICIAL_FROZEN_CATALOG_COMMITMENTS_V1
  if (
      source.name != trust["spec_filename"]
      or not source.is_file()
      or source.is_symlink()
  ):
    raise ValueError("frozen catalog source path differs")
  raw = source.read_bytes()
  file_sha256 = hashlib.sha256(raw).hexdigest()
  if file_sha256 != trust["spec_file_sha256"]:
    raise ValueError("frozen catalog source bytes differ")
  payload = _strict_json(raw)
  if not isinstance(payload, Mapping) or set(payload) != {
      "schema_version",
      "source_artifact_sha256",
      "producer_code_sha256",
      "roster",
      "spec_payload_sha256",
  }:
    raise ValueError("frozen catalog spec fields differ")
  unsigned = dict(payload)
  observed_payload_sha256 = unsigned.pop("spec_payload_sha256", None)
  if (
      payload.get("schema_version") != CATALOG_SPEC_SCHEMA_VERSION
      or observed_payload_sha256 != _canonical_sha256(unsigned)
      or observed_payload_sha256 != trust["spec_payload_sha256"]
      or payload.get("source_artifact_sha256")
      != trust["source_artifact_sha256"]
      or payload.get("producer_code_sha256")
      != trust["producer_code_sha256"]
  ):
    raise ValueError("frozen catalog spec commitments differ")
  raw_roster = payload.get("roster")
  if not isinstance(raw_roster, Mapping) or set(raw_roster) != {
      "schema_version",
      "entry_count",
      "entries",
      "catalog_sha256",
  }:
    raise ValueError("frozen catalog roster fields differ")
  raw_entries = raw_roster.get("entries")
  if (
      raw_roster.get("schema_version") != CATALOG_SCHEMA_VERSION
      or raw_roster.get("entry_count") != FORMAL_CATALOG_SIZE
      or not isinstance(raw_entries, list)
      or len(raw_entries) != FORMAL_CATALOG_SIZE
  ):
    raise ValueError("frozen catalog roster domain differs")
  entries: list[FixedProgramEntryV1] = []
  for index, row in enumerate(raw_entries):
    if not isinstance(row, Mapping) or set(row) != {
        "program_index",
        "program_id",
        "descriptor",
        "descriptor_sha256",
    }:
      raise ValueError("frozen catalog entry fields differ")
    descriptor = row.get("descriptor")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "relation_hint",
        "contact_type",
        "surface_type_a",
        "surface_type_b",
    }:
      raise ValueError("frozen catalog descriptor fields differ")
    if row.get("program_index") != index:
      raise ValueError("frozen catalog entry order differs")
    entries.append(FixedProgramEntryV1(
        program_index=index,
        program_id=str(row.get("program_id") or ""),
        descriptor=FixedProgramDescriptorV1(
            relation_hint=str(descriptor.get("relation_hint") or ""),
            contact_type=str(descriptor.get("contact_type") or ""),
            surface_type_a=str(descriptor.get("surface_type_a") or ""),
            surface_type_b=str(descriptor.get("surface_type_b") or ""),
        ),
        descriptor_sha256=str(row.get("descriptor_sha256") or ""),
    ))
  roster = FixedProgramRosterV1(
      entries=tuple(entries),
      catalog_sha256=str(raw_roster.get("catalog_sha256") or ""),
  )
  if roster.catalog_sha256 != trust["catalog_roster_sha256"]:
    raise ValueError("alternate frozen catalog roster is forbidden")
  result = FrozenProgramCatalogSpecV1(
      roster=roster,
      source_artifact_sha256=str(payload["source_artifact_sha256"]),
      producer_code_sha256=str(payload["producer_code_sha256"]),
      spec_payload_sha256=str(observed_payload_sha256),
      spec_file_sha256=file_sha256,
  )
  if any(
      _SHA256.fullmatch(value) is None
      for value in (
          result.source_artifact_sha256,
          result.producer_code_sha256,
          result.spec_payload_sha256,
          result.spec_file_sha256,
      )
  ):
    raise ValueError("frozen catalog spec hash is malformed")
  return result


def assert_torch_free_import_boundary_v1() -> None:
  """Fail if this protocol import caused Torch to enter the process."""

  import sys

  if "torch" in sys.modules:
    raise RuntimeError("torch-free catalog protocol imported Torch")


__all__ = [
    "FixedProgramDescriptorV1",
    "FixedProgramEntryV1",
    "FixedProgramRosterV1",
    "FrozenProgramCatalogSpecV1",
    "OFFICIAL_FROZEN_CATALOG_COMMITMENTS_V1",
    "assert_torch_free_import_boundary_v1",
    "load_official_frozen_program_catalog_v1",
]
