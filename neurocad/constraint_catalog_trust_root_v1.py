"""Version-controlled trust root for the only exact-executable V2 roster.

This is an allowlist, not a caller-supplied pin bundle.  The small frozen
catalog may be mirrored for reproducible tests, but its filename and all four
independent commitments must match this policy before an exact-execution
capability can be issued.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping


OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1: Mapping[str, str] = MappingProxyType({
    "trust_root_id": "p0_frozen_program_roster_20260719.v1",
    "spec_filename": "externally_frozen_program_catalog_spec_v3_p0_20260719.json",
    "catalog_roster_sha256": "7e97af33edd35186499f6e187a85a0b07a0a0f31afe105b36d46754b49755724",
    "spec_file_sha256": "e97aed9f936284429698d783ae3eb56ae6d42f8ec72d94da5c4eae26bdc760db",
    "spec_payload_sha256": "71200b9b894419f8519ea1dabd5809207702903d67b44820a2c59b54faaceb3d",
    "source_artifact_sha256": "2ca8cb6e13e2b176267ac1f81e13fcbc474f791e89dadb282e1ff6f4a6370966",
    "producer_code_sha256": "9f4151a5624d426516cd5d479ac54da5723413203afaa57cf0810c63f26d088c",
})


def require_official_constraint_catalog_trust_root_v1(
    source_spec_path: str | Path,
    *,
    catalog_roster_sha256: str,
    spec_file_sha256: str,
    source_artifact_sha256: str,
    producer_code_sha256: str,
) -> Path:
  """Reject a self-consistent alternate roster before parsing its payload."""

  trust = OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1
  observed = {
      "catalog_roster_sha256": catalog_roster_sha256,
      "spec_file_sha256": spec_file_sha256,
      "source_artifact_sha256": source_artifact_sha256,
      "producer_code_sha256": producer_code_sha256,
  }
  if any(observed[key] != trust[key] for key in observed):
    raise ValueError("constraint catalog differs from official trust root")
  path = Path(source_spec_path).resolve(strict=True)
  if path.name != trust["spec_filename"]:
    raise ValueError("constraint catalog path differs from official trust root")
  return path


__all__ = [
    "OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1",
    "require_official_constraint_catalog_trust_root_v1",
]
