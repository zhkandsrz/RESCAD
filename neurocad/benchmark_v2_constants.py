"""Dependency-free identities shared by benchmark-v2 producers and consumers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_VERSION = "benchmark_v2_independent_se3_v1"

EXACT_CERTIFICATE_SEMANTICS_VERSION = (
    "benchmark_v2_predicted_interface_physical_consistency.v1"
)
EXACT_CERTIFICATE_SCOPE = "predicted_interface_physical_consistency_only"
GOLD_TARGET_INTERFACE_STATUS = "unavailable"
GOLD_TARGET_INTERFACE_REASON_CODE = (
    "fusion360assembly_to_step_face_identity_unverified"
)
FINAL_FORMAL_ACCEPTANCE_REASON_CODE = "gold_target_interface_not_certified"

EXACT_FACE_BINDING_SCHEMA = "benchmark_v2_exact_face_binding.v2"
EXACT_FACE_IDENTITY_PROOFS = frozenset(
    {"occ_tshape_partner_bijection", "unique_signature_bijection"}
)


def exact_face_binding_sha256(
    *,
    namespace: str,
    raw_to_randomized_transform: dict[str, Any],
    interface_id: str,
    raw_face_indices: list[int],
    raw_face_signature_sha256s: list[str],
    face_identity_proof: str,
) -> str:
  """Bind a certificate-only face set to one private benchmark gauge."""

  clean_namespace = str(namespace)
  clean_interface = str(interface_id)
  clean_proof = str(face_identity_proof)
  supplied_indices = [int(index) for index in raw_face_indices]
  clean_indices = sorted(set(supplied_indices))
  clean_signatures = [str(value) for value in raw_face_signature_sha256s]
  if not clean_namespace or not clean_interface or not clean_indices:
    raise ValueError("exact face binding identity must be nonempty")
  if clean_indices[0] < 0:
    raise ValueError("exact face binding indices must be nonnegative")
  if supplied_indices != clean_indices:
    raise ValueError("exact face binding indices must be sorted and unique")
  if clean_proof not in EXACT_FACE_IDENTITY_PROOFS:
    raise ValueError("exact face binding proof is not protocol-approved")
  if len(clean_signatures) != len(clean_indices) or any(
      len(value) != 64
      or any(character not in "0123456789abcdef" for character in value)
      for value in clean_signatures
  ):
    raise ValueError("exact face binding signatures must be one SHA-256 per face")
  payload = {
      "schema": EXACT_FACE_BINDING_SCHEMA,
      "namespace": clean_namespace,
      "raw_to_randomized_transform": raw_to_randomized_transform,
      "interface_id": clean_interface,
      "raw_face_indices": clean_indices,
      "raw_face_signature_sha256s": clean_signatures,
      "face_identity_proof": clean_proof,
  }
  encoded = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()
