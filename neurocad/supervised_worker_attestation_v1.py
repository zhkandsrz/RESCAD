"""Single-use parent attestations for isolated development workers.

The attestation is intentionally local and ephemeral.  It is not a signing
scheme: its job is to make the hidden worker entry point parent-only, bind the
exact public invocation and native-thread environment, and leave an immutable
receipt in the published development artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


ATTESTATION_SCHEMA = "supervised_worker_parent_attestation.v1"
ROLE_ENV = "NEUROCAD_SUPERVISED_WORKER_ROLE"
NONCE_ENV = "NEUROCAD_SUPERVISED_WORKER_NONCE"
ATTESTATION_ENV = "NEUROCAD_SUPERVISED_WORKER_ATTESTATION"
NATIVE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _bytes(value: Any) -> bytes:
  return json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def _sha(value: Any) -> str:
  return hashlib.sha256(_bytes(value)).hexdigest()


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
  def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
      if key in result:
        raise ValueError(f"{label} contains duplicate key")
      result[key] = value
    return result

  try:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
  except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict JSON") from error
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be an object")
  return value


def native_environment_receipt() -> dict[str, Any]:
  """Return the actual process values, not merely requested defaults."""

  values = {key: os.environ.get(key) for key in sorted(NATIVE_THREAD_ENV)}
  return {
      "schema_version": "native_thread_environment_receipt.v1",
      "values": values,
      "all_expected_values_observed": values == dict(sorted(NATIVE_THREAD_ENV.items())),
      "values_sha256": _sha(values),
  }


def public_argv_commitment(
    *, executable: str | Path, script: str | Path, public_arguments: Sequence[str],
) -> str:
  return _sha({
      "executable": str(Path(executable).resolve()),
      "script": str(Path(script).resolve()),
      "public_arguments": list(public_arguments),
  })


def parent_launch_attestation(
    *, nonce: str, worker_role: str, output: str | Path,
    code_bundle_sha256: str, script: str | Path, public_arguments: Sequence[str],
    parent_pid: int | None = None,
) -> dict[str, Any]:
  if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
    raise ValueError("supervised worker nonce is malformed")
  if _SHA256.fullmatch(code_bundle_sha256) is None:
    raise ValueError("supervised worker code binding differs")
  environment = dict(sorted(NATIVE_THREAD_ENV.items()))
  payload = {
      "schema_version": ATTESTATION_SCHEMA,
      "worker_role": str(worker_role),
      "parent_pid": os.getpid() if parent_pid is None else int(parent_pid),
      "worker_executable": str(Path(sys.executable).resolve()),
      "worker_script": str(Path(script).resolve()),
      "public_argv_commitment_sha256": public_argv_commitment(
          executable=sys.executable, script=script,
          public_arguments=public_arguments,
      ),
      "code_bundle_sha256": code_bundle_sha256,
      "native_environment": environment,
      "native_environment_sha256": _sha(environment),
      "output_path": str(Path(output).resolve()),
      "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
      "issued_time_ns": time.time_ns(),
      "single_use": True,
  }
  payload["attestation_payload_sha256"] = _sha(payload)
  return payload


def claim_parent_launch_attestation(
    *, output: str | Path, worker_role: str, code_bundle_sha256: str,
    script: str | Path, public_arguments: Sequence[str],
) -> tuple[bytes, str, dict[str, Any]]:
  """Atomically claim and verify the one parent-created launch capability."""

  role = os.environ.get(ROLE_ENV)
  nonce = os.environ.get(NONCE_ENV)
  raw_path = os.environ.get(ATTESTATION_ENV)
  if role != worker_role or nonce is None or raw_path is None:
    raise PermissionError("direct supervised worker invocation is forbidden")
  if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
    raise PermissionError("supervised worker nonce is malformed")
  source = Path(raw_path)
  claimed = source.with_name(source.name + f".claimed-{os.getpid()}")
  try:
    os.replace(source, claimed)
  except OSError as error:
    raise PermissionError(
        "supervised worker attestation is absent or already claimed"
    ) from error
  raw = claimed.read_bytes()
  payload = _strict_json(raw, label="supervised worker parent attestation")
  unsigned = dict(payload)
  observed = unsigned.pop("attestation_payload_sha256", None)
  expected_keys = {
      "schema_version", "worker_role", "parent_pid", "worker_executable",
      "worker_script", "public_argv_commitment_sha256", "code_bundle_sha256",
      "native_environment", "native_environment_sha256", "output_path",
      "nonce_sha256", "issued_time_ns", "single_use",
  }
  environment_receipt = native_environment_receipt()
  age_ns = time.time_ns() - int(payload.get("issued_time_ns", -1))
  if (
      set(unsigned) != expected_keys
      or observed != _sha(unsigned)
      or payload.get("schema_version") != ATTESTATION_SCHEMA
      or payload.get("worker_role") != worker_role
      or payload.get("parent_pid") != os.getppid()
      or payload.get("worker_executable") != str(Path(sys.executable).resolve())
      or payload.get("worker_script") != str(Path(script).resolve())
      or payload.get("public_argv_commitment_sha256") != public_argv_commitment(
          executable=sys.executable, script=script,
          public_arguments=public_arguments,
      )
      or payload.get("code_bundle_sha256") != code_bundle_sha256
      or payload.get("native_environment") != dict(sorted(NATIVE_THREAD_ENV.items()))
      or payload.get("native_environment_sha256")
      != _sha(dict(sorted(NATIVE_THREAD_ENV.items())))
      or environment_receipt["all_expected_values_observed"] is not True
      or payload.get("output_path") != str(Path(output).resolve())
      or payload.get("nonce_sha256")
      != hashlib.sha256(nonce.encode("ascii")).hexdigest()
      or payload.get("single_use") is not True
      or not 0 <= age_ns <= 120_000_000_000
  ):
    raise PermissionError("supervised worker parent attestation verification failed")
  return raw, hashlib.sha256(raw).hexdigest(), environment_receipt


__all__ = [
    "ATTESTATION_ENV", "NATIVE_THREAD_ENV", "NONCE_ENV", "ROLE_ENV",
    "claim_parent_launch_attestation", "native_environment_receipt",
    "parent_launch_attestation", "public_argv_commitment",
]
