"""Supervised native OCC exact executor for manifold solver V4."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from .constraint_manifold_solver_v3 import file_sha256
from .constraint_manifold_solver_v4 import (
    ExactChildReceiptV4, ManifoldCandidateV4, canonical_sha256,
    issue_exact_child_receipt_v4,
)
from .domain_types import Transform
from .supervised_worker_attestation_v1 import NATIVE_THREAD_ENV


WORKER_REQUEST_SCHEMA_VERSION = "constraint_manifold_occ_worker_request.v4"
WORKER_TERMINAL_SCHEMA_VERSION = "constraint_manifold_occ_worker_terminal.v4"


def _matrix_tuple(transform: Transform) -> tuple[float, ...]:
  import numpy as np
  matrix = np.eye(4)
  matrix[:3, :3] = np.asarray(transform.rotation)
  matrix[:3, 3] = np.asarray(transform.translation)
  return tuple(float(value) for value in matrix.reshape(-1))


class OccExactManifoldExecutorV4:
  """Runs each selected top-16 candidate in a fresh killable process."""

  __slots__ = (
      "step_path_a", "step_path_b", "step_sha256_a", "step_sha256_b",
      "raw_face_index_a", "raw_face_index_b", "face_signature_a", "face_signature_b",
      "world_a_row_major", "plane_frame_a", "plane_origin_a", "_seen_nonces",
  )

  def __init__(
      self, *, step_paths: Sequence[str | Path], step_sha256s: Sequence[str],
      raw_face_indices: Sequence[int], face_signatures: Sequence[str],
      world_a: Transform, selected_endpoint_a: dict[str, Any],
  ) -> None:
    if not all(len(row) == 2 for row in (
        step_paths, step_sha256s, raw_face_indices, face_signatures,
    )):
      raise ValueError("V4 exact endpoint cardinality differs")
    self.step_path_a = str(Path(step_paths[0]).resolve(strict=True))
    self.step_path_b = str(Path(step_paths[1]).resolve(strict=True))
    self.step_sha256_a, self.step_sha256_b = map(str, step_sha256s)
    if file_sha256(self.step_path_a) != self.step_sha256_a or (
        file_sha256(self.step_path_b) != self.step_sha256_b
    ):
      raise ValueError("V4 exact STEP bytes differ")
    self.raw_face_index_a, self.raw_face_index_b = map(int, raw_face_indices)
    self.face_signature_a, self.face_signature_b = map(str, face_signatures)
    self.world_a_row_major = _matrix_tuple(world_a)
    self.plane_frame_a = selected_endpoint_a["rotation_local_to_world"]
    self.plane_origin_a = selected_endpoint_a["origin_world_mm"]
    self._seen_nonces: set[str] = set()

  def run_child(self, candidate: ManifoldCandidateV4, *, call_nonce: str,
                absolute_deadline: float) -> ExactChildReceiptV4:
    if type(candidate) is not ManifoldCandidateV4:
      raise TypeError("V4 exact executor requires a V4 candidate")
    if call_nonce in self._seen_nonces:
      raise ValueError("V4 exact nonce was reused")
    self._seen_nonces.add(call_nonce)
    arguments = {
        "operation": "candidate_exact_bundle",
        "step_path_a": self.step_path_a, "step_path_b": self.step_path_b,
        "step_sha256_a": self.step_sha256_a, "step_sha256_b": self.step_sha256_b,
        "raw_face_index_a": self.raw_face_index_a,
        "raw_face_index_b": self.raw_face_index_b,
        "face_signature_a": self.face_signature_a,
        "face_signature_b": self.face_signature_b,
        "world_a_row_major": self.world_a_row_major,
        "child_world_row_major": candidate.child_world_row_major,
        "plane_frame_a": self.plane_frame_a, "plane_origin_a": self.plane_origin_a,
        "call_nonce": call_nonce,
    }
    unsigned = {"schema_version": WORKER_REQUEST_SCHEMA_VERSION, "arguments": arguments}
    request = {**unsigned, "request_payload_sha256": canonical_sha256(unsigned)}
    worker = Path(__file__).resolve().parent / "tools" / "constraint_manifold_occ_worker_v4.py"
    environment = os.environ.copy()
    environment.update(NATIVE_THREAD_ENV)
    package_parent = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = package_parent + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["PATH"] = str(Path(sys.prefix).resolve() / "Library" / "bin") + os.pathsep + environment.get("PATH", "")
    process: subprocess.Popen[str] | None = None
    stdout = stderr = ""
    try:
      remaining = absolute_deadline - time.monotonic()
      if remaining <= 0.0:
        return issue_exact_child_receipt_v4(
            call_nonce=call_nonce, terminal_status="timeout", child_started=False,
            issuer="parent_observer", result={"error": "deadline_expired_before_spawn"},
        )
      try:
        process = subprocess.Popen(
            [sys.executable, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=environment,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                if os.name == "nt" else 0
            ),
        )
      except Exception as error:
        return issue_exact_child_receipt_v4(
            call_nonce=call_nonce, terminal_status="kernel_error", child_started=False,
            issuer="parent_observer",
            result={"error": f"spawn_failed:{type(error).__name__}:{error}"},
        )
      try:
        stdout, stderr = process.communicate(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            timeout=max(0.0, absolute_deadline - time.monotonic()),
        )
      except Exception as communicate_error:
        kill_error = wait_error = None
        try:
          process.kill()
        except Exception as error:
          kill_error = f"{type(error).__name__}:{error}"
        try:
          stdout, stderr = process.communicate(timeout=2.0)
        except Exception as error:
          wait_error = f"{type(error).__name__}:{error}"
        return issue_exact_child_receipt_v4(
            call_nonce=call_nonce,
            terminal_status=(
                "timeout" if isinstance(communicate_error, subprocess.TimeoutExpired)
                else "kernel_error"
            ),
            child_started=True, issuer="parent_observer",
            result={
                "error": f"communicate_failed:{type(communicate_error).__name__}:{communicate_error}",
                "kill_error": kill_error, "wait_error": wait_error,
                "child_pid": process.pid,
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            },
        )
      try:
        terminal = json.loads(stdout.strip().splitlines()[-1])
        terminal_unsigned = dict(terminal)
        terminal_hash = terminal_unsigned.pop("terminal_payload_sha256", None)
        valid = (
            terminal.get("schema_version") == WORKER_TERMINAL_SCHEMA_VERSION
            and terminal.get("request_payload_sha256") == request["request_payload_sha256"]
            and terminal_hash == canonical_sha256(terminal_unsigned)
        )
      except Exception:
        terminal, valid, terminal_hash = {}, False, None
      if not valid:
        return issue_exact_child_receipt_v4(
            call_nonce=call_nonce, terminal_status="kernel_error", child_started=True,
            issuer="parent_observer", result={
                "error": f"worker_terminal_invalid:exit={process.returncode}",
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            },
        )
      result = dict(terminal["result"])
      status = str(result.pop("status", "kernel_error"))
      echoed_nonce = result.pop("call_nonce", None)
      echoed_operation = result.pop("operation", None)
      if echoed_nonce != call_nonce or echoed_operation != "candidate_exact_bundle":
        status = "kernel_error"
        result = {"error": "child_nonce_or_operation_echo_mismatch", "child_pid": process.pid}
      if status not in {"ok", "timeout", "kernel_error"}:
        status = "kernel_error"
      result["worker_terminal_payload_sha256"] = terminal_hash
      return issue_exact_child_receipt_v4(
          call_nonce=call_nonce, terminal_status=status, child_started=True,
          issuer="child_process_echo", result=result,
      )
    finally:
      if process is not None and process.poll() is None:
        try:
          process.kill()
        except Exception:
          pass
        try:
          process.wait(timeout=2.0)
        except Exception:
          pass


__all__ = [
    "WORKER_REQUEST_SCHEMA_VERSION", "WORKER_TERMINAL_SCHEMA_VERSION",
    "OccExactManifoldExecutorV4",
]
