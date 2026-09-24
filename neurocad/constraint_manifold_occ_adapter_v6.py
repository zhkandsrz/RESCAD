"""V6 lifecycle-audited wrapper for the frozen V4 exact OCC operation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .constraint_manifold_occ_adapter_v4 import (
    OccExactManifoldExecutorV4, WORKER_REQUEST_SCHEMA_VERSION,
    WORKER_TERMINAL_SCHEMA_VERSION,
)
from .constraint_manifold_solver_v4 import (
    ExactChildReceiptV4, ManifoldCandidateV4, canonical_sha256,
    issue_exact_child_receipt_v4,
)
from .supervised_worker_attestation_v1 import NATIVE_THREAD_ENV


EXACT_CLEANUP_WINDOW_SECONDS_V6 = 0.5


class OccExactManifoldExecutorV6(OccExactManifoldExecutorV4):
  """Same V4 exact policy with explicit terminal/reap/cleanup evidence."""

  __slots__ = ()

  def run_child(self, candidate: ManifoldCandidateV4, *, call_nonce: str,
                absolute_deadline: float) -> ExactChildReceiptV4:
    if type(candidate) is not ManifoldCandidateV4:
      raise TypeError("V6 exact executor requires a V4 candidate")
    if call_nonce in self._seen_nonces:
      raise ValueError("V6 exact nonce was reused")
    self._seen_nonces.add(call_nonce)
    started_at = time.monotonic()
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
    environment["PATH"] = (
        str(Path(sys.prefix).resolve() / "Library" / "bin") + os.pathsep
        + environment.get("PATH", "")
    )
    process: subprocess.Popen[str] | None = None

    def receipt(status: str, *, child_started: bool, issuer: str,
                result: dict[str, Any], decision_at: float,
                cleanup_elapsed: float = 0.0) -> ExactChildReceiptV4:
      child_reaped = not child_started or (process is not None and process.poll() is not None)
      normalized = dict(result)
      if "error" in normalized:
        normalized["error_sha256"] = hashlib.sha256(
            str(normalized.pop("error")).encode("utf-8")
        ).hexdigest()
      normalized.update({
          "decision_elapsed_seconds": max(0.0, decision_at - started_at),
          "cleanup_window_seconds": EXACT_CLEANUP_WINDOW_SECONDS_V6,
          "cleanup_elapsed_seconds": cleanup_elapsed,
          "child_reaped": child_reaped,
          "unreaped": bool(child_started and not child_reaped),
      })
      return issue_exact_child_receipt_v4(
          call_nonce=call_nonce, terminal_status=status, child_started=child_started,
          issuer=issuer, result=normalized,
      )

    if time.monotonic() >= absolute_deadline:
      now = time.monotonic()
      return receipt(
          "timeout", child_started=False, issuer="parent_observer",
          result={"error": "deadline_expired_before_spawn"}, decision_at=now,
      )
    try:
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
        now = time.monotonic()
        return receipt(
            "kernel_error", child_started=False, issuer="parent_observer",
            result={"error": f"spawn_failed:{type(error).__name__}"}, decision_at=now,
        )
      try:
        stdout, stderr = process.communicate(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            timeout=max(0.0, absolute_deadline - time.monotonic()),
        )
      except Exception as communicate_error:
        decision_at = time.monotonic()
        kill_error = None
        try:
          process.kill()
        except Exception as error:
          kill_error = type(error).__name__
        cleanup_started = time.monotonic()
        stdout = stderr = ""
        wait_error = None
        try:
          stdout, stderr = process.communicate(timeout=EXACT_CLEANUP_WINDOW_SECONDS_V6)
        except Exception as error:
          wait_error = type(error).__name__
        cleanup_elapsed = time.monotonic() - cleanup_started
        return receipt(
            "timeout" if isinstance(communicate_error, subprocess.TimeoutExpired)
            else "kernel_error",
            child_started=True, issuer="parent_observer",
            result={
                "error": f"communicate_failed:{type(communicate_error).__name__}",
                "kill_error_type": kill_error, "wait_error_type": wait_error,
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            },
            decision_at=decision_at, cleanup_elapsed=cleanup_elapsed,
        )
      decision_at = time.monotonic()
      if decision_at > absolute_deadline:
        return receipt(
            "timeout", child_started=True, issuer="parent_observer",
            result={"error": "deadline_expired_before_parse",
                    "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest()},
            decision_at=decision_at,
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
        return receipt(
            "kernel_error", child_started=True, issuer="parent_observer",
            result={"error": "worker_terminal_invalid",
                    "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest()},
            decision_at=decision_at,
        )
      result = dict(terminal["result"])
      status = str(result.pop("status", "kernel_error"))
      echoed_nonce = result.pop("call_nonce", None)
      echoed_operation = result.pop("operation", None)
      if echoed_nonce != call_nonce or echoed_operation != "candidate_exact_bundle":
        status, result = "kernel_error", {"error": "child_echo_mismatch"}
      if status not in {"ok", "timeout", "kernel_error"}:
        status = "kernel_error"
      result["worker_terminal_payload_sha256"] = terminal_hash
      return receipt(
          status, child_started=True, issuer="child_process_echo", result=result,
          decision_at=decision_at,
      )
    finally:
      if process is not None and process.poll() is None:
        try:
          process.kill()
        except Exception:
          pass
        try:
          process.wait(timeout=EXACT_CLEANUP_WINDOW_SECONDS_V6)
        except Exception:
          pass


__all__ = ["EXACT_CLEANUP_WINDOW_SECONDS_V6", "OccExactManifoldExecutorV6"]
