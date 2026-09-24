"""Single-load, killable OCC exact batch executor for frozen V4 candidates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cadquery_backend import (
    boolean_intersection, load_step_shape, shape_bbox, shape_volume,
    source_face_signature_sha256, transform_shape,
)
from .constraint_manifold_occ_adapter_v3 import _transform, _transformed_bbox
from .constraint_manifold_occ_adapter_v6 import (
    EXACT_CLEANUP_WINDOW_SECONDS_V6, OccExactManifoldExecutorV6,
)
from .constraint_manifold_solver_v3 import file_sha256
from .constraint_manifold_solver_v4 import (
    BATCH_CANDIDATE_RECEIPT_SCHEMA_VERSION_V8,
    ExactBatchCandidateReceiptV8, ExactBatchChildReceiptV8, ManifoldCandidateV4,
    OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4, canonical_sha256,
    issue_exact_batch_candidate_receipt_v8, issue_exact_batch_child_receipt_v8,
)
from .supervised_worker_attestation_v1 import NATIVE_THREAD_ENV


WORKER_REQUEST_SCHEMA_VERSION_V8 = "constraint_manifold_occ_batch_worker_request.v8"
WORKER_TERMINAL_SCHEMA_VERSION_V8 = "constraint_manifold_occ_batch_worker_terminal.v8"


def _exact_metrics_from_loaded_v8(
    *, source_a: Any, source_b: Any, face_a: Any, face_b: Any,
    world_a_row_major: Sequence[float], child_world_row_major: Sequence[float],
    plane_frame_a: Sequence[Sequence[float]], plane_origin_a: Sequence[float],
) -> dict[str, Any]:
  """Evaluate the same two OCC observations as the legacy individual worker."""

  world_a, world_b = _transform(world_a_row_major), _transform(child_world_row_major)
  moved_face_a = transform_shape(face_a, world_a)
  moved_face_b = transform_shape(face_b, world_b)
  distance = float(moved_face_a.distance(moved_face_b))
  low_a, high_a, corners_a = _transformed_bbox(source_a, world_a)
  low_b, high_b, corners_b = _transformed_bbox(source_b, world_b)
  overlap = np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
  aabb_upper = 0.0 if np.any(overlap <= 0.0) else float(np.prod(overlap))
  threshold = OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.thresholds.whole_common_volume_mm3
  if aabb_upper <= threshold:
    volume = 0.0
  else:
    volume = max(0.0, float(shape_volume(boolean_intersection(
        transform_shape(source_a, world_a), transform_shape(source_b, world_b),
    ))))
  frame = np.asarray(plane_frame_a, dtype=float).reshape(3, 3)
  origin = np.asarray(plane_origin_a, dtype=float).reshape(3)
  local_a = (corners_a - origin) @ frame
  local_b = (corners_b - origin) @ frame
  proj_a = np.min(local_a[:, :2], axis=0), np.max(local_a[:, :2], axis=0)
  proj_b = np.min(local_b[:, :2], axis=0), np.max(local_b[:, :2], axis=0)
  epsilon = 1e-4
  offsets = (
      (float(proj_a[1][0] - proj_b[0][0] + epsilon), 0.0),
      (float(proj_a[0][0] - proj_b[1][0] - epsilon), 0.0),
      (0.0, float(proj_a[1][1] - proj_b[0][1] + epsilon)),
      (0.0, float(proj_a[0][1] - proj_b[1][1] - epsilon)),
  )
  return {
      "distance_mm": distance, "common_volume_mm3": volume,
      "aabb_intersection_upper_mm3": aabb_upper,
      "projected_aabb_separation_offsets_xy_mm": offsets,
  }


def execute_occ_exact_batch_v8(
    arguments: Mapping[str, Any], *, request_payload_sha256: str,
    emit: Callable[[ExactBatchCandidateReceiptV8], None],
) -> dict[str, Any]:
  """Load A/B once, evaluate frozen rank order, and stop at first feasible row."""

  if arguments.get("operation") != "candidate_exact_batch":
    raise ValueError("V8 OCC batch operation differs")
  if file_sha256(arguments["step_path_a"]) != arguments["step_sha256_a"] or (
      file_sha256(arguments["step_path_b"]) != arguments["step_sha256_b"]
  ):
    raise ValueError("V8 batch receipt-bound STEP bytes changed")
  source_a = load_step_shape(arguments["step_path_a"])
  source_b = load_step_shape(arguments["step_path_b"])
  faces_a, faces_b = tuple(source_a.Faces()), tuple(source_b.Faces())
  face_a = faces_a[int(arguments["raw_face_index_a"])]
  face_b = faces_b[int(arguments["raw_face_index_b"])]
  if source_face_signature_sha256(face_a) != arguments["face_signature_a"] or (
      source_face_signature_sha256(face_b) != arguments["face_signature_b"]
  ):
    raise ValueError("V8 batch reloaded STEP face signature differs")
  candidates = arguments.get("candidates")
  if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
    raise ValueError("V8 OCC batch candidate cardinality differs")
  batch_nonce = str(arguments.get("batch_nonce", ""))
  thresholds = OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.thresholds
  evaluated = 0
  stop_reason = "exhausted"
  for sequence, row in enumerate(candidates):
    if not isinstance(row, Mapping) or int(row.get("rank_zero_based", -1)) != sequence:
      raise ValueError("V8 OCC batch candidate order differs")
    candidate_key = str(row.get("candidate_key", ""))
    call_nonce = str(row.get("call_nonce", ""))
    try:
      result = _exact_metrics_from_loaded_v8(
          source_a=source_a, source_b=source_b, face_a=face_a, face_b=face_b,
          world_a_row_major=arguments["world_a_row_major"],
          child_world_row_major=row["child_world_row_major"],
          plane_frame_a=arguments["plane_frame_a"],
          plane_origin_a=arguments["plane_origin_a"],
      )
      status = "ok"
    except BaseException as error:
      status = "kernel_error"
      result = {"error_sha256": hashlib.sha256(
          f"{type(error).__name__}:{error}".encode("utf-8")
      ).hexdigest()}
    receipt = issue_exact_batch_candidate_receipt_v8(
        request_payload_sha256=request_payload_sha256, batch_nonce=batch_nonce,
        rank_zero_based=sequence, candidate_key=candidate_key,
        call_nonce=call_nonce, terminal_status=status, result=result,
    )
    emit(receipt)
    evaluated += 1
    if status != "ok":
      stop_reason = "candidate_kernel_error"
      break
    if (float(result["distance_mm"]) <= thresholds.selected_face_clearance_mm
        and float(result["common_volume_mm3"]) <= thresholds.whole_common_volume_mm3):
      stop_reason = "first_feasible"
      break
  return {
      "stop_reason": stop_reason, "evaluated_candidate_count": evaluated,
      "unassessed_candidate_count": len(candidates) - evaluated,
      "occ_calls_reserved": 2 * evaluated, "step_load_count": 2,
      "process_launch_count": 1,
  }


def _candidate_receipt_from_payload_v8(payload: Mapping[str, Any]) -> ExactBatchCandidateReceiptV8:
  result = payload.get("result")
  if not isinstance(result, Mapping):
    raise ValueError("V8 batch event result differs")
  return ExactBatchCandidateReceiptV8(
      request_payload_sha256=str(payload.get("request_payload_sha256", "")),
      batch_nonce=str(payload.get("batch_nonce", "")),
      rank_zero_based=int(payload.get("rank_zero_based", -1)),
      candidate_key=str(payload.get("candidate_key", "")),
      call_nonce=str(payload.get("call_nonce", "")),
      terminal_status=str(payload.get("terminal_status", "")),
      result_items=tuple(sorted((str(key), value) for key, value in result.items())),
      event_payload_sha256=str(payload.get("event_payload_sha256", "")),
      schema_version=str(payload.get("schema_version", "")),
  )


def _parse_worker_stdout_v8(
    stdout: str, *, request_payload_sha256: str, batch_nonce: str,
    worker_terminal_schema: str = WORKER_TERMINAL_SCHEMA_VERSION_V8,
) -> tuple[
    tuple[ExactBatchCandidateReceiptV8, ...], Mapping[str, Any] | None, str | None,
]:
  """Return the maximal verified JSONL prefix and a sanitized tail error."""

  events: list[ExactBatchCandidateReceiptV8] = []
  terminal: Mapping[str, Any] | None = None
  parse_error: str | None = None
  for raw in stdout.splitlines():
    if not raw.strip():
      continue
    try:
      payload = json.loads(raw)
    except Exception:
      parse_error = "invalid_json_record"
      break
    if not isinstance(payload, Mapping):
      parse_error = "non_mapping_worker_record"
      break
    if payload.get("schema_version") == BATCH_CANDIDATE_RECEIPT_SCHEMA_VERSION_V8:
      try:
        event = _candidate_receipt_from_payload_v8(payload)
        if (terminal is not None
            or event.request_payload_sha256 != request_payload_sha256
            or event.batch_nonce != batch_nonce or event.rank_zero_based != len(events)):
          raise ValueError("binding")
      except Exception:
        parse_error = "candidate_event_binding_or_commitment"
        break
      events.append(event)
    elif payload.get("schema_version") == worker_terminal_schema:
      if terminal is not None:
        parse_error = "duplicate_worker_terminal"
        break
      terminal = payload
    else:
      parse_error = "unknown_worker_record_schema"
      break
  return tuple(events), terminal, parse_error


class OccExactManifoldExecutorV8(OccExactManifoldExecutorV6):
  """One killable process for an ordered exact domain; individual replay stays inherited."""

  __slots__ = ()
  WORKER_REQUEST_SCHEMA = WORKER_REQUEST_SCHEMA_VERSION_V8
  WORKER_TERMINAL_SCHEMA = WORKER_TERMINAL_SCHEMA_VERSION_V8
  WORKER_FILENAME = "constraint_manifold_occ_batch_worker_v8.py"

  def run_batch(
      self, candidates: Sequence[ManifoldCandidateV4], *,
      call_nonces: Sequence[str], batch_nonce: str, absolute_deadline: float,
  ) -> ExactBatchChildReceiptV8:
    if not candidates or len(candidates) > 16 or len(candidates) != len(call_nonces):
      raise ValueError("V8 exact batch cardinality differs")
    if any(type(row) is not ManifoldCandidateV4 for row in candidates):
      raise TypeError("V8 exact batch requires V4 candidates")
    identities = (batch_nonce, *map(str, call_nonces))
    if any(not row or row in self._seen_nonces for row in identities) or (
        len(set(identities)) != len(identities)
    ):
      raise ValueError("V8 exact batch nonce was reused")
    self._seen_nonces.update(identities)
    arguments = {
        "operation": "candidate_exact_batch",
        "step_path_a": self.step_path_a, "step_path_b": self.step_path_b,
        "step_sha256_a": self.step_sha256_a, "step_sha256_b": self.step_sha256_b,
        "raw_face_index_a": self.raw_face_index_a,
        "raw_face_index_b": self.raw_face_index_b,
        "face_signature_a": self.face_signature_a,
        "face_signature_b": self.face_signature_b,
        "world_a_row_major": self.world_a_row_major,
        "plane_frame_a": self.plane_frame_a, "plane_origin_a": self.plane_origin_a,
        "batch_nonce": batch_nonce,
        "candidates": [
            {"rank_zero_based": index, "candidate_key": candidate.candidate_key,
             "call_nonce": str(call_nonces[index]),
             "child_world_row_major": candidate.child_world_row_major}
            for index, candidate in enumerate(candidates)
        ],
    }
    unsigned = {"schema_version": self.WORKER_REQUEST_SCHEMA, "arguments": arguments}
    request_sha = canonical_sha256(unsigned)
    request = {**unsigned, "request_payload_sha256": request_sha}
    worker = Path(__file__).resolve().parent / "tools" / self.WORKER_FILENAME
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
    started_at = time.monotonic()
    process: subprocess.Popen[str] | None = None

    def finish(
        status: str, *, child_started: bool, issuer: str,
        candidate_receipts: Sequence[ExactBatchCandidateReceiptV8],
        result: Mapping[str, Any], cleanup_elapsed: float = 0.0,
    ) -> ExactBatchChildReceiptV8:
      reaped = not child_started or (process is not None and process.poll() is not None)
      normalized = dict(result)
      normalized.setdefault("unreceipted_exact_candidates", 0)
      normalized.setdefault("unreceipted_occ_calls", 0)
      normalized.setdefault(
          "occ_calls_reserved",
          2 * len(candidate_receipts) + int(normalized["unreceipted_occ_calls"]),
      )
      normalized.update({
          "decision_elapsed_seconds": max(0.0, time.monotonic() - started_at),
          "cleanup_window_seconds": EXACT_CLEANUP_WINDOW_SECONDS_V6,
          "cleanup_elapsed_seconds": cleanup_elapsed,
          "child_reaped": reaped, "unreaped": bool(child_started and not reaped),
      })
      return issue_exact_batch_child_receipt_v8(
          batch_nonce=batch_nonce, request_payload_sha256=request_sha,
          terminal_status=status, child_started=child_started, issuer=issuer,
          candidate_receipts=candidate_receipts, result=normalized,
      )

    if time.monotonic() >= absolute_deadline:
      return finish(
          "timeout", child_started=False, issuer="parent_observer",
          candidate_receipts=(), result={"error_sha256": hashlib.sha256(
              b"deadline_expired_before_spawn"
          ).hexdigest()},
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
      try:
        stdout, stderr = process.communicate(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            timeout=max(0.0, absolute_deadline - time.monotonic()),
        )
      except Exception as error:
        timeout_output = getattr(error, "output", "") or ""
        if isinstance(timeout_output, bytes):
          timeout_output = timeout_output.decode("utf-8", errors="replace")
        decision_at = time.monotonic()
        try:
          process.kill()
        except Exception:
          pass
        cleanup_started = time.monotonic()
        try:
          killed_stdout, stderr = process.communicate(timeout=EXACT_CLEANUP_WINDOW_SECONDS_V6)
        except Exception:
          killed_stdout, stderr = "", ""
        cleanup_elapsed = time.monotonic() - cleanup_started
        stdout = killed_stdout or timeout_output
        try:
          events, _terminal, parse_error = _parse_worker_stdout_v8(
              stdout, request_payload_sha256=request_sha, batch_nonce=batch_nonce,
              worker_terminal_schema=self.WORKER_TERMINAL_SCHEMA,
          )
        except Exception:
          events, parse_error = (), "worker_prefix_parser_internal_error"
        inflight = int(len(events) < len(candidates))
        return finish(
            "timeout" if isinstance(error, subprocess.TimeoutExpired) else "kernel_error",
            child_started=True, issuer="parent_observer", candidate_receipts=events,
            cleanup_elapsed=cleanup_elapsed,
            result={
                "error_sha256": hashlib.sha256(
                    f"communicate_failed:{type(error).__name__}".encode("utf-8")
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                "terminal_decision_elapsed_seconds": decision_at - started_at,
                "unassessed_candidate_count": len(candidates) - len(events),
                "worker_stream_parse_error": parse_error,
                "unreceipted_exact_candidates": inflight,
                "unreceipted_occ_calls": 2 * inflight,
                "occ_calls_reserved": 2 * (len(events) + inflight),
            },
        )
      events, terminal, parse_error = _parse_worker_stdout_v8(
          stdout, request_payload_sha256=request_sha, batch_nonce=batch_nonce,
          worker_terminal_schema=self.WORKER_TERMINAL_SCHEMA,
      )
      if parse_error is not None:
        inflight = int(len(events) < len(candidates))
        return finish(
            "kernel_error", child_started=True, issuer="parent_observer",
            candidate_receipts=events,
            result={
                "error_sha256": hashlib.sha256(
                    b"worker_stream_parse_error"
                ).hexdigest(),
                "worker_stream_parse_error": parse_error,
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                "unassessed_candidate_count": len(candidates) - len(events),
                "unreceipted_exact_candidates": inflight,
                "unreceipted_occ_calls": 2 * inflight,
                "occ_calls_reserved": 2 * (len(events) + inflight),
            },
        )
      if time.monotonic() > absolute_deadline:
        inflight = int(len(events) < len(candidates))
        return finish(
            "timeout", child_started=True, issuer="parent_observer",
            candidate_receipts=events,
            result={
                "error_sha256": hashlib.sha256(
                    b"deadline_expired_before_terminal_validation"
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                "unassessed_candidate_count": len(candidates) - len(events),
                "unreceipted_exact_candidates": inflight,
                "unreceipted_occ_calls": 2 * inflight,
                "occ_calls_reserved": 2 * (len(events) + inflight),
            },
        )
      terminal_valid = False
      terminal_hash = None
      if terminal is not None:
        terminal_unsigned = dict(terminal)
        terminal_hash = terminal_unsigned.pop("terminal_payload_sha256", None)
        terminal_valid = (
            terminal.get("request_payload_sha256") == request_sha
            and terminal.get("batch_nonce") == batch_nonce
            and terminal_hash == canonical_sha256(terminal_unsigned)
            and terminal.get("candidate_event_payload_sha256s")
            == [row.event_payload_sha256 for row in events]
            and int(terminal.get("evaluated_candidate_count", -1)) == len(events)
            and int(terminal.get("unassessed_candidate_count", -1))
            == len(candidates) - len(events)
            and int(terminal.get("occ_calls_reserved", -1)) == 2 * len(events)
            and int(terminal.get("step_load_count", -1)) == 2
            and int(terminal.get("process_launch_count", -1)) == 1
        )
      if not terminal_valid:
        inflight = int(len(events) < len(candidates))
        return finish(
            "kernel_error", child_started=True, issuer="parent_observer",
            candidate_receipts=events,
            result={"error_sha256": hashlib.sha256(b"worker_terminal_invalid").hexdigest(),
                    "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                    "unreceipted_exact_candidates": inflight,
                    "unreceipted_occ_calls": 2 * inflight,
                    "occ_calls_reserved": 2 * (len(events) + inflight)},
        )
      status = str(terminal.get("terminal_status", "kernel_error"))
      if status not in {"ok", "kernel_error"}:
        status = "kernel_error"
      return finish(
          status, child_started=True, issuer="child_process_echo",
          candidate_receipts=events,
          result={
              "stop_reason": str(terminal.get("stop_reason", "invalid")),
              "evaluated_candidate_count": int(
                  terminal.get("evaluated_candidate_count", -1)
              ),
              "unassessed_candidate_count": int(
                  terminal.get("unassessed_candidate_count", -1)
              ),
              "occ_calls_reserved": int(terminal.get("occ_calls_reserved", -1)),
              "step_load_count": int(terminal.get("step_load_count", -1)),
              "process_launch_count": int(terminal.get("process_launch_count", -1)),
              "worker_terminal_payload_sha256": terminal_hash,
              "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
              "unreceipted_exact_candidates": 0, "unreceipted_occ_calls": 0,
          },
      )
    except Exception as error:
      inflight = int(process is not None and bool(candidates))
      return finish(
          "kernel_error", child_started=process is not None, issuer="parent_observer",
          candidate_receipts=(), result={"error_sha256": hashlib.sha256(
              f"batch_executor:{type(error).__name__}:{error}".encode("utf-8")
          ).hexdigest(), "unreceipted_exact_candidates": inflight,
              "unreceipted_occ_calls": 2 * inflight,
              "occ_calls_reserved": 2 * inflight},
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


__all__ = [
    "WORKER_REQUEST_SCHEMA_VERSION_V8", "WORKER_TERMINAL_SCHEMA_VERSION_V8",
    "OccExactManifoldExecutorV8", "execute_occ_exact_batch_v8",
]
