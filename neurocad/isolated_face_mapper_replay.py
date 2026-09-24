"""Fresh-process face-mapper replay for pilot/formal geometry authority.

The development mapper receipt is never accepted as face identity authority by
this module.  Every call creates a private temporary output, launches the
existing audit CLI in a new process, parses the live result into memory, and
deletes the output.  The serializable manifest is provenance only; callers
must use :meth:`IsolatedFaceMapperReplay.endpoint_lookup` from the live run.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .benchmark_v2_training_provenance import canonical_sha256
from .fusion_face_mapper import (
    FACE_MAP_AUDIT_SCHEMA_VERSION,
    MAPPER_VERSION,
    MappingTolerances,
    _implementation_source_sha256s,
    build_mapped_endpoint_lookup,
)


ISOLATED_REPLAY_SCHEMA_VERSION = "fusion_face_mapper_isolated_replay.v1"
ISOLATED_REPLAY_TIMEOUT_SECONDS = 900
_SHA256_CHARS = frozenset("0123456789abcdef")
_REPARSE_ATTRIBUTE = 0x400
_LOCKED_ENVIRONMENT = MappingProxyType({
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
})


class IsolatedFaceMapperReplayError(RuntimeError):
  """Raised when a fresh mapper replay cannot be authenticated."""


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key: {key}")
    result[key] = value
  return result


def _reject_constant(value: str) -> None:
  raise ValueError(f"nonstandard JSON constant: {value}")


def _strict_json(raw: bytes, *, label: str) -> Any:
  try:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
  except (UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise IsolatedFaceMapperReplayError(
        f"{label} is not duplicate-key-safe UTF-8 JSON"
    ) from error


def _is_sha256(value: Any) -> bool:
  return (
      isinstance(value, str)
      and len(value) == 64
      and all(character in _SHA256_CHARS for character in value)
  )


def _is_sha1(value: Any) -> bool:
  return (
      isinstance(value, str)
      and len(value) == 40
      and all(character in _SHA256_CHARS for character in value)
  )


def _is_reparse(path: Path) -> bool:
  try:
    return bool(int(getattr(path.lstat(), "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE)
  except OSError:
    return False


def _plain_directory(path: str | Path, *, label: str) -> Path:
  candidate = Path(path).expanduser()
  try:
    if candidate.is_symlink() or _is_reparse(candidate) or not candidate.is_dir():
      raise IsolatedFaceMapperReplayError(f"{label} must be a plain directory")
    return candidate.resolve(strict=True)
  except IsolatedFaceMapperReplayError:
    raise
  except (OSError, RuntimeError) as error:
    raise IsolatedFaceMapperReplayError(f"{label} is missing") from error


@dataclass(frozen=True, slots=True)
class _CapturedFile:
  path: Path
  resolved_path: Path
  sha256: str
  byte_count: int
  identity: tuple[int, ...]
  allow_hardlinks: bool = False

  def binding(self) -> dict[str, Any]:
    return {
        "path": str(self.resolved_path),
        "bytes": self.byte_count,
        "sha256": self.sha256,
    }

  def revalidate(self, *, label: str) -> None:
    current, _ = _capture_file(
        self.path,
        label=label,
        retain_bytes=False,
        allow_hardlinks=self.allow_hardlinks,
    )
    if current != self:
      raise IsolatedFaceMapperReplayError(f"{label} changed since capture")


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
  return (
      int(value.st_dev),
      int(value.st_ino),
      int(value.st_size),
      int(value.st_mtime_ns),
      int(value.st_nlink),
  )


def _capture_file(
    path: str | Path,
    *,
    label: str,
    retain_bytes: bool,
    allow_hardlinks: bool = False,
) -> tuple[_CapturedFile, bytes | None]:
  candidate = Path(os.path.abspath(Path(path)))
  try:
    if candidate.is_symlink() or _is_reparse(candidate) or not candidate.is_file():
      raise IsolatedFaceMapperReplayError(f"{label} must be a plain file")
    resolved = candidate.resolve(strict=True)
    before = candidate.stat()
    if not allow_hardlinks and int(before.st_nlink) != 1:
      raise IsolatedFaceMapperReplayError(f"{label} must not be hard-linked")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_bytes else None
    byte_count = 0
    with candidate.open("rb") as stream:
      opened_before = os.fstat(stream.fileno())
      while True:
        chunk = stream.read(8 * 1024 * 1024)
        if not chunk:
          break
        digest.update(chunk)
        byte_count += len(chunk)
        if chunks is not None:
          chunks.append(chunk)
      opened_after = os.fstat(stream.fileno())
    after = candidate.stat()
    identities = tuple(
        _stat_identity(value)
        for value in (before, opened_before, opened_after, after)
    )
    if (
        any(value != identities[0] for value in identities[1:])
        or candidate.is_symlink()
        or _is_reparse(candidate)
        or candidate.resolve(strict=True) != resolved
        or byte_count != int(after.st_size)
        or (
            not allow_hardlinks
            and any(int(value.st_nlink) != 1 for value in (opened_before, opened_after, after))
        )
    ):
      raise IsolatedFaceMapperReplayError(f"{label} changed during capture")
  except IsolatedFaceMapperReplayError:
    raise
  except OSError as error:
    raise IsolatedFaceMapperReplayError(f"{label} is missing or unreadable") from error
  captured = _CapturedFile(
      path=candidate,
      resolved_path=resolved,
      sha256=digest.hexdigest(),
      byte_count=byte_count,
      identity=_stat_identity(after),
      allow_hardlinks=allow_hardlinks,
  )
  return captured, (b"".join(chunks) if chunks is not None else None)


def _capture_json(path: str | Path, *, label: str) -> tuple[_CapturedFile, Any]:
  captured, raw = _capture_file(path, label=label, retain_bytes=True)
  return captured, _strict_json(raw or b"", label=label)


def _implementation_paths() -> dict[str, Path]:
  root = Path(__file__).resolve().parent
  return {
      "neurocad.isolated_face_mapper_replay": Path(__file__),
      "neurocad.fusion_face_mapper": root / "fusion_face_mapper.py",
      "neurocad.source_obj_analytic_support_v1": root / "source_obj_analytic_support_v1.py",
      "neurocad.analytic_trim_domain_v1": root / "analytic_trim_domain_v1.py",
      "neurocad.tools.audit_fusion_step_face_map": root / "tools" / "audit_fusion_step_face_map.py",
      "neurocad.cadquery_backend": root / "cadquery_backend.py",
      "neurocad.tools.restore_fusion_assembly_dataset": root / "tools" / "restore_fusion_assembly_dataset.py",
  }


def _capture_implementation() -> tuple[dict[str, Any], list[_CapturedFile]]:
  captures: list[_CapturedFile] = []
  hashes: dict[str, str] = {}
  for name, path in sorted(_implementation_paths().items()):
    captured, _ = _capture_file(path, label=f"isolated replay implementation {name}", retain_bytes=False)
    captures.append(captured)
    hashes[name] = captured.sha256
  return {"source_sha256s": hashes}, captures


def _capture_python() -> tuple[dict[str, Any], _CapturedFile]:
  captured, _ = _capture_file(
      sys.executable,
      label="isolated replay Python executable",
      retain_bytes=False,
      allow_hardlinks=True,
  )
  return {
      **captured.binding(),
      "version": sys.version,
  }, captured


def _safe_member(root: Path, relative: Any, *, label: str) -> Path:
  if not isinstance(relative, str) or not relative or "\\" in relative:
    raise IsolatedFaceMapperReplayError(f"{label} relative path is unsafe")
  posix = PurePosixPath(relative)
  if posix.is_absolute() or posix.as_posix() != relative or any(
      part in {"", ".", ".."} for part in posix.parts
  ):
    raise IsolatedFaceMapperReplayError(f"{label} relative path is unsafe")
  try:
    result = root.joinpath(*posix.parts).resolve(strict=True)
    result.relative_to(root)
  except (OSError, ValueError) as error:
    raise IsolatedFaceMapperReplayError(f"{label} escapes or is missing") from error
  return result


def _selected_source_material(
    family: Any,
    *,
    case_ids: Sequence[str],
    dataset_root: Path,
) -> tuple[list[str], list[tuple[str, Path]]]:
  if not isinstance(family, Mapping) or family.get("schema_version") != 2:
    raise IsolatedFaceMapperReplayError("mapper family source schema differs")
  public = family.get("cases")
  private = family.get("private_source_bindings", {}).get("cases")
  if not isinstance(public, list) or not isinstance(private, list):
    raise IsolatedFaceMapperReplayError("mapper family source case domains are malformed")
  source_order = [
      row.get("id") if isinstance(row, Mapping) else None for row in public
  ]
  if not all(isinstance(value, str) and value for value in source_order) or len(
      set(source_order)
  ) != len(source_order):
    raise IsolatedFaceMapperReplayError("mapper family source case identities are malformed")
  requested = list(case_ids)
  if (
      not requested
      or not all(isinstance(value, str) and value for value in requested)
      or len(set(requested)) != len(requested)
      or not set(requested).issubset(set(source_order))
  ):
    raise IsolatedFaceMapperReplayError("isolated replay case set is invalid")
  private_by_id = {
      str(row.get("case_id")): row
      for row in private
      if isinstance(row, Mapping) and isinstance(row.get("case_id"), str)
  }
  if not set(requested).issubset(private_by_id):
    raise IsolatedFaceMapperReplayError("isolated replay private case coverage differs")
  selected_order = [value for value in source_order if value in set(requested)]
  members: list[tuple[str, Path]] = []
  for case_id in selected_order:
    row = private_by_id[case_id]
    binding = row.get("source_receipt_binding")
    archive = binding.get("archive") if isinstance(binding, Mapping) else None
    if not isinstance(archive, str) or not archive.endswith(".7z"):
      raise IsolatedFaceMapperReplayError("isolated replay archive identity is malformed")
    volume_root = _plain_directory(dataset_root / archive[:-3], label=f"case {case_id} volume root")
    members.append((f"case {case_id} content receipt", volume_root / ".neurocad_content_receipt.json"))
    assembly = binding.get("assembly_json")
    members.append(
        (
            f"case {case_id} assembly JSON",
            _safe_member(
                volume_root,
                assembly.get("path") if isinstance(assembly, Mapping) else None,
                label=f"case {case_id} assembly JSON",
            ),
        )
    )
    steps = binding.get("body_steps")
    if not isinstance(steps, list) or not steps:
      raise IsolatedFaceMapperReplayError("isolated replay STEP domain is malformed")
    # Repeated occurrences may legitimately reuse one body-local STEP.  The
    # family-source capture above retains every instance/transform/endpoint;
    # physical files are captured once, provided every occurrence declares the
    # same immutable content/body binding for that path.
    step_bindings_by_path: dict[str, tuple[Any, ...]] = {}
    for step in steps:
      if not isinstance(step, Mapping):
        raise IsolatedFaceMapperReplayError("isolated replay STEP identity is malformed")
      relative = step.get("path")
      byte_count = step.get("bytes")
      sha1 = step.get("sha1")
      sha256 = step.get("sha256")
      body_uuid = step.get("body_uuid")
      geometry_asset = step.get("geometry_asset")
      if (
          not isinstance(relative, str)
          or type(byte_count) is not int
          or byte_count <= 0
          or not _is_sha1(sha1)
          or not _is_sha256(sha256)
          or not isinstance(body_uuid, str)
          or not body_uuid
          or not isinstance(geometry_asset, str)
          or not geometry_asset
      ):
        raise IsolatedFaceMapperReplayError(
            "isolated replay STEP identity is malformed"
        )
      binding = (
          relative,
          byte_count,
          sha1,
          sha256,
          body_uuid,
          geometry_asset,
      )
      prior = step_bindings_by_path.get(relative)
      if prior is not None:
        if prior != binding:
          raise IsolatedFaceMapperReplayError(
              "isolated replay STEP path has a conflicting binding"
          )
        continue
      step_bindings_by_path[relative] = binding
      step_path = _safe_member(volume_root, relative, label=f"case {case_id} STEP")
      members.append((f"case {case_id} STEP {relative}", step_path))
      obj_relative = PurePosixPath(relative).with_suffix(".obj").as_posix()
      members.append(
          (
              f"case {case_id} OBJ {obj_relative}",
              _safe_member(volume_root, obj_relative, label=f"case {case_id} OBJ"),
          )
      )
  return selected_order, members


def _capture_inputs(
    *,
    family_source_path: str | Path,
    dataset_root: Path,
    case_ids: Sequence[str],
) -> tuple[dict[str, Any], Any, list[_CapturedFile], list[str]]:
  family_capture, family = _capture_json(family_source_path, label="isolated replay family source")
  source_order, members = _selected_source_material(
      family,
      case_ids=case_ids,
      dataset_root=dataset_root,
  )
  captures = [family_capture]
  rows: list[dict[str, Any]] = []
  seen: set[Path] = set()
  for role, path in members:
    resolved = path.resolve(strict=True)
    if resolved in seen:
      continue
    seen.add(resolved)
    captured, _ = _capture_file(path, label=role, retain_bytes=False)
    captures.append(captured)
    rows.append({"role": role, **captured.binding()})
  return {
      "family_source": family_capture.binding(),
      "dataset_root": str(dataset_root),
      "files": rows,
  }, family, captures, source_order


def _verify_receipt_hash(payload: Mapping[str, Any]) -> None:
  unsigned = dict(payload)
  observed = unsigned.pop("receipt_payload_sha256", None)
  if not _is_sha256(observed) or canonical_sha256(unsigned) != observed:
    raise IsolatedFaceMapperReplayError("live mapper receipt payload hash differs")


def _lookup_payload(lookup: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
  return [
      {"key": list(key), "mapping": copy.deepcopy(dict(value))}
      for key, value in sorted(lookup.items())
  ]


def _validate_live_receipt(
    payload: Any,
    *,
    family_capture: _CapturedFile,
    requested_case_ids: Sequence[str],
    source_order_case_ids: Sequence[str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
  if not isinstance(payload, Mapping):
    raise IsolatedFaceMapperReplayError("live mapper receipt is not an object")
  _verify_receipt_hash(payload)
  expected_source_hashes = _implementation_source_sha256s()
  implementation = payload.get("environment", {}).get("implementation")
  source = payload.get("source")
  if (
      payload.get("schema_version") != FACE_MAP_AUDIT_SCHEMA_VERSION
      or payload.get("mapper_version") != MAPPER_VERSION
      or payload.get("tolerances") != asdict(MappingTolerances())
      or not isinstance(implementation, Mapping)
      or implementation.get("module") != "neurocad.fusion_face_mapper"
      or implementation.get("version") != MAPPER_VERSION
      or implementation.get("source_sha256s") != expected_source_hashes
      or not isinstance(source, Mapping)
      or Path(str(source.get("family_source_path") or "")).resolve(strict=False)
      != family_capture.resolved_path
      or source.get("family_source_bytes") != family_capture.byte_count
      or source.get("family_source_sha256") != family_capture.sha256
  ):
    raise IsolatedFaceMapperReplayError(
        "live mapper implementation/tolerance/source binding differs"
    )
  cases = payload.get("cases")
  if not isinstance(cases, list) or [
      row.get("case_id") if isinstance(row, Mapping) else None for row in cases
  ] != list(source_order_case_ids):
    raise IsolatedFaceMapperReplayError("live mapper exact case order differs")
  for case in cases:
    if not isinstance(case, Mapping) or case.get("status") != "mapped_complete":
      raise IsolatedFaceMapperReplayError("live mapper case is not mapped_complete")
    summary = case.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("all_bodies_ready") is not True
        or summary.get("all_endpoints_mapped") is not True
        or summary.get("case_complete") is not True
        or summary.get("mapped_endpoint_count") != summary.get("endpoint_count")
    ):
      raise IsolatedFaceMapperReplayError("live mapper case summary does not close")
  summary = payload.get("summary")
  if (
      not isinstance(summary, Mapping)
      or summary.get("case_count") != len(requested_case_ids)
      or summary.get("all_cases_complete") is not True
      or summary.get("all_endpoints_mapped") is not True
  ):
    raise IsolatedFaceMapperReplayError("live mapper run summary does not close")
  try:
    lookup = build_mapped_endpoint_lookup(payload)
  except Exception as error:
    raise IsolatedFaceMapperReplayError("live mapper lookup construction failed") from error
  if {key[0] for key in lookup} != set(requested_case_ids):
    raise IsolatedFaceMapperReplayError("live mapper lookup case domain differs")
  return {key: copy.deepcopy(dict(value)) for key, value in lookup.items()}


def _kill_process_tree(process: subprocess.Popen[Any]) -> None:
  if process.poll() is not None:
    return
  try:
    if os.name == "nt":
      subprocess.run(
          ["taskkill", "/PID", str(process.pid), "/T", "/F"],
          check=False,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          timeout=30,
      )
    else:
      os.killpg(process.pid, 9)
  except Exception:
    process.kill()


@dataclass(frozen=True, slots=True)
class IsolatedFaceMapperReplay:
  manifest: Mapping[str, Any]
  lookup_sha256: str
  wall_time_seconds: float
  _lookup: Mapping[tuple[str, str, int], Mapping[str, Any]]
  _receipt_payload: Mapping[str, Any]

  def endpoint_lookup(self) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        key: copy.deepcopy(dict(value)) for key, value in self._lookup.items()
    }

  def receipt_payload(self) -> dict[str, Any]:
    """Return the ephemeral live receipt as a defensive in-memory copy."""

    return copy.deepcopy(dict(self._receipt_payload))


def _manifest_without_hash(value: Mapping[str, Any]) -> dict[str, Any]:
  unsigned = copy.deepcopy(dict(value))
  unsigned.pop("manifest_payload_sha256", None)
  return unsigned


def run_isolated_face_mapper_replay(
    *,
    family_source_path: str | Path,
    dataset_root: str | Path,
    case_ids: Sequence[str],
    phase: str,
    timeout_seconds: int = ISOLATED_REPLAY_TIMEOUT_SECONDS,
) -> IsolatedFaceMapperReplay:
  """Run one exact case set through a fresh, non-reused mapper process."""

  if phase not in {"builder", "verifier"}:
    raise IsolatedFaceMapperReplayError("isolated replay phase is invalid")
  if type(timeout_seconds) is not int or timeout_seconds != ISOLATED_REPLAY_TIMEOUT_SECONDS:
    raise IsolatedFaceMapperReplayError("isolated replay timeout must remain locked at 900 seconds")
  requested_case_ids = list(case_ids)
  root = Path(__file__).resolve().parent
  dataset = _plain_directory(dataset_root, label="isolated replay dataset root")
  implementation, implementation_captures = _capture_implementation()
  python, python_capture = _capture_python()
  inputs, _family, input_captures, source_order_case_ids = _capture_inputs(
      family_source_path=family_source_path,
      dataset_root=dataset,
      case_ids=requested_case_ids,
  )
  captures = [*implementation_captures, python_capture, *input_captures]
  environment = os.environ.copy()
  environment.update(_LOCKED_ENVIRONMENT)
  environment["PYTHONDONTWRITEBYTECODE"] = "1"
  started = time.monotonic()
  with tempfile.TemporaryDirectory(prefix="neurocad-face-replay-") as temporary_name:
    temporary = _plain_directory(temporary_name, label="isolated replay private temporary directory")
    output_path = temporary / "live_mapper_receipt.json"
    stdout_path = temporary / "stdout.log"
    stderr_path = temporary / "stderr.log"
    if output_path.exists():
      raise IsolatedFaceMapperReplayError("isolated replay output was unexpectedly reused")
    command = [
        sys.executable,
        "-B",
        "-u",
        str((root / "tools" / "audit_fusion_step_face_map.py").resolve(strict=True)),
        "--family-source",
        str(Path(family_source_path).resolve(strict=True)),
        "--archive-root",
        str(dataset),
        "--output",
        str(output_path),
    ]
    for case_id in requested_case_ids:
      command.extend(["--development-case-id", case_id])
    creationflags = 0
    process_kwargs: dict[str, Any] = {}
    if os.name == "nt":
      creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | int(
          getattr(subprocess, "CREATE_NO_WINDOW", 0)
      )
    else:
      process_kwargs["start_new_session"] = True
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
      process = subprocess.Popen(
          command,
          cwd=root,
          env=environment,
          stdout=stdout,
          stderr=stderr,
          creationflags=creationflags,
          **process_kwargs,
      )
      try:
        returncode = process.wait(timeout=timeout_seconds)
      except subprocess.TimeoutExpired as error:
        _kill_process_tree(process)
        try:
          process.wait(timeout=30)
        except Exception:
          pass
        raise IsolatedFaceMapperReplayError(
            "isolated mapper replay timed out and its process tree was killed"
        ) from error
    wall_time_seconds = time.monotonic() - started
    if returncode != 0:
      stderr_tail = stderr_path.read_bytes()[-8192:].decode("utf-8", errors="replace")
      raise IsolatedFaceMapperReplayError(
          f"isolated mapper replay returned {returncode}: {stderr_tail}"
      )
    output_capture, raw_output = _capture_file(
        output_path,
        label="isolated replay private output",
        retain_bytes=True,
    )
    stdout_capture, _ = _capture_file(stdout_path, label="isolated replay stdout", retain_bytes=False)
    stderr_capture, _ = _capture_file(stderr_path, label="isolated replay stderr", retain_bytes=False)
    payload = _strict_json(raw_output or b"", label="isolated replay private output")
    lookup = _validate_live_receipt(
        payload,
        family_capture=input_captures[0],
        requested_case_ids=requested_case_ids,
        source_order_case_ids=source_order_case_ids,
    )
    lookup_sha256 = canonical_sha256(_lookup_payload(lookup))
    ephemeral_output = {
        "retained": False,
        "private_temporary_output": True,
        "receipt": {
            "bytes": output_capture.byte_count,
            "sha256": output_capture.sha256,
            "receipt_payload_sha256": payload["receipt_payload_sha256"],
        },
        "stdout": {
            "bytes": stdout_capture.byte_count,
            "sha256": stdout_capture.sha256,
        },
        "stderr": {
            "bytes": stderr_capture.byte_count,
            "sha256": stderr_capture.sha256,
        },
    }
  for index, captured in enumerate(captures):
    captured.revalidate(label=f"isolated replay captured input {index}")
  manifest: dict[str, Any] = {
      "schema_version": ISOLATED_REPLAY_SCHEMA_VERSION,
      "formal": False,
      "development": True,
      "phase": phase,
      "fresh_private_output": True,
      "receipt_reuse": False,
      "returncode": 0,
      "timeout_seconds": timeout_seconds,
      "process_tree_kill_on_timeout": True,
      "environment": dict(_LOCKED_ENVIRONMENT),
      "tolerances": asdict(MappingTolerances()),
      "implementation": implementation,
      "python": python,
      "inputs": inputs,
      "case_ids": requested_case_ids,
      "source_order_case_ids": source_order_case_ids,
      "lookup_entry_count": len(lookup),
      "lookup_payload_sha256": lookup_sha256,
      "ephemeral_output": ephemeral_output,
  }
  manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
  validate_isolated_face_mapper_replay_manifest(
      manifest,
      expected_phase=phase,
      expected_case_ids=requested_case_ids,
      expected_lookup_sha256=lookup_sha256,
  )
  return IsolatedFaceMapperReplay(
      manifest=MappingProxyType(copy.deepcopy(manifest)),
      lookup_sha256=lookup_sha256,
      wall_time_seconds=wall_time_seconds,
      _lookup=MappingProxyType(copy.deepcopy(lookup)),
      _receipt_payload=MappingProxyType(copy.deepcopy(dict(payload))),
  )


def validate_isolated_face_mapper_replay_manifest(
    value: Any,
    *,
    expected_phase: str,
    expected_case_ids: Sequence[str],
    expected_lookup_sha256: str,
) -> None:
  """Fail closed on any replay provenance/configuration drift."""

  required = {
      "schema_version", "formal", "development", "phase",
      "fresh_private_output", "receipt_reuse", "returncode",
      "timeout_seconds", "process_tree_kill_on_timeout", "environment",
      "tolerances", "implementation", "python", "inputs", "case_ids",
      "source_order_case_ids", "lookup_entry_count", "lookup_payload_sha256",
      "ephemeral_output", "manifest_payload_sha256",
  }
  if not isinstance(value, Mapping) or set(value) != required:
    raise IsolatedFaceMapperReplayError("isolated replay manifest contract differs")
  if (
      value.get("schema_version") != ISOLATED_REPLAY_SCHEMA_VERSION
      or value.get("formal") is not False
      or value.get("development") is not True
      or value.get("phase") != expected_phase
      or value.get("fresh_private_output") is not True
      or value.get("receipt_reuse") is not False
      or type(value.get("returncode")) is not int
      or value.get("returncode") != 0
      or type(value.get("timeout_seconds")) is not int
      or value.get("timeout_seconds") != ISOLATED_REPLAY_TIMEOUT_SECONDS
      or value.get("process_tree_kill_on_timeout") is not True
      or value.get("environment") != dict(_LOCKED_ENVIRONMENT)
      or value.get("tolerances") != asdict(MappingTolerances())
      or value.get("case_ids") != list(expected_case_ids)
      or value.get("lookup_payload_sha256") != expected_lookup_sha256
      or not _is_sha256(value.get("manifest_payload_sha256"))
      or canonical_sha256(_manifest_without_hash(value))
      != value.get("manifest_payload_sha256")
  ):
    raise IsolatedFaceMapperReplayError("isolated replay manifest values differ")
  current_implementation, _ = _capture_implementation()
  current_python, _ = _capture_python()
  if value.get("implementation") != current_implementation or value.get("python") != current_python:
    raise IsolatedFaceMapperReplayError("isolated replay implementation/Python binding differs")
  inputs = value.get("inputs")
  if not isinstance(inputs, Mapping) or set(inputs) != {"family_source", "dataset_root", "files"}:
    raise IsolatedFaceMapperReplayError("isolated replay input manifest differs")
  family = inputs.get("family_source")
  dataset_raw = inputs.get("dataset_root")
  if not isinstance(family, Mapping) or not isinstance(dataset_raw, str):
    raise IsolatedFaceMapperReplayError("isolated replay family/dataset binding is malformed")
  try:
    family_capture, _ = _capture_json(str(family.get("path") or ""), label="manifest family source")
    dataset = _plain_directory(dataset_raw, label="manifest dataset root")
  except (OSError, RuntimeError) as error:
    raise IsolatedFaceMapperReplayError("isolated replay manifest inputs are missing") from error
  if family_capture.binding() != dict(family):
    raise IsolatedFaceMapperReplayError("isolated replay family source binding differs")
  expected_inputs, _family, _captures, expected_source_order = _capture_inputs(
      family_source_path=family_capture.path,
      dataset_root=dataset,
      case_ids=expected_case_ids,
  )
  if dict(inputs) != expected_inputs or value.get("source_order_case_ids") != expected_source_order:
    raise IsolatedFaceMapperReplayError("isolated replay exact input captures differ")
  if type(value.get("lookup_entry_count")) is not int or value.get("lookup_entry_count") <= 0:
    raise IsolatedFaceMapperReplayError("isolated replay lookup count is invalid")
  ephemeral = value.get("ephemeral_output")
  if not isinstance(ephemeral, Mapping) or set(ephemeral) != {
      "retained", "private_temporary_output", "receipt", "stdout", "stderr"
  } or ephemeral.get("retained") is not False or ephemeral.get(
      "private_temporary_output"
  ) is not True:
    raise IsolatedFaceMapperReplayError("isolated replay ephemeral output contract differs")
  receipt = ephemeral.get("receipt")
  if not isinstance(receipt, Mapping) or set(receipt) != {
      "bytes", "sha256", "receipt_payload_sha256"
  } or type(receipt.get("bytes")) is not int or receipt.get("bytes") <= 0 or not _is_sha256(
      receipt.get("sha256")
  ) or not _is_sha256(receipt.get("receipt_payload_sha256")):
    raise IsolatedFaceMapperReplayError("isolated replay output receipt binding is malformed")
  for stream_name in ("stdout", "stderr"):
    stream = ephemeral.get(stream_name)
    if not isinstance(stream, Mapping) or set(stream) != {"bytes", "sha256"} or type(
        stream.get("bytes")
    ) is not int or stream.get("bytes") < 0 or not _is_sha256(stream.get("sha256")):
      raise IsolatedFaceMapperReplayError("isolated replay stream binding is malformed")


def require_equivalent_isolated_replay(
    stored_builder_manifest: Any,
    live_verifier_replay: IsolatedFaceMapperReplay,
) -> None:
  """Require a stored builder provenance commitment to match fresh live faces."""

  case_ids = list(live_verifier_replay.manifest.get("case_ids", []))
  validate_isolated_face_mapper_replay_manifest(
      stored_builder_manifest,
      expected_phase="builder",
      expected_case_ids=case_ids,
      expected_lookup_sha256=live_verifier_replay.lookup_sha256,
  )
  validate_isolated_face_mapper_replay_manifest(
      live_verifier_replay.manifest,
      expected_phase="verifier",
      expected_case_ids=case_ids,
      expected_lookup_sha256=live_verifier_replay.lookup_sha256,
  )
  deterministic_fields = (
      "schema_version", "formal", "development", "fresh_private_output",
      "receipt_reuse", "returncode", "timeout_seconds",
      "process_tree_kill_on_timeout", "environment", "tolerances",
      "implementation", "python", "inputs", "case_ids",
      "source_order_case_ids", "lookup_entry_count", "lookup_payload_sha256",
  )
  if any(
      stored_builder_manifest.get(field) != live_verifier_replay.manifest.get(field)
      for field in deterministic_fields
  ):
    raise IsolatedFaceMapperReplayError(
        "stored builder replay does not match fresh verifier replay"
    )


__all__ = [
    "ISOLATED_REPLAY_SCHEMA_VERSION",
    "ISOLATED_REPLAY_TIMEOUT_SECONDS",
    "IsolatedFaceMapperReplay",
    "IsolatedFaceMapperReplayError",
    "run_isolated_face_mapper_replay",
    "validate_isolated_face_mapper_replay_manifest",
    "require_equivalent_isolated_replay",
]
