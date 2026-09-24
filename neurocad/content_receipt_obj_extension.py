"""Direct archive-byte authority for restored Fusion OBJ members.

The legacy dataset content receipt remains immutable and keeps its existing
STEP/JSON semantics.  This additive v2 extension proves that selected OBJ
files are real members of the authenticated archive by hashing bytes streamed
directly from 7-Zip, then requiring byte identity with the restored OBJ.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import zlib

from .benchmark_v2_training_provenance import canonical_sha256


CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_VERSION = (
    "fusion_content_receipt_obj_extension.v2"
)
FORMAL_CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_ALLOWLIST: frozenset[str] = frozenset(
    {CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_VERSION}
)
CONTENT_RECEIPT_OBJ_EXTENSION_TIMEOUT_SECONDS = 900
_CONTENT_RECEIPT_NAME = ".neurocad_content_receipt.json"
_ROOT = Path(__file__).resolve().parent
_SCHEMA_PATH = _ROOT / "configs" / "fusion_content_receipt_obj_extension_v2.schema.json"
_RESTORE_TOOL_PATH = _ROOT / "tools" / "restore_fusion_assembly_dataset.py"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CRC32_RE = re.compile(r"^[0-9a-f]{8}$")
_ARCHIVE_RE = re.compile(r"^a[0-9]+\.[0-9]+\.[0-9]+_[0-9]{2}\.7z$")
_FACTORY_TOKEN = object()


class ContentReceiptObjExtensionError(RuntimeError):
  """Raised when direct archive/restored OBJ provenance cannot close."""


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key: {key}")
    result[key] = value
  return result


def _strict_json(raw: bytes, *, label: str) -> Any:
  try:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON constant: {value}")
        ),
    )
  except (UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise ContentReceiptObjExtensionError(
        f"{label} is not duplicate-key-safe UTF-8 JSON"
    ) from error


def _is_reparse(path: Path) -> bool:
  try:
    return bool(int(getattr(path.lstat(), "st_file_attributes", 0)) & 0x400)
  except OSError:
    return False


def _plain_directory(path: str | Path, *, label: str) -> Path:
  candidate = Path(path).expanduser()
  try:
    if candidate.is_symlink() or _is_reparse(candidate) or not candidate.is_dir():
      raise ContentReceiptObjExtensionError(f"{label} must be a plain directory")
    return candidate.resolve(strict=True)
  except ContentReceiptObjExtensionError:
    raise
  except OSError as error:
    raise ContentReceiptObjExtensionError(f"{label} is missing") from error


def _identity(stat: os.stat_result) -> tuple[int, ...]:
  return (
      int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
      int(stat.st_mtime_ns), int(stat.st_nlink),
  )


@dataclass(frozen=True, slots=True)
class _CapturedFile:
  path: Path
  resolved_path: Path
  byte_count: int
  sha256: str
  identity: tuple[int, ...]

  def binding(self) -> dict[str, Any]:
    return {
        "path": str(self.resolved_path),
        "bytes": self.byte_count,
        "sha256": self.sha256,
    }

  def revalidate(self, *, label: str) -> None:
    current, _ = _capture_file(self.path, label=label, retain_bytes=False)
    if current != self:
      raise ContentReceiptObjExtensionError(f"{label} changed since capture")


def _capture_file(
    path: str | Path,
    *,
    label: str,
    retain_bytes: bool,
) -> tuple[_CapturedFile, bytes | None]:
  candidate = Path(os.path.abspath(Path(path)))
  try:
    if candidate.is_symlink() or _is_reparse(candidate) or not candidate.is_file():
      raise ContentReceiptObjExtensionError(f"{label} must be a plain file")
    resolved = candidate.resolve(strict=True)
    before = candidate.stat()
    if int(before.st_nlink) != 1:
      raise ContentReceiptObjExtensionError(f"{label} must not be hard-linked")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_bytes else None
    with candidate.open("rb") as stream:
      opened_before = os.fstat(stream.fileno())
      byte_count = 0
      while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
        if chunks is not None:
          chunks.append(chunk)
      opened_after = os.fstat(stream.fileno())
    after = candidate.stat()
    if (
        any(_identity(value) != _identity(before) for value in (opened_before, opened_after, after))
        or candidate.is_symlink()
        or _is_reparse(candidate)
        or candidate.resolve(strict=True) != resolved
        or byte_count != int(after.st_size)
    ):
      raise ContentReceiptObjExtensionError(f"{label} changed during capture")
  except ContentReceiptObjExtensionError:
    raise
  except OSError as error:
    raise ContentReceiptObjExtensionError(f"{label} is missing or unreadable") from error
  captured = _CapturedFile(
      path=candidate,
      resolved_path=resolved,
      byte_count=byte_count,
      sha256=digest.hexdigest(),
      identity=_identity(after),
  )
  return captured, (b"".join(chunks) if chunks is not None else None)


def _capture_json(path: str | Path, *, label: str) -> tuple[_CapturedFile, Any]:
  captured, raw = _capture_file(path, label=label, retain_bytes=True)
  return captured, _strict_json(raw or b"", label=label)


def _safe_archive(value: Any) -> str:
  if not isinstance(value, str) or _ARCHIVE_RE.fullmatch(value) is None:
    raise ContentReceiptObjExtensionError("source archive identity is malformed")
  return value


def _safe_member(root: Path, value: Any) -> tuple[str, Path]:
  if not isinstance(value, str) or not value or "\\" in value:
    raise ContentReceiptObjExtensionError("OBJ archive member path is unsafe")
  relative = PurePosixPath(value)
  if (
      relative.is_absolute()
      or relative.as_posix() != value
      or relative.suffix.lower() != ".obj"
      or any(part in {"", ".", ".."} for part in relative.parts)
  ):
    raise ContentReceiptObjExtensionError("OBJ archive member path is unsafe")
  try:
    path = root.joinpath(*relative.parts).resolve(strict=True)
    if path.relative_to(root).as_posix() != value:
      raise ValueError
  except (OSError, ValueError) as error:
    raise ContentReceiptObjExtensionError(
        f"restored OBJ is missing or unsafe: {value}"
    ) from error
  return value, path


def _list_archive_members(
    archive: Path,
    seven_zip: Path,
    timeout_seconds: int,
) -> tuple[dict[str, object], ...]:
  from .tools.restore_fusion_assembly_dataset import (
      _list_7z_archive_file_members,
  )
  return _list_7z_archive_file_members(archive, seven_zip, timeout_seconds)


def _capture_archive_member_bytes(
    archive: Path,
    member: str,
    seven_zip: Path,
    *,
    deadline: float,
) -> dict[str, object]:
  remaining = deadline - time.monotonic()
  if not math.isfinite(remaining) or remaining <= 0.0:
    raise ContentReceiptObjExtensionError("direct archive OBJ replay timed out")
  with tempfile.TemporaryDirectory(prefix="neurocad-archive-obj-") as raw_temp:
    temporary = Path(raw_temp)
    output = temporary / "member.bin"
    stderr = temporary / "7z.stderr"
    try:
      with output.open("xb") as stdout_stream, stderr.open("xb") as stderr_stream:
        completed = subprocess.run(
            [
                str(seven_zip), "x", "-so", "-bd", "-bb0", "-y",
                str(archive), member,
            ],
            cwd=str(archive.parent),
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
      raise ContentReceiptObjExtensionError(
          f"direct archive OBJ replay timed out: {member}"
      ) from error
    if completed.returncode != 0:
      detail = stderr.read_text(encoding="utf-8", errors="replace")[-2000:]
      raise ContentReceiptObjExtensionError(
          f"direct archive OBJ replay failed: {member}: {detail}"
      )
    digest = hashlib.sha256()
    checksum = 0
    byte_count = 0
    with output.open("rb") as stream:
      while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
        checksum = zlib.crc32(chunk, checksum)
        byte_count += len(chunk)
  return {
      "path": member,
      "bytes": byte_count,
      "sha256": digest.hexdigest(),
      "crc32": f"{checksum & 0xFFFFFFFF:08x}",
  }


def _implementation_binding() -> dict[str, Any]:
  rows = {}
  for name, path in (
      ("neurocad.content_receipt_obj_extension", Path(__file__)),
      ("neurocad.tools.restore_fusion_assembly_dataset", _RESTORE_TOOL_PATH),
  ):
    captured, _ = _capture_file(path, label=f"implementation {name}", retain_bytes=False)
    rows[name] = captured.sha256
  schema, _ = _capture_file(_SCHEMA_PATH, label="OBJ extension schema", retain_bytes=False)
  return {"source_sha256s": rows, "schema_sha256": schema.sha256}


def _selected_domain(
    source: Any,
    *,
    requested_case_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], dict[tuple[str, str], list[dict[str, str]]]]:
  cases = source.get("cases") if isinstance(source, Mapping) else None
  if not isinstance(cases, list) or not cases:
    raise ContentReceiptObjExtensionError("private source cases are malformed")
  source_order: list[str] = []
  by_id: dict[str, Mapping[str, Any]] = {}
  for row in cases:
    case_id = row.get("case_id") if isinstance(row, Mapping) else None
    if not isinstance(case_id, str) or not case_id or case_id in by_id:
      raise ContentReceiptObjExtensionError("private source case identity is malformed")
    source_order.append(case_id)
    by_id[case_id] = row
  requested = list(requested_case_ids) if requested_case_ids is not None else source_order
  if (
      not requested
      or len(set(requested)) != len(requested)
      or any(case_id not in by_id for case_id in requested)
  ):
    raise ContentReceiptObjExtensionError("requested OBJ provenance case set is invalid")
  selected = tuple(case_id for case_id in source_order if case_id in set(requested))
  domain: dict[tuple[str, str], list[dict[str, str]]] = {}
  seen_parts: set[tuple[str, str]] = set()
  for case_id in selected:
    binding = by_id[case_id].get("source_receipt_binding")
    archive = _safe_archive(binding.get("archive") if isinstance(binding, Mapping) else None)
    steps = binding.get("body_steps") if isinstance(binding, Mapping) else None
    if not isinstance(steps, list) or not steps:
      raise ContentReceiptObjExtensionError("private source body STEP domain is malformed")
    for step in steps:
      part = step.get("part") if isinstance(step, Mapping) else None
      path = step.get("path") if isinstance(step, Mapping) else None
      body_uuid = step.get("body_uuid") if isinstance(step, Mapping) else None
      geometry_asset = step.get("geometry_asset") if isinstance(step, Mapping) else None
      if (
          not isinstance(part, str) or not part
          or (case_id, part) in seen_parts
          or not isinstance(path, str)
          or PurePosixPath(path).suffix.lower() != ".step"
          or not isinstance(body_uuid, str) or not body_uuid
          or not isinstance(geometry_asset, str) or not geometry_asset
      ):
        raise ContentReceiptObjExtensionError("private source body identity is malformed")
      seen_parts.add((case_id, part))
      obj_path = PurePosixPath(path).with_suffix(".obj").as_posix()
      domain.setdefault((archive, obj_path), []).append(
          {
              "case_id": case_id,
              "part": part,
              "body_uuid": body_uuid,
              "geometry_asset": geometry_asset,
          }
      )
  return selected, domain


def _base_receipt(
    *,
    archive: str,
    volume: Path,
    source_cases: Sequence[Mapping[str, Any]],
    archive_capture: _CapturedFile,
) -> tuple[_CapturedFile, dict[str, Any]]:
  receipt_capture, payload = _capture_json(
      volume / _CONTENT_RECEIPT_NAME,
      label=f"base content receipt {archive}",
  )
  expected_hashes = {
      str(row["source_receipt_binding"].get("receipt_sha256"))
      for row in source_cases
      if row["source_receipt_binding"].get("archive") == archive
  }
  archive_row = payload.get("archive") if isinstance(payload, Mapping) else None
  provenance = payload.get("integrity_provenance") if isinstance(payload, Mapping) else None
  if (
      len(expected_hashes) != 1
      or receipt_capture.sha256 not in expected_hashes
      or payload.get("schema_version") not in {1, 2}
      or payload.get("dataset_version") != "a1.0.0"
      or not isinstance(provenance, Mapping)
      or provenance.get("official") is not True
      or payload.get("archive_member_metadata_authenticated") is not True
      or not isinstance(archive_row, Mapping)
      or archive_row.get("name") != archive
      or archive_row.get("bytes") != archive_capture.byte_count
      or archive_row.get("sha256") != archive_capture.sha256
  ):
    raise ContentReceiptObjExtensionError(
        f"base content receipt/archive authority differs: {archive}"
    )
  return receipt_capture, {
      **receipt_capture.binding(),
      "schema_version": payload["schema_version"],
      "legacy_member_domain_preserved": True,
  }


def _produce(
    *,
    private_source_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    case_ids: Sequence[str] | None,
    seven_zip_path: str | Path,
    timeout_seconds: int,
    pilot_only: bool,
) -> tuple[dict[str, Any], tuple[_CapturedFile, ...]]:
  if not pilot_only and (
      CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_VERSION
      not in FORMAL_CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_ALLOWLIST
  ):
    raise ContentReceiptObjExtensionError(
        "formal OBJ extension schema is not explicitly allowlisted"
    )
  if timeout_seconds != CONTENT_RECEIPT_OBJ_EXTENSION_TIMEOUT_SECONDS:
    raise ContentReceiptObjExtensionError("OBJ provenance timeout must remain 900 seconds")
  archives_root = _plain_directory(archive_root, label="archive root")
  restored = _plain_directory(restored_root, label="restored root")
  source_capture, source = _capture_json(private_source_path, label="private source")
  selected, domain = _selected_domain(source, requested_case_ids=case_ids)
  source_cases = [
      row for row in source["cases"] if row.get("case_id") in set(selected)
  ]
  seven_zip, _ = _capture_file(seven_zip_path, label="7-Zip executable", retain_bytes=False)
  deadline = time.monotonic() + timeout_seconds
  captures: list[_CapturedFile] = [source_capture, seven_zip]
  archive_rows: list[dict[str, Any]] = []
  receipt_rows: list[dict[str, Any]] = []
  member_rows: list[dict[str, Any]] = []
  by_archive: dict[str, list[str]] = {}
  for archive, member in domain:
    by_archive.setdefault(archive, []).append(member)
  for archive in sorted(by_archive):
    archive_path = archives_root / archive
    archive_capture, _ = _capture_file(
        archive_path,
        label=f"source archive {archive}",
        retain_bytes=False,
    )
    captures.append(archive_capture)
    volume = _plain_directory(restored / archive[:-3], label=f"restored volume {archive}")
    receipt_capture, receipt_binding = _base_receipt(
        archive=archive,
        volume=volume,
        source_cases=source_cases,
        archive_capture=archive_capture,
    )
    captures.append(receipt_capture)
    receipt_rows.append({"archive": archive, **receipt_binding})
    archive_rows.append({"archive": archive, **archive_capture.binding()})
    remaining = max(1, math.ceil(deadline - time.monotonic()))
    try:
      listing = _list_archive_members(archive_path, seven_zip.resolved_path, remaining)
    except Exception as error:
      raise ContentReceiptObjExtensionError(
          f"archive member listing failed: {archive}"
      ) from error
    metadata = {
        str(row.get("path")): row for row in listing if isinstance(row, Mapping)
    }
    for member in sorted(by_archive[archive], key=lambda value: value.encode("utf-8")):
      listed = metadata.get(member)
      if (
          not isinstance(listed, Mapping)
          or isinstance(listed.get("bytes"), bool)
          or not isinstance(listed.get("bytes"), int)
          or listed.get("bytes") < 0
          or not isinstance(listed.get("crc32"), str)
          or _CRC32_RE.fullmatch(str(listed.get("crc32"))) is None
      ):
        raise ContentReceiptObjExtensionError(
            f"OBJ is absent from authenticated archive listing: {archive}/{member}"
        )
      direct = _capture_archive_member_bytes(
          archive_path,
          member,
          seven_zip.resolved_path,
          deadline=deadline,
      )
      if (
          direct.get("path") != member
          or direct.get("bytes") != listed.get("bytes")
          or direct.get("crc32") != listed.get("crc32")
          or not isinstance(direct.get("sha256"), str)
          or _SHA256_RE.fullmatch(str(direct.get("sha256"))) is None
      ):
        raise ContentReceiptObjExtensionError(
            f"direct archive member metadata differs: {archive}/{member}"
        )
      _relative, restored_path = _safe_member(volume, member)
      restored_capture, _ = _capture_file(
          restored_path,
          label=f"restored OBJ {archive}/{member}",
          retain_bytes=False,
      )
      captures.append(restored_capture)
      if (
          restored_capture.byte_count != direct["bytes"]
          or restored_capture.sha256 != direct["sha256"]
      ):
        raise ContentReceiptObjExtensionError(
            f"restored OBJ differs from direct archive bytes: {archive}/{member}"
        )
      member_rows.append(
          {
              "archive": archive,
              "path": member,
              "bytes": direct["bytes"],
              "crc32": direct["crc32"],
              "sha256": direct["sha256"],
              "restored_path": str(restored_capture.resolved_path),
              "binding_mode": "direct_7z_member_bytes_plus_restored_byte_identity",
              "references": sorted(
                  copy.deepcopy(domain[(archive, member)]),
                  key=lambda row: (row["case_id"], row["part"]),
              ),
          }
      )
  payload = {
      "schema_version": CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_VERSION,
      "visibility": "private_evaluation_only",
      "formal": not pilot_only,
      "pilot_only": bool(pilot_only),
      "binding_mode": "direct_7z_member_bytes_plus_restored_byte_identity",
      "input_bindings": {
          "private_source": source_capture.binding(),
          "archive_root": str(archives_root),
          "restored_root": str(restored),
          "seven_zip": seven_zip.binding(),
      },
      "implementation": _implementation_binding(),
      "case_ids": list(selected),
      "base_content_receipts": receipt_rows,
      "archives": archive_rows,
      "members": member_rows,
      "summary": {
          "case_count": len(selected),
          "archive_count": len(archive_rows),
          "unique_obj_member_count": len(member_rows),
          "instance_reference_count": sum(len(row["references"]) for row in member_rows),
          "all_archive_members_directly_replayed": True,
          "all_restored_objs_byte_identical": True,
          "member_set_sha256": canonical_sha256(member_rows),
      },
  }
  for index, captured in enumerate(captures):
    captured.revalidate(label=f"OBJ provenance capture {index}")
  return payload, tuple(captures)


def _validate_payload(payload: Any, *, pilot_only: bool) -> None:
  if not isinstance(payload, Mapping):
    raise ContentReceiptObjExtensionError("OBJ extension receipt is not an object")
  required = {
      "schema_version", "visibility", "formal", "pilot_only", "binding_mode",
      "input_bindings", "implementation", "case_ids", "base_content_receipts",
      "archives", "members", "summary", "receipt_payload_sha256",
  }
  unsigned = copy.deepcopy(dict(payload))
  expected_hash = unsigned.pop("receipt_payload_sha256", None)
  if (
      set(payload) != required
      or payload.get("schema_version") != CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_VERSION
      or payload.get("visibility") != "private_evaluation_only"
      or payload.get("formal") is not (not pilot_only)
      or payload.get("pilot_only") is not bool(pilot_only)
      or payload.get("binding_mode")
      != "direct_7z_member_bytes_plus_restored_byte_identity"
      or not isinstance(expected_hash, str)
      or expected_hash != canonical_sha256(unsigned)
  ):
    raise ContentReceiptObjExtensionError("OBJ extension receipt contract differs")
  case_ids = payload.get("case_ids")
  members = payload.get("members")
  summary = payload.get("summary")
  if (
      not isinstance(case_ids, list) or not case_ids
      or len(set(case_ids)) != len(case_ids)
      or not isinstance(members, list) or not members
      or not isinstance(summary, Mapping)
      or summary.get("case_count") != len(case_ids)
      or summary.get("unique_obj_member_count") != len(members)
      or summary.get("member_set_sha256") != canonical_sha256(members)
      or summary.get("all_archive_members_directly_replayed") is not True
      or summary.get("all_restored_objs_byte_identical") is not True
      or payload.get("implementation") != _implementation_binding()
  ):
    raise ContentReceiptObjExtensionError("OBJ extension receipt evidence differs")
  keys: list[tuple[str, str]] = []
  for row in members:
    if not isinstance(row, Mapping):
      raise ContentReceiptObjExtensionError("OBJ extension member is malformed")
    key = (row.get("archive"), row.get("path"))
    if (
        not all(isinstance(value, str) for value in key)
        or not isinstance(row.get("bytes"), int)
        or isinstance(row.get("bytes"), bool)
        or row.get("bytes") < 0
        or _SHA256_RE.fullmatch(str(row.get("sha256") or "")) is None
        or _CRC32_RE.fullmatch(str(row.get("crc32") or "")) is None
        or row.get("binding_mode") != payload.get("binding_mode")
        or not isinstance(row.get("references"), list)
        or not row["references"]
    ):
      raise ContentReceiptObjExtensionError("OBJ extension member is malformed")
    keys.append(key)
  if keys != sorted(keys) or len(set(keys)) != len(keys):
    raise ContentReceiptObjExtensionError("OBJ extension member domain is not canonical")


def build_content_receipt_obj_extension_v2(
    *,
    private_source_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    case_ids: Sequence[str] | None = None,
    seven_zip_path: str | Path,
    timeout_seconds: int = CONTENT_RECEIPT_OBJ_EXTENSION_TIMEOUT_SECONDS,
    pilot_only: bool = True,
) -> dict[str, Any]:
  payload, _captures = _produce(
      private_source_path=private_source_path,
      archive_root=archive_root,
      restored_root=restored_root,
      case_ids=case_ids,
      seven_zip_path=seven_zip_path,
      timeout_seconds=timeout_seconds,
      pilot_only=pilot_only,
  )
  payload["receipt_payload_sha256"] = canonical_sha256(payload)
  _validate_payload(payload, pilot_only=pilot_only)
  return payload


class AuthenticatedContentReceiptObjExtension:
  """Non-serializable capability issued only after fresh archive-byte replay."""

  __slots__ = (
      "case_ids", "formal", "pilot_only", "member_set_sha256",
      "input_commitment_sha256", "receipt_payload_sha256", "_members",
      "_captures", "_authority_binding", "_sealed",
  )

  def __init__(
      self,
      *,
      payload: Mapping[str, Any],
      receipt: _CapturedFile,
      captures: Sequence[_CapturedFile],
      token: object,
  ):
    if token is not _FACTORY_TOKEN:
      raise TypeError("OBJ extension authority must be created by its verifier")
    object.__setattr__(self, "case_ids", tuple(payload["case_ids"]))
    object.__setattr__(self, "formal", bool(payload["formal"]))
    object.__setattr__(self, "pilot_only", bool(payload["pilot_only"]))
    object.__setattr__(
        self,
        "member_set_sha256",
        str(payload["summary"]["member_set_sha256"]),
    )
    input_commitment = canonical_sha256(
        {
            "case_ids": payload["case_ids"],
            "input_bindings": payload["input_bindings"],
            "base_content_receipts": payload["base_content_receipts"],
            "archives": payload["archives"],
        }
    )
    object.__setattr__(self, "input_commitment_sha256", input_commitment)
    object.__setattr__(
        self,
        "receipt_payload_sha256",
        str(payload["receipt_payload_sha256"]),
    )
    object.__setattr__(self, "_members", MappingProxyType({
        (row["archive"], row["path"]): MappingProxyType(copy.deepcopy(dict(row)))
        for row in payload["members"]
    }))
    object.__setattr__(self, "_captures", tuple(captures))
    object.__setattr__(
        self,
        "_authority_binding",
        MappingProxyType(
            {
                "schema_version": (
                    "fusion_content_receipt_obj_extension_authority_binding.v1"
                ),
                "extension_schema_version": payload["schema_version"],
                "formal": bool(payload["formal"]),
                "pilot_only": bool(payload["pilot_only"]),
                "case_ids": list(payload["case_ids"]),
                "receipt": receipt.binding(),
                "receipt_payload_sha256": payload["receipt_payload_sha256"],
                "input_bindings": copy.deepcopy(payload["input_bindings"]),
                "base_content_receipts": copy.deepcopy(
                    payload["base_content_receipts"]
                ),
                "archives": copy.deepcopy(payload["archives"]),
                "member_set_sha256": payload["summary"]["member_set_sha256"],
                "input_commitment_sha256": input_commitment,
                "unique_obj_member_count": payload["summary"][
                    "unique_obj_member_count"
                ],
                "instance_reference_count": payload["summary"][
                    "instance_reference_count"
                ],
            }
        ),
    )
    object.__setattr__(self, "_sealed", True)

  def __setattr__(self, _name: str, _value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("OBJ extension authority is immutable")
    object.__setattr__(self, _name, _value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("OBJ extension authority is not serializable")

  def revalidate(self) -> None:
    for index, captured in enumerate(self._captures):
      captured.revalidate(label=f"authenticated OBJ provenance capture {index}")

  def authenticated_snapshot(
      self,
  ) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    self.revalidate()
    return (
        copy.deepcopy(dict(self._authority_binding)),
        {
            key: copy.deepcopy(dict(value))
            for key, value in self._members.items()
        },
    )

  def require_member(
      self,
      archive: str,
      path: str,
      *,
      byte_count: int | None = None,
      sha256: str | None = None,
  ) -> dict[str, Any]:
    _binding, members = self.authenticated_snapshot()
    try:
      member = members[(archive, path)]
    except KeyError as error:
      raise ContentReceiptObjExtensionError("OBJ member is outside authenticated domain") from error
    if (
        byte_count is not None and member["bytes"] != byte_count
    ) or (
        sha256 is not None and member["sha256"] != sha256
    ):
      raise ContentReceiptObjExtensionError(
          "OBJ member bytes differ from authenticated domain"
      )
    return member


def verify_content_receipt_obj_extension_v2(
    receipt_path: str | Path,
    *,
    private_source_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    case_ids: Sequence[str] | None = None,
    seven_zip_path: str | Path,
    timeout_seconds: int = CONTENT_RECEIPT_OBJ_EXTENSION_TIMEOUT_SECONDS,
    pilot_only: bool = True,
) -> AuthenticatedContentReceiptObjExtension:
  receipt_capture, stored = _capture_json(receipt_path, label="OBJ extension receipt")
  _validate_payload(stored, pilot_only=pilot_only)
  fresh, captures = _produce(
      private_source_path=private_source_path,
      archive_root=archive_root,
      restored_root=restored_root,
      case_ids=case_ids,
      seven_zip_path=seven_zip_path,
      timeout_seconds=timeout_seconds,
      pilot_only=pilot_only,
  )
  fresh["receipt_payload_sha256"] = canonical_sha256(fresh)
  if dict(stored) != fresh:
    raise ContentReceiptObjExtensionError(
        "stored OBJ extension differs from fresh archive-byte replay"
    )
  receipt_capture.revalidate(label="OBJ extension receipt")
  return AuthenticatedContentReceiptObjExtension(
      payload=fresh,
      receipt=receipt_capture,
      captures=(*captures, receipt_capture),
      token=_FACTORY_TOKEN,
  )


__all__ = [
    "AuthenticatedContentReceiptObjExtension",
    "CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_VERSION",
    "CONTENT_RECEIPT_OBJ_EXTENSION_TIMEOUT_SECONDS",
    "FORMAL_CONTENT_RECEIPT_OBJ_EXTENSION_SCHEMA_ALLOWLIST",
    "ContentReceiptObjExtensionError",
    "build_content_receipt_obj_extension_v2",
    "verify_content_receipt_obj_extension_v2",
]
