"""Fail-closed capture and routing for restored multi-archive datasets.

Archive member identity is the pair ``(archive, PurePosixPath(path))``.  The
archive string is retained in receipts even when two restored volumes contain
identical relative paths and identical bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata


_SHA256 = frozenset("0123456789abcdef")
_REPARSE_ATTRIBUTE = 0x400
_ARCHIVE_NAME = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9_-]*|a[0-9]+\.[0-9]+\.[0-9]+_[0-9]{2})\.7z",
    re.ASCII,
)
_MEMBER_SEGMENT = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
    re.ASCII,
)
_ROLE_SUFFIX = MappingProxyType({"assembly_json": ".json", "body_step": ".step"})


def _canonical_sha256(payload: Any) -> str:
  encoded = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
  return (
      isinstance(value, str)
      and len(value) == 64
      and all(character in _SHA256 for character in value)
  )


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
  return (
      int(value.st_dev),
      int(value.st_ino),
      int(value.st_size),
      int(value.st_mtime_ns),
      int(value.st_ctime_ns),
      int(value.st_nlink),
  )


def _file_object_identity(value: os.stat_result) -> tuple[int, int]:
  return (int(value.st_dev), int(value.st_ino))


def _reject_reparse_path(path: str | Path, *, label: str) -> None:
  requested = Path(os.path.abspath(Path(path)))
  for entry in (requested, *requested.parents):
    try:
      stat = entry.lstat()
    except OSError:
      continue
    is_junction = getattr(entry, "is_junction", None)
    if (
        entry.is_symlink()
        or int(getattr(stat, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
        or (callable(is_junction) and is_junction())
    ):
      raise ValueError(f"{label} path traverses a symlink or reparse point")


def _plain_parent_root(restored_archive_root: str | Path) -> Path:
  requested = Path(os.path.abspath(Path(restored_archive_root)))
  _reject_reparse_path(requested, label="restored_archive_root")
  try:
    if requested.is_symlink() or not requested.is_dir():
      raise ValueError(
          "restored_archive_root must be the parent of restored archive directories"
      )
    return requested.resolve(strict=True)
  except OSError as error:
    raise ValueError(
        "restored_archive_root must be the parent of restored archive directories"
    ) from error


def _archive_identity(raw_archive: Any) -> tuple[str, str]:
  if not isinstance(raw_archive, str):
    raise ValueError(
        "source receipt archive identity must be a canonical ASCII .7z filename"
    )
  archive = raw_archive
  posix = PurePosixPath(archive)
  stem = archive[:-3] if archive.endswith(".7z") else ""
  if (
      not archive
      or not archive.isascii()
      or unicodedata.normalize("NFKC", archive) != archive
      or _ARCHIVE_NAME.fullmatch(archive) is None
      or posix.is_absolute()
      or len(posix.parts) != 1
      or posix.name != archive
      or not stem
  ):
    raise ValueError(
        "source receipt archive identity must name one canonical ASCII direct-child .7z file"
    )
  return archive, stem


def _archive_child(parent: Path, archive: str, stem: str) -> Path:
  try:
    casefold_matches = tuple(
        entry for entry in parent.iterdir() if entry.name.casefold() == stem.casefold()
    )
  except OSError as error:
    raise ValueError(
        "restored_archive_root must be the parent of restored archive directories"
    ) from error
  if not casefold_matches:
    raise ValueError(
        "restored_archive_root must be the parent of restored archive directories"
    )
  exact_matches = tuple(entry for entry in casefold_matches if entry.name == stem)
  if len(exact_matches) != 1:
    raise ValueError(
        f"restored archive {archive!r} lacks one exact direct child named {stem!r}"
    )
  candidate = exact_matches[0]
  _reject_reparse_path(candidate, label=f"restored archive {archive!r}")
  try:
    if candidate.is_symlink() or not candidate.is_dir():
      raise ValueError(
          "restored_archive_root must be the parent of restored archive directories"
      )
    resolved = candidate.resolve(strict=True)
    if resolved.parent != parent or resolved.name != stem:
      raise ValueError(
          f"restored archive {archive!r} is not a plain direct child"
      )
    return resolved
  except OSError as error:
    raise ValueError(
        "restored_archive_root must be the parent of restored archive directories"
    ) from error


def _safe_member(
    root: Path,
    raw_path: Any,
    *,
    label: str,
    role: str | None = None,
) -> tuple[str, Path]:
  if not isinstance(raw_path, str):
    raise ValueError(
        f"{label} has an unsafe archive-relative path; expected canonical ASCII POSIX relative form"
    )
  text = raw_path
  relative = PurePosixPath(text)
  parts = text.split("/")
  if (
      not text
      or not text.isascii()
      or unicodedata.normalize("NFKC", text) != text
      or "\\" in text
      or "//" in text
      or relative.is_absolute()
      or any(
          part in {"", ".", ".."} or _MEMBER_SEGMENT.fullmatch(part) is None
          for part in parts
      )
      or relative.as_posix() != text
  ):
    raise ValueError(
        f"{label} has an unsafe archive-relative path; expected canonical ASCII POSIX relative form"
    )
  if role is not None:
    _validate_member_role(role, text)
  candidate = root.joinpath(*relative.parts)
  _reject_reparse_path(candidate, label=label)
  try:
    resolved = candidate.resolve(strict=True)
    actual_relative = resolved.relative_to(root)
  except (OSError, ValueError) as error:
    raise ValueError(f"{label} is missing from its restored archive") from error
  if actual_relative.as_posix() != text:
    raise ValueError(f"{label} path traverses an alias or reparse point")
  return text, candidate


def _validate_member_role(role: str, relative: str) -> None:
  suffix = _ROLE_SUFFIX.get(role)
  if suffix is None:
    raise ValueError("private source file receipt role is unsupported")
  if not relative.endswith(suffix):
    raise ValueError(f"{role} member must end with {suffix}")


def _archive_root_identity(root: Path) -> tuple[int, int]:
  try:
    stat = root.stat()
  except OSError as error:
    raise ValueError("restored archive root identity is unreadable") from error
  return _file_object_identity(stat)


@dataclass(frozen=True, slots=True)
class LightweightFileCapture:
  """Streaming hash plus filesystem identity; never retains raw STEP bytes."""

  path: Path
  resolved_path: Path
  sha256: str
  byte_count: int
  device: int
  inode: int
  link_count: int
  mtime_ns: int
  ctime_ns: int

  def revalidate(self) -> None:
    current, _payload = _capture_file(
        self.path,
        label="restored archive member",
        retain_bytes=False,
    )
    if current != self:
      raise ValueError("restored archive member changed since capture")


def _capture_file(
    path: Path,
    *,
    label: str,
    retain_bytes: bool,
) -> tuple[LightweightFileCapture, bytes | None]:
  _reject_reparse_path(path, label=label)
  try:
    if path.is_symlink() or not path.is_file():
      raise ValueError(f"{label} must be an existing plain file")
    resolved = path.resolve(strict=True)
    before = path.stat()
    if int(before.st_nlink) != 1:
      raise ValueError(f"{label} must not be hard-linked")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_bytes else None
    byte_count = 0
    with path.open("rb") as stream:
      opened_before = os.fstat(stream.fileno())
      while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
          break
        digest.update(chunk)
        byte_count += len(chunk)
        if chunks is not None:
          chunks.append(chunk)
      opened_after = os.fstat(stream.fileno())
    after = path.stat()
    if (
        path.is_symlink()
        or path.resolve(strict=True) != resolved
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(opened_before) != _stat_identity(opened_after)
        or _file_object_identity(before) != _file_object_identity(opened_before)
        or _file_object_identity(opened_after) != _file_object_identity(after)
        or byte_count != int(opened_after.st_size)
        or any(int(value.st_nlink) != 1 for value in (opened_before, opened_after, after))
    ):
      raise ValueError(f"{label} changed while being captured")
  except ValueError:
    raise
  except OSError as error:
    raise ValueError(f"{label} is missing or unreadable") from error
  captured = LightweightFileCapture(
      path=path,
      resolved_path=resolved,
      sha256=digest.hexdigest(),
      byte_count=byte_count,
      device=int(after.st_dev),
      inode=int(after.st_ino),
      link_count=int(after.st_nlink),
      mtime_ns=int(after.st_mtime_ns),
      ctime_ns=int(after.st_ctime_ns),
  )
  return captured, (b"".join(chunks) if chunks is not None else None)


@dataclass(frozen=True, slots=True)
class CapturedRestoredArchiveInputs:
  captures: tuple[LightweightFileCapture, ...]
  ledger: tuple[dict[str, Any], ...]
  ledger_sha256: str
  case_archive_roots: Mapping[str, tuple[str, Path]]

  def revalidate(self) -> None:
    for captured in self.captures:
      captured.revalidate()


def _case_identity(case: Mapping[str, Any]) -> str:
  case_id = str(case.get("case_id") or case.get("id") or "").strip()
  if not case_id:
    raise ValueError("private source case lacks a nonempty case identity")
  return case_id


def resolve_case_archive_root(
    case: Mapping[str, Any],
    *,
    restored_archive_root: str | Path,
) -> tuple[str, Path]:
  """Resolve exactly the receipt-declared archive for one private case."""

  parent = _plain_parent_root(restored_archive_root)
  receipt = case.get("source_receipt_binding")
  if not isinstance(receipt, Mapping):
    raise ValueError("private source case lacks a receipt binding")
  archive, stem = _archive_identity(receipt.get("archive"))
  return archive, _archive_child(parent, archive, stem)


def capture_restored_archive_inputs(
    source_payload: Any,
    *,
    restored_archive_root: str | Path,
) -> CapturedRestoredArchiveInputs:
  """Capture all receipt claims using ``(archive, path)`` member identity."""

  parent = _plain_parent_root(restored_archive_root)
  cases = source_payload.get("cases") if isinstance(source_payload, Mapping) else None
  if not isinstance(cases, list) or not cases:
    raise ValueError("private source receipt has no cases")
  claims: dict[tuple[str, str], tuple[int, str, str]] = {}
  captures: dict[tuple[str, str], LightweightFileCapture] = {}
  ledger: dict[tuple[str, str], dict[str, Any]] = {}
  case_roots: dict[str, tuple[str, Path]] = {}
  archive_names: dict[str, tuple[str, Path, tuple[int, int]]] = {}
  physical_roots: dict[tuple[int, int], str] = {}
  for raw_case in cases:
    if not isinstance(raw_case, Mapping):
      raise ValueError("private source case is malformed")
    case_id = _case_identity(raw_case)
    if case_id in case_roots:
      raise ValueError("private source case identity is duplicated")
    receipt = raw_case.get("source_receipt_binding")
    if not isinstance(receipt, Mapping):
      raise ValueError("private source case lacks a receipt binding")
    archive, stem = _archive_identity(receipt.get("archive"))
    folded_archive = archive.casefold()
    prior_archive = archive_names.get(folded_archive)
    if prior_archive is not None and prior_archive[0] != archive:
      raise ValueError("different claims use the same casefold archive identity")
    archive_root = _archive_child(parent, archive, stem)
    root_identity = _archive_root_identity(archive_root)
    if prior_archive is not None and (
        prior_archive[1] != archive_root or prior_archive[2] != root_identity
    ):
      raise ValueError("one archive identity resolves to different restored volumes")
    prior_root_archive = physical_roots.get(root_identity)
    if prior_root_archive is not None and prior_root_archive != archive:
      raise ValueError(
          "different archive identities resolve to the same physical restored volume"
      )
    archive_names[folded_archive] = (archive, archive_root, root_identity)
    physical_roots[root_identity] = archive
    case_roots[case_id] = (archive, archive_root)
    records: list[tuple[str, Any]] = [("assembly_json", receipt.get("assembly_json"))]
    body_steps = receipt.get("body_steps")
    if not isinstance(body_steps, list):
      raise ValueError("private source case lacks STEP receipts")
    records.extend(("body_step", record) for record in body_steps)
    for kind, raw_record in records:
      if not isinstance(raw_record, Mapping):
        raise ValueError("private source file receipt is malformed")
      relative, path = _safe_member(
          archive_root,
          raw_record.get("path"),
          label=f"receipt-bound archive member in {archive!r}",
          role=kind,
      )
      expected_bytes = raw_record.get("bytes")
      expected_sha256 = raw_record.get("sha256")
      if (
          isinstance(expected_bytes, bool)
          or not isinstance(expected_bytes, int)
          or expected_bytes < 0
          or not _is_sha256(expected_sha256)
      ):
        raise ValueError("private source file receipt byte binding is malformed")
      key = (archive, relative)
      claim = (expected_bytes, str(expected_sha256), kind)
      prior = claims.get(key)
      if prior is not None:
        if prior[:2] != claim[:2]:
          raise ValueError(
              "same archive member has conflicting receipt claims"
          )
        if prior[2] != kind:
          raise ValueError("same archive member has conflicting receipt roles")
        continue
      captured, raw_bytes = _capture_file(
          path,
          label=f"receipt-bound archive member {archive!r}:{relative!r}",
          retain_bytes=(kind == "assembly_json"),
      )
      if captured.byte_count != expected_bytes or captured.sha256 != expected_sha256:
        raise ValueError("receipt-bound archive member byte binding differs")
      if kind == "assembly_json":
        try:
          assembly = json.loads(
              (raw_bytes or b"").decode("utf-8"),
              object_pairs_hook=_unique_json_object,
              parse_constant=_reject_json_constant,
          )
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
          raise ValueError(
              "receipt-bound assembly JSON is invalid duplicate-key-safe JSON"
          ) from error
        if not isinstance(assembly, Mapping):
          raise ValueError("receipt-bound assembly JSON must be an object")
      claims[key] = claim
      captures[key] = captured
      ledger[key] = {
          "archive": archive,
          "path": relative,
          "bytes": captured.byte_count,
          "sha256": captured.sha256,
          "role": kind,
      }
  ordered_keys = sorted(ledger)
  ordered_ledger = tuple(ledger[key] for key in ordered_keys)
  return CapturedRestoredArchiveInputs(
      captures=tuple(captures[key] for key in ordered_keys),
      ledger=ordered_ledger,
      ledger_sha256=_canonical_sha256(list(ordered_ledger)),
      case_archive_roots=MappingProxyType(dict(sorted(case_roots.items()))),
  )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key: {key}")
    result[key] = value
  return result


def _reject_json_constant(value: str) -> None:
  raise ValueError(f"nonstandard JSON constant: {value}")


__all__ = [
    "CapturedRestoredArchiveInputs",
    "LightweightFileCapture",
    "capture_restored_archive_inputs",
    "resolve_case_archive_root",
]
