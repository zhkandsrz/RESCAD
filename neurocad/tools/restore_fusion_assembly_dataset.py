"""Resume-safe restore tool for the Fusion 360 Gallery Assembly Dataset.

The archive URLs are the eleven a1.0.0 links published by Autodesk AI Lab:
https://github.com/AutodeskAILab/Fusion360GalleryDataset/tree/master/tools/assembly_download
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping
import uuid
import zlib

import requests

from neurocad.paths import DEFAULT_DATASET_ARCHIVE_ROOT, PROJECT_ROOT, resolve_path


DATASET_VERSION = "a1.0.0"
ARCHIVE_NAMES = tuple(f"{DATASET_VERSION}_{index:02d}.7z" for index in range(11))
# Immutable object sizes captured from the official Autodesk S3 objects on
# 2026-07-14. Every normal download also compares these values with a live
# HEAD response, so a truncated local file or changed remote object fails
# closed instead of being accepted merely because two mutable values agree.
ARCHIVE_BYTES = {
    "a1.0.0_00.7z": 2_514_242_842,
    "a1.0.0_01.7z": 2_211_529_612,
    "a1.0.0_02.7z": 2_108_715_291,
    "a1.0.0_03.7z": 1_975_524_796,
    "a1.0.0_04.7z": 1_202_111_575,
    "a1.0.0_05.7z": 2_091_036_039,
    "a1.0.0_06.7z": 1_650_695_992,
    "a1.0.0_07.7z": 2_093_110_875,
    "a1.0.0_08.7z": 1_555_441_511,
    "a1.0.0_09.7z": 1_315_661_285,
    "a1.0.0_10.7z": 1_508_431_532,
}
INTEGRITY_MANIFEST_PATH = (
    PROJECT_ROOT / "config" / "fusion_assembly_a1.0.0_integrity.json"
)
BASE_URL = (
    "https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/"
    f"assembly/{DATASET_VERSION}"
)
PROGRESS_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_EXTRACTION_TIMEOUT_SECONDS = 7200
REPRESENTATIVE_ASSEMBLY_COUNT = 3
CONTENT_RECEIPT_NAME = ".neurocad_content_receipt.json"
CONTENT_RECEIPT_SCHEMA = 2
EXTRACTION_MARKER_NAME = ".neurocad_extract_complete.json"
_PRINT_LOCK = threading.Lock()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CRC32_RE = re.compile(r"^[0-9a-f]{8}$")
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400

IntegritySpec = Mapping[str, Mapping[str, object]]
ArchiveMemberLister = Callable[[Path, Path, int], Iterable[object]]
ArchiveExtractor = Callable[[Path, Path, Path, int], None]


def normalize_integrity_spec(spec: IntegritySpec) -> dict[str, dict[str, object]]:
  normalized: dict[str, dict[str, object]] = {}
  for raw_name, raw_entry in spec.items():
    name = str(raw_name)
    if Path(name).name != name or not name.lower().endswith(".7z"):
      raise ValueError(f"Unsafe archive filename in integrity spec: {name!r}")
    if not isinstance(raw_entry, Mapping):
      raise ValueError(f"Integrity entry for {name} is not an object")
    try:
      expected_bytes = int(raw_entry["bytes"])
      sha256 = str(raw_entry["sha256"]).lower()
    except (KeyError, TypeError, ValueError) as error:
      raise ValueError(f"Invalid integrity entry for {name}") from error
    if expected_bytes <= 0:
      raise ValueError(f"Integrity byte count must be positive for {name}")
    if _SHA256_RE.fullmatch(sha256) is None:
      raise ValueError(f"Integrity SHA256 is invalid for {name}")
    normalized[name] = {"bytes": expected_bytes, "sha256": sha256}
  if not normalized:
    raise ValueError("Integrity spec must contain at least one archive")
  return normalized


def load_integrity_manifest(
    path: str | Path = INTEGRITY_MANIFEST_PATH,
) -> dict[str, dict[str, object]]:
  """Load and validate the version-controlled official integrity manifest."""
  manifest_path = Path(path).expanduser().resolve(strict=True)
  try:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, ValueError, TypeError) as error:
    raise RuntimeError(f"Integrity manifest is unreadable: {manifest_path}") from error
  if not isinstance(payload, dict):
    raise RuntimeError("Integrity manifest must be a JSON object")
  if payload.get("schema_version") != 1:
    raise RuntimeError("Unsupported integrity manifest schema_version")
  if payload.get("dataset_version") != DATASET_VERSION:
    raise RuntimeError("Integrity manifest dataset_version mismatch")
  rows = payload.get("archives")
  if not isinstance(rows, list):
    raise RuntimeError("Integrity manifest archives must be a list")
  raw_spec: dict[str, dict[str, object]] = {}
  for row in rows:
    if not isinstance(row, dict):
      raise RuntimeError("Integrity manifest archive entry must be an object")
    name = str(row.get("filename") or "")
    if name in raw_spec:
      raise RuntimeError(f"Duplicate integrity manifest archive: {name}")
    raw_spec[name] = {
        "bytes": row.get("bytes"),
        "sha256": row.get("sha256"),
    }
  spec = normalize_integrity_spec(raw_spec)
  if tuple(spec) != ARCHIVE_NAMES:
    raise RuntimeError("Integrity manifest must list the exact official 00..10 set")
  manifest_bytes = {name: int(entry["bytes"]) for name, entry in spec.items()}
  if manifest_bytes != ARCHIVE_BYTES:
    raise RuntimeError("Integrity manifest byte sizes disagree with ARCHIVE_BYTES")
  return spec


ARCHIVE_INTEGRITY = load_integrity_manifest()
ARCHIVE_SHA256 = {
    name: str(entry["sha256"])
    for name, entry in ARCHIVE_INTEGRITY.items()
}


def _integrity_entry(
    name: str,
    integrity_spec: IntegritySpec | None,
) -> dict[str, object]:
  spec = ARCHIVE_INTEGRITY if integrity_spec is None else normalize_integrity_spec(
      integrity_spec
  )
  try:
    return spec[name]
  except KeyError as error:
    raise ValueError(f"Archive is absent from integrity spec: {name}") from error


def archive_url(name: str) -> str:
  if name not in ARCHIVE_NAMES:
    raise ValueError(f"Unknown Fusion Assembly archive: {name}")
  return f"{BASE_URL}/{name}"


def _print(message: str) -> None:
  with _PRINT_LOCK:
    print(message, flush=True)


def _remote_size(session: requests.Session, url: str) -> int:
  response = session.head(url, allow_redirects=True, timeout=(30, 120))
  response.raise_for_status()
  raw_size = response.headers.get("content-length")
  if raw_size is None:
    raise RuntimeError(f"Server did not report Content-Length for {url}")
  return int(raw_size)


def _content_range_total(value: str | None) -> int | None:
  if not value:
    return None
  match = re.search(r"/(\d+)$", value)
  return None if match is None else int(match.group(1))


def _download_archive_locked(name: str, archive_root: Path) -> dict[str, Any]:
  """Download one official archive while its caller holds the volume lock."""
  url = archive_url(name)
  target = archive_root / name
  partial = target.with_suffix(target.suffix + ".part")
  _assert_safe_direct_child(target, archive_root, expected_name=name)
  _assert_safe_direct_child(
      partial,
      archive_root,
      expected_name=f"{name}.part",
  )
  if _is_reparse_point(target):
    raise RuntimeError(f"Archive target is a reparse point: {target}")
  if _is_reparse_point(partial):
    raise RuntimeError(f"Partial archive is a reparse point: {partial}")
  if target.exists() and not target.is_file():
    raise RuntimeError(f"Archive target is not a regular file: {target}")
  if partial.exists() and not partial.is_file():
    raise RuntimeError(f"Partial archive is not a regular file: {partial}")

  with requests.Session() as session:
    expected_size = ARCHIVE_BYTES[name]
    expected_sha256 = ARCHIVE_SHA256[name]
    remote_size = _remote_size(session, url)
    if remote_size != expected_size:
      raise RuntimeError(
          f"Official manifest/remote size disagreement for {name}: "
          f"{expected_size} != {remote_size}"
      )
    if target.exists():
      actual_size = target.stat().st_size
      if actual_size == expected_size:
        actual_sha256 = _sha256(target)
        if actual_sha256 != expected_sha256:
          raise RuntimeError(
              f"Existing archive has wrong SHA256: {target} "
              f"({actual_sha256} != {expected_sha256})"
          )
        _print(f"[download] {name}: already complete ({actual_size} bytes)")
        return {
            "name": name,
            "url": url,
            "bytes": actual_size,
            "sha256": actual_sha256,
            "status": "present",
        }
      raise RuntimeError(
          f"Existing archive has wrong size: {target} "
          f"({actual_size} != {expected_size})"
      )

    resume_from = partial.stat().st_size if partial.exists() else 0
    if resume_from > expected_size:
      raise RuntimeError(
          f"Partial archive is larger than the server object: {partial}"
      )
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with session.get(
        url,
        headers=headers,
        allow_redirects=True,
        stream=True,
        timeout=(30, 300),
    ) as response:
      response.raise_for_status()
      append = resume_from > 0 and response.status_code == 206
      if resume_from > 0 and not append:
        _print(f"[download] {name}: server ignored Range; restarting")
        resume_from = 0
      response_total = _content_range_total(response.headers.get("content-range"))
      if response_total is not None and response_total != expected_size:
        raise RuntimeError(
            f"HEAD/GET size disagreement for {name}: "
            f"{expected_size} != {response_total}"
        )

      mode = "ab" if append else "wb"
      downloaded = resume_from
      next_report = ((downloaded // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
      started = time.monotonic()
      with partial.open(mode) as stream:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
          if not chunk:
            continue
          stream.write(chunk)
          downloaded += len(chunk)
          if downloaded >= next_report:
            # Publish a durable resume boundary. On Windows an open file may
            # otherwise continue to report length zero until the handle closes.
            stream.flush()
            os.fsync(stream.fileno())
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = (downloaded - resume_from) / elapsed / (1024 * 1024)
            percent = 100.0 * downloaded / expected_size
            _print(
                f"[download] {name}: {percent:5.1f}% "
                f"({downloaded / (1024**3):.2f} GiB, {rate:.1f} MiB/s)"
            )
            next_report += PROGRESS_BYTES

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
      raise RuntimeError(
          f"Incomplete archive after download: {partial} "
          f"({actual_size} != {expected_size})"
      )
    actual_sha256 = _sha256(partial)
    if actual_sha256 != expected_sha256:
      raise RuntimeError(
          f"Downloaded archive has wrong SHA256: {partial} "
          f"({actual_sha256} != {expected_sha256})"
      )
    partial.replace(target)
    _print(f"[download] {name}: complete ({actual_size} bytes)")
    return {
        "name": name,
        "url": url,
        "bytes": actual_size,
        "sha256": actual_sha256,
        "status": "downloaded",
    }


def download_archive(
    name: str,
    archive_root: Path,
    *,
    _lock_already_held: bool = False,
) -> dict[str, Any]:
  """Download one official archive with all target/partial I/O under its lock."""

  root = _resolve_plain_directory(
      archive_root,
      label="Archive root",
      create=True,
  )
  if _lock_already_held:
    return _download_archive_locked(name, root)
  with volume_restore_lock(name, root):
    return _download_archive_locked(name, root)


def download_archive_with_retries(
    name: str,
    archive_root: Path,
    *,
    attempts: int = 5,
) -> dict[str, Any]:
  """Download an archive with bounded retries over the durable partial file.

  The S3 connection can occasionally close cleanly before the advertised
  Content-Length is reached. ``download_archive`` detects that condition and
  leaves a flushed ``.part`` file; retrying therefore resumes rather than
  downloading the volume again.
  """
  if attempts < 1:
    raise ValueError("attempts must be at least one")
  root = _resolve_plain_directory(
      archive_root,
      label="Archive root",
      create=True,
  )
  with volume_restore_lock(name, root):
    for attempt in range(1, attempts + 1):
      try:
        return download_archive(name, root, _lock_already_held=True)
      except Exception as error:
        if attempt == attempts:
          raise
        delay_seconds = min(2 ** (attempt - 1), 30)
        _print(
            f"[download] {name}: attempt {attempt}/{attempts} failed "
            f"({type(error).__name__}: {error}); retrying in "
            f"{delay_seconds}s"
        )
        time.sleep(delay_seconds)
  raise AssertionError("unreachable")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(CHUNK_BYTES):
      digest.update(chunk)
  return digest.hexdigest()


def integrity_spec_provenance(
    integrity_spec: IntegritySpec | None,
) -> dict[str, object]:
  official = integrity_spec is None
  normalized = (
      ARCHIVE_INTEGRITY
      if official
      else normalize_integrity_spec(integrity_spec)
  )
  canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
  return {
      "official": official,
      "manifest_path": str(INTEGRITY_MANIFEST_PATH.resolve()) if official else None,
      "manifest_sha256": (
          _sha256(INTEGRITY_MANIFEST_PATH) if official else None
      ),
      "spec_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
  }


def _validate_archive_integrity_locked(
    name: str,
    archive_root: str | Path,
    *,
    integrity_spec: IntegritySpec | None = None,
) -> dict[str, Any]:
  """Read and validate one archive without writing anywhere.

  Omitting ``integrity_spec`` selects the checked-in official manifest. Tests
  may inject a tiny specification, but the returned provenance marks that
  result as non-official so a formal runner cannot mistake it for evidence.
  """

  provenance = integrity_spec_provenance(integrity_spec)
  errors: list[str] = []
  root = Path(archive_root)
  try:
    integrity = _integrity_entry(name, integrity_spec)
  except ValueError as error:
    errors.append(str(error))
    integrity = None
  archive = root / name
  try:
    _assert_safe_direct_child(archive, root, expected_name=name)
  except ValueError as error:
    errors.append(str(error))
  actual_bytes: int | None = None
  actual_sha256: str | None = None
  if not archive.is_file() or _is_reparse_point(archive):
    errors.append(f"archive missing or not a regular file: {archive}")
  elif integrity is not None:
    actual_bytes = archive.stat().st_size
    expected_bytes = int(integrity["bytes"])
    if actual_bytes != expected_bytes:
      errors.append(
          f"archive byte-size mismatch: {actual_bytes} != {expected_bytes}"
      )
    actual_sha256 = _sha256(archive)
    expected_sha256 = str(integrity["sha256"])
    if actual_sha256 != expected_sha256:
      errors.append(
          f"archive SHA256 mismatch: {actual_sha256} != {expected_sha256}"
      )
  return {
      "archive": name,
      "archive_bytes": actual_bytes,
      "archive_path": str(archive),
      "archive_sha256": actual_sha256,
      "errors": errors,
      "integrity_provenance": provenance,
      "valid": not errors,
  }


def validate_archive_integrity(
    name: str,
    archive_root: str | Path,
    *,
    integrity_spec: IntegritySpec | None = None,
    _lock_already_held: bool = False,
) -> dict[str, Any]:
  """Hash one archive under the same per-volume lock used by restore writes."""

  provenance = integrity_spec_provenance(integrity_spec)
  raw_root = Path(archive_root).expanduser()
  try:
    root = _resolve_plain_directory(raw_root, label="Archive root")
  except (OSError, ValueError) as error:
    return {
        "archive": name,
        "archive_path": str(raw_root / name),
        "errors": [str(error)],
        "integrity_provenance": provenance,
        "valid": False,
    }
  if _lock_already_held:
    return _validate_archive_integrity_locked(
        name,
        root,
        integrity_spec=integrity_spec,
    )
  with volume_restore_lock(name, root):
    return _validate_archive_integrity_locked(
        name,
        root,
        integrity_spec=integrity_spec,
    )


def find_7z(explicit: str | None) -> Path:
  workspace_tool = PROJECT_ROOT.parent / ".tools" / "7-Zip" / "7z.exe"
  candidates = [
      explicit,
      os.getenv("NEUROCAD_7Z"),
      str(workspace_tool),
      r"C:\Program Files\7-Zip\7z.exe",
      shutil.which("7z"),
      shutil.which("7za"),
  ]
  for raw in candidates:
    if not raw:
      continue
    candidate = Path(raw).expanduser().resolve(strict=False)
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(
      "7z.exe not found; pass --seven-zip or set NEUROCAD_7Z."
  )


def _top_level_directories(extract_root: Path) -> list[Path]:
  if _is_reparse_point(extract_root) or not extract_root.is_dir():
    raise RuntimeError(f"Extraction root is not a plain directory: {extract_root}")
  directories: list[Path] = []
  for path in extract_root.iterdir():
    if _is_reparse_point(path):
      raise RuntimeError(f"Top-level extraction member is a reparse point: {path}")
    if path.is_dir():
      directories.append(path)
  return sorted(directories, key=lambda path: path.name.lower())


def _representative_files(extract_root: Path, directories: list[Path]) -> list[str]:
  """Return bounded, deterministic evidence spanning an extracted volume."""
  if not directories:
    return []
  indexes = sorted({0, len(directories) // 2, len(directories) - 1})
  representatives: list[str] = []
  for index in indexes[:REPRESENTATIVE_ASSEMBLY_COUNT]:
    directory = directories[index]
    assembly_json = directory / "assembly.json"
    if assembly_json.is_file():
      representatives.append(assembly_json.relative_to(extract_root).as_posix())
    step = next(
        (path for path in sorted(directory.glob("*.step")) if path.is_file()),
        None,
    )
    if step is not None:
      representatives.append(step.relative_to(extract_root).as_posix())
  return representatives


def _validate_extraction_marker_locked(
    name: str,
    archive_root: Path,
    *,
    deep_audit: bool = False,
    integrity_spec: IntegritySpec | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
  """Validate a marker against bounded live filesystem evidence.

  Normal checks enumerate only immediate assembly directories and a maximum of
  three representative assemblies. The optional deep audit requires a marker-
  bound full-byte content receipt and compares its canonical member/stat records
  with the current tree without rereading every STEP/JSON payload byte.
  """
  archive = archive_root / name
  extract_root = archive_root / Path(name).stem
  marker = extract_root / EXTRACTION_MARKER_NAME
  try:
    integrity = _integrity_entry(name, integrity_spec)
  except ValueError as error:
    return False, str(error), None
  expected_archive_bytes = int(integrity["bytes"])
  expected_archive_sha256 = str(integrity["sha256"])
  if _is_reparse_point(archive) or not archive.is_file():
    return False, f"archive missing: {archive}", None
  actual_archive_bytes = archive.stat().st_size
  if actual_archive_bytes != expected_archive_bytes:
    return (
        False,
        "archive byte-size mismatch: "
        f"{actual_archive_bytes} != {expected_archive_bytes}",
        None,
    )
  if _is_reparse_point(extract_root) or not extract_root.is_dir():
    return False, f"extraction root is not a plain directory: {extract_root}", None
  if _is_reparse_point(marker) or not marker.is_file():
    return False, f"marker missing: {marker}", None
  try:
    state = json.loads(marker.read_text(encoding="utf-8"))
  except (OSError, ValueError, TypeError) as error:
    return False, f"marker unreadable: {type(error).__name__}: {error}", None
  if not isinstance(state, dict):
    return False, "marker is not a JSON object", None
  required_matches = {
      "archive": name,
      "archive_bytes": expected_archive_bytes,
      "archive_sha256": expected_archive_sha256,
      "dataset_version": DATASET_VERSION,
      "status": "ok",
  }
  for key, expected in required_matches.items():
    if state.get(key) != expected:
      return False, f"marker {key} mismatch", state
  expected_steps = int(state.get("step_files") or 0)
  expected_json = int(state.get("json_files") or 0)
  if expected_steps <= 0 or expected_json <= 0:
    return False, "marker contains non-positive file counts", state
  try:
    directories = _top_level_directories(extract_root)
  except RuntimeError as error:
    return False, str(error), state
  raw_expected_directories = state.get("top_level_directory_count")
  expected_directories = (
      int(raw_expected_directories)
      if raw_expected_directories is not None
      else None
  )
  if expected_directories is not None and len(directories) != expected_directories:
    return (
        False,
        "top-level directory count mismatch: "
        f"{len(directories)} != {expected_directories}",
        state,
    )

  representative_paths = state.get("representative_files")
  if not isinstance(representative_paths, list) or not representative_paths:
    # Backward-compatible bounded evidence for markers written before schema 1.
    representative_paths = _representative_files(extract_root, directories)
  if len(representative_paths) < 2:
    return False, "insufficient representative extraction evidence", state
  for raw in representative_paths:
    try:
      normalized = _normalize_archive_member(raw)
    except RuntimeError:
      return False, f"unsafe representative path in marker: {raw}", state
    candidate = extract_root / Path(normalized)
    if _is_reparse_point(candidate):
      return False, f"representative file is a reparse point: {raw}", state
    if not candidate.is_file() or candidate.stat().st_size <= 0:
      return False, f"representative file missing or empty: {raw}", state

  if deep_audit:
    valid_receipt, reason, receipt_summary = _validate_content_receipt_locked(
        name,
        archive_root,
        extract_root,
        state,
        integrity_spec=integrity_spec,
    )
    if not valid_receipt:
      return False, reason, state
    state = dict(state)
    state["content_receipt_provenance"] = receipt_summary
  return True, "ok", state


def validate_extraction_marker(
    name: str,
    archive_root: Path,
    *,
    deep_audit: bool = False,
    integrity_spec: IntegritySpec | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
  """Validate one volume while excluding concurrent restore/download writes."""

  try:
    root = _resolve_plain_directory(archive_root, label="Archive root")
  except (OSError, ValueError) as error:
    return False, str(error), None
  with volume_restore_lock(name, root):
    return _validate_extraction_marker_locked(
        name,
        root,
        deep_audit=deep_audit,
        integrity_spec=integrity_spec,
    )


def _is_reparse_point(path: Path) -> bool:
  is_junction = getattr(os.path, "isjunction", None)
  try:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
  except OSError:
    attributes = 0
  return (
      path.is_symlink()
      or bool(is_junction and is_junction(path))
      or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)
  )


def _resolve_plain_directory(
    raw_path: str | Path,
    *,
    label: str,
    create: bool = False,
) -> Path:
  """Resolve an exact directory while preserving and rejecting link identity."""

  path = Path(raw_path).expanduser()
  if not path.is_absolute():
    path = Path.cwd() / path
  if create:
    path.mkdir(parents=True, exist_ok=True)
  if _is_reparse_point(path):
    raise ValueError(f"{label} must not be a symlink, junction, or reparse point: {path}")
  resolved = path.resolve(strict=True)
  if not resolved.is_dir() or _is_reparse_point(resolved):
    raise ValueError(f"{label} is not a plain directory: {path}")
  return resolved


def _assert_safe_direct_child(
    path: Path,
    archive_root: Path,
    *,
    expected_name: str | None = None,
    expected_prefix: str | None = None,
) -> None:
  """Reject paths that could make a move/delete escape the exact archive root."""
  root = archive_root.resolve(strict=True)
  if not root.is_dir():
    raise ValueError(f"Archive root is not a directory: {root}")
  if path.parent.resolve(strict=True) != root:
    raise ValueError(f"Unsafe path outside exact archive root: {path}")
  if expected_name is not None and path.name != expected_name:
    raise ValueError(f"Unexpected managed path name: {path.name}")
  if expected_prefix is not None and not path.name.startswith(expected_prefix):
    raise ValueError(f"Unexpected managed path prefix: {path.name}")
  if _is_reparse_point(path):
    raise ValueError(f"Managed path must not be a symlink or junction: {path}")
  if path.exists() and path.resolve(strict=True).parent != root:
    raise ValueError(f"Resolved managed path escapes archive root: {path}")


def _safe_remove_tree(
    path: Path,
    archive_root: Path,
    *,
    expected_prefix: str,
) -> None:
  _assert_safe_direct_child(
      path,
      archive_root,
      expected_prefix=expected_prefix,
  )
  if not path.exists():
    return
  if not path.is_dir():
    raise ValueError(f"Refusing recursive removal of a non-directory: {path}")
  root = archive_root.resolve(strict=True)
  managed_root = path.resolve(strict=True)
  for raw_directory, directory_names, file_names in os.walk(
      path,
      topdown=True,
      followlinks=False,
  ):
    directory = Path(raw_directory)
    resolved_directory = directory.resolve(strict=True)
    if (
        not resolved_directory.is_relative_to(root)
        or not resolved_directory.is_relative_to(managed_root)
    ):
      raise ValueError(f"Recursive removal tree escapes managed root: {directory}")
    for child_name in (*directory_names, *file_names):
      child = directory / child_name
      if _is_reparse_point(child):
        raise ValueError(
            f"Refusing recursive removal through a symlink or junction: {child}"
        )
      resolved_child = child.resolve(strict=True)
      if (
          not resolved_child.is_relative_to(root)
          or not resolved_child.is_relative_to(managed_root)
      ):
        raise ValueError(f"Recursive removal child escapes managed root: {child}")
  shutil.rmtree(path)


def _canonical_json_sha256(value: object) -> str:
  encoded = json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _receipt_integrity_binding(
    integrity_spec: IntegritySpec | None,
) -> dict[str, object]:
  provenance = integrity_spec_provenance(integrity_spec)
  return {
      "official": provenance["official"],
      "manifest_path": (
          INTEGRITY_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()
          if provenance["official"]
          else None
      ),
      "manifest_sha256": provenance["manifest_sha256"],
      "spec_sha256": provenance["spec_sha256"],
  }


def _stat_record(relative_path: str, metadata: os.stat_result) -> dict[str, int | str]:
  return {
      "path": relative_path,
      "bytes": int(metadata.st_size),
      "mtime_ns": int(metadata.st_mtime_ns),
      "ctime_ns": int(metadata.st_ctime_ns),
      "mode": int(metadata.st_mode),
      "inode": int(metadata.st_ino),
      "device": int(metadata.st_dev),
  }


def _member_stat_view(member: Mapping[str, object]) -> dict[str, object]:
  return {
      key: member[key]
      for key in (
          "path",
          "bytes",
          "mtime_ns",
          "ctime_ns",
          "mode",
          "inode",
          "device",
      )
  }


def _member_content_view(member: Mapping[str, object]) -> dict[str, object]:
  return {
      "path": member["path"],
      "bytes": member["bytes"],
      "crc32": member["crc32"],
      "sha256": member["sha256"],
  }


def _member_archive_metadata_view(
    member: Mapping[str, object],
) -> dict[str, object]:
  return {
      "path": member["path"],
      "bytes": member["bytes"],
      "crc32": member["crc32"],
  }


def _hash_receipt_member(path: Path) -> tuple[str, str]:
  digest = hashlib.sha256()
  checksum = 0
  with path.open("rb") as stream:
    while chunk := stream.read(CHUNK_BYTES):
      digest.update(chunk)
      checksum = zlib.crc32(chunk, checksum)
  return digest.hexdigest(), f"{checksum & 0xFFFFFFFF:08x}"


def _scan_receipt_members(
    extract_root: Path,
    *,
    hash_bytes: bool,
) -> list[dict[str, object]]:
  """Enumerate a plain tree and optionally hash every STEP/JSON member byte."""

  if _is_reparse_point(extract_root) or not extract_root.is_dir():
    raise RuntimeError(
        "Extraction root must be a plain directory, not a symlink, junction, "
        f"or reparse point: {extract_root}"
    )
  records: list[dict[str, object]] = []
  stack = [extract_root]
  while stack:
    directory = stack.pop()
    if _is_reparse_point(directory):
      raise RuntimeError(f"Extraction directory is a reparse point: {directory}")
    with os.scandir(directory) as iterator:
      entries = sorted(iterator, key=lambda entry: entry.name.encode("utf-8"))
    child_directories: list[Path] = []
    for entry in entries:
      path = Path(entry.path)
      metadata = entry.stat(follow_symlinks=False)
      if entry.is_symlink() or bool(
          int(getattr(metadata, "st_file_attributes", 0))
          & _FILE_ATTRIBUTE_REPARSE_POINT
      ) or _is_reparse_point(path):
        raise RuntimeError(
            "Extracted member must not be a symlink, junction, or reparse "
            f"point: {path}"
        )
      if stat.S_ISDIR(metadata.st_mode):
        child_directories.append(path)
        continue
      if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Extracted member is not a regular file: {path}")
      relative = path.relative_to(extract_root).as_posix()
      if relative in {CONTENT_RECEIPT_NAME, EXTRACTION_MARKER_NAME}:
        continue
      if path.suffix.lower() not in {".step", ".json"}:
        continue
      record: dict[str, object] = dict(_stat_record(relative, path.lstat()))
      if hash_bytes:
        digest, crc32 = _hash_receipt_member(path)
        after = path.lstat()
        after_record = _stat_record(relative, after)
        stable_keys = ("path", "bytes", "mtime_ns", "mode", "inode", "device")
        if any(after_record[key] != record[key] for key in stable_keys):
          raise RuntimeError(f"Extracted member changed while hashing: {relative}")
        record = dict(after_record)
        record["sha256"] = digest
        record["crc32"] = crc32
      records.append(record)
    stack.extend(reversed(child_directories))
  return sorted(records, key=lambda row: str(row["path"]).encode("utf-8"))


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, object],
    *,
    replace: bool,
) -> None:
  if _is_reparse_point(path):
    raise RuntimeError(f"Refusing to replace reparse-point metadata: {path}")
  if path.exists() and not replace:
    raise FileExistsError(f"Metadata already exists: {path}")
  temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
  try:
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
      json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    if temporary.exists():
      temporary.unlink()


def _verified_receipt_archive_record(
    name: str,
    archive_root: Path,
    *,
    integrity_spec: IntegritySpec | None,
) -> dict[str, object]:
  integrity = _integrity_entry(name, integrity_spec)
  archive = archive_root / name
  _assert_safe_direct_child(archive, archive_root, expected_name=name)
  if _is_reparse_point(archive) or not archive.is_file():
    raise RuntimeError(f"Archive must be a regular non-reparse file: {archive}")
  archive_before = archive.lstat()
  archive_sha256 = _sha256(archive)
  archive_after = archive.lstat()
  archive_stat = _stat_record(name, archive_before)
  if archive_stat != _stat_record(name, archive_after):
    raise RuntimeError(f"Archive changed while hashing: {archive}")
  if int(archive_stat["bytes"]) != int(integrity["bytes"]):
    raise RuntimeError(f"Archive byte-size mismatch while creating receipt: {name}")
  if archive_sha256 != str(integrity["sha256"]):
    raise RuntimeError(f"Archive SHA256 mismatch while creating receipt: {name}")
  return {
      "name": name,
      "bytes": int(integrity["bytes"]),
      "sha256": archive_sha256,
      "stat": archive_stat,
  }


def _build_content_receipt_payload(
    name: str,
    archive_root: Path,
    extract_root: Path,
    *,
    integrity_spec: IntegritySpec | None,
    archive_member_metadata: Iterable[Mapping[str, object]] | None = None,
    verified_archive_record: Mapping[str, object] | None = None,
) -> dict[str, object]:
  integrity = _integrity_entry(name, integrity_spec)
  provenance = _receipt_integrity_binding(integrity_spec)
  archive = archive_root / name
  _assert_safe_direct_child(archive, archive_root, expected_name=name)
  archive_record = dict(
      verified_archive_record
      or _verified_receipt_archive_record(
          name,
          archive_root,
          integrity_spec=integrity_spec,
      )
  )
  if (
      archive_record.get("name") != name
      or archive_record.get("bytes") != int(integrity["bytes"])
      or archive_record.get("sha256") != str(integrity["sha256"])
  ):
    raise RuntimeError(f"Verified archive identity mismatch for receipt: {name}")
  if _is_reparse_point(archive) or not archive.is_file():
    raise RuntimeError(f"Archive must be a regular non-reparse file: {archive}")
  if archive_record.get("stat") != _stat_record(name, archive.lstat()):
    raise RuntimeError(f"Archive changed after SHA256 authentication: {name}")

  members = _scan_receipt_members(extract_root, hash_bytes=True)
  step_files = sum(str(row["path"]).lower().endswith(".step") for row in members)
  json_files = sum(str(row["path"]).lower().endswith(".json") for row in members)
  if step_files <= 0 or json_files <= 0:
    raise RuntimeError(
        f"Content receipt requires STEP and JSON members: {name} "
        f"(step={step_files}, json={json_files})"
    )
  stat_view = [_member_stat_view(row) for row in members]
  current_stat_view = [
      _member_stat_view(row)
      for row in _scan_receipt_members(extract_root, hash_bytes=False)
  ]
  if current_stat_view != stat_view:
    raise RuntimeError(f"Extracted tree changed while creating receipt: {name}")
  content_view = [_member_content_view(row) for row in members]
  live_archive_metadata = [
      _member_archive_metadata_view(row) for row in members
  ]
  metadata_authenticated = False
  if archive_member_metadata is not None:
    expected_archive_metadata = [
        dict(row)
        for row in archive_member_metadata
        if Path(str(row["path"])).suffix.lower() in {".step", ".json"}
    ]
    expected_by_path = {
        str(row["path"]): row for row in expected_archive_metadata
    }
    live_by_path = {str(row["path"]): row for row in live_archive_metadata}
    missing = sorted(set(expected_by_path) - set(live_by_path))
    extra = sorted(set(live_by_path) - set(expected_by_path))
    if missing or extra:
      raise RuntimeError(
          "Archive/live receipt member set mismatch: "
          f"missing={missing[:10]}, extra={extra[:10]}"
      )
    for path in sorted(expected_by_path, key=lambda value: value.encode("utf-8")):
      expected = expected_by_path[path]
      live = live_by_path[path]
      if int(live["bytes"]) != int(expected["bytes"]):
        raise RuntimeError(
            f"Archive member byte-size mismatch for {path}: "
            f"{live['bytes']} != {expected['bytes']}"
        )
      if str(live["crc32"]) != str(expected["crc32"]).lower():
        raise RuntimeError(
            f"Archive member CRC mismatch for {path}: "
            f"{live['crc32']} != {expected['crc32']}"
        )
    live_archive_metadata = expected_archive_metadata
    metadata_authenticated = integrity_spec is None
  elif integrity_spec is None:
    raise RuntimeError(
        "Official content receipt requires authenticated archive member metadata"
    )
  archive_member_metadata_set_sha256 = _canonical_json_sha256(
      live_archive_metadata
  )
  if archive_record.get("stat") != _stat_record(name, archive.lstat()):
    raise RuntimeError(f"Archive changed while creating receipt: {name}")
  paths = [str(row["path"]) for row in members]
  return {
      "schema_version": CONTENT_RECEIPT_SCHEMA,
      "dataset_version": DATASET_VERSION,
      "archive": {
          **archive_record,
      },
      "integrity_provenance": provenance,
      "member_count": len(members),
      "step_files": step_files,
      "json_files": json_files,
      "total_bytes": sum(int(row["bytes"]) for row in members),
      "path_set_sha256": _canonical_json_sha256(paths),
      "aggregate_sha256": _canonical_json_sha256(content_view),
      "archive_member_metadata_authenticated": metadata_authenticated,
      "archive_member_metadata_set_sha256": archive_member_metadata_set_sha256,
      "stat_fingerprint_sha256": _canonical_json_sha256(stat_view),
      "members": members,
  }


def _content_receipt_summary(
    receipt_path: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
  provenance = payload["integrity_provenance"]
  if not isinstance(provenance, Mapping):
    raise RuntimeError("Content receipt integrity provenance is malformed")
  metadata_authenticated = (
      payload.get("archive_member_metadata_authenticated") is True
  )
  return {
      "path": str(receipt_path.resolve(strict=True)),
      "sha256": _sha256(receipt_path),
      "official": (
          provenance.get("official") is True and metadata_authenticated
      ),
      "manifest_sha256": provenance.get("manifest_sha256"),
      "spec_sha256": provenance.get("spec_sha256"),
      "aggregate_sha256": payload["aggregate_sha256"],
      "archive_member_metadata_authenticated": metadata_authenticated,
      "archive_member_metadata_set_sha256": payload[
          "archive_member_metadata_set_sha256"
      ],
      "path_set_sha256": payload["path_set_sha256"],
      "stat_fingerprint_sha256": payload["stat_fingerprint_sha256"],
      "member_count": int(payload["member_count"]),
      "total_bytes": int(payload["total_bytes"]),
  }


def _marker_receipt_binding(summary: Mapping[str, object]) -> dict[str, object]:
  """Return the deterministic, extraction-root-relative marker binding."""

  return {
      "name": CONTENT_RECEIPT_NAME,
      **{
          key: summary.get(key)
          for key in (
              "sha256",
              "official",
              "manifest_sha256",
              "spec_sha256",
              "aggregate_sha256",
              "archive_member_metadata_authenticated",
              "archive_member_metadata_set_sha256",
              "path_set_sha256",
              "stat_fingerprint_sha256",
              "member_count",
              "total_bytes",
          )
      },
  }


def _validate_content_receipt_locked(
    name: str,
    archive_root: Path,
    extract_root: Path,
    marker_state: Mapping[str, object],
    *,
    integrity_spec: IntegritySpec | None,
) -> tuple[bool, str, dict[str, object] | None]:
  receipt_path = extract_root / CONTENT_RECEIPT_NAME
  if not receipt_path.exists():
    return (
        False,
        "content receipt missing; run restore_fusion_assembly_dataset.py "
        "--refresh-content-receipts",
        None,
    )
  if _is_reparse_point(receipt_path) or not receipt_path.is_file():
    return False, f"content receipt is not a regular file: {receipt_path}", None
  try:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
  except (OSError, ValueError, TypeError) as error:
    return False, f"content receipt unreadable: {type(error).__name__}: {error}", None
  if not isinstance(payload, dict):
    return False, "content receipt is not a JSON object", None
  if payload.get("schema_version") != CONTENT_RECEIPT_SCHEMA:
    return False, "content receipt schema_version mismatch", None
  if payload.get("dataset_version") != DATASET_VERSION:
    return False, "content receipt dataset_version mismatch", None

  expected_provenance = _receipt_integrity_binding(integrity_spec)
  if payload.get("integrity_provenance") != expected_provenance:
    return False, "content receipt integrity provenance mismatch", None
  if integrity_spec is None and expected_provenance.get("official") is not True:
    return False, "formal content receipt is not official", None
  metadata_authenticated = payload.get(
      "archive_member_metadata_authenticated"
  )
  if not isinstance(metadata_authenticated, bool):
    return False, "content receipt archive metadata authentication is malformed", None
  if integrity_spec is None and metadata_authenticated is not True:
    return False, "formal content receipt lacks authenticated archive metadata", None

  receipt_sha256 = _sha256(receipt_path)
  marker_binding = marker_state.get("content_receipt")
  if not isinstance(marker_binding, Mapping):
    return False, "marker does not bind the content receipt", None
  if (
      marker_binding.get("name") != CONTENT_RECEIPT_NAME
      or marker_binding.get("sha256") != receipt_sha256
  ):
    return False, "marker content receipt binding mismatch", None

  integrity = _integrity_entry(name, integrity_spec)
  archive_record = payload.get("archive")
  if not isinstance(archive_record, Mapping):
    return False, "content receipt archive record is malformed", None
  if (
      archive_record.get("name") != name
      or archive_record.get("bytes") != int(integrity["bytes"])
      or archive_record.get("sha256") != str(integrity["sha256"])
  ):
    return False, "content receipt archive identity mismatch", None
  archive = archive_root / name
  if _is_reparse_point(archive) or not archive.is_file():
    return False, f"archive is not a regular file: {archive}", None
  if archive_record.get("stat") != _stat_record(name, archive.lstat()):
    return False, "content receipt archive stat fingerprint mismatch", None

  raw_members = payload.get("members")
  if not isinstance(raw_members, list) or not raw_members:
    return False, "content receipt members are missing", None
  members: list[dict[str, object]] = []
  paths: list[str] = []
  try:
    for raw_member in raw_members:
      if not isinstance(raw_member, dict):
        raise ValueError("member is not an object")
      path = _normalize_archive_member(raw_member.get("path"))
      if path in {CONTENT_RECEIPT_NAME, EXTRACTION_MARKER_NAME}:
        raise ValueError(f"reserved metadata path in receipt: {path}")
      if Path(path).suffix.lower() not in {".step", ".json"}:
        raise ValueError(f"non STEP/JSON member in receipt: {path}")
      if _SHA256_RE.fullmatch(str(raw_member.get("sha256") or "")) is None:
        raise ValueError(f"invalid member SHA256: {path}")
      if _CRC32_RE.fullmatch(str(raw_member.get("crc32") or "")) is None:
        raise ValueError(f"invalid member CRC32: {path}")
      for key in (
          "bytes",
          "mtime_ns",
          "ctime_ns",
          "mode",
          "inode",
          "device",
      ):
        value = raw_member.get(key)
        if not isinstance(value, int) or value < 0:
          raise ValueError(f"invalid member {key}: {path}")
      member = dict(raw_member)
      member["path"] = path
      members.append(member)
      paths.append(path)
  except (KeyError, TypeError, ValueError) as error:
    return False, f"content receipt member record malformed: {error}", None
  if paths != sorted(paths, key=lambda path: path.encode("utf-8")):
    return False, "content receipt member paths are not canonical", None
  if len(set(paths)) != len(paths):
    return False, "content receipt contains duplicate member paths", None

  content_view = [_member_content_view(row) for row in members]
  archive_metadata_view = [
      _member_archive_metadata_view(row) for row in members
  ]
  stat_view = [_member_stat_view(row) for row in members]
  if payload.get("member_count") != len(members):
    return False, "content receipt member_count mismatch", None
  if payload.get("total_bytes") != sum(int(row["bytes"]) for row in members):
    return False, "content receipt total_bytes mismatch", None
  if payload.get("path_set_sha256") != _canonical_json_sha256(paths):
    return False, "content receipt path-set digest mismatch", None
  if payload.get("aggregate_sha256") != _canonical_json_sha256(content_view):
    return False, "content receipt aggregate digest mismatch", None
  if payload.get("archive_member_metadata_set_sha256") != _canonical_json_sha256(
      archive_metadata_view
  ):
    return False, "content receipt archive member-metadata digest mismatch", None
  if payload.get("stat_fingerprint_sha256") != _canonical_json_sha256(stat_view):
    return False, "content receipt stored stat fingerprint mismatch", None
  if int(marker_state.get("step_files") or 0) != int(payload.get("step_files") or 0):
    return False, "content receipt STEP count does not match marker", None
  if int(marker_state.get("json_files") or 0) != int(payload.get("json_files") or 0):
    return False, "content receipt JSON count does not match marker", None

  try:
    current_stat_view = [
        _member_stat_view(row)
        for row in _scan_receipt_members(extract_root, hash_bytes=False)
    ]
  except RuntimeError as error:
    return False, str(error), None
  current_fingerprint = _canonical_json_sha256(current_stat_view)
  if current_fingerprint != payload.get("stat_fingerprint_sha256"):
    return False, "content receipt stat fingerprint mismatch", None
  if current_stat_view != stat_view:
    return False, "content receipt current-tree stat records mismatch", None
  try:
    summary = _content_receipt_summary(receipt_path, payload)
  except (KeyError, RuntimeError, TypeError, ValueError) as error:
    return False, f"content receipt summary is malformed: {error}", None
  return True, "ok", summary


def _lock_file_name(name: str) -> str:
  if Path(name).name != name or not name.lower().endswith(".7z"):
    raise ValueError(f"Unsafe archive name for restore lock: {name!r}")
  return f".{Path(name).stem}.restore.lock"


def _acquire_file_lock(handle: Any) -> None:
  handle.seek(0)
  if os.name == "nt":
    import msvcrt

    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
  else:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(handle: Any) -> None:
  handle.seek(0)
  if os.name == "nt":
    import msvcrt

    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
  else:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def volume_restore_lock(
    name: str,
    archive_root: str | Path,
    *,
    timeout_seconds: float = 0.0,
) -> Iterator[Path]:
  """Hold an OS-backed, per-volume lock for restore and repair operations."""

  root = _resolve_plain_directory(archive_root, label="Archive root")
  lock_path = root / _lock_file_name(name)
  _assert_safe_direct_child(
      lock_path,
      root,
      expected_name=lock_path.name,
  )
  timeout_seconds = max(0.0, float(timeout_seconds))
  deadline = time.monotonic() + timeout_seconds
  if lock_path.exists() and _is_reparse_point(lock_path):
    raise ValueError(f"Restore lock must not be a reparse point: {lock_path}")
  with lock_path.open("a+b", buffering=0) as handle:
    if lock_path.stat().st_size == 0:
      handle.write(b"\0")
      handle.flush()
    if _is_reparse_point(lock_path) or not lock_path.is_file():
      raise ValueError(f"Restore lock must be a regular file: {lock_path}")
    acquired = False
    while not acquired:
      try:
        _acquire_file_lock(handle)
        acquired = True
      except OSError as error:
        if time.monotonic() >= deadline:
          raise RuntimeError(
              f"Fusion volume restore is already locked: {name}"
          ) from error
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    try:
      yield lock_path
    finally:
      _release_file_lock(handle)


def _create_content_receipt_locked(
    name: str,
    archive_root: Path,
    *,
    integrity_spec: IntegritySpec | None,
    refresh: bool,
    seven_zip: Path | None,
    timeout_seconds: int,
    archive_member_lister: ArchiveMemberLister | None,
) -> dict[str, object]:
  extract_root = archive_root / Path(name).stem
  _assert_safe_direct_child(
      extract_root,
      archive_root,
      expected_name=Path(name).stem,
  )
  if _is_reparse_point(extract_root) or not extract_root.is_dir():
    raise RuntimeError(f"Extraction root is not a plain directory: {extract_root}")
  marker = extract_root / EXTRACTION_MARKER_NAME
  if _is_reparse_point(marker) or not marker.is_file():
    raise RuntimeError(f"Extraction marker is missing or unsafe: {marker}")
  try:
    marker_state = json.loads(marker.read_text(encoding="utf-8"))
  except (OSError, TypeError, ValueError) as error:
    raise RuntimeError(f"Extraction marker is unreadable: {marker}") from error
  if not isinstance(marker_state, dict):
    raise RuntimeError(f"Extraction marker is not a JSON object: {marker}")
  integrity = _integrity_entry(name, integrity_spec)
  required_marker = {
      "archive": name,
      "archive_bytes": int(integrity["bytes"]),
      "archive_sha256": str(integrity["sha256"]),
      "dataset_version": DATASET_VERSION,
      "status": "ok",
  }
  for key, expected in required_marker.items():
    if marker_state.get(key) != expected:
      raise RuntimeError(f"Extraction marker {key} mismatch for {name}")

  verified_archive_record = _verified_receipt_archive_record(
      name,
      archive_root,
      integrity_spec=integrity_spec,
  )
  archive_member_metadata: tuple[dict[str, object], ...] | None = None
  if integrity_spec is None or archive_member_lister is not None:
    executable = seven_zip or find_7z(None)
    member_lister = archive_member_lister or _list_7z_archive_file_members
    raw_archive_members = tuple(
        member_lister(
            archive_root / name,
            executable,
            int(timeout_seconds),
        )
    )
    archive_member_metadata = _normalize_archive_member_metadata(
        raw_archive_members
    )
    if verified_archive_record["stat"] != _stat_record(
        name,
        (archive_root / name).lstat(),
    ):
      raise RuntimeError(f"Archive changed while listing members: {name}")

  payload = _build_content_receipt_payload(
      name,
      archive_root,
      extract_root,
      integrity_spec=integrity_spec,
      archive_member_metadata=archive_member_metadata,
      verified_archive_record=verified_archive_record,
  )
  if int(marker_state.get("step_files") or 0) != int(payload["step_files"]):
    raise RuntimeError(f"Extraction marker STEP count mismatch for {name}")
  if int(marker_state.get("json_files") or 0) != int(payload["json_files"]):
    raise RuntimeError(f"Extraction marker JSON count mismatch for {name}")
  receipt_path = extract_root / CONTENT_RECEIPT_NAME
  _write_json_atomic(receipt_path, payload, replace=refresh)
  summary = _content_receipt_summary(receipt_path, payload)
  marker_state["content_receipt"] = _marker_receipt_binding(summary)
  _write_json_atomic(marker, marker_state, replace=True)
  return summary


def create_content_receipt(
    name: str,
    archive_root: str | Path,
    *,
    integrity_spec: IntegritySpec | None = None,
    refresh: bool = False,
    seven_zip: Path | None = None,
    timeout_seconds: int = DEFAULT_EXTRACTION_TIMEOUT_SECONDS,
    archive_member_lister: ArchiveMemberLister | None = None,
) -> dict[str, object]:
  """Hash all extracted STEP/JSON bytes and atomically publish a receipt."""

  root = _resolve_plain_directory(archive_root, label="Archive root")
  with volume_restore_lock(name, root):
    return _create_content_receipt_locked(
        name,
        root,
        integrity_spec=integrity_spec,
        refresh=refresh,
        seven_zip=seven_zip,
        timeout_seconds=timeout_seconds,
        archive_member_lister=archive_member_lister,
    )


def _crash_artifacts(
    archive_root: Path,
    prefix: str,
) -> list[Path]:
  artifacts = sorted(
      (path for path in archive_root.iterdir() if path.name.startswith(prefix)),
      key=lambda path: path.name,
  )
  for path in artifacts:
    _assert_safe_direct_child(path, archive_root, expected_prefix=prefix)
    if not path.is_dir():
      raise ValueError(f"Crash-recovery artifact is not a directory: {path}")
  return artifacts


def _recover_interrupted_extraction_locked(
    name: str,
    archive_root: Path,
    *,
    integrity_spec: IntegritySpec | None = None,
) -> dict[str, object]:
  stem = Path(name).stem
  target = archive_root / stem
  _assert_safe_direct_child(target, archive_root, expected_name=stem)
  staging_prefix = f".{stem}.staging-"
  backup_prefix = f".{stem}.backup-"
  staging = _crash_artifacts(archive_root, staging_prefix)
  backups = _crash_artifacts(archive_root, backup_prefix)
  if len(backups) > 1:
    raise RuntimeError(
        "Crash recovery is ambiguous: multiple backups exist for "
        f"{name}: {[path.name for path in backups]}"
    )
  if len(staging) > 1:
    raise RuntimeError(
        "Crash recovery is ambiguous: multiple staging trees exist for "
        f"{name}: {[path.name for path in staging]}"
    )

  action = "none"
  if not target.exists() and backups:
    backups[0].rename(target)
    action = "restored_backup"
  elif target.exists() and backups:
    valid_target, reason, _ = _validate_extraction_marker_locked(
        name,
        archive_root,
        deep_audit=True,
        integrity_spec=integrity_spec,
    )
    if not valid_target:
      raise RuntimeError(
          "Crash recovery is ambiguous: target and backup both exist, but "
          f"the target does not validate for {name}: {reason}"
      )
    _safe_remove_tree(
        backups[0],
        archive_root,
        expected_prefix=backup_prefix,
    )
    action = "removed_completed_backup"

  if staging:
    _safe_remove_tree(
        staging[0],
        archive_root,
        expected_prefix=staging_prefix,
    )
    if action == "none":
      action = "discarded_staging"
  return {
      "action": action,
      "archive": name,
      "target": str(target),
  }


def recover_interrupted_extraction(
    name: str,
    archive_root: str | Path,
    *,
    integrity_spec: IntegritySpec | None = None,
) -> dict[str, object]:
  """Recover one interrupted staging/backup transaction under its lock."""

  root = _resolve_plain_directory(archive_root, label="Archive root")
  with volume_restore_lock(name, root):
    return _recover_interrupted_extraction_locked(
        name,
        root,
        integrity_spec=integrity_spec,
    )


def _replace_extraction_tree(
    staging_root: Path,
    extract_root: Path,
    archive_root: Path,
) -> None:
  """Install a staged tree with rollback if the final rename fails."""
  stem = extract_root.name
  staging_prefix = f".{stem}.staging-"
  backup_prefix = f".{stem}.backup-"
  _assert_safe_direct_child(
      staging_root,
      archive_root,
      expected_prefix=staging_prefix,
  )
  _assert_safe_direct_child(
      extract_root,
      archive_root,
      expected_name=stem,
  )
  backup_root = archive_root / f"{backup_prefix}{uuid.uuid4().hex}"
  _assert_safe_direct_child(
      backup_root,
      archive_root,
      expected_prefix=backup_prefix,
  )
  had_target = extract_root.exists()
  if had_target and not extract_root.is_dir():
    raise ValueError(f"Extraction target is not a directory: {extract_root}")
  if had_target:
    extract_root.rename(backup_root)
  try:
    staging_root.rename(extract_root)
  except BaseException:
    if had_target and backup_root.exists() and not extract_root.exists():
      backup_root.rename(extract_root)
    raise
  if had_target:
    try:
      _safe_remove_tree(
          backup_root,
          archive_root,
          expected_prefix=backup_prefix,
      )
    except BaseException as error:
      raise RuntimeError(
          "New extraction is installed, but safe cleanup of the in-root "
          f"backup failed: {backup_root}"
      ) from error


def _build_extraction_state(
    name: str,
    archive_bytes: int,
    archive_sha256: str,
    extract_root: Path,
    *,
    content_receipt: Mapping[str, object],
) -> dict[str, Any]:
  marker = extract_root / EXTRACTION_MARKER_NAME
  receipt = extract_root / CONTENT_RECEIPT_NAME
  step_count = 0
  json_count = 0
  for path in extract_root.rglob("*"):
    if not path.is_file():
      continue
    if path.suffix.lower() == ".step":
      step_count += 1
    elif path.suffix.lower() == ".json" and path not in {marker, receipt}:
      json_count += 1
  if step_count == 0 or json_count == 0:
    raise RuntimeError(
        f"Extraction produced no usable assembly data for {name}: "
        f"step={step_count}, json={json_count}"
    )
  directories = _top_level_directories(extract_root)
  representative_files = _representative_files(extract_root, directories)
  if len(representative_files) < 2:
    raise RuntimeError(
        f"Extraction lacks bounded representative evidence for {name}"
    )
  return {
      "archive": name,
      "archive_bytes": archive_bytes,
      "archive_sha256": archive_sha256,
      "content_receipt": _marker_receipt_binding(content_receipt),
      "dataset_version": DATASET_VERSION,
      "evidence_schema": 1,
      "json_files": json_count,
      "representative_files": representative_files,
      "status": "ok",
      "step_files": step_count,
      "top_level_directory_count": len(directories),
  }


def _normalize_archive_member(raw_member: object) -> str:
  text = str(raw_member).strip().replace("\\", "/")
  while text.startswith("./"):
    text = text[2:]
  member = PurePosixPath(text)
  if (
      not text
      or member.is_absolute()
      or any(part in {"", ".", ".."} for part in member.parts)
      or re.match(r"^[A-Za-z]:", text)
  ):
    raise RuntimeError(f"Unsafe archive member path: {raw_member!r}")
  normalized = member.as_posix()
  if normalized in {EXTRACTION_MARKER_NAME, CONTENT_RECEIPT_NAME}:
    raise RuntimeError(f"Archive contains reserved metadata path: {normalized}")
  return normalized


def _raw_archive_member_path(raw_member: object) -> object:
  if isinstance(raw_member, Mapping):
    try:
      return raw_member["path"]
    except KeyError as error:
      raise RuntimeError("Archive member metadata is missing path") from error
  return raw_member


def _normalize_archive_file_members(raw_members: Iterable[object]) -> set[str]:
  normalized: set[str] = set()
  raw_count = 0
  for raw in raw_members:
    raw_count += 1
    member = _normalize_archive_member(_raw_archive_member_path(raw))
    if member in normalized:
      raise RuntimeError(f"Duplicate archive file member: {member}")
    normalized.add(member)
  if raw_count == 0:
    raise RuntimeError("7-Zip archive listing contains no file members")
  return normalized


def _normalize_archive_member_metadata(
    raw_members: Iterable[object],
) -> tuple[dict[str, object], ...]:
  records: list[dict[str, object]] = []
  seen_paths: set[str] = set()
  for raw_member in raw_members:
    if not isinstance(raw_member, Mapping):
      raise RuntimeError(
          "Official archive listing lacks per-file Size/CRC metadata"
      )
    path = _normalize_archive_member(_raw_archive_member_path(raw_member))
    try:
      member_bytes = int(raw_member["bytes"])
    except (KeyError, TypeError, ValueError) as error:
      raise RuntimeError(
          f"Official archive member has invalid byte size: {path}"
      ) from error
    crc32 = str(raw_member.get("crc32") or "").lower()
    if member_bytes < 0:
      raise RuntimeError(f"Official archive member has negative byte size: {path}")
    if _CRC32_RE.fullmatch(crc32) is None:
      raise RuntimeError(f"Official archive member has invalid CRC: {path}")
    if path in seen_paths:
      raise RuntimeError(f"Duplicate archive file member: {path}")
    seen_paths.add(path)
    records.append({"path": path, "bytes": member_bytes, "crc32": crc32})
  if not records:
    raise RuntimeError("Official archive listing contains no file members")
  return tuple(
      sorted(records, key=lambda row: str(row["path"]).encode("utf-8"))
  )


def _parse_7z_slt_file_members(output: str) -> tuple[dict[str, object], ...]:
  records: list[dict[str, str]] = []
  current: dict[str, str] = {}
  for raw_line in output.splitlines():
    line = raw_line.strip()
    if not line:
      if current:
        records.append(current)
        current = {}
      continue
    if " = " not in line:
      continue
    key, value = line.split(" = ", 1)
    current[key] = value
  if current:
    records.append(current)

  members: list[dict[str, object]] = []
  seen_paths: set[str] = set()
  for record in records:
    path = record.get("Path")
    if not path or "Type" in record:
      continue
    attributes = record.get("Attributes", "")
    if record.get("Folder") == "+" or attributes.startswith("D"):
      continue
    normalized_path = _normalize_archive_member(path)
    try:
      size = int(record["Size"])
    except (KeyError, TypeError, ValueError) as error:
      raise RuntimeError(
          f"7-Zip regular-file member has invalid Size: {normalized_path}"
      ) from error
    crc32 = str(record.get("CRC") or "").lower()
    if size < 0:
      raise RuntimeError(f"7-Zip member Size is negative: {normalized_path}")
    if _CRC32_RE.fullmatch(crc32) is None:
      raise RuntimeError(
          f"7-Zip regular-file member has invalid CRC: {normalized_path}"
      )
    if normalized_path in seen_paths:
      raise RuntimeError(f"Duplicate archive file member: {normalized_path}")
    seen_paths.add(normalized_path)
    members.append(
        {"path": normalized_path, "bytes": size, "crc32": crc32}
    )
  if not members:
    raise RuntimeError("7-Zip archive listing contains no file members")
  return tuple(
      sorted(members, key=lambda row: str(row["path"]).encode("utf-8"))
  )


def _list_7z_archive_file_members(
    archive: Path,
    seven_zip: Path,
    timeout_seconds: int,
) -> tuple[dict[str, object], ...]:
  try:
    completed = subprocess.run(
        [str(seven_zip), "l", "-slt", "-ba", str(archive)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
  except subprocess.TimeoutExpired as error:
    raise RuntimeError(
        f"7-Zip listing for {archive.name} timed out after {timeout_seconds} seconds"
    ) from error
  if completed.returncode != 0:
    detail = (completed.stderr or completed.stdout).strip()[-2000:]
    raise RuntimeError(
        f"7-Zip listing failed for {archive.name}: exit {completed.returncode}: "
        f"{detail}"
    )
  return _parse_7z_slt_file_members(completed.stdout)


def _extract_7z_archive(
    archive: Path,
    staging_root: Path,
    seven_zip: Path,
    timeout_seconds: int,
) -> None:
  try:
    completed = subprocess.run(
        [
            str(seven_zip),
            "x",
            str(archive),
            f"-o{staging_root}",
            "-aoa",
            "-bso0",
            "-bsp0",
        ],
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
  except subprocess.TimeoutExpired as error:
    raise RuntimeError(
        f"7-Zip extraction for {archive.name} timed out after "
        f"{timeout_seconds} seconds"
    ) from error
  if completed.returncode != 0:
    raise RuntimeError(
        f"7-Zip failed for {archive.name}: exit {completed.returncode}"
    )


def _staged_file_members(staging_root: Path) -> set[str]:
  members: set[str] = set()
  for path in staging_root.rglob("*"):
    if _is_reparse_point(path):
      raise RuntimeError(
          f"Extracted archive member must not be a symlink or junction: {path}"
      )
    if not path.is_file():
      continue
    member = _normalize_archive_member(path.relative_to(staging_root).as_posix())
    members.add(member)
  return members


def _verify_extracted_member_set(
    staging_root: Path,
    expected_members: set[str],
) -> None:
  actual_members = _staged_file_members(staging_root)
  missing = sorted(expected_members - actual_members)
  extra = sorted(actual_members - expected_members)
  if missing or extra:
    raise RuntimeError(
        "Extracted archive member set mismatch: "
        f"missing={missing[:10]}, extra={extra[:10]}"
    )


def _extract_archive_locked(
    name: str,
    archive_root: Path,
    seven_zip: Path,
    *,
    timeout_seconds: int = DEFAULT_EXTRACTION_TIMEOUT_SECONDS,
    deep_audit_marker: bool = False,
    integrity_spec: IntegritySpec | None = None,
    archive_member_lister: ArchiveMemberLister | None = None,
    archive_extractor: ArchiveExtractor | None = None,
) -> dict[str, Any]:
  archive_root = _resolve_plain_directory(archive_root, label="Archive root")
  integrity = _integrity_entry(name, integrity_spec)
  expected_archive_bytes = int(integrity["bytes"])
  expected_archive_sha256 = str(integrity["sha256"])
  archive = archive_root / name
  _assert_safe_direct_child(archive, archive_root, expected_name=name)
  if not archive.is_file():
    raise FileNotFoundError(f"Archive is missing: {archive}")
  extract_root = archive_root / Path(name).stem
  _assert_safe_direct_child(
      extract_root,
      archive_root,
      expected_name=Path(name).stem,
  )
  marker = extract_root / ".neurocad_extract_complete.json"
  if marker.is_file():
    valid, reason, state = _validate_extraction_marker_locked(
        name,
        archive_root,
        deep_audit=deep_audit_marker,
        integrity_spec=integrity_spec,
    )
    if valid and state is not None:
      if int(state.get("evidence_schema") or 0) < 1:
        directories = _top_level_directories(extract_root)
        state["evidence_schema"] = 1
        state["representative_files"] = _representative_files(
            extract_root,
            directories,
        )
        state["top_level_directory_count"] = len(directories)
        marker.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _print(f"[extract] {name}: upgraded bounded marker evidence")
      _print(f"[extract] {name}: completion marker and live evidence valid")
      return state
    _print(f"[extract] {name}: stale marker ignored ({reason})")

  verified_archive_record = _verified_receipt_archive_record(
      name,
      archive_root,
      integrity_spec=integrity_spec,
  )

  timeout_seconds = int(timeout_seconds)
  if timeout_seconds <= 0:
    raise ValueError("timeout_seconds must be positive")
  member_lister = archive_member_lister or _list_7z_archive_file_members
  extractor = archive_extractor or _extract_7z_archive
  raw_archive_members = tuple(
      member_lister(archive, seven_zip, timeout_seconds)
  )
  expected_members = _normalize_archive_file_members(raw_archive_members)
  archive_member_metadata = None
  if integrity_spec is None or all(
      isinstance(row, Mapping) for row in raw_archive_members
  ):
    archive_member_metadata = _normalize_archive_member_metadata(
        raw_archive_members
    )
  staging_prefix = f".{extract_root.name}.staging-"
  staging_root = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=archive_root))
  _assert_safe_direct_child(
      staging_root,
      archive_root,
      expected_prefix=staging_prefix,
  )
  try:
    _print(f"[extract] {name}: extracting to staging {staging_root}")
    extractor(archive, staging_root, seven_zip, timeout_seconds)
    _verify_extracted_member_set(staging_root, expected_members)
    receipt_payload = _build_content_receipt_payload(
        name,
        archive_root,
        staging_root,
        integrity_spec=integrity_spec,
        archive_member_metadata=archive_member_metadata,
        verified_archive_record=verified_archive_record,
    )
    staging_receipt = staging_root / CONTENT_RECEIPT_NAME
    _write_json_atomic(staging_receipt, receipt_payload, replace=False)
    receipt_summary = _content_receipt_summary(
        staging_receipt,
        receipt_payload,
    )
    state = _build_extraction_state(
        name,
        expected_archive_bytes,
        expected_archive_sha256,
        staging_root,
        content_receipt=receipt_summary,
    )
    staging_marker = staging_root / EXTRACTION_MARKER_NAME
    staging_marker.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_extraction_tree(staging_root, extract_root, archive_root)
    _print(
        f"[extract] {name}: step={state['step_files']}, "
        f"json={state['json_files']}"
    )
    return state
  finally:
    if staging_root.exists() or _is_reparse_point(staging_root):
      _safe_remove_tree(
          staging_root,
          archive_root,
          expected_prefix=staging_prefix,
      )


def extract_archive(
    name: str,
    archive_root: Path,
    seven_zip: Path,
    *,
    timeout_seconds: int = DEFAULT_EXTRACTION_TIMEOUT_SECONDS,
    deep_audit_marker: bool = False,
    integrity_spec: IntegritySpec | None = None,
    archive_member_lister: ArchiveMemberLister | None = None,
    archive_extractor: ArchiveExtractor | None = None,
) -> dict[str, Any]:
  """Recover and extract one volume while holding its interprocess lock."""

  root = _resolve_plain_directory(archive_root, label="Archive root")
  with volume_restore_lock(name, root):
    _recover_interrupted_extraction_locked(
        name,
        root,
        integrity_spec=integrity_spec,
    )
    return _extract_archive_locked(
        name,
        root,
        seven_zip,
        timeout_seconds=timeout_seconds,
        deep_audit_marker=deep_audit_marker,
        integrity_spec=integrity_spec,
        archive_member_lister=archive_member_lister,
        archive_extractor=archive_extractor,
    )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--archive-root",
      default=str(DEFAULT_DATASET_ARCHIVE_ROOT),
      help="Archive and per-volume extraction root.",
  )
  parser.add_argument("--limit", type=int, default=len(ARCHIVE_NAMES))
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument(
      "--download-retries",
      type=int,
      default=5,
      help="Bounded attempts per archive; retries resume the durable .part file.",
  )
  parser.add_argument("--download-only", action="store_true")
  parser.add_argument("--extract-only", action="store_true")
  parser.add_argument("--check-only", action="store_true")
  parser.add_argument(
      "--refresh-content-receipts",
      action="store_true",
      help=(
          "Explicit full-byte audit mode: hash every extracted STEP/JSON member "
          "and atomically create or refresh each per-volume content receipt."
      ),
  )
  parser.add_argument("--seven-zip", default=None)
  parser.add_argument(
      "--extraction-timeout-seconds",
      type=int,
      default=DEFAULT_EXTRACTION_TIMEOUT_SECONDS,
      help="Hard timeout for each 7-Zip extraction (default: 7200 seconds).",
  )
  parser.add_argument(
      "--deep-audit-markers",
      action="store_true",
      help="Recount every extracted STEP/JSON member when validating markers.",
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()
  if args.download_only and args.extract_only:
    raise SystemExit("--download-only and --extract-only are mutually exclusive")
  if args.refresh_content_receipts and (
      args.download_only or args.extract_only or args.check_only
  ):
    raise SystemExit(
        "--refresh-content-receipts is mutually exclusive with download/extract/check modes"
    )
  archive_root = resolve_path(args.archive_root)
  names = ARCHIVE_NAMES[: max(0, min(int(args.limit), len(ARCHIVE_NAMES)))]
  if not names:
    raise SystemExit("--limit must select at least one archive")

  if args.refresh_content_receipts:
    seven_zip = find_7z(args.seven_zip)
    receipt_rows = [
        create_content_receipt(
            name,
            archive_root,
            refresh=True,
            seven_zip=seven_zip,
            timeout_seconds=int(args.extraction_timeout_seconds),
        )
        for name in names
    ]
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "full_byte_content_receipt_refresh",
                "archive_root": str(archive_root),
                "receipt_count": len(receipt_rows),
                "receipts": receipt_rows,
                "integrity_provenance": integrity_spec_provenance(None),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return

  if args.check_only:
    with requests.Session() as session:
      rows = []
      for name in names:
        url = archive_url(name)
        remote_bytes = _remote_size(session, url)
        expected_bytes = ARCHIVE_BYTES[name]
        if remote_bytes != expected_bytes:
          raise RuntimeError(
              f"Official manifest/remote size disagreement for {name}: "
              f"{expected_bytes} != {remote_bytes}"
          )
        rows.append(
            {
                "bytes": expected_bytes,
                "name": name,
                "remote_bytes": remote_bytes,
                "expected_sha256": ARCHIVE_SHA256[name],
                "url": url,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return

  download_rows: list[dict[str, Any]] = []
  if not args.extract_only:
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
      futures = {
          executor.submit(
              download_archive_with_retries,
              name,
              archive_root,
              attempts=max(1, int(args.download_retries)),
          ): name
          for name in names
      }
      for future in as_completed(futures):
        download_rows.append(future.result())

  extract_rows: list[dict[str, Any]] = []
  if not args.download_only:
    seven_zip = find_7z(args.seven_zip)
    for name in names:
      extract_rows.append(
          extract_archive(
              name,
              archive_root,
              seven_zip,
              timeout_seconds=int(args.extraction_timeout_seconds),
              deep_audit_marker=bool(args.deep_audit_markers),
          )
      )

  state = {
      "dataset": "Fusion 360 Gallery Assembly Dataset",
      "dataset_version": DATASET_VERSION,
      "download": sorted(download_rows, key=lambda row: row["name"]),
      "extraction": sorted(extract_rows, key=lambda row: row["archive"]),
      "integrity_provenance": integrity_spec_provenance(None),
      "official_source": (
          "https://github.com/AutodeskAILab/Fusion360GalleryDataset/"
          "tree/master/tools/assembly_download"
      ),
  }
  archive_root.mkdir(parents=True, exist_ok=True)
  (archive_root / "dataset_state.json").write_text(
      json.dumps(state, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(
      json.dumps(
          {
              "status": "ok",
              "root": str(archive_root),
              "archives": len(names),
              "integrity_provenance": integrity_spec_provenance(None),
          }
      )
  )


if __name__ == "__main__":
  main()
