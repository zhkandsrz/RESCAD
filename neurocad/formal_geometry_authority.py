"""Fail-closed formal geometry authorities for benchmark-v2.

These authorities are intentionally separate from the development face-map and
contact-census receipts.  A receipt becomes usable only after the verifier has
reopened every bound byte artifact and replayed the closed case/endpoint/pair
domains.  The returned objects are ephemeral capabilities, not serializable
provenance dictionaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import functools
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
from types import MappingProxyType
from typing import Any
import weakref

from .benchmark_v2_private_supervision import (
    BenchmarkV2PrivateSupervision,
    load_benchmark_v2_private_supervision,
)
from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    CapturedJsonArtifact,
    canonical_sha256,
    capture_file_artifact,
    reverify_captured_file_artifact,
    validate_family_split_binding,
)
from .content_receipt_obj_extension import (
    AuthenticatedContentReceiptObjExtension,
    ContentReceiptObjExtensionError,
)


FACE_MAP_SCHEMA_VERSION = "fusion_step_face_map_audit.v3"
FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION = "fusion_step_face_map_audit.v4"
CONTACT_CENSUS_SCHEMA_VERSION = "fusion_contact_census_authority.v1"
ENDPOINT_LOOKUP_SCHEMA_VERSION = "fusion_step_face_endpoint_lookup.v3"
MAPPER_EVIDENCE_SCHEMA_VERSION = "fusion_face_mapper_receipt_binding.v1"
# Only the isolated live-replay mapper is eligible for formal promotion.  The
# v3 development-receipt authority remains available to pilots but is never a
# formal authority.
FORMAL_FACE_MAP_AUTHORITY_SCHEMA_ALLOWLIST: frozenset[str] = frozenset(
    {FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION}
)
FORMAL_CONTACT_CENSUS_SCHEMA_ALLOWLIST: frozenset[str] = frozenset(
    {CONTACT_CENSUS_SCHEMA_VERSION}
)
DEFAULT_CONTACT_THRESHOLD_MM = 0.1
DEFAULT_INTERFERENCE_THRESHOLD_MM3 = 1e-7

_CONTENT_RECEIPT_NAME = ".neurocad_content_receipt.json"
_FACE_SCHEMA_PATH = Path(__file__).resolve().parent / "configs" / "fusion_step_face_map_audit_v3.schema.json"
_FACE_ISOLATED_REPLAY_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "fusion_step_face_map_audit_v4.schema.json"
)
_CENSUS_SCHEMA_PATH = Path(__file__).resolve().parent / "configs" / "fusion_contact_census_authority_v1.schema.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FACE_CAPABILITY_KIND = "formal_face_map"
_CENSUS_CAPABILITY_KIND = "formal_contact_census"
_CAPABILITY_STATE_LOCK = threading.RLock()
_CAPABILITY_STATE_REGISTRY: weakref.WeakKeyDictionary[Any, str] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True, slots=True)
class _VerifiedHandleConstruction:
  """Constructor material returned only after a verifier closes its replay."""

  kwargs: Mapping[str, Any]
  memory_state_commitment: str


def _make_verifier_only_issuance_boundary():
  """Create the private one-shot issuance boundary used by verifier wrappers.

  The context class and its issuer live only in the returned wrapper closures.
  A handle never stores or returns its context.  This boundary prevents misuse
  through public/ordinary APIs; it intentionally does not claim resistance to
  private closure introspection or monkeypatching this module.
  """

  class _VerifiedIssuanceContext:
    __slots__ = ()

  pending: dict[object, tuple[str, str]] = {}
  lock = threading.RLock()

  def require(context: object, capability_kind: str) -> None:
    with lock:
      try:
        expected = pending.get(context)
      except TypeError as error:
        raise TypeError(
            "authenticated capability requires a one-shot verified issuance context"
        ) from error
      if expected is None or expected[0] != capability_kind:
        raise TypeError(
            "authenticated capability requires a one-shot verified issuance context"
        )

  def consume(context: object, capability_kind: str, commitment: str) -> None:
    with lock:
      try:
        expected = pending.pop(context)
      except (KeyError, TypeError) as error:
        raise TypeError(
            "authenticated capability requires a one-shot verified issuance context"
        ) from error
      if expected != (capability_kind, commitment):
        raise TypeError(
            "verified issuance context does not match capability kind and "
            "validated memory-state commitment"
        )

  def wrap(
      verifier: Any,
      *,
      capability_kind: str,
      constructor: Any,
  ) -> Any:
    @functools.wraps(verifier)
    def verified_issuer(*args: Any, **kwargs: Any) -> Any:
      construction = verifier(*args, **kwargs)
      if type(construction) is not _VerifiedHandleConstruction:
        raise TypeError("verifier did not return validated construction material")
      context = _VerifiedIssuanceContext()
      with lock:
        pending[context] = (
            capability_kind,
            construction.memory_state_commitment,
        )
      try:
        return constructor(
            **dict(construction.kwargs),
            _issuance_context=context,
        )
      finally:
        # Construction and memory-state registration are both fail-closed.  A
        # constructor exception or a losing concurrent consumer cannot leak a
        # future authorization.
        with lock:
          pending.pop(context, None)

    verified_issuer.__annotations__["return"] = constructor
    return verified_issuer

  def pending_count() -> int:
    with lock:
      return len(pending)

  return consume, require, wrap, pending_count


(
    _consume_verified_issuance_context,
    _require_verified_issuance_context,
    _wrap_verified_verifier,
    _pending_verified_issuance_count,
) = _make_verifier_only_issuance_boundary()


class FormalGeometryAuthorityError(RuntimeError):
  """Raised when formal geometry evidence does not close exactly."""


def _require_face_map_promotion(
    *,
    pilot_only: bool,
    schema_version: str = FACE_MAP_SCHEMA_VERSION,
) -> None:
  if (
      not pilot_only
      and schema_version not in FORMAL_FACE_MAP_AUTHORITY_SCHEMA_ALLOWLIST
  ):
    raise FormalGeometryAuthorityError(
        "formal face-map promotion is unavailable until isolated mapper replay "
        "is implemented and explicitly allowlisted"
    )


def _require_contact_census_promotion(*, pilot_only: bool) -> None:
  if (
      not pilot_only
      and CONTACT_CENSUS_SCHEMA_VERSION
      not in FORMAL_CONTACT_CENSUS_SCHEMA_ALLOWLIST
  ):
    raise FormalGeometryAuthorityError(
        "formal contact-census schema is not explicitly allowlisted"
    )


def _recursive_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType(
        {_recursive_freeze(key): _recursive_freeze(item) for key, item in value.items()}
    )
  if isinstance(value, (list, tuple)):
    return tuple(_recursive_freeze(item) for item in value)
  if isinstance(value, (set, frozenset)):
    return frozenset(_recursive_freeze(item) for item in value)
  return value


def _recursive_mutable_copy(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {
        key: _recursive_mutable_copy(item)
        for key, item in value.items()
    }
  if isinstance(value, tuple):
    return [_recursive_mutable_copy(item) for item in value]
  if isinstance(value, frozenset):
    return {_recursive_mutable_copy(item) for item in value}
  return copy.deepcopy(value)


class _ImmutableAuthenticatedCapability:
  __slots__ = ("_sealed", "__weakref__")

  def __setattr__(self, name: str, value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("authenticated authority handles are immutable")
    object.__setattr__(self, name, value)

  def _seal(self) -> None:
    object.__setattr__(self, "_sealed", True)

  def _register_memory_state(
      self,
      payload: Mapping[str, Any],
      *,
      issuance_context: object,
      capability_kind: str,
  ) -> None:
    commitment = canonical_sha256(dict(payload))
    try:
      _consume_verified_issuance_context(
          issuance_context,
          capability_kind,
          commitment,
      )
    except TypeError as error:
      raise FormalGeometryAuthorityError(
          "authenticated capability has no matching verified issuance context"
      ) from error
    with _CAPABILITY_STATE_LOCK:
      if self in _CAPABILITY_STATE_REGISTRY:
        raise FormalGeometryAuthorityError(
            "authenticated capability memory state is already registered"
        )
      _CAPABILITY_STATE_REGISTRY[self] = commitment

  def _require_memory_state(self, payload: Mapping[str, Any]) -> None:
    try:
      commitment = canonical_sha256(dict(payload))
      with _CAPABILITY_STATE_LOCK:
        expected = _CAPABILITY_STATE_REGISTRY.get(self)
    except Exception as error:
      raise FormalGeometryAuthorityError(
          "authenticated capability memory state is malformed"
      ) from error
    if expected is None or commitment != expected:
      raise FormalGeometryAuthorityError(
          "authenticated capability memory state commitment differs"
      )


@dataclass(frozen=True, slots=True)
class _StrictJsonEvidence:
  file: CapturedFileArtifact
  json: CapturedJsonArtifact


@dataclass(frozen=True, slots=True)
class _StreamCapturedFile:
  """Constant-memory capture for multi-gigabyte immutable source archives."""

  path: Path
  resolved_path: Path
  sha256: str
  byte_count: int
  device: int
  inode: int
  link_count: int
  mtime_ns: int
  ctime_ns: int


@dataclass(frozen=True, slots=True)
class _FormalInputContext:
  source_split: str
  pilot_only: bool
  case_ids: tuple[str, ...]
  input_bindings: Mapping[str, Mapping[str, Any]]
  case_material: Mapping[str, Mapping[str, Any]]
  source_pair_counts: Mapping[str, Mapping[tuple[str, str], int]]
  expected_endpoints: Mapping[str, frozenset[tuple[str, int]]]
  source_cases: Mapping[str, Mapping[str, Any]]
  gold_cases: Mapping[str, Mapping[str, Any]]
  content_members: Mapping[str, Mapping[str, Mapping[str, Any]]]
  restored_volume_roots: Mapping[str, Path]
  captures: tuple[
      tuple[str, CapturedFileArtifact | _StreamCapturedFile], ...
  ]


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON object key: {key}")
    result[key] = value
  return result


def _capture_json_strict(path: str | Path, *, label: str) -> _StrictJsonEvidence:
  captured = capture_file_artifact(path, label=label)
  try:
    payload = json.loads(
        captured.raw_bytes.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
  except (UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise FormalGeometryAuthorityError(
        f"{label} is not duplicate-key-safe UTF-8 JSON"
    ) from error
  json_capture = CapturedJsonArtifact(
      path=captured.path,
      resolved_path=captured.resolved_path,
      raw_bytes=captured.raw_bytes,
      sha256=captured.sha256,
      byte_count=captured.byte_count,
      device=captured.device,
      inode=captured.inode,
      link_count=captured.link_count,
      payload=payload,
  )
  return _StrictJsonEvidence(captured, json_capture)


def _stream_capture(path: str | Path, *, label: str) -> _StreamCapturedFile:
  candidate = Path(path)
  try:
    if candidate.is_symlink() or not candidate.is_file():
      raise FormalGeometryAuthorityError(f"{label} must be a plain file")
    resolved = candidate.resolve(strict=True)
    before = candidate.stat()
    if int(before.st_nlink) != 1:
      raise FormalGeometryAuthorityError(f"{label} must not be hard-linked")
    digest = hashlib.sha256()
    byte_count = 0
    with candidate.open("rb") as stream:
      opened_before = os.fstat(stream.fileno())
      while True:
        chunk = stream.read(8 * 1024 * 1024)
        if not chunk:
          break
        digest.update(chunk)
        byte_count += len(chunk)
      opened_after = os.fstat(stream.fileno())
    after = candidate.stat()
    identities = [
        (
            int(value.st_dev), int(value.st_ino), int(value.st_size),
            int(value.st_nlink),
        )
        for value in (before, opened_before, opened_after, after)
    ]
    if (
        any(identity != identities[0] for identity in identities[1:])
        or byte_count != int(after.st_size)
        or candidate.is_symlink()
        or candidate.resolve(strict=True) != resolved
    ):
      raise FormalGeometryAuthorityError(
          f"{label} changed during streaming capture"
      )
  except FormalGeometryAuthorityError:
    raise
  except OSError as error:
    raise FormalGeometryAuthorityError(f"{label} is missing or unreadable") from error
  return _StreamCapturedFile(
      path=candidate,
      resolved_path=resolved,
      sha256=digest.hexdigest(),
      byte_count=byte_count,
      device=int(after.st_dev),
      inode=int(after.st_ino),
      link_count=int(after.st_nlink),
      mtime_ns=int(after.st_mtime_ns),
      ctime_ns=int(after.st_ctime_ns),
  )


def _file_binding(
    captured: CapturedFileArtifact | _StreamCapturedFile,
) -> dict[str, Any]:
  return {
      "path": str(captured.resolved_path),
      "bytes": captured.byte_count,
      "sha256": captured.sha256,
  }


def _require_exact_file_binding(
    value: Any,
    captured: CapturedFileArtifact | _StreamCapturedFile,
    *,
    label: str,
) -> None:
  if not isinstance(value, Mapping) or dict(value) != _file_binding(captured):
    raise FormalGeometryAuthorityError(f"{label} file binding differs")


def _plain_directory(path: str | Path, *, label: str) -> Path:
  try:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
      raise FormalGeometryAuthorityError(f"{label} must be a plain directory")
    return candidate.resolve(strict=True)
  except FormalGeometryAuthorityError:
    raise
  except (OSError, RuntimeError) as error:
    raise FormalGeometryAuthorityError(f"{label} is missing") from error


def _safe_member(root: Path, relative: Any, *, suffix: str | None = None) -> Path:
  if not isinstance(relative, str) or not relative or "\\" in relative:
    raise FormalGeometryAuthorityError("dataset member path is unsafe")
  posix = PurePosixPath(relative)
  if (
      posix.is_absolute()
      or posix.as_posix() != relative
      or any(part in {"", ".", ".."} for part in posix.parts)
      or (suffix is not None and posix.suffix.lower() != suffix.lower())
  ):
    raise FormalGeometryAuthorityError("dataset member path is unsafe")
  candidate = root.joinpath(*posix.parts)
  try:
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
  except (OSError, ValueError) as error:
    raise FormalGeometryAuthorityError("dataset member escapes or is missing") from error
  current = candidate
  while current != root:
    if current.is_symlink():
      raise FormalGeometryAuthorityError("dataset member crosses a symlink")
    current = current.parent
  return resolved


def _sha1(captured: CapturedFileArtifact) -> str:
  return hashlib.sha1(captured.raw_bytes).hexdigest()


def _transform_payload(transform: Any) -> dict[str, Any]:
  return {
      "rotation_row_major": [
          float(value) for value in transform.rotation.reshape(-1)
      ],
      "translation_mm": [float(value) for value in transform.translation],
  }


def _implementation_paths(*, authority_kind: str) -> dict[str, Path]:
  root = Path(__file__).resolve().parent
  paths = {
      "neurocad.formal_geometry_authority": Path(__file__),
      "neurocad.benchmark_v2_training_provenance": root / "benchmark_v2_training_provenance.py",
      "neurocad.benchmark_v2_private_supervision": root / "benchmark_v2_private_supervision.py",
      "neurocad.fusion_contact_auditor": root / "fusion_contact_auditor.py",
      "neurocad.cadquery_backend": root / "cadquery_backend.py",
  }
  if authority_kind in {"face_map", "face_map_isolated_replay"}:
    paths.update(
        {
            "neurocad.fusion_face_mapper": root / "fusion_face_mapper.py",
            "neurocad.tools.audit_fusion_step_face_map": (
                root / "tools" / "audit_fusion_step_face_map.py"
            ),
            "neurocad.tools.restore_fusion_assembly_dataset": (
                root / "tools" / "restore_fusion_assembly_dataset.py"
            ),
        }
    )
    if authority_kind == "face_map_isolated_replay":
      paths["neurocad.isolated_face_mapper_replay"] = (
          root / "isolated_face_mapper_replay.py"
      )
  if authority_kind == "contact_census":
    paths["neurocad.fusion_contact_census"] = root / "fusion_contact_census.py"
  elif authority_kind not in {"face_map", "face_map_isolated_replay"}:
    raise ValueError("formal authority implementation kind is invalid")
  return paths


def _implementation_binding(
    schema_path: Path,
    *,
    authority_kind: str,
) -> dict[str, Any]:
  sources = {
      name: capture_file_artifact(path, label=f"formal authority source {name}")
      for name, path in _implementation_paths(authority_kind=authority_kind).items()
  }
  schema = capture_file_artifact(schema_path, label="formal authority schema")
  return {
      "module": "neurocad.formal_geometry_authority",
      "source_sha256s": {
          name: captured.sha256 for name, captured in sorted(sources.items())
      },
      "schema_sha256": schema.sha256,
  }


def _validate_implementation_binding(
    value: Any,
    *,
    schema_path: Path,
    authority_kind: str,
    captures: list[tuple[str, CapturedFileArtifact]],
) -> None:
  sources = {
      name: capture_file_artifact(path, label=f"formal authority source {name}")
      for name, path in _implementation_paths(authority_kind=authority_kind).items()
  }
  schema = capture_file_artifact(schema_path, label="formal authority schema")
  captures.extend(
      (f"formal authority source {name}", captured)
      for name, captured in sorted(sources.items())
  )
  captures.append(("formal authority schema", schema))
  expected = {
      "module": "neurocad.formal_geometry_authority",
      "source_sha256s": {
          name: captured.sha256 for name, captured in sorted(sources.items())
      },
      "schema_sha256": schema.sha256,
  }
  if not isinstance(value, Mapping) or dict(value) != expected:
    raise FormalGeometryAuthorityError("formal authority implementation binding differs")


def _public_case_ids(payload: Any) -> tuple[str, ...]:
  if not isinstance(payload, list) or not payload:
    raise FormalGeometryAuthorityError("public family split cases are missing")
  case_ids: list[str] = []
  for row in payload:
    if not isinstance(row, Mapping):
      raise FormalGeometryAuthorityError("public family split case is malformed")
    case_id = row.get("id")
    if not isinstance(case_id, str) or not case_id or case_id in case_ids:
      raise FormalGeometryAuthorityError("public family split case IDs are invalid")
    case_ids.append(case_id)
  return tuple(case_ids)


def _content_members(payload: Any, *, archive: str) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
  if not isinstance(payload, Mapping):
    raise FormalGeometryAuthorityError("content receipt is not an object")
  archive_row = payload.get("archive")
  raw_members = payload.get("members")
  provenance = payload.get("integrity_provenance")
  if (
      payload.get("schema_version") != 2
      or payload.get("dataset_version") != "a1.0.0"
      or payload.get("archive_member_metadata_authenticated") is not True
      or not isinstance(provenance, Mapping)
      or provenance.get("official") is not True
      or not isinstance(archive_row, Mapping)
      or archive_row.get("name") != archive
      or not isinstance(raw_members, list)
      or payload.get("member_count") != len(raw_members)
  ):
    raise FormalGeometryAuthorityError("content receipt lacks formal archive authority")
  members: dict[str, Mapping[str, Any]] = {}
  for row in raw_members:
    if not isinstance(row, Mapping):
      raise FormalGeometryAuthorityError("content receipt member is malformed")
    path = row.get("path")
    size = row.get("bytes")
    digest = row.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or path in members
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
    ):
      raise FormalGeometryAuthorityError("content receipt member binding is malformed")
    members[path] = row
  return archive_row, members


def _capture_member(
    volume_root: Path,
    members: Mapping[str, Mapping[str, Any]],
    binding: Any,
    *,
    label: str,
    suffix: str,
) -> CapturedFileArtifact:
  if not isinstance(binding, Mapping):
    raise FormalGeometryAuthorityError(f"{label} source binding is malformed")
  relative = binding.get("path")
  path = _safe_member(volume_root, relative, suffix=suffix)
  captured = capture_file_artifact(path, label=label)
  member = members.get(str(relative))
  expected = {
      "bytes": captured.byte_count,
      "sha256": captured.sha256,
  }
  if (
      not isinstance(member, Mapping)
      or member.get("bytes") != expected["bytes"]
      or member.get("sha256") != expected["sha256"]
      or binding.get("bytes") != expected["bytes"]
      or binding.get("sha256") != expected["sha256"]
  ):
    raise FormalGeometryAuthorityError(f"{label} differs from source/content receipts")
  return captured


def _load_formal_inputs(
    *,
    family_split_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    source_split: str,
    pilot_only: bool = False,
) -> _FormalInputContext:
  split = str(source_split).strip().lower()
  if split not in {"train", "dev"}:
    raise FormalGeometryAuthorityError("formal authority split must be train or dev")
  captures: list[
      tuple[str, CapturedFileArtifact | _StreamCapturedFile]
  ] = []
  manifest = _capture_json_strict(family_split_manifest_path, label="family split manifest")
  public = _capture_json_strict(public_cases_path, label="public family split cases")
  source = _capture_json_strict(private_source_path, label="private source split")
  gold = _capture_json_strict(private_gold_path, label="private evaluation gold")
  captures.extend(
      (
          ("family split manifest", manifest.file),
          ("public family split cases", public.file),
          ("private source split", source.file),
          ("private evaluation gold", gold.file),
      )
  )
  case_ids = _public_case_ids(public.json.payload)
  try:
    validate_family_split_binding(
        cases_path=public_cases_path,
        manifest_path=family_split_manifest_path,
        source_split=split,
        case_ids=case_ids,
        _cases_capture=public.file,
        _manifest_capture=manifest.json,
        pilot_only=pilot_only,
    )
    supervision: BenchmarkV2PrivateSupervision = load_benchmark_v2_private_supervision(
        source_path=private_source_path,
        gold_path=private_gold_path,
        family_split_manifest_path=family_split_manifest_path,
        source_split=split,
        public_case_ids=case_ids,
        _source_capture=source.json,
        _gold_capture=gold.json,
        _family_manifest_capture=manifest.json,
        pilot_only=pilot_only,
    )
  except (TypeError, ValueError) as error:
    raise FormalGeometryAuthorityError("formal split/private supervision replay failed") from error

  archive_dir = _plain_directory(archive_root, label="archive root")
  restored_dir = _plain_directory(restored_root, label="restored dataset root")
  case_material: dict[str, Mapping[str, Any]] = {}
  pair_counts_by_case: dict[str, Mapping[tuple[str, str], int]] = {}
  endpoints_by_case: dict[str, frozenset[tuple[str, int]]] = {}
  source_cases: dict[str, Mapping[str, Any]] = {}
  gold_cases: dict[str, Mapping[str, Any]] = {}
  content_members_by_case: dict[str, Mapping[str, Mapping[str, Any]]] = {}
  restored_volume_roots: dict[str, Path] = {}
  auditor = __import__("neurocad.fusion_contact_auditor", fromlist=["reconstruct_instance_transforms"])
  archive_captures: dict[Path, _StreamCapturedFile] = {}

  for case_id in case_ids:
    if pilot_only:
      print(
          json.dumps(
              {"event": "pilot_input_replay", "case_id": case_id},
              sort_keys=True,
          ),
          file=sys.stderr,
          flush=True,
      )
    source_case = supervision.source_case(case_id)
    gold_case = supervision.gold_case(case_id)
    source_cases[case_id] = MappingProxyType(copy.deepcopy(source_case))
    gold_cases[case_id] = MappingProxyType(copy.deepcopy(gold_case))
    binding = source_case["source_receipt_binding"]
    archive = str(binding["archive"])
    archive_path = archive_dir / archive
    archive_capture = archive_captures.get(archive_path)
    if archive_capture is None:
      archive_capture = _stream_capture(
          archive_path,
          label=f"case {case_id} source archive",
      )
      archive_captures[archive_path] = archive_capture
      captures.append((f"source archive {archive}", archive_capture))
    if pilot_only:
      print(json.dumps({"event": "pilot_archive_bound", "case_id": case_id}), file=sys.stderr, flush=True)
    volume_root = (restored_dir / Path(archive).stem).resolve(strict=True)
    try:
      volume_root.relative_to(restored_dir)
    except ValueError as error:
      raise FormalGeometryAuthorityError("restored volume escapes dataset root") from error
    content = _capture_json_strict(
        volume_root / _CONTENT_RECEIPT_NAME,
        label=f"case {case_id} content receipt",
    )
    if pilot_only:
      print(json.dumps({"event": "pilot_content_receipt_bound", "case_id": case_id}), file=sys.stderr, flush=True)
    if content.file.sha256 != binding["receipt_sha256"]:
      raise FormalGeometryAuthorityError("private source content receipt binding differs")
    archive_row, members = _content_members(content.json.payload, archive=archive)
    content_members_by_case[case_id] = MappingProxyType(
        {key: MappingProxyType(copy.deepcopy(dict(value))) for key, value in members.items()}
    )
    restored_volume_roots[case_id] = volume_root
    if (
        archive_row.get("bytes") != archive_capture.byte_count
        or archive_row.get("sha256") != archive_capture.sha256
    ):
      raise FormalGeometryAuthorityError("live archive differs from content receipt")
    assembly_capture = _capture_member(
        volume_root,
        members,
        binding["assembly_json"],
        label=f"case {case_id} assembly JSON",
        suffix=".json",
    )
    if pilot_only:
      print(json.dumps({"event": "pilot_assembly_bound", "case_id": case_id}), file=sys.stderr, flush=True)
    try:
      assembly = json.loads(
          assembly_capture.raw_bytes.decode("utf-8"),
          object_pairs_hook=_reject_duplicate_keys,
      )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
      raise FormalGeometryAuthorityError("assembly JSON is not duplicate-key-safe") from error
    if not isinstance(assembly, Mapping):
      raise FormalGeometryAuthorityError("assembly JSON is not an object")
    instances = binding["instances"]
    try:
      certificates = auditor.reconstruct_instance_transforms(assembly, instances)
    except Exception as error:
      raise FormalGeometryAuthorityError("assembly instance transform replay failed") from error
    if pilot_only:
      print(json.dumps({"event": "pilot_transforms_replayed", "case_id": case_id}), file=sys.stderr, flush=True)
    steps_by_part = {str(row["part"]): row for row in binding["body_steps"]}
    raw_bodies = assembly.get("bodies")
    if not isinstance(raw_bodies, Mapping):
      raise FormalGeometryAuthorityError("assembly bodies are missing")
    step_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    for part in sorted(certificates):
      certificate = certificates[part]
      step = steps_by_part.get(part)
      body = raw_bodies.get(certificate.body_uuid)
      if not isinstance(step, Mapping) or not isinstance(body, Mapping):
        raise FormalGeometryAuthorityError("assembly/source body instance differs")
      step_capture = _capture_member(
          volume_root,
          members,
          step,
          label=f"case {case_id} STEP {part}",
          suffix=".step",
      )
      if (
          step.get("sha1") != _sha1(step_capture)
          or body.get("step") != Path(str(step["path"])).name
      ):
        raise FormalGeometryAuthorityError("STEP SHA1 or assembly body binding differs")
      source_instance = next(row for row in instances if row["part"] == part)
      step_rows.append(
          {
              "part": part,
              "geometry_asset": certificate.geometry_asset,
              "body_uuid": certificate.body_uuid,
              "file": _file_binding(step_capture),
              "source_relative_path": step["path"],
              "sha1": step["sha1"],
          }
      )
      instance_rows.append(
          {
              "part": part,
              "geometry_asset": certificate.geometry_asset,
              "body_uuid": certificate.body_uuid,
              "source_instance_key": copy.deepcopy(dict(certificate.source_instance_key)),
              "visible": source_instance.get("is_visible") is True,
              "world_transform_mm": _transform_payload(certificate.world_transform),
              "world_transform_chain_sha256": certificate.transform_chain_sha256,
          }
      )
      captures.append((f"case {case_id} STEP {part}", step_capture))
    if pilot_only:
      print(json.dumps({"event": "pilot_steps_bound", "case_id": case_id}), file=sys.stderr, flush=True)
    gold_contacts = gold_case.get("contacts")
    if not isinstance(gold_contacts, list) or not gold_contacts:
      raise FormalGeometryAuthorityError("formal case has no source-positive contacts")
    pair_counts: Counter[tuple[str, str]] = Counter()
    endpoints: set[tuple[str, int]] = set()
    visible_parts = {row["part"] for row in instance_rows if row["visible"] is True}
    for contact in gold_contacts:
      if not isinstance(contact, Mapping):
        raise FormalGeometryAuthorityError("private gold contact is malformed")
      parts: list[str] = []
      for role in ("a", "b"):
        endpoint = contact.get(f"endpoint_{role}")
        if not isinstance(endpoint, Mapping):
          raise FormalGeometryAuthorityError("private gold endpoint is malformed")
        part = str(endpoint.get("part") or "")
        face_index = endpoint.get("fusion_face_index")
        if part not in visible_parts or isinstance(face_index, bool) or not isinstance(face_index, int):
          raise FormalGeometryAuthorityError("source-positive endpoint is not a visible instance")
        endpoints.add((part, face_index))
        parts.append(part)
      pair_counts[tuple(sorted(parts))] += 1
    captures.extend(
        (
            (f"case {case_id} content receipt", content.file),
            (f"case {case_id} assembly JSON", assembly_capture),
        )
    )
    case_material[case_id] = {
        "archive": _file_binding(archive_capture),
        "content_receipt": _file_binding(content.file),
        "assembly_json": _file_binding(assembly_capture),
        "steps": step_rows,
        "instances": instance_rows,
    }
    pair_counts_by_case[case_id] = MappingProxyType(dict(pair_counts))
    endpoints_by_case[case_id] = frozenset(endpoints)

  input_bindings = {
      "family_split_manifest": _file_binding(manifest.file),
      "public_cases": _file_binding(public.file),
      "private_source": _file_binding(source.file),
      "private_gold": _file_binding(gold.file),
  }
  return _FormalInputContext(
      source_split=split,
      pilot_only=bool(pilot_only),
      case_ids=case_ids,
      input_bindings=MappingProxyType(input_bindings),
      case_material=MappingProxyType(case_material),
      source_pair_counts=MappingProxyType(pair_counts_by_case),
      expected_endpoints=MappingProxyType(endpoints_by_case),
      source_cases=MappingProxyType(source_cases),
      gold_cases=MappingProxyType(gold_cases),
      content_members=MappingProxyType(content_members_by_case),
      restored_volume_roots=MappingProxyType(restored_volume_roots),
      captures=tuple(captures),
  )


def _validate_occ_identity(indices: Any, signatures: Any) -> tuple[list[int], list[str]]:
  if (
      not isinstance(indices, list)
      or not indices
      or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices)
      or indices != sorted(set(indices))
      or not isinstance(signatures, list)
      or len(signatures) != len(indices)
      or any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in signatures)
  ):
    raise FormalGeometryAuthorityError("mapped endpoint OCC identity is malformed")
  return list(indices), list(signatures)


_MAPPER_IMPLEMENTATION_KEYS = (
    "neurocad.cadquery_backend",
    "neurocad.fusion_face_mapper",
    "neurocad.tools.audit_fusion_step_face_map",
    "neurocad.tools.restore_fusion_assembly_dataset",
)
_SOUND_MAPPER_COVERAGE_MODE = (
    "complete_chunked_forward_plus_bidirectional_occ_tessellation"
)


def _mapper_implementation_sha256s() -> dict[str, str]:
  paths = _implementation_paths(authority_kind="face_map")
  return {
      key: capture_file_artifact(
          paths[key], label=f"face mapper implementation {key}"
      ).sha256
      for key in _MAPPER_IMPLEMENTATION_KEYS
  }


def _mapper_number(value: Any, *, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise FormalGeometryAuthorityError(f"mapper receipt {label} is not numeric")
  result = float(value)
  if not math.isfinite(result) or result < 0.0:
    raise FormalGeometryAuthorityError(
        f"mapper receipt {label} is not finite and nonnegative"
    )
  return result


def _require_exact_keys(
    value: Any,
    expected_keys: set[str] | frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != set(expected_keys):
    raise FormalGeometryAuthorityError(f"{label} has unexpected fields")
  return value


def _require_strict_int_fields(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    label: str,
) -> None:
  for field in fields:
    if type(value.get(field)) is not int:
      raise FormalGeometryAuthorityError(
          f"{label} {field} has invalid integer type"
      )


def _require_strict_bool_fields(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    label: str,
) -> None:
  for field in fields:
    if type(value.get(field)) is not bool:
      raise FormalGeometryAuthorityError(
          f"{label} {field} has invalid boolean type"
      )


def _require_counter_mapping(value: Any, *, label: str) -> None:
  if not isinstance(value, Mapping) or any(
      not isinstance(key, str) or type(count) is not int or count < 0
      for key, count in value.items()
  ):
    raise FormalGeometryAuthorityError(f"{label} has invalid count type")


def _require_sound_mapper_mapping_evidence(
    mapping: Mapping[str, Any],
    *,
    tolerances: Mapping[str, Any],
    analytic_margin_proof_replayed: bool = False,
) -> None:
  """Validate the receipt's closed proof ledger without self-asserting a face.

  The full OCC/OBJ computation remains the development mapper's job.  Formal
  authority binds that immutable output and independently checks that its
  success row actually carries the locked ranking, margin, bidirectional
  coverage, boundary and work-completion predicates used by the sound mapper.
  """

  checks = mapping.get("checks")
  if not isinstance(checks, Mapping):
    raise FormalGeometryAuthorityError("mapper receipt mapping lacks proof checks")
  if checks.get("coverage_mode") != _SOUND_MAPPER_COVERAGE_MODE:
    raise FormalGeometryAuthorityError("mapper receipt coverage mode is not formal-safe")
  if _mapper_number(checks.get("obj_sample_coverage"), label="OBJ sample coverage") != 1.0:
    raise FormalGeometryAuthorityError("mapper receipt OBJ sample domain is incomplete")
  if _mapper_number(
      checks.get("maximum_best_distance_mm"), label="maximum best distance"
  ) > _mapper_number(tolerances.get("sample_distance_mm"), label="sample tolerance"):
    raise FormalGeometryAuthorityError("mapper receipt forward distance proof exceeds tolerance")
  if _mapper_number(
      checks.get("maximum_obj_to_occ_boundary_distance_mm"),
      label="OBJ-to-OCC boundary distance",
  ) > _mapper_number(tolerances.get("boundary_distance_mm"), label="boundary tolerance"):
    raise FormalGeometryAuthorityError("mapper receipt forward boundary proof exceeds tolerance")
  if _mapper_number(
      checks.get("maximum_occ_to_obj_boundary_distance_mm"),
      label="OCC-to-OBJ boundary distance",
  ) > _mapper_number(tolerances.get("boundary_distance_mm"), label="boundary tolerance"):
    raise FormalGeometryAuthorityError("mapper receipt reverse boundary proof exceeds tolerance")

  raw_indices, _ = _validate_occ_identity(
      mapping.get("raw_occ_face_indices"),
      mapping.get("source_face_signature_sha256s"),
  )
  if checks.get("forward_assigned_occ_face_indices") != raw_indices or checks.get(
      "reverse_covered_occ_face_indices"
  ) != raw_indices:
    raise FormalGeometryAuthorityError(
        "mapper receipt forward/reverse OCC face domains differ"
    )
  margin = checks.get("minimum_second_best_margin_mm")
  forward_work = checks.get("forward_distance_work")
  reverse_work = checks.get("reverse_distance_work")
  if not isinstance(forward_work, Mapping) or not isinstance(reverse_work, Mapping):
    raise FormalGeometryAuthorityError("mapper receipt proof work ledgers are missing")
  candidate_count = forward_work.get("candidate_occ_face_count")
  if margin is None:
    if candidate_count != 1:
      raise FormalGeometryAuthorityError(
          "mapper receipt omits margin for a multi-candidate ranking"
      )
  elif _mapper_number(margin, label="minimum second-best margin") < _mapper_number(
      tolerances.get("minimum_sample_margin_mm"), label="minimum margin tolerance"
  ) and not analytic_margin_proof_replayed:
    raise FormalGeometryAuthorityError("mapper receipt ranking margin is insufficient")
  if (
      forward_work.get("schema_version") != "occ_point_face_distance_call_work.v4"
      or forward_work.get("all_forward_samples_checked") is not True
      or forward_work.get("work_cap_exhausted") is not False
      or reverse_work.get("schema_version") != "exact_triangle_aabb_bvh_work.v1"
      or reverse_work.get("all_reverse_samples_checked") is not True
      or reverse_work.get("work_cap_exhausted") is not False
  ):
    raise FormalGeometryAuthorityError("mapper receipt proof work did not close")
  if (
      checks.get("curved_chord_proof_sample_count") != 0
      or checks.get("curved_chord_proofs") != []
      or _mapper_number(
          checks.get("maximum_curved_chord_error_upper_bound_mm"),
          label="curved chord upper bound",
      ) != 0.0
  ):
    raise FormalGeometryAuthorityError(
        "mapper receipt uses the invalidated curved-chord proof route"
    )
  signature = checks.get("obj_group_signature_sha256")
  if not isinstance(signature, str) or _SHA256_RE.fullmatch(signature) is None:
    raise FormalGeometryAuthorityError("mapper receipt OBJ group signature is missing")


def _unit_vector3(value: Any, *, label: str) -> tuple[float, float, float]:
  vector = tuple(float(item) for item in value)
  if len(vector) != 3 or any(not math.isfinite(item) for item in vector):
    raise FormalGeometryAuthorityError(f"{label} is not a finite 3-vector")
  norm = math.sqrt(sum(item * item for item in vector))
  if not math.isfinite(norm) or norm <= 0.0:
    raise FormalGeometryAuthorityError(f"{label} has zero norm")
  return tuple(item / norm for item in vector)


def _dot3(left: Sequence[float], right: Sequence[float]) -> float:
  return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub3(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
  return tuple(float(a) - float(b) for a, b in zip(left, right))


def _cross3(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
  return (
      float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
      float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
      float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
  )


def _norm3(value: Sequence[float]) -> float:
  return math.sqrt(sum(float(item) * float(item) for item in value))


def _formal_analytic_support(face: Any) -> tuple[str, tuple[Any, ...]]:
  """Read a regular analytic support without calling mapper proof helpers."""

  try:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
    from OCP.GeomAbs import (  # type: ignore
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
    )

    adaptor = BRepAdaptor_Surface(face.wrapped)
    surface_type = adaptor.GetType()
    if surface_type == GeomAbs_Plane:
      plane = adaptor.Plane()
      direction = plane.Axis().Direction()
      location = plane.Location()
      normal = _unit_vector3(
          (direction.X(), direction.Y(), direction.Z()),
          label="formal analytic plane normal",
      )
      point = (float(location.X()), float(location.Y()), float(location.Z()))
      return "PLANE", (normal, point)
    if surface_type == GeomAbs_Cylinder:
      cylinder = adaptor.Cylinder()
      direction = cylinder.Axis().Direction()
      location = cylinder.Axis().Location()
      axis = _unit_vector3(
          (direction.X(), direction.Y(), direction.Z()),
          label="formal analytic cylinder axis",
      )
      origin = (float(location.X()), float(location.Y()), float(location.Z()))
      radius = float(cylinder.Radius())
      if not math.isfinite(radius) or radius < 0.0:
        raise FormalGeometryAuthorityError("formal analytic cylinder radius is invalid")
      return "CYLINDER", (axis, origin, radius)
    if surface_type == GeomAbs_Sphere:
      sphere = adaptor.Sphere()
      location = sphere.Location()
      center = (float(location.X()), float(location.Y()), float(location.Z()))
      radius = float(sphere.Radius())
      if not math.isfinite(radius) or radius < 0.0:
        raise FormalGeometryAuthorityError("formal analytic sphere radius is invalid")
      return "SPHERE", (center, radius)
  except FormalGeometryAuthorityError:
    raise
  except Exception as error:
    raise FormalGeometryAuthorityError(
        "formal analytic STEP support cannot be replayed"
    ) from error
  raise FormalGeometryAuthorityError(
      "formal analytic proof references a non-regular support"
  )


def _formal_supports_are_same(
    reference: tuple[str, tuple[Any, ...]],
    candidate: tuple[str, tuple[Any, ...]],
    *,
    tolerance_mm: float,
) -> bool:
  reference_type, reference_values = reference
  candidate_type, candidate_values = candidate
  if reference_type != candidate_type:
    return False
  if reference_type == "PLANE":
    normal_a, point_a = reference_values
    normal_b, point_b = candidate_values
    if abs(_dot3(normal_a, normal_b)) < 1.0 - 1e-8:
      return False
    return abs(_dot3(normal_a, _sub3(point_b, point_a))) <= tolerance_mm
  if reference_type == "CYLINDER":
    axis_a, origin_a, radius_a = reference_values
    axis_b, origin_b, radius_b = candidate_values
    if abs(_dot3(axis_a, axis_b)) < 1.0 - 1e-8:
      return False
    axis_distance = _norm3(_cross3(_sub3(origin_b, origin_a), axis_a))
    return axis_distance <= tolerance_mm and abs(radius_a - radius_b) <= tolerance_mm
  if reference_type == "SPHERE":
    center_a, radius_a = reference_values
    center_b, radius_b = candidate_values
    return _norm3(_sub3(center_b, center_a)) <= tolerance_mm and abs(
        radius_a - radius_b
    ) <= tolerance_mm
  raise FormalGeometryAuthorityError("formal analytic support type is unsupported")


def _formal_mapper_analytic_margin_proof_replay(
    mapper_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    context: _FormalInputContext,
    tolerances: Mapping[str, Any],
) -> tuple[frozenset[tuple[str, str, int]], dict[str, Any]]:
  """Independently replay every low-margin analytic support exclusion."""

  locked_margin = _mapper_number(
      tolerances.get("minimum_sample_margin_mm"),
      label="minimum margin tolerance",
  )
  support_tolerance = _mapper_number(
      tolerances.get("analytic_support_distance_mm"),
      label="analytic support tolerance",
  )
  backend = __import__(
      "neurocad.cadquery_backend",
      fromlist=["load_step_shape"],
  )
  shape_cache: dict[tuple[str, str], list[Any]] = {}
  replayed_mappings: set[tuple[str, str, int]] = set()
  replay_rows: list[dict[str, Any]] = []
  for identity, mapping in sorted(mapper_lookup.items()):
    checks = mapping.get("checks")
    if not isinstance(checks, Mapping):
      raise FormalGeometryAuthorityError("live mapper mapping lacks proof checks")
    raw_margin = checks.get("minimum_second_best_margin_mm")
    if raw_margin is None or _mapper_number(
        raw_margin,
        label="minimum second-best margin",
    ) >= locked_margin:
      continue
    case_id, part, _fusion_index = identity
    step_by_part = {
        row["part"]: row for row in context.case_material[case_id]["steps"]
    }
    step = step_by_part.get(part)
    if not isinstance(step, Mapping):
      raise FormalGeometryAuthorityError("formal analytic replay STEP is missing")
    cache_key = (case_id, part)
    faces = shape_cache.get(cache_key)
    if faces is None:
      try:
        shape = backend.load_step_shape(Path(step["file"]["path"]))
        faces = list(shape.Faces())
      except Exception as error:
        raise FormalGeometryAuthorityError(
            "formal analytic replay cannot import the live STEP"
        ) from error
      shape_cache[cache_key] = faces
    proofs = checks.get("analytic_margin_disambiguation_proofs")
    claimed_count = checks.get("analytic_margin_disambiguated_sample_count")
    if (
        not isinstance(proofs, list)
        or not proofs
        or type(claimed_count) is not int
        or claimed_count != len(proofs)
    ):
      raise FormalGeometryAuthorityError(
          "low-margin mapping lacks a complete analytic proof ledger"
      )
    raw_indices, _ = _validate_occ_identity(
        mapping.get("raw_occ_face_indices"),
        mapping.get("source_face_signature_sha256s"),
    )
    proof_margins: list[float] = []
    for ordinal, proof in enumerate(proofs):
      proof = _require_exact_keys(
          proof,
          {
              "candidate_occ_face_index", "reference_occ_face_indices",
              "surface_type", "support_relation",
              "analytic_support_distance_tolerance_mm", "proof",
              "best_occ_face_index", "best_distance_mm",
              "second_best_distance_lower_bound_mm",
              "sample_margin_lower_bound_mm",
              "locked_minimum_sample_margin_mm",
          },
          label="formal analytic margin proof",
      )
      best_index = proof.get("best_occ_face_index")
      candidate_index = proof.get("candidate_occ_face_index")
      if (
          type(best_index) is not int
          or type(candidate_index) is not int
          or best_index not in raw_indices
          or candidate_index == best_index
          or min(best_index, candidate_index) < 0
          or max(best_index, candidate_index) >= len(faces)
          or proof.get("reference_occ_face_indices") != [best_index]
          or proof.get("support_relation") != "provably_distinct"
          or proof.get("proof")
          != "distinct_same_type_regular_analytic_supports_have_zero_area_intersection"
          or proof.get("analytic_support_distance_tolerance_mm") != support_tolerance
          or proof.get("locked_minimum_sample_margin_mm") != locked_margin
      ):
        raise FormalGeometryAuthorityError(
            "formal analytic margin proof identity/contract differs"
        )
      best_distance = _mapper_number(
          proof.get("best_distance_mm"), label="analytic best distance"
      )
      second_distance = _mapper_number(
          proof.get("second_best_distance_lower_bound_mm"),
          label="analytic second-best distance",
      )
      claimed_margin = _mapper_number(
          proof.get("sample_margin_lower_bound_mm"),
          label="analytic sample margin",
      )
      difference = max(0.0, second_distance - best_distance)
      expected_margin = (
          0.0 if difference == 0.0 else max(0.0, math.nextafter(difference, -math.inf))
      )
      if claimed_margin != expected_margin or claimed_margin >= locked_margin:
        raise FormalGeometryAuthorityError(
            "formal analytic margin arithmetic does not replay"
        )
      best_support = _formal_analytic_support(faces[best_index])
      candidate_support = _formal_analytic_support(faces[candidate_index])
      if (
          proof.get("surface_type") != best_support[0]
          or candidate_support[0] != best_support[0]
          or _formal_supports_are_same(
              best_support,
              candidate_support,
              tolerance_mm=support_tolerance,
          )
      ):
        raise FormalGeometryAuthorityError(
            "formal analytic supports are not provably distinct"
        )
      proof_margins.append(claimed_margin)
      replay_rows.append(
          {
              "case_id": case_id,
              "part": part,
              "fusion_face_index": identity[2],
              "proof_ordinal": ordinal,
              "best_occ_face_index": best_index,
              "candidate_occ_face_index": candidate_index,
              "surface_type": best_support[0],
              "support_relation": "provably_distinct",
              "sample_margin_lower_bound_mm": claimed_margin,
          }
      )
    if min(proof_margins) != float(raw_margin):
      raise FormalGeometryAuthorityError(
          "formal analytic proof ledger does not attain the mapping minimum margin"
      )
    replayed_mappings.add(identity)
  result = {
      "schema_version": "formal_mapper_analytic_margin_proof_replay.v1",
      "locked_minimum_sample_margin_mm": locked_margin,
      "analytic_support_distance_tolerance_mm": support_tolerance,
      "low_margin_mapping_count": len(replayed_mappings),
      "claimed_proof_count": len(replay_rows),
      "replayed_proof_count": len(replay_rows),
      "all_low_margin_proofs_replayed": True,
      "replay_payload_sha256": canonical_sha256(replay_rows),
  }
  return frozenset(replayed_mappings), result


def _case_row_by_id(rows: Any, case_id: str, *, label: str) -> Mapping[str, Any]:
  if not isinstance(rows, list):
    raise FormalGeometryAuthorityError(f"mapper family source {label} is malformed")
  matches = [row for row in rows if isinstance(row, Mapping) and row.get("case_id", row.get("id")) == case_id]
  if len(matches) != 1:
    raise FormalGeometryAuthorityError(
        f"mapper family source {label} does not uniquely contain the formal case"
    )
  return matches[0]


def _validate_mapper_family_source_case(
    family_source: Mapping[str, Any],
    *,
    case_id: str,
    context: _FormalInputContext,
) -> None:
  if family_source.get("schema_version") != 2:
    raise FormalGeometryAuthorityError("mapper family source schema differs")
  private = family_source.get("private_source_bindings")
  gold = family_source.get("evaluation_gold_contacts")
  if not isinstance(private, Mapping) or not isinstance(gold, Mapping):
    raise FormalGeometryAuthorityError("mapper family source private evidence is missing")
  private_case = _case_row_by_id(private.get("cases"), case_id, label="private source")
  gold_case = _case_row_by_id(gold.get("cases"), case_id, label="private gold")
  if dict(private_case) != dict(context.source_cases[case_id]):
    raise FormalGeometryAuthorityError(
        "mapper family source case differs from authenticated private source"
    )
  expected_gold = {"case_id": case_id, **dict(context.gold_cases[case_id])}
  normalized_gold = {"case_id": case_id}
  expected_gold_fields = set(expected_gold).difference({"case_id"})
  if set(gold_case).difference({"case_id", *expected_gold_fields}):
    raise FormalGeometryAuthorityError(
        "mapper family source endpoints contain unexpected fields"
    )
  for field in expected_gold_fields:
    if field in gold_case:
      normalized_gold[field] = gold_case[field]
    elif field in gold:
      normalized_gold[field] = gold[field]
    else:
      raise FormalGeometryAuthorityError(
          "mapper family source endpoints omit authenticated private gold metadata"
      )
  if normalized_gold != expected_gold:
    raise FormalGeometryAuthorityError(
        "mapper family source endpoints differ from authenticated private gold"
    )
  public_case = _case_row_by_id(family_source.get("cases"), case_id, label="public case")
  instances = context.case_material[case_id]["instances"]
  if (
      public_case.get("selected_part_names") != [row["part"] for row in instances]
      or public_case.get("selected_geometry_assets")
      != [row["geometry_asset"] for row in instances]
  ):
    raise FormalGeometryAuthorityError(
        "mapper family source public instance domain differs"
    )


def _authenticated_obj_extension_snapshot(
    authority: AuthenticatedContentReceiptObjExtension,
    *,
    context: _FormalInputContext,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
    CapturedFileArtifact,
]:
  if not isinstance(authority, AuthenticatedContentReceiptObjExtension):
    raise FormalGeometryAuthorityError(
        "formal face-map v4 requires authenticated OBJ extension authority"
    )
  if (
      authority.pilot_only is not context.pilot_only
      or authority.formal is not (not context.pilot_only)
      or authority.case_ids != context.case_ids
  ):
    raise FormalGeometryAuthorityError(
        "OBJ extension pilot/formal flags or exact case domain differ"
    )
  try:
    binding, members = authority.authenticated_snapshot()
  except (ContentReceiptObjExtensionError, OSError, ValueError) as error:
    raise FormalGeometryAuthorityError(
        "OBJ extension authority changed since authentication"
    ) from error
  required_binding = {
      "schema_version", "extension_schema_version", "formal", "pilot_only",
      "case_ids", "receipt", "receipt_payload_sha256", "input_bindings",
      "base_content_receipts", "archives", "member_set_sha256",
      "input_commitment_sha256", "unique_obj_member_count",
      "instance_reference_count",
  }
  input_bindings = binding.get("input_bindings")
  commitment_payload = {
      "case_ids": binding.get("case_ids"),
      "input_bindings": input_bindings,
      "base_content_receipts": binding.get("base_content_receipts"),
      "archives": binding.get("archives"),
  }
  if (
      set(binding) != required_binding
      or binding.get("schema_version")
      != "fusion_content_receipt_obj_extension_authority_binding.v1"
      or binding.get("extension_schema_version")
      != "fusion_content_receipt_obj_extension.v2"
      or binding.get("formal") is not (not context.pilot_only)
      or binding.get("pilot_only") is not context.pilot_only
      or binding.get("case_ids") != list(context.case_ids)
      or not isinstance(input_bindings, Mapping)
      or input_bindings.get("private_source")
      != dict(context.input_bindings["private_source"])
      or binding.get("input_commitment_sha256")
      != canonical_sha256(commitment_payload)
      or binding.get("unique_obj_member_count") != len(members)
  ):
    raise FormalGeometryAuthorityError(
        "OBJ extension authority binding/input commitment differs"
    )
  expected_members = {
      (
          str(context.source_cases[case_id]["source_receipt_binding"]["archive"]),
          PurePosixPath(str(step["path"])).with_suffix(".obj").as_posix(),
      )
      for case_id in context.case_ids
      for step in context.source_cases[case_id]["source_receipt_binding"][
          "body_steps"
      ]
  }
  member_rows = [members[key] for key in sorted(members)]
  if (
      set(members) != expected_members
      or binding.get("member_set_sha256") != canonical_sha256(member_rows)
  ):
    raise FormalGeometryAuthorityError(
        "OBJ extension archive/path member domain differs"
    )
  receipt_binding = binding.get("receipt")
  receipt_path = (
      receipt_binding.get("path")
      if isinstance(receipt_binding, Mapping)
      else None
  )
  try:
    receipt_capture = capture_file_artifact(
        receipt_path,
        label="OBJ extension authority receipt",
    )
  except (OSError, ValueError, TypeError) as error:
    raise FormalGeometryAuthorityError(
        "OBJ extension authority receipt is missing or unsafe"
    ) from error
  if (
      not isinstance(receipt_binding, Mapping)
      or receipt_binding.get("bytes") != receipt_capture.byte_count
      or receipt_binding.get("sha256") != receipt_capture.sha256
  ):
    raise FormalGeometryAuthorityError(
        "OBJ extension authority receipt binding differs"
    )
  return binding, members, receipt_capture


def _validate_mapper_case_inputs(
    case: Mapping[str, Any],
    *,
    case_id: str,
    context: _FormalInputContext,
    obj_extension_members: Mapping[
        tuple[str, str], Mapping[str, Any]
    ] | None = None,
) -> tuple[tuple[tuple[str, CapturedFileArtifact], ...], bool]:
  material = context.case_material[case_id]
  if case.get("case_id") != case_id or case.get("status") != "mapped_complete":
    raise FormalGeometryAuthorityError("mapper receipt case is not mapped_complete")
  if case.get("archive") != Path(str(material["archive"]["path"])).name:
    raise FormalGeometryAuthorityError("mapper receipt archive binding differs")
  inputs = case.get("inputs")
  assembly = inputs.get("assembly_json") if isinstance(inputs, Mapping) else None
  if not isinstance(assembly, Mapping) or any(
      assembly.get(key) != material["assembly_json"][key]
      for key in ("bytes", "sha256")
  ):
    raise FormalGeometryAuthorityError("mapper receipt assembly binding differs")

  source_binding = context.source_cases[case_id]["source_receipt_binding"]
  source_instances = {row["part"]: row for row in source_binding["instances"]}
  source_steps = {row["part"]: row for row in source_binding["body_steps"]}
  material_instances = {row["part"]: row for row in material["instances"]}
  bodies = case.get("bodies")
  if not isinstance(bodies, Mapping) or set(bodies) != set(material_instances):
    raise FormalGeometryAuthorityError("mapper receipt body instance domain differs")
  members = context.content_members[case_id]
  archive_sha = material["archive"]["sha256"]
  captured_objs: list[tuple[str, CapturedFileArtifact]] = []
  used_pilot_live_obj_revalidation = False
  for part, instance in material_instances.items():
    body = bodies.get(part)
    source_instance = source_instances[part]
    source_step = source_steps[part]
    if not isinstance(body, Mapping) or any(
        body.get(key) != value
        for key, value in {
            "geometry_asset": instance["geometry_asset"],
            "body_uuid": instance["body_uuid"],
            "source_instance_key": instance["source_instance_key"],
            "is_visible": source_instance["is_visible"],
            "is_grounded": source_instance["is_grounded"],
            "status": "ready_for_mapping",
        }.items()
    ):
      raise FormalGeometryAuthorityError("mapper receipt body/source identity differs")
    if body.get("step") != {
        key: source_step[key] for key in ("path", "bytes", "sha1", "sha256")
    }:
      raise FormalGeometryAuthorityError("mapper receipt STEP binding differs")
    obj = body.get("obj")
    obj_path = obj.get("path") if isinstance(obj, Mapping) else None
    member = members.get(obj_path) if isinstance(obj_path, str) else None
    extension_member = (
        obj_extension_members.get(
            (
                str(source_binding["archive"]),
                str(obj_path),
            )
        )
        if obj_extension_members is not None and isinstance(obj_path, str)
        else None
    )
    if (
        not isinstance(obj, Mapping)
        or not isinstance(obj_path, str)
        or PurePosixPath(obj_path).suffix.lower() != ".obj"
        or PurePosixPath(obj_path) != PurePosixPath(source_step["path"]).with_suffix(".obj")
        or PurePosixPath(obj_path).parent
        != PurePosixPath(str(context.source_cases[case_id]["assembly_dir"]))
        or obj.get("authenticated_archive_sha256") != archive_sha
        or obj.get("receipt_binding_mode")
        != "authenticated_archive_member_bytes"
    ):
      raise FormalGeometryAuthorityError("mapper receipt OBJ/archive binding differs")
    try:
      obj_capture = capture_file_artifact(
          _safe_member(
              context.restored_volume_roots[case_id], obj_path, suffix=".obj"
          ),
          label=f"case {case_id} restored OBJ {part}",
      )
    except (OSError, ValueError) as error:
      raise FormalGeometryAuthorityError(
          "mapper receipt restored OBJ is missing or unsafe"
      ) from error
    if (
        obj.get("bytes") != obj_capture.byte_count
        or obj.get("sha256") != obj_capture.sha256
        or (
            obj_extension_members is not None
            and (
                not isinstance(extension_member, Mapping)
                or extension_member.get("archive") != source_binding["archive"]
                or extension_member.get("path") != obj_path
                or extension_member.get("bytes") != obj_capture.byte_count
                or extension_member.get("sha256") != obj_capture.sha256
            )
        )
        or (
            isinstance(member, Mapping)
            and (
                member.get("bytes") != obj_capture.byte_count
                or member.get("sha256") != obj_capture.sha256
            )
        )
    ):
      raise FormalGeometryAuthorityError("mapper receipt OBJ/archive binding differs")
    if member is None and obj_extension_members is None:
      if context.pilot_only is not True:
        raise FormalGeometryAuthorityError(
            "formal mapper receipt OBJ is outside authenticated content member domain"
        )
      used_pilot_live_obj_revalidation = True
    captured_objs.append((f"case {case_id} restored OBJ {part}", obj_capture))
  return tuple(captured_objs), used_pilot_live_obj_revalidation


def _capture_and_validate_mapper_receipts(
    mapper_receipt_paths: Mapping[str, str | Path],
    *,
    context: _FormalInputContext,
) -> tuple[
    dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    dict[str, Any],
    tuple[tuple[str, CapturedFileArtifact], ...],
]:
  if not isinstance(mapper_receipt_paths, Mapping) or set(mapper_receipt_paths) != set(
      context.case_ids
  ):
    raise FormalGeometryAuthorityError(
        "formal face-map v3 requires one hash-bound mapper receipt per case"
    )
  mapper_module = __import__(
      "neurocad.fusion_face_mapper",
      fromlist=["build_mapped_endpoint_lookup", "MAPPER_VERSION"],
  )
  expected_source_hashes = _mapper_implementation_sha256s()
  locked_tolerances = asdict(mapper_module.MappingTolerances())
  lookups: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
  bindings: list[dict[str, Any]] = []
  captures: list[tuple[str, CapturedFileArtifact]] = []
  used_pilot_live_obj_revalidation = False
  family_cache: dict[Path, _StrictJsonEvidence] = {}
  for case_id in context.case_ids:
    receipt = _capture_json_strict(
        mapper_receipt_paths[case_id], label=f"case {case_id} mapper receipt"
    )
    payload = receipt.json.payload
    if not isinstance(payload, Mapping):
      raise FormalGeometryAuthorityError("mapper receipt is not an object")
    _validate_receipt_sha(payload)
    environment = payload.get("environment")
    implementation = (
        environment.get("implementation") if isinstance(environment, Mapping) else None
    )
    if not isinstance(implementation, Mapping) or dict(implementation) != {
        "module": "neurocad.fusion_face_mapper",
        "version": mapper_module.MAPPER_VERSION,
        "source_sha256s": expected_source_hashes,
    } or payload.get("mapper_version") != mapper_module.MAPPER_VERSION:
      raise FormalGeometryAuthorityError(
          "mapper receipt implementation binding differs"
      )
    source = payload.get("source")
    raw_family_path = source.get("family_source_path") if isinstance(source, Mapping) else None
    if not isinstance(raw_family_path, str):
      raise FormalGeometryAuthorityError("mapper receipt family source path is missing")
    try:
      family_path = Path(raw_family_path).resolve(strict=True)
    except OSError as error:
      raise FormalGeometryAuthorityError("mapper receipt family source is missing") from error
    family = family_cache.get(family_path)
    if family is None:
      family = _capture_json_strict(family_path, label="mapper family source")
      family_cache[family_path] = family
      captures.append(("mapper family source", family.file))
    if (
        source.get("family_source_bytes") != family.file.byte_count
        or source.get("family_source_sha256") != family.file.sha256
    ):
      raise FormalGeometryAuthorityError("mapper receipt family source hash differs")
    if not isinstance(family.json.payload, Mapping):
      raise FormalGeometryAuthorityError("mapper family source is not an object")
    _validate_mapper_family_source_case(
        family.json.payload, case_id=case_id, context=context
    )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 1 or not isinstance(
        raw_cases[0], Mapping
    ):
      raise FormalGeometryAuthorityError("mapper receipt must contain exactly one case")
    mapper_case = raw_cases[0]
    obj_captures, case_used_pilot_live_obj_revalidation = (
        _validate_mapper_case_inputs(
            mapper_case, case_id=case_id, context=context
        )
    )
    captures.extend(obj_captures)
    used_pilot_live_obj_revalidation = (
        used_pilot_live_obj_revalidation
        or case_used_pilot_live_obj_revalidation
    )
    try:
      lookup = mapper_module.build_mapped_endpoint_lookup(payload)
    except Exception as error:
      raise FormalGeometryAuthorityError(
          "mapper receipt endpoint lookup replay failed"
      ) from error
    expected_keys = {
        (case_id, part, fusion_index)
        for part, fusion_index in context.expected_endpoints[case_id]
    }
    if set(lookup) != expected_keys:
      raise FormalGeometryAuthorityError(
          "mapper receipt source-positive endpoint domain differs"
      )
    tolerances = payload.get("tolerances")
    if not isinstance(tolerances, Mapping):
      raise FormalGeometryAuthorityError("mapper receipt tolerances are missing")
    if canonical_sha256(dict(tolerances)) != canonical_sha256(locked_tolerances):
      raise FormalGeometryAuthorityError(
          "mapper receipt differs from the locked MappingTolerances configuration"
      )
    for row in lookup.values():
      _require_sound_mapper_mapping_evidence(row, tolerances=tolerances)
    case_summary = mapper_case.get("summary")
    if isinstance(case_summary, Mapping):
      _require_strict_int_fields(
          case_summary,
          ("endpoint_count", "mapped_endpoint_count"),
          label="mapper receipt case summary",
      )
      _require_strict_bool_fields(
          case_summary,
          (
              "all_endpoints_mapped",
              "all_bodies_ready",
              "case_complete",
          ),
          label="mapper receipt case summary",
      )
      _require_counter_mapping(
          case_summary.get("status_counts"),
          label="mapper receipt case summary status_counts",
      )
      _require_counter_mapping(
          case_summary.get("body_status_counts"),
          label="mapper receipt case summary body_status_counts",
      )
    if (
        not isinstance(case_summary, Mapping)
        or case_summary.get("all_endpoints_mapped") is not True
        or case_summary.get("case_complete") is not True
        or case_summary.get("mapped_endpoint_count")
        != case_summary.get("endpoint_count")
    ):
      raise FormalGeometryAuthorityError("mapper receipt case summary does not close")
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
      _require_strict_int_fields(
          summary,
          ("case_count", "endpoint_count", "mapped_endpoint_count"),
          label="mapper receipt run summary",
      )
      _require_strict_bool_fields(
          summary,
          ("all_cases_complete", "all_endpoints_mapped"),
          label="mapper receipt run summary",
      )
      _require_counter_mapping(
          summary.get("status_counts"),
          label="mapper receipt run summary status_counts",
      )
    if (
        not isinstance(summary, Mapping)
        or summary.get("all_cases_complete") is not True
        or summary.get("all_endpoints_mapped") is not True
    ):
      raise FormalGeometryAuthorityError("mapper receipt run summary does not close")
    lookups[case_id] = {key: copy.deepcopy(dict(value)) for key, value in lookup.items()}
    bindings.append(
        {
            "case_id": case_id,
            "receipt": _file_binding(receipt.file),
            "schema_version": str(payload.get("schema_version")),
            "receipt_payload_sha256": payload["receipt_payload_sha256"],
            "mapper_version": payload["mapper_version"],
            "family_source": _file_binding(family.file),
        }
    )
    captures.append((f"case {case_id} mapper receipt", receipt.file))
  return (
      lookups,
      {
          "schema_version": MAPPER_EVIDENCE_SCHEMA_VERSION,
          "exact_case_set": True,
          "provenance_mode": (
              "pilot_live_obj_revalidated"
              if used_pilot_live_obj_revalidation
              else "authenticated_content_member"
          ),
          "case_receipts": bindings,
      },
      tuple(captures),
  )


def _mapper_paths_from_formal_evidence(
    value: Any,
    *,
    case_ids: Sequence[str],
) -> dict[str, Path]:
  if not isinstance(value, Mapping) or set(value) != {
      "schema_version", "exact_case_set", "provenance_mode", "case_receipts"
  } or value.get("schema_version") != MAPPER_EVIDENCE_SCHEMA_VERSION or value.get(
      "exact_case_set"
  ) is not True or value.get("provenance_mode") not in {
      "authenticated_content_member", "pilot_live_obj_revalidated"
  }:
    raise FormalGeometryAuthorityError(
        "formal face-map v3 lacks hash-bound mapper evidence; rebuild the sidecar"
    )
  rows = value.get("case_receipts")
  if not isinstance(rows, list) or [
      row.get("case_id") if isinstance(row, Mapping) else None for row in rows
  ] != list(case_ids):
    raise FormalGeometryAuthorityError(
        "formal face-map mapper receipt case set/order differs"
    )
  result: dict[str, Path] = {}
  for row in rows:
    assert isinstance(row, Mapping)
    receipt = row.get("receipt")
    raw_path = receipt.get("path") if isinstance(receipt, Mapping) else None
    if not isinstance(raw_path, str) or not raw_path:
      raise FormalGeometryAuthorityError(
          "formal face-map mapper receipt path is missing"
      )
    result[str(row["case_id"])] = Path(raw_path)
  return result


def build_formal_face_map_audit(
    *,
    family_split_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    source_split: str,
    mapper_receipt_paths: Mapping[str, str | Path] | None = None,
    case_mappings: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    pilot_only: bool = False,
) -> dict[str, Any]:
  """Build v3 only from immutable, replayed development mapper receipts.

  ``case_mappings`` remains only to turn old callers into a clear fail-closed
  error.  Caller-provided OCC indices/signatures are never an authority input.
  """

  _require_face_map_promotion(pilot_only=pilot_only)

  context = _load_formal_inputs(
      family_split_manifest_path=family_split_manifest_path,
      public_cases_path=public_cases_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      archive_root=archive_root,
      restored_root=restored_root,
      source_split=source_split,
      pilot_only=pilot_only,
  )
  if mapper_receipt_paths is None:
    legacy = "; legacy case_mappings are not authoritative" if case_mappings is not None else ""
    raise FormalGeometryAuthorityError(
        "formal face-map v3 requires hash-bound mapper receipts" + legacy
    )
  if case_mappings is not None:
    raise FormalGeometryAuthorityError(
        "formal face-map v3 rejects caller-provided case_mappings"
    )
  mapper_lookups, mapper_evidence, mapper_captures = (
      _capture_and_validate_mapper_receipts(
          mapper_receipt_paths,
          context=context,
      )
  )
  cases: list[dict[str, Any]] = []
  endpoint_count = 0
  for case_id in context.case_ids:
    raw_rows = [
        mapper_lookups[case_id][key]
        for key in sorted(mapper_lookups[case_id])
    ]
    instances = {row["part"]: row for row in context.case_material[case_id]["instances"]}
    seen: set[tuple[str, int]] = set()
    mappings: list[dict[str, Any]] = []
    for raw in raw_rows:
      if not isinstance(raw, Mapping):
        raise FormalGeometryAuthorityError("face-map producer mapping is malformed")
      part = raw.get("part")
      index = raw.get("fusion_face_index")
      key = (str(part), index) if isinstance(index, int) and not isinstance(index, bool) else ("", -1)
      if key in seen or key not in context.expected_endpoints[case_id]:
        raise FormalGeometryAuthorityError("face-map producer endpoint identity differs")
      seen.add(key)
      occ_indices, signatures = _validate_occ_identity(
          raw.get("raw_occ_face_indices"), raw.get("source_face_signature_sha256s")
      )
      instance = instances[key[0]]
      mappings.append(
          {
              "part": key[0],
              "geometry_asset": instance["geometry_asset"],
              "body_uuid": instance["body_uuid"],
              "source_instance_key": copy.deepcopy(instance["source_instance_key"]),
              "fusion_face_index": key[1],
              "status": "mapped_unique",
              "mapping_mode": "authoritative_obj_face_group",
              "raw_occ_face_indices": occ_indices,
              "source_face_signature_sha256s": signatures,
              "mapper_mapping_payload_sha256": canonical_sha256(raw),
              "world_transform_mm": copy.deepcopy(instance["world_transform_mm"]),
              "world_transform_chain_sha256": instance["world_transform_chain_sha256"],
          }
      )
    if seen != set(context.expected_endpoints[case_id]):
      raise FormalGeometryAuthorityError("not all source-positive endpoints are mapped_unique")
    mappings.sort(key=lambda row: (row["part"], row["fusion_face_index"]))
    endpoint_count += len(mappings)
    cases.append(
        {
            "case_id": case_id,
            "inputs": copy.deepcopy(dict(context.case_material[case_id])),
            "face_mappings": mappings,
            "all_source_positive_endpoints_mapped_unique": True,
        }
    )
  receipt = {
      "schema_version": FACE_MAP_SCHEMA_VERSION,
      "formal": not context.pilot_only,
      "visibility": "private_evaluation_only",
      "source_split": context.source_split,
      "endpoint_lookup_contract": {
          "schema_version": ENDPOINT_LOOKUP_SCHEMA_VERSION,
          "key_fields": ["case_id", "part", "fusion_face_index"],
          "success_status": "mapped_unique",
          "part_semantics": "assembly_body_instance",
      },
      "input_bindings": copy.deepcopy(dict(context.input_bindings)),
      "mapper_evidence": mapper_evidence,
      "implementation": _implementation_binding(
          _FACE_SCHEMA_PATH,
          authority_kind="face_map",
      ),
      "cases": cases,
      "summary": {
          "case_count": len(cases),
          "endpoint_count": endpoint_count,
          "mapped_unique_count": endpoint_count,
          "exact_case_set": True,
          "all_source_positive_endpoints_mapped_unique": True,
      },
  }
  if context.pilot_only:
    receipt["pilot_only"] = True
    receipt["development"] = True
  receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
  _validate_face_payload(
      receipt,
      context=context,
      mapper_lookups=mapper_lookups,
      expected_mapper_evidence=mapper_evidence,
  )
  implementation_captures: list[tuple[str, CapturedFileArtifact]] = []
  _validate_implementation_binding(
      receipt["implementation"],
      schema_path=_FACE_SCHEMA_PATH,
      authority_kind="face_map",
      captures=implementation_captures,
  )
  _revalidate_captures(
      (*context.captures, *mapper_captures, *implementation_captures)
  )
  return receipt


def _validate_receipt_sha(payload: Mapping[str, Any]) -> None:
  expected = payload.get("receipt_payload_sha256")
  unsigned = dict(payload)
  unsigned.pop("receipt_payload_sha256", None)
  if (
      not isinstance(expected, str)
      or _SHA256_RE.fullmatch(expected) is None
      or canonical_sha256(unsigned) != expected
  ):
    raise FormalGeometryAuthorityError("formal authority receipt payload SHA256 is invalid")


def _validate_face_payload(
    payload: Any,
    *,
    context: _FormalInputContext,
    mapper_lookups: Mapping[
        str, Mapping[tuple[str, str, int], Mapping[str, Any]]
    ],
    expected_mapper_evidence: Mapping[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
  if not isinstance(payload, Mapping):
    raise FormalGeometryAuthorityError("formal face-map receipt is not an object")
  _validate_receipt_sha(payload)
  required = {
      "schema_version", "formal", "visibility", "source_split",
      "endpoint_lookup_contract", "input_bindings", "implementation",
      "mapper_evidence", "cases", "summary", "receipt_payload_sha256",
  }
  if context.pilot_only:
    required.update({"pilot_only", "development"})
  if set(payload) != required or (
      payload.get("schema_version") != FACE_MAP_SCHEMA_VERSION
      or payload.get("formal") is not (not context.pilot_only)
      or payload.get("pilot_only") is not (True if context.pilot_only else None)
      or payload.get("development") is not (True if context.pilot_only else None)
      or payload.get("visibility") != "private_evaluation_only"
      or payload.get("source_split") != context.source_split
  ):
    raise FormalGeometryAuthorityError("formal face-map receipt contract is invalid")
  expected_contract = {
      "schema_version": ENDPOINT_LOOKUP_SCHEMA_VERSION,
      "key_fields": ["case_id", "part", "fusion_face_index"],
      "success_status": "mapped_unique",
      "part_semantics": "assembly_body_instance",
  }
  if payload.get("endpoint_lookup_contract") != expected_contract:
    raise FormalGeometryAuthorityError("formal face-map endpoint lookup contract differs")
  if payload.get("input_bindings") != dict(context.input_bindings):
    raise FormalGeometryAuthorityError("formal face-map top-level input bindings differ")
  if payload.get("mapper_evidence") != dict(expected_mapper_evidence):
    raise FormalGeometryAuthorityError(
        "formal face-map mapper receipt binding differs"
    )
  raw_cases = payload.get("cases")
  if not isinstance(raw_cases, list):
    raise FormalGeometryAuthorityError("formal face-map cases are malformed")
  if [row.get("case_id") if isinstance(row, Mapping) else None for row in raw_cases] != list(context.case_ids):
    raise FormalGeometryAuthorityError("formal face-map case set/order differs")
  lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
  mapping_keys = {
      "part", "geometry_asset", "body_uuid", "source_instance_key",
      "fusion_face_index", "status", "mapping_mode",
      "raw_occ_face_indices", "source_face_signature_sha256s",
      "mapper_mapping_payload_sha256", "world_transform_mm",
      "world_transform_chain_sha256",
  }
  for raw_case in raw_cases:
    if not isinstance(raw_case, Mapping) or set(raw_case) != {
        "case_id", "inputs", "face_mappings", "all_source_positive_endpoints_mapped_unique"
    }:
      raise FormalGeometryAuthorityError("formal face-map case contract is malformed")
    case_id = str(raw_case["case_id"])
    if context.pilot_only:
      print(
          json.dumps(
              {"event": "pilot_face_signature_replay", "case_id": case_id},
              sort_keys=True,
          ),
          file=sys.stderr,
          flush=True,
      )
    if raw_case.get("inputs") != context.case_material[case_id]:
      raise FormalGeometryAuthorityError("formal face-map case input replay differs")
    if raw_case.get("all_source_positive_endpoints_mapped_unique") is not True:
      raise FormalGeometryAuthorityError("formal face-map case is not endpoint-complete")
    raw_mappings = raw_case.get("face_mappings")
    if not isinstance(raw_mappings, list):
      raise FormalGeometryAuthorityError("formal face mappings are malformed")
    instance_by_part = {
        row["part"]: row for row in context.case_material[case_id]["instances"]
    }
    step_by_part = {
        row["part"]: row for row in context.case_material[case_id]["steps"]
    }
    face_signature_cache: dict[str, list[str]] = {}
    backend = __import__(
        "neurocad.cadquery_backend",
        fromlist=["load_step_shape", "source_face_signature_sha256"],
    )
    actual: set[tuple[str, int]] = set()
    for raw in raw_mappings:
      raw = _require_exact_keys(
          raw,
          mapping_keys,
          label="formal face mapping row",
      )
      source_instance_key = raw.get("source_instance_key")
      source_kind = (
          source_instance_key.get("kind")
          if isinstance(source_instance_key, Mapping)
          else None
      )
      _require_exact_keys(
          source_instance_key,
          (
              {"kind", "body_uuid", "occurrence_uuid"}
              if source_kind == "occurrence"
              else {"kind", "body_uuid", "root_component_uuid"}
          ),
          label="formal face mapping source_instance_key",
      )
      _require_exact_keys(
          raw.get("world_transform_mm"),
          {"rotation_row_major", "translation_mm"},
          label="formal face mapping world_transform_mm",
      )
      part = raw.get("part")
      index = raw.get("fusion_face_index")
      if not isinstance(part, str) or isinstance(index, bool) or not isinstance(index, int):
        raise FormalGeometryAuthorityError("formal face mapping identity is malformed")
      identity = (part, index)
      if identity in actual:
        raise FormalGeometryAuthorityError("formal face mapping identity is duplicated")
      actual.add(identity)
      instance = instance_by_part.get(part)
      if instance is None or any(
          raw.get(field) != expected
          for field, expected in {
              "geometry_asset": instance["geometry_asset"],
              "body_uuid": instance["body_uuid"],
              "source_instance_key": instance["source_instance_key"],
              "status": "mapped_unique",
              "mapping_mode": "authoritative_obj_face_group",
              "world_transform_mm": instance["world_transform_mm"],
              "world_transform_chain_sha256": instance["world_transform_chain_sha256"],
          }.items()
      ):
        raise FormalGeometryAuthorityError("formal face mapping source/transform binding differs")
      _validate_occ_identity(raw.get("raw_occ_face_indices"), raw.get("source_face_signature_sha256s"))
      step = step_by_part[part]
      signatures = face_signature_cache.get(part)
      if signatures is None:
        try:
          shape = backend.load_step_shape(Path(step["file"]["path"]))
          if (
              str(shape.ShapeType()) != "Solid"
              or not bool(shape.isValid())
              or len(list(shape.Solids())) != 1
          ):
            raise FormalGeometryAuthorityError(
                "re-imported STEP is not one valid solid"
            )
          signatures = [
              backend.source_face_signature_sha256(face)
              for face in shape.Faces()
          ]
        except FormalGeometryAuthorityError:
          raise
        except Exception as error:
          raise FormalGeometryAuthorityError(
              "re-imported STEP face signatures cannot be replayed"
          ) from error
        face_signature_cache[part] = signatures
      indices = raw["raw_occ_face_indices"]
      expected_signatures = raw["source_face_signature_sha256s"]
      if any(index >= len(signatures) for index in indices) or [
          signatures[index] for index in indices
      ] != expected_signatures:
        raise FormalGeometryAuthorityError(
            "re-imported STEP face identity/signature differs"
        )
      mapper_row = mapper_lookups.get(case_id, {}).get((case_id, part, index))
      if not isinstance(mapper_row, Mapping) or any(
          raw.get(field) != mapper_row.get(field)
          for field in (
              "part",
              "geometry_asset",
              "body_uuid",
              "source_instance_key",
              "fusion_face_index",
              "status",
              "mapping_mode",
              "raw_occ_face_indices",
              "source_face_signature_sha256s",
          )
      ) or raw.get("mapper_mapping_payload_sha256") != canonical_sha256(
          mapper_row
      ):
        raise FormalGeometryAuthorityError(
            "formal face mapping differs from bound mapper receipt"
        )
      lookup[(case_id, part, index)] = copy.deepcopy(dict(raw))
    if actual != set(context.expected_endpoints[case_id]):
      raise FormalGeometryAuthorityError("formal face-map source-positive endpoint set differs")
  summary = payload.get("summary")
  summary_keys = {
      "case_count", "endpoint_count", "mapped_unique_count",
      "exact_case_set", "all_source_positive_endpoints_mapped_unique",
  }
  summary = _require_exact_keys(
      summary,
      summary_keys,
      label="formal face-map summary",
  )
  _require_strict_int_fields(
      summary,
      ("case_count", "endpoint_count", "mapped_unique_count"),
      label="formal face-map summary",
  )
  _require_strict_bool_fields(
      summary,
      ("exact_case_set", "all_source_positive_endpoints_mapped_unique"),
      label="formal face-map summary",
  )
  if summary != {
      "case_count": len(context.case_ids),
      "endpoint_count": len(lookup),
      "mapped_unique_count": len(lookup),
      "exact_case_set": True,
      "all_source_positive_endpoints_mapped_unique": True,
  }:
    raise FormalGeometryAuthorityError("formal face-map summary does not close")
  return lookup


def _revalidate_captures(
    captures: Sequence[
        tuple[str, CapturedFileArtifact | _StreamCapturedFile]
    ],
) -> None:
  for label, captured in captures:
    try:
      if isinstance(captured, _StreamCapturedFile):
        current = _stream_capture(captured.path, label=label)
        if current != captured:
          raise ValueError(f"{label} changed since streaming capture")
      else:
        reverify_captured_file_artifact(captured, label=label)
    except (TypeError, ValueError, FormalGeometryAuthorityError) as error:
      raise FormalGeometryAuthorityError(f"{label} changed since authentication") from error


def _capture_memory_binding(
    label: str,
    captured: CapturedFileArtifact | _StreamCapturedFile,
) -> dict[str, Any]:
  return {
      "label": label,
      "path": str(captured.resolved_path),
      "sha256": captured.sha256,
      "byte_count": captured.byte_count,
      "device": captured.device,
      "inode": captured.inode,
      "link_count": captured.link_count,
  }


def _face_map_memory_state_payload(
    *,
    schema_version: str,
    case_ids: Sequence[str],
    source_split: str,
    pilot_only: bool,
    receipt: CapturedFileArtifact,
    captures: Sequence[tuple[str, CapturedFileArtifact]],
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    case_parts: Mapping[str, Sequence[str]],
    case_material: Mapping[str, Mapping[str, Any]],
    source_pair_counts: Mapping[str, Mapping[tuple[str, str], int]],
) -> dict[str, Any]:
  return {
      "kind": "AuthenticatedFormalFaceMap",
      "schema_version": str(schema_version),
      "case_ids": list(case_ids),
      "source_split": str(source_split),
      "pilot_only": bool(pilot_only),
      "receipt": _capture_memory_binding("receipt", receipt),
      "captures": [
          _capture_memory_binding(label, captured)
          for label, captured in captures
      ],
      "lookup": [
          {"key": list(key), "value": _recursive_mutable_copy(value)}
          for key, value in sorted(lookup.items())
      ],
      "case_parts": {
          key: list(value) for key, value in sorted(case_parts.items())
      },
      "case_material": _recursive_mutable_copy(case_material),
      "source_pair_counts": [
          {
              "case_id": case_id,
              "pairs": [
                  [left, right, count]
                  for (left, right), count in sorted(pair_counts.items())
              ],
          }
          for case_id, pair_counts in sorted(source_pair_counts.items())
      ],
  }


class AuthenticatedFormalFaceMap(_ImmutableAuthenticatedCapability):
  """Ephemeral path-authenticated capability for formal face identities."""

  __slots__ = (
      "schema_version", "case_ids", "source_split", "pilot_only", "_receipt", "_captures", "_lookup",
      "_case_parts", "_case_material", "_source_pair_counts",
  )

  def __init__(
      self,
      *,
      schema_version: str,
      case_ids: Sequence[str],
      source_split: str,
      pilot_only: bool = False,
      receipt: CapturedFileArtifact,
      captures: Sequence[tuple[str, CapturedFileArtifact]],
      lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
      case_parts: Mapping[str, Sequence[str]],
      case_material: Mapping[str, Mapping[str, Any]],
      source_pair_counts: Mapping[str, Mapping[tuple[str, str], int]],
      _issuance_context: object,
  ) -> None:
    try:
      _require_verified_issuance_context(
          _issuance_context,
          _FACE_CAPABILITY_KIND,
      )
    except TypeError as error:
      raise TypeError(
          "AuthenticatedFormalFaceMap must be created by its verifier with a "
          "one-shot verified issuance context"
      ) from error
    self.schema_version = str(schema_version)
    self.case_ids = tuple(case_ids)
    self.source_split = str(source_split)
    self.pilot_only = bool(pilot_only)
    self._receipt = receipt
    self._captures = tuple(captures)
    self._lookup = _recursive_freeze(copy.deepcopy(dict(lookup)))
    self._case_parts = _recursive_freeze(
        {key: tuple(value) for key, value in case_parts.items()}
    )
    self._case_material = _recursive_freeze(copy.deepcopy(dict(case_material)))
    self._source_pair_counts = _recursive_freeze(
        {key: dict(value) for key, value in source_pair_counts.items()}
    )
    try:
      self._register_memory_state(
          self._memory_state_payload(),
          issuance_context=_issuance_context,
          capability_kind=_FACE_CAPABILITY_KIND,
      )
    except (FormalGeometryAuthorityError, TypeError) as error:
      raise TypeError(
          "AuthenticatedFormalFaceMap must be created by its verifier with a "
          "one-shot verified issuance context"
      ) from error
    self._seal()

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated formal face-map authority is not serializable")

  def _memory_state_payload(self) -> dict[str, Any]:
    return _face_map_memory_state_payload(
        schema_version=self.schema_version,
        case_ids=self.case_ids,
        source_split=self.source_split,
        pilot_only=self.pilot_only,
        receipt=self._receipt,
        captures=self._captures,
        lookup=self._lookup,
        case_parts=self._case_parts,
        case_material=self._case_material,
        source_pair_counts=self._source_pair_counts,
    )

  def revalidate(self) -> None:
    self._require_memory_state(self._memory_state_payload())
    _revalidate_captures((("formal face-map receipt", self._receipt), *self._captures))

  def require_mapped_endpoint(self, case_id: str, part: str, fusion_face_index: int) -> dict[str, Any]:
    self.revalidate()
    try:
      return _recursive_mutable_copy(
          self._lookup[(case_id, part, fusion_face_index)]
      )
    except KeyError as error:
      raise FormalGeometryAuthorityError("endpoint is not formally mapped_unique") from error

  def endpoint_lookup(self) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Return a defensive copy for deterministic producer-semantic replay."""

    self.revalidate()
    return {
        key: _recursive_mutable_copy(value) for key, value in self._lookup.items()
    }


def verify_formal_face_map_authority(
    receipt_path: str | Path,
    *,
    family_split_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    source_split: str,
    pilot_only: bool = False,
) -> _VerifiedHandleConstruction:
  """Authenticate a v3 face map by replaying every bound source input."""

  _require_face_map_promotion(pilot_only=pilot_only)

  context = _load_formal_inputs(
      family_split_manifest_path=family_split_manifest_path,
      public_cases_path=public_cases_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      archive_root=archive_root,
      restored_root=restored_root,
      source_split=source_split,
      pilot_only=pilot_only,
  )
  receipt = _capture_json_strict(receipt_path, label="formal face-map receipt")
  captures = list(context.captures)
  _validate_implementation_binding(
      receipt.json.payload.get("implementation") if isinstance(receipt.json.payload, Mapping) else None,
      schema_path=_FACE_SCHEMA_PATH,
      authority_kind="face_map",
      captures=captures,
  )
  raw_payload = receipt.json.payload
  raw_mapper_evidence = (
      raw_payload.get("mapper_evidence") if isinstance(raw_payload, Mapping) else None
  )
  mapper_paths = _mapper_paths_from_formal_evidence(
      raw_mapper_evidence,
      case_ids=context.case_ids,
  )
  mapper_lookups, mapper_evidence, mapper_captures = (
      _capture_and_validate_mapper_receipts(mapper_paths, context=context)
  )
  captures.extend(mapper_captures)
  lookup = _validate_face_payload(
      raw_payload,
      context=context,
      mapper_lookups=mapper_lookups,
      expected_mapper_evidence=mapper_evidence,
  )
  all_captures = (("formal face-map receipt", receipt.file), *captures)
  _revalidate_captures(all_captures)
  case_parts = {
      case_id: tuple(
          row["part"]
          for row in context.case_material[case_id]["instances"]
          if row["visible"] is True
      )
      for case_id in context.case_ids
  }
  constructor_kwargs = {
      "schema_version": FACE_MAP_SCHEMA_VERSION,
      "case_ids": context.case_ids,
      "source_split": context.source_split,
      "pilot_only": context.pilot_only,
      "receipt": receipt.file,
      "captures": tuple(captures),
      "lookup": lookup,
      "case_parts": case_parts,
      "case_material": context.case_material,
      "source_pair_counts": context.source_pair_counts,
  }
  return _VerifiedHandleConstruction(
      kwargs=MappingProxyType(constructor_kwargs),
      memory_state_commitment=canonical_sha256(
          _face_map_memory_state_payload(**constructor_kwargs)
      ),
  )


def _validate_live_isolated_mapper_replay(
    replay: Any,
    *,
    context: _FormalInputContext,
    obj_extension_members: Mapping[
        tuple[str, str], Mapping[str, Any]
    ],
) -> tuple[
    dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    dict[str, Any],
    tuple[tuple[str, CapturedFileArtifact], ...],
]:
  """Bind an ephemeral live replay to the authenticated formal input graph."""

  from .isolated_face_mapper_replay import IsolatedFaceMapperReplay

  if not isinstance(replay, IsolatedFaceMapperReplay):
    raise FormalGeometryAuthorityError("isolated mapper replay result type differs")
  manifest = replay.manifest
  family_binding = manifest.get("inputs", {}).get("family_source")
  family_path = family_binding.get("path") if isinstance(family_binding, Mapping) else None
  if not isinstance(family_path, str):
    raise FormalGeometryAuthorityError("isolated mapper family source is missing")
  family = _capture_json_strict(family_path, label="isolated mapper family source")
  if not isinstance(family.json.payload, Mapping):
    raise FormalGeometryAuthorityError("isolated mapper family source is not an object")
  for case_id in context.case_ids:
    _validate_mapper_family_source_case(
        family.json.payload,
        case_id=case_id,
        context=context,
    )
  payload = replay.receipt_payload()
  raw_cases = payload.get("cases") if isinstance(payload, Mapping) else None
  if not isinstance(raw_cases, list) or {
      row.get("case_id") if isinstance(row, Mapping) else None for row in raw_cases
  } != set(context.case_ids):
    raise FormalGeometryAuthorityError("isolated mapper live case set differs")
  case_by_id = {
      str(row["case_id"]): row for row in raw_cases if isinstance(row, Mapping)
  }
  captures: list[tuple[str, CapturedFileArtifact]] = [
      ("isolated mapper family source", family.file)
  ]
  for case_id in context.case_ids:
    obj_captures, used_live = _validate_mapper_case_inputs(
        case_by_id[case_id],
        case_id=case_id,
        context=context,
        obj_extension_members=obj_extension_members,
    )
    if used_live:
      raise FormalGeometryAuthorityError(
          "face-map v4 forbids pilot_live_obj_revalidated provenance"
      )
    captures.extend(obj_captures)
  full_lookup = replay.endpoint_lookup()
  expected_keys = {
      (case_id, part, fusion_index)
      for case_id in context.case_ids
      for part, fusion_index in context.expected_endpoints[case_id]
  }
  if set(full_lookup) != expected_keys:
    raise FormalGeometryAuthorityError(
        "isolated mapper source-positive endpoint domain differs"
    )
  tolerances = payload.get("tolerances")
  if not isinstance(tolerances, Mapping) or canonical_sha256(dict(tolerances)) != canonical_sha256(
      asdict(
          __import__(
              "neurocad.fusion_face_mapper",
              fromlist=["MappingTolerances"],
          ).MappingTolerances()
      )
  ):
    raise FormalGeometryAuthorityError(
        "isolated mapper differs from locked MappingTolerances"
    )
  replayed_low_margin, analytic_replay = (
      _formal_mapper_analytic_margin_proof_replay(
          full_lookup,
          context=context,
          tolerances=tolerances,
      )
  )
  for identity, mapping in full_lookup.items():
    _require_sound_mapper_mapping_evidence(
        mapping,
        tolerances=tolerances,
        analytic_margin_proof_replayed=identity in replayed_low_margin,
    )
  lookups = {
      case_id: {
          key: copy.deepcopy(dict(value))
          for key, value in full_lookup.items()
          if key[0] == case_id
      }
      for case_id in context.case_ids
  }
  evidence = {
      "schema_version": "fusion_formal_isolated_mapper_evidence.v1",
      "exact_case_set": True,
      "provenance_mode": "authenticated_obj_extension_member",
      "manifest": copy.deepcopy(dict(manifest)),
      "analytic_margin_proof_replay": analytic_replay,
  }
  return lookups, evidence, tuple(captures)


def _face_v4_shadow(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
  shadow = copy.deepcopy(dict(payload))
  shadow["schema_version"] = FACE_MAP_SCHEMA_VERSION
  shadow["mapper_evidence"] = shadow.pop("isolated_mapper_replay")
  shadow.pop("obj_extension_authority", None)
  shadow.pop("receipt_payload_sha256", None)
  shadow["receipt_payload_sha256"] = canonical_sha256(shadow)
  return shadow


def _validate_face_v4_payload(
    payload: Any,
    *,
    context: _FormalInputContext,
    mapper_lookups: Mapping[
        str, Mapping[tuple[str, str, int], Mapping[str, Any]]
    ],
    expected_replay_evidence: Mapping[str, Any],
    expected_obj_extension_binding: Mapping[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
  if not isinstance(payload, Mapping):
    raise FormalGeometryAuthorityError("isolated replay face-map receipt is not an object")
  _validate_receipt_sha(payload)
  required = {
      "schema_version", "formal", "visibility", "source_split",
      "endpoint_lookup_contract", "input_bindings", "implementation",
      "isolated_mapper_replay", "obj_extension_authority", "cases", "summary",
      "receipt_payload_sha256",
  }
  if context.pilot_only:
    required.update({"pilot_only", "development"})
  if set(payload) != required or (
      payload.get("schema_version") != FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION
      or payload.get("formal") is not (not context.pilot_only)
      or payload.get("pilot_only") is not (True if context.pilot_only else None)
      or payload.get("development") is not (True if context.pilot_only else None)
      or payload.get("isolated_mapper_replay") != dict(expected_replay_evidence)
      or payload.get("obj_extension_authority")
      != dict(expected_obj_extension_binding)
      or expected_replay_evidence.get("provenance_mode")
      != "authenticated_obj_extension_member"
  ):
    raise FormalGeometryAuthorityError(
        "isolated replay face-map receipt contract differs"
    )
  return _validate_face_payload(
      _face_v4_shadow(payload),
      context=context,
      mapper_lookups=mapper_lookups,
      expected_mapper_evidence=expected_replay_evidence,
  )


def _build_face_cases_from_live_lookup(
    *,
    context: _FormalInputContext,
    mapper_lookups: Mapping[
        str, Mapping[tuple[str, str, int], Mapping[str, Any]]
    ],
) -> tuple[list[dict[str, Any]], int]:
  cases: list[dict[str, Any]] = []
  endpoint_count = 0
  for case_id in context.case_ids:
    raw_rows = [mapper_lookups[case_id][key] for key in sorted(mapper_lookups[case_id])]
    instances = {
        row["part"]: row for row in context.case_material[case_id]["instances"]
    }
    mappings: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_rows:
      part = raw.get("part")
      index = raw.get("fusion_face_index")
      key = (
          (str(part), index)
          if type(index) is int
          else ("", -1)
      )
      if key in seen or key not in context.expected_endpoints[case_id]:
        raise FormalGeometryAuthorityError(
            "isolated replay producer endpoint identity differs"
        )
      seen.add(key)
      occ_indices, signatures = _validate_occ_identity(
          raw.get("raw_occ_face_indices"),
          raw.get("source_face_signature_sha256s"),
      )
      instance = instances[key[0]]
      mappings.append(
          {
              "part": key[0],
              "geometry_asset": instance["geometry_asset"],
              "body_uuid": instance["body_uuid"],
              "source_instance_key": copy.deepcopy(instance["source_instance_key"]),
              "fusion_face_index": key[1],
              "status": "mapped_unique",
              "mapping_mode": "authoritative_obj_face_group",
              "raw_occ_face_indices": occ_indices,
              "source_face_signature_sha256s": signatures,
              "mapper_mapping_payload_sha256": canonical_sha256(raw),
              "world_transform_mm": copy.deepcopy(instance["world_transform_mm"]),
              "world_transform_chain_sha256": instance[
                  "world_transform_chain_sha256"
              ],
          }
      )
    if seen != set(context.expected_endpoints[case_id]):
      raise FormalGeometryAuthorityError(
          "isolated replay does not map every source-positive endpoint"
      )
    mappings.sort(key=lambda row: (row["part"], row["fusion_face_index"]))
    endpoint_count += len(mappings)
    cases.append(
        {
            "case_id": case_id,
            "inputs": copy.deepcopy(dict(context.case_material[case_id])),
            "face_mappings": mappings,
            "all_source_positive_endpoints_mapped_unique": True,
        }
    )
  return cases, endpoint_count


def build_formal_face_map_audit_v4(
    *,
    family_split_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    source_split: str,
    mapper_family_source_path: str | Path,
    obj_extension_authority: AuthenticatedContentReceiptObjExtension,
    pilot_only: bool = False,
) -> dict[str, Any]:
  """Build v4 solely from a fresh isolated mapper lookup."""

  _require_face_map_promotion(
      pilot_only=pilot_only,
      schema_version=FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION,
  )
  context = _load_formal_inputs(
      family_split_manifest_path=family_split_manifest_path,
      public_cases_path=public_cases_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      archive_root=archive_root,
      restored_root=restored_root,
      source_split=source_split,
      pilot_only=pilot_only,
  )
  (
      obj_extension_binding,
      obj_extension_members,
      obj_extension_receipt_capture,
  ) = _authenticated_obj_extension_snapshot(
      obj_extension_authority,
      context=context,
  )
  from .isolated_face_mapper_replay import run_isolated_face_mapper_replay

  live = run_isolated_face_mapper_replay(
      family_source_path=mapper_family_source_path,
      dataset_root=restored_root,
      case_ids=context.case_ids,
      phase="builder",
  )
  mapper_lookups, replay_evidence, replay_captures = (
      _validate_live_isolated_mapper_replay(
          live,
          context=context,
          obj_extension_members=obj_extension_members,
      )
  )
  cases, endpoint_count = _build_face_cases_from_live_lookup(
      context=context,
      mapper_lookups=mapper_lookups,
  )
  receipt = {
      "schema_version": FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION,
      "formal": not context.pilot_only,
      "visibility": "private_evaluation_only",
      "source_split": context.source_split,
      "endpoint_lookup_contract": {
          "schema_version": ENDPOINT_LOOKUP_SCHEMA_VERSION,
          "key_fields": ["case_id", "part", "fusion_face_index"],
          "success_status": "mapped_unique",
          "part_semantics": "assembly_body_instance",
      },
      "input_bindings": copy.deepcopy(dict(context.input_bindings)),
      "isolated_mapper_replay": replay_evidence,
      "obj_extension_authority": copy.deepcopy(obj_extension_binding),
      "implementation": _implementation_binding(
          _FACE_ISOLATED_REPLAY_SCHEMA_PATH,
          authority_kind="face_map_isolated_replay",
      ),
      "cases": cases,
      "summary": {
          "case_count": len(cases),
          "endpoint_count": endpoint_count,
          "mapped_unique_count": endpoint_count,
          "exact_case_set": True,
          "all_source_positive_endpoints_mapped_unique": True,
      },
  }
  if context.pilot_only:
    receipt["pilot_only"] = True
    receipt["development"] = True
  receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
  _validate_face_v4_payload(
      receipt,
      context=context,
      mapper_lookups=mapper_lookups,
      expected_replay_evidence=replay_evidence,
      expected_obj_extension_binding=obj_extension_binding,
  )
  implementation_captures: list[tuple[str, CapturedFileArtifact]] = []
  _validate_implementation_binding(
      receipt["implementation"],
      schema_path=_FACE_ISOLATED_REPLAY_SCHEMA_PATH,
      authority_kind="face_map_isolated_replay",
      captures=implementation_captures,
  )
  _revalidate_captures(
      (
          *context.captures,
          ("OBJ extension authority receipt", obj_extension_receipt_capture),
          *replay_captures,
          *implementation_captures,
      )
  )
  return receipt


def verify_formal_face_map_authority_v4(
    receipt_path: str | Path,
    *,
    family_split_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    source_split: str,
    obj_extension_authority: AuthenticatedContentReceiptObjExtension,
    pilot_only: bool = False,
) -> _VerifiedHandleConstruction:
  """Verify v4 with a second fresh mapper process; stored faces are ignored."""

  _require_face_map_promotion(
      pilot_only=pilot_only,
      schema_version=FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION,
  )
  context = _load_formal_inputs(
      family_split_manifest_path=family_split_manifest_path,
      public_cases_path=public_cases_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      archive_root=archive_root,
      restored_root=restored_root,
      source_split=source_split,
      pilot_only=pilot_only,
  )
  receipt = _capture_json_strict(receipt_path, label="isolated replay face-map receipt")
  captures = list(context.captures)
  _validate_implementation_binding(
      receipt.json.payload.get("implementation")
      if isinstance(receipt.json.payload, Mapping)
      else None,
      schema_path=_FACE_ISOLATED_REPLAY_SCHEMA_PATH,
      authority_kind="face_map_isolated_replay",
      captures=captures,
  )
  payload = receipt.json.payload
  (
      obj_extension_binding,
      obj_extension_members,
      obj_extension_receipt_capture,
  ) = _authenticated_obj_extension_snapshot(
      obj_extension_authority,
      context=context,
  )
  if (
      not isinstance(payload, Mapping)
      or payload.get("obj_extension_authority")
      != obj_extension_binding
  ):
    raise FormalGeometryAuthorityError(
        "stored OBJ extension authority binding differs"
    )
  captures.append(
      ("OBJ extension authority receipt", obj_extension_receipt_capture)
  )
  stored_evidence = (
      payload.get("isolated_mapper_replay")
      if isinstance(payload, Mapping)
      else None
  )
  stored_manifest = (
      stored_evidence.get("manifest")
      if isinstance(stored_evidence, Mapping)
      else None
  )
  family_binding = (
      stored_manifest.get("inputs", {}).get("family_source")
      if isinstance(stored_manifest, Mapping)
      else None
  )
  family_path = (
      family_binding.get("path") if isinstance(family_binding, Mapping) else None
  )
  if not isinstance(family_path, str):
    raise FormalGeometryAuthorityError(
        "isolated replay receipt lacks mapper family provenance"
    )
  from .isolated_face_mapper_replay import (
      IsolatedFaceMapperReplayError,
      require_equivalent_isolated_replay,
      run_isolated_face_mapper_replay,
  )

  live = run_isolated_face_mapper_replay(
      family_source_path=family_path,
      dataset_root=restored_root,
      case_ids=context.case_ids,
      phase="verifier",
  )
  try:
    require_equivalent_isolated_replay(stored_manifest, live)
  except IsolatedFaceMapperReplayError as error:
    raise FormalGeometryAuthorityError(
        "stored mapper provenance differs from fresh verifier replay"
    ) from error
  mapper_lookups, live_evidence, replay_captures = (
      _validate_live_isolated_mapper_replay(
          live,
          context=context,
          obj_extension_members=obj_extension_members,
      )
  )
  expected_evidence = {
      **live_evidence,
      "manifest": copy.deepcopy(dict(stored_manifest)),
  }
  if not isinstance(stored_evidence, Mapping) or dict(stored_evidence) != expected_evidence:
    raise FormalGeometryAuthorityError(
        "stored analytic/live replay provenance differs"
    )
  captures.extend(replay_captures)
  lookup = _validate_face_v4_payload(
      payload,
      context=context,
      mapper_lookups=mapper_lookups,
      expected_replay_evidence=expected_evidence,
      expected_obj_extension_binding=obj_extension_binding,
  )
  _revalidate_captures((("isolated replay face-map receipt", receipt.file), *captures))
  case_parts = {
      case_id: tuple(
          row["part"]
          for row in context.case_material[case_id]["instances"]
          if row["visible"] is True
      )
      for case_id in context.case_ids
  }
  constructor_kwargs = {
      "schema_version": FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION,
      "case_ids": context.case_ids,
      "source_split": context.source_split,
      "pilot_only": context.pilot_only,
      "receipt": receipt.file,
      "captures": tuple(captures),
      "lookup": lookup,
      "case_parts": case_parts,
      "case_material": context.case_material,
      "source_pair_counts": context.source_pair_counts,
  }
  return _VerifiedHandleConstruction(
      kwargs=MappingProxyType(constructor_kwargs),
      memory_state_commitment=canonical_sha256(
          _face_map_memory_state_payload(**constructor_kwargs)
      ),
  )


def _finite_nonnegative(value: Any, *, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise FormalGeometryAuthorityError(f"{label} must be numeric")
  result = float(value)
  if not math.isfinite(result) or result < 0.0:
    raise FormalGeometryAuthorityError(f"{label} must be finite and nonnegative")
  return result


def _normalize_pair_evidence(value: Any, *, source_positive: bool) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise FormalGeometryAuthorityError("formal census pair evidence is malformed")
  route = value.get("route")
  if route == "exact_kernel":
    _require_exact_keys(
        value,
        {
            "route", "minimum_solid_distance_mm",
            "solid_common_volume_mm3", "kernel",
        },
        label="formal census exact-kernel evidence",
    )
    distance = _finite_nonnegative(value.get("minimum_solid_distance_mm"), label="minimum solid distance")
    volume = _finite_nonnegative(value.get("solid_common_volume_mm3"), label="solid common volume")
    kernel = value.get("kernel")
    if not isinstance(kernel, str) or not kernel:
      raise FormalGeometryAuthorityError("exact census kernel identity is missing")
    if source_positive:
      if distance > DEFAULT_CONTACT_THRESHOLD_MM or volume > DEFAULT_INTERFERENCE_THRESHOLD_MM3:
        raise FormalGeometryAuthorityError("source-positive pair fails formal contact/interference thresholds")
    elif (
        distance <= DEFAULT_CONTACT_THRESHOLD_MM
        or volume > DEFAULT_INTERFERENCE_THRESHOLD_MM3
    ):
      raise FormalGeometryAuthorityError(
          "exact negative fails contact-distance or interference threshold"
      )
    return {
        "route": route,
        "minimum_solid_distance_mm": distance,
        "solid_common_volume_mm3": volume,
        "kernel": kernel,
    }
  if route == "conservative_aabb_lower_bound" and not source_positive:
    _require_exact_keys(
        value,
        {"route", "aabb_distance_lower_bound_mm"},
        label="formal census AABB evidence",
    )
    lower = _finite_nonnegative(value.get("aabb_distance_lower_bound_mm"), label="AABB lower bound")
    if lower <= DEFAULT_CONTACT_THRESHOLD_MM:
      raise FormalGeometryAuthorityError("AABB lower bound is not above contact threshold")
    return {"route": route, "aabb_distance_lower_bound_mm": lower}
  raise FormalGeometryAuthorityError("formal census evidence route is not admissible")


def _census_geometry_inputs(
    face_map_authority: AuthenticatedFormalFaceMap,
    case_id: str,
) -> dict[str, Any]:
  import numpy as np

  census = __import__(
      "neurocad.fusion_contact_census",
      fromlist=["CensusGeometryInput"],
  )
  domain = __import__("neurocad.domain_types", fromlist=["Transform"])
  material = face_map_authority._case_material[case_id]
  steps = {row["part"]: row for row in material["steps"]}
  result: dict[str, Any] = {}
  for instance in material["instances"]:
    if instance["visible"] is not True:
      continue
    part = instance["part"]
    step = steps[part]
    transform = instance["world_transform_mm"]
    result[part] = census.CensusGeometryInput(
        part=part,
        geometry_asset=instance["geometry_asset"],
        step_path=Path(step["file"]["path"]),
        step_receipt_path=step["source_relative_path"],
        step_bytes=step["file"]["bytes"],
        step_sha256=step["file"]["sha256"],
        world_transform=domain.Transform(
            rotation=np.asarray(transform["rotation_row_major"], dtype=float).reshape(3, 3),
            translation=np.asarray(transform["translation_mm"], dtype=float),
        ),
        world_transform_chain_sha256=instance["world_transform_chain_sha256"],
        visible=True,
    )
  return result


def _replay_pair_geometry_evidence(
    evidence: Mapping[str, Any],
    *,
    instance_a: Any,
    instance_b: Any,
    kernel: Any,
    bounds_cache: dict[str, Mapping[str, Any]],
) -> None:
  route = evidence["route"]
  if route == "conservative_aabb_lower_bound":
    for instance in (instance_a, instance_b):
      if instance.part not in bounds_cache:
        bounds_cache[instance.part] = kernel.bounds(instance)
    bounds_a = bounds_cache[instance_a.part]
    bounds_b = bounds_cache[instance_b.part]
    try:
      minimum_a = [float(value) for value in bounds_a["minimum_mm"]]
      maximum_a = [float(value) for value in bounds_a["maximum_mm"]]
      minimum_b = [float(value) for value in bounds_b["minimum_mm"]]
      maximum_b = [float(value) for value in bounds_b["maximum_mm"]]
    except (KeyError, TypeError, ValueError) as error:
      raise FormalGeometryAuthorityError("OCC AABB replay is malformed") from error
    gaps = [
        max(0.0, minimum_b[index] - maximum_a[index], minimum_a[index] - maximum_b[index])
        for index in range(3)
    ]
    replayed = math.sqrt(sum(value * value for value in gaps))
    claimed = float(evidence["aabb_distance_lower_bound_mm"])
    if claimed > replayed + 1e-9:
      raise FormalGeometryAuthorityError(
          "claimed AABB lower bound exceeds replayed placed-solid bound"
      )
    return
  try:
    replayed = kernel.evaluate(instance_a, instance_b)
    distance = float(replayed["minimum_solid_distance_mm"])
    volume = float(replayed["solid_common_volume_mm3"])
  except Exception as error:
    raise FormalGeometryAuthorityError("exact OCC pair evidence replay failed") from error
  if (
      not math.isclose(
          float(evidence["minimum_solid_distance_mm"]),
          distance,
          rel_tol=1e-9,
          abs_tol=1e-9,
      )
      or not math.isclose(
          float(evidence["solid_common_volume_mm3"]),
          volume,
          rel_tol=1e-9,
          abs_tol=1e-9,
      )
  ):
    raise FormalGeometryAuthorityError("exact pair evidence differs from OCC replay")


def build_formal_contact_census_authority(
    face_map_authority: AuthenticatedFormalFaceMap,
    *,
    case_pairs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
  """Build a formal closed-world body-pair census from an authenticated face map."""

  if not isinstance(face_map_authority, AuthenticatedFormalFaceMap):
    raise TypeError("formal census producer requires an authenticated formal face map")
  _require_contact_census_promotion(pilot_only=face_map_authority.pilot_only)
  if (
      not face_map_authority.pilot_only
      and face_map_authority.schema_version
      != FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION
  ):
    raise FormalGeometryAuthorityError(
        "formal contact census requires the allowlisted v4 face-map authority"
    )
  face_map_authority.revalidate()
  if set(case_pairs) != set(face_map_authority.case_ids):
    raise FormalGeometryAuthorityError("formal census producer case set differs")
  cases: list[dict[str, Any]] = []
  negative_count = 0
  positive_count = 0
  for case_id in face_map_authority.case_ids:
    parts = tuple(sorted(face_map_authority._case_parts[case_id]))
    expected_pairs = tuple(itertools.combinations(parts, 2))
    source_counts = face_map_authority._source_pair_counts[case_id]
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in case_pairs[case_id]:
      if not isinstance(raw, Mapping) or set(raw) != {"parts", "evidence"}:
        raise FormalGeometryAuthorityError("formal census producer pair row is malformed")
      raw_parts = raw.get("parts")
      if not isinstance(raw_parts, list) or len(raw_parts) != 2 or any(not isinstance(part, str) for part in raw_parts):
        raise FormalGeometryAuthorityError("formal census producer pair identity is malformed")
      pair = tuple(sorted(raw_parts))
      if pair in indexed:
        raise FormalGeometryAuthorityError("formal census producer pair is duplicated")
      indexed[pair] = raw
    if set(indexed) != set(expected_pairs):
      raise FormalGeometryAuthorityError("formal census pair domain is incomplete or has extras")
    pair_rows: list[dict[str, Any]] = []
    for pair in expected_pairs:
      source_count = int(source_counts.get(pair, 0))
      source_positive = source_count > 0
      evidence = _normalize_pair_evidence(indexed[pair]["evidence"], source_positive=source_positive)
      classification = "verified_positive" if source_positive else "certified_negative"
      pair_rows.append(
          {
              "part_a": pair[0],
              "part_b": pair[1],
              "source_contact_count": source_count,
              "source_presence": source_positive,
              "absence_is_negative": False,
              "classification": classification,
              "certified_negative": not source_positive,
              "terminal_status": True,
              "evidence": evidence,
          }
      )
      positive_count += int(source_positive)
      negative_count += int(not source_positive)
    cases.append(
        {
            "case_id": case_id,
            "visible_parts": list(parts),
            "pair_domain": pair_rows,
            "completeness_ledger": {
                "visible_instance_count": len(parts),
                "expected_pair_count": len(expected_pairs),
                "enumerated_pair_count": len(pair_rows),
                "verified_positive_count": sum(row["classification"] == "verified_positive" for row in pair_rows),
                "certified_negative_count": sum(row["classification"] == "certified_negative" for row in pair_rows),
                "unknown_count": 0,
                "discovered_count": 0,
                "source_error_count": 0,
                "all_pairs_terminal": True,
                "formal_data_eligible": True,
            },
        }
    )
  receipt = {
      "schema_version": CONTACT_CENSUS_SCHEMA_VERSION,
      "formal": not face_map_authority.pilot_only,
      "visibility": "private_evaluation_only",
      "source_split": face_map_authority.source_split,
      "thresholds": {
          "contact_threshold_mm": DEFAULT_CONTACT_THRESHOLD_MM,
          "interference_threshold_mm3": DEFAULT_INTERFERENCE_THRESHOLD_MM3,
      },
      "face_map_authority": {
          **_file_binding(face_map_authority._receipt),
          "schema_version": face_map_authority.schema_version,
          "receipt_payload_sha256": _strict_payload_sha(face_map_authority._receipt.raw_bytes),
      },
      "implementation": _implementation_binding(
          _CENSUS_SCHEMA_PATH,
          authority_kind="contact_census",
      ),
      "cases": cases,
      "summary": {
          "case_count": len(cases),
          "instance_pair_count": positive_count + negative_count,
          "verified_positive_count": positive_count,
          "certified_negative_count": negative_count,
          "unknown_count": 0,
          "discovered_candidate_count": 0,
          "source_label_error_count": 0,
          "verified_discovered_count": 0,
          "exhaustive_pair_census_complete": True,
          "formal_eligible": True,
      },
  }
  if face_map_authority.pilot_only:
    receipt["pilot_only"] = True
    receipt["development"] = True
  receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
  _validate_census_payload(receipt, face_map_authority=face_map_authority)
  implementation_captures: list[tuple[str, CapturedFileArtifact]] = []
  _validate_implementation_binding(
      receipt["implementation"],
      schema_path=_CENSUS_SCHEMA_PATH,
      authority_kind="contact_census",
      captures=implementation_captures,
  )
  _revalidate_captures(implementation_captures)
  face_map_authority.revalidate()
  return receipt


def _strict_payload_sha(raw: bytes) -> str:
  try:
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
  except (UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise FormalGeometryAuthorityError("bound authority receipt is invalid JSON") from error
  if not isinstance(payload, Mapping):
    raise FormalGeometryAuthorityError("bound authority receipt is not an object")
  _validate_receipt_sha(payload)
  return str(payload["receipt_payload_sha256"])


def _validate_census_payload(
    payload: Any,
    *,
    face_map_authority: AuthenticatedFormalFaceMap,
) -> dict[tuple[str, str, str], dict[str, Any]]:
  if not isinstance(payload, Mapping):
    raise FormalGeometryAuthorityError("formal census authority is not an object")
  _validate_receipt_sha(payload)
  required = {
      "schema_version", "formal", "visibility", "source_split", "thresholds",
      "face_map_authority", "implementation", "cases", "summary",
      "receipt_payload_sha256",
  }
  if face_map_authority.pilot_only:
    required.update({"pilot_only", "development"})
  if set(payload) != required or (
      payload.get("schema_version") != CONTACT_CENSUS_SCHEMA_VERSION
      or payload.get("formal") is not (not face_map_authority.pilot_only)
      or payload.get("pilot_only") is not (
          True if face_map_authority.pilot_only else None
      )
      or payload.get("development") is not (
          True if face_map_authority.pilot_only else None
      )
      or payload.get("visibility") != "private_evaluation_only"
      or payload.get("source_split") != face_map_authority.source_split
      or payload.get("thresholds") != {
          "contact_threshold_mm": DEFAULT_CONTACT_THRESHOLD_MM,
          "interference_threshold_mm3": DEFAULT_INTERFERENCE_THRESHOLD_MM3,
      }
  ):
    raise FormalGeometryAuthorityError("formal census authority contract is invalid")
  expected_face_binding = {
      **_file_binding(face_map_authority._receipt),
      "schema_version": face_map_authority.schema_version,
      "receipt_payload_sha256": _strict_payload_sha(face_map_authority._receipt.raw_bytes),
  }
  if payload.get("face_map_authority") != expected_face_binding:
    raise FormalGeometryAuthorityError("formal census face-map authority binding differs")
  raw_cases = payload.get("cases")
  if not isinstance(raw_cases, list) or [
      row.get("case_id") if isinstance(row, Mapping) else None for row in raw_cases
  ] != list(face_map_authority.case_ids):
    raise FormalGeometryAuthorityError("formal census exact case set/order differs")
  negatives: dict[tuple[str, str, str], dict[str, Any]] = {}
  positive_count = 0
  negative_count = 0
  pair_count = 0
  census_module = __import__(
      "neurocad.fusion_contact_census",
      fromlist=["OccWholeSolidContactKernel"],
  )
  geometry_kernel = census_module.OccWholeSolidContactKernel()
  pair_keys = {
      "part_a", "part_b", "source_contact_count", "source_presence",
      "absence_is_negative", "classification", "certified_negative",
      "terminal_status", "evidence",
  }
  ledger_keys = {
      "visible_instance_count", "expected_pair_count",
      "enumerated_pair_count", "verified_positive_count",
      "certified_negative_count", "unknown_count", "discovered_count",
      "source_error_count", "all_pairs_terminal", "formal_data_eligible",
  }
  for case in raw_cases:
    if not isinstance(case, Mapping) or set(case) != {
        "case_id", "visible_parts", "pair_domain", "completeness_ledger"
    }:
      raise FormalGeometryAuthorityError("formal census case contract is malformed")
    case_id = str(case["case_id"])
    parts = tuple(sorted(face_map_authority._case_parts[case_id]))
    if case.get("visible_parts") != list(parts):
      raise FormalGeometryAuthorityError("formal census visible instance domain differs")
    expected_pairs = tuple(itertools.combinations(parts, 2))
    raw_pairs = case.get("pair_domain")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(expected_pairs):
      raise FormalGeometryAuthorityError("formal census pair domain is incomplete")
    seen: set[tuple[str, str]] = set()
    case_positive = 0
    case_negative = 0
    source_counts = face_map_authority._source_pair_counts[case_id]
    geometry_inputs = _census_geometry_inputs(face_map_authority, case_id)
    bounds_cache: dict[str, Mapping[str, Any]] = {}
    for row in raw_pairs:
      row = _require_exact_keys(
          row,
          pair_keys,
          label="formal census pair row",
      )
      _require_strict_int_fields(
          row,
          ("source_contact_count",),
          label="formal census pair row",
      )
      _require_strict_bool_fields(
          row,
          (
              "source_presence", "absence_is_negative",
              "certified_negative", "terminal_status",
          ),
          label="formal census pair row",
      )
      pair = (row.get("part_a"), row.get("part_b"))
      if pair not in expected_pairs or pair in seen:
        raise FormalGeometryAuthorityError("formal census pair identity is missing, extra, or duplicated")
      seen.add(pair)  # type: ignore[arg-type]
      source_count = int(source_counts.get(pair, 0))  # type: ignore[arg-type]
      source_positive = source_count > 0
      expected_classification = "verified_positive" if source_positive else "certified_negative"
      if any(
          row.get(key) != value
          for key, value in {
              "source_contact_count": source_count,
              "source_presence": source_positive,
              "absence_is_negative": False,
              "classification": expected_classification,
              "certified_negative": not source_positive,
              "terminal_status": True,
          }.items()
      ):
        raise FormalGeometryAuthorityError("formal census pair semantic classification differs")
      normalized = _normalize_pair_evidence(row.get("evidence"), source_positive=source_positive)
      if row.get("evidence") != normalized:
        raise FormalGeometryAuthorityError("formal census pair evidence is not canonical")
      _replay_pair_geometry_evidence(
          normalized,
          instance_a=geometry_inputs[pair[0]],
          instance_b=geometry_inputs[pair[1]],
          kernel=geometry_kernel,
          bounds_cache=bounds_cache,
      )
      pair_count += 1
      case_positive += int(source_positive)
      case_negative += int(not source_positive)
      if not source_positive:
        negatives[(case_id, pair[0], pair[1])] = copy.deepcopy(dict(row))  # type: ignore[index]
    if seen != set(expected_pairs):
      raise FormalGeometryAuthorityError("formal census pair domain does not close")
    expected_ledger = {
        "visible_instance_count": len(parts),
        "expected_pair_count": len(expected_pairs),
        "enumerated_pair_count": len(expected_pairs),
        "verified_positive_count": case_positive,
        "certified_negative_count": case_negative,
        "unknown_count": 0,
        "discovered_count": 0,
        "source_error_count": 0,
        "all_pairs_terminal": True,
        "formal_data_eligible": True,
    }
    ledger = _require_exact_keys(
        case.get("completeness_ledger"),
        ledger_keys,
        label="formal census case ledger",
    )
    _require_strict_int_fields(
        ledger,
        (
            "visible_instance_count", "expected_pair_count",
            "enumerated_pair_count", "verified_positive_count",
            "certified_negative_count", "unknown_count", "discovered_count",
            "source_error_count",
        ),
        label="formal census case ledger",
    )
    _require_strict_bool_fields(
        ledger,
        ("all_pairs_terminal", "formal_data_eligible"),
        label="formal census case ledger",
    )
    if ledger != expected_ledger:
      raise FormalGeometryAuthorityError("formal census case ledger does not close")
    positive_count += case_positive
    negative_count += case_negative
  expected_summary = {
      "case_count": len(face_map_authority.case_ids),
      "instance_pair_count": pair_count,
      "verified_positive_count": positive_count,
      "certified_negative_count": negative_count,
      "unknown_count": 0,
      "discovered_candidate_count": 0,
      "source_label_error_count": 0,
      "verified_discovered_count": 0,
      "exhaustive_pair_census_complete": True,
      "formal_eligible": True,
  }
  summary = _require_exact_keys(
      payload.get("summary"),
      set(expected_summary),
      label="formal census summary",
  )
  _require_strict_int_fields(
      summary,
      (
          "case_count", "instance_pair_count", "verified_positive_count",
          "certified_negative_count", "unknown_count",
          "discovered_candidate_count", "source_label_error_count",
          "verified_discovered_count",
      ),
      label="formal census summary",
  )
  _require_strict_bool_fields(
      summary,
      ("exhaustive_pair_census_complete", "formal_eligible"),
      label="formal census summary",
  )
  if summary != expected_summary:
    raise FormalGeometryAuthorityError("formal census summary does not close")
  return negatives


def _contact_census_memory_state_payload(
    *,
    case_ids: Sequence[str],
    source_split: str,
    pilot_only: bool,
    receipt: CapturedFileArtifact,
    captures: Sequence[tuple[str, CapturedFileArtifact]],
    face_map: AuthenticatedFormalFaceMap,
    certified_negatives: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
  return {
      "kind": "AuthenticatedFormalContactCensus",
      "case_ids": list(case_ids),
      "source_split": str(source_split),
      "pilot_only": bool(pilot_only),
      "receipt": _capture_memory_binding("receipt", receipt),
      "captures": [
          _capture_memory_binding(label, captured)
          for label, captured in captures
      ],
      "face_map_identity": id(face_map),
      "certified_negatives": [
          {"key": list(key), "value": _recursive_mutable_copy(value)}
          for key, value in sorted(certified_negatives.items())
      ],
  }


class AuthenticatedFormalContactCensus(_ImmutableAuthenticatedCapability):
  """Ephemeral capability exposing only receipt-certified body-pair negatives."""

  __slots__ = (
      "case_ids", "source_split", "pilot_only", "_receipt", "_captures", "_face_map",
      "_certified_negatives",
  )

  def __init__(
      self,
      *,
      case_ids: Sequence[str],
      source_split: str,
      receipt: CapturedFileArtifact,
      captures: Sequence[tuple[str, CapturedFileArtifact]],
      face_map: AuthenticatedFormalFaceMap,
      certified_negatives: Mapping[tuple[str, str, str], Mapping[str, Any]],
      _issuance_context: object,
  ) -> None:
    try:
      _require_verified_issuance_context(
          _issuance_context,
          _CENSUS_CAPABILITY_KIND,
      )
    except TypeError as error:
      raise TypeError(
          "AuthenticatedFormalContactCensus must be created by its verifier "
          "with a one-shot verified issuance context"
      ) from error
    self.case_ids = tuple(case_ids)
    self.source_split = source_split
    self.pilot_only = bool(face_map.pilot_only)
    self._receipt = receipt
    self._captures = tuple(captures)
    self._face_map = face_map
    self._certified_negatives = _recursive_freeze(
        copy.deepcopy(dict(certified_negatives))
    )
    try:
      self._register_memory_state(
          self._memory_state_payload(),
          issuance_context=_issuance_context,
          capability_kind=_CENSUS_CAPABILITY_KIND,
      )
    except (FormalGeometryAuthorityError, TypeError) as error:
      raise TypeError(
          "AuthenticatedFormalContactCensus must be created by its verifier "
          "with a one-shot verified issuance context"
      ) from error
    self._seal()

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated formal contact census is not serializable")

  def _memory_state_payload(self) -> dict[str, Any]:
    return _contact_census_memory_state_payload(
        case_ids=self.case_ids,
        source_split=self.source_split,
        pilot_only=self.pilot_only,
        receipt=self._receipt,
        captures=self._captures,
        face_map=self._face_map,
        certified_negatives=self._certified_negatives,
    )

  def revalidate(self) -> None:
    self._require_memory_state(self._memory_state_payload())
    self._face_map.revalidate()
    _revalidate_captures((("formal contact census authority", self._receipt), *self._captures))

  def require_certified_negative(self, case_id: str, part_a: str, part_b: str) -> dict[str, Any]:
    self.revalidate()
    left, right = sorted((part_a, part_b))
    try:
      return _recursive_mutable_copy(
          self._certified_negatives[(case_id, left, right)]
      )
    except KeyError as error:
      raise FormalGeometryAuthorityError("body pair is not a certified negative") from error


def verify_formal_contact_census_authority(
    receipt_path: str | Path,
    *,
    face_map_receipt_path: str | Path,
    family_split_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    source_split: str,
    pilot_only: bool = False,
    obj_extension_authority: AuthenticatedContentReceiptObjExtension | None = None,
) -> _VerifiedHandleConstruction:
  """Authenticate a closed-world census and its entire face/source evidence graph."""

  _require_contact_census_promotion(pilot_only=pilot_only)
  face_probe = _capture_json_strict(
      face_map_receipt_path,
      label="contact census face-map authority probe",
  )
  face_schema = (
      face_probe.json.payload.get("schema_version")
      if isinstance(face_probe.json.payload, Mapping)
      else None
  )
  common = {
      "family_split_manifest_path": family_split_manifest_path,
      "public_cases_path": public_cases_path,
      "private_source_path": private_source_path,
      "private_gold_path": private_gold_path,
      "archive_root": archive_root,
      "restored_root": restored_root,
      "source_split": source_split,
      "pilot_only": pilot_only,
  }
  if face_schema == FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION:
    if not isinstance(
        obj_extension_authority,
        AuthenticatedContentReceiptObjExtension,
    ):
      raise FormalGeometryAuthorityError(
          "v4 contact census verification requires authenticated OBJ extension"
      )
    face_map = verify_formal_face_map_authority_v4(
        face_map_receipt_path,
        obj_extension_authority=obj_extension_authority,
        **common,
    )
  elif face_schema == FACE_MAP_SCHEMA_VERSION and pilot_only:
    face_map = verify_formal_face_map_authority(
        face_map_receipt_path,
        **common,
    )
  else:
    raise FormalGeometryAuthorityError(
        "formal contact census rejects legacy or unknown face-map authority"
    )
  receipt = _capture_json_strict(receipt_path, label="formal contact census authority")
  captures: list[tuple[str, CapturedFileArtifact]] = []
  _validate_implementation_binding(
      receipt.json.payload.get("implementation") if isinstance(receipt.json.payload, Mapping) else None,
      schema_path=_CENSUS_SCHEMA_PATH,
      authority_kind="contact_census",
      captures=captures,
  )
  negatives = _validate_census_payload(receipt.json.payload, face_map_authority=face_map)
  _revalidate_captures((("formal contact census authority", receipt.file), *captures))
  face_map.revalidate()
  constructor_kwargs = {
      "case_ids": face_map.case_ids,
      "source_split": face_map.source_split,
      "receipt": receipt.file,
      "captures": tuple(captures),
      "face_map": face_map,
      "certified_negatives": negatives,
  }
  return _VerifiedHandleConstruction(
      kwargs=MappingProxyType(constructor_kwargs),
      memory_state_commitment=canonical_sha256(
          _contact_census_memory_state_payload(
              pilot_only=face_map.pilot_only,
              **constructor_kwargs,
          )
      ),
  )


verify_formal_face_map_authority = _wrap_verified_verifier(
    verify_formal_face_map_authority,
    capability_kind=_FACE_CAPABILITY_KIND,
    constructor=AuthenticatedFormalFaceMap,
)
verify_formal_face_map_authority_v4 = _wrap_verified_verifier(
    verify_formal_face_map_authority_v4,
    capability_kind=_FACE_CAPABILITY_KIND,
    constructor=AuthenticatedFormalFaceMap,
)
verify_formal_contact_census_authority = _wrap_verified_verifier(
    verify_formal_contact_census_authority,
    capability_kind=_CENSUS_CAPABILITY_KIND,
    constructor=AuthenticatedFormalContactCensus,
)
del _wrap_verified_verifier


__all__ = [
    "FACE_MAP_SCHEMA_VERSION",
    "FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION",
    "CONTACT_CENSUS_SCHEMA_VERSION",
    "ENDPOINT_LOOKUP_SCHEMA_VERSION",
    "FORMAL_FACE_MAP_AUTHORITY_SCHEMA_ALLOWLIST",
    "FORMAL_CONTACT_CENSUS_SCHEMA_ALLOWLIST",
    "AuthenticatedFormalFaceMap",
    "AuthenticatedFormalContactCensus",
    "FormalGeometryAuthorityError",
    "build_formal_face_map_audit",
    "verify_formal_face_map_authority",
    "build_formal_face_map_audit_v4",
    "verify_formal_face_map_authority_v4",
    "build_formal_contact_census_authority",
    "verify_formal_contact_census_authority",
]
