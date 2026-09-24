"""Independent semantic replay authority for formal mate-program libraries.

The mate library and its manifest are outputs under verification.  Neither may
authorize its own residual transforms, program identifiers, or coverage.  This
module therefore issues only ephemeral handles after replaying those semantics
from private source/gold receipts, bound STEP bytes, and the frozen mining
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import weakref

from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    CapturedJsonArtifact,
    canonical_sha256,
    capture_file_artifact,
    reverify_captured_file_artifact,
)


MATE_SEMANTIC_AUTHORITY_SCHEMA = "mate_semantic_authority.v1"
MATE_SEMANTIC_AUTHORITY_V2_SCHEMA = "mate_semantic_authority.v2"
FORMAL_MATE_SEMANTIC_AUTHORITY_SCHEMA_ALLOWLIST: frozenset[str] = frozenset(
    {MATE_SEMANTIC_AUTHORITY_V2_SCHEMA}
)

_REPLAY_CONTRACT = (
    "private_source_gold_step_face_map_occurrence_cm_mm_frame_seed_exact_rows.v1"
)
_REPLAY_CONTRACT_V2 = (
    "private_source_gold_multi_archive_step_face_map_occurrence_cm_mm_"
    "frame_seed_exact_rows.v2"
)
_SHA256 = frozenset("0123456789abcdef")

_MATE_CAPABILITY_KIND = "mate_semantic_authority"
_CAPABILITY_STATE_LOCK = threading.RLock()
_CAPABILITY_STATE_REGISTRY: weakref.WeakKeyDictionary[Any, str] = (
    weakref.WeakKeyDictionary()
)
_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "mate_semantic_authority.schema.json"
)
_SCHEMA_V2_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "mate_semantic_authority_v2.schema.json"
)


def _require_mate_promotion(
    *,
    pilot_only: bool,
    schema_version: str = MATE_SEMANTIC_AUTHORITY_SCHEMA,
) -> None:
  if pilot_only:
    return
  if (
      schema_version
      not in FORMAL_MATE_SEMANTIC_AUTHORITY_SCHEMA_ALLOWLIST
  ):
    raise ValueError(
        "formal mate semantic promotion requires an allowlisted mate semantic "
        "authority after multi-archive v2 full-chain verification"
    )


@dataclass(frozen=True, slots=True)
class _VerifiedHandleConstruction:
  """Constructor material returned only after semantic replay succeeds."""

  kwargs: Mapping[str, Any]
  memory_state_commitment: str


def _make_verifier_only_issuance_boundary():
  """Create an issuer available only inside the verifier wrapper closure.

  This protects against public/ordinary API misuse.  It does not claim to
  resist private closure introspection or module monkeypatching.
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
            "mate semantic authority requires a one-shot verified issuance context"
        ) from error
      if expected is None or expected[0] != capability_kind:
        raise TypeError(
            "mate semantic authority requires a one-shot verified issuance context"
        )

  def consume(context: object, capability_kind: str, commitment: str) -> None:
    with lock:
      try:
        expected = pending.pop(context)
      except (KeyError, TypeError) as error:
        raise TypeError(
            "mate semantic authority requires a one-shot verified issuance context"
        ) from error
      if expected != (capability_kind, commitment):
        raise TypeError(
            "verified issuance context does not match capability kind and "
            "validated memory-state commitment"
        )

  def wrap(verifier: Any, *, constructor: Any) -> Any:
    @functools.wraps(verifier)
    def verified_issuer(*args: Any, **kwargs: Any) -> Any:
      construction = verifier(*args, **kwargs)
      if type(construction) is not _VerifiedHandleConstruction:
        raise TypeError("verifier did not return validated construction material")
      context = _VerifiedIssuanceContext()
      with lock:
        pending[context] = (
            _MATE_CAPABILITY_KIND,
            construction.memory_state_commitment,
        )
      try:
        return constructor(
            **dict(construction.kwargs),
            _issuance_context=context,
        )
      finally:
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


def _implementation_paths(
    schema_version: str = MATE_SEMANTIC_AUTHORITY_SCHEMA,
) -> dict[str, Path]:
  root = Path(__file__).resolve().parent
  paths = {
      "neurocad.mate_semantic_authority": Path(__file__),
      "neurocad.mate_pose_miner": root / "mate_pose_miner.py",
  }
  if schema_version == MATE_SEMANTIC_AUTHORITY_V2_SCHEMA:
    paths["neurocad.formal_archive_inputs"] = root / "formal_archive_inputs.py"
  return paths


def _capture_implementation_binding(
    schema_version: str = MATE_SEMANTIC_AUTHORITY_SCHEMA,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
  sources = {
      name: capture_file_artifact(path, label=f"mate semantic source {name}")
      for name, path in _implementation_paths(schema_version).items()
  }
  schema_path = (
      _SCHEMA_V2_PATH
      if schema_version == MATE_SEMANTIC_AUTHORITY_V2_SCHEMA
      else _SCHEMA_PATH
  )
  schema = capture_file_artifact(
      schema_path,
      label="mate semantic authority schema",
  )
  binding = {
      "module": "neurocad.mate_semantic_authority",
      "source_sha256s": {
          name: captured.sha256 for name, captured in sorted(sources.items())
      },
      "schema": {
          "bytes": schema.byte_count,
          "sha256": schema.sha256,
      },
  }
  return binding, (*sources.values(), schema)


def _recursive_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType(
        {key: _recursive_freeze(item) for key, item in value.items()}
    )
  if isinstance(value, (list, tuple)):
    return tuple(_recursive_freeze(item) for item in value)
  if isinstance(value, (set, frozenset)):
    return frozenset(_recursive_freeze(item) for item in value)
  return value


def _immutable_byte_capture(captured: Any) -> CapturedFileArtifact:
  """Drop mutable parsed payloads while retaining the exact file identity."""

  from .formal_archive_inputs import LightweightFileCapture

  if isinstance(captured, LightweightFileCapture):
    return captured
  if isinstance(captured, CapturedFileArtifact):
    return captured
  return CapturedFileArtifact(
      path=captured.path,
      resolved_path=captured.resolved_path,
      raw_bytes=captured.raw_bytes,
      sha256=captured.sha256,
      byte_count=captured.byte_count,
      device=captured.device,
      inode=captured.inode,
      link_count=captured.link_count,
  )


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
      raise ValueError(
          "mate semantic authority has no matching verified issuance context"
      ) from error
    with _CAPABILITY_STATE_LOCK:
      if self in _CAPABILITY_STATE_REGISTRY:
        raise ValueError(
            "mate semantic authority memory state is already registered"
        )
      _CAPABILITY_STATE_REGISTRY[self] = commitment

  def _require_memory_state(self, payload: Mapping[str, Any]) -> None:
    try:
      commitment = canonical_sha256(dict(payload))
      with _CAPABILITY_STATE_LOCK:
        expected = _CAPABILITY_STATE_REGISTRY.get(self)
    except Exception as error:
      raise ValueError("mate semantic authority memory state is malformed") from error
    if expected is None or commitment != expected:
      raise ValueError("mate semantic authority memory state commitment differs")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key: {key}")
    result[key] = value
  return result


def _captured_json(path: str | Path, *, label: str):
  _reject_reparse_path(path, label=label)
  captured = capture_file_artifact(path, label=label)
  try:
    payload = json.loads(
        captured.raw_bytes.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
    )
  except (UnicodeError, json.JSONDecodeError, ValueError) as error:
    raise ValueError(f"{label} is invalid duplicate-key-safe JSON") from error
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
  return json_capture, payload


def _reverify_capture(captured: Any, *, label: str) -> None:
  """Revalidate either a byte capture or its parsed JSON counterpart."""

  from .formal_archive_inputs import LightweightFileCapture

  if isinstance(captured, LightweightFileCapture):
    captured.revalidate()
    return
  if isinstance(captured, CapturedJsonArtifact):
    current, _payload = _captured_json(captured.path, label=label)
    if current != captured:
      raise ValueError(f"{label} changed since capture")
    return
  reverify_captured_file_artifact(captured, label=label)


def _reject_reparse_path(path: str | Path, *, label: str) -> None:
  requested = Path(os.path.abspath(Path(path)))
  for entry in (requested, *requested.parents):
    try:
      stat = entry.lstat()
    except OSError:
      continue
    attributes = int(getattr(stat, "st_file_attributes", 0))
    is_junction = getattr(entry, "is_junction", None)
    if (
        entry.is_symlink()
        or attributes & 0x400
        or (callable(is_junction) and is_junction())
    ):
      raise ValueError(f"{label} path traverses a symlink or reparse point")


def _is_sha256(value: Any) -> bool:
  return (
      isinstance(value, str)
      and len(value) == 64
      and all(character in _SHA256 for character in value)
  )


def _safe_archive_member(root: Path, raw_path: Any, *, label: str) -> Path:
  text = str(raw_path or "").strip().replace("\\", "/")
  relative = PurePosixPath(text)
  if (
      not text
      or relative.is_absolute()
      or any(part in {"", ".", ".."} for part in relative.parts)
      or relative.as_posix() != text
  ):
    raise ValueError(f"{label} has an unsafe archive-relative path")
  candidate = root.joinpath(*relative.parts)
  try:
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
  except OSError as error:
    raise ValueError(f"{label} is missing from archive_root") from error
  try:
    actual_relative = resolved.relative_to(resolved_root)
  except ValueError as error:
    raise ValueError(f"{label} escapes archive_root") from error
  if actual_relative.as_posix() != text:
    raise ValueError(f"{label} path traverses an alias or reparse point")
  _reject_reparse_path(candidate, label=label)
  return candidate


def _library_rows(raw_bytes: bytes) -> list[dict[str, Any]]:
  try:
    lines = raw_bytes.decode("utf-8").splitlines()
  except UnicodeError as error:
    raise ValueError("mate library is not UTF-8") from error
  rows: list[dict[str, Any]] = []
  for row_number, line in enumerate(lines, start=1):
    if not line.strip():
      continue
    try:
      row = json.loads(line, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, ValueError) as error:
      raise ValueError(
          f"mate library row {row_number} is invalid duplicate-key-safe JSON"
      ) from error
    if not isinstance(row, dict):
      raise ValueError(f"mate library row {row_number} is not an object")
    rows.append(row)
  if not rows:
    raise ValueError("mate semantic authority refuses an empty library")
  return rows


def _row_commitment(row: Mapping[str, Any]) -> dict[str, Any]:
  source_contact = row.get("source_contact")
  metadata = row.get("metadata")
  if not isinstance(source_contact, Mapping) or not isinstance(metadata, Mapping):
    raise ValueError("mate semantic replay row lacks source/metadata bindings")
  residual = {
      "rotation": row.get("residual_rotation"),
      "translation": row.get("residual_translation"),
  }
  return {
      "program_id": str(row.get("program_id") or ""),
      "source_case_token_sha256": str(
          metadata.get("source_case_token_sha256") or ""
      ),
      "source_contact_ordinal": source_contact.get("contact_index"),
      "direction": source_contact.get("direction"),
      "endpoint_a_sha256": canonical_sha256(row.get("interface_a")),
      "endpoint_b_sha256": canonical_sha256(row.get("interface_b")),
      "residual_sha256": canonical_sha256(residual),
      "row_sha256": canonical_sha256(dict(row)),
  }


def _manifest_contract(
    manifest: Any,
    *,
    library_file_sha256: str,
) -> dict[str, Any]:
  if not isinstance(manifest, Mapping):
    raise ValueError("mate library manifest must be an object")
  unsigned = dict(manifest)
  stored_manifest_sha = unsigned.pop("manifest_sha256", None)
  if stored_manifest_sha != canonical_sha256(unsigned):
    raise ValueError("mate library manifest payload hash mismatch")
  expected = {
      "schema_version": "mate_program_library_manifest.v3",
      "model_input_protocol": "benchmark_v2",
      "source_split": "train",
      "library_sha256": library_file_sha256,
  }
  if any(manifest.get(key) != value for key, value in expected.items()):
    raise ValueError("mate library manifest protocol/library binding differs")
  mining_config = manifest.get("mining_config")
  if not isinstance(mining_config, Mapping):
    raise ValueError("mate library manifest lacks mining_config")
  if (
      type(mining_config.get("max_candidates_per_part")) is not int
      or mining_config.get("max_candidates_per_part") != 0
  ):
    raise ValueError(
        "mate semantic authority requires max_candidates_per_part == 0"
    )
  if (
      type(mining_config.get("max_candidates_per_contact_face")) is not int
      or int(mining_config["max_candidates_per_contact_face"]) < 1
  ):
    raise ValueError(
        "mate semantic authority requires max_candidates_per_contact_face >= 1"
    )
  if manifest.get("mining_config_sha256") != canonical_sha256(mining_config):
    raise ValueError("mate library manifest mining_config hash mismatch")
  from .mate_pose_retriever import mate_miner_code_bundle_identity

  expected_code_bundle = mate_miner_code_bundle_identity()
  if (
      manifest.get("miner_code_bundle") != expected_code_bundle
      or manifest.get("miner_code_bundle_sha256")
      != expected_code_bundle["bundle_sha256"]
  ):
    raise ValueError("mate semantic replay code bundle differs from the manifest")
  if not _is_sha256(manifest.get("family_split_manifest_sha256")):
    raise ValueError("mate library manifest family hash is invalid")
  if not _is_sha256(manifest.get("source_split_sha256")):
    raise ValueError("mate library manifest train split hash is invalid")
  row_count = manifest.get("row_count")
  if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
    raise ValueError("mate library manifest row_count is invalid")
  return dict(manifest)


def _capture_archive_inputs(
    source_payload: Any,
    *,
    archive_root: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
  if not archive_root.is_dir() or archive_root.is_symlink():
    raise ValueError("archive_root must be a plain existing directory")
  _reject_reparse_path(archive_root, label="archive_root")
  cases = source_payload.get("cases") if isinstance(source_payload, Mapping) else None
  if not isinstance(cases, list) or not cases:
    raise ValueError("private source receipt has no cases")
  captures = []
  ledger: list[dict[str, Any]] = []
  captured_by_path: dict[str, Any] = {}
  for case in cases:
    receipt = case.get("source_receipt_binding") if isinstance(case, Mapping) else None
    if not isinstance(receipt, Mapping):
      raise ValueError("private source case lacks a receipt binding")
    records = [("assembly_json", receipt.get("assembly_json"))]
    body_steps = receipt.get("body_steps")
    if not isinstance(body_steps, list):
      raise ValueError("private source case lacks STEP receipts")
    records.extend(("body_step", record) for record in body_steps)
    for record_kind, record in records:
      if not isinstance(record, Mapping):
        raise ValueError("private source file receipt is malformed")
      relative_path = str(record.get("path") or "")
      if relative_path in captured_by_path:
        # Repeated physical instances may legitimately share one immutable STEP
        # geometry asset. Instance identity remains distinct in the private
        # source receipt; byte capture is deduplicated by archive member path.
        continue
      path = _safe_archive_member(
          archive_root,
          relative_path,
          label=f"receipt-bound archive member {relative_path!r}",
      )
      captured = capture_file_artifact(
          path,
          label=f"receipt-bound archive member {relative_path!r}",
      )
      if record_kind == "assembly_json":
        try:
          assembly = json.loads(
              captured.raw_bytes.decode("utf-8"),
              object_pairs_hook=_unique_json_object,
          )
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
          raise ValueError(
              "receipt-bound assembly JSON is invalid duplicate-key-safe JSON"
          ) from error
        if not isinstance(assembly, Mapping):
          raise ValueError("receipt-bound assembly JSON must be an object")
        if (
            record.get("sha256") != captured.sha256
            or type(record.get("bytes")) is not int
            or record.get("bytes") != captured.byte_count
        ):
          raise ValueError("receipt-bound assembly JSON byte binding differs")
      captured_by_path[relative_path] = captured
      captures.append(captured)
      ledger.append(
          {
              "path": relative_path,
              "sha256": captured.sha256,
              "bytes": captured.byte_count,
          }
      )
  ledger.sort(key=lambda row: row["path"])
  return captures, ledger


def _formal_v4_replay_inputs(
    *,
    public_cases: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    family_split_manifest_path: str | Path,
    train_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    face_map_receipt_path: str | Path,
    archive_root: str | Path,
    restored_root: str | Path,
    obj_extension_authority: Any | None,
) -> tuple[tuple[dict[str, Any], ...], Any, Any]:
  from .content_receipt_obj_extension import (
      AuthenticatedContentReceiptObjExtension,
  )

  if not isinstance(
      obj_extension_authority,
      AuthenticatedContentReceiptObjExtension,
  ):
    raise TypeError(
        "formal v4 semantic replay requires authenticated OBJ extension authority"
    )
  normalized_case_ids = tuple(str(case_id) for case_id in case_ids)
  normalized_public = tuple(dict(case) for case in public_cases)
  if tuple(str(case.get("id") or "") for case in normalized_public) != (
      normalized_case_ids
  ):
    raise ValueError("formal v4 semantic public case order differs")
  from .benchmark_v2_training_provenance import validate_family_split_binding
  from .benchmark_v2_private_supervision import (
      load_benchmark_v2_private_supervision,
  )

  validate_family_split_binding(
      cases_path=train_path,
      manifest_path=family_split_manifest_path,
      source_split="train",
      case_ids=normalized_case_ids,
      pilot_only=False,
  )
  private_supervision = load_benchmark_v2_private_supervision(
      source_path=private_source_path,
      gold_path=private_gold_path,
      family_split_manifest_path=family_split_manifest_path,
      source_split="train",
      public_case_ids=normalized_case_ids,
      pilot_only=False,
  )
  from .formal_geometry_authority import verify_formal_face_map_authority_v4

  face_authority = verify_formal_face_map_authority_v4(
      face_map_receipt_path,
      family_split_manifest_path=family_split_manifest_path,
      public_cases_path=train_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      archive_root=archive_root,
      restored_root=restored_root,
      source_split="train",
      obj_extension_authority=obj_extension_authority,
      pilot_only=False,
  )
  face_authority.revalidate()
  if (
      tuple(face_authority.case_ids) != normalized_case_ids
      or face_authority.source_split != "train"
      or face_authority.pilot_only is not False
  ):
    raise ValueError("formal v4 face authority domain differs from semantic inputs")
  return normalized_public, private_supervision, face_authority.endpoint_lookup()


def _replay_inputs(
    *,
    library_path: str | Path,
    manifest_path: str | Path,
    family_split_manifest_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    face_map_receipt_path: str | Path,
    archive_root: str | Path,
    pilot_only: bool = False,
    geometry_dataset_root: str | Path | None = None,
    authority_schema: str = MATE_SEMANTIC_AUTHORITY_SCHEMA,
    authenticated_face_map: Any | None = None,
    obj_extension_authority: Any | None = None,
) -> tuple[dict[str, Any], list[Any]]:
  if authenticated_face_map is not None and not pilot_only:
    raise ValueError(
        "authenticated_face_map reuse is restricted to non-publishing pilots"
    )
  if obj_extension_authority is not None and pilot_only:
    raise ValueError(
        "obj_extension_authority is restricted to formal v4 semantic replay"
    )
  captures: list[Any] = []

  def capture_json(path: str | Path, label: str):
    captured, payload = _captured_json(path, label=label)
    captures.append(captured)
    return captured, payload

  _reject_reparse_path(library_path, label="mate library")
  library_capture = capture_file_artifact(library_path, label="mate library")
  captures.append(library_capture)
  library_rows = _library_rows(library_capture.raw_bytes)
  manifest_capture, manifest_payload = capture_json(
      manifest_path, "mate library manifest"
  )
  family_capture, _family_payload = capture_json(
      family_split_manifest_path, "family split manifest"
  )
  source_capture, source_payload = capture_json(
      private_source_path, "private source split"
  )
  gold_capture, _gold_payload = capture_json(
      private_gold_path, "private evaluation gold split"
  )
  face_capture, _face_payload = capture_json(
      face_map_receipt_path, "Fusion STEP face-map receipt"
  )
  face_schema = (
      _face_payload.get("schema_version")
      if isinstance(_face_payload, Mapping)
      else None
  )
  manifest = _manifest_contract(
      manifest_payload,
      library_file_sha256=library_capture.sha256,
  )

  if pilot_only:
    if not isinstance(_family_payload, Mapping):
      raise ValueError("pilot family manifest is malformed")
    artifact_hashes = _family_payload.get("artifact_hashes")
    train_binding = (
        artifact_hashes.get("train")
        if isinstance(artifact_hashes, Mapping)
        else None
    )
    if not isinstance(train_binding, Mapping):
      raise ValueError("pilot family manifest lacks its train binding")
    train_path = Path(str(train_binding.get("path") or ""))
    train_capture, train_payload = capture_json(
        train_path, "pilot family train split"
    )
    if train_capture.sha256 != train_binding.get("sha256"):
      raise ValueError("pilot family train bytes differ")
    if not isinstance(train_payload, list):
      raise ValueError("pilot family train split is malformed")
    case_ids = tuple(str(case.get("id") or "") for case in train_payload)
    from .benchmark_v2_training_provenance import validate_family_split_binding
    from .benchmark_v2_private_supervision import (
        load_benchmark_v2_private_supervision,
    )
    validate_family_split_binding(
        cases_path=train_path,
        manifest_path=family_split_manifest_path,
        source_split="train",
        case_ids=case_ids,
        _cases_capture=train_capture,
        _manifest_capture=family_capture,
        pilot_only=True,
    )
    private_supervision = load_benchmark_v2_private_supervision(
        source_path=private_source_path,
        gold_path=private_gold_path,
        family_split_manifest_path=family_split_manifest_path,
        source_split="train",
        public_case_ids=case_ids,
        _source_capture=source_capture,
        _gold_capture=gold_capture,
        _family_manifest_capture=family_capture,
        pilot_only=True,
    )
    if geometry_dataset_root is None:
      raise ValueError("pilot semantic replay requires geometry_dataset_root")
    from .formal_geometry_authority import (
        AuthenticatedFormalFaceMap,
        FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION,
        FACE_MAP_SCHEMA_VERSION,
        verify_formal_face_map_authority,
        verify_formal_face_map_authority_v4,
    )
    if authenticated_face_map is not None:
      if not isinstance(authenticated_face_map, AuthenticatedFormalFaceMap):
        raise TypeError(
            "authenticated_face_map must be issued by a formal face-map verifier"
        )
      face_authority = authenticated_face_map
      face_authority.revalidate()
      if (
          not face_authority.pilot_only
          or face_authority.source_split != "train"
          or tuple(face_authority.case_ids) != tuple(case_ids)
          or face_authority._receipt.sha256 != face_capture.sha256
      ):
        raise ValueError(
            "authenticated face-map handle differs from the pilot semantic inputs"
        )
    else:
      if face_schema == FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION:
        face_verifier = verify_formal_face_map_authority_v4
      elif face_schema == FACE_MAP_SCHEMA_VERSION:
        face_verifier = verify_formal_face_map_authority
      else:
        raise ValueError("pilot semantic replay face-map schema is unsupported")
      face_authority = face_verifier(
          face_map_receipt_path,
          family_split_manifest_path=family_split_manifest_path,
          public_cases_path=train_path,
          private_source_path=private_source_path,
          private_gold_path=private_gold_path,
          archive_root=geometry_dataset_root,
          restored_root=geometry_dataset_root,
          source_split="train",
          pilot_only=True,
      )
    public_cases = tuple(dict(case) for case in train_payload)
    face_map_lookup = face_authority.endpoint_lookup()
  else:
    # Public family replay authenticates the frozen train artifact and the
    # authoritative source split. It is deliberately independent of the library.
    from .mate_pose_retriever import validate_formal_family_split_manifest

    family = validate_formal_family_split_manifest(family_split_manifest_path)
    if family["manifest_sha256"] != family_capture.sha256:
      raise ValueError("mate semantic authority family manifest binding differs")
    train_path = Path(family["artifact_paths"]["train"])
    train_capture, train_payload = capture_json(train_path, "formal family train split")
    dev_path = Path(family["artifact_paths"]["dev"])
    capture_json(dev_path, "formal family dev split")
    source_family_path = Path(family["authoritative_source"]["path"])
    capture_json(source_family_path, "formal family authoritative source")
    if not isinstance(train_payload, list) or not all(
        isinstance(case, Mapping) for case in train_payload
    ):
      raise ValueError("formal family train split is malformed")
  if manifest.get("family_split_manifest_sha256") != family_capture.sha256:
    raise ValueError("mate library and semantic replay use different family splits")
  if manifest.get("source_split_sha256") != train_capture.sha256:
    raise ValueError("mate library and semantic replay use different train bytes")
  root = Path(os.path.abspath(Path(archive_root)))
  archive_members_sha256: str | None = None
  if authority_schema == MATE_SEMANTIC_AUTHORITY_V2_SCHEMA:
    from .formal_archive_inputs import capture_restored_archive_inputs

    restored = capture_restored_archive_inputs(
        source_payload,
        restored_archive_root=root,
    )
    archive_captures = list(restored.captures)
    archive_ledger = [dict(row) for row in restored.ledger]
    archive_members_sha256 = restored.ledger_sha256
  elif authority_schema == MATE_SEMANTIC_AUTHORITY_SCHEMA:
    archive_captures, archive_ledger = _capture_archive_inputs(
        source_payload,
        archive_root=root,
    )
  else:
    raise ValueError("mate semantic replay authority schema is unsupported")
  captures.extend(archive_captures)

  if not pilot_only:
    from .formal_geometry_authority import (
        FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION,
    )

    if face_schema == FACE_MAP_ISOLATED_REPLAY_SCHEMA_VERSION:
      if authority_schema != MATE_SEMANTIC_AUTHORITY_V2_SCHEMA:
        raise ValueError("formal v4 face maps require mate semantic authority v2")
      if geometry_dataset_root is None:
        raise ValueError("formal v4 semantic replay requires the archive root")
      public_cases, private_supervision, face_map_lookup = (
          _formal_v4_replay_inputs(
              public_cases=tuple(dict(case) for case in train_payload),
              case_ids=tuple(family["train_case_ids"]),
              family_split_manifest_path=family_split_manifest_path,
              train_path=train_path,
              private_source_path=private_source_path,
              private_gold_path=private_gold_path,
              face_map_receipt_path=face_map_receipt_path,
              archive_root=geometry_dataset_root,
              restored_root=root,
              obj_extension_authority=obj_extension_authority,
          )
      )
    else:
      if obj_extension_authority is not None:
        raise ValueError(
            "obj_extension_authority is accepted only for formal v4 face maps"
        )
      # Preserve the established v3/training-loader contract.  Crucially, v4
      # never reaches this compatibility route.
      from .build_interface_dataset import (
          load_benchmark_v2_authenticated_training_inputs,
      )

      with tempfile.TemporaryDirectory(prefix="neurocad_mate_semantic_") as raw_dir:
        snapshot_root = Path(raw_dir)
        snapshot_paths = {
            "train": snapshot_root / "train.json",
            "family": snapshot_root / "family.json",
            "source": snapshot_root / "source.json",
            "gold": snapshot_root / "gold.json",
            "face": snapshot_root / "face.json",
        }
        snapshot_paths["train"].write_bytes(train_capture.raw_bytes)
        snapshot_paths["family"].write_bytes(family_capture.raw_bytes)
        snapshot_paths["source"].write_bytes(source_capture.raw_bytes)
        snapshot_paths["gold"].write_bytes(gold_capture.raw_bytes)
        snapshot_paths["face"].write_bytes(face_capture.raw_bytes)
        authenticated_inputs = load_benchmark_v2_authenticated_training_inputs(
            public_cases_path=snapshot_paths["train"],
            private_source_path=snapshot_paths["source"],
            private_gold_path=snapshot_paths["gold"],
            family_split_manifest_path=snapshot_paths["family"],
            face_map_receipt_path=snapshot_paths["face"],
            source_split="train",
        )
      if tuple(family["train_case_ids"]) != authenticated_inputs.case_ids:
        raise ValueError("mate semantic replay private case join differs")
      public_cases = authenticated_inputs.public_cases
      private_supervision = authenticated_inputs.private_supervision
      face_map_lookup = authenticated_inputs.face_map_lookup
  if authority_schema == MATE_SEMANTIC_AUTHORITY_V2_SCHEMA:
    from .mate_pose_miner import replay_benchmark_v2_mate_semantics_v2

    replay = replay_benchmark_v2_mate_semantics_v2(
        public_cases=public_cases,
        private_supervision=private_supervision,
        face_map_lookup=face_map_lookup,
        restored_archive_root=root,
        mining_config=manifest["mining_config"],
        family_split_manifest_sha256=family_capture.sha256,
        source_split_sha256=train_capture.sha256,
    )
  else:
    from .mate_pose_miner import replay_benchmark_v2_mate_semantics

    replay = replay_benchmark_v2_mate_semantics(
        public_cases=public_cases,
        private_supervision=private_supervision,
        face_map_lookup=face_map_lookup,
        archive_root=root,
        mining_config=manifest["mining_config"],
        family_split_manifest_sha256=family_capture.sha256,
        source_split_sha256=train_capture.sha256,
    )
  replay_rows = [dict(row) for row in replay.rows]
  if len(library_rows) != int(manifest["row_count"]):
    raise ValueError("mate library row count differs from its manifest")
  if library_rows != replay_rows:
    raise ValueError(
        "mate library rows differ from independent producer-semantic replay"
    )
  if dict(replay.coverage) != manifest.get("coverage"):
    raise ValueError("mate coverage differs from independent semantic replay")
  if dict(replay.coverage_ledger) != manifest.get("coverage_ledger"):
    raise ValueError("mate coverage ledger differs from independent semantic replay")
  row_commitments = [_row_commitment(row) for row in replay_rows]
  replay_inputs = {
              "library_sha256": library_capture.sha256,
              "manifest_sha256": manifest_capture.sha256,
              "family_split_manifest_sha256": family_capture.sha256,
              "train_split_sha256": train_capture.sha256,
              "private_source_sha256": source_capture.sha256,
              "private_gold_sha256": gold_capture.sha256,
              "face_map_receipt_sha256": face_capture.sha256,
              "archive_members": archive_ledger,
  }
  if archive_members_sha256 is not None:
    replay_inputs["archive_members_sha256"] = archive_members_sha256
  return (
      {
          "inputs": replay_inputs,
          "mining_config_sha256": str(manifest["mining_config_sha256"]),
          "coverage_ledger_sha256": canonical_sha256(
              dict(replay.coverage_ledger)
          ),
          "row_count": len(row_commitments),
          "row_commitments": row_commitments,
          "replay_rows_sha256": canonical_sha256(row_commitments),
      },
      captures,
  )


def _expected_receipt(
    replayed: Mapping[str, Any],
    *,
    implementation: Mapping[str, Any],
    pilot_only: bool = False,
    authority_schema: str = MATE_SEMANTIC_AUTHORITY_SCHEMA,
) -> dict[str, Any]:
  replay_contract = (
      _REPLAY_CONTRACT_V2
      if authority_schema == MATE_SEMANTIC_AUTHORITY_V2_SCHEMA
      else _REPLAY_CONTRACT
  )
  payload = {
      "schema_version": authority_schema,
      "formal": not pilot_only,
      "replay_contract": replay_contract,
      "implementation": dict(implementation),
      **dict(replayed),
  }
  if pilot_only:
    payload["pilot_only"] = True
    payload["development"] = True
  payload["receipt_payload_sha256"] = canonical_sha256(payload)
  return payload


def build_mate_semantic_authority_receipt(
    output_path: str | Path,
    *,
    library_path: str | Path,
    manifest_path: str | Path,
    family_split_manifest_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    face_map_receipt_path: str | Path,
    archive_root: str | Path,
    pilot_only: bool = False,
    geometry_dataset_root: str | Path | None = None,
) -> dict[str, Any]:
  """Build a hash-bound receipt from a fresh independent semantic replay."""

  _require_mate_promotion(
      pilot_only=pilot_only,
      schema_version=MATE_SEMANTIC_AUTHORITY_SCHEMA,
  )
  implementation, implementation_captures = _capture_implementation_binding(
      MATE_SEMANTIC_AUTHORITY_SCHEMA
  )
  replayed, captures = _replay_inputs(
      library_path=library_path,
      manifest_path=manifest_path,
      family_split_manifest_path=family_split_manifest_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      face_map_receipt_path=face_map_receipt_path,
      archive_root=archive_root,
      pilot_only=pilot_only,
      geometry_dataset_root=geometry_dataset_root,
      authority_schema=MATE_SEMANTIC_AUTHORITY_SCHEMA,
  )
  for captured in captures:
    _reverify_capture(captured, label="mate semantic replay input")
  for captured in implementation_captures:
    _reverify_capture(captured, label="mate semantic implementation input")
  receipt = _expected_receipt(
      replayed,
      implementation=implementation,
      pilot_only=pilot_only,
      authority_schema=MATE_SEMANTIC_AUTHORITY_SCHEMA,
  )
  destination = Path(output_path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  serialized = json.dumps(
      receipt,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  with tempfile.NamedTemporaryFile(
      dir=destination.parent,
      prefix=f".{destination.name}.",
      suffix=".tmp",
      delete=False,
  ) as stream:
    temporary = Path(stream.name)
    stream.write(serialized)
    stream.flush()
    os.fsync(stream.fileno())
  try:
    os.replace(temporary, destination)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
  return receipt


def build_mate_semantic_authority_v2_receipt(
    output_path: str | Path,
    *,
    library_path: str | Path,
    manifest_path: str | Path,
    family_split_manifest_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    face_map_receipt_path: str | Path,
    restored_archive_root: str | Path,
    pilot_only: bool = False,
    geometry_dataset_root: str | Path | None = None,
    authenticated_face_map: Any | None = None,
    obj_extension_authority: Any | None = None,
) -> dict[str, Any]:
  """Build the explicit parent-root, multi-archive semantic authority v2."""

  _require_mate_promotion(
      pilot_only=pilot_only,
      schema_version=MATE_SEMANTIC_AUTHORITY_V2_SCHEMA,
  )
  implementation, implementation_captures = _capture_implementation_binding(
      MATE_SEMANTIC_AUTHORITY_V2_SCHEMA
  )
  replayed, captures = _replay_inputs(
      library_path=library_path,
      manifest_path=manifest_path,
      family_split_manifest_path=family_split_manifest_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      face_map_receipt_path=face_map_receipt_path,
      archive_root=restored_archive_root,
      pilot_only=pilot_only,
      geometry_dataset_root=geometry_dataset_root,
      authority_schema=MATE_SEMANTIC_AUTHORITY_V2_SCHEMA,
      authenticated_face_map=authenticated_face_map,
      obj_extension_authority=obj_extension_authority,
  )
  for captured in captures:
    _reverify_capture(captured, label="mate semantic v2 replay input")
  for captured in implementation_captures:
    _reverify_capture(captured, label="mate semantic v2 implementation input")
  receipt = _expected_receipt(
      replayed,
      implementation=implementation,
      pilot_only=pilot_only,
      authority_schema=MATE_SEMANTIC_AUTHORITY_V2_SCHEMA,
  )
  destination = Path(output_path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  serialized = json.dumps(
      receipt,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  with tempfile.NamedTemporaryFile(
      dir=destination.parent,
      prefix=f".{destination.name}.",
      suffix=".tmp",
      delete=False,
  ) as stream:
    temporary = Path(stream.name)
    stream.write(serialized)
    stream.flush()
    os.fsync(stream.fileno())
  try:
    os.replace(temporary, destination)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
  return receipt


def _capture_memory_binding(captured: Any) -> dict[str, Any]:
  return {
      "path": str(captured.resolved_path),
      "sha256": captured.sha256,
      "byte_count": captured.byte_count,
      "device": captured.device,
      "inode": captured.inode,
      "link_count": captured.link_count,
  }


def _mate_memory_state_payload(
    *,
    schema_version: str,
    receipt_sha256: str,
    library_sha256: str,
    manifest_sha256: str,
    archive_root: Path,
    row_commitments: Sequence[Mapping[str, Any]],
    receipt_capture: Any,
    captures: Sequence[Any],
) -> dict[str, Any]:
  return {
      "kind": "AuthenticatedMateSemanticAuthority",
      "schema_version": str(schema_version),
      "receipt_sha256": str(receipt_sha256),
      "library_sha256": str(library_sha256),
      "manifest_sha256": str(manifest_sha256),
      "archive_root": str(Path(archive_root)),
      "row_commitments": [dict(commitment) for commitment in row_commitments],
      "receipt_capture": _capture_memory_binding(receipt_capture),
      "captures": [_capture_memory_binding(captured) for captured in captures],
  }


class AuthenticatedMateSemanticAuthority(_ImmutableAuthenticatedCapability):
  """Non-serializable proof handle created only by the replay verifier."""

  __slots__ = (
      "schema_version",
      "receipt_sha256",
      "library_sha256",
      "manifest_sha256",
      "archive_root",
      "row_commitments",
      "_receipt_capture",
      "_captures",
  )

  def __init__(
      self,
      *,
      schema_version: str,
      receipt_sha256: str,
      library_sha256: str,
      manifest_sha256: str,
      archive_root: Path,
      row_commitments: Sequence[Mapping[str, Any]],
      receipt_capture: Any,
      captures: Sequence[Any],
      _issuance_context: object,
  ) -> None:
    try:
      _require_verified_issuance_context(
          _issuance_context,
          _MATE_CAPABILITY_KIND,
      )
    except TypeError as error:
      raise TypeError(
          "AuthenticatedMateSemanticAuthority must be created by the verifier "
          "with a one-shot verified issuance context"
      ) from error
    self.schema_version = str(schema_version)
    self.receipt_sha256 = str(receipt_sha256)
    self.library_sha256 = str(library_sha256)
    self.manifest_sha256 = str(manifest_sha256)
    self.archive_root = Path(archive_root)
    self.row_commitments = _recursive_freeze(tuple(row_commitments))
    self._receipt_capture = receipt_capture
    self._captures = tuple(captures)
    try:
      self._register_memory_state(
          self._memory_state_payload(),
          issuance_context=_issuance_context,
          capability_kind=_MATE_CAPABILITY_KIND,
      )
    except (TypeError, ValueError) as error:
      raise TypeError(
          "AuthenticatedMateSemanticAuthority must be created by the verifier "
          "with a one-shot verified issuance context"
      ) from error
    self._seal()

  def __reduce_ex__(self, _protocol):
    raise TypeError("authenticated mate semantic authority is not serializable")

  def __repr__(self) -> str:
    return (
        "<AuthenticatedMateSemanticAuthority "
        f"schema={self.schema_version!r} rows={len(self.row_commitments)}>"
    )

  @property
  def restored_archive_root(self) -> Path:
    """The parent restored-volume root bound by authority v2."""

    if self.schema_version != MATE_SEMANTIC_AUTHORITY_V2_SCHEMA:
      raise AttributeError("v1 authority has only a single archive_root")
    return self.archive_root

  def _memory_state_payload(self) -> dict[str, Any]:
    return _mate_memory_state_payload(
        schema_version=self.schema_version,
        receipt_sha256=self.receipt_sha256,
        library_sha256=self.library_sha256,
        manifest_sha256=self.manifest_sha256,
        archive_root=self.archive_root,
        row_commitments=self.row_commitments,
        receipt_capture=self._receipt_capture,
        captures=self._captures,
    )

  def revalidate(self) -> None:
    self._require_memory_state(self._memory_state_payload())
    _reverify_capture(
        self._receipt_capture,
        label="mate semantic authority receipt",
    )
    for captured in self._captures:
      _reverify_capture(captured, label="mate semantic authority bound input")


def verify_formal_mate_semantic_authority(
    authority_receipt_path: str | Path,
    *,
    library_path: str | Path,
    manifest_path: str | Path,
    family_split_manifest_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    face_map_receipt_path: str | Path,
    archive_root: str | Path,
    pilot_only: bool = False,
    geometry_dataset_root: str | Path | None = None,
) -> _VerifiedHandleConstruction:
  """Verify and replay one formal mate semantic authority receipt."""

  _require_mate_promotion(
      pilot_only=pilot_only,
      schema_version=MATE_SEMANTIC_AUTHORITY_SCHEMA,
  )
  receipt_capture, receipt = _captured_json(
      authority_receipt_path,
      label="mate semantic authority receipt",
  )
  schema = receipt.get("schema_version") if isinstance(receipt, Mapping) else None
  if (
      schema != MATE_SEMANTIC_AUTHORITY_SCHEMA
      or (
          receipt.get("formal") is not (not pilot_only)
          or receipt.get("pilot_only") is not (True if pilot_only else None)
          or receipt.get("development") is not (True if pilot_only else None)
      )
  ):
    raise ValueError(
        "formal mate loading requires an authenticated, allowlisted mate semantic "
        "authority schema"
    )
  implementation, implementation_captures = _capture_implementation_binding(
      MATE_SEMANTIC_AUTHORITY_SCHEMA
  )
  replayed, captures = _replay_inputs(
      library_path=library_path,
      manifest_path=manifest_path,
      family_split_manifest_path=family_split_manifest_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      face_map_receipt_path=face_map_receipt_path,
      archive_root=archive_root,
      pilot_only=pilot_only,
      geometry_dataset_root=geometry_dataset_root,
      authority_schema=MATE_SEMANTIC_AUTHORITY_SCHEMA,
  )
  expected = _expected_receipt(
      replayed,
      implementation=implementation,
      pilot_only=pilot_only,
      authority_schema=MATE_SEMANTIC_AUTHORITY_SCHEMA,
  )
  if receipt != expected:
    raise ValueError(
        "mate semantic authority receipt differs from independent replay"
    )
  for captured in captures:
    _reverify_capture(captured, label="mate semantic replay input")
  for captured in implementation_captures:
    _reverify_capture(captured, label="mate semantic implementation input")
  _reverify_capture(
      receipt_capture,
      label="mate semantic authority receipt",
  )
  inputs = replayed["inputs"]
  constructor_kwargs = {
      "schema_version": MATE_SEMANTIC_AUTHORITY_SCHEMA,
      "receipt_sha256": receipt_capture.sha256,
      "library_sha256": str(inputs["library_sha256"]),
      "manifest_sha256": str(inputs["manifest_sha256"]),
      "archive_root": Path(os.path.abspath(Path(archive_root))).resolve(strict=True),
      "row_commitments": replayed["row_commitments"],
      "receipt_capture": receipt_capture,
      "captures": tuple(
          _immutable_byte_capture(captured)
          for captured in (*captures, *implementation_captures)
      ),
  }
  return _VerifiedHandleConstruction(
      kwargs=MappingProxyType(constructor_kwargs),
      memory_state_commitment=canonical_sha256(
          _mate_memory_state_payload(**constructor_kwargs)
      ),
  )


def verify_formal_mate_semantic_authority_v2(
    authority_receipt_path: str | Path,
    *,
    library_path: str | Path,
    manifest_path: str | Path,
    family_split_manifest_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    face_map_receipt_path: str | Path,
    restored_archive_root: str | Path,
    pilot_only: bool = False,
    geometry_dataset_root: str | Path | None = None,
    authenticated_face_map: Any | None = None,
    obj_extension_authority: Any | None = None,
) -> _VerifiedHandleConstruction:
  """Verify v2 using the restored-archive parent root, never one volume."""

  _require_mate_promotion(
      pilot_only=pilot_only,
      schema_version=MATE_SEMANTIC_AUTHORITY_V2_SCHEMA,
  )
  receipt_capture, receipt = _captured_json(
      authority_receipt_path,
      label="mate semantic authority v2 receipt",
  )
  schema = receipt.get("schema_version") if isinstance(receipt, Mapping) else None
  if (
      schema != MATE_SEMANTIC_AUTHORITY_V2_SCHEMA
      or receipt.get("formal") is not (not pilot_only)
      or receipt.get("pilot_only") is not (True if pilot_only else None)
      or receipt.get("development") is not (True if pilot_only else None)
  ):
    raise ValueError(
        "formal mate loading requires an authenticated, allowlisted mate semantic "
        "authority v2 schema"
    )
  implementation, implementation_captures = _capture_implementation_binding(
      MATE_SEMANTIC_AUTHORITY_V2_SCHEMA
  )
  replayed, captures = _replay_inputs(
      library_path=library_path,
      manifest_path=manifest_path,
      family_split_manifest_path=family_split_manifest_path,
      private_source_path=private_source_path,
      private_gold_path=private_gold_path,
      face_map_receipt_path=face_map_receipt_path,
      archive_root=restored_archive_root,
      pilot_only=pilot_only,
      geometry_dataset_root=geometry_dataset_root,
      authority_schema=MATE_SEMANTIC_AUTHORITY_V2_SCHEMA,
      authenticated_face_map=authenticated_face_map,
      obj_extension_authority=obj_extension_authority,
  )
  expected = _expected_receipt(
      replayed,
      implementation=implementation,
      pilot_only=pilot_only,
      authority_schema=MATE_SEMANTIC_AUTHORITY_V2_SCHEMA,
  )
  if receipt != expected:
    raise ValueError(
        "mate semantic authority v2 receipt differs from independent replay"
    )
  for captured in captures:
    _reverify_capture(captured, label="mate semantic v2 replay input")
  for captured in implementation_captures:
    _reverify_capture(captured, label="mate semantic v2 implementation input")
  _reverify_capture(receipt_capture, label="mate semantic authority v2 receipt")
  inputs = replayed["inputs"]
  constructor_kwargs = {
      "schema_version": MATE_SEMANTIC_AUTHORITY_V2_SCHEMA,
      "receipt_sha256": receipt_capture.sha256,
      "library_sha256": str(inputs["library_sha256"]),
      "manifest_sha256": str(inputs["manifest_sha256"]),
      "archive_root": Path(
          os.path.abspath(Path(restored_archive_root))
      ).resolve(strict=True),
      "row_commitments": replayed["row_commitments"],
      "receipt_capture": receipt_capture,
      "captures": tuple(
          _immutable_byte_capture(captured)
          for captured in (*captures, *implementation_captures)
      ),
  }
  return _VerifiedHandleConstruction(
      kwargs=MappingProxyType(constructor_kwargs),
      memory_state_commitment=canonical_sha256(
          _mate_memory_state_payload(**constructor_kwargs)
      ),
  )


verify_formal_mate_semantic_authority = _wrap_verified_verifier(
    verify_formal_mate_semantic_authority,
    constructor=AuthenticatedMateSemanticAuthority,
)
verify_formal_mate_semantic_authority_v2 = _wrap_verified_verifier(
    verify_formal_mate_semantic_authority_v2,
    constructor=AuthenticatedMateSemanticAuthority,
)
del _wrap_verified_verifier


__all__ = [
    "AuthenticatedMateSemanticAuthority",
    "build_mate_semantic_authority_receipt",
    "build_mate_semantic_authority_v2_receipt",
    "FORMAL_MATE_SEMANTIC_AUTHORITY_SCHEMA_ALLOWLIST",
    "MATE_SEMANTIC_AUTHORITY_SCHEMA",
    "MATE_SEMANTIC_AUTHORITY_V2_SCHEMA",
    "verify_formal_mate_semantic_authority",
    "verify_formal_mate_semantic_authority_v2",
]
