"""Lightweight retrieval over mined mate-program libraries."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat as stat_module
import tempfile
from typing import Any, Iterable, Optional

import numpy as np

from .benchmark_v2_constants import PROTOCOL_VERSION
from .benchmark_v2_training_provenance import (
    validate_face_map_binding,
    validate_private_supervision_binding,
)
from .mate_programs import MateProgram
from .domain_types import Socket
from .interface_pair_features import (
    BENCHMARK_V2_PAIR_FEATURE_NAMES,
    pair_features_from_rows,
)
from .paths import resolve_path
from .mate_semantic_authority import (
    AuthenticatedMateSemanticAuthority,
    FORMAL_MATE_SEMANTIC_AUTHORITY_SCHEMA_ALLOWLIST,
    verify_formal_mate_semantic_authority,
    verify_formal_mate_semantic_authority_v2,
)


MATE_LIBRARY_MANIFEST_SCHEMA = "mate_program_library_manifest.v3"
MATE_MINING_CONFIG_SCHEMA = "mate_program_mining_config.v4"
MATE_COVERAGE_LEDGER_SCHEMA = "mate_program_coverage_ledger.v1"
MATE_MINER_CODE_BUNDLE_SCHEMA = "mate_program_miner_code_bundle.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROGRAM_ID = re.compile(r"^[0-9a-f]{16}$")
_OPAQUE_CASE_ID = re.compile(r"^opaque_case_[0-9a-f]{20}$")
_OPAQUE_PART_ID = re.compile(r"^opaque_part_[0-9]{3,}$")
_OPAQUE_BODY_ID = re.compile(r"^opaque_body_[0-9]{3,}$")

_FORMAL_PROGRAM_ROW_KEYS = frozenset(
    {
        "model_input_protocol",
        "program_id",
        "case_id",
        "assembly_dir",
        "part_a",
        "part_b",
        "body_uuid_a",
        "body_uuid_b",
        "interface_a",
        "interface_b",
        "relation_hint",
        "contact_type",
        "residual_rotation",
        "residual_translation",
        "source_contact",
        "features",
        "metadata",
    }
)
_FORMAL_ENDPOINT_KEYS = frozenset(
    {
        "model_input_protocol",
        "benchmark_v2_model_view",
        "benchmark_v2_model_view_sha256",
        "score",
    }
)
_FORMAL_SOURCE_CONTACT_KEYS = frozenset(
    {
        "contact_index",
        "direction",
        "face_index_a",
        "face_index_b",
        "surface_type_a",
        "surface_type_b",
    }
)
_FORMAL_METADATA_KEYS = frozenset(
    {
        "source",
        "leakage_scope",
        "part_frame",
        "bidirectional_direction",
        "model_input_protocol",
        "source_split",
        "source_split_sha256",
        "mining_config_sha256",
        "family_split_manifest_sha256",
        "source_case_set_sha256",
        "source_case_token_sha256",
    }
)
_FORMAL_RELATION_HINTS = frozenset({"insert", "support", "link"})
_FORMAL_CONTACT_TYPES = frozenset(
    {
        "seat_plane",
        "insert_axis",
        "shaft_in_bore",
        "screw_in_hole",
        "pin_in_slot",
        "boss_in_slot",
        "threaded_interference",
        "generic_contact",
    }
)
_FORMAL_DIRECTIONS = frozenset(
    {"entity_one_to_entity_two", "entity_two_to_entity_one"}
)
_FORMAL_SOURCE_SURFACES = frozenset(
    {
        "PlaneSurfaceType",
        "CylinderSurfaceType",
        "ConeSurfaceType",
        "SphereSurfaceType",
        "TorusSurfaceType",
        "NurbsSurfaceType",
        "EllipticalCylinderSurfaceType",
        "EllipticalConeSurfaceType",
    }
)

_MATE_MINER_CODE_FILES = (
    "batch_infer_assemble.py",
    "benchmark_v2_constants.py",
    "benchmark_v2_model_view.py",
    "benchmark_v2_protocol.py",
    "benchmark_v2_private_supervision.py",
    "benchmark_v2_training_provenance.py",
    "build_interface_dataset.py",
    "build_interface_pair_dataset.py",
    "cadquery_backend.py",
    "domain_types.py",
    "frame_randomization.py",
    "fusion_face_mapper.py",
    "interface_features.py",
    "interface_grounder.py",
    "interface_pair_features.py",
    "interface_pair_scorer.py",
    "interface_scorer.py",
    "math3d.py",
    "mate_semantic_authority.py",
    "mate_pose_miner.py",
    "mate_pose_retriever.py",
    "mate_programs.py",
    "offline.py",
    "paths.py",
)


def _stat_identity(stat_result: os.stat_result) -> tuple[int, ...]:
  """Return fields that identify both a file object and its observed version."""

  return (
      int(stat_result.st_dev),
      int(stat_result.st_ino),
      int(stat_result.st_mode),
      int(stat_result.st_size),
      int(stat_result.st_mtime_ns),
      int(getattr(stat_result, "st_birthtime_ns", 0)),
      int(stat_result.st_nlink),
  )


def _path_entry_is_reparse(path: Path) -> bool:
  try:
    value = path.lstat()
  except OSError:
    return False
  attributes = int(getattr(value, "st_file_attributes", 0))
  is_junction = getattr(path, "is_junction", None)
  return bool(
      stat_module.S_ISLNK(value.st_mode)
      or attributes & 0x400  # Windows FILE_ATTRIBUTE_REPARSE_POINT
      or (callable(is_junction) and is_junction())
  )


@dataclass(frozen=True, slots=True)
class _CapturedFile:
  """One immutable verifier input captured from a single open file handle."""

  path: Path
  resolved_path: Path
  identity: tuple[int, ...]
  path_entry_identity: tuple[int, ...]
  payload: bytes
  sha256: str
  label: str

  @classmethod
  def capture(cls, path: str | Path, *, label: str) -> "_CapturedFile":
    requested = Path(os.path.abspath(Path(path)))
    try:
      if any(
          _path_entry_is_reparse(entry)
          for entry in (requested, *requested.parents)
      ):
        raise ValueError(f"{label} path traverses a symlink or reparse point")
      path_entry_before = _stat_identity(requested.lstat())
      if path_entry_before[-1] != 1:
        raise ValueError(f"{label} must not be a hard-linked file")
      resolved_before = requested.resolve(strict=True)
      if os.path.normcase(str(resolved_before)) != os.path.normcase(
          os.path.abspath(requested)
      ):
        raise ValueError(f"{label} path is not a plain direct path")
      with requested.open("rb") as stream:
        identity_before = _stat_identity(os.fstat(stream.fileno()))
        if identity_before[-1] != 1:
          raise ValueError(f"{label} must not be a hard-linked file")
        payload = stream.read()
        identity_after = _stat_identity(os.fstat(stream.fileno()))
      resolved_after = requested.resolve(strict=True)
      path_identity = _stat_identity(requested.stat())
      path_entry_after = _stat_identity(requested.lstat())
    except ValueError:
      raise
    except OSError as error:
      raise ValueError(f"{label} is missing/unreadable") from error
    if (
        identity_before != identity_after
        or identity_after != path_identity
        or path_entry_before != path_entry_after
        or resolved_before != resolved_after
        or len(payload) != identity_after[3]
    ):
      raise ValueError(f"{label} changed while it was captured")
    return cls(
        path=requested,
        resolved_path=resolved_before,
        identity=identity_after,
        path_entry_identity=path_entry_after,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        label=label,
    )

  def reverify(self) -> None:
    """Reject replacement or mutation of the path after its captured use."""

    current = _CapturedFile.capture(self.path, label=self.label)
    if (
        current.resolved_path != self.resolved_path
        or current.identity != self.identity
        or current.path_entry_identity != self.path_entry_identity
        or current.payload != self.payload
        or current.sha256 != self.sha256
    ):
      raise ValueError(f"{self.label} changed during verification")


class _FormalVerificationSession:
  """Hold every external verifier input until one final revalidation barrier."""

  def __init__(self) -> None:
    self._captures: dict[Path, _CapturedFile] = {}

  def capture(self, path: str | Path, *, label: str) -> _CapturedFile:
    absolute = Path(os.path.abspath(Path(path)))
    existing = self._captures.get(absolute)
    if existing is not None:
      return existing
    captured = _CapturedFile.capture(absolute, label=label)
    self._captures[absolute] = captured
    return captured

  def revalidate_all(self) -> None:
    for path in sorted(self._captures, key=lambda value: os.path.normcase(str(value))):
      self._captures[path].reverify()


def _json_from_captured_file(captured: _CapturedFile) -> Any:
  try:
    return json.loads(
        captured.payload.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
    )
  except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise ValueError(f"{captured.label} is invalid JSON") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate JSON key: {key}")
    result[key] = value
  return result


@dataclass
class MateProgramRetrieval:
  program: MateProgram
  score: float
  reasons: list[str]
  proposal_trace: Optional[dict[str, Any]] = None

  def to_dict(self) -> dict[str, Any]:
    payload = {
        "program_id": self.program.program_id,
        "score": round(float(self.score), 6),
        "reasons": list(self.reasons),
        "relation_hint": self.program.relation_hint,
        "contact_type": self.program.contact_type,
        "parent_role": self.program.parent_role,
        "child_role": self.program.child_role,
        "residual_translation": list(self.program.residual_translation),
        "residual_rotation": list(self.program.residual_rotation),
        "source": {
            "case_id": self.program.case_id,
            "assembly_dir": self.program.assembly_dir,
            "part_a": self.program.part_a,
            "part_b": self.program.part_b,
        },
    }
    if self.proposal_trace is not None:
      payload["proposal_trace"] = dict(self.proposal_trace)
    return payload


class MateProgramRetriever:
  """A dependency-light retriever for directed interface mate programs."""

  def __init__(
      self,
      programs: Iterable[MateProgram],
      *,
      use_contact_priors: bool = False,
      library_manifest: Optional[dict[str, Any]] = None,
  ):
    self.programs = list(programs)
    self.use_contact_priors = bool(use_contact_priors)
    self.library_manifest = (
        None if library_manifest is None else dict(library_manifest)
    )
    self._seating_depths_exact, self._seating_depths_family = (
        _index_seating_depths(self.programs)
    )

  @classmethod
  def load(
      cls,
      path: str | Path,
      *,
      use_contact_priors: bool = False,
      input_protocol: str = "legacy",
      formal: bool = False,
      manifest_path: str | Path | None = None,
      family_split_manifest_path: str | Path | None = None,
      private_source_path: str | Path | None = None,
      private_gold_path: str | Path | None = None,
      face_map_receipt_path: str | Path | None = None,
      contact_census_receipt_path: str | Path | None = None,
      semantic_authority_receipt_path: str | Path | None = None,
      archive_root: str | Path | None = None,
      restored_archive_root: str | Path | None = None,
      _semantic_authority_v2: bool = False,
  ) -> "MateProgramRetriever":
    protocol = str(input_protocol or "legacy").strip().lower()
    if protocol not in {"legacy", "benchmark_v2"}:
      raise ValueError("mate library protocol must be legacy or benchmark_v2")
    if protocol != "benchmark_v2" and not formal:
      return cls(load_mate_programs(path), use_contact_priors=use_contact_priors)
    if protocol != "benchmark_v2":
      raise ValueError("formal mate libraries require benchmark_v2 protocol")
    library_path = Path(path)
    sidecar = (
        Path(manifest_path)
        if manifest_path is not None
        else library_path.with_suffix(library_path.suffix + ".manifest.json")
    )
    session = _FormalVerificationSession()
    library_capture = session.capture(
        library_path,
        label="formal mate library",
    )
    manifest_capture = session.capture(
        sidecar,
        label="formal mate library manifest",
    )
    manifest = _parse_v2_library_manifest(
        manifest_capture,
        session=session,
    )
    if family_split_manifest_path is None:
      raise ValueError("formal mate library requires a family split manifest")
    family_verification = _validate_family_split_manifest_binding(
        manifest,
        Path(family_split_manifest_path),
        session=session,
    )
    authenticated_receipts = _authenticate_formal_mate_receipts(
        manifest=manifest,
        family_verification=family_verification,
        family_manifest_path=Path(family_split_manifest_path),
        private_source_path=private_source_path,
        private_gold_path=private_gold_path,
        face_map_receipt_path=face_map_receipt_path,
        contact_census_receipt_path=contact_census_receipt_path,
        session=session,
    )
    programs = _load_mate_programs_payload(
        library_capture.payload,
        strict=True,
        expected_manifest=manifest,
    )
    if manifest.get("library_sha256") != library_capture.sha256:
      raise ValueError("formal mate library content hash mismatch")
    if int(manifest.get("row_count", -1)) != len(programs):
      raise ValueError("formal mate library row count mismatch")
    _validate_loaded_program_coverage(
        programs,
        manifest=manifest,
        family_verification=family_verification,
    )
    session.revalidate_all()
    _require_formal_mate_receipt_readiness(authenticated_receipts)
    semantic_common = {
        "library_path": library_path,
        "manifest_path": sidecar,
        "family_split_manifest_path": Path(family_split_manifest_path),
        "private_source_path": Path(private_source_path),
        "private_gold_path": Path(private_gold_path),
        "face_map_receipt_path": Path(face_map_receipt_path),
    }
    if _semantic_authority_v2:
      if (
          semantic_authority_receipt_path is None
          or restored_archive_root is None
          or archive_root is not None
      ):
        raise ValueError(
            "formal mate authority v2 requires its receipt and the restored "
            "archive parent root; archive_root must not be supplied"
        )
      semantic_authority = verify_formal_mate_semantic_authority_v2(
          semantic_authority_receipt_path,
          restored_archive_root=Path(restored_archive_root),
          **semantic_common,
      )
    else:
      if semantic_authority_receipt_path is None or archive_root is None:
        raise ValueError(
            "formal mate loading requires a producer-semantic replay authority "
            "receipt path and archive_root"
        )
      if restored_archive_root is not None:
        raise ValueError(
            "restored_archive_root is accepted only by the explicit v2 loader"
        )
      semantic_authority = verify_formal_mate_semantic_authority(
          semantic_authority_receipt_path,
          archive_root=Path(archive_root),
          **semantic_common,
      )
    _require_formal_mate_semantic_authority(
        manifest=manifest,
        authenticated_receipts=authenticated_receipts,
        authority=semantic_authority,
        library_sha256=library_capture.sha256,
        manifest_sha256=manifest_capture.sha256,
    )
    session.revalidate_all()
    semantic_authority.revalidate()
    return cls(
        programs,
        use_contact_priors=False,
        library_manifest=manifest,
    )

  @classmethod
  def load_formal_multi_archive_v2(
      cls,
      path: str | Path,
      *,
      manifest_path: str | Path | None = None,
      family_split_manifest_path: str | Path,
      private_source_path: str | Path,
      private_gold_path: str | Path,
      face_map_receipt_path: str | Path,
      semantic_authority_receipt_path: str | Path,
      restored_archive_root: str | Path,
      contact_census_receipt_path: str | Path | None = None,
  ) -> "MateProgramRetriever":
    """Load a formal library through the explicit multi-archive v2 gate."""

    return cls.load(
        path,
        input_protocol="benchmark_v2",
        formal=True,
        manifest_path=manifest_path,
        family_split_manifest_path=family_split_manifest_path,
        private_source_path=private_source_path,
        private_gold_path=private_gold_path,
        face_map_receipt_path=face_map_receipt_path,
        contact_census_receipt_path=contact_census_receipt_path,
        semantic_authority_receipt_path=semantic_authority_receipt_path,
        restored_archive_root=restored_archive_root,
        _semantic_authority_v2=True,
    )

  def query(
      self,
      *,
      parent_role: str,
      child_role: str,
      relation_hint: str = "",
      contact_type: str = "",
      parent_surface: str = "",
      child_surface: str = "",
      parent_radius: Optional[float] = None,
      child_radius: Optional[float] = None,
      top_k: int = 8,
  ) -> list[MateProgramRetrieval]:
    results: list[MateProgramRetrieval] = []
    for program in self.programs:
      score, reasons = _score_program(
          program=program,
          parent_role=parent_role,
          child_role=child_role,
          relation_hint=relation_hint,
          contact_type=contact_type,
          parent_surface=parent_surface,
          child_surface=child_surface,
          parent_radius=parent_radius,
          child_radius=child_radius,
          use_contact_priors=self.use_contact_priors,
      )
      if score <= -8.0:
        continue
      results.append(MateProgramRetrieval(program=program, score=score, reasons=reasons))
    results.sort(
        key=lambda item: (
            -float(item.score),
            item.program.contact_type,
            item.program.program_id,
        )
    )
    return results[: max(1, int(top_k))]

  def query_sockets(
      self,
      *,
      parent_socket: Socket,
      child_socket: Socket,
      relation_hint: str = "",
      contact_type: str = "",
      top_k: int = 8,
  ) -> list[MateProgramRetrieval]:
    return self.query(
        parent_role=_socket_role(parent_socket),
        child_role=_socket_role(child_socket),
        relation_hint=relation_hint,
        contact_type=contact_type,
        parent_surface=_socket_surface(parent_socket),
        child_surface=_socket_surface(child_socket),
        parent_radius=parent_socket.radius,
        child_radius=child_socket.radius,
        top_k=top_k,
    )

  def stats(self) -> dict[str, Any]:
    by_contact: dict[str, int] = {}
    by_role_pair: dict[str, int] = {}
    for program in self.programs:
      by_contact[program.contact_type] = by_contact.get(program.contact_type, 0) + 1
      key = f"{program.parent_role}->{program.child_role}"
      by_role_pair[key] = by_role_pair.get(key, 0) + 1
    return {
        "program_count": len(self.programs),
        "seating_depth_exact_key_count": len(self._seating_depths_exact),
        "seating_depth_family_key_count": len(self._seating_depths_family),
        "contact_type_counts": dict(sorted(by_contact.items())),
        "role_pair_counts": dict(
            sorted(by_role_pair.items(), key=lambda item: (-item[1], item[0]))[:50]
        ),
    }

  def seating_offset_deltas(
      self,
      program: MateProgram,
      *,
      limit: int = 3,
      max_abs: float = 250.0,
  ) -> list[float]:
    """Return finite axial residual-depth replacements as world offset deltas.

    ``instantiate_child_transform`` has already applied ``program``'s residual
    translation.  The values returned here are therefore *deltas* that replace
    the current program's residual-z with a few library-derived residual-z
    representatives from the same contact/role family.
    """

    limit = max(1, int(limit))
    max_abs = max(0.0, float(max_abs))
    current_depth = _residual_z(program)
    if current_depth is None:
      return [0.0]

    exact_key = _seating_exact_key(program)
    family_key = _seating_family_key(program)
    targets = _seating_depth_targets(
        self._seating_depths_exact.get(exact_key, ()),
        max_targets=limit,
    )
    if len(targets) < limit:
      targets = _merge_depth_targets(
          targets,
          _seating_depth_targets(
              self._seating_depths_family.get(family_key, ()),
              max_targets=limit,
          ),
          limit,
      )

    deltas = [0.0]
    for target_depth in targets:
      delta = float(target_depth) - float(current_depth)
      if abs(delta) <= 1e-6:
        continue
      if abs(delta) > max_abs:
        continue
      if _has_near_value(deltas, delta):
        continue
      deltas.append(float(delta))
      if len(deltas) >= limit:
        break
    return deltas[:limit]


def _authenticate_formal_mate_receipts(
    *,
    manifest: dict[str, Any],
    family_verification: dict[str, Any],
    family_manifest_path: Path,
    private_source_path: str | Path | None,
    private_gold_path: str | Path | None,
    face_map_receipt_path: str | Path | None,
    contact_census_receipt_path: str | Path | None,
    session: _FormalVerificationSession,
) -> Any:
  required_paths = {
      "private source split": private_source_path,
      "private evaluation gold split": private_gold_path,
      "Fusion STEP face-map receipt": face_map_receipt_path,
  }
  missing = [label for label, path in required_paths.items() if path is None]
  if missing:
    raise ValueError(
        "formal mate library requires path-authenticated receipts: "
        + ", ".join(missing)
    )
  artifact_paths = family_verification.get("artifact_paths")
  if not isinstance(artifact_paths, dict) or not isinstance(
      artifact_paths.get("train"), str
  ):
    raise ValueError("formal family verification lacks its train artifact path")
  train_capture = session.capture(
      artifact_paths["train"],
      label="formal family train artifact",
  )
  family_capture = session.capture(
      family_manifest_path,
      label="formal family split manifest",
  )
  private_source_capture = session.capture(
      Path(private_source_path),
      label="private source split",
  )
  private_gold_capture = session.capture(
      Path(private_gold_path),
      label="private evaluation gold split",
  )
  face_map_capture = session.capture(
      Path(face_map_receipt_path),
      label="Fusion STEP face-map receipt",
  )
  for captured in (
      private_source_capture,
      private_gold_capture,
      face_map_capture,
  ):
    _json_from_captured_file(captured)

  census_capture: _CapturedFile | None = None
  if contact_census_receipt_path is not None:
    census_capture = session.capture(
        Path(contact_census_receipt_path),
        label="private contact census receipt",
    )
    _json_from_captured_file(census_capture)

  mining_config = manifest.get("mining_config")
  if not isinstance(mining_config, dict):
    raise ValueError("formal mate library lacks its mining config")
  private_binding = mining_config.get("private_supervision_binding")
  if not isinstance(private_binding, dict):
    raise ValueError("formal mate library lacks private supervision binding")
  binary_claim = bool(
      private_binding.get("gold_label_semantics")
      != "source_positive_assertions_only"
      or private_binding.get("gold_interface_certificate_ready") is not False
      or private_binding.get("certified_negative_complete") is not False
  )
  if binary_claim and census_capture is None:
    raise ValueError(
        "formal binary mate supervision requires an authenticated, allowlisted "
        "contact census receipt; no formal census schema is implemented"
    )

  # The shared authenticated loader currently accepts paths. Feed it private
  # snapshots made only after duplicate-key parsing of every captured receipt.
  from .build_interface_dataset import (
      load_benchmark_v2_authenticated_training_inputs,
  )

  with tempfile.TemporaryDirectory(prefix="neurocad_mate_receipts_") as raw_dir:
    snapshot_root = Path(raw_dir)
    snapshots = {
        "train": snapshot_root / "train.json",
        "family": snapshot_root / "split_manifest.json",
        "private_source": snapshot_root / "private_source.json",
        "private_gold": snapshot_root / "private_gold.json",
        "face_map": snapshot_root / "face_map.json",
    }
    payloads = {
        "train": train_capture.payload,
        "family": family_capture.payload,
        "private_source": private_source_capture.payload,
        "private_gold": private_gold_capture.payload,
        "face_map": face_map_capture.payload,
    }
    for key, snapshot in snapshots.items():
      snapshot.write_bytes(payloads[key])
    census_snapshot: Path | None = None
    if census_capture is not None:
      census_snapshot = snapshot_root / "contact_census.json"
      census_snapshot.write_bytes(census_capture.payload)
    authenticated = load_benchmark_v2_authenticated_training_inputs(
        public_cases_path=snapshots["train"],
        private_source_path=snapshots["private_source"],
        private_gold_path=snapshots["private_gold"],
        family_split_manifest_path=snapshots["family"],
        face_map_receipt_path=snapshots["face_map"],
        source_split="train",
        contact_census_receipt_path=census_snapshot,
    )

  expected_case_ids = tuple(
      str(value) for value in family_verification.get("train_case_ids", [])
  )
  receipts = authenticated.receipts
  if (
      authenticated.case_ids != expected_case_ids
      or receipts.source_split != "train"
      or receipts.case_ids != expected_case_ids
      or receipts.family_split_manifest_sha256 != family_capture.sha256
      or receipts.public_cases_file_sha256 != train_capture.sha256
  ):
    raise ValueError(
        "authenticated mate receipts do not close over the frozen train split"
    )
  if set(receipts.fully_mapped_case_ids) != set(expected_case_ids):
    raise ValueError(
        "formal mate face-map receipt is not fully mapped for every train case"
    )
  if dict(receipts.private_supervision_binding) != private_binding:
    raise ValueError(
        "mate manifest private supervision binding differs from authenticated receipts"
    )
  face_binding = mining_config.get("fusion_step_face_map_binding")
  if not isinstance(face_binding, dict) or dict(receipts.face_map_binding) != face_binding:
    raise ValueError(
        "mate manifest face-map binding differs from authenticated receipt"
    )
  return receipts


def _require_formal_mate_receipt_readiness(receipts: Any) -> None:
  face_binding = dict(receipts.face_map_binding)
  if not (
      face_binding.get("binding_mode") == "private_formal_receipt"
      and face_binding.get("formal_gold_eligible") is True
      and face_binding.get("gold_interface_certificate_ready") is True
  ):
    raise ValueError(
        "formal mate loading is unavailable: the current authenticated face-map "
        "receipt schema is development-only and cannot assert formal readiness"
    )


def _require_formal_mate_semantic_authority(
    *,
    manifest: dict[str, Any],
    authenticated_receipts: Any,
    authority: AuthenticatedMateSemanticAuthority,
    library_sha256: str,
    manifest_sha256: str,
) -> None:
  """Require the factory-issued replay handle to close over this load."""

  if not isinstance(authority, AuthenticatedMateSemanticAuthority):
    raise TypeError("formal mate loading requires authenticated semantic authority")
  authority.revalidate()
  if authority.schema_version not in FORMAL_MATE_SEMANTIC_AUTHORITY_SCHEMA_ALLOWLIST:
    raise ValueError("mate semantic authority schema is not allowlisted")
  if (
      authority.library_sha256 != library_sha256
      or authority.manifest_sha256 != manifest_sha256
  ):
    raise ValueError("mate semantic authority binds different library bytes")
  mining_config = manifest.get("mining_config")
  if not isinstance(mining_config, dict):
    raise ValueError("mate semantic authority requires manifest mining_config")
  if dict(authenticated_receipts.private_supervision_binding) != mining_config.get(
      "private_supervision_binding"
  ):
    raise ValueError("mate semantic authority private supervision binding differs")
  if dict(authenticated_receipts.face_map_binding) != mining_config.get(
      "fusion_step_face_map_binding"
  ):
    raise ValueError("mate semantic authority face-map binding differs")


def _validate_loaded_program_coverage(
    programs: list[MateProgram],
    *,
    manifest: dict[str, Any],
    family_verification: dict[str, Any],
) -> None:
  case_identity_by_token = family_verification.get("case_identity_by_token")
  if not isinstance(case_identity_by_token, dict) or any(
      _SHA256.fullmatch(str(token)) is None
      or _OPAQUE_CASE_ID.fullmatch(str(opaque_case)) is None
      for token, opaque_case in case_identity_by_token.items()
  ):
    raise ValueError("formal family case identity replay is invalid")
  valid_case_tokens = set(case_identity_by_token)
  ledger_cases = manifest["coverage_ledger"]["cases"]
  ledger_tokens = {
      str(case["source_case_token_sha256"])
      for case in ledger_cases
  }
  if ledger_tokens != valid_case_tokens:
    raise ValueError("formal mate library coverage ledger does not cover train cases")

  actual_counts: Counter[tuple[str, int]] = Counter()
  program_ids: set[str] = set()
  for program in programs:
    if program.program_id in program_ids:
      raise ValueError("formal mate library contains duplicate program_id")
    program_ids.add(program.program_id)
    token = str(program.metadata.get("source_case_token_sha256") or "")
    contact_index = program.source_contact.get("contact_index")
    if token not in valid_case_tokens:
      raise ValueError("formal mate library row has an unknown source case token")
    if program.case_id != case_identity_by_token[token]:
      raise ValueError(
          "formal mate library row opaque case identity does not match its source token"
      )
    if type(contact_index) is not int or contact_index < 0:
      raise ValueError("formal mate library row has an invalid source contact index")
    actual_counts[(token, contact_index)] += 1

  expected_counts: Counter[tuple[str, int]] = Counter()
  for case in ledger_cases:
    token = str(case["source_case_token_sha256"])
    for contact in case["contacts"]:
      count = int(contact["directed_program_count"])
      if count > 0:
        expected_counts[(token, int(contact["contact_index"]))] = count
  if actual_counts != expected_counts:
    raise ValueError("formal mate library rows do not match the contact coverage ledger")


def load_mate_programs(
    path: str | Path,
    *,
    strict: bool = False,
    expected_manifest: Optional[dict[str, Any]] = None,
) -> list[MateProgram]:
  if strict:
    captured = _CapturedFile.capture(path, label="formal mate library")
    programs = _load_mate_programs_payload(
        captured.payload,
        strict=True,
        expected_manifest=expected_manifest,
    )
    captured.reverify()
    return programs
  with Path(path).open("r", encoding="utf-8") as stream:
    return _load_mate_program_lines(
        stream,
        strict=False,
        expected_manifest=expected_manifest,
    )


def _load_mate_programs_payload(
    payload: bytes,
    *,
    strict: bool,
    expected_manifest: Optional[dict[str, Any]],
) -> list[MateProgram]:
  try:
    lines = payload.decode("utf-8").splitlines()
  except UnicodeDecodeError as error:
    if strict:
      raise ValueError("formal mate library is not UTF-8") from error
    return []
  return _load_mate_program_lines(
      lines,
      strict=strict,
      expected_manifest=expected_manifest,
  )


def _load_mate_program_lines(
    lines: Iterable[str],
    *,
    strict: bool,
    expected_manifest: Optional[dict[str, Any]],
) -> list[MateProgram]:
  programs: list[MateProgram] = []
  for line_number, line in enumerate(lines, start=1):
    text = line.strip()
    if not text:
      continue
    try:
      data = json.loads(
          text,
          object_pairs_hook=_unique_json_object if strict else None,
      )
      if isinstance(data, dict):
        if strict:
          _validate_v2_program_row(
              data,
              row_number=line_number,
              expected_manifest=expected_manifest,
          )
        programs.append(MateProgram.from_dict(data))
      elif strict:
        raise ValueError(f"formal mate library row {line_number} is not an object")
    except Exception as error:
      if strict:
        if isinstance(error, ValueError) and f"row {line_number}" in str(error):
          raise
        raise ValueError(
            f"formal mate library row {line_number} is malformed"
        ) from error
      continue
  return programs


def _canonical_sha256(payload: Any) -> str:
  encoded = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def mate_miner_code_bundle_identity(
    *,
    _session: _FormalVerificationSession | None = None,
) -> dict[str, Any]:
  """Return the content-addressed identity of the declared mining code bundle."""

  module_dir = Path(__file__).resolve().parent
  files = []
  owns_session = _session is None
  session = _session or _FormalVerificationSession()
  for relative_path in _MATE_MINER_CODE_FILES:
    path = module_dir / relative_path
    captured = session.capture(
        path,
        label=f"mate miner code file {relative_path}",
    )
    files.append(
        {
            "path": relative_path,
            "sha256": captured.sha256,
            "bytes": len(captured.payload),
        }
    )
  payload: dict[str, Any] = {
      "schema_version": MATE_MINER_CODE_BUNDLE_SCHEMA,
      "files": files,
  }
  payload["bundle_sha256"] = _canonical_sha256(payload)
  if owns_session:
    session.revalidate_all()
  return payload


def _load_case_artifact(
    captured: _CapturedFile,
    *,
    split: str,
) -> list[dict[str, Any]]:
  document = _json_from_captured_file(captured)
  cases = document if isinstance(document, list) else (
      document.get("cases") if isinstance(document, dict) else None
  )
  if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
    raise ValueError(f"formal family {split} artifact must contain case objects")
  return cases


def _resolve_family_artifact(
    *,
    manifest_path: Path,
    record: dict[str, Any],
    split: str,
    override: Path | None = None,
) -> Path:
  if override is not None:
    if not override.is_file():
      raise ValueError(f"formal family {split} artifact override is missing")
    return override
  raw_path = str(record.get("path") or "").strip()
  if not raw_path:
    raise ValueError(f"formal family manifest {split} artifact has no path")
  recorded = Path(raw_path)
  candidates = [recorded]
  if not recorded.is_absolute():
    candidates.append(manifest_path.resolve().parent / recorded)
  candidates.append(manifest_path.resolve().parent / recorded.name)
  resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
  if resolved is None:
    raise ValueError(f"formal family {split} artifact cannot be resolved")
  return resolved


def _formal_case_identity_by_token(
    train_cases: list[dict[str, Any]],
    *,
    family_manifest_sha256: str,
) -> dict[str, str]:
  """Replay the producer's exact case-token and opaque-case algorithms."""

  result: dict[str, str] = {}
  seen_opaque: set[str] = set()
  for case in train_cases:
    case_id = str(case.get("id") or "")
    if not case_id:
      raise ValueError("formal family train case lacks an id")
    token = hashlib.sha256(
        f"benchmark_v2_case\0{family_manifest_sha256}\0{case_id}".encode("utf-8")
    ).hexdigest()
    opaque_case = "opaque_case_" + _canonical_sha256(
        {
            "id": case_id,
            "assembly_dir": str(case.get("assembly_dir") or ""),
        }
    )[:20]
    if token in result or opaque_case in seen_opaque:
      raise ValueError("formal family case identity mapping is not one-to-one")
    result[token] = opaque_case
    seen_opaque.add(opaque_case)
  return result


def validate_formal_family_split_manifest(
    family_manifest_path: str | Path,
    *,
    train_artifact_override: str | Path | None = None,
    _session: _FormalVerificationSession | None = None,
) -> dict[str, Any]:
  """Fully replay a formal family split from its authoritative source.

  This deliberately delegates to the paper-facing hygiene verifier instead of
  trusting self-asserted ``formal`` flags or a hand-written case list.
  """

  owns_session = _session is None
  session = _session or _FormalVerificationSession()
  manifest_path = Path(family_manifest_path)
  manifest_capture = session.capture(
      manifest_path,
      label="formal family split manifest",
  )
  family = _json_from_captured_file(manifest_capture)
  if not isinstance(family, dict):
    raise ValueError("formal family split manifest must be an object")
  if (
      family.get("benchmark") != "benchmark_v2"
      or family.get("formal") is not True
      or family.get("strict_hygiene_clean") is not True
  ):
    raise ValueError("mate library requires a clean formal benchmark_v2 family split")
  hashes = family.get("artifact_hashes")
  if not isinstance(hashes, dict):
    raise ValueError("formal family split manifest lacks artifact hashes")
  split_captures: dict[str, _CapturedFile] = {}
  raw_cases: dict[str, list[dict[str, Any]]] = {}
  for split in ("train", "dev"):
    record = hashes.get(split)
    if not isinstance(record, dict):
      raise ValueError(f"formal family split manifest does not bind {split}")
    override = (
        Path(train_artifact_override)
        if split == "train" and train_artifact_override is not None
        else None
    )
    artifact_path = _resolve_family_artifact(
        manifest_path=manifest_path,
        record=record,
        split=split,
        override=override,
    )
    split_captures[split] = session.capture(
        artifact_path,
        label=f"formal family {split} artifact",
    )
    raw_cases[split] = _load_case_artifact(split_captures[split], split=split)

  source_record = hashes.get("input_cases")
  if not isinstance(source_record, dict):
    raise ValueError(
        "formal family split manifest lacks authoritative input_cases"
    )
  source_path = _resolve_family_artifact(
      manifest_path=manifest_path,
      record=source_record,
      split="authoritative source",
  )
  source_capture = session.capture(
      source_path,
      label="formal family authoritative source",
  )
  source_document = _json_from_captured_file(source_capture)
  source_cases = (
      source_document
      if isinstance(source_document, list)
      else source_document.get("cases")
      if isinstance(source_document, dict)
      else None
  )
  if not isinstance(source_cases, list) or not all(
      isinstance(case, dict) for case in source_cases
  ):
    raise ValueError(
        "formal family authoritative source must contain case objects"
    )

  # Local import avoids the module cycle: split_hygiene_audit imports the
  # dependency-light row loader from this module.
  from .split_hygiene_audit import _verify_formal_split_manifest

  # The hygiene verifier historically re-opened live paths while recomputing
  # provenance.  Feed it private immutable snapshots made from the captured
  # bytes, then re-check every original path after all use.
  with tempfile.TemporaryDirectory(prefix="neurocad_mate_family_verify_") as raw_dir:
    snapshot_root = Path(raw_dir)
    snapshot_manifest = snapshot_root / "split_manifest.json"
    snapshot_source = snapshot_root / "source_cases.json"
    snapshot_paths = {
        split: snapshot_root / f"{split}.json" for split in ("train", "dev")
    }
    snapshot_manifest.write_bytes(manifest_capture.payload)
    snapshot_source.write_bytes(source_capture.payload)
    for split, captured in split_captures.items():
      snapshot_paths[split].write_bytes(captured.payload)
    verified_family, verification = _verify_formal_split_manifest(
        snapshot_manifest,
        split_paths=snapshot_paths,
        raw_cases_by_split=raw_cases,
        authoritative_source_path=snapshot_source,
    )
  if verified_family != family:
    raise ValueError("formal family snapshot manifest changed during replay")
  for split, captured in split_captures.items():
    artifact_verification = verification.get("artifacts", {}).get(split, {})
    if (
        artifact_verification.get("sha256") != captured.sha256
        or artifact_verification.get("bytes") != len(captured.payload)
    ):
      raise ValueError(f"formal family {split} snapshot identity mismatch")
  if not raw_cases["dev"]:
    raise ValueError("formal family split requires a nonempty dev artifact")
  train_case_ids = [str(case.get("id") or "") for case in raw_cases["train"]]
  if not train_case_ids or any(not case_id for case_id in train_case_ids):
    raise ValueError("formal family train artifact has invalid case ids")
  if len(train_case_ids) != len(set(train_case_ids)):
    raise ValueError("formal family train artifact has duplicate case ids")
  case_identity_by_token = _formal_case_identity_by_token(
      raw_cases["train"],
      family_manifest_sha256=manifest_capture.sha256,
  )
  if owns_session:
    session.revalidate_all()
  return {
      "manifest_sha256": manifest_capture.sha256,
      "artifacts": verification["artifacts"],
      "authoritative_source": {
          **verification["authoritative_source"],
          "path": str(source_capture.resolved_path),
          "sha256": source_capture.sha256,
      },
      "train_case_ids": train_case_ids,
      "case_identity_by_token": case_identity_by_token,
      "artifact_paths": {
          split: str(split_captures[split].path) for split in ("train", "dev")
      },
  }


def _parse_v2_library_manifest(
    captured: _CapturedFile,
    *,
    session: _FormalVerificationSession,
) -> dict[str, Any]:
  raw = _json_from_captured_file(captured)
  if not isinstance(raw, dict):
    raise ValueError("formal mate library manifest must be an object")
  required_top_level = {
      "schema_version",
      "model_input_protocol",
      "protocol_version",
      "source_split",
      "source_split_sha256",
      "mining_config",
      "mining_config_sha256",
      "miner_code_bundle",
      "miner_code_bundle_sha256",
      "family_split_manifest_sha256",
      "source_case_set_sha256",
      "coverage",
      "coverage_ledger",
      "library_sha256",
      "row_count",
      "manifest_sha256",
  }
  if set(raw) != required_top_level:
    raise ValueError("formal mate library manifest has an invalid top-level schema")
  unsigned = dict(raw)
  stored_hash = unsigned.pop("manifest_sha256", None)
  if stored_hash != _canonical_sha256(unsigned):
    raise ValueError("formal mate library manifest hash mismatch")
  expected = {
      "schema_version": MATE_LIBRARY_MANIFEST_SCHEMA,
      "model_input_protocol": "benchmark_v2",
      "source_split": "train",
  }
  if any(raw.get(key) != value for key, value in expected.items()):
    raise ValueError("formal mate library manifest protocol/split mismatch")
  for key in (
      "source_split_sha256",
      "mining_config_sha256",
      "library_sha256",
  ):
    if _SHA256.fullmatch(str(raw.get(key) or "")) is None:
      raise ValueError(f"formal mate library manifest lacks valid {key}")
  if raw.get("protocol_version") != PROTOCOL_VERSION:
    raise ValueError("formal mate library manifest protocol_version mismatch")
  if type(raw.get("row_count")) is not int or int(raw["row_count"]) < 1:
    raise ValueError("formal mate library manifest row_count must be positive")
  for key in ("family_split_manifest_sha256", "source_case_set_sha256"):
    if _SHA256.fullmatch(str(raw.get(key) or "")) is None:
      raise ValueError(f"formal mate library manifest lacks valid {key}")
  mining_config = raw.get("mining_config")
  mining_config_keys = {
      "schema_version",
      "model_input_protocol",
      "protocol_version",
      "source_split",
      "source_split_sha256",
      "family_split_manifest_sha256",
      "source_case_set_sha256",
      "max_cases",
      "max_candidates_per_part",
      "max_candidates_per_contact_face",
      "canonicalize_parts",
      "bidirectional",
      "filter_implausible_programs",
      "frame_randomization_seed",
      "benchmark_v2_translation_box_fraction",
      "interface_scorer_sha256",
      "fusion_step_face_map_binding",
      "private_supervision_binding",
      "label_policy",
      "source_pose_contract",
  }
  if not isinstance(mining_config, dict) or set(mining_config) != mining_config_keys:
    raise ValueError("formal mate library manifest has incomplete mining_config")
  expected_config_bindings = {
      "schema_version": MATE_MINING_CONFIG_SCHEMA,
      "model_input_protocol": "benchmark_v2",
      "protocol_version": PROTOCOL_VERSION,
      "source_split": "train",
      "source_split_sha256": raw["source_split_sha256"],
      "family_split_manifest_sha256": raw["family_split_manifest_sha256"],
      "source_case_set_sha256": raw["source_case_set_sha256"],
      "max_cases": 0,
  }
  if any(
      mining_config.get(key) != value
      for key, value in expected_config_bindings.items()
  ):
    raise ValueError("formal mate library mining_config provenance mismatch")
  if type(mining_config["max_cases"]) is not int:
    raise ValueError("formal mate library has invalid mining_config max_cases")
  if (
      type(mining_config["max_candidates_per_part"]) is not int
      or mining_config["max_candidates_per_part"] != 0
  ):
    raise ValueError(
        "formal mate library has invalid mining_config max_candidates_per_part"
    )
  if (
      type(mining_config["max_candidates_per_contact_face"]) is not int
      or mining_config["max_candidates_per_contact_face"] < 1
  ):
    raise ValueError(
        "formal mate library has invalid mining_config "
        "max_candidates_per_contact_face"
    )
  if type(mining_config["frame_randomization_seed"]) is not int:
    raise ValueError("formal mate library has invalid frame_randomization_seed")
  for key in (
      "canonicalize_parts",
      "bidirectional",
      "filter_implausible_programs",
  ):
    if not isinstance(mining_config[key], bool):
      raise ValueError(f"formal mate library has invalid mining_config {key}")
  translation_fraction = mining_config["benchmark_v2_translation_box_fraction"]
  if (
      not isinstance(translation_fraction, (int, float))
      or isinstance(translation_fraction, bool)
      or not math.isfinite(float(translation_fraction))
      or float(translation_fraction) <= 0.0
  ):
    raise ValueError("formal mate library has invalid translation box fraction")
  scorer_sha256 = mining_config["interface_scorer_sha256"]
  if scorer_sha256 is not None and _SHA256.fullmatch(str(scorer_sha256)) is None:
    raise ValueError("formal mate library has invalid interface scorer hash")
  validate_face_map_binding(
      mining_config["fusion_step_face_map_binding"],
      require_private_receipt=False,
  )
  supervision_binding = validate_private_supervision_binding(
      mining_config["private_supervision_binding"],
      require_certified_binary=False,
  )
  if (
      supervision_binding["split"] != "train"
      or supervision_binding["family_split_manifest_sha256"]
      != raw["family_split_manifest_sha256"]
      or supervision_binding["gold_contact_schema_version"]
      != "benchmark_v2_evaluation_gold_contacts.v2"
      or supervision_binding["gold_label_semantics"]
      != "source_positive_assertions_only"
      or supervision_binding["gold_interface_certificate_ready"] is not False
      or supervision_binding["certified_negative_complete"] is not False
  ):
    raise ValueError("formal mate library private supervision binding mismatch")
  if mining_config["label_policy"] != "source_positive_only_unknown_excluded":
    raise ValueError("formal mate library label policy is invalid")
  if mining_config["source_pose_contract"] != (
      "fusion_occurrence_tree_parent_chain_axes_columns_cm_to_mm.v1"
  ):
    raise ValueError("formal mate library source pose contract is invalid")
  if _canonical_sha256(mining_config) != raw["mining_config_sha256"]:
    raise ValueError("formal mate library mining_config hash mismatch")
  expected_code_bundle = mate_miner_code_bundle_identity(_session=session)
  if raw.get("miner_code_bundle") != expected_code_bundle:
    raise ValueError("formal mate library miner code bundle does not match this code")
  if raw.get("miner_code_bundle_sha256") != expected_code_bundle["bundle_sha256"]:
    raise ValueError("formal mate library miner code bundle hash mismatch")
  coverage = raw.get("coverage")
  coverage_keys = {
      "expected_case_count",
      "processed_case_count",
      "successful_case_count",
      "cases_with_programs",
      "skipped_case_count",
      "contacts_seen",
      "contacts_skipped",
      "contacts_mined",
      "directed_program_count",
  }
  if not isinstance(coverage, dict) or set(coverage) != coverage_keys:
    raise ValueError("formal mate library manifest has invalid coverage schema")
  if any(type(coverage[key]) is not int or coverage[key] < 0 for key in coverage_keys):
    raise ValueError("formal mate library manifest coverage counts must be nonnegative")
  if coverage["expected_case_count"] < 1:
    raise ValueError("formal mate library manifest has no expected cases")
  if coverage["processed_case_count"] != coverage["expected_case_count"]:
    raise ValueError("formal mate library did not process the full train split")
  if coverage["successful_case_count"] + coverage["skipped_case_count"] != coverage["processed_case_count"]:
    raise ValueError("formal mate library case coverage counts are inconsistent")
  if coverage["cases_with_programs"] > coverage["successful_case_count"]:
    raise ValueError("formal mate library program case coverage is inconsistent")
  if coverage["contacts_mined"] + coverage["contacts_skipped"] != coverage["contacts_seen"]:
    raise ValueError("formal mate library contact coverage counts are inconsistent")
  if coverage["directed_program_count"] != raw["row_count"]:
    raise ValueError("formal mate library directed program count mismatch")
  _validate_coverage_ledger(raw.get("coverage_ledger"), coverage=coverage)
  return raw


def _validate_coverage_ledger(
    ledger: Any,
    *,
    coverage: dict[str, int],
) -> None:
  if not isinstance(ledger, dict) or set(ledger) != {"schema_version", "cases"}:
    raise ValueError("formal mate library has invalid coverage ledger schema")
  if ledger.get("schema_version") != MATE_COVERAGE_LEDGER_SCHEMA:
    raise ValueError("formal mate library coverage ledger version mismatch")
  cases = ledger.get("cases")
  if not isinstance(cases, list):
    raise ValueError("formal mate library coverage ledger cases must be a list")
  if len(cases) != coverage["expected_case_count"]:
    raise ValueError("formal mate library coverage ledger case count mismatch")
  case_keys = {
      "source_case_token_sha256",
      "status",
      "reason_code",
      "contacts_seen",
      "contacts_mined",
      "contacts_skipped",
      "directed_program_count",
      "contacts",
  }
  contact_keys = {
      "contact_index",
      "status",
      "reason_code",
      "directed_program_count",
  }
  allowed_skip_reasons = {
      "invalid_contact_entities",
      "outside_selected_part_set",
      "outside_selected_or_same_body",
      "invalid_face_index",
      "no_contact_face_candidates",
      "all_candidate_programs_filtered_or_invalid",
  }
  tokens: set[str] = set()
  aggregate_contacts_seen = 0
  aggregate_contacts_mined = 0
  aggregate_contacts_skipped = 0
  aggregate_programs = 0
  cases_with_programs = 0
  for case in cases:
    if not isinstance(case, dict) or set(case) != case_keys:
      raise ValueError("formal mate library has invalid case coverage ledger")
    token = str(case.get("source_case_token_sha256") or "")
    if _SHA256.fullmatch(token) is None or token in tokens:
      raise ValueError("formal mate library coverage ledger has invalid case token")
    tokens.add(token)
    # Formal mining is zero-unknown-skip: a case failure must abort generation,
    # not be converted into a self-declared reason inside the receipt.
    if case.get("status") != "ok" or case.get("reason_code") is not None:
      raise ValueError("formal mate library coverage ledger contains a skipped case")
    contacts = case.get("contacts")
    if not isinstance(contacts, list):
      raise ValueError("formal mate library case coverage contacts must be a list")
    mined = 0
    skipped = 0
    directed = 0
    seen_contact_indices: set[int] = set()
    previous_contact_index = -1
    for contact in contacts:
      if not isinstance(contact, dict) or set(contact) != contact_keys:
        raise ValueError("formal mate library has invalid contact coverage ledger")
      contact_index = contact.get("contact_index")
      if (
          type(contact_index) is not int
          or contact_index < 0
          or contact_index in seen_contact_indices
          or contact_index <= previous_contact_index
      ):
        raise ValueError(
            "formal mate library contact coverage indices must be unique, "
            "nonnegative, and strictly increasing source ordinals"
        )
      seen_contact_indices.add(contact_index)
      previous_contact_index = contact_index
      row_count = contact.get("directed_program_count")
      if type(row_count) is not int or row_count < 0:
        raise ValueError("formal mate library contact coverage row count is invalid")
      if row_count > 0:
        if contact.get("status") != "mined" or contact.get("reason_code") is not None:
          raise ValueError("formal mate library mined contact ledger is inconsistent")
        mined += 1
      else:
        if (
            contact.get("status") != "skipped"
            or contact.get("reason_code") not in allowed_skip_reasons
        ):
          raise ValueError("formal mate library skipped contact lacks a known reason")
        skipped += 1
      directed += row_count
    expected_case_counts = {
        "contacts_seen": len(contacts),
        "contacts_mined": mined,
        "contacts_skipped": skipped,
        "directed_program_count": directed,
    }
    if any(
        type(case.get(key)) is not int or case.get(key) != value
        for key, value in expected_case_counts.items()
    ):
      raise ValueError("formal mate library case coverage ledger does not recompute")
    aggregate_contacts_seen += len(contacts)
    aggregate_contacts_mined += mined
    aggregate_contacts_skipped += skipped
    aggregate_programs += directed
    if directed > 0:
      cases_with_programs += 1
  recomputed = {
      "processed_case_count": len(cases),
      "successful_case_count": len(cases),
      "skipped_case_count": 0,
      "cases_with_programs": cases_with_programs,
      "contacts_seen": aggregate_contacts_seen,
      "contacts_mined": aggregate_contacts_mined,
      "contacts_skipped": aggregate_contacts_skipped,
      "directed_program_count": aggregate_programs,
  }
  if any(coverage[key] != value for key, value in recomputed.items()):
    raise ValueError("formal mate library aggregate coverage does not match its ledger")


def _validate_family_split_manifest_binding(
    mate_manifest: dict[str, Any],
    family_manifest_path: Path,
    *,
    session: _FormalVerificationSession,
) -> dict[str, Any]:
  verification = validate_formal_family_split_manifest(
      family_manifest_path,
      _session=session,
  )
  if verification["manifest_sha256"] != mate_manifest.get(
      "family_split_manifest_sha256"
  ):
    raise ValueError("mate library family split manifest hash mismatch")
  train_sha256 = verification["artifacts"]["train"]["sha256"]
  if train_sha256 != mate_manifest.get("source_split_sha256"):
    raise ValueError("mate library train split hash mismatch")
  train_case_ids = sorted(verification["train_case_ids"])
  if _canonical_sha256(train_case_ids) != mate_manifest.get("source_case_set_sha256"):
    raise ValueError("mate library source case set mismatch")
  if len(train_case_ids) != int(mate_manifest["coverage"]["expected_case_count"]):
    raise ValueError("mate library expected case count mismatches family split")
  return verification


def _validate_v2_program_row(
    row: dict[str, Any],
    *,
    row_number: int,
    expected_manifest: Optional[dict[str, Any]],
) -> None:
  if set(row) != set(_FORMAL_PROGRAM_ROW_KEYS):
    missing = sorted(_FORMAL_PROGRAM_ROW_KEYS - set(row))
    extras = sorted(set(row) - _FORMAL_PROGRAM_ROW_KEYS)
    raise ValueError(
        f"formal mate library row {row_number} has invalid top-level fields: "
        f"missing={','.join(missing) or '-'}; extras={','.join(extras) or '-'}"
    )
  if row.get("model_input_protocol") != "benchmark_v2":
    raise ValueError(
        f"formal mate library row {row_number} protocol mismatch"
    )
  if _PROGRAM_ID.fullmatch(str(row.get("program_id") or "")) is None:
    raise ValueError(f"formal mate library row {row_number} has invalid program_id")
  if _OPAQUE_CASE_ID.fullmatch(str(row.get("case_id") or "")) is None:
    raise ValueError(f"formal mate library row {row_number} exposes case identity")
  if row.get("assembly_dir") != "":
    raise ValueError(
        f"formal mate library row {row_number} exposes assembly_dir"
    )
  part_a = str(row.get("part_a") or "")
  part_b = str(row.get("part_b") or "")
  body_a = str(row.get("body_uuid_a") or "")
  body_b = str(row.get("body_uuid_b") or "")
  if (
      _OPAQUE_PART_ID.fullmatch(part_a) is None
      or _OPAQUE_PART_ID.fullmatch(part_b) is None
  ):
    raise ValueError(f"formal mate library row {row_number} exposes non-opaque parts")
  if (
      _OPAQUE_BODY_ID.fullmatch(body_a) is None
      or _OPAQUE_BODY_ID.fullmatch(body_b) is None
  ):
    raise ValueError(f"formal mate library row {row_number} exposes body identity")
  if part_a == part_b or body_a == body_b:
    raise ValueError(
        f"formal mate library row {row_number} repeats one opaque instance"
    )
  if part_a.removeprefix("opaque_part_") != body_a.removeprefix(
      "opaque_body_"
  ) or part_b.removeprefix("opaque_part_") != body_b.removeprefix(
      "opaque_body_"
  ):
    raise ValueError(
        f"formal mate library row {row_number} has inconsistent opaque identities"
    )
  if row.get("relation_hint") not in _FORMAL_RELATION_HINTS:
    raise ValueError(
        f"formal mate library row {row_number} has invalid relation_hint"
    )
  if row.get("contact_type") not in _FORMAL_CONTACT_TYPES:
    raise ValueError(
        f"formal mate library row {row_number} has invalid contact_type"
    )
  for key in ("interface_a", "interface_b", "source_contact", "features", "metadata"):
    if not isinstance(row.get(key), dict):
      raise ValueError(f"formal mate library row {row_number} has invalid {key}")
  for key in ("interface_a", "interface_b"):
    if set(row[key]) != set(_FORMAL_ENDPOINT_KEYS):
      raise ValueError(
          f"formal mate library row {row_number} has unsafe or incomplete {key}"
      )
    score = row[key].get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
      raise ValueError(
          f"formal mate library row {row_number} has invalid {key} score"
      )
  try:
    recomputed_features = pair_features_from_rows(
        row["interface_a"],
        row["interface_b"],
        relation_hint=str(row.get("relation_hint") or "link"),
        protocol="benchmark_v2",
    )
  except (TypeError, ValueError) as error:
    raise ValueError(
        f"formal mate library row {row_number} has invalid sanitized endpoints"
    ) from error
  if set(row["features"]) != set(BENCHMARK_V2_PAIR_FEATURE_NAMES):
    raise ValueError(
        f"formal mate library row {row_number} has unsafe feature schema"
    )
  if any(
      isinstance(value, bool)
      or not isinstance(value, (int, float))
      or not math.isfinite(float(value))
      for value in row["features"].values()
  ):
    raise ValueError(
        f"formal mate library row {row_number} has invalid feature values"
    )
  try:
    if any(
        not math.isclose(
            float(row["features"][name]),
            float(recomputed_features[name]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for name in BENCHMARK_V2_PAIR_FEATURE_NAMES
    ):
      raise ValueError
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError(
        f"formal mate library row {row_number} feature values do not recompute"
    ) from error
  source_contact = row["source_contact"]
  if set(source_contact) != set(_FORMAL_SOURCE_CONTACT_KEYS):
    raise ValueError(
      f"formal mate library row {row_number} exposes unsafe source contact fields"
    )
  if (
      type(source_contact.get("contact_index")) is not int
      or source_contact["contact_index"] < 0
      or source_contact.get("direction") not in _FORMAL_DIRECTIONS
      or source_contact.get("surface_type_a") not in _FORMAL_SOURCE_SURFACES
      or source_contact.get("surface_type_b") not in _FORMAL_SOURCE_SURFACES
  ):
    raise ValueError(
        f"formal mate library row {row_number} has invalid source contact identity"
    )
  if any(
      source_contact.get(key) is not None
      for key in ("face_index_a", "face_index_b")
  ):
    raise ValueError(
        f"formal mate library row {row_number} exposes private face indices"
    )
  raw_rotation = row["residual_rotation"]
  raw_translation = row["residual_translation"]
  rotation_shape_valid = (
      isinstance(raw_rotation, list)
      and len(raw_rotation) == 3
      and all(isinstance(values, list) and len(values) == 3 for values in raw_rotation)
  )
  rotation_values = (
      [value for values in raw_rotation for value in values]
      if rotation_shape_valid
      else []
  )
  translation_shape_valid = isinstance(raw_translation, list) and len(
      raw_translation
  ) == 3
  if (
      not rotation_shape_valid
      or not translation_shape_valid
      or any(
          isinstance(value, bool) or not isinstance(value, (int, float))
          for value in rotation_values + list(raw_translation if translation_shape_valid else [])
      )
  ):
    raise ValueError(
        f"formal mate library row {row_number} has invalid residual transform"
    )
  rotation = np.asarray(raw_rotation, dtype=float)
  translation = np.asarray(raw_translation, dtype=float)
  if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
    raise ValueError(
        f"formal mate library row {row_number} has non-finite residual transform"
    )
  orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
  determinant = float(np.linalg.det(rotation))
  if orthogonality_error > 1e-5 or abs(determinant - 1.0) > 1e-5:
    raise ValueError(
        f"formal mate library row {row_number} residual rotation is not SO(3)"
    )
  metadata = row["metadata"]
  if set(metadata) != set(_FORMAL_METADATA_KEYS):
    raise ValueError(
        f"formal mate library row {row_number} exposes unsafe metadata fields"
    )
  manifest = expected_manifest or {}
  bindings = {
      "model_input_protocol": "benchmark_v2",
      "source_split": "train",
      "source_split_sha256": manifest.get("source_split_sha256"),
      "mining_config_sha256": manifest.get("mining_config_sha256"),
      "family_split_manifest_sha256": manifest.get(
          "family_split_manifest_sha256"
      ),
      "source_case_set_sha256": manifest.get("source_case_set_sha256"),
  }
  if any(metadata.get(key) != value for key, value in bindings.items()):
    raise ValueError(
        f"formal mate library row {row_number} provenance binding mismatch"
    )
  if _SHA256.fullmatch(str(metadata.get("source_case_token_sha256") or "")) is None:
    raise ValueError(
        f"formal mate library row {row_number} lacks source case token"
    )
  expected_metadata = {
      "source": "assembly_contact_mined_mate_program",
      "leakage_scope": "train_or_dev_only",
      "part_frame": "benchmark_v2_independent_se3",
      "bidirectional_direction": source_contact["direction"],
  }
  if any(metadata.get(key) != value for key, value in expected_metadata.items()):
    raise ValueError(
        f"formal mate library row {row_number} metadata contract mismatch"
    )


def _index_seating_depths(
    programs: Iterable[MateProgram],
) -> tuple[dict[tuple[str, str, str], list[float]], dict[tuple[str, str, str], list[float]]]:
  exact: dict[tuple[str, str, str], list[float]] = {}
  family: dict[tuple[str, str, str], list[float]] = {}
  for program in programs:
    depth = _residual_z(program)
    if depth is None:
      continue
    exact.setdefault(_seating_exact_key(program), []).append(float(depth))
    family.setdefault(_seating_family_key(program), []).append(float(depth))
  for bucket in (exact, family):
    for key, values in list(bucket.items()):
      bucket[key] = sorted(float(value) for value in values)
  return exact, family


def _residual_z(program: MateProgram) -> Optional[float]:
  try:
    values = list(program.residual_translation)
    if len(values) < 3:
      return None
    depth = float(values[2])
  except Exception:
    return None
  if not math.isfinite(depth):
    return None
  return depth


def _seating_exact_key(program: MateProgram) -> tuple[str, str, str]:
  return (
      _contact_family(program.contact_type),
      str(program.parent_role or "").lower(),
      str(program.child_role or "").lower(),
  )


def _seating_family_key(program: MateProgram) -> tuple[str, str, str]:
  return (
      _contact_family(program.contact_type),
      _role_family(program.parent_role),
      _role_family(program.child_role),
  )


def _seating_depth_targets(
    values: Iterable[float],
    *,
    max_targets: int,
) -> list[float]:
  clean = sorted(float(value) for value in values if math.isfinite(float(value)))
  if not clean:
    return []
  candidates = [
      _quantile(clean, 0.50),
      _quantile(clean, 0.25),
      _quantile(clean, 0.75),
      _quantile(clean, 0.10),
      _quantile(clean, 0.90),
  ]
  targets: list[float] = []
  for value in candidates:
    if _has_near_value(targets, value):
      continue
    targets.append(float(value))
    if len(targets) >= max(1, int(max_targets)):
      break
  return targets


def _merge_depth_targets(
    primary: Iterable[float],
    secondary: Iterable[float],
    limit: int,
) -> list[float]:
  merged: list[float] = []
  for value in list(primary) + list(secondary):
    if _has_near_value(merged, float(value)):
      continue
    merged.append(float(value))
    if len(merged) >= max(1, int(limit)):
      break
  return merged


def _quantile(values: list[float], q: float) -> float:
  if not values:
    return 0.0
  if len(values) == 1:
    return float(values[0])
  q = max(0.0, min(1.0, float(q)))
  position = q * (len(values) - 1)
  lo = int(math.floor(position))
  hi = int(math.ceil(position))
  if lo == hi:
    return float(values[lo])
  alpha = position - lo
  return float((1.0 - alpha) * values[lo] + alpha * values[hi])


def _has_near_value(values: Iterable[float], query: float, tol: float = 1e-6) -> bool:
  query = float(query)
  return any(abs(float(value) - query) <= float(tol) for value in values)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--library", default="neurocad/mate_program_library_train.jsonl")
  parser.add_argument("--contact_priors", action="store_true")
  sub = parser.add_subparsers(dest="command", required=True)
  sub.add_parser("stats")
  query = sub.add_parser("query")
  query.add_argument("--parent_role", required=True)
  query.add_argument("--child_role", required=True)
  query.add_argument("--relation_hint", default="")
  query.add_argument("--contact_type", default="")
  query.add_argument("--parent_surface", default="")
  query.add_argument("--child_surface", default="")
  query.add_argument("--parent_radius", type=float, default=None)
  query.add_argument("--child_radius", type=float, default=None)
  query.add_argument("--top_k", type=int, default=8)
  return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  args.library = str(resolve_path(args.library))
  return args


def main() -> None:
  args = normalize_paths(build_parser().parse_args())
  retriever = MateProgramRetriever.load(
      args.library,
      use_contact_priors=bool(args.contact_priors),
  )
  if args.command == "stats":
    print(json.dumps(retriever.stats(), indent=2))
    return
  if args.command == "query":
    rows = retriever.query(
        parent_role=str(args.parent_role),
        child_role=str(args.child_role),
        relation_hint=str(args.relation_hint or ""),
        contact_type=str(args.contact_type or ""),
        parent_surface=str(args.parent_surface or ""),
        child_surface=str(args.child_surface or ""),
        parent_radius=args.parent_radius,
        child_radius=args.child_radius,
        top_k=int(args.top_k),
    )
    print(json.dumps([row.to_dict() for row in rows], indent=2))


def _score_program(
    *,
    program: MateProgram,
    parent_role: str,
    child_role: str,
    relation_hint: str,
    contact_type: str,
    parent_surface: str,
    child_surface: str,
    parent_radius: Optional[float],
    child_radius: Optional[float],
    use_contact_priors: bool,
) -> tuple[float, list[str]]:
  score = 0.0
  reasons: list[str] = []
  parent_role = parent_role.lower()
  child_role = child_role.lower()
  relation_hint = relation_hint.lower()
  contact_type = contact_type.lower()
  parent_surface = parent_surface.lower()
  child_surface = child_surface.lower()

  if program.parent_role == parent_role:
    score += 4.0
    reasons.append("parent_role_exact")
  elif _role_family(program.parent_role) == _role_family(parent_role):
    score += 1.2
    reasons.append("parent_role_family")
  else:
    score -= 3.5

  if program.child_role == child_role:
    score += 4.0
    reasons.append("child_role_exact")
  elif _role_family(program.child_role) == _role_family(child_role):
    score += 1.2
    reasons.append("child_role_family")
  else:
    score -= 3.5

  if relation_hint:
    if program.relation_hint.lower() == relation_hint:
      score += 2.0
      reasons.append("relation_exact")
    elif _relation_family(program.relation_hint) == _relation_family(relation_hint):
      score += 0.7
      reasons.append("relation_family")
    else:
      score -= 1.0

  if contact_type:
    if program.contact_type.lower() == contact_type:
      score += 3.0
      reasons.append("contact_type_exact")
    elif _contact_family(program.contact_type) == _contact_family(contact_type):
      score += 1.0
      reasons.append("contact_type_family")
    else:
      score -= 1.5

  if use_contact_priors:
    plausibility = _query_contact_plausibility(
        parent_role=parent_role,
        child_role=child_role,
        contact_type=program.contact_type,
    )
    if plausibility < 0.0:
      score += plausibility
      reasons.append("query_contact_implausible")
    elif plausibility > 0.0:
      score += plausibility
      reasons.append("query_contact_plausible")

    axis_prior = _residual_axis_prior(program)
    if axis_prior != 0.0:
      score += axis_prior
      reasons.append("residual_axis_prior")

  if parent_surface and program.parent_surface == parent_surface:
    score += 0.8
    reasons.append("parent_surface")
  if child_surface and program.child_surface == child_surface:
    score += 0.8
    reasons.append("child_surface")

  radius_score = _radius_score(parent_radius, program.parent_radius)
  if radius_score is not None:
    score += radius_score
    reasons.append("parent_radius")
  radius_score = _radius_score(child_radius, program.child_radius)
  if radius_score is not None:
    score += radius_score
    reasons.append("child_radius")

  prior = program.features.get("heuristic_prior")
  if isinstance(prior, (int, float)):
    score += 0.15 * float(prior)
  return score, reasons


def _radius_score(query_radius: Optional[float], program_radius: Optional[float]) -> Optional[float]:
  if query_radius is None or program_radius is None:
    return None
  if query_radius <= 0.0 or program_radius <= 0.0:
    return None
  rel = abs(float(query_radius) - float(program_radius)) / max(
      float(query_radius),
      float(program_radius),
      1e-6,
  )
  if rel <= 0.05:
    return 1.5
  if rel <= 0.18:
    return 0.7
  if rel <= 0.5:
    return -0.5
  return -1.5


def _socket_role(socket: Socket) -> str:
  return str(
      socket.metadata.get("interface_role")
      or socket.metadata.get("composite_role")
      or socket.kind
      or ""
  ).lower()


def _socket_surface(socket: Socket) -> str:
  return str(socket.metadata.get("surface_type") or "").lower()


def _role_family(role: str) -> str:
  role = role.lower()
  if role in {"center_bore", "threaded_hole", "hole_entry"}:
    return "hole"
  if role in {"shaft_axis", "pin_boss", "cylindrical_interface"}:
    return "pin"
  if role in {"planar_seat", "shoulder_stop"}:
    return "plane"
  if "slot" in role:
    return "slot"
  return role


def _query_contact_plausibility(
    *,
    parent_role: str,
    child_role: str,
    contact_type: str,
) -> float:
  family = _contact_family(contact_type)
  parent_family = _role_family(parent_role)
  child_family = _role_family(child_role)
  families = {parent_family, child_family}
  if family == "insert":
    if "hole" in families and "pin" in families:
      return 0.8
    return -12.0
  if family == "slot":
    if "slot" in families and "pin" in families:
      return 0.7
    return -10.0
  if family == "seat":
    if parent_family == "plane" and child_family == "plane":
      return 0.6
    return -8.0
  return 0.0


def _residual_axis_prior(program: MateProgram) -> float:
  family = _contact_family(program.contact_type)
  if family not in {"insert", "seat"}:
    return 0.0
  try:
    rot = np.asarray(program.residual_rotation, dtype=float).reshape(3, 3)
  except Exception:
    return 0.0
  z_alignment = abs(float(rot[2, 2]))
  if z_alignment >= 0.85:
    return 0.35
  if z_alignment >= 0.55:
    return -1.5
  if z_alignment >= 0.25:
    return -3.0
  return -7.0


def _relation_family(relation: str) -> str:
  relation = relation.lower()
  if any(token in relation for token in ("insert", "bore", "hole", "shaft", "screw")):
    return "insert"
  if any(token in relation for token in ("seat", "support", "plane")):
    return "seat"
  if "slot" in relation:
    return "slot"
  return relation


def _contact_family(contact_type: str) -> str:
  text = contact_type.lower()
  if text in {"shaft_in_bore", "screw_in_hole", "insert_axis", "threaded_interference"}:
    return "insert"
  if text in {"seat_plane", "flange_contact"}:
    return "seat"
  if "slot" in text:
    return "slot"
  return text


if __name__ == "__main__":
  main()
