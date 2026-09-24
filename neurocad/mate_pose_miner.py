"""Mine directed mate programs from training assemblies.

This script uses assembly metadata only for train/dev supervision.  It does not
belong in test-time inference.  The output JSONL is a reusable library of local
interface residual transforms:

  parent_mcf @ residual == child_mcf

At inference time, the residual can be composed with newly extracted semantic
MCFs to propose a concrete child SE(3) pose without hand-written axial docking.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .batch_infer_assemble import (
    _index_step_files,
    _load_cases,
    _parse_dataset_roots,
    _resolve_part_path,
)
from .benchmark_v2_protocol import (
    BenchmarkV2PartSpec,
    PreparedBenchmarkV2Assembly,
    PROTOCOL_VERSION,
    prepare_benchmark_v2_assembly,
)
from .benchmark_v2_training_provenance import (
    canonical_sha256 as _training_canonical_sha256,
    case_token_sha256,
    validate_private_supervision_binding,
)
from .benchmark_v2_private_supervision import (
    load_benchmark_v2_private_supervision,
)
from .build_interface_pair_dataset import (
    _assembly_json_for_case,
    _candidate_contact_face_indices,
    _contact_type_from_pair,
    _load_case_part_interfaces,
    _rank_candidates,
    _relation_hint_from_roles,
)
from .build_interface_dataset import (
    _load_fusion_step_face_map_receipt,
    _open_verified_benchmark_v2_case_inputs,
    _mapped_fusion_contact_endpoint,
    _reverify_benchmark_v2_case_inputs,
    _validated_face_map_binding,
)
from .cadquery_backend import load_step_shape, shape_bbox
from .domain_types import Transform
from .frame_randomization import compose_se3, inverse_se3
from .interface_features import CandidateInterface
from .interface_pair_features import heuristic_pair_prior, pair_features_from_rows
from .interface_scorer import InterfaceScorer
from .mate_pose_retriever import (
    MATE_COVERAGE_LEDGER_SCHEMA,
    MATE_LIBRARY_MANIFEST_SCHEMA,
    MATE_MINING_CONFIG_SCHEMA,
    _validate_coverage_ledger,
    mate_miner_code_bundle_identity,
    validate_formal_family_split_manifest,
)
from .mate_programs import (
    frame_matrix_from_dict,
    invert_transform_matrix,
    matrix_from_transform,
    residual_between_frames,
    rotation_angle_degrees,
    transform_from_matrix,
)
from .offline import SocketExtractionConfig
from .paths import resolve_dataset_roots, resolve_path


@dataclass(frozen=True, slots=True)
class BenchmarkV2MateSemanticReplay:
  """Deterministic mate rows and coverage replayed from producer inputs."""

  rows: tuple[Mapping[str, Any], ...]
  coverage: Mapping[str, int]
  coverage_ledger: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkV2QueryMateSemanticReplay:
  """Query-local mate rows plus an exhaustive selected-ordinal ledger."""

  rows: tuple[Mapping[str, Any], ...]
  query_ledger: tuple[Mapping[str, Any], ...]
  input_domain_sha256: str
  opaque_case_id_by_case_id: Mapping[str, str]


def _replay_benchmark_v2_mate_semantics_core(
    *,
    public_cases: Sequence[Mapping[str, Any]],
    private_supervision: Any,
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    archive_root: str | Path,
    mining_config: Mapping[str, Any],
    family_split_manifest_sha256: str,
    source_split_sha256: str,
    _case_archive_root_resolver: Any = None,
    _diagnostic_frame_sink: list[dict[str, Any]] | None = None,
    _source_contact_ordinals_by_case: Mapping[str, frozenset[int]] | None = None,
) -> BenchmarkV2MateSemanticReplay:
  """Replay mate semantics, optionally restricted to query-selected ordinals.

  The selector is deliberately private.  Formal callers enter through
  ``replay_benchmark_v2_mate_semantics`` and always replay every case and
  source contact.
  """

  cases = [dict(case) for case in public_cases]
  case_ids = [str(case.get("id") or "").strip() for case in cases]
  if (
      not case_ids
      or any(not case_id for case_id in case_ids)
      or len(set(case_ids)) != len(case_ids)
  ):
    raise ValueError("mate semantic replay requires unique nonempty case ids")
  root: Path | None = None
  if _case_archive_root_resolver is None:
    root = Path(archive_root)
    if not root.is_dir() or root.is_symlink():
      raise ValueError("mate semantic replay archive_root must be a plain directory")
  expected_config = {
      "max_cases": 0,
      "model_input_protocol": "benchmark_v2",
      "source_split": "train",
  }
  if any(mining_config.get(key) != value for key, value in expected_config.items()):
    raise ValueError("mate semantic replay mining configuration is not formal train")
  if (
      type(mining_config.get("max_candidates_per_part")) is not int
      or mining_config.get("max_candidates_per_part") != 0
  ):
    raise ValueError(
        "mate semantic replay requires max_candidates_per_part == 0"
    )
  if (
      type(mining_config.get("max_candidates_per_contact_face")) is not int
      or int(mining_config["max_candidates_per_contact_face"]) < 1
  ):
    raise ValueError(
        "mate semantic replay requires max_candidates_per_contact_face >= 1"
    )
  if mining_config.get("interface_scorer_sha256") is not None:
    raise ValueError(
        "mate semantic replay requires an explicit authenticated interface scorer"
    )
  source_case_set_sha256 = _training_canonical_sha256(sorted(case_ids))
  if mining_config.get("source_case_set_sha256") != source_case_set_sha256:
    raise ValueError("mate semantic replay case-set commitment differs")
  if mining_config.get("family_split_manifest_sha256") != str(
      family_split_manifest_sha256
  ):
    raise ValueError("mate semantic replay family commitment differs")
  if mining_config.get("source_split_sha256") != str(source_split_sha256):
    raise ValueError("mate semantic replay train artifact commitment differs")

  config_sha256 = _canonical_sha256(dict(mining_config))
  source_case_tokens = {
      case_id: case_token_sha256(
          family_manifest_sha256=str(family_split_manifest_sha256),
          case_id=case_id,
      )
      for case_id in case_ids
  }
  archive_index_cache: dict[Path, dict[str, list[Path]]] = {}
  if root is not None:
    root = root.resolve(strict=True)
    archive_index_cache[root] = _index_step_files([root])
  replayed_rows: list[dict[str, Any]] = []
  coverage_cases: list[dict[str, Any]] = []
  seen_program_ids: set[str] = set()

  active_cases = (
      cases
      if _source_contact_ordinals_by_case is None
      else [
          case
          for case in cases
          if str(case["id"]) in _source_contact_ordinals_by_case
      ]
  )
  for case in active_cases:
    case_id = str(case["id"])
    source_case = private_supervision.source_case(case_id)
    gold_case = private_supervision.gold_case(case_id)
    if _case_archive_root_resolver is None:
      assert root is not None
      case_archive_root = root
    else:
      case_archive_root = Path(
          _case_archive_root_resolver(source_case)
      ).resolve(strict=True)
    dataset_roots = [case_archive_root]
    if case_archive_root not in archive_index_cache:
      archive_index_cache[case_archive_root] = _index_step_files(dataset_roots)
    file_index = archive_index_cache[case_archive_root]
    with _open_verified_benchmark_v2_case_inputs(
        source_case,
        dataset_roots=dataset_roots,
        file_index=file_index,
    ) as verified_inputs:
      case_summary, rows = _mine_case_programs(
          case=case,
          dataset_roots=dataset_roots,
          file_index=file_index,
          scorer=None,
          max_candidates_per_part=int(mining_config["max_candidates_per_part"]),
          max_candidates_per_contact_face=int(
              mining_config["max_candidates_per_contact_face"]
          ),
          canonicalize_parts=bool(mining_config["canonicalize_parts"]),
          bidirectional=bool(mining_config["bidirectional"]),
          filter_implausible_programs=bool(
              mining_config["filter_implausible_programs"]
          ),
          input_protocol="benchmark_v2",
          frame_randomization_seed=int(mining_config["frame_randomization_seed"]),
          benchmark_v2_translation_box_fraction=float(
              mining_config["benchmark_v2_translation_box_fraction"]
          ),
          fusion_step_face_lookup=face_map_lookup,
          benchmark_v2_verified_inputs=verified_inputs,
          private_source_case=source_case,
          evaluation_gold_contacts=gold_case,
          diagnostic_frame_sink=_diagnostic_frame_sink,
          source_contact_ordinals=(
              None
              if _source_contact_ordinals_by_case is None
              else _source_contact_ordinals_by_case[case_id]
          ),
      )
    if case_summary.get("status") != "ok":
      raise ValueError(
          "mate semantic replay refuses a skipped case: "
          + str(case_summary.get("status") or "unknown")
      )
    token = source_case_tokens[case_id]
    for row in rows:
      materialized = dict(row)
      metadata = dict(materialized.get("metadata") or {})
      metadata.update(
          {
              "model_input_protocol": "benchmark_v2",
              "source_split": "train",
              "source_split_sha256": str(source_split_sha256),
              "mining_config_sha256": config_sha256,
              "family_split_manifest_sha256": str(
                  family_split_manifest_sha256
              ),
              "source_case_set_sha256": source_case_set_sha256,
              "source_case_token_sha256": token,
          }
      )
      materialized["metadata"] = metadata
      program_id = str(materialized.get("program_id") or "")
      if not program_id or program_id in seen_program_ids:
        raise ValueError("mate semantic replay produced duplicate program_id")
      seen_program_ids.add(program_id)
      replayed_rows.append(materialized)
    contact_ledger = [dict(row) for row in case_summary.get("contact_ledger") or []]
    contacts_mined = sum(
        int(row.get("directed_program_count", 0)) > 0 for row in contact_ledger
    )
    directed_count = sum(
        int(row.get("directed_program_count", 0)) for row in contact_ledger
    )
    coverage_cases.append(
        {
            "source_case_token_sha256": token,
            "status": "ok",
            "reason_code": None,
            "contacts_seen": len(contact_ledger),
            "contacts_mined": contacts_mined,
            "contacts_skipped": len(contact_ledger) - contacts_mined,
            "directed_program_count": directed_count,
            "contacts": contact_ledger,
        }
    )

  directed_program_count = len(replayed_rows)
  contacts_seen = sum(int(row["contacts_seen"]) for row in coverage_cases)
  contacts_mined = sum(int(row["contacts_mined"]) for row in coverage_cases)
  contacts_skipped = sum(int(row["contacts_skipped"]) for row in coverage_cases)
  coverage = {
      "expected_case_count": len(active_cases),
      "processed_case_count": len(active_cases),
      "successful_case_count": len(active_cases),
      "cases_with_programs": sum(
          int(row["directed_program_count"]) > 0 for row in coverage_cases
      ),
      "skipped_case_count": 0,
      "contacts_seen": contacts_seen,
      "contacts_skipped": contacts_skipped,
      "contacts_mined": contacts_mined,
      "directed_program_count": directed_program_count,
  }
  coverage_ledger = {
      "schema_version": MATE_COVERAGE_LEDGER_SCHEMA,
      "cases": coverage_cases,
  }
  _validate_coverage_ledger(coverage_ledger, coverage=coverage)
  return BenchmarkV2MateSemanticReplay(
      rows=tuple(replayed_rows),
      coverage=coverage,
      coverage_ledger=coverage_ledger,
  )


def replay_benchmark_v2_mate_semantics(
    *,
    public_cases: Sequence[Mapping[str, Any]],
    private_supervision: Any,
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    archive_root: str | Path,
    mining_config: Mapping[str, Any],
    family_split_manifest_sha256: str,
    source_split_sha256: str,
    _case_archive_root_resolver: Any = None,
    _diagnostic_frame_sink: list[dict[str, Any]] | None = None,
) -> BenchmarkV2MateSemanticReplay:
  """Replay a complete formal mate library without reading that library.

  This public seam always reconstructs every case and source contact.  Query
  selection is isolated to the query-specific replay helper below.
  """

  return _replay_benchmark_v2_mate_semantics_core(
      public_cases=public_cases,
      private_supervision=private_supervision,
      face_map_lookup=face_map_lookup,
      archive_root=archive_root,
      mining_config=mining_config,
      family_split_manifest_sha256=family_split_manifest_sha256,
      source_split_sha256=source_split_sha256,
      _case_archive_root_resolver=_case_archive_root_resolver,
      _diagnostic_frame_sink=_diagnostic_frame_sink,
  )


def replay_benchmark_v2_query_mate_semantics(
    *,
    public_cases: Sequence[Mapping[str, Any]],
    private_supervision: Any,
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    archive_root: str | Path,
    mining_config: Mapping[str, Any],
    family_split_manifest_sha256: str,
    source_split_sha256: str,
    source_contact_ordinals: Mapping[str, Sequence[int]],
    _case_archive_root_resolver: Any = None,
    _diagnostic_frame_sink: list[dict[str, Any]] | None = None,
) -> BenchmarkV2QueryMateSemanticReplay:
  """Replay only an exact, result-independent set of private-gold ordinals."""

  case_ids = tuple(str(case.get("id") or "").strip() for case in public_cases)
  known = frozenset(case_ids)
  if (
      not isinstance(source_contact_ordinals, Mapping)
      or not source_contact_ordinals
      or not set(source_contact_ordinals).issubset(known)
  ):
    raise ValueError("query mate replay selector case domain differs")
  normalized: dict[str, frozenset[int]] = {}
  selector_rows: list[dict[str, Any]] = []
  for raw_case_id, raw_ordinals in source_contact_ordinals.items():
    case_id = str(raw_case_id)
    if isinstance(raw_ordinals, (str, bytes)) or not isinstance(
        raw_ordinals, Sequence
    ):
      raise ValueError("query mate replay ordinals must be a sequence")
    ordinals = tuple(raw_ordinals)
    if (
        not ordinals
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ordinals
        )
        or len(set(ordinals)) != len(ordinals)
    ):
      raise ValueError("query mate replay ordinals must be unique nonnegative integers")
    normalized[case_id] = frozenset(ordinals)
    selector_rows.extend(
        {"case_id": case_id, "source_contact_ordinal": ordinal}
        for ordinal in sorted(ordinals)
    )
  selector_rows.sort(
      key=lambda row: (row["case_id"], row["source_contact_ordinal"])
  )
  original_case_by_opaque = {
      _opaque_case_id(dict(case)): str(case["id"])
      for case in public_cases
  }
  replayed_rows: list[Mapping[str, Any]] = []
  query_ledger: list[dict[str, Any]] = []

  def append_replay_results(
      replay: BenchmarkV2MateSemanticReplay,
      selected_rows: Sequence[Mapping[str, Any]],
  ) -> None:
    rows_by_query: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in replay.rows:
      key = (
          original_case_by_opaque.get(
              str(row.get("case_id") or ""),
              str(row.get("case_id") or ""),
          ),
          int(row["source_contact"]["contact_index"]),
      )
      rows_by_query.setdefault(key, []).append(row)
    reasons: dict[tuple[str, int], str | None] = {}
    active_case_ids = [
        str(case["id"])
        for case in public_cases
        if str(case["id"]) in {
            str(row["case_id"]) for row in selected_rows
        }
    ]
    for case_id, case_ledger in zip(
        active_case_ids, replay.coverage_ledger["cases"],
    ):
      for contact in case_ledger["contacts"]:
        reasons[(case_id, int(contact["contact_index"]))] = contact.get(
            "reason_code"
        )
    for selector in selected_rows:
      key = (
          str(selector["case_id"]),
          int(selector["source_contact_ordinal"]),
      )
      query_rows = rows_by_query.get(key, [])
      reason = None
      if not query_rows:
        reason = reasons.get(key) or "source_contact_ordinal_missing"
      replayed_rows.extend(query_rows)
      query_ledger.append(
          {
              **selector,
              "status": "authorized" if query_rows else "rejected",
              "reason": reason,
              "directed_program_count": len(query_rows),
          }
      )

  try:
    aggregate = _replay_benchmark_v2_mate_semantics_core(
        public_cases=public_cases,
        private_supervision=private_supervision,
        face_map_lookup=face_map_lookup,
        archive_root=archive_root,
        mining_config=mining_config,
        family_split_manifest_sha256=family_split_manifest_sha256,
        source_split_sha256=source_split_sha256,
        _case_archive_root_resolver=_case_archive_root_resolver,
        _diagnostic_frame_sink=_diagnostic_frame_sink,
        _source_contact_ordinals_by_case=normalized,
    )
  except Exception:  # pylint: disable=broad-except
    aggregate = None
  if aggregate is not None:
    append_replay_results(aggregate, selector_rows)
  else:
    for selector in selector_rows:
      case_id = selector["case_id"]
      ordinal = selector["source_contact_ordinal"]
      try:
        replay = _replay_benchmark_v2_mate_semantics_core(
            public_cases=public_cases,
            private_supervision=private_supervision,
            face_map_lookup=face_map_lookup,
            archive_root=archive_root,
            mining_config=mining_config,
            family_split_manifest_sha256=family_split_manifest_sha256,
            source_split_sha256=source_split_sha256,
            _case_archive_root_resolver=_case_archive_root_resolver,
            _diagnostic_frame_sink=_diagnostic_frame_sink,
            _source_contact_ordinals_by_case={
                case_id: frozenset({ordinal})
            },
        )
      except Exception as error:  # pylint: disable=broad-except
        query_ledger.append(
            {
                **selector,
                "status": "rejected",
                "reason": (
                    "query_replay_error:"
                    + type(error).__name__
                    + ":"
                    + str(error)
                ),
                "directed_program_count": 0,
            }
        )
      else:
        append_replay_results(replay, [selector])
  program_ids = [str(row.get("program_id") or "") for row in replayed_rows]
  if (
      any(not program_id for program_id in program_ids)
      or len(program_ids) != len(set(program_ids))
  ):
    raise ValueError("query mate replay produced duplicate or empty program_id")
  return BenchmarkV2QueryMateSemanticReplay(
      rows=tuple(
          sorted(
              replayed_rows,
              key=lambda row: (
                  str(row.get("case_id") or ""),
                  int(row["source_contact"]["contact_index"]),
                  str(row["source_contact"]["direction"]),
                  str(row.get("program_id") or ""),
              ),
          )
      ),
      query_ledger=tuple(query_ledger),
      input_domain_sha256=_training_canonical_sha256(selector_rows),
      opaque_case_id_by_case_id={
          original: opaque
          for opaque, original in sorted(original_case_by_opaque.items())
      },
  )


def replay_benchmark_v2_mate_semantics_v2(
    *,
    public_cases: Sequence[Mapping[str, Any]],
    private_supervision: Any,
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    restored_archive_root: str | Path,
    mining_config: Mapping[str, Any],
    family_split_manifest_sha256: str,
    source_split_sha256: str,
    _diagnostic_frame_sink: list[dict[str, Any]] | None = None,
) -> BenchmarkV2MateSemanticReplay:
  """Replay semantics with strict per-case restored-archive isolation.

  ``restored_archive_root`` is a parent directory.  Each case is routed only to
  ``root / Path(source_receipt_binding.archive).stem``; STEP indices are cached
  lazily per archive and are never shared across archive roots.
  """

  from .formal_archive_inputs import resolve_case_archive_root

  parent = Path(restored_archive_root)

  def case_archive_root(source_case: Mapping[str, Any]) -> Path:
    _archive, resolved = resolve_case_archive_root(
        source_case,
        restored_archive_root=parent,
    )
    return resolved

  return replay_benchmark_v2_mate_semantics(
      public_cases=public_cases,
      private_supervision=private_supervision,
      face_map_lookup=face_map_lookup,
      archive_root=parent,
      mining_config=mining_config,
      family_split_manifest_sha256=family_split_manifest_sha256,
      source_split_sha256=source_split_sha256,
      _case_archive_root_resolver=case_archive_root,
      _diagnostic_frame_sink=_diagnostic_frame_sink,
  )


class _MateArtifactTransaction:
  """Stage a mate JSONL and manifest, publishing the manifest last."""

  def __init__(self, final_library_path: Path) -> None:
    self.final_library_path = Path(final_library_path)
    self.final_manifest_path = self.final_library_path.with_suffix(
        self.final_library_path.suffix + ".manifest.json"
    )
    self._directory = tempfile.TemporaryDirectory(
        dir=str(self.final_library_path.parent),
        prefix=f".{self.final_library_path.name}.",
    )
    self.staged_library_path = (
        Path(self._directory.name) / self.final_library_path.name
    )
    self.staged_manifest_path = self.staged_library_path.with_suffix(
        self.staged_library_path.suffix + ".manifest.json"
    )
    self._closed = False

  @contextmanager
  def open_library(self):
    if self._closed:
      raise RuntimeError("mate artifact transaction is already closed")
    try:
      with self.staged_library_path.open("w", encoding="utf-8") as stream:
        yield stream
    except BaseException:
      self.close()
      raise

  def commit(self, *, formal: bool) -> None:
    """Publish data then the validated manifest commit marker, with rollback."""

    if self._closed:
      raise RuntimeError("mate artifact transaction is already closed")
    if not self.staged_library_path.is_file():
      raise ValueError("staged mate library is missing")
    if formal and not self.staged_manifest_path.is_file():
      raise ValueError("staged formal mate manifest is missing")
    targets = [self.final_library_path]
    if formal:
      targets.append(self.final_manifest_path)
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
      for index, target in enumerate(targets):
        if target.exists():
          backup = Path(self._directory.name) / f"backup_{index:02d}"
          os.replace(target, backup)
          backups[target] = backup
      os.replace(self.staged_library_path, self.final_library_path)
      published.append(self.final_library_path)
      if formal:
        # The manifest is the commit marker and is always published last.
        os.replace(self.staged_manifest_path, self.final_manifest_path)
        published.append(self.final_manifest_path)
    except Exception:
      for target in reversed(published):
        if target.exists():
          target.unlink()
      for target, backup in reversed(tuple(backups.items())):
        if backup.exists():
          os.replace(backup, target)
      raise
    else:
      for backup in backups.values():
        backup.unlink(missing_ok=True)
    finally:
      self.close()

  def close(self) -> None:
    if not self._closed:
      self._directory.cleanup()
      self._closed = True

  def __del__(self) -> None:
    if hasattr(self, "_closed"):
      self.close()


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )
  parser.add_argument("--frame_randomization_seed", type=int, default=1001)
  parser.add_argument(
      "--benchmark_v2_translation_box_fraction",
      type=float,
      default=1.0,
  )
  parser.add_argument(
      "--cases_json",
      default="neurocad/paper_splits_candidate_main_deepseek_v1/train_oracle.json",
      help="Train/dev split JSON. Never mine mate programs from held-out test.",
  )
  parser.add_argument(
      "--source_split",
      choices=["train", "dev"],
      default="train",
      help="Formal benchmark_v2 libraries are accepted only from train.",
  )
  parser.add_argument(
      "--family_split_manifest",
      default=None,
      help="Required in benchmark_v2; binds cases_json to the frozen train family split.",
  )
  parser.add_argument(
      "--fusion-step-face-map-receipt",
      dest="fusion_step_face_map_receipt",
      default=None,
      help=(
          "Private Fusion-to-STEP face-map receipt. Required for formal "
          "benchmark_v2 mate mining."
      ),
  )
  parser.add_argument(
      "--private-source-split",
      dest="private_source_split",
      default=None,
      help="Custodian-only source/instance sidecar for the train split.",
  )
  parser.add_argument(
      "--private-evaluation-gold-split",
      dest="private_evaluation_gold_split",
      default=None,
      help="Evaluator-only source-positive contact sidecar for the train split.",
  )
  parser.add_argument("--dataset_root", default=None)
  parser.add_argument("--dataset_roots", nargs="*", default=[])
  parser.add_argument("--output_jsonl", default="neurocad/mate_program_library_train.jsonl")
  parser.add_argument("--summary_json", default=None)
  parser.add_argument("--interface_scorer_model", default=None)
  parser.add_argument("--max_cases", type=int, default=0)
  parser.add_argument("--max_candidates_per_part", type=int, default=64)
  parser.add_argument("--max_candidates_per_contact_face", type=int, default=4)
  parser.add_argument(
      "--canonicalize_parts",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Translate mined MCF frames by the same bbox-center canonicalization "
          "used at inference time."
      ),
  )
  parser.add_argument(
      "--bidirectional",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Write both A->B and B->A directed programs for each contact.",
  )
  parser.add_argument(
      "--filter_implausible_programs",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Drop mined residuals whose contact family is inconsistent with the "
          "interface roles or whose insert/seat residual axes are perpendicular."
      ),
  )
  return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  """Anchor all CLI paths at the checkout, independent of process CWD."""
  roots = resolve_dataset_roots(args.dataset_root, args.dataset_roots)
  args.dataset_root = str(roots[0])
  args.dataset_roots = [str(path) for path in roots[1:]]
  args.cases_json = str(resolve_path(args.cases_json))
  if args.family_split_manifest:
    args.family_split_manifest = str(resolve_path(args.family_split_manifest))
  if args.fusion_step_face_map_receipt:
    args.fusion_step_face_map_receipt = str(
        resolve_path(args.fusion_step_face_map_receipt)
    )
  if args.private_source_split:
    args.private_source_split = str(resolve_path(args.private_source_split))
  if args.private_evaluation_gold_split:
    args.private_evaluation_gold_split = str(
        resolve_path(args.private_evaluation_gold_split)
    )
  args.output_jsonl = str(resolve_path(args.output_jsonl))
  if args.summary_json:
    args.summary_json = str(resolve_path(args.summary_json))
  if args.interface_scorer_model:
    args.interface_scorer_model = str(resolve_path(args.interface_scorer_model))
  return args


def main() -> None:
  args = normalize_paths(build_parser().parse_args())
  protocol = str(args.input_protocol).strip().lower()
  if protocol == "benchmark_v2" and str(args.source_split) != "train":
    raise ValueError("formal benchmark_v2 mate libraries must be mined from train")
  if protocol == "benchmark_v2" and not args.fusion_step_face_map_receipt:
    raise SystemExit(
        "formal benchmark_v2 mate mining requires --fusion-step-face-map-receipt"
    )
  if protocol == "benchmark_v2" and not args.private_source_split:
    raise SystemExit(
        "formal benchmark_v2 mate mining requires --private-source-split"
    )
  if protocol == "benchmark_v2" and not args.private_evaluation_gold_split:
    raise SystemExit(
        "formal benchmark_v2 mate mining requires --private-evaluation-gold-split"
    )
  dataset_roots = _parse_dataset_roots(
      dataset_root=str(args.dataset_root),
      dataset_roots=list(args.dataset_roots or []),
  )
  if not dataset_roots:
    raise SystemExit("No dataset roots provided.")

  cases = _load_cases(Path(args.cases_json))
  family_split_manifest_sha256: str | None = None
  source_case_set_sha256: str | None = None
  source_case_tokens: dict[str, str] = {}
  face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]] = {}
  face_map_binding: Mapping[str, Any] | None = None
  private_supervision = None
  if protocol == "benchmark_v2":
    if not args.family_split_manifest:
      raise ValueError("benchmark_v2 mate mining requires --family_split_manifest")
    if int(args.max_cases) != 0:
      raise ValueError("formal benchmark_v2 mate mining forbids --max_cases sampling")
    if type(args.max_candidates_per_part) is not int or args.max_candidates_per_part != 0:
      raise ValueError(
          "formal benchmark_v2 mate mining requires --max_candidates_per_part 0"
      )
    if (
        type(args.max_candidates_per_contact_face) is not int
        or args.max_candidates_per_contact_face < 1
    ):
      raise ValueError(
          "formal benchmark_v2 mate mining requires "
          "--max_candidates_per_contact_face >= 1"
      )
    case_ids = [str(case.get("id") or "") for case in cases]
    if not case_ids or any(not case_id for case_id in case_ids):
      raise ValueError("benchmark_v2 mate mining requires nonempty case ids")
    if len(case_ids) != len(set(case_ids)):
      raise ValueError("benchmark_v2 mate mining rejects duplicate case ids")
    family_verification = validate_formal_family_split_manifest(
        Path(args.family_split_manifest),
        train_artifact_override=Path(args.cases_json),
    )
    family_split_manifest_sha256 = str(family_verification["manifest_sha256"])
    if case_ids != list(family_verification["train_case_ids"]):
      raise ValueError(
          "benchmark_v2 mate mining cases are not the authoritative train artifact"
      )
    source_case_set_sha256 = _training_canonical_sha256(sorted(case_ids))
    source_case_tokens = {
        case_id: case_token_sha256(
            family_manifest_sha256=family_split_manifest_sha256,
            case_id=case_id,
        )
        for case_id in case_ids
    }
    private_supervision = load_benchmark_v2_private_supervision(
        source_path=Path(args.private_source_split),
        gold_path=Path(args.private_evaluation_gold_split),
        family_split_manifest_path=Path(args.family_split_manifest),
        source_split="train",
        public_case_ids=case_ids,
    )
    validate_private_supervision_binding(
        private_supervision.producer_binding,
        require_certified_binary=False,
    )
    face_map_lookup, face_map_binding = _load_fusion_step_face_map_receipt(
        Path(args.fusion_step_face_map_receipt),
        family_split_manifest=Path(args.family_split_manifest),
        case_ids=case_ids,
    )
    face_map_binding = _validated_face_map_binding(
        face_map_binding,
        require_private_receipt=True,
    )
  if int(args.max_cases) > 0:
    cases = cases[: int(args.max_cases)]
  file_index = _index_step_files(dataset_roots)
  scorer = (
      InterfaceScorer.load(
          Path(str(args.interface_scorer_model)),
          expected_protocol=protocol,
      )
      if args.interface_scorer_model
      else None
  )

  source_split_sha256 = _file_sha256(Path(args.cases_json))
  mining_config = {
      "schema_version": MATE_MINING_CONFIG_SCHEMA,
      "model_input_protocol": protocol,
      "protocol_version": PROTOCOL_VERSION if protocol == "benchmark_v2" else "legacy",
      "source_split": str(args.source_split),
      "source_split_sha256": source_split_sha256,
      "family_split_manifest_sha256": family_split_manifest_sha256,
      "source_case_set_sha256": source_case_set_sha256,
      "max_cases": int(args.max_cases),
      "max_candidates_per_part": int(args.max_candidates_per_part),
      "max_candidates_per_contact_face": int(args.max_candidates_per_contact_face),
      "canonicalize_parts": bool(args.canonicalize_parts),
      "bidirectional": bool(args.bidirectional),
      "filter_implausible_programs": bool(args.filter_implausible_programs),
      "frame_randomization_seed": int(args.frame_randomization_seed),
      "benchmark_v2_translation_box_fraction": float(
          args.benchmark_v2_translation_box_fraction
      ),
      "interface_scorer_sha256": (
          _file_sha256(Path(args.interface_scorer_model))
          if args.interface_scorer_model
          else None
      ),
  }
  if protocol == "benchmark_v2":
    mining_config["fusion_step_face_map_binding"] = dict(face_map_binding or {})
    mining_config["private_supervision_binding"] = dict(
        private_supervision.producer_binding
    )
    mining_config["label_policy"] = "source_positive_only_unknown_excluded"
    mining_config["source_pose_contract"] = (
        "fusion_occurrence_tree_parent_chain_axes_columns_cm_to_mm.v1"
    )
  mining_config_sha256 = _canonical_sha256(mining_config)

  output_path = Path(args.output_jsonl)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  rows_written = 0
  skipped_cases = 0
  skipped_contacts = 0
  duplicate_rows = 0
  case_summaries: list[dict[str, Any]] = []
  coverage_ledger_cases: list[dict[str, Any]] = []
  seen_programs: set[str] = set()

  artifact_transaction = _MateArtifactTransaction(output_path)
  with artifact_transaction.open_library() as f:
    for case in cases:
      case_id = str(case.get("id") or "")
      private_source_case = (
          private_supervision.source_case(case_id)
          if protocol == "benchmark_v2"
          else None
      )
      private_gold_case = (
          private_supervision.gold_case(case_id)
          if protocol == "benchmark_v2"
          else None
      )
      verified_context = (
          _open_verified_benchmark_v2_case_inputs(
              private_source_case,
              dataset_roots=dataset_roots,
              file_index=file_index,
          )
          if protocol == "benchmark_v2"
          else nullcontext(None)
      )
      with verified_context as verified_inputs:
        case_summary, rows = _mine_case_programs(
            case=case,
            dataset_roots=dataset_roots,
            file_index=file_index,
            scorer=scorer,
            max_candidates_per_part=int(args.max_candidates_per_part),
            max_candidates_per_contact_face=int(args.max_candidates_per_contact_face),
            canonicalize_parts=bool(args.canonicalize_parts),
            bidirectional=bool(args.bidirectional),
            filter_implausible_programs=bool(args.filter_implausible_programs),
            input_protocol=str(args.input_protocol),
            frame_randomization_seed=int(args.frame_randomization_seed),
            benchmark_v2_translation_box_fraction=float(
                args.benchmark_v2_translation_box_fraction
            ),
            fusion_step_face_lookup=face_map_lookup,
            benchmark_v2_verified_inputs=verified_inputs,
            private_source_case=private_source_case,
            evaluation_gold_contacts=private_gold_case,
        )
      if case_summary.get("status") != "ok":
        skipped_cases += 1
        case_summaries.append(case_summary)
        if protocol == "benchmark_v2":
          raise ValueError(
              "formal benchmark_v2 mate mining refuses skipped case: "
              + str(case_summary.get("status") or "unknown")
          )
        continue
      skipped_contacts += int(case_summary.get("skipped_contacts", 0))
      written_for_case = 0
      for row in rows:
        key = str(row.get("program_id") or "")
        if key in seen_programs:
          if protocol == "benchmark_v2":
            raise ValueError("formal benchmark_v2 mate mining produced duplicate program_id")
          duplicate_rows += 1
          continue
        if protocol == "benchmark_v2":
          metadata = row.setdefault("metadata", {})
          metadata.update(
              {
                  "model_input_protocol": "benchmark_v2",
                  "source_split": "train",
                  "source_split_sha256": source_split_sha256,
                  "mining_config_sha256": mining_config_sha256,
                  "family_split_manifest_sha256": family_split_manifest_sha256,
                  "source_case_set_sha256": source_case_set_sha256,
                  "source_case_token_sha256": source_case_tokens[
                      str(case.get("id") or "")
                  ],
              }
          )
        seen_programs.add(key)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows_written += 1
        written_for_case += 1
      case_summary["rows_written"] = written_for_case
      case_summaries.append(case_summary)
      if protocol == "benchmark_v2":
        contact_ledger = list(case_summary.get("contact_ledger") or [])
        mined_contacts = sum(
            1 for item in contact_ledger
            if int(item.get("directed_program_count", 0)) > 0
        )
        coverage_ledger_cases.append(
            {
                "source_case_token_sha256": source_case_tokens[
                    str(case.get("id") or "")
                ],
                "status": "ok",
                "reason_code": None,
                "contacts_seen": len(contact_ledger),
                "contacts_mined": mined_contacts,
                "contacts_skipped": len(contact_ledger) - mined_contacts,
                "directed_program_count": written_for_case,
                "contacts": contact_ledger,
            }
        )

  summary = {
      "cases_seen": len(cases),
      "cases_with_programs": sum(
          1 for item in case_summaries if int(item.get("rows_written", 0)) > 0
      ),
      "skipped_cases": skipped_cases,
      "skipped_contacts": skipped_contacts,
      "duplicate_rows": duplicate_rows,
      "rows_written": rows_written,
      "bidirectional": bool(args.bidirectional),
      "filter_implausible_programs": bool(args.filter_implausible_programs),
      "canonicalize_parts": bool(args.canonicalize_parts),
      "input_protocol": str(args.input_protocol),
      "frame_randomization_seed": int(args.frame_randomization_seed),
      "output_jsonl": str(output_path.resolve()),
      "source_split": str(args.source_split),
      "source_split_sha256": source_split_sha256,
      "mining_config": mining_config,
      "mining_config_sha256": mining_config_sha256,
  }
  if protocol == "benchmark_v2":
    contacts_seen = sum(int(item["contacts_seen"]) for item in coverage_ledger_cases)
    contacts_mined = sum(int(item["contacts_mined"]) for item in coverage_ledger_cases)
    contacts_skipped = sum(int(item["contacts_skipped"]) for item in coverage_ledger_cases)
    cases_with_programs = sum(
        1 for item in case_summaries if int(item.get("rows_written", 0)) > 0
    )
    coverage = {
        "expected_case_count": len(cases),
        "processed_case_count": len(case_summaries),
        "successful_case_count": sum(
            1 for item in case_summaries if item.get("status") == "ok"
        ),
        "cases_with_programs": cases_with_programs,
        "skipped_case_count": skipped_cases,
        "contacts_seen": contacts_seen,
        "contacts_skipped": contacts_skipped,
        "contacts_mined": contacts_mined,
        "directed_program_count": rows_written,
    }
    coverage_ledger = {
        "schema_version": MATE_COVERAGE_LEDGER_SCHEMA,
        "cases": coverage_ledger_cases,
    }
    try:
      manifest = _finalize_v2_library_manifest(
          library_path=artifact_transaction.staged_library_path,
          row_count=rows_written,
          source_split=str(args.source_split),
          source_split_sha256=source_split_sha256,
          mining_config_sha256=mining_config_sha256,
          family_split_manifest_sha256=str(family_split_manifest_sha256),
          source_case_set_sha256=str(source_case_set_sha256),
          mining_config=mining_config,
          coverage_ledger=coverage_ledger,
          coverage=coverage,
      )
    except BaseException:
      artifact_transaction.close()
      raise
    summary["library_manifest"] = str(
        output_path.with_suffix(output_path.suffix + ".manifest.json").resolve()
    )
    summary["library_manifest_sha256"] = manifest["manifest_sha256"]
    summary["family_split_manifest_sha256"] = family_split_manifest_sha256
    summary["source_case_set_sha256"] = source_case_set_sha256
    summary["coverage"] = coverage
  artifact_transaction.commit(formal=protocol == "benchmark_v2")
  summary_path = (
      Path(args.summary_json)
      if args.summary_json
      else output_path.with_suffix(".summary.json")
  )
  summary_path.write_text(
      json.dumps({"summary": summary, "cases": case_summaries}, indent=2),
      encoding="utf-8",
  )
  print(json.dumps(summary, indent=2))


def _mine_case_programs(
    *,
    case: dict[str, Any],
    dataset_roots: list[Path],
    file_index: dict[str, list[Path]],
    scorer: Optional[InterfaceScorer],
    max_candidates_per_part: int,
    max_candidates_per_contact_face: int,
    canonicalize_parts: bool,
    bidirectional: bool,
    filter_implausible_programs: bool,
    input_protocol: str = "legacy",
    frame_randomization_seed: int = 1001,
    benchmark_v2_translation_box_fraction: float = 1.0,
    fusion_step_face_lookup: Mapping[
        tuple[str, str, int], Mapping[str, Any]
    ] | None = None,
    benchmark_v2_verified_inputs: Any = None,
    private_source_case: Mapping[str, Any] | None = None,
    evaluation_gold_contacts: Mapping[str, Any] | None = None,
    diagnostic_frame_sink: list[dict[str, Any]] | None = None,
    source_contact_ordinals: frozenset[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  case_id = str(case.get("id") or "")
  protocol = str(input_protocol or "legacy").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("input_protocol must be 'legacy' or 'benchmark_v2'")
  if protocol == "benchmark_v2" and (
      not isinstance(private_source_case, Mapping)
      or not isinstance(evaluation_gold_contacts, Mapping)
  ):
    raise ValueError(
        "benchmark_v2 mate mining requires explicit private source/contact sidecars"
    )
  source_case = private_source_case if protocol == "benchmark_v2" else case
  verified_assembly_data: Mapping[str, Any] | None = None
  verified_part_specs: Sequence[BenchmarkV2PartSpec] | None = None
  live_input_verification = None
  if benchmark_v2_verified_inputs is not None:
    try:
      (
          verified_assembly_data,
          verified_part_specs,
          live_input_verification,
      ) = benchmark_v2_verified_inputs
    except (TypeError, ValueError) as error:
      raise ValueError("benchmark_v2 verified case inputs are malformed") from error
    if protocol != "benchmark_v2" or not isinstance(
        verified_assembly_data, Mapping
    ):
      raise ValueError("verified live inputs are valid only for benchmark_v2")
  assembly_json = _assembly_json_for_case(dict(source_case), dataset_roots)
  if assembly_json is None or not assembly_json.exists():
    return {
        "case_id": case_id,
        "assembly_dir": case.get("assembly_dir"),
        "status": "missing_assembly_json",
        "rows_written": 0,
      }, []

  prepared: PreparedBenchmarkV2Assembly | None = None
  try:
    if protocol == "benchmark_v2":
      part_records, prepared = _load_benchmark_v2_case_part_interfaces(
          case=dict(source_case),
          dataset_roots=dataset_roots,
          file_index=file_index,
          scorer=scorer,
          max_candidates_per_part=max_candidates_per_part,
          frame_randomization_seed=frame_randomization_seed,
          translation_box_fraction=benchmark_v2_translation_box_fraction,
          part_specs=verified_part_specs,
      )
    else:
      part_records = _load_case_part_interfaces(
          case=case,
          dataset_roots=dataset_roots,
          file_index=file_index,
          assembly_json=assembly_json,
          scorer=scorer,
          max_candidates_per_part=max_candidates_per_part,
      )
  except Exception as exc:  # pylint: disable=broad-except
    return {
        "case_id": case_id,
        "assembly_dir": case.get("assembly_dir"),
        "status": f"part_interface_error:{type(exc).__name__}:{exc}",
        "rows_written": 0,
    }, []

  part_records = {
      key: value for key, value in part_records.items() if not value.get("error")
  }
  if canonicalize_parts and protocol == "legacy":
    _canonicalize_part_record_frames(part_records)
  if len(part_records) < 2:
    return {
        "case_id": case_id,
        "assembly_dir": case.get("assembly_dir"),
        "status": "too_few_parts_with_interfaces",
        "rows_written": 0,
    }, []

  if prepared is None:
    body_to_part_name = _body_to_part_name(case)
  else:
    body_to_part_name = {
        prepared.private_identity_context.opaque_body_for(part_alias):
        prepared.private_identity_context.opaque_part_for(part_alias)
        for part_alias in (source_case.get("selected_part_names") or [])
        if str(part_alias).strip()
    }
  selected_bodies = set(part_records)
  if prepared is not None:
    if verified_assembly_data is None:
      raise ValueError("benchmark_v2 mate mining requires receipt-captured assembly JSON")
    source_world_transforms = _benchmark_v2_source_world_transforms(
        assembly_data=verified_assembly_data,
        private_source_case=source_case,
    )
    contacts = evaluation_gold_contacts.get("contacts")
    if (
        evaluation_gold_contacts.get("schema_version")
        != "benchmark_v2_evaluation_gold_contacts.v2"
        or not isinstance(contacts, list)
    ):
      raise ValueError("benchmark_v2 mate mining contact gold contract is invalid")
    source_identity_by_part = {
        str(part): (str(asset), str(body))
        for part, asset, body in zip(
            source_case.get("selected_part_names") or [],
            source_case.get("selected_geometry_assets") or [],
            source_case.get("selected_body_uuids") or [],
        )
    }
  else:
    source_world_transforms = {}
    source_identity_by_part = {}
    contacts = _load_contacts(assembly_json)
  rows: list[dict[str, Any]] = []
  skipped_contacts = 0
  contact_ledger: list[dict[str, Any]] = []

  def record_contact(
      contact_index: int,
      *,
      status: str,
      reason_code: str | None,
      directed_program_count: int = 0,
  ) -> None:
    contact_ledger.append(
        {
            "contact_index": int(contact_index),
            "status": str(status),
            "reason_code": reason_code,
            "directed_program_count": int(directed_program_count),
        }
    )

  for contact_offset, contact in enumerate(contacts):
    if source_contact_ordinals is not None:
      selected_ordinal = (
          contact.get("source_contact_ordinal")
          if isinstance(contact, Mapping)
          else None
      )
      if selected_ordinal not in source_contact_ordinals:
        continue
    if not isinstance(contact, Mapping):
      skipped_contacts += 1
      record_contact(
          contact_offset,
          status="skipped",
          reason_code="invalid_contact_entities",
      )
      continue
    raw_part_a: str | None = None
    raw_part_b: str | None = None
    if prepared is not None:
      contact_index = contact.get("source_contact_ordinal")
      entity_a = contact.get("endpoint_a")
      entity_b = contact.get("endpoint_b")
      if (
          isinstance(contact_index, bool)
          or not isinstance(contact_index, int)
          or contact_index < 0
          or not isinstance(entity_a, Mapping)
          or not isinstance(entity_b, Mapping)
      ):
        raise ValueError("benchmark_v2 private gold contact row is malformed")
    else:
      contact_index = contact_offset
      entity_a = contact.get("entity_one")
      entity_b = contact.get("entity_two")
      if not isinstance(entity_a, dict) or not isinstance(entity_b, dict):
        skipped_contacts += 1
        record_contact(
            contact_index,
            status="skipped",
            reason_code="invalid_contact_entities",
        )
        continue
    mapped_face_indices_a: tuple[int, ...] | None = None
    mapped_face_indices_b: tuple[int, ...] | None = None
    if prepared is not None:
      if not isinstance(fusion_step_face_lookup, Mapping):
        raise ValueError(
            "benchmark_v2 mate mining requires an explicit face-map lookup"
        )
      mapped_a = _mapped_fusion_contact_endpoint(
          prepared,
          entity_a,
          case_id=case_id,
          source_identity_by_part=source_identity_by_part,
          fusion_step_face_lookup=fusion_step_face_lookup,
      )
      mapped_b = _mapped_fusion_contact_endpoint(
          prepared,
          entity_b,
          case_id=case_id,
          source_identity_by_part=source_identity_by_part,
          fusion_step_face_lookup=fusion_step_face_lookup,
      )
      raw_part_a = mapped_a.raw_part
      raw_part_b = mapped_b.raw_part
      if raw_part_a == raw_part_b:
        raise ValueError("benchmark_v2 gold contact references one physical instance twice")
      mapped_face_indices_a = mapped_a.randomized_face_indices
      mapped_face_indices_b = mapped_b.randomized_face_indices
      entity_a = {
          "body": mapped_a.opaque_body,
          "index": mapped_face_indices_a[0],
          "surface_type": entity_a.get("surface_type"),
      }
      entity_b = {
          "body": mapped_b.opaque_body,
          "index": mapped_face_indices_b[0],
          "surface_type": entity_b.get("surface_type"),
      }
    body_a = str(entity_a.get("body") or "")
    body_b = str(entity_b.get("body") or "")
    if body_a not in selected_bodies or body_b not in selected_bodies or body_a == body_b:
      skipped_contacts += 1
      record_contact(
          contact_index,
          status="skipped",
          reason_code="outside_selected_or_same_body",
      )
      continue
    face_a = entity_a.get("index")
    face_b = entity_b.get("index")
    if not isinstance(face_a, int) or not isinstance(face_b, int):
      skipped_contacts += 1
      record_contact(
          contact_index,
          status="skipped",
          reason_code="invalid_face_index",
      )
      continue
    candidates_a = _candidates_for_contact_faces(
        part_records[body_a].get("candidates", []),
        (
            mapped_face_indices_a
            if mapped_face_indices_a is not None
            else (int(face_a),)
        ),
        max_count=max_candidates_per_contact_face,
    )
    candidates_b = _candidates_for_contact_faces(
        part_records[body_b].get("candidates", []),
        (
            mapped_face_indices_b
            if mapped_face_indices_b is not None
            else (int(face_b),)
        ),
        max_count=max_candidates_per_contact_face,
    )
    if not candidates_a or not candidates_b:
      skipped_contacts += 1
      record_contact(
          contact_index,
          status="skipped",
          reason_code="no_contact_face_candidates",
      )
      continue
    part_a_name = body_to_part_name.get(body_a, part_records[body_a].get("part_name", body_a))
    part_b_name = body_to_part_name.get(body_b, part_records[body_b].get("part_name", body_b))
    rows_before_contact = len(rows)
    for candidate_a in candidates_a:
      for candidate_b in candidates_b:
        child_to_parent = _benchmark_v2_child_to_parent_transform(
            prepared=prepared,
            child_part_key=raw_part_b or body_b,
            parent_part_key=raw_part_a or body_a,
            source_world_transforms=source_world_transforms,
        )
        row = _program_row(
            case=case,
            contact_index=contact_index,
            entity_a=entity_a,
            entity_b=entity_b,
            part_a_name=str(part_a_name),
            part_b_name=str(part_b_name),
            body_a=body_a,
            body_b=body_b,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            direction="entity_one_to_entity_two",
            filter_implausible_programs=filter_implausible_programs,
            child_to_parent_transform=child_to_parent,
            input_protocol=protocol,
        )
        if row is not None:
          rows.append(row)
          if diagnostic_frame_sink is not None:
            diagnostic_frame_sink.append(
                _semantic_frame_diagnostic_row(
                    row=row,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    prepared=prepared,
                    raw_part_a=raw_part_a or body_a,
                    raw_part_b=raw_part_b or body_b,
                    source_world_transforms=source_world_transforms,
                )
            )
        if bidirectional:
          reverse_child_to_parent = _benchmark_v2_child_to_parent_transform(
              prepared=prepared,
              child_part_key=raw_part_a or body_a,
              parent_part_key=raw_part_b or body_b,
              source_world_transforms=source_world_transforms,
          )
          row_rev = _program_row(
              case=case,
              contact_index=contact_index,
              entity_a=entity_b,
              entity_b=entity_a,
              part_a_name=str(part_b_name),
              part_b_name=str(part_a_name),
              body_a=body_b,
              body_b=body_a,
              candidate_a=candidate_b,
              candidate_b=candidate_a,
              direction="entity_two_to_entity_one",
              filter_implausible_programs=filter_implausible_programs,
              child_to_parent_transform=reverse_child_to_parent,
              input_protocol=protocol,
          )
          if row_rev is not None:
            rows.append(row_rev)
            if diagnostic_frame_sink is not None:
              diagnostic_frame_sink.append(
                  _semantic_frame_diagnostic_row(
                      row=row_rev,
                      candidate_a=candidate_b,
                      candidate_b=candidate_a,
                      prepared=prepared,
                      raw_part_a=raw_part_b or body_b,
                      raw_part_b=raw_part_a or body_a,
                      source_world_transforms=source_world_transforms,
                  )
              )
    directed_program_count = len(rows) - rows_before_contact
    if directed_program_count > 0:
      record_contact(
          contact_index,
          status="mined",
          reason_code=None,
          directed_program_count=directed_program_count,
      )
    else:
      skipped_contacts += 1
      record_contact(
          contact_index,
          status="skipped",
          reason_code="all_candidate_programs_filtered_or_invalid",
      )

  if live_input_verification is not None:
    _reverify_benchmark_v2_case_inputs(live_input_verification)
  return {
      "case_id": case_id,
      "assembly_dir": case.get("assembly_dir"),
      "status": "ok",
      "contacts": len(contacts),
      "skipped_contacts": skipped_contacts,
      "candidate_rows": len(rows),
      "rows_written": 0,
      "contact_ledger": contact_ledger,
      "input_protocol": protocol,
      "protocol_audit": (
          prepared.audit_provenance() if prepared is not None else {}
      ),
  }, rows


def _load_benchmark_v2_case_part_interfaces(
    *,
    case: dict[str, Any],
    dataset_roots: list[Path],
    file_index: dict[str, list[Path]],
    scorer: Optional[InterfaceScorer],
    max_candidates_per_part: int,
    frame_randomization_seed: int,
    translation_box_fraction: float,
    part_specs: Sequence[BenchmarkV2PartSpec] | None = None,
) -> tuple[dict[str, dict[str, Any]], PreparedBenchmarkV2Assembly]:
  """Prepare every mining part through the same single-load v2 boundary."""

  parts = case.get("parts")
  if not isinstance(parts, list) or not parts:
    raise ValueError("Case missing non-empty list field: parts")
  name_hints = case.get("selected_part_names")
  body_hints = case.get("selected_body_uuids")
  geometry_hints = case.get("selected_geometry_assets")
  if not isinstance(name_hints, list) or len(name_hints) != len(parts):
    raise ValueError("benchmark_v2 mining requires one opaque selected_part_name per part")
  if not isinstance(body_hints, list) or len(body_hints) != len(parts):
    raise ValueError("benchmark_v2 mining requires one opaque selected_body_uuid per part")
  if not isinstance(geometry_hints, list) or len(geometry_hints) != len(parts):
    raise ValueError("benchmark_v2 mining requires one geometry asset per part")
  assembly_dir = case.get("assembly_dir")
  if not isinstance(assembly_dir, str):
    assembly_dir = None
  root_hint = case.get("dataset_root_hint")
  if not isinstance(root_hint, str):
    root_hint = None

  if part_specs is None:
    specs: list[BenchmarkV2PartSpec] = []
    for index, token in enumerate(parts):
      step_path = _resolve_part_path(
          part_token=str(token),
          dataset_roots=dataset_roots,
          assembly_dir=assembly_dir,
          dataset_root_hint=root_hint,
          file_index=file_index,
      )
      body_uuid = str(body_hints[index]).strip()
      part_alias = str(name_hints[index]).strip()
      specs.append(
          BenchmarkV2PartSpec(
              part_key=part_alias,
              part_name=part_alias,
              body_uuid=body_uuid,
              step_path=step_path,
              geometry_asset=str(geometry_hints[index]).strip(),
          )
      )
  else:
    specs = list(part_specs)
    expected = [
        (
            str(name_hints[index]).strip(),
            str(name_hints[index]).strip(),
            str(body_hints[index]).strip(),
            str(geometry_hints[index]).strip(),
        )
        for index in range(len(parts))
    ]
    actual = [
        (spec.part_key, spec.part_name, spec.body_uuid, spec.geometry_asset)
        for spec in specs
    ]
    if actual != expected:
      raise ValueError("verified STEP specs do not match the benchmark_v2 case")

  prepared = prepare_benchmark_v2_assembly(
      specs,
      assembly_nonce=str(case.get("case_id") or case.get("id") or ""),
      seed=int(frame_randomization_seed),
      extraction_config=SocketExtractionConfig(
          enable_proxy_sockets=False,
          frame_protocol="benchmark_v2",
      ),
      translation_box_fraction=float(translation_box_fraction),
      max_candidates_per_part=int(max_candidates_per_part),
      interface_scorer=scorer,
  )
  records = {
      prepared.private_identity_context.opaque_body_for(spec.part_key): {
          "part_name": prepared.private_identity_context.opaque_part_for(
              spec.part_key
          ),
          "body_uuid": prepared.private_identity_context.opaque_body_for(
              spec.part_key
          ),
          "step_path": str(spec.step_path.resolve()),
          "candidates": list(prepared.part_for(spec.part_key).candidates),
      }
      for spec in specs
  }
  for spec in specs:
    part = prepared.part_for(spec.part_key)
    for candidate, view in zip(part.candidates, part.model_views):
      candidate.metadata.update(
          {
              "model_input_protocol": "benchmark_v2",
              "benchmark_v2_model_view": view.model_view,
              "benchmark_v2_model_view_sha256": view.sha256,
          }
      )
  return records, prepared


def _fusion_vector3(value: Any, *, label: str) -> np.ndarray:
  """Parse one Fusion JSON vector without accepting non-finite values."""

  if isinstance(value, Mapping):
    # Fusion's serialized Point3D/Vector3D objects carry descriptive metadata
    # (for example ``type`` and ``length``) in addition to x/y/z.  Those fields
    # are not pose inputs; require the coordinates and ignore the metadata just
    # as the geometry authority's transform replay does.
    if not all(axis in value for axis in ("x", "y", "z")):
      raise ValueError(f"{label} must contain x/y/z")
    raw = [value["x"], value["y"], value["z"]]
  elif isinstance(value, (list, tuple)) and len(value) == 3:
    raw = list(value)
  else:
    raise ValueError(f"{label} must be a three-vector")
  try:
    result = np.asarray(raw, dtype=float).reshape(3)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{label} must be numeric") from error
  if not np.isfinite(result).all():
    raise ValueError(f"{label} values must be finite")
  return result


def _fusion_occurrence_transform_mm(value: Any, *, occurrence_uuid: str) -> Transform:
  """Convert one Fusion local-to-parent coordinate system from cm to mm."""

  if not isinstance(value, Mapping) or set(value) != {
      "origin",
      "x_axis",
      "y_axis",
      "z_axis",
  }:
    raise ValueError(
        f"occurrence {occurrence_uuid!r} transform has an invalid field set"
    )
  origin_cm = _fusion_vector3(
      value.get("origin"),
      label=f"occurrence {occurrence_uuid} transform origin",
  )
  axes = [
      _fusion_vector3(
          value.get(name),
          label=f"occurrence {occurrence_uuid} transform {name}",
      )
      for name in ("x_axis", "y_axis", "z_axis")
  ]
  rotation = np.column_stack(axes)
  try:
    return Transform(rotation=rotation, translation=origin_cm * 10.0)
  except ValueError as error:
    raise ValueError(
        f"occurrence {occurrence_uuid!r} rotation is not finite SO(3)"
    ) from error


def _benchmark_v2_source_world_transforms(
    *,
    assembly_data: Mapping[str, Any],
    private_source_case: Mapping[str, Any],
) -> dict[str, Transform]:
  """Resolve every selected body instance to a body-local-to-root pose.

  Fusion occurrence transforms are local-to-parent. Axes form rotation columns,
  origins are centimetres, and the rooted composition is ``W_parent @ L_child``.
  No missing/invalid transform is allowed to fall back to identity.
  """

  if not isinstance(assembly_data, Mapping):
    raise ValueError("benchmark_v2 assembly pose source must be an object")
  root = assembly_data.get("root")
  occurrences = assembly_data.get("occurrences")
  tree = assembly_data.get("tree")
  if (
      not isinstance(root, Mapping)
      or not isinstance(occurrences, Mapping)
      or not isinstance(tree, Mapping)
      or set(tree) != {"root"}
      or not isinstance(tree.get("root"), Mapping)
  ):
    raise ValueError("benchmark_v2 assembly occurrence tree is malformed")
  root_component = str(root.get("component") or "").strip()
  # Fusion omits ``root.bodies`` when the root component contains no direct
  # bodies.  The occurrence table remains authoritative for selected child
  # instances, so treat that omitted optional field exactly like an empty map.
  root_bodies = root.get("bodies", {})
  if not root_component or not isinstance(root_bodies, Mapping):
    raise ValueError("benchmark_v2 assembly root identity is malformed")

  raw_occurrences: dict[str, Mapping[str, Any]] = {}
  for raw_uuid, raw_occurrence in occurrences.items():
    occurrence_uuid = str(raw_uuid).strip()
    if (
        not occurrence_uuid
        or occurrence_uuid in raw_occurrences
        or not isinstance(raw_occurrence, Mapping)
    ):
      raise ValueError("benchmark_v2 occurrence table is malformed")
    raw_occurrences[occurrence_uuid] = raw_occurrence

  world_by_occurrence: dict[str, Transform] = {}
  path_by_occurrence: dict[str, tuple[str, ...]] = {}
  bodies_by_occurrence: dict[str, frozenset[str]] = {}

  def walk(
      subtree: Mapping[str, Any],
      *,
      parent_world: Transform,
      parent_path: tuple[str, ...],
  ) -> None:
    for raw_uuid, raw_child_tree in subtree.items():
      occurrence_uuid = str(raw_uuid).strip()
      occurrence = raw_occurrences.get(occurrence_uuid)
      if occurrence is None:
        raise ValueError("benchmark_v2 tree references an unknown occurrence")
      if occurrence_uuid in world_by_occurrence:
        raise ValueError("benchmark_v2 occurrence tree is duplicated or cyclic")
      if not isinstance(raw_child_tree, Mapping):
        raise ValueError("benchmark_v2 occurrence subtree is malformed")
      local = _fusion_occurrence_transform_mm(
          occurrence.get("transform"),
          occurrence_uuid=occurrence_uuid,
      )
      world = compose_se3(parent_world, local)
      # Fusion omits ``bodies`` for an occurrence whose component owns no
      # direct solids.  This is an empty instance set, not malformed geometry.
      raw_bodies = occurrence.get("bodies", {})
      if not isinstance(raw_bodies, Mapping) or any(
          not isinstance(reference, Mapping) for reference in raw_bodies.values()
      ):
        raise ValueError("benchmark_v2 occurrence body references are malformed")
      path = (*parent_path, occurrence_uuid)
      world_by_occurrence[occurrence_uuid] = world
      path_by_occurrence[occurrence_uuid] = path
      bodies_by_occurrence[occurrence_uuid] = frozenset(
          str(body_uuid) for body_uuid in raw_bodies
      )
      walk(raw_child_tree, parent_world=world, parent_path=path)

  walk(tree["root"], parent_world=Transform.identity(), parent_path=())
  if set(world_by_occurrence) != set(raw_occurrences):
    raise ValueError("benchmark_v2 occurrence table contains unreachable entries")

  names = private_source_case.get("selected_part_names")
  bodies = private_source_case.get("selected_body_uuids")
  receipt = private_source_case.get("source_receipt_binding")
  instances = receipt.get("instances") if isinstance(receipt, Mapping) else None
  if (
      not isinstance(names, list)
      or not isinstance(bodies, list)
      or len(names) != len(bodies)
      or not isinstance(instances, list)
      or len(instances) != len(names)
  ):
    raise ValueError("benchmark_v2 private source instance binding is malformed")
  expected_body_by_part = {
      str(part): str(body) for part, body in zip(names, bodies)
  }
  if (
      len(expected_body_by_part) != len(names)
      or any(not part or not body for part, body in expected_body_by_part.items())
  ):
    raise ValueError("benchmark_v2 private source part aliases are invalid")

  result: dict[str, Transform] = {}
  seen_source_instances: set[str] = set()
  for raw_instance in instances:
    if not isinstance(raw_instance, Mapping):
      raise ValueError("benchmark_v2 private source instance is malformed")
    part = str(raw_instance.get("part") or "").strip()
    source_key = raw_instance.get("source_instance_key")
    occurrence_path = raw_instance.get("occurrence_path")
    if (
        part not in expected_body_by_part
        or part in result
        or not isinstance(source_key, Mapping)
        or not isinstance(occurrence_path, list)
    ):
      raise ValueError("benchmark_v2 private source instance identity is malformed")
    body_uuid = str(source_key.get("body_uuid") or "").strip()
    if body_uuid != expected_body_by_part[part]:
      raise ValueError("benchmark_v2 private source instance/body binding differs")
    canonical_source_key = json.dumps(
        dict(source_key),
        sort_keys=True,
        separators=(",", ":"),
    )
    if canonical_source_key in seen_source_instances:
      raise ValueError("benchmark_v2 private source instance is duplicated")
    seen_source_instances.add(canonical_source_key)
    kind = source_key.get("kind")
    if kind == "root":
      if set(source_key) != {"kind", "body_uuid", "root_component_uuid"}:
        raise ValueError("benchmark_v2 root instance key is malformed")
      if (
          str(source_key.get("root_component_uuid") or "") != root_component
          or occurrence_path
          or body_uuid not in {str(value) for value in root_bodies}
      ):
        raise ValueError("benchmark_v2 root instance binding is inconsistent")
      result[part] = Transform.identity()
      continue
    if kind != "occurrence" or set(source_key) != {
        "kind",
        "body_uuid",
        "occurrence_uuid",
    }:
      raise ValueError("benchmark_v2 occurrence instance key is malformed")
    occurrence_uuid = str(source_key.get("occurrence_uuid") or "").strip()
    if (
        occurrence_uuid not in world_by_occurrence
        or tuple(occurrence_path) != path_by_occurrence[occurrence_uuid]
    ):
      raise ValueError("benchmark_v2 occurrence instance path is inconsistent")
    if body_uuid not in bodies_by_occurrence[occurrence_uuid]:
      raise ValueError("benchmark_v2 occurrence instance body is not referenced")
    result[part] = world_by_occurrence[occurrence_uuid]
  if set(result) != set(expected_body_by_part):
    raise ValueError("benchmark_v2 source world transform coverage is incomplete")
  return {part: result[part] for part in sorted(result)}


def _benchmark_v2_child_to_parent_transform(
    *,
    prepared: PreparedBenchmarkV2Assembly | None,
    child_part_key: str,
    parent_part_key: str,
    source_world_transforms: Mapping[str, Transform] | None = None,
) -> Optional[Transform]:
  if prepared is None:
    return None
  if not isinstance(source_world_transforms, Mapping):
    raise ValueError("benchmark_v2 mate labels require source world transforms")
  try:
    child_world = source_world_transforms[str(child_part_key)]
    parent_world = source_world_transforms[str(parent_part_key)]
  except KeyError as error:
    raise ValueError("benchmark_v2 mate part lacks a source world transform") from error
  if not isinstance(child_world, Transform) or not isinstance(parent_world, Transform):
    raise ValueError("benchmark_v2 source world transform has an invalid type")
  raw_child_to_parent = compose_se3(
      inverse_se3(parent_world),
      child_world,
  )
  return prepared.transform_pair_supervision(
      source_part_key=child_part_key,
      target_part_key=parent_part_key,
      ground_truth=raw_child_to_parent,
  ).relative_transform


def _program_row(
    *,
    case: dict[str, Any],
    contact_index: int,
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
    part_a_name: str,
    part_b_name: str,
    body_a: str,
    body_b: str,
    candidate_a: CandidateInterface,
    candidate_b: CandidateInterface,
    direction: str,
    filter_implausible_programs: bool,
    child_to_parent_transform: Optional[Transform] = None,
    input_protocol: str = "legacy",
) -> Optional[dict[str, Any]]:
  protocol = str(input_protocol or "legacy").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("input_protocol must be 'legacy' or 'benchmark_v2'")
  try:
    residual = _program_residual(
        parent_frame=candidate_a.local_frame,
        child_frame=candidate_b.local_frame,
        child_to_parent_transform=child_to_parent_transform,
    )
  except Exception:
    return None

  row_a = candidate_a.to_dict()
  row_b = candidate_b.to_dict()
  interface_a = _compact_interface(candidate_a, input_protocol=protocol)
  interface_b = _compact_interface(candidate_b, input_protocol=protocol)
  semantic_a = _semantic_interface_row(interface_a) if protocol == "benchmark_v2" else row_a
  semantic_b = _semantic_interface_row(interface_b) if protocol == "benchmark_v2" else row_b
  relation_hint = _relation_hint_from_roles(semantic_a, semantic_b)
  contact_type = _contact_type_from_pair(
      part_a_name="" if protocol == "benchmark_v2" else part_a_name,
      part_b_name="" if protocol == "benchmark_v2" else part_b_name,
      row_a=semantic_a,
      row_b=semantic_b,
  )
  features = pair_features_from_rows(
      interface_a if protocol == "benchmark_v2" else row_a,
      interface_b if protocol == "benchmark_v2" else row_b,
      relation_hint=relation_hint,
      protocol=protocol,
  )
  features["heuristic_prior"] = heuristic_pair_prior(features)
  residual_translation = np.asarray(residual.translation, dtype=float).reshape(3)
  residual_rotation = np.asarray(residual.rotation, dtype=float).reshape(3, 3)
  if protocol == "legacy":
    features["residual_translation_norm"] = float(np.linalg.norm(residual_translation))
    features["residual_rotation_angle_deg"] = rotation_angle_degrees(residual_rotation)
    features["residual_axis_z_abs"] = abs(float(residual_rotation[2, 2]))
  if filter_implausible_programs and not _program_is_plausible(
      row_a=semantic_a,
      row_b=semantic_b,
      contact_type=contact_type,
      residual_rotation=residual_rotation,
  ):
    return None
  source_contact = {
      "contact_index": int(contact_index),
      "direction": direction,
      # Formal rows must not publish Fusion keys, private OCC indices, or the
      # receipt-derived randomized correspondence.
      "face_index_a": None if protocol == "benchmark_v2" else entity_a.get("index"),
      "face_index_b": None if protocol == "benchmark_v2" else entity_b.get("index"),
      "surface_type_a": entity_a.get("surface_type"),
      "surface_type_b": entity_b.get("surface_type"),
  }
  payload_for_id = {
      "case_id": case.get("id"),
      "body_a": body_a,
      "body_b": body_b,
      "iface_a": candidate_a.interface_id,
      "iface_b": candidate_b.interface_id,
      "contact_index": contact_index,
      "direction": direction,
      "rt": [round(float(x), 5) for x in residual_translation],
      "rr": [round(float(x), 5) for x in residual_rotation.reshape(-1)],
  }
  program_id = hashlib.sha1(
      json.dumps(payload_for_id, sort_keys=True).encode("utf-8")
  ).hexdigest()[:16]
  return {
      "model_input_protocol": protocol,
      "program_id": program_id,
      "case_id": (
          _opaque_case_id(case) if protocol == "benchmark_v2" else str(case.get("id") or "")
      ),
      "assembly_dir": "" if protocol == "benchmark_v2" else str(case.get("assembly_dir") or ""),
      "part_a": part_a_name,
      "part_b": part_b_name,
      "body_uuid_a": body_a,
      "body_uuid_b": body_b,
      "interface_a": interface_a,
      "interface_b": interface_b,
      "relation_hint": relation_hint,
      "contact_type": contact_type,
      "residual_rotation": residual_rotation.tolist(),
      "residual_translation": residual_translation.tolist(),
      "source_contact": source_contact,
      "features": features,
      "metadata": {
          "source": "assembly_contact_mined_mate_program",
          "leakage_scope": "train_or_dev_only",
          "part_frame": (
              "benchmark_v2_independent_se3"
              if child_to_parent_transform is not None
              else (
                  "bbox_center_canonical"
                  if candidate_a.metadata.get("mate_program_canonicalized")
                  else "raw_step_local"
              )
          ),
          "bidirectional_direction": direction,
      },
  }


def _program_residual(
    *,
    parent_frame: dict[str, Any],
    child_frame: dict[str, Any],
    child_to_parent_transform: Optional[Transform] = None,
) -> Transform:
  """Return the local mate program, correcting independently gauged frames."""

  if child_to_parent_transform is None:
    return residual_between_frames(parent_frame, child_frame)
  parent = frame_matrix_from_dict(parent_frame)
  child = frame_matrix_from_dict(child_frame)
  child_to_parent = matrix_from_transform(child_to_parent_transform)
  return transform_from_matrix(
      invert_transform_matrix(parent) @ child_to_parent @ child
  )


def _semantic_frame_diagnostic_row(
    *,
    row: Mapping[str, Any],
    candidate_a: CandidateInterface,
    candidate_b: CandidateInterface,
    prepared: PreparedBenchmarkV2Assembly | None,
    raw_part_a: str,
    raw_part_b: str,
    source_world_transforms: Mapping[str, Transform],
) -> dict[str, Any]:
  """Capture private semantic frames without changing the published mate row.

  Candidate frames live in independently randomized part coordinates.  The
  audit-only transform ``W_raw @ inverse(G_raw)`` places each such frame in the
  receipt-bound source world, where it can be compared with a full-graph OCC
  frame.  This helper is reachable only through an explicit diagnostic sink.
  """

  if prepared is None:
    raise ValueError("semantic-frame diagnostics require benchmark_v2 gauges")
  try:
    world_a = matrix_from_transform(source_world_transforms[str(raw_part_a)])
    world_b = matrix_from_transform(source_world_transforms[str(raw_part_b)])
  except KeyError as error:
    raise ValueError("semantic-frame diagnostic source world transform differs") from error
  gauge_a = matrix_from_transform(prepared.raw_transform_for(str(raw_part_a)))
  gauge_b = matrix_from_transform(prepared.raw_transform_for(str(raw_part_b)))
  local_a = frame_matrix_from_dict(candidate_a.local_frame)
  local_b = frame_matrix_from_dict(candidate_b.local_frame)
  semantic_world_a = world_a @ invert_transform_matrix(gauge_a) @ local_a
  semantic_world_b = world_b @ invert_transform_matrix(gauge_b) @ local_b
  target = np.eye(4, dtype=float)
  target[:3, :3] = np.asarray(row["residual_rotation"], dtype=float).reshape(3, 3)
  target[:3, 3] = np.asarray(row["residual_translation"], dtype=float).reshape(3)
  reconstructed_world_delta = (
      semantic_world_a @ target @ invert_transform_matrix(semantic_world_b)
  )
  return {
      "program_id": str(row["program_id"]),
      "case_id": str(row["case_id"]),
      "source_contact_ordinal": int(row["source_contact"]["contact_index"]),
      "direction": str(row["source_contact"]["direction"]),
      "raw_part_a": str(raw_part_a),
      "raw_part_b": str(raw_part_b),
      "semantic_frame_a_randomized": local_a.tolist(),
      "semantic_frame_b_randomized": local_b.tolist(),
      "raw_to_randomized_a": gauge_a.tolist(),
      "raw_to_randomized_b": gauge_b.tolist(),
      "source_world_a": world_a.tolist(),
      "source_world_b": world_b.tolist(),
      "semantic_frame_a_world": semantic_world_a.tolist(),
      "semantic_frame_b_world": semantic_world_b.tolist(),
      "semantic_target": target.tolist(),
      "semantic_world_delta": reconstructed_world_delta.tolist(),
      "surface_type_a": str(candidate_a.surface_type or "other").lower(),
      "surface_type_b": str(candidate_b.surface_type or "other").lower(),
  }


def _program_is_plausible(
    *,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    contact_type: str,
    residual_rotation: np.ndarray,
) -> bool:
  family = _contact_family(contact_type)
  role_a = str(row_a.get("role_hint") or "").lower()
  role_b = str(row_b.get("role_hint") or "").lower()
  family_a = _role_family(role_a)
  family_b = _role_family(role_b)
  families = {family_a, family_b}
  if family == "insert":
    if not ("hole" in families and "pin" in families):
      return False
    return abs(float(residual_rotation[2, 2])) >= 0.75
  if family == "slot":
    return "slot" in families and "pin" in families
  if family == "seat":
    if family_a != "plane" or family_b != "plane":
      return False
    return abs(float(residual_rotation[2, 2])) >= 0.75
  return True


def _contact_family(contact_type: str) -> str:
  text = str(contact_type or "").lower()
  if "slot" in text:
    return "slot"
  if any(token in text for token in ("insert", "bore", "hole", "shaft", "screw", "threaded")):
    return "insert"
  if any(token in text for token in ("seat", "support", "plane", "flange")):
    return "seat"
  return "generic"


def _role_family(role: str) -> str:
  text = str(role or "").lower()
  if text in {"center_bore", "threaded_hole", "hole_entry"}:
    return "hole"
  if text in {"shaft_axis", "pin_boss", "cylindrical_interface"}:
    return "pin"
  if "slot" in text:
    return "slot"
  if text in {"planar_seat", "shoulder_stop"}:
    return "plane"
  return text


def _candidates_for_contact_face(
    candidates: Any,
    face_index: int,
    *,
    max_count: int,
) -> list[CandidateInterface]:
  return _candidates_for_contact_faces(
      candidates,
      (int(face_index),),
      max_count=max_count,
  )


def _candidates_for_contact_faces(
    candidates: Any,
    face_indices: Sequence[int],
    *,
    max_count: int,
) -> list[CandidateInterface]:
  """Return candidates that cover the complete mapped OCC face group."""

  required = frozenset(int(index) for index in face_indices)
  if not required:
    raise ValueError("mapped contact face group must be nonempty")
  ranked = _rank_candidates(candidates)
  matched = [
      candidate
      for candidate in ranked
      if required <= frozenset(_candidate_contact_face_indices(candidate))
  ]
  matched.sort(
      key=lambda item: (
          -float(bool(item.metadata.get("composite_interface", False))),
          -(item.score if isinstance(item.score, (int, float)) else 0.0),
          -float(item.features.get("sqrt_area_ratio", 0.0)),
          item.interface_id,
      )
  )
  return matched[: max(1, int(max_count))]


def _compact_interface(
    candidate: CandidateInterface,
    *,
    input_protocol: str = "legacy",
) -> dict[str, Any]:
  metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
  if str(input_protocol).strip().lower() == "benchmark_v2":
    if metadata.get("model_input_protocol") != "benchmark_v2":
      raise ValueError("benchmark_v2 mate endpoint lacks model protocol binding")
    view = metadata.get("benchmark_v2_model_view")
    view_hash = metadata.get("benchmark_v2_model_view_sha256")
    if not isinstance(view, dict) or not isinstance(view_hash, str):
      raise ValueError("benchmark_v2 mate endpoint lacks sanitized model view")
    return {
        "model_input_protocol": "benchmark_v2",
        "benchmark_v2_model_view": view,
        "benchmark_v2_model_view_sha256": view_hash,
        "score": float(candidate.score or 0.0),
    }
  return {
      "interface_id": candidate.interface_id,
      "face_index": candidate.face_index,
      "surface_type": candidate.surface_type,
      "role_hint": candidate.role_hint,
      "score": candidate.score,
      "label": candidate.label,
      "label_source": candidate.label_source,
      "metadata": {
          "radius": metadata.get("radius"),
          "area": metadata.get("area"),
          "edge_count": metadata.get("edge_count"),
          "concavity": metadata.get("concavity"),
          "composite_interface": bool(metadata.get("composite_interface", False)),
          "member_face_indices": metadata.get("member_face_indices", []),
          "canonicalization_offset": metadata.get("canonicalization_offset"),
      },
  }


def _semantic_interface_row(endpoint: dict[str, Any]) -> dict[str, Any]:
  """Recover only invariant role/surface hints from a sanitized v2 endpoint."""

  view = endpoint.get("benchmark_v2_model_view")
  if not isinstance(view, dict):
    raise ValueError("benchmark_v2 endpoint is missing model view")
  topology = view.get("topology")
  if not isinstance(topology, dict):
    topology = {}
  role = str(topology.get("composite_role") or "").strip().lower()
  surface = str(view.get("surface_type") or "other").strip().lower()
  if not role:
    role = {
        "plane": "planar_seat",
        "cylinder": "cylindrical_interface",
        "cone": "shoulder_stop",
        "torus": "shoulder_stop",
    }.get(surface, "generic_interface")
  return {
      "role_hint": role,
      "surface_type": surface,
      "features": {
          "has_helical_edge": float(bool(topology.get("has_helical_edge", False)))
      },
  }


def _opaque_case_id(case: dict[str, Any]) -> str:
  payload = {
      "id": str(case.get("id") or ""),
      "assembly_dir": str(case.get("assembly_dir") or ""),
  }
  return "opaque_case_" + _canonical_sha256(payload)[:20]


def _canonical_sha256(payload: Any) -> str:
  encoded = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _finalize_v2_library_manifest(
    *,
    library_path: Path,
    row_count: int,
    source_split: str,
    source_split_sha256: str,
    mining_config_sha256: str,
    family_split_manifest_sha256: str,
    source_case_set_sha256: str,
    mining_config: dict[str, Any],
    coverage_ledger: dict[str, Any],
    coverage: dict[str, int],
) -> dict[str, Any]:
  if str(source_split) != "train":
    raise ValueError("formal benchmark_v2 mate libraries require train split")
  if int(row_count) < 1:
    raise ValueError("formal benchmark_v2 mate library contains no programs")
  if mining_config.get("schema_version") != MATE_MINING_CONFIG_SCHEMA:
    raise ValueError("formal mate mining config schema mismatch")
  if (
      type(mining_config.get("max_candidates_per_part")) is not int
      or mining_config.get("max_candidates_per_part") != 0
  ):
    raise ValueError("formal mate mining requires max_candidates_per_part == 0")
  if (
      type(mining_config.get("max_candidates_per_contact_face")) is not int
      or int(mining_config["max_candidates_per_contact_face"]) < 1
  ):
    raise ValueError(
        "formal mate mining requires max_candidates_per_contact_face >= 1"
    )
  if _canonical_sha256(mining_config) != str(mining_config_sha256):
    raise ValueError("formal mate mining config hash mismatch")
  if coverage_ledger.get("schema_version") != MATE_COVERAGE_LEDGER_SCHEMA:
    raise ValueError("formal mate coverage ledger schema mismatch")
  _validate_coverage_ledger(coverage_ledger, coverage=coverage)
  if int(coverage.get("directed_program_count", -1)) != int(row_count):
    raise ValueError("formal mate coverage does not match library row count")
  code_bundle = mate_miner_code_bundle_identity()
  manifest = {
      "schema_version": MATE_LIBRARY_MANIFEST_SCHEMA,
      "model_input_protocol": "benchmark_v2",
      "protocol_version": PROTOCOL_VERSION,
      "source_split": "train",
      "source_split_sha256": str(source_split_sha256),
      "mining_config": mining_config,
      "mining_config_sha256": str(mining_config_sha256),
      "miner_code_bundle": code_bundle,
      "miner_code_bundle_sha256": code_bundle["bundle_sha256"],
      "family_split_manifest_sha256": str(family_split_manifest_sha256),
      "source_case_set_sha256": str(source_case_set_sha256),
      "coverage": {str(key): int(value) for key, value in coverage.items()},
      "coverage_ledger": coverage_ledger,
      "library_sha256": _file_sha256(Path(library_path)),
      "row_count": int(row_count),
  }
  manifest["manifest_sha256"] = _canonical_sha256(manifest)
  sidecar = Path(library_path).with_suffix(
      Path(library_path).suffix + ".manifest.json"
  )
  sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return manifest


def _canonicalize_part_record_frames(part_records: dict[str, dict[str, Any]]) -> None:
  for record in part_records.values():
    step_path = record.get("step_path")
    candidates = record.get("candidates")
    if not isinstance(step_path, str) or not isinstance(candidates, list):
      continue
    try:
      bbox_min, bbox_max = shape_bbox(load_step_shape(step_path))
      center = (np.asarray(bbox_min, dtype=float) + np.asarray(bbox_max, dtype=float)) / 2.0
      offset = -center
    except Exception:
      continue
    for candidate in candidates:
      if not isinstance(candidate, CandidateInterface):
        continue
      _translate_candidate_frame(candidate, offset)


def _translate_candidate_frame(candidate: CandidateInterface, offset: np.ndarray) -> None:
  frame = candidate.local_frame
  if not isinstance(frame, dict):
    return
  origin = frame.get("origin")
  if isinstance(origin, (list, tuple)):
    try:
      frame["origin"] = (
          np.asarray(origin, dtype=float).reshape(3)
          + np.asarray(offset, dtype=float).reshape(3)
      ).tolist()
    except Exception:
      return
  candidate.metadata["mate_program_canonicalized"] = True
  candidate.metadata["canonicalization_offset"] = (
      np.asarray(offset, dtype=float).reshape(3).tolist()
  )


def _body_to_part_name(case: dict[str, Any]) -> dict[str, str]:
  names = case.get("selected_part_names")
  bodies = case.get("selected_body_uuids")
  if not isinstance(names, list) or not isinstance(bodies, list):
    return {}
  return {
      str(body): str(name)
      for name, body in zip(names, bodies)
      if str(body).strip() and str(name).strip()
  }


def _load_contacts(assembly_json: Path) -> list[dict[str, Any]]:
  try:
    data = json.loads(assembly_json.read_text(encoding="utf-8"))
  except Exception:
    return []
  return _contacts_from_assembly_data(data)


def _contacts_from_assembly_data(
    data: Mapping[str, Any],
) -> list[dict[str, Any]]:
  contacts = data.get("contacts")
  if not isinstance(contacts, list):
    return []
  return [item for item in contacts if isinstance(item, dict)]


if __name__ == "__main__":
  main()
