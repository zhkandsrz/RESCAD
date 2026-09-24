"""Development-only semantic replay for the complete V19 pending-query domain.

This is deliberately a separate protocol from
``query_mate_semantic_authority_v1``.  The older protocol is pinned to the
127-query overlay pilot.  This protocol selects every V19 query whose geometry
status is ``geometry_certified_pending_semantic`` while preserving the
pre-registered train/dev and family partition.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .benchmark_v2_training_provenance import canonical_sha256
from .mate_pose_miner import replay_benchmark_v2_query_mate_semantics
from . import query_mate_semantic_authority_v1 as pilot_authority


SCHEMA_VERSION = "neurocad_query_mate_semantic_full_domain_authority.v1"
SELECTION = (
    "all_v19_geometry_certified_pending_semantic_rows_"
    "schedule_order_split_preserving.v1"
)
PENDING_STATUS = "geometry_certified_pending_semantic"
ALLOWED_SPLITS = ("train", "dev")
_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "query_mate_semantic_full_domain_v1.schema.json"
)
_PROJECT_ROOT = Path(__file__).resolve().parent
_RUN_CONFIG_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "query_mate_semantic_full_domain_v1.run.json"
)
_EXPECTED_RUN_CONFIG_SHA256 = (
    "a0d681863838693f7d4dbff7ce82f9e7b5b87d4790cf55360a3e58c469f014b2"
)


def _sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _file_binding(path: str | Path) -> dict[str, Any]:
  resolved = Path(path).resolve(strict=True)
  if not resolved.is_file() or resolved.is_symlink():
    raise ValueError("full-domain authority input must be a plain file")
  return {
      "path": str(resolved),
      "bytes": resolved.stat().st_size,
      "sha256": _sha256_file(resolved),
  }


def _read_json(path: str | Path, *, label: str) -> Any:
  try:
    return json.loads(Path(path).read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not readable UTF-8 JSON") from error


def _load_fixed_run_contract(
    v19_manifest_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
  """Open the one committed run config and the one manifest it pins."""

  config_binding = _file_binding(_RUN_CONFIG_PATH)
  if config_binding["sha256"] != _EXPECTED_RUN_CONFIG_SHA256:
    raise ValueError("fixed full938 run config bytes differ")
  config = _read_json(_RUN_CONFIG_PATH, label="fixed full938 run config")
  if (
      not isinstance(config, Mapping)
      or config.get("schema_version")
      != "neurocad_query_mate_semantic_full_domain_run_config.v1"
      or config.get("development") is not True
      or config.get("formal") is not False
      or config.get("publication_eligible") is not False
      or config.get("final_test_touched") is not False
      or config.get("withheld_test_touched") is not False
  ):
    raise ValueError("fixed full938 run config boundary differs")
  expected_manifest_path = (
      _PROJECT_ROOT / Path(str(config.get("v19_manifest_path") or ""))
  ).resolve(strict=True)
  observed_manifest_path = Path(v19_manifest_path).resolve(strict=True)
  if observed_manifest_path != expected_manifest_path:
    raise ValueError("full938 authority refuses a substituted manifest path")
  manifest_binding = _file_binding(observed_manifest_path)
  if manifest_binding["sha256"] != config.get("v19_manifest_file_sha256"):
    raise ValueError("full938 authority refuses substituted manifest bytes")
  manifest = _read_json(
      observed_manifest_path, label="fixed V19 query manifest",
  )
  if (
      not isinstance(manifest, Mapping)
      or manifest.get("manifest_payload_sha256")
      != config.get("v19_manifest_payload_sha256")
  ):
    raise ValueError("full938 authority refuses a substituted manifest payload")
  return config, manifest, observed_manifest_path


def _write_new_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
  """Atomically publish a new JSON file without replacing any existing name."""

  destination = path.resolve()
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  if destination.exists() or temporary.exists():
    raise FileExistsError("full-domain receipt destination already exists")
  destination.parent.mkdir(parents=True, exist_ok=True)
  raw = json.dumps(
      payload, ensure_ascii=False, sort_keys=True,
  ).encode("utf-8")
  try:
    with temporary.open("xb") as stream:
      stream.write(raw)
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, destination)
  finally:
    temporary.unlink(missing_ok=True)


def _self_hash(value: Mapping[str, Any], field: str, *, label: str) -> None:
  unsigned = dict(value)
  observed = unsigned.pop(field, None)
  if observed != canonical_sha256(unsigned):
    raise ValueError(f"{label} self-hash differs")


def _schedule_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
  schedule = row.get("schedule")
  key = row.get("key")
  if not isinstance(schedule, Mapping) or not isinstance(key, Mapping):
    raise ValueError("V19 pending query lacks schedule or key")
  ordinal = schedule.get("global_schedule_ordinal")
  source_ordinal = key.get("source_contact_ordinal")
  if (
      isinstance(ordinal, bool)
      or not isinstance(ordinal, int)
      or ordinal < 0
      or isinstance(source_ordinal, bool)
      or not isinstance(source_ordinal, int)
      or source_ordinal < 0
  ):
    raise ValueError("V19 pending query schedule/key ordinal differs")
  case_id = str(key.get("case_id") or "")
  if not case_id:
    raise ValueError("V19 pending query case identity differs")
  return ordinal, case_id, source_ordinal


def derive_v19_full_pending_semantic_domain(
    manifest: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], str, dict[str, Any]]:
  """Return the complete pending-semantic domain in pre-registered order."""

  pilot_authority._validate_v19_development_boundary(manifest)
  domains = manifest.get("domains")
  raw_rows = domains.get("query_rows") if isinstance(domains, Mapping) else None
  if not isinstance(raw_rows, list):
    raise ValueError("V19 query row domain differs")
  selected: list[dict[str, Any]] = []
  seen_keys: set[tuple[str, int]] = set()
  split_by_case: dict[str, str] = {}
  split_by_family: dict[str, str] = {}
  for raw_row in raw_rows:
    if not isinstance(raw_row, Mapping):
      raise ValueError("V19 query row differs")
    if raw_row.get("status") != PENDING_STATUS:
      continue
    _self_hash(raw_row, "row_payload_sha256", label="V19 pending query")
    schedule_ordinal, case_id, source_ordinal = _schedule_key(raw_row)
    split = str(raw_row.get("split") or "")
    family_sha256 = str(raw_row.get("family_sha256") or "")
    if split not in ALLOWED_SPLITS:
      raise ValueError("full-domain query split is not train/dev")
    if (
        len(family_sha256) != 64
        or any(character not in "0123456789abcdef" for character in family_sha256)
    ):
      raise ValueError("full-domain query family identity differs")
    query_key = (case_id, source_ordinal)
    if query_key in seen_keys:
      raise ValueError("full-domain query identity is duplicated")
    seen_keys.add(query_key)
    previous_case_split = split_by_case.setdefault(case_id, split)
    previous_family_split = split_by_family.setdefault(family_sha256, split)
    if previous_case_split != split:
      raise ValueError("full-domain case crosses train/dev")
    if previous_family_split != split:
      raise ValueError("full-domain family crosses train/dev")
    selected.append(copy.deepcopy(dict(raw_row)))
  selected.sort(key=_schedule_key)
  if not selected:
    raise ValueError("V19 full pending-semantic query domain is empty")

  split_counts = {
      split: sum(row["split"] == split for row in selected)
      for split in ALLOWED_SPLITS
  }
  case_ids_by_split = {
      split: sorted({
          str(row["key"]["case_id"])
          for row in selected
          if row["split"] == split
      })
      for split in ALLOWED_SPLITS
  }
  family_ids_by_split = {
      split: sorted({
          str(row["family_sha256"])
          for row in selected
          if row["split"] == split
      })
      for split in ALLOWED_SPLITS
  }
  if set(case_ids_by_split["train"]).intersection(case_ids_by_split["dev"]):
    raise ValueError("full-domain train/dev case overlap")
  if set(family_ids_by_split["train"]).intersection(
      family_ids_by_split["dev"]
  ):
    raise ValueError("full-domain train/dev family overlap")

  preimage = {
      "selection": SELECTION,
      "rows": [
          {
              "global_schedule_ordinal": _schedule_key(row)[0],
              "case_id": row["key"]["case_id"],
              "source_contact_ordinal": row["key"]["source_contact_ordinal"],
              "split": row["split"],
              "family_sha256": row["family_sha256"],
              "row_payload_sha256": row["row_payload_sha256"],
          }
          for row in selected
      ],
  }
  summary = {
      "selected_query_count": len(selected),
      "split_query_counts": split_counts,
      "split_case_counts": {
          split: len(case_ids_by_split[split]) for split in ALLOWED_SPLITS
      },
      "split_family_counts": {
          split: len(family_ids_by_split[split]) for split in ALLOWED_SPLITS
      },
      "case_ids_by_split_sha256": {
          split: canonical_sha256(case_ids_by_split[split])
          for split in ALLOWED_SPLITS
      },
      "family_ids_by_split_sha256": {
          split: canonical_sha256(family_ids_by_split[split])
          for split in ALLOWED_SPLITS
      },
      "case_overlap_count": 0,
      "family_overlap_count": 0,
      "final_test_touched": False,
      "withheld_test_touched": False,
  }
  return tuple(selected), canonical_sha256(preimage), summary


def _rows_by_split(
    domain_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
  return {
      split: tuple(row for row in domain_rows if row["split"] == split)
      for split in ALLOWED_SPLITS
  }


def _require_fixed_domain_summary(
    *,
    config: Mapping[str, Any],
    domain_sha256: str,
    summary: Mapping[str, Any],
) -> None:
  expected_queries = config.get("expected_query_counts")
  expected_cases = config.get("expected_case_counts")
  if (
      domain_sha256 != config.get("full_query_domain_sha256")
      or not isinstance(expected_queries, Mapping)
      or expected_queries.get("total") != summary["selected_query_count"]
      or any(
          expected_queries.get(split)
          != summary["split_query_counts"][split]
          for split in ALLOWED_SPLITS
      )
      or not isinstance(expected_cases, Mapping)
      or expected_cases.get("total")
      != sum(summary["split_case_counts"].values())
      or any(
          expected_cases.get(split) != summary["split_case_counts"][split]
          for split in ALLOWED_SPLITS
      )
      or config.get("expected_case_ids_by_split_sha256")
      != summary["case_ids_by_split_sha256"]
      or config.get("expected_family_ids_by_split_sha256")
      != summary["family_ids_by_split_sha256"]
      or summary["case_overlap_count"] != 0
      or summary["family_overlap_count"] != 0
  ):
    raise ValueError("fixed full938 query domain differs")


def _selectors(
    query_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
  return pilot_authority._selectors(query_rows)


def _canonical_miner_selector_commitment(
    query_rows: Sequence[Mapping[str, Any]],
) -> str:
  """Mirror the miner's canonical case-id/ordinal selector serialization."""

  selector_rows = [
      {
          "case_id": str(row["key"]["case_id"]),
          "source_contact_ordinal": int(
              row["key"]["source_contact_ordinal"]
          ),
      }
      for row in query_rows
  ]
  selector_rows.sort(
      key=lambda row: (
          row["case_id"], row["source_contact_ordinal"],
      )
  )
  return canonical_sha256(selector_rows)


def _producer_binding() -> dict[str, Any]:
  root = Path(__file__).resolve().parent
  paths = {
      "neurocad.query_mate_semantic_full_domain_v1": Path(__file__),
      "neurocad.query_mate_semantic_authority_v1": (
          root / "query_mate_semantic_authority_v1.py"
      ),
      "neurocad.mate_pose_miner": root / "mate_pose_miner.py",
      "neurocad.tools.query_mate_semantic_full_domain_v1": (
          root / "tools" / "query_mate_semantic_full_domain_v1.py"
      ),
      "run_config": (
          root / "configs" / "query_mate_semantic_full_domain_v1.run.json"
      ),
      "schema": _SCHEMA_PATH,
  }
  return {
      "implementation": {
          name: _file_binding(path) for name, path in sorted(paths.items())
      },
      "replay_contract": (
          "v19_complete_pending_domain_private_gold_ordinal_"
          "authenticated_face_map_assembly_step_program_replay.v1"
      ),
  }


def _validate_split_inputs(
    *,
    split: str,
    domain_rows: Sequence[Mapping[str, Any]],
    split_inputs: Mapping[str, Any],
    fixed_frame_seed: int,
) -> None:
  required = {
      "public_cases_path",
      "private_source_path",
      "private_gold_path",
      "family_split_manifest_path",
      "face_map_receipt_paths",
      "family_split_manifest_sha256",
      "source_split_sha256",
      "mining_config",
  }
  if set(split_inputs) != required:
    raise ValueError(f"{split} replay input keys differ")
  public_cases = _read_json(
      split_inputs["public_cases_path"], label=f"{split} public cases",
  )
  if not isinstance(public_cases, list) or not public_cases:
    raise ValueError(f"{split} public replay case domain differs")
  expected_cases = {
      str(row["key"]["case_id"]) for row in domain_rows
  }
  observed_cases = [str(row.get("id") or "") for row in public_cases]
  if (
      len(observed_cases) != len(set(observed_cases))
      or set(observed_cases) != expected_cases
  ):
    raise ValueError(f"{split} public replay case set differs")
  mapper_paths = split_inputs["face_map_receipt_paths"]
  if not isinstance(mapper_paths, Mapping) or set(mapper_paths) != expected_cases:
    raise ValueError(f"{split} mapper receipt case set differs")
  if (
      _sha256_file(split_inputs["family_split_manifest_path"])
      != split_inputs["family_split_manifest_sha256"]
      or _sha256_file(split_inputs["public_cases_path"])
      != split_inputs["source_split_sha256"]
  ):
    raise ValueError(f"{split} replay split hashes differ")
  mining_config = split_inputs["mining_config"]
  if (
      not isinstance(mining_config, Mapping)
      or mining_config.get("source_split") != "train"
      or mining_config.get("development_target_split") != split
      or mining_config.get("frame_randomization_seed") != fixed_frame_seed
      or mining_config.get("family_split_manifest_sha256")
      != split_inputs["family_split_manifest_sha256"]
      or mining_config.get("source_split_sha256")
      != split_inputs["source_split_sha256"]
      or mining_config.get("label_policy")
      != "source_positive_only_unknown_excluded"
  ):
    raise ValueError(f"{split} development replay mining config differs")


@dataclass(frozen=True, slots=True)
class _SplitReplay:
  commitments: tuple[dict[str, Any], ...]
  program_rows: tuple[Mapping[str, Any], ...]
  query_ledger: tuple[Mapping[str, Any], ...]
  raw_program_rows: tuple[Mapping[str, Any], ...]
  raw_query_ledger: tuple[Mapping[str, Any], ...]
  private_supervision: Any
  face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]]
  public_case_ids: tuple[str, ...]
  opaque_case_id_by_case_id: Mapping[str, str]
  input_paths: Mapping[str, str | Path]
  selector_sha256: str


def _run_split_replay(
    *,
    split: str,
    domain_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    split_inputs: Mapping[str, Any],
    archive_root: str | Path,
    restored_archive_root: str | Path | None,
    fixed_frame_seed: int,
) -> _SplitReplay:
  _validate_split_inputs(
      split=split,
      domain_rows=domain_rows,
      split_inputs=split_inputs,
      fixed_frame_seed=fixed_frame_seed,
  )
  authenticated, family_source_path = (
      pilot_authority._independent_mapper_context(
          manifest=manifest,
          domain_rows=domain_rows,
          public_cases_path=split_inputs["public_cases_path"],
          private_source_path=split_inputs["private_source_path"],
          private_gold_path=split_inputs["private_gold_path"],
          family_split_manifest_path=split_inputs[
              "family_split_manifest_path"
          ],
          face_map_receipt_paths=split_inputs["face_map_receipt_paths"],
      )
  )
  mining_config = dict(split_inputs["mining_config"])
  replay = replay_benchmark_v2_query_mate_semantics(
      public_cases=authenticated.public_cases,
      private_supervision=authenticated.private_supervision,
      face_map_lookup=authenticated.face_map_lookup,
      archive_root=archive_root,
      mining_config=mining_config,
      family_split_manifest_sha256=split_inputs[
          "family_split_manifest_sha256"
      ],
      source_split_sha256=split_inputs["source_split_sha256"],
      source_contact_ordinals=_selectors(domain_rows),
      _case_archive_root_resolver=pilot_authority._archive_resolver(
          restored_archive_root
      ),
  )
  selector_sha256 = _canonical_miner_selector_commitment(domain_rows)
  if replay.input_domain_sha256 != selector_sha256:
    raise ValueError(f"{split} miner query selector commitment differs")
  frame_seed = mining_config.get("frame_randomization_seed")
  if isinstance(frame_seed, bool) or not isinstance(frame_seed, int):
    raise ValueError(f"{split} frame seed differs")
  commitments = pilot_authority._query_commitments(
      domain_rows=domain_rows,
      private_supervision=authenticated.private_supervision,
      face_map_lookup=authenticated.face_map_lookup,
      replay=replay,
      frame_seed=frame_seed,
  )
  split_commitments = tuple({
      **commitment,
      "development_split": split,
  } for commitment in commitments)
  input_paths: dict[str, str | Path] = {
      "public_cases": split_inputs["public_cases_path"],
      "private_source": split_inputs["private_source_path"],
      "private_gold": split_inputs["private_gold_path"],
      "family_split_manifest": split_inputs["family_split_manifest_path"],
      "v19_bound_family_source": family_source_path,
  }
  input_paths.update({
      f"development_mapper_receipt:{case_id}": path
      for case_id, path in sorted(
          split_inputs["face_map_receipt_paths"].items()
      )
  })
  return _SplitReplay(
      commitments=split_commitments,
      program_rows=tuple({
          **copy.deepcopy(dict(row)),
          "development_split": split,
      } for row in replay.rows),
      query_ledger=tuple({
          **copy.deepcopy(dict(row)),
          "development_split": split,
      } for row in replay.query_ledger),
      raw_program_rows=tuple(replay.rows),
      raw_query_ledger=tuple(replay.query_ledger),
      private_supervision=authenticated.private_supervision,
      face_map_lookup=authenticated.face_map_lookup,
      public_case_ids=tuple(
          str(case["id"]) for case in authenticated.public_cases
      ),
      opaque_case_id_by_case_id=dict(replay.opaque_case_id_by_case_id),
      input_paths=input_paths,
      selector_sha256=selector_sha256,
  )


def _coverage(
    replays: Mapping[str, _SplitReplay],
) -> dict[str, Any]:
  split_coverage: dict[str, dict[str, int]] = {}
  for split in ALLOWED_SPLITS:
    replay = replays[split]
    authorized = sum(
        row["status"] == "authorized" for row in replay.query_ledger
    )
    split_coverage[split] = {
        "selected_query_count": len(replay.query_ledger),
        "authorized_query_count": authorized,
        "rejected_query_count": len(replay.query_ledger) - authorized,
        "program_row_count": len(replay.program_rows),
    }
  return {
      "selected_query_count": sum(
          values["selected_query_count"]
          for values in split_coverage.values()
      ),
      "authorized_query_count": sum(
          values["authorized_query_count"]
          for values in split_coverage.values()
      ),
      "rejected_query_count": sum(
          values["rejected_query_count"]
          for values in split_coverage.values()
      ),
      "program_row_count": sum(
          values["program_row_count"]
          for values in split_coverage.values()
      ),
      "by_split": split_coverage,
  }


def build_query_mate_semantic_full_domain_v1_receipt(
    *,
    receipt_path: str | Path,
    v19_manifest_path: str | Path,
    split_replay_inputs: Mapping[str, Mapping[str, Any]],
    archive_root: str | Path,
    restored_archive_root: str | Path | None = None,
) -> dict[str, Any]:
  """Replay both model splits and write a development-only aggregate receipt."""

  destination = Path(receipt_path)
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  if destination.exists() or temporary.exists():
    raise FileExistsError("full-domain receipt destination already exists")
  config, fixed_manifest, fixed_manifest_path = _load_fixed_run_contract(
      v19_manifest_path
  )
  manifest = pilot_authority._validate_v19_development_boundary(
      fixed_manifest
  )
  domain_rows, domain_sha256, domain_summary = (
      derive_v19_full_pending_semantic_domain(manifest)
  )
  _require_fixed_domain_summary(
      config=config, domain_sha256=domain_sha256, summary=domain_summary,
  )
  if set(split_replay_inputs) != set(ALLOWED_SPLITS):
    raise ValueError("full-domain authority requires train and dev replays")
  rows_by_split = _rows_by_split(domain_rows)
  fixed_frame_seed = config.get("frame_randomization_seed")
  if isinstance(fixed_frame_seed, bool) or not isinstance(
      fixed_frame_seed, int
  ):
    raise ValueError("fixed full938 frame seed differs")
  for split in ALLOWED_SPLITS:
    _validate_split_inputs(
        split=split,
        domain_rows=rows_by_split[split],
        split_inputs=split_replay_inputs[split],
        fixed_frame_seed=fixed_frame_seed,
    )
  replays = {
      split: _run_split_replay(
          split=split,
          domain_rows=rows_by_split[split],
          manifest=manifest,
          split_inputs=split_replay_inputs[split],
          archive_root=archive_root,
          restored_archive_root=restored_archive_root,
          fixed_frame_seed=fixed_frame_seed,
      )
      for split in ALLOWED_SPLITS
  }
  commitment_by_key = {
      (
          commitment["case_id"],
          commitment["source_contact_ordinal"],
      ): commitment
      for replay in replays.values()
      for commitment in replay.commitments
  }
  ordered_commitments = [
      commitment_by_key[
          (row["key"]["case_id"], row["key"]["source_contact_ordinal"])
      ]
      for row in domain_rows
  ]
  input_paths: dict[str, str | Path] = {
      "fixed_run_config": _RUN_CONFIG_PATH,
      "v19_query_manifest": fixed_manifest_path,
  }
  for split, replay in replays.items():
    input_paths.update({
        f"{split}:{name}": path
        for name, path in replay.input_paths.items()
    })
  payload = {
      "schema_version": SCHEMA_VERSION,
      "development": True,
      "formal": False,
      "publication_eligible": False,
      "formal_promotion_blocked": True,
      "final_test_touched": False,
      "withheld_test_touched": False,
      "authority_source": (
          "development_full_domain_query_replay_cross_checked"
      ),
      "input_domain": {
          "selection": SELECTION,
          "commitment_sha256": domain_sha256,
          **domain_summary,
          "miner_selector_commitment_sha256_by_split": {
              split: replays[split].selector_sha256
              for split in ALLOWED_SPLITS
          },
      },
      "inputs": {
          name: _file_binding(path)
          for name, path in sorted(input_paths.items())
      },
      "mining_configs": {
          split: copy.deepcopy(dict(split_replay_inputs[split]["mining_config"]))
          for split in ALLOWED_SPLITS
      },
      "mining_config_sha256_by_split": {
          split: canonical_sha256(
              dict(split_replay_inputs[split]["mining_config"])
          )
          for split in ALLOWED_SPLITS
      },
      "producer": _producer_binding(),
      "query_commitments": ordered_commitments,
      "coverage": _coverage(replays),
  }
  receipt = copy.deepcopy(payload)
  receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
  _write_new_json_atomic(destination, receipt)
  return receipt


def _freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({
        key: _freeze(item) for key, item in value.items()
    })
  if isinstance(value, (list, tuple)):
    return tuple(_freeze(item) for item in value)
  return value


def _make_handle_type():
  token = object()

  class AuthenticatedQueryMateSemanticFullDomain:
    __slots__ = (
        "_program_rows_by_split", "_query_ledger_by_split",
        "_receipt_sha256", "_sealed",
    )

    def __init__(
        self,
        *,
        replays: Mapping[str, _SplitReplay],
        receipt_sha256: str,
        _token: object,
    ) -> None:
      if _token is not token:
        raise TypeError("full-domain semantic authority is verifier-only")
      object.__setattr__(
          self,
          "_program_rows_by_split",
          _freeze({
              split: replays[split].program_rows for split in ALLOWED_SPLITS
          }),
      )
      object.__setattr__(
          self,
          "_query_ledger_by_split",
          _freeze({
              split: replays[split].query_ledger for split in ALLOWED_SPLITS
          }),
      )
      object.__setattr__(self, "_receipt_sha256", receipt_sha256)
      object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: Any) -> None:
      if getattr(self, "_sealed", False):
        raise AttributeError("authenticated full-domain authority is immutable")
      object.__setattr__(self, _name, _value)

    @property
    def program_rows_by_split(self) -> Mapping[str, Sequence[Mapping[str, Any]]]:
      return self._program_rows_by_split

    @property
    def query_ledger_by_split(self) -> Mapping[str, Sequence[Mapping[str, Any]]]:
      return self._query_ledger_by_split

    @property
    def receipt_sha256(self) -> str:
      return self._receipt_sha256

    def __reduce_ex__(self, _protocol: int) -> Any:
      raise TypeError("authenticated full-domain authority is not serializable")

  def issue(
      *,
      replays: Mapping[str, _SplitReplay],
      receipt_sha256: str,
  ) -> AuthenticatedQueryMateSemanticFullDomain:
    return AuthenticatedQueryMateSemanticFullDomain(
        replays=replays, receipt_sha256=receipt_sha256, _token=token,
    )

  return AuthenticatedQueryMateSemanticFullDomain, issue


(
    AuthenticatedQueryMateSemanticFullDomain,
    _issue_authenticated_full_domain,
) = _make_handle_type()


def verify_query_mate_semantic_full_domain_v1(
    *,
    receipt_path: str | Path,
    v19_manifest_path: str | Path,
    split_replay_inputs: Mapping[str, Mapping[str, Any]],
    archive_root: str | Path,
    restored_archive_root: str | Path | None = None,
) -> AuthenticatedQueryMateSemanticFullDomain:
  """Independently replay all 938-domain inputs before issuing a handle."""

  config, fixed_manifest, fixed_manifest_path = _load_fixed_run_contract(
      v19_manifest_path
  )
  receipt_binding = _file_binding(receipt_path)
  receipt = _read_json(receipt_path, label="full-domain receipt")
  if not isinstance(receipt, Mapping):
    raise ValueError("full-domain receipt root differs")
  _self_hash(
      receipt, "receipt_payload_sha256", label="full-domain receipt",
  )
  expected_keys = {
      "schema_version", "development", "formal", "publication_eligible",
      "formal_promotion_blocked", "final_test_touched",
      "withheld_test_touched", "authority_source", "input_domain", "inputs",
      "mining_configs", "mining_config_sha256_by_split", "producer",
      "query_commitments", "coverage", "receipt_payload_sha256",
  }
  if (
      set(receipt) != expected_keys
      or receipt.get("schema_version") != SCHEMA_VERSION
      or receipt.get("development") is not True
      or receipt.get("formal") is not False
      or receipt.get("publication_eligible") is not False
      or receipt.get("formal_promotion_blocked") is not True
      or receipt.get("final_test_touched") is not False
      or receipt.get("withheld_test_touched") is not False
      or receipt.get("authority_source")
      != "development_full_domain_query_replay_cross_checked"
  ):
    raise ValueError("full-domain development authority boundary differs")
  manifest = pilot_authority._validate_v19_development_boundary(
      fixed_manifest
  )
  domain_rows, domain_sha256, domain_summary = (
      derive_v19_full_pending_semantic_domain(manifest)
  )
  _require_fixed_domain_summary(
      config=config, domain_sha256=domain_sha256, summary=domain_summary,
  )
  if set(split_replay_inputs) != set(ALLOWED_SPLITS):
    raise ValueError("verifier requires train and dev replay inputs")
  rows_by_split = _rows_by_split(domain_rows)
  fixed_frame_seed = config.get("frame_randomization_seed")
  if isinstance(fixed_frame_seed, bool) or not isinstance(
      fixed_frame_seed, int
  ):
    raise ValueError("fixed full938 frame seed differs")
  for split in ALLOWED_SPLITS:
    _validate_split_inputs(
        split=split,
        domain_rows=rows_by_split[split],
        split_inputs=split_replay_inputs[split],
        fixed_frame_seed=fixed_frame_seed,
    )
  replays = {
      split: _run_split_replay(
          split=split,
          domain_rows=rows_by_split[split],
          manifest=manifest,
          split_inputs=split_replay_inputs[split],
          archive_root=archive_root,
          restored_archive_root=restored_archive_root,
          fixed_frame_seed=fixed_frame_seed,
      )
      for split in ALLOWED_SPLITS
  }
  expected_domain = {
      "selection": SELECTION,
      "commitment_sha256": domain_sha256,
      **domain_summary,
      "miner_selector_commitment_sha256_by_split": {
          split: replays[split].selector_sha256
          for split in ALLOWED_SPLITS
      },
  }
  if receipt.get("input_domain") != expected_domain:
    raise ValueError("full-domain receipt domain differs")
  for split in ALLOWED_SPLITS:
    config = dict(split_replay_inputs[split]["mining_config"])
    if (
        receipt.get("mining_configs", {}).get(split) != config
        or receipt.get("mining_config_sha256_by_split", {}).get(split)
        != canonical_sha256(config)
    ):
      raise ValueError("full-domain mining config binding differs")

  receipt_rows = receipt.get("query_commitments")
  if not isinstance(receipt_rows, list):
    raise ValueError("full-domain query commitment rows differ")
  observed_order = [
      (
          row.get("development_split"),
          row.get("case_id"),
          row.get("source_contact_ordinal"),
      )
      for row in receipt_rows
      if isinstance(row, Mapping)
  ]
  expected_order = [
      (
          row["split"],
          row["key"]["case_id"],
          row["key"]["source_contact_ordinal"],
      )
      for row in domain_rows
  ]
  if observed_order != expected_order:
    raise ValueError("full-domain query commitment order differs")
  observed_by_split = {
      split: [
          {key: value for key, value in row.items()
           if key != "development_split"}
          for row in receipt_rows
          if isinstance(row, Mapping)
          and row.get("development_split") == split
      ]
      for split in ALLOWED_SPLITS
  }
  for split in ALLOWED_SPLITS:
    replay = replays[split]
    replay_namespace = type("_Replay", (), {
        "rows": replay.raw_program_rows,
        "query_ledger": replay.raw_query_ledger,
        "opaque_case_id_by_case_id": replay.opaque_case_id_by_case_id,
    })()
    frame_seed = split_replay_inputs[split]["mining_config"].get(
        "frame_randomization_seed"
    )
    pilot_authority._verify_query_commitments_independently(
        receipt_rows=observed_by_split[split],
        domain_rows=rows_by_split[split],
        private_supervision=replay.private_supervision,
        face_map_lookup=replay.face_map_lookup,
        replay=replay_namespace,
        frame_seed=frame_seed,
    )
  if len(receipt_rows) != len(domain_rows):
    raise ValueError("full-domain receipt query coverage differs")
  if receipt.get("coverage") != _coverage(replays):
    raise ValueError("full-domain receipt coverage differs")
  if receipt.get("producer") != _producer_binding():
    raise ValueError("full-domain producer binding differs")

  expected_input_paths: dict[str, str | Path] = {
      "fixed_run_config": _RUN_CONFIG_PATH,
      "v19_query_manifest": fixed_manifest_path,
  }
  for split, replay in replays.items():
    expected_input_paths.update({
        f"{split}:{name}": path
        for name, path in replay.input_paths.items()
    })
  expected_inputs = {
      name: _file_binding(path)
      for name, path in sorted(expected_input_paths.items())
  }
  if receipt.get("inputs") != expected_inputs:
    raise ValueError("full-domain input byte bindings differ")
  if _file_binding(receipt_path) != receipt_binding:
    raise ValueError("full-domain receipt changed during verification")
  for binding in expected_inputs.values():
    if _file_binding(binding["path"]) != binding:
      raise ValueError("full-domain input changed during verification")
  for split in ALLOWED_SPLITS:
    pilot_authority._reverify_live_source_bytes(
        private_supervision=replays[split].private_supervision,
        case_ids=replays[split].public_case_ids,
        archive_root=archive_root,
        restored_archive_root=restored_archive_root,
    )
  return _issue_authenticated_full_domain(
      replays=replays, receipt_sha256=receipt_binding["sha256"],
  )
