"""Development-only query-scoped mate semantic replay authority.

The V19 geometry manifest chooses a fixed query domain, but it is not mate
program authority.  This module reopens the private source/gold sidecars,
authenticated Fusion-to-STEP face map, assembly JSON, and STEP bytes and then
reconstructs the real miner rows for every selected source ordinal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from types import MappingProxyType
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .benchmark_v2_private_supervision import (
    load_benchmark_v2_private_supervision,
)
from .benchmark_v2_training_provenance import (
    canonical_sha256,
)
from .build_interface_dataset import (
    load_benchmark_v2_authenticated_training_inputs,
)
from .mate_pose_miner import replay_benchmark_v2_query_mate_semantics


SCHEMA_VERSION = "neurocad_query_mate_semantic_authority.v1"
V19_SCHEMA_VERSION = "neurocad_v19_query_certification_manifest.v1"
PENDING_STATUS = "geometry_certified_pending_semantic"
_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "query_mate_semantic_authority_v1.schema.json"
)


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _require_live_split_hashes(
    *,
    family_split_manifest_path: str | Path,
    public_cases_path: str | Path,
    family_split_manifest_sha256: str,
    source_split_sha256: str,
) -> None:
  if _sha256_file(family_split_manifest_path) != family_split_manifest_sha256:
    raise ValueError("family split manifest live SHA256 differs")
  if _sha256_file(public_cases_path) != source_split_sha256:
    raise ValueError("public source split live SHA256 differs")


def _require_complete_v2_6_face_map_receipt(payload: Any) -> None:
  cases = payload.get("cases") if isinstance(payload, Mapping) else None
  if (
      not isinstance(payload, Mapping)
      or payload.get("schema_version") != "fusion_step_face_map_audit.v2_6"
      or not isinstance(cases, list)
      or not cases
      or any(
          not isinstance(case, Mapping)
          or case.get("status") != "mapped_complete"
          for case in cases
      )
  ):
    raise ValueError(
        "query authority rejects legacy or partial mapper receipts"
    )


def _file_binding(path: str | Path) -> dict[str, Any]:
  resolved = Path(path).resolve(strict=True)
  if not resolved.is_file() or resolved.is_symlink():
    raise ValueError("query authority input must be a plain file")
  return {
      "path": str(resolved),
      "bytes": resolved.stat().st_size,
      "sha256": _sha256_file(resolved),
  }


def _read_json(path: str | Path, *, label: str) -> Any:
  try:
    return json.loads(Path(path).read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not canonical readable JSON") from error


def _validate_development_pilot_public_binding(
    *,
    public_cases_path: str | Path,
    family_split_manifest_path: str | Path,
    case_ids: Sequence[str],
) -> None:
  manifest = _read_json(
      family_split_manifest_path, label="development family manifest",
  )
  if (
      not isinstance(manifest, Mapping)
      or manifest.get("development") is not True
      or manifest.get("pilot_only") is not True
      or manifest.get("formal") is not False
      or manifest.get("publication_eligible") is not False
      or manifest.get("final_test_generated") is not False
  ):
    raise ValueError("development family pilot boundary differs")
  artifact = manifest.get("artifact_hashes", {}).get("train")
  binding = _file_binding(public_cases_path)
  if (
      not isinstance(artifact, Mapping)
      or artifact.get("sha256") != binding["sha256"]
      or artifact.get("bytes") != binding["bytes"]
      or str(Path(str(artifact.get("path") or "")).resolve(strict=True))
      != binding["path"]
      or manifest.get("split_case_counts", {}).get("train")
      != len(case_ids)
  ):
    raise ValueError("development public split artifact binding differs")


def _validate_v19_development_boundary(manifest: Any) -> Mapping[str, Any]:
  if not isinstance(manifest, Mapping):
    raise ValueError("V19 query manifest root differs")
  if (
      manifest.get("schema_version") != V19_SCHEMA_VERSION
      or manifest.get("development") is not True
      or manifest.get("formal") is not False
      or manifest.get("publication_eligible") is not False
      or manifest.get("final_test_touched") is not False
      or manifest.get("withheld_test_touched") is not False
  ):
    raise ValueError("V19 query manifest development boundary differs")
  observed_hash = manifest.get("manifest_payload_sha256")
  unsigned = dict(manifest)
  unsigned.pop("manifest_payload_sha256", None)
  if observed_hash != canonical_sha256(unsigned):
    raise ValueError("V19 query manifest self-hash differs")
  policy = manifest.get("overlay_selection_policy")
  if (
      not isinstance(policy, Mapping)
      or policy.get("mode")
      != "development_outcome_informed_explicit_allowlist"
      or policy.get("formal_promotion_prohibited") is not True
  ):
    raise ValueError("V19 overlay selection policy differs")
  return manifest


def derive_v19_overlay_query_domain(
    manifest: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], str]:
  """Take every pending-semantic row of every allowlisted overlay case."""

  _validate_v19_development_boundary(manifest)
  allowlist = manifest["overlay_selection_policy"].get("allowlist")
  if not isinstance(allowlist, list) or not allowlist:
    raise ValueError("V19 overlay allowlist is empty or malformed")
  overlay_case_ids: set[str] = set()
  for row in allowlist:
    if not isinstance(row, Mapping):
      raise ValueError("V19 overlay allowlist row differs")
    case_id = str(row.get("case_id") or "")
    if not case_id:
      raise ValueError("V19 overlay allowlist case_id differs")
    overlay_case_ids.add(case_id)
  domains = manifest.get("domains")
  query_rows = domains.get("query_rows") if isinstance(domains, Mapping) else None
  if not isinstance(query_rows, list):
    raise ValueError("V19 query row domain differs")
  selected: list[dict[str, Any]] = []
  seen: set[tuple[str, int]] = set()
  for raw_row in query_rows:
    if not isinstance(raw_row, Mapping):
      raise ValueError("V19 query row differs")
    key = raw_row.get("key")
    if not isinstance(key, Mapping):
      raise ValueError("V19 query key differs")
    case_id = str(key.get("case_id") or "")
    ordinal = key.get("source_contact_ordinal")
    if (
        case_id not in overlay_case_ids
        or raw_row.get("status") != PENDING_STATUS
    ):
      continue
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or raw_row.get("split") != "train"
    ):
      raise ValueError("V19 selected query key differs")
    query_key = (case_id, ordinal)
    if query_key in seen:
      raise ValueError("V19 selected query key is duplicated")
    seen.add(query_key)
    selected.append(copy.deepcopy(dict(raw_row)))
  selected.sort(
      key=lambda row: (
          str(row["key"]["case_id"]),
          int(row["key"]["source_contact_ordinal"]),
      )
  )
  if not selected:
    raise ValueError("V19 overlay query domain is empty")
  domain_preimage = [
      {
          "case_id": row["key"]["case_id"],
          "source_contact_ordinal": row["key"]["source_contact_ordinal"],
          "row_payload_sha256": row.get("row_payload_sha256"),
      }
      for row in selected
  ]
  return tuple(selected), canonical_sha256(domain_preimage)


def _selectors(
    query_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
  result: dict[str, list[int]] = {}
  for row in query_rows:
    key = row["key"]
    result.setdefault(str(key["case_id"]), []).append(
        int(key["source_contact_ordinal"])
    )
  return result


def _gold_contact(
    private_supervision: Any,
    *,
    case_id: str,
    ordinal: int,
) -> dict[str, Any]:
  contacts = private_supervision.gold_case(case_id).get("contacts")
  matches = [
      copy.deepcopy(dict(contact))
      for contact in contacts or []
      if isinstance(contact, Mapping)
      and contact.get("source_contact_ordinal") == ordinal
  ]
  if len(matches) != 1:
    raise ValueError("private gold query ordinal is missing or duplicated")
  return matches[0]


def _mapped_endpoint_commitment(
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    case_id: str,
    endpoint: Mapping[str, Any],
) -> dict[str, Any]:
  key = (
      case_id,
      str(endpoint.get("part") or ""),
      int(endpoint.get("fusion_face_index")),
  )
  try:
    return copy.deepcopy(dict(face_map_lookup[key]))
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError("authenticated face-map endpoint is absent") from error


def _join_query_program_rows(
    replay: Any,
) -> dict[tuple[str, int], tuple[Mapping[str, Any], ...]]:
  """Join miner rows to source queries through explicit opaque case identity."""

  raw_case_map = getattr(replay, "opaque_case_id_by_case_id", None)
  if not isinstance(raw_case_map, Mapping):
    raise ValueError("query replay lacks its opaque case identity map")
  opaque_by_original = {
      str(case_id): str(opaque_case_id)
      for case_id, opaque_case_id in raw_case_map.items()
  }
  ledger_by_opaque_key: dict[
      tuple[str, int], tuple[str, Mapping[str, Any]]
  ] = {}
  for ledger in replay.query_ledger:
    if not isinstance(ledger, Mapping):
      raise ValueError("query replay ledger row differs")
    case_id = str(ledger.get("case_id") or "")
    ordinal = ledger.get("source_contact_ordinal")
    opaque_case_id = opaque_by_original.get(case_id)
    if (
        not opaque_case_id
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
    ):
      raise ValueError("query replay ledger identity differs")
    opaque_key = (opaque_case_id, ordinal)
    if opaque_key in ledger_by_opaque_key:
      raise ValueError("query replay ledger identity is duplicated")
    ledger_by_opaque_key[opaque_key] = (case_id, ledger)

  rows_by_opaque_key: dict[
      tuple[str, int], list[Mapping[str, Any]]
  ] = {key: [] for key in ledger_by_opaque_key}
  seen_program_rows: set[tuple[str, int, str, str]] = set()
  for row in replay.rows:
    if not isinstance(row, Mapping):
      raise ValueError("query replay program row differs")
    source_contact = row.get("source_contact")
    if not isinstance(source_contact, Mapping):
      raise ValueError("query replay program source contact differs")
    opaque_case_id = str(row.get("case_id") or "")
    ordinal = source_contact.get("contact_index")
    direction = str(source_contact.get("direction") or "")
    program_id = str(row.get("program_id") or "")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or not opaque_case_id
        or not direction
        or not program_id
    ):
      raise ValueError("query replay program identity differs")
    opaque_key = (opaque_case_id, ordinal)
    if opaque_key not in rows_by_opaque_key:
      raise ValueError("query replay contains a program outside its ledger")
    row_key = (opaque_case_id, ordinal, direction, program_id)
    if row_key in seen_program_rows:
      raise ValueError("query replay contains a duplicate program row")
    seen_program_rows.add(row_key)
    rows_by_opaque_key[opaque_key].append(row)

  joined: dict[tuple[str, int], tuple[Mapping[str, Any], ...]] = {}
  for opaque_key, (case_id, ledger) in ledger_by_opaque_key.items():
    rows = sorted(
        rows_by_opaque_key[opaque_key],
        key=lambda row: (
            str(row["source_contact"]["direction"]),
            str(row["program_id"]),
        ),
    )
    expected_count = ledger.get("directed_program_count")
    status = ledger.get("status")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(rows)
        or (status == "authorized") != bool(rows)
        or status not in {"authorized", "rejected"}
    ):
      raise ValueError("query replay program/ledger coverage differs")
    joined[(case_id, opaque_key[1])] = tuple(rows)
  if sum(map(len, joined.values())) != len(replay.rows):
    raise ValueError("query replay program rows are missing or duplicated")
  return joined


def _query_commitments(
    *,
    domain_rows: Sequence[Mapping[str, Any]],
    private_supervision: Any,
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    replay: Any,
    frame_seed: int,
) -> list[dict[str, Any]]:
  replay_rows = _join_query_program_rows(replay)
  opaque_case_ids = {
      str(case_id): str(opaque_case_id)
      for case_id, opaque_case_id in (
          replay.opaque_case_id_by_case_id.items()
      )
  }
  commitments: list[dict[str, Any]] = []

  domain_by_key = {
      (
          str(row["key"]["case_id"]),
          int(row["key"]["source_contact_ordinal"]),
      ): row
      for row in domain_rows
  }
  for ledger in replay.query_ledger:
    case_id = str(ledger["case_id"])
    ordinal = int(ledger["source_contact_ordinal"])
    domain_row = domain_by_key[(case_id, ordinal)]
    contact = _gold_contact(
        private_supervision, case_id=case_id, ordinal=ordinal,
    )
    if domain_row.get("source_contact") != contact:
      raise ValueError("V19 source contact differs from private gold ordinal")
    authenticated_mapped = {
        f"endpoint_{role}": _mapped_endpoint_commitment(
            face_map_lookup,
            case_id=case_id,
            endpoint=contact[f"endpoint_{role}"],
        )
        for role in ("a", "b")
    }
    expected_mapped = domain_row.get("endpoint_identity")
    if not isinstance(expected_mapped, Mapping):
      raise ValueError("V19 mapped endpoint commitment is absent")
    mapped: dict[str, Any] = {}
    for role in ("a", "b"):
      endpoint_key = f"endpoint_{role}"
      expected_endpoint = expected_mapped.get(endpoint_key)
      actual_endpoint = authenticated_mapped[endpoint_key]
      if not isinstance(expected_endpoint, Mapping):
        raise ValueError("V19 mapped endpoint projection differs")
      expected_core = dict(expected_endpoint)
      endpoint_role = expected_core.pop("endpoint_role", role)
      if endpoint_role != role or any(
          key not in actual_endpoint or actual_endpoint[key] != value
          for key, value in expected_core.items()
      ):
        raise ValueError(
            "V19 mapped endpoint commitment differs from authenticated face map"
        )
      mapped[endpoint_key] = {
          "v19_endpoint_identity": copy.deepcopy(dict(expected_endpoint)),
          "development_mapper_lookup_sha256": canonical_sha256(
              actual_endpoint
          ),
      }
    directions: list[dict[str, Any]] = []
    for program_row in replay_rows[(case_id, ordinal)]:
      residual = {
          "rotation": program_row.get("residual_rotation"),
          "translation": program_row.get("residual_translation"),
      }
      directions.append(
          {
              "direction": program_row["source_contact"]["direction"],
              "program_id": program_row["program_id"],
              "residual_sha256": canonical_sha256(residual),
              "program_row_sha256": canonical_sha256(program_row),
              "program_row": copy.deepcopy(dict(program_row)),
          }
      )
    directions.sort(
        key=lambda row: (row["direction"], row["program_id"])
    )
    source_case = private_supervision.source_case(case_id)
    receipt_binding = copy.deepcopy(
        dict(source_case["source_receipt_binding"])
    )
    commitments.append(
        {
            "case_id": case_id,
            "source_contact_ordinal": ordinal,
            "status": ledger["status"],
            "rejection_reason": ledger["reason"],
            "source_contact": contact,
            "source_endpoint_identities_sha256": canonical_sha256(
                {
                    "endpoint_a": contact["endpoint_a"],
                    "endpoint_b": contact["endpoint_b"],
                }
            ),
            "mapped_face_commitments": mapped,
            "mapped_face_commitments_sha256": canonical_sha256(mapped),
            "assembly_step_bytes_commitment": receipt_binding,
            "assembly_step_bytes_commitment_sha256": canonical_sha256(
                receipt_binding
            ),
            "frame_randomization_seed": frame_seed,
            "opaque_program_case_id": opaque_case_ids[case_id],
            "programs": directions,
        }
    )
  return commitments


def _producer_binding() -> dict[str, Any]:
  root = Path(__file__).resolve().parent
  paths = {
      "neurocad.query_mate_semantic_authority_v1": Path(__file__),
      "neurocad.mate_pose_miner": root / "mate_pose_miner.py",
      "schema": _SCHEMA_PATH,
  }
  return {
      "implementation": {
          name: _file_binding(path)
          for name, path in sorted(paths.items())
      },
      "replay_contract": (
          "private_gold_ordinal_authenticated_face_map_assembly_step_"
          "frame_residual_program_replay.v1"
      ),
  }


@dataclass(frozen=True, slots=True)
class _ComputedAuthority:
  payload: dict[str, Any]
  program_rows: tuple[Mapping[str, Any], ...]
  query_ledger: tuple[Mapping[str, Any], ...]


def _compute_authority(
    *,
    v19_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    family_split_manifest_path: str | Path,
    face_map_receipt_path: str | Path | None = None,
    face_map_receipt_paths: Mapping[str, str | Path] | None = None,
    archive_root: str | Path,
    restored_archive_root: str | Path | None = None,
    mining_config: Mapping[str, Any],
    family_split_manifest_sha256: str,
    source_split_sha256: str,
) -> _ComputedAuthority:
  if not isinstance(mining_config, Mapping):
    raise ValueError("query authority mining config differs")
  _require_live_split_hashes(
      family_split_manifest_path=family_split_manifest_path,
      public_cases_path=public_cases_path,
      family_split_manifest_sha256=family_split_manifest_sha256,
      source_split_sha256=source_split_sha256,
  )
  manifest = _validate_v19_development_boundary(
      _read_json(v19_manifest_path, label="V19 query manifest")
  )
  domain_rows, domain_sha256 = derive_v19_overlay_query_domain(manifest)
  if face_map_receipt_paths is not None:
    from .fusion_face_mapper import build_mapped_endpoint_lookup

    selected_case_ids = {
        str(row["key"]["case_id"]) for row in domain_rows
    }
    if set(face_map_receipt_paths) != selected_case_ids:
      raise ValueError("query authority mapper receipt case domain differs")
    v19_bindings: dict[str, Mapping[str, Any]] = {}
    for binding in manifest.get("inputs", {}).get("mapper_receipts", []):
      if isinstance(binding, Mapping):
        v19_bindings[str(binding.get("case_id") or "")] = binding
    for binding in manifest.get("inputs", {}).get("overlay_receipts", []):
      if (
          isinstance(binding, Mapping)
          and binding.get("overlay_role") == "mapper"
      ):
        v19_bindings[str(binding.get("case_id") or "")] = binding
    face_map_lookup: dict[
        tuple[str, str, int], Mapping[str, Any]
    ] = {}
    for case_id, raw_path in sorted(face_map_receipt_paths.items()):
      binding = v19_bindings.get(case_id)
      if (
          not isinstance(binding, Mapping)
          or _file_binding(raw_path)["sha256"] != binding.get("sha256")
          or str(Path(raw_path).resolve(strict=True))
          != str(Path(str(binding.get("path") or "")).resolve(strict=True))
      ):
        raise ValueError("query authority mapper/V19 byte binding differs")
      mapper_payload = _read_json(
          raw_path, label="authenticated mapper receipt",
      )
      mapper_cases = (
          mapper_payload.get("cases")
          if isinstance(mapper_payload, Mapping)
          else None
      )
      if (
          not isinstance(mapper_payload, Mapping)
          or mapper_payload.get("schema_version")
          != "fusion_step_face_map_audit.v2_6"
          or not isinstance(mapper_cases, list)
          or len(mapper_cases) != 1
          or mapper_cases[0].get("case_id") != case_id
          or mapper_cases[0].get("status") != "mapped_complete"
      ):
        raise ValueError(
            "query authority rejects legacy or partial mapper receipts"
        )
      lookup = build_mapped_endpoint_lookup(mapper_payload)
      overlap = set(face_map_lookup).intersection(lookup)
      if overlap:
        raise ValueError("query authority mapper lookup keys overlap")
      face_map_lookup.update(lookup)
    public_cases = _read_json(public_cases_path, label="public cases")
    if not isinstance(public_cases, list) or not public_cases:
      raise ValueError("query authority public cases differ")
    case_ids = tuple(str(case.get("id") or "") for case in public_cases)
    _validate_development_pilot_public_binding(
        public_cases_path=public_cases_path,
        family_split_manifest_path=family_split_manifest_path,
        case_ids=case_ids,
    )
    private_supervision = load_benchmark_v2_private_supervision(
        source_path=private_source_path,
        gold_path=private_gold_path,
        family_split_manifest_path=family_split_manifest_path,
        source_split="train",
        public_case_ids=case_ids,
        pilot_only=True,
    )
    family_source_binding = manifest.get("inputs", {}).get("family_source")
    if not isinstance(family_source_binding, Mapping):
      raise ValueError("V19 family source binding differs")
    family_source_path = Path(
        str(family_source_binding.get("path") or "")
    ).resolve(strict=True)
    if (
        _file_binding(family_source_path)["sha256"]
        != family_source_binding.get("sha256")
    ):
      raise ValueError("V19 family source bytes differ")
    family_source = _read_json(
        family_source_path, label="V19 bound family source",
    )
    source_by_case = {
        str(row.get("case_id") or ""): row
        for row in family_source.get(
            "private_source_bindings", {}
        ).get("cases", [])
        if isinstance(row, Mapping)
    }
    gold_by_case = {
        str(row.get("case_id") or ""): row
        for row in family_source.get(
            "evaluation_gold_contacts", {}
        ).get("cases", [])
        if isinstance(row, Mapping)
    }
    public_by_case = {
        str(row.get("id") or ""): row
        for row in family_source.get("cases", [])
        if isinstance(row, Mapping)
    }
    for public_case in public_cases:
      case_id = str(public_case.get("id") or "")
      source_row = source_by_case.get(case_id)
      gold_row = gold_by_case.get(case_id)
      public_row = public_by_case.get(case_id)
      if (
          not isinstance(source_row, Mapping)
          or not isinstance(gold_row, Mapping)
          or not isinstance(public_row, Mapping)
          or private_supervision.source_case(case_id) != source_row
          or private_supervision.gold_case(case_id).get("contacts")
          != gold_row.get("contacts")
          or private_supervision.gold_case(case_id).get(
              "source_statistics"
          )
          != gold_row.get("source_statistics")
          or dict(public_case)
          != {
              key: copy.deepcopy(value)
              for key, value in public_row.items()
              if key != "family_provenance"
          }
      ):
        raise ValueError(
            "development sidecars differ from the V19-bound private source"
        )
    authenticated = SimpleNamespace(
        public_cases=tuple(dict(case) for case in public_cases),
        private_supervision=private_supervision,
        face_map_lookup=face_map_lookup,
    )
  elif face_map_receipt_path is None:
    raise ValueError("query authority requires authenticated mapper receipts")
  else:
    face_payload = _read_json(
        face_map_receipt_path, label="authenticated face-map receipt",
    )
    _require_complete_v2_6_face_map_receipt(face_payload)
    authenticated = load_benchmark_v2_authenticated_training_inputs(
        public_cases_path=public_cases_path,
        private_source_path=private_source_path,
        private_gold_path=private_gold_path,
        family_split_manifest_path=family_split_manifest_path,
        face_map_receipt_path=face_map_receipt_path,
        source_split="train",
    )
  archive_resolver = None
  if restored_archive_root is not None:
    from .formal_archive_inputs import resolve_case_archive_root

    def resolve_archive(source_case: Mapping[str, Any]) -> Path:
      _archive, resolved = resolve_case_archive_root(
          source_case,
          restored_archive_root=Path(restored_archive_root),
      )
      return resolved

    archive_resolver = resolve_archive
  replay = replay_benchmark_v2_query_mate_semantics(
      public_cases=authenticated.public_cases,
      private_supervision=authenticated.private_supervision,
      face_map_lookup=authenticated.face_map_lookup,
      archive_root=archive_root,
      mining_config=dict(mining_config),
      family_split_manifest_sha256=family_split_manifest_sha256,
      source_split_sha256=source_split_sha256,
      source_contact_ordinals=_selectors(domain_rows),
      _case_archive_root_resolver=archive_resolver,
  )
  if replay.input_domain_sha256 != canonical_sha256(
      [
          {
              "case_id": row["key"]["case_id"],
              "source_contact_ordinal": row["key"]["source_contact_ordinal"],
          }
          for row in domain_rows
      ]
  ):
    raise ValueError("miner query selector domain commitment differs")
  frame_seed = mining_config.get("frame_randomization_seed")
  if isinstance(frame_seed, bool) or not isinstance(frame_seed, int):
    raise ValueError("query authority frame seed differs")
  query_commitments = _query_commitments(
      domain_rows=domain_rows,
      private_supervision=authenticated.private_supervision,
      face_map_lookup=authenticated.face_map_lookup,
      replay=replay,
      frame_seed=frame_seed,
  )
  input_paths = {
      "v19_query_manifest": v19_manifest_path,
      "public_cases": public_cases_path,
      "private_source": private_source_path,
      "private_gold": private_gold_path,
      "family_split_manifest": family_split_manifest_path,
  }
  if face_map_receipt_paths is None:
    input_paths["authenticated_face_map"] = face_map_receipt_path
  else:
    input_paths["v19_bound_family_source"] = family_source_path
    input_paths.update(
        {
            f"development_mapper_receipt:{case_id}": path
            for case_id, path in sorted(face_map_receipt_paths.items())
        }
    )
  authorized = sum(row["status"] == "authorized" for row in query_commitments)
  payload = {
      "schema_version": SCHEMA_VERSION,
      "development": True,
      "formal": False,
      "publication_eligible": False,
      "final_test_touched": False,
      "withheld_test_touched": False,
      "authority_source": "development_query_replay_cross_checked",
      "mapper_authority_level": (
          "development_training_label_receipt_with_live_step_replay"
      ),
      "formal_promotion_blocked": True,
      "input_domain": {
          "selection": (
              "all_pending_semantic_rows_for_all_overlay_allowlist_cases"
          ),
          "selected_query_count": len(domain_rows),
          "commitment_sha256": domain_sha256,
          "miner_selector_commitment_sha256": replay.input_domain_sha256,
      },
      "inputs": {
          name: _file_binding(path)
          for name, path in sorted(input_paths.items())
      },
      "mining_config": copy.deepcopy(dict(mining_config)),
      "mining_config_sha256": canonical_sha256(dict(mining_config)),
      "family_split_manifest_sha256": family_split_manifest_sha256,
      "source_split_sha256": source_split_sha256,
      "producer": _producer_binding(),
      "query_commitments": query_commitments,
      "coverage": {
          "selected_query_count": len(query_commitments),
          "authorized_query_count": authorized,
          "rejected_query_count": len(query_commitments) - authorized,
          "program_row_count": len(replay.rows),
      },
  }
  return _ComputedAuthority(
      payload=payload,
      program_rows=tuple(replay.rows),
      query_ledger=tuple(replay.query_ledger),
  )


def build_query_mate_semantic_authority_v1_receipt(
    *,
    receipt_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
  """Produce a development receipt; the receipt itself grants no authority."""

  computed = _compute_authority(**kwargs)
  receipt = copy.deepcopy(computed.payload)
  receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
  destination = Path(receipt_path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  temporary.write_text(
      json.dumps(receipt, ensure_ascii=False, sort_keys=True),
      encoding="utf-8",
  )
  temporary.replace(destination)
  return receipt


def _recursive_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType(
        {key: _recursive_freeze(item) for key, item in value.items()}
    )
  if isinstance(value, (list, tuple)):
    return tuple(_recursive_freeze(item) for item in value)
  return value


def _make_authenticated_handle_type():
  factory_token = object()

  class AuthenticatedQueryMateSemanticAuthority:
    __slots__ = ("_program_rows", "_query_ledger", "_receipt_sha256", "_sealed")

    def __init__(
        self,
        *,
        program_rows: Sequence[Mapping[str, Any]],
        query_ledger: Sequence[Mapping[str, Any]],
        receipt_sha256: str,
        _token: object,
    ) -> None:
      if _token is not factory_token:
        raise TypeError("query mate semantic authority is verifier-only")
      object.__setattr__(self, "_program_rows", _recursive_freeze(program_rows))
      object.__setattr__(self, "_query_ledger", _recursive_freeze(query_ledger))
      object.__setattr__(self, "_receipt_sha256", receipt_sha256)
      object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: Any) -> None:
      if getattr(self, "_sealed", False):
        raise AttributeError("authenticated query authority is immutable")
      object.__setattr__(self, _name, _value)

    @property
    def program_rows(self) -> tuple[Mapping[str, Any], ...]:
      return self._program_rows

    @property
    def query_ledger(self) -> tuple[Mapping[str, Any], ...]:
      return self._query_ledger

    @property
    def receipt_sha256(self) -> str:
      return self._receipt_sha256

    def __reduce_ex__(self, _protocol: int) -> Any:
      raise TypeError("authenticated query authority is not serializable")

  def issue(
      *,
      program_rows: Sequence[Mapping[str, Any]],
      query_ledger: Sequence[Mapping[str, Any]],
      receipt_sha256: str,
  ) -> AuthenticatedQueryMateSemanticAuthority:
    return AuthenticatedQueryMateSemanticAuthority(
        program_rows=program_rows,
        query_ledger=query_ledger,
        receipt_sha256=receipt_sha256,
        _token=factory_token,
    )

  return AuthenticatedQueryMateSemanticAuthority, issue


(
    AuthenticatedQueryMateSemanticAuthority,
    _issue_authenticated_query_authority,
) = _make_authenticated_handle_type()


def _archive_resolver(
    restored_archive_root: str | Path | None,
) -> Any:
  if restored_archive_root is None:
    return None
  from .formal_archive_inputs import resolve_case_archive_root

  def resolve(source_case: Mapping[str, Any]) -> Path:
    _archive, root = resolve_case_archive_root(
        source_case,
        restored_archive_root=Path(restored_archive_root),
    )
    return root

  return resolve


def _reverify_live_source_bytes(
    *,
    private_supervision: Any,
    case_ids: Sequence[str],
    archive_root: str | Path,
    restored_archive_root: str | Path | None,
) -> None:
  resolver = _archive_resolver(restored_archive_root)
  for case_id in case_ids:
    source_case = private_supervision.source_case(case_id)
    root = (
        Path(archive_root).resolve(strict=True)
        if resolver is None
        else resolver(source_case)
    )
    receipt = source_case["source_receipt_binding"]
    bindings = [
        receipt["assembly_json"],
        *receipt["body_steps"],
    ]
    for binding in bindings:
      path = root / Path(str(binding["path"]))
      if not path.is_file() or path.is_symlink():
        raise ValueError("live assembly/STEP input is not a plain file")
      resolved = path.resolve(strict=True)
      if (
          resolved.stat().st_size != binding["bytes"]
          or _sha256_file(resolved) != binding["sha256"]
      ):
        raise ValueError("live assembly/STEP bytes changed before issuance")


def _independent_mapper_context(
    *,
    manifest: Mapping[str, Any],
    domain_rows: Sequence[Mapping[str, Any]],
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    family_split_manifest_path: str | Path,
    face_map_receipt_paths: Mapping[str, str | Path],
) -> tuple[Any, Path]:
  from .fusion_face_mapper import build_mapped_endpoint_lookup

  selected_case_ids = {
      str(row["key"]["case_id"]) for row in domain_rows
  }
  if set(face_map_receipt_paths) != selected_case_ids:
    raise ValueError("verifier mapper receipt case domain differs")
  v19_bindings: dict[str, Mapping[str, Any]] = {}
  for binding in manifest.get("inputs", {}).get("mapper_receipts", []):
    if isinstance(binding, Mapping):
      v19_bindings[str(binding.get("case_id") or "")] = binding
  for binding in manifest.get("inputs", {}).get("overlay_receipts", []):
    if (
        isinstance(binding, Mapping)
        and binding.get("overlay_role") == "mapper"
    ):
      v19_bindings[str(binding.get("case_id") or "")] = binding
  lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
  for case_id, raw_path in sorted(face_map_receipt_paths.items()):
    file_binding = _file_binding(raw_path)
    v19_binding = v19_bindings.get(case_id)
    if (
        not isinstance(v19_binding, Mapping)
        or file_binding["sha256"] != v19_binding.get("sha256")
        or file_binding["bytes"] != v19_binding.get("bytes")
        or file_binding["path"]
        != str(Path(str(v19_binding.get("path") or "")).resolve(strict=True))
    ):
      raise ValueError("verifier mapper/V19 byte binding differs")
    mapper = _read_json(raw_path, label="verifier mapper receipt")
    cases = mapper.get("cases") if isinstance(mapper, Mapping) else None
    if (
        not isinstance(mapper, Mapping)
        or mapper.get("schema_version")
        != "fusion_step_face_map_audit.v2_6"
        or not isinstance(cases, list)
        or len(cases) != 1
        or cases[0].get("case_id") != case_id
        or cases[0].get("status") != "mapped_complete"
    ):
      raise ValueError("verifier rejects legacy or partial mapper receipt")
    case_lookup = build_mapped_endpoint_lookup(mapper)
    if set(lookup).intersection(case_lookup):
      raise ValueError("verifier mapper lookup keys overlap")
    lookup.update(case_lookup)

  public_cases = _read_json(public_cases_path, label="verifier public cases")
  if not isinstance(public_cases, list) or not public_cases:
    raise ValueError("verifier public cases differ")
  case_ids = tuple(str(case.get("id") or "") for case in public_cases)
  _validate_development_pilot_public_binding(
      public_cases_path=public_cases_path,
      family_split_manifest_path=family_split_manifest_path,
      case_ids=case_ids,
  )
  supervision = load_benchmark_v2_private_supervision(
      source_path=private_source_path,
      gold_path=private_gold_path,
      family_split_manifest_path=family_split_manifest_path,
      source_split="train",
      public_case_ids=case_ids,
      pilot_only=True,
  )
  family_binding = manifest.get("inputs", {}).get("family_source")
  if not isinstance(family_binding, Mapping):
    raise ValueError("verifier V19 family source binding differs")
  family_source_path = Path(
      str(family_binding.get("path") or "")
  ).resolve(strict=True)
  if (
      _file_binding(family_source_path)["sha256"]
      != family_binding.get("sha256")
  ):
    raise ValueError("verifier V19 family source bytes differ")
  family_source = _read_json(
      family_source_path, label="verifier V19 family source",
  )
  source_by_case = {
      str(row.get("case_id") or ""): row
      for row in family_source.get(
          "private_source_bindings", {}
      ).get("cases", [])
      if isinstance(row, Mapping)
  }
  gold_by_case = {
      str(row.get("case_id") or ""): row
      for row in family_source.get(
          "evaluation_gold_contacts", {}
      ).get("cases", [])
      if isinstance(row, Mapping)
  }
  public_by_case = {
      str(row.get("id") or ""): row
      for row in family_source.get("cases", [])
      if isinstance(row, Mapping)
  }
  for public_case in public_cases:
    case_id = str(public_case.get("id") or "")
    source = source_by_case.get(case_id)
    gold = gold_by_case.get(case_id)
    original_public = public_by_case.get(case_id)
    if (
        supervision.source_case(case_id) != source
        or supervision.gold_case(case_id).get("contacts")
        != (gold or {}).get("contacts")
        or supervision.gold_case(case_id).get("source_statistics")
        != (gold or {}).get("source_statistics")
        or dict(public_case)
        != {
            key: copy.deepcopy(value)
            for key, value in (original_public or {}).items()
            if key != "family_provenance"
        }
    ):
      raise ValueError("verifier sidecars differ from V19 family source")
  return (
      SimpleNamespace(
          public_cases=tuple(dict(case) for case in public_cases),
          private_supervision=supervision,
          face_map_lookup=lookup,
      ),
      family_source_path,
  )


def _verify_query_commitments_independently(
    *,
    receipt_rows: Any,
    domain_rows: Sequence[Mapping[str, Any]],
    private_supervision: Any,
    face_map_lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    replay: Any,
    frame_seed: int,
) -> None:
  if not isinstance(receipt_rows, list):
    raise ValueError("receipt query commitments differ")
  observed: dict[tuple[str, int], Mapping[str, Any]] = {}
  for row in receipt_rows:
    if not isinstance(row, Mapping):
      raise ValueError("receipt query commitment differs")
    key = (str(row.get("case_id") or ""), row.get("source_contact_ordinal"))
    if (
        isinstance(key[1], bool)
        or not isinstance(key[1], int)
        or key in observed
    ):
      raise ValueError("receipt query commitment key is duplicated")
    observed[key] = row
  domain_by_key = {
      (
          str(row["key"]["case_id"]),
          int(row["key"]["source_contact_ordinal"]),
      ): row
      for row in domain_rows
  }
  if set(observed) != set(domain_by_key):
    raise ValueError("receipt query commitment coverage differs")
  ledger_by_key = {
      (
          str(row["case_id"]),
          int(row["source_contact_ordinal"]),
      ): row
      for row in replay.query_ledger
  }
  if set(ledger_by_key) != set(domain_by_key):
    raise ValueError("verifier replay ledger domain differs")
  program_rows = _join_query_program_rows(replay)
  opaque_ids = dict(replay.opaque_case_id_by_case_id)
  for key, domain_row in domain_by_key.items():
    case_id, ordinal = key
    ledger = ledger_by_key[key]
    contact = _gold_contact(
        private_supervision, case_id=case_id, ordinal=ordinal,
    )
    if domain_row.get("source_contact") != contact:
      raise ValueError("verifier V19/private gold contact differs")
    expected_endpoint_identity = domain_row.get("endpoint_identity")
    if not isinstance(expected_endpoint_identity, Mapping):
      raise ValueError("verifier V19 endpoint identity is absent")
    mapped_commitments: dict[str, Any] = {}
    for role in ("a", "b"):
      endpoint_key = f"endpoint_{role}"
      actual_mapping = _mapped_endpoint_commitment(
          face_map_lookup,
          case_id=case_id,
          endpoint=contact[endpoint_key],
      )
      expected_mapping = expected_endpoint_identity.get(endpoint_key)
      if not isinstance(expected_mapping, Mapping):
        raise ValueError("verifier V19 endpoint projection differs")
      core = dict(expected_mapping)
      if core.pop("endpoint_role", role) != role or any(
          actual_mapping.get(field) != value
          for field, value in core.items()
      ):
        raise ValueError("verifier mapper/V19 endpoint differs")
      mapped_commitments[endpoint_key] = {
          "v19_endpoint_identity": copy.deepcopy(dict(expected_mapping)),
          "development_mapper_lookup_sha256": canonical_sha256(
              actual_mapping
          ),
      }
    programs = []
    for program in program_rows[key]:
      residual = {
          "rotation": program.get("residual_rotation"),
          "translation": program.get("residual_translation"),
      }
      programs.append(
          {
              "direction": program["source_contact"]["direction"],
              "program_id": program["program_id"],
              "residual_sha256": canonical_sha256(residual),
              "program_row_sha256": canonical_sha256(program),
              "program_row": copy.deepcopy(dict(program)),
          }
      )
    programs.sort(key=lambda row: (row["direction"], row["program_id"]))
    source_binding = copy.deepcopy(
        dict(
            private_supervision.source_case(case_id)[
                "source_receipt_binding"
            ]
        )
    )
    expected = {
        "case_id": case_id,
        "source_contact_ordinal": ordinal,
        "status": ledger["status"],
        "rejection_reason": ledger["reason"],
        "source_contact": contact,
        "source_endpoint_identities_sha256": canonical_sha256(
            {
                "endpoint_a": contact["endpoint_a"],
                "endpoint_b": contact["endpoint_b"],
            }
        ),
        "mapped_face_commitments": mapped_commitments,
        "mapped_face_commitments_sha256": canonical_sha256(
            mapped_commitments
        ),
        "assembly_step_bytes_commitment": source_binding,
        "assembly_step_bytes_commitment_sha256": canonical_sha256(
            source_binding
        ),
        "frame_randomization_seed": frame_seed,
        "opaque_program_case_id": opaque_ids[case_id],
        "programs": programs,
    }
    if dict(observed[key]) != expected:
      raise ValueError(
          "receipt query commitment differs from verifier replay"
      )


def verify_query_mate_semantic_authority_v1(
    *,
    receipt_path: str | Path,
    v19_manifest_path: str | Path,
    public_cases_path: str | Path,
    private_source_path: str | Path,
    private_gold_path: str | Path,
    family_split_manifest_path: str | Path,
    archive_root: str | Path,
    mining_config: Mapping[str, Any],
    family_split_manifest_sha256: str,
    source_split_sha256: str,
    face_map_receipt_path: str | Path | None = None,
    face_map_receipt_paths: Mapping[str, str | Path] | None = None,
    restored_archive_root: str | Path | None = None,
) -> AuthenticatedQueryMateSemanticAuthority:
  """Reopen every input and issue an ephemeral handle only after exact replay."""

  _require_live_split_hashes(
      family_split_manifest_path=family_split_manifest_path,
      public_cases_path=public_cases_path,
      family_split_manifest_sha256=family_split_manifest_sha256,
      source_split_sha256=source_split_sha256,
  )
  receipt_binding = _file_binding(receipt_path)
  receipt = _read_json(receipt_path, label="query authority receipt")
  if not isinstance(receipt, Mapping):
    raise ValueError("query authority receipt root differs")
  observed_hash = receipt.get("receipt_payload_sha256")
  unsigned = dict(receipt)
  unsigned.pop("receipt_payload_sha256", None)
  if observed_hash != canonical_sha256(unsigned):
    raise ValueError("query authority receipt self-hash differs")
  expected_keys = {
      "schema_version", "development", "formal", "publication_eligible",
      "final_test_touched", "withheld_test_touched", "authority_source",
      "mapper_authority_level", "formal_promotion_blocked", "input_domain",
      "inputs", "mining_config", "mining_config_sha256",
      "family_split_manifest_sha256", "source_split_sha256", "producer",
      "query_commitments", "coverage", "receipt_payload_sha256",
  }
  if (
      set(receipt) != expected_keys
      or receipt.get("schema_version") != SCHEMA_VERSION
      or receipt.get("development") is not True
      or receipt.get("formal") is not False
      or receipt.get("publication_eligible") is not False
      or receipt.get("final_test_touched") is not False
      or receipt.get("withheld_test_touched") is not False
      or receipt.get("authority_source")
      != "development_query_replay_cross_checked"
      or receipt.get("mapper_authority_level")
      != "development_training_label_receipt_with_live_step_replay"
      or receipt.get("formal_promotion_blocked") is not True
  ):
    raise ValueError(
        "surface, census, library, or manifest self-signing is prohibited"
    )
  manifest = _validate_v19_development_boundary(
      _read_json(v19_manifest_path, label="verifier V19 manifest")
  )
  domain_rows, domain_sha256 = derive_v19_overlay_query_domain(manifest)
  if face_map_receipt_paths is not None:
    authenticated, family_source_path = _independent_mapper_context(
        manifest=manifest,
        domain_rows=domain_rows,
        public_cases_path=public_cases_path,
        private_source_path=private_source_path,
        private_gold_path=private_gold_path,
        family_split_manifest_path=family_split_manifest_path,
        face_map_receipt_paths=face_map_receipt_paths,
    )
  elif face_map_receipt_path is not None:
    family_source_path = None
    _require_complete_v2_6_face_map_receipt(
        _read_json(
            face_map_receipt_path,
            label="verifier authenticated face-map receipt",
        )
    )
    authenticated = load_benchmark_v2_authenticated_training_inputs(
        public_cases_path=public_cases_path,
        private_source_path=private_source_path,
        private_gold_path=private_gold_path,
        family_split_manifest_path=family_split_manifest_path,
        face_map_receipt_path=face_map_receipt_path,
        source_split="train",
    )
  else:
    raise ValueError("verifier requires authenticated mapper inputs")
  replay = replay_benchmark_v2_query_mate_semantics(
      public_cases=authenticated.public_cases,
      private_supervision=authenticated.private_supervision,
      face_map_lookup=authenticated.face_map_lookup,
      archive_root=archive_root,
      mining_config=dict(mining_config),
      family_split_manifest_sha256=family_split_manifest_sha256,
      source_split_sha256=source_split_sha256,
      source_contact_ordinals=_selectors(domain_rows),
      _case_archive_root_resolver=_archive_resolver(
          restored_archive_root
      ),
  )
  selector_commitment = canonical_sha256(
      [
          {
              "case_id": row["key"]["case_id"],
              "source_contact_ordinal": row["key"][
                  "source_contact_ordinal"
              ],
          }
          for row in domain_rows
      ]
  )
  expected_input_domain = {
      "selection": (
          "all_pending_semantic_rows_for_all_overlay_allowlist_cases"
      ),
      "selected_query_count": len(domain_rows),
      "commitment_sha256": domain_sha256,
      "miner_selector_commitment_sha256": selector_commitment,
  }
  if receipt.get("input_domain") != expected_input_domain:
    raise ValueError("receipt input domain differs from verifier domain")
  if (
      receipt.get("mining_config") != dict(mining_config)
      or receipt.get("mining_config_sha256")
      != canonical_sha256(dict(mining_config))
      or receipt.get("family_split_manifest_sha256")
      != family_split_manifest_sha256
      or receipt.get("source_split_sha256") != source_split_sha256
      or receipt.get("producer") != _producer_binding()
  ):
    raise ValueError("receipt replay/producer binding differs")
  input_paths: dict[str, str | Path] = {
      "v19_query_manifest": v19_manifest_path,
      "public_cases": public_cases_path,
      "private_source": private_source_path,
      "private_gold": private_gold_path,
      "family_split_manifest": family_split_manifest_path,
  }
  if face_map_receipt_paths is None:
    input_paths["authenticated_face_map"] = face_map_receipt_path
  else:
    assert family_source_path is not None
    input_paths["v19_bound_family_source"] = family_source_path
    input_paths.update(
        {
            f"development_mapper_receipt:{case_id}": path
            for case_id, path in sorted(face_map_receipt_paths.items())
        }
    )
  expected_inputs = {
      name: _file_binding(path)
      for name, path in sorted(input_paths.items())
  }
  if receipt.get("inputs") != expected_inputs:
    raise ValueError("receipt input byte bindings differ")
  frame_seed = mining_config.get("frame_randomization_seed")
  if isinstance(frame_seed, bool) or not isinstance(frame_seed, int):
    raise ValueError("verifier frame seed differs")
  _verify_query_commitments_independently(
      receipt_rows=receipt.get("query_commitments"),
      domain_rows=domain_rows,
      private_supervision=authenticated.private_supervision,
      face_map_lookup=authenticated.face_map_lookup,
      replay=replay,
      frame_seed=frame_seed,
  )
  authorized = sum(
      row["status"] == "authorized" for row in replay.query_ledger
  )
  expected_coverage = {
      "selected_query_count": len(replay.query_ledger),
      "authorized_query_count": authorized,
      "rejected_query_count": len(replay.query_ledger) - authorized,
      "program_row_count": len(replay.rows),
  }
  if receipt.get("coverage") != expected_coverage:
    raise ValueError("receipt coverage differs from verifier replay")
  # Final byte checks close replacement races across both receipt and inputs.
  if _file_binding(receipt_path) != receipt_binding:
    raise ValueError("query authority receipt changed during verification")
  for binding in expected_inputs.values():
    if _file_binding(binding["path"]) != binding:
      raise ValueError("query authority input changed during verification")
  _reverify_live_source_bytes(
      private_supervision=authenticated.private_supervision,
      case_ids=tuple(str(case["id"]) for case in authenticated.public_cases),
      archive_root=archive_root,
      restored_archive_root=restored_archive_root,
  )
  return _issue_authenticated_query_authority(
      program_rows=replay.rows,
      query_ledger=replay.query_ledger,
      receipt_sha256=receipt_binding["sha256"],
  )


def _pickle_guard() -> None:
  # Keep static analyzers aware that pickle support is deliberately rejected.
  _ = pickle.HIGHEST_PROTOCOL
