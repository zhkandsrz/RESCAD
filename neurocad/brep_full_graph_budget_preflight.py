"""Receipt-bound full-STEP graph-budget preflight for the 600 development cases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from . import benchmark_v2_model_view_v2 as model_view
from .brep_program_learner_v1 import AuthenticatedShardedBRepTensorCacheV2
PREFLIGHT_SCHEMA_VERSION = "brep_full_graph_budget_preflight.v1"
PREFLIGHT_ARTIFACT_SCHEMA_VERSION = "brep_full_graph_budget_preflight_artifact.v1"
MAX_FACES = 1024
MAX_ADJACENCIES = 4096
_FACTORY_TOKEN = object()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREFLIGHT_SOURCE_NAMES = (
    "brep_full_graph_budget_preflight.py",
    "tools/preflight_unlabeled_full_graph_budget_v1.py",
    "benchmark_v2_model_view_v2.py",
    "benchmark_v2_training_provenance.py",
    "brep_program_learner_v1.py",
)


def canonical_bytes(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path, *, name: str) -> dict[str, Any]:
  return {"name": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != expected:
    raise ValueError(f"{label} keys differ")
  return value


def _strict_json(raw: bytes, *, label: str) -> Any:
  def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
      if key in result:
        raise ValueError(f"{label} contains duplicate JSON keys")
      result[key] = value
    return result
  try:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _sha256_value(value: Any, *, label: str) -> str:
  if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
    raise ValueError(f"{label} is not a SHA-256 value")
  return value


def _preflight_source_paths() -> Mapping[str, Path]:
  root = Path(__file__).resolve().parent
  return MappingProxyType(
      {
          "brep_full_graph_budget_preflight.py": root / "brep_full_graph_budget_preflight.py",
          "tools/preflight_unlabeled_full_graph_budget_v1.py": root / "tools" / "preflight_unlabeled_full_graph_budget_v1.py",
          "benchmark_v2_model_view_v2.py": root / "benchmark_v2_model_view_v2.py",
          "benchmark_v2_training_provenance.py": root / "benchmark_v2_training_provenance.py",
          "brep_program_learner_v1.py": root / "brep_program_learner_v1.py",
      }
  )


def preflight_producer_binding(*, expected_revision: str) -> dict[str, Any]:
  """Bind only the preflight scanner and its authenticated topology dependencies."""

  if not isinstance(expected_revision, str) or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
    raise ValueError("preflight expected revision is not a full Git object ID")
  root = Path(__file__).resolve().parent
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
  if revision != expected_revision:
    raise ValueError("full-graph preflight producer revision differs")
  dirty = subprocess.check_output(
      ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
  )
  if dirty.strip():
    raise ValueError("full-graph preflight producer tracked worktree is not clean")
  return {
      "revision": revision,
      "source_sha256s": {
          name: _file_sha256(path) for name, path in _preflight_source_paths().items()
      },
  }


def historical_preflight_producer_binding(*, revision: str) -> dict[str, Any]:
  """Rebuild a preflight producer binding from immutable Git objects."""

  if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError("preflight historical revision is not a full Git object ID")
  root = Path(__file__).resolve().parent
  sources: dict[str, str] = {}
  for name in _PREFLIGHT_SOURCE_NAMES:
    try:
      raw = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
    except subprocess.CalledProcessError as error:
      raise ValueError("preflight producer source is absent from its revision") from error
    sources[name] = hashlib.sha256(raw).hexdigest()
  return {"revision": revision, "source_sha256s": sources}


def _percentile(values: Sequence[int], fraction: float) -> int:
  if not values:
    raise ValueError("preflight distribution is empty")
  ordered = sorted(values)
  return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _distribution(values: Sequence[int]) -> dict[str, int]:
  return {
      "min": min(values),
      "p50": _percentile(values, 0.50),
      "p95": _percentile(values, 0.95),
      "max": max(values),
  }


def _identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
  return {
      key: row[key]
      for key in (
          "split", "case_index", "case_id", "part", "source_instance_key",
          "step_sha256",
      )
  }


def build_preflight_ledger(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    case_domain: Sequence[Mapping[str, Any]],
    projection: Mapping[str, Any],
    projection_artifact_sha256: str,
    source_cache_artifact_sha256: str,
    revision: str,
    producer: Mapping[str, Any],
    emit: Callable[[str, Mapping[str, Any]], None],
    emit_part_counts: bool = True,
) -> dict[str, Any]:
  """Deduplicate part identities, emit counts, then make whole-case decisions."""

  identities: dict[str, dict[str, Any]] = {}
  for occurrence in occurrences:
    identity = _identity_payload(occurrence)
    identity_sha256 = canonical_sha256(identity)
    current = identities.get(identity_sha256)
    if current is None:
      current = {
          **identity,
          "identity_sha256": identity_sha256,
          "face_count": int(occurrence["face_count"]),
          "adjacency_count": int(occurrence["adjacency_count"]),
          "row_ordinals": [],
          "program_indices": [],
          "endpoint_occurrence_count": 0,
      }
      identities[identity_sha256] = current
    elif (
        current["face_count"] != occurrence["face_count"]
        or current["adjacency_count"] != occurrence["adjacency_count"]
    ):
      raise ValueError("preflight reused part identity has inconsistent topology counts")
    current["row_ordinals"].append(int(occurrence["row_ordinal"]))
    current["program_indices"].append(int(occurrence["program_index"]))
    current["endpoint_occurrence_count"] += 1

  ordered = sorted(
      identities.values(), key=lambda row: (row["split"], row["case_index"], row["part"])
  )
  excluded: dict[tuple[str, int], list[str]] = {}
  over_faces: list[dict[str, Any]] = []
  over_adjacencies: list[dict[str, Any]] = []
  for row in ordered:
    row["row_ordinals"] = sorted(set(row["row_ordinals"]))
    row["program_indices"] = sorted(set(row["program_indices"]))
    row["identity_reuse_count"] = row.pop("endpoint_occurrence_count") - 1
    # This event is deliberately flushed by the CLI before exclusion is decided.
    if emit_part_counts:
      emit(
          "part_counted",
          {
              key: row[key]
              for key in (
                  "split", "case_index", "case_id", "part", "step_sha256",
                  "face_count", "adjacency_count",
              )
          }
          | {"row_ordinal": row["row_ordinals"][0]},
      )
    key = (str(row["split"]), int(row["case_index"]))
    reasons = excluded.setdefault(key, [])
    if row["face_count"] > MAX_FACES:
      reasons.append("faces_over_1024")
      over_faces.append(dict(row))
    if row["adjacency_count"] > MAX_ADJACENCIES:
      reasons.append("adjacencies_over_4096")
      over_adjacencies.append(dict(row))

  decisions: list[dict[str, Any]] = []
  for case in case_domain:
    key = (str(case["split"]), int(case["case_index"]))
    reasons = sorted(set(excluded.get(key, ())))
    identity_hashes = [
        row["identity_sha256"]
        for row in ordered
        if (row["split"], row["case_index"]) == key
    ]
    if not identity_hashes:
      raise ValueError("preflight case has no bound STEP part identities")
    decisions.append(
        {
            **dict(case),
            "part_identity_sha256s": identity_hashes,
            "admitted": not reasons,
            "exclusion_reasons": reasons,
        }
    )
    if reasons:
      emit(
          "case_excluded",
          {
              "split": key[0],
              "case_index": key[1],
              "case_id": case["case_id"],
              "exclusion_reasons": reasons,
          },
      )
  face_counts = [int(row["face_count"]) for row in ordered]
  adjacency_counts = [int(row["adjacency_count"]) for row in ordered]
  excluded_cases = [
      {key: row[key] for key in ("split", "case_index", "case_id", "exclusion_reasons")}
      for row in decisions if not row["admitted"]
  ]
  ledger = {
      "schema_version": PREFLIGHT_SCHEMA_VERSION,
      "scope": "p0_train_dev_600_cases_full_step_no_final_access",
      "final_test_touched": False,
      "projection": dict(projection),
      "projection_artifact_sha256": projection_artifact_sha256,
      "source_cache_artifact_sha256": source_cache_artifact_sha256,
      "revision": revision,
      "producer_binding": dict(producer),
      "budget": {"max_faces_per_graph": MAX_FACES, "max_edges_per_graph": MAX_ADJACENCIES},
      "case_count": len(decisions),
      "part_identity_count": len(ordered),
      "unique_step_count": len({row["step_sha256"] for row in ordered}),
      "endpoint_occurrence_count": len(occurrences),
      "part_identities": ordered,
      "case_decisions": decisions,
      "summary": {
          "admitted_case_count": sum(row["admitted"] for row in decisions),
          "excluded_case_count": len(excluded_cases),
          "face_count_distribution": _distribution(face_counts),
          "adjacency_count_distribution": _distribution(adjacency_counts),
          "over_1024_faces": over_faces,
          "over_4096_adjacencies": over_adjacencies,
          "case_level_exclusions": excluded_cases,
      },
      "case_domain_sha256": canonical_sha256(list(case_domain)),
      "case_exclusion_set_sha256": canonical_sha256(excluded_cases),
  }
  ledger["ledger_payload_sha256"] = canonical_sha256(ledger)
  return ledger


@dataclass(frozen=True, slots=True)
class AuthenticatedFullGraphBudgetPreflight:
  artifact_sha256: str
  binding: Mapping[str, Any]
  _decisions: Mapping[tuple[str, int], bool]
  _factory_token: object

  def __post_init__(self) -> None:
    if self._factory_token is not _FACTORY_TOKEN:
      raise TypeError("full-graph preflight is verifier-factory-only")
    object.__setattr__(self, "binding", MappingProxyType(dict(self.binding)))
    object.__setattr__(self, "_decisions", MappingProxyType(dict(self._decisions)))

  def allows(self, *, split: str, case_index: int) -> bool:
    key = (str(split), int(case_index))
    if key not in self._decisions:
      raise ValueError("row case is outside the preflight case domain")
    return bool(self._decisions[key])


def load_full_graph_budget_preflight(
    root: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_projection_artifact_sha256: str,
    expected_source_cache_artifact_sha256: str,
    expected_revision: str,
) -> AuthenticatedFullGraphBudgetPreflight:
  directory = Path(root)
  artifact_path = directory / "artifact.json"
  if _file_sha256(artifact_path) != expected_artifact_sha256:
    raise ValueError("full-graph preflight artifact pin differs")
  artifact = _strict_json(artifact_path.read_bytes(), label="full-graph preflight artifact")
  _exact_keys(artifact, {"schema_version", "ledger", "artifact_payload_sha256"}, label="full-graph preflight artifact")
  if artifact["schema_version"] != PREFLIGHT_ARTIFACT_SCHEMA_VERSION:
    raise ValueError("full-graph preflight artifact schema differs")
  artifact_copy = dict(artifact)
  if artifact_copy.pop("artifact_payload_sha256") != canonical_sha256(artifact_copy):
    raise ValueError("full-graph preflight artifact self hash differs")
  ledger_binding = _exact_keys(artifact["ledger"], {"name", "bytes", "sha256"}, label="full-graph preflight ledger binding")
  ledger_path = directory / str(ledger_binding["name"])
  if dict(ledger_binding) != _binding(ledger_path, name=str(ledger_binding["name"])):
    raise ValueError("full-graph preflight ledger changed")
  ledger = _strict_json(ledger_path.read_bytes(), label="full-graph preflight ledger")
  ledger_copy = dict(ledger)
  if ledger_copy.pop("ledger_payload_sha256", None) != canonical_sha256(ledger_copy):
    raise ValueError("full-graph preflight ledger self hash differs")
  if (
      ledger.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
      or ledger.get("scope") != "p0_train_dev_600_cases_full_step_no_final_access"
      or ledger.get("final_test_touched") is not False
      or ledger.get("case_count") != 600
      or ledger.get("projection_artifact_sha256") != expected_projection_artifact_sha256
      or ledger.get("source_cache_artifact_sha256") != expected_source_cache_artifact_sha256
      or ledger.get("revision") != expected_revision
      or ledger.get("producer_binding") != historical_preflight_producer_binding(
          revision=expected_revision
      )
  ):
    raise ValueError("full-graph preflight authority binding differs")
  decisions = ledger.get("case_decisions")
  if not isinstance(decisions, list) or len(decisions) != 600:
    raise ValueError("full-graph preflight case decisions differ")
  decision_map: dict[tuple[str, int], bool] = {}
  for row in decisions:
    if not isinstance(row, Mapping) or row.get("split") not in {"train", "dev"} or type(row.get("case_index")) is not int or type(row.get("admitted")) is not bool:
      raise ValueError("full-graph preflight decision differs")
    key = (str(row["split"]), int(row["case_index"]))
    if key in decision_map:
      raise ValueError("full-graph preflight decision is duplicated")
    decision_map[key] = bool(row["admitted"])
  binding = {
      "artifact_sha256": expected_artifact_sha256,
      "ledger_sha256": str(ledger_binding["sha256"]),
      "ledger_payload_sha256": str(ledger["ledger_payload_sha256"]),
      "case_domain_sha256": str(ledger["case_domain_sha256"]),
      "case_exclusion_set_sha256": str(ledger["case_exclusion_set_sha256"]),
  }
  for key, value in binding.items():
    _sha256_value(value, label=f"full-graph preflight {key}")
  return AuthenticatedFullGraphBudgetPreflight(
      artifact_sha256=expected_artifact_sha256,
      binding=binding,
      _decisions=decision_map,
      _factory_token=_FACTORY_TOKEN,
  )


def publish_full_graph_budget_preflight(
    output_directory: str | Path,
    *,
    index: Any,
    source_cache: AuthenticatedShardedBRepTensorCacheV2,
    projection_path: Path,
    projection_artifact_sha256: str,
    expected_revision: str,
    emit: Callable[[str, Mapping[str, Any]], None],
) -> str:
  if type(source_cache) is not AuthenticatedShardedBRepTensorCacheV2:
    raise TypeError("preflight requires an authenticated source cache")
  build_binding = preflight_producer_binding(expected_revision=expected_revision)
  source_cache.revalidate()
  projection_binding = _binding(projection_path, name=projection_path.name)
  if projection_binding["sha256"] != projection_artifact_sha256:
    raise ValueError("preflight projection differs from external pin")
  state = model_view._get_private_state(index, model_view._DevelopmentIndexState)
  case_domain = [
      {"split": split, "case_index": case_index, "case_id": binding.case_id}
      for split in ("train", "dev")
      for case_index, binding in enumerate(state.bindings[split])
  ]
  if len(case_domain) != 600:
    raise ValueError("preflight requires exactly 600 development cases")
  cache_rows = list(source_cache._manifest["rows"])
  cache_positions = {
      (row["split"], row["case_index"], row["program_index"]): row["row_ordinal"]
      for row in cache_rows
  }
  topology_counts: dict[str, tuple[int, int]] = {}
  occurrences: list[dict[str, Any]] = []
  emitted_identities: set[str] = set()
  for case in case_domain:
    split = str(case["split"])
    case_index = int(case["case_index"])
    binding = state.bindings[split][case_index]
    rows = model_view._library_rows(binding.library_capture)
    commitments = binding.semantic_payload.get("row_commitments")
    if not isinstance(commitments, list) or len(commitments) != len(rows):
      raise ValueError("preflight semantic authority differs")
    for program_index, (row, commitment) in enumerate(zip(rows, commitments, strict=True)):
      position = (split, case_index, program_index)
      if position not in cache_positions or commitment.get("row_sha256") != model_view._canonical_sha256(row):
        raise ValueError("preflight source-cache replay differs")
      ordinal = commitment.get("source_contact_ordinal")
      contact = model_view._private_gold_contact(binding, source_contact_ordinal=ordinal)
      for endpoint in (contact.get("endpoint_a"), contact.get("endpoint_b")):
        _mapping, step, instance = model_view._face_endpoint_binding(binding, endpoint=endpoint)
        capture = model_view._captured_bound_file(step.get("file"), label="full-graph preflight STEP")
        counts = topology_counts.get(capture.sha256)
        if counts is None:
          # Retain only integer counts across STEP identities.  Holding every
          # OCC shape/topology until the 600-case scan ends is unbounded.
          streams = model_view._CapturedStepStreamCache()
          faces, _face_edges, edge_groups, _signatures = streams.load_step_topology(capture)
          counts = (
              len(faces),
              sum(1 for group in edge_groups if len(group) == 2 and group[0][0] != group[1][0]),
          )
          topology_counts[capture.sha256] = counts
          del faces, edge_groups, streams
        occurrence = {
                **case,
                "part": str(endpoint["part"]),
                "source_instance_key": str(instance.get("source_instance_key")),
                "step_sha256": capture.sha256,
                "row_ordinal": cache_positions[position],
                "program_index": program_index,
                "face_count": counts[0],
                "adjacency_count": counts[1],
            }
        identity_sha256 = canonical_sha256(_identity_payload(occurrence))
        if identity_sha256 not in emitted_identities:
          emitted_identities.add(identity_sha256)
          emit(
              "part_counted",
              {
                  key: occurrence[key]
                  for key in (
                      "split", "case_index", "case_id", "part", "step_sha256",
                      "face_count", "adjacency_count", "row_ordinal",
                  )
              },
          )
        occurrences.append(occurrence)
  ledger = build_preflight_ledger(
      occurrences,
      case_domain=case_domain,
      projection=projection_binding,
      projection_artifact_sha256=projection_artifact_sha256,
      source_cache_artifact_sha256=source_cache.artifact_sha256,
      revision=expected_revision,
      producer=build_binding,
      emit=emit,
      emit_part_counts=False,
  )
  output = Path(output_directory)
  if output.exists():
    raise FileExistsError("preflight output already exists")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
  try:
    ledger_path = staging / "preflight_ledger.json"
    ledger_path.write_bytes(canonical_bytes(ledger))
    artifact = {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "ledger": _binding(ledger_path, name=ledger_path.name),
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    artifact_bytes = canonical_bytes(artifact)
    (staging / "artifact.json").write_bytes(artifact_bytes)
    source_cache.revalidate()
    os.replace(staging, output)
    return hashlib.sha256(artifact_bytes).hexdigest()
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise
