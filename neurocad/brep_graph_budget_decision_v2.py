"""Model-independent full-graph budget decision derived from immutable preflight."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

from .brep_full_graph_budget_preflight import (
    AuthenticatedFullGraphBudgetPreflight,
    _binding,
    _exact_keys,
    _file_sha256,
    _sha256_value,
    _strict_json,
    canonical_bytes,
    canonical_sha256,
)


DECISION_SCHEMA_VERSION = "brep_graph_budget_decision.v2"
DECISION_ARTIFACT_SCHEMA_VERSION = "brep_graph_budget_decision_artifact.v2"
MAX_FACES = 2048
MAX_ADJACENCIES = 4096
_FACTORY_TOKEN = object()
_SOURCE_NAMES = (
    "brep_graph_budget_decision_v2.py",
    "tools/publish_brep_graph_budget_decision_v2.py",
    "brep_full_graph_budget_preflight.py",
)


def _source_paths() -> Mapping[str, Path]:
  root = Path(__file__).resolve().parent
  return MappingProxyType(
      {
          "brep_graph_budget_decision_v2.py": root / "brep_graph_budget_decision_v2.py",
          "tools/publish_brep_graph_budget_decision_v2.py": root / "tools" / "publish_brep_graph_budget_decision_v2.py",
          "brep_full_graph_budget_preflight.py": root / "brep_full_graph_budget_preflight.py",
      }
  )


def decision_producer_binding(*, expected_revision: str) -> dict[str, Any]:
  if not isinstance(expected_revision, str) or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
    raise ValueError("decision expected revision is not a full Git object ID")
  root = Path(__file__).resolve().parent
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
  dirty = subprocess.check_output(
      ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
  )
  if revision != expected_revision or dirty.strip():
    raise ValueError("graph-budget decision requires its expected clean revision")
  return {
      "revision": revision,
      "source_sha256s": {name: _file_sha256(path) for name, path in _source_paths().items()},
  }


def historical_decision_producer_binding(*, revision: str) -> dict[str, Any]:
  if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError("decision historical revision is not a full Git object ID")
  root = Path(__file__).resolve().parent
  sources: dict[str, str] = {}
  for name in _SOURCE_NAMES:
    try:
      raw = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
    except subprocess.CalledProcessError as error:
      raise ValueError("decision producer source is absent from its revision") from error
    sources[name] = hashlib.sha256(raw).hexdigest()
  return {"revision": revision, "source_sha256s": sources}


def _load_bound_preflight_ledger(
    root: Path,
    *,
    preflight: AuthenticatedFullGraphBudgetPreflight,
) -> Mapping[str, Any]:
  artifact_path = root / "artifact.json"
  if _file_sha256(artifact_path) != preflight.artifact_sha256:
    raise ValueError("decision preflight artifact changed")
  artifact = _strict_json(artifact_path.read_bytes(), label="decision preflight artifact")
  binding = artifact.get("ledger")
  if not isinstance(binding, Mapping) or set(binding) != {"name", "bytes", "sha256"}:
    raise ValueError("decision preflight ledger binding differs")
  ledger_path = root / str(binding["name"])
  if _binding(ledger_path, name=str(binding["name"])) != dict(binding):
    raise ValueError("decision preflight ledger changed")
  if binding["sha256"] != preflight.binding["ledger_sha256"]:
    raise ValueError("decision preflight ledger pin differs")
  ledger = _strict_json(ledger_path.read_bytes(), label="decision preflight ledger")
  if (
      ledger.get("ledger_payload_sha256") != preflight.binding["ledger_payload_sha256"]
      or ledger.get("case_domain_sha256") != preflight.binding["case_domain_sha256"]
      or ledger.get("case_exclusion_set_sha256") != preflight.binding["case_exclusion_set_sha256"]
  ):
    raise ValueError("decision preflight ledger authority differs")
  return ledger


def _decision_body(
    *,
    preflight: AuthenticatedFullGraphBudgetPreflight,
    preflight_ledger: Mapping[str, Any],
    revision: str,
    producer_binding: Mapping[str, Any],
) -> dict[str, Any]:
  parts = preflight_ledger.get("part_identities")
  cases = preflight_ledger.get("case_decisions")
  if not isinstance(parts, list) or not isinstance(cases, list) or len(cases) != 600:
    raise ValueError("decision preflight domain differs")
  parts_by_case: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
  for part in parts:
    if not isinstance(part, Mapping):
      raise ValueError("decision preflight part differs")
    parts_by_case.setdefault((str(part["split"]), int(part["case_index"])), []).append(part)
  decisions: list[dict[str, Any]] = []
  for case in cases:
    key = (str(case["split"]), int(case["case_index"]))
    case_parts = parts_by_case.get(key, [])
    reasons: list[str] = []
    if any(int(part["face_count"]) > MAX_FACES for part in case_parts):
      reasons.append("faces_over_2048")
    if any(int(part["adjacency_count"]) > MAX_ADJACENCIES for part in case_parts):
      reasons.append("adjacencies_over_4096")
    decisions.append(
        {
            "split": key[0],
            "case_index": key[1],
            "case_id": case["case_id"],
            "admitted": not reasons,
            "exclusion_reasons": reasons,
        }
    )
  max_faces = max(int(part["face_count"]) for part in parts)
  max_adjacencies = max(int(part["adjacency_count"]) for part in parts)
  excluded = [row for row in decisions if not row["admitted"]]
  body = {
      "schema_version": DECISION_SCHEMA_VERSION,
      "scope": "model_independent_full_step_budget_for_p0_600_cases",
      "final_test_touched": False,
      "revision": revision,
      "producer_binding": dict(producer_binding),
      "preflight_binding": dict(preflight.binding),
      "preflight_original_exclusion_set_sha256": preflight.binding["case_exclusion_set_sha256"],
      "budget": {
          "max_faces_per_graph": MAX_FACES,
          "max_edges_per_graph": MAX_ADJACENCIES,
          "policy": "fixed_model_independent_complete_graph_budget",
          "geometry_tolerances_changed": False,
      },
      "observed": {
          "max_face_count": max_faces,
          "max_adjacency_count": max_adjacencies,
      },
      "headroom": {
          "face_absolute": MAX_FACES - max_faces,
          "face_relative_to_budget": (MAX_FACES - max_faces) / MAX_FACES,
          "adjacency_absolute": MAX_ADJACENCIES - max_adjacencies,
          "adjacency_relative_to_budget": (MAX_ADJACENCIES - max_adjacencies) / MAX_ADJACENCIES,
      },
      "case_count": len(decisions),
      "admitted_case_count": sum(row["admitted"] for row in decisions),
      "excluded_case_count": len(excluded),
      "case_domain_sha256": preflight.binding["case_domain_sha256"],
      "case_decisions": decisions,
      "decision_exclusion_set_sha256": canonical_sha256(excluded),
  }
  return body


@dataclass(frozen=True, slots=True)
class AuthenticatedGraphBudgetDecisionV2:
  artifact_sha256: str
  binding: Mapping[str, Any]
  preflight_binding: Mapping[str, Any]
  max_faces_per_graph: int
  max_edges_per_graph: int
  _decisions: Mapping[tuple[str, int], bool]
  _factory_token: object

  def __post_init__(self) -> None:
    if self._factory_token is not _FACTORY_TOKEN:
      raise TypeError("graph-budget decisions are verifier-factory-only")
    object.__setattr__(self, "binding", MappingProxyType(dict(self.binding)))
    object.__setattr__(self, "preflight_binding", MappingProxyType(dict(self.preflight_binding)))
    object.__setattr__(self, "_decisions", MappingProxyType(dict(self._decisions)))

  def allows(self, *, split: str, case_index: int) -> bool:
    key = (str(split), int(case_index))
    if key not in self._decisions:
      raise ValueError("case is outside the graph-budget decision domain")
    return bool(self._decisions[key])


def publish_graph_budget_decision_v2(
    output_directory: str | Path,
    *,
    preflight_root: str | Path,
    preflight: AuthenticatedFullGraphBudgetPreflight,
    expected_revision: str,
) -> str:
  if type(preflight) is not AuthenticatedFullGraphBudgetPreflight:
    raise TypeError("decision publisher requires authenticated preflight")
  producer = decision_producer_binding(expected_revision=expected_revision)
  body = _decision_body(
      preflight=preflight,
      preflight_ledger=_load_bound_preflight_ledger(Path(preflight_root), preflight=preflight),
      revision=expected_revision,
      producer_binding=producer,
  )
  body["decision_payload_sha256"] = canonical_sha256(body)
  output = Path(output_directory)
  if output.exists():
    raise FileExistsError("graph-budget decision output already exists")
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
  try:
    decision_path = staging / "decision.json"
    decision_path.write_bytes(canonical_bytes(body))
    artifact = {
        "schema_version": DECISION_ARTIFACT_SCHEMA_VERSION,
        "decision": _binding(decision_path, name=decision_path.name),
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    artifact_bytes = canonical_bytes(artifact)
    (staging / "artifact.json").write_bytes(artifact_bytes)
    os.replace(staging, output)
    return hashlib.sha256(artifact_bytes).hexdigest()
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise


def load_graph_budget_decision_v2(
    root: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_revision: str,
    preflight_root: str | Path,
    preflight: AuthenticatedFullGraphBudgetPreflight,
) -> AuthenticatedGraphBudgetDecisionV2:
  directory = Path(root)
  artifact_path = directory / "artifact.json"
  if _file_sha256(artifact_path) != expected_artifact_sha256:
    raise ValueError("graph-budget decision artifact pin differs")
  artifact = _strict_json(artifact_path.read_bytes(), label="graph-budget decision artifact")
  _exact_keys(artifact, {"schema_version", "decision", "artifact_payload_sha256"}, label="graph-budget decision artifact")
  artifact_copy = dict(artifact)
  if (
      artifact_copy.pop("artifact_payload_sha256") != canonical_sha256(artifact_copy)
      or artifact["schema_version"] != DECISION_ARTIFACT_SCHEMA_VERSION
  ):
    raise ValueError("graph-budget decision artifact differs")
  decision_binding = artifact["decision"]
  decision_path = directory / str(decision_binding["name"])
  if _binding(decision_path, name=str(decision_binding["name"])) != dict(decision_binding):
    raise ValueError("graph-budget decision changed")
  decision = _strict_json(decision_path.read_bytes(), label="graph-budget decision")
  decision_copy = dict(decision)
  if decision_copy.pop("decision_payload_sha256", None) != canonical_sha256(decision_copy):
    raise ValueError("graph-budget decision self hash differs")
  if (
      decision.get("revision") != expected_revision
      or decision.get("producer_binding") != historical_decision_producer_binding(revision=expected_revision)
      or decision.get("preflight_binding") != dict(preflight.binding)
  ):
    raise ValueError("graph-budget decision authority binding differs")
  expected = _decision_body(
      preflight=preflight,
      preflight_ledger=_load_bound_preflight_ledger(Path(preflight_root), preflight=preflight),
      revision=expected_revision,
      producer_binding=decision["producer_binding"],
  )
  if decision_copy != expected:
    raise ValueError("graph-budget decision recomputation differs")
  decisions = {
      (str(row["split"]), int(row["case_index"])): bool(row["admitted"])
      for row in decision["case_decisions"]
  }
  binding = {
      "artifact_sha256": expected_artifact_sha256,
      "decision_sha256": decision_binding["sha256"],
      "decision_payload_sha256": decision["decision_payload_sha256"],
      "case_domain_sha256": decision["case_domain_sha256"],
      "decision_exclusion_set_sha256": decision["decision_exclusion_set_sha256"],
  }
  for key, value in binding.items():
    _sha256_value(value, label=f"graph-budget decision {key}")
  return AuthenticatedGraphBudgetDecisionV2(
      artifact_sha256=expected_artifact_sha256,
      binding=binding,
      preflight_binding=decision["preflight_binding"],
      max_faces_per_graph=int(decision["budget"]["max_faces_per_graph"]),
      max_edges_per_graph=int(decision["budget"]["max_edges_per_graph"]),
      _decisions=decisions,
      _factory_token=_FACTORY_TOKEN,
  )
