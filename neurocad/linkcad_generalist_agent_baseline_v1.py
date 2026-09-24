"""Frozen general-purpose language-agent baseline for LinkCAD part ranking.

The agent sees only public role descriptions, connection requests, and compact
intrinsic summaries of each candidate B-rep.  It ranks every candidate for
each role.  LinkCAD's symbolic code then converts those rankings into the same
feasible assignment beam and interface budget used by the learned method.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Callable, Mapping, Sequence
from urllib import error as url_error
from urllib import request as url_request

from .linkcad_port_contract_v1 import describe_port_v1
from .linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2


SCHEMA_VERSION = "linkcad_generalist_agent_baseline.v1"
SUMMARY_SCHEMA_VERSION = "linkcad_generalist_public_summary.v1"
RANKING_SCHEMA_VERSION = "linkcad_generalist_role_rankings.v1"
PROMPT_VERSION = "linkcad_generalist_agent_prompt.v1"
DEFAULT_MODEL = "deepseek-v4-pro"


def _canonical(value: Any) -> str:
  return json.dumps(
      value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
      allow_nan=False,
  )


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def system_prompt_v1() -> str:
  return """You are a general CAD assembly agent. Rank every public candidate for every semantic role using only the supplied role descriptions, connection requests, and intrinsic B-rep summaries. Candidate IDs are opaque labels, not semantic hints. Prefer parts whose whole-shape structure and available endpoint contracts jointly fit all incident connections. Return one JSON object and no prose. Preserve every query_id, role_id, and candidate_id exactly. For each query, output every role exactly once and every candidate for that role exactly once, ordered from most to least suitable. Output exactly {\"queries\":[{\"query_id\":str,\"role_rankings\":[{\"role_id\":str,\"candidate_ids\":[str,...]}]}]}."""


def prompt_sha256_v1() -> str:
  return hashlib.sha256(system_prompt_v1().encode("utf-8")).hexdigest()


def _contract_key(value: Mapping[str, Any]) -> str:
  return _canonical(value)


def build_public_query_summary_v1(
    query: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
) -> dict[str, Any]:
  """Summarize public intrinsic candidate evidence without file identities."""

  query_id = str(query["query_id"])
  roles = {str(row["role_id"]): str(row["description"]) for row in query["roles"]}
  if set(roles) != set(query["candidate_sets"]):
    raise ValueError("LinkCAD generalist summary role domain differs")
  connections = []
  incident: dict[str, list[tuple[Mapping[str, Any], str]]] = {
      role_id: [] for role_id in roles
  }
  for edge in query["functional_edges"]:
    role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
    if role_a not in incident or role_b not in incident:
      raise ValueError("LinkCAD generalist summary edge domain differs")
    incident[role_a].append((edge, "a"))
    incident[role_b].append((edge, "b"))
    connections.append({
        "edge_id": str(edge["edge_id"]),
        "role_a": role_a, "role_b": role_b,
        "instruction": str(edge["instruction"]),
        "requested_mobility": str(edge.get("requested_mobility", "unknown")),
        "port_contract_status": str(edge.get("port_contract_status", "unknown")),
        "endpoint_a": edge.get("port_contract_a"),
        "endpoint_b": edge.get("port_contract_b"),
    })
  role_summaries = []
  for role_id, description in roles.items():
    candidates = []
    for candidate in query["candidate_sets"][role_id]:
      graph = cache.graph(str(candidate["step_sha256"]))
      if graph is None:
        primitive_type_counts: dict[str, int] = {}
        primitive_kind_counts: dict[str, int] = {}
        contracts: list[str] = []
      else:
        primitive_type_counts = {}
        primitive_kind_counts = {}
        contracts = []
        for ordinal in range(int(graph.primitive_features.shape[0])):
          primitive_type = str(graph.primitive_type(ordinal))
          primitive_kind = str(graph.primitive_kinds[ordinal])
          primitive_type_counts[primitive_type] = primitive_type_counts.get(
              primitive_type, 0,
          ) + 1
          primitive_kind_counts[primitive_kind] = primitive_kind_counts.get(
              primitive_kind, 0,
          ) + 1
          contracts.append(_contract_key(describe_port_v1(graph, ordinal)))
      endpoint_matches = []
      for edge, side in incident[role_id]:
        contract = (
            edge.get(f"port_contract_{side}")
            if edge.get("port_contract_status") == "specified" else None
        )
        endpoint_matches.append({
            "edge_id": str(edge["edge_id"]),
            "side": side,
            "exact_matching_primitive_count": (
                None if contract is None
                else sum(value == _contract_key(contract) for value in contracts)
            ),
        })
      candidates.append({
          "candidate_id": str(candidate["candidate_id"]),
          "volume_mm3": round(float(candidate.get("volume", 0.0)), 4),
          "area_mm2": round(float(candidate.get("area", 0.0)), 4),
          "primitive_type_counts": dict(sorted(primitive_type_counts.items())),
          "primitive_kind_counts": dict(sorted(primitive_kind_counts.items())),
          "incident_endpoint_matches": endpoint_matches,
      })
    role_summaries.append({
        "role_id": role_id, "description": description,
        "candidates": candidates,
    })
  return {
      "schema_version": SUMMARY_SCHEMA_VERSION,
      "query_id": query_id,
      "roles": role_summaries,
      "connections": connections,
      "contains_private_targets": False,
  }


def validate_agent_response_v1(
    summaries: Sequence[Mapping[str, Any]], response: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
  if not isinstance(response, Mapping) or set(response) != {"queries"}:
    raise ValueError("LinkCAD generalist response envelope differs")
  rows = response["queries"]
  if not isinstance(rows, list) or len(rows) != len(summaries):
    raise ValueError("LinkCAD generalist response coverage differs")
  expected = {str(row["query_id"]): row for row in summaries}
  result: dict[str, dict[str, list[str]]] = {}
  for row in rows:
    if not isinstance(row, Mapping) or set(row) != {"query_id", "role_rankings"}:
      raise ValueError("LinkCAD generalist response query fields differ")
    query_id = str(row["query_id"])
    if query_id in result or query_id not in expected:
      raise ValueError("LinkCAD generalist response query identity differs")
    summary = expected[query_id]
    expected_roles = {
        str(role["role_id"]): {
            str(candidate["candidate_id"]) for candidate in role["candidates"]
        }
        for role in summary["roles"]
    }
    role_rankings = row["role_rankings"]
    if not isinstance(role_rankings, list) or len(role_rankings) != len(expected_roles):
      raise ValueError("LinkCAD generalist response role coverage differs")
    rankings: dict[str, list[str]] = {}
    for role in role_rankings:
      if not isinstance(role, Mapping) or set(role) != {"role_id", "candidate_ids"}:
        raise ValueError("LinkCAD generalist response role fields differ")
      role_id = str(role["role_id"])
      candidate_ids = [str(value) for value in role["candidate_ids"]]
      if (
          role_id in rankings or role_id not in expected_roles
          or len(candidate_ids) != len(set(candidate_ids))
          or set(candidate_ids) != expected_roles[role_id]
      ):
        raise ValueError("LinkCAD generalist response candidate ranking differs")
      rankings[role_id] = candidate_ids
    if set(rankings) != set(expected_roles):
      raise ValueError("LinkCAD generalist response role domain differs")
    result[query_id] = rankings
  if set(result) != set(expected):
    raise ValueError("LinkCAD generalist response query domain differs")
  return result


Transport = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class FrozenGeneralistAgentBatchV1:
  rankings: dict[str, dict[str, list[str]]]
  receipt: dict[str, Any]


class FrozenGeneralistCADAgentV1:
  """Zero-temperature general-purpose candidate ranker with strict JSON IO."""

  def __init__(
      self, *, model: str = DEFAULT_MODEL,
      base_url: str = "https://api.deepseek.com/v1",
      timeout_seconds: int = 180, transport: Transport | None = None,
  ) -> None:
    self.model = str(model)
    self.base_url = str(base_url).rstrip("/")
    self.timeout_seconds = int(timeout_seconds)
    self.transport = transport

  def rank(
      self, summaries: Sequence[Mapping[str, Any]],
  ) -> FrozenGeneralistAgentBatchV1:
    if not summaries:
      raise ValueError("LinkCAD generalist request is empty")
    user_payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "queries": list(summaries),
    }
    request_payload = {
        "model": self.model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt_v1()},
            {"role": "user", "content": json.dumps(
                user_payload, ensure_ascii=False, sort_keys=True,
            )},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    raw = (
        self.transport(request_payload)
        if self.transport is not None else self._remote_call(request_payload)
    )
    response = self._extract_response(raw)
    rankings = validate_agent_response_v1(summaries, response)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "provider": "deepseek", "model": self.model,
        "temperature": 0.0, "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256_v1(),
        "request_query_count": len(summaries),
        "request_sha256": canonical_sha256(user_payload),
        "response_sha256": canonical_sha256(response),
        "credential_material_included": False,
        "private_targets_opened": False,
    }
    return FrozenGeneralistAgentBatchV1(rankings=rankings, receipt=receipt)

  def _remote_call(self, payload: dict[str, Any]) -> Mapping[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
      raise RuntimeError("DEEPSEEK_API_KEY is required for the frozen baseline")
    request = url_request.Request(
        self.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
      with url_request.urlopen(request, timeout=self.timeout_seconds) as response:
        value = json.loads(response.read().decode("utf-8"))
    except (OSError, url_error.URLError, json.JSONDecodeError) as error:
      raise RuntimeError("Frozen LinkCAD generalist-agent call failed") from error
    if not isinstance(value, Mapping):
      raise RuntimeError("Frozen LinkCAD generalist-agent response differs")
    return value

  @staticmethod
  def _extract_response(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) == {"queries"}:
      return dict(raw)
    try:
      text = str(raw["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as error:
      raise ValueError("LinkCAD generalist provider response differs") from error
    if text.startswith("```"):
      text = text.removeprefix("```json").removeprefix("```")
      text = text.removesuffix("```").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
      raise ValueError("LinkCAD generalist JSON root differs")
    return value


__all__ = [
    "DEFAULT_MODEL", "FrozenGeneralistAgentBatchV1",
    "FrozenGeneralistCADAgentV1", "PROMPT_VERSION", "RANKING_SCHEMA_VERSION",
    "SCHEMA_VERSION", "SUMMARY_SCHEMA_VERSION", "build_public_query_summary_v1",
    "canonical_sha256", "prompt_sha256_v1", "system_prompt_v1",
    "validate_agent_response_v1",
]
