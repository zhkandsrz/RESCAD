"""Generalist CAD agent with a rich, target-free serialized B-Rep view."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .linkcad_brep_encoder_v1 import SURFACE_TYPES
from .linkcad_generalist_agent_baseline_v1 import (
    DEFAULT_MODEL,
    FrozenGeneralistAgentBatchV1,
    FrozenGeneralistCADAgentV1,
    canonical_sha256,
    validate_agent_response_v1,
)
from .linkcad_port_contract_v1 import describe_port_v1
from .linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2


SCHEMA_VERSION = "linkcad_serialized_brep_agent_baseline.v2"
SUMMARY_SCHEMA_VERSION = "linkcad_serialized_brep_public_summary.v2"
PROMPT_VERSION = "linkcad_serialized_brep_agent_prompt.v2"
COMPACT_SCHEMA_VERSION = "linkcad_serialized_brep_agent_baseline.v3"
COMPACT_SUMMARY_SCHEMA_VERSION = "linkcad_serialized_brep_public_summary.v3"
COMPACT_PROMPT_VERSION = "linkcad_serialized_brep_agent_prompt.v3"


def system_prompt_v2() -> str:
  return """You are a general CAD assembly agent. Rank every opaque candidate ID for every semantic role using the supplied functional language and serialized intrinsic B-rep geometry. The serialization exposes the same public geometry categories as the specialist: face-surface topology, structural orbit sizes, primitive descriptions, and normalized continuous primitive features. Candidate IDs and list order carry no semantics. Jointly reason over every incident connection, but do not invent candidates or interfaces. Return one JSON object and no prose. Preserve every query_id, role_id, and candidate_id exactly. For each query, output every role exactly once and every candidate for that role exactly once, ordered from most to least suitable. Output exactly {"queries":[{"query_id":str,"role_rankings":[{"role_id":str,"candidate_ids":[str,...]}]}]}."""


def prompt_sha256_v2() -> str:
  return hashlib.sha256(system_prompt_v2().encode("utf-8")).hexdigest()


def system_prompt_v3() -> str:
  return """You are a general CAD assembly agent. For each semantic role, rank only the opaque candidate IDs listed inside that role, using the functional language and compact serialized intrinsic B-rep geometry. The serialization contains face-surface topology, structural orbit sizes, primitive descriptions, and normalized continuous-feature summaries. Candidate IDs and list order carry no semantics. Use every incident connection, but never move a candidate ID between roles and never invent an ID. Return one JSON object and no prose. Preserve every query_id and role_id exactly. For each role, return exactly the candidate IDs listed for that role, each exactly once, ordered from most to least suitable. Output exactly {"queries":[{"query_id":str,"role_rankings":[{"role_id":str,"candidate_ids":[str,...]}]}]}."""


def prompt_sha256_v3() -> str:
  return hashlib.sha256(system_prompt_v3().encode("utf-8")).hexdigest()


def _surface_types(graph) -> list[str]:
  one_hot = graph.face_graph.node_features[:, -len(SURFACE_TYPES):]
  return [SURFACE_TYPES[int(row.argmax())] for row in one_hot]


def _serialized_graph(graph) -> dict[str, Any]:
  surfaces = _surface_types(graph)
  adjacency = Counter()
  degrees = [0] * len(surfaces)
  for source, target in graph.face_graph.edge_index.tolist():
    source, target = int(source), int(target)
    degrees[source] += 1
    degrees[target] += 1
    adjacency["--".join(sorted((surfaces[source], surfaces[target])))] += 1
  contract_rows: dict[str, dict[str, Any]] = {}
  for ordinal in range(int(graph.primitive_features.shape[0])):
    contract = describe_port_v1(graph, ordinal)
    key = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    row = contract_rows.setdefault(key, {
        "description": contract,
        "count": 0,
        "orbit_member_counts": [],
        "intrinsic_feature_vectors": [],
    })
    row["count"] += 1
    row["orbit_member_counts"].append(len(graph.primitive_members[ordinal]))
    row["intrinsic_feature_vectors"].append([
        round(float(value), 5)
        for value in graph.primitive_features[ordinal, :8].tolist()
    ])
  return {
      "face_count": len(surfaces),
      "face_adjacency_count": int(graph.face_graph.edge_index.shape[0]),
      "surface_type_counts": dict(sorted(Counter(surfaces).items())),
      "face_degree_histogram": {
          str(key): value for key, value in sorted(Counter(degrees).items())
      },
      "surface_adjacency_type_counts": dict(sorted(adjacency.items())),
      "face_orbit_sizes": sorted(
          len(members) for members in graph.face_graph.orbit_members
      ),
      "primitive_count": int(graph.primitive_features.shape[0]),
      "primitive_contract_groups": sorted(
          contract_rows.values(),
          key=lambda row: json.dumps(row["description"], sort_keys=True),
      ),
  }


def _serialized_graph_v3(graph) -> dict[str, Any]:
  """Compact the public graph without dropping geometry categories."""

  serialized = _serialized_graph(graph)
  groups = []
  for source in serialized["primitive_contract_groups"]:
    vectors = source["intrinsic_feature_vectors"]
    feature_summary = []
    for dimension in range(len(vectors[0]) if vectors else 0):
      values = [float(vector[dimension]) for vector in vectors]
      mean = sum(values) / len(values)
      variance = sum((value - mean) ** 2 for value in values) / len(values)
      feature_summary.append({
          "mean": round(mean, 5),
          "std": round(math.sqrt(variance), 5),
          "min": round(min(values), 5),
          "max": round(max(values), 5),
      })
    groups.append({
        "description": source["description"],
        "count": source["count"],
        "orbit_member_count_histogram": {
            str(key): value for key, value in sorted(
                Counter(source["orbit_member_counts"]).items()
            )
        },
        "feature_summary": feature_summary,
    })
  serialized["primitive_contract_groups"] = groups
  return serialized


def build_public_query_summary_v2(
    query: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
) -> dict[str, Any]:
  roles = {str(row["role_id"]): str(row["description"]) for row in query["roles"]}
  connections = [{
      "edge_id": str(edge["edge_id"]),
      "role_a": str(edge["role_a"]), "role_b": str(edge["role_b"]),
      "instruction": str(edge["instruction"]),
      "requested_mobility": str(edge.get("requested_mobility", "unknown")),
      "endpoint_a": edge.get("port_contract_a"),
      "endpoint_b": edge.get("port_contract_b"),
  } for edge in query["functional_edges"]]
  role_rows = []
  for role_id, description in roles.items():
    candidates = []
    for candidate in query["candidate_sets"][role_id]:
      graph = cache.graph(str(candidate["step_sha256"]))
      candidates.append({
          "candidate_id": str(candidate["candidate_id"]),
          "volume_mm3": round(float(candidate.get("volume", 0.0)), 4),
          "area_mm2": round(float(candidate.get("area", 0.0)), 4),
          "serialized_brep": None if graph is None else _serialized_graph(graph),
      })
    role_rows.append({
        "role_id": role_id, "description": description,
        "candidates": candidates,
    })
  return {
      "schema_version": SUMMARY_SCHEMA_VERSION,
      "query_id": str(query["query_id"]),
      "roles": role_rows, "connections": connections,
      "contains_private_targets": False,
  }


def build_public_query_summary_v3(
    query: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
) -> dict[str, Any]:
  roles = {str(row["role_id"]): str(row["description"]) for row in query["roles"]}
  connections = [{
      "edge_id": str(edge["edge_id"]),
      "role_a": str(edge["role_a"]), "role_b": str(edge["role_b"]),
      "instruction": str(edge["instruction"]),
      "requested_mobility": str(edge.get("requested_mobility", "unknown")),
      "endpoint_a": edge.get("port_contract_a"),
      "endpoint_b": edge.get("port_contract_b"),
  } for edge in query["functional_edges"]]
  role_rows = []
  for role_id, description in roles.items():
    candidates = []
    for candidate in query["candidate_sets"][role_id]:
      graph = cache.graph(str(candidate["step_sha256"]))
      candidates.append({
          "candidate_id": str(candidate["candidate_id"]),
          "volume_mm3": round(float(candidate.get("volume", 0.0)), 4),
          "area_mm2": round(float(candidate.get("area", 0.0)), 4),
          "serialized_brep": None if graph is None else _serialized_graph_v3(graph),
      })
    role_rows.append({
        "role_id": role_id, "description": description,
        "candidates": candidates,
    })
  return {
      "schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
      "query_id": str(query["query_id"]),
      "roles": role_rows, "connections": connections,
      "contains_private_targets": False,
  }


class FrozenSerializedBRepAgentV2(FrozenGeneralistCADAgentV1):
  def rank(
      self, summaries: Sequence[Mapping[str, Any]],
  ) -> FrozenGeneralistAgentBatchV1:
    if not summaries:
      raise ValueError("LinkCAD serialized B-rep request is empty")
    user_payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION, "queries": list(summaries),
    }
    request_payload = {
        "model": self.model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt_v2()},
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
        "prompt_sha256": prompt_sha256_v2(),
        "request_query_count": len(summaries),
        "request_sha256": canonical_sha256(user_payload),
        "response_sha256": canonical_sha256(response),
        "credential_material_included": False,
        "private_targets_opened": False,
    }
    return FrozenGeneralistAgentBatchV1(rankings=rankings, receipt=receipt)


class FrozenSerializedBRepAgentV3(FrozenGeneralistCADAgentV1):
  def rank(
      self, summaries: Sequence[Mapping[str, Any]],
  ) -> FrozenGeneralistAgentBatchV1:
    if not summaries:
      raise ValueError("LinkCAD compact serialized B-rep request is empty")
    user_payload = {
        "schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
        "queries": list(summaries),
    }
    request_payload = {
        "model": self.model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt_v3()},
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
        "schema_version": COMPACT_SCHEMA_VERSION,
        "provider": "deepseek", "model": self.model,
        "temperature": 0.0, "prompt_version": COMPACT_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256_v3(),
        "request_query_count": len(summaries),
        "request_sha256": canonical_sha256(user_payload),
        "response_sha256": canonical_sha256(response),
        "credential_material_included": False,
        "private_targets_opened": False,
    }
    return FrozenGeneralistAgentBatchV1(rankings=rankings, receipt=receipt)


__all__ = [
    "COMPACT_PROMPT_VERSION", "COMPACT_SCHEMA_VERSION",
    "COMPACT_SUMMARY_SCHEMA_VERSION", "DEFAULT_MODEL",
    "FrozenSerializedBRepAgentV2", "FrozenSerializedBRepAgentV3", "PROMPT_VERSION",
    "SCHEMA_VERSION", "SUMMARY_SCHEMA_VERSION", "build_public_query_summary_v2",
    "build_public_query_summary_v3", "prompt_sha256_v2", "prompt_sha256_v3",
    "system_prompt_v2", "system_prompt_v3",
]
