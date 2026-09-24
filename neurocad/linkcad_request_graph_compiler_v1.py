"""Frozen whole-request compiler for LinkCAD role-level connection graphs.

The compiler receives the role labels exposed by the candidate pools and one
natural-language assembly request.  It returns the role-level connection graph,
per-edge local interface descriptions, and requested motion.  Candidate IDs,
STEP paths, primitive ordinals, and reference edges are never exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Any, Callable, Mapping, Sequence
from urllib import error as url_error
from urllib import request as url_request

from .linkcad_factorized_model_v1 import EXECUTABLE_MOBILITY_NAMES
from .linkcad_language_planner_v1 import (
    DEFAULT_MODEL,
    PORT_KEYS,
    canonical_sha256,
    normalize_port_contract_v1,
    render_free_language_edge_v1,
)
from .linkcad_language_contracts import render_mobility_instruction
from .linkcad_port_contract_v1 import render_port_conditioned_instruction_v1


SCHEMA_VERSION = "linkcad_request_graph_compiler.v1"
INPUT_SCHEMA_VERSION = "linkcad_request_graph_inputs.v1"
TARGET_SCHEMA_VERSION = "linkcad_request_graph_targets.v1"
PREDICTION_SCHEMA_VERSION = "linkcad_request_graph_predictions.v1"
PROMPT_VERSION = "linkcad_request_graph_prompt.v1"

_FORBIDDEN_KEYS = {
    "candidate_id", "candidate_index", "primitive_ordinal",
    "primitive_orbit_ordinal", "source_joint_id", "occurrence_id",
    "body_id", "step_sha256", "step_path", "path", "filename",
}


def _contains_forbidden_identity(value: Any) -> bool:
  if isinstance(value, Mapping):
    return any(
        str(key).lower() in _FORBIDDEN_KEYS or _contains_forbidden_identity(item)
        for key, item in value.items()
    )
  if isinstance(value, (list, tuple)):
    return any(_contains_forbidden_identity(item) for item in value)
  return False


def _role_rows(query: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
  raw = query.get("roles")
  if not isinstance(raw, list) or len(raw) < 2:
    raise ValueError("LinkCAD request-graph roles differ")
  output = []
  seen = set()
  for item in raw:
    if not isinstance(item, Mapping) or "role_id" not in item:
      raise ValueError("LinkCAD request-graph role row differs")
    role_id = str(item["role_id"]).strip()
    description = str(item.get("description", role_id)).strip()
    if not role_id or role_id in seen:
      raise ValueError("LinkCAD request-graph role identity differs")
    seen.add(role_id)
    output.append({"role_id": role_id, "description": description or role_id})
  return tuple(output)


def _edge_pair(role_a: str, role_b: str) -> tuple[str, str]:
  return tuple(sorted((str(role_a), str(role_b))))


def _target_connection(edge: Mapping[str, Any]) -> dict[str, Any]:
  status = str(edge.get("port_contract_status", "unknown")).strip().lower()
  if status == "specified":
    port_a = normalize_port_contract_v1(edge["port_contract_a"])
    port_b = normalize_port_contract_v1(edge["port_contract_b"])
  elif status == "unknown":
    port_a = None
    port_b = None
  else:
    raise ValueError("LinkCAD request-graph target status differs")
  mobility = str(edge.get("requested_mobility", "")).strip().lower()
  if mobility not in EXECUTABLE_MOBILITY_NAMES:
    raise ValueError("LinkCAD request-graph target mobility differs")
  return {
      "role_a": str(edge["role_a"]),
      "role_b": str(edge["role_b"]),
      "port_contract_status": status,
      "port_a": port_a,
      "port_b": port_b,
      "requested_mobility": mobility,
  }


def _whole_request(query: Mapping[str, Any]) -> str:
  edges = query.get("functional_edges")
  if not isinstance(edges, list) or not edges:
    raise ValueError("LinkCAD request-graph source edges differ")
  instructions = [str(edge.get("instruction", "")).strip() for edge in edges]
  if any(not instruction for instruction in instructions):
    raise ValueError("LinkCAD request-graph source instruction differs")
  return "Build one assembly from the available role pools. " + " ".join(
      instructions
  )


def build_request_graph_benchmark_v1(
    public: Mapping[str, Any], *, split_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  """Build prompt-only graph inputs and sealed reference graph targets."""

  queries = public.get("queries")
  if not isinstance(queries, list) or not queries:
    raise ValueError("LinkCAD request-graph public queries differ")
  inputs = []
  targets = []
  edge_count = 0
  for query in queries:
    query_id = str(query["query_id"])
    roles = _role_rows(query)
    connections = tuple(
        _target_connection(edge) for edge in query["functional_edges"]
    )
    pairs = [_edge_pair(row["role_a"], row["role_b"]) for row in connections]
    if len(set(pairs)) != len(pairs):
      raise ValueError("LinkCAD request-graph target pair repeats")
    role_ids = {row["role_id"] for row in roles}
    if any(set(pair) - role_ids for pair in pairs):
      raise ValueError("LinkCAD request-graph target role differs")
    inputs.append({
        "query_id": query_id,
        "available_roles": list(roles),
        "request": _whole_request(query),
    })
    targets.append({
        "query_id": query_id,
        "roles": [row["role_id"] for row in roles],
        "connections": list(connections),
    })
    edge_count += len(connections)
  input_payload = {
      "schema_version": INPUT_SCHEMA_VERSION,
      "scope": "whole_request_and_role_labels_no_reference_graph",
      "split_name": str(split_name),
      "item_count": len(inputs),
      "contains_reference_connection_graph": False,
      "items": inputs,
  }
  target_payload = {
      "schema_version": TARGET_SCHEMA_VERSION,
      "scope": "reference_only_not_compiler_input",
      "split_name": str(split_name),
      "item_count": len(targets),
      "items": targets,
  }
  summary = {
      "schema_version": "linkcad_request_graph_benchmark_summary.v1",
      "split_name": str(split_name),
      "query_count": len(inputs),
      "edge_count": edge_count,
      "input_sha256": canonical_sha256(input_payload),
      "target_sha256": canonical_sha256(target_payload),
      "evaluation_scope": "frozen_frontend_component_audit",
  }
  return input_payload, target_payload, summary


def system_prompt_v1() -> str:
  return """You compile complete natural-language CAD assembly requests into role-level connection graphs. Return one JSON object and no prose. Each input item provides query_id, available_roles, and request. Copy every available role_id exactly into roles. Infer which role pairs must connect from the request. For each connection, extract requested_mobility and the two local port descriptions. Do not invent or omit roles, and do not use candidate IDs, primitive indices, file names, paths, hashes, or source-joint IDs. Allowed primitive_kind: face, edge. Allowed geometry_type: plane, cylinder, cone, sphere, torus, spline, other, circle, line, ellipse, bspline_curve, other_curve. Allowed size_class: only, smallest, small, medium, large, largest. Allowed adjacent_surface_types: plane, cylinder, cone, sphere, torus, spline, other; emit zero to two values. Allowed symmetry_class: single, repeated. Allowed requested_mobility: fixed, revolute, prismatic, cylindrical, planar, ball, pin_slot. If the request identifies a connection but does not describe both ports, set port_contract_status to unknown and port_a and port_b to null. Return exactly {\"items\":[{\"query_id\":str,\"roles\":[str],\"connections\":[{\"role_a\":str,\"role_b\":str,\"port_contract_status\":\"specified\"|\"unknown\",\"port_a\":object|null,\"port_b\":object|null,\"requested_mobility\":str}]}]}. Each port object has exactly primitive_kind, geometry_type, size_class, adjacent_surface_types, symmetry_class."""


def prompt_sha256_v1() -> str:
  return hashlib.sha256(system_prompt_v1().encode("utf-8")).hexdigest()


def compiler_user_payload_v1(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  normalized = []
  for row in items:
    if set(row) != {"query_id", "available_roles", "request"}:
      raise ValueError("LinkCAD request-graph compiler input fields differ")
    roles = []
    for role in row["available_roles"]:
      if set(role) != {"role_id", "description"}:
        raise ValueError("LinkCAD request-graph compiler role fields differ")
      roles.append({
          "role_id": str(role["role_id"]),
          "description": str(role["description"]),
      })
    normalized.append({
        "query_id": str(row["query_id"]),
        "available_roles": roles,
        "request": str(row["request"]),
    })
  if not normalized or _contains_forbidden_identity(normalized):
    raise ValueError("LinkCAD request-graph compiler identity boundary differs")
  return {"schema_version": INPUT_SCHEMA_VERSION, "items": normalized}


def _graph_is_connected(roles: Sequence[str], connections: Sequence[Mapping[str, Any]]) -> bool:
  if not roles:
    return False
  adjacency = {role: set() for role in roles}
  for edge in connections:
    a = str(edge["role_a"])
    b = str(edge["role_b"])
    adjacency[a].add(b)
    adjacency[b].add(a)
  visited = set()
  stack = [roles[0]]
  while stack:
    current = stack.pop()
    if current in visited:
      continue
    visited.add(current)
    stack.extend(adjacency[current] - visited)
  return visited == set(roles)


def validate_compiler_response_v1(
    request_items: Sequence[Mapping[str, Any]], response: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
  """Fail closed unless every request yields one valid connected graph."""

  if not isinstance(response, Mapping) or set(response) != {"items"}:
    raise ValueError("LinkCAD request-graph compiler envelope differs")
  rows = response["items"]
  if not isinstance(rows, list):
    raise ValueError("LinkCAD request-graph compiler response differs")
  expected = {str(row["query_id"]): row for row in request_items}
  if len(expected) != len(request_items) or len(rows) != len(expected):
    raise ValueError("LinkCAD request-graph compiler coverage differs")
  output = []
  seen_queries = set()
  connection_keys = {
      "role_a", "role_b", "port_contract_status", "port_a", "port_b",
      "requested_mobility",
  }
  for row in rows:
    if not isinstance(row, Mapping) or set(row) != {"query_id", "roles", "connections"}:
      raise ValueError("LinkCAD request-graph compiler item fields differ")
    query_id = str(row["query_id"])
    if query_id in seen_queries or query_id not in expected:
      raise ValueError("LinkCAD request-graph compiler query identity differs")
    seen_queries.add(query_id)
    allowed_roles = [
        str(role["role_id"]) for role in expected[query_id]["available_roles"]
    ]
    roles = [str(role) for role in row["roles"]]
    if len(roles) != len(set(roles)) or set(roles) != set(allowed_roles):
      raise ValueError("LinkCAD request-graph compiler role set differs")
    raw_connections = row["connections"]
    if not isinstance(raw_connections, list) or not raw_connections:
      raise ValueError("LinkCAD request-graph compiler connections differ")
    connections = []
    seen_pairs = set()
    for edge in raw_connections:
      if not isinstance(edge, Mapping) or set(edge) != connection_keys:
        raise ValueError("LinkCAD request-graph compiler connection fields differ")
      role_a = str(edge["role_a"])
      role_b = str(edge["role_b"])
      pair = _edge_pair(role_a, role_b)
      if role_a == role_b or set(pair) - set(allowed_roles) or pair in seen_pairs:
        raise ValueError("LinkCAD request-graph compiler connection identity differs")
      seen_pairs.add(pair)
      status = str(edge["port_contract_status"]).strip().lower()
      attribute_status = "valid"
      if status == "specified":
        try:
          port_a = normalize_port_contract_v1(edge["port_a"])
          port_b = normalize_port_contract_v1(edge["port_b"])
        except (TypeError, ValueError):
          # Topology remains usable, while untrusted attributes fail closed.
          status = "unknown"
          port_a = None
          port_b = None
          attribute_status = "invalid_to_unknown"
      elif status == "unknown" and edge["port_a"] is None and edge["port_b"] is None:
        port_a = None
        port_b = None
      else:
        raise ValueError("LinkCAD request-graph compiler port status differs")
      mobility = str(edge["requested_mobility"]).strip().lower()
      if mobility not in EXECUTABLE_MOBILITY_NAMES:
        raise ValueError("LinkCAD request-graph compiler mobility differs")
      connections.append({
          "role_a": role_a,
          "role_b": role_b,
          "port_contract_status": status,
          "port_a": port_a,
          "port_b": port_b,
          "attribute_status": attribute_status,
          "requested_mobility": mobility,
      })
    if not _graph_is_connected(allowed_roles, connections):
      raise ValueError("LinkCAD request-graph compiler graph is disconnected")
    output.append({
        "query_id": query_id,
        "roles": allowed_roles,
        "connections": connections,
    })
  output.sort(key=lambda row: list(expected).index(row["query_id"]))
  return tuple(output)


def _decode_json_text(text: str) -> dict[str, Any]:
  value = text.strip()
  if value.startswith("```"):
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
  parsed = json.loads(value)
  if not isinstance(parsed, dict):
    raise ValueError("LinkCAD request-graph compiler JSON root differs")
  return parsed


Transport = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class FrozenRequestGraphBatchV1:
  predictions: tuple[dict[str, Any], ...]
  receipt: dict[str, Any]


class FrozenLinkCADRequestGraphCompilerV1:
  """Zero-temperature whole-request graph compiler."""

  def __init__(
      self, *, model: str = DEFAULT_MODEL,
      base_url: str = "https://api.deepseek.com/v1",
      timeout_seconds: int = 180, transport: Transport | None = None,
  ) -> None:
    self.model = str(model)
    self.base_url = str(base_url).rstrip("/")
    self.timeout_seconds = int(timeout_seconds)
    self.transport = transport

  def compile(
      self, items: Sequence[Mapping[str, Any]],
  ) -> FrozenRequestGraphBatchV1:
    user_payload = compiler_user_payload_v1(items)
    request_payload = {
        "model": self.model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt_v1()},
            {"role": "user", "content": json.dumps(
                user_payload, ensure_ascii=False, sort_keys=True,
            )},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    raw = self.transport(request_payload) if self.transport else self._remote_call(
        request_payload
    )
    response = self._extract_response(raw)
    predictions = validate_compiler_response_v1(items, response)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "provider": "deepseek",
        "model": self.model,
        "temperature": 0.0,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256_v1(),
        "request_item_count": len(items),
        "request_sha256": canonical_sha256(user_payload),
        "response_sha256": canonical_sha256(response),
        "credential_material_included": False,
        "prediction_uses_candidate_or_reference_graph_identity": False,
    }
    return FrozenRequestGraphBatchV1(predictions, receipt)

  def _remote_call(self, payload: dict[str, Any]) -> Mapping[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
      raise RuntimeError("DEEPSEEK_API_KEY is required for the graph compiler")
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
        result = json.loads(response.read().decode("utf-8"))
    except (OSError, url_error.URLError, json.JSONDecodeError) as error:
      raise RuntimeError("Frozen LinkCAD request-graph call failed") from error
    if not isinstance(result, Mapping):
      raise RuntimeError("Frozen LinkCAD request-graph provider response differs")
    return result

  @staticmethod
  def _extract_response(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) == {"items"}:
      return dict(raw)
    try:
      content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
      raise ValueError("LinkCAD request-graph provider response differs") from error
    return _decode_json_text(str(content))


def _oriented_prediction(
    prediction: Mapping[str, Any], target: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
  same = (
      str(prediction["role_a"]) == str(target["role_a"])
      and str(prediction["role_b"]) == str(target["role_b"])
  )
  reverse = (
      str(prediction["role_a"]) == str(target["role_b"])
      and str(prediction["role_b"]) == str(target["role_a"])
  )
  if not same and not reverse:
    raise ValueError("LinkCAD request-graph evaluated pair differs")
  return prediction, reverse


def evaluate_request_graph_compiler_v1(
    targets: Mapping[str, Any], predictions: Mapping[str, Any],
) -> dict[str, Any]:
  """Evaluate graph structure and edge attributes with all queries retained."""

  target_rows = targets.get("items")
  prediction_rows = predictions.get("items")
  if not isinstance(target_rows, list) or not isinstance(prediction_rows, list):
    raise ValueError("LinkCAD request-graph evaluation rows differ")
  predicted = {str(row["query_id"]): row for row in prediction_rows}
  query_count = len(target_rows)
  role_exact = 0
  graph_exact = 0
  full_exact = 0
  matched_edges = 0
  predicted_edges = 0
  target_edges = 0
  mobility_exact = 0
  port_pair_exact = 0
  per_query = []
  for target in target_rows:
    query_id = str(target["query_id"])
    prediction = predicted.get(query_id)
    target_map = {
        _edge_pair(edge["role_a"], edge["role_b"]): edge
        for edge in target["connections"]
    }
    target_edges += len(target_map)
    if prediction is None:
      per_query.append({
          "query_id": query_id, "terminal_status": "compiler_error",
          "role_set_exact": False, "graph_exact": False,
          "full_request_exact": False,
      })
      continue
    roles_ok = set(prediction["roles"]) == set(target["roles"])
    role_exact += int(roles_ok)
    prediction_map = {
        _edge_pair(edge["role_a"], edge["role_b"]): edge
        for edge in prediction["connections"]
    }
    predicted_edges += len(prediction_map)
    shared = set(target_map) & set(prediction_map)
    matched_edges += len(shared)
    graph_ok = set(target_map) == set(prediction_map)
    graph_exact += int(graph_ok)
    all_attributes = graph_ok
    for pair in shared:
      pred, reverse = _oriented_prediction(prediction_map[pair], target_map[pair])
      target_edge = target_map[pair]
      mobility_ok = pred["requested_mobility"] == target_edge["requested_mobility"]
      mobility_exact += int(mobility_ok)
      status_ok = pred["port_contract_status"] == target_edge["port_contract_status"]
      if reverse:
        ports_ok = (
            pred["port_a"] == target_edge["port_b"]
            and pred["port_b"] == target_edge["port_a"]
        )
      else:
        ports_ok = (
            pred["port_a"] == target_edge["port_a"]
            and pred["port_b"] == target_edge["port_b"]
        )
      port_ok = status_ok and ports_ok
      port_pair_exact += int(port_ok)
      all_attributes = all_attributes and mobility_ok and port_ok
    request_ok = roles_ok and graph_ok and all_attributes
    full_exact += int(request_ok)
    per_query.append({
        "query_id": query_id, "terminal_status": "compiled",
        "role_set_exact": roles_ok, "graph_exact": graph_ok,
        "full_request_exact": request_ok,
        "target_edge_count": len(target_map),
        "predicted_edge_count": len(prediction_map),
        "matched_edge_count": len(shared),
    })
  precision = matched_edges / predicted_edges if predicted_edges else 0.0
  recall = matched_edges / target_edges if target_edges else 0.0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
  return {
      "schema_version": "linkcad_request_graph_evaluation.v1",
      "query_count": query_count,
      "prediction_count": len(predicted),
      "compiler_failure_count": query_count - len(predicted),
      "role_set_exact_count": role_exact,
      "role_set_exact_rate": role_exact / query_count if query_count else 0.0,
      "graph_exact_count": graph_exact,
      "graph_exact_rate": graph_exact / query_count if query_count else 0.0,
      "full_request_exact_count": full_exact,
      "full_request_exact_rate": full_exact / query_count if query_count else 0.0,
      "edge_precision": precision,
      "edge_recall": recall,
      "edge_f1": f1,
      "matched_edge_count": matched_edges,
      "predicted_edge_count": predicted_edges,
      "target_edge_count": target_edges,
      "matched_edge_mobility_exact_rate": (
          mobility_exact / matched_edges if matched_edges else 0.0
      ),
      "matched_edge_port_pair_exact_rate": (
          port_pair_exact / matched_edges if matched_edges else 0.0
      ),
      "per_query": per_query,
  }


def materialize_compiler_public_v1(
    public: Mapping[str, Any], predictions: Mapping[str, Any],
) -> dict[str, Any]:
  """Replace reference graph rows with compiler-produced rows for downstream use."""

  prediction_map = {
      str(row["query_id"]): row for row in predictions.get("items", [])
  }
  output_queries = []
  for query in public["queries"]:
    query_id = str(query["query_id"])
    prediction = prediction_map.get(query_id)
    base = {key: value for key, value in query.items() if key != "functional_edges"}
    if prediction is None:
      output_queries.append({
          **base, "compiler_status": "error", "functional_edges": [],
      })
      continue
    edges = []
    for ordinal, row in enumerate(prediction["connections"]):
      if row["port_contract_status"] == "specified":
        instruction = render_free_language_edge_v1(
            role_a=row["role_a"], role_b=row["role_b"],
            port_a=row["port_a"], port_b=row["port_b"],
            mobility=row["requested_mobility"], variant=0,
        )
      else:
        instruction = (
            f"Connect {row['role_a']} to {row['role_b']} with "
            f"{row['requested_mobility']} motion."
        )
      edges.append({
          "edge_id": f"edge_{ordinal:02d}",
          "role_a": row["role_a"], "role_b": row["role_b"],
          "instruction": instruction,
          "port_contract_status": row["port_contract_status"],
          "port_contract_a": row["port_a"],
          "port_contract_b": row["port_b"],
          "requested_mobility": row["requested_mobility"],
      })
    output_queries.append({
        **base, "compiler_status": "compiled", "functional_edges": edges,
    })
  return {
      **{key: value for key, value in public.items() if key != "queries"},
      "schema_version": "linkcad_compiler_conditioned_public.v1",
      "scope": "whole_request_compiler_output_for_downstream_linker",
      "queries": output_queries,
      "query_count": len(output_queries),
      "compiler_error_count": sum(
          row["compiler_status"] == "error" for row in output_queries
      ),
  }


def _request_edge_instruction(
    request: str, role_a: str, role_b: str,
) -> tuple[int, str]:
  """Recover the clauses for one generated edge without a reference graph."""

  chunks = [
      chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", str(request))
      if chunk.strip()
  ]
  matched = [
      (index, chunk) for index, chunk in enumerate(chunks)
      if str(role_a) in chunk and str(role_b) in chunk
  ]
  if not matched:
    raise ValueError("LinkCAD request-graph edge has no request clause")
  return matched[0][0], " ".join(chunk for _, chunk in matched)


def materialize_two_pass_compiler_public_v1(
    public: Mapping[str, Any], graph_inputs: Mapping[str, Any],
    graph_predictions: Mapping[str, Any], edge_predictions: Mapping[str, Any],
) -> dict[str, Any]:
  """Build linker input from generated topology and frozen edge parsing.

  The topology comes only from the whole-request compiler.  The second frozen
  pass parses the clauses selected by each generated role pair.  Reference
  functional edges are removed before either output is materialized.
  """

  requests = {
      str(row["query_id"]): str(row["request"])
      for row in graph_inputs.get("items", [])
  }
  graphs = {
      str(row["query_id"]): row for row in graph_predictions.get("items", [])
  }
  parsed_edges: dict[tuple[str, tuple[str, str]], Mapping[str, Any]] = {}
  for row in edge_predictions.get("items", []):
    key = (str(row["query_id"]), _edge_pair(row["role_a"], row["role_b"]))
    if key in parsed_edges:
      raise ValueError("LinkCAD two-pass edge parser pair repeats")
    parsed_edges[key] = row

  queries = []
  missing_graph = 0
  missing_edge_parse = 0
  for query in public["queries"]:
    query_id = str(query["query_id"])
    base = {key: value for key, value in query.items() if key != "functional_edges"}
    graph = graphs.get(query_id)
    request = requests.get(query_id)
    if graph is None or request is None:
      missing_graph += 1
      queries.append({
          **base, "compiler_status": "error", "functional_edges": [],
      })
      continue
    ordered = []
    for graph_edge in graph["connections"]:
      position, natural_instruction = _request_edge_instruction(
          request, graph_edge["role_a"], graph_edge["role_b"],
      )
      ordered.append((position, natural_instruction, graph_edge))
    ordered.sort(key=lambda row: (row[0], _edge_pair(
        row[2]["role_a"], row[2]["role_b"],
    )))
    edges = []
    for ordinal, (_, natural_instruction, graph_edge) in enumerate(ordered):
      key = (query_id, _edge_pair(
          graph_edge["role_a"], graph_edge["role_b"],
      ))
      parsed = parsed_edges.get(key)
      if parsed is None:
        missing_edge_parse += 1
        status = "unknown"
        port_a = None
        port_b = None
        mobility = graph_edge["requested_mobility"]
        instruction = natural_instruction
      else:
        same_orientation = (
            str(parsed["role_a"]) == str(graph_edge["role_a"])
            and str(parsed["role_b"]) == str(graph_edge["role_b"])
        )
        status = parsed["port_contract_status"]
        if same_orientation:
          port_a = parsed.get("port_contract_a")
          port_b = parsed.get("port_contract_b")
        else:
          port_a = parsed.get("port_contract_b")
          port_b = parsed.get("port_contract_a")
        mobility = parsed["requested_mobility"]
        mobility_text = render_mobility_instruction(
            mobility, role_a=graph_edge["role_a"],
            role_b=graph_edge["role_b"], variant=0,
        )
        instruction = (
            render_port_conditioned_instruction_v1(
                role_a=graph_edge["role_a"], role_b=graph_edge["role_b"],
                port_a=port_a, port_b=port_b,
                mobility_instruction=mobility_text,
            )
            if status == "specified"
            else mobility_text
        )
      edges.append({
          "edge_id": f"edge_{ordinal:02d}",
          "role_a": graph_edge["role_a"], "role_b": graph_edge["role_b"],
          "instruction": instruction,
          "port_contract_status": status,
          "port_contract_a": port_a, "port_contract_b": port_b,
          "requested_mobility": mobility,
      })
    queries.append({
        **base, "compiler_status": "compiled", "functional_edges": edges,
    })
  return {
      **{key: value for key, value in public.items() if key != "queries"},
      "schema_version": "linkcad_two_pass_compiler_public.v1",
      "scope": "whole_request_graph_then_frozen_edge_parser",
      "connection_graph_source": "frozen_whole_request_compiler",
      "edge_constraint_source": "frozen_edge_language_parser",
      "compiler_error_count": missing_graph,
      "missing_edge_parse_count": missing_edge_parse,
      "query_count": len(queries),
      "queries": queries,
  }


__all__ = [
    "FrozenLinkCADRequestGraphCompilerV1", "PREDICTION_SCHEMA_VERSION",
    "build_request_graph_benchmark_v1", "compiler_user_payload_v1",
    "evaluate_request_graph_compiler_v1", "materialize_compiler_public_v1",
    "materialize_two_pass_compiler_public_v1", "prompt_sha256_v1",
    "validate_compiler_response_v1",
]
