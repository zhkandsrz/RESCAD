"""Frozen natural-language front end for target-ID-free LinkCAD contracts.

The linker consumes a small symbolic port vocabulary.  This module defines the
auditable boundary that asks a frozen language model to compile ordinary role
and interface descriptions into that vocabulary.  Neither requests nor
responses contain candidate identities, STEP hashes, primitive ordinals, or
source-joint identifiers.
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

from .linkcad_brep_encoder_v1 import SURFACE_TYPES
from .linkcad_factorized_model_v1 import EXECUTABLE_MOBILITY_NAMES
from .linkcad_port_contract_v1 import SCHEMA_VERSION as PORT_SCHEMA_VERSION
from .linkcad_primitive_graph_v2 import PRIMITIVE_TYPES


SCHEMA_VERSION = "linkcad_language_planner.v1"
INPUT_SCHEMA_VERSION = "linkcad_language_planner_inputs.v1"
PREDICTION_SCHEMA_VERSION = "linkcad_language_planner_predictions.v1"
PROMPT_VERSION = "linkcad_language_planner_prompt.v1"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-pro"

PORT_KEYS = (
    "primitive_kind", "geometry_type", "size_class",
    "adjacent_surface_types", "symmetry_class",
)
SIZE_CLASSES = ("only", "smallest", "small", "medium", "large", "largest")
PRIMITIVE_KINDS = ("face", "edge")
SYMMETRY_CLASSES = ("single", "repeated")
FORBIDDEN_IDENTITY_KEYS = {
    "candidate_id", "candidate_index", "primitive_ordinal",
    "primitive_orbit_ordinal", "source_joint_id", "occurrence_id",
    "body_id", "step_sha256", "path", "filename",
}
FORBIDDEN_IDENTITY_TEXT_TOKENS = FORBIDDEN_IDENTITY_KEYS - {"path", "filename"}

_MOBILITY_PROSE = {
    "fixed": "remain rigidly fixed with no relative motion",
    "revolute": "rotate about their shared axis without sliding",
    "prismatic": "slide along their shared axis without rotating",
    "cylindrical": "both rotate and slide along their shared axis",
    "planar": "translate in the shared plane and rotate about its normal",
    "ball": "rotate freely about a common center without translating",
    "pin_slot": "slide along the slot while rotating about the pin axis",
}


def canonical_json_bytes(value: Any) -> bytes:
  return json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
  ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _contains_forbidden_identity(value: Any) -> bool:
  if isinstance(value, Mapping):
    return any(
        str(key).lower() in FORBIDDEN_IDENTITY_KEYS
        or _contains_forbidden_identity(item)
        for key, item in value.items()
    )
  if isinstance(value, (list, tuple)):
    return any(_contains_forbidden_identity(item) for item in value)
  if isinstance(value, str):
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in FORBIDDEN_IDENTITY_TEXT_TOKENS)
  return False


def normalize_port_contract_v1(value: Mapping[str, Any]) -> dict[str, Any]:
  """Validate and canonicalize one identity-free symbolic port contract."""

  if not isinstance(value, Mapping) or _contains_forbidden_identity(value):
    raise ValueError("LinkCAD language-planner port exposes forbidden identity")
  allowed = set(PORT_KEYS) | {"schema_version"}
  if set(value) - allowed:
    raise ValueError("LinkCAD language-planner port fields differ")
  kind = str(value.get("primitive_kind", "")).strip().lower()
  geometry = str(value.get("geometry_type", "")).strip().lower()
  size = str(value.get("size_class", "")).strip().lower()
  symmetry = str(value.get("symmetry_class", "")).strip().lower()
  adjacency_raw = value.get("adjacent_surface_types")
  if kind not in PRIMITIVE_KINDS:
    raise ValueError("LinkCAD language-planner primitive kind differs")
  if geometry not in PRIMITIVE_TYPES:
    raise ValueError("LinkCAD language-planner geometry type differs")
  if size not in SIZE_CLASSES:
    raise ValueError("LinkCAD language-planner size class differs")
  if symmetry not in SYMMETRY_CLASSES:
    raise ValueError("LinkCAD language-planner symmetry class differs")
  if not isinstance(adjacency_raw, list) or len(adjacency_raw) > 2:
    raise ValueError("LinkCAD language-planner adjacency differs")
  adjacency = []
  for raw in adjacency_raw:
    surface = str(raw).strip().lower()
    if surface not in SURFACE_TYPES or surface in adjacency:
      raise ValueError("LinkCAD language-planner adjacent surface differs")
    adjacency.append(surface)
  adjacency.sort(key=SURFACE_TYPES.index)
  return {
      "schema_version": PORT_SCHEMA_VERSION,
      "primitive_kind": kind,
      "geometry_type": geometry,
      "size_class": size,
      "adjacent_surface_types": adjacency,
      "symmetry_class": symmetry,
  }


def _port_prose(contract: Mapping[str, Any]) -> str:
  port = normalize_port_contract_v1(contract)
  size = port["size_class"]
  prefix = "sole" if size == "only" else size
  result = f"its {prefix} {port['geometry_type']} {port['primitive_kind']}"
  adjacency = port["adjacent_surface_types"]
  if adjacency:
    result += " next to " + " and ".join(adjacency) + " surfaces"
  if port["symmetry_class"] == "repeated":
    result += " from the repeated symmetric group"
  else:
    result += " that occurs once"
  return result


def render_free_language_edge_v1(
    *, role_a: str, role_b: str, port_a: Mapping[str, Any],
    port_b: Mapping[str, Any], mobility: str, variant: int,
) -> str:
  """Render a deterministic prose variant without candidate/primitive IDs."""

  if mobility not in _MOBILITY_PROSE:
    raise ValueError("LinkCAD language-planner mobility differs")
  left = _port_prose(port_a)
  right = _port_prose(port_b)
  motion = _MOBILITY_PROSE[mobility]
  templates = (
      "Join {a} to {b}. On {a}, choose {left}; on {b}, choose {right}. "
      "The connection should {motion}.",
      "Build a connection between {a} and {b} that can {motion}. "
      "Use {left} on {a} together with {right} on {b}.",
      "For the {a}--{b} interface, pair {left} belonging to {a} with "
      "{right} belonging to {b}; allow the parts to {motion}.",
      "Connect {a} through {left}, and connect {b} through {right}. "
      "After assembly they must {motion}.",
      "The port of {a} is {left}. The matching port of {b} is {right}. "
      "Create their joint so the two parts {motion}.",
      "Make {a} and {b} meet at the following local features: {left} for "
      "{a}, {right} for {b}. Their permitted relative motion is to {motion}.",
  )
  return templates[int(variant) % len(templates)].format(
      a=role_a, b=role_b, left=left, right=right, motion=motion,
  )


def build_language_planner_benchmark_v1(
    public: Mapping[str, Any], *, split_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  """Separate natural-language inputs from reference symbolic contracts."""

  queries = public.get("queries")
  if not isinstance(queries, list) or not queries:
    raise ValueError("LinkCAD language-planner public queries differ")
  inputs = []
  targets = []
  for query in queries:
    query_id = str(query["query_id"])
    for edge in query["functional_edges"]:
      edge_id = str(edge["edge_id"])
      role_a = str(edge["role_a"])
      role_b = str(edge["role_b"])
      status = str(edge.get("port_contract_status", "unknown"))
      mobility = str(edge.get("requested_mobility", ""))
      if mobility not in EXECUTABLE_MOBILITY_NAMES:
        raise ValueError("LinkCAD language-planner reference mobility differs")
      if status == "specified":
        port_a = normalize_port_contract_v1(edge["port_contract_a"])
        port_b = normalize_port_contract_v1(edge["port_contract_b"])
        variant = int(
            hashlib.sha256(f"{split_name}:{query_id}:{edge_id}".encode()).hexdigest(),
            16,
        ) % 6
        instruction = render_free_language_edge_v1(
            role_a=role_a, role_b=role_b, port_a=port_a, port_b=port_b,
            mobility=mobility, variant=variant,
        )
      elif status == "unknown":
        port_a = None
        port_b = None
        instruction = str(edge["instruction"])
      else:
        raise ValueError("LinkCAD language-planner reference status differs")
      inputs.append({
          "query_id": query_id,
          "edge_id": edge_id,
          "role_a": role_a,
          "role_b": role_b,
          "instruction": instruction,
      })
      targets.append({
          "query_id": query_id,
          "edge_id": edge_id,
          "role_a": role_a,
          "role_b": role_b,
          "port_contract_status": status,
          "port_contract_a": port_a,
          "port_contract_b": port_b,
          "requested_mobility": mobility,
      })
  input_payload = {
      "schema_version": INPUT_SCHEMA_VERSION,
      "scope": "target_id_free_natural_language_only",
      "split_name": str(split_name),
      "item_count": len(inputs),
      "contains_structured_port_targets": False,
      "items": inputs,
  }
  target_payload = {
      "schema_version": "linkcad_language_planner_targets.v1",
      "scope": "reference_only_not_planner_input",
      "split_name": str(split_name),
      "item_count": len(targets),
      "items": targets,
  }
  summary = {
      "schema_version": "linkcad_language_planner_benchmark_summary.v1",
      "split_name": str(split_name),
      "query_count": len({row["query_id"] for row in inputs}),
      "edge_count": len(inputs),
      "specified_edge_count": sum(
          row["port_contract_status"] == "specified" for row in targets
      ),
      "unknown_edge_count": sum(
          row["port_contract_status"] == "unknown" for row in targets
      ),
      "input_sha256": canonical_sha256(input_payload),
      "target_sha256": canonical_sha256(target_payload),
      "paraphrase_template_count": 6,
      "evaluation_scope": "posthoc_frozen_frontend_audit",
  }
  return input_payload, target_payload, summary


def system_prompt_v1() -> str:
  return """You compile natural-language CAD assembly requests into a small symbolic port vocabulary. Return one JSON object and no prose. Preserve every query_id, edge_id, role_a, and role_b exactly. For each item, extract requested_mobility and the two local port descriptions. Allowed primitive_kind: face, edge. Allowed geometry_type: plane, cylinder, cone, sphere, torus, spline, other, circle, line, ellipse, bspline_curve, other_curve. Allowed size_class: only, smallest, small, medium, large, largest. Allowed adjacent_surface_types: plane, cylinder, cone, sphere, torus, spline, other; emit zero to two values. Allowed symmetry_class: single, repeated. Allowed requested_mobility: fixed, revolute, prismatic, cylindrical, planar, ball, pin_slot. If the text does not describe both ports, set port_contract_status to unknown and port_a and port_b to null; still extract mobility when it is unambiguous. Do not invent candidate IDs, primitive indices, file names, paths, hashes, or source-joint IDs. Output exactly {\"items\":[{\"query_id\":str,\"edge_id\":str,\"role_a\":str,\"role_b\":str,\"port_contract_status\":\"specified\"|\"unknown\",\"port_a\":object|null,\"port_b\":object|null,\"requested_mobility\":str}]} where each port object has exactly primitive_kind, geometry_type, size_class, adjacent_surface_types, symmetry_class."""


def prompt_sha256_v1() -> str:
  return hashlib.sha256(system_prompt_v1().encode("utf-8")).hexdigest()


def planner_user_payload_v1(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  normalized = []
  for row in items:
    if set(row) != {"query_id", "edge_id", "role_a", "role_b", "instruction"}:
      raise ValueError("LinkCAD language-planner input fields differ")
    normalized.append({key: str(row[key]) for key in (
        "query_id", "edge_id", "role_a", "role_b", "instruction",
    )})
  if not normalized or _contains_forbidden_identity(normalized):
    raise ValueError("LinkCAD language-planner input identity boundary differs")
  return {"schema_version": INPUT_SCHEMA_VERSION, "items": normalized}


def validate_planner_response_v1(
    request_items: Sequence[Mapping[str, Any]], response: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
  """Fail closed unless the response covers exactly the requested edge set."""

  if not isinstance(response, Mapping) or set(response) != {"items"}:
    raise ValueError("LinkCAD language-planner response envelope differs")
  rows = response["items"]
  if not isinstance(rows, list) or _contains_forbidden_identity(rows):
    raise ValueError("LinkCAD language-planner response identity boundary differs")
  expected = {
      (str(row["query_id"]), str(row["edge_id"])): row
      for row in request_items
  }
  if len(expected) != len(request_items) or len(rows) != len(expected):
    raise ValueError("LinkCAD language-planner response coverage differs")
  allowed_keys = {
      "query_id", "edge_id", "role_a", "role_b", "port_contract_status",
      "port_a", "port_b", "requested_mobility",
  }
  output = []
  seen = set()
  for row in rows:
    if not isinstance(row, Mapping) or set(row) != allowed_keys:
      raise ValueError("LinkCAD language-planner response item fields differ")
    key = (str(row["query_id"]), str(row["edge_id"]))
    if key in seen or key not in expected:
      raise ValueError("LinkCAD language-planner response identity differs")
    seen.add(key)
    source = expected[key]
    if (
        str(row["role_a"]) != str(source["role_a"])
        or str(row["role_b"]) != str(source["role_b"])
    ):
      raise ValueError("LinkCAD language-planner response roles differ")
    mobility = str(row["requested_mobility"]).strip().lower()
    if mobility not in EXECUTABLE_MOBILITY_NAMES:
      raise ValueError("LinkCAD language-planner response mobility differs")
    status = str(row["port_contract_status"]).strip().lower()
    if status == "specified":
      port_a = normalize_port_contract_v1(row["port_a"])
      port_b = normalize_port_contract_v1(row["port_b"])
    elif status == "unknown" and row["port_a"] is None and row["port_b"] is None:
      port_a = None
      port_b = None
    else:
      raise ValueError("LinkCAD language-planner response status differs")
    output.append({
        "query_id": key[0], "edge_id": key[1],
        "role_a": str(row["role_a"]), "role_b": str(row["role_b"]),
        "port_contract_status": status,
        "port_contract_a": port_a, "port_contract_b": port_b,
        "requested_mobility": mobility,
    })
  output.sort(key=lambda row: list(expected).index((row["query_id"], row["edge_id"])))
  return tuple(output)


def _decode_json_text(text: str) -> dict[str, Any]:
  value = text.strip()
  if value.startswith("```"):
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
  parsed = json.loads(value)
  if not isinstance(parsed, dict):
    raise ValueError("LinkCAD language-planner JSON root differs")
  return parsed


Transport = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class FrozenLanguagePlannerBatchV1:
  predictions: tuple[dict[str, Any], ...]
  receipt: dict[str, Any]


class FrozenLinkCADLanguagePlannerV1:
  """A zero-temperature, prompt-committed JSON compiler."""

  def __init__(
      self, *, model: str = DEFAULT_MODEL,
      base_url: str = "https://api.deepseek.com/v1",
      timeout_seconds: int = 120, transport: Transport | None = None,
  ) -> None:
    self.model = str(model)
    self.base_url = str(base_url).rstrip("/")
    self.timeout_seconds = int(timeout_seconds)
    self.transport = transport

  def plan(self, items: Sequence[Mapping[str, Any]]) -> FrozenLanguagePlannerBatchV1:
    user_payload = planner_user_payload_v1(items)
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
    raw = (
        self.transport(request_payload)
        if self.transport is not None else self._remote_call(request_payload)
    )
    response = self._extract_response(raw)
    predictions = validate_planner_response_v1(items, response)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "provider": DEFAULT_PROVIDER,
        "model": self.model,
        "temperature": 0.0,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256_v1(),
        "request_item_count": len(items),
        "request_sha256": canonical_sha256(user_payload),
        "response_sha256": canonical_sha256(response),
        "credential_material_included": False,
        "prediction_uses_candidate_or_target_identity": False,
    }
    return FrozenLanguagePlannerBatchV1(predictions, receipt)

  def _remote_call(self, payload: dict[str, Any]) -> Mapping[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
      raise RuntimeError("DEEPSEEK_API_KEY is required for the frozen planner")
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
      raise RuntimeError("Frozen LinkCAD language-planner call failed") from error
    if not isinstance(result, Mapping):
      raise RuntimeError("Frozen LinkCAD language-planner response differs")
    return result

  @staticmethod
  def _extract_response(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) == {"items"}:
      return dict(raw)
    try:
      content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
      raise ValueError("LinkCAD language-planner provider response differs") from error
    return _decode_json_text(str(content))


def evaluate_language_planner_v1(
    targets: Mapping[str, Any], predictions: Mapping[str, Any],
) -> dict[str, Any]:
  target_rows = targets.get("items")
  prediction_rows = predictions.get("items")
  if not isinstance(target_rows, list) or not isinstance(prediction_rows, list):
    raise ValueError("LinkCAD language-planner evaluation rows differ")
  predicted = {
      (str(row["query_id"]), str(row["edge_id"])): row
      for row in prediction_rows
  }
  field_correct = {key: 0 for key in PORT_KEYS}
  endpoint_count = 0
  specified_count = 0
  status_correct = 0
  mobility_correct = 0
  port_pair_correct = 0
  full_edge_correct = 0
  per_query: dict[str, list[bool]] = {}
  for target in target_rows:
    key = (str(target["query_id"]), str(target["edge_id"]))
    row = predicted.get(key)
    expected_status = str(target["port_contract_status"])
    status_hit = bool(row and row.get("port_contract_status") == expected_status)
    status_correct += int(status_hit)
    mobility_hit = bool(
        row and row.get("requested_mobility") == target["requested_mobility"]
    )
    mobility_correct += int(mobility_hit)
    pair_hit = expected_status == "unknown" and status_hit
    if expected_status == "specified":
      specified_count += 1
      pair_hit = bool(row and status_hit)
      for side in ("a", "b"):
        endpoint_count += 1
        expected_port = normalize_port_contract_v1(target[f"port_contract_{side}"])
        predicted_port = (
            row.get(f"port_contract_{side}") if row is not None else None
        )
        if predicted_port is None:
          pair_hit = False
          continue
        actual_port = normalize_port_contract_v1(predicted_port)
        pair_hit = pair_hit and actual_port == expected_port
        for field in PORT_KEYS:
          field_correct[field] += int(actual_port[field] == expected_port[field])
    port_pair_correct += int(pair_hit)
    edge_hit = bool(pair_hit and mobility_hit)
    full_edge_correct += int(edge_hit)
    per_query.setdefault(key[0], []).append(edge_hit)
  total = len(target_rows)
  if not total:
    raise ValueError("LinkCAD language-planner evaluation is empty")
  return {
      "schema_version": "linkcad_language_planner_evaluation.v1",
      "edge_count": total,
      "specified_edge_count": specified_count,
      "prediction_count": len(prediction_rows),
      "status_accuracy": status_correct / total,
      "mobility_accuracy": mobility_correct / total,
      "port_pair_exact_accuracy": port_pair_correct / total,
      "full_edge_exact_accuracy": full_edge_correct / total,
      "all_edges_exact_query_accuracy": sum(all(values) for values in per_query.values()) / len(per_query),
      "specified_endpoint_field_accuracy": {
          key: (field_correct[key] / endpoint_count if endpoint_count else 0.0)
          for key in PORT_KEYS
      },
      "missing_prediction_count": total - sum(
          (str(row["query_id"]), str(row["edge_id"])) in predicted
          for row in target_rows
      ),
  }


__all__ = [
    "DEFAULT_MODEL", "FrozenLanguagePlannerBatchV1",
    "FrozenLinkCADLanguagePlannerV1", "INPUT_SCHEMA_VERSION", "PORT_KEYS",
    "PREDICTION_SCHEMA_VERSION", "PROMPT_VERSION", "SCHEMA_VERSION",
    "build_language_planner_benchmark_v1", "canonical_sha256",
    "evaluate_language_planner_v1", "normalize_port_contract_v1",
    "planner_user_payload_v1", "prompt_sha256_v1",
    "render_free_language_edge_v1", "system_prompt_v1",
    "validate_planner_response_v1",
]
