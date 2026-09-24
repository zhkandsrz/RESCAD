from __future__ import annotations

import pytest

from neurocad.linkcad_language_planner_v1 import (
    FrozenLinkCADLanguagePlannerV1,
    build_language_planner_benchmark_v1,
    evaluate_language_planner_v1,
    normalize_port_contract_v1,
    planner_user_payload_v1,
    validate_planner_response_v1,
)
from neurocad.tools.materialize_linkcad_planner_conditioned_public_v1 import (
    materialize_planner_conditioned_public_v1,
)


def _port(kind="face", geometry="cylinder", size="smallest", repeated=False):
  return {
      "schema_version": "linkcad_port_contract.v1",
      "primitive_kind": kind,
      "geometry_type": geometry,
      "size_class": size,
      "adjacent_surface_types": ["plane", "cylinder"],
      "symmetry_class": "repeated" if repeated else "single",
  }


def _public():
  return {
      "queries": [{
          "query_id": "q1",
          "functional_edges": [{
              "edge_id": "e1", "role_a": "shaft", "role_b": "housing",
              "instruction": "Connect the shaft and housing.",
              "port_contract_status": "specified",
              "port_contract_a": _port(),
              "port_contract_b": _port("edge", "circle", "largest", True),
              "requested_mobility": "revolute",
          }],
      }],
  }


def test_benchmark_separates_natural_language_from_structured_target():
  inputs, targets, summary = build_language_planner_benchmark_v1(
      _public(), split_name="heldout",
  )
  row = inputs["items"][0]
  assert set(row) == {"query_id", "edge_id", "role_a", "role_b", "instruction"}
  assert "candidate_id" not in str(row)
  assert "primitive_ordinal" not in str(row)
  assert targets["items"][0]["port_contract_a"]["geometry_type"] == "cylinder"
  assert summary["specified_edge_count"] == 1


def test_contract_validation_canonicalizes_adjacency_and_rejects_identity():
  port = _port()
  port["adjacent_surface_types"] = ["cylinder", "plane"]
  assert normalize_port_contract_v1(port)["adjacent_surface_types"] == [
      "plane", "cylinder",
  ]
  port["candidate_id"] = "secret"
  with pytest.raises(ValueError, match="identity"):
    normalize_port_contract_v1(port)

  with pytest.raises(ValueError, match="identity"):
    planner_user_payload_v1([{
        "query_id": "q1", "edge_id": "e1", "role_a": "a", "role_b": "b",
        "instruction": "Use candidate ID 7.",
    }])
  payload = planner_user_payload_v1([{
      "query_id": "q1", "edge_id": "e1", "role_a": "a", "role_b": "b",
      "instruction": "Let the pin travel along the slot path.",
  }])
  assert payload["items"][0]["edge_id"] == "e1"


def test_response_validation_requires_exact_coverage_and_roles():
  inputs, _, _ = build_language_planner_benchmark_v1(
      _public(), split_name="heldout",
  )
  source = inputs["items"]
  response = {"items": [{
      "query_id": "q1", "edge_id": "e1", "role_a": "shaft",
      "role_b": "housing", "port_contract_status": "specified",
      "port_a": {key: value for key, value in _port().items() if key != "schema_version"},
      "port_b": {
          key: value for key, value in _port("edge", "circle", "largest", True).items()
          if key != "schema_version"
      },
      "requested_mobility": "revolute",
  }]}
  parsed = validate_planner_response_v1(source, response)
  assert parsed[0]["port_contract_a"]["schema_version"] == "linkcad_port_contract.v1"
  response["items"][0]["role_b"] = "other"
  with pytest.raises(ValueError, match="roles"):
    validate_planner_response_v1(source, response)


def test_frozen_planner_receipt_excludes_secret_and_binds_prompt():
  inputs, _, _ = build_language_planner_benchmark_v1(
      _public(), split_name="heldout",
  )
  def transport(_payload):
    return {"items": [{
        "query_id": "q1", "edge_id": "e1", "role_a": "shaft",
        "role_b": "housing", "port_contract_status": "specified",
        "port_a": {key: value for key, value in _port().items() if key != "schema_version"},
        "port_b": {
            key: value for key, value in _port("edge", "circle", "largest", True).items()
            if key != "schema_version"
        },
        "requested_mobility": "revolute",
    }]}
  batch = FrozenLinkCADLanguagePlannerV1(transport=transport).plan(inputs["items"])
  assert batch.receipt["temperature"] == 0.0
  assert len(batch.receipt["prompt_sha256"]) == 64
  assert batch.receipt["credential_material_included"] is False
  assert "api_key" not in str(batch.receipt).lower()


def test_evaluation_reports_end_to_end_edge_and_query_exactness():
  inputs, targets, _ = build_language_planner_benchmark_v1(
      _public(), split_name="heldout",
  )
  prediction = {"items": [{
      "query_id": "q1", "edge_id": "e1", "role_a": "shaft",
      "role_b": "housing", "port_contract_status": "specified",
      "port_contract_a": _port(),
      "port_contract_b": _port("edge", "circle", "largest", True),
      "requested_mobility": "revolute",
  }]}
  metrics = evaluate_language_planner_v1(targets, prediction)
  assert metrics["full_edge_exact_accuracy"] == 1.0
  assert metrics["all_edges_exact_query_accuracy"] == 1.0
  assert metrics["specified_endpoint_field_accuracy"]["geometry_type"] == 1.0


def test_planner_conditioned_protocol_canonicalizes_before_linking():
  inputs, _, _ = build_language_planner_benchmark_v1(
      _public(), split_name="heldout",
  )
  predictions = {"items": [{
      "query_id": "q1", "edge_id": "e1", "role_a": "shaft",
      "role_b": "housing", "port_contract_status": "specified",
      "port_contract_a": _port(),
      "port_contract_b": _port("edge", "circle", "largest", True),
      "requested_mobility": "revolute",
  }]}
  output = materialize_planner_conditioned_public_v1(
      public=_public(), inputs=inputs, predictions=predictions,
  )
  edge = output["queries"][0]["functional_edges"][0]
  assert edge["instruction"].startswith("Use the smallest cylinder face")
  assert "Connect shaft to housing so it can rotate" in edge["instruction"]
  assert edge["instruction"] != inputs["items"][0]["instruction"]

  natural = materialize_planner_conditioned_public_v1(
      public=_public(), inputs=inputs, predictions=predictions,
      linker_text_mode="natural",
  )
  assert natural["linker_text_mode"] == "natural"
  assert natural["queries"][0]["functional_edges"][0]["instruction"] == (
      inputs["items"][0]["instruction"]
  )
