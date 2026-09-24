from neurocad.linkcad_request_graph_compiler_v1 import (
    FrozenLinkCADRequestGraphCompilerV1,
    build_request_graph_benchmark_v1,
    evaluate_request_graph_compiler_v1,
    materialize_compiler_public_v1,
    materialize_two_pass_compiler_public_v1,
    validate_compiler_response_v1,
)


def _port(kind="face", geometry="cylinder"):
  return {
      "schema_version": "linkcad_port_contract.v1",
      "primitive_kind": kind,
      "geometry_type": geometry,
      "size_class": "only",
      "adjacent_surface_types": ["plane"],
      "symmetry_class": "single",
  }


def _public():
  return {
      "schema_version": "fixture.v1",
      "queries": [{
          "query_id": "q0",
          "roles": [
              {"role_id": "shaft", "description": "shaft candidates"},
              {"role_id": "bearing", "description": "bearing candidates"},
              {"role_id": "gear", "description": "gear candidates"},
          ],
          "candidate_sets": {"shaft": [], "bearing": [], "gear": []},
          "functional_edges": [
              {
                  "edge_id": "edge_00", "role_a": "shaft", "role_b": "bearing",
                  "instruction": "Put the shaft through the bearing so it can turn.",
                  "port_contract_status": "specified",
                  "port_contract_a": _port(), "port_contract_b": _port(),
                  "requested_mobility": "revolute",
              },
              {
                  "edge_id": "edge_01", "role_a": "gear", "role_b": "shaft",
                  "instruction": "Fix the gear on the shaft.",
                  "port_contract_status": "unknown",
                  "port_contract_a": None, "port_contract_b": None,
                  "requested_mobility": "fixed",
              },
          ],
      }],
  }


def _response():
  return {"items": [{
      "query_id": "q0",
      "roles": ["shaft", "bearing", "gear"],
      "connections": [
          {
              "role_a": "shaft", "role_b": "bearing",
              "port_contract_status": "specified",
              "port_a": {key: value for key, value in _port().items()
                         if key != "schema_version"},
              "port_b": {key: value for key, value in _port().items()
                         if key != "schema_version"},
              "requested_mobility": "revolute",
          },
          {
              "role_a": "gear", "role_b": "shaft",
              "port_contract_status": "unknown",
              "port_a": None, "port_b": None,
              "requested_mobility": "fixed",
          },
      ],
  }]}


def test_benchmark_hides_reference_graph_from_compiler_input():
  inputs, targets, summary = build_request_graph_benchmark_v1(
      _public(), split_name="test",
  )
  item = inputs["items"][0]
  assert set(item) == {"query_id", "available_roles", "request"}
  assert "functional_edges" not in item
  assert "edge_id" not in str(item)
  assert targets["items"][0]["connections"][0]["role_a"] == "shaft"
  assert summary["query_count"] == 1
  assert summary["edge_count"] == 2


def test_validator_and_evaluator_accept_exact_graph():
  inputs, targets, _ = build_request_graph_benchmark_v1(
      _public(), split_name="test",
  )
  rows = validate_compiler_response_v1(inputs["items"], _response())
  predictions = {"items": list(rows)}
  evaluation = evaluate_request_graph_compiler_v1(targets, predictions)
  assert evaluation["graph_exact_rate"] == 1.0
  assert evaluation["full_request_exact_rate"] == 1.0
  assert evaluation["edge_f1"] == 1.0


def test_validator_rejects_spurious_role_and_duplicate_pair():
  inputs, _, _ = build_request_graph_benchmark_v1(_public(), split_name="test")
  spurious = _response()
  spurious["items"][0]["connections"][0]["role_b"] = "housing"
  try:
    validate_compiler_response_v1(inputs["items"], spurious)
  except ValueError as error:
    assert "connection identity" in str(error)
  else:
    raise AssertionError("spurious role was accepted")

  duplicate = _response()
  duplicate["items"][0]["connections"].append(
      dict(duplicate["items"][0]["connections"][0])
  )
  try:
    validate_compiler_response_v1(inputs["items"], duplicate)
  except ValueError as error:
    assert "connection identity" in str(error)
  else:
    raise AssertionError("duplicate pair was accepted")


def test_invalid_port_attributes_fail_closed_without_losing_graph():
  inputs, _, _ = build_request_graph_benchmark_v1(_public(), split_name="test")
  response = _response()
  response["items"][0]["connections"][0]["port_a"] = {
      "candidate_id": "forbidden", "primitive_kind": "face",
  }
  rows = validate_compiler_response_v1(inputs["items"], response)
  edge = rows[0]["connections"][0]
  assert edge["port_contract_status"] == "unknown"
  assert edge["attribute_status"] == "invalid_to_unknown"
  assert edge["port_a"] is None and edge["port_b"] is None


def test_frozen_compiler_receipt_and_downstream_materialization():
  inputs, _, _ = build_request_graph_benchmark_v1(_public(), split_name="test")

  def transport(payload):
    assert payload["temperature"] == 0.0
    assert "functional_edges" not in payload["messages"][1]["content"]
    return _response()

  batch = FrozenLinkCADRequestGraphCompilerV1(transport=transport).compile(
      inputs["items"]
  )
  assert batch.receipt["credential_material_included"] is False
  conditioned = materialize_compiler_public_v1(
      _public(), {"items": list(batch.predictions)},
  )
  query = conditioned["queries"][0]
  assert query["compiler_status"] == "compiled"
  assert len(query["functional_edges"]) == 2
  assert query["functional_edges"][0]["requested_mobility"] == "revolute"


def test_missing_prediction_stays_in_evaluation_denominator():
  _, targets, _ = build_request_graph_benchmark_v1(_public(), split_name="test")
  evaluation = evaluate_request_graph_compiler_v1(targets, {"items": []})
  assert evaluation["query_count"] == 1
  assert evaluation["compiler_failure_count"] == 1
  assert evaluation["graph_exact_rate"] == 0.0


def test_two_pass_public_uses_generated_graph_and_edge_parser():
  inputs, _, _ = build_request_graph_benchmark_v1(_public(), split_name="test")
  graph_rows = validate_compiler_response_v1(inputs["items"], _response())
  edge_predictions = {"items": [
      {
          "query_id": "q0", "edge_id": "source_a",
          "role_a": "shaft", "role_b": "bearing",
          "port_contract_status": "specified",
          "port_contract_a": _port(), "port_contract_b": _port(),
          "requested_mobility": "revolute",
      },
      {
          "query_id": "q0", "edge_id": "source_b",
          "role_a": "gear", "role_b": "shaft",
          "port_contract_status": "unknown",
          "port_contract_a": None, "port_contract_b": None,
          "requested_mobility": "fixed",
      },
  ]}
  result = materialize_two_pass_compiler_public_v1(
      _public(), inputs, {"items": list(graph_rows)}, edge_predictions,
  )
  assert result["connection_graph_source"] == "frozen_whole_request_compiler"
  assert result["missing_edge_parse_count"] == 0
  assert [edge["edge_id"] for edge in result["queries"][0]["functional_edges"]] == [
      "edge_00", "edge_01",
  ]
  assert result["queries"][0]["functional_edges"][0][
      "port_contract_status"
  ] == "specified"
