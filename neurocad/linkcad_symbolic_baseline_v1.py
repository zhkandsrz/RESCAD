"""Target-free symbolic baselines and ambiguity accounting for LinkCAD.

The baselines deliberately have no trainable parameters.  They use only the
public port contract and intrinsic B-rep primitive descriptions, then apply the
same assignment-beam and per-edge interface budgets as the learned linker.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from statistics import mean, median
from typing import Any, Mapping

from .linkcad_contract_strength_v1 import port_contract_matches_v1
from .linkcad_port_contract_v1 import describe_port_v1
from .linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2


PREDICTION_SCHEMA_VERSION = "linkcad_symbolic_predictions.v1"
EXTERNAL_PREDICTION_SCHEMA_VERSION = "linkcad_external_ranked_predictions.v1"
AMBIGUITY_SCHEMA_VERSION = "linkcad_symbolic_ambiguity_audit.v1"
POLICIES = ("minimum_port_ambiguity", "deterministic_hash")


def _canonical(value: Any) -> str:
  return json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  )


def _sha(value: Any) -> str:
  return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _quantile(values: list[int], probability: float) -> int:
  if not values:
    return 0
  ordered = sorted(values)
  index = max(0, math.ceil(probability * len(ordered)) - 1)
  return int(ordered[index])


def _matching_primitives(
    *, graph: Any, contract: Mapping[str, Any] | None,
) -> tuple[int, ...]:
  if contract is None:
    return tuple(range(int(graph.primitive_features.shape[0])))
  return tuple(
      ordinal
      for ordinal in range(int(graph.primitive_features.shape[0]))
      if port_contract_matches_v1(describe_port_v1(graph, ordinal), contract)
  )


def _primitive_sort_key(graph: Any, ordinal: int) -> tuple[Any, ...]:
  features = tuple(
      round(float(value), 10)
      for value in graph.primitive_features[ordinal].tolist()
  )
  return (
      str(graph.primitive_kinds[ordinal]), features,
      len(graph.primitive_members[ordinal]), int(ordinal),
  )


def _support_family(
    left_type: str, right_type: str,
) -> str:
  axes = {"circle", "cylinder", "cone", "line"}
  if left_type == right_type == "line":
    return "linear_support"
  if left_type in axes and right_type in axes:
    return "axial_support"
  if left_type == right_type == "plane":
    return "planar_support"
  if {left_type, right_type} <= axes | {"plane"} and (
      left_type == "plane" or right_type == "plane"
  ):
    return "axis_plane_support"
  return "other"


def _query_symbolic_state(
    public_query: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
) -> dict[str, Any]:
  role_ids = tuple(str(row["role_id"]) for row in public_query["roles"])
  role_index = {role_id: index for index, role_id in enumerate(role_ids)}
  candidate_rows = tuple(
      tuple(public_query["candidate_sets"][role_id]) for role_id in role_ids
  )
  counts = [len(rows) for rows in candidate_rows]
  if not counts or min(counts) < 2:
    raise ValueError("LinkCAD symbolic candidate-set sizes differ")
  graphs = tuple(tuple(
      cache.graph(str(candidate["step_sha256"])) for candidate in rows
  ) for rows in candidate_rows)
  endpoint_matches: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
  edge_rows = []
  for edge_ordinal, edge in enumerate(public_query["functional_edges"]):
    sides = []
    for side, role_field in (("a", "role_a"), ("b", "role_b")):
      role = role_index[str(edge[role_field])]
      contract = (
          edge.get(f"port_contract_{side}")
          if edge.get("port_contract_status") == "specified" else None
      )
      matches = tuple(
          () if graph is None else _matching_primitives(
              graph=graph, contract=contract,
          )
          for graph in graphs[role]
      )
      endpoint_matches[(edge_ordinal, 0 if side == "a" else 1)] = matches
      sides.append({
          "side": side,
          "role_id": role_ids[role],
          "matching_primitive_counts": [len(row) for row in matches],
          "compatible_candidate_count": sum(bool(row) for row in matches),
      })
    edge_rows.append({"edge_id": str(edge["edge_id"]), "sides": sides})
  return {
      "role_ids": role_ids,
      "role_index": role_index,
      "candidate_rows": candidate_rows,
      "graphs": graphs,
      "endpoint_matches": endpoint_matches,
      "edge_rows": edge_rows,
  }


def _feasible_assignments(
    public_query: Mapping[str, Any], state: Mapping[str, Any],
) -> list[tuple[tuple[int, ...], float, str]]:
  candidate_rows = state["candidate_rows"]
  role_index = state["role_index"]
  endpoint_matches = state["endpoint_matches"]
  assignments = []
  for assignment in itertools.product(*[range(len(rows)) for rows in candidate_rows]):
    ambiguity = 0.0
    feasible = True
    for edge_ordinal, edge in enumerate(public_query["functional_edges"]):
      left = assignment[role_index[str(edge["role_a"])]]
      right = assignment[role_index[str(edge["role_b"])]]
      left_count = len(endpoint_matches[(edge_ordinal, 0)][left])
      right_count = len(endpoint_matches[(edge_ordinal, 1)][right])
      if left_count == 0 or right_count == 0:
        feasible = False
        break
      ambiguity += math.log(left_count) + math.log(right_count)
    if not feasible:
      continue
    identity = tuple(
        str(candidate_rows[role][candidate]["step_sha256"])
        for role, candidate in enumerate(assignment)
    )
    assignments.append((assignment, -ambiguity, _sha(identity)))
  return assignments


def _ordered_assignments(
    assignments: list[tuple[tuple[int, ...], float, str]], *, policy: str,
) -> list[tuple[tuple[int, ...], float]]:
  if policy == "minimum_port_ambiguity":
    ordered = sorted(assignments, key=lambda row: (-row[1], row[2]))
  elif policy == "deterministic_hash":
    ordered = sorted(assignments, key=lambda row: row[2])
  else:
    raise ValueError("LinkCAD symbolic policy differs")
  return [(assignment, score) for assignment, score, _ in ordered]


def _hypotheses_from_assignments(
    *, public_query: Mapping[str, Any], state: Mapping[str, Any],
    assignments: list[tuple[tuple[int, ...], float]],
    interface_alternatives_per_edge: int,
) -> list[dict[str, Any]]:
  hypotheses = []
  for rank, (assignment, score) in enumerate(assignments):
    candidate_by_role = {
        role_id: str(state["candidate_rows"][role][assignment[role]]["candidate_id"])
        for role, role_id in enumerate(state["role_ids"])
    }
    edge_programs = []
    for edge_ordinal, edge in enumerate(public_query["functional_edges"]):
      role_a = state["role_index"][str(edge["role_a"])]
      role_b = state["role_index"][str(edge["role_b"])]
      candidate_a, candidate_b = assignment[role_a], assignment[role_b]
      graph_a = state["graphs"][role_a][candidate_a]
      graph_b = state["graphs"][role_b][candidate_b]
      matches_a = state["endpoint_matches"][(edge_ordinal, 0)][candidate_a]
      matches_b = state["endpoint_matches"][(edge_ordinal, 1)][candidate_b]
      alternatives = sorted(
          itertools.product(matches_a, matches_b),
          key=lambda pair: (
              _primitive_sort_key(graph_a, pair[0]),
              _primitive_sort_key(graph_b, pair[1]),
          ),
      )[:interface_alternatives_per_edge]
      best = alternatives[0] if alternatives else (-1, -1)
      support = (
          _support_family(
              graph_a.primitive_type(best[0]), graph_b.primitive_type(best[1])
          ) if min(best) >= 0 else "other"
      )
      edge_programs.append({
          "edge_id": str(edge["edge_id"]),
          "support_family": support,
          "mobility": str(edge.get("requested_mobility", "fixed")),
          "interface_primitive_a": int(best[0]),
          "interface_primitive_b": int(best[1]),
          "primitive_alternatives": [
              {
                  "rank": alternative_rank,
                  "interface_primitive_a": int(pair[0]),
                  "interface_primitive_b": int(pair[1]),
                  "model_score": 0.0,
              }
              for alternative_rank, pair in enumerate(alternatives)
          ],
      })
    hypothesis = {
        "rank": rank, "score": float(score),
        "candidate_by_role": candidate_by_role,
        "edge_programs": edge_programs,
    }
    hypothesis["hypothesis_payload_sha256"] = _sha(hypothesis)
    hypotheses.append(hypothesis)
  return hypotheses


def materialize_symbolic_predictions_v1(
    *, public: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
    policy: str = "minimum_port_ambiguity", beam_size: int = 25,
    interface_alternatives_per_edge: int = 8,
) -> dict[str, Any]:
  """Create a matched-budget prediction ledger without learned scores."""

  if public.get("contains_private_targets") is not False:
    raise ValueError("LinkCAD symbolic public prediction scope differs")
  if policy not in POLICIES or beam_size < 1 or interface_alternatives_per_edge < 1:
    raise ValueError("LinkCAD symbolic prediction configuration differs")
  rows = []
  for public_query in public["queries"]:
    state = _query_symbolic_state(public_query, cache)
    assignments = _ordered_assignments(
        _feasible_assignments(public_query, state), policy=policy,
    )[:beam_size]
    hypotheses = _hypotheses_from_assignments(
        public_query=public_query, state=state, assignments=assignments,
        interface_alternatives_per_edge=interface_alternatives_per_edge,
    )
    row = {
        "query_id": str(public_query["query_id"]),
        "terminal_status": "prediction_ready" if hypotheses else "no_candidate",
        "hypotheses": hypotheses,
    }
    row["row_payload_sha256"] = _sha(row)
    rows.append(row)
  payload = {
      "schema_version": PREDICTION_SCHEMA_VERSION,
      "scope": "public_contract_and_intrinsic_brep_only_predictions",
      "policy": policy,
      "selection_uses_private_targets_or_model_outputs": False,
      "private_targets_opened": False,
      "checkpoint_sha256": "not_applicable_symbolic_only",
      "model_kind": "symbolic_only",
      "beam_size": beam_size,
      "interface_alternatives_per_edge": interface_alternatives_per_edge,
      "query_count": len(rows),
      "rows": rows,
  }
  payload["prediction_payload_sha256"] = _sha(payload)
  return payload


def materialize_external_assignment_predictions_v1(
    *, public: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
    role_rankings_by_query: Mapping[str, Mapping[str, list[str]]],
    method_id: str, model_id: str, beam_size: int = 25,
    interface_alternatives_per_edge: int = 8,
    failed_query_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
  """Turn public external role rankings into a matched LinkCAD beam."""

  if public.get("contains_private_targets") is not False:
    raise ValueError("LinkCAD external ranking public scope differs")
  if beam_size < 1 or interface_alternatives_per_edge < 1:
    raise ValueError("LinkCAD external ranking budget differs")
  rows = []
  failed = {str(value) for value in failed_query_ids}
  for public_query in public["queries"]:
    query_id = str(public_query["query_id"])
    if query_id in failed:
      row = {
          "query_id": query_id,
          "terminal_status": "no_candidate",
          "hypotheses": [],
      }
      row["row_payload_sha256"] = _sha(row)
      rows.append(row)
      continue
    state = _query_symbolic_state(public_query, cache)
    rankings = role_rankings_by_query.get(query_id)
    if not isinstance(rankings, Mapping) or set(rankings) != set(state["role_ids"]):
      raise ValueError("LinkCAD external ranking role domain differs")
    rank_by_role = []
    for role_index, role_id in enumerate(state["role_ids"]):
      expected = [
          str(row["candidate_id"]) for row in state["candidate_rows"][role_index]
      ]
      ordered = [str(value) for value in rankings[role_id]]
      if len(ordered) != len(set(ordered)) or set(ordered) != set(expected):
        raise ValueError("LinkCAD external ranking candidate domain differs")
      rank_by_role.append({candidate_id: rank for rank, candidate_id in enumerate(ordered)})
    scored = []
    for assignment, _, identity in _feasible_assignments(public_query, state):
      score = -float(sum(
          rank_by_role[role][str(
              state["candidate_rows"][role][candidate]["candidate_id"]
          )]
          for role, candidate in enumerate(assignment)
      ))
      scored.append((assignment, score, identity))
    assignments = [
        (assignment, score)
        for assignment, score, _ in sorted(
            scored, key=lambda row: (-row[1], row[2])
        )[:beam_size]
    ]
    hypotheses = _hypotheses_from_assignments(
        public_query=public_query, state=state, assignments=assignments,
        interface_alternatives_per_edge=interface_alternatives_per_edge,
    )
    row = {
        "query_id": query_id,
        "terminal_status": "prediction_ready" if hypotheses else "no_candidate",
        "hypotheses": hypotheses,
    }
    row["row_payload_sha256"] = _sha(row)
    rows.append(row)
  expected_queries = {str(row["query_id"]) for row in public["queries"]}
  if (
      set(role_rankings_by_query) & failed
      or set(role_rankings_by_query) | failed != expected_queries
  ):
    raise ValueError("LinkCAD external ranking query domain differs")
  payload = {
      "schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
      "scope": "public_external_role_ranking_with_matched_symbolic_programs",
      "method_id": str(method_id), "model_id": str(model_id),
      "private_targets_opened": False,
      "selection_uses_private_targets": False,
      "checkpoint_sha256": "not_applicable_external_frozen_agent",
      "model_kind": "external_role_ranker",
      "beam_size": beam_size,
      "interface_alternatives_per_edge": interface_alternatives_per_edge,
      "failed_query_count": len(failed),
      "failed_query_ids": sorted(failed),
      "query_count": len(rows), "rows": rows,
  }
  payload["prediction_payload_sha256"] = _sha(payload)
  return payload


def materialize_symbolic_ambiguity_audit_v1(
    *, public: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
) -> dict[str, Any]:
  """Measure how much ambiguity remains after public symbolic feasibility."""

  if public.get("contains_private_targets") is not False:
    raise ValueError("LinkCAD symbolic ambiguity scope differs")
  rows = []
  endpoint_candidate_counts: list[int] = []
  matching_primitive_counts: list[int] = []
  feasible_assignment_counts: list[int] = []
  for public_query in public["queries"]:
    state = _query_symbolic_state(public_query, cache)
    assignments = _feasible_assignments(public_query, state)
    feasible_assignment_counts.append(len(assignments))
    for edge in state["edge_rows"]:
      for side in edge["sides"]:
        endpoint_candidate_counts.append(side["compatible_candidate_count"])
        matching_primitive_counts.extend(
            count for count in side["matching_primitive_counts"] if count > 0
        )
    rows.append({
        "query_id": str(public_query["query_id"]),
        "role_count": len(state["role_ids"]),
        "edge_count": len(state["edge_rows"]),
        "raw_assignment_count": math.prod(
            len(role) for role in state["candidate_rows"]
        ),
        "feasible_assignment_count": len(assignments),
        "unique_feasible_assignment": len(assignments) == 1,
        "edges": state["edge_rows"],
    })
  query_count = len(rows)
  payload = {
      "schema_version": AMBIGUITY_SCHEMA_VERSION,
      "scope": "public_post_contract_ambiguity",
      "selection_uses_private_targets_or_model_outputs": False,
      "query_count": query_count,
      "edge_count": sum(row["edge_count"] for row in rows),
      "endpoint_count": len(endpoint_candidate_counts),
      "zero_feasible_query_count": sum(value == 0 for value in feasible_assignment_counts),
      "unique_feasible_query_count": sum(value == 1 for value in feasible_assignment_counts),
      "ambiguous_feasible_query_count": sum(value > 1 for value in feasible_assignment_counts),
      "unique_feasible_query_rate": (
          0.0 if query_count == 0
          else sum(value == 1 for value in feasible_assignment_counts) / query_count
      ),
      "compatible_candidates_per_endpoint": {
          "mean": mean(endpoint_candidate_counts) if endpoint_candidate_counts else 0.0,
          "median": median(endpoint_candidate_counts) if endpoint_candidate_counts else 0.0,
          "p90": _quantile(endpoint_candidate_counts, 0.90),
          "maximum": max(endpoint_candidate_counts, default=0),
      },
      "matching_primitives_per_compatible_candidate_endpoint": {
          "mean": mean(matching_primitive_counts) if matching_primitive_counts else 0.0,
          "median": median(matching_primitive_counts) if matching_primitive_counts else 0.0,
          "p90": _quantile(matching_primitive_counts, 0.90),
          "maximum": max(matching_primitive_counts, default=0),
          "unique_rate": (
              0.0 if not matching_primitive_counts
              else sum(value == 1 for value in matching_primitive_counts)
              / len(matching_primitive_counts)
          ),
      },
      "feasible_assignments_per_query": {
          "mean": mean(feasible_assignment_counts) if feasible_assignment_counts else 0.0,
          "median": median(feasible_assignment_counts) if feasible_assignment_counts else 0.0,
          "p90": _quantile(feasible_assignment_counts, 0.90),
          "maximum": max(feasible_assignment_counts, default=0),
      },
      "rows": rows,
  }
  payload["audit_payload_sha256"] = _sha(payload)
  return payload


__all__ = [
    "AMBIGUITY_SCHEMA_VERSION", "EXTERNAL_PREDICTION_SCHEMA_VERSION",
    "POLICIES", "PREDICTION_SCHEMA_VERSION",
    "materialize_external_assignment_predictions_v1",
    "materialize_symbolic_ambiguity_audit_v1",
    "materialize_symbolic_predictions_v1",
]
