"""Controlled missing-field interventions for LinkCAD port contracts.

The original five-field contract remains unchanged.  This module projects it
to a partial contract whose omitted fields are wildcards, allowing a frozen
model to be evaluated as interface language becomes less informative.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "linkcad_partial_port_contract.v1"
CONTRACT_FIELDS = (
    "primitive_kind",
    "geometry_type",
    "size_class",
    "adjacent_surface_types",
    "symmetry_class",
)
LEVEL_FIELDS = {
    "fields_5": CONTRACT_FIELDS,
    "fields_4": CONTRACT_FIELDS[:-1],
    "fields_3": CONTRACT_FIELDS[:-2],
    "fields_2": CONTRACT_FIELDS[:2],
    "fields_1": CONTRACT_FIELDS[:1],
}


def project_port_contract_v1(
    contract: Mapping[str, Any], *, active_fields: Sequence[str],
) -> dict[str, Any]:
  """Return a target-ID-free partial contract; omitted fields are wildcards."""

  active = tuple(str(value) for value in active_fields)
  if not active or any(value not in CONTRACT_FIELDS for value in active):
    raise ValueError("LinkCAD partial contract field domain differs")
  if tuple(value for value in CONTRACT_FIELDS if value in active) != active:
    raise ValueError("LinkCAD partial contract field order differs")
  missing = [value for value in active if value not in contract]
  if missing:
    raise ValueError("LinkCAD source contract fields differ")
  return {
      "schema_version": SCHEMA_VERSION,
      "active_fields": list(active),
      **{field: deepcopy(contract[field]) for field in active},
  }


def port_contract_matches_v1(
    candidate_contract: Mapping[str, Any],
    requested_contract: Mapping[str, Any],
) -> bool:
  """Match a complete intrinsic port description to exact or partial input."""

  if requested_contract.get("schema_version") != SCHEMA_VERSION:
    return dict(candidate_contract) == dict(requested_contract)
  active = tuple(requested_contract.get("active_fields", ()))
  if not active or any(field not in CONTRACT_FIELDS for field in active):
    raise ValueError("LinkCAD partial contract active fields differ")
  return all(
      candidate_contract.get(field) == requested_contract.get(field)
      for field in active
  )


def partial_port_phrase_v1(contract: Mapping[str, Any]) -> str:
  if contract.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("LinkCAD partial port contract schema differs")
  active = set(contract["active_fields"])
  kind = str(contract["primitive_kind"])
  noun = (
      f"{contract['geometry_type']} {kind}"
      if "geometry_type" in active else kind
  )
  if "size_class" in active and contract["size_class"] != "only":
    noun = f"{contract['size_class']} {noun}"
  if "adjacent_surface_types" in active:
    adjacency = tuple(str(value) for value in contract["adjacent_surface_types"])
    if adjacency:
      noun += " adjacent to " + " and ".join(adjacency)
  if "symmetry_class" in active and contract["symmetry_class"] == "repeated":
    noun += " in a repeated symmetric set"
  return noun


def project_public_contract_strength_v1(
    public: Mapping[str, Any], *, level: str,
) -> dict[str, Any]:
  """Project all endpoint contracts and their language to one fixed level."""

  if public.get("contains_private_targets") is not False:
    raise ValueError("LinkCAD contract-strength public scope differs")
  if level not in LEVEL_FIELDS:
    raise ValueError("LinkCAD contract-strength level differs")
  projected = deepcopy(public)
  projected["contract_strength_intervention"] = {
      "schema_version": "linkcad_contract_strength_intervention.v1",
      "level": level,
      "active_fields": list(LEVEL_FIELDS[level]),
      "omitted_fields_are_wildcards": True,
  }
  if level == "fields_5":
    return projected
  active = LEVEL_FIELDS[level]
  for query in projected["queries"]:
    for edge in query["functional_edges"]:
      if edge.get("port_contract_status") != "specified":
        continue
      port_a = project_port_contract_v1(
          edge["port_contract_a"], active_fields=active,
      )
      port_b = project_port_contract_v1(
          edge["port_contract_b"], active_fields=active,
      )
      instruction = str(edge["instruction"])
      mobility_sentence = (
          instruction.split(". ", 1)[1]
          if ". " in instruction else instruction
      )
      edge["port_contract_a"] = port_a
      edge["port_contract_b"] = port_b
      edge["instruction"] = (
          f"Use the {partial_port_phrase_v1(port_a)} on {edge['role_a']} and "
          f"the {partial_port_phrase_v1(port_b)} on {edge['role_b']}. "
          f"{mobility_sentence}"
      )
  return projected


__all__ = [
    "CONTRACT_FIELDS", "LEVEL_FIELDS", "SCHEMA_VERSION",
    "partial_port_phrase_v1", "port_contract_matches_v1",
    "project_port_contract_v1", "project_public_contract_strength_v1",
]
