"""Target-ID-free language contracts for intended B-Rep interface ports."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, Mapping

from .linkcad_primitive_graph_v2 import (
    PRIMITIVE_TYPES,
    LinkCADPrimitiveGraphV2,
)
from .linkcad_brep_encoder_v1 import SURFACE_TYPES


SCHEMA_VERSION = "linkcad_port_contract.v1"


def _measure(graph: LinkCADPrimitiveGraphV2, ordinal: int) -> float:
  primitive_type = graph.primitive_type(ordinal)
  radius = float(graph.primitive_features[ordinal, 1])
  return radius if primitive_type in {
      "circle", "cylinder", "cone", "sphere", "torus"
  } and radius > 0.0 else float(graph.primitive_features[ordinal, 0])


def _size_class(graph: LinkCADPrimitiveGraphV2, ordinal: int) -> str:
  primitive_type = graph.primitive_type(ordinal)
  peers = [
      index for index in range(int(graph.primitive_features.shape[0]))
      if graph.primitive_type(index) == primitive_type
  ]
  if len(peers) == 1:
    return "only"
  target = _measure(graph, ordinal)
  lower = sum(_measure(graph, index) < target - 1e-7 for index in peers)
  upper = sum(_measure(graph, index) > target + 1e-7 for index in peers)
  if lower == 0 and upper > 0:
    return "smallest"
  if upper == 0 and lower > 0:
    return "largest"
  quantile = (lower + 0.5 * (len(peers) - lower - upper)) / len(peers)
  if quantile < 1.0 / 3.0:
    return "small"
  if quantile > 2.0 / 3.0:
    return "large"
  return "medium"


def describe_port_v1(
    graph: LinkCADPrimitiveGraphV2, primitive_ordinal: int,
) -> dict[str, Any]:
  if not 0 <= primitive_ordinal < int(graph.primitive_features.shape[0]):
    raise ValueError("LinkCAD port primitive ordinal differs")
  primitive_type = graph.primitive_type(primitive_ordinal)
  kind = graph.primitive_kinds[primitive_ordinal]
  histogram_start = 8 + len(PRIMITIVE_TYPES)
  histogram = graph.primitive_features[
      primitive_ordinal,
      histogram_start:histogram_start + len(SURFACE_TYPES),
  ]
  adjacency = [
      surface.lower().replace("surfacetype", " surface")
      for index, surface in enumerate(SURFACE_TYPES)
      if float(histogram[index]) >= 0.24
  ][:2]
  member_count = len(graph.primitive_members[primitive_ordinal])
  return {
      "schema_version": SCHEMA_VERSION,
      "primitive_kind": kind,
      "geometry_type": primitive_type,
      "size_class": _size_class(graph, primitive_ordinal),
      "adjacent_surface_types": adjacency,
      "symmetry_class": "repeated" if member_count > 1 else "single",
  }


def describe_ports_v1(
    graph: LinkCADPrimitiveGraphV2,
) -> tuple[dict[str, Any], ...]:
  """Describe every primitive with one grouped size-ranking pass."""

  count = int(graph.primitive_features.shape[0])
  measures = [_measure(graph, ordinal) for ordinal in range(count)]
  types = [graph.primitive_type(ordinal) for ordinal in range(count)]
  grouped: dict[str, list[float]] = {}
  for primitive_type, measure in zip(types, measures, strict=True):
    grouped.setdefault(primitive_type, []).append(measure)
  sorted_measures = {
      primitive_type: sorted(values)
      for primitive_type, values in grouped.items()
  }
  histogram_start = 8 + len(PRIMITIVE_TYPES)
  contracts = []
  for ordinal, (primitive_type, target) in enumerate(
      zip(types, measures, strict=True)
  ):
    peers = sorted_measures[primitive_type]
    if len(peers) == 1:
      size_class = "only"
    else:
      lower = bisect_left(peers, target - 1e-7)
      upper = len(peers) - bisect_right(peers, target + 1e-7)
      if lower == 0 and upper > 0:
        size_class = "smallest"
      elif upper == 0 and lower > 0:
        size_class = "largest"
      else:
        quantile = (lower + 0.5 * (len(peers) - lower - upper)) / len(peers)
        if quantile < 1.0 / 3.0:
          size_class = "small"
        elif quantile > 2.0 / 3.0:
          size_class = "large"
        else:
          size_class = "medium"
    histogram = graph.primitive_features[
        ordinal,
        histogram_start:histogram_start + len(SURFACE_TYPES),
    ]
    adjacency = [
        surface.lower().replace("surfacetype", " surface")
        for index, surface in enumerate(SURFACE_TYPES)
        if float(histogram[index]) >= 0.24
    ][:2]
    contracts.append({
        "schema_version": SCHEMA_VERSION,
        "primitive_kind": graph.primitive_kinds[ordinal],
        "geometry_type": primitive_type,
        "size_class": size_class,
        "adjacent_surface_types": adjacency,
        "symmetry_class": (
            "repeated" if len(graph.primitive_members[ordinal]) > 1 else "single"
        ),
    })
  return tuple(contracts)


def port_phrase_v1(contract: Mapping[str, Any]) -> str:
  if contract.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("LinkCAD port contract schema differs")
  kind = str(contract["primitive_kind"])
  geometry = str(contract["geometry_type"])
  size = str(contract["size_class"])
  noun = f"{geometry} {kind}"
  if size != "only":
    noun = f"{size} {noun}"
  adjacency = tuple(str(value) for value in contract["adjacent_surface_types"])
  if adjacency:
    noun += " adjacent to " + " and ".join(adjacency)
  if contract["symmetry_class"] == "repeated":
    noun += " in a repeated symmetric set"
  return noun


def render_port_conditioned_instruction_v1(
    *, role_a: str, role_b: str,
    port_a: Mapping[str, Any], port_b: Mapping[str, Any],
    mobility_instruction: str,
) -> str:
  return (
      f"Use the {port_phrase_v1(port_a)} on {role_a} and the "
      f"{port_phrase_v1(port_b)} on {role_b}. {mobility_instruction.strip()}"
  )


__all__ = [
    "SCHEMA_VERSION", "describe_port_v1", "describe_ports_v1", "port_phrase_v1",
    "render_port_conditioned_instruction_v1",
]
