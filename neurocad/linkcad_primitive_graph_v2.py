"""Face-and-edge interface primitive graph for LinkCAD V2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from .benchmark_v2_model_view_v2 import (
    GraphSizeBudget,
    _extract_intrinsic_brep_graph_with_identity,
)
from .linkcad_brep_encoder_v1 import (
    LinkCADPartGraph,
    SURFACE_TYPES,
    graph_payload_to_tensors,
)


PRIMITIVE_TYPES = (
    "plane", "cylinder", "cone", "sphere", "torus", "spline", "other",
    "circle", "line", "ellipse", "bspline_curve", "other_curve",
)
PRIMITIVE_FEATURE_DIM = 8 + len(PRIMITIVE_TYPES) + len(SURFACE_TYPES) + 2
_GRAPH_BUDGET = GraphSizeBudget(
    max_graphs=1, max_faces_per_graph=1024, max_edges_per_graph=4096,
    max_examples=1, max_program_candidates_per_example=1,
)


@dataclass(frozen=True, slots=True)
class LinkCADPrimitiveGraphV2:
  face_graph: LinkCADPartGraph
  primitive_features: Tensor
  primitive_kinds: tuple[str, ...]
  primitive_members: tuple[tuple[int, ...], ...]
  face_orbit_count: int

  def to(self, device: str | torch.device) -> "LinkCADPrimitiveGraphV2":
    return LinkCADPrimitiveGraphV2(
        face_graph=self.face_graph.to(device),
        primitive_features=self.primitive_features.to(device),
        primitive_kinds=self.primitive_kinds,
        primitive_members=self.primitive_members,
        face_orbit_count=self.face_orbit_count,
    )

  def primitive_type(self, ordinal: int) -> str:
    if not 0 <= ordinal < int(self.primitive_features.shape[0]):
      raise ValueError("LinkCAD primitive type ordinal differs")
    start = 8
    values = self.primitive_features[ordinal, start:start + len(PRIMITIVE_TYPES)]
    return PRIMITIVE_TYPES[int(values.argmax())]


def _same(left: Any, right: Any) -> bool:
  for left_shape, right_shape in (
      (getattr(left, "wrapped", left), getattr(right, "wrapped", right)),
      (left, right),
  ):
    for name in ("IsSame", "IsEqual", "isSame", "isEqual"):
      method = getattr(left_shape, name, None)
      if callable(method):
        try:
          if bool(method(right_shape)):
            return True
        except (TypeError, RuntimeError):
          pass
  return False


def _edge_radius(edge: Any, *, scale: float) -> tuple[float, bool]:
  if str(edge.geomType()).upper() != "CIRCLE":
    return 0.0, False
  try:
    value = float(edge.radius()) / scale
    if math.isfinite(value) and value > 0.0:
      return value, True
  except Exception:
    pass
  return 0.0, False


def _curve_type(edge: Any) -> str:
  return {
      "CIRCLE": "circle",
      "LINE": "line",
      "ELLIPSE": "ellipse",
      "BSPLINE": "bspline_curve",
  }.get(str(edge.geomType()).upper(), "other_curve")


def structural_edge_orbits_v2(shape: Any) -> tuple[tuple[int, ...], ...]:
  """Deterministic intrinsic edge partition used by labels and model view."""

  edges = list(shape.Edges())
  faces = list(shape.Faces())
  if not edges or not faces:
    raise ValueError("LinkCAD primitive STEP topology is empty")
  scale = math.sqrt(sum(float(face.Area()) for face in faces))
  face_edges = [list(face.Edges()) for face in faces]
  signatures = []
  for edge in edges:
    owners = [
        str(face.geomType()).upper()
        for face, boundary in zip(faces, face_edges, strict=True)
        if any(_same(edge, candidate) for candidate in boundary)
    ]
    radius, radius_known = _edge_radius(edge, scale=scale)
    vertices = list(edge.Vertices())
    chord = 0.0
    if len(vertices) >= 2:
      a = vertices[0].Center().toTuple()
      b = vertices[-1].Center().toTuple()
      chord = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    signatures.append((
        str(edge.geomType()).upper(),
        round(float(edge.Length()) / scale, 7),
        round(radius, 7), radius_known,
        round(chord / scale, 7), len(vertices), tuple(sorted(owners)),
    ))
  groups: dict[tuple[Any, ...], list[int]] = {}
  for raw_index, signature in enumerate(signatures):
    groups.setdefault(signature, []).append(raw_index)
  return tuple(
      tuple(indices)
      for _signature, indices in sorted(groups.items(), key=lambda row: row[1][0])
  )


def _surface_histogram(surface_rows: list[str]) -> list[float]:
  count = max(1, len(surface_rows))
  return [surface_rows.count(surface) / count for surface in SURFACE_TYPES]


def extract_primitive_graph_v2(shape: Any) -> LinkCADPrimitiveGraphV2:
  graph_payload, raw_to_graph, _signatures = (
      _extract_intrinsic_brep_graph_with_identity(shape, budget=_GRAPH_BUDGET)
  )
  face_graph = graph_payload_to_tensors(graph_payload)
  raw_faces = list(shape.Faces())
  raw_edges = list(shape.Edges())
  scale = math.sqrt(sum(float(face.Area()) for face in raw_faces))
  graph_to_raw = {graph_index: raw for raw, graph_index in raw_to_graph.items()}
  graph_surface = [str(row["surface_type"]) for row in graph_payload["nodes"]]
  neighbors = [[] for _ in graph_surface]
  for edge in graph_payload["edges"]:
    left, right = int(edge["source"]), int(edge["target"])
    neighbors[left].append(graph_surface[right])
    neighbors[right].append(graph_surface[left])

  feature_rows = []
  primitive_kinds = []
  primitive_members = []
  for members in face_graph.orbit_members:
    representative = members[0]
    surface = graph_surface[representative]
    measure = sum(
        float(graph_payload["nodes"][index]["features"][0]) for index in members
    ) / len(members)
    adjacent = [value for index in members for value in neighbors[index]]
    continuous = [
        measure, 0.0, 0.0, math.log1p(len(members)),
        min(1.0, len(adjacent) / max(1.0, 4.0 * len(members))),
        0.0, 0.0, 0.0,
    ]
    feature_rows.append([
        *continuous,
        *[float(surface == value) for value in PRIMITIVE_TYPES],
        *_surface_histogram(adjacent),
        1.0, 0.0,
    ])
    primitive_kinds.append("face")
    primitive_members.append(tuple(graph_to_raw[index] for index in members))

  face_edges = [list(face.Edges()) for face in raw_faces]
  edge_orbits = structural_edge_orbits_v2(shape)
  for members in edge_orbits:
    representative_edge = raw_edges[members[0]]
    curve = _curve_type(representative_edge)
    owner_surfaces = []
    owner_counts = []
    for raw_edge_index in members:
      edge = raw_edges[raw_edge_index]
      owners = [
          str(graph_payload["nodes"][raw_to_graph[raw_face]]["surface_type"])
          for raw_face, boundary in enumerate(face_edges)
          if any(_same(edge, candidate) for candidate in boundary)
      ]
      owner_surfaces.extend(owners)
      owner_counts.append(len(owners))
    radius, radius_known = _edge_radius(representative_edge, scale=scale)
    vertices = list(representative_edge.Vertices())
    chord = 0.0
    if len(vertices) >= 2:
      a = vertices[0].Center().toTuple()
      b = vertices[-1].Center().toTuple()
      chord = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    mean_owners = sum(owner_counts) / len(owner_counts)
    continuous = [
        float(representative_edge.Length()) / scale,
        radius,
        chord / scale,
        math.log1p(len(members)),
        min(1.0, mean_owners / 2.0),
        float(chord <= 1e-8 * scale),
        float(mean_owners <= 1.0),
        float(radius_known),
    ]
    feature_rows.append([
        *continuous,
        *[float(curve == value) for value in PRIMITIVE_TYPES],
        *_surface_histogram(owner_surfaces),
        0.0, 1.0,
    ])
    primitive_kinds.append("edge")
    primitive_members.append(tuple(members))
  features = torch.tensor(feature_rows, dtype=torch.float32)
  if features.ndim != 2 or features.shape[1] != PRIMITIVE_FEATURE_DIM:
    raise ValueError("LinkCAD primitive feature dimension differs")
  return LinkCADPrimitiveGraphV2(
      face_graph=face_graph,
      primitive_features=features,
      primitive_kinds=tuple(primitive_kinds),
      primitive_members=tuple(primitive_members),
      face_orbit_count=len(face_graph.orbit_members),
  )


__all__ = [
    "LinkCADPrimitiveGraphV2", "PRIMITIVE_FEATURE_DIM", "PRIMITIVE_TYPES",
    "extract_primitive_graph_v2", "structural_edge_orbits_v2",
]
