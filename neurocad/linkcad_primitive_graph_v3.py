"""Analytic-radius extension of the LinkCAD face-and-edge primitive graph."""

from __future__ import annotations

import math
from typing import Any

from OCP.BRepAdaptor import BRepAdaptor_Surface

from .linkcad_primitive_graph_v2 import (
    LinkCADPrimitiveGraphV2,
    extract_primitive_graph_v2,
)


FEATURE_SCHEMA_VERSION = "linkcad_primitive_features.v3"


def _surface_scales(face: Any, surface: str, scale: float):
  adaptor = BRepAdaptor_Surface(face.wrapped)
  if surface == "cylinder":
    return float(adaptor.Cylinder().Radius()) / scale, 0.0, True
  if surface == "sphere":
    return float(adaptor.Sphere().Radius()) / scale, 0.0, True
  if surface == "cone":
    cone = adaptor.Cone()
    return (
        float(cone.RefRadius()) / scale,
        abs(float(cone.SemiAngle())) / math.pi,
        True,
    )
  if surface == "torus":
    torus = adaptor.Torus()
    return (
        float(torus.MajorRadius()) / scale,
        float(torus.MinorRadius()) / scale,
        True,
    )
  return 0.0, 0.0, False


def extract_primitive_graph_v3(shape: Any) -> LinkCADPrimitiveGraphV2:
  """Add normalized analytic surface radii without changing orbit identity."""

  graph = extract_primitive_graph_v2(shape)
  features = graph.primitive_features.clone()
  faces = list(shape.Faces())
  scale = math.sqrt(sum(float(face.Area()) for face in faces))
  if not math.isfinite(scale) or scale <= 0.0:
    raise ValueError("LinkCAD V3 primitive scale differs")
  for ordinal in range(graph.face_orbit_count):
    surface = graph.primitive_type(ordinal)
    rows = []
    for raw_index in graph.primitive_members[ordinal]:
      try:
        rows.append(_surface_scales(faces[raw_index], surface, scale))
      except (AttributeError, RuntimeError, ValueError):
        continue
    known = [row for row in rows if row[2]]
    if known:
      features[ordinal, 1] = sum(row[0] for row in known) / len(known)
      features[ordinal, 2] = sum(row[1] for row in known) / len(known)
      features[ordinal, 7] = 1.0
  return LinkCADPrimitiveGraphV2(
      face_graph=graph.face_graph,
      primitive_features=features,
      primitive_kinds=graph.primitive_kinds,
      primitive_members=graph.primitive_members,
      face_orbit_count=graph.face_orbit_count,
  )


__all__ = ["FEATURE_SCHEMA_VERSION", "extract_primitive_graph_v3"]
