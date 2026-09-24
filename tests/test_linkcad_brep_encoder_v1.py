from __future__ import annotations

import torch

from neurocad.linkcad_brep_encoder_v1 import (
    LinkCADBRepEncoderV1,
    graph_payload_to_tensors,
    normalize_node_features,
)


def _graph():
  node = {
      "surface_type": "plane",
      "features": [0.5, 1.0, 0.0, 0.0, 1.0, 4.0],
      "feature_mask": [True, True, False, False, True, True],
  }
  return {
      "nodes": [node, dict(node)],
      "edges": [{"source": 0, "target": 1, "features": [0.2, 1.0, 0.0]}],
  }


def test_graph_tensorization_finds_structural_orbit_and_encodes():
  graph = graph_payload_to_tensors(_graph())
  assert graph.node_features.shape == (2, 19)
  assert graph.edge_features.shape == (1, 3)
  assert graph.orbit_members == ((0, 1),)
  encoder = LinkCADBRepEncoderV1(hidden_dim=16, layers=2)
  attention = encoder(graph, orbit_mode="attention")
  canonical = encoder(graph, orbit_mode="canonical")
  assert attention.shape == canonical.shape == (16,)
  assert torch.isfinite(attention).all()
  batch = encoder.forward_many((graph, graph), orbit_mode="attention")
  assert batch.shape == (2, 16)
  assert torch.allclose(batch[0], batch[1])
  parts, orbits = encoder.forward_many_with_orbits(
      (graph, graph), orbit_mode="attention"
  )
  assert parts.shape == (2, 16)
  assert len(orbits) == 2 and orbits[0].shape == (1, 16)


def test_node_normalization_bounds_float32_curvature_long_tails():
  features = torch.zeros((1, 19), dtype=torch.float32)
  features[0, 2] = torch.inf
  features[0, 3] = 1e30
  normalized = normalize_node_features(features)
  assert torch.isfinite(normalized).all()
  assert float(normalized.abs().max()) <= 70.0
