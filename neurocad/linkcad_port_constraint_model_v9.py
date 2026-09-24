"""Neuro-symbolic port-feasibility constraints for LinkCAD assignment."""

from __future__ import annotations

from dataclasses import replace

import torch

from .linkcad_port_conditioned_model_v6 import PortConditionedLinkCADV6


class PortConstraintLinkCADV9(PortConditionedLinkCADV6):
  """Mask candidate pairs that cannot realize the public port contract."""

  def forward(self, query):
    output = super().forward(query)
    if query.port_candidate_mask is None:
      return output
    expected = (int(query.edge_index.shape[0]), 2, query.candidate_count)
    if tuple(query.port_candidate_mask.shape) != expected:
      raise ValueError("LinkCAD port candidate mask dimensions differ")
    pair_mask = (
        query.port_candidate_mask[:, 0, :, None]
        & query.port_candidate_mask[:, 1, None, :]
    )
    constrained = output.edge_pair_logits.masked_fill(~pair_mask, -torch.inf)
    return replace(output, edge_pair_logits=constrained)


__all__ = ["PortConstraintLinkCADV9"]
