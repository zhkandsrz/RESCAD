"""Part-contextual face-and-edge primitive LinkCAD model."""

from __future__ import annotations

import torch

from .linkcad_primitive_factorized_model_v2 import (
    PrimitiveFactorizedLinkCADV2,
)


class PrimitiveFactorizedLinkCADV3(PrimitiveFactorizedLinkCADV2):
  """Fuse local primitive geometry with its whole-part B-Rep embedding."""

  def _encode_graphs(self, available_graphs):
    part_embeddings, primitive_sets = super()._encode_graphs(available_graphs)
    contextual_sets = []
    for part, primitives in zip(
        part_embeddings, primitive_sets, strict=True
    ):
      context = part[None, :].expand_as(primitives)
      update = self.part_primitive_fusion(torch.cat((
          primitives,
          context,
          primitives * context,
      ), dim=-1))
      contextual_sets.append(self.primitive_norm(primitives + update))
    return part_embeddings, tuple(contextual_sets)


__all__ = ["PrimitiveFactorizedLinkCADV3"]
