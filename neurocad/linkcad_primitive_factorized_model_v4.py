"""LinkCAD primitive linker paired with analytic-radius feature cache V3."""

from __future__ import annotations

from .linkcad_primitive_factorized_model_v2 import PrimitiveFactorizedLinkCADV2


class PrimitiveFactorizedLinkCADV4(PrimitiveFactorizedLinkCADV2):
  """Retain V2 factorization while consuming V3 analytic primitive features."""


__all__ = ["PrimitiveFactorizedLinkCADV4"]
