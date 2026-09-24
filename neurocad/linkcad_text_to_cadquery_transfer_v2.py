"""Corpus-native normalized Text-to-CadQuery confirmation schedule."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

from .linkcad_step_llm_transfer_v1 import canonical_sha256
from .linkcad_text_to_cadquery_generator_v1 import (
    MODEL_REPO,
    MODEL_REVISION,
    OFFICIAL_REPOSITORY_REVISION,
)
from .linkcad_text_to_cadquery_transfer_v1 import (
    CANDIDATE_DIAMETERS_MM,
    ROLE_SPECS,
)


SCHEMA_VERSION = "linkcad_text_to_cadquery_transfer_schedule.v2"


@dataclass(frozen=True, slots=True)
class TextToCadQueryTransferConfigV2:
  seed: int = 2026082329
  query_count: int = 30
  candidates_per_role: int = 4
  target_diameter_mm: float = 16.0
  query_variant_offset: int = 9
  model_context_tokens: int = 1024

  def validate(self) -> None:
    if (
        self.seed < 0 or self.query_count < 1
        or self.candidates_per_role != len(CANDIDATE_DIAMETERS_MM)
        or self.target_diameter_mm != 16.0
        or self.query_variant_offset != 9
        or self.model_context_tokens != 1024
    ):
      raise ValueError("Text-to-CadQuery transfer V2 configuration differs")

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": self.seed,
        "query_count": self.query_count,
        "candidates_per_role": self.candidates_per_role,
        "target_diameter_mm": self.target_diameter_mm,
        "query_variant_offset": self.query_variant_offset,
        "model_context_tokens": self.model_context_tokens,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "do_sample": False,
        "terminator": "<|endoftext|>",
        "generator_output_scale_semantics": "normalized_candidate_local_unit",
        "query_oracle_requires_prompt_fidelity": False,
        "query_oracle_uses_interface_scale_calibration": True,
        "interface_scale_factor_range": [0.1, 100.0],
    }


def _instruction(*, family: str, variant: int) -> str:
  if family == "ring":
    outer = 1.62 + 0.001 * variant
    height = 0.30 + 0.0005 * variant
    styles = (
        "On the XY plane, draw two concentric circular profiles.",
        "Create an XY sketch from two concentric circles.",
        "Start with two concentric circles centered at the XY origin.",
    )
    return (
        "Use normalized dimensions. "
        f"{styles[variant % len(styles)]} The outer diameter is {outer:.3f} "
        "units and the inner-to-outer diameter ratio is the reciprocal of "
        f"{outer:.3f}, so the inner interface is one normalized unit. "
        f"Extrude the annular region by {height:.4f} units to create exactly "
        "one hollow cylindrical solid."
    )
  if family == "shaft":
    height = 1.50 + 0.005 * variant
    styles = (
        "Start from a coordinate system aligned with the default axes.",
        "Create a new coordinate system aligned with the default axes.",
        "Use the default axes as the local coordinate system.",
    )
    return (
        "Use normalized dimensions. "
        f"{styles[variant % len(styles)]} Draw one circular shape and extrude "
        "it along the normal to create exactly one solid cylinder. The final "
        f"dimensions are 1.0 units in length and width and {height:.3f} "
        "units in height."
    )
  raise ValueError("Text-to-CadQuery V2 family differs")


def build_text_to_cadquery_transfer_schedule_v2(
    config: TextToCadQueryTransferConfigV2,
) -> tuple[dict[str, Any], ...]:
  config.validate()
  queries = []
  for query_ordinal in range(config.query_count):
    query_variant = query_ordinal + config.query_variant_offset
    query_seed = int(hashlib.sha256(
        f"{config.seed}:query:{query_ordinal}:variant:{query_variant}".encode()
    ).hexdigest()[:16], 16)
    roles = []
    for role_ordinal, (role_id, description, family) in enumerate(ROLE_SPECS):
      diameters = list(CANDIDATE_DIAMETERS_MM)
      random.Random(query_seed ^ (role_ordinal << 18)).shuffle(diameters)
      candidates = []
      for candidate_ordinal, diameter in enumerate(diameters):
        variant = query_variant * 8 + role_ordinal * 4 + candidate_ordinal
        instruction = _instruction(family=family, variant=variant)
        digest = canonical_sha256({
            "query_seed": query_seed,
            "role_id": role_id,
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": diameter,
            "instruction": instruction,
        })
        candidates.append({
            "candidate_key": "t2cq2_" + digest[:24],
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": diameter,
            "instruction": instruction,
            "is_target": diameter == config.target_diameter_mm,
        })
      roles.append({
          "role_id": role_id,
          "role_ordinal": role_ordinal,
          "description": description,
          "family": family,
          "target_requested_diameter_mm": config.target_diameter_mm,
          "candidates": candidates,
      })
    query_payload = {
        "query_ordinal": query_ordinal,
        "query_variant": query_variant,
        "query_seed": query_seed,
        "roles": roles,
    }
    queries.append({
        "query_id": "linkcad_t2cq2_" + canonical_sha256(query_payload)[:20],
        **query_payload,
    })
  return tuple(queries)


__all__ = [
    "SCHEMA_VERSION", "TextToCadQueryTransferConfigV2",
    "build_text_to_cadquery_transfer_schedule_v2",
]
