"""Frozen candidate schedule for learned Text-to-CadQuery transfer."""

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


SCHEMA_VERSION = "linkcad_text_to_cadquery_transfer_schedule.v1"
CANDIDATE_DIAMETERS_MM = (8.0, 16.0, 30.0, 45.0)
ROLE_SPECS = (
    ("role_00", "annular connector with a 16 mm bore", "ring"),
    ("role_01", "axial shaft 16 mm in diameter", "shaft"),
)


@dataclass(frozen=True, slots=True)
class TextToCadQueryTransferConfigV1:
  seed: int = 2026082317
  query_count: int = 30
  candidates_per_role: int = 4
  target_diameter_mm: float = 16.0
  query_variant_offset: int = 7
  model_context_tokens: int = 1024

  def validate(self) -> None:
    if (
        self.seed < 0 or self.query_count < 1
        or self.candidates_per_role != len(CANDIDATE_DIAMETERS_MM)
        or self.target_diameter_mm != 16.0
        or self.query_variant_offset != 7
        or self.model_context_tokens != 1024
    ):
      raise ValueError("Text-to-CadQuery transfer configuration differs")

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
        "query_oracle_requires_prompt_fidelity": False,
        "query_oracle_uses_interface_scale_calibration": True,
        "interface_scale_factor_range": [1e-6, 1e6],
    }


def _instruction(*, family: str, diameter: float, variant: int) -> str:
  if family == "ring":
    outer = diameter + 10.0 + 0.25 * variant
    height = 5.0 + 0.05 * variant
    styles = (
        "Start with a sketch of two concentric circles on the XY plane.",
        "On the XY plane, draw two concentric circular profiles.",
        "Create one annular XY sketch from a pair of concentric circles.",
    )
    return (
        f"{styles[variant % len(styles)]} The outer diameter is {outer:g} "
        f"millimeters and the inner diameter is {diameter:g} millimeters. "
        f"Extrude the annular region by {height:g} millimeters to create "
        "exactly one hollow cylindrical solid."
    )
  if family == "shaft":
    length = 24.0 + 0.5 * variant
    styles = (
        "Draw one circular sketch on the XY plane.",
        "On the XY plane, create a single circular profile.",
        "Begin from one circle centered at the XY origin.",
    )
    return (
        f"{styles[variant % len(styles)]} Its diameter is {diameter:g} "
        f"millimeters. Extrude it by {length:g} millimeters to create "
        "exactly one solid cylindrical shaft."
    )
  raise ValueError("Text-to-CadQuery transfer family differs")


def build_text_to_cadquery_transfer_schedule_v1(
    config: TextToCadQueryTransferConfigV1,
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
      random.Random(query_seed ^ (role_ordinal << 17)).shuffle(diameters)
      candidates = []
      for candidate_ordinal, diameter in enumerate(diameters):
        candidate_variant = query_variant * 8 + role_ordinal * 4 + candidate_ordinal
        instruction = _instruction(
            family=family, diameter=diameter, variant=candidate_variant,
        )
        key_payload = {
            "query_seed": query_seed,
            "role_id": role_id,
            "candidate_ordinal": candidate_ordinal,
            "instruction": instruction,
        }
        digest = canonical_sha256(key_payload)
        candidates.append({
            "candidate_key": "t2cq_" + digest[:24],
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
        "query_id": "linkcad_t2cq_" + canonical_sha256(query_payload)[:20],
        **query_payload,
    })
  return tuple(queries)


__all__ = [
    "CANDIDATE_DIAMETERS_MM", "ROLE_SPECS", "SCHEMA_VERSION",
    "TextToCadQueryTransferConfigV1",
    "build_text_to_cadquery_transfer_schedule_v1",
]
