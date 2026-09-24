"""Frozen scale-calibrated STEP-LLM candidate transfer schedule."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

from .linkcad_step_llm_generator_v1 import (
    STEP_LLM_ADAPTER_REPO,
    STEP_LLM_ADAPTER_REVISION,
    STEP_LLM_BASE_REPO,
    STEP_LLM_BASE_REVISION,
)
from .linkcad_step_llm_transfer_v1 import (
    CANDIDATE_DIAMETERS_MM,
    canonical_sha256,
)


SCHEMA_VERSION = "linkcad_step_llm_transfer_schedule.v3"
ROLE_SPECS = (
    ("role_00", "annular connector with a 16 mm bore", "ring"),
    ("role_01", "axial shaft 16 mm in diameter", "shaft"),
)


@dataclass(frozen=True, slots=True)
class StepLLMTransferConfigV3:
  seed: int = 2026082467
  query_count: int = 30
  candidates_per_role: int = 4
  target_diameter_mm: float = 16.0
  query_variant_offset: int = 1
  max_new_tokens: int = 6000

  def validate(self) -> None:
    if (
        self.seed < 0 or self.query_count < 1
        or self.candidates_per_role != len(CANDIDATE_DIAMETERS_MM)
        or self.target_diameter_mm != 16.0
        or self.query_variant_offset != 1
        or self.max_new_tokens != 6000
    ):
      raise ValueError("STEP-LLM transfer v3 configuration differs")

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": self.seed,
        "query_count": self.query_count,
        "candidates_per_role": self.candidates_per_role,
        "target_diameter_mm": self.target_diameter_mm,
        "query_variant_offset": self.query_variant_offset,
        "max_new_tokens": self.max_new_tokens,
        "adapter_repo": STEP_LLM_ADAPTER_REPO,
        "adapter_revision": STEP_LLM_ADAPTER_REVISION,
        "base_repo": STEP_LLM_BASE_REPO,
        "base_revision": STEP_LLM_BASE_REVISION,
        "do_sample": False,
        "query_oracle_requires_prompt_fidelity": False,
        "query_oracle_uses_interface_scale_calibration": True,
        "interface_scale_factor_range": [0.05, 20.0],
    }


def _caption(*, family: str, diameter: float, variant: int) -> str:
  if family == "ring":
    outer = diameter + 12.0 + 0.5 * variant
    thickness = 6.0 + 0.1 * variant
    return (
        f"A thick annular ring {outer:g} mm in outer diameter and {diameter:g} mm "
        f"in inner diameter, extruded to a thickness of {thickness:g} mm."
    )
  if family == "shaft":
    length = 30.0 + 0.5 * variant
    return (
        f"A simple cylindrical shaft {length:g} mm long and {diameter:g} mm in diameter."
    )
  raise ValueError("STEP-LLM transfer v3 family differs")


def build_step_llm_transfer_schedule_v3(
    config: StepLLMTransferConfigV3,
) -> tuple[dict[str, Any], ...]:
  config.validate()
  queries = []
  for query_ordinal in range(config.query_count):
    variant = query_ordinal + config.query_variant_offset
    query_seed = int(hashlib.sha256(
        f"{config.seed}:query:{query_ordinal}:variant:{variant}".encode("utf-8")
    ).hexdigest()[:16], 16)
    roles = []
    for role_ordinal, (role_id, description, family) in enumerate(ROLE_SPECS):
      diameters = list(CANDIDATE_DIAMETERS_MM)
      random.Random(query_seed ^ (role_ordinal << 16)).shuffle(diameters)
      candidates = []
      for candidate_ordinal, diameter in enumerate(diameters):
        caption = _caption(family=family, diameter=diameter, variant=variant)
        key_payload = {
            "query_seed": query_seed,
            "role_id": role_id,
            "candidate_ordinal": candidate_ordinal,
            "caption": caption,
        }
        digest = canonical_sha256(key_payload)
        candidates.append({
            "candidate_key": "stepllm3_" + digest[:24],
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": diameter,
            "caption": caption,
            "generation_seed": int(hashlib.sha256(
                digest.encode("ascii")
            ).hexdigest()[:8], 16),
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
        "query_variant": variant,
        "query_seed": query_seed,
        "roles": roles,
    }
    queries.append({
        "query_id": "linkcad_stepllm3_" + canonical_sha256(query_payload)[:20],
        **query_payload,
    })
  return tuple(queries)


__all__ = [
    "ROLE_SPECS", "SCHEMA_VERSION", "StepLLMTransferConfigV3",
    "build_step_llm_transfer_schedule_v3",
]
