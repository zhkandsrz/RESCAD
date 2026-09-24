"""Smoke-informed frozen schedule for STEP-LLM geometric-oracle transfer."""

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


SCHEMA_VERSION = "linkcad_step_llm_transfer_schedule.v2"
ROLE_SPECS = (
    ("role_00", "annular connector with a cylindrical bore", "ring", 8.0),
    ("role_01", "axial shaft inserted into the connector", "shaft", 45.0),
)


@dataclass(frozen=True, slots=True)
class StepLLMTransferConfigV2:
  seed: int = 2026082461
  query_count: int = 30
  candidates_per_role: int = 4
  max_new_tokens: int = 6000

  def validate(self) -> None:
    if (
        self.seed < 0 or self.query_count < 1
        or self.candidates_per_role != len(CANDIDATE_DIAMETERS_MM)
        or self.max_new_tokens != 6000
    ):
      raise ValueError("STEP-LLM transfer v2 configuration differs")

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": self.seed,
        "query_count": self.query_count,
        "candidates_per_role": self.candidates_per_role,
        "target_requested_diameter_by_family_mm": {"ring": 8.0, "shaft": 45.0},
        "max_new_tokens": self.max_new_tokens,
        "adapter_repo": STEP_LLM_ADAPTER_REPO,
        "adapter_revision": STEP_LLM_ADAPTER_REVISION,
        "base_repo": STEP_LLM_BASE_REPO,
        "base_revision": STEP_LLM_BASE_REVISION,
        "do_sample": False,
        "query_oracle_requires_prompt_fidelity": False,
    }


def _caption(*, family: str, diameter: float, query_ordinal: int) -> str:
  if family == "ring":
    outer = diameter + 12.0 + 0.5 * query_ordinal
    thickness = 6.0 + 0.1 * query_ordinal
    return (
        f"A thick annular ring {outer:g} mm in outer diameter and {diameter:g} mm "
        f"in inner diameter, extruded to a thickness of {thickness:g} mm."
    )
  if family == "shaft":
    length = 30.0 + 0.5 * query_ordinal
    return (
        f"A simple cylindrical shaft {length:g} mm long and {diameter:g} mm in diameter."
    )
  raise ValueError("STEP-LLM transfer v2 family differs")


def build_step_llm_transfer_schedule_v2(
    config: StepLLMTransferConfigV2,
) -> tuple[dict[str, Any], ...]:
  config.validate()
  queries = []
  for query_ordinal in range(config.query_count):
    query_seed = int(hashlib.sha256(
        f"{config.seed}:query:{query_ordinal}".encode("utf-8")
    ).hexdigest()[:16], 16)
    roles = []
    for role_ordinal, (role_id, description, family, target_diameter) in enumerate(
        ROLE_SPECS
    ):
      diameters = list(CANDIDATE_DIAMETERS_MM)
      random.Random(query_seed ^ (role_ordinal << 16)).shuffle(diameters)
      candidates = []
      for candidate_ordinal, diameter in enumerate(diameters):
        caption = _caption(
            family=family, diameter=diameter, query_ordinal=query_ordinal,
        )
        key_payload = {
            "query_seed": query_seed,
            "role_id": role_id,
            "candidate_ordinal": candidate_ordinal,
            "caption": caption,
        }
        digest = canonical_sha256(key_payload)
        candidates.append({
            "candidate_key": "stepllm2_" + digest[:24],
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": diameter,
            "caption": caption,
            "generation_seed": int(hashlib.sha256(
                digest.encode("ascii")
            ).hexdigest()[:8], 16),
            "is_target": diameter == target_diameter,
        })
      roles.append({
          "role_id": role_id,
          "role_ordinal": role_ordinal,
          "description": description,
          "family": family,
          "target_requested_diameter_mm": target_diameter,
          "candidates": candidates,
      })
    query_payload = {
        "query_ordinal": query_ordinal,
        "query_seed": query_seed,
        "roles": roles,
    }
    queries.append({
        "query_id": "linkcad_stepllm2_" + canonical_sha256(query_payload)[:20],
        **query_payload,
    })
  return tuple(queries)


__all__ = [
    "ROLE_SPECS", "SCHEMA_VERSION", "StepLLMTransferConfigV2",
    "build_step_llm_transfer_schedule_v2",
]
