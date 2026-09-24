"""Frozen schedule for learned STEP-LLM candidate transfer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any

from .linkcad_step_llm_generator_v1 import (
    STEP_LLM_ADAPTER_REPO,
    STEP_LLM_ADAPTER_REVISION,
    STEP_LLM_BASE_REPO,
    STEP_LLM_BASE_REVISION,
)


SCHEMA_VERSION = "linkcad_step_llm_transfer_schedule.v1"
ROLE_SPECS = (
    ("role_00", "annular connector with a 45 mm bore", "ring"),
    ("role_01", "axial shaft 45 mm in diameter", "shaft"),
)
CANDIDATE_DIAMETERS_MM = (8.0, 16.0, 30.0, 45.0)


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
  ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StepLLMTransferConfigV1:
  seed: int = 2026082455
  query_count: int = 30
  candidates_per_role: int = 4
  target_diameter_mm: float = 45.0
  max_new_tokens: int = 6000

  def validate(self) -> None:
    if (
        self.seed < 0 or self.query_count < 1
        or self.candidates_per_role != len(CANDIDATE_DIAMETERS_MM)
        or self.target_diameter_mm not in CANDIDATE_DIAMETERS_MM
        or self.max_new_tokens != 6000
    ):
      raise ValueError("STEP-LLM transfer configuration differs")

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": self.seed,
        "query_count": self.query_count,
        "candidates_per_role": self.candidates_per_role,
        "target_diameter_mm": self.target_diameter_mm,
        "max_new_tokens": self.max_new_tokens,
        "adapter_repo": STEP_LLM_ADAPTER_REPO,
        "adapter_revision": STEP_LLM_ADAPTER_REVISION,
        "base_repo": STEP_LLM_BASE_REPO,
        "base_revision": STEP_LLM_BASE_REVISION,
        "do_sample": False,
  }


def _caption(
    *, family: str, diameter: float, query_ordinal: int,
) -> str:
  if family == "ring":
    outer = diameter + 14.0 + 0.5 * query_ordinal
    thickness = 5.0 + 0.1 * query_ordinal
    return (
        f"A thick annular connector with a central through bore exactly {diameter:g} mm "
        f"in diameter, an outer diameter {outer:g} mm, and thickness {thickness:g} mm."
    )
  if family == "shaft":
    length = 26.0 + 0.5 * query_ordinal
    return (
        f"A simple cylindrical shaft exactly {diameter:g} mm in diameter and "
        f"{length:g} mm long."
    )
  raise ValueError("STEP-LLM transfer family differs")


def build_step_llm_transfer_schedule_v1(
    config: StepLLMTransferConfigV1,
) -> tuple[dict[str, Any], ...]:
  config.validate()
  queries = []
  for query_ordinal in range(config.query_count):
    query_seed = int(hashlib.sha256(
        f"{config.seed}:query:{query_ordinal}".encode("utf-8")
    ).hexdigest()[:16], 16)
    roles = []
    for role_ordinal, (role_id, description, family) in enumerate(ROLE_SPECS):
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
        candidates.append({
            "candidate_key": "stepllm_" + canonical_sha256(key_payload)[:24],
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": diameter,
            "caption": caption,
            "generation_seed": int(hashlib.sha256(
                canonical_sha256(key_payload).encode("ascii")
            ).hexdigest()[:8], 16),
            "is_target": diameter == config.target_diameter_mm,
        })
      roles.append({
          "role_id": role_id,
          "role_ordinal": role_ordinal,
          "description": description,
          "family": family,
          "candidates": candidates,
      })
    query_payload = {
        "query_ordinal": query_ordinal,
        "query_seed": query_seed,
        "roles": roles,
    }
    queries.append({
        "query_id": "linkcad_stepllm_" + canonical_sha256(query_payload)[:20],
        **query_payload,
    })
  return tuple(queries)


__all__ = [
    "CANDIDATE_DIAMETERS_MM", "ROLE_SPECS", "SCHEMA_VERSION",
    "StepLLMTransferConfigV1", "build_step_llm_transfer_schedule_v1",
    "canonical_sha256",
]
