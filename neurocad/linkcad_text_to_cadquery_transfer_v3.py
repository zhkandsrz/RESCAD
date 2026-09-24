"""Eight-candidate Text-to-CadQuery schedule for source adaptation."""

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
from .linkcad_text_to_cadquery_transfer_v2 import _instruction


SCHEMA_VERSION = "linkcad_text_to_cadquery_transfer_schedule.v3"
CANDIDATE_DIAMETERS_MM = (8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 36.0, 45.0)
TARGET_DIAMETERS_MM = (12.0, 16.0, 20.0, 24.0, 30.0, 36.0)
SPLIT_VARIANT_OFFSETS = {
    "pilot": 100,
    "train": 1000,
    "dev": 2000,
    "test": 3000,
}


@dataclass(frozen=True, slots=True)
class TextToCadQueryTransferConfigV3:
  seed: int = 2026082341
  query_count: int = 12
  split_name: str = "pilot"
  candidates_per_role: int = 8
  model_context_tokens: int = 1024

  def validate(self) -> None:
    if (
        self.seed < 0
        or self.query_count < 1
        or self.split_name not in SPLIT_VARIANT_OFFSETS
        or self.candidates_per_role != len(CANDIDATE_DIAMETERS_MM)
        or self.model_context_tokens != 1024
    ):
      raise ValueError("Text-to-CadQuery transfer V3 configuration differs")

  def to_dict(self) -> dict[str, Any]:
    self.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": self.seed,
        "query_count": self.query_count,
        "split_name": self.split_name,
        "candidates_per_role": self.candidates_per_role,
        "candidate_diameters_mm": list(CANDIDATE_DIAMETERS_MM),
        "target_diameters_mm": list(TARGET_DIAMETERS_MM),
        "query_variant_offset": SPLIT_VARIANT_OFFSETS[self.split_name],
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
        "minimum_ready_candidates_per_role": 6,
        "minimum_complete_assignment_count": 36,
    }


def build_text_to_cadquery_transfer_schedule_v3(
    config: TextToCadQueryTransferConfigV3,
) -> tuple[dict[str, Any], ...]:
  config.validate()
  offset = SPLIT_VARIANT_OFFSETS[config.split_name]
  target_shift = config.seed % len(TARGET_DIAMETERS_MM)
  queries = []
  for query_ordinal in range(config.query_count):
    query_variant = query_ordinal + offset
    target_diameter = TARGET_DIAMETERS_MM[
        (query_ordinal + target_shift) % len(TARGET_DIAMETERS_MM)
    ]
    query_seed = int(hashlib.sha256(
        f"v3:{config.seed}:{config.split_name}:{query_ordinal}:"
        f"{query_variant}:{target_diameter}".encode()
    ).hexdigest()[:16], 16)
    role_specs = (
        (
            "role_00",
            f"annular connector with a {target_diameter:g} mm bore",
            "ring",
        ),
        (
            "role_01",
            f"axial shaft {target_diameter:g} mm in diameter",
            "shaft",
        ),
    )
    roles = []
    for role_ordinal, (role_id, description, family) in enumerate(role_specs):
      diameters = list(CANDIDATE_DIAMETERS_MM)
      random.Random(query_seed ^ (role_ordinal << 19)).shuffle(diameters)
      candidates = []
      for candidate_ordinal, diameter in enumerate(diameters):
        variant = query_variant * 16 + role_ordinal * 8 + candidate_ordinal
        instruction = _instruction(family=family, variant=variant)
        digest = canonical_sha256({
            "schema_version": SCHEMA_VERSION,
            "split_name": config.split_name,
            "query_seed": query_seed,
            "role_id": role_id,
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": diameter,
            "instruction": instruction,
        })
        candidates.append({
            "candidate_key": "t2cq3_" + digest[:24],
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": diameter,
            "instruction": instruction,
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
        "query_variant": query_variant,
        "query_seed": query_seed,
        "split_name": config.split_name,
        "roles": roles,
    }
    queries.append({
        "query_id": "linkcad_t2cq3_" + canonical_sha256(query_payload)[:20],
        **query_payload,
    })
  return tuple(queries)


__all__ = [
    "CANDIDATE_DIAMETERS_MM", "SCHEMA_VERSION", "SPLIT_VARIANT_OFFSETS",
    "TARGET_DIAMETERS_MM", "TextToCadQueryTransferConfigV3",
    "build_text_to_cadquery_transfer_schedule_v3",
]
