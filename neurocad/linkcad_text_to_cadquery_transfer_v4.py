"""Style-conditioned Text-to-CadQuery schedule for source adaptation."""

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
from .linkcad_text_to_cadquery_transfer_v3 import (
    SPLIT_VARIANT_OFFSETS,
    TARGET_DIAMETERS_MM,
)


SCHEMA_VERSION = "linkcad_text_to_cadquery_transfer_schedule.v4"
RING_STYLES = (
    ("wafer_spacer", "wafer-thin spacer ring", 2.00, 0.08),
    ("thin_spacer", "thin spacer ring", 2.00, 0.20),
    ("low_profile_collar", "low-profile retaining collar", 2.00, 0.45),
    ("compact_collar", "compact retaining collar", 2.00, 0.80),
    ("standard_bushing", "standard bearing bushing", 2.00, 1.50),
    ("deep_bushing", "deep cylindrical bushing", 2.00, 2.50),
    ("long_sleeve", "long cylindrical sleeve", 2.00, 4.00),
    ("extra_long_sleeve", "extra-long cylindrical sleeve", 2.00, 6.50),
)
SHAFT_STYLES = (
    ("short_pin", "short locating pin", 0.80),
    ("stub_shaft", "stub shaft", 1.20),
    ("compact_spindle", "compact spindle", 1.80),
    ("standard_axle", "standard axle", 2.50),
    ("long_axle", "long axle", 3.50),
    ("extended_spindle", "extended spindle", 4.50),
    ("deep_reach_rod", "deep-reach rod", 5.50),
    ("extra_long_mandrel", "extra-long mandrel", 6.50),
)


@dataclass(frozen=True, slots=True)
class TextToCadQueryTransferConfigV4:
  seed: int = 2026082353
  query_count: int = 12
  split_name: str = "pilot"
  candidates_per_role: int = 8
  model_context_tokens: int = 1024

  def validate(self) -> None:
    if (
        self.seed < 0
        or self.query_count < 1
        or self.split_name not in SPLIT_VARIANT_OFFSETS
        or self.candidates_per_role != 8
        or self.model_context_tokens != 1024
    ):
      raise ValueError("Text-to-CadQuery transfer V4 configuration differs")

  def to_dict(self) -> dict[str, Any]:
    self.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": self.seed,
        "query_count": self.query_count,
        "split_name": self.split_name,
        "candidates_per_role": self.candidates_per_role,
        "ring_style_ids": [row[0] for row in RING_STYLES],
        "shaft_style_ids": [row[0] for row in SHAFT_STYLES],
        "target_diameters_mm": list(TARGET_DIAMETERS_MM),
        "query_variant_offset": SPLIT_VARIANT_OFFSETS[self.split_name] + 4000,
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
        "candidate_interface_diameter_within_query": "shared",
        "target_selection_semantics": "language_conditioned_shape_style",
    }


def _ring_instruction(
    *, outer_ratio: float, height_ratio: float, variant: int,
) -> str:
  styles = (
      "On the XY plane, draw two concentric circular profiles.",
      "Create an XY sketch from two concentric circles.",
      "Start with two concentric circles centered at the XY origin.",
  )
  return (
      "Use normalized dimensions. "
      f"{styles[variant % len(styles)]} The inner diameter is exactly 1.000 "
      f"units and the outer diameter is {outer_ratio:.4f} units. Extrude the "
      f"annular region by {height_ratio:.4f} units to create exactly one "
      "hollow cylindrical solid."
  )


def _shaft_instruction(*, length_ratio: float, variant: int) -> str:
  styles = (
      "Draw one circular sketch on the XY plane.",
      "On the XY plane, create one circle centered at the origin.",
      "Begin with a circular profile centered on the default axis.",
  )
  return (
      "Use normalized dimensions. "
      f"{styles[variant % len(styles)]} Set the cylinder diameter to exactly "
      f"1.000 units and extrude it by {length_ratio:.4f} units to create "
      "exactly one solid cylindrical shaft."
  )


def build_text_to_cadquery_transfer_schedule_v4(
    config: TextToCadQueryTransferConfigV4,
) -> tuple[dict[str, Any], ...]:
  config.validate()
  offset = SPLIT_VARIANT_OFFSETS[config.split_name] + 4000
  target_shift = config.seed % len(RING_STYLES)
  diameter_shift = (config.seed // len(RING_STYLES)) % len(TARGET_DIAMETERS_MM)
  queries = []
  for query_ordinal in range(config.query_count):
    query_variant = query_ordinal + offset
    interface_diameter = TARGET_DIAMETERS_MM[
        (query_ordinal + diameter_shift) % len(TARGET_DIAMETERS_MM)
    ]
    ring_target = RING_STYLES[(query_ordinal + target_shift) % len(RING_STYLES)]
    shaft_target = SHAFT_STYLES[
        (3 * query_ordinal + target_shift + 1) % len(SHAFT_STYLES)
    ]
    query_seed = int(hashlib.sha256(
        f"v4:{config.seed}:{config.split_name}:{query_ordinal}:"
        f"{query_variant}:{interface_diameter}:"
        f"{ring_target[0]}:{shaft_target[0]}".encode()
    ).hexdigest()[:16], 16)
    role_specs = (
        (
            "role_00", "ring",
            f"{ring_target[1]} with a {interface_diameter:g} mm bore",
            RING_STYLES, ring_target[0],
        ),
        (
            "role_01", "shaft",
            f"{shaft_target[1]} with a {interface_diameter:g} mm bearing diameter",
            SHAFT_STYLES, shaft_target[0],
        ),
    )
    roles = []
    for role_ordinal, (
        role_id, family, description, raw_styles, target_style_id,
    ) in enumerate(role_specs):
      candidate_styles = list(raw_styles)
      random.Random(query_seed ^ (role_ordinal << 20)).shuffle(candidate_styles)
      candidates = []
      for candidate_ordinal, style in enumerate(candidate_styles):
        style_id, style_description, *dimensions = style
        variant = query_variant * 16 + role_ordinal * 8 + candidate_ordinal
        jitter_rng = random.Random(query_seed ^ (role_ordinal << 24) ^ variant)
        jitter = 1.0 + jitter_rng.uniform(-0.015, 0.015)
        if family == "ring":
          instruction = _ring_instruction(
              outer_ratio=float(dimensions[0]) * jitter,
              height_ratio=float(dimensions[1]) / jitter,
              variant=variant,
          )
        else:
          instruction = _shaft_instruction(
              length_ratio=float(dimensions[0]) * jitter,
              variant=variant,
          )
        digest = canonical_sha256({
            "schema_version": SCHEMA_VERSION,
            "split_name": config.split_name,
            "query_seed": query_seed,
            "role_id": role_id,
            "candidate_ordinal": candidate_ordinal,
            "style_id": style_id,
            "requested_diameter_mm": interface_diameter,
            "instruction": instruction,
        })
        candidates.append({
            "candidate_key": "t2cq4_" + digest[:24],
            "candidate_ordinal": candidate_ordinal,
            "requested_diameter_mm": interface_diameter,
            "style_id": style_id,
            "style_description": style_description,
            "instruction": instruction,
            "is_target": style_id == target_style_id,
        })
      roles.append({
          "role_id": role_id,
          "role_ordinal": role_ordinal,
          "description": description,
          "family": family,
          "target_requested_diameter_mm": interface_diameter,
          "target_style_id": target_style_id,
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
        "query_id": "linkcad_t2cq4_" + canonical_sha256(query_payload)[:20],
        **query_payload,
    })
  return tuple(queries)


__all__ = [
    "RING_STYLES", "SCHEMA_VERSION", "SHAFT_STYLES",
    "TextToCadQueryTransferConfigV4",
    "build_text_to_cadquery_transfer_schedule_v4",
]
