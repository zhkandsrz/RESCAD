"""Controlled language contracts for LinkCAD mobility supervision."""

from __future__ import annotations

from typing import Mapping

from .linkcad_functional_authority import MobilityContract


SCHEMA_VERSION = "linkcad_functional_language.v1"


MOBILITY_INSTRUCTIONS: Mapping[str, tuple[str, ...]] = {
    "fixed": (
        "Fasten {a} to {b} so that no relative motion remains.",
        "Join {a} and {b} as one rigid connection.",
    ),
    "revolute": (
        "Connect {a} to {b} so it can rotate about their shared axis without sliding.",
        "Make a hinge between {a} and {b}: allow rotation, but no axial translation.",
    ),
    "prismatic": (
        "Connect {a} to {b} so it can slide along their shared axis without rotating.",
        "Make a linear guide between {a} and {b}: allow translation, but no rotation.",
    ),
    "cylindrical": (
        "Connect {a} to {b} so it can both rotate and slide along their shared axis.",
        "Make a cylindrical connection between {a} and {b} with axial sliding and rotation.",
    ),
    "planar": (
        "Keep {a} on the support plane of {b} while allowing in-plane motion.",
        "Join {a} to {b} with a planar connection that can move within the plane.",
    ),
    "ball": (
        "Connect {a} to {b} at one center so it may rotate freely without translating.",
        "Make a ball joint between {a} and {b} with rotational freedom about one center.",
    ),
    "pin_slot": (
        "Place the pin of {a} in the slot of {b} so it can travel along the slot.",
        "Connect {a} and {b} with a pin-slot motion constrained to the slot path.",
    ),
}


def render_mobility_instruction(
    mobility: MobilityContract | str,
    *,
    role_a: str,
    role_b: str,
    variant: int = 0,
) -> str:
  """Render a deterministic instruction with no geometry identifiers."""

  mobility_name = mobility.name if isinstance(mobility, MobilityContract) else mobility
  templates = MOBILITY_INSTRUCTIONS.get(mobility_name)
  if templates is None:
    raise ValueError(f"unsupported mobility instruction: {mobility_name}")
  if not role_a.strip() or not role_b.strip() or role_a == role_b:
    raise ValueError("functional edge roles must be distinct non-empty strings")
  return templates[variant % len(templates)].format(a=role_a, b=role_b)


def counterfactual_mobility_names(mobility_name: str) -> tuple[str, ...]:
  """Return functionally distinct prompts for a fixed geometry/candidate set."""

  if mobility_name not in MOBILITY_INSTRUCTIONS:
    raise ValueError(f"unsupported mobility instruction: {mobility_name}")
  dominant = ("fixed", "revolute", "prismatic", "cylindrical")
  candidates = dominant if mobility_name in dominant else tuple(MOBILITY_INSTRUCTIONS)
  return tuple(name for name in candidates if name != mobility_name)
