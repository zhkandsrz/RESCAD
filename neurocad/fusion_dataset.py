"""Fusion 360 Gallery-style assembly dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Optional


def _as_dict(value: Any) -> dict[str, Any]:
  if isinstance(value, dict):
    return value
  return {}


def _safe_text(value: Any, fallback: str) -> str:
  if isinstance(value, str) and value.strip():
    return value.strip()
  return fallback


def _normalize_pair(a: str, b: str) -> tuple[str, str]:
  if a <= b:
    return (a, b)
  return (b, a)


def _sanitize_name(text: str, fallback: str = "part") -> str:
  cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower())
  cleaned = re.sub(r"_+", "_", cleaned).strip("_")
  if cleaned:
    return cleaned
  return fallback


_GENERIC_OCCURRENCE_NAMES = {
    "root",
    "body",
    "body1",
    "body2",
    "component",
    "part",
    "occurrence",
}


def _preferred_part_label(
    occurrence_name: Optional[str],
    body_name: str,
    step_stem: str,
) -> str:
  occ = str(occurrence_name or "").strip()
  occ_key = _sanitize_name(occ, fallback="")
  body_key = _sanitize_name(body_name, fallback="")
  step_key = _sanitize_name(step_stem, fallback="")
  if occ_key and occ_key not in _GENERIC_OCCURRENCE_NAMES:
    return occ
  if body_key and body_key not in _GENERIC_OCCURRENCE_NAMES:
    return body_name
  if step_key and step_key not in _GENERIC_OCCURRENCE_NAMES:
    return step_stem
  if occ:
    return occ
  if body_name:
    return body_name
  return step_stem or "part"


@dataclass
class FusionBodyCandidate:
  body_uuid: str
  step_path: Path
  part_name: str
  body_name: str
  occurrence_name: Optional[str]
  is_grounded: bool
  is_visible: bool
  volume: float
  contact_degree: int = 0


@dataclass
class FusionAssemblySample:
  assembly_id: str
  assembly_dir: Path
  assembly_json: Path
  body_candidates: list[FusionBodyCandidate]
  selected_bodies: list[FusionBodyCandidate]
  contacts_uuid: list[tuple[str, str]]
  selected_contact_pairs: list[tuple[str, str]]
  body_uuid_to_part: dict[str, str]
  instruction: str


def discover_fusion_assembly_dirs(
    dataset_root: str | Path,
    max_assemblies: int = 0,
) -> list[Path]:
  root = Path(dataset_root)
  if not root.exists() or not root.is_dir():
    raise ValueError(f"dataset_root not found or not dir: {root}")
  dirs = sorted(
      [
          p
          for p in root.iterdir()
          if p.is_dir() and (p / "assembly.json").exists()
      ],
      key=lambda p: p.name.lower(),
  )
  if max_assemblies > 0:
    dirs = dirs[:max_assemblies]
  return dirs


def load_fusion_assembly_sample(
    assembly_dir: str | Path,
    max_parts: int = 8,
    body_selection_mode: str = "contact_degree",
    rng=None,
) -> FusionAssemblySample:
  folder = Path(assembly_dir)
  if not folder.exists() or not folder.is_dir():
    raise ValueError(f"assembly_dir not found or not dir: {folder}")
  json_path = folder / "assembly.json"
  if not json_path.exists():
    raise ValueError(f"assembly.json not found in: {folder}")

  with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

  bodies_table = _as_dict(data.get("bodies"))
  occurrences = _as_dict(data.get("occurrences"))
  root = _as_dict(data.get("root"))
  root_bodies = _as_dict(root.get("bodies"))

  body_occ_map: dict[str, dict[str, Any]] = {}

  def _set_occ_hint(
      body_uuid: str,
      occ_name: Optional[str],
      grounded: bool,
      visible: bool,
  ) -> None:
    previous = body_occ_map.get(body_uuid)
    incoming = {
        "occurrence_name": occ_name,
        "is_grounded": bool(grounded),
        "is_visible": bool(visible),
    }
    if previous is None:
      body_occ_map[body_uuid] = incoming
      return
    prev_score = (
        int(bool(previous.get("is_grounded"))),
        int(bool(previous.get("is_visible"))),
        int(bool(previous.get("occurrence_name"))),
    )
    new_score = (
        int(bool(incoming.get("is_grounded"))),
        int(bool(incoming.get("is_visible"))),
        int(bool(incoming.get("occurrence_name"))),
    )
    if new_score > prev_score:
      body_occ_map[body_uuid] = incoming

  for occ_id, raw_occ in occurrences.items():
    occ = _as_dict(raw_occ)
    occ_name = _safe_text(occ.get("name"), occ_id)
    grounded = bool(occ.get("is_grounded", False))
    occ_bodies = _as_dict(occ.get("bodies"))
    for body_uuid, raw_body_ref in occ_bodies.items():
      body_ref = _as_dict(raw_body_ref)
      visible = bool(body_ref.get("is_visible", True))
      _set_occ_hint(
          body_uuid=body_uuid,
          occ_name=occ_name,
          grounded=grounded,
          visible=visible,
      )

  for body_uuid, raw_body_ref in root_bodies.items():
    body_ref = _as_dict(raw_body_ref)
    visible = bool(body_ref.get("is_visible", True))
    _set_occ_hint(
        body_uuid=body_uuid,
        occ_name="root",
        grounded=False,
        visible=visible,
    )

  contact_pairs: set[tuple[str, str]] = set()
  contacts = data.get("contacts")
  if isinstance(contacts, list):
    for item in contacts:
      if not isinstance(item, dict):
        continue
      entity_one = _as_dict(item.get("entity_one"))
      entity_two = _as_dict(item.get("entity_two"))
      body_a = entity_one.get("body")
      body_b = entity_two.get("body")
      if not isinstance(body_a, str) or not isinstance(body_b, str):
        continue
      if body_a == body_b:
        continue
      contact_pairs.add(_normalize_pair(body_a, body_b))

  contact_degree: dict[str, int] = {}
  for body_a, body_b in contact_pairs:
    contact_degree[body_a] = contact_degree.get(body_a, 0) + 1
    contact_degree[body_b] = contact_degree.get(body_b, 0) + 1

  candidates: list[FusionBodyCandidate] = []
  used_names: set[str] = set()
  for body_uuid, raw_body in bodies_table.items():
    body = _as_dict(raw_body)
    step_name = _safe_text(body.get("step"), f"{body_uuid}.step")
    step_path = folder / step_name
    if not step_path.exists():
      fallback_step = folder / f"{body_uuid}.step"
      if fallback_step.exists():
        step_path = fallback_step
      else:
        continue

    body_name = _safe_text(body.get("name"), f"body_{body_uuid[:8]}")
    occ_hint = _as_dict(body_occ_map.get(body_uuid))
    occurrence_name = occ_hint.get("occurrence_name")
    is_grounded = bool(occ_hint.get("is_grounded", False))
    is_visible = bool(occ_hint.get("is_visible", True))

    physical = _as_dict(body.get("physical_properties"))
    try:
      volume = float(physical.get("volume", 0.0))
    except (TypeError, ValueError):
      volume = 0.0

    preferred_label = _preferred_part_label(
        occurrence_name=(
            None if not isinstance(occurrence_name, str) else occurrence_name
        ),
        body_name=body_name,
        step_stem=step_path.stem,
    )
    base_label = _sanitize_name(preferred_label, fallback="part")
    short_id = body_uuid.split("-", 1)[0][:8]
    part_name = f"{base_label}_{short_id}"
    suffix = 1
    while part_name in used_names:
      part_name = f"{base_label}_{short_id}_{suffix:02d}"
      suffix += 1
    used_names.add(part_name)

    candidates.append(
      FusionBodyCandidate(
          body_uuid=body_uuid,
          step_path=step_path,
          part_name=part_name,
          body_name=body_name,
          occurrence_name=(
              None if not isinstance(occurrence_name, str) else occurrence_name
          ),
          is_grounded=is_grounded,
          is_visible=is_visible,
          volume=volume,
          contact_degree=contact_degree.get(body_uuid, 0),
      )
    )

  if not candidates:
    raise ValueError(f"No valid body STEP files found in: {folder}")

  selected_bodies = select_fusion_bodies(
      bodies=candidates,
      max_parts=max_parts,
      selection_mode=body_selection_mode,
      rng=rng,
  )
  if len(selected_bodies) < 2 and len(candidates) >= 2:
    fallback = sorted(
        candidates,
        key=lambda c: (c.contact_degree, c.volume),
        reverse=True,
    )[:2]
    selected_bodies = fallback

  body_uuid_to_part = {
      body.body_uuid: body.part_name for body in selected_bodies
  }
  selected_contact_pairs = sorted(
      [
          (
              body_uuid_to_part[body_a],
              body_uuid_to_part[body_b],
          )
          for body_a, body_b in sorted(contact_pairs)
          if body_a in body_uuid_to_part and body_b in body_uuid_to_part
      ],
      key=lambda pair: (pair[0], pair[1]),
  )

  instruction = build_fusion_instruction(
      assembly_id=folder.name,
      selected_bodies=selected_bodies,
      selected_contact_pairs=selected_contact_pairs,
  )
  return FusionAssemblySample(
      assembly_id=folder.name,
      assembly_dir=folder,
      assembly_json=json_path,
      body_candidates=candidates,
      selected_bodies=selected_bodies,
      contacts_uuid=sorted(contact_pairs),
      selected_contact_pairs=selected_contact_pairs,
      body_uuid_to_part=body_uuid_to_part,
      instruction=instruction,
  )


def select_fusion_bodies(
    bodies: list[FusionBodyCandidate],
    max_parts: int = 8,
    selection_mode: str = "contact_degree",
    rng=None,
) -> list[FusionBodyCandidate]:
  if not bodies:
    return []
  max_parts = max(1, min(int(max_parts), len(bodies)))
  mode = str(selection_mode).strip().lower()
  if mode not in {"contact_degree", "volume", "random"}:
    mode = "contact_degree"

  selected: list[FusionBodyCandidate] = []
  selected_ids: set[str] = set()

  grounded = sorted(
      [body for body in bodies if body.is_grounded],
      key=lambda body: (body.contact_degree, body.volume),
      reverse=True,
  )
  for body in grounded:
    if body.body_uuid in selected_ids:
      continue
    selected.append(body)
    selected_ids.add(body.body_uuid)
    if len(selected) >= max_parts:
      return selected

  remaining = [body for body in bodies if body.body_uuid not in selected_ids]
  if mode == "random":
    if rng is None:
      import random as _random
      _random.shuffle(remaining)
    else:
      rng.shuffle(remaining)
  elif mode == "volume":
    remaining = sorted(
        remaining,
        key=lambda body: (body.volume, body.contact_degree),
        reverse=True,
    )
  else:
    remaining = sorted(
        remaining,
        key=lambda body: (body.contact_degree, body.volume),
        reverse=True,
    )

  for body in remaining:
    selected.append(body)
    if len(selected) >= max_parts:
      break
  return selected


def build_fusion_instruction(
    assembly_id: str,
    selected_bodies: list[FusionBodyCandidate],
    selected_contact_pairs: list[tuple[str, str]],
) -> str:
  part_names = [body.part_name for body in selected_bodies]
  shown_parts = ", ".join(part_names[:12])
  if len(part_names) > 12:
    shown_parts += f", ... ({len(part_names)} total)"

  text = (
      f"Assemble a connected mechanism for assembly {assembly_id}. "
      f"Parts: {shown_parts}. "
      "Every part must be connected to the final graph using legal mechanical constraints."
  )

  if selected_contact_pairs:
    hints = ", ".join(
        f"{a}<->{b}" for a, b in selected_contact_pairs[:14]
    )
    text += (
        " Prefer constraints consistent with these known contact pairs: "
        + hints
        + "."
    )

  text += (
      " Use concentric for hole/pin/axis alignment and coincident for planar mating."
  )
  return text
