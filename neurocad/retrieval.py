"""Natural-language part retrieval for prompt-to-CAD workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Optional
from urllib import error as url_error
from urllib import request as url_request

from .fusion_dataset import (
    FusionBodyCandidate,
    discover_fusion_assembly_dirs,
    load_fusion_assembly_sample,
)


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_CONCEPT_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "gear": {
        "strong": (
            "gear",
            "gear wheel",
            "spur gear",
            "bevel gear",
            "worm gear",
            "ratchet gear",
            "sprocket",
            "cog",
            "\u9f7f\u8f6e",
        ),
        "weak": ("pinion", "wheel"),
        "negative": ("gearbox", "gear box", "gearmotor", "gear motor"),
    },
    "shaft": {
        "strong": (
            "shaft",
            "axle",
            "arbor",
            "spindle",
            "driveshaft",
            "drive shaft",
            "camshaft",
            "crankshaft",
            "\u8f74",
            "\u8f6c\u8f74",
            "\u5fc3\u8f74",
        ),
        "weak": ("rod",),
        "negative": (),
    },
    "base": {
        "strong": (
            "base",
            "housing",
            "frame",
            "bracket",
            "mount",
            "holder",
            "stand",
            "chassis",
            "support",
            "\u5e95\u5ea7",
            "\u57fa\u5ea7",
            "\u6846\u67b6",
            "\u673a\u67b6",
            "\u58f3\u4f53",
            "\u652f\u67b6",
        ),
        "weak": ("plate", "panel", "baseplate", "\u5e95\u677f"),
        "negative": (),
    },
    "bearing": {
        "strong": ("bearing", "bushing", "insert", "\u8f74\u627f"),
        "weak": (),
        "negative": (),
    },
    "collar": {
        "strong": (
            "collar",
            "sleeve",
            "spacer",
            "washer",
            "bushing",
            "\u5957\u73af",
            "\u8f74\u5957",
            "\u886c\u5957",
            "\u57ab\u5708",
            "\u9694\u5957",
        ),
        "weak": (),
        "negative": (),
    },
    "bolt": {
        "strong": (
            "bolt",
            "screw",
            "fastener",
            "stud",
            "threaded rod",
            "\u87ba\u6813",
            "\u87ba\u4e1d",
            "\u87ba\u9489",
            "\u7d27\u56fa\u4ef6",
            "\u87ba\u6746",
        ),
        "weak": ("unc", "unf", "thread"),
        "negative": (),
    },
}

_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "onto",
    "into",
    "on",
    "in",
    "of",
    "and",
    "then",
    "with",
    "for",
    "using",
    "put",
    "place",
    "mount",
    "attach",
    "fix",
    "assemble",
    "connected",
    "mechanism",
    "part",
    "parts",
    "cad",
    "\u4e00\u4e2a",
    "\u628a",
    "\u88c5\u5230",
    "\u653e\u5230",
    "\u518d",
    "\u7136\u540e",
    "\u56fa\u5b9a",
    "\u7ec4\u88c5",
}


@dataclass
class RetrievedPartEntry:
  assembly_id: str
  assembly_dir: Path
  step_path: Path
  part_name: str
  body_uuid: str
  body_name: str
  occurrence_name: Optional[str]
  is_grounded: bool
  is_visible: bool
  volume: float
  contact_degree: int
  search_tokens: set[str] = field(default_factory=set)
  search_concepts: set[str] = field(default_factory=set)
  concept_scores: dict[str, float] = field(default_factory=dict)
  anchor_tokens: set[str] = field(default_factory=set)
  semantic_text: str = ""
  vlm_caption: str = ""
  caption_embedding_text: str = ""
  thumbnail_paths: tuple[str, ...] = ()
  embedding: Optional[list[float]] = None
  bbox_dims: tuple[float, float, float] | None = None
  hole_radii: tuple[float, ...] = ()
  pin_radii: tuple[float, ...] = ()
  hole_count: int = 0
  pin_count: int = 0
  threaded_hole_count: int = 0
  plane_count: int = 0
  score: float = 0.0
  match_reasons: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
        "assembly_id": self.assembly_id,
        "assembly_dir": str(self.assembly_dir.resolve()),
        "step_path": str(self.step_path.resolve()),
        "part_name": self.part_name,
        "body_uuid": self.body_uuid,
        "body_name": self.body_name,
        "occurrence_name": self.occurrence_name,
        "is_grounded": bool(self.is_grounded),
        "is_visible": bool(self.is_visible),
        "volume": round(float(self.volume), 6),
        "contact_degree": int(self.contact_degree),
        "search_tokens": sorted(self.search_tokens),
        "search_concepts": sorted(self.search_concepts),
        "concept_scores": {
            key: round(float(value), 4)
            for key, value in sorted(self.concept_scores.items())
            if float(value) > 0.0
        },
        "anchor_tokens": sorted(self.anchor_tokens),
        "semantic_text": self.semantic_text,
        "vlm_caption": self.vlm_caption,
        "caption_embedding_text": self.caption_embedding_text,
        "thumbnail_paths": list(self.thumbnail_paths),
        "bbox_dims": (
            None if self.bbox_dims is None else [round(v, 4) for v in self.bbox_dims]
        ),
        "hole_radii": [round(v, 4) for v in self.hole_radii[:8]],
        "pin_radii": [round(v, 4) for v in self.pin_radii[:8]],
        "hole_count": int(self.hole_count),
        "pin_count": int(self.pin_count),
        "threaded_hole_count": int(self.threaded_hole_count),
        "plane_count": int(self.plane_count),
        "score": round(float(self.score), 4),
        "match_reasons": list(self.match_reasons),
    }


@dataclass
class PartRetrievalIndex:
  dataset_root: Path
  dataset_roots: list[Path]
  entries: list[RetrievedPartEntry]
  assembly_contact_pairs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
  embedding_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedPartSelection:
  instruction: str
  selected_parts: list[RetrievedPartEntry]
  prompt_tokens: list[str]
  prompt_concepts: list[str]
  notes: list[str] = field(default_factory=list)
  contact_pairs: list[tuple[str, str]] = field(default_factory=list)
  preferred_assembly_id: Optional[str] = None
  role_assignments: dict[str, str] = field(default_factory=dict)
  semantic_score: float = 0.0
  candidate_pool: list[RetrievedPartEntry] = field(default_factory=list)

  def to_case(self, case_id: str) -> dict[str, Any]:
    assembly_id = self._single_assembly_id()
    assembly_dir = assembly_id if assembly_id else None
    return {
        "id": case_id,
        "instruction": self.instruction,
        "parts": [str(entry.step_path.resolve()) for entry in self.selected_parts],
        "assembly_dir": assembly_dir,
        "contact_pairs": [list(pair) for pair in self.contact_pairs],
        "selected_part_names": [entry.part_name for entry in self.selected_parts],
        "selected_body_uuids": [entry.body_uuid for entry in self.selected_parts],
        "retrieval": self.to_dict(),
    }

  def to_dict(self) -> dict[str, Any]:
    return {
        "instruction": self.instruction,
        "prompt_tokens": list(self.prompt_tokens),
        "prompt_concepts": list(self.prompt_concepts),
        "preferred_assembly_id": self.preferred_assembly_id,
        "role_assignments": dict(self.role_assignments),
        "semantic_score": round(float(self.semantic_score), 4),
        "notes": list(self.notes),
        "contact_pairs": [list(pair) for pair in self.contact_pairs],
        "selected_parts": [entry.to_dict() for entry in self.selected_parts],
        "candidate_pool": [entry.to_dict() for entry in self.candidate_pool[:16]],
    }

  def _single_assembly_id(self) -> Optional[str]:
    assembly_ids = {entry.assembly_id for entry in self.selected_parts}
    if len(assembly_ids) == 1:
      return next(iter(assembly_ids))
    return None


def build_part_retrieval_index(
    dataset_root: str | Path | list[str] | list[Path],
    max_assemblies: int = 0,
) -> PartRetrievalIndex:
  roots = _normalize_dataset_roots(dataset_root)
  if not roots:
    raise ValueError("At least one dataset root is required.")
  entries: list[RetrievedPartEntry] = []
  assembly_contact_pairs: dict[str, list[tuple[str, str]]] = {}

  if max_assemblies > 0:
    per_root_limit = max(1, int(max_assemblies))
  else:
    per_root_limit = 0

  for root in roots:
    for assembly_dir in discover_fusion_assembly_dirs(
        root, max_assemblies=per_root_limit
    ):
      sample = load_fusion_assembly_sample(
          assembly_dir=assembly_dir,
          max_parts=1000000,
          body_selection_mode="contact_degree",
      )
      assembly_contact_pairs[sample.assembly_id] = list(sample.selected_contact_pairs)
      for body in sample.selected_bodies:
        entries.append(
            _entry_from_body_candidate(
                assembly_id=sample.assembly_id,
                assembly_dir=sample.assembly_dir,
                body=body,
            )
        )

  return PartRetrievalIndex(
      dataset_root=roots[0],
      dataset_roots=list(roots),
      entries=entries,
      assembly_contact_pairs=assembly_contact_pairs,
  )


def enable_embedding_rerank(
    index: PartRetrievalIndex,
    *,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: int = 45,
    batch_size: int = 48,
    score_weight: float = 2.25,
) -> PartRetrievalIndex:
  if api_key:
    raise ValueError(
        "Direct embedding API keys are disabled; use "
        "NEUROCAD_EMBEDDING_API_KEY or a provider environment variable."
    )
  model_name = str(model or "").strip()
  if not model_name:
    raise ValueError("embedding model is required")
  texts = [
      entry.caption_embedding_text or entry.semantic_text or entry.part_name
      for entry in index.entries
  ]
  vectors = _embed_texts(
      texts=texts,
      model=model_name,
      base_url=base_url,
      api_key=api_key,
      timeout_seconds=timeout_seconds,
      batch_size=batch_size,
  )
  if len(vectors) != len(index.entries):
    raise RuntimeError("embedding count does not match retrieval entries")
  for entry, vector in zip(index.entries, vectors):
    entry.embedding = vector
  index.embedding_info = {
      "enabled": True,
      "model": model_name,
      "mode": "visual_semantic",
      "base_url": (base_url or "https://api.openai.com/v1").rstrip("/"),
      "api_key_source": "environment",
      "timeout_seconds": int(timeout_seconds),
      "batch_size": int(batch_size),
      "score_weight": float(score_weight),
  }
  return index


def infer_prompt_concepts(text: str) -> list[str]:
  return _ordered_prompt_concepts(text)


def infer_prompt_tokens(text: str) -> list[str]:
  return sorted(_prompt_tokens(text))


def infer_part_text_concepts(text: str) -> list[str]:
  scores = _concept_scores_from_text(text)
  return sorted(
      concept
      for concept, score in scores.items()
      if float(score) >= 0.45
  )


def _normalize_dataset_roots(
    dataset_root: str | Path | list[str] | list[Path],
) -> list[Path]:
  if isinstance(dataset_root, (str, Path)):
    raw_items = [dataset_root]
  else:
    raw_items = list(dataset_root)

  roots: list[Path] = []
  seen: set[str] = set()
  for item in raw_items:
    if item is None:
      continue
    if isinstance(item, Path):
      parts = [str(item)]
    else:
      parts = re.split(r"[;\n\r]+", str(item))
    for part in parts:
      text = str(part).strip().strip('"').strip("'")
      if not text:
        continue
      path = Path(text)
      key = str(path.resolve()) if path.exists() else str(path)
      if key in seen:
        continue
      seen.add(key)
      roots.append(path)
  return roots


def retrieve_candidate_parts(
    instruction: str,
    index: PartRetrievalIndex,
    part_count: int = 0,
    prefer_same_assembly: bool = True,
) -> RetrievedPartSelection:
  selections = retrieve_candidate_part_sets(
      instruction=instruction,
      index=index,
      part_count=part_count,
      prefer_same_assembly=prefer_same_assembly,
  )
  if not selections:
    raise ValueError("No retrieval candidates matched the prompt.")
  return selections[0]


def retrieve_candidate_part_sets(
    instruction: str,
    index: PartRetrievalIndex,
    part_count: int = 0,
    prefer_same_assembly: bool = True,
    per_concept_limit: int = 6,
    max_sets: int = 12,
) -> list[RetrievedPartSelection]:
  prompt_tokens = _prompt_tokens(instruction)
  prompt_concepts = _ordered_prompt_concepts(instruction)
  prompt_anchor_tokens = _spec_anchor_tokens(instruction)
  target_count = _target_part_count(
      instruction=instruction,
      prompt_concepts=prompt_concepts,
      part_count=part_count,
  )

  scored = [
      _score_entry(
          instruction=instruction,
          prompt_tokens=prompt_tokens,
          prompt_concepts=prompt_concepts,
          prompt_anchor_tokens=prompt_anchor_tokens,
          entry=entry,
      )
      for entry in index.entries
  ]
  scored = [entry for entry in scored if entry.score > 0.0]
  query_embedding = _query_embedding(instruction=instruction, index=index)
  if query_embedding is not None:
    embedding_weight = float(index.embedding_info.get("score_weight", 2.25))
    reranked: list[RetrievedPartEntry] = []
    for entry in scored:
      updated = RetrievedPartEntry(**entry.__dict__)
      if entry.embedding:
        sim = _cosine_similarity(query_embedding, entry.embedding)
        anchor_coverage = 0.0
        if prompt_anchor_tokens:
          anchor_coverage = (
              float(len(prompt_anchor_tokens & entry.anchor_tokens))
              / float(len(prompt_anchor_tokens))
          )
        rerank_scale = (
            1.0
            if not prompt_anchor_tokens
            else (0.0 if anchor_coverage <= 0.0 else 0.2 + 0.8 * anchor_coverage)
        )
        updated.score += embedding_weight * sim * rerank_scale
        if sim > 0.0:
          updated.match_reasons = list(updated.match_reasons) + [
              f"visual_embed={sim:.3f}*{rerank_scale:.2f}"
          ]
      reranked.append(updated)
    scored = reranked
  scored.sort(
      key=lambda item: (
          -float(item.score),
          -int(item.contact_degree),
          -float(item.volume),
          item.part_name,
      )
  )
  if not scored:
    raise ValueError("No retrieval candidates matched the prompt.")

  candidate_pool = list(scored[: max(16, per_concept_limit * max(1, target_count))])
  if not prompt_concepts:
    return [
        _selection_from_entries(
            instruction=instruction,
            prompt_tokens=prompt_tokens,
            prompt_concepts=prompt_concepts,
            entries=candidate_pool[:target_count],
            preferred_assembly_id=None,
            index=index,
            notes=[
                f"retrieval_target_count={target_count}",
                "retrieval_prompt_concepts=none",
                "retrieval_prompt_anchors="
                + (",".join(sorted(prompt_anchor_tokens)) or "none"),
            ],
            semantic_score=sum(float(entry.score) for entry in candidate_pool[:target_count]),
            role_assignments={},
            candidate_pool=candidate_pool,
        )
    ]

  same_assembly_sets = _same_assembly_role_assignments(
      prompt_concepts=prompt_concepts,
      prompt_anchor_tokens=prompt_anchor_tokens,
      entries=scored,
      prefer_same_assembly=prefer_same_assembly,
  )
  cross_sets = _cross_assembly_role_assignments(
      prompt_concepts=prompt_concepts,
      prompt_anchor_tokens=prompt_anchor_tokens,
      entries=scored,
      per_concept_limit=max(2, int(per_concept_limit)),
      max_sets=max(4, int(max_sets) * 2),
  )
  raw_assignments = same_assembly_sets + cross_sets
  if not raw_assignments:
    fallback = _select_entries(
        prompt_concepts=prompt_concepts,
        entries=scored,
        target_count=target_count,
        preferred_assembly_id=None,
    )
    return [
        _selection_from_entries(
            instruction=instruction,
            prompt_tokens=prompt_tokens,
            prompt_concepts=prompt_concepts,
            entries=fallback,
            preferred_assembly_id=None,
            index=index,
            notes=[
                f"retrieval_target_count={target_count}",
                "retrieval_prompt_concepts=" + ",".join(prompt_concepts),
                "retrieval_prompt_anchors="
                + (",".join(sorted(prompt_anchor_tokens)) or "none"),
                "retrieval_assignment_fallback=true",
            ],
            semantic_score=sum(float(entry.score) for entry in fallback),
            role_assignments={},
            candidate_pool=candidate_pool,
        )
    ]

  deduped: list[tuple[dict[str, RetrievedPartEntry], float, list[str]]] = []
  seen_keys: set[tuple[str, ...]] = set()
  for assignment, semantic_score, assignment_notes in sorted(
      raw_assignments,
      key=lambda item: (
          -float(item[1]),
          0 if _single_assembly_id_from_assignment(item[0]) else 1,
          tuple(sorted(item[0].keys())),
      ),
  ):
    ordered_names = tuple(
        sorted(entry.part_name for entry in assignment.values())
    )
    if ordered_names in seen_keys:
      continue
    seen_keys.add(ordered_names)
    deduped.append((assignment, semantic_score, assignment_notes))
    if len(deduped) >= max(2, int(max_sets)):
      break

  selections: list[RetrievedPartSelection] = []
  for assignment, semantic_score, assignment_notes in deduped:
    preferred_assembly_id = _single_assembly_id_from_assignment(assignment)
    ordered_entries = _ordered_entries_from_assignment(
        prompt_concepts=prompt_concepts,
        assignment=assignment,
        target_count=target_count,
        candidate_pool=scored,
        preferred_assembly_id=preferred_assembly_id,
    )
    notes = [
        f"retrieval_target_count={target_count}",
        "retrieval_prompt_concepts=" + ",".join(prompt_concepts),
        "retrieval_prompt_anchors="
        + (",".join(sorted(prompt_anchor_tokens)) or "none"),
        f"retrieval_semantic_score={semantic_score:.4f}",
        *assignment_notes,
    ]
    selection = _selection_from_entries(
        instruction=instruction,
        prompt_tokens=prompt_tokens,
        prompt_concepts=prompt_concepts,
        entries=ordered_entries,
        preferred_assembly_id=preferred_assembly_id,
        index=index,
        notes=notes,
        semantic_score=semantic_score,
        role_assignments={
            concept: entry.part_name for concept, entry in assignment.items()
        },
        candidate_pool=candidate_pool,
    )
    selections.append(selection)
  return selections


def _entry_from_body_candidate(
    assembly_id: str,
    assembly_dir: Path,
    body: FusionBodyCandidate,
) -> RetrievedPartEntry:
  text = " ".join(
      [
          body.part_name,
          body.body_name,
          body.occurrence_name or "",
          body.step_path.stem,
          assembly_id,
      ]
  )
  semantic_text = _entry_semantic_text(body, assembly_id=assembly_id)
  token_text = " ".join([text, semantic_text])
  tokens = _tokenize_text(token_text)
  concept_scores = _concept_scores_from_text(token_text)
  concepts = {
      concept
      for concept, score in concept_scores.items()
      if float(score) >= 0.45
  }
  sidecar_info = _load_sidecar_geometry_info(body.step_path)
  return RetrievedPartEntry(
      assembly_id=assembly_id,
      assembly_dir=assembly_dir,
      step_path=body.step_path,
      part_name=body.part_name,
      body_uuid=body.body_uuid,
      body_name=body.body_name,
      occurrence_name=body.occurrence_name,
      is_grounded=body.is_grounded,
      is_visible=body.is_visible,
      volume=float(body.volume),
      contact_degree=int(body.contact_degree),
      search_tokens=tokens,
      search_concepts=concepts,
      concept_scores=concept_scores,
      anchor_tokens=_spec_anchor_tokens(token_text),
      semantic_text=semantic_text,
      vlm_caption=str(sidecar_info.get("vlm_caption") or ""),
      caption_embedding_text=str(
          sidecar_info.get("caption_embedding_text") or semantic_text
      ),
      thumbnail_paths=tuple(
          str(item) for item in sidecar_info.get("thumbnail_paths", [])
      ),
      bbox_dims=sidecar_info.get("bbox_dims"),
      hole_radii=tuple(sidecar_info.get("hole_radii", [])),
      pin_radii=tuple(sidecar_info.get("pin_radii", [])),
      hole_count=int(sidecar_info.get("hole_count", 0) or 0),
      pin_count=int(sidecar_info.get("pin_count", 0) or 0),
      threaded_hole_count=int(sidecar_info.get("threaded_hole_count", 0) or 0),
      plane_count=int(sidecar_info.get("plane_count", 0) or 0),
  )


def _entry_semantic_text(body: FusionBodyCandidate, assembly_id: str) -> str:
  pieces = [
      str(body.part_name),
      str(body.body_name),
      str(body.occurrence_name or ""),
      str(body.step_path.stem),
      str(assembly_id),
  ]
  sidecars = _load_semantic_sidecars(body.step_path)
  if sidecars:
    pieces.extend(sidecars)
  return " | ".join(piece.strip() for piece in pieces if piece and piece.strip())


def _load_semantic_sidecars(step_path: Path) -> list[str]:
  candidates = [
      step_path.with_suffix(".caption.txt"),
      step_path.with_suffix(".desc.txt"),
      step_path.with_suffix(".sockets.json"),
  ]
  parts: list[str] = []
  for path in candidates:
    try:
      if not path.exists() or not path.is_file():
        continue
      if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
          kinds = payload.get("socket_kinds")
          summary = payload.get("summary")
          caption = payload.get("vlm_caption")
          caption_embedding_text = payload.get("caption_embedding_text")
          if isinstance(kinds, list):
            parts.append("socket kinds: " + ", ".join(str(item) for item in kinds[:8]))
          if isinstance(summary, str) and summary.strip():
            parts.append(summary.strip())
          if isinstance(caption, str) and caption.strip():
            parts.append(caption.strip()[:500])
          if (
              isinstance(caption_embedding_text, str)
              and caption_embedding_text.strip()
              and caption_embedding_text.strip() != str(caption or "").strip()
          ):
            parts.append(caption_embedding_text.strip()[:500])
      else:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
          parts.append(text[:500])
    except Exception:
      continue
  return parts


def _load_sidecar_geometry_info(step_path: Path) -> dict[str, Any]:
  path = step_path.with_suffix(".sockets.json")
  if not path.exists() or not path.is_file():
    return {}
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return {}
  if not isinstance(payload, dict):
    return {}
  bbox_dims = payload.get("bbox_dims")
  if isinstance(bbox_dims, list) and len(bbox_dims) >= 3:
    dims = tuple(float(value) for value in bbox_dims[:3])
  else:
    dims = None
  thumbnail_paths: list[str] = []
  raw_thumbnail_paths = payload.get("thumbnail_paths")
  if isinstance(raw_thumbnail_paths, list):
    for item in raw_thumbnail_paths:
      text = str(item or "").strip()
      if text:
        thumbnail_paths.append(text)
  def _radii_of(kind: str) -> list[float]:
    result: list[float] = []
    sockets = payload.get("sockets")
    if not isinstance(sockets, dict):
      return result
    for item in sockets.values():
      if not isinstance(item, dict):
        continue
      if str(item.get("kind")).strip().lower() != kind:
        continue
      radius = item.get("radius")
      if isinstance(radius, (int, float)):
        result.append(float(radius))
    return sorted(result)
  def _count_kind(*kinds: str) -> int:
    sockets = payload.get("sockets")
    if not isinstance(sockets, dict):
      return 0
    wanted = {str(kind).strip().lower() for kind in kinds}
    count = 0
    for item in sockets.values():
      if not isinstance(item, dict):
        continue
      if str(item.get("kind")).strip().lower() in wanted:
        count += 1
    return count
  return {
      "bbox_dims": dims,
      "hole_radii": _radii_of("hole") + _radii_of("threaded_hole"),
      "pin_radii": _radii_of("pin"),
      "hole_count": _count_kind("hole"),
      "pin_count": _count_kind("pin"),
      "threaded_hole_count": _count_kind("threaded_hole"),
      "plane_count": _count_kind("plane", "flange_plane"),
      "vlm_caption": str(payload.get("vlm_caption") or "").strip(),
      "caption_embedding_text": str(
          payload.get("caption_embedding_text") or ""
      ).strip(),
      "thumbnail_paths": thumbnail_paths,
  }


def _prompt_tokens(text: str) -> set[str]:
  raw = _tokenize_text(text)
  filtered = {token for token in raw if token not in _STOPWORDS}
  for alias in (
      "\u9f7f\u8f6e",
      "\u8f74",
      "\u5e95\u5ea7",
      "\u57fa\u5ea7",
      "\u6846\u67b6",
      "\u58f3\u4f53",
  ):
    if alias in str(text):
      filtered.add(alias)
  return filtered


def _spec_anchor_tokens(text: str) -> set[str]:
  tokens = _tokenize_text(text)
  anchors: set[str] = set()
  for token in tokens:
    if token in _STOPWORDS or len(token) < 2:
      continue
    if re.fullmatch(r"(body|part|component)\d+", token):
      continue
    if re.fullmatch(r"[0-9a-f]{8,}", token):
      continue
    has_alpha = any(char.isalpha() for char in token)
    has_digit = any(char.isdigit() for char in token)
    if re.fullmatch(r"m\d{1,3}", token):
      anchors.add(token)
      continue
    if re.fullmatch(r"\d{2,4}mm", token):
      anchors.add(token)
      continue
    if has_alpha and has_digit:
      anchors.add(token)
      continue
    if token.isdigit() and len(token) >= 4:
      anchors.add(token)
  return anchors


def _tokenize_text(text: str) -> set[str]:
  lowered = str(text).strip().lower()
  ascii_only = re.sub(r"[^a-z0-9_]+", " ", lowered)
  tokens = {
      token.strip("_")
      for token in _TOKEN_RE.findall(ascii_only)
      if token.strip("_")
  }
  extra: set[str] = set()
  for token in list(tokens):
    if re.fullmatch(r"m\d+", token):
      extra.add("bolt")
      extra.add("fastener")
    if token in {"unc", "unf"}:
      extra.add("bolt")
      extra.add("fastener")
  split_tokens: set[str] = set()
  for token in tokens:
    if "_" not in token:
      continue
    for piece in token.split("_"):
      piece = piece.strip("_")
      if piece:
        split_tokens.add(piece)
  return tokens | split_tokens | extra


def _ordered_prompt_concepts(text: str) -> list[str]:
  lowered = str(text).strip().lower()
  tokens = _tokenize_text(lowered)
  hits: list[tuple[int, str]] = []
  for concept, profile in _CONCEPT_PROFILES.items():
    pos = _first_alias_position(lowered, tokens, profile)
    if pos >= 0:
      hits.append((pos, concept))
  hits.sort(key=lambda item: item[0])
  return [concept for _, concept in hits]


def _first_alias_position(
    lowered: str,
    tokens: set[str],
    profile: dict[str, tuple[str, ...]],
) -> int:
  positions: list[int] = []
  for group_name in ("strong", "weak"):
    for alias in profile.get(group_name, ()):
      alias_lower = alias.lower()
      pos = lowered.find(alias_lower)
      if pos >= 0:
        positions.append(pos)
      elif alias_lower in tokens:
        positions.append(0)
  if not positions:
    return -1
  return min(positions)


def _concept_scores_from_text(text: str) -> dict[str, float]:
  lowered = str(text).strip().lower()
  tokens = _tokenize_text(lowered)
  scores = {concept: 0.0 for concept in _CONCEPT_PROFILES}

  for concept, profile in _CONCEPT_PROFILES.items():
    score = 0.0
    if any(_alias_matches(lowered, tokens, alias) for alias in profile["strong"]):
      score = max(score, 1.0)
    if any(_alias_matches(lowered, tokens, alias) for alias in profile["weak"]):
      score = max(score, 0.45)
    if any(_alias_matches(lowered, tokens, alias) for alias in profile["negative"]):
      score = min(score, 0.35)
    scores[concept] = score

  if (
      scores["gear"] > 0.0
      and scores["shaft"] >= 1.0
      and "gear" not in tokens
      and "gear" not in lowered
  ):
    scores["gear"] = min(scores["gear"], 0.25)

  if (
      scores["gear"] >= 1.0
      and scores["shaft"] >= 1.0
      and any(token in lowered for token in ("gear and shaft", "gear shaft"))
  ):
    scores["gear"] = min(scores["gear"], 0.55)

  if (
      scores["shaft"] > 0.0
      and "shaft" not in lowered
      and "axle" not in lowered
      and "spindle" not in lowered
      and "driveshaft" not in lowered
      and "drive shaft" not in lowered
      and any(token in lowered for token in ("arbor", "rod"))
      and any(
          token in lowered
          for token in ("frame", "base", "housing", "mount", "holder", "bracket")
      )
  ):
    scores["shaft"] = min(scores["shaft"], 0.35)

  if scores["base"] > 0.0 and "plate" in tokens and not any(
      token in lowered for token in ("base", "frame", "housing", "mount", "bracket")
  ):
    scores["base"] = min(scores["base"], 0.35)

  if scores["base"] > 0.0 and "mount" in lowered and not any(
      token in lowered for token in ("base", "frame", "housing", "bracket")
  ):
    scores["base"] = min(scores["base"], 0.75)

  if any(token in lowered for token in ("table plate", "gib plate")):
    scores["base"] = min(scores["base"], 0.25)

  if any(token in lowered for token in ("keyway", "gasket")):
    scores["gear"] = min(scores["gear"], 0.25)
    scores["base"] = min(scores["base"], 0.45)

  if (
      scores["gear"] > 0.0
      and any(
          token in lowered
          for token in ("housing", "frame", "bracket", "mount", "case", "casing")
      )
      and "shaft" not in lowered
  ):
    scores["gear"] = min(scores["gear"], 0.35)

  if (
      scores["gear"] > 0.0
      and scores["base"] > 0.0
      and "gear" in lowered
      and any(token in lowered for token in ("housing", "gasket", "cover"))
  ):
    scores["gear"] = min(scores["gear"], 0.35)

  return scores


def _alias_matches(lowered: str, tokens: set[str], alias: str) -> bool:
  alias_lower = alias.lower()
  if all(ord(char) < 128 for char in alias_lower):
    if " " in alias_lower:
      return alias_lower in lowered
    return alias_lower in tokens or alias_lower in lowered
  return alias_lower in str(lowered)


def _score_entry(
    instruction: str,
    prompt_tokens: set[str],
    prompt_concepts: list[str],
    prompt_anchor_tokens: set[str],
    entry: RetrievedPartEntry,
) -> RetrievedPartEntry:
  scored = RetrievedPartEntry(**entry.__dict__)
  score = 0.0
  reasons: list[str] = []

  for concept in prompt_concepts:
    concept_score = float(entry.concept_scores.get(concept, 0.0))
    if concept_score <= 0.0:
      continue
    score += 4.8 * concept_score
    reasons.append(f"{concept}={concept_score:.2f}")

  token_overlap = sorted(prompt_tokens & entry.search_tokens)
  if token_overlap:
    score += 0.9 * float(len(token_overlap))
    reasons.append("token_overlap=" + ",".join(token_overlap[:6]))

  if entry.vlm_caption:
    caption_tokens = _tokenize_text(entry.vlm_caption)
    caption_overlap = sorted(prompt_tokens & caption_tokens)
    if caption_overlap:
      score += min(2.4, 0.55 * float(len(caption_overlap)))
      reasons.append("caption_overlap=" + ",".join(caption_overlap[:6]))

  anchor_overlap = sorted(prompt_anchor_tokens & entry.anchor_tokens)
  if anchor_overlap:
    score += 7.0 * float(len(anchor_overlap))
    reasons.append("anchor_overlap=" + ",".join(anchor_overlap[:6]))
  elif prompt_anchor_tokens:
    score -= min(4.0, 1.0 + 0.8 * float(len(prompt_anchor_tokens)))
    reasons.append("anchor_stage1_penalty")

  for concept in prompt_concepts:
    concept_score = float(entry.concept_scores.get(concept, 0.0))
    if concept_score < 0.45:
      continue
    role_socket_score = _role_socket_score(entry, concept)
    if abs(role_socket_score) <= 1e-8:
      continue
    score += role_socket_score
    reasons.append(f"socket_prior_{concept}={role_socket_score:.2f}")

  if "base" in prompt_concepts and entry.is_grounded:
    if float(entry.concept_scores.get("base", 0.0)) >= 0.55:
      score += 1.2
      reasons.append("grounded_base_bonus")

  if "gear" in prompt_concepts and float(entry.concept_scores.get("gear", 0.0)) >= 0.85:
    score += 0.9
    reasons.append("strong_gear_bonus")

  if (
      "gear" in prompt_concepts
      and "shaft" in prompt_concepts
      and float(entry.concept_scores.get("gear", 0.0)) < 0.45
      and float(entry.concept_scores.get("shaft", 0.0)) >= 0.85
      and "pinion" in " ".join(sorted(entry.search_tokens))
  ):
    score -= 0.75
    reasons.append("compound_pinion_shaft_penalty")

  if (
      "base" in prompt_concepts
      and "plate" in entry.search_tokens
      and float(entry.concept_scores.get("base", 0.0)) < 0.55
  ):
    score -= 0.35
    reasons.append("weak_plate_base_penalty")

  if (
      "gear" in prompt_concepts
      and any(
          token in " ".join(sorted(entry.search_tokens))
          for token in ("gearbox", "gearmotor")
      )
  ):
    score -= 0.8
    reasons.append("gearbox_penalty")

  if _is_hybrid_base_gear(entry):
    score -= 0.9
    reasons.append("hybrid_base_gear_penalty")

  if _is_low_level_subpart(entry):
    score -= 0.8
    reasons.append("subpart_penalty")

  score += 0.08 * min(float(entry.contact_degree), 10.0)
  if entry.volume > 0.0:
    score += min(0.8, 0.04 * float(len(str(int(entry.volume + 1.0)))))

  if not entry.is_visible:
    score -= 0.5

  lowered = instruction.lower()
  if any(token in lowered for token in ("base", "frame", "\u5e95\u5ea7", "\u6846\u67b6")):
    if entry.is_grounded:
      score += 0.25
      reasons.append("grounded_hint")

  scored.score = float(score)
  scored.match_reasons = reasons
  return scored


def _target_part_count(
    instruction: str,
    prompt_concepts: list[str],
    part_count: int,
) -> int:
  if int(part_count) > 0:
    return max(2, int(part_count))
  enumerated_count = _enumerated_part_count(instruction)
  if enumerated_count >= 3:
    concept_floor = len(prompt_concepts) if prompt_concepts else 0
    return max(2, min(8, max(enumerated_count, concept_floor)))
  if prompt_concepts:
    return max(2, min(6, len(prompt_concepts)))
  lowered = instruction.lower()
  if "then" in lowered or "\u7136\u540e" in instruction:
    return 3
  return 2


def _enumerated_part_count(instruction: str) -> int:
  text = str(instruction or "").strip()
  if not text:
    return 0
  lowered = text.lower()
  candidate_segments: list[str] = []

  using_match = re.search(r"\busing\b(.+?)(?:\.|$)", lowered, re.IGNORECASE)
  if using_match:
    start, end = using_match.span(1)
    candidate_segments.append(text[start:end])

  parts_match = re.search(r"\bparts?\s*:\s*(.+?)(?:\.|$)", lowered, re.IGNORECASE)
  if parts_match:
    start, end = parts_match.span(1)
    candidate_segments.append(text[start:end])

  best = 0
  for segment in candidate_segments:
    cleaned = re.sub(r"\s+", " ", segment).strip(" .")
    if not cleaned:
      continue
    cleaned = re.sub(r"\b(use|connect|assemble)\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" ,.")
    if not cleaned:
      continue
    parts = [
        piece.strip(" ,.")
        for piece in re.split(r",|\band\b|\bas well as\b|/", cleaned, flags=re.IGNORECASE)
        if piece and piece.strip(" ,.")
    ]
    normalized = []
    for piece in parts:
      token_count = len(_tokenize_text(piece))
      if token_count <= 0:
        continue
      normalized.append(piece)
    best = max(best, len(normalized))
  return best


def _best_assembly_for_prompt(
    prompt_concepts: list[str],
    prompt_anchor_tokens: set[str],
    entries: list[RetrievedPartEntry],
    target_count: int,
) -> Optional[str]:
  assignments = _same_assembly_role_assignments(
      prompt_concepts=prompt_concepts,
      prompt_anchor_tokens=prompt_anchor_tokens,
      entries=entries,
      prefer_same_assembly=True,
  )
  if not assignments:
    return None
  best_assignment = assignments[0]
  coverage = len(best_assignment[0])
  required_coverage = min(len(prompt_concepts), max(2, target_count))
  if coverage < required_coverage:
    return None
  return _single_assembly_id_from_assignment(best_assignment[0])


def _select_entries(
    prompt_concepts: list[str],
    entries: list[RetrievedPartEntry],
    target_count: int,
    preferred_assembly_id: Optional[str],
) -> list[RetrievedPartEntry]:
  selected: list[RetrievedPartEntry] = []
  seen_part_names: set[str] = set()

  def _try_add(candidate: RetrievedPartEntry) -> None:
    if candidate.part_name in seen_part_names:
      return
    selected.append(candidate)
    seen_part_names.add(candidate.part_name)

  scoped_entries = list(entries)
  if preferred_assembly_id is not None:
    preferred_entries = [
        entry for entry in entries if entry.assembly_id == preferred_assembly_id
    ]
    if preferred_entries:
      scoped_entries = preferred_entries + [
          entry for entry in entries if entry.assembly_id != preferred_assembly_id
      ]

  for concept in prompt_concepts:
    selected_assemblies = {entry.assembly_id for entry in selected}
    pool = [
        entry
        for entry in scoped_entries
        if entry.part_name not in seen_part_names
        and float(entry.concept_scores.get(concept, 0.0)) >= 0.45
    ]
    if not pool:
      continue
    pool.sort(
        key=lambda item: (
            -(
                1
                if item.assembly_id in selected_assemblies and selected_assemblies
                else 0
            ),
            -float(item.concept_scores.get(concept, 0.0)),
            -float(item.score),
            -int(item.contact_degree),
        )
    )
    _try_add(pool[0])
    if len(selected) >= target_count:
      return selected[:target_count]

  fill_assembly_id = preferred_assembly_id or _modal_assembly_id(selected)
  preferred_fill = [
      entry
      for entry in entries
      if entry.part_name not in seen_part_names
      and fill_assembly_id is not None
      and entry.assembly_id == fill_assembly_id
  ]
  remaining = preferred_fill + [
      entry
      for entry in entries
      if entry.part_name not in seen_part_names and entry not in preferred_fill
  ]
  for entry in remaining:
    _try_add(entry)
    if len(selected) >= target_count:
      break
  return selected[:target_count]


def _same_assembly_role_assignments(
    prompt_concepts: list[str],
    prompt_anchor_tokens: set[str],
    entries: list[RetrievedPartEntry],
    prefer_same_assembly: bool,
) -> list[tuple[dict[str, RetrievedPartEntry], float, list[str]]]:
  if not prompt_concepts:
    return []
  by_assembly: dict[str, list[RetrievedPartEntry]] = {}
  for entry in entries:
    by_assembly.setdefault(entry.assembly_id, []).append(entry)

  ranked: list[tuple[dict[str, RetrievedPartEntry], float, list[str]]] = []
  for assembly_id, assembly_entries in by_assembly.items():
    assignment = _best_distinct_role_assignment(prompt_concepts, assembly_entries)
    if not assignment:
      continue
    role_map, score = assignment
    if len(role_map) < 2:
      continue
    notes = [
        f"retrieval_assignment_mode=same_assembly:{assembly_id}",
        "retrieval_assignment_roles="
        + ",".join(f"{concept}:{role_map[concept].part_name}" for concept in prompt_concepts if concept in role_map),
    ]
    if prefer_same_assembly:
      score += 1.5 + 0.2 * float(len(role_map))
    score += _assembly_contact_bonus(role_map, assembly_id)
    score += _dimension_compatibility_bonus(role_map)
    anchor_bonus, anchor_notes, matched = _assignment_anchor_bonus(
        role_map, prompt_anchor_tokens
    )
    if prompt_anchor_tokens and not matched:
      continue
    score += anchor_bonus
    notes.extend(anchor_notes)
    ranked.append((role_map, score, notes))
  ranked.sort(
      key=lambda item: (
          -float(item[1]),
          -len(item[0]),
          _single_assembly_id_from_assignment(item[0]) or "",
      )
  )
  return ranked


def _cross_assembly_role_assignments(
    prompt_concepts: list[str],
    prompt_anchor_tokens: set[str],
    entries: list[RetrievedPartEntry],
    per_concept_limit: int,
    max_sets: int,
) -> list[tuple[dict[str, RetrievedPartEntry], float, list[str]]]:
  if not prompt_concepts:
    return []
  pools: dict[str, list[RetrievedPartEntry]] = {}
  for concept in prompt_concepts:
    pool = [
        entry
        for entry in entries
        if float(entry.concept_scores.get(concept, 0.0)) >= 0.45
    ]
    pool.sort(
        key=lambda item: (
            -float(item.concept_scores.get(concept, 0.0)),
            -float(item.score),
            -int(item.contact_degree),
        )
    )
    pools[concept] = pool[: max(1, int(per_concept_limit))]

  ranked: list[tuple[dict[str, RetrievedPartEntry], float, list[str]]] = []

  def _search(
      idx: int,
      current: dict[str, RetrievedPartEntry],
      used_parts: set[str],
      score: float,
  ) -> None:
    if len(ranked) >= max_sets * 3:
      return
    if idx >= len(prompt_concepts):
      if len(current) >= 2:
        notes = [
            "retrieval_assignment_mode=cross_assembly",
            "retrieval_assignment_roles="
            + ",".join(
                f"{concept}:{current[concept].part_name}"
                for concept in prompt_concepts
                if concept in current
            ),
        ]
        score_local = float(score)
        if _single_assembly_id_from_assignment(current) is None:
          score_local -= 0.35 * max(0, len(current) - 1)
        score_local += _dimension_compatibility_bonus(current)
        anchor_bonus, anchor_notes, matched = _assignment_anchor_bonus(
            current, prompt_anchor_tokens
        )
        if prompt_anchor_tokens and not matched:
          return
        score_local += anchor_bonus
        notes.extend(anchor_notes)
        ranked.append((dict(current), score_local, notes))
      return

    concept = prompt_concepts[idx]
    pool = pools.get(concept, [])
    progressed = False
    for entry in pool:
      if entry.part_name in used_parts:
        continue
      progressed = True
      current[concept] = entry
      used_parts.add(entry.part_name)
      _search(
          idx + 1,
          current,
          used_parts,
          score + _entry_role_score(entry, concept),
      )
      used_parts.remove(entry.part_name)
      current.pop(concept, None)
    if not progressed:
      _search(idx + 1, current, used_parts, score)

  _search(0, {}, set(), 0.0)
  ranked.sort(
      key=lambda item: (
          -float(item[1]),
          0 if _single_assembly_id_from_assignment(item[0]) else 1,
      )
  )
  return ranked[: max_sets]


def _best_distinct_role_assignment(
    prompt_concepts: list[str],
    entries: list[RetrievedPartEntry],
) -> Optional[tuple[dict[str, RetrievedPartEntry], float]]:
  pools: dict[str, list[RetrievedPartEntry]] = {}
  for concept in prompt_concepts:
    pool = [
        entry
        for entry in entries
        if float(entry.concept_scores.get(concept, 0.0)) >= 0.45
    ]
    if not pool:
      continue
    pool.sort(
        key=lambda item: (
            -float(item.concept_scores.get(concept, 0.0)),
            -float(item.score),
            -int(item.contact_degree),
        )
    )
    pools[concept] = pool[:8]
  if not pools:
    return None

  best_assignment: Optional[dict[str, RetrievedPartEntry]] = None
  best_score = -1e9

  def _search(
      idx: int,
      current: dict[str, RetrievedPartEntry],
      used_parts: set[str],
      score: float,
  ) -> None:
    nonlocal best_assignment, best_score
    if idx >= len(prompt_concepts):
      if len(current) >= 2 and score > best_score:
        best_assignment = dict(current)
        best_score = float(score)
      return
    concept = prompt_concepts[idx]
    pool = pools.get(concept, [])
    if not pool:
      _search(idx + 1, current, used_parts, score)
      return
    for entry in pool:
      if entry.part_name in used_parts:
        continue
      current[concept] = entry
      used_parts.add(entry.part_name)
      _search(idx + 1, current, used_parts, score + _entry_role_score(entry, concept))
      used_parts.remove(entry.part_name)
      current.pop(concept, None)
    _search(idx + 1, current, used_parts, score)

  _search(0, {}, set(), 0.0)
  if best_assignment is None:
    return None
  return best_assignment, best_score


def _entry_role_score(entry: RetrievedPartEntry, concept: str) -> float:
  concept_score = float(entry.concept_scores.get(concept, 0.0))
  impurity_penalty = 0.18 * max(0, len(entry.search_concepts) - 1)
  socket_score = _role_socket_score(entry, concept)
  return (
      3.0 * concept_score
      + 0.45 * float(entry.score)
      + 0.9 * socket_score
      + 0.03 * min(float(entry.contact_degree), 10.0)
      - impurity_penalty
  )


def _ordered_entries_from_assignment(
    prompt_concepts: list[str],
    assignment: dict[str, RetrievedPartEntry],
    target_count: int,
    candidate_pool: list[RetrievedPartEntry],
    preferred_assembly_id: Optional[str],
) -> list[RetrievedPartEntry]:
  ordered: list[RetrievedPartEntry] = []
  seen: set[str] = set()
  for concept in prompt_concepts:
    entry = assignment.get(concept)
    if entry is None or entry.part_name in seen:
      continue
    ordered.append(entry)
    seen.add(entry.part_name)
  if len(ordered) >= target_count:
    return ordered[:target_count]

  supplement = [
      entry
      for entry in candidate_pool
      if entry.part_name not in seen
  ]
  supplement.sort(
      key=lambda item: (
          0 if preferred_assembly_id and item.assembly_id == preferred_assembly_id else 1,
          -int(item.contact_degree),
          -float(item.score),
          -float(item.volume),
          item.part_name,
      )
  )
  for entry in supplement:
    ordered.append(entry)
    seen.add(entry.part_name)
    if len(ordered) >= target_count:
      break
  return ordered


def _refine_role_assignments_for_entries(
    *,
    instruction: str,
    prompt_concepts: list[str],
    entries: list[RetrievedPartEntry],
    role_assignments: dict[str, str],
) -> dict[str, str]:
  if not entries or not prompt_concepts:
    return dict(role_assignments)
  prompt_anchor_tokens = _spec_anchor_tokens(instruction)
  by_name = {entry.part_name: entry for entry in entries}
  refined = dict(role_assignments)
  used_names = {
      name for name in refined.values() if isinstance(name, str) and name in by_name
  }

  def _role_signature(entry: RetrievedPartEntry, concept: str) -> tuple[float, float, float, float]:
    anchor_hits = float(len(prompt_anchor_tokens & entry.anchor_tokens))
    concept_score = float(entry.concept_scores.get(concept, 0.0))
    socket_score = _role_socket_score(entry, concept)
    lexical_bonus = 0.0
    lowered = " ".join(sorted(entry.search_tokens))
    profile = _CONCEPT_PROFILES.get(concept, {})
    for token in profile.get("strong", ()):
      if token and token in lowered:
        lexical_bonus += 1.0
        break
    return (
        anchor_hits,
        lexical_bonus,
        concept_score + 0.35 * socket_score,
        float(entry.score),
    )

  for concept in prompt_concepts:
    pool = list(entries)
    current_name = refined.get(concept)
    if current_name in by_name:
      current_entry = by_name[current_name]
      current_sig = _role_signature(current_entry, concept)
    else:
      current_entry = None
      current_sig = (-1.0, -1.0, -1.0, -1.0)

    best_entry = current_entry
    best_sig = current_sig
    for entry in pool:
      if entry.part_name in used_names and entry.part_name != current_name:
        continue
      sig = _role_signature(entry, concept)
      if sig > best_sig:
        best_entry = entry
        best_sig = sig
    if best_entry is None:
      continue
    if current_name in used_names:
      used_names.discard(current_name)
    refined[concept] = best_entry.part_name
    used_names.add(best_entry.part_name)
  return refined


def _selection_from_entries(
    instruction: str,
    prompt_tokens: set[str],
    prompt_concepts: list[str],
    entries: list[RetrievedPartEntry],
    preferred_assembly_id: Optional[str],
    index: PartRetrievalIndex,
    notes: list[str],
    semantic_score: float,
    role_assignments: dict[str, str],
    candidate_pool: list[RetrievedPartEntry],
) -> RetrievedPartSelection:
  if len(entries) < 2:
    raise ValueError("Need at least two retrieved parts to attempt assembly.")

  role_assignments = _refine_role_assignments_for_entries(
      instruction=instruction,
      prompt_concepts=prompt_concepts,
      entries=entries,
      role_assignments=role_assignments,
  )

  common_assembly = {entry.assembly_id for entry in entries}
  contact_pairs: list[tuple[str, str]] = []
  full_notes = list(notes)
  missing = [
      concept
      for concept in prompt_concepts
      if concept not in role_assignments
  ]
  if missing:
    full_notes.append("retrieval_missing_concepts=" + ",".join(missing))

  if len(common_assembly) == 1:
    assembly_id = next(iter(common_assembly))
    all_pairs = index.assembly_contact_pairs.get(assembly_id, [])
    selected_names = {entry.part_name for entry in entries}
    contact_pairs = [
        pair
        for pair in all_pairs
        if pair[0] in selected_names and pair[1] in selected_names
    ]
    full_notes.append(
        f"retrieval_selected_single_assembly={assembly_id};contact_pairs={len(contact_pairs)}"
    )
  else:
    full_notes.append("retrieval_selected_cross_assembly=true")

  return RetrievedPartSelection(
      instruction=instruction,
      selected_parts=entries,
      prompt_tokens=sorted(prompt_tokens),
      prompt_concepts=prompt_concepts,
      notes=full_notes,
      contact_pairs=contact_pairs,
      preferred_assembly_id=preferred_assembly_id,
      role_assignments=dict(role_assignments),
      semantic_score=float(semantic_score),
      candidate_pool=list(candidate_pool),
  )


def _query_embedding(
    instruction: str,
    index: PartRetrievalIndex,
) -> Optional[list[float]]:
  if not bool(index.embedding_info.get("enabled")):
    return None
  try:
    vectors = _embed_texts(
        texts=[instruction],
        model=str(index.embedding_info.get("model")),
        base_url=index.embedding_info.get("base_url"),
        api_key=None,
        timeout_seconds=int(index.embedding_info.get("timeout_seconds", 45)),
        batch_size=1,
    )
  except Exception:
    return None
  if not vectors:
    return None
  return vectors[0]


def _embed_texts(
    *,
    texts: list[str],
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    timeout_seconds: int,
    batch_size: int,
) -> list[list[float]]:
  if not texts:
    return []
  url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/embeddings"
  if api_key:
    raise ValueError("Embedding API keys must not be passed as function arguments.")
  token = (
      os.getenv("NEUROCAD_EMBEDDING_API_KEY")
      or os.getenv("OPENAI_API_KEY")
      or os.getenv("DEEPSEEK_API_KEY")
      or os.getenv("OLLAMA_API_KEY")
      or "ollama"
  )
  results: list[list[float]] = []
  batch_size = max(1, int(batch_size))
  for start in range(0, len(texts), batch_size):
    batch = texts[start : start + batch_size]
    payload = {
        "model": model,
        "input": batch,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = _post_json(
        url=url,
        payload=payload,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    data = response.get("data")
    if not isinstance(data, list):
      raise RuntimeError("embedding response missing data list")
    ordered = sorted(
        [item for item in data if isinstance(item, dict)],
        key=lambda item: int(item.get("index", 0)),
    )
    for item in ordered:
      raw = item.get("embedding")
      if not isinstance(raw, list):
        raise RuntimeError("embedding item missing vector")
      vec = [float(value) for value in raw]
      results.append(_normalize_vector(vec))
  return results


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
  raw = json.dumps(payload).encode("utf-8")
  req = url_request.Request(url=url, method="POST", data=raw, headers=headers)
  try:
    with url_request.urlopen(req, timeout=int(timeout_seconds)) as resp:
      body = resp.read().decode("utf-8")
  except url_error.HTTPError as exc:
    details = exc.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"HTTP {exc.code}: {details[:300]}") from exc
  except url_error.URLError as exc:
    raise RuntimeError(f"Embedding request failed: {exc}") from exc
  parsed = json.loads(body)
  if not isinstance(parsed, dict):
    raise RuntimeError("embedding response is not a JSON object")
  return parsed


def _normalize_vector(vector: list[float]) -> list[float]:
  norm = sum(value * value for value in vector) ** 0.5
  if norm <= 1e-12:
    return vector
  return [value / norm for value in vector]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
  size = min(len(a), len(b))
  if size <= 0:
    return 0.0
  return float(sum(a[idx] * b[idx] for idx in range(size)))


def _assembly_contact_bonus(
    role_map: dict[str, RetrievedPartEntry],
    assembly_id: str,
) -> float:
  entries = list(role_map.values())
  if not entries:
    return 0.0
  if any(entry.assembly_id != assembly_id for entry in entries):
    return 0.0
  degree_bonus = 0.06 * sum(min(entry.contact_degree, 8) for entry in entries)
  unique_entries = {entry.part_name for entry in entries}
  return float(degree_bonus + 0.12 * max(0, len(unique_entries) - 1))


def _assignment_anchor_bonus(
    role_map: dict[str, RetrievedPartEntry],
    prompt_anchor_tokens: set[str],
) -> tuple[float, list[str], set[str]]:
  if not prompt_anchor_tokens:
    return 0.0, [], set()
  matched: set[str] = set()
  for entry in role_map.values():
    matched.update(prompt_anchor_tokens & entry.anchor_tokens)
  missing = prompt_anchor_tokens - matched
  notes = [
      "retrieval_anchor_match="
      + (",".join(sorted(matched)) if matched else "none"),
  ]
  score = 4.0 * float(len(matched)) - 3.0 * float(len(missing))
  if matched and not missing:
    score += 2.0
    notes.append("retrieval_anchor_full_coverage=true")
  elif missing:
    notes.append("retrieval_anchor_missing=" + ",".join(sorted(missing)))
  return score, notes, matched


def _role_socket_score(entry: RetrievedPartEntry, concept: str) -> float:
  hole_like = int(entry.hole_count) + int(entry.threaded_hole_count)
  pin_like = int(entry.pin_count)
  plane_like = int(entry.plane_count)
  if concept == "shaft":
    if pin_like > 0:
      return 1.5
    if hole_like > 0:
      return -1.2
    return -0.4
  if concept == "bolt":
    if pin_like > 0:
      return 1.3
    if hole_like > 0:
      return -1.0
    return -0.4
  if concept == "gear":
    if hole_like > 0:
      return 0.9
    if pin_like > 0:
      return -0.75
    return -0.25
  if concept in {"base", "bearing"}:
    score = 0.0
    if hole_like > 0:
      score += 1.0
    if plane_like > 0:
      score += 0.55
    return score if score > 0.0 else -1.35
  if concept == "collar":
    if hole_like > 0:
      return 0.95
    return -0.85
  return 0.0


def _dimension_compatibility_bonus(
    role_map: dict[str, RetrievedPartEntry],
) -> float:
  bonus = 0.0
  shaft = role_map.get("shaft")
  gear = role_map.get("gear")
  base = role_map.get("base") or role_map.get("bearing")
  collar = role_map.get("collar")
  bolt = role_map.get("bolt")
  if shaft is not None and gear is not None:
    if shaft.pin_count <= 0:
      bonus -= 1.8
    if (gear.hole_count + gear.threaded_hole_count) <= 0:
      bonus -= 1.4
    bonus += _radius_fit_bonus(shaft.pin_radii, gear.hole_radii, weight=1.4)
  if shaft is not None and base is not None:
    if shaft.pin_count <= 0:
      bonus -= 1.4
    if (base.hole_count + base.threaded_hole_count) <= 0:
      bonus -= 1.8
    bonus += _radius_fit_bonus(shaft.pin_radii, base.hole_radii, weight=1.2)
  if shaft is not None and collar is not None:
    if shaft.pin_count <= 0:
      bonus -= 1.0
    if (collar.hole_count + collar.threaded_hole_count) <= 0:
      bonus -= 1.2
    bonus += _radius_fit_bonus(shaft.pin_radii, collar.hole_radii, weight=1.1)
  if bolt is not None and base is not None:
    if bolt.pin_count <= 0:
      bonus -= 1.0
    if (base.hole_count + base.threaded_hole_count) <= 0:
      bonus -= 1.4
    bonus += _radius_fit_bonus(bolt.pin_radii, base.hole_radii, weight=0.9)
  bonus += _bbox_scale_bonus(role_map)
  return float(bonus)


def _radius_fit_bonus(
    pins: tuple[float, ...],
    holes: tuple[float, ...],
    weight: float,
) -> float:
  if not pins or not holes:
    return 0.0
  best = None
  for pin_radius in pins[:8]:
    for hole_radius in holes[:8]:
      clearance = hole_radius - pin_radius
      diff = abs(clearance)
      if best is None or diff < best:
        best = diff
  if best is None:
    return 0.0
  if best <= 0.35:
    return float(weight)
  if best <= 1.0:
    return float(weight * 0.55)
  if best <= 2.5:
    return float(weight * 0.2)
  return float(-0.35 * weight)


def _bbox_scale_bonus(
    role_map: dict[str, RetrievedPartEntry],
) -> float:
  entries = [entry for entry in role_map.values() if entry.bbox_dims is not None]
  if len(entries) < 2:
    return 0.0
  ratios: list[float] = []
  for entry in entries:
    dims = sorted([abs(float(v)) for v in entry.bbox_dims or ()], reverse=True)
    if len(dims) < 1:
      continue
    ratios.append(dims[0] / max(dims[-1], 1e-6))
  if not ratios:
    return 0.0
  spread = max(ratios) - min(ratios)
  if spread <= 2.0:
    return 0.25
  if spread <= 4.5:
    return 0.1
  return -0.15


def _single_assembly_id_from_assignment(
    assignment: dict[str, RetrievedPartEntry],
) -> Optional[str]:
  assembly_ids = {entry.assembly_id for entry in assignment.values()}
  if len(assembly_ids) == 1:
    return next(iter(assembly_ids))
  return None


def _modal_assembly_id(entries: list[RetrievedPartEntry]) -> Optional[str]:
  counts: dict[str, int] = {}
  for entry in entries:
    counts[entry.assembly_id] = counts.get(entry.assembly_id, 0) + 1
  if not counts:
    return None
  return max(counts.items(), key=lambda item: item[1])[0]


def _is_hybrid_base_gear(entry: RetrievedPartEntry) -> bool:
  lowered = " ".join(sorted(entry.search_tokens))
  return (
      float(entry.concept_scores.get("gear", 0.0)) > 0.0
      and float(entry.concept_scores.get("base", 0.0)) > 0.0
      and any(
          token in lowered
          for token in ("housing", "frame", "cover", "gasket", "case", "casing")
      )
  )


def _is_low_level_subpart(entry: RetrievedPartEntry) -> bool:
  lowered = " ".join(sorted(entry.search_tokens))
  return any(
      token in lowered
      for token in ("keyway", "gasket", "washer", "retaining", "ring", "ball")
  )
