"""Batch inference: Prompt -> LLM JSON -> CadQuery assembly -> STEP export."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional

import numpy as np

from .benchmark_v2_public_artifacts import public_batch_result
from .benchmark_v2_protocol import (
    BenchmarkV2PartSpec,
    PreparedBenchmarkV2Assembly,
    prepare_benchmark_v2_assembly,
)
from .cadquery_backend import (
    build_template_from_step,
    export_assembly_step,
    transform_shape,
)
from .demo import build_collision_checker, build_extraction_config, build_planner
from .evaluation import (
    derive_validated_success,
    evaluate_generation_quality,
)
from .fusion_dataset import load_fusion_assembly_sample
from .interface_grounder import augment_template_with_scored_interfaces
from .interface_pair_scorer import InterfacePairScorer
from .interface_scorer import InterfaceScorer
from .pipeline import NeuroSymbolicCadPipeline
from .pipeline import PipelineAttempt, PipelineResult
from .physics_grounder import (
    PhysicsGuidedGroundingPlanner,
    PhysicsGroundingConfig,
)
from .paths import resolve_dataset_roots, resolve_path
from .planner import RuleBasedNeuralPlanner
from .retrieval import (
    PartRetrievalIndex,
    RetrievedPartSelection,
    _entry_from_body_candidate,
    build_part_retrieval_index,
    enable_embedding_rerank,
    retrieve_candidate_part_sets,
)
from .solver import SymbolicCadSolver
from .taxonomy import classify_failure, summarize_failure_taxonomy
from .domain_types import (
    AssemblyState,
    Constraint,
    ConstraintGraph,
    ConstraintType,
    PartTemplate,
    Transform,
)


_STEP_SUFFIXES = {".step", ".stp"}


def _parse_dataset_roots(
    dataset_root: Optional[str],
    dataset_roots: Optional[list[str]] = None,
) -> list[Path]:
  """Compatibility wrapper around the CWD-independent path contract."""
  return resolve_dataset_roots(dataset_root, dataset_roots)


def _sanitize_name(raw: str, fallback: str) -> str:
  cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.strip().lower())
  cleaned = re.sub(r"_+", "_", cleaned).strip("_")
  return cleaned or fallback


def _build_preferred_neighbor_map(
    actual_part_names: list[str],
    alias_map: dict[str, str],
    raw_contact_pairs: Any,
) -> dict[str, list[str]]:
  neighbors: dict[str, set[str]] = {
      name: set() for name in actual_part_names
  }
  if not isinstance(raw_contact_pairs, list):
    return {name: [] for name in actual_part_names}

  for item in raw_contact_pairs:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
      continue
    raw_a = str(item[0]).strip()
    raw_b = str(item[1]).strip()
    if not raw_a or not raw_b:
      continue
    part_a = alias_map.get(raw_a) or alias_map.get(_sanitize_name(raw_a, raw_a))
    part_b = alias_map.get(raw_b) or alias_map.get(_sanitize_name(raw_b, raw_b))
    if not part_a or not part_b or part_a == part_b:
      continue
    if part_a not in neighbors or part_b not in neighbors:
      continue
    neighbors[part_a].add(part_b)
    neighbors[part_b].add(part_a)

  return {
      name: sorted(linked)
      for name, linked in neighbors.items()
  }


def _translate_socket_origins(socket, offset: np.ndarray) -> None:
  socket.origin = np.asarray(socket.origin, dtype=float) + offset
  metadata = socket.metadata
  frame_variants = metadata.get("frame_variants")
  if isinstance(frame_variants, dict):
    for frame in frame_variants.values():
      if not isinstance(frame, dict):
        continue
      origin = frame.get("origin")
      if isinstance(origin, (list, tuple)):
        try:
          frame["origin"] = (
              np.asarray(origin, dtype=float).reshape(3) + offset
          ).tolist()
        except Exception:
          continue


def _canonicalize_part_asset_to_local_coordinates(asset) -> str:
  """Translate a single STEP body to a part-local coordinate frame.

  Fusion body STEP exports often retain assembly-world coordinates. If the
  solver then applies a fresh SE(3) mate transform, the part is effectively
  moved twice. This local canonicalization uses only the single part's own bbox,
  so it does not leak any pairwise or assembly pose information.
  """

  bbox_min = np.asarray(asset.template.local_bbox_min, dtype=float)
  bbox_max = np.asarray(asset.template.local_bbox_max, dtype=float)
  center = (bbox_min + bbox_max) / 2.0
  offset = -center
  if float(np.linalg.norm(offset)) <= 1e-9:
    asset.template.metadata["part_local_canonicalization"] = "already_centered"
    return "part_local_canonicalization=already_centered"

  asset.shape = transform_shape(
      asset.shape,
      Transform(rotation=np.eye(3, dtype=float), translation=offset),
  )
  asset.template.local_bbox_min = bbox_min + offset
  asset.template.local_bbox_max = bbox_max + offset
  for socket in asset.template.sockets.values():
    _translate_socket_origins(socket, offset)
  asset.template.metadata["part_local_canonicalization"] = "bbox_center"
  asset.template.metadata["canonicalization_offset"] = offset.tolist()
  return (
      "part_local_canonicalization=bbox_center;"
      f"offset={','.join(f'{float(x):.4f}' for x in offset)}"
  )


def _load_cases(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    raise ValueError(f"test_cases file not found: {path}")
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  if not isinstance(data, list):
    raise ValueError("test_cases JSON must be a list.")
  result: list[dict[str, Any]] = []
  for item in data:
    if isinstance(item, dict):
      result.append(item)
  return result


def _index_step_files(dataset_roots: list[Path]) -> dict[str, list[Path]]:
  index: dict[str, list[Path]] = {}
  for dataset_root in dataset_roots:
    for path in dataset_root.rglob("*"):
      if not path.is_file() or path.suffix.lower() not in _STEP_SUFFIXES:
        continue
      key = path.name.lower()
      index.setdefault(key, []).append(path)
  return index


def _resolve_part_path(
    part_token: str,
    dataset_roots: list[Path],
    assembly_dir: Optional[str],
    dataset_root_hint: Optional[str],
    file_index: dict[str, list[Path]],
) -> Path:
  token = part_token.strip()
  if not token:
    raise ValueError("Empty part token in test case.")

  prioritized_roots = list(dataset_roots)
  if dataset_root_hint:
    hint_path = Path(str(dataset_root_hint))
    resolved_hint = str(hint_path.resolve()) if hint_path.exists() else str(hint_path)
    prioritized = []
    remaining = []
    for root in dataset_roots:
      root_key = str(root.resolve()) if root.exists() else str(root)
      if root_key == resolved_hint:
        prioritized.append(root)
      else:
        remaining.append(root)
    if prioritized:
      prioritized_roots = prioritized + remaining

  as_path = Path(token)
  if as_path.is_absolute():
    if as_path.exists():
      return as_path
    raise ValueError(f"Absolute part path does not exist: {as_path}")

  for dataset_root in prioritized_roots:
    if assembly_dir:
      direct = dataset_root / assembly_dir / token
      if direct.exists():
        return direct

    direct = dataset_root / token
    if direct.exists():
      return direct

  matches = file_index.get(as_path.name.lower(), [])
  if assembly_dir:
    for dataset_root in prioritized_roots:
      assembly_root = (dataset_root / assembly_dir).resolve()
      scoped = [
          path for path in matches
          if str(path.resolve()).startswith(str(assembly_root))
      ]
      if scoped:
        return sorted(scoped, key=lambda p: str(p))[0]
  if matches:
    return sorted(matches, key=lambda p: str(p))[0]

  roots_text = ", ".join(str(root) for root in dataset_roots)
  raise ValueError(f"Cannot resolve part '{part_token}' under [{roots_text}]")


def _build_case_catalog(
    case: dict[str, Any],
    dataset_roots: list[Path],
    file_index: dict[str, list[Path]],
    extraction_config,
    interface_scorer: Optional[InterfaceScorer] = None,
    interface_top_k: int = 0,
    interface_min_score: float = 0.25,
    canonicalize_parts: bool = True,
    input_protocol: str = "legacy",
    frame_randomization_seed: int = 1001,
    benchmark_v2_translation_box_fraction: float = 1.0,
    protocol_audit: Optional[dict[str, Any]] = None,
    benchmark_v2_prepared_out: Optional[list[PreparedBenchmarkV2Assembly]] = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], list[str]]:
  protocol = str(input_protocol or "legacy").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("input_protocol must be 'legacy' or 'benchmark_v2'")
  parts = case.get("parts")
  if not isinstance(parts, list) or not parts:
    raise ValueError("Case missing non-empty list field: parts")
  part_tokens = [str(item).strip() for item in parts if str(item).strip()]
  if len(part_tokens) < 2:
    raise ValueError("Case must include at least 2 parts.")

  name_hints = case.get("selected_part_names")
  if not isinstance(name_hints, list):
    name_hints = case.get("part_names")
  if not isinstance(name_hints, list):
    name_hints = []
  body_uuid_hints = case.get("selected_body_uuids")
  if not isinstance(body_uuid_hints, list):
    body_uuid_hints = []

  assembly_dir = case.get("assembly_dir")
  if not isinstance(assembly_dir, str):
    assembly_dir = None
  dataset_root_hint = case.get("dataset_root_hint")
  if not isinstance(dataset_root_hint, str):
    dataset_root_hint = None

  catalog: dict[str, Any] = {}
  shape_library: dict[str, Any] = {}
  selected_part_names: list[str] = []
  selected_step_paths: list[str] = []
  notes: list[str] = []
  used_names: set[str] = set()
  part_specs: list[dict[str, Any]] = []
  alias_map: dict[str, str] = {}

  for idx, token in enumerate(part_tokens):
    step_path = _resolve_part_path(
        part_token=token,
        dataset_roots=dataset_roots,
        assembly_dir=assembly_dir,
        dataset_root_hint=dataset_root_hint,
        file_index=file_index,
    )
    hint = None
    if idx < len(name_hints) and isinstance(name_hints[idx], str):
      hint = name_hints[idx].strip()
    if hint:
      base_name = _sanitize_name(hint, fallback=f"part_{idx:03d}")
    else:
      base_name = _sanitize_name(step_path.stem, fallback=f"part_{idx:03d}")

    part_name = base_name
    suffix = 1
    while part_name in used_names:
      part_name = f"{base_name}_{suffix:02d}"
      suffix += 1
    used_names.add(part_name)
    selected_part_names.append(part_name)
    selected_step_paths.append(str(step_path.resolve()))
    part_specs.append(
        {
            "part_name": part_name,
            "raw_part_name": hint or step_path.stem,
            "step_path": step_path,
            "hint": hint,
            "body_uuid": (
                str(body_uuid_hints[idx]).strip()
                if idx < len(body_uuid_hints)
                and str(body_uuid_hints[idx]).strip()
                else step_path.stem
            ),
        }
    )
    alias_map[part_name] = part_name
    alias_map[step_path.stem] = part_name
    alias_map[_sanitize_name(step_path.stem, part_name)] = part_name
    if hint:
      alias_map[hint] = part_name
      alias_map[_sanitize_name(hint, part_name)] = part_name

  preferred_neighbors = _build_preferred_neighbor_map(
      actual_part_names=selected_part_names,
      alias_map=alias_map,
      raw_contact_pairs=case.get("contact_pairs"),
  )
  retrieval_role_map = {}
  raw_retrieval = case.get("retrieval")
  if isinstance(raw_retrieval, dict):
    raw_roles = raw_retrieval.get("role_assignments")
    if isinstance(raw_roles, dict):
      for role_name, part_name in raw_roles.items():
        if isinstance(role_name, str) and isinstance(part_name, str):
          retrieval_role_map[part_name] = role_name
  benchmark_role_map = {}
  raw_benchmark = case.get("benchmark_metadata")
  if isinstance(raw_benchmark, dict):
    raw_roles = raw_benchmark.get("role_assignments")
    if isinstance(raw_roles, dict):
      for role_name, part_name in raw_roles.items():
        if isinstance(role_name, str) and isinstance(part_name, str):
          benchmark_role_map[part_name] = role_name

  if protocol == "benchmark_v2":
    protocol_specs = [
        BenchmarkV2PartSpec(
            part_key=str(spec["body_uuid"]),
            part_name=str(spec["raw_part_name"]),
            body_uuid=str(spec["body_uuid"]),
            step_path=Path(spec["step_path"]),
        )
        for spec in part_specs
    ]
    prepared = prepare_benchmark_v2_assembly(
        protocol_specs,
        assembly_nonce=str(case.get("id") or ""),
        seed=int(frame_randomization_seed),
        extraction_config=extraction_config,
        translation_box_fraction=float(benchmark_v2_translation_box_fraction),
        interface_scorer=interface_scorer,
        interface_top_k=int(interface_top_k),
        interface_min_score=float(interface_min_score),
    )
    if protocol_audit is not None:
      protocol_audit.clear()
      protocol_audit.update(prepared.audit_provenance())
    if benchmark_v2_prepared_out is not None:
      benchmark_v2_prepared_out.clear()
      benchmark_v2_prepared_out.append(prepared)
    notes.append("input_protocol=benchmark_v2")
    ordered_opaque_parts = [
        prepared.private_identity_context.opaque_part_for(spec.part_key)
        for spec in protocol_specs
    ]
    spec_by_opaque = dict(zip(ordered_opaque_parts, protocol_specs))
    opaque_step_paths: list[str] = []
    # Preserve the case input order at this private adapter boundary.  The
    # downstream alias builder zips selected raw names/body UUIDs/paths with
    # this list.  Sorting opaque HMAC ranks here would silently attach raw
    # contacts and anchors to another physical part.
    for opaque_part in ordered_opaque_parts:
      spec = spec_by_opaque[opaque_part]
      part = prepared.part_for(opaque_part)
      catalog[opaque_part] = part.template
      shape_library[opaque_part] = part.shape
      opaque_step_paths.append(str(spec.step_path.resolve()))
      notes.extend(part.grounding_notes)
      notes.append(f"part_loaded={opaque_part};protocol=benchmark_v2")
    return (
        catalog,
        shape_library,
        ordered_opaque_parts,
        opaque_step_paths,
        notes,
    )

  for spec in part_specs:
    part_name = str(spec["part_name"])
    step_path = Path(spec["step_path"])
    asset = build_template_from_step(
        name=part_name,
        step_path=step_path,
        extraction_config=extraction_config,
        metadata={
            "assembly_dir": assembly_dir,
            "test_case_id": case.get("id"),
            "source_step_name": step_path.name,
            "preferred_neighbors": preferred_neighbors.get(part_name, []),
            "retrieval_role_hint": retrieval_role_map.get(part_name),
            "benchmark_role_hint": benchmark_role_map.get(part_name),
        },
    )
    if interface_scorer is not None:
      body_uuid = str(spec.get("body_uuid") or step_path.stem)
      try:
        notes.extend(
            augment_template_with_scored_interfaces(
                template=asset.template,
                step_path=step_path,
                scorer=interface_scorer,
                part_name=part_name,
                body_uuid=body_uuid,
                top_k=int(interface_top_k),
                min_score=float(interface_min_score),
            )
        )
      except Exception as exc:  # pylint: disable=broad-except
        notes.append(f"learned_interfaces_failed={part_name}:{type(exc).__name__}:{exc}")
    if canonicalize_parts:
      try:
        notes.append(
            f"{part_name}:"
            + _canonicalize_part_asset_to_local_coordinates(asset)
        )
      except Exception as exc:  # pylint: disable=broad-except
        notes.append(
            f"part_local_canonicalization_failed={part_name}:{type(exc).__name__}:{exc}"
        )
    catalog[part_name] = asset.template
    shape_library[part_name] = asset.shape
    neighbors = preferred_neighbors.get(part_name, [])
    if neighbors:
      notes.append(
          f"part_loaded={part_name}:{step_path.name};preferred={','.join(neighbors)}"
      )
    else:
      notes.append(f"part_loaded={part_name}:{step_path.name}")

  return catalog, shape_library, selected_part_names, selected_step_paths, notes


def _collision_volume_from_payload(payload: dict[str, Any]) -> float:
  attempts = payload.get("attempts") or []
  total = 0.0
  for attempt in attempts:
    if not isinstance(attempt, dict):
      continue
    for message in attempt.get("feedback_messages", []) or []:
      text = str(message)
      marker = "volume="
      if marker not in text:
        continue
      raw = text.split(marker, 1)[1].strip()
      raw = raw.split(" ", 1)[0].strip()
      raw = raw.replace(",", "")
      try:
        total += float(raw)
      except ValueError:
        continue
  return float(total)


def _final_feedback_collision_volume(payload: dict[str, Any]) -> float:
  total = 0.0
  for message in payload.get("final_feedback", []) or []:
    text = str(message)
    marker = "volume="
    if marker not in text:
      continue
    raw = text.split(marker, 1)[1].strip()
    raw = raw.split(" ", 1)[0].strip().replace(",", "")
    try:
      total += float(raw)
    except ValueError:
      continue
  return float(total)


def _attempt_count(payload: dict[str, Any]) -> int:
  attempts = payload.get("attempts")
  if isinstance(attempts, list):
    return len(attempts)
  return 0


def _tolerated_collision_volume_from_payload(payload: dict[str, Any]) -> float:
  total = 0.0
  final_tolerated = payload.get("final_tolerated_collisions")
  if isinstance(final_tolerated, list):
    for item in final_tolerated:
      if not isinstance(item, dict):
        continue
      volume = item.get("volume")
      if isinstance(volume, (int, float)):
        total += float(volume)
  if total > 0.0:
    return float(total)

  attempts = payload.get("attempts") or []
  for attempt in attempts:
    if not isinstance(attempt, dict):
      continue
    for text in attempt.get("tolerated_collision_messages", []) or []:
      message = str(text)
      if "volume=" not in message:
        continue
      token = message.split("volume=", 1)[1].strip()
      token = token.split(" ", 1)[0].replace(",", "")
      try:
        total += float(token)
      except ValueError:
        continue
  return float(total)


def _tolerated_collision_count_from_payload(payload: dict[str, Any]) -> int:
  final_tolerated = payload.get("final_tolerated_collisions")
  if isinstance(final_tolerated, list):
    return len(
        [
            item
            for item in final_tolerated
            if isinstance(item, dict)
            and not _is_intentional_interference_collision(item)
        ]
    )
  attempts = payload.get("attempts") or []
  total = 0
  for attempt in attempts:
    if not isinstance(attempt, dict):
      continue
    total += len(attempt.get("tolerated_collision_messages", []) or [])
  return total


def _is_intentional_interference_collision(item: dict[str, Any]) -> bool:
  metadata = item.get("metadata")
  if not isinstance(metadata, dict):
    return False
  return bool(metadata.get("intentional_interference")) or str(
      metadata.get("predicted_contact_type") or ""
  ) in {"threaded_interference", "screw_in_hole"}


def _strict_success_from_payload(payload: dict[str, Any]) -> bool:
  if not bool(payload.get("success", False)):
    return False
  return _tolerated_collision_count_from_payload(payload) == 0


def _source_pose_socket_name(template: PartTemplate) -> str:
  if "base_plane" in template.sockets:
    return "base_plane"
  for preferred_kind in ("plane", "hole", "pin", "axis"):
    for name, socket in template.sockets.items():
      if str(socket.kind).lower() == preferred_kind:
        return name
  if template.sockets:
    return next(iter(template.sockets))
  raise ValueError(f"Part template '{template.name}' has no sockets.")


def _source_pose_anchor(
    case: dict[str, Any],
    selected_parts: list[str],
) -> str:
  meta = case.get("benchmark_metadata")
  if isinstance(meta, dict):
    roles = meta.get("role_assignments")
    if isinstance(roles, dict):
      for role in ("base", "housing", "frame", "anchor", "shaft"):
        value = roles.get(role)
        if isinstance(value, str) and value in selected_parts:
          return value
    anchor = meta.get("anchor_part")
    if isinstance(anchor, str) and anchor in selected_parts:
      return anchor
  return selected_parts[0]


def _source_pose_contact_pairs(
    case: dict[str, Any],
    selected_parts: list[str],
) -> list[tuple[str, str]]:
  selected = set(selected_parts)
  pairs: list[tuple[str, str]] = []
  seen: set[tuple[str, str]] = set()
  for item in case.get("contact_pairs") or []:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
      continue
    part_a = str(item[0]).strip()
    part_b = str(item[1]).strip()
    if part_a not in selected or part_b not in selected or part_a == part_b:
      continue
    key = tuple(sorted((part_a, part_b)))
    if key in seen:
      continue
    seen.add(key)
    pairs.append((part_a, part_b))
  if pairs:
    return pairs
  return [
      (selected_parts[idx], selected_parts[idx + 1])
      for idx in range(max(0, len(selected_parts) - 1))
  ]


def _run_source_pose_result(
    *,
    instruction: str,
    case: dict[str, Any],
    catalog: dict[str, PartTemplate],
    selected_parts: list[str],
) -> PipelineResult:
  instances = {}
  for part_id in selected_parts:
    template = catalog.get(part_id)
    if template is None:
      raise ValueError(f"Cannot preserve source pose; missing part '{part_id}'.")
    instance = template.instantiate(part_id)
    instance.transform = Transform.identity()
    instances[part_id] = instance

  anchor = _source_pose_anchor(case, selected_parts)
  constraints: list[Constraint] = []
  for part_a, part_b in _source_pose_contact_pairs(case, selected_parts):
    template_a = catalog.get(part_a)
    template_b = catalog.get(part_b)
    if template_a is None or template_b is None:
      continue
    constraints.append(
        Constraint(
            ctype=ConstraintType.COINCIDENT,
            part_a=part_a,
            socket_a=_source_pose_socket_name(template_a),
            part_b=part_b,
            socket_b=_source_pose_socket_name(template_b),
            label=f"source_pose_contact_{part_a}_{part_b}",
            metadata={
                "relation_hint": "source_pose_contact",
                "alignment_mode": "source_pose",
                "source_pose_preserve": True,
            },
        )
    )
  graph = ConstraintGraph(
      instances=instances,
      constraints=constraints,
      anchor=anchor,
      metadata={
          "source_pose_preserve": True,
          "planner_notes": [
              "planner=source_pose_oracle",
              "source_pose_preserve=true",
              f"source_pose_contact_edges={len(constraints)}",
          ],
      },
  )
  assembly = AssemblyState(
      instances=instances,
      constraint_graph=graph,
      logs=[
          "source_pose_preserve: kept per-body STEP coordinates unchanged",
          "source_pose_preserve: intended for given-part/oracle sanity checks",
      ],
  )
  return PipelineResult(
      success=True,
      instruction=instruction,
      attempts=[
          PipelineAttempt(
              iteration=0,
              planner_notes=list(graph.metadata.get("planner_notes", [])),
              solver_logs=list(assembly.logs),
              feedback_messages=[],
              tolerated_collision_messages=[],
          )
      ],
      final_graph=graph,
      final_assembly=assembly,
      final_feedback=[],
      final_tolerated_collisions=[],
  )


def _evaluate_contact_pairs(
    result_payload: dict[str, Any],
    expected_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
  expected = {tuple(sorted(pair)) for pair in expected_pairs}
  predicted: set[tuple[str, str]] = set()
  final_graph = result_payload.get("final_graph") or {}
  for constraint in final_graph.get("constraints", []) or []:
    if not isinstance(constraint, dict):
      continue
    part_a = constraint.get("part_a")
    part_b = constraint.get("part_b")
    if not isinstance(part_a, str) or not isinstance(part_b, str):
      continue
    if part_a == part_b:
      continue
    predicted.add(tuple(sorted((part_a, part_b))))

  matched = expected & predicted
  precision = None if not predicted else len(matched) / len(predicted)
  recall = None if not expected else len(matched) / len(expected)
  f1 = None
  if precision is not None and recall is not None and precision + recall > 0.0:
    f1 = 2.0 * precision * recall / (precision + recall)
  return {
      "expected_contact_pairs": len(expected),
      "predicted_pairs": len(predicted),
      "matched_pairs": len(matched),
      "contact_precision": (
          None if precision is None else round(float(precision), 4)
      ),
      "contact_recall": None if recall is None else round(float(recall), 4),
      "contact_f1": None if f1 is None else round(float(f1), 4),
  }


def _evaluate_exact_contact_distances(
    *,
    assembly: Optional[AssemblyState],
    shape_library: dict[str, Any],
    expected_pairs: list[tuple[str, str]],
    distance_tolerance: float,
    min_contact_recall: float,
) -> dict[str, Any]:
  """Use exact OCC shape distance to catch visually scattered false positives."""

  if assembly is None:
    return {"available": False, "valid": True, "reason": "missing_assembly"}

  expected: list[tuple[str, str]] = []
  seen: set[tuple[str, str]] = set()
  for raw_a, raw_b in expected_pairs:
    part_a = str(raw_a).strip()
    part_b = str(raw_b).strip()
    if not part_a or not part_b or part_a == part_b:
      continue
    key = tuple(sorted((part_a, part_b)))
    if key in seen:
      continue
    seen.add(key)
    expected.append((part_a, part_b))
  if not expected:
    return {"available": False, "valid": True, "reason": "no_expected_pairs"}

  world_shapes: dict[str, Any] = {}

  def _world_shape(part_id: str) -> Any:
    if part_id in world_shapes:
      return world_shapes[part_id]
    instance = assembly.instances.get(part_id)
    if instance is None or instance.transform is None:
      raise ValueError(f"missing solved instance '{part_id}'")
    base_shape = shape_library.get(instance.template_name)
    if base_shape is None:
      base_shape = shape_library.get(instance.instance_id)
    if base_shape is None:
      raise ValueError(f"missing shape for '{part_id}'")
    world_shapes[part_id] = transform_shape(base_shape, instance.transform)
    return world_shapes[part_id]

  tolerance = max(0.0, float(distance_tolerance))
  records: list[dict[str, Any]] = []
  errors: list[dict[str, Any]] = []
  distances: list[float] = []
  within = 0
  for part_a, part_b in expected:
    try:
      distance = float(_world_shape(part_a).distance(_world_shape(part_b)))
      distances.append(distance)
      ok = distance <= tolerance
      if ok:
        within += 1
      records.append(
          {
              "part_a": part_a,
              "part_b": part_b,
              "distance": round(distance, 6),
              "within_tolerance": bool(ok),
          }
      )
    except Exception as exc:  # pylint: disable=broad-except
      errors.append(
          {
              "part_a": part_a,
              "part_b": part_b,
              "error": f"{type(exc).__name__}:{exc}",
          }
      )

  exact_contact_recall = float(within) / float(max(1, len(expected)))
  valid = (
      bool(records)
      and not errors
      and exact_contact_recall >= float(min_contact_recall)
  )
  return {
      "available": True,
      "valid": bool(valid),
      "checked_pairs": len(records),
      "expected_pairs": len(expected),
      "distance_tolerance": round(tolerance, 6),
      "min_contact_recall": round(float(min_contact_recall), 4),
      "exact_contact_recall": round(exact_contact_recall, 4),
      "max_contact_distance": None if not distances else round(max(distances), 6),
      "mean_contact_distance": (
          None if not distances else round(sum(distances) / len(distances), 6)
      ),
      "violations": [
          item for item in records if not bool(item.get("within_tolerance"))
      ][:20],
      "errors": errors[:20],
  }


def _evaluate_retrieval_selection(
    case: dict[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
  retrieval = result_payload.get("retrieval")
  if not isinstance(retrieval, dict):
    return {}

  selected_parts = retrieval.get("selected_parts")
  if not isinstance(selected_parts, list) or not selected_parts:
    return {}

  reference_uuids = {
      str(item).strip()
      for item in (case.get("selected_body_uuids") or [])
      if str(item).strip()
  }
  reference_names = {
      str(item).strip()
      for item in (case.get("selected_part_names") or [])
      if str(item).strip()
  }
  if not reference_uuids and not reference_names:
    return {}

  retrieved_uuids = {
      str(item.get("body_uuid")).strip()
      for item in selected_parts
      if isinstance(item, dict) and str(item.get("body_uuid")).strip()
  }
  retrieved_names = {
      str(item.get("part_name")).strip()
      for item in selected_parts
      if isinstance(item, dict) and str(item.get("part_name")).strip()
  }
  retrieved_assembly_ids = {
      str(item.get("assembly_id")).strip()
      for item in selected_parts
      if isinstance(item, dict) and str(item.get("assembly_id")).strip()
  }
  reference_assembly = str(case.get("assembly_dir") or "").strip()

  matched_uuid_count = len(reference_uuids & retrieved_uuids)
  matched_name_count = len(reference_names & retrieved_names)
  reference_count = max(len(reference_uuids), len(reference_names))
  retrieved_count = max(len(retrieved_uuids), len(retrieved_names))

  recall = (
      None
      if reference_count <= 0
      else float(max(matched_uuid_count, matched_name_count)) / float(reference_count)
  )
  precision = (
      None
      if retrieved_count <= 0
      else float(max(matched_uuid_count, matched_name_count)) / float(retrieved_count)
  )
  f1 = None
  if precision is not None and recall is not None and precision + recall > 0.0:
    f1 = 2.0 * precision * recall / (precision + recall)
  exact_match = False
  if reference_uuids:
    exact_match = reference_uuids == retrieved_uuids
  elif reference_names:
    exact_match = reference_names == retrieved_names

  assembly_match = None
  if reference_assembly:
    assembly_match = retrieved_assembly_ids == {reference_assembly}

  return {
      "reference_part_count": reference_count,
      "retrieved_part_count": retrieved_count,
      "matched_body_uuid_count": matched_uuid_count,
      "matched_part_name_count": matched_name_count,
      "retrieval_precision": (
          None if precision is None else round(float(precision), 4)
      ),
      "retrieval_recall": None if recall is None else round(float(recall), 4),
      "retrieval_f1": None if f1 is None else round(float(f1), 4),
      "exact_part_set_match": bool(exact_match),
      "same_reference_assembly": assembly_match,
  }


def _selected_part_set_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
  retrieval = payload.get("retrieval")
  if not isinstance(retrieval, dict):
    return ()
  selected = retrieval.get("selected_parts")
  if not isinstance(selected, list):
    return ()
  names: list[str] = []
  for item in selected:
    if not isinstance(item, dict):
      continue
    name = str(item.get("part_name") or "").strip()
    if name:
      names.append(name)
  if not names:
    return ()
  return tuple(sorted(set(names)))


def _case_needs_retrieval(
    case: dict[str, Any],
    *,
    input_protocol: str = "legacy",
) -> bool:
  protocol = str(input_protocol or "legacy").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("input_protocol must be 'legacy' or 'benchmark_v2'")
  if protocol == "benchmark_v2" and _is_fixed_candidate_tray_case(case):
    return False
  if _case_task_mode(case) == "retrieval":
    return True
  parts = case.get("parts")
  if not isinstance(parts, list):
    return True
  return len([item for item in parts if str(item).strip()]) == 0


def _is_fixed_candidate_tray_case(case: dict[str, Any]) -> bool:
  candidate_parts = case.get("candidate_parts")
  if not isinstance(candidate_parts, list):
    return False
  tokens = [str(item).strip() for item in candidate_parts if str(item).strip()]
  if len(tokens) < 2:
    return False
  mode = _case_task_mode(case)
  return mode in {"candidate_tray", "fixed_candidate_tray"} or mode.startswith(
      "candidate_tray_"
  )


def _materialize_fixed_candidate_tray_case(
    case: dict[str, Any],
) -> dict[str, Any]:
  """Create the non-oracle execution view of an immutable candidate tray.

  Candidate names, UUIDs, paths, gold part labels, contact pairs, roles, and
  reference metadata remain outside the model boundary. An empty ``parts``
  list exposes the complete fixed tray. A preselected subset is rejected
  because, without a benchmark-v2 selector attestation, it is indistinguishable
  from an oracle part set. Identities are always taken from the candidate
  arrays, never from gold ``selected_*`` fields.
  """

  if not _is_fixed_candidate_tray_case(case):
    raise ValueError("Case is not a valid fixed candidate tray")
  candidate_parts = [str(item).strip() for item in case["candidate_parts"]]
  candidate_names = case.get("candidate_part_names")
  candidate_uuids = case.get("candidate_body_uuids")
  if not isinstance(candidate_names, list) or not isinstance(candidate_uuids, list):
    raise ValueError(
        "fixed candidate tray requires candidate_part_names and "
        "candidate_body_uuids"
    )
  names = [str(item).strip() for item in candidate_names]
  uuids = [str(item).strip() for item in candidate_uuids]
  count = len(candidate_parts)
  if len(names) != count or len(uuids) != count:
    raise ValueError("fixed candidate tray identity arrays must have equal lengths")
  if any(not value for value in candidate_parts + names + uuids):
    raise ValueError("fixed candidate tray identities must be nonempty")
  if (
      len(set(candidate_parts)) != count
      or len(set(names)) != count
      or len(set(uuids)) != count
  ):
    raise ValueError("fixed candidate tray identities must be unique")

  selected_tokens = [
      str(item).strip()
      for item in (case.get("parts") or [])
      if str(item).strip()
  ]
  if not selected_tokens:
    selected_tokens = list(candidate_parts)
  elif selected_tokens != candidate_parts:
    raise ValueError(
        "benchmark_v2 fixed candidate tray rejects a preselected/oracle subset"
    )
  candidate_index = {token: index for index, token in enumerate(candidate_parts)}
  if len(set(selected_tokens)) != len(selected_tokens):
    raise ValueError("fixed candidate tray selected parts must be unique")
  if any(token not in candidate_index for token in selected_tokens):
    raise ValueError("fixed candidate tray selected parts must be a tray subset")
  selected_indices = [candidate_index[token] for token in selected_tokens]

  tray_payload = [
      {
          "step_file": Path(candidate_parts[index]).name,
          "name": names[index],
          "body_uuid": uuids[index],
      }
      for index in range(count)
  ]
  tray_sha256 = hashlib.sha256(
      json.dumps(
          tray_payload,
          sort_keys=True,
          separators=(",", ":"),
          ensure_ascii=True,
      ).encode("utf-8")
  ).hexdigest()
  materialized: dict[str, Any] = {
      "id": str(case.get("id") or ""),
      "instruction": str(case.get("instruction") or ""),
      "parts": selected_tokens,
      "selected_part_names": [names[index] for index in selected_indices],
      "selected_body_uuids": [uuids[index] for index in selected_indices],
  }
  for location_key in ("assembly_dir", "dataset_root_hint"):
    location_value = case.get(location_key)
    if isinstance(location_value, str) and location_value.strip():
      materialized[location_key] = location_value
  materialized["benchmark_metadata"] = {
      "case_mode": "fixed_candidate_tray",
      "candidate_tray_protocol": "benchmark_v2_fixed_tray_v1",
      "candidate_tray_part_count": count,
      "candidate_tray_sha256": tray_sha256,
  }
  return materialized


def require_safe_retrieval_protocol(
    *,
    input_protocol: str,
    retrieval_context_protocol: str,
    case_needs_retrieval: bool,
    fixed_candidate_tray: bool = False,
) -> str:
  """Fail closed before role/contact/reference-assisted retrieval can run."""

  protocol = str(input_protocol or "legacy").strip().lower()
  context = str(retrieval_context_protocol or "fixed_tray").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("input_protocol must be 'legacy' or 'benchmark_v2'")
  if context not in {"fixed_tray", "context_assisted_dev"}:
    raise ValueError(
        "retrieval_context_protocol must be fixed_tray or context_assisted_dev"
    )
  if protocol == "benchmark_v2":
    if fixed_candidate_tray and case_needs_retrieval:
      raise ValueError(
          "benchmark_v2 fixed candidate tray cannot also request retrieval"
      )
    if context != "fixed_tray":
      raise ValueError(
          "benchmark_v2 permits only an immutable fixed candidate tray; "
          "context-assisted retrieval/oracle context is forbidden"
      )
    if case_needs_retrieval:
      raise ValueError(
          "benchmark_v2 requires an explicit fixed candidate tray; "
          "contact/role/reference-assisted retrieval is forbidden"
      )
    return "fixed_tray"
  if case_needs_retrieval and context != "context_assisted_dev":
    raise ValueError(
        "context-assisted retrieval is development-only and requires "
        "--retrieval_context_protocol context_assisted_dev"
    )
  return context


def _case_task_mode(case: dict[str, Any]) -> str:
  metadata = case.get("benchmark_metadata")
  if isinstance(metadata, dict):
    mode = str(metadata.get("case_mode") or "").strip().lower()
    if mode:
      return mode
  if isinstance(case.get("candidate_parts"), list):
    return "candidate_tray"
  parts = case.get("parts")
  if isinstance(parts, list) and any(str(item).strip() for item in parts):
    return "oracle"
  return "retrieval"


def _reference_part_count_for_case(case: Optional[dict[str, Any]]) -> int:
  if not isinstance(case, dict):
    return 0
  uuid_count = len(
      [
          item
          for item in (case.get("selected_body_uuids") or [])
          if str(item).strip()
      ]
  )
  name_count = len(
      [
          item
          for item in (case.get("selected_part_names") or [])
          if str(item).strip()
      ]
  )
  meta = case.get("benchmark_metadata")
  meta_count = 0
  if isinstance(meta, dict):
    for key in ("candidate_tray_gold_part_count", "part_count"):
      value = meta.get(key)
      if isinstance(value, int):
        meta_count = max(meta_count, value)
      elif isinstance(value, float):
        meta_count = max(meta_count, int(value))
  return max(uuid_count, name_count, meta_count)


def _candidate_tray_index_for_case(
    *,
    case: dict[str, Any],
    retrieval_index: PartRetrievalIndex,
    dataset_roots: list[Path],
    file_index: dict[str, list[Path]],
) -> PartRetrievalIndex:
  candidate_parts = case.get("candidate_parts")
  if not isinstance(candidate_parts, list) or not candidate_parts:
    raise ValueError("candidate_tray case is missing candidate_parts")
  assembly_dir = case.get("assembly_dir")
  if not isinstance(assembly_dir, str):
    assembly_dir = None
  dataset_root_hint = case.get("dataset_root_hint")
  if not isinstance(dataset_root_hint, str):
    dataset_root_hint = None

  allowed_paths: set[str] = set()
  for token in candidate_parts:
    resolved = _resolve_part_path(
        part_token=str(token),
        dataset_roots=dataset_roots,
        assembly_dir=assembly_dir,
        dataset_root_hint=dataset_root_hint,
        file_index=file_index,
    )
    allowed_paths.add(str(resolved.resolve()))

  entries = [
      entry
      for entry in retrieval_index.entries
      if str(entry.step_path.resolve()) in allowed_paths
  ]
  if len(entries) < 2 and assembly_dir:
    local_root: Optional[Path] = None
    if dataset_root_hint:
      hint_path = Path(dataset_root_hint)
      if hint_path.exists():
        local_root = hint_path
    if local_root is None:
      for root in dataset_roots:
        assembly_path = root / assembly_dir
        if assembly_path.exists():
          local_root = root
          break
    if local_root is not None:
      try:
        sample = load_fusion_assembly_sample(
            assembly_dir=(local_root / assembly_dir),
            max_parts=max(16, len(candidate_parts) + 4),
            body_selection_mode="contact_degree",
        )
        existing_paths = {str(entry.step_path.resolve()) for entry in entries}
        for body in sample.body_candidates:
          body_path = str(body.step_path.resolve())
          if body_path not in allowed_paths or body_path in existing_paths:
            continue
          entries.append(
              _entry_from_body_candidate(
                  assembly_id=sample.assembly_id,
                  assembly_dir=sample.assembly_dir,
                  body=body,
              )
          )
          existing_paths.add(body_path)
      except Exception:
        pass
  if len(entries) < 2:
    raise ValueError("candidate_tray retrieval pool has fewer than two indexed parts")

  contact_pairs: dict[str, list[tuple[str, str]]] = {}
  if assembly_dir:
    all_pairs = case.get("contact_pairs")
    if isinstance(all_pairs, list):
      filtered_pairs: list[tuple[str, str]] = []
      entry_names = {entry.part_name for entry in entries}
      for item in all_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
          continue
        part_a = str(item[0]).strip()
        part_b = str(item[1]).strip()
        if part_a in entry_names and part_b in entry_names and part_a != part_b:
          filtered_pairs.append((part_a, part_b))
      if filtered_pairs:
        shared_assembly = entries[0].assembly_id
        contact_pairs[shared_assembly] = filtered_pairs

  tray_index = PartRetrievalIndex(
      dataset_root=retrieval_index.dataset_root,
      dataset_roots=list(retrieval_index.dataset_roots),
      entries=list(entries),
      assembly_contact_pairs=contact_pairs or dict(retrieval_index.assembly_contact_pairs),
      embedding_info=dict(retrieval_index.embedding_info),
  )
  return tray_index


def _retrieval_anchor_part(selection: RetrievedPartSelection) -> str:
  roles = selection.role_assignments
  for concept in ("base", "bearing", "shaft", "gear", "collar", "bolt"):
    part_name = roles.get(concept)
    if part_name:
      return part_name
  return selection.selected_parts[0].part_name


def _retrieval_relation_plan(
    selection: RetrievedPartSelection,
) -> list[tuple[str, str, str, str]]:
  roles = selection.role_assignments
  entry_map = {entry.part_name: entry for entry in selection.selected_parts}
  selected_names = [entry.part_name for entry in selection.selected_parts]
  edges: list[tuple[str, str, str, str]] = []
  seen_pairs: set[tuple[str, str]] = set()

  def _try_add(parent: Optional[str], child: Optional[str], relation: str, label: str) -> None:
    if not parent or not child or parent == child:
      return
    key = (parent, child)
    if key in seen_pairs:
      return
    seen_pairs.add(key)
    edges.append((parent, child, relation, label))

  anchor = _retrieval_anchor_part(selection)
  _try_add(roles.get("base"), roles.get("shaft"), "coaxial", "base_to_shaft")
  _try_add(roles.get("base"), roles.get("bearing"), "coaxial", "base_to_bearing")
  _try_add(roles.get("bearing"), roles.get("shaft"), "coaxial", "bearing_to_shaft")
  _try_add(roles.get("shaft"), roles.get("gear"), "coaxial", "shaft_to_gear")
  _try_add(roles.get("shaft"), roles.get("collar"), "coaxial", "shaft_to_collar")
  _try_add(roles.get("base"), roles.get("bolt"), "fasten", "base_to_bolt")
  _try_add(roles.get("bolt"), roles.get("collar"), "fasten", "bolt_to_collar")

  connected = {anchor}
  for parent, child, _, _ in edges:
    if parent in connected:
      connected.add(child)

  contact_pairs = []
  for item in selection.contact_pairs:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
      continue
    a = str(item[0]).strip()
    b = str(item[1]).strip()
    if a in entry_map and b in entry_map and a != b:
      contact_pairs.append((a, b))

  progressed = True
  while progressed:
    progressed = False
    for part_a, part_b in contact_pairs:
      if part_a in connected and part_b not in connected:
        relation = _infer_retrieval_relation(
            entry_map.get(part_a), entry_map.get(part_b)
        )
        _try_add(part_a, part_b, relation, f"contact_{part_a}_to_{part_b}")
        connected.add(part_b)
        progressed = True
      elif part_b in connected and part_a not in connected:
        relation = _infer_retrieval_relation(
            entry_map.get(part_b), entry_map.get(part_a)
        )
        _try_add(part_b, part_a, relation, f"contact_{part_b}_to_{part_a}")
        connected.add(part_a)
        progressed = True

  for part_name in selected_names:
    if part_name == anchor or part_name in connected:
      continue
    parent = roles.get("shaft") or roles.get("base") or anchor
    relation = _infer_retrieval_relation(entry_map.get(parent), entry_map.get(part_name))
    _try_add(parent, part_name, relation, f"fallback_to_{part_name}")
    connected.add(part_name)

  return edges


def _infer_retrieval_relation(
    parent_entry,
    child_entry,
) -> str:
  parent_tokens = set() if parent_entry is None else set(parent_entry.search_tokens)
  child_tokens = set() if child_entry is None else set(child_entry.search_tokens)
  parent_concepts = (
      set() if parent_entry is None else set(parent_entry.search_concepts)
  )
  child_concepts = (
      set() if child_entry is None else set(child_entry.search_concepts)
  )
  all_concepts = parent_concepts | child_concepts
  all_tokens = parent_tokens | child_tokens
  if "bolt" in all_concepts or "thread" in all_tokens or "screw" in all_tokens:
    return "fasten"
  if (
      {"shaft", "gear"} <= all_concepts
      or {"shaft", "collar"} <= all_concepts
      or {"shaft", "bearing"} <= all_concepts
      or {"shaft", "base"} <= all_concepts
  ):
    return "coaxial"
  if any(
      token in all_tokens
      for token in (
          "panel",
          "plate",
          "frame",
          "housing",
          "bracket",
          "mount",
          "support",
          "bezel",
          "cover",
          "side",
      )
  ):
    return "planar"
  return "link"


def _pair_key(part_a: str, part_b: str) -> tuple[str, str]:
  return tuple(sorted((str(part_a), str(part_b))))


def _is_recoverable_preflight_solve_error(
    error_text: str,
    *,
    same_assembly_selection: bool,
) -> bool:
  text = str(error_text or "").strip().lower()
  if not same_assembly_selection:
    return False
  if "final validation failed on" not in text:
    return False
  if "seat_on_" not in text and "_seat_on_" not in text:
    return False
  match = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)\s*$", text)
  if match is None:
    return True
  try:
    value = float(match.group(1))
  except ValueError:
    return True
  return value <= 25.0


def _preflight_relation_metadata(
    relation_edges: list[tuple[str, str, str, str]],
    constraints,
    instances: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
  pair_info: dict[tuple[str, str], dict[str, Any]] = {}
  for parent, child, relation, label in relation_edges:
    key = _pair_key(parent, child)
    info = pair_info.setdefault(
        key,
        {
            "relations": set(),
            "labels": set(),
            "constraint_types": set(),
            "socket_kinds": set(),
        },
    )
    info["relations"].add(str(relation))
    info["labels"].add(str(label))
  for constraint in constraints:
    key = _pair_key(constraint.part_a, constraint.part_b)
    info = pair_info.setdefault(
        key,
        {
            "relations": set(),
            "labels": set(),
            "constraint_types": set(),
            "socket_kinds": set(),
        },
    )
    info["constraint_types"].add(str(constraint.ctype.value))
    socket_a = instances[constraint.part_a].sockets.get(constraint.socket_a)
    socket_b = instances[constraint.part_b].sockets.get(constraint.socket_b)
    if socket_a is not None:
      info["socket_kinds"].add(str(socket_a.kind))
    if socket_b is not None:
      info["socket_kinds"].add(str(socket_b.kind))
  return pair_info


def _classify_preflight_collisions(
    *,
    collisions,
    relation_edges: list[tuple[str, str, str, str]],
    constraints,
    instances: dict[str, Any],
    same_assembly_selection: bool,
    args,
) -> dict[str, Any]:
  pair_info = _preflight_relation_metadata(relation_edges, constraints, instances)
  base_soft_tolerance = max(
      float(getattr(args, "mate_collision_volume_tolerance", 0.0) or 0.0),
      5.0,
  )
  soft_collisions: list[dict[str, Any]] = []
  hard_collisions: list[dict[str, Any]] = []

  for collision in collisions:
    key = _pair_key(collision.part_a, collision.part_b)
    info = pair_info.get(key, {})
    relations = set(info.get("relations", set()))
    socket_kinds = set(info.get("socket_kinds", set()))
    is_mated_pair = bool(relations)
    axial_like = bool(relations & {"coaxial", "fasten"}) and bool(
        socket_kinds & {"hole", "threaded_hole", "pin", "axis"}
    )
    threshold = base_soft_tolerance
    if same_assembly_selection and is_mated_pair:
      threshold = max(threshold * 4.0, 50.0)
    if axial_like:
      threshold = max(threshold, base_soft_tolerance * 1.5)
    is_soft = bool(is_mated_pair) and float(collision.volume) <= float(threshold)
    payload = {
        "part_a": collision.part_a,
        "part_b": collision.part_b,
        "volume": float(collision.volume),
        "relations": sorted(relations),
        "socket_kinds": sorted(socket_kinds),
        "soft_threshold": float(threshold),
    }
    if is_soft:
      soft_collisions.append(payload)
    else:
      hard_collisions.append(payload)

  return {
      "soft_collisions": soft_collisions,
      "hard_collisions": hard_collisions,
      "soft_collision_volume_sum": float(
          sum(item["volume"] for item in soft_collisions)
      ),
      "hard_collision_volume_sum": float(
          sum(item["volume"] for item in hard_collisions)
      ),
  }


def _preflight_retrieved_selection(
    selection: RetrievedPartSelection,
    catalog: dict[str, Any],
    shape_library: dict[str, Any],
    args,
) -> dict[str, Any]:
  planner = RuleBasedNeuralPlanner(default_clearance=1.0)
  solver = SymbolicCadSolver(radius_clearance=0.2)
  anchor = _retrieval_anchor_part(selection)
  relation_edges = _retrieval_relation_plan(selection)
  same_assembly_selection = selection._single_assembly_id() is not None
  instances = {
      part_name: template.instantiate(part_name)
      for part_name, template in catalog.items()
  }
  socket_usage: dict[tuple[str, str], int] = {}
  constraints = []
  notes: list[str] = []

  for parent, child, relation, label in relation_edges:
    if parent not in instances or child not in instances:
      return {
          "success": False,
          "anchor": anchor,
          "relation_edges": relation_edges,
          "solve_error": f"missing_instance={parent}->{child}",
          "collision_count": 0,
          "collision_volume_sum": float("inf"),
          "hard_collision_volume_sum": float("inf"),
          "soft_collision_volume_sum": 0.0,
          "recoverable_solve_error": False,
          "same_assembly_selection": same_assembly_selection,
          "collision_messages": [],
          "planner_notes": notes,
          "preflight_score": float("inf"),
      }
    candidate = planner._best_connection(  # pylint: disable=protected-access
        parent=instances[parent],
        child=instances[child],
        socket_usage=socket_usage,
        blocked_pair=False,
        relation_hint=relation,
    )
    if candidate is None:
      return {
          "success": False,
          "anchor": anchor,
          "relation_edges": relation_edges,
          "solve_error": f"no_connection={parent}->{child}:{relation}",
          "collision_count": 0,
          "collision_volume_sum": float("inf"),
          "hard_collision_volume_sum": float("inf"),
          "soft_collision_volume_sum": 0.0,
          "recoverable_solve_error": False,
          "same_assembly_selection": same_assembly_selection,
          "collision_messages": [],
          "planner_notes": notes,
          "preflight_score": float("inf"),
      }
    constraints.extend(candidate.constraints)
    planner._reserve_candidate(candidate, socket_usage)  # pylint: disable=protected-access
    notes.append(f"{label}:{candidate.note}")

  graph = ConstraintGraph(
      instances=instances,
      constraints=constraints,
      anchor=anchor,
      metadata={"preflight": True, "planner_notes": notes},
  )
  try:
    assembly = solver.solve(graph)
  except Exception as exc:  # pylint: disable=broad-except
    error_text = f"{type(exc).__name__}:{exc}"
    recoverable = _is_recoverable_preflight_solve_error(
        error_text,
        same_assembly_selection=same_assembly_selection,
    )
    return {
        "success": False,
        "anchor": anchor,
        "relation_edges": relation_edges,
        "solve_error": error_text,
        "collision_count": 0,
        "collision_volume_sum": 0.0 if recoverable else float("inf"),
        "hard_collision_volume_sum": 0.0 if recoverable else float("inf"),
        "soft_collision_volume_sum": 0.0,
        "recoverable_solve_error": bool(recoverable),
        "same_assembly_selection": same_assembly_selection,
        "collision_messages": [],
        "planner_notes": notes,
        "preflight_score": 0.75 if recoverable else float("inf"),
    }

  collision_checker, kernel_notes = build_collision_checker(
      kernel="cadquery",
      shape_library=shape_library,
      allow_fallback=True,
      cadquery_exact_timeout_seconds=args.cadquery_exact_timeout_seconds,
      cadquery_skip_exact_for_complex_pairs=True,
      cadquery_complexity_face_threshold=args.cadquery_complexity_face_threshold,
      cadquery_complexity_score_threshold=args.cadquery_complexity_score_threshold,
      cadquery_complex_pair_aabb_ratio_threshold=(
          args.cadquery_complex_pair_aabb_ratio_threshold
      ),
  )
  collisions = collision_checker.check(
      assembly, epsilon=max(float(args.collision_epsilon), 1e-5)
  )
  collision_summary = _classify_preflight_collisions(
      collisions=collisions,
      relation_edges=relation_edges,
      constraints=constraints,
      instances=instances,
      same_assembly_selection=same_assembly_selection,
      args=args,
  )
  hard_collisions = collision_summary["hard_collisions"]
  soft_collisions = collision_summary["soft_collisions"]
  collision_messages = [
      (
          f"soft_collision({item['part_a']},{item['part_b']}) "
          f"volume={item['volume']:.6f}"
      )
      for item in soft_collisions
  ] + [
      (
          f"hard_collision({item['part_a']},{item['part_b']}) "
          f"volume={item['volume']:.6f}"
      )
      for item in hard_collisions
  ]
  collision_volume_sum = float(sum(item.volume for item in collisions))
  hard_collision_volume_sum = float(collision_summary["hard_collision_volume_sum"])
  soft_collision_volume_sum = float(collision_summary["soft_collision_volume_sum"])
  return {
      "success": len(hard_collisions) == 0,
      "anchor": anchor,
      "relation_edges": relation_edges,
      "solve_error": None,
      "collision_count": len(collisions),
      "collision_volume_sum": collision_volume_sum,
      "hard_collision_volume_sum": hard_collision_volume_sum,
      "soft_collision_volume_sum": soft_collision_volume_sum,
      "recoverable_solve_error": False,
      "same_assembly_selection": same_assembly_selection,
      "collision_messages": collision_messages,
      "planner_notes": notes + list(kernel_notes),
      "preflight_score": hard_collision_volume_sum + 0.05 * soft_collision_volume_sum,
  }


def _augment_instruction_with_retrieval_hints(
    instruction: str,
    selection: RetrievedPartSelection,
) -> str:
  roles = selection.role_assignments
  if not roles:
    return instruction
  role_chunks = [
      f"{role}={part_name}"
      for role, part_name in roles.items()
      if isinstance(role, str) and isinstance(part_name, str)
  ]
  relation_chunks = []
  for parent, child, relation, _ in _retrieval_relation_plan(selection):
    relation_chunks.append(f"{parent}->{child}:{relation}")
  hint_lines = [
      instruction.strip(),
      "Role hints: " + "; ".join(role_chunks) + ".",
      "Use the base or housing role as the anchor when available.",
      "Keep every selected part in one connected graph.",
  ]
  if relation_chunks:
    hint_lines.append(
        "Preferred topology: " + "; ".join(relation_chunks) + "."
    )
  if "shaft" in roles and "gear" in roles and "base" in roles:
    hint_lines.append(
        "Do not connect the gear directly to the base when a shaft role is present."
    )
  return " ".join(chunk for chunk in hint_lines if chunk)


def _base_like_name(name: str) -> bool:
  lowered = str(name or "").strip().lower()
  if not lowered:
    return False
  return any(
      token in lowered
      for token in ("base", "housing", "frame", "bracket", "mount", "holder", "stand", "chassis")
  )


def _candidate_tray_core_variants(
    selection: RetrievedPartSelection,
) -> list[RetrievedPartSelection]:
  parts = list(selection.selected_parts)
  if len(parts) <= 2:
    return []
  by_name = {entry.part_name: entry for entry in parts}
  adjacency: dict[str, set[str]] = {entry.part_name: set() for entry in parts}
  for pair in selection.contact_pairs:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
      continue
    left = str(pair[0]).strip()
    right = str(pair[1]).strip()
    if left in adjacency and right in adjacency and left != right:
      adjacency[left].add(right)
      adjacency[right].add(left)

  seed_names: list[str] = []
  for part_name in selection.role_assignments.values():
    name = str(part_name or "").strip()
    if name and name in by_name and name not in seed_names:
      seed_names.append(name)
  if not any(_base_like_name(name) for name in seed_names):
    base_candidates = [entry.part_name for entry in parts if _base_like_name(entry.part_name)]
    if base_candidates:
      if base_candidates[0] not in seed_names:
        seed_names.insert(0, base_candidates[0])
    else:
      anchor_like = max(
          parts,
          key=lambda item: (int(item.contact_degree), float(item.score), -float(item.volume)),
      )
      if anchor_like.part_name not in seed_names:
        seed_names.insert(0, anchor_like.part_name)
  if len(seed_names) < 2:
    for entry in sorted(parts, key=lambda item: (-int(item.contact_degree), -float(item.score), item.part_name)):
      if entry.part_name not in seed_names:
        seed_names.append(entry.part_name)
      if len(seed_names) >= 2:
        break

  max_subset = min(4, len(parts))
  variants: list[RetrievedPartSelection] = []
  seen_keys: set[tuple[str, ...]] = set()
  original_order = [entry.part_name for entry in parts]
  for target_size in range(max(2, len(seed_names)), max_subset + 1):
    chosen = list(seed_names[:target_size])
    while len(chosen) < target_size:
      pool = [entry for entry in parts if entry.part_name not in chosen]
      if not pool:
        break
      best = max(
          pool,
          key=lambda item: (
              sum(1 for neighbor in adjacency.get(item.part_name, set()) if neighbor in chosen),
              int(item.contact_degree),
              float(item.score),
              -float(item.volume),
          ),
      )
      chosen.append(best.part_name)
    ordered_names = tuple(name for name in original_order if name in set(chosen))
    if len(ordered_names) < 2 or ordered_names in seen_keys:
      continue
    seen_keys.add(ordered_names)
    selected_parts = [by_name[name] for name in ordered_names]
    selected_names = set(ordered_names)
    filtered_roles = {
        role: name
        for role, name in selection.role_assignments.items()
        if name in selected_names
    }
    filtered_contacts = [
        pair
        for pair in selection.contact_pairs
        if pair[0] in selected_names and pair[1] in selected_names
    ]
    variants.append(
        RetrievedPartSelection(
            instruction=selection.instruction,
            selected_parts=selected_parts,
            prompt_tokens=list(selection.prompt_tokens),
            prompt_concepts=list(selection.prompt_concepts),
            notes=list(selection.notes)
            + [f"candidate_tray_core_subset={len(selected_parts)}"],
            contact_pairs=filtered_contacts,
            preferred_assembly_id=selection.preferred_assembly_id,
            role_assignments=filtered_roles,
            semantic_score=float(selection.semantic_score),
            candidate_pool=list(selection.candidate_pool),
        )
    )
  return variants


def _choose_retrieved_case(
    case_id: str,
    instruction: str,
    candidate_sets: list[RetrievedPartSelection],
    dataset_roots: list[Path],
    file_index: dict[str, list[Path]],
    extraction_config,
    args,
    task_mode: str = "retrieval",
    reference_case: Optional[dict[str, Any]] = None,
    interface_scorer: Optional[InterfaceScorer] = None,
) -> tuple[dict[str, Any], RetrievedPartSelection, dict[str, Any]]:
  diagnostics: list[dict[str, Any]] = []
  best_case: Optional[dict[str, Any]] = None
  best_selection: Optional[RetrievedPartSelection] = None
  best_rank = None
  best_diagnostic: Optional[dict[str, Any]] = None
  expanded_sets: list[RetrievedPartSelection] = []
  for selection in candidate_sets[:6]:
    expanded_sets.append(selection)
    if task_mode == "candidate_tray":
      expanded_sets.extend(_candidate_tray_core_variants(selection))

  for rank_index, selection in enumerate(expanded_sets[:18], start=1):
    candidate_case = selection.to_case(case_id=case_id)
    try:
      catalog, shape_library, selected_parts, _, load_notes = _build_case_catalog(
          case=candidate_case,
          dataset_roots=dataset_roots,
          file_index=file_index,
          extraction_config=extraction_config,
          interface_scorer=interface_scorer,
          interface_top_k=int(getattr(args, "interface_top_k", 0) or 0),
          interface_min_score=float(getattr(args, "interface_min_score", 0.25) or 0.25),
          canonicalize_parts=(
              not bool(getattr(args, "no_part_local_canonicalization", False))
              and str(getattr(args, "source_pose_mode", "off")).lower() != "force"
          ),
          input_protocol=str(getattr(args, "input_protocol", "legacy")),
          frame_randomization_seed=int(
              getattr(args, "frame_randomization_seed", 1001)
          ),
          benchmark_v2_translation_box_fraction=float(
              getattr(args, "benchmark_v2_translation_box_fraction", 1.0)
          ),
      )
      preflight = _preflight_retrieved_selection(
          selection=selection,
          catalog=catalog,
          shape_library=shape_library,
          args=args,
      )
      diagnostic = {
          "rank_index": rank_index,
          "selected_parts": list(selected_parts),
          "assembly_dir": candidate_case.get("assembly_dir"),
          "semantic_score": round(float(selection.semantic_score), 4),
          "role_assignments": dict(selection.role_assignments),
          "notes": list(selection.notes),
          "load_notes": list(load_notes),
          "preflight": preflight,
      }
    except Exception as exc:  # pylint: disable=broad-except
      diagnostic = {
          "rank_index": rank_index,
          "selected_parts": [entry.part_name for entry in selection.selected_parts],
          "assembly_dir": selection._single_assembly_id(),
          "semantic_score": round(float(selection.semantic_score), 4),
          "role_assignments": dict(selection.role_assignments),
          "notes": list(selection.notes),
          "preflight": {
              "success": False,
              "solve_error": f"{type(exc).__name__}:{exc}",
              "collision_count": 0,
              "collision_volume_sum": float("inf"),
              "collision_messages": [],
          },
      }
    diagnostics.append(diagnostic)

    preflight = diagnostic["preflight"]
    same_assembly_selection = bool(preflight.get("same_assembly_selection"))
    recoverable = bool(preflight.get("recoverable_solve_error"))
    preflight_score = float(preflight.get("preflight_score", float("inf")))
    tier = 4
    if bool(preflight.get("success")) and same_assembly_selection:
      tier = 0
    elif recoverable and same_assembly_selection:
      tier = 1
    elif bool(preflight.get("success")):
      tier = 2
    elif recoverable:
      tier = 3
    selected_count = len(diagnostic.get("selected_parts", []))
    reference_count = (
        _reference_part_count_for_case(reference_case)
        if task_mode == "candidate_tray"
        else 0
    )
    missing_reference_count = (
        max(0, int(reference_count) - int(selected_count))
        if reference_count > 0
        else 0
    )
    extra_reference_count = (
        max(0, int(selected_count) - int(reference_count))
        if reference_count > 0
        else 0
    )
    rank = (
        tier,
        missing_reference_count,
        extra_reference_count if task_mode == "candidate_tray" else 0,
        selected_count if task_mode != "candidate_tray" else 0,
        preflight_score,
        0 if diagnostic.get("assembly_dir") else 1,
        -float(diagnostic.get("semantic_score", 0.0)),
    )
    if best_rank is None or rank < best_rank:
      best_rank = rank
      best_case = candidate_case
      best_selection = selection
      best_diagnostic = diagnostic

  if best_case is None or best_selection is None or best_diagnostic is None:
    raise ValueError("No retrieved candidate sets were available.")

  best_diagnostic = {
      "selected_rank_index": best_diagnostic["rank_index"],
      "candidates": diagnostics,
  }
  return best_case, best_selection, best_diagnostic


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
      help=(
          "Geometry input protocol. benchmark_v2 single-loads each B-Rep, "
          "applies independent nonce-bound SE(3), and exposes only sanitized views."
      ),
  )
  parser.add_argument("--frame_randomization_seed", type=int, default=1001)
  parser.add_argument(
      "--benchmark_v2_translation_box_fraction",
      type=float,
      default=1.0,
  )
  parser.add_argument(
      "--test_cases_json",
      default=None,
      help="Path to test_cases JSON file.",
  )
  parser.add_argument(
      "--instruction",
      default=None,
      help="Optional direct natural-language prompt. When set, test_cases_json is not required.",
  )
  parser.add_argument(
      "--case_id",
      default="prompt_case_0001",
      help="Case id used with --instruction mode.",
  )
  parser.add_argument(
      "--dataset_root",
      default=None,
      help=(
          "Dataset root that stores assembly folders and STEP files. Defaults "
          "to NEUROCAD_DATASET_ROOT(S), then the project-local archive root."
      ),
  )
  parser.add_argument(
      "--dataset_roots",
      nargs="*",
      default=None,
      help=(
          "Optional additional dataset roots. You can pass multiple values or a "
          "single semicolon-separated string. These are indexed together with "
          "--dataset_root."
      ),
  )
  parser.add_argument(
      "--output_dir",
      default=None,
      help=(
          "Output directory for result JSON and exported STEP assemblies. "
          "Defaults to NEUROCAD_OUTPUT_DIR, then <project>/results_batch."
      ),
  )
  parser.add_argument(
      "--max_cases",
      type=int,
      default=0,
      help="Optional limit of cases processed (0 means all).",
  )
  parser.add_argument(
      "--case_timeout_seconds",
      type=int,
      default=0,
      help=(
          "Optional hard timeout per case. When >0, each case runs in a child "
          "process so CAD-kernel hangs are recorded as failures instead of "
          "stalling the batch."
      ),
  )
  parser.add_argument(
      "--max_iterations",
      type=int,
      default=6,
      help="Closed-loop max retries per case.",
  )
  parser.add_argument(
      "--save_failed_case_json",
      action="store_true",
      help="Save case result JSON even when case loading fails.",
  )

  parser.add_argument(
      "--planner",
      default="ollama",
      choices=["rule", "openai", "anthropic", "ollama", "deepseek"],
      help="Neural planner provider.",
  )
  parser.add_argument("--model", default=None, help="LLM model override.")
  parser.add_argument(
      "--base_url",
      default=None,
      help="Optional provider base URL override.",
  )
  parser.add_argument(
      "--llm_timeout_seconds",
      type=int,
      default=60,
      help="LLM request timeout.",
  )
  parser.add_argument(
      "--llm_temperature",
      type=float,
      default=0.28,
      help="LLM base decoding temperature.",
  )
  parser.add_argument(
      "--llm_temperature_step",
      type=float,
      default=0.14,
      help="Adaptive temperature increment.",
  )
  parser.add_argument(
      "--llm_temperature_max",
      type=float,
      default=0.85,
      help="Maximum adaptive temperature.",
  )
  parser.add_argument(
      "--llm_retry_on_invalid",
      type=int,
      default=4,
      help="Extra in-iteration retries for invalid planner output.",
  )
  parser.add_argument(
      "--llm_max_prompt_sockets_per_part",
      type=int,
      default=12,
      help="Max sockets per part exposed to LLM.",
  )
  parser.add_argument(
      "--llm_max_prompt_total_sockets",
      type=int,
      default=84,
      help="Global max sockets exposed to LLM prompt.",
  )
  parser.add_argument(
      "--llm_no_few_shot",
      action="store_true",
      help="Disable few-shot examples.",
  )
  parser.add_argument(
      "--llm_few_shot_example_count",
      type=int,
      default=2,
      help="Few-shot example count in prompt.",
  )
  parser.add_argument(
      "--llm_reasoning_mode",
      choices=["none", "brief"],
      default="brief",
      help="Reasoning mode for planner prompt.",
  )
  parser.add_argument(
      "--llm_thinking_mode",
      choices=["disabled", "two_stage"],
      default="disabled",
      help=(
          "DeepSeek thinking ablation mode. Final JSON emission always disables "
          "thinking."
      ),
  )
  parser.add_argument(
      "--planning_mode",
      choices=["topology_graph", "constraint_graph"],
      default="topology_graph",
      help="Planning abstraction level for LLM ablations.",
  )
  parser.add_argument(
      "--source_pose_mode",
      choices=["off", "force"],
      default="off",
      help=(
          "Preserve original per-body STEP coordinates for controlled "
          "oracle/candidate-tray sanity checks. This is a diagnostic upper "
          "bound, not a learned pose solver."
      ),
  )
  parser.add_argument(
      "--interface_scorer_model",
      default=None,
      help=(
          "Optional learned interface scorer model. When set, each STEP is "
          "augmented with top-k single-part interface sockets before planning."
      ),
  )
  parser.add_argument("--interface_top_k", type=int, default=12)
  parser.add_argument("--interface_min_score", type=float, default=0.25)
  parser.add_argument(
      "--interface_pair_scorer_model",
      default=None,
      help=(
          "Optional learned interface-pair scorer. When set, physics grounding "
          "uses it to rank socket pairs before pose search."
      ),
  )
  parser.add_argument(
      "--interface_pair_top_k",
      type=int,
      default=12,
      help="Maximum learned interface pairs kept per graph edge after pair scoring.",
  )
  parser.add_argument(
      "--no_part_local_canonicalization",
      action="store_true",
      help=(
          "Disable per-STEP bbox-center canonicalization. Useful only for "
          "source-pose diagnostics when body STEP files already carry assembly "
          "world coordinates."
      ),
  )
  parser.add_argument(
      "--no_fallback_to_rules",
      action="store_true",
      help="Disable fallback to rule planner.",
  )

  parser.add_argument(
      "--kernel",
      default="cadquery",
      choices=["aabb", "cadquery"],
      help="Collision kernel.",
  )
  parser.add_argument(
      "--no_kernel_fallback_to_aabb",
      action="store_true",
      help="Disable fallback to AABB in collision checker.",
  )
  parser.add_argument(
      "--collision_epsilon",
      type=float,
      default=1e-5,
      help="Collision volume threshold.",
  )
  parser.add_argument(
      "--mate_collision_volume_tolerance",
      type=float,
      default=50.0,
      help=(
          "Allowed collision volume for already-mated hole-pin / hole-axis pairs. "
          "Non-mated pairs remain strict."
      ),
  )
  parser.add_argument(
      "--threaded_interference_collision_tolerance",
      type=float,
      default=500.0,
      help=(
          "Allowed collision volume for constraints predicted as threaded or "
          "intentional interference contacts."
      ),
  )
  parser.add_argument(
      "--exact_contact_distance_tolerance",
      type=float,
      default=2.0,
      help=(
          "Maximum OCC shape-to-shape distance for an expected contact pair. "
          "Used to reject scattered false positives in oracle/candidate-tray runs."
      ),
  )
  parser.add_argument(
      "--exact_contact_min_recall",
      type=float,
      default=0.8,
      help=(
          "Minimum fraction of expected contact pairs that must be within "
          "--exact_contact_distance_tolerance."
      ),
  )
  parser.add_argument(
      "--no_exact_contact_validation",
      action="store_true",
      help="Disable exact contact distance validation.",
  )
  parser.add_argument(
      "--no_physics_guided_grounding",
      action="store_true",
      help="Disable physics-guided interface pair reranking.",
  )
  parser.add_argument(
      "--physics_grounding_max_pairs",
      type=int,
      default=8,
      help="Maximum learned interface pairs to physically score per graph edge.",
  )
  parser.add_argument(
      "--physics_grounding_max_pose_candidates",
      type=int,
      default=12,
      help="Maximum axial seat candidates to score per interface pair.",
  )
  parser.add_argument(
      "--physics_grounding_edge_timeout_seconds",
      type=float,
      default=60.0,
      help="Soft per-edge deadline for physics-guided grounding search.",
  )
  parser.add_argument(
      "--skip_obround_slot_grounding",
      action="store_true",
      help="Ignore obround-slot interfaces in physics grounding for main-relation runs.",
  )
  parser.add_argument(
      "--no_physics_exact_contact_rerank",
      action="store_true",
      help=(
          "Disable exact OCC distance/collision reranking for the selected "
          "physics-grounded pose."
      ),
  )
  parser.add_argument(
      "--physics_exact_contact_rerank_timeout_seconds",
      type=float,
      default=2.0,
      help="Timeout for exact distance checks used by physics pose reranking.",
  )
  parser.add_argument(
      "--physics_exact_contact_rerank_tolerance",
      type=float,
      default=None,
      help=(
          "Distance tolerance for physics pose reranking. Defaults to "
          "--exact_contact_distance_tolerance."
      ),
  )
  parser.add_argument(
      "--no_physics_micro_retreat",
      action="store_true",
      help="Disable pair-conditioned micro axial retreat after pose search.",
  )
  parser.add_argument(
      "--physics_micro_retreat_max_distance",
      type=float,
      default=2.0,
      help="Maximum retreat distance in mm for pair-conditioned pose cleanup.",
  )
  parser.add_argument(
      "--physics_micro_retreat_timeout_seconds",
      type=float,
      default=3.0,
      help="Timeout for each exact boolean used by micro-retreat pose search.",
  )
  parser.add_argument(
      "--cadquery_exact_timeout_seconds",
      type=float,
      default=8.0,
      help="Hard timeout for exact OCC boolean in child process.",
  )
  parser.add_argument(
      "--cadquery_no_skip_exact_for_complex_pairs",
      action="store_true",
      help="Always try exact boolean, even on complex pairs.",
  )
  parser.add_argument(
      "--cadquery_complexity_face_threshold",
      type=int,
      default=180,
      help="Face-count threshold that marks a part as geometrically complex.",
  )
  parser.add_argument(
      "--cadquery_complexity_score_threshold",
      type=float,
      default=240.0,
      help="Complexity score threshold that marks a part as geometrically complex.",
  )
  parser.add_argument(
      "--cadquery_complex_pair_aabb_ratio_threshold",
      type=float,
      default=0.02,
      help=(
          "For complex pairs below this AABB overlap ratio, skip exact boolean "
          "and use conservative fallback."
      ),
  )

  parser.add_argument("--socket_max_holes", type=int, default=12)
  parser.add_argument("--socket_max_pins", type=int, default=8)
  parser.add_argument("--socket_max_threads", type=int, default=6)
  parser.add_argument("--socket_max_planes", type=int, default=8)
  parser.add_argument("--socket_max_flanges", type=int, default=4)
  parser.add_argument("--socket_max_guides", type=int, default=4)
  parser.add_argument("--socket_min_plane_area_ratio", type=float, default=0.02)
  parser.add_argument("--socket_min_cyl_area_ratio", type=float, default=0.0015)
  parser.add_argument(
      "--retrieval_part_count",
      type=int,
      default=0,
      help="Auto-retrieval part count (0 means infer from prompt concepts).",
  )
  parser.add_argument(
      "--retrieval_context_protocol",
      choices=["fixed_tray", "context_assisted_dev"],
      default="fixed_tray",
      help=(
          "Formal/default runs accept only an explicit fixed tray. "
          "Role/contact/reference-assisted adaptation is development-only."
      ),
  )
  parser.add_argument(
      "--retrieval_max_assemblies_loaded",
      type=int,
      default=400,
      help="How many assembly folders to index for natural-language retrieval.",
  )
  parser.add_argument(
      "--retrieval_embedding_model",
      default=None,
      help="Optional OpenAI-compatible embedding model for retrieval reranking.",
  )
  parser.add_argument(
      "--retrieval_embedding_base_url",
      default="http://localhost:11434/v1",
      help="Embedding endpoint base URL.",
  )
  parser.add_argument(
      "--retrieval_embedding_timeout_seconds",
      type=int,
      default=45,
      help="Timeout for retrieval embedding requests.",
  )
  parser.add_argument(
      "--retrieval_embedding_batch_size",
      type=int,
      default=48,
      help="Batch size when embedding retrieval entries.",
  )
  parser.add_argument(
      "--retrieval_embedding_weight",
      type=float,
      default=2.25,
      help="How strongly embedding similarity reranks lexical retrieval.",
  )
  parser.add_argument(
      "--retrieval_no_prefer_same_assembly",
      action="store_true",
      help="Disable the preference for semantically coherent same-assembly retrieval.",
  )
  return parser


def normalize_args(args):
  args.fallback_to_rules = not bool(args.no_fallback_to_rules)
  args.kernel_fallback_to_aabb = not bool(args.no_kernel_fallback_to_aabb)
  args.llm_use_few_shot = not bool(args.llm_no_few_shot)
  args.cadquery_skip_exact_for_complex_pairs = not bool(
      args.cadquery_no_skip_exact_for_complex_pairs
  )
  args.retrieval_prefer_same_assembly = not bool(
      args.retrieval_no_prefer_same_assembly
  )
  args.dataset_roots_resolved = _parse_dataset_roots(
      dataset_root=getattr(args, "dataset_root", None),
      dataset_roots=getattr(args, "dataset_roots", None),
  )
  args.dataset_root = str(args.dataset_roots_resolved[0])
  args.dataset_roots = [str(path) for path in args.dataset_roots_resolved[1:]]
  args.output_dir = str(
      resolve_path(
          getattr(args, "output_dir", None),
          env_var="NEUROCAD_OUTPUT_DIR",
          default="results_batch",
      )
  )
  for name in (
      "test_cases_json",
      "interface_scorer_model",
      "interface_pair_scorer_model",
  ):
    value = getattr(args, name, None)
    if value:
      setattr(args, name, str(resolve_path(value)))
  return args


def _run_single_case_timeout_worker(
    args_dict: dict[str, Any],
    case: dict[str, Any],
    queue: mp.Queue,
) -> None:
  args = argparse.Namespace(**args_dict)
  args.case_timeout_seconds = 0
  args.max_cases = 0
  try:
    summary = run_batch(args, cases=[case])
    queue.put({"ok": True, "summary": summary})
  except BaseException as exc:  # pylint: disable=broad-except
    queue.put({"ok": False, "error": f"{type(exc).__name__}:{exc}"})


def _timeout_payload(
    *,
    case: dict[str, Any],
    case_id: str,
    instruction: str,
    task_mode: str,
    timeout_seconds: int,
) -> dict[str, Any]:
  payload: dict[str, Any] = {
      "id": case_id,
      "success": False,
      "tolerant_success": False,
      "strict_success": False,
      "instruction": instruction,
      "benchmark_task": task_mode,
      "attempts": [
          {
              "iteration": 0,
              "planner_notes": [],
              "solver_logs": [],
              "feedback_messages": [
                  f"case_timeout_seconds={int(timeout_seconds)}"
              ],
              "tolerated_collision_messages": [],
          }
      ],
      "final_graph": None,
      "final_assembly": None,
      "final_feedback": [f"case_timeout_seconds={int(timeout_seconds)}"],
      "case": case,
      "output_step": None,
      "output_step_error": "case_timeout",
      "assembly_plan": None,
      "self_correction_count": 0,
      "cumulative_collision_volume_sum": 0.0,
      "collision_volume_sum": 0.0,
      "blocking_collision_volume_sum": 0.0,
      "tolerated_collision_volume_sum": 0.0,
      "strict_collision_volume_sum": 0.0,
      "tolerated_collision_count": 0,
  }
  payload["evaluation"] = evaluate_generation_quality(payload)
  payload["failure_taxonomy"] = classify_failure(payload)
  return payload


def _evaluate_timeout_batch_output(output_dir: Path) -> dict[str, Any]:
  summary_path = output_dir / "summary.json"
  cmd = [
      sys.executable,
      "-m",
      "neurocad.evaluate_batch_results",
      "--results_dir",
      str(output_dir),
      "--output_json",
      str(summary_path),
  ]
  completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if summary_path.exists():
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
      return payload
  return {
      "summary": {
          "output_dir": str(output_dir.resolve()),
          "evaluation_error": completed.stderr[-2000:] or completed.stdout[-2000:],
      },
      "results": [],
  }


def _build_assembly_plan_payload(
    *,
    case_id: str,
    payload: dict[str, Any],
    output_step: Optional[str],
) -> dict[str, Any]:
  final_graph = payload.get("final_graph")
  final_assembly = payload.get("final_assembly")
  graph = final_graph if isinstance(final_graph, dict) else {}
  assembly = final_assembly if isinstance(final_assembly, dict) else {}
  instances = assembly.get("instances")
  if not isinstance(instances, dict):
    instances = {}

  parts = []
  selected_step_paths = payload.get("selected_step_paths")
  selected_parts = payload.get("selected_parts")
  if isinstance(selected_step_paths, dict):
    step_by_name = selected_step_paths
  elif isinstance(selected_step_paths, list) and isinstance(selected_parts, list):
    step_by_name = {
        str(name): str(path)
        for name, path in zip(selected_parts, selected_step_paths)
    }
  else:
    step_by_name = {}
  for instance_id, instance in sorted(instances.items()):
    if not isinstance(instance, dict):
      continue
    parts.append(
        {
            "instance_id": instance_id,
            "template_name": instance.get("template_name"),
            "source_step": step_by_name.get(instance_id),
            "transform": instance.get("transform"),
            "local_bbox_min": instance.get("local_bbox_min"),
            "local_bbox_max": instance.get("local_bbox_max"),
        }
    )

  attempts = payload.get("attempts")
  solver_decisions = []
  planner_decisions = []
  if isinstance(attempts, list):
    for attempt in attempts:
      if not isinstance(attempt, dict):
        continue
      solver_decisions.extend(
          str(item) for item in attempt.get("solver_logs", []) or []
      )
      planner_decisions.extend(
          str(item) for item in attempt.get("planner_notes", []) or []
      )

  return {
      "id": case_id,
      "instruction": payload.get("instruction"),
      "output_step": output_step,
      "benchmark_task": payload.get("benchmark_task"),
      "success": payload.get("success"),
      "strict_success": payload.get("strict_success"),
      "tolerant_success": payload.get("tolerant_success"),
      "parts": parts,
      "mate_graph": payload.get("canonical_mate_graph"),
      "constraint_graph": {
          "anchor": graph.get("anchor"),
          "constraints": graph.get("constraints", []),
      },
      "solver_decisions": solver_decisions,
      "planner_decisions": planner_decisions,
      "failure_taxonomy": payload.get("failure_taxonomy"),
  }


def run_batch_with_case_timeouts(
    args,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "assembled_step").mkdir(parents=True, exist_ok=True)
  timeout_seconds = int(args.case_timeout_seconds)
  args_dict = vars(args).copy()
  args_dict["case_timeout_seconds"] = 0
  args_dict["max_cases"] = 0

  ctx = mp.get_context("spawn")
  for index, case in enumerate(cases, start=1):
    case_id = str(case.get("id") or f"case_{index:04d}")
    instruction = str(case.get("instruction") or "").strip()
    task_mode = _case_task_mode(case)
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_run_single_case_timeout_worker,
        args=(args_dict, case, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
      process.terminate()
      process.join(timeout=5.0)
      if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=2.0)
      payload = _timeout_payload(
          case=case,
          case_id=case_id,
          instruction=instruction,
          task_mode=task_mode,
          timeout_seconds=timeout_seconds,
      )
      result_path = output_dir / f"result_{case_id}.json"
      result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
      continue
    if process.exitcode not in (0, None):
      result_path = output_dir / f"result_{case_id}.json"
      if not result_path.exists():
        payload = _timeout_payload(
            case=case,
            case_id=case_id,
            instruction=instruction,
            task_mode=task_mode,
            timeout_seconds=timeout_seconds,
        )
        payload["final_feedback"] = [
            f"case_process_error=exitcode:{process.exitcode}"
        ]
        payload["output_step_error"] = "case_process_error"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
      _ = queue.get_nowait()
    except Exception:
      pass

  summary_payload = _evaluate_timeout_batch_output(output_dir)
  if isinstance(summary_payload.get("summary"), dict):
    summary_payload["summary"]["case_timeout_seconds"] = timeout_seconds
    summary_payload["summary"]["output_dir"] = str(output_dir.resolve())
  summary_path = output_dir / "summary.json"
  summary_path.write_text(
      json.dumps(summary_payload, indent=2, ensure_ascii=False),
      encoding="utf-8",
  )
  return summary_payload


def run_batch(
    args,
    cases: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
  args = normalize_args(args)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  step_output_dir = output_dir / "assembled_step"
  step_output_dir.mkdir(parents=True, exist_ok=True)
  assembly_plan_dir = output_dir / "assembly_plan"
  assembly_plan_dir.mkdir(parents=True, exist_ok=True)

  dataset_roots = [Path(path) for path in args.dataset_roots_resolved]
  file_index = _index_step_files(dataset_roots)
  if cases is None and args.instruction:
    cases = [
        {
            "id": str(args.case_id),
            "instruction": str(args.instruction).strip(),
        }
    ]
  elif cases is None and args.test_cases_json:
    cases = _load_cases(Path(args.test_cases_json))
  elif cases is None:
    raise SystemExit("Provide either --test_cases_json or --instruction.")

  assert cases is not None
  if args.max_cases and int(args.max_cases) > 0 and not args.instruction:
    cases = cases[: int(args.max_cases)]
  if not cases:
    raise SystemExit("No test cases to run.")
  if (
      int(getattr(args, "case_timeout_seconds", 0) or 0) > 0
      and not args.instruction
  ):
    return run_batch_with_case_timeouts(args, cases)

  retrieval_index: Optional[PartRetrievalIndex] = None
  runtime_input_protocol = str(getattr(args, "input_protocol", "legacy"))
  if any(
      _case_needs_retrieval(case, input_protocol=runtime_input_protocol)
      for case in cases
  ):
    retrieval_index = build_part_retrieval_index(
        dataset_root=dataset_roots,
        max_assemblies=max(0, int(args.retrieval_max_assemblies_loaded)),
    )
    if args.retrieval_embedding_model:
      retrieval_index = enable_embedding_rerank(
          retrieval_index,
          model=str(args.retrieval_embedding_model),
          base_url=str(args.retrieval_embedding_base_url),
          api_key=None,
          timeout_seconds=int(args.retrieval_embedding_timeout_seconds),
          batch_size=int(args.retrieval_embedding_batch_size),
          score_weight=float(args.retrieval_embedding_weight),
      )

  extraction_config = build_extraction_config(args)
  interface_scorer = None
  if getattr(args, "interface_scorer_model", None):
    interface_scorer = InterfaceScorer.load(
        Path(str(args.interface_scorer_model)),
        expected_protocol=str(getattr(args, "input_protocol", "legacy")),
    )
  interface_pair_scorer = None
  if getattr(args, "interface_pair_scorer_model", None):
    interface_pair_scorer = InterfacePairScorer.load(
        Path(str(args.interface_pair_scorer_model)),
        expected_protocol=str(getattr(args, "input_protocol", "legacy")),
    )
  planner = build_planner(
      planner=args.planner,
      model=args.model,
      api_key=None,
      base_url=args.base_url,
      llm_timeout_seconds=args.llm_timeout_seconds,
      llm_temperature=args.llm_temperature,
      llm_temperature_step=args.llm_temperature_step,
      llm_temperature_max=args.llm_temperature_max,
      llm_retry_on_invalid=args.llm_retry_on_invalid,
      fallback_to_rules=args.fallback_to_rules,
      llm_max_prompt_sockets_per_part=args.llm_max_prompt_sockets_per_part,
      llm_max_prompt_total_sockets=args.llm_max_prompt_total_sockets,
      llm_use_few_shot=args.llm_use_few_shot,
      llm_few_shot_example_count=args.llm_few_shot_example_count,
      llm_reasoning_mode=args.llm_reasoning_mode,
      llm_thinking_mode=args.llm_thinking_mode,
      planning_mode=args.planning_mode,
      model_input_protocol=str(getattr(args, "input_protocol", "legacy")),
  )
  solver = SymbolicCadSolver(radius_clearance=0.2)

  rows: list[dict[str, Any]] = []
  for index, case in enumerate(cases, start=1):
    case_id = str(case.get("id") or f"case_{index:04d}")
    instruction = str(case.get("instruction") or "").strip()
    task_mode = _case_task_mode(case)
    case_result: Optional[PipelineResult] = None
    case_shape_library: dict[str, Any] = {}
    if not instruction:
      row = {
          "id": case_id,
          "error": "missing_instruction",
      }
      rows.append(row)
      if args.save_failed_case_json:
        (output_dir / f"result_{case_id}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
      continue

    payload: dict[str, Any]
    try:
      fixed_candidate_tray = _is_fixed_candidate_tray_case(case)
      case_for_run = (
          _materialize_fixed_candidate_tray_case(case)
          if fixed_candidate_tray
          and str(getattr(args, "input_protocol", "legacy")) == "benchmark_v2"
          else case
      )
      retrieval_selection: Optional[RetrievedPartSelection] = None
      retrieval_preflight: Optional[dict[str, Any]] = None
      retrieval_notes: list[str] = []
      retrieval_context_protocol = require_safe_retrieval_protocol(
          input_protocol=str(getattr(args, "input_protocol", "legacy")),
          retrieval_context_protocol=str(
              getattr(args, "retrieval_context_protocol", "fixed_tray")
          ),
          case_needs_retrieval=_case_needs_retrieval(
              case,
              input_protocol=str(getattr(args, "input_protocol", "legacy")),
          ),
          fixed_candidate_tray=fixed_candidate_tray,
      )
      if _case_needs_retrieval(
          case,
          input_protocol=str(getattr(args, "input_protocol", "legacy")),
      ):
        if retrieval_index is None:
          raise ValueError("retrieval_index was not initialized.")
        case_retrieval_index = retrieval_index
        if task_mode == "candidate_tray":
          case_retrieval_index = _candidate_tray_index_for_case(
              case=case,
              retrieval_index=retrieval_index,
              dataset_roots=dataset_roots,
              file_index=file_index,
          )
        retrieval_candidates = retrieve_candidate_part_sets(
            instruction=instruction,
            index=case_retrieval_index,
            part_count=int(args.retrieval_part_count),
            prefer_same_assembly=bool(args.retrieval_prefer_same_assembly),
        )
        (
            case_for_run,
            retrieval_selection,
            retrieval_preflight,
        ) = _choose_retrieved_case(
            case_id=case_id,
            instruction=instruction,
            candidate_sets=retrieval_candidates,
            dataset_roots=dataset_roots,
            file_index=file_index,
            extraction_config=extraction_config,
            args=args,
            task_mode=task_mode,
            reference_case=case,
            interface_scorer=interface_scorer,
        )
        retrieval_notes = list(retrieval_selection.notes)
        if retrieval_preflight is not None:
          retrieval_notes.append(
              "retrieval_preflight_selected_rank="
              + str(retrieval_preflight.get("selected_rank_index"))
          )
        instruction = _augment_instruction_with_retrieval_hints(
            str(case_for_run.get("instruction") or instruction),
            retrieval_selection,
        )
      protocol_audit: dict[str, Any] = {}
      benchmark_v2_prepared: list[PreparedBenchmarkV2Assembly] = []
      (
          catalog,
          shape_library,
          selected_parts,
          selected_step_paths,
          load_notes,
      ) = _build_case_catalog(
          case=case_for_run,
          dataset_roots=dataset_roots,
          file_index=file_index,
          extraction_config=extraction_config,
          interface_scorer=interface_scorer,
          interface_top_k=int(getattr(args, "interface_top_k", 0) or 0),
          interface_min_score=float(getattr(args, "interface_min_score", 0.25) or 0.25),
          canonicalize_parts=(
              not bool(getattr(args, "no_part_local_canonicalization", False))
              and str(getattr(args, "source_pose_mode", "off")).lower() != "force"
          ),
          input_protocol=str(getattr(args, "input_protocol", "legacy")),
          frame_randomization_seed=int(
              getattr(args, "frame_randomization_seed", 1001)
          ),
          benchmark_v2_translation_box_fraction=float(
              getattr(args, "benchmark_v2_translation_box_fraction", 1.0)
          ),
          protocol_audit=protocol_audit,
          benchmark_v2_prepared_out=benchmark_v2_prepared,
      )
      model_instruction = instruction
      if benchmark_v2_prepared:
        model_instruction = benchmark_v2_prepared[0].sanitize_model_instruction(
            instruction
        )
      case_shape_library = shape_library
      collision_checker, kernel_notes = build_collision_checker(
          kernel=args.kernel,
          shape_library=shape_library,
          allow_fallback=args.kernel_fallback_to_aabb,
          cadquery_exact_timeout_seconds=args.cadquery_exact_timeout_seconds,
          cadquery_skip_exact_for_complex_pairs=(
              args.cadquery_skip_exact_for_complex_pairs
          ),
          cadquery_complexity_face_threshold=(
              args.cadquery_complexity_face_threshold
          ),
          cadquery_complexity_score_threshold=(
              args.cadquery_complexity_score_threshold
          ),
          cadquery_complex_pair_aabb_ratio_threshold=(
              args.cadquery_complex_pair_aabb_ratio_threshold
          ),
      )
      if str(args.source_pose_mode).lower() == "force":
        result = _run_source_pose_result(
            instruction=model_instruction,
            case=case_for_run,
            catalog=catalog,
            selected_parts=selected_parts,
        )
      else:
        case_planner = planner
        if not bool(getattr(args, "no_physics_guided_grounding", False)):
          case_planner = PhysicsGuidedGroundingPlanner(
              planner,
              shape_library=shape_library,
              config=PhysicsGroundingConfig(
                  input_protocol=str(getattr(args, "input_protocol", "legacy")),
                  max_pairs_per_edge=int(args.physics_grounding_max_pairs),
                  max_pose_candidates_per_pair=int(
                      args.physics_grounding_max_pose_candidates
                  ),
                  edge_timeout_seconds=float(
                      getattr(args, "physics_grounding_edge_timeout_seconds", 60.0)
                      or 60.0
                  ),
                  contact_distance_tolerance=float(
                      args.exact_contact_distance_tolerance
                  ),
                  collision_epsilon=float(args.collision_epsilon),
                  pair_scorer_top_k=int(
                      getattr(args, "interface_pair_top_k", 12) or 12
                  ),
                  micro_retreat_enabled=not bool(
                      getattr(args, "no_physics_micro_retreat", False)
                  ),
                  micro_retreat_max_distance=float(
                      getattr(args, "physics_micro_retreat_max_distance", 2.0)
                      or 2.0
                  ),
                  micro_retreat_exact_timeout_seconds=float(
                      getattr(args, "physics_micro_retreat_timeout_seconds", 3.0)
                      or 3.0
                  ),
                  skip_obround_slot_grounding=bool(
                      getattr(args, "skip_obround_slot_grounding", False)
                  ),
                  exact_contact_rerank_enabled=not bool(
                      getattr(args, "no_physics_exact_contact_rerank", False)
                  ),
                  exact_contact_rerank_tolerance=float(
                      getattr(
                          args,
                          "physics_exact_contact_rerank_tolerance",
                          None,
                      )
                      or args.exact_contact_distance_tolerance
                  ),
                  exact_contact_rerank_timeout_seconds=float(
                      getattr(
                          args,
                          "physics_exact_contact_rerank_timeout_seconds",
                          2.0,
                      )
                      or 2.0
                  ),
              ),
              pair_scorer=interface_pair_scorer,
          )
        pipeline = NeuroSymbolicCadPipeline(
            planner=case_planner,
            solver=solver,
            collision_checker=collision_checker,
            collision_epsilon=float(args.collision_epsilon),
            mate_collision_volume_tolerance=float(
                args.mate_collision_volume_tolerance
            ),
            threaded_interference_collision_tolerance=float(
                getattr(args, "threaded_interference_collision_tolerance", 500.0)
                or 500.0
            ),
        )
        result = pipeline.run(
            instruction=model_instruction,
            catalog=catalog,
            max_iterations=int(args.max_iterations),
        )
      case_result = result
      payload = result.to_dict()
      if benchmark_v2_prepared:
        payload = benchmark_v2_prepared[0].restore_model_output(payload)
      payload["id"] = case_id
      payload["case"] = case_for_run
      payload["benchmark_task"] = task_mode
      payload["selected_parts"] = (
          benchmark_v2_prepared[0].restore_model_output(selected_parts)
          if benchmark_v2_prepared
          else selected_parts
      )
      payload["selected_step_paths"] = selected_step_paths
      payload["runtime_notes"] = retrieval_notes + load_notes + kernel_notes
      payload["input_protocol"] = str(getattr(args, "input_protocol", "legacy"))
      payload["retrieval_context_protocol"] = retrieval_context_protocol
      if protocol_audit:
        payload["protocol_audit"] = protocol_audit
      if retrieval_selection is not None:
        payload["retrieval"] = retrieval_selection.to_dict()
      if retrieval_preflight is not None:
        payload["retrieval_preflight"] = retrieval_preflight
      final_graph = payload.get("final_graph")
      if isinstance(final_graph, dict):
        graph_metadata = final_graph.get("metadata")
        if isinstance(graph_metadata, dict):
          canonical = graph_metadata.get("canonical_mate_graph")
          if canonical is not None:
            payload["canonical_mate_graph"] = canonical
      output_step = None
      output_step_error = None
      if payload.get("success") and result.final_assembly is not None:
        output_step_path = step_output_dir / f"output_{case_id}.step"
        try:
          saved = export_assembly_step(
              assembly=result.final_assembly,
              shape_library=shape_library,
              output_step_path=output_step_path,
          )
          output_step = str(saved.resolve())
        except Exception as exc:  # pylint: disable=broad-except
          output_step_error = f"{type(exc).__name__}:{exc}"
      payload["output_step"] = output_step
      payload["output_step_error"] = output_step_error
      assembly_plan_path = None
      if payload.get("success"):
        plan_payload = _build_assembly_plan_payload(
            case_id=case_id,
            payload=payload,
            output_step=output_step,
        )
        plan_path = assembly_plan_dir / f"assembly_plan_{case_id}.json"
        plan_path.write_text(
            json.dumps(plan_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        assembly_plan_path = str(plan_path.resolve())
      payload["assembly_plan"] = assembly_plan_path
    except Exception as exc:  # pylint: disable=broad-except
      payload = {
          "id": case_id,
          "success": False,
          "instruction": instruction,
          "benchmark_task": task_mode,
          "attempts": [],
          "final_graph": None,
          "final_assembly": None,
          "final_feedback": [f"case_error={type(exc).__name__}:{exc}"],
          "case": case,
          "output_step": None,
          "output_step_error": None,
          "assembly_plan": None,
      }

    case_contact_pairs = case.get("contact_pairs")
    if isinstance(case_contact_pairs, list):
      expected_pairs = []
      for item in case_contact_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
          continue
        expected_pairs.append((str(item[0]), str(item[1])))
      if expected_pairs:
        payload["contact_metrics"] = _evaluate_contact_pairs(
            result_payload=payload,
            expected_pairs=expected_pairs,
        )
        if (
            not bool(getattr(args, "no_exact_contact_validation", False))
            and case_result is not None
            and case_result.final_assembly is not None
            and case_shape_library
        ):
          payload["exact_contact_validation"] = _evaluate_exact_contact_distances(
              assembly=case_result.final_assembly,
              shape_library=case_shape_library,
              expected_pairs=expected_pairs,
              distance_tolerance=float(args.exact_contact_distance_tolerance),
              min_contact_recall=float(args.exact_contact_min_recall),
          )
    retrieval_metrics = _evaluate_retrieval_selection(case=case, result_payload=payload)
    if retrieval_metrics:
      payload["retrieval_metrics"] = retrieval_metrics

    payload["evaluation"] = evaluate_generation_quality(payload)
    payload["self_correction_count"] = max(0, _attempt_count(payload) - 1)
    payload["cumulative_collision_volume_sum"] = _collision_volume_from_payload(payload)
    payload["collision_volume_sum"] = payload["cumulative_collision_volume_sum"]
    payload["blocking_collision_volume_sum"] = _final_feedback_collision_volume(
        payload
    )
    payload["tolerated_collision_volume_sum"] = _tolerated_collision_volume_from_payload(
        payload
    )
    payload["strict_collision_volume_sum"] = (
        float(payload["blocking_collision_volume_sum"])
        + float(payload["tolerated_collision_volume_sum"])
    )
    payload["tolerated_collision_count"] = _tolerated_collision_count_from_payload(
        payload
    )
    validated = derive_validated_success(
        payload,
        tolerated_collision_count=int(payload["tolerated_collision_count"]),
    )
    payload["raw_success"] = bool(validated.get("raw_success", False))
    payload["retrieval_confidence"] = validated.get("retrieval_confidence", {})
    payload["mate_contact_validation"] = validated.get(
        "mate_contact_validation", {}
    )
    payload["contact_coverage"] = validated.get("contact_coverage", {})
    payload["exact_contact_validation"] = validated.get(
        "exact_contact_validation",
        payload.get("exact_contact_validation", {}),
    )
    payload["validation_rejection_reasons"] = validated.get(
        "rejection_reasons", []
    )
    payload["success"] = bool(validated.get("success", False))
    payload["tolerant_success"] = bool(validated.get("tolerant_success", False))
    payload["strict_success"] = bool(validated.get("strict_success", False))
    payload["failure_taxonomy"] = classify_failure(payload)

    result_path = output_dir / f"result_{case_id}.json"
    artifact_payload = payload
    if str(getattr(args, "input_protocol", "legacy")) == "benchmark_v2":
      artifact_payload = public_batch_result(
          payload,
          private_protocol_audit=protocol_audit,
      )
    result_path.write_text(
        json.dumps(artifact_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows.append(
      {
          "id": case_id,
          "benchmark_task": task_mode,
          "success": bool(payload.get("success", False)),
          "tolerant_success": bool(payload.get("tolerant_success", False)),
          "strict_success": bool(payload.get("strict_success", False)),
          "score": payload["evaluation"]["score"],
          "attempts": _attempt_count(payload),
          "self_correction_count": payload["self_correction_count"],
          "collision_volume_sum": payload["collision_volume_sum"],
          "blocking_collision_volume_sum": payload["blocking_collision_volume_sum"],
          "tolerated_collision_volume_sum": payload["tolerated_collision_volume_sum"],
          "contact_recall": (
              payload.get("contact_metrics", {}).get("contact_recall")
          ),
          "contact_precision": (
              payload.get("contact_metrics", {}).get("contact_precision")
          ),
          "retrieval_recall": (
              payload.get("retrieval_metrics", {}).get("retrieval_recall")
          ),
          "retrieval_precision": (
              payload.get("retrieval_metrics", {}).get("retrieval_precision")
          ),
          "retrieval_exact_match": (
              payload.get("retrieval_metrics", {}).get("exact_part_set_match")
          ),
          "frame_grounding_coverage": (
              payload.get("evaluation", {}).get("frame_grounding_coverage")
          ),
          "direct_frame_solve_rate": (
              payload.get("evaluation", {}).get("direct_frame_solve_rate")
          ),
          "axial_fallback_rate": (
              payload.get("evaluation", {}).get("axial_fallback_rate")
          ),
          "failure_category": (
              payload.get("failure_taxonomy", {}).get("category")
          ),
          "exact_contact_recall": (
              payload.get("exact_contact_validation", {}).get("exact_contact_recall")
          ),
          "max_contact_distance": (
              payload.get("exact_contact_validation", {}).get("max_contact_distance")
          ),
          "selected_part_set": _selected_part_set_from_payload(payload),
          "output_step": payload.get("output_step"),
          "assembly_plan": payload.get("assembly_plan"),
          "result_json": str(result_path.resolve()),
      }
    )

  tolerant_solved = [
      row for row in rows if row.get("tolerant_success") is True
  ]
  strict_solved = [row for row in rows if row.get("strict_success") is True]
  task_counts: dict[str, int] = {}
  duplicate_part_set_counts: dict[tuple[str, ...], int] = {}
  contact_recalls = [
      float(row["contact_recall"])
      for row in rows
      if isinstance(row.get("contact_recall"), (int, float))
  ]
  contact_precisions = [
      float(row["contact_precision"])
      for row in rows
      if isinstance(row.get("contact_precision"), (int, float))
  ]
  retrieval_recalls = [
      float(row["retrieval_recall"])
      for row in rows
      if isinstance(row.get("retrieval_recall"), (int, float))
  ]
  retrieval_precisions = [
      float(row["retrieval_precision"])
      for row in rows
      if isinstance(row.get("retrieval_precision"), (int, float))
  ]
  retrieval_exact_matches = [
      row for row in rows if row.get("retrieval_exact_match") is True
  ]
  retrieval_cases = [
      row for row in rows if row.get("retrieval_exact_match") is not None
  ]
  frame_coverages = [
      float(row["frame_grounding_coverage"])
      for row in rows
      if isinstance(row.get("frame_grounding_coverage"), (int, float))
  ]
  direct_frame_rates = [
      float(row["direct_frame_solve_rate"])
      for row in rows
      if isinstance(row.get("direct_frame_solve_rate"), (int, float))
  ]
  axial_fallback_rates = [
      float(row["axial_fallback_rate"])
      for row in rows
      if isinstance(row.get("axial_fallback_rate"), (int, float))
  ]
  for row in rows:
    task_name = str(row.get("benchmark_task") or "unknown").strip() or "unknown"
    task_counts[task_name] = task_counts.get(task_name, 0) + 1
    selected_part_set = row.get("selected_part_set")
    if isinstance(selected_part_set, tuple) and selected_part_set:
      duplicate_part_set_counts[selected_part_set] = (
          duplicate_part_set_counts.get(selected_part_set, 0) + 1
      )
  taxonomy_counts = summarize_failure_taxonomy(rows)
  duplicate_rows = 0
  duplicate_unique_sets = 0
  for count in duplicate_part_set_counts.values():
    if count <= 1:
      continue
    duplicate_rows += count
    duplicate_unique_sets += 1
  retrieval_low_confidence_count = int(
      taxonomy_counts.get("retrieval_low_confidence", 0)
  )
  grounding_failure_count = int(
      taxonomy_counts.get("grounding_failure", 0)
  )
  tolerant_asr = len(tolerant_solved) / max(1, len(rows))
  strict_asr = len(strict_solved) / max(1, len(rows))
  summary = {
      "test_cases_json": (
          None if not args.test_cases_json else str(Path(args.test_cases_json).resolve())
      ),
      "instruction": None if not args.instruction else str(args.instruction),
      "dataset_root": str(dataset_roots[0].resolve()),
      "dataset_roots": [str(path.resolve()) for path in dataset_roots],
      "case_count": len(rows),
      "strict_success_count": len(strict_solved),
      "tolerant_success_count": len(tolerant_solved),
      "strict_ASR": round(strict_asr, 4),
      "tolerant_ASR": round(tolerant_asr, 4),
      "ASR": round(tolerant_asr, 4),
      "validated_strict_ASR": round(strict_asr, 4),
      "validated_tolerant_ASR": round(tolerant_asr, 4),
      "validated_ASR": round(tolerant_asr, 4),
      "mean_score": round(
          sum(float(row["score"]) for row in rows) / max(1, len(rows)), 2
      ),
      "mean_self_correction_count": round(
          sum(float(row["self_correction_count"]) for row in rows)
          / max(1, len(rows)),
          4,
      ),
      "mean_collision_volume_sum": round(
          sum(float(row["collision_volume_sum"]) for row in rows)
          / max(1, len(rows)),
          6,
      ),
      "mean_blocking_collision_volume_sum": round(
          sum(float(row.get("blocking_collision_volume_sum", 0.0)) for row in rows)
          / max(1, len(rows)),
          6,
      ),
      "mean_tolerated_collision_volume_sum": round(
          sum(float(row.get("tolerated_collision_volume_sum", 0.0)) for row in rows)
          / max(1, len(rows)),
          6,
      ),
      "mean_contact_recall": (
          None
          if not contact_recalls
          else round(sum(contact_recalls) / len(contact_recalls), 4)
      ),
      "mean_contact_precision": (
          None
          if not contact_precisions
          else round(sum(contact_precisions) / len(contact_precisions), 4)
      ),
      "mean_retrieval_recall": (
          None
          if not retrieval_recalls
          else round(sum(retrieval_recalls) / len(retrieval_recalls), 4)
      ),
      "mean_retrieval_precision": (
          None
          if not retrieval_precisions
          else round(sum(retrieval_precisions) / len(retrieval_precisions), 4)
      ),
      "retrieval_exact_match_rate": (
          None
          if not retrieval_cases
          else round(len(retrieval_exact_matches) / len(retrieval_cases), 4)
      ),
      "frame_grounding_coverage": (
          None
          if not frame_coverages
          else round(sum(frame_coverages) / len(frame_coverages), 4)
      ),
      "direct_frame_solve_rate": (
          None
          if not direct_frame_rates
          else round(sum(direct_frame_rates) / len(direct_frame_rates), 4)
      ),
      "axial_fallback_rate": (
          None
          if not axial_fallback_rates
          else round(sum(axial_fallback_rates) / len(axial_fallback_rates), 4)
      ),
      "duplicate_part_set_rate": round(
          float(duplicate_rows) / float(max(1, len(rows))),
          4,
      ),
      "duplicate_part_set_unique_count": int(duplicate_unique_sets),
      "retrieval_low_confidence_rate": round(
          float(retrieval_low_confidence_count) / float(max(1, len(rows))),
          4,
      ),
      "grounding_failure_rate": round(
          float(grounding_failure_count) / float(max(1, len(rows))),
          4,
      ),
      "benchmark_task_counts": task_counts,
      "failure_taxonomy": taxonomy_counts,
      "output_dir": str(output_dir.resolve()),
      "assembled_step_dir": str(step_output_dir.resolve()),
  }
  summary_payload = {"summary": summary, "results": rows}
  summary_path = output_dir / "summary.json"
  summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
  return summary_payload


def main() -> None:
  args = build_parser().parse_args()
  summary_payload = run_batch(args)
  print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
  main()
