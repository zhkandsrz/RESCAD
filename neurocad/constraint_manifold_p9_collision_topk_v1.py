"""Non-oracle collision-aware structural top-K for Program 9 only."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neurocad.constraint_manifold_solver_v3 import canonical_sha256
from neurocad.constraint_manifold_solver_v4 import (
    candidate_rank_payload_v4,
    canonical_candidate_payload_v4,
)


P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1 = (
    "constraint_manifold_p9_collision_aware_structural_topk_rank.v1"
)
P9_COLLISION_AWARE_STRUCTURAL_TOPK_SCHEMA_V1 = (
    "constraint_manifold_p9_collision_aware_structural_topk.v1"
)
P9_COLLISION_METRIC_SCHEMA_V1 = "constraint_manifold_p9_body_aabb_collision.v1"
LEGACY_PROTECTION_COUNT_V1 = 8
STRUCTURAL_SELECTION_COUNT_V1 = 8
MAXIMUM_EXACT_CANDIDATES_V1 = 16
COMPACT_TOP_K_V1 = 32
_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY_FRAGMENTS = (
    "source_label", "source_pose", "ground_truth", "gold_pose",
    "row_ordinal",
)


def _reject_oracle_fields(value: Any, *, path: str = "candidate_domain") -> None:
  if isinstance(value, Mapping):
    for key, item in value.items():
      normalized = str(key).lower()
      mate_field = normalized == "mate" or normalized.startswith("mate_")
      if mate_field or any(
          fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS
      ):
        raise ValueError(f"P9 collision ranking forbidden field at {path}.{key}")
      _reject_oracle_fields(item, path=f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _reject_oracle_fields(item, path=f"{path}[{index}]")


def p9_collision_policy_v1() -> dict[str, Any]:
  unsigned = {
      "schema_version": "constraint_manifold_preprocess_policy.p9_collision_topk.v1",
      "ranking_schema": P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1,
      "authorized_program": 9,
      "legacy_protection_slots": LEGACY_PROTECTION_COUNT_V1,
      "structural_selection_slots": STRUCTURAL_SELECTION_COUNT_V1,
      "maximum_exact_candidates": MAXIMUM_EXACT_CANDIDATES_V1,
      "compact_top_k": COMPACT_TOP_K_V1,
      "structural_strata": "canonical_yaw_mod_pi_x_alignment_family",
      "within_stratum_order": (
          "body_frame_aabb_normalized_overlap_ascending_then_legacy_rank_then_key"
      ),
      "structural_schedule": "deterministic_sorted_stratum_round_robin",
      "structural_eligibility": "frozen_v4_analytic_pass_only",
      "remaining_transport_order": "legacy_rank_then_candidate_key",
      "metric_claim": "ranking_proxy_only_never_negative_or_certificate_evidence",
      "oracle_and_external_identity_inputs_forbidden": True,
      "thresholds_changed": False,
      "exact_occ_or_time_budget_changed": False,
      "formal_authorized": False,
      "final_test_touched": False,
  }
  return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


def body_frame_aabb_collision_metrics_p9_v1(
    corners_a: Sequence[Sequence[float]],
    corners_b_current: Sequence[Sequence[float]],
    world_delta_row_major: Sequence[float], *,
    reference_origin_world_mm: Sequence[float],
    reference_rotation_local_to_world: Sequence[Sequence[float]],
) -> dict[str, Any]:
  """Return deterministic target-frame AABB overlap proxy for one placement."""

  first = np.asarray(corners_a, dtype=float)
  second = np.asarray(corners_b_current, dtype=float)
  delta = np.asarray(world_delta_row_major, dtype=float).reshape(4, 4)
  origin = np.asarray(reference_origin_world_mm, dtype=float)
  rotation = np.asarray(reference_rotation_local_to_world, dtype=float)
  if (
      first.shape != (8, 3) or second.shape != (8, 3)
      or origin.shape != (3,) or rotation.shape != (3, 3)
      or not all(np.isfinite(value).all() for value in (
          first, second, delta, origin, rotation,
      ))
  ):
    raise ValueError("P9 collision AABB inputs differ")
  # Fixed-order scalar arithmetic avoids loading an additional BLAS/OpenMP
  # runtime inside the OCC worker and is bitwise deterministic across inputs.
  def moved_point(point: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(
        delta[axis, 0] * point[0] + delta[axis, 1] * point[1]
        + delta[axis, 2] * point[2] + delta[axis, 3]
    ) for axis in range(3))  # type: ignore[return-value]

  def local_point(point: Sequence[float]) -> tuple[float, float, float]:
    shifted = tuple(float(point[axis] - origin[axis]) for axis in range(3))
    return tuple(float(
        shifted[0] * rotation[0, axis]
        + shifted[1] * rotation[1, axis]
        + shifted[2] * rotation[2, axis]
    ) for axis in range(3))  # type: ignore[return-value]

  local_a = tuple(local_point(point) for point in first)
  local_b = tuple(local_point(moved_point(point)) for point in second)
  low_a = tuple(min(point[axis] for point in local_a) for axis in range(3))
  high_a = tuple(max(point[axis] for point in local_a) for axis in range(3))
  low_b = tuple(min(point[axis] for point in local_b) for axis in range(3))
  high_b = tuple(max(point[axis] for point in local_b) for axis in range(3))
  overlap_extent = tuple(max(
      0.0, min(high_a[axis], high_b[axis]) - max(low_a[axis], low_b[axis]),
  ) for axis in range(3))
  volume_a = math.prod(max(0.0, high_a[i] - low_a[i]) for i in range(3))
  volume_b = math.prod(max(0.0, high_b[i] - low_b[i]) for i in range(3))
  overlap = math.prod(overlap_extent)
  denominator = min(volume_a, volume_b)
  normalized = 0.0 if denominator <= 0.0 else min(1.0, overlap / denominator)
  unsigned = {
      "schema_version": P9_COLLISION_METRIC_SCHEMA_V1,
      "body_frame_aabb_intersection_volume_mm3": float(round(overlap, 12)),
      "body_frame_aabb_normalized_overlap": float(round(normalized, 15)),
      "claim": "deterministic_ranking_proxy_only",
  }
  return {**unsigned, "metric_payload_sha256": canonical_sha256(unsigned)}


def _yaw_bucket(candidate: Mapping[str, Any]) -> str:
  raw = candidate.get("yaw_radians")
  if isinstance(raw, bool) or not isinstance(raw, (int, float)):
    raise ValueError("P9 collision yaw differs")
  yaw = float(raw) % math.pi
  if math.isclose(yaw, math.pi, abs_tol=5e-13, rel_tol=0.0):
    yaw = 0.0
  return float(round(yaw, 12)).hex()


def _alignment_families(candidate: Mapping[str, Any]) -> tuple[str, ...]:
  events = candidate.get("alignment_events")
  if not isinstance(events, (list, tuple)) or not events:
    raise ValueError("P9 collision alignment events differ")
  families = sorted({
      str(event.get("kind")) for event in events
      if isinstance(event, Mapping) and isinstance(event.get("kind"), str)
  })
  if not families:
    raise ValueError("P9 collision alignment family differs")
  return tuple(families)


def _proxy(candidate: Mapping[str, Any]) -> float:
  metrics = candidate.get("cheap_metrics")
  if not isinstance(metrics, Mapping):
    raise ValueError("P9 collision metrics differ")
  value = metrics.get("body_frame_aabb_normalized_overlap")
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError("P9 collision normalized overlap differs")
  result = float(value)
  if not math.isfinite(result) or not 0.0 <= result <= 1.0:
    raise ValueError("P9 collision normalized overlap differs")
  return result


def _candidate_strata(candidate: Mapping[str, Any]) -> tuple[str, ...]:
  yaw = _yaw_bucket(candidate)
  return tuple(f"yaw={yaw}|family={family}" for family in _alignment_families(candidate))


def select_p9_collision_aware_topk_v1(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Commit full P9 domain and choose exact slots without oracle information."""

  _reject_oracle_fields(candidate_rows)
  unique: dict[str, dict[str, Any]] = {}
  for value in candidate_rows:
    row = json.loads(json.dumps(value, allow_nan=False))
    if row.get("program_index") != 9:
      raise ValueError("P9 collision candidate program differs")
    key = str(row.get("candidate_key", ""))
    if _SHA256_RE.fullmatch(key) is None:
      raise ValueError("P9 collision candidate key differs")
    _proxy(row); _yaw_bucket(row); _alignment_families(row)
    prior = unique.setdefault(key, row)
    if prior != row:
      raise ValueError("P9 collision duplicate candidate evidence differs")
  canonical_domain = [unique[key] for key in sorted(unique)]
  if not canonical_domain:
    raise ValueError("P9 collision candidate domain is empty")
  legacy = sorted(
      canonical_domain, key=lambda row: candidate_rank_payload_v4(9, row),
  )
  protected = legacy[:LEGACY_PROTECTION_COUNT_V1]
  protected_keys = {row["candidate_key"] for row in protected}
  analytic = [
      row for row in legacy
      if row["candidate_key"] not in protected_keys
      and candidate_rank_payload_v4(9, row)[0] == 0
  ]
  buckets: dict[str, list[dict[str, Any]]] = {}
  for row in analytic:
    for stratum in _candidate_strata(row):
      buckets.setdefault(stratum, []).append(row)
  for rows in buckets.values():
    rows.sort(key=lambda row: (
        round(_proxy(row), 15), candidate_rank_payload_v4(9, row),
        row["candidate_key"],
    ))
  selected = list(protected)
  selected_keys = set(protected_keys)
  structural: list[dict[str, Any]] = []
  bucket_offsets = {key: 0 for key in buckets}
  while len(structural) < STRUCTURAL_SELECTION_COUNT_V1:
    changed = False
    for stratum in sorted(buckets):
      rows = buckets[stratum]
      offset = bucket_offsets[stratum]
      while offset < len(rows) and rows[offset]["candidate_key"] in selected_keys:
        offset += 1
      bucket_offsets[stratum] = offset
      if offset >= len(rows):
        continue
      row = rows[offset]
      bucket_offsets[stratum] = offset + 1
      structural.append(row); selected.append(row)
      selected_keys.add(row["candidate_key"]); changed = True
      if len(structural) == STRUCTURAL_SELECTION_COUNT_V1:
        break
    if not changed:
      break
  if len(structural) < STRUCTURAL_SELECTION_COUNT_V1:
    fill = sorted(analytic, key=lambda row: (
        round(_proxy(row), 15), candidate_rank_payload_v4(9, row),
        row["candidate_key"],
    ))
    for row in fill:
      if row["candidate_key"] not in selected_keys:
        structural.append(row); selected.append(row)
        selected_keys.add(row["candidate_key"])
        if len(structural) == STRUCTURAL_SELECTION_COUNT_V1:
          break
  exact_keys = [row["candidate_key"] for row in selected]
  remaining = [row for row in legacy if row["candidate_key"] not in selected_keys]
  compact = (selected + remaining)[:COMPACT_TOP_K_V1]
  covered = sorted({
      stratum for row in structural for stratum in _candidate_strata(row)
  })
  domain_sha = canonical_sha256([
      canonical_candidate_payload_v4(9, row) for row in canonical_domain
  ])
  selection_unsigned = {
      "schema_version": P9_COLLISION_AWARE_STRUCTURAL_TOPK_SCHEMA_V1,
      "ranking_schema": P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1,
      "policy_payload_sha256": p9_collision_policy_v1()["policy_payload_sha256"],
      "candidate_domain_count": len(canonical_domain),
      "candidate_domain_sha256": domain_sha,
      "legacy_protected_candidate_keys": [
          row["candidate_key"] for row in protected
      ],
      "structural_selected_candidate_keys": [
          row["candidate_key"] for row in structural
      ],
      "exact_selected_candidate_keys": exact_keys,
      "compact_top32_candidate_keys": [row["candidate_key"] for row in compact],
      "covered_yaw_alignment_strata": covered,
      "legacy_protection_count": len(protected),
      "structural_selection_count": len(structural),
      "exact_selected_count": len(exact_keys),
      "compact_selected_count": len(compact),
      "maximum_exact_candidates": MAXIMUM_EXACT_CANDIDATES_V1,
      "maximum_occ_calls": 32,
      "total_wall_seconds": 30.0,
      "oracle_inputs_absent": True,
      "formal_authorized": False,
      "final_test_touched": False,
  }
  selection = {
      **selection_unsigned,
      "selection_payload_sha256": canonical_sha256(selection_unsigned),
  }
  return {
      "ranking_schema": P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1,
      "candidate_domain_count": len(canonical_domain),
      "candidate_domain_sha256": domain_sha,
      "compact_top_k": COMPACT_TOP_K_V1,
      "compact_topk_payload_sha256": canonical_sha256(compact),
      "cheap_geometry_candidates": compact,
      "collision_selection": selection,
  }


def verify_p9_collision_selection_v1(
    *, candidate_domain_count: int, candidate_domain_sha256: str,
    compact_candidates: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> None:
  _reject_oracle_fields(selection)
  unsigned = dict(selection); digest = unsigned.pop("selection_payload_sha256", None)
  compact_keys = [row.get("candidate_key") for row in compact_candidates]
  protected = selection.get("legacy_protected_candidate_keys")
  structural = selection.get("structural_selected_candidate_keys")
  exact = selection.get("exact_selected_candidate_keys")
  if not all((
      digest == canonical_sha256(unsigned),
      selection.get("schema_version") == P9_COLLISION_AWARE_STRUCTURAL_TOPK_SCHEMA_V1,
      selection.get("ranking_schema")
      == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1,
      selection.get("policy_payload_sha256")
      == p9_collision_policy_v1()["policy_payload_sha256"],
      selection.get("candidate_domain_count") == candidate_domain_count,
      selection.get("candidate_domain_sha256") == candidate_domain_sha256,
      selection.get("compact_top32_candidate_keys") == compact_keys,
      isinstance(protected, list), isinstance(structural, list),
      isinstance(exact, list), exact == protected + structural,
      compact_keys[:len(exact)] == exact,
      len(protected) <= LEGACY_PROTECTION_COUNT_V1,
      len(structural) <= STRUCTURAL_SELECTION_COUNT_V1,
      len(exact) <= MAXIMUM_EXACT_CANDIDATES_V1,
      selection.get("exact_selected_count") == len(exact),
      selection.get("compact_selected_count") == len(compact_keys),
      selection.get("maximum_exact_candidates") == 16,
      selection.get("maximum_occ_calls") == 32,
      selection.get("total_wall_seconds") == 30.0,
      selection.get("oracle_inputs_absent") is True,
      selection.get("formal_authorized") is False,
      selection.get("final_test_touched") is False,
      len(set(compact_keys)) == len(compact_keys),
  )):
    raise ValueError("P9 collision selection commitment differs")


__all__ = [
    "COMPACT_TOP_K_V1", "LEGACY_PROTECTION_COUNT_V1",
    "MAXIMUM_EXACT_CANDIDATES_V1",
    "P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1",
    "P9_COLLISION_AWARE_STRUCTURAL_TOPK_SCHEMA_V1",
    "STRUCTURAL_SELECTION_COUNT_V1", "body_frame_aabb_collision_metrics_p9_v1",
    "p9_collision_policy_v1", "select_p9_collision_aware_topk_v1",
    "verify_p9_collision_selection_v1",
]
