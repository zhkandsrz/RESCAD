"""Two-stage conditional-feasibility solver for executable programs 9 and 11.

V4 intentionally has no compatibility path for V1--V3 requests, receipts, or
certificates.  A killable preprocessing child supplies a sanitized, replayable
geometry candidate authority.  This module applies the frozen cheap ranking,
executes at most sixteen candidates (two OCC observations each), and issues a
certificate whose claim is *conditional physical feasibility* of the selected
face/program hypothesis.  Intendedness remains an outer evaluation decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import secrets
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
from weakref import WeakKeyDictionary

import numpy as np


SCHEMA_VERSION = "constraint_manifold_solver.v4"
POLICY_SCHEMA_VERSION = "constraint_manifold_frozen_policy.v4"
P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1 = (
    "constraint_manifold_p9_collision_aware_structural_topk_rank.v1"
)
P9_COLLISION_AWARE_STRUCTURAL_TOPK_POLICY_SCHEMA_VERSION_V1 = (
    "constraint_manifold_frozen_policy.p9_collision_structural_topk.v1"
)
P11_BODY_AABB_RANK_SCHEMA_V1 = "constraint_manifold_p11_body_aabb_rank.v1"
P11_BODY_AABB_POLICY_SCHEMA_VERSION_V1 = (
    "constraint_manifold_frozen_policy.p11_body_aabb.v1"
)
P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1 = (
    "constraint_manifold_p11_stratified_diversity_rank.v1"
)
P11_STRATIFIED_DIVERSITY_POLICY_SCHEMA_VERSION_V1 = (
    "constraint_manifold_frozen_policy.p11_stratified_diversity.v1"
)
P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2 = (
    "constraint_manifold_p11_factorized_structural_topk.v2"
)
P11_FACTORIZED_STRUCTURAL_TOPK_POLICY_SCHEMA_VERSION_V2 = (
    "constraint_manifold_frozen_policy.p11_factorized_structural_topk.v2"
)
CHILD_RECEIPT_SCHEMA_VERSION = "constraint_manifold_exact_child_receipt.v4"
BATCH_CANDIDATE_RECEIPT_SCHEMA_VERSION_V8 = (
    "constraint_manifold_exact_batch_candidate_receipt.v8"
)
BATCH_CHILD_RECEIPT_SCHEMA_VERSION_V8 = "constraint_manifold_exact_batch_child_receipt.v8"
BATCH_CANDIDATE_EVIDENCE_SCHEMA_VERSION_V8 = (
    "constraint_manifold_exact_batch_candidate_evidence.v8"
)
CERTIFICATE_SCHEMA_VERSION = "constraint_manifold_certificate.v4"
COUNTEREVIDENCE_SCHEMA_VERSION = "constraint_manifold_counterevidence.v4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CERTIFICATE_TOKEN = object()
_COUNTEREVIDENCE_TOKEN = object()
_RADIUS_GEOMETRIC_EVIDENCE_INVALID = object()


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
      allow_nan=False,
  ).encode("utf-8")).hexdigest()


def _deep_freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
  if isinstance(value, (list, tuple)):
    return tuple(_deep_freeze(item) for item in value)
  return value


def _deep_thaw(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(key): _deep_thaw(item) for key, item in value.items()}
  if isinstance(value, tuple):
    return [_deep_thaw(item) for item in value]
  return value


def _require_sha256(value: str, *, label: str) -> str:
  if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
    raise ValueError(f"{label} must be lowercase hexadecimal SHA-256")
  return value


@dataclass(frozen=True, slots=True)
class ConstraintManifoldThresholdsV4:
  plane_normal_error_degrees: float = 1e-5
  plane_normal_gap_mm: float = 1e-6
  cylinder_axis_error_degrees: float = 1e-5
  cylinder_axis_distance_mm: float = 1e-6
  selected_face_clearance_mm: float = 0.1
  whole_common_volume_mm3: float = 1e-7
  minimum_plane_overlap_mm2: float = 1e-4
  minimum_cylinder_axial_overlap_mm: float = 1e-3
  minimum_cylinder_angular_overlap_radians: float = 1e-3
  minimum_trim_overlap_ratio: float = 0.05
  maximum_radial_clearance_mm: float = 0.1


@dataclass(frozen=True, slots=True)
class ConstraintManifoldBudgetV4:
  maximum_cheap_candidates: int = 250_000
  maximum_exact_candidates: int = 16
  maximum_occ_calls: int = 32
  total_wall_seconds: float = 30.0
  exact_child_seconds: float = 8.0


class FrozenConstraintManifoldPolicyV4:
  __slots__ = ()

  @property
  def thresholds(self) -> ConstraintManifoldThresholdsV4:
    return ConstraintManifoldThresholdsV4()

  @property
  def budget(self) -> ConstraintManifoldBudgetV4:
    return ConstraintManifoldBudgetV4()

  def payload(self) -> dict[str, Any]:
    unsigned = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "authorized_programs": {
            "9": "opposed_plane_contact_with_free_tangent_translation_and_edge_yaw.v4",
            "11": "coaxial_shaft_bore_with_free_axis_translation_and_periodic_yaw.v4",
        },
        "cheap_domain": (
            "canonical trim vertex/edge-midpoint/loop-centroid and adjacent-shared-edge "
            "landmark pairs crossed with deterministic boundary/shared-edge yaw pairs"
        ),
        "cheap_score_order": [
            "analytic_pre_occ_pass_descending",
            "body_projected_aabb_separation_descending",
            "body_projected_aabb_overlap_upper_ascending",
            "trim_overlap_ratio_lower_descending",
            "trim_overlap_lower_descending",
            "landmark_provenance_tier_ascending_equal_geometry_only",
            "fixed_displacement_ascending",
            "candidate_key_ascending",
        ],
        "thresholds": asdict(self.thresholds),
        "budget": asdict(self.budget),
        "claim": "selected_face_program_conditional_physical_feasibility_only",
        "non_gold_policy": "accepted_is_alternate_feasible_not_certificate_negative",
        "bounded_rejection_policy": "unknown_not_physical_negative",
        "source_or_gold_inputs_forbidden": True,
    }
    return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4 = FrozenConstraintManifoldPolicyV4()


class FrozenConstraintManifoldPolicyP9CollisionStructuralTopkV1(
    FrozenConstraintManifoldPolicyV4
):
  """P9-only non-oracle rank override with unchanged thresholds and budgets."""

  __slots__ = ()

  def payload(self) -> dict[str, Any]:
    unsigned = dict(OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.payload())
    unsigned.pop("policy_payload_sha256")
    unsigned["schema_version"] = (
        P9_COLLISION_AWARE_STRUCTURAL_TOPK_POLICY_SCHEMA_VERSION_V1
    )
    unsigned["ranking_schema"] = (
        P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
    )
    unsigned["cheap_score_order"] = [
        "legacy_rank_first_8_protection_slots",
        "analytic_pass_yaw_alignment_family_round_robin_next_8",
        "body_frame_aabb_normalized_overlap_ascending_within_stratum",
        "legacy_rank_remaining_transport",
        "candidate_key_ascending_final_tie_break",
    ]
    unsigned["transport_order"] = "worker_committed_p9_collision_structural_order"
    unsigned["source_or_gold_inputs_forbidden"] = True
    unsigned["row_ordinal_input_forbidden"] = True
    return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


OFFICIAL_P9_COLLISION_STRUCTURAL_TOPK_POLICY_V1 = (
    FrozenConstraintManifoldPolicyP9CollisionStructuralTopkV1()
)


class FrozenConstraintManifoldPolicyP11BodyAabbV1(FrozenConstraintManifoldPolicyV4):
  """Explicit p11 rank override; thresholds, budget, and claims remain V4."""

  __slots__ = ()

  def payload(self) -> dict[str, Any]:
    unsigned = dict(OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.payload())
    unsigned.pop("policy_payload_sha256")
    unsigned["schema_version"] = P11_BODY_AABB_POLICY_SCHEMA_VERSION_V1
    unsigned["ranking_schema"] = P11_BODY_AABB_RANK_SCHEMA_V1
    unsigned["cheap_score_order"] = [
        "analytic_pre_occ_pass_descending",
        "body_projected_aabb_separation_descending",
        "body_projected_aabb_overlap_upper_ascending",
        "body_frame_aabb_overlap_upper_mm3_ascending_heuristic_only",
        "trim_overlap_ratio_lower_descending",
        "trim_overlap_lower_descending",
        "landmark_provenance_tier_ascending_equal_geometry_only",
        "fixed_displacement_ascending",
        "candidate_key_ascending",
    ]
    unsigned["body_aabb_overlap_claim"] = (
        "ranking_heuristic_only_never_negative_or_certificate_evidence"
    )
    return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


OFFICIAL_P11_BODY_AABB_POLICY_V1 = FrozenConstraintManifoldPolicyP11BodyAabbV1()


class FrozenConstraintManifoldPolicyP11StratifiedDiversityV1(
    FrozenConstraintManifoldPolicyV4
):
  """Non-oracle p11 stratum coverage with unchanged exact/OCC budgets."""

  __slots__ = ()

  def payload(self) -> dict[str, Any]:
    unsigned = dict(OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.payload())
    unsigned.pop("policy_payload_sha256")
    unsigned["schema_version"] = P11_STRATIFIED_DIVERSITY_POLICY_SCHEMA_VERSION_V1
    unsigned["ranking_schema"] = P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1
    unsigned["cheap_score_order"] = [
        "stage_a_sign_yaw_zero_pi_baseline_anchor_coverage",
        "stage_b_sign_offset_family_minimum_coverage",
        "deterministic_yaw_bucket_round_robin",
        "legacy_cheap_rank_within_yaw_bucket",
        "candidate_key_ascending_final_tie_break",
    ]
    unsigned["stratum_inputs"] = [
        "candidate_normal_sign", "candidate_canonical_yaw_bucket",
        "candidate_offset_event_family",
    ]
    unsigned["transport_order"] = "worker_committed_stratified_order"
    unsigned["source_or_gold_inputs_forbidden"] = True
    return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


OFFICIAL_P11_STRATIFIED_DIVERSITY_POLICY_V1 = (
    FrozenConstraintManifoldPolicyP11StratifiedDiversityV1()
)


class FrozenConstraintManifoldPolicyP11FactorizedStructuralTopkV2(
    FrozenConstraintManifoldPolicyV4
):
  """Factor-level non-oracle top32 with unchanged exact/OCC budgets."""

  __slots__ = ()

  def payload(self) -> dict[str, Any]:
    unsigned = dict(OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.payload())
    unsigned.pop("policy_payload_sha256")
    unsigned["schema_version"] = (
        P11_FACTORIZED_STRUCTURAL_TOPK_POLICY_SCHEMA_VERSION_V2
    )
    unsigned["ranking_schema"] = P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2
    unsigned["cheap_score_order"] = [
        "stage_a_sign_yaw_zero_pi_baseline_factor_anchors",
        "stage_b_sign_offset_family_factor_minimum_coverage",
        "deterministic_factor_yaw_round_robin",
        "factor_landmark_provenance_tier_ascending",
        "factor_fixed_displacement_ascending",
        "selected_candidate_identity_ascending_final_tie_break",
    ]
    unsigned["factor_domain"] = (
        "normal_sign_join_canonical_yaw_x_axial_offset_without_cartesian_expansion"
    )
    unsigned["transport_order"] = "worker_committed_factor_membership_order"
    unsigned["materialization_budget"] = 32
    unsigned["source_or_gold_inputs_forbidden"] = True
    return {**unsigned, "policy_payload_sha256": canonical_sha256(unsigned)}


OFFICIAL_P11_FACTORIZED_STRUCTURAL_TOPK_POLICY_V2 = (
    FrozenConstraintManifoldPolicyP11FactorizedStructuralTopkV2()
)


def _official_policy_for_authenticated_candidate_payload_v1(
    program_index: int, authority_payload: Mapping[str, Any],
) -> FrozenConstraintManifoldPolicyV4:
  """Select the only policy authorized by a candidate authority's rank schema."""

  if type(program_index) is not int or program_index not in {9, 11}:
    raise ValueError("V4 certificate program differs")
  authority_program = authority_payload.get("program_index")
  if type(authority_program) is not int or authority_program != program_index:
    raise ValueError("V4 certificate preprocessing program differs")
  if "ranking_schema" not in authority_payload:
    if program_index == 9:
      return OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4
    raise ValueError("V4 candidate ranking schema is required for program 11")
  ranking_schema = authority_payload["ranking_schema"]
  if (
      program_index == 9
      and ranking_schema == P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
  ):
    return OFFICIAL_P9_COLLISION_STRUCTURAL_TOPK_POLICY_V1
  if program_index != 11:
    raise ValueError("V4 candidate ranking schema/program mismatch")
  if ranking_schema == P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1:
    return OFFICIAL_P11_STRATIFIED_DIVERSITY_POLICY_V1
  if ranking_schema == P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2:
    return OFFICIAL_P11_FACTORIZED_STRUCTURAL_TOPK_POLICY_V2
  if ranking_schema == P11_BODY_AABB_RANK_SCHEMA_V1:
    return OFFICIAL_P11_BODY_AABB_POLICY_V1
  raise ValueError("V4 candidate ranking schema differs")


class SanitizedCandidateAuthorityV4(Protocol):
  """Structural boundary implemented by the killable preprocess supervisor."""

  @property
  def sha256(self) -> str: ...

  def payload(self) -> Mapping[str, Any]: ...

  def revalidate(self) -> None: ...

  def authenticated_snapshot(self) -> tuple[str, Mapping[str, Any]]: ...


def _authenticated_candidate_authority_snapshot_v1(
    authority: SanitizedCandidateAuthorityV4,
) -> tuple[str, Mapping[str, Any]]:
  snapshotter = getattr(authority, "authenticated_snapshot", None)
  if callable(snapshotter):
    snapshot = snapshotter()
    if not isinstance(snapshot, tuple) or len(snapshot) != 2:
      raise ValueError("V4 candidate authority snapshot differs")
    authority_sha256, payload = snapshot
  else:
    # Compatibility for test/provisional authorities; production authorities
    # implement the one-pass snapshot API below.
    authority.revalidate()
    authority_sha256, payload = authority.sha256, authority.payload()
  _require_sha256(authority_sha256, label="candidate authority snapshot")
  if not isinstance(payload, Mapping):
    raise ValueError("V4 candidate authority snapshot payload differs")
  return authority_sha256, payload


@dataclass(frozen=True, slots=True)
class ConstraintManifoldRequestV4:
  query_id: str
  program_index: int
  child_world_row_major: tuple[float, ...]
  identity_authority_sha256: str
  selected_face_authority_sha256: str
  candidate_authority: SanitizedCandidateAuthorityV4
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != SCHEMA_VERSION or not self.query_id:
      raise ValueError("V4 request identity differs")
    if self.program_index not in {9, 11}:
      raise ValueError("V4 authorizes only finite programs 9 and 11")
    _require_sha256(self.identity_authority_sha256, label="identity authority")
    _require_sha256(self.selected_face_authority_sha256, label="selected face authority")
    matrix = np.asarray(self.child_world_row_major, dtype=float)
    if matrix.shape != (16,) or not np.isfinite(matrix).all():
      raise ValueError("V4 child transform differs")
    matrix = matrix.reshape(4, 4)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10):
      raise ValueError("V4 child transform is not homogeneous")
    _authority_sha256, payload = _authenticated_candidate_authority_snapshot_v1(
        self.candidate_authority,
    )
    if int(payload.get("program_index", -1)) != self.program_index:
      raise ValueError("preprocessing program differs")
    if payload.get("oracle_inputs_absent") is not True:
      raise ValueError("preprocessing authority did not prove oracle inputs absent")

  @property
  def policy(self) -> FrozenConstraintManifoldPolicyV4:
    _authority_sha256, payload = _authenticated_candidate_authority_snapshot_v1(
        self.candidate_authority,
    )
    return _official_policy_for_authenticated_candidate_payload_v1(
        self.program_index, payload,
    )

  def _payload_from_authenticated_snapshot(
      self, snapshot: tuple[str, Mapping[str, Any]],
  ) -> dict[str, Any]:
    authority_sha256, authority_payload = snapshot
    policy = _official_policy_for_authenticated_candidate_payload_v1(
        self.program_index, authority_payload,
    )
    return {
        "schema_version": self.schema_version,
        "query_id": self.query_id,
        "program_index": self.program_index,
        "child_world_row_major": list(self.child_world_row_major),
        "identity_authority_sha256": self.identity_authority_sha256,
        "selected_face_authority_sha256": self.selected_face_authority_sha256,
        "candidate_authority_sha256": authority_sha256,
        "frozen_policy": policy.payload(),
    }

  def payload(self) -> dict[str, Any]:
    return self._payload_from_authenticated_snapshot(
        _authenticated_candidate_authority_snapshot_v1(self.candidate_authority),
    )


@dataclass(frozen=True, slots=True)
class ManifoldCandidateV4:
  candidate_key: str
  program_index: int
  world_delta_row_major: tuple[float, ...]
  child_world_row_major: tuple[float, ...]
  cheap_metric_items: tuple[tuple[str, float], ...]
  grammar_tags: tuple[str, ...]

  def __post_init__(self) -> None:
    _require_sha256(self.candidate_key, label="candidate key")
    if self.program_index not in {9, 11}:
      raise ValueError("candidate program differs")
    for label, values in (
        ("world delta", self.world_delta_row_major),
        ("child world", self.child_world_row_major),
    ):
      array = np.asarray(values, dtype=float)
      if array.shape != (16,) or not np.isfinite(array).all():
        raise ValueError(f"candidate {label} differs")
    metrics = dict(self.cheap_metric_items)
    required = {
        "trim_overlap_lower_bound", "trim_overlap_ratio_lower_bound",
        "body_projected_aabb_separation_mm",
        "body_projected_aabb_overlap_upper_mm2", "fixed_displacement_mm",
    }
    if not required.issubset(metrics) or not all(math.isfinite(float(v)) for v in metrics.values()):
      raise ValueError("candidate cheap metrics differ")

  @property
  def cheap_metrics(self) -> Mapping[str, float]:
    return MappingProxyType(dict(self.cheap_metric_items))

  def payload(self) -> dict[str, Any]:
    return {
        "candidate_key": self.candidate_key,
        "program_index": self.program_index,
        "world_delta_row_major": list(self.world_delta_row_major),
        "child_world_row_major": list(self.child_world_row_major),
        "cheap_metrics": dict(self.cheap_metric_items),
        "grammar_tags": list(self.grammar_tags),
    }


def _candidate_from_payload(value: Mapping[str, Any], program_index: int) -> ManifoldCandidateV4:
  grammar_tags = {str(row) for row in value.get("grammar_tags", ())}
  for event in value.get("alignment_events", ()):
    if isinstance(event, Mapping) and isinstance(event.get("kind"), str):
      grammar_tags.add(f"alignment:{event['kind']}")
  unsigned = {
      "program_index": int(value.get("program_index", program_index)),
      "world_delta_row_major": [float(row) for row in value["world_delta_row_major"]],
      "child_world_row_major": [float(row) for row in value["child_world_row_major"]],
      "cheap_metrics": {str(k): float(v) for k, v in value["cheap_metrics"].items()},
      "grammar_tags": sorted(grammar_tags),
  }
  observed = str(value.get("candidate_key", ""))
  # Candidate identity is defined only by rigid placement and program.  Tags
  # and scores are replayed evidence, not alternative identities.
  expected = canonical_sha256({
      "schema_version": "constraint_manifold_candidate_identity.v4",
      "program_index": unsigned["program_index"],
      "world_delta_row_major": unsigned["world_delta_row_major"],
      "child_world_row_major": unsigned["child_world_row_major"],
  })
  if observed != expected:
    raise ValueError("candidate key differs from canonical placement")
  return ManifoldCandidateV4(
      candidate_key=observed, program_index=unsigned["program_index"],
      world_delta_row_major=tuple(unsigned["world_delta_row_major"]),
      child_world_row_major=tuple(unsigned["child_world_row_major"]),
      cheap_metric_items=tuple(sorted(unsigned["cheap_metrics"].items())),
      grammar_tags=tuple(unsigned["grammar_tags"]),
  )


def _analytic_reasons(request: ConstraintManifoldRequestV4,
                      candidate: ManifoldCandidateV4) -> tuple[str, ...]:
  return _analytic_reasons_for_program_v4(request.program_index, candidate)


def _analytic_reasons_for_program_v4(
    program_index: int, candidate: ManifoldCandidateV4,
) -> tuple[str, ...]:
  metrics = candidate.cheap_metrics
  thresholds = OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.thresholds
  reasons: list[str] = []
  if program_index == 9:
    if float(metrics.get("normal_error_degrees", math.inf)) > thresholds.plane_normal_error_degrees:
      reasons.append("plane_normal_error")
    if float(metrics.get("normal_gap_mm", math.inf)) > thresholds.plane_normal_gap_mm:
      reasons.append("plane_normal_gap")
    if float(metrics["trim_overlap_lower_bound"]) < thresholds.minimum_plane_overlap_mm2:
      reasons.append("plane_trim_overlap_lower_bound")
  else:
    if float(metrics.get("axis_error_degrees", math.inf)) > thresholds.cylinder_axis_error_degrees:
      reasons.append("cylinder_axis_error")
    if float(metrics.get("radial_axis_distance_mm", math.inf)) > thresholds.cylinder_axis_distance_mm:
      reasons.append("cylinder_axis_distance")
    if float(metrics.get("axial_overlap_mm", 0.0)) < thresholds.minimum_cylinder_axial_overlap_mm:
      reasons.append("cylinder_axial_overlap")
    if float(metrics.get("angular_overlap_radians", 0.0)) < thresholds.minimum_cylinder_angular_overlap_radians:
      reasons.append("cylinder_angular_overlap")
  if float(metrics["trim_overlap_ratio_lower_bound"]) < thresholds.minimum_trim_overlap_ratio:
    reasons.append("trim_overlap_ratio_lower_bound")
  return tuple(sorted(set(reasons)))


def _candidate_rank(request: ConstraintManifoldRequestV4,
                    candidate: ManifoldCandidateV4) -> tuple[Any, ...]:
  return candidate_rank_payload_v4(request.program_index, candidate.payload())


def candidate_rank_payload_v4(
    program_index: int, candidate_payload: Mapping[str, Any],
) -> tuple[Any, ...]:
  """Frozen cheap rank usable by both the worker and parent solver.

  The worker applies this exact order before truncating IPC to the first 32
  candidates.  Keeping the rank here prevents the compact transport from
  silently changing the solver proposal order.
  """

  candidate = _candidate_from_payload(candidate_payload, program_index)
  metrics = candidate.cheap_metrics
  return (
      int(bool(_analytic_reasons_for_program_v4(program_index, candidate))),
      -round(float(metrics["body_projected_aabb_separation_mm"]), 12),
      round(float(metrics["body_projected_aabb_overlap_upper_mm2"]), 9),
      -round(float(metrics["trim_overlap_ratio_lower_bound"]), 12),
      -round(float(metrics["trim_overlap_lower_bound"]), 9),
      _landmark_provenance_tier(candidate),
      round(float(metrics["fixed_displacement_mm"]), 9),
      candidate.candidate_key,
  )


def candidate_rank_payload_p11_body_aabb_v1(
    candidate_payload: Mapping[str, Any],
) -> tuple[Any, ...]:
  """Rank p11 collision risk without changing candidate identity or claims."""

  candidate = _candidate_from_payload(candidate_payload, 11)
  metrics = candidate.cheap_metrics
  overlap_3d = float(metrics.get("body_frame_aabb_overlap_upper_mm3", math.nan))
  if not math.isfinite(overlap_3d) or overlap_3d < 0.0:
    raise ValueError("p11 body-frame AABB overlap heuristic differs")
  return (
      int(bool(_analytic_reasons_for_program_v4(11, candidate))),
      -round(float(metrics["body_projected_aabb_separation_mm"]), 12),
      round(float(metrics["body_projected_aabb_overlap_upper_mm2"]), 9),
      round(overlap_3d, 9),
      -round(float(metrics["trim_overlap_ratio_lower_bound"]), 12),
      -round(float(metrics["trim_overlap_lower_bound"]), 9),
      _landmark_provenance_tier(candidate),
      round(float(metrics["fixed_displacement_mm"]), 9),
      candidate.candidate_key,
  )


def _candidate_rank_for_request_v1(
    request: ConstraintManifoldRequestV4, candidate: ManifoldCandidateV4,
) -> tuple[Any, ...]:
  return _candidate_rank_for_policy_v1(request.program_index, request.policy, candidate)


def _candidate_rank_for_policy_v1(
    program_index: int, policy: FrozenConstraintManifoldPolicyV4,
    candidate: ManifoldCandidateV4,
) -> tuple[Any, ...]:
  if policy is OFFICIAL_P11_BODY_AABB_POLICY_V1:
    return candidate_rank_payload_p11_body_aabb_v1(candidate.payload())
  return candidate_rank_payload_v4(program_index, candidate.payload())


def canonical_candidate_payload_v4(
    program_index: int, candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
  """Return the exact legacy V4 domain-commitment representation."""

  return _candidate_from_payload(candidate_payload, program_index).payload()


def _landmark_provenance_tier(candidate: ManifoldCandidateV4) -> int:
  """Prefer exact B-Rep landmarks only after all geometric scores tie."""

  kinds = {
      tag.removeprefix("alignment:") for tag in candidate.grammar_tags
      if tag.startswith("alignment:")
  }
  if kinds & {
      "adjacent_shared_edge_vertices_local_2d",
      "adjacent_shared_edge_midpoints_local_2d",
  }:
    return 0
  if kinds & {
      "trim_vertices_local_2d", "trim_edge_midpoints_local_2d",
      "loop_centroids_local_2d",
  }:
    return 1
  if kinds & {
      "triangle_centroid", "lower_extrema", "upper_extrema", "bbox_center",
      "interval_center", "lower_endpoint", "upper_endpoint", "opposed_endpoint",
  }:
    return 2
  return 3


def generate_and_rank_candidates_v4(
    request: ConstraintManifoldRequestV4,
    *, _authenticated_snapshot: tuple[str, Mapping[str, Any]] | None = None,
) -> tuple[tuple[ManifoldCandidateV4, ...], tuple[ManifoldCandidateV4, ...]]:
  snapshot = (
      _authenticated_snapshot
      if _authenticated_snapshot is not None
      else _authenticated_candidate_authority_snapshot_v1(request.candidate_authority)
  )
  _authority_sha256, authority = snapshot
  policy = _official_policy_for_authenticated_candidate_payload_v1(
      request.program_index, authority,
  )
  raw = authority.get("cheap_geometry_candidates")
  if not isinstance(raw, (tuple, list)) or not raw:
    raise ValueError("V4 cheap candidate domain is empty")
  if len(raw) > policy.budget.maximum_cheap_candidates:
    raise ValueError("V4 cheap candidate domain exceeds frozen budget")
  unique: dict[str, ManifoldCandidateV4] = {}
  for value in raw:
    if not isinstance(value, Mapping):
      raise ValueError("V4 cheap candidate row differs")
    row = _candidate_from_payload(value, request.program_index)
    if row.program_index != request.program_index:
      raise ValueError("V4 cheap candidate program differs")
    prior = unique.setdefault(row.candidate_key, row)
    if prior.payload() != row.payload():
      raise ValueError("V4 duplicate candidate key has divergent evidence")
  compact = (
      tuple(unique.values())
      if authority.get("ranking_schema") in {
          P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1,
          P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1,
          P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2,
      }
      else tuple(sorted(
          unique.values(), key=lambda row: _candidate_rank_for_policy_v1(
              request.program_index, policy, row,
          ),
      ))
  )
  if len(compact) > 32:
    raise ValueError("V4 compact candidate transport exceeds top32")
  if authority.get("ranking_schema") == (
      P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1
  ):
    selection = authority.get("collision_selection")
    if not isinstance(selection, Mapping):
      raise ValueError("P9 collision selection is absent")
    exact_keys = selection.get("exact_selected_candidate_keys")
    if (
        not isinstance(exact_keys, (list, tuple))
        or len(exact_keys) > policy.budget.maximum_exact_candidates
        or list(exact_keys) != [
            row.candidate_key for row in compact[:len(exact_keys)]
        ]
    ):
      raise ValueError("P9 collision exact selection differs")
    exact = compact[:len(exact_keys)]
  else:
    exact = compact[:policy.budget.maximum_exact_candidates]
  return compact, exact


def _committed_domain_count(authority_payload: Mapping[str, Any]) -> int:
  value = authority_payload.get("candidate_domain_count")
  if value is None:
    # Test-only structural authorities from the pre-compact V4 contract remain
    # usable; supervisor-issued authorities always carry the committed count.
    value = len(authority_payload.get("cheap_geometry_candidates", ()))
  if type(value) is not int or value <= 0:
    raise ValueError("V4 candidate domain count differs")
  if value > OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4.budget.maximum_cheap_candidates:
    raise ValueError("V4 cheap candidate domain exceeds frozen budget")
  return value


def _committed_domain_sha256(
    authority_payload: Mapping[str, Any], compact: Sequence[ManifoldCandidateV4],
) -> str:
  value = authority_payload.get("candidate_domain_sha256")
  if value is None:
    return canonical_sha256([row.payload() for row in sorted(
        compact, key=lambda row: row.candidate_key,
    )])
  return _require_sha256(str(value), label="candidate domain")


@dataclass(frozen=True, slots=True)
class ExactChildReceiptV4:
  call_nonce: str
  terminal_status: str
  child_started: bool
  issuer: str
  result_items: tuple[tuple[str, Any], ...]
  receipt_payload_sha256: str
  operation: str = "candidate_exact_bundle"
  schema_version: str = CHILD_RECEIPT_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CHILD_RECEIPT_SCHEMA_VERSION:
      raise ValueError("V4 exact receipt schema differs")
    if not self.call_nonce or self.operation != "candidate_exact_bundle":
      raise ValueError("V4 exact receipt identity differs")
    if self.terminal_status not in {"ok", "timeout", "kernel_error"}:
      raise ValueError("V4 exact receipt status differs")
    if self.issuer not in {"child_process_echo", "parent_observer"}:
      raise ValueError("V4 exact receipt issuer differs")
    _require_sha256(self.receipt_payload_sha256, label="exact child receipt")
    if self.receipt_payload_sha256 != canonical_sha256(self.unsigned_payload()):
      raise ValueError("V4 exact receipt commitment differs")

  @property
  def result(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.result_items))

  def unsigned_payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version, "call_nonce": self.call_nonce,
        "operation": self.operation, "terminal_status": self.terminal_status,
        "child_started": self.child_started, "issuer": self.issuer,
        "result": dict(self.result_items),
    }

  def payload(self) -> dict[str, Any]:
    return {**self.unsigned_payload(), "receipt_payload_sha256": self.receipt_payload_sha256}


def issue_exact_child_receipt_v4(
    *, call_nonce: str, terminal_status: str, child_started: bool,
    result: Mapping[str, Any] | None = None, issuer: str = "child_process_echo",
) -> ExactChildReceiptV4:
  items = tuple(sorted((str(key), value) for key, value in (result or {}).items()))
  unsigned = {
      "schema_version": CHILD_RECEIPT_SCHEMA_VERSION, "call_nonce": call_nonce,
      "operation": "candidate_exact_bundle", "terminal_status": terminal_status,
      "child_started": child_started, "issuer": issuer, "result": dict(items),
  }
  return ExactChildReceiptV4(
      call_nonce=call_nonce, terminal_status=terminal_status,
      child_started=child_started, issuer=issuer, result_items=items,
      receipt_payload_sha256=canonical_sha256(unsigned),
  )


class ExactManifoldExecutorV4(Protocol):
  def run_child(self, candidate: ManifoldCandidateV4, *, call_nonce: str,
                absolute_deadline: float) -> ExactChildReceiptV4: ...


@dataclass(frozen=True, slots=True)
class ExactBatchCandidateReceiptV8:
  request_payload_sha256: str
  batch_nonce: str
  rank_zero_based: int
  candidate_key: str
  call_nonce: str
  terminal_status: str
  result_items: tuple[tuple[str, Any], ...]
  event_payload_sha256: str
  schema_version: str = BATCH_CANDIDATE_RECEIPT_SCHEMA_VERSION_V8

  def __post_init__(self) -> None:
    if self.schema_version != BATCH_CANDIDATE_RECEIPT_SCHEMA_VERSION_V8:
      raise ValueError("V8 batch candidate receipt schema differs")
    _require_sha256(self.request_payload_sha256, label="batch request")
    _require_sha256(self.candidate_key, label="batch candidate")
    _require_sha256(self.event_payload_sha256, label="batch candidate event")
    if not self.batch_nonce or not self.call_nonce or self.rank_zero_based < 0:
      raise ValueError("V8 batch candidate identity differs")
    if self.terminal_status not in {"ok", "kernel_error"}:
      raise ValueError("V8 batch candidate terminal status differs")
    if self.event_payload_sha256 != canonical_sha256(self.unsigned_payload()):
      raise ValueError("V8 batch candidate event commitment differs")

  @property
  def result(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.result_items))

  def unsigned_payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version,
        "request_payload_sha256": self.request_payload_sha256,
        "batch_nonce": self.batch_nonce, "rank_zero_based": self.rank_zero_based,
        "candidate_key": self.candidate_key, "call_nonce": self.call_nonce,
        "terminal_status": self.terminal_status, "result": dict(self.result_items),
    }

  def payload(self) -> dict[str, Any]:
    return {**self.unsigned_payload(), "event_payload_sha256": self.event_payload_sha256}


def issue_exact_batch_candidate_receipt_v8(
    *, request_payload_sha256: str, batch_nonce: str, rank_zero_based: int,
    candidate_key: str, call_nonce: str, terminal_status: str,
    result: Mapping[str, Any],
) -> ExactBatchCandidateReceiptV8:
  items = tuple(sorted((str(key), value) for key, value in result.items()))
  unsigned = {
      "schema_version": BATCH_CANDIDATE_RECEIPT_SCHEMA_VERSION_V8,
      "request_payload_sha256": request_payload_sha256,
      "batch_nonce": batch_nonce, "rank_zero_based": int(rank_zero_based),
      "candidate_key": candidate_key, "call_nonce": call_nonce,
      "terminal_status": terminal_status, "result": dict(items),
  }
  return ExactBatchCandidateReceiptV8(
      request_payload_sha256=request_payload_sha256, batch_nonce=batch_nonce,
      rank_zero_based=int(rank_zero_based), candidate_key=candidate_key,
      call_nonce=call_nonce, terminal_status=terminal_status, result_items=items,
      event_payload_sha256=canonical_sha256(unsigned),
  )


@dataclass(frozen=True, slots=True)
class ExactBatchChildReceiptV8:
  batch_nonce: str
  request_payload_sha256: str
  terminal_status: str
  child_started: bool
  issuer: str
  candidate_receipts: tuple[ExactBatchCandidateReceiptV8, ...]
  result_items: tuple[tuple[str, Any], ...]
  receipt_payload_sha256: str
  operation: str = "candidate_exact_batch"
  schema_version: str = BATCH_CHILD_RECEIPT_SCHEMA_VERSION_V8

  def __post_init__(self) -> None:
    if self.schema_version != BATCH_CHILD_RECEIPT_SCHEMA_VERSION_V8:
      raise ValueError("V8 exact batch receipt schema differs")
    if not self.batch_nonce or self.operation != "candidate_exact_batch":
      raise ValueError("V8 exact batch receipt identity differs")
    _require_sha256(self.request_payload_sha256, label="batch request")
    _require_sha256(self.receipt_payload_sha256, label="batch child receipt")
    if self.terminal_status not in {"ok", "timeout", "kernel_error"}:
      raise ValueError("V8 exact batch receipt status differs")
    if self.issuer not in {"child_process_echo", "parent_observer"}:
      raise ValueError("V8 exact batch receipt issuer differs")
    for sequence, receipt in enumerate(self.candidate_receipts):
      if (type(receipt) is not ExactBatchCandidateReceiptV8
          or receipt.batch_nonce != self.batch_nonce
          or receipt.request_payload_sha256 != self.request_payload_sha256
          or receipt.rank_zero_based != sequence):
        raise ValueError("V8 exact batch candidate order/binding differs")
    if self.receipt_payload_sha256 != canonical_sha256(self.unsigned_payload()):
      raise ValueError("V8 exact batch receipt commitment differs")

  @property
  def result(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.result_items))

  def unsigned_payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version, "batch_nonce": self.batch_nonce,
        "operation": self.operation,
        "request_payload_sha256": self.request_payload_sha256,
        "terminal_status": self.terminal_status,
        "child_started": self.child_started, "issuer": self.issuer,
        "candidate_receipts": [row.payload() for row in self.candidate_receipts],
        "result": dict(self.result_items),
    }

  def payload(self) -> dict[str, Any]:
    return {**self.unsigned_payload(), "receipt_payload_sha256": self.receipt_payload_sha256}


def issue_exact_batch_child_receipt_v8(
    *, batch_nonce: str, request_payload_sha256: str, terminal_status: str,
    child_started: bool, candidate_receipts: Sequence[ExactBatchCandidateReceiptV8],
    result: Mapping[str, Any], issuer: str = "child_process_echo",
) -> ExactBatchChildReceiptV8:
  candidates = tuple(candidate_receipts)
  items = tuple(sorted((str(key), value) for key, value in result.items()))
  unsigned = {
      "schema_version": BATCH_CHILD_RECEIPT_SCHEMA_VERSION_V8,
      "batch_nonce": batch_nonce, "operation": "candidate_exact_batch",
      "request_payload_sha256": request_payload_sha256,
      "terminal_status": terminal_status, "child_started": child_started,
      "issuer": issuer, "candidate_receipts": [row.payload() for row in candidates],
      "result": dict(items),
  }
  return ExactBatchChildReceiptV8(
      batch_nonce=batch_nonce, request_payload_sha256=request_payload_sha256,
      terminal_status=terminal_status, child_started=child_started, issuer=issuer,
      candidate_receipts=candidates, result_items=items,
      receipt_payload_sha256=canonical_sha256(unsigned),
  )


class ExactBatchManifoldExecutorV8(Protocol):
  def run_batch(
      self, candidates: Sequence[ManifoldCandidateV4], *,
      call_nonces: Sequence[str], batch_nonce: str, absolute_deadline: float,
  ) -> ExactBatchChildReceiptV8: ...


def _post_exact_reasons(request: ConstraintManifoldRequestV4,
                        observation: Mapping[str, Any], *,
                        _policy: FrozenConstraintManifoldPolicyV4 | None = None,
                        ) -> tuple[str, ...]:
  thresholds = (request.policy if _policy is None else _policy).thresholds
  reasons = []
  if float(observation.get("selected_face_distance_mm", math.inf)) > thresholds.selected_face_clearance_mm:
    reasons.append("selected_face_clearance")
  if float(observation.get("whole_solid_common_volume_mm3", math.inf)) > thresholds.whole_common_volume_mm3:
    reasons.append("whole_solid_interference")
  return tuple(reasons)


def _validate_ledger(payload: Mapping[str, Any]) -> None:
  receipts = payload.get("terminal_child_receipts", ())
  if not isinstance(receipts, (list, tuple)):
    raise ValueError("V4 ledger receipts differ")
  started = sum(bool(row.get("child_started")) for row in receipts)
  if int(payload.get("terminal_receipt_count", -1)) != len(receipts):
    raise ValueError("V4 ledger terminal receipt arithmetic differs")
  if int(payload.get("started_occ_children", -1)) != started:
    raise ValueError("V4 ledger started child arithmetic differs")
  if int(payload.get("unreceipted_started_children", -1)) != 0:
    raise ValueError("V4 ledger contains an unreceipted child")
  if int(payload.get("reserved_occ_calls", -1)) > 32:
    raise ValueError("V4 ledger exceeds OCC budget")
  for raw in receipts:
    if not isinstance(raw, Mapping):
      raise ValueError("V4 ledger receipt row differs")
    result = raw.get("result")
    if not isinstance(result, Mapping):
      raise ValueError("V4 ledger receipt result differs")
    if raw.get("schema_version") == BATCH_CHILD_RECEIPT_SCHEMA_VERSION_V8:
      candidate_receipts = []
      for candidate in raw.get("candidate_receipts", ()):
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("result"), Mapping):
          raise ValueError("V8 ledger batch candidate receipt differs")
        candidate_receipts.append(ExactBatchCandidateReceiptV8(
            request_payload_sha256=str(candidate.get("request_payload_sha256", "")),
            batch_nonce=str(candidate.get("batch_nonce", "")),
            rank_zero_based=int(candidate.get("rank_zero_based", -1)),
            candidate_key=str(candidate.get("candidate_key", "")),
            call_nonce=str(candidate.get("call_nonce", "")),
            terminal_status=str(candidate.get("terminal_status", "")),
            result_items=tuple(sorted(
                (str(key), value) for key, value in candidate["result"].items()
            )),
            event_payload_sha256=str(candidate.get("event_payload_sha256", "")),
            schema_version=str(candidate.get("schema_version", "")),
        ))
      ExactBatchChildReceiptV8(
          batch_nonce=str(raw.get("batch_nonce", "")),
          request_payload_sha256=str(raw.get("request_payload_sha256", "")),
          operation=str(raw.get("operation", "")),
          terminal_status=str(raw.get("terminal_status", "")),
          child_started=bool(raw.get("child_started")),
          issuer=str(raw.get("issuer", "")),
          candidate_receipts=tuple(candidate_receipts),
          result_items=tuple(sorted((str(key), value) for key, value in result.items())),
          receipt_payload_sha256=str(raw.get("receipt_payload_sha256", "")),
          schema_version=str(raw.get("schema_version", "")),
      )
    else:
      ExactChildReceiptV4(
          call_nonce=str(raw.get("call_nonce", "")),
          operation=str(raw.get("operation", "")),
          terminal_status=str(raw.get("terminal_status", "")),
          child_started=bool(raw.get("child_started")),
          issuer=str(raw.get("issuer", "")),
          result_items=tuple(sorted((str(key), value) for key, value in result.items())),
          receipt_payload_sha256=str(raw.get("receipt_payload_sha256", "")),
          schema_version=str(raw.get("schema_version", "")),
      )
  exact_evidence = payload.get("exact_candidate_receipts", ())
  if not isinstance(exact_evidence, (list, tuple)):
    raise ValueError("V8 ledger exact candidate evidence differs")
  unreceipted_exact = int(payload.get("unreceipted_exact_candidates", 0))
  unreceipted_occ = int(payload.get("unreceipted_occ_calls", 0))
  if (unreceipted_exact < 0 or unreceipted_occ != 2 * unreceipted_exact
      or int(payload.get("reserved_occ_calls", -1))
      != 2 * len(exact_evidence) + unreceipted_occ):
    if exact_evidence or unreceipted_exact or unreceipted_occ:
      raise ValueError("V8 ledger exact candidate OCC arithmetic differs")
  if exact_evidence:
    for sequence, evidence in enumerate(exact_evidence):
      if not isinstance(evidence, Mapping):
        raise ValueError("V8 ledger exact candidate evidence row differs")
      unsigned = dict(evidence)
      observed = unsigned.pop("candidate_evidence_sha256", None)
      if (unsigned.get("schema_version") != BATCH_CANDIDATE_EVIDENCE_SCHEMA_VERSION_V8
          or int(unsigned.get("batch_sequence_zero_based", -1)) != sequence
          or int(unsigned.get("occ_calls_reserved", -1)) != 2
          or observed != canonical_sha256(unsigned)):
        raise ValueError("V8 ledger exact candidate evidence commitment differs")
  batch_rows = [
      row for row in receipts
      if row.get("schema_version") == BATCH_CHILD_RECEIPT_SCHEMA_VERSION_V8
  ]
  if batch_rows:
    if len(batch_rows) != 1:
      raise ValueError("V8 ledger exact batch receipt cardinality differs")
    result = batch_rows[0]["result"]
    if (int(result.get("unreceipted_exact_candidates", -1)) != unreceipted_exact
        or int(result.get("unreceipted_occ_calls", -1)) != unreceipted_occ
        or int(result.get("occ_calls_reserved", -1))
        != int(payload.get("reserved_occ_calls", -1))):
      raise ValueError("V8 ledger exact batch accounting binding differs")


_CERTIFICATE_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


class ConstraintManifoldCertificateV4:
  __slots__ = ("__weakref__",)

  def __init__(self, payload: Mapping[str, Any], *, authority: SanitizedCandidateAuthorityV4,
               replay: Callable[[], Mapping[str, Any]], _factory_token: object) -> None:
    if _factory_token is not _CERTIFICATE_TOKEN or not callable(replay):
      raise TypeError("V4 certificate is loader/solver factory-only")
    frozen = json.loads(json.dumps(payload))
    observed = frozen.pop("certificate_payload_sha256", None)
    if frozen.get("schema_version") != CERTIFICATE_SCHEMA_VERSION or observed != canonical_sha256(frozen):
      raise ValueError("V4 certificate commitment differs")
    frozen["certificate_payload_sha256"] = observed
    _CERTIFICATE_STATES[self] = MappingProxyType({
        "payload": _deep_freeze(frozen), "authority": authority, "replay": replay,
    })
    self.revalidate(full_replay=True)

  def revalidate(self, *, full_replay: bool = False) -> None:
    state = _CERTIFICATE_STATES[self]
    authority_sha256, authority_payload = (
        _authenticated_candidate_authority_snapshot_v1(state["authority"])
    )
    payload = _deep_thaw(state["payload"])
    observed = payload.pop("certificate_payload_sha256")
    if payload["candidate_authority_sha256"] != authority_sha256:
      raise ValueError("V4 certificate candidate authority differs")
    expected_policy = _official_policy_for_authenticated_candidate_payload_v1(
        payload["program_index"], authority_payload,
    )
    if payload["frozen_policy"] != expected_policy.payload():
      raise ValueError("V4 certificate policy differs")
    _validate_ledger(payload["ledger"])
    if canonical_sha256(payload) != observed:
      raise ValueError("V4 certificate bytes changed")
    if full_replay:
      replayed = json.loads(json.dumps(state["replay"]()))
      if replayed != payload["replay_core"]:
        raise ValueError("V4 certificate full geometry replay differs")

  @property
  def accepted(self) -> bool:
    self.revalidate()
    return bool(_CERTIFICATE_STATES[self]["payload"]["accepted"])

  def payload(self) -> dict[str, Any]:
    self.revalidate()
    return _deep_thaw(_CERTIFICATE_STATES[self]["payload"])

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("V4 certificate is immutable")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("V4 certificate is not serializable")


_COUNTEREVIDENCE_STATES: WeakKeyDictionary[Any, Mapping[str, Any]] = WeakKeyDictionary()


class ConstraintManifoldCounterevidenceV4:
  __slots__ = ("__weakref__",)

  def __init__(self, payload: Mapping[str, Any], *, _factory_token: object) -> None:
    if _factory_token is not _COUNTEREVIDENCE_TOKEN:
      raise TypeError("V4 counterevidence is solver-factory-only")
    row = json.loads(json.dumps(payload))
    observed = row.pop("counterevidence_payload_sha256", None)
    if row.get("schema_version") != COUNTEREVIDENCE_SCHEMA_VERSION or observed != canonical_sha256(row):
      raise ValueError("V4 counterevidence commitment differs")
    row["counterevidence_payload_sha256"] = observed
    _COUNTEREVIDENCE_STATES[self] = _deep_freeze(row)

  def payload(self) -> dict[str, Any]:
    return _deep_thaw(_COUNTEREVIDENCE_STATES[self])

  def __setattr__(self, _name: str, _value: Any) -> None:
    raise TypeError("V4 counterevidence is immutable")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("V4 counterevidence is not serializable")


@dataclass(frozen=True, slots=True)
class ConstraintManifoldSolveResultV4:
  selected: ManifoldCandidateV4 | None
  certificate: ConstraintManifoldCertificateV4 | None
  counterevidence: ConstraintManifoldCounterevidenceV4 | None
  terminal_status: str
  full_candidate_count: int
  exact_domain_count: int
  ledger_items: tuple[tuple[str, Any], ...]

  @property
  def ledger(self) -> Mapping[str, Any]:
    return MappingProxyType(dict(self.ledger_items))


def _finite_positive_radius_or_none_v4(value: Any) -> float | None:
  if type(value) in {int, float}:
    radius = float(value)
    if math.isfinite(radius) and radius > 0.0:
      return radius
  return None


def _radius_counterevidence(
    request: ConstraintManifoldRequestV4, *,
    _authenticated_snapshot: tuple[str, Mapping[str, Any]],
    _policy: FrozenConstraintManifoldPolicyV4,
) -> ConstraintManifoldCounterevidenceV4 | object | None:
  if request.program_index != 11:
    return None
  authority_sha256, authority_payload = _authenticated_snapshot
  endpoints = authority_payload.get("endpoints", ())
  if len(endpoints) != 2:
    raise ValueError("V4 cylinder endpoints differ")
  trims = [row.get("trim", {}) for row in endpoints]
  sides = [row.get("surface_side") for row in trims]
  if any(
      type(side) is not str or side not in {"interior", "exterior"}
      for side in sides
  ):
    return _RADIUS_GEOMETRIC_EVIDENCE_INVALID
  radii = [
      _finite_positive_radius_or_none_v4(row.get("radius_mm"))
      for row in trims
  ]
  reason = None
  clearance = None
  if sides[0] == sides[1]:
    reason = "shaft_bore_side_relation_all_representatives"
  elif any(row is None for row in radii):
    return _RADIUS_GEOMETRIC_EVIDENCE_INVALID
  else:
    interior = radii[next(
        index for index, row in enumerate(trims)
        if row["surface_side"] == "interior"
    )]
    exterior = radii[next(
        index for index, row in enumerate(trims)
        if row["surface_side"] == "exterior"
    )]
    if interior is None or exterior is None:
      return _RADIUS_GEOMETRIC_EVIDENCE_INVALID
    clearance = interior - exterior
    if clearance < -1e-9 or clearance > _policy.thresholds.maximum_radial_clearance_mm:
      reason = "shaft_bore_radius_relation_all_representatives"
  if reason is None:
    return None
  unsigned = {
      "schema_version": COUNTEREVIDENCE_SCHEMA_VERSION,
      "query_id": request.query_id, "program_index": request.program_index,
      "candidate_authority_sha256": authority_sha256,
      "reason_code": reason, "scope": "all_pose_representatives",
      "radius_values_mm": radii, "radial_clearance_mm": clearance,
      "uses_gold_or_source_absence": False, "occ_calls_reserved": 0,
      "claim": "conditional_physical_infeasibility",
  }
  return ConstraintManifoldCounterevidenceV4(
      {**unsigned, "counterevidence_payload_sha256": canonical_sha256(unsigned)},
      _factory_token=_COUNTEREVIDENCE_TOKEN,
  )


def solve_constraint_manifold_v4(
    request: ConstraintManifoldRequestV4, *, exact_executor: ExactManifoldExecutorV4,
    query_started_at: float, clock: Callable[[], float] = time.monotonic,
) -> ConstraintManifoldSolveResultV4:
  """Solve one selected hypothesis under the frozen two-stage contract."""

  if type(request) is not ConstraintManifoldRequestV4:
    raise TypeError("V4 solver requires a typed V4 request")
  authority_snapshot = _authenticated_candidate_authority_snapshot_v1(
      request.candidate_authority,
  )
  authority_sha256, authority_payload = authority_snapshot
  policy = _official_policy_for_authenticated_candidate_payload_v1(
      request.program_index, authority_payload,
  )
  start = float(query_started_at)
  now = clock()
  if not math.isfinite(start) or start > now:
    raise ValueError("V4 total-wall start differs")
  deadline = start + policy.budget.total_wall_seconds
  if now >= deadline:
    ledger = {
        "full_candidate_count": 0, "exact_domain_count": 0,
        "reserved_occ_calls": 0, "started_occ_children": 0,
        "terminal_receipt_count": 0, "unreceipted_started_children": 0,
        "terminal_child_receipts": [], "elapsed_seconds": now - start,
    }
    return ConstraintManifoldSolveResultV4(
        None, None, None, "unknown_preprocessing_timeout", 0, 0,
        tuple(ledger.items()),
    )
  domain_count = _committed_domain_count(authority_payload)
  counter = _radius_counterevidence(
      request, _authenticated_snapshot=authority_snapshot, _policy=policy,
  )
  if counter is not None:
    ledger = {
        "full_candidate_count": domain_count,
        "exact_domain_count": 0, "reserved_occ_calls": 0,
        "started_occ_children": 0, "terminal_receipt_count": 0,
        "unreceipted_started_children": 0, "terminal_child_receipts": [],
        "elapsed_seconds": clock() - start,
    }
    if counter is _RADIUS_GEOMETRIC_EVIDENCE_INVALID:
      return ConstraintManifoldSolveResultV4(
          None, None, None,
          "unknown_preprocessing_geometric_evidence_invalid",
          int(ledger["full_candidate_count"]), 0, tuple(ledger.items()),
      )
    return ConstraintManifoldSolveResultV4(
        None, None, counter, "physically_infeasible_all_representatives", int(ledger["full_candidate_count"]),
        0, tuple(ledger.items()),
    )
  compact, exact_domain = generate_and_rank_candidates_v4(
      request, _authenticated_snapshot=authority_snapshot,
  )
  if clock() >= deadline:
    ledger = {
        "full_candidate_count": domain_count, "exact_domain_count": len(exact_domain),
        "reserved_occ_calls": 0, "started_occ_children": 0,
        "terminal_receipt_count": 0, "unreceipted_started_children": 0,
        "terminal_child_receipts": [], "elapsed_seconds": clock() - start,
    }
    return ConstraintManifoldSolveResultV4(
        None, None, None, "unknown_preprocessing_timeout", domain_count, len(exact_domain),
        tuple(ledger.items()),
    )

  terminal_receipts: list[ExactChildReceiptV4 | ExactBatchChildReceiptV8] = []
  exact_candidate_evidence: list[dict[str, Any]] = []
  unreceipted_exact_candidates = 0
  unreceipted_occ_calls = 0
  evaluated: list[tuple[ManifoldCandidateV4, Mapping[str, Any], tuple[str, ...]]] = []
  analytic_rejected = 0
  failure: str | None = None
  exact_candidates: list[ManifoldCandidateV4] = []
  for candidate in exact_domain:
    analytic = _analytic_reasons(request, candidate)
    if analytic:
      analytic_rejected += 1
      evaluated.append((candidate, {}, analytic))
      continue
    exact_candidates.append(candidate)

  batch_method = getattr(exact_executor, "run_batch", None)
  used_batch_executor = bool(exact_candidates and callable(batch_method))
  if used_batch_executor:
    batch_nonce = secrets.token_hex(16)
    call_nonces = tuple(secrets.token_hex(16) for _ in exact_candidates)
    if clock() >= deadline:
      failure = "unknown_total_wall_timeout"
    else:
      child_deadline = min(deadline, clock() + policy.budget.exact_child_seconds)
      try:
        batch_receipt = batch_method(
            tuple(exact_candidates), call_nonces=call_nonces,
            batch_nonce=batch_nonce, absolute_deadline=child_deadline,
        )
      except Exception as error:
        batch_receipt = issue_exact_batch_child_receipt_v8(
            batch_nonce=batch_nonce,
            request_payload_sha256=canonical_sha256({
                "batch_nonce": batch_nonce,
                "candidate_keys": [row.candidate_key for row in exact_candidates],
            }),
            terminal_status="kernel_error", child_started=False,
            candidate_receipts=(), issuer="parent_observer",
            result={"error": f"executor_escape:{type(error).__name__}:{error}"},
        )
      if type(batch_receipt) is not ExactBatchChildReceiptV8:
        batch_receipt = issue_exact_batch_child_receipt_v8(
            batch_nonce=batch_nonce,
            request_payload_sha256=canonical_sha256({
                "batch_nonce": batch_nonce,
                "candidate_keys": [row.candidate_key for row in exact_candidates],
            }),
            terminal_status="kernel_error", child_started=False,
            candidate_receipts=(), issuer="parent_observer",
            result={"error": "executor_returned_non_v8_batch_receipt"},
        )
      terminal_receipts.append(batch_receipt)
      if batch_receipt.batch_nonce != batch_nonce:
        failure = "unknown_exact_receipt_binding"
      else:
        try:
          unreceipted_exact_candidates = int(
              batch_receipt.result.get("unreceipted_exact_candidates", -1)
          )
          unreceipted_occ_calls = int(
              batch_receipt.result.get("unreceipted_occ_calls", -1)
          )
          reported_reserved_occ_calls = int(
              batch_receipt.result.get("occ_calls_reserved", -1)
          )
          if (unreceipted_exact_candidates < 0
              or unreceipted_occ_calls != 2 * unreceipted_exact_candidates
              or reported_reserved_occ_calls != (
                  2 * len(batch_receipt.candidate_receipts) + unreceipted_occ_calls
              )
              or len(batch_receipt.candidate_receipts) + unreceipted_exact_candidates
              > len(exact_candidates)):
            raise ValueError("batch OCC accounting")
        except Exception:
          failure = "unknown_exact_receipt_binding"
        accepted_in_batch = False
        for sequence, candidate_receipt in enumerate(batch_receipt.candidate_receipts):
          if sequence >= len(exact_candidates):
            failure = "unknown_exact_receipt_binding"
            break
          candidate = exact_candidates[sequence]
          if (candidate_receipt.candidate_key != candidate.candidate_key
              or candidate_receipt.call_nonce != call_nonces[sequence]):
            failure = "unknown_exact_receipt_binding"
            break
          if candidate_receipt.terminal_status != "ok":
            failure = f"unknown_exact_{candidate_receipt.terminal_status}"
            reasons: tuple[str, ...] = ()
            observation: Mapping[str, Any] = {}
          else:
            observation = {
                "selected_face_distance_mm": float(
                    candidate_receipt.result.get("distance_mm", math.inf)
                ),
                "whole_solid_common_volume_mm3": float(
                    candidate_receipt.result.get("common_volume_mm3", math.inf)
                ),
                "aabb_intersection_upper_mm3": float(
                    candidate_receipt.result.get("aabb_intersection_upper_mm3", math.inf)
                ),
            }
            reasons = _post_exact_reasons(request, observation, _policy=policy)
            evaluated.append((candidate, observation, reasons))
          evidence_unsigned = {
              "schema_version": BATCH_CANDIDATE_EVIDENCE_SCHEMA_VERSION_V8,
              "batch_nonce": batch_nonce,
              "batch_sequence_zero_based": sequence,
              "candidate_key": candidate.candidate_key,
              "call_nonce": call_nonces[sequence],
              "terminal_status": candidate_receipt.terminal_status,
              "worker_event_payload_sha256": candidate_receipt.event_payload_sha256,
              "result": dict(candidate_receipt.result),
              "post_exact_reason_codes": list(reasons),
              "occ_calls_reserved": 2,
          }
          exact_candidate_evidence.append({
              **evidence_unsigned,
              "candidate_evidence_sha256": canonical_sha256(evidence_unsigned),
          })
          if failure is not None:
            break
          if not reasons:
            accepted_in_batch = True
            if sequence + 1 != len(batch_receipt.candidate_receipts):
              failure = "unknown_exact_receipt_binding"
            break
        if failure is None and batch_receipt.terminal_status != "ok":
          failure = f"unknown_exact_{batch_receipt.terminal_status}"
        if (failure is None and not accepted_in_batch
            and len(batch_receipt.candidate_receipts) != len(exact_candidates)):
          failure = "unknown_exact_receipt_binding"
  elif exact_candidates:
    for candidate in exact_candidates:
      if clock() >= deadline:
        failure = "unknown_total_wall_timeout"
        break
      nonce = secrets.token_hex(16)
      child_deadline = min(deadline, clock() + policy.budget.exact_child_seconds)
      try:
        receipt = exact_executor.run_child(
            candidate, call_nonce=nonce, absolute_deadline=child_deadline,
        )
      except Exception as error:
        receipt = issue_exact_child_receipt_v4(
            call_nonce=nonce, terminal_status="kernel_error", child_started=False,
            issuer="parent_observer",
            result={"error": f"executor_escape:{type(error).__name__}:{error}"},
        )
      if type(receipt) is not ExactChildReceiptV4:
        receipt = issue_exact_child_receipt_v4(
            call_nonce=nonce, terminal_status="kernel_error", child_started=False,
            issuer="parent_observer", result={"error": "executor_returned_non_v4_receipt"},
        )
      terminal_receipts.append(receipt)
      if receipt.call_nonce != nonce:
        failure = "unknown_exact_receipt_binding"
        break
      if receipt.terminal_status != "ok":
        failure = f"unknown_exact_{receipt.terminal_status}"
        break
      observation = {
          "selected_face_distance_mm": float(receipt.result.get("distance_mm", math.inf)),
          "whole_solid_common_volume_mm3": float(receipt.result.get("common_volume_mm3", math.inf)),
          "aabb_intersection_upper_mm3": float(receipt.result.get("aabb_intersection_upper_mm3", math.inf)),
      }
      reasons = _post_exact_reasons(request, observation, _policy=policy)
      evaluated.append((candidate, observation, reasons))
      # The exact domain is already in the frozen total order.  Therefore the
      # first feasible row is exactly the selected row; later OCC calls cannot
      # change the decision and would only inflate wall time.
      if not reasons:
        break

  exact_evaluated_count = (
      len(exact_candidate_evidence) if used_batch_executor else len(terminal_receipts)
  )
  ledger = {
      "full_candidate_count": domain_count, "exact_domain_count": len(exact_domain),
      "analytic_rejected_candidates": analytic_rejected,
      "exact_evaluated_candidates": exact_evaluated_count,
      "unreached_exact_candidates": (
          len(exact_domain) - exact_evaluated_count - analytic_rejected
          - unreceipted_exact_candidates
      ),
      "reserved_occ_calls": 2 * exact_evaluated_count + unreceipted_occ_calls,
      "unreceipted_exact_candidates": unreceipted_exact_candidates,
      "unreceipted_occ_calls": unreceipted_occ_calls,
      "started_occ_children": sum(row.child_started for row in terminal_receipts),
      "terminal_receipt_count": len(terminal_receipts),
      "unreceipted_started_children": 0,
      "terminal_child_receipts": [row.payload() for row in terminal_receipts],
      "exact_candidate_receipts": exact_candidate_evidence,
      "elapsed_seconds": clock() - start,
  }
  if ledger["reserved_occ_calls"] > policy.budget.maximum_occ_calls:
    raise RuntimeError("internal V4 OCC reservation bug")
  if failure is not None:
    return ConstraintManifoldSolveResultV4(
        None, None, None, failure, domain_count, len(exact_domain), tuple(ledger.items()),
    )
  accepted = [row for row in evaluated if row[1] and not row[2]]
  if not accepted:
    # Bounded-search failure is deliberately unknown, never a physical negative.
    return ConstraintManifoldSolveResultV4(
        None, None, None, "unknown_bounded_no_feasible_representative",
        domain_count, len(exact_domain), tuple(ledger.items()),
    )
  selected, observation, _ = min(
      accepted, key=lambda row: _candidate_rank_for_policy_v1(
          request.program_index, policy, row[0],
      ),
  )
  rank = next(index for index, row in enumerate(exact_domain) if row.candidate_key == selected.candidate_key)
  replay_core = {
      "candidate_key": selected.candidate_key,
      "candidate_domain_sha256": _committed_domain_sha256(authority_payload, compact),
      "exact_top16_sha256": canonical_sha256([row.payload() for row in exact_domain]),
      "selected_rank_zero_based": rank,
      "cheap_metrics": dict(selected.cheap_metric_items),
      "observation": dict(observation),
      "analytic_reason_codes": list(_analytic_reasons(request, selected)),
      "post_exact_reason_codes": list(_post_exact_reasons(
          request, observation, _policy=policy,
      )),
      "accepted": True,
  }

  def replay() -> Mapping[str, Any]:
    replay_snapshot = _authenticated_candidate_authority_snapshot_v1(
        request.candidate_authority,
    )
    _replay_authority_sha256, replay_authority_payload = replay_snapshot
    if _replay_authority_sha256 != authority_sha256:
      raise ValueError("V4 certificate replay authority differs")
    replay_policy = _official_policy_for_authenticated_candidate_payload_v1(
        request.program_index, replay_authority_payload,
    )
    replay_compact, replay_top = generate_and_rank_candidates_v4(
        request, _authenticated_snapshot=replay_snapshot,
    )
    matches = [row for row in replay_top if row.candidate_key == selected.candidate_key]
    if len(matches) != 1:
      raise ValueError("selected V4 candidate is absent from replayed top16")
    # Native values are independently rerun by certificate loaders through a
    # replay executor.  During solve construction they are bound to the child
    # receipt and checked here deterministically.
    return {
        "candidate_key": selected.candidate_key,
        "candidate_domain_sha256": _committed_domain_sha256(
            replay_authority_payload, replay_compact,
        ),
        "exact_top16_sha256": canonical_sha256([row.payload() for row in replay_top]),
        "selected_rank_zero_based": next(i for i, row in enumerate(replay_top)
                                         if row.candidate_key == selected.candidate_key),
        "cheap_metrics": dict(matches[0].cheap_metric_items),
        "observation": dict(observation),
        "analytic_reason_codes": list(_analytic_reasons(request, matches[0])),
        "post_exact_reason_codes": list(_post_exact_reasons(
            request, observation, _policy=replay_policy,
        )),
        "accepted": True,
    }

  unsigned = {
      "schema_version": CERTIFICATE_SCHEMA_VERSION,
      "query_id": request.query_id, "program_index": request.program_index,
      "request_sha256": canonical_sha256(
          request._payload_from_authenticated_snapshot(authority_snapshot)
      ),
      "identity_authority_sha256": request.identity_authority_sha256,
      "selected_face_authority_sha256": request.selected_face_authority_sha256,
      "candidate_authority_sha256": authority_sha256,
      "frozen_policy": policy.payload(),
      "candidate": selected.payload(), "replay_core": replay_core,
      "ledger": ledger, "accepted": True, "reason_codes": [],
      "claim_scope": "selected_face_program_conditional_physical_feasibility_only",
      "source_pose_reproduction_claimed": False,
      "intended_interface_claimed": False,
      "alternate_feasible_is_negative": False,
      "oracle_inputs_absent": True,
  }
  certificate = ConstraintManifoldCertificateV4(
      {**unsigned, "certificate_payload_sha256": canonical_sha256(unsigned)},
      authority=request.candidate_authority, replay=replay,
      _factory_token=_CERTIFICATE_TOKEN,
  )
  return ConstraintManifoldSolveResultV4(
      selected, certificate, None, "accepted", domain_count, len(exact_domain),
      tuple(ledger.items()),
  )


def load_constraint_manifold_certificate_v4(
    certificate_path: str | Path, *, request: ConstraintManifoldRequestV4,
    replay_exact_executor: ExactManifoldExecutorV4,
    preprocess_replay: Callable[[], tuple[SanitizedCandidateAuthorityV4, Mapping[str, Any]]],
) -> ConstraintManifoldCertificateV4:
  """Load a V4 certificate only after a complete independent replay.

  The loader replays preprocessing through ``candidate_authority.revalidate``,
  reconstructs the complete cheap domain and frozen top-16 order, proves the
  selected candidate's membership/rank, validates every historical child
  receipt and ledger equation, and launches a fresh exact child for the
  selected rigid placement.  Source placement and gold identity are not inputs.
  """

  path = Path(certificate_path).resolve(strict=True)
  original_bytes = path.read_bytes()
  payload = json.loads(original_bytes)
  if not isinstance(payload, Mapping):
    raise ValueError("V4 certificate file is not a mapping")
  observed = dict(payload)
  commitment = observed.pop("certificate_payload_sha256", None)
  if commitment != canonical_sha256(observed):
    raise ValueError("V4 certificate file commitment differs")
  if observed.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
    raise ValueError("V4 certificate file schema differs")
  if observed.get("request_sha256") != canonical_sha256(request.payload()):
    raise ValueError("V4 certificate request binding differs")
  if observed.get("identity_authority_sha256") != request.identity_authority_sha256 or (
      observed.get("selected_face_authority_sha256") != request.selected_face_authority_sha256
  ):
    raise ValueError("V4 certificate identity/selected-face binding differs")
  if observed.get("candidate_authority_sha256") != request.candidate_authority.sha256:
    raise ValueError("V4 certificate preprocessing binding differs")
  _validate_ledger(observed["ledger"])
  stored_candidate = _candidate_from_payload(observed["candidate"], request.program_index)

  def replay() -> Mapping[str, Any]:
    if path.read_bytes() != original_bytes:
      raise ValueError("V4 certificate file changed after capture")
    replay_authority, preprocess_receipt = preprocess_replay()
    replay_authority.revalidate()
    if replay_authority.payload() != request.candidate_authority.payload():
      raise ValueError("V4 certificate preprocessing geometry replay differs")
    if (
        preprocess_receipt.get("terminal_status") != "ok"
        or preprocess_receipt.get("child_started") is not True
        or preprocess_receipt.get("issuer") != "child_process_echo"
    ):
      raise ValueError("V4 certificate preprocessing child receipt differs")
    receipt_unsigned = dict(preprocess_receipt)
    receipt_hash = receipt_unsigned.pop("receipt_payload_sha256", None)
    if receipt_hash != canonical_sha256(receipt_unsigned):
      raise ValueError("V4 certificate preprocessing receipt commitment differs")
    replay_request = ConstraintManifoldRequestV4(
        query_id=request.query_id, program_index=request.program_index,
        child_world_row_major=request.child_world_row_major,
        identity_authority_sha256=request.identity_authority_sha256,
        selected_face_authority_sha256=request.selected_face_authority_sha256,
        candidate_authority=replay_authority,
    )
    compact, top = generate_and_rank_candidates_v4(replay_request)
    matches = [row for row in top if row.candidate_key == stored_candidate.candidate_key]
    if len(matches) != 1 or matches[0].payload() != stored_candidate.payload():
      raise ValueError("V4 selected candidate domain/rank replay differs")
    rank = next(index for index, row in enumerate(top)
                if row.candidate_key == stored_candidate.candidate_key)
    nonce = secrets.token_hex(16)
    receipt = replay_exact_executor.run_child(
        stored_candidate, call_nonce=nonce,
        absolute_deadline=time.monotonic() + request.policy.budget.exact_child_seconds,
    )
    if (
        type(receipt) is not ExactChildReceiptV4
        or receipt.call_nonce != nonce or receipt.terminal_status != "ok"
    ):
      raise ValueError("V4 selected exact observation replay failed")
    observation = {
        "selected_face_distance_mm": float(receipt.result.get("distance_mm", math.inf)),
        "whole_solid_common_volume_mm3": float(receipt.result.get("common_volume_mm3", math.inf)),
        "aabb_intersection_upper_mm3": float(receipt.result.get("aabb_intersection_upper_mm3", math.inf)),
    }
    stored_core = observed["replay_core"]
    stored_observation = stored_core["observation"]
    for key, value in observation.items():
      if not math.isclose(float(stored_observation[key]), value, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("V4 selected native exact observation replay differs")
    replayed = {
        "candidate_key": stored_candidate.candidate_key,
        "candidate_domain_sha256": _committed_domain_sha256(
            replay_authority.payload(), compact,
        ),
        "exact_top16_sha256": canonical_sha256([row.payload() for row in top]),
        "selected_rank_zero_based": rank,
        "cheap_metrics": dict(matches[0].cheap_metric_items),
        "observation": dict(stored_observation),
        "analytic_reason_codes": list(_analytic_reasons(request, matches[0])),
        "post_exact_reason_codes": list(_post_exact_reasons(request, observation)),
        "accepted": not _analytic_reasons(request, matches[0]) and not _post_exact_reasons(
            request, observation,
        ),
    }
    if replayed["accepted"] is not True:
      raise ValueError("V4 replayed acceptance predicate rejected selected candidate")
    return replayed

  return ConstraintManifoldCertificateV4(
      payload, authority=request.candidate_authority, replay=replay,
      _factory_token=_CERTIFICATE_TOKEN,
  )


def exact_task_success_v4(*, intended_face_match: bool, intended_program_match: bool,
                          certificate_accepted: bool) -> bool:
  return bool(intended_face_match and intended_program_match and certificate_accepted)


def classify_hypothesis_v4(*, intended_face_match: bool, intended_program_match: bool,
                           terminal_status: str) -> str:
  if terminal_status == "accepted":
    return "intended_feasible_success" if (
        intended_face_match and intended_program_match
    ) else "alternate_feasible_interface"
  if terminal_status == "physically_infeasible_all_representatives":
    return "physically_infeasible"
  return "unknown"


__all__ = [
    "BATCH_CANDIDATE_EVIDENCE_SCHEMA_VERSION_V8",
    "BATCH_CANDIDATE_RECEIPT_SCHEMA_VERSION_V8",
    "BATCH_CHILD_RECEIPT_SCHEMA_VERSION_V8",
    "CERTIFICATE_SCHEMA_VERSION", "CHILD_RECEIPT_SCHEMA_VERSION",
    "COUNTEREVIDENCE_SCHEMA_VERSION", "OFFICIAL_CONSTRAINT_MANIFOLD_POLICY_V4",
    "OFFICIAL_P11_FACTORIZED_STRUCTURAL_TOPK_POLICY_V2",
    "OFFICIAL_P9_COLLISION_STRUCTURAL_TOPK_POLICY_V1",
    "P9_COLLISION_AWARE_STRUCTURAL_TOPK_RANK_SCHEMA_V1",
    "P11_FACTORIZED_STRUCTURAL_TOPK_RANK_SCHEMA_V2",
    "OFFICIAL_P11_BODY_AABB_POLICY_V1", "P11_BODY_AABB_RANK_SCHEMA_V1",
    "OFFICIAL_P11_STRATIFIED_DIVERSITY_POLICY_V1",
    "P11_STRATIFIED_DIVERSITY_RANK_SCHEMA_V1",
    "POLICY_SCHEMA_VERSION", "SCHEMA_VERSION", "ConstraintManifoldBudgetV4",
    "ConstraintManifoldCertificateV4", "ConstraintManifoldCounterevidenceV4",
    "ConstraintManifoldRequestV4", "ConstraintManifoldSolveResultV4",
    "ConstraintManifoldThresholdsV4", "ExactBatchCandidateReceiptV8",
    "ExactBatchChildReceiptV8", "ExactBatchManifoldExecutorV8",
    "ExactChildReceiptV4", "ExactManifoldExecutorV4", "FrozenConstraintManifoldPolicyV4",
    "ManifoldCandidateV4", "SanitizedCandidateAuthorityV4", "canonical_sha256",
    "candidate_rank_payload_p11_body_aabb_v1", "candidate_rank_payload_v4",
    "canonical_candidate_payload_v4",
    "classify_hypothesis_v4", "exact_task_success_v4",
    "generate_and_rank_candidates_v4", "issue_exact_batch_candidate_receipt_v8",
    "issue_exact_batch_child_receipt_v8", "issue_exact_child_receipt_v4",
    "load_constraint_manifold_certificate_v4", "solve_constraint_manifold_v4",
]
