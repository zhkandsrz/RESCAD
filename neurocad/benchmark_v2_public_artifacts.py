"""Fail-closed public serializers for benchmark-v2 result artifacts.

Private gauges, source geometry identities, paths, gold labels, and free-form
kernel errors belong only in custodian-side audit state.  Public paper rows are
constructed from explicit allowlists here instead of recursively redacting an
already serialized private object.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_OPAQUE_PART_ID = re.compile(r"^opaque_part_[0-9]{3,}$")
_FORBIDDEN_KEY_FRAGMENTS = (
    "private_",
    "raw_face",
    "fusion_face",
    "step_path",
    "selected_step_paths",
    "evaluation_gold",
    "raw_to_randomized",
    "gauge",
    "protocol_audit",
    "point_on_shape",
    "vector_a_to_b",
)

_PUBLIC_POSE_ROW_KEYS = frozenset(
    {
        "case_index",
        "case_id",
        "status",
        "success",
        "aabb_clean_success",
        "strict_aabb_success",
        "pose_search_completed",
        "placement_complete",
        "candidate_no_unexpected_aabb",
        "proxy_certified",
        "proxy_certificate_category",
        "proxy_certificate_reasons",
        "proxy_certificate_metrics",
        "failure_category",
        "part_count",
        "placed_part_count",
        "topology_edge_count",
        "constraint_count",
        "realized_constraint_edge_count",
        "realized_expected_edge_count",
        "missing_expected_edge_count",
        "extra_constraint_edge_count",
        "realized_edge_coverage",
        "realized_constraint_edges",
        "missing_expected_edges",
        "extra_constraint_edges",
        "anchor",
        "selected_part_names",
        "final_verifier",
        "pair_rank_mode",
        "learned_pair_scorer",
        "final_collision_count",
        "final_expected_collision_count",
        "final_unexpected_collision_count",
        "final_collision_volume",
        "final_expected_collision_volume",
        "final_unexpected_collision_volume",
        "runtime_seconds",
        "input_protocol",
        "protocol_receipt",
        "error_type",
        "exact_certificate_enabled",
        "exact_certificate_ran",
        "exact_certificate_status",
        "exact_certificate_semantics_version",
        "exact_certificate_scope",
        "exact_collision_count",
        "exact_expected_collision_count",
        "exact_unexpected_collision_count",
        "exact_collision_volume",
        "exact_expected_collision_volume",
        "exact_unexpected_collision_volume",
        "exact_clean",
        "exact_no_unexpected",
        "exact_safety_certified",
        "exact_strict_certified",
        "exact_contact_distance_enabled",
        "exact_contact_distance_status",
        "exact_contact_distance_error_count",
        "exact_expected_contact_distances",
        "exact_expected_contact_distance_count",
        "exact_expected_contact_distance_max",
        "exact_expected_contact_distance_mean",
        "exact_contact_distance_tolerance",
        "exact_contact_distance_certified",
        "exact_contact_scope",
        "exact_physical_certified",
        "predicted_interface_physical_consistency_certified",
        "gold_target_interface_certification_status",
        "gold_target_interface_certified",
        "gold_target_interface_reason_code",
        "final_formal_acceptance_evidence",
        "final_formal_acceptance_reason_code",
        "exact_failure_category",
        "exact_failure_taxonomy",
    }
)

_PUBLIC_CONTACT_DISTANCE_KEYS = frozenset(
    {
        "part_a",
        "part_b",
        "distance",
        "contact_scope",
        "socket_a",
        "socket_b",
        "interface_id_a",
        "interface_id_b",
    }
)


def _canonical_sha256(value: Any) -> str:
  encoded = json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def public_protocol_receipt(audit: Mapping[str, Any]) -> dict[str, Any]:
  """Commit to a private audit while exposing only non-geometric evidence."""

  receipt = {
      "schema_version": "benchmark_v2_public_protocol_receipt.v1",
      "protocol_version": str(audit.get("protocol_version") or ""),
      "custodian_audit_sha256": _canonical_sha256(dict(audit)),
      "model_view_sha256": copy.deepcopy(audit.get("model_view_sha256") or {}),
      "face_identity_proof": copy.deepcopy(
          audit.get("face_identity_proof") or {}
      ),
  }
  for key in ("model_view_sha256", "face_identity_proof"):
    mapping = receipt[key]
    if not isinstance(mapping, Mapping) or any(
        not _OPAQUE_PART_ID.fullmatch(str(part_id)) for part_id in mapping
    ):
      raise ValueError("public protocol receipt contains a non-opaque part identity")
  validate_public_artifact(receipt)
  return receipt


def public_pose_evaluation_row(
    row: Mapping[str, Any],
    *,
    private_protocol_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Construct one publication-safe pose evaluation row from a private row."""

  result = {
      key: copy.deepcopy(row[key])
      for key in _PUBLIC_POSE_ROW_KEYS
      if key in row
  }
  distances = result.get("exact_expected_contact_distances")
  if isinstance(distances, list):
    result["exact_expected_contact_distances"] = [
        {
            key: copy.deepcopy(item[key])
            for key in _PUBLIC_CONTACT_DISTANCE_KEYS
            if isinstance(item, Mapping) and key in item
        }
        for item in distances
        if isinstance(item, Mapping)
    ]
  if private_protocol_audit:
    result["protocol_receipt"] = public_protocol_receipt(
        private_protocol_audit
    )
  if str(result.get("input_protocol") or "") == "benchmark_v2":
    _validate_opaque_pose_identities(result)
  validate_public_artifact(result)
  return result


def _validate_opaque_pose_identities(row: Mapping[str, Any]) -> None:
  parts = row.get("selected_part_names")
  if isinstance(parts, list) and any(
      not _OPAQUE_PART_ID.fullmatch(str(value)) for value in parts
  ):
    raise ValueError("benchmark_v2 public row contains a non-opaque part identity")
  anchor = row.get("anchor")
  if anchor not in (None, "") and not _OPAQUE_PART_ID.fullmatch(str(anchor)):
    raise ValueError("benchmark_v2 public row contains a non-opaque anchor")
  for key in (
      "realized_constraint_edges",
      "missing_expected_edges",
      "extra_constraint_edges",
  ):
    edges = row.get(key)
    if not isinstance(edges, list):
      continue
    for edge in edges:
      endpoints = str(edge).split("--")
      if len(endpoints) != 2 or any(
          not _OPAQUE_PART_ID.fullmatch(endpoint) for endpoint in endpoints
      ):
        raise ValueError(
            "benchmark_v2 public row contains a non-opaque edge identity"
        )
  distances = row.get("exact_expected_contact_distances")
  if isinstance(distances, list):
    for distance in distances:
      if not isinstance(distance, Mapping):
        continue
      for key in ("part_a", "part_b"):
        if key in distance and not _OPAQUE_PART_ID.fullmatch(
            str(distance[key])
        ):
          raise ValueError(
              "benchmark_v2 public contact row contains a non-opaque part identity"
          )


def public_batch_result(
    payload: Mapping[str, Any],
    *,
    private_protocol_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Return a metrics-only public view of a generic batch pipeline result."""

  allowed = {
      "id",
      "benchmark_task",
      "input_protocol",
      "success",
      "raw_success",
      "tolerant_success",
      "strict_success",
      "failure_taxonomy",
      "collision_volume_sum",
      "blocking_collision_volume_sum",
      "tolerated_collision_volume_sum",
      "tolerated_collision_count",
      "self_correction_count",
  }
  result = {key: copy.deepcopy(payload[key]) for key in allowed if key in payload}
  for metrics_name in ("contact_metrics", "retrieval_metrics", "contact_coverage"):
    raw_metrics = payload.get(metrics_name)
    if not isinstance(raw_metrics, Mapping):
      continue
    scalar_metrics = {
        str(key): value
        for key, value in raw_metrics.items()
        if isinstance(value, (bool, int, float))
    }
    if scalar_metrics:
      result[metrics_name] = scalar_metrics
  evaluation = payload.get("evaluation")
  if isinstance(evaluation, Mapping) and isinstance(
      evaluation.get("score"), (int, float)
  ):
    result["evaluation_score"] = float(evaluation["score"])
  if private_protocol_audit:
    result["protocol_receipt"] = public_protocol_receipt(
        private_protocol_audit
    )
  validate_public_artifact(result)
  return result


def validate_public_artifact(value: Any, *, location: str = "$") -> None:
  """Reject sensitive keys, absolute paths, and non-finite numeric values."""

  if isinstance(value, Mapping):
    for raw_key, item in value.items():
      key = str(raw_key)
      lowered = key.lower()
      if any(fragment in lowered for fragment in _FORBIDDEN_KEY_FRAGMENTS):
        raise ValueError(f"public artifact contains forbidden key at {location}.{key}")
      validate_public_artifact(item, location=f"{location}.{key}")
    return
  if isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      validate_public_artifact(item, location=f"{location}[{index}]")
    return
  if isinstance(value, bool) or value is None:
    return
  if isinstance(value, (int, float)):
    if not math.isfinite(float(value)):
      raise ValueError(f"public artifact contains non-finite number at {location}")
    return
  if isinstance(value, str):
    if (
        _WINDOWS_ABSOLUTE_PATH.match(value)
        or value.startswith("\\\\")
        or value.startswith("/")
    ):
      raise ValueError(f"public artifact contains an absolute path at {location}")
