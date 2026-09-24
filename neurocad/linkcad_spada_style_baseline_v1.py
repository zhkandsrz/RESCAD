"""SPADA-style compile--test--repair baseline for LinkCAD.

The baseline starts from the frozen pure-LLM assembly, executes that assembly
in the CAD kernel, and gives only target-free geometric test failures back to
the same frozen language model.  Each repair predicts a complete replacement
assembly.  Part selection is held fixed so that the repair loop cannot inspect
private candidate identities or evaluation targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .linkcad_generalist_agent_baseline_v1 import (
    DEFAULT_MODEL,
    FrozenGeneralistCADAgentV1,
)
from .linkcad_pure_llm_baseline_v1 import (
    canonical_sha256,
    validate_direct_pose_response_v1,
)


SCHEMA_VERSION = "linkcad_spada_style_repair.v1"
PROMPT_VERSION = "linkcad_spada_style_repair_prompt.v1"


def system_prompt_v1() -> str:
  return """You are a test-driven CAD assembly repair agent. You receive a public CAD assembly request, your previous complete assembly prediction, and deterministic CAD-kernel test results. Return one corrected complete assembly prediction. Use only the listed interface primitive ordinals and keep candidate_by_role unchanged. The anchor role, which is the lexicographically first role, must remain at the identity transform. A world transform maps a local column point p to R p + t in millimetres. Reduce every reported interface angle, distance, and axial-gap error below its threshold, remove every reported positive-volume collision, and preserve each requested mobility. Return exactly one JSON object and no prose with fields query_id, candidate_by_role, edge_programs, and role_pose_row_major. edge_programs contains edge_id, interface_primitive_a, interface_primitive_b, and mobility. role_pose_row_major maps every role to a row-major rigid homogeneous 4x4 matrix of 16 finite numbers."""


def prompt_sha256_v1() -> str:
  return hashlib.sha256(system_prompt_v1().encode("utf-8")).hexdigest()


def public_test_report_v1(manifest: Mapping[str, Any]) -> dict[str, Any]:
  """Project an execution manifest to target-free geometric test feedback."""

  verification = manifest.get("verification", {})
  collisions = []
  for row in verification.get("collision_pairs", []):
    if row.get("positive_volume_collision") is True:
      collisions.append({
          "role_a": str(row["role_a"]),
          "role_b": str(row["role_b"]),
          "whole_solid_common_volume_mm3": float(
              row["whole_solid_common_volume_mm3"]
          ),
      })
  connections = []
  for row in manifest.get("connections", []):
    observation = row.get("selected_observation", {})
    connections.append({
        "edge_id": str(row["edge_id"]),
        "role_a": str(row["role_a"]),
        "role_b": str(row["role_b"]),
        "interface_primitive_a": int(row["interface_primitive_a"]),
        "interface_primitive_b": int(row["interface_primitive_b"]),
        "requested_mobility": str(row["requested_mobility"]),
        "predicted_mobility": str(row["predicted_mobility"]),
        "mobility_correct": row.get("mobility_correct") is True,
        "interface_satisfied": row.get("connection_satisfied") is True,
        "support_family": str(row.get("support_family", "unknown")),
        "axis_angle_error_deg": observation.get("axis_angle_error_deg"),
        "interface_distance_error_mm": observation.get(
            "interface_distance_error_mm"
        ),
        "axial_interval_gap_mm": observation.get("axial_interval_gap_mm"),
    })
  return {
      "schema_version": "linkcad_public_geometry_tests.v1",
      "query_id": str(manifest["query_id"]),
      "thresholds": {
          "axis_angle_error_deg": 1.0,
          "interface_distance_error_mm": 0.1,
          "axial_interval_gap_mm": 0.1,
          "positive_collision_volume_mm3": 1e-7,
      },
      "all_connections_satisfied": verification.get(
          "all_connections_satisfied"
      ) is True,
      "raw_collision_free": verification.get("raw_collision_free") is True,
      "connections": connections,
      "positive_volume_collisions": collisions,
      "contains_private_targets": False,
  }


@dataclass(frozen=True, slots=True)
class SPADAStyleRepairResultV1:
  prediction: dict[str, Any]
  receipt: dict[str, Any]


class FrozenSPADAStyleRepairAgentV1(FrozenGeneralistCADAgentV1):
  """Zero-temperature repair call with deterministic geometric feedback."""

  def repair(
      self, *, request: Mapping[str, Any], previous_prediction: Mapping[str, Any],
      test_report: Mapping[str, Any], repair_round: int,
  ) -> SPADAStyleRepairResultV1:
    if repair_round < 1:
      raise ValueError("SPADA-style repair round must be positive")
    if test_report.get("contains_private_targets") is not False:
      raise ValueError("SPADA-style feedback target scope differs")
    user_payload = {
        "schema_version": SCHEMA_VERSION,
        "repair_round": int(repair_round),
        "assembly_request": request,
        "previous_prediction": previous_prediction,
        "cad_kernel_tests": test_report,
    }
    payload = {
        "model": self.model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt_v1()},
            {"role": "user", "content": json.dumps(
                user_payload, ensure_ascii=False, sort_keys=True,
            )},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    raw = self.transport(payload) if self.transport else self._remote_call(payload)
    if isinstance(raw, Mapping) and set(raw) == {
        "query_id", "candidate_by_role", "edge_programs",
        "role_pose_row_major",
    }:
      response = dict(raw)
    else:
      response = self._extract_response(raw)
    prediction = validate_direct_pose_response_v1(request, response)
    return SPADAStyleRepairResultV1(prediction=prediction, receipt={
        "schema_version": SCHEMA_VERSION,
        "provider": "deepseek",
        "model": self.model,
        "temperature": 0.0,
        "repair_round": int(repair_round),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256_v1(),
        "request_sha256": canonical_sha256(user_payload),
        "response_sha256": canonical_sha256(prediction),
        "credential_material_included": False,
        "private_targets_opened": False,
    })


__all__ = [
    "DEFAULT_MODEL", "FrozenSPADAStyleRepairAgentV1", "PROMPT_VERSION",
    "SCHEMA_VERSION", "SPADAStyleRepairResultV1", "prompt_sha256_v1",
    "public_test_report_v1", "system_prompt_v1",
]
