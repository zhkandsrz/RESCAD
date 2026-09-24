"""Pure-LLM end-to-end baseline for LinkCAD.

The language model selects parts, interface primitives, and one world pose for
every selected part.  No LinkCAD scorer, pose generator, search, or repair is
used after the response.  A CAD kernel may subsequently *score* that response,
but it is not allowed to change it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cadquery_backend import load_step_shape
from .linkcad_generalist_agent_baseline_v1 import (
    DEFAULT_MODEL,
    FrozenGeneralistCADAgentV1,
)
from .linkcad_kinematic_execution_v2 import _edge_frame, _face_frame
from .linkcad_primitive_graph_v2 import extract_primitive_graph_v2


SCHEMA_VERSION = "linkcad_pure_llm_prediction.v1"
PROMPT_VERSION = "linkcad_pure_llm_direct_pose_prompt.v1"


def _canonical(value: Any) -> str:
  return json.dumps(
      value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
      allow_nan=False,
  )


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def system_prompt_v1() -> str:
  return """You are a pure CAD assembly baseline. You receive selected STEP parts, natural-language connection instructions, and local geometric interface frames. Directly choose one listed interface primitive at each connection endpoint and directly predict one rigid world transform for every role. The evaluator will execute your answer exactly once: there is no geometric search, snapping, collision repair, or alternative pose selection. Keep the lexicographically first role at the identity transform. A world transform maps a local column point p to R p + t in millimetres. Use the instructions to make requested interfaces coincide, keep parts free of positive-volume intersections, and preserve the requested motion. Return exactly one JSON object and no prose with fields query_id, candidate_by_role, edge_programs, and role_pose_row_major. edge_programs contains edge_id, interface_primitive_a, interface_primitive_b, and mobility. role_pose_row_major maps every role to a row-major homogeneous 4x4 matrix of 16 finite numbers. Preserve every supplied ID exactly and use only listed executable primitive ordinals."""


def prompt_sha256_v1() -> str:
  return hashlib.sha256(system_prompt_v1().encode("utf-8")).hexdigest()


def _candidate_index(
    query: Mapping[str, Any], candidate_by_role: Mapping[str, str],
) -> dict[str, Mapping[str, Any]]:
  roles = {str(row["role_id"]) for row in query["roles"]}
  if set(candidate_by_role) != roles:
    raise ValueError("Pure-LLM selected role domain differs")
  selected = {}
  for role_id in sorted(roles):
    by_id = {
        str(row["candidate_id"]): row
        for row in query["candidate_sets"][role_id]
    }
    candidate_id = str(candidate_by_role[role_id])
    if candidate_id not in by_id:
      raise ValueError("Pure-LLM selected candidate differs")
    selected[role_id] = by_id[candidate_id]
  return selected


def frozen_selection_v1(
    query: Mapping[str, Any], rankings: Mapping[str, Sequence[str]],
) -> dict[str, str]:
  """Use the previously frozen LLM ranking without symbolic post-processing."""

  result = {}
  for role in query["roles"]:
    role_id = str(role["role_id"])
    if role_id not in rankings or not rankings[role_id]:
      raise ValueError("Pure-LLM frozen ranking is incomplete")
    result[role_id] = str(rankings[role_id][0])
  _candidate_index(query, result)
  return result


def _frame_payload(frame: Any) -> dict[str, Any]:
  return {
      "geometry_type": str(frame.geometry_type).lower(),
      "origin_local_mm": [round(float(v), 6) for v in frame.origin_local_mm],
      "rotation_local_row_major": [
          round(float(v), 7) for v in frame.rotation_local
      ],
  }


def _part_geometry_summary(
    *, candidate: Mapping[str, Any], dataset_root: Path,
) -> dict[str, Any]:
  source = (dataset_root / str(candidate["step_path"])).resolve()
  try:
    source.relative_to(dataset_root)
  except ValueError as error:
    raise ValueError("Pure-LLM STEP escapes dataset root") from error
  shape = load_step_shape(source)
  graph = extract_primitive_graph_v2(shape)
  faces, edges = list(shape.Faces()), list(shape.Edges())
  primitives = []
  for ordinal, (kind, members) in enumerate(zip(
      graph.primitive_kinds, graph.primitive_members, strict=True,
  )):
    frames = []
    for raw_index in members:
      try:
        frame = (
            _face_frame(faces[raw_index], raw_index)
            if kind == "face" else _edge_frame(edges[raw_index], raw_index)
        )
      except ValueError:
        continue
      frames.append(_frame_payload(frame))
    if frames:
      primitives.append({
          "primitive_ordinal": ordinal,
          "kind": str(kind),
          "type": str(graph.primitive_type(ordinal)),
          "symmetric_member_count": len(members),
          "representative_frames": frames,
      })
  box = shape.BoundingBox()
  return {
      "candidate_id": str(candidate["candidate_id"]),
      "volume_mm3": round(float(candidate.get("volume", 0.0)), 5),
      "area_mm2": round(float(candidate.get("area", 0.0)), 5),
      "aabb_local_mm": {
          "minimum": [round(float(v), 5) for v in (box.xmin, box.ymin, box.zmin)],
          "maximum": [round(float(v), 5) for v in (box.xmax, box.ymax, box.zmax)],
      },
      "executable_primitives": primitives,
  }


def build_direct_pose_request_v1(
    *, query: Mapping[str, Any], candidate_by_role: Mapping[str, str],
    dataset_root: str | Path,
) -> dict[str, Any]:
  root = Path(dataset_root).resolve()
  selected = _candidate_index(query, candidate_by_role)
  descriptions = {
      str(row["role_id"]): str(row["description"]) for row in query["roles"]
  }
  parts = []
  for role_id in sorted(selected):
    parts.append({
        "role_id": role_id,
        "role_description": descriptions[role_id],
        **_part_geometry_summary(candidate=selected[role_id], dataset_root=root),
    })
  connections = [{
      "edge_id": str(row["edge_id"]),
      "role_a": str(row["role_a"]),
      "role_b": str(row["role_b"]),
      "instruction": str(row["instruction"]),
      "requested_mobility": str(row["requested_mobility"]),
  } for row in query["functional_edges"]]
  return {
      "schema_version": "linkcad_pure_llm_public_request.v1",
      "query_id": str(query["query_id"]),
      "candidate_by_role": dict(sorted(candidate_by_role.items())),
      "selected_parts": parts,
      "connections": connections,
      "contains_private_targets": False,
  }


def _rigid_matrix(values: Any) -> list[float]:
  if not isinstance(values, list) or len(values) != 16:
    raise ValueError("Pure-LLM pose must contain sixteen values")
  try:
    matrix = np.asarray([float(value) for value in values], dtype=float).reshape(4, 4)
  except (TypeError, ValueError) as error:
    raise ValueError("Pure-LLM pose values differ") from error
  if not np.isfinite(matrix).all() or not np.allclose(
      matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5,
  ):
    raise ValueError("Pure-LLM pose is not homogeneous")
  rotation = matrix[:3, :3]
  if (
      not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
      or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3)
  ):
    raise ValueError("Pure-LLM pose is not rigid")
  return [float(value) for value in matrix.reshape(-1)]


def validate_direct_pose_response_v1(
    request: Mapping[str, Any], response: Mapping[str, Any],
) -> dict[str, Any]:
  required = {
      "query_id", "candidate_by_role", "edge_programs", "role_pose_row_major",
  }
  if not isinstance(response, Mapping) or set(response) != required:
    raise ValueError("Pure-LLM response fields differ")
  if str(response["query_id"]) != str(request["query_id"]):
    raise ValueError("Pure-LLM response query differs")
  expected_candidates = {
      str(k): str(v) for k, v in request["candidate_by_role"].items()
  }
  candidates = {
      str(k): str(v) for k, v in dict(response["candidate_by_role"]).items()
  }
  if candidates != expected_candidates:
    raise ValueError("Pure-LLM response changed the frozen part selection")
  allowed_ordinals = {
      str(part["role_id"]): {
          int(row["primitive_ordinal"])
          for row in part["executable_primitives"]
      }
      for part in request["selected_parts"]
  }
  edges = {str(row["edge_id"]): row for row in request["connections"]}
  rows = response["edge_programs"]
  if not isinstance(rows, list) or len(rows) != len(edges):
    raise ValueError("Pure-LLM response edge coverage differs")
  programs = []
  seen = set()
  for row in rows:
    if not isinstance(row, Mapping) or set(row) != {
        "edge_id", "interface_primitive_a", "interface_primitive_b", "mobility",
    }:
      raise ValueError("Pure-LLM edge program fields differ")
    edge_id = str(row["edge_id"])
    if edge_id in seen or edge_id not in edges:
      raise ValueError("Pure-LLM edge identity differs")
    seen.add(edge_id)
    edge = edges[edge_id]
    ordinal_a, ordinal_b = int(row["interface_primitive_a"]), int(
        row["interface_primitive_b"]
    )
    if (
        ordinal_a not in allowed_ordinals[str(edge["role_a"])]
        or ordinal_b not in allowed_ordinals[str(edge["role_b"])]
    ):
      raise ValueError("Pure-LLM selected an unavailable primitive")
    programs.append({
        "edge_id": edge_id,
        "interface_primitive_a": ordinal_a,
        "interface_primitive_b": ordinal_b,
        "mobility": str(row["mobility"]),
    })
  poses_raw = response["role_pose_row_major"]
  if not isinstance(poses_raw, Mapping) or set(poses_raw) != set(candidates):
    raise ValueError("Pure-LLM response pose role domain differs")
  poses = {str(role): _rigid_matrix(values) for role, values in poses_raw.items()}
  anchor = min(poses)
  if not np.allclose(np.asarray(poses[anchor]).reshape(4, 4), np.eye(4), atol=2e-3):
    raise ValueError("Pure-LLM anchor pose is not identity")
  return {
      "schema_version": SCHEMA_VERSION,
      "query_id": str(request["query_id"]),
      "candidate_by_role": candidates,
      "edge_programs": sorted(programs, key=lambda row: row["edge_id"]),
      "role_pose_row_major": dict(sorted(poses.items())),
      "contains_private_targets": False,
  }


@dataclass(frozen=True, slots=True)
class PureLLMResultV1:
  prediction: dict[str, Any]
  receipt: dict[str, Any]


class FrozenPureLLMAssemblerV1(FrozenGeneralistCADAgentV1):
  """Zero-temperature direct-pose call using the existing frozen provider."""

  def assemble(self, request: Mapping[str, Any]) -> PureLLMResultV1:
    payload = {
        "model": self.model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt_v1()},
            {"role": "user", "content": json.dumps(
                request, ensure_ascii=False, sort_keys=True,
            )},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    raw = self.transport(payload) if self.transport else self._remote_call(payload)
    if isinstance(raw, Mapping) and set(raw) == {
        "query_id", "candidate_by_role", "edge_programs", "role_pose_row_major",
    }:
      response = dict(raw)
    else:
      response = self._extract_response(raw)
    prediction = validate_direct_pose_response_v1(request, response)
    return PureLLMResultV1(prediction=prediction, receipt={
        "schema_version": "linkcad_pure_llm_call_receipt.v1",
        "provider": "deepseek",
        "model": self.model,
        "temperature": 0.0,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256_v1(),
        "request_sha256": canonical_sha256(request),
        "response_sha256": canonical_sha256(prediction),
        "credential_material_included": False,
        "private_targets_opened": False,
    })


__all__ = [
    "DEFAULT_MODEL", "FrozenPureLLMAssemblerV1", "PROMPT_VERSION",
    "PureLLMResultV1", "SCHEMA_VERSION", "build_direct_pose_request_v1",
    "canonical_sha256", "frozen_selection_v1", "prompt_sha256_v1",
    "system_prompt_v1", "validate_direct_pose_response_v1",
]
