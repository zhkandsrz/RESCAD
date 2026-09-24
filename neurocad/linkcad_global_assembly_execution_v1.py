"""Target-free composition and exact collision audit for LinkCAD programs.

Pair execution places the ``role_b`` candidate in the local frame of
``role_a``.  This module composes those public relative transforms into one
assembly frame, checks cycle closure when the functional graph has cycles, and
then audits every unordered solid pair.  Missing execution evidence and kernel
failures remain unknown; they are never converted into collision-free labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark_v2_training_provenance import capture_file_artifact
from .cadquery_backend import (
    boolean_intersection,
    load_step_shape,
    shape_volume,
    transform_shape,
)
from .domain_types import Transform


SCHEMA_VERSION = "linkcad_global_assembly_execution.v1"


def _sha(value: Any) -> str:
  raw = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(raw).hexdigest()


def _execution_payload(execution: Mapping[str, Any]) -> Mapping[str, Any]:
  payload = execution.get("execution")
  if payload is None:
    return execution
  if (
      not isinstance(payload, Mapping)
      or execution.get("private_targets_read_by_executor") is not False
  ):
    raise ValueError("LinkCAD global execution wrapper differs")
  return payload


def _matrix(row_major: Sequence[float]) -> np.ndarray:
  if len(row_major) != 16:
    raise ValueError("LinkCAD relative pose must contain sixteen values")
  matrix = np.asarray(row_major, dtype=float).reshape(4, 4)
  if not np.isfinite(matrix).all() or not np.allclose(
      matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8,
  ):
    raise ValueError("LinkCAD relative pose is not homogeneous")
  rotation = matrix[:3, :3]
  if (
      not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
      or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
  ):
    raise ValueError("LinkCAD relative pose is not rigid")
  return matrix


def _inverse(matrix: np.ndarray) -> np.ndarray:
  rotation = matrix[:3, :3]
  translation = matrix[:3, 3]
  result = np.eye(4)
  result[:3, :3] = rotation.T
  result[:3, 3] = -rotation.T @ translation
  return result


def _rotation_error_radians(left: np.ndarray, right: np.ndarray) -> float:
  delta = left[:3, :3].T @ right[:3, :3]
  cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
  return float(math.acos(cosine))


def _bbox_overlap_upper(left: Any, right: Any) -> float:
  a, b = left.BoundingBox(), right.BoundingBox()
  overlap = np.asarray((
      min(a.xmax, b.xmax) - max(a.xmin, b.xmin),
      min(a.ymax, b.ymax) - max(a.ymin, b.ymin),
      min(a.zmax, b.zmax) - max(a.zmin, b.zmin),
  ))
  return 0.0 if np.any(overlap <= 0.0) else float(np.prod(overlap))


def _candidate_index(
    query: Mapping[str, Any], hypothesis: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
  selected = {}
  for role_id, rows in query["candidate_sets"].items():
    by_id = {str(row["candidate_id"]): row for row in rows}
    candidate_id = str(hypothesis["candidate_by_role"][role_id])
    if candidate_id not in by_id:
      raise ValueError("LinkCAD global candidate selection differs")
    selected[str(role_id)] = by_id[candidate_id]
  if set(selected) != set(hypothesis["candidate_by_role"]):
    raise ValueError("LinkCAD global role domain differs")
  return selected


def compose_role_poses_v1(
    *, query: Mapping[str, Any], hypothesis: Mapping[str, Any],
    execution_rows: Sequence[Mapping[str, Any]],
    closure_translation_tolerance_mm: float = 1e-5,
    closure_rotation_tolerance_radians: float = 1e-6,
) -> dict[str, Any]:
  """Compose accepted pair poses into one deterministic assembly frame."""

  role_ids = tuple(str(row["role_id"]) for row in query["roles"])
  edge_by_id = {
      str(row["edge_id"]): row for row in query["functional_edges"]
  }
  if (
      not role_ids or len(set(role_ids)) != len(role_ids)
      or not edge_by_id or len(edge_by_id) != len(query["functional_edges"])
  ):
    raise ValueError("LinkCAD global functional graph differs")
  candidates = _candidate_index(query, hypothesis)
  accepted_by_edge: dict[str, Mapping[str, Any]] = {}
  for row in execution_rows:
    edge_id = str(row["edge_id"])
    if edge_id not in edge_by_id or int(row["hypothesis_rank"]) != 0:
      continue
    edge = edge_by_id[edge_id]
    if (
        str(row["candidate_id_a"])
        != candidates[str(edge["role_a"])]["candidate_id"]
        or str(row["candidate_id_b"])
        != candidates[str(edge["role_b"])]["candidate_id"]
    ):
      raise ValueError("LinkCAD global execution identity differs")
    if row.get("conditional_kinematic_feasibility_accepted") is True:
      if edge_id in accepted_by_edge:
        raise ValueError("LinkCAD global accepted edge pose duplicates")
      if not isinstance(row.get("selected_observation"), Mapping):
        raise ValueError("LinkCAD global accepted edge lacks a pose")
      accepted_by_edge[edge_id] = row
  if set(accepted_by_edge) != set(edge_by_id):
    return {
        "status": "unknown_incomplete_pair_execution",
        "role_pose_row_major": {},
        "accepted_edge_count": len(accepted_by_edge),
        "functional_edge_count": len(edge_by_id),
        "closure_checks": [],
    }

  adjacency: dict[str, list[tuple[str, str, np.ndarray]]] = {
      role_id: [] for role_id in role_ids
  }
  for edge_id, edge in edge_by_id.items():
    role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
    if role_a not in adjacency or role_b not in adjacency or role_a == role_b:
      raise ValueError("LinkCAD global edge endpoints differ")
    relative = _matrix(accepted_by_edge[edge_id]["selected_observation"][
        "child_world_row_major"
    ])
    adjacency[role_a].append((role_b, edge_id, relative))
    adjacency[role_b].append((role_a, edge_id, _inverse(relative)))

  anchor = min(role_ids)
  world_by_role = {anchor: np.eye(4)}
  queue = [anchor]
  while queue:
    role = queue.pop(0)
    for neighbor, _, neighbor_to_role in sorted(
        adjacency[role], key=lambda item: (item[0], item[1]),
    ):
      proposed = world_by_role[role] @ neighbor_to_role
      if neighbor not in world_by_role:
        world_by_role[neighbor] = proposed
        queue.append(neighbor)
  if set(world_by_role) != set(role_ids):
    return {
        "status": "unknown_disconnected_functional_graph",
        "role_pose_row_major": {
            role: [float(value) for value in pose.reshape(-1)]
            for role, pose in sorted(world_by_role.items())
        },
        "accepted_edge_count": len(accepted_by_edge),
        "functional_edge_count": len(edge_by_id),
        "closure_checks": [],
    }

  closure_checks = []
  for edge_id, edge in sorted(edge_by_id.items()):
    role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
    relative = _matrix(accepted_by_edge[edge_id]["selected_observation"][
        "child_world_row_major"
    ])
    expected_b = world_by_role[role_a] @ relative
    translation_error = float(np.linalg.norm(
        expected_b[:3, 3] - world_by_role[role_b][:3, 3]
    ))
    rotation_error = _rotation_error_radians(
        expected_b, world_by_role[role_b]
    )
    closure_checks.append({
        "edge_id": edge_id,
        "translation_error_mm": translation_error,
        "rotation_error_radians": rotation_error,
        "within_tolerance": (
            translation_error <= closure_translation_tolerance_mm
            and rotation_error <= closure_rotation_tolerance_radians
        ),
    })
  status = (
      "composed"
      if all(row["within_tolerance"] for row in closure_checks)
      else "unknown_pose_graph_closure_failure"
  )
  return {
      "status": status,
      "anchor_role": anchor,
      "role_pose_row_major": {
          role: [float(value) for value in pose.reshape(-1)]
          for role, pose in sorted(world_by_role.items())
      },
      "accepted_edge_count": len(accepted_by_edge),
      "functional_edge_count": len(edge_by_id),
      "closure_checks": closure_checks,
  }


def execute_global_assembly_query_v1(
    *, query: Mapping[str, Any], hypothesis: Mapping[str, Any],
    execution_rows: Sequence[Mapping[str, Any]], dataset_root: str | Path,
    collision_tolerance_mm3: float = 1e-7,
) -> dict[str, Any]:
  """Compose a Top-1 program and audit every solid pair in one frame."""

  started = time.monotonic()
  query_id = str(query["query_id"])
  base = {
      "query_id": query_id,
      "private_targets_opened": False,
      "collision_tolerance_mm3": collision_tolerance_mm3,
  }
  try:
    composition = compose_role_poses_v1(
        query=query, hypothesis=hypothesis, execution_rows=execution_rows,
    )
    if composition["status"] != "composed":
      result = {
          **base, "status": composition["status"],
          "global_assembly_conditionally_feasible": False,
          "composition": composition, "pair_observations": [],
      }
    else:
      candidates = _candidate_index(query, hypothesis)
      root = Path(dataset_root).resolve()
      world_shapes = {}
      for role_id, candidate in sorted(candidates.items()):
        path = (root / str(candidate["step_path"])).resolve()
        capture = capture_file_artifact(path, label="LinkCAD global STEP")
        if capture.sha256 != str(candidate["step_sha256"]):
          raise ValueError("LinkCAD global STEP bytes differ")
        matrix = _matrix(composition["role_pose_row_major"][role_id])
        world_shapes[role_id] = transform_shape(
            load_step_shape(path),
            Transform(rotation=matrix[:3, :3], translation=matrix[:3, 3]),
        )
      functional_pairs = {
          frozenset((str(edge["role_a"]), str(edge["role_b"])))
          for edge in query["functional_edges"]
      }
      observations = []
      role_ids = sorted(world_shapes)
      for left_index, role_a in enumerate(role_ids):
        for role_b in role_ids[left_index + 1:]:
          left, right = world_shapes[role_a], world_shapes[role_b]
          upper = _bbox_overlap_upper(left, right)
          if upper <= collision_tolerance_mm3:
            common_volume = 0.0
            volume_status = "certified_by_aabb_upper_bound"
          else:
            common_volume = max(0.0, float(shape_volume(
                boolean_intersection(left, right)
            )))
            volume_status = "exact_occ_boolean"
          observations.append({
              "role_a": role_a, "role_b": role_b,
              "functional_edge_pair": (
                  frozenset((role_a, role_b)) in functional_pairs
              ),
              "aabb_intersection_upper_mm3": upper,
              "whole_solid_common_volume_mm3": common_volume,
              "common_volume_status": volume_status,
              "collision_free": common_volume <= collision_tolerance_mm3,
          })
      collision_count = sum(
          not row["collision_free"] for row in observations
      )
      non_edge_collision_count = sum(
          not row["collision_free"] and not row["functional_edge_pair"]
          for row in observations
      )
      result = {
          **base,
          "status": (
              "accepted_global_assembly"
              if collision_count == 0 else "rejected_global_collision"
          ),
          "global_assembly_conditionally_feasible": collision_count == 0,
          "composition": composition,
          "pair_observations": observations,
          "pair_count": len(observations),
          "collision_pair_count": collision_count,
          "non_edge_collision_pair_count": non_edge_collision_count,
      }
  except Exception as error:
    result = {
        **base, "status": "unknown_pre_execution_error",
        "global_assembly_conditionally_feasible": False,
        "composition": None, "pair_observations": [],
        "failure_reason": f"{type(error).__name__}:{error}",
    }
  result["elapsed_seconds"] = time.monotonic() - started
  result["result_payload_sha256"] = _sha(result)
  return result


def _pose_options(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
  source = row.get("accepted_observations")
  if not isinstance(source, list) or not source:
    source = [row.get("selected_observation")]
  options = []
  identities = set()
  for observation in source:
    if not isinstance(observation, Mapping):
      continue
    matrix = _matrix(observation["child_world_row_major"])
    identity = tuple(np.round(matrix.reshape(-1), 12))
    if identity not in identities:
      identities.add(identity)
      options.append(observation)
  return options


def search_global_assembly_query_v1(
    *, query: Mapping[str, Any], hypothesis: Mapping[str, Any],
    execution_rows: Sequence[Mapping[str, Any]], dataset_root: str | Path,
    collision_tolerance_mm3: float = 1e-7,
    maximum_global_states: int = 512,
) -> dict[str, Any]:
  """Search finite public poses for one collision-free, cycle-consistent assembly."""

  if maximum_global_states < 1:
    raise ValueError("LinkCAD global state budget must be positive")
  started = time.monotonic()
  query_id = str(query["query_id"])
  base = {
      "query_id": query_id,
      "private_targets_opened": False,
      "collision_tolerance_mm3": collision_tolerance_mm3,
      "maximum_global_states": maximum_global_states,
  }
  try:
    role_ids = tuple(str(row["role_id"]) for row in query["roles"])
    edge_by_id = {
        str(row["edge_id"]): row for row in query["functional_edges"]
    }
    candidates = _candidate_index(query, hypothesis)
    accepted_by_edge: dict[str, Mapping[str, Any]] = {}
    for row in execution_rows:
      edge_id = str(row["edge_id"])
      if (
          edge_id not in edge_by_id or int(row["hypothesis_rank"]) != 0
          or row.get("conditional_kinematic_feasibility_accepted") is not True
      ):
        continue
      edge = edge_by_id[edge_id]
      if (
          str(row["candidate_id_a"])
          != candidates[str(edge["role_a"])]["candidate_id"]
          or str(row["candidate_id_b"])
          != candidates[str(edge["role_b"])]["candidate_id"]
      ):
        raise ValueError("LinkCAD global search execution identity differs")
      if edge_id in accepted_by_edge:
        raise ValueError("LinkCAD global search accepted edge duplicates")
      accepted_by_edge[edge_id] = row
    if set(accepted_by_edge) != set(edge_by_id):
      result = {
          **base, "status": "unknown_incomplete_pair_execution",
          "global_assembly_conditionally_feasible": False,
          "search_status": "incomplete_pair_execution",
          "pair_observations": [], "composition": None,
      }
    else:
      adjacency: dict[str, list[tuple[str, str]]] = {
          role: [] for role in role_ids
      }
      for edge_id, edge in edge_by_id.items():
        role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
        adjacency[role_a].append((role_b, edge_id))
        adjacency[role_b].append((role_a, edge_id))
      anchor = min(role_ids)
      visited = {anchor}
      queue = [anchor]
      traversal = []
      tree_edge_ids = set()
      while queue:
        parent = queue.pop(0)
        for child, edge_id in sorted(adjacency[parent]):
          if child in visited:
            continue
          visited.add(child)
          queue.append(child)
          traversal.append((parent, child, edge_id))
          tree_edge_ids.add(edge_id)
      if set(visited) != set(role_ids):
        raise ValueError("LinkCAD global search graph is disconnected")
      closure_edge_ids = sorted(set(edge_by_id) - tree_edge_ids)

      root = Path(dataset_root).resolve()
      local_shapes = {}
      for role_id, candidate in sorted(candidates.items()):
        path = (root / str(candidate["step_path"])).resolve()
        capture = capture_file_artifact(path, label="LinkCAD global search STEP")
        if capture.sha256 != str(candidate["step_sha256"]):
          raise ValueError("LinkCAD global search STEP bytes differ")
        local_shapes[role_id] = load_step_shape(path)

      states = [{
          "world_by_role": {anchor: np.eye(4)},
          "shape_by_role": {anchor: local_shapes[anchor]},
          "chosen_by_edge": {}, "choice_ranks": (),
      }]
      explored_state_count = 1
      collision_pruned_state_count = 0
      closure_pruned_state_count = 0
      kernel_error_count = 0
      truncated_state_count = 0
      exhausted_stage = "tree_placement"
      transform_cache: dict[tuple[str, tuple[float, ...]], Any] = {}
      for parent, child, edge_id in traversal:
        edge = edge_by_id[edge_id]
        role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
        options = _pose_options(accepted_by_edge[edge_id])
        if not options:
          states = []
          break
        next_states = []
        for state in states:
          for option_rank, observation in enumerate(options):
            relative = _matrix(observation["child_world_row_major"])
            child_to_parent = relative if (
                parent == role_a and child == role_b
            ) else _inverse(relative)
            child_world = state["world_by_role"][parent] @ child_to_parent
            cache_key = (
                child, tuple(np.round(child_world.reshape(-1), 12))
            )
            try:
              moved = transform_cache.get(cache_key)
              if moved is None:
                moved = transform_shape(
                    local_shapes[child], Transform(
                        rotation=child_world[:3, :3],
                        translation=child_world[:3, 3],
                    ),
                )
                transform_cache[cache_key] = moved
              collides = False
              for placed_shape in state["shape_by_role"].values():
                upper = _bbox_overlap_upper(moved, placed_shape)
                if upper <= collision_tolerance_mm3:
                  continue
                common = max(0.0, float(shape_volume(
                    boolean_intersection(moved, placed_shape)
                )))
                if common > collision_tolerance_mm3:
                  collides = True
                  break
            except Exception:
              kernel_error_count += 1
              continue
            explored_state_count += 1
            if collides:
              collision_pruned_state_count += 1
              continue
            next_states.append({
                "world_by_role": {
                    **state["world_by_role"], child: child_world,
                },
                "shape_by_role": {
                    **state["shape_by_role"], child: moved,
                },
                "chosen_by_edge": {
                    **state["chosen_by_edge"], edge_id: observation,
                },
                "choice_ranks": state["choice_ranks"] + (option_rank,),
            })
        next_states.sort(key=lambda state: state["choice_ranks"])
        if len(next_states) > maximum_global_states:
          truncated_state_count += len(next_states) - maximum_global_states
          next_states = next_states[:maximum_global_states]
        states = next_states
        if not states:
          break

      if states and closure_edge_ids:
        exhausted_stage = "cycle_closure"
        for edge_id in closure_edge_ids:
          edge = edge_by_id[edge_id]
          role_a, role_b = str(edge["role_a"]), str(edge["role_b"])
          options = _pose_options(accepted_by_edge[edge_id])
          if not options:
            states = []
            break
          next_states = []
          for state in states:
            for option_rank, observation in enumerate(options):
              relative = _matrix(observation["child_world_row_major"])
              expected_b = state["world_by_role"][role_a] @ relative
              actual_b = state["world_by_role"][role_b]
              translation_error = float(np.linalg.norm(
                  expected_b[:3, 3] - actual_b[:3, 3]
              ))
              rotation_error = _rotation_error_radians(expected_b, actual_b)
              explored_state_count += 1
              if translation_error > 1e-5 or rotation_error > 1e-6:
                closure_pruned_state_count += 1
                continue
              next_states.append({
                  **state,
                  "chosen_by_edge": {
                      **state["chosen_by_edge"], edge_id: observation,
                  },
                  "choice_ranks": state["choice_ranks"] + (option_rank,),
              })
          next_states.sort(key=lambda state: state["choice_ranks"])
          if len(next_states) > maximum_global_states:
            truncated_state_count += len(next_states) - maximum_global_states
            next_states = next_states[:maximum_global_states]
          states = next_states
          if not states:
            break

      if not states:
        if kernel_error_count:
          status = "unknown_global_search_kernel_failure"
        elif truncated_state_count:
          status = "unknown_bounded_no_global_pose"
        elif exhausted_stage == "cycle_closure":
          status = "rejected_no_cycle_consistent_global_pose"
        else:
          status = "rejected_no_collision_free_global_pose"
        result = {
            **base, "status": status,
            "global_assembly_conditionally_feasible": False,
            "search_status": f"exhausted_during_{exhausted_stage}",
            "pair_observations": [], "composition": None,
            "explored_state_count": explored_state_count,
            "collision_pruned_state_count": collision_pruned_state_count,
            "closure_pruned_state_count": closure_pruned_state_count,
            "closure_edge_count": len(closure_edge_ids),
            "kernel_error_count": kernel_error_count,
            "truncated_state_count": truncated_state_count,
        }
      else:
        selected = states[0]
        selected_rows = []
        for edge_id, source in accepted_by_edge.items():
          row = dict(source)
          row["selected_observation"] = selected["chosen_by_edge"][edge_id]
          selected_rows.append(row)
        composition = compose_role_poses_v1(
            query=query, hypothesis=hypothesis,
            execution_rows=selected_rows,
        )
        observations = []
        functional_pairs = {
            frozenset((str(edge["role_a"]), str(edge["role_b"])))
            for edge in query["functional_edges"]
        }
        ordered_roles = sorted(selected["shape_by_role"])
        for left_index, role_a in enumerate(ordered_roles):
          for role_b in ordered_roles[left_index + 1:]:
            left = selected["shape_by_role"][role_a]
            right = selected["shape_by_role"][role_b]
            upper = _bbox_overlap_upper(left, right)
            if upper <= collision_tolerance_mm3:
              common_volume = 0.0
              volume_status = "certified_by_aabb_upper_bound"
            else:
              common_volume = max(0.0, float(shape_volume(
                  boolean_intersection(left, right)
              )))
              volume_status = "exact_occ_boolean"
            observations.append({
                "role_a": role_a, "role_b": role_b,
                "functional_edge_pair": (
                    frozenset((role_a, role_b)) in functional_pairs
                ),
                "aabb_intersection_upper_mm3": upper,
                "whole_solid_common_volume_mm3": common_volume,
                "common_volume_status": volume_status,
                "collision_free": common_volume <= collision_tolerance_mm3,
            })
        result = {
            **base, "status": "accepted_global_assembly",
            "global_assembly_conditionally_feasible": True,
            "search_status": "collision_free_pose_found",
            "composition": composition,
            "selected_pose_rank_by_edge": {
                edge_id: selected["chosen_by_edge"][edge_id].get(
                    "pose_rank", 0
                )
                for edge_id in sorted(selected["chosen_by_edge"])
            },
            "pair_observations": observations,
            "pair_count": len(observations), "collision_pair_count": 0,
            "non_edge_collision_pair_count": 0,
            "explored_state_count": explored_state_count,
            "collision_pruned_state_count": collision_pruned_state_count,
            "closure_pruned_state_count": closure_pruned_state_count,
            "closure_edge_count": len(closure_edge_ids),
            "kernel_error_count": kernel_error_count,
            "truncated_state_count": truncated_state_count,
        }
  except Exception as error:
    result = {
        **base, "status": "unknown_pre_execution_error",
        "global_assembly_conditionally_feasible": False,
        "search_status": "pre_execution_error",
        "composition": None, "pair_observations": [],
        "failure_reason": f"{type(error).__name__}:{error}",
    }
  result["elapsed_seconds"] = time.monotonic() - started
  result["result_payload_sha256"] = _sha(result)
  return result


def execute_global_assembly_subset_v1(
    *, public: Mapping[str, Any], predictions: Mapping[str, Any],
    pair_execution: Mapping[str, Any], dataset_root: str | Path,
    query_limit: int | None = None,
) -> dict[str, Any]:
  """Audit Top-1 assemblies using public predictions and pair receipts."""

  execution = _execution_payload(pair_execution)
  if (
      public.get("contains_private_targets") is not False
      or predictions.get("private_targets_opened") is not False
      or execution.get("private_targets_opened") is not False
      or execution.get("prediction_payload_sha256")
      != predictions.get("prediction_payload_sha256")
  ):
    raise ValueError("LinkCAD global public evidence scope differs")
  forbidden = (
      "target_candidate_by_role", "target_candidate_id", "target_mobility",
      "source_joint_id", "source_joint_type",
  )
  if any(key in repr((public, predictions, execution)) for key in forbidden):
    raise ValueError("LinkCAD global execution contains private target material")
  public_by_id = {str(row["query_id"]): row for row in public["queries"]}
  prediction_by_id = {
      str(row["query_id"]): row for row in predictions["rows"]
  }
  if set(public_by_id) != set(prediction_by_id):
    raise ValueError("LinkCAD global query domain differs")
  selected_ids = sorted(public_by_id)
  if query_limit is not None:
    if query_limit < 1:
      raise ValueError("LinkCAD global query limit differs")
    selected_ids = selected_ids[:query_limit]
  execution_by_query: dict[str, list[Mapping[str, Any]]] = {}
  for row in execution["rows"]:
    execution_by_query.setdefault(str(row["query_id"]), []).append(row)
  results = []
  for query_id in selected_ids:
    prediction_row = prediction_by_id[query_id]
    if not prediction_row["hypotheses"]:
      result = {
          "query_id": query_id, "private_targets_opened": False,
          "status": "unknown_no_hypothesis",
          "global_assembly_conditionally_feasible": False,
          "composition": None, "pair_observations": [],
          "elapsed_seconds": 0.0,
      }
      result["result_payload_sha256"] = _sha(result)
    else:
      result = search_global_assembly_query_v1(
          query=public_by_id[query_id],
          hypothesis=prediction_row["hypotheses"][0],
          execution_rows=execution_by_query.get(query_id, ()),
          dataset_root=dataset_root,
      )
    results.append(result)
  status_counts: dict[str, int] = {}
  for row in results:
    status = str(row["status"])
    status_counts[status] = status_counts.get(status, 0) + 1
  payload = {
      "schema_version": SCHEMA_VERSION,
      "scope": "public_top1_global_pose_and_all_pair_collision_audit",
      "private_targets_opened": False,
      "prediction_payload_sha256": predictions["prediction_payload_sha256"],
      "pair_execution_subset_payload_sha256": execution[
          "subset_payload_sha256"
      ],
      "query_count": len(results),
      "accepted_query_count": sum(
          row["global_assembly_conditionally_feasible"] is True
          for row in results
      ),
      "status_counts": dict(sorted(status_counts.items())),
      "rows": results,
  }
  payload["global_execution_payload_sha256"] = _sha(payload)
  return payload


__all__ = [
    "SCHEMA_VERSION", "compose_role_poses_v1",
    "execute_global_assembly_query_v1", "execute_global_assembly_subset_v1",
    "search_global_assembly_query_v1",
]
