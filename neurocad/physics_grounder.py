"""Physics-guided symbolic grounding for learned CAD interfaces.

The planner decides which parts should connect. This module refines the concrete
interface pair for each edge by trying several local socket pairs and scoring
the resulting SE(3) pose with geometry, not language.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import queue as py_queue
import time
from typing import Any, Optional

import numpy as np

from .cadquery_backend import (
    _safe_exact_boolean_worker,
    _safe_exact_distance_worker,
    boolean_intersection,
    shape_bbox,
    shape_volume,
    transform_shape,
)
from .interface_grounder import InterfacePairCandidate, rank_interface_socket_pairs
from .interface_pair_scorer import InterfacePairScorer
from .math3d import normalize
from .planner import PlannerProtocol
from .solver import SolveError, SymbolicCadSolver
from .domain_types import (
    Constraint,
    ConstraintGraph,
    ConstraintType,
    PartInstance,
    Socket,
    Transform,
)


@dataclass
class PhysicsGroundingConfig:
  input_protocol: str = "legacy"
  max_pairs_per_edge: int = 8
  max_pose_candidates_per_pair: int = 12
  edge_timeout_seconds: float = 60.0
  contact_distance_tolerance: float = 2.0
  collision_epsilon: float = 1e-5
  exact_distance_for_pose_search: bool = False
  exact_collision_for_pose_search: bool = False
  max_pair_collision_volume: float = 25.0
  exact_collision_volume_weight: float = 0.08
  aabb_overlap_weight: float = 0.006
  max_global_aabb_overlap_ratio: float = 0.45
  pair_scorer_top_k: int = 12
  pair_scorer_weight: float = 10.0
  min_grounding_score: float = -6.0
  reject_failed_interface_grounding: bool = True
  skip_obround_slot_grounding: bool = False
  exact_contact_rerank_enabled: bool = True
  exact_contact_rerank_tolerance: float = 0.5
  exact_contact_rerank_timeout_seconds: float = 2.0
  micro_retreat_enabled: bool = True
  micro_retreat_max_distance: float = 2.0
  micro_retreat_binary_steps: int = 5
  micro_retreat_collision_tolerance: float = 1e-4
  micro_retreat_exact_timeout_seconds: float = 3.0


@dataclass
class _PoseCandidate:
  score: float
  transform: Transform
  distance: float
  aabb_overlap_volume: float
  collision_volume: float
  delta: float


class PhysicsGuidedGroundingPlanner:
  """Planner wrapper that refines interface sockets using geometry."""

  def __init__(
      self,
      base_planner: PlannerProtocol,
      *,
      shape_library: dict[str, Any],
      config: Optional[PhysicsGroundingConfig] = None,
      pair_scorer: Optional[InterfacePairScorer] = None,
  ):
    self.base_planner = base_planner
    self.shape_library = dict(shape_library)
    self.config = config or PhysicsGroundingConfig()
    self.input_protocol = str(self.config.input_protocol or "legacy").strip().lower()
    if self.input_protocol not in {"legacy", "benchmark_v2"}:
      raise ValueError("input protocol must be 'legacy' or 'benchmark_v2'")
    if pair_scorer is not None and pair_scorer.model_protocol != self.input_protocol:
      raise ValueError(
          "interface-pair scorer protocol does not match physics input protocol"
      )
    self.pair_scorer = pair_scorer
    self._solver = SymbolicCadSolver(
        enable_axial_snapping=False,
        max_passes=4,
    )

  def plan(
      self,
      instruction: str,
      catalog: dict[str, Any],
      feedback=None,
  ) -> ConstraintGraph:
    graph = self.base_planner.plan(
        instruction=instruction,
        catalog=catalog,
        feedback=feedback,
    )
    refined_constraints: list[Constraint] = []
    notes = list(graph.metadata.get("planner_notes", []))
    refined_count = 0
    used_sockets: set[tuple[str, str]] = set()
    placed: dict[str, PartInstance] = {}
    if graph.anchor in graph.instances:
      anchor_inst = graph.instances[graph.anchor].copy()
      anchor_inst.transform = Transform.identity()
      placed[graph.anchor] = anchor_inst
    for constraint in graph.constraints:
      refined, note = self._refine_constraint(
          graph=graph,
          constraint=constraint,
          used_sockets=used_sockets,
          placed_instances=placed,
      )
      if refined is None:
        if note:
          notes.append(note)
        continue
      refined_constraints.append(refined)
      if note:
        notes.append(note)
      if refined is not constraint:
        refined_count += 1
      used_sockets.add((refined.part_a, refined.socket_a))
      used_sockets.add((refined.part_b, refined.socket_b))
      self._update_placed_instance(
          graph=graph,
          refined=refined,
          placed=placed,
      )
    metadata = dict(graph.metadata)
    metadata["planner_notes"] = notes
    metadata["physics_guided_grounding_count"] = refined_count
    return ConstraintGraph(
        instances=graph.instances,
        constraints=refined_constraints,
        anchor=graph.anchor,
        metadata=metadata,
    )

  def _refine_constraint(
      self,
      graph: ConstraintGraph,
      constraint: Constraint,
      used_sockets: set[tuple[str, str]],
      placed_instances: dict[str, PartInstance],
  ) -> tuple[Optional[Constraint], str]:
    if not bool((constraint.metadata or {}).get("interface_grounding", False)):
      return constraint, ""
    started_at = time.monotonic()
    deadline = _deadline_after(started_at, float(self.config.edge_timeout_seconds))
    part_a = graph.instances.get(constraint.part_a)
    part_b = graph.instances.get(constraint.part_b)
    if part_a is None or part_b is None:
      return constraint, ""
    relation_hint = str(
        (constraint.metadata or {}).get("relation_hint")
        or _constraint_relation_hint(constraint)
    )
    pairs = rank_interface_socket_pairs(
        parent_id=constraint.part_a,
        child_id=constraint.part_b,
        parent_sockets=list(part_a.sockets.values()),
        child_sockets=list(part_b.sockets.values()),
        relation_hint=relation_hint,
        protocol=self.input_protocol,
    )
    if not pairs:
      return constraint, ""
    pairs = [
        pair
        for pair in pairs
        if _passes_hard_geometry_pruning(
            pair,
            skip_obround_slot=bool(self.config.skip_obround_slot_grounding),
        )
    ]
    if not pairs:
      return self._failed_grounding(
          constraint,
          "physics_grounding no_pair_after_hard_pruning",
      )
    pairs = self._apply_pair_scorer(
        pairs=pairs,
        relation_hint=relation_hint,
        parent_id=constraint.part_a,
        child_id=constraint.part_b,
        parent_metadata=part_a.metadata,
        child_metadata=part_b.metadata,
    )

    best: Optional[tuple[float, Constraint, _PoseCandidate, InterfacePairCandidate]] = None
    evaluated_pairs = 0
    for pair in pairs[: max(1, int(self.config.max_pairs_per_edge))]:
      if _deadline_expired(deadline):
        if best is not None:
          break
        return self._failed_grounding(
            constraint,
            "physics_grounding edge_timeout "
            f"seconds={float(self.config.edge_timeout_seconds):.3f} "
            f"evaluated_pairs={evaluated_pairs}",
        )
      if (constraint.part_a, pair.parent_socket.name) in used_sockets:
        continue
      if (constraint.part_b, pair.child_socket.name) in used_sockets:
        continue
      evaluated_pairs += 1
      candidate_constraint = self._candidate_constraint(
          original=constraint,
          pair=pair,
          relation_hint=relation_hint,
      )
      pose = self._best_pose_for_pair(
          parent=part_a,
          child=part_b,
          constraint=candidate_constraint,
          placed_instances=placed_instances,
          deadline=deadline,
      )
      if pose is None:
        continue
      if pose.distance > float(self.config.contact_distance_tolerance):
        continue
      if (
          self.config.exact_collision_for_pose_search
          and pose.collision_volume > float(self.config.max_pair_collision_volume)
      ):
        continue
      score = pair.score + pose.score
      if best is None or score > best[0]:
        best = (score, candidate_constraint, pose, pair)

    if best is None:
      if _deadline_expired(deadline):
        return self._failed_grounding(
            constraint,
            "physics_grounding edge_timeout "
            f"seconds={float(self.config.edge_timeout_seconds):.3f} "
            f"evaluated_pairs={evaluated_pairs}",
        )
      return self._failed_grounding(
          constraint,
          "physics_grounding no_valid_interface_pair "
          f"evaluated_pairs={evaluated_pairs}",
      )
    score, best_constraint, pose, pair = best
    if float(score) < float(self.config.min_grounding_score):
      return self._failed_grounding(
          constraint,
          "physics_grounding rejected_low_score "
          f"score={float(score):.3f} threshold={float(self.config.min_grounding_score):.3f}",
      )
    # Store the axial offset in the same signed convention used by the solver's
    # coincident constraint validation: dot(child_origin - parent_origin,
    # parent_normal) == value.
    best_constraint.value = float(pose.delta)
    best_constraint.metadata["physics_guided_grounding"] = True
    best_constraint.metadata["physics_score"] = float(score)
    best_constraint.metadata["physics_contact_distance"] = float(pose.distance)
    best_constraint.metadata["physics_collision_volume"] = float(pose.collision_volume)
    best_constraint.metadata["physics_axial_delta"] = float(pose.delta)
    note = (
        "physics_grounding "
        f"{constraint.part_a}.{best_constraint.socket_a}->"
        f"{constraint.part_b}.{best_constraint.socket_b} "
        f"score={score:.3f} distance={pose.distance:.4f} "
        f"collision={pose.collision_volume:.6f} delta={pose.delta:.4f} "
        f"from={pair.note}"
    )
    return best_constraint, note

  def _failed_grounding(
      self,
      constraint: Constraint,
      note: str,
  ) -> tuple[Optional[Constraint], str]:
    if bool(self.config.reject_failed_interface_grounding):
      return None, f"{note};edge_skipped"
    return constraint, note

  def _update_placed_instance(
      self,
      *,
      graph: ConstraintGraph,
      refined: Constraint,
      placed: dict[str, PartInstance],
  ) -> None:
    if refined.part_a not in placed or refined.part_b in placed:
      return
    parent = graph.instances.get(refined.part_a)
    child = graph.instances.get(refined.part_b)
    if parent is None or child is None:
      return
    parent_placed = placed.get(refined.part_a)
    if parent_placed is None or parent_placed.transform is None:
      return
    child_copy = child.copy()
    pose = self._best_pose_for_pair(
        parent=parent,
        child=child,
        constraint=refined,
        placed_instances=placed,
        force_socket_pair=True,
    )
    if pose is None:
      return
    child_copy.transform = pose.transform
    placed[refined.part_b] = child_copy

  def _apply_pair_scorer(
      self,
      *,
      pairs: list[InterfacePairCandidate],
      relation_hint: str,
      parent_id: str,
      child_id: str,
      parent_metadata: Optional[dict[str, Any]] = None,
      child_metadata: Optional[dict[str, Any]] = None,
  ) -> list[InterfacePairCandidate]:
    benchmark_v2 = self.input_protocol == "benchmark_v2"
    if self.pair_scorer is None:
      if benchmark_v2:
        return sorted(
            pairs,
            key=lambda item: (
                -float(item.score),
                item.parent_socket.name,
                item.child_socket.name,
            ),
        )
      adjusted_pairs: list[InterfacePairCandidate] = []
      for pair in pairs:
        if _invalid_support_fastener_pair(
            parent_id=parent_id,
            child_id=child_id,
            parent_metadata=parent_metadata,
            child_metadata=child_metadata,
            pair=pair,
        ):
          continue
        context_adjustment, context_note = _mechanical_context_pair_adjustment(
            parent_id=parent_id,
            child_id=child_id,
            parent_metadata=parent_metadata,
            child_metadata=child_metadata,
            relation_hint=relation_hint,
            pair=pair,
        )
        support_adjustment, support_note = _support_fastener_penalty(
            parent_id=parent_id,
            child_id=child_id,
            parent_metadata=parent_metadata,
            child_metadata=child_metadata,
            pair=pair,
        )
        context_adjustment += support_adjustment
        context_note += support_note
        adjusted_pairs.append(
            InterfacePairCandidate(
                score=float(pair.score) + float(context_adjustment),
                parent_socket=pair.parent_socket,
                child_socket=pair.child_socket,
                relation_mode=pair.relation_mode,
                note=f"{pair.note}{context_note}",
            )
        )
      adjusted_pairs.sort(
          key=lambda item: (-item.score, item.parent_socket.name, item.child_socket.name)
      )
      return adjusted_pairs
    original_pairs = list(pairs)
    scored: list[InterfacePairCandidate] = []
    for pair in pairs:
      if not benchmark_v2 and _invalid_support_fastener_pair(
          parent_id=parent_id,
          child_id=child_id,
          parent_metadata=parent_metadata,
          child_metadata=child_metadata,
          pair=pair,
      ):
        continue
      try:
        pair_score = self.pair_scorer.score_sockets(
            pair.parent_socket,
            pair.child_socket,
            relation_hint=relation_hint,
            protocol=self.input_protocol,
        )
        contact_type, contact_type_probs = self.pair_scorer.predict_contact_type_sockets(
            pair.parent_socket,
            pair.child_socket,
            relation_hint=relation_hint,
            protocol=self.input_protocol,
        )
      except Exception:
        if benchmark_v2:
          raise
        pair_score = 0.0
        contact_type = "generic_contact"
        contact_type_probs = {}
      if not benchmark_v2:
        contact_type = _contact_type_with_mechanical_prior(
            contact_type=contact_type,
            parent_id=parent_id,
            child_id=child_id,
            parent_metadata=parent_metadata,
            child_metadata=child_metadata,
            pair=pair,
        )
      type_prob = float(contact_type_probs.get(contact_type, 0.0))
      adjusted_score = (
          float(pair.score)
          + float(self.config.pair_scorer_weight) * (float(pair_score) - 0.5)
      )
      if benchmark_v2:
        context_adjustment, context_note = 0.0, ""
        support_adjustment, support_note = 0.0, ""
      else:
        context_adjustment, context_note = _mechanical_context_pair_adjustment(
            parent_id=parent_id,
            child_id=child_id,
            parent_metadata=parent_metadata,
            child_metadata=child_metadata,
            relation_hint=relation_hint,
            pair=pair,
        )
        support_adjustment, support_note = _support_fastener_penalty(
            parent_id=parent_id,
            child_id=child_id,
            parent_metadata=parent_metadata,
            child_metadata=child_metadata,
            pair=pair,
        )
      context_adjustment += support_adjustment
      context_note += support_note
      adjusted_score += context_adjustment
      if contact_type in {"threaded_interference", "screw_in_hole"}:
        adjusted_score += 0.75
      elif contact_type == "none":
        adjusted_score -= 2.5
      scored.append(
          InterfacePairCandidate(
              score=float(adjusted_score),
              parent_socket=pair.parent_socket,
              child_socket=pair.child_socket,
              relation_mode=pair.relation_mode,
              note=(
                  f"{pair.note};learned_pair_score={pair_score:.3f};"
                  f"contact_type={contact_type};contact_type_prob={type_prob:.3f}"
                  f"{context_note}"
              ),
          )
      )
    scored.sort(key=lambda item: (-item.score, item.parent_socket.name, item.child_socket.name))
    top_k = int(self.config.pair_scorer_top_k)
    if top_k > 0:
      retained = scored[:top_k]
      retained_keys = {
          (item.parent_socket.name, item.child_socket.name) for item in retained
      }
      # Keep a small heuristic tail so a young pair scorer cannot fully hide a
      # physically plausible pair. This is still leakage-free and functions as
      # recall protection during early training.
      tail_budget = max(2, top_k // 2)
      for pair in ([] if benchmark_v2 else original_pairs):
        if not benchmark_v2 and _invalid_support_fastener_pair(
            parent_id=parent_id,
            child_id=child_id,
            parent_metadata=parent_metadata,
            child_metadata=child_metadata,
            pair=pair,
        ):
          continue
        key = (pair.parent_socket.name, pair.child_socket.name)
        if key in retained_keys:
          continue
        retained.append(pair)
        retained_keys.add(key)
        if len(retained) >= top_k + tail_budget:
          break
      scored = retained
    return scored

  def _candidate_constraint(
      self,
      *,
      original: Constraint,
      pair: InterfacePairCandidate,
      relation_hint: str,
  ) -> Constraint:
    return Constraint(
        ctype=ConstraintType.COINCIDENT,
        part_a=original.part_a,
        socket_a=pair.parent_socket.name,
        part_b=original.part_b,
        socket_b=pair.child_socket.name,
        value=None,
        label=f"{original.part_b}_physics_interface_to_{original.part_a}",
        metadata={
            **dict(original.metadata or {}),
            "source_frame_hint": "primary",
            "target_frame_hint": "primary",
            "stack_phase": "seat",
            "alignment_mode": "interface_frame",
            "relation_hint": relation_hint,
            "interface_grounding": True,
            "interface_relation_mode": pair.relation_mode,
            "learned_pair_score": _pair_score(pair),
            "predicted_contact_type": _pair_contact_type(pair),
            "predicted_contact_type_prob": _pair_contact_type_prob(pair),
            "collision_policy": _collision_policy_for_contact_type(
                _pair_contact_type(pair)
            ),
            "source_interface_role": _role(pair.parent_socket),
            "target_interface_role": _role(pair.child_socket),
            "source_interface_score": _interface_score(pair.parent_socket),
            "target_interface_score": _interface_score(pair.child_socket),
        },
    )

  def _best_pose_for_pair(
      self,
      *,
      parent: PartInstance,
      child: PartInstance,
      constraint: Constraint,
      placed_instances: Optional[dict[str, PartInstance]] = None,
      force_socket_pair: bool = False,
      deadline: Optional[float] = None,
  ) -> Optional[_PoseCandidate]:
    if _deadline_expired(deadline):
      return None
    parent_instance = parent.copy()
    child_instance = child.copy()
    parent_instance.transform = Transform.identity()
    child_instance.transform = None
    graph = ConstraintGraph(
        instances={
            parent_instance.instance_id: parent_instance,
            child_instance.instance_id: child_instance,
        },
        constraints=[constraint],
        anchor=parent_instance.instance_id,
    )
    try:
      assembly = self._solver.solve(graph, allow_morphing=False)
    except SolveError:
      return None
    solved_child = assembly.instances.get(child.instance_id)
    solved_parent = assembly.instances.get(parent.instance_id)
    if solved_child is None or solved_child.transform is None or solved_parent is None:
      return None
    parent_world_transform = Transform.identity()
    if placed_instances:
      placed_parent = placed_instances.get(parent.instance_id)
      if placed_parent is not None and placed_parent.transform is not None:
        parent_world_transform = placed_parent.transform

    axis = self._candidate_slide_axis(
        parent=parent_instance,
        socket_name=constraint.socket_a,
    )
    base_transform = solved_child.transform
    deltas = self._candidate_axial_deltas(
        parent=parent_instance,
        child=child_instance,
        child_transform=base_transform,
        axis=axis,
    )
    best: Optional[_PoseCandidate] = None
    predicted_contact_type = str(
        (constraint.metadata or {}).get("predicted_contact_type") or ""
    )
    for delta in deltas[: max(1, int(self.config.max_pose_candidates_per_pair))]:
      if _deadline_expired(deadline):
        break
      transform = Transform(
          rotation=base_transform.rotation.copy(),
          translation=base_transform.translation + axis * float(delta),
      )
      world_transform = _compose_transform(parent_world_transform, transform)
      pose = self._score_pose(
          parent=parent_instance,
          parent_transform=parent_world_transform,
          child=child_instance,
          child_transform=world_transform,
          delta=float(delta),
          placed_instances=placed_instances,
      )
      if pose is None:
        continue
      if best is None or pose.score > best.score:
        best = pose
    if (
        best is not None
        and bool(self.config.micro_retreat_enabled)
        and predicted_contact_type not in {"threaded_interference", "screw_in_hole"}
    ):
      retreated = self._micro_retreat_pose(
          parent=parent_instance,
          parent_transform=parent_world_transform,
          child=child_instance,
          initial_pose=best,
          axis=axis,
          placed_instances=placed_instances,
          deadline=deadline,
      )
      if retreated is not None and (
          retreated.collision_volume < best.collision_volume
          or retreated.score >= best.score - 2.0
      ):
        best = retreated
    if (
        best is not None
        and bool(self.config.exact_contact_rerank_enabled)
        and predicted_contact_type
        in {
            "seat_plane",
            "planar_seat",
            "insert_axis",
            "shaft_in_bore",
            "screw_in_hole",
            "threaded_interference",
        }
    ):
      best = self._exact_contact_rerank_pose(
          parent=parent_instance,
          parent_transform=parent_world_transform,
          child=child_instance,
          pose=best,
          contact_type=predicted_contact_type,
      )
    return best

  def _exact_contact_rerank_pose(
      self,
      *,
      parent: PartInstance,
      parent_transform: Transform,
      child: PartInstance,
      pose: _PoseCandidate,
      contact_type: str,
  ) -> _PoseCandidate:
    distance = self._exact_distance_for_pose(
        parent=parent,
        parent_transform=parent_transform,
        child=child,
        child_transform=pose.transform,
    )
    if distance is None:
      return pose
    tolerance = max(1e-6, float(self.config.exact_contact_rerank_tolerance))
    contact_bonus = 14.0 if float(distance) <= tolerance else 0.0
    distance_penalty = min(80.0, 8.0 * float(distance) / tolerance)
    collision_volume = float(pose.collision_volume)
    collision_penalty = 0.0
    if (
        _strict_zero_penetration_contact_type(contact_type)
        and float(pose.aabb_overlap_volume) > float(self.config.collision_epsilon)
    ):
      exact_collision = self._exact_collision_volume_for_pose(
          parent=parent,
          parent_transform=parent_transform,
          child=child,
          child_transform=pose.transform,
      )
      if exact_collision is not None:
        collision_volume = float(exact_collision)
        collision_penalty = min(
            180.0,
            max(0.0, collision_volume - float(self.config.collision_epsilon))
            * max(0.12, float(self.config.exact_collision_volume_weight)),
        )
        if collision_volume > float(self.config.max_pair_collision_volume):
          collision_penalty += 80.0
    return _PoseCandidate(
        score=float(pose.score) + contact_bonus - distance_penalty - collision_penalty,
        transform=pose.transform,
        distance=float(distance),
        aabb_overlap_volume=float(pose.aabb_overlap_volume),
        collision_volume=float(collision_volume),
        delta=float(pose.delta),
    )

  def _micro_retreat_pose(
      self,
      *,
      parent: PartInstance,
      parent_transform: Transform,
      child: PartInstance,
      initial_pose: _PoseCandidate,
      axis: np.ndarray,
      placed_instances: Optional[dict[str, PartInstance]] = None,
      deadline: Optional[float] = None,
  ) -> Optional[_PoseCandidate]:
    """Search a tiny axial retreat that removes exact penetration.

    The closed-form frame alignment often lands at a mathematically exact
    contact, but real STEP exports include threads, chamfers, or interference
    fits. This routine keeps the chosen interface pair fixed and searches only
    one scalar along the mate axis.
    """

    if _deadline_expired(deadline):
      return initial_pose
    if initial_pose.aabb_overlap_volume <= max(1e-9, self.config.collision_epsilon):
      return initial_pose
    initial_volume = self._exact_collision_volume_for_pose(
        parent=parent,
        parent_transform=parent_transform,
        child=child,
        child_transform=initial_pose.transform,
    )
    if initial_volume is None:
      return initial_pose
    initial = self._pose_with_collision_volume(initial_pose, initial_volume)
    if initial_volume <= float(self.config.micro_retreat_collision_tolerance):
      return initial

    axis_world = normalize(parent_transform.apply_direction(axis))
    max_distance = max(0.0, float(self.config.micro_retreat_max_distance))
    if max_distance <= 1e-9:
      return initial
    probes = _micro_retreat_probe_distances(max_distance)
    best = initial
    for sign in (1.0, -1.0):
      low_distance = 0.0
      high_distance: Optional[float] = None
      high_pose: Optional[_PoseCandidate] = None
      for distance in probes:
        if _deadline_expired(deadline):
          return best
        candidate = self._pose_at_retreat(
            parent=parent,
            parent_transform=parent_transform,
            child=child,
            initial_pose=initial,
            axis_world=axis_world,
            retreat=float(sign) * float(distance),
            placed_instances=placed_instances,
        )
        if candidate is None:
          continue
        if candidate.collision_volume < best.collision_volume or (
            candidate.collision_volume <= best.collision_volume
            and candidate.score > best.score
        ):
          best = candidate
        if candidate.collision_volume <= float(self.config.micro_retreat_collision_tolerance):
          high_distance = float(distance)
          high_pose = candidate
          break
        low_distance = float(distance)
      if high_distance is None or high_pose is None:
        continue
      # Refine the first collision-free point. The retained pose is the closest
      # zero-penetration pose to the original alignment, not the largest gap.
      for _ in range(max(0, int(self.config.micro_retreat_binary_steps))):
        if _deadline_expired(deadline):
          return best
        mid = 0.5 * (low_distance + high_distance)
        candidate = self._pose_at_retreat(
            parent=parent,
            parent_transform=parent_transform,
            child=child,
            initial_pose=initial,
            axis_world=axis_world,
            retreat=float(sign) * float(mid),
            placed_instances=placed_instances,
        )
        if candidate is None:
          break
        if candidate.collision_volume <= float(self.config.micro_retreat_collision_tolerance):
          high_distance = mid
          high_pose = candidate
          if candidate.score > best.score or best.collision_volume > float(
              self.config.micro_retreat_collision_tolerance
          ):
            best = candidate
        else:
          low_distance = mid
          if candidate.collision_volume < best.collision_volume:
            best = candidate
      if high_pose is not None and high_pose.score > best.score - 4.0:
        best = high_pose
    return best if best is not initial else initial

  def _pose_at_retreat(
      self,
      *,
      parent: PartInstance,
      parent_transform: Transform,
      child: PartInstance,
      initial_pose: _PoseCandidate,
      axis_world: np.ndarray,
      retreat: float,
      placed_instances: Optional[dict[str, PartInstance]],
  ) -> Optional[_PoseCandidate]:
    transform = Transform(
        rotation=initial_pose.transform.rotation.copy(),
        translation=(
            initial_pose.transform.translation
            + normalize(axis_world) * float(retreat)
        ),
    )
    pose = self._score_pose(
        parent=parent,
        parent_transform=parent_transform,
        child=child,
        child_transform=transform,
        delta=float(initial_pose.delta) + float(retreat),
        placed_instances=placed_instances,
    )
    if pose is None:
      return None
    if pose.distance > float(self.config.contact_distance_tolerance):
      return None
    volume = self._exact_collision_volume_for_pose(
        parent=parent,
        parent_transform=parent_transform,
        child=child,
        child_transform=transform,
    )
    if volume is None:
      return pose
    return self._pose_with_collision_volume(pose, volume)

  def _pose_with_collision_volume(
      self,
      pose: _PoseCandidate,
      collision_volume: float,
  ) -> _PoseCandidate:
    collision_penalty = min(
        120.0,
        max(
            float(self.config.exact_collision_volume_weight),
            0.08,
        )
        * max(0.0, float(collision_volume)),
    )
    zero_bonus = (
        8.0
        if float(collision_volume)
        <= float(self.config.micro_retreat_collision_tolerance)
        else 0.0
    )
    return _PoseCandidate(
        score=float(pose.score) + zero_bonus - collision_penalty,
        transform=pose.transform,
        distance=float(pose.distance),
        aabb_overlap_volume=float(pose.aabb_overlap_volume),
        collision_volume=float(collision_volume),
        delta=float(pose.delta),
    )

  def _exact_collision_volume_for_pose(
      self,
      *,
      parent: PartInstance,
      parent_transform: Transform,
      child: PartInstance,
      child_transform: Transform,
  ) -> Optional[float]:
    step_path_parent = parent.metadata.get("step_path")
    step_path_child = child.metadata.get("step_path")
    if not isinstance(step_path_parent, str) or not isinstance(step_path_child, str):
      return None
    timeout = max(0.25, float(self.config.micro_retreat_exact_timeout_seconds))
    ctx = mp.get_context("spawn")
    result_queue = None
    process = None
    try:
      result_queue = ctx.Queue(maxsize=1)
      process = ctx.Process(
          target=_safe_exact_boolean_worker,
          args=(
              step_path_parent,
              parent_transform.rotation.tolist(),
              parent_transform.translation.tolist(),
              _canonicalization_offset(parent),
              step_path_child,
              child_transform.rotation.tolist(),
              child_transform.translation.tolist(),
              _canonicalization_offset(child),
              result_queue,
          ),
          daemon=True,
      )
      process.start()
      process.join(timeout)
      if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
        return None
      try:
        result = result_queue.get_nowait()
      except py_queue.Empty:
        return None
      if not isinstance(result, dict) or result.get("status") != "ok":
        return None
      raw_volume = result.get("volume")
      return float(raw_volume) if isinstance(raw_volume, (int, float)) else None
    except Exception:
      return None
    finally:
      if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=0.25)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.25)
      if result_queue is not None:
        try:
          result_queue.close()
        except Exception:
          pass

  def _exact_distance_for_pose(
      self,
      *,
      parent: PartInstance,
      parent_transform: Transform,
      child: PartInstance,
      child_transform: Transform,
  ) -> Optional[float]:
    direct = self._direct_exact_distance_for_pose(
        parent=parent,
        parent_transform=parent_transform,
        child=child,
        child_transform=child_transform,
    )
    if direct is not None:
      return direct
    step_path_parent = parent.metadata.get("step_path")
    step_path_child = child.metadata.get("step_path")
    if not isinstance(step_path_parent, str) or not isinstance(step_path_child, str):
      return None
    timeout = max(0.25, float(self.config.exact_contact_rerank_timeout_seconds))
    ctx = mp.get_context("spawn")
    result_queue = None
    process = None
    try:
      result_queue = ctx.Queue(maxsize=1)
      process = ctx.Process(
          target=_safe_exact_distance_worker,
          args=(
              step_path_parent,
              parent_transform.rotation.tolist(),
              parent_transform.translation.tolist(),
              _canonicalization_offset(parent),
              step_path_child,
              child_transform.rotation.tolist(),
              child_transform.translation.tolist(),
              _canonicalization_offset(child),
              result_queue,
          ),
          daemon=True,
      )
      process.start()
      process.join(timeout)
      if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.5)
        return None
      try:
        result = result_queue.get_nowait()
      except py_queue.Empty:
        return None
      if not isinstance(result, dict) or result.get("status") != "ok":
        return None
      distance = result.get("distance")
      if not isinstance(distance, (int, float)):
        return None
      return max(0.0, float(distance))
    except Exception:
      return None
    finally:
      if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=0.25)
        if process.is_alive() and hasattr(process, "kill"):
          process.kill()
          process.join(timeout=0.25)
      if result_queue is not None:
        try:
          result_queue.close()
        except Exception:
          pass

  def _direct_exact_distance_for_pose(
      self,
      *,
      parent: PartInstance,
      parent_transform: Transform,
      child: PartInstance,
      child_transform: Transform,
  ) -> Optional[float]:
    shape_parent = self.shape_library.get(parent.template_name)
    if shape_parent is None:
      shape_parent = self.shape_library.get(parent.instance_id)
    shape_child = self.shape_library.get(child.template_name)
    if shape_child is None:
      shape_child = self.shape_library.get(child.instance_id)
    if shape_parent is None or shape_child is None:
      return None
    try:
      world_parent = transform_shape(shape_parent, parent_transform)
      world_child = transform_shape(shape_child, child_transform)
      return max(0.0, float(world_parent.distance(world_child)))
    except Exception:
      return None

  def _candidate_slide_axis(
      self,
      *,
      parent: PartInstance,
      socket_name: str,
  ) -> np.ndarray:
    socket = parent.sockets.get(socket_name)
    if socket is None:
      return np.array([0.0, 0.0, 1.0], dtype=float)
    for value in (socket.z_axis, socket.axis, socket.normal):
      if value is not None:
        return normalize(np.asarray(value, dtype=float))
    return np.array([0.0, 0.0, 1.0], dtype=float)

  def _candidate_axial_deltas(
      self,
      *,
      parent: PartInstance,
      child: PartInstance,
      child_transform: Transform,
      axis: np.ndarray,
  ) -> list[float]:
    values: list[float] = [0.0]
    pmin, pmax = _bbox_interval(parent, Transform.identity(), axis)
    cmin, cmax = _bbox_interval(child, child_transform, axis)
    values.extend([pmin - cmax, pmax - cmin, pmin - cmin, pmax - cmax])

    parent_planes = _axial_stop_positions(parent, Transform.identity(), axis)
    child_planes = _axial_stop_positions(child, child_transform, axis)
    for pp in parent_planes[:12]:
      for cp in child_planes[:12]:
        values.append(pp - cp)

    expanded: list[float] = []
    for value in values:
      for extra in (0.0, -0.05, 0.05, -0.25, 0.25):
        expanded.append(float(value) + extra)
    unique = sorted(
        {round(float(item), 6) for item in expanded},
        key=lambda item: (abs(item), item),
    )
    return [float(item) for item in unique]

  def _score_pose(
      self,
      *,
      parent: PartInstance,
      parent_transform: Transform,
      child: PartInstance,
      child_transform: Transform,
      delta: float,
      placed_instances: Optional[dict[str, PartInstance]] = None,
  ) -> Optional[_PoseCandidate]:
    shape_parent = self.shape_library.get(parent.template_name)
    if shape_parent is None:
      shape_parent = self.shape_library.get(parent.instance_id)
    shape_child = self.shape_library.get(child.template_name)
    if shape_child is None:
      shape_child = self.shape_library.get(child.instance_id)
    if shape_parent is None or shape_child is None:
      return None
    try:
      world_parent = transform_shape(shape_parent, parent_transform)
      world_child = transform_shape(shape_child, child_transform)
    except Exception:
      return None
    aabb_overlap = _aabb_overlap_volume(world_parent, world_child)
    if bool(self.config.exact_distance_for_pose_search):
      try:
        distance = float(world_parent.distance(world_child))
      except Exception:
        return None
    else:
      distance = _aabb_distance(world_parent, world_child)
    global_overlap = 0.0
    child_bbox_volume = _shape_bbox_volume(world_child)
    if placed_instances:
      for placed_id, placed in placed_instances.items():
        if placed_id in {parent.instance_id, child.instance_id}:
          continue
        if placed.transform is None:
          continue
        placed_shape = self.shape_library.get(placed.template_name)
        if placed_shape is None:
          placed_shape = self.shape_library.get(placed.instance_id)
        if placed_shape is None:
          continue
        try:
          world_placed = transform_shape(placed_shape, placed.transform)
          global_overlap += _aabb_overlap_volume(world_child, world_placed)
        except Exception:
          continue
    if (
        child_bbox_volume > 1e-9
        and global_overlap / child_bbox_volume
        > float(self.config.max_global_aabb_overlap_ratio)
    ):
      return None
    collision_volume = 0.0
    if (
        self.config.exact_collision_for_pose_search
        and aabb_overlap > self.config.collision_epsilon
        and distance <= max(1e-6, self.config.contact_distance_tolerance)
    ):
      try:
        collision_volume = float(shape_volume(boolean_intersection(world_parent, world_child)))
      except Exception:
        collision_volume = 0.0
    contact_reward = 8.0 if distance <= self.config.contact_distance_tolerance else 0.0
    distance_penalty = min(
        40.0,
        4.0 * distance / max(1e-6, self.config.contact_distance_tolerance),
    )
    collision_penalty = min(
        80.0,
        self.config.exact_collision_volume_weight * max(0.0, collision_volume),
    )
    aabb_penalty = min(35.0, self.config.aabb_overlap_weight * max(0.0, aabb_overlap))
    global_penalty = min(70.0, 0.012 * max(0.0, global_overlap))
    score = (
        contact_reward
        - distance_penalty
        - collision_penalty
        - aabb_penalty
        - global_penalty
    )
    return _PoseCandidate(
        score=float(score),
        transform=child_transform,
        distance=float(distance),
        aabb_overlap_volume=float(aabb_overlap),
        collision_volume=float(collision_volume),
        delta=float(delta),
    )


def _constraint_relation_hint(constraint: Constraint) -> str:
  if constraint.ctype == ConstraintType.CONCENTRIC:
    return "coaxial"
  if constraint.ctype == ConstraintType.COINCIDENT:
    return "planar"
  return "link"


def _compose_transform(parent: Transform, child_relative: Transform) -> Transform:
  return Transform(
      rotation=parent.rotation @ child_relative.rotation,
      translation=parent.rotation @ child_relative.translation + parent.translation,
  )


def _role(socket: Socket) -> str:
  return str(socket.metadata.get("interface_role") or "generic_interface").strip().lower()


def _interface_score(socket: Socket) -> float:
  raw = socket.metadata.get("interface_score", socket.metadata.get("quality", 0.0))
  return float(raw) if isinstance(raw, (int, float)) else 0.0


def _pair_score(pair: InterfacePairCandidate) -> float:
  marker = "learned_pair_score="
  if marker not in pair.note:
    return 0.0
  try:
    return float(pair.note.split(marker, 1)[1].split(";", 1)[0])
  except Exception:
    return 0.0


def _pair_contact_type(pair: InterfacePairCandidate) -> str:
  marker = "contact_type="
  if marker not in pair.note:
    return "generic_contact"
  try:
    raw = pair.note.split(marker, 1)[1].split(";", 1)[0].strip()
    return raw or "generic_contact"
  except Exception:
    return "generic_contact"


def _pair_contact_type_prob(pair: InterfacePairCandidate) -> float:
  marker = "contact_type_prob="
  if marker not in pair.note:
    return 0.0
  try:
    return float(pair.note.split(marker, 1)[1].split(";", 1)[0])
  except Exception:
    return 0.0


def _collision_policy_for_contact_type(contact_type: str) -> str:
  if str(contact_type) in {"threaded_interference", "screw_in_hole"}:
    return "intentional_interference"
  if str(contact_type) in {
      "seat_plane",
      "planar_seat",
      "insert_axis",
      "shaft_in_bore",
      "pin_in_slot",
      "boss_in_slot",
  }:
    return "strict_zero_penetration"
  return "standard_validation"


def _strict_zero_penetration_contact_type(contact_type: str) -> bool:
  return str(contact_type) in {
      "seat_plane",
      "planar_seat",
      "insert_axis",
      "shaft_in_bore",
      "pin_in_slot",
      "boss_in_slot",
  }


def _contact_type_with_mechanical_prior(
    *,
    contact_type: str,
    parent_id: str,
    child_id: str,
    parent_metadata: Optional[dict[str, Any]] = None,
    child_metadata: Optional[dict[str, Any]] = None,
    pair: InterfacePairCandidate,
) -> str:
  text = (
      _part_context_text(parent_id, parent_metadata)
      + " "
      + _part_context_text(child_id, child_metadata)
  ).lower()
  fastener_like = any(
      token in text
      for token in (
          "screw",
          "bolt",
          "fastener",
          "thread",
          "nut",
          "stud",
      )
  )
  role_a = _role(pair.parent_socket)
  role_b = _role(pair.child_socket)
  roles = {role_a, role_b}
  pin_roles = {"shaft_axis", "cylindrical_interface", "pin_boss"}
  if "obround_slot" in roles and "pin_boss" in roles:
    return "boss_in_slot"
  if "obround_slot" in roles and bool(roles & {"shaft_axis", "cylindrical_interface"}):
    return "pin_in_slot"
  if "threaded_hole" in roles and bool(roles & pin_roles):
    return "screw_in_hole"
  if "center_bore" in roles and bool(roles & pin_roles):
    if fastener_like:
      return "screw_in_hole"
    return "shaft_in_bore"
  if not fastener_like:
    return contact_type
  cylindrical_roles = {
      "hole_entry",
      "center_bore",
      "threaded_hole",
      "shaft_axis",
      "cylindrical_interface",
      "pin_boss",
  }
  if role_a in cylindrical_roles and role_b in cylindrical_roles:
    return "threaded_interference"
  return contact_type


def _mechanical_context_pair_adjustment(
    *,
    parent_id: str,
    child_id: str,
    parent_metadata: Optional[dict[str, Any]] = None,
    child_metadata: Optional[dict[str, Any]] = None,
    relation_hint: str,
    pair: InterfacePairCandidate,
) -> tuple[float, str]:
  """Contextual prior that prevents gear outer arcs from masquerading as bores."""

  hint = str(relation_hint or "").lower()
  if hint not in {"coaxial", "fasten", "insert", "link"}:
    return 0.0, ""
  parent_text = _part_context_text(parent_id, parent_metadata)
  child_text = _part_context_text(child_id, child_metadata)
  parent_gear = _is_gear_like(parent_text)
  child_gear = _is_gear_like(child_text)
  if parent_gear == child_gear:
    return 0.0, ""
  parent_shaft = _is_shaft_or_fastener_like(parent_text)
  child_shaft = _is_shaft_or_fastener_like(child_text)
  if parent_gear and not child_shaft:
    return 0.0, ""
  if child_gear and not parent_shaft:
    return 0.0, ""

  gear_socket = pair.parent_socket if parent_gear else pair.child_socket
  other_socket = pair.child_socket if parent_gear else pair.parent_socket
  gear_role = _role(gear_socket)
  other_role = _role(other_socket)
  if gear_role not in {"hole_entry", "center_bore", "shaft_axis", "cylindrical_interface", "pin_boss"}:
    return -8.0, f";gear_center_prior=-8.000;gear_socket_role={gear_role}"
  if other_role not in {"shaft_axis", "cylindrical_interface", "pin_boss", "hole_entry", "center_bore"}:
    return -5.0, f";gear_center_prior=-5.000;gear_mate_role={other_role}"

  centrality = _socket_centeredness_score(gear_socket)
  boundary = _socket_boundary_score(gear_socket)
  radius_ratio = _socket_radius_ratio(gear_socket)
  adjustment = 0.0
  # Center bores should be close to the part-local bbox center. Outer rim arcs
  # and decorative/gear tooth cylindrical slots tend to live near bbox borders.
  adjustment += 8.0 * (centrality - 0.5)
  adjustment -= 5.5 * boundary
  if centrality < 0.65:
    adjustment -= 4.0
  if centrality < 0.45:
    adjustment -= 5.0
  if radius_ratio > 0.16:
    adjustment -= min(12.0, 24.0 * (radius_ratio - 0.16))
  if gear_role == "center_bore" and radius_ratio > 0.22:
    adjustment -= 4.0
  if boundary > 0.5 and centrality < 0.65:
    adjustment -= 4.0
  if gear_role in {"hole_entry", "center_bore"}:
    adjustment += 1.0
  if gear_role == "center_bore":
    adjustment += 1.75
  if gear_role in {"shaft_axis", "pin_boss"} and child_gear:
    adjustment -= 1.0
  note = (
      f";gear_center_prior={adjustment:.3f}"
      f";gear_socket_centeredness={centrality:.3f}"
      f";gear_socket_boundary={boundary:.3f}"
      f";gear_socket_radius_ratio={radius_ratio:.3f}"
  )
  return float(adjustment), note


def _support_fastener_penalty(
    *,
    parent_id: str,
    child_id: str,
    parent_metadata: Optional[dict[str, Any]],
    child_metadata: Optional[dict[str, Any]],
    pair: InterfacePairCandidate,
) -> tuple[float, str]:
  parent_text = _part_context_text(parent_id, parent_metadata)
  child_text = _part_context_text(child_id, child_metadata)
  parent_support = any(
      token in parent_text for token in ("holder", "base", "frame", "bracket", "plate", "support")
  )
  child_support = any(
      token in child_text for token in ("holder", "base", "frame", "bracket", "plate", "support")
  )
  parent_fastener = _is_shaft_or_fastener_like(parent_text)
  child_fastener = _is_shaft_or_fastener_like(child_text)
  role_a = _role(pair.parent_socket)
  role_b = _role(pair.child_socket)
  penalty = 0.0
  if parent_support and child_fastener and role_a == "pin_boss":
    penalty -= 4.0
  if child_support and parent_fastener and role_b == "pin_boss":
    penalty -= 4.0
  if penalty == 0.0:
    return 0.0, ""
  return penalty, f";support_fastener_penalty={penalty:.3f}"


def _invalid_support_fastener_pair(
    *,
    parent_id: str,
    child_id: str,
    parent_metadata: Optional[dict[str, Any]],
    child_metadata: Optional[dict[str, Any]],
    pair: InterfacePairCandidate,
) -> bool:
  parent_text = _part_context_text(parent_id, parent_metadata)
  child_text = _part_context_text(child_id, child_metadata)
  parent_support = any(
      token in parent_text
      for token in ("holder", "base", "frame", "bracket", "plate", "support")
  )
  child_support = any(
      token in child_text
      for token in ("holder", "base", "frame", "bracket", "plate", "support")
  )
  parent_fastener = _is_shaft_or_fastener_like(parent_text)
  child_fastener = _is_shaft_or_fastener_like(child_text)
  support_hole_roles = {"hole_entry", "center_bore", "threaded_hole"}
  if parent_support and child_fastener:
    return _role(pair.parent_socket) not in support_hole_roles
  if child_support and parent_fastener:
    return _role(pair.child_socket) not in support_hole_roles
  return False


def _is_gear_like(part_id: str) -> bool:
  text = str(part_id or "").lower()
  return any(
      token in text
      for token in (
          "gear",
          "wheel",
          "pulley",
          "sprocket",
          "pinion",
          "cam",
      )
  )


def _part_context_text(
    part_id: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
  pieces = [str(part_id or "")]
  if isinstance(metadata, dict):
    for key in (
        "benchmark_role_hint",
        "retrieval_role_hint",
        "source_step_name",
        "assembly_dir",
    ):
      raw = metadata.get(key)
      if raw is not None:
        pieces.append(str(raw))
  return " ".join(pieces).lower()


def _is_shaft_or_fastener_like(part_id: str) -> bool:
  text = str(part_id or "").lower()
  return any(
      token in text
      for token in (
          "shaft",
          "axle",
          "arbor",
          "spindle",
          "rod",
          "pin",
          "bolt",
          "screw",
          "fastener",
          "stud",
      )
  )


def _socket_centeredness_score(socket: Socket) -> float:
  features = socket.metadata.get("interface_features")
  if not isinstance(features, dict):
    return 0.5
  coords = []
  for key in ("center_x_norm", "center_y_norm", "center_z_norm"):
    raw = features.get(key)
    if not isinstance(raw, (int, float)):
      return 0.5
    coords.append(float(raw))
  axis = socket.axis if socket.axis is not None else socket.z_axis
  used = [0, 1, 2]
  if axis is not None:
    try:
      dominant = int(np.argmax(np.abs(np.asarray(axis, dtype=float).reshape(3))))
      used = [idx for idx in used if idx != dominant]
    except Exception:
      used = [0, 1, 2]
  # For a cylindrical through-bore, position along the cylinder axis can lie on
  # one side face. The perpendicular coordinates are what distinguish center
  # bore from outer rim slots.
  arr = np.asarray([coords[idx] for idx in used], dtype=float)
  distance = float(np.linalg.norm(arr - 0.5))
  divisor = 0.5 if len(used) == 2 else 0.72
  return max(0.0, min(1.0, 1.0 - distance / divisor))


def _socket_boundary_score(socket: Socket) -> float:
  features = socket.metadata.get("interface_features")
  if isinstance(features, dict):
    raw = features.get("near_bbox_boundary")
    if isinstance(raw, (int, float)):
      return max(0.0, min(1.0, float(raw)))
  return 0.0


def _socket_radius_ratio(socket: Socket) -> float:
  features = socket.metadata.get("interface_features")
  if isinstance(features, dict):
    raw = features.get("radius_ratio")
    if isinstance(raw, (int, float)):
      return max(0.0, float(raw))
  return 0.0


def _passes_hard_geometry_pruning(
    pair: InterfacePairCandidate,
    *,
    skip_obround_slot: bool = False,
) -> bool:
  role_a = _role(pair.parent_socket)
  role_b = _role(pair.child_socket)
  if "slot_arc" in {role_a, role_b}:
    return False
  if skip_obround_slot and "obround_slot" in {role_a, role_b}:
    return False
  planar = {"planar_seat", "shoulder_stop"}
  hole_roles = {"hole_entry", "center_bore", "threaded_hole"}
  pin_roles = {"shaft_axis", "cylindrical_interface", "pin_boss"}
  slot_roles = {"obround_slot"}
  cylindrical = hole_roles | pin_roles | slot_roles
  if (role_a in planar and role_b in cylindrical) or (
      role_b in planar and role_a in cylindrical
  ):
    return False
  if (role_a in slot_roles and role_b not in pin_roles) or (
      role_b in slot_roles and role_a not in pin_roles
  ):
    return False
  radius_a = pair.parent_socket.radius
  radius_b = pair.child_socket.radius
  if radius_a is None or radius_b is None:
    return True
  if role_a in hole_roles and role_b in pin_roles:
    return float(radius_a) + 0.25 >= float(radius_b)
  if role_b in hole_roles and role_a in pin_roles:
    return float(radius_b) + 0.25 >= float(radius_a)
  if role_a == "obround_slot" and role_b in pin_roles:
    return _slot_accepts_pin(pair.parent_socket, pair.child_socket)
  if role_b == "obround_slot" and role_a in pin_roles:
    return _slot_accepts_pin(pair.child_socket, pair.parent_socket)
  return True


def _slot_accepts_pin(slot_socket: Socket, pin_socket: Socket) -> bool:
  slot_width = slot_socket.metadata.get("slot_width")
  pin_radius = pin_socket.radius
  if not isinstance(slot_width, (int, float)) or pin_radius is None:
    return True
  return float(slot_width) + 0.35 >= 2.0 * float(pin_radius)


def _bbox_interval(
    part: PartInstance,
    transform: Transform,
    axis: np.ndarray,
) -> tuple[float, float]:
  mins = np.asarray(part.local_bbox_min, dtype=float)
  maxs = np.asarray(part.local_bbox_max, dtype=float)
  values = []
  for x in (mins[0], maxs[0]):
    for y in (mins[1], maxs[1]):
      for z in (mins[2], maxs[2]):
        point = transform.apply_point(np.array([x, y, z], dtype=float))
        values.append(float(np.dot(point, axis)))
  return min(values), max(values)


def _axial_stop_positions(
    part: PartInstance,
    transform: Transform,
    axis: np.ndarray,
) -> list[float]:
  values: list[tuple[float, float]] = []
  for socket in part.sockets.values():
    role = str(
        socket.metadata.get("interface_role")
        or socket.metadata.get("support_role")
        or socket.kind
    ).lower()
    if not any(token in role for token in ("plane", "seat", "stop", "shoulder")):
      continue
    direction = socket.normal
    if direction is None:
      direction = socket.axis
    if direction is None:
      continue
    world_dir = normalize(transform.apply_direction(np.asarray(direction, dtype=float)))
    parallel = abs(float(np.dot(world_dir, axis)))
    if parallel < 0.75:
      continue
    world_origin = transform.apply_point(socket.origin)
    values.append((parallel, float(np.dot(world_origin, axis))))
  values.sort(key=lambda item: (-item[0], abs(item[1])))
  return [value for _, value in values]


def _aabb_overlap_volume(shape_a: Any, shape_b: Any) -> float:
  try:
    min_a, max_a = shape_bbox(shape_a)
    min_b, max_b = shape_bbox(shape_b)
  except Exception:
    return 0.0
  overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
  if np.any(overlap <= 0.0):
    return 0.0
  return float(np.prod(overlap))


def _aabb_distance(shape_a: Any, shape_b: Any) -> float:
  try:
    min_a, max_a = shape_bbox(shape_a)
    min_b, max_b = shape_bbox(shape_b)
  except Exception:
    return 0.0
  gap_low = np.asarray(min_b, dtype=float) - np.asarray(max_a, dtype=float)
  gap_high = np.asarray(min_a, dtype=float) - np.asarray(max_b, dtype=float)
  gap = np.maximum(np.maximum(gap_low, gap_high), 0.0)
  return float(np.linalg.norm(gap))


def _shape_bbox_volume(shape: Any) -> float:
  try:
    bbox_min, bbox_max = shape_bbox(shape)
  except Exception:
    return 0.0
  extents = np.maximum(np.asarray(bbox_max, dtype=float) - np.asarray(bbox_min, dtype=float), 0.0)
  return float(np.prod(extents))


def _micro_retreat_probe_distances(max_distance: float) -> list[float]:
  base = [0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.65, 1.0, 1.5, 2.0]
  values = [value for value in base if value <= max_distance + 1e-9]
  if not values or values[-1] < max_distance:
    values.append(float(max_distance))
  return sorted({round(float(item), 6) for item in values if item > 0.0})


def _deadline_after(started_at: float, seconds: float) -> Optional[float]:
  if seconds <= 0.0:
    return None
  return float(started_at) + float(seconds)


def _deadline_expired(deadline: Optional[float]) -> bool:
  return deadline is not None and time.monotonic() >= float(deadline)


def _canonicalization_offset(instance: PartInstance) -> list[float]:
  raw = instance.metadata.get("canonicalization_offset")
  if isinstance(raw, (list, tuple)) and len(raw) == 3:
    try:
      return [float(raw[0]), float(raw[1]), float(raw[2])]
    except (TypeError, ValueError):
      pass
  return [0.0, 0.0, 0.0]
