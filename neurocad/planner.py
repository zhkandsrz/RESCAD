"""Neural-layer planners: rule baseline and real LLM-based planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Optional, Protocol
from urllib import error as url_error
from urllib import request as url_request

import numpy as np

from .benchmark_v2_model_view import validate_benchmark_v2_model_view
from .interface_grounder import rank_interface_socket_pairs
from .math3d import normalize
from .domain_types import (
    CanonicalMateEdge,
    CanonicalMateGraph,
    CanonicalStackGroup,
    CollisionRecord,
    Constraint,
    ConstraintGraph,
    ConstraintType,
    PartInstance,
    PartTemplate,
    PlannerFeedback,
    Socket,
)


class PlannerProtocol(Protocol):
  """Planner interface shared by rule-based and LLM planners."""

  def plan(
      self,
      instruction: str,
      catalog: dict[str, PartTemplate],
      feedback: Optional[PlannerFeedback] = None,
  ) -> ConstraintGraph:
    ...


@dataclass
class _ConnectionCandidate:
  score: float
  constraints: list[Constraint]
  note: str
  reserved_endpoints: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _TopologyConnectionProposal:
  part_a: str
  part_b: str
  relation_type: str = "link"
  label: str = ""
  stack_group: Optional[str] = None
  order_hint: Optional[int] = None
  source_frame_hint: Optional[str] = None
  target_frame_hint: Optional[str] = None
  source_stop_hint: Optional[str] = None
  target_stop_hint: Optional[str] = None
  stack_phase: Optional[str] = None
  alignment_mode: Optional[str] = None


class RuleBasedNeuralPlanner:
  """A small planning baseline that mimics an LLM constraint planner."""

  def __init__(
      self,
      default_clearance: float = 1.0,
      single_use_kinds: Optional[set[str]] = None,
      plane_capacity: int = 6,
  ):
    self.default_clearance = float(default_clearance)
    self.single_use_kinds = (
        {
            "hole",
            "threaded_hole",
            "guide_slot",
        }
        if single_use_kinds is None
        else set(single_use_kinds)
    )
    self.plane_capacity = max(1, int(plane_capacity))

  def plan(
      self,
      instruction: str,
      catalog: dict[str, PartTemplate],
      feedback: Optional[PlannerFeedback] = None,
  ) -> ConstraintGraph:
    ordered_parts = self._rank_parts_from_instruction(instruction, catalog)
    instances = {
        part_name: catalog[part_name].instantiate(part_name)
        for part_name in ordered_parts
    }
    anchor = self._choose_anchor(ordered_parts)
    ordered_parts = self._contact_guided_order(
        anchor=anchor,
        ordered_parts=ordered_parts,
        instances=instances,
    )
    planner_notes = [
        f"instruction={instruction}",
        f"ordered_parts={ordered_parts}",
        f"anchor={anchor}",
    ]
    blocked_pairs = self._blocked_pairs_from_feedback(feedback)
    if blocked_pairs:
      planner_notes.append(
          "blocked_pairs="
          + str(sorted(tuple(sorted(pair)) for pair in blocked_pairs))
      )

    constraints: list[Constraint] = []
    socket_usage: dict[tuple[str, str], int] = {}
    selected = [anchor]
    for part_name in ordered_parts:
      if part_name == anchor:
        continue
      candidate_parents = self._preferred_parent_candidates(
          child=part_name,
          selected=selected,
          instances=instances,
      )
      if not candidate_parents:
        candidate_parents = list(selected)
      parent, candidate = self._best_parent_for_part(
          part_name,
          candidate_parents,
          instances,
          socket_usage=socket_usage,
          blocked_pairs=blocked_pairs,
      )
      selected.append(part_name)
      if candidate is None:
        planner_notes.append(
            f"fallback connection used for {parent}->{part_name}"
        )
        continue
      constraints.extend(candidate.constraints)
      self._reserve_candidate(candidate, socket_usage)
      planner_notes.append(candidate.note)

    if feedback:
      planner_notes.extend([f"feedback={m}" for m in feedback.messages])
      feedback_constraints = self._constraints_from_feedback(
          feedback.collisions, instances, constraints
      )
      for constraint in feedback_constraints:
        self._append_unique(constraints, constraint)
      if feedback_constraints:
        planner_notes.append(
            f"added_feedback_constraints={len(feedback_constraints)}"
        )

    return ConstraintGraph(
        instances=instances,
        constraints=constraints,
        anchor=anchor,
        metadata={
            "planner_notes": planner_notes,
            "canonical_mate_graph": self._canonical_graph_from_constraints(
                anchor=anchor,
                constraints=constraints,
            ),
        },
    )

  def _canonical_graph_from_constraints(
      self,
      anchor: str,
      constraints: list[Constraint],
  ) -> dict[str, Any]:
    edges: list[CanonicalMateEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for constraint in constraints:
      relation_type = "link"
      if constraint.ctype == ConstraintType.CONCENTRIC:
        relation_type = "coaxial"
      elif constraint.ctype == ConstraintType.COINCIDENT:
        relation_type = "planar"
      key = tuple(sorted((constraint.part_a, constraint.part_b)) + [relation_type])
      if key in seen:
        continue
      seen.add(key)
      edges.append(
          CanonicalMateEdge(
              part_a=constraint.part_a,
              part_b=constraint.part_b,
              relation_type=relation_type,
              label=constraint.label,
              stack_group=(
                  str(constraint.metadata.get("stack_group")).strip()
                  if constraint.metadata.get("stack_group") is not None
                  else None
              ),
              order_hint=(
                  int(constraint.metadata.get("order_hint"))
                  if isinstance(constraint.metadata.get("order_hint"), int)
                  else None
              ),
              source_frame_hint=(
                  str(constraint.metadata.get("source_frame_hint")).strip()
                  if constraint.metadata.get("source_frame_hint") is not None
                  else None
              ),
              target_frame_hint=(
                  str(constraint.metadata.get("target_frame_hint")).strip()
                  if constraint.metadata.get("target_frame_hint") is not None
                  else None
              ),
              source_stop_hint=(
                  str(constraint.metadata.get("source_stop_hint")).strip()
                  if constraint.metadata.get("source_stop_hint") is not None
                  else None
              ),
              target_stop_hint=(
                  str(constraint.metadata.get("target_stop_hint")).strip()
                  if constraint.metadata.get("target_stop_hint") is not None
                  else None
              ),
              stack_phase=(
                  str(constraint.metadata.get("stack_phase")).strip().lower()
                  if constraint.metadata.get("stack_phase") is not None
                  else None
              ),
              alignment_mode=(
                  str(constraint.metadata.get("alignment_mode")).strip().lower()
                  if constraint.metadata.get("alignment_mode") is not None
                  else None
              ),
          )
      )
    return CanonicalMateGraph(
        anchor=anchor,
        edges=edges,
        stack_groups=self._infer_stack_groups_from_constraints(
            anchor=anchor,
            constraints=constraints,
        ),
        plan_summary="",
        metadata={"source": "rule_planner"},
    ).to_dict()

  def _infer_stack_groups_from_constraints(
      self,
      anchor: str,
      constraints: list[Constraint],
  ) -> list[CanonicalStackGroup]:
    carrier_parent: dict[str, str] = {}
    members: set[str] = set()
    for constraint in constraints:
      carrier_id = str(constraint.metadata.get("carrier_part_id") or "").strip()
      inserted_id = str(constraint.metadata.get("inserted_part_id") or "").strip()
      if not carrier_id or not inserted_id or carrier_id == inserted_id:
        continue
      carrier_parent.setdefault(inserted_id, carrier_id)
      members.add(carrier_id)
      members.add(inserted_id)

    groups: list[CanonicalStackGroup] = []
    if members:
      children: dict[str, list[str]] = {}
      for child_id, parent_id in carrier_parent.items():
        children.setdefault(parent_id, []).append(child_id)
      roots = sorted(
          part_id
          for part_id in members
          if part_id not in carrier_parent
      )
      seen_paths: set[tuple[str, ...]] = set()
      for root in roots:
        stack: list[tuple[str, list[str]]] = [(root, [root])]
        while stack:
          current, path = stack.pop()
          next_children = sorted(children.get(current, []))
          if not next_children:
            if len(path) >= 2 and tuple(path) not in seen_paths:
              seen_paths.add(tuple(path))
              groups.append(
                  CanonicalStackGroup(
                      group_id=f"stack_{len(groups):02d}",
                      parts=path,
                      anchor_part=path[0],
                      relation_type="coaxial",
                      metadata={"source": "rule_planner_chain"},
                  )
              )
            continue
          for child_id in reversed(next_children):
            stack.append((child_id, path + [child_id]))
    if groups:
      return groups

    adjacency: dict[str, set[str]] = {}
    for constraint in constraints:
      if constraint.ctype != ConstraintType.CONCENTRIC:
        continue
      adjacency.setdefault(constraint.part_a, set()).add(constraint.part_b)
      adjacency.setdefault(constraint.part_b, set()).add(constraint.part_a)
    visited: set[str] = set()
    for start in sorted(adjacency):
      if start in visited:
        continue
      stack = [start]
      component: list[str] = []
      while stack:
        current = stack.pop()
        if current in visited:
          continue
        visited.add(current)
        component.append(current)
        for nxt in sorted(adjacency.get(current, set())):
          if nxt not in visited:
            stack.append(nxt)
      if len(component) < 2:
        continue
      ordered: list[str] = []
      if anchor in component:
        ordered.append(anchor)
      ordered.extend(part_id for part_id in sorted(component) if part_id not in ordered)
      groups.append(
          CanonicalStackGroup(
              group_id=f"stack_{len(groups):02d}",
              parts=ordered,
              anchor_part=ordered[0],
              relation_type="coaxial",
              metadata={"source": "rule_planner_component"},
          )
      )
    return groups

  def _rank_parts_from_instruction(
      self, instruction: str, catalog: dict[str, PartTemplate]
  ) -> list[str]:
    text = instruction.lower()
    mentioned: list[tuple[int, str]] = []
    unmentioned: list[tuple[int, str]] = []
    for name in catalog:
      idx = text.find(name.lower())
      if idx >= 0:
        mentioned.append((idx, name))
      else:
        role_hint = str(
            catalog[name].metadata.get("benchmark_role_hint")
            or catalog[name].metadata.get("retrieval_role_hint")
            or ""
        ).lower()
        role_priority = {
            "base": 0,
            "bearing": 1,
            "shaft": 2,
            "gear": 3,
            "collar": 4,
            "bolt": 5,
        }.get(role_hint, 9)
        unmentioned.append((role_priority, name))

    mentioned = sorted(mentioned, key=lambda x: x[0])
    ordered = [name for _, name in mentioned]
    ordered.extend(
        [
            name
            for _, name in sorted(
                unmentioned,
                key=lambda item: (item[0], item[1]),
            )
        ]
    )
    if not ordered:
      raise ValueError("Catalog cannot be empty.")
    return ordered

  def _choose_anchor(self, ordered_parts: list[str]) -> str:
    for name in ordered_parts:
      if self._role_hint_from_name(name) == "base":
        return name
    for name in ordered_parts:
      lowered = name.lower()
      if any(
          token in lowered
          for token in ("base", "ground", "frame", "housing", "bracket", "mount")
      ):
        return name
    return ordered_parts[0]

  def _contact_guided_order(
      self,
      anchor: str,
      ordered_parts: list[str],
      instances: dict[str, PartInstance],
  ) -> list[str]:
    adjacency: dict[str, set[str]] = {}
    for part_id, instance in instances.items():
      neighbors = {
          neighbor
          for neighbor in self._instance_preferred_neighbors(instance)
          if neighbor in instances and neighbor != part_id
      }
      if not neighbors:
        continue
      adjacency.setdefault(part_id, set()).update(neighbors)
      for neighbor in neighbors:
        adjacency.setdefault(neighbor, set()).add(part_id)
    if not adjacency:
      return ordered_parts

    rank_index = {part_id: idx for idx, part_id in enumerate(ordered_parts)}
    traversal: list[str] = []
    visited: set[str] = set()
    queue: list[str] = [anchor]
    while queue:
      current = queue.pop(0)
      if current in visited:
        continue
      visited.add(current)
      traversal.append(current)
      next_parts = sorted(
          adjacency.get(current, set()) - visited,
          key=lambda part_id: (
              self._role_priority(part_id, instances),
              rank_index.get(part_id, 10**6),
              part_id,
          ),
      )
      queue.extend(next_parts)
    if len(traversal) <= 1:
      return ordered_parts
    traversal.extend(
        part_id for part_id in ordered_parts if part_id not in visited
    )
    return traversal

  def _preferred_parent_candidates(
      self,
      child: str,
      selected: list[str],
      instances: dict[str, PartInstance],
  ) -> list[str]:
    child_instance = instances.get(child)
    if child_instance is None:
      return []
    child_neighbors = set(self._instance_preferred_neighbors(child_instance))
    candidates: list[str] = []
    for parent_id in selected:
      parent_instance = instances.get(parent_id)
      if parent_instance is None:
        continue
      parent_neighbors = set(self._instance_preferred_neighbors(parent_instance))
      if child in parent_neighbors or parent_id in child_neighbors:
        candidates.append(parent_id)
    return candidates

  def _role_priority(
      self,
      part_id: str,
      instances: dict[str, PartInstance],
  ) -> int:
    instance = instances.get(part_id)
    if instance is None:
      return 99
    role = self._role_hint_from_instance(instance)
    return {
        "base": 0,
        "bearing": 1,
        "shaft": 2,
        "gear": 3,
        "bolt": 4,
        "collar": 5,
    }.get(role, 99)

  def _best_parent_for_part(
      self,
      child: str,
      candidate_parents: list[str],
      instances: dict[str, PartInstance],
      socket_usage: dict[tuple[str, str], int],
      blocked_pairs: set[frozenset[str]],
  ) -> tuple[str, Optional[_ConnectionCandidate]]:
    best_parent = candidate_parents[-1]
    best_candidate: Optional[_ConnectionCandidate] = None
    for parent in reversed(candidate_parents):
      blocked = frozenset((parent, child)) in blocked_pairs
      relation_hint = self._infer_relation_hint(
          instances[parent],
          instances[child],
      )
      candidate = self._best_connection(
          instances[parent],
          instances[child],
          socket_usage=socket_usage,
          blocked_pair=blocked,
          relation_hint=relation_hint,
      )
      if candidate is None:
        continue
      if blocked:
        candidate.score -= 0.8
        candidate.note += " [collision-avoid]"
      if best_candidate is None or candidate.score > best_candidate.score:
        best_parent = parent
        best_candidate = candidate
    return best_parent, best_candidate

  def _best_connection(
      self,
      parent: PartInstance,
      child: PartInstance,
      socket_usage: dict[tuple[str, str], int],
      blocked_pair: bool = False,
      relation_hint: Optional[str] = None,
  ) -> Optional[_ConnectionCandidate]:
    relation_hint = self._normalize_relation_hint(relation_hint)
    parent_holes = self._prioritize_hole_like(
        self._sorted_sockets(
            parent,
            {"hole", "threaded_hole"},
            relation_hint=relation_hint,
            partner_id=child.instance_id,
        )
    )
    parent_pins = self._sorted_sockets(
        parent,
        {"pin"},
        relation_hint=relation_hint,
        partner_id=child.instance_id,
    )
    parent_planes = self._sorted_sockets(
        parent,
        {"flange_plane", "plane"},
        relation_hint=relation_hint,
        partner_id=child.instance_id,
    )
    parent_axes = self._sorted_sockets(
        parent,
        {"axis", "guide_slot"},
        relation_hint=relation_hint,
        partner_id=child.instance_id,
    )

    child_holes = self._prioritize_hole_like(
        self._sorted_sockets(
            child,
            {"hole", "threaded_hole"},
            relation_hint=relation_hint,
            partner_id=parent.instance_id,
        )
    )
    child_pins = self._sorted_sockets(
        child,
        {"pin"},
        relation_hint=relation_hint,
        partner_id=parent.instance_id,
    )
    child_planes = self._sorted_sockets(
        child,
        {"flange_plane", "plane"},
        relation_hint=relation_hint,
        partner_id=parent.instance_id,
    )
    child_axes = self._sorted_sockets(
        child,
        {"axis", "guide_slot"},
        relation_hint=relation_hint,
        partner_id=parent.instance_id,
    )

    parent_id = parent.instance_id
    child_id = child.instance_id
    candidates: list[_ConnectionCandidate] = []
    allow_cyl = relation_hint != "none"
    allow_plane = relation_hint != "fasten"

    interface_pairs = rank_interface_socket_pairs(
        parent_id=parent_id,
        child_id=child_id,
        parent_sockets=self._sorted_sockets(
            parent,
            {"interface"},
            relation_hint=relation_hint,
            partner_id=child.instance_id,
        ),
        child_sockets=self._sorted_sockets(
            child,
            {"interface"},
            relation_hint=relation_hint,
            partner_id=parent.instance_id,
        ),
        relation_hint=relation_hint,
    )
    for pair in interface_pairs[:6]:
      if not self._can_use_socket(parent, pair.parent_socket, socket_usage):
        continue
      if not self._can_use_socket(child, pair.child_socket, socket_usage):
        continue
      score = (
          pair.score
          + self._preferred_neighbor_bonus(parent, child)
          + self._context_bonus(parent_id, child_id, mode="interface")
          + self._relation_hint_bonus(relation_hint, "interface")
          + self._blocked_pair_mode_adjustment(
              blocked_pair=blocked_pair,
              relation_hint=relation_hint,
              candidate_mode="interface",
          )
      )
      constraint = Constraint(
          ctype=ConstraintType.COINCIDENT,
          part_a=parent_id,
          socket_a=pair.parent_socket.name,
          part_b=child_id,
          socket_b=pair.child_socket.name,
          label=f"{child_id}_interface_to_{parent_id}_interface",
          metadata={
              "source_frame_hint": "primary",
              "target_frame_hint": "primary",
              "stack_phase": "seat",
              "alignment_mode": "interface_frame",
              "relation_hint": relation_hint,
              "interface_grounding": True,
              "interface_relation_mode": pair.relation_mode,
              "source_interface_role": pair.parent_socket.metadata.get("interface_role"),
              "target_interface_role": pair.child_socket.metadata.get("interface_role"),
              "source_interface_score": pair.parent_socket.metadata.get("interface_score"),
              "target_interface_score": pair.child_socket.metadata.get("interface_score"),
          },
      )
      candidates.append(
          _ConnectionCandidate(
              score=score,
              constraints=[constraint],
              note=pair.note + " (learned interface frame)",
              reserved_endpoints=[
                  (parent_id, pair.parent_socket.name),
                  (child_id, pair.child_socket.name),
              ],
          )
      )

    if allow_cyl:
      for parent_hole in parent_holes[:4]:
        if not self._can_use_socket(parent, parent_hole, socket_usage):
          continue
        for child_pin in child_pins[:4]:
          if not self._can_use_socket(child, child_pin, socket_usage):
            continue
          axial_value = self._concentric_constraint_value(
              parent=parent,
              parent_socket=parent_hole,
              child=child,
              child_socket=child_pin,
          )
          score = (
              5.2
              + 0.08
              * (
                  self._socket_priority(parent_hole)
                  + self._socket_priority(child_pin)
              )
              + self._context_bonus(parent_id, child_id, mode="pin_into_hole")
              + self._connection_bonus(
                  parent=parent,
                  child=child,
                  parent_socket=parent_hole,
                  child_socket=child_pin,
                  mode="pin_into_hole",
              )
              - self._socket_reuse_penalty(parent, parent_hole, socket_usage)
              - self._socket_reuse_penalty(child, child_pin, socket_usage)
              + self._relation_hint_bonus(relation_hint, "coaxial")
              + self._blocked_pair_mode_adjustment(
                  blocked_pair=blocked_pair,
                  relation_hint=relation_hint,
                  candidate_mode="coaxial",
              )
          )
          constraint = Constraint(
              ctype=ConstraintType.CONCENTRIC,
              part_a=parent_id,
              socket_a=parent_hole.name,
              part_b=child_id,
              socket_b=child_pin.name,
              value=axial_value,
              label=f"{child_id}_pin_into_{parent_id}_hole",
              metadata={
                  "relation_hint": relation_hint,
                  "stack_mode": "pin_into_hole",
                  "source_frame_hint": "entry_plus",
                  "target_frame_hint": "tip_plus",
                  "source_stop_hint": "seat_plus",
                  "target_stop_hint": "stop_plus",
                  "stack_phase": "entry",
                  "alignment_mode": "insert",
                  "carrier_part_id": parent_id,
                  "inserted_part_id": child_id,
                  "carrier_socket_kind": parent_hole.kind,
                  "inserted_socket_kind": child_pin.kind,
              },
          )
          constraints = [constraint]
          seat_diagnostics: list[str] = []
          seat = self._best_axial_seat_constraint(
              parent=parent,
              parent_socket=parent_hole,
              child=child,
              child_socket=child_pin,
              axial_value=axial_value,
              parent_id=parent_id,
              child_id=child_id,
              relation_hint=relation_hint,
              diagnostics=seat_diagnostics,
          )
          if seat is not None:
            constraints.append(seat)
            score += 0.38
          candidates.append(
              _ConnectionCandidate(
                  score=score,
                  constraints=constraints,
                  note=(
                      f"connect {parent_id}.{parent_hole.name} -> "
                      f"{child_id}.{child_pin.name} (concentric"
                      + (" + seat)" if seat is not None else ")")
                      + (
                          f" | {' | '.join(seat_diagnostics[:1])}"
                          if seat_diagnostics
                          else ""
                      )
                  ),
                  reserved_endpoints=[
                      (parent_id, parent_hole.name),
                      (child_id, child_pin.name),
                  ],
              )
          )

      for parent_pin in parent_pins[:4]:
        if not self._can_use_socket(parent, parent_pin, socket_usage):
          continue
        for child_hole in child_holes[:4]:
          if not self._can_use_socket(child, child_hole, socket_usage):
            continue
          axial_value = self._concentric_constraint_value(
              parent=parent,
              parent_socket=parent_pin,
              child=child,
              child_socket=child_hole,
          )
          score = (
              5.05
              + 0.08
              * (
                  self._socket_priority(parent_pin)
                  + self._socket_priority(child_hole)
              )
              + self._context_bonus(parent_id, child_id, mode="hole_on_pin")
              + self._connection_bonus(
                  parent=parent,
                  child=child,
                  parent_socket=parent_pin,
                  child_socket=child_hole,
                  mode="hole_on_pin",
              )
              - self._socket_reuse_penalty(parent, parent_pin, socket_usage)
              - self._socket_reuse_penalty(child, child_hole, socket_usage)
              + self._relation_hint_bonus(relation_hint, "coaxial")
              + self._blocked_pair_mode_adjustment(
                  blocked_pair=blocked_pair,
                  relation_hint=relation_hint,
                  candidate_mode="coaxial",
              )
          )
          constraint = Constraint(
              ctype=ConstraintType.CONCENTRIC,
              part_a=parent_id,
              socket_a=parent_pin.name,
              part_b=child_id,
              socket_b=child_hole.name,
              value=axial_value,
              label=f"{child_id}_hole_on_{parent_id}_pin",
              metadata={
                  "relation_hint": relation_hint,
                  "stack_mode": "hole_on_pin",
                  "source_frame_hint": "tip_plus",
                  "target_frame_hint": "entry_plus",
                  "source_stop_hint": "stop_plus",
                  "target_stop_hint": "seat_plus",
                  "stack_phase": "entry",
                  "alignment_mode": "insert",
                  "carrier_part_id": parent_id,
                  "inserted_part_id": child_id,
                  "carrier_socket_kind": parent_pin.kind,
                  "inserted_socket_kind": child_hole.kind,
              },
          )
          constraints = [constraint]
          seat_diagnostics = []
          seat = self._best_axial_seat_constraint(
              parent=parent,
              parent_socket=parent_pin,
              child=child,
              child_socket=child_hole,
              axial_value=axial_value,
              parent_id=parent_id,
              child_id=child_id,
              relation_hint=relation_hint,
              diagnostics=seat_diagnostics,
          )
          if seat is not None:
            constraints.append(seat)
            score += 0.38
          candidates.append(
              _ConnectionCandidate(
                  score=score,
                  constraints=constraints,
                  note=(
                      f"connect {parent_id}.{parent_pin.name} -> "
                      f"{child_id}.{child_hole.name} (concentric"
                      + (" + seat)" if seat is not None else ")")
                      + (
                          f" | {' | '.join(seat_diagnostics[:1])}"
                          if seat_diagnostics
                          else ""
                      )
                  ),
                  reserved_endpoints=[
                      (parent_id, parent_pin.name),
                      (child_id, child_hole.name),
                  ],
              )
          )

    plane_base = 2.8 if not blocked_pair else 2.3
    if allow_plane:
      for parent_plane in parent_planes[:4]:
        if not self._can_use_socket(parent, parent_plane, socket_usage):
          continue
        for child_plane in child_planes[:4]:
          if not self._can_use_socket(child, child_plane, socket_usage):
            continue
          score = (
              plane_base
              + 0.06
              * (
                  self._socket_priority(parent_plane)
                  + self._socket_priority(child_plane)
              )
              + self._context_bonus(parent_id, child_id, mode="plane")
              + self._connection_bonus(
                  parent=parent,
                  child=child,
                  parent_socket=parent_plane,
                  child_socket=child_plane,
                  mode="plane",
              )
              + self._relation_hint_bonus(relation_hint, "planar")
              + self._blocked_pair_mode_adjustment(
                  blocked_pair=blocked_pair,
                  relation_hint=relation_hint,
                  candidate_mode="planar",
              )
          )
          constraint = Constraint(
              ctype=ConstraintType.COINCIDENT,
              part_a=parent_id,
              socket_a=parent_plane.name,
              part_b=child_id,
              socket_b=child_plane.name,
              label=f"{child_id}_plane_to_{parent_id}_plane",
              metadata={
                  "source_frame_hint": "support",
                  "target_frame_hint": "support",
                  "stack_phase": "seat",
                  "alignment_mode": "support",
              },
          )
          candidates.append(
              _ConnectionCandidate(
                  score=score,
                  constraints=[constraint],
                  note=(
                      f"connect {parent_id}.{parent_plane.name} -> "
                      f"{child_id}.{child_plane.name} (coincident)"
                  ),
                  reserved_endpoints=[
                      (parent_id, parent_plane.name),
                      (child_id, child_plane.name),
                  ],
              )
          )

    if allow_cyl:
      for parent_axis in parent_axes[:3]:
        if not self._can_use_socket(parent, parent_axis, socket_usage):
          continue
        for child_axis in child_axes[:3]:
          if not self._can_use_socket(child, child_axis, socket_usage):
            continue
          score = (
              2.3
              + 0.08
              * (
                  self._socket_priority(parent_axis)
                  + self._socket_priority(child_axis)
              )
              - self._socket_reuse_penalty(parent, parent_axis, socket_usage)
              - self._socket_reuse_penalty(child, child_axis, socket_usage)
              + self._preferred_neighbor_bonus(parent, child)
              + self._relation_hint_bonus(relation_hint, "coaxial")
              + self._blocked_pair_mode_adjustment(
                  blocked_pair=blocked_pair,
                  relation_hint=relation_hint,
                  candidate_mode="coaxial",
              )
          )
          constraint = Constraint(
              ctype=ConstraintType.CONCENTRIC,
              part_a=parent_id,
              socket_a=parent_axis.name,
              part_b=child_id,
              socket_b=child_axis.name,
              label=f"{child_id}_axis_to_{parent_id}_axis",
              metadata={
                  "source_frame_hint": "primary",
                  "target_frame_hint": "primary",
                  "stack_phase": "through_stop",
                  "alignment_mode": "stack",
              },
          )
          candidates.append(
              _ConnectionCandidate(
                  score=score,
                  constraints=[constraint],
                  note=(
                      f"connect {parent_id}.{parent_axis.name} -> "
                      f"{child_id}.{child_axis.name} (axis)"
                  ),
                  reserved_endpoints=[
                      (parent_id, parent_axis.name),
                      (child_id, child_axis.name),
                  ],
              )
          )

    if not candidates:
      if relation_hint in {"coaxial", "fasten"}:
        return None
      fallback_kinds = {"plane", "flange_plane", "guide_slot"}
      parent_any = self._first_usable_socket(
          parent, socket_usage, fallback_kinds
      )
      child_any = self._first_usable_socket(
          child, socket_usage, fallback_kinds
      )
      if (parent_any is None or child_any is None) and relation_hint == "link":
        parent_any = self._first_usable_socket(parent, socket_usage, set())
        child_any = self._first_usable_socket(child, socket_usage, set())
      if not parent_any or not child_any:
        return None
      score = (
          0.2
          + 0.04
          * (self._socket_priority(parent_any) + self._socket_priority(child_any))
      )
      fallback = Constraint(
          ctype=ConstraintType.COINCIDENT,
          part_a=parent_id,
          socket_a=parent_any.name,
          part_b=child_id,
          socket_b=child_any.name,
          label=f"{child_id}_fallback",
          metadata={
              "source_frame_hint": "support",
              "target_frame_hint": "support",
              "stack_phase": "seat",
              "alignment_mode": "mate",
          },
      )
      return _ConnectionCandidate(
          score=score,
          constraints=[fallback],
          note=(
              f"fallback {parent_id}.{parent_any.name} -> "
              f"{child_id}.{child_any.name}"
          ),
          reserved_endpoints=[
              (parent_id, parent_any.name),
              (child_id, child_any.name),
          ],
      )

    best = max(candidates, key=lambda c: c.score)
    return best

  def _concentric_constraint_value(
      self,
      parent: PartInstance,
      parent_socket: Socket,
      child: PartInstance,
      child_socket: Socket,
  ) -> Optional[float]:
    if {parent_socket.kind, child_socket.kind} != {"hole", "pin"}:
      return None
    axis_parent = self._socket_axis_or_normal(parent_socket)
    axis_child = self._socket_axis_or_normal(child_socket)

    parent_interval = self._bbox_projection_interval(parent, parent_socket, axis_parent)
    child_interval = self._bbox_projection_interval(child, child_socket, axis_child)
    parent_support = self._support_projections_along_socket(
        parent,
        parent_socket,
        axis_parent,
    )
    child_support = self._support_projections_along_socket(
        child,
        child_socket,
        axis_child,
    )

    candidate_offsets = {0.0}
    for parent_proj in parent_support:
      for child_proj in child_support:
        candidate_offsets.add(float(parent_proj - child_proj))

    best_value = 0.0
    best_rank = None
    for value in candidate_offsets:
      child_shifted = (
          float(child_interval[0] + value),
          float(child_interval[1] + value),
      )
      overlap = self._interval_overlap_length(parent_interval, child_shifted)
      rank = (
          overlap,
          0 if abs(value) > self.default_clearance else 1,
          abs(value),
      )
      if best_rank is None or rank < best_rank:
        best_rank = rank
        best_value = float(value)

    if abs(best_value) <= 1e-6:
      return None
    return best_value

  def _best_axial_seat_constraint(
      self,
      parent: PartInstance,
      parent_socket: Socket,
      child: PartInstance,
      child_socket: Socket,
      axial_value: Optional[float],
      parent_id: str,
      child_id: str,
      relation_hint: Optional[str] = None,
      diagnostics: Optional[list[str]] = None,
  ) -> Optional[Constraint]:
    if axial_value is None:
      axial_value = 0.0
    axis_parent = self._socket_axis_or_normal(parent_socket)
    axis_child = self._socket_axis_or_normal(child_socket)
    parent_planes = self._support_plane_candidates(parent, parent_socket, axis_parent)
    child_planes = self._support_plane_candidates(child, child_socket, axis_child)
    if not parent_planes or not child_planes:
      return None

    best: Optional[
        tuple[tuple[float, float, float], Socket, Socket, float, float]
    ] = None
    local_best: Optional[
        tuple[tuple[float, float, float], Socket, Socket, float, float]
    ] = None
    candidate_diags: list[tuple[tuple[float, float, float], str]] = []
    tolerance = max(
        0.35,
        0.02 * max(
            np.linalg.norm(parent.local_bbox_max - parent.local_bbox_min),
            np.linalg.norm(child.local_bbox_max - child.local_bbox_min),
        ),
    )
    socket_radii = [
        float(value)
        for value in (parent_socket.radius, child_socket.radius)
        if value is not None and float(value) > 0.0
    ]
    socket_radius_scale = min(socket_radii) if socket_radii else 1.0
    hint = self._normalize_relation_hint(relation_hint)
    if (
        hint in {"coaxial", "fasten"}
        and socket_radius_scale < 1.2
        and {parent_socket.kind, child_socket.kind} == {"hole", "pin"}
    ):
      return None
    radial_tolerance = min(
        max(0.18, 0.12 * socket_radius_scale),
        max(0.45, 0.25 * tolerance),
    )
    if hint in {"coaxial", "fasten"}:
      radial_tolerance = min(radial_tolerance, 0.28)
    for parent_plane, parent_proj, parent_sign in parent_planes:
      for child_plane, child_proj, child_sign in child_planes:
        if parent_sign * child_sign >= 0.0:
          continue
        target_delta = float(parent_proj - child_proj)
        err = abs(target_delta - float(axial_value))
        parent_radial = self._plane_radial_offset(
            socket=parent_socket,
            plane=parent_plane,
            axis=axis_parent,
        )
        child_radial = self._plane_radial_offset(
            socket=child_socket,
            plane=child_plane,
            axis=axis_child,
        )
        radial_err = abs(parent_radial - child_radial)
        if radial_err > radial_tolerance:
          continue
        radial_band = max(0.25, 0.28 * socket_radius_scale)
        if hint in {"coaxial", "fasten"} and (
            max(parent_radial, child_radial) > 0.65 or radial_err > 0.28
        ):
          continue
        source_penalty = 0.0
        quality_bonus = 0.0
        for plane in (parent_plane, child_plane):
          source = str(plane.metadata.get("source") or "").strip().lower()
          if source == "axial_proxy_support":
            source_penalty += 0.18
          elif source == "frame_variant_support":
            source_penalty += 0.03
          elif bool(plane.metadata.get("proxy_socket", False)):
            source_penalty += 0.10
          quality_bonus += 0.04 * float(plane.metadata.get("quality", 0.0) or 0.0)
        parent_role = str(
            parent_plane.metadata.get("support_role") or "seat"
        ).strip().lower()
        child_role = str(
            child_plane.metadata.get("support_role") or "seat"
        ).strip().lower()
        role_penalty = 0.0
        if "through_stop" in {parent_role, child_role}:
          role_penalty += 0.24
        if "shoulder" in {parent_role, child_role}:
          role_penalty -= 0.10
        if (
            (parent_role == "seat" and child_role == "through_stop")
            or (child_role == "seat" and parent_role == "through_stop")
        ):
          role_penalty += 0.14
        radial_band_penalty = 0.15 * max(
            0.0,
            max(parent_radial, child_radial) - 0.5 * radial_band,
        )
        rank = (
            err
            + 0.85 * radial_err
            + source_penalty
            + radial_band_penalty
            + role_penalty,
            radial_err,
            radial_band_penalty + source_penalty + role_penalty - quality_bonus,
        )
        parent_source = str(parent_plane.metadata.get("source") or "").strip().lower()
        child_source = str(child_plane.metadata.get("source") or "").strip().lower()
        candidate_diags.append(
            (
                rank,
                (
                    f"{parent_plane.name}[{parent_role}/{parent_source}]"
                    f" <-> {child_plane.name}[{child_role}/{child_source}]"
                    f" err={rank[0]:.3f} radial={radial_err:.3f}"
                ),
            )
        )
        if best is None or rank < best[0]:
          best = (rank, parent_plane, child_plane, parent_sign, child_sign)
        local_sources = {"frame_variant_support", "flange_backface_proxy"}
        if (
            (parent_source in local_sources or child_source in local_sources)
            and "shoulder" in {parent_role, child_role}
        ):
          if local_best is None or rank < local_best[0]:
            local_best = (rank, parent_plane, child_plane, parent_sign, child_sign)
    if best is None:
      if diagnostics is not None:
        diagnostics.append("seat_diag none candidates=0")
      return None
    if best[0][0] > tolerance:
      if socket_radius_scale < 1.2:
        if diagnostics is not None:
          top = "; ".join(text for _, text in sorted(candidate_diags, key=lambda item: item[0])[:3])
          diagnostics.append(
              f"seat_diag tiny_socket_reject best={best[0][0]:.3f} tol={tolerance:.3f} "
              f"top=[{top}]"
          )
        return None
      relaxed_tolerance = max(tolerance * 2.5, 18.0)
      if local_best is None or local_best[0][0] > relaxed_tolerance:
        if diagnostics is not None:
          top = "; ".join(text for _, text in sorted(candidate_diags, key=lambda item: item[0])[:3])
          diagnostics.append(
              f"seat_diag rejected best={best[0][0]:.3f} tol={tolerance:.3f} "
              f"relaxed={relaxed_tolerance:.3f} top=[{top}]"
          )
        return None
      best = local_best
      if diagnostics is not None:
        top = "; ".join(text for _, text in sorted(candidate_diags, key=lambda item: item[0])[:3])
        diagnostics.append(
            f"seat_diag relaxed best={best[0][0]:.3f} tol={tolerance:.3f} "
            f"top=[{top}]"
        )
    elif diagnostics is not None:
      top = "; ".join(text for _, text in sorted(candidate_diags, key=lambda item: item[0])[:3])
      diagnostics.append(
          f"seat_diag selected best={best[0][0]:.3f} tol={tolerance:.3f} top=[{top}]"
      )
    parent_plane = best[1]
    child_plane = best[2]
    parent_sign = float(best[3])
    child_sign = float(best[4])
    parent_role = str(parent_plane.metadata.get("support_role") or "seat").strip().lower()
    child_role = str(child_plane.metadata.get("support_role") or "seat").strip().lower()
    role_set = {parent_role, child_role}
    stack_phase = (
        "through_stop"
        if role_set == {"through_stop"} or role_set == {"through_stop", "shoulder"}
        else "seat"
    )
    if stack_phase == "through_stop":
      carrier_seat_hint = "through_plus" if parent_sign > 0.0 else "through_minus"
    else:
      carrier_seat_hint = "seat_plus" if parent_sign > 0.0 else "seat_minus"
    inserted_stop_hint = "stop_plus" if child_sign > 0.0 else "stop_minus"
    return Constraint(
        ctype=ConstraintType.COINCIDENT,
        part_a=parent_id,
        socket_a=parent_plane.name,
        part_b=child_id,
        socket_b=child_plane.name,
        label=f"{child_id}_seat_on_{parent_id}",
        metadata={
            "secondary_seat": True,
            "seat_parent_socket": parent_plane.name,
            "seat_child_socket": child_plane.name,
            "source_frame_hint": "support",
            "target_frame_hint": "support",
            "source_stop_hint": carrier_seat_hint,
            "target_stop_hint": inserted_stop_hint,
            "carrier_stop_hint": carrier_seat_hint,
            "inserted_stop_hint": inserted_stop_hint,
            "stack_phase": stack_phase,
            "alignment_mode": "through_stop" if stack_phase == "through_stop" else "support",
            "carrier_support_role": parent_role,
            "inserted_support_role": child_role,
        },
    )

  def _plane_radial_offset(
      self,
      socket: Socket,
      plane: Socket,
      axis: np.ndarray,
  ) -> float:
    delta = plane.origin - socket.origin
    delta = delta - float(np.dot(delta, axis)) * axis
    return float(np.linalg.norm(delta))

  def _support_plane_candidates(
      self,
      instance: PartInstance,
      socket: Socket,
      axis: np.ndarray,
  ) -> list[tuple[Socket, float, float]]:
    candidates: list[tuple[Socket, float, float]] = []
    support_limit = self._socket_support_distance_limit(socket)
    for other in list(instance.sockets.values()):
      if other.name == socket.name:
        continue
      if other.kind not in {"plane", "flange_plane"}:
        continue
      if other.normal is None:
        continue
      align = float(np.dot(other.normal, axis))
      if abs(align) < 0.92:
        continue
      proj = float(np.dot(other.origin - socket.origin, axis))
      if support_limit is not None and abs(proj) > support_limit:
        continue
      other.metadata.setdefault("support_role", "seat")
      candidates.append((other, proj, float(np.sign(align) or 1.0)))
    candidates.extend(
        self._frame_variant_support_candidates(
            instance=instance,
            socket=socket,
            axis=axis,
            existing_candidates=candidates,
        )
    )
    candidates.extend(
        self._flange_backface_support_candidates(
            instance=instance,
            socket=socket,
            axis=axis,
            existing_candidates=candidates,
        )
    )
    candidates.extend(
        self._proxy_support_plane_candidates(
            instance=instance,
            socket=socket,
            axis=axis,
            existing_candidates=candidates,
        )
    )
    return sorted(candidates, key=lambda item: abs(item[1]))

  def _socket_support_distance_limit(self, socket: Socket) -> Optional[float]:
    if socket.kind not in {"hole", "threaded_hole", "pin", "axis"}:
      return None
    variants = socket.local_frame_variants()
    plus = variants.get("end_plus")
    minus = variants.get("end_minus")
    if plus is None or minus is None:
      return None
    length_est = float(np.linalg.norm(plus.origin - minus.origin))
    if length_est <= 1e-4:
      return None
    return max(8.0, 0.75 * length_est + 6.0)

  def _frame_variant_support_candidates(
      self,
      instance: PartInstance,
      socket: Socket,
      axis: np.ndarray,
      existing_candidates: list[tuple[Socket, float, float]],
  ) -> list[tuple[Socket, float, float]]:
    role_map = {
        "seat_plus": "seat",
        "seat_minus": "seat",
        "through_plus": "through_stop",
        "through_minus": "through_stop",
        "through_stop_plus": "through_stop",
        "through_stop_minus": "through_stop",
        "hole_depth_stop_plus": "through_stop",
        "hole_depth_stop_minus": "through_stop",
        "stop_plus": "shoulder",
        "stop_minus": "shoulder",
        "shoulder_plus": "shoulder",
        "shoulder_minus": "shoulder",
        "pin_shoulder_plus": "shoulder",
        "pin_shoulder_minus": "shoulder",
        "support": "seat",
    }
    variants = socket.local_frame_variants()
    if len(variants) <= 1:
      return []
    existing_proj = [float(item[1]) for item in existing_candidates]
    proxy_candidates: list[tuple[Socket, float, float]] = []
    for variant_name, role in role_map.items():
      frame = variants.get(variant_name)
      if frame is None:
        continue
      normal = frame.normal if frame.normal is not None else frame.axis
      if normal is None:
        continue
      normal = normalize(normal)
      align = float(np.dot(normal, axis))
      if abs(align) < 0.92:
        continue
      proj = float(np.dot(frame.origin - socket.origin, axis))
      if any(abs(proj - item) <= 0.2 for item in existing_proj) and not variant_name.startswith("stop_"):
        continue
      sign = float(np.sign(align) or np.sign(proj) or 1.0)
      proxy_name = f"{socket.name}__frame_{variant_name}"
      proxy_socket = instance.sockets.get(proxy_name)
      quality = 0.10 if role == "seat" else (0.14 if role == "shoulder" else -0.06)
      if proxy_socket is None:
        proxy_socket = Socket(
            name=proxy_name,
            kind="plane",
            origin=frame.origin,
            normal=normal,
            x_axis=frame.x_axis,
            y_axis=frame.y_axis,
            z_axis=frame.z_axis,
            radius=frame.radius,
            metadata={
                "source": "frame_variant_support",
                "proxy_socket": True,
                "mating_enabled": True,
                "support_role": role,
                "frame_variant_name": variant_name,
                "support_projection": float(proj),
                "quality": quality,
            },
        )
        instance.sockets[proxy_name] = proxy_socket
      else:
        proxy_socket.origin = frame.origin
        proxy_socket.normal = normal
        proxy_socket.x_axis = frame.x_axis
        proxy_socket.y_axis = frame.y_axis
        proxy_socket.z_axis = frame.z_axis
        proxy_socket.radius = frame.radius
        proxy_socket.metadata.update(
            {
                "source": "frame_variant_support",
                "proxy_socket": True,
                "mating_enabled": True,
                "support_role": role,
                "frame_variant_name": variant_name,
                "support_projection": float(proj),
                "quality": quality,
            }
        )
      proxy_candidates.append((proxy_socket, float(proj), sign))
    return proxy_candidates

  def _flange_backface_support_candidates(
      self,
      instance: PartInstance,
      socket: Socket,
      axis: np.ndarray,
      existing_candidates: list[tuple[Socket, float, float]],
  ) -> list[tuple[Socket, float, float]]:
    support_limit = self._socket_support_distance_limit(socket)
    existing_proj = [float(item[1]) for item in existing_candidates]
    proxy_candidates: list[tuple[Socket, float, float]] = []
    for other in list(instance.sockets.values()):
      if other.name == socket.name or other.kind != "flange_plane" or other.normal is None:
        continue
      local_axis = normalize(other.normal)
      if abs(float(np.dot(local_axis, axis))) < 0.92:
        continue
      local_projections = self._support_projections_along_socket(
          instance=instance,
          socket=other,
          axis=local_axis,
      )
      negative = [value for value in local_projections if value < -0.25]
      positive = [value for value in local_projections if value > 0.25]
      tagged: list[tuple[float, np.ndarray]] = []
      if negative:
        tagged.append((negative[-1], -local_axis))
      if positive:
        tagged.append((positive[0], local_axis))
      for local_proj, normal in tagged:
        origin = other.origin + local_axis * float(local_proj)
        proj = float(np.dot(origin - socket.origin, axis))
        if support_limit is not None and abs(proj) > support_limit:
          continue
        if any(abs(proj - item) <= 0.25 for item in existing_proj):
          continue
        sign = float(np.sign(np.dot(normal, axis)) or np.sign(proj) or 1.0)
        suffix = "plus" if sign > 0.0 else "minus"
        proxy_name = f"{other.name}__proxy_backface_{suffix}"
        proxy_socket = instance.sockets.get(proxy_name)
        if proxy_socket is None:
          proxy_socket = Socket(
              name=proxy_name,
              kind="plane",
              origin=origin,
              normal=normal,
              x_axis=other.x_axis,
              y_axis=other.y_axis,
              z_axis=other.z_axis,
              radius=other.radius,
              metadata={
                  "source": "flange_backface_proxy",
                  "proxy_socket": True,
                  "mating_enabled": True,
                  "support_role": "shoulder",
                  "quality": 0.14,
              },
          )
          instance.sockets[proxy_name] = proxy_socket
        else:
          proxy_socket.origin = origin
          proxy_socket.normal = normal
          proxy_socket.x_axis = other.x_axis
          proxy_socket.y_axis = other.y_axis
          proxy_socket.z_axis = other.z_axis
          proxy_socket.radius = other.radius
          proxy_socket.metadata.update(
              {
                  "source": "flange_backface_proxy",
                  "proxy_socket": True,
                  "mating_enabled": True,
                  "support_role": "shoulder",
                  "quality": 0.14,
              }
          )
        proxy_candidates.append((proxy_socket, proj, sign))
    return proxy_candidates

  def _proxy_support_plane_candidates(
      self,
      instance: PartInstance,
      socket: Socket,
      axis: np.ndarray,
      existing_candidates: list[tuple[Socket, float, float]],
  ) -> list[tuple[Socket, float, float]]:
    projections = self._support_projections_along_socket(instance, socket, axis)
    if not projections:
      return []
    existing_proj = [float(item[1]) for item in existing_candidates]
    proxy_candidates: list[tuple[Socket, float, float]] = []
    support_limit = self._socket_support_distance_limit(socket)
    negative = sorted(value for value in projections if value < -1e-4)
    positive = sorted(value for value in projections if value > 1e-4)
    tagged: list[tuple[float, str]] = []
    if negative:
      tagged.append((negative[-1], "shoulder"))
      tagged.append((negative[0], "through_stop"))
    if positive:
      tagged.append((positive[0], "shoulder"))
      tagged.append((positive[-1], "through_stop"))
    seen_keys: set[tuple[int, str]] = set()
    for proj, role in tagged:
      if support_limit is not None and abs(proj) > support_limit:
        continue
      if any(abs(proj - item) <= 0.25 for item in existing_proj):
        continue
      sign = float(np.sign(proj) or 1.0)
      key = (int(round(proj * 1000.0)), 1 if role == "through_stop" else 0)
      if key in seen_keys:
        continue
      seen_keys.add(key)
      normal = axis if sign > 0.0 else -axis
      suffix = "plus" if sign > 0.0 else "minus"
      proxy_name = f"{socket.name}__proxy_{role}_{suffix}"
      proxy_socket = instance.sockets.get(proxy_name)
      if proxy_socket is None:
        proxy_socket = Socket(
          name=proxy_name,
          kind="plane",
          origin=socket.origin + axis * float(proj),
          normal=normal,
          metadata={
              "source": "axial_proxy_support",
              "proxy_socket": True,
              "mating_enabled": True,
              "support_role": role,
              "support_projection": float(proj),
              "quality": (0.08 if role == "shoulder" else -0.06) - 0.03 * abs(float(proj)),
          },
        )
        instance.sockets[proxy_name] = proxy_socket
      else:
        proxy_socket.origin = socket.origin + axis * float(proj)
        proxy_socket.normal = normal
        proxy_socket.metadata.update(
            {
                "source": "axial_proxy_support",
                "proxy_socket": True,
                "mating_enabled": True,
                "support_role": role,
                "support_projection": float(proj),
                "quality": (0.08 if role == "shoulder" else -0.06) - 0.03 * abs(float(proj)),
            }
        )
      proxy_candidates.append((proxy_socket, float(proj), sign))
    return proxy_candidates

  def _support_projections_along_socket(
      self,
      instance: PartInstance,
      socket: Socket,
      axis: np.ndarray,
  ) -> list[float]:
    projections: set[float] = set()
    bbox_min, bbox_max = self._bbox_projection_interval(instance, socket, axis)
    projections.add(float(bbox_min))
    projections.add(float(bbox_max))
    for other in list(instance.sockets.values()):
      if other.name == socket.name:
        continue
      if other.normal is None:
        continue
      if other.kind not in {"plane", "flange_plane"}:
        continue
      if abs(float(np.dot(other.normal, axis))) < 0.92:
        continue
      proj = float(np.dot(other.origin - socket.origin, axis))
      projections.add(proj)
    variants = socket.local_frame_variants()
    for variant_name in (
        "seat_plus",
        "seat_minus",
        "through_plus",
        "through_minus",
        "stop_plus",
        "stop_minus",
        "support",
    ):
      frame = variants.get(variant_name)
      if frame is None:
        continue
      projections.add(float(np.dot(frame.origin - socket.origin, axis)))
    ordered = sorted(projections)
    if not ordered:
      return [0.0]
    negative = [value for value in ordered if value <= 0.0]
    positive = [value for value in ordered if value >= 0.0]
    compact: list[float] = []
    if negative:
      compact.append(negative[0])
      compact.append(negative[-1])
    if positive:
      compact.append(positive[0])
      compact.append(positive[-1])
    compact.append(0.0)
    return sorted({round(float(value), 6) for value in compact})

  def _bbox_projection_interval(
      self,
      instance: PartInstance,
      socket: Socket,
      axis: np.ndarray,
  ) -> tuple[float, float]:
    mins = instance.local_bbox_min
    maxs = instance.local_bbox_max
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=float,
    )
    centered = corners - socket.origin.reshape(1, 3)
    projections = centered @ axis.reshape(3, 1)
    values = projections.reshape(-1)
    return float(np.min(values)), float(np.max(values))

  def _interval_overlap_length(
      self,
      interval_a: tuple[float, float],
      interval_b: tuple[float, float],
  ) -> float:
    lo = max(float(interval_a[0]), float(interval_b[0]))
    hi = min(float(interval_a[1]), float(interval_b[1]))
    return max(0.0, hi - lo)

  def _socket_reuse_penalty(
      self,
      instance: PartInstance,
      socket: Socket,
      socket_usage: dict[tuple[str, str], int],
  ) -> float:
    usage = socket_usage.get((instance.instance_id, socket.name), 0)
    if usage <= 0:
      return 0.0
    if socket.kind in {"pin", "axis"}:
      return 1.35 * float(usage)
    if socket.kind in {"plane", "flange_plane"}:
      return 0.2 * float(usage)
    return 0.5 * float(usage)

  def _socket_axis_or_normal(self, socket: Socket) -> np.ndarray:
    if socket.axis is not None:
      return np.asarray(socket.axis, dtype=float).reshape(3)
    if socket.normal is not None:
      return np.asarray(socket.normal, dtype=float).reshape(3)
    raise ValueError(f"Socket '{socket.name}' has no axis or normal.")

  def _constraints_from_feedback(
      self,
      collisions: list[CollisionRecord],
      instances: dict[str, PartInstance],
      existing_constraints: list[Constraint],
  ) -> list[Constraint]:
    concentric_pairs = {
        frozenset((constraint.part_a, constraint.part_b))
        for constraint in existing_constraints
        if constraint.ctype == ConstraintType.CONCENTRIC
    }
    constraints: list[Constraint] = []
    for collision in collisions:
      part_a = instances.get(collision.part_a)
      part_b = instances.get(collision.part_b)
      if not part_a or not part_b:
        continue
      pair = frozenset((part_a.instance_id, part_b.instance_id))
      if pair in concentric_pairs and collision.volume <= max(
          self.default_clearance**3, 1e-4
      ):
        # Small contact on already-mated pair can be acceptable.
        continue
      socket_a = self._find_socket(part_a, {"flange_plane", "plane"})
      socket_b = self._find_socket(part_b, {"flange_plane", "plane"})
      if not socket_a or not socket_b:
        continue

      clearance = self._clearance_from_overlap(collision, socket_a)
      constraints.append(
          Constraint(
              ctype=ConstraintType.DISTANCE,
              part_a=part_a.instance_id,
              socket_a=socket_a.name,
              part_b=part_b.instance_id,
              socket_b=socket_b.name,
              value=clearance,
              label=f"clearance_{part_a.instance_id}_{part_b.instance_id}",
              metadata={"reason": "collision_feedback"},
          )
      )
    return constraints

  def _clearance_from_overlap(
      self, collision: CollisionRecord, reference_socket: Socket
  ) -> float:
    overlap = collision.overlap_max - collision.overlap_min
    if reference_socket.normal is None:
      overlap_depth = float(np.max(overlap))
    else:
      axis = int(np.argmax(np.abs(reference_socket.normal)))
      overlap_depth = float(overlap[axis])
    return max(self.default_clearance, overlap_depth + self.default_clearance)

  def _append_unique(
      self, constraints: list[Constraint], incoming: Constraint
  ) -> None:
    signatures = {constraint.signature() for constraint in constraints}
    if incoming.signature() in signatures:
      return
    constraints.append(incoming)

  def _blocked_pairs_from_feedback(
      self, feedback: Optional[PlannerFeedback]
  ) -> set[frozenset[str]]:
    if feedback is None:
      return set()
    return {
        frozenset((collision.part_a, collision.part_b))
        for collision in feedback.collisions
    }

  def _reserve_candidate(
      self,
      candidate: _ConnectionCandidate,
      socket_usage: dict[tuple[str, str], int],
  ) -> None:
    for endpoint in candidate.reserved_endpoints:
      socket_usage[endpoint] = socket_usage.get(endpoint, 0) + 1

  def _can_use_socket(
      self,
      instance: PartInstance,
      socket: Socket,
      socket_usage: dict[tuple[str, str], int],
  ) -> bool:
    capacity = self._socket_capacity(socket)
    key = (instance.instance_id, socket.name)
    return socket_usage.get(key, 0) < capacity

  def _socket_capacity(self, socket: Socket) -> int:
    if socket.kind in self.single_use_kinds:
      return 1
    if socket.kind in {"pin", "axis"}:
      return max(2, self.plane_capacity)
    if socket.kind in {"plane", "flange_plane"}:
      return self.plane_capacity
    if socket.kind == "interface":
      return 1
    return 2

  def _sorted_sockets(
      self,
      instance: PartInstance,
      accepted_kinds: set[str],
      relation_hint: Optional[str] = None,
      partner_id: Optional[str] = None,
  ) -> list[Socket]:
    sockets = list(instance.sockets.values())
    if accepted_kinds:
      sockets = [socket for socket in sockets if socket.kind in accepted_kinds]
    sockets = [
        socket
        for socket in sockets
        if self._socket_allowed_for_mating(
            instance_id=instance.instance_id,
            socket=socket,
            relation_hint=relation_hint,
            partner_id=partner_id,
        )
    ]
    return sorted(sockets, key=self._socket_priority, reverse=True)

  def _prioritize_hole_like(self, sockets: list[Socket]) -> list[Socket]:
    # For assembly defaults, plain holes are usually more reliable than
    # threaded holes as insertion targets.
    return sorted(
        sockets,
        key=lambda socket: (
            0 if socket.kind == "hole" else 2,
            -self._socket_priority(socket),
        ),
    )

  def _first_usable_socket(
      self,
      instance: PartInstance,
      socket_usage: dict[tuple[str, str], int],
      accepted_kinds: set[str],
  ) -> Optional[Socket]:
    for socket in self._sorted_sockets(instance, accepted_kinds):
      if self._can_use_socket(instance, socket, socket_usage):
        return socket
    return None

  def _socket_allowed_for_mating(
      self,
      instance_id: str,
      socket: Socket,
      relation_hint: Optional[str],
      partner_id: Optional[str],
  ) -> bool:
    relation_hint = self._normalize_relation_hint(relation_hint)
    partner_id = None if partner_id is None else str(partner_id)

    if bool(socket.metadata.get("non_mating_surface", False)):
      return False

    fastener_context = bool(partner_id) and self._is_fastener_like(partner_id)
    if socket.kind == "threaded_hole":
      if relation_hint != "fasten" and not fastener_context:
        return False

    mating_enabled = socket.metadata.get("mating_enabled")
    if mating_enabled is False and not (
        socket.kind == "threaded_hole"
        and (relation_hint == "fasten" or fastener_context)
    ):
      return False

    if relation_hint == "planar":
      return socket.kind in {"plane", "flange_plane", "guide_slot", "interface"}
    if relation_hint in {"coaxial", "fasten"}:
      return socket.kind in {"hole", "threaded_hole", "pin", "axis", "guide_slot", "interface"}
    return True

  def _socket_priority(self, socket: Socket) -> float:
    score = float(socket.metadata.get("quality", 0.0))
    if socket.kind == "guide_slot":
      score *= 0.12
    if bool(socket.metadata.get("non_mating_surface", False)):
      score -= 4.0
    if socket.metadata.get("mating_enabled") is False:
      score -= 2.0
    area = socket.metadata.get("face_area")
    if isinstance(area, (int, float)):
      score += 0.05 * float(np.log1p(max(float(area), 0.0)))
    if socket.kind == "hole":
      score += 0.9
    elif socket.kind == "threaded_hole":
      score += 0.25
    elif socket.kind == "pin":
      score += 0.5
    elif socket.kind in {"flange_plane", "plane"}:
      score += 0.25
    elif socket.kind == "interface":
      score += 1.25 + float(socket.metadata.get("interface_score", 0.0) or 0.0)
    if socket.radius is not None:
      score += 0.05 * float(socket.radius)
    return score

  def _context_bonus(self, parent_id: str, child_id: str, mode: str) -> float:
    parent = parent_id.lower()
    child = child_id.lower()

    if mode == "pin_into_hole":
      bonus = 0.0
      if any(token in parent for token in {"base", "housing", "frame"}):
        bonus += 0.25
      if "gear" in child and any(token in parent for token in {"base", "housing"}):
        bonus -= 0.35
      return bonus

    if mode == "hole_on_pin":
      bonus = 0.0
      if any(token in parent for token in {"shaft", "axle"}):
        bonus += 0.45
      if "gear" in child and any(token in parent for token in {"shaft", "axle"}):
        bonus += 0.35
      if "gear" in child and any(token in parent for token in {"base", "housing"}):
        bonus -= 0.5
      return bonus

    if mode == "plane":
      if any(token in parent for token in {"base", "housing", "frame"}):
        return 0.2
      return 0.0

    return 0.0

  def _connection_bonus(
      self,
      parent: PartInstance,
      child: PartInstance,
      parent_socket: Socket,
      child_socket: Socket,
      mode: str,
  ) -> float:
    bonus = self._preferred_neighbor_bonus(parent, child)

    if mode == "pin_into_hole":
      bonus += self._radius_fit_bonus(parent_socket, child_socket)
      bonus += self._threaded_receiver_penalty(
          receiver_socket=parent_socket,
          inserted_part_id=child.instance_id,
      )
      if self._is_base_like(parent.instance_id) and self._is_shaft_like(
          child.instance_id
      ):
        bonus += 0.65
      if self._is_base_like(parent.instance_id) and self._is_gear_like(
          child.instance_id
      ):
        bonus -= 0.45
      return bonus

    if mode == "hole_on_pin":
      bonus += self._radius_fit_bonus(child_socket, parent_socket)
      bonus += self._threaded_receiver_penalty(
          receiver_socket=child_socket,
          inserted_part_id=parent.instance_id,
      )
      if self._is_base_like(parent.instance_id) and self._is_shaft_like(
          child.instance_id
      ):
        bonus -= 0.95
      if self._is_shaft_like(parent.instance_id) and (
          self._is_gear_like(child.instance_id)
          or self._is_collar_like(child.instance_id)
      ):
        bonus += 0.75
      return bonus

    if mode == "plane":
      if self._is_base_like(parent.instance_id):
        bonus += 0.1
      return bonus

    if mode == "interface":
      bonus += 0.2
      if self._is_base_like(parent.instance_id):
        bonus += 0.15
      return bonus

    return bonus

  def _normalize_relation_hint(self, relation_hint: Optional[str]) -> str:
    if relation_hint is None:
      return "link"
    text = str(relation_hint).strip().lower()
    if not text:
      return "link"
    if text in {"coaxial", "concentric", "insert", "mount", "align"}:
      return "coaxial"
    if text in {"planar", "plane", "coincident", "flush", "mate"}:
      return "planar"
    if text in {"fasten", "threaded", "screw", "bolt"}:
      return "fasten"
    if text in {"support", "rest", "seat"}:
      return "support"
    return "link"

  def _relation_hint_bonus(
      self,
      relation_hint: Optional[str],
      candidate_mode: str,
  ) -> float:
    hint = self._normalize_relation_hint(relation_hint)
    if hint == "link":
      return 0.0
    if hint == "support":
      if candidate_mode == "planar":
        return 0.35
      if candidate_mode == "interface":
        return 0.28
      if candidate_mode == "coaxial":
        return 0.05
      return 0.0
    if hint == "fasten":
      if candidate_mode == "interface":
        return 0.35
      return 0.55 if candidate_mode == "coaxial" else -0.35
    if hint == "planar":
      if candidate_mode == "interface":
        return 0.45
      return 0.7 if candidate_mode == "planar" else -0.45
    if hint == "coaxial":
      if candidate_mode == "interface":
        return 0.45
      return 0.7 if candidate_mode == "coaxial" else -0.45
    return 0.0

  def _blocked_pair_mode_adjustment(
      self,
      blocked_pair: bool,
      relation_hint: Optional[str],
      candidate_mode: str,
  ) -> float:
    if not blocked_pair:
      return 0.0
    hint = self._normalize_relation_hint(relation_hint)
    if hint == "planar":
      if candidate_mode == "coaxial":
        return 0.95
      if candidate_mode == "planar":
        return -1.2
    if hint == "support":
      if candidate_mode == "coaxial":
        return 0.45
      if candidate_mode == "planar":
        return -0.85
    if hint == "coaxial":
      if candidate_mode == "coaxial":
        return -0.65
      if candidate_mode == "planar":
        return 0.25
    if hint == "fasten":
      if candidate_mode == "coaxial":
        return -0.2
      if candidate_mode == "planar":
        return -0.7
    return -0.15

  def _infer_relation_hint(
      self,
      parent: PartInstance,
      child: PartInstance,
  ) -> str:
    parent_role = self._role_hint_from_instance(parent)
    child_role = self._role_hint_from_instance(child)
    role_set = {parent_role, child_role}
    if "bolt" in role_set:
      return "fasten"
    if "gear" in role_set and "shaft" in role_set:
      return "coaxial"
    if "base" in role_set and ("shaft" in role_set or "gear" in role_set):
      return "coaxial"
    if self._has_hole_pin_pair(parent, child):
      return "coaxial"
    if self._has_planar_support_pair(parent, child):
      return "planar"
    return "link"

  def _has_hole_pin_pair(
      self,
      parent: PartInstance,
      child: PartInstance,
  ) -> bool:
    parent_has_hole = any(
        socket.kind in {"hole", "threaded_hole"}
        for socket in parent.sockets.values()
    )
    parent_has_pin = any(
        socket.kind == "pin" for socket in parent.sockets.values()
    )
    child_has_hole = any(
        socket.kind in {"hole", "threaded_hole"}
        for socket in child.sockets.values()
    )
    child_has_pin = any(
        socket.kind == "pin" for socket in child.sockets.values()
    )
    return (parent_has_hole and child_has_pin) or (parent_has_pin and child_has_hole)

  def _has_planar_support_pair(
      self,
      parent: PartInstance,
      child: PartInstance,
  ) -> bool:
    parent_has_plane = any(
        socket.kind in {"plane", "flange_plane"}
        for socket in parent.sockets.values()
    )
    child_has_plane = any(
        socket.kind in {"plane", "flange_plane"}
        for socket in child.sockets.values()
    )
    return parent_has_plane and child_has_plane

  def _role_hint_from_instance(self, instance: PartInstance) -> str:
    raw = (
        instance.metadata.get("benchmark_role_hint")
        or instance.metadata.get("retrieval_role_hint")
        or ""
    )
    text = str(raw).strip().lower()
    if text:
      return text
    return self._role_hint_from_name(instance.instance_id)

  def _role_hint_from_name(self, name: str) -> str:
    lowered = str(name).lower()
    if any(
        token in lowered
        for token in {"base", "housing", "frame", "bracket", "mount", "chassis"}
    ):
      return "base"
    if any(
        token in lowered
        for token in {"shaft", "axle", "arbor", "spindle", "rotor"}
    ):
      return "shaft"
    if any(token in lowered for token in {"gear", "pinion", "flywheel", "wheel"}):
      return "gear"
    if any(token in lowered for token in {"bolt", "screw", "fastener"}):
      return "bolt"
    if any(token in lowered for token in {"collar", "bushing", "spacer"}):
      return "collar"
    if any(token in lowered for token in {"bearing", "locator"}):
      return "bearing"
    return ""

  def _instance_preferred_neighbors(self, instance: PartInstance) -> list[str]:
    raw = instance.metadata.get("preferred_neighbors", [])
    if not isinstance(raw, list):
      return []
    return [
        str(item).strip()
        for item in raw
        if isinstance(item, str) and str(item).strip()
    ]

  def _preferred_neighbor_bonus(
      self,
      parent: PartInstance,
      child: PartInstance,
  ) -> float:
    parent_neighbors = {
        str(item)
        for item in parent.metadata.get("preferred_neighbors", []) or []
        if isinstance(item, str)
    }
    child_neighbors = {
        str(item)
        for item in child.metadata.get("preferred_neighbors", []) or []
        if isinstance(item, str)
    }
    mutual = (
        child.instance_id in parent_neighbors
        and parent.instance_id in child_neighbors
    )
    if mutual:
      return 0.9
    if child.instance_id in parent_neighbors or parent.instance_id in child_neighbors:
      return 0.55
    return 0.0

  def _radius_fit_bonus(self, hole_like: Socket, pin_like: Socket) -> float:
    if hole_like.radius is None or pin_like.radius is None:
      return 0.0
    hole_radius = float(hole_like.radius)
    pin_radius = float(pin_like.radius)
    scale = max(hole_radius, pin_radius, 1.0)
    diff = hole_radius - pin_radius
    mismatch_ratio = abs(diff) / scale
    bonus = 1.2 - 1.8 * mismatch_ratio
    if diff < 0.0:
      bonus -= 1.4 + 2.0 * min(1.0, abs(diff) / scale)
    return bonus

  def _threaded_receiver_penalty(
      self,
      receiver_socket: Socket,
      inserted_part_id: str,
  ) -> float:
    if receiver_socket.kind != "threaded_hole":
      return 0.0
    if self._is_fastener_like(inserted_part_id):
      return -0.15
    return -1.25

  def _is_base_like(self, part_id: str) -> bool:
    lowered = part_id.lower()
    return any(token in lowered for token in {"base", "housing", "frame"})

  def _is_shaft_like(self, part_id: str) -> bool:
    lowered = part_id.lower()
    return any(
        token in lowered
        for token in {"shaft", "axle", "arbor", "spindle", "pinion_shaft"}
    )

  def _is_gear_like(self, part_id: str) -> bool:
    lowered = part_id.lower()
    return any(token in lowered for token in {"gear", "pinion", "sprocket"})

  def _is_collar_like(self, part_id: str) -> bool:
    lowered = part_id.lower()
    return any(token in lowered for token in {"collar", "bushing", "spacer"})

  def _is_fastener_like(self, part_id: str) -> bool:
    lowered = part_id.lower()
    return any(
        token in lowered
        for token in {"bolt", "screw", "fastener", "stud", "thread"}
    )

  def _find_socket(
      self, instance: PartInstance, accepted_kinds: set[str]
  ) -> Optional[Socket]:
    sockets = self._sorted_sockets(instance, accepted_kinds)
    if not sockets:
      return None
    return sockets[0]


class LLMPlannerError(RuntimeError):
  """Raised when remote LLM planner call or parse fails."""


class LLMNeuralPlanner:
  """Planner that calls real LLM APIs and expects strict JSON output."""

  def __init__(
      self,
      provider: str = "openai",
      model: Optional[str] = None,
      api_key: Optional[str] = None,
      base_url: Optional[str] = None,
      timeout_seconds: int = 60,
      temperature: float = 0.1,
      default_clearance: float = 1.0,
      fallback_to_rules: bool = True,
      max_prompt_sockets_per_part: int = 18,
      max_prompt_total_sockets: int = 56,
      prompt_kind_budgets: Optional[dict[str, int]] = None,
      adaptive_temperature: bool = True,
      temperature_step: float = 0.12,
      temperature_max: float = 0.75,
      llm_retry_on_invalid: int = 3,
      use_few_shot: bool = True,
      few_shot_example_count: int = 2,
      reasoning_mode: str = "brief",
      llm_thinking_mode: str = "disabled",
      planning_mode: str = "topology_graph",
      model_input_protocol: str = "legacy",
  ):
    provider = provider.lower().strip()
    if provider not in {"openai", "anthropic", "ollama", "deepseek"}:
      raise ValueError(
          "provider must be one of: openai, anthropic, ollama, deepseek"
      )
    self.provider = provider
    if model is not None:
      self.model = model
    elif provider == "openai":
      self.model = "gpt-4o"
    elif provider == "anthropic":
      self.model = "claude-3-5-sonnet-latest"
    elif provider == "deepseek":
      self.model = "deepseek-v4-pro"
    else:
      self.model = "qwen2.5:14b"
    self.api_key = api_key
    self.base_url = base_url
    self.timeout_seconds = int(timeout_seconds)
    self.temperature = float(temperature)
    self.fallback_to_rules = bool(fallback_to_rules)
    self.adaptive_temperature = bool(adaptive_temperature)
    self.temperature_step = max(0.0, float(temperature_step))
    self.temperature_max = max(float(temperature), float(temperature_max))
    self.llm_retry_on_invalid = max(1, int(llm_retry_on_invalid))
    self.use_few_shot = bool(use_few_shot)
    self.few_shot_example_count = max(0, int(few_shot_example_count))
    self.reasoning_mode = str(reasoning_mode).strip().lower()
    if self.reasoning_mode not in {"none", "brief"}:
      raise ValueError("reasoning_mode must be one of: none, brief")
    self.llm_thinking_mode = str(llm_thinking_mode).strip().lower()
    if self.llm_thinking_mode not in {"disabled", "two_stage"}:
      raise ValueError("llm_thinking_mode must be one of: disabled, two_stage")
    self.planning_mode = str(planning_mode).strip().lower()
    if self.planning_mode not in {"topology_graph", "constraint_graph"}:
      raise ValueError(
          "planning_mode must be one of: topology_graph, constraint_graph"
      )
    self.model_input_protocol = str(model_input_protocol or "legacy").strip().lower()
    if self.model_input_protocol not in {"legacy", "benchmark_v2"}:
      raise ValueError(
          "model_input_protocol must be one of: legacy, benchmark_v2"
      )
    if self.model_input_protocol == "benchmark_v2":
      if self.planning_mode != "topology_graph":
        raise ValueError(
            "benchmark_v2 currently requires planning_mode='topology_graph'"
        )
      if self.fallback_to_rules:
        raise ValueError(
            "benchmark_v2 requires fallback_to_rules=False until the rule "
            "planner has an audited benchmark input boundary"
        )
    self.max_prompt_sockets_per_part = max(4, int(max_prompt_sockets_per_part))
    self.max_prompt_total_sockets = max(12, int(max_prompt_total_sockets))
    self.prompt_kind_budgets = (
        {
            "hole": 6,
            "threaded_hole": 4,
            "pin": 5,
            "flange_plane": 3,
            "guide_slot": 3,
            "plane": 4,
            "axis": 2,
        }
        if prompt_kind_budgets is None
        else {k: max(0, int(v)) for k, v in prompt_kind_budgets.items()}
    )
    self.rule_fallback = RuleBasedNeuralPlanner(
        default_clearance=default_clearance
    )
    self._last_feedback_signature: Optional[str] = None
    self._feedback_repeat_count: int = 0
    self._last_canonical_mate_graph: Optional[CanonicalMateGraph] = None

  def plan(
      self,
      instruction: str,
      catalog: dict[str, PartTemplate],
      feedback: Optional[PlannerFeedback] = None,
  ) -> ConstraintGraph:
    ordered_parts = self.rule_fallback._rank_parts_from_instruction(  # pylint: disable=protected-access
        instruction, catalog
    )
    instances = {
        part_name: catalog[part_name].instantiate(part_name)
        for part_name in ordered_parts
    }
    default_anchor = self.rule_fallback._choose_anchor(ordered_parts)  # pylint: disable=protected-access

    planner_notes = [
      f"instruction={instruction}",
      f"ordered_parts={ordered_parts}",
      f"default_anchor={default_anchor}",
      f"planner=llm:{self.provider}:{self.model}",
      f"planning_mode={self.planning_mode}",
    ]

    feedback_messages = [] if feedback is None else list(feedback.messages)
    feedback_repeat_count = self._update_feedback_repeat(feedback_messages)
    blocked_part_pairs = []
    if feedback is not None:
      blocked_part_pairs = sorted(
          [
              sorted([collision.part_a, collision.part_b])
              for collision in feedback.collisions
          ]
      )

    if (
        feedback is not None
        and self.fallback_to_rules
        and self._should_use_contact_guided_fallback(
            feedback_messages=feedback_messages,
            instances=instances,
        )
    ):
      fallback_graph = self.rule_fallback.plan(
          instruction,
          catalog,
          feedback=None,
      )
      notes = list(fallback_graph.metadata.get("planner_notes", []))
      notes.extend([f"feedback={m}" for m in feedback_messages])
      notes.append("fallback=contact_guided_rule")
      fallback_graph.metadata["planner_notes"] = notes
      fallback_graph.metadata["canonical_mate_graph"] = self._fallback_canonical_graph(
          fallback_graph
      )
      return fallback_graph

    response: dict[str, Any] = {}
    anchor = default_anchor
    constraints: list[Constraint] = []
    warnings: list[str] = []
    llm_errors: list[str] = []
    self._last_canonical_mate_graph = None

    for local_retry in range(self.llm_retry_on_invalid):
      llm_temperature = self._compute_llm_temperature(
          feedback_messages=feedback_messages,
          local_retry=local_retry,
          feedback_repeat_count=feedback_repeat_count,
      )
      planner_notes.append(
          f"llm_try={local_retry + 1}/{self.llm_retry_on_invalid};"
          f"temperature={llm_temperature:.3f}"
      )
      try:
        response = self._call_llm(
            instruction=instruction,
            instances=instances,
            default_anchor=default_anchor,
            feedback_messages=feedback_messages,
            blocked_part_pairs=blocked_part_pairs,
            llm_temperature=llm_temperature,
            local_retry=local_retry,
            feedback_repeat_count=feedback_repeat_count,
            prior_errors=llm_errors[-2:],
        )
        anchor, constraints, warnings = self._parse_llm_plan_json(
            raw=response,
            instances=instances,
            default_anchor=default_anchor,
            feedback=feedback,
        )
        self._validate_constraint_graph_connectivity(
            anchor=anchor, constraints=constraints, instances=instances
        )
        break
      except Exception as exc:  # pylint: disable=broad-except
        llm_errors.append(str(exc))
        if local_retry + 1 >= self.llm_retry_on_invalid:
          if not self.fallback_to_rules:
            raise LLMPlannerError(str(exc)) from exc
          fallback_graph = self.rule_fallback.plan(instruction, catalog, feedback)
          notes = list(fallback_graph.metadata.get("planner_notes", []))
          notes.extend([f"llm_error={err}" for err in llm_errors])
          notes.append("fallback=rule_based")
          fallback_graph.metadata["planner_notes"] = notes
          fallback_graph.metadata["canonical_mate_graph"] = self._fallback_canonical_graph(
              fallback_graph
          )
          return fallback_graph

    if feedback:
      planner_notes.extend([f"feedback={m}" for m in feedback.messages])
    planner_notes.extend(warnings)

    metadata = {"planner_notes": planner_notes, "llm_raw": response}
    if self._last_canonical_mate_graph is not None:
      metadata["canonical_mate_graph"] = self._last_canonical_mate_graph.to_dict()
    return ConstraintGraph(
        instances=instances,
        constraints=constraints,
        anchor=anchor,
        metadata=metadata,
    )

  def serialize_model_parts(
      self,
      instances: dict[str, PartInstance],
  ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Serialize the exact part view allowed by the selected protocol."""

    if self.model_input_protocol == "legacy":
      return self._serialize_parts(instances)
    return self._serialize_parts_benchmark_v2(instances)

  def _serialize_parts_benchmark_v2(
      self,
      instances: dict[str, PartInstance],
  ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload: dict[str, dict[str, Any]] = {}
    for name in sorted(instances):
      instance = instances[name]
      self._validate_opaque_part_key(name)
      if instance.instance_id != name:
        raise ValueError("benchmark_v2 instance_id must match its opaque catalog key")
      if instance.template_name != name:
        raise ValueError(
            "benchmark_v2 template_name must equal its opaque runtime part key"
        )
      if str(instance.metadata.get("model_input_protocol") or "") != "benchmark_v2":
        raise ValueError(
            f"benchmark_v2 part {name} is missing model_input_protocol metadata"
        )

      kind_counts = self._socket_kind_histogram(list(instance.sockets.values()))
      surface_counts: dict[str, int] = {}
      role_counts: dict[str, int] = {}
      radius_ratios: list[float] = []
      for socket in instance.sockets.values():
        metadata = socket.metadata if isinstance(socket.metadata, dict) else {}
        if metadata.get("model_input_protocol") != "benchmark_v2":
          continue
        view = metadata.get("benchmark_v2_model_view")
        expected_hash = metadata.get("benchmark_v2_model_view_sha256")
        if not isinstance(view, dict) or not isinstance(expected_hash, str):
          raise ValueError(
              f"benchmark_v2 learned socket on {name} lacks sanitized view/hash"
          )
        validate_benchmark_v2_model_view(
            view,
            expected_sha256=expected_hash,
        )
        surface = str(view.get("surface_type") or "other")
        surface_counts[surface] = surface_counts.get(surface, 0) + 1
        topology = view.get("topology", {})
        role = str(topology.get("composite_role") or "generic_interface")
        role_counts[role] = role_counts.get(role, 0) + 1
        geometry = view.get("geometry", {})
        radius_ratio = geometry.get("radius_ratio")
        if isinstance(radius_ratio, (int, float)):
          radius_ratios.append(round(float(radius_ratio), 8))

      payload[name] = {
          "template_name": name,
          "socket_count": len(instance.sockets),
          "socket_kinds": dict(sorted(kind_counts.items())),
          "invariant_interface_summary": {
              "surface_counts": dict(sorted(surface_counts.items())),
              "role_counts": dict(sorted(role_counts.items())),
              "radius_ratios": sorted(radius_ratios)[:16],
          },
      }
    return payload, []

  def build_model_payload(
      self,
      *,
      instruction: str,
      instances: dict[str, PartInstance],
      default_anchor: str,
      feedback_messages: list[str],
      blocked_part_pairs: list[list[str]],
      llm_temperature: float,
      local_retry: int,
      feedback_repeat_count: int,
      prior_errors: list[str],
  ) -> dict[str, Any]:
    """Build the externally visible planner payload for benchmark-v2."""

    if self.model_input_protocol != "benchmark_v2":
      raise ValueError(
          "build_model_payload is the formal benchmark_v2 boundary; legacy "
          "calls retain the historical _call_llm payload"
      )
    parts, _ = self.serialize_model_parts(instances)
    self._validate_opaque_part_key(default_anchor)
    if default_anchor not in instances:
      raise ValueError("benchmark_v2 default_anchor is not in the part catalog")
    safe_blocked_pairs: list[list[str]] = []
    for pair in blocked_part_pairs:
      if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError("benchmark_v2 blocked_part_pairs must contain pairs")
      first, second = str(pair[0]), str(pair[1])
      self._validate_opaque_part_key(first)
      self._validate_opaque_part_key(second)
      if first not in instances or second not in instances:
        raise ValueError("benchmark_v2 blocked pair references an unknown part")
      safe_blocked_pairs.append([first, second])

    failure_signals: list[str] = []
    combined_feedback = "\n".join(str(item).lower() for item in feedback_messages)
    for token, signal in (
        ("collision", "collision"),
        ("solve_error", "solve_error"),
        ("validation_error", "validation_error"),
        ("unplaced_part", "unplaced_part"),
    ):
      if token in combined_feedback:
        failure_signals.append(signal)

    return {
        "model_input_protocol": "benchmark_v2",
        "instruction": self._sanitize_benchmark_v2_instruction(
            instruction,
            instances,
        ),
        "default_anchor": default_anchor,
        "planning_mode": "topology_graph",
        "parts": parts,
        "must_cover_parts": sorted(instances),
        "focus_parts": sorted(instances),
        "failure_signals": failure_signals,
        "blocked_part_pairs": sorted(safe_blocked_pairs),
        "prior_error_count": len(prior_errors),
        "replan_state": {
            "local_retry": int(local_retry),
            "feedback_repeat_count": int(feedback_repeat_count),
            "temperature": round(float(llm_temperature), 3),
        },
        "reasoning_mode": self.reasoning_mode,
        "planning_rules": [
            "Every opaque part in must_cover_parts must appear in a connection edge.",
            "Output one connected part graph rooted at anchor.",
            "Do not output socket names, coordinates, axes, normals, or low-level constraints.",
            "Use coaxial for compatible cylindrical interfaces.",
            "Use planar or support for compatible planar interfaces.",
            "Use fasten only when invariant interface evidence supports a threaded relation.",
            "Avoid a blocked pair relation after collision feedback when another connected graph exists.",
        ],
        "relation_types": {
            "coaxial": "compatible cylindrical-interface relation",
            "planar": "compatible planar-interface relation",
            "fasten": "thread-supported fastening relation",
            "support": "planar support relation",
            "link": "generic direct connection",
        },
        "output_schema_hint": {
            "anchor": "opaque_part_name",
            "connections": [
                {
                    "part_a": "opaque_part_name",
                    "part_b": "opaque_part_name",
                    "relation_type": "coaxial|planar|fasten|support|link",
                    "order_hint": "integer_optional",
                }
            ],
            "plan_summary": "string_optional_short",
        },
        "reasoning_checklist": [
            "check every opaque part appears in at least one edge",
            "check graph is connected from anchor",
            "check no coordinate, direction, path, or source-name assumption is used",
            "check blocked collision relations are not repeated without necessity",
        ],
    }

  @staticmethod
  def _validate_opaque_part_key(name: str) -> None:
    if re.fullmatch(r"opaque_part_[0-9]{3,}", str(name)) is None:
      raise ValueError(
          "benchmark_v2 requires runtime part keys matching "
          "^opaque_part_[0-9]{3,}$"
      )

  def _sanitize_benchmark_v2_instruction(
      self,
      instruction: str,
      instances: dict[str, PartInstance],
  ) -> str:
    text = str(instruction or "")
    for opaque_name, instance in instances.items():
      sensitive_tokens: set[str] = set()
      for key in (
          "original_name",
          "source_name",
          "source_hint",
          "body_uuid",
          "step_path",
      ):
        raw = instance.metadata.get(key)
        if isinstance(raw, str) and raw.strip():
          sensitive_tokens.add(raw.strip())
          if key == "step_path":
            sensitive_tokens.add(Path(raw).stem)
      preferred = instance.metadata.get("preferred_neighbors")
      if isinstance(preferred, list):
        sensitive_tokens.update(
            str(item).strip()
            for item in preferred
            if isinstance(item, str) and item.strip()
        )
      for token in sorted(sensitive_tokens, key=len, reverse=True):
        text = re.sub(re.escape(token), opaque_name, text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?i)\b[a-z]:\\+(?:[^\s,;]+)",
        "[path]",
        text,
    )
    text = re.sub(
        r"(?<!\w)/(?:[^\s,;]+/)*[^\s,;]+",
        "[path]",
        text,
    )
    text = re.sub(
        r"(?i)\bassembly[_-]?[a-z0-9]+\b",
        "assembly",
        text,
    )
    return " ".join(text.split())

  def _call_llm(
      self,
      instruction: str,
      instances: dict[str, PartInstance],
      default_anchor: str,
      feedback_messages: list[str],
      blocked_part_pairs: list[list[str]],
      llm_temperature: float,
      local_retry: int,
      feedback_repeat_count: int,
      prior_errors: list[str],
  ) -> dict[str, Any]:
    api_key = self._resolve_api_key()
    part_payload, compaction_notes = self.serialize_model_parts(instances)
    must_cover_parts = sorted(instances.keys())
    focus_parts = self._focus_parts_from_feedback(feedback_messages, instances)

    user_payload = {
        "instruction": instruction,
        "default_anchor": default_anchor,
        "planning_mode": self.planning_mode,
        "parts": part_payload,
        "must_cover_parts": must_cover_parts,
        "focus_parts": focus_parts,
        "feedback_messages": feedback_messages,
        "blocked_part_pairs": blocked_part_pairs,
        "prior_errors": prior_errors,
        "replan_state": {
            "local_retry": local_retry,
            "feedback_repeat_count": feedback_repeat_count,
            "temperature": round(llm_temperature, 3),
        },
        "reasoning_mode": self.reasoning_mode,
    }
    if self.planning_mode == "topology_graph":
      user_payload["planning_rules"] = [
          "Every part in must_cover_parts must appear in at least one connection edge.",
          "Output a connected part graph rooted at anchor.",
          "Do not output socket names or low-level geometric constraints.",
          "Use relation_type=coaxial for shaft-hole or pin-hole style assembly intent.",
          "Use relation_type=planar for face-on-face support or flange mating intent.",
          "Use relation_type=fasten only when a threaded interface or fastener-like part is clearly involved.",
          "Honor preferred_neighbors when choosing which parts should connect directly.",
          "Prefer gears and collars to connect through shafts/arbors rather than directly into base holes.",
          "If feedback indicates collision or failed grounding, choose a different connection graph.",
          "If a blocked_part_pair previously collided, avoid repeating the same face-on-face planar relation for that pair when a cylindrical chain exists.",
      ]
      user_payload["relation_types"] = {
          "coaxial": "hole-pin or shaft-axis style alignment",
          "planar": "planar or flange support/mating",
          "fasten": "threaded or screw-fastened relation",
          "support": "resting/support relation when planar support is primary",
          "link": "generic direct connection when unsure",
      }
      user_payload["output_schema_hint"] = {
          "anchor": "string",
          "role_assignments": {"base|shaft|gear|bolt|collar|bearing": "part_name"},
          "stack_groups": [
              {
                  "group_id": "string_optional",
                  "parts": ["ordered_part_names"],
                  "anchor_part": "part_name_optional",
                  "relation_type": "coaxial|fasten_optional",
              }
          ],
          "connections": [
              {
                  "part_a": "part_name",
                  "part_b": "part_name",
                  "relation_type": "coaxial|planar|fasten|support|link",
                  "label": "string_optional",
                  "stack_group": "string_optional",
                  "order_hint": "integer_optional",
                  "source_frame_hint": "entry_plus|entry_minus|seat_plus|seat_minus|through_plus|through_minus|through_stop_plus|through_stop_minus|tip_plus|tip_minus|stop_plus|stop_minus|shoulder_plus|shoulder_minus|support|primary_optional",
                  "target_frame_hint": "entry_plus|entry_minus|seat_plus|seat_minus|through_plus|through_minus|through_stop_plus|through_stop_minus|tip_plus|tip_minus|stop_plus|stop_minus|shoulder_plus|shoulder_minus|support|primary_optional",
                  "source_stop_hint": "seat_plus|seat_minus|through_plus|through_minus|through_stop_plus|through_stop_minus|hole_depth_stop_plus|hole_depth_stop_minus|stop_plus|stop_minus|shoulder_plus|shoulder_minus_optional",
                  "target_stop_hint": "seat_plus|seat_minus|through_plus|through_minus|through_stop_plus|through_stop_minus|hole_depth_stop_plus|hole_depth_stop_minus|stop_plus|stop_minus|shoulder_plus|shoulder_minus_optional",
                  "stack_phase": "entry|seat|through_stop_optional",
                  "alignment_mode": "mate|insert|stack|support|through_stop_optional",
              }
          ],
          "plan_summary": "string_optional_short",
      }
      user_payload["reasoning_checklist"] = [
          "check every part appears in at least one connection edge",
          "check graph is connected from anchor",
          "check preferred_neighbors are respected when possible",
          "check graph avoids repeating failed pairings from feedback",
      ]
    else:
      user_payload["planning_rules"] = [
          "Every part in must_cover_parts must appear in at least one constraint edge.",
          "Output a connected assembly graph rooted at anchor.",
          "Use concentric for hole-pin, hole-axis, pin-axis, axis-axis alignment.",
          "Use coincident only for plane/flange_plane/guide_slot mating.",
          "Never use coincident between hole/pin/axis sockets.",
          "Do not reuse the same hole or threaded_hole socket across multiple different mate links.",
          "Prefer plain holes over threaded_hole unless the mating part is clearly a fastener.",
          "Prefer cylindrical pairs with similar radii; avoid very loose or impossible fits.",
          "Honor preferred_neighbors when choosing which parts should connect directly.",
          "Prefer mating gear-like parts onto shaft-like pins/axes, not directly into base holes.",
          "Use distance only for spacing or collision clearance.",
      ]
      user_payload["constraint_legality"] = {
          "concentric_allowed_kinds": [
              "hole", "threaded_hole", "pin", "axis", "guide_slot"
          ],
          "coincident_allowed_kinds": [
              "plane", "flange_plane", "guide_slot"
          ],
          "distance_allowed_reference_kinds": [
              "plane", "flange_plane", "axis", "guide_slot"
          ],
      }
      user_payload["output_schema_hint"] = {
          "anchor": "string",
          "constraints": [
              {
                  "type": "concentric|coincident|distance",
                  "part_a": "part_name",
                  "socket_a": "socket_name",
                  "part_b": "part_name",
                  "socket_b": "socket_name",
                  "value": "number|null",
                  "label": "string_optional",
              }
          ],
          "plan_summary": "string_optional_short",
      }
      user_payload["reasoning_checklist"] = [
          "check every part appears in at least one edge",
          "check graph is connected from anchor",
          "check socket-kind legality for each constraint type",
          "check collisions from previous feedback are avoided",
      ]
    if compaction_notes:
      user_payload["prompt_compaction_notes"] = compaction_notes
    if self.use_few_shot and self.few_shot_example_count > 0:
      user_payload["few_shot_examples"] = self._few_shot_examples_for_prompt()[
          : self.few_shot_example_count
      ]
    if self.model_input_protocol == "benchmark_v2":
      # Overwrite the complete legacy prompt construction before any provider
      # call.  No legacy names, paths, preferred-neighbour hints, raw feedback,
      # socket frames, or few-shot semantic names cross this boundary.
      user_payload = self.build_model_payload(
          instruction=instruction,
          instances=instances,
          default_anchor=default_anchor,
          feedback_messages=feedback_messages,
          blocked_part_pairs=blocked_part_pairs,
          llm_temperature=llm_temperature,
          local_retry=local_retry,
          feedback_repeat_count=feedback_repeat_count,
          prior_errors=prior_errors,
      )
    system_prompt = (
        "You are a CAD neuro-symbolic planner. "
        "Return ONLY valid JSON that follows the required schema. "
        "Think step by step internally, but do not expose long reasoning traces. "
        "Plan high-level mechanical assembly structure first, then leave low-level geometric grounding to a deterministic symbolic layer. "
        "Never leave a part unconnected. "
        "If feedback indicates previous failure, produce a different and corrected graph. "
        "If plan_summary is returned, keep it to one short sentence."
    )
    schema = (
        self._topology_graph_schema()
        if self.planning_mode == "topology_graph"
        else self._constraint_graph_schema()
    )
    thinking_summary: Optional[dict[str, Any]] = None
    if self.provider == "deepseek" and self.llm_thinking_mode == "two_stage":
      thinking_summary = self._call_deepseek_thinking_summary(
          system_prompt=system_prompt,
          user_payload=user_payload,
          api_key=api_key,
          llm_temperature=llm_temperature,
      )
      user_payload["thinking_summary"] = thinking_summary
      user_payload["planning_rules"].append(
          "Use thinking_summary only as private planning guidance; final output must still follow the required JSON schema exactly."
      )
    if self.provider in {"openai", "ollama", "deepseek"}:
      response = self._call_openai_json(
          system_prompt=system_prompt,
          user_payload=user_payload,
          schema=schema,
          api_key=api_key,
          llm_temperature=llm_temperature,
          deepseek_thinking="disabled",
      )
      if thinking_summary is not None:
        response["_thinking_summary"] = thinking_summary
      return response
    return self._call_anthropic_json(
        system_prompt=system_prompt,
        user_payload=user_payload,
        schema=schema,
        api_key=api_key,
        llm_temperature=llm_temperature,
    )

  def _resolve_api_key(self) -> str:
    if self.api_key:
      raise LLMPlannerError(
          "Direct API-key arguments are disabled because command lines and run "
          "manifests can expose credentials. Set the provider environment "
          "variable instead."
      )
    if self.provider == "openai":
      key = os.getenv("OPENAI_API_KEY")
    elif self.provider == "anthropic":
      key = os.getenv("ANTHROPIC_API_KEY")
    elif self.provider == "deepseek":
      key = os.getenv("DEEPSEEK_API_KEY")
    else:
      key = os.getenv("OLLAMA_API_KEY") or "ollama"
    if key:
      return key
    raise LLMPlannerError(
        f"Missing API key for provider={self.provider}. "
        "Set OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OLLAMA_API_KEY."
    )

  def _update_feedback_repeat(self, feedback_messages: list[str]) -> int:
    if not feedback_messages:
      self._last_feedback_signature = None
      self._feedback_repeat_count = 0
      return 0
    signature = "||".join(sorted(str(message) for message in feedback_messages))
    if signature == self._last_feedback_signature:
      self._feedback_repeat_count += 1
    else:
      self._last_feedback_signature = signature
      self._feedback_repeat_count = 1
    return self._feedback_repeat_count

  def _compute_llm_temperature(
      self,
      feedback_messages: list[str],
      local_retry: int,
      feedback_repeat_count: int,
  ) -> float:
    temperature = self.temperature
    if self.adaptive_temperature:
      temperature += self.temperature_step * float(local_retry)
      if feedback_repeat_count > 1:
        temperature += self.temperature_step * float(feedback_repeat_count - 1)
      if any("unplaced_part=" in msg for msg in feedback_messages):
        temperature += self.temperature_step * 0.65
      if any("solve_error=" in msg for msg in feedback_messages):
        temperature += self.temperature_step * 0.5
      if any("validation_error=" in msg for msg in feedback_messages):
        temperature += self.temperature_step * 0.4
    return float(min(max(0.0, temperature), self.temperature_max))

  def _focus_parts_from_feedback(
      self,
      feedback_messages: list[str],
      instances: dict[str, PartInstance],
  ) -> list[str]:
    part_names = sorted(instances.keys())
    focused: list[str] = []
    for message in feedback_messages:
      text = str(message)
      if "collision(" in text and ")" in text:
        pair_text = text.split("collision(", 1)[1].split(")", 1)[0]
        pair_items = [item.strip() for item in pair_text.split(",")]
        for item in pair_items[:2]:
          if item in instances and item not in focused:
            focused.append(item)
      if "unplaced_part=" in text:
        token = text.split("unplaced_part=", 1)[1].split(";", 1)[0].strip()
        if token and token in instances and token not in focused:
          focused.append(token)
    if focused:
      return focused
    return part_names

  def _should_use_contact_guided_fallback(
      self,
      feedback_messages: list[str],
      instances: dict[str, PartInstance],
  ) -> bool:
    if not feedback_messages:
      return False
    has_hard_feedback = any(
        ("collision(" in message)
        or ("solve_error=" in message)
        or ("validation_error=" in message)
        for message in feedback_messages
    )
    if not has_hard_feedback:
      return False
    covered = 0
    for instance in instances.values():
      raw = instance.metadata.get("preferred_neighbors", [])
      if isinstance(raw, list) and raw:
        covered += 1
    return covered >= max(2, len(instances) // 2)

  def _call_openai_json(
      self,
      system_prompt: str,
      user_payload: dict[str, Any],
      schema: dict[str, Any],
      api_key: str,
      llm_temperature: float,
      deepseek_thinking: str = "disabled",
  ) -> dict[str, Any]:
    if self.provider == "openai":
      default_url = "https://api.openai.com/v1"
      response_format: dict[str, Any] = {
          "type": "json_schema",
          "json_schema": {
              "name": "constraint_graph",
              "strict": True,
              "schema": schema,
          },
      }
    elif self.provider == "deepseek":
      default_url = "https://api.deepseek.com/v1"
      response_format = {"type": "json_object"}
    else:
      # Ollama OpenAI-compatible endpoint.
      default_url = "http://localhost:11434/v1"
      response_format = {"type": "json_object"}
    url = (self.base_url or default_url).rstrip("/")
    url += "/chat/completions"

    payload = {
        "model": self.model,
        "temperature": llm_temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        "response_format": response_format,
    }
    if self.provider == "deepseek":
      # Planner calls need machine-readable JSON only; disabling thinking mode
      # prevents hidden/reasoning text from leaking before the JSON object.
      payload["thinking"] = {"type": deepseek_thinking}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = self._http_post_json_with_openai_sdk(
        url=url, payload=payload, api_key=api_key
    )
    if data is None:
      data = self._http_post_json(url, payload, headers)
    try:
      content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
      raise LLMPlannerError(
          f"Unexpected OpenAI response structure: {data}"
      ) from exc

    if isinstance(content, list):
      text = "".join(
          c.get("text", "") if isinstance(c, dict) else str(c)
          for c in content
      )
    else:
      text = str(content)
    return self._decode_json_text(text)

  def _call_deepseek_thinking_summary(
      self,
      *,
      system_prompt: str,
      user_payload: dict[str, Any],
      api_key: str,
      llm_temperature: float,
  ) -> dict[str, Any]:
    summary_payload = dict(user_payload)
    summary_payload["task"] = (
        "Reason about the assembly plan before final JSON emission. "
        "Do not choose raw coordinates."
    )
    summary_schema = {
        "type": "object",
        "properties": {
            "anchor": {"type": "string"},
            "connection_order": {
                "type": "array",
                "items": {"type": "string"},
            },
            "critical_mates": {
                "type": "array",
                "items": {"type": "string"},
            },
            "seat_stop_hypotheses": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risk_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": True,
    }
    thinking_system_prompt = (
        system_prompt
        + " This is a first-pass reasoning call. Return ONLY a compact JSON "
        "summary with anchor, connection_order, critical_mates, "
        "seat_stop_hypotheses, and risk_notes. The next call will emit the "
        "final machine-readable graph."
    )
    try:
      return self._call_openai_json(
          system_prompt=thinking_system_prompt,
          user_payload=summary_payload,
          schema=summary_schema,
          api_key=api_key,
          llm_temperature=min(0.8, max(llm_temperature, 0.2)),
          deepseek_thinking="enabled",
      )
    except Exception as exc:  # pylint: disable=broad-except
      return {"thinking_error": f"{type(exc).__name__}:{exc}"}

  def _http_post_json_with_openai_sdk(
      self, url: str, payload: dict[str, Any], api_key: str
  ) -> Optional[dict[str, Any]]:
    """Try OpenAI python sdk first; fallback to raw HTTP on failure."""
    try:
      from openai import OpenAI
    except ImportError:
      return None

    # We pass the full /chat/completions URL externally.
    base_url = url.rsplit("/chat/completions", 1)[0]
    try:
      client = OpenAI(base_url=base_url, api_key=api_key)
      sdk_payload = dict(payload)
      thinking = sdk_payload.pop("thinking", None)
      if thinking is not None:
        extra_body = dict(sdk_payload.get("extra_body") or {})
        extra_body["thinking"] = thinking
        sdk_payload["extra_body"] = extra_body
      completion = client.chat.completions.create(**sdk_payload)
      dumped = completion.model_dump()
      if isinstance(dumped, dict):
        return dumped
      return None
    except Exception:
      return None

  def _call_anthropic_json(
      self,
      system_prompt: str,
      user_payload: dict[str, Any],
      schema: dict[str, Any],
      api_key: str,
      llm_temperature: float,
  ) -> dict[str, Any]:
    _ = schema  # Anthropic does not currently enforce JSON schema natively.
    url = (self.base_url or "https://api.anthropic.com/v1").rstrip("/")
    url += "/messages"

    user_text = (
        "Return ONLY JSON with keys: anchor and constraints.\n"
        + json.dumps(user_payload)
    )
    payload = {
        "model": self.model,
        "temperature": llm_temperature,
        "max_tokens": 1800,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_text}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    data = self._http_post_json(url, payload, headers)
    blocks = data.get("content", [])
    text = "\n".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    )
    if not text:
      raise LLMPlannerError(f"Empty Anthropic response: {data}")
    return self._decode_json_text(text)

  def _decode_json_text(self, text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
      raise LLMPlannerError("LLM returned empty JSON content.")
    try:
      result = json.loads(text)
      if isinstance(result, dict):
        return result
    except json.JSONDecodeError:
      pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
      raise LLMPlannerError(f"Cannot decode JSON from text: {text[:250]}")
    snippet = text[start : end + 1]
    try:
      decoded = json.loads(snippet)
    except json.JSONDecodeError as exc:
      raise LLMPlannerError(
          f"Failed to parse JSON snippet from LLM output: {snippet[:250]}"
      ) from exc
    if not isinstance(decoded, dict):
      raise LLMPlannerError("Parsed JSON is not an object.")
    return decoded

  def _http_post_json(
      self, url: str, payload: dict[str, Any], headers: dict[str, str]
  ) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = url_request.Request(url=url, method="POST", data=raw, headers=headers)
    try:
      with url_request.urlopen(req, timeout=self.timeout_seconds) as resp:
        body = resp.read().decode("utf-8")
    except url_error.HTTPError as exc:
      details = exc.read().decode("utf-8", errors="replace")
      raise LLMPlannerError(
          f"HTTP {exc.code} calling {url}: {details[:500]}"
      ) from exc
    except url_error.URLError as exc:
      raise LLMPlannerError(f"Network error calling {url}: {exc}") from exc
    try:
      parsed = json.loads(body)
    except json.JSONDecodeError as exc:
      raise LLMPlannerError(
          f"Server did not return JSON: {body[:500]}"
      ) from exc
    if isinstance(parsed, dict) and parsed.get("error"):
      raise LLMPlannerError(f"Provider error: {parsed['error']}")
    if not isinstance(parsed, dict):
      raise LLMPlannerError(f"Unexpected non-dict response: {parsed}")
    return parsed

  def _serialize_parts(
      self, instances: dict[str, PartInstance]
  ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload: dict[str, dict[str, Any]] = {}
    selected_by_part: dict[str, list[tuple[Socket, float]]] = {}
    compaction_notes: list[str] = []

    for name, instance in instances.items():
      selected = self._select_sockets_for_prompt(instance)
      selected_by_part[name] = selected
      raw_count = len(instance.sockets)
      trimmed = max(0, raw_count - len(selected))
      if trimmed > 0:
        compaction_notes.append(
            f"{name}:trimmed_sockets={trimmed}/{raw_count}"
        )

    total_selected = sum(len(items) for items in selected_by_part.values())
    if total_selected > self.max_prompt_total_sockets:
      dropped = self._enforce_total_socket_budget(selected_by_part)
      if dropped > 0:
        compaction_notes.append(
            f"global_socket_trim={dropped};"
            f"max_total={self.max_prompt_total_sockets}"
        )

    for name, instance in instances.items():
      chosen = selected_by_part.get(name, [])
      raw_kind_counts = self._socket_kind_histogram(
          list(instance.sockets.values())
      )
      prompt_kind_counts = self._socket_kind_histogram(
          [socket for socket, _ in chosen]
      )
      base_payload = {
          "template_name": instance.template_name,
          "source_hint": self._instance_source_hint(instance),
          "preferred_neighbors": self._instance_preferred_neighbors(instance),
          "geometry_complexity": self._instance_geometry_complexity(instance),
          "socket_count_raw": len(instance.sockets),
          "socket_count_prompt": len(chosen),
          "socket_kinds_raw": raw_kind_counts,
          "socket_kinds_prompt": prompt_kind_counts,
      }
      if self.planning_mode == "topology_graph":
        payload[name] = {
            **base_payload,
            "interface_summary": self._topology_interface_summary(chosen),
            "blocked_interface_summary": self._blocked_interface_summary(instance),
        }
      else:
        payload[name] = {
            **base_payload,
            "sockets": [
                {
                    "name": socket.name,
                    "kind": socket.kind,
                    "axis": (
                        None
                        if socket.axis is None
                        else [round(float(x), 6) for x in socket.axis]
                    ),
                    "normal": (
                        None
                        if socket.normal is None
                        else [round(float(x), 6) for x in socket.normal]
                    ),
                    "radius": socket.radius,
                    "quality": round(float(score), 4),
                }
                for socket, score in chosen
            ],
        }
    return payload, compaction_notes

  def _instance_source_hint(self, instance: PartInstance) -> str:
    step_path = instance.metadata.get("step_path")
    if isinstance(step_path, str) and step_path.strip():
      return Path(step_path).stem
    return instance.template_name

  def _instance_preferred_neighbors(self, instance: PartInstance) -> list[str]:
    raw = instance.metadata.get("preferred_neighbors", [])
    if not isinstance(raw, list):
      return []
    result: list[str] = []
    for item in raw:
      if isinstance(item, str) and item not in result:
        result.append(item)
    return result[:8]

  def _instance_geometry_complexity(
      self, instance: PartInstance
  ) -> dict[str, Any]:
    face_count = int(instance.metadata.get("face_count", 0) or 0)
    helical_face_count = int(
        instance.metadata.get("helical_face_count", 0) or 0
    )
    spline_face_count = int(
        instance.metadata.get("spline_face_count", 0) or 0
    )
    complexity_score = float(
        instance.metadata.get("complexity_score", 0.0) or 0.0
    )
    risk = "low"
    if (
        face_count >= 180
        or complexity_score >= 240.0
        or helical_face_count > 0
        or spline_face_count > 12
    ):
      risk = "high"
    elif face_count >= 90 or complexity_score >= 120.0:
      risk = "medium"
    return {
        "risk": risk,
        "face_count": face_count,
        "helical_face_count": helical_face_count,
        "spline_face_count": spline_face_count,
    }

  def _topology_interface_summary(
      self,
      chosen: list[tuple[Socket, float]],
  ) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for socket, score in chosen:
      family = socket.kind
      entry = grouped.setdefault(
          family,
          {
              "kind": family,
              "count": 0,
              "top_radii": [],
              "max_quality": 0.0,
              "frame_hints": set(),
          },
      )
      entry["count"] += 1
      entry["max_quality"] = max(float(entry["max_quality"]), float(score))
      if socket.radius is not None:
        entry["top_radii"].append(round(float(socket.radius), 4))
      for hint in socket.metadata.get("frame_variant_names", []) or []:
        if isinstance(hint, str) and hint.strip():
          entry["frame_hints"].add(hint.strip())

    ordered: list[dict[str, Any]] = []
    for family, entry in grouped.items():
      radii = sorted(
          {float(value) for value in entry["top_radii"]},
          reverse=True,
      )[:4]
      ordered.append(
          {
              "kind": family,
              "count": int(entry["count"]),
              "top_radii": [round(value, 4) for value in radii],
              "max_quality": round(float(entry["max_quality"]), 4),
              "frame_hints": sorted(entry["frame_hints"])[:8],
          }
      )
    ordered.sort(
        key=lambda item: (
            -int(item["count"]),
            -float(item["max_quality"]),
            str(item["kind"]),
        )
    )
    return ordered

  def _blocked_interface_summary(
      self,
      instance: PartInstance,
  ) -> dict[str, int]:
    blocked: dict[str, int] = {}
    for socket in instance.sockets.values():
      if not self.rule_fallback._socket_allowed_for_mating(  # pylint: disable=protected-access
          instance_id=instance.instance_id,
          socket=socket,
          relation_hint="link",
          partner_id=None,
      ):
        blocked[socket.kind] = blocked.get(socket.kind, 0) + 1
    return blocked

  def _select_sockets_for_prompt(
      self, instance: PartInstance
  ) -> list[tuple[Socket, float]]:
    sockets = list(instance.sockets.values())
    if self.planning_mode == "topology_graph":
      filtered = [
          socket
          for socket in sockets
          if self.rule_fallback._socket_allowed_for_mating(  # pylint: disable=protected-access
              instance_id=instance.instance_id,
              socket=socket,
              relation_hint="link",
              partner_id=None,
          )
      ]
      if filtered:
        sockets = filtered
    if not sockets:
      return []

    scored = [(socket, self._socket_prompt_score(socket)) for socket in sockets]
    scored.sort(key=lambda item: item[1], reverse=True)
    by_kind: dict[str, list[tuple[Socket, float]]] = {}
    for socket, score in scored:
      by_kind.setdefault(socket.kind, []).append((socket, score))

    selected: list[tuple[Socket, float]] = []
    selected_names: set[str] = set()

    priority = [
        "hole",
        "threaded_hole",
        "pin",
        "flange_plane",
        "guide_slot",
        "plane",
        "axis",
    ]
    for kind in priority:
      budget = self.prompt_kind_budgets.get(kind, 0)
      if budget <= 0:
        continue
      for socket, score in by_kind.get(kind, [])[:budget]:
        if socket.name in selected_names:
          continue
        selected.append((socket, score))
        selected_names.add(socket.name)

    # Ensure at least one cylindrical and one planar anchor if present.
    cyl_kinds = {"hole", "threaded_hole", "pin", "axis"}
    if not any(socket.kind in cyl_kinds for socket, _ in selected):
      for socket, score in scored:
        if socket.kind not in cyl_kinds or socket.name in selected_names:
          continue
        selected.append((socket, score))
        selected_names.add(socket.name)
        break

    plane_kinds = {"plane", "flange_plane"}
    if not any(socket.kind in plane_kinds for socket, _ in selected):
      for socket, score in scored:
        if socket.kind not in plane_kinds or socket.name in selected_names:
          continue
        selected.append((socket, score))
        selected_names.add(socket.name)
        break

    if len(selected) < self.max_prompt_sockets_per_part:
      for socket, score in scored:
        if socket.name in selected_names:
          continue
        selected.append((socket, score))
        selected_names.add(socket.name)
        if len(selected) >= self.max_prompt_sockets_per_part:
          break

    selected.sort(key=lambda item: item[1], reverse=True)
    return selected[: self.max_prompt_sockets_per_part]

  def _socket_prompt_score(self, socket: Socket) -> float:
    score = float(socket.metadata.get("quality", 0.0))
    if socket.kind == "guide_slot":
      score *= 0.15
    if bool(socket.metadata.get("non_mating_surface", False)):
      score -= 5.0
    if socket.metadata.get("mating_enabled") is False:
      score -= 2.5
    if socket.kind == "hole":
      score += 1.2
    elif socket.kind == "threaded_hole":
      score += 0.45
    elif socket.kind == "pin":
      score += 0.9
    elif socket.kind in {"flange_plane", "plane"}:
      score += 0.45
    elif socket.kind == "guide_slot":
      score += 0.35
    if socket.radius is not None:
      score += 0.06 * float(socket.radius)
    area = socket.metadata.get("face_area")
    if isinstance(area, (int, float)):
      score += 0.05 * float(np.log1p(max(float(area), 0.0)))
    return score

  def _socket_kind_histogram(self, sockets: list[Socket]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for socket in sockets:
      histogram[socket.kind] = histogram.get(socket.kind, 0) + 1
    return histogram

  def _enforce_total_socket_budget(
      self, selected_by_part: dict[str, list[tuple[Socket, float]]]
  ) -> int:
    total = sum(len(items) for items in selected_by_part.values())
    dropped = 0
    while total > self.max_prompt_total_sockets:
      # Remove the lowest-score socket from the currently largest list.
      target_name = None
      target_size = -1
      for part_name, items in selected_by_part.items():
        if len(items) <= 1:
          continue
        if len(items) > target_size:
          target_name = part_name
          target_size = len(items)
      if target_name is None:
        break
      selected_by_part[target_name].sort(key=lambda item: item[1], reverse=True)
      selected_by_part[target_name].pop()
      dropped += 1
      total -= 1
    return dropped

  def _parse_llm_plan_json(
      self,
      raw: dict[str, Any],
      instances: dict[str, PartInstance],
      default_anchor: str,
      feedback: Optional[PlannerFeedback],
  ) -> tuple[str, list[Constraint], list[str]]:
    if self.planning_mode == "topology_graph":
      has_connections = any(
          key in raw for key in ("connections", "relations", "edges")
      )
      has_constraints = bool(raw.get("constraints"))
      if has_connections or not has_constraints:
        (
            anchor,
            proposals,
            warnings,
            role_assignments,
            stack_groups,
            plan_summary_text,
        ) = self._parse_topology_graph_json(
            raw=raw,
            instances=instances,
            default_anchor=default_anchor,
        )
        self._last_canonical_mate_graph = CanonicalMateGraph(
            anchor=anchor,
            edges=[
                CanonicalMateEdge(
                    part_a=proposal.part_a,
                    part_b=proposal.part_b,
                    relation_type=proposal.relation_type,
                    label=proposal.label,
                    stack_group=proposal.stack_group,
                    order_hint=proposal.order_hint,
                    source_frame_hint=proposal.source_frame_hint,
                    target_frame_hint=proposal.target_frame_hint,
                    source_stop_hint=proposal.source_stop_hint,
                    target_stop_hint=proposal.target_stop_hint,
                    stack_phase=proposal.stack_phase,
                    alignment_mode=proposal.alignment_mode,
                )
                for proposal in proposals
            ],
            role_assignments=role_assignments,
            stack_groups=stack_groups,
            plan_summary=plan_summary_text,
            metadata={
                "planning_mode": "topology_graph",
                "part_count": len(instances),
            },
        )
        constraints, grounding_warnings = self._ground_topology_graph(
            anchor=anchor,
            proposals=proposals,
            instances=instances,
            feedback=feedback,
        )
        warnings.extend(grounding_warnings)
        return anchor, constraints, warnings
    return self._parse_constraint_graph_json(
        raw=raw,
        instances=instances,
        default_anchor=default_anchor,
    )

  def _parse_topology_graph_json(
      self,
      raw: dict[str, Any],
      instances: dict[str, PartInstance],
      default_anchor: str,
  ) -> tuple[
      str,
      list[_TopologyConnectionProposal],
      list[str],
      dict[str, str],
      list[CanonicalStackGroup],
      str,
  ]:
    warnings: list[str] = []
    anchor = raw.get("anchor", default_anchor)
    if anchor not in instances:
      warnings.append(
          f"llm_invalid_anchor={anchor}, fallback_anchor={default_anchor}"
      )
      anchor = default_anchor
    plan_summary_text = ""
    plan_summary = raw.get("plan_summary")
    if isinstance(plan_summary, str) and plan_summary.strip():
      cleaned = " ".join(plan_summary.strip().split())
      plan_summary_text = cleaned[:220]
      warnings.append(f"plan_summary={cleaned[:220]}")

    incoming = raw.get("connections", [])
    if not incoming:
      incoming = raw.get("relations", [])
    if not incoming:
      incoming = raw.get("edges", [])
    if not isinstance(incoming, list):
      raise LLMPlannerError("connections must be a JSON array.")

    proposals: list[_TopologyConnectionProposal] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for item in incoming:
      if not isinstance(item, dict):
        continue
      part_a = (
          item.get("part_a")
          or item.get("source_part")
          or item.get("from_part")
          or item.get("part")
      )
      part_b = (
          item.get("part_b")
          or item.get("target_part")
          or item.get("to_part")
          or item.get("other_part")
      )
      if part_a not in instances or part_b not in instances or part_a == part_b:
        warnings.append(f"skip_invalid_connection={part_a}<->{part_b}")
        continue
      relation_type = self.rule_fallback._normalize_relation_hint(  # pylint: disable=protected-access
          item.get("relation_type")
          or item.get("relation")
          or item.get("kind")
          or item.get("mate_type")
          or item.get("constraint_hint")
      )
      label = str(item.get("label", "") or item.get("name", "")).strip()
      pair_key = tuple(sorted((str(part_a), str(part_b))) + [relation_type])
      if pair_key in seen_pairs:
        continue
      seen_pairs.add(pair_key)
      proposals.append(
          _TopologyConnectionProposal(
              part_a=str(part_a),
              part_b=str(part_b),
              relation_type=relation_type,
              label=label,
              stack_group=(
                  str(item.get("stack_group")).strip()
                  if item.get("stack_group") is not None
                  else None
              ) or None,
              order_hint=(
                  int(item.get("order_hint"))
                  if isinstance(item.get("order_hint"), int)
                  else None
              ),
              source_frame_hint=(
                  str(item.get("source_frame_hint")).strip()
                  if item.get("source_frame_hint") is not None
                  else None
              ) or None,
              target_frame_hint=(
                  str(item.get("target_frame_hint")).strip()
                  if item.get("target_frame_hint") is not None
                  else None
              ) or None,
              source_stop_hint=(
                  str(item.get("source_stop_hint")).strip()
                  if item.get("source_stop_hint") is not None
                  else None
              ) or None,
              target_stop_hint=(
                  str(item.get("target_stop_hint")).strip()
                  if item.get("target_stop_hint") is not None
                  else None
              ) or None,
              stack_phase=(
                  str(item.get("stack_phase")).strip().lower()
                  if item.get("stack_phase") is not None
                  else None
              ) or None,
              alignment_mode=(
                  str(item.get("alignment_mode")).strip().lower()
                  if item.get("alignment_mode") is not None
                  else None
              ) or None,
          )
      )

    if not proposals:
      raise LLMPlannerError("LLM returned no valid topology connections.")
    self._validate_topology_connectivity(anchor, proposals, instances)
    role_assignments = self._parse_role_assignments(raw, instances)
    stack_groups = self._parse_stack_groups(
        raw=raw,
        anchor=anchor,
        proposals=proposals,
        instances=instances,
    )
    return (
        anchor,
        proposals,
        warnings,
        role_assignments,
        stack_groups,
        plan_summary_text,
    )

  def _validate_topology_connectivity(
      self,
      anchor: str,
      proposals: list[_TopologyConnectionProposal],
      instances: dict[str, PartInstance],
  ) -> None:
    part_names = set(instances.keys())
    if anchor not in part_names:
      raise LLMPlannerError(f"anchor_not_in_instances={anchor}")

    adjacency = {name: set() for name in part_names}
    incident = {name: 0 for name in part_names}
    for proposal in proposals:
      if proposal.part_a not in part_names or proposal.part_b not in part_names:
        continue
      adjacency[proposal.part_a].add(proposal.part_b)
      adjacency[proposal.part_b].add(proposal.part_a)
      incident[proposal.part_a] += 1
      incident[proposal.part_b] += 1

    missing = sorted(
        name for name in part_names if name != anchor and incident[name] == 0
    )
    if missing:
      raise LLMPlannerError("missing_part_connections=" + ",".join(missing))

    stack = [anchor]
    visited: set[str] = set()
    while stack:
      current = stack.pop()
      if current in visited:
        continue
      visited.add(current)
      for nxt in adjacency[current]:
        if nxt not in visited:
          stack.append(nxt)

    disconnected = sorted(name for name in part_names if name not in visited)
    if disconnected:
      raise LLMPlannerError(
          "disconnected_parts_from_anchor=" + ",".join(disconnected)
      )

  def _parse_role_assignments(
      self,
      raw: dict[str, Any],
      instances: dict[str, PartInstance],
  ) -> dict[str, str]:
    incoming = raw.get("role_assignments")
    if not isinstance(incoming, dict):
      return {}
    result: dict[str, str] = {}
    for role_name, part_name in incoming.items():
      if not isinstance(role_name, str) or not isinstance(part_name, str):
        continue
      cleaned_role = role_name.strip().lower()
      cleaned_part = part_name.strip()
      if not cleaned_role or cleaned_part not in instances:
        continue
      result[cleaned_role] = cleaned_part
    return result

  def _parse_stack_groups(
      self,
      raw: dict[str, Any],
      anchor: str,
      proposals: list[_TopologyConnectionProposal],
      instances: dict[str, PartInstance],
  ) -> list[CanonicalStackGroup]:
    incoming = raw.get("stack_groups")
    groups: list[CanonicalStackGroup] = []
    if isinstance(incoming, list):
      for idx, item in enumerate(incoming):
        if not isinstance(item, dict):
          continue
        raw_parts = item.get("parts")
        if not isinstance(raw_parts, list):
          continue
        parts = [
            str(part).strip()
            for part in raw_parts
            if isinstance(part, str) and str(part).strip() in instances
        ]
        if len(parts) < 2:
          continue
        group_id = str(item.get("group_id") or f"stack_{idx:02d}").strip()
        if not group_id:
          group_id = f"stack_{idx:02d}"
        anchor_part = (
            str(item.get("anchor_part")).strip()
            if isinstance(item.get("anchor_part"), str)
            else ""
        )
        if anchor_part not in parts:
          anchor_part = anchor if anchor in parts else parts[0]
        groups.append(
            CanonicalStackGroup(
                group_id=group_id,
                parts=parts,
                anchor_part=anchor_part,
                relation_type=str(item.get("relation_type") or "coaxial").strip().lower(),
                metadata={"source": "llm"},
            )
        )
    if groups:
      return groups
    return self._infer_stack_groups(anchor, proposals)

  def _infer_stack_groups(
      self,
      anchor: str,
      proposals: list[_TopologyConnectionProposal],
  ) -> list[CanonicalStackGroup]:
    adjacency: dict[str, set[str]] = {}
    for proposal in proposals:
      if proposal.relation_type not in {"coaxial", "fasten"}:
        continue
      adjacency.setdefault(proposal.part_a, set()).add(proposal.part_b)
      adjacency.setdefault(proposal.part_b, set()).add(proposal.part_a)
    visited: set[str] = set()
    groups: list[CanonicalStackGroup] = []
    for start in sorted(adjacency):
      if start in visited:
        continue
      stack = [start]
      component: list[str] = []
      while stack:
        current = stack.pop()
        if current in visited:
          continue
        visited.add(current)
        component.append(current)
        for nxt in sorted(adjacency.get(current, set())):
          if nxt not in visited:
            stack.append(nxt)
      if len(component) < 2:
        continue
      ordered: list[str] = []
      if anchor in component:
        ordered.append(anchor)
      ordered.extend(item for item in sorted(component) if item not in ordered)
      groups.append(
          CanonicalStackGroup(
              group_id=f"stack_{len(groups):02d}",
              parts=ordered,
              anchor_part=ordered[0],
              relation_type="coaxial",
              metadata={"source": "inferred"},
          )
      )
    return groups

  def _ground_topology_graph(
      self,
      anchor: str,
      proposals: list[_TopologyConnectionProposal],
      instances: dict[str, PartInstance],
      feedback: Optional[PlannerFeedback],
  ) -> tuple[list[Constraint], list[str]]:
    warnings: list[str] = []
    constraints: list[Constraint] = []
    socket_usage: dict[tuple[str, str], int] = {}
    blocked_pairs = self.rule_fallback._blocked_pairs_from_feedback(  # pylint: disable=protected-access
        feedback
    )
    pending = list(proposals)
    connected: set[str] = {anchor}
    used_pairs: set[frozenset[str]] = set()

    while pending and len(connected) < len(instances):
      progressed = False
      for index, proposal in enumerate(pending):
        pair = frozenset((proposal.part_a, proposal.part_b))
        if pair in used_pairs:
          continue
        part_a_connected = proposal.part_a in connected
        part_b_connected = proposal.part_b in connected
        if part_a_connected == part_b_connected:
          continue

        parent_id = proposal.part_a if part_a_connected else proposal.part_b
        child_id = proposal.part_b if part_a_connected else proposal.part_a
        candidate, used_hint = self._ground_relation_with_fallback(
            parent=instances[parent_id],
            child=instances[child_id],
            socket_usage=socket_usage,
            blocked_pair=(pair in blocked_pairs),
            relation_hint=proposal.relation_type,
        )
        if candidate is None:
          continue
        for constraint in candidate.constraints:
          constraint.metadata["grounded_from_relation"] = proposal.relation_type
          constraint.metadata["grounded_with_hint"] = used_hint
          constraint.metadata["topology_pair"] = sorted([proposal.part_a, proposal.part_b])
          if proposal.stack_group:
            constraint.metadata["stack_group"] = proposal.stack_group
          if proposal.order_hint is not None:
            constraint.metadata["order_hint"] = proposal.order_hint
          if proposal.source_frame_hint:
            constraint.metadata["source_frame_hint"] = proposal.source_frame_hint
          if proposal.target_frame_hint:
            constraint.metadata["target_frame_hint"] = proposal.target_frame_hint
          if proposal.source_stop_hint:
            constraint.metadata["source_stop_hint"] = proposal.source_stop_hint
          if proposal.target_stop_hint:
            constraint.metadata["target_stop_hint"] = proposal.target_stop_hint
          if proposal.stack_phase:
            constraint.metadata["stack_phase"] = proposal.stack_phase
          if proposal.alignment_mode:
            constraint.metadata["alignment_mode"] = proposal.alignment_mode
        constraints.extend(candidate.constraints)
        self.rule_fallback._reserve_candidate(  # pylint: disable=protected-access
            candidate, socket_usage
        )
        connected.add(child_id)
        used_pairs.add(pair)
        pending.pop(index)
        label = proposal.label or proposal.relation_type
        warnings.append(
            f"grounded_relation={parent_id}->{child_id}:{label}:{used_hint}:{candidate.note}"
        )
        progressed = True
        break
      if not progressed:
        break

    if len(connected) < len(instances):
      missing = sorted(name for name in instances if name not in connected)
      raise LLMPlannerError(
          "ungrounded_topology_parts=" + ",".join(missing)
      )

    extra = max(0, len(proposals) - len(used_pairs))
    if extra > 0:
      warnings.append(f"topology_extra_edges_ignored={extra}")
    if not constraints:
      raise LLMPlannerError("topology_grounding_produced_no_constraints")
    return constraints, warnings

  def _fallback_canonical_graph(
      self,
      graph: ConstraintGraph,
  ) -> dict[str, Any]:
    proposals: list[_TopologyConnectionProposal] = []
    for constraint in graph.constraints:
      relation_type = "link"
      if constraint.ctype == ConstraintType.CONCENTRIC:
        relation_type = "coaxial"
      elif constraint.ctype == ConstraintType.COINCIDENT:
        relation_type = "planar"
      proposals.append(
          _TopologyConnectionProposal(
              part_a=constraint.part_a,
              part_b=constraint.part_b,
              relation_type=relation_type,
              label=constraint.label,
              source_frame_hint=(
                  str(constraint.metadata.get("source_frame_hint")).strip()
                  if constraint.metadata.get("source_frame_hint") is not None
                  else None
              ),
              target_frame_hint=(
                  str(constraint.metadata.get("target_frame_hint")).strip()
                  if constraint.metadata.get("target_frame_hint") is not None
                  else None
              ),
              source_stop_hint=(
                  str(constraint.metadata.get("source_stop_hint")).strip()
                  if constraint.metadata.get("source_stop_hint") is not None
                  else None
              ),
              target_stop_hint=(
                  str(constraint.metadata.get("target_stop_hint")).strip()
                  if constraint.metadata.get("target_stop_hint") is not None
                  else None
              ),
              stack_phase=(
                  str(constraint.metadata.get("stack_phase")).strip().lower()
                  if constraint.metadata.get("stack_phase") is not None
                  else None
              ),
              alignment_mode=(
                  str(constraint.metadata.get("alignment_mode")).strip().lower()
                  if constraint.metadata.get("alignment_mode") is not None
                  else None
              ),
          )
      )
    return CanonicalMateGraph(
        anchor=graph.anchor,
        edges=[
            CanonicalMateEdge(
                part_a=proposal.part_a,
                part_b=proposal.part_b,
                relation_type=proposal.relation_type,
                label=proposal.label,
                source_frame_hint=proposal.source_frame_hint,
                target_frame_hint=proposal.target_frame_hint,
                source_stop_hint=proposal.source_stop_hint,
                target_stop_hint=proposal.target_stop_hint,
                stack_phase=proposal.stack_phase,
                alignment_mode=proposal.alignment_mode,
            )
            for proposal in proposals
        ],
        stack_groups=self._infer_stack_groups(graph.anchor, proposals),
        plan_summary="",
        metadata={"source": "rule_fallback"},
    ).to_dict()

  def _ground_relation_with_fallback(
      self,
      parent: PartInstance,
      child: PartInstance,
      socket_usage: dict[tuple[str, str], int],
      blocked_pair: bool,
      relation_hint: str,
  ) -> tuple[Optional[_ConnectionCandidate], str]:
    hint = self.rule_fallback._normalize_relation_hint(  # pylint: disable=protected-access
        relation_hint
    )
    candidate = self.rule_fallback._best_connection(  # pylint: disable=protected-access
        parent=parent,
        child=child,
        socket_usage=socket_usage,
        blocked_pair=blocked_pair,
        relation_hint=hint,
    )
    if candidate is not None:
      return candidate, hint

    fallback_sequences = {
        "coaxial": ["link", "planar"],
        "planar": ["link", "coaxial"],
        "fasten": ["coaxial", "planar", "link"],
        "support": ["planar", "link", "coaxial"],
        "link": ["planar", "coaxial"],
    }
    for alt_hint in fallback_sequences.get(hint, ["link"]):
      candidate = self.rule_fallback._best_connection(  # pylint: disable=protected-access
          parent=parent,
          child=child,
          socket_usage=socket_usage,
          blocked_pair=blocked_pair,
          relation_hint=alt_hint,
      )
      if candidate is not None:
        return candidate, alt_hint
    return None, hint

  def _parse_constraint_graph_json(
      self,
      raw: dict[str, Any],
      instances: dict[str, PartInstance],
      default_anchor: str,
  ) -> tuple[str, list[Constraint], list[str]]:
    warnings: list[str] = []
    anchor = raw.get("anchor", default_anchor)
    if anchor not in instances:
      warnings.append(
          f"llm_invalid_anchor={anchor}, fallback_anchor={default_anchor}"
      )
      anchor = default_anchor
    plan_summary = raw.get("plan_summary")
    if isinstance(plan_summary, str) and plan_summary.strip():
      cleaned = " ".join(plan_summary.strip().split())
      warnings.append(f"plan_summary={cleaned[:220]}")

    incoming_constraints = raw.get("constraints", [])
    if not incoming_constraints:
      incoming_constraints = raw.get("relations", [])
    if not incoming_constraints:
      incoming_constraints = raw.get("edges", [])
    if not isinstance(incoming_constraints, list):
      raise LLMPlannerError("constraints must be a JSON array.")

    constraints: list[Constraint] = []
    for item in incoming_constraints:
      if not isinstance(item, dict):
        continue
      ctype_str = str(item.get("type", "")).lower()
      if ctype_str not in {
          ConstraintType.CONCENTRIC.value,
          ConstraintType.COINCIDENT.value,
          ConstraintType.DISTANCE.value,
      }:
        warnings.append(f"skip_invalid_constraint_type={ctype_str}")
        continue

      part_a = (
          item.get("part_a")
          or item.get("source_part")
          or item.get("from_part")
      )
      part_b = (
          item.get("part_b")
          or item.get("target_part")
          or item.get("to_part")
      )
      socket_a = (
          item.get("socket_a")
          or item.get("source_socket")
          or item.get("from_socket")
      )
      socket_b = (
          item.get("socket_b")
          or item.get("target_socket")
          or item.get("to_socket")
      )
      if (
          part_a not in instances
          or part_b not in instances
          or socket_a not in instances[part_a].sockets
          or socket_b not in instances[part_b].sockets
      ):
        warnings.append(
            "skip_invalid_endpoint="
            f"{part_a}.{socket_a}->{part_b}.{socket_b}"
        )
        continue
      socket_obj_a = instances[part_a].sockets[socket_a]
      socket_obj_b = instances[part_b].sockets[socket_b]
      if not self._constraint_endpoints_compatible(
          ctype=ConstraintType(ctype_str),
          socket_a=socket_obj_a,
          socket_b=socket_obj_b,
      ):
        warnings.append(
            "skip_incompatible_endpoint="
            f"{ctype_str}:{part_a}.{socket_a}->{part_b}.{socket_b}"
        )
        continue

      value = item.get("value")
      if value is not None:
        try:
          value = float(value)
        except (TypeError, ValueError):
          warnings.append(
              f"invalid_value_for_{part_a}_{socket_a}_{part_b}_{socket_b}"
          )
          value = None

      label = str(item.get("label", "") or item.get("name", "")).strip()
      constraints.append(
          Constraint(
              ctype=ConstraintType(ctype_str),
              part_a=str(part_a),
              socket_a=str(socket_a),
              part_b=str(part_b),
              socket_b=str(socket_b),
              value=value,
              label=label,
          )
      )

    if not constraints:
      raise LLMPlannerError("LLM returned no valid constraints.")
    return anchor, constraints, warnings

  def _validate_constraint_graph_connectivity(
      self,
      anchor: str,
      constraints: list[Constraint],
      instances: dict[str, PartInstance],
  ) -> None:
    part_names = set(instances.keys())
    if anchor not in part_names:
      raise LLMPlannerError(f"anchor_not_in_instances={anchor}")

    adjacency = {name: set() for name in part_names}
    incident = {name: 0 for name in part_names}
    for constraint in constraints:
      if constraint.part_a not in part_names or constraint.part_b not in part_names:
        continue
      adjacency[constraint.part_a].add(constraint.part_b)
      adjacency[constraint.part_b].add(constraint.part_a)
      incident[constraint.part_a] += 1
      incident[constraint.part_b] += 1

    missing = sorted(
        name for name in part_names if name != anchor and incident[name] == 0
    )
    if missing:
      raise LLMPlannerError(
          "missing_part_constraints=" + ",".join(missing)
      )

    stack = [anchor]
    visited: set[str] = set()
    while stack:
      current = stack.pop()
      if current in visited:
        continue
      visited.add(current)
      for nxt in adjacency[current]:
        if nxt not in visited:
          stack.append(nxt)

    disconnected = sorted(name for name in part_names if name not in visited)
    if disconnected:
      raise LLMPlannerError(
          "disconnected_parts_from_anchor=" + ",".join(disconnected)
      )

  def _constraint_endpoints_compatible(
      self, ctype: ConstraintType, socket_a: Socket, socket_b: Socket
  ) -> bool:
    if ctype == ConstraintType.CONCENTRIC:
      cyl_kinds = {"hole", "threaded_hole", "pin", "axis", "guide_slot"}
      if socket_a.kind not in cyl_kinds or socket_b.kind not in cyl_kinds:
        return False
      hole_like = {"hole", "threaded_hole"}
      pair = frozenset((socket_a.kind, socket_b.kind))

      # Hard reject mechanically weak defaults.
      if pair == {"pin"}:
        return False
      if pair <= hole_like:
        return False

      # Strongly preferred valid pairs.
      allowed_pairs = {
          frozenset(("hole", "pin")),
          frozenset(("threaded_hole", "pin")),
          frozenset(("hole", "axis")),
          frozenset(("threaded_hole", "axis")),
          frozenset(("axis", "axis")),
          frozenset(("axis", "pin")),
          frozenset(("guide_slot", "pin")),
          frozenset(("guide_slot", "axis")),
      }
      if pair not in allowed_pairs:
        return False

      # Concentric requires usable directions for robust symbolic solving.
      if socket_a.axis is None or socket_b.axis is None:
        return False
      return True

    if ctype == ConstraintType.COINCIDENT:
      planar_kinds = {"plane", "flange_plane", "guide_slot", "interface"}
      if socket_a.normal is not None and socket_b.normal is not None:
        return True
      return socket_a.kind in planar_kinds and socket_b.kind in planar_kinds

    if ctype == ConstraintType.DISTANCE:
      return (
          socket_a.normal is not None
          or socket_a.axis is not None
          or socket_b.normal is not None
          or socket_b.axis is not None
      )

    return False

  def _constraint_graph_schema(self) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["anchor", "constraints"],
        "properties": {
            "anchor": {"type": "string"},
            "plan_summary": {"type": "string"},
            "constraints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "type",
                        "part_a",
                        "socket_a",
                        "part_b",
                        "socket_b",
                    ],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["concentric", "coincident", "distance"],
                        },
                        "part_a": {"type": "string"},
                        "socket_a": {"type": "string"},
                        "part_b": {"type": "string"},
                        "socket_b": {"type": "string"},
                        "value": {"type": ["number", "null"]},
                        "label": {"type": "string"},
                    },
                },
            },
        },
    }

  def _topology_graph_schema(self) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["anchor", "connections"],
        "properties": {
            "anchor": {"type": "string"},
            "plan_summary": {"type": "string"},
            "role_assignments": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "stack_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["parts"],
                    "properties": {
                        "group_id": {"type": "string"},
                        "parts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "anchor_part": {"type": "string"},
                        "relation_type": {"type": "string"},
                    },
                },
            },
            "connections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["part_a", "part_b", "relation_type"],
                    "properties": {
                        "part_a": {"type": "string"},
                        "part_b": {"type": "string"},
                        "relation_type": {
                            "type": "string",
                            "enum": [
                                "coaxial",
                                "planar",
                                "fasten",
                                "support",
                                "link",
                            ],
                        },
                        "label": {"type": "string"},
                        "stack_group": {"type": "string"},
                        "order_hint": {"type": "integer"},
                        "source_frame_hint": {"type": "string"},
                        "target_frame_hint": {"type": "string"},
                        "source_stop_hint": {"type": "string"},
                        "target_stop_hint": {"type": "string"},
                        "stack_phase": {
                            "type": "string",
                            "enum": ["entry", "seat", "through_stop"],
                        },
                        "alignment_mode": {
                            "type": "string",
                            "enum": ["mate", "insert", "stack", "support", "through_stop"],
                        },
                    },
                },
            },
        },
    }

  def _few_shot_examples_for_prompt(self) -> list[dict[str, Any]]:
    return [
        {
            "name": "shaft_gear_chain",
            "instruction": (
                "Assemble shaft into base, then mount gear onto shaft."
            ),
            "parts": {
                "base": {
                    "interface_summary": [
                        {"kind": "hole", "count": 1, "top_radii": [4.0]},
                        {"kind": "plane", "count": 1, "top_radii": []},
                    ]
                },
                "shaft": {
                    "interface_summary": [
                        {"kind": "pin", "count": 1, "top_radii": [4.0]},
                        {"kind": "plane", "count": 1, "top_radii": []},
                    ]
                },
                "gear": {
                    "interface_summary": [
                        {"kind": "hole", "count": 1, "top_radii": [4.0]},
                        {"kind": "plane", "count": 1, "top_radii": []},
                    ]
                },
            },
            "good_output": {
                "anchor": "base",
                "connections": [
                    {
                        "part_a": "base",
                        "part_b": "shaft",
                        "relation_type": "coaxial",
                        "label": "shaft_into_base",
                    },
                    {
                        "part_a": "shaft",
                        "part_b": "gear",
                        "relation_type": "coaxial",
                        "label": "gear_on_shaft",
                    },
                ],
                "plan_summary": "Base anchors shaft; shaft then carries gear.",
            },
            "bad_pattern": "Do not output socket names in topology mode.",
        },
        {
            "name": "rail_slider_alignment",
            "instruction": (
                "Mount slider onto rail and then attach cover onto slider."
            ),
            "parts": {
                "rail": {
                    "interface_summary": [
                        {"kind": "guide_slot", "count": 1, "top_radii": []},
                        {"kind": "plane", "count": 1, "top_radii": []},
                    ]
                },
                "slider": {
                    "interface_summary": [
                        {"kind": "pin", "count": 1, "top_radii": [3.0]},
                        {"kind": "plane", "count": 1, "top_radii": []},
                    ]
                },
                "cover": {
                    "interface_summary": [
                        {"kind": "plane", "count": 1, "top_radii": []},
                        {"kind": "hole", "count": 1, "top_radii": [3.0]},
                    ]
                },
            },
            "good_output": {
                "anchor": "rail",
                "connections": [
                    {
                        "part_a": "rail",
                        "part_b": "slider",
                        "relation_type": "coaxial",
                        "label": "slider_guided_on_rail",
                    },
                    {
                        "part_a": "slider",
                        "part_b": "cover",
                        "relation_type": "planar",
                        "label": "cover_on_slider",
                    },
                ],
                "plan_summary": "Guide slider on rail first, then mate cover by planes.",
            },
            "bad_pattern": "Do not leave any part disconnected from anchor.",
        },
    ]
