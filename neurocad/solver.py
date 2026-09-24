"""Deterministic symbolic CAD solver with constrained parametric morphing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .math3d import norm, normalize, rotation_about_axis, rotation_from_to
from .domain_types import (
    AssemblyState,
    Constraint,
    ConstraintGraph,
    ConstraintType,
    PartInstance,
    Socket,
    SocketFrame,
    Transform,
)


class SolveError(RuntimeError):
  """Raised when symbolic solving fails."""


@dataclass
class SymbolicCadSolver:
  """Constraint solver that treats geometry operations as deterministic."""

  tolerance: float = 1e-5
  radius_clearance: float = 0.15
  max_passes: int = 24
  enable_axial_snapping: bool = True
  axial_snap_max_span_scale: float = 1.5
  axial_stack_step: float = 0.5
  axial_safety_clearance: float = 0.05
  secondary_seat_clearance: float = 0.05
  secondary_seat_soft_tolerance: float = 12.0

  def solve(
      self, graph: ConstraintGraph, allow_morphing: bool = True
  ) -> AssemblyState:
    instances = {name: inst.copy() for name, inst in graph.instances.items()}
    logs: list[str] = []
    cluster_context = self._build_cluster_context(graph)

    if graph.anchor not in instances:
      raise SolveError(f"Anchor '{graph.anchor}' is not in instance set.")
    instances[graph.anchor].transform = Transform.identity()
    logs.append(f"anchor {graph.anchor}=identity")

    unresolved = self._ordered_constraints(graph.constraints)
    for pass_idx in range(self.max_passes):
      if not unresolved:
        break
      logs.append(f"solve_pass={pass_idx} unresolved={len(unresolved)}")

      progressed = False
      next_unresolved: list[Constraint] = []
      handled_pair_groups: set[tuple[str, str]] = set()
      for constraint in unresolved:
        part_a = instances[constraint.part_a]
        part_b = instances[constraint.part_b]
        solved_a = part_a.transform is not None
        solved_b = part_b.transform is not None

        pair_key = tuple(sorted((constraint.part_a, constraint.part_b)))
        if pair_key in handled_pair_groups:
          continue

        if solved_a and not solved_b:
          oriented_group = self._collect_oriented_pair_constraints(
              unresolved=unresolved,
              known_id=constraint.part_a,
              unknown_id=constraint.part_b,
          )
          new_transform = self._solve_transform_group(
              known_part=part_a,
              unknown_part=part_b,
              constraints=oriented_group,
              all_instances=instances,
              cluster_context=cluster_context,
              allow_morphing=allow_morphing,
              logs=logs,
          )
          part_b.transform = new_transform
          logs.append(f"solved {part_b.instance_id} via {constraint.label}")
          handled_pair_groups.add(pair_key)
          progressed = True
          continue

        if solved_b and not solved_a:
          oriented_group = self._collect_oriented_pair_constraints(
              unresolved=unresolved,
              known_id=constraint.part_b,
              unknown_id=constraint.part_a,
          )
          new_transform = self._solve_transform_group(
              known_part=part_b,
              unknown_part=part_a,
              constraints=oriented_group,
              all_instances=instances,
              cluster_context=cluster_context,
              allow_morphing=allow_morphing,
              logs=logs,
          )
          part_a.transform = new_transform
          logs.append(f"solved {part_a.instance_id} via {constraint.label}")
          handled_pair_groups.add(pair_key)
          progressed = True
          continue

        if solved_a and solved_b:
          if constraint.ctype == ConstraintType.DISTANCE:
            if self._enforce_distance(part_a, part_b, constraint, logs):
              progressed = True
          err = self._constraint_error(part_a, part_b, constraint)
          if self._is_soft_secondary_seat(constraint, err):
            continue
          if err > max(self.tolerance * 20.0, 1e-4):
            raise SolveError(
                f"constraint '{constraint.label or constraint.ctype.value}' "
                f"violated with error={err:.6f}"
            )
          continue

        next_unresolved.append(constraint)

      unresolved = next_unresolved
      if unresolved and not progressed:
        break

    if unresolved:
      missing = [c.label or c.ctype.value for c in unresolved]
      raise SolveError(
          "unresolved constraints after propagation: " + ", ".join(missing)
      )

    # Final validation across all constraints.
    for constraint in graph.constraints:
      part_a = instances[constraint.part_a]
      part_b = instances[constraint.part_b]
      if part_a.transform is None or part_b.transform is None:
        raise SolveError(
            f"unsolved transform for {constraint.part_a} or {constraint.part_b}"
        )
      err = self._constraint_error(part_a, part_b, constraint)
      if self._is_soft_secondary_seat(constraint, err):
        continue
      if err > max(self.tolerance * 30.0, 5e-4):
        raise SolveError(
            f"final validation failed on '{constraint.label}': {err:.6f}"
        )

    solved_graph = ConstraintGraph(
        instances=instances,
        constraints=[Constraint(**c.__dict__) for c in graph.constraints],
        anchor=graph.anchor,
        metadata=dict(graph.metadata),
    )
    return AssemblyState(instances=instances, constraint_graph=solved_graph, logs=logs)

  def _ordered_constraints(
      self,
      constraints: list[Constraint],
  ) -> list[Constraint]:
    def _type_rank(item: Constraint) -> int:
      if item.ctype == ConstraintType.CONCENTRIC:
        return 0
      if item.ctype == ConstraintType.COINCIDENT:
        return 1
      return 2

    def _key(item: Constraint) -> tuple[int, str, int, int, str]:
      order_hint = item.metadata.get("order_hint")
      if not isinstance(order_hint, int):
        order_hint = 10**6
      stack_group = str(item.metadata.get("stack_group") or "")
      secondary_rank = 1 if bool(item.metadata.get("secondary_seat", False)) else 0
      return (
          int(order_hint),
          stack_group,
          _type_rank(item),
          secondary_rank,
          item.label or "",
      )

    return sorted(constraints, key=_key)

  def _build_cluster_context(
      self,
      graph: ConstraintGraph,
  ) -> dict[str, object]:
    carrier_parent: dict[str, str] = {}
    for constraint in graph.constraints:
      carrier_id = str(constraint.metadata.get("carrier_part_id") or "").strip()
      inserted_id = str(constraint.metadata.get("inserted_part_id") or "").strip()
      if carrier_id and inserted_id and carrier_id != inserted_id:
        carrier_parent.setdefault(inserted_id, carrier_id)

    stack_groups: dict[str, list[str]] = {}
    part_order: dict[tuple[str, str], int] = {}
    raw_graph = graph.metadata.get("canonical_mate_graph")
    if isinstance(raw_graph, dict):
      raw_groups = raw_graph.get("stack_groups")
      if isinstance(raw_groups, list):
        for item in raw_groups:
          if not isinstance(item, dict):
            continue
          group_id = str(item.get("group_id") or "").strip()
          raw_parts = item.get("parts")
          if not group_id or not isinstance(raw_parts, list):
            continue
          ordered = [
              str(part).strip()
              for part in raw_parts
              if isinstance(part, str) and str(part).strip()
          ]
          if len(ordered) < 2:
            continue
          stack_groups[group_id] = ordered
          for index, part_id in enumerate(ordered):
            part_order[(group_id, part_id)] = index

    if not stack_groups:
      fallback_members: dict[str, list[tuple[int, str]]] = {}
      for constraint in graph.constraints:
        group_id = str(constraint.metadata.get("stack_group") or "").strip()
        if not group_id:
          continue
        order_hint = constraint.metadata.get("order_hint")
        if not isinstance(order_hint, int):
          order_hint = 10**6
        fallback_members.setdefault(group_id, []).append((order_hint, constraint.part_a))
        fallback_members.setdefault(group_id, []).append((order_hint + 1, constraint.part_b))
      for group_id, items in fallback_members.items():
        ordered_parts: list[str] = []
        for _, part_id in sorted(items, key=lambda item: (item[0], item[1])):
          if part_id not in ordered_parts:
            ordered_parts.append(part_id)
        if len(ordered_parts) < 2:
          continue
        stack_groups[group_id] = ordered_parts
        for index, part_id in enumerate(ordered_parts):
          part_order[(group_id, part_id)] = index

    if not stack_groups and carrier_parent:
      children: dict[str, list[str]] = {}
      members: set[str] = set()
      for child_id, parent_id in carrier_parent.items():
        children.setdefault(parent_id, []).append(child_id)
        members.add(parent_id)
        members.add(child_id)
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
              group_id = f"stack_chain_{len(stack_groups):02d}"
              seen_paths.add(tuple(path))
              stack_groups[group_id] = path
              for index, part_id in enumerate(path):
                part_order[(group_id, part_id)] = index
            continue
          for child_id in reversed(next_children):
            stack.append((child_id, path + [child_id]))

    return {
        "carrier_parent": carrier_parent,
        "stack_groups": stack_groups,
        "part_order": part_order,
    }

  def _solve_transform(
      self,
      known_part: PartInstance,
      known_socket_name: str,
      unknown_part: PartInstance,
      unknown_socket_name: str,
      constraint: Constraint,
      allow_morphing: bool,
      logs: list[str],
  ) -> Transform:
    known_socket = self._must_get_socket(known_part, known_socket_name)
    unknown_socket = self._must_get_socket(unknown_part, unknown_socket_name)
    known_frame = known_part.world_socket(known_socket_name)

    if constraint.ctype == ConstraintType.CONCENTRIC and allow_morphing:
      self._maybe_morph_radii(
          known_part,
          known_socket_name,
          known_socket,
          unknown_part,
          unknown_socket_name,
          unknown_socket,
          logs,
      )

    direct_frame_transform = self._solve_transform_from_frames(
        known_part=known_part,
        known_socket_name=known_socket_name,
        known_socket=known_socket,
        unknown_part=unknown_part,
        unknown_socket_name=unknown_socket_name,
        unknown_socket=unknown_socket,
        constraint=constraint,
        logs=logs,
    )
    if direct_frame_transform is not None:
      return direct_frame_transform

    if constraint.ctype == ConstraintType.CONCENTRIC:
      axis_world = self._frame_axis_or_normal(known_frame)
      axis_local = self._socket_axis_or_normal(unknown_socket)
      rotation = rotation_from_to(axis_local, axis_world)
      target_origin = known_frame.origin.copy()
      if constraint.value is not None:
        target_origin += axis_world * float(constraint.value)
      translation = target_origin - rotation @ unknown_socket.origin
      return Transform(rotation=rotation, translation=translation)

    if constraint.ctype == ConstraintType.COINCIDENT:
      normal_world = self._frame_normal_or_axis(known_frame)
      normal_local = self._socket_normal_or_axis(unknown_socket)
      rotation = rotation_from_to(normal_local, -normal_world)
      target_origin = known_frame.origin.copy()
      if constraint.value is not None:
        target_origin += normal_world * float(constraint.value)
      translation = target_origin - rotation @ unknown_socket.origin
      return Transform(rotation=rotation, translation=translation)

    if constraint.ctype == ConstraintType.DISTANCE:
      direction_world = self._frame_normal_or_axis(known_frame)
      direction_local = self._socket_normal_or_axis(unknown_socket)
      rotation = rotation_from_to(direction_local, direction_world)
      distance = 0.0 if constraint.value is None else float(constraint.value)
      target_origin = known_frame.origin + direction_world * distance
      translation = target_origin - rotation @ unknown_socket.origin
      return Transform(rotation=rotation, translation=translation)

    raise SolveError(f"Unsupported constraint type: {constraint.ctype}")

  def _solve_transform_from_frames(
      self,
      known_part: PartInstance,
      known_socket_name: str,
      known_socket: Socket,
      unknown_part: PartInstance,
      unknown_socket_name: str,
      unknown_socket: Socket,
      constraint: Constraint,
      logs: list[str],
  ) -> Optional[Transform]:
    if known_part.transform is None:
      return None
    if constraint.ctype not in {
        ConstraintType.CONCENTRIC,
        ConstraintType.COINCIDENT,
    }:
      return None

    known_variants = known_socket.world_frame_variants(known_part.transform)
    unknown_variants = unknown_socket.local_frame_variants()
    if not known_variants or not unknown_variants:
      return None

    known_names = self._ordered_frame_variant_names(
        available=known_variants,
        preferred_hint=self._preferred_frame_hint(
            primary_hint=str(constraint.metadata.get("source_frame_hint") or ""),
            stop_hint=str(constraint.metadata.get("source_stop_hint") or ""),
            stack_phase=str(constraint.metadata.get("stack_phase") or ""),
        ),
        socket=known_socket,
        relation_hint=str(constraint.metadata.get("relation_hint") or ""),
    )
    unknown_names = self._ordered_frame_variant_names(
        available=unknown_variants,
        preferred_hint=self._preferred_frame_hint(
            primary_hint=str(constraint.metadata.get("target_frame_hint") or ""),
            stop_hint=str(constraint.metadata.get("target_stop_hint") or ""),
            stack_phase=str(constraint.metadata.get("stack_phase") or ""),
        ),
        socket=unknown_socket,
        relation_hint=str(constraint.metadata.get("relation_hint") or ""),
    )
    if not known_names or not unknown_names:
      return None

    anti_parallel = self._frame_alignment_requires_opposed_z(
        known_socket=known_socket,
        unknown_socket=unknown_socket,
        constraint=constraint,
    )
    best: Optional[tuple[tuple[int, float], Transform, str, str]] = None
    for known_name in known_names:
      known_frame = known_variants.get(known_name)
      if known_frame is None or known_frame.basis_matrix() is None:
        continue
      for unknown_name in unknown_names:
        unknown_frame = unknown_variants.get(unknown_name)
        if unknown_frame is None or unknown_frame.basis_matrix() is None:
          continue
        try:
          transform = self._align_frame_pair(
              known_frame=known_frame,
              unknown_frame=unknown_frame,
              anti_parallel=anti_parallel,
              axial_offset=constraint.value,
          )
        except Exception:
          continue
        rank = (
            0
            if (
                known_name == str(constraint.metadata.get("source_frame_hint") or "")
                or unknown_name
                == str(constraint.metadata.get("target_frame_hint") or "")
            )
            else 1,
            float(np.linalg.norm(transform.translation)),
        )
        if best is None or rank < best[0]:
          best = (rank, transform, known_name, unknown_name)
    if best is None:
      return None
    _, transform, known_name, unknown_name = best
    log_prefix = (
        "interface_frame_align"
        if bool(constraint.metadata.get("interface_grounding", False))
        else "frame_align"
    )
    logs.append(
        f"{log_prefix} "
        f"{unknown_part.instance_id}.{unknown_socket_name} -> "
        f"{known_part.instance_id}.{known_socket_name} "
        f"[{unknown_name}->{known_name}]"
    )
    return transform

  def _solve_transform_group(
      self,
      known_part: PartInstance,
      unknown_part: PartInstance,
      constraints: list[Constraint],
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      allow_morphing: bool,
      logs: list[str],
  ) -> Transform:
    if not constraints:
      raise SolveError("empty constraint group")
    entry_constraints = [
        item
        for item in constraints
        if self._constraint_stack_phase([item]) == "entry"
    ]
    seat_constraints = [
        item
        for item in constraints
        if self._constraint_stack_phase([item]) == "seat"
    ]
    through_constraints = [
        item
        for item in constraints
        if self._constraint_stack_phase([item]) == "through_stop"
    ]
    residual_constraints = [
        item
        for item in constraints
        if item not in entry_constraints
        and item not in seat_constraints
        and item not in through_constraints
    ]
    primary_pool = entry_constraints or constraints
    primary = min(
        primary_pool,
        key=lambda item: (
            0
            if item.ctype == ConstraintType.CONCENTRIC
            else 1 if item.ctype == ConstraintType.COINCIDENT else 2,
            0 if not bool(item.metadata.get("secondary_seat", False)) else 1,
        ),
    )
    transform = self._solve_transform(
        known_part=known_part,
        known_socket_name=primary.socket_a,
        unknown_part=unknown_part,
        unknown_socket_name=primary.socket_b,
        constraint=primary,
        allow_morphing=allow_morphing,
        logs=logs,
    )

    staged_unknown = unknown_part.copy()
    staged_unknown.transform = transform.copy()
    for extra in entry_constraints:
      if extra.signature() == primary.signature():
        continue
      refined = self._refine_transform_with_constraint(
          known_part=known_part,
          unknown_part=staged_unknown,
          transform=staged_unknown.transform,
          constraint=extra,
          primary_constraint=primary,
          logs=logs,
      )
      if refined is not None:
        staged_unknown.transform = refined
    if (
        self.enable_axial_snapping
        and primary.ctype == ConstraintType.CONCENTRIC
        and staged_unknown.transform is not None
    ):
      logs.append(f"stack_phase entry {unknown_part.instance_id}")
      staged_unknown.transform = self._maybe_snap_along_axis(
          known_part=known_part,
          known_socket_name=primary.socket_a,
          unknown_part=staged_unknown,
          unknown_socket_name=primary.socket_b,
          constraints=constraints,
          all_instances=all_instances,
          cluster_context=cluster_context,
          transform=staged_unknown.transform,
          logs=logs,
      )
    for extra in residual_constraints:
      if extra.signature() == primary.signature():
        continue
      refined = self._refine_transform_with_constraint(
          known_part=known_part,
          unknown_part=staged_unknown,
          transform=staged_unknown.transform,
          constraint=extra,
          primary_constraint=primary,
          logs=logs,
      )
      if refined is not None:
        staged_unknown.transform = refined
    if seat_constraints and staged_unknown.transform is not None:
      logs.append(f"stack_phase seat {unknown_part.instance_id}")
      for extra in seat_constraints:
        if extra.signature() == primary.signature():
          continue
        refined = self._refine_transform_with_constraint(
            known_part=known_part,
            unknown_part=staged_unknown,
            transform=staged_unknown.transform,
            constraint=extra,
            primary_constraint=primary,
            logs=logs,
        )
        if refined is not None:
          staged_unknown.transform = refined
      if (
          primary.ctype == ConstraintType.CONCENTRIC
          and not self._seat_constraints_satisfied(
              known_part=known_part,
              unknown_part=staged_unknown,
              seat_constraints=seat_constraints,
          )
      ):
        staged_unknown.transform = self._maybe_snap_along_axis(
            known_part=known_part,
            known_socket_name=primary.socket_a,
            unknown_part=staged_unknown,
            unknown_socket_name=primary.socket_b,
            constraints=[primary] + seat_constraints,
            all_instances=all_instances,
            cluster_context=cluster_context,
            transform=staged_unknown.transform,
            logs=logs,
        )
    if through_constraints and staged_unknown.transform is not None:
      logs.append(f"stack_phase through_stop {unknown_part.instance_id}")
      if primary.ctype == ConstraintType.CONCENTRIC:
        staged_unknown.transform = self._maybe_snap_along_axis(
            known_part=known_part,
            known_socket_name=primary.socket_a,
            unknown_part=staged_unknown,
            unknown_socket_name=primary.socket_b,
            constraints=[primary] + through_constraints,
            all_instances=all_instances,
            cluster_context=cluster_context,
            transform=staged_unknown.transform,
            logs=logs,
        )
    if (
        primary.ctype == ConstraintType.COINCIDENT
        and staged_unknown.transform is not None
    ):
      staged_unknown.transform = self._maybe_snap_planar_support(
          known_part=known_part,
          known_socket_name=primary.socket_a,
          unknown_part=staged_unknown,
          unknown_socket_name=primary.socket_b,
          constraints=constraints,
          all_instances=all_instances,
          cluster_context=cluster_context,
          transform=staged_unknown.transform,
          logs=logs,
      )
    if staged_unknown.transform is None:
      raise SolveError("group refinement produced no transform")
    return staged_unknown.transform

  def _collect_oriented_pair_constraints(
      self,
      unresolved: list[Constraint],
      known_id: str,
      unknown_id: str,
  ) -> list[Constraint]:
    oriented: list[Constraint] = []
    for constraint in unresolved:
      if constraint.part_a == known_id and constraint.part_b == unknown_id:
        oriented.append(constraint)
      elif constraint.part_a == unknown_id and constraint.part_b == known_id:
        swapped_metadata = dict(constraint.metadata)
        source_hint = swapped_metadata.get("source_frame_hint")
        target_hint = swapped_metadata.get("target_frame_hint")
        source_stop_hint = swapped_metadata.get("source_stop_hint")
        target_stop_hint = swapped_metadata.get("target_stop_hint")
        if source_hint is not None or target_hint is not None:
          swapped_metadata["source_frame_hint"] = target_hint
          swapped_metadata["target_frame_hint"] = source_hint
        if source_stop_hint is not None or target_stop_hint is not None:
          swapped_metadata["source_stop_hint"] = target_stop_hint
          swapped_metadata["target_stop_hint"] = source_stop_hint
        oriented.append(
            Constraint(
                ctype=constraint.ctype,
                part_a=known_id,
                socket_a=constraint.socket_b,
                part_b=unknown_id,
                socket_b=constraint.socket_a,
                value=constraint.value,
                label=constraint.label + "_inverted",
                metadata=swapped_metadata,
            )
        )
    return oriented

  def _refine_transform_with_constraint(
      self,
      known_part: PartInstance,
      unknown_part: PartInstance,
      transform: Optional[Transform],
      constraint: Constraint,
      primary_constraint: Optional[Constraint],
      logs: list[str],
  ) -> Optional[Transform]:
    if transform is None:
      return None
    working = transform.copy()
    unknown_part.transform = working
    frame_a = known_part.world_socket(constraint.socket_a)
    frame_b = unknown_part.world_socket(constraint.socket_b)
    if constraint.ctype == ConstraintType.DISTANCE:
      direction = self._frame_normal_or_axis(frame_a)
      delta = frame_b.origin - frame_a.origin
      along = float(np.dot(delta, direction))
      target = 0.0 if constraint.value is None else float(constraint.value)
      shift = (target - along) * direction
      if norm(shift) > self.tolerance:
        working = working.translated(shift)
        unknown_part.transform = working
        logs.append(
            f"group_refine distance {unknown_part.instance_id} by {shift.round(6).tolist()}"
        )
      return working
    if constraint.ctype == ConstraintType.COINCIDENT:
      if primary_constraint is not None and primary_constraint.ctype == ConstraintType.CONCENTRIC:
        twisted = self._twist_transform_for_seat(
            known_part=known_part,
            unknown_part=unknown_part,
            transform=working,
            primary_constraint=primary_constraint,
            seat_constraint=constraint,
            logs=logs,
        )
        if twisted is not None:
          working = twisted
          unknown_part.transform = working
          frame_b = unknown_part.world_socket(constraint.socket_b)
      if bool(constraint.metadata.get("secondary_seat", False)):
        alignment_mode = str(
            constraint.metadata.get("alignment_mode", "")
        ).strip().lower()
        if alignment_mode in {"support", "through_stop"}:
          return working
      direction = self._frame_normal_or_axis(frame_a)
      delta = frame_a.origin - frame_b.origin
      shift = float(np.dot(delta, direction)) * direction
      if norm(shift) > self.tolerance:
        working = working.translated(shift)
        unknown_part.transform = working
        logs.append(
            f"group_refine coincident {unknown_part.instance_id} by {shift.round(6).tolist()}"
        )
      return working
    return working

  def _twist_transform_for_seat(
      self,
      known_part: PartInstance,
      unknown_part: PartInstance,
      transform: Transform,
      primary_constraint: Constraint,
      seat_constraint: Constraint,
      logs: list[str],
  ) -> Optional[Transform]:
    unknown_part.transform = transform
    known_primary = known_part.world_socket(primary_constraint.socket_a)
    unknown_primary = unknown_part.world_socket(primary_constraint.socket_b)
    axis_world = self._frame_axis_or_normal(known_primary)
    seat_a = known_part.world_socket(seat_constraint.socket_a)
    seat_b = unknown_part.world_socket(seat_constraint.socket_b)
    if seat_a.normal is None or seat_b.normal is None:
      return None
    if (
        abs(abs(float(np.dot(seat_a.normal, axis_world))) - 1.0) > 0.2
        or abs(abs(float(np.dot(seat_b.normal, axis_world))) - 1.0) > 0.2
    ):
      return None

    pivot = known_primary.origin.copy()
    vec_a = seat_a.origin - pivot
    vec_b = seat_b.origin - pivot
    vec_a = vec_a - float(np.dot(vec_a, axis_world)) * axis_world
    vec_b = vec_b - float(np.dot(vec_b, axis_world)) * axis_world
    if norm(vec_a) <= 1e-6 or norm(vec_b) <= 1e-6:
      return None

    vec_a_u = normalize(vec_a)
    vec_b_u = normalize(vec_b)
    sin_term = float(np.dot(np.cross(vec_b_u, vec_a_u), axis_world))
    cos_term = float(np.clip(np.dot(vec_b_u, vec_a_u), -1.0, 1.0))
    angle = float(np.arctan2(sin_term, cos_term))
    if abs(angle) <= 1e-7:
      return None

    rotation = rotation_about_axis(axis_world, angle)
    new_rotation = rotation @ transform.rotation
    new_translation = pivot + rotation @ (transform.translation - pivot)
    logs.append(
        f"group_refine twist {unknown_part.instance_id} by {angle:.6f}rad"
    )
    return Transform(rotation=new_rotation, translation=new_translation)

  def _maybe_snap_along_axis(
      self,
      known_part: PartInstance,
      known_socket_name: str,
      unknown_part: PartInstance,
      unknown_socket_name: str,
      constraints: list[Constraint],
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      transform: Transform,
      logs: list[str],
  ) -> Transform:
    known_socket = self._must_get_socket(known_part, known_socket_name)
    unknown_socket = self._must_get_socket(unknown_part, unknown_socket_name)
    allowed_kinds = {"hole", "threaded_hole", "pin", "axis"}
    if (
        known_socket.kind not in allowed_kinds
        and unknown_socket.kind not in allowed_kinds
    ):
      return transform

    known_frame = known_part.world_socket(known_socket_name)
    axis_world = self._frame_axis_or_normal(known_frame)
    axis_origin = known_frame.origin.copy()
    working = transform.copy()
    unknown_part.transform = working
    relation_hint = self._constraint_relation_hint(constraints)
    stack_mode = self._constraint_stack_mode(constraints)
    stack_phase = self._constraint_stack_phase(constraints)
    preferred_sign = self._preferred_stack_direction(
        carrier_part=known_part,
        unknown_part=unknown_part,
        axis_world=axis_world,
    )
    seat_constraints = [
        item for item in constraints if bool(item.metadata.get("secondary_seat", False))
    ]
    seat_ok = self._seat_constraints_satisfied(
        known_part=known_part,
        unknown_part=unknown_part,
        seat_constraints=seat_constraints,
    )
    if relation_hint == "coaxial" and seat_constraints and seat_ok:
      return working

    shift: Optional[float] = None
    cluster_chain = self._cluster_support_chain(
        carrier_part=known_part,
        unknown_part=unknown_part,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
    )
    if stack_phase in {"seat", "through_stop"} and seat_constraints:
      shift = self._direct_secondary_seat_shift(
          carrier_part=known_part,
          unknown_part=unknown_part,
          axis_world=axis_world,
          all_instances=all_instances,
          cluster_context=cluster_context,
          constraints=constraints,
          relation_hint=relation_hint,
          stack_mode=stack_mode,
          preferred_sign=preferred_sign,
          logs=logs,
      )
    if stack_phase in {"seat", "through_stop"} or len(cluster_chain) >= 2:
      if shift is None:
        shift = self._joint_stack_shift_global(
            carrier_part=known_part,
            unknown_part=unknown_part,
            axis_world=axis_world,
            axis_origin=axis_origin,
            all_instances=all_instances,
            cluster_context=cluster_context,
            constraints=constraints,
            relation_hint=relation_hint,
            stack_mode=stack_mode,
            stack_phase=stack_phase,
            preferred_sign=preferred_sign,
        )
    if shift is None:
      shift = self._plane_contact_snap_shift_global(
          carrier_part=known_part,
          unknown_part=unknown_part,
          axis_world=axis_world,
          axis_origin=axis_origin,
          all_instances=all_instances,
          cluster_context=cluster_context,
          constraints=constraints,
          relation_hint=relation_hint,
          stack_mode=stack_mode,
          preferred_sign=preferred_sign,
      )
    if shift is None:
      shift = self._bbox_contact_snap_shift_global(
          carrier_part=known_part,
          unknown_part=unknown_part,
          axis_world=axis_world,
          all_instances=all_instances,
          cluster_context=cluster_context,
          constraints=constraints,
          relation_hint=relation_hint,
          stack_mode=stack_mode,
          preferred_sign=preferred_sign,
      )
    if shift is None or abs(float(shift)) <= self.tolerance:
      return working

    max_span = (
        self._projected_bbox_span(known_part, axis_world)
        + self._projected_bbox_span(unknown_part, axis_world)
    )
    max_shift = self.axial_snap_max_span_scale * max(max_span, 1.0)
    if abs(float(shift)) > max_shift:
      return working

    working = working.translated(axis_world * float(shift))
    unknown_part.transform = working
    logs.append(
        f"axial_fallback {unknown_part.instance_id}"
    )
    logs.append(
        f"axial_snap {unknown_part.instance_id} by {float(shift):.6f}"
    )
    if seat_constraints:
      retreat = min(
          float(self.secondary_seat_clearance),
          max(0.0, 0.25 * abs(float(shift))),
      )
      if retreat > self.tolerance:
        retreat_dir = -1.0 if float(shift) >= 0.0 else 1.0
        working = working.translated(axis_world * (retreat_dir * retreat))
        unknown_part.transform = working
        logs.append(
            f"seat_clearance_retreat {unknown_part.instance_id} by {retreat:.6f}"
        )
    else:
      retreat = min(
          float(self.axial_safety_clearance),
          max(0.0, 0.5 * abs(float(shift))),
      )
      if retreat > self.tolerance:
        retreat_dir = -1.0 if float(shift) >= 0.0 else 1.0
        working = working.translated(axis_world * (retreat_dir * retreat))
        unknown_part.transform = working
        logs.append(
            f"axial_safety_retreat {unknown_part.instance_id} by {retreat:.6f}"
        )
    return working

  def _direct_secondary_seat_shift(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      axis_world: np.ndarray,
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      constraints: list[Constraint],
      relation_hint: str,
      stack_mode: str,
      preferred_sign: float,
      logs: Optional[list[str]] = None,
  ) -> Optional[float]:
    seat_constraints = [
        item for item in constraints if bool(item.metadata.get("secondary_seat", False))
    ]
    if not seat_constraints:
      return None
    blockers = self._placed_stack_support_parts(
        carrier_part=carrier_part,
        unknown_part=unknown_part,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
        relation_hint=relation_hint,
        stack_mode=stack_mode,
        include_carrier=True,
    )
    noncarrier_blockers = [
        blocker
        for blocker in blockers
        if blocker.instance_id != carrier_part.instance_id
    ]
    best_shift: Optional[float] = None
    best_rank: Optional[tuple[int, int, float]] = None
    for constraint in seat_constraints:
      try:
        frame_a = carrier_part.world_socket(constraint.socket_a)
        frame_b = unknown_part.world_socket(constraint.socket_b)
      except Exception:
        continue
      target_shift = float(np.dot(frame_a.origin - frame_b.origin, axis_world))
      if abs(target_shift) <= self.tolerance:
        return 0.0
      candidate_shift = target_shift
      if noncarrier_blockers and not self._shift_causes_stack_collision(
          unknown_part=unknown_part,
          blockers=noncarrier_blockers,
          axis_world=axis_world,
          shift=target_shift,
      ):
        candidate_shift = target_shift
      elif noncarrier_blockers:
        candidate_shift = self._ray_cast_snap_shift(
            unknown_part=unknown_part,
            blockers=noncarrier_blockers,
            axis_world=axis_world,
            target_shift=target_shift,
        )
        if candidate_shift is None:
          candidate_shift = target_shift
      carrier_only = [
          blocker
          for blocker in blockers
          if blocker.instance_id == carrier_part.instance_id
      ]
      if carrier_only and self._shift_causes_stack_collision(
          unknown_part=unknown_part,
          blockers=carrier_only,
          axis_world=axis_world,
          shift=candidate_shift,
      ):
        retreated = self._retreat_from_collision_toward_zero(
            unknown_part=unknown_part,
            blockers=carrier_only,
            axis_world=axis_world,
            target_shift=float(candidate_shift),
        )
        if retreated is not None:
          candidate_shift = retreated
      if noncarrier_blockers and self._shift_causes_stack_collision(
          unknown_part=unknown_part,
          blockers=noncarrier_blockers,
          axis_world=axis_world,
          shift=candidate_shift,
      ):
        continue
      support_penalty = 0
      if str(constraint.metadata.get("carrier_support_role") or "").strip().lower() == "through_stop":
        support_penalty += 1
      if str(constraint.metadata.get("inserted_support_role") or "").strip().lower() == "through_stop":
        support_penalty += 1
      if str(constraint.metadata.get("carrier_support_role") or "").strip().lower() == "shoulder":
        support_penalty -= 1
      if str(constraint.metadata.get("inserted_support_role") or "").strip().lower() == "shoulder":
        support_penalty -= 1
      sign_penalty = 0
      if preferred_sign and float(candidate_shift) * preferred_sign < -self.tolerance:
        sign_penalty = 1
      residual_penalty = abs(float(target_shift - candidate_shift))
      rank = (support_penalty, sign_penalty, residual_penalty, abs(float(candidate_shift)))
      if best_rank is None or rank < best_rank:
        best_rank = rank
        best_shift = float(candidate_shift)
        if logs is not None:
          logs.append(
              "seat_resolve "
              f"{unknown_part.instance_id} target={target_shift:.6f} "
              f"resolved={candidate_shift:.6f}"
          )
    return best_shift

  def _retreat_from_collision_toward_zero(
      self,
      unknown_part: PartInstance,
      blockers: list[PartInstance],
      axis_world: np.ndarray,
      target_shift: float,
  ) -> Optional[float]:
    if abs(float(target_shift)) <= self.tolerance:
      return None
    if not self._shift_causes_stack_collision(
        unknown_part=unknown_part,
        blockers=blockers,
        axis_world=axis_world,
        shift=float(target_shift),
    ):
      return float(target_shift)
    lower = 0.0
    upper = float(target_shift)
    for _ in range(14):
      mid = 0.5 * (lower + upper)
      if self._shift_causes_stack_collision(
          unknown_part=unknown_part,
          blockers=blockers,
          axis_world=axis_world,
          shift=mid,
      ):
        upper = mid
      else:
        lower = mid
    if abs(lower) <= self.tolerance:
      return None
    return float(lower)

  def _maybe_snap_planar_support(
      self,
      known_part: PartInstance,
      known_socket_name: str,
      unknown_part: PartInstance,
      unknown_socket_name: str,
      constraints: list[Constraint],
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      transform: Transform,
      logs: list[str],
  ) -> Transform:
    known_socket = self._must_get_socket(known_part, known_socket_name)
    unknown_socket = self._must_get_socket(unknown_part, unknown_socket_name)
    allowed_kinds = {"plane", "flange_plane"}
    if (
        known_socket.kind not in allowed_kinds
        and unknown_socket.kind not in allowed_kinds
    ):
      return transform

    known_frame = known_part.world_socket(known_socket_name)
    normal_world = self._frame_normal_or_axis(known_frame)
    basis_u, basis_v = self._orthogonal_basis(normal_world)
    working = transform.copy()
    unknown_part.transform = working
    relation_hint = self._constraint_relation_hint(constraints)

    shift_vec = self._planar_contact_snap_shift_global(
        carrier_part=known_part,
        unknown_part=unknown_part,
        basis_u=basis_u,
        basis_v=basis_v,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
        relation_hint=relation_hint,
    )
    if shift_vec is None:
      return working
    shift_norm = norm(shift_vec)
    if shift_norm <= self.tolerance:
      return working

    max_shift = self.axial_snap_max_span_scale * max(
        self._projected_bbox_span(known_part, basis_u)
        + self._projected_bbox_span(known_part, basis_v)
        + self._projected_bbox_span(unknown_part, basis_u)
        + self._projected_bbox_span(unknown_part, basis_v),
        1.0,
    )
    if shift_norm > max_shift:
      return working

    working = working.translated(shift_vec)
    unknown_part.transform = working
    logs.append(f"planar_fallback {unknown_part.instance_id}")
    logs.append(
        "planar_snap "
        f"{unknown_part.instance_id} by {shift_vec.round(6).tolist()}"
    )
    return working

  def _planar_contact_snap_shift_global(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      basis_u: np.ndarray,
      basis_v: np.ndarray,
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      constraints: list[Constraint],
      relation_hint: str,
  ) -> Optional[np.ndarray]:
    blockers = [
        part
        for part in all_instances.values()
        if part.transform is not None and part.instance_id != unknown_part.instance_id
    ]
    if not blockers:
      return None

    blocker_groups: list[list[PartInstance]] = []
    carrier_group = [
        part for part in blockers if part.instance_id == carrier_part.instance_id
    ]
    if carrier_group:
      blocker_groups.append(carrier_group)
    cluster_chain = self._cluster_support_chain(
        carrier_part=carrier_part,
        unknown_part=unknown_part,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
    )
    if cluster_chain:
      blocker_groups.append(cluster_chain)
    if relation_hint in {"planar", "support", "mate"} and len(blockers) > len(carrier_group):
      blocker_groups.append(blockers)

    for blocker_group in blocker_groups or [blockers]:
      shift_vec = self._nearest_planar_bbox_contact_shift(
          unknown_part=unknown_part,
          blockers=blocker_group,
          basis_u=basis_u,
          basis_v=basis_v,
          preferred_carrier_id=carrier_part.instance_id,
      )
      if shift_vec is not None and norm(shift_vec) > self.tolerance:
        return shift_vec
    return None

  def _nearest_planar_bbox_contact_shift(
      self,
      unknown_part: PartInstance,
      blockers: list[PartInstance],
      basis_u: np.ndarray,
      basis_v: np.ndarray,
      preferred_carrier_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    unknown_u = self._projected_bbox_interval(unknown_part, basis_u)
    unknown_v = self._projected_bbox_interval(unknown_part, basis_v)
    if unknown_u is None or unknown_v is None:
      return None
    current_score = self._planar_overlap_score(
        unknown_u=unknown_u,
        unknown_v=unknown_v,
        shift_u=0.0,
        shift_v=0.0,
        blockers=blockers,
        basis_u=basis_u,
        basis_v=basis_v,
    )
    if current_score[0] == 0:
      return None

    best_shift: Optional[np.ndarray] = None
    best_rank: Optional[tuple[int, float, float]] = None
    for blocker in blockers:
      blocker_u = self._projected_bbox_interval(blocker, basis_u)
      blocker_v = self._projected_bbox_interval(blocker, basis_v)
      if blocker_u is None or blocker_v is None:
        continue
      overlap_u = min(unknown_u[1], blocker_u[1]) - max(unknown_u[0], blocker_u[0])
      overlap_v = min(unknown_v[1], blocker_v[1]) - max(unknown_v[0], blocker_v[0])
      if overlap_u <= -self.tolerance or overlap_v <= -self.tolerance:
        continue
      candidate_offsets = [
          (float(blocker_u[0] - unknown_u[1]), 0.0),
          (float(blocker_u[1] - unknown_u[0]), 0.0),
          (0.0, float(blocker_v[0] - unknown_v[1])),
          (0.0, float(blocker_v[1] - unknown_v[0])),
      ]
      carrier_penalty = 0
      if (
          preferred_carrier_id is not None
          and blocker.instance_id != preferred_carrier_id
      ):
        carrier_penalty = 1
      for shift_u, shift_v in candidate_offsets:
        score = self._planar_overlap_score(
            unknown_u=unknown_u,
            unknown_v=unknown_v,
            shift_u=shift_u,
            shift_v=shift_v,
            blockers=blockers,
            basis_u=basis_u,
            basis_v=basis_v,
        )
        rank = (
            score[0] + carrier_penalty,
            score[1],
            abs(float(shift_u)) + abs(float(shift_v)),
        )
        if best_rank is None or rank < best_rank:
          best_rank = rank
          best_shift = basis_u * float(shift_u) + basis_v * float(shift_v)
    return best_shift

  def _planar_overlap_score(
      self,
      *,
      unknown_u: tuple[float, float],
      unknown_v: tuple[float, float],
      shift_u: float,
      shift_v: float,
      blockers: list[PartInstance],
      basis_u: np.ndarray,
      basis_v: np.ndarray,
  ) -> tuple[int, float]:
    moved_u = (float(unknown_u[0] + shift_u), float(unknown_u[1] + shift_u))
    moved_v = (float(unknown_v[0] + shift_v), float(unknown_v[1] + shift_v))
    overlap_count = 0
    overlap_area = 0.0
    for blocker in blockers:
      blocker_u = self._projected_bbox_interval(blocker, basis_u)
      blocker_v = self._projected_bbox_interval(blocker, basis_v)
      if blocker_u is None or blocker_v is None:
        continue
      overlap_u = min(moved_u[1], blocker_u[1]) - max(moved_u[0], blocker_u[0])
      overlap_v = min(moved_v[1], blocker_v[1]) - max(moved_v[0], blocker_v[0])
      if overlap_u > -self.tolerance and overlap_v > -self.tolerance:
        overlap_count += 1
        overlap_area += max(0.0, overlap_u) * max(0.0, overlap_v)
    return overlap_count, float(overlap_area)

  def _plane_contact_snap_shift_global(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      axis_world: np.ndarray,
      axis_origin: np.ndarray,
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      constraints: list[Constraint],
      relation_hint: str,
      stack_mode: str,
      preferred_sign: float,
  ) -> Optional[float]:
    unknown_candidates = self._support_plane_candidates(
        part=unknown_part,
        axis_world=axis_world,
        axis_origin=axis_origin,
    )
    if not unknown_candidates:
      return None
    placed_parts = self._placed_stack_support_parts(
        carrier_part=carrier_part,
        unknown_part=unknown_part,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
        relation_hint=relation_hint,
        stack_mode=stack_mode,
    )
    for support_group in self._support_groups_for_stack(
        carrier_part=carrier_part,
        placed_parts=placed_parts,
        relation_hint=relation_hint,
        stack_mode=stack_mode,
    ):
      best_shift: Optional[float] = None
      best_rank: Optional[tuple[int, float, float]] = None
      for support_part in support_group:
        support_candidates = self._support_plane_candidates(
            part=support_part,
            axis_world=axis_world,
            axis_origin=axis_origin,
        )
        for _, known_frame, known_score in support_candidates:
          if known_frame.normal is None:
            continue
          for _, unknown_frame, unknown_score in unknown_candidates:
            if unknown_frame.normal is None:
              continue
            dot = float(np.dot(known_frame.normal, unknown_frame.normal))
            if abs(dot) < 0.65:
              continue
            shift = float(
                np.dot(known_frame.origin - unknown_frame.origin, axis_world)
            )
            sign_penalty = 0
            if preferred_sign and shift * preferred_sign < -self.tolerance:
              sign_penalty = 1
            carrier_penalty = (
                0 if support_part.instance_id == carrier_part.instance_id else 1
            )
            quality_bonus = 0.05 * float(known_score + unknown_score)
            rank = (
                sign_penalty + carrier_penalty,
                abs(shift),
                -quality_bonus,
            )
            if best_rank is None or rank < best_rank:
              best_rank = rank
              best_shift = shift
      if best_shift is not None:
        return best_shift
    return None

  def _support_plane_candidates(
      self,
      part: PartInstance,
      axis_world: np.ndarray,
      axis_origin: Optional[np.ndarray] = None,
  ) -> list[tuple[str, SocketFrame, float]]:
    candidates: list[tuple[str, SocketFrame, float]] = []
    if part.transform is None:
      return candidates
    for name, socket in part.sockets.items():
      if socket.kind not in {"plane", "flange_plane"}:
        continue
      try:
        frame = part.world_socket(name)
      except Exception:
        continue
      if frame.normal is None:
        continue
      alignment = abs(float(np.dot(frame.normal, axis_world)))
      if alignment < 0.65:
        continue
      quality = float(socket.metadata.get("quality", 0.0) or 0.0)
      if bool(socket.metadata.get("proxy_socket", False)):
        quality -= 0.35
      radial_penalty = 0.0
      if axis_origin is not None:
        delta = frame.origin - axis_origin
        delta = delta - float(np.dot(delta, axis_world)) * axis_world
        radial_penalty = float(np.linalg.norm(delta))
      candidates.append((name, frame, quality + alignment - 0.28 * radial_penalty))
    candidates.sort(key=lambda item: -float(item[2]))
    return candidates[:8]

  def _bbox_contact_snap_shift_global(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      axis_world: np.ndarray,
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      constraints: list[Constraint],
      relation_hint: str,
      stack_mode: str,
      preferred_sign: float,
  ) -> Optional[float]:
    unknown_interval = self._projected_bbox_interval(unknown_part, axis_world)
    if unknown_interval is None:
      return None
    blockers = self._placed_stack_support_parts(
        carrier_part=carrier_part,
        unknown_part=unknown_part,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
        relation_hint=relation_hint,
        stack_mode=stack_mode,
        include_carrier=True,
    )
    if not blockers:
      return None
    for blocker_group in self._support_groups_for_stack(
        carrier_part=carrier_part,
        placed_parts=blockers,
        relation_hint=relation_hint,
        stack_mode=stack_mode,
    ):
      exact = self._nearest_bbox_contact_shift(
          unknown_part=unknown_part,
          blockers=blocker_group,
          axis_world=axis_world,
          preferred_sign=preferred_sign,
          preferred_carrier_id=carrier_part.instance_id,
      )
      if exact is None:
        continue
      refined = self._ray_cast_snap_shift(
          unknown_part=unknown_part,
          blockers=blocker_group,
          axis_world=axis_world,
          target_shift=exact,
      )
      if refined is None:
        continue
      if self._shift_causes_stack_collision(
          unknown_part=unknown_part,
          blockers=blockers,
          axis_world=axis_world,
          shift=float(refined),
          ignore_blockers={item.instance_id for item in blocker_group},
      ):
        continue
      return refined
    return None

  def _placed_stack_support_parts(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      constraints: list[Constraint],
      relation_hint: str,
      stack_mode: str,
      include_carrier: bool = True,
  ) -> list[PartInstance]:
    cluster_chain = self._cluster_support_chain(
        carrier_part=carrier_part,
        unknown_part=unknown_part,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
    )
    if cluster_chain:
      return cluster_chain
    placed: list[PartInstance] = []
    for part in all_instances.values():
      if part.instance_id == unknown_part.instance_id:
        continue
      if part.transform is None:
        continue
      if not include_carrier and part.instance_id == carrier_part.instance_id:
        continue
      placed.append(part)
    if relation_hint == "fasten":
      placed.sort(
          key=lambda item: (
              0 if item.instance_id != carrier_part.instance_id else 1,
              item.instance_id,
          )
      )
    elif stack_mode in {"hole_on_pin", "pin_into_hole"}:
      placed.sort(
          key=lambda item: (
              0 if item.instance_id == carrier_part.instance_id else 1,
              item.instance_id,
          )
      )
    else:
      placed.sort(
          key=lambda item: (
              0 if item.instance_id == carrier_part.instance_id else 1,
              item.instance_id,
          )
      )
    return placed

  def _cluster_support_chain(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      constraints: list[Constraint],
  ) -> list[PartInstance]:
    carrier_parent = cluster_context.get("carrier_parent")
    if not isinstance(carrier_parent, dict):
      carrier_parent = {}
    group_ids = self._active_stack_group_ids(
        constraints,
        cluster_context=cluster_context,
        focus_parts={carrier_part.instance_id, unknown_part.instance_id},
    )
    stack_groups = cluster_context.get("stack_groups")
    part_order = cluster_context.get("part_order")
    if not isinstance(stack_groups, dict):
      stack_groups = {}
    if not isinstance(part_order, dict):
      part_order = {}

    ordered_ids: list[str] = []
    for group_id in group_ids:
      members = stack_groups.get(group_id)
      if not isinstance(members, list) or unknown_part.instance_id not in members:
        continue
      unknown_idx = part_order.get((group_id, unknown_part.instance_id))
      if not isinstance(unknown_idx, int):
        unknown_idx = members.index(unknown_part.instance_id)
      for part_id in members[:unknown_idx]:
        if part_id not in ordered_ids:
          ordered_ids.append(part_id)

    current = carrier_part.instance_id
    visited: set[str] = set()
    while current and current not in visited:
      visited.add(current)
      if current != unknown_part.instance_id and current not in ordered_ids:
        ordered_ids.append(current)
      parent = carrier_parent.get(current)
      current = parent if isinstance(parent, str) else ""

    result: list[PartInstance] = []
    seen: set[str] = set()
    for part_id in ordered_ids:
      part = all_instances.get(part_id)
      if part is None or part.transform is None or part_id in seen:
        continue
      seen.add(part_id)
      result.append(part)
    return result

  def _active_stack_group_ids(
      self,
      constraints: list[Constraint],
      cluster_context: Optional[dict[str, object]] = None,
      focus_parts: Optional[set[str]] = None,
  ) -> list[str]:
    group_ids: list[str] = []
    for constraint in constraints:
      group_id = str(constraint.metadata.get("stack_group") or "").strip()
      if group_id and group_id not in group_ids:
        group_ids.append(group_id)
    if group_ids or cluster_context is None or not focus_parts:
      return group_ids
    stack_groups = cluster_context.get("stack_groups")
    if not isinstance(stack_groups, dict):
      return group_ids
    for group_id, members in stack_groups.items():
      if not isinstance(members, list):
        continue
      member_set = {
          str(item).strip()
          for item in members
          if isinstance(item, str) and str(item).strip()
      }
      if focus_parts.intersection(member_set) and group_id not in group_ids:
        group_ids.append(group_id)
    return group_ids

  def _joint_stack_shift_global(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      axis_world: np.ndarray,
      axis_origin: np.ndarray,
      all_instances: dict[str, PartInstance],
      cluster_context: dict[str, object],
      constraints: list[Constraint],
      relation_hint: str,
      stack_mode: str,
      stack_phase: str,
      preferred_sign: float,
  ) -> Optional[float]:
    blockers = self._placed_stack_support_parts(
        carrier_part=carrier_part,
        unknown_part=unknown_part,
        all_instances=all_instances,
        cluster_context=cluster_context,
        constraints=constraints,
        relation_hint=relation_hint,
        stack_mode=stack_mode,
        include_carrier=True,
    )
    if not blockers:
      return None
    support_order = [part.instance_id for part in blockers]
    unknown_interval = self._projected_bbox_interval(unknown_part, axis_world)
    if unknown_interval is None:
      return None
    candidate_shifts: set[float] = {0.0}
    unknown_support = self._support_plane_candidates(
        part=unknown_part,
        axis_world=axis_world,
        axis_origin=axis_origin,
    )
    for blocker in blockers:
      blocker_interval = self._projected_bbox_interval(blocker, axis_world)
      if blocker_interval is not None and self._perpendicular_bbox_overlap(
          part_a=unknown_part,
          part_b=blocker,
          axis_world=axis_world,
      ):
        candidate_shifts.add(float(blocker_interval[0] - unknown_interval[1]))
        candidate_shifts.add(float(blocker_interval[1] - unknown_interval[0]))
      support_candidates = self._support_plane_candidates(
          part=blocker,
          axis_world=axis_world,
          axis_origin=axis_origin,
      )
      for _, blocker_frame, _ in support_candidates:
        if blocker_frame.normal is None:
          continue
        for _, unknown_frame, _ in unknown_support:
          if unknown_frame.normal is None:
            continue
          dot = float(np.dot(blocker_frame.normal, unknown_frame.normal))
          if abs(dot) < 0.65:
            continue
          candidate_shifts.add(
              float(np.dot(blocker_frame.origin - unknown_frame.origin, axis_world))
          )

    best_shift: Optional[float] = None
    best_rank: Optional[tuple[int, int, int, int, float, float]] = None
    for shift in candidate_shifts:
      rank = self._joint_stack_shift_rank(
          unknown_part=unknown_part,
          blockers=blockers,
          axis_world=axis_world,
          shift=float(shift),
          preferred_sign=preferred_sign,
          support_order=support_order,
          phase=stack_phase,
      )
      if best_rank is None or rank < best_rank:
        best_rank = rank
        best_shift = float(shift)
    return best_shift

  def _joint_stack_shift_rank(
      self,
      unknown_part: PartInstance,
      blockers: list[PartInstance],
      axis_world: np.ndarray,
      shift: float,
      preferred_sign: float,
      support_order: list[str],
      phase: str,
  ) -> tuple[int, int, int, int, float, float]:
    moved_interval = self._projected_bbox_interval_with_shift(
        part=unknown_part,
        axis_world=axis_world,
        shift=float(shift),
    )
    if moved_interval is None:
      return (10**6, 10**6, 10**6, 10**6, 10**6, abs(float(shift)))
    moved_min, moved_max = moved_interval
    collision_count = 0
    collision_overlap = 0.0
    contact_ids: list[str] = []
    for blocker in blockers:
      blocker_interval = self._projected_bbox_interval(blocker, axis_world)
      if blocker_interval is None:
        continue
      if not self._perpendicular_bbox_overlap(
          part_a=unknown_part,
          part_b=blocker,
          axis_world=axis_world,
      ):
        continue
      blocker_min, blocker_max = blocker_interval
      overlap = min(moved_max, blocker_max) - max(moved_min, blocker_min)
      if overlap > max(self.tolerance, 1e-3):
        collision_count += 1
        collision_overlap += float(overlap)
      if (
          abs(moved_min - blocker_max) <= max(self.tolerance, 1e-3)
          or abs(moved_max - blocker_min) <= max(self.tolerance, 1e-3)
      ):
        contact_ids.append(blocker.instance_id)
    sign_penalty = 0
    if preferred_sign and float(shift) * preferred_sign < -self.tolerance:
      sign_penalty = 1
    preferred_support_penalty = 2
    if support_order:
      deepest_id = support_order[-1]
      if deepest_id in contact_ids:
        preferred_support_penalty = 0
      elif any(item in contact_ids for item in support_order):
        preferred_support_penalty = 1
    any_contact_penalty = 0 if contact_ids else 1
    phase_penalty = 0 if phase in {"seat", "through_stop"} else 1
    return (
        collision_count,
        preferred_support_penalty,
        any_contact_penalty,
        sign_penalty + phase_penalty,
        float(collision_overlap),
        abs(float(shift)),
    )

  def _support_groups_for_stack(
      self,
      carrier_part: PartInstance,
      placed_parts: list[PartInstance],
      relation_hint: str,
      stack_mode: str,
  ) -> list[list[PartInstance]]:
    if not placed_parts:
      return []
    axial_chain = relation_hint == "fasten" or stack_mode in {
        "hole_on_pin",
        "pin_into_hole",
    }
    if not axial_chain:
      return [placed_parts]
    carrier_group = [
        part
        for part in placed_parts
        if part.instance_id == carrier_part.instance_id
    ]
    groups: list[list[PartInstance]] = []
    if carrier_group:
      groups.append(carrier_group)
    if len(placed_parts) > len(carrier_group):
      groups.append(placed_parts)
    return groups or [placed_parts]

  def _nearest_bbox_contact_shift(
      self,
      unknown_part: PartInstance,
      blockers: list[PartInstance],
      axis_world: np.ndarray,
      preferred_sign: float = 0.0,
      preferred_carrier_id: Optional[str] = None,
  ) -> Optional[float]:
    unknown_interval = self._projected_bbox_interval(unknown_part, axis_world)
    if unknown_interval is None:
      return None
    unknown_min, unknown_max = unknown_interval
    best_shift: Optional[float] = None
    best_rank: Optional[tuple[int, int, float]] = None
    for blocker in blockers:
      blocker_interval = self._projected_bbox_interval(blocker, axis_world)
      if blocker_interval is None:
        continue
      if not self._perpendicular_bbox_overlap(
          part_a=unknown_part,
          part_b=blocker,
          axis_world=axis_world,
      ):
        continue
      blocker_min, blocker_max = blocker_interval
      candidates = [
          float(blocker_min - unknown_max),
          float(blocker_max - unknown_min),
      ]
      for shift in candidates:
        sign_penalty = 0
        if preferred_sign and float(shift) * preferred_sign < -self.tolerance:
          sign_penalty = 1
        carrier_penalty = 0
        if (
            preferred_carrier_id is not None
            and blocker.instance_id != preferred_carrier_id
        ):
          carrier_penalty = 1
        rank = (carrier_penalty, sign_penalty, abs(float(shift)))
        if best_rank is None or rank < best_rank:
          best_rank = rank
          best_shift = float(shift)
    return best_shift

  def _ray_cast_snap_shift(
      self,
      unknown_part: PartInstance,
      blockers: list[PartInstance],
      axis_world: np.ndarray,
      target_shift: float,
  ) -> Optional[float]:
    if abs(float(target_shift)) <= self.tolerance:
      return float(target_shift)
    direction = 1.0 if float(target_shift) >= 0.0 else -1.0
    step = max(0.1, min(float(self.axial_stack_step), abs(float(target_shift))))
    travelled = 0.0
    max_travel = abs(float(target_shift))
    last_free = 0.0
    while travelled + step < max_travel:
      travelled += step
      shift = direction * travelled
      if self._stack_contact_reached(
          unknown_part=unknown_part,
          blockers=blockers,
          axis_world=axis_world,
          shift=shift,
      ):
        lower = direction * last_free
        upper = shift
        for _ in range(8):
          mid = 0.5 * (lower + upper)
          if self._stack_contact_reached(
              unknown_part=unknown_part,
              blockers=blockers,
              axis_world=axis_world,
              shift=mid,
          ):
            upper = mid
          else:
            lower = mid
        return lower
      last_free = travelled
    return float(target_shift)

  def _stack_contact_reached(
      self,
      unknown_part: PartInstance,
      blockers: list[PartInstance],
      axis_world: np.ndarray,
      shift: float,
  ) -> bool:
    interval = self._projected_bbox_interval_with_shift(
        part=unknown_part,
        axis_world=axis_world,
        shift=float(shift),
    )
    if interval is None:
      return False
    unknown_min, unknown_max = interval
    for blocker in blockers:
      blocker_interval = self._projected_bbox_interval(blocker, axis_world)
      if blocker_interval is None:
        continue
      if not self._perpendicular_bbox_overlap(
          part_a=unknown_part,
          part_b=blocker,
          axis_world=axis_world,
      ):
        continue
      blocker_min, blocker_max = blocker_interval
      overlap = min(unknown_max, blocker_max) - max(unknown_min, blocker_min)
      if overlap >= -max(self.tolerance, 1e-3):
        return True
    return False

  def _shift_causes_stack_collision(
      self,
      unknown_part: PartInstance,
      blockers: list[PartInstance],
      axis_world: np.ndarray,
      shift: float,
      ignore_blockers: Optional[set[str]] = None,
  ) -> bool:
    moved_interval = self._projected_bbox_interval_with_shift(
        part=unknown_part,
        axis_world=axis_world,
        shift=float(shift),
    )
    if moved_interval is None:
      return False
    unknown_min, unknown_max = moved_interval
    ignored = ignore_blockers or set()
    for blocker in blockers:
      if blocker.instance_id in ignored:
        continue
      blocker_interval = self._projected_bbox_interval(blocker, axis_world)
      if blocker_interval is None:
        continue
      if not self._perpendicular_bbox_overlap(
          part_a=unknown_part,
          part_b=blocker,
          axis_world=axis_world,
      ):
        continue
      blocker_min, blocker_max = blocker_interval
      overlap = min(unknown_max, blocker_max) - max(unknown_min, blocker_min)
      if overlap > max(self.tolerance, 1e-3):
        return True
    return False

  def _projected_bbox_interval_with_shift(
      self,
      part: PartInstance,
      axis_world: np.ndarray,
      shift: float,
  ) -> Optional[tuple[float, float]]:
    interval = self._projected_bbox_interval(part, axis_world)
    if interval is None:
      return None
    return float(interval[0] + shift), float(interval[1] + shift)

  def _perpendicular_bbox_overlap(
      self,
      part_a: PartInstance,
      part_b: PartInstance,
      axis_world: np.ndarray,
  ) -> bool:
    if part_a.transform is None or part_b.transform is None:
      return False
    basis_u, basis_v = self._orthogonal_basis(axis_world)
    interval_a_u = self._projected_bbox_interval(part_a, basis_u)
    interval_a_v = self._projected_bbox_interval(part_a, basis_v)
    interval_b_u = self._projected_bbox_interval(part_b, basis_u)
    interval_b_v = self._projected_bbox_interval(part_b, basis_v)
    if None in (interval_a_u, interval_a_v, interval_b_u, interval_b_v):
      return False
    overlap_u = min(interval_a_u[1], interval_b_u[1]) - max(interval_a_u[0], interval_b_u[0])
    overlap_v = min(interval_a_v[1], interval_b_v[1]) - max(interval_a_v[0], interval_b_v[0])
    return overlap_u > -1e-3 and overlap_v > -1e-3

  def _orthogonal_basis(self, axis_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis_world)
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, trial))) > 0.9:
      trial = np.array([0.0, 1.0, 0.0], dtype=float)
    basis_u = normalize(np.cross(axis, trial))
    basis_v = normalize(np.cross(axis, basis_u))
    return basis_u, basis_v

  def _constraint_relation_hint(self, constraints: list[Constraint]) -> str:
    for constraint in constraints:
      hint = str(constraint.metadata.get("relation_hint", "")).strip().lower()
      if hint:
        return hint
    return ""

  def _constraint_stack_mode(self, constraints: list[Constraint]) -> str:
    for constraint in constraints:
      mode = str(constraint.metadata.get("stack_mode", "")).strip().lower()
      if mode:
        return mode
    return ""

  def _constraint_stack_phase(self, constraints: list[Constraint]) -> str:
    saw_entry = False
    for constraint in constraints:
      if bool(constraint.metadata.get("secondary_seat", False)):
        return "seat"
      phase = str(constraint.metadata.get("stack_phase", "")).strip().lower()
      if phase == "through_stop":
        return "through_stop"
      if phase == "seat":
        return "seat"
      if phase == "entry":
        saw_entry = True
    if saw_entry:
      return "entry"
    return ""

  def _preferred_stack_direction(
      self,
      carrier_part: PartInstance,
      unknown_part: PartInstance,
      axis_world: np.ndarray,
  ) -> float:
    carrier_interval = self._projected_bbox_interval(carrier_part, axis_world)
    unknown_interval = self._projected_bbox_interval(unknown_part, axis_world)
    if carrier_interval is None or unknown_interval is None:
      return 0.0
    carrier_center = 0.5 * float(carrier_interval[0] + carrier_interval[1])
    unknown_center = 0.5 * float(unknown_interval[0] + unknown_interval[1])
    delta = carrier_center - unknown_center
    if abs(delta) <= self.tolerance:
      return 0.0
    return 1.0 if delta > 0.0 else -1.0

  def _seat_constraints_satisfied(
      self,
      known_part: PartInstance,
      unknown_part: PartInstance,
      seat_constraints: list[Constraint],
  ) -> bool:
    if not seat_constraints:
      return False
    for constraint in seat_constraints:
      err = self._constraint_error(known_part, unknown_part, constraint)
      if err > max(self.tolerance * 20.0, 1e-3):
        return False
    return True

  def _projected_bbox_span(
      self,
      part: PartInstance,
      axis_world: np.ndarray,
  ) -> float:
    interval = self._projected_bbox_interval(part, axis_world)
    if interval is None:
      return 0.0
    return float(interval[1] - interval[0])

  def _projected_bbox_interval(
      self,
      part: PartInstance,
      axis_world: np.ndarray,
  ) -> Optional[tuple[float, float]]:
    if part.transform is None:
      return None
    mins = part.local_bbox_min
    maxs = part.local_bbox_max
    corners = np.array(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=float,
    )
    projections: list[float] = []
    for corner in corners:
      world = part.transform.apply_point(corner)
      projections.append(float(np.dot(world, axis_world)))
    return min(projections), max(projections)

  def _enforce_distance(
      self,
      part_a: PartInstance,
      part_b: PartInstance,
      constraint: Constraint,
      logs: list[str],
  ) -> bool:
    if constraint.value is None:
      return False
    frame_a = part_a.world_socket(constraint.socket_a)
    frame_b = part_b.world_socket(constraint.socket_b)
    direction = self._frame_normal_or_axis(frame_a)
    delta = frame_b.origin - frame_a.origin
    along = float(np.dot(delta, direction))
    shift = (float(constraint.value) - along) * direction
    if norm(shift) <= self.tolerance:
      return False
    if part_b.transform is None:
      return False
    part_b.transform = part_b.transform.translated(shift)
    logs.append(
        f"distance_adjust {part_b.instance_id} by {shift.round(6).tolist()}"
    )
    return True

  def _constraint_error(
      self, part_a: PartInstance, part_b: PartInstance, constraint: Constraint
  ) -> float:
    frame_a = part_a.world_socket(constraint.socket_a)
    frame_b = part_b.world_socket(constraint.socket_b)

    if constraint.ctype == ConstraintType.CONCENTRIC:
      axis_a = self._frame_axis_or_normal(frame_a)
      axis_b = self._frame_axis_or_normal(frame_b)
      axis_err = 1.0 - abs(float(np.dot(axis_a, axis_b)))
      delta = frame_b.origin - frame_a.origin
      radial = norm(delta - np.dot(delta, axis_a) * axis_a)
      radius_err = 0.0
      if frame_a.radius is not None and frame_b.radius is not None:
        socket_a = self._must_get_socket(part_a, constraint.socket_a)
        socket_b = self._must_get_socket(part_b, constraint.socket_b)
        if {socket_a.kind, socket_b.kind} == {"hole", "pin"}:
          hole = frame_a if socket_a.kind == "hole" else frame_b
          pin = frame_a if socket_a.kind == "pin" else frame_b
          shortfall = (pin.radius + self.radius_clearance) - hole.radius
          radius_err = max(0.0, shortfall)
        else:
          radius_err = abs(frame_a.radius - frame_b.radius)
      return float(radial + axis_err + radius_err)

    if constraint.ctype == ConstraintType.COINCIDENT:
      normal_a = self._frame_normal_or_axis(frame_a)
      normal_b = self._frame_normal_or_axis(frame_b)
      delta = frame_b.origin - frame_a.origin
      target = 0.0 if constraint.value is None else float(constraint.value)
      if bool(constraint.metadata.get("physics_guided_grounding", False)):
        # Physics-guided grounding may intentionally leave a nonzero axial
        # offset after 1D seat search. The geometric validity of that offset is
        # checked by collision and exact contact-distance validation, so the
        # symbolic validator should only ensure the local frames are oriented.
        target = float(np.dot(delta, normal_a))
      plane_sep = abs(float(np.dot(delta, normal_a)) - target)
      tangent = norm(delta - float(np.dot(delta, normal_a)) * normal_a)
      normal_err = 1.0 - abs(float(np.dot(normal_a, -normal_b)))
      alignment_mode = str(
          constraint.metadata.get("alignment_mode", "")
      ).strip().lower()
      relation_hint = str(
          constraint.metadata.get("relation_hint", "")
      ).strip().lower()
      if alignment_mode in {"support", "mate"} or relation_hint == "planar":
        tangential_weight = 0.0
      elif bool(constraint.metadata.get("physics_guided_grounding", False)):
        tangential_weight = 0.0
      elif bool(constraint.metadata.get("secondary_seat", False)):
        tangential_weight = 0.02
      else:
        tangential_weight = 0.05
      if bool(constraint.metadata.get("secondary_seat", False)):
        plane_sep = max(0.0, plane_sep - float(self.secondary_seat_clearance))
      return float(plane_sep + normal_err + tangential_weight * tangent)

    if constraint.ctype == ConstraintType.DISTANCE:
      direction = self._frame_normal_or_axis(frame_a)
      delta = frame_b.origin - frame_a.origin
      along = float(np.dot(delta, direction))
      target = 0.0 if constraint.value is None else float(constraint.value)
      perp = norm(delta - along * direction)
      return float(abs(along - target) + perp)

    return 1e9

  def _is_soft_secondary_seat(
      self,
      constraint: Constraint,
      err: float,
  ) -> bool:
    if not bool(constraint.metadata.get("secondary_seat", False)):
      return False
    soft_tol = float(self.secondary_seat_soft_tolerance)
    support_roles = {
        str(constraint.metadata.get("carrier_support_role") or "").strip().lower(),
        str(constraint.metadata.get("inserted_support_role") or "").strip().lower(),
    }
    if "through_stop" in support_roles:
      soft_tol = max(soft_tol, 14.0)
    return float(err) <= soft_tol

  def _maybe_morph_radii(
      self,
      known_part: PartInstance,
      known_socket_name: str,
      known_socket: Socket,
      unknown_part: PartInstance,
      unknown_socket_name: str,
      unknown_socket: Socket,
      logs: list[str],
  ) -> None:
    if known_socket.radius is None or unknown_socket.radius is None:
      return
    if {known_socket.kind, unknown_socket.kind} != {"hole", "pin"}:
      return

    if known_socket.kind == "hole":
      hole_part, hole_name, hole_socket = (
          known_part,
          known_socket_name,
          known_socket,
      )
      pin_part, pin_name, pin_socket = (
          unknown_part,
          unknown_socket_name,
          unknown_socket,
      )
    else:
      hole_part, hole_name, hole_socket = (
          unknown_part,
          unknown_socket_name,
          unknown_socket,
      )
      pin_part, pin_name, pin_socket = (
          known_part,
          known_socket_name,
          known_socket,
      )

    required_hole = pin_socket.radius + self.radius_clearance
    if hole_socket.radius >= required_hole:
      return

    old_hole_radius = hole_socket.radius
    if hole_part.try_set_socket_radius(hole_name, required_hole):
      logs.append(
          f"param_morph grow {hole_part.instance_id}.{hole_name} "
          f"{old_hole_radius:.4f}->{required_hole:.4f}"
      )
      return

    old_pin_radius = pin_socket.radius
    shrink_pin = max(hole_socket.radius - self.radius_clearance, self.tolerance)
    if pin_part.try_set_socket_radius(pin_name, shrink_pin):
      logs.append(
          f"param_morph shrink {pin_part.instance_id}.{pin_name} "
          f"{old_pin_radius:.4f}->{shrink_pin:.4f}"
      )
      return

    raise SolveError(
        "radius mismatch cannot be repaired: "
        f"hole={hole_socket.radius:.4f}, pin={pin_socket.radius:.4f}"
    )

  def _must_get_socket(self, part: PartInstance, socket_name: str) -> Socket:
    if socket_name not in part.sockets:
      raise SolveError(
          f"Socket '{socket_name}' not found in part '{part.instance_id}'."
      )
    return part.sockets[socket_name]

  def _frame_axis_or_normal(self, frame: SocketFrame) -> np.ndarray:
    if frame.axis is not None:
      return normalize(frame.axis)
    if frame.normal is not None:
      return normalize(frame.normal)
    return np.array([0.0, 0.0, 1.0], dtype=float)

  def _frame_normal_or_axis(self, frame: SocketFrame) -> np.ndarray:
    if frame.normal is not None:
      return normalize(frame.normal)
    if frame.axis is not None:
      return normalize(frame.axis)
    return np.array([0.0, 0.0, 1.0], dtype=float)

  def _socket_axis_or_normal(self, socket: Socket) -> np.ndarray:
    if socket.axis is not None:
      return normalize(socket.axis)
    if socket.normal is not None:
      return normalize(socket.normal)
    return np.array([0.0, 0.0, 1.0], dtype=float)

  def _socket_normal_or_axis(self, socket: Socket) -> np.ndarray:
    if socket.normal is not None:
      return normalize(socket.normal)
    if socket.axis is not None:
      return normalize(socket.axis)
    return np.array([0.0, 0.0, 1.0], dtype=float)

  def _ordered_frame_variant_names(
      self,
      *,
      available: dict[str, SocketFrame],
      preferred_hint: str,
      socket: Socket,
      relation_hint: str,
  ) -> list[str]:
    names: list[str] = []
    hint = preferred_hint.strip().lower()
    if hint and hint in available:
      names.append(hint)
    if socket.kind in {"hole", "threaded_hole"}:
      defaults = [
          "entry_plus",
          "entry_minus",
          "seat_plus",
          "seat_minus",
          "through_plus",
          "through_minus",
          "through_stop_plus",
          "through_stop_minus",
          "hole_depth_stop_plus",
          "hole_depth_stop_minus",
          "primary",
      ]
    elif socket.kind in {"pin", "axis"}:
      defaults = [
          "tip_plus",
          "tip_minus",
          "stop_plus",
          "stop_minus",
          "shoulder_plus",
          "shoulder_minus",
          "pin_shoulder_plus",
          "pin_shoulder_minus",
          "end_plus",
          "end_minus",
          "primary",
      ]
    else:
      defaults = ["support", "primary"]
    if relation_hint in {"coaxial", "fasten"}:
      defaults.extend([
          "through_plus",
          "through_minus",
          "through_stop_plus",
          "through_stop_minus",
          "hole_depth_stop_plus",
          "hole_depth_stop_minus",
          "stop_plus",
          "stop_minus",
          "shoulder_plus",
          "shoulder_minus",
          "pin_shoulder_plus",
          "pin_shoulder_minus",
          "end_plus",
          "end_minus",
          "primary",
      ])
    for name in defaults:
      if name in available and name not in names:
        names.append(name)
    for name in sorted(available):
      if name not in names:
        names.append(name)
    return names

  def _preferred_frame_hint(
      self,
      *,
      primary_hint: str,
      stop_hint: str,
      stack_phase: str,
  ) -> str:
    phase = stack_phase.strip().lower()
    if phase in {"seat", "through_stop"} and stop_hint.strip():
      return stop_hint.strip().lower()
    return primary_hint.strip().lower()

  def _frame_alignment_requires_opposed_z(
      self,
      known_socket: Socket,
      unknown_socket: Socket,
      constraint: Constraint,
  ) -> bool:
    if constraint.ctype == ConstraintType.COINCIDENT:
      return True
    alignment_mode = str(constraint.metadata.get("alignment_mode") or "").strip().lower()
    if alignment_mode in {"insert", "mate", "support", "through_stop"}:
      return True
    if known_socket.kind == "axis" and unknown_socket.kind == "axis":
      return False
    return True

  def _align_frame_pair(
      self,
      *,
      known_frame: SocketFrame,
      unknown_frame: SocketFrame,
      anti_parallel: bool,
      axial_offset: Optional[float],
  ) -> Transform:
    known_basis = known_frame.basis_matrix()
    unknown_basis = unknown_frame.basis_matrix()
    if known_basis is None or unknown_basis is None:
      raise SolveError("frame basis unavailable")
    desired_basis = known_basis.copy()
    if anti_parallel:
      desired_basis = desired_basis @ np.diag([1.0, -1.0, -1.0])
    rotation = desired_basis @ unknown_basis.T
    target_origin = known_frame.origin.copy()
    if axial_offset is not None:
      target_z = desired_basis[:, 2]
      target_origin = target_origin + target_z * float(axial_offset)
    translation = target_origin - rotation @ unknown_frame.origin
    return Transform(rotation=rotation, translation=translation)
