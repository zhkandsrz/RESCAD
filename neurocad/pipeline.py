"""Neuro-symbolic CAD orchestration pipeline with closed-loop correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .collision import CollisionChecker
from .planner import PlannerProtocol, RuleBasedNeuralPlanner
from .solver import SolveError, SymbolicCadSolver
from .domain_types import (
    AssemblyState,
    CollisionRecord,
    ConstraintGraph,
    ConstraintType,
    PartTemplate,
    PlannerFeedback,
)


@dataclass
class PipelineAttempt:
  iteration: int
  planner_notes: list[str]
  solver_logs: list[str]
  feedback_messages: list[str]
  tolerated_collision_messages: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, object]:
    return {
        "iteration": self.iteration,
        "planner_notes": list(self.planner_notes),
        "solver_logs": list(self.solver_logs),
        "feedback_messages": list(self.feedback_messages),
        "tolerated_collision_messages": list(self.tolerated_collision_messages),
    }


@dataclass
class PipelineResult:
  success: bool
  instruction: str
  attempts: list[PipelineAttempt] = field(default_factory=list)
  final_graph: Optional[ConstraintGraph] = None
  final_assembly: Optional[AssemblyState] = None
  final_feedback: list[str] = field(default_factory=list)
  final_tolerated_collisions: list[CollisionRecord] = field(default_factory=list)
  artifacts: dict[str, Any] = field(default_factory=dict, repr=False)

  def to_dict(self) -> dict[str, object]:
    return {
        "success": self.success,
        "instruction": self.instruction,
        "attempts": [attempt.to_dict() for attempt in self.attempts],
        "final_graph": None if self.final_graph is None else self.final_graph.to_dict(),
        "final_assembly": (
            None if self.final_assembly is None else self.final_assembly.to_dict()
        ),
        "final_feedback": list(self.final_feedback),
        "final_tolerated_collisions": [
            record.to_dict() for record in self.final_tolerated_collisions
        ],
    }


class NeuroSymbolicCadPipeline:
  """Full loop: plan -> symbolic solve -> collision check -> replan."""

  def __init__(
      self,
      planner: Optional[PlannerProtocol] = None,
      solver: Optional[SymbolicCadSolver] = None,
      collision_checker: Optional[CollisionChecker] = None,
      collision_epsilon: float = 1e-6,
      mate_collision_volume_tolerance: float = 50.0,
      threaded_interference_collision_tolerance: float = 500.0,
  ):
    self.planner = planner or RuleBasedNeuralPlanner()
    self.solver = solver or SymbolicCadSolver()
    self.collision_checker = collision_checker or CollisionChecker()
    self.collision_epsilon = collision_epsilon
    self.mate_collision_volume_tolerance = max(
        0.0, float(mate_collision_volume_tolerance)
    )
    self.threaded_interference_collision_tolerance = max(
        self.mate_collision_volume_tolerance,
        float(threaded_interference_collision_tolerance),
    )

  def run(
      self,
      instruction: str,
      catalog: dict[str, PartTemplate],
      max_iterations: int = 4,
      allow_morphing: bool = True,
  ) -> PipelineResult:
    feedback: Optional[PlannerFeedback] = None
    attempts: list[PipelineAttempt] = []
    last_graph: Optional[ConstraintGraph] = None
    last_feedback_messages: list[str] = []

    for iteration in range(max_iterations):
      try:
        graph = self.planner.plan(
            instruction=instruction, catalog=catalog, feedback=feedback
        )
      except Exception as exc:  # pylint: disable=broad-except
        message = f"planner_error={type(exc).__name__}:{exc}"
        attempts.append(
            PipelineAttempt(
                iteration=iteration,
                planner_notes=[],
                solver_logs=[],
                feedback_messages=[message],
            )
        )
        feedback = PlannerFeedback(messages=[message], collisions=[])
        last_feedback_messages = [message]
        continue
      last_graph = graph
      planner_notes = list(graph.metadata.get("planner_notes", []))

      try:
        assembly = self.solver.solve(graph, allow_morphing=allow_morphing)
      except SolveError as exc:
        message = f"solve_error={exc}"
        attempts.append(
            PipelineAttempt(
                iteration=iteration,
                planner_notes=planner_notes,
                solver_logs=[],
                feedback_messages=[message],
            )
        )
        feedback = PlannerFeedback(messages=[message], collisions=[])
        last_feedback_messages = [message]
        continue

      placement_feedback = self._unplaced_part_feedback(graph, assembly)
      if placement_feedback:
        attempts.append(
            PipelineAttempt(
                iteration=iteration,
                planner_notes=planner_notes,
                solver_logs=list(assembly.logs),
                feedback_messages=placement_feedback,
            )
        )
        feedback = PlannerFeedback(messages=placement_feedback, collisions=[])
        last_feedback_messages = placement_feedback
        continue

      try:
        collisions = self.collision_checker.check(
            assembly, epsilon=self.collision_epsilon
        )
      except Exception as exc:  # pylint: disable=broad-except
        message = f"validation_error={type(exc).__name__}:{exc}"
        attempts.append(
            PipelineAttempt(
                iteration=iteration,
                planner_notes=planner_notes,
                solver_logs=list(assembly.logs),
                feedback_messages=[message],
            )
        )
        feedback = PlannerFeedback(messages=[message], collisions=[])
        last_feedback_messages = [message]
        continue
      blocking_collisions, tolerated_collisions = self._partition_collisions(
          collisions, graph
      )
      tolerated_messages = [
          record.message() for record in tolerated_collisions
      ]
      if not blocking_collisions:
        attempts.append(
            PipelineAttempt(
                iteration=iteration,
                planner_notes=planner_notes,
                solver_logs=list(assembly.logs),
                feedback_messages=[],
                tolerated_collision_messages=tolerated_messages,
            )
        )
        return PipelineResult(
            success=True,
            instruction=instruction,
            attempts=attempts,
            final_graph=graph,
            final_assembly=assembly,
            final_feedback=[],
            final_tolerated_collisions=tolerated_collisions,
        )

      collision_messages = [record.message() for record in blocking_collisions]
      attempts.append(
          PipelineAttempt(
              iteration=iteration,
              planner_notes=planner_notes,
              solver_logs=list(assembly.logs),
              feedback_messages=collision_messages,
              tolerated_collision_messages=tolerated_messages,
          )
      )
      feedback = PlannerFeedback(
          messages=collision_messages,
          collisions=blocking_collisions,
      )
      last_feedback_messages = collision_messages

    return PipelineResult(
        success=False,
        instruction=instruction,
        attempts=attempts,
        final_graph=last_graph,
        final_assembly=None,
        final_feedback=last_feedback_messages,
    )

  def _partition_collisions(
      self,
      collisions: list[CollisionRecord],
      graph: ConstraintGraph,
  ) -> tuple[list[CollisionRecord], list[CollisionRecord]]:
    tolerated_pairs = self._tolerated_mate_pairs(graph)
    blocking: list[CollisionRecord] = []
    tolerated: list[CollisionRecord] = []
    for collision in collisions:
      pair = frozenset((collision.part_a, collision.part_b))
      tolerance = tolerated_pairs.get(pair, 0.0)
      if tolerance > 0.0 and collision.volume <= tolerance:
        collision.metadata["tolerated"] = True
        collision.metadata["tolerance_volume"] = tolerance
        collision.metadata.update(tolerated_pairs.get((pair, "metadata"), {}))
        tolerated.append(collision)
      else:
        blocking.append(collision)
    return blocking, tolerated

  def _tolerated_mate_pairs(
      self,
      graph: ConstraintGraph,
  ) -> dict[Any, Any]:
    if (
        self.mate_collision_volume_tolerance <= 0.0
        and self.threaded_interference_collision_tolerance <= 0.0
    ):
      return {}
    tolerated: dict[Any, Any] = {}
    for constraint in graph.constraints:
      contact_type = str(
          (constraint.metadata or {}).get("predicted_contact_type") or ""
      )
      collision_policy = str(
          (constraint.metadata or {}).get("collision_policy") or ""
      )
      instance_a = graph.instances.get(constraint.part_a)
      instance_b = graph.instances.get(constraint.part_b)
      if instance_a is None or instance_b is None:
        continue
      socket_a = instance_a.sockets.get(constraint.socket_a)
      socket_b = instance_b.sockets.get(constraint.socket_b)
      if socket_a is None or socket_b is None:
        continue
      pair_key = frozenset((constraint.part_a, constraint.part_b))
      if (
          contact_type == "threaded_interference"
          or collision_policy == "intentional_interference"
      ):
        if self.threaded_interference_collision_tolerance <= 0.0:
          continue
        tolerated[pair_key] = self.threaded_interference_collision_tolerance
        tolerated[(pair_key, "metadata")] = {
            "intentional_interference": True,
            "predicted_contact_type": contact_type or "threaded_interference",
            "collision_policy": "intentional_interference",
        }
        continue
      if constraint.ctype != ConstraintType.CONCENTRIC:
        continue
      if self.mate_collision_volume_tolerance <= 0.0:
        continue
      if self._is_tolerant_mate_socket_pair(socket_a.kind, socket_b.kind):
        tolerated[pair_key] = self.mate_collision_volume_tolerance
        tolerated[(pair_key, "metadata")] = {
            "predicted_contact_type": contact_type,
            "collision_policy": collision_policy,
        }
    return tolerated

  def _is_tolerant_mate_socket_pair(
      self,
      kind_a: str,
      kind_b: str,
  ) -> bool:
    pair = frozenset((str(kind_a), str(kind_b)))
    return pair in {
        frozenset(("hole", "pin")),
        frozenset(("threaded_hole", "pin")),
        frozenset(("hole", "axis")),
        frozenset(("threaded_hole", "axis")),
    }

  def _unplaced_part_feedback(
      self, graph: ConstraintGraph, assembly: AssemblyState
  ) -> list[str]:
    unplaced = [
        part_id
        for part_id, instance in assembly.instances.items()
        if instance.transform is None
    ]
    if not unplaced:
      return []

    connected = {graph.anchor}
    for constraint in graph.constraints:
      connected.add(constraint.part_a)
      connected.add(constraint.part_b)

    placed = sorted(
        part_id
        for part_id, instance in assembly.instances.items()
        if instance.transform is not None
    )
    placed_hint = ",".join(placed[:4]) if placed else "none"

    messages: list[str] = []
    for part_id in sorted(unplaced):
      if part_id not in connected:
        reason = "not_connected_in_constraint_graph"
      else:
        reason = "transform_not_solved"
      messages.append(
          "unplaced_part="
          f"{part_id};reason={reason};connect_to={placed_hint}"
      )
    return messages
