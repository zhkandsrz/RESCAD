"""Unified finite mate-program proposal interface.

The proposer is the single ranking boundary between symbolic candidate
generation and geometric execution.  Candidate generation may use the mined
train library, but benchmark-v2 formal ranking is performed only by a loaded
learned model and never silently falls back to retrieval order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from .domain_types import Socket
from .v19_program_proposal_protocol_v1 import ProposalBudget
from .interface_pair_features import pair_features_from_sockets
from .mate_pose_retriever import MateProgramRetrieval
from .mate_programs import MateProgram
from .program_class_reranker import (
    ProgramClassReranker,
    mate_program_class_from_program,
    role_radius_score,
)


PROGRAM_PROPOSER_PROTOCOL = "finite_program_proposer.v1"


@dataclass(frozen=True)
class ProgramHypothesis:
  """One traceable, executable program hypothesis."""

  program: MateProgram = field(repr=False, compare=False)
  score: float
  program_id: str
  program_class: str
  query_parent_interface_id: str
  query_child_interface_id: str
  residual_rotation: tuple[tuple[float, float, float], ...]
  residual_translation: tuple[float, float, float]
  score_components: dict[str, float]
  model_sha256: str
  retrieval_rank: int
  proposer_mode: str
  proposal_budget: dict[str, Any]
  proposer_protocol: str = PROGRAM_PROPOSER_PROTOCOL

  def to_retrieval(self) -> MateProgramRetrieval:
    proposed_rotation = np.asarray(self.residual_rotation, dtype=float).reshape(3, 3)
    proposed_translation = np.asarray(self.residual_translation, dtype=float).reshape(3)
    source_rotation = np.asarray(self.program.residual_rotation, dtype=float).reshape(3, 3)
    source_translation = np.asarray(self.program.residual_translation, dtype=float).reshape(3)
    if np.array_equal(proposed_rotation, source_rotation) and np.array_equal(
        proposed_translation, source_translation
    ):
      execution_program = self.program
    else:
      # Keep the authority/retrieval candidate immutable.  Only a genuinely
      # refined hypothesis receives a derived executable program.
      execution_program = replace(
          self.program,
          residual_rotation=proposed_rotation.tolist(),
          residual_translation=proposed_translation.tolist(),
          metadata={
              **dict(self.program.metadata),
              "execution_residual_source": "program_hypothesis.v1",
              "execution_model_sha256": self.model_sha256,
          },
      )
    return MateProgramRetrieval(
        program=execution_program,
        score=float(self.score),
        reasons=[
            "finite_program_proposer",
            f"proposer_protocol={self.proposer_protocol}",
            f"model_sha256={self.model_sha256}",
            f"retrieval_rank={int(self.retrieval_rank)}",
            f"proposer_mode={self.proposer_mode}",
        ],
        proposal_trace=self.trace_dict(),
    )

  def trace_dict(self) -> dict[str, Any]:
    return {
        "program_id": self.program_id,
        "program_class": self.program_class,
        "query_parent_interface_id": self.query_parent_interface_id,
        "query_child_interface_id": self.query_child_interface_id,
        "residual_rotation": [list(row) for row in self.residual_rotation],
        "residual_translation": list(self.residual_translation),
        "score": float(self.score),
        "score_components": dict(self.score_components),
        "model_sha256": self.model_sha256,
        "retrieval_rank": int(self.retrieval_rank),
        "proposer_mode": self.proposer_mode,
        "proposal_budget": dict(self.proposal_budget),
        "proposer_protocol": self.proposer_protocol,
    }


@runtime_checkable
class ProgramProposer(Protocol):
  """Contract shared by all finite-program proposal strategies."""

  mode: str
  model_sha256: str

  def propose(
      self,
      query: MateProgram,
      candidates: Sequence[MateProgramRetrieval],
      *,
      budget: ProposalBudget,
  ) -> list[ProgramHypothesis]: ...


class RetrievalProgramProposer:
  """Legacy baseline that preserves deterministic retrieval order."""

  mode = "retrieval"
  model_sha256 = "none"

  def propose(
      self,
      query: MateProgram,
      candidates: Sequence[MateProgramRetrieval],
      *,
      budget: ProposalBudget,
  ) -> list[ProgramHypothesis]:
    pool = list(candidates)[: int(budget.candidate_pool_size)]
    return [
        _hypothesis(
            query=query,
            row=row,
            score=float(row.score),
            score_components={"retrieval_score": float(row.score)},
            model_sha256=self.model_sha256,
            retrieval_rank=rank,
            proposer_mode=self.mode,
            budget=budget,
        )
        for rank, row in enumerate(pool[: int(budget.top_k)])
    ]


class HeuristicProgramProposer:
  """Matched-pool non-learning baseline using fixed role/radius compatibility."""

  mode = "heuristic"
  model_sha256 = "none"

  def propose(
      self,
      query: MateProgram,
      candidates: Sequence[MateProgramRetrieval],
      *,
      budget: ProposalBudget,
  ) -> list[ProgramHypothesis]:
    pool = list(candidates)[: int(budget.candidate_pool_size)]
    scored = [
        (float(role_radius_score(query, row.program)), rank, row)
        for rank, row in enumerate(pool)
    ]
    scored.sort(key=lambda item: (-item[0], item[2].program.program_id))
    return [
        _hypothesis(
            query=query,
            row=row,
            score=score,
            score_components={"fixed_role_radius_score": score},
            model_sha256=self.model_sha256,
            retrieval_rank=retrieval_rank,
            proposer_mode=self.mode,
            budget=budget,
        )
        for score, retrieval_rank, row in scored[: int(budget.top_k)]
    ]


class RetrievalDiversityProgramProposer:
  """Strong retrieval baseline with unique program classes before backfill."""

  mode = "retrieval_diversity"
  model_sha256 = "none"

  def propose(
      self,
      query: MateProgram,
      candidates: Sequence[MateProgramRetrieval],
      *,
      budget: ProposalBudget,
  ) -> list[ProgramHypothesis]:
    pool = list(candidates)[: int(budget.candidate_pool_size)]
    ranked = sorted(
        enumerate(pool),
        key=lambda item: (-float(item[1].score), item[1].program.program_id),
    )
    unique: list[tuple[int, MateProgramRetrieval]] = []
    deferred: list[tuple[int, MateProgramRetrieval]] = []
    seen_classes: set[str] = set()
    for retrieval_rank, row in ranked:
      program_class = mate_program_class_from_program(row.program)
      target = deferred if program_class in seen_classes else unique
      target.append((retrieval_rank, row))
      seen_classes.add(program_class)
    selected = (unique + deferred)[: int(budget.top_k)]
    return [
        _hypothesis(
            query=query,
            row=row,
            score=float(row.score),
            score_components={
                "retrieval_score": float(row.score),
                "class_diversity_first": float(
                    rank < len(unique)
                ),
            },
            model_sha256=self.model_sha256,
            retrieval_rank=retrieval_rank,
            proposer_mode=self.mode,
            budget=budget,
        )
        for rank, (retrieval_rank, row) in enumerate(selected)
    ]


class LearnedProgramProposer:
  """Rank a fixed symbolic pool using only a learned reranker score."""

  mode = "learned"

  def __init__(self, reranker: ProgramClassReranker) -> None:
    self.reranker = reranker
    self.model_sha256 = str(reranker.artifact_sha256 or "")
    if len(self.model_sha256) != 64:
      raise ValueError("learned proposer requires a hashed model artifact")

  def propose(
      self,
      query: MateProgram,
      candidates: Sequence[MateProgramRetrieval],
      *,
      budget: ProposalBudget,
  ) -> list[ProgramHypothesis]:
    pool = list(candidates)[: int(budget.candidate_pool_size)]
    scored: list[tuple[float, int, MateProgramRetrieval]] = []
    for retrieval_rank, row in enumerate(pool):
      score = float(
          self.reranker.score(
              query,
              row.program,
              # A formal learned proposer receives a fixed pool but no
              # retrieval-derived ranking features.
              base_score=(
                  0.0
                  if self.reranker.formal_validated
                  else float(row.score)
              ),
              base_rank=(0 if self.reranker.formal_validated else retrieval_rank),
          )
      )
      if not math.isfinite(score):
        raise ValueError("learned program proposer produced a non-finite score")
      scored.append((score, retrieval_rank, row))
    scored.sort(
        key=lambda item: (
            -float(item[0]),
            item[2].program.program_id,
        )
    )
    return [
        _hypothesis(
            query=query,
            row=row,
            score=score,
            score_components={"learned_score": score},
            model_sha256=self.model_sha256,
            retrieval_rank=retrieval_rank,
            proposer_mode=self.mode,
            budget=budget,
        )
        for score, retrieval_rank, row in scored[: int(budget.top_k)]
    ]


def query_program_from_sockets(
    *,
    parent_socket: Socket,
    child_socket: Socket,
    relation_hint: str,
    contact_type: str,
    protocol: str,
) -> MateProgram:
  """Build the label-free runtime query consumed by a proposer."""

  normalized = str(protocol or "legacy").strip().lower()
  if normalized not in {"legacy", "benchmark_v2"}:
    raise ValueError("query program protocol must be legacy or benchmark_v2")
  parent = _socket_endpoint(parent_socket, protocol=normalized)
  child = _socket_endpoint(child_socket, protocol=normalized)
  features = pair_features_from_sockets(
      parent_socket,
      child_socket,
      relation_hint=relation_hint,
      protocol=normalized,
  )
  identity_payload = {
      "parent": parent,
      "child": child,
      "relation_hint": str(relation_hint or ""),
      "contact_type": str(contact_type or ""),
      "protocol": normalized,
  }
  query_hash = _canonical_sha256(identity_payload)
  return MateProgram(
      program_id=f"query_{query_hash[:20]}",
      case_id="",
      assembly_dir="",
      part_a="",
      part_b="",
      body_uuid_a="",
      body_uuid_b="",
      interface_a=parent,
      interface_b=child,
      relation_hint=str(relation_hint or ""),
      contact_type=str(contact_type or ""),
      residual_rotation=np.eye(3, dtype=float).tolist(),
      residual_translation=[0.0, 0.0, 0.0],
      source_contact={},
      features={str(key): float(value) for key, value in features.items()},
      metadata={"model_input_protocol": normalized, "label_free_query": True},
  )


def _socket_endpoint(socket: Socket, *, protocol: str) -> dict[str, Any]:
  metadata = socket.metadata if isinstance(socket.metadata, dict) else {}
  score = metadata.get("interface_score", metadata.get("quality", 0.0))
  score = float(score) if isinstance(score, (int, float)) else 0.0
  if protocol == "benchmark_v2":
    if metadata.get("model_input_protocol") != "benchmark_v2":
      raise ValueError("benchmark_v2 proposer socket lacks protocol binding")
    view = metadata.get("benchmark_v2_model_view")
    view_hash = metadata.get("benchmark_v2_model_view_sha256")
    if not isinstance(view, dict) or not isinstance(view_hash, str):
      raise ValueError("benchmark_v2 proposer socket lacks sanitized model view")
    return {
        "model_input_protocol": "benchmark_v2",
        "benchmark_v2_model_view": view,
        "benchmark_v2_model_view_sha256": view_hash,
        "score": score,
    }
  return {
      "role_hint": str(metadata.get("interface_role") or socket.kind or ""),
      "surface_type": str(metadata.get("surface_type") or ""),
      "score": score,
      "metadata": {
          "radius": None if socket.radius is None else float(socket.radius),
      },
  }


def _hypothesis(
    *,
    query: MateProgram,
    row: MateProgramRetrieval,
    score: float,
    score_components: dict[str, float],
    model_sha256: str,
    retrieval_rank: int,
    proposer_mode: str,
    budget: ProposalBudget,
) -> ProgramHypothesis:
  rotation = np.asarray(row.program.residual_rotation, dtype=float).reshape(3, 3)
  translation = np.asarray(row.program.residual_translation, dtype=float).reshape(3)
  if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
    raise ValueError("program hypothesis residual transform is non-finite")
  return ProgramHypothesis(
      program=row.program,
      score=float(score),
      program_id=str(row.program.program_id),
      program_class=mate_program_class_from_program(row.program),
      query_parent_interface_id=_canonical_sha256(query.interface_a)[:24],
      query_child_interface_id=_canonical_sha256(query.interface_b)[:24],
      residual_rotation=tuple(
          tuple(float(value) for value in rotation_row) for rotation_row in rotation
      ),
      residual_translation=tuple(float(value) for value in translation),
      score_components={str(key): float(value) for key, value in score_components.items()},
      model_sha256=str(model_sha256),
      retrieval_rank=int(retrieval_rank),
      proposer_mode=str(proposer_mode),
      proposal_budget=budget.trace_dict(),
  )


def _canonical_sha256(payload: Any) -> str:
  encoded = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()
