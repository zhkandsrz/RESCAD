"""Dependency-minimal proposal budget shared by training and execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class ProposalBudget:
  top_k: int
  candidate_pool_size: int
  max_occ_calls: int | None = None
  wall_time_seconds: float | None = None

  def __post_init__(self) -> None:
    if type(self.top_k) is not int or self.top_k < 1:
      raise ValueError("proposal top_k must be a positive actual integer")
    if (
        type(self.candidate_pool_size) is not int
        or self.candidate_pool_size < self.top_k
    ):
      raise ValueError("candidate_pool_size must be at least top_k")
    if self.max_occ_calls is not None and (
        type(self.max_occ_calls) is not int or self.max_occ_calls < 0
    ):
      raise ValueError("proposal max_occ_calls must be nonnegative")
    if self.wall_time_seconds is not None and (
        not math.isfinite(float(self.wall_time_seconds))
        or float(self.wall_time_seconds) <= 0.0
    ):
      raise ValueError("proposal wall_time_seconds must be finite and positive")

  def trace_dict(self) -> dict[str, Any]:
    return {
        "top_k": self.top_k,
        "candidate_pool_size": self.candidate_pool_size,
        "max_occ_calls": self.max_occ_calls,
        "wall_time_seconds": self.wall_time_seconds,
    }
