"""Gold-free analytic-surface compatibility for finite-program Top-K.

The frozen 19-program roster states an ordered analytic surface type for each
endpoint.  This module uses only that public roster and the model-visible V5
safe graph one-hot features.  It never accepts labels, endpoint authorities,
OCC measurements, certificates, or transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark_v3_unlabeled_step_view import (
    INTRINSIC_NODE_FEATURE_NAMES,
    SURFACE_TYPES,
    BenchmarkV3UnlabeledStepView,
    UnlabeledIntrinsicBRepGraph,
    require_unlabeled_step_model_input,
)
from .joint_interface_program_learner_v2 import FiniteProgramCatalogRosterV2
from .constraint_catalog_trust_root_v1 import (
    OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1,
)


SCHEMA_VERSION = "semantic_surface_compatible_topk_decoder.v1"
COMPATIBILITY_SCHEMA_VERSION = "semantic_surface_compatibility.v1"
_SURFACE_OFFSET = INTRINSIC_NODE_FEATURE_NAMES.index("surface_type_plane")
_EXPLICIT_ROSTER_SURFACES = frozenset(SURFACE_TYPES) - {"other"}


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
      allow_nan=False,
  ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticRankedJointCandidateV1:
  score: float
  face_index_a: int
  face_index_b: int
  program_index: int
  residual: tuple[float, ...]

  def __post_init__(self) -> None:
    if not math.isfinite(float(self.score)):
      raise ValueError("semantic candidate score is non-finite")
    if any(type(value) is not int or value < 0 for value in (
        self.face_index_a, self.face_index_b, self.program_index,
    )):
      raise ValueError("semantic candidate identity differs")
    if len(self.residual) != 6 or any(
        not math.isfinite(float(value)) for value in self.residual
    ):
      raise ValueError("semantic candidate residual differs")

  @property
  def identity(self) -> tuple[int, int, int]:
    return self.face_index_a, self.face_index_b, self.program_index

  def commitment_payload(self) -> dict[str, Any]:
    return {
        "score": float(self.score),
        "face_index_a": self.face_index_a,
        "face_index_b": self.face_index_b,
        "program_index": self.program_index,
        "residual": [float(value) for value in self.residual],
    }


@dataclass(frozen=True, slots=True)
class MaskedSemanticCandidateV1:
  identity: tuple[int, int, int]
  reason: str
  observed_surface_type_a: str | None
  observed_surface_type_b: str | None
  expected_surface_type_a: str | None
  expected_surface_type_b: str | None


@dataclass(frozen=True, slots=True)
class SemanticSurfaceCompatibleTopKResultV1:
  selected: tuple[SemanticRankedJointCandidateV1, ...]
  compatible_ranked: tuple[SemanticRankedJointCandidateV1, ...]
  masked: tuple[MaskedSemanticCandidateV1, ...]
  masked_reason_counts: Mapping[str, int]
  top_k: int
  shortfall: int
  candidates_evaluated: int
  compatible_candidates: int
  candidate_pool_commitment_sha256: str
  catalog_sha256: str
  safe_input_sha256: str
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != SCHEMA_VERSION:
      raise ValueError("semantic decoder schema differs")
    if self.shortfall != self.top_k - len(self.selected) or self.shortfall < 0:
      raise ValueError("semantic decoder shortfall differs")
    if self.selected != self.compatible_ranked[:self.top_k]:
      raise ValueError("semantic decoder selected prefix differs")
    if self.compatible_candidates + len(self.masked) != self.candidates_evaluated:
      raise ValueError("semantic decoder candidate ledger differs")
    object.__setattr__(
        self, "masked_reason_counts", MappingProxyType(dict(self.masked_reason_counts))
    )


def surface_type_for_graph_face_v1(
    graph: UnlabeledIntrinsicBRepGraph, face_index: int,
) -> str:
  """Decode exactly one model-visible surface one-hot; ambiguity fails closed."""

  if type(graph) is not UnlabeledIntrinsicBRepGraph:
    raise TypeError("surface decoder requires a V5 safe graph")
  if type(face_index) is not int or not 0 <= face_index < graph.node_features.shape[0]:
    raise ValueError("surface decoder face index is outside the safe graph")
  values = np.asarray(
      graph.node_features[face_index, _SURFACE_OFFSET:_SURFACE_OFFSET + len(SURFACE_TYPES)],
      dtype=np.float64,
  )
  active = np.flatnonzero(np.isclose(values, 1.0, atol=1e-6, rtol=0.0))
  inactive = np.delete(values, active)
  if (
      active.size != 1
      or not np.all(np.isclose(inactive, 0.0, atol=1e-6, rtol=0.0))
  ):
    raise ValueError("surface decoder requires a unique allowlisted one-hot")
  return SURFACE_TYPES[int(active[0])]


def _compatibility(
    candidate: SemanticRankedJointCandidateV1, *,
    safe_view: BenchmarkV3UnlabeledStepView,
    catalog: FiniteProgramCatalogRosterV2,
) -> tuple[bool, MaskedSemanticCandidateV1 | None]:
  try:
    entry = catalog.entry(candidate.program_index)
  except ValueError as error:
    raise ValueError("semantic candidate program index differs from roster") from error
  descriptor = entry.descriptor
  expected = (descriptor.surface_type_a, descriptor.surface_type_b)
  if any(value not in _EXPLICIT_ROSTER_SURFACES for value in expected):
    return False, MaskedSemanticCandidateV1(
        identity=candidate.identity, reason="unsupported_catalog_surface_type",
        observed_surface_type_a=None, observed_surface_type_b=None,
        expected_surface_type_a=expected[0], expected_surface_type_b=expected[1],
    )
  try:
    observed = (
        surface_type_for_graph_face_v1(safe_view.graphs[0], candidate.face_index_a),
        surface_type_for_graph_face_v1(safe_view.graphs[1], candidate.face_index_b),
    )
  except ValueError:
    return False, MaskedSemanticCandidateV1(
        identity=candidate.identity, reason="unknown_endpoint_surface_type",
        observed_surface_type_a=None, observed_surface_type_b=None,
        expected_surface_type_a=expected[0], expected_surface_type_b=expected[1],
    )
  if observed != expected:
    return False, MaskedSemanticCandidateV1(
        identity=candidate.identity, reason="endpoint_surface_type_mismatch",
        observed_surface_type_a=observed[0], observed_surface_type_b=observed[1],
        expected_surface_type_a=expected[0], expected_surface_type_b=expected[1],
    )
  return True, None


def semantic_surface_compatible_v1(
    candidate: SemanticRankedJointCandidateV1, *,
    safe_view: BenchmarkV3UnlabeledStepView,
    catalog: FiniteProgramCatalogRosterV2,
) -> bool:
  """Public predicate shared by matched ranking and exact proposal materialization."""

  if type(candidate) is not SemanticRankedJointCandidateV1:
    raise TypeError("semantic predicate requires a ranked joint candidate")
  require_unlabeled_step_model_input(safe_view)
  if type(catalog) is not FiniteProgramCatalogRosterV2:
    raise TypeError("semantic predicate requires the exact frozen roster")
  if (
      catalog.catalog_sha256
      != OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1["catalog_roster_sha256"]
  ):
    raise ValueError("semantic predicate catalog differs from official roster")
  return _compatibility(candidate, safe_view=safe_view, catalog=catalog)[0]


def decode_semantic_surface_compatible_topk_v1(
    candidates: Sequence[SemanticRankedJointCandidateV1], *,
    safe_view: BenchmarkV3UnlabeledStepView,
    catalog: FiniteProgramCatalogRosterV2,
    top_k: int,
) -> SemanticSurfaceCompatibleTopKResultV1:
  """Mask incompatible joints, preserving scores and refilling K from deeper rank."""

  require_unlabeled_step_model_input(safe_view)
  if type(catalog) is not FiniteProgramCatalogRosterV2:
    raise TypeError("semantic decoder requires the exact frozen roster")
  if (
      catalog.catalog_sha256
      != OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1["catalog_roster_sha256"]
  ):
    raise ValueError("semantic decoder catalog differs from official roster")
  if type(top_k) is not int or top_k < 1:
    raise ValueError("semantic decoder Top-K differs")
  frozen = tuple(candidates)
  if any(type(row) is not SemanticRankedJointCandidateV1 for row in frozen):
    raise TypeError("semantic decoder candidate domain differs")
  identities = [row.identity for row in frozen]
  if len(set(identities)) != len(identities):
    raise ValueError("semantic decoder candidate identities repeat")
  ranked = tuple(sorted(
      frozen,
      key=lambda row: (-float(row.score), row.face_index_a, row.face_index_b, row.program_index),
  ))
  commitment = _canonical_sha256({
      "schema_version": "semantic_surface_candidate_pool_commitment.v1",
      "decoder_schema_version": SCHEMA_VERSION,
      "compatibility_schema_version": COMPATIBILITY_SCHEMA_VERSION,
      "safe_input_sha256": safe_view.input_sha256,
      "catalog_sha256": catalog.catalog_sha256,
      "top_k": top_k,
      "ranked_candidates": [row.commitment_payload() for row in ranked],
  })
  compatible: list[SemanticRankedJointCandidateV1] = []
  masked: list[MaskedSemanticCandidateV1] = []
  reason_counts: dict[str, int] = {}
  for candidate in ranked:
    accepted, rejection = _compatibility(
        candidate, safe_view=safe_view, catalog=catalog,
    )
    if accepted:
      compatible.append(candidate)
    else:
      assert rejection is not None
      masked.append(rejection)
      reason_counts[rejection.reason] = reason_counts.get(rejection.reason, 0) + 1
  return SemanticSurfaceCompatibleTopKResultV1(
      selected=tuple(compatible[:top_k]), compatible_ranked=tuple(compatible),
      masked=tuple(masked),
      masked_reason_counts=reason_counts, top_k=top_k,
      shortfall=max(0, top_k - len(compatible)),
      candidates_evaluated=len(ranked), compatible_candidates=len(compatible),
      candidate_pool_commitment_sha256=commitment,
      catalog_sha256=catalog.catalog_sha256,
      safe_input_sha256=safe_view.input_sha256,
  )


def decode_joint_output_semantic_surface_topk_v1(
    output: Any, *, safe_view: BenchmarkV3UnlabeledStepView,
    catalog: FiniteProgramCatalogRosterV2, top_l: int, top_k: int,
) -> SemanticSurfaceCompatibleTopKResultV1:
  """Flatten the complete Joint-V3 beam/catalog domain into the shared decoder."""

  require_unlabeled_step_model_input(safe_view)
  if type(top_l) is not int or top_l < 1:
    raise ValueError("semantic Joint decoder face beam differs")
  selected = output.selected_face_indices[0].cpu()
  logits = output.inference_joint_logits[0].cpu()
  if selected.ndim != 2 or selected.shape[0] != 2 or selected.shape[1] < top_l:
    raise ValueError("semantic Joint decoder selected-face domain differs")
  if logits.ndim != 3 or logits.shape[0] < top_l or logits.shape[1] < top_l:
    raise ValueError("semantic Joint decoder logit domain differs")
  program_count = int(logits.shape[2])
  if program_count != len(catalog.entries):
    raise ValueError("semantic Joint decoder program pool differs from roster")
  candidates: list[SemanticRankedJointCandidateV1] = []
  for rank_a in range(top_l):
    face_a = int(selected[0, rank_a])
    if face_a < 0:
      continue
    for rank_b in range(top_l):
      face_b = int(selected[1, rank_b])
      if face_b < 0:
        continue
      for position in range(program_count):
        program_index = int(output.candidate_program_indices[0, position])
        residual = tuple(float(value) for value in (
            *output.inference_residual_translation_local[
                0, rank_a, rank_b, position
            ].cpu(),
            *output.inference_residual_rotation_vector_local[
                0, rank_a, rank_b, position
            ].cpu(),
        ))
        candidates.append(SemanticRankedJointCandidateV1(
            score=float(logits[rank_a, rank_b, position]),
            face_index_a=face_a, face_index_b=face_b,
            program_index=program_index, residual=residual,
        ))
  return decode_semantic_surface_compatible_topk_v1(
      candidates, safe_view=safe_view, catalog=catalog, top_k=top_k,
  )


__all__ = [
    "COMPATIBILITY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "MaskedSemanticCandidateV1",
    "SemanticRankedJointCandidateV1",
    "SemanticSurfaceCompatibleTopKResultV1",
    "decode_semantic_surface_compatible_topk_v1",
    "decode_joint_output_semantic_surface_topk_v1",
    "semantic_surface_compatible_v1",
    "surface_type_for_graph_face_v1",
]
