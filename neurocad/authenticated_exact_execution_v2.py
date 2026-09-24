"""Development-only contact-feasibility interfaces for matched Top-K rows.

Proposal materialization is deliberately gold-blind.  Target interface and
program labels enter only through :func:`score_executed_topk_v2`. Geometry is
authorized only after the categorical roster is joined to the evidence-backed
constraint catalog; unsupported descriptors stop before STEP loading or OCC.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .domain_types import Transform
from .benchmark_v2_constants import exact_face_binding_sha256
from .benchmark_v2_model_view_v2 import _load_captured_step_shape
from .cadquery_backend import (
    CadKernelChildProcessError,
    CadKernelError,
    CadQueryCollisionChecker,
    shape_bbox,
)
from .domain_types import AssemblyState, ConstraintGraph, PartTemplate, Socket
from .full_graph_face_frame_authority_v1 import _matmul, _matvec
from .joint_interface_program_learner_v2 import (
    PAIR_FRAME_POLICY_SHA256,
    FiniteProgramDescriptorV2,
)
from .constraint_executable_program_catalog_v1 import (
    AppliedConstraintRefinementV1,
    AuthenticatedConstraintExecutableProgramCatalogV1,
    ProgramConstraintCertificateV1,
    SurfaceConstraintFrameV1,
    UnsupportedProgramSemantics,
    execute_surface_constraint_v1,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BLIND_SAFE_VIEW_PROPOSAL_V4_FACTORY_TOKEN = object()


class FiniteProgramBaseSE3SchemaMissing(ValueError):
  """Raised when a categorical program is mistaken for executable SE(3)."""


class SurfaceTypeMismatch(ValueError):
  """The blind face selection is incompatible with the finite program."""


def _canonical_sha256(value: Any) -> str:
  encoded = json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ExactExecutionBudgetV2:
  top_k: int = 5
  max_occ_calls: int = 10
  query_wall_time_seconds: float = 60.0
  per_occ_timeout_seconds: float = 8.0
  contact_tolerance_mm: float = 0.1
  interference_epsilon_mm3: float = 1e-7

  def __post_init__(self) -> None:
    if type(self.top_k) is not int or self.top_k < 1:
      raise ValueError("exact Top-K must be a positive integer")
    if type(self.max_occ_calls) is not int or self.max_occ_calls < 2 * self.top_k:
      raise ValueError("exact budget must allow two geometry calls per candidate")
    for value in (
        self.query_wall_time_seconds, self.per_occ_timeout_seconds,
        self.contact_tolerance_mm, self.interference_epsilon_mm3,
    ):
      if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError("exact geometry thresholds and deadlines must be finite and positive")

  def payload(self) -> dict[str, Any]:
    return {
        "top_k": self.top_k,
        "max_occ_calls": self.max_occ_calls,
        "query_wall_time_seconds": float(self.query_wall_time_seconds),
        "per_occ_timeout_seconds": float(self.per_occ_timeout_seconds),
        "contact_tolerance_mm": float(self.contact_tolerance_mm),
        "interference_epsilon_mm3": float(self.interference_epsilon_mm3),
    }


@dataclass(slots=True)
class ExactOccCallLedgerV3:
  """Separate budget reservations from actual isolated OCC child calls."""

  reserved: int = 0
  attempted: int = 0
  executed: int = 0
  broadphase_skipped: int = 0
  _child_receipts: list[dict[str, Any]] | None = None
  _terminal_attempt_receipts: list[dict[str, Any]] | None = None

  def __post_init__(self) -> None:
    if self._child_receipts is None:
      self._child_receipts = []
    if self._terminal_attempt_receipts is None:
      self._terminal_attempt_receipts = []
    self._validate_counts()

  def _validate_counts(self) -> None:
    values = (self.reserved, self.attempted, self.executed, self.broadphase_skipped)
    if any(type(value) is not int or value < 0 for value in values):
      raise ValueError("OCC call ledger counts must be non-negative integers")
    if self.executed > self.attempted or self.attempted + self.broadphase_skipped > self.reserved:
      raise ValueError("OCC call ledger transition counts are inconsistent")

  def reserve(self, count: int = 1) -> None:
    if type(count) is not int or count < 1:
      raise ValueError("OCC call reservation must be positive")
    self.reserved += count

  def record_attempt(self) -> None:
    if self.attempted + self.broadphase_skipped >= self.reserved:
      raise ValueError("OCC call attempt lacks a reservation")
    self.attempted += 1

  def record_broadphase_skip(self) -> None:
    if self.attempted + self.broadphase_skipped >= self.reserved:
      raise ValueError("OCC broadphase skip lacks a reservation")
    self.broadphase_skipped += 1

  def record_executed_child(self, receipt: Mapping[str, Any]) -> None:
    if self.executed >= self.attempted:
      raise ValueError("OCC child completion lacks an attempted call")
    required = {
        "schema_version", "call_id", "operation", "status", "exit_code",
        "result_payload_sha256", "receipt_payload_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
      raise ValueError("OCC child receipt schema differs")
    unsigned = dict(receipt)
    observed = unsigned.pop("receipt_payload_sha256")
    if (
        receipt["schema_version"] != "exact_occ_child_receipt.v1"
        or receipt["operation"] not in {"intended_interface_distance", "whole_solid_interference"}
        or receipt["status"] not in {"ok", "error", "timeout", "native_exit"}
        or type(receipt["exit_code"]) is not int
        or _SHA256.fullmatch(str(receipt["result_payload_sha256"])) is None
        or observed != _canonical_sha256(unsigned)
    ):
      raise ValueError("OCC child receipt binding differs")
    assert self._child_receipts is not None
    self._child_receipts.append(dict(receipt))
    self.executed += 1
    self.record_terminal_attempt(
        call_id=str(receipt["call_id"]), operation=str(receipt["operation"]),
        status=str(receipt["status"]), exit_code=int(receipt["exit_code"]),
        detail="child_receipt_authenticated",
    )

  def record_terminal_attempt(
      self, *, call_id: str, operation: str, status: str, exit_code: int,
      detail: str,
  ) -> None:
    if status not in {"ok", "error", "timeout", "native_exit", "pre_spawn"}:
      raise ValueError("OCC terminal attempt status differs")
    assert self._terminal_attempt_receipts is not None
    if len(self._terminal_attempt_receipts) >= self.attempted:
      raise ValueError("OCC terminal attempt lacks a pending attempt")
    receipt = {
        "schema_version": "exact_occ_attempt_terminal_receipt.v1",
        "call_id": call_id, "operation": operation, "status": status,
        "exit_code": int(exit_code), "detail": str(detail),
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    self._terminal_attempt_receipts.append(receipt)

  def payload(self) -> dict[str, Any]:
    self._validate_counts()
    assert self._child_receipts is not None
    assert self._terminal_attempt_receipts is not None
    if len(self._terminal_attempt_receipts) != self.attempted:
      raise ValueError("OCC attempted call lacks a terminal receipt")
    return {
        "reserved": self.reserved,
        "attempted": self.attempted,
        "executed": self.executed,
        "broadphase_skipped": self.broadphase_skipped,
        "child_receipts": tuple(dict(row) for row in self._child_receipts),
        "child_receipts_domain_sha256": _canonical_sha256(self._child_receipts),
        "terminal_attempt_receipts": tuple(
            dict(row) for row in self._terminal_attempt_receipts
        ),
        "terminal_attempt_receipts_domain_sha256": _canonical_sha256(
            self._terminal_attempt_receipts
        ),
    }


def _occ_child_receipt_v1(
    *, call_id: str, operation: str, status: str, exit_code: int,
    result_payload: Mapping[str, Any],
) -> dict[str, Any]:
  receipt = {
      "schema_version": "exact_occ_child_receipt.v1",
      "call_id": call_id,
      "operation": operation,
      "status": status,
      "exit_code": exit_code,
      "result_payload_sha256": _canonical_sha256(dict(result_payload)),
  }
  receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
  return receipt


def _observed_occ_call_v1(
    ledger: ExactOccCallLedgerV3,
    *, call_id: str, operation: str, invoke: Callable[[], Any],
) -> Any:
  """Record one parent-observed isolated OCC child attempt and completion."""

  ledger.record_attempt()
  try:
    result = invoke()
  except CadKernelChildProcessError as error:
    payload = {
        "error": str(error.error),
        "terminal_status": error.status,
    }
    ledger.record_executed_child(_occ_child_receipt_v1(
        call_id=call_id,
        operation=operation,
        status=error.status,
        exit_code=error.exit_code,
        result_payload=payload,
    ))
    raise
  except Exception as error:
    # Be conservative: the public checker can reject before spawning its OCC
    # worker.  Without a child-owned completion payload, attempted must not be
    # upgraded to executed.
    ledger.record_terminal_attempt(
        call_id=call_id, operation=operation, status="pre_spawn", exit_code=-1,
        detail=f"{type(error).__name__}:{error}",
    )
    raise
  if operation == "intended_interface_distance":
    payload = {"distance_mm": float(result["distance"])}
  else:
    payload = {
        "collision_count": len(result),
        "maximum_interference_volume_mm3": max(
            (float(row.volume) for row in result), default=0.0
        ),
    }
  ledger.record_executed_child(_occ_child_receipt_v1(
      call_id=call_id,
      operation=operation,
      status="ok",
      exit_code=0,
      result_payload=payload,
  ))
  return result


def _whole_solid_broadphase_skip_v1(
    candidate: "ExactCandidateAssemblyV2", *, epsilon: float,
) -> bool:
  """Replay the exact-worker two-body AABB gate without claiming an OCC call."""

  checker = candidate.checker
  if type(checker) is not CadQueryCollisionChecker or not checker.exact_worker_only:
    return False
  ids = (candidate.part_a, candidate.part_b)
  try:
    first = candidate.assembly.instances[ids[0]]
    second = candidate.assembly.instances[ids[1]]
    min_a, max_a = checker._world_aabb_without_native_matmul(first)
    min_b, max_b = checker._world_aabb_without_native_matmul(second)
  except (AttributeError, KeyError, ValueError, CadKernelError):
    return False
  overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
  return bool(np.any(overlap <= 0.0) or float(np.prod(overlap)) <= epsilon)


@dataclass(frozen=True, slots=True)
class BlindMatchedProposalV2:
  method_id: str
  query_id: str
  case_id: str
  row_ordinal: int
  safe_input_sha256: str
  label_commitment_sha256: str
  rank: int
  face_index_a: int
  face_index_b: int
  program_index: int
  program_id: str
  descriptor: FiniteProgramDescriptorV2
  descriptor_sha256: str
  catalog_sha256: str
  checkpoint_sha256: str
  residual_translation_pair_local: tuple[float, float, float]
  residual_rotation_vector_pair_local: tuple[float, float, float]
  pair_frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256

  def __post_init__(self) -> None:
    if any(_SHA256.fullmatch(value) is None for value in (
        self.safe_input_sha256, self.label_commitment_sha256,
        self.descriptor_sha256, self.catalog_sha256, self.checkpoint_sha256,
        self.pair_frame_policy_sha256,
    )):
      raise ValueError("blind proposal hash binding differs")
    if min(self.rank, self.row_ordinal, self.face_index_a, self.face_index_b,
           self.program_index) < 0:
      raise ValueError("blind proposal identity is negative")
    if type(self.descriptor) is not FiniteProgramDescriptorV2 or self.descriptor.sha256 != self.descriptor_sha256:
      raise ValueError("blind proposal descriptor binding differs")
    residual = self.residual_translation_pair_local + self.residual_rotation_vector_pair_local
    if len(residual) != 6 or any(not math.isfinite(float(value)) for value in residual):
      raise ValueError("blind proposal residual differs")

  def payload(self) -> dict[str, Any]:
    return {
        "method_id": self.method_id,
        "query_id": self.query_id,
        "case_id": self.case_id,
        "row_ordinal": self.row_ordinal,
        "safe_input_sha256": self.safe_input_sha256,
        "label_commitment_sha256": self.label_commitment_sha256,
        "rank": self.rank,
        "face_index_a": self.face_index_a,
        "face_index_b": self.face_index_b,
        "program_index": self.program_index,
        "program_id": self.program_id,
        "descriptor": self.descriptor.payload(),
        "descriptor_sha256": self.descriptor_sha256,
        "catalog_sha256": self.catalog_sha256,
        "checkpoint_sha256": self.checkpoint_sha256,
        "residual_translation_pair_local": list(self.residual_translation_pair_local),
        "residual_rotation_vector_pair_local": list(self.residual_rotation_vector_pair_local),
        "pair_frame_policy_sha256": self.pair_frame_policy_sha256,
    }


@dataclass(frozen=True, slots=True)
class BlindSafeViewProposalV3:
  """Deprecated v3 evidence shape retained only for artifact compatibility.

  Exact semantic-surface execution intentionally accepts V4 only.  V3 has no
  typed candidate-pool commitment and therefore cannot authorize OCC work.
  """

  method_id: str
  query_id: str
  case_id: str
  row_ordinal: int
  safe_input_sha256: str
  proposal_input_commitment_sha256: str
  rank: int
  face_index_a: int
  face_index_b: int
  program_index: int
  program_id: str
  descriptor: FiniteProgramDescriptorV2
  descriptor_sha256: str
  catalog_sha256: str
  checkpoint_sha256: str
  residual_translation_pair_local: tuple[float, float, float]
  residual_rotation_vector_pair_local: tuple[float, float, float]
  candidate_pool_commitment_sha256: str
  pair_frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256

  def __post_init__(self) -> None:
    if any(_SHA256.fullmatch(value) is None for value in (
        self.safe_input_sha256, self.proposal_input_commitment_sha256,
        self.descriptor_sha256, self.catalog_sha256, self.checkpoint_sha256,
        self.pair_frame_policy_sha256, self.candidate_pool_commitment_sha256,
    )):
      raise ValueError("blind safe-view proposal hash binding differs")
    if min(self.rank, self.row_ordinal, self.face_index_a, self.face_index_b,
           self.program_index) < 0:
      raise ValueError("blind safe-view proposal identity is negative")
    if (
        type(self.descriptor) is not FiniteProgramDescriptorV2
        or self.descriptor.sha256 != self.descriptor_sha256
    ):
      raise ValueError("blind safe-view proposal descriptor binding differs")
    residual = self.residual_translation_pair_local + self.residual_rotation_vector_pair_local
    if len(residual) != 6 or any(not math.isfinite(float(value)) for value in residual):
      raise ValueError("blind safe-view proposal residual differs")

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": "blind_safe_view_proposal.v3",
        "method_id": self.method_id, "query_id": self.query_id,
        "case_id": self.case_id, "row_ordinal": self.row_ordinal,
        "safe_input_sha256": self.safe_input_sha256,
        "proposal_input_commitment_sha256": self.proposal_input_commitment_sha256,
        "candidate_pool_commitment_sha256": self.candidate_pool_commitment_sha256,
        "rank": self.rank, "face_index_a": self.face_index_a,
        "face_index_b": self.face_index_b, "program_index": self.program_index,
        "program_id": self.program_id, "descriptor": self.descriptor.payload(),
        "descriptor_sha256": self.descriptor_sha256,
        "catalog_sha256": self.catalog_sha256,
        "checkpoint_sha256": self.checkpoint_sha256,
        "residual_translation_pair_local": list(self.residual_translation_pair_local),
        "residual_rotation_vector_pair_local": list(self.residual_rotation_vector_pair_local),
        "pair_frame_policy_sha256": self.pair_frame_policy_sha256,
    }


@dataclass(frozen=True, slots=True)
class BlindSafeViewProposalV4:
  """A blind proposal bound to the typed semantic-compatible candidate pool."""

  method_id: str
  query_id: str
  case_id: str
  row_ordinal: int
  safe_input_sha256: str
  proposal_input_commitment_sha256: str
  rank: int
  face_index_a: int
  face_index_b: int
  program_index: int
  program_id: str
  descriptor: FiniteProgramDescriptorV2
  descriptor_sha256: str
  catalog_sha256: str
  checkpoint_sha256: str
  residual_translation_pair_local: tuple[float, float, float]
  residual_rotation_vector_pair_local: tuple[float, float, float]
  candidate_pool_commitment_sha256: str
  candidate_pool_commitment_schema_version: str
  _factory_token: InitVar[object]
  pair_frame_policy_sha256: str = PAIR_FRAME_POLICY_SHA256

  def __post_init__(self, _factory_token: object) -> None:
    if _factory_token is not _BLIND_SAFE_VIEW_PROPOSAL_V4_FACTORY_TOKEN:
      raise TypeError("blind safe-view proposal v4 is decoder-factory-only")
    if any(_SHA256.fullmatch(value) is None for value in (
        self.safe_input_sha256, self.proposal_input_commitment_sha256,
        self.descriptor_sha256, self.catalog_sha256, self.checkpoint_sha256,
        self.pair_frame_policy_sha256, self.candidate_pool_commitment_sha256,
    )):
      raise ValueError("blind safe-view proposal hash binding differs")
    if (
        self.candidate_pool_commitment_schema_version
        != "semantic_surface_candidate_pool_commitment.v1"
    ):
      raise ValueError("blind safe-view proposal candidate-pool schema differs")
    if min(self.rank, self.row_ordinal, self.face_index_a, self.face_index_b,
           self.program_index) < 0:
      raise ValueError("blind safe-view proposal identity is negative")
    if (
        type(self.descriptor) is not FiniteProgramDescriptorV2
        or self.descriptor.sha256 != self.descriptor_sha256
    ):
      raise ValueError("blind safe-view proposal descriptor binding differs")
    residual = self.residual_translation_pair_local + self.residual_rotation_vector_pair_local
    if len(residual) != 6 or any(not math.isfinite(float(value)) for value in residual):
      raise ValueError("blind safe-view proposal residual differs")

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": "blind_safe_view_proposal.v4",
        "method_id": self.method_id, "query_id": self.query_id,
        "case_id": self.case_id, "row_ordinal": self.row_ordinal,
        "safe_input_sha256": self.safe_input_sha256,
        "proposal_input_commitment_sha256": self.proposal_input_commitment_sha256,
        "candidate_pool_commitment_sha256": self.candidate_pool_commitment_sha256,
        "candidate_pool_commitment_schema_version": (
            self.candidate_pool_commitment_schema_version
        ),
        "rank": self.rank, "face_index_a": self.face_index_a,
        "face_index_b": self.face_index_b, "program_index": self.program_index,
        "program_id": self.program_id, "descriptor": self.descriptor.payload(),
        "descriptor_sha256": self.descriptor_sha256,
        "catalog_sha256": self.catalog_sha256,
        "checkpoint_sha256": self.checkpoint_sha256,
        "residual_translation_pair_local": list(self.residual_translation_pair_local),
        "residual_rotation_vector_pair_local": list(self.residual_rotation_vector_pair_local),
        "pair_frame_policy_sha256": self.pair_frame_policy_sha256,
    }


def materialize_blind_topk_v2(
    matched_row: Any,
    *,
    catalog: Any,
    checkpoint_sha256: str,
    budget: ExactExecutionBudgetV2,
) -> tuple[BlindMatchedProposalV2, ...]:
  """Copy the executable prediction allowlist without touching gold fields."""

  if matched_row.status != "ok":
    return ()
  if _SHA256.fullmatch(checkpoint_sha256) is None:
    raise ValueError("blind proposal checkpoint pin differs")
  joint = tuple(matched_row.ranked_joint)
  residuals = tuple(matched_row.ranked_joint_residuals)
  if len(joint) != len(residuals) or len(joint) != budget.top_k:
    raise ValueError("matched executable Top-K payload differs from exact budget")
  result = []
  for rank, (identity, residual) in enumerate(zip(joint, residuals, strict=True)):
    if len(identity) != 3 or len(residual) != 6:
      raise ValueError("matched executable hypothesis schema differs")
    face_a, face_b, program_index = (int(value) for value in identity)
    entry = catalog.entry(program_index)
    result.append(
        BlindMatchedProposalV2(
            method_id=str(matched_row.method_id),
            query_id=str(matched_row.query_id),
            case_id=str(matched_row.case_id),
            row_ordinal=int(matched_row.row_ordinal),
            safe_input_sha256=str(matched_row.safe_input_sha256),
            label_commitment_sha256=str(matched_row.label_commitment_sha256),
            rank=rank,
            face_index_a=face_a,
            face_index_b=face_b,
            program_index=program_index,
            program_id=str(entry.program_id),
            descriptor=entry.descriptor,
            descriptor_sha256=str(entry.descriptor_sha256),
            catalog_sha256=str(catalog.catalog_sha256),
            checkpoint_sha256=checkpoint_sha256,
            residual_translation_pair_local=tuple(float(value) for value in residual[:3]),
            residual_rotation_vector_pair_local=tuple(float(value) for value in residual[3:]),
        )
    )
  return tuple(result)


@dataclass(frozen=True, slots=True)
class BlindSafeViewTopKMaterializationV4:
  proposals: tuple[BlindSafeViewProposalV4, ...]
  requested_top_k: int
  shortfall: int
  compatible_candidates: int
  candidates_evaluated: int
  masked_reason_counts: tuple[tuple[str, int], ...]
  candidate_pool_commitment_sha256: str
  proposal_input_commitment_sha256: str

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": "blind_safe_view_topk_materialization.v4",
        "requested_top_k": self.requested_top_k,
        "proposal_count": len(self.proposals), "shortfall": self.shortfall,
        "compatible_candidates": self.compatible_candidates,
        "candidates_evaluated": self.candidates_evaluated,
        "masked_reason_counts": dict(self.masked_reason_counts),
        "candidate_pool_commitment_schema_version": (
            "semantic_surface_candidate_pool_commitment.v1"
        ),
        "candidate_pool_commitment_sha256": self.candidate_pool_commitment_sha256,
        "proposal_input_commitment_sha256": self.proposal_input_commitment_sha256,
    }


def materialize_seed11_blind_topk_with_receipt_v4(
    model: Any, safe_view: Any, *, catalog: Any, checkpoint_sha256: str,
    budget: ExactExecutionBudgetV2, query_id: str, case_id: str,
    row_ordinal: int,
) -> BlindSafeViewTopKMaterializationV4:
  """Run the shared decoder and retain an explicit shortfall/mask receipt."""

  import torch

  from .full_graph_face_frame_authority_v1 import (
      graph_program_inputs_from_safe_view_v1,
  )
  from .semantic_surface_compatible_topk_decoder_v1 import (
      decode_joint_output_semantic_surface_topk_v1,
  )

  if _SHA256.fullmatch(checkpoint_sha256) is None:
    raise ValueError("blind inference checkpoint pin differs")
  inputs = graph_program_inputs_from_safe_view_v1(safe_view, catalog=catalog)
  model.eval()
  with torch.inference_mode():
    output = model(inputs).core
  decoded = decode_joint_output_semantic_surface_topk_v1(
      output, safe_view=safe_view, catalog=catalog,
      top_l=model.config.top_l, top_k=budget.top_k,
  )
  safe_input_sha256 = str(safe_view.input_sha256)
  nonlabel_commitment = _canonical_sha256({
      "schema_version": "blind_safe_view_proposal_input.v1",
      "row_ordinal": int(row_ordinal), "safe_input_sha256": safe_input_sha256,
      "checkpoint_sha256": checkpoint_sha256,
      "semantic_decoder_schema_version": decoded.schema_version,
      "candidate_pool_commitment_sha256": decoded.candidate_pool_commitment_sha256,
  })
  result = []
  for rank, candidate in enumerate(decoded.selected):
    face_a, face_b = candidate.face_index_a, candidate.face_index_b
    program_index, residual = candidate.program_index, candidate.residual
    entry = catalog.entry(program_index)
    result.append(BlindSafeViewProposalV4(
        method_id="joint_v3_seed11", query_id=query_id, case_id=case_id,
        row_ordinal=int(row_ordinal), safe_input_sha256=safe_input_sha256,
        proposal_input_commitment_sha256=nonlabel_commitment, rank=rank,
        face_index_a=face_a, face_index_b=face_b,
        program_index=program_index, program_id=str(entry.program_id),
        descriptor=entry.descriptor,
        descriptor_sha256=str(entry.descriptor_sha256),
        catalog_sha256=str(catalog.catalog_sha256),
        checkpoint_sha256=checkpoint_sha256,
        residual_translation_pair_local=residual[:3],
        residual_rotation_vector_pair_local=residual[3:],
        candidate_pool_commitment_sha256=decoded.candidate_pool_commitment_sha256,
        candidate_pool_commitment_schema_version=(
            "semantic_surface_candidate_pool_commitment.v1"
        ),
        _factory_token=_BLIND_SAFE_VIEW_PROPOSAL_V4_FACTORY_TOKEN,
    ))
  return BlindSafeViewTopKMaterializationV4(
      proposals=tuple(result), requested_top_k=budget.top_k,
      shortfall=decoded.shortfall,
      compatible_candidates=decoded.compatible_candidates,
      candidates_evaluated=decoded.candidates_evaluated,
      masked_reason_counts=tuple(sorted(decoded.masked_reason_counts.items())),
      candidate_pool_commitment_sha256=decoded.candidate_pool_commitment_sha256,
      proposal_input_commitment_sha256=nonlabel_commitment,
  )


def materialize_seed11_blind_topk_from_safe_view_v3(
    model: Any, safe_view: Any, *, catalog: Any, checkpoint_sha256: str,
    budget: ExactExecutionBudgetV2, query_id: str, case_id: str,
    row_ordinal: int,
) -> tuple[BlindSafeViewProposalV4, ...]:
  """Compatibility wrapper; exact runners should use the V4 receipt seam."""

  materialized = materialize_seed11_blind_topk_with_receipt_v4(
      model, safe_view, catalog=catalog, checkpoint_sha256=checkpoint_sha256,
      budget=budget, query_id=query_id, case_id=case_id,
      row_ordinal=row_ordinal,
  )
  if materialized.shortfall:
    raise ValueError(
        "semantic_surface_compatible_topk_shortfall:"
        f"requested={budget.top_k}:selected={len(materialized.proposals)}:"
        f"pool={materialized.candidate_pool_commitment_sha256}"
    )
  return materialized.proposals


@dataclass(frozen=True, slots=True)
class PredictedWorldPlacementV2:
  world_delta: Transform
  child_world_transform: Transform
  axis_alignment_degrees: float
  absolute_normal_gap_mm: float
  constraint_template_id: str
  semantic_catalog_sha256: str
  applied_refinement: AppliedConstraintRefinementV1
  program_constraint_certificate: ProgramConstraintCertificateV1


@dataclass(frozen=True, slots=True)
class ExactCandidateAssemblyV2:
  assembly: AssemblyState
  checker: CadQueryCollisionChecker
  part_a: str
  part_b: str
  socket_a: str
  socket_b: str


def _validated_proposal_surface_frames_v1(
    proposal: BlindMatchedProposalV2 | BlindSafeViewProposalV4,
    frame_authority: Any,
    *,
    expected_surface_type_a: str,
    expected_surface_type_b: str,
) -> tuple[Any, Any]:
  """Authenticate selected face classes before base delta, STEP, or OCC."""

  frame_a = frame_authority.frame_for(
      part_slot="a", graph_local_face_index=proposal.face_index_a
  )
  frame_b = frame_authority.frame_for(
      part_slot="b", graph_local_face_index=proposal.face_index_b
  )
  observed = (getattr(frame_a, "surface_type", None), getattr(frame_b, "surface_type", None))
  expected = (expected_surface_type_a, expected_surface_type_b)
  if observed != expected:
    raise SurfaceTypeMismatch(
        "surface_type_mismatch:"
        f"expected={expected[0]}/{expected[1]}:observed={observed[0]}/{observed[1]}"
    )
  return frame_a, frame_b


def world_placement_for_blind_proposal_v2(
    proposal: BlindMatchedProposalV2 | BlindSafeViewProposalV4,
    frame_authority: Any,
    *,
    constraint_catalog: AuthenticatedConstraintExecutableProgramCatalogV1,
) -> PredictedWorldPlacementV2:
  """Execute a finite constraint and its bounded symmetry-quotient refinement."""

  if type(constraint_catalog) is not AuthenticatedConstraintExecutableProgramCatalogV1:
    raise TypeError("exact placement requires a loader-authenticated semantic catalog")
  entry = constraint_catalog.require_supported(proposal)
  assert entry.template is not None
  frame_a, frame_b = _validated_proposal_surface_frames_v1(
      proposal,
      frame_authority,
      expected_surface_type_a=entry.descriptor.surface_type_a,
      expected_surface_type_b=entry.descriptor.surface_type_b,
  )
  current_world = frame_authority.part_world_transforms
  if not isinstance(current_world, tuple) or len(current_world) != 2:
    raise ValueError("exact placement part-world transform domain differs")

  def surface_frame(value: Any, *, slot: str) -> SurfaceConstraintFrameV1:
    center = getattr(value, "analytic_center_world_mm", None)
    return SurfaceConstraintFrameV1(
        part_slot=slot,
        origin_world_mm=tuple(float(item) for item in value.origin_world_mm),
        rotation_local_to_world=tuple(
            tuple(float(item) for item in row)
            for row in value.rotation_local_to_world
        ),
        analytic_center_world_mm=(
            None if center is None else tuple(float(item) for item in center)
        ),
    )

  execution = execute_surface_constraint_v1(
      template=entry.template,
      frame_a=surface_frame(frame_a, slot="a"),
      frame_b=surface_frame(frame_b, slot="b"),
      current_child_world=current_world[1],
      residual_translation_pair_local=proposal.residual_translation_pair_local,
      residual_rotation_vector_pair_local=proposal.residual_rotation_vector_pair_local,
  )
  delta = execution.world_delta_matrix
  child = execution.child_world_matrix
  metrics = execution.certificate.metrics
  return PredictedWorldPlacementV2(
      world_delta=Transform(rotation=delta[:3, :3], translation=delta[:3, 3]),
      child_world_transform=Transform(
          rotation=child[:3, :3], translation=child[:3, 3]
      ),
      axis_alignment_degrees=float(
          metrics.get("normal_error_degrees", metrics.get("axis_error_degrees", 0.0))
      ),
      absolute_normal_gap_mm=float(metrics.get("normal_gap_mm", 0.0)),
      constraint_template_id=execution.template_id,
      semantic_catalog_sha256=constraint_catalog.semantic_catalog_sha256,
      applied_refinement=execution.refinement,
      program_constraint_certificate=execution.certificate,
  )


def _local_socket_from_world_frame(frame: Any, world: Transform, *, name: str, metadata: dict[str, Any]) -> Socket:
  rotation_world = np.asarray(world.rotation, dtype=float)
  inverse_rotation = rotation_world.T
  local_origin = _matvec(
      inverse_rotation,
      np.asarray(frame.origin_world_mm, dtype=float) - np.asarray(world.translation, dtype=float),
  )
  local_rotation = _matmul(
      inverse_rotation, np.asarray(frame.rotation_local_to_world, dtype=float)
  )
  return Socket(
      name=name,
      kind="predicted_interface",
      origin=local_origin,
      axis=local_rotation[:, 2],
      normal=local_rotation[:, 2],
      x_axis=local_rotation[:, 0],
      y_axis=local_rotation[:, 1],
      z_axis=local_rotation[:, 2],
      metadata=metadata,
  )


def build_exact_candidate_assembly_v2(
    proposal: BlindMatchedProposalV2 | BlindSafeViewProposalV4,
    *,
    frame_authority: Any,
    source_context: Any,
    constraint_catalog: AuthenticatedConstraintExecutableProgramCatalogV1,
    budget: ExactExecutionBudgetV2,
) -> ExactCandidateAssemblyV2:
  """Construct the receipt-bound two-solid OCC input for one hypothesis."""

  if len(source_context.parts) != 2:
    raise ValueError("exact execution source context must contain two parts")
  frame_authority.revalidate()
  source_bindings = source_context.source_bindings
  if (
      not source_bindings.get("view_execution_pose_row_receipt_sha256")
      or source_bindings.get("source_pose_bound") is not False
      or source_bindings.get("execution_input_row_receipt_sha256")
  ):
    raise ValueError(
        "exact execution requires a V5-bound independently scrambled current pose"
    )
  if (
      proposal.query_id != frame_authority.row_id
      or proposal.query_id != source_context.row_id
      or proposal.case_id != source_context.case_id
      or proposal.safe_input_sha256 != frame_authority.safe_input_sha256
      or source_context.binding_sha256 != frame_authority.source_context_binding_sha256
      or source_bindings.get("safe_input_sha256") != proposal.safe_input_sha256
  ):
    raise ValueError("proposal/frame/source authenticated query binding differs")
  frame_a = frame_authority.frame_for(
      part_slot="a", graph_local_face_index=proposal.face_index_a
  )
  frame_b = frame_authority.frame_for(
      part_slot="b", graph_local_face_index=proposal.face_index_b
  )
  placement = world_placement_for_blind_proposal_v2(
      proposal, frame_authority, constraint_catalog=constraint_catalog
  )
  current_world = frame_authority.part_world_transforms
  instances: dict[str, Any] = {}
  shapes: dict[str, Any] = {}
  socket_names = ("predicted_a", "predicted_b")
  part_names = ("part_a", "part_b")
  selected_frames = (frame_a, frame_b)
  selected_world = (current_world[0], placement.child_world_transform)
  identity_gauge = Transform.identity().to_dict()
  namespace = _canonical_sha256(
      {
          "schema": "authenticated_exact_identity_namespace.v2",
          "safe_input_sha256": proposal.safe_input_sha256,
          "source_context_binding_sha256": str(source_context.binding_sha256),
      }
  )
  for slot, (part_name, socket_name, query_part, frame, world) in enumerate(
      zip(
          part_names, socket_names, source_context.parts, selected_frames,
          selected_world, strict=True,
      )
  ):
    interface_id = _canonical_sha256(
        {
            "schema": "authenticated_predicted_interface_id.v2",
            "query_id": proposal.query_id,
            "rank": proposal.rank,
            "slot": slot,
            "raw_occ_face_index": frame.raw_occ_face_index,
            "occ_face_signature_sha256": frame.occ_face_signature_sha256,
        }
    )
    raw_indices = [int(frame.raw_occ_face_index)]
    signatures = [str(frame.occ_face_signature_sha256)]
    socket_metadata = {
        "private_exact_interface_id": interface_id,
        "private_exact_raw_face_indices": raw_indices,
        "private_exact_raw_face_signature_sha256s": signatures,
        "private_exact_face_identity_proof": "occ_tshape_partner_bijection",
    }
    socket_metadata["private_exact_binding_sha256"] = exact_face_binding_sha256(
        namespace=namespace,
        raw_to_randomized_transform=identity_gauge,
        interface_id=interface_id,
        raw_face_indices=raw_indices,
        raw_face_signature_sha256s=signatures,
        face_identity_proof="occ_tshape_partner_bijection",
    )
    shape = _load_captured_step_shape(query_part.step)
    bbox_min, bbox_max = shape_bbox(shape)
    template = PartTemplate(
        name=part_name,
        sockets={
            socket_name: _local_socket_from_world_frame(
                frame, current_world[slot], name=socket_name, metadata=socket_metadata
            )
        },
        local_bbox_min=bbox_min,
        local_bbox_max=bbox_max,
        metadata={
            "step_path": str(query_part.step.resolved_path),
            "frame_protocol": "benchmark_v2_predicted_full_graph_frame.v2",
            "benchmark_v2_raw_to_randomized_transform": identity_gauge,
            "benchmark_v2_exact_identity_namespace": namespace,
        },
    )
    instance = template.instantiate(part_name)
    instance.transform = world
    instances[part_name] = instance
    shapes[part_name] = shape
  graph = ConstraintGraph(
      instances=instances, constraints=[], anchor="part_a",
      metadata={"schema_version": "authenticated_exact_candidate_graph.v2"},
  )
  assembly = AssemblyState(instances=instances, constraint_graph=graph)
  return ExactCandidateAssemblyV2(
      assembly=assembly,
      checker=CadQueryCollisionChecker(
          shape_library=shapes,
          allow_aabb_fallback=False,
          exact_timeout_seconds=budget.per_occ_timeout_seconds,
          skip_exact_for_complex_pairs=False,
          exact_worker_only=True,
      ),
      part_a="part_a", part_b="part_b", socket_a="predicted_a", socket_b="predicted_b",
  )


def execute_blind_topk_v2(
    proposals: Sequence[BlindMatchedProposalV2 | BlindSafeViewProposalV4],
    *,
    frame_authority: Any,
    source_context: Any,
    constraint_catalog: AuthenticatedConstraintExecutableProgramCatalogV1,
    budget: ExactExecutionBudgetV2,
    telemetry: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
  """Execute supported finite programs; unsupported rows consume zero OCC calls."""

  if any(type(proposal) is BlindSafeViewProposalV3 for proposal in proposals):
    raise ValueError(
        "legacy blind_safe_view_proposal.v3 cannot authorize exact execution"
    )
  if type(constraint_catalog) is not AuthenticatedConstraintExecutableProgramCatalogV1:
    raise TypeError("exact executor requires a loader-authenticated semantic catalog")
  constraint_catalog.revalidate()
  start = time.monotonic()
  if len(proposals) != budget.top_k:
    raise ValueError("exact executor proposal count differs from frozen Top-K")
  if len({proposal.method_id for proposal in proposals}) != 1 or len(
      {proposal.query_id for proposal in proposals}
  ) != 1:
    raise ValueError("exact executor received a mixed query or method domain")
  results: list[dict[str, Any]] = []
  reserved_calls = 0
  for proposal in proposals:
    ledger = ExactOccCallLedgerV3()
    if telemetry is not None:
      telemetry({
          "event": "candidate_start", "query_id": proposal.query_id,
          "method_id": proposal.method_id, "rank": proposal.rank,
      })
    base = {
        "method_id": proposal.method_id,
        "query_id": proposal.query_id,
        "case_id": proposal.case_id,
        "row_ordinal": proposal.row_ordinal,
        "rank": proposal.rank,
        "face_index_a": proposal.face_index_a,
        "face_index_b": proposal.face_index_b,
        "program_index": proposal.program_index,
        "program_id": proposal.program_id,
        "semantic_catalog_sha256": constraint_catalog.semantic_catalog_sha256,
        "program_supported": False,
        "surface_type_match": False,
        "constraint_template_id": None,
        "program_constraint_certificate": None,
        "intended_interface_distance_mm": None,
        "whole_body_interference_volume_mm3": None,
        "contact_feasibility_accepted": False,
    }
    try:
      semantic_entry = constraint_catalog.require_supported(proposal)
    except UnsupportedProgramSemantics as error:
      results.append({
          **base,
          "status": "unsupported_program_semantics",
          "failure_reason": str(error),
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - start,
      })
      if telemetry is not None:
        telemetry({
            "event": "candidate_complete", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
            "status": "unsupported_program_semantics",
        })
      continue
    assert semantic_entry.template is not None
    base["program_supported"] = True
    try:
      _validated_proposal_surface_frames_v1(
          proposal,
          frame_authority,
          expected_surface_type_a=semantic_entry.descriptor.surface_type_a,
          expected_surface_type_b=semantic_entry.descriptor.surface_type_b,
      )
    except SurfaceTypeMismatch as error:
      results.append({
          **base,
          "status": "surface_type_mismatch",
          "failure_reason": str(error),
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - start,
      })
      if telemetry is not None:
        telemetry({
            "event": "candidate_complete", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
            "status": "surface_type_mismatch", "failure_reason": str(error),
        })
      continue
    base["surface_type_match"] = True
    elapsed = time.monotonic() - start
    if elapsed >= budget.query_wall_time_seconds or reserved_calls + 2 > budget.max_occ_calls:
      results.append({
          **base,
          "status": "budget_exhausted",
          "failure_reason": "query_budget_exhausted",
          "occ_call_ledger": ledger.payload(),
      })
      if telemetry is not None:
        telemetry({"event": "candidate_complete", "query_id": proposal.query_id,
                   "method_id": proposal.method_id, "rank": proposal.rank,
                   "status": "budget_exhausted"})
      continue
    reserved_calls += 2
    ledger.reserve(2)
    pregeometry: dict[str, Any] = {}
    try:
      if telemetry is not None:
        telemetry({
            "event": "candidate_build_start", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
        })
      candidate = build_exact_candidate_assembly_v2(
          proposal,
          frame_authority=frame_authority,
          source_context=source_context,
          constraint_catalog=constraint_catalog,
          budget=budget,
      )
      placement = world_placement_for_blind_proposal_v2(
          proposal, frame_authority, constraint_catalog=constraint_catalog
      )
      pregeometry = {
          "constraint_template_id": placement.constraint_template_id,
          "program_constraint_certificate": (
              placement.program_constraint_certificate.payload()
          ),
          "applied_bounded_refinement": placement.applied_refinement.payload(),
          "axis_alignment_degrees": placement.axis_alignment_degrees,
          "absolute_normal_gap_mm": placement.absolute_normal_gap_mm,
      }
      if telemetry is not None:
        telemetry({
            "event": "candidate_build_complete", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
            "transformed_shape_count": 2,
        })
      remaining = budget.query_wall_time_seconds - (time.monotonic() - start)
      if remaining < 0.5:
        raise TimeoutError("query_wall_time_exhausted_before_interface_distance")
      candidate.checker.exact_timeout_seconds = min(
          budget.per_occ_timeout_seconds, max(0.5, remaining)
      )
      if telemetry is not None:
        telemetry({
            "event": "occ_interface_distance_start", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
            "timeout_seconds": candidate.checker.exact_timeout_seconds,
        })
      distance = _observed_occ_call_v1(
          ledger,
          call_id=f"{proposal.query_id}:{proposal.method_id}:{proposal.rank}:distance",
          operation="intended_interface_distance",
          invoke=lambda: candidate.checker.interface_distance_details_between(
              candidate.assembly, candidate.part_a, candidate.socket_a,
              candidate.part_b, candidate.socket_b,
          ),
      )
      if telemetry is not None:
        telemetry({
            "event": "occ_interface_distance_complete", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
            "distance_mm": float(distance["distance"]),
        })
      remaining = budget.query_wall_time_seconds - (time.monotonic() - start)
      if remaining < 0.5:
        raise TimeoutError("query_wall_time_exhausted_before_interference")
      candidate.checker.exact_timeout_seconds = min(
          budget.per_occ_timeout_seconds, max(0.5, remaining)
      )
      if telemetry is not None:
        telemetry({
            "event": "occ_whole_solid_interference_start", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
            "timeout_seconds": candidate.checker.exact_timeout_seconds,
        })
      if _whole_solid_broadphase_skip_v1(
          candidate, epsilon=budget.interference_epsilon_mm3
      ):
        ledger.record_broadphase_skip()
        collisions = []
      else:
        collisions = _observed_occ_call_v1(
            ledger,
            call_id=f"{proposal.query_id}:{proposal.method_id}:{proposal.rank}:interference",
            operation="whole_solid_interference",
            invoke=lambda: candidate.checker.check(
                candidate.assembly, epsilon=budget.interference_epsilon_mm3
            ),
        )
      if telemetry is not None:
        telemetry({
            "event": "occ_whole_solid_interference_complete", "query_id": proposal.query_id,
            "method_id": proposal.method_id, "rank": proposal.rank,
            "collision_count": len(collisions),
        })
      exact_distance = float(distance["distance"])
      maximum_interference = max((float(row.volume) for row in collisions), default=0.0)
      accepted = bool(
          exact_distance <= budget.contact_tolerance_mm
          and maximum_interference <= budget.interference_epsilon_mm3
      )
      results.append(
          {
              **base,
              **pregeometry,
              "status": "accepted" if accepted else "rejected",
              "failure_reason": None if accepted else "contact_feasibility_rejected",
              "intended_interface_distance_mm": exact_distance,
              "whole_body_interference_volume_mm3": maximum_interference,
              "axis_and_gap_role": "program_specific_pregeometry_certificate",
              "contact_feasibility_accepted": accepted,
              "occ_call_ledger": ledger.payload(),
              "elapsed_seconds": time.monotonic() - start,
          }
      )
      if telemetry is not None:
        telemetry({"event": "candidate_complete", "query_id": proposal.query_id,
                   "method_id": proposal.method_id, "rank": proposal.rank,
                   "status": "accepted" if accepted else "rejected"})
    except (CadKernelError, TimeoutError, ValueError, OSError, RuntimeError) as error:
      message = f"{type(error).__name__}:{error}"
      ledger_payload = ledger.payload()
      terminal_statuses = tuple(
          str(row["status"])
          for row in ledger_payload["terminal_attempt_receipts"]
      )
      if isinstance(error, CadKernelChildProcessError):
        failure = f"occ_{error.status}"
      elif ledger.attempted == 0:
        failure = "pre_spawn_error"
      elif "pre_spawn" in terminal_statuses:
        failure = "occ_pre_spawn_error"
      elif "timeout" in message.lower():
        failure = "occ_timeout"
      else:
        failure = "kernel_error"
      results.append(
          {
              **base,
              **pregeometry,
              "status": failure,
              "failure_reason": message,
              "occ_call_ledger": ledger_payload,
              "elapsed_seconds": time.monotonic() - start,
          }
      )
      if telemetry is not None:
        telemetry({"event": "candidate_complete", "query_id": proposal.query_id,
                   "method_id": proposal.method_id, "rank": proposal.rank,
                   "status": failure, "failure_reason": message})
  return tuple(results)


def score_executed_topk_v2(
    executed_rows: Sequence[Mapping[str, Any]],
    *,
    target_face_indices_a: Sequence[int],
    target_face_indices_b: Sequence[int],
    target_program_index: int,
) -> tuple[dict[str, Any], ...]:
  """Join dev gold without upgrading contact feasibility to mate correctness."""

  targets_a = frozenset(int(value) for value in target_face_indices_a)
  targets_b = frozenset(int(value) for value in target_face_indices_b)
  if not targets_a or not targets_b or not 0 <= int(target_program_index) < 19:
    raise ValueError("post-execution target identity differs")
  result = []
  for row in executed_rows:
    item = dict(row)
    gold_joint = bool(
        int(item["face_index_a"]) in targets_a
        and int(item["face_index_b"]) in targets_b
        and int(item["program_index"]) == int(target_program_index)
    )
    item["gold_joint_match"] = gold_joint
    item["gold_joint_contact_feasibility"] = bool(
        gold_joint and item.get("contact_feasibility_accepted") is True
    )
    result.append(item)
  return tuple(result)


def aggregate_query_method_topk_v3(
    scored_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
  """Aggregate candidate rows with the preregistered Top-K-any definition."""

  grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
  for row in scored_rows:
    query_id = str(row.get("query_id") or "")
    method_id = str(row.get("method_id") or "")
    rank = row.get("rank")
    if not query_id or not method_id or type(rank) is not int or rank < 0:
      raise ValueError("Top-K aggregate row identity differs")
    grouped.setdefault((query_id, method_id), []).append(row)
  result = []
  for (query_id, method_id), rows in sorted(grouped.items()):
    ranks = [int(row["rank"]) for row in rows]
    if len(set(ranks)) != len(ranks):
      raise ValueError("Top-K aggregate repeats a candidate rank")
    ordered = sorted(rows, key=lambda row: int(row["rank"]))
    result.append({
        "schema_version": "query_method_topk_contact_feasibility.v1",
        "query_id": query_id,
        "method_id": method_id,
        "top_k": len(ordered),
        "any_contact_feasibility": any(
            row.get("contact_feasibility_accepted") is True for row in ordered
        ),
        "any_gold_joint_match": any(
            row.get("gold_joint_match") is True for row in ordered
        ),
        "any_gold_joint_contact_feasibility": any(
            row.get("gold_joint_contact_feasibility") is True for row in ordered
        ),
        "candidate_rank_domain_sha256": _canonical_sha256(ranks),
    })
  return tuple(result)
