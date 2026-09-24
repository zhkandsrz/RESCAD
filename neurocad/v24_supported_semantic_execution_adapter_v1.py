"""Executable finite-program placement for the dominant V19 semantics.

This development adapter is intentionally narrower than the 19-way
classification roster.  It executes only the two source-backed finite
programs:

* program 9: opposed coincident planar seats;
* program 11: an unoriented shaft/bore common axis.

All other categorical programs stop before STEP loading and consume no OCC
calls.  The predicted six-vector is projected onto the selected program's
free degrees of freedom and bounded by the authenticated semantic catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .authenticated_exact_execution_v2 import (
    ExactCandidateAssemblyV2,
    ExactExecutionBudgetV2,
    ExactOccCallLedgerV3,
    _observed_occ_call_v1,
    _whole_solid_broadphase_skip_v1,
)
from .benchmark_v2_model_view_v2 import _load_captured_step_shape
from .cadquery_backend import (
    CadKernelChildProcessError,
    CadKernelError,
    CadQueryCollisionChecker,
    shape_bbox,
)
from .constraint_catalog_trust_root_v1 import (
    OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1,
)
from .constraint_executable_program_catalog_v1 import (
    AuthenticatedConstraintExecutableProgramCatalogV1,
    SurfaceConstraintFrameV1,
    UnsupportedProgramSemantics,
    execute_surface_constraint_v1,
    load_authenticated_constraint_executable_catalog_v1,
)
from .domain_types import (
    AssemblyState,
    ConstraintGraph,
    PartTemplate,
    Transform,
)
from .full_graph_face_frame_authority_v1 import _constraint_surface_type
from .v19_exact_execution_adapter_v1 import (
    FIXED_EXACT_BUDGET,
    ResolvedExactGeometryV1,
    _face_frame,
    _socket,
)
from .v19_exact_program_proposal_v1 import (
    ExactProgramProposalV1,
    NoCandidateProposalV1,
    canonical_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_SPEC_PATH = PROJECT_ROOT / (
    "artifacts/formal/"
    "externally_frozen_program_catalog_spec_v3_p0_20260719.json"
)
TASK_SCOPE = "supported_finite_program_plus_bounded_residual_diagnostic"
SUPPORTED_PROGRAM_INDICES = (9, 11)


class SemanticSurfaceTypeMismatchV1(ValueError):
  """Selected OCC faces differ from the program's analytic surface types."""


def load_supported_semantic_catalog_v1(
) -> AuthenticatedConstraintExecutableProgramCatalogV1:
  trust = OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1
  catalog = load_authenticated_constraint_executable_catalog_v1(
      CATALOG_SPEC_PATH,
      expected_catalog_roster_sha256=trust["catalog_roster_sha256"],
      expected_spec_file_sha256=trust["spec_file_sha256"],
      expected_source_artifact_sha256=trust["source_artifact_sha256"],
      expected_producer_code_sha256=trust["producer_code_sha256"],
  )
  if catalog.supported_program_indices != SUPPORTED_PROGRAM_INDICES:
    raise ValueError("supported finite-program domain differs")
  return catalog


def _surface_frame(
    frame: np.ndarray,
    *,
    slot: str,
) -> SurfaceConstraintFrameV1:
  return SurfaceConstraintFrameV1(
      part_slot=slot,
      origin_world_mm=tuple(float(value) for value in frame[:3, 3]),
      rotation_local_to_world=tuple(
          tuple(float(value) for value in row) for row in frame[:3, :3]
      ),
  )


@dataclass(frozen=True, slots=True)
class SupportedSemanticCandidateV1:
  candidate: ExactCandidateAssemblyV2
  semantic_catalog_sha256: str
  constraint_template_id: str
  constraint_certificate: Mapping[str, Any]
  refinement: Mapping[str, Any]


def build_supported_semantic_candidate_v1(
    proposal: ExactProgramProposalV1,
    geometry: ResolvedExactGeometryV1,
    *,
    semantic_catalog: AuthenticatedConstraintExecutableProgramCatalogV1,
    budget: ExactExecutionBudgetV2 = FIXED_EXACT_BUDGET,
) -> SupportedSemanticCandidateV1:
  """Build one two-solid candidate from authenticated program semantics."""

  if proposal.sample.graph_work_id != geometry.graph_work_id:
    raise ValueError("proposal/geometry graph-work binding differs")
  if type(semantic_catalog) is not (
      AuthenticatedConstraintExecutableProgramCatalogV1
  ):
    raise TypeError("semantic execution requires authenticated catalog")
  entry = semantic_catalog.require_supported(proposal.program_index)
  if proposal.program_id != entry.descriptor.program_id:
    raise ValueError("proposal program ID differs from semantic catalog")
  assert entry.template is not None
  shape_a = _load_captured_step_shape(geometry.step_a)
  shape_b = _load_captured_step_shape(geometry.step_b)
  frame_a, face_a = _face_frame(
      shape_a,
      raw_index=geometry.raw_face_index_a,
      expected_signature=geometry.face_signature_a,
  )
  frame_b, face_b = _face_frame(
      shape_b,
      raw_index=geometry.raw_face_index_b,
      expected_signature=geometry.face_signature_b,
  )
  observed_types = (
      _constraint_surface_type(face_a),
      _constraint_surface_type(face_b),
  )
  expected_types = (
      entry.descriptor.surface_type_a,
      entry.descriptor.surface_type_b,
  )
  if observed_types != expected_types:
    raise SemanticSurfaceTypeMismatchV1(
        "surface_type_mismatch:"
        f"expected={expected_types[0]}/{expected_types[1]}:"
        f"observed={observed_types[0]}/{observed_types[1]}"
    )
  execution = execute_surface_constraint_v1(
      template=entry.template,
      frame_a=_surface_frame(frame_a, slot="a"),
      frame_b=_surface_frame(frame_b, slot="b"),
      current_child_world=Transform.identity(),
      residual_translation_pair_local=(
          proposal.residual_twist_pair_local[:3]
      ),
      residual_rotation_vector_pair_local=(
          proposal.residual_twist_pair_local[3:]
      ),
  )
  child_matrix = execution.child_world_matrix
  child_world = Transform(
      rotation=np.asarray(child_matrix[:3, :3], dtype=float),
      translation=np.asarray(child_matrix[:3, 3], dtype=float),
  )
  namespace = canonical_sha256({
      "schema_version": "v24_supported_semantic_identity_namespace.v1",
      "geometry_binding_sha256": geometry.geometry_binding_sha256,
      "semantic_catalog_sha256": semantic_catalog.semantic_catalog_sha256,
      "proposal_input_commitment_sha256": (
          proposal.proposal_input_commitment_sha256
      ),
  })
  interfaces = (
      canonical_sha256({
          "namespace": namespace,
          "slot": "a",
          "raw_occ_face_index": geometry.raw_face_index_a,
          "face_signature_sha256": geometry.face_signature_a,
      }),
      canonical_sha256({
          "namespace": namespace,
          "slot": "b",
          "raw_occ_face_index": geometry.raw_face_index_b,
          "face_signature_sha256": geometry.face_signature_b,
      }),
  )
  shapes = {"part_a": shape_a, "part_b": shape_b}
  instances = {}
  for (
      slot,
      name,
      capture,
      shape,
      frame,
      raw_index,
      signature,
      world,
  ) in (
      (
          "a",
          "part_a",
          geometry.step_a,
          shape_a,
          frame_a,
          geometry.raw_face_index_a,
          geometry.face_signature_a,
          Transform.identity(),
      ),
      (
          "b",
          "part_b",
          geometry.step_b,
          shape_b,
          frame_b,
          geometry.raw_face_index_b,
          geometry.face_signature_b,
          child_world,
      ),
  ):
    socket_name = f"predicted_{slot}"
    bbox_min, bbox_max = shape_bbox(shape)
    template = PartTemplate(
        name=name,
        sockets={
            socket_name: _socket(
                name=socket_name,
                frame=frame,
                raw_index=raw_index,
                signature=signature,
                namespace=namespace,
                interface_id=interfaces[0 if slot == "a" else 1],
            )
        },
        local_bbox_min=bbox_min,
        local_bbox_max=bbox_max,
        metadata={
            "step_path": str(capture.resolved_path),
            "benchmark_v2_raw_to_randomized_transform": (
                Transform.identity().to_dict()
            ),
            "benchmark_v2_exact_identity_namespace": namespace,
            "task_scope": TASK_SCOPE,
        },
    )
    instance = template.instantiate(name)
    instance.transform = world
    instances[name] = instance
  graph = ConstraintGraph(
      instances=instances,
      constraints=[],
      anchor="part_a",
      metadata={
          "schema_version": (
              "v24_supported_semantic_candidate_graph.v1"
          ),
          "task_scope": TASK_SCOPE,
      },
  )
  assembly = AssemblyState(instances=instances, constraint_graph=graph)
  return SupportedSemanticCandidateV1(
      candidate=ExactCandidateAssemblyV2(
          assembly=assembly,
          checker=CadQueryCollisionChecker(
              shape_library=shapes,
              allow_aabb_fallback=False,
              exact_timeout_seconds=budget.per_occ_timeout_seconds,
              skip_exact_for_complex_pairs=False,
              exact_worker_only=True,
          ),
          part_a="part_a",
          part_b="part_b",
          socket_a="predicted_a",
          socket_b="predicted_b",
      ),
      semantic_catalog_sha256=semantic_catalog.semantic_catalog_sha256,
      constraint_template_id=execution.template_id,
      constraint_certificate=execution.certificate.payload(),
      refinement=execution.refinement.payload(),
  )


def _base_row(
    proposal: ExactProgramProposalV1 | NoCandidateProposalV1,
) -> dict[str, Any]:
  return {
      "schema_version": "v24_supported_semantic_candidate_terminal.v1",
      "task_scope": TASK_SCOPE,
      "development": True,
      "formal": False,
      "publication_eligible": False,
      "final_test_touched": False,
      "withheld_test_touched": False,
      "method_id": proposal.method_id,
      "opaque_sample_commitment_sha256": (
          proposal.sample.opaque_sample_commitment_sha256
      ),
      "graph_work_id": proposal.sample.graph_work_id,
      "rank": proposal.rank,
      "program_index": proposal.program_index,
      "program_id": getattr(proposal, "program_id", None),
      "program_semantics_certified": False,
      "semantic_catalog_sha256": None,
      "constraint_template_id": None,
      "constraint_certificate": None,
      "bounded_refinement": None,
  }


def execute_supported_semantic_topk_v1(
    proposals: Sequence[
        ExactProgramProposalV1 | NoCandidateProposalV1
    ],
    *,
    resolver: Any,
    semantic_catalog: AuthenticatedConstraintExecutableProgramCatalogV1,
    budget: ExactExecutionBudgetV2 = FIXED_EXACT_BUDGET,
) -> tuple[dict[str, Any], ...]:
  """Execute supported Top-K programs; close unsupported ranks at zero OCC."""

  rows = tuple(proposals)
  if (
      len(rows) != budget.top_k
      or [row.rank for row in rows] != list(range(budget.top_k))
      or len({row.method_id for row in rows}) != 1
      or len({
          row.sample.opaque_sample_commitment_sha256 for row in rows
      }) != 1
  ):
    raise ValueError("supported semantic Top-K domain differs")
  if type(semantic_catalog) is not (
      AuthenticatedConstraintExecutableProgramCatalogV1
  ):
    raise TypeError("supported semantic Top-K catalog differs")
  started = time.monotonic()
  reserved_calls = 0
  result = []
  geometry: ResolvedExactGeometryV1 | None = None
  geometry_error: Exception | None = None
  for proposal in rows:
    ledger = ExactOccCallLedgerV3()
    base = _base_row(proposal)
    if isinstance(proposal, NoCandidateProposalV1):
      terminal = {
          **base,
          "status": "no_candidate",
          "failure_reason": proposal.reason,
          "intended_interface_distance_mm": None,
          "whole_body_interference_volume_mm3": None,
          "contact_feasibility_accepted": False,
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - started,
      }
      terminal["terminal_payload_sha256"] = canonical_sha256(terminal)
      result.append(terminal)
      continue
    try:
      entry = semantic_catalog.require_supported(proposal.program_index)
      if proposal.program_id != entry.descriptor.program_id:
        raise ValueError(
            "proposal program ID differs from semantic catalog"
        )
    except UnsupportedProgramSemantics as error:
      terminal = {
          **base,
          "status": "unsupported_program_semantics",
          "failure_reason": str(error),
          "intended_interface_distance_mm": None,
          "whole_body_interference_volume_mm3": None,
          "contact_feasibility_accepted": False,
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - started,
      }
      terminal["terminal_payload_sha256"] = canonical_sha256(terminal)
      result.append(terminal)
      continue
    if time.monotonic() - started >= budget.query_wall_time_seconds:
      terminal = {
          **base,
          "status": "budget_exhausted",
          "failure_reason": "query_wall_time_exhausted",
          "intended_interface_distance_mm": None,
          "whole_body_interference_volume_mm3": None,
          "contact_feasibility_accepted": False,
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - started,
      }
      terminal["terminal_payload_sha256"] = canonical_sha256(terminal)
      result.append(terminal)
      continue
    if reserved_calls + 2 > budget.max_occ_calls:
      raise RuntimeError("supported semantic OCC reservation drifted")
    reserved_calls += 2
    ledger.reserve(2)
    try:
      if geometry is None and geometry_error is None:
        try:
          geometry = resolver.resolve(proposal.sample.graph_work_id)
        except Exception as error:
          geometry_error = error
      if geometry_error is not None:
        raise geometry_error
      assert geometry is not None
      built = build_supported_semantic_candidate_v1(
          proposal,
          geometry,
          semantic_catalog=semantic_catalog,
          budget=budget,
      )
      candidate = built.candidate
      remaining = budget.query_wall_time_seconds - (
          time.monotonic() - started
      )
      if remaining < 0.5:
        raise TimeoutError(
            "query_wall_time_exhausted_before_distance"
        )
      candidate.checker.exact_timeout_seconds = min(
          budget.per_occ_timeout_seconds, max(0.5, remaining)
      )
      distance = _observed_occ_call_v1(
          ledger,
          call_id=(
              f"{proposal.sample.opaque_sample_commitment_sha256}:"
              f"{proposal.method_id}:{proposal.rank}:distance"
          ),
          operation="intended_interface_distance",
          invoke=lambda: (
              candidate.checker.interface_distance_details_between(
                  candidate.assembly,
                  candidate.part_a,
                  candidate.socket_a,
                  candidate.part_b,
                  candidate.socket_b,
              )
          ),
      )
      remaining = budget.query_wall_time_seconds - (
          time.monotonic() - started
      )
      if remaining < 0.5:
        raise TimeoutError(
            "query_wall_time_exhausted_before_interference"
        )
      candidate.checker.exact_timeout_seconds = min(
          budget.per_occ_timeout_seconds, max(0.5, remaining)
      )
      if _whole_solid_broadphase_skip_v1(
          candidate, epsilon=budget.interference_epsilon_mm3
      ):
        ledger.record_broadphase_skip()
        collisions = []
      else:
        collisions = _observed_occ_call_v1(
            ledger,
            call_id=(
                f"{proposal.sample.opaque_sample_commitment_sha256}:"
                f"{proposal.method_id}:{proposal.rank}:interference"
            ),
            operation="whole_solid_interference",
            invoke=lambda: candidate.checker.check(
                candidate.assembly,
                epsilon=budget.interference_epsilon_mm3,
            ),
        )
      exact_distance = float(distance["distance"])
      interference = max(
          (float(row.volume) for row in collisions), default=0.0
      )
      accepted = bool(
          exact_distance <= budget.contact_tolerance_mm
          and interference <= budget.interference_epsilon_mm3
      )
      terminal = {
          **base,
          "program_semantics_certified": True,
          "semantic_catalog_sha256": (
              built.semantic_catalog_sha256
          ),
          "constraint_template_id": built.constraint_template_id,
          "constraint_certificate": built.constraint_certificate,
          "bounded_refinement": built.refinement,
          "status": "accepted" if accepted else "rejected",
          "failure_reason": (
              None if accepted else "contact_feasibility_rejected"
          ),
          "intended_interface_distance_mm": exact_distance,
          "whole_body_interference_volume_mm3": interference,
          "contact_feasibility_accepted": accepted,
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - started,
      }
    except SemanticSurfaceTypeMismatchV1 as error:
      terminal = {
          **base,
          "status": "surface_type_mismatch",
          "failure_reason": str(error),
          "intended_interface_distance_mm": None,
          "whole_body_interference_volume_mm3": None,
          "contact_feasibility_accepted": False,
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - started,
      }
    except (
        CadKernelError,
        TimeoutError,
        ValueError,
        OSError,
        RuntimeError,
    ) as error:
      if isinstance(error, CadKernelChildProcessError):
        status = f"occ_{error.status}"
      elif isinstance(error, TimeoutError):
        status = "occ_timeout"
      elif ledger.attempted == 0:
        status = "pre_spawn_error"
      else:
        status = "kernel_error"
      terminal = {
          **base,
          "status": status,
          "failure_reason": f"{type(error).__name__}:{error}",
          "intended_interface_distance_mm": None,
          "whole_body_interference_volume_mm3": None,
          "contact_feasibility_accepted": False,
          "occ_call_ledger": ledger.payload(),
          "elapsed_seconds": time.monotonic() - started,
      }
    terminal["terminal_payload_sha256"] = canonical_sha256(terminal)
    result.append(terminal)
  if len(result) != budget.top_k:
    raise RuntimeError("supported semantic terminal coverage differs")
  return tuple(result)


__all__ = [
    "CATALOG_SPEC_PATH",
    "SUPPORTED_PROGRAM_INDICES",
    "SemanticSurfaceTypeMismatchV1",
    "SupportedSemanticCandidateV1",
    "TASK_SCOPE",
    "build_supported_semantic_candidate_v1",
    "execute_supported_semantic_topk_v1",
    "load_supported_semantic_catalog_v1",
]
