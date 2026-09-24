"""Public prediction transport and pairwise OCC execution for LinkCAD.

This module is deliberately split into two boundaries.  Public prediction
never opens private targets.  Pairwise execution replays a selected structural
orbit to a raw OCC face and invokes the existing authenticated P9/P11 support
executor.  Multi-edge pose consistency is a later authority and is not claimed
by this pairwise diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .benchmark_v2_model_view_v2 import (
    GraphSizeBudget,
    _extract_intrinsic_brep_graph_with_identity,
    _load_captured_step_shape,
)
from .benchmark_v2_training_provenance import capture_file_artifact
from .full_graph_face_frame_authority_v1 import _constraint_surface_type
from .linkcad_brep_cache_v1 import LinkCADBRepGraphCacheV1, attach_brep_graphs
from .linkcad_brep_encoder_v1 import graph_payload_to_tensors
from .linkcad_brep_factorized_model_v1 import BRepFactorizedLinkCADV1
from .linkcad_factorized_model_v1 import tensorize_linkcad_query
from .v19_exact_execution_adapter_v1 import ResolvedExactGeometryV1
from .v19_exact_program_proposal_v1 import (
    ExactProgramProposalV1,
    Fixed25SampleV1,
    canonical_sha256,
)
from .v24_supported_semantic_execution_adapter_v1 import (
    execute_supported_semantic_topk_v1,
    load_supported_semantic_catalog_v1,
)
from .authenticated_exact_execution_v2 import ExactExecutionBudgetV2
from .constraint_manifold_occ_adapter_v9 import OccExactManifoldExecutorV9
from .constraint_manifold_preprocess_v4 import (
    ConstraintManifoldPreprocessEndpointV4,
    ConstraintManifoldPreprocessRequestV4,
    ConstraintManifoldPreprocessSupervisorV4,
)
from .constraint_manifold_solver_v4 import (
    ConstraintManifoldRequestV4,
    solve_constraint_manifold_v4,
)
from .domain_types import Transform


PREDICTION_SCHEMA_VERSION = "linkcad_public_predictions.v1"
PAIR_EXACT_SCHEMA_VERSION = "linkcad_pair_exact_execution.v1"
PAIR_MANIFOLD_SCHEMA_VERSION = "linkcad_pair_manifold_execution.v1"
_GRAPH_BUDGET = GraphSizeBudget(
    max_graphs=1,
    max_faces_per_graph=1024,
    max_edges_per_graph=4096,
    max_examples=1,
    max_program_candidates_per_example=1,
)


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def materialize_public_predictions_v1(
    *,
    public: Mapping[str, Any],
    cache: LinkCADBRepGraphCacheV1,
    checkpoint_path: str | Path,
    beam_size: int = 25,
    seed: int = 1701,
) -> dict[str, Any]:
  """Decode public queries without reading or accepting private target rows."""

  if public.get("contains_private_targets") is not False or beam_size < 1:
    raise ValueError("LinkCAD public prediction scope differs")
  checkpoint = Path(checkpoint_path).resolve()
  model = BRepFactorizedLinkCADV1(
      hidden_dim=64,
      use_language=True,
      use_global=False,
      orbit_mode="attention",
      max_orbits_per_candidate=12,
      seed=seed,
  )
  model.load_state_dict(
      torch.load(checkpoint, map_location="cpu", weights_only=True)
  )
  model.eval()
  rows = []
  with torch.no_grad():
    for public_query in public["queries"]:
      query = tensorize_linkcad_query(public_query, None)
      query = attach_brep_graphs(query, public_query, cache)
      predictions = model.decode_beam(query, beam_size=beam_size)
      role_ids = tuple(str(row["role_id"]) for row in public_query["roles"])
      edge_ids = tuple(str(row["edge_id"]) for row in public_query["functional_edges"])
      hypotheses = []
      for rank, prediction in enumerate(predictions):
        hypothesis = {
            "rank": rank,
            "score": float(prediction.score),
            "candidate_by_role": {
                role_id: candidate_id
                for role_id, candidate_id in zip(
                    role_ids, prediction.candidate_ids, strict=True
                )
            },
            "edge_programs": [
                {
                    "edge_id": edge_id,
                    "mobility": mobility,
                    "support_family": support,
                    "interface_orbit_a": int(orbits[0]),
                    "interface_orbit_b": int(orbits[1]),
                    "interface_alternatives": [
                        {
                            "rank": alternative_rank,
                            "interface_orbit_a": int(alternative[0]),
                            "interface_orbit_b": int(alternative[1]),
                            "model_score": float(alternative[2]),
                        }
                        for alternative_rank, alternative in enumerate(alternatives)
                    ],
                }
                for edge_id, mobility, support, orbits, alternatives in zip(
                    edge_ids,
                    prediction.mobility,
                    prediction.support,
                    prediction.interface_orbits,
                    prediction.interface_alternatives,
                    strict=True,
                )
            ],
        }
        hypothesis["hypothesis_payload_sha256"] = _payload_sha256(hypothesis)
        hypotheses.append(hypothesis)
      row = {
          "query_id": str(public_query["query_id"]),
          "terminal_status": "prediction_ready" if hypotheses else "no_candidate",
          "hypotheses": hypotheses,
      }
      row["row_payload_sha256"] = _payload_sha256(row)
      rows.append(row)
  payload = {
      "schema_version": PREDICTION_SCHEMA_VERSION,
      "scope": "public_candidate_set_top25_orbit_mobility_predictions",
      "private_targets_opened": False,
      "model_kind": "brep_pairwise_factor_linker",
      "learned_higher_order_global_head": False,
      "beam_size": beam_size,
      "checkpoint_sha256": _file_sha256(checkpoint),
      "query_count": len(rows),
      "rows": rows,
  }
  payload["prediction_payload_sha256"] = _payload_sha256(payload)
  return payload


@dataclass(frozen=True, slots=True)
class MobilitySupportContractV1:
  mobility: str
  support_program_index: int
  constrained_dofs: tuple[str, ...]
  free_dofs: tuple[str, ...]

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": "linkcad_mobility_support_contract.v1",
        "mobility": self.mobility,
        "support_program_index": self.support_program_index,
        "constrained_dofs": list(self.constrained_dofs),
        "free_dofs": list(self.free_dofs),
    }


def mobility_support_contract_v1(
    *,
    mobility: str,
    support_program_index: int,
) -> MobilitySupportContractV1:
  """Bind requested motion to a certified geometric support program."""

  if support_program_index == 9:
    free = {
        "fixed": (),
        "revolute": ("normal_yaw",),
    }.get(mobility)
    supported = ("normal_translation", "tilt_x", "tilt_y")
  elif support_program_index == 11:
    free = {
        "fixed": (),
        "revolute": ("axis_yaw",),
        "prismatic": ("axial_translation",),
        "cylindrical": ("axis_yaw", "axial_translation"),
    }.get(mobility)
    supported = ("radial_x", "radial_y", "tilt_x", "tilt_y")
  else:
    raise ValueError("LinkCAD support program is not executable")
  if free is None:
    raise ValueError(
        f"LinkCAD mobility {mobility} is incompatible with support P{support_program_index}"
    )
  return MobilitySupportContractV1(
      mobility=mobility,
      support_program_index=support_program_index,
      constrained_dofs=supported,
      free_dofs=free,
  )


@dataclass(frozen=True, slots=True)
class _ResolvedOrbitFaceV1:
  step_capture: Any
  graph_face_index: int
  raw_face_index: int
  face_signature_sha256: str
  surface_type: str
  orbit_ordinal: int
  orbit_member_count: int


def _resolve_orbit_face_v1(
    *,
    candidate: Mapping[str, Any],
    orbit_ordinal: int,
    dataset_root: Path,
) -> _ResolvedOrbitFaceV1:
  if orbit_ordinal < 0:
    raise ValueError("LinkCAD prediction has no interface orbit")
  path = (dataset_root / str(candidate["step_path"])).resolve()
  capture = capture_file_artifact(path, label="LinkCAD exact STEP")
  if capture.sha256 != candidate["step_sha256"]:
    raise ValueError("LinkCAD exact STEP bytes differ from public candidate")
  shape = _load_captured_step_shape(capture)
  graph_payload, raw_to_graph, signatures = (
      _extract_intrinsic_brep_graph_with_identity(shape, budget=_GRAPH_BUDGET)
  )
  graph = graph_payload_to_tensors(graph_payload)
  if not 0 <= orbit_ordinal < len(graph.orbit_members):
    raise ValueError("LinkCAD interface orbit is outside the replayed graph")
  members = graph.orbit_members[orbit_ordinal]
  representative_graph_node = min(members)
  graph_to_raw = {graph_node: raw for raw, graph_node in raw_to_graph.items()}
  raw_index = graph_to_raw.get(representative_graph_node)
  if raw_index is None:
    raise ValueError("LinkCAD graph node has no raw OCC face")
  faces = list(shape.Faces())
  surface_type = _constraint_surface_type(faces[raw_index])
  return _ResolvedOrbitFaceV1(
      step_capture=capture,
      graph_face_index=representative_graph_node,
      raw_face_index=raw_index,
      face_signature_sha256=signatures[raw_index],
      surface_type=surface_type,
      orbit_ordinal=orbit_ordinal,
      orbit_member_count=len(members),
  )


_IDENTITY_ROW_MAJOR = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _manifold_endpoint_v1(
    *, part_slot: str, resolved: _ResolvedOrbitFaceV1,
) -> ConstraintManifoldPreprocessEndpointV4:
  return ConstraintManifoldPreprocessEndpointV4(
      part_slot=part_slot,
      step_path=str(resolved.step_capture.path),
      step_sha256=resolved.step_capture.sha256,
      graph_face_index=resolved.graph_face_index,
      raw_occ_face_index=resolved.raw_face_index,
      face_signature_sha256=resolved.face_signature_sha256,
      current_world_row_major=_IDENTITY_ROW_MAJOR,
  )


def execute_pair_edge_manifold_v1(
    *,
    query_id: str,
    hypothesis_rank: int,
    edge_id: str,
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
    orbit_a: int,
    orbit_b: int,
    mobility: str,
    dataset_root: str | Path,
) -> dict[str, Any]:
  """Search the selected face-pair manifold, then verify it with exact OCC.

  The search is target-free: it uses only the public candidate parts, predicted
  structural orbits, and predicted mobility.  Its certificate is deliberately
  limited to conditional pairwise physical feasibility; it does not assert
  functional intendedness or simultaneous consistency across all graph edges.
  """

  started = time.monotonic()
  base = {
      "schema_version": PAIR_MANIFOLD_SCHEMA_VERSION,
      "scope": "pairwise_conditional_physical_feasibility_not_intendedness",
      "query_id": query_id,
      "hypothesis_rank": hypothesis_rank,
      "edge_id": edge_id,
      "candidate_id_a": candidate_a["candidate_id"],
      "candidate_id_b": candidate_b["candidate_id"],
      "predicted_interface_orbit_a": orbit_a,
      "predicted_interface_orbit_b": orbit_b,
      "predicted_mobility": mobility,
      "private_targets_opened": False,
  }
  try:
    root = Path(dataset_root).resolve()
    resolved_a = _resolve_orbit_face_v1(
        candidate=candidate_a, orbit_ordinal=orbit_a, dataset_root=root,
    )
    resolved_b = _resolve_orbit_face_v1(
        candidate=candidate_b, orbit_ordinal=orbit_b, dataset_root=root,
    )
    program_index = _support_program(
        resolved_a.surface_type, resolved_b.surface_type,
    )
    mobility_contract = mobility_support_contract_v1(
        mobility=mobility, support_program_index=program_index,
    )
    execution_id = canonical_sha256({
        "schema": PAIR_MANIFOLD_SCHEMA_VERSION,
        "query_id": query_id,
        "hypothesis_rank": hypothesis_rank,
        "edge_id": edge_id,
        "candidate_id_a": candidate_a["candidate_id"],
        "candidate_id_b": candidate_b["candidate_id"],
        "step_sha256_a": resolved_a.step_capture.sha256,
        "step_sha256_b": resolved_b.step_capture.sha256,
        "raw_face_index_a": resolved_a.raw_face_index,
        "raw_face_index_b": resolved_b.raw_face_index,
        "mobility": mobility,
        "program_index": program_index,
    })
    preprocess_request = ConstraintManifoldPreprocessRequestV4(
        query_id=f"linkcad_manifold_{execution_id}",
        program_index=program_index,
        endpoints=(
            _manifold_endpoint_v1(part_slot="a", resolved=resolved_a),
            _manifold_endpoint_v1(part_slot="b", resolved=resolved_b),
        ),
        child_world_row_major=_IDENTITY_ROW_MAJOR,
    )
    preprocess_run = ConstraintManifoldPreprocessSupervisorV4().run(
        preprocess_request,
        wall_time_seconds=30.0,
    )
    preprocess_payload = preprocess_run.receipt.payload()
    if preprocess_run.geometry_authority is None:
      result = {
          **base,
          "status": f"unknown_preprocess_{preprocess_run.receipt.terminal_status}",
          "support_program_index": program_index,
          "mobility_support_contract": mobility_contract.payload(),
          "surface_type_a": resolved_a.surface_type,
          "surface_type_b": resolved_b.surface_type,
          "preprocess_receipt": preprocess_payload,
          "full_candidate_count": 0,
          "exact_domain_count": 0,
          "selected_pose_row_major": None,
          "conditional_physical_feasibility_accepted": False,
          "certificate": None,
          "counterevidence": None,
          "ledger": None,
          "elapsed_seconds": time.monotonic() - started,
      }
    else:
      authority = preprocess_run.geometry_authority
      selected_face_binding = canonical_sha256({
          "schema": "linkcad_selected_face_binding.v1",
          "execution_id": execution_id,
          "face_signature_a": resolved_a.face_signature_sha256,
          "face_signature_b": resolved_b.face_signature_sha256,
          "orbit_a": orbit_a,
          "orbit_b": orbit_b,
      })
      request = ConstraintManifoldRequestV4(
          query_id=f"linkcad_manifold_{execution_id}",
          program_index=program_index,
          child_world_row_major=_IDENTITY_ROW_MAJOR,
          identity_authority_sha256=execution_id,
          selected_face_authority_sha256=selected_face_binding,
          candidate_authority=authority,
      )
      endpoint_payloads = authority.payload()["endpoints"]
      executor = OccExactManifoldExecutorV9(
          step_paths=(resolved_a.step_capture.path, resolved_b.step_capture.path),
          step_sha256s=(
              resolved_a.step_capture.sha256, resolved_b.step_capture.sha256,
          ),
          raw_face_indices=(resolved_a.raw_face_index, resolved_b.raw_face_index),
          face_signatures=(
              resolved_a.face_signature_sha256,
              resolved_b.face_signature_sha256,
          ),
          world_a=Transform(rotation=np.eye(3), translation=np.zeros(3)),
          selected_endpoint_a=endpoint_payloads[0],
      )
      solved = solve_constraint_manifold_v4(
          request,
          exact_executor=executor,
          query_started_at=started,
      )
      certificate = (
          solved.certificate.payload() if solved.certificate is not None else None
      )
      counterevidence = (
          solved.counterevidence.payload()
          if solved.counterevidence is not None else None
      )
      result = {
          **base,
          "status": solved.terminal_status,
          "support_program_index": program_index,
          "mobility_support_contract": mobility_contract.payload(),
          "surface_type_a": resolved_a.surface_type,
          "surface_type_b": resolved_b.surface_type,
          "representative_graph_face_index_a": resolved_a.graph_face_index,
          "representative_graph_face_index_b": resolved_b.graph_face_index,
          "representative_raw_face_index_a": resolved_a.raw_face_index,
          "representative_raw_face_index_b": resolved_b.raw_face_index,
          "orbit_member_count_a": resolved_a.orbit_member_count,
          "orbit_member_count_b": resolved_b.orbit_member_count,
          "preprocess_receipt": preprocess_payload,
          "full_candidate_count": solved.full_candidate_count,
          "exact_domain_count": solved.exact_domain_count,
          "selected_pose_row_major": (
              list(solved.selected.child_world_row_major)
              if solved.selected is not None else None
          ),
          "conditional_physical_feasibility_accepted": (
              solved.certificate is not None and solved.certificate.accepted
          ),
          "certificate": certificate,
          "counterevidence": counterevidence,
          "ledger": dict(solved.ledger),
          "elapsed_seconds": time.monotonic() - started,
      }
  except Exception as error:
    reason = f"{type(error).__name__}:{error}"
    if "unsupported LinkCAD support surfaces" in reason:
      status = "unsupported_support_surfaces"
    elif "incompatible with support" in reason:
      status = "unsupported_mobility_support_pair"
    elif "no interface orbit" in reason:
      status = "no_interface_orbit"
    else:
      status = "pre_execution_error"
    result = {
        **base,
        "status": status,
        "support_program_index": None,
        "mobility_support_contract": None,
        "surface_type_a": None,
        "surface_type_b": None,
        "preprocess_receipt": None,
        "full_candidate_count": 0,
        "exact_domain_count": 0,
        "selected_pose_row_major": None,
        "conditional_physical_feasibility_accepted": False,
        "certificate": None,
        "counterevidence": None,
        "ledger": None,
        "failure_reason": reason,
        "elapsed_seconds": time.monotonic() - started,
    }
  result["result_payload_sha256"] = _payload_sha256(result)
  return result


def _support_program(surface_a: str, surface_b: str) -> int:
  if (surface_a, surface_b) == ("plane", "plane"):
    return 9
  if (surface_a, surface_b) == ("cylinder", "cylinder"):
    return 11
  raise ValueError(
      f"unsupported LinkCAD support surfaces: {surface_a}/{surface_b}"
  )


class _OneGeometryResolverV1:
  def __init__(self, geometry: ResolvedExactGeometryV1) -> None:
    self.geometry = geometry

  def resolve(self, graph_work_id: str) -> ResolvedExactGeometryV1:
    if graph_work_id != self.geometry.graph_work_id:
      raise ValueError("LinkCAD exact graph-work identity differs")
    return self.geometry


def execute_pair_edge_v1(
    *,
    query_id: str,
    hypothesis_rank: int,
    edge_id: str,
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
    orbit_a: int,
    orbit_b: int,
    mobility: str,
    dataset_root: str | Path,
    budget: ExactExecutionBudgetV2 | None = None,
) -> dict[str, Any]:
  """Execute one predicted edge with two isolated OCC calls."""

  started = time.monotonic()
  if budget is None:
    budget = ExactExecutionBudgetV2(
        top_k=1,
        max_occ_calls=2,
        query_wall_time_seconds=30.0,
        per_occ_timeout_seconds=8.0,
        contact_tolerance_mm=0.1,
        interference_epsilon_mm3=1e-7,
    )
  base = {
      "schema_version": PAIR_EXACT_SCHEMA_VERSION,
      "scope": "pairwise_edge_feasibility_not_multi_part_pose_consistency",
      "query_id": query_id,
      "hypothesis_rank": hypothesis_rank,
      "edge_id": edge_id,
      "candidate_id_a": candidate_a["candidate_id"],
      "candidate_id_b": candidate_b["candidate_id"],
      "predicted_interface_orbit_a": orbit_a,
      "predicted_interface_orbit_b": orbit_b,
      "predicted_mobility": mobility,
  }
  try:
    resolved_a = _resolve_orbit_face_v1(
        candidate=candidate_a,
        orbit_ordinal=orbit_a,
        dataset_root=Path(dataset_root).resolve(),
    )
    resolved_b = _resolve_orbit_face_v1(
        candidate=candidate_b,
        orbit_ordinal=orbit_b,
        dataset_root=Path(dataset_root).resolve(),
    )
    program_index = _support_program(
        resolved_a.surface_type, resolved_b.surface_type
    )
    mobility_contract = mobility_support_contract_v1(
        mobility=mobility, support_program_index=program_index
    )
    graph_work_id = canonical_sha256({
        "schema": "linkcad_exact_graph_work.v1",
        "query_id": query_id,
        "rank": hypothesis_rank,
        "edge_id": edge_id,
        "candidate_id_a": candidate_a["candidate_id"],
        "candidate_id_b": candidate_b["candidate_id"],
        "orbit_a": orbit_a,
        "orbit_b": orbit_b,
    })
    sample_identity = canonical_sha256({
        "schema": "linkcad_exact_sample.v1",
        "graph_work_id": graph_work_id,
    })
    sample = Fixed25SampleV1(
        dev_row_index=0,
        bundle_global_index=0,
        opaque_sample_commitment_sha256=sample_identity,
        opaque_query_identity=canonical_sha256({"query_id": query_id}),
        opaque_case_identity=canonical_sha256({"query_id": query_id, "case": "linkcad"}),
        direction="forward",
        graph_work_id=graph_work_id,
        source_contact_ordinal=0,
    )
    catalog = load_supported_semantic_catalog_v1()
    entry = catalog.require_supported(program_index)
    proposal_input = canonical_sha256({
        "schema": "linkcad_exact_proposal_input.v1",
        "graph_work_id": graph_work_id,
        "program_index": program_index,
        "mobility": mobility,
    })
    proposal = ExactProgramProposalV1(
        method_id="learned_fused",
        sample=sample,
        rank=0,
        program_index=program_index,
        program_id=entry.descriptor.program_id,
        residual_twist_pair_local=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        residual_source="linkcad_zero_free_dof_refinement.v1",
        residual_source_commitment_sha256=canonical_sha256({
            "schema": "linkcad_zero_refinement.v1"
        }),
        proposal_input_commitment_sha256=proposal_input,
    )
    geometry = ResolvedExactGeometryV1(
        graph_work_id=graph_work_id,
        step_a=resolved_a.step_capture,
        step_b=resolved_b.step_capture,
        raw_face_index_a=resolved_a.raw_face_index,
        raw_face_index_b=resolved_b.raw_face_index,
        face_signature_a=resolved_a.face_signature_sha256,
        face_signature_b=resolved_b.face_signature_sha256,
        geometry_binding_sha256=canonical_sha256({
            "schema": "linkcad_exact_geometry_binding.v1",
            "graph_work_id": graph_work_id,
            "step_a": resolved_a.step_capture.sha256,
            "step_b": resolved_b.step_capture.sha256,
            "raw_face_a": resolved_a.raw_face_index,
            "raw_face_b": resolved_b.raw_face_index,
        }),
    )
    terminal = execute_supported_semantic_topk_v1(
        (proposal,),
        resolver=_OneGeometryResolverV1(geometry),
        semantic_catalog=catalog,
        budget=budget,
    )[0]
    result = {
        **base,
        "status": terminal["status"],
        "support_program_index": program_index,
        "mobility_support_contract": mobility_contract.payload(),
        "surface_type_a": resolved_a.surface_type,
        "surface_type_b": resolved_b.surface_type,
        "representative_raw_face_index_a": resolved_a.raw_face_index,
        "representative_raw_face_index_b": resolved_b.raw_face_index,
        "orbit_member_count_a": resolved_a.orbit_member_count,
        "orbit_member_count_b": resolved_b.orbit_member_count,
        "program_semantics_certified": terminal["program_semantics_certified"],
        "intended_interface_distance_mm": terminal[
            "intended_interface_distance_mm"
        ],
        "whole_body_interference_volume_mm3": terminal[
            "whole_body_interference_volume_mm3"
        ],
        "contact_feasibility_accepted": terminal[
            "contact_feasibility_accepted"
        ],
        "failure_reason": terminal["failure_reason"],
        "occ_call_ledger": terminal["occ_call_ledger"],
        "elapsed_seconds": time.monotonic() - started,
    }
  except Exception as error:
    reason = f"{type(error).__name__}:{error}"
    if "unsupported LinkCAD support surfaces" in reason:
      status = "unsupported_support_surfaces"
    elif "incompatible with support" in reason:
      status = "unsupported_mobility_support_pair"
    elif "no interface orbit" in reason:
      status = "no_interface_orbit"
    else:
      status = "pre_execution_error"
    result = {
        **base,
        "status": status,
        "support_program_index": None,
        "mobility_support_contract": None,
        "surface_type_a": None,
        "surface_type_b": None,
        "representative_raw_face_index_a": None,
        "representative_raw_face_index_b": None,
        "orbit_member_count_a": None,
        "orbit_member_count_b": None,
        "program_semantics_certified": False,
        "intended_interface_distance_mm": None,
        "whole_body_interference_volume_mm3": None,
        "contact_feasibility_accepted": False,
        "failure_reason": reason,
        "occ_call_ledger": None,
        "elapsed_seconds": time.monotonic() - started,
    }
  result["result_payload_sha256"] = _payload_sha256(result)
  return result


def execute_prediction_subset_v1(
    *,
    public: Mapping[str, Any],
    predictions: Mapping[str, Any],
    dataset_root: str | Path,
    query_limit: int,
    hypotheses_per_query: int = 1,
    execution_mode: str = "manifold",
    interface_alternatives_per_edge: int = 3,
    maximum_attempts_per_query: int = 25,
) -> dict[str, Any]:
  """Run pairwise exact execution for a deterministic public subset."""

  if (
      predictions.get("schema_version") != PREDICTION_SCHEMA_VERSION
      or predictions.get("private_targets_opened") is not False
      or query_limit < 1
      or hypotheses_per_query < 1
      or execution_mode not in {"manifold", "zero_pose"}
      or interface_alternatives_per_edge < 1
      or maximum_attempts_per_query < 1
  ):
    raise ValueError("LinkCAD exact subset scope differs")
  query_by_id = {row["query_id"]: row for row in public["queries"]}
  results = []
  selected_prediction_rows = sorted(
      predictions["rows"], key=lambda row: row["query_id"]
  )[:query_limit]
  for prediction_row in selected_prediction_rows:
    public_query = query_by_id[prediction_row["query_id"]]
    edge_by_id = {
        row["edge_id"]: row for row in public_query["functional_edges"]
    }
    candidates_by_role = {
        role_id: {row["candidate_id"]: row for row in rows}
        for role_id, rows in public_query["candidate_sets"].items()
    }
    attempts_for_query = 0
    for hypothesis in prediction_row["hypotheses"][:hypotheses_per_query]:
      for edge_program in hypothesis["edge_programs"]:
        if attempts_for_query >= maximum_attempts_per_query:
          break
        edge = edge_by_id[edge_program["edge_id"]]
        candidate_id_a = hypothesis["candidate_by_role"][edge["role_a"]]
        candidate_id_b = hypothesis["candidate_by_role"][edge["role_b"]]
        executor = (
            execute_pair_edge_manifold_v1
            if execution_mode == "manifold" else execute_pair_edge_v1
        )
        alternatives = edge_program.get("interface_alternatives") or ({
            "rank": 0,
            "interface_orbit_a": edge_program["interface_orbit_a"],
            "interface_orbit_b": edge_program["interface_orbit_b"],
            "model_score": None,
        },)
        for alternative in alternatives[:interface_alternatives_per_edge]:
          if attempts_for_query >= maximum_attempts_per_query:
            break
          row = executor(
              query_id=prediction_row["query_id"],
              hypothesis_rank=int(hypothesis["rank"]),
              edge_id=edge_program["edge_id"],
              candidate_a=candidates_by_role[edge["role_a"]][candidate_id_a],
              candidate_b=candidates_by_role[edge["role_b"]][candidate_id_b],
              orbit_a=int(alternative["interface_orbit_a"]),
              orbit_b=int(alternative["interface_orbit_b"]),
              mobility=str(edge_program["mobility"]),
              dataset_root=dataset_root,
          )
          row["interface_alternative_rank"] = int(alternative["rank"])
          row["interface_model_score"] = alternative["model_score"]
          row.pop("result_payload_sha256", None)
          row["result_payload_sha256"] = _payload_sha256(row)
          results.append(row)
          attempts_for_query += 1
          accepted = (
              row.get("conditional_physical_feasibility_accepted") is True
              if execution_mode == "manifold"
              else row.get("contact_feasibility_accepted") is True
          )
          if accepted:
            break
  status_counts: dict[str, int] = {}
  for row in results:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
  edge_targets = {
      (row["query_id"], row["hypothesis_rank"], row["edge_id"])
      for row in results
  }
  accepted_edge_targets = {
      (row["query_id"], row["hypothesis_rank"], row["edge_id"])
      for row in results
      if (
          row.get("conditional_physical_feasibility_accepted") is True
          if execution_mode == "manifold"
          else row.get("contact_feasibility_accepted") is True
      )
  }
  payload = {
      "schema_version": "linkcad_pair_exact_subset.v1",
      "scope": "development_public_prediction_pairwise_exact_subset",
      "private_targets_opened": False,
      "execution_mode": execution_mode,
      "prediction_payload_sha256": predictions["prediction_payload_sha256"],
      "query_limit": query_limit,
      "hypotheses_per_query": hypotheses_per_query,
      "interface_alternatives_per_edge": interface_alternatives_per_edge,
      "maximum_attempts_per_query": maximum_attempts_per_query,
      "edge_execution_count": len(edge_targets),
      "manifold_attempt_count": len(results),
      "status_counts": dict(sorted(status_counts.items())),
      "accepted_count": len(accepted_edge_targets),
      "rows": results,
  }
  payload["subset_payload_sha256"] = _payload_sha256(payload)
  return payload


def execute_development_oracle_subset_v1(
    *,
    public: Mapping[str, Any],
    private_targets: Mapping[str, Any],
    direct_supervision: Mapping[str, Any],
    dataset_root: str | Path,
    query_limit: int,
) -> dict[str, Any]:
  """Measure executor ceiling on an explicitly opened development pilot.

  This entry point is intentionally separate from public prediction transport.
  It accepts only the already-development candidate-set schemas and labels every
  output as private-opened, so its numbers cannot be mistaken for blind model
  evidence.
  """

  if (
      public.get("contains_private_targets") is not False
      or private_targets.get("schema_version")
      != "linkcad_candidate_set_private_targets.v1"
      or direct_supervision.get("schema_version")
      != "linkcad_direct_face_orbit_supervision.v1"
      or query_limit < 1
  ):
    raise ValueError("LinkCAD development oracle scope differs")
  public_by_id = {row["query_id"]: row for row in public["queries"]}
  private_by_id = {row["query_id"]: row for row in private_targets["targets"]}
  direct_by_key = {
      (row["query_id"], row["edge_id"], row["side"]): row
      for row in direct_supervision["rows"]
      if row.get("status") == "mapped_type_consistent"
  }
  results = []
  omitted = []
  selected_query_ids = []
  for query_id in sorted(public_by_id):
    public_query = public_by_id[query_id]
    target = private_by_id.get(query_id)
    if target is None:
      continue
    ready_edges = [
        edge["edge_id"]
        for edge in public_query["functional_edges"]
        if (
            (query_id, edge["edge_id"], "a") in direct_by_key
            and (query_id, edge["edge_id"], "b") in direct_by_key
        )
    ]
    if len(ready_edges) != len(public_query["functional_edges"]):
      continue
    selected_query_ids.append(query_id)
    if len(selected_query_ids) == query_limit:
      break
  for query_id in selected_query_ids:
    public_query = public_by_id[query_id]
    target = private_by_id[query_id]
    candidates_by_role = {
        role_id: {row["candidate_id"]: row for row in rows}
        for role_id, rows in public_query["candidate_sets"].items()
    }
    edge_by_id = {row["edge_id"]: row for row in public_query["functional_edges"]}
    target_edge_by_id = {
        row["edge_id"]: row for row in target["functional_edge_targets"]
    }
    for edge_id, edge in edge_by_id.items():
      left = direct_by_key.get((query_id, edge_id, "a"))
      right = direct_by_key.get((query_id, edge_id, "b"))
      edge_target = target_edge_by_id.get(edge_id)
      if left is None or right is None or edge_target is None:
        omitted.append({
            "query_id": query_id,
            "edge_id": edge_id,
            "reason": "complete_direct_orbit_target_absent",
        })
        continue
      candidate_id_a = target["target_candidate_by_role"][edge["role_a"]]
      candidate_id_b = target["target_candidate_by_role"][edge["role_b"]]
      result = execute_pair_edge_manifold_v1(
          query_id=query_id,
          hypothesis_rank=0,
          edge_id=edge_id,
          candidate_a=candidates_by_role[edge["role_a"]][candidate_id_a],
          candidate_b=candidates_by_role[edge["role_b"]][candidate_id_b],
          orbit_a=int(left["structural_orbit_ordinal"]),
          orbit_b=int(right["structural_orbit_ordinal"]),
          mobility=str(edge_target["target_mobility"]),
          dataset_root=dataset_root,
      )
      result["evidence_role"] = "development_oracle_executor_ceiling"
      result["private_targets_opened"] = True
      result.pop("result_payload_sha256", None)
      result["result_payload_sha256"] = _payload_sha256(result)
      results.append(result)
  status_counts: dict[str, int] = {}
  for row in results:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
  payload = {
      "schema_version": "linkcad_pair_manifold_oracle_subset.v1",
      "scope": "development_oracle_executor_ceiling_not_blind_model_evidence",
      "private_targets_opened": True,
      "query_limit": query_limit,
      "selection_policy": (
          "query_id_ascending_among_queries_with_all_edge_endpoints_"
          "mapped_type_consistent_before_exact_execution"
      ),
      "selected_query_ids": selected_query_ids,
      "selected_query_count": len(selected_query_ids),
      "edge_execution_count": len(results),
      "accepted_count": sum(
          row["conditional_physical_feasibility_accepted"] is True
          for row in results
      ),
      "status_counts": dict(sorted(status_counts.items())),
      "omitted_count": len(omitted),
      "omitted": omitted,
      "rows": results,
  }
  payload["subset_payload_sha256"] = _payload_sha256(payload)
  return payload
