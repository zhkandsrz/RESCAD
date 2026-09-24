"""Direct two-body MCF placement for the fixed V19 development diagnostic.

The adapter applies the predicted six-vector as a direct residual between the
two receipt-bound mating coordinate frames.  It does not infer or certify a
finite-program base transform.  In particular, 17/19 categorical descriptors
still have no certified executable base semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
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
from .benchmark_v2_constants import exact_face_binding_sha256
from .benchmark_v2_model_view_v2 import (
    _captured_brep_topology,
    _load_captured_step_shape,
)
from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    capture_file_artifact,
    reverify_captured_file_artifact,
)
from .cadquery_backend import (
    CadKernelChildProcessError,
    CadKernelError,
    CadQueryCollisionChecker,
    shape_bbox,
)
from .domain_types import (
    AssemblyState,
    ConstraintGraph,
    PartTemplate,
    Socket,
    Transform,
)
from .full_graph_face_frame_authority_v1 import _local_face_frame
from .v19_exact_program_proposal_v1 import (
    ExactProgramProposalV1,
    FIXED_INDEX_PATH,
    FIXED_INDEX_SHA256,
    NoCandidateProposalV1,
    canonical_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "cad" / "abc_step_archives"
AUTHORITY_PATH = PROJECT_ROOT / (
    "artifacts/development/"
    "query_mate_semantic_full_domain_v1_938_commit7748530_retry3_20260724/"
    "query_mate_semantic_full_domain_authority_v1.json"
)
AUTHORITY_SHA256 = (
    "a286f9b65076a2a1156b24b75b9321623cba2ac083f15f5a091ec6807b8a2d68"
)
TASK_SCOPE = "predicted_mcf_residual_placement_diagnostic"
SUPPORTED_BASE_SEMANTICS = frozenset({9, 11})
FIXED_EXACT_BUDGET = ExactExecutionBudgetV2(
    top_k=5,
    max_occ_calls=10,
    per_occ_timeout_seconds=8.0,
    query_wall_time_seconds=60.0,
    contact_tolerance_mm=0.1,
    interference_epsilon_mm3=1e-7,
)


def _strict_json(raw: bytes, *, label: str) -> Any:
  def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
      if key in result:
        raise ValueError(f"{label} contains duplicate JSON keys")
      result[key] = value
    return result

  try:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _proper_rotation(value: Any, *, label: str) -> np.ndarray:
  rotation = np.asarray(value, dtype=float).reshape(3, 3)
  if (
      not np.isfinite(rotation).all()
      or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
      or not math.isclose(
          float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7, rel_tol=0
      )
  ):
    raise ValueError(f"{label} is not a proper rotation")
  return rotation


def rotation_vector_matrix_v1(value: Sequence[float]) -> np.ndarray:
  vector = np.asarray(tuple(value), dtype=float).reshape(3)
  if not np.isfinite(vector).all():
    raise ValueError("residual rotation vector is non-finite")
  angle = float(np.linalg.norm(vector))
  if angle <= 1e-12:
    cross = np.asarray([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])
    return _proper_rotation(np.eye(3) + cross, label="small residual rotation")
  axis = vector / angle
  cross = np.asarray([
      [0.0, -axis[2], axis[1]],
      [axis[2], 0.0, -axis[0]],
      [-axis[1], axis[0], 0.0],
  ])
  result = (
      np.eye(3)
      + math.sin(angle) * cross
      + (1.0 - math.cos(angle)) * (cross @ cross)
  )
  return _proper_rotation(result, label="residual rotation")


def direct_mcf_child_transform_v1(
    *,
    frame_a_local: np.ndarray,
    frame_b_local: np.ndarray,
    residual_twist_pair_local: Sequence[float],
) -> Transform:
  """Solve ``I @ F_a @ residual == child_world @ F_b``."""

  left = np.asarray(frame_a_local, dtype=float).reshape(4, 4)
  right = np.asarray(frame_b_local, dtype=float).reshape(4, 4)
  twist = tuple(float(value) for value in residual_twist_pair_local)
  if len(twist) != 6 or any(not math.isfinite(value) for value in twist):
    raise ValueError("direct MCF residual differs")
  for label, frame in (("A", left), ("B", right)):
    _proper_rotation(frame[:3, :3], label=f"MCF {label}")
    if (
        not np.isfinite(frame).all()
        or not np.allclose(frame[3], [0, 0, 0, 1], atol=0, rtol=0)
    ):
      raise ValueError(f"MCF {label} homogeneous frame differs")
  residual = np.eye(4)
  residual[:3, :3] = rotation_vector_matrix_v1(twist[3:])
  residual[:3, 3] = twist[:3]
  child = left @ residual @ np.linalg.inv(right)
  return Transform(
      rotation=_proper_rotation(child[:3, :3], label="child world"),
      translation=np.asarray(child[:3, 3], dtype=float),
  )


@dataclass(frozen=True, slots=True)
class _StepCommitmentV1:
  archive: str
  relative_path: str
  sha256: str
  byte_count: int


@dataclass(frozen=True, slots=True)
class ResolvedExactGeometryV1:
  graph_work_id: str
  step_a: CapturedFileArtifact
  step_b: CapturedFileArtifact
  raw_face_index_a: int
  raw_face_index_b: int
  face_signature_a: str
  face_signature_b: str
  geometry_binding_sha256: str


class FixedExactGeometryResolverV1:
  """Resolve only geometry committed by the pinned V2 index and authority."""

  def __init__(self) -> None:
    index_raw = FIXED_INDEX_PATH.read_bytes()
    authority_raw = AUTHORITY_PATH.read_bytes()
    if hashlib.sha256(index_raw).hexdigest() != FIXED_INDEX_SHA256:
      raise ValueError("fixed V2 exact index bytes differ")
    if hashlib.sha256(authority_raw).hexdigest() != AUTHORITY_SHA256:
      raise ValueError("fixed semantic authority bytes differ")
    index = _strict_json(index_raw, label="fixed V2 exact index")
    authority = _strict_json(authority_raw, label="fixed semantic authority")
    index_unsigned = dict(index)
    index_payload_sha = index_unsigned.pop("index_payload_sha256", None)
    authority_unsigned = dict(authority)
    authority_payload_sha = authority_unsigned.pop(
        "receipt_payload_sha256", None
    )
    source = index.get("source_authority", {})
    if (
        index_payload_sha != canonical_sha256(index_unsigned)
        or authority_payload_sha != canonical_sha256(authority_unsigned)
        or source.get("receipt", {}).get("sha256") != AUTHORITY_SHA256
        or source.get("receipt_payload_sha256") != authority_payload_sha
        or index.get("development") is not True
        or index.get("formal") is not False
        or authority.get("development") is not True
        or authority.get("formal") is not False
        or any(
            payload.get(field) is not False
            for payload in (index, authority)
            for field in ("final_test_touched", "withheld_test_touched")
        )
    ):
      raise ValueError("fixed exact geometry lineage differs")
    self._work = {
        str(row["graph_work_id"]): row for row in index["graph_work"]
    }
    commitments: dict[str, _StepCommitmentV1] = {}
    for query in authority["query_commitments"]:
      assembly = query["assembly_step_bytes_commitment"]
      archive = str(assembly["archive"])
      if Path(archive).name != archive or not archive.endswith(".7z"):
        raise ValueError("authority archive identity differs")
      for body in assembly["body_steps"]:
        relative = str(body["path"])
        key = canonical_sha256(relative)
        value = _StepCommitmentV1(
            archive=archive,
            relative_path=relative,
            sha256=str(body["sha256"]),
            byte_count=int(body["bytes"]),
        )
        previous = commitments.get(key)
        if previous is not None and previous != value:
          raise ValueError("authority STEP commitment conflicts")
        commitments[key] = value
    self._commitments = commitments

  def _capture_endpoint(self, endpoint: Mapping[str, Any]) -> tuple[
      CapturedFileArtifact, int, str
  ]:
    indices = tuple(int(value) for value in endpoint["raw_occ_face_indices"])
    signatures = tuple(
        str(value) for value in endpoint["source_face_signature_sha256s"]
    )
    if len(indices) != 1 or len(signatures) != 1:
      raise ValueError("interface_frame_not_unique")
    source_path_sha = str(endpoint["step"]["source_path_sha256"])
    commitment = self._commitments.get(source_path_sha)
    if commitment is None:
      raise ValueError("STEP path absent from semantic authority")
    relative = PurePosixPath(commitment.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError("authority STEP path escapes dataset")
    candidate = (
        DATASET_ROOT
        / Path(commitment.archive).stem
        / Path(*relative.parts)
    )
    captured = capture_file_artifact(candidate, label="exact diagnostic STEP")
    if (
        canonical_sha256(commitment.relative_path) != source_path_sha
        or captured.sha256 != commitment.sha256
        or captured.byte_count != commitment.byte_count
        or captured.sha256 != endpoint["step"]["sha256"]
        or captured.byte_count != endpoint["step"]["bytes"]
    ):
      raise ValueError("exact diagnostic STEP bytes differ")
    return captured, indices[0], signatures[0]

  def resolve(self, graph_work_id: str) -> ResolvedExactGeometryV1:
    row = self._work.get(graph_work_id)
    if row is None:
      raise ValueError("graph work is absent from fixed index")
    a = self._capture_endpoint(row["endpoints"]["endpoint_a"])
    b = self._capture_endpoint(row["endpoints"]["endpoint_b"])
    binding = canonical_sha256({
        "schema_version": "v19_exact_geometry_binding.v1",
        "fixed_index_sha256": FIXED_INDEX_SHA256,
        "semantic_authority_sha256": AUTHORITY_SHA256,
        "graph_work_id": graph_work_id,
        "endpoint_a": {
            "step_sha256": a[0].sha256,
            "raw_occ_face_index": a[1],
            "face_signature_sha256": a[2],
        },
        "endpoint_b": {
            "step_sha256": b[0].sha256,
            "raw_occ_face_index": b[1],
            "face_signature_sha256": b[2],
        },
    })
    return ResolvedExactGeometryV1(
        graph_work_id=graph_work_id,
        step_a=a[0],
        step_b=b[0],
        raw_face_index_a=a[1],
        raw_face_index_b=b[1],
        face_signature_a=a[2],
        face_signature_b=b[2],
        geometry_binding_sha256=binding,
    )


def _face_frame(
    shape: Any, *, raw_index: int, expected_signature: str
) -> tuple[np.ndarray, Any]:
  faces, _face_edges, _edge_groups, signatures = _captured_brep_topology(shape)
  if not 0 <= raw_index < len(faces):
    raise ValueError("raw OCC face index is outside STEP")
  if signatures[raw_index] != expected_signature:
    raise ValueError("raw OCC face signature differs")
  origin, rotation = _local_face_frame(faces[raw_index])
  frame = np.eye(4)
  frame[:3, :3] = rotation
  frame[:3, 3] = origin
  return frame, faces[raw_index]


def _socket(
    *,
    name: str,
    frame: np.ndarray,
    raw_index: int,
    signature: str,
    namespace: str,
    interface_id: str,
) -> Socket:
  identity = Transform.identity().to_dict()
  metadata = {
      "private_exact_interface_id": interface_id,
      "private_exact_raw_face_indices": [raw_index],
      "private_exact_raw_face_signature_sha256s": [signature],
      "private_exact_face_identity_proof": "occ_tshape_partner_bijection",
  }
  metadata["private_exact_binding_sha256"] = exact_face_binding_sha256(
      namespace=namespace,
      raw_to_randomized_transform=identity,
      interface_id=interface_id,
      raw_face_indices=[raw_index],
      raw_face_signature_sha256s=[signature],
      face_identity_proof="occ_tshape_partner_bijection",
  )
  return Socket(
      name=name,
      kind="predicted_mcf_interface",
      origin=frame[:3, 3],
      axis=frame[:3, 2],
      normal=frame[:3, 2],
      x_axis=frame[:3, 0],
      y_axis=frame[:3, 1],
      z_axis=frame[:3, 2],
      metadata=metadata,
  )


def build_direct_mcf_candidate_v1(
    proposal: ExactProgramProposalV1,
    geometry: ResolvedExactGeometryV1,
    *,
    budget: ExactExecutionBudgetV2 = FIXED_EXACT_BUDGET,
) -> ExactCandidateAssemblyV2:
  if proposal.sample.graph_work_id != geometry.graph_work_id:
    raise ValueError("proposal/geometry graph-work binding differs")
  for capture in (geometry.step_a, geometry.step_b):
    reverify_captured_file_artifact(capture, label="exact diagnostic STEP")
  shape_a = _load_captured_step_shape(geometry.step_a)
  shape_b = _load_captured_step_shape(geometry.step_b)
  frame_a, _ = _face_frame(
      shape_a,
      raw_index=geometry.raw_face_index_a,
      expected_signature=geometry.face_signature_a,
  )
  frame_b, _ = _face_frame(
      shape_b,
      raw_index=geometry.raw_face_index_b,
      expected_signature=geometry.face_signature_b,
  )
  child = direct_mcf_child_transform_v1(
      frame_a_local=frame_a,
      frame_b_local=frame_b,
      residual_twist_pair_local=proposal.residual_twist_pair_local,
  )
  namespace = canonical_sha256({
      "schema_version": "v19_direct_mcf_identity_namespace.v1",
      "geometry_binding_sha256": geometry.geometry_binding_sha256,
      "proposal_input_commitment_sha256": (
          proposal.proposal_input_commitment_sha256
      ),
  })
  interfaces = (
      canonical_sha256({
          "namespace": namespace, "slot": "a",
          "raw_occ_face_index": geometry.raw_face_index_a,
          "face_signature_sha256": geometry.face_signature_a,
      }),
      canonical_sha256({
          "namespace": namespace, "slot": "b",
          "raw_occ_face_index": geometry.raw_face_index_b,
          "face_signature_sha256": geometry.face_signature_b,
      }),
  )
  shapes = {"part_a": shape_a, "part_b": shape_b}
  instances = {}
  for slot, name, capture, shape, frame, raw_index, signature, world in (
      (
          "a", "part_a", geometry.step_a, shape_a, frame_a,
          geometry.raw_face_index_a, geometry.face_signature_a,
          Transform.identity(),
      ),
      (
          "b", "part_b", geometry.step_b, shape_b, frame_b,
          geometry.raw_face_index_b, geometry.face_signature_b, child,
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
          "schema_version": "v19_direct_mcf_candidate_graph.v1",
          "task_scope": TASK_SCOPE,
      },
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
      part_a="part_a",
      part_b="part_b",
      socket_a="predicted_a",
      socket_b="predicted_b",
  )


def _base_row(
    proposal: ExactProgramProposalV1 | NoCandidateProposalV1,
) -> dict[str, Any]:
  base_supported = proposal.program_index in SUPPORTED_BASE_SEMANTICS
  return {
      "schema_version": "v19_exact_candidate_terminal.v1",
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
      "base_semantics_available": base_supported,
      "base_semantics_used": False,
      "base_semantics_claim": (
          "not_used_direct_mcf_residual_only"
          if base_supported
          else "unsupported_program_semantics_not_certified"
      ),
  }


def execute_direct_mcf_topk_v1(
    proposals: Sequence[ExactProgramProposalV1 | NoCandidateProposalV1],
    *,
    resolver: FixedExactGeometryResolverV1,
    budget: ExactExecutionBudgetV2 = FIXED_EXACT_BUDGET,
) -> tuple[dict[str, Any], ...]:
  """Close all five candidate ranks under one common exact budget."""

  rows = tuple(proposals)
  if (
      len(rows) != budget.top_k
      or [row.rank for row in rows] != list(range(budget.top_k))
      or len({row.method_id for row in rows}) != 1
      or len({
          row.sample.opaque_sample_commitment_sha256 for row in rows
      }) != 1
  ):
    raise ValueError("exact direct-MCF Top-K domain differs")
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
      raise RuntimeError("fixed top5/max10 OCC reservation drifted")
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
      candidate = build_direct_mcf_candidate_v1(
          proposal, geometry, budget=budget
      )
      remaining = budget.query_wall_time_seconds - (
          time.monotonic() - started
      )
      if remaining < 0.5:
        raise TimeoutError("query_wall_time_exhausted_before_distance")
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
          invoke=lambda: candidate.checker.interface_distance_details_between(
              candidate.assembly,
              candidate.part_a,
              candidate.socket_a,
              candidate.part_b,
              candidate.socket_b,
          ),
      )
      remaining = budget.query_wall_time_seconds - (
          time.monotonic() - started
      )
      if remaining < 0.5:
        raise TimeoutError("query_wall_time_exhausted_before_interference")
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
          "geometry_binding_sha256": geometry.geometry_binding_sha256,
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
    except (CadKernelError, TimeoutError, ValueError, OSError, RuntimeError) as error:
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
    raise RuntimeError("exact direct-MCF terminal coverage differs")
  return tuple(result)


__all__ = [
    "FIXED_EXACT_BUDGET",
    "FixedExactGeometryResolverV1",
    "ResolvedExactGeometryV1",
    "SUPPORTED_BASE_SEMANTICS",
    "TASK_SCOPE",
    "build_direct_mcf_candidate_v1",
    "direct_mcf_child_transform_v1",
    "execute_direct_mcf_topk_v1",
    "rotation_vector_matrix_v1",
]
