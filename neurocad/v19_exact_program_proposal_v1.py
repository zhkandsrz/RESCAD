"""Gold-blind fixed-25 proposals for the V19 exact placement diagnostic.

This module deliberately emits only development proposals.  The selected
interface pair is the receipt-bound pair in the fixed V2 graph-work index.
Program targets are not read while proposals are materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .v19_authenticated_training_bundle_v1 import (
    load_v19_authenticated_training_bundle_v1,
)
from .v19_brep_program_model_v1 import BRepProgramLearnerConfig
from .v19_descriptor_brep_program_model_v1 import (
    DescriptorConditionedBRepProgramLearnerV1,
)
from .v19_descriptor_brep_pilot_v1 import (
    FIXED_BUNDLE_DIRECTORY,
    FIXED_CANDIDATE_DOMAIN,
)
from .v19_dual_objective_program_proposer_eval_v1 import (
    BUNDLE_ARTIFACT_SHA256,
    JOINT_CHECKPOINT_SHA256,
    load_fixed_dual_objective_checkpoints_v1,
    symmetric_reciprocal_rank_fusion_v1,
)
from .v19_program_proposal_protocol_v1 import ProposalBudget


PROJECT_ROOT = Path(__file__).resolve().parent
FIXED_INDEX_PATH = PROJECT_ROOT / (
    "artifacts/development/"
    "v19_query_semantic_adapter_v2_full938_commit5b58150_20260725_run1/"
    "authenticated_query_graph_work_index.v2.json"
)
FIXED_INDEX_SHA256 = (
    "d12dbe5c21da200bd5c16470340bdc570807365ec135241bee439e5da3f85f2c"
)
FIXED_RETRIEVAL_DIRECTORY = PROJECT_ROOT / (
    "artifacts/development/"
    "v19_matched_brep_retrieval_baseline_v1_commit9d25c9a_20260726_run1"
)
FIXED_RETRIEVAL_RECEIPT_SHA256 = (
    "7726a6cf30f800e78f06b6ece7d1204ffbaed8d0d56a6b90bfeeab631f5a77de"
)
FIXED_RETRIEVAL_RANKINGS_SHA256 = (
    "497ef9dfdfc55b8ce1a8378edd0aa0b5b938fa299be07f6a4662f39057c10552"
)
FIXED_SELECTION_SIZE = 25
FIXED_PROPOSAL_BUDGET = ProposalBudget(
    top_k=5,
    candidate_pool_size=19,
    max_occ_calls=10,
    wall_time_seconds=60.0,
)
_SHA256_CHARS = frozenset("0123456789abcdef")


def canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def require_sha256(value: str, *, label: str) -> str:
  if (
      type(value) is not str
      or len(value) != 64
      or set(value) - _SHA256_CHARS
  ):
    raise ValueError(f"{label} is not SHA-256")
  return value


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


def sample_commitment_v1(sample: Mapping[str, Any]) -> str:
  """Commit only to label-free row identity."""

  return canonical_sha256({
      "schema_version": "v19_dual_objective_sample_commitment.v1",
      "opaque_query_identity": require_sha256(
          sample["opaque_query_identity"], label="opaque query"
      ),
      "direction": str(sample["direction"]),
      "graph_work_id": require_sha256(
          sample["graph_work_id"], label="graph work"
      ),
  })


@dataclass(frozen=True, slots=True)
class Fixed25SampleV1:
  dev_row_index: int
  bundle_global_index: int
  opaque_sample_commitment_sha256: str
  opaque_query_identity: str
  opaque_case_identity: str
  direction: str
  graph_work_id: str
  source_contact_ordinal: int

  def __post_init__(self) -> None:
    for value in (
        self.opaque_sample_commitment_sha256,
        self.opaque_query_identity,
        self.opaque_case_identity,
        self.graph_work_id,
    ):
      require_sha256(value, label="fixed25 sample identity")
    if (
        type(self.dev_row_index) is not int
        or self.dev_row_index < 0
        or type(self.bundle_global_index) is not int
        or self.bundle_global_index < 0
        or type(self.source_contact_ordinal) is not int
        or self.source_contact_ordinal < 0
    ):
      raise ValueError("fixed25 sample ordinal differs")

  def payload(self) -> dict[str, Any]:
    return {
        "dev_row_index": self.dev_row_index,
        "bundle_global_index": self.bundle_global_index,
        "opaque_sample_commitment_sha256": (
            self.opaque_sample_commitment_sha256
        ),
        "opaque_query_identity": self.opaque_query_identity,
        "opaque_case_identity": self.opaque_case_identity,
        "direction": self.direction,
        "graph_work_id": self.graph_work_id,
        "source_contact_ordinal": self.source_contact_ordinal,
    }


@dataclass(frozen=True, slots=True)
class ExactProgramProposalV1:
  method_id: str
  sample: Fixed25SampleV1
  rank: int
  program_index: int
  program_id: str
  residual_twist_pair_local: tuple[float, ...]
  residual_source: str
  residual_source_commitment_sha256: str
  proposal_input_commitment_sha256: str

  def __post_init__(self) -> None:
    if (
        self.method_id not in {"learned_fused", "retrieval_diversity"}
        or type(self.sample) is not Fixed25SampleV1
        or type(self.rank) is not int
        or not 0 <= self.rank < FIXED_PROPOSAL_BUDGET.top_k
        or type(self.program_index) is not int
        or self.program_index not in FIXED_CANDIDATE_DOMAIN
        or not self.program_id
        or len(self.residual_twist_pair_local) != 6
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in self.residual_twist_pair_local
        )
    ):
      raise ValueError("exact program proposal differs")
    require_sha256(
        self.residual_source_commitment_sha256,
        label="residual source commitment",
    )
    require_sha256(
        self.proposal_input_commitment_sha256,
        label="proposal input commitment",
    )

  def payload(self) -> dict[str, Any]:
    row = {
        "schema_version": "v19_exact_program_proposal.v1",
        "task_scope": "predicted_mcf_residual_placement_diagnostic",
        "development": True,
        "formal": False,
        "publication_eligible": False,
        "final_test_touched": False,
        "withheld_test_touched": False,
        "method_id": self.method_id,
        "sample": self.sample.payload(),
        "rank": self.rank,
        "program_index": self.program_index,
        "program_id": self.program_id,
        "residual_twist_pair_local": list(
            self.residual_twist_pair_local
        ),
        "residual_source": self.residual_source,
        "residual_source_commitment_sha256": (
            self.residual_source_commitment_sha256
        ),
        "proposal_input_commitment_sha256": (
            self.proposal_input_commitment_sha256
        ),
        "proposal_budget": FIXED_PROPOSAL_BUDGET.trace_dict(),
    }
    row["proposal_payload_sha256"] = canonical_sha256(row)
    return row


@dataclass(frozen=True, slots=True)
class NoCandidateProposalV1:
  method_id: str
  sample: Fixed25SampleV1
  rank: int
  program_index: int
  reason: str
  proposal_input_commitment_sha256: str

  def payload(self) -> dict[str, Any]:
    require_sha256(
        self.proposal_input_commitment_sha256,
        label="no-candidate input commitment",
    )
    row = {
        "schema_version": "v19_exact_program_no_candidate.v1",
        "task_scope": "predicted_mcf_residual_placement_diagnostic",
        "development": True,
        "formal": False,
        "publication_eligible": False,
        "final_test_touched": False,
        "withheld_test_touched": False,
        "method_id": self.method_id,
        "sample": self.sample.payload(),
        "rank": self.rank,
        "program_index": self.program_index,
        "status": "no_candidate",
        "reason": self.reason,
        "proposal_input_commitment_sha256": (
            self.proposal_input_commitment_sha256
        ),
    }
    row["proposal_payload_sha256"] = canonical_sha256(row)
    return row


def _load_fixed_index_payload() -> Mapping[str, Any]:
  raw = FIXED_INDEX_PATH.read_bytes()
  if hashlib.sha256(raw).hexdigest() != FIXED_INDEX_SHA256:
    raise ValueError("fixed V2 index bytes differ")
  payload = _strict_json(raw, label="fixed V2 index")
  unsigned = dict(payload)
  observed = unsigned.pop("index_payload_sha256", None)
  if (
      payload.get("schema_version")
      != "benchmark_v2_authenticated_query_graph_work_index.v2"
      or payload.get("development") is not True
      or payload.get("formal") is not False
      or payload.get("final_test_touched") is not False
      or payload.get("withheld_test_touched") is not False
      or observed != canonical_sha256(unsigned)
      or len(payload.get("samples", ())) != 1870
  ):
    raise ValueError("fixed V2 index boundary differs")
  return payload


def fixed25_selection_v1() -> tuple[Fixed25SampleV1, ...]:
  """Select the lexicographically first 25 commitments without reading gold."""

  payload = _load_fixed_index_payload()
  dev_rows = [
      row for row in payload["samples"]
      if row.get("development_split") == "dev"
  ]
  if len(dev_rows) != 394:
    raise ValueError("fixed development domain differs")
  global_by_sample = {
      str(row["sample_identity_sha256"]): index
      for index, row in enumerate(payload["samples"])
  }
  candidates = []
  for dev_index, row in enumerate(dev_rows):
    # Deliberately do not access row["targets"] here.
    commitment = sample_commitment_v1(row)
    candidates.append((commitment, dev_index, row))
  selected = sorted(candidates, key=lambda value: value[0])[
      :FIXED_SELECTION_SIZE
  ]
  result = tuple(
      Fixed25SampleV1(
          dev_row_index=dev_index,
          bundle_global_index=global_by_sample[str(row["sample_identity_sha256"])],
          opaque_sample_commitment_sha256=commitment,
          opaque_query_identity=str(row["opaque_query_identity"]),
          opaque_case_identity=str(row["opaque_case_identity"]),
          direction=str(row["direction"]),
          graph_work_id=str(row["graph_work_id"]),
          source_contact_ordinal=int(row["source_contact_ordinal"]),
      )
      for commitment, dev_index, row in selected
  )
  if len(result) != 25 or len({
      row.opaque_sample_commitment_sha256 for row in result
  }) != 25:
    raise ValueError("fixed25 selection differs")
  return result


def fixed25_selection_commitment_v1(
    selected: Sequence[Fixed25SampleV1],
) -> str:
  rows = tuple(selected)
  if len(rows) != FIXED_SELECTION_SIZE:
    raise ValueError("fixed25 selection count differs")
  return canonical_sha256({
      "schema_version": "v19_exact_fixed25_selection.v1",
      "selection_rule": (
          "lexicographic_first25_opaque_sample_commitment_sha256"
      ),
      "selected": [row.payload() for row in rows],
  })


def _load_retrieval_rows() -> tuple[Mapping[str, Any], ...]:
  receipt_path = FIXED_RETRIEVAL_DIRECTORY / "receipt.json"
  rankings_path = FIXED_RETRIEVAL_DIRECTORY / "rankings.jsonl"
  receipt_raw = receipt_path.read_bytes()
  if hashlib.sha256(receipt_raw).hexdigest() != FIXED_RETRIEVAL_RECEIPT_SHA256:
    raise ValueError("committed retrieval receipt bytes differ")
  receipt = _strict_json(receipt_raw, label="retrieval receipt")
  unsigned = dict(receipt)
  observed = unsigned.pop("receipt_payload_sha256", None)
  if (
      observed != canonical_sha256(unsigned)
      or receipt.get("development") is not True
      or receipt.get("formal") is not False
      or receipt.get("final_test_touched") is not False
      or receipt.get("withheld_test_touched") is not False
      or receipt.get("rankings", {}).get("sha256")
      != FIXED_RETRIEVAL_RANKINGS_SHA256
      or file_sha256(rankings_path) != FIXED_RETRIEVAL_RANKINGS_SHA256
  ):
    raise ValueError("committed retrieval binding differs")
  rows = tuple(
      _strict_json(line, label="retrieval ranking row")
      for line in rankings_path.read_bytes().splitlines()
      if line
  )
  if len(rows) != 394:
    raise ValueError("retrieval ranking count differs")
  for index, row in enumerate(rows):
    unsigned_row = dict(row)
    row_hash = unsigned_row.pop("row_payload_sha256", None)
    if (
        row.get("row_index") != index
        or row_hash != canonical_sha256(unsigned_row)
        or len(row.get("ranking", ())) != 19
        or set(row["ranking"]) != set(FIXED_CANDIDATE_DOMAIN)
        or len(row.get("neighbor_global_indices", ())) != 1472
    ):
      raise ValueError("retrieval ranking row differs")
  return rows


def first_same_class_neighbor_residual_v1(
    *,
    program_index: int,
    neighbor_global_indices: Sequence[int],
    train_targets_by_global_index: Mapping[
        int, Mapping[int, tuple[float, ...]]
    ],
) -> tuple[int, tuple[float, ...]] | None:
  """Return the first neighbor exemplar in committed neighbor order."""

  if program_index not in FIXED_CANDIDATE_DOMAIN:
    raise ValueError("retrieval candidate is outside fixed19")
  for global_index in neighbor_global_indices:
    by_class = train_targets_by_global_index.get(int(global_index))
    if by_class is None:
      raise ValueError("retrieval neighbor is outside authenticated train")
    residual = by_class.get(program_index)
    if residual is not None:
      return int(global_index), residual
  return None


def _train_residual_lookup(handle: Any) -> Mapping[
    int, Mapping[int, tuple[float, ...]]
]:
  result: dict[int, Mapping[int, tuple[float, ...]]] = {}
  train = handle.training_indices()
  for offset in range(0, len(train), 32):
    batch_indices = train[offset:offset + 32]
    batch = handle.collate(batch_indices)
    for global_index, sample in zip(
        batch_indices, batch.targets.samples, strict=True
    ):
      result[global_index] = {
          group.catalog_index: group.alternatives[0].residual_twist_pair_local
          for group in sample.known_catalog_targets
      }
  if len(result) != 1472:
    raise ValueError("authenticated train residual lookup differs")
  return result


def materialize_fixed25_proposals_v1() -> tuple[
    tuple[Fixed25SampleV1, ...],
    Mapping[str, tuple[ExactProgramProposalV1 | NoCandidateProposalV1, ...]],
]:
  """Replay learned/retrieval proposals without dereferencing dev targets."""

  selected = fixed25_selection_v1()
  selection_sha = fixed25_selection_commitment_v1(selected)
  handle = load_v19_authenticated_training_bundle_v1(
      FIXED_BUNDLE_DIRECTORY,
      expected_artifact_sha256=BUNDLE_ARTIFACT_SHA256,
  )
  checkpoints = load_fixed_dual_objective_checkpoints_v1()
  config = BRepProgramLearnerConfig(
      **dict(checkpoints.classification.model_config)
  )
  classification = DescriptorConditionedBRepProgramLearnerV1(
      config, seed=7, catalog_descriptors=handle.catalog_descriptors
  )
  joint = DescriptorConditionedBRepProgramLearnerV1(
      config, seed=7, catalog_descriptors=handle.catalog_descriptors
  )
  classification.load_state_dict(
      dict(checkpoints.classification.model_state_dict), strict=True
  )
  joint.load_state_dict(dict(checkpoints.joint.model_state_dict), strict=True)
  classification.eval()
  joint.eval()
  catalog = {row.program_index: row for row in handle.catalog_descriptors}
  learned: list[ExactProgramProposalV1] = []
  with torch.inference_mode():
    for offset in range(0, len(selected), 25):
      sample_rows = selected[offset:offset + 25]
      # collate creates a typed batch, but this branch reads inputs only.
      batch = handle.collate(
          tuple(row.bundle_global_index for row in sample_rows)
      )
      classification_output = classification(batch.inputs)
      joint_output = joint(batch.inputs)
      classification_rankings = torch.argsort(
          classification_output.logits,
          dim=1,
          descending=True,
          stable=True,
      ).tolist()
      joint_rankings = torch.argsort(
          joint_output.logits,
          dim=1,
          descending=True,
          stable=True,
      ).tolist()
      joint_residuals = torch.cat((
          joint_output.residual_translation,
          joint_output.residual_rotation_vector,
      ), dim=-1).detach().cpu()
      for local, sample in enumerate(sample_rows):
        ranking = symmetric_reciprocal_rank_fusion_v1(
            classification_rankings[local], joint_rankings[local]
        )
        input_sha = canonical_sha256({
            "schema_version": "v19_exact_learned_proposal_input.v1",
            "selection_sha256": selection_sha,
            "sample_commitment_sha256": (
                sample.opaque_sample_commitment_sha256
            ),
            "classification_checkpoint_sha256": (
                checkpoints.classification.checkpoint_sha256
            ),
            "joint_checkpoint_sha256": (
                checkpoints.joint.checkpoint_sha256
            ),
            "fusion_alpha": 0.5,
            "proposal_budget": FIXED_PROPOSAL_BUDGET.trace_dict(),
        })
        for rank, program_index in enumerate(
            ranking[:FIXED_PROPOSAL_BUDGET.top_k]
        ):
          residual = tuple(
              float(value)
              for value in joint_residuals[local, program_index].tolist()
          )
          learned.append(ExactProgramProposalV1(
              method_id="learned_fused",
              sample=sample,
              rank=rank,
              program_index=program_index,
              program_id=catalog[program_index].program_id,
              residual_twist_pair_local=residual,
              residual_source="fixed_joint_checkpoint_candidate_head",
              residual_source_commitment_sha256=canonical_sha256({
                  "checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
                  "sample_commitment_sha256": (
                      sample.opaque_sample_commitment_sha256
                  ),
                  "program_index": program_index,
                  "residual": list(residual),
              }),
              proposal_input_commitment_sha256=input_sha,
          ))
  retrieval_rows = _load_retrieval_rows()
  train_lookup = _train_residual_lookup(handle)
  retrieval: list[ExactProgramProposalV1 | NoCandidateProposalV1] = []
  for sample in selected:
    ranking_row = retrieval_rows[sample.dev_row_index]
    input_sha = canonical_sha256({
        "schema_version": "v19_exact_retrieval_proposal_input.v1",
        "selection_sha256": selection_sha,
        "sample_commitment_sha256": sample.opaque_sample_commitment_sha256,
        "retrieval_rankings_sha256": FIXED_RETRIEVAL_RANKINGS_SHA256,
        "proposal_budget": FIXED_PROPOSAL_BUDGET.trace_dict(),
    })
    for rank, program_index in enumerate(
        ranking_row["ranking"][:FIXED_PROPOSAL_BUDGET.top_k]
    ):
      exemplar = first_same_class_neighbor_residual_v1(
          program_index=int(program_index),
          neighbor_global_indices=ranking_row["neighbor_global_indices"],
          train_targets_by_global_index=train_lookup,
      )
      if exemplar is None:
        retrieval.append(NoCandidateProposalV1(
            method_id="retrieval_diversity",
            sample=sample,
            rank=rank,
            program_index=int(program_index),
            reason="no_same_class_authenticated_train_exemplar",
            proposal_input_commitment_sha256=input_sha,
        ))
        continue
      exemplar_global_index, residual = exemplar
      retrieval.append(ExactProgramProposalV1(
          method_id="retrieval_diversity",
          sample=sample,
          rank=rank,
          program_index=int(program_index),
          program_id=catalog[int(program_index)].program_id,
          residual_twist_pair_local=residual,
          residual_source="first_same_class_committed_train_neighbor",
          residual_source_commitment_sha256=canonical_sha256({
              "retrieval_rankings_sha256": (
                  FIXED_RETRIEVAL_RANKINGS_SHA256
              ),
              "exemplar_global_index": exemplar_global_index,
              "program_index": int(program_index),
              "residual": list(residual),
          }),
          proposal_input_commitment_sha256=input_sha,
      ))
  if len(learned) != 125 or len(retrieval) != 125:
    raise ValueError("fixed25 proposal count differs")
  return selected, {
      "learned_fused": tuple(learned),
      "retrieval_diversity": tuple(retrieval),
  }


def load_fixed25_gold_v1(
    selected: Sequence[Fixed25SampleV1],
) -> Mapping[str, tuple[int, ...]]:
  """Load development targets only after the runner closes all candidates."""

  rows = tuple(selected)
  if fixed25_selection_commitment_v1(rows) != (
      fixed25_selection_commitment_v1(fixed25_selection_v1())
  ):
    raise ValueError("fixed25 gold request selection differs")
  handle = load_v19_authenticated_training_bundle_v1(
      FIXED_BUNDLE_DIRECTORY,
      expected_artifact_sha256=BUNDLE_ARTIFACT_SHA256,
  )
  result: dict[str, tuple[int, ...]] = {}
  for sample in rows:
    target = handle.collate((sample.bundle_global_index,)).targets.samples[0]
    result[sample.opaque_sample_commitment_sha256] = tuple(
        group.catalog_index for group in target.known_catalog_targets
    )
  if len(result) != 25:
    raise ValueError("fixed25 gold coverage differs")
  return result


__all__ = [
    "ExactProgramProposalV1",
    "FIXED_PROPOSAL_BUDGET",
    "Fixed25SampleV1",
    "NoCandidateProposalV1",
    "canonical_sha256",
    "first_same_class_neighbor_residual_v1",
    "fixed25_selection_commitment_v1",
    "fixed25_selection_v1",
    "load_fixed25_gold_v1",
    "materialize_fixed25_proposals_v1",
    "sample_commitment_v1",
]
