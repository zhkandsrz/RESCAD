"""Development evaluator for the fixed V19 dual-objective program proposer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, InitVar
import hashlib
import io
import math
from pathlib import Path
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Any
from typing import Mapping

import torch

from .v19_descriptor_brep_pilot_v1 import (
    FIXED_BUNDLE_DIRECTORY,
    FIXED_CANDIDATE_DOMAIN,
    evaluate_descriptor_pilot_rankings_v1,
)
from .v19_authenticated_training_bundle_v1 import (
    load_v19_authenticated_training_bundle_v1,
)
from .v19_brep_program_model_v1 import BRepProgramLearnerConfig
from .v19_descriptor_brep_program_model_v1 import (
    DescriptorConditionedBRepProgramLearnerV1,
)
from .v19_set_valued_program_learning_v1 import SetValuedProgramTargets
from .v19_training_bundle_protocol_v1 import (
    atomic_write_exclusive,
    canonical_bytes,
    canonical_sha256,
    capture_plain_file,
    cleanup_owned_directory,
    directory_identity,
    plain_child,
    plain_directory,
    publish_directory_noreplace,
    require_sha256,
    strict_json,
)


FUSION_ALPHA = 0.5
PROJECT_ROOT = Path(__file__).resolve().parent
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts" / "development"
BUNDLE_ARTIFACT_SHA256 = (
    "72a6bc8a5acd8ac612047090ce9047f3538177bea42c3fe81fbe59eafa6ccd5d"
)
JOINT_CHECKPOINT_SHA256 = (
    "fe15882e87ea7be71a8803011127e3caae9435a856f624e848f50f4b3fbdd427"
)
CLASSIFICATION_CHECKPOINT_SHA256 = (
    "a956b040c536f7ccca15a49ad28fa6c3ece7e51854b29c4019645c833a988757"
)
_CHECKPOINT_KEYS = {
    "schema_version",
    "development",
    "formal",
    "final_test_touched",
    "withheld_test_touched",
    "bundle_artifact_sha256",
    "config_source_sha256",
    "config_payload_sha256",
    "code_bundle_sha256",
    "model_config",
    "seed",
    "candidate_catalog_indices",
    "train_prior_ranking",
    "model_state_dict",
    "optimizer_state_dict",
}
_CAPABILITY_TOKEN = object()
_METHODS = ("classification", "joint", "fused", "prior")


@dataclass(frozen=True, slots=True)
class _PinnedCheckpointSpecV1:
  role: str
  directory_name: str
  revision: str
  receipt_file_sha256: str
  checkpoint_sha256: str
  residual_lambda: float


_JOINT_SPEC = _PinnedCheckpointSpecV1(
    role="joint_residual_consistency",
    directory_name=(
        "v19_descriptor_brep_pilot_v1_commita6d93e9_20260725_run1"
    ),
    revision="a6d93e9fba7549b1bd49a2a2199494e42d9a7000",
    receipt_file_sha256=(
        "d33fbe8cebfaa7c7c067f0373d419dbbba26528d4b248dc0a0f759fddcbf63bd"
    ),
    checkpoint_sha256=JOINT_CHECKPOINT_SHA256,
    residual_lambda=1.0,
)
_CLASSIFICATION_SPEC = _PinnedCheckpointSpecV1(
    role="classification_proposal",
    directory_name=(
        "v19_descriptor_brep_classification_pilot_v1_"
        "commitbbfbc3e_20260725_run1"
    ),
    revision="bbfbc3e817d4fad28dbf9423ab606681bad94aca",
    receipt_file_sha256=(
        "54af5bce3eb6e5ba5d74ca309f4a94b1274ac87d77ebcf18cb79cb923f961251"
    ),
    checkpoint_sha256=CLASSIFICATION_CHECKPOINT_SHA256,
    residual_lambda=0.0,
)


def _model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
  digest = hashlib.sha256()
  for key in sorted(state):
    value = state[key].detach().cpu().contiguous()
    digest.update(canonical_bytes({
        "name": key,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }))
    digest.update(value.numpy().tobytes(order="C"))
  return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedPilotCheckpointV1:
  role: str
  revision: str
  receipt_file_sha256: str
  checkpoint_sha256: str
  checkpoint_bytes: int
  code_bundle_sha256: str
  config_source_sha256: str
  config_payload_sha256: str
  residual_lambda: float
  model_config: Mapping[str, Any]
  model_state_dict: Mapping[str, torch.Tensor]
  train_prior_ranking: tuple[int, ...]
  _token: InitVar[object]
  model_state_sha256: str = field(init=False)

  def __post_init__(self, _token: object) -> None:
    if _token is not _CAPABILITY_TOKEN:
      raise TypeError("verified pilot checkpoints are factory-only")
    cloned = {
        key: value.detach().clone()
        for key, value in self.model_state_dict.items()
    }
    object.__setattr__(self, "model_state_dict", MappingProxyType(cloned))
    object.__setattr__(
        self, "model_state_sha256", _model_state_sha256(cloned)
    )


@dataclass(frozen=True, slots=True)
class VerifiedDualObjectiveCheckpointsV1:
  classification: VerifiedPilotCheckpointV1
  joint: VerifiedPilotCheckpointV1
  _token: InitVar[object]

  def __post_init__(self, _token: object) -> None:
    if _token is not _CAPABILITY_TOKEN:
      raise TypeError("verified dual checkpoints are factory-only")
    if (
        type(self.classification) is not VerifiedPilotCheckpointV1
        or type(self.joint) is not VerifiedPilotCheckpointV1
    ):
      raise TypeError("verified dual checkpoint members differ")


def _git_blob(revision: str, relative_path: str) -> bytes:
  resolved = subprocess.run(
      ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
      cwd=PROJECT_ROOT,
      capture_output=True,
      check=False,
  )
  if (
      resolved.returncode != 0
      or resolved.stdout.decode("ascii").strip() != revision
  ):
    raise ValueError("pinned pilot revision is unavailable")
  blob = subprocess.run(
      ["git", "show", f"{revision}:{relative_path}"],
      cwd=PROJECT_ROOT,
      capture_output=True,
      check=False,
  )
  if blob.returncode != 0:
    raise ValueError("pinned pilot source blob is unavailable")
  return blob.stdout


def _verify_receipt_self_hash(receipt: Mapping[str, Any]) -> None:
  unsigned = dict(receipt)
  observed = unsigned.pop("receipt_payload_sha256", None)
  if observed != canonical_sha256(unsigned):
    raise ValueError("pinned pilot receipt self-hash differs")


def _load_pinned_checkpoint_v1(
    spec: _PinnedCheckpointSpecV1,
) -> VerifiedPilotCheckpointV1:
  development = plain_directory(DEVELOPMENT_ROOT, label="development root")
  root = plain_directory(
      development / spec.directory_name, label=f"{spec.role} pilot artifact"
  )
  if root.parent != development:
    raise ValueError("pinned pilot artifact escaped development root")
  receipt_capture = capture_plain_file(
      plain_child(root, "receipt.json", label="pilot receipt", kind="file"),
      label="pilot receipt",
  )
  if receipt_capture.sha256 != spec.receipt_file_sha256:
    raise ValueError("pinned pilot receipt bytes differ")
  receipt = strict_json(receipt_capture.raw_bytes, label="pilot receipt")
  if not isinstance(receipt, Mapping):
    raise ValueError("pinned pilot receipt is not an object")
  _verify_receipt_self_hash(receipt)
  expected_boundary = (
      receipt.get("schema_version") == "v19_descriptor_brep_pilot_receipt.v1"
      and receipt.get("development") is True
      and receipt.get("formal") is False
      and receipt.get("publication_eligible") is False
      and receipt.get("final_test_touched") is False
      and receipt.get("withheld_test_touched") is False
      and receipt.get("bundle_artifact_sha256") == BUNDLE_ARTIFACT_SHA256
      and receipt.get("candidate_catalog_indices")
      == list(FIXED_CANDIDATE_DOMAIN)
  )
  if not expected_boundary:
    raise ValueError("pinned pilot receipt boundary differs")
  source_bindings = receipt.get("source_bindings")
  if not isinstance(source_bindings, list) or len(source_bindings) != 3:
    raise ValueError("pinned pilot source binding roster differs")
  allowed_sources = {
      "v19_descriptor_brep_program_model_v1.py",
      "v19_descriptor_brep_pilot_v1.py",
      "tools/run_v19_descriptor_brep_pilot_v1.py",
  }
  for binding in source_bindings:
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"relative_path", "bytes", "sha256"}
        or binding.get("relative_path") not in allowed_sources
    ):
      raise ValueError("pinned pilot source binding differs")
    raw = _git_blob(spec.revision, str(binding["relative_path"]))
    if (
        len(raw) != binding["bytes"]
        or hashlib.sha256(raw).hexdigest() != binding["sha256"]
    ):
      raise ValueError("pinned pilot historical source bytes differ")
  if {row["relative_path"] for row in source_bindings} != allowed_sources:
    raise ValueError("pinned pilot historical source roster differs")
  if receipt.get("code_bundle_sha256") != canonical_sha256(source_bindings):
    raise ValueError("pinned pilot code bundle binding differs")
  config_binding = receipt.get("config")
  if (
      not isinstance(config_binding, Mapping)
      or set(config_binding) != {
          "relative_path", "source_sha256", "payload_sha256",
      }
      or config_binding.get("relative_path")
      != "configs/v19_descriptor_brep_pilot_v1.json"
  ):
    raise ValueError("pinned pilot config binding differs")
  config_raw = _git_blob(spec.revision, str(config_binding["relative_path"]))
  config_payload = strict_json(config_raw, label="historical pilot config")
  if (
      hashlib.sha256(config_raw).hexdigest()
      != config_binding["source_sha256"]
      or canonical_sha256(config_payload) != config_binding["payload_sha256"]
      or not isinstance(config_payload, Mapping)
      or config_payload.get("development") is not True
      or config_payload.get("formal") is not False
      or config_payload.get("final_test_touched") is not False
      or config_payload.get("withheld_test_touched") is not False
      or float(config_payload.get("residual_lambda", -1.0))
      != spec.residual_lambda
      or config_payload.get("candidate_catalog_indices")
      != list(FIXED_CANDIDATE_DOMAIN)
  ):
    raise ValueError("pinned pilot historical config differs")
  checkpoint_binding = receipt.get("checkpoint")
  if (
      not isinstance(checkpoint_binding, Mapping)
      or set(checkpoint_binding) != {"name", "bytes", "sha256"}
      or checkpoint_binding.get("name") != "checkpoint.pt"
      or checkpoint_binding.get("sha256") != spec.checkpoint_sha256
  ):
    raise ValueError("pinned pilot checkpoint binding differs")
  checkpoint_capture = capture_plain_file(
      plain_child(root, "checkpoint.pt", label="pilot checkpoint", kind="file"),
      label="pilot checkpoint",
  )
  if (
      checkpoint_capture.sha256 != spec.checkpoint_sha256
      or checkpoint_capture.byte_count != checkpoint_binding["bytes"]
  ):
    raise ValueError("pinned pilot checkpoint bytes differ")
  checkpoint = torch.load(
      io.BytesIO(checkpoint_capture.raw_bytes),
      map_location="cpu",
      weights_only=True,
  )
  if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_KEYS:
    raise ValueError("pinned pilot checkpoint schema differs")
  prior = _fixed_ranking(
      checkpoint.get("train_prior_ranking"), label="checkpoint train-only prior"
  )
  model_config = checkpoint.get("model_config")
  model_state = checkpoint.get("model_state_dict")
  result = receipt.get("result")
  evaluation_contract = (
      result.get("evaluation_contract") if isinstance(result, Mapping) else None
  )
  if (
      checkpoint.get("schema_version")
      != "v19_descriptor_brep_pilot_checkpoint.v1"
      or checkpoint.get("development") is not True
      or checkpoint.get("formal") is not False
      or checkpoint.get("final_test_touched") is not False
      or checkpoint.get("withheld_test_touched") is not False
      or checkpoint.get("bundle_artifact_sha256") != BUNDLE_ARTIFACT_SHA256
      or checkpoint.get("config_source_sha256")
      != config_binding["source_sha256"]
      or checkpoint.get("config_payload_sha256")
      != config_binding["payload_sha256"]
      or checkpoint.get("code_bundle_sha256")
      != receipt["code_bundle_sha256"]
      or checkpoint.get("candidate_catalog_indices")
      != list(FIXED_CANDIDATE_DOMAIN)
      or receipt.get("train_prior_ranking") != list(prior)
      or not isinstance(model_config, Mapping)
      or dict(model_config) != config_payload.get("model")
      or checkpoint.get("seed") != config_payload.get("seed")
      or not isinstance(model_state, Mapping)
      or not model_state
      or any(
          type(key) is not str
          or not isinstance(value, torch.Tensor)
          or not torch.isfinite(value).all()
          for key, value in model_state.items()
      )
      or not isinstance(evaluation_contract, Mapping)
      or evaluation_contract.get("prior_source") != "train_only"
      or result.get("train_usable_sample_count") != 1472
      or result.get("dev_sample_count") != 394
      or result.get("model_metrics", {}).get("oov_only_miss_count") != 2
      or result.get("model_metrics", {}).get(
          "classification_evaluable_count"
      ) != 392
  ):
    raise ValueError("pinned pilot checkpoint/receipt replay differs")
  return VerifiedPilotCheckpointV1(
      role=spec.role,
      revision=spec.revision,
      receipt_file_sha256=receipt_capture.sha256,
      checkpoint_sha256=checkpoint_capture.sha256,
      checkpoint_bytes=checkpoint_capture.byte_count,
      code_bundle_sha256=str(receipt["code_bundle_sha256"]),
      config_source_sha256=str(config_binding["source_sha256"]),
      config_payload_sha256=str(config_binding["payload_sha256"]),
      residual_lambda=spec.residual_lambda,
      model_config=MappingProxyType(dict(model_config)),
      model_state_dict=MappingProxyType(dict(model_state)),
      train_prior_ranking=prior,
      _token=_CAPABILITY_TOKEN,
  )


def load_fixed_dual_objective_checkpoints_v1(
) -> VerifiedDualObjectiveCheckpointsV1:
  """Replay both immutable pilot lineages and load only tensor-safe weights."""

  classification = _load_pinned_checkpoint_v1(_CLASSIFICATION_SPEC)
  joint = _load_pinned_checkpoint_v1(_JOINT_SPEC)
  if (
      classification.model_config != joint.model_config
      or classification.train_prior_ranking != joint.train_prior_ranking
      or tuple(classification.model_state_dict)
      != tuple(joint.model_state_dict)
  ):
    raise ValueError("dual-objective checkpoint architecture/prior differs")
  return VerifiedDualObjectiveCheckpointsV1(
      classification=classification,
      joint=joint,
      _token=_CAPABILITY_TOKEN,
  )


def _safe_new_output(path: str | Path) -> Path:
  output = Path(path)
  development = plain_directory(DEVELOPMENT_ROOT, label="development root")
  if output.exists():
    raise FileExistsError(output)
  if (
      not output.is_absolute()
      or output.parent.resolve(strict=True) != development
      or output.name in {"", ".", ".."}
      or any(
          token in output.name.casefold()
          for token in ("formal", "final", "withheld")
      )
  ):
    raise ValueError("dual evaluator output must be a new development child")
  return output


def _source_bindings() -> list[dict[str, Any]]:
  paths = (
      PROJECT_ROOT / "v19_dual_objective_program_proposer_eval_v1.py",
      PROJECT_ROOT / "tools"
      / "run_v19_dual_objective_program_proposer_eval_v1.py",
  )
  return [
      {
          "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
          "bytes": path.stat().st_size,
          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      }
      for path in paths
  ]


def _checkpoint_identity(
    value: VerifiedPilotCheckpointV1,
) -> tuple[Any, ...]:
  return (
      value.role,
      value.revision,
      value.receipt_file_sha256,
      value.checkpoint_sha256,
      value.checkpoint_bytes,
      value.code_bundle_sha256,
      value.config_source_sha256,
      value.config_payload_sha256,
      value.residual_lambda,
      dict(value.model_config),
      value.train_prior_ranking,
      value.model_state_sha256,
  )


def _verify_checkpoint_capability(
    supplied: VerifiedDualObjectiveCheckpointsV1,
) -> None:
  if type(supplied) is not VerifiedDualObjectiveCheckpointsV1:
    raise TypeError("dual evaluator publication requires verified checkpoints")
  for branch in (supplied.classification, supplied.joint):
    if (
        type(branch) is not VerifiedPilotCheckpointV1
        or _model_state_sha256(branch.model_state_dict)
        != branch.model_state_sha256
    ):
      raise ValueError("verified checkpoint tensor state changed")
  fresh = load_fixed_dual_objective_checkpoints_v1()
  if (
      _checkpoint_identity(supplied.classification)
      != _checkpoint_identity(fresh.classification)
      or _checkpoint_identity(supplied.joint)
      != _checkpoint_identity(fresh.joint)
  ):
    raise ValueError("verified checkpoint capability differs from fixed inputs")


def verify_fixed_paired_evaluation_v1(
    evaluation: Mapping[str, Any],
) -> None:
  """Replay exact paired-row schema, hashes, commitments, and aggregates."""

  if (
      not isinstance(evaluation, Mapping)
      or set(evaluation) != {
          "fusion", "candidate_catalog_indices", "summary", "paired_rows",
      }
      or evaluation.get("candidate_catalog_indices")
      != list(FIXED_CANDIDATE_DOMAIN)
  ):
    raise ValueError("dual evaluator publication payload differs")
  fusion = evaluation["fusion"]
  if (
      not isinstance(fusion, Mapping)
      or set(fusion) != {
          "method",
          "alpha",
          "rank_origin",
          "tie_break",
          "alpha_selection_used_development_metrics",
          "grid_search_performed_during_immutable_replay",
      }
      or fusion.get("method") != "symmetric_reciprocal_rank_fusion.v1"
      or fusion.get("alpha") != FUSION_ALPHA
      or fusion.get("rank_origin") != 1
      or fusion.get("tie_break") != "catalog_index_ascending"
      or fusion.get("alpha_selection_used_development_metrics") is not True
      or fusion.get(
          "grid_search_performed_during_immutable_replay"
      ) is not False
  ):
    raise ValueError("dual evaluator fusion receipt differs")
  rows = evaluation["paired_rows"]
  summary = evaluation["summary"]
  if (
      not isinstance(rows, list)
      or len(rows) != 394
      or not isinstance(summary, Mapping)
      or set(summary) != {
          "sample_count",
          "classification_evaluable_count",
          "oov_only_miss_count",
          *_METHODS,
      }
      or summary.get("sample_count") != 394
      or summary.get("classification_evaluable_count") != 392
      or summary.get("oov_only_miss_count") != 2
  ):
    raise ValueError("dual evaluator fixed development coverage differs")
  expected_row_keys = {
      "row_index",
      "opaque_sample_commitment_sha256",
      "family_cluster_commitment_sha256",
      "assembly_cluster_commitment_sha256",
      "classification_evaluable",
      "oov_only",
      "hits",
      "row_payload_sha256",
  }
  expected_hit_keys = {"program_hit_at_1", "program_hit_at_5"}
  hit_counts = {
      method: {"program_hit_at_1": 0, "program_hit_at_5": 0}
      for method in _METHODS
  }
  sample_commitments: set[str] = set()
  evaluable_count = 0
  oov_count = 0
  for row_index, row in enumerate(rows):
    if not isinstance(row, Mapping) or set(row) != expected_row_keys:
      raise ValueError("dual evaluator paired-row schema differs")
    unsigned = dict(row)
    observed_hash = unsigned.pop("row_payload_sha256")
    require_sha256(observed_hash, label="paired-row self-hash")
    for field_name in (
        "opaque_sample_commitment_sha256",
        "family_cluster_commitment_sha256",
        "assembly_cluster_commitment_sha256",
    ):
      require_sha256(row[field_name], label=f"paired-row {field_name}")
    if (
        observed_hash != canonical_sha256(unsigned)
        or type(row["row_index"]) is not int
        or row["row_index"] != row_index
        or type(row["classification_evaluable"]) is not bool
        or type(row["oov_only"]) is not bool
        or row["oov_only"] is row["classification_evaluable"]
        or not isinstance(row["hits"], Mapping)
        or set(row["hits"]) != set(_METHODS)
    ):
      raise ValueError("dual evaluator paired-row identity/hash differs")
    sample_commitment = row["opaque_sample_commitment_sha256"]
    if sample_commitment in sample_commitments:
      raise ValueError("dual evaluator paired sample commitment repeats")
    sample_commitments.add(sample_commitment)
    evaluable_count += int(row["classification_evaluable"])
    oov_count += int(row["oov_only"])
    for method in _METHODS:
      hits = row["hits"][method]
      if (
          not isinstance(hits, Mapping)
          or set(hits) != expected_hit_keys
          or any(type(hits[key]) is not bool for key in expected_hit_keys)
          or (hits["program_hit_at_1"] and not hits["program_hit_at_5"])
          or (
              row["oov_only"]
              and (
                  hits["program_hit_at_1"]
                  or hits["program_hit_at_5"]
              )
          )
      ):
        raise ValueError("dual evaluator paired hit evidence differs")
      for key in expected_hit_keys:
        hit_counts[method][key] += int(hits[key])
  if evaluable_count != 392 or oov_count != 2:
    raise ValueError("dual evaluator row coverage totals differ")
  expected_metric_keys = {
      "sample_count",
      "classification_evaluable_count",
      "oov_only_miss_count",
      "program_hit_count_at_1",
      "program_hit_count_at_5",
      "program_r_at_1",
      "program_r_at_5",
  }
  for method in _METHODS:
    metric = summary[method]
    count_1 = hit_counts[method]["program_hit_at_1"]
    count_5 = hit_counts[method]["program_hit_at_5"]
    if (
        not isinstance(metric, Mapping)
        or set(metric) != expected_metric_keys
        or metric.get("sample_count") != 394
        or metric.get("classification_evaluable_count") != 392
        or metric.get("oov_only_miss_count") != 2
        or metric.get("program_hit_count_at_1") != count_1
        or metric.get("program_hit_count_at_5") != count_5
        or metric.get("program_r_at_1") != count_1 / 394
        or metric.get("program_r_at_5") != count_5 / 394
    ):
      raise ValueError("dual evaluator paired summary replay differs")


def publish_dual_objective_evaluation_v1(
    output_directory: str | Path,
    *,
    checkpoints: VerifiedDualObjectiveCheckpointsV1,
    evaluation: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
  """Atomically publish paired development rows and their immutable receipt."""

  output = _safe_new_output(output_directory)
  _verify_checkpoint_capability(checkpoints)
  verify_fixed_paired_evaluation_v1(evaluation)
  summary = evaluation["summary"]
  rows = evaluation["paired_rows"]
  runtime_keys = {
      "classification_forward_seconds",
      "joint_forward_seconds",
      "evaluation_wall_seconds",
      "classification_forward_call_count",
      "joint_forward_call_count",
  }
  if (
      not isinstance(runtime, Mapping)
      or set(runtime) != runtime_keys
      or any(
          type(runtime[key]) not in (int, float)
          or not math.isfinite(float(runtime[key]))
          or float(runtime[key]) < 0.0
          for key in (
              "classification_forward_seconds",
              "joint_forward_seconds",
              "evaluation_wall_seconds",
          )
      )
      or any(
          type(runtime[key]) is not int or runtime[key] < 1
          for key in (
              "classification_forward_call_count",
              "joint_forward_call_count",
          )
      )
      or runtime["classification_forward_call_count"]
      != runtime["joint_forward_call_count"]
      or runtime["classification_forward_call_count"] != 13
  ):
    raise ValueError("dual evaluator runtime accounting differs")
  jsonl = b"".join(canonical_bytes(row) + b"\n" for row in rows)
  source_bindings = _source_bindings()
  staging = Path(tempfile.mkdtemp(
      prefix=f".{output.name}.staging-", dir=DEVELOPMENT_ROOT
  ))
  staging_identity = directory_identity(staging)
  try:
    paired_path = staging / "paired_rows.jsonl"
    atomic_write_exclusive(paired_path, jsonl)
    paired_binding = {
        "name": paired_path.name,
        "bytes": len(jsonl),
        "sha256": hashlib.sha256(jsonl).hexdigest(),
        "row_count": len(rows),
    }
    branch_bindings = {
        branch.role: {
            "revision": branch.revision,
            "receipt_file_sha256": branch.receipt_file_sha256,
            "checkpoint_sha256": branch.checkpoint_sha256,
            "checkpoint_bytes": branch.checkpoint_bytes,
            "config_source_sha256": branch.config_source_sha256,
            "config_payload_sha256": branch.config_payload_sha256,
            "code_bundle_sha256": branch.code_bundle_sha256,
            "residual_lambda": branch.residual_lambda,
        }
        for branch in (checkpoints.classification, checkpoints.joint)
    }
    receipt: dict[str, Any] = {
        "schema_version": "v19_dual_objective_program_proposer_eval.v1",
        "task_scope": "oracle_interface_pair_conditioned_program_prediction.v1",
        "development": True,
        "development_tuned_on_dev": True,
        "formal": False,
        "publication_eligible": False,
        "final_test_touched": False,
        "withheld_test_touched": False,
        "bundle_artifact_sha256": BUNDLE_ARTIFACT_SHA256,
        "branch_bindings": branch_bindings,
        "fusion": dict(evaluation["fusion"]),
        "evaluation_contract": {
            "development_split": "dev",
            "set_valued_any_known_target": True,
            "oov_only_is_automatic_miss": True,
            "common_candidate_count": 19,
            "common_candidate_catalog_indices": list(
                FIXED_CANDIDATE_DOMAIN
            ),
            "top_k": [1, 5],
            "two_forward_branches": True,
            "classification_branch": "classification_proposal",
            "joint_branch": "joint_residual_consistency",
            "prior_source": "train_only_checkpoint_replayed",
            "alpha_selection_used_development_metrics": True,
            "grid_search_performed_during_immutable_replay": False,
        },
        "runtime": dict(runtime),
        "summary": dict(summary),
        "paired_rows": paired_binding,
        "source_bindings": source_bindings,
        "code_bundle_sha256": canonical_sha256(source_bindings),
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_exclusive(
        staging / "receipt.json", canonical_bytes(receipt)
    )
    publish_directory_noreplace(
        staging,
        output,
        parent=DEVELOPMENT_ROOT,
        stage_identity=staging_identity,
    )
    return receipt
  finally:
    if staging.exists():
      cleanup_owned_directory(
          staging,
          expected_parent=DEVELOPMENT_ROOT,
          identity=staging_identity,
      )


def run_dual_objective_program_proposer_eval_v1(
    output_directory: str | Path,
) -> dict[str, Any]:
  """Run the two fixed branches once over all 394 development samples."""

  _safe_new_output(output_directory)
  checkpoints = load_fixed_dual_objective_checkpoints_v1()
  handle = load_v19_authenticated_training_bundle_v1(
      FIXED_BUNDLE_DIRECTORY,
      expected_artifact_sha256=BUNDLE_ARTIFACT_SHA256,
  )
  dev_indices = handle.development_evaluation_indices()
  if len(dev_indices) != 394 or handle.counts["catalog_size"] != 19:
    raise ValueError("dual evaluator fixed development domain differs")
  model_config = BRepProgramLearnerConfig(
      **dict(checkpoints.classification.model_config)
  )
  classification = DescriptorConditionedBRepProgramLearnerV1(
      model_config,
      seed=7,
      catalog_descriptors=handle.catalog_descriptors,
  )
  joint = DescriptorConditionedBRepProgramLearnerV1(
      model_config,
      seed=7,
      catalog_descriptors=handle.catalog_descriptors,
  )
  classification.load_state_dict(
      dict(checkpoints.classification.model_state_dict), strict=True
  )
  joint.load_state_dict(dict(checkpoints.joint.model_state_dict), strict=True)
  classification.eval()
  joint.eval()
  classification_rankings: list[tuple[int, ...]] = []
  joint_rankings: list[tuple[int, ...]] = []
  samples = []
  classification_seconds = 0.0
  joint_seconds = 0.0
  classification_calls = 0
  joint_calls = 0
  evaluation_started = time.perf_counter()
  with torch.no_grad():
    for offset in range(0, len(dev_indices), 32):
      batch = handle.collate(dev_indices[offset:offset + 32])
      branch_started = time.perf_counter()
      classification_output = classification(batch.inputs)
      classification_seconds += time.perf_counter() - branch_started
      classification_calls += 1
      branch_started = time.perf_counter()
      joint_output = joint(batch.inputs)
      joint_seconds += time.perf_counter() - branch_started
      joint_calls += 1
      classification_rankings.extend(
          tuple(int(index) for index in row)
          for row in torch.argsort(
              classification_output.logits,
              dim=1,
              descending=True,
              stable=True,
          ).tolist()
      )
      joint_rankings.extend(
          tuple(int(index) for index in row)
          for row in torch.argsort(
              joint_output.logits,
              dim=1,
              descending=True,
              stable=True,
          ).tolist()
      )
      samples.extend(batch.targets.samples)
  wall_seconds = time.perf_counter() - evaluation_started
  targets = SetValuedProgramTargets(samples=tuple(samples), catalog_size=19)
  evaluation = evaluate_dual_objective_rankings_v1(
      classification_rankings,
      joint_rankings,
      targets,
      train_only_prior_ranking=(
          checkpoints.classification.train_prior_ranking
      ),
  )
  summary = evaluation["summary"]
  if (
      summary["sample_count"] != 394
      or summary["classification_evaluable_count"] != 392
      or summary["oov_only_miss_count"] != 2
  ):
    raise ValueError("dual evaluator complete dev accounting differs")
  return publish_dual_objective_evaluation_v1(
      output_directory,
      checkpoints=checkpoints,
      evaluation=evaluation,
      runtime={
          "classification_forward_seconds": classification_seconds,
          "joint_forward_seconds": joint_seconds,
          "evaluation_wall_seconds": wall_seconds,
          "classification_forward_call_count": classification_calls,
          "joint_forward_call_count": joint_calls,
      },
  )


def _fixed_ranking(value: Sequence[int], *, label: str) -> tuple[int, ...]:
  ranking = tuple(value)
  if (
      len(ranking) != 19
      or any(type(index) is not int for index in ranking)
      or set(ranking) != set(FIXED_CANDIDATE_DOMAIN)
  ):
    raise ValueError(f"{label} differs from exact fixed19 candidates")
  return ranking


def symmetric_reciprocal_rank_fusion_v1(
    classification_ranking: Sequence[int],
    joint_ranking: Sequence[int],
) -> tuple[int, ...]:
  """Fuse two fixed19 rankings with the frozen symmetric alpha=0.5 rule."""

  classification = _fixed_ranking(
      classification_ranking, label="classification ranking"
  )
  joint = _fixed_ranking(joint_ranking, label="joint ranking")
  classification_rank = {
      candidate: rank for rank, candidate in enumerate(classification, start=1)
  }
  joint_rank = {
      candidate: rank for rank, candidate in enumerate(joint, start=1)
  }
  scores = {
      candidate: (
          FUSION_ALPHA / classification_rank[candidate]
          + (1.0 - FUSION_ALPHA) / joint_rank[candidate]
      )
      for candidate in FIXED_CANDIDATE_DOMAIN
  }
  return tuple(sorted(
      FIXED_CANDIDATE_DOMAIN,
      key=lambda candidate: (-scores[candidate], candidate),
  ))


def _cluster_commitment(kind: str, opaque_cluster_id: str) -> str:
  return canonical_sha256({
      "schema_version": "v19_dual_objective_cluster_commitment.v1",
      "cluster_kind": kind,
      "opaque_cluster_id": opaque_cluster_id,
  })


def evaluate_dual_objective_rankings_v1(
    classification_rankings: Sequence[Sequence[int]],
    joint_rankings: Sequence[Sequence[int]],
    targets: SetValuedProgramTargets,
    *,
    train_only_prior_ranking: Sequence[int],
) -> dict[str, Any]:
  """Return aggregate metrics and paired per-row hits for clustered analysis."""

  if type(targets) is not SetValuedProgramTargets or targets.catalog_size != 19:
    raise ValueError("dual-objective evaluator requires fixed19 targets")
  classification = tuple(
      _fixed_ranking(row, label="classification ranking")
      for row in classification_rankings
  )
  joint = tuple(
      _fixed_ranking(row, label="joint ranking") for row in joint_rankings
  )
  if len(classification) != len(targets.samples) or len(joint) != len(
      targets.samples
  ):
    raise ValueError("dual-objective ranking sample count differs")
  prior = _fixed_ranking(
      train_only_prior_ranking, label="train-only prior ranking"
  )
  fused = tuple(
      symmetric_reciprocal_rank_fusion_v1(left, right)
      for left, right in zip(classification, joint, strict=True)
  )
  prior_rows = tuple(prior for _ in targets.samples)
  rankings = {
      "classification": classification,
      "joint": joint,
      "fused": fused,
      "prior": prior_rows,
  }
  metrics = {
      name: evaluate_descriptor_pilot_rankings_v1(rows, targets)
      for name, rows in rankings.items()
  }
  paired_rows: list[dict[str, Any]] = []
  for row_index, sample in enumerate(targets.samples):
    known = set(sample.known_catalog_indices)
    row = {
        "row_index": row_index,
        "opaque_sample_commitment_sha256": canonical_sha256({
            "schema_version": "v19_dual_objective_sample_commitment.v1",
            "opaque_query_identity": sample.opaque_query_identity,
            "direction": sample.direction,
            "graph_work_id": sample.graph_work_id,
        }),
        "family_cluster_commitment_sha256": _cluster_commitment(
            "family", sample.family_cluster_id
        ),
        "assembly_cluster_commitment_sha256": _cluster_commitment(
            "assembly", sample.assembly_cluster_id
        ),
        "classification_evaluable": sample.classification_evaluable,
        "oov_only": not sample.classification_evaluable,
        "hits": {
            name: {
                "program_hit_at_1": bool(known.intersection(rows[row_index][:1])),
                "program_hit_at_5": bool(known.intersection(rows[row_index][:5])),
            }
            for name, rows in rankings.items()
        },
    }
    row["row_payload_sha256"] = canonical_sha256(row)
    paired_rows.append(row)
  baseline = metrics["classification"]
  summary: dict[str, Any] = {
      "sample_count": baseline["sample_count"],
      "classification_evaluable_count": baseline[
          "classification_evaluable_count"
      ],
      "oov_only_miss_count": baseline["oov_only_miss_count"],
      **{
          name: {
              key: value
              for key, value in metric.items()
              if key != "candidate_catalog_indices"
          }
          for name, metric in metrics.items()
      },
  }
  return {
      "fusion": {
          "method": "symmetric_reciprocal_rank_fusion.v1",
          "alpha": FUSION_ALPHA,
          "rank_origin": 1,
          "tie_break": "catalog_index_ascending",
          "alpha_selection_used_development_metrics": True,
          "grid_search_performed_during_immutable_replay": False,
      },
      "candidate_catalog_indices": list(FIXED_CANDIDATE_DOMAIN),
      "summary": summary,
      "paired_rows": paired_rows,
  }


__all__ = [
    "CLASSIFICATION_CHECKPOINT_SHA256",
    "FUSION_ALPHA",
    "JOINT_CHECKPOINT_SHA256",
    "VerifiedDualObjectiveCheckpointsV1",
    "VerifiedPilotCheckpointV1",
    "evaluate_dual_objective_rankings_v1",
    "load_fixed_dual_objective_checkpoints_v1",
    "publish_dual_objective_evaluation_v1",
    "run_dual_objective_program_proposer_eval_v1",
    "symmetric_reciprocal_rank_fusion_v1",
    "verify_fixed_paired_evaluation_v1",
]
