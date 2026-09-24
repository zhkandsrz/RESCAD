"""Development-only descriptor-conditioned B-Rep pilot orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile
import time
from types import MappingProxyType
from typing import Mapping
from typing import Any

import torch

from .v19_authenticated_training_bundle_v1 import (
    V19AuthenticatedTrainingBatchV1,
    load_v19_authenticated_training_bundle_v1,
)
from .v19_brep_program_model_v1 import BRepProgramLearnerConfig
from .v19_descriptor_brep_program_model_v1 import (
    DescriptorConditionedBRepProgramLearnerV1,
)
from .v19_set_valued_program_learning_v1 import (
    JointSetProgramResidualLossV1,
    SetValuedProgramSampleV1,
    SetValuedProgramTargets,
    evaluate_set_valued_topk_v1,
    set_valued_program_training_loss_v1,
)
from .v19_training_bundle_protocol_v1 import (
    atomic_write_exclusive,
    canonical_sha256,
    cleanup_owned_directory,
    directory_identity,
    publish_directory_noreplace,
)


FIXED_CANDIDATE_DOMAIN = tuple(range(19))
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PILOT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "v19_descriptor_brep_pilot_v1.json"
)
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts" / "development"
FIXED_BUNDLE_DIRECTORY = (
    DEVELOPMENT_ROOT
    / "v19_authenticated_training_bundle_v1_commit118c9db_20260725_run1"
)
FIXED_BUNDLE_ARTIFACT_SHA256 = (
    "72a6bc8a5acd8ac612047090ce9047f3538177bea42c3fe81fbe59eafa6ccd5d"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_KEYS = {
    "schema_version",
    "development",
    "formal",
    "final_test_touched",
    "withheld_test_touched",
    "seed",
    "epochs",
    "batch_size",
    "optimizer",
    "model",
    "residual_lambda",
    "candidate_catalog_indices",
    "device",
    "deterministic_algorithms",
}


@dataclass(frozen=True, slots=True)
class DescriptorBRepPilotConfigV1:
  seed: int
  epochs: int
  batch_size: int
  optimizer_name: str
  learning_rate: float
  weight_decay: float
  model_config: Mapping[str, Any]
  residual_lambda: float
  candidate_catalog_indices: tuple[int, ...]
  device: str
  deterministic_algorithms: bool
  development: bool
  formal: bool
  final_test_touched: bool
  withheld_test_touched: bool
  source_sha256: str
  payload_sha256: str


def _strict_json_object(raw: bytes, *, label: str) -> Mapping[str, Any]:
  def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
      if key in result:
        raise ValueError(f"{label} contains duplicate keys")
      result[key] = value
    return result

  def reject_constant(value: str) -> None:
    raise ValueError(f"{label} contains non-finite number {value}")

  try:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
  except (UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict JSON") from error
  if not isinstance(payload, Mapping):
    raise ValueError(f"{label} must be a JSON object")
  return payload


def load_descriptor_pilot_config_v1(
    path: str | Path,
) -> DescriptorBRepPilotConfigV1:
  """Load the one explicit, hash-receipted pilot hyperparameter source."""

  config_path = Path(path)
  if not config_path.is_file() or config_path.is_symlink():
    raise ValueError("descriptor pilot config must be a plain file")
  raw = config_path.read_bytes()
  payload = _strict_json_object(raw, label="descriptor pilot config")
  if (
      set(payload) != _CONFIG_KEYS
      or payload.get("schema_version")
      != "v19_descriptor_brep_pilot_config.v1"
      or payload.get("development") is not True
      or payload.get("formal") is not False
      or payload.get("final_test_touched") is not False
      or payload.get("withheld_test_touched") is not False
      or payload.get("deterministic_algorithms") is not True
      or payload.get("device") != "cpu"
  ):
    raise ValueError("descriptor pilot config boundary differs")
  seed = payload["seed"]
  epochs = payload["epochs"]
  batch_size = payload["batch_size"]
  residual_lambda = payload["residual_lambda"]
  candidates = payload["candidate_catalog_indices"]
  optimizer = payload["optimizer"]
  model = payload["model"]
  if (
      type(seed) is not int
      or type(epochs) is not int
      or epochs < 1
      or type(batch_size) is not int
      or not 1 <= batch_size <= 32
      or type(residual_lambda) not in (int, float)
      or not math.isfinite(float(residual_lambda))
      or float(residual_lambda) < 0.0
      or not isinstance(candidates, list)
      or tuple(candidates) != FIXED_CANDIDATE_DOMAIN
      or any(type(value) is not int for value in candidates)
      or not isinstance(optimizer, Mapping)
      or set(optimizer) != {"name", "learning_rate", "weight_decay"}
      or optimizer.get("name") != "AdamW"
      or not isinstance(model, Mapping)
      or set(model) != {
          "catalog_size",
          "hidden_dim",
          "layers",
          "dropout",
          "translation_loss_weight",
          "rotation_loss_weight",
      }
      or model.get("catalog_size") != 19
  ):
    raise ValueError("descriptor pilot hyperparameter contract differs")
  learning_rate = optimizer["learning_rate"]
  weight_decay = optimizer["weight_decay"]
  if any(
      type(value) not in (int, float)
      or not math.isfinite(float(value))
      or float(value) < 0.0
      for value in (learning_rate, weight_decay)
  ) or float(learning_rate) == 0.0:
    raise ValueError("descriptor pilot AdamW hyperparameters differ")
  model_config = {
      str(key): value for key, value in model.items()
  }
  # Construction is the final validation for model scalar ranges/types.
  from .v19_brep_program_model_v1 import BRepProgramLearnerConfig
  BRepProgramLearnerConfig(**model_config)
  return DescriptorBRepPilotConfigV1(
      seed=seed,
      epochs=epochs,
      batch_size=batch_size,
      optimizer_name="AdamW",
      learning_rate=float(learning_rate),
      weight_decay=float(weight_decay),
      model_config=MappingProxyType(model_config),
      residual_lambda=float(residual_lambda),
      candidate_catalog_indices=tuple(candidates),
      device="cpu",
      deterministic_algorithms=True,
      development=True,
      formal=False,
      final_test_touched=False,
      withheld_test_touched=False,
      source_sha256=hashlib.sha256(raw).hexdigest(),
      payload_sha256=hashlib.sha256(
          json.dumps(
              payload,
              ensure_ascii=False,
              sort_keys=True,
              separators=(",", ":"),
              allow_nan=False,
          ).encode("utf-8")
      ).hexdigest(),
  )


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def _safe_new_development_output(path: str | Path) -> Path:
  output = Path(path)
  parent = DEVELOPMENT_ROOT.resolve(strict=True)
  if (
      output.parent.resolve(strict=True) != parent
      or output.name in {"", ".", ".."}
      or any(
          token in output.name.casefold()
          for token in ("formal", "final", "withheld")
      )
      or output.exists()
      or output.is_symlink()
  ):
    if output.exists():
      raise FileExistsError(output)
    raise ValueError("pilot output must be a new direct development child")
  return output


def publish_descriptor_pilot_artifact_v1(
    output_directory: str | Path,
    *,
    model: DescriptorConditionedBRepProgramLearnerV1,
    optimizer: torch.optim.Optimizer,
    config: DescriptorBRepPilotConfigV1,
    bundle_artifact_sha256: str,
    train_prior_ranking: Sequence[int],
    result: Mapping[str, Any],
) -> dict[str, Any]:
  """Atomically publish a hash-bound development checkpoint and receipt."""

  output = _safe_new_development_output(output_directory)
  prior_ranking = tuple(train_prior_ranking)
  if (
      type(model) is not DescriptorConditionedBRepProgramLearnerV1
      or type(config) is not DescriptorBRepPilotConfigV1
      or not isinstance(optimizer, torch.optim.AdamW)
      or model.seed != config.seed
      or asdict(model.config) != dict(config.model_config)
      or len(prior_ranking) != 19
      or any(type(index) is not int for index in prior_ranking)
      or set(prior_ranking) != set(FIXED_CANDIDATE_DOMAIN)
      or _SHA256.fullmatch(bundle_artifact_sha256) is None
      or not isinstance(result, Mapping)
  ):
    raise ValueError("descriptor pilot publication binding differs")
  if config != load_descriptor_pilot_config_v1(DEFAULT_PILOT_CONFIG_PATH):
    raise ValueError("descriptor pilot config differs from its sole source")
  default_config_raw = DEFAULT_PILOT_CONFIG_PATH.read_bytes()
  if hashlib.sha256(default_config_raw).hexdigest() != config.source_sha256:
    raise ValueError("descriptor pilot config source changed after loading")
  try:
    _canonical_bytes(result)
  except (TypeError, ValueError) as error:
    raise ValueError("descriptor pilot result is not canonical JSON") from error

  source_paths = (
      PROJECT_ROOT / "v19_descriptor_brep_program_model_v1.py",
      PROJECT_ROOT / "v19_descriptor_brep_pilot_v1.py",
      PROJECT_ROOT / "tools" / "run_v19_descriptor_brep_pilot_v1.py",
  )
  source_bindings = [
      {
          "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
          "bytes": path.stat().st_size,
          "sha256": _file_sha256(path),
      }
      for path in source_paths
  ]
  code_bundle_sha256 = canonical_sha256(source_bindings)
  staging = Path(tempfile.mkdtemp(
      prefix=f".{output.name}.staging-",
      dir=DEVELOPMENT_ROOT,
  ))
  staging_identity = directory_identity(staging)
  published = False
  try:
    checkpoint_path = staging / "checkpoint.pt"
    checkpoint_buffer = io.BytesIO()
    torch.save(
        {
            "schema_version": "v19_descriptor_brep_pilot_checkpoint.v1",
            "development": True,
            "formal": False,
            "final_test_touched": False,
            "withheld_test_touched": False,
            "bundle_artifact_sha256": bundle_artifact_sha256,
            "config_source_sha256": config.source_sha256,
            "config_payload_sha256": config.payload_sha256,
            "code_bundle_sha256": code_bundle_sha256,
            "model_config": dict(config.model_config),
            "seed": config.seed,
            "candidate_catalog_indices": list(FIXED_CANDIDATE_DOMAIN),
            "train_prior_ranking": list(prior_ranking),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_buffer,
    )
    atomic_write_exclusive(checkpoint_path, checkpoint_buffer.getvalue())
    checkpoint_binding = {
        "name": "checkpoint.pt",
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _file_sha256(checkpoint_path),
    }
    receipt: dict[str, Any] = {
        "schema_version": "v19_descriptor_brep_pilot_receipt.v1",
        "development": True,
        "formal": False,
        "publication_eligible": False,
        "final_test_touched": False,
        "withheld_test_touched": False,
        "task_scope": "oracle_interface_pair_conditioned_program_prediction.v1",
        "bundle_artifact_sha256": bundle_artifact_sha256,
        "config": {
            "relative_path": DEFAULT_PILOT_CONFIG_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "source_sha256": config.source_sha256,
            "payload_sha256": config.payload_sha256,
        },
        "source_bindings": source_bindings,
        "code_bundle_sha256": code_bundle_sha256,
        "candidate_catalog_indices": list(FIXED_CANDIDATE_DOMAIN),
        "train_prior_ranking": list(prior_ranking),
        "checkpoint": checkpoint_binding,
        "result": dict(result),
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_exclusive(
        staging / "receipt.json", _canonical_bytes(receipt)
    )
    publish_directory_noreplace(
        staging,
        output,
        parent=DEVELOPMENT_ROOT,
        stage_identity=staging_identity,
    )
    published = True
    return receipt
  finally:
    if not published and staging.exists():
      cleanup_owned_directory(
          staging,
          expected_parent=DEVELOPMENT_ROOT,
          identity=staging_identity,
      )


def deterministic_epoch_order_v1(
    sample_count: int,
    *,
    seed: int,
    epoch: int,
) -> tuple[int, ...]:
  """Produce the sole seeded training order, independent of dev data."""

  if (
      type(sample_count) is not int
      or sample_count < 1
      or type(seed) is not int
      or type(epoch) is not int
      or epoch < 0
  ):
    raise ValueError("descriptor pilot epoch-order request differs")
  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed + 1_000_003 * epoch)
  return tuple(int(value) for value in torch.randperm(
      sample_count, generator=generator
  ).tolist())


def _batches(values: Sequence[int], size: int) -> tuple[tuple[int, ...], ...]:
  rows = tuple(values)
  return tuple(
      rows[offset:offset + size] for offset in range(0, len(rows), size)
  )


def run_descriptor_brep_pilot_v1(
    output_directory: str | Path,
) -> dict[str, Any]:
  """Train and evaluate the fixed development pilot, then publish once."""

  _safe_new_development_output(output_directory)
  started = time.perf_counter()
  config = load_descriptor_pilot_config_v1(DEFAULT_PILOT_CONFIG_PATH)
  torch.use_deterministic_algorithms(config.deterministic_algorithms)
  torch.manual_seed(config.seed)
  handle = load_v19_authenticated_training_bundle_v1(
      FIXED_BUNDLE_DIRECTORY,
      expected_artifact_sha256=FIXED_BUNDLE_ARTIFACT_SHA256,
  )
  counts = handle.counts
  train_indices = handle.training_indices()
  dev_indices = handle.development_evaluation_indices()
  if (
      len(train_indices) != 1472
      or len(dev_indices) != 394
      or counts["catalog_size"] != 19
  ):
    raise ValueError("descriptor pilot fixed bundle counts differ")
  model = DescriptorConditionedBRepProgramLearnerV1(
      BRepProgramLearnerConfig(**dict(config.model_config)),
      seed=config.seed,
      catalog_descriptors=handle.catalog_descriptors,
  ).to(config.device)
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=config.learning_rate,
      weight_decay=config.weight_decay,
  )

  training_started = time.perf_counter()
  epoch_losses: list[float] = []
  train_samples: list[SetValuedProgramSampleV1] = []
  model.train()
  for epoch in range(config.epochs):
    order = deterministic_epoch_order_v1(
        len(train_indices), seed=config.seed, epoch=epoch
    )
    shuffled = tuple(train_indices[position] for position in order)
    weighted_loss_sum = 0.0
    observed = 0
    for index_batch in _batches(shuffled, config.batch_size):
      batch = handle.collate(index_batch).to(config.device)
      if epoch == 0:
        train_samples.extend(batch.targets.samples)
      optimizer.zero_grad(set_to_none=True)
      loss = descriptor_brep_set_valued_training_loss_v1(
          model,
          batch,
          residual_lambda=config.residual_lambda,
      )
      loss.total.backward()
      optimizer.step()
      count = loss.training_sample_count
      weighted_loss_sum += float(loss.total.detach().cpu()) * count
      observed += count
    if observed != len(train_indices):
      raise ValueError("descriptor pilot epoch sample accounting differs")
    epoch_losses.append(weighted_loss_sum / observed)
  training_seconds = time.perf_counter() - training_started
  if (
      len(train_samples) != len(train_indices)
      or len({sample.sample_key for sample in train_samples})
      != len(train_indices)
  ):
    raise ValueError("descriptor pilot train-only prior coverage differs")
  train_prior = train_only_program_prior_v1(train_samples)

  evaluation_started = time.perf_counter()
  model.eval()
  dev_logits: list[torch.Tensor] = []
  dev_residuals: list[torch.Tensor] = []
  dev_samples: list[SetValuedProgramSampleV1] = []
  global_state = None
  with torch.no_grad():
    for index_batch in _batches(dev_indices, config.batch_size):
      batch = handle.collate(index_batch).to(config.device)
      output = model(batch.inputs)
      dev_logits.append(output.logits.detach().cpu())
      dev_residuals.append(torch.cat(
          (output.residual_translation, output.residual_rotation_vector),
          dim=-1,
      ).detach().cpu())
      dev_samples.extend(batch.targets.samples)
      if global_state is None:
        global_state = batch.global_state
  if len(dev_samples) != len(dev_indices) or global_state is None:
    raise ValueError("descriptor pilot dev coverage differs")
  logits = torch.cat(dev_logits, dim=0)
  residuals = torch.cat(dev_residuals, dim=0)
  dev_targets = SetValuedProgramTargets(
      samples=tuple(dev_samples),
      catalog_size=19,
  )
  model_rankings = tuple(
      tuple(int(index) for index in row)
      for row in torch.argsort(
          logits, dim=1, descending=True, stable=True
      ).tolist()
  )
  model_metrics = evaluate_descriptor_pilot_rankings_v1(
      model_rankings, dev_targets
  )
  prior_rankings = tuple(train_prior for _ in dev_samples)
  prior_metrics = evaluate_descriptor_pilot_rankings_v1(
      prior_rankings, dev_targets
  )
  dev_loss = set_valued_program_training_loss_v1(
      logits,
      residuals,
      dev_targets,
      global_cluster_weights=global_state.global_cluster_weights,
      catalog_class_weights=None,
      residual_lambda=config.residual_lambda,
  )
  evaluation_seconds = time.perf_counter() - evaluation_started
  total_seconds = time.perf_counter() - started
  result = {
      "train_usable_sample_count": len(train_indices),
      "dev_sample_count": len(dev_indices),
      "model_metrics": model_metrics,
      "prior_metrics": prior_metrics,
      "loss": {
          "train_epoch_sample_weighted_mean": epoch_losses,
          "train_final_epoch": epoch_losses[-1],
          "dev_exact_full_split": float(dev_loss.total),
      },
      "runtime_seconds": {
          "training": training_seconds,
          "evaluation": evaluation_seconds,
          "total": total_seconds,
      },
      "evaluation_contract": {
          "training_stage": "program_classification_pretraining",
          "residual_lambda": config.residual_lambda,
          "set_valued_any_known_target": True,
          "oov_only_is_automatic_miss": True,
          "top_k": [1, 5],
          "common_candidate_count": 19,
          "common_candidate_catalog_indices": list(FIXED_CANDIDATE_DOMAIN),
          "prior_source": "train_only",
          "dev_used_for_vocabulary_or_hyperparameters": False,
      },
  }
  return publish_descriptor_pilot_artifact_v1(
      output_directory,
      model=model,
      optimizer=optimizer,
      config=config,
      bundle_artifact_sha256=FIXED_BUNDLE_ARTIFACT_SHA256,
      train_prior_ranking=train_prior,
      result=result,
  )


def descriptor_brep_set_valued_training_loss_v1(
    model: DescriptorConditionedBRepProgramLearnerV1,
    batch: V19AuthenticatedTrainingBatchV1,
    *,
    residual_lambda: float = 1.0,
) -> JointSetProgramResidualLossV1:
  """Bridge descriptor-conditioned outputs to the frozen set-valued objective."""

  if type(model) is not DescriptorConditionedBRepProgramLearnerV1:
    raise TypeError("descriptor pilot requires the descriptor-conditioned model")
  if type(batch) is not V19AuthenticatedTrainingBatchV1:
    raise TypeError("descriptor pilot requires an authenticated bundle batch")
  output = model(batch.inputs)
  residual = torch.cat(
      (output.residual_translation, output.residual_rotation_vector), dim=-1
  )
  expected = (len(batch.targets.samples), 19)
  if output.logits.shape != expected or residual.shape != (*expected, 6):
    raise ValueError("descriptor pilot output shape differs from fixed19")
  return set_valued_program_training_loss_v1(
      output.logits,
      residual,
      batch.targets,
      global_cluster_weights=batch.global_state.global_cluster_weights,
      catalog_class_weights=(
          batch.global_state.catalog_class_weights
          if batch.development_split == "train"
          else None
      ),
      residual_lambda=residual_lambda,
  )


def evaluate_descriptor_pilot_rankings_v1(
    ranked_catalog_indices: Sequence[Sequence[int]],
    targets: SetValuedProgramTargets,
) -> dict[str, Any]:
  """Evaluate exact set-valued R@1/R@5 over the common fixed19 domain."""

  if type(targets) is not SetValuedProgramTargets or targets.catalog_size != 19:
    raise ValueError("descriptor pilot evaluation requires fixed19 targets")
  rankings = tuple(tuple(row) for row in ranked_catalog_indices)
  if any(
      len(row) != 19
      or any(type(index) is not int for index in row)
      or set(row) != set(FIXED_CANDIDATE_DOMAIN)
      for row in rankings
  ):
    raise ValueError("descriptor pilot ranking differs from fixed19 budget")
  at_1 = evaluate_set_valued_topk_v1(rankings, targets, top_k=1)
  at_5 = evaluate_set_valued_topk_v1(rankings, targets, top_k=5)
  if (
      at_1.sample_count != at_5.sample_count
      or at_1.classification_evaluable_count
      != at_5.classification_evaluable_count
      or at_1.oov_only_miss_count != at_5.oov_only_miss_count
  ):
    raise ValueError("descriptor pilot metric accounting differs")
  return {
      "sample_count": at_1.sample_count,
      "classification_evaluable_count": at_1.classification_evaluable_count,
      "oov_only_miss_count": at_1.oov_only_miss_count,
      "program_r_at_1": at_1.recall,
      "program_r_at_5": at_5.recall,
      "program_hit_count_at_1": at_1.hit_count,
      "program_hit_count_at_5": at_5.hit_count,
      "candidate_catalog_indices": list(FIXED_CANDIDATE_DOMAIN),
  }


def train_only_program_prior_v1(
    samples: Sequence[SetValuedProgramSampleV1],
) -> tuple[int, ...]:
  """Return a deterministic frequency prior derived only from usable train."""

  rows = tuple(samples)
  if (
      not rows
      or any(type(row) is not SetValuedProgramSampleV1 for row in rows)
      or any(
          row.development_split != "train"
          or not row.classification_evaluable
          for row in rows
      )
  ):
    raise ValueError("program prior requires usable train-only samples")
  counts = [0] * 19
  for row in rows:
    for index in row.known_catalog_indices:
      if index not in FIXED_CANDIDATE_DOMAIN:
        raise ValueError("train target lies outside fixed19")
      counts[index] += 1
  return tuple(sorted(FIXED_CANDIDATE_DOMAIN, key=lambda index: (-counts[index], index)))


__all__ = [
    "DEFAULT_PILOT_CONFIG_PATH",
    "DescriptorBRepPilotConfigV1",
    "FIXED_BUNDLE_ARTIFACT_SHA256",
    "FIXED_BUNDLE_DIRECTORY",
    "FIXED_CANDIDATE_DOMAIN",
    "descriptor_brep_set_valued_training_loss_v1",
    "deterministic_epoch_order_v1",
    "evaluate_descriptor_pilot_rankings_v1",
    "load_descriptor_pilot_config_v1",
    "publish_descriptor_pilot_artifact_v1",
    "run_descriptor_brep_pilot_v1",
    "train_only_program_prior_v1",
]
