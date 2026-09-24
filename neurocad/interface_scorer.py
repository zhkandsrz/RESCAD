"""Lightweight learned interface affordance scorer.

The default model is a dependency-free logistic scorer trained on face features
with labels produced from train assemblies. It predicts P(interface | single
part geometry), not pairwise poses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .benchmark_v2_model_view import (
    BENCHMARK_V2_FEATURE_NAMES,
    ModelViewSanitization,
    benchmark_v2_numeric_feature_vector,
    sanitize_benchmark_v2_model_view,
    validate_benchmark_v2_model_view,
)
from .benchmark_v2_constants import PROTOCOL_VERSION
from .benchmark_v2_training_provenance import (
    INTERFACE_TRAINING_ROW_KEYS,
    INTERFACE_TRAINING_ROW_SCHEMA,
    FORMAL_MODEL_TRAINING_MANIFEST_KEYS,
    MODEL_TRAINING_MANIFEST_SCHEMA,
    TrainingDatasetBinding,
    model_parameter_sha256,
    trainer_code_sha256,
    validate_dataset_manifest,
    validate_finite_numeric_tree,
    validate_formal_training_manifest_counts,
    validate_rows_against_binding,
    validate_training_hyperparameters,
)
from .interface_features import (
    FEATURE_NAMES,
    CandidateInterface,
    extract_candidate_interfaces_from_step,
    interface_feature_vector,
)
from .paths import resolve_path


INTERFACE_SCORER_V2_SCHEMA = "interface_scorer.v2"
INTERFACE_MODEL_PARAMETER_KEYS = (
    "model_type",
    "model_schema",
    "model_input_protocol",
    "feature_names",
    "feature_names_sha256",
    "weights",
    "mean",
    "scale",
    "threshold",
    "metrics_sha256",
)
INTERFACE_MODEL_ALLOWED_KEYS = frozenset(
    {
        *INTERFACE_MODEL_PARAMETER_KEYS,
        "model_parameters_sha256",
        "training_manifest",
        "metrics",
    }
)


def _normalized_model_protocol(protocol: str) -> str:
  value = str(protocol or "legacy").strip().lower()
  if value not in {"legacy", "benchmark_v2"}:
    raise ValueError(
        "model protocol must be 'legacy' or 'benchmark_v2', "
        f"got {protocol!r}"
    )
  return value


@dataclass(frozen=True)
class InterfaceModelInput:
  """Auditable input actually consumed by an interface scorer."""

  protocol: str
  feature_names: tuple[str, ...]
  vector: tuple[float, ...]
  sanitization: Optional[ModelViewSanitization] = None


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="command", required=True)

  train = sub.add_parser("train", help="Train a binary interface scorer.")
  train.add_argument("--train_jsonl", default="neurocad/interface_train.jsonl")
  train.add_argument("--dev_jsonl", default=None)
  train.add_argument("--output_model", default="neurocad/interface_scorer_model.json")
  train.add_argument("--epochs", type=int, default=600)
  train.add_argument("--learning_rate", type=float, default=0.08)
  train.add_argument("--l2", type=float, default=1e-4)
  train.add_argument("--positive_weight", type=float, default=0.0)
  train.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )

  score = sub.add_parser("score", help="Score candidate interfaces from one STEP.")
  score.add_argument("--model", default="neurocad/interface_scorer_model.json")
  score.add_argument("--step", required=True)
  score.add_argument("--part_name", default="")
  score.add_argument("--body_uuid", default="")
  score.add_argument("--top_k", type=int, default=16)
  score.add_argument("--output_json", default=None)
  score.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )

  evaluate = sub.add_parser("eval", help="Evaluate a scorer on labeled JSONL.")
  evaluate.add_argument("--model", default="neurocad/interface_scorer_model.json")
  evaluate.add_argument("--jsonl", required=True)
  evaluate.add_argument("--ks", default="1,3,5,10,16")
  evaluate.add_argument("--output_json", default=None)
  evaluate.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )
  return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  """Resolve the selected scorer subcommand's filesystem arguments."""
  names_by_command = {
      "train": ("train_jsonl", "dev_jsonl", "output_model"),
      "score": ("model", "step", "output_json"),
      "eval": ("model", "jsonl", "output_json"),
  }
  for name in names_by_command.get(str(args.command), ()):
    value = getattr(args, name, None)
    if value:
      setattr(args, name, str(resolve_path(value)))
  return args


def main() -> None:
  args = normalize_paths(build_parser().parse_args())
  if args.command == "train":
    payload = train_model(
        train_jsonl=Path(args.train_jsonl),
        dev_jsonl=None if args.dev_jsonl is None else Path(args.dev_jsonl),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        positive_weight=float(args.positive_weight),
        input_protocol=str(args.input_protocol),
    )
    output = Path(args.output_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    return

  if args.command == "score":
    scorer = InterfaceScorer.load(
        Path(args.model),
        expected_protocol=str(args.input_protocol),
    )
    candidates = extract_candidate_interfaces_from_step(
        args.step,
        part_name=str(args.part_name or ""),
        body_uuid=str(args.body_uuid or ""),
        protocol=str(args.input_protocol),
    )
    scored = scorer.score_candidates(
        candidates,
        protocol=str(args.input_protocol),
    )
    scored.sort(key=lambda item: (-(item.score or 0.0), item.interface_id))
    if int(args.top_k) > 0:
      scored = scored[: int(args.top_k)]
    result = [item.to_dict() for item in scored]
    if args.output_json:
      path = Path(args.output_json)
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result[: min(5, len(result))], indent=2))
    return

  if args.command == "eval":
    scorer = InterfaceScorer.load(
        Path(args.model),
        expected_protocol=str(args.input_protocol),
    )
    rows = _load_rows(
        Path(args.jsonl),
        input_protocol=scorer.model_protocol,
    )
    metrics = evaluate_rows(
        rows=rows,
        scorer=scorer,
        ks=[
            int(item)
            for item in str(args.ks).split(",")
            if str(item).strip().isdigit()
        ],
    )
    if args.output_json:
      path = Path(args.output_json)
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return


class InterfaceScorer:
  """Standardized logistic interface scorer."""

  def __init__(
      self,
      weights: np.ndarray,
      mean: np.ndarray,
      scale: np.ndarray,
      threshold: float = 0.5,
      feature_names: Sequence[str] | None = None,
      model_protocol: str = "legacy",
  ) -> None:
    self.model_protocol = _normalized_model_protocol(model_protocol)
    default_names: Sequence[str] = (
        BENCHMARK_V2_FEATURE_NAMES
        if self.model_protocol == "benchmark_v2"
        else FEATURE_NAMES
    )
    self.feature_names = tuple(str(name) for name in (feature_names or default_names))
    self.weights = np.asarray(weights, dtype=float).reshape(-1)
    self.mean = np.asarray(mean, dtype=float).reshape(-1)
    self.scale = np.asarray(scale, dtype=float).reshape(-1)
    self.threshold = float(threshold)
    expected = len(self.feature_names)
    if not (
        len(self.weights) == len(self.mean) == len(self.scale) == expected
    ):
      raise ValueError(
          "Interface scorer dimensions must match feature_names: "
          f"weights={len(self.weights)}, mean={len(self.mean)}, "
          f"scale={len(self.scale)}, feature_names={expected}"
      )
    if np.any(~np.isfinite(self.weights)) or np.any(~np.isfinite(self.mean)):
      raise ValueError("Interface scorer weights and mean must be finite")
    if np.any(~np.isfinite(self.scale)) or np.any(self.scale == 0.0):
      raise ValueError("Interface scorer scale must be finite and non-zero")
    if not math.isfinite(self.threshold):
      raise ValueError("Interface scorer threshold must be finite")
    if self.model_protocol == "benchmark_v2" and self.feature_names != tuple(
        BENCHMARK_V2_FEATURE_NAMES
    ):
      raise ValueError(
          "benchmark_v2 interface scorer must use the fixed "
          "BENCHMARK_V2_FEATURE_NAMES schema"
      )

  @staticmethod
  def load(
      path: str | Path,
      *,
      expected_protocol: str | None = None,
  ) -> "InterfaceScorer":
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model_protocol = str(
        data.get("model_input_protocol", data.get("protocol", "legacy"))
    )
    model_protocol = _normalized_model_protocol(model_protocol)
    if expected_protocol is not None:
      expected = _normalized_model_protocol(expected_protocol)
      if model_protocol != expected:
        raise ValueError(
            f"Expected {expected} interface model, found {model_protocol}"
        )
    if model_protocol == "benchmark_v2":
      _validate_bound_model_metadata(
          data,
          expected_schema=INTERFACE_SCORER_V2_SCHEMA,
          expected_protocol=model_protocol,
          expected_feature_names=BENCHMARK_V2_FEATURE_NAMES,
      )
    return InterfaceScorer(
        weights=np.asarray(data["weights"], dtype=float),
        mean=np.asarray(data["mean"], dtype=float),
        scale=np.asarray(data["scale"], dtype=float),
        threshold=float(data.get("threshold", 0.5)),
        feature_names=data.get("feature_names"),
        model_protocol=model_protocol,
    )

  def score_vector(self, features: list[float]) -> float:
    x = np.asarray(features, dtype=float).reshape(-1)
    if len(x) != len(self.feature_names):
      raise ValueError(
          f"Expected {len(self.feature_names)} scorer features, got {len(x)}"
      )
    if np.any(~np.isfinite(x)):
      raise ValueError("Interface scorer features must be finite")
    x = (x - self.mean) / self.scale
    logit = float(np.dot(x, self.weights))
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))

  def score_candidates(
      self,
      candidates: list[CandidateInterface],
      *,
      protocol: str = "legacy",
      invariant_part_scale: float | None = None,
      part_surface_area: float | None = None,
  ) -> list[CandidateInterface]:
    protocol = _normalized_model_protocol(protocol)
    if protocol != self.model_protocol:
      raise ValueError(
          f"Cannot use {self.model_protocol} model for {protocol} model input"
      )
    if protocol == "legacy":
      for candidate in candidates:
        candidate.score = self.score_vector(interface_feature_vector(candidate))
      return candidates

    if part_surface_area is None:
      part_surface_area = self._benchmark_part_surface_area(candidates)
    if invariant_part_scale is None:
      invariant_part_scale = math.sqrt(float(part_surface_area))
    for candidate in candidates:
      model_input = self.model_input_for_candidate(
          candidate,
          protocol=protocol,
          invariant_part_scale=invariant_part_scale,
          part_surface_area=part_surface_area,
      )
      candidate.score = self.score_vector(list(model_input.vector))
    return candidates

  def model_input_for_candidate(
      self,
      candidate: CandidateInterface,
      *,
      protocol: str = "legacy",
      invariant_part_scale: float | None = None,
      part_surface_area: float | None = None,
  ) -> InterfaceModelInput:
    """Return the exact auditable vector consumed for one candidate."""

    protocol = _normalized_model_protocol(protocol)
    if protocol != self.model_protocol:
      raise ValueError(
          f"Cannot use {self.model_protocol} model for {protocol} model input"
      )
    if protocol == "legacy":
      vector = tuple(interface_feature_vector(candidate))
      return InterfaceModelInput(
          protocol="legacy",
          feature_names=self.feature_names,
          vector=vector,
      )
    if invariant_part_scale is None or part_surface_area is None:
      raise ValueError(
          "benchmark_v2 model input requires invariant_part_scale and "
          "part_surface_area"
      )
    sanitization = sanitize_benchmark_v2_model_view(
        candidate.to_dict(),
        invariant_part_scale=float(invariant_part_scale),
        part_surface_area=float(part_surface_area),
        formal=True,
    )
    vector = tuple(benchmark_v2_numeric_feature_vector(sanitization))
    return InterfaceModelInput(
        protocol="benchmark_v2",
        feature_names=self.feature_names,
        vector=vector,
        sanitization=sanitization,
    )

  def score_model_views(
      self,
      model_views: Sequence[Mapping[str, Any]],
  ) -> list[float]:
    """Score pre-sanitized views supplied by the shared benchmark boundary."""

    if self.model_protocol != "benchmark_v2":
      raise ValueError(
          "score_model_views is available only for a benchmark_v2 model"
      )
    scores: list[float] = []
    for index, model_view in enumerate(model_views):
      if not isinstance(model_view, Mapping):
        raise TypeError(f"model_views[{index}] must be a mapping")
      sha256 = validate_benchmark_v2_model_view(model_view)
      canonical_json = json.dumps(
          model_view,
          ensure_ascii=False,
          sort_keys=True,
          separators=(",", ":"),
          allow_nan=False,
      )
      # Canonical round-trip prevents a caller from mutating a custom Mapping
      # after validation but before vectorization.
      canonical_view = json.loads(canonical_json)
      sanitization = ModelViewSanitization(
          model_view=canonical_view,
          audit={
              "schema_version": 1,
              "protocol": "benchmark_v2",
              "formal": True,
              "source": "validated_pre_sanitized_model_view",
          },
          canonical_json=canonical_json,
          sha256=sha256,
      )
      scores.append(
          self.score_vector(
              benchmark_v2_numeric_feature_vector(sanitization)
          )
      )
    return scores

  @staticmethod
  def _benchmark_part_surface_area(
      candidates: list[CandidateInterface],
  ) -> float:
    area = sum(
        max(0.0, float(candidate.metadata.get("area", 0.0) or 0.0))
        for candidate in candidates
        if not bool(candidate.metadata.get("composite_interface", False))
    )
    if not math.isfinite(area) or area <= 0.0:
      raise ValueError(
          "benchmark_v2 requires positive invariant base-face surface area"
      )
    return float(area)


def train_model(
    *,
    train_jsonl: Path,
    dev_jsonl: Path | None,
    epochs: int,
    learning_rate: float,
    l2: float,
    positive_weight: float,
    input_protocol: str = "legacy",
) -> dict[str, Any]:
  protocol = _normalized_model_protocol(input_protocol)
  if protocol == "benchmark_v2":
    validate_training_hyperparameters(
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        positive_weight=positive_weight,
    )
  feature_names = (
      BENCHMARK_V2_FEATURE_NAMES if protocol == "benchmark_v2" else FEATURE_NAMES
  )
  train_binding: TrainingDatasetBinding | None = None
  dev_binding: TrainingDatasetBinding | None = None
  if protocol == "benchmark_v2":
    train_binding = validate_dataset_manifest(
        train_jsonl,
        expected_row_schema=INTERFACE_TRAINING_ROW_SCHEMA,
        expected_feature_names=BENCHMARK_V2_FEATURE_NAMES,
        required_split="train",
        producer_kind="interface",
    )
    if dev_jsonl is not None:
      dev_binding = validate_dataset_manifest(
          dev_jsonl,
          expected_row_schema=INTERFACE_TRAINING_ROW_SCHEMA,
          expected_feature_names=BENCHMARK_V2_FEATURE_NAMES,
          required_split="dev",
          producer_kind="interface",
      )
      if (
          dev_binding.family_split_manifest_sha256
          != train_binding.family_split_manifest_sha256
      ):
        raise ValueError("benchmark_v2 train/dev use different frozen family splits")
      overlap = set(train_binding.case_token_sha256) & set(
          dev_binding.case_token_sha256
      )
      if overlap:
        raise ValueError("benchmark_v2 train/dev case overlap detected")
  train_rows = _load_rows(train_jsonl, input_protocol=protocol)
  if not train_rows:
    raise ValueError(f"No labeled rows loaded from {train_jsonl}")
  dev_rows = (
      _load_rows(dev_jsonl, input_protocol=protocol)
      if dev_jsonl is not None
      else []
  )
  if train_binding is not None:
    validate_rows_against_binding(
        train_rows,
        train_binding,
        allowed_keys=set(INTERFACE_TRAINING_ROW_KEYS),
    )
  if dev_binding is not None:
    validate_rows_against_binding(
        dev_rows,
        dev_binding,
        allowed_keys=set(INTERFACE_TRAINING_ROW_KEYS),
    )

  X_train, y_train = _rows_to_xy(train_rows, input_protocol=protocol)
  if protocol == "benchmark_v2" and set(y_train.tolist()) != {0.0, 1.0}:
    raise ValueError("benchmark_v2 interface training requires both binary classes")
  X_dev, y_dev = (
      _rows_to_xy(dev_rows, input_protocol=protocol)
      if dev_rows
      else (None, None)
  )
  mean = X_train.mean(axis=0)
  scale = X_train.std(axis=0)
  scale = np.where(scale < 1e-8, 1.0, scale)
  # Keep the explicit bias feature stable.
  mean[0] = 0.0
  scale[0] = 1.0
  Xs = (X_train - mean) / scale

  if positive_weight <= 0.0:
    pos = max(1.0, float(np.sum(y_train == 1.0)))
    neg = max(1.0, float(np.sum(y_train == 0.0)))
    positive_weight = neg / pos
  sample_weights = np.where(y_train > 0.5, float(positive_weight), 1.0)
  sample_weights = sample_weights / max(1e-9, float(sample_weights.mean()))

  weights = np.zeros(Xs.shape[1], dtype=float)
  lr = float(learning_rate)
  for epoch in range(max(1, int(epochs))):
    logits = Xs @ weights
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    error = (probs - y_train) * sample_weights
    grad = (Xs.T @ error) / float(len(y_train))
    grad += float(l2) * weights
    grad[0] -= float(l2) * weights[0]
    weights -= lr * grad
    if epoch > 0 and epoch % 200 == 0:
      lr *= 0.75

  train_metrics = _metrics(y_train, _predict_scores(X_train, weights, mean, scale))
  metrics: dict[str, Any] = {
      "train": train_metrics,
      "train_rows": int(len(train_rows)),
      "positive_weight": round(float(positive_weight), 4),
  }
  if X_dev is not None and y_dev is not None and len(y_dev) > 0:
    metrics["dev"] = _metrics(y_dev, _predict_scores(X_dev, weights, mean, scale))
    metrics["dev_rows"] = int(len(dev_rows))

  feature_names_sha256 = _canonical_sha256(list(feature_names))
  model_schema = (
      INTERFACE_SCORER_V2_SCHEMA
      if protocol == "benchmark_v2"
      else "interface_scorer.legacy.v1"
  )
  payload: dict[str, Any] = {
      "model_type": "logistic_interface_affordance_scorer",
      "model_schema": model_schema,
      "model_input_protocol": protocol,
      "feature_names": list(feature_names),
      "feature_names_sha256": feature_names_sha256,
      "weights": weights.tolist(),
      "mean": mean.tolist(),
      "scale": scale.tolist(),
      "threshold": 0.5,
      "metrics": metrics,
  }
  payload["metrics_sha256"] = _canonical_sha256(metrics)
  parameters_sha256 = model_parameter_sha256(
      payload,
      keys=INTERFACE_MODEL_PARAMETER_KEYS,
  )
  payload["model_parameters_sha256"] = parameters_sha256
  payload["training_manifest"] = _training_manifest(
      model_schema=model_schema,
      input_protocol=protocol,
      feature_names_sha256=feature_names_sha256,
      train_jsonl=train_jsonl,
      dev_jsonl=dev_jsonl,
      train_rows=train_rows,
      dev_rows=dev_rows,
      epochs=epochs,
      learning_rate=learning_rate,
      l2=l2,
      positive_weight=positive_weight,
      train_binding=train_binding,
      dev_binding=dev_binding,
      model_parameters_sha256=parameters_sha256,
  )
  return payload


def _load_rows(
    path: Path | None,
    *,
    input_protocol: str = "legacy",
) -> list[dict[str, Any]]:
  protocol = _normalized_model_protocol(input_protocol)
  if path is None or not path.exists():
    return []
  rows = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      item = json.loads(line)
      if not isinstance(item, dict):
        if protocol == "benchmark_v2":
          raise ValueError("benchmark_v2 training rows must be JSON objects")
        continue
      if item.get("label") not in {0, 1}:
        if protocol == "benchmark_v2":
          raise ValueError("benchmark_v2 training rows require binary labels")
        continue
      if protocol == "benchmark_v2":
        _benchmark_v2_model_input_from_row(item)
      rows.append(item)
  return rows


def _rows_to_xy(
    rows: list[dict[str, Any]],
    *,
    input_protocol: str = "legacy",
) -> tuple[np.ndarray, np.ndarray]:
  protocol = _normalized_model_protocol(input_protocol)
  X = []
  y = []
  for row in rows:
    if protocol == "benchmark_v2":
      model_input = _benchmark_v2_model_input_from_row(row)
      X.append(benchmark_v2_numeric_feature_vector(model_input))
      y.append(float(row.get("label", 0)))
      continue
    features = row.get("features")
    if not isinstance(features, dict):
      continue
    X.append([float(features.get(name, 0.0)) for name in FEATURE_NAMES])
    y.append(float(row.get("label", 0)))
  if not X:
    raise ValueError("No usable feature rows.")
  return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _benchmark_v2_model_input_from_row(
    row: Mapping[str, Any],
) -> ModelViewSanitization:
  if str(row.get("model_input_protocol") or "") != "benchmark_v2":
    raise ValueError("benchmark_v2 training row lacks model_input_protocol")
  view = row.get("benchmark_v2_model_view")
  expected_hash = row.get("benchmark_v2_model_view_sha256")
  if not isinstance(view, Mapping) or not isinstance(expected_hash, str):
    raise ValueError("benchmark_v2 training row lacks sanitized model view/hash")
  actual_hash = validate_benchmark_v2_model_view(
      view,
      expected_sha256=expected_hash,
  )
  canonical_json = json.dumps(
      view,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  )
  return ModelViewSanitization(
      model_view=json.loads(canonical_json),
      audit={"protocol": "benchmark_v2", "source": "training_jsonl"},
      canonical_json=canonical_json,
      sha256=actual_hash,
  )


def _canonical_sha256(payload: Any) -> str:
  canonical = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  )
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path | None) -> str | None:
  if path is None:
    return None
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _training_manifest(
    *,
    model_schema: str,
    input_protocol: str,
    feature_names_sha256: str,
    train_jsonl: Path,
    dev_jsonl: Path | None,
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    epochs: int,
    learning_rate: float,
    l2: float,
    positive_weight: float,
    train_binding: TrainingDatasetBinding | None,
    dev_binding: TrainingDatasetBinding | None,
    model_parameters_sha256: str,
) -> dict[str, Any]:
  optimizer = {
      "epochs": int(epochs),
      "learning_rate": float(learning_rate),
      "l2": float(l2),
      "positive_weight": float(positive_weight),
  }
  payload: dict[str, Any] = {
      "schema_version": MODEL_TRAINING_MANIFEST_SCHEMA,
      "model_schema": model_schema,
      "model_input_protocol": input_protocol,
      "protocol_version": (
          PROTOCOL_VERSION if input_protocol == "benchmark_v2" else "legacy"
      ),
      "feature_names_sha256": feature_names_sha256,
      "train_jsonl_sha256": _file_sha256(train_jsonl),
      "dev_jsonl_sha256": _file_sha256(dev_jsonl),
      "train_rows": len(train_rows),
      "dev_rows": len(dev_rows),
      "train_case_count": 0 if train_binding is None else train_binding.case_count,
      "dev_case_count": 0 if dev_binding is None else dev_binding.case_count,
      "train_positive_rows": sum(int(row.get("label") == 1) for row in train_rows),
      "dev_positive_rows": sum(int(row.get("label") == 1) for row in dev_rows),
      "optimizer": optimizer,
      "training_config_sha256": _canonical_sha256(optimizer),
      "model_parameters_sha256": model_parameters_sha256,
  }
  if input_protocol == "benchmark_v2":
    if train_binding is None:
      raise ValueError("benchmark_v2 training lacks a validated train binding")
    payload.update(
        {
            "frozen_family_train_split_sha256": (
                train_binding.source_split_artifact_sha256
            ),
            "frozen_family_dev_split_sha256": (
                None
                if dev_binding is None
                else dev_binding.source_split_artifact_sha256
            ),
            "family_split_manifest_sha256": (
                train_binding.family_split_manifest_sha256
            ),
            "train_dataset_manifest_sha256": train_binding.manifest_sha256,
            "dev_dataset_manifest_sha256": (
                None if dev_binding is None else dev_binding.manifest_sha256
            ),
            "train_source_split": train_binding.source_split,
            "dev_source_split": (
                None if dev_binding is None else dev_binding.source_split
            ),
            "trainer_code_sha256": trainer_code_sha256("interface"),
        }
    )
  payload["manifest_sha256"] = _canonical_sha256(payload)
  return payload


def _validate_bound_model_metadata(
    data: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_protocol: str,
    expected_feature_names: Sequence[str],
) -> None:
  if set(data) != set(INTERFACE_MODEL_ALLOWED_KEYS):
    raise ValueError("benchmark_v2 model top-level schema mismatch")
  if data.get("model_type") != "logistic_interface_affordance_scorer":
    raise ValueError("benchmark_v2 model type mismatch")
  if data.get("model_schema") != expected_schema:
    raise ValueError("benchmark_v2 model schema mismatch")
  if data.get("model_input_protocol") != expected_protocol:
    raise ValueError("benchmark_v2 model input protocol mismatch")
  expected_feature_hash = _canonical_sha256(list(expected_feature_names))
  if data.get("feature_names_sha256") != expected_feature_hash:
    raise ValueError("benchmark_v2 model feature names hash mismatch")
  if tuple(data.get("feature_names") or ()) != tuple(expected_feature_names):
    raise ValueError("benchmark_v2 model feature schema mismatch")
  for key in ("weights", "mean", "scale"):
    validate_finite_numeric_tree(data.get(key), name=f"interface model {key}")
  validate_finite_numeric_tree(
      data.get("threshold"),
      name="interface model threshold",
  )
  expected_parameter_hash = model_parameter_sha256(
      data,
      keys=INTERFACE_MODEL_PARAMETER_KEYS,
  )
  if data.get("model_parameters_sha256") != expected_parameter_hash:
    raise ValueError("benchmark_v2 model parameter hash mismatch")
  metrics = data.get("metrics")
  if not isinstance(metrics, Mapping) or data.get("metrics_sha256") != _canonical_sha256(metrics):
    raise ValueError("benchmark_v2 model metrics hash mismatch")
  manifest = data.get("training_manifest")
  if not isinstance(manifest, Mapping):
    raise ValueError("benchmark_v2 model lacks training manifest")
  if set(manifest) != set(FORMAL_MODEL_TRAINING_MANIFEST_KEYS):
    raise ValueError("benchmark_v2 model training manifest schema mismatch")
  validate_formal_training_manifest_counts(manifest)
  unsigned = dict(manifest)
  stored_hash = unsigned.pop("manifest_sha256", None)
  if stored_hash != _canonical_sha256(unsigned):
    raise ValueError("benchmark_v2 model training manifest hash mismatch")
  required = {
      "schema_version": MODEL_TRAINING_MANIFEST_SCHEMA,
      "model_schema": expected_schema,
      "model_input_protocol": expected_protocol,
      "feature_names_sha256": expected_feature_hash,
      "protocol_version": PROTOCOL_VERSION,
      "train_source_split": "train",
      "model_parameters_sha256": expected_parameter_hash,
      "trainer_code_sha256": trainer_code_sha256("interface"),
  }
  if any(manifest.get(key) != value for key, value in required.items()):
    raise ValueError("benchmark_v2 model training manifest binding mismatch")
  for key in (
      "frozen_family_train_split_sha256",
      "family_split_manifest_sha256",
      "train_dataset_manifest_sha256",
      "training_config_sha256",
  ):
    value = str(manifest.get(key) or "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
      raise ValueError(f"benchmark_v2 model {key} is not a SHA-256 binding")
  optimizer = manifest.get("optimizer")
  if not isinstance(optimizer, Mapping):
    raise ValueError("benchmark_v2 model lacks optimizer configuration")
  if manifest.get("training_config_sha256") != _canonical_sha256(optimizer):
    raise ValueError("benchmark_v2 model training config hash mismatch")


def _predict_scores(
    X: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
  Xs = (X - mean) / scale
  logits = Xs @ weights
  return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
  preds = scores >= 0.5
  truth = labels > 0.5
  tp = int(np.sum(preds & truth))
  fp = int(np.sum(preds & ~truth))
  fn = int(np.sum(~preds & truth))
  tn = int(np.sum(~preds & ~truth))
  precision = tp / max(1, tp + fp)
  recall = tp / max(1, tp + fn)
  f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
  return {
      "positive_rate": round(float(np.mean(truth)), 4),
      "accuracy": round(float((tp + tn) / max(1, len(labels))), 4),
      "precision": round(float(precision), 4),
      "recall": round(float(recall), 4),
      "f1": round(float(f1), 4),
      "tp": tp,
      "fp": fp,
      "fn": fn,
      "tn": tn,
  }


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    scorer: InterfaceScorer,
    ks: list[int],
) -> dict[str, Any]:
  if not rows:
    return {"rows": 0}
  X, y = _rows_to_xy(rows, input_protocol=scorer.model_protocol)
  scores = _predict_scores(X, scorer.weights, scorer.mean, scorer.scale)
  base = _metrics(y, scores)

  grouped: dict[tuple[str, str, str], list[tuple[float, int]]] = {}
  for row, score in zip(rows, scores):
    key = (
        str(row.get("case_id") or ""),
        str(row.get("body_uuid") or ""),
        str(row.get("part_name") or ""),
    )
    grouped.setdefault(key, []).append((float(score), int(row.get("label", 0))))

  at_k: dict[str, Any] = {}
  for k in sorted(set(max(1, int(item)) for item in ks)):
    recalls = []
    precisions = []
    hit_rates = []
    for items in grouped.values():
      positives = sum(1 for _, label in items if label == 1)
      if positives <= 0:
        continue
      ranked = sorted(items, key=lambda item: -item[0])[:k]
      hits = sum(1 for _, label in ranked if label == 1)
      recalls.append(hits / max(1, positives))
      precisions.append(hits / max(1, len(ranked)))
      hit_rates.append(1.0 if hits > 0 else 0.0)
    at_k[f"recall@{k}"] = round(float(np.mean(recalls)) if recalls else 0.0, 4)
    at_k[f"precision@{k}"] = round(float(np.mean(precisions)) if precisions else 0.0, 4)
    at_k[f"hit_rate@{k}"] = round(float(np.mean(hit_rates)) if hit_rates else 0.0, 4)

  return {
      "rows": int(len(rows)),
      "groups": int(len(grouped)),
      "threshold_metrics": base,
      **at_k,
  }


if __name__ == "__main__":
  main()
