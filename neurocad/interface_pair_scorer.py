"""Lightweight learned scorer for interface-pair compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark_v2_constants import PROTOCOL_VERSION
from .benchmark_v2_training_provenance import (
    INTERFACE_PAIR_TRAINING_ROW_KEYS,
    INTERFACE_PAIR_TRAINING_ROW_SCHEMA,
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
from .interface_pair_features import (
    BENCHMARK_V2_PAIR_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    pair_feature_vector,
    pair_features_from_rows,
    pair_features_from_sockets,
)
from .domain_types import Socket
from .paths import resolve_path


INTERFACE_PAIR_SCORER_V2_SCHEMA = "interface_pair_scorer.v2"
INTERFACE_PAIR_MODEL_PARAMETER_KEYS = (
    "model_type",
    "model_schema",
    "model_input_protocol",
    "feature_names",
    "feature_names_sha256",
    "weights",
    "type_weights",
    "contact_types",
    "mean",
    "scale",
    "threshold",
    "metrics_sha256",
)
INTERFACE_PAIR_MODEL_ALLOWED_KEYS = frozenset(
    {
        *INTERFACE_PAIR_MODEL_PARAMETER_KEYS,
        "model_parameters_sha256",
        "training_manifest",
        "metrics",
    }
)


CONTACT_TYPES = (
    "none",
    "seat_plane",
    "insert_axis",
    "shaft_in_bore",
    "screw_in_hole",
    "pin_in_slot",
    "boss_in_slot",
    "threaded_interference",
    "generic_contact",
)


def _normalized_pair_model_protocol(protocol: str) -> str:
  value = str(protocol or "legacy").strip().lower()
  if value not in {"legacy", "benchmark_v2"}:
    raise ValueError("pair model protocol must be 'legacy' or 'benchmark_v2'")
  return value


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="command", required=True)

  train = sub.add_parser("train", help="Train a binary interface-pair scorer.")
  train.add_argument("--train_jsonl", default="neurocad/interface_pair_train.jsonl")
  train.add_argument("--dev_jsonl", default=None)
  train.add_argument("--output_model", default="neurocad/interface_pair_scorer_model.json")
  train.add_argument("--epochs", type=int, default=700)
  train.add_argument("--learning_rate", type=float, default=0.06)
  train.add_argument("--l2", type=float, default=2e-4)
  train.add_argument("--positive_weight", type=float, default=0.0)
  train.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )

  evaluate = sub.add_parser("eval", help="Evaluate pair scorer on labeled JSONL.")
  evaluate.add_argument("--model", default="neurocad/interface_pair_scorer_model.json")
  evaluate.add_argument("--jsonl", required=True)
  evaluate.add_argument("--ks", default="1,3,5,10")
  evaluate.add_argument("--output_json", default=None)
  evaluate.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )

  return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  """Resolve pair-scorer inputs and outputs at the project root."""
  names = (
      ("train_jsonl", "dev_jsonl", "output_model")
      if args.command == "train"
      else ("model", "jsonl", "output_json")
  )
  for name in names:
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

  if args.command == "eval":
    scorer = InterfacePairScorer.load(
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
      output = Path(args.output_json)
      output.parent.mkdir(parents=True, exist_ok=True)
      output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return


class InterfacePairScorer:
  """Standardized logistic scorer for two local interface sockets."""

  def __init__(
      self,
      weights: np.ndarray,
      mean: np.ndarray,
      scale: np.ndarray,
      type_weights: np.ndarray | None = None,
      threshold: float = 0.5,
      contact_types: tuple[str, ...] = CONTACT_TYPES,
      feature_names: Sequence[str] | None = None,
      model_protocol: str = "legacy",
  ) -> None:
    self.model_protocol = _normalized_pair_model_protocol(model_protocol)
    default_names: Sequence[str] = (
        BENCHMARK_V2_PAIR_FEATURE_NAMES
        if self.model_protocol == "benchmark_v2"
        else PAIR_FEATURE_NAMES
    )
    self.feature_names = tuple(str(name) for name in (feature_names or default_names))
    self.weights = np.asarray(weights, dtype=float).reshape(-1)
    self.mean = np.asarray(mean, dtype=float).reshape(-1)
    self.scale = np.asarray(scale, dtype=float).reshape(-1)
    self.type_weights = (
        None
        if type_weights is None
        else np.asarray(type_weights, dtype=float).reshape((-1, len(contact_types)))
    )
    self.threshold = float(threshold)
    self.contact_types = tuple(contact_types)
    expected = len(self.feature_names)
    if not (
        len(self.weights) == len(self.mean) == len(self.scale) == expected
    ):
      raise ValueError("Interface pair scorer dimensions do not match feature_names")
    if np.any(~np.isfinite(self.weights)) or np.any(~np.isfinite(self.mean)):
      raise ValueError("Interface pair scorer weights and mean must be finite")
    if np.any(~np.isfinite(self.scale)) or np.any(self.scale == 0.0):
      raise ValueError("Interface pair scorer scale must be finite and non-zero")
    if self.type_weights is not None and self.type_weights.shape[0] != expected:
      raise ValueError("Interface pair type_weights do not match feature_names")
    if self.type_weights is not None and np.any(~np.isfinite(self.type_weights)):
      raise ValueError("Interface pair type_weights must be finite")
    if not math.isfinite(self.threshold):
      raise ValueError("Interface pair scorer threshold must be finite")
    if self.model_protocol == "benchmark_v2" and self.feature_names != tuple(
        BENCHMARK_V2_PAIR_FEATURE_NAMES
    ):
      raise ValueError(
          "benchmark_v2 pair scorer must use fixed benchmark pair features"
      )

  @staticmethod
  def load(
      path: str | Path,
      *,
      expected_protocol: str | None = None,
  ) -> "InterfacePairScorer":
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_feature_names = list(data.get("feature_names", PAIR_FEATURE_NAMES))
    stored_contact_types = tuple(data.get("contact_types", CONTACT_TYPES))
    model_protocol = _normalized_pair_model_protocol(
        str(data.get("model_input_protocol", data.get("protocol", "legacy")))
    )
    if expected_protocol is not None:
      expected = _normalized_pair_model_protocol(expected_protocol)
      if model_protocol != expected:
        raise ValueError(
            f"Expected {expected} interface-pair model, found {model_protocol}"
        )
    if model_protocol == "benchmark_v2":
      _validate_bound_model_metadata(
          data,
          expected_schema=INTERFACE_PAIR_SCORER_V2_SCHEMA,
          expected_feature_names=BENCHMARK_V2_PAIR_FEATURE_NAMES,
      )
      if tuple(stored_feature_names) != tuple(BENCHMARK_V2_PAIR_FEATURE_NAMES):
        raise ValueError("benchmark_v2 pair model feature schema mismatch")
      if stored_contact_types != tuple(CONTACT_TYPES):
        raise ValueError("benchmark_v2 pair model contact type schema mismatch")
      return InterfacePairScorer(
          weights=np.asarray(data["weights"], dtype=float),
          mean=np.asarray(data["mean"], dtype=float),
          scale=np.asarray(data["scale"], dtype=float),
          type_weights=(
              None
              if data.get("type_weights") is None
              else np.asarray(data["type_weights"], dtype=float)
          ),
          threshold=float(data.get("threshold", 0.5)),
          contact_types=tuple(CONTACT_TYPES),
          feature_names=BENCHMARK_V2_PAIR_FEATURE_NAMES,
          model_protocol="benchmark_v2",
      )
    contact_types = tuple(CONTACT_TYPES)
    weights, mean, scale = _align_vector_payload(
        feature_names=stored_feature_names,
        values=data["weights"],
        default=0.0,
    )
    mean = _align_named_values(
        feature_names=stored_feature_names,
        values=data["mean"],
        default=0.0,
    )
    scale = _align_named_values(
        feature_names=stored_feature_names,
        values=data["scale"],
        default=1.0,
    )
    type_weights = _align_type_weights_payload(
        feature_names=stored_feature_names,
        contact_types=stored_contact_types,
        values=data.get("type_weights"),
        target_contact_types=contact_types,
    )
    return InterfacePairScorer(
        weights=weights,
        mean=mean,
        scale=scale,
        type_weights=type_weights,
        threshold=float(data.get("threshold", 0.5)),
        contact_types=contact_types,
        feature_names=PAIR_FEATURE_NAMES,
        model_protocol="legacy",
    )

  def score_features(
      self,
      features: dict[str, float],
      *,
      protocol: str = "legacy",
  ) -> float:
    protocol = _normalized_pair_model_protocol(protocol)
    if protocol != self.model_protocol:
      raise ValueError(
          f"Cannot use {self.model_protocol} pair model for {protocol} input"
      )
    x = np.asarray(
        pair_feature_vector(features, protocol=protocol), dtype=float
    ).reshape(-1)
    if np.any(~np.isfinite(x)):
      raise ValueError("Interface pair scorer features must be finite")
    x = (x - self.mean) / self.scale
    logit = float(np.dot(x, self.weights))
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))

  def score_sockets(
      self,
      socket_a: Socket,
      socket_b: Socket,
      *,
      relation_hint: str = "link",
      protocol: str = "legacy",
  ) -> float:
    features = pair_features_from_sockets(
        socket_a,
        socket_b,
        relation_hint=relation_hint,
        protocol=protocol,
    )
    return self.score_features(features, protocol=protocol)

  def predict_contact_type_features(
      self,
      features: dict[str, float],
      *,
      protocol: str = "legacy",
  ) -> dict[str, float]:
    protocol = _normalized_pair_model_protocol(protocol)
    if protocol != self.model_protocol:
      raise ValueError(
          f"Cannot use {self.model_protocol} pair model for {protocol} input"
      )
    if self.type_weights is None:
      mate = self.score_features(features, protocol=protocol)
      fallback = _heuristic_contact_type(features)
      probs = {name: 0.0 for name in self.contact_types}
      probs["none"] = max(0.0, 1.0 - mate)
      probs[fallback] = mate
      return probs
    x = np.asarray(
        pair_feature_vector(features, protocol=protocol), dtype=float
    ).reshape(-1)
    if np.any(~np.isfinite(x)):
      raise ValueError("Interface pair scorer features must be finite")
    x = (x - self.mean) / self.scale
    logits = x @ self.type_weights
    logits = logits - np.max(logits)
    exp = np.exp(np.clip(logits, -60.0, 60.0))
    probs = exp / max(1e-12, float(np.sum(exp)))
    return {
        name: float(prob)
        for name, prob in zip(self.contact_types, probs)
    }

  def predict_contact_type_sockets(
      self,
      socket_a: Socket,
      socket_b: Socket,
      *,
      relation_hint: str = "link",
      protocol: str = "legacy",
  ) -> tuple[str, dict[str, float]]:
    features = pair_features_from_sockets(
        socket_a,
        socket_b,
        relation_hint=relation_hint,
        protocol=protocol,
    )
    probs = self.predict_contact_type_features(features, protocol=protocol)
    label = max(probs.items(), key=lambda item: item[1])[0] if probs else "none"
    return label, probs


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
  protocol = _normalized_pair_model_protocol(input_protocol)
  if protocol == "benchmark_v2":
    validate_training_hyperparameters(
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        positive_weight=positive_weight,
    )
  feature_names = (
      BENCHMARK_V2_PAIR_FEATURE_NAMES
      if protocol == "benchmark_v2"
      else PAIR_FEATURE_NAMES
  )
  train_binding: TrainingDatasetBinding | None = None
  dev_binding: TrainingDatasetBinding | None = None
  if protocol == "benchmark_v2":
    train_binding = validate_dataset_manifest(
        train_jsonl,
        expected_row_schema=INTERFACE_PAIR_TRAINING_ROW_SCHEMA,
        expected_feature_names=BENCHMARK_V2_PAIR_FEATURE_NAMES,
        required_split="train",
        producer_kind="pair",
    )
    if dev_jsonl is not None:
      dev_binding = validate_dataset_manifest(
          dev_jsonl,
          expected_row_schema=INTERFACE_PAIR_TRAINING_ROW_SCHEMA,
          expected_feature_names=BENCHMARK_V2_PAIR_FEATURE_NAMES,
          required_split="dev",
          producer_kind="pair",
      )
      if (
          dev_binding.family_split_manifest_sha256
          != train_binding.family_split_manifest_sha256
      ):
        raise ValueError("benchmark_v2 pair train/dev use different frozen family splits")
      if set(train_binding.case_token_sha256) & set(dev_binding.case_token_sha256):
        raise ValueError("benchmark_v2 pair train/dev case overlap detected")
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
        allowed_keys=set(INTERFACE_PAIR_TRAINING_ROW_KEYS),
    )
  if dev_binding is not None:
    validate_rows_against_binding(
        dev_rows,
        dev_binding,
        allowed_keys=set(INTERFACE_PAIR_TRAINING_ROW_KEYS),
    )

  X_train, y_train = _rows_to_xy(train_rows, input_protocol=protocol)
  if protocol == "benchmark_v2" and set(y_train.tolist()) != {0.0, 1.0}:
    raise ValueError("benchmark_v2 pair training requires both binary classes")
  y_type_train = _rows_to_type_y(train_rows)
  if protocol == "benchmark_v2" and len(set(y_type_train.tolist())) < 2:
    raise ValueError("benchmark_v2 pair training lacks contact-type class diversity")
  X_dev, y_dev = (
      _rows_to_xy(dev_rows, input_protocol=protocol)
      if dev_rows
      else (None, None)
  )
  y_type_dev = _rows_to_type_y(dev_rows) if dev_rows else None
  mean = X_train.mean(axis=0)
  scale = X_train.std(axis=0)
  scale = np.where(scale < 1e-8, 1.0, scale)
  mean[0] = 0.0
  scale[0] = 1.0
  Xs = (X_train - mean) / scale

  if positive_weight <= 0.0:
    pos = max(1.0, float(np.sum(y_train == 1.0)))
    neg = max(1.0, float(np.sum(y_train == 0.0)))
    positive_weight = min(20.0, neg / pos)
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
    if epoch > 0 and epoch % 250 == 0:
      lr *= 0.72

  type_weights = _train_type_weights(
      Xs=Xs,
      y_type=y_type_train,
      sample_weights=sample_weights,
      epochs=max(1, int(epochs)),
      learning_rate=float(learning_rate),
      l2=float(l2),
  )

  metrics: dict[str, Any] = {
      "train": _metrics(y_train, _predict_scores(X_train, weights, mean, scale)),
      "train_rank": _ranking_metrics(train_rows, _predict_scores(X_train, weights, mean, scale), [1, 3, 5, 10]),
      "train_contact_type": _contact_type_metrics(
          y_type_train,
          _predict_type_probs(X_train, type_weights, mean, scale),
      ),
      "train_rows": int(len(train_rows)),
      "positive_weight": round(float(positive_weight), 4),
  }
  if X_dev is not None and y_dev is not None and len(y_dev) > 0:
    dev_scores = _predict_scores(X_dev, weights, mean, scale)
    metrics["dev"] = _metrics(y_dev, dev_scores)
    metrics["dev_rank"] = _ranking_metrics(dev_rows, dev_scores, [1, 3, 5, 10])
    if y_type_dev is not None:
      metrics["dev_contact_type"] = _contact_type_metrics(
          y_type_dev,
          _predict_type_probs(X_dev, type_weights, mean, scale),
      )
    metrics["dev_rows"] = int(len(dev_rows))

  feature_names_sha256 = _canonical_sha256(list(feature_names))
  model_schema = (
      INTERFACE_PAIR_SCORER_V2_SCHEMA
      if protocol == "benchmark_v2"
      else "interface_pair_scorer.legacy.v1"
  )
  payload: dict[str, Any] = {
      "model_type": "logistic_interface_pair_compatibility_scorer",
      "model_schema": model_schema,
      "model_input_protocol": protocol,
      "feature_names": list(feature_names),
      "feature_names_sha256": feature_names_sha256,
      "weights": weights.tolist(),
      "type_weights": type_weights.tolist(),
      "contact_types": list(CONTACT_TYPES),
      "mean": mean.tolist(),
      "scale": scale.tolist(),
      "threshold": 0.5,
      "metrics": metrics,
  }
  payload["metrics_sha256"] = _canonical_sha256(metrics)
  parameters_sha256 = model_parameter_sha256(
      payload,
      keys=INTERFACE_PAIR_MODEL_PARAMETER_KEYS,
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


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    scorer: InterfacePairScorer,
    ks: list[int],
) -> dict[str, Any]:
  if not rows:
    return {"rows": 0}
  X, y = _rows_to_xy(rows, input_protocol=scorer.model_protocol)
  y_type = _rows_to_type_y(rows)
  scores = _predict_scores(X, scorer.weights, scorer.mean, scorer.scale)
  result = {
      "rows": int(len(rows)),
      "threshold_metrics": _metrics(y, scores),
      "rank_metrics": _ranking_metrics(rows, scores, ks),
  }
  if scorer.type_weights is not None:
    result["contact_type_metrics"] = _contact_type_metrics(
        y_type,
        _predict_type_probs(X, scorer.type_weights, scorer.mean, scorer.scale),
    )
  return result


def _load_rows(
    path: Path | None,
    *,
    input_protocol: str = "legacy",
) -> list[dict[str, Any]]:
  protocol = _normalized_pair_model_protocol(input_protocol)
  if path is None or not path.exists():
    return []
  rows = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      item = json.loads(line)
      if isinstance(item, dict) and item.get("label") in {0, 1}:
        if protocol == "benchmark_v2":
          _benchmark_v2_pair_features_from_row(item)
        rows.append(item)
      elif protocol == "benchmark_v2":
        raise ValueError("benchmark_v2 pair training rows require object/binary label")
  return rows


def _rows_to_xy(
    rows: list[dict[str, Any]],
    *,
    input_protocol: str = "legacy",
) -> tuple[np.ndarray, np.ndarray]:
  protocol = _normalized_pair_model_protocol(input_protocol)
  feature_names = (
      BENCHMARK_V2_PAIR_FEATURE_NAMES
      if protocol == "benchmark_v2"
      else PAIR_FEATURE_NAMES
  )
  X = []
  y = []
  for row in rows:
    if protocol == "benchmark_v2":
      features = _benchmark_v2_pair_features_from_row(row)
      X.append([float(features[name]) for name in feature_names])
      y.append(float(row.get("label", 0)))
      continue
    features = row.get("features")
    if not isinstance(features, dict):
      continue
    X.append([float(features.get(name, 0.0)) for name in PAIR_FEATURE_NAMES])
    y.append(float(row.get("label", 0)))
  if not X:
    raise ValueError("No usable pair feature rows.")
  return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _benchmark_v2_pair_features_from_row(
    row: Mapping[str, Any],
) -> dict[str, float]:
  if str(row.get("model_input_protocol") or "") != "benchmark_v2":
    raise ValueError("benchmark_v2 pair row lacks model_input_protocol")
  endpoint_a = row.get("interface_a")
  endpoint_b = row.get("interface_b")
  if not isinstance(endpoint_a, dict) or not isinstance(endpoint_b, dict):
    raise ValueError("benchmark_v2 pair row requires two sanitized interfaces")
  expected_endpoint_keys = {
      "model_input_protocol",
      "benchmark_v2_model_view",
      "benchmark_v2_model_view_sha256",
      "score",
  }
  if set(endpoint_a) != expected_endpoint_keys or set(endpoint_b) != expected_endpoint_keys:
    raise ValueError("benchmark_v2 pair endpoints contain a forbidden extra channel")
  features = pair_features_from_rows(
      endpoint_a,
      endpoint_b,
      relation_hint=str(row.get("relation_hint") or "link"),
      protocol="benchmark_v2",
  )
  if tuple(features) != tuple(BENCHMARK_V2_PAIR_FEATURE_NAMES):
    raise ValueError("benchmark_v2 pair row feature schema mismatch")
  return features


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
      raise ValueError("benchmark_v2 pair training lacks a validated train binding")
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
            "trainer_code_sha256": trainer_code_sha256("pair"),
        }
    )
  payload["manifest_sha256"] = _canonical_sha256(payload)
  return payload


def _validate_bound_model_metadata(
    data: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_feature_names: Sequence[str],
) -> None:
  if set(data) != set(INTERFACE_PAIR_MODEL_ALLOWED_KEYS):
    raise ValueError("benchmark_v2 pair model top-level schema mismatch")
  if data.get("model_type") != "logistic_interface_pair_compatibility_scorer":
    raise ValueError("benchmark_v2 pair model type mismatch")
  if data.get("model_schema") != expected_schema:
    raise ValueError("benchmark_v2 pair model schema mismatch")
  expected_hash = _canonical_sha256(list(expected_feature_names))
  if data.get("feature_names_sha256") != expected_hash:
    raise ValueError("benchmark_v2 pair model feature names hash mismatch")
  if tuple(data.get("feature_names") or ()) != tuple(expected_feature_names):
    raise ValueError("benchmark_v2 pair model feature schema mismatch")
  for key in ("weights", "type_weights", "mean", "scale"):
    validate_finite_numeric_tree(data.get(key), name=f"interface pair model {key}")
  validate_finite_numeric_tree(
      data.get("threshold"),
      name="interface pair model threshold",
  )
  expected_parameter_hash = model_parameter_sha256(
      data,
      keys=INTERFACE_PAIR_MODEL_PARAMETER_KEYS,
  )
  if data.get("model_parameters_sha256") != expected_parameter_hash:
    raise ValueError("benchmark_v2 pair model parameter hash mismatch")
  metrics = data.get("metrics")
  if not isinstance(metrics, Mapping) or data.get("metrics_sha256") != _canonical_sha256(metrics):
    raise ValueError("benchmark_v2 pair model metrics hash mismatch")
  manifest = data.get("training_manifest")
  if not isinstance(manifest, Mapping):
    raise ValueError("benchmark_v2 pair model lacks training manifest")
  if set(manifest) != set(FORMAL_MODEL_TRAINING_MANIFEST_KEYS):
    raise ValueError("benchmark_v2 pair model training manifest schema mismatch")
  validate_formal_training_manifest_counts(manifest)
  unsigned = dict(manifest)
  stored_hash = unsigned.pop("manifest_sha256", None)
  if stored_hash != _canonical_sha256(unsigned):
    raise ValueError("benchmark_v2 pair model training manifest hash mismatch")
  required = {
      "schema_version": MODEL_TRAINING_MANIFEST_SCHEMA,
      "model_schema": expected_schema,
      "model_input_protocol": "benchmark_v2",
      "feature_names_sha256": expected_hash,
      "protocol_version": PROTOCOL_VERSION,
      "train_source_split": "train",
      "model_parameters_sha256": expected_parameter_hash,
      "trainer_code_sha256": trainer_code_sha256("pair"),
  }
  if any(manifest.get(key) != value for key, value in required.items()):
    raise ValueError("benchmark_v2 pair model training manifest binding mismatch")
  for key in (
      "frozen_family_train_split_sha256",
      "family_split_manifest_sha256",
      "train_dataset_manifest_sha256",
      "training_config_sha256",
  ):
    value = str(manifest.get(key) or "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
      raise ValueError(f"benchmark_v2 pair model {key} is not a SHA-256 binding")
  optimizer = manifest.get("optimizer")
  if not isinstance(optimizer, Mapping):
    raise ValueError("benchmark_v2 pair model lacks optimizer configuration")
  if manifest.get("training_config_sha256") != _canonical_sha256(optimizer):
    raise ValueError("benchmark_v2 pair model training config hash mismatch")


def _rows_to_type_y(rows: list[dict[str, Any]]) -> np.ndarray:
  index = {name: idx for idx, name in enumerate(CONTACT_TYPES)}
  labels = []
  for row in rows:
    raw = str(row.get("contact_type") or "").strip()
    if raw not in index:
      raw = "none" if int(row.get("label", 0)) == 0 else "generic_contact"
    labels.append(index[raw])
  return np.asarray(labels, dtype=int)


def _train_type_weights(
    *,
    Xs: np.ndarray,
    y_type: np.ndarray,
    sample_weights: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
  class_count = len(CONTACT_TYPES)
  weights = np.zeros((Xs.shape[1], class_count), dtype=float)
  # Rebalance non-none classes so the classifier does not collapse to "none".
  class_counts = np.bincount(y_type, minlength=class_count).astype(float)
  inv = np.where(class_counts > 0.0, np.max(class_counts) / np.maximum(1.0, class_counts), 0.0)
  inv = np.clip(inv, 0.5, 12.0)
  type_weights = sample_weights * inv[y_type]
  type_weights = type_weights / max(1e-9, float(np.mean(type_weights)))
  lr = float(learning_rate)
  y_onehot = np.zeros((len(y_type), class_count), dtype=float)
  y_onehot[np.arange(len(y_type)), y_type] = 1.0
  for epoch in range(max(1, int(epochs))):
    # Avoid a matrix-matrix BLAS dependency for this tiny linear classifier.
    # The elementwise reduction is deterministic across the locked Windows env.
    logits = np.sum(Xs[:, :, None] * weights[None, :, :], axis=1)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -60.0, 60.0))
    probs = exp / np.maximum(1e-12, np.sum(exp, axis=1, keepdims=True))
    error = (probs - y_onehot) * type_weights[:, None]
    grad = np.sum(Xs[:, :, None] * error[:, None, :], axis=0) / float(
        len(y_type)
    )
    grad += float(l2) * weights
    grad[0, :] -= float(l2) * weights[0, :]
    weights -= lr * grad
    if epoch > 0 and epoch % 250 == 0:
      lr *= 0.72
  return weights


def _predict_scores(
    X: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
  Xs = (X - mean) / scale
  logits = Xs @ weights
  return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def _predict_type_probs(
    X: np.ndarray,
    type_weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
  Xs = (X - mean) / scale
  logits = np.sum(Xs[:, :, None] * type_weights[None, :, :], axis=1)
  logits = logits - np.max(logits, axis=1, keepdims=True)
  exp = np.exp(np.clip(logits, -60.0, 60.0))
  return exp / np.maximum(1e-12, np.sum(exp, axis=1, keepdims=True))


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


def _contact_type_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
  if len(labels) <= 0:
    return {"rows": 0}
  preds = np.argmax(probs, axis=1)
  result: dict[str, Any] = {
      "accuracy": round(float(np.mean(preds == labels)), 4),
      "rows": int(len(labels)),
  }
  for idx, name in enumerate(CONTACT_TYPES):
    truth = labels == idx
    pred = preds == idx
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result[f"{name}_precision"] = round(float(precision), 4)
    result[f"{name}_recall"] = round(float(recall), 4)
    result[f"{name}_support"] = int(np.sum(truth))
  return result


def _heuristic_contact_type(features: dict[str, float]) -> str:
  if float(features.get("pair_threaded_fastener", 0.0)) > 0.5:
    return "screw_in_hole"
  if float(features.get("pair_center_bore_pin", 0.0)) > 0.5:
    return "shaft_in_bore"
  if float(features.get("pair_boss_slot", 0.0)) > 0.5:
    return "boss_in_slot"
  if float(features.get("pair_pin_slot", 0.0)) > 0.5:
    return "pin_in_slot"
  if float(features.get("pair_planar_planar", 0.0)) > 0.5:
    return "seat_plane"
  if float(features.get("pair_hole_shaft", 0.0)) > 0.5:
    return "insert_axis"
  if float(features.get("pair_cylindrical_cylindrical", 0.0)) > 0.5:
    return "insert_axis"
  return "generic_contact"


def _align_named_values(
    *,
    feature_names: list[str],
    values: Any,
    default: float,
) -> np.ndarray:
  raw = np.asarray(values, dtype=float).reshape(-1)
  if len(raw) == len(PAIR_FEATURE_NAMES) and feature_names == list(PAIR_FEATURE_NAMES):
    return raw
  mapping = {name: idx for idx, name in enumerate(feature_names)}
  aligned = np.full(len(PAIR_FEATURE_NAMES), float(default), dtype=float)
  for idx, name in enumerate(PAIR_FEATURE_NAMES):
    old_idx = mapping.get(name)
    if old_idx is None or old_idx >= len(raw):
      continue
    aligned[idx] = float(raw[old_idx])
  return aligned


def _align_vector_payload(
    *,
    feature_names: list[str],
    values: Any,
    default: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  weights = _align_named_values(
      feature_names=feature_names,
      values=values,
      default=default,
  )
  return weights, np.zeros_like(weights), np.ones_like(weights)


def _align_type_weights_payload(
    *,
    feature_names: list[str],
    contact_types: tuple[str, ...],
    values: Any,
    target_contact_types: tuple[str, ...],
) -> np.ndarray | None:
  if values is None:
    return None
  raw = np.asarray(values, dtype=float)
  if raw.ndim != 2:
    return None
  feature_map = {name: idx for idx, name in enumerate(feature_names)}
  type_map = {name: idx for idx, name in enumerate(contact_types)}
  aligned = np.zeros((len(PAIR_FEATURE_NAMES), len(target_contact_types)), dtype=float)
  for feature_idx, feature_name in enumerate(PAIR_FEATURE_NAMES):
    old_feature_idx = feature_map.get(feature_name)
    if old_feature_idx is None or old_feature_idx >= raw.shape[0]:
      continue
    for type_idx, type_name in enumerate(target_contact_types):
      old_type_idx = type_map.get(type_name)
      if old_type_idx is None or old_type_idx >= raw.shape[1]:
        continue
      aligned[feature_idx, type_idx] = float(raw[old_feature_idx, old_type_idx])
  return aligned


def _ranking_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    ks: list[int],
) -> dict[str, Any]:
  grouped: dict[tuple[str, str, str, str, str], list[tuple[float, int]]] = {}
  for row, score in zip(rows, scores):
    key = (
        str(row.get("case_id") or ""),
        str(row.get("part_a") or ""),
        str(row.get("part_b") or ""),
        str(row.get("body_uuid_a") or ""),
        str(row.get("body_uuid_b") or ""),
    )
    grouped.setdefault(key, []).append((float(score), int(row.get("label", 0))))
  result: dict[str, Any] = {"groups": int(len(grouped))}
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
    result[f"recall@{k}"] = round(float(np.mean(recalls)) if recalls else 0.0, 4)
    result[f"precision@{k}"] = round(float(np.mean(precisions)) if precisions else 0.0, 4)
    result[f"hit_rate@{k}"] = round(float(np.mean(hit_rates)) if hit_rates else 0.0, 4)
  return result


if __name__ == "__main__":
  main()
