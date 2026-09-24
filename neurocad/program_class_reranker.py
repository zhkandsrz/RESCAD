"""Learned reranker for symbolic mate-program candidates.

The model is intentionally small and neuro-symbolic:

* symbolic retrievers generate a finite candidate set of CAD-mined mate programs;
* a learned pairwise ranker orders those executable templates;
* downstream CAD verification still accepts or rejects the instantiated pose.

The ranker never uses the held-out query residual as an input feature.  Query
residuals are used only as training/evaluation labels through ``program_class``.
Candidate residual features are allowed because they are the train-library mate
program templates being selected.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .benchmark_v2_constants import PROTOCOL_VERSION
from .interface_pair_features import PAIR_FEATURE_NAMES
from .mate_pose_retriever import MateProgramRetrieval, MateProgramRetriever, load_mate_programs
from .mate_programs import MateProgram, rotation_angle_degrees
from .paths import resolve_path


ROLE_FAMILIES = ("hole", "pin", "slot", "plane", "generic")
CONTACT_FAMILIES = ("insert", "seat", "slot", "generic")
MOTION_TYPES = ("flush", "axial_pos", "axial_neg", "lateral", "oblique")
DISTANCE_BINS = ("short", "medium", "long", "far")
ROTATION_BINS = (
    "aligned",
    "tilted",
    "quarter",
    "oblique",
    "half_turn",
    "axis_aligned",
    "axis_tilted",
    "axis_quarter",
    "axis_oblique",
    "axis_half_turn",
)
SAFE_PAIR_FEATURE_NAMES = tuple(
    name for name in PAIR_FEATURE_NAMES if name != "bias" and not name.startswith("residual_")
)

# Fixed decoder terms shared by learned and non-learned matched-pool rows.
# Keeping them explicit lets the benchmark isolate the contribution of s_theta.
DECODER_ROLE_RADIUS_WEIGHT = 0.20
DECODER_RETRIEVAL_WEIGHT = 0.05
PROGRAM_CLASS_SCHEMA = "mate_program_class.v1"
FORMAL_PROGRAM_FEATURE_POLICY = "learned_geometry_only.v1"

# These fields encode the retrieval order or a hand-written score.  Legacy
# ablations may use them, but formal benchmark-v2 models must mask them so the
# learned proposer cannot collapse into a disguised heuristic/retrieval hybrid.
FORMAL_FORBIDDEN_PROGRAM_FEATURES = frozenset(
    {
        "base_score",
        "base_rank_inv",
        "role_radius_score",
        "candidate_heuristic_prior",
    }
)


PROGRAM_RERANK_FEATURE_NAMES = [
    "bias",
    "base_score",
    "base_rank_inv",
    "role_radius_score",
    "parent_role_exact",
    "child_role_exact",
    "parent_role_family",
    "child_role_family",
    "role_pair_exact",
    "role_pair_family",
    "parent_surface_exact",
    "child_surface_exact",
    "relation_exact",
    "relation_family",
    "contact_type_exact",
    "contact_family_match",
    "parent_radius_abs_diff",
    "child_radius_abs_diff",
    "parent_radius_rel_diff",
    "child_radius_rel_diff",
    "candidate_translation_norm_log",
    "candidate_axial_abs_log",
    "candidate_lateral_abs_log",
    "candidate_heuristic_prior",
    *[f"query_parent_role_family_{name}" for name in ROLE_FAMILIES],
    *[f"query_child_role_family_{name}" for name in ROLE_FAMILIES],
    *[f"candidate_parent_role_family_{name}" for name in ROLE_FAMILIES],
    *[f"candidate_child_role_family_{name}" for name in ROLE_FAMILIES],
    *[f"query_contact_family_{name}" for name in CONTACT_FAMILIES],
    *[f"candidate_contact_family_{name}" for name in CONTACT_FAMILIES],
    *[f"candidate_motion_{name}" for name in MOTION_TYPES],
    *[f"candidate_distance_{name}" for name in DISTANCE_BINS],
    *[f"candidate_rotation_{name}" for name in ROTATION_BINS],
    *[f"query_feature_{name}" for name in SAFE_PAIR_FEATURE_NAMES],
    *[f"candidate_feature_{name}" for name in SAFE_PAIR_FEATURE_NAMES],
    *[f"pair_feature_absdiff_{name}" for name in SAFE_PAIR_FEATURE_NAMES],
]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="command", required=True)

  train = sub.add_parser("train", help="Train a pairwise program-class reranker.")
  train.add_argument("--library", required=True)
  train.add_argument(
      "--input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )
  train.add_argument("--library_manifest", default=None)
  train.add_argument("--family_split_manifest", default=None)
  train.add_argument("--output_model", required=True)
  train.add_argument("--pool_size", type=int, default=256)
  train.add_argument("--pairs_per_query", type=int, default=16)
  train.add_argument("--epochs", type=int, default=350)
  train.add_argument("--learning_rate", type=float, default=0.18)
  train.add_argument("--l2", type=float, default=1e-4)
  train.add_argument("--seed", type=int, default=17)
  train.add_argument(
      "--model_family",
      choices=["linear", "mlp"],
      default="linear",
      help=(
          "linear trains the original pairwise ranker; mlp trains a lightweight "
          "one-hidden-layer neural reranker on the same symbolic candidate pool."
      ),
  )
  train.add_argument("--hidden_dim", type=int, default=64)
  train.add_argument("--exclude_same_case", action=argparse.BooleanOptionalAction, default=True)
  train.add_argument(
      "--feature_drop_group",
      action="append",
      default=[],
      choices=[
          "role_contact",
          "radius_geometry",
          "retrieval_score",
          "safe_pair_features",
          "motion_bins",
      ],
      help=(
          "Feature-group ablation. May be supplied multiple times; dropped "
          "features receive zero train-time and test-time weights."
      ),
  )
  train.add_argument(
      "--negative_mode",
      choices=["hard", "random", "mixed"],
      default="hard",
      help="Negative mining mode for pairwise rank training.",
  )
  train.add_argument(
      "--train_fraction",
      type=float,
      default=1.0,
      help=(
          "Fraction of train mate programs used as supervised training "
          "queries. The train-only retrieval library remains unchanged."
      ),
  )

  inspect = sub.add_parser("inspect", help="Print model metadata.")
  inspect.add_argument("--model", required=True)

  return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  """Resolve training and inspection paths at the project root."""
  names = (
      ("library", "library_manifest", "family_split_manifest", "output_model")
      if args.command == "train"
      else ("model",)
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
        library=Path(str(args.library)),
        input_protocol=str(args.input_protocol),
        library_manifest=(
            Path(str(args.library_manifest)) if args.library_manifest else None
        ),
        family_split_manifest=(
            Path(str(args.family_split_manifest))
            if args.family_split_manifest
            else None
        ),
        pool_size=int(args.pool_size),
        pairs_per_query=int(args.pairs_per_query),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        seed=int(args.seed),
        model_family=str(args.model_family),
        hidden_dim=int(args.hidden_dim),
        exclude_same_case=bool(args.exclude_same_case),
        feature_drop_groups=list(args.feature_drop_group or []),
        negative_mode=str(args.negative_mode),
        train_fraction=float(args.train_fraction),
    )
    output = Path(str(args.output_model))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return
  if args.command == "inspect":
    model = ProgramClassReranker.load(Path(str(args.model)))
    print(json.dumps(model.metadata, indent=2, sort_keys=True))
    return


class ProgramClassReranker:
  def __init__(
      self,
      *,
      weights: np.ndarray,
      mean: np.ndarray,
      scale: np.ndarray,
      feature_names: list[str],
      metadata: dict[str, Any],
      layers: list[dict[str, np.ndarray]] | None = None,
      active_mask: np.ndarray | None = None,
      artifact_sha256: str = "",
      formal_validated: bool = False,
  ) -> None:
    self.weights = np.asarray(weights, dtype=float).reshape(-1)
    self.mean = np.asarray(mean, dtype=float).reshape(-1)
    self.scale = np.asarray(scale, dtype=float).reshape(-1)
    self.feature_names = list(feature_names)
    self.metadata = dict(metadata)
    self.model_type = str(self.metadata.get("model_type") or "linear_pairwise_program_class_reranker")
    self.layers = list(layers or [])
    self.active_mask = (
        np.asarray(active_mask, dtype=float).reshape(-1)
        if active_mask is not None
        else np.ones_like(self.weights, dtype=float)
    )
    self.artifact_sha256 = str(artifact_sha256 or "")
    self.formal_validated = bool(formal_validated)
    self._validate_numeric_state()

  @staticmethod
  def load(
      path: str | Path,
      *,
      expected_protocol: str | None = None,
      formal: bool = False,
      mate_library_path: str | Path | None = None,
      mate_library_manifest_path: str | Path | None = None,
      family_split_manifest_path: str | Path | None = None,
      expected_source_split_sha256: str | None = None,
  ) -> "ProgramClassReranker":
    model_path = Path(path)
    raw_bytes = model_path.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
      raise ValueError("program reranker artifact must be a JSON object")
    feature_names = list(data.get("feature_names") or PROGRAM_RERANK_FEATURE_NAMES)
    weights = _align_vector(feature_names, data.get("weights"), default=0.0)
    mean = _align_named_vector(feature_names, data.get("mean"), default=0.0)
    scale = _align_named_vector(feature_names, data.get("scale"), default=1.0)
    raw_layers = data.get("layers") or []
    layers: list[dict[str, np.ndarray]] = []
    for layer in raw_layers:
      if not isinstance(layer, dict):
        continue
      layers.append(
          {
              "weights": np.asarray(layer.get("weights"), dtype=float),
              "bias": np.asarray(layer.get("bias"), dtype=float),
          }
      )
    active_mask = _align_named_vector(feature_names, data.get("active_mask"), default=1.0)
    metadata = dict(data.get("metadata") or {})
    if "model_type" not in metadata:
      metadata["model_type"] = str(data.get("model_type") or "linear_pairwise_program_class_reranker")
    artifact_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if formal or expected_protocol is not None:
      _validate_formal_program_reranker_artifact(
          data,
          metadata=metadata,
          expected_protocol=(
              "benchmark_v2" if expected_protocol is None else str(expected_protocol)
          ),
          mate_library_path=mate_library_path,
          mate_library_manifest_path=mate_library_manifest_path,
          family_split_manifest_path=family_split_manifest_path,
          expected_source_split_sha256=expected_source_split_sha256,
      )
    strict_scale = bool(formal or expected_protocol is not None)
    effective_scale = (
        scale
        if strict_scale
        else np.where(np.abs(scale) < 1e-8, 1.0, scale)
    )
    return ProgramClassReranker(
        weights=weights,
        mean=mean,
        scale=effective_scale,
        feature_names=PROGRAM_RERANK_FEATURE_NAMES,
        metadata=metadata,
        layers=layers,
        active_mask=active_mask,
        artifact_sha256=artifact_sha256,
        formal_validated=bool(formal or expected_protocol is not None),
    )

  def _validate_numeric_state(self) -> None:
    width = len(PROGRAM_RERANK_FEATURE_NAMES)
    for name, vector in (
        ("weights", self.weights),
        ("mean", self.mean),
        ("scale", self.scale),
        ("active_mask", self.active_mask),
    ):
      if vector.shape != (width,):
        raise ValueError(f"program reranker {name} has invalid shape")
      if not np.all(np.isfinite(vector)):
        raise ValueError(f"program reranker {name} contains non-finite values")
    if np.any(np.abs(self.scale) < 1e-12):
      raise ValueError("program reranker scale contains zeros")
    input_width = width
    for index, layer in enumerate(self.layers):
      weights = np.asarray(layer.get("weights"), dtype=float)
      bias = np.asarray(layer.get("bias"), dtype=float).reshape(-1)
      if weights.ndim != 2 or weights.shape[0] != input_width:
        raise ValueError(f"program reranker layer {index} has invalid weights")
      if bias.shape != (weights.shape[1],):
        raise ValueError(f"program reranker layer {index} has invalid bias")
      if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
        raise ValueError(f"program reranker layer {index} is non-finite")
      input_width = int(weights.shape[1])
    if self.layers and input_width != 1:
      raise ValueError("program reranker final layer must emit one score")

  def score(
      self,
      query: MateProgram,
      candidate: MateProgram,
      *,
      base_score: float = 0.0,
      base_rank: int = 0,
  ) -> float:
    x = np.asarray(
        program_rerank_feature_vector(
            query,
            candidate,
            base_score=base_score,
            base_rank=base_rank,
        ),
        dtype=float,
    )
    x = (x - self.mean) / self.scale
    if self.layers:
      z = x * self.active_mask
      for index, layer in enumerate(self.layers):
        weights = np.asarray(layer["weights"], dtype=float)
        bias = np.asarray(layer["bias"], dtype=float)
        z = z @ weights + bias
        if index < len(self.layers) - 1:
          z = np.tanh(z)
      return float(np.asarray(z).reshape(-1)[0])
    return float(np.dot(x, self.weights))

  def rerank(
      self,
      query: MateProgram,
      candidates: list[MateProgramRetrieval],
      *,
      contact_guard: bool = False,
      base_score_weight: float = DECODER_RETRIEVAL_WEIGHT,
      role_radius_weight: float = DECODER_ROLE_RADIUS_WEIGHT,
      diversify_program_class: bool = True,
  ) -> list[MateProgramRetrieval]:
    scored = []
    for rank, row in enumerate(candidates):
      model_score = self.score(
          query,
          row.program,
          base_score=float(row.score),
          base_rank=rank,
      )
      final_score = model_score + fixed_decoder_score(
          query,
          row.program,
          retrieval_score=float(row.score) if contact_guard else 0.0,
          role_radius_weight=role_radius_weight,
          retrieval_weight=base_score_weight,
      )
      tier = contact_guard_tier(query, row.program) if contact_guard else 0
      scored.append((tier, final_score, rank, row))
    scored.sort(
        key=lambda item: (
            -int(item[0]),
            -float(item[1]),
            item[2],
            item[3].program.program_id,
        )
    )
    rows = [
        MateProgramRetrieval(
            program=row.program,
            score=float(score),
            reasons=[
                *row.reasons,
                "program_class_reranker",
                *([f"contact_guard_tier={tier}"] if contact_guard else []),
            ],
        )
        for tier, score, _rank, row in scored
    ]
    if diversify_program_class:
      rows = diverse_top_k(rows)
    return rows


def program_reranker_parameter_sha256(payload: dict[str, Any]) -> str:
  """Hash every numeric/model-schema field that affects inference."""

  bound = {
      "model_type": payload.get("model_type"),
      "feature_names": payload.get("feature_names"),
      "weights": payload.get("weights"),
      "mean": payload.get("mean"),
      "scale": payload.get("scale"),
      "layers": payload.get("layers"),
      "active_mask": payload.get("active_mask"),
  }
  encoded = json.dumps(
      bound,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(payload: Any) -> str:
  encoded = json.dumps(
      payload,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _program_trainer_code_sha256() -> str:
  """Hash the code bundle that constructs formal program-ranker artifacts."""

  root = Path(__file__).resolve().parent
  rows = []
  for path in (
      root / "program_class_reranker.py",
      root / "mate_pose_retriever.py",
      root / "mate_programs.py",
      root / "interface_pair_features.py",
  ):
    rows.append({"name": path.name, "sha256": _file_sha256(path)})
  return _canonical_sha256(rows)


def _validate_formal_program_reranker_artifact(
    payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
    expected_protocol: str,
    mate_library_path: str | Path | None,
    mate_library_manifest_path: str | Path | None,
    family_split_manifest_path: str | Path | None,
    expected_source_split_sha256: str | None,
) -> None:
  protocol = str(expected_protocol or "").strip().lower()
  if protocol != "benchmark_v2":
    raise ValueError("formal program reranker supports only benchmark_v2")
  expected = {
      "model_input_protocol": "benchmark_v2",
      "protocol_version": "benchmark_v2_independent_se3_v1",
      "source_split": "train",
      "program_class_schema": PROGRAM_CLASS_SCHEMA,
      "feature_policy": FORMAL_PROGRAM_FEATURE_POLICY,
  }
  for key, value in expected.items():
    if metadata.get(key) != value:
      raise ValueError(f"formal program reranker {key} mismatch")
  for key in (
      "source_split_sha256",
      "training_data_sha256",
      "training_config_sha256",
      "model_parameters_sha256",
      "mate_library_manifest_sha256",
      "family_split_manifest_sha256",
      "trainer_code_sha256",
  ):
    value = str(metadata.get(key) or "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
      raise ValueError(f"formal program reranker lacks valid {key}")
  actual_parameters = program_reranker_parameter_sha256(payload)
  if metadata.get("model_parameters_sha256") != actual_parameters:
    raise ValueError("formal program reranker parameter hash mismatch")
  family_splits = metadata.get("observed_splits")
  if family_splits != ["train"]:
    raise ValueError("formal program reranker must be trained only on train")
  class_count = metadata.get("program_class_count")
  if not isinstance(class_count, int) or int(class_count) < 2:
    raise ValueError("formal program reranker requires at least two program classes")
  if payload.get("feature_names") != list(PROGRAM_RERANK_FEATURE_NAMES):
    raise ValueError("formal program reranker feature schema mismatch")
  raw_active_mask = payload.get("active_mask")
  if not isinstance(raw_active_mask, dict) or set(raw_active_mask) != set(
      PROGRAM_RERANK_FEATURE_NAMES
  ):
    raise ValueError("formal program reranker active-mask schema mismatch")
  for name in FORMAL_FORBIDDEN_PROGRAM_FEATURES:
    if float(raw_active_mask.get(name, 1.0)) != 0.0:
      raise ValueError(
          f"formal program reranker must mask retrieval/heuristic feature {name}"
      )
  if not _formal_program_model_is_non_degenerate(payload):
    raise ValueError("formal program reranker is numerically degenerate")
  metrics = payload.get("metrics")
  if not isinstance(metrics, dict) or metadata.get("metrics") != metrics:
    raise ValueError("formal program reranker training diagnostics mismatch")
  for key in ("query_count", "ranking_pair_count"):
    if not isinstance(metrics.get(key), int) or int(metrics[key]) < 1:
      raise ValueError(f"formal program reranker has invalid {key}")
  for key in (
      "learned_score_tie_rate",
      "retrieval_pair_accuracy",
      "learned_vs_retrieval_pair_disagreement_rate",
  ):
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
      raise ValueError(f"formal program reranker lacks finite {key}")
    if not 0.0 <= float(value) <= 1.0:
      raise ValueError(f"formal program reranker has out-of-range {key}")
  if float(metrics["learned_score_tie_rate"]) >= 1.0:
    raise ValueError("formal program reranker ties every supervised pair")
  if (
      mate_library_path is None
      or mate_library_manifest_path is None
      or family_split_manifest_path is None
  ):
    raise ValueError(
        "formal program reranker requires bound mate library and mate/family manifests"
    )
  if metadata["training_data_sha256"] != _file_sha256(mate_library_path):
    raise ValueError("formal program reranker training library hash mismatch")
  if metadata["mate_library_manifest_sha256"] != _file_sha256(
      mate_library_manifest_path
  ):
    raise ValueError("formal program reranker mate manifest hash mismatch")
  if metadata["family_split_manifest_sha256"] != _file_sha256(
      family_split_manifest_path
  ):
    raise ValueError("formal program reranker family manifest hash mismatch")
  training_config = payload.get("training_config")
  if not isinstance(training_config, dict):
    raise ValueError("formal program reranker lacks locked training config")
  if metadata["training_config_sha256"] != _canonical_sha256(training_config):
    raise ValueError("formal program reranker training config hash mismatch")
  if metadata["trainer_code_sha256"] != _program_trainer_code_sha256():
    raise ValueError("formal program reranker trainer code hash mismatch")
  if (
      expected_source_split_sha256 is None
      or metadata["source_split_sha256"] != str(expected_source_split_sha256)
  ):
    raise ValueError("formal program reranker source split hash mismatch")


def _formal_program_model_is_non_degenerate(payload: dict[str, Any]) -> bool:
  """Check that an allowed learned input can change the serialized score."""

  names = list(PROGRAM_RERANK_FEATURE_NAMES)
  active = _align_named_vector(names, payload.get("active_mask"), default=0.0)
  allowed_indices = [
      index
      for index, name in enumerate(names)
      if name != "bias"
      and name not in FORMAL_FORBIDDEN_PROGRAM_FEATURES
      and float(active[index]) > 0.0
  ]
  if not allowed_indices:
    return False
  weights = _align_vector(names, payload.get("weights"), default=0.0)
  layers = payload.get("layers") or []
  if not layers:
    return bool(np.max(np.abs(weights[allowed_indices])) > 1e-10)

  def evaluate(vector: np.ndarray) -> float:
    z = vector * active
    for layer_index, layer in enumerate(layers):
      matrix = np.asarray(layer.get("weights"), dtype=float)
      bias = np.asarray(layer.get("bias"), dtype=float)
      z = z @ matrix + bias
      if layer_index < len(layers) - 1:
        z = np.tanh(z)
    return float(np.asarray(z).reshape(-1)[0])

  probe_scores = [evaluate(np.zeros(len(names), dtype=float))]
  for index in allowed_indices:
    for sign in (-1.0, 1.0):
      probe = np.zeros(len(names), dtype=float)
      probe[index] = sign
      probe_scores.append(evaluate(probe))
  return bool(max(probe_scores) - min(probe_scores) > 1e-10)


def train_model(
    *,
    library: Path,
    input_protocol: str = "legacy",
    library_manifest: Path | None = None,
    family_split_manifest: Path | None = None,
    pool_size: int,
    pairs_per_query: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    model_family: str = "linear",
    hidden_dim: int = 64,
    exclude_same_case: bool,
    feature_drop_groups: list[str] | tuple[str, ...] | None = None,
    negative_mode: str = "hard",
    train_fraction: float = 1.0,
) -> dict[str, Any]:
  protocol = str(input_protocol or "legacy").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("program reranker input protocol must be legacy or benchmark_v2")
  if protocol == "benchmark_v2" and not bool(exclude_same_case):
    raise ValueError("formal program training requires exclude_same_case")
  mate_manifest: dict[str, Any] | None = None
  if protocol == "benchmark_v2":
    if family_split_manifest is None:
      raise ValueError("benchmark_v2 program training requires a family split manifest")
    retriever = MateProgramRetriever.load(
        library,
        input_protocol="benchmark_v2",
        formal=True,
        manifest_path=library_manifest,
        family_split_manifest_path=family_split_manifest,
    )
    programs = list(retriever.programs)
    mate_manifest = dict(retriever.library_manifest or {})
  else:
    programs = load_mate_programs(library)
  if not programs:
    raise ValueError(f"No mate programs loaded from {library}")

  rng = np.random.default_rng(int(seed))
  feature_drop_groups = sorted({str(item) for item in (feature_drop_groups or [])})
  active_mask = _feature_active_mask(feature_drop_groups)
  if protocol == "benchmark_v2":
    for name in FORMAL_FORBIDDEN_PROGRAM_FEATURES:
      active_mask[PROGRAM_RERANK_FEATURE_NAMES.index(name)] = 0.0
  model_family = str(model_family or "linear").lower()
  if model_family not in {"linear", "mlp"}:
    raise ValueError(f"unsupported model_family={model_family}")
  negative_mode = str(negative_mode or "hard").lower()
  if negative_mode not in {"hard", "random", "mixed"}:
    raise ValueError(f"unsupported negative_mode={negative_mode}")
  train_fraction = max(0.0, min(1.0, float(train_fraction)))
  query_programs = list(programs)
  if train_fraction < 1.0:
    keep = max(1, int(round(train_fraction * len(query_programs))))
    order = rng.permutation(len(query_programs))[:keep]
    query_programs = [query_programs[int(index)] for index in order]
  pair_pos: list[list[float]] = []
  pair_neg: list[list[float]] = []
  query_count = 0
  skipped_no_positive = 0
  positive_recall_at_pool = 0
  positive_counts: Counter[str] = Counter()

  retrievers: dict[str, MateProgramRetriever] = {}
  for query in query_programs:
    query_count += 1
    key = query.case_id if exclude_same_case else "__all__"
    if key not in retrievers:
      candidates = [
          program for program in programs
          if not (exclude_same_case and program.case_id == key)
      ]
      retrievers[key] = MateProgramRetriever(candidates, use_contact_priors=True)
    pool = symbolic_candidate_pool(
        query,
        retrievers[key],
        retrievers[key].programs,
        pool_size=int(pool_size),
    )
    if not pool:
      skipped_no_positive += 1
      continue
    target_class = mate_program_class_from_program(query)
    positives = [
        (rank, row)
        for rank, row in enumerate(pool)
        if mate_program_class_from_program(row.program) == target_class
    ]
    negatives = [
        (rank, row)
        for rank, row in enumerate(pool)
        if mate_program_class_from_program(row.program) != target_class
    ]
    if not positives or not negatives:
      skipped_no_positive += int(not positives)
      continue
    positive_recall_at_pool += 1
    positive_counts[target_class] += 1
    train_negatives = _training_negatives(
        query=query,
        negatives=negatives,
        rng=rng,
        mode=negative_mode,
    )
    sampled_pairs = _sample_rank_pairs(
        positives=positives,
        negatives=train_negatives,
        rng=rng,
        limit=int(pairs_per_query),
    )
    for pos_rank, pos_row, neg_rank, neg_row in sampled_pairs:
      pair_pos.append(
          program_rerank_feature_vector(
              query,
              pos_row.program,
              base_score=float(pos_row.score),
              base_rank=int(pos_rank),
          )
      )
      pair_neg.append(
          program_rerank_feature_vector(
              query,
              neg_row.program,
              base_score=float(neg_row.score),
              base_rank=int(neg_rank),
          )
      )

  if not pair_pos:
    raise ValueError("No ranking pairs generated for program-class reranker.")

  X_pos = np.asarray(pair_pos, dtype=float)
  X_neg = np.asarray(pair_neg, dtype=float)
  X_all = np.vstack([X_pos, X_neg])
  mean = X_all.mean(axis=0)
  scale = X_all.std(axis=0)
  scale = np.where(scale < 1e-8, 1.0, scale)
  mean[0] = 0.0
  scale[0] = 1.0
  D = (X_pos - mean) / scale - (X_neg - mean) / scale
  D = D * active_mask

  layers: list[dict[str, Any]] = []
  if model_family == "linear":
    weights = np.zeros(D.shape[1], dtype=float)
    initial_objective = pairwise_logistic_objective(D, weights, l2=float(l2))
    lr = float(learning_rate)
    for epoch in range(max(1, int(epochs))):
      margins = np.clip(D @ weights, -60.0, 60.0)
      wrong_prob = 1.0 / (1.0 + np.exp(margins))
      grad = -(wrong_prob[:, None] * D).mean(axis=0) + float(l2) * weights
      grad[0] -= 0.0
      weights -= lr * grad
      if epoch and epoch % 120 == 0:
        lr *= 0.72
    margins = D @ weights
    weights = weights * active_mask
    final_objective = pairwise_logistic_objective(D, weights, l2=float(l2))
  else:
    weights, layers, margins = _train_mlp_pairwise(
        X_pos=(X_pos - mean) / scale * active_mask,
        X_neg=(X_neg - mean) / scale * active_mask,
        hidden_dim=int(hidden_dim),
        epochs=max(1, int(epochs)),
        learning_rate=float(learning_rate),
        l2=float(l2),
        rng=rng,
    )
    initial_objective = float("nan")
    final_objective = float(np.mean(np.logaddexp(0.0, -margins)))
  train_pair_accuracy = float(np.mean(margins > 0.0))
  retrieval_index = PROGRAM_RERANK_FEATURE_NAMES.index("base_score")
  retrieval_margins = X_pos[:, retrieval_index] - X_neg[:, retrieval_index]
  learned_ties = np.abs(margins) <= 1e-10
  retrieval_ties = np.abs(retrieval_margins) <= 1e-10
  comparable = ~(learned_ties | retrieval_ties)
  disagreement = (
      float(
          np.mean(
              (margins[comparable] > 0.0)
              != (retrieval_margins[comparable] > 0.0)
          )
      )
      if np.any(comparable)
      else 0.0
  )
  metrics = {
      "query_count": query_count,
      "supervised_train_query_fraction": round(float(train_fraction), 4),
      "program_count": len(programs),
      "ranking_pair_count": int(D.shape[0]),
      "skipped_no_positive": skipped_no_positive,
      "positive_recall_at_pool": round(float(positive_recall_at_pool) / max(1.0, float(query_count)), 4),
      "train_pair_accuracy": round(train_pair_accuracy, 4),
      "learned_score_tie_rate": round(float(np.mean(learned_ties)), 6),
      "retrieval_pair_accuracy": round(
          float(np.mean(retrieval_margins > 0.0)), 4
      ),
      "learned_vs_retrieval_pair_disagreement_rate": round(disagreement, 6),
      "initial_pairwise_objective": (
          None if not math.isfinite(initial_objective) else round(initial_objective, 6)
      ),
      "final_pairwise_objective": round(final_objective, 6),
      "negative_mode": negative_mode,
      "model_family": model_family,
      "hidden_dim": int(hidden_dim) if model_family == "mlp" else 0,
      "feature_drop_groups": list(feature_drop_groups),
      "active_feature_count": int(np.sum(active_mask > 0.0)),
      "unique_positive_program_classes": len(positive_counts),
      "top_positive_program_classes": dict(positive_counts.most_common(12)),
  }
  training_config = {
      "pool_size": int(pool_size),
      "pairs_per_query": int(pairs_per_query),
      "epochs": int(epochs),
      "learning_rate": float(learning_rate),
      "l2": float(l2),
      "seed": int(seed),
      "model_family": model_family,
      "hidden_dim": int(hidden_dim) if model_family == "mlp" else 0,
      "exclude_same_case": bool(exclude_same_case),
      "feature_drop_groups": list(feature_drop_groups),
      "negative_mode": negative_mode,
      "train_fraction": round(float(train_fraction), 4),
      "input_protocol": protocol,
  }
  payload = {
      "model_type": f"{model_family}_pairwise_program_class_reranker",
      "feature_names": list(PROGRAM_RERANK_FEATURE_NAMES),
      "weights": [float(value) for value in weights],
      "layers": layers,
      "active_mask": {
          name: float(value)
          for name, value in zip(PROGRAM_RERANK_FEATURE_NAMES, active_mask)
      },
      "mean": {
          name: float(value)
          for name, value in zip(PROGRAM_RERANK_FEATURE_NAMES, mean)
      },
      "scale": {
          name: float(value)
          for name, value in zip(PROGRAM_RERANK_FEATURE_NAMES, scale)
      },
      "metadata": {
          "library": str(library),
          "pool_size": int(pool_size),
          "pairs_per_query": int(pairs_per_query),
          "epochs": int(epochs),
          "learning_rate": float(learning_rate),
          "l2": float(l2),
          "seed": int(seed),
          "model_type": f"{model_family}_pairwise_program_class_reranker",
          "model_family": model_family,
          "hidden_dim": int(hidden_dim) if model_family == "mlp" else 0,
          "exclude_same_case": bool(exclude_same_case),
          "feature_drop_groups": list(feature_drop_groups),
          "negative_mode": negative_mode,
          "train_fraction": round(float(train_fraction), 4),
          "metrics": metrics,
      },
      "metrics": metrics,
  }
  if protocol == "benchmark_v2":
    assert mate_manifest is not None
    program_classes = sorted(
        {mate_program_class_from_program(program) for program in programs}
    )
    if len(program_classes) < 2:
      raise ValueError("formal program reranker requires at least two program classes")
    metadata = payload["metadata"]
    payload["training_config"] = training_config
    metadata.update(
        {
            "model_input_protocol": "benchmark_v2",
            "protocol_version": PROTOCOL_VERSION,
            "source_split": "train",
            "observed_splits": ["train"],
            "program_class_schema": PROGRAM_CLASS_SCHEMA,
            "feature_policy": FORMAL_PROGRAM_FEATURE_POLICY,
            "program_class_count": len(program_classes),
            "source_split_sha256": str(mate_manifest["source_split_sha256"]),
            "training_data_sha256": _file_sha256(library),
            "training_config_sha256": _canonical_sha256(training_config),
            "mate_library_manifest_sha256": _file_sha256(
                library_manifest
                if library_manifest is not None
                else library.with_suffix(library.suffix + ".manifest.json")
            ),
            "family_split_manifest_sha256": _file_sha256(
                Path(family_split_manifest)
            ),
            "trainer_code_sha256": _program_trainer_code_sha256(),
        }
    )
    metadata["model_parameters_sha256"] = program_reranker_parameter_sha256(
        payload
    )
  return payload


def pairwise_logistic_objective(
    differences: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
) -> float:
  """Mean pairwise logistic loss with L2 regularization."""

  D = np.asarray(differences, dtype=float)
  w = np.asarray(weights, dtype=float).reshape(-1)
  margins = D @ w
  data_loss = float(np.mean(np.logaddexp(0.0, -margins)))
  return data_loss + 0.5 * float(l2) * float(np.dot(w, w))


def _train_mlp_pairwise(
    *,
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
  """Train a tiny one-hidden-layer pairwise ranker.

  This is deliberately small: the paper claim is about the finite mate-program
  target, not a large geometry backbone.  The neural variant gives reviewers a
  direct check that the representation is not tied to the linear scorer.
  """
  input_dim = int(X_pos.shape[1])
  hidden_dim = max(4, int(hidden_dim))
  W1 = rng.normal(0.0, 1.0 / math.sqrt(max(1, input_dim)), size=(input_dim, hidden_dim))
  b1 = np.zeros(hidden_dim, dtype=float)
  W2 = rng.normal(0.0, 1.0 / math.sqrt(max(1, hidden_dim)), size=(hidden_dim, 1))
  b2 = np.zeros(1, dtype=float)

  def forward(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H_pre = X @ W1 + b1
    H = np.tanh(H_pre)
    Y = H @ W2 + b2
    return H, Y.reshape(-1)

  mW1 = np.zeros_like(W1)
  vW1 = np.zeros_like(W1)
  mb1 = np.zeros_like(b1)
  vb1 = np.zeros_like(b1)
  mW2 = np.zeros_like(W2)
  vW2 = np.zeros_like(W2)
  mb2 = np.zeros_like(b2)
  vb2 = np.zeros_like(b2)
  beta1 = 0.9
  beta2 = 0.999
  eps = 1e-8
  lr = float(learning_rate)
  n = max(1, int(X_pos.shape[0]))

  for epoch in range(max(1, int(epochs))):
    H_pos, y_pos = forward(X_pos)
    H_neg, y_neg = forward(X_neg)
    margins = np.clip(y_pos - y_neg, -60.0, 60.0)
    d_margin = -(1.0 / (1.0 + np.exp(margins))) / float(n)
    d_y_pos = d_margin
    d_y_neg = -d_margin

    grad_W2 = H_pos.T @ d_y_pos[:, None] + H_neg.T @ d_y_neg[:, None] + float(l2) * W2
    grad_b2 = np.asarray([float(np.sum(d_y_pos) + np.sum(d_y_neg))])
    d_H_pos = d_y_pos[:, None] @ W2.T
    d_H_neg = d_y_neg[:, None] @ W2.T
    d_pre_pos = d_H_pos * (1.0 - H_pos * H_pos)
    d_pre_neg = d_H_neg * (1.0 - H_neg * H_neg)
    grad_W1 = X_pos.T @ d_pre_pos + X_neg.T @ d_pre_neg + float(l2) * W1
    grad_b1 = np.sum(d_pre_pos + d_pre_neg, axis=0)

    step = epoch + 1
    for param, grad, m, v in (
        (W1, grad_W1, mW1, vW1),
        (b1, grad_b1, mb1, vb1),
        (W2, grad_W2, mW2, vW2),
        (b2, grad_b2, mb2, vb2),
    ):
      m *= beta1
      m += (1.0 - beta1) * grad
      v *= beta2
      v += (1.0 - beta2) * (grad * grad)
      m_hat = m / (1.0 - beta1 ** step)
      v_hat = v / (1.0 - beta2 ** step)
      param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    if epoch and epoch % 180 == 0:
      lr *= 0.7

  _H_pos, y_pos = forward(X_pos)
  _H_neg, y_neg = forward(X_neg)
  margins = y_pos - y_neg
  linear_proxy = np.zeros(input_dim, dtype=float)
  layers = [
      {
          "weights": [[float(value) for value in row] for row in W1],
          "bias": [float(value) for value in b1],
      },
      {
          "weights": [[float(value) for value in row] for row in W2],
          "bias": [float(value) for value in b2],
      },
  ]
  return linear_proxy, layers, margins


def symbolic_candidate_pool(
    query: MateProgram,
    retriever: MateProgramRetriever,
    programs: list[MateProgram],
    *,
    pool_size: int,
) -> list[MateProgramRetrieval]:
  pool_size = max(1, int(pool_size))
  retrieved = retriever.query(
      parent_role=query.parent_role,
      child_role=query.child_role,
      relation_hint=query.relation_hint,
      contact_type=query.contact_type,
      parent_surface=query.parent_surface,
      child_surface=query.child_surface,
      parent_radius=query.parent_radius,
      child_radius=query.child_radius,
      top_k=pool_size,
  )
  role_rows = []
  for program in programs:
    score = role_radius_score(query, program)
    role_rows.append(
        MateProgramRetrieval(
            program=program,
            score=float(score),
            reasons=["role_radius_symbolic_pool"],
        )
    )
  role_rows.sort(key=lambda item: (-float(item.score), item.program.program_id))
  return _dedupe_retrievals([*retrieved, *role_rows[:pool_size]], limit=pool_size)


def program_rerank_feature_vector(
    query: MateProgram,
    candidate: MateProgram,
    *,
    base_score: float,
    base_rank: int,
) -> list[float]:
  values = {name: 0.0 for name in PROGRAM_RERANK_FEATURE_NAMES}
  values["bias"] = 1.0
  values["base_score"] = float(base_score)
  values["base_rank_inv"] = 1.0 / float(int(base_rank) + 1)
  values["role_radius_score"] = role_radius_score(query, candidate)

  query_parent_family = _role_family(query.parent_role)
  query_child_family = _role_family(query.child_role)
  candidate_parent_family = _role_family(candidate.parent_role)
  candidate_child_family = _role_family(candidate.child_role)
  query_contact_family = _contact_family(query.contact_type)
  candidate_contact_family = _contact_family(candidate.contact_type)

  values["parent_role_exact"] = float(query.parent_role == candidate.parent_role)
  values["child_role_exact"] = float(query.child_role == candidate.child_role)
  values["parent_role_family"] = float(query_parent_family == candidate_parent_family)
  values["child_role_family"] = float(query_child_family == candidate_child_family)
  values["role_pair_exact"] = float(
      query.parent_role == candidate.parent_role and query.child_role == candidate.child_role
  )
  values["role_pair_family"] = float(
      query_parent_family == candidate_parent_family and query_child_family == candidate_child_family
  )
  values["parent_surface_exact"] = float(query.parent_surface == candidate.parent_surface)
  values["child_surface_exact"] = float(query.child_surface == candidate.child_surface)
  values["relation_exact"] = float(str(query.relation_hint).lower() == str(candidate.relation_hint).lower())
  values["relation_family"] = float(_relation_family(query.relation_hint) == _relation_family(candidate.relation_hint))
  values["contact_type_exact"] = float(str(query.contact_type).lower() == str(candidate.contact_type).lower())
  values["contact_family_match"] = float(query_contact_family == candidate_contact_family)

  parent_abs, parent_rel = _radius_diffs(query.parent_radius, candidate.parent_radius)
  child_abs, child_rel = _radius_diffs(query.child_radius, candidate.child_radius)
  values["parent_radius_abs_diff"] = parent_abs
  values["child_radius_abs_diff"] = child_abs
  values["parent_radius_rel_diff"] = parent_rel
  values["child_radius_rel_diff"] = child_rel

  trans = np.asarray(candidate.residual_translation, dtype=float).reshape(3)
  values["candidate_translation_norm_log"] = math.log1p(float(np.linalg.norm(trans)))
  values["candidate_axial_abs_log"] = math.log1p(abs(float(trans[2])))
  values["candidate_lateral_abs_log"] = math.log1p(float(np.linalg.norm(trans[:2])))
  prior = candidate.features.get("heuristic_prior")
  values["candidate_heuristic_prior"] = float(prior) if isinstance(prior, (int, float)) else 0.0

  _set_one_hot(values, "query_parent_role_family", _known(query_parent_family, ROLE_FAMILIES, "generic"))
  _set_one_hot(values, "query_child_role_family", _known(query_child_family, ROLE_FAMILIES, "generic"))
  _set_one_hot(values, "candidate_parent_role_family", _known(candidate_parent_family, ROLE_FAMILIES, "generic"))
  _set_one_hot(values, "candidate_child_role_family", _known(candidate_child_family, ROLE_FAMILIES, "generic"))
  _set_one_hot(values, "query_contact_family", _known(query_contact_family, CONTACT_FAMILIES, "generic"))
  _set_one_hot(values, "candidate_contact_family", _known(candidate_contact_family, CONTACT_FAMILIES, "generic"))

  motion, distance_bin, rotation_bin = program_class_components(candidate)
  _set_one_hot(values, "candidate_motion", motion)
  _set_one_hot(values, "candidate_distance", distance_bin)
  _set_one_hot(values, "candidate_rotation", rotation_bin)

  for name in SAFE_PAIR_FEATURE_NAMES:
    q_value = _feature_value(query, name)
    c_value = _feature_value(candidate, name)
    values[f"query_feature_{name}"] = q_value
    values[f"candidate_feature_{name}"] = c_value
    values[f"pair_feature_absdiff_{name}"] = abs(q_value - c_value)

  return [float(values.get(name, 0.0)) for name in PROGRAM_RERANK_FEATURE_NAMES]


def mate_program_class_from_program(program: MateProgram) -> str:
  return mate_program_class(
      contact_type=program.contact_type,
      parent_role=program.parent_role,
      child_role=program.child_role,
      residual_rotation=np.asarray(program.residual_rotation, dtype=float),
      residual_translation=np.asarray(program.residual_translation, dtype=float),
  )


def mate_program_class(
    *,
    contact_type: str,
    parent_role: str,
    child_role: str,
    residual_rotation: np.ndarray,
    residual_translation: np.ndarray,
) -> str:
  family = _contact_family(contact_type)
  role_pair = f"{_role_family(parent_role)}->{_role_family(child_role)}"
  motion, distance_bin, rotation_bin = _program_class_components_from_arrays(
      contact_type=contact_type,
      residual_rotation=residual_rotation,
      residual_translation=residual_translation,
  )
  return ";".join([family, role_pair, _motion_label(motion, distance_bin), rotation_bin])


def program_class_components(program: MateProgram) -> tuple[str, str, str]:
  return _program_class_components_from_arrays(
      contact_type=program.contact_type,
      residual_rotation=np.asarray(program.residual_rotation, dtype=float),
      residual_translation=np.asarray(program.residual_translation, dtype=float),
  )


def role_radius_score(query: MateProgram, candidate: MateProgram) -> float:
  score = 0.0
  score -= 8.0 * float(candidate.parent_role != query.parent_role)
  score -= 8.0 * float(candidate.child_role != query.child_role)
  score -= 2.0 * float(_role_family(candidate.parent_role) != _role_family(query.parent_role))
  score -= 2.0 * float(_role_family(candidate.child_role) != _role_family(query.child_role))
  score -= _radius_distance(candidate.parent_radius, query.parent_radius)
  score -= _radius_distance(candidate.child_radius, query.child_radius)
  return float(score)


def fixed_decoder_score(
    query: MateProgram,
    candidate: MateProgram,
    *,
    retrieval_score: float,
    role_radius_weight: float = DECODER_ROLE_RADIUS_WEIGHT,
    retrieval_weight: float = DECODER_RETRIEVAL_WEIGHT,
) -> float:
  """Fixed score terms shared by matched learned/non-learned decoders."""

  return float(
      float(role_radius_weight) * role_radius_score(query, candidate)
      + float(retrieval_weight) * float(retrieval_score)
  )


def _program_class_components_from_arrays(
    *,
    contact_type: str,
    residual_rotation: np.ndarray,
    residual_translation: np.ndarray,
) -> tuple[str, str, str]:
  trans = np.asarray(residual_translation, dtype=float).reshape(3)
  rot = np.asarray(residual_rotation, dtype=float).reshape(3, 3)
  t_norm = float(np.linalg.norm(trans))
  axial = abs(float(trans[2]))
  lateral = float(np.linalg.norm(trans[:2]))
  if t_norm <= 1.0:
    motion = "flush"
    distance_bin = "short"
  elif axial >= 0.75 * max(t_norm, 1e-9):
    motion = "axial_pos" if trans[2] > 0.0 else "axial_neg"
    distance_bin = _coarse_distance_bin(axial)
  elif lateral >= 0.75 * max(t_norm, 1e-9):
    motion = "lateral"
    distance_bin = _coarse_distance_bin(lateral)
  else:
    motion = "oblique"
    distance_bin = _coarse_distance_bin(t_norm)

  family = _contact_family(contact_type)
  if family in {"insert", "seat", "slot"}:
    angle = _symmetry_rotation_error_degrees(rot, np.eye(3, dtype=float), contact_family=family)
    rotation_bin = "axis_" + _coarse_rotation_bin(angle)
  else:
    rotation_bin = _coarse_rotation_bin(rotation_angle_degrees(rot))
  return motion, distance_bin, rotation_bin


def _motion_label(motion: str, distance_bin: str) -> str:
  if motion == "flush":
    return "flush"
  if motion in {"axial_pos", "axial_neg"}:
    sign = "pos" if motion == "axial_pos" else "neg"
    return f"axial_{sign}_{distance_bin}"
  return f"{motion}_{distance_bin}"


def _hard_negatives(
    query: MateProgram,
    negatives: list[tuple[int, MateProgramRetrieval]],
) -> list[tuple[int, MateProgramRetrieval]]:
  def priority(item: tuple[int, MateProgramRetrieval]) -> tuple[float, float, int]:
    rank, row = item
    contact = float(str(row.program.contact_type).lower() == str(query.contact_type).lower())
    family = float(_contact_family(row.program.contact_type) == _contact_family(query.contact_type))
    role = float(row.program.parent_role == query.parent_role) + float(row.program.child_role == query.child_role)
    return (contact + 0.5 * family + 0.25 * role, float(row.score), -rank)

  return sorted(negatives, key=priority, reverse=True)


def _training_negatives(
    *,
    query: MateProgram,
    negatives: list[tuple[int, MateProgramRetrieval]],
    rng: np.random.Generator,
    mode: str,
) -> list[tuple[int, MateProgramRetrieval]]:
  if mode == "hard":
    return _hard_negatives(query, negatives)
  shuffled = list(negatives)
  if shuffled:
    order = rng.permutation(len(shuffled))
    shuffled = [shuffled[int(index)] for index in order]
  if mode == "random":
    return shuffled
  hard = _hard_negatives(query, negatives)
  seen: set[str] = set()
  mixed: list[tuple[int, MateProgramRetrieval]] = []
  for source in (hard[: max(16, len(hard) // 4)], shuffled):
    for item in source:
      program_id = item[1].program.program_id
      if program_id in seen:
        continue
      seen.add(program_id)
      mixed.append(item)
  return mixed


def _sample_rank_pairs(
    *,
    positives: list[tuple[int, MateProgramRetrieval]],
    negatives: list[tuple[int, MateProgramRetrieval]],
    rng: np.random.Generator,
    limit: int,
) -> list[tuple[int, MateProgramRetrieval, int, MateProgramRetrieval]]:
  limit = max(1, int(limit))
  positives = positives[: min(len(positives), max(4, limit))]
  negatives = negatives[: min(len(negatives), max(16, 4 * limit))]
  pairs = []
  for _ in range(limit):
    pos_rank, pos_row = positives[int(rng.integers(0, len(positives)))]
    neg_rank, neg_row = negatives[int(rng.integers(0, len(negatives)))]
    pairs.append((pos_rank, pos_row, neg_rank, neg_row))
  return pairs


def _dedupe_retrievals(
    rows: Iterable[MateProgramRetrieval],
    *,
    limit: int,
) -> list[MateProgramRetrieval]:
  merged: dict[str, MateProgramRetrieval] = {}
  for row in rows:
    program_id = row.program.program_id
    existing = merged.get(program_id)
    if existing is None or row.score > existing.score:
      merged[program_id] = row
  result = list(merged.values())
  result.sort(key=lambda item: (-float(item.score), item.program.program_id))
  return result[: max(1, int(limit))]


def contact_guard_tier(query: MateProgram, candidate: MateProgram) -> int:
  """Return the deterministic contact-priority tier used during decoding.

  Exact contact labels outrank candidates that only match the broader contact
  family; incompatible candidates form the fallback tier. Ranking within each
  tier is still learned.
  """

  query_contact = str(query.contact_type or "").strip().lower()
  candidate_contact = str(candidate.contact_type or "").strip().lower()
  if query_contact and candidate_contact == query_contact:
    return 2
  if _contact_family(candidate_contact) == _contact_family(query_contact):
    return 1
  return 0


def diverse_top_k(
    rows: list[MateProgramRetrieval],
    *,
    k: int | None = None,
) -> list[MateProgramRetrieval]:
  """Stable unique-program-class-first decoding with duplicate backfill.

  ``rows`` must already be sorted by the desired primary ranking. The first
  pass retains the highest-ranked candidate from every executable program
  class. If fewer than ``k`` distinct classes exist, deferred duplicates are
  appended in their original rank order.
  """

  selected: list[MateProgramRetrieval] = []
  deferred: list[MateProgramRetrieval] = []
  seen: set[str] = set()
  for row in rows:
    program_class = mate_program_class_from_program(row.program)
    if program_class in seen:
      deferred.append(row)
      continue
    selected.append(row)
    seen.add(program_class)
  result = selected + deferred
  return result if k is None else result[: max(0, int(k))]


def _diversify_by_program_class(rows: list[MateProgramRetrieval]) -> list[MateProgramRetrieval]:
  """Backward-compatible alias for older experiment scripts."""

  return diverse_top_k(rows)


def _feature_active_mask(drop_groups: list[str]) -> np.ndarray:
  dropped = set()
  for group in drop_groups:
    dropped.update(_feature_names_for_group(group))
  return np.asarray(
      [0.0 if name in dropped else 1.0 for name in PROGRAM_RERANK_FEATURE_NAMES],
      dtype=float,
  )


def _feature_names_for_group(group: str) -> set[str]:
  group = str(group)
  if group == "role_contact":
    tokens = (
        "role",
        "relation",
        "contact",
        "surface",
        "pair_hole",
        "pair_center",
        "pair_threaded",
        "pair_pin",
        "pair_boss",
        "pair_planar",
        "pair_cylindrical",
        "illegal_",
        "same_role",
        "same_surface",
    )
    return {
        name for name in PROGRAM_RERANK_FEATURE_NAMES
        if any(token in name for token in tokens)
    }
  if group == "radius_geometry":
    tokens = (
        "radius",
        "area",
        "edge_count",
        "concavity",
        "boundary",
        "composite",
        "axis_",
        "normal_",
        "z_dot",
        "facing_normals",
        "parallel_axes",
    )
    return {
        name for name in PROGRAM_RERANK_FEATURE_NAMES
        if any(token in name for token in tokens)
    }
  if group == "retrieval_score":
    return {
        "base_score",
        "base_rank_inv",
        "role_radius_score",
        "candidate_heuristic_prior",
        "query_feature_score_a",
        "query_feature_score_b",
        "query_feature_score_min",
        "query_feature_score_product",
        "query_feature_heuristic_prior",
        "candidate_feature_score_a",
        "candidate_feature_score_b",
        "candidate_feature_score_min",
        "candidate_feature_score_product",
        "candidate_feature_heuristic_prior",
        "pair_feature_absdiff_score_a",
        "pair_feature_absdiff_score_b",
        "pair_feature_absdiff_score_min",
        "pair_feature_absdiff_score_product",
        "pair_feature_absdiff_heuristic_prior",
    }
  if group == "safe_pair_features":
    return {
        name for name in PROGRAM_RERANK_FEATURE_NAMES
        if (
            name.startswith("query_feature_")
            or name.startswith("candidate_feature_")
            or name.startswith("pair_feature_absdiff_")
        )
    }
  if group == "motion_bins":
    return {
        name for name in PROGRAM_RERANK_FEATURE_NAMES
        if (
            name.startswith("candidate_motion_")
            or name.startswith("candidate_distance_")
            or name.startswith("candidate_rotation_")
            or name in {
                "candidate_translation_norm_log",
                "candidate_axial_abs_log",
                "candidate_lateral_abs_log",
            }
        )
    }
  return set()


def _feature_value(program: MateProgram, name: str) -> float:
  value = program.features.get(name)
  if isinstance(value, (int, float)) and math.isfinite(float(value)):
    return float(value)
  return 0.0


def _set_one_hot(values: dict[str, float], prefix: str, name: str) -> None:
  key = f"{prefix}_{name}"
  if key in values:
    values[key] = 1.0


def _known(value: str, choices: tuple[str, ...], fallback: str) -> str:
  return value if value in choices else fallback


def _radius_diffs(a: float | None, b: float | None) -> tuple[float, float]:
  if a is None or b is None:
    return 0.0, 1.0
  try:
    a = abs(float(a))
    b = abs(float(b))
  except Exception:
    return 0.0, 1.0
  if a <= 0.0 or b <= 0.0:
    return 0.0, 1.0
  abs_diff = abs(a - b)
  return math.log1p(abs_diff), min(5.0, abs_diff / max(a, b, 1e-6))


def _radius_distance(a: float | None, b: float | None) -> float:
  if a is None or b is None:
    return 1.0
  a = abs(float(a))
  b = abs(float(b))
  if a <= 0.0 or b <= 0.0:
    return 1.0
  return min(5.0, abs(a - b) / max(a, b, 1e-6))


def _coarse_distance_bin(value: float) -> str:
  value = float(value)
  if value <= 5.0:
    return "short"
  if value <= 25.0:
    return "medium"
  if value <= 100.0:
    return "long"
  return "far"


def _coarse_rotation_bin(angle_deg: float) -> str:
  angle_deg = abs(float(angle_deg))
  if angle_deg <= 5.0:
    return "aligned"
  if angle_deg <= 30.0:
    return "tilted"
  if angle_deg <= 75.0:
    return "quarter"
  if angle_deg <= 135.0:
    return "oblique"
  return "half_turn"


def _symmetry_rotation_error_degrees(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    contact_family: str,
) -> float:
  if contact_family not in {"insert", "seat", "slot"}:
    return _rotation_error_degrees(pred, target)
  try:
    a = np.asarray(pred, dtype=float).reshape(3, 3)[:, 2]
    b = np.asarray(target, dtype=float).reshape(3, 3)[:, 2]
    dot = abs(float(np.dot(_normalize(a), _normalize(b))))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))
  except Exception:
    return 180.0


def _rotation_error_degrees(pred: np.ndarray, target: np.ndarray) -> float:
  try:
    delta = np.asarray(pred, dtype=float).reshape(3, 3).T @ np.asarray(target, dtype=float).reshape(3, 3)
    return rotation_angle_degrees(delta)
  except Exception:
    return 180.0


def _normalize(value: np.ndarray) -> np.ndarray:
  arr = np.asarray(value, dtype=float).reshape(3)
  norm = float(np.linalg.norm(arr))
  if norm <= 1e-12:
    return np.array([0.0, 0.0, 1.0], dtype=float)
  return arr / norm


def _relation_family(relation: str) -> str:
  text = str(relation or "").lower()
  if any(token in text for token in ("insert", "bore", "hole", "shaft", "screw")):
    return "insert"
  if any(token in text for token in ("seat", "support", "plane")):
    return "seat"
  if "slot" in text:
    return "slot"
  return text or "generic"


def _contact_family(contact_type: str) -> str:
  text = str(contact_type or "").lower()
  if "slot" in text:
    return "slot"
  if any(token in text for token in ("insert", "bore", "hole", "shaft", "screw", "threaded")):
    return "insert"
  if any(token in text for token in ("seat", "support", "plane", "flange")):
    return "seat"
  return "generic"


def _role_family(role: str) -> str:
  text = str(role or "").lower()
  if text in {"center_bore", "threaded_hole", "hole_entry"}:
    return "hole"
  if text in {"shaft_axis", "pin_boss", "cylindrical_interface"}:
    return "pin"
  if "slot" in text:
    return "slot"
  if text in {"planar_seat", "shoulder_stop"}:
    return "plane"
  return text or "generic"


def _align_vector(
    stored_feature_names: list[str],
    values: Any,
    *,
    default: float,
) -> np.ndarray:
  if not isinstance(values, list):
    values = []
  mapping = {
      name: float(values[index])
      for index, name in enumerate(stored_feature_names)
      if index < len(values) and isinstance(values[index], (int, float))
  }
  return np.asarray(
      [float(mapping.get(name, default)) for name in PROGRAM_RERANK_FEATURE_NAMES],
      dtype=float,
  )


def _align_named_vector(
    stored_feature_names: list[str],
    values: Any,
    *,
    default: float,
) -> np.ndarray:
  if isinstance(values, dict):
    mapping = {
        str(name): float(value)
        for name, value in values.items()
        if isinstance(value, (int, float))
    }
  elif isinstance(values, list):
    mapping = {
        name: float(values[index])
        for index, name in enumerate(stored_feature_names)
        if index < len(values) and isinstance(values[index], (int, float))
    }
  else:
    mapping = {}
  return np.asarray(
      [float(mapping.get(name, default)) for name in PROGRAM_RERANK_FEATURE_NAMES],
      dtype=float,
  )


if __name__ == "__main__":
  main()
