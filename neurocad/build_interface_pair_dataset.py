"""Build train data for pair-level interface compatibility.

This uses assembly metadata only for labels in train/dev splits. Rows contain
local single-part interface features and a binary label for whether two local
interfaces were a true contact pair. It never serializes ground-truth relative
part transforms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import random
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

from .benchmark_v2_constants import PROTOCOL_VERSION
from .benchmark_v2_training_provenance import (
    BenchmarkV2AuthenticatedTrainingReceipts,
    CapturedFileArtifact,
    INTERFACE_PAIR_TRAINING_ROW_KEYS,
    INTERFACE_PAIR_TRAINING_ROW_SCHEMA,
    canonical_sha256,
    capture_file_artifact,
    reverify_captured_file_artifact,
    require_formal_binary_training_available,
    validate_private_supervision_binding,
    write_dataset_artifact,
)
from .batch_infer_assemble import (
    _index_step_files,
    _load_cases,
    _parse_dataset_roots,
    _resolve_part_path,
)
from .interface_features import (
    CandidateInterface,
    extract_candidate_interfaces_from_step,
    label_interfaces_from_assembly_contacts,
)
from .interface_pair_features import (
    BENCHMARK_V2_PAIR_FEATURE_NAMES,
    heuristic_pair_prior,
    pair_features_from_rows,
)
from .interface_scorer import InterfaceScorer
from .build_interface_dataset import (
    BenchmarkV2InterfaceRecord,
    BenchmarkV2ProducedInterfaceCase,
    _open_verified_benchmark_v2_case_inputs,
    _reverify_benchmark_v2_case_inputs,
    _validated_face_map_binding,
    load_benchmark_v2_authenticated_training_inputs,
    produce_benchmark_v2_interface_case,
)
from .benchmark_v2_private_supervision import (
    require_certified_binary_training_gold,
)
from .paths import resolve_dataset_roots, resolve_path


@dataclass(frozen=True, slots=True)
class BenchmarkV2InterfaceScorerArtifact:
  """A scorer parsed only from one captured model artifact."""

  scorer: InterfaceScorer
  capture: CapturedFileArtifact

  @property
  def sha256(self) -> str:
    return self.capture.sha256

  def reverify(self) -> None:
    reverify_captured_file_artifact(
        self.capture,
        label="benchmark_v2 interface scorer model",
    )


def load_benchmark_v2_interface_scorer_artifact(
    path: str | Path,
) -> BenchmarkV2InterfaceScorerArtifact:
  """Load a benchmark-v2 scorer from exact captured bytes via private staging."""

  captured = capture_file_artifact(
      path,
      label="benchmark_v2 interface scorer model",
  )
  with tempfile.TemporaryDirectory(prefix="neurocad-v2-scorer-") as raw_dir:
    staging = Path(raw_dir) / "captured_interface_scorer.json"
    staging.write_bytes(captured.raw_bytes)
    staged_capture = capture_file_artifact(
        staging,
        label="staged benchmark_v2 interface scorer model",
    )
    if (
        staged_capture.sha256 != captured.sha256
        or staged_capture.byte_count != captured.byte_count
    ):
      raise ValueError("staged interface scorer differs from captured bytes")
    scorer = InterfaceScorer.load(
        staging,
        expected_protocol="benchmark_v2",
    )
    reverify_captured_file_artifact(
        staged_capture,
        label="staged benchmark_v2 interface scorer model",
    )
  return BenchmarkV2InterfaceScorerArtifact(
      scorer=scorer,
      capture=captured,
  )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--cases_json",
      default="neurocad/paper_splits_candidate_main_deepseek_v1/train_oracle.json",
      help="Train/dev split JSON. Do not build pair labels from held-out test.",
  )
  parser.add_argument("--dataset_root", default=None)
  parser.add_argument("--dataset_roots", nargs="*", default=[])
  parser.add_argument("--output_jsonl", default="neurocad/interface_pair_train.jsonl")
  parser.add_argument("--summary_json", default=None)
  parser.add_argument("--interface_scorer_model", default=None)
  parser.add_argument("--max_cases", type=int, default=0)
  parser.add_argument("--max_candidates_per_part", type=int, default=64)
  parser.add_argument("--top_candidates_per_part_pair", type=int, default=24)
  parser.add_argument("--negative_ratio", type=int, default=8)
  parser.add_argument(
      "--slot_positive_oversample",
      type=int,
      default=1,
      help="Repeat positive rows involving obround_slot this many times.",
  )
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument(
      "--input_protocol",
      "--protocol",
      dest="input_protocol",
      choices=["legacy", "benchmark_v2"],
      default="legacy",
  )
  parser.add_argument("--frame_randomization_seed", type=int, default=17)
  parser.add_argument(
      "--benchmark_v2_translation_box_fraction",
      type=float,
      default=1.0,
  )
  parser.add_argument("--source_split", choices=["train", "dev"], default="train")
  parser.add_argument("--family_split_manifest", default=None)
  parser.add_argument(
      "--private-source-split",
      dest="private_source_split",
      default=None,
  )
  parser.add_argument(
      "--private-evaluation-gold-split",
      dest="private_evaluation_gold_split",
      default=None,
  )
  parser.add_argument(
      "--fusion-step-face-map-receipt",
      dest="fusion_step_face_map_receipt",
      default=None,
      help=(
          "Private receipt mapping Fusion contact face keys to verified STEP/OCC "
          "faces. Required for formal benchmark_v2 label production."
      ),
  )
  return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  """Anchor every filesystem argument at the checkout root."""
  roots = resolve_dataset_roots(args.dataset_root, args.dataset_roots)
  args.dataset_root = str(roots[0])
  args.dataset_roots = [str(path) for path in roots[1:]]
  args.cases_json = str(resolve_path(args.cases_json))
  args.output_jsonl = str(resolve_path(args.output_jsonl))
  if args.summary_json:
    args.summary_json = str(resolve_path(args.summary_json))
  if args.interface_scorer_model:
    args.interface_scorer_model = str(resolve_path(args.interface_scorer_model))
  if args.family_split_manifest:
    args.family_split_manifest = str(resolve_path(args.family_split_manifest))
  if args.private_source_split:
    args.private_source_split = str(resolve_path(args.private_source_split))
  if args.private_evaluation_gold_split:
    args.private_evaluation_gold_split = str(
        resolve_path(args.private_evaluation_gold_split)
    )
  if args.fusion_step_face_map_receipt:
    args.fusion_step_face_map_receipt = str(
        resolve_path(args.fusion_step_face_map_receipt)
    )
  return args


def main() -> None:
  args = normalize_paths(build_parser().parse_args())
  if str(args.input_protocol) == "benchmark_v2":
    _main_benchmark_v2(args)
    return
  _main_legacy(args)


def _main_legacy(args: argparse.Namespace) -> None:
  dataset_roots = _parse_dataset_roots(
      dataset_root=str(args.dataset_root),
      dataset_roots=list(args.dataset_roots or []),
  )
  if not dataset_roots:
    raise SystemExit("No dataset roots provided.")
  cases = _load_cases(Path(args.cases_json))
  if int(args.max_cases) > 0:
    cases = cases[: int(args.max_cases)]
  file_index = _index_step_files(dataset_roots)
  scorer = (
      InterfaceScorer.load(Path(str(args.interface_scorer_model)))
      if args.interface_scorer_model
      else None
  )
  output_path = Path(args.output_jsonl)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  rng = random.Random(int(args.seed))

  rows_written = 0
  positive_count = 0
  negative_count = 0
  skipped_cases = 0
  skipped_parts = 0
  skipped_no_positive_pairs = 0
  case_summaries: list[dict[str, Any]] = []
  with output_path.open("w", encoding="utf-8") as f:
    for case in cases:
      case_id = str(case.get("id") or "")
      assembly_json = _assembly_json_for_case(case, dataset_roots)
      if assembly_json is None or not assembly_json.exists():
        skipped_cases += 1
        continue
      part_records = _load_case_part_interfaces(
          case=case,
          dataset_roots=dataset_roots,
          file_index=file_index,
          assembly_json=assembly_json,
          scorer=scorer,
          max_candidates_per_part=int(args.max_candidates_per_part),
      )
      skipped_parts += sum(1 for item in part_records.values() if item.get("error"))
      part_records = {
          key: value for key, value in part_records.items() if not value.get("error")
      }
      if len(part_records) < 2:
        skipped_cases += 1
        continue
      contact_face_pairs = _contact_face_pairs_by_body(assembly_json)
      generated = _generate_pair_rows_for_case(
          case=case,
          part_records=part_records,
          contact_face_pairs=contact_face_pairs,
          top_candidates_per_part_pair=int(args.top_candidates_per_part_pair),
          negative_ratio=int(args.negative_ratio),
          rng=rng,
      )
      if not any(int(row["label"]) == 1 for row in generated):
        skipped_no_positive_pairs += 1
        continue
      case_pos = 0
      case_neg = 0
      for row in generated:
        repeats = _row_repeat_count(
            row,
            slot_positive_oversample=int(args.slot_positive_oversample),
        )
        for _ in range(repeats):
          f.write(json.dumps(row, ensure_ascii=False) + "\n")
          rows_written += 1
          if int(row["label"]) == 1:
            positive_count += 1
            case_pos += 1
          else:
            negative_count += 1
            case_neg += 1
      case_summaries.append(
          {
              "case_id": case_id,
              "assembly_dir": case.get("assembly_dir"),
              "rows": len(generated),
              "positives": case_pos,
              "negatives": case_neg,
          }
      )

  summary = {
      "cases_seen": len(cases),
      "cases_with_rows": len(case_summaries),
      "skipped_cases": skipped_cases,
      "skipped_parts": skipped_parts,
      "skipped_no_positive_pairs": skipped_no_positive_pairs,
      "rows_written": rows_written,
      "positive_count": positive_count,
      "negative_count": negative_count,
      "positive_rate": round(positive_count / max(1, rows_written), 4),
      "negative_ratio": int(args.negative_ratio),
      "slot_positive_oversample": int(args.slot_positive_oversample),
      "output_jsonl": str(output_path.resolve()),
  }
  summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
  summary_path.write_text(
      json.dumps({"summary": summary, "cases": case_summaries}, indent=2),
      encoding="utf-8",
  )
  print(json.dumps(summary, indent=2))


def _benchmark_v2_endpoint(record: BenchmarkV2InterfaceRecord) -> dict[str, Any]:
  score = record.candidate.score
  return {
      "model_input_protocol": "benchmark_v2",
      "benchmark_v2_model_view": record.model_view.model_view,
      "benchmark_v2_model_view_sha256": record.model_view.sha256,
      "score": 0.0 if score is None else float(score),
  }


def _record_face_indices(record: BenchmarkV2InterfaceRecord) -> frozenset[int]:
  result: set[int] = set()
  candidate = record.candidate
  if candidate.face_index is not None:
    result.add(int(candidate.face_index))
  members = candidate.metadata.get("member_face_indices")
  if isinstance(members, list):
    result.update(int(value) for value in members if isinstance(value, int))
  return frozenset(result)


def _benchmark_v2_pair_label(
    record_a: BenchmarkV2InterfaceRecord,
    record_b: BenchmarkV2InterfaceRecord,
    contact_group_pairs: frozenset[
        tuple[str, frozenset[int], str, frozenset[int]]
    ],
) -> int:
  faces_a = _record_face_indices(record_a)
  faces_b = _record_face_indices(record_b)
  return int(
      any(
          body_a == record_a.opaque_body
          and body_b == record_b.opaque_body
          and required_a <= faces_a
          and required_b <= faces_b
          for body_a, required_a, body_b, required_b in contact_group_pairs
      )
  )


def _intrinsic_contact_type(
    record_a: BenchmarkV2InterfaceRecord,
    record_b: BenchmarkV2InterfaceRecord,
) -> str:
  view_a = record_a.model_view.model_view
  view_b = record_b.model_view.model_view
  topology_a = view_a.get("topology")
  topology_b = view_b.get("topology")
  if not isinstance(topology_a, Mapping):
    topology_a = {}
  if not isinstance(topology_b, Mapping):
    topology_b = {}
  roles = {
      str(topology_a.get("composite_role") or ""),
      str(topology_b.get("composite_role") or ""),
  }
  roles.discard("")
  surfaces = {str(view_a.get("surface_type")), str(view_b.get("surface_type"))}
  helical = bool(
      topology_a.get("has_helical_edge") or topology_b.get("has_helical_edge")
  )
  if helical:
    return "screw_in_hole"
  if "obround_slot" in roles and "pin_boss" in roles:
    return "boss_in_slot"
  if "obround_slot" in roles:
    return "pin_in_slot"
  if "center_bore" in roles and roles & {"shaft_axis", "pin_boss"}:
    return "shaft_in_bore"
  if surfaces == {"plane"}:
    return "seat_plane"
  if surfaces <= {"cylinder"}:
    return "insert_axis"
  return "generic_contact"


def _select_v2_records(
    records: Sequence[BenchmarkV2InterfaceRecord],
    *,
    contacted_face_groups: set[frozenset[int]],
    limit: int,
) -> list[BenchmarkV2InterfaceRecord]:
  ranked = sorted(
      records,
      key=lambda record: (
          -(float(record.candidate.score) if record.candidate.score is not None else 0.0),
          record.candidate.interface_id,
      ),
  )
  if limit <= 0 or len(ranked) <= limit:
    return ranked
  positives = [
      record
      for record in ranked
      if any(
          required <= _record_face_indices(record)
          for required in contacted_face_groups
      )
  ]
  selected = list(positives)
  for record in ranked:
    if record in selected:
      continue
    selected.append(record)
    if len(selected) >= max(limit, len(positives)):
      break
  return selected


def produce_benchmark_v2_pair_rows(
    produced: BenchmarkV2ProducedInterfaceCase,
    *,
    negative_ratio: int,
    top_candidates_per_part_pair: int,
    seed: int,
    interface_scorer_artifact_sha256: str | None = None,
) -> list[dict[str, Any]]:
  """Create pair rows without interpreting unlisted source contacts as negatives."""

  if not produced.rows:
    return []
  if int(negative_ratio) < 1:
    raise ValueError("benchmark_v2 pair negative_ratio must be at least 1")
  if int(top_candidates_per_part_pair) < 0:
    raise ValueError("top_candidates_per_part_pair must be nonnegative")
  has_upstream_scores = any(
      record.candidate.score is not None
      for records in produced.records_by_body.values()
      for record in records
  )
  if has_upstream_scores and not interface_scorer_artifact_sha256:
    raise ValueError(
        "benchmark_v2 pair rows with learned interface scores require the "
        "interface scorer artifact SHA-256"
    )
  if interface_scorer_artifact_sha256 is not None and (
      len(interface_scorer_artifact_sha256) != 64
      or any(
          character not in "0123456789abcdef"
          for character in interface_scorer_artifact_sha256
      )
  ):
    raise ValueError("interface scorer artifact SHA-256 is malformed")
  if not math.isfinite(float(seed)):
    raise ValueError("benchmark_v2 pair seed must be finite")
  first = produced.rows[0]
  label_policy = str(produced.producer_config.get("label_policy") or "")
  if label_policy not in {
      "source_positive_only_unknown_excluded",
      "closed_world_certified_binary",
  }:
    raise ValueError("benchmark_v2 pair producer lacks an explicit label policy")
  split = str(first["source_split"])
  family_sha = str(first["family_split_manifest_sha256"])
  case_token = str(first["case_token_sha256"])
  config = {
      "frame_randomization_seed": int(produced.seed),
      "interface_producer_config": dict(produced.producer_config),
      "negative_sampling_seed": int(seed),
      "negative_ratio": int(negative_ratio),
      "top_candidates_per_part_pair": int(top_candidates_per_part_pair),
      "interface_scorer_artifact_sha256": interface_scorer_artifact_sha256,
      "label_policy": label_policy,
  }
  config_sha = canonical_sha256(config)
  feature_sha = canonical_sha256(list(BENCHMARK_V2_PAIR_FEATURE_NAMES))
  rng = random.Random(int(seed))
  bodies = sorted(produced.records_by_body)
  rows: list[dict[str, Any]] = []
  for body_index, body_a in enumerate(bodies):
    for body_b in bodies[body_index + 1 :]:
      directed_contacts = {
          (group_a, group_b)
          for contact_body_a, group_a, contact_body_b, group_b in (
              produced.contact_face_group_pairs
          )
          if contact_body_a == body_a and contact_body_b == body_b
      }
      if not directed_contacts:
        continue
      contacted_a = {first_group for first_group, _ in directed_contacts}
      contacted_b = {second_group for _, second_group in directed_contacts}
      records_a = _select_v2_records(
          produced.records_by_body[body_a],
          contacted_face_groups=contacted_a,
          limit=int(top_candidates_per_part_pair),
      )
      records_b = _select_v2_records(
          produced.records_by_body[body_b],
          contacted_face_groups=contacted_b,
          limit=int(top_candidates_per_part_pair),
      )
      positives: list[tuple[float, dict[str, Any]]] = []
      negatives: list[tuple[float, dict[str, Any]]] = []
      for record_a in records_a:
        for record_b in records_b:
          label = _benchmark_v2_pair_label(
              record_a,
              record_b,
              produced.contact_face_group_pairs,
          )
          endpoint_a = _benchmark_v2_endpoint(record_a)
          endpoint_b = _benchmark_v2_endpoint(record_b)
          pair_features = pair_features_from_rows(
              endpoint_a,
              endpoint_b,
              relation_hint="link",
              protocol="benchmark_v2",
          )
          prior = float(pair_features["heuristic_prior"])
          opaque_pair = "opaque_pair_" + hashlib.sha256(
              (
                  f"{case_token}\0{record_a.candidate.interface_id}\0"
                  f"{record_b.candidate.interface_id}"
              ).encode("utf-8")
          ).hexdigest()[:24]
          row = {
              "schema_version": INTERFACE_PAIR_TRAINING_ROW_SCHEMA,
              "model_input_protocol": "benchmark_v2",
              "protocol_version": PROTOCOL_VERSION,
              "source_split": split,
              "family_split_manifest_sha256": family_sha,
              "feature_names_sha256": feature_sha,
              "producer_config_sha256": config_sha,
              "case_token_sha256": case_token,
              "opaque_pair": opaque_pair,
              "opaque_interface_a": record_a.candidate.interface_id,
              "opaque_interface_b": record_b.candidate.interface_id,
              "interface_a": endpoint_a,
              "interface_b": endpoint_b,
              "relation_hint": "link",
              "contact_type": (
                  _intrinsic_contact_type(record_a, record_b) if label else "none"
              ),
              "label": label,
          }
          if label == 0 and label_policy == (
              "source_positive_only_unknown_excluded"
          ):
            continue
          if set(row) != INTERFACE_PAIR_TRAINING_ROW_KEYS:
            raise AssertionError("internal benchmark_v2 pair row schema drift")
          (positives if label else negatives).append((prior, row))
      if not positives:
        continue
      negatives.sort(
          key=lambda value: (
              -value[0],
              str(value[1]["opaque_interface_a"]),
              str(value[1]["opaque_interface_b"]),
          )
      )
      quota = max(1, int(negative_ratio)) * len(positives)
      hard_quota = min(len(negatives), int(quota * 0.75))
      selected_negatives = negatives[:hard_quota]
      remainder = negatives[hard_quota:]
      rng.shuffle(remainder)
      selected_negatives.extend(remainder[: max(0, quota - hard_quota)])
      rows.extend(row for _prior, row in positives)
      rows.extend(row for _prior, row in selected_negatives)
  rows.sort(
      key=lambda row: (
          -int(row["label"]),
          str(row["opaque_pair"]),
      )
  )
  return rows


def write_benchmark_v2_pair_dataset(
    output_path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    source_split: str,
    family_split_manifest: str | Path,
    seed: int,
    producer_config: Mapping[str, Any],
    authenticated_receipts: BenchmarkV2AuthenticatedTrainingReceipts | None = None,
) -> dict[str, Any]:
  require_formal_binary_training_available()
  if not isinstance(
      authenticated_receipts,
      BenchmarkV2AuthenticatedTrainingReceipts,
  ):
    raise ValueError(
        "formal benchmark_v2 pair writing requires a path-authenticated receipt "
        "set; receipt Mappings are provenance only"
    )
  interface_config = (
      producer_config.get("interface_producer_config")
      if isinstance(producer_config, Mapping)
      else None
  )
  _validated_face_map_binding(
      (
          interface_config.get("fusion_step_face_map_binding")
          if isinstance(interface_config, Mapping)
          else None
      ),
      require_private_receipt=True,
  )
  validate_private_supervision_binding(
      (
          interface_config.get("private_supervision_binding")
          if isinstance(interface_config, Mapping)
          else None
      ),
      require_certified_binary=True,
  )
  path = Path(output_path)
  return write_dataset_artifact(
      path,
      rows=rows,
      row_schema=INTERFACE_PAIR_TRAINING_ROW_SCHEMA,
      feature_names=BENCHMARK_V2_PAIR_FEATURE_NAMES,
      source_split=source_split,
      family_split_manifest=family_split_manifest,
      seed=seed,
      producer_config=producer_config,
      producer_kind="pair",
      authenticated_receipts=authenticated_receipts,
  )


def _main_benchmark_v2(args: argparse.Namespace) -> None:
  if not args.family_split_manifest:
    raise SystemExit("benchmark_v2 requires --family_split_manifest")
  if not getattr(args, "private_source_split", None):
    raise SystemExit("benchmark_v2 requires --private-source-split")
  if not getattr(args, "private_evaluation_gold_split", None):
    raise SystemExit("benchmark_v2 requires --private-evaluation-gold-split")
  if not args.fusion_step_face_map_receipt:
    raise SystemExit("benchmark_v2 requires --fusion-step-face-map-receipt")
  if int(args.max_cases) > 0:
    raise ValueError(
        "formal benchmark_v2 production must consume the complete frozen split; "
        "--max_cases is legacy smoke-test only"
    )
  dataset_roots = _parse_dataset_roots(
      dataset_root=str(args.dataset_root),
      dataset_roots=list(args.dataset_roots or []),
  )
  authenticated_inputs = load_benchmark_v2_authenticated_training_inputs(
      public_cases_path=Path(args.cases_json),
      private_source_path=Path(args.private_source_split),
      private_gold_path=Path(args.private_evaluation_gold_split),
      family_split_manifest_path=Path(args.family_split_manifest),
      face_map_receipt_path=Path(args.fusion_step_face_map_receipt),
      source_split=str(args.source_split),
  )
  cases = [dict(case) for case in authenticated_inputs.public_cases]
  case_ids = list(authenticated_inputs.case_ids)
  family_sha = authenticated_inputs.receipts.family_split_manifest_sha256
  private_supervision = authenticated_inputs.private_supervision
  face_map_lookup = authenticated_inputs.face_map_lookup
  face_map_binding = authenticated_inputs.face_map_binding
  for case_id in case_ids:
    require_certified_binary_training_gold(
        private_supervision.gold_case(case_id)
    )
  file_index = _index_step_files(dataset_roots)
  scorer_artifact = (
      load_benchmark_v2_interface_scorer_artifact(
          Path(args.interface_scorer_model)
      )
      if args.interface_scorer_model
      else None
  )
  scorer = None if scorer_artifact is None else scorer_artifact.scorer
  scorer_sha256 = None if scorer_artifact is None else scorer_artifact.sha256
  rows: list[dict[str, Any]] = []
  for case in cases:
    case_id = str(case.get("id") or "")
    private_source_case = private_supervision.source_case(case_id)
    private_gold_case = private_supervision.gold_case(case_id)
    with _open_verified_benchmark_v2_case_inputs(
        private_source_case,
        dataset_roots=dataset_roots,
        file_index=file_index,
    ) as (_assembly_data, part_specs, live_inputs):
      produced = produce_benchmark_v2_interface_case(
          part_specs=part_specs,
          evaluation_gold_contacts=private_gold_case,
          case_id=case_id,
          seed=int(args.frame_randomization_seed),
          source_split=str(args.source_split),
          family_split_manifest_sha256=family_sha,
          # Formal pair mining labels all interfaces before applying the
          # pair-level top-candidate budget, so positives cannot be truncated.
          max_candidates_per_part=0,
          translation_box_fraction=float(
              args.benchmark_v2_translation_box_fraction
          ),
          interface_scorer=scorer,
          fusion_step_face_lookup=face_map_lookup,
          fusion_step_face_map_binding=face_map_binding,
          private_supervision_binding=private_supervision.producer_binding,
      )
      _reverify_benchmark_v2_case_inputs(live_inputs)
      case_rows = produce_benchmark_v2_pair_rows(
          produced,
          negative_ratio=int(args.negative_ratio),
          top_candidates_per_part_pair=int(
              args.top_candidates_per_part_pair
          ),
          seed=int(args.seed),
          interface_scorer_artifact_sha256=scorer_sha256,
      )
      if not case_rows or not any(int(row["label"]) == 1 for row in case_rows):
        raise ValueError(
            f"benchmark_v2 case {case.get('id')!r} has no contact-grounded pair rows"
        )
      rows.extend(case_rows)
  if scorer_artifact is not None:
    scorer_artifact.reverify()
  config = {
      "frame_randomization_seed": int(args.frame_randomization_seed),
      "interface_producer_config": {
          "max_candidates_per_part": 0,
          "translation_box_fraction": float(
              args.benchmark_v2_translation_box_fraction
          ),
          "fusion_step_face_map_binding": face_map_binding,
          "private_supervision_binding": dict(
              private_supervision.producer_binding
          ),
          "label_policy": "closed_world_certified_binary",
      },
      "negative_sampling_seed": int(args.seed),
      "negative_ratio": int(args.negative_ratio),
      "top_candidates_per_part_pair": int(args.top_candidates_per_part_pair),
      "interface_scorer_artifact_sha256": scorer_sha256,
      "label_policy": "closed_world_certified_binary",
  }
  manifest = write_benchmark_v2_pair_dataset(
      Path(args.output_jsonl),
      rows=rows,
      source_split=str(args.source_split),
      family_split_manifest=Path(args.family_split_manifest),
      seed=int(args.frame_randomization_seed),
      producer_config=config,
      authenticated_receipts=authenticated_inputs.receipts,
  )
  summary = {
      "model_input_protocol": "benchmark_v2",
      "protocol_version": PROTOCOL_VERSION,
      "source_split": str(args.source_split),
      "cases_seen": len(cases),
      "rows_written": len(rows),
      "positive_count": sum(int(row["label"] == 1) for row in rows),
      "negative_count": sum(int(row["label"] == 0) for row in rows),
      "dataset_manifest_sha256": manifest["manifest_sha256"],
  }
  summary_path = (
      Path(args.summary_json)
      if args.summary_json
      else Path(args.output_jsonl).with_suffix(".summary.json")
  )
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


def _load_case_part_interfaces(
    *,
    case: dict[str, Any],
    dataset_roots: list[Path],
    file_index: dict[str, Path],
    assembly_json: Path,
    scorer: InterfaceScorer | None,
    max_candidates_per_part: int,
) -> dict[str, dict[str, Any]]:
  parts = case.get("parts")
  if not isinstance(parts, list):
    return {}
  part_names = case.get("selected_part_names")
  if not isinstance(part_names, list):
    part_names = []
  body_uuids = case.get("selected_body_uuids")
  if not isinstance(body_uuids, list):
    body_uuids = []
  assembly_dir = case.get("assembly_dir") if isinstance(case.get("assembly_dir"), str) else None
  dataset_root_hint = (
      case.get("dataset_root_hint") if isinstance(case.get("dataset_root_hint"), str) else None
  )
  records: dict[str, dict[str, Any]] = {}
  for idx, token in enumerate(parts):
    part_name = ""
    body_uuid = ""
    try:
      step_path = _resolve_part_path(
          part_token=str(token),
          dataset_roots=dataset_roots,
          assembly_dir=assembly_dir,
          dataset_root_hint=dataset_root_hint,
          file_index=file_index,
      )
      part_name = (
          str(part_names[idx]).strip()
          if idx < len(part_names) and str(part_names[idx]).strip()
          else step_path.stem
      )
      body_uuid = (
          str(body_uuids[idx]).strip()
          if idx < len(body_uuids) and str(body_uuids[idx]).strip()
          else step_path.stem
      )
      candidates = extract_candidate_interfaces_from_step(
          step_path,
          part_name=part_name,
          body_uuid=body_uuid,
          max_candidates=max_candidates_per_part,
      )
      candidates = label_interfaces_from_assembly_contacts(candidates, assembly_json)
      if scorer is not None:
        scorer.score_candidates(candidates)
      records[body_uuid] = {
          "part_name": part_name,
          "body_uuid": body_uuid,
          "step_path": str(step_path.resolve()),
          "candidates": candidates,
      }
    except Exception as exc:
      if body_uuid:
        records[body_uuid] = {
            "part_name": part_name,
            "body_uuid": body_uuid,
            "error": str(exc),
          }
  return records


def _generate_pair_rows_for_case(
    *,
    case: dict[str, Any],
    part_records: dict[str, dict[str, Any]],
    contact_face_pairs: set[tuple[str, int, str, int]],
    top_candidates_per_part_pair: int,
    negative_ratio: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
  contact_pairs = case.get("contact_pairs")
  selected_part_names = case.get("selected_part_names")
  selected_body_uuids = case.get("selected_body_uuids")
  if not isinstance(contact_pairs, list):
    contact_pairs = []
  if not isinstance(selected_part_names, list):
    selected_part_names = []
  if not isinstance(selected_body_uuids, list):
    selected_body_uuids = []
  name_to_body = {
      str(name): str(body)
      for name, body in zip(selected_part_names, selected_body_uuids)
      if str(name).strip() and str(body).strip()
  }

  rows: list[dict[str, Any]] = []
  seen_keys: set[tuple[str, str, str, str, int]] = set()
  for pair in contact_pairs:
    if not isinstance(pair, list) or len(pair) != 2:
      continue
    part_a_name = str(pair[0])
    part_b_name = str(pair[1])
    body_a = name_to_body.get(part_a_name)
    body_b = name_to_body.get(part_b_name)
    if not body_a or not body_b:
      continue
    record_a = part_records.get(body_a)
    record_b = part_records.get(body_b)
    if record_a is None or record_b is None:
      continue
    candidates_a = _rank_candidates(record_a.get("candidates", []))
    candidates_b = _rank_candidates(record_b.get("candidates", []))
    if top_candidates_per_part_pair > 0:
      candidates_a = candidates_a[: int(top_candidates_per_part_pair)]
      candidates_b = candidates_b[: int(top_candidates_per_part_pair)]
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for candidate_a in candidates_a:
      for candidate_b in candidates_b:
        label = _pair_label(
            body_a=body_a,
            body_b=body_b,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            contact_face_pairs=contact_face_pairs,
        )
        row = _row_for_candidate_pair(
            case=case,
            part_a_name=part_a_name,
            part_b_name=part_b_name,
            body_a=body_a,
            body_b=body_b,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            label=label,
        )
        key = (
            body_a,
            body_b,
            str(candidate_a.interface_id),
            str(candidate_b.interface_id),
            int(label),
        )
        if key in seen_keys:
          continue
        seen_keys.add(key)
        if label == 1:
          positives.append(row)
        else:
          negatives.append(row)
    if not positives:
      continue
    negatives.sort(
        key=lambda item: (
            -float(item["features"].get("heuristic_prior", 0.0)),
            item["interface_a"]["interface_id"],
            item["interface_b"]["interface_id"],
        )
    )
    max_negatives = max(1, int(negative_ratio)) * len(positives)
    hard_quota = int(max_negatives * 0.75)
    selected_negatives = negatives[:hard_quota]
    remaining = negatives[hard_quota:]
    rng.shuffle(remaining)
    selected_negatives.extend(remaining[: max(0, max_negatives - len(selected_negatives))])
    rows.extend(positives)
    rows.extend(selected_negatives)
  return rows


def _rank_candidates(candidates: Any) -> list[CandidateInterface]:
  if not isinstance(candidates, list):
    return []
  result = [item for item in candidates if isinstance(item, CandidateInterface)]
  result.sort(
      key=lambda item: (
          -(item.score if isinstance(item.score, (int, float)) else 0.0),
          -float(item.features.get("sqrt_area_ratio", 0.0)),
          item.interface_id,
      )
  )
  return result


def _pair_label(
    *,
    body_a: str,
    body_b: str,
    candidate_a: CandidateInterface,
    candidate_b: CandidateInterface,
    contact_face_pairs: set[tuple[str, int, str, int]],
) -> int:
  indices_a = _candidate_contact_face_indices(candidate_a)
  indices_b = _candidate_contact_face_indices(candidate_b)
  if not indices_a or not indices_b:
    return 0
  for idx_a in indices_a:
    for idx_b in indices_b:
      key = (body_a, int(idx_a), body_b, int(idx_b))
      rev = (body_b, int(idx_b), body_a, int(idx_a))
      if key in contact_face_pairs or rev in contact_face_pairs:
        return 1
  return 0


def _candidate_contact_face_indices(candidate: CandidateInterface) -> list[int]:
  indices: list[int] = []
  if candidate.face_index is not None:
    try:
      indices.append(int(candidate.face_index))
    except (TypeError, ValueError):
      pass
  raw_members = candidate.metadata.get("member_face_indices")
  if isinstance(raw_members, list):
    for item in raw_members:
      try:
        value = int(item)
      except (TypeError, ValueError):
        continue
      if value not in indices:
        indices.append(value)
  return indices


def _row_for_candidate_pair(
    *,
    case: dict[str, Any],
    part_a_name: str,
    part_b_name: str,
    body_a: str,
    body_b: str,
    candidate_a: CandidateInterface,
    candidate_b: CandidateInterface,
    label: int,
) -> dict[str, Any]:
  row_a = candidate_a.to_dict()
  row_b = candidate_b.to_dict()
  relation_hint = _relation_hint_from_roles(row_a, row_b)
  contact_type = (
      _contact_type_from_pair(
          part_a_name=part_a_name,
          part_b_name=part_b_name,
          row_a=row_a,
          row_b=row_b,
      )
      if int(label) == 1
      else "none"
  )
  features = pair_features_from_rows(row_a, row_b, relation_hint=relation_hint)
  # Recompute the prior after relation-specific features are filled.
  features["heuristic_prior"] = heuristic_pair_prior(features)
  return {
      "case_id": str(case.get("id") or ""),
      "assembly_dir": case.get("assembly_dir"),
      "part_a": part_a_name,
      "part_b": part_b_name,
      "body_uuid_a": body_a,
      "body_uuid_b": body_b,
      "interface_a": _compact_interface(row_a),
      "interface_b": _compact_interface(row_b),
      "relation_hint": relation_hint,
      "contact_type": contact_type,
      "features": features,
      "label": int(label),
      "label_source": "positive_contact_face_pair" if label else "hard_negative_non_contact_pair",
  }


def _compact_interface(row: dict[str, Any]) -> dict[str, Any]:
  return {
      "interface_id": row.get("interface_id"),
      "face_index": row.get("face_index"),
      "surface_type": row.get("surface_type"),
      "role_hint": row.get("role_hint"),
      "score": row.get("score"),
      "label": row.get("label"),
      "metadata": {
          "radius": (row.get("metadata") or {}).get("radius")
          if isinstance(row.get("metadata"), dict)
          else None,
          "area": (row.get("metadata") or {}).get("area")
          if isinstance(row.get("metadata"), dict)
          else None,
      },
  }


def _row_repeat_count(row: dict[str, Any], *, slot_positive_oversample: int) -> int:
  if int(row.get("label", 0)) != 1:
    return 1
  factor = max(1, int(slot_positive_oversample))
  if factor <= 1:
    return 1
  role_a = str((row.get("interface_a") or {}).get("role_hint") or "").lower()
  role_b = str((row.get("interface_b") or {}).get("role_hint") or "").lower()
  if "obround_slot" in {role_a, role_b}:
    return factor
  return 1


def _relation_hint_from_roles(row_a: dict[str, Any], row_b: dict[str, Any]) -> str:
  roles = {str(row_a.get("role_hint") or ""), str(row_b.get("role_hint") or "")}
  if "obround_slot" in roles and roles & {"pin_boss", "shaft_axis", "cylindrical_interface"}:
    return "insert"
  if roles & {"hole_entry", "center_bore", "threaded_hole"} and roles & {
      "shaft_axis",
      "cylindrical_interface",
      "pin_boss",
  }:
    return "insert"
  if roles <= {"planar_seat", "shoulder_stop"}:
    return "support"
  return "link"


def _contact_type_from_pair(
    *,
    part_a_name: str,
    part_b_name: str,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
) -> str:
  role_a = str(row_a.get("role_hint") or "")
  role_b = str(row_b.get("role_hint") or "")
  surface_a = str(row_a.get("surface_type") or "")
  surface_b = str(row_b.get("surface_type") or "")
  roles = {role_a, role_b}
  surfaces = {surface_a, surface_b}
  names = f"{part_a_name} {part_b_name}".lower()
  threaded_name = any(
      token in names
      for token in (
          "screw",
          "bolt",
          "thread",
          "fastener",
          "nut",
          "stud",
      )
  )
  features_a = row_a.get("features") if isinstance(row_a.get("features"), dict) else {}
  features_b = row_b.get("features") if isinstance(row_b.get("features"), dict) else {}
  helical = bool(features_a.get("has_helical_edge") or features_b.get("has_helical_edge"))
  if threaded_name or helical:
    if roles & {"threaded_hole", "center_bore", "hole_entry"} and roles & {
        "shaft_axis",
        "cylindrical_interface",
        "pin_boss",
    }:
      return "screw_in_hole"
    return "threaded_interference"
  if roles <= {"planar_seat", "shoulder_stop"} or surfaces <= {"plane"}:
    return "seat_plane"
  if "obround_slot" in roles and "pin_boss" in roles:
    return "boss_in_slot"
  if "obround_slot" in roles and roles & {"shaft_axis", "cylindrical_interface"}:
    return "pin_in_slot"
  if "center_bore" in roles and roles & {"shaft_axis", "cylindrical_interface", "pin_boss"}:
    return "shaft_in_bore"
  if roles & {"hole_entry", "threaded_hole"} and roles & {
      "shaft_axis",
      "cylindrical_interface",
      "pin_boss",
  }:
    return "insert_axis"
  if surfaces <= {"cylinder"} or roles <= {
      "shaft_axis",
      "hole_entry",
      "center_bore",
      "threaded_hole",
      "pin_boss",
      "obround_slot",
      "cylindrical_interface",
  }:
    return "insert_axis"
  return "generic_contact"


def _contact_face_pairs_by_body(assembly_json: str | Path) -> set[tuple[str, int, str, int]]:
  data = json.loads(Path(assembly_json).read_text(encoding="utf-8"))
  contacts = data.get("contacts")
  result: set[tuple[str, int, str, int]] = set()
  if not isinstance(contacts, list):
    return result
  for contact in contacts:
    if not isinstance(contact, dict):
      continue
    e1 = contact.get("entity_one")
    e2 = contact.get("entity_two")
    if not isinstance(e1, dict) or not isinstance(e2, dict):
      continue
    body_a = e1.get("body")
    body_b = e2.get("body")
    idx_a = e1.get("index")
    idx_b = e2.get("index")
    if (
        isinstance(body_a, str)
        and isinstance(body_b, str)
        and isinstance(idx_a, int)
        and isinstance(idx_b, int)
    ):
      result.add((body_a, int(idx_a), body_b, int(idx_b)))
      result.add((body_b, int(idx_b), body_a, int(idx_a)))
  return result


def _assembly_json_for_case(
    case: dict[str, Any],
    dataset_roots: list[Path],
) -> Optional[Path]:
  assembly_dir = case.get("assembly_dir")
  if not isinstance(assembly_dir, str) or not assembly_dir.strip():
    return None
  dataset_root_hint = case.get("dataset_root_hint")
  if isinstance(dataset_root_hint, str) and dataset_root_hint.strip():
    candidate = Path(dataset_root_hint) / assembly_dir / "assembly.json"
    if candidate.exists():
      return candidate
  for root in dataset_roots:
    candidate = root / assembly_dir / "assembly.json"
    if candidate.exists():
      return candidate
  return None


if __name__ == "__main__":
  main()
