"""Geometry split-hygiene audit for NeuroCAD benchmarks.

This script is intentionally paper-facing rather than search-facing.  It checks
whether train/dev/test splits share exact or near-exact geometric evidence that
could make mate-program retrieval look better than it is:

* STEP file SHA1 overlap,
* rigid-transform-invariant intrinsic B-Rep signatures,
* case-level part-graph fingerprints,
* optional mined mate residual near-duplicate overlap.

The audit does not modify splits.  It produces a JSON summary and, optionally, a
small Markdown table that can go directly into the paper appendix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from .batch_infer_assemble import (
    _index_step_files,
    _parse_dataset_roots,
    _resolve_part_path,
)
from .cadquery_backend import load_step_shape
from .family_split_builder import (
    FamilySplitConfig,
    _code_identity,
    _family_source_schema_identity,
    _identity_for_payload,
    audit_family_splits,
    build_family_splits,
)
from .mate_programs import MateProgram, rotation_angle_degrees
from .mate_pose_retriever import load_mate_programs
from .paths import resolve_dataset_roots, resolve_path


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_root", default=None)
  parser.add_argument("--dataset_roots", nargs="*", default=[])
  parser.add_argument(
      "--splits",
      nargs="+",
      required=True,
      help="Split specs like train=path/to/train_oracle.json dev=... test=...",
  )
  parser.add_argument(
      "--mate_program_splits",
      nargs="*",
      default=[],
      help=(
          "Optional mate-program specs like train=mate_train.jsonl "
          "dev=mate_dev.jsonl test=mate_test.jsonl for residual overlap audit."
      ),
  )
  parser.add_argument("--max_cases_per_split", type=int, default=0)
  parser.set_defaults(formal=True)
  parser.add_argument(
      "--formal",
      dest="formal",
      action="store_true",
      help="Run the full, unsampled, manifest-bound paper-facing audit (default).",
  )
  parser.add_argument(
      "--development_nonformal",
      "--development-nonformal",
      dest="formal",
      action="store_false",
      help="Allow exploratory sampling and omit formal manifest requirements.",
  )
  parser.add_argument("--split_manifest", "--split-manifest", default="")
  parser.add_argument(
      "--family_source_cases",
      "--family-source-cases",
      default="",
      help="Optional local override for a moved authoritative input_cases artifact.",
  )
  parser.add_argument(
      "--family_pair_review",
      "--family-pair-review",
      default="",
  )
  parser.add_argument(
      "--review_verification_key_file",
      "--review-verification-key-file",
      default="",
      help="Trusted RSA public-key JSON for the independent family-pair review.",
  )
  parser.add_argument(
      "--required_evidence_coverage",
      "--required-evidence-coverage",
      type=float,
      default=1.0,
  )
  parser.add_argument("--interface_max_candidates", type=int, default=0)
  parser.add_argument(
      "--skip_intrinsic_descriptors",
      "--skip_interface_histograms",
      dest="skip_intrinsic_descriptors",
      action="store_true",
      help="Skip intrinsic B-Rep descriptors and run exact STEP-hash checks only.",
  )
  # Accepted for command-line compatibility only; pose-dependent descriptors
  # are intentionally absent from benchmark-v2 hygiene.
  parser.add_argument("--bbox_round_mm", type=float, default=0.05)
  parser.add_argument("--radius_round_mm", type=float, default=0.05)
  parser.add_argument("--residual_translation_round_mm", type=float, default=0.5)
  parser.add_argument("--residual_rotation_round_deg", type=float, default=5.0)
  parser.add_argument("--max_examples", type=int, default=8)
  parser.add_argument(
      "--nearest_neighbor_top_k",
      type=int,
      default=20,
      help="Number of cross-split approximate family neighbors to retain.",
  )
  parser.add_argument(
      "--neighbor_warning_threshold",
      type=float,
      default=0.9,
      help="Similarity at or above this value requires family review.",
  )
  parser.add_argument("--output_json", default="")
  parser.add_argument("--output_md", default="")
  return parser


def _normalize_named_path_specs(items: Iterable[str]) -> list[str]:
  normalized: list[str] = []
  for item in items or []:
    text = str(item).strip()
    if "=" not in text:
      raise ValueError(f"Expected name=path spec, got: {text}")
    name, raw_path = text.split("=", 1)
    normalized.append(f"{name.strip()}={resolve_path(raw_path.strip())}")
  return normalized


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
  """Anchor split, model, and output paths at the checkout root."""
  roots = resolve_dataset_roots(args.dataset_root, args.dataset_roots)
  args.dataset_root = str(roots[0])
  args.dataset_roots = [str(path) for path in roots[1:]]
  args.splits = _normalize_named_path_specs(args.splits)
  args.mate_program_splits = _normalize_named_path_specs(args.mate_program_splits)
  if args.output_json:
    args.output_json = str(resolve_path(args.output_json))
  if args.output_md:
    args.output_md = str(resolve_path(args.output_md))
  for attribute in (
      "split_manifest",
      "family_source_cases",
      "family_pair_review",
      "review_verification_key_file",
  ):
    value = str(getattr(args, attribute, "") or "").strip()
    if value:
      setattr(args, attribute, str(resolve_path(value)))
  return args


def main() -> None:
  args = normalize_paths(build_parser().parse_args())
  summary = audit(args)
  if args.output_json:
    path = Path(str(args.output_json))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  if args.output_md:
    path = Path(str(args.output_md))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown(summary), encoding="utf-8")
  print(json.dumps(summary, indent=2, sort_keys=True))


def _load_cases_strict(path: Path) -> list[dict[str, Any]]:
  """Load every row; formal hygiene must never filter malformed inputs."""

  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except FileNotFoundError as exc:
    raise ValueError(f"Split file not found: {path}") from exc
  except json.JSONDecodeError as exc:
    raise ValueError(f"Split JSON is invalid: {path}: {exc}") from exc
  if isinstance(payload, list):
    raw_cases = payload
  elif isinstance(payload, dict):
    raw_cases = payload.get("cases")
  else:
    raw_cases = None
  if not isinstance(raw_cases, list):
    raise ValueError(f"Split JSON must be a list or object with cases list: {path}")
  cases: list[dict[str, Any]] = []
  seen_ids: set[str] = set()
  for index, row in enumerate(raw_cases):
    if not isinstance(row, dict):
      raise ValueError(f"Split {path} row {index} is not an object")
    case_id = str(row.get("id") or "").strip()
    if not case_id:
      raise ValueError(f"Split {path} row {index} is missing a nonempty id")
    if case_id in seen_ids:
      raise ValueError(f"Split {path} contains duplicate case id: {case_id}")
    seen_ids.add(case_id)
    cases.append(row)
  if not cases:
    raise ValueError(f"Split {path} is empty")
  return cases


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except FileNotFoundError as exc:
    raise ValueError(f"{label} not found: {path}") from exc
  except json.JSONDecodeError as exc:
    raise ValueError(f"{label} is invalid JSON: {path}: {exc}") from exc
  if not isinstance(payload, dict):
    raise ValueError(f"{label} must be a JSON object: {path}")
  return payload


def _formal_manifest_path(
    args: argparse.Namespace,
    split_paths: dict[str, Path],
) -> Path:
  configured = str(getattr(args, "split_manifest", "") or "").strip()
  if configured:
    return Path(configured)
  parents = {path.resolve().parent for path in split_paths.values()}
  if len(parents) == 1:
    inferred = next(iter(parents)) / "split_manifest.json"
    if inferred.is_file():
      return inferred
  raise ValueError(
      "Formal hygiene requires --split_manifest (or a shared adjacent split_manifest.json)"
  )


def _verify_formal_split_manifest(
    manifest_path: Path,
    *,
    split_paths: dict[str, Path],
    raw_cases_by_split: dict[str, list[dict[str, Any]]],
    authoritative_source_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  manifest = _load_json_object(manifest_path, label="Split manifest")
  if manifest.get("formal") is not True:
    raise ValueError("Formal hygiene requires a manifest with formal=true")
  counts = manifest.get("split_case_counts")
  hashes = manifest.get("artifact_hashes")
  if not isinstance(counts, dict) or not isinstance(hashes, dict):
    raise ValueError("Split manifest is missing split_case_counts or artifact_hashes")
  verified_artifacts: dict[str, Any] = {}
  for split, path in split_paths.items():
    expected_count = counts.get(split)
    actual_count = len(raw_cases_by_split[split])
    if expected_count != actual_count:
      raise ValueError(
          f"Split manifest count mismatch for {split}: "
          f"expected {expected_count!r}, actual {actual_count}"
      )
    record = hashes.get(split)
    if not isinstance(record, dict):
      raise ValueError(f"Split manifest has no artifact hash for {split}")
    expected_sha = str(record.get("sha256") or "").lower()
    if expected_sha.startswith("sha256:"):
      expected_sha = expected_sha[7:]
    actual_sha = _sha256_file(path)
    actual_bytes = path.stat().st_size
    if expected_sha != actual_sha:
      raise ValueError(f"Split manifest SHA256 mismatch for {split}")
    if record.get("bytes") != actual_bytes:
      raise ValueError(f"Split manifest byte-count mismatch for {split}")
    verified_artifacts[split] = {
        "case_count": actual_count,
        "sha256": actual_sha,
        "bytes": actual_bytes,
    }
  family_records = manifest.get("family_case_records")
  if not isinstance(family_records, list):
    raise ValueError("Split manifest is missing family_case_records")
  policy_identity = manifest.get("frozen_policy_identity")
  if not isinstance(policy_identity, dict) or not isinstance(
      policy_identity.get("payload"), dict
  ):
    raise ValueError("Split manifest is missing its full frozen policy identity")
  recomputed_policy_identity = _identity_for_payload(policy_identity["payload"])
  if policy_identity != recomputed_policy_identity:
    raise ValueError("Split manifest frozen policy identity is not self-consistent")
  expected_schema_identity = _family_source_schema_identity()
  if manifest.get("family_provenance_schema_identity") != expected_schema_identity:
    raise ValueError("Split manifest provenance schema identity does not match this code")
  expected_code_identity = _code_identity()
  if manifest.get("code_identity") != expected_code_identity:
    raise ValueError("Split manifest code identity does not match this audit code")

  input_record = hashes.get("input_cases")
  if not isinstance(input_record, dict):
    raise ValueError("Formal split manifest requires authoritative input_cases identity")
  source_path = authoritative_source_path
  if source_path is None:
    raw_source_path = str(input_record.get("path") or "").strip()
    if not raw_source_path:
      raise ValueError("Formal split manifest input_cases identity has no path")
    recorded = Path(raw_source_path)
    candidates = [recorded]
    if not recorded.is_absolute():
      candidates.append(manifest_path.resolve().parent / recorded)
    candidates.append(manifest_path.resolve().parent / recorded.name)
    source_path = next((path for path in candidates if path.is_file()), None)
  if source_path is None or not source_path.is_file():
    raise ValueError(
        "Formal hygiene cannot resolve the authoritative family source cases"
    )
  expected_source_sha = str(input_record.get("sha256") or "").lower()
  if expected_source_sha.startswith("sha256:"):
    expected_source_sha = expected_source_sha[7:]
  if _sha256_file(source_path) != expected_source_sha:
    raise ValueError("Authoritative input_cases SHA256 does not match the manifest")
  if source_path.stat().st_size != input_record.get("bytes"):
    raise ValueError("Authoritative input_cases byte count does not match the manifest")
  source_document = json.loads(source_path.read_text(encoding="utf-8"))
  if isinstance(source_document, list):
    source_cases = source_document
  elif isinstance(source_document, dict):
    source_cases = source_document.get("cases")
  else:
    source_cases = None
  if not isinstance(source_cases, list) or not all(
      isinstance(case, dict) for case in source_cases
  ):
    raise ValueError("Authoritative input_cases must contain a case-object list")

  policy = policy_identity["payload"]
  try:
    dev_policy = policy["dev_subset"]
    near_policy = policy["near_family_union"]
    review_policy = policy["hygiene_review"]
    final_policy = policy["final_test"]
    recomputed = build_family_splits(
        source_cases,
        config=FamilySplitConfig(
            seed=int(dev_policy["seed"]),
            dev_ratio=float(dev_policy["target_ratio"]),
            dev_ratio_tolerance=float(dev_policy["ratio_tolerance"]),
            fail_if_ratio_impossible=bool(dev_policy["fail_if_impossible"]),
            near_family_threshold=(
                float(near_policy["threshold"])
                if near_policy.get("enabled")
                else None
            ),
            neighbor_top_k=int(review_policy["top_k"]),
            neighbor_warning_threshold=float(review_policy["warning_threshold"]),
            required_evidence_coverage=float(
                review_policy["required_descriptor_coverage"]
            ),
            development_synthetic_mode=False,
            final_test_generation_rule=str(final_policy["generation_rule"]),
            final_test_salt_commitment=final_policy.get("salt_commitment"),
        ),
    )
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(
        "Split manifest frozen policy cannot reproduce family assignment"
    ) from exc
  recomputed_manifest = recomputed["manifest"]
  if family_records != recomputed_manifest["family_case_records"]:
    raise ValueError(
        "authoritative source recomputation disagrees with family_case_records"
    )
  for split in split_paths:
    if raw_cases_by_split[split] != recomputed[split]:
      raise ValueError(
          f"Authoritative source recomputation disagrees with {split} split cases"
      )
  for field in (
      "family_groups",
      "merge_edges",
      "family_count",
      "split_case_counts",
      "split_family_counts",
      "frozen_policy_identity",
      "family_provenance_schema_identity",
      "code_identity",
      "data_identity",
  ):
    if manifest.get(field) != recomputed_manifest.get(field):
      raise ValueError(
          f"Authoritative source recomputation disagrees with manifest {field}"
      )
  verification = {
      "verified": True,
      "manifest_path": str(manifest_path.resolve()),
      "manifest_sha256": _sha256_file(manifest_path),
      "artifacts": verified_artifacts,
      "authoritative_source": {
          "path": str(source_path.resolve()),
          "sha256": expected_source_sha,
          "case_count": len(source_cases),
      },
  }
  return manifest, verification


def audit(args: argparse.Namespace) -> dict[str, Any]:
  split_paths = _parse_named_paths(args.splits)
  if len(split_paths) < 2:
    raise ValueError("Provide at least two --splits entries.")
  formal = bool(getattr(args, "formal", True))
  if formal:
    if set(split_paths) != {"train", "dev"}:
      raise ValueError("Formal hygiene accepts exactly train and dev split files")
    if int(getattr(args, "max_cases_per_split", 0)) != 0:
      raise ValueError("Formal hygiene forbids max_cases_per_split sampling")
    forbidden_path_tokens = []
    for split, path in split_paths.items():
      tokens = {
          token
          for token in re.split(r"[^a-z0-9]+", path.name.lower())
          if token
      }
      if tokens & {"test", "final", "finaltest"}:
        forbidden_path_tokens.append(f"{split}={path}")
    if forbidden_path_tokens:
      raise ValueError(
          "Formal train/dev hygiene rejects final/test file variants: "
          + ", ".join(forbidden_path_tokens)
      )
  dataset_roots = _parse_dataset_roots(
      dataset_root=str(args.dataset_root),
      dataset_roots=list(args.dataset_roots or []),
  )
  file_index = _index_step_files(dataset_roots)
  part_cache: dict[str, dict[str, Any]] = {}
  split_parts: dict[str, list[dict[str, Any]]] = {}
  split_cases: dict[str, list[dict[str, Any]]] = {}
  raw_cases_by_split: dict[str, list[dict[str, Any]]] = {}
  skip_intrinsic_descriptors = bool(
      getattr(
          args,
          "skip_intrinsic_descriptors",
          getattr(args, "skip_interface_histograms", False),
      )
  )
  if formal and skip_intrinsic_descriptors:
    raise ValueError(
        "Formal hygiene requires intrinsic shape descriptors and forbids skipping them"
    )

  for split, path in split_paths.items():
    cases = _load_cases_strict(path)
    if int(args.max_cases_per_split) > 0:
      cases = cases[: int(args.max_cases_per_split)]
    raw_cases_by_split[split] = cases
    case_rows: list[dict[str, Any]] = []
    part_rows: list[dict[str, Any]] = []
    for case in cases:
      resolved = _resolve_case_parts(
          case=case,
          dataset_roots=dataset_roots,
          file_index=file_index,
      )
      for part in resolved:
        key = str(part["step_path"])
        if key not in part_cache:
          part_cache[key] = _part_fingerprints(
              Path(key),
              skip_intrinsic_descriptors=skip_intrinsic_descriptors,
          )
        part_rows.append(
            {
                "split": split,
                "case_id": str(case.get("id") or ""),
                "assembly_dir": str(case.get("assembly_dir") or ""),
                "part_name": str(part["part_name"]),
                "body_uuid": str(part["body_uuid"]),
                "step_name": Path(key).name,
                "step_path": key,
                **part_cache[key],
            }
        )
      case_rows.append(
          _case_fingerprint(
              split=split,
              case=case,
              resolved_parts=resolved,
              part_cache=part_cache,
          )
      )
    split_parts[split] = part_rows
    split_cases[split] = case_rows

  manifest: dict[str, Any] | None = None
  manifest_verification: dict[str, Any] = {
      "verified": False,
      "status": "nonformal_not_required",
  }
  if formal:
    manifest_path = _formal_manifest_path(args, split_paths)
    manifest, manifest_verification = _verify_formal_split_manifest(
        manifest_path,
        split_paths=split_paths,
        raw_cases_by_split=raw_cases_by_split,
        authoritative_source_path=(
            Path(str(getattr(args, "family_source_cases", "") or ""))
            if str(getattr(args, "family_source_cases", "") or "").strip()
            else None
        ),
    )

  overlap_summary = {
      "step_sha1": _overlap_report(split_parts, "step_sha1", int(args.max_examples)),
      "intrinsic_shape_signature": _overlap_report(
          split_parts,
          "intrinsic_shape_signature",
          int(args.max_examples),
      ),
      "case_part_graph_fingerprint": _overlap_report(
          split_cases,
          "case_part_graph_fingerprint",
          int(args.max_examples),
      ),
  }

  residual_summary = _residual_overlap_summary(
      specs=_parse_named_paths(args.mate_program_splits),
      translation_round_mm=float(args.residual_translation_round_mm),
      rotation_round_deg=float(args.residual_rotation_round_deg),
      max_examples=int(args.max_examples),
  )
  if residual_summary:
    overlap_summary["mate_residual_near_duplicate"] = residual_summary

  review_artifact = getattr(args, "family_pair_review_artifact", None)
  review_path = str(getattr(args, "family_pair_review", "") or "").strip()
  if review_artifact is None and review_path:
    review_artifact = _load_json_object(
        Path(review_path), label="Family-pair review artifact"
    )
  review_key = getattr(args, "review_verification_key", None)
  key_path = str(
      getattr(args, "review_verification_key_file", "") or ""
  ).strip()
  if review_key is None and key_path:
    review_key = Path(key_path).read_bytes().strip()

  family_hygiene = audit_family_splits(
      raw_cases_by_split,
      top_k=int(getattr(args, "nearest_neighbor_top_k", 20)),
      warning_threshold=float(getattr(args, "neighbor_warning_threshold", 0.9)),
      allow_missing=not formal,
      required_evidence_coverage=float(
          getattr(args, "required_evidence_coverage", 1.0)
      ),
      review_artifact=review_artifact,
      review_verification_key=review_key,
      family_records=(
          manifest.get("family_case_records")
          if isinstance(manifest, dict)
          else None
      ),
  )
  family_overlaps = family_hygiene.get("overlaps", {})
  for output_name, family_name in (
      ("family_id", "family_id"),
      ("source_lineage", "source_lineage"),
      ("provenance_body_sha1", "body_sha1"),
      ("provenance_graph_fingerprint", "graph_fingerprint"),
  ):
    item = family_overlaps.get(family_name)
    if isinstance(item, dict):
      overlap_summary[output_name] = item

  split_stats = {}
  for split in split_paths:
    split_stats[split] = {
        "case_count": len(split_cases.get(split, [])),
        "part_instance_count": len(split_parts.get(split, [])),
        "unique_step_sha1_count": len(
            {row.get("step_sha1") for row in split_parts.get(split, []) if row.get("step_sha1")}
        ),
        "unique_intrinsic_shape_signature_count": len(
            {
                row.get("intrinsic_shape_signature")
                for row in split_parts.get(split, [])
                if row.get("intrinsic_shape_signature")
            }
        ),
        "unique_case_graph_fingerprint_count": len(
            {
                row.get("case_part_graph_fingerprint")
                for row in split_cases.get(split, [])
                if row.get("case_part_graph_fingerprint")
            }
        ),
    }

  geometry_overlap_keys = {
      "step_sha1",
      "case_part_graph_fingerprint",
  }
  if not skip_intrinsic_descriptors:
    geometry_overlap_keys.add("intrinsic_shape_signature")
  geometry_clean = all(
      item.get("total_pair_overlap_count", 0) == 0
      for key, item in overlap_summary.items()
      if key in geometry_overlap_keys
  )
  clean = bool(
      geometry_clean
      and (not formal or manifest_verification.get("verified", False))
      and family_hygiene.get("evidence_complete", False)
      and family_hygiene.get("strict_hygiene_clean", False)
  )
  return {
      "audit": "neurocad_split_hygiene",
      "formal": formal,
      "dataset_roots": [str(path.resolve()) for path in dataset_roots],
      "split_paths": {key: str(path) for key, path in split_paths.items()},
      "config": {
          "residual_translation_round_mm": float(args.residual_translation_round_mm),
          "residual_rotation_round_deg": float(args.residual_rotation_round_deg),
          "intrinsic_descriptor_version": "intrinsic_brep.v1",
          "skip_intrinsic_descriptors": skip_intrinsic_descriptors,
          "max_cases_per_split": int(args.max_cases_per_split),
          "nearest_neighbor_top_k": int(
              getattr(args, "nearest_neighbor_top_k", 20)
          ),
          "neighbor_warning_threshold": float(
              getattr(args, "neighbor_warning_threshold", 0.9)
          ),
      },
      "split_stats": split_stats,
      "overlaps": overlap_summary,
      "family_hygiene": family_hygiene,
      "manifest_verification": manifest_verification,
      "cross_split_nearest_neighbors": family_hygiene.get(
          "nearest_neighbor_top20", []
      ),
      "geometry_clean": bool(geometry_clean),
      "strict_hygiene_clean": bool(clean),
      "notes": [
          "STEP SHA1, source lineage, family id, and graph fingerprints must have zero cross-split overlap.",
          "Intrinsic B-Rep signatures use only area, volume, topology, and surface type; no AABB, world pose, or role_hint enters hygiene.",
          "Requested intrinsic descriptor failures abort the audit instead of emitting placeholder signatures.",
          "Residual near-duplicate overlap requires per-split mined mate-program JSONL inputs.",
          "Missing formal family provenance fails strict hygiene instead of being inferred from names or paths.",
          "The top cross-split nearest neighbors require recorded manual review before paper use.",
      ],
  }


def _parse_named_paths(items: Iterable[str]) -> dict[str, Path]:
  result: dict[str, Path] = {}
  seen_names: set[str] = set()
  for item in items or []:
    text = str(item).strip()
    if not text:
      raise ValueError("Split path specs must not be empty")
    if "=" not in text:
      raise ValueError(f"Expected name=path spec, got: {text}")
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
      raise ValueError(f"Invalid split name in spec: {text}")
    canonical_name = name.lower()
    if canonical_name in seen_names:
      raise ValueError(f"Duplicate split name: {name}")
    seen_names.add(canonical_name)
    if not raw_path.strip():
      raise ValueError(f"Empty split path in spec: {text}")
    result[name] = Path(raw_path.strip())
  return result


def _resolve_case_parts(
    *,
    case: dict[str, Any],
    dataset_roots: list[Path],
    file_index: dict[str, list[Path]],
) -> list[dict[str, Any]]:
  parts = case.get("parts")
  if not isinstance(parts, list) or not parts:
    parts = case.get("candidate_parts")
  if not isinstance(parts, list) or not parts:
    return []
  names = case.get("selected_part_names")
  if not isinstance(names, list) or len(names) != len(parts):
    names = case.get("candidate_part_names")
  if not isinstance(names, list):
    names = []
  uuids = case.get("selected_body_uuids")
  if not isinstance(uuids, list) or len(uuids) != len(parts):
    uuids = case.get("candidate_body_uuids")
  if not isinstance(uuids, list):
    uuids = []
  assembly_dir = case.get("assembly_dir") if isinstance(case.get("assembly_dir"), str) else None
  root_hint = (
      case.get("dataset_root_hint")
      if isinstance(case.get("dataset_root_hint"), str)
      else None
  )
  rows = []
  for index, token in enumerate(parts):
    step_path = _resolve_part_path(
        part_token=str(token),
        dataset_roots=dataset_roots,
        assembly_dir=assembly_dir,
        dataset_root_hint=root_hint,
        file_index=file_index,
    )
    part_name = (
        str(names[index]).strip()
        if index < len(names) and str(names[index]).strip()
        else step_path.stem
    )
    body_uuid = (
        str(uuids[index]).strip()
        if index < len(uuids) and str(uuids[index]).strip()
        else step_path.stem
    )
    rows.append(
        {
            "part_name": part_name,
            "body_uuid": body_uuid,
            "step_path": str(step_path.resolve()),
        }
    )
  return rows


def _part_fingerprints(
    step_path: Path,
    *,
    skip_intrinsic_descriptors: bool,
) -> dict[str, Any]:
  step_sha1 = _sha1_file(step_path)
  if skip_intrinsic_descriptors:
    return {
        "step_sha1": step_sha1,
        "intrinsic_shape_signature": "skipped",
        "intrinsic_shape_descriptor": None,
    }
  try:
    shape = load_step_shape(step_path)
    faces = list(shape.Faces())
    edges = list(shape.Edges())
    vertices = list(shape.Vertices())
    solids = list(shape.Solids())
    volume = abs(float(shape.Volume()))
    area = float(shape.Area())
    if not math.isfinite(volume) or not math.isfinite(area) or area <= 0.0:
      raise ValueError("non-finite or non-positive intrinsic measure")
    face_rows = []
    surface_types: Counter[str] = Counter()
    for face in faces:
      surface_type = str(face.geomType()).strip().lower()
      face_area = float(face.Area())
      if not surface_type or not math.isfinite(face_area) or face_area <= 0.0:
        raise ValueError("invalid intrinsic face descriptor")
      edge_types = sorted(
          str(edge.geomType()).strip().lower() for edge in list(face.Edges())
      )
      if any(not edge_type for edge_type in edge_types):
        raise ValueError("invalid intrinsic edge descriptor")
      surface_types[surface_type] += 1
      face_rows.append(
          {
              "surface_type": surface_type,
              "area": round(face_area, 9),
              "edge_types": edge_types,
          }
      )
    descriptor = {
        "version": "intrinsic_brep.v1",
        "volume": round(volume, 9),
        "area": round(area, 9),
        "solid_count": len(solids),
        "face_count": len(faces),
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "surface_type_histogram": sorted(surface_types.items()),
        "faces": sorted(face_rows, key=lambda row: json.dumps(row, sort_keys=True)),
    }
  except Exception as exc:  # pylint: disable=broad-except
    raise RuntimeError(
        f"Intrinsic B-Rep descriptor extraction failed for {step_path}: "
        f"{type(exc).__name__}: {exc}"
    ) from exc
  canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
  return {
      "step_sha1": step_sha1,
      "intrinsic_shape_signature": (
          "intrinsic_brep_sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
      ),
      "intrinsic_shape_descriptor": descriptor,
  }


def _case_fingerprint(
    *,
    split: str,
    case: dict[str, Any],
    resolved_parts: list[dict[str, Any]],
    part_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  name_to_sig = {}
  part_sigs = []
  for part in resolved_parts:
    path = str(part["step_path"])
    cached = part_cache.get(path, {})
    sig = str(
        cached.get("intrinsic_shape_signature")
        if cached.get("intrinsic_shape_signature") not in {None, "skipped"}
        else cached.get("step_sha1")
    )
    name_to_sig[str(part["part_name"])] = sig
    name_to_sig[str(part["body_uuid"])] = sig
    name_to_sig[Path(path).stem] = sig
    part_sigs.append(sig)
  edge_sigs = []
  degree: Counter[str] = Counter()
  for raw in case.get("contact_pairs") or []:
    if not isinstance(raw, list) or len(raw) != 2:
      continue
    a = name_to_sig.get(str(raw[0]))
    b = name_to_sig.get(str(raw[1]))
    if not a or not b:
      continue
    edge = tuple(sorted((a, b)))
    edge_sigs.append(edge)
    degree[a] += 1
    degree[b] += 1
  payload = {
      "part_sigs": sorted(part_sigs),
      "edge_sigs": sorted(edge_sigs),
      "degree_sequence": sorted(degree.values()),
  }
  return {
      "split": split,
      "case_id": str(case.get("id") or ""),
      "assembly_dir": str(case.get("assembly_dir") or ""),
      "part_count": len(resolved_parts),
      "edge_count": len(edge_sigs),
      "case_part_graph_fingerprint": _hash_text(json.dumps(payload, sort_keys=True)),
  }


def _overlap_report(
    rows_by_split: dict[str, list[dict[str, Any]]],
    key: str,
    max_examples: int,
) -> dict[str, Any]:
  values_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
  for split, rows in rows_by_split.items():
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
      value = str(row.get(key) or "")
      if not value or value in {"unavailable", "skipped"}:
        continue
      bucket[value].append(row)
    values_by_split[split] = bucket

  pair_counts = {}
  examples = []
  splits = list(rows_by_split)
  total = 0
  for i, left in enumerate(splits):
    for right in splits[i + 1 :]:
      common = sorted(set(values_by_split[left]) & set(values_by_split[right]))
      pair_key = f"{left}__{right}"
      pair_counts[pair_key] = len(common)
      total += len(common)
      for value in common[: max(0, int(max_examples))]:
        examples.append(
            {
                "pair": pair_key,
                "value": value,
                "left_examples": _compact_examples(values_by_split[left][value]),
                "right_examples": _compact_examples(values_by_split[right][value]),
            }
        )
  return {
      "key": key,
      "pair_overlap_counts": pair_counts,
      "total_pair_overlap_count": total,
      "examples": examples[: max(0, int(max_examples))],
  }


def _compact_examples(rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
  result = []
  for row in rows[:limit]:
    result.append(
        {
            "case_id": row.get("case_id"),
            "assembly_dir": row.get("assembly_dir"),
            "part_name": row.get("part_name"),
            "step_name": row.get("step_name"),
        }
    )
  return result


def _residual_overlap_summary(
    *,
    specs: dict[str, Path],
    translation_round_mm: float,
    rotation_round_deg: float,
    max_examples: int,
) -> dict[str, Any]:
  if len(specs) < 2:
    return {}
  rows_by_split = {}
  for split, path in specs.items():
    programs = load_mate_programs(path)
    rows = []
    for program in programs:
      rows.append(
          {
              "split": split,
              "case_id": program.case_id,
              "assembly_dir": program.assembly_dir,
              "program_id": program.program_id,
              "source": f"{program.part_a}->{program.part_b}",
              "residual_near_signature": _residual_signature(
                  program,
                  translation_round_mm=translation_round_mm,
                  rotation_round_deg=rotation_round_deg,
              ),
          }
      )
    rows_by_split[split] = rows
  return _overlap_report(rows_by_split, "residual_near_signature", max_examples)


def _residual_signature(
    program: MateProgram,
    *,
    translation_round_mm: float,
    rotation_round_deg: float,
) -> str:
  try:
    trans = np.asarray(program.residual_translation, dtype=float).reshape(3)
    rot = np.asarray(program.residual_rotation, dtype=float).reshape(3, 3)
    t_key = tuple(_round_to(float(value), translation_round_mm) for value in trans)
    angle = rotation_angle_degrees(rot)
    angle_key = _round_to(angle, rotation_round_deg)
  except Exception:
    t_key = (0.0, 0.0, 0.0)
    angle_key = 0.0
  payload = {
      "contact_family": _contact_family(program.contact_type),
      "parent_role": program.parent_role,
      "child_role": program.child_role,
      "translation": t_key,
      "rotation_angle": angle_key,
  }
  return _hash_text(json.dumps(payload, sort_keys=True))


def _sha1_file(path: Path) -> str:
  digest = hashlib.sha1()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _hash_text(text: str) -> str:
  return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _round_to(value: float, step: float) -> float:
  step = max(float(step), 1e-9)
  return round(round(float(value) / step) * step, 6)


def _contact_family(contact_type: str) -> str:
  text = str(contact_type or "").lower()
  if "slot" in text:
    return "slot"
  if any(token in text for token in ("insert", "bore", "hole", "shaft", "screw", "threaded")):
    return "insert"
  if any(token in text for token in ("seat", "support", "plane", "flange")):
    return "seat"
  return "generic"


def _render_markdown(summary: dict[str, Any]) -> str:
  lines = [
      "# NeuroCAD Split Hygiene Audit",
      "",
      "## Split Stats",
      "",
      "| Split | Cases | Part Instances | Unique STEP SHA1 | Unique Intrinsic B-Rep | Unique Case Graph |",
      "|---|---:|---:|---:|---:|---:|",
  ]
  for split, stats in summary.get("split_stats", {}).items():
    lines.append(
        "| {split} | {case_count} | {part_instance_count} | {unique_step_sha1_count} | "
        "{unique_intrinsic_shape_signature_count} | "
        "{unique_case_graph_fingerprint_count} |".format(split=split, **stats)
    )
  lines.extend(["", "## Overlap Checks", ""])
  lines.append("| Check | Total Pair Overlap | Pair Counts |")
  lines.append("|---|---:|---|")
  for name, item in summary.get("overlaps", {}).items():
    lines.append(
        f"| {name} | {int(item.get('total_pair_overlap_count', 0))} | "
        f"`{json.dumps(item.get('pair_overlap_counts', {}), sort_keys=True)}` |"
    )
  family_hygiene = summary.get("family_hygiene", {})
  lines.extend(
      [
          "",
          "## Cross-Split Family Nearest Neighbors",
          "",
          (
              "Family evidence complete: "
              + str(bool(family_hygiene.get("evidence_complete", False)))
              + "; manual review status: "
              + str(family_hygiene.get("manual_review", {}).get("status", "unknown"))
              + "."
          ),
          "",
          "| Rank | Left | Right | Similarity | Warning | Components |",
          "|---:|---|---|---:|---|---|",
      ]
  )
  for rank, row in enumerate(
      summary.get("cross_split_nearest_neighbors", []), start=1
  ):
    lines.append(
        "| {rank} | {left_split}:{left_case} | {right_split}:{right_case} | "
        "{score:.6f} | {warning} | `{components}` |".format(
            rank=rank,
            left_split=row.get("left_split"),
            left_case=row.get("left_case_id"),
            right_split=row.get("right_split"),
            right_case=row.get("right_case_id"),
            score=float(row.get("score", 0.0)),
            warning=bool(row.get("warning", False)),
            components=json.dumps(row.get("components", {}), sort_keys=True),
        )
    )
  lines.extend(
      [
          "",
          "## Notes",
          "",
      ]
  )
  for note in summary.get("notes", []):
    lines.append(f"- {note}")
  lines.append("")
  return "\n".join(lines)


if __name__ == "__main__":
  main()
