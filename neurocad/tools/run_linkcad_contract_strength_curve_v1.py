"""Run LinkCAD's frozen five-to-one-field contract-strength experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from neurocad.linkcad_contract_strength_evaluation_v1 import (
    SCHEMA_VERSION,
    ambiguity_summary_v1,
    paired_seed_median_bootstrap_v1,
    target_retention_audit_v1,
)
from neurocad.linkcad_contract_strength_v1 import (
    LEVEL_FIELDS,
    project_public_contract_strength_v1,
)
from neurocad.linkcad_kinematic_execution_v2 import (
    materialize_public_primitive_predictions_v2,
)
from neurocad.linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2
from neurocad.linkcad_symbolic_baseline_v1 import (
    materialize_symbolic_ambiguity_audit_v1,
    materialize_symbolic_predictions_v1,
)
from neurocad.tools.evaluate_linkcad_confirmation_v2 import (
    _primitive_targets,
    evaluate_confirmation_predictions_v3,
)


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )


def _file_sha(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--phase", choices=("target-free", "evaluate", "all"), default="all",
  )
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--private-targets", type=Path)
  parser.add_argument("--primitive-supervision", type=Path)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, action="append", required=True)
  parser.add_argument("--seed", type=int, action="append", required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--bootstrap-samples", type=int, default=10_000)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if len(args.checkpoint) != len(args.seed) or len(args.checkpoint) < 1:
    raise ValueError("LinkCAD contract-strength checkpoint/seed domain differs")
  public = _read(args.public)
  cache = LinkCADPrimitiveGraphCacheV2(args.primitive_cache)
  cache.load_all()

  target_free: dict[str, dict[str, Any]] = {}
  if args.phase in {"target-free", "all"}:
    # This phase never opens the private target or primitive-supervision files.
    artifact_files = []
    for level in LEVEL_FIELDS:
      level_root = args.output_root / level
      projected = project_public_contract_strength_v1(public, level=level)
      _write(level_root / "public.projected.json", projected)
      ambiguity = materialize_symbolic_ambiguity_audit_v1(
          public=projected, cache=cache,
      )
      symbolic = materialize_symbolic_predictions_v1(
          public=projected, cache=cache, policy="minimum_port_ambiguity",
          beam_size=25, interface_alternatives_per_edge=8,
      )
      _write(level_root / "symbolic_ambiguity.json", ambiguity)
      _write(level_root / "symbolic_predictions.json", symbolic)
      learned = []
      for checkpoint, seed in zip(args.checkpoint, args.seed, strict=True):
        prediction = materialize_public_primitive_predictions_v2(
            public=projected, cache=cache, checkpoint_path=checkpoint,
            beam_size=25, interface_alternatives_per_edge=8,
            seed=seed, model_kind="port_v9",
        )
        path = level_root / f"linkcad_seed{seed}.json"
        _write(path, prediction)
        learned.append(prediction)
      target_free[level] = {
          "public": projected, "ambiguity": ambiguity,
          "symbolic": symbolic, "learned": learned,
      }
      artifact_files.extend([
          level_root / "public.projected.json",
          level_root / "symbolic_ambiguity.json",
          level_root / "symbolic_predictions.json",
          *[level_root / f"linkcad_seed{seed}.json" for seed in args.seed],
      ])
    terminal = {
        "schema_version": "linkcad_contract_strength_target_free_terminal.v1",
        "query_count": int(public["query_count"]),
        "levels": list(LEVEL_FIELDS), "seeds": list(args.seed),
        "private_targets_opened": False,
        "artifact_sha256s": {
            path.relative_to(args.output_root).as_posix(): _file_sha(path)
            for path in artifact_files
        },
    }
    _write(args.output_root / "target_free_terminal.json", terminal)
    if args.phase == "target-free":
      print(json.dumps({
          "query_count": terminal["query_count"],
          "private_targets_opened": False,
          "artifact_count": len(terminal["artifact_sha256s"]),
      }, sort_keys=True))
      return

  if args.private_targets is None or args.primitive_supervision is None:
    raise ValueError("LinkCAD contract-strength evaluation inputs are required")
  terminal_path = args.output_root / "target_free_terminal.json"
  if not terminal_path.exists():
    raise ValueError("LinkCAD contract-strength target-free terminal is absent")
  terminal = _read(terminal_path)
  if terminal.get("private_targets_opened") is not False:
    raise ValueError("LinkCAD contract-strength target-free terminal differs")
  for relative, expected in terminal["artifact_sha256s"].items():
    if _file_sha(args.output_root / relative) != expected:
      raise ValueError("LinkCAD contract-strength target-free artifact differs")
  if not target_free:
    for level in LEVEL_FIELDS:
      level_root = args.output_root / level
      target_free[level] = {
          "public": _read(level_root / "public.projected.json"),
          "ambiguity": _read(level_root / "symbolic_ambiguity.json"),
          "symbolic": _read(level_root / "symbolic_predictions.json"),
          "learned": [
              _read(level_root / f"linkcad_seed{seed}.json")
              for seed in args.seed
          ],
      }

  # Phase two joins labels only for evaluation; no model or schedule changes.
  private = _read(args.private_targets)
  primitive_targets = _primitive_targets(_read(args.primitive_supervision))
  levels = []
  for level, rows in target_free.items():
    level_root = args.output_root / level
    symbolic_evaluation = evaluate_confirmation_predictions_v3(
        public=rows["public"], private=private,
        primitive_targets=primitive_targets, predictions=rows["symbolic"],
    )
    _write(level_root / "symbolic_evaluation.json", symbolic_evaluation)
    learned_evaluations = []
    for seed, prediction in zip(args.seed, rows["learned"], strict=True):
      evaluation = evaluate_confirmation_predictions_v3(
          public=rows["public"], private=private,
          primitive_targets=primitive_targets, predictions=prediction,
      )
      _write(level_root / f"linkcad_seed{seed}_evaluation.json", evaluation)
      learned_evaluations.append(evaluation)
    retention = target_retention_audit_v1(
        public=rows["public"], private=private,
        primitive_targets=primitive_targets, cache=cache,
    )
    _write(level_root / "target_retention.json", retention)
    level_summary = {
        "level": level,
        "active_fields": list(LEVEL_FIELDS[level]),
        "ambiguity": ambiguity_summary_v1(rows["ambiguity"]),
        "target_retention": {
            key: retention[key] for key in (
                "target_endpoint_candidate_retention",
                "target_primitive_retention", "target_query_retention",
            )
        },
        "symbolic": {
            key: symbolic_evaluation["metrics"][key] for key in (
                "top1_strict_part_accuracy", "assignment_recall_at_25",
                "edge_part_and_primitive_top8_recall_at_25",
            )
        },
        "linkcad_seed_median": {
            key: median(
                row["metrics"][key] for row in learned_evaluations
            )
            for key in (
                "top1_strict_part_accuracy", "assignment_recall_at_25",
                "edge_part_and_primitive_top8_recall_at_25",
            )
        },
        "paired_bootstrap_vs_symbolic": {
            metric: paired_seed_median_bootstrap_v1(
                learned=learned_evaluations, symbolic=symbolic_evaluation,
                metric=metric, samples=args.bootstrap_samples,
                seed=20260823 + index,
            )
            for index, metric in enumerate((
                "top1_strict_part_accuracy", "assignment_recall_at_25",
            ))
        },
    }
    _write(level_root / "summary.json", level_summary)
    levels.append(level_summary)

  manifest = {
      "schema_version": SCHEMA_VERSION,
      "scope": (
          "prospective_contract_strength_confirmation"
          if args.phase == "evaluate"
          else "post_confirmation_fixed_schedule_contract_strength_ablation"
      ),
      "prospective_confirmation": args.phase == "evaluate",
      "private_targets_read_only_after_all_target_free_predictions": True,
      "omitted_contract_fields_are_wildcards": True,
      "query_count": int(public["query_count"]),
      "beam_size": 25,
      "interface_alternatives_per_edge": 8,
      "bootstrap_samples": args.bootstrap_samples,
      "inputs": {
          "public": _file_sha(args.public),
          "private_targets": _file_sha(args.private_targets),
          "primitive_supervision": _file_sha(args.primitive_supervision),
          "primitive_cache_manifest": _file_sha(
              args.primitive_cache / "manifest.json"
          ),
          "checkpoints": [
              {"seed": seed, "sha256": _file_sha(path)}
              for seed, path in zip(args.seed, args.checkpoint, strict=True)
          ],
      },
      "target_free_terminal_sha256": _file_sha(terminal_path),
      "levels": levels,
  }
  _write(args.output_root / "contract_strength_curve.json", manifest)
  print(json.dumps({
      "query_count": manifest["query_count"],
      "levels": [{
          "level": row["level"],
          "median_feasible": row["ambiguity"]["median_feasible_assignment_count"],
          "symbolic_t1": row["symbolic"]["top1_strict_part_accuracy"],
          "linkcad_t1": row["linkcad_seed_median"]["top1_strict_part_accuracy"],
      } for row in levels],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
