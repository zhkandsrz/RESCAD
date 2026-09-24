"""Join frozen LinkCAD V2 predictions with unsealed confirmation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping


SCHEMA_VERSION = "linkcad_confirmation_private_evaluation.v3"


def _read(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson(successes: int, count: int) -> list[float]:
  if count <= 0:
    return [0.0, 0.0]
  z = 1.959963984540054
  p = successes / count
  denominator = 1.0 + z * z / count
  center = (p + z * z / (2.0 * count)) / denominator
  half = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
  return [max(0.0, center - half), min(1.0, center + half)]


def _primitive_targets(supervision: Mapping[str, Any]):
  by_key = {
      (row["query_id"], row["edge_id"], row["side"]): int(
          row["primitive_orbit_ordinal"]
      )
      for row in supervision["rows"]
      if row.get("status") == "mapped_type_consistent"
  }
  result = {}
  for query_id, edge_id, _side in sorted(by_key):
    left = by_key.get((query_id, edge_id, "a"))
    right = by_key.get((query_id, edge_id, "b"))
    if left is not None and right is not None:
      result[(query_id, edge_id)] = (left, right)
  return result


def evaluate_confirmation_predictions_v3(
    *, public: Mapping[str, Any], private: Mapping[str, Any],
    primitive_targets: Mapping[tuple[str, str], tuple[int, int]],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
  public_by_id = {row["query_id"]: row for row in public["queries"]}
  private_by_id = {row["query_id"]: row for row in private["targets"]}
  prediction_by_id = {row["query_id"]: row for row in predictions["rows"]}
  counts = {
      "query_count": len(public_by_id),
      "role_count": 0,
      "edge_count": 0,
      "top1_role_correct": 0,
      "top1_strict_part": 0,
      "top1_strict_joint_mobility": 0,
      "top1_strict_support_mobility_program": 0,
      "assignment_recall_at_25": 0,
      "joint_mobility_recall_at_25": 0,
      "support_mobility_program_recall_at_25": 0,
      "edge_part_pair_recall_at_25": 0,
      "primitive_supervised_edge_count": 0,
      "primitive_supervised_edge_part_hit_count": 0,
      "edge_part_and_primitive_top1_recall_at_25": 0,
      "edge_part_and_primitive_top8_recall_at_25": 0,
      "primitive_complete_query_count": 0,
      "full_primitive_top1_program_recall_at_25": 0,
      "full_primitive_top8_program_recall_at_25": 0,
      "no_prediction_query_count": 0,
  }
  per_query = []
  for query_id, query in public_by_id.items():
    target = private_by_id[query_id]
    prediction_row = prediction_by_id.get(query_id)
    hypotheses = [] if prediction_row is None else prediction_row["hypotheses"]
    if not hypotheses:
      counts["no_prediction_query_count"] += 1
      counts["role_count"] += len(query["roles"])
      counts["edge_count"] += len(query["functional_edges"])
      supervised_edges = {
          edge["edge_id"]
          for edge in query["functional_edges"]
          if (query_id, edge["edge_id"]) in primitive_targets
      }
      counts["primitive_supervised_edge_count"] += len(supervised_edges)
      complete = len(supervised_edges) == len(query["functional_edges"])
      counts["primitive_complete_query_count"] += int(complete)
      per_query.append({
          "query_id": query_id,
          "role_correct": 0,
          "role_count": len(query["roles"]),
          "assignment_hit": False,
          "joint_mobility_hit": False,
          "support_mobility_program_hit": False,
          "top1_strict_part": False,
          "top1_strict_joint_mobility": False,
          "top1_strict_support_mobility_program": False,
          "primitive_complete": complete,
          "full_primitive_top1_program_hit": False,
          "full_primitive_top8_program_hit": False,
          "edges": [
              {
                  "edge_id": edge["edge_id"],
                  "edge_part_hit": False,
                  "primitive_supervised": edge["edge_id"] in supervised_edges,
                  "edge_part_and_primitive_top1_hit": False,
                  "edge_part_and_primitive_top8_hit": False,
              }
              for edge in query["functional_edges"]
          ],
      })
      continue
    role_ids = tuple(str(row["role_id"]) for row in query["roles"])
    target_assignment = {
        role: str(target["target_candidate_by_role"][role]) for role in role_ids
    }
    public_edge = {row["edge_id"]: row for row in query["functional_edges"]}
    target_edge = {row["edge_id"]: row for row in target["functional_edge_targets"]}
    counts["role_count"] += len(role_ids)
    counts["edge_count"] += len(public_edge)

    def assignment_ok(hypothesis):
      return hypothesis["candidate_by_role"] == target_assignment

    def mobility_ok(hypothesis):
      programs = {row["edge_id"]: row for row in hypothesis["edge_programs"]}
      return all(
          programs[edge_id]["mobility"] == truth["target_mobility"]
          for edge_id, truth in target_edge.items()
      )

    def support_ok(hypothesis):
      programs = {row["edge_id"]: row for row in hypothesis["edge_programs"]}
      return all(
          programs[edge_id]["support_family"] == truth["support_family"]
          for edge_id, truth in target_edge.items()
      )

    top = hypotheses[0]
    role_matches = [
        top["candidate_by_role"][role] == target_assignment[role]
        for role in role_ids
    ]
    counts["top1_role_correct"] += sum(role_matches)
    counts["top1_strict_part"] += int(all(role_matches))
    counts["top1_strict_joint_mobility"] += int(
        all(role_matches) and mobility_ok(top)
    )
    counts["top1_strict_support_mobility_program"] += int(
        all(role_matches) and mobility_ok(top) and support_ok(top)
    )
    assignment_hit = any(map(assignment_ok, hypotheses))
    joint_hit = any(
        assignment_ok(row) and mobility_ok(row) for row in hypotheses
    )
    support_program_hit = any(
        assignment_ok(row) and mobility_ok(row) and support_ok(row)
        for row in hypotheses
    )
    counts["assignment_recall_at_25"] += int(assignment_hit)
    counts["joint_mobility_recall_at_25"] += int(joint_hit)
    counts["support_mobility_program_recall_at_25"] += int(support_program_hit)
    edge_outcomes = []
    for edge_id, edge in public_edge.items():
      pair_hypotheses = [
          row for row in hypotheses
          if row["candidate_by_role"][edge["role_a"]]
          == target_assignment[edge["role_a"]]
          and row["candidate_by_role"][edge["role_b"]]
          == target_assignment[edge["role_b"]]
      ]
      counts["edge_part_pair_recall_at_25"] += int(bool(pair_hypotheses))
      target_pair = primitive_targets.get((query_id, edge_id))
      if target_pair is None:
        edge_outcomes.append({
            "edge_id": edge_id,
            "edge_part_hit": bool(pair_hypotheses),
            "primitive_supervised": False,
            "edge_part_and_primitive_top1_hit": False,
            "edge_part_and_primitive_top8_hit": False,
        })
        continue
      counts["primitive_supervised_edge_count"] += 1
      counts["primitive_supervised_edge_part_hit_count"] += int(
          bool(pair_hypotheses)
      )
      top1_hit = top8_hit = False
      for hypothesis in pair_hypotheses:
        program = next(
            row for row in hypothesis["edge_programs"] if row["edge_id"] == edge_id
        )
        top1_hit = top1_hit or (
            int(program["interface_primitive_a"]),
            int(program["interface_primitive_b"]),
        ) == target_pair
        alternatives = {
            (
                int(row["interface_primitive_a"]),
                int(row["interface_primitive_b"]),
            )
            for row in program["primitive_alternatives"]
        }
        top8_hit = top8_hit or target_pair in alternatives
      counts["edge_part_and_primitive_top1_recall_at_25"] += int(top1_hit)
      counts["edge_part_and_primitive_top8_recall_at_25"] += int(top8_hit)
      edge_outcomes.append({
          "edge_id": edge_id,
          "edge_part_hit": bool(pair_hypotheses),
          "primitive_supervised": True,
          "edge_part_and_primitive_top1_hit": top1_hit,
          "edge_part_and_primitive_top8_hit": top8_hit,
      })
    complete = all(
        (query_id, edge_id) in primitive_targets for edge_id in public_edge
    )
    if complete:
      counts["primitive_complete_query_count"] += 1
      full_top1 = full_top8 = False
      for hypothesis in hypotheses:
        if not (
            assignment_ok(hypothesis)
            and mobility_ok(hypothesis)
            and support_ok(hypothesis)
        ):
          continue
        programs = {row["edge_id"]: row for row in hypothesis["edge_programs"]}
        top1_ok = top8_ok = True
        for edge_id in public_edge:
          target_pair = primitive_targets[(query_id, edge_id)]
          program = programs[edge_id]
          top1_ok = top1_ok and (
              int(program["interface_primitive_a"]),
              int(program["interface_primitive_b"]),
          ) == target_pair
          top8_pairs = {
              (
                  int(row["interface_primitive_a"]),
                  int(row["interface_primitive_b"]),
              )
              for row in program["primitive_alternatives"]
          }
          top8_ok = top8_ok and target_pair in top8_pairs
        full_top1 = full_top1 or top1_ok
        full_top8 = full_top8 or top8_ok
      counts["full_primitive_top1_program_recall_at_25"] += int(full_top1)
      counts["full_primitive_top8_program_recall_at_25"] += int(full_top8)
    else:
      full_top1 = False
      full_top8 = False
    per_query.append({
        "query_id": query_id,
        "role_correct": sum(role_matches),
        "role_count": len(role_matches),
        "assignment_hit": assignment_hit,
        "joint_mobility_hit": joint_hit,
        "support_mobility_program_hit": support_program_hit,
        "top1_strict_part": all(role_matches),
        "top1_strict_joint_mobility": all(role_matches) and mobility_ok(top),
        "top1_strict_support_mobility_program": (
            all(role_matches) and mobility_ok(top) and support_ok(top)
        ),
        "primitive_complete": complete,
        "full_primitive_top1_program_hit": full_top1,
        "full_primitive_top8_program_hit": full_top8,
        "edges": edge_outcomes,
    })
  denominators = {
      "top1_role_accuracy": (counts["top1_role_correct"], counts["role_count"]),
      "top1_strict_part_accuracy": (counts["top1_strict_part"], counts["query_count"]),
      "top1_strict_joint_mobility_accuracy": (
          counts["top1_strict_joint_mobility"], counts["query_count"]
      ),
      "top1_strict_support_mobility_program_accuracy": (
          counts["top1_strict_support_mobility_program"], counts["query_count"]
      ),
      "assignment_recall_at_25": (counts["assignment_recall_at_25"], counts["query_count"]),
      "joint_mobility_recall_at_25": (
          counts["joint_mobility_recall_at_25"], counts["query_count"]
      ),
      "support_mobility_program_recall_at_25": (
          counts["support_mobility_program_recall_at_25"], counts["query_count"]
      ),
      "edge_part_pair_recall_at_25": (
          counts["edge_part_pair_recall_at_25"], counts["edge_count"]
      ),
      "edge_part_and_primitive_top1_recall_at_25": (
          counts["edge_part_and_primitive_top1_recall_at_25"],
          counts["primitive_supervised_edge_count"],
      ),
      "edge_part_and_primitive_top8_recall_at_25": (
          counts["edge_part_and_primitive_top8_recall_at_25"],
          counts["primitive_supervised_edge_count"],
      ),
      "primitive_top1_given_edge_part_hit": (
          counts["edge_part_and_primitive_top1_recall_at_25"],
          counts["primitive_supervised_edge_part_hit_count"],
      ),
      "primitive_top8_given_edge_part_hit": (
          counts["edge_part_and_primitive_top8_recall_at_25"],
          counts["primitive_supervised_edge_part_hit_count"],
      ),
      "full_primitive_top1_program_recall_at_25": (
          counts["full_primitive_top1_program_recall_at_25"],
          counts["primitive_complete_query_count"],
      ),
      "full_primitive_top8_program_recall_at_25": (
          counts["full_primitive_top8_program_recall_at_25"],
          counts["primitive_complete_query_count"],
      ),
  }
  metrics = {}
  for name, (successes, count) in denominators.items():
    metrics[name] = successes / count if count else 0.0
    metrics[name + "_wilson95"] = _wilson(successes, count)
  return {"counts": counts, "metrics": metrics, "per_query": per_query}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--private-targets", type=Path, required=True)
  parser.add_argument("--primitive-supervision", type=Path, required=True)
  parser.add_argument("--unseal-receipt", type=Path, required=True)
  parser.add_argument("--predictions", type=Path, nargs="+", required=True)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  receipt = _read(args.unseal_receipt)
  bound = {row["file_sha256"] for row in receipt["prediction_rows"]}
  observed = {_sha(path) for path in args.predictions}
  if (
      receipt.get("private_targets_opened_after_prediction_closure") is not True
      or not observed
      or not observed <= bound
  ):
    raise ValueError("confirmation evaluation predictions differ from unseal receipt")
  public = _read(args.public)
  private = _read(args.private_targets)
  supervision = _read(args.primitive_supervision)
  targets = _primitive_targets(supervision)
  per_seed = []
  for path in args.predictions:
    prediction = _read(path)
    result = evaluate_confirmation_predictions_v3(
        public=public, private=private, primitive_targets=targets,
        predictions=prediction,
    )
    per_seed.append({
        "prediction_path": path.resolve().as_posix(),
        "prediction_file_sha256": _sha(path),
        "checkpoint_sha256": prediction["checkpoint_sha256"],
        **result,
    })
  metric_names = sorted(per_seed[0]["metrics"])
  aggregate = {
      name: {
          "median": median(row["metrics"][name] for row in per_seed),
          "minimum": min(row["metrics"][name] for row in per_seed),
          "maximum": max(row["metrics"][name] for row in per_seed),
      }
      for name in metric_names
      if not name.endswith("_wilson95")
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "scope": "untouched_confirmation_private_join_after_frozen_predictions",
      "unseal_receipt_sha256": _sha(args.unseal_receipt),
      "query_count": public["query_count"],
      "primitive_supervised_edge_count": len(targets),
      "seed_count": len(per_seed),
      "per_seed": per_seed,
      "aggregate": aggregate,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
  main()


_evaluate_one = evaluate_confirmation_predictions_v3
