"""Evaluate a frozen LinkCAD V2 checkpoint with a unified stagewise funnel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neurocad.linkcad_factorized_model_v1 import MOBILITY_NAMES, SUPPORT_NAMES
from neurocad.linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2
from neurocad.linkcad_primitive_factorized_model_v2 import (
    PrimitiveFactorizedLinkCADV2,
)
from neurocad.linkcad_port_conditioned_model_v6 import PortConditionedLinkCADV6
from neurocad.tools.train_linkcad_factorized_pilot_v1 import _load_rows, evaluate


SCHEMA_VERSION = "linkcad_stagewise_checkpoint_evaluation.v2"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--eval-root", type=Path, required=True)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--beam-size", type=int, default=25)
  parser.add_argument("--seed", type=int, default=1701)
  parser.add_argument(
      "--model-kind", choices=("primitive_v2", "port_v6"),
      default="primitive_v2",
  )
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


@torch.no_grad()
def stagewise(model, rows, interface, *, device, beam_size):
  counts = {
      "supervised_edge_count": 0,
      "target_part_pair_in_beam": 0,
      "target_primitives_generated_at_oracle_parts": 0,
      "target_primitive_pair_top1_at_oracle_parts": 0,
      "target_support_at_oracle_parts": 0,
      "target_mobility_at_oracle_parts": 0,
      "complete_primitive_query_count": 0,
      "assignment_support_mobility_recall_at_beam": 0,
      "full_primitive_top1_recall_at_beam": 0,
      "full_primitive_top8_recall_at_beam": 0,
  }
  for cpu_query in rows:
    query = cpu_query.to(device)
    output = model(query)
    beam = model.decode_beam(query, beam_size=beam_size)
    target_assignment = tuple(int(value) for value in query.target_assignment.tolist())
    target_mobility = tuple(
        MOBILITY_NAMES[int(value)] for value in query.target_mobility.tolist()
    )
    target_support = tuple(
        SUPPORT_NAMES[int(value)] for value in query.target_support.tolist()
    )
    query_interface = interface.get(query.query_id, {})
    complete = len(query_interface) == len(query.edge_ids)
    if complete:
      counts["complete_primitive_query_count"] += 1
    for edge_id, (target_a, target_b) in query_interface.items():
      counts["supervised_edge_count"] += 1
      edge_ordinal = query.edge_ids.index(edge_id)
      role_a, role_b = query.edge_index[edge_ordinal].tolist()
      candidate_a = target_assignment[role_a]
      candidate_b = target_assignment[role_b]
      counts["target_part_pair_in_beam"] += int(any(
          prediction.assignment[role_a] == candidate_a
          and prediction.assignment[role_b] == candidate_b
          for prediction in beam
      ))
      selected_a = output.selected_orbit_indices[edge_ordinal][0][candidate_a]
      selected_b = output.selected_orbit_indices[edge_ordinal][1][candidate_b]
      generated = target_a in selected_a and target_b in selected_b
      counts["target_primitives_generated_at_oracle_parts"] += int(generated)
      block = output.interface_score_blocks[edge_ordinal][candidate_a][candidate_b]
      if generated and block is not None:
        flat = int(block.reshape(-1).argmax())
        predicted = (
            selected_a[flat // int(block.shape[1])],
            selected_b[flat % int(block.shape[1])],
        )
        counts["target_primitive_pair_top1_at_oracle_parts"] += int(
            predicted == (target_a, target_b)
        )
      counts["target_mobility_at_oracle_parts"] += int(
          MOBILITY_NAMES[int(output.mobility_logits[
              edge_ordinal, candidate_a, candidate_b
          ].argmax())] == target_mobility[edge_ordinal]
      )
      counts["target_support_at_oracle_parts"] += int(
          SUPPORT_NAMES[int(output.support_logits[
              edge_ordinal, candidate_a, candidate_b
          ].argmax())] == target_support[edge_ordinal]
      )
    if not complete:
      continue
    support_mobility_hit = any(
        prediction.assignment == target_assignment
        and prediction.mobility == target_mobility
        and prediction.support == target_support
        for prediction in beam
    )
    counts["assignment_support_mobility_recall_at_beam"] += int(
        support_mobility_hit
    )
    top1_hit = False
    top8_hit = False
    for prediction in beam:
      if not (
          prediction.assignment == target_assignment
          and prediction.mobility == target_mobility
          and prediction.support == target_support
      ):
        continue
      top1_ok = True
      top8_ok = True
      for edge_ordinal, edge_id in enumerate(query.edge_ids):
        target_pair = query_interface[edge_id]
        top1_ok = top1_ok and prediction.interface_orbits[edge_ordinal] == target_pair
        top8_pairs = {
            (int(row[0]), int(row[1]))
            for row in prediction.interface_alternatives[edge_ordinal]
        }
        top8_ok = top8_ok and target_pair in top8_pairs
      top1_hit = top1_hit or top1_ok
      top8_hit = top8_hit or top8_ok
    counts["full_primitive_top1_recall_at_beam"] += int(top1_hit)
    counts["full_primitive_top8_recall_at_beam"] += int(top8_hit)
  edges = counts["supervised_edge_count"]
  queries = counts["complete_primitive_query_count"]
  rates = {
      key + "_rate": value / edges
      for key, value in counts.items()
      if key in {
          "target_part_pair_in_beam",
          "target_primitives_generated_at_oracle_parts",
          "target_primitive_pair_top1_at_oracle_parts",
          "target_support_at_oracle_parts",
          "target_mobility_at_oracle_parts",
      }
  }
  rates.update({
      key + "_rate": value / queries if queries else 0.0
      for key, value in counts.items()
      if key in {
          "assignment_support_mobility_recall_at_beam",
          "full_primitive_top1_recall_at_beam",
          "full_primitive_top8_recall_at_beam",
      }
  })
  return {**counts, **rates}


def main() -> None:
  args = _parse_args()
  cache = LinkCADPrimitiveGraphCacheV2(args.primitive_cache)
  cache.load_all()
  rows, counterfactual, interface = _load_rows(
      args.eval_root, primitive_cache=cache
  )
  device = torch.device(args.device)
  model_class = {
      "primitive_v2": PrimitiveFactorizedLinkCADV2,
      "port_v6": PortConditionedLinkCADV6,
  }[args.model_kind]
  model = model_class(
      hidden_dim=64, use_language=True, use_global=False,
      orbit_mode="attention", max_orbits_per_candidate=16, seed=args.seed,
  ).to(device)
  model.load_state_dict(
      torch.load(args.checkpoint, map_location=device, weights_only=True)
  )
  standard = evaluate(
      model, rows, counterfactual, device=device,
      beam_size=args.beam_size, interface_by_query=interface,
  )
  funnel = stagewise(
      model, rows, interface, device=device, beam_size=args.beam_size,
  )
  payload = {
      "schema_version": SCHEMA_VERSION,
      "scope": "development_checkpoint_stagewise_evaluation",
      "checkpoint": args.checkpoint.resolve().as_posix(),
      "query_count": len(rows),
      "beam_size": args.beam_size,
      "model_kind": args.model_kind,
      "standard_metrics": standard,
      "stagewise": funnel,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
  main()
