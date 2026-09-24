"""Adapt only LinkCAD's primitive-pair ranker to a generated source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from neurocad.linkcad_port_constraint_model_v9 import PortConstraintLinkCADV9
from neurocad.linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2
from neurocad.linkcad_source_adaptation_v1 import (
    SCHEMA_VERSION,
    configure_interface_ranker_adaptation_v1,
    interface_ranking_loss_v1,
    trainable_parameter_count,
)
from neurocad.tools.train_linkcad_factorized_pilot_v1 import _load_rows, evaluate


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-root", type=Path, required=True)
  parser.add_argument("--eval-root", type=Path, required=True)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=12)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--seed", type=int, default=4701)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()
  if args.epochs < 1 or args.learning_rate <= 0:
    raise ValueError("LinkCAD interface-adaptation configuration differs")
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  cache = LinkCADPrimitiveGraphCacheV2(args.primitive_cache)
  cache.load_all()
  train_rows, _train_counterfactual, train_interface = _load_rows(
      args.train_root, primitive_cache=cache,
  )
  eval_rows, eval_counterfactual, eval_interface = _load_rows(
      args.eval_root, primitive_cache=cache,
  )
  device = torch.device(args.device)
  model = PortConstraintLinkCADV9(
      hidden_dim=64, use_language=True, use_global=False,
      orbit_mode="attention", max_orbits_per_candidate=16, seed=args.seed,
  ).to(device)
  model.load_state_dict(torch.load(
      args.checkpoint, map_location=device, weights_only=True,
  ))
  parameters = configure_interface_ranker_adaptation_v1(model)
  optimizer = torch.optim.AdamW(
      parameters, lr=args.learning_rate, weight_decay=1e-4,
  )
  order = list(range(len(train_rows)))
  epoch_losses = []
  for epoch in range(args.epochs):
    random.Random(args.seed + epoch).shuffle(order)
    model.train()
    cumulative = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, index in enumerate(order, start=1):
      query = train_rows[index].to(device)
      loss = interface_ranking_loss_v1(
          model, query, train_interface.get(query.query_id, {}),
      )
      (loss / 8.0).backward()
      cumulative += float(loss.detach())
      if step % 8 == 0 or step == len(order):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    epoch_losses.append(cumulative / len(order))
  metrics = evaluate(
      model, eval_rows, eval_counterfactual, device=device, beam_size=25,
      interface_by_query=eval_interface,
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  checkpoint = args.output_dir / f"source_complete_seed{args.seed}.pt"
  torch.save(model.state_dict(), checkpoint)
  payload = {
      "schema_version": SCHEMA_VERSION,
      "scope": "development_generated_source_interface_ranker_adaptation",
      "source_checkpoint": args.checkpoint.resolve().as_posix(),
      "checkpoint": checkpoint.resolve().as_posix(),
      "train_query_count": len(train_rows),
      "eval_query_count": len(eval_rows),
      "epochs": args.epochs, "learning_rate": args.learning_rate,
      "seed": args.seed,
      "trainable_parameter_count": trainable_parameter_count(parameters),
      "total_parameter_count": sum(row.numel() for row in model.parameters()),
      "epoch_losses": epoch_losses, "metrics": metrics,
  }
  (args.output_dir / "summary.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
  main()
