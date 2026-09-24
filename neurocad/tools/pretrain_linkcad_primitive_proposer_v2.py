"""Pretrain LinkCAD V2's language-conditioned face-or-edge proposer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from neurocad.linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2
from neurocad.linkcad_primitive_cache_v3 import LinkCADPrimitiveGraphCacheV3
from neurocad.linkcad_primitive_factorized_model_v2 import (
    PrimitiveFactorizedLinkCADV2,
)
from neurocad.linkcad_primitive_factorized_model_v3 import (
    PrimitiveFactorizedLinkCADV3,
)
from neurocad.linkcad_primitive_factorized_model_v4 import (
    PrimitiveFactorizedLinkCADV4,
)
from neurocad.tools.train_linkcad_factorized_pilot_v1 import _load_rows


SCHEMA_VERSION = "linkcad_primitive_proposer_pretraining.v2"


def _samples(rows, interface_by_query):
  result = []
  for query in rows:
    for edge_id, (primitive_a, primitive_b) in interface_by_query.get(
        query.query_id, {}
    ).items():
      edge_ordinal = query.edge_ids.index(edge_id)
      role_a, role_b = query.edge_index[edge_ordinal].tolist()
      for role, target in ((role_a, primitive_a), (role_b, primitive_b)):
        candidate = int(query.target_assignment[role])
        graph = query.candidate_graphs[role][candidate]
        if graph is None or not 0 <= target < int(graph.primitive_features.shape[0]):
          continue
        result.append((graph, query.edge_text_features[edge_ordinal], target))
  return result


def _proposal_logits(model, primitive_tensor, language_features):
  language = model.text_encoder(language_features[None, :])[0]
  if not model.use_language:
    language = torch.zeros_like(language)
  expanded = language[None, :].expand_as(primitive_tensor)
  return model.orbit_proposal(torch.cat((
      primitive_tensor,
      expanded,
      primitive_tensor * expanded,
      torch.abs(primitive_tensor - expanded),
  ), dim=-1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, samples, *, device, top_k):
  model.eval()
  top1 = topk = 0
  for graph, language, target in samples:
    _parts, primitive_sets = model._encode_graphs((graph,))
    logits = _proposal_logits(model, primitive_sets[0], language.to(device))
    order = torch.topk(logits, k=min(top_k, int(logits.shape[0]))).indices.tolist()
    top1 += int(order[0] == target)
    topk += int(target in order)
  count = len(samples)
  return {
      "endpoint_count": count,
      "top1_accuracy": top1 / count if count else 0.0,
      "topk_coverage": topk / count if count else 0.0,
      "top_k": top_k,
  }


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-root", type=Path, required=True)
  parser.add_argument("--eval-root", type=Path, required=True)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=12)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--top-k", type=int, default=16)
  parser.add_argument("--seed", type=int, default=1701)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--no-language", action="store_true")
  parser.add_argument(
      "--model-kind", choices=("primitive_v2", "primitive_v3", "primitive_v4"),
      default="primitive_v2",
  )
  parser.add_argument(
      "--primitive-cache-version", choices=("v2", "v3"), default="v2"
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  expected_cache_version = "v3" if args.model_kind == "primitive_v4" else "v2"
  if args.primitive_cache_version != expected_cache_version:
    raise ValueError("LinkCAD primitive model/cache feature versions differ")
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  cache_class = (
      LinkCADPrimitiveGraphCacheV3
      if args.primitive_cache_version == "v3"
      else LinkCADPrimitiveGraphCacheV2
  )
  cache = cache_class(args.primitive_cache)
  cache.load_all()
  train_rows, _counterfactual, train_interface = _load_rows(
      args.train_root, primitive_cache=cache
  )
  eval_rows, _eval_counterfactual, eval_interface = _load_rows(
      args.eval_root, primitive_cache=cache
  )
  train_samples = _samples(train_rows, train_interface)
  eval_samples = _samples(eval_rows, eval_interface)
  device = torch.device(args.device)
  model_class = {
      "primitive_v2": PrimitiveFactorizedLinkCADV2,
      "primitive_v3": PrimitiveFactorizedLinkCADV3,
      "primitive_v4": PrimitiveFactorizedLinkCADV4,
  }[args.model_kind]
  model = model_class(
      hidden_dim=64,
      use_language=not args.no_language,
      use_global=False,
      orbit_mode="attention",
      max_orbits_per_candidate=args.top_k,
      seed=args.seed,
  ).to(device)
  parameters = [
      *model.brep_encoder.parameters(),
      *model.primitive_encoder.parameters(),
      *model.primitive_norm.parameters(),
      *model.text_encoder.parameters(),
      *model.orbit_proposal.parameters(),
  ]
  if args.model_kind == "primitive_v3":
    parameters.extend(model.part_primitive_fusion.parameters())
  optimizer = torch.optim.AdamW(
      parameters, lr=args.learning_rate, weight_decay=1e-4
  )
  order = list(range(len(train_samples)))
  epoch_losses = []
  for epoch in range(args.epochs):
    random.Random(args.seed + epoch).shuffle(order)
    cumulative = 0.0
    model.train()
    for start in range(0, len(order), args.batch_size):
      batch = [train_samples[index] for index in order[start:start + args.batch_size]]
      graphs = tuple(row[0] for row in batch)
      _parts, primitive_sets = model._encode_graphs(graphs)
      losses = []
      for primitives, (_graph, language, target) in zip(
          primitive_sets, batch, strict=True
      ):
        logits = _proposal_logits(model, primitives, language.to(device))
        losses.append(F.cross_entropy(
            logits[None, :],
            torch.tensor([target], dtype=torch.long, device=device),
        ))
      loss = torch.stack(losses).mean()
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
      optimizer.step()
      cumulative += float(loss.detach()) * len(batch)
    epoch_losses.append(cumulative / len(train_samples))
  metrics = evaluate(model, eval_samples, device=device, top_k=args.top_k)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  checkpoint = args.output_dir / f"primitive_proposer_seed{args.seed}.pt"
  torch.save(model.state_dict(), checkpoint)
  result: dict[str, Any] = {
      "schema_version": SCHEMA_VERSION,
      "scope": "development_face_or_edge_primitive_proposer_pretraining",
      "train_endpoint_count": len(train_samples),
      "eval_endpoint_count": len(eval_samples),
      "epochs": args.epochs,
      "batch_size": args.batch_size,
      "learning_rate": args.learning_rate,
      "seed": args.seed,
      "epoch_losses": epoch_losses,
      "metrics": metrics,
      "checkpoint": checkpoint.resolve().as_posix(),
      "use_language": not args.no_language,
      "model_kind": args.model_kind,
      "primitive_cache_version": args.primitive_cache_version,
  }
  (args.output_dir / "summary.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
