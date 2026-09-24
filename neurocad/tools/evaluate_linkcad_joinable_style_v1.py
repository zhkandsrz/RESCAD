"""Evaluate frozen JoinABLe-style checkpoints with LinkCAD's formal metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import torch

from neurocad.linkcad_factorized_model_v1 import (
    LinkCADPrediction,
    LinkCADQueryTensors,
    tensorize_linkcad_query,
)
from neurocad.linkcad_joinable_style_baseline_v1 import (
    PREDICTION_SCHEMA_VERSION,
    JoinABLeStyleLinkCADV1,
)
from neurocad.linkcad_primitive_cache_v2 import (
    LinkCADPrimitiveGraphCacheV2,
    attach_primitive_graphs_v2,
)
from neurocad.tools.evaluate_linkcad_confirmation_v2 import (
    _primitive_targets,
    evaluate_confirmation_predictions_v3,
)


SCHEMA_VERSION = "linkcad_joinable_style_evaluation.v2"


def resolve_primitive_supervision_path_v1(eval_root: Path) -> Path:
  """Resolve the two supported confirmation-supervision layouts."""

  preferred = eval_root / "direct_primitive_orbit_supervision_v2_geometry.json"
  if preferred.is_file():
    return preferred
  fallback = eval_root / "primitive_supervision.json"
  if fallback.is_file():
    return fallback
  raise ValueError("JoinABLe-style primitive supervision is unavailable")


def public_query_tensors_v1(
    public: Mapping[str, Any], cache: LinkCADPrimitiveGraphCacheV2,
) -> list[LinkCADQueryTensors]:
  """Tensorize predictions from public inputs without opening target labels."""

  if public.get("contains_private_targets") is not False:
    raise ValueError("JoinABLe-style public prediction scope differs")
  return [
      attach_primitive_graphs_v2(
          tensorize_linkcad_query(public_query), public_query, cache,
      )
      for public_query in public["queries"]
  ]


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _payload_sha256(value: Any) -> str:
  encoded = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--eval-root", type=Path, required=True)
  parser.add_argument("--primitive-cache", type=Path, required=True)
  parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
  parser.add_argument("--seeds", type=int, nargs="+", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--beam-size", type=int, default=25)
  parser.add_argument("--max-orbits-per-candidate", type=int, default=16)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument(
      "--scope",
      default="post_hoc_external_baseline_on_previously_opened_confirmation",
  )
  return parser.parse_args()


def joinable_style_prediction_row_v2(
    query: LinkCADQueryTensors,
    beam: Sequence[LinkCADPrediction],
    *,
    role_ids: Sequence[str],
) -> dict[str, Any]:
  """Convert a decoded beam to the frozen confirmation-prediction interface."""

  if len(role_ids) != query.role_count or len(set(role_ids)) != len(role_ids):
    raise ValueError("JoinABLe-style role identities differ")
  hypotheses = []
  for rank, prediction in enumerate(beam):
    if len(prediction.candidate_ids) != len(role_ids):
      raise ValueError("JoinABLe-style candidate assignment differs")
    if not (
        len(prediction.mobility)
        == len(prediction.support)
        == len(prediction.interface_orbits)
        == len(prediction.interface_alternatives)
        == len(query.edge_ids)
    ):
      raise ValueError("JoinABLe-style edge program domain differs")
    edge_programs = []
    for edge_ordinal, edge_id in enumerate(query.edge_ids):
      primitive_a, primitive_b = prediction.interface_orbits[edge_ordinal]
      alternatives = [
          {
              "rank": alternative_rank,
              "interface_primitive_a": int(alternative_a),
              "interface_primitive_b": int(alternative_b),
              "model_score": float(alternative_score),
          }
          for alternative_rank, (
              alternative_a, alternative_b, alternative_score,
          ) in enumerate(
              prediction.interface_alternatives[edge_ordinal], start=1
          )
      ]
      edge_programs.append({
          "edge_id": edge_id,
          "mobility": prediction.mobility[edge_ordinal],
          "support_family": prediction.support[edge_ordinal],
          "interface_primitive_a": int(primitive_a),
          "interface_primitive_b": int(primitive_b),
          "primitive_alternatives": alternatives,
      })
    hypotheses.append({
        "rank": rank,
        "score": float(prediction.score),
        "candidate_by_role": {
            str(role_id): str(candidate_id)
            for role_id, candidate_id in zip(
                role_ids, prediction.candidate_ids, strict=True
            )
        },
        "edge_programs": edge_programs,
    })
  return {
      "query_id": query.query_id,
      "terminal_status": "predicted" if hypotheses else "no_valid_hypothesis",
      "hypotheses": hypotheses,
  }


@torch.no_grad()
def _predict(
    model: JoinABLeStyleLinkCADV1,
    rows: Sequence[LinkCADQueryTensors],
    *,
    role_ids_by_query: dict[str, tuple[str, ...]],
    device: torch.device,
    beam_size: int,
) -> list[dict[str, Any]]:
  model.eval()
  prediction_rows = []
  for cpu_query in rows:
    query = cpu_query.to(device)
    beam = model.decode_beam(
        query, beam_size=beam_size, interface_alternatives_per_edge=8
    )
    prediction_rows.append(joinable_style_prediction_row_v2(
        cpu_query,
        beam,
        role_ids=role_ids_by_query[cpu_query.query_id],
    ))
  return prediction_rows


def main() -> None:
  args = _parse_args()
  if len(args.checkpoints) != len(args.seeds) or len(set(args.seeds)) != len(args.seeds):
    raise ValueError("JoinABLe-style checkpoint/seed domain differs")
  cache = LinkCADPrimitiveGraphCacheV2(args.primitive_cache)
  cache.load_all()
  public = json.loads((args.eval_root / "public.json").read_text(encoding="utf-8"))
  private = json.loads(
      (args.eval_root / "private_targets.sealed.json").read_text(encoding="utf-8")
  )
  supervision_path = resolve_primitive_supervision_path_v1(args.eval_root)
  supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
  primitive_targets = _primitive_targets(supervision)
  total_query_count = len(public["queries"])
  role_ids_by_query = {
      row["query_id"]: tuple(str(role["role_id"]) for role in row["roles"])
      for row in public["queries"]
  }
  rows = public_query_tensors_v1(public, cache)
  supported_query_count = len(rows)
  if not 0 < supported_query_count <= total_query_count:
    raise ValueError("JoinABLe-style supported query accounting differs")
  if any(row.query_id not in role_ids_by_query for row in rows):
    raise ValueError("JoinABLe-style public query identities differ")
  device = torch.device(args.device)
  runs = []
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for checkpoint, seed in zip(args.checkpoints, args.seeds, strict=True):
    model = JoinABLeStyleLinkCADV1(
        hidden_dim=64,
        use_language=False,
        use_global=False,
        max_orbits_per_candidate=args.max_orbits_per_candidate,
        seed=seed,
    ).to(device)
    model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=True
    ))
    checkpoint_sha256 = _file_sha256(checkpoint)
    prediction_rows = _predict(
        model,
        rows,
        role_ids_by_query=role_ids_by_query,
        device=device,
        beam_size=args.beam_size,
    )
    prediction_payload = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "scope": args.scope,
        "method": "JoinABLe-style (adapted)",
        "private_targets_opened": False,
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_seed": seed,
        "beam_size": args.beam_size,
        "interface_alternatives_per_edge": 8,
        "query_count": supported_query_count,
        "total_query_count": total_query_count,
        "rows": prediction_rows,
    }
    prediction_payload["prediction_payload_sha256"] = _payload_sha256(
        prediction_payload
    )
    prediction_path = args.output.parent / (
        f"joinable_style_seed{seed}.predictions.json"
    )
    prediction_path.write_text(
        json.dumps(prediction_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = evaluate_confirmation_predictions_v3(
        public=public,
        private=private,
        primitive_targets=primitive_targets,
        predictions=prediction_payload,
    )
    runs.append({
        "optimizer_seed": seed,
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_path": prediction_path.resolve().as_posix(),
        "prediction_sha256": _file_sha256(prediction_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **result,
    })
  metric_names = sorted(
      name for name in runs[0]["metrics"] if not name.endswith("_wilson95")
  )
  aggregate = {
      name: {
          "median": statistics.median(
              float(run["metrics"][name]) for run in runs
          ),
          "minimum": min(float(run["metrics"][name]) for run in runs),
          "maximum": max(float(run["metrics"][name]) for run in runs),
      }
      for name in metric_names
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "method": "JoinABLe-style (adapted)",
      "scope": args.scope,
      "native_joinable_code_or_checkpoint_reproduction": False,
      "language_conditioned": False,
      "same_public_port_filter": True,
      "formal_confirmation_claim": False,
      "beam_size": args.beam_size,
      "max_orbits_per_candidate": args.max_orbits_per_candidate,
      "query_count": total_query_count,
      "supported_query_count": supported_query_count,
      "unsupported_query_count": total_query_count - supported_query_count,
      "primitive_supervised_edge_count": len(primitive_targets),
      "seed_count": len(runs),
      "runs": runs,
      "aggregate": aggregate,
  }
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "method": payload["method"],
      "scope": payload["scope"],
      "query_count": payload["query_count"],
      "supported_query_count": payload["supported_query_count"],
      "aggregate": aggregate,
  }, sort_keys=True))


if __name__ == "__main__":
  main()
