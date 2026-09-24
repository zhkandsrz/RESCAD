"""Join a preunseal LinkCAD joint-execution top choice with private targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from neurocad.linkcad_joint_execution_evaluation_v1 import (
    evaluate_joint_execution_selection_v1,
)


def _read(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--private-targets", type=Path, required=True)
  parser.add_argument("--primitive-supervision", type=Path, required=True)
  parser.add_argument("--unseal-receipt", type=Path, required=True)
  parser.add_argument("--rerank-preunseal-receipt", type=Path, required=True)
  parser.add_argument("--execution-receipt", type=Path, required=True)
  parser.add_argument("--reranked-predictions", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  unseal = _read(args.unseal_receipt)
  preunseal = _read(args.rerank_preunseal_receipt)
  execution_hash = _sha(args.execution_receipt)
  reranked_hash = _sha(args.reranked_predictions)
  bound = next((
      row for row in preunseal["rows"]
      if row["execution_receipt_file_sha256"] == execution_hash
      and row["reranked_prediction_file_sha256"] == reranked_hash
  ), None)
  if (
      bound is None
      or unseal.get("private_targets_opened_after_prediction_closure") is not True
      or preunseal.get("prediction_precommit_sha256")
      != unseal.get("prediction_precommit_sha256")
      or _sha(args.private_targets) != unseal.get("private_target_sha256")
      or _sha(args.primitive_supervision)
      != unseal.get("primitive_supervision_sha256")
  ):
    raise ValueError("LinkCAD joint execution private join binding differs")
  payload = evaluate_joint_execution_selection_v1(
      public=_read(args.public),
      private_targets=_read(args.private_targets),
      primitive_supervision=_read(args.primitive_supervision),
      execution_receipt=_read(args.execution_receipt),
      reranked_predictions=_read(args.reranked_predictions),
  )
  payload.update({
      "seed": int(bound["seed"]),
      "execution_receipt_sha256": execution_hash,
      "reranked_prediction_sha256": reranked_hash,
      "unseal_receipt_sha256": _sha(args.unseal_receipt),
      "rerank_preunseal_receipt_sha256": _sha(
          args.rerank_preunseal_receipt
      ),
  })
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
  main()
