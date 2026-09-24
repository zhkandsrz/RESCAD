"""Replace oracle-style port fields with frozen language-planner predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurocad.linkcad_factorized_model_v1 import EXECUTABLE_MOBILITY_NAMES
from neurocad.linkcad_language_contracts import render_mobility_instruction
from neurocad.linkcad_port_contract_v1 import (
    render_port_conditioned_instruction_v1,
)


def materialize_planner_conditioned_public_v1(
    *, public, inputs, predictions, linker_text_mode="canonical",
):
  if linker_text_mode not in {"canonical", "natural"}:
    raise ValueError("LinkCAD planner-conditioned linker text mode differs")
  instruction_by_key = {
      (row["query_id"], row["edge_id"]): row["instruction"]
      for row in inputs["items"]
  }
  predicted = {
      (row["query_id"], row["edge_id"]): row
      for row in predictions["items"]
  }
  specified = 0
  missing = 0
  queries = []
  for query in public["queries"]:
    edges = []
    for edge in query["functional_edges"]:
      key = (query["query_id"], edge["edge_id"])
      row = predicted.get(key)
      output = {
          key_name: value for key_name, value in edge.items()
          if key_name not in {
              "port_contract_a", "port_contract_b", "requested_mobility",
              "port_contract_status", "instruction",
          }
      }
      if row is not None and row["port_contract_status"] == "specified":
        output["port_contract_status"] = "specified"
        output["port_contract_a"] = row["port_contract_a"]
        output["port_contract_b"] = row["port_contract_b"]
        specified += 1
      else:
        output["port_contract_status"] = "unknown"
        missing += 1
      mobility = None if row is None else row.get("requested_mobility")
      output["requested_mobility"] = (
          mobility if mobility in EXECUTABLE_MOBILITY_NAMES else "unknown"
      )
      if linker_text_mode == "natural":
        output["instruction"] = instruction_by_key[key]
      elif mobility in EXECUTABLE_MOBILITY_NAMES:
        mobility_instruction = render_mobility_instruction(
            mobility, role_a=edge["role_a"], role_b=edge["role_b"], variant=0,
        )
        output["instruction"] = (
            render_port_conditioned_instruction_v1(
                role_a=edge["role_a"], role_b=edge["role_b"],
                port_a=row["port_contract_a"], port_b=row["port_contract_b"],
                mobility_instruction=mobility_instruction,
            )
            if row is not None and row["port_contract_status"] == "specified"
            else mobility_instruction
        )
      else:
        output["instruction"] = instruction_by_key[key]
      edges.append(output)
    queries.append({**query, "functional_edges": edges})
  return {
      **public,
      "scope": "frozen_language_planner_conditioned_candidate_set_inputs",
      "port_contract_source": "frozen_language_planner_predictions",
      "linker_text_mode": linker_text_mode,
      "planner_prediction_count": len(predicted),
      "planner_specified_edge_count": specified,
      "planner_unknown_or_missing_edge_count": missing,
      "queries": queries,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--public", type=Path, required=True)
  parser.add_argument("--inputs", type=Path, required=True)
  parser.add_argument("--predictions", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--linker-text-mode", choices=("canonical", "natural"),
      default="canonical",
  )
  args = parser.parse_args()
  load = lambda path: json.loads(path.read_text(encoding="utf-8"))
  result = materialize_planner_conditioned_public_v1(
      public=load(args.public), inputs=load(args.inputs),
      predictions=load(args.predictions), linker_text_mode=args.linker_text_mode,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "query_count": len(result["queries"]),
      "planner_prediction_count": result["planner_prediction_count"],
      "planner_specified_edge_count": result["planner_specified_edge_count"],
      "planner_unknown_or_missing_edge_count": result[
          "planner_unknown_or_missing_edge_count"
      ],
      "linker_text_mode": result["linker_text_mode"],
  }, sort_keys=True))


if __name__ == "__main__":
  main()
