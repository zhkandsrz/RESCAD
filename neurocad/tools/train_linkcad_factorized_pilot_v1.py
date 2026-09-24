"""Train and evaluate the fast factorized LinkCAD protocol smoke model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from neurocad.linkcad_factorized_model_v1 import (
    FactorizedLinkCADV1,
    LinkCADQueryTensors,
    MOBILITY_NAMES,
    SUPPORT_NAMES,
    frozen_text_features,
    linkcad_training_loss,
    tensorize_linkcad_query,
)
from neurocad.linkcad_brep_cache_v1 import (
    LinkCADBRepGraphCacheV1,
    attach_brep_graphs,
)
from neurocad.linkcad_brep_factorized_model_v1 import BRepFactorizedLinkCADV1
from neurocad.linkcad_primitive_cache_v2 import (
    LinkCADPrimitiveGraphCacheV2,
    attach_primitive_graphs_v2,
)
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
from neurocad.linkcad_joinable_style_baseline_v1 import JoinABLeStyleLinkCADV1
from neurocad.linkcad_external_brep_baselines_v1 import (
    AutoMateStyleLinkCADV1,
    PointerCADStyleLinkCADV1,
)
from neurocad.linkcad_port_conditioned_model_v6 import PortConditionedLinkCADV6
from neurocad.linkcad_port_constraint_model_v9 import PortConstraintLinkCADV9


RESULT_SCHEMA_VERSION = "linkcad_factorized_pilot_result.v1"


def _load_rows(
    root: Path,
    *,
    brep_cache: LinkCADBRepGraphCacheV1 | None = None,
    primitive_cache: LinkCADPrimitiveGraphCacheV2 | None = None,
    filter_unsupported_mobility: bool = False,
) -> tuple[
    list[LinkCADQueryTensors],
    dict[str, Any],
    dict[str, dict[str, tuple[int, int]]],
]:
  public = json.loads((root / "public.json").read_text(encoding="utf-8"))
  private_path = root / "private_targets.json"
  if not private_path.exists():
    private_path = root / "private_targets.sealed.json"
  private = json.loads(private_path.read_text(encoding="utf-8"))
  counterfactual_path = root / "counterfactual.json"
  counterfactual = (
      json.loads(counterfactual_path.read_text(encoding="utf-8"))
      if counterfactual_path.exists() else {"pairs": []}
  )
  private_by_id = {row["query_id"]: row for row in private["targets"]}
  rows = []
  for row in public["queries"]:
    target = private_by_id[row["query_id"]]
    target_mobility = {
        str(edge["target_mobility"])
        for edge in target["functional_edge_targets"]
    }
    if filter_unsupported_mobility and not target_mobility <= set(MOBILITY_NAMES):
      continue
    query = tensorize_linkcad_query(row, target)
    if brep_cache is not None:
      query = attach_brep_graphs(query, row, brep_cache)
    elif primitive_cache is not None:
      query = attach_primitive_graphs_v2(query, row, primitive_cache)
    rows.append(query)
  counterfactual_by_id = {
      row["query_id"]: row for row in counterfactual["pairs"]
  }
  interface_by_query: dict[str, dict[str, tuple[int, int]]] = {}
  interface_path = root / "direct_primitive_orbit_supervision_v2_geometry.json"
  if not interface_path.exists():
    interface_path = root / "direct_face_orbit_supervision.json"
  if not interface_path.exists():
    interface_path = root / "primitive_supervision.json"
  if interface_path.exists():
    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    mapped: dict[tuple[str, str], dict[str, int]] = {}
    for row in interface["rows"]:
      if row["status"] == "mapped_type_consistent":
        mapped.setdefault((row["query_id"], row["edge_id"]), {})[
            row["side"]
        ] = int(row.get(
            "primitive_orbit_ordinal", row["structural_orbit_ordinal"]
        ))
    for (query_id, edge_id), sides in mapped.items():
      if set(sides) == {"a", "b"}:
        interface_by_query.setdefault(query_id, {})[edge_id] = (
            sides["a"], sides["b"]
        )
  return rows, counterfactual_by_id, interface_by_query


def _counterfactual_query(
    query: LinkCADQueryTensors,
    pair: Mapping[str, Any],
) -> LinkCADQueryTensors:
  edge_id = str(pair["counterfactual_edge"]["edge_id"])
  edge_ordinal = query.edge_ids.index(edge_id)
  edge_text = query.edge_text_features.clone()
  edge_text[edge_ordinal] = frozen_text_features(
      str(pair["counterfactual_edge"]["instruction"])
  )
  mobility = query.target_mobility.clone()
  mobility[edge_ordinal] = MOBILITY_NAMES.index(
      str(pair["counterfactual_target_mobility"])
  )
  return LinkCADQueryTensors(
      query_id=query.query_id,
      candidate_features=query.candidate_features,
      role_text_features=query.role_text_features,
      edge_index=query.edge_index,
      edge_text_features=edge_text,
      candidate_mask=query.candidate_mask,
      target_assignment=query.target_assignment,
      target_mobility=mobility,
      target_support=query.target_support,
      candidate_ids=query.candidate_ids,
      edge_ids=query.edge_ids,
      candidate_graphs=query.candidate_graphs,
      port_candidate_mask=query.port_candidate_mask,
      requested_mobility=(
          tuple(
              str(pair["counterfactual_target_mobility"])
              if ordinal == edge_ordinal else value
              for ordinal, value in enumerate(query.requested_mobility)
          )
          if query.requested_mobility else ()
      ),
  )


def _counterfactual_loss(
    model: FactorizedLinkCADV1,
    query: LinkCADQueryTensors,
    pair: Mapping[str, Any],
) -> torch.Tensor:
  changed = _counterfactual_query(query, pair)
  output = model(changed)
  edge_id = str(pair["counterfactual_edge"]["edge_id"])
  edge_ordinal = changed.edge_ids.index(edge_id)
  role_a, role_b = changed.edge_index[edge_ordinal].tolist()
  candidate_a = changed.target_assignment[role_a]
  candidate_b = changed.target_assignment[role_b]
  return F.cross_entropy(
      output.mobility_logits[edge_ordinal, candidate_a, candidate_b][None, :],
      changed.target_mobility[edge_ordinal][None],
  )


def _independent_role_beam(output, query, *, beam_size: int):
  """Matched-budget baseline using edge max-marginals but no joint pair factors."""

  role_scores = output.unary_logits.clone()
  degree = torch.zeros(query.role_count, device=role_scores.device)
  for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
    pair = output.edge_pair_logits[edge_ordinal]
    role_scores[role_a] = role_scores[role_a] + pair.max(dim=1).values
    role_scores[role_b] = role_scores[role_b] + pair.max(dim=0).values
    degree[role_a] += 1
    degree[role_b] += 1
  role_scores = role_scores / degree.clamp_min(1.0)[:, None]
  beams = [((), 0.0)]
  for role in range(query.role_count):
    expanded = []
    for prefix, score in beams:
      for candidate in range(query.candidate_count):
        value = float(role_scores[role, candidate])
        if not np.isfinite(value):
          continue
        expanded.append((prefix + (candidate,), score + value))
    expanded.sort(key=lambda row: (-row[1], row[0]))
    beams = expanded[:beam_size]
  return [assignment for assignment, _score in beams]


@torch.no_grad()
def evaluate(
    model: FactorizedLinkCADV1,
    rows: list[LinkCADQueryTensors],
    counterfactual_by_id: Mapping[str, Any],
    *,
    device: torch.device,
    beam_size: int,
    interface_by_query: Mapping[str, Mapping[str, tuple[int, int]]],
) -> dict[str, Any]:
  model.eval()
  role_correct = 0
  role_total = 0
  strict_parts = 0
  edge_pair_correct = 0
  edge_total = 0
  oracle_mobility_correct = 0
  predicted_mobility_correct = 0
  joint_strict = 0
  assignment_recall_at_beam = 0
  joint_recall_at_beam = 0
  edge_pair_recall_at_beam = 0
  counterfactual_correct = 0
  counterfactual_changed = 0
  executed_counterfactual_correct = 0
  executed_counterfactual_changed = 0
  counterfactual_count = 0
  interface_generator_covered = 0
  interface_correct = 0
  interface_total = 0
  support_edge_total = 0
  oracle_support_correct = 0
  predicted_support_correct = 0
  strict_program = 0
  program_recall_at_beam = 0
  local_edge_pair_correct = 0
  local_mobility_correct = 0
  local_consistent_queries = 0
  local_strict_parts = 0
  local_strict_joint = 0
  marginal_assignment_recall_at_beam = 0
  marginal_joint_recall_at_beam = 0
  for cpu_query in rows:
    query = cpu_query.to(device)
    output = model(query)
    prediction_beam = model.decode_beam(query, beam_size=beam_size)
    prediction = prediction_beam[0]
    target_assignment = tuple(int(value) for value in query.target_assignment.tolist())
    target_mobility = tuple(
        MOBILITY_NAMES[int(value)] for value in query.target_mobility.tolist()
    )
    target_support = (
        ()
        if query.target_support is None
        else tuple(SUPPORT_NAMES[int(value)] for value in query.target_support.tolist())
    )
    target_available = all(
        query.candidate_mask is None
        or bool(query.candidate_mask[role, target])
        for role, target in enumerate(target_assignment)
    )
    local_role_choices: dict[int, list[int]] = {
        role: [] for role in range(query.role_count)
    }
    marginal_beam = _independent_role_beam(
        output, query, beam_size=beam_size
    )
    marginal_assignment_recall_at_beam += int(
        target_available and target_assignment in marginal_beam
    )
    marginal_joint_hit = False
    if target_available and target_assignment in marginal_beam:
      marginal_mobility = []
      for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
        logits = output.mobility_logits[
            edge_ordinal,
            target_assignment[role_a],
            target_assignment[role_b],
        ]
        marginal_mobility.append(MOBILITY_NAMES[int(logits.argmax())])
      marginal_joint_hit = tuple(marginal_mobility) == target_mobility
    marginal_joint_recall_at_beam += int(marginal_joint_hit)
    local_mobility = []
    for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
      flat = int(output.edge_pair_logits[edge_ordinal].reshape(-1).argmax())
      candidate_a = flat // query.candidate_count
      candidate_b = flat % query.candidate_count
      local_role_choices[role_a].append(candidate_a)
      local_role_choices[role_b].append(candidate_b)
      local_edge_pair_correct += int(
          candidate_a == target_assignment[role_a]
          and candidate_b == target_assignment[role_b]
      )
      mobility_index = int(output.mobility_logits[
          edge_ordinal, candidate_a, candidate_b
      ].argmax())
      mobility_name = MOBILITY_NAMES[mobility_index]
      local_mobility.append(mobility_name)
      local_mobility_correct += int(
          mobility_name == target_mobility[edge_ordinal]
      )
    local_consistent = all(
        choices and len(set(choices)) == 1
        for choices in local_role_choices.values()
    )
    local_consistent_queries += int(local_consistent)
    if local_consistent:
      local_assignment = tuple(
          choices[0] for _, choices in sorted(local_role_choices.items())
      )
      local_parts_exact = target_available and local_assignment == target_assignment
      local_strict_parts += int(local_parts_exact)
      local_strict_joint += int(
          local_parts_exact and tuple(local_mobility) == target_mobility
      )
    for edge_id, (orbit_a, orbit_b) in interface_by_query.get(
        cpu_query.query_id, {}
    ).items():
      interface_total += 1
      if output.interface_score_blocks is None:
        continue
      edge_ordinal = query.edge_ids.index(edge_id)
      role_a, role_b = query.edge_index[edge_ordinal].tolist()
      candidate_a = target_assignment[role_a]
      candidate_b = target_assignment[role_b]
      block = output.interface_score_blocks[edge_ordinal][candidate_a][candidate_b]
      selected_a = output.selected_orbit_indices[edge_ordinal][0][candidate_a]
      selected_b = output.selected_orbit_indices[edge_ordinal][1][candidate_b]
      if (
          block is None
          or orbit_a not in selected_a
          or orbit_b not in selected_b
      ):
        continue
      interface_generator_covered += 1
      predicted_flat = int(block.reshape(-1).argmax())
      predicted_local = (
          predicted_flat // int(block.shape[1]),
          predicted_flat % int(block.shape[1]),
      )
      predicted_pair = (
          selected_a[predicted_local[0]],
          selected_b[predicted_local[1]],
      )
      interface_correct += int(predicted_pair == (orbit_a, orbit_b))
    part_matches = [
        predicted == target
        for predicted, target in zip(prediction.assignment, target_assignment)
    ]
    assignment_recall_at_beam += int(
        target_available
        and any(row.assignment == target_assignment for row in prediction_beam)
    )
    joint_recall_at_beam += int(
        target_available
        and any(
            row.assignment == target_assignment
            and tuple(row.mobility) == target_mobility
            for row in prediction_beam
        )
    )
    if output.support_logits is not None and target_support:
      program_recall_at_beam += int(
          target_available
          and any(
              row.assignment == target_assignment
              and tuple(row.mobility) == target_mobility
              and tuple(row.support) == target_support
              for row in prediction_beam
          )
      )
    if not target_available:
      part_matches = [False for _ in part_matches]
    role_correct += sum(part_matches)
    role_total += len(part_matches)
    parts_exact = all(part_matches)
    strict_parts += int(parts_exact)
    for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
      edge_pair_correct += int(
          prediction.assignment[role_a] == target_assignment[role_a]
          and prediction.assignment[role_b] == target_assignment[role_b]
      )
      edge_pair_recall_at_beam += int(
          target_available
          and any(
              row.assignment[role_a] == target_assignment[role_a]
              and row.assignment[role_b] == target_assignment[role_b]
              for row in prediction_beam
          )
      )
      edge_total += 1
      a = target_assignment[role_a]
      b = target_assignment[role_b]
      edge_target_available = (
          query.candidate_mask is None
          or (
              bool(query.candidate_mask[role_a, a])
              and bool(query.candidate_mask[role_b, b])
          )
      )
      if edge_target_available:
        oracle_prediction = int(output.mobility_logits[edge_ordinal, a, b].argmax())
        oracle_mobility_correct += int(
            MOBILITY_NAMES[oracle_prediction] == target_mobility[edge_ordinal]
        )
      predicted_mobility_correct += int(
          prediction.mobility[edge_ordinal] == target_mobility[edge_ordinal]
      )
      if output.support_logits is not None and target_support:
        support_edge_total += 1
        if edge_target_available:
          oracle_support = int(
              output.support_logits[edge_ordinal, a, b].argmax()
          )
          oracle_support_correct += int(
              SUPPORT_NAMES[oracle_support] == target_support[edge_ordinal]
          )
        predicted_support_correct += int(
            prediction.support[edge_ordinal] == target_support[edge_ordinal]
        )
    joint_strict += int(
        parts_exact and tuple(prediction.mobility) == target_mobility
    )
    strict_program += int(
        bool(target_support)
        and parts_exact
        and tuple(prediction.mobility) == target_mobility
        and tuple(prediction.support) == target_support
    )

    pair = counterfactual_by_id.get(cpu_query.query_id)
    if pair is None:
      continue
    counterfactual_count += 1
    changed = _counterfactual_query(cpu_query, pair).to(device)
    changed_output = model(changed)
    changed_edge_id = str(pair["counterfactual_edge"]["edge_id"])
    edge_ordinal = changed.edge_ids.index(changed_edge_id)
    role_a, role_b = changed.edge_index[edge_ordinal].tolist()
    a = int(changed.target_assignment[role_a])
    b = int(changed.target_assignment[role_b])
    changed_target_available = (
        changed.candidate_mask is None
        or (
            bool(changed.candidate_mask[role_a, a])
            and bool(changed.candidate_mask[role_b, b])
        )
    )
    changed_index = int(changed_output.mobility_logits[edge_ordinal, a, b].argmax())
    changed_name = MOBILITY_NAMES[changed_index]
    counterfactual_correct += int(
        changed_target_available
        and changed_name == pair["counterfactual_target_mobility"]
    )
    original_index = int(output.mobility_logits[edge_ordinal, a, b].argmax())
    counterfactual_changed += int(changed_index != original_index)
    changed_executed = model._decode_mobility(
        changed,
        edge_ordinal,
        changed_output.mobility_logits[edge_ordinal, a, b],
    )
    original_executed = model._decode_mobility(
        query,
        edge_ordinal,
        output.mobility_logits[edge_ordinal, a, b],
    )
    executed_counterfactual_correct += int(
        changed_target_available
        and changed_executed == pair["counterfactual_target_mobility"]
    )
    executed_counterfactual_changed += int(
        changed_executed != original_executed
    )

  query_count = len(rows)
  return {
      "query_count": query_count,
      "role_candidate_accuracy": role_correct / role_total,
      "strict_part_assignment_accuracy": strict_parts / query_count,
      "edge_part_pair_accuracy": edge_pair_correct / edge_total,
      "oracle_part_mobility_accuracy": oracle_mobility_correct / edge_total,
      "predicted_part_mobility_accuracy": predicted_mobility_correct / edge_total,
      "strict_joint_accuracy": joint_strict / query_count,
      "assignment_recall_at_beam": assignment_recall_at_beam / query_count,
      "joint_recall_at_beam": joint_recall_at_beam / query_count,
      "program_recall_at_beam": (
          program_recall_at_beam / query_count if support_edge_total else 0.0
      ),
      "edge_pair_recall_at_beam": edge_pair_recall_at_beam / edge_total,
      "beam_size": beam_size,
      "independent_edge_pair_accuracy": local_edge_pair_correct / edge_total,
      "independent_edge_mobility_accuracy": local_mobility_correct / edge_total,
      "independent_edge_consistency_rate": local_consistent_queries / query_count,
      "independent_edge_strict_part_accuracy": local_strict_parts / query_count,
      "independent_edge_strict_joint_accuracy": local_strict_joint / query_count,
      "independent_marginal_assignment_recall_at_beam": (
          marginal_assignment_recall_at_beam / query_count
      ),
      "independent_marginal_joint_recall_at_beam": (
          marginal_joint_recall_at_beam / query_count
      ),
      "counterfactual_mobility_accuracy": (
          counterfactual_correct / counterfactual_count if counterfactual_count else 0.0
      ),
      "counterfactual_prediction_change_rate": (
          counterfactual_changed / counterfactual_count if counterfactual_count else 0.0
      ),
      "learned_head_counterfactual_mobility_accuracy": (
          counterfactual_correct / counterfactual_count if counterfactual_count else 0.0
      ),
      "learned_head_counterfactual_change_rate": (
          counterfactual_changed / counterfactual_count if counterfactual_count else 0.0
      ),
      "executed_contract_counterfactual_mobility_accuracy": (
          executed_counterfactual_correct / counterfactual_count
          if counterfactual_count else 0.0
      ),
      "executed_contract_counterfactual_change_rate": (
          executed_counterfactual_changed / counterfactual_count
          if counterfactual_count else 0.0
      ),
      "interface_supervised_edge_count": interface_total,
      "interface_generator_coverage": (
          interface_generator_covered / interface_total if interface_total else 0.0
      ),
      "oracle_part_interface_orbit_accuracy": (
          interface_correct / interface_total if interface_total else 0.0
      ),
      "oracle_part_support_accuracy": (
          oracle_support_correct / support_edge_total if support_edge_total else 0.0
      ),
      "predicted_part_support_accuracy": (
          predicted_support_correct / support_edge_total if support_edge_total else 0.0
      ),
      "strict_support_mobility_program_accuracy": (
          strict_program / query_count if support_edge_total else 0.0
      ),
  }


def train_one(
    *,
    train_rows: list[LinkCADQueryTensors],
    train_counterfactual: Mapping[str, Any],
    eval_rows: list[LinkCADQueryTensors],
    eval_counterfactual: Mapping[str, Any],
    train_interface: Mapping[str, Mapping[str, tuple[int, int]]],
    eval_interface: Mapping[str, Mapping[str, tuple[int, int]]],
    mode: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    beam_size: int,
    model_kind: str,
    orbit_mode: str,
    max_orbits_per_candidate: int,
    init_checkpoint: Path | None,
    part_init_checkpoint: Path | None,
    full_init_checkpoint: Path | None,
    freeze_proposer_stack: bool,
    interface_weight: float,
) -> tuple[FactorizedLinkCADV1, dict[str, Any]]:
  if not np.isfinite(interface_weight) or interface_weight < 0.0:
    raise ValueError("LinkCAD interface loss weight differs")
  if mode not in {
      "full", "no_global", "no_language", "no_language_no_global"
  }:
    raise ValueError("LinkCAD pilot mode differs")
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  model_class = {
      "descriptor": FactorizedLinkCADV1,
      "brep": BRepFactorizedLinkCADV1,
      "primitive_v2": PrimitiveFactorizedLinkCADV2,
      "primitive_v3": PrimitiveFactorizedLinkCADV3,
      "primitive_v4": PrimitiveFactorizedLinkCADV4,
      "joinable_style": JoinABLeStyleLinkCADV1,
      "automate_style": AutoMateStyleLinkCADV1,
      "pointercad_style": PointerCADStyleLinkCADV1,
      "port_v6": PortConditionedLinkCADV6,
      "port_v9": PortConstraintLinkCADV9,
  }[model_kind]
  model_kwargs = {
      "hidden_dim": 128 if model_kind == "pointercad_style" else 64,
      "use_language": mode not in {"no_language", "no_language_no_global"},
      "use_global": mode not in {"no_global", "no_language_no_global"},
      "seed": seed,
  }
  if model_kind in {"joinable_style", "automate_style"}:
    model_kwargs["use_language"] = False
    model_kwargs["use_global"] = False
  if model_kind == "pointercad_style":
    model_kwargs["use_language"] = True
  if model_kind in {
      "brep", "primitive_v2", "primitive_v3", "primitive_v4",
      "joinable_style", "automate_style", "pointercad_style",
      "port_v6", "port_v9",
  }:
    model_kwargs["orbit_mode"] = orbit_mode
    model_kwargs["max_orbits_per_candidate"] = max_orbits_per_candidate
  model = model_class(**model_kwargs).to(device)
  if full_init_checkpoint is not None:
    model.load_state_dict(torch.load(
        full_init_checkpoint, map_location=device, weights_only=True,
    ))
  if init_checkpoint is not None:
    state = torch.load(init_checkpoint, map_location=device, weights_only=True)
    if model_kind in {"port_v6", "port_v9"}:
      current = model.state_dict()
      transferred = 0
      for key, value in state.items():
        destination = (
            "interface_text_encoder." + key.removeprefix("text_encoder.")
            if key.startswith("text_encoder.") else key
        )
        if (
            destination.startswith((
                "brep_encoder.", "primitive_encoder.", "primitive_norm.",
                "orbit_proposal.", "interface_text_encoder.",
            ))
            and destination in current
            and current[destination].shape == value.shape
        ):
          current[destination] = value
          transferred += 1
      if transferred == 0:
        raise ValueError("V6 port proposer initialization transferred no parameters")
      model.load_state_dict(current)
    else:
      model.load_state_dict(state)
  if part_init_checkpoint is not None:
    if model_kind not in {
        "primitive_v2", "primitive_v3", "primitive_v4", "port_v6",
        "port_v9",
    }:
      raise ValueError("part-only initialization requires primitive V2")
    source = torch.load(
        part_init_checkpoint, map_location=device, weights_only=True
    )
    current = model.state_dict()
    prefixes = (
        "part_encoder.", "text_encoder.", "unary.", "edge.",
        "edge_score.", "global_score.", "brep_encoder.",
    )
    transferred = 0
    for key, value in source.items():
      destination = key
      if model_kind in {"port_v6", "port_v9"} and key.startswith("brep_encoder."):
        destination = "part_brep_encoder." + key.removeprefix("brep_encoder.")
      if (
          key.startswith(prefixes)
          and destination in current
          and current[destination].shape == value.shape
      ):
        current[destination] = value
        transferred += 1
    if transferred == 0:
      raise ValueError("part initialization transferred no parameters")
    model.load_state_dict(current)
  if freeze_proposer_stack:
    if model_kind not in {
        "brep", "primitive_v2", "primitive_v3", "primitive_v4", "port_v6",
        "port_v9",
    }:
      raise ValueError("orbit proposer freezing requires the B-Rep model")
    modules = [model.brep_encoder, model.orbit_proposal]
    if model_kind in {"port_v6", "port_v9"}:
      modules.extend([
          model.primitive_encoder, model.primitive_norm,
          model.interface_text_encoder,
      ])
    else:
      modules.append(model.text_encoder)
      if model_kind in {"primitive_v2", "primitive_v3", "primitive_v4"}:
        modules.extend([model.primitive_encoder, model.primitive_norm])
    for module in modules:
      for parameter in module.parameters():
        parameter.requires_grad_(False)
  trainable_parameters = [
      parameter for parameter in model.parameters() if parameter.requires_grad
  ]
  optimizer = torch.optim.AdamW(
      trainable_parameters, lr=learning_rate, weight_decay=1e-4
  )
  order = list(range(len(train_rows)))
  epoch_losses = []
  for epoch in range(epochs):
    random.Random(seed + epoch).shuffle(order)
    model.train()
    cumulative = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, index in enumerate(order, start=1):
      query = train_rows[index].to(device)
      loss = linkcad_training_loss(
          model,
          query,
          interface_target_orbits=train_interface.get(query.query_id),
          interface_weight=interface_weight,
      )
      if (
          mode not in {"no_language", "no_language_no_global"}
          and not query.requested_mobility
      ):
        loss = loss + 0.5 * _counterfactual_loss(
            model, query, train_counterfactual[query.query_id]
        )
      (loss / 8.0).backward()
      cumulative += float(loss.detach())
      if step % 8 == 0 or step == len(order):
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    epoch_losses.append(cumulative / len(order))
  metrics = evaluate(
      model,
      eval_rows,
      eval_counterfactual,
      device=device,
      beam_size=beam_size,
      interface_by_query=eval_interface,
  )
  metrics.update({
      "mode": mode,
      "seed": seed,
      "epochs": epochs,
      "learning_rate": learning_rate,
      "epoch_losses": epoch_losses,
      "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
      "model_kind": model_kind,
      "orbit_mode": (
          orbit_mode if model_kind in {
              "brep", "primitive_v2", "primitive_v3", "primitive_v4",
              "joinable_style", "automate_style", "pointercad_style",
              "port_v6", "port_v9",
          }
          else "not_applicable"
      ),
      "max_orbits_per_candidate": (
          max_orbits_per_candidate
          if model_kind in {
              "brep", "primitive_v2", "primitive_v3", "primitive_v4",
              "joinable_style", "automate_style", "pointercad_style",
              "port_v6", "port_v9",
          } else 0
      ),
      "init_checkpoint": (
          None if init_checkpoint is None else init_checkpoint.resolve().as_posix()
      ),
      "freeze_proposer_stack": freeze_proposer_stack,
      "part_init_checkpoint": (
          None if part_init_checkpoint is None
          else part_init_checkpoint.resolve().as_posix()
      ),
      "full_init_checkpoint": (
          None if full_init_checkpoint is None
          else full_init_checkpoint.resolve().as_posix()
      ),
      "interface_weight": interface_weight,
  })
  return model, metrics


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-root", type=Path, required=True)
  parser.add_argument("--eval-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--modes", nargs="+", default=["full", "no_global", "no_language"])
  parser.add_argument("--seeds", nargs="+", type=int, default=[1701])
  parser.add_argument("--epochs", type=int, default=8)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--beam-size", type=int, default=25)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument(
      "--model-kind",
      choices=(
          "descriptor", "brep", "primitive_v2", "primitive_v3", "primitive_v4",
          "joinable_style", "automate_style", "pointercad_style",
          "port_v6", "port_v9",
      ),
      default="descriptor",
  )
  parser.add_argument("--brep-cache", type=Path)
  parser.add_argument("--primitive-cache", type=Path)
  parser.add_argument(
      "--primitive-cache-version", choices=("v2", "v3"), default="v2"
  )
  parser.add_argument("--orbit-mode", choices=("attention", "canonical"), default="attention")
  parser.add_argument("--max-train-queries", type=int, default=0)
  parser.add_argument("--max-eval-queries", type=int, default=0)
  parser.add_argument("--max-orbits-per-candidate", type=int, default=12)
  parser.add_argument("--init-checkpoint", type=Path)
  parser.add_argument("--part-init-checkpoint", type=Path)
  parser.add_argument("--full-init-checkpoint", type=Path)
  parser.add_argument("--filter-unsupported-mobility", action="store_true")
  parser.add_argument("--freeze-proposer-stack", action="store_true")
  parser.add_argument("--interface-weight", type=float, default=0.5)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  cache = None
  primitive_cache = None
  if args.model_kind == "brep":
    if args.brep_cache is None:
      raise ValueError("B-Rep model requires --brep-cache")
    cache = LinkCADBRepGraphCacheV1(args.brep_cache)
    cache.load_all()
  elif args.model_kind in {
      "primitive_v2", "primitive_v3", "primitive_v4", "joinable_style",
      "automate_style", "pointercad_style",
      "port_v6", "port_v9"
  }:
    if args.primitive_cache is None:
      raise ValueError("primitive V2 model requires --primitive-cache")
    expected_cache_version = (
        "v3" if args.model_kind == "primitive_v4" else "v2"
    )
    if args.primitive_cache_version != expected_cache_version:
      raise ValueError("LinkCAD primitive model/cache feature versions differ")
    cache_class = (
        LinkCADPrimitiveGraphCacheV3
        if args.primitive_cache_version == "v3"
        else LinkCADPrimitiveGraphCacheV2
    )
    primitive_cache = cache_class(args.primitive_cache)
    primitive_cache.load_all()
  train_rows, train_counterfactual, train_interface = _load_rows(
      args.train_root, brep_cache=cache, primitive_cache=primitive_cache,
      filter_unsupported_mobility=args.filter_unsupported_mobility,
  )
  eval_rows, eval_counterfactual, eval_interface = _load_rows(
      args.eval_root, brep_cache=cache, primitive_cache=primitive_cache,
      filter_unsupported_mobility=args.filter_unsupported_mobility,
  )
  train_rows = [
      row for row in train_rows
      if row.candidate_mask is None
      or all(
          bool(row.candidate_mask[role, target])
          for role, target in enumerate(row.target_assignment.tolist())
      )
  ]
  if args.max_train_queries > 0:
    train_rows = train_rows[:args.max_train_queries]
  if args.max_eval_queries > 0:
    eval_rows = eval_rows[:args.max_eval_queries]
  device = torch.device(args.device)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  results = []
  for mode in args.modes:
    for seed in args.seeds:
      model, metrics = train_one(
          train_rows=train_rows,
          train_counterfactual=train_counterfactual,
          eval_rows=eval_rows,
          eval_counterfactual=eval_counterfactual,
          train_interface=train_interface,
          eval_interface=eval_interface,
          mode=mode,
          seed=seed,
          epochs=args.epochs,
          learning_rate=args.learning_rate,
          device=device,
          beam_size=args.beam_size,
          model_kind=args.model_kind,
          orbit_mode=args.orbit_mode,
          max_orbits_per_candidate=args.max_orbits_per_candidate,
          init_checkpoint=args.init_checkpoint,
          part_init_checkpoint=args.part_init_checkpoint,
          full_init_checkpoint=args.full_init_checkpoint,
          freeze_proposer_stack=args.freeze_proposer_stack,
          interface_weight=args.interface_weight,
      )
      stem = f"{mode}_seed{seed}"
      torch.save(model.state_dict(), args.output_dir / f"{stem}.pt")
      (args.output_dir / f"{stem}.metrics.json").write_text(
          json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
      )
      results.append(metrics)
      print(json.dumps(metrics, sort_keys=True), flush=True)
  summary = {
      "schema_version": RESULT_SCHEMA_VERSION,
      "model_scope": "descriptor_protocol_smoke_not_paper_result",
      "train_query_count": len(train_rows),
      "eval_query_count": len(eval_rows),
      "device": str(device),
      "model_kind": args.model_kind,
      "orbit_mode": args.orbit_mode,
      "max_orbits_per_candidate": args.max_orbits_per_candidate,
      "init_checkpoint": (
          None
          if args.init_checkpoint is None
          else args.init_checkpoint.resolve().as_posix()
      ),
      "freeze_proposer_stack": args.freeze_proposer_stack,
      "part_init_checkpoint": (
          None if args.part_init_checkpoint is None
          else args.part_init_checkpoint.resolve().as_posix()
      ),
      "full_init_checkpoint": (
          None if args.full_init_checkpoint is None
          else args.full_init_checkpoint.resolve().as_posix()
      ),
      "unsupported_mobility_filtered": args.filter_unsupported_mobility,
      "primitive_cache_version": args.primitive_cache_version,
      "results": results,
  }
  (args.output_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


if __name__ == "__main__":
  main()
