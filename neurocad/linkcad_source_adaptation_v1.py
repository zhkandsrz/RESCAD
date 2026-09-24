"""Parameter-efficient source adaptation for LinkCAD part ranking."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

from .linkcad_factorized_model_v1 import LinkCADQueryTensors
from .linkcad_port_constraint_model_v9 import PortConstraintLinkCADV9


SCHEMA_VERSION = "linkcad_source_adaptation.v1"


def part_assignment_output_v1(
    model: PortConstraintLinkCADV9, query: LinkCADQueryTensors,
):
  """Run only the part branch and public port mask during source adaptation."""

  if query.candidate_graphs is None:
    raise ValueError("LinkCAD source-adaptation graphs are missing")
  graphs = []
  positions = []
  for role, role_graphs in enumerate(query.candidate_graphs):
    for candidate, graph in enumerate(role_graphs):
      if graph is not None:
        graphs.append(graph.face_graph)
        positions.append((role, candidate))
  if not graphs:
    raise ValueError("LinkCAD source-adaptation graph domain is empty")
  encoded = model.part_brep_encoder.forward_many(
      graphs, orbit_mode=model.orbit_mode,
  )
  part = torch.zeros(
      (query.role_count, query.candidate_count, model.hidden_dim),
      dtype=encoded.dtype, device=encoded.device,
  )
  for row, (role, candidate) in enumerate(positions):
    part[role, candidate] = encoded[row]
  output = model._forward_from_part_embeddings(query, part)
  if query.port_candidate_mask is None:
    return output
  pair_mask = (
      query.port_candidate_mask[:, 0, :, None]
      & query.port_candidate_mask[:, 1, None, :]
  )
  return replace(
      output,
      edge_pair_logits=output.edge_pair_logits.masked_fill(~pair_mask, -torch.inf),
  )


def configure_part_ranker_adaptation_v1(
    model: PortConstraintLinkCADV9, *, adapt_part_encoder: bool = False,
) -> tuple[torch.nn.Parameter, ...]:
  """Freeze execution/interface stacks and expose only assignment parameters."""

  for parameter in model.parameters():
    parameter.requires_grad_(False)
  modules: list[torch.nn.Module] = [
      model.text_encoder, model.unary, model.edge, model.edge_score,
  ]
  if adapt_part_encoder:
    modules.append(model.part_brep_encoder)
  for module in modules:
    for parameter in module.parameters():
      parameter.requires_grad_(True)
  return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def configure_interface_ranker_adaptation_v1(
    model: PortConstraintLinkCADV9,
) -> tuple[torch.nn.Parameter, ...]:
  """Expose only the language-conditioned primitive-pair ranking head."""

  for parameter in model.parameters():
    parameter.requires_grad_(False)
  for module in (
      model.interface_text_encoder, model.interface_edge,
      model.interface_score,
  ):
    for parameter in module.parameters():
      parameter.requires_grad_(True)
  return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def part_assignment_loss_v1(
    model: PortConstraintLinkCADV9, query: LinkCADQueryTensors,
) -> torch.Tensor:
  if query.target_assignment is None:
    raise ValueError("LinkCAD source-adaptation target is missing")
  output = part_assignment_output_v1(model, query)
  target = query.target_assignment
  unary_losses = [
      F.cross_entropy(output.unary_logits[role][None, :], target[role][None])
      for role in range(query.role_count)
  ]
  pair_losses = []
  for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
    pair_target = (
        target[role_a] * query.candidate_count + target[role_b]
    ).reshape(1)
    pair_losses.append(F.cross_entropy(
        output.edge_pair_logits[edge_ordinal].reshape(1, -1), pair_target,
    ))
  return torch.stack(unary_losses).mean() + torch.stack(pair_losses).mean()


def interface_ranking_loss_v1(
    model: PortConstraintLinkCADV9, query: LinkCADQueryTensors,
    interface_target_orbits: Mapping[str, tuple[int, int]],
) -> torch.Tensor:
  """Rank the intended primitive pair at the already supervised part pair."""

  if query.target_assignment is None or not interface_target_orbits:
    raise ValueError("LinkCAD interface-adaptation target is missing")
  output = model(query)
  if output.interface_score_blocks is None:
    raise ValueError("LinkCAD interface-adaptation scores are missing")
  losses = []
  for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
    edge_id = query.edge_ids[edge_ordinal]
    if edge_id not in interface_target_orbits:
      continue
    candidate_a = int(query.target_assignment[role_a])
    candidate_b = int(query.target_assignment[role_b])
    target_a, target_b = interface_target_orbits[edge_id]
    selected_a = output.selected_orbit_indices[edge_ordinal][0][candidate_a]
    selected_b = output.selected_orbit_indices[edge_ordinal][1][candidate_b]
    if target_a not in selected_a or target_b not in selected_b:
      raise ValueError("LinkCAD interface-adaptation target was not proposed")
    block = output.interface_score_blocks[edge_ordinal][candidate_a][candidate_b]
    if block is None:
      raise ValueError("LinkCAD interface-adaptation score block is missing")
    local_a = selected_a.index(target_a)
    local_b = selected_b.index(target_b)
    target = torch.tensor(
        [local_a * block.shape[1] + local_b],
        dtype=torch.long, device=block.device,
    )
    losses.append(F.cross_entropy(block.reshape(1, -1), target))
  if not losses:
    raise ValueError("LinkCAD interface-adaptation edge domain is empty")
  return torch.stack(losses).mean()


def trainable_parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
  return sum(parameter.numel() for parameter in parameters)


__all__ = [
    "SCHEMA_VERSION", "configure_interface_ranker_adaptation_v1",
    "configure_part_ranker_adaptation_v1", "interface_ranking_loss_v1",
    "part_assignment_loss_v1", "part_assignment_output_v1",
    "trainable_parameter_count",
]
