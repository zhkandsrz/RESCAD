"""B-Rep graph instantiation of the factorized LinkCAD scorer."""

from __future__ import annotations

import torch

from .linkcad_brep_encoder_v1 import LinkCADBRepEncoderV1
from .linkcad_factorized_model_v1 import (
    FactorizedLinkCADV1,
    FactorizedOutput,
    LinkCADQueryTensors,
)


class BRepFactorizedLinkCADV1(FactorizedLinkCADV1):
  def __init__(
      self,
      *,
      hidden_dim: int = 64,
      use_language: bool = True,
      use_global: bool = True,
      orbit_mode: str = "attention",
      max_orbits_per_candidate: int = 12,
      seed: int = 1701,
  ) -> None:
    super().__init__(
        hidden_dim=hidden_dim,
        use_language=use_language,
        use_global=use_global,
        seed=seed,
    )
    if orbit_mode not in {"attention", "canonical"}:
      raise ValueError("LinkCAD orbit mode differs")
    self.orbit_mode = orbit_mode
    if max_orbits_per_candidate < 2:
      raise ValueError("LinkCAD interface orbit budget differs")
    self.max_orbits_per_candidate = max_orbits_per_candidate
    self.brep_encoder = LinkCADBRepEncoderV1(hidden_dim=hidden_dim, layers=3)
    self.orbit_proposal = torch.nn.Sequential(
        torch.nn.Linear(4 * hidden_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, 1),
    )

  def forward(self, query: LinkCADQueryTensors) -> FactorizedOutput:
    if query.candidate_graphs is None:
      raise ValueError("LinkCAD B-Rep candidate graphs are missing")
    available_graphs = []
    positions = []
    for role, role_graphs in enumerate(query.candidate_graphs):
      for candidate, graph in enumerate(role_graphs):
        if graph is not None:
          available_graphs.append(graph)
          positions.append((role, candidate))
    if not available_graphs:
      raise ValueError("LinkCAD B-Rep query has no available candidate graphs")
    encoded, orbit_sets = self._encode_graphs(available_graphs)
    part = torch.zeros(
        (query.role_count, query.candidate_count, self.hidden_dim),
        device=encoded.device,
        dtype=encoded.dtype,
    )
    for row, (role, candidate) in enumerate(positions):
      part[role, candidate] = encoded[row]
    base = self._forward_from_part_embeddings(query, part)
    orbit_by_position = {}
    for row, (role, candidate) in enumerate(positions):
      orbits = orbit_sets[row]
      orbit_by_position[(role, candidate)] = orbits
    edge_text = self._encode_interface_text(query.edge_text_features)
    if not self.use_language:
      edge_text = torch.zeros_like(edge_text)
    pair_logits = []
    pair_embeddings = []
    mobility_logits = []
    support_logits = []
    interface_score_blocks = []
    edge_proposal_logits = []
    edge_selected_orbits = []
    candidate_mask = (
        torch.ones(
            (query.role_count, query.candidate_count),
            dtype=torch.bool,
            device=part.device,
        )
        if query.candidate_mask is None else query.candidate_mask
    )
    for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
      left_rows = []
      left_candidates = []
      right_rows = []
      right_candidates = []
      side_proposals = [[], []]
      side_selected = [[], []]
      edge_language = edge_text[edge_ordinal]
      for side_ordinal, role in enumerate((role_a, role_b)):
        for candidate in range(query.candidate_count):
          orbits = orbit_by_position.get((role, candidate))
          if orbits is None:
            side_proposals[side_ordinal].append(None)
            side_selected[side_ordinal].append(())
            continue
          language = edge_language[None, :].expand_as(orbits)
          proposal_logits = self.orbit_proposal(torch.cat((
              orbits,
              language,
              orbits * language,
              torch.abs(orbits - language),
          ), dim=-1)).squeeze(-1)
          selected = torch.topk(
              proposal_logits,
              k=min(self.max_orbits_per_candidate, int(orbits.shape[0])),
          ).indices.sort().values
          side_proposals[side_ordinal].append(proposal_logits)
          side_selected[side_ordinal].append(tuple(
              int(value) for value in selected.tolist()
          ))
      edge_proposal_logits.append((
          tuple(side_proposals[0]), tuple(side_proposals[1])
      ))
      edge_selected_orbits.append((
          tuple(side_selected[0]), tuple(side_selected[1])
      ))
      for candidate in range(query.candidate_count):
        left_all = orbit_by_position.get((role_a, candidate))
        right_all = orbit_by_position.get((role_b, candidate))
        left_orbits = (
            None
            if left_all is None
            else left_all[list(side_selected[0][candidate])]
        )
        right_orbits = (
            None
            if right_all is None
            else right_all[list(side_selected[1][candidate])]
        )
        if left_orbits is not None:
          left_rows.append(left_orbits)
          left_candidates.extend([candidate] * int(left_orbits.shape[0]))
        if right_orbits is not None:
          right_rows.append(right_orbits)
          right_candidates.extend([candidate] * int(right_orbits.shape[0]))
      left = torch.cat(left_rows)
      right = torch.cat(right_rows)
      left_grid = left[:, None, :].expand(-1, right.shape[0], -1)
      right_grid = right[None, :, :].expand(left.shape[0], -1, -1)
      language = edge_text[edge_ordinal][None, None, :].expand_as(left_grid)
      orbit_pair_hidden = self._interface_hidden(torch.cat((
          left_grid,
          right_grid,
          language,
          left_grid * right_grid,
          torch.abs(left_grid - right_grid),
          left_grid * language,
          right_grid * language,
      ), dim=-1))
      orbit_pair_scores = self._interface_scores(orbit_pair_hidden).squeeze(-1)
      scores = torch.full(
          (query.candidate_count, query.candidate_count),
          -torch.inf,
          device=part.device,
      )
      hidden = torch.zeros(
          (query.candidate_count, query.candidate_count, self.hidden_dim),
          device=part.device,
      )
      mobility = torch.zeros(
          (
              query.candidate_count,
              query.candidate_count,
              self.mobility.out_features,
          ),
          device=part.device,
      )
      support_head = getattr(self, "support_head", None)
      support = (
          None
          if support_head is None
          else torch.zeros(
              (
                  query.candidate_count,
                  query.candidate_count,
                  support_head.out_features,
              ),
              device=part.device,
          )
      )
      left_candidate_ids = torch.tensor(
          left_candidates, dtype=torch.long, device=part.device
      )
      right_candidate_ids = torch.tensor(
          right_candidates, dtype=torch.long, device=part.device
      )
      blocks: list[list[torch.Tensor | None]] = [
          [None for _ in range(query.candidate_count)]
          for _ in range(query.candidate_count)
      ]
      for candidate_a in range(query.candidate_count):
        for candidate_b in range(query.candidate_count):
          if not (
              bool(candidate_mask[role_a, candidate_a])
              and bool(candidate_mask[role_b, candidate_b])
          ):
            continue
          local_mask = (
              (left_candidate_ids[:, None] == candidate_a)
              & (right_candidate_ids[None, :] == candidate_b)
          )
          local_scores = orbit_pair_scores.masked_fill(~local_mask, -torch.inf)
          left_mask = left_candidate_ids == candidate_a
          right_mask = right_candidate_ids == candidate_b
          blocks[candidate_a][candidate_b] = orbit_pair_scores[left_mask][
              :, right_mask
          ]
          flat_index = int(local_scores.reshape(-1).argmax())
          left_index = flat_index // int(right.shape[0])
          right_index = flat_index % int(right.shape[0])
          selected_hidden = orbit_pair_hidden[left_index, right_index]
          scores[candidate_a, candidate_b] = local_scores[left_index, right_index]
          hidden[candidate_a, candidate_b] = selected_hidden
          mobility[candidate_a, candidate_b] = self._interface_mobility(
              selected_hidden
          )
          if support is not None:
            support[candidate_a, candidate_b] = support_head(selected_hidden)
      pair_logits.append(self._assignment_pair_scores(
          base.edge_pair_logits[edge_ordinal], scores,
      ))
      pair_embeddings.append(hidden)
      mobility_logits.append(mobility)
      if support is not None:
        support_logits.append(support)
      interface_score_blocks.append(
          tuple(tuple(row) for row in blocks)
      )
    return FactorizedOutput(
        unary_logits=base.unary_logits,
        edge_pair_logits=torch.stack(pair_logits),
        mobility_logits=torch.stack(mobility_logits),
        candidate_embeddings=part,
        edge_pair_embeddings=torch.stack(pair_embeddings),
        interface_score_blocks=tuple(interface_score_blocks),
        orbit_proposal_logits=tuple(edge_proposal_logits),
        selected_orbit_indices=tuple(edge_selected_orbits),
        support_logits=(
            torch.stack(support_logits) if support_logits else None
        ),
    )

  def _encode_graphs(self, available_graphs):
    return self.brep_encoder.forward_many_with_orbits(
        available_graphs, orbit_mode=self.orbit_mode
    )

  def _encode_interface_text(self, features):
    return self.text_encoder(features)

  def _interface_hidden(self, inputs):
    return self.edge(inputs)

  def _interface_scores(self, hidden):
    return self.edge_score(hidden)

  def _interface_mobility(self, hidden):
    return self.mobility(hidden)

  def _assignment_pair_scores(self, _part_scores, interface_scores):
    return interface_scores
