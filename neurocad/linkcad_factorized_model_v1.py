"""Minimal factorized LinkCAD model and bounded structured decoder.

This v1 model is the fast protocol smoke test.  It uses target-free intrinsic
part descriptors and a deterministic frozen text projection.  The scoring and
decoder interfaces are shared with the later B-Rep graph encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODEL_SCHEMA_VERSION = "linkcad_factorized_linker.v1"
MOBILITY_NAMES = ("fixed", "revolute", "prismatic", "cylindrical")
EXECUTABLE_MOBILITY_NAMES = (
    *MOBILITY_NAMES, "planar", "ball", "pin_slot",
)
SUPPORT_NAMES = (
    "axial_support", "planar_support", "axis_plane_support", "other"
)
ANCHOR_SHAPE_TYPES = (
    "Arc3DCurveType",
    "Circle3DCurveType",
    "ConeSurfaceType",
    "CylinderSurfaceType",
    "Ellipse3DCurveType",
    "EllipticalConeSurfaceType",
    "EllipticalCylinderSurfaceType",
    "Line3DCurveType",
    "NurbsCurve3DCurveType",
    "NurbsSurfaceType",
    "PlaneSurfaceType",
    "SphereSurfaceType",
    "TorusSurfaceType",
)
TEXT_DIM = 128
PART_FEATURE_DIM = 4 + len(ANCHOR_SHAPE_TYPES)


def frozen_text_features(text: str, *, dimension: int = TEXT_DIM) -> Tensor:
  """Deterministic signed-hash word/character features; no fitted vocabulary."""

  if not isinstance(text, str) or not text.strip() or dimension < 8:
    raise ValueError("text feature input differs")
  normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
  words = normalized.split()
  tokens = [f"w:{word}" for word in words]
  compact = "_".join(words)
  tokens.extend(f"c3:{compact[i:i + 3]}" for i in range(max(0, len(compact) - 2)))
  result = torch.zeros(dimension, dtype=torch.float32)
  for token in tokens:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dimension
    sign = 1.0 if digest[4] & 1 else -1.0
    result[index] += sign
  return result / result.norm().clamp_min(1.0)


@dataclass(frozen=True, slots=True)
class LinkCADQueryTensors:
  query_id: str
  candidate_features: Tensor
  role_text_features: Tensor
  edge_index: Tensor
  edge_text_features: Tensor
  candidate_mask: Tensor | None = None
  target_assignment: Tensor | None = None
  target_mobility: Tensor | None = None
  target_support: Tensor | None = None
  candidate_ids: tuple[tuple[str, ...], ...] = ()
  edge_ids: tuple[str, ...] = ()
  candidate_graphs: tuple[tuple[Any | None, ...], ...] | None = None
  requested_mobility: tuple[str, ...] = ()
  port_candidate_mask: Tensor | None = None

  @property
  def role_count(self) -> int:
    return int(self.candidate_features.shape[0])

  @property
  def candidate_count(self) -> int:
    return int(self.candidate_features.shape[1])

  def to(self, device: str | torch.device) -> "LinkCADQueryTensors":
    return LinkCADQueryTensors(
        query_id=self.query_id,
        candidate_features=self.candidate_features.to(device),
        role_text_features=self.role_text_features.to(device),
        edge_index=self.edge_index.to(device),
        edge_text_features=self.edge_text_features.to(device),
        candidate_mask=(
            None if self.candidate_mask is None else self.candidate_mask.to(device)
        ),
        target_assignment=(
            None if self.target_assignment is None else self.target_assignment.to(device)
        ),
        target_mobility=(
            None if self.target_mobility is None else self.target_mobility.to(device)
        ),
        target_support=(
            None if self.target_support is None else self.target_support.to(device)
        ),
        candidate_ids=self.candidate_ids,
        edge_ids=self.edge_ids,
        candidate_graphs=(
            None
            if self.candidate_graphs is None
            else tuple(
                tuple(None if graph is None else graph.to(device) for graph in role)
                for role in self.candidate_graphs
            )
        ),
        requested_mobility=self.requested_mobility,
        port_candidate_mask=(
            None
            if self.port_candidate_mask is None
            else self.port_candidate_mask.to(device)
        ),
    )


def _candidate_features(rows: Sequence[Mapping[str, Any]]) -> Tensor:
  raw = []
  for row in rows:
    volume = math.log1p(max(0.0, float(row["volume"])))
    area = math.log1p(max(0.0, float(row["area"])))
    # The public projection intentionally excludes source-joint anchor labels.
    # This descriptor smoke view therefore uses only intrinsic physical values.
    shapes: set[str] = set()
    raw.append([
        volume,
        area,
        *[float(shape in shapes) for shape in ANCHOR_SHAPE_TYPES],
    ])
  tensor = torch.tensor(raw, dtype=torch.float32)
  scalar = tensor[:, :2]
  mean = scalar.mean(dim=0, keepdim=True)
  deviation = scalar.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
  normalized = (scalar - mean) / deviation
  return torch.cat((normalized, scalar, tensor[:, 2:]), dim=-1)


def tensorize_linkcad_query(
    public_query: Mapping[str, Any],
    private_target: Mapping[str, Any] | None = None,
) -> LinkCADQueryTensors:
  roles = public_query.get("roles")
  candidate_sets = public_query.get("candidate_sets")
  edges = public_query.get("functional_edges")
  if not isinstance(roles, list) or not isinstance(candidate_sets, Mapping):
    raise ValueError("LinkCAD public query roles differ")
  if not isinstance(edges, list) or not 2 <= len(roles) <= 5:
    raise ValueError("LinkCAD public query graph differs")
  role_ids = [str(row["role_id"]) for row in roles]
  if len(set(role_ids)) != len(role_ids):
    raise ValueError("LinkCAD role identities differ")
  candidate_rows = [candidate_sets[role_id] for role_id in role_ids]
  counts = [len(rows) for rows in candidate_rows]
  if not counts or min(counts) < 2:
    raise ValueError("LinkCAD candidate-set sizes differ")
  candidate_count = max(counts)
  feature_rows = [_candidate_features(rows) for rows in candidate_rows]
  candidate_features = torch.stack([
      F.pad(features, (0, 0, 0, candidate_count - int(features.shape[0])))
      for features in feature_rows
  ])
  candidate_mask = torch.tensor([
      [index < count for index in range(candidate_count)]
      for count in counts
  ], dtype=torch.bool)
  role_text = torch.stack([
      frozen_text_features(str(row["description"])) for row in roles
  ])
  role_index = {role_id: index for index, role_id in enumerate(role_ids)}
  edge_index = torch.tensor([
      [role_index[row["role_a"]], role_index[row["role_b"]]] for row in edges
  ], dtype=torch.long)
  edge_text = torch.stack([
      frozen_text_features(str(row["instruction"])) for row in edges
  ])
  candidate_ids = tuple(
      tuple(str(row["candidate_id"]) for row in candidates) + tuple(
          f"__linkcad_padding__:{role_id}:{index}"
          for index in range(len(candidates), candidate_count)
      )
      for role_id, candidates in zip(role_ids, candidate_rows, strict=True)
  )
  target_assignment = None
  target_mobility = None
  target_support = None
  if private_target is not None:
    target_by_role = private_target["target_candidate_by_role"]
    target_assignment = torch.tensor([
        candidate_ids[index].index(str(target_by_role[role_id]))
        for index, role_id in enumerate(role_ids)
    ], dtype=torch.long)
    target_edges = {
        str(row["edge_id"]): row for row in private_target["functional_edge_targets"]
    }
    target_mobility = torch.tensor([
        MOBILITY_NAMES.index(str(target_edges[row["edge_id"]]["target_mobility"]))
        for row in edges
    ], dtype=torch.long)
    target_support = torch.tensor([
        SUPPORT_NAMES.index(
            str(target_edges[row["edge_id"]].get("support_family", "other"))
            if str(target_edges[row["edge_id"]].get("support_family", "other"))
            in SUPPORT_NAMES else "other"
        )
        for row in edges
    ], dtype=torch.long)
  return LinkCADQueryTensors(
      query_id=str(public_query["query_id"]),
      candidate_features=candidate_features,
      role_text_features=role_text,
      edge_index=edge_index,
      edge_text_features=edge_text,
      candidate_mask=candidate_mask,
      target_assignment=target_assignment,
      target_mobility=target_mobility,
      target_support=target_support,
      candidate_ids=candidate_ids,
      edge_ids=tuple(str(row["edge_id"]) for row in edges),
      requested_mobility=(
          tuple(str(row["requested_mobility"]) for row in edges)
          if all(
              str(row.get("requested_mobility", ""))
              in EXECUTABLE_MOBILITY_NAMES
              for row in edges
          ) else ()
      ),
  )


@dataclass(frozen=True, slots=True)
class FactorizedOutput:
  unary_logits: Tensor
  edge_pair_logits: Tensor
  mobility_logits: Tensor
  candidate_embeddings: Tensor
  edge_pair_embeddings: Tensor
  interface_score_blocks: Any = None
  orbit_proposal_logits: Any = None
  selected_orbit_indices: Any = None
  support_logits: Any = None


@dataclass(frozen=True, slots=True)
class LinkCADPrediction:
  assignment: tuple[int, ...]
  candidate_ids: tuple[str, ...]
  mobility: tuple[str, ...]
  support: tuple[str, ...]
  interface_orbits: tuple[tuple[int, int], ...]
  interface_alternatives: tuple[
      tuple[tuple[int, int, float], ...], ...
  ]
  score: float


class FactorizedLinkCADV1(nn.Module):
  def __init__(
      self,
      *,
      hidden_dim: int = 64,
      use_language: bool = True,
      use_global: bool = True,
      seed: int = 1701,
  ) -> None:
    super().__init__()
    if hidden_dim < 16:
      raise ValueError("LinkCAD hidden dimension differs")
    self.hidden_dim = hidden_dim
    self.use_language = use_language
    self.use_global = use_global
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(seed)
      self.part_encoder = nn.Sequential(
          nn.Linear(PART_FEATURE_DIM, hidden_dim), nn.SiLU(),
          nn.Linear(hidden_dim, hidden_dim),
      )
      self.text_encoder = nn.Sequential(
          nn.Linear(TEXT_DIM, hidden_dim), nn.SiLU(),
          nn.Linear(hidden_dim, hidden_dim),
      )
      self.unary = nn.Sequential(
          nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
      )
      self.edge = nn.Sequential(
          nn.Linear(7 * hidden_dim, 2 * hidden_dim), nn.SiLU(),
          nn.Linear(2 * hidden_dim, hidden_dim),
      )
      self.edge_score = nn.Linear(hidden_dim, 1)
      self.mobility = nn.Linear(hidden_dim, len(MOBILITY_NAMES))
      self.global_score = nn.Sequential(
          nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
      )

  def forward(self, query: LinkCADQueryTensors) -> FactorizedOutput:
    part = self.part_encoder(query.candidate_features)
    return self._forward_from_part_embeddings(query, part)

  def _forward_from_part_embeddings(
      self,
      query: LinkCADQueryTensors,
      part: Tensor,
  ) -> FactorizedOutput:
    role_text = self.text_encoder(query.role_text_features)
    edge_text = self.text_encoder(query.edge_text_features)
    if not self.use_language:
      role_text = torch.zeros_like(role_text)
      edge_text = torch.zeros_like(edge_text)
    role = role_text[:, None, :].expand_as(part)
    unary_input = torch.cat((part, role, part * role, torch.abs(part - role)), dim=-1)
    unary_logits = self.unary(unary_input).squeeze(-1)
    candidate_mask = (
        torch.ones_like(unary_logits, dtype=torch.bool)
        if query.candidate_mask is None else query.candidate_mask
    )
    unary_logits = unary_logits.masked_fill(~candidate_mask, -torch.inf)
    pair_embeddings = []
    pair_logits = []
    mobility_logits = []
    for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
      left = part[role_a][:, None, :].expand(-1, part.shape[1], -1)
      right = part[role_b][None, :, :].expand(part.shape[1], -1, -1)
      language = edge_text[edge_ordinal][None, None, :].expand_as(left)
      pair = self.edge(torch.cat((
          left, right, language, left * right,
          torch.abs(left - right), left * language, right * language,
      ), dim=-1))
      pair_embeddings.append(pair)
      valid_pairs = candidate_mask[role_a][:, None] & candidate_mask[role_b][None, :]
      pair_logits.append(
          self.edge_score(pair).squeeze(-1).masked_fill(~valid_pairs, -torch.inf)
      )
      mobility_logits.append(self.mobility(pair))
    return FactorizedOutput(
        unary_logits=unary_logits,
        edge_pair_logits=torch.stack(pair_logits),
        mobility_logits=torch.stack(mobility_logits),
        candidate_embeddings=part,
        edge_pair_embeddings=torch.stack(pair_embeddings),
    )

  def assignment_score(
      self,
      query: LinkCADQueryTensors,
      output: FactorizedOutput,
      assignment: Sequence[int],
  ) -> Tensor:
    indices = torch.tensor(assignment, dtype=torch.long, device=output.unary_logits.device)
    roles = torch.arange(query.role_count, device=indices.device)
    score = output.unary_logits[roles, indices].sum()
    edge_values = []
    for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
      a, b = indices[role_a], indices[role_b]
      score = score + output.edge_pair_logits[edge_ordinal, a, b]
      edge_values.append(output.edge_pair_embeddings[edge_ordinal, a, b])
    if self.use_global:
      selected = output.candidate_embeddings[roles, indices].mean(dim=0)
      edge_mean = torch.stack(edge_values).mean(dim=0)
      score = score + self.global_score(torch.cat((selected, edge_mean))).squeeze()
    return score

  @torch.no_grad()
  def decode(self, query: LinkCADQueryTensors, *, beam_size: int = 25) -> LinkCADPrediction:
    return self.decode_beam(query, beam_size=beam_size)[0]

  @torch.no_grad()
  def decode_beam(
      self,
      query: LinkCADQueryTensors,
      *,
      beam_size: int = 25,
      interface_alternatives_per_edge: int = 8,
  ) -> list[LinkCADPrediction]:
    if beam_size < 1 or interface_alternatives_per_edge < 1:
      raise ValueError("LinkCAD beam size differs")
    output = self(query)
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for role_index in range(query.role_count):
      expanded = []
      for prefix, prefix_score in beams:
        for candidate_index in range(query.candidate_count):
          if query.candidate_mask is not None and not bool(
              query.candidate_mask[role_index, candidate_index]
          ):
            continue
          score = prefix_score + float(output.unary_logits[role_index, candidate_index])
          assignment = prefix + (candidate_index,)
          for edge_ordinal, (a, b) in enumerate(query.edge_index.tolist()):
            if max(a, b) == role_index and a < len(assignment) and b < len(assignment):
              score += float(output.edge_pair_logits[
                  edge_ordinal, assignment[a], assignment[b]
              ])
          if math.isfinite(score):
            expanded.append((assignment, score))
      expanded.sort(key=lambda row: (-row[1], row[0]))
      beams = expanded[:beam_size]
    rescored = []
    for assignment, _ in beams:
      score = float(self.assignment_score(query, output, assignment))
      if math.isfinite(score):
        rescored.append((assignment, score))
    rescored.sort(key=lambda row: (-row[1], row[0]))
    predictions = []
    for assignment, score in rescored:
      mobility = []
      support = []
      interface_orbits = []
      interface_alternatives = []
      for edge_ordinal, (a, b) in enumerate(query.edge_index.tolist()):
        logits = output.mobility_logits[edge_ordinal, assignment[a], assignment[b]]
        mobility.append(self._decode_mobility(query, edge_ordinal, logits))
        if output.support_logits is None:
          learned_support = "other"
        else:
          support_logits = output.support_logits[
              edge_ordinal, assignment[a], assignment[b]
          ]
          learned_support = SUPPORT_NAMES[int(support_logits.argmax())]
        if output.interface_score_blocks is None:
          interface_orbits.append((-1, -1))
          interface_alternatives.append(())
          support.append(self._decode_support(
              query, edge_ordinal, assignment[a], assignment[b],
              (-1, -1), learned_support,
          ))
          continue
        block = output.interface_score_blocks[edge_ordinal][
            assignment[a]
        ][assignment[b]]
        selected_a = output.selected_orbit_indices[edge_ordinal][0][
            assignment[a]
        ]
        selected_b = output.selected_orbit_indices[edge_ordinal][1][
            assignment[b]
        ]
        if block is None or not selected_a or not selected_b:
          interface_orbits.append((-1, -1))
          interface_alternatives.append(())
          support.append(self._decode_support(
              query, edge_ordinal, assignment[a], assignment[b],
              (-1, -1), learned_support,
          ))
          continue
        alternatives = []
        for local_a in range(int(block.shape[0])):
          for local_b in range(int(block.shape[1])):
            value = float(block[local_a, local_b])
            if math.isfinite(value):
              alternatives.append((
                  selected_a[local_a], selected_b[local_b], value,
              ))
        alternatives.sort(key=lambda row: (-row[2], row[0], row[1]))
        alternatives = alternatives[:interface_alternatives_per_edge]
        interface_alternatives.append(tuple(alternatives))
        interface_orbits.append(
            (alternatives[0][0], alternatives[0][1])
            if alternatives else (-1, -1)
        )
        support.append(self._decode_support(
            query, edge_ordinal, assignment[a], assignment[b],
            interface_orbits[-1], learned_support,
        ))
      predictions.append(LinkCADPrediction(
          assignment=assignment,
          candidate_ids=tuple(
              query.candidate_ids[role][candidate]
              for role, candidate in enumerate(assignment)
          ),
          mobility=tuple(mobility),
          support=tuple(support),
          interface_orbits=tuple(interface_orbits),
          interface_alternatives=tuple(interface_alternatives),
          score=score,
      ))
    return predictions

  def _decode_support(
      self, _query, _edge_ordinal, _candidate_a, _candidate_b,
      _primitive_pair, learned_support,
  ):
    return learned_support

  def _decode_mobility(self, query, edge_ordinal, logits):
    if query.requested_mobility:
      return query.requested_mobility[edge_ordinal]
    return MOBILITY_NAMES[int(logits.argmax())]


def linkcad_training_loss(
    model: FactorizedLinkCADV1,
    query: LinkCADQueryTensors,
    *,
    structured_weight: float = 0.25,
    interface_target_orbits: Mapping[str, tuple[int, int]] | None = None,
    interface_weight: float = 0.5,
) -> Tensor:
  if query.target_assignment is None or query.target_mobility is None:
    raise ValueError("LinkCAD training targets are missing")
  if query.requested_mobility:
    target_names = tuple(
        MOBILITY_NAMES[int(value)] for value in query.target_mobility.tolist()
    )
    if query.requested_mobility != target_names:
      raise ValueError("LinkCAD requested mobility differs from training target")
  output = model(query)
  target = query.target_assignment
  available_mask = (
      torch.ones_like(output.unary_logits, dtype=torch.bool)
      if query.candidate_mask is None else query.candidate_mask
  )
  if not bool(torch.isfinite(output.unary_logits[available_mask]).all()):
    raise FloatingPointError(
        f"LinkCAD unary logits became non-finite for {query.query_id}"
    )
  if query.candidate_mask is not None and any(
      not bool(query.candidate_mask[role, target[role]])
      for role in range(query.role_count)
  ):
    raise ValueError("LinkCAD target candidate is unavailable")
  unary = torch.stack([
      F.cross_entropy(output.unary_logits[role][None, :], target[role][None])
      for role in range(query.role_count)
  ]).mean()
  pair_losses = []
  mobility_losses = []
  support_losses = []
  interface_losses = []
  orbit_proposal_losses = []
  for edge_ordinal, (role_a, role_b) in enumerate(query.edge_index.tolist()):
    a, b = target[role_a], target[role_b]
    pair_target = (a * query.candidate_count + b).reshape(1)
    pair_losses.append(F.cross_entropy(
        output.edge_pair_logits[edge_ordinal].reshape(1, -1), pair_target
    ))
    mobility_losses.append(F.cross_entropy(
        output.mobility_logits[edge_ordinal, a, b][None, :],
        query.target_mobility[edge_ordinal][None],
    ))
    if output.support_logits is not None and query.target_support is not None:
      support_losses.append(F.cross_entropy(
          output.support_logits[edge_ordinal, a, b][None, :],
          query.target_support[edge_ordinal][None],
      ))
    if (
        interface_target_orbits
        and output.interface_score_blocks is not None
        and query.edge_ids[edge_ordinal] in interface_target_orbits
    ):
      orbit_a, orbit_b = interface_target_orbits[query.edge_ids[edge_ordinal]]
      proposal_a = output.orbit_proposal_logits[edge_ordinal][0][int(a)]
      proposal_b = output.orbit_proposal_logits[edge_ordinal][1][int(b)]
      if orbit_a < proposal_a.shape[0]:
        orbit_proposal_losses.append(F.cross_entropy(
            proposal_a[None, :],
            torch.tensor([orbit_a], dtype=torch.long, device=proposal_a.device),
        ))
      if orbit_b < proposal_b.shape[0]:
        orbit_proposal_losses.append(F.cross_entropy(
            proposal_b[None, :],
            torch.tensor([orbit_b], dtype=torch.long, device=proposal_b.device),
        ))
      selected_a = output.selected_orbit_indices[edge_ordinal][0][int(a)]
      selected_b = output.selected_orbit_indices[edge_ordinal][1][int(b)]
      local_a = (
          selected_a.index(orbit_a) if orbit_a in selected_a else None
      )
      local_b = (
          selected_b.index(orbit_b) if orbit_b in selected_b else None
      )
      block = output.interface_score_blocks[edge_ordinal][int(a)][int(b)]
      if (
          block is not None
          and local_a is not None
          and local_b is not None
      ):
        interface_losses.append(F.cross_entropy(
            block.reshape(1, -1),
            torch.tensor(
                [local_a * block.shape[1] + local_b],
                dtype=torch.long,
                device=block.device,
            ),
        ))
  positive_assignment = tuple(int(value) for value in target.tolist())
  positive_score = model.assignment_score(query, output, positive_assignment)
  structured = []
  for role in range(query.role_count):
    available_indices = torch.where(available_mask[role])[0]
    if available_indices.numel() < 1:
      raise ValueError("LinkCAD role has no available candidate")
    available_logits = output.unary_logits[role, available_indices]
    wrong_local = int(torch.topk(
        available_logits, k=min(2, int(available_indices.numel()))
    ).indices[-1])
    wrong = int(available_indices[wrong_local])
    if wrong == positive_assignment[role]:
      alternatives = [
          int(value) for value in available_indices.tolist()
          if int(value) != positive_assignment[role]
      ]
      if not alternatives:
        continue
      wrong = alternatives[0]
    corrupted = list(positive_assignment)
    corrupted[role] = wrong
    negative_score = model.assignment_score(query, output, corrupted)
    structured.append(F.softplus(1.0 + negative_score - positive_score))
  return (
      unary
      + torch.stack(pair_losses).mean()
      + (
          torch.zeros((), device=unary.device)
          if query.requested_mobility
          else torch.stack(mobility_losses).mean()
      )
      + (
          torch.stack(support_losses).mean()
          if support_losses else torch.zeros((), device=unary.device)
      )
      + interface_weight * (
          torch.stack(interface_losses).mean()
          if interface_losses else torch.zeros((), device=unary.device)
      )
      + interface_weight * (
          torch.stack(orbit_proposal_losses).mean()
          if orbit_proposal_losses else torch.zeros((), device=unary.device)
      )
      + structured_weight * (
          torch.stack(structured).mean()
          if structured else torch.zeros((), device=unary.device)
      )
  )
