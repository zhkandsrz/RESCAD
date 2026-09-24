"""Ground learned single-part interfaces into planner sockets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .benchmark_v2_model_view import (
    BENCHMARK_V2_FEATURE_NAMES,
    ModelViewSanitization,
    benchmark_v2_numeric_feature_vector,
    sanitize_benchmark_v2_model_view,
    validate_benchmark_v2_model_view,
)
from .interface_features import (
    CandidateInterface,
    extract_candidate_interfaces_from_shape,
    extract_candidate_interfaces_from_step,
)
from .interface_pair_scorer import InterfacePairScorer
from .interface_scorer import InterfaceModelInput, InterfaceScorer
from .domain_types import PartTemplate, Socket


@dataclass
class InterfacePairCandidate:
  score: float
  parent_socket: Socket
  child_socket: Socket
  relation_mode: str
  note: str
  heuristic_score: float = 0.0
  learned_pair_score: float = 0.0
  predicted_contact_type: str = ""
  contact_type_probs: dict[str, float] = field(default_factory=dict)


def augment_template_with_scored_interfaces(
    *,
    template: PartTemplate,
    step_path: str | Path,
    scorer: InterfaceScorer,
    part_name: str,
    body_uuid: str = "",
    top_k: int = 16,
    min_score: float = 0.25,
    shape: Any = None,
    candidates: Optional[Sequence[CandidateInterface]] = None,
    sanitized_model_views: Optional[Sequence[ModelViewSanitization]] = None,
    private_exact_bindings: Optional[dict[str, dict[str, Any]]] = None,
    protocol: str = "legacy",
) -> list[str]:
  """Add learned interface sockets to a template using only one STEP file."""

  protocol = str(protocol or "legacy").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("protocol must be 'legacy' or 'benchmark_v2'")
  if protocol == "legacy" and sanitized_model_views is not None:
    raise ValueError("sanitized_model_views is valid only for benchmark_v2")
  if protocol == "legacy" and private_exact_bindings is not None:
    raise ValueError("private_exact_bindings is valid only for benchmark_v2")
  if protocol == "benchmark_v2" and shape is None and candidates is None:
    raise ValueError(
        "benchmark_v2 interface grounding requires an already gauged shape; "
        "STEP reload is forbidden"
    )
  if candidates is not None:
    extracted_candidates = list(candidates)
  elif shape is None:
    extracted_candidates = extract_candidate_interfaces_from_step(
        step_path=step_path,
        part_name=part_name,
        body_uuid=body_uuid,
    )
  else:
    extracted_candidates = extract_candidate_interfaces_from_shape(
        shape,
        part_name=part_name,
        body_uuid=body_uuid,
        source_step_path=step_path,
        protocol=protocol,
    )
  candidates = extracted_candidates
  model_inputs: dict[int, InterfaceModelInput] = {}
  safe_roles: dict[int, str] = {}
  if protocol == "legacy":
    scorer.score_candidates(candidates)
    for candidate in candidates:
      if not bool(candidate.metadata.get("composite_interface", False)):
        continue
      raw = 0.0 if candidate.score is None else float(candidate.score)
      role = str(candidate.role_hint or "").lower()
      boost = 0.18
      if role in {"center_bore", "threaded_hole", "obround_slot", "pin_boss"}:
        boost = 0.26
      candidate.score = min(1.0, raw + boost)
    candidates = [
        item for item in candidates if item.score is not None and item.score >= min_score
    ]
    candidates.sort(key=lambda item: (-(item.score or 0.0), item.interface_id))
    if top_k > 0:
      candidates = _select_with_role_guards(candidates, int(top_k))
  else:
    raw_area = template.metadata.get("benchmark_v2_part_surface_area")
    raw_scale = template.metadata.get("benchmark_v2_invariant_part_scale")
    if not isinstance(raw_area, (int, float)) or not isinstance(
        raw_scale, (int, float)
    ):
      raise ValueError(
          "benchmark_v2 template is missing invariant surface normalization"
      )
    part_surface_area = float(raw_area)
    invariant_part_scale = float(raw_scale)
    if sanitized_model_views is not None:
      supplied_views = list(sanitized_model_views)
      if len(supplied_views) != len(candidates):
        raise ValueError(
            "sanitized_model_views and candidates must have the same length"
        )
      for candidate, supplied in zip(candidates, supplied_views):
        if not isinstance(supplied, ModelViewSanitization):
          raise TypeError(
              "sanitized_model_views must contain ModelViewSanitization values"
          )
        validate_benchmark_v2_model_view(
            supplied.model_view,
            expected_sha256=supplied.sha256,
        )
        recomputed = sanitize_benchmark_v2_model_view(
            candidate.to_dict(),
            invariant_part_scale=invariant_part_scale,
            part_surface_area=part_surface_area,
            formal=True,
        )
        if recomputed.sha256 != supplied.sha256:
          raise ValueError(
              "sanitized_model_views must align one-to-one with candidates"
          )
        if candidate.score is None or not np.isfinite(float(candidate.score)):
          raise ValueError(
              "pre-sanitized benchmark_v2 candidates must be pre-scored"
          )
        model_inputs[id(candidate)] = InterfaceModelInput(
            protocol="benchmark_v2",
            feature_names=tuple(BENCHMARK_V2_FEATURE_NAMES),
            vector=tuple(benchmark_v2_numeric_feature_vector(supplied)),
            sanitization=supplied,
        )
    else:
      scorer.score_candidates(
          candidates,
          protocol="benchmark_v2",
          invariant_part_scale=invariant_part_scale,
          part_surface_area=part_surface_area,
      )
      for candidate in candidates:
        model_inputs[id(candidate)] = scorer.model_input_for_candidate(
            candidate,
            protocol="benchmark_v2",
            invariant_part_scale=invariant_part_scale,
            part_surface_area=part_surface_area,
        )
    for candidate in candidates:
      model_input = model_inputs[id(candidate)]
      if model_input.sanitization is None:
        raise ValueError("benchmark_v2 scorer returned no formal sanitization")
      role = _benchmark_role_from_model_input(model_input)
      safe_roles[id(candidate)] = role
      if bool(
          model_input.sanitization.model_view["topology"].get(
              "composite_interface", False
          )
      ):
        raw = 0.0 if candidate.score is None else float(candidate.score)
        boost = 0.26 if role in {
            "center_bore",
            "threaded_hole",
            "obround_slot",
            "pin_boss",
        } else 0.18
        candidate.score = min(1.0, raw + boost)
    candidates = [
        item for item in candidates if item.score is not None and item.score >= min_score
    ]
    candidates.sort(
        key=lambda item: (
            -(item.score or 0.0),
            model_inputs[id(item)].sanitization.canonical_json,
            item.interface_id,
        )
    )
    if top_k > 0:
      candidates = candidates[: int(top_k)]

  notes: list[str] = []
  for index, candidate in enumerate(candidates):
    private_exact_binding = (
        None
        if private_exact_bindings is None
        else private_exact_bindings.get(candidate.interface_id)
    )
    if private_exact_bindings is not None and private_exact_binding is None:
      raise ValueError(
          "benchmark_v2 candidate lacks its private exact-interface binding"
      )
    socket = _socket_from_candidate(
        candidate,
        index,
        protocol=protocol,
        model_input=model_inputs.get(id(candidate)),
        role_override=safe_roles.get(id(candidate)),
        private_exact_binding=private_exact_binding,
    )
    name = socket.name
    suffix = 1
    while name in template.sockets:
      name = f"{socket.name}_{suffix:02d}"
      suffix += 1
    socket.name = name
    template.sockets[name] = socket
  if candidates:
    template.metadata["learned_interface_socket_count"] = len(candidates)
    template.metadata["learned_interface_model"] = "logistic_interface_affordance_scorer"
    notes.append(
        f"learned_interfaces={part_name}:{len(candidates)};top_score={float(candidates[0].score or 0.0):.4f}"
    )
  return notes


def _select_with_role_guards(
    candidates: list[CandidateInterface],
    top_k: int,
) -> list[CandidateInterface]:
  """Keep high-value mechanical roles even when raw scorer ranks them lower.

  Industrial STEP faces often produce many tiny fillet or grip interfaces. A
  purely score-sorted top-k list can therefore drop the only central bore or
  obround slot, after which grounding is impossible no matter how good the pair
  scorer is. This guard is still leakage-free: it only uses the single-part
  role hypothesis, not assembly contacts or target pose.
  """

  if top_k <= 0 or len(candidates) <= top_k:
    return candidates
  quotas = {
      "threaded_hole": 4,
      "center_bore": 4,
      "obround_slot": 4,
      "pin_boss": 3,
      "shaft_axis": 3,
      "planar_seat": 2,
      "shoulder_stop": 2,
  }
  selected: list[CandidateInterface] = []
  seen: set[str] = set()
  by_role: dict[str, list[CandidateInterface]] = {}
  for candidate in candidates:
    by_role.setdefault(str(candidate.role_hint or "").lower(), []).append(candidate)
  for role, quota in quotas.items():
    for candidate in by_role.get(role, [])[:quota]:
      if candidate.interface_id in seen:
        continue
      selected.append(candidate)
      seen.add(candidate.interface_id)
      if len(selected) >= top_k:
        return selected[:top_k]
  for candidate in candidates:
    if candidate.interface_id in seen:
      continue
    selected.append(candidate)
    seen.add(candidate.interface_id)
    if len(selected) >= top_k:
      break
  selected.sort(key=lambda item: (-(item.score or 0.0), item.interface_id))
  return selected[:top_k]


def rank_interface_socket_pairs(
    *,
    parent_id: str,
    child_id: str,
    parent_sockets: list[Socket],
    child_sockets: list[Socket],
    relation_hint: str,
    radius_penalty_scale: float = 3.2,
    radius_penalty_cap: float = 3.25,
    pair_scorer: Optional[InterfacePairScorer] = None,
    pair_rank_mode: str = "heuristic",
    learned_pair_score_weight: float = 4.0,
    protocol: str = "legacy",
) -> list[InterfacePairCandidate]:
  protocol = str(protocol or "legacy").strip().lower()
  if protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("protocol must be 'legacy' or 'benchmark_v2'")
  parent_interfaces = [s for s in parent_sockets if _is_learned_interface(s)]
  child_interfaces = [s for s in child_sockets if _is_learned_interface(s)]
  pairs: list[InterfacePairCandidate] = []
  for parent_socket in parent_interfaces:
    for child_socket in child_interfaces:
      parent_role = _role(parent_socket)
      child_role = _role(child_socket)
      if protocol == "legacy" and _invalid_support_fastener_pair(
          parent_id=parent_id,
          child_id=child_id,
          parent_role=parent_role,
          child_role=child_role,
      ):
        continue
      compatibility = _role_compatibility(
          parent_role,
          child_role,
          relation_hint=relation_hint,
      )
      if compatibility <= 0.0:
        continue
      radius_penalty = _radius_penalty(
          parent_socket,
          child_socket,
          scale=radius_penalty_scale,
          cap=radius_penalty_cap,
      )
      context_adjustment = 0.0
      if protocol == "legacy":
        context_adjustment = _contextual_pair_adjustment(
            parent_id=parent_id,
            child_id=child_id,
            parent_role=parent_role,
            child_role=child_role,
            relation_hint=relation_hint,
        )
      parent_score = _score(parent_socket)
      child_score = _score(child_socket)
      heuristic_score = (
          4.25
          + 1.25 * parent_score
          + 1.25 * child_score
          + compatibility
          + context_adjustment
          - radius_penalty
      )
      learned_score = 0.0
      predicted_contact_type = ""
      contact_type_probs: dict[str, float] = {}
      if pair_scorer is not None:
        try:
          learned_score = float(
              pair_scorer.score_sockets(
                  parent_socket,
                  child_socket,
                  relation_hint=relation_hint,
                  protocol=protocol,
              )
          )
          predicted_contact_type, contact_type_probs = (
              pair_scorer.predict_contact_type_sockets(
                  parent_socket,
                  child_socket,
                  relation_hint=relation_hint,
                  protocol=protocol,
              )
          )
        except Exception:
          if protocol == "benchmark_v2":
            raise
          learned_score = 0.0
          predicted_contact_type = ""
          contact_type_probs = {}
      mode = str(pair_rank_mode or "heuristic").lower()
      if pair_scorer is None or mode == "heuristic":
        score = heuristic_score
      elif mode == "learned":
        score = 10.0 * learned_score
      else:
        score = heuristic_score + float(learned_pair_score_weight) * (
            learned_score - 0.5
        )
      if score <= 0.0:
        continue
      relation_mode = _relation_mode(parent_role, child_role)
      pairs.append(
          InterfacePairCandidate(
              score=float(score),
              parent_socket=parent_socket,
              child_socket=child_socket,
              relation_mode=relation_mode,
              note=(
                  f"interface_pair {parent_id}.{parent_socket.name}"
                  f"({parent_role},{parent_score:.3f}) -> "
                  f"{child_id}.{child_socket.name}"
                  f"({child_role},{child_score:.3f}) "
                  f"heuristic={heuristic_score:.3f} learned={learned_score:.3f}"
              ),
              heuristic_score=float(heuristic_score),
              learned_pair_score=float(learned_score),
              predicted_contact_type=str(predicted_contact_type or ""),
              contact_type_probs=dict(contact_type_probs),
          )
      )
  pairs.sort(key=lambda item: (-item.score, item.parent_socket.name, item.child_socket.name))
  return pairs


def _benchmark_role_from_model_input(model_input: InterfaceModelInput) -> str:
  if model_input.sanitization is None:
    return "generic_interface"
  view = model_input.sanitization.model_view
  topology = view.get("topology", {})
  role = str(topology.get("composite_role") or "").strip().lower()
  if role:
    return role
  surface = str(view.get("surface_type") or "other").strip().lower()
  if surface == "plane":
    return "planar_seat"
  if surface == "cylinder":
    return "cylindrical_interface"
  if surface in {"cone", "torus"}:
    return "shoulder_stop"
  return "generic_interface"


def _socket_from_candidate(
    candidate: CandidateInterface,
    index: int,
    *,
    protocol: str = "legacy",
    model_input: InterfaceModelInput | None = None,
    role_override: str | None = None,
    private_exact_binding: dict[str, Any] | None = None,
) -> Socket:
  frame = candidate.local_frame
  if protocol == "benchmark_v2":
    if model_input is None or model_input.sanitization is None:
      raise ValueError("benchmark_v2 socket requires a sanitized model input")
    if private_exact_binding is not None:
      expected_keys = {
          "private_exact_interface_id",
          "private_exact_raw_face_indices",
          "private_exact_raw_face_signature_sha256s",
          "private_exact_face_identity_proof",
          "private_exact_binding_sha256",
      }
      if set(private_exact_binding) != expected_keys:
        raise ValueError("private exact-interface binding has an invalid schema")
      if private_exact_binding["private_exact_interface_id"] != candidate.interface_id:
        raise ValueError("private exact-interface binding targets another candidate")
      raw_indices = private_exact_binding["private_exact_raw_face_indices"]
      if (
          not isinstance(raw_indices, list)
          or not raw_indices
          or any(not isinstance(value, int) or value < 0 for value in raw_indices)
      ):
        raise ValueError("private exact-interface binding has invalid face identities")
      raw_signatures = private_exact_binding[
          "private_exact_raw_face_signature_sha256s"
      ]
      if (
          not isinstance(raw_signatures, list)
          or len(raw_signatures) != len(raw_indices)
          or any(
              not isinstance(value, str)
              or len(value) != 64
              or any(character not in "0123456789abcdef" for character in value)
              for value in raw_signatures
          )
      ):
        raise ValueError(
            "private exact-interface binding has invalid STEP face signatures"
        )
      if private_exact_binding["private_exact_face_identity_proof"] not in {
          "occ_tshape_partner_bijection",
          "unique_signature_bijection",
      }:
        raise ValueError("private exact-interface binding has an invalid proof")
      binding_sha256 = str(private_exact_binding["private_exact_binding_sha256"])
      if len(binding_sha256) != 64 or any(
          character not in "0123456789abcdef" for character in binding_sha256
      ):
        raise ValueError("private exact-interface binding has an invalid hash")
    score = 0.0 if candidate.score is None else float(candidate.score)
    role = str(role_override or "generic_interface")
    metadata = {
        "source": "benchmark_v2_interface_scorer",
        "learned_interface": True,
        "model_input_protocol": "benchmark_v2",
        "benchmark_v2_model_view": model_input.sanitization.model_view,
        "benchmark_v2_model_view_sha256": model_input.sanitization.sha256,
        "interface_score": score,
        "interface_role": role,
        "surface_type": str(
            model_input.sanitization.model_view.get("surface_type") or "other"
        ),
        "quality": score,
        "mating_enabled": True,
    }
    for key in (
        "private_exact_interface_id",
        "private_exact_raw_face_indices",
        "private_exact_raw_face_signature_sha256s",
        "private_exact_face_identity_proof",
        "private_exact_binding_sha256",
    ):
      if private_exact_binding is not None and key in private_exact_binding:
        value = private_exact_binding[key]
        metadata[key] = list(value) if isinstance(value, list) else value
    return Socket(
        name=f"interface_{index:02d}_{_safe_role(role)}",
        kind="interface",
        origin=np.asarray(frame["origin"], dtype=float),
        axis=None if frame.get("axis") is None else np.asarray(frame.get("axis"), dtype=float),
        normal=None if frame.get("normal") is None else np.asarray(frame.get("normal"), dtype=float),
        x_axis=None if frame.get("x_axis") is None else np.asarray(frame.get("x_axis"), dtype=float),
        y_axis=None if frame.get("y_axis") is None else np.asarray(frame.get("y_axis"), dtype=float),
        z_axis=None if frame.get("z_axis") is None else np.asarray(frame.get("z_axis"), dtype=float),
        radius=(
            None
            if candidate.metadata.get("radius") is None
            else float(candidate.metadata.get("radius"))
        ),
        metadata=metadata,
    )
  metadata = dict(candidate.metadata)
  score = 0.0 if candidate.score is None else float(candidate.score)
  role = str(candidate.role_hint or "generic_interface")
  metadata.update(
      {
          "source": "learned_interface_scorer",
          "learned_interface": True,
          "interface_id": candidate.interface_id,
          "interface_score": score,
          "interface_features": dict(candidate.features),
          "interface_role": role,
          "surface_type": candidate.surface_type,
          "face_index": candidate.face_index,
          "quality": score,
          "mating_enabled": True,
          "frame_variants": {
              "primary": {
                  "origin": frame.get("origin"),
                  "axis": frame.get("axis"),
                  "normal": frame.get("normal"),
                  "x_axis": frame.get("x_axis"),
                  "y_axis": frame.get("y_axis"),
                  "z_axis": frame.get("z_axis"),
                  "radius": metadata.get("radius"),
              },
              "support": {
                  "origin": frame.get("origin"),
                  "axis": frame.get("axis"),
                  "normal": frame.get("normal"),
                  "x_axis": frame.get("x_axis"),
                  "y_axis": frame.get("y_axis"),
                  "z_axis": frame.get("z_axis"),
                  "radius": metadata.get("radius"),
              },
          },
          "frame_variant_names": ["primary", "support"],
      }
  )
  return Socket(
      name=f"interface_{index:02d}_{_safe_role(role)}",
      kind="interface",
      origin=np.asarray(frame["origin"], dtype=float),
      axis=None if frame.get("axis") is None else np.asarray(frame.get("axis"), dtype=float),
      normal=None if frame.get("normal") is None else np.asarray(frame.get("normal"), dtype=float),
      x_axis=None if frame.get("x_axis") is None else np.asarray(frame.get("x_axis"), dtype=float),
      y_axis=None if frame.get("y_axis") is None else np.asarray(frame.get("y_axis"), dtype=float),
      z_axis=None if frame.get("z_axis") is None else np.asarray(frame.get("z_axis"), dtype=float),
      radius=None if metadata.get("radius") is None else float(metadata.get("radius")),
      metadata=metadata,
  )


def _is_learned_interface(socket: Socket) -> bool:
  return bool(socket.metadata.get("learned_interface", False))


def _role(socket: Socket) -> str:
  return str(socket.metadata.get("interface_role") or "generic_interface").strip().lower()


def _score(socket: Socket) -> float:
  raw = socket.metadata.get("interface_score", socket.metadata.get("quality", 0.0))
  return float(raw) if isinstance(raw, (int, float)) else 0.0


def _safe_role(role: str) -> str:
  return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in role.lower())[:32] or "generic"


def _role_compatibility(role_a: str, role_b: str, relation_hint: str) -> float:
  hint = str(relation_hint or "link").strip().lower()
  roles = {role_a, role_b}
  if "slot_arc" in roles:
    return 0.0
  planar_roles = {"planar_seat", "shoulder_stop"}
  hole_roles = {"hole_entry", "center_bore", "threaded_hole"}
  pin_roles = {"shaft_axis", "cylindrical_interface", "pin_boss"}
  slot_roles = {"obround_slot"}
  cylindrical_roles = hole_roles | pin_roles | slot_roles
  if (
      (role_a in planar_roles and role_b in cylindrical_roles)
      or (role_b in planar_roles and role_a in cylindrical_roles)
  ):
    return 0.0
  if (role_a in slot_roles and role_b not in pin_roles) or (
      role_b in slot_roles and role_a not in pin_roles
  ):
    return 0.0
  if role_a == "shaft_axis" and role_b == "shaft_axis":
    base = 0.12
  elif role_a == role_b and role_a in hole_roles:
    base = 0.10
  elif (role_a in {"center_bore", "threaded_hole"} and role_b in pin_roles) or (
      role_b in {"center_bore", "threaded_hole"} and role_a in pin_roles
  ):
    base = 3.10
  elif (role_a == "obround_slot" and role_b in {"pin_boss", "shaft_axis"}) or (
      role_b == "obround_slot" and role_a in {"pin_boss", "shaft_axis"}
  ):
    base = 3.00
  elif role_a == "cylindrical_interface" and role_b == "cylindrical_interface":
    base = 0.35
  elif "generic_interface" in roles:
    base = 0.35
  elif roles <= planar_roles:
    base = 1.45
  elif bool(roles & hole_roles) and bool(roles & pin_roles):
    base = 2.15
  elif roles <= cylindrical_roles:
    base = 0.65
  else:
    base = 0.55

  if hint in {"coaxial", "fasten"}:
    if bool(roles & (hole_roles | pin_roles)):
      base += 0.65
      if role_a == role_b:
        base -= 0.75
    else:
      base -= 0.45
  elif hint in {"planar", "support"}:
    if role_a in {"planar_seat", "shoulder_stop"} and role_b in {"planar_seat", "shoulder_stop"}:
      base += 0.75
    else:
      base -= 0.25
  return max(0.0, float(base))


def _contextual_pair_adjustment(
    *,
    parent_id: str,
    child_id: str,
    parent_role: str,
    child_role: str,
    relation_hint: str,
) -> float:
  """Mechanical prior for choosing which two learned interfaces form a mate.

  The single-interface scorer answers "is this face useful?". This prior answers a
  different question: "do these two useful faces make a legal pair?" It is still
  leakage-free because it only uses role labels, relation hints, and part names.
  """

  hint = str(relation_hint or "link").strip().lower()
  adjustment = 0.0
  parent_is_shaft = _is_shaft_like(parent_id)
  child_is_shaft = _is_shaft_like(child_id)
  parent_is_support = _is_support_like(parent_id)
  child_is_support = _is_support_like(child_id)

  if parent_role == child_role == "shaft_axis":
    adjustment -= 1.6
  if parent_role == child_role and parent_role in {"hole_entry", "center_bore", "threaded_hole"}:
    adjustment -= 1.3

  parent_cyl = parent_role in {"shaft_axis", "cylindrical_interface", "pin_boss"}
  child_cyl = child_role in {"shaft_axis", "cylindrical_interface", "pin_boss"}
  parent_hole = parent_role in {"hole_entry", "center_bore", "threaded_hole"}
  child_hole = child_role in {"hole_entry", "center_bore", "threaded_hole"}
  parent_slot = parent_role == "obround_slot"
  child_slot = child_role == "obround_slot"
  parent_planar = parent_role in {"planar_seat", "shoulder_stop"}
  child_planar = child_role in {"planar_seat", "shoulder_stop"}

  if hint in {"planar", "support"}:
    if parent_planar and child_planar:
      adjustment += 0.65
    elif parent_cyl or child_cyl or parent_hole or child_hole:
      adjustment -= 0.55

  if parent_is_support and child_is_support:
    if parent_planar and child_planar:
      adjustment += 2.0
    if (parent_slot and child_cyl) or (child_slot and parent_cyl):
      if "slot" in hint or "slide" in hint or "guide" in hint:
        adjustment += 0.4
      else:
        adjustment -= 2.25

  if hint in {"coaxial", "fasten"}:
    if (parent_hole and child_cyl) or (child_hole and parent_cyl):
      adjustment += 0.75
    elif parent_role == child_role:
      adjustment -= 0.55

  if parent_is_support and child_is_shaft:
    if parent_hole and child_cyl:
      adjustment += 1.2
    if parent_cyl and child_hole:
      adjustment += 0.4
    if parent_role == "pin_boss" and child_cyl:
      adjustment -= 0.8
  if child_is_support and parent_is_shaft:
    if child_hole and parent_cyl:
      adjustment += 1.2
    if child_cyl and parent_hole:
      adjustment += 0.4
    if child_role == "pin_boss" and parent_cyl:
      adjustment -= 0.8

  if parent_is_shaft and not child_is_shaft:
    if parent_cyl and child_hole:
      adjustment += 0.95
    if parent_hole and child_cyl:
      adjustment -= 0.75
  if child_is_shaft and not parent_is_shaft:
    if child_cyl and parent_hole:
      adjustment += 0.95
    if child_hole and parent_cyl:
      adjustment -= 0.75

  if (parent_slot and child_cyl) or (child_slot and parent_cyl):
    adjustment += 1.25
    if "slot" in hint or "insert" in hint:
      adjustment += 0.75

  return float(adjustment)


def _is_shaft_like(part_id: str) -> bool:
  text = part_id.lower()
  tokens = ("shaft", "axle", "pin", "bolt", "screw", "rod", "arbor", "spindle")
  return any(token in text for token in tokens)


def _is_support_like(part_id: str) -> bool:
  text = part_id.lower()
  tokens = (
      "base",
      "frame",
      "plate",
      "bracket",
      "housing",
      "mount",
      "support",
      "block",
      "holder",
      "toolholder",
  )
  return any(token in text for token in tokens)


def _invalid_support_fastener_pair(
    *,
    parent_id: str,
    child_id: str,
    parent_role: str,
    child_role: str,
) -> bool:
  support_hole_roles = {"hole_entry", "center_bore", "threaded_hole"}
  parent_support = _is_support_like(parent_id)
  child_support = _is_support_like(child_id)
  parent_fastener = _is_shaft_like(parent_id)
  child_fastener = _is_shaft_like(child_id)
  if parent_support and child_fastener:
    return parent_role not in (support_hole_roles | {"shaft_axis", "cylindrical_interface", "pin_boss"})
  if child_support and parent_fastener:
    return child_role not in (support_hole_roles | {"shaft_axis", "cylindrical_interface", "pin_boss"})
  return False


def _relation_mode(role_a: str, role_b: str) -> str:
  roles = {role_a, role_b}
  if "obround_slot" in roles and bool(roles & {"pin_boss", "shaft_axis", "cylindrical_interface"}):
    return "pin_in_slot"
  if bool(roles & {"hole_entry", "center_bore", "threaded_hole"}) and bool(
      roles & {"shaft_axis", "cylindrical_interface", "pin_boss"}
  ):
    return "interface_insert"
  if roles <= {"planar_seat", "shoulder_stop"}:
    return "interface_support"
  return "interface_mate"


def _radius_penalty(
    socket_a: Socket,
    socket_b: Socket,
    *,
    scale: float = 3.2,
    cap: float = 3.25,
) -> float:
  if socket_a.radius is None or socket_b.radius is None:
    return 0.0
  max_radius = max(abs(float(socket_a.radius)), abs(float(socket_b.radius)), 1e-6)
  rel = abs(float(socket_a.radius) - float(socket_b.radius)) / max_radius
  return min(float(cap), float(scale) * rel)
