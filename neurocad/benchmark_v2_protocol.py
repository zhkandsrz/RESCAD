"""Shared production boundary for the pose-scrambled benchmark-v2 protocol.

This module is the only supported route from raw STEP paths to benchmark-v2
templates, shapes, and sanitized model views.  It deliberately keeps gauge
provenance separate from every object that can be consumed by a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import hmac
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .benchmark_v2_model_view import (
    ModelViewSanitization,
    sanitize_benchmark_v2_model_view,
)
from .benchmark_v2_constants import PROTOCOL_VERSION, exact_face_binding_sha256
from .cadquery_backend import (
    build_template_from_shape,
    extract_brep_faces_from_shape,
    load_step_shape,
    source_face_signature_sha256,
)
from .domain_types import PartTemplate, Transform
from .frame_randomization import (
    PartGaugeSet,
    RandomizedPartShapes,
    randomize_loaded_part_shapes,
    transform_interface_frame,
    transform_relative_label,
)
from .interface_features import CandidateInterface, extract_candidate_interfaces_from_shape
from .interface_grounder import augment_template_with_scored_interfaces
from .offline import SocketExtractionConfig


_RUNTIME_OPAQUE_ID_PATTERN = re.compile(r"^opaque_(?:part|body)_[0-9]{3,}$")


def _is_runtime_opaque_identity(value: Any) -> bool:
  """Return whether ``value`` belongs to the reserved model-facing namespace."""

  return bool(_RUNTIME_OPAQUE_ID_PATTERN.fullmatch(str(value).strip()))


@dataclass(frozen=True, slots=True)
class BenchmarkV2PartSpec:
  """One assembly-body instance at the non-model loading boundary.

  ``part_key`` is the unique instance alias. ``body_uuid``, ``geometry_asset``,
  and ``step_path`` identify reusable source geometry and therefore are not, by
  themselves, safe instance lookup keys.
  """

  part_key: str
  part_name: str
  body_uuid: str
  step_path: Path
  geometry_asset: str | None = None

  def __post_init__(self) -> None:
    for name in ("part_key", "part_name", "body_uuid"):
      if not str(getattr(self, name)).strip():
        raise ValueError(f"{name} must be a nonempty string")
    if self.geometry_asset is not None and not str(self.geometry_asset).strip():
      raise ValueError("geometry_asset must be a nonempty string when supplied")
    object.__setattr__(self, "step_path", Path(self.step_path))


class BenchmarkV2PrivateIdentityContext:
  """Ephemeral raw/opaque identity bridge for supervision and audit code only.

  The object deliberately provides no serialization method and refuses pickle.
  Model-facing structures receive only values returned by ``opaque_*`` methods.
  """

  __slots__ = (
      "_opaque_part_by_source",
      "_opaque_body_by_source",
      "_raw_body_by_opaque",
      "_raw_part_by_opaque",
      "_gauge_key_by_source",
      "_interface_namespace_by_source",
      "_source_by_alias",
      "_instruction_aliases_by_source",
      "_face_maps_by_source",
      "_face_inverse_maps_by_source",
      "_face_proofs_by_source",
      "_raw_face_signatures_by_source",
  )

  def __init__(
      self,
      *,
      opaque_part_by_source: Mapping[str, str],
      opaque_body_by_source: Mapping[str, str],
      raw_body_by_opaque: Mapping[str, str],
      raw_part_by_opaque: Mapping[str, str],
      gauge_key_by_source: Mapping[str, str],
      interface_namespace_by_source: Mapping[str, str],
      source_by_alias: Mapping[str, str],
      instruction_aliases_by_source: Mapping[str, Sequence[str]],
  ) -> None:
    self._opaque_part_by_source = MappingProxyType(dict(opaque_part_by_source))
    self._opaque_body_by_source = MappingProxyType(dict(opaque_body_by_source))
    self._raw_body_by_opaque = MappingProxyType(dict(raw_body_by_opaque))
    self._raw_part_by_opaque = MappingProxyType(dict(raw_part_by_opaque))
    self._gauge_key_by_source = MappingProxyType(dict(gauge_key_by_source))
    self._interface_namespace_by_source = MappingProxyType(
        dict(interface_namespace_by_source)
    )
    self._source_by_alias = MappingProxyType(dict(source_by_alias))
    self._instruction_aliases_by_source = MappingProxyType(
        {
            key: tuple(str(value) for value in values if str(value).strip())
            for key, values in instruction_aliases_by_source.items()
        }
    )
    self._face_maps_by_source: dict[str, Mapping[int, int]] = {}
    self._face_inverse_maps_by_source: dict[str, Mapping[int, int]] = {}
    self._face_proofs_by_source: dict[str, str] = {}
    self._raw_face_signatures_by_source: dict[str, Mapping[int, str]] = {}

  def __repr__(self) -> str:
    return "<BenchmarkV2PrivateIdentityContext redacted>"

  def __reduce_ex__(self, _protocol):
    raise TypeError("benchmark-v2 private identity context is not serializable")

  def _source_for(self, identity: str) -> str:
    value = str(identity).strip()
    source = self._source_by_alias.get(value)
    if source is None:
      raise KeyError(f"Unknown benchmark-v2 private identity: {identity!r}")
    return source

  def opaque_part_for(self, raw_identity: str) -> str:
    return self._opaque_part_by_source[self._source_for(raw_identity)]

  def opaque_body_for(self, raw_identity: str) -> str:
    return self._opaque_body_by_source[self._source_for(raw_identity)]

  def raw_body_for(self, opaque_body: str) -> str:
    try:
      return self._raw_body_by_opaque[str(opaque_body)]
    except KeyError as error:
      raise KeyError(f"Unknown opaque body identity: {opaque_body!r}") from error

  def raw_part_for(self, opaque_part: str) -> str:
    try:
      return self._raw_part_by_opaque[str(opaque_part)]
    except KeyError as error:
      raise KeyError(f"Unknown opaque part identity: {opaque_part!r}") from error

  def private_gauge_key_for(self, raw_identity: str) -> str:
    return self._gauge_key_by_source[self._source_for(raw_identity)]

  def private_interface_namespace_for(self, raw_identity: str) -> str:
    return self._interface_namespace_by_source[self._source_for(raw_identity)]

  def _register_face_identity_map(
      self,
      raw_identity: str,
      raw_to_randomized: Mapping[int, int],
      proof: str,
      raw_face_signature_by_index: Mapping[int, str],
  ) -> None:
    source = self._source_for(raw_identity)
    mapping = {int(key): int(value) for key, value in raw_to_randomized.items()}
    if len(set(mapping.values())) != len(mapping):
      raise ValueError("benchmark_v2 face identity map must be bijective")
    signatures = {
        int(index): str(signature)
        for index, signature in raw_face_signature_by_index.items()
    }
    if set(signatures) != set(mapping) or any(
        re.fullmatch(r"[0-9a-f]{64}", signature) is None
        for signature in signatures.values()
    ):
      raise ValueError(
          "benchmark_v2 raw face signatures must exactly cover the identity map"
      )
    self._face_maps_by_source[source] = MappingProxyType(mapping)
    self._face_inverse_maps_by_source[source] = MappingProxyType(
        {value: key for key, value in mapping.items()}
    )
    self._face_proofs_by_source[source] = str(proof)
    self._raw_face_signatures_by_source[source] = MappingProxyType(signatures)

  def randomized_face_index_for(self, raw_identity: str, face_index: int) -> int:
    source = self._source_for(raw_identity)
    try:
      return int(self._face_maps_by_source[source][int(face_index)])
    except KeyError as error:
      raise KeyError(
          f"No stable randomized face for {raw_identity!r} face {face_index}"
      ) from error

  def raw_face_index_for(self, raw_identity: str, face_index: int) -> int:
    source = self._source_for(raw_identity)
    try:
      return int(self._face_inverse_maps_by_source[source][int(face_index)])
    except KeyError as error:
      raise KeyError(
          f"No stable raw face for {raw_identity!r} face {face_index}"
      ) from error

  def face_identity_proof_for(self, raw_identity: str) -> str:
    source = self._source_for(raw_identity)
    try:
      return self._face_proofs_by_source[source]
    except KeyError as error:
      raise KeyError(f"No face identity proof for {raw_identity!r}") from error

  def validate_raw_face_signatures(
      self,
      raw_identity: str,
      face_indices: Sequence[int],
      expected_signatures: Sequence[str],
  ) -> None:
    """Bind private receipt rows to the faces in the live loaded STEP."""

    source = self._source_for(raw_identity)
    actual = self._raw_face_signatures_by_source.get(source)
    if actual is None:
      raise KeyError(f"No loaded STEP face signatures for {raw_identity!r}")
    for index, expected in zip(face_indices, expected_signatures):
      if actual.get(index) != expected:
        raise ValueError(
            "receipt source face signature does not match the loaded STEP face"
        )

  def sanitize_model_instruction(self, instruction: str) -> str:
    """Replace known raw identity channels before an instruction reaches a model."""

    sanitized = str(instruction)
    aliases: dict[str, set[str]] = {}
    for source, values in self._instruction_aliases_by_source.items():
      for value in values:
        aliases.setdefault(value.casefold(), set()).add(source)
        for token in re.findall(r"[A-Za-z0-9]+", value):
          if len(token) >= 4 and token.casefold() not in {
              "part",
              "body",
              "component",
              "assembly",
              "step",
          }:
            aliases.setdefault(token.casefold(), set()).add(source)
    ordered = sorted(aliases, key=lambda value: (-len(value), value))
    for alias_key in ordered:
      sources = aliases[alias_key]
      pattern = re.compile(re.escape(alias_key), flags=re.IGNORECASE)
      if len(sources) == 1:
        source = next(iter(sources))
        sanitized = pattern.sub(self._opaque_part_by_source[source], sanitized)
      elif pattern.search(sanitized):
        raise ValueError(
            f"benchmark_v2 instruction contains ambiguous raw identity token: {alias_key!r}"
        )
    for alias_key in ordered:
      if re.search(re.escape(alias_key), sanitized, flags=re.IGNORECASE):
        raise ValueError(
            f"benchmark_v2 instruction retained raw identity token: {alias_key!r}"
        )
    return sanitized

  def restore_model_output(self, value: Any) -> Any:
    """Restore raw audit identities only after opaque model execution completes."""

    replacements = {
        **self._raw_part_by_opaque,
        **self._raw_body_by_opaque,
    }
    if isinstance(value, str):
      if not replacements:
        return value
      pattern = re.compile(
          "|".join(
              re.escape(opaque)
              for opaque in sorted(replacements, key=lambda item: -len(item))
          )
      )
      return pattern.sub(lambda match: replacements[match.group(0)], value)
    if isinstance(value, list):
      return [self.restore_model_output(item) for item in value]
    if isinstance(value, tuple):
      return tuple(self.restore_model_output(item) for item in value)
    if isinstance(value, Mapping):
      return {
          self.restore_model_output(key): self.restore_model_output(item)
          for key, item in value.items()
      }
    return value


def _canonical_source_identity(spec: BenchmarkV2PartSpec) -> str:
  """Return a path/order-independent private identity for HMAC binding."""

  payload = {
      "body_uuid": str(spec.body_uuid).strip().casefold(),
      "part_key": str(spec.part_key).strip().casefold(),
  }
  if spec.geometry_asset is not None:
    payload["geometry_asset"] = str(spec.geometry_asset).strip().casefold()
  return json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
  )


def build_benchmark_v2_identity_context(
    part_specs: Sequence[BenchmarkV2PartSpec],
    *,
    assembly_nonce: str,
    seed: int,
) -> BenchmarkV2PrivateIdentityContext:
  """Assign deterministic opaque IDs by a nonce/seed-bound HMAC permutation."""

  nonce = str(assembly_nonce).strip()
  if not nonce:
    raise ValueError("benchmark_v2 requires a nonempty assembly_nonce/case identity")
  specs = tuple(part_specs)
  if not specs:
    raise ValueError("benchmark_v2 requires at least one part")
  part_keys = [str(spec.part_key).strip() for spec in specs]
  if len(set(part_keys)) != len(part_keys):
    raise ValueError("benchmark_v2 part_key instance aliases must be unique")
  canonical_by_spec = [(spec, _canonical_source_identity(spec)) for spec in specs]
  identities = [identity for _, identity in canonical_by_spec]
  if len(set(identities)) != len(identities):
    raise ValueError("benchmark_v2 raw part identities must be unique")
  key = hashlib.sha256(
      f"{PROTOCOL_VERSION}\0{int(seed)}\0{nonce}".encode("utf-8")
  ).digest()
  ranked = sorted(
      (
          hmac.new(key, identity.encode("utf-8"), hashlib.sha256).hexdigest(),
          identity,
          spec,
      )
      for spec, identity in canonical_by_spec
  )
  opaque_part_by_source: dict[str, str] = {}
  opaque_body_by_source: dict[str, str] = {}
  raw_body_by_opaque: dict[str, str] = {}
  raw_part_by_opaque: dict[str, str] = {}
  gauge_key_by_source: dict[str, str] = {}
  interface_namespace_by_source: dict[str, str] = {}
  source_by_alias: dict[str, str] = {}
  instruction_aliases_by_source: dict[str, tuple[str, ...]] = {}
  for ordinal, (digest, identity, spec) in enumerate(ranked):
    opaque_part = f"opaque_part_{ordinal:03d}"
    opaque_body = f"opaque_body_{ordinal:03d}"
    opaque_part_by_source[identity] = opaque_part
    opaque_body_by_source[identity] = opaque_body
    raw_body_by_opaque[opaque_body] = str(spec.body_uuid)
    raw_part_by_opaque[opaque_part] = str(spec.part_name)
    gauge_key_by_source[identity] = "gauge_" + digest
    interface_key = hashlib.sha256(
        f"{PROTOCOL_VERSION}\0interface\0{nonce}".encode("utf-8")
    ).digest()
    interface_namespace_by_source[identity] = hmac.new(
        interface_key,
        identity.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    instruction_aliases_by_source[identity] = tuple(
        value
        for value in (
            str(spec.part_key),
            str(spec.part_name),
            str(spec.body_uuid),
            None if spec.geometry_asset is None else str(spec.geometry_asset),
            str(spec.step_path),
            spec.step_path.name,
            spec.step_path.stem,
        )
        if value is not None and not _is_runtime_opaque_identity(value)
    )
  # Generated identifiers own the reserved runtime namespace.  Some legacy
  # fixtures already use opaque-looking raw names; those remain available via
  # their non-reserved part keys without shadowing another generated identity.
  for identity, opaque_part in opaque_part_by_source.items():
    source_by_alias[opaque_part] = identity
    source_by_alias[opaque_body_by_source[identity]] = identity
  body_alias_counts: dict[str, int] = {}
  for spec in specs:
    raw_body = str(spec.body_uuid).strip()
    body_alias_counts[raw_body] = body_alias_counts.get(raw_body, 0) + 1
  for spec, identity in canonical_by_spec:
    aliases = [spec.part_key, spec.part_name]
    raw_body = str(spec.body_uuid).strip()
    if body_alias_counts[raw_body] == 1:
      aliases.append(raw_body)
    for alias in aliases:
      value = str(alias).strip()
      if _is_runtime_opaque_identity(value):
        continue
      previous = source_by_alias.get(value)
      if previous is not None and previous != identity:
        raise ValueError(f"benchmark_v2 identity alias is ambiguous: {value!r}")
      source_by_alias[value] = identity
  return BenchmarkV2PrivateIdentityContext(
      opaque_part_by_source=opaque_part_by_source,
      opaque_body_by_source=opaque_body_by_source,
      raw_body_by_opaque=raw_body_by_opaque,
      raw_part_by_opaque=raw_part_by_opaque,
      gauge_key_by_source=gauge_key_by_source,
      interface_namespace_by_source=interface_namespace_by_source,
      source_by_alias=source_by_alias,
      instruction_aliases_by_source=instruction_aliases_by_source,
  )


def _face_identity_signature(face: Any, *, part_scale: float) -> tuple[Any, ...]:
  scale = max(1e-12, float(part_scale))
  metadata = face.metadata if isinstance(face.metadata, dict) else {}
  axial_extent = metadata.get("axial_extent")
  return (
      str(face.surface_type).strip().lower(),
      round(float(face.area) / (scale * scale), 8),
      None if face.radius is None else round(float(face.radius) / scale, 8),
      int(face.edge_count),
      bool(face.has_helical_edge),
      int(face.concavity),
      None
      if not isinstance(axial_extent, (int, float))
      else round(float(axial_extent) / scale, 8),
  )


def _cadquery_face_tshape_partner_map(
    raw_shape: Any,
    randomized_shape: Any,
    *,
    raw_indices: set[int],
    randomized_indices: set[int],
) -> Optional[dict[int, int]]:
  """Return a proven OCC face bijection, or ``None`` when ancestry is absent.

  ``transformShape`` applies a rigid location while preserving each face's
  underlying OCCT TShape. ``IsPartner`` therefore proves source ancestry even
  when OCCT changes face enumeration. It intentionally ignores Location while
  retaining TShape identity. Reused TShapes can produce multiple partners; in
  that case this routine refuses to choose by order.
  """

  raw_value = raw_shape.val() if hasattr(raw_shape, "val") else raw_shape
  randomized_value = (
      randomized_shape.val()
      if hasattr(randomized_shape, "val")
      else randomized_shape
  )
  try:
    raw_topology = list(raw_value.Faces())
    randomized_topology = list(randomized_value.Faces())
  except Exception:
    return None
  if any(index < 0 or index >= len(raw_topology) for index in raw_indices):
    raise ValueError("Raw face descriptor index is outside OCC face topology")
  if any(
      index < 0 or index >= len(randomized_topology)
      for index in randomized_indices
  ):
    raise ValueError("Randomized face descriptor index is outside OCC face topology")

  mapping: dict[int, int] = {}
  for raw_index in sorted(raw_indices):
    raw_wrapped = getattr(raw_topology[raw_index], "wrapped", None)
    if raw_wrapped is None or not hasattr(raw_wrapped, "IsPartner"):
      return None
    partners: list[int] = []
    for randomized_index in sorted(randomized_indices):
      randomized_wrapped = getattr(
          randomized_topology[randomized_index], "wrapped", None
      )
      if randomized_wrapped is None:
        return None
      try:
        if bool(raw_wrapped.IsPartner(randomized_wrapped)):
          partners.append(randomized_index)
      except Exception:
        return None
    if len(partners) != 1:
      return None
    mapping[raw_index] = partners[0]
  if len(set(mapping.values())) != len(mapping):
    return None
  if set(mapping.values()) != randomized_indices:
    return None
  return mapping


def _stable_face_identity_map(
    raw_shape: Any,
    randomized_shape: Any,
    *,
    part_scale: float,
) -> tuple[dict[int, int], str]:
  """Prove OCC ancestry or construct an invariant unique-signature map."""

  raw_faces = extract_brep_faces_from_shape(raw_shape, protocol="benchmark_v2")
  randomized_faces = extract_brep_faces_from_shape(
      randomized_shape,
      protocol="benchmark_v2",
  )
  if len(raw_faces) != len(randomized_faces):
    raise ValueError("Rigid benchmark transform changed the B-Rep face count")
  raw_rows = [
      (int(face.metadata["face_index"]), _face_identity_signature(face, part_scale=part_scale))
      for face in raw_faces
  ]
  randomized_rows = [
      (int(face.metadata["face_index"]), _face_identity_signature(face, part_scale=part_scale))
      for face in randomized_faces
  ]
  raw_indices = {index for index, _signature in raw_rows}
  randomized_indices = {index for index, _signature in randomized_rows}
  if len(raw_indices) != len(raw_rows) or len(randomized_indices) != len(
      randomized_rows
  ):
    raise ValueError("B-Rep face descriptors contain duplicate face indices")
  occ_mapping = _cadquery_face_tshape_partner_map(
      raw_shape,
      randomized_shape,
      raw_indices=raw_indices,
      randomized_indices=randomized_indices,
  )
  if occ_mapping is not None:
    raw_signature_by_index = dict(raw_rows)
    randomized_signature_by_index = dict(randomized_rows)
    if any(
        raw_signature_by_index[raw_index]
        != randomized_signature_by_index[randomized_index]
        for raw_index, randomized_index in occ_mapping.items()
    ):
      raise ValueError(
          "Rigid benchmark transform changed an OCC-partner face signature"
      )
    return occ_mapping, "occ_tshape_partner_bijection"

  raw_by_signature: dict[tuple[Any, ...], list[int]] = {}
  randomized_by_signature: dict[tuple[Any, ...], list[int]] = {}
  for index, signature in raw_rows:
    raw_by_signature.setdefault(signature, []).append(index)
  for index, signature in randomized_rows:
    randomized_by_signature.setdefault(signature, []).append(index)
  if set(raw_by_signature) != set(randomized_by_signature):
    raise ValueError("Rigid benchmark transform changed invariant face signatures")
  mapping: dict[int, int] = {}
  for signature in raw_by_signature:
    raw_indices = raw_by_signature[signature]
    randomized_indices = randomized_by_signature[signature]
    if len(raw_indices) != 1 or len(randomized_indices) != 1:
      raise ValueError(
          "Face identity is ambiguous: a duplicate invariant signature group "
          "has no unique OCC TShape ancestry"
      )
    mapping[raw_indices[0]] = randomized_indices[0]
  return mapping, "unique_signature_bijection"


@dataclass(frozen=True, slots=True)
class PreparedBenchmarkV2Part:
  """One randomized part plus formally sanitized candidate model views."""

  template: PartTemplate
  shape: Any
  candidates: tuple[CandidateInterface, ...]
  model_views: tuple[ModelViewSanitization, ...]
  grounding_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkV2PairSupervision:
  """Gauge-updated labels kept outside all model-visible payloads."""

  relative_transform: Transform
  source_contact_point: np.ndarray | None
  target_contact_point: np.ndarray | None
  source_interface_frame: dict[str, Any] | None
  target_interface_frame: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class PreparedBenchmarkV2Assembly:
  """Prepared assembly; ``gauges`` and transforms are audit-only provenance."""

  parts: Mapping[str, PreparedBenchmarkV2Part]
  randomized: RandomizedPartShapes
  private_identity_context: BenchmarkV2PrivateIdentityContext = field(
      repr=False,
      compare=False,
  )
  protocol_version: str = PROTOCOL_VERSION

  @property
  def gauges(self) -> PartGaugeSet:
    source = self.randomized.gauges
    transforms = {
        opaque_part: source.transform_for(
            self.private_identity_context.private_gauge_key_for(opaque_part)
        )
        for opaque_part in self.model_part_names
    }
    part_seeds = {
        opaque_part: int(
            source.part_seeds[
                self.private_identity_context.private_gauge_key_for(opaque_part)
            ]
        )
        for opaque_part in self.model_part_names
    }
    return PartGaugeSet(
        seed=source.seed,
        assembly_scale=source.assembly_scale,
        translation_box_fraction=source.translation_box_fraction,
        transforms=MappingProxyType(transforms),
        part_seeds=MappingProxyType(part_seeds),
        assembly_nonce=source.assembly_nonce,
        protocol=source.protocol,
    )

  @property
  def model_part_names(self) -> tuple[str, ...]:
    return tuple(sorted(self.parts))

  def part_for(self, part_key: str) -> PreparedBenchmarkV2Part:
    identity = str(part_key)
    if identity not in self.parts:
      identity = self.private_identity_context.opaque_part_for(identity)
    try:
      return self.parts[identity]
    except KeyError as error:
      raise KeyError(f"Unknown benchmark-v2 part key: {part_key}") from error

  def raw_transform_for(self, part_identity: str) -> Transform:
    gauge_key = self.private_identity_context.private_gauge_key_for(part_identity)
    return self.randomized.raw_transform_for(gauge_key)

  def face_identity_proof_for(self, part_identity: str) -> str:
    return self.private_identity_context.face_identity_proof_for(part_identity)

  def private_source_part_for(self, part_identity: str) -> str:
    """Return the receipt-side part alias at a private producer boundary."""

    opaque_part = self.private_identity_context.opaque_part_for(part_identity)
    return self.private_identity_context.raw_part_for(opaque_part)

  def randomized_face_indices_for_mapped_occ_faces(
      self,
      part_identity: str,
      raw_occ_face_indices: Sequence[int],
      source_face_signature_sha256s: Sequence[str],
  ) -> tuple[int, ...]:
    """Map receipt-verified STEP/OCC faces into the randomized runtime frame.

    Fusion collection indices are deliberately absent from this API.  They may
    select a private receipt row, but must never be passed as OCC face indices.
    """

    supplied = list(raw_occ_face_indices)
    supplied_signatures = list(source_face_signature_sha256s)
    if (
        not supplied
        or any(not isinstance(index, int) or isinstance(index, bool) or index < 0
               for index in supplied)
        or supplied != sorted(set(supplied))
    ):
      raise ValueError(
          "receipt-mapped raw OCC face indices must be nonempty, sorted, and unique"
      )
    if len(supplied_signatures) != len(supplied) or any(
        not isinstance(signature, str)
        or re.fullmatch(r"[0-9a-f]{64}", signature) is None
        for signature in supplied_signatures
    ):
      raise ValueError(
          "receipt-mapped source face signatures must align with OCC face indices"
      )
    self.private_identity_context.validate_raw_face_signatures(
        part_identity,
        supplied,
        supplied_signatures,
    )
    mapped = tuple(
        self.private_identity_context.randomized_face_index_for(
            part_identity,
            index,
        )
        for index in supplied
    )
    if len(set(mapped)) != len(mapped):
      raise ValueError("receipt-mapped randomized face identities are ambiguous")
    return mapped

  def sanitize_model_instruction(self, instruction: str) -> str:
    return self.private_identity_context.sanitize_model_instruction(instruction)

  def restore_model_output(self, value: Any) -> Any:
    return self.private_identity_context.restore_model_output(value)

  def remap_contact_entity(self, entity: Mapping[str, Any]) -> dict[str, Any]:
    """Reject the unsafe legacy Fusion-index-as-OCC contact mapping API."""

    del entity
    raise ValueError(
        "A raw Fusion face index is only a receipt lookup key; "
        "use remap_mapped_contact_entity with a receipt-verified OCC face"
    )

  def remap_mapped_contact_entity(
      self,
      entity: Mapping[str, Any],
      *,
      raw_occ_face_index: int,
      source_face_signature_sha256: str,
  ) -> dict[str, Any]:
    """Move one receipt-verified OCC endpoint into its randomized frame."""

    raw_body = str(entity.get("body") or "").strip()
    if not raw_body:
      raise ValueError("Fusion contact entity is missing body identity")
    transform = self.raw_transform_for(raw_body)
    remapped = dict(entity)
    remapped["body"] = self.private_identity_context.opaque_body_for(raw_body)
    remapped["index"] = self.randomized_face_indices_for_mapped_occ_faces(
        raw_body,
        [raw_occ_face_index],
        [source_face_signature_sha256],
    )
    remapped["index"] = remapped["index"][0]
    for key in ("point_on_entity", "point", "origin"):
      value = entity.get(key)
      if isinstance(value, (list, tuple, np.ndarray)):
        remapped[key] = transform.apply_point(value).tolist()
    for key in ("axis", "normal", "x_axis", "y_axis", "z_axis"):
      value = entity.get(key)
      if isinstance(value, (list, tuple, np.ndarray)):
        remapped[key] = transform.apply_direction(value).tolist()
    frame = entity.get("local_frame")
    if isinstance(frame, Mapping):
      remapped["local_frame"] = transform_interface_frame(frame, transform)
    return remapped

  def audit_provenance(self) -> dict[str, object]:
    return {
        "protocol_version": self.protocol_version,
        "gauges": self.gauges.to_provenance(),
        "raw_to_randomized": {
            key: self.raw_transform_for(key).to_dict()
            for key in self.model_part_names
        },
        "model_view_sha256": {
            key: [view.sha256 for view in self.parts[key].model_views]
            for key in sorted(self.parts)
        },
        "face_identity_proof": {
            key: self.face_identity_proof_for(key)
            for key in self.model_part_names
        },
    }

  def transform_pair_supervision(
      self,
      *,
      source_part_key: str,
      target_part_key: str,
      ground_truth: Transform,
      source_contact_point: Any = None,
      target_contact_point: Any = None,
      source_interface_frame: Mapping[str, Any] | None = None,
      target_interface_frame: Mapping[str, Any] | None = None,
  ) -> BenchmarkV2PairSupervision:
    """Apply the complete raw-to-randomized transforms to pair supervision."""

    source_transform = self.raw_transform_for(source_part_key)
    target_transform = self.raw_transform_for(target_part_key)

    def _point(value: Any, transform: Transform) -> np.ndarray | None:
      if value is None:
        return None
      return transform.apply_point(np.asarray(value, dtype=float).reshape(3))

    return BenchmarkV2PairSupervision(
        relative_transform=transform_relative_label(
            source_gauge=source_transform,
            target_gauge=target_transform,
            ground_truth=ground_truth,
        ),
        source_contact_point=_point(source_contact_point, source_transform),
        target_contact_point=_point(target_contact_point, target_transform),
        source_interface_frame=(
            None
            if source_interface_frame is None
            else transform_interface_frame(source_interface_frame, source_transform)
        ),
        target_interface_frame=(
            None
            if target_interface_frame is None
            else transform_interface_frame(target_interface_frame, target_transform)
        ),
    )


def prepare_benchmark_v2_assembly(
    part_specs: Sequence[BenchmarkV2PartSpec],
    *,
    assembly_nonce: str,
    seed: int,
    extraction_config: SocketExtractionConfig | None = None,
    translation_box_fraction: float = 1.0,
    max_candidates_per_part: int = 0,
    interface_scorer: Any = None,
    interface_top_k: int = 0,
    interface_min_score: float = 0.25,
    shape_loader: Callable[[Path], Any] | None = None,
) -> PreparedBenchmarkV2Assembly:
  """Load each raw B-Rep once, randomize it, then derive all model geometry."""

  nonce = str(assembly_nonce).strip()
  if not nonce:
    raise ValueError("benchmark_v2 requires a nonempty assembly_nonce/case identity")
  specs = tuple(part_specs)
  if not specs:
    raise ValueError("benchmark_v2 requires at least one part")
  keys = [spec.part_key for spec in specs]
  names = [spec.part_name for spec in specs]
  if len(set(keys)) != len(keys):
    raise ValueError("benchmark_v2 part_key values must be unique")
  if len(set(names)) != len(names):
    raise ValueError("benchmark_v2 part_name values must be unique")
  identity_context = build_benchmark_v2_identity_context(
      specs,
      assembly_nonce=nonce,
      seed=int(seed),
  )

  # Loading is intentionally isolated in this single comprehension.  Every
  # downstream extractor receives only the already randomized in-memory shape.
  loader = load_step_shape if shape_loader is None else shape_loader
  raw_shapes = {
      identity_context.private_gauge_key_for(spec.part_key): loader(spec.step_path)
      for spec in specs
  }
  randomized = randomize_loaded_part_shapes(
      raw_shapes,
      seed=int(seed),
      assembly_nonce=nonce,
      translation_box_fraction=float(translation_box_fraction),
  )
  config = (
      SocketExtractionConfig(
          enable_proxy_sockets=False,
          frame_protocol="benchmark_v2",
      )
      if extraction_config is None
      else replace(
          extraction_config,
          enable_proxy_sockets=False,
          frame_protocol="benchmark_v2",
      )
  )

  prepared: dict[str, PreparedBenchmarkV2Part] = {}
  for spec in specs:
    opaque_part = identity_context.opaque_part_for(spec.part_key)
    opaque_body = identity_context.opaque_body_for(spec.part_key)
    gauge_key = identity_context.private_gauge_key_for(spec.part_key)
    shape = randomized.shape_for(gauge_key)
    geometry = randomized.intrinsic_geometry[gauge_key]
    raw_shape = raw_shapes[gauge_key]
    raw_shape_value = raw_shape.val() if hasattr(raw_shape, "val") else raw_shape
    raw_topology = list(raw_shape_value.Faces())
    raw_face_signature_by_index = {
        index: source_face_signature_sha256(face)
        for index, face in enumerate(raw_topology)
    }
    raw_to_randomized_faces, identity_proof = _stable_face_identity_map(
        raw_shape,
        shape,
        part_scale=geometry.characteristic_scale,
    )
    identity_context._register_face_identity_map(
        spec.part_key,
        raw_to_randomized_faces,
        identity_proof,
        raw_face_signature_by_index,
    )
    interface_namespace = identity_context.private_interface_namespace_for(
        spec.part_key
    )
    gauge_payload = randomized.raw_transform_for(gauge_key).to_dict()
    asset = build_template_from_shape(
        name=opaque_part,
        shape=shape,
        metadata={
            "frame_protocol": PROTOCOL_VERSION,
            # Exact OCC workers reload the immutable source STEP in an
            # isolated process.  Persist the private raw->randomized gauge so
            # that those workers evaluate the same geometry seen by VGT/VSEC.
            # This metadata is never part of the sanitized model view.
            "step_path": str(spec.step_path.resolve()),
            "benchmark_v2_raw_to_randomized_transform": gauge_payload,
            "benchmark_v2_exact_identity_namespace": interface_namespace,
        },
        extraction_config=config,
        protocol="benchmark_v2",
    )
    for socket in asset.template.sockets.values():
      randomized_indices: list[int] = []
      raw_face_index = socket.metadata.get("face_index")
      if isinstance(raw_face_index, int):
        randomized_indices.append(int(raw_face_index))
      for value in socket.metadata.get("member_face_indices", []):
        index = int(value)
        if index not in randomized_indices:
          randomized_indices.append(index)
      if not randomized_indices:
        continue
      raw_indices = sorted(
          identity_context.raw_face_index_for(spec.part_key, index)
          for index in randomized_indices
      )
      raw_signatures = [
          raw_face_signature_by_index[raw_index] for raw_index in raw_indices
      ]
      exact_interface_id = "opaque_exact_iface_" + hashlib.sha256(
          json.dumps(
              {
                  "namespace": interface_namespace,
                  "raw_face_indices": raw_indices,
              },
              sort_keys=True,
              separators=(",", ":"),
          ).encode("utf-8")
      ).hexdigest()[:24]
      socket.metadata.update(
          {
              "private_exact_interface_id": exact_interface_id,
              "private_exact_raw_face_indices": raw_indices,
              "private_exact_raw_face_signature_sha256s": raw_signatures,
              "private_exact_face_identity_proof": identity_proof,
              "private_exact_binding_sha256": exact_face_binding_sha256(
                  namespace=interface_namespace,
                  raw_to_randomized_transform=gauge_payload,
                  interface_id=exact_interface_id,
                  raw_face_indices=raw_indices,
                  raw_face_signature_sha256s=raw_signatures,
                  face_identity_proof=identity_proof,
              ),
          }
      )
    if any(
        bool(socket.metadata.get("proxy_socket"))
        for socket in asset.template.sockets.values()
    ):
      raise RuntimeError("benchmark_v2 emitted a forbidden AABB/world-axis proxy socket")
    candidates = tuple(
        extract_candidate_interfaces_from_shape(
            shape,
            part_name=opaque_part,
            body_uuid=opaque_body,
            max_candidates=int(max_candidates_per_part),
            protocol="benchmark_v2",
        )
    )
    model_views = tuple(
        sanitize_benchmark_v2_model_view(
            candidate.to_dict(),
            invariant_part_scale=geometry.characteristic_scale,
            part_surface_area=geometry.surface_area,
            formal=True,
        )
        for candidate in candidates
    )
    stable_ids: set[str] = set()
    private_exact_bindings: dict[str, dict[str, Any]] = {}
    for candidate, view in zip(candidates, model_views):
      randomized_indices: list[int] = []
      if candidate.face_index is not None:
        randomized_indices.append(int(candidate.face_index))
      for value in candidate.metadata.get("member_face_indices", []):
        index = int(value)
        if index not in randomized_indices:
          randomized_indices.append(index)
      raw_indices = sorted(
          identity_context.raw_face_index_for(spec.part_key, index)
          for index in randomized_indices
      )
      raw_signatures = [
          raw_face_signature_by_index[raw_index] for raw_index in raw_indices
      ]
      identity_payload = json.dumps(
          {
              "namespace": interface_namespace,
              "raw_face_indices": raw_indices,
              "model_view_sha256": view.sha256,
          },
          sort_keys=True,
          separators=(",", ":"),
      )
      stable_id = "opaque_iface_" + hashlib.sha256(
          identity_payload.encode("utf-8")
      ).hexdigest()[:24]
      if stable_id in stable_ids:
        raise ValueError("benchmark_v2 produced an ambiguous stable interface identity")
      stable_ids.add(stable_id)
      candidate.interface_id = stable_id
      private_exact_bindings[stable_id] = {
          "private_exact_interface_id": stable_id,
          "private_exact_raw_face_indices": raw_indices,
          "private_exact_raw_face_signature_sha256s": raw_signatures,
          "private_exact_face_identity_proof": identity_proof,
          "private_exact_binding_sha256": exact_face_binding_sha256(
              namespace=interface_namespace,
              raw_to_randomized_transform=gauge_payload,
              interface_id=stable_id,
              raw_face_indices=raw_indices,
              raw_face_signature_sha256s=raw_signatures,
              face_identity_proof=identity_proof,
          ),
      }
    grounding_notes: tuple[str, ...] = ()
    if interface_scorer is not None:
      score_model_views = getattr(interface_scorer, "score_model_views", None)
      if not callable(score_model_views):
        raise TypeError(
            "benchmark_v2 interface scorers must implement score_model_views; "
            "legacy score_candidates would receive forbidden raw features"
        )
      raw_scores = list(
          score_model_views(tuple(view.model_view for view in model_views))
      )
      if len(raw_scores) != len(candidates):
        raise ValueError(
            "benchmark_v2 scorer returned a different number of scores than views"
        )
      for candidate, score in zip(candidates, raw_scores):
        candidate.score = float(score)

      grounding_notes = tuple(
          augment_template_with_scored_interfaces(
              template=asset.template,
              step_path=spec.step_path,
              scorer=interface_scorer,
              part_name=opaque_part,
              body_uuid=opaque_body,
              top_k=int(interface_top_k),
              min_score=float(interface_min_score),
              shape=shape,
              candidates=candidates,
              sanitized_model_views=model_views,
              private_exact_bindings=private_exact_bindings,
              protocol="benchmark_v2",
          )
      )
    prepared[opaque_part] = PreparedBenchmarkV2Part(
        template=asset.template,
        shape=shape,
        candidates=candidates,
        model_views=model_views,
        grounding_notes=grounding_notes,
    )
  return PreparedBenchmarkV2Assembly(
      parts=MappingProxyType(prepared),
      randomized=randomized,
      private_identity_context=identity_context,
  )
