"""V19 set-valued finite-program learning and evaluation primitives.

The authenticated benchmark-v2 adapter emits one sample for each
``(opaque_query_identity, direction)`` and preserves every authorized semantic
program target.  This module consumes that exact typed sample.  It exposes only
``graph_work_id`` to the model-input side and keeps all semantic targets on a
separate supervision side.

Multiple targets may share a finite catalog entry and have different valid
program-local residuals.  They are alternatives, not duplicate training rows
and not negatives.  OOV-only samples remain present for fail-closed evaluation
but do not enter a training denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
from torch import Tensor

from .v19_program_proposal_protocol_v1 import ProposalBudget

if TYPE_CHECKING:
  from .benchmark_v2_query_semantic_adapter_verifier_v2 import (
      FixedCatalogMetadataV2,
      FixedLineageMetadataV2,
      GeometryMaterializationRequestV2,
      StructurallyAndExactRebuildCheckedQueryGraphWorkIndexV2,
      SupervisionSampleV2,
      SupervisionTargetV2,
  )


SCHEMA_VERSION = "v19_set_valued_program_learning.v1"
BENCHMARK_V2_SAMPLE_SCHEMA = "benchmark_v2_query_program_sample.v2"
_SHA256_CHARS = frozenset("0123456789abcdef")
_DIRECTIONS = frozenset({
    "entity_one_to_entity_two",
    "entity_two_to_entity_one",
})
_SPLITS = frozenset({"train", "dev"})
_SAMPLE_FIELDS = frozenset({
    "schema_version",
    "sample_identity_sha256",
    "opaque_case_identity",
    "opaque_query_identity",
    "source_contact_ordinal",
    "development_split",
    "direction",
    "graph_work_id",
    "family_cluster_id",
    "assembly_cluster_id",
    "target_count",
    "known_target_count",
    "oov_target_count",
    "classification_evaluable",
    "targets",
    "sample_payload_sha256",
})
_TARGET_FIELDS = frozenset({
    "program_id",
    "source_program_row_sha256",
    "descriptor",
    "descriptor_sha256",
    "catalog_index",
    "catalog_program_id",
    "oov",
    "residual_translation",
    "residual_rotation_row_major",
    "target_payload_sha256",
})
_GLOBAL_WEIGHT_TOKEN = object()
_CLASS_WEIGHT_TOKEN = object()
_DATASET_TOKEN = object()
_DESCRIPTOR_FIELDS = frozenset({
    "relation_hint",
    "contact_type",
    "surface_type_a",
    "surface_type_b",
})


def _canonical_sha256(value: Any) -> str:
  raw = json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(raw).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
  if (
      type(value) is not str
      or len(value) != 64
      or set(value) - _SHA256_CHARS
  ):
    raise ValueError(f"{label} is not a SHA-256 value")
  return value


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
  if type(value) is not int or value < minimum:
    raise ValueError(f"{label} must be an exact integer")
  return value


def _finite_numeric_vector(
    value: Any, *, width: int, label: str
) -> tuple[float, ...]:
  if not isinstance(value, list) or len(value) != width:
    raise ValueError(f"{label} shape differs")
  result: list[float] = []
  for item in value:
    if type(item) not in {int, float}:
      raise ValueError(f"{label} contains a non-numeric value")
    normalized = float(item)
    if not math.isfinite(normalized):
      raise ValueError(f"{label} contains a non-finite value")
    result.append(normalized)
  return tuple(result)


def rotation_matrix_to_vector_v1(
    residual_rotation_row_major: Sequence[float],
) -> tuple[float, float, float]:
  """Validate SO(3) and return a stable principal logarithm.

  The angle lies in ``[0, pi]``.  At exactly pi the axis sign is inherently
  ambiguous, so a deterministic first-nonzero-positive convention is used.
  """

  if (
      not isinstance(residual_rotation_row_major, (list, tuple))
      or isinstance(residual_rotation_row_major, (str, bytes))
  ):
    raise ValueError("residual rotation shape differs")
  values = _finite_numeric_vector(
      list(residual_rotation_row_major),
      width=9,
      label="residual rotation",
  )
  rotation = (
      values[0:3],
      values[3:6],
      values[6:9],
  )
  row_dots = tuple(
      sum(rotation[row][axis] * rotation[column][axis] for axis in range(3))
      for row in range(3)
      for column in range(3)
  )
  expected_dots = (
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  )
  determinant = (
      rotation[0][0]
      * (
          rotation[1][1] * rotation[2][2]
          - rotation[1][2] * rotation[2][1]
      )
      - rotation[0][1]
      * (
          rotation[1][0] * rotation[2][2]
          - rotation[1][2] * rotation[2][0]
      )
      + rotation[0][2]
      * (
          rotation[1][0] * rotation[2][1]
          - rotation[1][1] * rotation[2][0]
      )
  )
  if (
      any(
          not math.isclose(
              observed, expected, rel_tol=0.0, abs_tol=1e-6
          )
          for observed, expected in zip(
              row_dots, expected_dots, strict=True
          )
      )
      or not math.isclose(
          determinant, 1.0, rel_tol=0.0, abs_tol=1e-6
      )
  ):
    raise ValueError("residual rotation is not in SO(3)")
  trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
  cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
  vee = (
      rotation[2][1] - rotation[1][2],
      rotation[0][2] - rotation[2][0],
      rotation[1][0] - rotation[0][1],
  )
  sine = 0.5 * math.sqrt(sum(value * value for value in vee))
  angle = math.atan2(sine, cosine)
  if angle < 1e-7:
    vector = tuple(0.5 * value for value in vee)
  elif math.pi - angle < 5e-5:
    diagonal = tuple(
        max(0.0, 0.5 * (rotation[index][index] + 1.0))
        for index in range(3)
    )
    major = max(range(3), key=lambda index: diagonal[index])
    axis_values = [0.0, 0.0, 0.0]
    axis_values[major] = math.sqrt(diagonal[major])
    if axis_values[major] < 1e-10:
      raise ValueError("near-pi residual rotation has no stable axis")
    other = [index for index in range(3) if index != major]
    for index in other:
      axis_values[index] = (
          rotation[major][index] + rotation[index][major]
      ) / (4.0 * axis_values[major])
    norm = math.sqrt(sum(value * value for value in axis_values))
    axis = tuple(value / norm for value in axis_values)
    vee_norm = math.sqrt(sum(value * value for value in vee))
    if vee_norm > 1e-10:
      if sum(a * b for a, b in zip(axis, vee, strict=True)) < 0.0:
        axis = tuple(-value for value in axis)
    else:
      for component in axis:
        if abs(component) > 1e-12:
          if component < 0.0:
            axis = tuple(-value for value in axis)
          break
    vector = tuple(angle * value for value in axis)
  else:
    scale = angle / (2.0 * math.sin(angle))
    vector = tuple(scale * value for value in vee)
  result = tuple(float(value) for value in vector)
  if any(not math.isfinite(value) for value in result):
    raise ValueError("residual rotation logarithm is non-finite")
  return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ModelFeatureRefV1:
  """The only adapter sample field allowed to select model-visible graph data."""

  graph_work_id: str

  def __post_init__(self) -> None:
    _sha256(self.graph_work_id, label="graph work ID")


@dataclass(frozen=True, slots=True)
class ProgramResidualAlternativeV1:
  semantic_program_id: str
  residual_twist_pair_local: tuple[float, ...]

  def __post_init__(self) -> None:
    if type(self.semantic_program_id) is not str or not self.semantic_program_id:
      raise ValueError("semantic program ID is empty")
    if (
        not isinstance(self.residual_twist_pair_local, tuple)
        or len(self.residual_twist_pair_local) != 6
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in self.residual_twist_pair_local
        )
    ):
      raise ValueError("program-local residual twist differs")


@dataclass(frozen=True, slots=True)
class CatalogTargetAlternativesV1:
  catalog_index: int
  catalog_program_id: str
  alternatives: tuple[ProgramResidualAlternativeV1, ...]

  def __post_init__(self) -> None:
    _exact_int(self.catalog_index, label="catalog target index")
    if (
        type(self.catalog_program_id) is not str
        or not self.catalog_program_id
    ):
      raise ValueError("catalog program ID is empty")
    alternatives = tuple(self.alternatives)
    if not alternatives or any(
        type(value) is not ProgramResidualAlternativeV1
        for value in alternatives
    ):
      raise ValueError("catalog alternatives are empty or malformed")
    identities = [
        (value.semantic_program_id, value.residual_twist_pair_local)
        for value in alternatives
    ]
    if len(identities) != len(set(identities)):
      raise ValueError("catalog alternatives contain an exact duplicate")
    object.__setattr__(
        self,
        "alternatives",
        tuple(
            sorted(
                alternatives,
                key=lambda value: (
                    value.semantic_program_id,
                    value.residual_twist_pair_local,
                ),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class OOVProgramAlternativeV1:
  semantic_program_id: str
  residual_twist_pair_local: tuple[float, ...]

  def __post_init__(self) -> None:
    ProgramResidualAlternativeV1(
        semantic_program_id=self.semantic_program_id,
        residual_twist_pair_local=self.residual_twist_pair_local,
    )


@dataclass(frozen=True, slots=True)
class SetValuedProgramSampleV1:
  """Supervision and statistical cluster identity for one directed query."""

  opaque_query_identity: str
  direction: str
  graph_work_id: str
  development_split: str
  family_cluster_id: str
  assembly_cluster_id: str
  known_catalog_targets: tuple[CatalogTargetAlternativesV1, ...]
  oov_targets: tuple[OOVProgramAlternativeV1, ...]

  def __post_init__(self) -> None:
    _sha256(self.opaque_query_identity, label="opaque query identity")
    _sha256(self.graph_work_id, label="sample graph work ID")
    _sha256(self.family_cluster_id, label="family cluster ID")
    _sha256(self.assembly_cluster_id, label="assembly cluster ID")
    if self.direction not in _DIRECTIONS:
      raise ValueError("query direction differs")
    if self.development_split not in _SPLITS:
      raise ValueError("query development split differs")
    groups = tuple(self.known_catalog_targets)
    oov = tuple(self.oov_targets)
    if any(type(value) is not CatalogTargetAlternativesV1 for value in groups):
      raise TypeError("known target group type differs")
    if any(type(value) is not OOVProgramAlternativeV1 for value in oov):
      raise TypeError("OOV target type differs")
    indices = [value.catalog_index for value in groups]
    catalog_ids = [value.catalog_program_id for value in groups]
    if (
        len(indices) != len(set(indices))
        or len(catalog_ids) != len(set(catalog_ids))
    ):
      raise ValueError("known catalog groups are not unique")
    program_to_group: dict[str, int] = {}
    for group in groups:
      for alternative in group.alternatives:
        prior = program_to_group.setdefault(
            alternative.semantic_program_id, group.catalog_index
        )
        if prior != group.catalog_index:
          raise ValueError("one semantic program maps to two catalog groups")
    if set(program_to_group).intersection(
        value.semantic_program_id for value in oov
    ):
      raise ValueError("one semantic program is both known and OOV")
    if not groups and not oov:
      raise ValueError("set-valued sample has no targets")
    object.__setattr__(
        self,
        "known_catalog_targets",
        tuple(sorted(groups, key=lambda value: value.catalog_index)),
    )
    object.__setattr__(
        self,
        "oov_targets",
        tuple(
            sorted(
                oov,
                key=lambda value: (
                    value.semantic_program_id,
                    value.residual_twist_pair_local,
                ),
            )
        ),
    )

  @property
  def known_catalog_indices(self) -> tuple[int, ...]:
    return tuple(value.catalog_index for value in self.known_catalog_targets)

  @property
  def classification_evaluable(self) -> bool:
    return bool(self.known_catalog_targets)

  @property
  def cluster_key(self) -> tuple[str, str]:
    return self.family_cluster_id, self.assembly_cluster_id

  @property
  def sample_key(self) -> tuple[str, str]:
    return self.opaque_query_identity, self.direction


@dataclass(frozen=True, slots=True)
class SetValuedProgramTargets:
  samples: tuple[SetValuedProgramSampleV1, ...]
  catalog_size: int
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    samples = tuple(self.samples)
    if not samples or any(
        type(value) is not SetValuedProgramSampleV1 for value in samples
    ):
      raise ValueError("set-valued target batch is empty or malformed")
    _exact_int(self.catalog_size, label="catalog size", minimum=1)
    if self.schema_version != SCHEMA_VERSION:
      raise ValueError("set-valued schema differs")
    identities = [
        (value.opaque_query_identity, value.direction) for value in samples
    ]
    if len(identities) != len(set(identities)):
      raise ValueError("query-direction samples are not unique")
    if any(
        index >= self.catalog_size
        for sample in samples
        for index in sample.known_catalog_indices
    ):
      raise ValueError("known target lies outside the finite catalog")
    object.__setattr__(self, "samples", samples)

  @property
  def classification_evaluable_mask(self) -> tuple[bool, ...]:
    return tuple(value.classification_evaluable for value in self.samples)


@dataclass(frozen=True, slots=True)
class ParsedSetValuedAdapterSampleV1:
  model_feature_ref: ModelFeatureRefV1
  supervision: SetValuedProgramSampleV1


def _validate_descriptor(value: Any) -> None:
  if not isinstance(value, Mapping) or set(value) != _DESCRIPTOR_FIELDS:
    raise ValueError("target descriptor schema differs")
  if any(type(item) is not str or not item for item in value.values()):
    raise ValueError("target descriptor value differs")


def _parse_benchmark_v2_mapping_sample_v1(
    sample_payload: Mapping[str, Any],
) -> ParsedSetValuedAdapterSampleV1:
  """Private strict-JSON row parser retained for diagnostic equivalence."""

  if not isinstance(sample_payload, Mapping) or set(sample_payload) != _SAMPLE_FIELDS:
    raise ValueError("benchmark-v2 sample schema differs")
  if sample_payload["schema_version"] != BENCHMARK_V2_SAMPLE_SCHEMA:
    raise ValueError("benchmark-v2 sample version differs")
  for field in (
      "sample_identity_sha256",
      "opaque_case_identity",
      "opaque_query_identity",
      "graph_work_id",
      "family_cluster_id",
      "assembly_cluster_id",
      "sample_payload_sha256",
  ):
    _sha256(sample_payload[field], label=field)
  _exact_int(
      sample_payload["source_contact_ordinal"],
      label="source contact ordinal",
  )
  if sample_payload["development_split"] not in _SPLITS:
    raise ValueError("development split differs")
  if sample_payload["direction"] not in _DIRECTIONS:
    raise ValueError("sample direction differs")
  target_count = _exact_int(
      sample_payload["target_count"], label="target count", minimum=1
  )
  known_count = _exact_int(
      sample_payload["known_target_count"], label="known target count"
  )
  oov_count = _exact_int(
      sample_payload["oov_target_count"], label="OOV target count"
  )
  if type(sample_payload["classification_evaluable"]) is not bool:
    raise ValueError("classification evaluable flag is not boolean")
  raw_targets = sample_payload["targets"]
  if not isinstance(raw_targets, list) or len(raw_targets) != target_count:
    raise ValueError("target array/count differs")

  grouped: dict[int, tuple[str, list[ProgramResidualAlternativeV1]]] = {}
  oov_targets: list[OOVProgramAlternativeV1] = []
  semantic_program_ids: set[str] = set()
  for target_index, raw in enumerate(raw_targets):
    if not isinstance(raw, Mapping) or set(raw) != _TARGET_FIELDS:
      raise ValueError("benchmark-v2 target schema differs")
    program_id = raw["program_id"]
    if type(program_id) is not str or not program_id:
      raise ValueError("semantic program ID differs")
    if program_id in semantic_program_ids:
      raise ValueError("semantic program ID is duplicated")
    semantic_program_ids.add(program_id)
    for field in (
        "source_program_row_sha256",
        "descriptor_sha256",
        "target_payload_sha256",
    ):
      _sha256(raw[field], label=f"target[{target_index}].{field}")
    _validate_descriptor(raw["descriptor"])
    if _canonical_sha256(raw["descriptor"]) != raw["descriptor_sha256"]:
      raise ValueError("target descriptor hash differs")
    if type(raw["oov"]) is not bool:
      raise ValueError("target OOV flag is not boolean")
    translation = _finite_numeric_vector(
        raw["residual_translation"],
        width=3,
        label="residual translation",
    )
    rotation_vector = rotation_matrix_to_vector_v1(
        raw["residual_rotation_row_major"]
    )
    twist = tuple(float(value) for value in (*translation, *rotation_vector))
    if raw["oov"]:
      if raw["catalog_index"] is not None or raw["catalog_program_id"] is not None:
        raise ValueError("OOV target retained a catalog identity")
      oov_targets.append(
          OOVProgramAlternativeV1(
              semantic_program_id=program_id,
              residual_twist_pair_local=twist,
          )
      )
    else:
      catalog_index = _exact_int(
          raw["catalog_index"], label="target catalog index"
      )
      catalog_program_id = raw["catalog_program_id"]
      if type(catalog_program_id) is not str or not catalog_program_id:
        raise ValueError("known target catalog program ID differs")
      prior = grouped.get(catalog_index)
      if prior is None:
        grouped[catalog_index] = (catalog_program_id, [])
      elif prior[0] != catalog_program_id:
        raise ValueError("one catalog index has multiple catalog program IDs")
      grouped[catalog_index][1].append(
          ProgramResidualAlternativeV1(
              semantic_program_id=program_id,
              residual_twist_pair_local=twist,
          )
      )
    unsigned_target = dict(raw)
    observed_target_hash = unsigned_target.pop("target_payload_sha256")
    if _canonical_sha256(unsigned_target) != observed_target_hash:
      raise ValueError("target payload hash differs")

  groups = tuple(
      CatalogTargetAlternativesV1(
          catalog_index=index,
          catalog_program_id=catalog_program_id,
          alternatives=tuple(alternatives),
      )
      for index, (catalog_program_id, alternatives) in sorted(grouped.items())
  )
  if (
      len(groups) != len(grouped)
      or sum(len(value.alternatives) for value in groups) != known_count
      or len(oov_targets) != oov_count
      or target_count != known_count + oov_count
      or sample_payload["classification_evaluable"] != bool(groups)
  ):
    raise ValueError("sample known/OOV coverage differs")
  unsigned_sample = dict(sample_payload)
  observed_sample_hash = unsigned_sample.pop("sample_payload_sha256")
  if _canonical_sha256(unsigned_sample) != observed_sample_hash:
    raise ValueError("sample payload hash differs")

  return ParsedSetValuedAdapterSampleV1(
      model_feature_ref=ModelFeatureRefV1(
          graph_work_id=sample_payload["graph_work_id"]
      ),
      supervision=SetValuedProgramSampleV1(
          opaque_query_identity=sample_payload["opaque_query_identity"],
          direction=sample_payload["direction"],
          graph_work_id=sample_payload["graph_work_id"],
          development_split=sample_payload["development_split"],
          family_cluster_id=sample_payload["family_cluster_id"],
          assembly_cluster_id=sample_payload["assembly_cluster_id"],
          known_catalog_targets=groups,
          oov_targets=tuple(oov_targets),
      ),
  )


def _typed_residual_twist_v1(
    target: SupervisionTargetV2,
) -> tuple[float, ...]:
  from .benchmark_v2_query_semantic_adapter_verifier_v2 import (
      SupervisionTargetV2,
  )

  if type(target) is not SupervisionTargetV2:
    raise TypeError("supervision target type differs")
  if (
      not isinstance(target.residual_translation, tuple)
      or len(target.residual_translation) != 3
      or any(type(value) is not float for value in target.residual_translation)
      or not isinstance(target.residual_rotation_row_major, tuple)
      or len(target.residual_rotation_row_major) != 9
      or any(
          type(value) is not float
          for value in target.residual_rotation_row_major
      )
  ):
    raise ValueError("typed supervision residual differs")
  rotation = rotation_matrix_to_vector_v1(
      list(target.residual_rotation_row_major)
  )
  return tuple(float(value) for value in (
      *target.residual_translation,
      *rotation,
  ))


def _parse_typed_supervision_sample_v1(
    sample: SupervisionSampleV2,
    *,
    catalog: FixedCatalogMetadataV2,
) -> SetValuedProgramSampleV1:
  """Private exact-capability parser; never accepts a public Mapping."""

  from .benchmark_v2_query_semantic_adapter_verifier_v2 import (
      CatalogEntryMetadataV2,
      FixedCatalogMetadataV2,
      ProgramDescriptorMetadataV2,
      SupervisionSampleV2,
      SupervisionTargetV2,
  )

  if type(sample) is not SupervisionSampleV2:
    raise TypeError("dataset supervision sample type differs")
  if type(catalog) is not FixedCatalogMetadataV2:
    raise TypeError("dataset fixed catalog metadata type differs")
  if (
      sample.schema_version != BENCHMARK_V2_SAMPLE_SCHEMA
      or type(sample.source_contact_ordinal) is not int
      or type(sample.classification_evaluable) is not bool
      or type(sample.target_count) is not int
      or type(sample.known_target_count) is not int
      or type(sample.oov_target_count) is not int
      or sample.target_count != len(sample.targets)
      or sample.target_count
      != sample.known_target_count + sample.oov_target_count
  ):
    raise ValueError("typed supervision sample contract differs")
  for field in (
      "sample_identity_sha256",
      "opaque_case_identity",
      "opaque_query_identity",
      "graph_work_id",
      "family_cluster_id",
      "assembly_cluster_id",
      "sample_payload_sha256",
  ):
    _sha256(getattr(sample, field), label=f"typed sample {field}")
  if (
      sample.direction not in _DIRECTIONS
      or sample.development_split not in _SPLITS
  ):
    raise ValueError("typed supervision sample split/direction differs")
  entries = {
      entry.program_index: entry for entry in catalog.entries
  }
  grouped: dict[int, tuple[str, list[ProgramResidualAlternativeV1]]] = {}
  oov_targets: list[OOVProgramAlternativeV1] = []
  seen_programs: set[str] = set()
  known_count = 0
  for target in sample.targets:
    if type(target) is not SupervisionTargetV2:
      raise TypeError("typed supervision target type differs")
    if type(target.descriptor) is not ProgramDescriptorMetadataV2:
      raise TypeError("typed supervision descriptor type differs")
    if (
        type(target.program_id) is not str
        or not target.program_id
        or target.program_id in seen_programs
        or type(target.oov) is not bool
    ):
      raise ValueError("typed semantic program identity differs")
    seen_programs.add(target.program_id)
    for field in (
        "source_program_row_sha256",
        "descriptor_sha256",
        "target_payload_sha256",
    ):
      _sha256(getattr(target, field), label=f"typed target {field}")
    twist = _typed_residual_twist_v1(target)
    if target.oov:
      if target.catalog_index is not None or target.catalog_program_id is not None:
        raise ValueError("typed OOV target retained catalog identity")
      oov_targets.append(
          OOVProgramAlternativeV1(
              semantic_program_id=target.program_id,
              residual_twist_pair_local=twist,
          )
      )
      continue
    if type(target.catalog_index) is not int:
      raise ValueError("typed target catalog index is not an exact integer")
    entry = entries.get(target.catalog_index)
    if (
        type(entry) is not CatalogEntryMetadataV2
        or target.catalog_program_id != entry.program_id
        or target.descriptor_sha256 != entry.descriptor_sha256
    ):
      raise ValueError("typed target/fixed catalog join differs")
    known_count += 1
    prior = grouped.get(target.catalog_index)
    if prior is None:
      grouped[target.catalog_index] = (entry.program_id, [])
    grouped[target.catalog_index][1].append(
        ProgramResidualAlternativeV1(
            semantic_program_id=target.program_id,
            residual_twist_pair_local=twist,
        )
    )
  groups = tuple(
      CatalogTargetAlternativesV1(
          catalog_index=index,
          catalog_program_id=program_id,
          alternatives=tuple(alternatives),
      )
      for index, (program_id, alternatives) in sorted(grouped.items())
  )
  if (
      known_count != sample.known_target_count
      or len(oov_targets) != sample.oov_target_count
      or sample.classification_evaluable != bool(groups)
  ):
    raise ValueError("typed sample known/OOV coverage differs")
  return SetValuedProgramSampleV1(
      opaque_query_identity=sample.opaque_query_identity,
      direction=sample.direction,
      graph_work_id=sample.graph_work_id,
      development_split=sample.development_split,
      family_cluster_id=sample.family_cluster_id,
      assembly_cluster_id=sample.assembly_cluster_id,
      known_catalog_targets=groups,
      oov_targets=tuple(oov_targets),
  )


def _sample_commitment_v1(sample: SetValuedProgramSampleV1) -> str:
  """Commit every field that can affect model work, supervision, or weighting."""

  if type(sample) is not SetValuedProgramSampleV1:
    raise TypeError("sample commitment requires exact set-valued sample")
  return _canonical_sha256({
      "schema_version": "v19_authenticated_set_valued_sample_commitment.v1",
      "opaque_query_identity": sample.opaque_query_identity,
      "direction": sample.direction,
      "graph_work_id": sample.graph_work_id,
      "development_split": sample.development_split,
      "family_cluster_id": sample.family_cluster_id,
      "assembly_cluster_id": sample.assembly_cluster_id,
      "classification_evaluable": sample.classification_evaluable,
      "known_catalog_targets": [
          {
              "catalog_index": group.catalog_index,
              "catalog_program_id": group.catalog_program_id,
              "alternatives": [
                  {
                      "semantic_program_id": alternative.semantic_program_id,
                      "residual_twist_pair_local": list(
                          alternative.residual_twist_pair_local
                      ),
                  }
                  for alternative in group.alternatives
              ],
          }
          for group in sample.known_catalog_targets
      ],
      "oov_targets": [
          {
              "semantic_program_id": alternative.semantic_program_id,
              "residual_twist_pair_local": list(
                  alternative.residual_twist_pair_local
              ),
          }
          for alternative in sample.oov_targets
      ],
  })


class FrozenGlobalClusterWeightsV1:
  """Dataset-global cluster counts and per-sample lookup capability."""

  __slots__ = (
      "_cluster_counts",
      "_weights_by_sample",
      "_commitments_by_sample",
      "_complete_sample_keys",
      "_catalog_size",
      "_sealed",
  )

  def __init__(
      self,
      *,
      cluster_counts: tuple[tuple[str, str, str, int], ...],
      weights_by_sample: tuple[tuple[str, str, float], ...],
      commitments_by_sample: tuple[tuple[str, str, str], ...],
      complete_sample_keys: frozenset[tuple[str, str]],
      catalog_size: int,
      _token: object,
  ) -> None:
    if _token is not _GLOBAL_WEIGHT_TOKEN:
      raise TypeError("global cluster weights are dataset-builder only")
    object.__setattr__(self, "_cluster_counts", cluster_counts)
    object.__setattr__(self, "_weights_by_sample", weights_by_sample)
    object.__setattr__(
        self, "_commitments_by_sample", commitments_by_sample
    )
    object.__setattr__(self, "_complete_sample_keys", complete_sample_keys)
    object.__setattr__(self, "_catalog_size", catalog_size)
    object.__setattr__(self, "_sealed", True)

  def __setattr__(self, _name: str, _value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("global cluster weights are immutable")
    object.__setattr__(self, _name, _value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("global cluster weights are not serializable")

  @property
  def cluster_counts(self) -> tuple[tuple[str, str, str, int], ...]:
    return self._cluster_counts

  def weights_for(
      self, targets: SetValuedProgramTargets
  ) -> tuple[float, ...]:
    if targets.catalog_size != self._catalog_size:
      raise ValueError(
          "mini-batch catalog size differs from authenticated dataset"
      )
    lookup = {
        (query, direction): weight
        for query, direction, weight in self._weights_by_sample
    }
    commitments = {
        (query, direction): commitment
        for query, direction, commitment in self._commitments_by_sample
    }
    result: list[float] = []
    for sample in targets.samples:
      if sample.sample_key not in self._complete_sample_keys:
        raise ValueError("mini-batch sample is outside frozen dataset weights")
      if commitments[sample.sample_key] != _sample_commitment_v1(sample):
        raise ValueError(
            "mini-batch sample differs from authenticated dataset commitment"
        )
      result.append(lookup[sample.sample_key])
    return tuple(result)


def _freeze_global_cluster_weights_v1(
    complete_targets: SetValuedProgramTargets,
) -> FrozenGlobalClusterWeightsV1:
  """Freeze split-local cluster totals over one complete dataset capability."""

  counts: dict[tuple[str, str, str], int] = {}
  for sample in complete_targets.samples:
    if sample.classification_evaluable:
      key = (sample.development_split, *sample.cluster_key)
      counts[key] = counts.get(key, 0) + 1
  weights = tuple(
      (
          sample.opaque_query_identity,
          sample.direction,
          (
              1.0
              / counts[(sample.development_split, *sample.cluster_key)]
              if sample.classification_evaluable
              else 0.0
          ),
      )
      for sample in complete_targets.samples
  )
  commitments = tuple(
      (
          sample.opaque_query_identity,
          sample.direction,
          _sample_commitment_v1(sample),
      )
      for sample in complete_targets.samples
  )
  return FrozenGlobalClusterWeightsV1(
      cluster_counts=tuple(
          (*key, count) for key, count in sorted(counts.items())
      ),
      weights_by_sample=weights,
      commitments_by_sample=commitments,
      complete_sample_keys=frozenset(
          sample.sample_key for sample in complete_targets.samples
      ),
      catalog_size=complete_targets.catalog_size,
      _token=_GLOBAL_WEIGHT_TOKEN,
  )


class AuthenticatedCatalogClassWeightsV1:
  """Optional train-only class weights rooted in verified V2 catalog stats."""

  __slots__ = (
      "_weights",
      "_catalog_payload_sha256",
      "_source_commitment_sha256",
      "_sealed",
  )

  def __init__(
      self,
      *,
      weights: tuple[float, ...],
      catalog_payload_sha256: str,
      source_commitment_sha256: str,
      _token: object,
  ) -> None:
    if _token is not _CLASS_WEIGHT_TOKEN:
      raise TypeError("catalog class weights are authenticated-dataset only")
    if (
        len(weights) != 19
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
    ):
      raise ValueError("authenticated class weights differ")
    _sha256(catalog_payload_sha256, label="catalog payload hash")
    _sha256(
        source_commitment_sha256,
        label="train class-weight source commitment",
    )
    object.__setattr__(self, "_weights", weights)
    object.__setattr__(
        self, "_catalog_payload_sha256", catalog_payload_sha256
    )
    object.__setattr__(
        self, "_source_commitment_sha256", source_commitment_sha256
    )
    object.__setattr__(self, "_sealed", True)

  def __setattr__(self, _name: str, _value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("catalog class weights are immutable")
    object.__setattr__(self, _name, _value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("catalog class weights are not serializable")

  @property
  def values(self) -> tuple[float, ...]:
    return self._weights

  @property
  def source_commitment_sha256(self) -> str:
    return self._source_commitment_sha256


@dataclass(frozen=True, slots=True)
class BundleSetValuedTrainingStateV1:
  """Torch-side state reconstructed only after bundle byte verification."""

  complete_targets: SetValuedProgramTargets
  global_cluster_weights: FrozenGlobalClusterWeightsV1
  catalog_class_weights: AuthenticatedCatalogClassWeightsV1

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("bundle training state is not serializable")


def build_bundle_set_valued_training_state_v1(
    *,
    samples: Sequence[SetValuedProgramSampleV1],
    class_weights: Sequence[float],
    catalog_payload_sha256: str,
    source_commitment_sha256: str,
) -> BundleSetValuedTrainingStateV1:
  """Restore the fixed19 loss state from a byte-verified training bundle."""

  targets = SetValuedProgramTargets(
      samples=tuple(samples),
      catalog_size=19,
  )
  weights = tuple(float(value) for value in class_weights)
  if len(weights) != 19:
    raise ValueError("bundle fixed-catalog class weights differ")
  class_weight_state = AuthenticatedCatalogClassWeightsV1(
      weights=weights,
      catalog_payload_sha256=catalog_payload_sha256,
      source_commitment_sha256=source_commitment_sha256,
      _token=_CLASS_WEIGHT_TOKEN,
  )
  return BundleSetValuedTrainingStateV1(
      complete_targets=targets,
      global_cluster_weights=_freeze_global_cluster_weights_v1(targets),
      catalog_class_weights=class_weight_state,
  )


class AuthenticatedSetValuedDatasetV1:
  """Non-serializable bridge from the exact V2 verifier to learning."""

  __slots__ = (
      "_geometry_requests",
      "_model_feature_refs",
      "_targets",
      "_global_cluster_weights",
      "_catalog_class_weights",
      "_fixed_catalog_domain",
      "_lineage",
      "_catalog",
      "_sealed",
  )

  def __init__(
      self,
      *,
      geometry_requests: tuple[GeometryMaterializationRequestV2, ...],
      model_feature_refs: tuple[ModelFeatureRefV1, ...],
      targets: SetValuedProgramTargets,
      global_cluster_weights: FrozenGlobalClusterWeightsV1,
      catalog_class_weights: AuthenticatedCatalogClassWeightsV1,
      fixed_catalog_domain: frozenset[int],
      lineage: FixedLineageMetadataV2,
      catalog: FixedCatalogMetadataV2,
      _token: object,
  ) -> None:
    if _token is not _DATASET_TOKEN:
      raise TypeError("authenticated set-valued dataset is builder-only")
    object.__setattr__(self, "_geometry_requests", geometry_requests)
    object.__setattr__(self, "_model_feature_refs", model_feature_refs)
    object.__setattr__(self, "_targets", targets)
    object.__setattr__(
        self, "_global_cluster_weights", global_cluster_weights
    )
    object.__setattr__(
        self, "_catalog_class_weights", catalog_class_weights
    )
    object.__setattr__(self, "_fixed_catalog_domain", fixed_catalog_domain)
    object.__setattr__(self, "_lineage", lineage)
    object.__setattr__(self, "_catalog", catalog)
    object.__setattr__(self, "_sealed", True)

  def __setattr__(self, _name: str, _value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("authenticated set-valued dataset is immutable")
    object.__setattr__(self, _name, _value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("authenticated set-valued dataset is not serializable")

  @property
  def model_feature_refs(self) -> tuple[ModelFeatureRefV1, ...]:
    return self._model_feature_refs

  @property
  def supervision_targets(self) -> SetValuedProgramTargets:
    return self._targets

  @property
  def global_cluster_weights(self) -> FrozenGlobalClusterWeightsV1:
    return self._global_cluster_weights

  @property
  def catalog_class_weights(self) -> AuthenticatedCatalogClassWeightsV1:
    return self._catalog_class_weights

  @property
  def fixed_catalog_domain(self) -> frozenset[int]:
    return self._fixed_catalog_domain

  @property
  def lineage_metadata(self) -> FixedLineageMetadataV2:
    return self._lineage

  @property
  def catalog_metadata(self) -> FixedCatalogMetadataV2:
    return self._catalog

  def geometry_requests(
      self,
  ) -> tuple[GeometryMaterializationRequestV2, ...]:
    """Return only verifier-issued geometry requests, never supervision."""

    return self._geometry_requests


def build_authenticated_set_valued_dataset_v1(
    verified_index: StructurallyAndExactRebuildCheckedQueryGraphWorkIndexV2,
) -> AuthenticatedSetValuedDatasetV1:
  """Build the only public V19 adapter-to-learning dataset capability."""

  from .benchmark_v2_query_semantic_adapter_verifier_v2 import (
      CatalogEntryMetadataV2,
      FixedCatalogMetadataV2,
      FixedLineageMetadataV2,
      GeometryMaterializationRequestV2,
      ProgramDescriptorMetadataV2,
      StructurallyAndExactRebuildCheckedQueryGraphWorkIndexV2,
      SupervisionSampleV2,
      TrainCatalogStatisticsV2,
  )

  if type(verified_index) is not (
      StructurallyAndExactRebuildCheckedQueryGraphWorkIndexV2
  ):
    raise TypeError(
        "set-valued dataset requires the exact V2 verifier capability"
    )
  verified_index.revalidate()
  geometry = verified_index.geometry_requests()
  supervision = verified_index.supervision_samples()
  lineage = verified_index.fixed_lineage_metadata()
  catalog = verified_index.catalog_metadata()
  if (
      type(lineage) is not FixedLineageMetadataV2
      or type(catalog) is not FixedCatalogMetadataV2
      or type(catalog.training_statistics) is not TrainCatalogStatisticsV2
      or lineage.schema_version
      != "benchmark_v2_authenticated_query_graph_work_index.v2"
      or catalog.schema_version
      != "benchmark_v2_pre_v19_fixed_program_catalog.v2"
      or catalog.candidate_policy
      != "pre_v19_fixed19_catalog_with_explicit_oov_failure.v1"
      or catalog.candidate_order
      != "pre_v19_roster_program_index.v1"
      or catalog.target_policy
      != "all_targets_retained_catalog_index_or_null_oov.v1"
      or catalog.entry_count != 19
      or len(catalog.entries) != 19
      or tuple(entry.program_index for entry in catalog.entries)
      != tuple(range(19))
  ):
    raise ValueError("verified V2 fixed lineage/catalog contract differs")
  for field in (
      "index_payload_sha256",
      "authenticated_receipt_sha256",
      "receipt_payload_sha256",
      "domain_sha256",
      "fixed_v19_manifest_sha256",
      "fixed_family_source_sha256",
  ):
    _sha256(getattr(lineage, field), label=f"fixed lineage {field}")
  if (
      type(lineage.source_revision) is not str
      or len(lineage.source_revision) != 40
      or set(lineage.source_revision) - _SHA256_CHARS
  ):
    raise ValueError("fixed lineage source revision differs")
  if any(
      type(entry) is not CatalogEntryMetadataV2
      or type(entry.descriptor) is not ProgramDescriptorMetadataV2
      for entry in catalog.entries
  ):
    raise TypeError("verified fixed catalog entry type differs")
  _sha256(
      catalog.catalog_payload_sha256, label="fixed catalog payload hash"
  )
  _sha256(
      catalog.catalog_roster_sha256, label="fixed catalog roster hash"
  )
  statistics = catalog.training_statistics
  if (
      statistics.source_split != "train"
      or len(statistics.target_frequency_by_program_index) != 19
      or len(statistics.class_weight_by_program_index) != 19
      or type(statistics.source_row_count) is not int
      or statistics.source_row_count
      != statistics.known_target_count + statistics.oov_target_count
  ):
    raise ValueError("verified V2 train catalog statistics differ")
  _sha256(
      statistics.source_commitment_sha256,
      label="train catalog statistics commitment",
  )

  geometry_by_id: dict[str, GeometryMaterializationRequestV2] = {}
  for request in geometry:
    if type(request) is not GeometryMaterializationRequestV2:
      raise TypeError("verified geometry request type differs")
    _sha256(request.graph_work_id, label="geometry graph work ID")
    if (
        request.graph_work_id in geometry_by_id
        or request.model_view_schema_version != "benchmark_v2_model_view.v2"
        or request.graph_schema_version
        != "benchmark_v2_intrinsic_brep_graph.v2"
    ):
      raise ValueError("verified geometry request contract differs")
    geometry_by_id[request.graph_work_id] = request
  parsed_samples: list[SetValuedProgramSampleV1] = []
  feature_refs: list[ModelFeatureRefV1] = []
  for sample in supervision:
    if type(sample) is not SupervisionSampleV2:
      raise TypeError("verified supervision sample type differs")
    if sample.graph_work_id not in geometry_by_id:
      raise ValueError("supervision sample lacks geometry request")
    parsed_samples.append(
        _parse_typed_supervision_sample_v1(sample, catalog=catalog)
    )
    feature_refs.append(ModelFeatureRefV1(sample.graph_work_id))
  if (
      not parsed_samples
      or len(parsed_samples) != len(geometry_by_id)
      or len({value.graph_work_id for value in feature_refs})
      != len(feature_refs)
  ):
    raise ValueError("geometry/supervision typed join is not one-to-one")
  targets = SetValuedProgramTargets(
      samples=tuple(parsed_samples), catalog_size=19
  )
  global_weights = _freeze_global_cluster_weights_v1(targets)
  class_weights = AuthenticatedCatalogClassWeightsV1(
      weights=tuple(
          float(value)
          for value in statistics.class_weight_by_program_index
      ),
      catalog_payload_sha256=catalog.catalog_payload_sha256,
      source_commitment_sha256=statistics.source_commitment_sha256,
      _token=_CLASS_WEIGHT_TOKEN,
  )
  verified_index.revalidate()
  return AuthenticatedSetValuedDatasetV1(
      geometry_requests=tuple(
          geometry_by_id[value.graph_work_id] for value in feature_refs
      ),
      model_feature_refs=tuple(feature_refs),
      targets=targets,
      global_cluster_weights=global_weights,
      catalog_class_weights=class_weights,
      fixed_catalog_domain=frozenset(range(19)),
      lineage=lineage,
      catalog=catalog,
      _token=_DATASET_TOKEN,
  )


def _validate_prediction_tensors(
    program_logits: Tensor,
    residual_predictions_pair_local: Tensor,
    targets: SetValuedProgramTargets,
) -> None:
  batch = len(targets.samples)
  if program_logits.shape != (batch, targets.catalog_size):
    raise ValueError("program logits shape differs")
  if residual_predictions_pair_local.shape != (
      batch,
      targets.catalog_size,
      6,
  ):
    raise ValueError("residual prediction shape differs")
  if (
      not program_logits.is_floating_point()
      or not residual_predictions_pair_local.is_floating_point()
      or program_logits.dtype != residual_predictions_pair_local.dtype
      or program_logits.device != residual_predictions_pair_local.device
  ):
    raise TypeError("prediction dtype/device differs")
  if (
      not bool(torch.isfinite(program_logits).all())
      or not bool(torch.isfinite(residual_predictions_pair_local).all())
  ):
    raise ValueError("prediction contains a non-finite value")


def _normalized_residual_softmin(
    prediction: Tensor,
    alternatives: Sequence[ProgramResidualAlternativeV1],
    *,
    temperature: float | None,
) -> Tensor:
  losses = torch.stack([
      torch.mean(
          (
              prediction
              - prediction.new_tensor(value.residual_twist_pair_local)
          )
          ** 2
      )
      for value in alternatives
  ])
  if temperature is None:
    return torch.min(losses)
  return -temperature * (
      torch.logsumexp(-losses / temperature, dim=0)
      - math.log(losses.numel())
  )


def _weighted_unique_sample_mean(
    rows: Sequence[Tensor],
    weights: Sequence[float],
    *,
    zero: Tensor,
) -> Tensor:
  if not rows:
    return zero
  denominator = float(sum(weights))
  if not math.isfinite(denominator) or denominator <= 0.0:
    raise ValueError("effective cluster-equal weight is not positive")
  return torch.stack([
      row * float(weight)
      for row, weight in zip(rows, weights, strict=True)
  ]).sum() / denominator


def pure_set_program_nll_v1(
    program_logits: Tensor,
    residual_predictions_pair_local: Tensor,
    targets: SetValuedProgramTargets,
    *,
    global_cluster_weights: FrozenGlobalClusterWeightsV1,
) -> Tensor:
  """Classification-only set NLL for audit; not the default training API."""

  _validate_prediction_tensors(
      program_logits, residual_predictions_pair_local, targets
  )
  if type(global_cluster_weights) is not FrozenGlobalClusterWeightsV1:
    raise TypeError("pure set NLL requires frozen global cluster weights")
  cluster_weights = global_cluster_weights.weights_for(targets)
  accumulator_logits = (
      program_logits.float()
      if program_logits.dtype in {torch.float16, torch.bfloat16}
      else program_logits
  )
  accumulator_residuals = (
      residual_predictions_pair_local.float()
      if residual_predictions_pair_local.dtype
      in {torch.float16, torch.bfloat16}
      else residual_predictions_pair_local
  )
  rows: list[Tensor] = []
  weights: list[float] = []
  for index, sample in enumerate(targets.samples):
    if not sample.classification_evaluable:
      continue
    gold = torch.tensor(
        sample.known_catalog_indices,
        device=accumulator_logits.device,
        dtype=torch.long,
    )
    logits = accumulator_logits[index]
    rows.append(
        torch.logsumexp(logits, dim=0)
        - torch.logsumexp(logits.index_select(0, gold), dim=0)
    )
    weights.append(cluster_weights[index])
  zero = (
      accumulator_logits.sum() * 0.0
      + accumulator_residuals.sum() * 0.0
  )
  return _weighted_unique_sample_mean(rows, weights, zero=zero)


@dataclass(frozen=True, slots=True)
class JointSetProgramResidualLossV1:
  total: Tensor
  joint_set_program_residual_nll: Tensor
  audit_pure_set_program_nll: Tensor
  diagnostic_best_residual_loss: Tensor
  training_sample_count: int
  excluded_oov_only_count: int
  unique_cluster_count: int
  cluster_equal_weight_sum: float
  effective_weight_sum: float


def joint_set_program_residual_loss_v1(
    program_logits: Tensor,
    residual_predictions_pair_local: Tensor,
    targets: SetValuedProgramTargets,
    *,
    global_cluster_weights: FrozenGlobalClusterWeightsV1,
    catalog_class_weights: AuthenticatedCatalogClassWeightsV1 | None = None,
    residual_lambda: float = 1.0,
    residual_softmin_temperature: float | None = None,
) -> JointSetProgramResidualLossV1:
  """Default V19 objective coupling class score and local refinement.

  For every known catalog class, its alternative residuals are reduced first.
  The gold numerator is then
  ``logsumexp(logit[class] - lambda * residual_loss[class])``.
  """

  if type(targets) is not SetValuedProgramTargets:
    raise TypeError("joint set loss requires SetValuedProgramTargets")
  _validate_prediction_tensors(
      program_logits, residual_predictions_pair_local, targets
  )
  if (
      not math.isfinite(float(residual_lambda))
      or float(residual_lambda) < 0.0
  ):
    raise ValueError("residual lambda must be finite and non-negative")
  if residual_softmin_temperature is not None and (
      not math.isfinite(float(residual_softmin_temperature))
      or float(residual_softmin_temperature) <= 0.0
  ):
    raise ValueError("residual soft-min temperature must be positive")
  temperature = (
      None
      if residual_softmin_temperature is None
      else float(residual_softmin_temperature)
  )
  if type(global_cluster_weights) is not FrozenGlobalClusterWeightsV1:
    raise TypeError("joint set loss requires frozen global cluster weights")
  if catalog_class_weights is not None and (
      type(catalog_class_weights) is not AuthenticatedCatalogClassWeightsV1
  ):
    raise TypeError(
        "class weights must come from authenticated V2 train catalog stats"
    )
  if catalog_class_weights is not None and targets.catalog_size != 19:
    raise ValueError("authenticated class weights require the fixed19 catalog")
  cluster_weights = global_cluster_weights.weights_for(targets)
  accumulator_logits = (
      program_logits.float()
      if program_logits.dtype in {torch.float16, torch.bfloat16}
      else program_logits
  )
  accumulator_residuals = (
      residual_predictions_pair_local.float()
      if residual_predictions_pair_local.dtype
      in {torch.float16, torch.bfloat16}
      else residual_predictions_pair_local
  )
  joint_rows: list[Tensor] = []
  pure_rows: list[Tensor] = []
  residual_rows: list[Tensor] = []
  weights: list[float] = []
  pure_weights: list[float] = []
  for sample_index, sample in enumerate(targets.samples):
    if not sample.classification_evaluable:
      continue
    logits = accumulator_logits[sample_index]
    group_scores: list[Tensor] = []
    group_residuals: list[Tensor] = []
    for group in sample.known_catalog_targets:
      residual_loss = _normalized_residual_softmin(
          accumulator_residuals[
              sample_index, group.catalog_index
          ],
          group.alternatives,
          temperature=temperature,
      )
      group_residuals.append(residual_loss)
      group_scores.append(
          logits[group.catalog_index]
          - float(residual_lambda) * residual_loss
      )
    joint_rows.append(
        torch.logsumexp(logits, dim=0)
        - torch.logsumexp(torch.stack(group_scores), dim=0)
    )
    gold_indices = torch.tensor(
        sample.known_catalog_indices,
        dtype=torch.long,
        device=accumulator_logits.device,
    )
    pure_rows.append(
        torch.logsumexp(logits, dim=0)
        - torch.logsumexp(logits.index_select(0, gold_indices), dim=0)
    )
    residual_rows.append(torch.min(torch.stack(group_residuals)))
    sample_weight = cluster_weights[sample_index]
    if catalog_class_weights is not None:
      if sample.development_split != "train":
        raise ValueError("train catalog class weights cannot weight dev loss")
      class_values = tuple(
          catalog_class_weights.values[index]
          for index in sample.known_catalog_indices
      )
      class_factor = sum(class_values) / len(class_values)
      if not math.isfinite(class_factor) or class_factor <= 0.0:
        raise ValueError(
            "train sample target has no positive authenticated class weight"
        )
      sample_weight *= class_factor
    weights.append(sample_weight)
    pure_weights.append(cluster_weights[sample_index])
  zero = (
      accumulator_logits.sum() * 0.0
      + accumulator_residuals.sum() * 0.0
  )
  joint = _weighted_unique_sample_mean(joint_rows, weights, zero=zero)
  residual = _weighted_unique_sample_mean(
      residual_rows, weights, zero=zero
  )
  pure = _weighted_unique_sample_mean(
      pure_rows,
      pure_weights,
      zero=zero,
  )
  return JointSetProgramResidualLossV1(
      total=joint,
      joint_set_program_residual_nll=joint,
      audit_pure_set_program_nll=pure,
      diagnostic_best_residual_loss=residual,
      training_sample_count=len(joint_rows),
      excluded_oov_only_count=sum(
          not value for value in targets.classification_evaluable_mask
      ),
      unique_cluster_count=len({
          sample.cluster_key
          for sample in targets.samples
          if sample.classification_evaluable
      }),
      cluster_equal_weight_sum=float(sum(cluster_weights)),
      effective_weight_sum=float(sum(weights)),
  )


def set_valued_program_training_loss_v1(
    program_logits: Tensor,
    residual_predictions_pair_local: Tensor,
    targets: SetValuedProgramTargets,
    *,
    global_cluster_weights: FrozenGlobalClusterWeightsV1,
    catalog_class_weights: AuthenticatedCatalogClassWeightsV1 | None = None,
    residual_lambda: float = 1.0,
    residual_softmin_temperature: float | None = None,
) -> JointSetProgramResidualLossV1:
  """Canonical training entry point; classification-only NLL is audit-only."""

  return joint_set_program_residual_loss_v1(
      program_logits,
      residual_predictions_pair_local,
      targets,
      global_cluster_weights=global_cluster_weights,
      catalog_class_weights=catalog_class_weights,
      residual_lambda=residual_lambda,
      residual_softmin_temperature=residual_softmin_temperature,
  )


@dataclass(frozen=True, slots=True)
class SetValuedTopKResultV1:
  hits: tuple[bool, ...]
  hit_count: int
  sample_count: int
  classification_evaluable_count: int
  oov_only_miss_count: int

  @property
  def recall(self) -> float:
    return self.hit_count / self.sample_count


def evaluate_set_valued_topk_v1(
    ranked_catalog_indices: Sequence[Sequence[int]],
    targets: SetValuedProgramTargets,
    *,
    top_k: int,
) -> SetValuedTopKResultV1:
  """Hit any known target once per query; OOV-only is an automatic miss."""

  if type(targets) is not SetValuedProgramTargets:
    raise TypeError("top-K evaluation requires SetValuedProgramTargets")
  if type(top_k) is not int or not 1 <= top_k <= targets.catalog_size:
    raise ValueError("top-K lies outside the finite catalog")
  if len(ranked_catalog_indices) != len(targets.samples):
    raise ValueError("ranked sample count differs")
  hits: list[bool] = []
  for ranking, sample in zip(
      ranked_catalog_indices, targets.samples, strict=True
  ):
    values = tuple(
        _exact_int(value, label="ranked catalog index")
        for value in ranking
    )
    if (
        len(values) < top_k
        or len(values) != len(set(values))
        or any(value >= targets.catalog_size for value in values)
    ):
      raise ValueError("ranked catalog indices are incomplete or malformed")
    hits.append(
        bool(set(values[:top_k]).intersection(sample.known_catalog_indices))
    )
  frozen = tuple(hits)
  return SetValuedTopKResultV1(
      hits=frozen,
      hit_count=sum(frozen),
      sample_count=len(frozen),
      classification_evaluable_count=sum(
          targets.classification_evaluable_mask
      ),
      oov_only_miss_count=sum(
          not evaluable
          for evaluable in targets.classification_evaluable_mask
      ),
  )


@dataclass(frozen=True, slots=True)
class MatchedCandidateContractV1:
  """Diagnostic proposal contract; it makes no runtime/execution claim."""

  opaque_query_identity: str
  direction: str
  diagnostic_topk_hit: bool
  automatic_oov_miss: bool
  candidate_catalog_indices: tuple[int, ...]
  ranked_catalog_indices: tuple[int, ...]
  proposal_budget: Mapping[str, Any]
  runner_budget_enforcement_required: tuple[str, ...] = (
      "wall_time_seconds",
      "max_occ_calls",
      "timeout",
  )
  runner_budget_validated: bool = False
  execution_claimed: bool = False


def validate_multi_target_proposal_contract_v1(
    *,
    ranked_catalog_indices: Sequence[int],
    candidate_catalog_indices: Sequence[int],
    target: SetValuedProgramSampleV1,
    budget: ProposalBudget,
    fixed_catalog_domain: frozenset[int] | None = None,
    catalog_size: int | None = None,
) -> MatchedCandidateContractV1:
  """Validate an explicit, possibly non-contiguous matched candidate domain.

  The caller/runner must separately enforce wall time, OCC calls, and timeout.
  This function only validates membership, K, and the unchanged budget trace.
  """

  if type(target) is not SetValuedProgramSampleV1:
    raise TypeError("matched contract requires one set-valued target")
  if type(budget) is not ProposalBudget:
    raise TypeError("matched contract requires ProposalBudget")
  if (fixed_catalog_domain is None) == (catalog_size is None):
    raise ValueError(
        "matched contract requires exactly one explicit fixed catalog domain"
    )
  if fixed_catalog_domain is not None:
    if type(fixed_catalog_domain) is not frozenset or any(
        type(value) is not int for value in fixed_catalog_domain
    ):
      raise ValueError("fixed catalog domain must contain exact integers")
    catalog_domain = fixed_catalog_domain
  else:
    if type(catalog_size) is not int or catalog_size != 19:
      raise ValueError("fixed catalog size must be exactly 19")
    catalog_domain = frozenset(range(catalog_size))
  if catalog_domain != frozenset(range(19)):
    raise ValueError("matched contract requires the fixed catalog domain 0..18")
  if not set(target.known_catalog_indices).issubset(catalog_domain):
    raise ValueError("known gold target lies outside the fixed catalog domain")
  candidates = tuple(
      _exact_int(value, label="candidate catalog index")
      for value in candidate_catalog_indices
  )
  ranking = tuple(
      _exact_int(value, label="ranked catalog index")
      for value in ranked_catalog_indices
  )
  if (
      len(candidates) != budget.candidate_pool_size
      or len(candidates) != len(set(candidates))
      or len(ranking) != budget.top_k
      or len(ranking) != len(set(ranking))
      or not set(candidates).issubset(catalog_domain)
      or not set(ranking).issubset(candidates)
  ):
    raise ValueError("matched candidate/ranking domain differs from budget")
  return MatchedCandidateContractV1(
      opaque_query_identity=target.opaque_query_identity,
      direction=target.direction,
      diagnostic_topk_hit=bool(
          set(ranking).intersection(target.known_catalog_indices)
      ),
      automatic_oov_miss=not target.classification_evaluable,
      candidate_catalog_indices=candidates,
      ranked_catalog_indices=ranking,
      proposal_budget=MappingProxyType(dict(budget.trace_dict())),
  )
