"""Evidence-backed executable mate constraints for the frozen 19-way roster.

The old finite-program roster is categorical: it names a relation/contact and
two surface types, but it does not authorize an arbitrary rigid transform.
This module gives executable meaning only to descriptors already supported by
the repository's symbolic solver:

* plane/plane ``seat_plane`` -> opposed coincident planes;
* cylinder/cylinder ``shaft_in_bore`` -> an unoriented common axis.

Every other frozen descriptor remains an explicit
``unsupported_program_semantics`` entry.  Surface templates for analytic
sphere centres exist for fixtures and future authorities, but the current
``generic_contact`` sphere roster entry does not authorize that interpretation.

Predicted six-vectors are projected onto each constraint's free/symmetry DOFs
and clipped by immutable catalog bounds.  They can therefore refine a finite
program but cannot replace it with a full SE(3) prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .domain_types import Transform
from .constraint_catalog_trust_root_v1 import (
    OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1,
    require_official_constraint_catalog_trust_root_v1,
)


SCHEMA_VERSION = "constraint_executable_program_catalog.v1"
TEMPLATE_SCHEMA_VERSION = "surface_constraint_template.v1"
EXECUTION_SCHEMA_VERSION = "constraint_program_execution.v1"
CERTIFICATE_SCHEMA_VERSION = "program_constraint_certificate.v1"
HANDLER_REGISTRY_SCHEMA_VERSION = "constraint_handler_registry.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUTHENTICATED_CATALOG_FACTORY_TOKEN = object()
_SPHERE_AUTHORITY_TOKEN = object()


class UnsupportedProgramSemantics(ValueError):
  """The frozen descriptor does not carry enough evidence to execute a mate."""


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _unit(value: Any, *, label: str) -> np.ndarray:
  vector = np.asarray(value, dtype=float).reshape(3)
  length = float(np.linalg.norm(vector))
  if not np.isfinite(vector).all() or not math.isfinite(length) or length <= 1e-12:
    raise ValueError(f"{label} is degenerate")
  return vector / length


def _dot(first: Any, second: Any) -> float:
  left = np.asarray(first, dtype=float).reshape(-1)
  right = np.asarray(second, dtype=float).reshape(-1)
  if left.shape != right.shape:
    raise ValueError("dot-product dimensions differ")
  return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _matmul(first: Any, second: Any) -> np.ndarray:
  left = np.asarray(first, dtype=float)
  right = np.asarray(second, dtype=float)
  if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
    raise ValueError("matrix-product dimensions differ")
  return np.asarray([
      [sum(float(left[row, inner]) * float(right[inner, column])
           for inner in range(left.shape[1]))
       for column in range(right.shape[1])]
      for row in range(left.shape[0])
  ], dtype=float)


def _matvec(matrix: Any, vector: Any) -> np.ndarray:
  value = np.asarray(vector, dtype=float).reshape(-1, 1)
  return _matmul(matrix, value).reshape(-1)


def _proper_rotation(value: Any, *, label: str) -> np.ndarray:
  rotation = np.asarray(value, dtype=float).reshape(3, 3)
  if not np.isfinite(rotation).all():
    raise ValueError(f"{label} is non-finite")
  if not np.allclose(_matmul(rotation.T, rotation), np.eye(3), atol=1e-7, rtol=0.0):
    raise ValueError(f"{label} is not orthogonal")
  determinant = float(
      rotation[0, 0] * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
      - rotation[0, 1] * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
      + rotation[0, 2] * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
  )
  if not math.isclose(determinant, 1.0, abs_tol=1e-7):
    raise ValueError(f"{label} is not a proper rotation")
  return rotation


def _transform_matrix(transform: Transform) -> np.ndarray:
  if type(transform) is not Transform:
    raise TypeError("constraint execution requires an immutable Transform")
  result = np.eye(4, dtype=float)
  result[:3, :3] = transform.rotation
  result[:3, 3] = transform.translation
  return result


def _rotation_vector_matrix(vector: Any) -> np.ndarray:
  value = np.asarray(vector, dtype=float).reshape(3)
  angle = float(np.linalg.norm(value))
  if not np.isfinite(value).all():
    raise ValueError("rotation refinement is non-finite")
  if angle <= 1e-12:
    cross = np.asarray(
        [[0.0, -value[2], value[1]],
         [value[2], 0.0, -value[0]],
         [-value[1], value[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + cross
  axis = value / angle
  cross = np.asarray(
      [[0.0, -axis[2], axis[1]],
       [axis[2], 0.0, -axis[0]],
       [-axis[1], axis[0], 0.0]],
      dtype=float,
  )
  return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * _matmul(cross, cross)


def _rotation_from_to(source: Any, target: Any) -> np.ndarray:
  first = _unit(source, label="source direction")
  second = _unit(target, label="target direction")
  cosine = max(-1.0, min(1.0, _dot(first, second)))
  if cosine >= 1.0 - 1e-12:
    return np.eye(3)
  if cosine <= -1.0 + 1e-12:
    seeds = np.eye(3)
    seed = seeds[int(np.argmin([abs(_dot(row, first)) for row in seeds]))]
    axis = _unit(np.cross(first, seed), label="anti-parallel rotation axis")
    return _rotation_vector_matrix(axis * math.pi)
  axis_cross = np.cross(first, second)
  sine = float(np.linalg.norm(axis_cross))
  axis = axis_cross / sine
  angle = math.atan2(sine, cosine)
  return _rotation_vector_matrix(axis * angle)


def _angle_degrees(first: Any, second: Any, *, unoriented: bool) -> float:
  left = _unit(first, label="certificate first direction")
  right = _unit(second, label="certificate second direction")
  cosine = _dot(left, right)
  if unoriented:
    cosine = abs(cosine)
  return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _clip_norm(value: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
  norm = float(np.linalg.norm(value))
  if norm <= maximum or norm <= 1e-15:
    return value, False
  return value * (maximum / norm), True


@dataclass(frozen=True, slots=True)
class ConstraintProgramDescriptorV1:
  program_index: int
  program_id: str
  relation_hint: str
  contact_type: str
  surface_type_a: str
  surface_type_b: str

  def __post_init__(self) -> None:
    if type(self.program_index) is not int or self.program_index < 0:
      raise ValueError("constraint descriptor program index differs")
    if not self.program_id:
      raise ValueError("constraint descriptor program ID is empty")
    for field in (
        "relation_hint", "contact_type", "surface_type_a", "surface_type_b"
    ):
      value = getattr(self, field)
      if not isinstance(value, str) or not value or value != value.strip().lower():
        raise ValueError("constraint descriptor fields must be normalized")

  def semantic_key(self) -> tuple[str, str, str, str]:
    return (
        self.relation_hint, self.contact_type,
        self.surface_type_a, self.surface_type_b,
    )

  def payload(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class ConstraintProgramTemplateV1:
  template_id: str
  constraint_kind: str
  surface_type_a: str
  surface_type_b: str
  constrained_dofs: tuple[str, ...]
  free_dofs: tuple[str, ...]
  symmetry_quotient: tuple[str, ...]
  max_free_translation_mm: float
  max_free_rotation_degrees: float
  certificate_metrics: tuple[str, ...]
  schema_version: str = TEMPLATE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != TEMPLATE_SCHEMA_VERSION or not self.template_id:
      raise ValueError("constraint template identity differs")
    if self.constraint_kind not in {
        "opposed_coincident_planes", "unoriented_coaxial_cylinders",
        "coincident_analytic_sphere_centers",
    }:
      raise ValueError("constraint template kind differs")
    if self.max_free_translation_mm < 0 or self.max_free_rotation_degrees < 0:
      raise ValueError("constraint refinement bounds must be non-negative")

  def payload(self) -> dict[str, Any]:
    return asdict(self)


PLANE_COINCIDENT_TEMPLATE_V1 = ConstraintProgramTemplateV1(
    template_id="plane_plane_seat_coincident.v1",
    constraint_kind="opposed_coincident_planes",
    surface_type_a="plane",
    surface_type_b="plane",
    constrained_dofs=("normal_translation", "tilt_x", "tilt_y"),
    free_dofs=("tangent_x", "tangent_y", "normal_yaw"),
    symmetry_quotient=("in_plane_translation", "rotation_about_normal"),
    max_free_translation_mm=5.0,
    max_free_rotation_degrees=30.0,
    certificate_metrics=("normal_error_degrees", "normal_gap_mm"),
)

CYLINDER_COAXIAL_TEMPLATE_V1 = ConstraintProgramTemplateV1(
    template_id="cylinder_cylinder_shaft_bore_coaxial.v1",
    constraint_kind="unoriented_coaxial_cylinders",
    surface_type_a="cylinder",
    surface_type_b="cylinder",
    constrained_dofs=("radial_x", "radial_y", "tilt_x", "tilt_y"),
    free_dofs=("axial_translation", "axis_yaw"),
    symmetry_quotient=("axis_sign", "rotation_about_axis", "translation_along_axis"),
    max_free_translation_mm=5.0,
    max_free_rotation_degrees=30.0,
    certificate_metrics=("axis_error_degrees", "radial_axis_distance_mm"),
)

_SPHERE_CENTER_TEMPLATE_V1 = ConstraintProgramTemplateV1(
    template_id="sphere_sphere_analytic_center_coincident.v1",
    constraint_kind="coincident_analytic_sphere_centers",
    surface_type_a="sphere",
    surface_type_b="sphere",
    constrained_dofs=("center_x", "center_y", "center_z"),
    free_dofs=("rotation_x", "rotation_y", "rotation_z"),
    symmetry_quotient=("all_rotations"),
    max_free_translation_mm=0.0,
    max_free_rotation_degrees=30.0,
    certificate_metrics=("center_distance_mm",),
)


_SUPPORTED_SEMANTICS: Mapping[
    tuple[str, str, str, str], tuple[ConstraintProgramTemplateV1, str]
] = MappingProxyType({
    ("support", "seat_plane", "plane", "plane"): (
        PLANE_COINCIDENT_TEMPLATE_V1,
        "supported_by_existing_coincident_solver",
    ),
    ("insert", "shaft_in_bore", "cylinder", "cylinder"): (
        CYLINDER_COAXIAL_TEMPLATE_V1,
        "supported_by_existing_concentric_solver",
    ),
})


def _unsupported_reason(descriptor: ConstraintProgramDescriptorV1) -> str:
  if descriptor.contact_type in {"boss_in_slot", "pin_in_slot"}:
    return "slot_semantics_not_reducible_to_coaxial_cylinders"
  if descriptor.surface_type_a == descriptor.surface_type_b == "sphere":
    return "generic_sphere_contact_lacks_center_distance_semantics"
  if "spline" in {descriptor.surface_type_a, descriptor.surface_type_b}:
    return "spline_constraint_template_missing"
  if "torus" in {descriptor.surface_type_a, descriptor.surface_type_b}:
    return "torus_constraint_template_missing"
  if "cone" in {descriptor.surface_type_a, descriptor.surface_type_b}:
    return "cone_constraint_template_missing"
  if descriptor.contact_type == "generic_contact":
    return "generic_contact_does_not_define_constrained_dofs"
  return "descriptor_constraint_template_missing"


@dataclass(frozen=True, slots=True)
class ConstraintProgramEntryV1:
  descriptor: ConstraintProgramDescriptorV1
  status: str
  reason_code: str
  template: ConstraintProgramTemplateV1 | None
  evidence_refs: tuple[str, ...]

  def __post_init__(self) -> None:
    if self.status not in {"supported", "unsupported_program_semantics"}:
      raise ValueError("constraint program status differs")
    if (self.status == "supported") != (self.template is not None):
      raise ValueError("constraint program template/status differs")

  def payload(self) -> dict[str, Any]:
    return {
        "descriptor": self.descriptor.payload(),
        "status": self.status,
        "reason_code": self.reason_code,
        "template": None if self.template is None else self.template.payload(),
        "evidence_refs": list(self.evidence_refs),
    }


@dataclass(frozen=True, slots=True)
class ConstraintExecutableProgramCatalogV1:
  entries: tuple[ConstraintProgramEntryV1, ...]
  roster_sha256: str
  source_catalog_sha256: str
  semantic_catalog_sha256: str
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != SCHEMA_VERSION or len(self.entries) != 19:
      raise ValueError("constraint executable catalog domain differs")
    if tuple(row.descriptor.program_index for row in self.entries) != tuple(range(19)):
      raise ValueError("constraint executable catalog order differs")
    expected_roster = _canonical_sha256([
        row.descriptor.payload() for row in self.entries
    ])
    expected_catalog = _canonical_sha256({
        "schema_version": self.schema_version,
        "roster_sha256": expected_roster,
        "source_catalog_sha256": self.source_catalog_sha256,
        "entries": [row.payload() for row in self.entries],
    })
    if (
        self.roster_sha256 != expected_roster
        or len(self.source_catalog_sha256) != 64
        or any(value not in "0123456789abcdef" for value in self.source_catalog_sha256)
        or self.semantic_catalog_sha256 != expected_catalog
    ):
      raise ValueError("constraint executable catalog hash differs")

  @property
  def entry_count(self) -> int:
    return len(self.entries)

  @property
  def supported_program_indices(self) -> tuple[int, ...]:
    return tuple(
        row.descriptor.program_index for row in self.entries if row.status == "supported"
    )

  @property
  def coverage_fraction(self) -> float:
    return len(self.supported_program_indices) / len(self.entries)

  def entry(self, program_index: int) -> ConstraintProgramEntryV1:
    if type(program_index) is not int or not 0 <= program_index < len(self.entries):
      raise ValueError("constraint program index is outside [0,19)")
    return self.entries[program_index]

  def require_supported(self, proposal_or_index: Any) -> ConstraintProgramEntryV1:
    program_index = (
        proposal_or_index if type(proposal_or_index) is int
        else getattr(proposal_or_index, "program_index", None)
    )
    entry = self.entry(program_index)
    if type(proposal_or_index) is not int:
      descriptor = getattr(proposal_or_index, "descriptor", None)
      proposal_payload = descriptor.payload() if hasattr(descriptor, "payload") else None
      expected_payload = {
          "relation_hint": entry.descriptor.relation_hint,
          "contact_type": entry.descriptor.contact_type,
          "surface_type_a": entry.descriptor.surface_type_a,
          "surface_type_b": entry.descriptor.surface_type_b,
      }
      if (
          getattr(proposal_or_index, "program_id", None) != entry.descriptor.program_id
          or getattr(proposal_or_index, "catalog_sha256", None)
          != self.source_catalog_sha256
          or proposal_payload != expected_payload
      ):
        raise ValueError("proposal differs from constraint semantic catalog")
    if entry.status != "supported" or entry.template is None:
      raise UnsupportedProgramSemantics(
          f"unsupported_program_semantics:{entry.reason_code}"
      )
    return entry


@dataclass(frozen=True, slots=True)
class AuthenticatedConstraintExecutableProgramCatalogV1:
  """Factory-only semantic catalog rooted in the externally frozen roster.

  The public definition builder remains useful for unit-level mathematics, but
  it is deliberately not accepted by exact execution.  This capability
  reopens the frozen spec on every use and binds its path, bytes, lineage,
  producer pin, handler registry and implementation bytes.
  """

  _definition: ConstraintExecutableProgramCatalogV1
  _source_spec_path: str
  source_spec_file_sha256: str
  source_spec_payload_sha256: str
  source_artifact_sha256: str
  source_producer_code_sha256: str
  handler_registry_sha256: str
  semantic_source_sha256: str
  trust_root_source_sha256: str
  semantic_catalog_sha256: str
  _factory_token: object
  schema_version: str = "authenticated_constraint_executable_program_catalog.v1"

  def __post_init__(self) -> None:
    if self._factory_token is not _AUTHENTICATED_CATALOG_FACTORY_TOKEN:
      raise TypeError("authenticated semantic catalogs are loader-only")
    if type(self._definition) is not ConstraintExecutableProgramCatalogV1:
      raise TypeError("authenticated semantic catalog definition differs")
    if any(_SHA256.fullmatch(value) is None for value in (
        self.source_spec_file_sha256,
        self.source_spec_payload_sha256,
        self.source_artifact_sha256,
        self.source_producer_code_sha256,
        self.handler_registry_sha256,
        self.semantic_source_sha256,
        self.trust_root_source_sha256,
        self.semantic_catalog_sha256,
    )):
      raise ValueError("authenticated semantic catalog hash differs")
    if not Path(self._source_spec_path).is_absolute():
      raise ValueError("authenticated semantic catalog source path must be absolute")
    if self.semantic_catalog_sha256 != self._expected_semantic_sha256():
      raise ValueError("authenticated semantic catalog commitment differs")

  def _expected_semantic_sha256(self) -> str:
    return _canonical_sha256({
        "schema_version": self.schema_version,
        "definition_sha256": self._definition.semantic_catalog_sha256,
        "source_catalog_sha256": self._definition.source_catalog_sha256,
        "source_spec_file_sha256": self.source_spec_file_sha256,
        "source_spec_payload_sha256": self.source_spec_payload_sha256,
        "source_artifact_sha256": self.source_artifact_sha256,
        "source_producer_code_sha256": self.source_producer_code_sha256,
        "handler_registry_sha256": self.handler_registry_sha256,
        "semantic_source_sha256": self.semantic_source_sha256,
        "trust_root_source_sha256": self.trust_root_source_sha256,
    })

  @property
  def entries(self) -> tuple[ConstraintProgramEntryV1, ...]:
    return self._definition.entries

  @property
  def source_catalog_sha256(self) -> str:
    return self._definition.source_catalog_sha256

  @property
  def supported_program_indices(self) -> tuple[int, ...]:
    return self._definition.supported_program_indices

  @property
  def coverage_fraction(self) -> float:
    return self._definition.coverage_fraction

  def entry(self, program_index: int) -> ConstraintProgramEntryV1:
    return self._definition.entry(program_index)

  def require_supported(self, proposal_or_index: Any) -> ConstraintProgramEntryV1:
    self.revalidate()
    return self._definition.require_supported(proposal_or_index)

  def revalidate(self) -> None:
    """Reopen and replay the official roster loader and every external pin."""

    from .joint_interface_program_learner_v3 import (
        ExternallyFrozenCatalogSpecV3,
        load_externally_frozen_catalog_spec_v3,
    )

    path = Path(self._source_spec_path)
    trust_source = Path(__file__).resolve().with_name("constraint_catalog_trust_root_v1.py")
    require_official_constraint_catalog_trust_root_v1(
        path,
        catalog_roster_sha256=self.source_catalog_sha256,
        spec_file_sha256=self.source_spec_file_sha256,
        source_artifact_sha256=self.source_artifact_sha256,
        producer_code_sha256=self.source_producer_code_sha256,
    )
    if path.resolve(strict=True) != path or _file_sha256(path) != self.source_spec_file_sha256:
      raise ValueError("authenticated semantic catalog source path/bytes changed")
    spec = load_externally_frozen_catalog_spec_v3(
        path,
        expected_catalog_roster_sha256=self.source_catalog_sha256,
        expected_spec_file_sha256=self.source_spec_file_sha256,
    )
    if type(spec) is not ExternallyFrozenCatalogSpecV3:
      raise TypeError("authenticated semantic catalog official loader type differs")
    replay = _definition_from_authenticated_spec(spec)
    if (
        spec.source_artifact_sha256 != self.source_artifact_sha256
        or spec.spec_payload_sha256 != self.source_spec_payload_sha256
        or spec.producer_code_sha256 != self.source_producer_code_sha256
        or replay != self._definition
        or _handler_registry_sha256() != self.handler_registry_sha256
        or _file_sha256(Path(__file__).resolve()) != self.semantic_source_sha256
        or _file_sha256(trust_source) != self.trust_root_source_sha256
        or self.semantic_catalog_sha256 != self._expected_semantic_sha256()
    ):
      raise ValueError("authenticated semantic catalog lineage/code replay differs")


def build_constraint_executable_catalog_v1(
    descriptors: Sequence[ConstraintProgramDescriptorV1],
    *,
    source_catalog_sha256: str | None = None,
) -> ConstraintExecutableProgramCatalogV1:
  ordered = tuple(sorted(descriptors, key=lambda row: row.program_index))
  if (
      len(ordered) != 19
      or tuple(row.program_index for row in ordered) != tuple(range(19))
      or len({row.program_id for row in ordered}) != 19
  ):
    raise ValueError("constraint catalog requires the exact unique 19-way roster")
  entries = []
  for descriptor in ordered:
    supported = _SUPPORTED_SEMANTICS.get(descriptor.semantic_key())
    if supported is None:
      entries.append(ConstraintProgramEntryV1(
          descriptor=descriptor,
          status="unsupported_program_semantics",
          reason_code=_unsupported_reason(descriptor),
          template=None,
          evidence_refs=(),
      ))
      continue
    template, reason = supported
    entries.append(ConstraintProgramEntryV1(
        descriptor=descriptor,
        status="supported",
        reason_code=reason,
        template=template,
        evidence_refs=(
            "solver.py:ConstraintType.COINCIDENT"
            if template is PLANE_COINCIDENT_TEMPLATE_V1
            else "solver.py:ConstraintType.CONCENTRIC",
            "pose_library_certifier.py:_axis_error_for_family",
        ),
    ))
  frozen = tuple(entries)
  roster_sha256 = _canonical_sha256([row.descriptor.payload() for row in frozen])
  source_sha256 = roster_sha256 if source_catalog_sha256 is None else source_catalog_sha256
  catalog_sha256 = _canonical_sha256({
      "schema_version": SCHEMA_VERSION,
      "roster_sha256": roster_sha256,
      "source_catalog_sha256": source_sha256,
      "entries": [row.payload() for row in frozen],
  })
  return ConstraintExecutableProgramCatalogV1(
      entries=frozen,
      roster_sha256=roster_sha256,
      source_catalog_sha256=source_sha256,
      semantic_catalog_sha256=catalog_sha256,
  )


def build_constraint_executable_catalog_from_roster_v2(
    roster: Any,
) -> ConstraintExecutableProgramCatalogV1:
  """Build an unauthenticated mathematical definition for tests only.

  Exact execution rejects this type.  Production callers must use
  :func:`load_authenticated_constraint_executable_catalog_v1`.
  """

  entries = getattr(roster, "entries", None)
  source_catalog_sha256 = getattr(roster, "catalog_sha256", None)
  if not isinstance(entries, tuple) or not isinstance(source_catalog_sha256, str):
    raise TypeError("constraint catalog requires a finite-program V2 roster")
  descriptors = []
  for expected_index, entry in enumerate(entries):
    descriptor = getattr(entry, "descriptor", None)
    payload = descriptor.payload() if hasattr(descriptor, "payload") else None
    if (
        getattr(entry, "program_index", None) != expected_index
        or not isinstance(payload, Mapping)
        or set(payload) != {
            "relation_hint", "contact_type", "surface_type_a", "surface_type_b"
        }
    ):
      raise ValueError("finite-program V2 roster entry differs")
    descriptors.append(ConstraintProgramDescriptorV1(
        program_index=expected_index,
        program_id=str(entry.program_id),
        relation_hint=str(payload["relation_hint"]),
        contact_type=str(payload["contact_type"]),
        surface_type_a=str(payload["surface_type_a"]),
        surface_type_b=str(payload["surface_type_b"]),
    ))
  return build_constraint_executable_catalog_v1(
      descriptors, source_catalog_sha256=source_catalog_sha256
  )


def _definition_from_authenticated_spec(spec: Any) -> ConstraintExecutableProgramCatalogV1:
  from .joint_interface_program_learner_v3 import ExternallyFrozenCatalogSpecV3

  if type(spec) is not ExternallyFrozenCatalogSpecV3:
    raise TypeError("semantic catalog requires the official external catalog loader type")
  return build_constraint_executable_catalog_from_roster_v2(spec.roster)


def load_authenticated_constraint_executable_catalog_v1(
    source_spec_path: str | Path,
    *,
    expected_catalog_roster_sha256: str,
    expected_spec_file_sha256: str,
    expected_source_artifact_sha256: str,
    expected_producer_code_sha256: str,
) -> AuthenticatedConstraintExecutableProgramCatalogV1:
  """Create the only catalog capability accepted by the exact executor."""

  from .joint_interface_program_learner_v3 import (
      ExternallyFrozenCatalogSpecV3,
      load_externally_frozen_catalog_spec_v3,
  )

  for value in (
      expected_catalog_roster_sha256,
      expected_spec_file_sha256,
      expected_source_artifact_sha256,
      expected_producer_code_sha256,
  ):
    if _SHA256.fullmatch(value) is None:
      raise ValueError("semantic catalog loader requires external SHA-256 pins")
  path = require_official_constraint_catalog_trust_root_v1(
      source_spec_path,
      catalog_roster_sha256=expected_catalog_roster_sha256,
      spec_file_sha256=expected_spec_file_sha256,
      source_artifact_sha256=expected_source_artifact_sha256,
      producer_code_sha256=expected_producer_code_sha256,
  )
  spec = load_externally_frozen_catalog_spec_v3(
      path,
      expected_catalog_roster_sha256=expected_catalog_roster_sha256,
      expected_spec_file_sha256=expected_spec_file_sha256,
  )
  if type(spec) is not ExternallyFrozenCatalogSpecV3:
    raise TypeError("semantic catalog official loader type differs")
  if (
      spec.source_artifact_sha256 != expected_source_artifact_sha256
      or spec.spec_payload_sha256
      != OFFICIAL_CONSTRAINT_CATALOG_TRUST_ROOT_V1["spec_payload_sha256"]
      or spec.producer_code_sha256 != expected_producer_code_sha256
  ):
    raise ValueError("semantic catalog external lineage/code pin differs")
  definition = _definition_from_authenticated_spec(spec)
  handler_hash = _handler_registry_sha256()
  source_hash = _file_sha256(Path(__file__).resolve())
  trust_source_hash = _file_sha256(
      Path(__file__).resolve().with_name("constraint_catalog_trust_root_v1.py")
  )
  base = {
      "schema_version": "authenticated_constraint_executable_program_catalog.v1",
      "definition_sha256": definition.semantic_catalog_sha256,
      "source_catalog_sha256": definition.source_catalog_sha256,
      "source_spec_file_sha256": expected_spec_file_sha256,
      "source_spec_payload_sha256": spec.spec_payload_sha256,
      "source_artifact_sha256": expected_source_artifact_sha256,
      "source_producer_code_sha256": expected_producer_code_sha256,
      "handler_registry_sha256": handler_hash,
      "semantic_source_sha256": source_hash,
      "trust_root_source_sha256": trust_source_hash,
  }
  result = AuthenticatedConstraintExecutableProgramCatalogV1(
      _definition=definition,
      _source_spec_path=str(path),
      source_spec_file_sha256=expected_spec_file_sha256,
      source_spec_payload_sha256=spec.spec_payload_sha256,
      source_artifact_sha256=expected_source_artifact_sha256,
      source_producer_code_sha256=expected_producer_code_sha256,
      handler_registry_sha256=handler_hash,
      semantic_source_sha256=source_hash,
      trust_root_source_sha256=trust_source_hash,
      semantic_catalog_sha256=_canonical_sha256(base),
      _factory_token=_AUTHENTICATED_CATALOG_FACTORY_TOKEN,
  )
  result.revalidate()
  return result


@dataclass(frozen=True, slots=True)
class SurfaceConstraintFrameV1:
  part_slot: str
  origin_world_mm: tuple[float, float, float]
  rotation_local_to_world: tuple[tuple[float, float, float], ...]
  analytic_center_world_mm: tuple[float, float, float] | None = None

  def __post_init__(self) -> None:
    if self.part_slot not in {"a", "b"}:
      raise ValueError("surface constraint frame part slot differs")
    origin = np.asarray(self.origin_world_mm, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
      raise ValueError("surface constraint frame origin differs")
    _proper_rotation(self.rotation_local_to_world, label="surface constraint frame")
    if self.analytic_center_world_mm is not None:
      center = np.asarray(self.analytic_center_world_mm, dtype=float)
      if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("surface constraint analytic center differs")


@dataclass(frozen=True, slots=True)
class AppliedConstraintRefinementV1:
  translation_world_mm: tuple[float, float, float]
  rotation_vector_world: tuple[float, float, float]
  translation_norm_mm: float
  rotation_angle_degrees: float
  translation_clipped: bool
  rotation_clipped: bool
  blocked_translation_dofs: tuple[str, ...]
  blocked_rotation_dofs: tuple[str, ...]

  def payload(self) -> dict[str, Any]:
    return {
        "translation_world_mm": list(self.translation_world_mm),
        "rotation_vector_world": list(self.rotation_vector_world),
        "translation_norm_mm": self.translation_norm_mm,
        "rotation_angle_degrees": self.rotation_angle_degrees,
        "translation_clipped": self.translation_clipped,
        "rotation_clipped": self.rotation_clipped,
        "blocked_translation_dofs": list(self.blocked_translation_dofs),
        "blocked_rotation_dofs": list(self.blocked_rotation_dofs),
    }


@dataclass(frozen=True, slots=True)
class ProgramConstraintCertificateV1:
  template_id: str
  valid: bool
  metric_items: tuple[tuple[str, float], ...]
  reason_codes: tuple[str, ...]
  schema_version: str = CERTIFICATE_SCHEMA_VERSION

  @property
  def metrics(self) -> Mapping[str, float]:
    return MappingProxyType(dict(self.metric_items))

  def payload(self) -> dict[str, Any]:
    return {
        "schema_version": self.schema_version,
        "template_id": self.template_id,
        "valid": self.valid,
        "metrics": dict(self.metric_items),
        "reason_codes": list(self.reason_codes),
    }


@dataclass(frozen=True, slots=True)
class ConstraintProgramExecutionV1:
  template_id: str
  _world_delta_row_major: tuple[float, ...]
  _child_world_row_major: tuple[float, ...]
  refinement: AppliedConstraintRefinementV1
  certificate: ProgramConstraintCertificateV1
  schema_version: str = EXECUTION_SCHEMA_VERSION

  @property
  def world_delta_matrix(self) -> np.ndarray:
    return np.asarray(self._world_delta_row_major, dtype=float).reshape(4, 4).copy()

  @property
  def child_world_matrix(self) -> np.ndarray:
    return np.asarray(self._child_world_row_major, dtype=float).reshape(4, 4).copy()


def _pair_rotation(
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
) -> np.ndarray:
  rotation_a = np.asarray(frame_a.rotation_local_to_world, dtype=float)
  rotation_b = np.asarray(frame_b.rotation_local_to_world, dtype=float)
  normal_a = rotation_a[:, 2]
  normal_b = rotation_b[:, 2]
  z_seed = normal_a - normal_b
  if float(np.linalg.norm(z_seed)) < 1e-10:
    z_seed = normal_a
  z_axis = _unit(z_seed, label="pair-frame z")
  x_axis = None
  for seed in (rotation_a[:, 0], rotation_b[:, 0], rotation_a[:, 1]):
    projected = seed - _dot(seed, z_axis) * z_axis
    if float(np.linalg.norm(projected)) >= 1e-10:
      x_axis = _unit(projected, label="pair-frame x")
      break
  if x_axis is None:
    raise ValueError("constraint pair frame is geometrically degenerate")
  y_axis = _unit(np.cross(z_axis, x_axis), label="pair-frame y")
  x_axis = _unit(np.cross(y_axis, z_axis), label="pair-frame reorthogonalized x")
  return np.stack((x_axis, y_axis, z_axis), axis=1)


@dataclass(frozen=True, slots=True)
class _ConstraintHandlerV1:
  kind: str
  handler_id: str
  certificate_limits: tuple[tuple[str, float], ...]
  base_delta: Callable[[SurfaceConstraintFrameV1, SurfaceConstraintFrameV1], tuple[np.ndarray, np.ndarray]]
  project: Callable[[SurfaceConstraintFrameV1, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]]
  metrics: Callable[[SurfaceConstraintFrameV1, SurfaceConstraintFrameV1, np.ndarray], Mapping[str, float]]
  authority_scope: str = "frozen_roster"

  def payload(self) -> dict[str, Any]:
    return {
        "kind": self.kind,
        "handler_id": self.handler_id,
        "certificate_limits": dict(self.certificate_limits),
        "authority_scope": self.authority_scope,
        "schema_version": HANDLER_REGISTRY_SCHEMA_VERSION,
    }


def _plane_base(
    frame_a: SurfaceConstraintFrameV1, frame_b: SurfaceConstraintFrameV1,
) -> tuple[np.ndarray, np.ndarray]:
  origin_a = np.asarray(frame_a.origin_world_mm, dtype=float)
  origin_b = np.asarray(frame_b.origin_world_mm, dtype=float)
  z_a = np.asarray(frame_a.rotation_local_to_world, dtype=float)[:, 2]
  z_b = np.asarray(frame_b.rotation_local_to_world, dtype=float)[:, 2]
  rotation = _rotation_from_to(z_b, -z_a)
  pivot = origin_b - _dot(origin_b - origin_a, z_a) * z_a
  delta = np.eye(4)
  delta[:3, :3] = rotation
  delta[:3, 3] = pivot - _matvec(rotation, origin_b)
  return delta, pivot


def _cylinder_base(
    frame_a: SurfaceConstraintFrameV1, frame_b: SurfaceConstraintFrameV1,
) -> tuple[np.ndarray, np.ndarray]:
  origin_a = np.asarray(frame_a.origin_world_mm, dtype=float)
  origin_b = np.asarray(frame_b.origin_world_mm, dtype=float)
  z_a = np.asarray(frame_a.rotation_local_to_world, dtype=float)[:, 2]
  z_b = np.asarray(frame_b.rotation_local_to_world, dtype=float)[:, 2]
  target_axis = z_a if _dot(z_a, z_b) >= 0.0 else -z_a
  rotation = _rotation_from_to(z_b, target_axis)
  pivot = origin_a + _dot(origin_b - origin_a, z_a) * z_a
  delta = np.eye(4)
  delta[:3, :3] = rotation
  delta[:3, 3] = pivot - _matvec(rotation, origin_b)
  return delta, pivot


def _sphere_base(
    frame_a: SurfaceConstraintFrameV1, frame_b: SurfaceConstraintFrameV1,
) -> tuple[np.ndarray, np.ndarray]:
  if frame_a.analytic_center_world_mm is None or frame_b.analytic_center_world_mm is None:
    raise UnsupportedProgramSemantics(
        "unsupported_program_semantics:analytic sphere center authority missing"
    )
  center_a = np.asarray(frame_a.analytic_center_world_mm, dtype=float)
  center_b = np.asarray(frame_b.analytic_center_world_mm, dtype=float)
  delta = np.eye(4)
  delta[:3, 3] = center_a - center_b
  return delta, center_a


def _plane_project(
    frame_a: SurfaceConstraintFrameV1,
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
  z_axis = np.asarray(frame_a.rotation_local_to_world, dtype=float)[:, 2]
  return (
      translation_world - _dot(translation_world, z_axis) * z_axis,
      _dot(rotation_world, z_axis) * z_axis,
      ("normal",),
      ("tilt_x", "tilt_y"),
  )


def _cylinder_project(
    frame_a: SurfaceConstraintFrameV1,
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
  z_axis = np.asarray(frame_a.rotation_local_to_world, dtype=float)[:, 2]
  return (
      _dot(translation_world, z_axis) * z_axis,
      _dot(rotation_world, z_axis) * z_axis,
      ("radial_x", "radial_y"),
      ("tilt_x", "tilt_y"),
  )


def _sphere_project(
    _frame_a: SurfaceConstraintFrameV1,
    _translation_world: np.ndarray,
    rotation_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
  return np.zeros(3), rotation_world, ("x", "y", "z"), ()


def _moved_frame(
    frame_b: SurfaceConstraintFrameV1, world_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  origin_b = np.asarray(frame_b.origin_world_mm, dtype=float)
  rotation_b = np.asarray(frame_b.rotation_local_to_world, dtype=float)
  return (
      _matvec(world_delta[:3, :3], origin_b) + world_delta[:3, 3],
      _matmul(world_delta[:3, :3], rotation_b),
  )


def _plane_metrics(
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
    world_delta: np.ndarray,
) -> Mapping[str, float]:
  moved_origin_b, moved_rotation_b = _moved_frame(frame_b, world_delta)
  origin_a = np.asarray(frame_a.origin_world_mm, dtype=float)
  z_a = np.asarray(frame_a.rotation_local_to_world, dtype=float)[:, 2]
  return {
      "normal_error_degrees": _angle_degrees(moved_rotation_b[:, 2], -z_a, unoriented=False),
      "normal_gap_mm": abs(_dot(moved_origin_b - origin_a, z_a)),
  }


def _cylinder_metrics(
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
    world_delta: np.ndarray,
) -> Mapping[str, float]:
  moved_origin_b, moved_rotation_b = _moved_frame(frame_b, world_delta)
  origin_a = np.asarray(frame_a.origin_world_mm, dtype=float)
  z_a = np.asarray(frame_a.rotation_local_to_world, dtype=float)[:, 2]
  delta = moved_origin_b - origin_a
  radial = delta - _dot(delta, z_a) * z_a
  return {
      "axis_error_degrees": _angle_degrees(moved_rotation_b[:, 2], z_a, unoriented=True),
      "radial_axis_distance_mm": float(np.linalg.norm(radial)),
  }


def _sphere_metrics(
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
    world_delta: np.ndarray,
) -> Mapping[str, float]:
  if frame_a.analytic_center_world_mm is None or frame_b.analytic_center_world_mm is None:
    raise UnsupportedProgramSemantics(
        "unsupported_program_semantics:analytic sphere center authority missing"
    )
  center_a = np.asarray(frame_a.analytic_center_world_mm, dtype=float)
  center_b = np.asarray(frame_b.analytic_center_world_mm, dtype=float)
  moved_center_b = _matvec(world_delta[:3, :3], center_b) + world_delta[:3, 3]
  return {"center_distance_mm": float(np.linalg.norm(moved_center_b - center_a))}


_HANDLER_REGISTRY: Mapping[str, _ConstraintHandlerV1] = MappingProxyType({
    "opposed_coincident_planes": _ConstraintHandlerV1(
        kind="opposed_coincident_planes",
        handler_id="opposed_coincident_planes.v1",
        certificate_limits=(("normal_error_degrees", 1e-6), ("normal_gap_mm", 1e-7)),
        base_delta=_plane_base,
        project=_plane_project,
        metrics=_plane_metrics,
    ),
    "unoriented_coaxial_cylinders": _ConstraintHandlerV1(
        kind="unoriented_coaxial_cylinders",
        handler_id="unoriented_coaxial_cylinders.v1",
        certificate_limits=(("axis_error_degrees", 1e-6), ("radial_axis_distance_mm", 1e-7)),
        base_delta=_cylinder_base,
        project=_cylinder_project,
        metrics=_cylinder_metrics,
    ),
    "coincident_analytic_sphere_centers": _ConstraintHandlerV1(
        kind="coincident_analytic_sphere_centers",
        handler_id="coincident_analytic_sphere_centers.internal.v1",
        certificate_limits=(("center_distance_mm", 1e-7),),
        base_delta=_sphere_base,
        project=_sphere_project,
        metrics=_sphere_metrics,
        authority_scope="internal_analytic_fixture_only_not_roster_program_18",
    ),
})


def _handler_registry_sha256() -> str:
  return _canonical_sha256({
      "schema_version": HANDLER_REGISTRY_SCHEMA_VERSION,
      "handlers": [handler.payload() for _, handler in sorted(_HANDLER_REGISTRY.items())],
  })


def _handler(template: ConstraintProgramTemplateV1) -> _ConstraintHandlerV1:
  try:
    return _HANDLER_REGISTRY[template.constraint_kind]
  except KeyError as error:
    raise UnsupportedProgramSemantics(
        "unsupported_program_semantics:constraint handler missing"
    ) from error


def _base_constraint_delta(
    template: ConstraintProgramTemplateV1,
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
) -> tuple[np.ndarray, np.ndarray]:
  return _handler(template).base_delta(frame_a, frame_b)


def _project_refinement(
    template: ConstraintProgramTemplateV1,
    *,
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
    translation_pair_local: Sequence[float],
    rotation_vector_pair_local: Sequence[float],
) -> AppliedConstraintRefinementV1:
  translation_local = np.asarray(translation_pair_local, dtype=float).reshape(3)
  rotation_local = np.asarray(rotation_vector_pair_local, dtype=float).reshape(3)
  if not np.isfinite(translation_local).all() or not np.isfinite(rotation_local).all():
    raise ValueError("constraint refinement contains a non-finite value")
  pair_rotation = _pair_rotation(frame_a, frame_b)
  translation_world = _matvec(pair_rotation, translation_local)
  rotation_world = _matvec(pair_rotation, rotation_local)
  (
      translation_projected,
      rotation_projected,
      blocked_translation,
      blocked_rotation,
  ) = _handler(template).project(frame_a, translation_world, rotation_world)

  translation_projected, translation_clipped = _clip_norm(
      translation_projected, template.max_free_translation_mm
  )
  rotation_projected, rotation_clipped = _clip_norm(
      rotation_projected, math.radians(template.max_free_rotation_degrees)
  )
  return AppliedConstraintRefinementV1(
      translation_world_mm=tuple(float(value) for value in translation_projected),
      rotation_vector_world=tuple(float(value) for value in rotation_projected),
      translation_norm_mm=float(np.linalg.norm(translation_projected)),
      rotation_angle_degrees=math.degrees(float(np.linalg.norm(rotation_projected))),
      translation_clipped=translation_clipped,
      rotation_clipped=rotation_clipped,
      blocked_translation_dofs=blocked_translation,
      blocked_rotation_dofs=blocked_rotation,
  )


def _certificate(
    template: ConstraintProgramTemplateV1,
    *,
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
    world_delta: np.ndarray,
) -> ProgramConstraintCertificateV1:
  handler = _handler(template)
  metrics = dict(handler.metrics(frame_a, frame_b, world_delta))
  limits = dict(handler.certificate_limits)
  if set(metrics) != set(limits):
    raise ValueError("constraint handler metric/threshold registry drift")
  reasons = tuple(
      f"{name}_exceeds_{limits[name]:.9g}"
      for name, value in metrics.items() if value > limits[name]
  )
  return ProgramConstraintCertificateV1(
      template_id=template.template_id,
      valid=not reasons,
      metric_items=tuple(sorted((name, float(value)) for name, value in metrics.items())),
      reason_codes=reasons,
  )


def execute_surface_constraint_v1(
    *,
    template: ConstraintProgramTemplateV1,
    frame_a: SurfaceConstraintFrameV1,
    frame_b: SurfaceConstraintFrameV1,
    current_child_world: Transform,
    residual_translation_pair_local: Sequence[float],
    residual_rotation_vector_pair_local: Sequence[float],
    _analytic_authority_token: object | None = None,
) -> ConstraintProgramExecutionV1:
  """Execute one finite constraint and a bounded symmetry-quotient refinement."""

  if type(template) is not ConstraintProgramTemplateV1:
    raise TypeError("constraint execution requires a catalog template")
  if type(frame_a) is not SurfaceConstraintFrameV1 or type(frame_b) is not SurfaceConstraintFrameV1:
    raise TypeError("constraint execution requires authenticated surface frames")
  if frame_a.part_slot != "a" or frame_b.part_slot != "b":
    raise ValueError("constraint execution frame order differs")
  if (
      template.constraint_kind == "coincident_analytic_sphere_centers"
      and _analytic_authority_token is not _SPHERE_AUTHORITY_TOKEN
  ):
    raise UnsupportedProgramSemantics(
        "unsupported_program_semantics:analytic sphere execution is internal and "
        "does not authorize frozen roster program 18"
    )
  base_delta, pivot = _base_constraint_delta(template, frame_a, frame_b)
  refinement = _project_refinement(
      template,
      frame_a=frame_a,
      frame_b=frame_b,
      translation_pair_local=residual_translation_pair_local,
      rotation_vector_pair_local=residual_rotation_vector_pair_local,
  )
  refinement_rotation = _rotation_vector_matrix(refinement.rotation_vector_world)
  refinement_translation = np.asarray(refinement.translation_world_mm, dtype=float)
  refinement_delta = np.eye(4)
  refinement_delta[:3, :3] = refinement_rotation
  refinement_delta[:3, 3] = pivot + refinement_translation - _matvec(refinement_rotation, pivot)
  world_delta = _matmul(refinement_delta, base_delta)
  certificate = _certificate(
      template, frame_a=frame_a, frame_b=frame_b, world_delta=world_delta
  )
  if not certificate.valid:
    raise ValueError(
        "program-specific constraint certificate failed:" + ",".join(certificate.reason_codes)
    )
  child_world = _matmul(world_delta, _transform_matrix(current_child_world))
  return ConstraintProgramExecutionV1(
      template_id=template.template_id,
      _world_delta_row_major=tuple(float(value) for value in world_delta.reshape(-1)),
      _child_world_row_major=tuple(float(value) for value in child_world.reshape(-1)),
      refinement=refinement,
      certificate=certificate,
  )


__all__ = [
    "CYLINDER_COAXIAL_TEMPLATE_V1",
    "PLANE_COINCIDENT_TEMPLATE_V1",
    "AppliedConstraintRefinementV1",
    "ConstraintExecutableProgramCatalogV1",
    "AuthenticatedConstraintExecutableProgramCatalogV1",
    "ConstraintProgramDescriptorV1",
    "ConstraintProgramEntryV1",
    "ConstraintProgramExecutionV1",
    "ConstraintProgramTemplateV1",
    "ProgramConstraintCertificateV1",
    "SurfaceConstraintFrameV1",
    "UnsupportedProgramSemantics",
    "build_constraint_executable_catalog_v1",
    "build_constraint_executable_catalog_from_roster_v2",
    "load_authenticated_constraint_executable_catalog_v1",
    "execute_surface_constraint_v1",
]
