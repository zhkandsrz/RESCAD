"""Independent procedural candidate source for LinkCAD transfer evaluation.

The generator creates identity-disjoint CadQuery solids, exports each candidate
in an independently perturbed local frame, and separates public candidate sets
from private assignment/interface targets.  Target schedules depend only on the
declared seed and are materialized before any model prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import cadquery as cq

from .linkcad_language_contracts import render_mobility_instruction
from .linkcad_port_contract_v1 import (
    describe_ports_v1,
    render_port_conditioned_instruction_v1,
)
from .linkcad_primitive_graph_v2 import (
    LinkCADPrimitiveGraphV2,
    extract_primitive_graph_v2,
)


SCHEMA_VERSION = "linkcad_procedural_transfer.v1"
PUBLIC_SCHEMA_VERSION = "linkcad_candidate_set_public.v1"
PRIVATE_SCHEMA_VERSION = "linkcad_candidate_set_private_targets.v1"
SUPERVISION_SCHEMA_VERSION = "linkcad_direct_primitive_orbit_supervision.v2"
ROLE_COUNT = 4
CANDIDATE_COUNT = 4
LARGE_CANDIDATE_COUNT = 8


def canonical_json_bytes(value: Any) -> bytes:
  return json.dumps(
      value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
  ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ProceduralTransferConfigV1:
  seed: int = 271828
  query_count: int = 40
  candidates_per_role: int = CANDIDATE_COUNT
  roles_per_query: int = ROLE_COUNT

  def validate(self) -> None:
    if self.seed < 0 or self.query_count < 1:
      raise ValueError("LinkCAD procedural transfer config differs")
    if (
        self.candidates_per_role not in (CANDIDATE_COUNT, LARGE_CANDIDATE_COUNT)
        or self.roles_per_query != ROLE_COUNT
    ):
      raise ValueError("LinkCAD procedural transfer dimensions differ")

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": self.seed,
        "query_count": self.query_count,
        "candidates_per_role": self.candidates_per_role,
        "roles_per_query": self.roles_per_query,
    }


ROLE_SPECS = (
    {
        "description": "central multi-port connector",
        "target_family": "hub",
        "candidate_families": ("hub", "shaft", "plate", "block"),
    },
    {
        "description": "axial rotating member",
        "target_family": "shaft",
        "candidate_families": ("shaft", "ring", "plate", "bracket"),
    },
    {
        "description": "planar mounting member",
        "target_family": "plate",
        "candidate_families": ("plate", "block", "shaft", "ring"),
    },
    {
        "description": "guided pin member",
        "target_family": "pin",
        "candidate_families": ("pin", "plate", "ring", "bracket"),
    },
)

LARGE_CANDIDATE_FAMILIES = (
    ("hub", "shaft", "plate", "block", "ring", "bracket", "pin", "ring"),
    ("shaft", "ring", "plate", "bracket", "pin", "hub", "block", "pin"),
    ("plate", "block", "shaft", "ring", "bracket", "hub", "pin", "block"),
    ("pin", "plate", "ring", "bracket", "shaft", "block", "hub", "shaft"),
)

EDGE_SPECS = (
    {
        "role_a": "role_00", "role_b": "role_01",
        "port_a": ("face", "cylinder", "largest"),
        "port_b": ("face", "cylinder", "smallest"),
        "support_family": "axial_support",
        "mobility_cycle": ("revolute", "cylindrical", "prismatic"),
    },
    {
        "role_a": "role_00", "role_b": "role_02",
        "port_a": ("face", "plane", "largest"),
        "port_b": ("face", "plane", "largest"),
        "support_family": "planar_support",
        "mobility_cycle": ("fixed", "planar"),
    },
    {
        "role_a": "role_00", "role_b": "role_03",
        "port_a": ("edge", "circle", "smallest"),
        "port_b": ("face", "plane", "largest"),
        "support_family": "axis_plane_support",
        "mobility_cycle": ("prismatic",),
    },
)


def build_procedural_schedule_v1(
    config: ProceduralTransferConfigV1,
) -> tuple[dict[str, Any], ...]:
  config.validate()
  rows = []
  for query_ordinal in range(config.query_count):
    query_seed = int(hashlib.sha256(
        f"{config.seed}:query:{query_ordinal}".encode("utf-8")
    ).hexdigest()[:16], 16)
    rng = random.Random(query_seed)
    roles = []
    for role_ordinal, spec in enumerate(ROLE_SPECS):
      families = list(
          spec["candidate_families"]
          if config.candidates_per_role == CANDIDATE_COUNT
          else LARGE_CANDIDATE_FAMILIES[role_ordinal]
      )
      rng.shuffle(families)
      roles.append({
          "role_id": f"role_{role_ordinal:02d}",
          "description": spec["description"],
          "target_family": spec["target_family"],
          "candidate_families": families,
      })
    rows.append({
        "query_ordinal": query_ordinal,
        "query_seed": query_seed,
        "query_id": "linkcad_proc_" + hashlib.sha256(
            canonical_json_bytes({"seed": query_seed, "roles": roles})
        ).hexdigest()[:20],
        "roles": roles,
    })
  return tuple(rows)


def _shape_for_family(
    family: str, *, query_ordinal: int, candidate_ordinal: int,
) -> cq.Shape:
  phase = 0.17 * query_ordinal + 0.31 * candidate_ordinal
  jitter = 0.35 * (1.0 + math.sin(phase))
  if family == "hub":
    outer = 9.0 + jitter
    bore = 3.2 + 0.15 * ((query_ordinal + candidate_ordinal) % 4)
    length = 5.5 + 0.25 * (query_ordinal % 3)
    flange = outer + 3.0 + 0.2 * (candidate_ordinal % 2)
    return (
        cq.Workplane("XY").circle(outer).circle(bore).extrude(length)
        .faces(">Z").workplane().circle(flange).circle(bore).extrude(1.8)
        .val()
    )
  if family in {"shaft", "pin"}:
    base = (3.0 if family == "shaft" else 2.5) + 0.2 * jitter
    length = (17.0 if family == "shaft" else 11.0) + 0.35 * query_ordinal % 2.5
    head = base + (2.2 if family == "shaft" else 2.8)
    return (
        cq.Workplane("XY").circle(base).extrude(length)
        .faces(">Z").workplane().circle(head).extrude(2.2)
        .val()
    )
  if family == "plate":
    width = 22.0 + 0.5 * (query_ordinal % 5)
    height = 16.0 + 0.4 * (candidate_ordinal % 4)
    thickness = 3.0 + 0.15 * jitter
    hole = 5.0 + 0.2 * ((query_ordinal + candidate_ordinal) % 3)
    return (
        cq.Workplane("XY").box(width, height, thickness)
        .faces(">Z").workplane().hole(hole).val()
    )
  if family == "ring":
    outer = 8.0 + jitter
    inner = 3.5 + 0.25 * (candidate_ordinal % 3)
    return cq.Workplane("XY").circle(outer).circle(inner).extrude(4.0).val()
  if family == "bracket":
    length = 18.0 + 0.4 * (query_ordinal % 4)
    base = cq.Workplane("XY").box(length, 10.0, 3.0)
    upright = cq.Workplane("XZ").box(3.0, 10.0, 12.0).translate(
        (-(length - 3.0) / 2.0, 0.0, 4.5)
    )
    return base.union(upright).val()
  if family == "block":
    return cq.Workplane("XY").box(
        17.0 + jitter, 14.0 + 0.2 * query_ordinal % 2.0, 6.0,
    ).val()
  raise ValueError("LinkCAD procedural candidate family differs")


def _perturb_shape(
    shape: cq.Shape, *, query_seed: int, role_ordinal: int,
    candidate_ordinal: int,
) -> tuple[cq.Shape, list[float]]:
  rng = random.Random(query_seed ^ (role_ordinal << 12) ^ candidate_ordinal)
  angles = [rng.uniform(-170.0, 170.0) for _ in range(3)]
  translation = [rng.uniform(-80.0, 80.0) for _ in range(3)]
  transformed = shape
  for axis, angle in zip(
      (cq.Vector(1, 0, 0), cq.Vector(0, 1, 0), cq.Vector(0, 0, 1)),
      angles,
  ):
    transformed = transformed.rotate(cq.Vector(0, 0, 0), axis, angle)
  transformed = transformed.translate(cq.Vector(*translation))
  return transformed, [*angles, *translation]


def _select_port_ordinal(
    graph: LinkCADPrimitiveGraphV2, selector: tuple[str, str, str],
) -> int:
  kind, geometry, size = selector
  contracts = describe_ports_v1(graph)
  preferred = [
      ordinal for ordinal, contract in enumerate(contracts)
      if contract["primitive_kind"] == kind
      and contract["geometry_type"] == geometry
      and contract["size_class"] == size
  ]
  if preferred:
    return preferred[0]
  fallback = [
      ordinal for ordinal, contract in enumerate(contracts)
      if contract["primitive_kind"] == kind
      and contract["geometry_type"] == geometry
  ]
  if not fallback:
    raise ValueError("LinkCAD procedural target port is absent")
  return fallback[0] if size in {"only", "smallest", "small"} else fallback[-1]


def _write_json(path: Path, payload: Any) -> None:
  path.write_bytes(
      (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
  )


def _file_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_procedural_transfer_v1(
    *, config: ProceduralTransferConfigV1, dataset_root: Path,
    artifact_root: Path, historical_step_sha256s: set[str],
    code_revision: str, checkpoint_paths: tuple[Path, ...],
) -> dict[str, Any]:
  """Materialize a frozen public/private transfer protocol and STEP source."""

  config.validate()
  dataset_root = dataset_root.resolve()
  artifact_root = artifact_root.resolve()
  dataset_root.mkdir(parents=True, exist_ok=True)
  artifact_root.mkdir(parents=True, exist_ok=True)
  schedule = build_procedural_schedule_v1(config)
  public_queries = []
  private_targets = []
  supervision_rows = []
  candidate_programs = []
  all_step_sha256s: set[str] = set()
  for schedule_row in schedule:
    query_id = schedule_row["query_id"]
    query_ordinal = int(schedule_row["query_ordinal"])
    query_seed = int(schedule_row["query_seed"])
    public_roles = []
    candidate_sets: dict[str, list[dict[str, Any]]] = {}
    target_by_role: dict[str, str] = {}
    provenance_by_role: dict[str, list[dict[str, Any]]] = {}
    target_graph_by_role: dict[str, LinkCADPrimitiveGraphV2] = {}
    target_candidate_by_role: dict[str, dict[str, Any]] = {}
    for role_ordinal, role in enumerate(schedule_row["roles"]):
      role_id = str(role["role_id"])
      public_roles.append({
          "role_id": role_id, "description": str(role["description"]),
      })
      public_candidates = []
      provenance = []
      for candidate_ordinal, family in enumerate(role["candidate_families"]):
        shape = _shape_for_family(
            str(family), query_ordinal=query_ordinal,
            candidate_ordinal=candidate_ordinal,
        )
        shape, perturbation = _perturb_shape(
            shape, query_seed=query_seed, role_ordinal=role_ordinal,
            candidate_ordinal=candidate_ordinal,
        )
        relative_path = Path(
            f"query_{query_ordinal:04d}/role_{role_ordinal:02d}/"
            f"candidate_{candidate_ordinal:02d}.step"
        )
        output_path = dataset_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(shape, str(output_path), exportType="STEP")
        step_sha = _file_sha256(output_path)
        if step_sha in all_step_sha256s:
          raise ValueError("LinkCAD procedural STEP identity is duplicated")
        all_step_sha256s.add(step_sha)
        candidate_id = "proc_part_" + step_sha[:20]
        row = {
            "candidate_id": candidate_id,
            "step_path": relative_path.as_posix(),
            "step_sha256": step_sha,
            "volume": float(shape.Volume()),
            "area": float(shape.Area()),
        }
        public_candidates.append(row)
        provenance.append({
            "candidate_id": candidate_id,
            "step_sha256": step_sha,
            "generator": "cadquery_procedural_v1",
            "family": family,
            "query_ordinal": query_ordinal,
            "role_ordinal": role_ordinal,
            "candidate_ordinal": candidate_ordinal,
            "se3_angles_deg_then_translation_mm": perturbation,
        })
        candidate_programs.append(provenance[-1])
        if family == role["target_family"]:
          if role_id in target_by_role:
            raise ValueError("LinkCAD procedural target family is duplicated")
          target_by_role[role_id] = candidate_id
          target_candidate_by_role[role_id] = row
          target_graph_by_role[role_id] = extract_primitive_graph_v2(shape)
      if role_id not in target_by_role:
        raise ValueError("LinkCAD procedural target family is missing")
      candidate_sets[role_id] = public_candidates
      provenance_by_role[role_id] = provenance

    functional_edges = []
    target_edges = []
    for edge_ordinal, edge_spec in enumerate(EDGE_SPECS):
      edge_id = f"edge_{edge_ordinal:02d}"
      role_a = str(edge_spec["role_a"])
      role_b = str(edge_spec["role_b"])
      ordinal_a = _select_port_ordinal(
          target_graph_by_role[role_a], edge_spec["port_a"],
      )
      ordinal_b = _select_port_ordinal(
          target_graph_by_role[role_b], edge_spec["port_b"],
      )
      port_a = describe_ports_v1(target_graph_by_role[role_a])[ordinal_a]
      port_b = describe_ports_v1(target_graph_by_role[role_b])[ordinal_b]
      mobility_cycle = edge_spec["mobility_cycle"]
      mobility = mobility_cycle[query_ordinal % len(mobility_cycle)]
      mobility_instruction = render_mobility_instruction(
          mobility, role_a=role_a, role_b=role_b, variant=query_ordinal % 2,
      )
      functional_edges.append({
          "edge_id": edge_id, "role_a": role_a, "role_b": role_b,
          "instruction": render_port_conditioned_instruction_v1(
              role_a=role_a, role_b=role_b, port_a=port_a, port_b=port_b,
              mobility_instruction=mobility_instruction,
          ),
          "port_contract_status": "specified",
          "port_contract_a": port_a, "port_contract_b": port_b,
          "requested_mobility": mobility,
      })
      target_edges.append({
          "edge_id": edge_id,
          "support_family": edge_spec["support_family"],
          "target_mobility": mobility,
          "endpoint_a": {
              "candidate_id": target_by_role[role_a],
              "primitive_orbit_ordinal": ordinal_a,
          },
          "endpoint_b": {
              "candidate_id": target_by_role[role_b],
              "primitive_orbit_ordinal": ordinal_b,
          },
      })
      for side, role_id, ordinal in (
          ("a", role_a, ordinal_a), ("b", role_b, ordinal_b),
      ):
        graph = target_graph_by_role[role_id]
        candidate = target_candidate_by_role[role_id]
        supervision_rows.append({
            "query_id": query_id, "edge_id": edge_id, "side": side,
            "status": "mapped_type_consistent",
            "candidate_id": candidate["candidate_id"],
            "step_sha256": candidate["step_sha256"],
            "primitive_orbit_ordinal": ordinal,
            "primitive_kind": graph.primitive_kinds[ordinal],
            "observed_geometry_type": graph.primitive_type(ordinal),
            "source_shape_type": graph.primitive_type(ordinal),
            "structural_orbit_ordinal": ordinal,
            "structural_orbit_count": int(graph.primitive_features.shape[0]),
        })
    public_queries.append({
        "query_id": query_id, "roles": public_roles,
        "candidate_sets": candidate_sets, "functional_edges": functional_edges,
    })
    private_targets.append({
        "query_id": query_id, "target_candidate_by_role": target_by_role,
        "functional_edge_targets": target_edges,
        "candidate_provenance_by_role": provenance_by_role,
    })

  overlap = sorted(all_step_sha256s & historical_step_sha256s)
  if overlap:
    raise ValueError("LinkCAD procedural STEP overlaps historical candidates")
  public = {
      "schema_version": PUBLIC_SCHEMA_VERSION,
      "scope": "prospective_independent_procedural_candidate_transfer",
      "query_count": len(public_queries),
      "contains_private_targets": False,
      "require_semantic_role_names": True,
      "selection_uses_model_or_geometry_outcomes": False,
      "port_contract_schema_version": "linkcad_port_contract.v1",
      "port_contract_contains_candidate_or_primitive_identity": False,
      "queries": public_queries,
  }
  private = {
      "schema_version": PRIVATE_SCHEMA_VERSION,
      "scope": "procedural_transfer_private_targets_sealed",
      "query_count": len(private_targets), "targets": private_targets,
  }
  supervision = {
      "schema_version": SUPERVISION_SCHEMA_VERSION,
      "scope": "procedural_constructive_primitive_orbit_supervision",
      "query_count": len(public_queries),
      "endpoint_count": len(supervision_rows),
      "fully_mapped_edge_count": len(supervision_rows) // 2,
      "status_counts": {"mapped_type_consistent": len(supervision_rows)},
      "rows": supervision_rows,
  }
  generator_manifest = {
      "schema_version": SCHEMA_VERSION,
      "source": "independent_cadquery_procedural_programs",
      "config": config.to_dict(),
      "schedule_sha256": canonical_sha256(schedule),
      "candidate_program_count": len(candidate_programs),
      "candidate_programs": candidate_programs,
      "unique_step_sha256_count": len(all_step_sha256s),
      "historical_step_sha256_count": len(historical_step_sha256s),
      "historical_step_overlap_count": 0,
      "target_schedule_uses_model_outputs": False,
      "candidate_generation_uses_fusion_geometry": False,
  }
  paths = {
      "public": artifact_root / "public.json",
      "private": artifact_root / "private_targets.sealed.json",
      "supervision": artifact_root / "primitive_supervision.json",
      "manifest": artifact_root / "generator_manifest.json",
      "schedule": artifact_root / "schedule.json",
  }
  _write_json(paths["public"], public)
  _write_json(paths["private"], private)
  _write_json(paths["supervision"], supervision)
  _write_json(paths["manifest"], generator_manifest)
  _write_json(paths["schedule"], schedule)
  precommit = {
      "schema_version": "linkcad_procedural_transfer_precommit.v1",
      "code_revision": code_revision,
      "config_sha256": canonical_sha256(config.to_dict()),
      "public_file_sha256": _file_sha256(paths["public"]),
      "private_target_file_sha256": _file_sha256(paths["private"]),
      "primitive_supervision_file_sha256": _file_sha256(paths["supervision"]),
      "generator_manifest_file_sha256": _file_sha256(paths["manifest"]),
      "checkpoint_sha256s": [_file_sha256(path) for path in checkpoint_paths],
      "prediction_files_exist_at_precommit": False,
      "private_targets_used_by_prediction_runner": False,
      "public_private_signing_used": False,
  }
  _write_json(artifact_root / "precommit.json", precommit)
  return {
      "query_count": len(public_queries),
      "edge_count": sum(len(row["functional_edges"]) for row in public_queries),
      "candidate_step_count": len(all_step_sha256s),
      "historical_step_overlap_count": 0,
      "public": public, "private": private,
      "supervision": supervision, "precommit": precommit,
  }


def historical_step_sha256s_v1(public_paths: tuple[Path, ...]) -> set[str]:
  result = set()
  for path in public_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for query in payload["queries"]:
      for candidates in query["candidate_sets"].values():
        result.update(str(row["step_sha256"]) for row in candidates)
  return result


__all__ = [
    "CANDIDATE_COUNT", "LARGE_CANDIDATE_COUNT",
    "ProceduralTransferConfigV1", "ROLE_COUNT",
    "SCHEMA_VERSION", "build_procedural_schedule_v1",
    "historical_step_sha256s_v1", "materialize_procedural_transfer_v1",
]
