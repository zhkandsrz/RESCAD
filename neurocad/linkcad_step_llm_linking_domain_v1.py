"""Materialize the oracle-covered learned-generator subset for LinkCAD."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import cadquery as cq

from .linkcad_language_contracts import render_mobility_instruction
from .linkcad_port_contract_v1 import (
    describe_ports_v1,
    render_port_conditioned_instruction_v1,
)
from .linkcad_primitive_graph_v2 import extract_primitive_graph_v2


PUBLIC_SCHEMA_VERSION = "linkcad_candidate_set_public.v1"
PRIVATE_SCHEMA_VERSION = "linkcad_candidate_set_private_targets.v1"
SUPERVISION_SCHEMA_VERSION = "linkcad_direct_primitive_orbit_supervision.v2"


def generator_identity_v1(precommit: Mapping[str, Any]) -> str:
  schema = str(precommit.get("schema_version", ""))
  if schema.startswith("linkcad_text_to_cadquery_generation_precommit."):
    return "text_to_cadquery"
  if schema.startswith("linkcad_step_llm_generation_precommit."):
    return "step_llm"
  raise ValueError("learned-generator precommit schema differs")


def _canonical_sha(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
  ).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  temporary.replace(path)


def select_linking_queries_v1(
    generation_receipt: Mapping[str, Any], *, min_ready_per_role: int = 2,
) -> dict[str, Any]:
  if min_ready_per_role < 2:
    raise ValueError("STEP-LLM linking candidate minimum differs")
  selected = []
  exclusions = []
  for row in generation_receipt["query_oracles"]:
    query_id = str(row["query_id"])
    if row["intended_candidate_oracle_ready"] is not True:
      exclusions.append({
          "query_id": query_id,
          "reason": "intended_candidate_oracle_absent",
      })
      continue
    counts = row["candidate_ready_by_role"]
    if any(int(counts.get(family, 0)) < min_ready_per_role
           for family in ("ring", "shaft")):
      exclusions.append({
          "query_id": query_id,
          "reason": "candidate_pool_too_small",
      })
      continue
    selected.append(query_id)
  return {
      "selected_query_ids": sorted(selected),
      "excluded_queries": exclusions,
      "exclusion_counts": dict(sorted(Counter(
          row["reason"] for row in exclusions
      ).items())),
  }


def _posed_shape(shape: cq.Shape, candidate_key: str) -> tuple[cq.Shape, list[float]]:
  rng = random.Random(int(hashlib.sha256(
      candidate_key.encode("utf-8")
  ).hexdigest()[:16], 16))
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


def _cylinder_port_ordinal(graph, *, prefer_smallest: bool) -> int:
  contracts = describe_ports_v1(graph)
  candidates = [
      ordinal for ordinal, contract in enumerate(contracts)
      if contract["primitive_kind"] == "face"
      and contract["geometry_type"] == "cylinder"
  ]
  if not candidates:
    raise ValueError("STEP-LLM target cylinder port is absent")
  preferred_sizes = {"smallest", "small", "only"}
  preferred = [
      ordinal for ordinal in candidates
      if contracts[ordinal]["size_class"] in preferred_sizes
  ]
  if prefer_smallest and preferred:
    return preferred[0]
  return candidates[0]


def materialize_step_llm_linking_domain_v1(
    *, generation_root: Path, dataset_root: Path, artifact_root: Path,
    historical_step_sha256s: set[str], min_ready_per_role: int = 2,
) -> dict[str, Any]:
  generation_root = generation_root.resolve()
  dataset_root = dataset_root.resolve()
  artifact_root = artifact_root.resolve()
  output_names = (
      "public.json", "private_targets.sealed.json", "primitive_supervision.json",
      "generator_funnel.json", "materialization_manifest.json",
  )
  if any((artifact_root / name).exists() for name in output_names):
    raise ValueError("STEP-LLM linking domain already exists")
  if dataset_root.exists() and any(dataset_root.iterdir()):
    raise ValueError("STEP-LLM posed candidate root is not empty")
  receipt_path = generation_root / "generation_receipt.json"
  schedule_path = generation_root / "generation_schedule.sealed.json"
  precommit_path = generation_root / "generation_precommit.json"
  if not receipt_path.exists():
    raise ValueError("STEP-LLM generation is not terminal")
  receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
  schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
  precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
  generator_kind = generator_identity_v1(precommit)
  if (
      receipt.get("private_linking_targets_opened") is not False
      or receipt["schedule_sha256"] != precommit["schedule_sha256"]
      or _file_sha(schedule_path) != precommit["schedule_sha256"]
  ):
    raise ValueError("STEP-LLM generation authority differs")
  calibrate_interface_scale = bool(
      precommit.get("query_oracle_uses_interface_scale_calibration", False)
  )
  scale_min, scale_max = precommit.get(
      "interface_scale_factor_range", [1.0, 1.0]
  )
  selection = select_linking_queries_v1(
      receipt, min_ready_per_role=min_ready_per_role,
  )
  selected_ids = set(selection["selected_query_ids"])
  schedule_by_id = {
      str(row["query_id"]): row for row in schedule["queries"]
  }
  generated_by_query = defaultdict(list)
  for row in receipt["rows"]:
    generated_by_query[str(row["query_id"])].append(row)
  dataset_root.mkdir(parents=True, exist_ok=True)
  artifact_root.mkdir(parents=True, exist_ok=True)
  public_queries = []
  private_targets = []
  supervision_rows = []
  provenance_rows = []
  all_step_sha256s = set()
  for query_id in sorted(selected_ids):
    schedule_query = schedule_by_id[query_id]
    query_rows = generated_by_query[query_id]
    role_rows = {str(role["role_id"]): role for role in schedule_query["roles"]}
    public_roles = []
    candidate_sets = {}
    graph_by_key = {}
    public_by_key = {}
    target_key_by_role = {}
    for role_id, schedule_role in role_rows.items():
      family = str(schedule_role["family"])
      public_roles.append({
          "role_id": role_id,
          "description": str(schedule_role["description"]),
      })
      ready = sorted((
          row for row in query_rows
          if row["role_id"] == role_id
          and row["terminal_status"] == "candidate_ready"
      ), key=lambda row: int(row["candidate_ordinal"]))
      candidates = []
      for row in ready:
        raw_path = generation_root / row["raw_step_path"]
        solid = cq.importers.importStep(str(raw_path)).solids().vals()[0]
        scale_factor = 1.0
        if calibrate_interface_scale:
          scale_factor = (
              float(row["requested_diameter_mm"])
              / float(row["actual_interface_diameter_mm"])
          )
          if not float(scale_min) <= scale_factor <= float(scale_max):
            raise ValueError("STEP-LLM interface scale factor differs")
          solid = solid.scale(scale_factor)
        posed, perturbation = _posed_shape(solid, str(row["candidate_key"]))
        relative = Path(
            f"query_{int(row['query_ordinal']):04d}/{role_id}/"
            f"candidate_{int(row['candidate_ordinal']):02d}.step"
        )
        output = dataset_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(posed, str(output), exportType="STEP")
        step_sha = _file_sha(output)
        if step_sha in all_step_sha256s:
          raise ValueError("STEP-LLM posed candidate identity is duplicated")
        all_step_sha256s.add(step_sha)
        candidate_id = "stepllm_part_" + step_sha[:20]
        candidate = {
            "candidate_id": candidate_id,
            "step_path": relative.as_posix(),
            "step_sha256": step_sha,
            "volume": float(posed.Volume()),
            "area": float(posed.Area()),
        }
        candidates.append(candidate)
        graph_by_key[row["candidate_key"]] = extract_primitive_graph_v2(posed)
        public_by_key[row["candidate_key"]] = candidate
        if row["is_target"]:
          target_key_by_role[role_id] = row["candidate_key"]
        provenance_rows.append({
            "query_id": query_id,
            "role_id": role_id,
            "candidate_key": row["candidate_key"],
            "candidate_id": candidate_id,
            "family": family,
            "requested_diameter_mm": row["requested_diameter_mm"],
            "actual_interface_diameter_mm": row["actual_interface_diameter_mm"],
            "interface_scale_factor": scale_factor,
            "calibrated_interface_diameter_mm": (
                float(row["requested_diameter_mm"])
                if calibrate_interface_scale
                else float(row["actual_interface_diameter_mm"])
            ),
            "raw_step_sha256": row["raw_step_sha256"],
            "posed_step_sha256": step_sha,
            "se3_angles_deg_then_translation_mm": perturbation,
            **(
                {
                    "style_id": str(row["style_id"]),
                    "style_description": str(row["style_description"]),
                }
                if "style_id" in row and "style_description" in row else {}
            ),
        })
      if len(candidates) < min_ready_per_role or role_id not in target_key_by_role:
        raise ValueError("STEP-LLM selected query candidate closure differs")
      candidate_sets[role_id] = candidates
    ring_role, shaft_role = "role_00", "role_01"
    ring_key = target_key_by_role[ring_role]
    shaft_key = target_key_by_role[shaft_role]
    ring_ordinal = _cylinder_port_ordinal(
        graph_by_key[ring_key], prefer_smallest=True,
    )
    shaft_ordinal = _cylinder_port_ordinal(
        graph_by_key[shaft_key], prefer_smallest=True,
    )
    ring_port = describe_ports_v1(graph_by_key[ring_key])[ring_ordinal]
    shaft_port = describe_ports_v1(graph_by_key[shaft_key])[shaft_ordinal]
    mobility_instruction = render_mobility_instruction(
        "revolute", role_a=ring_role, role_b=shaft_role,
        variant=int(schedule_query["query_ordinal"]) % 2,
    )
    edge_id = "edge_00"
    public_edge = {
        "edge_id": edge_id,
        "role_a": ring_role,
        "role_b": shaft_role,
        "instruction": render_port_conditioned_instruction_v1(
            role_a=ring_role, role_b=shaft_role,
            port_a=ring_port, port_b=shaft_port,
            mobility_instruction=mobility_instruction,
        ),
        "port_contract_status": "specified",
        "port_contract_a": ring_port,
        "port_contract_b": shaft_port,
        "requested_mobility": "revolute",
    }
    public_queries.append({
        "query_id": query_id,
        "roles": public_roles,
        "candidate_sets": candidate_sets,
        "functional_edges": [public_edge],
    })
    target_candidate_by_role = {
        ring_role: public_by_key[ring_key]["candidate_id"],
        shaft_role: public_by_key[shaft_key]["candidate_id"],
    }
    target_edge = {
        "edge_id": edge_id,
        "support_family": "axial_support",
        "target_mobility": "revolute",
        "endpoint_a": {
            "candidate_id": target_candidate_by_role[ring_role],
            "primitive_orbit_ordinal": ring_ordinal,
        },
        "endpoint_b": {
            "candidate_id": target_candidate_by_role[shaft_role],
            "primitive_orbit_ordinal": shaft_ordinal,
        },
    }
    private_targets.append({
        "query_id": query_id,
        "target_candidate_by_role": target_candidate_by_role,
        "functional_edge_targets": [target_edge],
    })
    for side, role_id, key, ordinal in (
        ("a", ring_role, ring_key, ring_ordinal),
        ("b", shaft_role, shaft_key, shaft_ordinal),
    ):
      graph = graph_by_key[key]
      candidate = public_by_key[key]
      supervision_rows.append({
          "query_id": query_id,
          "edge_id": edge_id,
          "side": side,
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
  overlap = sorted(all_step_sha256s & historical_step_sha256s)
  if overlap:
    raise ValueError("STEP-LLM posed STEP overlaps historical candidates")
  public = {
      "schema_version": PUBLIC_SCHEMA_VERSION,
      "scope": f"prospective_learned_{generator_kind}_candidate_transfer",
      "generator_kind": generator_kind,
      "query_count": len(public_queries),
      "contains_private_targets": False,
      "require_semantic_role_names": True,
      "selection_uses_model_or_geometry_outcomes": True,
      "selection_conditions_on_generator_oracle": True,
      "selection_uses_linking_model_outputs": False,
      "interface_scale_calibration_applied": calibrate_interface_scale,
      "private_candidate_identity_withheld_from_prediction_methods": True,
      "port_contract_schema_version": "linkcad_port_contract.v1",
      "port_contract_contains_candidate_or_primitive_identity": False,
      "queries": public_queries,
  }
  private = {
      "schema_version": PRIVATE_SCHEMA_VERSION,
      "scope": f"{generator_kind}_transfer_private_targets_sealed",
      "generator_kind": generator_kind,
      "query_count": len(private_targets),
      "targets": private_targets,
  }
  supervision = {
      "schema_version": SUPERVISION_SCHEMA_VERSION,
      "scope": f"{generator_kind}_constructive_primitive_orbit_supervision",
      "generator_kind": generator_kind,
      "query_count": len(public_queries),
      "endpoint_count": len(supervision_rows),
      "fully_mapped_edge_count": len(supervision_rows) // 2,
      "status_counts": {"mapped_type_consistent": len(supervision_rows)},
      "rows": supervision_rows,
  }
  funnel = {
      "schema_version": "linkcad_learned_generator_transfer_funnel.v1",
      "scope": f"{generator_kind}_full_generation_and_linking_funnel",
      "generator_kind": generator_kind,
      "generation_receipt_sha256": _file_sha(receipt_path),
      "attempted_query_count": int(receipt["query_count"]),
      "attempted_candidate_count": int(receipt["candidate_attempt_count"]),
      "generation_status_counts": receipt["status_counts"],
      "intended_candidate_oracle_query_count": int(
          receipt["intended_candidate_oracle_query_count"]
      ),
      "raw_compatible_pair_query_count": int(
          receipt.get("raw_compatible_pair_query_count", 0)
      ),
      "calibrated_compatible_pair_query_count": int(
          receipt.get("calibrated_compatible_pair_query_count", 0)
      ),
      "linking_evaluable_query_count": len(public_queries),
      **selection,
  }
  manifest = {
      "schema_version": "linkcad_learned_generator_materialization.v1",
      "generator_kind": generator_kind,
      "materializer_source_sha256": _file_sha(Path(__file__).resolve()),
      "generation_precommit_sha256": _file_sha(precommit_path),
      "generation_receipt_sha256": _file_sha(receipt_path),
      "public_sha256": _canonical_sha(public),
      "private_sha256": _canonical_sha(private),
      "supervision_sha256": _canonical_sha(supervision),
      "dataset_root": dataset_root.as_posix(),
      "candidate_count": len(provenance_rows),
      "unique_step_sha256_count": len(all_step_sha256s),
      "historical_step_overlap_count": 0,
      "min_ready_per_role": min_ready_per_role,
      "interface_scale_calibration_applied": calibrate_interface_scale,
      "interface_scale_factor_range": [scale_min, scale_max],
      "private_targets_opened": False,
      "private_candidate_schedule_used_only_for_oracle_filtering": True,
      "linking_predictions_exist_at_materialization": False,
      "candidate_provenance": provenance_rows,
  }
  for name, payload in (
      ("public.json", public),
      ("private_targets.sealed.json", private),
      ("primitive_supervision.json", supervision),
      ("generator_funnel.json", funnel),
      ("materialization_manifest.json", manifest),
  ):
    _write(artifact_root / name, payload)
  return {
      "query_count": len(public_queries),
      "candidate_count": len(provenance_rows),
      "historical_step_overlap_count": 0,
      "selection": selection,
  }


__all__ = [
    "generator_identity_v1", "materialize_step_llm_linking_domain_v1",
    "select_linking_queries_v1",
]
