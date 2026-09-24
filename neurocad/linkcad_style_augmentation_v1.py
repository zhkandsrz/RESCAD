"""Development-only language/style augmentation for generated candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .linkcad_language_contracts import render_mobility_instruction
from .linkcad_port_contract_v1 import (
    describe_ports_v1,
    render_port_conditioned_instruction_v1,
)
from .linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2
from .linkcad_step_llm_linking_domain_v1 import _cylinder_port_ordinal


SCHEMA_VERSION = "linkcad_generated_style_augmentation.v1"


def _sha(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
  ).encode("utf-8")).hexdigest()


def materialize_style_augmentation_v1(
    *, public: Mapping[str, Any], manifest: Mapping[str, Any],
    cache: LinkCADPrimitiveGraphCacheV2,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
  """Enumerate target-style combinations without creating new candidate bytes."""

  if (
      public.get("contains_private_targets") is not False
      or public.get("generator_kind") != "text_to_cadquery"
      or manifest.get("generator_kind") != "text_to_cadquery"
      or manifest.get("private_targets_opened") is not False
  ):
    raise ValueError("LinkCAD style augmentation source differs")
  provenance = manifest.get("candidate_provenance")
  if not isinstance(provenance, list) or not provenance:
    raise ValueError("LinkCAD style augmentation provenance differs")
  by_candidate = {
      str(row["candidate_id"]): row for row in provenance
      if row.get("style_id") and row.get("style_description")
  }
  if len(by_candidate) != len(provenance):
    raise ValueError("LinkCAD style augmentation labels differ")
  public_rows = []
  private_rows = []
  supervision_rows = []
  source_counts = {}
  for base in public["queries"]:
    base_query_id = str(base["query_id"])
    role_ids = [str(row["role_id"]) for row in base["roles"]]
    if role_ids != ["role_00", "role_01"] or len(base["functional_edges"]) != 1:
      raise ValueError("LinkCAD style augmentation graph differs")
    candidates_by_role: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for role_id in role_ids:
      rows = []
      for candidate in base["candidate_sets"][role_id]:
        metadata = by_candidate.get(str(candidate["candidate_id"]))
        if metadata is None or str(metadata["role_id"]) != role_id:
          raise ValueError("LinkCAD style augmentation candidate differs")
        rows.append((candidate, metadata))
      style_ids = [str(metadata["style_id"]) for _, metadata in rows]
      if len(rows) < 6 or len(style_ids) != len(set(style_ids)):
        raise ValueError("LinkCAD style augmentation style coverage differs")
      candidates_by_role[role_id] = rows
    diameters = {
        float(metadata["requested_diameter_mm"])
        for rows in candidates_by_role.values() for _, metadata in rows
    }
    if len(diameters) != 1:
      raise ValueError("LinkCAD style augmentation interface scale differs")
    diameter = next(iter(diameters))
    created = 0
    for ring_candidate, ring_meta in candidates_by_role["role_00"]:
      for shaft_candidate, shaft_meta in candidates_by_role["role_01"]:
        target_rows = {
            "role_00": (ring_candidate, ring_meta),
            "role_01": (shaft_candidate, shaft_meta),
        }
        target_payload = {
            "base_query_id": base_query_id,
            "ring_style_id": str(ring_meta["style_id"]),
            "shaft_style_id": str(shaft_meta["style_id"]),
        }
        query_id = "linkcad_styleaug_" + _sha(target_payload)[:24]
        target_ports = {}
        target_ordinals = {}
        for role_id, (candidate, _metadata) in target_rows.items():
          graph = cache.graph(str(candidate["step_sha256"]))
          if graph is None:
            raise ValueError("LinkCAD style augmentation primitive graph differs")
          ordinal = _cylinder_port_ordinal(graph, prefer_smallest=True)
          target_ordinals[role_id] = ordinal
          target_ports[role_id] = describe_ports_v1(graph)[ordinal]
        mobility_instruction = render_mobility_instruction(
            "revolute", role_a="role_00", role_b="role_01",
            variant=created % 2,
        )
        edge_id = "edge_00"
        public_rows.append({
            "query_id": query_id,
            "source_query_id": base_query_id,
            "roles": [
                {
                    "role_id": role_id,
                    "description": (
                        f"{target_rows[role_id][1]['style_description']} with a "
                        f"{diameter:g} mm "
                        f"{'bore' if role_id == 'role_00' else 'bearing diameter'}"
                    ),
                }
                for role_id in role_ids
            ],
            "candidate_sets": base["candidate_sets"],
            "functional_edges": [{
                "edge_id": edge_id,
                "role_a": "role_00",
                "role_b": "role_01",
                "instruction": render_port_conditioned_instruction_v1(
                    role_a="role_00", role_b="role_01",
                    port_a=target_ports["role_00"],
                    port_b=target_ports["role_01"],
                    mobility_instruction=mobility_instruction,
                ),
                "port_contract_status": "specified",
                "port_contract_a": target_ports["role_00"],
                "port_contract_b": target_ports["role_01"],
                "requested_mobility": "revolute",
            }],
        })
        target_candidate_by_role = {
            role_id: str(target_rows[role_id][0]["candidate_id"])
            for role_id in role_ids
        }
        private_rows.append({
            "query_id": query_id,
            "source_query_id": base_query_id,
            "target_candidate_by_role": target_candidate_by_role,
            "functional_edge_targets": [{
                "edge_id": edge_id,
                "support_family": "axial_support",
                "target_mobility": "revolute",
                "endpoint_a": {
                    "candidate_id": target_candidate_by_role["role_00"],
                    "primitive_orbit_ordinal": target_ordinals["role_00"],
                },
                "endpoint_b": {
                    "candidate_id": target_candidate_by_role["role_01"],
                    "primitive_orbit_ordinal": target_ordinals["role_01"],
                },
            }],
        })
        for side, role_id in (("a", "role_00"), ("b", "role_01")):
          candidate = target_rows[role_id][0]
          graph = cache.graph(str(candidate["step_sha256"]))
          ordinal = target_ordinals[role_id]
          supervision_rows.append({
              "query_id": query_id,
              "edge_id": edge_id,
              "side": side,
              "status": "mapped_type_consistent",
              "candidate_id": str(candidate["candidate_id"]),
              "step_sha256": str(candidate["step_sha256"]),
              "primitive_orbit_ordinal": ordinal,
              "primitive_kind": graph.primitive_kinds[ordinal],
              "observed_geometry_type": graph.primitive_type(ordinal),
              "source_shape_type": graph.primitive_type(ordinal),
              "structural_orbit_ordinal": ordinal,
              "structural_orbit_count": int(graph.primitive_features.shape[0]),
          })
        created += 1
    source_counts[base_query_id] = created
  augmented_public = {
      "schema_version": "linkcad_candidate_set_public.v1",
      "scope": "development_generated_style_augmentation",
      "generator_kind": "text_to_cadquery",
      "query_count": len(public_rows),
      "contains_private_targets": False,
      "require_semantic_role_names": True,
      "selection_uses_model_or_geometry_outcomes": False,
      "selection_conditions_on_generator_oracle": True,
      "selection_uses_linking_model_outputs": False,
      "interface_scale_calibration_applied": True,
      "private_candidate_identity_withheld_from_prediction_methods": True,
      "port_contract_schema_version": "linkcad_port_contract.v1",
      "port_contract_contains_candidate_or_primitive_identity": False,
      "queries": public_rows,
  }
  augmented_private = {
      "schema_version": "linkcad_candidate_set_private_targets.v1",
      "scope": "development_generated_style_augmentation_targets",
      "generator_kind": "text_to_cadquery",
      "query_count": len(private_rows),
      "targets": private_rows,
  }
  supervision = {
      "schema_version": "linkcad_direct_primitive_orbit_supervision.v2",
      "scope": "development_generated_style_augmentation_supervision",
      "generator_kind": "text_to_cadquery",
      "query_count": len(public_rows),
      "endpoint_count": len(supervision_rows),
      "fully_mapped_edge_count": len(supervision_rows) // 2,
      "status_counts": {"mapped_type_consistent": len(supervision_rows)},
      "rows": supervision_rows,
  }
  augmentation_manifest = {
      "schema_version": SCHEMA_VERSION,
      "scope": "development_only_source_adaptation_training",
      "source_query_count": len(public["queries"]),
      "augmented_query_count": len(public_rows),
      "augmented_queries_per_source": source_counts,
      "candidate_bytes_reused_only_within_development_training": True,
      "eligible_as_independent_confirmation": False,
      "uses_linking_model_outputs": False,
  }
  return augmented_public, augmented_private, supervision, augmentation_manifest


def write_style_augmentation_v1(
    *, domain_root: Path, primitive_cache_root: Path, output_root: Path,
) -> dict[str, Any]:
  if output_root.exists() and any(output_root.iterdir()):
    raise ValueError("LinkCAD style augmentation output exists")
  public = json.loads((domain_root / "public.json").read_text(encoding="utf-8"))
  manifest = json.loads(
      (domain_root / "materialization_manifest.json").read_text(encoding="utf-8")
  )
  cache = LinkCADPrimitiveGraphCacheV2(primitive_cache_root)
  cache.load_all()
  payloads = materialize_style_augmentation_v1(
      public=public, manifest=manifest, cache=cache,
  )
  output_root.mkdir(parents=True, exist_ok=True)
  names = (
      "public.json", "private_targets.sealed.json",
      "primitive_supervision.json", "augmentation_manifest.json",
  )
  for name, payload in zip(names, payloads, strict=True):
    temporary = (output_root / name).with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_root / name)
  return payloads[-1]


__all__ = [
    "SCHEMA_VERSION", "materialize_style_augmentation_v1",
    "write_style_augmentation_v1",
]
