from __future__ import annotations

import cadquery as cq

from neurocad.linkcad_port_contract_v1 import (
    describe_port_v1,
    describe_ports_v1,
    render_port_conditioned_instruction_v1,
)
from neurocad.linkcad_primitive_graph_v2 import extract_primitive_graph_v2


def test_port_contract_exposes_intrinsic_description_without_identity():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  contract = describe_port_v1(graph, 0)
  assert set(contract) == {
      "schema_version", "primitive_kind", "geometry_type", "size_class",
      "adjacent_surface_types", "symmetry_class",
  }
  assert not any(
      token in str(contract).lower()
      for token in ("candidate_id", "ordinal", "source_joint", "occurrence")
  )


def test_port_instruction_keeps_port_fixed_while_mobility_can_change():
  graph = extract_primitive_graph_v2(cq.Workplane("XY").box(2, 3, 4).val())
  port = describe_port_v1(graph, 0)
  fixed = render_port_conditioned_instruction_v1(
      role_a="shaft", role_b="housing", port_a=port, port_b=port,
      mobility_instruction="Keep the parts fixed.",
  )
  sliding = render_port_conditioned_instruction_v1(
      role_a="shaft", role_b="housing", port_a=port, port_b=port,
      mobility_instruction="Allow sliding along their shared axis.",
  )
  assert fixed.split(". ")[0] == sliding.split(". ")[0]
  assert fixed != sliding


def test_batch_port_descriptions_match_single_primitive_contracts():
  graph = extract_primitive_graph_v2(
      cq.Workplane("XY").box(2, 3, 4).faces(">Z").hole(0.7).val()
  )

  batch = describe_ports_v1(graph)

  assert batch == tuple(
      describe_port_v1(graph, ordinal)
      for ordinal in range(int(graph.primitive_features.shape[0]))
  )
