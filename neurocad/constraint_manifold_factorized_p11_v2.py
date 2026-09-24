"""Factorized, non-oracle P11 candidate-domain commitments.

V2 commits the structural Cartesian product before any overlap or body metric is
evaluated.  The exhaustive-row adapter exists only for equivalence tests and
development migration; production builders construct the same factors directly.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .constraint_manifold_solver_v4 import canonical_sha256


P11_FACTORIZED_DOMAIN_SCHEMA_V2 = "constraint_manifold_p11_factorized_domain.v2"
P11_FACTORIZED_EXPANSION_SCHEMA_V2 = (
    "sign_join_canonical_yaw_x_axial_offset_candidate_identity.v2"
)
P11_FACTORIZED_STRUCTURAL_TOPK_SCHEMA_V2 = (
    "constraint_manifold_p11_factorized_structural_topk.v2"
)
P11_FACTORIZED_MEMBERSHIP_SCHEMA_V2 = (
    "constraint_manifold_p11_factorized_membership.v2"
)
P11_FACTORIZED_COVERAGE_SCHEMA_V2 = (
    "constraint_manifold_p11_factorized_diversity_coverage.v2"
)
P11_FACTORIZED_TOP_K_V2 = 32
P11_IDENTITY_ARITHMETIC_SCHEMA_V1 = (
    "fixed_scalar_pairwise4_se3_matmul_ieee754_binary64.v1"
)
_SIGN_SCHEMA = "constraint_manifold_p11_sign_factor.v2"
_YAW_SCHEMA = "constraint_manifold_p11_canonical_yaw_factor.v2"
_OFFSET_SCHEMA = "constraint_manifold_p11_axial_offset_factor.v2"
_BASELINE_OFFSET_FAMILIES = (
    "interval_center", "lower_endpoint", "upper_endpoint", "opposed_endpoint",
)
_POINT_YAW_EVENT_KINDS = frozenset((
    "trim_vertices_local_2d", "trim_edge_midpoints_local_2d",
    "loop_centroids_local_2d", "adjacent_shared_edge_vertices_local_2d",
    "adjacent_shared_edge_midpoints_local_2d",
))
_OFFSET_EVENT_KINDS = frozenset((*_BASELINE_OFFSET_FAMILIES,
    "trim_vertices_local_2d", "mesh_vertices_local_2d",
    "trim_edge_midpoints_local_2d", "mesh_edge_midpoints_local_2d",
    "triangle_centroids_local_2d", "loop_centroids_local_2d",
    "adjacent_shared_edge_vertices_local_2d",
    "adjacent_shared_edge_midpoints_local_2d",
))


def _copy(value: Any) -> Any:
  return json.loads(json.dumps(value, allow_nan=False))


def _matrix(value: Any, *, label: str) -> np.ndarray:
  matrix = np.asarray(value, dtype=float)
  if matrix.shape != (16,) or not np.isfinite(matrix).all():
    raise ValueError(f"P11 factor {label} differs")
  matrix = matrix.reshape(4, 4)
  if not np.array_equal(matrix[3], (0.0, 0.0, 0.0, 1.0)):
    raise ValueError(f"P11 factor {label} is not homogeneous")
  rotation = matrix[:3, :3]
  for first in range(3):
    for second in range(3):
      dot = (
          float(rotation[0, first]) * float(rotation[0, second])
          + float(rotation[1, first]) * float(rotation[1, second])
          + float(rotation[2, first]) * float(rotation[2, second])
      )
      expected = 1.0 if first == second else 0.0
      if not math.isclose(dot, expected, abs_tol=1e-8, rel_tol=0.0):
        raise ValueError(f"P11 factor {label} is not SE(3)")
  determinant = (
      float(rotation[0, 0]) * (
          float(rotation[1, 1]) * float(rotation[2, 2])
          - float(rotation[1, 2]) * float(rotation[2, 1])
      )
      - float(rotation[0, 1]) * (
          float(rotation[1, 0]) * float(rotation[2, 2])
          - float(rotation[1, 2]) * float(rotation[2, 0])
      )
      + float(rotation[0, 2]) * (
          float(rotation[1, 0]) * float(rotation[2, 1])
          - float(rotation[1, 1]) * float(rotation[2, 0])
      )
  )
  if (
      not math.isclose(determinant, 1.0, abs_tol=1e-8, rel_tol=0.0)
  ):
    raise ValueError(f"P11 factor {label} is not SE(3)")
  return matrix


def multiply_se3_row_major_v2(left: Any, right: Any) -> np.ndarray:
  """Multiply two SE(3) matrices with fixed scalar arithmetic and no BLAS."""

  first = _matrix(
      np.asarray(left, dtype=float).reshape(-1),
      label="left multiplication transform",
  )
  second = _matrix(
      np.asarray(right, dtype=float).reshape(-1),
      label="right multiplication transform",
  )
  rows: list[list[float]] = []
  for row in range(4):
    values = []
    for column in range(4):
      first_pair = (
          float(first[row, 0]) * float(second[0, column])
          + float(first[row, 1]) * float(second[1, column])
      )
      second_pair = (
          float(first[row, 2]) * float(second[2, column])
          + float(first[row, 3]) * float(second[3, column])
      )
      # NumPy's fixed-size dot reduction groups four products pairwise.  Keep
      # that association so historical candidate identity bytes are unchanged.
      value = first_pair + second_pair
      values.append(value)
    rows.append(values)
  return _matrix(
      np.asarray(rows, dtype=float).reshape(-1), label="multiplied transform",
  )


def _real(value: Any, *, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"P11 factor {label} scalar differs")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"P11 factor {label} scalar differs")
  return result


def _point2(value: Any, *, label: str) -> tuple[float, float]:
  if not isinstance(value, list) or len(value) != 2:
    raise ValueError(f"P11 factor {label} point differs")
  return (_real(value[0], label=label), _real(value[1], label=label))


def _events(
    value: Any, *, label: str, allowed: frozenset[str], yaw: bool,
) -> list[Mapping[str, Any]]:
  if not isinstance(value, list) or not value:
    raise ValueError(f"P11 factor {label} events differ")
  for event in value:
    if not isinstance(event, Mapping) or set(event) != {"kind", "a", "b"}:
      raise ValueError(f"P11 factor {label} event schema differs")
    kind = event["kind"]
    if not isinstance(kind, str) or kind not in allowed:
      raise ValueError(f"P11 factor {label} event kind differs")
    if yaw and kind == "trim_angular_midpoint":
      _real(event["a"], label=f"{label} event a")
      _real(event["b"], label=f"{label} event b")
    else:
      _point2(event["a"], label=f"{label} event a")
      _point2(event["b"], label=f"{label} event b")
  if value != sorted(value, key=canonical_sha256):
    raise ValueError(f"P11 factor {label} event order differs")
  if len({canonical_sha256(event) for event in value}) != len(value):
    raise ValueError(f"P11 factor {label} event identity is duplicated")
  return value


def _validate_sign_factor(unsigned: Mapping[str, Any]) -> None:
  if set(unsigned) != {"schema_version", "normal_sign"}:
    raise ValueError("P11 factor sign exact schema differs")
  if type(unsigned["normal_sign"]) is not int or unsigned["normal_sign"] not in {-1, 1}:
    raise ValueError("P11 factor sign value differs")


def _validate_yaw_factor(unsigned: Mapping[str, Any]) -> None:
  if set(unsigned) != {
      "schema_version", "normal_sign", "yaw_radians", "yaw_events",
      "base_world_delta_row_major", "child_world_input_row_major",
  }:
    raise ValueError("P11 factor yaw exact schema differs")
  if type(unsigned["normal_sign"]) is not int or unsigned["normal_sign"] not in {-1, 1}:
    raise ValueError("P11 factor yaw sign differs")
  yaw = _real(unsigned["yaw_radians"], label="yaw")
  if not 0.0 <= yaw < 2.0 * math.pi:
    raise ValueError("P11 factor yaw is not canonical")
  _events(
      unsigned["yaw_events"], label="yaw", yaw=True,
      allowed=frozenset(("trim_angular_midpoint", *_POINT_YAW_EVENT_KINDS)),
  )
  _matrix(unsigned["base_world_delta_row_major"], label="base world transform")
  _matrix(unsigned["child_world_input_row_major"], label="child input transform")


def _validate_offset_factor(unsigned: Mapping[str, Any]) -> None:
  if set(unsigned) != {
      "schema_version", "normal_sign", "axial_offset_mm",
      "translation_delta_world_mm", "a_landmark", "b_landmark",
      "alignment_events", "offset_families", "fixed_displacement_mm",
  }:
    raise ValueError("P11 factor offset exact schema differs")
  if type(unsigned["normal_sign"]) is not int or unsigned["normal_sign"] not in {-1, 1}:
    raise ValueError("P11 factor offset sign differs")
  _real(unsigned["axial_offset_mm"], label="axial offset")
  translation = unsigned["translation_delta_world_mm"]
  if not isinstance(translation, list) or len(translation) != 3:
    raise ValueError("P11 factor offset translation differs")
  for value in translation:
    _real(value, label="offset translation")
  a = _point2(unsigned["a_landmark"], label="a landmark")
  b = _point2(unsigned["b_landmark"], label="b landmark")
  events = _events(
      unsigned["alignment_events"], label="offset", yaw=False,
      allowed=_OFFSET_EVENT_KINDS,
  )
  families = unsigned["offset_families"]
  expected_families = sorted({str(event["kind"]) for event in events})
  if (
      not isinstance(families, list) or families != expected_families
      or any(family not in _OFFSET_EVENT_KINDS for family in families)
  ):
    raise ValueError("P11 factor offset family differs")
  representative = sorted(events, key=canonical_sha256)[0]
  if list(a) != representative["a"] or list(b) != representative["b"]:
    raise ValueError("P11 factor offset representative differs")
  displacement = _real(unsigned["fixed_displacement_mm"], label="fixed displacement")
  if displacement < 0.0:
    raise ValueError("P11 factor fixed displacement differs")


def _factor(unsigned: Mapping[str, Any]) -> dict[str, Any]:
  payload = _copy(unsigned)
  return {**payload, "factor_payload_sha256": canonical_sha256(payload)}


def _factor_unsigned(payload: Mapping[str, Any]) -> dict[str, Any]:
  unsigned = dict(payload)
  observed = unsigned.pop("factor_payload_sha256", None)
  if observed != canonical_sha256(unsigned):
    raise ValueError("P11 factor commitment differs")
  return unsigned


def _identity(
    yaw_factor: Mapping[str, Any], offset_factor: Mapping[str, Any],
) -> dict[str, Any]:
  world = _matrix(
      yaw_factor["base_world_delta_row_major"], label="base world transform",
  ).copy()
  translation = np.asarray(offset_factor["translation_delta_world_mm"], dtype=float)
  if translation.shape != (3,) or not np.isfinite(translation).all():
    raise ValueError("P11 factor offset translation differs")
  world[:3, 3] += translation
  child_input = _matrix(
      yaw_factor["child_world_input_row_major"], label="child input transform",
  )
  # Match worker materialization operation-for-operation.  Adding translation
  # to a precomputed ``base @ child`` changes floating-point association and
  # can alter all three translation entries by one ULP for real assembly poses.
  child = multiply_se3_row_major_v2(world, child_input)
  unsigned = {
      "schema_version": "constraint_manifold_candidate_identity.v4",
      "program_index": 11,
      "world_delta_row_major": world.reshape(-1).tolist(),
      "child_world_row_major": child.reshape(-1).tolist(),
  }
  return {
      **unsigned, "candidate_key": canonical_sha256(unsigned),
      "normal_sign": int(yaw_factor["normal_sign"]),
      "yaw_factor_sha256": yaw_factor["factor_payload_sha256"],
      "axial_offset_factor_sha256": offset_factor["factor_payload_sha256"],
  }


def _domain_commitment_payload(domain: Mapping[str, Any]) -> dict[str, Any]:
  return {
      "schema_version": P11_FACTORIZED_DOMAIN_SCHEMA_V2,
      "expansion_schema": P11_FACTORIZED_EXPANSION_SCHEMA_V2,
      "candidate_identity_schema": "constraint_manifold_candidate_identity.v4",
      "identity_arithmetic_schema": P11_IDENTITY_ARITHMETIC_SCHEMA_V1,
      "sign_factors": domain["sign_factors"],
      "yaw_factors": domain["yaw_factors"],
      "axial_offset_factors": domain["axial_offset_factors"],
  }


def canonical_yaw_v2(value: Any) -> float:
  yaw = float(value)
  if not math.isfinite(yaw):
    raise ValueError("P11 factor yaw differs")
  if math.isclose(yaw, 2.0 * math.pi, abs_tol=5e-13, rel_tol=0.0):
    return 0.0
  yaw %= 2.0 * math.pi
  if math.isclose(yaw, 2.0 * math.pi, abs_tol=5e-13, rel_tol=0.0):
    return 0.0
  return 0.0 if yaw == 0.0 else float(yaw)


def _canonical_yaw(value: Any) -> float:
  return canonical_yaw_v2(value)


def _circular_yaw_distance(value: float, target: float) -> float:
  difference = abs(value - target) % (2.0 * math.pi)
  return min(difference, 2.0 * math.pi - difference)


def _offset_provenance_tier(offset: Mapping[str, Any]) -> int:
  families = set(map(str, offset["offset_families"]))
  if families & {
      "adjacent_shared_edge_vertices_local_2d",
      "adjacent_shared_edge_midpoints_local_2d",
  }:
    return 0
  if families & {
      "trim_vertices_local_2d", "trim_edge_midpoints_local_2d",
      "loop_centroids_local_2d",
  }:
    return 1
  return 2


def _offset_base_rank(offset: Mapping[str, Any]) -> tuple[int, float]:
  displacement = float(offset["fixed_displacement_mm"])
  if not math.isfinite(displacement) or displacement < 0.0:
    raise ValueError("P11 factor fixed displacement differs")
  return _offset_provenance_tier(offset), round(displacement, 9)


def _ordered_offsets_for_yaw(
    offsets: Sequence[Mapping[str, Any]], yaw: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
  """Use candidate identity only for equal provenance/displacement factors."""

  groups: dict[tuple[int, float], list[Mapping[str, Any]]] = {}
  for offset in offsets:
    groups.setdefault(_offset_base_rank(offset), []).append(offset)
  ordered: list[Mapping[str, Any]] = []
  for rank in sorted(groups):
    rows = groups[rank]
    # A singleton rank group has no tie to resolve. Avoid materializing and
    # hashing its SE(3) candidate identity; high-cardinality real domains are
    # overwhelmingly singleton by the frozen provenance/displacement rank.
    if len(rows) == 1:
      ordered.append(rows[0])
    else:
      ordered.extend(sorted(
          rows, key=lambda offset: _identity(yaw, offset)["candidate_key"],
      ))
  return ordered


def _membership(
    *, rank: int, stage: str, stratum: str, yaw: Mapping[str, Any],
    offset: Mapping[str, Any],
) -> dict[str, Any]:
  identity = _identity(yaw, offset)
  return {
      "schema_version": P11_FACTORIZED_MEMBERSHIP_SCHEMA_V2,
      "selected_rank": rank, "selection_stage": stage,
      "selection_stratum": stratum,
      "normal_sign": int(yaw["normal_sign"]),
      "yaw_factor_sha256": yaw["factor_payload_sha256"],
      "axial_offset_factor_sha256": offset["factor_payload_sha256"],
      "candidate_key": identity["candidate_key"],
  }


def _selection_commitment_payload(selection: Mapping[str, Any]) -> dict[str, Any]:
  return {
      key: selection[key] for key in (
          "ranking_schema", "factorized_domain_schema",
          "factorized_domain_sha256", "candidate_domain_count", "compact_top_k",
          "selected_candidate_count", "materialized_candidate_count",
          "selected_factor_memberships", "diversity_coverage",
      )
  }


def commit_p11_factorized_domain_v2(
    *, yaw_factor_payloads: Sequence[Mapping[str, Any]],
    axial_offset_factor_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Commit direct geometry factors without expanding their Cartesian product."""

  yaw_rows: dict[str, dict[str, Any]] = {}
  for raw in yaw_factor_payloads:
    unsigned = {"schema_version": _YAW_SCHEMA, **_copy(raw)}
    if set(unsigned) != {
        "schema_version", "normal_sign", "yaw_radians", "yaw_events",
        "base_world_delta_row_major", "child_world_input_row_major",
    }:
      raise ValueError("P11 direct yaw factor schema differs")
    factor = _factor(unsigned)
    yaw_rows.setdefault(factor["factor_payload_sha256"], factor)
  offset_rows: dict[str, dict[str, Any]] = {}
  for raw in axial_offset_factor_payloads:
    unsigned = {"schema_version": _OFFSET_SCHEMA, **_copy(raw)}
    if set(unsigned) != {
        "schema_version", "normal_sign", "axial_offset_mm",
        "translation_delta_world_mm", "a_landmark", "b_landmark",
        "alignment_events", "offset_families", "fixed_displacement_mm",
    }:
      raise ValueError("P11 direct offset factor schema differs")
    factor = _factor(unsigned)
    offset_rows.setdefault(factor["factor_payload_sha256"], factor)
  signs = sorted({
      int(row["normal_sign"]) for row in (*yaw_rows.values(), *offset_rows.values())
  })
  sign_rows = [_factor({
      "schema_version": _SIGN_SCHEMA, "normal_sign": sign,
  }) for sign in signs]
  domain: dict[str, Any] = {
      "schema_version": P11_FACTORIZED_DOMAIN_SCHEMA_V2,
      "expansion_schema": P11_FACTORIZED_EXPANSION_SCHEMA_V2,
      "candidate_identity_schema": "constraint_manifold_candidate_identity.v4",
      "identity_arithmetic_schema": P11_IDENTITY_ARITHMETIC_SCHEMA_V1,
      "sign_factors": sorted(sign_rows, key=lambda row: row["factor_payload_sha256"]),
      "yaw_factors": sorted(yaw_rows.values(), key=lambda row: row["factor_payload_sha256"]),
      "axial_offset_factors": sorted(
          offset_rows.values(), key=lambda row: row["factor_payload_sha256"],
      ),
  }
  yaw_signs = [int(row["normal_sign"]) for row in domain["yaw_factors"]]
  offset_signs = [int(row["normal_sign"]) for row in domain["axial_offset_factors"]]
  domain["candidate_domain_count"] = sum(
      yaw_signs.count(sign) * offset_signs.count(sign) for sign in signs
  )
  domain["candidate_domain_sha256"] = canonical_sha256(
      _domain_commitment_payload(domain)
  )
  verify_p11_factorized_domain_v2(domain)
  return domain


def verify_p11_factorized_domain_v2(domain: Mapping[str, Any]) -> None:
  expected_keys = {
      "schema_version", "expansion_schema", "candidate_identity_schema",
      "identity_arithmetic_schema",
      "sign_factors", "yaw_factors", "axial_offset_factors",
      "candidate_domain_count", "candidate_domain_sha256",
  }
  if not isinstance(domain, Mapping) or set(domain) != expected_keys:
    raise ValueError("P11 factorized domain schema differs")
  if (
      domain["schema_version"] != P11_FACTORIZED_DOMAIN_SCHEMA_V2
      or domain["expansion_schema"] != P11_FACTORIZED_EXPANSION_SCHEMA_V2
      or domain["candidate_identity_schema"]
      != "constraint_manifold_candidate_identity.v4"
      or domain["identity_arithmetic_schema"]
      != P11_IDENTITY_ARITHMETIC_SCHEMA_V1
  ):
    raise ValueError("P11 factorized domain identity differs")
  factors_by_kind = (
      ("sign_factors", _SIGN_SCHEMA, _validate_sign_factor),
      ("yaw_factors", _YAW_SCHEMA, _validate_yaw_factor),
      ("axial_offset_factors", _OFFSET_SCHEMA, _validate_offset_factor),
  )
  unsigned_by_kind: dict[str, list[dict[str, Any]]] = {}
  for key, schema, validator in factors_by_kind:
    rows = domain[key]
    if not isinstance(rows, list) or not rows:
      raise ValueError("P11 factor domain is empty")
    unsigned = [_factor_unsigned(row) for row in rows]
    if any(row.get("schema_version") != schema for row in unsigned):
      raise ValueError("P11 factor schema differs")
    for row in unsigned:
      validator(row)
    if rows != sorted(rows, key=lambda row: row["factor_payload_sha256"]):
      raise ValueError("P11 factor order differs")
    if len({row["factor_payload_sha256"] for row in rows}) != len(rows):
      raise ValueError("P11 factor identity is duplicated")
    unsigned_by_kind[key] = unsigned
  signs = {int(row["normal_sign"]) for row in unsigned_by_kind["sign_factors"]}
  if signs != {-1, 1}:
    raise ValueError("P11 factor sign domain differs")
  yaw_signs = [int(row["normal_sign"]) for row in unsigned_by_kind["yaw_factors"]]
  offset_signs = [
      int(row["normal_sign"]) for row in unsigned_by_kind["axial_offset_factors"]
  ]
  if set(yaw_signs) - signs or set(offset_signs) - signs:
    raise ValueError("P11 factor sign membership differs")
  child_inputs = {
      canonical_sha256(row["child_world_input_row_major"])
      for row in unsigned_by_kind["yaw_factors"]
  }
  if len(child_inputs) != 1:
    raise ValueError("P11 factor child input identity differs")
  count = sum(
      yaw_signs.count(sign) * offset_signs.count(sign) for sign in signs
  )
  if int(domain["candidate_domain_count"]) != count:
    raise ValueError("P11 factorized domain count differs")
  if domain["candidate_domain_sha256"] != canonical_sha256(
      _domain_commitment_payload(domain)
  ):
    raise ValueError("P11 factorized domain commitment differs")


def _select_p11_factorized_topk_v2(
    domain: Mapping[str, Any],
) -> dict[str, Any]:
  verify_p11_factorized_domain_v2(domain)
  yaws_by_sign = {
      sign: sorted(
          (row for row in domain["yaw_factors"]
           if int(row["normal_sign"]) == sign),
          key=lambda row: (
              _canonical_yaw(row["yaw_radians"]), row["factor_payload_sha256"],
          ),
      ) for sign in (-1, 1)
  }
  offsets_by_sign = {
      sign: [
          row for row in domain["axial_offset_factors"]
          if int(row["normal_sign"]) == sign
      ] for sign in (-1, 1)
  }
  strata = tuple(sorted({
      f"sign={sign}|offset_family={family}"
      for sign in (-1, 1) for offset in offsets_by_sign[sign]
      for family in offset["offset_families"]
  }))
  factors_by_stratum: dict[str, tuple[int, str, list[Mapping[str, Any]]]] = {}
  for stratum in strata:
    sign_text, family_text = stratum.split("|", 1)
    sign = int(sign_text.removeprefix("sign="))
    family = family_text.removeprefix("offset_family=")
    factors_by_stratum[stratum] = (
        sign, family, [
            row for row in offsets_by_sign[sign]
            if family in set(map(str, row["offset_families"]))
        ],
    )
  limit = min(P11_FACTORIZED_TOP_K_V2, int(domain["candidate_domain_count"]))
  selected: list[dict[str, Any]] = []
  selected_keys: set[str] = set()
  covered: set[str] = set()

  def add(
      yaw: Mapping[str, Any], offset: Mapping[str, Any], *, stage: str,
      stratum: str,
  ) -> bool:
    candidate_key = _identity(yaw, offset)["candidate_key"]
    if candidate_key in selected_keys or len(selected) >= limit:
      return False
    selected.append(_membership(
        rank=len(selected) + 1, stage=stage, stratum=stratum,
        yaw=yaw, offset=offset,
    ))
    selected_keys.add(candidate_key)
    sign = int(offset["normal_sign"])
    covered.update(
        f"sign={sign}|offset_family={family}"
        for family in offset["offset_families"]
    )
    return True

  # Stage A is fixed before any pose, label, overlap, or exact-body observation.
  anchor_specs = (
      (0.0, ("lower_endpoint", "upper_endpoint")),
      (math.pi, ("interval_center", "opposed_endpoint")),
  )
  for sign in (-1, 1):
    yaws = yaws_by_sign[sign]
    for target, families in anchor_specs:
      if not yaws:
        continue
      yaw = min(yaws, key=lambda row: (
          _circular_yaw_distance(_canonical_yaw(row["yaw_radians"]), target),
          _canonical_yaw(row["yaw_radians"]), row["factor_payload_sha256"],
      ))
      for family in families:
        stratum = f"sign={sign}|offset_family={family}"
        available = factors_by_stratum.get(stratum, (sign, family, []))[2]
        ordered = _ordered_offsets_for_yaw(available, yaw)
        if ordered:
          add(yaw, ordered[0], stage="stage_a_anchor", stratum=stratum)
  stage_a_count = len(selected)

  def pair_stream(stratum: str):
    sign, _, available = factors_by_stratum[stratum]
    yaws = yaws_by_sign[sign]
    ordered_by_yaw = [
        _ordered_offsets_for_yaw(available, yaw) for yaw in yaws
    ]
    depth = 0
    while any(depth < len(rows) for rows in ordered_by_yaw):
      for yaw, rows in zip(yaws, ordered_by_yaw, strict=True):
        if depth < len(rows):
          yield yaw, rows[depth]
      depth += 1

  # Stage B1 guarantees every available sign x offset-family stratum once.
  for stratum in strata:
    if stratum in covered or len(selected) >= limit:
      continue
    for yaw, offset in pair_stream(stratum):
      if add(yaw, offset, stage="stage_b_quota", stratum=stratum):
        break
  if len(strata) <= limit and set(strata) - covered:
    raise ValueError("P11 factorized minimum stratum coverage failed")
  stage_b_quota_count = len(selected) - stage_a_count

  # Stage B2 advances one yaw bucket per stratum at a time.  No Cartesian
  # candidate list is built; identities are produced only as an iterator is read.
  streams = {stratum: pair_stream(stratum) for stratum in strata}
  exhausted: set[str] = set()
  while len(selected) < limit and len(exhausted) < len(strata):
    progressed = False
    for stratum in strata:
      if stratum in exhausted:
        continue
      stream = streams[stratum]
      for yaw, offset in stream:
        if add(yaw, offset, stage="stage_b_round_robin", stratum=stratum):
          progressed = True
          break
      else:
        exhausted.add(stratum)
      if len(selected) >= limit:
        break
    if not progressed and len(exhausted) < len(strata):
      continue
  if len(selected) != limit:
    raise ValueError("P11 factorized selection did not fill top-k")
  coverage: dict[str, Any] = {
      "schema_version": P11_FACTORIZED_COVERAGE_SCHEMA_V2,
      "stratum_definition": "factor_normal_sign_x_factor_offset_family",
      "yaw_bucket_definition": "canonical_factor_yaw_radians_mod_2pi",
      "quota_strata": list(strata), "covered_quota_strata": sorted(covered),
      "stage_a_candidate_count": stage_a_count,
      "stage_b_quota_candidate_count": stage_b_quota_count,
      "selected_candidate_count": len(selected),
      "selected_yaw_factor_count": len({
          row["yaw_factor_sha256"] for row in selected
      }),
  }
  coverage["coverage_payload_sha256"] = canonical_sha256(coverage)
  result: dict[str, Any] = {
      "ranking_schema": P11_FACTORIZED_STRUCTURAL_TOPK_SCHEMA_V2,
      "factorized_domain_schema": P11_FACTORIZED_DOMAIN_SCHEMA_V2,
      "factorized_domain_sha256": domain["candidate_domain_sha256"],
      "candidate_domain_count": domain["candidate_domain_count"],
      "compact_top_k": P11_FACTORIZED_TOP_K_V2,
      "selected_candidate_count": len(selected),
      "materialized_candidate_count": len(selected),
      "selected_factor_memberships": selected,
      "diversity_coverage": coverage,
  }
  result["selection_payload_sha256"] = canonical_sha256(
      _selection_commitment_payload(result)
  )
  return result


def select_p11_factorized_topk_v2(
    domain: Mapping[str, Any],
) -> dict[str, Any]:
  """Select the bounded non-oracle P11 structural proposal set."""

  result = _select_p11_factorized_topk_v2(domain)
  verify_p11_factorized_selection_v2(domain, result)
  return result


def verify_p11_factorized_selection_v2(
    domain: Mapping[str, Any], selection: Mapping[str, Any],
) -> None:
  """Replay factor membership, order, identity, and the complete selection."""

  verify_p11_factorized_domain_v2(domain)
  expected_keys = {
      "ranking_schema", "factorized_domain_schema", "factorized_domain_sha256",
      "candidate_domain_count", "compact_top_k", "selected_candidate_count",
      "materialized_candidate_count", "selected_factor_memberships",
      "diversity_coverage", "selection_payload_sha256",
  }
  if not isinstance(selection, Mapping) or set(selection) != expected_keys:
    raise ValueError("P11 factorized selection schema differs")
  if (
      selection["ranking_schema"] != P11_FACTORIZED_STRUCTURAL_TOPK_SCHEMA_V2
      or selection["factorized_domain_schema"] != P11_FACTORIZED_DOMAIN_SCHEMA_V2
      or selection["factorized_domain_sha256"] != domain["candidate_domain_sha256"]
      or selection["candidate_domain_count"] != domain["candidate_domain_count"]
      or selection["compact_top_k"] != P11_FACTORIZED_TOP_K_V2
  ):
    raise ValueError("P11 factorized selection authority differs")
  memberships = selection["selected_factor_memberships"]
  limit = min(P11_FACTORIZED_TOP_K_V2, int(domain["candidate_domain_count"]))
  if (
      not isinstance(memberships, list) or len(memberships) != limit
      or selection["selected_candidate_count"] != limit
      or selection["materialized_candidate_count"] != limit
  ):
    raise ValueError("P11 factorized selection count differs")
  yaws = {row["factor_payload_sha256"]: row for row in domain["yaw_factors"]}
  offsets = {
      row["factor_payload_sha256"]: row
      for row in domain["axial_offset_factors"]
  }
  candidate_keys: set[str] = set()
  for rank, membership in enumerate(memberships, start=1):
    if set(membership) != {
        "schema_version", "selected_rank", "selection_stage",
        "selection_stratum", "normal_sign", "yaw_factor_sha256",
        "axial_offset_factor_sha256", "candidate_key",
    } or (
        membership["schema_version"] != P11_FACTORIZED_MEMBERSHIP_SCHEMA_V2
        or membership["selected_rank"] != rank
    ):
      raise ValueError("P11 factorized selection membership differs")
    yaw = yaws.get(membership["yaw_factor_sha256"])
    offset = offsets.get(membership["axial_offset_factor_sha256"])
    if yaw is None or offset is None or (
        int(yaw["normal_sign"]) != int(offset["normal_sign"])
        or int(membership["normal_sign"]) != int(yaw["normal_sign"])
        or membership["candidate_key"] != _identity(yaw, offset)["candidate_key"]
    ):
      raise ValueError("P11 factorized selection membership replay differs")
    if membership["candidate_key"] in candidate_keys:
      raise ValueError("P11 factorized selection membership is duplicated")
    candidate_keys.add(membership["candidate_key"])
  coverage = selection["diversity_coverage"]
  if not isinstance(coverage, Mapping) or coverage.get(
      "coverage_payload_sha256"
  ) != canonical_sha256({
      key: value for key, value in coverage.items()
      if key != "coverage_payload_sha256"
  }):
    raise ValueError("P11 factorized selection coverage differs")
  if selection["selection_payload_sha256"] != canonical_sha256(
      _selection_commitment_payload(selection)
  ):
    raise ValueError("P11 factorized selection commitment differs")
  if _copy(selection) != _select_p11_factorized_topk_v2(domain):
    raise ValueError("P11 factorized selection replay differs")


def expand_factorized_candidate_identities_v2(
    domain: Mapping[str, Any],
) -> list[dict[str, Any]]:
  """Diagnostic exhaustive expansion; production selection never calls this."""

  verify_p11_factorized_domain_v2(domain)
  rows = []
  for yaw in domain["yaw_factors"]:
    for offset in domain["axial_offset_factors"]:
      if int(yaw["normal_sign"]) == int(offset["normal_sign"]):
        rows.append(_identity(yaw, offset))
  if len(rows) != int(domain["candidate_domain_count"]):
    raise ValueError("P11 factor expansion count differs")
  if len({row["candidate_key"] for row in rows}) != len(rows):
    raise ValueError("P11 factor expansion candidate identity is duplicated")
  return rows


def factorize_exhaustive_p11_candidates_v2(
    candidate_rows: Sequence[Mapping[str, Any]], *,
    child_world_row_major: Sequence[float],
) -> dict[str, Any]:
  """Development adapter proving V2 factors equal a frozen exhaustive domain."""

  if not candidate_rows:
    raise ValueError("P11 factor exhaustive domain is empty")
  sign_rows: dict[int, dict[str, Any]] = {}
  yaw_rows: dict[str, dict[str, Any]] = {}
  offset_rows: dict[str, dict[str, Any]] = {}
  original: dict[str, dict[str, Any]] = {}
  child_input = _matrix(child_world_row_major, label="child input transform")
  for raw in candidate_rows:
    row = _copy(raw)
    if int(row.get("program_index", -1)) != 11:
      raise ValueError("P11 factor exhaustive candidate program differs")
    sign = int(row.get("normal_sign", 0))
    if sign not in {-1, 1}:
      raise ValueError("P11 factor candidate sign differs")
    world = _matrix(row["world_delta_row_major"], label="world transform")
    child = _matrix(row["child_world_row_major"], label="child transform")
    base = _matrix(row["base_world_delta_row_major"], label="base transform")
    translation = world[:3, 3] - base[:3, 3]
    sign_rows[canonical_sha256({"normal_sign": sign})] = _factor({
        "schema_version": _SIGN_SCHEMA, "normal_sign": sign,
    })
    yaw_unsigned = {
        "schema_version": _YAW_SCHEMA, "normal_sign": sign,
        "yaw_radians": float(row["yaw_radians"]),
        "yaw_events": sorted(row["yaw_events"], key=canonical_sha256),
        "base_world_delta_row_major": base.reshape(-1).tolist(),
        "child_world_input_row_major": child_input.reshape(-1).tolist(),
    }
    yaw_factor = _factor(yaw_unsigned)
    prior_yaw = yaw_rows.setdefault(yaw_factor["factor_payload_sha256"], yaw_factor)
    if prior_yaw != yaw_factor:
      raise ValueError("P11 factor yaw evidence differs")
    events = sorted(row["alignment_events"], key=canonical_sha256)
    families = sorted({str(event["kind"]) for event in events})
    fixed = float(row.get("cheap_metrics", {}).get("fixed_displacement_mm", math.nan))
    if not math.isfinite(fixed) or fixed < 0.0:
      raise ValueError("P11 factor fixed displacement differs")
    offset_unsigned = {
        "schema_version": _OFFSET_SCHEMA, "normal_sign": sign,
        "axial_offset_mm": float(row["offset_local_2d"][1]),
        "translation_delta_world_mm": translation.tolist(),
        "a_landmark": row["a_landmark"], "b_landmark": row["b_landmark"],
        "alignment_events": events, "offset_families": families,
        "fixed_displacement_mm": fixed,
    }
    offset_factor = _factor(offset_unsigned)
    prior_offset = offset_rows.setdefault(
        offset_factor["factor_payload_sha256"], offset_factor,
    )
    if prior_offset != offset_factor:
      raise ValueError("P11 factor offset evidence differs")
    key = str(row["candidate_key"])
    if key != canonical_sha256({
        "schema_version": "constraint_manifold_candidate_identity.v4",
        "program_index": 11,
        "world_delta_row_major": world.reshape(-1).tolist(),
        "child_world_row_major": child.reshape(-1).tolist(),
    }):
      raise ValueError("P11 factor candidate identity differs")
    prior = original.setdefault(key, row)
    if prior != row:
      raise ValueError("P11 factor exhaustive candidate is duplicated")
  domain: dict[str, Any] = {
      "schema_version": P11_FACTORIZED_DOMAIN_SCHEMA_V2,
      "expansion_schema": P11_FACTORIZED_EXPANSION_SCHEMA_V2,
      "candidate_identity_schema": "constraint_manifold_candidate_identity.v4",
      "identity_arithmetic_schema": P11_IDENTITY_ARITHMETIC_SCHEMA_V1,
      "sign_factors": sorted(sign_rows.values(), key=lambda row: row["factor_payload_sha256"]),
      "yaw_factors": sorted(yaw_rows.values(), key=lambda row: row["factor_payload_sha256"]),
      "axial_offset_factors": sorted(
          offset_rows.values(), key=lambda row: row["factor_payload_sha256"],
      ),
  }
  yaw_signs = [int(row["normal_sign"]) for row in domain["yaw_factors"]]
  offset_signs = [int(row["normal_sign"]) for row in domain["axial_offset_factors"]]
  domain["candidate_domain_count"] = sum(
      yaw_signs.count(sign) * offset_signs.count(sign) for sign in (-1, 1)
  )
  domain["candidate_domain_sha256"] = canonical_sha256(
      _domain_commitment_payload(domain)
  )
  verify_p11_factorized_domain_v2(domain)
  expanded = expand_factorized_candidate_identities_v2(domain)
  if {row["candidate_key"] for row in expanded} != set(original):
    raise ValueError("P11 factor exhaustive identity membership differs")
  for row in expanded:
    expected = original[row["candidate_key"]]
    if (
        row["world_delta_row_major"] != expected["world_delta_row_major"]
        or row["child_world_row_major"] != expected["child_world_row_major"]
    ):
      raise ValueError("P11 factor exhaustive placement replay differs")
  return domain


__all__ = [
    "P11_FACTORIZED_DOMAIN_SCHEMA_V2", "P11_FACTORIZED_EXPANSION_SCHEMA_V2",
    "P11_FACTORIZED_STRUCTURAL_TOPK_SCHEMA_V2", "P11_FACTORIZED_TOP_K_V2",
    "P11_IDENTITY_ARITHMETIC_SCHEMA_V1",
    "commit_p11_factorized_domain_v2", "expand_factorized_candidate_identities_v2",
    "canonical_yaw_v2",
    "factorize_exhaustive_p11_candidates_v2", "select_p11_factorized_topk_v2",
    "multiply_se3_row_major_v2",
    "verify_p11_factorized_domain_v2", "verify_p11_factorized_selection_v2",
]
