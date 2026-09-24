"""Label-free STEP model boundary for the NeuroCAD benchmark.

Version 2 cached endpoint-centred graphs duplicated the private endpoint mask
in two places: ``endpoint_markers`` and the final two node-feature columns.
This module defines the replacement *model-visible* boundary.  It intentionally
has no target, residual, endpoint marker, source-face identity, case identity,
or authority-row identity.  Supervision belongs to the verifier-owned cache
module and is never accepted by this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "benchmark_v3_unlabeled_step_view.v1"
GRAPH_SCHEMA_VERSION = "benchmark_v3_unlabeled_intrinsic_brep_graph.v1"

SURFACE_TYPES = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "spline",
    "other",
)
INTRINSIC_NODE_FEATURE_NAMES = (
    "area_fraction",
    "perimeter_over_sqrt_total_area",
    "mean_curvature_times_sqrt_total_area",
    "gaussian_curvature_times_total_area",
    "boundary_loop_count",
    "boundary_edge_count",
    "area_fraction_present",
    "perimeter_over_sqrt_total_area_present",
    "mean_curvature_times_sqrt_total_area_present",
    "gaussian_curvature_times_total_area_present",
    "boundary_loop_count_present",
    "boundary_edge_count_present",
    *(f"surface_type_{name}" for name in SURFACE_TYPES),
)
EDGE_FEATURE_NAMES = (
    "shared_edge_length_over_sqrt_total_area",
    "dihedral_cos",
    "dihedral_sin_abs",
)
INTRINSIC_NODE_INPUT_DIM = len(INTRINSIC_NODE_FEATURE_NAMES)
EDGE_INPUT_DIM = len(EDGE_FEATURE_NAMES)

FORBIDDEN_MODEL_VISIBLE_TOKENS = frozenset(
    {
        "endpoint_marker",
        "endpoint_markers",
        "endpoint_a",
        "endpoint_b",
        "face_index",
        "face_indices",
        "source_face",
        "source_face_identity",
        "source_face_signature",
        "target",
        "target_program",
        "target_program_index",
        "residual",
        "residual_mask",
        "residual_translation",
        "residual_rotation",
        "program_index",
        "case_index",
        "model_view_sha256",
    }
)
_VIEW_FACTORY_TOKEN = object()


def canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any],
    ndim: int,
    trailing_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
  array = np.asarray(value, dtype=dtype)
  if array.ndim != ndim or tuple(array.shape[-len(trailing_shape):]) != trailing_shape:
    raise ValueError(f"{label} shape differs from the unlabeled-view schema")
  if not np.isfinite(array).all() if np.issubdtype(array.dtype, np.floating) else False:
    raise ValueError(f"{label} contains non-finite values")
  immutable = np.array(array, dtype=dtype, copy=True, order="C")
  immutable.setflags(write=False)
  return immutable


def tensor_receipt(name: str, array: np.ndarray) -> dict[str, Any]:
  contiguous = np.ascontiguousarray(array)
  return {
      "name": name,
      "dtype": str(contiguous.dtype),
      "shape": list(contiguous.shape),
      "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
  }


@dataclass(frozen=True, slots=True)
class UnlabeledIntrinsicBRepGraph:
  """One model-visible graph with no endpoint/source-face annotations."""

  node_features: np.ndarray
  edge_index: np.ndarray
  edge_features: np.ndarray

  def __post_init__(self) -> None:
    nodes = _readonly_array(
        self.node_features,
        dtype=np.dtype("float32"),
        ndim=2,
        trailing_shape=(INTRINSIC_NODE_INPUT_DIM,),
        label="unlabeled graph nodes",
    )
    edges = _readonly_array(
        self.edge_index,
        dtype=np.dtype("int64"),
        ndim=2,
        trailing_shape=(2,),
        label="unlabeled graph edge index",
    )
    features = _readonly_array(
        self.edge_features,
        dtype=np.dtype("float32"),
        ndim=2,
        trailing_shape=(EDGE_INPUT_DIM,),
        label="unlabeled graph edge features",
    )
    if edges.shape[0] != features.shape[0]:
      raise ValueError("unlabeled graph edge rows differ")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= nodes.shape[0]):
      raise ValueError("unlabeled graph edge index is outside its node domain")
    object.__setattr__(self, "node_features", nodes)
    object.__setattr__(self, "edge_index", edges)
    object.__setattr__(self, "edge_features", features)

  def tensor_receipts(self, *, prefix: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        MappingProxyType(row)
        for row in (
            tensor_receipt(f"{prefix}_nodes", self.node_features),
            tensor_receipt(f"{prefix}_edge_index", self.edge_index),
            tensor_receipt(f"{prefix}_edge_features", self.edge_features),
        )
    )


class BenchmarkV3UnlabeledStepView:
  """Complete object accepted by a label-free model/proposer boundary."""

  __slots__ = ("_graphs", "_input_sha256", "_schema_version")

  def __init__(
      self,
      *,
      graphs: Sequence[UnlabeledIntrinsicBRepGraph],
      input_sha256: str,
      schema_version: str,
      _factory_token: object,
  ) -> None:
    if _factory_token is not _VIEW_FACTORY_TOKEN:
      raise TypeError("unlabeled STEP views are cache-factory-only")
    if schema_version != SCHEMA_VERSION:
      raise ValueError("unlabeled STEP view schema differs")
    frozen_graphs = tuple(graphs)
    if len(frozen_graphs) != 2 or any(
        type(graph) is not UnlabeledIntrinsicBRepGraph for graph in frozen_graphs
    ):
      raise TypeError("unlabeled STEP view requires exactly two full part graphs")
    self._graphs = frozen_graphs
    self._input_sha256 = str(input_sha256)
    self._schema_version = schema_version
    expected = canonical_sha256(self.receipt_payload(include_input_sha256=False))
    if self._input_sha256 != expected:
      raise ValueError("unlabeled STEP view input hash differs")

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("unlabeled STEP views are cache-owned and not serializable")

  @property
  def graphs(self) -> tuple[UnlabeledIntrinsicBRepGraph, ...]:
    return self._graphs

  @property
  def input_sha256(self) -> str:
    return self._input_sha256

  @property
  def schema_version(self) -> str:
    return self._schema_version

  def receipt_payload(self, *, include_input_sha256: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "schema_version": self.schema_version,
        "graphs": [
            {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "tensors": [dict(row) for row in graph.tensor_receipts(prefix=f"g{i}")],
            }
            for i, graph in enumerate(self.graphs)
        ],
        "feature_contract": {
            "intrinsic_node_feature_names": list(INTRINSIC_NODE_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "endpoint_annotation_columns": 0,
        },
    }
    if include_input_sha256:
      payload["input_sha256"] = self.input_sha256
    assert_no_forbidden_model_visible_fields(payload)
    return payload


def _build_cache_owned_unlabeled_step_view(
    *,
    graphs: Sequence[UnlabeledIntrinsicBRepGraph],
) -> BenchmarkV3UnlabeledStepView:
  """Internal cache factory; callers cannot directly instantiate a view."""

  frozen_graphs = tuple(graphs)
  if len(frozen_graphs) != 2:
    raise ValueError("unlabeled STEP input requires exactly two full graphs")
  provisional = {
      "schema_version": SCHEMA_VERSION,
      "graphs": [
          {
              "schema_version": GRAPH_SCHEMA_VERSION,
              "tensors": [dict(row) for row in graph.tensor_receipts(prefix=f"g{i}")],
          }
          for i, graph in enumerate(frozen_graphs)
      ],
      "feature_contract": {
          "intrinsic_node_feature_names": list(INTRINSIC_NODE_FEATURE_NAMES),
          "edge_feature_names": list(EDGE_FEATURE_NAMES),
          "endpoint_annotation_columns": 0,
      },
  }
  assert_no_forbidden_model_visible_fields(provisional)
  return BenchmarkV3UnlabeledStepView(
      graphs=frozen_graphs,
      input_sha256=canonical_sha256(provisional),
      schema_version=SCHEMA_VERSION,
      _factory_token=_VIEW_FACTORY_TOKEN,
  )


def _walk_keys(value: Any) -> Sequence[str]:
  found: list[str] = []
  if isinstance(value, Mapping):
    for key, child in value.items():
      found.append(str(key).strip().lower())
      found.extend(_walk_keys(child))
  elif isinstance(value, (list, tuple)):
    for child in value:
      found.extend(_walk_keys(child))
  return found


def assert_no_forbidden_model_visible_fields(value: Any) -> None:
  """Fail closed if a model-visible mapping acquires supervision or face IDs."""

  for key in _walk_keys(value):
    normalized = key.replace("-", "_")
    if normalized in FORBIDDEN_MODEL_VISIBLE_TOKENS or any(
        token in normalized
        for token in (
            "endpoint_marker",
            "source_face",
            "target_program",
            "residual_translation",
            "residual_rotation",
        )
    ):
      raise ValueError(f"forbidden model-visible field: {key}")


def require_unlabeled_step_model_input(value: Any) -> BenchmarkV3UnlabeledStepView:
  """Exact forward/proposer firewall; wrappers containing gold are rejected."""

  if type(value) is not BenchmarkV3UnlabeledStepView:
    raise TypeError(
        "forward/proposer accepts BenchmarkV3UnlabeledStepView only; "
        "training-row or label capabilities are forbidden"
    )
  assert_no_forbidden_model_visible_fields(value.receipt_payload())
  return value
