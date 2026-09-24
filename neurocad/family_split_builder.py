"""Build deterministic family-disjoint benchmark-v2 train/dev splits.

Formal construction is provenance driven.  It never guesses design families
from filenames, assembly ids, or source paths, and it never emits a final-test
split.
"""

from __future__ import annotations

import argparse
from array import array
import base64
import binascii
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Sequence
import uuid

from .paths import resolve_path


FAMILY_PROVENANCE_VERSION = "family_provenance.v1"
GRAPH_FINGERPRINT_EXTERNAL_VERSION = "external.v1"
GRAPH_FINGERPRINT_CANONICAL_VERSION = "attributed_wl_bucket.v2"
FAMILY_SPLIT_POLICY_VERSION = "family_split_policy.v1"
FAMILY_SOURCE_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "benchmark_v2_family_source.schema.json"
)


class FamilyEvidenceError(ValueError):
  """Raised when a case cannot support a formal family assignment."""


@dataclass(frozen=True)
class FamilySplitConfig:
  """Configuration for the train/dev-only family split builder."""

  seed: int = 2027
  dev_ratio: float = 0.2
  dev_ratio_tolerance: float = 0.05
  fail_if_ratio_impossible: bool = True
  near_family_threshold: float | None = None
  neighbor_top_k: int = 20
  neighbor_warning_threshold: float = 0.9
  required_evidence_coverage: float = 1.0
  family_pair_review_artifact: dict[str, Any] | None = None
  family_pair_review_verification_key: dict[str, Any] | str | bytes | None = None
  development_synthetic_mode: bool = False
  final_test_generation_rule: str = (
      "Apply this frozen family-grouping policy to an externally held source "
      "pool only after model selection is complete."
  )
  final_test_salt_commitment: str | None = None
  external_final_test_attestation: dict[str, Any] | None = None
  external_final_test_verification_key: dict[str, Any] | str | bytes | None = None


@dataclass(frozen=True)
class FamilyEvidence:
  """Normalized provenance used for family construction and auditing."""

  case_id: str
  provenance_version: str
  formal: bool
  source_lineages: tuple[str, ...]
  body_sha1s: tuple[str, ...]
  graph_fingerprint: str
  graph_fingerprint_version: str
  assembly_graph: dict[str, Any] | None = None
  geometry_embedding: tuple[float, ...] | None = None
  geometry_embedding_producer: str | None = None
  shape_signature: tuple[str, ...] = ()
  shape_signature_producer: str | None = None
  mate_graph_signature: tuple[str, ...] = ()
  mate_graph_signature_producer: str | None = None
  synthetic_fields: tuple[str, ...] = ()

  @property
  def synthetic(self) -> bool:
    return bool(self.synthetic_fields)

  def to_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provenance_version": self.provenance_version,
        "formal": self.formal,
        "source_lineage": list(self.source_lineages),
        "body_sha1_set": list(self.body_sha1s),
        "graph_fingerprint": self.graph_fingerprint,
        "graph_fingerprint_version": self.graph_fingerprint_version,
        "synthetic": self.synthetic,
        "synthetic_fields": list(self.synthetic_fields),
    }
    if self.geometry_embedding is not None:
      result["geometry_embedding"] = list(self.geometry_embedding)
      result["geometry_embedding_provenance"] = json.loads(
          str(self.geometry_embedding_producer)
      )
    if self.shape_signature:
      result["shape_signature"] = list(self.shape_signature)
      result["shape_signature_provenance"] = json.loads(
          str(self.shape_signature_producer)
      )
    if self.mate_graph_signature:
      result["mate_graph_signature"] = list(self.mate_graph_signature)
      result["mate_graph_signature_provenance"] = json.loads(
          str(self.mate_graph_signature_producer)
      )
    return result

  @property
  def has_near_family_descriptor(self) -> bool:
    return bool(
        (self.geometry_embedding is not None and self.geometry_embedding_producer)
        or (self.shape_signature and self.shape_signature_producer)
        or (self.mate_graph_signature and self.mate_graph_signature_producer)
    )


class _UnionFind:
  def __init__(self, size: int) -> None:
    self.parent = list(range(size))
    self.rank = [0] * size

  def find(self, item: int) -> int:
    parent = self.parent[item]
    if parent != item:
      self.parent[item] = self.find(parent)
    return self.parent[item]

  def union(self, left: int, right: int) -> bool:
    root_left = self.find(left)
    root_right = self.find(right)
    if root_left == root_right:
      return False
    if self.rank[root_left] < self.rank[root_right]:
      root_left, root_right = root_right, root_left
    self.parent[root_right] = root_left
    if self.rank[root_left] == self.rank[root_right]:
      self.rank[root_left] += 1
    return True


def _canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _family_source_schema_identity() -> dict[str, Any]:
  """Return the immutable identity of the authoritative provenance schema."""

  raw = FAMILY_SOURCE_SCHEMA_PATH.read_bytes()
  payload = json.loads(raw.decode("utf-8"))
  return {
      "schema_version": FAMILY_PROVENANCE_VERSION,
      "schema_id": str(payload.get("$id") or ""),
      "path": "configs/benchmark_v2_family_source.schema.json",
      "sha256": "sha256:" + _sha256_bytes(raw),
  }


def _code_identity() -> dict[str, Any]:
  module_dir = Path(__file__).resolve().parent
  files = [
      {
          "module": module,
          "path": relative_path,
          "sha256": "sha256:"
          + _sha256_bytes((module_dir / relative_path).read_bytes()),
      }
      for module, relative_path in (
          ("neurocad.family_split_builder", "family_split_builder.py"),
          ("neurocad.split_hygiene_audit", "split_hygiene_audit.py"),
      )
  ]
  return {
      "schema_version": "family_code_identity.v2",
      "files": files,
      "sha256": "sha256:" + _sha256_text(_canonical_json(files)),
  }


_PROVENANCE_ALLOWED_KEYS = {
    "provenance_version",
    "formal",
    "source_lineage",
    "body_sha1_set",
    "graph_fingerprint",
    "graph_fingerprint_version",
    "assembly_graph",
    "geometry_embedding",
    "geometry_embedding_provenance",
    "shape_signature",
    "shape_signature_provenance",
    "mate_graph_signature",
    "mate_graph_signature_provenance",
    # These two fields appear only in audit records emitted by this module.
    "synthetic",
    "synthetic_fields",
}


def _validate_authoritative_provenance_schema(
    raw: dict[str, Any],
    *,
    case_id: str,
) -> None:
  """Validate the strict family_provenance.v1 schema without optional deps."""

  unknown = sorted(set(raw) - _PROVENANCE_ALLOWED_KEYS)
  if unknown:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance has schema-unknown fields: "
        + ", ".join(unknown)
    )
  if raw.get("provenance_version") != FAMILY_PROVENANCE_VERSION:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.provenance_version must equal "
        f"{FAMILY_PROVENANCE_VERSION!r}"
    )
  if raw.get("formal") is not True:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.formal must be true"
    )
  lineages = raw.get("source_lineage")
  if isinstance(lineages, str):
    valid_lineages = bool(lineages.strip())
  elif isinstance(lineages, list):
    valid_lineages = bool(lineages) and all(
        isinstance(value, str) and bool(value.strip()) for value in lineages
    ) and len(lineages) == len(set(lineages))
  else:
    valid_lineages = False
  if not valid_lineages:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.source_lineage violates the schema"
    )
  hashes = raw.get("body_sha1_set")
  if not isinstance(hashes, list) or not hashes or len(hashes) != len(set(hashes)):
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.body_sha1_set must be a nonempty unique list"
    )
  invalid_hashes = [
      value
      for value in hashes
      if not isinstance(value, str)
      or len(value) != 40
      or any(char not in "0123456789abcdef" for char in value)
  ]
  if invalid_hashes:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.body_sha1_set violates the SHA1 schema"
    )

  explicit_graph = raw.get("graph_fingerprint")
  graph_payload = raw.get("assembly_graph")
  if bool(isinstance(explicit_graph, str) and explicit_graph.strip()) == bool(
      graph_payload is not None
  ):
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance must provide exactly one of "
        "graph_fingerprint or assembly_graph"
    )
  graph_version = raw.get("graph_fingerprint_version")
  if explicit_graph and graph_version != GRAPH_FINGERPRINT_EXTERNAL_VERSION:
    raise FamilyEvidenceError(
        f"Case {case_id} external graph_fingerprint requires "
        f"graph_fingerprint_version={GRAPH_FINGERPRINT_EXTERNAL_VERSION!r}"
    )
  if graph_payload is not None:
    if graph_version != GRAPH_FINGERPRINT_CANONICAL_VERSION:
      raise FamilyEvidenceError(
          f"Case {case_id} derived assembly_graph requires "
          f"graph_fingerprint_version={GRAPH_FINGERPRINT_CANONICAL_VERSION!r}"
      )
    if not isinstance(graph_payload, dict):
      raise FamilyEvidenceError(
          f"Case {case_id} family_provenance.assembly_graph must be an object"
      )
    if set(graph_payload) - {"nodes", "edges"}:
      raise FamilyEvidenceError(
          f"Case {case_id} assembly_graph has schema-unknown fields"
      )
    if not isinstance(graph_payload.get("nodes"), list) or not graph_payload["nodes"]:
      raise FamilyEvidenceError(f"Case {case_id} assembly_graph.nodes must be nonempty")
    if not isinstance(graph_payload.get("edges"), list):
      raise FamilyEvidenceError(f"Case {case_id} assembly_graph.edges must be a list")

  for descriptor in (
      "geometry_embedding",
      "shape_signature",
      "mate_graph_signature",
  ):
    descriptor_present = descriptor in raw
    producer_key = f"{descriptor}_provenance"
    producer_present = producer_key in raw
    if descriptor_present != producer_present:
      raise FamilyEvidenceError(
          f"Case {case_id} {descriptor} and {producer_key} must appear together"
      )
    if not descriptor_present:
      continue
    value = raw[descriptor]
    if descriptor == "geometry_embedding" and (
        not isinstance(value, list) or not value
    ):
      raise FamilyEvidenceError(
          f"Case {case_id} geometry_embedding must be a nonempty list"
      )
    if descriptor != "geometry_embedding" and not _signature_tokens(value):
      raise FamilyEvidenceError(f"Case {case_id} {descriptor} must be nonempty")
    producer = raw[producer_key]
    if not isinstance(producer, dict) or set(producer) != {
        "producer", "version", "artifact_sha256"
    }:
      raise FamilyEvidenceError(
          f"Case {case_id} {producer_key} violates the producer schema"
      )
    _descriptor_producer(
        raw,
        descriptor_name=descriptor,
        present=True,
        case_id=case_id,
    )


def _clean_text_values(value: Any) -> tuple[str, ...]:
  if isinstance(value, str):
    values: Iterable[Any] = [value]
  elif isinstance(value, (list, tuple, set, frozenset)):
    values = value
  else:
    values = []
  return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _signature_tokens(value: Any) -> tuple[str, ...]:
  if value is None:
    return ()
  if isinstance(value, dict):
    return (_canonical_json(value),)
  return _clean_text_values(value)


def _provenance_layers(case: dict[str, Any]) -> list[dict[str, Any]]:
  layers: list[dict[str, Any]] = []
  benchmark = case.get("benchmark_metadata")
  if isinstance(benchmark, dict):
    for key in ("provenance", "family_provenance"):
      value = benchmark.get(key)
      if isinstance(value, dict):
        layers.append(value)
  for key in ("provenance", "family_provenance"):
    value = case.get(key)
    if isinstance(value, dict):
      layers.append(value)
  layers.append(case)
  return layers


def _authoritative_provenance(
    case: dict[str, Any],
    *,
    case_id: str,
    development_synthetic_mode: bool,
) -> dict[str, Any]:
  """Return the only provenance record accepted by the formal protocol."""

  raw = case.get("family_provenance")
  if development_synthetic_mode and raw is None:
    return {}
  if not isinstance(raw, dict):
    raise FamilyEvidenceError(
        f"Case {case_id} requires authoritative top-level family_provenance"
    )
  if not development_synthetic_mode:
    _validate_authoritative_provenance_schema(raw, case_id=case_id)
  version = raw.get("provenance_version")
  if version != FAMILY_PROVENANCE_VERSION:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.provenance_version must equal "
        f"{FAMILY_PROVENANCE_VERSION!r}"
    )
  if not development_synthetic_mode and raw.get("formal") is not True:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.formal must be true"
    )
  if development_synthetic_mode and not isinstance(raw.get("formal"), bool):
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance.formal must be boolean"
    )

  relevant_keys = {
      "source_lineage",
      "source_lineages",
      "body_sha1_set",
      "body_sha1s",
      "graph_fingerprint",
      "graph_fingerprint_version",
      "assembly_graph",
      "geometry_embedding",
      "geometry_embedding_provenance",
      "shape_signature",
      "shape_signature_provenance",
      "mate_graph_signature",
      "mate_graph_signature_provenance",
  }
  shadow_layers: list[tuple[str, dict[str, Any]]] = []
  shadow = case.get("provenance")
  if isinstance(shadow, dict):
    shadow_layers.append(("provenance", shadow))
  benchmark = case.get("benchmark_metadata")
  if isinstance(benchmark, dict):
    for key in ("provenance", "family_provenance"):
      value = benchmark.get(key)
      if isinstance(value, dict):
        shadow_layers.append((f"benchmark_metadata.{key}", value))
  direct = {key: case[key] for key in relevant_keys if key in case}
  if direct:
    shadow_layers.append(("case", direct))
  aliases = {
      "source_lineage": ("source_lineage", "source_lineages"),
      "body_sha1_set": ("body_sha1_set", "body_sha1s"),
      "graph_fingerprint": ("graph_fingerprint",),
      "graph_fingerprint_version": ("graph_fingerprint_version",),
      "assembly_graph": ("assembly_graph",),
      "geometry_embedding": ("geometry_embedding",),
      "geometry_embedding_provenance": ("geometry_embedding_provenance",),
      "shape_signature": ("shape_signature",),
      "shape_signature_provenance": ("shape_signature_provenance",),
      "mate_graph_signature": ("mate_graph_signature",),
      "mate_graph_signature_provenance": ("mate_graph_signature_provenance",),
  }
  for location, layer in shadow_layers:
    conflicts: list[str] = []
    for canonical, names in aliases.items():
      shadow_values = [layer[name] for name in names if name in layer]
      if not shadow_values:
        continue
      if canonical not in raw:
        conflicts.append(canonical)
        continue
      if any(
          _canonical_json(value) != _canonical_json(raw[canonical])
          for value in shadow_values
      ):
        conflicts.append(canonical)
    # Legacy nested aliases are still treated as shadow claims; they may not
    # silently disagree with the authoritative schema record.
    nested_lineages = _lineages_from_layers([layer])
    authoritative_lineages = _clean_text_values(raw.get("source_lineage"))
    if nested_lineages and nested_lineages != authoritative_lineages:
      conflicts.append("source_lineage")
    nested_hashes = _body_sha1s_from_layers([layer])
    authoritative_hashes = tuple(
        item.lower() for item in _clean_text_values(raw.get("body_sha1_set"))
    )
    if nested_hashes and nested_hashes != authoritative_hashes:
      conflicts.append("body_sha1_set")
    nested_graph = _graph_fingerprint_from_layers([layer])
    authoritative_graph = str(raw.get("graph_fingerprint") or "").strip()
    if nested_graph and nested_graph != authoritative_graph:
      conflicts.append("graph_fingerprint")
    conflicts = sorted(set(conflicts))
    if conflicts:
      raise FamilyEvidenceError(
          f"Case {case_id} has conflicting family provenance in {location}: "
          + ", ".join(conflicts)
      )
  return raw


def _last_present(layers: Sequence[dict[str, Any]], keys: Sequence[str]) -> Any:
  for layer in reversed(layers):
    for key in keys:
      if key in layer and layer[key] is not None:
        return layer[key]
  return None


def _lineages_from_layers(layers: Sequence[dict[str, Any]]) -> tuple[str, ...]:
  value = _last_present(
      layers,
      ("source_lineage", "source_lineages", "lineage", "lineage_id"),
  )
  result = _clean_text_values(value)
  if result:
    return result
  for layer in reversed(layers):
    source = layer.get("source")
    if not isinstance(source, dict):
      continue
    result = _clean_text_values(
        source.get("lineage") or source.get("lineage_id") or source.get("source_lineage")
    )
    if result:
      return result
  return ()


def _body_sha1s_from_layers(layers: Sequence[dict[str, Any]]) -> tuple[str, ...]:
  value = _last_present(
      layers,
      ("body_sha1_set", "body_sha1s", "body_sha1", "step_sha1s", "step_sha1"),
  )
  result = _clean_text_values(value)
  if result:
    return tuple(item.lower() for item in result)
  hashes: set[str] = set()
  for layer in layers:
    parts = layer.get("parts")
    if not isinstance(parts, list):
      continue
    for part in parts:
      if not isinstance(part, dict):
        continue
      raw = (
          part.get("body_sha1")
          or part.get("step_sha1")
          or part.get("sha1")
          or part.get("content_sha1")
      )
      if isinstance(raw, str) and raw.strip():
        hashes.add(raw.strip().lower())
  return tuple(sorted(hashes))


def _graph_node_id(node: Any, index: int) -> str:
  if isinstance(node, dict):
    for key in ("id", "node_id", "body", "body_uuid", "part_id", "name"):
      value = node.get(key)
      if isinstance(value, (str, int)) and str(value).strip():
        return str(value)
  if isinstance(node, (str, int)) and str(node).strip():
    return str(node)
  return f"__node_{index}"


def _graph_node_attributes(node: Any) -> Any:
  if not isinstance(node, dict):
    return {}
  identifier_keys = {"id", "node_id", "body", "body_uuid", "part_id", "name"}
  return {key: value for key, value in node.items() if key not in identifier_keys}


def _graph_edge(edge: Any) -> tuple[str, str, Any]:
  if isinstance(edge, (list, tuple)) and len(edge) >= 2:
    return str(edge[0]), str(edge[1]), list(edge[2:])
  if isinstance(edge, dict):
    left = next(
        (edge.get(key) for key in ("source", "from", "left", "part_a", "u") if key in edge),
        None,
    )
    right = next(
        (edge.get(key) for key in ("target", "to", "right", "part_b", "v") if key in edge),
        None,
    )
    if left is None or right is None:
      raise FamilyEvidenceError("assembly_graph edge is missing both endpoints")
    endpoint_keys = {
        "source", "from", "left", "part_a", "u",
        "target", "to", "right", "part_b", "v",
    }
    attributes = {key: value for key, value in edge.items() if key not in endpoint_keys}
    return str(left), str(right), attributes
  raise FamilyEvidenceError("assembly_graph edges must be endpoint pairs or objects")


def canonical_graph_fingerprint(payload: Any) -> str:
  """Return an ID/order-invariant WL bucket, never an isomorphism proof."""

  if not isinstance(payload, dict):
    raise FamilyEvidenceError("assembly_graph must be an object")
  raw_nodes = payload.get("nodes")
  raw_edges = payload.get("edges")
  if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
    raise FamilyEvidenceError("assembly_graph requires nodes and edges lists")

  attributes_by_id: dict[str, Any] = {}
  for index, node in enumerate(raw_nodes):
    node_id = _graph_node_id(node, index)
    if node_id in attributes_by_id:
      raise FamilyEvidenceError(f"assembly_graph has duplicate node id: {node_id}")
    attributes_by_id[node_id] = _graph_node_attributes(node)

  parsed_edges: list[tuple[str, str, str]] = []
  for raw_edge in raw_edges:
    left, right, attributes = _graph_edge(raw_edge)
    attributes_by_id.setdefault(left, {})
    attributes_by_id.setdefault(right, {})
    parsed_edges.append((left, right, _canonical_json(attributes)))
  if not attributes_by_id:
    raise FamilyEvidenceError("assembly_graph must contain at least one node")

  adjacency: dict[str, list[tuple[str, str]]] = {
      node_id: [] for node_id in attributes_by_id
  }
  for left, right, edge_attributes in parsed_edges:
    adjacency[left].append((right, edge_attributes))
    if left != right:
      adjacency[right].append((left, edge_attributes))

  colors = {
      node_id: _sha256_text(_canonical_json(attributes))
      for node_id, attributes in attributes_by_id.items()
  }
  for _ in range(len(colors)):
    updated = {
        node_id: _sha256_text(
            _canonical_json(
                {
                    "self": colors[node_id],
                    "neighbors": sorted(
                        (edge_attributes, colors[neighbor])
                        for neighbor, edge_attributes in adjacency[node_id]
                    ),
                }
            )
        )
        for node_id in colors
    }
    if updated == colors:
      break
    colors = updated

  canonical_edges = sorted(
      (
          min(colors[left], colors[right]),
          max(colors[left], colors[right]),
          edge_attributes,
      )
      for left, right, edge_attributes in parsed_edges
  )
  canonical = {
      "version": GRAPH_FINGERPRINT_CANONICAL_VERSION,
      "node_colors": sorted(colors.values()),
      "edges": canonical_edges,
  }
  return "graph_wl_bucket_sha256:" + _sha256_text(_canonical_json(canonical))


def _attributed_graph_isomorphic(left_payload: Any, right_payload: Any) -> bool:
  """Return exact isomorphism after a WL fingerprint has selected a bucket."""

  def parse(
      payload: Any,
  ) -> tuple[
      dict[str, str],
      dict[tuple[str, str], tuple[str, ...]],
      dict[str, tuple[str, ...]],
  ]:
    if not isinstance(payload, dict):
      raise FamilyEvidenceError("assembly_graph must be an object")
    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
      raise FamilyEvidenceError("assembly_graph requires nodes and edges lists")
    attributes: dict[str, str] = {}
    for index, node in enumerate(raw_nodes):
      node_id = _graph_node_id(node, index)
      if node_id in attributes:
        raise FamilyEvidenceError(f"assembly_graph has duplicate node id: {node_id}")
      attributes[node_id] = _canonical_json(_graph_node_attributes(node))
    pair_attributes: dict[tuple[str, str], list[str]] = {}
    incident_attributes: dict[str, list[str]] = {
        node_id: [] for node_id in attributes
    }
    for raw_edge in raw_edges:
      left, right, edge_attributes = _graph_edge(raw_edge)
      attributes.setdefault(left, _canonical_json({}))
      attributes.setdefault(right, _canonical_json({}))
      incident_attributes.setdefault(left, [])
      incident_attributes.setdefault(right, [])
      edge_label = _canonical_json(edge_attributes)
      pair = tuple(sorted((left, right)))
      pair_attributes.setdefault(pair, []).append(edge_label)
      incident_attributes[left].append(edge_label)
      if right != left:
        incident_attributes[right].append(edge_label)
    return (
        attributes,
        {key: tuple(sorted(values)) for key, values in pair_attributes.items()},
        {key: tuple(sorted(values)) for key, values in incident_attributes.items()},
    )

  left_attributes, left_edges, left_incident = parse(left_payload)
  right_attributes, right_edges, right_incident = parse(right_payload)
  if len(left_attributes) != len(right_attributes):
    return False
  if sum(len(values) for values in left_edges.values()) != sum(
      len(values) for values in right_edges.values()
  ):
    return False

  def node_signature(
      node_id: str,
      attributes: dict[str, str],
      edges: dict[tuple[str, str], tuple[str, ...]],
      incident: dict[str, tuple[str, ...]],
  ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return (
        attributes[node_id],
        incident[node_id],
        edges.get((node_id, node_id), ()),
    )

  right_by_signature: dict[
      tuple[str, tuple[str, ...], tuple[str, ...]], list[str]
  ] = {}
  for node_id in right_attributes:
    signature = node_signature(
        node_id, right_attributes, right_edges, right_incident
    )
    right_by_signature.setdefault(signature, []).append(node_id)
  candidates: dict[str, tuple[str, ...]] = {}
  for node_id in left_attributes:
    signature = node_signature(
        node_id, left_attributes, left_edges, left_incident
    )
    matches = tuple(sorted(right_by_signature.get(signature, [])))
    if not matches:
      return False
    candidates[node_id] = matches
  ordered_left = sorted(
      left_attributes,
      key=lambda node_id: (
          len(candidates[node_id]),
          -len(left_incident[node_id]),
          node_id,
      ),
  )
  mapping: dict[str, str] = {}
  used_right: set[str] = set()

  def edges_between(
      edges: dict[tuple[str, str], tuple[str, ...]],
      first: str,
      second: str,
  ) -> tuple[str, ...]:
    return edges.get(tuple(sorted((first, second))), ())

  def search(offset: int) -> bool:
    if offset == len(ordered_left):
      return True
    left_node = ordered_left[offset]
    for right_node in candidates[left_node]:
      if right_node in used_right:
        continue
      if any(
          edges_between(left_edges, left_node, mapped_left)
          != edges_between(right_edges, right_node, mapped_right)
          for mapped_left, mapped_right in mapping.items()
      ):
        continue
      mapping[left_node] = right_node
      used_right.add(right_node)
      if search(offset + 1):
        return True
      used_right.remove(right_node)
      del mapping[left_node]
    return False

  return search(0)


def _graph_fingerprint_from_layers(layers: Sequence[dict[str, Any]]) -> str:
  explicit = _last_present(
      layers,
      (
          "graph_fingerprint",
          "assembly_graph_fingerprint",
          "case_part_graph_fingerprint",
      ),
  )
  if isinstance(explicit, str) and explicit.strip():
    return explicit.strip()
  for layer in reversed(layers):
    for key in ("assembly_graph", "graph_payload", "mate_graph"):
      payload = layer.get(key)
      if isinstance(payload, dict):
        return canonical_graph_fingerprint(payload)
  return ""


def _embedding_from_layers(
    layers: Sequence[dict[str, Any]],
) -> tuple[float, ...] | None:
  raw = _last_present(
      layers,
      ("geometry_embedding", "family_embedding", "similarity_embedding"),
  )
  if raw is None:
    return None
  if not isinstance(raw, (list, tuple)) or not raw:
    raise FamilyEvidenceError("geometry_embedding must be a non-empty numeric list")
  try:
    vector = tuple(float(value) for value in raw)
  except (TypeError, ValueError) as exc:
    raise FamilyEvidenceError("geometry_embedding must contain only numbers") from exc
  if not all(math.isfinite(value) for value in vector):
    raise FamilyEvidenceError("geometry_embedding must contain only finite numbers")
  return vector


def _descriptor_producer(
    provenance: dict[str, Any],
    *,
    descriptor_name: str,
    present: bool,
    case_id: str,
) -> str | None:
  raw = provenance.get(f"{descriptor_name}_provenance")
  if not present:
    if raw is not None:
      raise FamilyEvidenceError(
          f"Case {case_id} has {descriptor_name}_provenance without {descriptor_name}"
      )
    return None
  if not isinstance(raw, dict):
    raise FamilyEvidenceError(
        f"Case {case_id} {descriptor_name} requires producer provenance"
    )
  required = ("producer", "version", "artifact_sha256")
  missing = [key for key in required if not str(raw.get(key) or "").strip()]
  if missing:
    raise FamilyEvidenceError(
        f"Case {case_id} {descriptor_name}_provenance missing: "
        + ", ".join(missing)
    )
  artifact = str(raw["artifact_sha256"]).strip().lower()
  if not artifact.startswith("sha256:") or len(artifact) != 71 or any(
      char not in "0123456789abcdef" for char in artifact[7:]
  ):
    raise FamilyEvidenceError(
        f"Case {case_id} {descriptor_name}_provenance.artifact_sha256 is invalid"
    )
  normalized = {
      "producer": str(raw["producer"]).strip(),
      "version": str(raw["version"]).strip(),
      "artifact_sha256": artifact,
  }
  return _canonical_json(normalized)


def _synthetic_family_values(
    case: dict[str, Any],
    case_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
  assembly = str(case.get("assembly_dir") or case_id).strip()
  dataset = str(case.get("dataset_root_hint") or "development").strip()
  lineage = (f"synthetic:{dataset}:{assembly}",)
  raw_parts = case.get("parts")
  if not isinstance(raw_parts, list) or not raw_parts:
    raw_parts = case.get("candidate_parts")
  if not isinstance(raw_parts, list) or not raw_parts:
    raw_parts = [case_id]
  body_hashes = tuple(
      sorted(
          {
              hashlib.sha1(str(part).strip().encode("utf-8")).hexdigest()
              for part in raw_parts
          }
      )
  )
  graph_payload = {
      "body_sha1_set": body_hashes,
      "contact_pairs": case.get("contact_pairs") or [],
  }
  graph = "synthetic_sha256:" + _sha256_text(_canonical_json(graph_payload))
  return lineage, body_hashes, graph


def extract_family_evidence(
    case: dict[str, Any],
    *,
    case_index: int = 0,
    development_synthetic_mode: bool = False,
) -> FamilyEvidence:
  """Normalize explicit family provenance, optionally with marked dev fallbacks."""

  case_id = str(case.get("id") or "").strip()
  if not case_id:
    raise FamilyEvidenceError(f"Case at index {case_index} is missing a non-empty id")
  provenance = _authoritative_provenance(
      case,
      case_id=case_id,
      development_synthetic_mode=development_synthetic_mode,
  )
  explicit_graph = provenance.get("graph_fingerprint")
  graph_payload = provenance.get("assembly_graph")
  if explicit_graph not in (None, "") and graph_payload is not None:
    raise FamilyEvidenceError(
        f"Case {case_id} family_provenance must provide either "
        "graph_fingerprint or assembly_graph, not both"
    )
  layers = [provenance]
  lineages = _lineages_from_layers(layers)
  body_sha1s = _body_sha1s_from_layers(layers)
  graph_fingerprint = _graph_fingerprint_from_layers(layers)
  graph_fingerprint_version = str(
      provenance.get("graph_fingerprint_version") or ""
  ).strip()
  if graph_payload is not None and (
      graph_fingerprint_version != GRAPH_FINGERPRINT_CANONICAL_VERSION
  ):
    raise FamilyEvidenceError(
        f"Case {case_id} derived assembly_graph requires "
        f"graph_fingerprint_version={GRAPH_FINGERPRINT_CANONICAL_VERSION!r}"
    )
  synthetic_fields: list[str] = []
  if development_synthetic_mode and (
      not lineages or not body_sha1s or not graph_fingerprint
  ):
    synthetic_lineages, synthetic_hashes, synthetic_graph = _synthetic_family_values(
        case, case_id
    )
    if not lineages:
      lineages = synthetic_lineages
      synthetic_fields.append("source_lineage")
    if not body_sha1s:
      body_sha1s = synthetic_hashes
      synthetic_fields.append("body_sha1_set")
    if not graph_fingerprint:
      graph_fingerprint = synthetic_graph
      graph_fingerprint_version = "synthetic_graph.v1"
      synthetic_fields.append("graph_fingerprint")

  missing = []
  if not lineages:
    missing.append("source_lineage")
  if not body_sha1s:
    missing.append("body_sha1_set")
  if not graph_fingerprint:
    missing.append("graph_fingerprint")
  if not graph_fingerprint_version:
    missing.append("graph_fingerprint_version")
  if missing:
    raise FamilyEvidenceError(
        f"Case {case_id} is missing formal family evidence: " + ", ".join(missing)
    )
  invalid_hashes = [
      value
      for value in body_sha1s
      if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value)
  ]
  if invalid_hashes and not development_synthetic_mode:
    raise FamilyEvidenceError(
        f"Case {case_id} has invalid body_sha1_set values; expected 40 hex characters"
    )
  geometry_embedding = _embedding_from_layers(layers)
  shape_signature = _signature_tokens(
      _last_present(layers, ("shape_signature", "shape_signatures"))
  )
  mate_graph_signature = _signature_tokens(
      _last_present(layers, ("mate_graph_signature", "graph_signature"))
  )
  return FamilyEvidence(
      case_id=case_id,
      provenance_version=FAMILY_PROVENANCE_VERSION,
      formal=bool(provenance.get("formal") is True and not synthetic_fields),
      source_lineages=lineages,
      body_sha1s=body_sha1s,
      graph_fingerprint=graph_fingerprint,
      graph_fingerprint_version=graph_fingerprint_version,
      assembly_graph=(copy.deepcopy(graph_payload) if graph_payload is not None else None),
      geometry_embedding=geometry_embedding,
      geometry_embedding_producer=_descriptor_producer(
          provenance,
          descriptor_name="geometry_embedding",
          present=geometry_embedding is not None,
          case_id=case_id,
      ),
      shape_signature=shape_signature,
      shape_signature_producer=_descriptor_producer(
          provenance,
          descriptor_name="shape_signature",
          present=bool(shape_signature),
          case_id=case_id,
      ),
      mate_graph_signature=mate_graph_signature,
      mate_graph_signature_producer=_descriptor_producer(
          provenance,
          descriptor_name="mate_graph_signature",
          present=bool(mate_graph_signature),
          case_id=case_id,
      ),
      synthetic_fields=tuple(sorted(synthetic_fields)),
  )


def _family_id(member_evidence: Sequence[FamilyEvidence]) -> str:
  payload = {
      "case_ids": sorted(item.case_id for item in member_evidence),
      "source_lineage": sorted(
          {value for item in member_evidence for value in item.source_lineages}
      ),
      "body_sha1_set": sorted(
          {value for item in member_evidence for value in item.body_sha1s}
      ),
      "graph_fingerprints": sorted(
          {item.graph_fingerprint for item in member_evidence}
      ),
  }
  return "fam_" + _sha256_text(_canonical_json(payload))[:20]


def _stable_family_order(family_ids: Iterable[str], seed: int) -> list[str]:
  return sorted(
      family_ids,
      key=lambda family_id: (
          _sha256_text(f"{int(seed)}:{family_id}"),
          family_id,
      ),
  )


def _optimize_dev_families(
    families: dict[str, list[int]],
    *,
    seed: int,
    dev_ratio: float,
    tolerance: float,
    fail_if_impossible: bool,
) -> tuple[set[str], dict[str, Any]]:
  """Solve deterministic family-level subset sum under a declared tolerance."""

  total_cases = sum(len(indices) for indices in families.values())
  ordered = _stable_family_order(families, seed)
  reachable = bytearray(total_cases + 1)
  reachable[0] = 1
  reachable_counts = [0]
  parent_count = array("q", [-2]) * (total_cases + 1)
  parent_family = array("i", [-1]) * (total_cases + 1)
  parent_count[0] = -1
  transition_count = 0
  for family_index, family_id in enumerate(ordered):
    size = len(families[family_id])
    prior_state_count = len(reachable_counts)
    for state_index in range(prior_state_count):
      count = reachable_counts[state_index]
      transition_count += 1
      new_count = count + size
      if reachable[new_count]:
        continue
      reachable[new_count] = 1
      reachable_counts.append(new_count)
      parent_count[new_count] = count
      parent_family[new_count] = family_index

  candidates = [count for count in reachable_counts if 0 < count < total_cases]
  target_count = float(total_cases) * float(dev_ratio)
  if candidates:
    achieved_count = min(
        candidates,
        key=lambda count: (
            abs(float(count) - target_count),
            count,
        ),
    )
  else:
    achieved_count = 0
  selected_ids: list[str] = []
  cursor = achieved_count
  while cursor > 0:
    family_index = int(parent_family[cursor])
    previous = int(parent_count[cursor])
    if family_index < 0 or previous < 0:
      raise RuntimeError("Subset-sum parent chain is corrupt")
    selected_ids.append(ordered[family_index])
    cursor = previous
  selected_tuple = tuple(reversed(selected_ids))
  achieved_ratio = achieved_count / max(1, total_cases)
  absolute_error = abs(achieved_ratio - float(dev_ratio))
  within_tolerance = bool(absolute_error <= float(tolerance) + 1e-12)
  diagnostics = {
      "target_ratio": float(dev_ratio),
      "achieved_ratio": round(achieved_ratio, 12),
      "absolute_error": round(absolute_error, 12),
      "tolerance": float(tolerance),
      "within_tolerance": within_tolerance,
      "fail_if_impossible": bool(fail_if_impossible),
      "algorithm": "parent_pointer_subset_sum.v2",
      "reachable_state_count": len(reachable_counts),
      "transition_count": transition_count,
      "parent_node_count": len(reachable_counts) - 1,
      "parent_storage_slots": len(parent_count) + len(parent_family),
  }
  if fail_if_impossible and not within_tolerance:
    raise FamilyEvidenceError(
        "No nonempty train/dev family subset can achieve the requested dev ratio "
        f"within tolerance: {diagnostics}"
    )
  return set(selected_tuple), diagnostics


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float | None:
  left_set = set(left)
  right_set = set(right)
  if not left_set or not right_set:
    return None
  return len(left_set & right_set) / len(left_set | right_set)


def family_similarity(
    left: FamilyEvidence,
    right: FamilyEvidence,
) -> dict[str, Any]:
  """Return a provenance-only approximate design-family similarity."""

  components: dict[str, float] = {}
  weights: dict[str, float] = {}
  if (
      left.geometry_embedding is not None
      and right.geometry_embedding is not None
      and left.geometry_embedding_producer == right.geometry_embedding_producer
      and len(left.geometry_embedding) == len(right.geometry_embedding)
  ):
    dot = sum(a * b for a, b in zip(left.geometry_embedding, right.geometry_embedding))
    norm_left = math.sqrt(sum(value * value for value in left.geometry_embedding))
    norm_right = math.sqrt(sum(value * value for value in right.geometry_embedding))
    if norm_left > 0.0 and norm_right > 0.0:
      cosine = max(0.0, min(1.0, dot / (norm_left * norm_right)))
      components["geometry_embedding_cosine"] = cosine
      weights["geometry_embedding_cosine"] = 0.60
  shape_score = (
      _jaccard(left.shape_signature, right.shape_signature)
      if left.shape_signature_producer == right.shape_signature_producer
      else None
  )
  if shape_score is not None:
    components["shape_signature_jaccard"] = shape_score
    weights["shape_signature_jaccard"] = 0.25
  graph_score = (
      _jaccard(left.mate_graph_signature, right.mate_graph_signature)
      if left.mate_graph_signature_producer == right.mate_graph_signature_producer
      else None
  )
  if graph_score is not None:
    components["mate_graph_signature_jaccard"] = graph_score
    weights["mate_graph_signature_jaccard"] = 0.15
  total_weight = sum(weights.values())
  score = 0.0
  if total_weight > 0.0:
    score = sum(components[name] * weights[name] for name in components) / total_weight
  return {
      "score": float(score),
      "comparable": bool(components),
      "components": components,
  }


def _exact_overlap(values_by_split: dict[str, set[str]]) -> dict[str, Any]:
  pair_counts: dict[str, int] = {}
  examples: list[dict[str, Any]] = []
  total = 0
  split_names = sorted(values_by_split)
  for index, left in enumerate(split_names):
    for right in split_names[index + 1 :]:
      common = sorted(values_by_split[left] & values_by_split[right])
      pair = f"{left}__{right}"
      pair_counts[pair] = len(common)
      total += len(common)
      examples.extend({"pair": pair, "value": value} for value in common[:20])
  return {
      "pair_overlap_counts": pair_counts,
      "total_pair_overlap_count": total,
      "examples": examples[:20],
  }


def _exact_graph_overlap(
    evidence_by_split: dict[str, list[tuple[str, FamilyEvidence]]],
) -> dict[str, Any]:
  buckets_by_split: dict[
      str, dict[tuple[str, str], list[FamilyEvidence]]
  ] = {}
  for split, rows in evidence_by_split.items():
    buckets: dict[tuple[str, str], list[FamilyEvidence]] = {}
    for _, evidence in rows:
      key = (evidence.graph_fingerprint_version, evidence.graph_fingerprint)
      buckets.setdefault(key, []).append(evidence)
    buckets_by_split[split] = buckets
  pair_counts: dict[str, int] = {}
  examples: list[dict[str, Any]] = []
  total = 0
  split_names = sorted(buckets_by_split)
  for index, left_split in enumerate(split_names):
    for right_split in split_names[index + 1 :]:
      common: list[tuple[str, str]] = []
      left_buckets = buckets_by_split[left_split]
      right_buckets = buckets_by_split[right_split]
      for bucket in sorted(set(left_buckets) & set(right_buckets)):
        overlaps = any(
            (
                left.assembly_graph is None
                and right.assembly_graph is None
            )
            or (
                left.assembly_graph is not None
                and right.assembly_graph is not None
                and _attributed_graph_isomorphic(
                    left.assembly_graph, right.assembly_graph
                )
            )
            for left in left_buckets[bucket]
            for right in right_buckets[bucket]
        )
        if overlaps:
          common.append(bucket)
      pair = f"{left_split}__{right_split}"
      pair_counts[pair] = len(common)
      total += len(common)
      examples.extend(
          {
              "pair": pair,
              "value": fingerprint,
              "graph_fingerprint_version": version,
          }
          for version, fingerprint in common[:20]
      )
  return {
      "pair_overlap_counts": pair_counts,
      "total_pair_overlap_count": total,
      "examples": examples[:20],
  }


def build_family_pair_review_request(
    nearest_family_pairs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
  """Freeze the exact top-family-pair payload that a custodian must review."""

  pairs = []
  for row in nearest_family_pairs:
    pair_id = str(row.get("pair_id") or "").strip()
    if not pair_id:
      pair_id = "pair_" + _sha256_text(
          _canonical_json(
              {
                  "left_split": row.get("left_split"),
                  "right_split": row.get("right_split"),
                  "left_family_id": row.get("left_family_id"),
                  "right_family_id": row.get("right_family_id"),
              }
          )
      )[:20]
    pairs.append(
        {
            "pair_id": pair_id,
            "left_split": str(row.get("left_split") or ""),
            "right_split": str(row.get("right_split") or ""),
            "left_family_id": str(row.get("left_family_id") or ""),
            "right_family_id": str(row.get("right_family_id") or ""),
            "left_case_id": str(row.get("left_case_id") or ""),
            "right_case_id": str(row.get("right_case_id") or ""),
            "score": float(row.get("score") or 0.0),
            "components": dict(row.get("components") or {}),
        }
    )
  payload = {
      "schema_version": "family_pair_review_request.v2",
      "pair_count": len(pairs),
      "pairs": pairs,
  }
  payload["request_sha256"] = "sha256:" + _sha256_text(_canonical_json(payload))
  return payload


def verify_family_pair_review(
    request: dict[str, Any],
    artifact: dict[str, Any],
    *,
    verification_key: dict[str, Any] | str | bytes,
) -> dict[str, Any]:
  """Verify signature, request binding, exact coverage, and review decisions."""

  try:
    expected_request = build_family_pair_review_request(request.get("pairs") or [])
    if request != expected_request:
      raise FamilyEvidenceError("Family-pair review request is not canonical")
    if artifact.get("schema_version") != "family_pair_review.v2":
      raise FamilyEvidenceError("Unsupported signed family-pair review schema")
    if set(artifact) != {
        "schema_version",
        "request_sha256",
        "reviewer",
        "timestamp_utc",
        "decisions",
        "signer_key_id",
        "signature_scheme",
        "signature",
    }:
      raise FamilyEvidenceError("Signed review contains an incomplete or mutable field set")
    if artifact.get("request_sha256") != request.get("request_sha256"):
      raise FamilyEvidenceError("Signed review is bound to a different pair request")
    if artifact.get("signature_scheme") != "rsa-pkcs1v15-sha256.v1":
      raise FamilyEvidenceError("Signed review has an unsupported signature scheme")
    if not str(artifact.get("reviewer") or "").strip():
      raise FamilyEvidenceError("Signed review is missing reviewer identity")
    _require_rfc3339_utc(
        artifact.get("timestamp_utc"), field="Signed review timestamp_utc"
    )
    unsigned = {key: value for key, value in artifact.items() if key != "signature"}
    key_id, modulus, exponent = _parse_rsa_public_key(verification_key)
    if artifact.get("signer_key_id") != key_id:
      raise FamilyEvidenceError("Signed review signer key is not trusted")
    if not _verify_rsa_pkcs1v15_sha256(
        _canonical_json(unsigned).encode("utf-8"),
        str(artifact.get("signature") or ""),
        modulus=modulus,
        exponent=exponent,
    ):
      raise FamilyEvidenceError("Signed review signature verification failed")
    expected_pair_ids = [str(row["pair_id"]) for row in request["pairs"]]
    decisions = artifact.get("decisions")
    if not isinstance(decisions, list):
      raise FamilyEvidenceError("Signed review decisions must be a list")
    if any(
        not isinstance(row, dict)
        or set(row) != {"pair_id", "decision"}
        or row.get("decision") not in {"confirmed_distinct", "merge_required"}
        for row in decisions
    ):
      raise FamilyEvidenceError("Signed review decisions violate the frozen schema")
    actual_pair_ids = [str(row.get("pair_id") or "") for row in decisions]
    if actual_pair_ids != expected_pair_ids:
      raise FamilyEvidenceError("Signed review does not cover the immutable pair order")
    merge_required = [
        row["pair_id"] for row in decisions if row.get("decision") != "confirmed_distinct"
    ]
    if merge_required:
      raise FamilyEvidenceError(
          "Signed review identifies family pairs that require merging: "
          + ", ".join(merge_required)
      )
  except (FamilyEvidenceError, KeyError, TypeError, ValueError) as exc:
    return {"verified": False, "error": str(exc)}
  return {
      "verified": True,
      "reviewer": artifact.get("reviewer"),
      "timestamp_utc": artifact.get("timestamp_utc"),
      "request_sha256": request.get("request_sha256"),
      "signer_key_id": artifact.get("signer_key_id"),
      "signature_scheme": artifact.get("signature_scheme"),
  }


def audit_family_splits(
    splits: dict[str, Sequence[dict[str, Any]]],
    *,
    top_k: int = 20,
    warning_threshold: float = 0.9,
    allow_missing: bool = False,
    development_synthetic_mode: bool = False,
    required_evidence_coverage: float = 1.0,
    similarity_block_size: int = 256,
    review_artifact: dict[str, Any] | None = None,
    review_verification_key: dict[str, Any] | str | bytes | None = None,
    family_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  """Audit exact overlap and rank cross-split approximate family neighbors."""

  if int(top_k) < 0:
    raise ValueError("top_k must be non-negative")
  if not 0.0 <= float(warning_threshold) <= 1.0:
    raise ValueError("warning_threshold must be in [0, 1]")
  if not 0.0 <= float(required_evidence_coverage) <= 1.0:
    raise ValueError("required_evidence_coverage must be in [0, 1]")
  if int(similarity_block_size) <= 0:
    raise ValueError("similarity_block_size must be positive")

  family_values: dict[str, set[str]] = {}
  lineage_values: dict[str, set[str]] = {}
  body_values: dict[str, set[str]] = {}
  evidence_by_split: dict[str, list[tuple[str, FamilyEvidence]]] = {}
  missing: list[dict[str, str]] = []
  record_by_case_id: dict[str, dict[str, Any]] = {}
  if family_records is not None:
    for record_index, record in enumerate(family_records):
      if not isinstance(record, dict):
        raise FamilyEvidenceError(
            f"Family audit record at index {record_index} must be an object"
        )
      record_case_id = str(record.get("case_id") or "").strip()
      if not record_case_id:
        raise FamilyEvidenceError(
            f"Family audit record at index {record_index} is missing case_id"
        )
      if record_case_id in record_by_case_id:
        raise FamilyEvidenceError(
            f"Duplicate family audit record for case: {record_case_id}"
        )
      record_by_case_id[record_case_id] = record
  seen_case_ids: set[str] = set()
  for split, cases in splits.items():
    family_values[split] = set()
    lineage_values[split] = set()
    body_values[split] = set()
    evidence_by_split[split] = []
    for index, case in enumerate(cases):
      if not isinstance(case, dict):
        raise FamilyEvidenceError(
            f"Split {split} case at index {index} must be an object"
        )
      case_id = str(case.get("id") or "").strip()
      if not case_id:
        raise FamilyEvidenceError(f"Split {split} case at index {index} is missing id")
      if case_id in seen_case_ids:
        raise FamilyEvidenceError(f"Duplicate case id across audited splits: {case_id}")
      seen_case_ids.add(case_id)
      audit_case = case
      if family_records is not None:
        record = record_by_case_id.get(case_id)
        if record is None:
          if not allow_missing:
            raise FamilyEvidenceError(
                f"Case {case_id} has no family audit record"
            )
          missing.append(
              {"split": split, "case_id": case_id, "error": "missing family audit record"}
          )
          continue
        if str(record.get("split") or "") != split:
          raise FamilyEvidenceError(
              f"Case {case_id} family audit record names split "
              f"{record.get('split')!r}, expected {split!r}"
          )
        audit_case = dict(case)
        audit_case["family_id"] = record.get("family_id")
        audit_case["family_provenance"] = record.get("family_provenance")
      try:
        evidence = extract_family_evidence(
            audit_case,
            case_index=index,
            development_synthetic_mode=development_synthetic_mode,
        )
      except FamilyEvidenceError as exc:
        if not allow_missing:
          raise
        missing.append({"split": split, "case_id": case_id, "error": str(exc)})
        continue
      family_id = str(audit_case.get("family_id") or "").strip()
      if not family_id:
        if not allow_missing:
          raise FamilyEvidenceError(f"Case {case_id} is missing family_id")
        missing.append(
            {"split": split, "case_id": case_id, "error": "missing family_id"}
        )
        continue
      family_values[split].add(family_id)
      lineage_values[split].update(evidence.source_lineages)
      body_values[split].update(evidence.body_sha1s)
      evidence_by_split[split].append((family_id, evidence))
  unreferenced_records = sorted(set(record_by_case_id) - seen_case_ids)
  if unreferenced_records:
    raise FamilyEvidenceError(
        "Family audit records reference cases outside the audited splits: "
        + ", ".join(unreferenced_records)
    )
  overlaps = {
      "family_id": _exact_overlap(family_values),
      "source_lineage": _exact_overlap(lineage_values),
      "body_sha1": _exact_overlap(body_values),
      "graph_fingerprint": _exact_graph_overlap(evidence_by_split),
  }


  exact_clean = all(
      item["total_pair_overlap_count"] == 0 for item in overlaps.values()
  )
  neighbor_rows: list[dict[str, Any]] = []
  neighbor_limit = max(0, int(top_k))
  cross_split_pair_count = 0
  warning_count = 0
  comparable_pair_count = 0
  similarity_block_size = int(similarity_block_size)
  pairs_since_prune = 0
  peak_neighbor_buffer_size = 0

  def _neighbor_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["score"]),
        str(row["left_split"]),
        str(row["right_split"]),
        str(row["left_case_id"]),
        str(row["right_case_id"]),
    )

  split_names = sorted(evidence_by_split)
  families_by_split: dict[str, dict[str, list[FamilyEvidence]]] = {}
  for split, rows in evidence_by_split.items():
    grouped: dict[str, list[FamilyEvidence]] = {}
    for family_id, item in rows:
      grouped.setdefault(family_id, []).append(item)
    families_by_split[split] = grouped
  for split_index, left_split in enumerate(split_names):
    for right_split in split_names[split_index + 1 :]:
      for left_family_id, left_members in sorted(families_by_split[left_split].items()):
        for right_family_id, right_members in sorted(families_by_split[right_split].items()):
          cross_split_pair_count += 1
          best: tuple[float, FamilyEvidence, FamilyEvidence, dict[str, Any]] | None = None
          for left in left_members:
            for right in right_members:
              similarity = family_similarity(left, right)
              if not bool(similarity["comparable"]):
                continue
              candidate = (float(similarity["score"]), left, right, similarity)
              if best is None or (
                  candidate[0], left.case_id, right.case_id
              ) > (best[0], best[1].case_id, best[2].case_id):
                best = candidate
          if best is None:
            continue
          score, left, right, similarity = best
          comparable_pair_count += 1
          warning = bool(score >= float(warning_threshold))
          warning_count += int(warning)
          if neighbor_limit <= 0:
            continue
          pair_id = "pair_" + _sha256_text(
              _canonical_json(
                  {
                      "left_split": left_split,
                      "right_split": right_split,
                      "left_family_id": left_family_id,
                      "right_family_id": right_family_id,
                  }
              )
          )[:20]
          neighbor_rows.append(
              {
                  "pair_id": pair_id,
                  "left_split": left_split,
                  "right_split": right_split,
                  "left_family_id": left_family_id,
                  "right_family_id": right_family_id,
                  "left_case_id": left.case_id,
                  "right_case_id": right.case_id,
                  "score": round(score, 8),
                  "components": similarity["components"],
                  "warning": warning,
              }
          )
          pairs_since_prune += 1
          peak_neighbor_buffer_size = max(
              peak_neighbor_buffer_size, len(neighbor_rows)
          )
          if pairs_since_prune >= similarity_block_size:
            neighbor_rows.sort(key=_neighbor_key)
            del neighbor_rows[neighbor_limit:]
            pairs_since_prune = 0
  neighbor_rows.sort(key=_neighbor_key)
  nearest = neighbor_rows[:neighbor_limit]
  evidence_rows = [item for rows in evidence_by_split.values() for _, item in rows]
  evidence_covered = sum(item.has_near_family_descriptor for item in evidence_rows)
  evidence_coverage = evidence_covered / max(1, len(evidence_rows))
  case_coverage_met = bool(
      evidence_rows and evidence_coverage >= float(required_evidence_coverage)
  )
  comparable_pair_coverage = comparable_pair_count / max(1, cross_split_pair_count)
  pair_coverage_met = bool(
      cross_split_pair_count
      and comparable_pair_coverage >= float(required_evidence_coverage)
  )
  coverage_met = bool(case_coverage_met and pair_coverage_met)
  review_request = build_family_pair_review_request(nearest)
  if review_artifact is None or review_verification_key is None:
    review_verification = {
        "verified": False,
        "error": "missing signed review artifact or verification key",
    }
    review_status = "missing_signed_review"
  else:
    review_verification = verify_family_pair_review(
        review_request,
        review_artifact,
        verification_key=review_verification_key,
    )
    review_status = "verified" if review_verification["verified"] else "invalid"
  required_splits_nonempty = bool(
      evidence_by_split.get("train") and evidence_by_split.get("dev")
  )
  return {
      "evidence_complete": not missing,
      "missing_family_evidence": missing,
      "overlaps": overlaps,
      "nearest_neighbor_top20": nearest,
      "nearest_neighbor_config": {
          "top_k": int(top_k),
          "warning_threshold": float(warning_threshold),
          "cross_split_pair_count": cross_split_pair_count,
          "comparable_pair_count": comparable_pair_count,
          "ranking_unit": "family_pair",
          "similarity_block_size": similarity_block_size,
          "peak_neighbor_buffer_size": peak_neighbor_buffer_size,
      },
      "near_family_evidence": {
          "case_count": len(evidence_rows),
          "covered_case_count": evidence_covered,
          "coverage": round(evidence_coverage, 8),
          "case_coverage": round(evidence_coverage, 8),
          "case_coverage_met": case_coverage_met,
          "cross_split_family_pair_count": cross_split_pair_count,
          "comparable_family_pair_count": comparable_pair_count,
          "comparable_family_pair_coverage": round(comparable_pair_coverage, 8),
          "pair_coverage_met": pair_coverage_met,
          "required_coverage": float(required_evidence_coverage),
          "coverage_met": coverage_met,
      },
      "near_family_warning_count": warning_count,
      "manual_review": {
          "status": review_status,
          "required_pair_count": len(nearest),
          "request": review_request,
          "verification": review_verification,
      },
      "required_splits_nonempty": required_splits_nonempty,
      "top20_review_configured": int(top_k) == 20,
      "strict_hygiene_clean": bool(
          exact_clean
          and not missing
          and required_splits_nonempty
          and coverage_met
          and int(top_k) == 20
          and review_verification.get("verified", False)
      ),
  }


def _frozen_family_split_policy(
    settings: FamilySplitConfig,
    *,
    salt_commitment: str | None,
) -> dict[str, Any]:
  """Materialize every decision that can affect formal family assignment."""

  return {
      "schema_version": FAMILY_SPLIT_POLICY_VERSION,
      "benchmark": "benchmark_v2",
      "formal": not bool(settings.development_synthetic_mode),
      "provenance": {
          "authoritative_location": "case.family_provenance",
          "schema_version": FAMILY_PROVENANCE_VERSION,
          "schema_sha256": _family_source_schema_identity()["sha256"],
          "conflicting_shadow_provenance": "reject",
      },
      "exact_family_union": {
          "algorithm": "transitive_union_find_canonical_case_order.v2",
          "keys": ["source_lineage", "body_sha1", "external_graph_fingerprint"],
          "derived_graph_rule": "wl_bucket_then_exact_attributed_isomorphism",
      },
      "graph_fingerprint": {
          "external_version": GRAPH_FINGERPRINT_EXTERNAL_VERSION,
          "derived_version": GRAPH_FINGERPRINT_CANONICAL_VERSION,
          "algorithm": "undirected_attributed_wl_bucket_plus_exact_isomorphism.v2",
      },
      "near_family_union": {
          "enabled": settings.near_family_threshold is not None,
          "threshold": (
              None
              if settings.near_family_threshold is None
              else float(settings.near_family_threshold)
          ),
          "descriptor_producer_identity_required": True,
          "incomparable_pairs_excluded": True,
      },
      "dev_subset": {
          "algorithm": "parent_pointer_subset_sum.v2",
          "seed": int(settings.seed),
          "target_ratio": float(settings.dev_ratio),
          "ratio_tolerance": float(settings.dev_ratio_tolerance),
          "fail_if_impossible": bool(settings.fail_if_ratio_impossible),
          "required_nonempty_splits": ["train", "dev"],
      },
      "hygiene_review": {
          "ranking_unit": "family_pair",
          "top_k": int(settings.neighbor_top_k),
          "warning_threshold": float(settings.neighbor_warning_threshold),
          "required_descriptor_coverage": float(settings.required_evidence_coverage),
          "signed_immutable_review_required": True,
          "signature_scheme": "rsa-pkcs1v15-sha256.v1",
          "representative_case_ids_signed": True,
      },
      "model_input": {
          "family_and_split_metadata": "audit_manifest_only",
          "excluded_fields": [
              "family_provenance",
              "family_id",
              "family_assignment",
              "evaluation_gold_contacts",
              "private_source_binding",
          ],
          "private_evaluation_gold_sidecar": "evaluator_custodian_only",
          "private_source_binding_sidecar": "source_custodian_only",
      },
      "final_test": {
          "generated_by_builder": False,
          "generation_rule": str(settings.final_test_generation_rule),
          "salt_commitment": salt_commitment,
          "external_attestation_required_for_sealed_status": True,
          "signature_scheme": "rsa-pkcs1v15-sha256.v1",
          "full_identity_binding_required": True,
      },
  }


def _identity_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
  return {
      "schema_version": str(payload.get("schema_version") or ""),
      "sha256": "sha256:" + _sha256_text(_canonical_json(payload)),
      "payload": payload,
  }


def _family_data_identity(
    cases: Sequence[dict[str, Any]],
    family_case_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
  canonical_cases = sorted(
      (copy.deepcopy(case) for case in cases),
      key=lambda case: str(case.get("id") or ""),
  )
  canonical_records = sorted(
      (copy.deepcopy(record) for record in family_case_records),
      key=lambda record: str(record.get("case_id") or ""),
  )
  return {
      "schema_version": "family_source_data.v1",
      "case_count": len(canonical_cases),
      "cases_sha256": "sha256:" + _sha256_text(_canonical_json(canonical_cases)),
      "family_records_sha256": (
          "sha256:" + _sha256_text(_canonical_json(canonical_records))
      ),
  }


def _private_evaluation_gold_split_payload(
    split: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
  """Build a custodian-only sidecar that is never copied into model cases."""

  canonical_rows = sorted(
      (copy.deepcopy(row) for row in rows),
      key=lambda row: str(row.get("case_id") or ""),
  )
  case_ids = [str(row.get("case_id") or "") for row in canonical_rows]
  if any(not case_id for case_id in case_ids):
    raise FamilyEvidenceError("Private evaluation gold row lacks a case_id")
  if len(case_ids) != len(set(case_ids)):
    raise FamilyEvidenceError("Private evaluation gold case IDs are not unique")
  return {
      "schema_version": "benchmark_v2_private_evaluation_gold_split.v1",
      "visibility": "private_evaluator_only",
      "split": str(split),
      "case_count": len(canonical_rows),
      "cases": canonical_rows,
  }


def _private_source_split_payload(
    split: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
  """Build one custodian-only raw source receipt split artifact."""

  canonical_rows = sorted(
      (copy.deepcopy(row) for row in rows),
      key=lambda row: str(row.get("case_id") or ""),
  )
  case_ids = [str(row.get("case_id") or "") for row in canonical_rows]
  if any(not case_id for case_id in case_ids) or len(case_ids) != len(
      set(case_ids)
  ):
    raise FamilyEvidenceError("Private source split case IDs are invalid")
  return {
      "schema_version": "benchmark_v2_private_source_split.v1",
      "visibility": "private_custodian_only",
      "split": str(split),
      "case_count": len(canonical_rows),
      "cases": canonical_rows,
  }


_RAW_UUID_PATTERN = re.compile(
    r"(?i)(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])"
)
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def _validate_v2_public_split_case(case: dict[str, Any]) -> None:
  """Fail if a v2 model case retains any custodian-only source channel."""

  expected_fields = {
      "id",
      "selected_part_names",
      "selected_geometry_assets",
      "instance_identity_contract",
      "contact_pairs",
  }
  if set(case) != expected_fields:
    extras = sorted(set(case) - expected_fields)
    missing = sorted(expected_fields - set(case))
    raise FamilyEvidenceError(
        "v2 public case field boundary failed; "
        f"extra={extras}, missing={missing}"
    )

  forbidden_keys = {
      "assembly_dir",
      "parts",
      "selected_body_uuids",
      "source_receipt_binding",
      "private_source_binding",
      "evaluation_gold_contacts",
      "occurrence_path",
      "source_instance_key",
      "transform",
      "step_path",
      "body_uuid",
      "occurrence_uuid",
      "root_component_uuid",
  }

  def visit(value: Any, location: str) -> None:
    if isinstance(value, dict):
      for raw_key, child in value.items():
        key = str(raw_key)
        child_location = f"{location}.{key}" if location else key
        if key in forbidden_keys:
          raise FamilyEvidenceError(
              f"v2 public case exposes private source key: {child_location}"
          )
        visit(child, child_location)
      return
    if isinstance(value, (list, tuple)):
      for index, child in enumerate(value):
        visit(child, f"{location}[{index}]")
      return
    if isinstance(value, str):
      lowered = value.casefold()
      if (
          _RAW_UUID_PATTERN.search(value)
          or _WINDOWS_ABSOLUTE_PATTERN.match(value)
          or value.startswith(("/", "\\"))
          or lowered.startswith("raw-")
          or lowered.endswith((".step", ".stp", ".smt", ".obj"))
      ):
        raise FamilyEvidenceError(
            f"v2 public case exposes a raw source identity at {location}"
        )

  visit(case, "")


def _validate_public_manifest_boundary(manifest: dict[str, Any]) -> None:
  """Reject custodian source identities anywhere in the public manifest."""

  forbidden_keys = {
      "assembly_dir",
      "parts",
      "selected_body_uuids",
      "source_receipt_binding",
      "occurrence_path",
      "source_instance_key",
      "transform",
      "step_path",
      "body_uuid",
      "occurrence_uuid",
      "root_component_uuid",
  }

  def visit(value: Any, location: str) -> None:
    if isinstance(value, dict):
      for raw_key, child in value.items():
        key = str(raw_key)
        child_location = f"{location}.{key}" if location else key
        if key in forbidden_keys:
          raise FamilyEvidenceError(
              f"Public manifest exposes private source key: {child_location}"
          )
        visit(child, child_location)
      return
    if isinstance(value, (list, tuple)):
      for index, child in enumerate(value):
        visit(child, f"{location}[{index}]")
      return
    if isinstance(value, str):
      lowered = value.casefold()
      if (
          _RAW_UUID_PATTERN.search(value)
          or _WINDOWS_ABSOLUTE_PATTERN.match(value)
          or value.startswith(("/", "\\"))
          or lowered.startswith("raw-")
          or lowered.endswith((".step", ".stp", ".smt", ".obj"))
      ):
        raise FamilyEvidenceError(
            f"Public manifest exposes a raw source identity at {location}"
        )

  source_document = manifest.get("source_document")
  if not isinstance(source_document, dict) or set(source_document) != {
      "schema_version",
      "metadata_sha256",
      "private_sidecars_excluded",
  }:
    raise FamilyEvidenceError("Public source_document boundary is invalid")
  if source_document.get("private_sidecars_excluded") is not True:
    raise FamilyEvidenceError("Public source_document must exclude private sidecars")
  _validated_sha256_identity(
      source_document.get("metadata_sha256"),
      field="source_document.metadata_sha256",
  )
  visit(manifest, "")


def _parse_rsa_public_key(
    value: dict[str, Any] | str | bytes,
) -> tuple[str, int, int]:
  if isinstance(value, dict):
    payload = value
  else:
    raw = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
      raise FamilyEvidenceError("Trusted verification key must be a JSON object")
    payload = parsed
  if set(payload) != {
      "schema_version",
      "key_id",
      "algorithm",
      "public_exponent",
      "modulus_hex",
  }:
    raise FamilyEvidenceError("Trusted verification key has an invalid field set")
  if payload.get("schema_version") != "rsa_public_key.v1":
    raise FamilyEvidenceError("Trusted verification key schema is unsupported")
  if payload.get("algorithm") != "rsa-pkcs1v15-sha256":
    raise FamilyEvidenceError("Trusted verification key algorithm is unsupported")
  key_id = str(payload.get("key_id") or "").strip()
  modulus_hex = str(payload.get("modulus_hex") or "").strip().lower()
  try:
    modulus = int(modulus_hex, 16)
    exponent = int(payload.get("public_exponent"))
  except (TypeError, ValueError) as exc:
    raise FamilyEvidenceError("Trusted RSA public key is malformed") from exc
  if not key_id:
    raise FamilyEvidenceError("Trusted RSA public key requires key_id")
  if (
      len(modulus_hex) < 512
      or len(modulus_hex) % 2
      or any(char not in "0123456789abcdef" for char in modulus_hex)
      or modulus.bit_length() < 2048
      or modulus.bit_length() > 8192
  ):
    raise FamilyEvidenceError("Trusted RSA modulus must be between 2048 and 8192 bits")
  if exponent < 3 or exponent > 0xFFFFFFFF or exponent % 2 == 0:
    raise FamilyEvidenceError("Trusted RSA public exponent is invalid")
  return key_id, modulus, exponent


def _verify_rsa_pkcs1v15_sha256(
    message: bytes,
    signature_text: str,
    *,
    modulus: int,
    exponent: int,
) -> bool:
  if not re.fullmatch(r"[A-Za-z0-9_-]+", signature_text):
    return False
  try:
    signature = base64.urlsafe_b64decode(
        signature_text + "=" * (-len(signature_text) % 4)
    )
  except (binascii.Error, ValueError, TypeError):
    return False
  encoded_size = (modulus.bit_length() + 7) // 8
  if len(signature) != encoded_size:
    return False
  signature_integer = int.from_bytes(signature, "big")
  if signature_integer >= modulus:
    return False
  encoded = pow(signature_integer, exponent, modulus).to_bytes(encoded_size, "big")
  digest_info = bytes.fromhex("3031300d060960864801650304020105000420")
  digest_info += hashlib.sha256(message).digest()
  padding_size = encoded_size - len(digest_info) - 3
  if padding_size < 8:
    return False
  expected = b"\x00\x01" + b"\xff" * padding_size + b"\x00" + digest_info
  return hmac.compare_digest(encoded, expected)


def _require_rfc3339_utc(value: Any, *, field: str) -> str:
  text = str(value or "").strip()
  if not text.endswith("Z"):
    raise FamilyEvidenceError(f"{field} must be RFC3339 UTC")
  try:
    parsed = datetime.fromisoformat(text[:-1] + "+00:00")
  except ValueError as exc:
    raise FamilyEvidenceError(f"{field} must be RFC3339 UTC") from exc
  if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
    raise FamilyEvidenceError(f"{field} must be RFC3339 UTC")
  return text


def verify_external_final_test_attestation(
    artifact: dict[str, Any] | None,
    *,
    policy_identity: dict[str, Any],
    schema_identity: dict[str, Any],
    code_identity: dict[str, Any],
    data_identity: dict[str, Any],
    verification_key: dict[str, Any] | str | bytes | None,
) -> dict[str, Any]:
  if artifact is None:
    return {"verified": False, "status": "missing_external_attestation"}
  if verification_key is None:
    return {
        "verified": False,
        "status": "invalid_external_attestation",
        "error": "External final-test attestation requires a trusted public key",
    }
  try:
    allowed = {
        "schema_version",
        "source_pool_sha256",
        "policy_identity",
        "provenance_schema_identity",
        "code_identity",
        "data_identity",
        "custodian",
        "timestamp_utc",
        "signer_key_id",
        "signature_scheme",
        "signature",
    }
    if not isinstance(artifact, dict) or set(artifact) != allowed:
      raise FamilyEvidenceError(
          "External final-test attestation must contain exactly the frozen fields"
      )
    if artifact.get("schema_version") != "external_final_test_attestation.v2":
      raise FamilyEvidenceError("External final-test attestation schema is unsupported")
    _validated_sha256_identity(
        artifact.get("source_pool_sha256"), field="source_pool_sha256"
    )
    if not str(artifact.get("custodian") or "").strip():
      raise FamilyEvidenceError("External final-test attestation requires custodian")
    _require_rfc3339_utc(
        artifact.get("timestamp_utc"),
        field="External final-test attestation timestamp_utc",
    )
    if artifact.get("signature_scheme") != "rsa-pkcs1v15-sha256.v1":
      raise FamilyEvidenceError("External final-test signature scheme is unsupported")
    key_id, modulus, exponent = _parse_rsa_public_key(verification_key)
    if artifact.get("signer_key_id") != key_id:
      raise FamilyEvidenceError("External final-test signer key is not trusted")
    expected = {
        "policy_identity": policy_identity,
        "provenance_schema_identity": schema_identity,
        "code_identity": code_identity,
        "data_identity": data_identity,
    }
    mismatched = [key for key, value in expected.items() if artifact[key] != value]
    if mismatched:
      raise FamilyEvidenceError(
          "External final-test attestation identity mismatch: "
          + ", ".join(mismatched)
      )
    unsigned = {key: value for key, value in artifact.items() if key != "signature"}
    if not _verify_rsa_pkcs1v15_sha256(
        _canonical_json(unsigned).encode("utf-8"),
        str(artifact.get("signature") or ""),
        modulus=modulus,
        exponent=exponent,
    ):
      raise FamilyEvidenceError("External final-test signature verification failed")
  except (FamilyEvidenceError, KeyError, TypeError, ValueError) as exc:
    return {
        "verified": False,
        "status": "invalid_external_attestation",
        "error": str(exc),
    }
  return {
      "verified": True,
      "status": "verified_external_attestation",
      "source_pool_sha256": artifact["source_pool_sha256"],
      "custodian": artifact["custodian"],
      "timestamp_utc": artifact["timestamp_utc"],
      "signer_key_id": artifact["signer_key_id"],
      "signature_scheme": artifact["signature_scheme"],
      "artifact_sha256": "sha256:" + _sha256_text(_canonical_json(artifact)),
  }


def build_family_splits(
    cases: Sequence[dict[str, Any]],
    *,
    config: FamilySplitConfig | None = None,
) -> dict[str, Any]:
  """Build benchmark-v2 train/dev splits from case-level provenance."""

  settings = config or FamilySplitConfig()
  if not 0.0 <= float(settings.dev_ratio) < 1.0:
    raise ValueError("dev_ratio must be in [0, 1)")
  if not 0.0 <= float(settings.dev_ratio_tolerance) <= 1.0:
    raise ValueError("dev_ratio_tolerance must be in [0, 1]")
  if not 0.0 <= float(settings.neighbor_warning_threshold) <= 1.0:
    raise ValueError("neighbor_warning_threshold must be in [0, 1]")
  if not 0.0 <= float(settings.required_evidence_coverage) <= 1.0:
    raise ValueError("required_evidence_coverage must be in [0, 1]")
  if int(settings.neighbor_top_k) < 0:
    raise ValueError("neighbor_top_k must be non-negative")
  if settings.near_family_threshold is not None and not (
      0.0 <= float(settings.near_family_threshold) <= 1.0
  ):
    raise ValueError("near_family_threshold must be in [0, 1]")
  if not str(settings.final_test_generation_rule).strip():
    raise FamilyEvidenceError("final_test_generation_rule must be non-empty")
  salt_commitment = _validated_salt_commitment(
      settings.final_test_salt_commitment
  )

  if not cases:
    raise FamilyEvidenceError("At least one input case is required")
  evidence = [
      extract_family_evidence(
          case,
          case_index=index,
          development_synthetic_mode=bool(settings.development_synthetic_mode),
      )
      for index, case in enumerate(cases)
  ]
  id_counts = Counter(item.case_id for item in evidence)
  duplicates = sorted(case_id for case_id, count in id_counts.items() if count > 1)
  if duplicates:
    raise FamilyEvidenceError("Duplicate case ids: " + ", ".join(duplicates))

  union_find = _UnionFind(len(evidence))
  merge_edges: list[dict[str, Any]] = []
  canonical_indices = sorted(
      range(len(evidence)), key=lambda index: evidence[index].case_id
  )
  for evidence_type, values_for_case in (
      ("source_lineage", [item.source_lineages for item in evidence]),
      ("body_sha1", [item.body_sha1s for item in evidence]),
  ):
    owner: dict[str, int] = {}
    for index in canonical_indices:
      for value in sorted(values_for_case[index]):
        previous = owner.get(value)
        if previous is None:
          owner[value] = index
          continue
        if union_find.union(previous, index):
          left_case_id, right_case_id = sorted(
              (evidence[previous].case_id, evidence[index].case_id)
          )
          merge_edges.append(
              {
                  "left_case_id": left_case_id,
                  "right_case_id": right_case_id,
                  "reason": evidence_type,
                  "value": value,
              }
          )

  external_graph_owner: dict[tuple[str, str], int] = {}
  graph_bucket_representatives: dict[tuple[str, str], list[int]] = {}
  for index in canonical_indices:
    item = evidence[index]
    bucket = (item.graph_fingerprint_version, item.graph_fingerprint)
    if item.assembly_graph is None:
      previous = external_graph_owner.get(bucket)
      if previous is None:
        external_graph_owner[bucket] = index
        continue
    else:
      previous = next(
          (
              representative
              for representative in graph_bucket_representatives.get(bucket, [])
              if _attributed_graph_isomorphic(
                  evidence[representative].assembly_graph,
                  item.assembly_graph,
              )
          ),
          None,
      )
      if previous is None:
        graph_bucket_representatives.setdefault(bucket, []).append(index)
        continue
    if union_find.union(previous, index):
      left_case_id, right_case_id = sorted(
          (evidence[previous].case_id, item.case_id)
      )
      merge_edges.append(
          {
              "left_case_id": left_case_id,
              "right_case_id": right_case_id,
              "reason": "graph_isomorphism",
              "value": item.graph_fingerprint,
              "bucket_version": item.graph_fingerprint_version,
          }
      )

  if settings.near_family_threshold is not None:
    threshold = float(settings.near_family_threshold)
    for left_offset, left in enumerate(canonical_indices):
      for right in canonical_indices[left_offset + 1 :]:
        similarity = family_similarity(evidence[left], evidence[right])
        if not similarity["comparable"] or float(similarity["score"]) < threshold:
          continue
        if union_find.union(left, right):
          merge_edges.append(
              {
                  "left_case_id": evidence[left].case_id,
                  "right_case_id": evidence[right].case_id,
                  "reason": "near_family_similarity",
                  "score": round(float(similarity["score"]), 8),
                  "components": similarity["components"],
                  "threshold": threshold,
              }
          )
  merge_edges.sort(
      key=lambda row: (
          str(row.get("left_case_id") or ""),
          str(row.get("right_case_id") or ""),
          str(row.get("reason") or ""),
          _canonical_json(row),
      )
  )

  members_by_root: dict[int, list[int]] = {}
  for index in range(len(evidence)):
    members_by_root.setdefault(union_find.find(index), []).append(index)
  families: dict[str, list[int]] = {}
  family_for_index: dict[int, str] = {}
  for indices in members_by_root.values():
    family_id = _family_id([evidence[index] for index in indices])
    families[family_id] = sorted(indices, key=lambda index: evidence[index].case_id)
    for index in indices:
      family_for_index[index] = family_id

  dev_families, dev_subset_optimization = _optimize_dev_families(
      families,
      seed=int(settings.seed),
      dev_ratio=float(settings.dev_ratio),
      tolerance=float(settings.dev_ratio_tolerance),
      fail_if_impossible=bool(
          settings.fail_if_ratio_impossible
          and not settings.development_synthetic_mode
      ),
  )
  split_cases: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
  private_evaluation_gold_rows: dict[str, list[dict[str, Any]]] = {
      "train": [],
      "dev": [],
  }
  private_source_rows: dict[str, list[dict[str, Any]]] = {
      "train": [],
      "dev": [],
  }
  family_case_records: list[dict[str, Any]] = []
  for index, source_case in enumerate(cases):
    family_id = family_for_index[index]
    split = "dev" if family_id in dev_families else "train"
    case = copy.deepcopy(source_case)
    for audit_only_key in ("family_id", "family_provenance", "family_assignment"):
      case.pop(audit_only_key, None)
    private_gold = case.pop("evaluation_gold_contacts", None)
    if private_gold is not None:
      if not isinstance(private_gold, dict):
        raise FamilyEvidenceError(
            f"Case {evidence[index].case_id} evaluation_gold_contacts must be "
            "an object"
        )
      if private_gold.get("visibility") != "private_evaluation_only":
        raise FamilyEvidenceError(
            f"Case {evidence[index].case_id} evaluation gold lacks its private "
            "visibility marker"
        )
      private_evaluation_gold_rows[split].append(
          {
              "case_id": evidence[index].case_id,
              "evaluation_gold_contacts": copy.deepcopy(private_gold),
          }
      )
    private_source = case.pop("private_source_binding", None)
    if private_source is not None:
      if (
          not isinstance(private_source, dict)
          or private_source.get("schema_version")
          != "benchmark_v2_private_source_bindings.v1"
          or private_source.get("visibility") != "private_custodian_only"
          or not isinstance(private_source.get("source"), dict)
      ):
        raise FamilyEvidenceError(
            f"Case {evidence[index].case_id} private source binding is invalid"
        )
      source_row = copy.deepcopy(private_source["source"])
      if source_row.get("case_id") != evidence[index].case_id:
        raise FamilyEvidenceError("Private source binding case join changed")
      private_source_rows[split].append(source_row)
      _validate_v2_public_split_case(case)
    split_cases[split].append(case)
    normalized_provenance = evidence[index].to_dict()
    if not settings.development_synthetic_mode:
      # Preserve graph payload/version consistency in the authoritative audit record.
      normalized_provenance = copy.deepcopy(source_case["family_provenance"])
    family_case_records.append(
        {
            "case_id": evidence[index].case_id,
            "split": split,
            "family_id": family_id,
            "family_provenance": normalized_provenance,
        }
    )
  for split in split_cases:
    split_cases[split].sort(key=lambda case: str(case.get("id") or ""))
  family_case_records.sort(key=lambda row: str(row["case_id"]))
  private_evaluation_gold = {
      split: _private_evaluation_gold_split_payload(
          split,
          private_evaluation_gold_rows[split],
      )
      for split in ("train", "dev")
  }
  private_source_bindings = {
      split: _private_source_split_payload(
          split,
          private_source_rows[split],
      )
      for split in ("train", "dev")
  }

  audit = audit_family_splits(
      split_cases,
      top_k=int(settings.neighbor_top_k),
      warning_threshold=float(settings.neighbor_warning_threshold),
      development_synthetic_mode=bool(settings.development_synthetic_mode),
      required_evidence_coverage=float(settings.required_evidence_coverage),
      review_artifact=settings.family_pair_review_artifact,
      review_verification_key=settings.family_pair_review_verification_key,
      family_records=family_case_records,
  )
  groups = [
      {
          "family_id": family_id,
          "split": "dev" if family_id in dev_families else "train",
          "case_ids": [evidence[index].case_id for index in indices],
          "case_count": len(indices),
      }
      for family_id, indices in sorted(families.items())
  ]
  schema_identity = _family_source_schema_identity()
  code_identity = _code_identity()
  frozen_policy = _frozen_family_split_policy(
      settings, salt_commitment=salt_commitment
  )
  policy_identity = _identity_for_payload(frozen_policy)
  data_identity = _family_data_identity(cases, family_case_records)
  final_attestation = verify_external_final_test_attestation(
      settings.external_final_test_attestation,
      policy_identity=policy_identity,
      schema_identity=schema_identity,
      code_identity=code_identity,
      data_identity=data_identity,
      verification_key=settings.external_final_test_verification_key,
  )
  final_status = (
      "sealed_not_generated_external_attestation"
      if final_attestation["verified"]
      else "not_generated_unsealed"
  )
  manifest = {
      "schema_version": 1,
      "benchmark": "benchmark_v2",
      "formal": not settings.development_synthetic_mode,
      "split_policy": "transitive_family_disjoint_train_dev_only",
      "seed": int(settings.seed),
      "dev_ratio": float(settings.dev_ratio),
      "dev_subset_optimization": dev_subset_optimization,
      "input_case_count": len(cases),
      "family_count": len(families),
      "synthetic_case_count": sum(1 for item in evidence if item.synthetic),
      "split_case_counts": {
          split: len(items) for split, items in split_cases.items()
      },
      "split_family_counts": {
          "train": len(families) - len(dev_families),
          "dev": len(dev_families),
      },
      "required_family_evidence": [
          "source_lineage",
          "body_sha1_set",
          "graph_fingerprint",
      ],
      "family_case_records": family_case_records,
      "family_groups": groups,
      "merge_edges": merge_edges,
      "frozen_policy_identity": policy_identity,
      "family_provenance_schema_identity": schema_identity,
      "code_identity": code_identity,
      "data_identity": data_identity,
      "private_evaluation_gold": {
          "structurally_isolated_from_model_cases": True,
          "schema_version": "benchmark_v2_private_evaluation_gold_split.v1",
          "splits": {
              split: {
                  "case_count": int(private_evaluation_gold[split]["case_count"]),
                  "sha256": "sha256:"
                  + _sha256_text(_canonical_json(private_evaluation_gold[split])),
              }
              for split in ("train", "dev")
          },
      },
      "private_source_bindings": {
          "structurally_isolated_from_model_cases": True,
          "schema_version": "benchmark_v2_private_source_split.v1",
          "splits": {
              split: {
                  "case_count": int(
                      private_source_bindings[split]["case_count"]
                  ),
                  "sha256": "sha256:"
                  + _sha256_text(
                      _canonical_json(private_source_bindings[split])
                  ),
              }
              for split in ("train", "dev")
          },
      },
      "final_test": {
          "status": final_status,
          "generated": False,
          "case_count": None,
          "generation_rule": settings.final_test_generation_rule,
          "salt_commitment": (
              salt_commitment or "pending_external_escrow"
          ),
          "external_attestation": final_attestation,
      },
  }
  return {
      "train": split_cases["train"],
      "dev": split_cases["dev"],
      "manifest": manifest,
      "audit": audit,
      "private_evaluation_gold": private_evaluation_gold,
      "private_source_bindings": private_source_bindings,
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description=(
          "Build provenance-driven benchmark-v2 train/dev family splits. "
          "This command never generates a final-test split."
      )
  )
  parser.add_argument("--input_cases", "--input-cases", required=True)
  parser.add_argument(
      "--output_dir",
      "--output-dir",
      default="neurocad/benchmark_v2",
  )
  parser.add_argument(
      "--private_evaluation_output_dir",
      "--private-evaluation-output-dir",
      default="",
      help=(
          "Required when source cases contain evaluation gold. Must be a "
          "separate custodian directory outside the public split directory."
      ),
  )
  parser.add_argument(
      "--private_source_output_dir",
      "--private-source-output-dir",
      default="",
      help=(
          "Required when source cases contain private raw source bindings. "
          "Must be a separate custodian directory outside both the public "
          "split directory and private evaluation directory."
      ),
  )
  parser.add_argument("--seed", type=int, default=2027)
  parser.add_argument("--dev_ratio", "--dev-ratio", type=float, default=0.2)
  parser.add_argument(
      "--dev_ratio_tolerance",
      "--dev-ratio-tolerance",
      type=float,
      default=0.05,
  )
  parser.add_argument(
      "--allow_ratio_outside_tolerance",
      "--allow-ratio-outside-tolerance",
      action="store_true",
      help="Development-only escape hatch; formal construction should fail impossible ratios.",
  )
  parser.add_argument(
      "--near_family_threshold",
      "--near-family-threshold",
      type=float,
      default=None,
      help="Optional provenance-similarity threshold for conservative family merging.",
  )
  parser.add_argument(
      "--neighbor_top_k",
      "--neighbor-top-k",
      type=int,
      default=20,
  )
  parser.add_argument(
      "--neighbor_warning_threshold",
      "--neighbor-warning-threshold",
      type=float,
      default=0.9,
  )
  parser.add_argument(
      "--required_evidence_coverage",
      "--required-evidence-coverage",
      type=float,
      default=1.0,
  )
  parser.add_argument(
      "--family_pair_review",
      "--family-pair-review",
      default="",
  )
  parser.add_argument(
      "--review_verification_key_file",
      "--review-verification-key-file",
      default="",
      help="Trusted RSA public-key JSON for the independent family-pair review.",
  )
  parser.add_argument(
      "--development_synthetic_mode",
      "--development-synthetic-mode",
      action="store_true",
      help=(
          "Permit visibly marked synthetic fallback provenance for hermetic "
          "development only. Output is non-formal."
      ),
  )
  parser.add_argument(
      "--final_test_generation_rule",
      "--final-test-generation-rule",
      default=FamilySplitConfig.final_test_generation_rule,
  )
  parser.add_argument(
      "--final_test_salt_commitment",
      "--final-test-salt-commitment",
      default=None,
      help="Public commitment only; never pass or record the secret final-test salt.",
  )
  parser.add_argument(
      "--external_final_test_attestation",
      "--external-final-test-attestation",
      default="",
      help="Externally signed custodian attestation; never a held-out payload.",
  )
  parser.add_argument(
      "--external_final_test_verification_key_file",
      "--external-final-test-verification-key-file",
      default="",
      help="Trusted RSA public-key JSON used only to verify the external attestation.",
  )
  return parser


def _name_tokens(value: Any) -> set[str]:
  text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
  return {
      token
      for token in re.split(r"[^a-z0-9]+", text.lower())
      if token
  }


def _forbidden_source_exposures(payload: Any) -> list[str]:
  """Find nested raw-salt and final/test payload/file variants."""

  exposed: list[str] = []

  def visit(value: Any, location: str) -> None:
    if isinstance(value, dict):
      for raw_key, child in value.items():
        key = str(raw_key)
        tokens = _name_tokens(key)
        child_location = f"{location}.{key}" if location else key
        if "salt" in tokens:
          exposed.append(child_location)
        if tokens & {"test", "final", "finaltest"}:
          exposed.append(child_location)
        visit(child, child_location)
      return
    if isinstance(value, list):
      for index, child in enumerate(value):
        visit(child, f"{location}[{index}]")
      return
    if isinstance(value, str):
      suffixes = {".json", ".jsonl", ".step", ".stp", ".zip", ".tar", ".gz"}
      candidate = Path(value.replace("\\", "/")).name
      if any(candidate.lower().endswith(suffix) for suffix in suffixes):
        if _name_tokens(candidate) & {"test", "final", "finaltest"}:
          exposed.append(location + "=<forbidden-file-variant>")

  visit(payload, "")
  return sorted(set(exposed))


def _normalized_top_level_evaluation_gold(
    payload: Any,
    *,
    source_case_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  """Validate and index the producer's custodian-only top-level gold sidecar.

  The authoritative Fusion source deliberately keeps labels outside model cases.
  ``build_family_splits`` has a single case-oriented isolation boundary, so the
  loader joins each row to its case only in memory.  The split builder removes
  it again before any public train/dev case is materialized.
  """

  if not isinstance(payload, dict):
    raise FamilyEvidenceError("evaluation_gold_contacts must be an object")
  expected_fields = {
      "schema_version",
      "visibility",
      "storage_boundary",
      "multiplicity_semantics",
      "face_identity_status",
      "gold_interface_certificate_ready",
      "blocking_reason",
      "cases",
  }
  if set(payload) != expected_fields:
    raise FamilyEvidenceError(
        "evaluation_gold_contacts has an invalid field set"
    )
  expected_values = {
      "schema_version": "benchmark_v2_evaluation_gold_contacts.v2",
      "visibility": "private_evaluation_only",
      "storage_boundary": "family_source_top_level_sidecar_not_model_case",
      "multiplicity_semantics": (
          "one_record_per_fusion_source_contact_instance_pair"
      ),
      "face_identity_status": "fusion_source_index_unverified_step_bijection",
      "blocking_reason": "fusion_to_step_face_identity_bijection_not_verified",
  }
  for field, expected in expected_values.items():
    if payload.get(field) != expected:
      raise FamilyEvidenceError(
          f"evaluation_gold_contacts.{field} is unsupported"
      )
  if payload.get("gold_interface_certificate_ready") is not False:
    raise FamilyEvidenceError(
        "Unverified Fusion face labels cannot claim certificate readiness"
    )

  rows = payload.get("cases")
  if not isinstance(rows, list) or not rows:
    raise FamilyEvidenceError("evaluation_gold_contacts.cases must be non-empty")
  expected_case_ids = list(source_case_ids)
  if any(not case_id for case_id in expected_case_ids):
    raise FamilyEvidenceError("Every source case must have a non-empty id")
  if len(expected_case_ids) != len(set(expected_case_ids)):
    raise FamilyEvidenceError("Source case IDs are not unique")

  indexed: dict[str, dict[str, Any]] = {}
  contact_ids: set[str] = set()
  for row in rows:
    if not isinstance(row, dict) or set(row) != {
        "case_id",
        "source_statistics",
        "contacts",
    }:
      raise FamilyEvidenceError(
          "Every evaluation gold case must contain only case_id, "
          "source_statistics, and contacts"
      )
    case_id = str(row.get("case_id") or "")
    if (
        not re.fullmatch(r"fusionv2_[0-9a-f]{24}", case_id)
        or case_id in indexed
    ):
      raise FamilyEvidenceError("Evaluation gold case IDs are empty or duplicated")
    statistics = row.get("source_statistics")
    contacts = row.get("contacts")
    if not isinstance(statistics, dict) or set(statistics) != {
        "audit_performed",
        "source_contact_count",
        "gold_contact_count",
        "excluded_contact_count",
        "exclusions",
    }:
      raise FamilyEvidenceError(
          f"Evaluation gold case {case_id} has invalid source_statistics"
      )
    if not isinstance(contacts, list) or not contacts:
      raise FamilyEvidenceError(
          f"Evaluation gold case {case_id} must contain at least one contact"
      )
    source_count = statistics.get("source_contact_count")
    gold_count = statistics.get("gold_contact_count")
    excluded_count = statistics.get("excluded_contact_count")
    exclusions = statistics.get("exclusions")
    if (
        statistics.get("audit_performed") is not True
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (source_count, gold_count, excluded_count)
        )
        or int(source_count) < 1
        or int(gold_count) < 1
        or int(excluded_count) < 0
        or not isinstance(exclusions, dict)
        or any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for reason, count in (
                exclusions.items() if isinstance(exclusions, dict) else ()
            )
        )
        or int(gold_count) != len(contacts)
        or int(source_count) != int(gold_count) + int(excluded_count)
        or sum(exclusions.values()) != int(excluded_count)
    ):
      raise FamilyEvidenceError(
          f"Evaluation gold case {case_id} statistics accounting is invalid"
      )
    source_ordinals: set[int] = set()
    for contact in contacts:
      if not isinstance(contact, dict) or set(contact) != {
          "contact_id",
          "source_contact_ordinal",
          "endpoint_a",
          "endpoint_b",
      }:
        raise FamilyEvidenceError(
            f"Evaluation gold case {case_id} has an invalid contact field set"
        )
      contact_id = str(contact.get("contact_id") or "")
      if not re.fullmatch(r"gold_contact_[0-9a-f]{24}", contact_id):
        raise FamilyEvidenceError(
            f"Evaluation gold case {case_id} has an invalid contact_id"
        )
      if contact_id in contact_ids:
        raise FamilyEvidenceError("Evaluation gold contact IDs are not unique")
      contact_ids.add(contact_id)
      ordinal = contact.get("source_contact_ordinal")
      if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise FamilyEvidenceError(
            f"Evaluation gold case {case_id} has an invalid contact ordinal"
        )
      if ordinal >= int(source_count):
        raise FamilyEvidenceError(
            f"Evaluation gold case {case_id} contact ordinal is out of range"
        )
      if ordinal in source_ordinals:
        raise FamilyEvidenceError(
            f"Evaluation gold case {case_id} source contact ordinals must be unique"
        )
      source_ordinals.add(ordinal)
      for endpoint_name in ("endpoint_a", "endpoint_b"):
        endpoint = contact.get(endpoint_name)
        if not isinstance(endpoint, dict) or set(endpoint) != {
            "part",
            "geometry_asset",
            "fusion_face_index",
            "entity_type",
            "surface_type",
        }:
          raise FamilyEvidenceError(
              f"Evaluation gold case {case_id} has an invalid {endpoint_name}"
          )
        if not re.fullmatch(r"part_[0-9]{3,}", str(endpoint.get("part") or "")):
          raise FamilyEvidenceError(
              f"Evaluation gold case {case_id} has an invalid endpoint part"
          )
        if not re.fullmatch(
            r"geometry_[0-9]{3,}",
            str(endpoint.get("geometry_asset") or ""),
        ):
          raise FamilyEvidenceError(
              f"Evaluation gold case {case_id} has an invalid geometry asset"
          )
        face_index = endpoint.get("fusion_face_index")
        if (
            isinstance(face_index, bool)
            or not isinstance(face_index, int)
            or face_index < 0
        ):
          raise FamilyEvidenceError(
              f"Evaluation gold case {case_id} has an invalid Fusion face index"
          )
        if endpoint.get("entity_type") != "BRepFace" or not str(
            endpoint.get("surface_type") or ""
        ).strip():
          raise FamilyEvidenceError(
              f"Evaluation gold case {case_id} has invalid endpoint semantics"
          )
      if contact["endpoint_a"]["part"] == contact["endpoint_b"]["part"]:
        raise FamilyEvidenceError(
            f"Evaluation gold case {case_id} has a same-instance contact"
        )
    indexed[case_id] = {
        "schema_version": payload["schema_version"],
        "visibility": payload["visibility"],
        "multiplicity_semantics": payload["multiplicity_semantics"],
        "face_identity_status": payload["face_identity_status"],
        "gold_interface_certificate_ready": False,
        "blocking_reason": payload["blocking_reason"],
        "source_statistics": copy.deepcopy(statistics),
        "contacts": copy.deepcopy(contacts),
    }

  if set(indexed) != set(expected_case_ids):
    missing = sorted(set(expected_case_ids) - set(indexed))
    extra = sorted(set(indexed) - set(expected_case_ids))
    raise FamilyEvidenceError(
        "Evaluation gold case set must exactly match source cases; "
        f"missing={missing[:10]}, extra={extra[:10]}"
    )
  receipt = {
      "schema_version": str(payload["schema_version"]),
      "case_count": len(indexed),
      "sha256": "sha256:" + _sha256_text(_canonical_json(payload)),
  }
  return indexed, receipt


def _safe_private_source_path(value: Any, *, suffix: str) -> str:
  if (
      not isinstance(value, str)
      or not value
      or "\\" in value
      or ":" in value
  ):
    raise FamilyEvidenceError("private source contains an unsafe member path")
  parsed = PurePosixPath(value)
  if (
      parsed.is_absolute()
      or any(part in {"", ".", ".."} for part in parsed.parts)
      or parsed.suffix.lower() != suffix
      or parsed.as_posix() != value
  ):
    raise FamilyEvidenceError("private source contains an unsafe member path")
  return value


def _lower_hex(value: Any, *, length: int, field: str) -> str:
  text = str(value or "")
  if len(text) != length or any(character not in "0123456789abcdef" for character in text):
    raise FamilyEvidenceError(f"{field} is not a lowercase hexadecimal digest")
  return text


def _normalized_top_level_private_sources(
    payload: Any,
    *,
    source_cases: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  """Validate and index the custodian-only source/instance receipt sidecar."""

  if not isinstance(payload, dict) or set(payload) != {
      "schema_version",
      "visibility",
      "storage_boundary",
      "instance_identity_schema_version",
      "cases",
  }:
    raise FamilyEvidenceError("private_source_bindings has an invalid field set")
  expected = {
      "schema_version": "benchmark_v2_private_source_bindings.v1",
      "visibility": "private_custodian_only",
      "storage_boundary": "family_source_top_level_sidecar_not_model_case",
      "instance_identity_schema_version": "fusion_assembly_instance_identity.v1",
  }
  for field, value in expected.items():
    if payload.get(field) != value:
      raise FamilyEvidenceError(f"private_source_bindings.{field} is unsupported")
  rows = payload.get("cases")
  if not isinstance(rows, list) or not rows:
    raise FamilyEvidenceError("private_source_bindings.cases must be non-empty")

  public_by_case = {str(case.get("id") or ""): case for case in source_cases}
  if (
      any(not case_id for case_id in public_by_case)
      or len(public_by_case) != len(source_cases)
  ):
    raise FamilyEvidenceError("authoritative source case IDs are empty or duplicated")

  indexed: dict[str, dict[str, Any]] = {}
  for row in rows:
    if not isinstance(row, dict) or set(row) != {
        "case_id",
        "assembly_dir",
        "parts",
        "selected_part_names",
        "selected_body_uuids",
        "selected_geometry_assets",
        "source_receipt_binding",
    }:
      raise FamilyEvidenceError("private source case has an invalid field set")
    case_id = str(row.get("case_id") or "")
    if case_id not in public_by_case or case_id in indexed:
      raise FamilyEvidenceError("private source case ID is unknown or duplicated")
    assembly_dir = row.get("assembly_dir")
    if (
        not isinstance(assembly_dir, str)
        or not assembly_dir
        or assembly_dir in {".", ".."}
        or "/" in assembly_dir
        or "\\" in assembly_dir
        or ":" in assembly_dir
    ):
      raise FamilyEvidenceError("private source assembly_dir is malformed")
    parts = row.get("parts")
    names = row.get("selected_part_names")
    bodies = row.get("selected_body_uuids")
    assets = row.get("selected_geometry_assets")
    if not all(isinstance(value, list) for value in (parts, names, bodies, assets)):
      raise FamilyEvidenceError("private source instance arrays are malformed")
    count = len(names)
    if (
        not 2 <= count <= 12
        or any(len(value) != count for value in (parts, bodies, assets))
        or len(set(str(value) for value in names)) != count
        or any(not re.fullmatch(r"part_[0-9]{3,}", str(value)) for value in names)
        or any(
            not re.fullmatch(r"geometry_[0-9]{3,}", str(value))
            for value in assets
        )
        or any(
            not isinstance(value, str) or not value.strip() for value in bodies
        )
    ):
      raise FamilyEvidenceError("private source instance arrays disagree")
    for part_path in parts:
      _safe_private_source_path(part_path, suffix=".step")
    public_case = public_by_case[case_id]
    if public_case.get("selected_part_names") != names or public_case.get(
        "selected_geometry_assets"
    ) != assets:
      raise FamilyEvidenceError(
          "private source aliases do not match the public source case"
      )

    receipt = row.get("source_receipt_binding")
    if not isinstance(receipt, dict) or set(receipt) != {
        "archive",
        "receipt_sha256",
        "assembly_json",
        "body_steps",
        "instances",
    }:
      raise FamilyEvidenceError("private source receipt binding is malformed")
    if not re.fullmatch(r"a1\.0\.0_[0-9]{2}\.7z", str(receipt.get("archive") or "")):
      raise FamilyEvidenceError("private source archive identity is malformed")
    _lower_hex(
        receipt.get("receipt_sha256"),
        length=64,
        field="private source receipt_sha256",
    )
    assembly = receipt.get("assembly_json")
    if not isinstance(assembly, dict) or set(assembly) != {"path", "bytes", "sha256"}:
      raise FamilyEvidenceError("private source assembly receipt is malformed")
    assembly_path = _safe_private_source_path(
        assembly.get("path"), suffix=".json"
    )
    if assembly_path != f"{assembly_dir}/assembly.json":
      raise FamilyEvidenceError("private source assembly path binding is invalid")
    if not isinstance(assembly.get("bytes"), int) or isinstance(
        assembly.get("bytes"), bool
    ) or int(assembly["bytes"]) <= 0:
      raise FamilyEvidenceError("private source assembly byte count is invalid")
    _lower_hex(
        assembly.get("sha256"),
        length=64,
        field="private source assembly sha256",
    )

    step_rows = receipt.get("body_steps")
    instance_rows = receipt.get("instances")
    if (
        not isinstance(step_rows, list)
        or not isinstance(instance_rows, list)
        or len(step_rows) != count
        or len(instance_rows) != count
    ):
      raise FamilyEvidenceError("private source receipt does not cover all instances")
    steps_by_part: dict[str, dict[str, Any]] = {}
    for step in step_rows:
      if not isinstance(step, dict) or set(step) != {
          "part",
          "geometry_asset",
          "body_uuid",
          "path",
          "bytes",
          "sha1",
          "sha256",
      }:
        raise FamilyEvidenceError("private source STEP receipt is malformed")
      part = str(step.get("part") or "")
      if part in steps_by_part:
        raise FamilyEvidenceError("private source STEP part is duplicated")
      _safe_private_source_path(step.get("path"), suffix=".step")
      if not isinstance(step.get("bytes"), int) or isinstance(
          step.get("bytes"), bool
      ) or int(step["bytes"]) <= 0:
        raise FamilyEvidenceError("private source STEP byte count is invalid")
      _lower_hex(step.get("sha1"), length=40, field="private source STEP sha1")
      _lower_hex(step.get("sha256"), length=64, field="private source STEP sha256")
      steps_by_part[part] = step

    instances_by_part: dict[str, dict[str, Any]] = {}
    for instance in instance_rows:
      if not isinstance(instance, dict) or set(instance) != {
          "part",
          "geometry_asset",
          "source_instance_key",
          "occurrence_path",
          "is_visible",
          "is_grounded",
      }:
        raise FamilyEvidenceError("private source instance receipt is malformed")
      part = str(instance.get("part") or "")
      if part in instances_by_part:
        raise FamilyEvidenceError("private source instance part is duplicated")
      source_key = instance.get("source_instance_key")
      if not isinstance(source_key, dict):
        raise FamilyEvidenceError("private source instance key is malformed")
      kind = source_key.get("kind")
      expected_key_fields = (
          {"kind", "body_uuid", "occurrence_uuid"}
          if kind == "occurrence"
          else {"kind", "body_uuid", "root_component_uuid"}
          if kind == "root"
          else set()
      )
      if set(source_key) != expected_key_fields or any(
          not isinstance(source_key.get(field), str)
          or not str(source_key.get(field)).strip()
          for field in expected_key_fields - {"kind"}
      ):
        raise FamilyEvidenceError("private source instance key is malformed")
      occurrence_path = instance.get("occurrence_path")
      if not isinstance(occurrence_path, list) or any(
          not isinstance(value, str) or not value for value in occurrence_path
      ):
        raise FamilyEvidenceError("private source occurrence path is malformed")
      if (
          len(occurrence_path) != len(set(occurrence_path))
          or
          (kind == "root" and occurrence_path)
          or (
              kind == "occurrence"
              and (
                  not occurrence_path
                  or occurrence_path[-1] != source_key["occurrence_uuid"]
              )
          )
          or not isinstance(instance.get("is_visible"), bool)
          or not isinstance(instance.get("is_grounded"), bool)
      ):
        raise FamilyEvidenceError("private source instance placement is inconsistent")
      instances_by_part[part] = instance

    expected_parts = [str(value) for value in names]
    if set(steps_by_part) != set(expected_parts) or set(instances_by_part) != set(
        expected_parts
    ):
      raise FamilyEvidenceError("private source receipt part coverage is incomplete")
    for index, part in enumerate(expected_parts):
      step = steps_by_part[part]
      instance = instances_by_part[part]
      source_key = instance["source_instance_key"]
      if (
          step.get("geometry_asset") != assets[index]
          or instance.get("geometry_asset") != assets[index]
          or step.get("body_uuid") != bodies[index]
          or source_key.get("body_uuid") != bodies[index]
          or step.get("path") != f"{assembly_dir}/{parts[index]}"
      ):
        raise FamilyEvidenceError(
            "private source per-instance geometry binding is inconsistent"
        )
    indexed[case_id] = {
        "schema_version": payload["schema_version"],
        "visibility": payload["visibility"],
        "source": copy.deepcopy(row),
    }

  if set(indexed) != set(public_by_case):
    raise FamilyEvidenceError(
        "private source case set must exactly match authoritative source cases"
    )
  commitment = {
      "schema_version": str(payload["schema_version"]),
      "case_count": len(indexed),
      "sha256": "sha256:" + _sha256_text(_canonical_json(payload)),
  }
  return indexed, commitment


def _load_cases_document(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except FileNotFoundError as exc:
    raise FamilyEvidenceError(f"Input cases file not found: {path}") from exc
  except json.JSONDecodeError as exc:
    raise FamilyEvidenceError(f"Input cases JSON is invalid: {path}: {exc}") from exc
  exposures = _forbidden_source_exposures(payload)
  if exposures:
    raise FamilyEvidenceError(
        "Input document exposes forbidden final/test data, file variants, or raw salt: "
        + ", ".join(exposures[:20])
    )
  metadata: dict[str, Any] = {}
  if isinstance(payload, list):
    raw_cases = payload
  elif isinstance(payload, dict):
    raw_cases = payload.get("cases")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {
            "cases",
            "evaluation_gold_contacts",
            "private_source_bindings",
        }
    }
  else:
    raw_cases = None
  if not isinstance(raw_cases, list):
    raise FamilyEvidenceError("Input JSON must be a case list or an object with a cases list")
  cases = [case for case in raw_cases if isinstance(case, dict)]
  if len(cases) != len(raw_cases):
    raise FamilyEvidenceError("Every input case must be a JSON object")
  if not cases:
    raise FamilyEvidenceError("Input cases list is empty")
  if isinstance(payload, dict) and payload.get("formal") is True:
    if payload.get("schema_version") != 2:
      raise FamilyEvidenceError(
          "Formal instance-unaware family source schema is forbidden; rebuild v2"
      )
    missing = {
        "evaluation_gold_contacts",
        "private_source_bindings",
    } - set(payload)
    if missing:
      raise FamilyEvidenceError(
          "Formal benchmark-v2 family source requires both private top-level "
          "sidecars: " + ", ".join(sorted(missing))
      )
  if isinstance(payload, dict) and payload.get("schema_version") == 2:
    missing = {
        "evaluation_gold_contacts",
        "private_source_bindings",
    } - set(payload)
    if missing:
      raise FamilyEvidenceError(
          "Family source v2 has an incomplete private sidecar set: "
          + ", ".join(sorted(missing))
      )
  if isinstance(payload, dict) and "evaluation_gold_contacts" in payload:
    indexed_gold, receipt = _normalized_top_level_evaluation_gold(
        payload["evaluation_gold_contacts"],
        source_case_ids=[str(case.get("id") or "") for case in cases],
    )
    for case in cases:
      case_id = str(case.get("id") or "")
      if "evaluation_gold_contacts" in case:
        raise FamilyEvidenceError(
            f"Case {case_id} duplicates the top-level evaluation gold sidecar"
        )
      case["evaluation_gold_contacts"] = indexed_gold[case_id]
    metadata["evaluation_label_receipt"] = receipt
  if isinstance(payload, dict) and "private_source_bindings" in payload:
    indexed_sources, receipt = _normalized_top_level_private_sources(
        payload["private_source_bindings"],
        source_cases=cases,
    )
    for case in cases:
      case_id = str(case.get("id") or "")
      if "private_source_binding" in case:
        raise FamilyEvidenceError(
            f"Case {case_id} duplicates the top-level private source sidecar"
        )
      case["private_source_binding"] = indexed_sources[case_id]
    metadata["private_source_binding_receipt"] = receipt

  if isinstance(payload, dict) and payload.get("schema_version") == 2:
    for case in cases:
      case_id = str(case.get("id") or "")
      names = case.get("selected_part_names")
      assets = case.get("selected_geometry_assets")
      if (
          not isinstance(names, list)
          or not isinstance(assets, list)
          or len(names) != len(assets)
          or len(set(str(value) for value in names)) != len(names)
      ):
        raise FamilyEvidenceError(
            f"Case {case_id} has an invalid public instance alias contract"
        )
      asset_by_part = {
          str(part): str(asset) for part, asset in zip(names, assets)
      }
      private_gold = case.get("evaluation_gold_contacts")
      if not isinstance(private_gold, dict):
        raise FamilyEvidenceError(f"Case {case_id} lacks joined gold v2")
      for contact in private_gold.get("contacts") or []:
        for endpoint_name in ("endpoint_a", "endpoint_b"):
          endpoint = contact.get(endpoint_name) if isinstance(contact, dict) else None
          if (
              not isinstance(endpoint, dict)
              or asset_by_part.get(str(endpoint.get("part") or ""))
              != str(endpoint.get("geometry_asset") or "")
          ):
            raise FamilyEvidenceError(
                f"Case {case_id} gold endpoint does not match public aliases"
            )
  return cases, metadata


def _validated_salt_commitment(value: Any) -> str | None:
  if value is None or not str(value).strip():
    return None
  text = str(value).strip().lower()
  prefix = "sha256:"
  digest = text[len(prefix) :] if text.startswith(prefix) else ""
  if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
    raise FamilyEvidenceError(
        "final_test_salt_commitment must be sha256:<64 hex>; never pass the raw salt"
    )
  return prefix + digest


def _validated_sha256_identity(value: Any, *, field: str) -> str:
  text = str(value or "").strip().lower()
  if not text.startswith("sha256:") or len(text) != 71 or any(
      char not in "0123456789abcdef" for char in text[7:]
  ):
    raise FamilyEvidenceError(f"{field} must be sha256:<64 hex>")
  return text


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_json_fsync(path: Path, payload: Any) -> None:
  with path.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(
        payload,
        stream,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


class _JsonBundleTransaction:
  """Publish public and custodian JSON files as one rollback-safe bundle."""

  def __init__(self) -> None:
    self._token = uuid.uuid4().hex
    self._staged: dict[Path, Path] = {}
    self._backups: dict[Path, Path] = {}
    self._created_directories: list[Path] = []
    self._closed = False

  def _ensure_parent(self, parent: Path) -> None:
    missing: list[Path] = []
    cursor = parent
    while not cursor.exists():
      missing.append(cursor)
      if cursor.parent == cursor:
        break
      cursor = cursor.parent
    parent.mkdir(parents=True, exist_ok=True)
    self._created_directories.extend(reversed(missing))

  def stage_json(self, target: Path, payload: Any) -> None:
    if self._closed:
      raise RuntimeError("JSON bundle transaction is closed")
    target = Path(target)
    if target in self._staged:
      raise ValueError(f"Duplicate JSON bundle target: {target}")
    self._ensure_parent(target.parent)
    if target.is_symlink():
      raise FamilyEvidenceError("Refusing to replace a symlink output target")
    if target.exists() and not target.is_file():
      raise FamilyEvidenceError("Refusing to replace a non-file output target")
    staged = target.with_name(f".{target.name}.stage-{self._token}")
    try:
      _write_json_fsync(staged, payload)
    except BaseException:
      staged.unlink(missing_ok=True)
      raise
    self._staged[target] = staged

  def staged_path_for(self, target: Path) -> Path:
    try:
      return self._staged[Path(target)]
    except KeyError as error:
      raise KeyError(f"JSON bundle target is not staged: {target}") from error

  def commit(self, *, commit_marker: Path) -> None:
    if self._closed:
      raise RuntimeError("JSON bundle transaction is closed")
    marker = Path(commit_marker)
    if marker not in self._staged:
      raise ValueError("JSON bundle commit marker was not staged")
    ordered = sorted(
        (target for target in self._staged if target != marker),
        key=lambda path: str(path).casefold(),
    ) + [marker]
    published: list[Path] = []
    try:
      for target in ordered:
        if target.exists():
          backup = target.with_name(f".{target.name}.backup-{self._token}")
          os.replace(target, backup)
          self._backups[target] = backup
      for target in ordered:
        os.replace(self._staged[target], target)
        published.append(target)
    except BaseException as publish_error:
      rollback_errors: list[str] = []
      published_set = set(published)
      # Restore the manifest marker last so readers never observe an old
      # manifest over a partially restored bundle.
      for target in ordered:
        backup = self._backups.get(target)
        try:
          if backup is not None and backup.exists():
            os.replace(backup, target)
          elif target in published_set:
            target.unlink(missing_ok=True)
        except BaseException as rollback_error:
          rollback_errors.append(f"{target}: {rollback_error}")
      for staged in self._staged.values():
        try:
          staged.unlink(missing_ok=True)
        except OSError as cleanup_error:
          rollback_errors.append(f"{staged}: {cleanup_error}")
      if not rollback_errors:
        for directory in reversed(self._created_directories):
          try:
            directory.rmdir()
          except OSError:
            pass
      self._closed = True
      if rollback_errors:
        preserved = [
            str(path) for path in self._backups.values() if path.exists()
        ]
        detail = "; ".join(rollback_errors[:10])
        if preserved:
          detail += "; preserved backups: " + ", ".join(preserved[:10])
        raise FamilyEvidenceError(
            "JSON bundle publication failed and rollback was incomplete: "
            + detail
        ) from publish_error
      raise
    else:
      for backup in self._backups.values():
        try:
          backup.unlink(missing_ok=True)
        except OSError:
          # All targets and the commit marker are already durable. A stale
          # hidden backup is safer than reporting a false publication failure.
          pass
      self._closed = True

  def close(self, *, remove_created_directories: bool = True) -> None:
    if self._closed:
      return
    for staged in self._staged.values():
      try:
        staged.unlink(missing_ok=True)
      except OSError:
        pass
    for backup in self._backups.values():
      try:
        backup.unlink(missing_ok=True)
      except OSError:
        pass
    if remove_created_directories:
      for directory in reversed(self._created_directories):
        try:
          directory.rmdir()
        except OSError:
          pass
    self._closed = True

  def __del__(self) -> None:
    if hasattr(self, "_closed"):
      self.close()


def _validate_isolated_output_directories(
    directories: dict[str, Path],
) -> dict[str, Path]:
  resolved = {
      name: Path(path).resolve(strict=False) for name, path in directories.items()
  }
  names = sorted(resolved)
  for offset, left_name in enumerate(names):
    left = resolved[left_name]
    for right_name in names[offset + 1 :]:
      right = resolved[right_name]
      if left == right:
        raise FamilyEvidenceError(
            f"{left_name} and {right_name} output directories must differ"
        )
      try:
        left.relative_to(right)
      except ValueError:
        pass
      else:
        raise FamilyEvidenceError(
            f"{left_name} must not be nested in {right_name}"
        )
      try:
        right.relative_to(left)
      except ValueError:
        pass
      else:
        raise FamilyEvidenceError(
            f"{right_name} must not be nested in {left_name}"
        )
  return resolved


def _validate_input_outside_output_directories(
    input_path: Path,
    directories: dict[str, Path],
) -> None:
  authoritative_input = Path(input_path).resolve(strict=True)
  for name, directory in directories.items():
    try:
      authoritative_input.relative_to(directory)
    except ValueError:
      continue
    raise FamilyEvidenceError(
        f"input_cases must be outside the {name} output directory"
    )


def main() -> None:
  args = build_parser().parse_args()
  input_path = resolve_path(args.input_cases)
  output_dir = resolve_path(args.output_dir)
  forbidden_existing = []
  if output_dir.exists():
    forbidden_existing = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and _name_tokens(path.name) & {"test", "final", "finaltest"}
    ]
  if forbidden_existing:
    raise FamilyEvidenceError(
        "Refusing to write benchmark-v2 train/dev beside final/test file variants: "
        + ", ".join(str(path) for path in forbidden_existing[:20])
    )
  cases, source_metadata = _load_cases_document(input_path)
  review_artifact = None
  if str(args.family_pair_review or "").strip():
    review_path = resolve_path(args.family_pair_review)
    review_artifact = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review_artifact, dict):
      raise FamilyEvidenceError("family_pair_review must contain a JSON object")
  review_key: bytes | None = None
  if str(args.review_verification_key_file or "").strip():
    review_key = resolve_path(args.review_verification_key_file).read_bytes().strip()
  external_attestation = None
  if str(args.external_final_test_attestation or "").strip():
    attestation_path = resolve_path(args.external_final_test_attestation)
    external_attestation = json.loads(
        attestation_path.read_text(encoding="utf-8")
    )
    if not isinstance(external_attestation, dict):
      raise FamilyEvidenceError(
          "external_final_test_attestation must contain a JSON object"
      )
  external_attestation_key: dict[str, Any] | None = None
  if str(args.external_final_test_verification_key_file or "").strip():
    key_path = resolve_path(args.external_final_test_verification_key_file)
    external_attestation_key = json.loads(key_path.read_text(encoding="utf-8"))
    if not isinstance(external_attestation_key, dict):
      raise FamilyEvidenceError(
          "external_final_test_verification_key_file must contain a JSON object"
      )
  config = FamilySplitConfig(
      seed=int(args.seed),
      dev_ratio=float(args.dev_ratio),
      dev_ratio_tolerance=float(args.dev_ratio_tolerance),
      fail_if_ratio_impossible=not bool(args.allow_ratio_outside_tolerance),
      near_family_threshold=args.near_family_threshold,
      neighbor_top_k=int(args.neighbor_top_k),
      neighbor_warning_threshold=float(args.neighbor_warning_threshold),
      required_evidence_coverage=float(args.required_evidence_coverage),
      family_pair_review_artifact=review_artifact,
      family_pair_review_verification_key=review_key,
      development_synthetic_mode=bool(args.development_synthetic_mode),
      final_test_generation_rule=str(args.final_test_generation_rule),
      final_test_salt_commitment=(
          _validated_salt_commitment(args.final_test_salt_commitment)
      ),
      external_final_test_attestation=external_attestation,
      external_final_test_verification_key=external_attestation_key,
  )
  result = build_family_splits(cases, config=config)

  has_private_gold = any(
      int(result["private_evaluation_gold"][split].get("case_count", 0)) > 0
      for split in ("train", "dev")
  )
  has_private_sources = any(
      int(result["private_source_bindings"][split].get("case_count", 0)) > 0
      for split in ("train", "dev")
  )
  private_gold_dir: Path | None = None
  if has_private_gold:
    raw_private_dir = str(args.private_evaluation_output_dir or "").strip()
    if not raw_private_dir:
      raise FamilyEvidenceError(
          "Source cases contain private evaluation gold; provide a separate "
          "--private_evaluation_output_dir"
      )
    private_gold_dir = resolve_path(raw_private_dir)
  private_source_dir: Path | None = None
  if has_private_sources:
    raw_private_source_dir = str(args.private_source_output_dir or "").strip()
    if not raw_private_source_dir:
      raise FamilyEvidenceError(
          "Source cases contain private raw source bindings; provide a separate "
          "--private_source_output_dir"
      )
    private_source_dir = resolve_path(raw_private_source_dir)

  requested_directories = {"public": output_dir}
  if private_gold_dir is not None:
    requested_directories["private_evaluation"] = private_gold_dir
  if private_source_dir is not None:
    requested_directories["private_source"] = private_source_dir
  isolated_directories = _validate_isolated_output_directories(
      requested_directories
  )
  _validate_input_outside_output_directories(
      input_path,
      isolated_directories,
  )
  output_dir = isolated_directories["public"]
  private_gold_dir = isolated_directories.get("private_evaluation")
  private_source_dir = isolated_directories.get("private_source")
  for name, directory in isolated_directories.items():
    if name == "public" or not directory.exists():
      continue
    forbidden_private_files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and _name_tokens(path.name) & {"test", "final", "finaltest"}
    ]
    if forbidden_private_files:
      raise FamilyEvidenceError(
          f"Refusing to write {name} beside final/test file variants: "
          + ", ".join(str(path) for path in forbidden_private_files[:20])
      )

  train_path = output_dir / "train.json"
  dev_path = output_dir / "dev.json"
  audit_path = output_dir / "leakage_audit.json"
  manifest_path = output_dir / "split_manifest.json"
  transaction = _JsonBundleTransaction()
  try:
    transaction.stage_json(train_path, result["train"])
    transaction.stage_json(dev_path, result["dev"])
    transaction.stage_json(audit_path, result["audit"])
    if has_private_gold:
      if private_gold_dir is None:
        raise RuntimeError("Private evaluation gold directory was not validated")
      for split in ("train", "dev"):
        transaction.stage_json(
            private_gold_dir / f"private_{split}_evaluation_gold.json",
            result["private_evaluation_gold"][split],
        )
    if has_private_sources:
      if private_source_dir is None:
        raise RuntimeError("Private source directory was not validated")
      for split in ("train", "dev"):
        transaction.stage_json(
            private_source_dir / f"private_{split}_source_bindings.json",
            result["private_source_bindings"][split],
        )

    source_schema_version = source_metadata.get("schema_version")
    if not isinstance(source_schema_version, (int, str)):
      source_schema_version = "unversioned_case_list"
    manifest = dict(result["manifest"])
    manifest["source_document"] = {
        "schema_version": source_schema_version,
        "metadata_sha256": "sha256:"
        + _sha256_text(_canonical_json(source_metadata)),
        "private_sidecars_excluded": True,
    }
    public_artifact_paths = {
        "train": train_path,
        "dev": dev_path,
        "leakage_audit": audit_path,
    }
    manifest["artifact_hashes"] = {
        "input_cases": {
            "path": input_path.name,
            "sha256": _sha256_file(input_path),
            "bytes": input_path.stat().st_size,
        },
        **{
            name: {
                "path": path.name,
                "sha256": _sha256_file(transaction.staged_path_for(path)),
                "bytes": transaction.staged_path_for(path).stat().st_size,
            }
            for name, path in public_artifact_paths.items()
        },
    }
    manifest["strict_hygiene_clean"] = bool(
        result["audit"].get("strict_hygiene_clean", False)
    )
    _validate_public_manifest_boundary(manifest)
    transaction.stage_json(manifest_path, manifest)
    transaction.commit(commit_marker=manifest_path)
  except BaseException:
    transaction.close(remove_created_directories=True)
    raise
  print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
  main()
