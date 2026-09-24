"""Exact-rebuild verifier for serialized V19 v2 graph-work indexes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Any, Mapping

import jsonschema

from .benchmark_v2_query_semantic_adapter_v2 import (
    AuthenticatedBenchmarkV2QuerySemanticIndexV2,
    AuthenticatedQueryMateSemanticFullDomain,
    PROJECT_ROOT,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    _canonical_bytes,
    _capture_project_file,
    _json_from_capture,
    _reverify_capture,
    build_benchmark_v2_query_semantic_index_v2,
)
from .benchmark_v2_training_provenance import (
    CapturedFileArtifact,
    canonical_sha256,
    capture_file_artifact,
)
@dataclass(frozen=True, slots=True)
class GeometryEndpointBindingV2:
  step_sha256: str
  step_bytes: int
  source_path_sha256: str
  face_map_receipt_sha256: str
  face_map_receipt_bytes: int
  mapper_endpoint_identity_sha256: str
  fusion_face_index: int
  raw_occ_face_indices: tuple[int, ...]
  source_face_signature_sha256s: tuple[str, ...]
  endpoint_graph_cache_key_sha256: str


@dataclass(frozen=True, slots=True)
class GeometryMaterializationRequestV2:
  graph_work_id: str
  model_view_schema_version: str
  graph_schema_version: str
  endpoint_a: GeometryEndpointBindingV2
  endpoint_b: GeometryEndpointBindingV2


@dataclass(frozen=True, slots=True)
class ProgramDescriptorMetadataV2:
  relation_hint: str
  contact_type: str
  surface_type_a: str
  surface_type_b: str


@dataclass(frozen=True, slots=True)
class SupervisionTargetV2:
  program_id: str
  source_program_row_sha256: str
  descriptor: ProgramDescriptorMetadataV2
  descriptor_sha256: str
  catalog_index: int | None
  catalog_program_id: str | None
  oov: bool
  residual_translation: tuple[float, float, float]
  residual_rotation_row_major: tuple[
      float, float, float, float, float, float, float, float, float
  ]
  target_payload_sha256: str


@dataclass(frozen=True, slots=True)
class SupervisionSampleV2:
  schema_version: str
  sample_identity_sha256: str
  opaque_case_identity: str
  opaque_query_identity: str
  source_contact_ordinal: int
  development_split: str
  direction: str
  graph_work_id: str
  family_cluster_id: str
  assembly_cluster_id: str
  target_count: int
  known_target_count: int
  oov_target_count: int
  classification_evaluable: bool
  targets: tuple[SupervisionTargetV2, ...]
  sample_payload_sha256: str


@dataclass(frozen=True, slots=True)
class FixedLineageMetadataV2:
  schema_version: str
  index_payload_sha256: str
  authenticated_receipt_sha256: str
  receipt_payload_sha256: str
  domain_sha256: str
  source_revision: str
  fixed_v19_manifest_sha256: str
  fixed_family_source_sha256: str


@dataclass(frozen=True, slots=True)
class CatalogEntryMetadataV2:
  program_index: int
  program_id: str
  program_type: str
  descriptor: ProgramDescriptorMetadataV2
  descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class TrainCatalogStatisticsV2:
  source_split: str
  source_binding_count: int
  source_row_count: int
  source_commitment_sha256: str
  known_target_count: int
  oov_target_count: int
  positive_class_count: int
  target_frequency_by_program_index: tuple[int, ...]
  class_weight_by_program_index: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FixedCatalogMetadataV2:
  schema_version: str
  catalog_payload_sha256: str
  catalog_roster_sha256: str
  candidate_policy: str
  candidate_order: str
  target_policy: str
  entry_count: int
  entries: tuple[CatalogEntryMetadataV2, ...]
  training_statistics: TrainCatalogStatisticsV2


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, label: str
) -> None:
  unsigned = dict(payload)
  observed = unsigned.pop(field, None)
  if observed != canonical_sha256(unsigned):
    raise ValueError(f"{label} self-hash differs")


def _run_git_for_revision_check(
    arguments: list[str],
    *,
    label: str,
) -> subprocess.CompletedProcess[bytes]:
  try:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        timeout=10,
    )
  except (OSError, subprocess.SubprocessError) as error:
    raise ValueError(f"{label} is unavailable") from error


def _verify_revision_neutral_producer(
    *,
    serialized: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
  serialized_producer = serialized.get("producer")
  expected_producer = expected.get("producer")
  if (
      not isinstance(serialized_producer, Mapping)
      or not isinstance(expected_producer, Mapping)
  ):
    raise ValueError("serialized producer binding differs")
  serialized_without_revision = dict(serialized_producer)
  serialized_without_revision.pop("git_revision", None)
  expected_without_revision = dict(expected_producer)
  expected_without_revision.pop("git_revision", None)
  if serialized_without_revision != expected_without_revision:
    raise ValueError("serialized producer implementation differs")
  serialized_revision = serialized_producer.get("git_revision")
  expected_revision = expected_producer.get("git_revision")
  if (
      not isinstance(serialized_revision, str)
      or re.fullmatch(r"[0-9a-f]{40}", serialized_revision) is None
  ):
    raise ValueError("serialized producer revision differs")
  commit = _run_git_for_revision_check(
      [
          "rev-parse",
          "--verify",
          "--quiet",
          f"{serialized_revision}^{{commit}}",
      ],
      label="serialized producer revision",
  )
  if commit.returncode != 0:
    raise ValueError("serialized producer revision does not exist")
  if (
      not isinstance(expected_revision, str)
      or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None
  ):
    raise ValueError("current producer revision differs")
  if serialized_revision != expected_revision:
    ancestor = _run_git_for_revision_check(
        [
            "merge-base",
            "--is-ancestor",
            serialized_revision,
            expected_revision,
        ],
        label="producer revision ancestry",
    )
    if ancestor.returncode != 0:
      raise ValueError(
          "serialized producer revision is not an ancestor of current revision"
      )
  implementation = expected_producer.get("implementation")
  if not isinstance(implementation, Mapping):
    raise ValueError("current producer implementation differs")
  for binding in implementation.values():
    if not isinstance(binding, Mapping):
      raise ValueError("current producer implementation differs")
    logical_id = binding.get("logical_id")
    if not isinstance(logical_id, str) or not logical_id:
      raise ValueError("current producer logical ID differs")
    blob = _run_git_for_revision_check(
        ["show", f"{serialized_revision}:{logical_id}"],
        label=f"serialized producer blob {logical_id}",
    )
    if blob.returncode != 0:
      raise ValueError(
          f"serialized producer blob {logical_id} does not exist"
      )
    if (
        binding.get("bytes") != len(blob.stdout)
        or binding.get("sha256")
        != hashlib.sha256(blob.stdout).hexdigest()
    ):
      raise ValueError(
          f"serialized producer blob {logical_id} differs"
      )
  if serialized_revision == expected_revision:
    return copy.deepcopy(dict(expected))
  canonical_expected = copy.deepcopy(dict(expected))
  canonical_expected["producer"]["git_revision"] = serialized_revision
  unsigned_expected = dict(canonical_expected)
  unsigned_expected.pop("index_payload_sha256", None)
  canonical_expected["index_payload_sha256"] = canonical_sha256(
      unsigned_expected
  )
  return canonical_expected


def _verify_exact_rebuild_payload(
    *,
    serialized: Mapping[str, Any],
    serialized_bytes: bytes,
    expected: Mapping[str, Any],
    label: str,
) -> None:
  canonical_expected = _verify_revision_neutral_producer(
      serialized=serialized,
      expected=expected,
  )
  if (
      serialized != canonical_expected
      or serialized_bytes != _canonical_bytes(canonical_expected)
  ):
    raise ValueError(f"serialized graph-work index differs from {label}")


def _verify_fixed_catalog(payload: Any) -> Mapping[str, Any]:
  if (
      not isinstance(payload, Mapping)
      or payload.get("schema_version")
      != "benchmark_v2_pre_v19_fixed_program_catalog.v2"
      or payload.get("candidate_policy")
      != "pre_v19_fixed19_catalog_with_explicit_oov_failure.v1"
      or payload.get("candidate_order")
      != "pre_v19_roster_program_index.v1"
      or payload.get("target_policy")
      != "all_targets_retained_catalog_index_or_null_oov.v1"
      or payload.get("entry_count") != 19
  ):
    raise ValueError("serialized fixed19 catalog contract differs")
  _verify_self_hash(
      payload, field="catalog_payload_sha256", label="fixed19 catalog"
  )
  source = payload.get("fixed_source")
  expected_source = {
      "policy_source_revision": (
          "ee922bbbbb28753b6123bd5e951afeb0b5b5481b"
      ),
      "source_spec_payload_sha256": (
          "71200b9b894419f8519ea1dabd5809207702903d67b44820a2c59b54faaceb3d"
      ),
      "catalog_roster_sha256": (
          "7e97af33edd35186499f6e187a85a0b07a0a0f31afe105b36d46754b49755724"
      ),
      "source_artifact_sha256": (
          "2ca8cb6e13e2b176267ac1f81e13fcbc474f791e89dadb282e1ff6f4a6370966"
      ),
      "producer_code_sha256": (
          "9f4151a5624d426516cd5d479ac54da5723413203afaa57cf0810c63f26d088c"
      ),
      "trust_root_id": "p0_frozen_program_roster_20260719.v1",
  }
  if not isinstance(source, Mapping) or any(
      source.get(key) != value for key, value in expected_source.items()
  ):
    raise ValueError("serialized fixed19 source commitments differ")
  entries = payload.get("entries")
  if not isinstance(entries, list) or len(entries) != 19:
    raise ValueError("serialized fixed19 roster differs")
  for index, entry in enumerate(entries):
    if not isinstance(entry, Mapping):
      raise ValueError("serialized fixed19 entry differs")
    descriptor = entry.get("descriptor")
    if not isinstance(descriptor, Mapping):
      raise ValueError("serialized fixed19 descriptor differs")
    descriptor_sha256 = canonical_sha256(descriptor)
    if (
        entry.get("program_index") != index
        or entry.get("descriptor_sha256") != descriptor_sha256
        or entry.get("program_id")
        != f"catalog_{descriptor_sha256[:16]}"
        or entry.get("program_type") != descriptor.get("relation_hint")
    ):
      raise ValueError("serialized fixed19 roster identity differs")
  roster_payload = {
      "schema_version": "finite_program_catalog_roster.v2",
      "entry_count": 19,
      "entries": [
          {
              key: entry[key]
              for key in (
                  "program_index",
                  "program_id",
                  "descriptor",
                  "descriptor_sha256",
              )
          }
          for entry in entries
      ],
  }
  if canonical_sha256(roster_payload) != source.get(
      "catalog_roster_sha256"
  ):
    raise ValueError("serialized fixed19 roster hash differs")
  statistics = payload.get("training_statistics")
  frequencies = (
      statistics.get("target_frequency_by_program_index")
      if isinstance(statistics, Mapping)
      else None
  )
  weights = (
      statistics.get("class_weight_by_program_index")
      if isinstance(statistics, Mapping)
      else None
  )
  if (
      not isinstance(frequencies, list)
      or len(frequencies) != 19
      or any(
          isinstance(value, bool) or not isinstance(value, int) or value < 0
          for value in frequencies
      )
      or not isinstance(weights, list)
      or len(weights) != 19
  ):
    raise ValueError("serialized fixed19 train statistics differ")
  known = sum(frequencies)
  positive = sum(value > 0 for value in frequencies)
  expected_weights = [
      known / (positive * count) if count > 0 and positive > 0 else 0.0
      for count in frequencies
  ]
  if (
      statistics.get("schema_version")
      != "v19_train_only_class_statistics.v1"
      or statistics.get("source_split") != "train"
      or statistics.get("frequency_policy")
      != "exact_authority_target_count_by_fixed_catalog_index.v1"
      or statistics.get("weight_policy")
      != "balanced_inverse_frequency_positive_classes_zero_for_unseen.v1"
      or statistics.get("known_target_count") != known
      or statistics.get("positive_class_count") != positive
      or statistics.get("source_row_count")
      != known + statistics.get("oov_target_count", -1)
      or weights != expected_weights
  ):
    raise ValueError("serialized fixed19 train statistics do not close")
  return payload


def _verify_serialized_structure_v2(payload: Any) -> None:
  if not isinstance(payload, Mapping):
    raise ValueError("serialized graph-work index root differs")
  _verify_self_hash(
      payload, field="index_payload_sha256", label="graph-work index"
  )
  catalog = _verify_fixed_catalog(payload.get("program_catalog"))
  model_protocol = payload.get("model_protocol")
  group_contract = (
      model_protocol.get("supervision_group_contract")
      if isinstance(model_protocol, Mapping)
      else None
  )
  if (
      not isinstance(group_contract, Mapping)
      or group_contract.get("schema_version")
      != "benchmark_v2_supervision_group_contract.v1"
      or group_contract.get("fields")
      != ["assembly_cluster_id", "family_cluster_id"]
      or group_contract.get("derivation")
      != "fixed_v19_manifest_and_domain_lineage_hash.v1"
      or group_contract.get("allowed_uses")
      != ["clustered_resampling", "training_weighting"]
      or group_contract.get("model_feature_access") != "forbidden"
  ):
    raise ValueError("serialized supervision group contract differs")
  graph_work = payload.get("graph_work")
  samples = payload.get("samples")
  rejections = payload.get("rejected_queries")
  coverage = payload.get("coverage")
  if (
      not isinstance(graph_work, list)
      or not isinstance(samples, list)
      or not isinstance(rejections, list)
      or not isinstance(coverage, Mapping)
  ):
    raise ValueError("serialized graph-work index ledgers differ")

  graph_by_id: dict[str, Mapping[str, Any]] = {}
  endpoint_cache_keys: set[str] = set()
  for row in graph_work:
    if not isinstance(row, Mapping):
      raise ValueError("serialized graph-work row differs")
    _verify_self_hash(
        row, field="graph_work_payload_sha256", label="graph-work row"
    )
    graph_id = str(row.get("graph_work_id") or "")
    if graph_id in graph_by_id:
      raise ValueError("serialized graph-work ID is duplicated")
    graph_by_id[graph_id] = row
    endpoints = row.get("endpoints")
    if not isinstance(endpoints, Mapping):
      raise ValueError("serialized graph-work endpoints differ")
    for role in ("endpoint_a", "endpoint_b"):
      endpoint = endpoints.get(role)
      if not isinstance(endpoint, Mapping):
        raise ValueError("serialized graph endpoint differs")
      unsigned_endpoint = dict(endpoint)
      observed_cache_key = unsigned_endpoint.pop(
          "endpoint_graph_cache_key_sha256", None
      )
      if observed_cache_key != canonical_sha256(unsigned_endpoint):
        raise ValueError("serialized endpoint graph binding hash differs")
      endpoint_cache_keys.add(str(observed_cache_key))

  seen_sample_ids: set[str] = set()
  seen_input_keys: set[tuple[Any, ...]] = set()
  target_count = 0
  known_target_count = 0
  oov_target_count = 0
  evaluable_sample_count = 0
  sample_split_counts = {"train": 0, "dev": 0}
  target_split_counts = {"train": 0, "dev": 0}
  known_split_counts = {"train": 0, "dev": 0}
  oov_split_counts = {"train": 0, "dev": 0}
  evaluable_split_counts = {"train": 0, "dev": 0}
  clusters_by_case: dict[str, tuple[str, str]] = {}
  for sample in samples:
    if not isinstance(sample, Mapping):
      raise ValueError("serialized graph-work sample differs")
    _verify_self_hash(
        sample, field="sample_payload_sha256", label="graph-work sample"
    )
    sample_id = str(sample.get("sample_identity_sha256") or "")
    input_key = (
        sample.get("opaque_query_identity"),
        sample.get("source_contact_ordinal"),
        sample.get("development_split"),
        sample.get("direction"),
    )
    if sample_id in seen_sample_ids or input_key in seen_input_keys:
      raise ValueError("serialized query/direction sample is duplicated")
    seen_sample_ids.add(sample_id)
    seen_input_keys.add(input_key)
    targets = sample.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or sample.get("target_count") != len(targets)
    ):
      raise ValueError("serialized multi-positive targets differ")
    target_ids: list[str] = []
    sample_known_count = 0
    sample_oov_count = 0
    for target in targets:
      if not isinstance(target, Mapping):
        raise ValueError("serialized target row differs")
      _verify_self_hash(
          target, field="target_payload_sha256", label="target row"
      )
      target_ids.append(str(target.get("program_id") or ""))
      descriptor = target.get("descriptor")
      if (
          not isinstance(descriptor, Mapping)
          or target.get("descriptor_sha256")
          != canonical_sha256(descriptor)
      ):
        raise ValueError("serialized target descriptor differs")
      catalog_index = target.get("catalog_index")
      is_oov = target.get("oov")
      if is_oov is True:
        if (
            catalog_index is not None
            or target.get("catalog_program_id") is not None
            or any(
                entry["descriptor_sha256"]
                == target.get("descriptor_sha256")
                for entry in catalog["entries"]
            )
        ):
          raise ValueError("serialized OOV target catalog join differs")
        sample_oov_count += 1
      elif is_oov is False:
        if (
            isinstance(catalog_index, bool)
            or not isinstance(catalog_index, int)
            or catalog_index < 0
            or catalog_index >= len(catalog["entries"])
        ):
          raise ValueError("serialized target catalog index differs")
        entry = catalog["entries"][catalog_index]
        if (
            target.get("catalog_program_id") != entry["program_id"]
            or target.get("descriptor_sha256")
            != entry["descriptor_sha256"]
        ):
          raise ValueError("serialized target catalog join differs")
        sample_known_count += 1
      else:
        raise ValueError("serialized target OOV flag differs")
    if target_ids != sorted(set(target_ids)):
      raise ValueError("serialized targets are not sorted and unique")
    if (
        sample.get("known_target_count") != sample_known_count
        or sample.get("oov_target_count") != sample_oov_count
        or sample_known_count + sample_oov_count != len(targets)
        or sample.get("classification_evaluable")
        != (sample_known_count > 0)
    ):
      raise ValueError("serialized sample OOV coverage differs")
    graph = graph_by_id.get(str(sample.get("graph_work_id") or ""))
    if (
        graph is None
        or graph.get("opaque_query_identity")
        != sample.get("opaque_query_identity")
        or graph.get("source_contact_ordinal")
        != sample.get("source_contact_ordinal")
        or graph.get("development_split")
        != sample.get("development_split")
        or graph.get("direction") != sample.get("direction")
    ):
      raise ValueError("serialized sample/graph-work join differs")
    cluster_pair = (
        str(sample.get("family_cluster_id") or ""),
        str(sample.get("assembly_cluster_id") or ""),
    )
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in cluster_pair):
      raise ValueError("serialized sample supervision cluster ID differs")
    opaque_case = str(sample.get("opaque_case_identity") or "")
    prior_clusters = clusters_by_case.setdefault(opaque_case, cluster_pair)
    if prior_clusters != cluster_pair:
      raise ValueError("serialized case supervision clusters are unstable")
    split = str(sample["development_split"])
    sample_split_counts[split] += 1
    target_split_counts[split] += len(targets)
    known_split_counts[split] += sample_known_count
    oov_split_counts[split] += sample_oov_count
    evaluable_split_counts[split] += int(sample_known_count > 0)
    target_count += len(targets)
    known_target_count += sample_known_count
    oov_target_count += sample_oov_count
    evaluable_sample_count += int(sample_known_count > 0)

  for rejection in rejections:
    if not isinstance(rejection, Mapping):
      raise ValueError("serialized rejection row differs")
    _verify_self_hash(
        rejection,
        field="rejection_payload_sha256",
        label="rejection row",
    )

  if (
      coverage.get("sample_count") != len(samples)
      or coverage.get("target_count") != target_count
      or coverage.get("known_target_count") != known_target_count
      or coverage.get("oov_target_count") != oov_target_count
      or coverage.get("classification_evaluable_sample_count")
      != evaluable_sample_count
      or coverage.get("classification_failure_sample_count")
      != len(samples) - evaluable_sample_count
      or coverage.get("rejected_query_count") != len(rejections)
      or coverage.get("unique_graph_work_count") != len(graph_by_id)
      or coverage.get("unique_endpoint_graph_count")
      != len(endpoint_cache_keys)
      or len(samples) != len(graph_work)
  ):
    raise ValueError("serialized graph-work coverage differs")
  by_split = coverage.get("by_split")
  if not isinstance(by_split, Mapping):
    raise ValueError("serialized split coverage differs")
  for split in ("train", "dev"):
    row = by_split.get(split)
    if (
        not isinstance(row, Mapping)
        or row.get("sample_count") != sample_split_counts[split]
        or row.get("target_count") != target_split_counts[split]
        or row.get("known_target_count") != known_split_counts[split]
        or row.get("oov_target_count") != oov_split_counts[split]
        or row.get("classification_evaluable_sample_count")
        != evaluable_split_counts[split]
        or row.get("classification_failure_sample_count")
        != sample_split_counts[split] - evaluable_split_counts[split]
        or row.get("unique_graph_work_count")
        != sample_split_counts[split]
    ):
      raise ValueError("serialized per-split coverage differs")


class StructurallyAndExactRebuildCheckedQueryGraphWorkIndexV2:
  """Capability issued after direct structure and same-producer rebuild checks."""

  __slots__ = ("_payload_bytes", "_source", "_authority", "_sealed")

  def __init__(
      self,
      *,
      payload_bytes: bytes,
      source: Any,
      authority: AuthenticatedQueryMateSemanticFullDomain,
      _token: object,
  ) -> None:
    if _token is not _VERIFIER_TOKEN:
      raise TypeError("serialized graph-work index is verifier-only")
    object.__setattr__(self, "_payload_bytes", bytes(payload_bytes))
    object.__setattr__(self, "_source", source)
    object.__setattr__(self, "_authority", authority)
    object.__setattr__(self, "_sealed", True)

  def __setattr__(self, _name: str, _value: Any) -> None:
    if getattr(self, "_sealed", False):
      raise AttributeError("serialized graph-work capability is immutable")
    object.__setattr__(self, _name, _value)

  def __reduce_ex__(self, _protocol: int) -> Any:
    raise TypeError("serialized graph-work capability is not serializable")

  def revalidate(self) -> None:
    _reverify_capture(self._source, label="serialized graph-work index")
    payload = json.loads(self._payload_bytes)
    _verify_serialized_structure_v2(payload)
    rebuilt = build_benchmark_v2_query_semantic_index_v2(self._authority)
    expected = dict(rebuilt.projection())
    _verify_exact_rebuild_payload(
        serialized=payload,
        serialized_bytes=self._payload_bytes,
        expected=expected,
        label="exact current rebuild",
    )

  def capture_index_artifact(self) -> CapturedFileArtifact:
    """Return a fresh public capture of the verified serialized index."""

    self.revalidate()
    captured = capture_file_artifact(
        self._source.resolved_path,
        label="serialized graph-work index public binding",
    )
    if (
        captured.resolved_path != self._source.resolved_path
        or captured.sha256 != self._source.sha256
        or captured.byte_count != self._source.byte_count
        or captured.device != self._source.device
        or captured.inode != self._source.inode
    ):
      raise ValueError("serialized graph-work index public binding differs")
    return captured

  @property
  def coverage(self) -> Mapping[str, Any]:
    self.revalidate()
    payload = json.loads(self._payload_bytes)
    return MappingProxyType(copy.deepcopy(payload["coverage"]))

  def geometry_requests(
      self,
  ) -> tuple[GeometryMaterializationRequestV2, ...]:
    """Return the strict label-free geometry materialization channel."""

    self.revalidate()
    payload = json.loads(self._payload_bytes)

    def endpoint(raw: Mapping[str, Any]) -> GeometryEndpointBindingV2:
      return GeometryEndpointBindingV2(
          step_sha256=str(raw["step"]["sha256"]),
          step_bytes=int(raw["step"]["bytes"]),
          source_path_sha256=str(raw["step"]["source_path_sha256"]),
          face_map_receipt_sha256=str(raw["face_map_receipt"]["sha256"]),
          face_map_receipt_bytes=int(raw["face_map_receipt"]["bytes"]),
          mapper_endpoint_identity_sha256=str(
              raw["mapper_endpoint_identity_sha256"]
          ),
          fusion_face_index=int(raw["fusion_face_index"]),
          raw_occ_face_indices=tuple(raw["raw_occ_face_indices"]),
          source_face_signature_sha256s=tuple(
              raw["source_face_signature_sha256s"]
          ),
          endpoint_graph_cache_key_sha256=str(
              raw["endpoint_graph_cache_key_sha256"]
          ),
      )

    return tuple(
        GeometryMaterializationRequestV2(
            graph_work_id=str(row["graph_work_id"]),
            model_view_schema_version=str(
                row["model_view_schema_version"]
            ),
            graph_schema_version=str(row["graph_schema_version"]),
            endpoint_a=endpoint(row["endpoints"]["endpoint_a"]),
            endpoint_b=endpoint(row["endpoints"]["endpoint_b"]),
        )
        for row in payload["graph_work"]
    )

  def supervision_samples(self) -> tuple[SupervisionSampleV2, ...]:
    """Return immutable identities, clusters, and labels without geometry."""

    self.revalidate()
    payload = json.loads(self._payload_bytes)

    def descriptor(raw: Mapping[str, Any]) -> ProgramDescriptorMetadataV2:
      return ProgramDescriptorMetadataV2(
          relation_hint=str(raw["relation_hint"]),
          contact_type=str(raw["contact_type"]),
          surface_type_a=str(raw["surface_type_a"]),
          surface_type_b=str(raw["surface_type_b"]),
      )

    def target(raw: Mapping[str, Any]) -> SupervisionTargetV2:
      return SupervisionTargetV2(
          program_id=str(raw["program_id"]),
          source_program_row_sha256=str(raw["source_program_row_sha256"]),
          descriptor=descriptor(raw["descriptor"]),
          descriptor_sha256=str(raw["descriptor_sha256"]),
          catalog_index=raw["catalog_index"],
          catalog_program_id=raw["catalog_program_id"],
          oov=bool(raw["oov"]),
          residual_translation=tuple(
              float(value) for value in raw["residual_translation"]
          ),
          residual_rotation_row_major=tuple(
              float(value)
              for value in raw["residual_rotation_row_major"]
          ),
          target_payload_sha256=str(raw["target_payload_sha256"]),
      )

    return tuple(
        SupervisionSampleV2(
            schema_version=str(row["schema_version"]),
            sample_identity_sha256=str(row["sample_identity_sha256"]),
            opaque_case_identity=str(row["opaque_case_identity"]),
            opaque_query_identity=str(row["opaque_query_identity"]),
            source_contact_ordinal=int(row["source_contact_ordinal"]),
            development_split=str(row["development_split"]),
            direction=str(row["direction"]),
            graph_work_id=str(row["graph_work_id"]),
            family_cluster_id=str(row["family_cluster_id"]),
            assembly_cluster_id=str(row["assembly_cluster_id"]),
            target_count=int(row["target_count"]),
            known_target_count=int(row["known_target_count"]),
            oov_target_count=int(row["oov_target_count"]),
            classification_evaluable=bool(
                row["classification_evaluable"]
            ),
            targets=tuple(target(value) for value in row["targets"]),
            sample_payload_sha256=str(row["sample_payload_sha256"]),
        )
        for row in payload["samples"]
    )

  def fixed_lineage_metadata(self) -> FixedLineageMetadataV2:
    self.revalidate()
    payload = json.loads(self._payload_bytes)
    source = payload["source_authority"]
    return FixedLineageMetadataV2(
        schema_version=str(payload["schema_version"]),
        index_payload_sha256=str(payload["index_payload_sha256"]),
        authenticated_receipt_sha256=str(
            source["authenticated_receipt_sha256"]
        ),
        receipt_payload_sha256=str(source["receipt_payload_sha256"]),
        domain_sha256=str(source["domain_sha256"]),
        source_revision=str(source["source_revision"]),
        fixed_v19_manifest_sha256=str(
            source["fixed_v19_manifest"]["sha256"]
        ),
        fixed_family_source_sha256=str(
            source["fixed_family_source"]["sha256"]
        ),
    )

  def catalog_metadata(self) -> FixedCatalogMetadataV2:
    self.revalidate()
    payload = json.loads(self._payload_bytes)
    catalog = payload["program_catalog"]
    statistics = catalog["training_statistics"]

    def descriptor(raw: Mapping[str, Any]) -> ProgramDescriptorMetadataV2:
      return ProgramDescriptorMetadataV2(
          relation_hint=str(raw["relation_hint"]),
          contact_type=str(raw["contact_type"]),
          surface_type_a=str(raw["surface_type_a"]),
          surface_type_b=str(raw["surface_type_b"]),
      )

    return FixedCatalogMetadataV2(
        schema_version=str(catalog["schema_version"]),
        catalog_payload_sha256=str(catalog["catalog_payload_sha256"]),
        catalog_roster_sha256=str(
            catalog["fixed_source"]["catalog_roster_sha256"]
        ),
        candidate_policy=str(catalog["candidate_policy"]),
        candidate_order=str(catalog["candidate_order"]),
        target_policy=str(catalog["target_policy"]),
        entry_count=int(catalog["entry_count"]),
        entries=tuple(
            CatalogEntryMetadataV2(
                program_index=int(row["program_index"]),
                program_id=str(row["program_id"]),
                program_type=str(row["program_type"]),
                descriptor=descriptor(row["descriptor"]),
                descriptor_sha256=str(row["descriptor_sha256"]),
            )
            for row in catalog["entries"]
        ),
        training_statistics=TrainCatalogStatisticsV2(
            source_split=str(statistics["source_split"]),
            source_binding_count=int(statistics["source_binding_count"]),
            source_row_count=int(statistics["source_row_count"]),
            source_commitment_sha256=str(
                statistics["source_commitment_sha256"]
            ),
            known_target_count=int(statistics["known_target_count"]),
            oov_target_count=int(statistics["oov_target_count"]),
            positive_class_count=int(statistics["positive_class_count"]),
            target_frequency_by_program_index=tuple(
                statistics["target_frequency_by_program_index"]
            ),
            class_weight_by_program_index=tuple(
                float(value)
                for value in statistics["class_weight_by_program_index"]
            ),
        ),
    )

  def _audit_payload(self) -> Mapping[str, Any]:
    """Private verifier/tool diagnostic; never a model input API."""

    self.revalidate()
    return MappingProxyType(json.loads(self._payload_bytes))


_VERIFIER_TOKEN = object()


def verify_benchmark_v2_query_semantic_index_v2(
    *,
    index_path: str | Path,
    authority: AuthenticatedQueryMateSemanticFullDomain,
) -> StructurallyAndExactRebuildCheckedQueryGraphWorkIndexV2:
  """Run strict structural checks, then an exact same-producer rebuild check."""

  source = _capture_project_file(
      index_path, label="serialized graph-work index"
  )
  allowed_root = (
      PROJECT_ROOT / "artifacts" / "development"
  ).resolve(strict=True)
  try:
    relative = source.resolved_path.relative_to(allowed_root)
  except ValueError as error:
    raise ValueError(
        "serialized graph-work index is outside artifacts/development"
    ) from error
  if source.resolved_path.name != (
      "authenticated_query_graph_work_index.v2.json"
  ) or any(
      re.search(r"(^|[_\-.])(formal|final|withheld)($|[_\-.])", part.lower())
      for part in relative.parts
  ):
    raise ValueError("serialized graph-work index name differs")
  try:
    payload = _json_from_capture(
        source, label="serialized graph-work index"
    )
  except ValueError as error:
    raise ValueError("serialized graph-work index is not strict JSON") from error

  schema_capture = _capture_project_file(
      SCHEMA_PATH, label="graph-work index schema"
  )
  schema = _json_from_capture(
      schema_capture, label="graph-work index schema"
  )
  jsonschema.Draft202012Validator.check_schema(schema)
  jsonschema.Draft202012Validator(schema).validate(payload)
  _verify_serialized_structure_v2(payload)

  rebuilt: AuthenticatedBenchmarkV2QuerySemanticIndexV2 = (
      build_benchmark_v2_query_semantic_index_v2(authority)
  )
  expected = dict(rebuilt.projection())
  _verify_exact_rebuild_payload(
      serialized=payload,
      serialized_bytes=source.raw_bytes,
      expected=expected,
      label="fixed rebuild",
  )
  _reverify_capture(schema_capture, label="graph-work index schema")
  _reverify_capture(source, label="serialized graph-work index")
  return StructurallyAndExactRebuildCheckedQueryGraphWorkIndexV2(
      payload_bytes=source.raw_bytes,
      source=source,
      authority=authority,
      _token=_VERIFIER_TOKEN,
  )
