"""Export a verified LinkCAD result as a product-structured STEP assembly.

The learned system already returns a role graph and one world pose per selected
part.  This module packages that result as named STEP component occurrences and
writes the richer interface and motion semantics to a JSON sidecar.  The STEP
file is re-imported through XCAF before either output is published.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from .benchmark_v2_training_provenance import capture_file_artifact
from .cadquery_backend import load_step_shape


MANIFEST_SCHEMA_VERSION = "linkcad_assembly_manifest.v1"
RECEIPT_SCHEMA_VERSION = "linkcad_assembly_export_receipt.v1"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json_sha256(value: Any) -> str:
  encoded = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _matrix(row_major: Sequence[float]) -> np.ndarray:
  if len(row_major) != 16:
    raise ValueError("LinkCAD assembly pose must contain sixteen values")
  matrix = np.asarray(row_major, dtype=float).reshape(4, 4)
  if not np.isfinite(matrix).all() or not np.allclose(
      matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8,
  ):
    raise ValueError("LinkCAD assembly pose is not homogeneous")
  rotation = matrix[:3, :3]
  if (
      not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
      or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
  ):
    raise ValueError("LinkCAD assembly pose is not rigid")
  return matrix


def _safe_name(value: str) -> str:
  cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
  return cleaned or "assembly"


def _location(matrix: np.ndarray):
  import cadquery as cq
  from OCP.gp import gp_Trsf

  transform = gp_Trsf()
  transform.SetValues(
      float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]),
      float(matrix[0, 3]),
      float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]),
      float(matrix[1, 3]),
      float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]),
      float(matrix[2, 3]),
  )
  return cq.Location(transform)


def _label_name(label: Any) -> str:
  from OCP.TDataStd import TDataStd_Name

  attribute = TDataStd_Name()
  if not label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
    return ""
  return str(attribute.Get().ToExtString())


def inspect_assembly_step_v1(step_path: str | Path) -> dict[str, Any]:
  """Re-import a STEP file and report its direct product structure."""

  from OCP.BinXCAFDrivers import BinXCAFDrivers
  from OCP.IFSelect import IFSelect_ReturnStatus
  from OCP.STEPCAFControl import STEPCAFControl_Reader
  from OCP.TCollection import TCollection_ExtendedString
  from OCP.TDF import TDF_LabelSequence
  from OCP.TDocStd import TDocStd_Document
  from OCP.XCAFApp import XCAFApp_Application
  from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

  path = Path(step_path).resolve()
  if not path.is_file():
    raise ValueError(f"LinkCAD assembly STEP does not exist: {path}")
  application = XCAFApp_Application.GetApplication_s()
  BinXCAFDrivers.DefineFormat_s(application)
  document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
  application.InitDocument(document)
  reader = STEPCAFControl_Reader()
  reader.SetNameMode(True)
  if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
    raise ValueError("LinkCAD assembly STEP re-import did not parse")
  if not reader.Transfer(document):
    raise ValueError("LinkCAD assembly STEP re-import did not transfer")

  shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
  roots = TDF_LabelSequence()
  shape_tool.GetFreeShapes(roots)
  root_is_assembly = (
      roots.Length() == 1 and XCAFDoc_ShapeTool.IsAssembly_s(roots.Value(1))
  )
  components = TDF_LabelSequence()
  if root_is_assembly:
    XCAFDoc_ShapeTool.GetComponents_s(roots.Value(1), components, False)

  rows = []
  for component in components:
    location = XCAFDoc_ShapeTool.GetLocation_s(component).Transformation()
    matrix = [
        float(location.Value(row, column))
        for row in range(1, 4) for column in range(1, 5)
    ] + [0.0, 0.0, 0.0, 1.0]
    rows.append({
        "name": _label_name(component),
        "world_pose_row_major": matrix,
        "translation_mm": [matrix[3], matrix[7], matrix[11]],
        "is_reference": XCAFDoc_ShapeTool.IsReference_s(component),
    })
  rows.sort(key=lambda row: row["name"])
  return {
      "schema_version": "linkcad_assembly_step_inspection.v1",
      "step_sha256": _sha256(path),
      "root_count": roots.Length(),
      "root_is_assembly": root_is_assembly,
      "root_name": _label_name(roots.Value(1)) if roots.Length() == 1 else None,
      "component_count": len(rows),
      "component_names": [row["name"] for row in rows],
      "translation_by_component_mm": {
          row["name"]: row["translation_mm"] for row in rows
      },
      "components": rows,
  }


def _candidate_index(
    query: Mapping[str, Any], hypothesis: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
  result = {}
  selected = hypothesis.get("candidate_by_role")
  if not isinstance(selected, Mapping):
    raise ValueError("LinkCAD assembly hypothesis lacks candidate assignments")
  for role in query["roles"]:
    role_id = str(role["role_id"])
    by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in query["candidate_sets"][role_id]
    }
    candidate_id = str(selected[role_id])
    if candidate_id not in by_id:
      raise ValueError("LinkCAD assembly selected candidate differs")
    result[role_id] = by_id[candidate_id]
  if set(result) != set(selected):
    raise ValueError("LinkCAD assembly role domain differs")
  return result


def _connections(
    *, query: Mapping[str, Any], hypothesis: Mapping[str, Any],
    pair_execution_rows: Sequence[Mapping[str, Any]],
    selected_pose_rank_by_edge: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
  programs = {
      str(row["edge_id"]): row for row in hypothesis.get("edge_programs", ())
  }
  pair_rows = {}
  for row in pair_execution_rows:
    if (
        str(row.get("query_id")) == str(query["query_id"])
        and int(row.get("hypothesis_rank", -1)) == int(hypothesis.get("rank", 0))
        and row.get("conditional_kinematic_feasibility_accepted") is True
    ):
      edge_id = str(row["edge_id"])
      if edge_id in pair_rows:
        raise ValueError("LinkCAD assembly accepted edge evidence duplicates")
      pair_rows[edge_id] = row

  result = []
  for edge in query["functional_edges"]:
    edge_id = str(edge["edge_id"])
    if edge_id not in programs or edge_id not in pair_rows:
      raise ValueError("LinkCAD assembly connection evidence is incomplete")
    program = programs[edge_id]
    pair_row = pair_rows[edge_id]
    observation = pair_row.get("selected_observation")
    if selected_pose_rank_by_edge is not None:
      if edge_id not in selected_pose_rank_by_edge:
        raise ValueError("LinkCAD assembly global pose selection is incomplete")
      selected_rank = int(selected_pose_rank_by_edge[edge_id])
      alternatives = pair_row.get("accepted_observations")
      if isinstance(alternatives, list):
        matches = [
            row for row in alternatives
            if isinstance(row, Mapping)
            and int(row.get("pose_rank", -1)) == selected_rank
        ]
        if len(matches) != 1:
          raise ValueError("LinkCAD assembly selected global pose differs")
        observation = matches[0]
      elif (
          not isinstance(observation, Mapping)
          or int(observation.get("pose_rank", 0)) != selected_rank
      ):
        raise ValueError("LinkCAD assembly selected global pose is unavailable")
    if not isinstance(observation, Mapping):
      raise ValueError("LinkCAD assembly connection lacks a selected pose")
    result.append({
        "edge_id": edge_id,
        "role_a": str(edge["role_a"]),
        "role_b": str(edge["role_b"]),
        "instruction": edge.get("instruction"),
        "requested_mobility": edge.get("requested_mobility"),
        "predicted_mobility": program.get("mobility"),
        "support_family": program.get("support_family"),
        "interface_primitive_a": program.get("interface_primitive_a"),
        "interface_primitive_b": program.get("interface_primitive_b"),
        "selected_observation": dict(observation),
    })
  return result


def export_assembly_package_v1(
    *, query: Mapping[str, Any], hypothesis: Mapping[str, Any],
    global_execution: Mapping[str, Any],
    pair_execution_rows: Sequence[Mapping[str, Any]],
    dataset_root: str | Path, output_step: str | Path,
    output_manifest: str | Path,
    artifact_role: str = "prediction",
) -> dict[str, Any]:
  """Write and re-import-verify a LinkCAD STEP assembly plus JSON manifest."""

  import cadquery as cq

  if artifact_role not in {"prediction", "evaluation_reference"}:
    raise ValueError("LinkCAD assembly artifact role differs")
  if (
      global_execution.get("query_id") != query.get("query_id")
      or global_execution.get("private_targets_opened") is not False
      or global_execution.get("status") != "accepted_global_assembly"
      or global_execution.get("global_assembly_conditionally_feasible") is not True
  ):
    raise ValueError("LinkCAD assembly export requires one accepted public result")
  composition = global_execution.get("composition")
  if not isinstance(composition, Mapping) or composition.get("status") != "composed":
    raise ValueError("LinkCAD assembly export requires composed role poses")

  candidates = _candidate_index(query, hypothesis)
  pose_rows = composition.get("role_pose_row_major")
  if not isinstance(pose_rows, Mapping) or set(pose_rows) != set(candidates):
    raise ValueError("LinkCAD assembly composed pose domain differs")
  connections = _connections(
      query=query, hypothesis=hypothesis,
      pair_execution_rows=pair_execution_rows,
      selected_pose_rank_by_edge=global_execution.get(
          "selected_pose_rank_by_edge"
      ),
  )

  root = Path(dataset_root).resolve()
  root_name = f"linkcad_{_safe_name(str(query['query_id']))}"
  assembly = cq.Assembly(name=root_name)
  components = []
  for role_id, candidate in sorted(candidates.items()):
    source = (root / str(candidate["step_path"])).resolve()
    try:
      source.relative_to(root)
    except ValueError as error:
      raise ValueError("LinkCAD assembly STEP escapes dataset root") from error
    capture = capture_file_artifact(source, label="LinkCAD assembly source STEP")
    if capture.sha256 != str(candidate["step_sha256"]):
      raise ValueError("LinkCAD assembly source STEP bytes differ")
    matrix = _matrix(pose_rows[role_id])
    assembly.add(
        load_step_shape(source), name=role_id, loc=_location(matrix),
    )
    components.append({
        "role_id": role_id,
        "candidate_id": str(candidate["candidate_id"]),
        "source_step_path": str(candidate["step_path"]),
        "source_step_sha256": capture.sha256,
        "world_pose_row_major": [float(value) for value in matrix.reshape(-1)],
    })

  output_step_path = Path(output_step).resolve()
  output_manifest_path = Path(output_manifest).resolve()
  output_step_path.parent.mkdir(parents=True, exist_ok=True)
  output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
  token = uuid.uuid4().hex
  temporary_step = output_step_path.with_name(
      f".{output_step_path.stem}.{token}.tmp{output_step_path.suffix}"
  )
  temporary_manifest = output_manifest_path.with_name(
      f".{output_manifest_path.name}.{token}.tmp"
  )
  try:
    assembly.export(
        str(temporary_step), exportType="STEP", mode="default", unit="MM",
        outputUnit="MM",
    )
    inspection = inspect_assembly_step_v1(temporary_step)
    expected_names = sorted(candidates)
    if (
        inspection["root_count"] != 1
        or inspection["root_is_assembly"] is not True
        or inspection["component_names"] != expected_names
    ):
      raise ValueError("LinkCAD assembly STEP product structure differs")
    inspected = {row["name"]: row for row in inspection["components"]}
    for role_id in expected_names:
      if not np.allclose(
          inspected[role_id]["world_pose_row_major"],
          _matrix(pose_rows[role_id]).reshape(-1), atol=1e-6,
      ):
        raise ValueError("LinkCAD assembly STEP occurrence pose differs")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "query_id": str(query["query_id"]),
        "root_assembly_name": root_name,
        "unit": "mm",
        "anchor_role": composition.get("anchor_role"),
        "components": components,
        "connections": connections,
        "verification": {
            "global_execution_status": global_execution["status"],
            "global_assembly_conditionally_feasible": True,
            "step_reimport": inspection,
        },
        "assembly_step_sha256": inspection["step_sha256"],
    }
    manifest["manifest_payload_sha256"] = _json_sha256(manifest)
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_step.replace(output_step_path)
    temporary_manifest.replace(output_manifest_path)
  finally:
    temporary_step.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)

  receipt = {
      "schema_version": RECEIPT_SCHEMA_VERSION,
      "artifact_role": artifact_role,
      "status": "exported_and_reimport_verified",
      "query_id": str(query["query_id"]),
      "output_step": str(output_step_path),
      "output_step_sha256": _sha256(output_step_path),
      "output_manifest": str(output_manifest_path),
      "output_manifest_sha256": _sha256(output_manifest_path),
      "component_count": len(components),
      "connection_count": len(connections),
      "reimport_inspection": inspection,
  }
  receipt["receipt_payload_sha256"] = _json_sha256(receipt)
  return receipt
