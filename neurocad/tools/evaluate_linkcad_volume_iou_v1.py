"""Add output validity and mean part volume IoU to frozen assembly outputs."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from neurocad.cadquery_backend import load_step_shape
from neurocad.linkcad_assembly_export_v1 import _location
from neurocad.linkcad_assembly_geometry_evaluation_v1 import _matrix


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def volume_iou(left, right):
    a, b = float(left.Volume()), float(right.Volume())
    if a <= 0 or b <= 0 or not left.isValid() or not right.isValid():
        raise ValueError("invalid solid")
    common = left.intersect(right)
    if not common.isNull() and not common.isValid():
        raise ValueError("invalid intersection")
    intersection = 0.0 if common.isNull() else float(common.Volume())
    if not np.isfinite(intersection) or intersection < -1e-7 or intersection > min(a, b) + max(a,b)*1e-7:
        raise ValueError("inconsistent intersection volume")
    intersection = min(max(intersection, 0.0), a, b)
    return intersection / (a + b - intersection)


def mean_part_iou(prediction, reference, dataset_root):
    pred = {r["role_id"]: r for r in prediction["components"]}
    ref = {r["role_id"]: r for r in reference["components"]}
    if set(pred) != set(ref) or not ref:
        raise ValueError("incomplete role domain")
    anchor = reference.get("anchor_role") or min(ref)
    gauge = _matrix(ref[anchor]["world_pose_row_major"]) @ np.linalg.inv(_matrix(pred[anchor]["world_pose_row_major"]))
    values = {}
    root = Path(dataset_root).resolve()
    for role in sorted(ref):
        shapes = []
        for component, alignment in ((pred[role], gauge), (ref[role], np.eye(4))):
            path = (root / component["source_step_path"]).resolve()
            path.relative_to(root)
            shape = load_step_shape(path)
            shapes.append(shape.moved(_location(alignment @ _matrix(component["world_pose_row_major"]))))
        values[role] = volume_iou(*shapes)
    return float(np.mean(list(values.values()))), values


def evaluate_one(job):
    qid, prediction_root, reference_root, dataset_root = job
    directory = Path(prediction_root) / qid
    reference_path = Path(reference_root) / qid / "reference.manifest.json"
    row = {"query_id": qid, "valid_output": False, "mean_part_volume_iou": 0.0,
           "reference_available": reference_path.is_file()}
    if not (directory / "assembly.step").is_file():
        row["status"] = "missing_output"
        return row
    try:
        shape = load_step_shape(directory / "assembly.step")
        solids = shape.Solids()
        row["valid_output"] = bool(solids) and shape.isValid() and all(s.isValid() and s.Volume() > 0 for s in solids)
        if not row["valid_output"]:
            row["status"] = "invalid_step_output"
            return row
    except Exception as error:
        row.update(status="step_import_failure", error_type=type(error).__name__)
        return row
    if not reference_path.is_file():
        row["status"] = "reference_unavailable"
        return row
    try:
        value, roles = mean_part_iou(read(directory / "assembly.manifest.json"), read(reference_path), dataset_root)
        row.update(status="scored", mean_part_volume_iou=value, role_volume_iou=roles)
    except Exception as error:
        row.update(status="metric_failure", error_type=type(error).__name__)
    return row


def summarize(rows):
    return {"query_count": len(rows),
        "valid_output_rate": sum(r["valid_output"] for r in rows) / len(rows),
        "mean_part_volume_iou": sum(r["mean_part_volume_iou"] for r in rows) / len(rows),
        "reference_available_count": sum(r["reference_available"] for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prediction-root", "reference-root", "dataset-root", "geometry-evaluation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ids = [r["query_id"] for r in read(args.geometry_evaluation)["rows"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate queries")
    jobs = [(qid, args.prediction_root, args.reference_root, args.dataset_root) for qid in ids]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(evaluate_one, jobs):
            rows.append(row)
    result = {"schema_version": "linkcad_volume_iou_evaluation.v1",
        "prediction_root": str(args.prediction_root), "reference_root": str(args.reference_root),
        "metric_definition": "Unweighted mean role-wise solid intersection-over-union after one anchor alignment; failures and unavailable references contribute zero over all requests.",
        "validity_definition": "Reimported STEP has nonempty valid positive-volume solids; collision is evaluated separately.",
        "rows": rows, "summary": summarize(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
