import copy
import cadquery as cq
import numpy as np
import pytest
from neurocad.tools.evaluate_linkcad_volume_iou_v1 import volume_iou, mean_part_iou, evaluate_one, summarize


def pose(x):
    result = np.eye(4)
    result[0, 3] = x
    return result.reshape(-1).tolist()


def test_analytic_box_intersections():
    box = cq.Workplane("XY").box(1, 1, 1).val()
    assert volume_iou(box, box) == pytest.approx(1)
    assert volume_iou(box, box.translate((2, 0, 0))) == pytest.approx(0)
    assert volume_iou(box, box.translate((.5, 0, 0))) == pytest.approx(1/3)


def test_one_anchor_alignment_preserves_relative_error(tmp_path):
    cq.exporters.export(cq.Workplane("XY").box(1, 1, 1), str(tmp_path / "box.step"))
    ref = {"anchor_role": "a", "components": [
        {"role_id": "a", "source_step_path": "box.step", "world_pose_row_major": pose(0)},
        {"role_id": "b", "source_step_path": "box.step", "world_pose_row_major": pose(3)}]}
    pred = copy.deepcopy(ref)
    pred["components"][0]["world_pose_row_major"] = pose(8)
    pred["components"][1]["world_pose_row_major"] = pose(11.5)
    value, roles = mean_part_iou(pred, ref, tmp_path)
    assert roles["a"] == pytest.approx(1)
    assert roles["b"] == pytest.approx(1/3)
    assert value == pytest.approx(2/3)


def test_missing_outputs_remain_in_denominator(tmp_path):
    row = evaluate_one(("missing", tmp_path, tmp_path, tmp_path))
    summary = summarize([row, {"query_id": "valid", "valid_output": True,
        "reference_available": True, "mean_part_volume_iou": 1, "status": "scored"}])
    assert summary["valid_output_rate"] == .5
    assert summary["mean_part_volume_iou"] == .5
