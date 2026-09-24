from __future__ import annotations

import math

import numpy as np

from neurocad.linkcad_pose_equivalence_v1 import (
    mobility_quotient_errors_v1,
)


_ENDPOINT = {
    "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
    "primary_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
    "secondary_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
}


def _pose(*, yaw_degrees: float = 0.0, translation=(0.0, 0.0, 0.0)):
  angle = math.radians(yaw_degrees)
  matrix = np.eye(4)
  matrix[:3, :3] = np.asarray((
      (math.cos(angle), -math.sin(angle), 0.0),
      (math.sin(angle), math.cos(angle), 0.0),
      (0.0, 0.0, 1.0),
  ))
  matrix[:3, 3] = translation
  return matrix


def test_revolute_quotient_accepts_yaw_but_not_translation():
  accepted = mobility_quotient_errors_v1(
      source_child_world=np.eye(4),
      predicted_child_world=_pose(yaw_degrees=90.0),
      child_endpoint_world=_ENDPOINT,
      mobility="revolute",
  )
  rejected = mobility_quotient_errors_v1(
      source_child_world=np.eye(4),
      predicted_child_world=_pose(yaw_degrees=90.0, translation=(0.0, 0.0, 1.0)),
      child_endpoint_world=_ENDPOINT,
      mobility="revolute",
  )
  assert accepted["equivalent"] is True
  assert rejected["equivalent"] is False


def test_prismatic_quotient_accepts_axis_translation_but_not_yaw():
  accepted = mobility_quotient_errors_v1(
      source_child_world=np.eye(4),
      predicted_child_world=_pose(translation=(0.0, 0.0, 25.0)),
      child_endpoint_world=_ENDPOINT,
      mobility="prismatic",
  )
  rejected = mobility_quotient_errors_v1(
      source_child_world=np.eye(4),
      predicted_child_world=_pose(yaw_degrees=10.0),
      child_endpoint_world=_ENDPOINT,
      mobility="prismatic",
  )
  assert accepted["equivalent"] is True
  assert rejected["equivalent"] is False


def test_cylindrical_quotient_accepts_axis_translation_and_yaw():
  result = mobility_quotient_errors_v1(
      source_child_world=np.eye(4),
      predicted_child_world=_pose(
          yaw_degrees=120.0, translation=(0.0, 0.0, 25.0),
      ),
      child_endpoint_world=_ENDPOINT,
      mobility="cylindrical",
  )
  assert result["equivalent"] is True
