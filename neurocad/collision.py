"""Collision validation for assembled part instances."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from .domain_types import AssemblyState, CollisionRecord, PartInstance


class CollisionChecker:
  """Axis-aligned broad-phase collision checker."""

  def world_aabb(self, instance: PartInstance) -> tuple[np.ndarray, np.ndarray]:
    if instance.transform is None:
      raise ValueError(
          f"Part '{instance.instance_id}' does not have a solved transform."
      )
    mins = instance.local_bbox_min
    maxs = instance.local_bbox_max
    corners = np.array(
        list(
            product(
                [mins[0], maxs[0]],
                [mins[1], maxs[1]],
                [mins[2], maxs[2]],
            )
        ),
        dtype=float,
    )
    world = (instance.transform.rotation @ corners.T).T
    world += instance.transform.translation
    return world.min(axis=0), world.max(axis=0)

  def check(
      self, assembly: AssemblyState, epsilon: float = 1e-6
  ) -> list[CollisionRecord]:
    records: list[CollisionRecord] = []
    ids = list(assembly.instances.keys())
    bounds = {
        part_id: self.world_aabb(assembly.instances[part_id]) for part_id in ids
    }

    for part_a, part_b in combinations(ids, 2):
      min_a, max_a = bounds[part_a]
      min_b, max_b = bounds[part_b]
      overlap_min = np.maximum(min_a, min_b)
      overlap_max = np.minimum(max_a, max_b)
      overlap = overlap_max - overlap_min
      if np.any(overlap <= 0.0):
        continue
      volume = float(np.prod(overlap))
      if volume <= epsilon:
        continue
      records.append(
          CollisionRecord(
              part_a=part_a,
              part_b=part_b,
              volume=volume,
              overlap_min=overlap_min,
              overlap_max=overlap_max,
          )
      )

    return records
