"""Interpret accepted near-rigid LLM matrices consistently at kernel import.

This numerical conversion uses no reference, interface, or collision information.
The v1 acceptance tolerance is unchanged. Raw predictions remain immutable.
"""
from __future__ import annotations

import copy
import numpy as np
from .linkcad_pure_llm_execution_v1 import _matrix


def normalize_prediction_v2(prediction):
    result = copy.deepcopy(prediction)
    audit = {"schema_version": "linkcad_llm_rotation_normalization.v2",
             "rule": "closest proper rotation by polar/SVD projection within the unchanged v1 matrix acceptance tolerance",
             "translation_unchanged": True, "candidate_and_interface_ids_unchanged": True,
             "reference_or_geometric_feedback_used": False, "roles": {}}
    for role, raw in prediction["role_pose_row_major"].items():
        matrix = _matrix(raw).copy()
        rotation = matrix[:3, :3].copy()
        u, _, vt = np.linalg.svd(rotation)
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(u @ vt)
        normalized = u @ correction @ vt
        audit["roles"][role] = {
            "raw_max_orthogonality_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
            "max_rotation_entry_change": float(np.max(np.abs(normalized - rotation))),
            "rotation_change_frobenius_norm": float(np.linalg.norm(normalized - rotation)),
        }
        matrix[:3, :3] = normalized
        matrix[3] = (0.0, 0.0, 0.0, 1.0)
        result["role_pose_row_major"][role] = matrix.reshape(-1).tolist()
    return result, audit
