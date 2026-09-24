"""Offline semantic socket extraction from B-Rep descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

from .math3d import as_vector, normalize
from .domain_types import PartTemplate, Socket


@dataclass
class BRepFace:
  """Face descriptor used by socket extraction heuristics."""

  surface_type: str
  area: float
  center: np.ndarray
  normal: Optional[np.ndarray] = None
  axis: Optional[np.ndarray] = None
  radius: Optional[float] = None
  concavity: int = 0  # <0 concave, >0 convex
  edge_count: int = 0
  has_helical_edge: bool = False
  bbox_extents: Optional[np.ndarray] = None
  metadata: dict[str, object] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.surface_type = self.surface_type.lower().strip()
    self.center = as_vector(self.center)
    self.area = float(self.area)
    self.concavity = int(np.sign(self.concavity))
    self.edge_count = int(self.edge_count)
    self.has_helical_edge = bool(self.has_helical_edge)
    if self.normal is not None:
      self.normal = normalize(self.normal)
    if self.axis is not None:
      self.axis = normalize(self.axis)
    if self.radius is not None:
      self.radius = float(self.radius)
    if self.bbox_extents is not None:
      self.bbox_extents = as_vector(self.bbox_extents)


@dataclass
class SocketExtractionConfig:
  # Frame protocol. Legacy preserves world-AABB heuristics; benchmark_v2 uses
  # only rigid-invariant/equivariant ranking and clustering inputs.
  frame_protocol: str = "legacy"

  # Primitive filtering.
  min_area: float = 1e-3
  min_plane_area_ratio: float = 0.012
  min_cylinder_area_ratio: float = 0.001
  min_cylinder_radius_abs: float = 0.05
  min_cylinder_radius_ratio: float = 0.0008

  # Geometric similarity.
  plane_similarity_dot: float = 0.95
  coaxial_dot: float = 0.96
  axis_cluster_step: float = 0.18
  origin_merge_abs: float = 0.2
  origin_merge_ratio: float = 0.01
  cluster_linear_ratio: float = 0.08
  radius_merge_abs: float = 0.15
  radius_merge_rel: float = 0.06

  # Semantic structure.
  flange_min_holes: int = 3
  guide_slot_aspect_ratio: float = 3.0
  max_guide_slot_aspect_ratio: float = 40.0
  min_guide_slot_gap_abs: float = 0.2
  min_guide_slot_gap_ratio: float = 0.0015
  threaded_hole_radius_max: float = 30.0
  min_guide_slot_plane_area_ratio: float = 0.03
  guide_slot_plane_budget: int = 16
  non_mating_plane_edge_threshold: int = 18

  # Count caps to avoid feature explosion.
  max_hole_sockets: int = 18
  max_pin_sockets: int = 10
  max_threaded_hole_sockets: int = 8
  max_plane_sockets: int = 12
  max_flange_sockets: int = 4
  max_guide_slot_sockets: int = 6

  # Proxy socket fallback for spline-heavy / weakly-annotated industrial parts.
  enable_proxy_sockets: bool = True
  proxy_min_total_mating_sockets: int = 4
  proxy_axis_min_aspect_ratio: float = 1.55
  proxy_spline_ratio_trigger: float = 0.32


@dataclass
class _SocketCandidate:
  socket: Socket
  score: float


def extract_semantic_sockets(
    faces: list[BRepFace], min_area: float = 1e-3
) -> dict[str, Socket]:
  """Backward-compatible socket extraction entrypoint."""
  config = SocketExtractionConfig(min_area=min_area)
  return extract_semantic_sockets_robust(faces, config=config)


def extract_semantic_sockets_robust(
    faces: list[BRepFace],
    config: Optional[SocketExtractionConfig] = None,
) -> dict[str, Socket]:
  """Robust extraction of sockets from face primitives.

  Extracted socket kinds include:
  - `hole` / `pin`
  - `threaded_hole`
  - `plane` / `flange_plane`
  - `guide_slot`
  """
  cfg = config or SocketExtractionConfig()
  if not faces:
    return {}
  frame_protocol = str(cfg.frame_protocol or "legacy").strip().lower()
  if frame_protocol not in {"legacy", "benchmark_v2"}:
    raise ValueError("socket frame_protocol must be 'legacy' or 'benchmark_v2'")

  if frame_protocol == "benchmark_v2":
    model_center, model_diag = _estimate_intrinsic_model_frame(faces)
  else:
    model_center, model_diag = _estimate_model_frame(faces)
  sockets: dict[str, Socket] = {}

  raw_plane_faces = [
      face
      for face in faces
      if face.surface_type == "plane"
      and face.normal is not None
      and face.area >= cfg.min_area
      and not _is_non_mating_face(face, cfg, candidate_kind="plane")
  ]
  raw_cyl_faces = [
      face
      for face in faces
      if face.surface_type == "cylinder"
      and face.axis is not None
      and face.area >= cfg.min_area
  ]

  max_plane_area = max((face.area for face in raw_plane_faces), default=cfg.min_area)
  plane_area_floor = max(cfg.min_area, max_plane_area * cfg.min_plane_area_ratio)
  plane_faces = [face for face in raw_plane_faces if face.area >= plane_area_floor]

  max_cyl_area = max((face.area for face in raw_cyl_faces), default=cfg.min_area)
  cyl_area_floor = max(cfg.min_area, max_cyl_area * cfg.min_cylinder_area_ratio)
  cyl_radius_floor = max(
      cfg.min_cylinder_radius_abs,
      model_diag * cfg.min_cylinder_radius_ratio,
  )
  cyl_faces: list[BRepFace] = []
  for face in raw_cyl_faces:
    if face.area < cyl_area_floor:
      continue
    if face.radius is not None and face.radius < cyl_radius_floor:
      continue
    cyl_faces.append(face)

  # Candidate sockets from plane faces.
  plane_candidates: list[_SocketCandidate] = []
  for face in plane_faces:
    score = _score_plane_face(face, model_center=model_center, model_diag=model_diag)
    candidate = Socket(
        name="plane",
        kind="plane",
        origin=face.center,
        normal=face.normal,
        x_axis=_intrinsic_face_reference(face, face.normal),
        metadata={
            **dict(face.metadata),
            "source": "plane_face",
            "face_area": face.area,
            "quality": score,
            "mating_enabled": True,
        },
    )
    plane_candidates.append(_SocketCandidate(socket=candidate, score=score))

  selected_planes = _compact_candidates(
      plane_candidates,
      max_count=cfg.max_plane_sockets,
      config=cfg,
      model_center=model_center,
      model_diag=model_diag,
  )

  base_plane_socket: Optional[Socket] = None
  if selected_planes:
    base_plane_socket = max(selected_planes, key=lambda c: c.score).socket.copy()
  elif plane_faces:
    base_face = max(plane_faces, key=lambda face: face.area)
    base_score = _score_plane_face(
        base_face, model_center=model_center, model_diag=model_diag
    )
    base_plane_socket = Socket(
        name="base_plane",
        kind="plane",
        origin=base_face.center,
        normal=base_face.normal,
        x_axis=_intrinsic_face_reference(base_face, base_face.normal),
        metadata={
            **dict(base_face.metadata),
            "source": "largest_plane",
            "face_area": base_face.area,
            "quality": base_score,
        },
    )
  if base_plane_socket is not None:
    base_plane_socket.name = "base_plane"
    base_plane_socket.metadata["source"] = "largest_plane"
    sockets["base_plane"] = base_plane_socket

  # Candidate sockets from cylinder faces.
  cylinder_candidates: dict[str, list[_SocketCandidate]] = {
      "hole": [],
      "pin": [],
      "threaded_hole": [],
  }
  for face in cyl_faces:
    kind = _classify_cylinder_face(face, cfg)
    score = _score_cylinder_face(
        face,
        kind=kind,
        model_center=model_center,
        model_diag=model_diag,
    )
    metadata = {
        **dict(face.metadata),
        "source": "cylinder_face",
        "face_area": face.area,
        "quality": score,
        "mating_enabled": True,
    }
    if kind == "threaded_hole":
      metadata["source"] = "cylinder_thread"
      metadata["mating_enabled"] = False
      metadata["requires_fastener_context"] = True
      metadata["non_mating_surface"] = True
    candidate = Socket(
        name=kind,
        kind=kind,
        origin=face.center,
        axis=face.axis,
        x_axis=_intrinsic_face_reference(face, face.axis),
        radius=face.radius,
        metadata=metadata,
    )
    cylinder_candidates[kind].append(_SocketCandidate(candidate, score))

  selected_holes = _compact_candidates(
      cylinder_candidates["hole"],
      max_count=cfg.max_hole_sockets,
      config=cfg,
      model_center=model_center,
      model_diag=model_diag,
  )
  selected_pins = _compact_candidates(
      cylinder_candidates["pin"],
      max_count=cfg.max_pin_sockets,
      config=cfg,
      model_center=model_center,
      model_diag=model_diag,
  )
  selected_threads = _compact_candidates(
      cylinder_candidates["threaded_hole"],
      max_count=cfg.max_threaded_hole_sockets,
      config=cfg,
      model_center=model_center,
      model_diag=model_diag,
  )

  # Flange faces: selected plane aligns with at least N hole/threaded_hole axes.
  hole_like = [
      candidate.socket
      for candidate in (selected_holes + selected_threads)
      if candidate.socket.axis is not None
  ]
  flange_candidates: list[_SocketCandidate] = []
  for plane_candidate in selected_planes:
    plane_socket = plane_candidate.socket
    if plane_socket.normal is None:
      continue
    aligned_count = 0
    for hole_socket in hole_like:
      if abs(float(np.dot(plane_socket.normal, hole_socket.axis))) >= cfg.coaxial_dot:
        aligned_count += 1
    if aligned_count < cfg.flange_min_holes:
      continue
    score = plane_candidate.score + 0.35 * float(aligned_count)
    flange_socket = plane_socket.copy()
    flange_socket.kind = "flange_plane"
    flange_socket.metadata = {
        **dict(flange_socket.metadata),
        "source": "plane_with_hole_pattern",
        "aligned_hole_count": aligned_count,
        "quality": score,
    }
    flange_candidates.append(_SocketCandidate(flange_socket, score))

  selected_flanges = _compact_candidates(
      flange_candidates,
      max_count=cfg.max_flange_sockets,
      config=cfg,
      model_center=model_center,
      model_diag=model_diag,
  )

  # Guide slots from top planes only to avoid O(N^2) explosion on dense models.
  guide_faces = sorted(plane_faces, key=lambda face: face.area, reverse=True)
  if frame_protocol == "benchmark_v2":
    # The legacy slot heuristic infers an in-plane axis from world-AABB face
    # extents. Until an intrinsic edge-loop slot descriptor replaces it, the
    # formal protocol must exclude these candidates.
    guide_faces = []
  guide_faces = [
      face
      for face in guide_faces
      if face.area >= max_plane_area * cfg.min_guide_slot_plane_area_ratio
  ]
  guide_faces = guide_faces[: max(0, cfg.guide_slot_plane_budget)]
  slot_candidates: list[_SocketCandidate] = []
  for face_a, face_b in combinations(guide_faces, 2):
    if _is_non_mating_face(face_a, cfg, candidate_kind="guide_slot"):
      continue
    if _is_non_mating_face(face_b, cfg, candidate_kind="guide_slot"):
      continue
    if abs(float(np.dot(face_a.normal, face_b.normal))) < cfg.plane_similarity_dot:
      continue
    gap = float(abs(np.dot(face_a.center - face_b.center, face_a.normal)))
    min_gap = max(cfg.min_guide_slot_gap_abs, model_diag * cfg.min_guide_slot_gap_ratio)
    if gap <= min_gap:
      continue

    ext_a = face_a.bbox_extents
    ext_b = face_b.bbox_extents
    if ext_a is None or ext_b is None:
      continue
    mean_ext = (ext_a + ext_b) / 2.0
    planar_dims = sorted(
        [float(value) for value in mean_ext if float(value) > min_gap * 0.5],
        reverse=True,
    )
    if len(planar_dims) < 2:
      continue
    long_dim = planar_dims[0]
    short_dim = planar_dims[1]
    aspect = long_dim / short_dim
    if aspect < cfg.guide_slot_aspect_ratio:
      continue
    if aspect > cfg.max_guide_slot_aspect_ratio:
      continue
    if gap > short_dim * 1.5:
      continue

    origin = (face_a.center + face_b.center) / 2.0
    axis_hint = _slot_axis_from_extents(mean_ext, face_a.normal)
    score = (
        float(np.log1p(face_a.area + face_b.area))
        + 0.7 * aspect
        - 0.25 * (gap / max(short_dim, 1e-8))
    )
    slot_socket = Socket(
        name="guide_slot",
        kind="guide_slot",
        origin=origin,
        axis=axis_hint,
        normal=face_a.normal,
        x_axis=_intrinsic_face_reference(face_a, axis_hint),
        metadata={
            "source": "parallel_plane_pair",
            "gap": gap,
            "aspect": aspect,
            "quality": score,
        },
    )
    slot_candidates.append(_SocketCandidate(slot_socket, score))

  selected_slots = _compact_candidates(
      slot_candidates,
      max_count=cfg.max_guide_slot_sockets,
      config=cfg,
      model_center=model_center,
      model_diag=model_diag,
  )

  hole_idx = 0
  pin_idx = 0
  thread_idx = 0
  plane_idx = 0
  flange_idx = 0
  slot_idx = 0

  for candidate in selected_planes:
    socket = candidate.socket.copy()
    if base_plane_socket is not None and _socket_similar(
        socket, base_plane_socket, cfg, model_diag
    ):
      continue
    socket.name = f"plane_{plane_idx:02d}"
    plane_idx += 1
    sockets[socket.name] = socket

  for candidate in selected_holes:
    socket = candidate.socket.copy()
    socket.name = f"hole_{hole_idx:02d}"
    hole_idx += 1
    sockets[socket.name] = socket

  for candidate in selected_pins:
    socket = candidate.socket.copy()
    socket.name = f"pin_{pin_idx:02d}"
    pin_idx += 1
    sockets[socket.name] = socket

  for candidate in selected_threads:
    socket = candidate.socket.copy()
    socket.name = f"threaded_hole_{thread_idx:02d}"
    thread_idx += 1
    sockets[socket.name] = socket

  for candidate in selected_flanges:
    socket = candidate.socket.copy()
    socket.name = f"flange_plane_{flange_idx:02d}"
    flange_idx += 1
    sockets[socket.name] = socket

  for candidate in selected_slots:
    socket = candidate.socket.copy()
    socket.name = f"guide_slot_{slot_idx:02d}"
    slot_idx += 1
    sockets[socket.name] = socket

  if cfg.enable_proxy_sockets:
    _maybe_add_proxy_sockets(
        sockets=sockets,
        faces=faces,
        config=cfg,
    )

  for socket in sockets.values():
    _attach_frame_variants(socket)

  return sockets


def build_template_from_faces(
    name: str,
    faces: list[BRepFace],
    local_bbox_min: np.ndarray,
    local_bbox_max: np.ndarray,
    default_params: Optional[dict[str, float]] = None,
    metadata: Optional[dict[str, object]] = None,
    extraction_config: Optional[SocketExtractionConfig] = None,
) -> PartTemplate:
  return PartTemplate(
      name=name,
      sockets=extract_semantic_sockets_robust(faces, config=extraction_config),
      local_bbox_min=np.asarray(local_bbox_min, dtype=float),
      local_bbox_max=np.asarray(local_bbox_max, dtype=float),
      default_params={} if default_params is None else dict(default_params),
      metadata={} if metadata is None else dict(metadata),
  )


def _estimate_model_frame(faces: list[BRepFace]) -> tuple[np.ndarray, float]:
  mins: list[np.ndarray] = []
  maxs: list[np.ndarray] = []
  for face in faces:
    center = face.center
    if face.bbox_extents is None:
      mins.append(center)
      maxs.append(center)
      continue
    half = 0.5 * face.bbox_extents
    mins.append(center - half)
    maxs.append(center + half)
  if not mins:
    return np.zeros(3, dtype=float), 1.0
  bbox_min = np.min(np.stack(mins, axis=0), axis=0)
  bbox_max = np.max(np.stack(maxs, axis=0), axis=0)
  center = (bbox_min + bbox_max) / 2.0
  diag = float(np.linalg.norm(bbox_max - bbox_min))
  return center, max(diag, 1e-6)


def _estimate_intrinsic_model_frame(
    faces: list[BRepFace],
) -> tuple[np.ndarray, float]:
  """Equivariant surface centroid plus rigid-invariant surface length scale."""

  positive = [face for face in faces if float(face.area) > 0.0]
  if not positive:
    raise ValueError("benchmark_v2 socket extraction requires positive face area")
  total_area = sum(float(face.area) for face in positive)
  center = sum(
      (float(face.area) * np.asarray(face.center, dtype=float) for face in positive),
      start=np.zeros(3, dtype=float),
  ) / total_area
  return center, max(float(np.sqrt(total_area)), 1e-6)


def _estimate_model_bbox(
    faces: list[BRepFace],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  mins: list[np.ndarray] = []
  maxs: list[np.ndarray] = []
  for face in faces:
    center = face.center
    if face.bbox_extents is None:
      mins.append(center)
      maxs.append(center)
      continue
    half = 0.5 * face.bbox_extents
    mins.append(center - half)
    maxs.append(center + half)
  if not mins:
    zeros = np.zeros(3, dtype=float)
    return zeros, zeros, np.ones(3, dtype=float)
  bbox_min = np.min(np.stack(mins, axis=0), axis=0)
  bbox_max = np.max(np.stack(maxs, axis=0), axis=0)
  dims = np.maximum(bbox_max - bbox_min, 1e-6)
  return bbox_min, bbox_max, dims


def _maybe_add_proxy_sockets(
    sockets: dict[str, Socket],
    faces: list[BRepFace],
    config: SocketExtractionConfig,
) -> None:
  real_mating = [
      socket
      for socket in sockets.values()
      if socket.kind in {"hole", "pin", "plane", "flange_plane", "guide_slot", "axis"}
      and not bool(socket.metadata.get("proxy_socket", False))
      and bool(socket.metadata.get("mating_enabled", True))
  ]
  spline_like_count = sum(
      1
      for face in faces
      if any(
          token in face.surface_type
          for token in ("spline", "bspline", "bezier", "nurbs")
      )
  )
  spline_ratio = float(spline_like_count) / max(1, len(faces))
  bbox_min, bbox_max, dims = _estimate_model_bbox(faces)
  long_idx = int(np.argmax(dims))
  short_dims = [float(dims[idx]) for idx in range(3) if idx != long_idx]
  long_dim = float(dims[long_idx])
  short_dim = max(short_dims) if short_dims else long_dim
  aspect_ratio = long_dim / max(short_dim, 1e-6)
  if (
      len(real_mating) >= int(config.proxy_min_total_mating_sockets)
      and spline_ratio < float(config.proxy_spline_ratio_trigger)
  ):
    return

  axis = np.zeros(3, dtype=float)
  axis[long_idx] = 1.0
  center = (bbox_min + bbox_max) / 2.0
  end_origin_lo = center.copy()
  end_origin_lo[long_idx] = bbox_min[long_idx]
  end_origin_hi = center.copy()
  end_origin_hi[long_idx] = bbox_max[long_idx]

  if (
      aspect_ratio >= float(config.proxy_axis_min_aspect_ratio)
      and "axis_00" not in sockets
  ):
    sockets["axis_00"] = Socket(
        name="axis_00",
        kind="axis",
        origin=center,
        axis=axis,
        metadata={
            "source": "proxy_principal_axis",
            "proxy_socket": True,
            "quality": 2.25 + min(2.0, 0.3 * aspect_ratio),
            "aspect_ratio": aspect_ratio,
            "mating_enabled": True,
        },
    )

  proxy_planes = [
      (
          "proxy_plane_00",
          end_origin_lo,
          -axis,
      ),
      (
          "proxy_plane_01",
          end_origin_hi,
          axis,
      ),
  ]
  for name, origin, normal in proxy_planes:
    if name in sockets:
      continue
    sockets[name] = Socket(
        name=name,
        kind="plane",
        origin=origin,
        normal=normal,
        metadata={
            "source": "proxy_bbox_plane",
            "proxy_socket": True,
            "quality": 1.75,
            "aspect_ratio": aspect_ratio,
            "mating_enabled": True,
        },
    )


def _intrinsic_face_reference(
    face: BRepFace,
    preferred_z: Optional[np.ndarray],
) -> Optional[np.ndarray]:
  """Return an intrinsic in-plane direction carried by the source B-Rep face."""

  if preferred_z is None:
    return None
  raw = face.metadata.get("reference_direction")
  if raw is None:
    return None
  try:
    direction = as_vector(np.asarray(raw, dtype=float))
    z_axis = normalize(preferred_z)
    tangent = direction - float(np.dot(direction, z_axis)) * z_axis
    if float(np.linalg.norm(tangent)) <= 1e-10:
      return None
    return normalize(tangent)
  except (TypeError, ValueError):
    return None


def _attach_frame_variants(socket: Socket) -> None:
  variants = _socket_frame_variants(socket)
  if not variants:
    return
  socket.metadata["frame_variants"] = variants
  socket.metadata["primary_frame"] = "primary"
  socket.metadata["frame_variant_count"] = len(variants)
  socket.metadata["frame_variant_names"] = sorted(variants.keys())


def _socket_frame_variants(socket: Socket) -> dict[str, dict[str, object]]:
  variants: dict[str, dict[str, object]] = {
      "primary": _frame_payload(
          origin=socket.origin,
          axis=socket.axis,
          normal=socket.normal,
          x_axis=socket.x_axis,
          y_axis=socket.y_axis,
          z_axis=socket.z_axis,
          radius=socket.radius,
      )
  }
  if socket.kind in {"plane", "flange_plane", "guide_slot"} and socket.normal is not None:
    variants["support"] = _frame_payload(
        origin=socket.origin,
        axis=None,
        normal=socket.normal,
        x_axis=socket.x_axis,
        y_axis=socket.y_axis,
        z_axis=socket.z_axis,
        radius=socket.radius,
    )
    return variants

  if socket.kind not in {"hole", "threaded_hole", "pin", "axis"} or socket.axis is None:
    return variants

  length_est = _estimate_socket_length(socket)
  if length_est <= 1e-4:
    return variants
  half_span = 0.5 * float(length_est)
  axis = normalize(socket.axis)
  plus_origin = socket.origin + axis * half_span
  minus_origin = socket.origin - axis * half_span
  x_axis = socket.x_axis
  y_axis = socket.y_axis

  variants["end_plus"] = _frame_payload(
      origin=plus_origin,
      axis=axis,
      normal=axis,
      x_axis=x_axis,
      y_axis=y_axis,
      z_axis=axis,
      radius=socket.radius,
  )
  variants["end_minus"] = _frame_payload(
      origin=minus_origin,
      axis=-axis,
      normal=-axis,
      x_axis=x_axis,
      y_axis=None if y_axis is None else -y_axis,
      z_axis=-axis,
      radius=socket.radius,
  )
  if socket.kind in {"hole", "threaded_hole"}:
    variants["entry_plus"] = dict(variants["end_plus"])
    variants["entry_minus"] = dict(variants["end_minus"])
    variants["seat_plus"] = dict(variants["end_plus"])
    variants["seat_minus"] = dict(variants["end_minus"])
    variants["through_plus"] = dict(variants["end_minus"])
    variants["through_minus"] = dict(variants["end_plus"])
    variants["through_stop_plus"] = dict(variants["through_plus"])
    variants["through_stop_minus"] = dict(variants["through_minus"])
    variants["hole_depth_stop_plus"] = dict(variants["through_plus"])
    variants["hole_depth_stop_minus"] = dict(variants["through_minus"])
  elif socket.kind in {"pin", "axis"}:
    variants["tip_plus"] = dict(variants["end_plus"])
    variants["tip_minus"] = dict(variants["end_minus"])
    variants["stop_plus"] = dict(variants["end_minus"])
    variants["stop_minus"] = dict(variants["end_plus"])
    variants["shoulder_plus"] = dict(variants["stop_plus"])
    variants["shoulder_minus"] = dict(variants["stop_minus"])
    variants["pin_shoulder_plus"] = dict(variants["stop_plus"])
    variants["pin_shoulder_minus"] = dict(variants["stop_minus"])
  return variants


def _frame_payload(
    *,
    origin: np.ndarray,
    axis: Optional[np.ndarray],
    normal: Optional[np.ndarray],
    x_axis: Optional[np.ndarray],
    y_axis: Optional[np.ndarray],
    z_axis: Optional[np.ndarray],
    radius: Optional[float],
) -> dict[str, object]:
  return {
      "origin": np.asarray(origin, dtype=float).tolist(),
      "axis": None if axis is None else np.asarray(axis, dtype=float).tolist(),
      "normal": (
          None if normal is None else np.asarray(normal, dtype=float).tolist()
      ),
      "x_axis": (
          None if x_axis is None else np.asarray(x_axis, dtype=float).tolist()
      ),
      "y_axis": (
          None if y_axis is None else np.asarray(y_axis, dtype=float).tolist()
      ),
      "z_axis": (
          None if z_axis is None else np.asarray(z_axis, dtype=float).tolist()
      ),
      "radius": None if radius is None else float(radius),
  }


def _estimate_socket_length(socket: Socket) -> float:
  face_area = socket.metadata.get("face_area")
  radius = socket.radius
  if (
      isinstance(face_area, (int, float))
      and isinstance(radius, (int, float))
      and float(radius) > 1e-6
  ):
    lateral = max(0.0, float(face_area))
    length = lateral / max(2.0 * np.pi * float(radius), 1e-6)
    if length > 1e-4:
      return float(length)
  bbox_extents = socket.metadata.get("bbox_extents")
  if isinstance(bbox_extents, (list, tuple)) and len(bbox_extents) >= 3:
    dims = [abs(float(value)) for value in bbox_extents[:3]]
    if dims:
      return max(dims)
  return 0.0


def _score_plane_face(
    face: BRepFace, model_center: np.ndarray, model_diag: float
) -> float:
  centrality = _centrality(face.center, None, model_center, model_diag)
  return float(np.log1p(face.area) + 0.75 * centrality)


def _is_non_mating_face(
    face: BRepFace,
    config: SocketExtractionConfig,
    candidate_kind: str,
) -> bool:
  surface_type = face.surface_type.lower().strip()
  if face.has_helical_edge:
    return True
  if any(token in surface_type for token in ("spline", "bspline", "bezier", "nurbs")):
    return True
  if candidate_kind in {"plane", "guide_slot"}:
    if face.edge_count >= max(8, int(config.non_mating_plane_edge_threshold)):
      return True
  return bool(face.metadata.get("non_mating_surface", False))


def _classify_cylinder_face(
    face: BRepFace, config: SocketExtractionConfig
) -> str:
  is_thread_hint = bool(face.metadata.get("thread_hint", False))
  is_threaded = (
      is_thread_hint
      or face.has_helical_edge
      or (
          face.concavity < 0
          and face.radius is not None
          and face.radius <= config.threaded_hole_radius_max
          and face.edge_count >= 8
      )
  )
  if face.concavity < 0 and is_threaded:
    return "threaded_hole"
  return "hole" if face.concavity < 0 else "pin"


def _score_cylinder_face(
    face: BRepFace,
    kind: str,
    model_center: np.ndarray,
    model_diag: float,
) -> float:
  centrality = _centrality(face.center, face.axis, model_center, model_diag)
  score = float(np.log1p(face.area) + 2.1 * centrality)
  if kind in {"hole", "threaded_hole"}:
    score += 0.65
  if kind == "threaded_hole":
    score += 0.6
  if face.has_helical_edge:
    score += 0.35
  if face.edge_count in {2, 3, 4, 6, 8}:
    score += 0.2
  if face.radius is not None:
    score += 0.2 * min(face.radius / max(model_diag, 1e-6), 1.0)
  return score


def _centrality(
    origin: np.ndarray,
    axis: Optional[np.ndarray],
    model_center: np.ndarray,
    model_diag: float,
) -> float:
  delta = origin - model_center
  if axis is not None:
    axial = float(np.dot(delta, axis))
    radial_vec = delta - axial * axis
    radial = float(np.linalg.norm(radial_vec))
  else:
    radial = float(np.linalg.norm(delta))
  scale = max(0.5 * model_diag, 1e-6)
  return max(0.0, 1.0 - radial / scale)


def _compact_candidates(
    candidates: list[_SocketCandidate],
    max_count: int,
    config: SocketExtractionConfig,
    model_center: np.ndarray,
    model_diag: float,
) -> list[_SocketCandidate]:
  if max_count <= 0 or not candidates:
    return []

  if str(config.frame_protocol or "legacy").strip().lower() == "benchmark_v2":
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (-float(item[1].score), int(item[0])),
    )
    selected: list[_SocketCandidate] = []
    for _, candidate in ordered:
      if any(
          _socket_similar(candidate.socket, kept.socket, config, model_diag)
          for kept in selected
      ):
        continue
      selected.append(candidate)
      if len(selected) >= max_count:
        break
    return selected

  deduped_by_key: dict[tuple[object, ...], _SocketCandidate] = {}
  for candidate in candidates:
    key = _socket_cluster_key(
        candidate.socket, config, model_center=model_center, model_diag=model_diag
    )
    previous = deduped_by_key.get(key)
    if previous is None or candidate.score > previous.score:
      deduped_by_key[key] = candidate

  ordered = sorted(deduped_by_key.values(), key=lambda c: c.score, reverse=True)
  selected: list[_SocketCandidate] = []
  for candidate in ordered:
    if any(
        _socket_similar(candidate.socket, kept.socket, config, model_diag)
        for kept in selected
    ):
      continue
    selected.append(candidate)
    if len(selected) >= max_count:
      break
  return selected


def _socket_cluster_key(
    socket: Socket,
    config: SocketExtractionConfig,
    model_center: np.ndarray,
    model_diag: float,
) -> tuple[object, ...]:
  linear_step = max(
      config.origin_merge_abs, model_diag * config.cluster_linear_ratio
  )
  origin_delta = socket.origin - model_center

  if socket.axis is None:
    axis_key = (0, 0, 0)
    axial_bin = 0
    radial_bin = 0
  else:
    axis = _canonical_direction(socket.axis)
    axis_key = _vector_key(axis, config.axis_cluster_step)
    axial = float(np.dot(origin_delta, axis))
    radial = float(np.linalg.norm(origin_delta - axial * axis))
    axial_bin = _quantize(axial, linear_step)
    radial_bin = _quantize(radial, linear_step)

  if socket.normal is None:
    normal_key = (0, 0, 0)
    offset_bin = 0
  else:
    normal = _canonical_direction(socket.normal)
    normal_key = _vector_key(normal, config.axis_cluster_step)
    offset_bin = _quantize(
        float(np.dot(origin_delta, normal)),
        linear_step,
    )

  if socket.radius is None:
    radius_bin = -1
  else:
    radius_step = max(
        config.radius_merge_abs,
        abs(socket.radius) * config.radius_merge_rel,
    )
    radius_bin = _quantize(socket.radius, radius_step)

  return (
      socket.kind,
      axis_key,
      normal_key,
      axial_bin,
      radial_bin,
      offset_bin,
      radius_bin,
  )


def _socket_similar(
    socket_a: Socket,
    socket_b: Socket,
    config: SocketExtractionConfig,
    model_diag: float,
) -> bool:
  origin_tol = max(config.origin_merge_abs, model_diag * config.origin_merge_ratio)
  if float(np.linalg.norm(socket_a.origin - socket_b.origin)) > origin_tol:
    return False

  if socket_a.axis is not None and socket_b.axis is not None:
    dot = abs(float(np.dot(socket_a.axis, socket_b.axis)))
    if dot < config.coaxial_dot:
      return False

  if socket_a.normal is not None and socket_b.normal is not None:
    dot = abs(float(np.dot(socket_a.normal, socket_b.normal)))
    if dot < config.plane_similarity_dot:
      return False

  if socket_a.radius is not None and socket_b.radius is not None:
    radius_tol = max(
        config.radius_merge_abs,
        max(abs(socket_a.radius), abs(socket_b.radius)) * config.radius_merge_rel,
    )
    if abs(socket_a.radius - socket_b.radius) > radius_tol:
      return False

  return True


def _canonical_direction(direction: np.ndarray) -> np.ndarray:
  axis = normalize(direction)
  index = int(np.argmax(np.abs(axis)))
  if axis[index] < 0:
    axis = -axis
  return axis


def _vector_key(direction: np.ndarray, step: float) -> tuple[int, int, int]:
  step = max(abs(step), 1e-6)
  return tuple(int(round(float(value) / step)) for value in direction)


def _quantize(value: float, step: float) -> int:
  step = max(abs(step), 1e-6)
  return int(round(float(value) / step))


def _slot_axis_from_extents(
    extents: np.ndarray, face_normal: np.ndarray
) -> Optional[np.ndarray]:
  # We only have extents (not principal directions); keep normal and leave axis
  # as a stable heuristic aligned with global largest extent axis.
  axis_index = int(np.argmax(extents))
  candidate = np.zeros(3, dtype=float)
  candidate[axis_index] = 1.0
  if abs(float(np.dot(candidate, face_normal))) > 0.9:
    # Avoid near-collinearity with normal for slot axis.
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, face_normal))) > 0.9:
      candidate = np.array([0.0, 1.0, 0.0], dtype=float)
  try:
    return normalize(candidate)
  except ValueError:
    return None
