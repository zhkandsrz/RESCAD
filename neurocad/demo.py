"""Runnable demo for the neuro-symbolic generative CAD pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .cadquery_backend import (
    CadKernelError,
    CadQueryCollisionChecker,
    build_template_from_step,
)
from .collision import CollisionChecker
from .offline import BRepFace, SocketExtractionConfig, build_template_from_faces
from .pipeline import NeuroSymbolicCadPipeline
from .planner import LLMNeuralPlanner, RuleBasedNeuralPlanner
from .solver import SymbolicCadSolver


def _synthetic_faces() -> dict[str, list[BRepFace]]:
  return {
      "base": [
          BRepFace(
              surface_type="plane",
              area=1600.0,
              center=[0.0, 0.0, 0.0],
              normal=[0.0, 0.0, 1.0],
          ),
          BRepFace(
              surface_type="cylinder",
              area=400.0,
              center=[0.0, 0.0, 5.0],
              axis=[0.0, 0.0, 1.0],
              radius=4.0,
              concavity=-1,
              metadata={"radius_param": "hole_radius"},
          ),
      ],
      "shaft": [
          BRepFace(
              surface_type="plane",
              area=120.0,
              center=[0.0, 0.0, 0.0],
              normal=[0.0, 0.0, -1.0],
          ),
          BRepFace(
              surface_type="plane",
              area=120.0,
              center=[0.0, 0.0, 40.0],
              normal=[0.0, 0.0, 1.0],
          ),
          BRepFace(
              surface_type="cylinder",
              area=628.0,
              center=[0.0, 0.0, 20.0],
              axis=[0.0, 0.0, 1.0],
              radius=5.0,
              concavity=1,
              metadata={"radius_param": "shaft_radius"},
          ),
      ],
      "gear": [
          BRepFace(
              surface_type="plane",
              area=706.0,
              center=[0.0, 0.0, 0.0],
              normal=[0.0, 0.0, -1.0],
          ),
          BRepFace(
              surface_type="cylinder",
              area=188.0,
              center=[0.0, 0.0, 5.0],
              axis=[0.0, 0.0, 1.0],
              radius=3.0,
              concavity=-1,
              metadata={"radius_param": "bore_radius"},
          ),
      ],
  }


def build_demo_catalog(
    base_step: Optional[str] = None,
    shaft_step: Optional[str] = None,
    gear_step: Optional[str] = None,
    cad_dir: Optional[str] = None,
    try_step_loading: bool = True,
    extraction_config: Optional[SocketExtractionConfig] = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
  """Build catalog, optional STEP overrides, and optional shape library."""
  faces = _synthetic_faces()
  default_params = {
      "base": {"hole_radius": 4.0},
      "shaft": {"shaft_radius": 5.0},
      "gear": {"bore_radius": 3.0},
  }
  bboxes = {
      "base": ([-20.0, -20.0, 0.0], [20.0, 20.0, 10.0]),
      "shaft": ([-5.0, -5.0, 0.0], [5.0, 5.0, 40.0]),
      "gear": ([-15.0, -15.0, 0.0], [15.0, 15.0, 10.0]),
  }

  auto_step_map, auto_notes = _discover_step_map(cad_dir)
  step_map = {
      "base": base_step or auto_step_map.get("base"),
      "shaft": shaft_step or auto_step_map.get("shaft"),
      "gear": gear_step or auto_step_map.get("gear"),
  }
  catalog = {}
  shape_library = {}
  notes: list[str] = list(auto_notes)

  for name in ["base", "shaft", "gear"]:
    step_path = step_map[name]
    if step_path and try_step_loading:
      try:
        asset = build_template_from_step(
            name=name,
            step_path=step_path,
            default_params=default_params[name],
            extraction_config=extraction_config,
        )
        catalog[name] = asset.template
        shape_library[name] = asset.shape
        notes.append(f"loaded_step={name}:{step_path}")
        continue
      except CadKernelError as exc:
        notes.append(f"step_load_failed={name}:{step_path}:{exc}")
    elif step_path and not try_step_loading:
      notes.append(f"step_load_skipped={name}:{step_path}")

    bbox_min, bbox_max = bboxes[name]
    catalog[name] = build_template_from_faces(
        name=name,
        faces=faces[name],
        local_bbox_min=bbox_min,
        local_bbox_max=bbox_max,
        default_params=default_params[name],
        extraction_config=extraction_config,
    )
    notes.append(f"loaded_synthetic={name}")

  return catalog, shape_library, notes


def _discover_step_map(cad_dir: Optional[str]) -> tuple[dict[str, str], list[str]]:
  if not cad_dir:
    return {}, ["cad_discovery=disabled"]

  root = Path(cad_dir)
  if not root.exists() or not root.is_dir():
    return {}, [f"cad_discovery=missing:{root}"]

  files = sorted(
      [
          p
          for p in root.rglob("*")
          if p.is_file() and p.suffix.lower() in {".step", ".stp"}
      ],
      key=lambda p: p.name.lower(),
  )
  if not files:
    return {}, [f"cad_discovery=empty:{root}"]

  mapping: dict[str, str] = {}
  notes = [f"cad_discovery=count={len(files)}@{root}"]
  for path in files:
    lname = path.stem.lower()
    slot = None
    if "gear" in lname and "gear" not in mapping:
      slot = "gear"
    elif "shaft" in lname and "shaft" not in mapping:
      slot = "shaft"
    elif (
        any(k in lname for k in ["base", "bearing", "housing", "insert"])
        and "base" not in mapping
    ):
      slot = "base"
    if slot is None:
      continue
    mapping[slot] = str(path.resolve())
    notes.append(f"cad_discovery_map={slot}:{path.name}")

  return mapping, notes


def build_planner(
    planner: str,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    llm_timeout_seconds: int,
    llm_temperature: float,
    llm_temperature_step: float,
    llm_temperature_max: float,
    llm_retry_on_invalid: int,
    fallback_to_rules: bool,
    llm_max_prompt_sockets_per_part: int,
    llm_max_prompt_total_sockets: int,
    llm_use_few_shot: bool,
    llm_few_shot_example_count: int,
    llm_reasoning_mode: str,
    llm_thinking_mode: str = "disabled",
    planning_mode: str = "topology_graph",
    model_input_protocol: str = "legacy",
):
  planner = planner.lower().strip()
  if planner == "rule":
    if str(model_input_protocol).strip().lower() == "benchmark_v2":
      raise ValueError(
          "benchmark_v2 requires the leakage-safe topology-graph LLM planner; "
          "the metadata-sensitive rule planner is not allowed"
      )
    return RuleBasedNeuralPlanner(default_clearance=1.0)
  if planner in {"openai", "anthropic", "ollama", "deepseek"}:
    return LLMNeuralPlanner(
        provider=planner,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=llm_timeout_seconds,
        temperature=llm_temperature,
        temperature_step=llm_temperature_step,
        temperature_max=llm_temperature_max,
        llm_retry_on_invalid=llm_retry_on_invalid,
        fallback_to_rules=fallback_to_rules,
        max_prompt_sockets_per_part=llm_max_prompt_sockets_per_part,
        max_prompt_total_sockets=llm_max_prompt_total_sockets,
        use_few_shot=llm_use_few_shot,
        few_shot_example_count=llm_few_shot_example_count,
        reasoning_mode=llm_reasoning_mode,
        llm_thinking_mode=llm_thinking_mode,
        planning_mode=planning_mode,
        model_input_protocol=model_input_protocol,
    )
  raise ValueError(
      "planner must be one of: rule, openai, anthropic, ollama, deepseek"
  )


def build_collision_checker(
    kernel: str,
    shape_library: dict[str, Any],
    allow_fallback: bool,
    cadquery_exact_timeout_seconds: float = 8.0,
    cadquery_skip_exact_for_complex_pairs: bool = True,
    cadquery_complexity_face_threshold: int = 180,
    cadquery_complexity_score_threshold: float = 240.0,
    cadquery_complex_pair_aabb_ratio_threshold: float = 0.02,
):
  kernel = kernel.lower().strip()
  if kernel == "aabb":
    return CollisionChecker(), ["collision_kernel=aabb"]
  if kernel != "cadquery":
    raise ValueError("kernel must be one of: aabb, cadquery")

  if not shape_library:
    return CollisionChecker(), ["collision_kernel=aabb (no step shapes loaded)"]

  try:
    checker = CadQueryCollisionChecker(
        shape_library=shape_library,
        allow_aabb_fallback=allow_fallback,
        exact_timeout_seconds=cadquery_exact_timeout_seconds,
        skip_exact_for_complex_pairs=cadquery_skip_exact_for_complex_pairs,
        complexity_face_threshold=cadquery_complexity_face_threshold,
        complexity_score_threshold=cadquery_complexity_score_threshold,
        complex_pair_aabb_ratio_threshold=(
            cadquery_complex_pair_aabb_ratio_threshold
        ),
    )
  except CadKernelError:
    if not allow_fallback:
      raise
    return CollisionChecker(), ["collision_kernel=aabb (cadquery unavailable)"]
  return checker, [
      "collision_kernel=cadquery_boolean",
      f"cadquery_exact_timeout_seconds={cadquery_exact_timeout_seconds}",
      (
          "cadquery_skip_exact_for_complex_pairs="
          f"{bool(cadquery_skip_exact_for_complex_pairs)}"
      ),
  ]


def build_extraction_config(args) -> SocketExtractionConfig:
  return SocketExtractionConfig(
      max_hole_sockets=args.socket_max_holes,
      max_pin_sockets=args.socket_max_pins,
      max_threaded_hole_sockets=args.socket_max_threads,
      max_plane_sockets=args.socket_max_planes,
      max_flange_sockets=args.socket_max_flanges,
      max_guide_slot_sockets=args.socket_max_guides,
      min_plane_area_ratio=args.socket_min_plane_area_ratio,
      min_cylinder_area_ratio=args.socket_min_cyl_area_ratio,
  )


def run_demo(args):
  extraction_config = build_extraction_config(args)
  catalog, shape_library, catalog_notes = build_demo_catalog(
      base_step=args.base_step,
      shaft_step=args.shaft_step,
      gear_step=args.gear_step,
      cad_dir=args.cad_dir,
      try_step_loading=args.try_step_loading,
      extraction_config=extraction_config,
  )
  planner = build_planner(
      planner=args.planner,
      model=args.model,
      api_key=None,
      base_url=args.base_url,
      llm_timeout_seconds=args.llm_timeout_seconds,
      llm_temperature=args.llm_temperature,
      llm_temperature_step=args.llm_temperature_step,
      llm_temperature_max=args.llm_temperature_max,
      llm_retry_on_invalid=args.llm_retry_on_invalid,
      fallback_to_rules=args.fallback_to_rules,
      llm_max_prompt_sockets_per_part=args.llm_max_prompt_sockets_per_part,
      llm_max_prompt_total_sockets=args.llm_max_prompt_total_sockets,
      llm_use_few_shot=args.llm_use_few_shot,
      llm_few_shot_example_count=args.llm_few_shot_example_count,
      llm_reasoning_mode=args.llm_reasoning_mode,
      llm_thinking_mode=args.llm_thinking_mode,
  )
  collision_checker, kernel_notes = build_collision_checker(
      kernel=args.kernel,
      shape_library=shape_library,
      allow_fallback=args.kernel_fallback_to_aabb,
      cadquery_exact_timeout_seconds=args.cadquery_exact_timeout_seconds,
      cadquery_skip_exact_for_complex_pairs=(
          args.cadquery_skip_exact_for_complex_pairs
      ),
      cadquery_complexity_face_threshold=(
          args.cadquery_complexity_face_threshold
      ),
      cadquery_complexity_score_threshold=(
          args.cadquery_complexity_score_threshold
      ),
      cadquery_complex_pair_aabb_ratio_threshold=(
          args.cadquery_complex_pair_aabb_ratio_threshold
      ),
  )
  solver = SymbolicCadSolver(radius_clearance=0.2)
  pipeline = NeuroSymbolicCadPipeline(
      planner=planner,
      solver=solver,
      collision_checker=collision_checker,
      collision_epsilon=args.collision_epsilon,
      mate_collision_volume_tolerance=args.mate_collision_volume_tolerance,
  )
  result = pipeline.run(
      instruction=args.instruction,
      catalog=catalog,
      max_iterations=args.max_iterations,
  )
  metadata = result.to_dict()
  metadata["runtime_notes"] = catalog_notes + kernel_notes
  return metadata


def _validated_path(path: Optional[str]) -> Optional[str]:
  if not path:
    return None
  p = Path(path)
  if not p.exists():
    raise ValueError(f"STEP file does not exist: {path}")
  return str(p.resolve())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--instruction",
      default="Assemble shaft into the base, then mount gear onto shaft.",
      help="Natural language assembly request.",
  )
  parser.add_argument(
      "--max_iterations", type=int, default=4, help="Closed-loop max retries."
  )
  parser.add_argument(
      "--planner",
      default="ollama",
      choices=["rule", "openai", "anthropic", "ollama", "deepseek"],
      help="Neural planner provider.",
  )
  parser.add_argument("--model", default=None, help="LLM model name override.")
  parser.add_argument(
      "--base_url",
      default=None,
      help="Optional provider base URL override.",
  )
  parser.add_argument(
      "--llm_timeout_seconds", type=int, default=60, help="LLM request timeout."
  )
  parser.add_argument(
      "--llm_temperature", type=float, default=0.22, help="LLM base decoding temp."
  )
  parser.add_argument(
      "--llm_temperature_step",
      type=float,
      default=0.12,
      help="Adaptive temperature increment per retry/failure repeat.",
  )
  parser.add_argument(
      "--llm_temperature_max",
      type=float,
      default=0.75,
      help="Max adaptive temperature.",
  )
  parser.add_argument(
      "--llm_retry_on_invalid",
      type=int,
      default=3,
      help="Extra LLM retries within one iteration when output is invalid.",
  )
  parser.add_argument(
      "--llm_max_prompt_sockets_per_part",
      type=int,
      default=18,
      help="Max sockets per part exposed to LLM.",
  )
  parser.add_argument(
      "--llm_max_prompt_total_sockets",
      type=int,
      default=56,
      help="Global max sockets exposed to LLM prompt.",
  )
  parser.add_argument(
      "--llm_no_few_shot",
      action="store_true",
      help="Disable few-shot examples in prompt.",
  )
  parser.add_argument(
      "--llm_few_shot_example_count",
      type=int,
      default=2,
      help="How many built-in few-shot examples to include in prompt.",
  )
  parser.add_argument(
      "--llm_reasoning_mode",
      choices=["none", "brief"],
      default="brief",
      help="Reasoning prompt mode: brief asks model for short plan_summary.",
  )
  parser.add_argument(
      "--llm_thinking_mode",
      choices=["disabled", "two_stage"],
      default="disabled",
      help=(
          "DeepSeek thinking ablation mode. Final JSON emission always disables "
          "thinking."
      ),
  )
  parser.add_argument(
      "--no_fallback_to_rules",
      action="store_true",
      help="Disable fallback to rule-based planner when LLM call fails.",
  )
  parser.add_argument(
      "--kernel",
      default="cadquery",
      choices=["aabb", "cadquery"],
      help="Collision kernel.",
  )
  parser.add_argument(
      "--no_kernel_fallback_to_aabb",
      action="store_true",
      help="Disable fallback to AABB collision when CadQuery is unavailable.",
  )
  parser.add_argument(
      "--collision_epsilon",
      type=float,
      default=1e-5,
      help="Minimum collision volume threshold.",
  )
  parser.add_argument(
      "--mate_collision_volume_tolerance",
      type=float,
      default=50.0,
      help=(
          "Allowed collision volume for already-mated hole-pin / hole-axis pairs. "
          "Other collisions remain strict."
      ),
  )
  parser.add_argument(
      "--cadquery_exact_timeout_seconds",
      type=float,
      default=8.0,
      help="Hard timeout for exact OCC boolean in child process.",
  )
  parser.add_argument(
      "--cadquery_no_skip_exact_for_complex_pairs",
      action="store_true",
      help="Always try exact boolean even for complex pairs.",
  )
  parser.add_argument(
      "--cadquery_complexity_face_threshold",
      type=int,
      default=180,
      help="Face-count threshold that marks a part as geometrically complex.",
  )
  parser.add_argument(
      "--cadquery_complexity_score_threshold",
      type=float,
      default=240.0,
      help="Complexity score threshold that marks a part as geometrically complex.",
  )
  parser.add_argument(
      "--cadquery_complex_pair_aabb_ratio_threshold",
      type=float,
      default=0.02,
      help=(
          "For complex pairs below this AABB overlap ratio, skip exact boolean "
          "and use conservative fallback."
      ),
  )
  parser.add_argument("--base_step", default=None, help="Path to base STEP.")
  parser.add_argument("--shaft_step", default=None, help="Path to shaft STEP.")
  parser.add_argument("--gear_step", default=None, help="Path to gear STEP.")
  parser.add_argument(
      "--cad_dir",
      default="neurocad/cad",
      help="Directory to auto-discover STEP files (for base/shaft/gear mapping).",
  )
  parser.add_argument(
      "--no_try_step_loading",
      action="store_true",
      help="Disable loading STEP geometry even if step paths are discovered.",
  )
  parser.add_argument(
      "--socket_max_holes",
      type=int,
      default=18,
      help="Max extracted hole sockets per part.",
  )
  parser.add_argument(
      "--socket_max_pins",
      type=int,
      default=10,
      help="Max extracted pin sockets per part.",
  )
  parser.add_argument(
      "--socket_max_threads",
      type=int,
      default=8,
      help="Max extracted threaded-hole sockets per part.",
  )
  parser.add_argument(
      "--socket_max_planes",
      type=int,
      default=12,
      help="Max extracted plane sockets per part.",
  )
  parser.add_argument(
      "--socket_max_flanges",
      type=int,
      default=4,
      help="Max extracted flange-plane sockets per part.",
  )
  parser.add_argument(
      "--socket_max_guides",
      type=int,
      default=6,
      help="Max extracted guide-slot sockets per part.",
  )
  parser.add_argument(
      "--socket_min_plane_area_ratio",
      type=float,
      default=0.012,
      help="Plane area floor as ratio of largest plane area.",
  )
  parser.add_argument(
      "--socket_min_cyl_area_ratio",
      type=float,
      default=0.001,
      help="Cylinder area floor as ratio of largest cylinder area.",
  )
  args = parser.parse_args()

  args.base_step = _validated_path(args.base_step)
  args.shaft_step = _validated_path(args.shaft_step)
  args.gear_step = _validated_path(args.gear_step)
  args.fallback_to_rules = not bool(args.no_fallback_to_rules)
  args.kernel_fallback_to_aabb = not bool(args.no_kernel_fallback_to_aabb)
  args.try_step_loading = not bool(args.no_try_step_loading)
  args.llm_use_few_shot = not bool(args.llm_no_few_shot)
  args.cadquery_skip_exact_for_complex_pairs = not bool(
      args.cadquery_no_skip_exact_for_complex_pairs
  )

  result = run_demo(args)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
