"""Frozen official STEP-LLM adapter for prospective LinkCAD transfer tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any


SCHEMA_VERSION = "linkcad_step_llm_generator.v1"
STEP_LLM_ADAPTER_REPO = "JasonShiii/step-llm-llama3b-no_rag"
STEP_LLM_ADAPTER_REVISION = "b40c8b3a4acb3b41836b07debf85006f3536c1cd"
STEP_LLM_BASE_REPO = "unsloth/Llama-3.2-3B-Instruct"
STEP_LLM_BASE_REVISION = "006f5dcd1393c3add266de40994ba96225e9689d"

NO_RAG_PROMPT = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description, generate a STEP file that accurately represents the described object.

### caption:
{}

### output:
{}"""

STEP_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION( ( '' ), ' ' );
FILE_NAME( '/vol/tmp/translate-8579754438183730235/5ae5839f3947920fcf80d878.step', '2018-04-29T08:34:40', ( '' ), ( '' ), ' ', ' ', ' ' );
FILE_SCHEMA( ( 'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }' ) );
ENDSEC;"""


def build_no_rag_prompt_v1(caption: str) -> str:
  caption = str(caption).strip()
  if not caption:
    raise ValueError("STEP-LLM caption is empty")
  return NO_RAG_PROMPT.format(caption, "")


def extract_step_file_v1(decoded: str) -> str:
  """Extract one complete model-emitted DATA section and prepend its header."""

  text = str(decoded)
  if "### output:" in text:
    text = text.split("### output:")[-1]
  data_start = text.find("DATA;")
  if data_start < 0:
    raise ValueError("STEP-LLM output lacks a DATA section")
  terminator = "END-ISO-10303-21;"
  data_end = text.find(terminator, data_start)
  if data_end < 0:
    raise ValueError("STEP-LLM output lacks the STEP terminator")
  data = text[data_start:data_end + len(terminator)].strip()
  return STEP_HEADER + "\n" + data + "\n"


def inspect_step_candidate_v1(*, path: Path, family: str) -> dict[str, Any]:
  """Require a single positive solid, replay the encoder, and expose its interface."""

  import cadquery as cq
  from OCP.BRepAdaptor import BRepAdaptor_Surface

  from .linkcad_primitive_graph_v2 import extract_primitive_graph_v2

  workplane = cq.importers.importStep(str(path))
  solids = workplane.solids().vals()
  if len(solids) != 1 or solids[0].Volume() <= 0.0:
    raise ValueError("STEP-LLM candidate requires one positive-volume solid")
  solid = solids[0]
  graph = extract_primitive_graph_v2(solid)
  radii = sorted(
      float(BRepAdaptor_Surface(face.wrapped).Cylinder().Radius())
      for face in solid.Faces() if face.geomType() == "CYLINDER"
  )
  if not radii:
    raise ValueError("STEP-LLM candidate lacks a cylindrical interface")
  if family not in {"ring", "shaft"}:
    raise ValueError("STEP-LLM candidate family differs")
  return {
      "solid_count": 1,
      "volume_mm3": float(solid.Volume()),
      "area_mm2": float(solid.Area()),
      "face_count": len(solid.Faces()),
      "primitive_orbit_count": int(graph.primitive_features.shape[0]),
      "cylinder_radii_mm": radii,
      "interface_radius_mm": radii[0],
      "interface_role": "bore" if family == "ring" else "shaft",
  }


@dataclass(frozen=True, slots=True)
class FrozenStepLLMRuntimeV1:
  model: Any
  tokenizer: Any
  base_path: Path
  adapter_path: Path


@dataclass(frozen=True, slots=True)
class GeneratedStepV1:
  step_text: str
  generated_token_count: int
  wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class GeneratedStepAttemptV1:
  step_text: str | None
  generated_token_count: int
  wall_time_seconds: float
  terminal_status: str
  failure_message: str | None


def load_frozen_step_llm_v1(
    *, base_path: Path, adapter_path: Path,
) -> FrozenStepLLMRuntimeV1:
  import torch
  from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer

  base_path = base_path.resolve()
  adapter_path = adapter_path.resolve()
  if not base_path.is_dir() or not adapter_path.is_dir():
    raise FileNotFoundError("STEP-LLM local model directory is missing")
  tokenizer = AutoTokenizer.from_pretrained(
      adapter_path, local_files_only=True, use_fast=True,
  )
  base = AutoModelForCausalLM.from_pretrained(
      base_path,
      local_files_only=True,
      dtype=torch.bfloat16,
      device_map={"": 0},
      low_cpu_mem_usage=True,
      attn_implementation="sdpa",
  )
  model = PeftModel.from_pretrained(
      base, adapter_path, local_files_only=True, is_trainable=False,
  )
  model.eval()
  return FrozenStepLLMRuntimeV1(
      model=model, tokenizer=tokenizer,
      base_path=base_path, adapter_path=adapter_path,
  )


def generate_step_v1(
    runtime: FrozenStepLLMRuntimeV1,
    *, caption: str, seed: int, max_new_tokens: int = 8192,
    temperature: float = 0.7, top_p: float = 0.9,
    do_sample: bool = True,
) -> GeneratedStepV1:
  import torch

  if max_new_tokens < 1:
    raise ValueError("STEP-LLM generation budget must be positive")
  if do_sample and (temperature <= 0.0 or not 0.0 < top_p <= 1.0):
    raise ValueError("STEP-LLM sampling parameters differ")
  prompt = build_no_rag_prompt_v1(caption)
  inputs = runtime.tokenizer(prompt, return_tensors="pt")
  inputs = {key: value.to(runtime.model.device) for key, value in inputs.items()}
  torch.manual_seed(int(seed))
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(seed))
  generation = {
      "max_new_tokens": int(max_new_tokens),
      "do_sample": bool(do_sample),
      "use_cache": True,
      "pad_token_id": runtime.tokenizer.eos_token_id,
      "stop_strings": ["END-ISO-10303-21;"],
      "tokenizer": runtime.tokenizer,
  }
  if do_sample:
    generation.update({"temperature": float(temperature), "top_p": float(top_p)})
  started = time.perf_counter()
  with torch.inference_mode():
    output = runtime.model.generate(**inputs, **generation)
  elapsed = time.perf_counter() - started
  input_tokens = int(inputs["input_ids"].shape[1])
  continuation = output[0, input_tokens:]
  decoded = runtime.tokenizer.decode(continuation, skip_special_tokens=True)
  return GeneratedStepV1(
      step_text=extract_step_file_v1(decoded),
      generated_token_count=int(continuation.shape[0]),
      wall_time_seconds=elapsed,
  )


def generate_step_batch_v1(
    runtime: FrozenStepLLMRuntimeV1,
    *, captions: list[str] | tuple[str, ...], seed: int,
    max_new_tokens: int = 8192, do_sample: bool = False,
    temperature: float = 0.7, top_p: float = 0.9,
) -> tuple[GeneratedStepAttemptV1, ...]:
  """Generate an ordered batch while preserving per-candidate failure rows."""

  import torch

  if not captions or max_new_tokens < 1:
    raise ValueError("STEP-LLM batch generation domain differs")
  if do_sample and (temperature <= 0.0 or not 0.0 < top_p <= 1.0):
    raise ValueError("STEP-LLM batch sampling parameters differ")
  prompts = [build_no_rag_prompt_v1(caption) for caption in captions]
  tokenizer = runtime.tokenizer
  tokenizer.padding_side = "left"
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  inputs = tokenizer(prompts, return_tensors="pt", padding=True)
  inputs = {key: value.to(runtime.model.device) for key, value in inputs.items()}
  torch.manual_seed(int(seed))
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(seed))
  generation = {
      "max_new_tokens": int(max_new_tokens),
      "do_sample": bool(do_sample),
      "use_cache": True,
      "pad_token_id": tokenizer.pad_token_id,
      "stop_strings": ["END-ISO-10303-21;"],
      "tokenizer": tokenizer,
  }
  if do_sample:
    generation.update({"temperature": float(temperature), "top_p": float(top_p)})
  started = time.perf_counter()
  with torch.inference_mode():
    output = runtime.model.generate(**inputs, **generation)
  elapsed = time.perf_counter() - started
  input_tokens = int(inputs["input_ids"].shape[1])
  rows = []
  for sequence in output:
    continuation = sequence[input_tokens:]
    decoded = tokenizer.decode(continuation, skip_special_tokens=True)
    try:
      step_text = extract_step_file_v1(decoded)
    except ValueError as error:
      rows.append(GeneratedStepAttemptV1(
          step_text=None,
          generated_token_count=int(continuation.shape[0]),
          wall_time_seconds=elapsed,
          terminal_status="incomplete_step_text",
          failure_message=str(error),
      ))
    else:
      rows.append(GeneratedStepAttemptV1(
          step_text=step_text,
          generated_token_count=int(continuation.shape[0]),
          wall_time_seconds=elapsed,
          terminal_status="complete_step_text",
          failure_message=None,
      ))
  return tuple(rows)


__all__ = [
    "GeneratedStepAttemptV1", "GeneratedStepV1", "FrozenStepLLMRuntimeV1",
    "SCHEMA_VERSION",
    "STEP_LLM_ADAPTER_REPO", "STEP_LLM_ADAPTER_REVISION",
    "STEP_LLM_BASE_REPO", "STEP_LLM_BASE_REVISION", "build_no_rag_prompt_v1",
    "extract_step_file_v1", "generate_step_batch_v1", "generate_step_v1",
    "inspect_step_candidate_v1", "load_frozen_step_llm_v1",
]
