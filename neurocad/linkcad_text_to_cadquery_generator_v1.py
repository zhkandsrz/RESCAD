"""Pinned Text-to-CadQuery inference and restricted CadQuery execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import cadquery as cq
from OCP.BRepCheck import BRepCheck_Analyzer


SCHEMA_VERSION = "linkcad_text_to_cadquery_generator.v1"
MODEL_REPO = "ricemonster/qwen2.5-3B-SFT"
MODEL_REVISION = "aaa2cd16993bc67067b4c198a910d99cf2b7affe"
OFFICIAL_REPOSITORY_REVISION = "4f7f50176f3c642f6d897c4a8670f7a83acb31bd"


@dataclass(frozen=True, slots=True)
class SanitizedCadQueryProgramV1:
  source: str
  result_name: str


@dataclass(frozen=True, slots=True)
class TextToCadQueryRuntimeV1:
  model: Any
  tokenizer: Any
  model_path: Path


@dataclass(frozen=True, slots=True)
class GeneratedCadQueryV1:
  source: str
  generated_token_count: int
  wall_time_seconds: float


_ALLOWED_BUILTINS = {
    "abs": abs, "float": float, "int": int, "len": len, "max": max,
    "min": min, "range": range, "round": round, "sum": sum,
}
_FORBIDDEN_NAMES = {
    "__import__", "builtins", "compile", "delattr", "eval", "exec",
    "getattr", "globals", "input", "locals", "open", "os", "pathlib",
    "setattr", "subprocess", "sys", "vars",
}
_ALLOWED_IMPORTS = {
    ("cadquery", None), ("math", None),
    ("cadquery", "exporters"), ("cadquery.vis", "show"),
}
_ALLOWED_CQ_ROOT_ATTRIBUTES = {
    "Assembly", "Color", "Compound", "Location", "Matrix", "Plane",
    "Shape", "Solid", "Vector", "Wire", "Workplane", "selectors",
}
_ALLOWED_NODES = {
    ast.Module, ast.Assign, ast.AnnAssign, ast.Expr, ast.Call, ast.Attribute,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.Tuple, ast.List, ast.Dict, ast.keyword,
    ast.Subscript, ast.Slice, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or, ast.IfExp,
}


def _strip_markdown_fence(source: str) -> str:
  text = source.split("<|endoftext|>", 1)[0].strip()
  if "```" not in text:
    return text
  blocks = text.split("```")
  if len(blocks) < 3:
    raise ValueError("Text-to-CadQuery response fence is incomplete")
  code = blocks[1]
  first, separator, remainder = code.partition("\n")
  if separator and first.strip().lower() in {"python", "py"}:
    code = remainder
  return code.strip()


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
  values = []
  current = node
  while isinstance(current, ast.Attribute):
    values.append(current.attr)
    current = current.value
  if isinstance(current, ast.Name):
    values.append(current.id)
  return tuple(reversed(values))


def _export_result_name(statement: ast.stmt) -> str | None:
  if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
    return None
  call = statement.value
  chain = _attribute_chain(call.func)
  if chain not in {("cq", "exporters", "export"), ("exporters", "export")}:
    return None
  if not call.args or not isinstance(call.args[0], ast.Name):
    raise ValueError("Text-to-CadQuery export target must be a named result")
  return call.args[0].id


def _is_visualization_call(statement: ast.stmt) -> bool:
  return (
      isinstance(statement, ast.Expr)
      and isinstance(statement.value, ast.Call)
      and isinstance(statement.value.func, ast.Name)
      and statement.value.func.id == "show"
  )


def _validate_import(statement: ast.stmt) -> bool:
  if isinstance(statement, ast.Import):
    if any((alias.name, None) not in _ALLOWED_IMPORTS for alias in statement.names):
      raise ValueError("Text-to-CadQuery import is not allowed")
    return True
  if isinstance(statement, ast.ImportFrom):
    module = statement.module or ""
    if any((module, alias.name) not in _ALLOWED_IMPORTS for alias in statement.names):
      raise ValueError("Text-to-CadQuery from-import is not allowed")
    return True
  return False


def sanitize_cadquery_program_v1(source: str) -> SanitizedCadQueryProgramV1:
  """Remove trusted boilerplate and reject non-geometric Python behavior."""

  try:
    tree = ast.parse(_strip_markdown_fence(source), mode="exec")
  except SyntaxError as error:
    raise ValueError(f"Text-to-CadQuery syntax differs: {error}") from error
  body = []
  result_name = None
  for statement in tree.body:
    if _validate_import(statement):
      continue
    exported = _export_result_name(statement)
    if exported is not None:
      if result_name is None:
        result_name = exported
      elif result_name != exported:
        raise ValueError("Text-to-CadQuery export targets differ")
      continue
    if _is_visualization_call(statement):
      continue
    body.append(statement)
  tree.body = body
  if result_name is None:
    raise ValueError("Text-to-CadQuery export statement is absent")
  for node in ast.walk(tree):
    if type(node) not in _ALLOWED_NODES:
      raise ValueError(
          f"Text-to-CadQuery syntax node is not allowed: {type(node).__name__}"
      )
    if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
      raise ValueError("Text-to-CadQuery forbidden name is present")
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
      raise ValueError("Text-to-CadQuery private attribute is not allowed")
    if isinstance(node, ast.Attribute):
      chain = _attribute_chain(node)
      if (
          len(chain) >= 2 and chain[0] == "cq"
          and chain[1] not in _ALLOWED_CQ_ROOT_ATTRIBUTES
      ):
        raise ValueError("Text-to-CadQuery CadQuery capability is not allowed")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      if node.func.id not in _ALLOWED_BUILTINS:
        raise ValueError("Text-to-CadQuery direct call is not allowed")
  ast.fix_missing_locations(tree)
  return SanitizedCadQueryProgramV1(
      source=ast.unparse(tree).strip() + "\n", result_name=result_name,
  )


def execute_sanitized_cadquery_v1(
    source: str, *, output_path: Path,
) -> dict[str, Any]:
  """Execute a restricted program and export its named result as STEP."""

  program = sanitize_cadquery_program_v1(source)
  globals_dict = {
      "__builtins__": _ALLOWED_BUILTINS, "cq": cq, "math": math,
  }
  locals_dict: dict[str, Any] = {}
  exec(compile(program.source, "<text-to-cadquery>", "exec"), globals_dict, locals_dict)
  if program.result_name not in locals_dict:
    raise ValueError("Text-to-CadQuery named result is absent after execution")
  result = locals_dict[program.result_name]
  if not isinstance(result, (cq.Workplane, cq.Shape)):
    raise ValueError("Text-to-CadQuery result is not CAD geometry")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  cq.exporters.export(result, str(output_path), exportType="STEP")
  imported = cq.importers.importStep(str(output_path))
  solids = imported.solids().vals()
  if len(solids) != 1 or solids[0].Volume() <= 0.0:
    raise ValueError("Text-to-CadQuery result requires one positive solid")
  if not BRepCheck_Analyzer(solids[0].wrapped).IsValid():
    raise ValueError("Text-to-CadQuery result B-rep is invalid")
  return {
      "schema_version": "linkcad_sanitized_cadquery_execution.v1",
      "result_name": program.result_name,
      "sanitized_source_sha256": hashlib.sha256(
          program.source.encode("utf-8")
      ).hexdigest(),
      "step_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
      "solid_count": 1,
      "volume": float(solids[0].Volume()),
      "area": float(solids[0].Area()),
  }


def load_text_to_cadquery_v1(model_path: Path) -> TextToCadQueryRuntimeV1:
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model_path = model_path.resolve()
  if not model_path.is_dir():
    raise FileNotFoundError("Text-to-CadQuery local model directory is absent")
  tokenizer = AutoTokenizer.from_pretrained(
      model_path, local_files_only=True, trust_remote_code=False,
      use_fast=False, model_max_length=1024,
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  model = AutoModelForCausalLM.from_pretrained(
      model_path, local_files_only=True, trust_remote_code=False,
      dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
      attn_implementation="sdpa",
  )
  model.eval()
  return TextToCadQueryRuntimeV1(
      model=model, tokenizer=tokenizer, model_path=model_path,
  )


def generate_cadquery_v1(
    runtime: TextToCadQueryRuntimeV1, *, instruction: str,
) -> GeneratedCadQueryV1:
  import torch

  prompt = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"
  inputs = runtime.tokenizer(
      prompt, return_tensors="pt", truncation=True,
  ).to(runtime.model.device)
  input_tokens = int(inputs["input_ids"].shape[1])
  maximum = max(1, 1024 - input_tokens)
  started = time.perf_counter()
  with torch.inference_mode():
    output = runtime.model.generate(
        **inputs, max_new_tokens=maximum,
        eos_token_id=runtime.tokenizer.eos_token_id,
        pad_token_id=runtime.tokenizer.eos_token_id,
        do_sample=False, use_cache=True,
        stop_strings=["<|endoftext|>"], tokenizer=runtime.tokenizer,
    )
  elapsed = time.perf_counter() - started
  continuation = output[0, input_tokens:]
  decoded = runtime.tokenizer.decode(
      continuation, skip_special_tokens=True,
  ).strip()
  return GeneratedCadQueryV1(
      source=decoded,
      generated_token_count=int(continuation.shape[0]),
      wall_time_seconds=elapsed,
  )


__all__ = [
    "GeneratedCadQueryV1", "MODEL_REPO", "MODEL_REVISION",
    "OFFICIAL_REPOSITORY_REVISION", "SCHEMA_VERSION",
    "SanitizedCadQueryProgramV1", "TextToCadQueryRuntimeV1",
    "execute_sanitized_cadquery_v1", "generate_cadquery_v1",
    "load_text_to_cadquery_v1", "sanitize_cadquery_program_v1",
]
