"""Stable, current-working-directory independent project path resolution.

New code should resolve user-facing paths through this module instead of
constructing ``Path("neurocad/...")`` values directly.  Explicit CLI values
take precedence over environment variables; project defaults are used only
when neither is present.  Relative values are always anchored at the project
root, including legacy values that begin with ``neurocad/``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ARCHIVE_ROOT = PROJECT_ROOT / "cad" / "abc_step_archives"
DEFAULT_DATASET_ROOT = DEFAULT_DATASET_ARCHIVE_ROOT / "a1.0.0_00"
# The ABC dataset is unrelated to Autodesk's Fusion Assembly ``a1`` volumes.
# Keep its downloads in a visibly distinct tree so a generic archive helper
# can never overwrite or mix with the official Fusion data.
DEFAULT_ABC_ARCHIVE_ROOT = (PROJECT_ROOT / "cad" / "abc_dataset_archives").resolve()
DEFAULT_ABC_EXTRACT_ROOT = (PROJECT_ROOT / "cad" / "abc_dataset_extracted").resolve()
DEFAULT_ABC_STEP_POOL_ROOT = (PROJECT_ROOT / "cad" / "abc_dataset_step_pool").resolve()
DEFAULT_ABC_NAMESPACE_ROOT = (PROJECT_ROOT / "cad").resolve()


def _nonempty(value: object) -> Optional[str]:
  if value is None:
    return None
  text = str(value).strip().strip('"').strip("'")
  return text or None


def _strip_legacy_project_prefix(path: Path) -> Path:
  """Convert legacy ``neurocad/...`` paths to project-relative paths."""
  parts = list(path.parts)
  if parts and parts[0] in {".", ""}:
    parts = parts[1:]
  if parts and parts[0].lower() == PROJECT_ROOT.name.lower():
    parts = parts[1:]
  return Path(*parts) if parts else Path()


def resolve_path(
    value: str | os.PathLike[str] | None = None,
    *,
    env_var: str | None = None,
    default: str | os.PathLike[str] | None = None,
    must_exist: bool = False,
) -> Path:
  """Resolve a CLI/config path without depending on the process CWD.

  Resolution order is explicit ``value`` (absolute or project-relative), the
  named environment variable, then ``default``.  A legacy relative path such
  as ``neurocad/cad/...`` is interpreted relative to :data:`PROJECT_ROOT`, not
  relative to the caller's current working directory.
  """
  raw = _nonempty(value)
  if raw is None and env_var:
    raw = _nonempty(os.getenv(env_var))
  if raw is None:
    raw = _nonempty(default)
  if raw is None:
    raise ValueError("No path value, environment value, or default was provided.")

  candidate = Path(raw).expanduser()
  if not candidate.is_absolute():
    candidate = PROJECT_ROOT / _strip_legacy_project_prefix(candidate)
  resolved = candidate.resolve(strict=False)
  if must_exist and not resolved.exists():
    source = f" ({env_var})" if env_var else ""
    raise FileNotFoundError(f"Required path{source} does not exist: {resolved}")
  return resolved


def absolute_project_path_no_follow(
    value: str | os.PathLike[str],
) -> Path:
  """Return an absolute project-anchored path without resolving link identity.

  This is for security gates that must inspect the lexical path supplied by a
  caller before any symlink, junction, or other reparse point is followed.
  """

  raw = _nonempty(value)
  if raw is None:
    raise ValueError("A non-empty path value is required.")
  candidate = Path(raw).expanduser()
  if not candidate.is_absolute():
    candidate = PROJECT_ROOT / _strip_legacy_project_prefix(candidate)
  return Path(os.path.abspath(candidate))


def _split_path_list(values: Iterable[object]) -> list[str]:
  items: list[str] = []
  for value in values:
    text = _nonempty(value)
    if text is None:
      continue
    # Semicolon is portable for this project and does not split Windows drive
    # letters.  Newlines are accepted for CI/environment readability.
    items.extend(
        piece.strip()
        for piece in re.split(r"[;\n\r]+", text)
        if piece.strip()
    )
  return items


def resolve_dataset_roots(
    dataset_root: str | os.PathLike[str] | None = None,
    dataset_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> list[Path]:
  """Resolve and de-duplicate Fusion Assembly archive roots.

  Explicit CLI roots win.  When none are supplied,
  ``NEUROCAD_DATASET_ROOTS`` (semicolon/newline separated) is tried before
  ``NEUROCAD_DATASET_ROOT`` and the project-local ``a1.0.0_00`` default.
  """
  explicit = _split_path_list(
      ([dataset_root] if _nonempty(dataset_root) is not None else [])
      + list(dataset_roots or [])
  )
  if explicit:
    raw_roots = explicit
  else:
    raw_roots = _split_path_list([os.getenv("NEUROCAD_DATASET_ROOTS")])
    if not raw_roots:
      raw_roots = _split_path_list([os.getenv("NEUROCAD_DATASET_ROOT")])
    if not raw_roots:
      raw_roots = [str(DEFAULT_DATASET_ROOT)]

  resolved: list[Path] = []
  seen: set[str] = set()
  for item in raw_roots:
    path = resolve_path(item)
    key = os.path.normcase(str(path))
    if key in seen:
      continue
    seen.add(key)
    resolved.append(path)
  return resolved


def resolve_fusion_protected_roots(
    explicit_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> list[Path]:
  """Return every Fusion root that ABC tooling must never overlap.

  Unlike :func:`resolve_dataset_roots`, this is intentionally additive: an
  explicit CLI/config root must not hide a default or environment-provided
  Fusion location from destructive-operation checks.
  """
  raw_roots: list[object] = [DEFAULT_DATASET_ARCHIVE_ROOT, DEFAULT_DATASET_ROOT]
  raw_roots.extend(
      _split_path_list(
          [
              os.getenv("NEUROCAD_DATASET_ARCHIVE_ROOT"),
              os.getenv("NEUROCAD_DATASET_ROOTS"),
              os.getenv("NEUROCAD_DATASET_ROOT"),
          ]
      )
  )
  raw_roots.extend(list(explicit_roots or []))

  resolved: list[Path] = []
  seen: set[str] = set()
  for raw in raw_roots:
    if _nonempty(raw) is None:
      continue
    path = resolve_path(raw)
    key = os.path.normcase(str(path))
    if key in seen:
      continue
    seen.add(key)
    resolved.append(path)
  return resolved


def project_relative(path: str | os.PathLike[str]) -> str:
  """Return a stable project-relative display path when possible."""
  resolved = resolve_path(path)
  try:
    return str(resolved.relative_to(PROJECT_ROOT))
  except ValueError:
    return str(resolved)


__all__ = [
    "DEFAULT_ABC_ARCHIVE_ROOT",
    "DEFAULT_ABC_EXTRACT_ROOT",
    "DEFAULT_ABC_NAMESPACE_ROOT",
    "DEFAULT_ABC_STEP_POOL_ROOT",
    "DEFAULT_DATASET_ARCHIVE_ROOT",
    "DEFAULT_DATASET_ROOT",
    "PROJECT_ROOT",
    "project_relative",
    "resolve_dataset_roots",
    "resolve_fusion_protected_roots",
    "resolve_path",
]
