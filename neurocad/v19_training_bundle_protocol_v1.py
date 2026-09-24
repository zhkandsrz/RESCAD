"""Pure byte/path protocol shared by V19 bundle publishing and loading."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


BUNDLE_SCHEMA_VERSION = "v19_authenticated_training_bundle.v1"
MANIFEST_SCHEMA_VERSION = "v19_authenticated_training_bundle_manifest.v1"
SHARD_SCHEMA_VERSION = "v19_authenticated_training_tensor_shard.v1"
TASK_SCOPE = "oracle_interface_pair_conditioned_program_prediction.v1"
FIXED_COUNTS = {
    "tensor_count": 1406,
    "graph_work_count": 1870,
    "train_sample_count": 1476,
    "dev_sample_count": 394,
    "target_count": 5106,
    "catalog_size": 19,
}
GRAPH_CACHE_ARTIFACT_SHA256 = (
    "131721588d6c3bf4df99e6f1c8483d6a92871e7b9b7d83ab5790dc91832b8609"
)
GRAPH_CACHE_RELATIVE_PATH = (
    "artifacts/development/"
    "v19_brep_graph_cache_occ_single_shard_commit7674f03_20260725_run1/cache/"
    "artifact.json"
)
FIXED_INDEX_RELATIVE_PATH = (
    "artifacts/development/"
    "v19_query_semantic_adapter_v2_full938_commit5b58150_20260725_run1/"
    "authenticated_query_graph_work_index.v2.json"
)
FIXED_INDEX_SHA256 = (
    "d12dbe5c21da200bd5c16470340bdc570807365ec135241bee439e5da3f85f2c"
)
FIXED_INDEX_BYTES = 9_554_189
IMPLEMENTATION_SOURCE_ROSTER = (
    "__init__.py",
    "v19_training_bundle_protocol_v1.py",
    "v19_authenticated_training_bundle_publisher_v1.py",
    "v19_authenticated_training_bundle_v1.py",
    "v19_brep_program_model_v1.py",
    "v19_set_valued_program_learning_v1.py",
    "v19_program_proposal_protocol_v1.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_ATTRIBUTE = 0x400
TENSOR_FIELDS = (
    "node_features",
    "edge_index",
    "edge_features",
    "center_node_indices",
)
TENSOR_DTYPES = {
    "node_features": np.dtype("<f4"),
    "edge_index": np.dtype("<i8"),
    "edge_features": np.dtype("<f4"),
    "center_node_indices": np.dtype("<i8"),
}
MAX_NPY_HEADER_BYTES = 512
MAX_NPY_MEMBER_BYTES = 512 * 1024


@dataclass(frozen=True, slots=True)
class PlainFileCaptureV1:
  path: Path
  resolved_path: Path
  raw_bytes: bytes
  sha256: str
  byte_count: int
  device: int
  inode: int
  link_count: int


def capture_plain_file(path: str | Path, *, label: str) -> PlainFileCaptureV1:
  candidate = Path(path)
  _check_absolute_chain(candidate)
  if (
      not candidate.is_file()
      or candidate.is_symlink()
      or has_reparse_attribute(candidate)
  ):
    raise ValueError(f"{label} must be a plain file")
  before = candidate.stat(follow_symlinks=False)
  if int(before.st_nlink) != 1:
    raise ValueError(f"{label} must have one physical link")
  raw = candidate.read_bytes()
  after = candidate.stat(follow_symlinks=False)
  identity = lambda value: (
      int(value.st_dev), int(value.st_ino), int(value.st_size),
      int(value.st_mtime_ns), int(value.st_ctime_ns), int(value.st_nlink),
  )
  if identity(before) != identity(after) or len(raw) != int(after.st_size):
    raise ValueError(f"{label} changed while captured")
  return PlainFileCaptureV1(
      path=candidate,
      resolved_path=candidate.resolve(strict=True),
      raw_bytes=raw,
      sha256=hashlib.sha256(raw).hexdigest(),
      byte_count=len(raw),
      device=int(after.st_dev),
      inode=int(after.st_ino),
      link_count=int(after.st_nlink),
  )


def canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json(raw: bytes, *, label: str) -> Any:
  def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
      if key in result:
        raise ValueError(f"{label} contains duplicate keys")
      result[key] = value
    return result

  try:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def require_sha256(value: Any, *, label: str) -> str:
  if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
    raise ValueError(f"{label} is not SHA-256")
  return value


def actual_int(value: Any, *, label: str, minimum: int = 0) -> int:
  if type(value) is not int or value < minimum:
    raise ValueError(f"{label} is not an actual integer")
  return value


def has_reparse_attribute(path: Path) -> bool:
  attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
  return bool(int(attributes) & _REPARSE_ATTRIBUTE)


def _check_absolute_chain(path: Path, *, allow_missing_leaf: bool = False) -> None:
  absolute = path.absolute()
  anchor = Path(absolute.anchor)
  current = anchor
  parts = absolute.parts[1:] if absolute.anchor else absolute.parts
  for ordinal, part in enumerate(parts):
    current = current / part
    final = ordinal == len(parts) - 1
    if not current.exists():
      if allow_missing_leaf and final:
        return
      raise ValueError("plain path chain contains a missing component")
    if current.is_symlink() or has_reparse_attribute(current):
      raise ValueError("plain path chain contains a redirected component")


def plain_directory(path: str | Path, *, label: str) -> Path:
  candidate = Path(path)
  _check_absolute_chain(candidate)
  if (
      not candidate.is_dir()
      or candidate.is_symlink()
      or has_reparse_attribute(candidate)
  ):
    raise ValueError(f"{label} must be a plain directory")
  resolved = candidate.resolve(strict=True)
  if resolved != candidate.absolute().resolve(strict=True):
    raise ValueError(f"{label} resolution differs")
  return resolved


def plain_child(
    root: Path,
    name: Any,
    *,
    label: str,
    kind: str,
) -> Path:
  if (
      not isinstance(name, str)
      or not name
      or Path(name).name != name
      or "/" in name
      or "\\" in name
  ):
    raise ValueError(f"{label} is not a basename")
  root = plain_directory(root, label=f"{label} root")
  candidate = root / name
  if candidate.is_symlink() or (
      candidate.exists() and has_reparse_attribute(candidate)
  ):
    raise ValueError(f"{label} is redirected")
  if kind == "file" and not candidate.is_file():
    raise ValueError(f"{label} is not a file")
  if kind == "directory" and not candidate.is_dir():
    raise ValueError(f"{label} is not a directory")
  if kind == "absent" and os.path.lexists(candidate):
    raise FileExistsError(f"{label} already exists")
  if kind not in {"file", "directory", "absent"}:
    raise ValueError("plain child kind differs")
  if candidate.exists() and candidate.resolve(strict=True).parent != root:
    raise ValueError(f"{label} escapes its root")
  return candidate


def atomic_write_exclusive(path: Path, raw: bytes) -> None:
  path = Path(path)
  parent = plain_directory(path.parent, label="atomic publication parent")
  if Path(path.name).name != path.name or not path.name:
    raise ValueError("atomic publication destination is not a basename")
  destination = parent / path.name
  temporary = parent / (
      f".{path.name}.tmp-{os.getpid()}-{hashlib.sha256(raw).hexdigest()[:12]}"
  )
  descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
  temporary_identity: tuple[int, int] | None = None
  try:
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
      descriptor = -1
      stream.write(raw)
      stream.flush()
      os.fsync(stream.fileno())
    temporary_identity = directory_identity(temporary)
    os.link(temporary, destination)
    if directory_identity(destination) != temporary_identity:
      raise ValueError("atomic publication destination identity differs")
    temporary.unlink()
    _fsync_parent_directory(parent)
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    if os.path.lexists(temporary):
      if (
          temporary_identity is None
          or temporary.is_symlink()
          or has_reparse_attribute(temporary)
          or directory_identity(temporary) != temporary_identity
      ):
        raise ValueError("refusing to clean a changed atomic temporary")
      temporary.unlink()
      _fsync_parent_directory(parent)


def directory_identity(path: Path) -> tuple[int, int]:
  value = path.stat(follow_symlinks=False)
  return int(value.st_dev), int(value.st_ino)


def cleanup_owned_directory(
    path: Path,
    *,
    expected_parent: Path,
    identity: tuple[int, int],
) -> None:
  if not path.exists():
    return
  parent = plain_directory(expected_parent, label="staging parent")
  if (
      path.parent.resolve(strict=True) != parent
      or path.is_symlink()
      or has_reparse_attribute(path)
      or directory_identity(path) != identity
  ):
    raise ValueError("refusing to clean a changed staging directory")
  shutil.rmtree(path)


def _fsync_parent_directory(parent: Path) -> None:
  if os.name == "nt":
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    flush = kernel32.FlushFileBuffers
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    close = kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ_WRITE_DELETE = 0x00000007
    OPEN_EXISTING = 3
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        str(parent),
        GENERIC_WRITE,
        FILE_SHARE_READ_WRITE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    if handle in (None, invalid_handle):
      code = ctypes.get_last_error()
      raise OSError(code, "opening publication parent for flush failed")
    try:
      if not flush(handle):
        code = ctypes.get_last_error()
        raise OSError(code, "flushing publication parent failed")
    finally:
      close(handle)
    return
  descriptor = os.open(parent, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
  if os.name == "nt":
    # Win32 MoveFile semantics, exposed by os.rename, refuse replacement.
    os.rename(source, destination)
    return
  if not sys.platform.startswith("linux"):
    raise OSError(
        errno.ENOTSUP,
        "directory no-replace publication is unavailable on this platform",
    )
  try:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    AT_FDCWD = -100
    RENAME_NOREPLACE = 1
    result = renameat2(
        AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
      error = ctypes.get_errno()
      if error == errno.EEXIST:
        raise FileExistsError("publication destination already exists")
      raise OSError(error, "renameat2 no-clobber publication failed")
  except AttributeError as error:
    raise OSError(
        errno.ENOTSUP,
        "renameat2 no-replace publication is unavailable",
    ) from error


def publish_directory_noreplace(
    stage: Path,
    output: Path,
    *,
    parent: Path,
    stage_identity: tuple[int, int],
) -> None:
  """Reserve with a hard link, then atomically publish without replacement."""

  parent = plain_directory(parent, label="publication parent")
  if (
      stage.parent.resolve(strict=True) != parent
      or directory_identity(stage) != stage_identity
      or output.parent.resolve(strict=True) != parent
  ):
    raise ValueError("publication directory identity differs")
  claim_raw = canonical_bytes({
      "schema_version": "v19_training_bundle_publication_claim.v1",
      "output_name": output.name,
      "stage_identity": list(stage_identity),
  })
  claim_source = parent / (
      f".{output.name}.claim-source-{os.getpid()}-"
      f"{hashlib.sha256(claim_raw).hexdigest()[:12]}"
  )
  claim = parent / f".{output.name}.publish-claim"
  atomic_write_exclusive(claim_source, claim_raw)
  source_identity = directory_identity(claim_source)
  claim_linked = False
  try:
    try:
      os.link(claim_source, claim)
      claim_linked = True
    except FileExistsError as error:
      raise FileExistsError("publication claim already exists") from error
    _rename_directory_noreplace(stage, output)
    if directory_identity(output) != stage_identity:
      raise ValueError("published directory identity differs")
    _fsync_parent_directory(parent)
  finally:
    for candidate in (claim if claim_linked else None, claim_source):
      if candidate is None or not candidate.exists():
        continue
      if directory_identity(candidate) != source_identity:
        raise ValueError("publication claim identity changed")
      candidate.unlink()
    _fsync_parent_directory(parent)


def file_binding(path: Path) -> dict[str, Any]:
  raw = path.read_bytes()
  return {
      "name": path.name,
      "bytes": len(raw),
      "sha256": hashlib.sha256(raw).hexdigest(),
  }


def array_receipt(array: np.ndarray) -> dict[str, Any]:
  contiguous = np.ascontiguousarray(array)
  return {
      "dtype": contiguous.dtype.str,
      "shape": list(contiguous.shape),
      "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
  }


def freeze_array(array: np.ndarray) -> np.ndarray:
  contiguous = np.ascontiguousarray(array)
  frozen = np.frombuffer(
      contiguous.tobytes(order="C"), dtype=contiguous.dtype
  ).reshape(contiguous.shape)
  frozen.setflags(write=False)
  return frozen


def validate_graph_tensor_arrays(
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
  if set(arrays) != set(TENSOR_FIELDS):
    raise ValueError("graph tensor field roster differs")
  values = {
      name: np.asarray(arrays[name]) for name in TENSOR_FIELDS
  }
  for name, expected_dtype in TENSOR_DTYPES.items():
    if values[name].dtype != expected_dtype:
      raise ValueError(f"graph tensor {name} dtype differs")
    if values[name].flags.f_contiguous and not values[name].flags.c_contiguous:
      raise ValueError(f"graph tensor {name} is Fortran ordered")
  node = values["node_features"]
  edge_index = values["edge_index"]
  edge = values["edge_features"]
  centers = values["center_node_indices"]
  if (
      node.ndim != 2
      or node.shape[1] != 19
      or not 1 <= node.shape[0] <= 1024
      or edge_index.ndim != 2
      or edge_index.shape[1] != 2
      or edge_index.shape[0] > 4096
      or edge.shape != (edge_index.shape[0], 3)
      or centers.ndim != 1
      or not 1 <= centers.size <= node.shape[0]
      or len(set(int(value) for value in centers.tolist())) != centers.size
      or not np.isfinite(node).all()
      or not np.isfinite(edge).all()
      or int(centers.min()) < 0
      or int(centers.max()) >= node.shape[0]
      or (
          edge_index.size
          and (
              int(edge_index.min()) < 0
              or int(edge_index.max()) >= node.shape[0]
          )
      )
  ):
    raise ValueError("graph tensor geometry/budget contract differs")
  return {
      name: freeze_array(values[name]) for name in TENSOR_FIELDS
  }


def _inspect_npy_member(
    raw: bytes,
    *,
    expected_dtype: np.dtype[Any],
    expected_shape: Sequence[int],
) -> np.ndarray:
  stream = io.BytesIO(raw)
  try:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
      shape, fortran, dtype = np.lib.format.read_array_header_1_0(
          stream, max_header_size=MAX_NPY_HEADER_BYTES
      )
    elif version == (2, 0):
      shape, fortran, dtype = np.lib.format.read_array_header_2_0(
          stream, max_header_size=MAX_NPY_HEADER_BYTES
      )
    else:
      raise ValueError("NPY version differs")
  except (EOFError, ValueError) as error:
    raise ValueError("NPY header differs") from error
  header_end = stream.tell()
  dtype = np.dtype(dtype)
  expected_shape_tuple = tuple(int(value) for value in expected_shape)
  if (
      fortran
      or dtype.hasobject
      or dtype.fields is not None
      or dtype != expected_dtype
      or tuple(shape) != expected_shape_tuple
      or any(type(value) is not int or value < 0 for value in shape)
      or header_end > MAX_NPY_HEADER_BYTES
  ):
    raise ValueError("NPY dtype/shape/order differs")
  element_count = math.prod(shape)
  expected_bytes = header_end + element_count * dtype.itemsize
  if expected_bytes != len(raw) or len(raw) > MAX_NPY_MEMBER_BYTES:
    raise ValueError("NPY byte extent differs")
  array = np.frombuffer(raw, dtype=dtype, count=element_count, offset=header_end)
  return freeze_array(array.reshape(shape))


def read_validated_tensor_npz(
    raw: bytes,
    *,
    declared: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, np.ndarray]:
  grouped_fields: dict[str, set[str]] = {}
  field_by_name: dict[str, str] = {}
  for name in declared:
    matches = [
        field for field in TENSOR_FIELDS if name.endswith(f"_{field}")
    ]
    if len(matches) != 1:
      raise ValueError("tensor archive member field name differs")
    field = matches[0]
    prefix = name[: -(len(field) + 1)]
    if not prefix:
      raise ValueError("tensor archive member prefix differs")
    grouped_fields.setdefault(prefix, set()).add(field)
    field_by_name[name] = field
  if (
      not grouped_fields
      or any(fields != set(TENSOR_FIELDS) for fields in grouped_fields.values())
      or len(declared) != 4 * len(grouped_fields)
  ):
    raise ValueError("tensor archive complete field roster differs")
  expected_names = {f"{name}.npy" for name in declared}
  if len(declared) > 4 * 64:
    raise ValueError("tensor archive member count exceeds shard budget")
  try:
    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
      infos = archive.infolist()
      names = [info.filename for info in infos]
      if (
          len(names) != len(set(names))
          or set(names) != expected_names
          or any(
              info.compress_type != zipfile.ZIP_STORED
              or info.file_size > MAX_NPY_MEMBER_BYTES
              or info.compress_size != info.file_size
              for info in infos
          )
      ):
        raise ValueError("tensor ZIP roster/size/compression differs")
      result: dict[str, np.ndarray] = {}
      for name, receipt in declared.items():
        if set(receipt) != {"dtype", "shape", "sha256"}:
          raise ValueError("tensor array receipt schema differs")
        dtype = np.dtype(str(receipt["dtype"]))
        if dtype != TENSOR_DTYPES[field_by_name[name]]:
          raise ValueError("tensor array receipt dtype differs")
        shape = receipt["shape"]
        if (
            not isinstance(shape, list)
            or any(type(value) is not int or value < 0 for value in shape)
        ):
          raise ValueError("tensor array receipt shape differs")
        member = archive.read(f"{name}.npy")
        array = _inspect_npy_member(
            member, expected_dtype=dtype, expected_shape=shape
        )
        if hashlib.sha256(array.tobytes(order="C")).hexdigest() != receipt["sha256"]:
          raise ValueError("tensor array receipt hash differs")
        result[name] = array
      return result
  except (zipfile.BadZipFile, KeyError) as error:
    raise ValueError("tensor ZIP differs") from error


def deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
  if not arrays:
    raise ValueError("tensor archive is empty")
  output = io.BytesIO()
  with zipfile.ZipFile(
      output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=False
  ) as archive:
    for name in sorted(arrays):
      payload = io.BytesIO()
      np.lib.format.write_array(
          payload,
          np.ascontiguousarray(arrays[name]),
          allow_pickle=False,
      )
      info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
      info.compress_type = zipfile.ZIP_STORED
      info.external_attr = 0o600 << 16
      archive.writestr(info, payload.getvalue())
  return output.getvalue()
