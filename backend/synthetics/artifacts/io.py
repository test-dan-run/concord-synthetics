"""Atomic artifact writers (temp file in the same directory, then rename) and readers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_model(path: Path, model: BaseModel) -> None:
    """Deterministic JSON: the same model always gives the same bytes."""
    atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def read_model[M: BaseModel](path: Path, cls: type[M]) -> M:
    return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash of a file, or of every file under a directory (relative names + contents)."""
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")):
        h.update(str(f.relative_to(path)).encode())
        h.update(file_sha256(f).encode())
    return h.hexdigest()
