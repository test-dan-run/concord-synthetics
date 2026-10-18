"""Load the run config: default YAMLs, then user YAMLs, then overrides, deep-merged in that order."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .schema import RunConfig

DEFAULTS_DIR = Path(__file__).with_name("defaults")
DEFAULT_FILES = ("room_classes.yaml", "seating.yaml", "acoustics.yaml", "turn_taking.yaml", "personas.yaml",
                 "stages.yaml", "backends.yaml", "phases.yaml")


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and "dist" not in value:
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value  # a distribution replaces the old one whole, so its fields never mix
    return out


def set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    """`set_dotted(d, "acoustics.hybrid.large.p", 0.5)`; numeric path parts stay strings (YAML keys)."""
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def default_data() -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in DEFAULT_FILES:
        data = deep_merge(data, read_yaml(DEFAULTS_DIR / name))
    return data


def load_run_config(*yaml_paths: Path, overrides: dict[str, Any] | None = None, defaults: bool = True) -> RunConfig:
    """Defaults < each YAML in order < overrides. Override keys may be dotted (`"turn_taking.zipf.exponent"`)."""
    data = default_data() if defaults else {}
    for path in yaml_paths:
        data = deep_merge(data, read_yaml(path))
    for key, value in (overrides or {}).items():
        if "." in key:
            set_dotted(data, key, value)
        else:
            data = deep_merge(data, {key: value})
    return RunConfig.model_validate(data)


def canonical_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(obj: Any) -> str:
    """Canonical JSON -> sha256[:12]. The UI shows it in the run summary line."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:12]


def dump_run_config(cfg: RunConfig, path: Path) -> None:
    from ..artifacts.io import atomic_write_text

    atomic_write_text(path, yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


def read_run_config(path: Path) -> RunConfig:
    """A frozen config.yaml is complete, so defaults are not merged in again."""
    return load_run_config(path, defaults=False)
