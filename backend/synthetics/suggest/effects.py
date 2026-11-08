"""Turn chosen options into builder settings: each effect edits the current value at its path, clamped to bounds.

Effects work on the draft's full config (defaults plus the draft's own changes), so "a bit noisier" moves the SNR
the user already has, not the default. The output is dotted overrides in the run builder's own form, which then go
through the builder's validation and preview like any typed change.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .questions import Effect, QuestionSet

CLASSES = {"both": ("small", "large"), "small": ("small",), "large": ("large",)}


@dataclass(frozen=True)
class Change:
    path: str  # dotted override key
    before: Any
    after: Any
    overlay: bool  # outside the builder form (written to the YAML overlay)


def get_path(data: dict, path: str) -> Any:
    x: Any = data
    for k in path.split("."):
        x = x[k]
    return x


def _set_path(data: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    x = data
    for k in keys[:-1]:
        x = x[k]
    x[keys[-1]] = value


def expand(path: str, scope: str) -> list[str]:
    if "{class}" not in path:
        return [path]
    return [path.replace("{class}", c) for c in CLASSES[scope]]


def _bounds(qs: QuestionSet, path: str, leaf: str) -> tuple[float, float] | None:
    if leaf == "p":
        return qs.bounds.get("*p", (0.0, 1.0))
    return qs.bounds.get(path)


def _clamp(v: float, b: tuple[float, float] | None) -> float:
    return v if b is None else min(max(v, b[0]), b[1])


def _range(cur: dict, lo: float, hi: float, b: tuple[float, float] | None) -> dict:
    lo, hi = _clamp(lo, b), _clamp(hi, b)
    if lo > hi:
        lo, hi = hi, lo
    if cur.get("dist") == "int_uniform":
        lo, hi = round(lo), round(hi)
    else:
        lo, hi = round(lo, 4), round(hi, 4)
    return {**cur, "lo": lo, "hi": hi}


def apply_effect(cfg: dict, e: Effect, path: str, qs: QuestionSet) -> dict[str, Any]:
    """New values at dotted paths, from one effect on one (class-expanded) path. `cfg` is updated in place, so
    later effects of the same request see earlier ones."""
    cur = get_path(cfg, path)
    out: dict[str, Any] = {}
    if e.op in ("shift", "scale", "range"):
        b = _bounds(qs, path, "range")
        if isinstance(cur, dict) and "lo" in cur:
            if e.op == "shift":
                new = _range(cur, cur["lo"] + e.by, cur["hi"] + e.by, b)
            elif e.op == "scale":
                new = _range(cur, cur["lo"] * e.by, cur["hi"] * e.by, b)
            else:
                new = _range(cur, e.lo, e.hi, b)
            out = {f"{path}.lo": new["lo"], f"{path}.hi": new["hi"]}
        elif isinstance(cur, (int, float)):
            v = cur + e.by if e.op == "shift" else cur * e.by if e.op == "scale" else e.lo
            out = {path: round(_clamp(v, b), 4)}
        else:
            raise ValueError(f"{e.op} needs a range or a number at {path}, found {cur!r}")
    elif e.op in ("p", "scale_p"):
        if not (isinstance(cur, dict) and cur.get("dist") == "bernoulli"):
            raise ValueError(f"{e.op} needs a Bernoulli at {path}, found {cur!r}")
        v = e.value if e.op == "p" else cur["p"] * e.by
        out = {f"{path}.p": round(_clamp(float(v), _bounds(qs, path, "p")), 4)}
    elif e.op == "weights":
        if not (isinstance(cur, dict) and "weights" in cur):
            raise ValueError(f"weights needs a categorical at {path}, found {cur!r}")
        out = {f"{path}.weights": dict(e.value)}
    else:  # set
        out = {path: copy.deepcopy(e.value)}
    for k, v in out.items():
        _set_path(cfg, k, v)
    return out


def apply_option(cfg: dict, effects: list[Effect], scope: str, qs: QuestionSet, *, overlay: bool) -> list[Change]:
    changes = []
    for e in effects:
        for path in expand(e.path, scope):
            before = copy.deepcopy(cfg)
            for k, v in apply_effect(cfg, e, path, qs).items():
                old = get_path(before, k) if _has(before, k) else None
                if old != v:
                    changes.append(Change(k, old, v, overlay))
    return changes


def _has(data: dict, path: str) -> bool:
    try:
        get_path(data, path)
        return True
    except (KeyError, TypeError):
        return False
