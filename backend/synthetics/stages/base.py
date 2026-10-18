"""The uniform stage contract (§3.3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from ..artifacts.paths import MeetingPaths
from ..backends.registry import Backends
from ..config.schema import Phase, Role, RunConfig, StageId


@dataclass
class StageContext:
    paths: MeetingPaths
    cfg: RunConfig
    rng: np.random.Generator  # seeded from (run seed, meeting seed, stage id, attempt)
    backends: Backends  # only the roles this stage declared
    feedback: str | None  # judge or voice-check feedback from the previous round
    log: logging.Logger
    attempt: int = 1
    seed: int = 0  # integer seed for backends (same derivation as rng)


@dataclass
class StageResult:
    state: Literal["pass", "fail", "retry", "defer"]  # defer: unfinished, needs a role leased in a later phase
    detail: str  # one line, shown on the UI stage tile
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    feedback: str | None = None  # sent to the upstream stage on retry
    retry_stage: StageId | None = None  # stage 3 -> 2, stage 5 -> 4


@runtime_checkable
class Stage(Protocol):
    id: StageId
    name: str
    phase: Phase
    roles: tuple[Role, ...]
    reads: tuple[str, ...]  # artifact keys (MeetingPaths attributes, plus run-level speaker_splits)
    writes: tuple[str, ...]
    version: str  # bump when the code changes outputs; part of the cache key

    def config_slice(self, cfg: RunConfig) -> BaseModel: ...

    def run(self, ctx: StageContext) -> StageResult: ...
