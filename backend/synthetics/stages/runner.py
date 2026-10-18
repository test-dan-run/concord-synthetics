"""Runs one stage on one meeting: toggle, cache key, seeded RNG, status record, retry hand-off.

Cache key = sha256(stage id, version, config slice, input artifact hashes, pending feedback). A stage is
skipped when status.json already holds a `pass` with the same key and its outputs exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..artifacts.io import tree_sha256
from ..artifacts.paths import MeetingPaths
from ..backends.registry import Backends
from ..config.loader import canonical_json
from ..config.schema import RunConfig, StageId
from ..orchestrator.status import StatusStore
from .base import Stage, StageContext, StageResult

log = logging.getLogger("synthetics")


def meeting_id_for(index: int) -> str:
    return f"m-{index:06d}"


def meeting_seed(run_seed: int, meeting_id: str) -> int:
    index = int(meeting_id.rsplit("-", 1)[-1])
    return int(np.random.SeedSequence([run_seed, index]).generate_state(1, dtype=np.uint32)[0])


def stage_seed_sequence(run_seed: int, m_seed: int, stage: int, attempt: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([run_seed, m_seed, int(stage), attempt])


def input_hashes(stage: Stage, paths: MeetingPaths) -> dict[str, str]:
    out = {}
    for key in stage.reads:
        p = paths.artifact(key)
        out[key] = tree_sha256(p) if p.exists() else "missing"
    return out


def cache_key(stage: Stage, cfg: RunConfig, paths: MeetingPaths, feedback: str | None) -> str:
    body = {"stage": int(stage.id), "version": stage.version, "config": json.loads(canonical_json(stage.config_slice(cfg))),
            "inputs": input_hashes(stage, paths), "feedback": feedback}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()[:16]


def outputs_exist(stage: Stage, paths: MeetingPaths) -> bool:
    return all(paths.artifact(k).exists() for k in stage.writes)


def is_fresh(stage: Stage, run_dir: Path, meeting_id: str, cfg: RunConfig) -> bool:
    """A pass whose cache key still matches and whose outputs exist: running it again would be a cache hit."""
    paths = MeetingPaths(Path(run_dir), meeting_id)
    st = StatusStore(run_dir).get(meeting_id).get(stage.id)
    if st.state != "pass" or any(not paths.artifact(k).exists() for k in stage.reads):
        return False
    return st.cache_key == cache_key(stage, cfg, paths, st.feedback) and outputs_exist(stage, paths)


@dataclass
class RunOutcome:
    meeting_id: str
    stage: StageId
    state: str  # pass | fail | retry | skipped | cached | blocked
    detail: str = ""


def run_stage(stage: Stage, run_dir: Path, meeting_id: str, cfg: RunConfig, backends: Backends, *,
              force: bool = False) -> RunOutcome:
    store = StatusStore(run_dir)
    paths = MeetingPaths(Path(run_dir), meeting_id)
    status = store.get(meeting_id).get(stage.id)
    if not cfg.enabled(stage.id):
        if status.state != "skipped":
            store.mark(meeting_id, stage.id, "skipped", detail="disabled in config")
        return RunOutcome(meeting_id, stage.id, "skipped", "disabled in config")
    if status.state == "dropped" and not force:
        return RunOutcome(meeting_id, stage.id, "blocked", "dropped")
    missing = [k for k in stage.reads if not paths.artifact(k).exists()]
    if missing:
        return RunOutcome(meeting_id, stage.id, "blocked", f"missing inputs: {', '.join(missing)}")

    feedback = status.feedback
    key = cache_key(stage, cfg, paths, feedback)
    if not force and status.state == "pass" and status.cache_key == key and outputs_exist(stage, paths):
        return RunOutcome(meeting_id, stage.id, "cached", status.detail)

    started = store.start(meeting_id, stage.id)
    m_seed = meeting_seed(cfg.seed, meeting_id)
    seq = stage_seed_sequence(cfg.seed, m_seed, stage.id, started.attempts)
    ctx = StageContext(paths=paths, cfg=cfg, rng=np.random.default_rng(seq),
                       backends=backends.for_roles(stage.roles), feedback=feedback,
                       log=logging.getLogger(f"synthetics.s{int(stage.id)}"), attempt=started.attempts,
                       seed=int(seq.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF))
    try:
        result = stage.run(ctx)
    except Exception as exc:  # noqa: BLE001 - recorded as a stage failure
        log.error("%s stage %d failed:\n%s", meeting_id, int(stage.id), traceback.format_exc())
        result = StageResult(state="fail", detail=f"error: {type(exc).__name__}: {exc}"[:300])
    if result.state == "pass" and not outputs_exist(stage, paths):
        result = StageResult(state="fail", detail="declared outputs missing after run")
    state = "pending" if result.state == "defer" else result.state
    store.mark(meeting_id, stage.id, state, cache_key=key, detail=result.detail, metrics=result.metrics)
    if result.state == "pass" and feedback is not None:
        store.update(meeting_id, stage.id, feedback=None, cache_key=cache_key(stage, cfg, paths, None))
    if result.state == "retry" and result.retry_stage is not None:
        store.update(meeting_id, result.retry_stage, state="retry", feedback=result.feedback or "")
    return RunOutcome(meeting_id, stage.id, result.state, result.detail)
