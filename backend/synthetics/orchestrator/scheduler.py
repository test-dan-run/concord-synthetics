"""Phase scheduler (§3.5).

1. stage 1 for every meeting (CPU)
2. rounds: lease A -> stage 2 for pending meetings; lease B -> stage 3 (primary judge); lease B2 -> stage 3 for
   meetings waiting on the tie-breaker; failed scripts go back to A with feedback; after `retries.script_rounds`
   rounds an unresolved script is dropped.
3. lease C -> stages 4 and 5 with re-synthesis inline (`phase_c.py`)
4. stages 6 and 7 for each meeting (CPU), then lease D -> stage 8, then manifest.jsonl (`phase_post.py`)

Everything is sequential: one meeting at a time and one request at a time.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts.paths import RunPaths
from ..backends.registry import Backends, check_endpoints
from ..config.schema import Phase, Role, RunConfig, StageId
from ..stages.runner import RunOutcome, is_fresh, run_stage
from .lease import GpuLease, LeaseError, check_vram
from .phase_c import run_phase_c
from .phase_post import run_post
from .status import StatusStore

log = logging.getLogger("synthetics.scheduler")


def stages() -> dict[StageId, object]:
    from ..stages.s1_sample import SampleStage
    from ..stages.s2_script import ScriptStage
    from ..stages.s3_script_val import ScriptValidationStage

    return {StageId.SAMPLE: SampleStage(), StageId.SCRIPT: ScriptStage(), StageId.SCRIPT_VAL: ScriptValidationStage()}


@dataclass
class RunReport:
    outcomes: list[RunOutcome] = field(default_factory=list)
    phase_seconds: dict[str, float] = field(default_factory=dict)
    rounds: int = 0

    def add(self, o: RunOutcome) -> None:
        self.outcomes.append(o)


class Stopwatch:
    """Time spent in one phase, recorded in the report and the run's ledger at every `tick` (each meeting)."""

    def __init__(self, run_dir: Path, report: RunReport, phase: str):
        self.run_dir, self.report, self.phase = run_dir, report, phase
        self.t = time.monotonic()

    def tick(self) -> None:
        from .run_report import record_seconds

        now = time.monotonic()
        dt, self.t = now - self.t, now
        self.report.phase_seconds[self.phase] = self.report.phase_seconds.get(self.phase, 0.0) + dt
        record_seconds(self.run_dir, self.phase, dt)


class PhaseScheduler:
    def __init__(self, run_dir: Path, cfg: RunConfig, *, backends: Backends | None = None,
                 lease_factory: Callable[..., GpuLease] = GpuLease, keep_services: bool = False):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.backends = backends or Backends(cfg.backends)
        self.store = StatusStore(run_dir)
        self.lease_factory = lease_factory
        self.keep_services = keep_services
        self.stages = stages()

    # ---------------------------------------------------------------- helpers

    @property
    def live(self) -> bool:
        return self.cfg.backends.mode in ("aicore", "record")

    def phase_roles(self, phase: Phase) -> list[Role]:
        spec = self.cfg.phases.phases.get(phase)
        return list(spec.roles) if spec else []

    def services(self, phase: Phase) -> list[str]:
        return sorted({self.cfg.backends.roles[r] for r in self.phase_roles(phase) if r in self.cfg.backends.roles})

    def lease(self, phase: Phase) -> GpuLease:
        ph = self.cfg.phases
        return self.lease_factory(self.services(phase), platform_dir=ph.platform_dir, no_evict=ph.no_evict,
                                  timeout_s=ph.lease_timeout_s, enabled=self.live, keep=self.keep_services)

    def preflight(self, phases: list[Phase]) -> None:
        """Every role resolves to a platform service, and every phase's set fits under util_limit."""
        roles = {r for p in phases for r in self.phase_roles(p)}
        missing = check_endpoints(self.cfg.backends, roles)
        if missing:
            raise LeaseError("services missing from the platform endpoint map: " + "; ".join(missing))
        if not self.live:
            return
        for p in phases:
            if not self.services(p):
                continue
            ok, out = check_vram(self.services(p), platform_dir=self.cfg.phases.platform_dir)
            if not ok:
                raise LeaseError(f"phase {p.value} is over the platform VRAM budget:\n{out}")

    def state(self, meeting_id: str, stage: StageId) -> str:
        return self.store.get(meeting_id).get(stage).state

    def _run(self, stage: StageId, ids: list[str], report: RunReport, available: list[Role] | None = None,
             watch: Stopwatch | None = None) -> None:
        backends = self.backends.with_available(available)
        for mid in ids:
            out = run_stage(self.stages[stage], self.run_dir, mid, self.cfg, backends)
            report.add(out)
            log.info("%s s%d %s %s", mid, int(stage), out.state, out.detail)
            if watch is not None:
                watch.tick()

    def _phase(self, phase: Phase, stage: StageId, ids: list[str], report: RunReport) -> None:
        if not ids:
            return
        watch = Stopwatch(self.run_dir, report, phase.value)
        log.info("phase %s: stage %d on %d meetings (services: %s)", phase.value, int(stage), len(ids),
                 ", ".join(self.services(phase)) or "none")
        with self.lease(phase):
            self._run(stage, ids, report, available=self.phase_roles(phase), watch=watch)
        watch.tick()  # the lease's release

    # ---------------------------------------------------------------- the run

    def run(self, *, until: StageId = StageId.SCRIPT_VAL, only: Phase | None = None,
            meetings: list[str] | None = None) -> RunReport:
        report = RunReport()
        ids = meetings or RunPaths(self.run_dir).meeting_ids()
        def want(p: Phase) -> bool:
            return only is None or only == p

        gpu_phases = [p for p in (Phase.A, Phase.B, Phase.B2) if want(p) and p in self.cfg.phases.phases]
        if until >= StageId.SCRIPT:
            self.preflight([p for p in gpu_phases if (p == Phase.A) or until >= StageId.SCRIPT_VAL])

        if want(Phase.CPU):
            watch = Stopwatch(self.run_dir, report, "CPU")
            self._run(StageId.SAMPLE, ids, report)
            watch.tick()
        if until < StageId.SCRIPT:
            return report

        max_rounds = self.cfg.retries.script_rounds
        s2, s3 = self.stages[StageId.SCRIPT], self.stages[StageId.SCRIPT_VAL]
        for rnd in range(1, max_rounds + 2):
            report.rounds = rnd
            done1 = [m for m in ids if self.state(m, StageId.SAMPLE) == "pass"]
            todo2 = [m for m in done1 if self.state(m, StageId.SCRIPT) not in ("dropped", "skipped")
                     and not is_fresh(s2, self.run_dir, m, self.cfg) and not self._exhausted(m)]
            if want(Phase.A):
                self._phase(Phase.A, StageId.SCRIPT, todo2, report)
            if until < StageId.SCRIPT_VAL:
                break
            done2 = [m for m in done1 if self.state(m, StageId.SCRIPT) == "pass"]
            todo3 = [m for m in done2 if self.state(m, StageId.SCRIPT_VAL) not in ("dropped", "skipped")
                     and not is_fresh(s3, self.run_dir, m, self.cfg)]
            if want(Phase.B):
                self._phase(Phase.B, StageId.SCRIPT_VAL, todo3, report)
            waiting = [m for m in done2 if self.state(m, StageId.SCRIPT_VAL) == "pending"]
            if want(Phase.B2) and Phase.B2 in self.cfg.phases.phases:
                self._phase(Phase.B2, StageId.SCRIPT_VAL, waiting, report)
            retry = [m for m in ids if self.state(m, StageId.SCRIPT) == "retry"]
            for m in retry:
                if self._exhausted(m):
                    self._drop(m, max_rounds)
            if not [m for m in retry if not self._exhausted(m)]:
                break
        if until >= StageId.VOICE and want(Phase.C):
            run_phase_c(self, ids, report)
        run_post(self, ids, report, until=until, only=only)
        return report

    def _exhausted(self, meeting_id: str) -> bool:
        st = self.store.get(meeting_id).get(StageId.SCRIPT)
        return st.state == "retry" and st.attempts >= self.cfg.retries.script_rounds

    def _drop(self, meeting_id: str, rounds: int) -> None:
        detail = f"dropped after {rounds} rounds"
        self.store.mark(meeting_id, StageId.SCRIPT, "dropped", detail=detail)
        self.store.mark(meeting_id, StageId.SCRIPT_VAL, "dropped", detail=detail)
        log.info("%s dropped: script unresolved after %d rounds", meeting_id, rounds)


def summarise(run_dir: Path, until: StageId = StageId.SCRIPT_VAL) -> dict:
    """Per-stage state counts plus the stage 2/3 regeneration figures."""
    store = StatusStore(run_dir)
    ids = RunPaths(run_dir).meeting_ids()
    by_stage = {int(s): Counter(store.get(m).get(s).state for m in ids) for s in StageId if s <= until}
    passed3 = [m for m in ids if store.get(m).get(StageId.SCRIPT_VAL).state == "pass"]
    rounds = Counter(store.get(m).get(StageId.SCRIPT).attempts for m in passed3)
    regenerated = sum(n for r, n in rounds.items() if r > 1)
    return {"meetings": len(ids), "stages": {k: dict(v) for k, v in by_stage.items()},
            "scripts_passed": len(passed3), "rounds_to_pass": dict(sorted(rounds.items())),
            "regeneration_rate": round(regenerated / len(passed3), 4) if passed3 else None,
            "dropped": sum(1 for m in ids if store.get(m).get(StageId.SCRIPT).state == "dropped")}
