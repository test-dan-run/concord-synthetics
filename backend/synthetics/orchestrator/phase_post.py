"""Stages 6-8 for the phase scheduler: stages 6 and 7 for each meeting, one meeting at a time (CPU); then phase D
(the aligner lease) for stage 8; then the run's manifest.jsonl.

The scheduler calls it in one line at the end of `PhaseScheduler.run`:

    run_post(self, ids, report, until=until, only=only)

Stage 6 takes meetings whose stage 5 passed (or is switched off). Everything is sequential.
"""

from __future__ import annotations

import logging

from ..config.schema import Phase, StageId
from ..export.manifest import write_run_manifest
from ..stages.runner import is_fresh

log = logging.getLogger("synthetics.post")

READY = ("pass", "skipped")


def post_stages() -> dict[StageId, object]:
    from ..stages.s6_assemble import AssembleStage
    from ..stages.s7_acoustic import AcousticStage
    from ..stages.s8_labels import LabelsStage

    return {StageId.ASSEMBLE: AssembleStage(), StageId.ACOUSTIC: AcousticStage(), StageId.LABELS: LabelsStage()}


def run_post(sched, ids: list[str], report, *, until: StageId = StageId.LABELS, only: Phase | None = None) -> None:
    """`sched` is a PhaseScheduler (its stages, status store, backends and leases are reused)."""
    if until < StageId.ASSEMBLE:
        return
    sched.stages.update({k: v for k, v in post_stages().items() if k not in sched.stages})

    def want(p: Phase) -> bool:
        return only is None or only == p

    if want(Phase.CPU):
        from .scheduler import Stopwatch

        watch = Stopwatch(sched.run_dir, report, "CPU-post")
        voiced = [m for m in ids if sched.state(m, StageId.VOICE_VAL) in READY
                  and sched.state(m, StageId.VOICE) in READY and not sched.store.dropped(m)]
        for mid in voiced:
            sched._run(StageId.ASSEMBLE, [mid], report)
            if until >= StageId.ACOUSTIC and sched.state(mid, StageId.ASSEMBLE) in READY:
                sched._run(StageId.ACOUSTIC, [mid], report)
            watch.tick()
    if until < StageId.LABELS:
        return
    s8 = sched.stages[StageId.LABELS]
    todo = [m for m in ids if sched.state(m, StageId.ACOUSTIC) in READY
            and sched.state(m, StageId.LABELS) not in ("dropped", "skipped")
            and not is_fresh(s8, sched.run_dir, m, sched.cfg)]
    if want(Phase.D) and todo:
        sched.preflight([Phase.D])
        sched._phase(Phase.D, StageId.LABELS, todo, report)
    path = write_run_manifest(sched.run_dir)
    log.info("manifest: %s", path)

