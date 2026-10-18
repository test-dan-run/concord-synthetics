"""Phase C (§3.5 step 3): lease the voice services, then stages 4 and 5 for each meeting whose script passed,
with re-synthesis inline (stage 5 -> stage 4 -> stage 5 ... inside one lease).

The scheduler calls it in one line after the script rounds:

    if until >= StageId.VOICE and want(Phase.C):
        run_phase_c(self, ids, report)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..backends.registry import Backends
from ..config.schema import Phase, RunConfig, StageId
from ..stages.runner import RunOutcome, is_fresh, run_stage
from .status import StatusStore

log = logging.getLogger("synthetics.phase_c")


def stages() -> dict[StageId, object]:
    from ..stages.s4_voice import VoiceStage
    from ..stages.s5_voice_val import VoiceValidationStage

    return {StageId.VOICE: VoiceStage(), StageId.VOICE_VAL: VoiceValidationStage()}


MAX_SPEAKERS = 20  # the largest meeting (room_classes)


def max_rounds(cfg: RunConfig) -> int:
    """A guard on the stage 4 <-> 5 loop for one meeting: every round either re-synthesises (at most
    `tts_attempts` per reference) or swaps a reference, so stage 5 stops asking well before this."""
    v = cfg.voice
    return cfg.retries.tts_attempts * (1 + v.max_speaker_swaps + MAX_SPEAKERS * v.max_clip_swaps) + 1


def pending(run_dir: Path, cfg: RunConfig, ids: list[str]) -> list[str]:
    """Meetings whose script passed and whose stages 4-5 are not both fresh passes."""
    store = StatusStore(run_dir)
    s = stages()
    out = []
    for mid in ids:
        st = store.get(mid)
        if st.get(StageId.SCRIPT_VAL).state != "pass" or st.get(StageId.VOICE_VAL).state in ("dropped", "skipped"):
            continue
        if is_fresh(s[StageId.VOICE], run_dir, mid, cfg) and is_fresh(s[StageId.VOICE_VAL], run_dir, mid, cfg):
            continue
        out.append(mid)
    return out


def voice_meeting(run_dir: Path, meeting_id: str, cfg: RunConfig, backends: Backends) -> list[RunOutcome]:
    """Stage 4, then stage 5, again while stage 5 asks for a retry."""
    s = stages()
    outcomes = []
    for _ in range(max_rounds(cfg)):
        o4 = run_stage(s[StageId.VOICE], run_dir, meeting_id, cfg, backends)
        outcomes.append(o4)
        log.info("%s s4 %s %s", meeting_id, o4.state, o4.detail)
        if o4.state not in ("pass", "cached"):
            return outcomes
        o5 = run_stage(s[StageId.VOICE_VAL], run_dir, meeting_id, cfg, backends)
        outcomes.append(o5)
        log.info("%s s5 %s %s", meeting_id, o5.state, o5.detail)
        if o5.state != "retry":
            return outcomes
    StatusStore(run_dir).mark(meeting_id, StageId.VOICE_VAL, "fail", detail=f"gave up after {max_rounds(cfg)} rounds")
    outcomes.append(RunOutcome(meeting_id, StageId.VOICE_VAL, "fail", "round limit"))
    return outcomes


def run_phase_c(sched, ids: list[str], report) -> None:
    """`sched` is the PhaseScheduler: its lease, phase roles, backends and preflight are reused."""
    todo = pending(sched.run_dir, sched.cfg, ids)
    if not todo:
        return
    sched.preflight([Phase.C])
    t0 = time.monotonic()
    log.info("phase C: stages 4-5 on %d meetings (services: %s)", len(todo), ", ".join(sched.services(Phase.C)))
    with sched.lease(Phase.C):
        backends = sched.backends.with_available(sched.phase_roles(Phase.C))
        for mid in todo:
            for o in voice_meeting(sched.run_dir, mid, sched.cfg, backends):
                report.add(o)
    report.phase_seconds["C"] = report.phase_seconds.get("C", 0.0) + time.monotonic() - t0


def summarise_c(run_dir: Path) -> dict:
    """Phase C figures for the M5 exit check: pass, re-synthesis and ASR-limited rates."""
    import json

    from ..artifacts.paths import RunPaths

    store = StatusStore(run_dir)
    rp = RunPaths(run_dir)
    ids = rp.meeting_ids()
    states: dict[str, int] = {}
    utts = resynth = limited = swaps = 0
    for mid in ids:
        st = store.get(mid).get(StageId.VOICE_VAL).state
        states[st] = states.get(st, 0) + 1
        vv = rp.meeting(mid).voice_validation
        if st == "pass" and vv.is_file():
            data = json.loads(vv.read_text())
            utts += len(data["utterances"])
            resynth += sum(u["attempts"] > 1 for u in data["utterances"])
            limited += sum(u["asr_limited"] for u in data["utterances"])
            swaps += len(data.get("swaps") or [])
    ran = sum(v for k, v in states.items() if k in ("pass", "fail"))
    return {"meetings": len(ids), "stage5": states,
            "pass_rate": round(states.get("pass", 0) / ran, 4) if ran else None,
            "utterances": utts, "resynthesis_rate": round(resynth / utts, 4) if utts else None,
            "asr_limited_rate": round(limited / utts, 4) if utts else None, "reference_swaps": swaps}
