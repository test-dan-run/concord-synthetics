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
    """Phase C figures for the M5 exit check, over the meetings stage 5 decided (pass or fail).

    Stage 4 counts a re-render after a reference swap as an attempt too, so an utterance's re-syntheses for failed
    checks are its attempts, less one, less the swaps of its speaker.
    """
    import json

    import numpy as np

    from ..artifacts.paths import RunPaths

    store = StatusStore(run_dir)
    rp = RunPaths(run_dir)
    ids = rp.meeting_ids()
    states: dict[str, int] = {}
    failed: dict[str, str] = {}
    utts = first = resynth = limited = kept = 0
    swaps = {"clip": 0, "speaker": 0}
    swapped = {"clip": 0, "speaker": 0}  # meetings with at least one swap of that kind
    sv: dict[str, list[float]] = {}
    mos: list[float] = []
    for mid in ids:
        st = store.get(mid).get(StageId.VOICE_VAL)
        states[st.state] = states.get(st.state, 0) + 1
        if st.state == "fail":
            failed[mid] = st.detail or ""
        vv = rp.meeting(mid).voice_validation
        if st.state not in ("pass", "fail") or not vv.is_file():
            continue
        data = json.loads(vv.read_text())
        per_speaker = {}
        for sw in data.get("swaps") or []:
            swaps[sw["kind"]] += 1
            per_speaker[sw["speaker_id"]] = per_speaker.get(sw["speaker_id"], 0) + 1
        for kind in swapped:
            swapped[kind] += any(sw["kind"] == kind for sw in data.get("swaps") or [])
        for u in data["utterances"]:
            utts += 1
            first += u["attempts"] == 1 and u["passed"]
            resynth += u["attempts"] - 1 - per_speaker.get(u["speaker_id"], 0) > 0
            limited += u["asr_limited"]
            for name, v in u["sv"].items():
                sv.setdefault(name, []).append(v)
            if u.get("mos") is not None:
                mos.append(u["mos"])
        kept += len(data.get("kept_failed") or [])
    ran = states.get("pass", 0) + states.get("fail", 0)

    def rate(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    return {"meetings": len(ids), "ran": ran, "stage5": states,
            "pass_rate": rate(states.get("pass", 0), ran),
            "utterances": utts, "first_pass_rate": rate(first, utts), "resynthesis_rate": rate(resynth, utts),
            "asr_limited_rate": rate(limited, utts), "kept_failed": kept,
            "clip_swaps": swaps["clip"], "speaker_swaps": swaps["speaker"],
            "clip_swap_meeting_rate": rate(swapped["clip"], ran),
            "speaker_swap_meeting_rate": rate(swapped["speaker"], ran),
            "sv_mean": {n: round(float(np.mean(v)), 4) for n, v in sorted(sv.items())},
            "mos_mean": round(float(np.mean(mos)), 3) if mos else None,
            "failed": failed}
