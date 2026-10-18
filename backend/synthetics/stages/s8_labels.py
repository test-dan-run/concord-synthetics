"""Stage 8: labels and packaging (phase D). Writes labels/meeting.rttm, labels/transcript.json and
labels/manifest_entry.json; the run's manifest.jsonl is rebuilt from the entries afterwards
(`export/manifest.py`).

Quality gates (audio and labels must agree; any failure fails the stage):
- the RTTM's total speech equals the sum of the timeline's segments
- every aligned word lies inside its segment, give or take `thresholds.align_slack_s`
- every utterance's alignment coverage is at least `thresholds.align_coverage_min`
"""

from __future__ import annotations

from pydantic import BaseModel

from ..artifacts.io import read_model, write_model
from ..artifacts.labels import Timeline, Transcript
from ..artifacts.models import Script
from ..backends.base import Audio
from ..config.schema import LabelsConfig, Phase, Role, RunConfig, StageId
from ..export.manifest import build_entry
from ..export.rttm import read_rttm, write_rttm
from ..export.transcript import utterance
from .base import StageContext, StageResult

RTTM_TOL_PER_SEGMENT = 0.0011  # millisecond rounding of start and duration


class Stage8Config(BaseModel):
    labels: LabelsConfig
    align_coverage_min: float
    align_slack_s: float
    aligner: str
    run_id: str
    stages_enabled: dict[str, bool]


class LabelsStage:
    id = StageId.LABELS
    name = "Labels"
    phase = Phase.D
    roles = (Role.ALIGNER,)
    reads = ("meeting", "script", "script_validation", "synthesis", "voice_validation", "timeline", "utterances_dir",
             "render")
    writes = ("rttm", "transcript", "manifest_entry")
    version = "1"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        return Stage8Config(labels=cfg.labels, align_coverage_min=cfg.thresholds.align_coverage_min,
                            align_slack_s=cfg.thresholds.align_slack_s,
                            aligner=cfg.backends.roles.get(Role.ALIGNER, ""), run_id=cfg.run_id,
                            stages_enabled={s.name.lower(): cfg.enabled(s) for s in StageId})

    def run(self, ctx: StageContext) -> StageResult:
        cfg, th = ctx.cfg, ctx.cfg.thresholds
        if not ctx.backends.has(Role.ALIGNER):
            return StageResult(state="defer", detail="waiting for the aligner (phase D)")
        aligner = ctx.backends.get(Role.ALIGNER)
        service = ctx.backends.service(Role.ALIGNER)
        tl = read_model(ctx.paths.timeline, Timeline)
        script = read_model(ctx.paths.script, Script)
        turns = {t.turn_id: t for t in script.turns}

        utts = []
        outside: list[str] = []
        low: list[str] = []
        for seg in tl.segments:
            turn = turns[seg.turn_id]
            timings = []
            if turn.text_norm:
                audio = Audio.load(ctx.paths.utterance(seg.turn_id))
                timings = aligner.align(audio, turn.text_norm, language=cfg.labels.align_language)
            u = utterance(seg, turn, timings)
            utts.append(u)
            if any(w.start < seg.start - th.align_slack_s or w.end > seg.end + th.align_slack_s or w.end < w.start
                   for w in u.words):
                outside.append(seg.turn_id)
            if u.align_coverage < th.align_coverage_min:
                low.append(seg.turn_id)

        write_rttm(ctx.paths.rttm, tl)
        rttm_total = sum(t.duration for t in read_rttm(ctx.paths.rttm))
        speech = sum(s.end - s.start for s in tl.segments)
        rttm_ok = abs(rttm_total - speech) <= RTTM_TOL_PER_SEGMENT * max(1, len(tl.segments))
        write_model(ctx.paths.transcript, Transcript(meeting_id=tl.meeting_id, aligner=service, utterances=utts))
        write_model(ctx.paths.manifest_entry, build_entry(ctx.paths, cfg, aligner=service))

        n_words = sum(len(u.words) for u in utts)
        covs = [u.align_coverage for u in utts if u.text_norm]
        metrics = {"utterances": len(utts), "words": n_words, "rttm_speech_s": round(rttm_total, 3),
                   "timeline_speech_s": round(speech, 3), "words_outside": len(outside), "low_coverage": len(low),
                   "align_coverage_min": round(min(covs), 4) if covs else 1.0}
        problems = []
        if not rttm_ok:
            problems.append(f"RTTM speech {rttm_total:.3f} s != timeline {speech:.3f} s")
        if outside:
            problems.append(f"{len(outside)} utterances with words outside their segment ({', '.join(outside[:3])})")
        if low:
            problems.append(f"{len(low)} utterances under {th.align_coverage_min:.0%} alignment coverage "
                            f"({', '.join(low[:3])})")
        if problems:
            return StageResult(state="fail", detail="; ".join(problems)[:300], metrics=metrics)
        return StageResult(state="pass", detail=f"{len(utts)} utterances · {n_words} words · "
                                                f"coverage ≥ {metrics['align_coverage_min']:.0%}", metrics=metrics)
