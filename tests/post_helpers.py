"""Test helpers for stages 6-8: a stand-in for phase C (stages 4-5) built on the fake audio backends, and
synthetic meetings for the large-sample timeline checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthetics.artifacts.io import read_model, write_model
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.models import Script
from synthetics.artifacts.paths import MeetingPaths, RunPaths
from synthetics.artifacts.voice import Synthesis, UtteranceSynth, VoiceAssignment, VoiceValidation
from synthetics.backends.base import VoiceRef
from synthetics.backends.fakes_audio import FakeVoiceCloner, fake_reference
from synthetics.config.schema import RunConfig, StageId
from synthetics.orchestrator.scheduler import PhaseScheduler
from synthetics.orchestrator.status import StatusStore
from synthetics.stages.s6_assemble import Utt


def fake_phase_c(run_dir: Path, cfg: RunConfig, meeting_ids: list[str] | None = None) -> list[str]:
    """What stages 4 and 5 hand to stage 6: one utterance WAV per script turn, synthesis.json, a passing
    voice_validation.json and pass records for stages 4-5. Returns the meetings it filled."""
    store = StatusStore(run_dir)
    ids = meeting_ids or RunPaths(run_dir).meeting_ids()
    done = []
    tts = FakeVoiceCloner(sr=cfg.sample_rate)
    for mid in ids:
        if store.get(mid).get(StageId.SCRIPT_VAL).state != "pass":
            continue
        paths = MeetingPaths(run_dir, mid)
        m = read_model(paths.meeting, MeetingSpec)
        script = read_model(paths.script, Script)
        voices, utts = {}, []
        for p in m.people:
            clip = f"{p.nsc_speaker.speaker}_0001"
            voices[p.speaker_id] = VoiceAssignment(speaker_id=p.speaker_id, nsc_speaker=p.nsc_speaker.speaker,
                                                   clip_id=clip, ref_text="reference words", ref_duration=8.0)
        for k, t in enumerate(script.turns):
            clip = voices[t.speaker_id].clip_id
            audio = tts.clone(t.text, VoiceRef(fake_reference(clip, sr=cfg.sample_rate), "x", clip), seed=k)
            audio.save(paths.utterance(t.turn_id))
            utts.append(UtteranceSynth(turn_id=t.turn_id, speaker_id=t.speaker_id, source="tts", tts_text=t.text,
                                       seed=k, attempt=1, duration=round(audio.duration, 4)))
        write_model(paths.synthesis, Synthesis(meeting_id=mid, service="fake-tts", tts_sample_rate=cfg.sample_rate,
                                               sample_rate=cfg.sample_rate, voices=voices, utterances=utts))
        write_model(paths.voice_validation, VoiceValidation(meeting_id=mid, utterances=[], speakers=[], pairs=[],
                                                            decision="pass"))
        for stage in (StageId.VOICE, StageId.VOICE_VAL):
            store.mark(mid, stage, "pass", cache_key="fake-phase-c", detail="fake phase C")
        done.append(mid)
    return done


def run_to_phase_c(run_dir: Path, cfg: RunConfig) -> list[str]:
    PhaseScheduler(run_dir, cfg).run()
    return fake_phase_c(run_dir, cfg)


def synthetic_utts(m: MeetingSpec, cfg: RunConfig, rng: np.random.Generator) -> list[Utt]:
    """A stage 2 turn plan with utterance lengths like TTS output, for timeline statistics over many meetings."""
    from synthetics.stages.script_plan import build_plan

    plan = build_plan(m, cfg.turn_taking, rng)
    out = []
    for j, t in enumerate(plan.turns):
        if t.interaction == "backchannel":
            dur = rng.uniform(0.4, 0.9)
        elif t.interaction == "presentation":
            dur = rng.uniform(5.0, 10.0)
        else:
            dur = float(np.clip(rng.lognormal(np.log(4.0), 0.5), 0.8, 15.0))
        out.append(Utt(f"t{j + 1:04d}", t.speaker_id, t.interaction, dur))
    return out
