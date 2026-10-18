"""The M4-M8 contract: phase C / stage 6-8 artifact models and the fake audio backends behind them."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ROOT
from synthetics.artifacts.io import read_model, write_model
from synthetics.artifacts.labels import Segment, Timeline, TimelineStats
from synthetics.artifacts.paths import MeetingPaths
from synthetics.artifacts.voice import Synthesis, UtteranceSynth, VoiceAssignment
from synthetics.backends.base import Audio, VoiceRef
from synthetics.backends.fakes_audio import (
    FakeAligner,
    FakeSpeakerEmbedder,
    FakeTranscriber,
    FakeVoiceCloner,
    fake_reference,
    spoken_words,
)
from synthetics.backends.registry import Backends
from synthetics.config import read_run_config
from synthetics.config.schema import Role


def _trim(a: Audio, pad_s: float = 0.05) -> Audio:
    idx = np.flatnonzero(np.abs(a.samples) > 0.05 * np.abs(a.samples).max())
    pad = int(pad_s * a.sr)
    return Audio(a.samples[max(0, idx[0] - pad): idx[-1] + 1 + pad], a.sr)


def _round_trip(a: Audio, tmp_path, name: str) -> Audio:
    """What stage 4 does before stage 5 reads the clip: resample, trim, save as PCM16, load."""
    path = tmp_path / f"{name}.wav"
    _trim(a.resample(16000)).save(path)
    return Audio.load(path)


def test_transcriber_recovers_cloned_text_after_stage4_processing(tmp_path):
    tts = FakeVoiceCloner(sr=24000)
    texts = ["Ok so the vendor (lah) cannot make it [laugh] by Friday.", "(mm)", "We wan- we want to push it."]
    clips = [_round_trip(tts.clone(t, VoiceRef(fake_reference("00860519_0001"), "x", "00860519_0001"), seed=i),
                         tmp_path, f"u{i}") for i, t in enumerate(texts)]
    asr = FakeTranscriber()
    assert [asr.transcribe(c) for c in clips] == [" ".join(spoken_words(t)) for t in texts]
    assert spoken_words(texts[0]) == ["ok", "so", "the", "vendor", "lah", "cannot", "make", "it", "by", "friday"]
    assert FakeTranscriber(drop_every=2).transcribe(clips[2]).split() == ["we", "we", "to", "it"]


def test_embedder_matches_speakers_and_separates_voices(tmp_path):
    tts, sv = FakeVoiceCloner(), FakeSpeakerEmbedder("sv_wavlm")
    ref_a1, ref_a2 = fake_reference("00000001_0001"), fake_reference("00000001_0002")
    a1 = _round_trip(tts.clone("first line here", VoiceRef(ref_a1, "x", "00000001_0001"), seed=1), tmp_path, "a1")
    a2 = _round_trip(tts.clone("another clip same voice", VoiceRef(ref_a2, "x", "00000001_0002"), seed=2), tmp_path, "a2")
    others = [tts.clone("hello there", VoiceRef(fake_reference(f"{k:08d}_0001"), "x", f"{k:08d}_0001"), seed=k)
              for k in range(2, 22)]
    e = sv.embed([ref_a1, a1, a2, *others])
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0)
    assert e[0] @ e[1] == pytest.approx(1.0) and e[1] @ e[2] == pytest.approx(1.0)  # clip swap keeps the voice
    cross = e[3:] @ e[3:].T
    assert np.max(cross[~np.eye(len(others), dtype=bool)]) < 0.9  # 20 voices, none nearly identical


def test_embedder_matches_a_real_reference_to_its_clones():
    rng = np.random.default_rng(0)
    real = Audio((0.1 * rng.standard_normal(16000 * 6)).astype(np.float32), 16000)  # no fake tones
    clone = FakeVoiceCloner(sr=16000).clone("some words to say", VoiceRef(real, "x", "00999999_0007"), seed=3)
    e = FakeSpeakerEmbedder().embed([real, clone])
    assert e[0] @ e[1] == pytest.approx(1.0)


def test_aligner_spreads_words_inside_the_active_region():
    clip = FakeVoiceCloner(sr=16000).clone("can or not meh", VoiceRef(fake_reference("1_1"), "x", "1_1"), seed=0)
    padded = Audio(np.concatenate([np.zeros(8000, np.float32), clip.samples, np.zeros(8000, np.float32)]), 16000)
    words = FakeAligner().align(padded, "Can or not (meh)?", language="en")
    assert [w.word for w in words] == ["can", "or", "not", "meh"]
    assert words[0].start >= 0.45 and words[-1].end <= 0.5 + clip.duration + 0.05
    assert all(a.end == pytest.approx(b.start) for a, b in zip(words, words[1:], strict=False))


def test_fake_mode_builds_every_phase_c_and_d_role(cfg):
    b = Backends(cfg.backends)
    for role in (Role.TTS, Role.ASR_PRIMARY, Role.ASR_SINGLISH, Role.SV_WAVLM, Role.SV_REDIMNET, Role.QUALITY,
                 Role.ALIGNER):
        assert b.get(role) is not None


def test_models_round_trip(tmp_path):
    paths = MeetingPaths(tmp_path, "m-000001")
    syn = Synthesis(meeting_id="m-000001", service="voxcpm2", tts_sample_rate=24000, sample_rate=16000,
                    voices={"S01": VoiceAssignment(speaker_id="S01", nsc_speaker="00860519", clip_id="00860519_0845",
                                                   ref_text="ya twenty four hours", ref_duration=9.06)},
                    utterances=[UtteranceSynth(turn_id="t0001", speaker_id="S01", source="tts", tts_text="Ok lah.",
                                               seed=7, attempt=1, duration=1.2)])
    write_model(paths.synthesis, syn)
    assert read_model(paths.artifact("synthesis"), Synthesis) == syn
    tl = Timeline(meeting_id="m-000001", sample_rate=16000,
                  segments=[Segment(turn_id="t0001", speaker_id="S01", kind="normal", start=1.0, end=2.2)],
                  stats=TimelineStats(duration=3.2, speech_time=1.2, overlap_share=0.0, backchannels_per_min=0.0,
                                      interruption_share=0.0, long_pauses=0))
    write_model(paths.timeline, tl)
    assert read_model(paths.timeline, Timeline) == tl


def test_existing_frozen_run_configs_still_load():
    for path in sorted((ROOT / "runs").glob("*/config.yaml")):
        cfg = read_run_config(path)
        assert cfg.voice.trim_pad_s == 0.05 and cfg.thresholds.align_coverage_min == 0.90
