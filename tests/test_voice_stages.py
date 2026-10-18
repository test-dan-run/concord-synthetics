"""Stages 4-5 and phase C against the fake audio backends: the stage 4 -> 6 contract, re-synthesis, ASR-limited
tagging, reference swaps, exhaustion, the sound library, and cache hits."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml

from helpers import fake_run
from synthetics.artifacts.io import read_model, write_model
from synthetics.artifacts.models import Script
from synthetics.artifacts.paths import RunPaths
from synthetics.artifacts.voice import Synthesis, VoiceValidation
from synthetics.backends.base import Audio
from synthetics.backends.fakes_audio import FakeSpeakerEmbedder, FakeTranscriber
from synthetics.backends.registry import Backends
from synthetics.config.schema import Role, StageId
from synthetics.orchestrator.phase_c import run_phase_c, summarise_c
from synthetics.orchestrator.scheduler import PhaseScheduler, RunReport
from synthetics.orchestrator.status import StatusStore
from synthetics.stages.s4_voice import split_text
from synthetics.voice.audio_ops import join, trim
from synthetics.voice.nonlexical import SoundLibrary, synthetic
from voice_helpers import fake_refs


@pytest.fixture(scope="session")
def refs_dir(tmp_path_factory):
    """nsc-part3-refs/clean/ with two fake reference clips for every pool speaker."""
    return fake_refs(tmp_path_factory.mktemp("refs"))


@pytest.fixture(autouse=True)
def voice_env(refs_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("NSC_REFS_DIR", str(refs_dir))
    monkeypatch.setenv("SOUND_LIBRARY_DIR", str(tmp_path / "no-library"))


def scripted_run(tmp_path, n=1, seed=11, **overrides):
    rd, cfg = fake_run(tmp_path, n=n, seed=seed, **overrides)
    PhaseScheduler(rd, cfg).run(until=StageId.SCRIPT_VAL)
    return rd, cfg


def phase_c(rd, cfg, **overrides):
    sched = PhaseScheduler(rd, cfg, backends=Backends(cfg.backends, overrides=overrides or None))
    report = RunReport()
    run_phase_c(sched, RunPaths(rd).meeting_ids(), report)
    return report, sched


# ---------------------------------------------------------------- helpers


def test_split_text_keeps_pieces_under_the_limit():
    text = "First sentence here. Second one is a bit longer, with a clause; and more. Third!"
    assert split_text(text, 200) == [text]
    parts = split_text(text, 40)
    assert all(len(p) <= 40 for p in parts) and " ".join(parts).split() == text.split()
    long = " ".join(["word"] * 60)
    assert all(len(p) <= 50 for p in split_text(long, 50))


def test_trim_and_join():
    sr = 16000
    x = np.zeros(sr, np.float32)
    x[4000:8000] = 0.5
    t = trim(Audio(x, sr), pad_s=0.05)
    assert t.duration == pytest.approx(0.25 + 0.1, abs=0.011)
    j = join([t, t], gap_s=0.2)
    assert j.duration == pytest.approx(2 * t.duration + 0.2, abs=1e-3)
    assert trim(Audio(np.zeros(100, np.float32), sr), pad_s=0.05).samples.size == 100  # silent: unchanged


@pytest.mark.parametrize("tag", ["laugh", "cough", "breath", "noise"])
def test_synthetic_sounds(tag):
    a = synthetic(tag, 16000, 3)
    assert 0.2 < a.duration < 1.0 and np.abs(a.samples).max() > 0.01
    assert np.array_equal(a.samples, synthetic(tag, 16000, 3).samples)


def test_sound_library_picks_gender_matched_clips(tmp_path):
    lib_dir = tmp_path / "nonlexical"
    (lib_dir / "laugh").mkdir(parents=True)
    for cid in ("laugh-f-1", "laugh-m-1"):
        Audio(np.full(800, 0.1, np.float32), 16000).save(lib_dir / f"laugh/{cid}.wav")
    (lib_dir / "index.yaml").write_text(yaml.safe_dump({"version": "nl-test", "clips": [
        {"id": "laugh-f-1", "tag": "laugh", "file": "laugh/laugh-f-1.wav", "gender": "F"},
        {"id": "laugh-m-1", "tag": "laugh", "file": "laugh/laugh-m-1.wav", "gender": "M"}]}))
    lib = SoundLibrary.load(lib_dir)
    assert lib.version == "nl-test"
    assert lib.pick("laugh", "F", "k1")[0] == "laugh-f-1" and lib.pick("laugh", "M", "k1")[0] == "laugh-m-1"
    assert lib.pick("cough", "F", "k1") is None
    assert SoundLibrary.load(tmp_path / "missing").version == "none"


# ---------------------------------------------------------------- phase C end to end


def test_phase_c_meets_the_stage6_contract(tmp_path):
    rd, cfg = scripted_run(tmp_path, n=2)
    report, _ = phase_c(rd, cfg)
    rp = RunPaths(rd)
    store = StatusStore(rd)
    for mid in rp.meeting_ids():
        mp = rp.meeting(mid)
        assert store.get(mid).get(StageId.VOICE_VAL).state == "pass", store.get(mid).get(StageId.VOICE_VAL).detail
        script = read_model(mp.script, Script)
        syn = read_model(mp.synthesis, Synthesis)
        assert [u.turn_id for u in syn.utterances] == [t.turn_id for t in script.turns]
        assert sorted(p.name for p in mp.utterances_dir.iterdir()) == sorted(f"{t.turn_id}.wav" for t in script.turns)
        assert syn.sample_rate == cfg.sample_rate and syn.tts_sample_rate == 24000
        for u in syn.utterances[:25]:
            a = Audio.load(mp.utterance(u.turn_id))
            assert a.sr == cfg.sample_rate and a.duration == pytest.approx(u.duration, abs=1e-3)
            lead = np.argmax(np.abs(a.samples) > 0.01 * np.abs(a.samples).max()) / a.sr
            assert lead <= cfg.voice.trim_pad_s + 0.011
        vv = read_model(mp.voice_validation, VoiceValidation)
        assert vv.decision == "pass" and all(u.passed for u in vv.utterances)
        for u in vv.utterances:
            assert set(u.sv) == ({"sv_wavlm", "sv_redimnet"} if "sv" not in u.skipped else set())
            assert u.asr_limited is False
        assert {v.speaker_id for v in syn.voices.values()} == {s for s in syn.voices}
    s = summarise_c(rd)
    assert s["pass_rate"] == 1.0 and s["asr_limited_rate"] == 0.0


def test_phase_c_is_a_cache_hit_the_second_time(tmp_path):
    rd, cfg = scripted_run(tmp_path)
    phase_c(rd, cfg)
    report, sched = phase_c(rd, cfg)
    assert report.outcomes == []  # nothing pending
    tts = sched.backends.get(Role.TTS) if sched.backends._clients else None
    assert tts is None or tts.calls == []


def test_primary_asr_failures_pass_through_the_singlish_asr(tmp_path):
    rd, cfg = scripted_run(tmp_path)
    phase_c(rd, cfg, **{Role.ASR_PRIMARY: FakeTranscriber("bad-asr", drop_every=2)})
    mid = RunPaths(rd).meeting_ids()[0]
    vv = read_model(RunPaths(rd).meeting(mid).voice_validation, VoiceValidation)
    assert vv.decision == "pass"
    checked = [u for u in vv.utterances if u.asr]
    limited = [u for u in checked if u.asr_limited]
    assert limited and all(len(u.asr) == 2 and not u.asr[0].passed and u.asr[1].passed for u in limited)
    assert all(u.asr[0].service == cfg.backends.roles[Role.ASR_PRIMARY] for u in checked)


class FirstTakeFails:
    """Both ASRs hear nothing on the first take of every turn whose text contains `word`; later takes are fine."""

    seen: dict[str, str] = {}

    def __init__(self, word: str, *, always: bool = False):
        self.inner, self.word, self.always = FakeTranscriber(), word, always

    def transcribe(self, audio, *, language=None):
        text = self.inner.transcribe(audio, language=language)
        if self.word not in text.split():
            return text
        sha = hashlib.sha256(audio.samples.tobytes()).hexdigest()
        first = FirstTakeFails.seen.setdefault(text, sha)
        return "" if (self.always or first == sha) else text


def unique_word(script: Script) -> str:
    """A word in exactly one turn, whose text no other turn repeats."""
    texts = [t.text_norm for t in script.turns]
    for t in script.turns:
        if texts.count(t.text_norm) == 1:
            for w in t.text_norm.split():
                if len(w) > 3 and sum(w in x.split() for x in texts) == 1:
                    return w
    pytest.skip("no turn with a unique word in this fake script")


def test_failing_utterances_are_resynthesised_with_a_new_seed(tmp_path):
    FirstTakeFails.seen = {}
    rd, cfg = scripted_run(tmp_path)
    mid = RunPaths(rd).meeting_ids()[0]
    script = read_model(RunPaths(rd).meeting(mid).script, Script)
    word = unique_word(script)
    phase_c(rd, cfg, **{Role.ASR_PRIMARY: FirstTakeFails(word), Role.ASR_SINGLISH: FirstTakeFails(word)})
    mp = RunPaths(rd).meeting(mid)
    syn = read_model(mp.synthesis, Synthesis)
    vv = read_model(mp.voice_validation, VoiceValidation)
    hit = [t.turn_id for t in script.turns if word in t.text_norm.split()]
    assert vv.decision == "pass"
    by_id = {u.turn_id: u for u in syn.utterances}
    assert all(by_id[t].attempt == 2 for t in hit)
    assert all(u.attempt == 1 for u in syn.utterances if u.turn_id not in hit)
    seeds = {u.turn_id: u.seed for u in syn.utterances}
    assert len({seeds[t] for t in hit}) == len(hit)
    st = StatusStore(rd).get(mid)
    assert st.get(StageId.VOICE).attempts == 2 and st.get(StageId.VOICE).feedback is None
    assert summarise_c(rd)["resynthesis_rate"] > 0


def test_meeting_fails_when_an_utterance_never_passes(tmp_path):
    FirstTakeFails.seen = {}
    rd, cfg = scripted_run(tmp_path, **{"retries.tts_attempts": 2})
    mid = RunPaths(rd).meeting_ids()[0]
    script = read_model(RunPaths(rd).meeting(mid).script, Script)
    word = next(w for t in script.turns for w in t.text_norm.split() if len(w) > 5)
    asr = FirstTakeFails(word, always=True)
    phase_c(rd, cfg, **{Role.ASR_PRIMARY: asr, Role.ASR_SINGLISH: asr})
    st = StatusStore(rd).get(mid).get(StageId.VOICE_VAL)
    assert st.state == "fail" and "still failing after the last attempt" in st.detail
    syn = read_model(RunPaths(rd).meeting(mid).synthesis, Synthesis)
    assert max(u.attempt for u in syn.utterances) == 2


def test_keep_flagged_passes_with_the_failures_listed(tmp_path):
    FirstTakeFails.seen = {}
    rd, cfg = scripted_run(tmp_path, **{"retries.tts_attempts": 1, "voice.on_exhausted": "keep_flagged"})
    mid = RunPaths(rd).meeting_ids()[0]
    script = read_model(RunPaths(rd).meeting(mid).script, Script)
    word = next(w for t in script.turns for w in t.text_norm.split() if len(w) > 5)
    asr = FirstTakeFails(word, always=True)
    phase_c(rd, cfg, **{Role.ASR_PRIMARY: asr, Role.ASR_SINGLISH: asr})
    vv = read_model(RunPaths(rd).meeting(mid).voice_validation, VoiceValidation)
    assert vv.decision == "pass" and vv.kept_failed
    assert set(vv.kept_failed) == {t.turn_id for t in script.turns if word in t.text_norm.split()}


class TwoClose(FakeSpeakerEmbedder):
    """The first two voices it sees (the references of S01 and S02) embed to one vector: that pair is too close
    until one of them gets another NSC speaker's voice."""

    def __init__(self, name):
        super().__init__(name)
        self.first: list[bytes] = []

    def embed(self, clips):
        e = super().embed(clips)
        for row in e:
            key = row.tobytes()
            if len(self.first) < 2 and key not in self.first:
                self.first.append(key)
        out = e.copy()
        if len(self.first) == 2:
            for i, row in enumerate(e):
                if row.tobytes() == self.first[1]:
                    out[i] = np.frombuffer(self.first[0], dtype=e.dtype)
        return out


def test_too_close_speakers_get_a_clip_swap_then_a_speaker_swap(tmp_path):
    rd, cfg = scripted_run(tmp_path)
    phase_c(rd, cfg, **{Role.SV_REDIMNET: TwoClose("sv_redimnet")})
    rp = RunPaths(rd)
    mid = rp.meeting_ids()[0]
    vv = read_model(rp.meeting(mid).voice_validation, VoiceValidation)
    syn = read_model(rp.meeting(mid).synthesis, Synthesis)
    splits = json.loads(rp.speaker_splits.read_text())["speakers"]
    assert vv.decision == "pass", StatusStore(rd).get(mid).get(StageId.VOICE_VAL).detail
    assert [s.kind for s in syn.swaps] == ["clip", "speaker"] and vv.swaps == syn.swaps
    clip, sp = syn.swaps
    assert clip.speaker_id == sp.speaker_id in ("S01", "S02") and "too close" in clip.reason
    assert clip.from_speaker == clip.to_speaker and clip.from_clip != clip.to_clip
    a, b = splits[sp.from_speaker], splits[sp.to_speaker]
    assert (a["split"], a["gender"], a["ethnicity"]) == (b["split"], b["gender"], b["ethnicity"])
    assert syn.voices[sp.speaker_id].nsc_speaker == sp.to_speaker
    people = json.loads(rp.meeting(mid).meeting.read_text())["people"]
    assert sp.to_speaker not in {p["nsc_speaker"]["speaker"] for p in people}
    turns = [u for u in syn.utterances if u.speaker_id == sp.speaker_id]
    assert all(u.attempt == 3 for u in turns)  # made with the first clip, the second clip, the new speaker


def test_meeting_fails_when_no_swap_is_left(tmp_path):
    rd, cfg = scripted_run(tmp_path, **{"voice.max_speaker_swaps": 0})
    phase_c(rd, cfg, **{Role.SV_REDIMNET: TwoClose("sv_redimnet")})
    mid = RunPaths(rd).meeting_ids()[0]
    st = StatusStore(rd).get(mid).get(StageId.VOICE_VAL)
    assert st.state == "fail" and "no swap left" in st.detail
    vv = read_model(RunPaths(rd).meeting(mid).voice_validation, VoiceValidation)
    assert [s.kind for s in vv.swaps] == ["clip"]


def test_nonspeech_tags_use_library_clips_or_stand_ins(tmp_path, monkeypatch):
    lib_root = tmp_path / "lib"
    (lib_root / "nonlexical/cough").mkdir(parents=True)
    Audio(np.full(4000, 0.2, np.float32), 16000).save(lib_root / "nonlexical/cough/c1.wav")
    (lib_root / "nonlexical/index.yaml").write_text(yaml.safe_dump(
        {"version": "nl-t", "clips": [{"id": "cough-1", "tag": "cough", "file": "cough/c1.wav"}]}))
    monkeypatch.setenv("SOUND_LIBRARY_DIR", str(lib_root))
    rd, cfg = scripted_run(tmp_path)
    mp = RunPaths(rd).meeting(RunPaths(rd).meeting_ids()[0])
    script = read_model(mp.script, Script)
    t0, t1, t2 = script.turns[0], script.turns[1], script.turns[2]
    turns = [t0.model_copy(update={"text": "[cough] " + t0.text, "nonspeech": ["cough"]}),
             t1.model_copy(update={"text": t1.text + " [laugh]", "nonspeech": ["laugh"]}),
             t2.model_copy(update={"text": "[breath]", "text_norm": "", "nonspeech": ["breath"]}), *script.turns[3:]]
    write_model(mp.script, script.model_copy(update={"turns": turns}))
    phase_c(rd, cfg)
    syn = {u.turn_id: u for u in read_model(mp.synthesis, Synthesis).utterances}
    assert syn[t0.turn_id].source == "tts+library" and syn[t0.turn_id].library_clips == ["cough-1"]
    assert syn[t1.turn_id].source == "tts" and syn[t1.turn_id].tts_text.endswith("haha")  # laugh cloned
    assert syn[t2.turn_id].source == "library" and syn[t2.turn_id].library_clips == ["synthetic:breath"]
    assert syn[t2.turn_id].tts_text == ""
    vv = {u.turn_id: u for u in read_model(mp.voice_validation, VoiceValidation).utterances}
    assert vv[t1.turn_id].passed  # "haha" is not held against the ASR check
    assert "asr" in vv[t2.turn_id].skipped
