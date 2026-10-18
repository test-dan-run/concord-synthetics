"""The phase C/D platform clients against stub services: request encoding, parameters, parsing, record/replay."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from synthetics.backends import aicore_impl
from synthetics.backends.aicore_impl import (
    AicoreAligner,
    AicoreSpeakerEmbedder,
    AicoreTranscriber,
    AicoreVoiceCloner,
    parse_transcription,
)
from synthetics.backends.base import Audio, VoiceRef
from synthetics.backends.registry import Backends
from synthetics.backends.transport import ReplayMiss, Transport
from synthetics.config.schema import BackendsConfig, Role


class StubOIP:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def infer(self, inputs, parameters=None):
        self.calls.append((inputs, parameters))
        return self.fn(inputs)


class Broken:
    def __getattr__(self, name):
        raise AssertionError("replay must not call the service")


def ref() -> VoiceRef:
    return VoiceRef(Audio(np.linspace(-0.1, 0.1, 16000).astype(np.float32), 16000), "my reference clip", "00000001_0001")


def test_cloner_sends_text_reference_seed_and_params(tmp_path):
    stub = StubOIP(lambda x: {"AUDIO": np.full(4800, 0.1, np.float32), "SAMPLE_RATE": np.array([48000], np.int32)})
    tts = AicoreVoiceCloner("voxcpm2", Transport("record", tmp_path), params={"mode": "ultimate"}, client=stub)
    a = tts.clone("Ok lah, we go first.", ref(), seed=42)
    assert a.sr == 48000 and a.duration == pytest.approx(0.1)
    inputs, params = stub.calls[0]
    assert inputs["TEXT"].tolist() == ["Ok lah, we go first."] and inputs["REF_TEXT"].tolist() == ["my reference clip"]
    assert inputs["SEED"].dtype == np.int64 and inputs["SEED"].tolist() == [42]
    assert inputs["REF_SAMPLE_RATE"].tolist() == [16000] and params == {"mode": "ultimate"}
    replay = AicoreVoiceCloner("voxcpm2", Transport("replay", tmp_path), params={"mode": "ultimate"}, client=Broken())
    assert np.array_equal(replay.clone("Ok lah, we go first.", ref(), seed=42).samples, a.samples)
    with pytest.raises(ReplayMiss):  # the text is part of the recording key
        replay.clone("Something else.", ref(), seed=42)
    with pytest.raises(ReplayMiss):  # so are the parameters
        AicoreVoiceCloner("voxcpm2", Transport("replay", tmp_path), params={"mode": "reference"},
                          client=Broken()).clone("Ok lah, we go first.", ref(), seed=42)


class StubTranscriptions:
    def __init__(self, text):
        self.text, self.calls = text, []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(with_raw_response=self))

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(http_response=SimpleNamespace(json=lambda: {"text": self.text}))


def test_transcriber_sends_16k_wav_and_strips_the_qwen_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr("aicore.model_name", lambda name: f"served-{name}")
    stub = StubTranscriptions("language English<asr_text>Ok lah we go first.")
    asr = AicoreTranscriber("qwen3-asr-1-7b", Transport("record", tmp_path), client=stub)
    clip = Audio(np.zeros(48000, np.float32), 48000)
    assert asr.transcribe(clip) == "Ok lah we go first."
    kw = stub.calls[0]
    name, data, mime = kw["file"]
    x, sr = sf.read(io.BytesIO(data))
    assert sr == 16000 and len(x) == 16000 and mime == "audio/wav" and kw["model"] == "served-qwen3-asr-1-7b"
    assert "language" not in kw
    asr.transcribe(clip, language="en")
    assert stub.calls[1]["language"] == "en"
    replay = AicoreTranscriber("qwen3-asr-1-7b", Transport("replay", tmp_path), client=Broken())
    assert replay.transcribe(clip) == "Ok lah we go first."


@pytest.mark.parametrize("raw, text", [({"text": " hello "}, "hello"), ("plain", "plain"),
                                       ({"text": "language Chinese<asr_text>好"}, "好"), ({}, "")])
def test_parse_transcription(raw, text):
    assert parse_transcription(raw) == text


def test_embedder_one_request_per_clip_and_unit_rows(tmp_path):
    stub = StubOIP(lambda x: {"EMBEDDINGS": np.array([[3.0, 4.0]], np.float32)})
    sv = AicoreSpeakerEmbedder("wavlm-sv", Transport("aicore"), name="sv_wavlm", client=stub)
    e = sv.embed([Audio(np.zeros(800, np.float32), 16000), Audio(np.zeros(1600, np.float32), 16000)])
    assert e.shape == (2, 2) and np.allclose(e[0], [0.6, 0.8])
    assert [c[0]["LENGTHS"].tolist() for c in stub.calls] == [[800], [1600]] and sv.name == "sv_wavlm"


def test_aligner_returns_word_timings(tmp_path):
    stub = StubOIP(lambda x: {"WORDS": np.array(["ok", "lah"], dtype=object),
                              "TIMES": np.array([[0.1, 0.3], [0.3, 0.5]], np.float32),
                              "SCORES": np.array([0.9, 0.8], np.float32)})
    al = AicoreAligner("qwen3-forced-aligner", Transport("record", tmp_path), client=stub)
    words = al.align(Audio(np.zeros(16000, np.float32), 16000), "ok lah", language="en")
    assert [(w.word, w.start, w.end) for w in words] == [("ok", pytest.approx(0.1), pytest.approx(0.3)),
                                                         ("lah", pytest.approx(0.3), pytest.approx(0.5))]
    inputs, _ = stub.calls[0]
    assert inputs["TEXT"].tolist() == ["ok lah"] and inputs["LANGUAGE"].tolist() == ["en"]
    replay = AicoreAligner("qwen3-forced-aligner", Transport("replay", tmp_path), client=Broken())
    assert [w.word for w in replay.align(Audio(np.zeros(16000, np.float32), 16000), "ok lah", language="en")] == ["ok", "lah"]


def test_registry_builds_the_platform_clients():
    roles = {Role.TTS: "voxcpm2", Role.ASR_PRIMARY: "qwen3-asr-1-7b", Role.ASR_SINGLISH: "meralion-3-3b-asr",
             Role.SV_WAVLM: "wavlm-sv", Role.SV_REDIMNET: "redimnet2-sv", Role.ALIGNER: "qwen3-forced-aligner"}
    b = Backends(BackendsConfig(mode="aicore", roles=roles))
    assert isinstance(b.get(Role.TTS), aicore_impl.AicoreVoiceCloner)
    assert isinstance(b.get(Role.ASR_SINGLISH), aicore_impl.AicoreTranscriber)
    assert b.get(Role.SV_REDIMNET).name == "sv_redimnet"
    assert isinstance(b.get(Role.ALIGNER), aicore_impl.AicoreAligner)
