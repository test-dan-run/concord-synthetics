"""Deterministic fake audio backends (cloner, transcriber, speaker embedder, aligner) for unit tests and
`mode: fake` runs. No GPU, no network.

`FakeVoiceCloner` renders a harmonic tone at a pitch chosen by the voice, plus two quiet chords of pure tones:
- `ID_TONES` encode a per-process utterance id; `FakeTranscriber` decodes it and returns the cloned text.
- `VOICE_TONES` encode the voice (the NSC speaker in the clip id); `FakeSpeakerEmbedder` decodes them, so
  clones of one speaker match and different speakers are near-orthogonal.
Both chords survive resampling, PCM16 round trips, gain changes and silence trimming. A pilot tone marks fake
audio. The text registry lives in the process: a fake run that restarts between stages 4 and 5 transcribes ""
for clips made before the restart.

Use `fake_reference(clip_id)` to build reference clips for tests; real NSC references (no chords) are matched
to the voice they were cloned with through a registry of their sample hashes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from .base import Audio, VoiceRef, WordTiming

PILOT_HZ = 900.0
ID_BITS = 20
ID_TONES = PILOT_HZ + 150.0 * np.arange(1, ID_BITS + 1)  # 1050 .. 3900 Hz
VOICE_BITS = 16
VOICE_TONES = 4000.0 + 180.0 * np.arange(VOICE_BITS)  # 4000 .. 6700 Hz, inside a 16 kHz passband
TONE_AMP = 0.012
F0_RANGE = (90.0, 260.0)

_SPOKEN: dict[int, str] = {}  # utterance id -> cloned text
_REF_VOICE: dict[str, int] = {}  # sha of a reference clip's samples -> voice bits


def _h(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def spoken_words(text: str) -> list[str]:
    """Lowercase words a listener would hear: non-speech tags dropped, particles kept, cut-off hyphens removed."""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return [w.strip("-'").lower() for w in re.findall(r"[A-Za-z0-9'][A-Za-z0-9'-]*", text) if w.strip("-'")]


def voice_key(clip_id: str) -> str:
    """NSC clip ids are <speaker>_<nnnn>; every clip of a speaker clones the same fake voice."""
    return clip_id.split("_")[0] if clip_id else ""


def voice_bits(key: str) -> int:
    return _h("voice", key) & ((1 << VOICE_BITS) - 1)


def _tones(freqs: np.ndarray, bits: int, t: np.ndarray) -> np.ndarray:
    on = [f for k, f in enumerate(freqs) if (bits >> k) & 1]
    return TONE_AMP * sum((np.sin(2 * np.pi * f * t) for f in on), np.zeros_like(t))


def _render(key: str, n_words: int, sr: int, seed: int, uid: int, sec_per_word: float, min_s: float) -> Audio:
    dur = max(min_s, 0.2 + sec_per_word * n_words)
    t = np.arange(int(round(dur * sr))) / sr
    rng = np.random.default_rng(_h(key, n_words, seed, uid))
    f0 = F0_RANGE[0] + (_h("f0", key) % 1000) / 1000 * (F0_RANGE[1] - F0_RANGE[0])
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi))  # syllable rate
    voice = env * (0.3 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(4 * np.pi * f0 * t) + 0.08 * np.sin(6 * np.pi * f0 * t))
    chord = TONE_AMP * np.sin(2 * np.pi * PILOT_HZ * t) + _tones(ID_TONES, uid, t) + _tones(VOICE_TONES, voice_bits(key), t)
    ramp = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.005)
    return Audio(((voice + chord) * ramp * (0.9 + 0.1 * rng.random())).astype(np.float32), sr)


def fake_reference(clip_id: str, *, sr: int = 16000, n_words: int = 25) -> Audio:
    """A reference clip in the fake voice of `clip_id`'s speaker (utterance id 0: transcribes as "")."""
    return _render(voice_key(clip_id), n_words, sr, 0, 0, 0.32, 0.4)


def _decode(audio: Audio, freqs: np.ndarray) -> int | None:
    x = audio.samples
    if len(x) < int(0.1 * audio.sr):
        return None
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    hz = np.fft.rfftfreq(len(x), 1 / audio.sr)

    def mag(f: float) -> float:
        i = int(np.searchsorted(hz, f))
        return float(spec[max(0, i - 2): i + 3].max()) if i < len(spec) else 0.0

    pilot = mag(PILOT_HZ)
    floor = float(np.median(spec[(hz > 950) & (hz < 7000)])) if len(spec) else 0.0
    if pilot <= 0 or pilot < 10 * floor:
        return None
    return sum(1 << k for k, f in enumerate(freqs) if mag(f) > 0.5 * pilot)


def _sha(audio: Audio) -> str:
    return hashlib.sha256(audio.samples.tobytes() + str(audio.sr).encode()).hexdigest()


class FakeVoiceCloner:
    """Duration grows with the word count; the voice comes from the reference clip's NSC speaker."""

    def __init__(self, *, sr: int = 24000, sec_per_word: float = 0.32, min_s: float = 0.4):
        self.sr = sr
        self.sec_per_word = sec_per_word
        self.min_s = min_s
        self.calls: list[dict] = []

    def clone(self, text: str, ref: VoiceRef, *, seed: int) -> Audio:
        key = voice_key(ref.clip_id)
        _REF_VOICE[_sha(ref.audio)] = voice_bits(key)
        uid = (len(_SPOKEN) + 1) % (1 << ID_BITS) or 1
        _SPOKEN[uid] = text
        self.calls.append({"text": text, "clip_id": ref.clip_id, "seed": seed})
        return _render(key, len(spoken_words(text)), self.sr, seed, uid, self.sec_per_word, self.min_s)


class FakeTranscriber:
    """Returns the cloned text as plain lowercase words. `drop_every=k` drops every k-th word (WER tests)."""

    def __init__(self, service: str = "fake-asr", *, drop_every: int = 0):
        self.service = service
        self.drop_every = drop_every

    def transcribe(self, audio: Audio, *, language: str | None = None) -> str:
        uid = _decode(audio, ID_TONES)
        words = spoken_words(_SPOKEN.get(uid, "")) if uid else []
        if self.drop_every:
            words = [w for i, w in enumerate(words, 1) if i % self.drop_every]
        return " ".join(words)


class FakeSpeakerEmbedder:
    """±1 per voice bit, L2-normalised: same voice -> 1.0, two random voices -> about 0 (16 bits)."""

    def __init__(self, name: str = "fake-sv"):
        self.name = name

    def embed(self, clips: Sequence[Audio]) -> np.ndarray:
        rows = []
        for c in clips:
            bits = _decode(c, VOICE_TONES)
            if bits is None:
                bits = _REF_VOICE.get(_sha(c), _h("unknown", _sha(c)) & ((1 << VOICE_BITS) - 1))
            v = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(VOICE_BITS)], dtype=np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows) if rows else np.zeros((0, VOICE_BITS), np.float32)


class FakeAligner:
    """Spreads the spoken words over the active part of the clip, in proportion to their length."""

    def align(self, audio: Audio, text: str, *, language: str) -> list[WordTiming]:
        words = spoken_words(text)
        x = np.abs(audio.samples)
        active = np.flatnonzero(x > 0.1 * x.max()) if x.size and x.max() > 0 else np.array([], int)
        if not words or active.size == 0:
            return []
        t0, t1 = active[0] / audio.sr, (active[-1] + 1) / audio.sr
        edges = np.concatenate([[0], np.cumsum([len(w) + 1 for w in words])]) / (sum(len(w) + 1 for w in words))
        return [WordTiming(w, float(t0 + a * (t1 - t0)), float(t0 + b * (t1 - t0)), 0.9)
                for w, a, b in zip(words, edges[:-1], edges[1:], strict=True)]
