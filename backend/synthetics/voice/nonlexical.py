"""Non-lexical sounds for the script's square-bracket tags: [laugh] [cough] [breath] [noise].

For each tag, stage 4 uses, in order:
1. a real clip from the sound library, `<SOUND_LIBRARY_DIR>/nonlexical/index.yaml` (gender-matched when the clip
   records a gender), chosen deterministically from the meeting, speaker and turn;
2. else the text configured in `voice.nonlexical_tts` (laugh -> "haha"), cloned in the speaker's own voice;
3. else a synthetic sound (shaped, band-limited noise), so every tag still makes some sound.

index.yaml:
    version: nl-1
    clips:
      - {id: laugh-f-001, tag: laugh, file: laugh/laugh-f-001.wav, gender: F, source: <url>, licence: CC-BY-4.0}
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from ..backends.base import Audio

DEFAULT_ROOT = Path("/home/user/code/datasets/sound-library")
SYNTH_VERSION = "syn-1"


def library_root() -> Path:
    return Path(os.environ.get("SOUND_LIBRARY_DIR") or DEFAULT_ROOT) / "nonlexical"


@dataclass(frozen=True)
class LibraryClip:
    id: str
    tag: str
    path: Path
    gender: str | None


class SoundLibrary:
    def __init__(self, clips: list[LibraryClip], version: str):
        self.clips = clips
        self.version = version  # part of the stage 4 cache key

    @classmethod
    def load(cls, root: Path | None = None) -> SoundLibrary:
        return _load(str(root or library_root()))

    def pick(self, tag: str, gender: str | None, key: str) -> tuple[str, Audio] | None:
        cands = [c for c in self.clips if c.tag == tag and (c.gender in (None, "any") or gender is None or c.gender == gender)]
        if not cands:
            return None
        c = sorted(cands, key=lambda c: c.id)[_h(key) % len(cands)]
        return c.id, _audio(str(c.path))


@lru_cache(maxsize=4)
def _load(root: str) -> SoundLibrary:
    index = Path(root) / "index.yaml"
    if not index.is_file():
        return SoundLibrary([], "none")
    data = yaml.safe_load(index.read_text(encoding="utf-8")) or {}
    clips = [LibraryClip(id=str(c["id"]), tag=str(c["tag"]), path=Path(root) / c["file"], gender=c.get("gender"))
             for c in data.get("clips") or []]
    return SoundLibrary(clips, str(data.get("version", "unversioned")))


@lru_cache(maxsize=256)
def _audio(path: str) -> Audio:
    return Audio.load(path)


def _h(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def _band_noise(n: int, sr: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    hz = np.fft.rfftfreq(n, 1 / sr)
    spec[(hz < lo) | (hz > min(hi, sr / 2 - 1))] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() or 1.0)


def _env(n: int, attack: float, sr: int) -> np.ndarray:
    a = max(1, int(attack * sr))
    e = np.ones(n)
    e[:a] = np.linspace(0, 1, a)
    e[a:] = np.exp(-np.linspace(0, 4, n - a))
    return e


def synthetic(tag: str, sr: int, seed: int) -> Audio:
    """A plain stand-in sound for `tag`: breath (soft 0.45 s hiss), cough (two sharp bursts), laugh (four short
    voiced bursts), noise (a 0.3 s rustle)."""
    rng = np.random.default_rng(seed)
    if tag == "breath":
        n = int(0.45 * sr)
        x = _band_noise(n, sr, 300, 3000, rng) * np.sin(np.linspace(0, np.pi, n)) * 0.08
    elif tag == "cough":
        bursts = []
        for k in range(2):
            n = int((0.16 if k == 0 else 0.12) * sr)
            bursts += [_band_noise(n, sr, 200, 4000, rng) * _env(n, 0.005, sr) * 0.5, np.zeros(int(0.08 * sr))]
        x = np.concatenate(bursts[:-1])
    elif tag == "laugh":
        t = np.arange(int(0.11 * sr)) / sr
        f0 = 180 + 60 * rng.random()
        burst = (np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t)) * np.sin(np.pi * t / t[-1]) * 0.3
        x = np.concatenate([np.concatenate([burst, np.zeros(int(0.07 * sr))]) for _ in range(4)])
    else:  # noise
        n = int(0.3 * sr)
        x = _band_noise(n, sr, 1000, 7000, rng) * _env(n, 0.01, sr) * 0.15
    return Audio(x.astype(np.float32), sr)
