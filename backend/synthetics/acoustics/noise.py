"""Noise for stage 7: stationary beds (HVAC, projector fan), keyboard typing and short events (door, chair,
cup, paper).

Every kind has a procedural generator, so stage 7 never waits on data. When the sound library
(`SimulationConfig.noise_library`, `<dir>/index.yaml`) lists real clips of a kind, one of them is used instead:
beds are looped with a crossfade to the meeting length, events are used as they are. Clips are resampled to the
run rate and mixed down to mono.

    # index.yaml
    clips:
      - {id: hvac-office-01, kind: hvac, file: hvac/office_01.wav, source: <url>, licence: CC0}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import signal

from ..backends.base import Audio

BEDS = ("hvac", "projector_fan")
EVENTS = ("door", "chair", "cup", "paper")
CHUNK_S = 60.0


@dataclass(frozen=True)
class LibraryClip:
    id: str
    kind: str
    path: Path
    source: str
    licence: str


def load_library(root: Path | None) -> dict[str, list[LibraryClip]]:
    if root is None or not (Path(root) / "index.yaml").is_file():
        return {}
    data = yaml.safe_load((Path(root) / "index.yaml").read_text(encoding="utf-8")) or {}
    out: dict[str, list[LibraryClip]] = {}
    for c in data.get("clips") or []:
        clip = LibraryClip(str(c["id"]), str(c["kind"]), Path(root) / c["file"], str(c.get("source", "")),
                           str(c.get("licence", "")))
        out.setdefault(clip.kind, []).append(clip)
    return out


def _filtered_noise(n: int, sos: np.ndarray, rng: np.random.Generator, sr: int) -> np.ndarray:
    """White noise through `sos`, generated in chunks so a 45 min bed never needs float64 at full length."""
    out = np.empty(n, np.float32)
    zi = np.zeros((sos.shape[0], 2))
    step = int(CHUNK_S * sr)
    for i in range(0, n, step):
        w = rng.standard_normal(min(step, n - i))
        y, zi = signal.sosfilt(sos, w, zi=zi)
        out[i:i + len(y)] = y
    return out


def _slow_am(n: int, sr: int, rng: np.random.Generator, depth_db: float, rate_hz: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    phase = rng.uniform(0, 2 * math.pi)
    return np.power(10.0, (depth_db / 20.0) * np.sin(2 * math.pi * rate_hz * t + phase)).astype(np.float32)


def hvac(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Broadband rumble falling about 6 dB per octave above 150 Hz, a faint blade-pass hum, slow drift."""
    sos = np.concatenate([signal.butter(2, 30, "highpass", fs=sr, output="sos"),
                          signal.butter(1, 150, "lowpass", fs=sr, output="sos"),
                          signal.butter(2, 0.45 * sr, "lowpass", fs=sr, output="sos")])
    x = _filtered_noise(n, sos, rng, sr)
    x += 0.15 * _filtered_noise(n, signal.butter(2, [800, 4000], "bandpass", fs=sr, output="sos"), rng, sr)
    f0 = rng.uniform(45, 110)
    t = np.arange(n, dtype=np.float32) / sr
    x += 0.05 * float(np.std(x)) * np.sin(2 * np.pi * f0 * t).astype(np.float32)
    x *= _slow_am(n, sr, rng, 1.0, rng.uniform(0.02, 0.1))
    return x


def projector_fan(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Mid-band whoosh plus a fan tone and two harmonics."""
    x = _filtered_noise(n, signal.butter(2, [250, 3500], "bandpass", fs=sr, output="sos"), rng, sr)
    f0 = rng.uniform(110, 260)
    t = np.arange(n, dtype=np.float32) / sr
    s = float(np.std(x))
    for h, a in ((1, 0.35), (2, 0.15), (3, 0.08)):
        x += a * s * np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
    x *= _slow_am(n, sr, rng, 0.5, rng.uniform(0.05, 0.2))
    return x


def keystroke(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * rng.uniform(0.02, 0.05))
    t = np.arange(n) / sr
    click = rng.standard_normal(n) * np.exp(-t / rng.uniform(0.003, 0.008))
    lo, hi = rng.uniform(1200, 2500), min(rng.uniform(5000, 7000), 0.45 * sr)
    click = signal.sosfilt(signal.butter(2, [lo, hi], "bandpass", fs=sr, output="sos"), click)
    thump = np.sin(2 * np.pi * rng.uniform(150, 300) * t) * np.exp(-t / 0.006)
    y = click / (np.abs(click).max() + 1e-9) + 0.4 * thump
    return (y * rng.uniform(0.5, 1.0)).astype(np.float32)


def keyboard(n: int, sr: int, rng: np.random.Generator, *, sessions_per_min: float = 2.0) -> np.ndarray:
    """Typing sessions (2-15 s, 4-8 keys a second) at random times, the first within about 15 s."""
    x = np.zeros(n, np.float32)
    dur = n / sr
    t = rng.exponential(15.0)
    while t < dur:
        length = rng.uniform(2.0, 15.0)
        rate = rng.uniform(4.0, 8.0)
        k = t
        while k < min(t + length, dur - 0.1):
            ks = keystroke(sr, rng)
            i = int(k * sr)
            x[i:i + len(ks)] += ks[: n - i]
            k += rng.exponential(1.0 / rate) + 0.03
        t += length + rng.exponential(60.0 / sessions_per_min)
    return x


def event(kind: str, sr: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "door":
        n = int(sr * rng.uniform(0.4, 0.8))
        t = np.arange(n) / sr
        thump = np.sin(2 * np.pi * rng.uniform(55, 110) * t) * np.exp(-t / 0.08)
        click = rng.standard_normal(n) * np.exp(-t / 0.004)
        click = signal.sosfilt(signal.butter(2, [1500, 0.45 * sr], "bandpass", fs=sr, output="sos"), click)
        y = thump + 0.5 * click / (np.abs(click).max() + 1e-9)
    elif kind == "chair":
        n = int(sr * rng.uniform(0.4, 1.5))
        t = np.arange(n) / sr
        y = signal.sosfilt(signal.butter(2, [rng.uniform(250, 600), rng.uniform(2000, 4000)], "bandpass", fs=sr,
                                         output="sos"), rng.standard_normal(n))
        y *= (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(8, 25) * t))) * np.sin(np.pi * t / t[-1])
    elif kind == "cup":
        n = int(sr * rng.uniform(0.25, 0.6))
        t = np.arange(n) / sr
        y = sum(rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t) * np.exp(-t / rng.uniform(0.05, 0.2))
                for f in rng.uniform(1800, min(6500, 0.45 * sr), size=4))
        y += 0.3 * rng.standard_normal(n) * np.exp(-t / 0.002)
    elif kind == "paper":
        n = int(sr * rng.uniform(0.5, 2.0))
        t = np.arange(n) / sr
        y = signal.sosfilt(signal.butter(2, [1000, min(7500, 0.45 * sr)], "bandpass", fs=sr, output="sos"),
                           rng.standard_normal(n))
        bursts = signal.sosfilt(signal.butter(1, 12, "lowpass", fs=sr, output="sos"), np.abs(rng.standard_normal(n)))
        y *= bursts / (bursts.max() + 1e-9) * np.sin(np.pi * t / t[-1])
    else:
        raise ValueError(f"unknown event kind {kind!r}")
    y = np.asarray(y, np.float64)
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def _library_audio(clip: LibraryClip, sr: int) -> np.ndarray:
    return Audio.load(clip.path).resample(sr).samples


def loop_to(x: np.ndarray, n: int, sr: int, xfade_s: float = 0.5) -> np.ndarray:
    """Repeat a clip to `n` samples with crossfades."""
    xf = min(int(xfade_s * sr), len(x) // 4)
    if len(x) >= n:
        return x[:n].copy()
    out = np.zeros(n, np.float32)
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32) if xf else np.zeros(0, np.float32)
    pos = 0
    while pos < n:
        seg = x.copy()
        if xf and pos:
            seg[:xf] *= ramp
        if xf:
            seg[-xf:] *= ramp[::-1]
        m = min(len(seg), n - pos)
        out[pos:pos + m] += seg[:m]
        pos += len(seg) - xf
    return out


def make(kind: str, n: int, sr: int, rng: np.random.Generator,
         library: dict[str, list[LibraryClip]]) -> tuple[np.ndarray, str]:
    """(mono signal, source) for a bed (`n` samples) or an event (its natural length; `n` ignored)."""
    clips = library.get(kind, [])
    if clips:
        clip = clips[int(rng.integers(len(clips)))]
        x = _library_audio(clip, sr)
        return (loop_to(x, n, sr) if kind in BEDS else x), f"library:{clip.id}"
    if kind == "hvac":
        return hvac(n, sr, rng), "procedural"
    if kind == "projector_fan":
        return projector_fan(n, sr, rng), "procedural"
    if kind == "keyboard":
        return keyboard(n, sr, rng), "procedural"
    return event(kind, sr, rng), "procedural"
