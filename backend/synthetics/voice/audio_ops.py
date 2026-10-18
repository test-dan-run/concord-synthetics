"""Small audio operations for stage 4: silence trimming and joining the pieces of one turn."""

from __future__ import annotations

import numpy as np

from ..backends.base import Audio

FRAME_S = 0.01


def trim(audio: Audio, *, pad_s: float, below_db: float = 40.0) -> Audio:
    """Cut leading and trailing silence, keeping at most `pad_s` before the first and after the last frame whose
    peak is within `below_db` of the clip's peak. An all-silent clip comes back unchanged."""
    x = audio.samples
    if x.size == 0:
        return audio
    peak = float(np.abs(x).max())
    if peak <= 0:
        return audio
    hop = max(1, int(FRAME_S * audio.sr))
    n = -(-len(x) // hop)
    frames = np.abs(np.pad(x, (0, n * hop - len(x)))).reshape(n, hop).max(axis=1)
    loud = np.flatnonzero(frames >= peak * 10 ** (-below_db / 20))
    pad = int(round(pad_s * audio.sr))
    start = max(0, loud[0] * hop - pad)
    end = min(len(x), (loud[-1] + 1) * hop + pad)
    return Audio(x[start:end], audio.sr)


def join(pieces: list[Audio], *, gap_s: float) -> Audio:
    """Concatenate same-rate pieces with `gap_s` of silence between them."""
    if not pieces:
        raise ValueError("nothing to join")
    sr = pieces[0].sr
    if any(p.sr != sr for p in pieces):
        raise ValueError("pieces must share one sample rate")
    gap = np.zeros(int(round(gap_s * sr)), np.float32)
    parts: list[np.ndarray] = []
    for i, p in enumerate(pieces):
        if i:
            parts.append(gap)
        parts.append(p.samples)
    return Audio(np.concatenate(parts), sr)


def rms(audio: Audio) -> float:
    return float(np.sqrt(np.mean(audio.samples.astype(np.float64) ** 2))) if audio.samples.size else 0.0


def match_level(audio: Audio, target_rms: float, *, offset_db: float = -6.0) -> Audio:
    """Scale to `target_rms` (+ offset_db), never above full scale."""
    r = rms(audio)
    if r <= 0 or target_rms <= 0:
        return audio
    g = target_rms * 10 ** (offset_db / 20) / r
    peak = float(np.abs(audio.samples).max()) * g
    if peak > 0.99:
        g *= 0.99 / peak
    return Audio(audio.samples * g, audio.sr)
