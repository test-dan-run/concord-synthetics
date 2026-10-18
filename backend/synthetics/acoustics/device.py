"""The recording device chain for stage 7, and the far-end path of remote participants.

Order on the room signal: band-limiting (mic and converter), AGC (slow, linked across channels), clipping, then
the Opus codec. Hybrid meetings add echo-cancellation artifacts before the chain: while a remote participant is
heard through the speakerphone, the canceller's suppressor ducks local talkers (double-talk) and clips the onset
of each remote turn while it converges.

Opus runs through the `ffmpeg` binary (libopus). Its output is realigned to the input by cross-correlation, so the
labels keep their timing.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
from scipy import signal

CHUNK = 1 << 20


def band_limit(x: np.ndarray, sr: int, hz: float, highpass_hz: float = 80.0) -> None:
    """In place, per channel: high-pass at `highpass_hz` and low-pass at `hz` (8th-order Butterworth)."""
    parts = [signal.butter(2, highpass_hz, "highpass", fs=sr, output="sos")]
    if hz < 0.49 * sr:
        parts.append(signal.butter(8, hz, "lowpass", fs=sr, output="sos"))
    sos = np.concatenate(parts)
    for ch in np.atleast_2d(x):
        zi = np.zeros((sos.shape[0], 2))
        for i in range(0, len(ch), CHUNK):
            y, zi = signal.sosfilt(sos, ch[i:i + CHUNK], zi=zi)
            ch[i:i + CHUNK] = y


def frame_rms_db(x: np.ndarray, sr: int, frame_s: float = 0.05) -> np.ndarray:
    hop = int(frame_s * sr)
    n = len(x) // hop
    fr = x[: n * hop].astype(np.float32).reshape(n, hop)
    return 10 * np.log10(np.mean(fr**2, axis=1) + 1e-12)


def agc_gain(ref: np.ndarray, sr: int, *, target_db: float, max_gain_db: float, attack_s: float, release_s: float,
             gate_db: float, frame_s: float = 0.05) -> np.ndarray:
    """Per-sample linear gain from a slow AGC on `ref` (the derived channel). The gain holds below the gate, so
    silences are not pumped up; it falls with `attack_s` and rises with `release_s`."""
    lev = frame_rms_db(ref, sr, frame_s)
    g = np.empty(len(lev))
    cur = 0.0
    a_att, a_rel = np.exp(-frame_s / attack_s), np.exp(-frame_s / release_s)
    for i, level in enumerate(lev):
        if level > gate_db:
            want = float(np.clip(target_db - level, -max_gain_db, max_gain_db))
            a = a_att if want < cur else a_rel
            cur = a * cur + (1 - a) * want
        g[i] = cur
    hop = int(frame_s * sr)
    t_frames = (np.arange(len(g)) + 0.5) * hop
    gain_db = np.interp(np.arange(len(ref)), t_frames, g) if len(g) else np.zeros(len(ref))
    return np.power(10.0, gain_db / 20.0).astype(np.float32)


def hard_clip(x: np.ndarray, threshold: float) -> float:
    """In place; returns the share of samples clipped."""
    n_clip = int(np.count_nonzero(np.abs(x) > threshold))
    np.clip(x, -threshold, threshold, out=x)
    return n_clip / x.size if x.size else 0.0


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def opus(x: np.ndarray, sr: int, kbps: int, *, application: str = "voip") -> np.ndarray:
    """Encode and decode one channel with libopus; output aligned to and as long as the input."""
    if not have_ffmpeg():
        raise RuntimeError("the Opus codec needs the ffmpeg binary (with libopus) on PATH")
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    enc = subprocess.run(base + ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-c:a", "libopus",
                                 "-b:a", f"{kbps}k", "-application", application, "-f", "ogg", "pipe:1"],
                         input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True, check=True)
    dec = subprocess.run(base + ["-i", "pipe:0", "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"],
                         input=enc.stdout, capture_output=True, check=True)
    y = np.frombuffer(dec.stdout, np.float32)
    lag = codec_lag(x, y, sr)
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag, np.float32), y])
    out = np.zeros(len(x), np.float32)
    out[: min(len(x), len(y))] = y[: len(x)]
    return out


def codec_lag(x: np.ndarray, y: np.ndarray, sr: int, max_lag_s: float = 0.1) -> int:
    """Lag of `y` behind `x` in samples, from the cross-correlation of their loudest 20 s."""
    win = min(len(x), len(y), int(20 * sr))
    if win < sr // 10:
        return 0
    hop = int(sr * 0.5)
    energies = [float(np.sum(x[i:i + win] ** 2)) for i in range(0, max(1, min(len(x), len(y)) - win + 1), hop)]
    i0 = int(np.argmax(energies)) * hop
    a, b = x[i0:i0 + win].astype(np.float64), y[i0:i0 + win].astype(np.float64)
    max_lag = int(max_lag_s * sr)
    c = signal.correlate(b, a, mode="full", method="fft")
    mid = len(a) - 1
    lo, hi = max(0, mid - max_lag), min(len(c), mid + max_lag + 1)
    return int(np.argmax(c[lo:hi]) + lo - mid)


def far_end(x: np.ndarray, sr: int, rng: np.random.Generator, *, narrowband_p: float,
            kbps: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """A remote participant's uplink before the room speakerphone plays it: band-limited (narrowband with
    `narrowband_p`, else wideband) and Opus-coded."""
    narrow = bool(rng.random() < narrowband_p)
    hz = 3400.0 if narrow else 7000.0
    y = x.astype(np.float32, copy=True)
    band_limit(y, sr, hz, highpass_hz=300.0 if narrow else 100.0)
    rate = int(rng.integers(kbps[0], kbps[1] + 1))
    y = opus(y, sr, rate)
    return y, {"band_hz": hz, "codec_kbps": rate}


def activity_mask(n: int, sr: int, spans: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(n, bool)
    for s, e in spans:
        m[max(0, int(s * sr)): min(n, int(e * sr))] = True
    return m


def smooth_gain(target: np.ndarray, sr: int, attack_s: float, release_s: float, frame_s: float = 0.005) -> np.ndarray:
    """Per-sample gain following `target` (linear, per sample) with separate fall and rise times."""
    hop = max(1, int(frame_s * sr))
    t = target[::hop].astype(np.float64)
    g = np.empty_like(t)
    cur = 1.0
    a_att, a_rel = np.exp(-frame_s / attack_s), np.exp(-frame_s / release_s)
    for i, v in enumerate(t):
        a = a_att if v < cur else a_rel
        cur = a * cur + (1 - a) * v
        g[i] = cur
    return np.repeat(g, hop)[: len(target)].astype(np.float32)


def aec_artifacts(local: np.ndarray, remote: np.ndarray, sr: int, remote_spans: list[tuple[float, float]],
                  rng: np.random.Generator, *, duck_db: tuple[float, float], onset_s: tuple[float, float]) -> dict:
    """In place. Double-talk suppression ducks `local` while a remote talker is active; the first moments of each
    remote turn are attenuated while the canceller converges."""
    n = local.shape[-1]
    duck = float(rng.uniform(*duck_db))
    active = activity_mask(n, sr, remote_spans)
    g_local = smooth_gain(np.where(active, 10 ** (-duck / 20), 1.0), sr, attack_s=0.01, release_s=0.15)
    local *= g_local
    onset_target = np.ones(n, np.float32)
    clipped = 0.0
    for s, _ in remote_spans:
        d = float(rng.uniform(*onset_s))
        i0, i1 = int(s * sr), min(n, int((s + d) * sr))
        if i1 > i0:
            onset_target[i0:i1] = np.linspace(0.1, 1.0, i1 - i0, dtype=np.float32)
            clipped += d
    remote *= onset_target
    return {"double_talk_duck_db": round(duck, 2), "onset_attenuated_s": round(clipped, 3)}
