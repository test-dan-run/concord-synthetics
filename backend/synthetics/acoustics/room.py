"""Room simulation for stage 7: a pyroomacoustics shoebox built from meeting.json, its absorption calibrated
to the sampled RT60, and impulse responses per source pose.

- Every surface takes a frequency-dependent absorption profile from the pyroomacoustics database
  (`SimulationConfig.materials`). One factor `s` scales all of them in the Eyring sense,
  -ln(1 - a') = s * -ln(1 - a), so absorption never passes 1 and the relative colour of the surfaces (reflective
  glass, absorbing tile) is kept. `s` is chosen so that the mid-band RT60 (mean T20 of the 500 Hz and 1 kHz
  octaves), measured on simulated impulse responses at a few talker positions (median), matches
  `room.rt60_target`.
- pyroomacoustics supports source directivity only in the image-source method (ISM). A pose's impulse response
  is therefore the directive ISM part up to `ism_max_order`, plus, in rooms that use ray tracing, the late part of
  an omnidirectional source at the same position (hybrid ISM + ray tracing, minus omnidirectional ISM), scaled by
  the directivity's power factor. Rooms without ray tracing use ISM alone, to an order long enough for the T20 fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
from scipy import signal

from ..artifacts.meeting import MeetingSpec
from ..config.schema import SimulationConfig

pra.constants.set("num_threads", 1)  # sequential and reproducible
SPEED = 343.0
BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
PRA_WALLS = {"W": "west", "E": "east", "S": "south", "N": "north"}


@dataclass(frozen=True)
class Pose:
    """A source position, and the direction it faces (None: omnidirectional, e.g. a loudspeaker or noise)."""

    xyz: tuple[float, float, float]
    facing_deg: float | None = None

    def key(self) -> tuple:
        return (*(round(c, 3) for c in self.xyz), None if self.facing_deg is None else round(self.facing_deg, 1))


def surface_profiles(m: MeetingSpec, sim: SimulationConfig) -> dict[str, np.ndarray]:
    """pyroomacoustics surface name -> energy absorption per octave band (unscaled)."""
    table = pra.materials_absorption_table
    names = {PRA_WALLS[w]: sim.materials.walls[mat] for w, mat in m.room.walls.items()}
    names["floor"] = sim.materials.floor[m.room.floor]
    names["ceiling"] = sim.materials.ceiling[m.room.ceiling]
    out = {}
    for surface, name in names.items():
        c = np.asarray(table[name]["coeffs"], float)
        out[surface] = np.concatenate([c, np.full(len(BANDS) - len(c), c[-1])])  # some entries stop at 4 kHz
    return out


def surface_areas(dims: tuple[float, float, float]) -> dict[str, float]:
    L, W, H = dims
    return {"west": W * H, "east": W * H, "south": L * H, "north": L * H, "floor": L * W, "ceiling": L * W}


def scaled(profile: np.ndarray, s: float) -> np.ndarray:
    return 1.0 - np.power(1.0 - np.clip(profile, 0.0, 0.999), s)


def eyring_scale(dims: tuple[float, float, float], profiles: dict[str, np.ndarray], rt60: float) -> float:
    """Scale that gives `rt60` by Eyring's formula at mid band, the starting point of calibration."""
    area = surface_areas(dims)
    mid = [BANDS.index(500), BANDS.index(1000)]
    a1 = sum(area[s] * -math.log(1.0 - float(np.mean(p[mid]))) for s, p in profiles.items())
    volume = dims[0] * dims[1] * dims[2]
    return 0.161 * volume / rt60 / a1


def octave_rt60(h: np.ndarray, fs: int, centre: float) -> float:
    """T20 of one octave band (Schroeder backward integration, fit from -5 to -25 dB), NaN if too short."""
    lo, hi = centre / math.sqrt(2), min(centre * math.sqrt(2), 0.45 * fs)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, h.astype(np.float64))
    edc = np.cumsum((x**2)[::-1])[::-1]
    if edc[0] <= 0:
        return math.nan
    db = 10 * np.log10(np.maximum(edc / edc[0], 1e-30))
    i5, i25 = np.argmax(db <= -5.0), np.argmax(db <= -25.0)
    if i25 <= i5:
        return math.nan
    t = np.arange(i5, i25) / fs
    slope = np.polyfit(t, db[i5:i25], 1)[0]
    return float(-60.0 / slope) if slope < 0 else math.nan


def rt60_mid(h: np.ndarray, fs: int) -> float:
    vals = [octave_rt60(h, fs, c) for c in (500.0, 1000.0)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else math.nan


class RoomModel:
    def __init__(self, m: MeetingSpec, sim: SimulationConfig, fs: int, *, seed: int):
        self.m = m
        self.sim = sim
        self.fs = fs
        self.seed = seed
        self.dims = (m.room.length, m.room.width, m.room.height)
        self.profiles = surface_profiles(m, sim)
        # simulated decays in these rooms (absorbing floor and ceiling, reflective walls) run about twice Eyring's
        # estimate, so calibration starts from twice its scale
        self.scale = 2.0 * eyring_scale(self.dims, self.profiles, m.room.rt60_target)
        self.ray_tracing = m.room_class in sim.ray_tracing
        self.mics = mic_positions(m)  # (3, n_mics)
        self.directivity_p = sim.directivity_p
        self.power_factor = math.sqrt(self.directivity_p**2 + (1 - self.directivity_p) ** 2 / 3)

    @property
    def ism_order(self) -> int:
        if self.ray_tracing:
            return self.sim.ism_max_order
        # ISM alone: long enough to reach the -25 dB point of the T20 fit (about 0.45 T60), with some margin
        reach = SPEED * 0.6 * self.m.room.rt60_target
        return int(min(self.sim.ism_order_cap, max(self.sim.ism_max_order, math.ceil(reach / min(self.dims)))))

    def materials(self) -> dict[str, pra.Material]:
        out = {}
        for surface, prof in self.profiles.items():
            coeffs = np.clip(scaled(prof, self.scale), self.sim.absorption_min, self.sim.absorption_max)
            out[surface] = pra.Material({"description": surface, "coeffs": [float(c) for c in coeffs],
                                         "center_freqs": BANDS}, self.sim.scattering)
        return out

    def _room(self, *, order: int, ray_tracing: bool) -> pra.ShoeBox:
        room = pra.ShoeBox(list(self.dims), fs=self.fs, materials=self.materials(), max_order=order,
                           ray_tracing=ray_tracing, air_absorption=True)
        if ray_tracing:
            room.set_ray_tracing(n_rays=self.sim.rt_n_rays, receiver_radius=self.sim.rt_receiver_radius)
        room.add_microphone_array(self.mics)
        return room

    def _directivity(self, facing_deg: float) -> CardioidFamily:
        return CardioidFamily(orientation=DirectionVector(azimuth=facing_deg, colatitude=90.0, degrees=True),
                              p=self.directivity_p)

    def _compute(self, room: pra.ShoeBox) -> list[list[np.ndarray]]:
        # both generators: wall scattering in C++, and the numpy one that turns ray-tracing histograms into noise
        pra.random.seed(numpy=self.seed, libroom=self.seed)
        room.compute_rir()
        return room.rir  # [mic][source]

    def rirs(self, poses: list[Pose]) -> dict[tuple, np.ndarray]:
        """Pose key -> (n_mics, n) float32 impulse responses, truncated after the reverberant tail."""
        uniq = list({p.key(): p for p in poses}.values())
        if not uniq:
            return {}
        early_room = self._room(order=self.ism_order, ray_tracing=False)
        for p in uniq:
            early_room.add_source(list(p.xyz), directivity=None if p.facing_deg is None else self._directivity(p.facing_deg))
        early = self._compute(early_room)
        late: dict[tuple, list[np.ndarray]] = {}
        if self.ray_tracing:
            positions = list({tuple(round(c, 3) for c in p.xyz): p.xyz for p in uniq}.items())
            hyb_room = self._room(order=self.ism_order, ray_tracing=True)
            ism_room = self._room(order=self.ism_order, ray_tracing=False)
            for _, xyz in positions:
                hyb_room.add_source(list(xyz))
                ism_room.add_source(list(xyz))
            hyb, ism = self._compute(hyb_room), self._compute(ism_room)
            for j, (pk, _) in enumerate(positions):
                late[pk] = [_minus(hyb[mi][j], ism[mi][j]) for mi in range(self.mics.shape[1])]
        n_keep = int(round((self.sim.rir_tail_factor * self.m.room.rt60_target + 0.05) * self.fs))
        out = {}
        for j, p in enumerate(uniq):
            chans = []
            for mi in range(self.mics.shape[1]):
                h = early[mi][j].astype(np.float64)
                if self.ray_tracing:
                    lt = late[tuple(round(c, 3) for c in p.xyz)][mi]
                    gain = self.power_factor if p.facing_deg is not None else 1.0
                    h = _plus(h, gain * lt)
                chans.append(h)
            n = min(n_keep, max(len(h) for h in chans))
            arr = np.zeros((len(chans), n), np.float32)
            for mi, h in enumerate(chans):
                arr[mi, :min(n, len(h))] = h[:n]
            fade = min(n // 10, int(0.05 * self.fs))
            if fade:
                arr[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            out[p.key()] = arr
        return out

    def measure(self, probes: list[Pose]) -> float:
        """Median mid-band RT60 at mic 0 over `probes`."""
        rirs = self.rirs(probes)
        vals = [rt60_mid(h[0], self.fs) for h in rirs.values()]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.median(vals)) if vals else math.nan

    def calibrate(self, probes: list[Pose]) -> tuple[float, list[float]]:
        """Rescale absorption until the probes' median mid-band RT60 is within tolerance of the target (secant
        steps on log RT60 against log scale). Returns (measured RT60, history)."""
        target = self.m.room.rt60_target
        history: list[float] = []
        best = (math.inf, self.scale, math.nan)
        prev: tuple[float, float] | None = None
        for _ in range(self.sim.calibration_iters):
            measured = self.measure(probes)
            history.append(round(measured, 4))
            if math.isnan(measured):
                break
            err = abs(measured - target) / target
            if err < best[0]:
                best = (err, self.scale, measured)
            if err <= self.sim.rt60_calibration_tol:
                break
            slope = -1.0  # Eyring: RT60 inverse to the scale
            if prev is not None and prev[0] != self.scale and prev[1] != measured:
                slope = float(np.clip(math.log(measured / prev[1]) / math.log(self.scale / prev[0]), -2.0, -0.6))
            prev = (self.scale, measured)
            self.scale *= float(np.clip((target / measured) ** (1.0 / slope), 1 / 3, 3.0))
        _, self.scale, measured = best
        return measured, history


def _minus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[:len(a)] += a
    out[:len(b)] -= b
    return out


def _plus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def mic_positions(m: MeetingSpec) -> np.ndarray:
    base = np.asarray(m.mic.position, float)
    geom = m.mic.array_geometry or [(0.0, 0.0, 0.0)]
    return np.stack([base + np.asarray(g, float) for g in geom], axis=1)
