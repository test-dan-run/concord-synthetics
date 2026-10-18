"""Stage 7: acoustic simulation (CPU). Writes mixture.wav, mixture_mc.wav (array mics) and render.json.

1. Room from meeting.json; absorption calibrated so the talkers' mid-band RT60 matches the sampled target
   (`acoustics/room.py`).
2. Per segment, the talker's pose (fixed seats with per-turn facing, presenter drift, the speakerphone for remote
   participants; `acoustics/scene.py`) picks an impulse response, and the segment's slice of the dry track is
   convolved with it into the room signal. Levels are set at the source only, so far talkers (wall seats) stay
   quieter and more reverberant at the mic, and the laptop's near talker stays loud.
3. Remote participants first pass through their far-end path (band limit, Opus); in hybrid meetings the echo
   canceller's artifacts are applied (`acoustics/device.py`).
4. Noise beds (HVAC, projector fan) at the sampled SNR; keyboard typing and events relative to speech
   (`acoustics/noise.py`).
5. Device chain on every channel: band limit, AGC, clipping -> mixture_mc.wav. The single channel is the mean of
   the channels, then the codec -> mixture.wav.

Memory stays at about two (channels x samples) float32 buffers plus one channel of temporaries: under 2 GB for a
45 min meeting on a 4-capsule array.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel
from scipy import signal

from ..acoustics import device, noise
from ..acoustics.room import Pose, RoomModel, rt60_mid
from ..acoustics.scene import segment_poses
from ..artifacts.io import read_model, write_model
from ..artifacts.labels import AcousticRender, NoiseRender, Timeline
from ..artifacts.meeting import MeetingSpec
from ..artifacts.paths import MeetingPaths
from ..config.schema import Phase, RunConfig, SimulationConfig, StageId
from .base import StageContext, StageResult


def db(x: float) -> float:
    return 10 * math.log10(max(x, 1e-20))


def active_power(x: np.ndarray, sr: int, floor_db: float = 30.0) -> float:
    """Mean power over the frames within `floor_db` of the loudest one (speech-active level of a clip)."""
    lev = device.frame_rms_db(x, sr, 0.02)
    if not len(lev):
        return float(np.mean(x.astype(np.float64) ** 2)) if len(x) else 0.0
    keep = lev > lev.max() - floor_db
    return float(np.mean(np.power(10.0, lev[keep] / 10)))


CONV_CHUNK = 1 << 20  # samples (about a minute at 16 kHz): long signals are convolved piece by piece


def add_convolved(buf: np.ndarray, x: np.ndarray, rir: np.ndarray, start: int) -> None:
    """buf[:, start:] += x * rir (per channel), clipped to the buffer. Overlap-add in chunks, one channel at a time,
    so a 45 min noise bed never needs a full-length temporary per channel."""
    total = buf.shape[1]
    for i in range(0, len(x), CONV_CHUNK):
        s0 = start + i
        if s0 >= total:
            break
        piece = x[i:i + CONV_CHUNK].astype(np.float32, copy=False)
        for ch in range(buf.shape[0]):
            y = signal.oaconvolve(piece, rir[ch])
            n = min(len(y), total - s0)
            buf[ch, s0:s0 + n] += y[:n]


def event_positions(m: MeetingSpec, kind: str, rng: np.random.Generator) -> tuple[float, float, float]:
    L, W = m.room.length, m.room.width
    seated = [p for p in m.people if p.position_type in ("table", "wall") and p.mouth]
    if kind == "door":
        wall = ("N", "S", "E")[int(rng.integers(3))]
        if wall == "E":
            return (L - 0.3, float(rng.uniform(0.5, W - 0.5)), 1.0)
        return (float(rng.uniform(1.0, L - 0.5)), 0.3 if wall == "S" else W - 0.3, 1.0)
    if kind == "chair" and seated:
        p = seated[int(rng.integers(len(seated)))]
        return (p.mouth[0], p.mouth[1], 0.45)
    slot = m.table.slots[int(rng.integers(len(m.table.slots)))]
    return (slot.edge_xy[0], slot.edge_xy[1], m.table.height + 0.05)


def plan_noise(m: MeetingSpec, duration: float, sim: SimulationConfig, rng: np.random.Generator) -> list[NoiseRender]:
    """Beds and keyboard from meeting.json's noise sources, events at random times; levels drawn here."""
    out = []
    centre = (m.room.length / 2, m.room.width / 2, m.room.height - 0.05)
    for src in m.acoustics.noise_sources:
        pos = tuple(src.position) if src.position else centre
        if src.kind == "hvac":
            level = 0.0
        elif src.kind == "projector_fan":
            level = sim.bed_balance_db.sample(rng)
        else:
            level = sim.keyboard_level_db.sample(rng)
        out.append(NoiseRender(kind=src.kind, source="", position=pos, level_db=round(level, 2)))
    for kind, count in sorted(m.acoustics.events.items()):
        for _ in range(count):
            t = float(rng.uniform(1.0, max(1.5, duration - 2.0)))
            out.append(NoiseRender(kind=kind, source="", position=event_positions(m, kind, rng),
                                   level_db=round(sim.event_level_db.sample(rng), 2), start=round(t, 3)))
    return out


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """(channels, n) or (n,) float32 -> PCM16 WAV, written in blocks to a temp file that is then renamed (atomic,
    and no full-size copy in memory)."""
    x2 = np.atleast_2d(x)
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        with sf.SoundFile(tmp, "w", samplerate=sr, channels=x2.shape[0], subtype="PCM_16", format="WAV") as f:
            step = 1 << 18
            for i in range(0, x2.shape[1], step):
                f.write(np.ascontiguousarray(x2[:, i:i + step].T))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def mean_power(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean power of the channel mean of `x` (channels, n), over `mask`, in chunks."""
    total, count = 0.0, 0
    for i in range(0, x.shape[1], CONV_CHUNK):
        d = x[:, i:i + CONV_CHUNK].mean(axis=0, dtype=np.float64)
        if mask is not None:
            d = d[mask[i:i + CONV_CHUNK]]
        total += float(np.sum(d * d))
        count += len(d)
    return total / count if count else 0.0


def render(paths: MeetingPaths, m: MeetingSpec, tl: Timeline, cfg: RunConfig, rng: np.random.Generator,
           seed: int) -> AcousticRender:
    sim, sr = cfg.simulation, cfg.sample_rate
    n_total = int(round(tl.stats.duration * sr))
    model = RoomModel(m, sim, sr, seed=seed)

    # 1. calibrate absorption on talkers' first poses (as the RT60 check measures them), then every impulse
    #    response the meeting needs
    poses = segment_poses(m, tl, sim, rng)
    first_pose: dict[str, Pose] = {}
    for s in sorted(tl.segments, key=lambda s: s.turn_id):
        if m.person(s.speaker_id).position_type != "remote":
            first_pose.setdefault(s.speaker_id, poses[s.turn_id])
    talkers = [first_pose[k] for k in sorted(first_pose)]
    picks = np.linspace(0, len(talkers) - 1, min(sim.calibration_probes, len(talkers))).round().astype(int)
    _, calibration = model.calibrate([talkers[j] for j in sorted(set(picks.tolist()))])
    noise_plan = plan_noise(m, tl.stats.duration, sim, rng)
    noise_poses = {i: Pose(tuple(nz.position), None) for i, nz in enumerate(noise_plan)}
    rirs = model.rirs(list(poses.values()) + list(noise_poses.values()))
    n_mics = model.mics.shape[1]
    per_speaker = {spk: round(rt60_mid(rirs[p.key()][0], sr), 4) for spk, p in sorted(first_pose.items())}
    finite = [v for v in per_speaker.values() if not math.isnan(v)]
    rt60 = float(np.median(finite)) if finite else math.nan

    # 2-3. speech, local talkers and remote participants in separate buffers (the canceller treats them apart)
    remote_ids = {p.speaker_id for p in m.people if p.position_type == "remote"}
    local = np.zeros((n_mics, n_total), np.float32)
    remote = np.zeros((n_mics, n_total), np.float32) if remote_ids else None
    spk_db = {p.speaker_id: rng.normal(0.0, sim.speaker_level_sd_db)
              + (sim.presenter_offset_db if p.position_type == "presenter" else 0.0) for p in m.people}
    remote_info: dict[str, dict[str, float]] = {}
    by_spk: dict[str, list] = {}
    for s in tl.segments:
        by_spk.setdefault(s.speaker_id, []).append(s)
    for spk in sorted(by_spk):
        segs = by_spk[spk]
        track_path = paths.track(spk)
        whole = None
        if spk in remote_ids:
            whole, _ = sf.read(str(track_path), dtype="float32")
            whole, remote_info[spk] = device.far_end(whole, sr, rng, narrowband_p=sim.far_end_narrowband_p,
                                                     kbps=(sim.far_end_kbps.lo, sim.far_end_kbps.hi))
        target = remote if spk in remote_ids else local
        for s in segs:
            s0, n = int(round(s.start * sr)), int(round((s.end - s.start) * sr))
            if whole is not None:
                x = whole[s0:s0 + n]
            else:
                x, _ = sf.read(str(track_path), start=s0, stop=s0 + n, dtype="float32")
            p = active_power(x, sr)
            if p <= 0:
                continue
            lvl = sim.speech_level_db + spk_db[spk] + rng.normal(0.0, sim.utterance_level_sd_db)
            if s.kind == "backchannel":
                lvl += sim.backchannel_offset_db
            gain = 10 ** ((lvl - db(p)) / 20)
            add_convolved(target, x * gain, rirs[poses[s.turn_id].key()], s0)
        del whole

    dev: dict = {}
    if remote is not None:
        if m.device_chain.aec_artifacts:
            spans = [(s.start, s.end) for s in tl.segments if s.speaker_id in remote_ids]
            dev.update(device.aec_artifacts(local, remote, sr, spans, rng, duck_db=(sim.aec_duck_db.lo, sim.aec_duck_db.hi),
                                            onset_s=(sim.aec_onset_s.lo, sim.aec_onset_s.hi)))
        local += remote
        del remote
    speech = local  # (n_mics, n): from here on the buffer that becomes the mixture

    voiced = device.activity_mask(n_total, sr, [(s.start, s.end) for s in tl.segments])
    p_speech = mean_power(speech, voiced)

    # 4. noise: beds scaled to the SNR together, then keyboard and events relative to speech
    library = noise.load_library(_library_root(sim))
    noise_buf = np.zeros((n_mics, n_total), np.float32)
    for i, nz in enumerate(noise_plan):
        if nz.kind not in noise.BEDS:
            continue
        x, nz.source = noise.make(nz.kind, n_total, sr, rng, library)
        x *= 10 ** (nz.level_db / 20) / (np.std(x) + 1e-12)
        add_convolved(noise_buf, x, rirs[noise_poses[i].key()], 0)
        del x
    p_bed = mean_power(noise_buf)
    snr_target = m.acoustics.snr_db
    snr = math.inf
    if p_bed > 0 and p_speech > 0:
        bed_gain = math.sqrt(p_speech / p_bed / 10 ** (snr_target / 10))
        noise_buf *= bed_gain
        snr = db(p_speech) - db(p_bed * bed_gain**2)
    for i, nz in enumerate(noise_plan):
        if nz.kind in noise.BEDS:
            continue
        if nz.kind == "keyboard":
            x, nz.source = noise.make("keyboard", n_total, sr, rng, library)
            s0 = 0
        else:
            x, nz.source = noise.make(nz.kind, 0, sr, rng, library)
            s0 = int(round(nz.start * sr))
            nz.end = round(nz.start + len(x) / sr, 3)
        h = rirs[noise_poses[i].key()]
        ref = np.zeros((1, len(x) + h.shape[1] - 1), np.float32)
        add_convolved(ref, x, h[:1], 0)
        p = active_power(ref[0], sr)
        del ref
        if p <= 0:
            continue
        x *= math.sqrt(p_speech / p * 10 ** (nz.level_db / 10))
        add_convolved(noise_buf, x, h, s0)
        del x
    speech += noise_buf
    del noise_buf
    mix = speech

    # 5. device chain
    chain = m.device_chain
    if chain.band_limit_hz:
        device.band_limit(mix, sr, chain.band_limit_hz)
        dev["band_limit_hz"] = chain.band_limit_hz
    fixed = 10 ** ((sim.output_level_db - db(p_speech)) / 20) if p_speech > 0 else 1.0
    mix *= fixed
    if chain.agc:
        g = device.agc_gain(mix.mean(axis=0, dtype=np.float32), sr, target_db=sim.agc_target_db, max_gain_db=sim.agc_max_gain_db, attack_s=sim.agc_attack_s, release_s=sim.agc_release_s,
                            gate_db=sim.agc_gate_db)
        dev["agc_gain_db_min"] = round(db(float(g.min()) ** 2), 2)
        dev["agc_gain_db_max"] = round(db(float(g.max()) ** 2), 2)
        mix *= g
        del g
    peak = float(np.abs(mix).max()) if mix.size else 0.0
    if chain.clipping_level and peak > 0:
        threshold = chain.clipping_level * peak
        dev["clipped_share"] = round(device.hard_clip(mix, threshold), 6)
        mix *= 0.99 / threshold  # the clipped peaks land just under full scale
    elif peak > 0.99:
        mix *= 0.99 / peak  # headroom only: the recording is never clipped unless the chain says so
        dev["headroom_gain_db"] = round(db((0.99 / peak) ** 2), 2)
    files = {"mixture": "mixture.wav"}
    if n_mics > 1:
        write_wav(paths.mixture_mc, mix, sr)
        files["mixture_mc"] = "mixture_mc.wav"
    mono = mix.mean(axis=0, dtype=np.float32) if n_mics > 1 else mix[0].copy()
    del mix, speech
    if chain.codec_kbps:
        mono = device.opus(mono, sr, chain.codec_kbps)
        dev["codec_kbps"] = chain.codec_kbps
        peak = float(np.abs(mono).max())
        if peak > 0.999:
            mono *= 0.999 / peak
    write_wav(paths.mixture, mono, sr)

    err = (rt60 - m.room.rt60_target) / m.room.rt60_target if not math.isnan(rt60) else math.nan
    return AcousticRender(meeting_id=m.meeting_id, sample_rate=sr, channels=n_mics, duration=tl.stats.duration,
                          files=files, rt60_target=m.room.rt60_target, rt60_measured=round(rt60, 4),
                          rt60_error=round(err, 4), rt60_per_speaker=per_speaker, absorption_scale=round(model.scale, 4),
                          calibration=calibration, ism_order=model.ism_order, ray_tracing=model.ray_tracing,
                          n_impulse_responses=len(rirs), snr_target_db=snr_target, snr_db=round(snr, 2),
                          noise=noise_plan, remote=remote_info, device=dev)


def _library_root(sim: SimulationConfig):
    if sim.noise_library is None:
        return None
    from ..orchestrator.run import app_path

    return app_path(sim.noise_library)


# ---------------------------------------------------------------- the stage


class Stage7Config(BaseModel):
    simulation: SimulationConfig
    sample_rate: int


class AcousticStage:
    id = StageId.ACOUSTIC
    name = "Acoustics"
    phase = Phase.CPU
    roles = ()
    reads = ("meeting", "timeline", "tracks_dir")
    writes = ("mixture", "render")
    version = "1"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        return Stage7Config(simulation=cfg.simulation, sample_rate=cfg.sample_rate)

    def run(self, ctx: StageContext) -> StageResult:
        from .runner import meeting_seed

        cfg = ctx.cfg
        m = read_model(ctx.paths.meeting, MeetingSpec)
        tl = read_model(ctx.paths.timeline, Timeline)
        m_seed = meeting_seed(cfg.seed, m.meeting_id)
        seq = np.random.SeedSequence([cfg.seed, m_seed, int(self.id)])
        rng = np.random.default_rng(seq)
        seed = int(seq.generate_state(1, dtype=np.uint32)[0])
        if m.mic.channels == 1:
            ctx.paths.mixture_mc.unlink(missing_ok=True)  # left over from an earlier render of another layout
        t0 = time.monotonic()
        rep = render(ctx.paths, m, tl, cfg, rng, seed)
        write_model(ctx.paths.render, rep)
        metrics = {"rt60_target": rep.rt60_target, "rt60_measured": rep.rt60_measured, "rt60_error": rep.rt60_error,
                   "snr_db": rep.snr_db, "channels": rep.channels, "impulse_responses": rep.n_impulse_responses,
                   "render_s": round(time.monotonic() - t0, 2)}
        detail = (f"RT60 {rep.rt60_measured:.2f} s (target {rep.rt60_target:.2f}) · SNR {rep.snr_db:.0f} dB · "
                  f"{rep.channels} ch")
        if math.isnan(rep.rt60_error) or abs(rep.rt60_error) > cfg.simulation.rt60_tolerance:
            return StageResult(state="fail", detail=f"RT60 off target: {detail}", metrics=metrics)
        return StageResult(state="pass", detail=detail, metrics=metrics)
