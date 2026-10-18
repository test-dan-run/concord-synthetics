"""Stage 7 building blocks: room calibration, poses, noise and the device chain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from synthetics.acoustics import device, noise
from synthetics.acoustics.room import Pose, RoomModel, rt60_mid, scaled
from synthetics.acoustics.scene import segment_poses
from synthetics.artifacts.labels import Segment, Timeline, TimelineStats
from synthetics.stages.s1_sample import sample_meeting

SR = 16000


def _meeting(cfg, splits, room_class: str, seed: int, **people):
    rng = np.random.default_rng(seed)
    from conftest import make_cfg

    c = make_cfg(**{"room_classes.ratio": {"dist": "categorical", "weights": {room_class: 1.0}}, **people})
    return sample_meeting(c, splits, "m-000001", seed, rng), c


@pytest.mark.parametrize("room_class,seed", [("small", 1), ("large", 2)])
def test_calibrated_rt60_matches_the_target(cfg, splits, room_class, seed):
    m, c = _meeting(cfg, splits, room_class, seed)
    model = RoomModel(m, c.simulation, SR, seed=seed)
    talkers = [Pose(tuple(p.mouth), p.facing_deg) for p in m.people if p.mouth][:4]
    measured, history = model.calibrate(talkers)
    assert abs(measured - m.room.rt60_target) / m.room.rt60_target <= c.simulation.rt60_calibration_tol + 0.05, history
    assert model.ray_tracing == (room_class == "large")


def test_absorption_scaling_stays_below_one_and_keeps_order():
    prof = np.array([0.02, 0.05, 0.3, 0.8])
    for s in (0.5, 1.0, 4.0, 20.0):
        out = scaled(prof, s)
        assert np.all(out < 1.0) and np.all(np.diff(out) > 0)
    assert np.allclose(scaled(prof, 1.0), prof)


def test_rt60_measure_recovers_a_synthetic_decay():
    for t60 in (0.3, 0.6, 0.9):
        t = np.arange(int(1.5 * t60 * SR)) / SR
        vals = [rt60_mid(np.random.default_rng(s).standard_normal(len(t)) * 10 ** (-3 * t / t60), SR) for s in range(8)]
        assert np.mean(vals) == pytest.approx(t60, rel=0.03)  # one noise realisation scatters by about 5%


def test_directivity_lowers_the_level_behind_the_talker(cfg, splits):
    m, c = _meeting(cfg, splits, "small", 3)
    model = RoomModel(m, c.simulation, SR, seed=0)
    mic = tuple(model.mics[:, 0])
    src = (mic[0] + 1.0, mic[1], mic[2])
    toward, away = Pose(src, 180.0), Pose(src, 0.0)
    h = model.rirs([toward, away])
    direct = slice(0, int(0.01 * SR) + 100)  # first 10 ms after the onset holds the direct path
    e_to = float(np.sum(h[toward.key()][0][direct] ** 2))
    e_away = float(np.sum(h[away.key()][0][direct] ** 2))
    assert 10 * math.log10(e_to / e_away) > 4.0


def test_poses_keep_seats_fixed_and_drift_presenters(cfg, splits):
    over = {"seating.presenters.presentation": {"small": {"dist": "bernoulli", "p": 1.0},
                                                "large": {"dist": "bernoulli", "p": 1.0}},
            "acoustics.hybrid": {"small": {"dist": "bernoulli", "p": 0.0}, "large": {"dist": "bernoulli", "p": 1.0}}}
    m, c = _meeting(cfg, splits, "large", 5, **over)
    seg = m.dynamics.presentation
    assert seg is not None
    spk = sorted(p.speaker_id for p in m.people)
    segs = [Segment(turn_id=f"t{i + 1:04d}", speaker_id=spk[i % len(spk)], kind="normal", start=float(i),
                    end=float(i) + 0.5) for i in range(400)]
    tl = Timeline(meeting_id=m.meeting_id, sample_rate=SR, segments=segs,
                  stats=TimelineStats(duration=401.0, speech_time=200.0, overlap_share=0, backchannels_per_min=0,
                                      interruption_share=0, long_pauses=0))
    poses = segment_poses(m, tl, c.simulation, np.random.default_rng(0))
    for p in m.people:
        mine = [poses[s.turn_id] for s in segs if s.speaker_id == p.speaker_id]
        if p.position_type in ("table", "wall"):
            assert {q.xyz for q in mine} == {tuple(p.mouth)}
            to_table = math.degrees(math.atan2(m.table.center[1] - p.mouth[1], m.table.center[0] - p.mouth[0]))
            for q in mine[1:]:
                dev = (q.facing_deg - to_table + 180) % 360 - 180
                assert abs(dev) <= m.acoustics.orientation_max_deg + c.simulation.facing_step_deg
        elif p.position_type == "remote":
            assert all(q.xyz == tuple(m.speakerphone.position) and q.facing_deg is None for q in mine)
            assert math.dist(mine[0].xyz, m.mic.position) >= c.simulation.speaker_mic_gap_m
        elif p.position_type == "presenter":
            standing = [q for q in mine if abs(q.xyz[2] - p.mouth[2]) < 1e-9]
            assert standing
            for q in standing:
                d = math.dist(q.xyz[:2], p.presenter.spot)
                assert d <= p.presenter.drift_radius + c.simulation.drift_grid_m
            assert len({q.xyz for q in standing}) > 1, "presenters drift between turns"


def test_a_laptop_speakerphone_is_kept_off_the_mic(cfg, splits):
    from synthetics.acoustics.scene import speakerphone_xyz

    over = {"acoustics.hybrid": {"small": {"dist": "bernoulli", "p": 1.0}, "large": {"dist": "bernoulli", "p": 1.0}}}
    m, c = _meeting(cfg, splits, "small", 4, **over)
    assert m.speakerphone is not None and tuple(m.speakerphone.position) == tuple(m.mic.position)
    xyz = speakerphone_xyz(m, c.simulation.speaker_mic_gap_m)
    assert math.dist(xyz, m.mic.position) >= c.simulation.speaker_mic_gap_m - 0.03
    owner = m.person(m.mic.near_speaker).mouth
    assert math.dist(xyz[:2], owner[:2]) < math.dist(m.mic.position[:2], owner[:2])


def test_procedural_noise_is_finite_audible_and_seeded():
    for kind in ("hvac", "projector_fan", "keyboard", "door", "chair", "cup", "paper"):
        a, src = noise.make(kind, SR * 60, SR, np.random.default_rng(1), {})
        b, _ = noise.make(kind, SR * 60, SR, np.random.default_rng(1), {})
        assert src == "procedural" and np.all(np.isfinite(a)) and np.std(a) > 0
        assert np.array_equal(a, b)
        if kind in noise.BEDS or kind == "keyboard":
            assert len(a) == SR * 60


def test_noise_library_clips_replace_procedural_ones(tmp_path):
    from synthetics.backends.base import Audio

    (tmp_path / "hvac").mkdir()
    Audio(np.sin(np.arange(SR * 3) * 0.05).astype(np.float32) * 0.1, 22050).save(tmp_path / "hvac/a.wav")
    (tmp_path / "index.yaml").write_text("clips:\n  - {id: hvac-a, kind: hvac, file: hvac/a.wav, source: test, "
                                         "licence: CC0}\n")
    lib = noise.load_library(tmp_path)
    x, src = noise.make("hvac", SR * 10, SR, np.random.default_rng(0), lib)
    assert src == "library:hvac-a" and len(x) == SR * 10 and np.std(x[-SR:]) > 0
    assert noise.make("door", 0, SR, np.random.default_rng(0), lib)[1] == "procedural"


def test_band_limit_removes_energy_above_the_cutoff():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, SR * 2)).astype(np.float32)
    device.band_limit(x, SR, 4000.0)
    f, p = __import__("scipy.signal", fromlist=["welch"]).welch(x[1], SR, nperseg=1024)
    assert p[f > 5500].mean() < 1e-3 * p[(f > 500) & (f < 3000)].mean()


def test_agc_brings_quiet_and_loud_speech_to_the_target():
    t = np.arange(SR * 40) / SR
    x = (np.sin(2 * np.pi * 200 * t) * np.where(t < 20, 0.01, 0.3)).astype(np.float32)
    g = device.agc_gain(x, SR, target_db=-23, max_gain_db=30, attack_s=0.5, release_s=3.0, gate_db=-80)
    y = x * g
    lev = device.frame_rms_db(y, SR)
    assert abs(np.median(lev[100:380]) + 23) < 2 and abs(np.median(lev[500:780]) + 23) < 2


def test_clipping_reports_its_share():
    x = np.linspace(-1, 1, 1001, dtype=np.float32)
    share = device.hard_clip(x, 0.8)
    assert share == pytest.approx(0.2, abs=0.01) and np.abs(x).max() == pytest.approx(0.8)


@pytest.mark.skipif(not device.have_ffmpeg(), reason="ffmpeg with libopus not installed")
def test_opus_keeps_length_and_alignment():
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 6, np.float32)
    for _ in range(10):
        i = int(rng.integers(SR // 2, SR * 5))
        x[i:i + 400] += np.hanning(400).astype(np.float32) * np.sin(np.arange(400) * 0.3).astype(np.float32)
    y = device.opus(x, SR, 24)
    assert len(y) == len(x)
    assert device.codec_lag(x, y, SR) == 0


def test_aec_ducks_local_talk_during_remote_speech():
    local = np.ones((1, SR * 4), np.float32)
    remote = np.ones((1, SR * 4), np.float32)
    info = device.aec_artifacts(local, remote, SR, [(1.0, 2.0)], np.random.default_rng(0), duck_db=(6.0, 6.0),
                                onset_s=(0.1, 0.1))
    assert info["double_talk_duck_db"] == 6.0
    assert local[0, int(1.5 * SR)] == pytest.approx(10 ** (-6 / 20), rel=0.02)
    assert local[0, int(3.5 * SR)] == pytest.approx(1.0, abs=0.01)
    assert remote[0, int(1.0 * SR) + 10] < 0.2 and remote[0, int(1.5 * SR)] == 1.0
