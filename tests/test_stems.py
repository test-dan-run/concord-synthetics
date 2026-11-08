"""Stage 7 stems: each speaker as heard at the mic and the noise alone, which add up to the mixture before clipping
and the codec, never change the mixture, and stay out of old runs' cache keys."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

from helpers import fake_run
from post_helpers import run_to_phase_c
from synthetics.artifacts.io import read_model
from synthetics.artifacts.labels import AcousticRender, ManifestEntry
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.paths import RunPaths
from synthetics.config.schema import StageId
from synthetics.orchestrator.lease import GpuLease
from synthetics.orchestrator.phase_post import run_post
from synthetics.orchestrator.scheduler import PhaseScheduler, RunReport
from synthetics.stages.s7_acoustic import AcousticStage

ON, OFF = {"dist": "bernoulli", "p": 1.0}, {"dist": "bernoulli", "p": 0.0}
# a linear device chain (band limit and AGC, no clipping or codec) and no echo canceller, so the sum is exact
LINEAR = {"simulation.noise_library": None, "acoustics.hybrid": {"small": OFF, "large": OFF},
          "acoustics.device_chain.band_limit": ON, "acoustics.device_chain.agc": ON,
          "acoustics.device_chain.codec": OFF, "acoustics.device_chain.clipping": OFF,
          "acoustics.mic.centre_array": ON}


def _post(rd, cfg, until=StageId.LABELS):
    sched = PhaseScheduler(rd, cfg, lease_factory=lambda s, **k: GpuLease(s, **{**k, "enabled": False}))
    run_post(sched, RunPaths(rd).meeting_ids(), RunReport(), until=until)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One run up to stage 6, copied; stage 7 and 8 then run on each copy, with stems on and off, so both renders
    start from the very same tracks (fake voices differ from run to run)."""
    root = tmp_path_factory.mktemp("stems")
    rd, cfg = fake_run(root / "on", n=2, seed=9, **LINEAR, **{"simulation.write_stems": True})
    run_to_phase_c(rd, cfg)
    _post(rd, cfg, until=StageId.ASSEMBLE)
    off_dir = root / "off" / rd.name
    shutil.copytree(rd, off_dir)
    cfg_off = cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update={"write_stems": False})})
    _post(rd, cfg)
    _post(off_dir, cfg_off)
    return (rd, cfg), (off_dir, cfg_off)


def test_stems_and_noise_add_up_to_the_mixture(runs):
    (rd, _), _ = runs
    for mid in RunPaths(rd).meeting_ids():
        p = RunPaths(rd).meeting(mid)
        m = read_model(p.meeting, MeetingSpec)
        mix, sr = sf.read(p.mixture, dtype="float64")
        parts = [sf.read(p.stem(x.speaker_id), dtype="float64")[0] for x in m.people]
        noise, _ = sf.read(p.noise, dtype="float64")
        assert sr == 16000 and all(len(x) == len(mix) for x in [*parts, noise])
        resid = mix - (np.sum(parts, axis=0) + noise)
        assert np.abs(resid).max() < (len(parts) + 2) / 32768  # 16-bit rounding of each file, nothing more
        files = read_model(p.render, AcousticRender).files
        assert files["noise"] == "noise.wav" and {f"stem:{x.speaker_id}" for x in m.people} <= set(files)
        entry = read_model(p.dir / "labels/manifest_entry.json", ManifestEntry)
        assert entry.files["stems"].endswith("/stems") and entry.files["noise"].endswith("/noise.wav")


def test_stems_do_not_change_the_mixture(runs):
    (on, _), (off, _) = runs
    for mid in RunPaths(on).meeting_ids():
        a, b = RunPaths(on).meeting(mid), RunPaths(off).meeting(mid)
        assert a.mixture.read_bytes() == b.mixture.read_bytes()
        assert not b.stems_dir.exists() and not b.noise.exists()
        assert not list(a.dir.glob(".stems-*")), "scratch removed"


def test_stems_stay_out_of_the_cache_key_while_off(runs):
    (_, cfg_on), (_, cfg_off) = runs
    off = AcousticStage().config_slice(cfg_off).model_dump()
    assert "write_stems" not in off["simulation"]
    assert AcousticStage().config_slice(cfg_on).model_dump()["simulation"]["write_stems"] is True
