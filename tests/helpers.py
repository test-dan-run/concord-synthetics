"""Shared helpers for tests that need a run directory with fake backends."""

from __future__ import annotations

from pathlib import Path

from conftest import ROOT
from synthetics.orchestrator.run import init_run, load_run


def fake_run(tmp_path: Path, n: int = 2, seed: int = 11, **overrides):
    """A run with fake backends and short meetings, so a full stage 1-3 pass takes well under a second.

    Stage 5 thresholds suit the fakes, not the real models: two random 16-bit fake voices often reach cosine 0.5,
    over the calibrated pair limits, and the fake MOS hashes the samples, so it is unrelated to the reference's.
    Tests that exercise those checks set them back.
    """
    base = {"backends.mode": "fake",
            "thresholds.sv_pair_max": {"sv_wavlm": 0.85, "sv_redimnet": 0.75},
            "thresholds.mos_drop_max": 5.0,
            "speaker_pool.path": str(ROOT / "data/nsc_part3_reference_speakers.csv"),
            "lexicon.path": str(ROOT / "backend/synthetics/config/defaults/singlish_lexicon.yaml"),
            "dynamics.duration_min": {"small": {"dist": "uniform", "lo": 4, "hi": 6},
                                      "large": {"dist": "uniform", "lo": 6, "hi": 8}}}
    base.update(overrides)
    rd = init_run("t", n_meetings=n, seed=seed, overrides=base, root=tmp_path)
    cfg, _ = load_run(rd)
    return rd, cfg
