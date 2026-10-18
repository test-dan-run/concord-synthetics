from __future__ import annotations

from pathlib import Path

import pytest

from synthetics.config import load_run_config
from synthetics.config.schema import RunConfig
from synthetics.nsc.pool import speaker_splits

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(**overrides) -> RunConfig:
    base = {"run_id": "test", "seed": 1234, "n_meetings": 5, "backends.mode": "fake",
            "speaker_pool.path": str(ROOT / "data/nsc_part3_reference_speakers.csv"),
            "lexicon.path": str(ROOT / "backend/synthetics/config/defaults/singlish_lexicon.yaml")}
    base.update(overrides)
    return load_run_config(overrides=base)


@pytest.fixture(scope="session")
def cfg() -> RunConfig:
    return make_cfg()


@pytest.fixture(scope="session")
def splits(cfg):
    return speaker_splits(cfg.speaker_pool.path, cfg.speaker_pool.version)
