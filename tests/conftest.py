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


ALWAYS = {"dist": "bernoulli", "p": 1.0}
# large hybrid rooms with presenters and wall seats by choice, so every position type the viewer draws is present
VIEWER_RUN = {"simulation.noise_library": None,
              "room_classes.ratio": {"dist": "categorical", "weights": {"small": 0.0, "large": 1.0}},
              "room_classes.classes.large.speakers": {"dist": "int_uniform", "lo": 7, "hi": 9},
              "acoustics.hybrid": {"small": ALWAYS, "large": ALWAYS},
              "seating.presenters.presentation": {"small": ALWAYS, "large": ALWAYS},
              "seating.wall.by_choice": ALWAYS}


@pytest.fixture(scope="session")
def viewer_run(tmp_path_factory) -> Path:
    """Two meetings through all eight stages on fake backends (the viewer's data contract and API tests)."""
    from helpers import fake_run
    from synthetics.config.schema import StageId
    from synthetics.orchestrator.lease import GpuLease
    from synthetics.orchestrator.scheduler import PhaseScheduler
    from voice_helpers import fake_refs

    tmp = tmp_path_factory.mktemp("viewer")
    mp = pytest.MonkeyPatch()
    mp.setenv("NSC_REFS_DIR", str(fake_refs(tmp / "refs")))
    mp.setenv("SOUND_LIBRARY_DIR", str(tmp / "no-library"))
    try:
        rd, cfg = fake_run(tmp / "runs", n=2, seed=21, **VIEWER_RUN)
        PhaseScheduler(rd, cfg, lease_factory=lambda s, **kw: GpuLease(s, **{**kw, "enabled": False})).run(
            until=StageId.LABELS)
    finally:
        mp.undo()
    return rd
