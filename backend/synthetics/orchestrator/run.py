"""Runs on disk: creating a run, loading its frozen config, resolving app-relative paths."""

from __future__ import annotations

import os
from pathlib import Path

from ..artifacts.io import write_model
from ..artifacts.paths import RunPaths
from ..config.loader import config_hash, dump_run_config, load_run_config, read_run_config
from ..config.schema import RunConfig
from ..nsc.pool import speaker_splits
from ..stages.runner import meeting_id_for

APP_ROOT = Path(__file__).resolve().parents[3]


def app_path(p: Path | str) -> Path:
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else APP_ROOT / p


def runs_root() -> Path:
    return app_path(os.environ.get("SYNTHETICS_RUNS", "runs"))


def resolve(cfg: RunConfig) -> RunConfig:
    """Absolute paths for files the run reads (the frozen config keeps them app-relative)."""
    platform_dir = Path(os.environ["PLATFORM_DIR"]) if os.environ.get("PLATFORM_DIR") else cfg.phases.platform_dir
    no_evict = os.environ.get("NO_EVICT", "").lower() in ("1", "true", "yes") or cfg.phases.no_evict
    return cfg.model_copy(update={
        "speaker_pool": cfg.speaker_pool.model_copy(update={"path": app_path(cfg.speaker_pool.path)}),
        "lexicon": cfg.lexicon.model_copy(update={"path": app_path(cfg.lexicon.path)}),
        "backends": cfg.backends.model_copy(update={"recordings": app_path(cfg.backends.recordings)}),
        "phases": cfg.phases.model_copy(update={"platform_dir": app_path(platform_dir), "no_evict": no_evict}),
    })


def write_splits(run_dir: Path, cfg: RunConfig) -> Path:
    rc = resolve(cfg)
    rp = RunPaths(run_dir)
    write_model(rp.speaker_splits, speaker_splits(rc.speaker_pool.path, rc.speaker_pool.version))
    return rp.speaker_splits


def init_run(run_id: str, *, n_meetings: int, seed: int, configs: list[Path] | None = None,
             overrides: dict | None = None, root: Path | None = None, exist_ok: bool = False) -> Path:
    cfg = load_run_config(*(configs or []), overrides={"run_id": run_id, "seed": seed, "n_meetings": n_meetings,
                                                       **(overrides or {})})
    run_dir = (root or runs_root()) / run_id
    rp = RunPaths(run_dir)
    if rp.config.exists() and not exist_ok:
        raise FileExistsError(f"run {run_id} already exists at {run_dir}")
    dump_run_config(cfg, rp.config)
    write_splits(run_dir, cfg)
    for i in range(1, n_meetings + 1):
        rp.meeting(meeting_id_for(i)).dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_run(run_dir: Path) -> tuple[RunConfig, str]:
    """(resolved config, config hash of the frozen file)."""
    cfg = read_run_config(RunPaths(run_dir).config)
    return resolve(cfg), config_hash(cfg)
