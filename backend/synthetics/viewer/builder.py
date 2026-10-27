"""The viewer's run builder: preview a config, create a run, start and stop it.

These are the only routes that write. Creating a run writes what `init-run` writes (the frozen config.yaml,
speaker_splits.json and empty meeting folders); starting it runs the pipeline in a background process
(`orchestrator/background.py`), which writes the run's files as the CLI would. Nothing here edits a frozen
config or a stage's output. `synthetics viewer --read-only` leaves these routes out.

    GET  /api/defaults                  the default config, complete (as a new run would freeze it)
    POST /api/preview                   stage 1 in memory for a draft: the meetings, the sampler checks
    POST /api/runs                      create the run from a draft (init-run)
    GET  /api/runs/{id}/process         the background process, its log tail and the GPU services it leases
    POST /api/runs/{id}/start           start (or resume) the run in the background
    POST /api/runs/{id}/stop            interrupt it; Start again resumes

A draft is the run id, meeting count and seed, plus dotted overrides from the form and an optional YAML overlay,
applied in that order over the defaults, exactly as `init-run --set` and `--config` apply them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from ..config.loader import config_hash
from ..config.schema import Phase, RunConfig
from ..orchestrator import background
from .api import RUN_ID, Runs, plan_of

PREVIEW_MAX = 500  # meetings sampled for a preview; a larger run is previewed on its first 500
PREVIEW_PLANS = 6  # full plans sent for the mini floor plans
RESERVED = ("run_id", "seed", "n_meetings")  # set by the draft's own fields
# Real backends: m9-e2e (20 meetings, 7.3 h of audio) took about 9 h over every phase, so about 74 s per minute of
# audio. A rough guide for the confirmation step, not a promise.
SECONDS_PER_AUDIO_MIN = 9 * 3600 / 436.5
GPU_PHASES = (Phase.A, Phase.B, Phase.B2, Phase.C, Phase.D)


class Draft(BaseModel):
    run_id: str = ""
    n_meetings: int = Field(20, ge=1, le=10_000)
    seed: int = Field(1, ge=0)
    overrides: dict[str, Any] = Field(default_factory=dict)  # dotted keys, e.g. "acoustics.hybrid.large.p"
    yaml: str = ""  # an overlay like a --config file; dotted keys work too


class StartRequest(BaseModel):
    confirm: bool = False  # required for a run that leases GPU services


def _error(status: int, message: str, errors: list[dict] | None = None) -> HTTPException:
    return HTTPException(status, {"message": message, "errors": errors or []})


def draft_overrides(draft: Draft) -> dict[str, Any]:
    try:
        extra = yaml.safe_load(draft.yaml) if draft.yaml.strip() else {}
    except yaml.YAMLError as exc:
        raise _error(422, "The YAML overlay does not parse.", [{"loc": "yaml", "msg": str(exc)}]) from exc
    if not isinstance(extra, dict):
        raise _error(422, "The YAML overlay must be a mapping.", [{"loc": "yaml", "msg": "top level is not a mapping"}])
    out = {**draft.overrides, **{str(k): v for k, v in extra.items()}}
    reserved = [k for k in out if k in RESERVED]
    if reserved:
        raise _error(422, "Set the run id, meeting count and seed in their own fields.",
                     [{"loc": k, "msg": "set by the run fields"} for k in reserved])
    return out


def build_config(draft: Draft) -> tuple[RunConfig, dict[str, Any]]:
    """(validated config, the overrides it was built with)."""
    from ..config.loader import load_run_config

    overrides = draft_overrides(draft)
    base = {"run_id": draft.run_id or "preview", "seed": draft.seed, "n_meetings": draft.n_meetings}
    try:
        return load_run_config(overrides={**base, **overrides}), overrides
    except ValidationError as exc:
        errors = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        raise _error(422, f"The config is not valid ({len(errors)} problem{'s' * (len(errors) != 1)}).",
                     errors) from exc
    except (TypeError, ValueError, AttributeError) as exc:  # a dotted key through a non-mapping, and the like
        raise _error(422, "The overrides do not fit the config.", [{"loc": "overrides", "msg": str(exc)}]) from exc


def is_live(cfg: RunConfig) -> bool:
    return cfg.backends.mode in ("aicore", "record")


def gpu_services(cfg: RunConfig) -> dict[str, list[str]]:
    """Platform services each GPU phase leases, as the scheduler computes them."""
    out = {}
    for ph in GPU_PHASES:
        spec = cfg.phases.phases.get(ph)
        if spec:
            out[ph.value] = sorted({cfg.backends.roles[r] for r in spec.roles if r in cfg.backends.roles})
    return out


def _row(m) -> dict:
    kinds = [p.position_type for p in m.people]
    return {"meeting_id": m.meeting_id, "room_class": m.room_class, "n_speakers": m.n_speakers, "split": m.split,
            "rt60": m.acoustics.rt60, "snr_db": m.acoustics.snr_db, "hybrid": m.dynamics.hybrid,
            "presentation": m.dynamics.presentation is not None, "duration_min": m.dynamics.duration_min,
            "table_shape": m.table.shape, "mic_type": m.mic.type, "fit_resamples": m.fit_resamples,
            "positions": {k: kinds.count(k) for k in ("table", "wall", "presenter", "remote")}}


def preview(draft: Draft) -> dict:
    """Stage 1 for the draft's meetings, in memory. A meeting is a function of the run seed, its id and the
    config, so these are the meetings the run will sample."""
    from ..nsc.pool import speaker_splits
    from ..orchestrator.run import resolve
    from ..stages.runner import meeting_id_for
    from ..stages.s1_sample import run_meeting
    from ..validators.sampler_report import as_dicts, sampler_checks

    cfg, overrides = build_config(draft)
    rc = resolve(cfg)
    try:
        sp = speaker_splits(rc.speaker_pool.path, rc.speaker_pool.version)
    except (OSError, ValueError) as exc:
        raise _error(422, f"The speaker pool cannot be read: {exc}") from exc
    n = min(cfg.n_meetings, PREVIEW_MAX)
    meetings, failures = [], []
    for i in range(1, n + 1):
        mid = meeting_id_for(i)
        try:
            meetings.append(run_meeting(rc, sp, mid))
        except Exception as exc:  # an infeasible room or seating: shown against the meeting, not as a crash
            failures.append({"meeting_id": mid, "error": f"{type(exc).__name__}: {exc}"})
    minutes = sum(m.dynamics.duration_min for m in meetings) * (cfg.n_meetings / n if n else 0)
    plans = [{"meeting_id": m.meeting_id, **plan_of(m)} for m in meetings[:PREVIEW_PLANS]]
    return {"config_hash": config_hash(cfg), "revision": cfg.revision, "n_meetings": cfg.n_meetings,
            "n_sampled": n, "overrides": sorted(overrides), "meetings": [_row(m) for m in meetings], "plans": plans,
            "checks": as_dicts(sampler_checks(rc, meetings)) if meetings else [], "failures": failures,
            "live": is_live(cfg), "gpu_services": gpu_services(cfg) if is_live(cfg) else {},
            "stages_enabled": {str(int(s)): cfg.enabled(s) for s in cfg.stages},
            "audio_hours": round(minutes / 60, 2),
            "estimate_hours": round(minutes * SECONDS_PER_AUDIO_MIN / 3600, 1) if is_live(cfg) else None}


def add_routes(app: FastAPI, runs: Runs, *, start=background.start) -> None:
    """The write routes, on the viewer app. `start` launches the background run (tests replace it)."""

    @app.get("/api/defaults")
    def defaults() -> dict:
        """Every field, so the form can compare a frozen config.yaml with it leaf by leaf."""
        from ..config.loader import load_run_config

        cfg = load_run_config(overrides={"run_id": "new", "seed": 1, "n_meetings": 20}).model_dump(mode="json")
        return {"config": {k: v for k, v in cfg.items() if k not in RESERVED}, "preview_max": PREVIEW_MAX,
                "reserved": list(RESERVED)}

    @app.post("/api/preview")
    def preview_route(draft: Draft) -> dict:
        return preview(draft)

    @app.post("/api/runs", status_code=201)
    def create_run(draft: Draft) -> dict:
        from ..orchestrator.run import init_run

        if not RUN_ID.match(draft.run_id):
            raise _error(422, "Run ids use letters, digits, dot, dash and underscore, and start with a letter or digit.",
                         [{"loc": "run_id", "msg": "not a valid run id"}])
        if (runs.root / draft.run_id).exists():
            raise _error(409, f"There is already a run called {draft.run_id}.",
                         [{"loc": "run_id", "msg": "already exists"}])
        cfg, overrides = build_config(draft)
        try:
            run_dir = init_run(draft.run_id, n_meetings=draft.n_meetings, seed=draft.seed, overrides=overrides,
                               root=runs.root)
        except FileExistsError as exc:
            raise _error(409, str(exc), [{"loc": "run_id", "msg": "already exists"}]) from exc
        return {"id": run_dir.name, "config_hash": config_hash(cfg), "live": is_live(cfg)}

    def run_config(run_dir: Path) -> RunConfig:
        from ..orchestrator.run import load_run

        return load_run(run_dir)[0]

    def process_info(run_id: str) -> dict:
        rp = runs.run(run_id)
        cfg = run_config(rp.run_dir)
        return {"run": run_id, "process": background.read_process(rp.run_dir).as_dict(),
                "log": background.log_tail(rp.run_dir), "live": is_live(cfg),
                "gpu_services": gpu_services(cfg) if is_live(cfg) else {}}

    @app.get("/api/runs/{run_id}/process")
    def run_process(run_id: str) -> dict:
        return process_info(run_id)

    @app.post("/api/runs/{run_id}/start")
    def start_run(run_id: str, req: StartRequest | None = None) -> dict:
        rp = runs.run(run_id)
        busy = [r for r in runs.ids() if background.running(runs.root / r)]
        if run_id in busy:
            raise _error(409, f"{run_id} is already running.")
        if busy:
            raise _error(409, f"{busy[0]} is running. One run at a time: stop it first, or wait for it to finish.")
        cfg = run_config(rp.run_dir)
        if is_live(cfg) and not (req and req.confirm):
            raise _error(409, "This run leases GPU services on the platform. Confirm to start it.",
                         [{"loc": "confirm", "msg": "required for real backends"}])
        start(rp.run_dir, live=is_live(cfg))
        return process_info(run_id)

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict:
        rp = runs.run(run_id)
        if not background.stop(rp.run_dir):
            raise _error(409, f"{run_id} is not running.")
        return process_info(run_id)
