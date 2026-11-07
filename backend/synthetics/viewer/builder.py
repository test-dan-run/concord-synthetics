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
from typing import Any, Literal

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
    # A pilot: the first PILOT_MEETINGS meetings of the draft as `<run_id>-pilot`, in one wave, to listen to before
    # the full run. Same seed and settings, so its meetings are the full run's first meetings.
    pilot: bool = False


PILOT_MEETINGS = 3
PILOT_SUFFIX = "-pilot"


class VoiceSwapIn(BaseModel):
    speaker_id: str


class StartRequest(BaseModel):
    confirm: bool = False  # required for a run that leases GPU services


class ReviewIn(BaseModel):
    meeting_id: str
    decision: Literal["keep", "exclude"] | None
    note: str = Field("", max_length=500)


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


def _ethnicity_error(short: list[str]) -> HTTPException:
    return _error(422, "The race mix cannot fill the largest meetings from the speaker pool.",
                  [{"loc": "speaker_pool.ethnicity_mix", "msg": msg} for msg in short])


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
    from ..stages.s1_sample import ethnicity_shortfalls, run_meeting
    from ..validators.sampler_report import as_dicts, sampler_checks

    cfg, overrides = build_config(draft)
    rc = resolve(cfg)
    try:
        sp = speaker_splits(rc.speaker_pool.path, rc.speaker_pool.version)
    except (OSError, ValueError) as exc:
        raise _error(422, f"The speaker pool cannot be read: {exc}") from exc
    short = ethnicity_shortfalls(rc, sp)
    if short:
        raise _ethnicity_error(short)
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
            "checks": as_dicts(sampler_checks(rc, meetings, sp)) if meetings else [], "failures": failures,
            "live": is_live(cfg), "gpu_services": gpu_services(cfg) if is_live(cfg) else {},
            "stages_enabled": {str(int(s)): cfg.enabled(s) for s in cfg.stages},
            "audio_hours": round(minutes / 60, 2),
            "estimate_hours": round(minutes * SECONDS_PER_AUDIO_MIN / 3600, 1) if is_live(cfg) else None}


def add_routes(app: FastAPI, runs: Runs, *, start=background.start) -> None:
    """The write routes, on the viewer app. `start` launches the background run (tests replace it)."""
    from . import export_routes, suggest_routes

    suggest_routes.add_routes(app)
    export_routes.add_routes(app, runs)

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
        from ..orchestrator.run import PoolShortfall, init_run

        if draft.pilot:
            draft = draft.model_copy(update={
                "run_id": draft.run_id + PILOT_SUFFIX, "n_meetings": min(draft.n_meetings, PILOT_MEETINGS),
                "overrides": {**draft.overrides, "phases.wave_size": PILOT_MEETINGS}, "pilot": False})
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
        except PoolShortfall as exc:
            raise _ethnicity_error(exc.problems) from exc
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

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_pilot(run_id: str) -> None:
        """Delete a pilot run (only a `-pilot` run that is not running): its meetings were for listening only."""
        import shutil

        rp = runs.run(run_id)
        if not run_id.endswith(PILOT_SUFFIX):
            raise _error(409, "Only pilot runs can be deleted here.")
        if background.running(rp.run_dir):
            raise _error(409, f"{run_id} is running. Stop it first.")
        shutil.rmtree(rp.run_dir)

    @app.post("/api/runs/{run_id}/meetings/{meeting_id}/voice-swap")
    def voice_swap(run_id: str, meeting_id: str, body: VoiceSwapIn) -> dict:
        """Give one speaker of one meeting another voice: a same-split speaker with the same gender and ethnicity
        (first language and age band when possible), the swap stage 5 would make. Stage 4 re-synthesises that
        speaker on the next start, and the later stages follow; nothing else in the run changes."""
        import json

        from ..artifacts.io import read_model
        from ..artifacts.models import SpeakerSplits
        from ..artifacts.voice import ReferenceSwap, Synthesis
        from ..config.schema import StageId
        from ..nsc.references import references_for
        from ..orchestrator.status import StatusStore

        mp = runs.meeting(run_id, meeting_id)
        rp = runs.run(run_id)
        if background.running(rp.run_dir):
            raise _error(409, f"{run_id} is running. Stop it first, or wait for it to finish.")
        if not mp.synthesis.is_file():
            raise _error(409, f"{meeting_id} has no voices yet.")
        syn = read_model(mp.synthesis, Synthesis)
        va = syn.voices.get(body.speaker_id)
        if va is None:
            raise _error(404, f"{meeting_id} has no speaker {body.speaker_id}.")
        cfg = run_config(rp.run_dir)
        refs = references_for(cfg)
        splits = read_model(rp.speaker_splits, SpeakerSplits)
        in_use = {v.nsc_speaker for v in syn.voices.values()} | {x for s in syn.swaps
                                                                  for x in (s.from_speaker, s.to_speaker)}
        new = refs.replacement(va.nsc_speaker, splits, in_use)
        if new is None:
            raise _error(409, "No other voice in this split matches the speaker's gender and ethnicity.")
        swap = ReferenceSwap(speaker_id=body.speaker_id, kind="speaker", from_speaker=va.nsc_speaker,
                             to_speaker=new, from_clip=va.clip_id, to_clip=refs.best(new).pair_id,
                             reason="asked for in review")
        StatusStore(rp.run_dir).update(meeting_id, StageId.VOICE, state="retry",
                                       feedback=json.dumps({"resynth": [], "swaps": [swap.model_dump()]}))
        return {"swap": swap.model_dump(), "next": "start the run to re-synthesise this speaker"}

    @app.post("/api/runs/{run_id}/review")
    def review_meeting(run_id: str, body: ReviewIn) -> dict:
        """Mark a meeting kept or left out of the export (`decision: null` clears the mark). review.json sits
        beside the run's results and nothing in the pipeline reads it, so a mark never changes a meeting."""
        import json
        import time

        from ..artifacts.io import atomic_write_text
        from .api import REVIEW

        runs.meeting(run_id, body.meeting_id)  # 404 for an unknown run or meeting
        rp = runs.run(run_id)
        f = rp.run_dir / REVIEW
        marks = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        if body.decision is None:
            marks.pop(body.meeting_id, None)
        else:
            marks[body.meeting_id] = {"decision": body.decision, "note": body.note,
                                      "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        atomic_write_text(f, json.dumps(dict(sorted(marks.items())), indent=2) + "\n")
        return marks
