"""The Room Config Viewer's read-only API over `runs/<run_id>/` (M12), and the built frontend.

A browser cannot read the local disk, so this layer lists runs and serves their files. It adds no fields of its
own: every number the viewer shows is either in a pipeline artifact or computed in the browser. There are no
write routes. Run ids, meeting ids and artifact names are checked against what is on disk and a whitelist before
any path is built, so no request reaches a file outside a run.

    GET /api/runs                                   id, meeting counts, seed, config hash
    GET /api/runs/{id}/config                       config.yaml as JSON
    GET /api/runs/{id}/manifest                     manifest.jsonl as a JSON array
    GET /api/runs/{id}/summary                      run_report.json and voice_report.json
    GET /api/runs/{id}/speaker_splits               speaker_splits.json
    GET /api/runs/{id}/meetings                     one summary per meeting (manifest line, status, plan)
    GET /api/runs/{id}/meetings/{mid}/{artifact}    whitelisted JSON files and prompts/<speaker_id>.txt
    GET /api/runs/{id}/jobs                         progress from the status.json files
    GET /api/runs/{id}/export?...                   the manifest slice (JSONL), the viewer's one action
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from ..artifacts.io import read_model
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import MeetingStatus
from ..artifacts.paths import MeetingPaths, RunPaths
from ..config.loader import config_hash, read_yaml
from ..config.schema import StageId

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MEETING_ID = re.compile(r"^m-\d{6}$")
PROMPT = re.compile(r"^prompts/S\d{2,3}\.txt$")
# meeting files the viewer reads; audio is not served (playback is out of scope)
ARTIFACTS = ("meeting.json", "script.json", "script_validation.json", "status.json", "synthesis.json",
             "voice_validation.json", "timeline.json", "render.json", "labels/transcript.json",
             "labels/manifest_entry.json")
RUN_FILES = {"summary": ("run_report.json", "voice_report.json"), "speaker_splits": ("speaker_splits.json",)}
STAGE_NAMES = {StageId.SAMPLE: "Config", StageId.SCRIPT: "Script", StageId.SCRIPT_VAL: "Script check",
               StageId.VOICE: "Voices", StageId.VOICE_VAL: "Voice check", StageId.ASSEMBLE: "Timeline",
               StageId.ACOUSTIC: "Acoustics", StageId.LABELS: "Labels"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Runs:
    """The runs under one root directory. Every lookup checks the id before it touches the disk."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and RUN_ID.match(p.name) and (p / "config.yaml").is_file())

    def run(self, run_id: str) -> RunPaths:
        if not RUN_ID.match(run_id) or run_id not in self.ids():
            raise HTTPException(404, f"no run {run_id!r} under {self.root}")
        return RunPaths(self.root / run_id)

    def meeting(self, run_id: str, meeting_id: str) -> MeetingPaths:
        rp = self.run(run_id)
        if not MEETING_ID.match(meeting_id) or meeting_id not in rp.meeting_ids():
            raise HTTPException(404, f"no meeting {meeting_id!r} in run {run_id!r}")
        return rp.meeting(meeting_id)


def manifest_entries(rp: RunPaths) -> list[dict]:
    if not rp.manifest.is_file():
        return []
    return [json.loads(x) for x in rp.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]


def status_of(paths: MeetingPaths) -> dict:
    """status.json as the viewer reads it: every stage, with pending ones filled in."""
    st = read_model(paths.status, MeetingStatus) if paths.status.is_file() else MeetingStatus(meeting_id=paths.meeting_id)
    stages = {}
    for s in StageId:
        x = st.get(s)
        stages[str(int(s))] = {"name": STAGE_NAMES[s], **x.model_dump(mode="json", exclude={"cache_key", "feedback"})}
    return stages


def meeting_state(stages: dict) -> str:
    """done | failed | running | pending, from the per-stage states."""
    states = [v["state"] for v in stages.values()]
    if stages[str(int(StageId.LABELS))]["state"] == "pass":
        return "done"
    if any(s in ("fail", "dropped") for s in states):
        return "failed"
    if any(s in ("running", "retry") for s in states) or any(s in ("pass", "skipped") for s in states):
        return "running"
    return "pending"


def plan_of(m: MeetingSpec) -> dict:
    """The parts of meeting.json the browser grid needs for its cards, filters and mini floor plans."""
    return {
        "room_class": m.room_class, "n_speakers": m.n_speakers, "split": m.split, "seed": m.seed,
        "fit_resamples": m.fit_resamples,
        "room": m.room.model_dump(mode="json"),
        "table": {"shape": m.table.shape, "center": m.table.center, "length": m.table.length, "width": m.table.width,
                  "capacity": m.table.capacity,
                  "slots": [{"xy": s.seat_xy, "occupied_by": s.occupied_by} for s in m.table.slots]},
        "wall_chairs": [{"xy": c.xy, "wall": c.wall, "occupied_by": c.occupied_by} for c in m.wall_chairs],
        "people": [{"speaker_id": p.speaker_id, "position_type": p.position_type, "wall_reason": p.wall_reason,
                    "mouth": p.mouth, "is_chair": p.is_chair, "turn_share": p.turn_share} for p in m.people],
        "mic": {"type": m.mic.type, "position": m.mic.position, "channels": m.mic.channels},
        "speakerphone": m.speakerphone.model_dump(mode="json") if m.speakerphone else None,
        "hybrid": m.dynamics.hybrid, "presentation": m.dynamics.presentation is not None,
        "acoustics": {"rt60": m.acoustics.rt60, "snr_db": m.acoustics.snr_db},
        "device_chain": m.device_chain.model_dump(mode="json"),
    }


def meeting_summary(paths: MeetingPaths, entry: dict | None) -> dict:
    stages = status_of(paths)
    plan = None
    if paths.meeting.is_file():
        try:
            plan = plan_of(read_model(paths.meeting, MeetingSpec))
        except (ValidationError, ValueError):
            plan = None
    return {"meeting_id": paths.meeting_id, "state": meeting_state(stages), "stages": stages, "plan": plan,
            "manifest": entry}


def _matches(s: dict, *, room_class: str | None, split: str | None, min_speakers: int, remote: bool,
             presenters: bool, wall: bool) -> bool:
    """The browser's filters, on a meeting summary (the same rules as the meeting browser screen)."""
    plan = s["plan"]
    if plan is None:
        return False
    kinds = [p["position_type"] for p in plan["people"]]
    return ((room_class in (None, "", "all") or plan["room_class"] == room_class)
            and (split in (None, "", "all") or plan["split"] == split)
            and plan["n_speakers"] >= min_speakers
            and (not remote or "remote" in kinds) and (not presenters or "presenter" in kinds)
            and (not wall or "wall" in kinds))


def create_app(runs_root: Path | None = None, frontend_dist: Path | None = None) -> FastAPI:
    from ..orchestrator.run import APP_ROOT
    from ..orchestrator.run import runs_root as default_root

    runs = Runs(runs_root or default_root())
    app = FastAPI(title="Synthetics viewer", version="0.1.0")
    app.state.runs = runs

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        out = []
        for rid in runs.ids():
            rp = RunPaths(runs.root / rid)
            cfg = read_yaml(rp.config)
            out.append({"id": rid, "meetings": len(rp.meeting_ids()), "done": len(manifest_entries(rp)),
                        "seed": cfg.get("seed"), "n_meetings": cfg.get("n_meetings"), "config_hash": config_hash(cfg),
                        "revision": cfg.get("revision", 1), "updated": rp.config.stat().st_mtime})
        return out

    @app.get("/api/runs/{run_id}/config")
    def run_config(run_id: str) -> dict:
        rp = runs.run(run_id)
        cfg = read_yaml(rp.config)
        return {"config_hash": config_hash(cfg), "config": cfg}

    @app.get("/api/runs/{run_id}/manifest")
    def run_manifest(run_id: str) -> list[dict]:
        return manifest_entries(runs.run(run_id))

    def run_files(run_id: str, key: str) -> dict:
        rp = runs.run(run_id)
        return {name.removesuffix(".json"): (_json(rp.run_dir / name) if (rp.run_dir / name).is_file() else None)
                for name in RUN_FILES[key]}

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict:
        return run_files(run_id, "summary")

    @app.get("/api/runs/{run_id}/speaker_splits")
    def run_splits(run_id: str) -> dict:
        data = run_files(run_id, "speaker_splits")["speaker_splits"]
        if data is None:
            raise HTTPException(404, "speaker_splits.json missing")
        return data

    def summaries(run_id: str) -> list[dict]:
        rp = runs.run(run_id)
        entries = {e["meeting_id"]: e for e in manifest_entries(rp)}
        return [meeting_summary(rp.meeting(mid), entries.get(mid)) for mid in rp.meeting_ids()]

    @app.get("/api/runs/{run_id}/meetings")
    def run_meetings(run_id: str) -> list[dict]:
        return summaries(run_id)

    @app.get("/api/runs/{run_id}/meetings/{meeting_id}/{artifact:path}")
    def meeting_artifact(run_id: str, meeting_id: str, artifact: str):
        paths = runs.meeting(run_id, meeting_id)
        if artifact not in ARTIFACTS and not PROMPT.match(artifact):
            raise HTTPException(404, f"{artifact!r} is not served")
        f = paths.dir / artifact
        if not f.is_file():
            raise HTTPException(404, f"{meeting_id} has no {artifact} yet")
        if artifact.endswith(".txt"):
            return PlainTextResponse(f.read_text(encoding="utf-8"))
        return FileResponse(f, media_type="application/json")

    @app.get("/api/runs/{run_id}/jobs")
    def run_jobs(run_id: str) -> dict:
        rp = runs.run(run_id)
        per_stage = {str(int(s)): {} for s in StageId}
        states: dict[str, int] = {}
        latest = None
        for mid in rp.meeting_ids():
            stages = status_of(rp.meeting(mid))
            st = meeting_state(stages)
            states[st] = states.get(st, 0) + 1
            for k, v in stages.items():
                per_stage[k][v["state"]] = per_stage[k].get(v["state"], 0) + 1
                for t in (v.get("started_at"), v.get("ended_at")):
                    if t and (latest is None or t > latest):
                        latest = t
        running = [k for k, v in per_stage.items() if v.get("running")]
        return {"run": run_id, "meetings": len(rp.meeting_ids()), "states": states, "stages": per_stage,
                "running_stages": running, "active": bool(running), "last_update": latest}

    @app.get("/api/runs/{run_id}/export")
    def run_export(run_id: str, room_class: str | None = None, split: str | None = None,
                   min_speakers: int = Query(0, ge=0), remote: bool = False, presenters: bool = False,
                   wall: bool = False, ids: str | None = Query(None, description="comma-separated meeting ids")):
        rp = runs.run(run_id)
        wanted = {x for x in (ids or "").split(",") if x}
        picked = {s["meeting_id"] for s in summaries(run_id)
                  if _matches(s, room_class=room_class, split=split, min_speakers=min_speakers, remote=remote,
                              presenters=presenters, wall=wall)}
        if wanted:
            picked &= wanted
        lines = [json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in manifest_entries(rp)
                 if e["meeting_id"] in picked]
        body = "\n".join(lines) + ("\n" if lines else "")
        return Response(body, media_type="application/x-ndjson",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}-manifest-slice.jsonl"',
                                 "X-Meetings": str(len(lines))})

    dist = frontend_dist or APP_ROOT / "frontend" / "dist"
    if (dist / "index.html").is_file():
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404, "Not Found")  # unknown API routes never fall through to the SPA
            f = (dist / path).resolve()
            if path and f.is_file() and f.is_relative_to(dist.resolve()):
                return FileResponse(f)
            return FileResponse(dist / "index.html")

    return app


def app() -> FastAPI:
    """uvicorn factory: `uvicorn --factory synthetics.viewer.api:app`."""
    return create_app()
