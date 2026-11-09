"""Export routes: size estimates, a pack of the kept meetings written to a folder, and one meeting as a zip.

A pack of a large run is many gigabytes, so it is written on the server into `$SYNTHETICS_EXPORT_ROOT` (default
/home/user/code/datasets/synthetics-exports, outside the app) by a background thread, and the browser polls its
progress; only one meeting at a time is downloaded through the browser. The kept meetings are the run's finished
meetings (in manifest.jsonl) without a leave-out mark in review.json. Nothing here writes into a run.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from ..artifacts.io import atomic_write_text, read_model
from ..artifacts.labels import Timeline
from ..export.pack import ROOM_TAIL_S, SAMPLE_RATE, PackOptions, export_run, missing, pack_meeting
from .api import REVIEW, Runs, manifest_entries

DEFAULT_ROOT = Path("/home/user/code/datasets/synthetics-exports")
BYTES_PER_S = SAMPLE_RATE * 2  # 16-bit mono


class OptionsIn(BaseModel):
    clips: bool = True
    full: bool = False
    clean: bool = True
    room: bool = True
    noise: bool = False

    def pack(self) -> PackOptions:
        return PackOptions(**self.model_dump())


class ExportIn(BaseModel):
    options: OptionsIn = Field(default_factory=OptionsIn)
    meetings: list[str] | None = None  # None: every kept meeting


def export_root() -> Path:
    return Path(os.environ.get("SYNTHETICS_EXPORT_ROOT") or DEFAULT_ROOT)


def kept(runs: Runs, run_id: str) -> list[str]:
    rp = runs.run(run_id)
    marks = json.loads((rp.run_dir / REVIEW).read_text()) if (rp.run_dir / REVIEW).is_file() else {}
    return [e["meeting_id"] for e in manifest_entries(rp)
            if marks.get(e["meeting_id"], {}).get("decision") != "exclude"]


def estimate(runs: Runs, run_id: str, meeting_ids: list[str]) -> dict:
    """Bytes of each part of a pack, from the timelines: the recording, each speaker's own audio by cut, noise."""
    rp = runs.run(run_id)
    out = {"mixture": 0, "full": 0, "clips_clean": 0, "clips_room": 0, "noise": 0, "labels": 0}
    for mid in meeting_ids:
        p = rp.meeting(mid)
        tl = read_model(p.timeline, Timeline)
        n_spk = len({s.speaker_id for s in tl.segments})
        dur = tl.stats.duration
        out["mixture"] += int(dur * BYTES_PER_S)
        out["noise"] += int(dur * BYTES_PER_S)
        out["full"] += int(n_spk * dur * BYTES_PER_S)
        out["clips_clean"] += int(sum(s.end - s.start for s in tl.segments) * BYTES_PER_S)
        out["clips_room"] += int(sum(s.end - s.start + ROOM_TAIL_S for s in tl.segments) * BYTES_PER_S)
        out["labels"] += sum(f.stat().st_size for f in (p.meeting, p.timeline, p.rttm, p.transcript) if f.is_file())
    return {"meetings": len(meeting_ids), "bytes": out}


def add_routes(app: FastAPI, runs: Runs) -> None:
    jobs: dict[str, dict] = {}
    lock = threading.Lock()

    def job_file(job_id: str) -> Path:
        return export_root() / ".jobs" / f"{job_id}.json"

    def save(job: dict) -> None:
        with lock:
            jobs[job["id"]] = dict(job)
        try:
            job_file(job["id"]).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(job_file(job["id"]), json.dumps(job, indent=2) + "\n")
        except OSError:
            pass  # progress stays in memory when the export root is not writable

    @app.get("/api/runs/{run_id}/exports/estimate")
    def export_estimate(run_id: str) -> dict:
        return {**estimate(runs, run_id, kept(runs, run_id)), "root": str(export_root())}

    @app.post("/api/runs/{run_id}/exports", status_code=202)
    def start_export(run_id: str, body: ExportIn) -> dict:
        rp = runs.run(run_id)
        ids = body.meetings if body.meetings is not None else kept(runs, run_id)
        unknown = sorted(set(ids) - set(rp.meeting_ids()))
        if unknown or not ids:
            raise HTTPException(422, f"no meetings to export{': unknown ' + ', '.join(unknown) if unknown else ''}")
        opts = body.options.pack()
        gaps = {mid: g for mid in ids if (g := missing(rp.meeting(mid), opts, _speakers(rp, mid)))}
        if gaps:
            first = next(iter(gaps))
            raise HTTPException(422, f"{len(gaps)} meetings lack files these options need, e.g. {first}: "
                                     f"{', '.join(gaps[first])}")
        job_id = f"{run_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        out = export_root() / job_id
        job = {"id": job_id, "run": run_id, "state": "running", "done": 0, "total": len(ids), "out": str(out),
               "options": body.options.model_dump(), "error": None}
        save(job)

        def work() -> None:
            def progress(i: int, n: int, mid: str) -> None:
                save({**job, "done": i, "current": mid})

            try:
                res = export_run(rp.run_dir, out, ids, opts, progress)
                save({**job, "state": "finished", "done": len(ids), "files": res["files"]})
            except Exception as exc:  # noqa: BLE001 - reported to the page
                save({**job, "state": "failed", "error": f"{type(exc).__name__}: {exc}"})

        threading.Thread(target=work, daemon=True, name=f"export-{job_id}").start()
        return job

    @app.get("/api/runs/{run_id}/exports")
    def list_exports(run_id: str) -> list[dict]:
        runs.run(run_id)
        with lock:
            return sorted((j for j in jobs.values() if j["run"] == run_id), key=lambda j: j["id"], reverse=True)

    @app.get("/api/runs/{run_id}/meetings/{meeting_id}/download")
    def download_meeting(run_id: str, meeting_id: str, clips: bool = True, full: bool = False, clean: bool = True,
                         room: bool = True, noise: bool = False):
        p = runs.meeting(run_id, meeting_id)
        opts = PackOptions(clips=clips, full=full, clean=clean, room=room, noise=noise)
        gaps = missing(p, opts, _speakers(runs.run(run_id), meeting_id))
        if gaps:
            raise HTTPException(409, f"{meeting_id} has no {', '.join(gaps)}")
        tmp = Path(tempfile.mkdtemp(prefix="synthetics-dl-"))
        pack_meeting(p, tmp / "pack", opts)
        archive = shutil.make_archive(str(tmp / f"{run_id}-{meeting_id}"), "zip", tmp / "pack")
        return FileResponse(archive, media_type="application/zip", filename=f"{run_id}-{meeting_id}.zip",
                            background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))


def _speakers(rp, meeting_id: str) -> list[str]:
    tl = rp.meeting(meeting_id).timeline
    return sorted({s.speaker_id for s in read_model(tl, Timeline).segments}) if tl.is_file() else []
