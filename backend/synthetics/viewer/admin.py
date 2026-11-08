"""Admin routes for the speaker pool: import a curated voice folder as a draft, review it, publish a new pool.

Only `synthetics viewer --admin` adds these routes; the viewer has no sign-in, so the flag is the gate. One draft
exists at a time, kept in `<runs root>/.admin/pool-draft.json` (the dot keeps it out of the run list). A draft holds
only the admin's choices (source folder, version, main clips, speakers left out); the proposed splits are recomputed
from the folder on every read with the same seed, so they never drift. Publishing writes `data/pools/<version>.csv`
and its sidecar and never rewrites an existing pool, so runs on older pools keep their speakers and splits.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..artifacts.io import atomic_write_text
from ..nsc.pool import load_pool, pool_meta
from ..nsc.pool_build import MIN_CLIPS, MOS_FLOOR, SPLITS, Draft, base_version_of, draft_pool, publish
from ..orchestrator.run import APP_ROOT, app_path

SPEAKER = re.compile(r"^\d{8}$")
CLIP = re.compile(r"^\d{8}_\d{4}$")
VERSION = re.compile(r"^[a-z0-9][a-z0-9.-]{2,47}$")
SEED = 2026


class DraftIn(BaseModel):
    source: str
    version: str | None = None


class SpeakerPatch(BaseModel):
    main_clip: str | None = None
    excluded: bool | None = None


def _current(defaults_loader) -> tuple[Path, str, dict[str, float]]:
    sp = defaults_loader().speaker_pool
    return app_path(sp.path), sp.version, {str(k): float(v) for k, v in sp.split_mix.weights.items()}


def _coverage(pool) -> dict[str, dict[str, int]]:
    c = Counter((s.split, s.ethnicity) for s in pool)
    return {sp: {e: n for (s2, e), n in sorted(c.items()) if s2 == sp} for sp in SPLITS}


def _runs_using(runs_root: Path, version: str) -> int:
    n = 0
    for p in runs_root.glob("*/speaker_splits.json"):
        try:
            n += json.loads(p.read_text()).get("pool_version") == version
        except (OSError, ValueError):
            continue
    return n


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(APP_ROOT))
    except ValueError:
        return str(p)


def _next_version(total: int, pools_dir: Path) -> str:
    taken = {p.stem for p in pools_dir.glob("*.csv")}
    k = 2 + len(taken)
    while f"p3-{total}-v{k}" in taken:
        k += 1
    return f"p3-{total}-v{k}"


def add_routes(app: FastAPI, runs_root: Path, defaults_loader, pools_dir: Path | None = None) -> None:
    pools_dir = pools_dir or APP_ROOT / "data/pools"
    state_path = runs_root / ".admin" / "pool-draft.json"

    def read_state() -> dict | None:
        return json.loads(state_path.read_text()) if state_path.is_file() else None

    def write_state(s: dict) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(state_path, json.dumps(s, indent=2) + "\n")

    def current_draft() -> tuple[dict, Draft]:
        s = read_state()
        if s is None:
            raise HTTPException(404, "no pool draft; import a voice folder first")
        base, _, mix = _current(defaults_loader)
        return s, draft_pool(base, Path(s["source"]), split_mix=mix, seed=SEED)

    def draft_view(s: dict, d: Draft) -> dict:
        excluded = set(s.get("exclude", []))
        by_id = {x.speaker: x for x in d.added}
        for sid in list(excluded):  # a pair is left out together
            p = by_id.get(sid)
            if p is not None and p.partner in by_id:
                excluded.add(p.partner)
        rows = []
        for sid, v in sorted(d.voices.items()):
            if sid in {b.speaker for b in d.base}:
                continue
            ps = by_id.get(sid)
            usable = v.usable()
            status = ("left_out" if sid in d.left_out else "excluded" if sid in excluded else "ready")
            rows.append({
                "speaker": sid, "gender": v.row.get("gender", ""), "age": v.row.get("age", ""),
                "ethnicity": v.row.get("ethnicity", ""), "first_language": v.row.get("first_language", ""),
                "source": "NSC Part 3", "consent": "NSC licence", "split": ps.split if ps else None,
                "by_partner": sid in d.forced, "partner": v.partner, "status": status,
                "reason": d.left_out.get(sid), "usable_clips": len(usable),
                "best_mos": usable[0]["mos"] if usable else (v.clips[0]["mos"] if v.clips else None),
                "main_clip": s.get("main_clips", {}).get(sid) or (usable[0]["clip_id"] if usable else None),
                "clips": [{"clip_id": c["clip_id"], "text": c.get("text", ""), "mos": c.get("mos"),
                           "duration": c.get("duration")} for c in v.clips],
            })
        kept = [x for x in d.added if x.speaker not in excluded]
        after = Counter(x.split for x in d.base + kept)
        added = Counter(x.split for x in kept)
        return {"source": s["source"], "version": s["version"], "speakers": rows,
                "counts": {k: after.get(k, 0) for k in SPLITS}, "added": {k: added.get(k, 0) for k in SPLITS},
                "coverage": d.coverage(), "rules": {"mos_floor": MOS_FLOOR, "min_clips": MIN_CLIPS},
                "ready": sum(r["status"] == "ready" for r in rows),
                "not_ready": sum(r["status"] != "ready" for r in rows)}

    @app.get("/api/admin/pool")
    def pool_info() -> dict:
        base, version, _ = _current(defaults_loader)
        pool = load_pool(base)
        versions = [{"version": "p3-175-v1", "path": "data/nsc_part3_reference_speakers.csv"}]
        for p in sorted(pools_dir.glob("*.json")):
            meta = pool_meta(p.with_suffix(".csv"))
            if meta is not None:
                versions.append({"version": meta.version, "path": _rel(p.with_suffix(".csv")),
                                 "base": meta.base, "counts": meta.counts})
        counts = Counter(s.split for s in pool)
        s = read_state()
        return {"version": version, "path": str(base), "speakers": len(pool),
                "counts": {k: counts.get(k, 0) for k in SPLITS}, "coverage": _coverage(pool),
                "runs_using": _runs_using(runs_root, version), "versions": versions,
                "draft": {"source": s["source"], "version": s["version"]} if s else None}

    @app.post("/api/admin/pool/drafts", status_code=201)
    def new_draft(body: DraftIn) -> dict:
        src = Path(body.source).expanduser()
        if not (src / "speakers.csv").is_file() or not (src / "manifest.jsonl").is_file():
            raise HTTPException(422, f"{src} is not a voice folder (needs speakers.csv and manifest.jsonl)")
        base, base_ver, mix = _current(defaults_loader)
        d = draft_pool(base, src, split_mix=mix, seed=SEED)
        version = body.version or _next_version(len(d.base) + len(d.added), pools_dir)
        if not VERSION.match(version) or (pools_dir / f"{version}.csv").exists():
            raise HTTPException(422, f"pool version {version!r} is taken or not valid")
        s = {"source": str(src), "version": version, "base_version": base_version_of(base, base_ver),
             "main_clips": {}, "exclude": []}
        write_state(s)
        return draft_view(s, d)

    @app.get("/api/admin/pool/draft")
    def get_draft() -> dict:
        return draft_view(*current_draft())

    @app.patch("/api/admin/pool/draft/speakers/{speaker}")
    def patch_speaker(speaker: str, body: SpeakerPatch) -> dict:
        s, d = current_draft()
        v = d.voices.get(speaker) if SPEAKER.match(speaker) else None
        if v is None or speaker not in {x.speaker for x in d.added}:
            raise HTTPException(404, f"{speaker} is not a new speaker of this draft")
        if body.main_clip is not None:
            if body.main_clip not in {c["clip_id"] for c in v.usable()}:
                raise HTTPException(422, f"{body.main_clip} is not a usable clip of {speaker}")
            s["main_clips"][speaker] = body.main_clip
        if body.excluded is not None:
            ex = set(s["exclude"])
            if body.excluded:
                ex.add(speaker)
            else:
                ex.discard(speaker)
            s["exclude"] = sorted(ex)
        write_state(s)
        return draft_view(s, d)

    @app.post("/api/admin/pool/draft/publish")
    def publish_draft() -> dict:
        s, d = current_draft()
        base, _, mix = _current(defaults_loader)
        out = pools_dir / f"{s['version']}.csv"
        if out.exists():
            raise HTTPException(409, f"{out.name} exists; pools are never rewritten")
        try:
            meta = publish(d, out, version=s["version"], base_path=base, base_version=s["base_version"],
                           folder=Path(s["source"]), seed=SEED, split_mix=mix, main_clips=s["main_clips"],
                           exclude=set(s["exclude"]))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        state_path.unlink(missing_ok=True)
        return {"version": meta.version, "path": _rel(out), "counts": meta.counts,
                "speakers": meta.total,
                "next": f"set speaker_pool.path: {_rel(out)}, version: {meta.version} and "
                        f"refs_index: {meta.refs_index} in config/defaults/stages.yaml for new runs"}

    @app.delete("/api/admin/pool/draft", status_code=204)
    def drop_draft() -> None:
        state_path.unlink(missing_ok=True)

    @app.get("/api/admin/voices/{speaker}/{clip}")
    def voice_clip(speaker: str, clip: str):
        s = read_state()
        if s is None or not SPEAKER.match(speaker) or not CLIP.match(clip) or not clip.startswith(speaker):
            raise HTTPException(404, "Not Found")
        path = (Path(s["source"]) / "clean" / speaker / f"{clip}.wav").resolve()
        if not path.is_file() or not path.is_relative_to(Path(s["source"]).resolve()):
            raise HTTPException(404, "Not Found")
        return FileResponse(path, media_type="audio/wav")
