"""`synthetics nsc clean`: candidate references -> denoise (deepfilternet3) -> dereverb (wpe-dereverb) -> score
(dnsmos), keeping each speaker's top clips by MOS in `nsc-part3-refs/clean/<speaker>/`, written once and reused
by every run.

`clean/<speaker>/scores.jsonl` records every cleaned candidate's score (so re-runs skip finished clips);
`clean/<speaker>/index.jsonl` lists the kept references; `clean/index.jsonl` is all speakers together:
clip path, transcript (NSC and ours), duration, sheet metadata, cleanup version and MOS.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable

from ..artifacts.io import atomic_write_text
from ..backends.base import Audio
from ..backends.registry import Backends
from ..config.loader import load_run_config
from ..config.schema import Phase, Role
from .extract import load_candidates, wav_path
from .pool import PoolSpeaker, load_pool
from .settings import NscPaths

CLEAN_VERSION = "c1:deepfilternet3-0.5.6+wpe-10-3-3:dnsmos-p835"
ROLES = (Role.ENHANCER, Role.DEREVERB, Role.REF_QUALITY)
DEFAULT_MOS_MIN = 2.5  # DNSMOS OVRL floor for a usable reference


def rank_candidates(rows: list[dict]) -> list[dict]:
    """Which candidates to clean first: 7-12 s, little markup (fillers, particles, non-speech), more words."""

    def key(r: dict) -> tuple:
        d = r["duration"]
        length_pen = 0.0 if 7 <= d <= 12 else min(abs(d - 7), abs(d - 12))
        markup = len(re.findall(r"[(\[]", r["text"]))
        words = len(r["text"].split())
        return (round(length_pen, 1), markup, -words, r["pair_id"])

    return sorted(rows, key=key)


def _read_jsonl(path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_speaker(sp: PoolSpeaker, paths: NscPaths, backends: Backends, *, keep: int, max_candidates: int,
                  log: Callable[[str], None] = print) -> dict:
    out_dir = paths.clean_speaker(sp.speaker)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"
    done = {r["pair_id"]: r for r in _read_jsonl(scores_path) if r.get("cleanup_version") == CLEAN_VERSION}
    rows = rank_candidates(load_candidates(paths, sp.speaker))
    if max_candidates > 0:
        rows = rows[:max_candidates]
    enh, der, q = (backends.get(r) for r in ROLES)
    new = []
    for row in rows:
        if row["pair_id"] in done:
            continue
        audio = Audio.load(wav_path(paths, row))
        cleaned = der.clean(enh.clean(audio))
        cleaned = Audio(cleaned.samples, cleaned.sr).resample(audio.sr)
        mos = float(q.score([cleaned])[0])
        cleaned.save(out_dir / row["wav"])
        rec = {"pair_id": row["pair_id"], "mos": round(mos, 4), "cleanup_version": CLEAN_VERSION}
        done[row["pair_id"]] = rec
        new.append(rec)
    if new:
        atomic_write_text(scores_path, "".join(json.dumps(r) + "\n" for r in sorted(done.values(), key=lambda r: r["pair_id"])))

    by_id = {r["pair_id"]: r for r in rows}
    ranked = sorted((r for r in done.values() if r["pair_id"] in by_id), key=lambda r: (-r["mos"], r["pair_id"]))
    kept = ranked[:keep]
    kept_ids = {r["pair_id"] for r in kept}
    for r in ranked[keep:]:  # only the kept clips stay on disk; scores.jsonl keeps every score
        (out_dir / by_id[r["pair_id"]]["wav"]).unlink(missing_ok=True)
    sheet = {"gender": sp.gender, "age": sp.age, "ethnicity": sp.ethnicity, "first_language": sp.first_language,
             "languages": list(sp.languages), "split": sp.split}
    index = []
    for r in kept:
        c = by_id[r["pair_id"]]
        index.append({"speaker": sp.speaker, "pair_id": c["pair_id"], "clip": f"clean/{sp.speaker}/{c['wav']}",
                      "raw_clip": f"raw/{sp.speaker}/{c['wav']}", "duration": c["duration"], "text_nsc": c["text_nsc"],
                      "text": c["text"], "mos": r["mos"], "cleanup_version": CLEAN_VERSION, **sheet})
    atomic_write_text(out_dir / "index.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in index))
    log(f"{sp.speaker}: cleaned {len(new)} new, kept {len(kept)} (MOS {kept[0]['mos'] if kept else '-'}"
        f"..{kept[-1]['mos'] if kept else '-'})")
    assert kept_ids == {x["pair_id"] for x in index}
    return {"speaker": sp.speaker, "cleaned": len(new), "kept": len(kept), "best_mos": kept[0]["mos"] if kept else None}


def clean_all(paths: NscPaths, *, mode: str = "aicore", keep: int = 5, max_candidates: int = 12,
              speakers: list[str] | None = None, lease: bool = True, log: Callable[[str], None] = print) -> dict:
    from ..orchestrator.lease import GpuLease
    from ..orchestrator.run import resolve

    cfg = resolve(load_run_config(overrides={"run_id": "nsc-clean", "seed": 0, "n_meetings": 0, "backends.mode": mode}))
    backends = Backends(cfg.backends, ROLES)
    services = sorted({cfg.backends.roles[r] for r in cfg.phases.phases[Phase.PREP].roles})
    pool = [s for s in load_pool(paths.pool_csv) if not speakers or s.speaker in speakers]
    results = []
    with GpuLease(services, platform_dir=cfg.phases.platform_dir, no_evict=cfg.phases.no_evict,
                  enabled=lease and mode in ("aicore", "record"), keep=True):
        for i, sp in enumerate(pool, 1):
            results.append(clean_speaker(sp, paths, backends, keep=keep, max_candidates=max_candidates,
                                         log=lambda msg, i=i: log(f"[{i}/{len(pool)}] {msg}")))
    write_global_index(paths)
    return {"speakers": len(results), "cleaned_now": sum(r["cleaned"] for r in results),
            "kept": sum(r["kept"] for r in results), "index": str(paths.clean_index)}


def write_global_index(paths: NscPaths) -> int:
    lines = []
    for d in sorted(p for p in paths.clean.glob("*") if p.is_dir()):
        lines += (d / "index.jsonl").read_text(encoding="utf-8").splitlines() if (d / "index.jsonl").is_file() else []
    atomic_write_text(paths.clean_index, "".join(line + "\n" for line in lines if line.strip()))
    return len(lines)


def readiness(paths: NscPaths, *, min_refs: int = 3, meeting_size: int = 20, mos_min: float = DEFAULT_MOS_MIN) -> dict:
    """M2 exit check: every selected speaker has >= `min_refs` clean references (MOS >= `mos_min`), and each
    split can still fill a `meeting_size`-speaker meeting after removing speakers that fail cleanup."""
    pool = load_pool(paths.pool_csv)
    good: Counter = Counter()
    for row in _read_jsonl(paths.clean_index):
        if row.get("mos", 0) >= mos_min and row.get("cleanup_version") == CLEAN_VERSION:
            good[row["speaker"]] += 1
    failing = sorted(s.speaker for s in pool if good[s.speaker] < min_refs)
    by_split = Counter(s.split for s in pool if s.speaker not in failing)
    splits_ok = {sp: by_split.get(sp, 0) >= meeting_size for sp in ("train", "dev", "test")}
    groups = Counter((s.split, s.ethnicity, s.gender) for s in pool if s.speaker in failing)
    return {"ok": not failing and all(splits_ok.values()), "speakers": len(pool), "min_refs": min_refs,
            "mos_min": mos_min, "failing_speakers": failing, "usable_per_split": dict(by_split),
            "splits_can_fill_meeting": splits_ok,
            "replacements_needed": [{"split": a, "ethnicity": b, "gender": c, "count": n} for (a, b, c), n in groups.items()],
            "refs_per_speaker": {"min": min((good[s.speaker] for s in pool), default=0),
                                 "max": max((good[s.speaker] for s in pool), default=0)}}
