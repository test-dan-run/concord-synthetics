"""Grow a voice pool from a curated voice folder, giving each new speaker one split for good (§4.3).

The folder is in the `nsc-part3-voices` layout: `speakers.csv` (one row per speaker, the sheet's attributes) and
`manifest.jsonl` (one line per cleaned clip, with its MOS). Every speaker of the base pool keeps its split. A new
speaker whose recording partner is already in the pool joins the partner's split, and new partners stay together.
The rest are spread over the splits by `split_mix`, balanced within each gender and ethnicity group, with a seeded
draw, so the same inputs always give the same pool. Splits stay speaker-disjoint across every run.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..artifacts.io import file_sha256
from .pool import PoolMeta, PoolSpeaker, check_pool, load_pool, pool_meta, write_meta, write_pool

SPLITS = ("train", "dev", "test")
MOS_FLOOR = 2.5  # DNSMOS overall floor for a usable cloning reference (nsc/cleanup.py DEFAULT_MOS_MIN)
MIN_CLIPS = 3  # usable clips a speaker needs: stage 5 swaps to the next clip before another speaker


@dataclass
class VoiceSpeaker:
    """A speaker of a voice folder, with the clips that pass the MOS floor."""

    row: dict[str, str]
    clips: list[dict] = field(default_factory=list)

    @property
    def speaker(self) -> str:
        return self.row["speaker"].zfill(8)

    @property
    def partner(self) -> str | None:
        p = (self.row.get("partner_id") or "").strip()
        return p.zfill(8) if p else None

    def usable(self) -> list[dict]:
        return [c for c in self.clips if (c.get("mos") or 0) >= MOS_FLOOR]


@dataclass
class Draft:
    """A proposed pool: the base speakers, the new ones with their splits, and the speakers left out with why."""

    base: list[PoolSpeaker]
    added: list[PoolSpeaker]
    left_out: dict[str, str]
    forced: set[str]  # new speakers placed by their partner's split
    voices: dict[str, VoiceSpeaker]

    @property
    def pool(self) -> list[PoolSpeaker]:
        return sorted(self.base + self.added, key=lambda s: s.speaker)

    def counts(self) -> dict[str, int]:
        c = Counter(s.split for s in self.pool)
        return {k: c.get(k, 0) for k in SPLITS}

    def coverage(self) -> dict[str, dict[str, dict[str, int]]]:
        """split -> ethnicity -> {"base": n, "added": n}."""
        out: dict[str, dict[str, dict[str, int]]] = {s: {} for s in SPLITS}
        for tag, group in (("base", self.base), ("added", self.added)):
            for s in group:
                cell = out[s.split].setdefault(s.ethnicity, {"base": 0, "added": 0})
                cell[tag] += 1
        return out


def read_voices(folder: Path) -> dict[str, VoiceSpeaker]:
    folder = Path(folder)
    with open(folder / "speakers.csv", newline="", encoding="utf-8") as fh:
        voices = {v.speaker: v for v in (VoiceSpeaker(r) for r in csv.DictReader(fh))}
    with open(folder / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                c = json.loads(line)
                v = voices.get(str(c["speaker"]).zfill(8))
                if v is not None:
                    v.clips.append(c)
    for v in voices.values():
        v.clips.sort(key=lambda c: (-(c.get("mos") or 0), c["clip_id"]))
    return voices


def problems_of(v: VoiceSpeaker) -> str | None:
    """Why a voice cannot join the pool, or None."""
    missing = [k for k in ("gender", "age", "ethnicity", "first_language") if not (v.row.get(k) or "").strip()]
    if missing:
        return f"sheet is missing {', '.join(missing)}"
    if v.row["gender"] not in ("F", "M"):
        return f"gender {v.row['gender']!r} is not F or M"
    if len(v.usable()) < MIN_CLIPS:
        return f"{len(v.usable())} clips at MOS {MOS_FLOOR} or more, needs {MIN_CLIPS}"
    return None


def as_pool_speaker(v: VoiceSpeaker, split: str) -> PoolSpeaker:
    r = v.row
    zips = {c.get("source_zip") for c in v.clips if c.get("source_zip")}
    return PoolSpeaker(
        speaker=v.speaker, split=split, sdm_zip=(sorted(zips)[0] if zips else f"SDM/SPEAKER{v.speaker}.zip"),
        conf=r.get("conference_id", ""), partner=v.partner, gender=r["gender"], age=r["age"],
        ethnicity=r["ethnicity"], first_language=r["first_language"],
        languages=tuple(x.strip() for x in (r.get("languages") or "").split(",") if x.strip()),
        education=r.get("education", ""), occupation=r.get("occupation", ""), relationship=r.get("relationship", ""),
        n_segments=0, speech_min=0.0, ref_candidates=int(r.get("candidates") or 0),
        ref_candidates_min=0.0, main_clip=None)


def _units(new: list[VoiceSpeaker]) -> list[list[VoiceSpeaker]]:
    """New speakers grouped with any new partner, so a pair always lands in one split."""
    by_id = {v.speaker: v for v in new}
    seen: set[str] = set()
    units = []
    for v in sorted(new, key=lambda v: v.speaker):
        if v.speaker in seen:
            continue
        unit = [v]
        seen.add(v.speaker)
        p = by_id.get(v.partner or "")
        if p is not None and p.speaker not in seen:
            unit.append(p)
            seen.add(p.speaker)
        units.append(unit)
    return units


def assign_splits(units: list[list[VoiceSpeaker]], split_mix: dict[str, float], seed: int) -> dict[str, str]:
    """speaker -> split. Within each gender and ethnicity group (by the unit's first speaker), units go one by one,
    in a seeded order, to the split furthest below its target share of that group."""
    total = sum(split_mix.get(s, 0.0) for s in SPLITS)
    target = {s: split_mix.get(s, 0.0) / total for s in SPLITS}
    groups: dict[tuple[str, str], list[list[VoiceSpeaker]]] = defaultdict(list)
    for u in units:
        groups[(u[0].row["ethnicity"], u[0].row["gender"])].append(u)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 406]))
    out: dict[str, str] = {}
    for key in sorted(groups):
        members = groups[key]
        order = rng.permutation(len(members))
        counts = dict.fromkeys(SPLITS, 0)
        placed = 0
        for i in order:
            u = members[int(i)]
            placed += len(u)
            split = max(SPLITS, key=lambda s: (target[s] * placed - counts[s], -SPLITS.index(s)))
            counts[split] += len(u)
            for v in u:
                out[v.speaker] = split
    return out


def draft_pool(base_path: Path, folder: Path, *, split_mix: dict[str, float], seed: int) -> Draft:
    base = load_pool(base_path)
    base_split = {s.speaker: s.split for s in base}
    voices = read_voices(folder)
    left_out: dict[str, str] = {}
    candidates = []
    for sid, v in sorted(voices.items()):
        if sid in base_split:
            continue
        why = problems_of(v)
        if why:
            left_out[sid] = why
        else:
            candidates.append(v)
    forced = {v.speaker: base_split[v.partner] for v in candidates if v.partner in base_split}
    free = [v for v in candidates if v.speaker not in forced]
    splits = {**assign_splits(_units(free), split_mix, seed), **forced}
    added = [as_pool_speaker(v, splits[v.speaker]) for v in candidates]
    return Draft(base=base, added=added, left_out=left_out, forced=set(forced), voices=voices)


def publish(draft: Draft, out_path: Path, *, version: str, base_path: Path, base_version: str, folder: Path,
            seed: int, split_mix: dict[str, float], main_clips: dict[str, str] | None = None,
            exclude: set[str] | None = None) -> PoolMeta:
    """Write the pool CSV and its sidecar. `main_clips` (speaker -> clip id) and `exclude` carry admin choices.
    Excluding a speaker also excludes a new partner of theirs, so a pair is never split."""
    exclude = set(exclude or ())
    by_id = {s.speaker: s for s in draft.added}
    for sid in list(exclude):
        p = by_id.get(sid)
        if p is not None and p.partner in by_id:
            exclude.add(p.partner)
    main_clips = main_clips or {}
    for sid, clip in main_clips.items():
        v = draft.voices.get(sid)
        if v is None or clip not in {c["clip_id"] for c in v.usable()}:
            raise ValueError(f"{clip} is not a usable clip of {sid}")
    added = [replace(s, main_clip=main_clips.get(s.speaker)) for s in draft.added if s.speaker not in exclude]
    base = [replace(s, main_clip=main_clips.get(s.speaker, s.main_clip)) for s in draft.base]
    pool = sorted(base + added, key=lambda s: s.speaker)
    counts = Counter(s.split for s in pool)
    meta = PoolMeta(version=version, counts={k: counts.get(k, 0) for k in SPLITS if counts.get(k)},
                    base=base_version, base_sha256=file_sha256(base_path), source=str(folder),
                    refs_index=str(Path(folder) / "manifest.jsonl"), seed=seed, split_mix=dict(split_mix))
    write_pool(out_path, pool)
    write_meta(out_path, meta)
    problems = check_pool(out_path)
    if problems:
        raise ValueError("; ".join(problems))
    return meta


def base_version_of(path: Path, fallback: str) -> str:
    meta = pool_meta(path)
    return meta.version if meta is not None else fallback
