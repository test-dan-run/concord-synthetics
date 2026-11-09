"""The pinned NSC Part 3 voice pools with their fixed splits (§4.3).

A pool is a CSV whose `split` column is copied into each run's speaker_splits.json, so a speaker keeps one split in
every run that uses the pool. The first pool, `data/nsc_part3_reference_speakers.csv` (175 speakers), has no
sidecar and is checked against fixed counts. A later pool sits in `data/pools/<version>.csv` beside
`<version>.json`, a sidecar recording its split counts and where it came from, and is checked against those.
A run records its pool path and version in config.yaml, so a new pool never changes an old run.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..artifacts.io import atomic_write_text, file_sha256
from ..artifacts.models import SpeakerSplits, SplitSpeaker

EXPECTED_SPLITS = {"train": 50, "dev": 25, "test": 100}
EXPECTED_TOTAL = 175
COLUMNS = ("speaker", "split", "sdm_zip", "conf", "partner", "gender", "age", "ethnicity", "first_language",
           "languages", "education", "occupation", "relationship", "n_segments", "speech_min", "ref_candidates",
           "ref_candidates_min", "main_clip")


@dataclass(frozen=True)
class PoolSpeaker:
    speaker: str
    split: str
    sdm_zip: str
    conf: str
    partner: str | None
    gender: str
    age: str
    ethnicity: str
    first_language: str
    languages: tuple[str, ...]
    education: str
    occupation: str
    relationship: str
    n_segments: int
    speech_min: float
    ref_candidates: int
    ref_candidates_min: float
    main_clip: str | None = None  # the clip stage 4 clones from; None: the speaker's best clip by MOS


@dataclass(frozen=True)
class PoolMeta:
    """The sidecar of a pool CSV: what validation expects, and where the pool came from."""

    version: str
    counts: dict[str, int]
    base: str | None = None  # the pool version it extends; every base speaker keeps its split
    base_sha256: str | None = None
    source: str | None = None  # the voice folder new speakers came from
    refs_index: str | None = None  # the clip manifest runs on this pool clone from
    seed: int | None = None
    split_mix: dict[str, float] = field(default_factory=dict)  # target shares for speakers not tied to a partner

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class PoolError(ValueError):
    pass


def _int(v: str) -> int:
    return int(v) if v.strip() else 0


def _float(v: str) -> float:
    return float(v) if v.strip() else 0.0


def load_pool(path: Path) -> list[PoolSpeaker]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append(PoolSpeaker(
            speaker=r["speaker"].zfill(8), split=r["split"], sdm_zip=r["sdm_zip"], conf=r["conf"],
            partner=(r["partner"].zfill(8) if r["partner"].strip() else None), gender=r["gender"], age=r["age"],
            ethnicity=r["ethnicity"], first_language=r["first_language"],
            languages=tuple(x.strip() for x in r["languages"].split(",") if x.strip()),
            education=r["education"], occupation=r["occupation"], relationship=r["relationship"],
            n_segments=_int(r["n_segments"]), speech_min=_float(r["speech_min"]),
            ref_candidates=_int(r["ref_candidates"]), ref_candidates_min=_float(r["ref_candidates_min"]),
            main_clip=(r.get("main_clip") or "").strip() or None))
    return out


def write_pool(path: Path, pool: list[PoolSpeaker]) -> None:
    """The pool as CSV, sorted by speaker, in the column order of COLUMNS."""
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(COLUMNS), lineterminator="\n")
    w.writeheader()
    for s in sorted(pool, key=lambda s: s.speaker):
        row = asdict(s)
        row["languages"] = ",".join(s.languages)
        row["partner"] = s.partner or ""
        row["main_clip"] = s.main_clip or ""
        w.writerow({k: row[k] for k in COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, buf.getvalue())


def sidecar_path(path: Path) -> Path:
    return Path(path).with_suffix(".json")


def pool_meta(path: Path) -> PoolMeta | None:
    """The pool's sidecar, or None for a pool without one (the first pool)."""
    p = sidecar_path(path)
    if not p.is_file():
        return None
    return PoolMeta(**json.loads(p.read_text(encoding="utf-8")))


def write_meta(path: Path, meta: PoolMeta) -> None:
    atomic_write_text(sidecar_path(path), json.dumps(asdict(meta), indent=2) + "\n")


def expected_counts(path: Path) -> dict[str, int]:
    meta = pool_meta(path)
    return dict(meta.counts) if meta is not None else dict(EXPECTED_SPLITS)


def validate_pool(pool: list[PoolSpeaker], nsc_dir: Path | None = None, *,
                  expected_splits: dict[str, int] = EXPECTED_SPLITS, expected_total: int | None = None) -> list[str]:
    """Problems with the pool (empty list = valid). Zip presence is checked only when `nsc_dir` is given."""
    if expected_total is None:
        expected_total = sum(expected_splits.values())
    problems = []
    ids = [s.speaker for s in pool]
    dupes = [k for k, n in Counter(ids).items() if n > 1]
    if dupes:
        problems.append(f"duplicate speakers: {', '.join(dupes)}")
    if len(set(ids)) != expected_total:
        problems.append(f"{len(set(ids))} unique speakers, expected {expected_total}")
    counts = Counter(s.split for s in pool)
    if dict(counts) != expected_splits:
        problems.append(f"split sizes {dict(sorted(counts.items()))}, expected {expected_splits}")
    by_id = {s.speaker: s for s in pool}
    for s in pool:
        p = by_id.get(s.partner or "")
        if p is not None and p.split != s.split and s.speaker < p.speaker:
            problems.append(f"partner pair split across splits: {s.speaker} ({s.split}) / {p.speaker} ({p.split})")
    if nsc_dir is not None:
        missing = [s.sdm_zip for s in pool if not (Path(nsc_dir) / s.sdm_zip).is_file()]
        if missing:
            problems.append(f"{len(missing)} zips missing under {nsc_dir}: {', '.join(missing[:5])}"
                            + (" ..." if len(missing) > 5 else ""))
    return problems


def check_pool(path: Path, nsc_dir: Path | None = None) -> list[str]:
    """validate_pool against the counts this pool expects (its sidecar, or the first pool's fixed counts)."""
    return validate_pool(load_pool(path), nsc_dir, expected_splits=expected_counts(path))


def speaker_splits(path: Path, version: str) -> SpeakerSplits:
    """The run's speaker_splits.json content, copied from the pool CSV."""
    pool = load_pool(path)
    problems = validate_pool(pool, expected_splits=expected_counts(path))
    if problems:
        raise PoolError("; ".join(problems))
    speakers = {s.speaker: SplitSpeaker(
        split=s.split, gender=s.gender, age=s.age, ethnicity=s.ethnicity, first_language=s.first_language,
        languages=list(s.languages), partner=s.partner, education=s.education, relationship=s.relationship)
        for s in sorted(pool, key=lambda s: s.speaker)}
    return SpeakerSplits(pool_version=version, pool_path=str(path), pool_sha256=file_sha256(path), speakers=speakers,
                         counts=dict(sorted(Counter(s.split for s in pool).items())))


def main_clips(path: Path) -> dict[str, str]:
    """Speaker -> the clip an admin chose to clone from, for the speakers that have one."""
    return {s.speaker: s.main_clip for s in load_pool(path) if s.main_clip}
