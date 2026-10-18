"""The pinned NSC Part 3 voice pool (`data/nsc_part3_reference_speakers.csv`) with its fixed splits (§4.3).

The CSV is the source of truth: its `split` column is copied into each run's speaker_splits.json.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..artifacts.io import file_sha256
from ..artifacts.models import SpeakerSplits, SplitSpeaker

EXPECTED_SPLITS = {"train": 50, "dev": 25, "test": 100}
EXPECTED_TOTAL = 175


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


class PoolError(ValueError):
    pass


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
            n_segments=int(r["n_segments"]), speech_min=float(r["speech_min"]),
            ref_candidates=int(r["ref_candidates"]), ref_candidates_min=float(r["ref_candidates_min"])))
    return out


def validate_pool(pool: list[PoolSpeaker], nsc_dir: Path | None = None, *,
                  expected_splits: dict[str, int] = EXPECTED_SPLITS, expected_total: int = EXPECTED_TOTAL) -> list[str]:
    """Problems with the pool (empty list = valid). Zip presence is checked only when `nsc_dir` is given."""
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


def speaker_splits(path: Path, version: str) -> SpeakerSplits:
    """The run's speaker_splits.json content, copied from the pool CSV."""
    pool = load_pool(path)
    problems = validate_pool(pool)
    if problems:
        raise PoolError("; ".join(problems))
    speakers = {s.speaker: SplitSpeaker(
        split=s.split, gender=s.gender, age=s.age, ethnicity=s.ethnicity, first_language=s.first_language,
        languages=list(s.languages), partner=s.partner, education=s.education, relationship=s.relationship)
        for s in sorted(pool, key=lambda s: s.speaker)}
    return SpeakerSplits(pool_version=version, pool_path=str(path), pool_sha256=file_sha256(path), speakers=speakers,
                         counts=dict(sorted(Counter(s.split for s in pool).items())))
