"""Fake voice data for tests: a fake  (clean clips for every pool speaker) and a voice folder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import ROOT
from synthetics.backends.fakes_audio import fake_reference
from synthetics.nsc.pool import load_pool


def fake_refs(root: Path, *, clips: int = 2, raw: bool = False) -> Path:
    """`root/clean/<speaker>/<pair_id>.wav` plus clean/index.jsonl, like `synthetics nsc clean` writes; with `raw`,
    a raw copy of each clip too (`raw/...`)."""
    lines = []
    for sp in load_pool(ROOT / "data/nsc_part3_reference_speakers.csv"):
        for k in range(clips):
            pid = f"{sp.speaker}_{k + 1:04d}"
            rel = f"clean/{sp.speaker}/{pid}.wav"
            a = fake_reference(pid, n_words=5)
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            a.save(root / rel)
            row = {"speaker": sp.speaker, "pair_id": pid, "clip": rel, "duration": round(a.duration, 3),
                   "text": "this is my reference clip lah", "mos": 3.5 - 0.1 * k, "gender": sp.gender, "age": sp.age,
                   "ethnicity": sp.ethnicity, "first_language": sp.first_language, "split": sp.split}
            if raw:
                (root / f"raw/{sp.speaker}").mkdir(parents=True, exist_ok=True)
                a.save(root / f"raw/{sp.speaker}/{pid}.wav")
                row["raw_clip"] = f"raw/{sp.speaker}/{pid}.wav"
            lines.append(row)
    (root / "clean/index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return root


SHEET = ["speaker", "conference_id", "gender", "age", "ethnicity", "education", "occupation", "languages",
         "first_language", "partner_id", "relationship", "synthetics_split", "candidates", "clips", "seconds",
         "mos_best", "mos_min"]


def voice_folder(root: Path, speakers: list[dict], *, clips: int = 5, mos: float = 3.2) -> Path:
    """A voice folder in the nsc-part3-voices layout with `clips` clips per speaker."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "speakers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET)
        w.writeheader()
        for s in speakers:
            w.writerow({k: s.get(k, "") for k in SHEET})
    lines = []
    for s in speakers:
        n = s.get("_clips", clips)
        for k in range(n):
            cid = f"{s['speaker']}_{k + 1:04d}"
            lines.append({"clip_id": cid, "speaker": s["speaker"], "rank": k + 1,
                          "audio": f"clean/{s['speaker']}/{cid}.wav", "duration": 6.0, "text": "okay lah",
                          "mos": round(mos - 0.1 * k, 3), "source_zip": f"SDM/SPEAKER{s['speaker']}.zip",
                          "gender": s["gender"], "age": s["age"], "ethnicity": s["ethnicity"],
                          "first_language": s["first_language"]})
    (root / "manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return root


def new_speaker(sid: str, **kw) -> dict:
    return {"speaker": sid, "gender": "F", "age": "2X", "ethnicity": "CHINESE", "first_language": "ENGLISH",
            "languages": "ENGLISH", "conference_id": "9000", **kw}
