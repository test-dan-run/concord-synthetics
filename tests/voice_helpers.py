"""A fake `nsc-part3-refs/` (clean clips for every pool speaker) for the phase C tests."""

from __future__ import annotations

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
