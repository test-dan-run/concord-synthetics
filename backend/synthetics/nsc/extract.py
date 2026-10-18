"""`synthetics nsc extract`: candidate reference segments from the 175 selected SDM zips, read in place.

A candidate is a 5-15 s segment with at least 8 words and no `#name#` redaction, `<...>` tag, `**` or `~`
(the same rule the pool selection used to count `ref_candidates`). Each is written as
`raw/<speaker>/<pair_id>.wav` (16 kHz mono, copied byte for byte from the zip) plus one line in
`raw/<speaker>/index.jsonl` with the NSC transcript and our notation. No zip is ever unpacked in full.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

from ..artifacts.io import atomic_write_bytes, atomic_write_text
from ..text.singlish import Lexicon, nsc_to_ours
from .pool import PoolSpeaker, load_pool
from .settings import NscPaths

BAD = re.compile(r"#|<|\*\*|\[unk|~", re.I)
MIN_S, MAX_S, MIN_WORDS = 5.0, 15.0, 8
EXTRACT_VERSION = "x1"


def is_candidate(entry: dict) -> bool:
    dur = float(entry["source_end"]) - float(entry["source_start"])
    text = entry.get("text", "")
    return MIN_S <= dur <= MAX_S and not BAD.search(text) and len(text.split()) >= MIN_WORDS


def wav_name(pair_id: str) -> str:
    return f"WAVE/{pair_id.replace('_', '')}.wav"


def extract_speaker(sp: PoolSpeaker, paths: NscPaths, lex: Lexicon) -> dict:
    out_dir = paths.raw_speaker(sp.speaker)
    index = out_dir / "index.jsonl"
    if index.is_file():
        meta = json.loads((out_dir / "_done.json").read_text()) if (out_dir / "_done.json").is_file() else {}
        if meta.get("version") == EXTRACT_VERSION:
            return {"speaker": sp.speaker, "skipped": True, "candidates": meta.get("candidates", 0)}
    rows = []
    with zipfile.ZipFile(paths.nsc_dir / sp.sdm_zip) as z:
        names = set(z.namelist())
        manifest = z.read(f"SPEAKER{sp.speaker}/manifest.json").decode("utf-8")
        for line in manifest.splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if not is_candidate(e) or wav_name(e["pair_id"]) not in names:
                continue
            try:
                ours = nsc_to_ours(e["text"], lex)
            except ValueError:
                continue  # #name# redactions (also excluded by BAD); never used as references
            atomic_write_bytes(out_dir / f"{e['pair_id']}.wav", z.read(wav_name(e["pair_id"])))
            rows.append({"pair_id": e["pair_id"], "speaker": sp.speaker, "conf": e.get("conf"),
                         "duration": round(float(e["source_end"]) - float(e["source_start"]), 3),
                         "text_nsc": e["text"], "text": ours, "wav": f"{e['pair_id']}.wav"})
    atomic_write_text(index, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    atomic_write_text(out_dir / "_done.json", json.dumps({"version": EXTRACT_VERSION, "candidates": len(rows),
                                                          "lexicon": lex.version}))
    return {"speaker": sp.speaker, "skipped": False, "candidates": len(rows)}


def extract_all(paths: NscPaths, *, speakers: list[str] | None = None, log: Callable[[str], None] = print) -> dict:
    from .pool import validate_pool

    pool = load_pool(paths.pool_csv)
    problems = validate_pool(pool, paths.nsc_dir)
    if problems:
        raise RuntimeError("pool invalid: " + "; ".join(problems))
    lex = Lexicon.load()
    todo = [s for s in pool if not speakers or s.speaker in speakers]
    results = []
    for i, sp in enumerate(todo, 1):
        r = extract_speaker(sp, paths, lex)
        results.append(r)
        log(f"[{i}/{len(todo)}] {sp.speaker}: {r['candidates']} candidates{' (already extracted)' if r['skipped'] else ''}")
    counts = [r["candidates"] for r in results]
    return {"speakers": len(results), "candidates": sum(counts), "min_per_speaker": min(counts) if counts else 0,
            "raw_dir": str(paths.raw)}


def load_candidates(paths: NscPaths, speaker: str) -> list[dict]:
    index = paths.raw_speaker(speaker) / "index.jsonl"
    if not index.is_file():
        return []
    return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]


def wav_path(paths: NscPaths, row: dict) -> Path:
    return paths.raw_speaker(row["speaker"]) / row["wav"]
