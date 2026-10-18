"""`synthetics nsc nonlexical`: real laughter, coughs, breaths and noises for the stage 4 sound library, cut from
the 175 pool speakers' NSC Part 3 segments whose transcript is only a non-speech code: (ppl) laugh, (ppc) cough,
(ppb) breath, (ppo) noise. Zips are read in place, as `nsc extract` does.

Each clip is dereverberated with WPE (`wpe-dereverb`), but not denoised (DeepFilterNet would remove breaths and much
of a laugh), trimmed, cut at MAX_S with a fade, and written to `<SOUND_LIBRARY_DIR>/nonlexical/<tag>/<pair_id>.wav`.
index.yaml records the NSC speaker, split and gender of every clip, so stage 4 can use a speaker's own laugh and
never another split's speaker.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from ..artifacts.io import atomic_write_text
from ..backends.base import Audio
from ..voice.audio_ops import trim
from .extract import wav_name
from .pool import load_pool
from .settings import NscPaths

CODES = {"ppl": "laugh", "ppc": "cough", "ppb": "breath", "ppo": "noise"}
ONLY_CODE = re.compile(r"^\s*\((ppl|ppc|ppb|ppo)\)\s*$", re.I)
MIN_S, MAX_S = 0.3, 3.0
FADE_S = 0.05
VERSION = "nl-nsc-1"
SOURCE = "IMDA National Speech Corpus Part 3 (SDM), segments transcribed as one non-speech code"
LICENCE = "IMDA NSC licence (same terms as the reference voices)"


def candidates(manifest: str) -> list[tuple[str, str, float]]:
    """(pair_id, tag, duration) for segments whose whole transcript is one non-speech code."""
    out = []
    for line in manifest.splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        m = ONLY_CODE.match(e.get("text", ""))
        if m:
            out.append((e["pair_id"], CODES[m.group(1).lower()], float(e["source_end"]) - float(e["source_start"])))
    return out


def shape(audio: Audio) -> Audio | None:
    """Trim; drop clips shorter than MIN_S; cut at MAX_S with a short fade-out."""
    a = trim(audio, pad_s=0.02, below_db=35.0)
    if a.duration < MIN_S:
        return None
    n = int(MAX_S * a.sr)
    if len(a.samples) > n:
        x = a.samples[:n].copy()
        f = int(FADE_S * a.sr)
        x[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        a = Audio(x, a.sr)
    return a


def extract_nonlexical(paths: NscPaths, out_root: Path, dereverb, *, speakers: list[str] | None = None,
                       log: Callable[[str], None] = print) -> dict:
    """`dereverb`: an Enhancer (or None to keep the audio as recorded)."""
    out_root = Path(out_root)
    index_path = out_root / "index.yaml"
    old = yaml.safe_load(index_path.read_text()) if index_path.is_file() else {}
    done = {c["id"]: c for c in (old or {}).get("clips") or [] if (old or {}).get("version") == VERSION}
    pool = [s for s in load_pool(paths.pool_csv) if not speakers or s.speaker in speakers]
    made = Counter()
    for i, sp in enumerate(pool, 1):
        with zipfile.ZipFile(paths.nsc_dir / sp.sdm_zip) as z:
            names = set(z.namelist())
            for pid, tag, dur in candidates(z.read(f"SPEAKER{sp.speaker}/manifest.json").decode("utf-8")):
                if pid in done or dur < MIN_S or wav_name(pid) not in names:
                    continue
                audio = Audio.load(io.BytesIO(z.read(wav_name(pid))))
                if dereverb is not None:
                    audio = dereverb.clean(audio).resample(audio.sr)
                clip = shape(audio)
                if clip is None:
                    continue
                rel = f"{tag}/{pid}.wav"
                clip.save(out_root / rel)
                done[pid] = {"id": pid, "tag": tag, "file": rel, "gender": sp.gender, "speaker": sp.speaker,
                             "split": sp.split, "duration": round(clip.duration, 3)}
                made[tag] += 1
        log(f"[{i}/{len(pool)}] {sp.speaker}: {sum(made.values())} clips so far")
    clips = sorted(done.values(), key=lambda c: c["id"])
    atomic_write_text(index_path, yaml.safe_dump({"version": VERSION, "source": SOURCE, "licence": LICENCE,
                                                  "processing": "WPE dereverb (wpe-dereverb), trim, cut at 3 s",
                                                  "clips": clips}, sort_keys=False, allow_unicode=True))
    return {"clips": len(clips), "new": dict(made), "by_tag": dict(Counter(c["tag"] for c in clips)),
            "index": str(index_path)}
