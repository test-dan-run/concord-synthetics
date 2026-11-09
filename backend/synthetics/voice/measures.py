"""Stage 5's raw measurements per utterance audio, kept in `voice_measures.npz` beside voice_validation.json.

A stage 5 retry sends only some turns back to stage 4; the rest keep the same audio. Measuring them again (two
speaker embeddings, the ASR and the MOS model for every line) took most of each later round, so stage 5 reuses
what it measured before for audio it has seen: an embedding, a transcript or a MOS is reused only for the same
audio bytes (sha256) and the same service, backend mode and (for the ASR) language. Pass or fail is always decided
again from these numbers with the current thresholds, and the speaker and pair checks always use every line.

The file is a cache, not an output: no stage reads it as an input and nothing else uses it, so deleting it only
means the next stage 5 run measures everything. It holds the utterances of the latest synthesis only.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from ..artifacts.io import atomic_write_bytes

VERSION = 1


class Measures:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.utts: dict[str, dict] = {}  # sha -> {"emb": {key: vector}, "asr": {key: text}, "mos": {key: float}}
        self.reused = self.measured = 0
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"][()]))
                if meta.get("version") != VERSION:
                    return
                mats = {k: z[f"emb{i}"] for i, k in enumerate(meta["emb_keys"])}
            for sha, rec in meta["utts"].items():
                emb = {k: mats[k][row] for k, row in rec.pop("emb_rows", {}).items()}
                self.utts[sha] = {"emb": emb, "asr": rec.get("asr", {}), "mos": rec.get("mos", {})}
        except (OSError, ValueError, KeyError, IndexError):
            self.utts = {}  # an unreadable cache only costs a full measurement

    def _get(self, sha: str, kind: str, key: str, make):
        rec = self.utts.setdefault(sha, {"emb": {}, "asr": {}, "mos": {}})
        if key in rec[kind]:
            self.reused += 1
            return rec[kind][key]
        self.measured += 1
        rec[kind][key] = make()
        return rec[kind][key]

    def embedding(self, sha: str, key: str, make) -> np.ndarray:
        return self._get(sha, "emb", key, make)

    def transcript(self, sha: str, key: str, make) -> str:
        return self._get(sha, "asr", key, make)

    def mos(self, sha: str, key: str, make) -> float:
        return self._get(sha, "mos", key, make)

    def save(self, keep: set[str]) -> None:
        """Write the measurements of the audio in `keep` (the latest synthesis), dropping older takes."""
        emb_keys = sorted({k for sha in keep if sha in self.utts for k in self.utts[sha]["emb"]})
        rows: dict[str, list[np.ndarray]] = {k: [] for k in emb_keys}
        utts = {}
        for sha in sorted(keep & self.utts.keys()):
            rec = self.utts[sha]
            emb_rows = {}
            for k, vec in rec["emb"].items():
                emb_rows[k] = len(rows[k])
                rows[k].append(np.asarray(vec))
            utts[sha] = {"asr": rec["asr"], "mos": rec["mos"], "emb_rows": emb_rows}
        meta = {"version": VERSION, "emb_keys": emb_keys, "utts": utts}
        arrays = {f"emb{i}": np.stack(rows[k]) for i, k in enumerate(emb_keys)}
        buf = io.BytesIO()
        np.savez(buf, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
        atomic_write_bytes(self.path, buf.getvalue())
