"""Clean NSC reference clips for stage 4 (§4.3): `nsc-part3-refs/clean/index.jsonl`, written once by
`synthetics nsc clean` and shared by every run.

Each speaker keeps its top few clips by MOS. Stage 4 clones every utterance of a speaker from one clip (the best
by default); swaps go to the next clip of the same speaker, then to another same-split speaker whose sheet row
matches (gender and ethnicity always, first language and age band when possible).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..artifacts.models import SpeakerSplits
from ..backends.base import Audio, VoiceRef
from .settings import NscPaths

AGE_ORDER = ["1X", "2X", "3X", "4X", "5X", "6X"]


@dataclass(frozen=True)
class RefClip:
    speaker: str
    pair_id: str  # NSC pair id <speaker>_<nnnn>; the clip id in synthesis.json
    path: Path
    text: str  # transcript in our notation (the cloning prompt)
    duration: float
    mos: float | None
    gender: str
    age: str
    ethnicity: str
    first_language: str
    split: str
    raw_path: Path | None = None  # the same segment before cleanup (bake-off room-colour check)


class ReferenceIndex:
    def __init__(self, clips: Iterable[RefClip], sha256: str = ""):
        self.sha256 = sha256  # of index.jsonl: part of the stage 4 / 5 cache keys
        self._by_speaker: dict[str, list[RefClip]] = {}
        self._by_id: dict[str, RefClip] = {}
        for c in clips:
            self._by_speaker.setdefault(c.speaker, []).append(c)
            self._by_id[c.pair_id] = c
        for v in self._by_speaker.values():
            v.sort(key=lambda c: (-(c.mos if c.mos is not None else 0.0), c.pair_id))

    @classmethod
    def load(cls, paths: NscPaths) -> ReferenceIndex:
        return _load(str(paths.clean_index), str(paths.refs_dir))

    def speakers(self) -> list[str]:
        return sorted(self._by_speaker)

    def clips(self, speaker: str) -> list[RefClip]:
        return list(self._by_speaker.get(speaker, []))

    def clip(self, pair_id: str) -> RefClip:
        try:
            return self._by_id[pair_id]
        except KeyError:
            raise KeyError(f"reference clip {pair_id} is not in the clean index") from None

    def best(self, speaker: str) -> RefClip:
        clips = self._by_speaker.get(speaker)
        if not clips:
            raise KeyError(f"NSC speaker {speaker} has no clean reference clips")
        return clips[0]

    def next_clip(self, speaker: str, used: Iterable[str]) -> RefClip | None:
        """The best clip of `speaker` not in `used` (clip ids)."""
        used = set(used)
        return next((c for c in self._by_speaker.get(speaker, []) if c.pair_id not in used), None)

    def replacement(self, speaker: str, splits: SpeakerSplits, exclude: Iterable[str]) -> str | None:
        """Another speaker of the same split with the same gender and ethnicity (first language and age band
        preferred), not in `exclude` and with clean clips. Deterministic: ties go to the lowest speaker id."""
        me = splits.speakers[speaker]
        exclude = set(exclude) | {speaker}

        def rank(s: str) -> tuple:
            o = splits.speakers[s]
            age_gap = abs(AGE_ORDER.index(o.age) - AGE_ORDER.index(me.age)) if o.age in AGE_ORDER and me.age in AGE_ORDER else 9
            return (o.first_language != me.first_language, age_gap, s)

        pool = [s for s, o in splits.speakers.items()
                if s not in exclude and o.split == me.split and o.gender == me.gender and o.ethnicity == me.ethnicity
                and self._by_speaker.get(s)]
        return min(pool, key=rank) if pool else None

    def voice_ref(self, pair_id: str, *, fake: bool = False) -> VoiceRef:
        """`fake` (fake backends): the clip's fake voice in place of its audio. The fake speaker embedder reads the
        voice from tones in the audio, so a fake run on real NSC references would otherwise fail stage 5."""
        c = self.clip(pair_id)
        if fake:
            from ..backends.fakes_audio import fake_reference

            return VoiceRef(audio=fake_reference(c.pair_id, n_words=5), text=c.text, clip_id=c.pair_id)
        return VoiceRef(audio=_audio(str(c.path)), text=c.text, clip_id=c.pair_id)


@lru_cache(maxsize=4)
def _load(index_path: str, refs_dir: str) -> ReferenceIndex:
    path = Path(index_path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `synthetics nsc clean` (M2) first")
    raw = path.read_bytes()
    clips = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        clips.append(RefClip(speaker=r["speaker"], pair_id=r["pair_id"], path=Path(refs_dir) / r["clip"], text=r["text"],
                             duration=float(r["duration"]), mos=r.get("mos"), gender=r.get("gender", ""),
                             age=r.get("age", ""), ethnicity=r.get("ethnicity", ""),
                             first_language=r.get("first_language", ""), split=r.get("split", ""),
                             raw_path=Path(refs_dir) / r["raw_clip"] if r.get("raw_clip") else None))
    return ReferenceIndex(clips, hashlib.sha256(raw).hexdigest()[:16])


@lru_cache(maxsize=64)
def _audio(path: str) -> Audio:
    return Audio.load(path)
