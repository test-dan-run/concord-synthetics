"""Backend roles (§3.4): one small Protocol per kind of model, plus the shared value types.

Each role has an `Aicore*` implementation (platform service), a `Fake*` (deterministic, for unit tests) and
works under the record/replay wrapper. Stages only ever see these protocols.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


class BackendError(Exception):
    pass


@dataclass(frozen=True, eq=False)
class Audio:
    """float32 mono samples at `sr`."""

    samples: np.ndarray
    sr: int

    def __post_init__(self) -> None:
        a = np.asarray(self.samples, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError(f"Audio must be mono (1-D), got shape {a.shape}")
        object.__setattr__(self, "samples", a)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    def resample(self, sr: int) -> Audio:
        if sr == self.sr:
            return self
        import soxr

        return Audio(soxr.resample(self.samples, self.sr, sr).astype(np.float32), sr)

    def save(self, path: Path, subtype: str = "PCM_16") -> None:
        import soundfile as sf

        from ..artifacts.io import atomic_write_bytes

        buf = io.BytesIO()
        sf.write(buf, self.samples, self.sr, format="WAV", subtype=subtype)
        atomic_write_bytes(Path(path), buf.getvalue())

    def wav_bytes(self) -> bytes:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, self.samples, self.sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @classmethod
    def load(cls, path_or_file: Any) -> Audio:
        import soundfile as sf

        data, sr = sf.read(path_or_file, dtype="float32", always_2d=True)
        return cls(data.mean(axis=1), int(sr))


# ---------------------------------------------------------------- chat


@dataclass(frozen=True)
class Msg:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResult:
    text: str  # final answer, reasoning removed
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""

    def json(self) -> Any:
        """The answer parsed as JSON (tolerates a fenced block)."""
        import json

        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendError(f"{self.model}: answer is not valid JSON ({exc}); finish={self.finish_reason}; "
                               f"head={self.text[:200]!r}") from None


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float
    score: float


@dataclass(frozen=True)
class VoiceRef:
    """A clean reference clip and its transcript (our notation), for continuation cloning."""

    audio: Audio
    text: str
    clip_id: str = ""


# ---------------------------------------------------------------- roles


@runtime_checkable
class ChatLLM(Protocol):
    service: str

    def chat(self, messages: list[Msg], *, json_schema: dict | None = None, temperature: float, seed: int,
             max_tokens: int, thinking: bool | None = None) -> ChatResult: ...


@runtime_checkable
class VoiceCloner(Protocol):
    def clone(self, text: str, ref: VoiceRef, *, seed: int) -> Audio: ...


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, audio: Audio, *, language: str | None = None) -> str: ...


@runtime_checkable
class SpeakerEmbedder(Protocol):
    name: str

    def embed(self, clips: Sequence[Audio]) -> np.ndarray: ...  # (n, d), L2-normalised


@runtime_checkable
class QualityScorer(Protocol):
    def score(self, clips: Sequence[Audio]) -> np.ndarray: ...  # MOS-like, (n,)


@runtime_checkable
class Aligner(Protocol):
    def align(self, audio: Audio, text: str, *, language: str) -> list[WordTiming]: ...


@runtime_checkable
class Enhancer(Protocol):
    def clean(self, audio: Audio) -> Audio: ...  # denoise or dereverb
