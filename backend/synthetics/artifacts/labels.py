"""Artifact models for stages 6-8: `timeline.json`, `labels/transcript.json` and `manifest.jsonl` entries.

Times are seconds from the start of the meeting audio unless a field says otherwise. Every segment plays its
utterance from the beginning; a truncated segment stops `truncated_at` seconds into the utterance, so
`end - start` is always the audible length.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .meeting import PositionType, Split
from .models import Interaction


class _A(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- timeline.json (stage 6)


class Segment(_A):
    turn_id: str
    speaker_id: str
    kind: Interaction
    start: float
    end: float
    truncated_at: float | None = None  # seconds into the utterance where it was cut; None = played in full


class TimelineStats(_A):
    duration: float
    speech_time: float  # sum of segment lengths
    overlap_share: float  # share of speech time with two or more speakers active
    backchannels_per_min: float
    interruption_share: float  # interruptions / speaker changes
    long_pauses: int


class Timeline(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    sample_rate: int
    segments: list[Segment]  # sorted by start
    stats: TimelineStats


# ---------------------------------------------------------------- labels/transcript.json (stage 8)


class Word(_A):
    word: str
    start: float
    end: float
    score: float  # aligner confidence


class TranscriptUtterance(_A):
    turn_id: str
    speaker_id: str
    start: float
    end: float
    text: str  # what is audible: words after a truncation point are removed
    text_norm: str
    truncated: bool
    words: list[Word]
    align_coverage: float  # share of text_norm words that got a timing


class Transcript(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    aligner: str  # service name
    utterances: list[TranscriptUtterance]  # sorted by start


# ---------------------------------------------------------------- manifest.jsonl (stage 8, one line per meeting)


class ManifestSpeaker(_A):
    speaker_id: str
    nsc_speaker: str  # after any stage 4 speaker swap
    split: Split
    position_type: PositionType
    wall_reason: Literal["overflow", "by_choice"] | None = None
    turn_share: float  # realised share of speech time
    asr_limited: int  # utterances that passed only through the Singlish ASR


Scalar = float | int | str | bool | None


class ManifestEntry(_A):
    schema_version: Literal["1"] = "1"
    run_id: str
    meeting_id: str
    seed: int
    split: Split
    room_class: Literal["small", "large"]
    n_speakers: int
    duration: float
    sampled: dict[str, Scalar]  # every sampled parameter, flattened with dotted keys (room.length, mic.type, ...)
    stages_enabled: dict[str, bool]  # stage name -> toggle (ablations)
    position_counts: dict[str, int]
    speakers: list[ManifestSpeaker]
    judge: dict[str, Scalar]  # primary / tie-break overall, rounds to pass
    voice: dict[str, Scalar]  # pass, re-synthesis and ASR-limited counts, swaps
    turn_taking: TimelineStats
    files: dict[str, str] = Field(default_factory=dict)  # artifact key -> path relative to the run dir
