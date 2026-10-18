"""Artifact models for stages 6-8: `timeline.json`, `render.json`, `labels/transcript.json` and `manifest.jsonl`
entries.

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


Scalar = float | int | str | bool | None


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
    overlap_share: float  # share of speech time with two or more speakers active (overlap_time / voiced_time)
    backchannels_per_min: float
    interruption_share: float  # interruptions / speaker changes
    long_pauses: int
    voiced_time: float = 0.0  # time with at least one speaker active
    overlap_time: float = 0.0  # time with two or more speakers active
    overlap_target: float | None = None  # the meeting's sampled target (turn_taking.overlap_target)
    speaker_changes: int = 0  # main-turn transitions between different speakers
    interruptions: int = 0  # realised: the floor speaker was cut off
    backchannels: int = 0


class Timeline(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    sample_rate: int
    segments: list[Segment]  # sorted by start
    stats: TimelineStats
    dropped_turns: list[str] = Field(default_factory=list)  # script turns left out (thinned backchannels)


# ---------------------------------------------------------------- render.json (stage 7)


class NoiseRender(_A):
    kind: str  # hvac | projector_fan | keyboard | door | chair | cup | paper
    source: str  # procedural | library:<clip id>
    position: tuple[float, float, float]
    level_db: float  # beds: share of the noise floor; keyboard and events: dB relative to the speech level
    start: float | None = None  # events only
    end: float | None = None


class AcousticRender(_A):
    """What stage 7 measured and applied, for the manifest and the viewer."""

    schema_version: Literal["1"] = "1"
    meeting_id: str
    sample_rate: int
    channels: int  # mixture_mc.wav channels (1: no multichannel file)
    duration: float
    files: dict[str, str]  # artifact key -> path relative to the meeting dir
    rt60_target: float
    rt60_measured: float  # median mid-band T20 over the talkers' impulse responses at mic 0
    rt60_error: float  # relative
    rt60_per_speaker: dict[str, float]
    absorption_scale: float
    calibration: list[float]  # probe RT60 at each calibration step
    ism_order: int
    ray_tracing: bool
    n_impulse_responses: int
    snr_target_db: float
    snr_db: float  # realised: speech over the noise beds, on the single-channel mixture before the device chain
    noise: list[NoiseRender]
    remote: dict[str, dict[str, float]] = Field(default_factory=dict)  # speaker -> far-end band and codec
    device: dict[str, Scalar] = Field(default_factory=dict)  # what the device chain applied


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
    acoustic: dict[str, Scalar] = Field(default_factory=dict)  # stage 7 measurements (render.json), flattened
    labels: dict[str, Scalar] = Field(default_factory=dict)  # stage 8: aligner, coverage, words, gates
