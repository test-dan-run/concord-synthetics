"""Artifact models besides meeting.json: script, script validation, status, speaker splits, run process."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .meeting import Split

Interaction = Literal["normal", "interruption", "backchannel", "overlap_start", "side", "presentation"]
StageState = Literal["pending", "running", "pass", "fail", "retry", "skipped", "dropped"]


class _A(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- script.json (stage 2)


class Disfluency(_A):
    type: Literal["filler", "false_start", "repetition", "cutoff"]
    token: str


class Turn(_A):
    turn_id: str  # t0001 ...
    speaker_id: str
    interaction: Interaction
    text: str  # cased, punctuated; particles / fillers in parentheses, non-speech in square brackets
    text_norm: str  # lowercase, no punctuation or markup; used for WER and training
    disfluencies: list[Disfluency] = Field(default_factory=list)
    nonspeech: list[str] = Field(default_factory=list)  # laugh, cough, breath, noise


class GeneratorInfo(_A):
    service: str
    role: str = "generator"
    template_version: str
    thinking: bool
    seed: int
    temperature: float
    lexicon_version: str
    round: int
    chunks: int


class Script(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    brief: str  # the meeting brief shared by every persona prompt
    turns: list[Turn]
    generator: GeneratorInfo


# ---------------------------------------------------------------- script_validation.json (stage 3)


class RuleCheck(_A):
    name: str
    passed: bool
    value: float | int | str | None = None
    limit: float | int | str | None = None
    detail: str = ""


class JudgeScores(_A):
    coherence: float
    singlish: float
    realism: float  # realistic messiness, rewarded over polish
    persona_consistency: dict[str, float]  # speaker_id -> 1..5
    overall: float


class JudgeResult(_A):
    service: str
    role: str
    rubric_version: str
    rationale: str  # written before the scores
    scores: JudgeScores
    feedback: str
    passed: bool


class ScriptValidation(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    round: int
    script_sha256: str
    rules: list[RuleCheck]
    rules_passed: bool
    primary: JudgeResult | None = None
    tiebreak: JudgeResult | None = None
    borderline: bool = False
    decision: Literal["pass", "fail", "pending"]  # pending: a judge still has to run in a later phase
    feedback: str | None = None  # sent back to stage 2 on a failure


# ---------------------------------------------------------------- status.json


class StageStatus(_A):
    state: StageState = "pending"
    attempts: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    cache_key: str | None = None
    detail: str = ""  # the one-line detail on the UI stage tile
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    feedback: str | None = None  # waiting for this stage on its next attempt (from a downstream check)


class MeetingStatus(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    stages: dict[int, StageStatus] = Field(default_factory=dict)

    def get(self, stage: int) -> StageStatus:
        return self.stages.get(int(stage)) or StageStatus()


# ---------------------------------------------------------------- speaker_splits.json


class SplitSpeaker(_A):
    split: Split
    gender: Literal["F", "M"]
    age: str
    ethnicity: str
    first_language: str
    languages: list[str]
    partner: str | None
    education: str
    relationship: str


class SpeakerSplits(_A):
    schema_version: Literal["1"] = "1"
    pool_version: str
    pool_path: str
    pool_sha256: str
    speakers: dict[str, SplitSpeaker]
    counts: dict[str, int]

    def in_split(self, split: str) -> list[str]:
        return sorted(s for s, v in self.speakers.items() if v.split == split)


# ---------------------------------------------------------------- process.json (a run started from the viewer)

ProcessState = Literal["running", "finished", "failed", "stopped"]


class RunProcess(_A):
    """The background `run` the viewer started. Only that process and the viewer write it; the pipeline never
    reads it, so it is not part of any cache key."""

    schema_version: Literal["1"] = "1"
    pid: int
    state: ProcessState
    live: bool  # leases GPU services (backends.mode aicore or record)
    started_at: str
    ended_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
