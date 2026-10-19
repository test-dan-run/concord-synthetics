"""Artifact models for phase C: `synthesis.json` (stage 4) and `voice_validation.json` (stage 5).

Contract with stage 6: stage 4 writes exactly one `utterances/<turn_id>.wav` per script turn, mono, at the
run sample rate, trimmed so at most `voice.trim_pad_s` of silence is left at each end. `Synthesis.utterances`
lists them in script order with their durations. Stage 6 reads only passed meetings (stage 5 `pass`), so every
listed utterance is usable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _A(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- synthesis.json (stage 4)


class ReferenceSwap(_A):
    speaker_id: str
    kind: Literal["clip", "speaker"]  # another clip of the same NSC speaker, or another same-split speaker
    from_speaker: str  # NSC speaker ids
    to_speaker: str
    from_clip: str  # clip ids (NSC pair_id, e.g. 00860519_0845)
    to_clip: str
    reason: str  # e.g. "too close to S04 (sv_wavlm 0.91)"


class VoiceAssignment(_A):
    speaker_id: str
    nsc_speaker: str  # differs from meeting.json only after a speaker swap
    clip_id: str  # the one reference clip used for every utterance of this speaker
    ref_text: str  # its transcript in our notation (the cloning prompt)
    ref_duration: float
    ref_mos: float | None = None


class UtteranceSynth(_A):
    turn_id: str
    speaker_id: str
    source: Literal["tts", "library", "tts+library"]  # library: sound-library clips for non-lexical tags
    tts_text: str  # what the cloner was given (derived from the turn's `text`); "" for library-only turns
    library_clips: list[str] = Field(default_factory=list)  # sound-library ids, in order
    seed: int
    attempt: int  # 1 = first synthesis; >1 after a stage 5 re-synthesis request
    duration: float  # seconds of the trimmed WAV at the run sample rate


class Synthesis(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    service: str  # tts service name
    tts_sample_rate: int
    sample_rate: int  # run rate; every utterance WAV is at this rate
    voices: dict[str, VoiceAssignment]  # speaker_id -> assignment
    utterances: list[UtteranceSynth]  # one per script turn, script order
    swaps: list[ReferenceSwap] = Field(default_factory=list)
    script_sha256: str = ""  # the script these utterances were made from


# ---------------------------------------------------------------- voice_validation.json (stage 5)


class AsrCheck(_A):
    role: Literal["asr_primary", "asr_singlish"]
    service: str
    hypothesis: str
    wer: float | None = None
    cer: float | None = None  # used instead of WER for short utterances
    passed: bool


class UtteranceCheck(_A):
    turn_id: str
    speaker_id: str
    sv: dict[str, float]  # SV role (sv_wavlm, sv_redimnet) -> cosine to the speaker's reference; equal standing
    asr: list[AsrCheck]  # primary first; the Singlish fallback only when the primary fails
    asr_limited: bool  # passed only through the fallback ASR
    short: bool  # judged by CER (thresholds.short_utterance_words)
    mos: float | None = None
    attempts: int
    seed: int
    passed: bool
    reasons: list[str] = Field(default_factory=list)  # why it failed, one short phrase each
    skipped: list[str] = Field(default_factory=list)  # checks not applied: "sv" (too short, a backchannel or no
    # words), "mos" (too short), "asr" (no words)


class SpeakerCheck(_A):
    speaker_id: str
    consistency: dict[str, float]  # SV role -> mean pairwise cosine between this speaker's utterances
    passed: bool
    ref_mos: float | None = None  # MOS of the speaker's reference clip: the utterance MOS floors are relative to it


class PairCheck(_A):
    a: str
    b: str
    centroid_cos: dict[str, float]  # SV role -> cosine between the two speakers' utterance centroids
    too_close: bool  # either model above its thresholds.sv_pair_max


class VoiceValidation(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    utterances: list[UtteranceCheck]
    speakers: list[SpeakerCheck]
    pairs: list[PairCheck]
    swaps: list[ReferenceSwap] = Field(default_factory=list)
    resynth_turns: list[str] = Field(default_factory=list)  # sent back to stage 4 on a retry
    decision: Literal["pass", "fail", "retry"]
    kept_failed: list[str] = Field(default_factory=list)  # still failing after the last attempt, kept (voice.on_exhausted)
