"""Stage 4: voice synthesis (phase C). Writes synthesis.json and one utterances/<turn_id>.wav per script turn.

Each speaker is cloned from one clean reference clip of the NSC speaker assigned in stage 1 (the best clip by MOS),
kept for every utterance. A turn's `text` becomes TTS input (parentheses removed, non-speech tags split out):
speech is cloned piece by piece (split at sentence ends past `voice.max_tts_chars`), and each [tag] becomes a
sound-library clip, a cloned stand-in ("haha" for [laugh]) or a synthetic sound. The pieces are joined with short
gaps, resampled to the run rate and trimmed to at most `voice.trim_pad_s` of silence at each end.

A retry from stage 5 arrives as JSON feedback: {"resynth": [turn ids], "swaps": [ReferenceSwap...]}. Only those
turns, plus every turn of a swapped speaker, are made again (with a new seed); the rest of the previous synthesis
is kept.
"""

from __future__ import annotations

import json
import re

import numpy as np
from pydantic import BaseModel

from ..artifacts.io import file_sha256, read_model, write_model
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import Script, Turn
from ..artifacts.voice import ReferenceSwap, Synthesis, UtteranceSynth, VoiceAssignment
from ..backends.base import Audio, VoiceRef
from ..config.schema import Phase, Role, RunConfig, StageId
from ..nsc.references import ReferenceIndex
from ..nsc.settings import nsc_paths
from ..text.singlish import Lexicon, tts_input
from ..voice.audio_ops import join, match_level, rms, trim
from ..voice.nonlexical import SYNTH_VERSION, SoundLibrary, synthetic
from .base import StageContext, StageResult


class Stage4Config(BaseModel):
    sample_rate: int
    tts: str
    tts_mode: str
    cfg_value: float
    inference_timesteps: int
    max_tts_chars: int
    piece_gap_s: float
    trim_pad_s: float
    trim_db: float
    nonlexical_tts: dict[str, str]
    lexicon_version: str
    refs_index: str
    sound_library: str


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def split_text(text: str, max_chars: int) -> list[str]:
    """Pieces of at most `max_chars`, cut at sentence ends, then clause ends, then spaces."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    def pack(parts: list[str]) -> list[str]:
        out, cur = [], ""
        for p in parts:
            if cur and len(cur) + 1 + len(p) > max_chars:
                out.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}".strip()
        return [*out, cur] if cur else out

    pieces: list[str] = []
    for sent in _SENTENCE_END.split(text):
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        for clause in pack(_CLAUSE_END.split(sent)):
            pieces.extend(pack(clause.split()) if len(clause) > max_chars else [clause])
    return pack(pieces)


def utterance_seed(meeting_seed: int, turn_index: int, attempt: int) -> int:
    return int(np.random.SeedSequence([meeting_seed, turn_index, attempt]).generate_state(1, dtype=np.uint32)[0]
               & 0x7FFFFFFF)


def parse_feedback(feedback: str | None) -> tuple[set[str], list[ReferenceSwap]]:
    if not feedback:
        return set(), []
    data = json.loads(feedback)
    return set(data.get("resynth") or []), [ReferenceSwap.model_validate(s) for s in data.get("swaps") or []]


def assignment(speaker_id: str, refs: ReferenceIndex, clip_id: str) -> VoiceAssignment:
    c = refs.clip(clip_id)
    return VoiceAssignment(speaker_id=speaker_id, nsc_speaker=c.speaker, clip_id=c.pair_id, ref_text=c.text,
                           ref_duration=round(c.duration, 3), ref_mos=c.mos)


class VoiceStage:
    id = StageId.VOICE
    name = "Voices"
    phase = Phase.C
    roles = (Role.TTS,)
    reads = ("meeting", "script")
    writes = ("synthesis", "utterances_dir")
    version = "1"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        v = cfg.voice
        return Stage4Config(sample_rate=cfg.sample_rate, tts=cfg.backends.roles.get(Role.TTS, ""), tts_mode=v.tts_mode,
                            cfg_value=v.cfg_value, inference_timesteps=v.inference_timesteps,
                            max_tts_chars=v.max_tts_chars, piece_gap_s=v.piece_gap_s, trim_pad_s=v.trim_pad_s,
                            trim_db=v.trim_db, nonlexical_tts=v.nonlexical_tts,
                            lexicon_version=Lexicon.load(cfg.lexicon.path).version,
                            refs_index=ReferenceIndex.load(nsc_paths()).sha256,
                            sound_library=f"{SoundLibrary.load().version}+{SYNTH_VERSION}")

    def run(self, ctx: StageContext) -> StageResult:
        cfg, vc, paths = ctx.cfg, ctx.cfg.voice, ctx.paths
        m = read_model(paths.meeting, MeetingSpec)
        script = read_model(paths.script, Script)
        sha = file_sha256(paths.script)
        lex = Lexicon.load(cfg.lexicon.path)
        refs = ReferenceIndex.load(nsc_paths())
        lib = SoundLibrary.load()
        resynth, new_swaps = parse_feedback(ctx.feedback)

        prev = read_model(paths.synthesis, Synthesis) if paths.synthesis.is_file() else None
        if prev is not None and (prev.script_sha256 != sha or prev.meeting_id != m.meeting_id):
            prev = None
        if prev is None:
            voices = {p.speaker_id: assignment(p.speaker_id, refs, refs.best(p.nsc_speaker.speaker).pair_id)
                      for p in m.people}
            swaps: list[ReferenceSwap] = []
            old: dict[str, UtteranceSynth] = {}
        else:
            voices, swaps = dict(prev.voices), list(prev.swaps)
            old = {u.turn_id: u for u in prev.utterances}
        for s in new_swaps:
            voices[s.speaker_id] = assignment(s.speaker_id, refs, s.to_clip)
            swaps.append(s)
        swapped = {s.speaker_id for s in new_swaps}

        cloner = ctx.backends.get(Role.TTS)
        if hasattr(cloner, "params"):  # the platform client; fakes ignore these settings
            cloner.params = {"mode": vc.tts_mode, "cfg_value": vc.cfg_value, "inference_timesteps": vc.inference_timesteps}
        voice_refs: dict[str, VoiceRef] = {}
        gender = {p.speaker_id: p.nsc_speaker.gender for p in m.people}
        pick = {sid: {"speaker": va.nsc_speaker, "split": m.split} for sid, va in voices.items()}
        paths.utterances_dir.mkdir(parents=True, exist_ok=True)

        utterances, made = [], 0
        tts_sr = prev.tts_sample_rate if prev else 0
        for i, turn in enumerate(script.turns):
            prior = old.get(turn.turn_id)
            redo = (prior is None or turn.turn_id in resynth or turn.speaker_id in swapped
                    or not paths.utterance(turn.turn_id).is_file())
            if not redo:
                utterances.append(prior)
                continue
            attempt = prior.attempt + 1 if prior else 1
            seed = utterance_seed(m.seed, i, attempt)
            sid = turn.speaker_id
            if sid not in voice_refs:
                voice_refs[sid] = refs.voice_ref(voices[sid].clip_id, fake=cfg.backends.mode == "fake")
            audio, source, tts_text, clips, sr_used = self._render(turn, voice_refs[sid], cloner, lib, lex,
                                                                   gender.get(sid), pick[sid], seed,
                                                                   f"{m.meeting_id}|{turn.turn_id}", ctx)
            tts_sr = sr_used or tts_sr
            audio.save(paths.utterance(turn.turn_id))
            utterances.append(UtteranceSynth(turn_id=turn.turn_id, speaker_id=sid, source=source, tts_text=tts_text,
                                             library_clips=clips, seed=seed, attempt=attempt,
                                             duration=round(audio.duration, 4)))
            made += 1

        syn = Synthesis(meeting_id=m.meeting_id, service=cfg.backends.roles.get(Role.TTS, ""),
                        tts_sample_rate=int(tts_sr), sample_rate=cfg.sample_rate, voices=voices,
                        utterances=utterances, swaps=swaps, script_sha256=sha)
        write_model(paths.synthesis, syn)
        total = sum(u.duration for u in utterances)
        detail = f"{made} of {len(utterances)} utterances synthesised, {total / 60:.1f} min"
        if new_swaps:
            detail += f", {len(new_swaps)} reference swap(s)"
        return StageResult(state="pass", detail=detail,
                           metrics={"utterances": len(utterances), "synthesised": made, "swaps": len(swaps),
                                    "speech_min": round(total / 60, 2)})

    def _render(self, turn: Turn, ref: VoiceRef, cloner, lib: SoundLibrary, lex: Lexicon, gender: str | None,
                pick: dict, seed: int, key: str, ctx: StageContext) -> tuple[Audio, str, str, list[str], int]:
        """(trimmed audio at the run rate, source, text given to the cloner, library clip ids, TTS sample rate)."""
        vc, sr = ctx.cfg.voice, ctx.cfg.sample_rate
        pieces: list[tuple[str, Audio]] = []  # ("speech" | "sound", audio at the run rate)
        tts_texts: list[str] = []
        clips: list[str] = []
        n_clone, tts_sr = 0, 0

        def clone(text: str) -> Audio:
            nonlocal n_clone, tts_sr
            out = cloner.clone(text, ref, seed=(seed + 7919 * n_clone) & 0x7FFFFFFF)
            n_clone += 1
            tts_sr = out.sr
            tts_texts.append(text)
            return trim(out.resample(sr), pad_s=vc.trim_pad_s, below_db=vc.trim_db)

        for k, seg in enumerate(tts_input(turn.text, lex)):
            if seg.kind == "speech":
                for chunk in split_text(seg.value, vc.max_tts_chars):
                    pieces.append(("speech", clone(chunk)))
                continue
            tag = seg.value
            picked = lib.pick(tag, gender, f"{key}|{k}", **pick)
            if picked is not None:
                clip_id, a = picked
                pieces.append(("sound", trim(a.resample(sr), pad_s=0.0, below_db=vc.trim_db)))
                clips.append(clip_id)
            elif tag in vc.nonlexical_tts:
                pieces.append(("speech", clone(vc.nonlexical_tts[tag])))
            else:
                pieces.append(("sound", synthetic(tag, sr, (seed + 104729 * (k + 1)) & 0x7FFFFFFF)))
                clips.append(f"synthetic:{tag}")
        if not pieces:
            raise ValueError(f"turn {turn.turn_id} has nothing to synthesise: {turn.text!r}")

        speech = [a for kind, a in pieces if kind == "speech"]
        level = float(np.mean([rms(a) for a in speech])) if speech else rms(ref.audio)
        audio = join([a if kind == "speech" else match_level(a, level) for kind, a in pieces], gap_s=vc.piece_gap_s)
        audio = trim(audio, pad_s=vc.trim_pad_s, below_db=vc.trim_db)
        has_tts, has_lib = bool(tts_texts), bool(clips)
        source = "tts+library" if has_tts and has_lib else ("library" if has_lib else "tts")
        return audio, source, " ".join(tts_texts), clips, tts_sr
