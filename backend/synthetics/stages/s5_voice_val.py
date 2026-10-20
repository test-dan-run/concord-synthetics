"""Stage 5: voice validation (phase C). Writes voice_validation.json.

Per utterance:
- speaker similarity to the speaker's reference clip in both SV models, each against its own threshold
  (`thresholds.sv`), neither weighted above the other; skipped below `voice.sv_min_s`, and for backchannels and
  turns with no words (an "mm" or a laugh carries too little of the voice: real hums scored 0.07-0.54 against a
  0.56 threshold), which also keeps them out of the consistency and separability checks;
- ASR round trip: the primary ASR (WER on `text_norm` words, CER for short utterances); skipped when the turn has
  no words to check. A failure is re-synthesised like any other. The Singlish ASR is a fallback only when phase C
  leases `asr_singlish` (off by default: MERaLiON leaves about 1 GB of VRAM headroom and the bake-off never
  needed it); a pass there tags the utterance `asr_limited`;
- MOS floor: at least `thresholds.mos_min`, and at most `thresholds.mos_drop_max` below the MOS of the speaker's
  reference clip (UTMOS rates real NSC voices from about 1.3 to 3.3, so a fixed floor would reject whole voices);
  skipped below `voice.mos_min_s`.
Per speaker: within-speaker consistency (mean pairwise cosine of its utterances of at least
`voice.consistency_min_s`, the clip lengths its threshold was calibrated on) in both models; skipped with fewer
than `voice.consistency_min_utts` such utterances.
Per pair of speakers: mean cosine over all utterance pairs across the two, in both models; a pair is too close if
either model is above its `thresholds.sv_pair_max`. That threshold was fitted on single clip pairs (`calibrate-sv`),
so the pair figure stays on that scale: a centroid cosine reads about 1/sqrt(consistency_a * consistency_b) higher,
which flagged most real meetings.

Decision: failing utterances go back to stage 4 for re-synthesis with a new seed (up to `retries.tts_attempts`
per reference); a speaker that fails consistency, or the speaker with fewer turns in a too-close pair, gets a
reference swap (another clip of the same NSC speaker first, then another same-split speaker with a matching sheet
row). The retry carries {"resynth": [...], "swaps": [...]} as JSON feedback to stage 4. The meeting fails when an
utterance is still failing after its last attempt (`voice.on_exhausted: fail`) or a needed swap is not possible.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
from pydantic import BaseModel

from ..artifacts.io import read_model, write_model
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import Script, SpeakerSplits
from ..artifacts.voice import (
    AsrCheck,
    PairCheck,
    ReferenceSwap,
    SpeakerCheck,
    Synthesis,
    UtteranceCheck,
    VoiceValidation,
)
from ..backends.base import Audio
from ..config.schema import Phase, Role, RunConfig, StageId
from ..nsc.references import ReferenceIndex
from ..nsc.settings import nsc_paths
from ..text.singlish import Lexicon
from ..validators.voice_checks import asr_words, consistency, cross_cosine, score_asr
from .base import StageContext, StageResult

SV_ROLES = (Role.SV_WAVLM, Role.SV_REDIMNET)
LAUGH_WORDS = frozenset({"ha", "haha", "hahaha", "hehe", "heh"})


def asr_roles(cfg: RunConfig) -> tuple[Role, ...]:
    """The primary ASR, then the Singlish fallback if phase C leases it."""
    spec = cfg.phases.phases.get(Phase.C)
    return (Role.ASR_PRIMARY, Role.ASR_SINGLISH) if spec and Role.ASR_SINGLISH in spec.roles else (Role.ASR_PRIMARY,)


class Stage5Config(BaseModel):
    sv: dict[str, float]
    sv_consistency_min: dict[str, float]
    sv_pair_max: dict[str, float]
    wer_max: float
    cer_max_short: float
    short_utterance_words: int
    mos_min: float
    mos_drop_max: float
    tts_attempts: int
    asr_language: str | None
    wer_ignore_fillers: bool
    sv_min_s: float
    consistency_min_s: float
    consistency_min_utts: int
    mos_min_s: float
    on_exhausted: str
    max_speaker_swaps: int
    max_clip_swaps: int
    nonlexical_tts: dict[str, str]
    services: dict[str, str]
    lexicon_version: str
    refs_index: str


class VoiceValidationStage:
    id = StageId.VOICE_VAL
    name = "Voice checks"
    phase = Phase.C
    roles = (Role.ASR_PRIMARY, Role.ASR_SINGLISH, Role.SV_WAVLM, Role.SV_REDIMNET, Role.QUALITY)
    reads = ("meeting", "script", "synthesis", "utterances_dir", "speaker_splits")
    writes = ("voice_validation",)
    version = "5"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        th, v = cfg.thresholds, cfg.voice
        return Stage5Config(sv=th.sv, sv_consistency_min=th.sv_consistency_min, sv_pair_max=th.sv_pair_max,
                            wer_max=th.wer_max, cer_max_short=th.cer_max_short,
                            short_utterance_words=th.short_utterance_words, mos_min=th.mos_min,
                            mos_drop_max=th.mos_drop_max,
                            tts_attempts=cfg.retries.tts_attempts, asr_language=v.asr_language,
                            wer_ignore_fillers=v.wer_ignore_fillers, sv_min_s=v.sv_min_s, mos_min_s=v.mos_min_s,
                            consistency_min_s=v.consistency_min_s, consistency_min_utts=v.consistency_min_utts,
                            on_exhausted=v.on_exhausted, max_speaker_swaps=v.max_speaker_swaps,
                            max_clip_swaps=v.max_clip_swaps, nonlexical_tts=v.nonlexical_tts,
                            services={r.value: cfg.backends.roles.get(r, "")
                                      for r in (*asr_roles(cfg), *SV_ROLES, Role.QUALITY)},
                            lexicon_version=Lexicon.load(cfg.lexicon.path).version,
                            refs_index=ReferenceIndex.load(nsc_paths()).sha256)

    def run(self, ctx: StageContext) -> StageResult:
        cfg, th, vc, paths = ctx.cfg, ctx.cfg.thresholds, ctx.cfg.voice, ctx.paths
        m = read_model(paths.meeting, MeetingSpec)
        script = read_model(paths.script, Script)
        syn = read_model(paths.synthesis, Synthesis)
        splits = read_model(paths.artifact("speaker_splits"), SpeakerSplits)
        lex = Lexicon.load(cfg.lexicon.path)
        refs = ReferenceIndex.load(nsc_paths())
        turns = {t.turn_id: t for t in script.turns}
        b = ctx.backends
        asrs = [(r, b.get(r), cfg.backends.roles.get(r, "")) for r in asr_roles(cfg)]
        q = b.get(Role.QUALITY)
        embedders = {r.value: b.get(r) for r in SV_ROLES}
        laugh_ignore = LAUGH_WORDS | {w for t in vc.nonlexical_tts.values() for w in t.lower().split()}

        fake = cfg.backends.mode == "fake"
        ref_emb = {sid: {name: e.embed([refs.voice_ref(va.clip_id, fake=fake).audio])[0] for name, e in embedders.items()}
                   for sid, va in syn.voices.items()}
        ref_mos = {sid: round(float(q.score([refs.voice_ref(va.clip_id, fake=fake).audio])[0]), 3)
                   for sid, va in syn.voices.items()}
        swaps_for = {sid: sum(s.speaker_id == sid for s in syn.swaps) for sid in syn.voices}
        utt_emb: dict[str, dict[str, list[np.ndarray]]] = {sid: {n: [] for n in embedders} for sid in syn.voices}
        long_emb: dict[str, dict[str, list[np.ndarray]]] = {sid: {n: [] for n in embedders} for sid in syn.voices}

        checks: list[UtteranceCheck] = []
        for u in syn.utterances:
            turn = turns[u.turn_id]
            audio = Audio.load(paths.utterance(u.turn_id))
            reasons, skipped = [], []
            ref_words = asr_words(turn.text_norm, lex, ignore_fillers=vc.wer_ignore_fillers, normalised=True)
            sv: dict[str, float] = {}
            if audio.duration >= vc.sv_min_s and ref_words and turn.interaction != "backchannel":
                for name, e in embedders.items():
                    vec = e.embed([audio])[0]
                    utt_emb[u.speaker_id][name].append(vec)
                    if audio.duration >= vc.consistency_min_s:
                        long_emb[u.speaker_id][name].append(vec)
                    sv[name] = round(float(vec @ ref_emb[u.speaker_id][name]), 4)
                    if sv[name] < th.sv[name]:
                        reasons.append(f"{name} {sv[name]:.2f} < {th.sv[name]:.2f}")
            else:
                skipped.append("sv")

            asr: list[AsrCheck] = []
            limited, short = False, False
            if ref_words and u.source != "library":
                ignore = laugh_ignore if "laugh" in turn.nonspeech else frozenset()
                for role, client, service in asrs:
                    hyp = client.transcribe(audio, language=vc.asr_language)
                    s = score_asr(ref_words, hyp, lex, short_words=th.short_utterance_words,
                                  ignore_fillers=vc.wer_ignore_fillers, ignore=ignore)
                    short = s.short
                    ok = s.error <= (th.cer_max_short if s.short else th.wer_max)
                    asr.append(AsrCheck(role=role.value, service=service, hypothesis=hyp, wer=s.wer, cer=s.cer,
                                        passed=ok))
                    if ok:
                        limited = role == Role.ASR_SINGLISH
                        break
                if not asr[-1].passed:
                    rate = "CER" if short else "WER"
                    errors = " / ".join(str(a.cer if short else a.wer) for a in asr)
                    reasons.append(f"{rate} {errors}" + (" in both ASRs" if len(asr) > 1 else ""))
            else:
                skipped.append("asr")
                short = len(ref_words) <= th.short_utterance_words

            mos = None
            if audio.duration >= vc.mos_min_s:
                mos = round(float(q.score([audio])[0]), 3)
                floor = max(th.mos_min, ref_mos[u.speaker_id] - th.mos_drop_max)
                if mos < floor:
                    reasons.append(f"MOS {mos:.2f} < {floor:.2f}")
            else:
                skipped.append("mos")

            checks.append(UtteranceCheck(turn_id=u.turn_id, speaker_id=u.speaker_id, sv=sv, asr=asr,
                                         asr_limited=limited, short=short, mos=mos, attempts=u.attempt, seed=u.seed,
                                         passed=not reasons, reasons=reasons, skipped=skipped))

        speakers, embs_by = [], {}
        for sid in syn.voices:
            embs = {n: np.stack(v) for n, v in utt_emb[sid].items() if v}
            if not embs:
                continue
            long = {n: np.stack(v) for n, v in long_emb[sid].items() if len(v) >= vc.consistency_min_utts}
            cons = {n: round(consistency(e), 4) for n, e in long.items()}
            ok = all(cons[n] >= th.sv_consistency_min[n] for n in cons)
            n_long = min((len(e) for e in long.values()), default=0)
            speakers.append(SpeakerCheck(speaker_id=sid, consistency=cons, consistency_utts=n_long, passed=ok,
                                         ref_mos=ref_mos[sid]))
            embs_by[sid] = embs
        pairs = []
        for a, c in combinations(sorted(embs_by), 2):
            cos = {n: round(cross_cosine(embs_by[a][n], embs_by[c][n]), 4) for n in embs_by[a] if n in embs_by[c]}
            pairs.append(PairCheck(a=a, b=c, cross_cos=cos, too_close=any(cos[n] > th.sv_pair_max[n] for n in cos)))

        # ---------------------------------------------------------------- decision
        n_turns = {sid: sum(t.speaker_id == sid for t in script.turns) for sid in syn.voices}
        targets: list[tuple[str, str]] = [(s.speaker_id, "inconsistent across utterances") for s in speakers if not s.passed]
        for p in pairs:
            if p.too_close:
                worst = max(p.cross_cos, key=lambda n: p.cross_cos[n] - th.sv_pair_max[n])
                victim = min((p.a, p.b), key=lambda s: (n_turns[s], s))
                targets.append((victim, f"too close to {p.b if victim == p.a else p.a} ({worst} {p.cross_cos[worst]:.2f})"))
        new_swaps, unresolved, seen = [], [], set()
        for sid, reason in targets:
            if sid in seen:
                continue
            seen.add(sid)
            swap = self._swap(sid, reason, syn, new_swaps, refs, splits, vc.max_clip_swaps, vc.max_speaker_swaps)
            if swap is None:
                unresolved.append(sid)
            else:
                new_swaps.append(swap)

        swapped = {s.speaker_id for s in new_swaps}
        failed = [c for c in checks if not c.passed and c.speaker_id not in swapped]
        limit = {sid: cfg.retries.tts_attempts * (1 + swaps_for[sid]) for sid in syn.voices}
        exhausted = [c.turn_id for c in failed if c.attempts >= limit[c.speaker_id]]
        resynth = [c.turn_id for c in failed if c.attempts < limit[c.speaker_id]]

        kept: list[str] = []
        if unresolved or (exhausted and vc.on_exhausted == "fail"):
            decision = "fail"
        elif resynth or new_swaps:
            decision = "retry"
        else:
            decision = "pass"
            kept = exhausted
        vv = VoiceValidation(meeting_id=m.meeting_id, utterances=checks, speakers=speakers, pairs=pairs,
                             swaps=[*syn.swaps, *new_swaps] if decision == "retry" else list(syn.swaps),
                             resynth_turns=resynth if decision == "retry" else [],
                             decision=decision, kept_failed=kept)
        write_model(paths.voice_validation, vv)

        n = len(checks)
        n_fail = sum(not c.passed for c in checks)
        n_limited = sum(c.asr_limited for c in checks)
        metrics: dict[str, float | int | str] = {
            "utterances": n, "failed": n_fail, "asr_limited": n_limited, "resynth": len(resynth),
            "swaps_new": len(new_swaps), "swaps_total": len(vv.swaps), "pairs_too_close": sum(p.too_close for p in pairs),
            "resynthesised_ever": sum(c.attempts > 1 for c in checks)}
        for name in embedders:
            vals = [c.sv[name] for c in checks if name in c.sv]
            if vals:
                metrics[f"{name}_mean"] = round(float(np.mean(vals)), 4)
        mos_vals = [c.mos for c in checks if c.mos is not None]
        if mos_vals:
            metrics["mos_mean"] = round(float(np.mean(mos_vals)), 3)

        if decision == "pass":
            detail = f"{n} utterances pass, {n_limited} ASR-limited" + (f", {len(kept)} kept failing" if kept else "")
            return StageResult(state="pass", detail=detail, metrics=metrics)
        if decision == "retry":
            fb = json.dumps({"resynth": resynth, "swaps": [s.model_dump() for s in new_swaps]})
            detail = f"re-synthesise {len(resynth)} of {n}" + (f", swap {len(new_swaps)} reference(s)" if new_swaps else "")
            return StageResult(state="retry", detail=detail, metrics=metrics, feedback=fb, retry_stage=StageId.VOICE)
        why = []
        if exhausted:
            why.append(f"{len(exhausted)} utterance(s) still failing after the last attempt")
        if unresolved:
            why.append(f"no swap left for {', '.join(unresolved)}")
        return StageResult(state="fail", detail="; ".join(why), metrics=metrics)

    @staticmethod
    def _swap(sid: str, reason: str, syn: Synthesis, pending: list[ReferenceSwap], refs: ReferenceIndex,
              splits: SpeakerSplits, max_clip: int, max_speaker: int) -> ReferenceSwap | None:
        """A clip swap while the speaker has clip swaps left, else a speaker swap while the meeting has any."""
        history = [*syn.swaps, *pending]
        va = syn.voices[sid]
        clip_swaps = sum(s.speaker_id == sid and s.kind == "clip" for s in history)
        if clip_swaps < max_clip:
            used = {va.clip_id} | {c for s in history if s.speaker_id == sid for c in (s.from_clip, s.to_clip)}
            nxt = refs.next_clip(va.nsc_speaker, used)
            if nxt is not None:
                return ReferenceSwap(speaker_id=sid, kind="clip", from_speaker=va.nsc_speaker, to_speaker=va.nsc_speaker,
                                     from_clip=va.clip_id, to_clip=nxt.pair_id, reason=reason)
        if sum(s.kind == "speaker" for s in history) < max_speaker:
            in_use = {v.nsc_speaker for v in syn.voices.values()} | {x for s in history for x in (s.from_speaker, s.to_speaker)}
            new = refs.replacement(va.nsc_speaker, splits, in_use)
            if new is not None:
                return ReferenceSwap(speaker_id=sid, kind="speaker", from_speaker=va.nsc_speaker, to_speaker=new,
                                     from_clip=va.clip_id, to_clip=refs.best(new).pair_id, reason=reason)
        return None
