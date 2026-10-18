"""Stage 2: script generation (phase A). Writes script.json and prompts/<speaker_id>.txt.

1. Turn plan (script_plan): speakers and interaction types, fixed per meeting so regeneration rounds change
   only the words.
2. Meeting brief and one stance per speaker from the generator.
3. Persona prompt per speaker (versioned template), saved verbatim; the system prompt is the shared rules plus
   every persona prompt.
4. The generator writes the plan in chunks (structured JSON), seeing the recent turns as context.
5. Each turn is brought into the transcription convention (`text`, `text_norm`, disfluencies, non-speech).
"""

from __future__ import annotations

import json
import re

import numpy as np
from pydantic import BaseModel

from ..artifacts.io import atomic_write_text, read_model, write_model
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import Disfluency, GeneratorInfo, Script, Turn
from ..backends.base import BackendError, ChatLLM, Msg
from ..config.schema import Phase, Role, RunConfig, ScriptConfig, StageId, TurnTakingConfig
from ..text import pdpa
from ..text.singlish import Lexicon, analyse, canonical_text, ends_with_cutoff, text_norm
from . import prompts
from .base import StageContext, StageResult
from .script_plan import PlannedTurn, build_plan

CHUNK_RETRIES = 3


class Stage2Config(BaseModel):
    script: ScriptConfig
    turn_taking: TurnTakingConfig
    lexicon_version: str
    service: str


def known_names(cfg: RunConfig) -> list[str]:
    names = {n for by_gender in cfg.personas.names.values() for ns in by_gender.values() for n in ns}
    return sorted(names | {s.strip() for ss in cfg.personas.surnames.values() for s in ss})


def plan_rng(cfg: RunConfig, meeting_seed: int) -> np.random.Generator:
    """The plan depends on the meeting only, never on the round."""
    return np.random.default_rng(np.random.SeedSequence([cfg.seed, meeting_seed, int(StageId.SCRIPT), 0]))


def _line(t: Turn | PlannedTurn, text: str | None = None) -> str:
    return f"{t.speaker_id} [{t.interaction}]: {text if text is not None else getattr(t, 'text', '')}"


def apply_cutoffs(texts: list[str], plan: list[PlannedTurn]) -> int:
    """The turn before an interruption ends cut off with a trailing hyphen; returns how many were fixed."""
    fixed = 0
    for i in range(1, len(plan)):
        if plan[i].interaction != "interruption":
            continue
        j = i - 1
        while j >= 0 and plan[j].interaction in ("backchannel", "side"):
            j -= 1
        if j < 0 or ends_with_cutoff(texts[j]):
            continue
        texts[j] = cut_off(texts[j])
        fixed += 1
    return fixed


_TRAILING = re.compile(r"(\s*(\([^()]*\)|\[[^\[\]]*\]|[.!?,;:\"'”’…-]))+\s*$")


def cut_off(text: str) -> str:
    """End the turn mid-word: trailing particles, fillers, tags and punctuation go, then a hyphen."""
    t = _TRAILING.sub("", text).rstrip()
    return t + "-" if t else text


def to_turn(i: int, planned: PlannedTurn, raw: str, lex: Lexicon) -> Turn:
    text = canonical_text(raw, lex)
    mk = analyse(text, lex)
    return Turn(turn_id=f"t{i + 1:04d}", speaker_id=planned.speaker_id, interaction=planned.interaction, text=text,
                text_norm=text_norm(text, lex), disfluencies=[Disfluency(type=k, token=v) for k, v in mk.disfluencies],
                nonspeech=mk.nonspeech)


class ScriptStage:
    id = StageId.SCRIPT
    name = "Script"
    phase = Phase.A
    roles = (Role.GENERATOR,)
    reads = ("meeting",)
    writes = ("script", "prompts_dir")
    version = "4"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        return Stage2Config(script=cfg.script, turn_taking=cfg.turn_taking, lexicon_version=Lexicon.load(cfg.lexicon.path).version,
                            service=cfg.backends.roles.get(Role.GENERATOR, ""))

    def run(self, ctx: StageContext) -> StageResult:
        from .runner import meeting_seed

        cfg, sc = ctx.cfg, ctx.cfg.script
        lex = Lexicon.load(cfg.lexicon.path)
        m = read_model(ctx.paths.meeting, MeetingSpec)
        llm: ChatLLM = ctx.backends.get(Role.GENERATOR)
        plan = build_plan(m, cfg.turn_taking, plan_rng(cfg, meeting_seed(cfg.seed, m.meeting_id)))
        roster = [p.persona.display_name for p in m.people]
        feedback = pdpa.strip_identifiers(ctx.feedback, roster_names=roster, known_names=known_names(cfg)) \
            if ctx.feedback else None

        # brief + stances
        brief_msg = prompts.BRIEF[sc.template_version].format(
            meeting_type=m.dynamics.meeting_type, topic=m.dynamics.topic, duration=m.dynamics.duration_min,
            n=m.n_speakers, hybrid=(", some joining remotely" if m.dynamics.hybrid else ""),
            presentation=(" It includes a slide presentation followed by Q&A." if m.dynamics.presentation else ""),
            roster=prompts.roster_lines(m))
        res = llm.chat([Msg("user", brief_msg)], json_schema=prompts.BRIEF_SCHEMA, temperature=sc.temperature,
                       seed=ctx.seed, max_tokens=sc.max_tokens, thinking=sc.thinking)
        brief_out = res.json()
        brief = pdpa.strip_identifiers(brief_out["brief"], roster_names=roster, known_names=known_names(cfg))
        stances = {s["speaker_id"]: s["stance"] for s in brief_out.get("stances", []) if isinstance(s, dict)}

        # persona prompts, saved verbatim
        total_main = len(plan.main_turns())
        persona_text = {}
        for p in m.people:
            stance = stances.get(p.speaker_id) or "Follow the discussion and contribute from your role."
            stance = pdpa.strip_identifiers(stance, roster_names=roster, known_names=known_names(cfg))
            persona_text[p.speaker_id] = prompts.render_persona(
                m, p, version=sc.template_version, stance=stance, total_turns=total_main,
                target_turns=plan.targets.get(p.speaker_id, 0))
        ctx.paths.prompts_dir.mkdir(parents=True, exist_ok=True)
        for sid, text in persona_text.items():
            atomic_write_text(ctx.paths.prompt(sid), text)
        system = prompts.SCRIPT_SYSTEM[sc.template_version].format(
            brief=brief, personas="\n".join(persona_text[s] for s in sorted(persona_text)))

        # chunked generation
        texts: list[str] = []
        n_chunks = 0
        seg = m.dynamics.presentation
        for start in range(0, len(plan.turns), sc.chunk_turns):
            chunk = plan.turns[start:start + sc.chunk_turns]
            context_turns = [(plan.turns[k], texts[k]) for k in range(max(0, start - sc.context_turns), start)]
            context = ""
            if context_turns:
                context = "Turns so far (most recent last):\n" + "\n".join(_line(t, x) for t, x in context_turns) + "\n\n"
            phase_note = ""
            if start == 0:
                phase_note = " This is the opening of the meeting."
            elif start + len(chunk) >= len(plan.turns):
                phase_note = " The meeting wraps up at the end of this part."
            if seg is not None and any(t.interaction == "presentation" for t in chunk):
                phase_note += " The presentation is in progress in this part."
            plan_lines = "\n".join(f"{start + k + 1} | {t.speaker_id} | {t.interaction}" for k, t in enumerate(chunk))
            fb = f"\nFeedback on the previous version of this script (address it):\n{feedback}\n" if feedback else ""
            user = prompts.CHUNK_USER[sc.template_version].format(
                context=context, first=start + 1, last=start + len(chunk), total=len(plan.turns),
                phase_note=phase_note, plan=plan_lines, feedback=fb)
            texts.extend(self._chunk(llm, system, user, chunk, start, ctx, sc))
            n_chunks += 1

        fixed = apply_cutoffs(texts, plan.turns)
        turns = [to_turn(i, t, x, lex) for i, (t, x) in enumerate(zip(plan.turns, texts, strict=True))]
        script = Script(meeting_id=m.meeting_id, brief=brief, turns=turns, generator=GeneratorInfo(
            service=llm.service, template_version=sc.template_version, thinking=sc.thinking, seed=ctx.seed,
            temperature=sc.temperature, lexicon_version=lex.version, round=ctx.attempt, chunks=n_chunks))
        write_model(ctx.paths.script, script)
        return StageResult(state="pass", detail=f"{len(turns)} turns · round {ctx.attempt}",
                           metrics={"turns": len(turns), "round": ctx.attempt, "chunks": n_chunks,
                                    "cutoffs_added": fixed, "interruption_rate": round(plan.interruption_rate, 4),
                                    "backchannels_per_min": round(plan.backchannels_per_min, 3)})

    def _chunk(self, llm: ChatLLM, system: str, user: str, chunk: list[PlannedTurn], start: int, ctx: StageContext,
               sc: ScriptConfig) -> list[str]:
        max_bc_words = ctx.cfg.thresholds.script_rules.max_words_backchannel
        """One chunk of turns; the reply must match the plan line for line. Retries with a correction note."""
        note = ""
        last_err = ""
        for attempt in range(CHUNK_RETRIES):
            res = llm.chat([Msg("system", system), Msg("user", user + note)], json_schema=prompts.chunk_schema(len(chunk)),
                           temperature=sc.temperature, seed=ctx.seed + 7919 * (start + 1) + attempt,
                           max_tokens=sc.max_tokens, thinking=sc.thinking)
            try:
                rows = res.json()["turns"]
                by_i = {int(r["i"]): r for r in rows}
                out = []
                for k, t in enumerate(chunk):
                    r = by_i.get(start + k + 1) or (rows[k] if k < len(rows) else None)
                    if r is None or r.get("speaker_id") != t.speaker_id or not str(r.get("text", "")).strip():
                        raise ValueError(f"turn {start + k + 1}: expected {t.speaker_id} with text, got "
                                         f"{json.dumps(r, ensure_ascii=False)[:120] if r else 'nothing'}")
                    out.append(str(r["text"]).strip())
                long_bc = [start + k + 1 for k, (t, x) in enumerate(zip(chunk, out, strict=True))
                           if t.interaction == "backchannel" and len(text_norm(x).split()) > max_bc_words]
                if long_bc and attempt < CHUNK_RETRIES - 1:
                    raise ValueError(f"backchannel turns {', '.join(map(str, long_bc))} are longer than "
                                     f"{max_bc_words} words; a backchannel is a brief reaction such as (mm), ya, ok (lah)")
                return out
            except (BackendError, KeyError, TypeError, ValueError) as exc:
                last_err = str(exc)
                note = (f"\n\nYour previous reply did not follow the plan ({last_err}). Return exactly one entry per "
                        "plan line, same index and speaker, each with non-empty text, and keep backchannels brief.")
                ctx.log.warning("%s chunk at %d: %s", ctx.paths.meeting_id, start + 1, last_err)
        raise BackendError(f"chunk starting at turn {start + 1} did not follow the plan after {CHUNK_RETRIES} tries: "
                           f"{last_err}")
