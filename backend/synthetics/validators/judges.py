"""LLM judges for stage 3: the same rubric for the primary judge and the tie-breaker."""

from __future__ import annotations

from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import JudgeResult, JudgeScores, Script, Turn
from ..backends.base import ChatLLM, Msg
from ..config.schema import JudgeConfig
from ..stages import prompts


def excerpt(turns: list[Turn], limit: int) -> tuple[list[Turn], str]:
    """Whole script when it fits; otherwise the opening, middle and closing thirds of `limit` turns."""
    if len(turns) <= limit:
        return turns, ""
    k = limit // 3
    mid = len(turns) // 2
    picked = turns[:k] + turns[mid - k // 2: mid - k // 2 + k] + turns[-(limit - 2 * k):]
    return picked, "; opening, middle and closing excerpts"


def judge_script(llm: ChatLLM, role: str, m: MeetingSpec, script: Script, cfg: JudgeConfig, *, seed: int,
                 pass_mark: float) -> JudgeResult:
    shown, note = excerpt(script.turns, cfg.max_script_turns)
    lines = "\n".join(f"{t.turn_id} {t.speaker_id} [{t.interaction}]: {t.text}" for t in shown)
    user = prompts.JUDGE_USER[cfg.rubric_version].format(
        meeting_type=m.dynamics.meeting_type, topic=m.dynamics.topic, n=m.n_speakers,
        personas=prompts.judge_persona_lines(m), shown=len(shown), total=len(script.turns), excerpt_note=note,
        lines=lines)
    res = llm.chat([Msg("system", prompts.JUDGE_SYSTEM[cfg.rubric_version]), Msg("user", user)],
                   json_schema=prompts.judge_schema(), temperature=cfg.temperature, seed=seed,
                   max_tokens=cfg.max_tokens, thinking=cfg.thinking)
    out = res.json()
    ids = {p.speaker_id for p in m.people}
    persona = {str(r["speaker_id"]): _clip(r["score"]) for r in out.get("persona_consistency", [])
               if isinstance(r, dict) and str(r.get("speaker_id")) in ids}
    scores = JudgeScores(coherence=_clip(out["coherence"]), singlish=_clip(out["singlish"]),
                         realism=_clip(out["realism"]), persona_consistency=dict(sorted(persona.items())),
                         overall=_clip(out["overall"]))
    return JudgeResult(service=llm.service, role=role, rubric_version=cfg.rubric_version,
                       rationale=str(out.get("rationale", "")).strip(), scores=scores,
                       feedback=str(out.get("feedback", "")).strip(), passed=scores.overall >= pass_mark)


def _clip(x: object) -> float:
    return round(min(5.0, max(1.0, float(x))), 2)
