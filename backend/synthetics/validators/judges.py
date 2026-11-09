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
                 pass_mark: float, criterion_floor: float | None = None) -> JudgeResult:
    shown, note = excerpt(script.turns, cfg.max_script_turns)
    lines = "\n".join(f"{t.turn_id} {t.speaker_id} [{t.interaction}]: {t.text}" for t in shown)
    user = prompts.JUDGE_USER[cfg.rubric_version].format(
        meeting_type=m.dynamics.meeting_type, topic=m.dynamics.topic, n=m.n_speakers,
        personas=prompts.judge_persona_lines(m), shown=len(shown), total=len(script.turns), excerpt_note=note,
        lines=lines)
    res = llm.chat([Msg("system", prompts.JUDGE_SYSTEM[cfg.rubric_version]), Msg("user", user)],
                   json_schema=prompts.judge_schema(cfg.rubric_version), temperature=cfg.temperature, seed=seed,
                   max_tokens=cfg.max_tokens, thinking=cfg.thinking)
    out = res.json()
    ids = {p.speaker_id for p in m.people}
    persona = {str(r["speaker_id"]): _clip(r["score"]) for r in out.get("persona_consistency", [])
               if isinstance(r, dict) and str(r.get("speaker_id")) in ids}
    scores = JudgeScores(**{c: _clip(out[c]) for c in prompts.JUDGE_CRITERIA[cfg.rubric_version]},
                         persona_consistency=dict(sorted(persona.items())), overall=_clip(out["overall"]))
    return JudgeResult(service=llm.service, role=role, rubric_version=cfg.rubric_version,
                       rationale=str(out.get("rationale", "")).strip(), scores=scores,
                       feedback=str(out.get("feedback", "")).strip(),
                       passed=passes(scores, pass_mark, criterion_floor))


def passes(scores: JudgeScores, pass_mark: float, criterion_floor: float | None = None) -> bool:
    """Overall at or above the pass mark and, with a floor, no criterion (each speaker's persona score included)
    at or below the floor."""
    if scores.overall < pass_mark:
        return False
    if criterion_floor is None:
        return True
    return not low_criteria(scores, criterion_floor)


def low_criteria(scores: JudgeScores, floor: float) -> list[str]:
    """The criteria at or below `floor`, as "realism 2.0" or "persona S03 2.0", for the writer's feedback."""
    low = [f"{k.replace('_', ' ')} {v:.1f}" for k, v in scores.criteria().items() if v <= floor]
    return low + [f"persona {s} {v:.1f}" for s, v in scores.persona_consistency.items() if v <= floor]


def _clip(x: object) -> float:
    return round(min(5.0, max(1.0, float(x))), 2)
