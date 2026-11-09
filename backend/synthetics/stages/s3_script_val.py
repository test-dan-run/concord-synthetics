"""Stage 3: script validation (phase B, tie-break in B2). Writes script_validation.json.

Programmatic checks first; then the primary judge (rationale before scores); if its overall score is within
the borderline band of the pass mark, the tie-breaker scores independently with the same rubric. Agreement
decides; disagreement fails the script. A failure goes back to stage 2 with feedback (the scheduler drops the
meeting once the round limit is reached).

When a judge's service is not leased in the current phase, the stage defers: the partial result (rules,
primary judgement) is kept in script_validation.json and reused when the stage runs again in a later phase.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from ..artifacts.io import file_sha256, read_model, write_model
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import JudgeResult, Script, ScriptValidation
from ..config.schema import JudgeConfig, Phase, Role, RunConfig, ScriptRuleThresholds, StageId, TurnTakingConfig
from ..text.singlish import run_lexicon
from ..validators.judges import judge_script, low_criteria
from ..validators.script_rules import check_script
from .base import StageContext, StageResult

# Settings added after runs were frozen, with the default that reproduces those runs: the cache key leaves a setting
# out while it holds that default, so older runs' stage 3 stays a cache hit
ADDED_SETTINGS: tuple[tuple[tuple[str, ...], object], ...] = (
    (("judge_criterion_floor",), None),
    (("rules", "min_words_per_planned_min"), None),
    (("rules", "max_repeated_phrase_share"), None),
    (("rules", "max_self_introductions"), None),
)


class Stage3Config(BaseModel):
    judge: JudgeConfig
    judge_pass: float
    judge_band: float
    judge_criterion_floor: float | None = None
    rules: ScriptRuleThresholds
    turn_taking: TurnTakingConfig
    lexicon_version: str
    primary: str
    tiebreak: str

    @model_serializer(mode="wrap")
    def _without_unused_additions(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        for path, default in ADDED_SETTINGS:
            node = data
            for k in path[:-1]:
                node = node[k]
            if node.get(path[-1], default) == default:
                node.pop(path[-1], None)
        return data


class ScriptValidationStage:
    id = StageId.SCRIPT_VAL
    name = "Validation"
    phase = Phase.B
    roles = (Role.JUDGE_PRIMARY, Role.JUDGE_TIEBREAK)
    reads = ("meeting", "script")
    writes = ("script_validation",)
    version = "3"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        return Stage3Config(judge=cfg.judge, judge_pass=cfg.thresholds.judge_pass, judge_band=cfg.thresholds.judge_band,
                            judge_criterion_floor=cfg.thresholds.judge_criterion_floor,
                            rules=cfg.thresholds.script_rules, turn_taking=cfg.turn_taking,
                            lexicon_version=run_lexicon(cfg).version,
                            primary=cfg.backends.roles.get(Role.JUDGE_PRIMARY, ""),
                            tiebreak=cfg.backends.roles.get(Role.JUDGE_TIEBREAK, ""))

    def run(self, ctx: StageContext) -> StageResult:
        cfg, th = ctx.cfg, ctx.cfg.thresholds
        m = read_model(ctx.paths.meeting, MeetingSpec)
        script = read_model(ctx.paths.script, Script)
        sha = file_sha256(ctx.paths.script)
        rnd = script.generator.round
        lex = run_lexicon(cfg)

        prev = None
        if ctx.paths.script_validation.is_file():
            old = read_model(ctx.paths.script_validation, ScriptValidation)
            if old.script_sha256 == sha and old.round == rnd:
                prev = old

        rules = check_script(script, m, cfg, lex)
        rules_ok = all(r.passed for r in rules)
        base = {"meeting_id": m.meeting_id, "round": rnd, "script_sha256": sha, "rules": rules, "rules_passed": rules_ok}
        if not rules_ok:
            failed = [r for r in rules if not r.passed]
            fb = "Programmatic checks failed: " + "; ".join(
                f"{r.name} = {r.value} (limit {r.limit}){' — ' + r.detail if r.detail else ''}" for r in failed)
            write_model(ctx.paths.script_validation, ScriptValidation(**base, decision="fail", feedback=fb))
            return StageResult(state="retry", detail=f"rules failed: {', '.join(r.name for r in failed)}"[:120],
                               metrics={"round": rnd, "rules_failed": len(failed)}, feedback=fb,
                               retry_stage=StageId.SCRIPT)

        primary = prev.primary if prev else None
        if primary is None:
            if not ctx.backends.has(Role.JUDGE_PRIMARY):
                write_model(ctx.paths.script_validation, ScriptValidation(**base, decision="pending"))
                return StageResult(state="defer", detail="rules ok · awaiting primary judge", metrics={"round": rnd})
            primary = judge_script(ctx.backends.get(Role.JUDGE_PRIMARY), Role.JUDGE_PRIMARY.value, m, script, cfg.judge,
                                   seed=ctx.seed, pass_mark=th.judge_pass,
                                   criterion_floor=th.judge_criterion_floor)
        borderline = abs(primary.scores.overall - th.judge_pass) <= th.judge_band
        tiebreak: JudgeResult | None = prev.tiebreak if prev else None
        if borderline and tiebreak is None:
            if not ctx.backends.has(Role.JUDGE_TIEBREAK):
                write_model(ctx.paths.script_validation,
                            ScriptValidation(**base, primary=primary, borderline=True, decision="pending"))
                return StageResult(state="defer", detail=f"judge {primary.scores.overall:.1f} · awaiting tie-break",
                                   metrics={"round": rnd, "judge_overall": primary.scores.overall})
            tiebreak = judge_script(ctx.backends.get(Role.JUDGE_TIEBREAK), Role.JUDGE_TIEBREAK.value, m, script,
                                    cfg.judge, seed=ctx.seed + 1, pass_mark=th.judge_pass,
                                    criterion_floor=th.judge_criterion_floor)

        if borderline:
            assert tiebreak is not None
            passed = primary.passed and tiebreak.passed  # agreement decides; disagreement fails
        else:
            passed = primary.passed
        feedback = None
        if not passed:
            parts = [primary.feedback] + ([tiebreak.feedback] if tiebreak and not tiebreak.passed else [])
            if th.judge_criterion_floor is not None:
                low = low_criteria(primary.scores, th.judge_criterion_floor)
                parts.append(f"These scores were too low: {', '.join(low)}." if low else "")
            fallback = ("Make the conversation sound more natural." if cfg.judge.rubric_version == "rubric_v3"
                        else "Make the conversation sound more natural and Singaporean.")
            feedback = "\n".join(p for p in parts if p) or fallback
        write_model(ctx.paths.script_validation, ScriptValidation(
            **base, primary=primary, tiebreak=tiebreak if borderline else None, borderline=borderline,
            decision="pass" if passed else "fail", feedback=feedback))

        scores = f"judge {primary.scores.overall:.1f}"
        if borderline and tiebreak is not None:
            scores += f" · tie-break {tiebreak.scores.overall:.1f}"
        metrics = {"round": rnd, "judge_overall": primary.scores.overall, "borderline": int(borderline)}
        if tiebreak is not None:
            metrics["tiebreak_overall"] = tiebreak.scores.overall
        if passed:
            retries = rnd - 1
            return StageResult(state="pass", detail=f"{scores} · {retries} retr{'y' if retries == 1 else 'ies'}",
                               metrics=metrics)
        return StageResult(state="retry", detail=f"{scores} · regenerate", metrics=metrics, feedback=feedback,
                           retry_stage=StageId.SCRIPT)
