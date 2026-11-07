"""Natural-language suggestions for the run builder: `POST /api/suggest` and `POST /api/suggest/apply`.

A request ("mostly big boardroom meetings, noisy, lots of interrupting") goes to the decision model, which answers
the questions in `suggest/questions.yaml`; confident answers come back as builder overrides merged onto the draft,
less confident ones as offers, and risky ones as questions. Nothing is created or started: the caller previews the
new draft like any other. With no decision service on the platform the routes fall back to the word-matching fake
and say so (`model: "fake"`), so the flow still works in fake mode.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config.schema import Role
from ..suggest.engine import Answer, apply_answers, suggest
from ..suggest.questions import load_questions
from .builder import Draft, build_config


class SuggestIn(BaseModel):
    request: str = Field(min_length=1, max_length=2000)
    draft: Draft = Field(default_factory=Draft)


class Choice(BaseModel):
    group: str
    option: str
    scope: str = "both"


class ApplyIn(BaseModel):
    draft: Draft = Field(default_factory=Draft)
    choices: list[Choice]


def decision_model():
    """(model, name): the `decision` role's service when the platform has it, else the fake."""
    from ..backends.registry import Backends, check_endpoints
    from ..config.loader import load_run_config
    from ..suggest.models import FakeDecisionModel

    cfg = load_run_config(overrides={"run_id": "suggest", "seed": 1, "n_meetings": 1}).backends
    if cfg.mode != "fake" and not check_endpoints(cfg, [Role.DECISION]):
        return Backends(cfg, [Role.DECISION]).get(Role.DECISION), cfg.roles[Role.DECISION]
    return FakeDecisionModel(), "fake"


def _merged(draft: Draft, overrides: dict[str, Any]) -> dict:
    return {**draft.model_dump(), "overrides": {**draft.overrides, **overrides}}


def add_routes(app: FastAPI, *, model_factory=decision_model) -> None:
    @app.post("/api/suggest")
    def post_suggest(body: SuggestIn) -> dict:
        cfg, _ = build_config(body.draft)
        model, name = model_factory()
        s = suggest(model, body.request, cfg.model_dump(mode="json"))
        qs = load_questions()
        notes = {gq.id: gq.on_yes for gq in qs.global_ if gq.on_yes}
        return {**s.as_dict(), "notes": [{"id": n, "text": notes.get(n, "")} for n in s.notes],
                "draft": _merged(body.draft, s.overrides), "model": name}

    @app.post("/api/suggest/apply")
    def post_apply(body: ApplyIn) -> dict:
        qs = load_questions()
        answers = []
        for c in body.choices:
            try:
                g = qs.group(c.group)
                opt = g.option(c.option)
            except StopIteration:
                raise HTTPException(422, f"no option {c.option!r} in group {c.group!r}") from None
            answers.append(Answer(g.id, opt.id, opt.label, 1.0, c.scope, "user"))
        cfg, _ = build_config(body.draft)
        changes, overrides = apply_answers(answers, cfg.model_dump(mode="json"), qs)
        return {"changes": [c.__dict__ for c in changes], "overrides": overrides,
                "draft": _merged(body.draft, overrides)}
