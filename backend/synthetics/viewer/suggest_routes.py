"""Natural-language suggestions for the run builder: `POST /api/suggest` and `POST /api/suggest/apply`.

A request ("mostly big boardroom meetings, noisy, lots of interrupting") goes to the decision model, which answers
the questions in `suggest/questions.yaml`; confident answers come back as builder overrides merged onto the draft,
less confident ones as offers, and risky ones as questions. Nothing is created or started: the caller previews the
new draft like any other. With no decision service on the platform the routes fall back to the word-matching fake
and say so (`model: "fake"`), so the flow still works in fake mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
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


USE_SERVICE_ENV = "SYNTHETICS_DECISION_SERVICE"


def decision_model(*, use_service: bool | None = None, service: str | None = None):
    """(model, name): the `decision` role's service (or `service`) when it is switched on
    (`SYNTHETICS_DECISION_SERVICE=1`) and the platform has it, else the fake."""
    import os

    from ..backends.registry import Backends, check_endpoints
    from ..config.loader import load_run_config
    from ..suggest.models import FakeDecisionModel

    if use_service is None:
        use_service = os.environ.get(USE_SERVICE_ENV, "") not in ("", "0", "false")
    overrides = {"run_id": "suggest", "seed": 1, "n_meetings": 1}
    if service:
        overrides["backends.roles.decision"] = service
    cfg = load_run_config(overrides=overrides).backends
    if use_service and cfg.mode != "fake" and not check_endpoints(cfg, [Role.DECISION]):
        return Backends(cfg, [Role.DECISION]).get(Role.DECISION), cfg.roles[Role.DECISION]
    return FakeDecisionModel(), "fake"


def _merged(draft: Draft, overrides: dict[str, Any]) -> dict:
    return {**draft.model_dump(), "overrides": {**draft.overrides, **overrides}}


TOPIC_BANK = Path(__file__).resolve().parents[1] / "config/defaults/topic_bank.yaml"


def topic_bank() -> dict[str, list[str]]:
    import yaml

    return yaml.safe_load(TOPIC_BANK.read_text(encoding="utf-8"))["topics"]


def add_routes(app: FastAPI, *, model_factory=decision_model) -> None:
    @app.get("/api/topics")
    def suggest_topics(n: int = Query(5, ge=1, le=20), seed: int = Query(0, ge=0),
                       exclude: list[str] = Query(default_factory=list)) -> dict:
        """`n` topics from the curated bank, spread over its themes, leaving out `exclude` (the user's list).
        `seed` picks another set for "suggest others"."""
        import numpy as np

        taken = {t.strip().lower() for t in exclude}
        bank = {theme: [t for t in ts if t.lower() not in taken] for theme, ts in topic_bank().items()}
        rng = np.random.default_rng(seed)
        order = {theme: list(rng.permutation(ts)) for theme, ts in bank.items()}
        out: list[dict] = []
        while len(out) < n and any(order.values()):
            for theme in sorted(order):
                if order[theme] and len(out) < n:
                    out.append({"topic": str(order[theme].pop()), "theme": theme})
        return {"topics": out}

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
