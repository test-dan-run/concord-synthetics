"""Answer the decision questions for one request and sort the answers into changes, offers and questions.

The decision model gives a probability per option and never a value. For each group the request touches (its gate
says yes at `gate_min` or more, or a global bundle names it), the top option is:

- applied, when it is at least `apply_min` and the group or option is not marked for confirmation;
- offered with the runner-up, between `offer_min` and `apply_min`, so the user picks;
- asked, when the group is in `confirm_always` or the option has `confirm: true`.

An `unchanged` top option does nothing. A group the request names directly keeps its own answer over a bundle's.
The same engine serves the first description and every later chat message, since effects apply to the current
draft.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .effects import Change, apply_option
from .questions import Group, Option, QuestionSet, load_questions

YES_NO = ["Yes", "No"]


class DecisionModel(Protocol):
    def choose(self, request: str, question: str, options: list[str]) -> list[float]:
        """A probability per option, in order, summing to 1."""


def ask(model: DecisionModel, request: str, items: list[tuple[str, str, list[str]]]) -> dict[str, list[float]]:
    """Several questions at once: in one call when the model takes a batch (`choose_many`), else one by one."""
    many = getattr(model, "choose_many", None)
    if many is not None:
        return many(request, items)
    return {key: model.choose(request, question, options) for key, question, options in items}


@dataclass
class Answer:
    group: str
    option: str
    label: str
    prob: float
    scope: str = "both"
    source: str = "direct"  # "direct", or the global question whose bundle chose it


@dataclass
class Offer:
    group: str
    question: str
    options: list[dict]  # [{id, label, prob}], top two
    scope: str = "both"


@dataclass
class Suggestion:
    applied: list[Answer] = field(default_factory=list)
    offers: list[Offer] = field(default_factory=list)
    confirm: list[Offer] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    highlight_exact_numbers: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)  # every change, as dotted builder overrides

    def as_dict(self) -> dict:
        return asdict(self)


def _top(probs: list[float]) -> list[int]:
    return sorted(range(len(probs)), key=lambda i: -probs[i])


def _scope(qs: QuestionSet, probs: list[float]) -> str:
    best = _top(probs)[0]
    return qs.scope.options[best].id if probs[best] >= qs.policy.offer_min else qs.scope.default


def _offer(g: Group, probs: list[float], scope: str, n: int = 2) -> Offer:
    order = _top(probs)[:n]
    return Offer(g.id, g.question, [{"id": g.options[i].id, "label": g.options[i].label, "prob": round(probs[i], 3)}
                                    for i in order], scope)


def answer_groups(model: DecisionModel, request: str, qs: QuestionSet | None = None) -> Suggestion:
    """Ask every question for `request`; no changes are computed yet (see `suggest`).

    Two rounds: the global questions and every group's gate together, then the scope and the choice questions of
    the groups that are in play."""
    qs = qs or load_questions()
    p = qs.policy
    s = Suggestion()
    first = [(f"global:{gq.id}", gq.question, YES_NO if gq.options is None else [o.label for o in gq.options])
             for gq in qs.global_]
    first += [(f"gate:{g.id}", g.gate, YES_NO) for g in qs.groups]
    got = ask(model, request, first)

    bundles: dict[str, tuple[str, float, str]] = {}  # group -> (option, prob, global id)
    for gq in qs.global_:
        probs = got[f"global:{gq.id}"]
        if gq.options is None:
            if probs[0] >= p.gate_min:
                if gq.id == "exact_numbers":
                    s.highlight_exact_numbers = True
                if gq.on_yes:
                    s.notes.append(gq.id)
            continue
        best = _top(probs)[0]
        if probs[best] >= p.apply_min:
            for gid, oid in gq.options[best].bundle.items():
                bundles.setdefault(gid, (oid, probs[best], gq.id))

    direct = {g.id for g in qs.groups if got[f"gate:{g.id}"][0] >= p.gate_min}
    live = [g for g in qs.groups if g.id in direct or g.id in bundles]
    second = [(f"choice:{g.id}", g.question, [o.label for o in g.options]) for g in live if g.id in direct]
    if any(g.scope for g in live):
        second.append(("scope", qs.scope.question, [o.label for o in qs.scope.options]))
    got = ask(model, request, second)

    for g in live:
        scope = _scope(qs, got["scope"]) if g.scope else "both"
        if g.id in direct:
            probs = got[f"choice:{g.id}"]
            best = _top(probs)[0]
            opt, prob, source = g.options[best], probs[best], "direct"
        else:
            oid, prob, source = bundles[g.id]
            opt, probs = g.option(oid), None
        if opt.id == "unchanged":
            continue
        if g.id in p.confirm_always or opt.confirm:
            s.confirm.append(_offer(g, probs, scope) if probs else
                             Offer(g.id, g.question, [{"id": opt.id, "label": opt.label, "prob": prob}], scope))
        elif prob >= p.apply_min:
            s.applied.append(Answer(g.id, opt.id, opt.label, round(prob, 3), scope, source))
        elif prob >= p.offer_min and probs is not None:
            s.offers.append(_offer(g, probs, scope))
    return s


def apply_answers(answers: list[Answer], config: dict, qs: QuestionSet | None = None) -> tuple[list[Change],
                                                                                              dict[str, Any]]:
    """Changes and dotted overrides from chosen options, applied to a copy of `config` (a full config dump)."""
    qs = qs or load_questions()
    cfg = copy.deepcopy(config)
    changes: list[Change] = []
    for a in answers:
        opt: Option = qs.group(a.group).option(a.option)
        changes += apply_option(cfg, opt.effects, a.scope, qs, overlay=False)
        changes += apply_option(cfg, opt.overlay, a.scope, qs, overlay=True)
    overrides: dict[str, Any] = {}
    for c in changes:
        overrides[c.path] = c.after
    return changes, overrides


def suggest(model: DecisionModel, request: str, config: dict, qs: QuestionSet | None = None) -> Suggestion:
    """Answer the questions for `request` and apply the confident answers to `config` (the draft's full config,
    as `RunConfig.model_dump(mode="json")`)."""
    qs = qs or load_questions()
    s = answer_groups(model, request, qs)
    s.changes, s.overrides = apply_answers(s.applied, config, qs)
    return s
