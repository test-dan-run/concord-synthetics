"""Decision questions to builder settings: the question file against the config, the engine's policy, the effects,
the word-matching fake, the logprob reader and the suggest routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import V1_POOL
from synthetics.config.loader import load_run_config
from synthetics.suggest.effects import apply_option, expand, get_path
from synthetics.suggest.engine import YES_NO, Answer, apply_answers, suggest
from synthetics.suggest.models import FakeDecisionModel, letter_probs
from synthetics.suggest.questions import load_questions

QS = load_questions()


def defaults() -> dict:
    return load_run_config(overrides={"run_id": "t", "seed": 1, "n_meetings": 1}).model_dump(mode="json")


@pytest.mark.parametrize("group", [g.id for g in QS.groups])
def test_every_option_applies_to_the_default_config_and_stays_valid(group):
    g = QS.group(group)
    for opt in g.options:
        for scope in ("both", "small", "large") if g.scope else ("both",):
            for e in opt.effects + opt.overlay:
                for path in expand(e.path, scope):
                    get_path(defaults(), path)  # the path exists
            changes, overrides = apply_answers([Answer(g.id, opt.id, opt.label, 1.0, scope)], defaults())
            load_run_config(overrides={"run_id": "t", "seed": 1, "n_meetings": 1, **overrides})


def test_bundles_name_real_groups_and_options():
    for gq in QS.global_:
        for opt in gq.options or []:
            for gid, oid in opt.bundle.items():
                QS.group(gid).option(oid)


def test_effects_clamp_to_bounds_and_round_integer_ranges():
    cfg = defaults()
    small = QS.group("meeting_size").option("larger").effects
    apply_option(cfg, small, "both", QS, overlay=False)
    assert get_path(cfg, "room_classes.classes.large.speakers") == {"dist": "int_uniform", "lo": 12, "hi": 20}
    noisy = QS.group("noise").option("much_noisier").effects
    for _ in range(4):
        apply_option(cfg, noisy, "small", QS, overlay=False)
    assert get_path(cfg, "acoustics.snr_db.small")["lo"] == -5  # clamped at the bound
    assert get_path(cfg, "acoustics.snr_db.large")["lo"] == 5.0  # scope: small only


class Scripted:
    """A model that answers yes to the listed gates and picks the listed option for those groups."""

    def __init__(self, picks: dict[str, tuple[str, float]], gates: set[str]):
        self.picks, self.gates = picks, gates

    def choose(self, request, question, options):
        if options == YES_NO:
            g = next((g for g in QS.groups if g.gate == question), None)
            return [0.95, 0.05] if g is not None and g.id in self.gates else [0.05, 0.95]
        g = next((g for g in QS.groups if g.question == question), None)
        if g is not None and g.id in self.picks:
            oid, p = self.picks[g.id]
            i = [o.id for o in g.options].index(oid)
            rest = (1 - p) / (len(options) - 1)
            return [p if k == i else rest for k in range(len(options))]
        unchanged = next((k for k, o in enumerate(options) if o in ("Unchanged", "Not said", "Not asked")), 0)
        return [0.9 if k == unchanged else 0.1 / (len(options) - 1) for k in range(len(options))]


def test_policy_applies_offers_and_asks():
    m = Scripted({"hybrid": ("most", 0.9), "noise": ("a_bit_noisier", 0.5), "backends": ("fake", 0.99),
                  "interruptions": ("unchanged", 0.9)}, gates={"hybrid", "noise", "backends", "interruptions"})
    s = suggest(m, "lots of people dialling in, maybe a bit noisy, just a dry run", defaults())
    assert [(a.group, a.option) for a in s.applied] == [("hybrid", "most")]
    assert s.overrides == {"acoustics.hybrid.small.p": 0.85, "acoustics.hybrid.large.p": 0.85}
    assert [o.group for o in s.offers] == ["noise"] and len(s.offers[0].options) == 2
    assert [o.group for o in s.confirm] == ["backends"]  # confirm_always: never applied


def test_bundles_apply_unless_the_group_is_named_directly():
    class Harder(Scripted):
        def choose(self, request, question, options):
            if question.startswith("Does the request ask for harder"):
                return [0.9, 0.05, 0.05]
            return super().choose(request, question, options)

    s = suggest(Harder({"noise": ("much_cleaner", 0.9)}, gates={"noise"}), "harder data but clean audio", defaults())
    by = {a.group: a for a in s.applied}
    assert by["noise"].option == "much_cleaner" and by["noise"].source == "direct"
    assert by["overlap"].option == "more" and by["overlap"].source == "overall_difficulty"


@pytest.mark.parametrize(("request_text", "expected"), [
    ("lots of people dialling in on Zoom", {"hybrid"}),
    ("noisy rooms with aircon noise", {"noise"}),
    ("people talking over each other and interrupting a lot", {"overlap", "interruptions"}),
])
def test_the_fake_finds_the_obvious_groups(request_text, expected):
    s = suggest(FakeDecisionModel(), request_text, defaults())
    touched = {a.group for a in s.applied} | {o.group for o in s.offers}
    assert expected <= touched


def test_letter_probs_reads_top_logprobs_and_falls_back():
    raw = {"choices": [{"message": {"content": "B"}, "logprobs": {"content": [{"token": "B", "logprob": -0.1,
           "top_logprobs": [{"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.4},
                            {"token": "Z", "logprob": -3.0}]}]}}]}
    p = letter_probs(raw, 3)
    assert p[1] > 0.85 and p[2] == 0 and abs(sum(p) - 1) < 1e-9
    p = letter_probs({"choices": [{"message": {"content": "A"}}]}, 2)
    assert p == [0.8, pytest.approx(0.2)]


def test_suggest_routes(tmp_path):
    from synthetics.viewer.api import create_app
    from synthetics.viewer.suggest_routes import add_routes

    app = create_app(tmp_path, tmp_path / "no-dist")
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") not in ("/api/suggest",
                                                                                         "/api/suggest/apply")]
    add_routes(app, model_factory=lambda: (FakeDecisionModel(), "fake"))
    c = TestClient(app)
    draft = {"n_meetings": 4, "seed": 3, "overrides": dict(V1_POOL), "yaml": ""}
    r = c.post("/api/suggest", json={"request": "lots of people dialling in on Zoom", "draft": draft})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "fake"
    assert {a["group"] for a in body["applied"]} | {o["group"] for o in body["offers"]} >= {"hybrid"}
    assert body["draft"]["overrides"] == {**V1_POOL, **body["overrides"]}
    r = c.post("/api/suggest/apply", json={"draft": draft, "choices": [{"group": "noise", "option": "much_noisier",
                                                                        "scope": "large"}]})
    assert r.status_code == 200 and r.json()["overrides"] == {"acoustics.snr_db.large.lo": -5.0,
                                                               "acoustics.snr_db.large.hi": 15.0}
    assert c.post("/api/suggest/apply", json={"choices": [{"group": "noise", "option": "nope"}]}).status_code == 422
