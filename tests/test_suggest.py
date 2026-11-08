"""Decision questions to builder settings: the question file against the config, the engine's policy, the effects,
the word-matching fake, the System One client and the suggest routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import V1_POOL
from synthetics.config.loader import load_run_config
from synthetics.suggest.effects import apply_option, expand, get_path
from synthetics.suggest.engine import YES_NO, Answer, apply_answers, suggest
from synthetics.suggest.models import FakeDecisionModel, SystemOneDecisionModel, answer_probs
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


class FakeSystemOne:
    """Stands in for aicore's decision.systemone client: records each call, answers yes and the second option."""

    def __init__(self):
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        return {k: {"type": "noul", "noul": 0.8} if q["type"] == "noul" else
                {"type": "choice", "choice": "o1", "probabilities": {o: (0.7 if o == "o1" else 0.3 / (len(q["criteria"]) - 1))
                                                                     for o in q["criteria"]}}
                for k, q in questions.items()}


def test_systemone_questions_and_answers():
    from synthetics.backends.transport import Transport

    client = FakeSystemOne()
    m = SystemOneDecisionModel("julia-1", Transport("aicore"), client=client)
    got = m.choose_many(" noisy rooms ", [("g", "Does it mention noise?", YES_NO), ("c", "How noisy?", ["A", "B", "C"])])
    assert got["g"] == pytest.approx([0.8, 0.2]) and got["c"] == pytest.approx([0.15, 0.7, 0.15])
    state, questions = client.calls[0]
    assert state == "noisy rooms" and questions["g"]["type"] == "noul" and "criteria" in questions["g"]
    assert questions["c"] == {"type": "choice", "instructions": "How noisy?", "criteria": {"o0": "A", "o1": "B", "o2": "C"}}
    assert answer_probs({"probabilities": {}}, ["A", "B"]) == [0.5, 0.5]


def test_the_engine_asks_in_two_batches():
    client = FakeSystemOne()
    from synthetics.backends.transport import Transport

    suggest(SystemOneDecisionModel("julia-1", Transport("aicore"), client=client), "anything", defaults())
    assert len(client.calls) == 2
    first, second = (set(q) for _, q in client.calls)
    assert {f"gate:{g.id}" for g in QS.groups} <= first and {f"global:{g.id}" for g in QS.global_} <= first
    assert {f"choice:{g.id}" for g in QS.groups} | {"scope"} == second  # every gate said yes


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


def test_topic_suggestions_come_from_the_bank_and_skip_the_users_list(tmp_path):
    from synthetics.viewer.api import create_app
    from synthetics.viewer.suggest_routes import topic_bank

    c = TestClient(create_app(tmp_path, tmp_path / "no-dist"))
    bank = {t for ts in topic_bank().values() for t in ts}
    first = c.get("/api/topics", params={"n": 4}).json()["topics"]
    assert len(first) == 4 and {t["topic"] for t in first} <= bank and len({t["theme"] for t in first}) == 4
    mine = first[0]["topic"]
    again = c.get("/api/topics", params={"n": 20, "exclude": [mine]}).json()["topics"]
    assert mine not in {t["topic"] for t in again}


def test_viewer_uses_word_matching_unless_the_service_is_switched_on(monkeypatch):
    from synthetics.backends import registry
    from synthetics.viewer import suggest_routes

    monkeypatch.setattr(registry, "check_endpoints", lambda cfg, roles: [])
    monkeypatch.delenv(suggest_routes.USE_SERVICE_ENV, raising=False)
    assert suggest_routes.decision_model()[1] == "fake"
    monkeypatch.setenv(suggest_routes.USE_SERVICE_ENV, "1")
    assert suggest_routes.decision_model()[1] == "laya"
    assert suggest_routes.decision_model(use_service=False)[1] == "fake"
