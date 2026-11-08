"""Labelled decision requests: seeded labels from questions.yaml, the writer's brief, resuming, and the rows
handed to Laya's fine-tuning."""

from __future__ import annotations

import json

import numpy as np

from synthetics.backends.base import ChatResult
from synthetics.suggest.datagen import Target, brief, expected, generate, laya_rows, sample_target
from synthetics.suggest.engine import YES_NO
from synthetics.suggest.models import as_question
from synthetics.suggest.questions import load_questions

QS = load_questions()


class Writer:
    """Stands in for the generator: returns the brief's first item line as the request."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, *, json_schema, temperature, seed, max_tokens, thinking):
        self.calls += 1
        assert json_schema["title"] == "decision_request" and thinking is False
        return ChatResult(text=json.dumps({"request": f"  please\n {seed % 97}  "}))


def test_targets_are_seeded_and_valid():
    targets = [sample_target(np.random.default_rng(np.random.SeedSequence([1, i])), QS) for i in range(300)]
    again = sample_target(np.random.default_rng(np.random.SeedSequence([1, 7])), QS)
    assert again == targets[7]
    for t in targets:
        for gid, oid in t.groups.items():
            assert oid != "unchanged" and oid in [o.id for o in QS.group(gid).options]
        assert t.scope == "both" or any(QS.group(g).scope for g in t.groups)
    sizes = {len(t.groups) for t in targets}
    assert {0, 1, 2, 3} <= sizes and max(sizes) <= 4
    assert any(t.out_of_scope for t in targets) and any(t.purpose != "unchanged" for t in targets)


def test_expected_adds_bundles_but_direct_answers_win():
    t = Target(groups={"noise": "much_cleaner"}, overall_difficulty="harder")
    want = expected(t, QS)
    assert want["noise"] == "much_cleaner" and want["overlap"] == "more" and want["reverb"] == "more_echo"
    assert expected(Target(groups={}, purpose="trial"), QS) == {"length": "much_shorter", "backends": "fake"}


def test_brief_names_every_item():
    t = Target(groups={"hybrid": "most", "noise": "a_bit_noisier"}, scope="small", exact_numbers=True,
               out_of_scope="ask for some specific words or product names to come up", style="chat_edit",
               singlish=True, persona="an HR manager")
    text = brief(t, QS)
    assert "In most meetings" in text and "A bit noisier" in text and "only the small meetings" in text
    assert "9 of 10 meetings" in text and "product names" in text and "Singlish" in text and "follow-up" in text
    avoid = text.split("Say nothing at all about:")[1]
    assert "people talking over each other" in avoid and "how many people attend" in avoid
    assert "background noise" not in avoid and "remote" not in avoid  # the items asked for are never off limits
    assert "Nothing about the meetings at all" in brief(Target(groups={}), QS)
    assert "not a request to change" in brief(Target(groups={"noise": "much_noisier"}, style="terse"), QS)
    strong = brief(Target(groups={"noise": "much_noisier"}, style="statement"), QS)
    assert "a strong change" in strong and "as facts" in strong
    assert "a mild change" in brief(Target(groups={"noise": "a_bit_noisier"}), QS)
    assert "change:" not in brief(Target(groups={"race": "more_malay"}), QS)  # not a scale
    numbered = brief(Target(groups={"meeting_size": "largest"}, exact_numbers=True), QS)
    assert "20 people" in numbered and "Give one exact number" not in numbered


def test_generate_resumes_and_keeps_rows_stable():
    w = Writer()
    first = list(generate(w, 12, 5, workers=4, log=lambda _: None))
    assert [r.id for r in first] == [f"r5-{i:05d}" for i in range(12)] and w.calls == 12
    assert first[0].request == first[0].request.strip() and "\n" not in first[0].request
    rest = list(generate(Writer(), 15, 5, done=[r.id for r in first], workers=4, log=lambda _: None))
    assert [r.id for r in rest] == [f"r5-{i:05d}" for i in range(12, 15)]
    assert list(generate(Writer(), 12, 5, workers=1, log=lambda _: None))[3].expect == first[3].expect


def test_laya_rows_ask_what_the_engine_asks():
    row = {"id": "r1-00003", "request": "more people dialling in, and noisy small rooms",
           "expect": {"hybrid": "more", "noise": "a_bit_noisier"},
           "target": {"groups": {"hybrid": "more", "noise": "a_bit_noisier"}, "scope": "small",
                      "exact_numbers": False, "out_of_scope": None, "overall_difficulty": "unchanged",
                      "purpose": "unchanged"}}
    out = laya_rows(row, QS)
    q, want = out["questions"], out["expected"]
    assert out["state"] == row["request"] and set(q) == set(want)
    assert q["gate:noise"] == as_question(QS.group("noise").gate, YES_NO)
    assert want["gate:noise"] is True and want["gate:pace"] is False
    assert sum(want[f"gate:{g.id}"] for g in QS.groups) == 2
    assert q["choice:hybrid"]["criteria"][want["choice:hybrid"]] == "More often"
    extra = [k for k in q if k.startswith("choice:") and k.removeprefix("choice:") not in row["expect"]]
    assert len(extra) == 3 and all(q[k]["criteria"][want[k]] == "Unchanged" for k in extra)
    assert q["scope"]["criteria"][want["scope"]].startswith("Small rooms only")
    assert want["global:purpose"] == "o3" and want["global:out_of_scope"] is False
    assert laya_rows(row, QS) == out  # the extra choices are seeded by the row
