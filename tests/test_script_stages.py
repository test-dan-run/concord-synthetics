"""M3 with fake backends: turn plan, stage 2, stage 3 (rules, judges, tie-break deferral), scheduler rounds."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from helpers import fake_run
from synthetics.artifacts.io import read_model
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.models import Script, ScriptValidation
from synthetics.backends.fakes import FakeChatLLM
from synthetics.backends.registry import Backends
from synthetics.config.schema import Role, StageId
from synthetics.orchestrator.scheduler import PhaseScheduler, summarise
from synthetics.orchestrator.status import StatusStore
from synthetics.stages.runner import meeting_id_for
from synthetics.stages.s1_sample import sample_meeting
from synthetics.stages.script_plan import build_plan, order_no_repeats, share_counts
from synthetics.text.pdpa import strip_identifiers

# ---------------------------------------------------------------- plan


def test_share_counts_are_exact_and_cover_everyone():
    counts = share_counts({"S01": 0.5, "S02": 0.3, "S03": 0.2}, 10)
    assert counts == {"S01": 5, "S02": 3, "S03": 2}
    tiny = share_counts({f"S{i:02d}": (0.9 if i == 1 else 0.1 / 19) for i in range(1, 21)}, 30)
    assert sum(tiny.values()) == 30 and min(tiny.values()) >= 1


def test_order_has_no_repeats_when_possible():
    rng = np.random.default_rng(0)
    for _ in range(50):
        counts = {"a": 6, "b": 4, "c": 3, "d": 1}
        order = order_no_repeats(counts, rng)
        assert Counter(order) == Counter(counts)
        assert all(x != y for x, y in zip(order, order[1:], strict=False))


@pytest.fixture(scope="module")
def sample_meetings(cfg, splits):
    return [sample_meeting(cfg, splits, meeting_id_for(i), i, np.random.default_rng(i)) for i in range(40)]


def test_plan_matches_shares_and_rates(cfg, sample_meetings):
    tt = cfg.turn_taking
    for m in sample_meetings:
        plan = build_plan(m, tt, np.random.default_rng(1))
        main = plan.main_turns()
        assert abs(len(main) - m.dynamics.n_turns_target) <= len(m.people)
        counts = Counter(t.speaker_id for t in main)
        assert set(counts) == {p.speaker_id for p in m.people}, "everyone speaks"
        for sid, share in m.dynamics.turn_shares.items():
            assert abs(counts[sid] / len(main) - share) < 0.08, (m.meeting_id, sid)
        assert main[0].interaction == "normal"
        bcs = [t for t in plan.turns if t.interaction == "backchannel"]
        assert abs(len(bcs) - plan.backchannels_per_min * m.dynamics.duration_min) <= 1
        seq = plan.turns
        for i, t in enumerate(seq):
            if t.interaction == "backchannel":
                assert t.speaker_id != seq[i - 1].speaker_id, "backchannels come from listeners"
        if m.dynamics.presentation:
            pres = [t for t in plan.turns if t.interaction == "presentation"]
            assert pres and all(t.speaker_id in m.dynamics.presentation.presenters for t in pres)


def test_plan_is_deterministic(cfg, sample_meetings):
    m = sample_meetings[3]
    a = build_plan(m, cfg.turn_taking, np.random.default_rng(9))
    b = build_plan(m, cfg.turn_taking, np.random.default_rng(9))
    assert a.turns == b.turns


# ---------------------------------------------------------------- stages 2 and 3 through the scheduler


def test_fake_run_writes_scripts_prompts_and_validations(tmp_path):
    rd, cfg = fake_run(tmp_path, n=3)
    PhaseScheduler(rd, cfg).run()
    s = summarise(rd)
    assert s["stages"][3] == {"pass": 3} and s["regeneration_rate"] == 0.0
    for mid in ("m-000001", "m-000002", "m-000003"):
        d = rd / "meetings" / mid
        m = read_model(d / "meeting.json", MeetingSpec)
        script = read_model(d / "script.json", Script)
        assert {t.speaker_id for t in script.turns} == {p.speaker_id for p in m.people}
        for p in m.people:
            prompt = (d / "prompts" / f"{p.speaker_id}.txt").read_text()
            assert prompt.startswith(f"# Persona {p.speaker_id} · ")
            assert p.persona.display_name in prompt and m.dynamics.topic in prompt
            assert p.nsc_speaker.speaker not in prompt, "NSC identities never reach a prompt"
        for t in script.turns:
            assert t.text_norm == t.text_norm.lower()
        v = read_model(d / "script_validation.json", ScriptValidation)
        assert v.rules_passed and v.decision == "pass" and v.primary is not None


def test_second_run_is_all_cache_hits(tmp_path):
    rd, cfg = fake_run(tmp_path, n=2)
    PhaseScheduler(rd, cfg).run()
    report = PhaseScheduler(rd, cfg).run()
    assert {o.state for o in report.outcomes} == {"cached"}


def test_borderline_defers_to_the_tiebreak_phase(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1)
    primary = FakeChatLLM("judge-a", judge_overall=3.1)  # within 3.0 +- 0.5
    tiebreak = FakeChatLLM("judge-b", judge_overall=3.4)
    backends = Backends(cfg.backends, overrides={Role.JUDGE_PRIMARY: primary, Role.JUDGE_TIEBREAK: tiebreak})
    PhaseScheduler(rd, cfg, backends=backends).run()
    v = read_model(rd / "meetings/m-000001/script_validation.json", ScriptValidation)
    assert v.borderline and v.decision == "pass"
    assert v.primary.service == "judge-a" and v.tiebreak.service == "judge-b"
    assert len(primary.calls) == 1, "the primary judgement is reused when the stage resumes in phase B2"
    assert len(tiebreak.calls) == 1


def test_disagreement_fails_and_regenerates_with_feedback(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1)
    primary = FakeChatLLM("judge-a", judge_overall_by_call=[3.2, 4.5])  # borderline pass, then a clear pass
    tiebreak = FakeChatLLM("judge-b", judge_overall=2.6)  # disagrees
    generator = FakeChatLLM("gen")
    backends = Backends(cfg.backends, overrides={Role.JUDGE_PRIMARY: primary, Role.JUDGE_TIEBREAK: tiebreak,
                                                 Role.GENERATOR: generator})
    PhaseScheduler(rd, cfg, backends=backends).run()
    st = StatusStore(rd).get("m-000001")
    assert st.get(StageId.SCRIPT).attempts == 2 and st.get(StageId.SCRIPT_VAL).state == "pass"
    second_round_prompts = [c for c in generator.calls if c["title"] == "script_chunk"][-1]["messages"][-1].content
    assert "Feedback on the previous version" in second_round_prompts
    assert summarise(rd)["regeneration_rate"] == 1.0


def test_unresolved_scripts_are_dropped_after_max_rounds(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1)
    backends = Backends(cfg.backends, overrides={Role.JUDGE_PRIMARY: FakeChatLLM("judge-a", judge_overall=1.5)})
    PhaseScheduler(rd, cfg, backends=backends).run()
    st = StatusStore(rd).get("m-000001")
    assert st.get(StageId.SCRIPT).state == "dropped" and st.get(StageId.SCRIPT).attempts == cfg.retries.script_rounds
    assert summarise(rd)["dropped"] == 1


def test_rule_failure_skips_the_judges(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1)

    class NoCutoffs(FakeChatLLM):
        def chat(self, messages, **kw):
            res = super().chat(messages, **kw)
            if kw.get("json_schema", {}).get("title") == "script_chunk":
                out = json.loads(res.text)
                for t in out["turns"]:
                    t["text"] = "(sneeze) " + t["text"]  # markup outside the convention
                res.text = json.dumps(out)
            return res

    judge = FakeChatLLM("judge-a")
    backends = Backends(cfg.backends, overrides={Role.GENERATOR: NoCutoffs("gen"), Role.JUDGE_PRIMARY: judge})
    PhaseScheduler(rd, cfg, backends=backends).run()
    v = read_model(rd / "meetings/m-000001/script_validation.json", ScriptValidation)
    assert not v.rules_passed and v.primary is None and "markup in convention" in (v.feedback or "")
    assert judge.calls == []


def test_retry_keeps_the_plan_and_changes_only_the_seed(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1)
    primary = FakeChatLLM("judge-a", judge_overall_by_call=[1.5, 4.5])
    gen = FakeChatLLM("gen")
    backends = Backends(cfg.backends, overrides={Role.JUDGE_PRIMARY: primary, Role.GENERATOR: gen})
    PhaseScheduler(rd, cfg, backends=backends).run()
    chunks = [c for c in gen.calls if c["title"] == "script_chunk"]
    half = len(chunks) // 2
    first, second = chunks[:half], chunks[half:]
    assert [c["messages"][-1].content.split("Plan,")[1][:200] for c in first[:1]] == \
           [c["messages"][-1].content.split("Plan,")[1][:200] for c in second[:1]]
    assert first[0]["seed"] != second[0]["seed"]


def test_pdpa_strip():
    text = ("Ask Jun Jie Tan at 9123 4567 or jj@example.com about S1234567D; Mei Ling Lim agreed; "
            "see www.example.com, Singapore 123456")
    out = strip_identifiers(text, roster_names=["Mei Ling Lim"], known_names=["Jun Jie", "Tan", "Mei Ling", "Lim"])
    assert "9123" not in out and "@" not in out and "S1234567D" not in out and "123456" not in out
    assert "Jun Jie" not in out and "Tan" not in out
    assert "Mei Ling Lim" in out


def test_a_new_stage_version_reruns_passed_meetings(tmp_path, monkeypatch):
    from synthetics.stages.s2_script import ScriptStage

    rd, cfg = fake_run(tmp_path, n=1)
    PhaseScheduler(rd, cfg).run()
    monkeypatch.setattr(ScriptStage, "version", "test-bump")
    report = PhaseScheduler(rd, cfg).run()
    states = {(int(o.stage), o.state) for o in report.outcomes}
    assert (2, "pass") in states and (3, "pass") in states, states
