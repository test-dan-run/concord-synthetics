"""Stages 2 and 3 with fake backends: turn plan, stage 2, stage 3 (rules, judges, tie-break deferral), scheduler rounds."""

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


def test_two_person_meeting_plan_passes_the_rate_rules(cfg, splits):
    """A 2-person meeting at a 2:1 share is mostly continuations; rates count turn changes only."""
    from synthetics.artifacts.models import Script, Turn
    from synthetics.text import Lexicon
    from synthetics.validators.script_rules import check_script

    lex = Lexicon.load(cfg.lexicon.path)
    found = 0
    for i in range(300):
        m = sample_meeting(cfg, splits, meeting_id_for(i), i, np.random.default_rng(i))
        if m.n_speakers > 3:
            continue
        found += 1
        plan = build_plan(m, cfg.turn_taking, np.random.default_rng(i))
        turns = [Turn(turn_id=f"t{k:04d}", speaker_id=t.speaker_id, interaction=t.interaction,
                      text="ok (mm)" if t.interaction == "backchannel" else "we go first then see how-",
                      text_norm="ok mm" if t.interaction == "backchannel" else "we go first then see how")
                 for k, t in enumerate(plan.turns)]
        script = Script(meeting_id=m.meeting_id, brief="", turns=turns, generator={
            "service": "x", "template_version": "x", "thinking": False, "seed": 0, "temperature": 0.0,
            "lexicon_version": "x", "round": 1, "chunks": 1})
        rules = {r.name: r for r in check_script(script, m, cfg, lex)}
        for name in ("overlap-start rate", "interruption rate", "turn shares"):
            assert rules[name].passed, (m.meeting_id, m.n_speakers, name, rules[name])
    assert found >= 10


# ---------------------------------------------------------------- rubric_v3, the criterion floor and the code checks

V3 = {"judge.rubric_version": "rubric_v3"}
V3_CRITERIA = ("responsiveness", "consistency", "progression", "language_fit", "realism", "social_dynamics")


def test_rubric_v3_asks_for_its_criteria_and_records_only_those(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1, **V3)
    judge = FakeChatLLM("judge-a", judge_overall=4.0)
    PhaseScheduler(rd, cfg, backends=Backends(cfg.backends, overrides={Role.JUDGE_PRIMARY: judge})).run()
    system = judge.calls[0]["messages"][0].content
    assert all(f"- {c}:" in system for c in V3_CRITERIA) and "singlish:" not in system
    raw = json.loads((rd / "meetings/m-000001/script_validation.json").read_text())
    scores = raw["primary"]["scores"]
    assert set(scores) == {*V3_CRITERIA, "persona_consistency", "overall"}
    assert raw["primary"]["rubric_version"] == "rubric_v3" and raw["decision"] == "pass"


def test_rubric_v2_results_keep_their_fields():
    from synthetics.artifacts.models import JudgeScores

    s = JudgeScores(coherence=4, singlish=3, realism=3, persona_consistency={"S01": 4}, overall=3)
    assert list(json.loads(s.model_dump_json())) == ["coherence", "singlish", "realism", "persona_consistency",
                                                     "overall"]
    assert s.criteria() == {"coherence": 4, "singlish": 3, "realism": 3}


def test_judge_schema_follows_the_rubric():
    from synthetics.stages.prompts import judge_schema

    v2, v3 = judge_schema("rubric_v2"), judge_schema("rubric_v3")
    assert v2 == judge_schema() and "coherence" in v2["required"] and "language_fit" not in v2["properties"]
    assert all(c in v3["required"] for c in V3_CRITERIA) and "singlish" not in v3["properties"]


def test_criterion_floor_fails_a_low_criterion_or_speaker():
    from synthetics.artifacts.models import JudgeScores
    from synthetics.validators.judges import low_criteria, passes

    good = {c: 4.0 for c in V3_CRITERIA}
    s = JudgeScores(**good, persona_consistency={"S01": 4.0, "S02": 2.0}, overall=4.0)
    assert passes(s, 3.0) and not passes(s, 3.0, 2.0)
    assert low_criteria(s, 2.0) == ["persona S02 2.0"]
    low = JudgeScores(**{**good, "realism": 2.0}, persona_consistency={"S01": 4.0}, overall=4.0)
    assert low_criteria(low, 2.0) == ["realism 2.0"] and passes(low, 3.0, 1.5)


class LowRealismOnce(FakeChatLLM):
    """A judge that gives overall 4 but realism 2 on its first judgement."""

    def chat(self, messages, **kw):
        res = super().chat(messages, **kw)
        if kw.get("json_schema", {}).get("title") == "script_judgement" and len(self.calls) == 1:
            out = json.loads(res.text)
            out["realism"] = 2.0
            res.text = json.dumps(out)
        return res


def test_a_low_criterion_sends_the_script_back_with_the_scores_named(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1, **V3, **{"thresholds.judge_criterion_floor": 2.0})
    gen = FakeChatLLM("gen")
    judge = LowRealismOnce("judge-a", judge_overall=4.0)
    PhaseScheduler(rd, cfg, backends=Backends(cfg.backends, overrides={Role.JUDGE_PRIMARY: judge,
                                                                         Role.GENERATOR: gen})).run()
    st = StatusStore(rd).get("m-000001")
    assert st.get(StageId.SCRIPT).attempts == 2 and st.get(StageId.SCRIPT_VAL).state == "pass"
    second = [c for c in gen.calls if c["title"] == "script_chunk"][-1]["messages"][-1].content
    assert "These scores were too low: realism 2.0." in second


def test_new_stage3_settings_stay_out_of_the_cache_key_while_off(tmp_path):
    from synthetics.config.loader import canonical_json
    from synthetics.stages.s3_script_val import ScriptValidationStage

    _, off = fake_run(tmp_path / "off", n=1)
    on = off.model_copy(update={"thresholds": off.thresholds.model_copy(update={
        "judge_criterion_floor": 2.0,
        "script_rules": off.thresholds.script_rules.model_copy(update={"max_self_introductions": 0})})})
    stage = ScriptValidationStage()
    key_off = canonical_json(stage.config_slice(off))
    assert "judge_criterion_floor" not in key_off and "max_self_introductions" not in key_off
    assert "words_per_planned_min" not in key_off and "repeated_phrase" not in key_off
    key_on = json.loads(canonical_json(stage.config_slice(on)))
    assert key_on["judge_criterion_floor"] == 2.0 and key_on["rules"]["max_self_introductions"] == 0


def test_code_checks_run_only_when_set(tmp_path):
    from synthetics.text.singlish import run_lexicon
    from synthetics.validators.script_rules import check_script, repeated_phrase_share, self_introductions

    rd, cfg = fake_run(tmp_path, n=1)
    PhaseScheduler(rd, cfg).run(until=StageId.SCRIPT)
    d = rd / "meetings/m-000001"
    m, script = read_model(d / "meeting.json", MeetingSpec), read_model(d / "script.json", Script)
    names = {c.name for c in check_script(script, m, cfg, run_lexicon(cfg))}
    assert not names & {"words per planned minute", "repeated phrasing", "self-introductions"}

    # a speaker who names themself, and a stock line said twice
    first = m.people[0]
    turns = list(script.turns)
    i = next(k for k, t in enumerate(turns) if t.speaker_id == first.speaker_id and t.interaction == "normal")
    own = first.persona.display_name.split()[0]
    turns[i] = turns[i].model_copy(update={"text": f"{own} here, let me start."})
    script = script.model_copy(update={"turns": turns})
    assert self_introductions(script, m) == [turns[i].turn_id]
    share, _ = repeated_phrase_share(script)
    assert 0.0 <= share <= 1.0

    on = cfg.model_copy(update={"thresholds": cfg.thresholds.model_copy(update={
        "script_rules": cfg.thresholds.script_rules.model_copy(update={
            "min_words_per_planned_min": 1.0, "max_repeated_phrase_share": 1.0, "max_self_introductions": 0})})})
    checks = {c.name: c for c in check_script(script, m, on, run_lexicon(on))}
    assert checks["words per planned minute"].passed and checks["repeated phrasing"].passed
    assert not checks["self-introductions"].passed and checks["self-introductions"].value == 1


def test_the_rating_sheet_follows_the_runs_rubric(tmp_path):
    import csv

    from synthetics.validators.calibration import export_judge_set

    rd, cfg = fake_run(tmp_path, n=1, **V3)
    PhaseScheduler(rd, cfg).run(until=StageId.SCRIPT_VAL)
    sheet = export_judge_set(rd)
    with open(sheet, newline="") as fh:
        header = next(csv.reader(fh))
    assert all(c in header for c in V3_CRITERIA) and "singlish" not in header
    assert "language_fit" in (sheet.parent / "README.txt").read_text()


def test_qa_interruptions_follow_their_setting_and_old_keys_stay(cfg, sample_meetings):
    from synthetics.config.loader import canonical_json
    from synthetics.config.schema import QA_INTERRUPTION_P_V1, TurnTakingConfig

    tt = cfg.turn_taking
    old = tt.model_copy(update={"qa_interruption_p": QA_INTERRUPTION_P_V1})
    assert "qa_interruption_p" not in canonical_json(old), "frozen runs keep their cache keys"
    assert json.loads(canonical_json(tt))["qa_interruption_p"] == tt.qa_interruption_p == 0.1
    assert TurnTakingConfig.model_validate(json.loads(canonical_json(old))).qa_interruption_p == QA_INTERRUPTION_P_V1

    with_pres = [m for m in sample_meetings if m.dynamics.presentation][:5]
    assert with_pres

    def interruptions(p):
        t = tt.model_copy(update={"qa_interruption_p": p})
        return sum(sum(x.interaction == "interruption" for x in build_plan(m, t, np.random.default_rng(1)).turns)
                   for m in with_pres)

    assert interruptions(0.0) < interruptions(1.0)
