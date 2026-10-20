"""GPU lease (stack.py calls), status store, preflight, judge calibration files."""

from __future__ import annotations

import csv
import subprocess

import pytest

from helpers import fake_run
from synthetics.config.schema import Phase, StageId
from synthetics.orchestrator.lease import GpuLease, LeaseError
from synthetics.orchestrator.scheduler import PhaseScheduler
from synthetics.orchestrator.status import StatusStore
from synthetics.validators.calibration import export_judge_set, fit_threshold


class Recorder:
    def __init__(self, codes=None):
        self.calls, self.codes = [], list(codes or [])

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[cmd.index("tools/stack.py") + 1:] if "tools/stack.py" in cmd else cmd)
        code = self.codes.pop(0) if self.codes else 0
        return subprocess.CompletedProcess(cmd, code, stdout="", stderr="error: over budget\n" if code else "")


def test_lease_brings_services_up_and_down(tmp_path):
    rec = Recorder()
    with GpuLease(["b", "a"], platform_dir=tmp_path, runner=rec):
        pass
    assert rec.calls == [["up", "a", "b"], ["down", "a", "b"]]
    rec = Recorder()
    with GpuLease(["a"], platform_dir=tmp_path, no_evict=True, keep=True, runner=rec):
        pass
    assert rec.calls == [["up", "a", "--no-evict"]]


def test_lease_refusal_raises(tmp_path):
    with pytest.raises(LeaseError, match="over budget"), GpuLease(["a"], platform_dir=tmp_path, runner=Recorder([1])):
        pass


def test_lease_is_a_no_op_when_disabled(tmp_path):
    rec = Recorder()
    with GpuLease(["a"], platform_dir=tmp_path, enabled=False, runner=rec):
        pass
    assert rec.calls == []


def test_scheduler_leases_each_phase_in_order(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1)
    leases = []

    def factory(services, **kw):
        leases.append(tuple(services))
        return GpuLease(services, **{**kw, "enabled": False})

    PhaseScheduler(rd, cfg, lease_factory=factory).run()
    assert leases == [("qwen-sea-lion-v4-5-27b-it",), ("gemma-sea-lion-v4-27b-it-nvfp4",)]
    leases.clear()
    PhaseScheduler(rd, cfg, lease_factory=factory).run()
    assert leases == [], "nothing to do: no lease"


def test_only_phase_cpu_runs_stage_one(tmp_path):
    rd, cfg = fake_run(tmp_path, n=2)
    PhaseScheduler(rd, cfg).run(only=Phase.CPU)
    store = StatusStore(rd)
    assert [store.get(m).get(1).state for m in ("m-000001", "m-000002")] == ["pass", "pass"]
    assert store.get("m-000001").get(2).state == "pending"


def test_status_store_counts_attempts(tmp_path):
    store = StatusStore(tmp_path)
    store.start("m-000001", 2)
    store.mark("m-000001", 2, "retry", detail="x")
    store.start("m-000001", 2)
    st = store.get("m-000001").get(2)
    assert st.attempts == 2 and st.state == "running"
    assert store.pending(2, ["m-000001"]) == ["m-000001"]


def test_judge_set_export_and_threshold_fit(tmp_path):
    rd, cfg = fake_run(tmp_path, n=12)
    PhaseScheduler(rd, cfg).run()
    sheet = export_judge_set(rd, n=12, name="js")
    with open(sheet, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12 and "primary_overall" not in rows[0], "raters never see the judge's scores"
    assert all((sheet.parent / r["script_file"]).is_file() for r in rows)
    # raters: the first half pass, the second half fail; judge scores rewritten to separate them around 3.8
    js = sheet.parent / "judge_scores.csv"
    with open(js, newline="") as fh:
        judge = list(csv.DictReader(fh))
    for i, r in enumerate(judge):
        r["primary_overall"] = 4.4 - 0.1 * i
    with open(js, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(judge[0]))
        w.writeheader()
        w.writerows(judge)
    for i, r in enumerate(rows):
        r.update({"overall": 4 if i < 6 else 2, "rater": "a"})
    with open(sheet, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "thresholds.yaml"
    res = fit_threshold(sheet, out=out)
    assert res["balanced_accuracy"] == 1.0 and 3.8 < res["judge_pass"] < 3.9
    assert res["n_scripts"] == 12 and "judge_pass" in out.read_text()


def test_a_killed_run_keeps_the_phase_time_it_spent(tmp_path, monkeypatch):
    import json

    from synthetics.orchestrator import phase_c
    from synthetics.orchestrator.run_report import LEDGER, write_run_report
    from voice_helpers import fake_refs

    monkeypatch.setenv("NSC_REFS_DIR", str(fake_refs(tmp_path / "refs")))
    rd, cfg = fake_run(tmp_path / "run", n=2, seed=11)
    # a report from before the ledger existed: its totals carry over
    (rd / "run_report.json").write_text(json.dumps({"phase_seconds": {"A": 100.0}}))
    real = phase_c.voice_meeting
    calls = []

    def dies_on_the_second(*a, **kw):
        calls.append(a[1])
        if len(calls) == 2:
            raise KeyboardInterrupt  # killed in the middle of phase C
        return real(*a, **kw)

    monkeypatch.setattr(phase_c, "voice_meeting", dies_on_the_second)
    sched = PhaseScheduler(rd, cfg, lease_factory=lambda s, **kw: GpuLease(s, **{**kw, "enabled": False}))
    with pytest.raises(KeyboardInterrupt):
        sched.run(until=StageId.LABELS)
    lines = [json.loads(x) for x in (rd / LEDGER).read_text().splitlines()]
    assert lines[0] == {"phase": "A", "seconds": 100.0, "from": "run_report.json"}
    recorded = {x["phase"] for x in lines[1:]}
    assert recorded >= {"CPU", "A", "B", "C"}, recorded  # phase C's first meeting was recorded before the kill
    rep = write_run_report(rd)
    assert rep["phase_seconds"]["A"] >= 100.0 and rep["phase_seconds"]["C"] > 0
