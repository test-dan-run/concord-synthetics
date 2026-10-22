"""M14: the viewer's run builder (preview, create, start, stop) and the background run process."""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from conftest import ROOT
from synthetics.artifacts.io import read_model, write_model
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.models import RunProcess
from synthetics.artifacts.paths import RunPaths
from synthetics.orchestrator import background
from synthetics.viewer.api import create_app

POOL = {"speaker_pool.path": str(ROOT / "data/nsc_part3_reference_speakers.csv")}
FAKE = {**POOL, "backends.mode": "fake"}


class Launcher:
    """Records starts instead of launching a run."""

    def __init__(self):
        self.calls = []

    def __call__(self, run_dir, *, live):
        self.calls.append((run_dir.name, live))
        rec = RunProcess(pid=os.getpid(), state="finished", live=live, started_at="2026-10-21T00:00:00+00:00")
        write_model(RunPaths(run_dir).process, rec)
        return rec


@pytest.fixture
def launcher():
    return Launcher()


@pytest.fixture
def client(tmp_path, launcher):
    return TestClient(create_app(tmp_path, frontend_dist=tmp_path / "no-dist", start=launcher))


def test_defaults_are_the_merged_default_yamls(client):
    body = client.get("/api/defaults").json()
    cfg = body["config"]
    assert cfg["revision"] == 2 and cfg["room_classes"]["ratio"]["weights"] == {"small": 0.3, "large": 0.7}
    assert "run_id" not in cfg and cfg["simulation"]["ism_max_order"] == 3 and cfg["stages"]["3"]["enabled"]
    assert body["preview_max"] == 500
    assert client.get("/api/viewer").json()["read_only"] is False


def test_preview_samples_the_meetings_the_run_will_get(client, tmp_path):
    draft = {"run_id": "p", "n_meetings": 4, "seed": 9,
             "overrides": {**FAKE, "acoustics.hybrid.large.p": 1.0, "stages.3.enabled": False},
             "yaml": "turn_taking:\n  zipf: {dist: zipf, exponent: 1.4}\n"}
    pv = client.post("/api/preview", json=draft).json()
    assert pv["n_sampled"] == 4 and len(pv["meetings"]) == 4 and not pv["failures"]
    assert pv["live"] is False and pv["gpu_services"] == {} and pv["estimate_hours"] is None
    assert pv["stages_enabled"]["3"] is False and pv["checks"]
    assert all(m["hybrid"] for m in pv["meetings"] if m["room_class"] == "large")

    assert client.post("/api/runs", json=draft).status_code == 201
    cfg = yaml.safe_load((tmp_path / "p" / "config.yaml").read_text())
    assert cfg["turn_taking"]["zipf"]["exponent"] == 1.4 and cfg["stages"]["3"]["enabled"] is False
    assert cfg["revision"] == 2 and cfg["seed"] == 9 and cfg["n_meetings"] == 4

    from synthetics.config.schema import StageId
    from synthetics.orchestrator.run import load_run
    from synthetics.orchestrator.scheduler import PhaseScheduler

    rd = tmp_path / "p"
    run_cfg, _ = load_run(rd)
    PhaseScheduler(rd, run_cfg).run(until=StageId.SAMPLE)
    for row in pv["meetings"]:
        m = read_model(RunPaths(rd).meeting(row["meeting_id"]).meeting, MeetingSpec)
        assert (m.room_class, m.n_speakers, m.split) == (row["room_class"], row["n_speakers"], row["split"])
        assert m.acoustics.rt60 == pytest.approx(row["rt60"]) and m.dynamics.duration_min == row["duration_min"]


def test_preview_of_a_real_run_lists_its_gpu_services(client):
    pv = client.post("/api/preview", json={"n_meetings": 2, "overrides": POOL}).json()
    assert pv["live"] is True and pv["estimate_hours"] > 0
    assert pv["gpu_services"]["D"] == ["qwen3-forced-aligner"] and set(pv["gpu_services"]) == {"A", "B", "B2", "C", "D"}


def test_preview_reports_every_invalid_field(client):
    r = client.post("/api/preview", json={"overrides": {"turn_taking.gap.median": -1}, "yaml": "nonsense: 1"})
    assert r.status_code == 422
    locs = {e["loc"] for e in r.json()["detail"]["errors"]}
    assert locs == {"turn_taking.gap.median", "nonsense"}
    r = client.post("/api/preview", json={"yaml": "seed: 4"})
    assert r.status_code == 422 and r.json()["detail"]["errors"][0]["loc"] == "seed"
    r = client.post("/api/preview", json={"yaml": "a: [1"})
    assert r.status_code == 422 and r.json()["detail"]["errors"][0]["loc"] == "yaml"
    r = client.post("/api/preview", json={"yaml": "- 1\n- 2"})
    assert r.status_code == 422


def test_create_refuses_bad_and_existing_ids(client):
    assert client.post("/api/runs", json={"run_id": "../x", "overrides": FAKE}).status_code == 422
    assert client.post("/api/runs", json={"run_id": "", "overrides": FAKE}).status_code == 422
    assert client.post("/api/runs", json={"run_id": "a", "n_meetings": 1, "overrides": FAKE}).status_code == 201
    r = client.post("/api/runs", json={"run_id": "a", "n_meetings": 1, "overrides": FAKE})
    assert r.status_code == 409 and r.json()["detail"]["errors"][0]["loc"] == "run_id"
    assert client.post("/api/runs", json={"run_id": "b", "n_meetings": 0, "overrides": FAKE}).status_code == 422


def test_start_needs_confirmation_for_real_backends(client, launcher):
    client.post("/api/runs", json={"run_id": "real", "n_meetings": 1, "overrides": POOL})
    client.post("/api/runs", json={"run_id": "fake", "n_meetings": 1, "overrides": FAKE})
    r = client.post("/api/runs/real/start")
    assert r.status_code == 409 and r.json()["detail"]["errors"][0]["loc"] == "confirm" and not launcher.calls
    assert client.post("/api/runs/real/start", json={"confirm": True}).status_code == 200
    assert client.post("/api/runs/fake/start").status_code == 200
    assert launcher.calls == [("real", True), ("fake", False)]
    info = client.get("/api/runs/real/process").json()
    assert info["process"]["state"] == "finished" and info["live"] and "C" in info["gpu_services"]
    assert client.post("/api/runs/nope/start").status_code == 404


def _sleeper():
    # a process whose command line names the module, as a real background run's does
    return [sys.executable, "-c", "import time; time.sleep(30)", background.MODULE]


def test_one_run_at_a_time_and_stop(tmp_path):
    c = TestClient(create_app(tmp_path, frontend_dist=tmp_path / "no-dist",
                              start=lambda rd, live: background.start(rd, live=live, argv=_sleeper())))
    for rid in ("one", "two"):
        c.post("/api/runs", json={"run_id": rid, "n_meetings": 1, "overrides": FAKE})
    assert c.post("/api/runs/one/stop").status_code == 409
    started = c.post("/api/runs/one/start").json()
    assert started["process"]["state"] == "running"
    assert c.get("/api/runs/one/jobs").json()["process"] == "running"
    assert c.post("/api/runs/one/start").status_code == 409
    r = c.post("/api/runs/two/start")
    assert r.status_code == 409 and "one is running" in r.json()["detail"]["message"]
    assert c.post("/api/runs/one/stop").status_code == 200
    for _ in range(50):  # the sleeper dies on SIGINT without writing its end, so the record is lost
        if c.get("/api/runs/one/process").json()["process"]["state"] != "running":
            break
        time.sleep(0.1)
    assert c.get("/api/runs/one/process").json()["process"]["state"] == "lost"
    assert "started" in c.get("/api/runs/one/process").json()["log"]
    assert c.post("/api/runs/two/start").status_code == 200
    os.kill(c.get("/api/runs/two/process").json()["process"]["pid"], signal.SIGKILL)


def test_jobs_are_not_active_after_a_stop(viewer_run, tmp_path):
    import shutil

    from synthetics.orchestrator.status import StatusStore

    shutil.copytree(viewer_run, tmp_path / "s")
    rp = RunPaths(tmp_path / "s")
    StatusStore(rp.run_dir).start(rp.meeting_ids()[0], 8)
    c = TestClient(create_app(tmp_path, frontend_dist=tmp_path / "no-dist"))
    assert c.get("/api/runs/s/jobs").json()["active"] is True  # started from the CLI: stages decide
    write_model(rp.process, RunProcess(pid=os.getpid(), state="stopped", live=False, started_at="x"))
    jobs = c.get("/api/runs/s/jobs").json()
    assert jobs["active"] is False and jobs["running_stages"] == [] and jobs["process"] == "stopped"


@pytest.mark.parametrize(("raised", "state", "code"), [(None, "finished", 0), (KeyboardInterrupt, "stopped", 130),
                                                       (RuntimeError("boom"), "failed", 1)])
def test_background_main_records_how_the_run_ended(tmp_path, monkeypatch, raised, state, code):
    from synthetics.orchestrator import scheduler
    from synthetics.orchestrator.run import init_run

    rd = init_run("bg", n_meetings=1, seed=1, overrides=FAKE, root=tmp_path)
    write_model(RunPaths(rd).process, RunProcess(pid=os.getpid(), state="running", live=False, started_at="x"))

    class Sched:
        def __init__(self, *a, **kw):
            pass

        def run(self, **kw):
            if raised is not None:
                raise raised
            return scheduler.RunReport()

    monkeypatch.setattr(scheduler, "PhaseScheduler", Sched)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    assert background.main(rd) == code
    rec = read_model(RunPaths(rd).process, RunProcess)
    assert rec.state == state and rec.exit_code == code and rec.ended_at
    assert (rec.error == "RuntimeError: boom") if state == "failed" else rec.error is None


def test_read_only_viewer_has_no_write_routes(tmp_path):
    c = TestClient(create_app(tmp_path, frontend_dist=tmp_path / "no-dist", read_only=True))
    assert c.get("/api/viewer").json()["read_only"] is True
    assert c.post("/api/preview", json={}).status_code in (404, 405)
    assert c.post("/api/runs", json={"run_id": "x"}).status_code in (404, 405)
    assert c.get("/api/defaults").status_code == 404
