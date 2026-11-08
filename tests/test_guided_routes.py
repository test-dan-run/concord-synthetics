"""Guided-flow routes: pilot runs (create, delete) and the reviewer's voice swap, which stage 4 then applies."""

from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import V1_POOL
from synthetics.artifacts.io import read_model
from synthetics.artifacts.voice import Synthesis
from synthetics.config.schema import StageId
from synthetics.orchestrator.lease import GpuLease
from synthetics.orchestrator.run import load_run
from synthetics.orchestrator.scheduler import PhaseScheduler
from synthetics.orchestrator.status import StatusStore
from synthetics.viewer.api import create_app
from voice_helpers import fake_refs


class Launcher:
    def __init__(self):
        self.calls = []

    def __call__(self, run_dir, *, live):
        self.calls.append((run_dir.name, live))


@pytest.fixture
def env(tmp_path, monkeypatch, viewer_run):
    monkeypatch.setenv("NSC_REFS_DIR", str(fake_refs(tmp_path / "refs")))
    monkeypatch.setenv("SOUND_LIBRARY_DIR", str(tmp_path / "no-library"))
    root = tmp_path / "runs"
    shutil.copytree(viewer_run, root / "t")
    return TestClient(create_app(root, frontend_dist=tmp_path / "no-dist", start=Launcher())), root


def test_a_pilot_is_the_drafts_first_meetings_in_one_wave(env):
    c, root = env
    draft = {"run_id": "g1", "n_meetings": 40, "seed": 5, "overrides": {**V1_POOL, "backends.mode": "fake"},
             "pilot": True}
    r = c.post("/api/runs", json=draft)
    assert r.status_code == 201, r.text
    assert r.json()["id"] == "g1-pilot"
    cfg, _ = load_run(root / "g1-pilot")
    assert cfg.n_meetings == 3 and cfg.phases.wave_size == 3 and cfg.seed == 5
    assert next(x for x in c.get("/api/runs").json() if x["id"] == "g1-pilot")["pilot"] is True
    full = c.post("/api/runs", json={**draft, "pilot": False})
    assert full.status_code == 201 and load_run(root / "g1")[0].n_meetings == 40
    assert c.delete("/api/runs/g1").status_code == 409  # only pilots are deleted here
    assert c.delete("/api/runs/g1-pilot").status_code == 204 and not (root / "g1-pilot").exists()


def test_a_voice_swap_changes_one_speaker_on_the_next_start(env):
    c, root = env
    rd = root / "t"
    syn = read_model(rd / "meetings/m-000001/synthesis.json", Synthesis)
    before = {k: v.nsc_speaker for k, v in syn.voices.items()}
    r = c.post("/api/runs/t/meetings/m-000001/voice-swap", json={"speaker_id": "S02"})
    assert r.status_code == 200, r.text
    swap = r.json()["swap"]
    assert swap["from_speaker"] == before["S02"] and swap["to_speaker"] not in before.values()
    st = StatusStore(rd).get("m-000001").get(StageId.VOICE)
    assert st.state == "retry" and json.loads(st.feedback)["swaps"][0]["to_speaker"] == swap["to_speaker"]
    assert c.post("/api/runs/t/meetings/m-000001/voice-swap", json={"speaker_id": "S99"}).status_code == 404

    cfg, _ = load_run(rd)
    PhaseScheduler(rd, cfg, lease_factory=lambda s, **kw: GpuLease(s, **{**kw, "enabled": False})).run(
        until=StageId.LABELS, meetings=["m-000001"])
    after = {k: v.nsc_speaker for k, v in read_model(rd / "meetings/m-000001/synthesis.json", Synthesis).voices.items()}
    assert after["S02"] == swap["to_speaker"]
    assert {k: v for k, v in after.items() if k != "S02"} == {k: v for k, v in before.items() if k != "S02"}
