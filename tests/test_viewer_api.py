"""M12: the viewer's read-only API on a fake run of all eight stages."""

from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from synthetics.artifacts.paths import RunPaths
from synthetics.orchestrator.status import StatusStore
from synthetics.viewer.api import create_app


@pytest.fixture(scope="module")
def runs_root(viewer_run, tmp_path_factory):
    """The fake run, plus a copy whose second meeting failed stage 7 and whose first is still running stage 8."""
    root = tmp_path_factory.mktemp("api-runs")
    shutil.copytree(viewer_run, root / "t")
    shutil.copytree(viewer_run, root / "busy")
    rp = RunPaths(root / "busy")
    a, b = rp.meeting_ids()
    store = StatusStore(rp.run_dir)
    store.mark(b, 7, "fail", detail="RT60 off target")
    store.mark(b, 8, "pending")
    store.start(a, 8)
    lines = [x for x in rp.manifest.read_text().splitlines() if a not in x and b not in x]
    rp.manifest.write_text("".join(x + "\n" for x in lines))
    (root / "not-a-run").mkdir()
    return root


@pytest.fixture(scope="module")
def client(runs_root):
    return TestClient(create_app(runs_root, frontend_dist=runs_root / "no-dist"))


def test_runs_lists_only_run_directories(client):
    runs = {r["id"]: r for r in client.get("/api/runs").json()}
    assert set(runs) == {"t", "busy"}
    assert runs["t"]["meetings"] == 2 and runs["t"]["done"] == 2 and runs["t"]["seed"] == 21
    assert runs["busy"]["done"] == 0 and len(runs["t"]["config_hash"]) == 12 and runs["t"]["revision"] == 2


def test_config_manifest_and_summary(client):
    cfg = client.get("/api/runs/t/config").json()
    assert cfg["config"]["room_classes"]["ratio"]["weights"]["large"] == 1.0
    manifest = client.get("/api/runs/t/manifest").json()
    assert [e["meeting_id"] for e in manifest] == ["m-000001", "m-000002"]
    summary = client.get("/api/runs/t/summary").json()
    assert set(summary) == {"run_report", "voice_report"}
    assert client.get("/api/runs/t/speaker_splits").json()["counts"]


def test_meetings_join_the_manifest_with_status(client):
    done = client.get("/api/runs/t/meetings").json()
    assert [m["state"] for m in done] == ["done", "done"]
    m = done[0]
    assert m["manifest"]["meeting_id"] == m["meeting_id"] and set(m["stages"]) == {str(i) for i in range(1, 9)}
    assert m["stages"]["1"]["name"] == "Config" and m["stages"]["8"]["state"] == "pass"
    kinds = {p["position_type"] for p in m["plan"]["people"]}
    assert {"table", "presenter", "remote"} <= kinds and m["plan"]["room"]["front_wall"] == "W"
    assert "cache_key" not in m["stages"]["1"]

    busy = {x["meeting_id"]: x for x in client.get("/api/runs/busy/meetings").json()}
    a, b = sorted(busy)
    assert busy[a]["state"] == "running" and busy[b]["state"] == "failed"
    assert busy[b]["manifest"] is None and busy[b]["plan"] is not None  # failed meetings still get a card
    assert busy[b]["stages"]["7"]["detail"] == "RT60 off target"


def test_jobs_report_progress(client):
    jobs = client.get("/api/runs/busy/jobs").json()
    assert jobs["states"] == {"running": 1, "failed": 1} and jobs["active"] and jobs["running_stages"] == ["8"]
    assert jobs["stages"]["7"] == {"pass": 1, "fail": 1} and jobs["last_update"]
    assert client.get("/api/runs/t/jobs").json()["active"] is False


def test_artifacts_are_whitelisted(client):
    r = client.get("/api/runs/t/meetings/m-000001/meeting.json")
    assert r.status_code == 200 and r.json()["meeting_id"] == "m-000001"
    assert client.get("/api/runs/t/meetings/m-000001/labels/transcript.json").status_code == 200
    prompt = client.get("/api/runs/t/meetings/m-000001/prompts/S01.txt")
    assert prompt.status_code == 200 and prompt.text.startswith("# Persona S01")
    for bad in ("mixture.wav", "tracks/S01.wav", "../config.yaml", "..%2Fconfig.yaml", "prompts%2F..%2F..%2Fstatus.json",
                "prompts%2FS01.txt%2F..%2F..%2F..%2Fconfig.yaml", "status.json.bak"):
        assert client.get(f"/api/runs/t/meetings/m-000001/{bad}").status_code == 404, bad
    assert client.get("/api/runs/t/meetings/m-999999/meeting.json").status_code == 404
    assert client.get("/api/runs/t/meetings/..%2F..%2Ft/meeting.json").status_code == 404
    assert client.get("/api/runs/not-a-run/manifest").status_code == 404
    assert client.get("/api/runs/..%2Ft/manifest").status_code == 404


def test_no_write_routes_on_run_files(client):
    # the run builder (test_viewer_builder.py) creates runs with POST /api/runs; nothing writes a run's files
    for path in ("/api/runs", "/api/runs/t/manifest", "/api/runs/t/meetings/m-000001/meeting.json"):
        for method in ("post", "put", "patch", "delete"):
            if (method, path) == ("post", "/api/runs"):
                continue
            assert getattr(client, method)(path).status_code == 405, (method, path)


def test_export_slices_the_manifest_by_the_browser_filters(client):
    r = client.get("/api/runs/t/export")
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert 'filename="t-manifest-slice.jsonl"' in r.headers["content-disposition"]
    assert [json.loads(x)["meeting_id"] for x in r.text.splitlines()] == ["m-000001", "m-000002"]
    assert client.get("/api/runs/t/export", params={"room_class": "small"}).text == ""
    assert client.get("/api/runs/t/export", params={"min_speakers": 99}).headers["x-meetings"] == "0"
    one = client.get("/api/runs/t/export", params={"ids": "m-000002", "presenters": True, "remote": True})
    assert [json.loads(x)["meeting_id"] for x in one.text.splitlines()] == ["m-000002"]


def test_the_built_frontend_is_served_but_never_shadows_the_api(runs_root, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Synthetics</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    c = TestClient(create_app(runs_root, frontend_dist=dist))
    assert "Synthetics" in c.get("/").text and "Synthetics" in c.get("/runs/t/meetings/m-000001").text
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert c.get("/api/nothing").status_code == 404
    assert c.get("/../../etc/passwd").text.startswith("<!doctype")
