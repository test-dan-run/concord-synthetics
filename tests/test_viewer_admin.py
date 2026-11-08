"""Speaker pool admin routes: import a voice folder as a draft, review it, publish a pool; off without --admin."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import make_cfg
from synthetics.nsc.pool import check_pool, load_pool, pool_meta
from synthetics.viewer import admin
from synthetics.viewer.api import create_app
from voice_helpers import new_speaker, voice_folder


@pytest.fixture
def setup(tmp_path):
    rows = [new_speaker("90000002", partner_id="90000003"), new_speaker("90000003", partner_id="90000002"),
            new_speaker("90000004", _clips=1), *(new_speaker(f"9100{i:04d}") for i in range(6))]
    folder = voice_folder(tmp_path / "voices", rows)
    (folder / "clean/91000000").mkdir(parents=True)
    (folder / "clean/91000000/91000000_0001.wav").write_bytes(b"RIFF....WAVE")
    app = FastAPI()
    admin.add_routes(app, tmp_path / "runs", make_cfg, tmp_path / "pools")  # base: the first pool
    return TestClient(app), folder, tmp_path / "pools"


def test_import_review_and_publish(setup):
    c, folder, pools = setup
    r = c.post("/api/admin/pool/drafts", json={"source": str(folder), "version": "p3-test-v2"})
    assert r.status_code == 201, r.text
    d = r.json()
    by = {s["speaker"]: s for s in d["speakers"]}
    assert by["90000004"]["status"] == "left_out" and "clips" in by["90000004"]["reason"]
    assert by["90000002"]["split"] == by["90000003"]["split"]
    assert d["ready"] == 8 and sum(d["counts"].values()) == 175 + 8

    r = c.patch("/api/admin/pool/draft/speakers/91000000", json={"main_clip": "91000000_0002"})
    assert r.json()["speakers"][3]["main_clip"] == "91000000_0002"
    assert c.patch("/api/admin/pool/draft/speakers/91000000", json={"main_clip": "nope"}).status_code == 422
    r = c.patch("/api/admin/pool/draft/speakers/90000002", json={"excluded": True})
    by = {s["speaker"]: s for s in r.json()["speakers"]}
    assert by["90000002"]["status"] == by["90000003"]["status"] == "excluded"  # a pair goes together

    r = c.post("/api/admin/pool/draft/publish")
    assert r.status_code == 200, r.text
    out = pools / "p3-test-v2.csv"
    assert check_pool(out) == [] and pool_meta(out).total == 175 + 6
    assert {s.speaker: s.main_clip for s in load_pool(out)}["91000000"] == "91000000_0002"
    assert c.get("/api/admin/pool/draft").status_code == 404  # the draft is gone once published


def test_versions_are_never_reused(setup):
    c, folder, pools = setup
    c.post("/api/admin/pool/drafts", json={"source": str(folder), "version": "p3-test-v2"})
    c.post("/api/admin/pool/draft/publish")
    assert c.post("/api/admin/pool/drafts", json={"source": str(folder), "version": "p3-test-v2"}).status_code == 422
    assert c.post("/api/admin/pool/drafts", json={"source": str(folder / "nope")}).status_code == 422


def test_pool_info_and_clip_audio(setup):
    c, folder, _ = setup
    info = c.get("/api/admin/pool").json()
    assert info["speakers"] == 175 and info["counts"] == {"train": 50, "dev": 25, "test": 100}
    assert c.get("/api/admin/voices/91000000/91000000_0001").status_code == 404  # no draft yet
    c.post("/api/admin/pool/drafts", json={"source": str(folder)})
    assert c.get("/api/admin/voices/91000000/91000000_0001").content.startswith(b"RIFF")
    assert c.get("/api/admin/voices/91000000/..%2F..%2Fspeakers").status_code == 404
    assert c.get("/api/admin/voices/91000001/91000000_0001").status_code == 404


def test_admin_routes_are_off_by_default_and_in_read_only(tmp_path):
    for kw in ({}, {"admin": True, "read_only": True}):
        c = TestClient(create_app(tmp_path, tmp_path / "no-dist", **kw))
        assert c.get("/api/admin/pool").status_code == 404
        assert c.get("/api/viewer").json()["admin"] is False
    c = TestClient(create_app(tmp_path, tmp_path / "no-dist", admin=True, pools_dir=tmp_path / "pools"))
    assert c.get("/api/viewer").json()["admin"] is True and c.get("/api/admin/pool").status_code == 200
