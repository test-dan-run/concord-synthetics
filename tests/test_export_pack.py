"""Export packs: 16 kHz mono 16-bit WAVs, clips cut at the timeline's bounds and listed with the transcript's text,
full-length speakers lined up with the recording, kept meetings only, and the routes that start and report them."""

from __future__ import annotations

import csv
import io
import shutil
import time
import zipfile

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from synthetics.artifacts.io import read_model
from synthetics.artifacts.labels import Timeline, Transcript
from synthetics.artifacts.paths import RunPaths
from synthetics.export.pack import PackOptions, export_run
from synthetics.viewer.api import create_app

ALL = PackOptions(clips=True, full=True, clean=True, room=True, noise=True)


@pytest.fixture(scope="module")
def run_dir(viewer_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    shutil.copytree(viewer_run, root / "runs/t")
    return root / "runs/t"


def test_a_pack_has_every_part_in_16k_mono(run_dir, tmp_path):
    ids = RunPaths(run_dir).meeting_ids()
    res = export_run(run_dir, tmp_path / "pack", ids, ALL)
    assert res["meetings"] == len(ids) and (tmp_path / "pack/README.md").is_file()
    mid = ids[0]
    d = tmp_path / "pack" / mid
    tl = read_model(RunPaths(run_dir).meeting(mid).timeline, Timeline)
    tr = {u.turn_id: u for u in read_model(RunPaths(run_dir).meeting(mid).transcript, Transcript).utterances}
    mix = sf.info(d / "mixture.wav")
    for f in [d / "mixture.wav", d / "noise.wav", *d.glob("speakers/*/*.wav"), *d.glob("clips/*/*.wav")]:
        info = sf.info(f)
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16"), f
    speakers = {s.speaker_id for s in tl.segments}
    for sound in ("clean", "room"):
        assert {f.stem for f in (d / "speakers" / sound).glob("*.wav")} == speakers
        assert all(sf.info(f).frames == mix.frames for f in (d / "speakers" / sound).glob("*.wav"))
    rows = list(csv.DictReader((d / "clips/clips.csv").open()))
    assert len(rows) == len(tl.segments) == len(list((d / "clips/clean").glob("*.wav")))
    for r in rows:
        turn = r["clip"].removesuffix(".wav")
        if turn in tr:
            assert r["text"] == tr[turn].text
        seg = next(s for s in tl.segments if s.turn_id == turn)
        assert sf.info(d / "clips/clean" / r["clip"]).frames == round(seg.end * 16000) - round(seg.start * 16000)


def test_routes_estimate_export_kept_meetings_and_zip_one(run_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHETICS_EXPORT_ROOT", str(tmp_path / "exports"))
    c = TestClient(create_app(run_dir.parent, frontend_dist=tmp_path / "no-dist"))
    ids = RunPaths(run_dir).meeting_ids()
    est = c.get("/api/runs/t/exports/estimate").json()
    assert est["meetings"] == len(ids) and est["bytes"]["full"] > est["bytes"]["clips_clean"] > 0
    c.post("/api/runs/t/review", json={"meeting_id": ids[0], "decision": "exclude"})
    assert c.get("/api/runs/t/exports/estimate").json()["meetings"] == len(ids) - 1
    job = c.post("/api/runs/t/exports", json={"options": {"clips": True, "noise": True}}).json()
    for _ in range(200):
        j = next(x for x in c.get("/api/runs/t/exports").json() if x["id"] == job["id"])
        if j["state"] != "running":
            break
        time.sleep(0.1)
    assert j["state"] == "finished", j
    out = tmp_path / "exports" / job["id"]
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ids[1:]  # the left-out meeting is not packed
    z = c.get(f"/api/runs/t/meetings/{ids[1]}/download", params={"clips": False, "full": True, "room": False})
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert f"{ids[1]}/mixture.wav" in names and any(n.startswith(f"{ids[1]}/speakers/clean/") for n in names)
    assert c.post("/api/runs/t/exports", json={"meetings": ["m-999999"]}).status_code == 422
