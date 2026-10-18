"""M2: pool validation, extraction from a synthetic SDM zip, cleanup with fake services, readiness."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import ROOT
from synthetics.artifacts.io import read_model
from synthetics.artifacts.models import SpeakerSplits
from synthetics.backends.registry import Backends
from synthetics.config.loader import load_run_config
from synthetics.config.schema import Role
from synthetics.nsc import cleanup, extract
from synthetics.nsc.pool import load_pool, speaker_splits, validate_pool
from synthetics.nsc.settings import NscPaths

POOL = ROOT / "data/nsc_part3_reference_speakers.csv"


def test_the_pinned_pool_is_valid():
    pool = load_pool(POOL)
    assert validate_pool(pool) == []
    assert len(pool) == 175


def test_pool_validation_catches_problems(tmp_path):
    with open(POOL, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["split"] = "dev"  # split sizes change, and a partner pair may be split
    partner = next(r for r in rows if r["speaker"] == rows[1]["partner"])
    partner["split"] = "test" if rows[1]["split"] != "test" else "train"
    rows.append(dict(rows[2]))  # duplicate
    path = tmp_path / "pool.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    problems = " | ".join(validate_pool(load_pool(path), tmp_path))
    assert "duplicate speakers" in problems and "split sizes" in problems
    assert "partner pair split across splits" in problems and "zips missing" in problems


def test_speaker_splits_copy_the_csv():
    s = speaker_splits(POOL, "p3-175-v1")
    assert s.counts == {"dev": 25, "test": 100, "train": 50}
    assert all(v.split in ("train", "dev", "test") for v in s.speakers.values())
    assert len(s.in_split("dev")) == 25 and s.pool_sha256


def test_init_run_writes_speaker_splits(tmp_path):
    from synthetics.orchestrator.run import init_run

    rd = init_run("r", n_meetings=2, seed=1, overrides={"backends.mode": "fake"}, root=tmp_path)
    s = read_model(rd / "speaker_splits.json", SpeakerSplits)
    assert s.pool_version == "p3-175-v1" and len(s.speakers) == 175
    assert sorted(p.name for p in (rd / "meetings").iterdir()) == ["m-000001", "m-000002"]


# ---------------------------------------------------------------- extraction and cleanup on a synthetic zip


def wav_bytes(seconds: float, freq: float = 220.0, sr: int = 16000) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def nsc(tmp_path) -> NscPaths:
    spk = "00860519"
    entries = [
        ("00860519_0001", 0.0, 6.0, "so are we going to (um) wait for instructions [lah] then we go makan"),
        ("00860519_0002", 10.0, 18.5, "ok [lor] I think the one wearing the green shirt is the boss one"),
        ("00860519_0003", 20.0, 22.0, "too short to be a reference clip"),  # < 5 s
        ("00860519_0004", 30.0, 38.0, "we went to #Bishan# yesterday to see the new flat with them"),  # name
        ("00860519_0005", 40.0, 48.0, "few words here"),  # < 8 words
        ("00860519_0006", 50.0, 57.0, "(ppl) true [ah] the vendor said can deliver by next week only"),
        ("00860519_0007", 60.0, 76.0, "this one is much too long to be used as a reference clip ok"),  # > 15 s
    ]
    zpath = tmp_path / "NSC" / "SDM" / f"SPEAKER{spk}.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(f"SPEAKER{spk}/manifest.json", "\n".join(json.dumps(
            {"conf": "2692", "pair_id": pid, "source_start": a, "source_end": b, "text": text})
            for pid, a, b, text in entries))
        for pid, a, b, _ in entries:
            z.writestr(f"WAVE/{pid.replace('_', '')}.wav", wav_bytes(b - a, 200 + int(pid[-1]) * 50))
    with open(POOL, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["speaker"] == spk]
    pool = tmp_path / "pool.csv"
    with open(pool, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return NscPaths(nsc_dir=tmp_path / "NSC", refs_dir=tmp_path / "refs", pool_csv=pool)


def test_extract_keeps_only_candidates_in_our_notation(nsc):
    from synthetics.text import Lexicon

    sp = load_pool(nsc.pool_csv)[0]
    r = extract.extract_speaker(sp, nsc, Lexicon.load())
    assert r["candidates"] == 3
    rows = extract.load_candidates(nsc, sp.speaker)
    assert [x["pair_id"] for x in rows] == ["00860519_0001", "00860519_0002", "00860519_0006"]
    by = {x["pair_id"]: x for x in rows}
    assert by["00860519_0001"]["text"] == "so are we going to (um) wait for instructions (lah) then we go makan"
    assert by["00860519_0006"]["text"].startswith("[laugh] true (ah)")
    a, sr = sf.read(extract.wav_path(nsc, by["00860519_0002"]))
    assert sr == 16000 and abs(len(a) / sr - 8.5) < 0.01
    assert extract.extract_speaker(sp, nsc, Lexicon.load())["skipped"] is True


def test_clean_keeps_top_by_mos_and_writes_indexes(nsc):
    from synthetics.text import Lexicon

    sp = load_pool(nsc.pool_csv)[0]
    extract.extract_speaker(sp, nsc, Lexicon.load())
    cfg = load_run_config(overrides={"run_id": "x", "seed": 0, "n_meetings": 0, "backends.mode": "fake"})
    backends = Backends(cfg.backends, cleanup.ROLES)
    r = cleanup.clean_speaker(sp, nsc, backends, keep=2, max_candidates=0, log=lambda _: None)
    assert r["cleaned"] == 3 and r["kept"] == 2
    scores = [json.loads(x) for x in (nsc.clean_speaker(sp.speaker) / "scores.jsonl").read_text().splitlines()]
    assert len(scores) == 3
    index = [json.loads(x) for x in (nsc.clean_speaker(sp.speaker) / "index.jsonl").read_text().splitlines()]
    top2 = sorted(scores, key=lambda s: -s["mos"])[:2]
    assert [i["pair_id"] for i in index] == [s["pair_id"] for s in top2]
    for i in index:
        assert (nsc.refs_dir / i["clip"]).is_file() and i["split"] == sp.split and i["cleanup_version"] == cleanup.CLEAN_VERSION
    wavs = sorted(p.name for p in nsc.clean_speaker(sp.speaker).glob("*.wav"))
    assert len(wavs) == 2, "only kept clips stay on disk"
    # resumable: nothing is cleaned twice
    assert cleanup.clean_speaker(sp, nsc, backends, keep=2, max_candidates=0, log=lambda _: None)["cleaned"] == 0
    assert cleanup.write_global_index(nsc) == 2


def test_readiness(nsc):
    from synthetics.text import Lexicon

    sp = load_pool(nsc.pool_csv)[0]
    extract.extract_speaker(sp, nsc, Lexicon.load())
    cfg = load_run_config(overrides={"run_id": "x", "seed": 0, "n_meetings": 0, "backends.mode": "fake"})
    cleanup.clean_speaker(sp, nsc, Backends(cfg.backends, cleanup.ROLES), keep=3, max_candidates=0, log=lambda _: None)
    cleanup.write_global_index(nsc)
    res = cleanup.readiness(nsc, min_refs=3, meeting_size=0, mos_min=0.0)  # a one-speaker pool fills no meeting
    assert res["ok"] and res["failing_speakers"] == []
    res = cleanup.readiness(nsc, min_refs=4, meeting_size=0, mos_min=0.0)
    assert not res["ok"] and res["failing_speakers"] == [sp.speaker]
    assert res["replacements_needed"][0]["count"] == 1


def test_cleanup_roles_resolve_to_the_prep_services():
    cfg = load_run_config(overrides={"run_id": "x", "seed": 0, "n_meetings": 0})
    assert [cfg.backends.roles[r] for r in cleanup.ROLES] == ["deepfilternet3", "wpe-dereverb", "dnsmos"]
    assert set(cfg.phases.phases["prep"].roles) == {Role.ENHANCER, Role.DEREVERB, Role.REF_QUALITY}


def test_ranking_prefers_mid_length_clean_clips():
    rows = [{"pair_id": "a", "duration": 5.2, "text": "one two three four five six seven eight"},
            {"pair_id": "b", "duration": 9.0, "text": "(um) one two (lah) three four five six seven eight"},
            {"pair_id": "c", "duration": 9.5, "text": "one two three four five six seven eight nine"}]
    assert [r["pair_id"] for r in cleanup.rank_candidates(rows)] == ["c", "b", "a"]


def test_paths_default_outside_the_app():
    from synthetics.nsc.settings import nsc_paths

    p = nsc_paths()
    assert not str(p.refs_dir).startswith(str(ROOT)) and p.pool_csv == ROOT / "data/nsc_part3_reference_speakers.csv"
    assert isinstance(p.raw, Path)
