"""Growing a voice pool: base splits kept, partners together, the rest balanced; sidecar counts; main clips."""

from __future__ import annotations

from collections import Counter

import pytest

from conftest import ROOT, make_cfg
from synthetics.nsc.pool import check_pool, load_pool, pool_meta, speaker_splits
from synthetics.nsc.pool_build import MIN_CLIPS, draft_pool, publish
from synthetics.nsc.references import ReferenceIndex, references_for
from voice_helpers import new_speaker, voice_folder

V1 = ROOT / "data/nsc_part3_reference_speakers.csv"
V2 = ROOT / "data/pools/p3-406-v2.csv"
MIX = {"train": 0.7, "dev": 0.15, "test": 0.15}
@pytest.fixture
def folder(tmp_path):
    v1 = load_pool(V1)
    partnered = next(s for s in v1 if s.split == "dev")
    rows = [new_speaker("90000001", partner_id=partnered.speaker),  # joins the dev partner
            new_speaker("90000002", partner_id="90000003"), new_speaker("90000003", partner_id="90000002"),
            new_speaker("90000004", _clips=MIN_CLIPS - 1),  # too few clips
            new_speaker("90000005", gender=""),  # sheet incomplete
            *(new_speaker(f"9100{i:04d}", gender="FM"[i % 2], ethnicity=("CHINESE", "MALAY")[i % 2])
              for i in range(40))]
    return voice_folder(tmp_path / "voices", rows), partnered


def test_base_speakers_keep_their_split_and_partners_stay_together(folder):
    root, partnered = folder
    d = draft_pool(V1, root, split_mix=MIX, seed=3)
    base = {s.speaker: s.split for s in load_pool(V1)}
    new = {s.speaker: s.split for s in d.added}
    assert {s.speaker: s.split for s in d.base} == base
    assert new["90000001"] == partnered.split and "90000001" in d.forced
    assert new["90000002"] == new["90000003"]
    assert set(d.left_out) == {"90000004", "90000005"}
    assert "clips" in d.left_out["90000004"] and "gender" in d.left_out["90000005"]


def test_free_speakers_follow_the_split_mix_within_each_group(folder):
    root, _ = folder
    d = draft_pool(V1, root, split_mix=MIX, seed=3)
    free = [s for s in d.added if s.speaker.startswith("9100")]
    for group in ({"F", "CHINESE"}, {"M", "MALAY"}):
        c = Counter(s.split for s in free if {s.gender, s.ethnicity} == group)
        assert c["train"] == 14 and c["dev"] + c["test"] == 6  # 20 per group at 70/15/15
    assert draft_pool(V1, root, split_mix=MIX, seed=3).added == d.added  # seeded: same inputs, same pool


def test_publish_writes_a_pool_its_sidecar_validates(folder, tmp_path):
    root, _ = folder
    d = draft_pool(V1, root, split_mix=MIX, seed=3)
    out = tmp_path / "pools/p3-test-v2.csv"
    meta = publish(d, out, version="p3-test-v2", base_path=V1, base_version="p3-175-v1", folder=root, seed=3,
                   split_mix=MIX, main_clips={"90000001": "90000001_0003"}, exclude={"90000002"})
    pool = {s.speaker: s for s in load_pool(out)}
    assert "90000002" not in pool and "90000003" not in pool  # excluding one of a pair drops both
    assert pool["90000001"].main_clip == "90000001_0003"
    assert meta.total == len(pool) == 175 + 41 and pool_meta(out) == meta
    assert check_pool(out) == []
    sp = speaker_splits(out, "p3-test-v2")
    assert sp.counts == meta.counts and sum(sp.counts.values()) == len(pool)


def test_the_first_pool_still_validates_without_a_sidecar():
    assert pool_meta(V1) is None and check_pool(V1) == []


def test_the_grown_pool_in_the_repo_is_valid_and_extends_the_first():
    assert check_pool(V2) == []
    meta = pool_meta(V2)
    v1 = {s.speaker: s.split for s in load_pool(V1)}
    v2 = {s.speaker: s.split for s in load_pool(V2)}
    assert meta.total == len(v2) == 406 and meta.base == "p3-175-v1"
    assert all(v2[k] == split for k, split in v1.items())


def test_a_voice_folder_manifest_loads_as_a_reference_index(folder):
    root, _ = folder
    idx = ReferenceIndex.load_manifest(root / "manifest.jsonl")
    assert idx.best("90000001").pair_id == "90000001_0001"
    assert idx.best("90000001").path == root / "clean/90000001/90000001_0001.wav"
    chosen = idx.with_main_clips({"90000001": "90000001_0003"})
    assert chosen.best("90000001").pair_id == "90000001_0003" and chosen.sha256 != idx.sha256
    assert idx.with_main_clips({}) is idx
    with pytest.raises(KeyError):
        idx.with_main_clips({"90000001": "nope"})


def test_runs_without_refs_index_keep_the_shared_index(tmp_path, monkeypatch):
    from voice_helpers import fake_refs

    monkeypatch.setenv("NSC_REFS_DIR", str(fake_refs(tmp_path / "refs")))
    cfg = make_cfg()
    assert cfg.speaker_pool.refs_index is None
    from synthetics.nsc.settings import nsc_paths

    assert references_for(cfg).sha256 == ReferenceIndex.load(nsc_paths()).sha256


def test_new_runs_default_to_the_grown_pool():
    from synthetics.config.loader import load_run_config

    sp = load_run_config(overrides={"run_id": "x", "seed": 1, "n_meetings": 1}).speaker_pool
    assert sp.version == "p3-406-v2" and str(sp.path) == "data/pools/p3-406-v2.csv"
    assert sp.refs_index is not None and str(sp.refs_index).endswith("manifest.jsonl")
