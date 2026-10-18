"""`bakeoff` and `calibrate-sv` in fake mode: the report shape, the room-colour pairing and the fitted overlay."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from synthetics.config import load_run_config
from synthetics.nsc.references import ReferenceIndex
from synthetics.nsc.settings import nsc_paths
from synthetics.validators.bakeoff import Bakeoff, run_bakeoff, select_speakers
from synthetics.validators.sv_calibration import eer, run_calibration
from voice_helpers import fake_refs


@pytest.fixture(scope="module")
def refs_root(tmp_path_factory):
    return fake_refs(tmp_path_factory.mktemp("refs"), clips=3, raw=True)


@pytest.fixture(autouse=True)
def env(refs_root, tmp_path, monkeypatch):
    monkeypatch.setenv("NSC_REFS_DIR", str(refs_root))
    monkeypatch.setenv("SOUND_LIBRARY_DIR", str(tmp_path / "none"))


def test_select_speakers_spreads_over_groups():
    refs = ReferenceIndex.load(nsc_paths())
    picked = select_speakers(refs, 12, seed=1)
    assert len(picked) == 12 == len(set(picked))
    groups = {(refs.clips(s)[0].gender, refs.clips(s)[0].ethnicity) for s in picked}
    assert len(groups) >= 6  # 8 gender x ethnicity groups in the pool
    assert picked == select_speakers(refs, 12, seed=1)


def test_bakeoff_report(tmp_path):
    rep = run_bakeoff(out_dir=tmp_path / "bo", mode="fake", n_speakers=4, targets=2, modes=["ultimate", "continuation"],
                      raw_speakers=1, lease=False, log=lambda _: None)
    m = rep["by_mode"]["ultimate"]
    assert m["clones"] == 8 and rep["by_mode"]["continuation"]["clones"] == 8
    assert m["sv_wavlm_to_ref"]["mean"] == pytest.approx(1.0) and m["sv_redimnet_id_accuracy"] == 1.0
    assert m["asr_primary_pass_rate"] == 1.0 and m["asr_limited_rate"] == 0.0
    assert m["stage5_pass_rate"] == m["mos_pass_rate"]  # fake MOS is unrelated to the reference's; the rest passes
    assert rep["real_baseline"]["recordings"] == 8
    assert rep["separability"]["sv_wavlm"]["pairs"] == 6
    assert rep["room_colour"]["pairs"] == 2
    assert (tmp_path / "bo/report.md").read_text().startswith("# Phase C bake-off")
    rows = [json.loads(x) for x in (tmp_path / "bo/clips.jsonl").read_text().splitlines()]
    assert len(rows) == 8 + 8 + 2 and {r["ref_kind"] for r in rows} == {"clean", "raw"}
    assert any(p.name.endswith("_raw.wav") for p in (tmp_path / "bo/listen").iterdir())


def test_bakeoff_resumes_after_a_crash(tmp_path, monkeypatch):
    kw = dict(mode="fake", n_speakers=4, targets=2, modes=["ultimate"], raw_speakers=1, lease=False, log=lambda _: None)
    whole = run_bakeoff(out_dir=tmp_path / "whole", **kw)

    clone, calls = Bakeoff._clone, []

    def counted(crash_at):
        def f(self, *a, **k):
            calls.append(1)
            if len(calls) == crash_at:
                raise RuntimeError("killed")
            return clone(self, *a, **k)
        return f

    # speaker 1 (raw too) makes 4 clones, the others 2: the crash comes in speaker 3
    monkeypatch.setattr(Bakeoff, "_clone", counted(crash_at=7))
    with pytest.raises(RuntimeError):
        run_bakeoff(out_dir=tmp_path / "bo", **kw)
    assert len((tmp_path / "bo/progress.jsonl").read_text().splitlines()) == 1 + 2

    calls.clear()
    monkeypatch.setattr(Bakeoff, "_clone", counted(crash_at=0))
    resumed = run_bakeoff(out_dir=tmp_path / "bo", **kw)
    assert len(calls) == 4  # only speakers 3 and 4

    def rows(d):  # a fake clone's MOS hashes its samples, which carry a process-wide call counter
        return [{k: v for k, v in json.loads(x).items() if k != "mos"} for x in (d / "clips.jsonl").read_text().splitlines()]

    assert rows(tmp_path / "bo") == rows(tmp_path / "whole")
    assert resumed["separability"] == whole["separability"] and resumed["real_baseline"] == whole["real_baseline"]

    calls.clear()
    run_bakeoff(out_dir=tmp_path / "bo", **{**kw, "targets": 1})  # other settings: starts over
    assert len(calls) == 4 + 1 and (tmp_path / "bo/progress.jsonl.old").is_file()


def test_calibrate_sv_writes_an_overlay_the_config_accepts(tmp_path):
    out = tmp_path / "sv.yaml"
    res = run_calibration(mode="fake", out=out, lease=False, speakers=ReferenceIndex.load(nsc_paths()).speakers()[:20],
                          log=lambda _: None)
    for m in ("sv_wavlm", "sv_redimnet"):
        f = res["models"][m]
        assert f["eer"] < 0.05 and f["same"]["mean"] == pytest.approx(1.0) and f["same"]["n"] == 20 * 3
        assert f["diff"]["n"] == 20 * 19 // 2
    overlay = yaml.safe_load(out.read_text())
    cfg = load_run_config(out, overrides={"run_id": "x", "seed": 1, "n_meetings": 1})
    assert cfg.thresholds.sv == overlay["thresholds"]["sv"]
    assert cfg.thresholds.sv_consistency_min == overlay["thresholds"]["sv_consistency_min"]
    assert json.loads(out.with_suffix(".json").read_text())["clips"] == 60


def test_eer_on_separated_and_overlapping_scores():
    assert eer(np.array([0.9, 0.8, 0.85]), np.array([0.1, 0.2]))[0] == 0.0
    e, t = eer(np.array([0.5, 0.6, 0.7, 0.8]), np.array([0.55, 0.65, 0.1, 0.2]))
    assert 0 < e < 0.5 and 0.5 <= t <= 0.7
