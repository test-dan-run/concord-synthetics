"""M10: the UI data contract (UI_DESIGN_DOCUMENT.md, "Data contract") holds on a fake run of all eight stages, and
on the real m9-e2e run when it is present."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import ROOT
from synthetics.artifacts.io import read_model, write_model
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.paths import RunPaths
from synthetics.orchestrator.cli import app
from synthetics.viewer.contract import check_run

# every row the viewer depends on; a row that is never checked would pass vacuously
EXPECTED = {
    ("manifest.jsonl", "position counts: table, wall-seated, presenters, remote"),
    ("manifest.jsonl", "per-speaker position type"),
    ("manifest.jsonl", "tags: mic type, table shape"),
    ("manifest.jsonl", "hybrid"),
    ("manifest.jsonl", "RT60: sampled and measured"),
    ("manifest.jsonl", "SNR: sampled and measured"),
    ("manifest.jsonl", "turn-taking: overlap share, backchannels per minute, interruption share"),
    ("config.yaml", "room-class ratio"),
    ("config.yaml", "per-class ranges: speakers, RT60, SNR"),
    ("config.yaml", "hybrid and presentation probabilities"),
    ("meeting.json", "table positions"),
    ("meeting.json", "wall positions"),
    ("meeting.json", "presenter positions"),
    ("meeting.json", "remote positions"),
    ("meeting.json", "presentation segment and return seats"),
    ("meeting.json", "fit resample count"),
    ("status.json", "per-stage state, timings, attempts, detail"),
    ("status.json", "all eight stages recorded"),
    ("script.json", "turns: speaker, interaction, text, text_norm"),
    ("script_validation.json", "judge persona-consistency score per speaker"),
    ("voice_validation.json", "both speaker-verification scores per utterance"),
    ("timeline.json", "turn start times"),
    ("speaker_splits.json", "NSC reference speaker per persona, in the meeting's split"),
    ("prompts/<speaker_id>.txt", "rendered persona prompt per speaker, naming the position"),
}


@pytest.fixture(scope="module")
def run_dir(viewer_run):
    return viewer_run


def test_every_contract_row_holds_on_a_fake_run(run_dir):
    rows = check_run(run_dir)
    bad = [f"{r.artifact} / {r.fields}: {r.problems[:3]}" for r in rows if not r.ok]
    assert not bad, bad
    checked = {(r.artifact, r.fields) for r in rows if r.checked}
    assert checked >= EXPECTED, sorted(EXPECTED - checked)
    assert RunPaths(run_dir).manifest.read_text().strip(), "no meeting reached the manifest"


def test_a_broken_artifact_fails_its_row(run_dir, tmp_path):
    import shutil

    rd = tmp_path / "broken"
    shutil.copytree(run_dir, rd)
    mid = RunPaths(rd).meeting_ids()[0]
    p = RunPaths(rd).meeting(mid)
    p.prompt("S01").unlink()
    m = read_model(p.meeting, MeetingSpec)
    write_model(p.meeting, m.model_copy(update={"fit_resamples": m.fit_resamples + 1}))
    failed = {(r.artifact, r.fields) for r in check_run(rd) if not r.ok}
    assert ("prompts/<speaker_id>.txt", "rendered persona prompt per speaker, naming the position") in failed
    assert ("manifest.jsonl", "fit resample count") in failed  # the manifest no longer agrees with meeting.json
    assert CliRunner().invoke(app, ["viewer-contract", "-r", str(rd)]).exit_code == 1


def test_the_cli_reports_rows_as_json(run_dir):
    out = CliRunner().invoke(app, ["viewer-contract", "-r", str(run_dir), "--json"])
    assert out.exit_code == 0, out.output
    rows = json.loads(out.output)
    assert rows and all(r["ok"] for r in rows)


REAL = ROOT / "runs" / "m9-e2e"


@pytest.mark.skipif(not (REAL / "manifest.jsonl").is_file(), reason="the real M9 run is not on this machine")
def test_the_contract_holds_on_the_real_m9_run():
    rows = check_run(Path(REAL))
    assert [r for r in rows if not r.ok] == []
    checked = {(r.artifact, r.fields) for r in rows if r.checked}
    assert checked >= EXPECTED, sorted(EXPECTED - checked)
