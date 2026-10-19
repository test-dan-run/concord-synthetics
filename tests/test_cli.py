"""CLI smoke test with fake backends: init-run, run, status, export-judge-set, sampler-report, voice-report."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from synthetics.orchestrator.cli import app


def test_cli_end_to_end_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHETICS_RUNS", str(tmp_path))
    r = CliRunner()
    out = r.invoke(app, ["init-run", "cli", "--n", "2", "--seed", "4", "--set", "backends.mode=fake",
                         "--set", "dynamics.duration_min.large={dist: uniform, lo: 6, hi: 8}"])
    assert out.exit_code == 0, out.output
    out = r.invoke(app, ["run", "-r", "cli"])
    assert out.exit_code == 0, out.output
    summary = json.loads(out.output[out.output.index("{"):])
    assert summary["scripts_passed"] == 2
    out = r.invoke(app, ["status", "-r", "cli", "--meetings"])
    assert out.exit_code == 0 and "m-000002" in out.output
    out = r.invoke(app, ["export-judge-set", "-r", "cli", "--n", "2"])
    assert out.exit_code == 0 and (tmp_path / "cli/judge_sets/judge_set_1/sheet.csv").is_file()


def test_sampler_report_command(tmp_path):
    out = CliRunner().invoke(app, ["sampler-report", "--n", "300", "--out", str(tmp_path / "checks.json")])
    assert out.exit_code == 0, out.output
    assert "checks within tolerance" in out.output
    assert all(c["ok"] for c in json.loads((tmp_path / "checks.json").read_text()))


def test_voice_report_command(tmp_path, monkeypatch):
    from voice_helpers import fake_refs

    monkeypatch.setenv("SYNTHETICS_RUNS", str(tmp_path))
    monkeypatch.setenv("NSC_REFS_DIR", str(fake_refs(tmp_path / "refs")))
    monkeypatch.setenv("SOUND_LIBRARY_DIR", str(tmp_path / "no-library"))
    r = CliRunner()
    out = r.invoke(app, ["init-run", "vr", "--n", "1", "--seed", "4", "--set", "backends.mode=fake",
                         "--set", "thresholds.sv_pair_max={sv_wavlm: 0.85, sv_redimnet: 0.75}",
                         "--set", "thresholds.mos_drop_max=5.0",
                         "--set", "dynamics.duration_min.small={dist: uniform, lo: 4, hi: 6}",
                         "--set", "dynamics.duration_min.large={dist: uniform, lo: 6, hi: 8}"])
    assert out.exit_code == 0, out.output
    assert r.invoke(app, ["run", "-r", "vr", "--until", "5"]).exit_code == 0
    out = r.invoke(app, ["voice-report", "-r", "vr", "--min-meetings", "1", "--out", str(tmp_path / "c.json")])
    assert out.exit_code == 0, out.output
    res = json.loads((tmp_path / "c.json").read_text())
    assert res["ran"] == 1 and res["pass_rate"] == 1.0 and res["utterances"] > 0
    assert "1 of 1 meetings through phase C" in out.output
    assert r.invoke(app, ["voice-report", "-r", "vr"]).exit_code == 1  # fewer than 10
