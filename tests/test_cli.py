"""CLI smoke test with fake backends: init-run, run, status, export-judge-set, sampler-report."""

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
