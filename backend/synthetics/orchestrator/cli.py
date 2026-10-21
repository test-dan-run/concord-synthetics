"""`synthetics` CLI. Every command is idempotent and reads status.json to decide what is left."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from ..config.schema import Phase, StageId

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
nsc_app = typer.Typer(no_args_is_help=True, help="NSC Part 3 reference data (M2).")
app.add_typer(nsc_app, name="nsc")

RunOpt = Annotated[str, typer.Option("--run", "-r", help="run id under SYNTHETICS_RUNS, or a run directory")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "openai", "httpcore", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _run_dir(run: str) -> Path:
    from .run import runs_root

    p = Path(run)
    return p if (p / "config.yaml").is_file() else runs_root() / run


def _parse_overrides(items: list[str]) -> dict:
    import yaml

    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out


def _load_dotenv() -> None:
    """The app's .env, without overriding variables already set (as `just` does)."""
    import os

    from .run import APP_ROOT

    path = APP_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.removeprefix("export ").partition("=")
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _load_dotenv()
    _setup_logging(verbose)


@app.command("init-run")
def init_run_cmd(run_id: Annotated[str, typer.Argument()],
                 n: Annotated[int, typer.Option("--n", help="number of meetings")] = 20,
                 seed: Annotated[int, typer.Option()] = 1,
                 config: Annotated[list[Path], typer.Option("--config", "-c", help="YAML overlays, in order")] = [],  # noqa: B006
                 set_: Annotated[list[str], typer.Option("--set", help="dotted override, e.g. backends.mode=fake")] = []):  # noqa: B006
    """Create runs/<run_id>: frozen config.yaml, speaker_splits.json and empty meeting folders."""
    from .run import init_run

    run_dir = init_run(run_id, n_meetings=n, seed=seed, configs=list(config), overrides=_parse_overrides(set_))
    typer.echo(f"created {run_dir}")


@app.command()
def splits(run: RunOpt):
    """Copy the fixed splits from the pool CSV into the run's speaker_splits.json."""
    from .run import load_run, write_splits

    rd = _run_dir(run)
    cfg, _ = load_run(rd)
    path = write_splits(rd, cfg)
    data = json.loads(path.read_text())
    typer.echo(f"wrote {path}: pool {data['pool_version']}, counts {data['counts']}")


def _scheduler(run: str, keep_services: bool):
    from .run import load_run
    from .scheduler import PhaseScheduler

    rd = _run_dir(run)
    cfg, _ = load_run(rd)
    return rd, PhaseScheduler(rd, cfg, keep_services=keep_services)


@app.command()
def run(run: RunOpt,
        until: Annotated[int, typer.Option(help="last stage to run (1-8)")] = 8,
        only: Annotated[str | None, typer.Option(help="one phase only: CPU, A, B, B2, C or D")] = None,
        keep_services: Annotated[bool, typer.Option(help="leave leased services running afterwards")] = False):
    """Run the pipeline phase by phase, then rebuild run_report.json."""
    from .run_report import write_run_report
    from .scheduler import summarise

    rd, sched = _scheduler(run, keep_services)
    rep = sched.run(until=StageId(until), only=Phase(only) if only else None)
    write_run_report(rd, rep.phase_seconds)
    typer.echo(json.dumps(summarise(rd, StageId(until)), indent=2))


@app.command()
def resume(run: RunOpt, keep_services: Annotated[bool, typer.Option()] = False):
    """Continue a run where it stopped (same as `run`; finished stages are cache hits)."""
    from .run_report import write_run_report
    from .scheduler import summarise

    rd, sched = _scheduler(run, keep_services)
    rep = sched.run(until=StageId.LABELS)
    write_run_report(rd, rep.phase_seconds)
    typer.echo(json.dumps(summarise(rd, StageId.LABELS), indent=2))


@app.command()
def stage(n: Annotated[int, typer.Argument(help="stage number")], run: RunOpt,
          meeting: Annotated[str, typer.Option("--meeting", "-m")],
          force: Annotated[bool, typer.Option(help="ignore the cache")] = False):
    """Debug one stage on one meeting (services must already be up)."""
    from ..backends.registry import Backends
    from ..stages.runner import run_stage
    from .phase_c import stages as phase_c_stages
    from .phase_post import post_stages
    from .run import load_run
    from .scheduler import stages

    rd = _run_dir(run)
    cfg, _ = load_run(rd)
    every = {**stages(), **phase_c_stages(), **post_stages()}
    out = run_stage(every[StageId(n)], rd, meeting, cfg, Backends(cfg.backends), force=force)
    typer.echo(f"{out.meeting_id} stage {int(out.stage)}: {out.state} {out.detail}")


@app.command()
def post(run: RunOpt,
         until: Annotated[int, typer.Option(help="last stage to run (6-8)")] = 8,
         only: Annotated[str | None, typer.Option(help="one phase only: CPU (stages 6-7) or D (stage 8)")] = None,
         keep_services: Annotated[bool, typer.Option(help="leave leased services running afterwards")] = False):
    """Stages 6-8 only, on meetings that passed phase C, then the run manifest."""
    from ..artifacts.paths import RunPaths
    from .phase_post import run_post
    from .run_report import write_run_report
    from .scheduler import RunReport, summarise

    rd, sched = _scheduler(run, keep_services)
    rep = RunReport()
    run_post(sched, RunPaths(rd).meeting_ids(), rep, until=StageId(until), only=Phase(only) if only else None)
    write_run_report(rd, rep.phase_seconds)
    typer.echo(json.dumps(summarise(rd, StageId(until)), indent=2))


@app.command()
def manifest(run: RunOpt):
    """Rebuild manifest.jsonl from the meetings' labels/manifest_entry.json (stage 8 passed)."""
    from ..export.manifest import write_run_manifest

    path = write_run_manifest(_run_dir(run))
    n = sum(1 for line in path.read_text().splitlines() if line.strip())
    typer.echo(f"wrote {path}: {n} meetings")


@app.command()
def report(run: RunOpt):
    """Rebuild run_report.json: drops and failures, time per phase, phase C figures, split disjointness."""
    from .run_report import write_run_report

    rep = write_run_report(_run_dir(run))
    typer.echo(json.dumps({k: rep[k] for k in ("problems", "phase_seconds", "manifest")}, indent=2))
    raise typer.Exit(0 if rep["manifest"]["speakers_disjoint_across_splits"] else 1)


@app.command()
def status(run: RunOpt, meetings: Annotated[bool, typer.Option(help="one line per meeting")] = False):
    """Per-stage state counts and the regeneration rate."""
    from .run import load_run
    from .scheduler import summarise
    from .status import StatusStore

    rd = _run_dir(run)
    _, h = load_run(rd)
    s = summarise(rd, StageId.LABELS)
    typer.echo(f"run {rd.name} · config {h}")
    typer.echo(json.dumps(s, indent=2))
    if meetings:
        store = StatusStore(rd)
        for mid in store.run.meeting_ids():
            st = store.get(mid)
            cells = [f"{k}:{v.state}" for k, v in st.stages.items()]
            typer.echo(f"{mid}  " + "  ".join(cells))


@app.command("sampler-report")
def sampler_report(n: Annotated[int, typer.Option()] = 1000, seed: Annotated[int, typer.Option()] = 1,
                   config: Annotated[list[Path], typer.Option("--config", "-c")] = [],  # noqa: B006
                   out: Annotated[Path | None, typer.Option(help="also write the checks as JSON")] = None):
    """M1 exit check: sample N meetings (stage 1 only, in memory) and compare every rate with its target."""

    from ..config.loader import load_run_config
    from ..nsc.pool import speaker_splits
    from ..stages.runner import meeting_id_for
    from ..stages.s1_sample import run_meeting
    from ..validators.sampler_report import as_dicts, render, sampler_checks
    from .run import resolve

    cfg = resolve(load_run_config(*config, overrides={"run_id": "sampler-report", "seed": seed, "n_meetings": n}))
    sp = speaker_splits(cfg.speaker_pool.path, cfg.speaker_pool.version)
    ms = [run_meeting(cfg, sp, meeting_id_for(i)) for i in range(1, n + 1)]
    checks = sampler_checks(cfg, ms)
    typer.echo(render(checks))
    if out:
        out.write_text(json.dumps(as_dicts(checks), indent=2))
    raise typer.Exit(0 if all(c.ok for c in checks) else 1)


@app.command("coverage-seed")
def coverage_seed(n: Annotated[int, typer.Option(help="meetings in the run")] = 20,
                  start: Annotated[int, typer.Option(help="first seed to try")] = 1,
                  tries: Annotated[int, typer.Option()] = 500,
                  config: Annotated[list[Path], typer.Option("--config", "-c", help="YAML overlays, in order")] = []):  # noqa: B006
    """M9: the first run seed whose N meetings cover every split, room class, a 20-speaker test meeting,
    presenters, wall seats (overflow and by choice) and remote participants."""
    from ..config.loader import load_run_config
    from ..validators.coverage import find_seed
    from .run import resolve

    cfg = resolve(load_run_config(*config, overrides={"run_id": "coverage-seed", "seed": start, "n_meetings": n}))
    seed, cov = find_seed(cfg, n, start=start, tries=tries)
    for name, ok in cov.items():
        typer.echo(f"{'ok  ' if ok else 'MISS'} {name}")
    if seed is None:
        typer.echo(f"no seed in {start}..{start + tries - 1} covers everything")
        raise typer.Exit(1)
    typer.echo(f"seed {seed}")


@app.command("export-judge-set")
def export_judge_set(run: RunOpt, n: Annotated[int, typer.Option(help="scripts to export (50-100)")] = 60,
                     name: Annotated[str, typer.Option()] = "judge_set_1",
                     seed: Annotated[int, typer.Option()] = 1):
    """Write a CSV rating sheet (one row per script) plus readable script files for human raters."""
    from ..validators.calibration import export_judge_set as export

    path = export(_run_dir(run), n=n, name=name, seed=seed)
    typer.echo(f"wrote {path}")


@app.command("fit-judge-threshold")
def fit_judge_threshold(sheet: Annotated[Path, typer.Argument(help="the filled rating CSV")],
                        out: Annotated[Path | None, typer.Option(help="YAML overlay to write")] = None):
    """Fit the judge pass mark and borderline band from a filled rating sheet."""
    from ..validators.calibration import fit_threshold

    res = fit_threshold(sheet, out=out)
    typer.echo(json.dumps(res, indent=2))



# ---------------------------------------------------------------- phase C (M4)


@app.command()
def bakeoff(out: Annotated[Path, typer.Option(help="report directory")] = Path("runs/bakeoff"),
            mode: Annotated[str, typer.Option(help="aicore | fake | replay | record")] = "aicore",
            n_speakers: Annotated[int, typer.Option(help="NSC speakers sampled across gender x ethnicity")] = 30,
            targets: Annotated[int, typer.Option(help="target clips cloned per speaker")] = 3,
            modes: Annotated[str, typer.Option(help="TTS modes, comma-separated (ultimate, continuation, reference)")] = "",
            raw_speakers: Annotated[int, typer.Option(help="speakers also cloned from the raw reference")] = 10,
            seed: Annotated[int, typer.Option()] = 1,
            lease: Annotated[bool, typer.Option(help="start/stop the phase C services through stack.py")] = True,
            keep_services: Annotated[bool, typer.Option()] = False):
    """Clone real NSC speakers and measure SV similarity, ASR error, MOS, separability and room colour."""
    from ..validators.bakeoff import run_bakeoff
    from .run import app_path

    rep = run_bakeoff(out_dir=app_path(out), mode=mode, n_speakers=n_speakers, targets=targets,
                      modes=[m.strip() for m in modes.split(",") if m.strip()] or None, raw_speakers=raw_speakers,
                      seed=seed, lease=lease, keep_services=keep_services, log=typer.echo)
    typer.echo(json.dumps(rep["by_mode"], indent=2))
    typer.echo(f"report: {app_path(out) / 'report.md'}")


@app.command("voice-report")
def voice_report(run: RunOpt,
                 min_meetings: Annotated[int, typer.Option(help="meetings stage 5 must have decided (M5 exit: 10)")] = 10,
                 out: Annotated[Path | None, typer.Option(help="also write the figures as JSON")] = None):
    """M5 exit check: pass, first-pass, re-synthesis, ASR-limited and swap rates over phase C."""
    from .phase_c import summarise_c

    res = summarise_c(_run_dir(run))
    text = json.dumps(res, indent=2)
    typer.echo(text)
    if out:
        out.write_text(text + "\n")
    typer.echo(f"{res['ran']} of {min_meetings} meetings through phase C")
    raise typer.Exit(0 if res["ran"] >= min_meetings else 1)


@app.command("viewer-contract")
def viewer_contract(run: RunOpt, as_json: Annotated[bool, typer.Option("--json", help="rows as JSON")] = False):
    """M10 exit check: every field of the UI data contract, on every meeting of the run."""
    from ..viewer.contract import check_run, format_rows

    rows = check_run(_run_dir(run))
    typer.echo(json.dumps([r.as_dict() for r in rows], indent=2) if as_json else format_rows(rows))
    raise typer.Exit(0 if all(r.ok for r in rows) else 1)


@app.command("calibrate-sv")
def calibrate_sv(out: Annotated[Path, typer.Option(help="YAML overlay to write")] = Path("config/sv_thresholds.yaml"),
                 mode: Annotated[str, typer.Option(help="aicore | fake | replay | record")] = "aicore",
                 lease: Annotated[bool, typer.Option(help="start/stop the SV services through stack.py")] = True):
    """Fit per-model SV thresholds on real NSC same-speaker vs different-speaker pairs."""
    from ..validators.sv_calibration import run_calibration
    from .run import app_path

    res = run_calibration(mode=mode, out=app_path(out), lease=lease, log=typer.echo)
    typer.echo(json.dumps({"models": {m: {k: f[k] for k in ("eer", "eer_threshold", "eer_hard", "consistency_p05")}
                                      for m, f in res["models"].items()}, "overlay": res["overlay"]}, indent=2))
    typer.echo(f"wrote {app_path(out)}")


# ---------------------------------------------------------------- NSC (M2)


@nsc_app.command("validate")
def nsc_validate(nsc_dir: Annotated[Path | None, typer.Option(help="NSC Part 3 root (holds SDM/)")] = None):
    """Check the pool CSV: 175 unique speakers, every zip present, split sizes, partner pairs."""
    from ..nsc.pool import load_pool, validate_pool
    from ..nsc.settings import nsc_paths
    from .run import app_path

    paths = nsc_paths(nsc_dir)
    pool = load_pool(app_path("data/nsc_part3_reference_speakers.csv"))
    problems = validate_pool(pool, paths.nsc_dir)
    for p in problems:
        typer.echo(f"problem: {p}")
    typer.echo("pool ok" if not problems else f"{len(problems)} problem(s)")
    raise typer.Exit(0 if not problems else 1)


@nsc_app.command("extract")
def nsc_extract(nsc_dir: Annotated[Path | None, typer.Option()] = None,
                out_dir: Annotated[Path | None, typer.Option(help="nsc-part3-refs root")] = None,
                speakers: Annotated[list[str], typer.Option("--speaker")] = []):  # noqa: B006
    """Read the selected zips in place; write candidate 5-15 s segments (our notation) to nsc-part3-refs/raw/."""
    from ..nsc.extract import extract_all
    from ..nsc.settings import nsc_paths

    paths = nsc_paths(nsc_dir, out_dir)
    res = extract_all(paths, speakers=speakers or None, log=typer.echo)
    typer.echo(json.dumps(res, indent=2))


@nsc_app.command("clean")
def nsc_clean(out_dir: Annotated[Path | None, typer.Option(help="nsc-part3-refs root")] = None,
              mode: Annotated[str, typer.Option(help="aicore | fake | replay | record")] = "aicore",
              keep: Annotated[int, typer.Option(help="clean references kept per speaker (top by MOS)")] = 5,
              max_candidates: Annotated[int, typer.Option(help="candidates cleaned per speaker")] = 12,
              speakers: Annotated[list[str], typer.Option("--speaker")] = [],  # noqa: B006
              lease: Annotated[bool, typer.Option(help="start/stop the prep services through stack.py")] = True):
    """Denoise (DeepFilterNet3) then dereverb (WPE) candidates into nsc-part3-refs/clean/, keep the top by MOS."""
    from ..nsc.cleanup import clean_all
    from ..nsc.settings import nsc_paths

    paths = nsc_paths(None, out_dir)
    res = clean_all(paths, mode=mode, keep=keep, max_candidates=max_candidates, speakers=speakers or None,
                    lease=lease, log=typer.echo)
    typer.echo(json.dumps(res, indent=2))


@nsc_app.command("nonlexical")
def nsc_nonlexical(out_dir: Annotated[Path | None, typer.Option(help="sound library nonlexical/ dir")] = None,
                   dereverb: Annotated[bool, typer.Option(help="WPE through wpe-dereverb (must be running)")] = True,
                   speakers: Annotated[list[str], typer.Option("--speaker")] = []):  # noqa: B006
    """Cut real laughter, coughs, breaths and noises from NSC Part 3 into the stage 4 sound library."""
    from ..backends.registry import Backends
    from ..config.loader import load_run_config
    from ..config.schema import Role
    from ..nsc.nonlexical import extract_nonlexical
    from ..nsc.settings import nsc_paths
    from ..voice.nonlexical import library_root

    cfg = load_run_config(overrides={"run_id": "nsc-nonlexical", "seed": 0, "n_meetings": 0})
    der = Backends(cfg.backends, [Role.DEREVERB]).get(Role.DEREVERB) if dereverb else None
    res = extract_nonlexical(nsc_paths(), out_dir or library_root(), der, speakers=speakers or None, log=typer.echo)
    typer.echo(json.dumps(res, indent=2))


@nsc_app.command("report")
def nsc_report(out_dir: Annotated[Path | None, typer.Option()] = None,
               min_refs: Annotated[int, typer.Option()] = 3, meeting_size: Annotated[int, typer.Option()] = 20,
               mos_min: Annotated[float, typer.Option(help="DNSMOS OVRL floor for a usable reference")] = 2.5):
    """M2 exit check: clean references per speaker, and whether each split can still fill a meeting."""
    from ..nsc.cleanup import readiness
    from ..nsc.settings import nsc_paths

    res = readiness(nsc_paths(None, out_dir), min_refs=min_refs, meeting_size=meeting_size, mos_min=mos_min)
    typer.echo(json.dumps(res, indent=2))
    raise typer.Exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    app()


@app.command()
def viewer(host: Annotated[str, typer.Option()] = "127.0.0.1",
           port: Annotated[int, typer.Option(envvar="SYNTHETICS_PORT")] = 8610,
           reload: Annotated[bool, typer.Option(help="reload on backend changes")] = False,
           read_only: Annotated[bool, typer.Option("--read-only", envvar="SYNTHETICS_READ_ONLY",
                                                   help="leave out the run builder (create, start, stop)")] = False):
    """The Room Config Viewer: the API over SYNTHETICS_RUNS, the run builder, and the built frontend (frontend/dist)."""
    import os

    import uvicorn

    if read_only:
        os.environ["SYNTHETICS_READ_ONLY"] = "1"  # read by the app factory, also in reload workers

    uvicorn.run("synthetics.viewer.api:app", factory=True, host=host, port=port, reload=reload,
                reload_dirs=["backend"] if reload else None)
