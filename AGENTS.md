# Synthetics

Synthetics generates synthetic Singapore meeting recordings (2 to 20 speakers) with exact labels: RTTM speaker
turns, word timestamps and normalised transcripts. A separate fine-tuning application consumes the output;
Synthetics never trains models or touches real recordings. A web viewer browses runs and can create and start
new ones.

## Stack and layout

Python 3.12 (uv, Typer CLI, Pydantic v2, FastAPI, numpy, pyroomacoustics) and a React 19 + Vite + Tailwind 4
frontend (TypeScript, vitest). Models are platform services reached through the `aicore` client from
`../../platform`.

```
backend/synthetics/
  config/        schema.py (RunConfig), distributions, loader; defaults/*.yaml hold every default
  artifacts/     one Pydantic model per run file, MeetingPaths, atomic writers (io.py)
  stages/        s1_sample.py to s8_labels.py, base.py (Stage protocol), runner.py (cache keys, status)
  orchestrator/  cli.py (the `synthetics` command), scheduler, GPU lease, phase_c / phase_post loops
  backends/      role protocols, aicore clients, fakes, record/replay transport
  acoustics/ export/ validators/ text/ nsc/ voice/
  viewer/        api.py (read routes), builder.py (create/start/stop runs), contract.py
frontend/src/    app/ (routes), features/{browser,detail,distributions,builder,pipeline}, lib/, components/
data/            nsc_part3_reference_speakers.csv: the pinned speaker pool and its fixed splits
config/          sv_thresholds.yaml: calibrated SV thresholds, an overlay for init-run
tests/           pytest suite; helpers.py::fake_run builds a fake-backend run
runs/            run directories (git-ignored)
```

## Commands

Run from the repo root inside WSL. `just` loads `.env` (copy `.env.example` first). Node comes from nvm, so use
an interactive shell or load nvm before the frontend recipes.

```bash
uv sync                                       # Python env
just test                                     # pytest, no GPU or platform needed (about 5 min)
just lint                                     # ruff check backend tests
just viewer-setup                             # npm install in frontend/ (once)
just viewer-test                              # vitest
just viewer-build                             # tsc + vite build into frontend/dist
(cd frontend && npm run lint)                 # TypeScript type check
just init <run> <n> <seed> --set backends.mode=fake   # a run with no GPU (one meeting runs in about 2 min)
just run <run>                                # all 8 stages; `uv run synthetics resume -r <run>` after a stop
just status <run>                             # state counts per stage
just viewer                                   # API + built UI on http://127.0.0.1:8610 (--read-only drops the builder)
just viewer-dev                               # API with reload + Vite on :5174
```

`just test` skips tests marked `live` (need platform services, `just test-live`) and `slow`. `just --list`
shows the rest (one stage on one meeting, NSC prep, judge and SV calibration, bake-off).

## How the code is organised

- Each meeting goes through eight stages. A stage is a class with `id`, `phase`, `roles`, `reads`, `writes`,
  `version`, `config_slice(cfg)` and `run(ctx) -> StageResult` (`stages/base.py`). It reads earlier stages'
  files and writes only its own. `stages/runner.py` enforces this.
- A stage is skipped when `status.json` holds a pass with the same cache key: stage id and version, the stage's
  config slice, input file hashes, pending feedback and (revision 2 and up) the run revision.
- Randomness comes from `SeedSequence([run_seed, meeting_seed, stage, attempt])`. Never use an unseeded RNG.
- Only one GPU phase's models fit at a time, so the scheduler runs each stage across all meetings, leasing that
  phase's services with the platform's `tools/stack.py up/down` (`orchestrator/lease.py`). Execution is
  sequential: one meeting, one request at a time.
- Each model role maps to a service name in `config/defaults/backends.yaml`. Swapping a model is a change there.
  `backends.mode` is `aicore`, `fake`, `replay` or `record`; fake and replay never lease the GPU.
- Every run file is a Pydantic model in `artifacts/`, written with `artifacts/io.py` (atomic write). Config
  models are frozen with `extra="forbid"`, so a misspelt YAML key fails loudly.
- Stage 7 sets levels at the source and never normalises, so far talkers stay quiet.
- The viewer's read routes never write to a run and serve only whitelisted files, never audio. Derived numbers
  (distances, histograms) are computed in the browser. Speakers are never colour-coded; room classes use
  `cat-1` and `cat-2`.

## Conventions

- Every module starts with a docstring saying what it does and why; comments explain intent, not mechanics.
- `from __future__ import annotations`, relative imports inside the package, full type hints. Ruff with
  `E,F,W,I,B,UP,SIM` at line length 120.
- Tests import `synthetics.*` and `helpers` directly (pytest `pythonpath = backend, tests`) and run against the
  fake backends. Add a test for new behaviour.
- Docs are plain English: simple words, complete sentences, no dashes. They describe the system as it is now:
  no milestone names, dates or "was X until" notes, since git history holds those.

## Gotchas and things not to do

- Never edit a run's frozen `config.yaml`, and never change old runs' results. A change that would alter draws
  or cache keys must be gated on `RunConfig.revision` (1 for old runs, 2 from `stages.yaml`). A new setting gets
  a schema default that reproduces the old behaviour, and the YAML default turns it on for new runs; stage 1
  leaves a setting out of its cache slice while it equals that default (`s1_sample.py`).
- Bumping a stage's `version` re-runs that stage on every old run that is resumed. Do it only when outputs
  should change.
- Leave `runs/m3-10` untouched. `runs/m9-e2e` and `runs/m5-exit` are the reference real runs.
- Never start containers or load a model in process; never download weights or datasets into this repo. Weights
  live in `$AI_ASSETS`, datasets in `/home/user/code/datasets/`.
- Ask before starting a GPU phase or a real-backend run: they take hours and the GPU is shared.
- A meeting only clones voices from its own split (`speaker_splits.json`), so splits stay speaker-disjoint. NSC
  clips with `#name#` redactions are never used (PDPA).
- `status.json` locking uses `fcntl`, so the pipeline runs on Linux only (WSL here), not native Windows.
- If `just test` says "Failed to spawn: pytest", the `.venv` was built at an older path. Run
  `uv sync --reinstall` (or `uv run python -m pytest`).
- When an output file changes, update `docs/OUTPUT_CONTRACT.md`: the fine-tuning application depends on it. New
  fields get defaults; a rename, removal or change of meaning raises `schema_version`.
  `synthetics viewer-contract -r <run>` checks the fields the viewer reads.

## Further docs

`DESIGN_DOCUMENT.md` (pipeline design, sampling tables, models and thresholds), `UI_DESIGN_DOCUMENT.md` (viewer),
`docs/OUTPUT_CONTRACT.md` (output files). The code wins when they disagree.
