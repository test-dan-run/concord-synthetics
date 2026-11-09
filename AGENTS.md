# Synthetics agent instructions

Synthetics generates recordings of made-up Singapore meetings with 2 to 20 speakers, and it labels each recording
exactly with speaker turns (RTTM), word timestamps and normalised transcripts. A separate fine-tuning application
trains speech models on the output, so Synthetics never trains models and never touches real meeting recordings. A
web viewer lets people describe a dataset in plain English, listen to samples, review meetings and export them, and
it lets engineers inspect every run.

Read these files before you change anything:

- `README.md` has the setup commands and the source map, which is the folder tree of the project.
- `specs/REQUIREMENTS.md` lists what the system must do. Check that a change serves one of its user stories.
- `specs/ARCHITECTURE.md` says where each kind of code lives and which pattern it follows.

The code wins when it disagrees with a document, and you fix the document in the same change.

## Commands

Run every command from the repo root inside WSL. `just` loads `.env`, so copy `.env.example` to `.env` first. Node
comes from nvm, so use a login shell (`bash -l`) or load nvm before the frontend recipes.

```bash
uv sync                                       # Python environment
just test                                     # pytest on fake backends, no GPU or platform (about 10 minutes)
just lint                                     # ruff check backend tests
just viewer-setup                             # npm install in frontend/ (once)
just viewer-test                              # vitest
just viewer-build                             # tsc and vite build into frontend/dist
(cd frontend && npm run lint)                 # TypeScript type check
just init <run> <n> <seed> --set backends.mode=fake   # a run with no GPU (about 10 seconds per meeting)
just run <run>                                # all eight stages; `uv run synthetics resume -r <run>` after a stop
just status <run>                             # state counts per stage
just viewer                                   # API and built UI on port 8610 (--read-only drops the builder)
just viewer-dev                               # API with reload and Vite on port 5174
uv run synthetics viewer --admin              # also the speaker pool admin page
uv run synthetics pool build --from <folder> --version <v>   # grow the speaker pool from a curated voice folder
just suggest-eval --fake                      # score the word matching fallback on labelled requests
uv run synthetics suggest-data --n 2000       # training rows for the decision model (leases the generator)
```

`just --list` shows the rest, e.g., one stage on one meeting, NSC preparation, judge and SV calibration, and the
voice model comparison.

## Test suites

The Python tests in `tests/` run against the fake backends, so they need no GPU, platform or dataset. `just test`
skips tests marked `live`, which need running platform services and run with `just test-live`, and tests marked
`slow`. Tests import `synthetics.*` and the helpers in `tests/` directly, because pytest sets
`pythonpath = backend, tests`, and `helpers.py::fake_run` builds a whole fake run. The tests set
`SYNTHETICS_DECISION_SERVICE=0`, so the viewer uses word matching instead of the decision model.

The frontend tests are vitest files next to the code they test (`*.test.ts` and `*.test.tsx`), and they run with
`just viewer-test`. `npm run lint` in `frontend/` is the TypeScript type check.

`uv run synthetics viewer-contract -r <run>` checks every field the viewer reads on every meeting of a run, and
`tests/test_viewer_contract.py` runs the same check on a fake run.

## Architecture boundaries

Each boundary below is enforced in code or checked by a test, so a change that crosses one will fail.

- Stages
  - A meeting goes through eight stages. A stage is a class with `id`, `phase`, `roles`, `reads`, `writes`,
    `version`, `config_slice(cfg)` and `run(ctx) -> StageResult` (`stages/base.py`).
  - A stage reads earlier stages' files and writes only its own, and `stages/runner.py` enforces the rule.
  - A stage is skipped when `status.json` holds a pass with the same cache key. The key covers the stage id and
    version, the stage's config slice, the hashes of its input files, pending feedback and, from revision 2, the run
    revision.
- Models
  - Every model is a platform service in `../../platform`, reached only through the `aicore` client.
  - Each model role maps to a service name in `config/defaults/backends.yaml`, so swapping a model is a change
    there. `backends.mode` is `aicore`, `fake`, `replay` or `record`, and fake and replay never lease the GPU.
  - Phase C's four GPU services are the platform GPU pack `synthetics-phase-c`. Phase C's GPU roles must stay
    exactly that pack, or the platform refuses the lease (`tests/test_orchestrator.py` checks).
- Viewer
  - The read routes never write to a run, and they serve only whitelisted files, audio included.
  - The only writes beside a run's results are `review.json` (keep or leave out) and a reviewer's voice swap, which
    goes through stage 5's swap feedback.
  - Derived numbers, e.g., distances and histograms, are computed in the browser.
- Decision model
  - The decision role turns a user's words into settings by scoring the options in `suggest/questions.yaml`, and
    code applies the options' effects. The model never writes a value.
  - It runs on the CPU (`laya-synthetics` on the platform's `decision.systemone` API) and is never leased. When it
    is down, the viewer uses word matching and says so.

## File layout and naming

`specs/ARCHITECTURE.md` maps every module, and the source map in `README.md` shows the folder tree. When you add a
top-level folder or a module under `backend/synthetics/` or `frontend/src/`, or move one, update the source map and
the module map in the same change.

Names follow these rules:

- Backend files
  - Stage files are `stages/s<n>_<name>.py` with one `<Name>Stage` class each.
  - Every run file has one Pydantic model in `artifacts/`, and `artifacts/paths.py::MeetingPaths` names its path.
  - Defaults live in one YAML file per area in `config/defaults/`, e.g., `turn_taking.yaml`.
  - Python tests are `tests/test_<area>.py`.
- Frontend files
  - Screens live in `frontend/src/features/<area>/` as PascalCase `.tsx` files.
  - Shared components live in `frontend/src/components/`, and hooks and helpers live in `frontend/src/lib/` as
    camelCase `.ts` files.
- Identifiers in run files
  - Meetings are `m-000001`, speakers are `S01`, `S02` and so on (`S01` is the chair), and turns are `t0001`.
  - Room classes are `small` and `large`, and services keep their platform names, e.g., `voxcpm2`.

Code follows these conventions:

- Every module starts with a docstring that says what it does and why. Comments explain intent, not mechanics.
- Python uses `from __future__ import annotations`, relative imports inside the package and full type hints.
  Ruff checks `E,F,W,I,B,UP,SIM` at line length 120.
- Randomness comes from `SeedSequence([run_seed, meeting_seed, stage, attempt])`.
- Config models are frozen with `extra="forbid"`, so a misspelt YAML key fails loudly. Run files are written with
  `artifacts/io.py`, which writes atomically.

Docs follow the plain writing rules, so they use simple words, complete sentences and no dashes. They describe the
system as it is now, with no milestone names, dates or notes about how it used to work, because git history holds
those.

## Always do

- Add a test for new behaviour, and run `just lint` and the tests that cover your change before you say it works.
  Run `just viewer-test` and `npm run lint` for frontend changes.
- Give a new setting a schema default that reproduces the old behaviour, and turn it on for new runs in the YAML
  default.
  - Stages 1 and 7 leave a setting out of their cache slice while it equals that default (`s1_sample.py`,
    `s7_acoustic.py`). Other slices use the same serializer pattern.
  - Gate a change that would alter old runs' draws or cache keys on `RunConfig.revision`, which is 1 for old runs
    and 2 for new runs.
- Update `specs/ARCHITECTURE.md` ("Output contract") when an output file changes, because the fine-tuning
  application depends on it. New fields get defaults, and a rename, removal or change of meaning raises
  `schema_version`.
- Before you write a feature, map it onto the existing modules as `specs/ARCHITECTURE.md` describes, and reuse the
  pattern it names.

## Ask first

- Starting a GPU phase or a real backend run. They take hours and the GPU is shared, so check `nvidia-smi` first.
- Leasing the 27B generator (`qwen-sea-lion-v4-5-27b-it`) for any reason, including `suggest-data`.
- Changing `suggest/questions.yaml` questions or options. The decision model is trained on them, so a change needs
  new rows from `suggest-data` and a retrain, or the model and the questions drift apart. Changing only an
  option's effects needs no retrain.
- Bumping a stage's `version`, because every old run that is resumed then runs that stage again.
- Deleting runs, exports or datasets, and committing or pushing.
- Work outside the user stories in `specs/REQUIREMENTS.md`.

## Never do

- Runs and results
  - Never edit a run's frozen `config.yaml`, and never change old runs' results.
  - Never rewrite a speaker pool. A grown pool is a new file that keeps every earlier speaker's split.
- Models and data
  - Never start containers or load a model in process.
  - Never download weights or datasets into this repo. Weights live in `$AI_ASSETS`, and datasets live in
    `/home/user/code/datasets/`.
  - Never stop platform services that another project started.
- Privacy and splits
  - Never use NSC clips with `#name#` redactions (PDPA).
  - Never let a meeting clone a voice from outside its own split in `speaker_splits.json`.
- Code
  - Never use an unseeded random number generator.
  - Never colour-code speakers. Room classes use `cat-1` and `cat-2`.
  - Never normalise levels in stage 7, because far talkers must stay quiet.

## Environment notes

- `status.json` locking uses `fcntl`, so the pipeline runs on Linux only (WSL here), not on native Windows.
- If `just test` says "Failed to spawn: pytest", the `.venv` was built at an older path. Run `uv sync --reinstall`,
  or use `uv run python -m pytest`.
- If a lease fails with "catalog ... is not answering", the platform catalog is down. Put `AI_STACK_DIRECT=1` in
  front of the command.
- `/tmp` in WSL is held in memory, and pytest keeps the temporary folders of failed tests only
  (`tmp_path_retention_policy = "failed"`).
- Run git inside WSL. Git from the Windows path to the repo reports "dubious ownership".
