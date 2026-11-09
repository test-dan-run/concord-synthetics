# Synthetics

Synthetics generates synthetic recordings of Singapore meetings with 2 to 20 speakers, labelled with who spoke when
and when each word was said, so that a separate fine-tuning application can train transcription and diarisation
models on them.

The voices are cloned from the National Speech Corpus (NSC), a language model writes the conversations, and nothing
in a recording comes from a real meeting. You can describe the meetings you want in plain English in the viewer, which
is a web page, or you can make them with commands in a terminal.

## Quick start

Synthetics runs on Linux, which on Windows means inside WSL. You need Python 3.12 with
[uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just), git, and Node.js through nvm for the
viewer. The steps below make two practice meetings with fake models, so they need no GPU.

First, clone the repo next to the AI platform, because the settings expect the platform at `../../platform`.

```bash
mkdir -p ~/code/dev/apps
cd ~/code/dev/apps
git clone https://github.com/test-dan-run/concord-synthetics.git
cd concord-synthetics
```

Second, install the Python and web packages.

```bash
uv sync
just viewer-setup
```

Third, copy the example settings. `.env` says where runs are written and where the platform, the NSC data and the
sound library are, and the defaults match the paths on this machine.

```bash
cp .env.example .env
```

Fourth, make a practice run of 2 meetings with fake models. It takes under a minute, and the audio is beeps instead
of speech, but every file and label is where a real run puts it.

```bash
just init try-1 2 1 --set backends.mode=fake
just run try-1
just status try-1
```

Fifth, build and start the viewer, and open http://127.0.0.1:8610 in a browser.

```bash
just viewer-build
just viewer
```

For frontend work, `just viewer-dev` runs the API with reload and the Vite dev server on http://127.0.0.1:5174.

A real run needs the platform services and a GPU with about 32 GB, and the GPU is shared, so check `nvidia-smi`
first. Run `just init <run> <meetings> <seed>` without `--set backends.mode=fake`, then `just run <run>`, and
continue a stopped run with `uv run synthetics resume -r <run>`. If a run stops with "catalog ... is not
answering", put `AI_STACK_DIRECT=1` in front of the command.

## Project structure

Update this tree whenever you add a top-level folder, add or move a module under `backend/synthetics/` or
`frontend/src/`, or reorganise a folder. `specs/ARCHITECTURE.md` describes each module in more detail.

```
concord-synthetics/
  AGENTS.md                 instructions for AI coding agents: commands, boundaries and rules
  README.md                 this file
  Justfile                  the commands you run with `just`
  pyproject.toml, uv.lock   Python package, pytest and ruff settings
  .env.example              local settings to copy to .env
  specs/
    REQUIREMENTS.md         user stories and acceptance criteria
    ARCHITECTURE.md         modules, data flow, output files, API and styling
  backend/synthetics/       the Python package
    config/                 RunConfig schema, distributions, loader, defaults/*.yaml
    artifacts/              one Pydantic model per run file, paths, atomic writes
    stages/                 stages 1 to 8, the stage protocol, the runner, the turn plan and prompts
    orchestrator/           the `synthetics` CLI, scheduler, GPU lease, phase loops, reports, background runs
    backends/               model role protocols, aicore clients, fakes, record and replay
    validators/             script rules, LLM judges, voice checks, sampler report, calibration
    acoustics/              room, scene, noise and device chain for stage 7
    export/                 RTTM, transcript, manifest and export packs
    text/                   Singlish convention, numbers, PDPA stripping
    nsc/                    NSC speaker pools, pool growth, reference clips
    voice/                  audio operations, sound library, stage 5 measurement cache
    suggest/                decision questions, engine and effects (words to settings)
    viewer/                 FastAPI routes: read, builder, suggest, export, admin, data contract
  frontend/                 the viewer's web page (React, Vite, Tailwind)
    src/app/                routes and the app shell
    src/features/           one folder per screen group: guided, browser, detail, distributions, builder,
                            pipeline, admin
    src/components/         shared UI components
    src/lib/                API hooks, draft state and helpers that compute derived numbers
  tests/                    pytest suite on fake backends; helpers.py::fake_run builds a fake run
  data/                     speaker pool CSVs (pools/ holds grown pools), pool selection and sound library scripts
  config/                   calibrated SV thresholds (sv_thresholds.yaml)
  assets/                   the Synthetics icon and UI mockups
  runs/                     run folders (ignored by git)
```

## Verification commands

Run these before you commit. None of them needs a GPU or the platform.

```bash
just lint                        # ruff on backend/ and tests/
just test                        # pytest on fake backends (about 10 minutes)
just viewer-test                 # vitest for the frontend
(cd frontend && npm run lint)    # TypeScript type check
just viewer-build                # type check and production build of the frontend
```

The checks below need platform services, a finished run or more time:

- `just test-live` runs the tests that need running platform services.
- `uv run synthetics viewer-contract -r <run>` checks every field the viewer reads on a finished run.
- `just sampler-report 1000` compares stage 1's sampled rates with their targets.
- `just suggest-eval` scores the decision model on hand-written requests, and `just suggest-eval --fake` scores the
  word matching fallback.
