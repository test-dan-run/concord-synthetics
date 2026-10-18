# Synthetics. Run `just` for the recipe list. See README.md and IMPLEMENTATION_PLAN.md.

set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

cli := "uv run --quiet synthetics"

# list recipes
default:
    @just --list --unsorted

# install dependencies (uv)
setup:
    uv sync

# unit tests (no GPU, platform or NSC data needed)
test *args:
    uv run pytest {{args}}

# tests that need running platform services
test-live *args:
    uv run pytest -m live -o addopts="--import-mode=importlib" {{args}}

# ruff
lint:
    uv run ruff check backend tests

# create runs/<id>: frozen config, speaker splits, meeting folders (e.g. just init m3-50 50 2026)
init run_id n="20" seed="1" *flags:
    {{cli}} init-run {{run_id}} --n {{n}} --seed {{seed}} {{flags}}

# run the pipeline phase by phase (stages 1-3 so far); resumable
run run_id *flags:
    {{cli}} run -r {{run_id}} {{flags}}

# one stage on one meeting, for debugging (services must be up)
stage n run_id meeting *flags:
    {{cli}} stage {{n}} -r {{run_id}} -m {{meeting}} {{flags}}

# per-stage state counts and the regeneration rate
status run_id *flags:
    {{cli}} status -r {{run_id}} {{flags}}

# copy the fixed NSC splits into a run's speaker_splits.json
splits run_id:
    {{cli}} splits -r {{run_id}}

# M1 exit check: sample N meetings and compare every rate with its target
sampler-report n="1000":
    {{cli}} sampler-report --n {{n}}

# M2: validate the pool, extract candidates, clean references (prep services), readiness report
nsc-validate:
    {{cli}} nsc validate
nsc-extract:
    {{cli}} nsc extract
nsc-clean *flags:
    {{cli}} nsc clean {{flags}}
nsc-report *flags:
    {{cli}} nsc report {{flags}}

# M5: real laughter / coughs / breaths / noises from NSC into the stage 4 sound library (uses wpe-dereverb)
nsc-nonlexical *flags:
    {{cli}} nsc nonlexical {{flags}}

# M3 judge calibration: rating sheet out, fitted threshold back in
judge-set run_id n="60":
    {{cli}} export-judge-set -r {{run_id}} --n {{n}}
fit-judge sheet out="config/judge_thresholds.yaml":
    {{cli}} fit-judge-threshold {{sheet}} --out {{out}}

# M4: phase C bake-off on real NSC speakers (leases the phase C services), and SV threshold calibration
bakeoff *flags:
    {{cli}} bakeoff {{flags}}
calibrate-sv *flags:
    {{cli}} calibrate-sv {{flags}}
