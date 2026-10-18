# Synthetics

Synthetic Singapore meeting audio with full labels. See `DESIGN_DOCUMENT.md` for the pipeline and
`IMPLEMENTATION_PLAN.md` for the build plan. Built so far: milestones M1 to M3 (stages 1 to 3), on the M0
foundations they need (config, artifacts, stage contract, backends, orchestrator).

## Setup

```bash
uv sync
cp .env.example .env        # SYNTHETICS_RUNS, PLATFORM_DIR, NSC_DIR, NSC_REFS_DIR, AI_HOST, NO_EVICT
just test                   # unit tests, no GPU or platform needed
```

Models run as platform services (`../../platform`). Synthetics never starts containers itself: each GPU
phase is a lease through `platform/tools/stack.py up/down`.

| Phase | Role | Service |
|---|---|---|
| A | generator | `qwen-sea-lion-v4-5-27b-it` (NVFP4A16, quantised locally) |
| B | judge_primary | `gemma-sea-lion-v4-27b-it-nvfp4` |
| B2 | judge_tiebreak | `apertus-sea-lion-v4-8b-it` (own sub-phase: it does not fit beside the 27B judge) |
| prep | enhancer, dereverb, ref_quality | `deepfilternet3`, `wpe-dereverb`, `dnsmos` (CPU) |

## Running

```bash
just init m3-50 50 2026         # runs/m3-50: config.yaml, speaker_splits.json, meeting folders
just run m3-50                  # stage 1, then rounds of A (scripts) -> B (judge) -> B2 (tie-break)
just status m3-50               # state counts per stage, rounds to pass, regeneration rate
just stage 2 m3-50 m-000001     # one stage on one meeting (services already up)
```

Every command is idempotent: a stage is skipped when status.json holds a pass with the same cache key
(stage version, config slice, input hashes). Set `--set backends.mode=fake` on `init-run` for a run with fake
backends.

## Milestone checks

- **M1** `just sampler-report 1000`: stage 1 rates against every configured target (exits 1 on a miss).
  `tests/test_sampler.py` runs the same checks on 10,000 meetings, plus geometry properties and
  byte-identical `meeting.json` for the same seed.
- **M2** `just nsc-validate`, `just nsc-extract`, `just nsc-clean`, `just nsc-report`. Data goes to
  `/home/user/code/datasets/nsc-part3-refs/{raw,clean}` (never into the app).
- **M3** `just run <run>` then `just status <run>`. Judge calibration is file-based:
  `just judge-set <run>` writes `runs/<run>/judge_sets/judge_set_1/` (rating sheet, scripts, instructions);
  `just fit-judge <filled sheet>` fits `thresholds.judge_pass` / `judge_band` into a YAML overlay for
  `init-run --config`.

## Departures from the plan, and why

- **Phase B is split into B and B2.** Gemma-SEA-LION NVFP4 alone is 19.5 GB of weights, so it cannot share the
  GPU with Apertus under `util_limit` 0.93 (the plan's ~0.58 estimate was low). Stage 3 runs the primary judge
  in B and defers borderline scripts; B2 loads Apertus and finishes them, reusing the primary judgement.
- **The generator is NVFP4A16, not W4A4 NVFP4.** AI Singapore publishes no 4-bit Qwen-SEA-LION. W4A4 needs
  activation calibration on a loaded 52 GB model, which does not fit in this machine's 31 GB of RAM; the
  data-free path (`experiments/quantize-qwen-sea-lion-nvfp4`) supports weight-only FP4.
- **Reference ranking uses DNSMOS** (`ref_quality` role), a noise and reverberation score that fits the
  cleanup job. UTMOS (`quality`) stays the phase C scorer for TTS output (M4).
- **Stage 2 plans turns before writing them.** A turn plan fixes speakers (Zipf shares, exact counts),
  interaction types (exact overlap and interruption counts), the presentation segment, backchannels and side
  conversations; the generator writes the words chunk by chunk. Persona prompts are saved verbatim.
- **Cleanup cleans the 12 best-ranked candidates per speaker** (by length and markup), not every candidate:
  every candidate is ~26,000 clips (~13 h on CPU). `just nsc-clean --max-candidates 0` cleans all of them.
