# Synthetics

Synthetic Singapore meeting audio with full labels. Start with `AGENTS.md` for the layout, commands and rules.
See `DESIGN_DOCUMENT.md` for the pipeline, `UI_DESIGN_DOCUMENT.md` for the viewer and `docs/OUTPUT_CONTRACT.md`
for the files a run hands over.

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
| C | tts, asr_primary, sv_wavlm, sv_redimnet, quality | `voxcpm2`, `qwen3-asr-1-7b`, `wavlm-sv`, `redimnet2-sv`, `utmos` |
| D | aligner | `qwen3-forced-aligner` |
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

## Room Config Viewer

A web UI over `runs/` (`UI_DESIGN_DOCUMENT.md`): a meeting browser, each meeting's floor plan,
elevation, persona prompts and speakers, and the run's sampled distributions against `config.yaml`. Under
Generate, New dataset creates a run from a form with a live stage 1 preview, and Pipeline starts, stops and
follows it in the background. `synthetics viewer --read-only` (or `SYNTHETICS_READ_ONLY=1`) leaves those out.

```bash
just viewer-setup               # npm install in frontend/ (once)
just viewer-build               # build frontend/dist, which the API serves
just viewer                     # http://127.0.0.1:8610 (SYNTHETICS_PORT); reads SYNTHETICS_RUNS
just viewer-dev                 # API with reload + Vite dev server on http://127.0.0.1:5174
```

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
- **M10** `just viewer-contract <run>`: every field of the UI data contract, on every meeting
  (`tests/test_viewer_contract.py` runs it on a fake run, and on `runs/m9-e2e` when present).
- **M11** `tests/test_ablation.py` switches stages 3, 5 and 7 off and runs to the manifest; the acoustics
  cache test in `tests/test_post.py` covers run revision 2.
- **M12** `tests/test_viewer_api.py`: every route, the whitelist, traversal attempts, no write routes.
- **M13** `just viewer-test` (vitest), then look at the three screens against `assets/mockups/png/*`.

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

## Licences to carry into the dataset card

- **MERaLiON-3:** the MERaLiON Public Licence requires an acknowledgement, and the Gemma terms apply (used in
  the bake-off only).
- **WavLM-Large ECAPA:** CC BY-SA 3.0. It is only used to check our own audio.
- **ESC-50 noise clips:** CC BY-NC, non-commercial only.
- **NSC:** the IMDA licence covers the reference voices and the non-lexical clips cut from them.
