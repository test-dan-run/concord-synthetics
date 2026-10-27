# Synthetics

Synthetic Singapore meeting audio with full labels. Start with `AGENTS.md` for the layout, commands and rules.
See `DESIGN_DOCUMENT.md` for the pipeline, `UI_DESIGN_DOCUMENT.md` for the viewer and `docs/OUTPUT_CONTRACT.md`
for the files a run hands over.

## Setup

```bash
uv sync
cp .env.example .env        # SYNTHETICS_RUNS, PLATFORM_DIR, NSC_DIR, NSC_REFS_DIR, SOUND_LIBRARY_DIR, AI_HOST, NO_EVICT
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
just init d-50 50 2026          # runs/d-50: config.yaml, speaker_splits.json, meeting folders
just run d-50                   # stage 1, rounds of A, B and B2, then C, stages 6 and 7, then D
just status d-50                # state counts per stage, rounds to pass, regeneration rate
just stage 2 d-50 m-000001      # one stage on one meeting (services already up)
```

Every command is idempotent: a stage is skipped when status.json holds a pass with the same cache key
(stage version, config slice, input hashes), and `synthetics resume -r <run>` continues after any stop. Set
`--set backends.mode=fake` on `init-run` for a run with fake backends and no GPU.

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

## Checks

- **Sampler.** `just sampler-report 1000` compares stage 1 rates with every configured target and exits 1 on a
  miss. `tests/test_sampler.py` runs the same checks on 10,000 meetings, plus geometry properties and
  byte-identical `meeting.json` for the same seed.
- **NSC references.** `just nsc-validate`, `just nsc-extract`, `just nsc-clean` and `just nsc-report` build the
  reference clips in `/home/user/code/datasets/nsc-part3-refs/{raw,clean}` (never in the app). Cleanup cleans the
  12 best-ranked candidates per speaker (by length and markup) and keeps 5; `just nsc-clean --max-candidates 0`
  cleans every candidate, about 26,000 clips and 13 hours on CPU.
- **Judge calibration.** `just judge-set <run>` writes a rating sheet, scripts and instructions to
  `runs/<run>/judge_sets/judge_set_1/`, and `just fit-judge <filled sheet>` fits `thresholds.judge_pass` and
  `judge_band` into a YAML overlay for `init-run --config`.
- **Voice models.** `just bakeoff` clones real NSC speakers and measures similarity, ASR error, MOS and
  separability; `just calibrate-sv` fits the SV thresholds into `config/sv_thresholds.yaml`.
- **Viewer data.** `just viewer-contract <run>` checks every field the viewer reads, on every meeting
  (`tests/test_viewer_contract.py` runs it on a fake run, and on `runs/m9-e2e` when present).
  `tests/test_viewer_api.py` covers every route, the whitelist and traversal attempts.
- **Ablations and caching.** `tests/test_ablation.py` switches stages 3, 5 and 7 off and runs to the manifest;
  `tests/test_post.py` checks that changing sampled acoustics re-runs only stages 1, 7 and 8.
- **Viewer UI.** `just viewer-test` (vitest), then compare the screens with `assets/mockups/png/*`.

## Licences to carry into the dataset card

- **MERaLiON-3:** the MERaLiON Public Licence requires an acknowledgement, and the Gemma terms apply (used in
  the bake-off only).
- **WavLM-Large ECAPA:** CC BY-SA 3.0. It is only used to check our own audio.
- **ESC-50 noise clips:** CC BY-NC, non-commercial only.
- **DEMAND room-tone beds:** CC BY-SA 3.0.
- **NSC:** the IMDA licence covers the reference voices and the non-lexical clips cut from them.
