# Handoff: after M13 (Oct 21, 2026)

M0 to M13 are done. The pipeline runs all eight stages end to end, the M9 run is complete (20 of 20), and the
Room Config Viewer (read-only API plus web UI) reads any run. What is left is the performance backlog
(`IMPLEMENTATION_PLAN.md` §5, "After M13") and the open items below.

## Starting point

- **Code:** `main` in `apps/Synthetics`. 226 Python tests and 14 frontend tests pass; ruff is clean.
- **Real runs** (in `apps/Synthetics/runs`): `m9-e2e` (20 meetings, all through stage 8), `m5-exit` (10 meetings),
  `m3-10` (scripts only; leave it untouched). All three are run revision 1 (see below).
- **Viewer:** `just viewer-setup` once, `just viewer-build`, then `just viewer` and open http://127.0.0.1:8610.
  `just viewer-dev` runs the API with reload and Vite on port 5174.
- **History:** `docs/M5_EXIT.md`, `docs/VOICE_BAKEOFF.md`, `docs/OUTPUT_CONTRACT.md`, and `git log`.

## What M10 to M13 added (Oct 21)

- **M10.** `synthetics viewer-contract -r <run>` ticks off every row of the UI doc's data contract on every
  meeting (`backend/synthetics/viewer/contract.py`). It passes on `m9-e2e` (52 rows), `m5-exit` and a fake run
  that covers every position type. Per-turn facing stays out of v1 (recorded in `UI_DESIGN_DOCUMENT.md`).
  `docs/OUTPUT_CONTRACT.md` now covers status, prompts, synthesis and voice validation.
- **M11.**
  - RT60 noise in stage 7 fixed: each ray-traced source position gets its own seed, so calibration, the check
    and the render see the same impulse responses. `m9-e2e` m-000020 was re-run through stages 7 and 8
    (0.654 s against 0.65 s) and the run is now 20 of 20.
  - Acoustics cache fixed for new runs: `RunConfig.revision` 2 (the default from `stages.yaml`) draws the
    acoustic values from their own random stream and leaves them out of the stage 2 to 6 cache keys. Frozen
    configs without the field are revision 1 and keep their draws and cache keys.
  - Ablations: switching off stage 3, 5 or 7 now runs to the manifest (it stalled before). See
    `tests/test_ablation.py`.
  - Turn-taking defaults: 50% of transitions overlap by 0.3 to 2.0 s (was 30% and 0.1 to 1.0 s). Stage 6 now
    stretches leads about 2 times on average instead of 3, and converts 3% of transitions instead of 26%.
  - Small faults: fake runs work on the real NSC references; phase time goes to `phase_seconds.jsonl` as it is
    spent, so a killed run keeps it; `SOUND_LIBRARY_DIR` is in `.env.example`.
  - `DESIGN_DOCUMENT.md` has a table of every departure from the first draft, with the text corrected in line.
- **M12.** `synthetics viewer` / `just viewer`: FastAPI over `SYNTHETICS_RUNS` with the plan's routes, plus run
  summary and speaker splits. Read-only, whitelisted, no audio, and it serves `frontend/dist`.
- **M13.** `frontend/`: meeting browser, meeting detail and sampler distributions, built on AeroAudio. The
  categorical palette and the Synthetics icon were added to `/home/user/code/dev/DESIGN_SYSTEM.md` (the file
  before the change is `DESIGN_SYSTEM.md.bak` beside it). Checked by screenshot against the mock-ups at
  1440 px and 390 px, and by keyboard.

## Decisions waiting for you

1. **"(mm) X." backchannels.** They are the largest cause of utterances kept after failing every check: 29 of
   the 96 in `m9-e2e`. The TTS gets `mm good.` and fuses the hum into the next word ("them good"). Sending
   `Mm, good.` should fix it, but that changes synthesis and needs a short phase C test on real audio before it
   ships. It would apply to new runs only.
2. **Names in short sentences** cause 15 of the 96 (the ASR mishears "Wei Ming", "Kok Wai"). The scorer could
   leave persona names out of WER, since stage 5 knows them from `meeting.json`. Some of these also fail SV, so
   it would recover fewer than 15.
3. **Judge pass mark** (unchanged): 3.0 is still provisional; calibration needs human ratings
   (`just judge-set`, then `just fit-judge`).
4. **`DESIGN_SYSTEM.md.bak`** can be deleted once you are happy with the design-system change.

## Open items

- About a tenth of planned interruptions cannot be placed by stage 6 (a floor turn too short to cut, or the
  interrupter still talking), so a meeting drawn near 5% can land a little under it (2 of 20 in `m9-e2e`). The
  realised rate over many meetings stays inside the 5 to 10% range. This was the same before and after the
  turn-taking change.
- `asr_limited` is always zero while MERaLiON is out of phase C; the viewer shows it as "0 (fallback off)".
- The viewer's meeting list sends every meeting's summary in one response (about 12 KB per meeting). That is
  fine for hundreds of meetings; a run of thousands would want paging on the server.
- Concord's charts still use the speaker palette for categories; they can move to `cat-1…6` when next touched.

## Working notes

- **Commands:** `synthetics run | resume | status | report | voice-report | coverage-seed | post | stage |
  viewer | viewer-contract`. A new run starts with `init-run`; `resume` is safe after any stop.
- **A run's config is frozen.** A new default (revision 2, the new turn-taking) does not reach an existing run.
- **Stage versions and old runs.** Bumping a stage's `version` re-runs that stage on every old run that is
  resumed. The Oct 21 behaviour changes are gated on `revision` instead, so old runs stay cache hits.
- **Memory:** watch RAM and `/tmp`; other projects on the machine count too.
- **Docker** comes from Docker Desktop's WSL integration. **Platform:** `/home/user/code/dev/platform` (no
  git). Phase D (`qwen3-forced-aligner`) was the only GPU phase used on Oct 21.

## Licences to carry into the dataset card

- **MERaLiON-3:** the MERaLiON Public Licence requires an acknowledgement, and the Gemma terms apply (used in
  the bake-off only).
- **WavLM-Large ECAPA:** CC BY-SA 3.0. It is only used to check our own audio.
- **ESC-50 noise clips:** CC BY-NC, non-commercial only.
- **NSC:** the IMDA licence covers the reference voices and the non-lexical clips cut from them.
