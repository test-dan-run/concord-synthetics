# Handoff: developing M10 onwards (Oct 21, 2026)

M0 to M9 are done and merged to `main`, so the next work is M10: checking that the pipeline writes everything
the viewer needs. After that come hardening (M11), the read-only viewer API (M12) and the viewer UI (M13).
Read this with `IMPLEMENTATION_PLAN.md` §5 (M10 to M13), the "Data contract" section of `UI_DESIGN_DOCUMENT.md`,
and `docs/OUTPUT_CONTRACT.md`.

## Starting point

- **Code:** `main` in `apps/Synthetics`. All tests pass (206, plus one strict xfail for the acoustics cache).
  The `apps/Synthetics-voice` and `apps/Synthetics-post` worktrees are fully merged and can be removed.
- **Real runs to build against** (in `apps/Synthetics/runs`):
  - `m9-e2e`: 20 meetings, seed 9; 19 passed stage 8 and are in `manifest.jsonl`. It covers all splits, both room
    classes, a 20-speaker test meeting, presenters, wall seats (overflow and by choice) and remote participants.
    It is the main fixture for M10 and the viewer.
  - `m5-exit`: 10 meetings, all through stage 8. Several debug logs sit beside it from the M5 fixes.
  - `m3-10`: the M3 record, scripts only. Leave it untouched.
- **GPU:** M10 to M13 need no GPU unless you make a new run. Fake-mode runs (`backends.mode=fake`) cover tests.
- **History:** `docs/M5_EXIT.md` (phase C results and every stage 5, 7 and 8 fix), `docs/VOICE_BAKEOFF.md` (voice
  model choices), and `git log`.

## M10: run statistics for the viewer

Most of the fields already exist: an audit of `m9-e2e` against the UI data contract table found every field
present. What M10 still lacks is a test that pins this down.

| UI doc's "new field" | Where it is now |
|---|---|
| Rendered persona prompt per speaker | `prompts/<speaker_id>.txt`, written by stage 2 |
| Turn-taking statistics per meeting | `manifest.jsonl` → `turn_taking` (overlap share, backchannels per minute, interruption share, plus raw counts) |
| Fit resample count | `meeting.json` → `fit_resamples` (the UI doc allows either file) |
| Speaker facing per turn | Not written. Optional: v1 of the UI shows one direction per seat |

Other contract rows and where they come from:

- **Manifest tags** (mic type, table shape, split): `sampled["mic.type"]`, `sampled["table.shape"]` and `split`.
- **Wall-seated and presenter counts:** `position_counts`. Per-speaker position type: `speakers[].position_type`.
- **Hybrid, RT60, SNR:** `sampled["dynamics.hybrid"]`, `sampled["acoustics.rt60"]` and `sampled["acoustics.snr_db"]`.
  Measured RT60 is in `acoustic`.
- **Per-stage status, timings, retries:** `status.json` → `stages[n]` (`state`, `started_at`, `ended_at`,
  `attempts`, `detail`, `metrics`).
- **Turn start times:** `script.json` has none; the UI doc says to use `timeline.json`.
- **SV scores and ASR-limited flags:** `voice_validation.json`. `asr_limited` is always false now that MERaLiON
  is out of phase C, so the speaker card can hide it or show "n/a".

**To do:**

1. Write a contract test over a fake run: for each row of the table, the field exists with the right type.
   Optionally, add a live check over `m9-e2e`.
2. Decide on per-turn facing (probably not for v1), and record the decision in the UI doc.
3. Bring `docs/OUTPUT_CONTRACT.md` up to date. It is dated Oct 18 and predates `turn_taking` counts, `acoustic`,
   `labels`, `kept_failed`, `cross_cos` and `consistency_utts`.
4. Exit: the contract table is ticked off against a real run, and the test passes.

## M11: hardening

From the plan:

- Ablation runs: stage toggles are already recorded in `manifest.stages_enabled`. They need a run and a check
  that a disabled stage is skipped cleanly.
- Update `DESIGN_DOCUMENT.md` where the plan departs from it: platform contracts (§1, §4.2), weights in
  `$AI_ASSETS` (§1), the fixed NSC pool and splits (§4.3), voice assignment in stage 1 (§4.3), sequential
  execution (§3.5).
  Since Oct 19, there are also MERaLiON leaving phase C, `keep_flagged`, and the new stage 5 checks.

Faults carried in, roughly in order of value:

1. **RT60 noise in stage 7.** In `m9-e2e`, `m-000020` failed: calibration and the check measured the same 6
   talker poses at 0.667 s and 0.78 s, against a 0.65 s target. The ray-traced late part is drawn afresh each call,
   and per-talker estimates spread from 0.58 to 0.99 s in one room. Options: run the check on the calibration's own
   impulse responses, average several mics, or use more rays (`simulation.rt_n_rays`). Re-running stage 7 on
   `m-000020` takes seconds.
2. **Acoustics cache (M7 exit criterion).** Changing only the acoustics re-runs every stage. The fix needs
   `s1_sample.py` (a separate RNG stream) and `runner.py` (hash `meeting.json` without the acoustic fields). A
   strict xfail in `tests/test_post.py` flips when it is fixed.
3. **Kept-flagged utterances** (1.3% in M9; kept in the data by decision). The two big causes:
   - "(mm) X." backchannels go to the TTS as `mm good.`; the hum fuses into the next word. A stage 4 input change
     (for example `Mm, good.`) would fix it, but it re-synthesises every meeting.
   - Short sentences with a Singaporean name fail WER because the ASR gets the name wrong. The lexicon's `#name#`
     marker could take names out of the score, but scripts don't use it yet.
   - Smaller ones: particles ("(hor)" sent as "haw"), cut-off turns losing their last word.
4. **Judge pass mark.** Gemma scores nearly every script 3.0 (borderline), so Apertus decides every pass. The 3.0
   mark is provisional; `export-judge-set` and `fit-judge-threshold` exist for a human-rated calibration.
5. **Overlap defaults.** Default turn-taking gives about 5–8% overlap against a 10–25% target; stage 6 stretches
   to close the gap. Revisit `turn_taking.yaml`.
6. **Small ones:** `SOUND_LIBRARY_DIR` is missing from `.env.example`. A fake-mode run fails stage 5 on real NSC
   references (it needs `tests/voice_helpers.fake_refs`). `run_report.json` loses `phase_seconds` from a run that
   is killed rather than finished.

## M12: viewer backend

A thin read-only FastAPI layer over `runs/<run_id>/`. The plan lists the routes. Notes from the runs so far:

- `GET /api/runs/{id}/meetings` should join each manifest line with `status.json`, so the grid also shows meetings
  that failed or are still running (`m9-e2e` has one, `m-000020`).
- The artifact whitelist needs `voice_validation.json`, `render.json` and `labels/transcript.json`, as well as the
  files in the contract table. Audio is not in the contract; leave it out unless the UI asks for playback.
- `run_report.json` and `voice_report.json` sit at the run root and suit a run summary route.
- Runs can live outside the app (`SYNTHETICS_RUNS`); the real ones are in `apps/Synthetics/runs`.

## M13: viewer UI

The plan's order follows the UI doc. The inputs:

- Design system: `/home/user/code/dev/DESIGN_SYSTEM.md`. The categorical palette and the Synthetics icon (already
  drawn in `assets/icon.svg`) go into it first, in the same commit as any design-system change.
- Scaffold: copy from `apps/Concord` per DESIGN_SYSTEM §11.
- Mock-ups: `assets/mockups/png/*` and `assets/mockups/html/Room Config Viewer.html`.
- Build the meeting detail screen first, against `m9-e2e`.

## Working notes

- **Commands:** `synthetics run | resume | status | report | voice-report | coverage-seed | post | stage`. A new
  run starts with `init-run`. `resume` is safe after any stop, because finished stages are cache hits.
- **A run's config is frozen.** A new default doesn't reach an existing run; edit its `config.yaml` by hand, or
  start a new run.
- **New real runs:** choose a seed with `coverage-seed` when a small run must cover rare cases. Phase A writes a
  script in about 5.5 min, and phase C runs at about 2.5–3.2 s per turn. A 20-meeting run takes about 9 hours
  end to end.
- **Memory:** watch RAM and `/tmp` (a tmpfs). Claude Code kills background jobs under memory pressure, and other
  projects on the machine (an Android emulator and Gradle, about 13 GB) count too.
- **Docker:** it comes from Docker Desktop's WSL integration. If `docker` is missing, start Docker Desktop.
- **Fakes vs real thresholds:** stage tests pin `sv_pair_max` to 0.85/0.75 and `mos_drop_max` to 5.0. Fake voices
  and fake MOS don't behave like the real models.
- **Speaker swaps:** after a reference swap, `synthesis.voices[sid].nsc_speaker` is the authoritative NSC speaker,
  not `meeting.json`. The viewer's "NSC reference speaker per persona" should read it from there, or from the
  manifest's `speakers[].nsc_speaker`.

## Platform (no git; changes exist only on disk)

`/home/user/code/dev/platform`. Phase C loads about 21 GB: `voxcpm2` (8410), `qwen3-asr-1-7b` (8106), `wavlm-sv`
(8404), `redimnet2-sv` (8405), `utmos` (8406). Phase D loads `qwen3-forced-aligner` (8407; it rejects requests over
180 s). `meralion-3-3b-asr` (8107) is registered but unused. The app's `GpuLease` starts and stops services; by
hand, use `just up <services> --no-evict`, or `uv run tools/stack.py down <services>`. `.env` has `NO_EVICT=1`.
Only build images with at least 12 GB of RAM free.

## Licences to carry into the dataset card

- **MERaLiON-3:** the MERaLiON Public Licence requires an acknowledgement, and the Gemma terms apply (no longer
  used in phase C, but it was used in the bake-off).
- **WavLM-Large ECAPA:** CC BY-SA 3.0. It is only used to check our own audio.
- **ESC-50 noise clips:** CC BY-NC, non-commercial only.
- **NSC:** the IMDA licence covers the reference voices and the non-lexical clips cut from them.
