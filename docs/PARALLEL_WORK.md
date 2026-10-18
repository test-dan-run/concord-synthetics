# Parallel work on M4 to M8

Oct 18, 2026. This file holds the rules for the three agents working at the same time. Read it fully
before starting, and read `IMPLEMENTATION_PLAN.md` §3 to §5 for the milestones themselves.

| Agent | Milestones | Where it works | Branch | Compute |
|---|---|---|---|---|
| M3 agent | M1 to M3 (finishing) | `apps/Synthetics` (main tree) | `main` (uncommitted) | GPU now (phases A, B, B2) |
| Voice agent | M4, M5, and the aligner service for M8 | `apps/Synthetics-voice` | `m4-m5` | GPU, but only after M3 releases it |
| Post agent | M6, M7, and M8 except the aligner service | `apps/Synthetics-post` | `m6-m8` | CPU only |

Both worktrees branch from the commit "Add the coordination rules for parallel M4-M8 work". They sit next to the main tree, so
the relative paths `../../platform` in `pyproject.toml` and `phases.yaml` still resolve.

## The machine

- **One RTX 5090 (32 GB).** Only one phase's models fit at a time. Phase A alone uses util 0.80.
- **31 GB of RAM, often only about 7 GB free** while a vLLM model is loaded. Docker image builds with torch
  and long acoustic renders can push WSL out of memory.
- 24 CPU cores. 600 GB free on the disk that holds `$AI_ASSETS` and the datasets.

## GPU rules

1. **Only the voice agent may lease the GPU.** The post agent never runs `stack.py`, `just up`, `just conform`
   or a live test.
2. **The voice agent waits for M3.** Before any GPU step, run `docker ps` and check that none of
   `qwen-sea-lion-v4-5-27b-it`, `gemma-sea-lion-v4-27b-it-nvfp4` or `apertus-sea-lion-v4-8b-it` is running,
   and ask the user to confirm the M3 run is finished. If one is running, do non-GPU work instead.
3. **Never evict.** The voice agent's `.env` sets `NO_EVICT=1`, which the Synthetics lease honours. On the
   platform side always pass the flag: `just up <name> --no-evict`.
4. **Image builds need memory.** Build a custom image only when `free -g` shows at least 12 GB available, or
   when no vLLM container is running.
5. Leave the running CPU prep services (`deepfilternet3`, `wpe-dereverb`, `dnsmos`, `text-normalizer`) alone.

## The shared contract (already committed)

These files define how the stages hand data to each other. They are on `main`, so both branches start from
them.

- `artifacts/voice.py`: `Synthesis` (`synthesis.json`, stage 4) and `VoiceValidation` (stage 5).
- `artifacts/labels.py`: `Timeline`, `Transcript` and `ManifestEntry` (stages 6 to 8).
- `artifacts/paths.py`: `MeetingPaths.synthesis` was added.
- `backends/fakes_audio.py`: `FakeVoiceCloner`, `FakeTranscriber`, `FakeSpeakerEmbedder`, `FakeAligner`, and
  `fake_reference(clip_id)`. `mode: fake` builds them for every phase C and D role.
- `config/schema.py`: `VoiceConfig`, `AssemblyConfig`, `SimulationConfig`, `LabelsConfig`, and new thresholds
  (`sv_consistency_min`, `sv_pair_max`, `align_coverage_min`, `align_slack_s`). The SV values are guesses
  until `calibrate-sv` runs.

**The stage 4 to stage 6 handoff.** Stage 4 writes exactly one `utterances/<turn_id>.wav` per script turn:
mono, at the run sample rate, with at most `voice.trim_pad_s` of silence at each end. `synthesis.json` lists
the utterances in script order with their durations. Stage 6 runs only on meetings that passed stage 5, so
every listed utterance can be used. Clip ids are NSC pair ids (`<speaker>_<nnnn>`).

**Changing the contract.** Each agent may add fields to the models it owns (see below), with defaults, so
the other side keeps working. Do not rename or remove a field, and do not change the meaning of one. If a
change is needed, stop and tell the user, who will relay it.

## File ownership

Each agent creates new files freely in the areas it owns. To keep the merge easy, shared files are touched
as little as possible.

| Area | Voice agent | Post agent |
|---|---|---|
| Artifact models | `artifacts/voice.py` | `artifacts/labels.py` |
| Stages | `stages/s4_voice.py`, `stages/s5_voice_val.py` | `stages/s6_assemble.py`, `stages/s7_acoustic.py`, `stages/s8_labels.py` |
| Other packages | `validators/` voice checks, bake-off, SV calibration; `nsc/` reference loading for stage 4 | `acoustics/` (new modules; `geometry.py` from M1 is read only), `export/` |
| Backends | `Aicore*` clients for tts, asr, sv, quality and aligner in `backends/aicore_impl.py` | none (use the fakes) |
| Config | `VoiceConfig`, the SV thresholds | `AssemblyConfig`, `SimulationConfig`, `LabelsConfig`, the alignment thresholds |
| Scheduler wiring | new module `orchestrator/phase_c.py` | new module `orchestrator/phase_post.py` (stages 6 and 7, phase D) |
| Platform | new manifests in `platform/services/` and their tests | nothing |
| Datasets | `datasets/sound-library/nonlexical/` with its own `index.yaml` | `datasets/sound-library/noise/` with its own `index.yaml` |
| Docs | the bake-off report | `docs/OUTPUT_CONTRACT.md` |

Rules for shared files:

- **Do not edit `orchestrator/scheduler.py` or `stages/runner.py`.** The M3 agent is still changing them. Put
  each phase's loop in your own `orchestrator/phase_*.py`, written so the scheduler can call it in one line.
  That line is added at the merge.
- **`orchestrator/cli.py` and the `Justfile`:** put command logic in your own modules and add only the thin
  command and recipe. Expect small conflicts here; they are resolved at the merge.
- **Dependencies:** add them with `uv add`. Conflicts in `uv.lock` are resolved at the merge by running
  `uv lock` again.
- **Do not edit** `stages/s1_sample.py`, `s2_script.py`, `s3_script_val.py`, `script_plan.py`, `prompts.py`,
  `validators/judges.py`, `validators/script_rules.py` or `text/`. They belong to the M3 agent. If you need
  a change there, tell the user.
- **The platform has no git.** The voice agent adds new files there and only appends to shared platform
  files (endpoint map, pinned tests, README), as the `register-model` skill describes.

## Working in a worktree

```bash
cd /home/user/code/dev/apps/Synthetics-voice     # or Synthetics-post
uv sync
just test
```

`.env` is copied in, and `runs/` is not shared. Use `tests/helpers.py::fake_run` for test runs. A live run
on real scripts (the M5 and M8 exit checks) waits until the M3 work is merged.

Commit on your branch whenever a step is finished and its tests pass. End every commit message with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not merge, rebase onto `main` or push. The
user orchestrates the merge.

## What "done" means for each agent

**Voice agent**
1. Without the GPU: manifests and pinned `weights.yaml` for `voxcpm2`, `qwen3-asr-1-7b`, `meralion-3-3b-asr`,
   `wavlm-sv`, `redimnet2-sv`, `utmos`, `qwen3-forced-aligner`; weights fetched; the CPU service `utmos`
   built and conformant; `Aicore*` clients with record/replay; the `bakeoff` and `calibrate-sv` commands;
   stages 4 and 5 and `orchestrator/phase_c.py` working against the fakes, with unit tests.
2. With the GPU, in this order: check that VoxCPM2 runs on sm_120 (the plan's main risk; if it fails, report
   it before going further); `just vram` for the phase C set (the plan's ~0.74 estimate may be low, as
   phase B's was, so be ready to propose a C2 sub-phase); `just conform` for each service; the bake-off;
   `calibrate-sv`, with the thresholds written to config; register and conform the aligner.
3. Report: the bake-off results, the chosen TTS, SV and ASR models, the measured VRAM, the SV thresholds, and
   anything that changed in the contract.

**Post agent**
1. Stage 6 with the tests from the plan (realised overlap, backchannel and interruption rates within their
   ranges over many meetings; no speaker overlaps their own speech; truncation matches `truncated_at`).
2. Stage 7: room, directivity, per-segment rendering for facing and presenter drift, the mic array and its
   derived single channel, noise beds and events at the sampled SNR, and the device chain. Exit check: RT60
   measured on the simulated impulse responses is within the target, and changing only acoustics re-runs only
   stages 7 and 8. Keep test meetings short (`fake_run` uses 4 to 8 minutes). Keep one render under about
   4 GB of RAM. No process pools: execution stays sequential.
3. Stage 8 against `FakeAligner`: RTTM, alignment per utterance window, transcript, manifest entry, the
   quality gates, and `docs/OUTPUT_CONTRACT.md`.
4. Noise beds: start with procedural beds (HVAC, projector fan) so stage 7 never waits on data. Before
   downloading any real recordings, list the candidate sources and their licences and ask the user.
5. Report: what was built, the measured RT60 error, render time and peak memory for one meeting, and anything
   that changed in the contract.

## The merge (after all three are done)

1. Commit the M3 agent's work on `main`.
2. Merge `m6-m8` (it touches the fewest shared files), then `m4-m5`.
3. Add the one-line calls from `scheduler.py` into `phase_c.py` and `phase_post.py`, resolve the `cli.py`,
   `Justfile` and `uv.lock` conflicts, and run the full test suite.
4. Continue with M9: the 20-meeting end-to-end run.
