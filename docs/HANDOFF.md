# Handoff: where things stand (Oct 20, 2026, early morning)

This covers the overnight work of Oct 19–20: the M5 exit run, the stage 5, 7 and 8 fixes it led to, and the start
of the M9 run. Read it with `docs/M5_EXIT.md` (the M5 results and every fix), `docs/VOICE_BAKEOFF.md` (the voice
model measurements) and `IMPLEMENTATION_PLAN.md` §5 (M5 to M11).

## State

- **Code:** branch `phase-c-qwen-only`, worked on in the `apps/Synthetics-voice` worktree. It starts from `main`
  (`1c070b6`) and is **not merged yet**. All tests pass (206, plus the strict xfail for the acoustics cache).
- **Runs** (all in `apps/Synthetics/runs`):
  - `m3-10`: the M3 record, untouched.
  - `m5-exit`: a copy of `m3-10` with current settings; stages 4 to 8 done on all 10 meetings.
  - `m9-e2e`: the M9 run, 20 meetings, seed 9. Started at 03:23 on Oct 20 from this worktree's code.
- **Platform:** `/home/user/code/dev/platform` has no git, so its changes exist only on disk (listed below).

## Milestones

| Milestone | State |
|---|---|
| M4 bake-off | **Done** (`docs/VOICE_BAKEOFF.md`) |
| M5 stages 4 and 5 | **Done.** Exit met on Oct 20: all 10 M3 meetings pass phase C (`docs/M5_EXIT.md`) |
| M6, M7, M8 | Done, and now checked on real audio: all 10 `m5-exit` meetings pass stages 6 to 8 |
| M9 | **Running** (`m9-e2e`). Code added: `coverage-seed`, `run_report.json`, `synthetics report` |

## Decisions taken on Oct 19–20

- **No C2 sub-phase.** MERaLiON is out of phase C; a Qwen3-ASR failure is re-synthesised.
- **M5 exit cut to the 10 M3 meetings** (from 50). M9 reports the meeting rates again.
- **`voice.on_exhausted: keep_flagged`.** An utterance still failing after its last attempt is listed in
  `kept_failed`, and the meeting passes.
- **Stage 5:** backchannels skip SV; ASR language `en`; "yeah"/"okay" scored as "ya"/"ok"; pair check on the
  mean cross-speaker cosine; consistency only on utterances of 3 s or more.
- **Stage 7:** RT60 calibrated on every talker (`calibration_probes` 20).
- **Stage 8:** aligned times clamped to the clip for clips under 1 s.
- **M9 coverage by seed:** `synthetics coverage-seed` picked seed 9, whose 20 meetings cover every item on the
  plan's coverage list. The sampler's distributions are unchanged.

## Next steps

1. **Watch M9.** `synthetics status -r m9-e2e --meetings` for progress. If it stops, `synthetics resume -r m9-e2e`
   picks it up; finished stages are cache hits. When it finishes, `synthetics report -r m9-e2e` and
   `synthetics voice-report -r m9-e2e --min-meetings 1`.
2. **Merge `phase-c-qwen-only` into `main`** once the M9 run is judged.
3. **M10:** run statistics for the viewer (plan §M10).

## Platform changes on disk (no git)

Seven new services, all registered with weights fetched and images built:

| Service | Port | Notes |
|---|---|---|
| `voxcpm2` | 8410 | TTS; torch.compile on (needs gcc and python3.12-dev in the image); about 80 s to become healthy |
| `qwen3-asr-1-7b` | 8106 | vLLM, util 0.22; weights pulled by `just up` (hf-cache is owned by root) |
| `meralion-3-3b-asr` | 8107 | custom transformers server; not used by phase C since Oct 19 |
| `wavlm-sv` | 8404 | WavLM-Large ECAPA; input cut at 20 s |
| `redimnet2-sv` | 8405 | input cut at 20 s |
| `utmos` | 8406 | CPU |
| `qwen3-forced-aligner` | 8407 | rejects requests over 180 s |

Measured VRAM is in each manifest's `gpu` comment. Also on disk: the names in `tests/test_schema.py`, the rows
in the root `README.md`, and `just render`.

## How to bring things up

- Phase C: `just up voxcpm2 qwen3-asr-1-7b wavlm-sv redimnet2-sv utmos --no-evict`, from the platform dir. It
  loads about 21 GB. In normal runs, the app's `GpuLease` does this.
- Prep services, needed only for NSC cleanup or `nsc nonlexical`: `deepfilternet3`, `wpe-dereverb`, `dnsmos`,
  `text-normalizer`.
- `.env` has `NO_EVICT=1`.
- Only build images when `free -g` shows at least 12 GB available, or no vLLM container is running.
- Docker comes from Docker Desktop's WSL integration: if `docker` is missing in WSL, start Docker Desktop.

## Gotchas learned

- **Memory:** watch `/tmp` and RAM. Claude Code kills background commands under memory pressure. Other projects
  on the machine (an Android emulator and Gradle used about 13 GB on Oct 19) count too.
- **A run's config is frozen.** Changing a default does not reach an existing run. `m5-exit` and `m9-e2e` were
  updated by hand to match the defaults of `08c7975`.
- **Fakes vs real thresholds:** the stage tests pin `sv_pair_max` to 0.85/0.75 and `mos_drop_max` to 5.0. The
  fake embedder's random voices often reach cosine 0.5, and the fake MOS is a hash of the samples.
- **Short turns behave differently from the bake-off's clips.** Backchannels, cut-offs and one-word turns caused
  most of the M5 and post fixes. They also make up most of what stage 5 keeps flagged.
- **MOS floor:** relative to the speaker's own reference (`mos_drop_max` 0.75, absolute `mos_min` 1.0).
- **Known gap:** a one-word TTS slip (about WER 0.15) passes stage 5. `wer_max` 0.35 cannot separate it from
  transcript notation differences.
- **Noisy log line:** Qwen3-ASR logs `Please install vllm[audio]` on each request. It is harmless.
- **Seeds:** compiled VoxCPM2 output differs from uncompiled output for the same seed.
- **Speaker swaps:** after a reference swap, `synthesis.voices[sid].nsc_speaker` is the authoritative NSC
  speaker, not `meeting.json`.

## Open items

- **Acoustics cache:** changing only the acoustics re-runs every stage. The fix needs `s1_sample.py` (a separate
  RNG stream) and `runner.py` (hash `meeting.json` without the acoustic fields). A strict xfail in
  `tests/test_post.py` flips when it is fixed. This is the M7 exit criterion.
- **Overlap defaults:** default turn-taking gives about 5–8% overlap against a 10–25% target; stage 6 stretches to
  close the gap. Revisit `turn_taking.yaml`.
- **`SOUND_LIBRARY_DIR`:** not in `.env.example` yet. Both libraries default to
  `/home/user/code/datasets/sound-library`.
- **Fake mode with real references:** a fake-mode run passes stage 5 only with synthetic NSC references.
- **Judge pass mark:** the M3 judge threshold of 3.0 is still provisional. Gemma judges strictly and Apertus
  leniently.
- **From M5** (`docs/M5_EXIT.md`): particles in TTS input ("(hor)" → "haw"), cut-off turns losing their last
  word, and short-turn speakers having no consistency check.

## Licences to carry into the dataset card

- **MERaLiON-3:** the MERaLiON Public Licence requires an acknowledgement, and the Gemma terms apply.
- **WavLM-Large ECAPA:** CC BY-SA 3.0. It is only used to check our own audio.
- **ESC-50 noise clips:** CC BY-NC, non-commercial only.
- **NSC:** the IMDA licence covers the reference voices and the non-lexical clips cut from them.
