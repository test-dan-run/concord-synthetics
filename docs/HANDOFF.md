# Handoff: where things stand (Oct 19, 2026, night)

This is for picking up tomorrow. It covers the state after M4, the M5 code, and the merges. Read it with
`docs/VOICE_BAKEOFF.md` (the measurements behind the voice choices) and `IMPLEMENTATION_PLAN.md` §5 (M5 to M11).

## State tonight

- **Code:** `main` has M0 to M3 (main tree), M6 to M8 (`m6-m8`, merged as `a9e6e40`) and M4/M5 (`m4-m5`, merged
  twice; this file came with the second merge). The scheduler already calls `run_phase_c` and `run_post` on
  `main`.
- **Worktrees:** `apps/Synthetics-voice` (`m4-m5`) and `apps/Synthetics-post` (`m6-m8`) still exist, and both
  branches are fully merged. New work can go straight on `main` in `apps/Synthetics`, or on a new branch. The
  parallel-work rules in `docs/PARALLEL_WORK.md` were for M4 to M8 only.
- **Services:** all Docker services are down (`just down`). Nothing is on the GPU.
- **`/tmp`:** cleared. It is a tmpfs, so it lives in RAM. Claude session scratch had grown to 13 GB of it and
  starved the machine; that is what killed the first bake-off.
- **Platform:** `/home/user/code/dev/platform` has no git, so its changes exist only on disk. They are listed below.

## Milestones

| Milestone | State |
|---|---|
| M4 bake-off | **Done.** Choices confirmed and thresholds in config (`docs/VOICE_BAKEOFF.md`) |
| M5 stages 4 and 5 | Code done and tested on fakes. **Exit not met yet:** 50 meetings through phase C on real scripts, with pass, re-synthesis and ASR-limited rates. `orchestrator/phase_c.py:summarise_c` computes them, but no CLI command calls it yet |
| M6, M7, M8 | Done by the post agent (see its notes under Open items) |
| M9 | Next: a 20-meeting unattended end-to-end run, with the coverage list in the plan |

## Decision waiting: phase C VRAM

Under the bake-off load, phase C peaked at 31.27 GB of 32.6 GB (0.959), above `util_limit` 0.93. Compiled TTS has
since added about 0.5 GB. On paper, `just vram` says 0.907, so the stack will start it, but the margin is about
1 GB.

**Proposal:** a C2 sub-phase for `meralion-3-3b-asr`, like B/B2.

- **Phase C:** TTS, Qwen3-ASR, both SV models and UTMOS, about 23 GB.
- **C2:** loads MERaLiON only for utterances that failed Qwen3-ASR, and is skipped when there are none. The
  bake-off had none.
- **Where the change goes:** `orchestrator/phase_c.py`, `stages/s5_voice_val.py` (an utterance waits on the
  Singlish check instead of deciding at once) and `config/defaults/phases.yaml`.

Decide this before the M5 exit run.

## Next steps, in order

1. Decide on C2, and build it if wanted.
2. **M5 exit:** 50 meetings through phase C on real scripts from the M3 run. Report the pass, re-synthesis,
   ASR-limited and swap rates. The bake-off predicts about 100% first-pass on SV, ASR and MOS, and a speaker swap
   in about one meeting in seven.
3. **M9:** the 20-meeting end-to-end run, with the coverage list from the plan. Phases go A → B → B2 → C → D, one
   at a time on the GPU.

## Platform changes on disk (no git)

Seven new services, all registered with weights fetched and images built:

| Service | Port | Notes |
|---|---|---|
| `voxcpm2` | 8410 | TTS; torch.compile on (needs gcc and python3.12-dev in the image); about 80 s to become healthy |
| `qwen3-asr-1-7b` | 8106 | vLLM, util 0.22; weights pulled by `just up` (hf-cache is owned by root) |
| `meralion-3-3b-asr` | 8107 | custom transformers server |
| `wavlm-sv` | 8404 | WavLM-Large ECAPA; input cut at 20 s |
| `redimnet2-sv` | 8405 | input cut at 20 s |
| `utmos` | 8406 | CPU |
| `qwen3-forced-aligner` | 8407 | rejects requests over 180 s |

Measured VRAM is in each manifest's `gpu` comment. Also on disk: the names in `tests/test_schema.py`, the rows
in the root `README.md`, and `just render`.

## How to bring things up

- Phase C: `just up voxcpm2 qwen3-asr-1-7b meralion-3-3b-asr wavlm-sv redimnet2-sv utmos --no-evict`, from the
  platform dir. In normal runs, the app's `GpuLease` does this.
- Prep services, needed only for NSC cleanup or `nsc nonlexical`: `deepfilternet3`, `wpe-dereverb`, `dnsmos`,
  `text-normalizer`.
- `.env` has `NO_EVICT=1`.
- Only build images when `free -g` shows at least 12 GB available, or no vLLM container is running.

## Gotchas learned

- **Memory:** watch `/tmp` and RAM. Claude Code kills background commands under memory pressure. The bake-off now
  resumes: rerun the same command and it skips finished speakers (`runs/bakeoff/progress.jsonl`).
- **Fakes vs real thresholds:** the stage tests pin `sv_pair_max` to 0.85/0.75 and `mos_drop_max` to 5.0. The
  fake embedder's random voices often reach cosine 0.5, and the fake MOS is a hash of the samples. The real
  defaults are in `stages.yaml` and `schema.py`.
- **MOS floor:** relative to the speaker's own reference (`mos_drop_max` 0.75, absolute `mos_min` 1.0). A fixed
  3.0 rejected whole groups of voices. The `mos` in the clean reference index is DNSMOS and is not used here.
- **Known gap:** a one-word TTS slip (about WER 0.15) passes stage 5. `wer_max` 0.35 cannot separate it from
  transcript notation differences.
- **Noisy log line:** Qwen3-ASR logs `Please install vllm[audio]` on each request. It is harmless.
- **Seeds:** compiled VoxCPM2 output differs from uncompiled output for the same seed, though each is
  deterministic. Any audio made before the switch on Oct 19 used the uncompiled model.
- **Speaker swaps:** after a reference swap, `synthesis.voices[sid].nsc_speaker` is the authoritative NSC
  speaker, not `meeting.json`.

## Open items

- **Acoustics cache:** changing only the acoustics re-runs every stage. The fix needs `s1_sample.py` (a separate
  RNG stream) and `runner.py` (hash `meeting.json` without the acoustic fields). A strict xfail in
  `tests/test_post.py` flips when it is fixed. This is the M7 exit criterion.
- **Overlap defaults:** default turn-taking gives about 5–8% overlap against a 10–25% target; stage 6 stretches to
  close the gap. Revisit `turn_taking.yaml`.
- **`SOUND_LIBRARY_DIR`:** not in `.env.example` yet. Both libraries default to
  `/home/user/code/datasets/sound-library` (`nonlexical/` from NSC, `noise/` from DEMAND + ESC-50).
- **Fake mode with real references:** a fake-mode run passes all eight stages only with synthetic NSC references
  (`NSC_REFS_DIR` from `tests/voice_helpers.fake_refs`). `FakeSpeakerEmbedder` reads voice tones that real NSC
  clips do not have, so stage 5 fails on real references in fake mode. Real-model runs are unaffected.
- **Judge pass mark:** the M3 judge threshold of 3.0 is still provisional. Gemma judges strictly and Apertus
  leniently.

## Licences to carry into the dataset card

- **MERaLiON-3:** the MERaLiON Public Licence requires an acknowledgement, and the Gemma terms apply.
- **WavLM-Large ECAPA:** CC BY-SA 3.0. It is only used to check our own audio.
- **ESC-50 noise clips:** CC BY-NC, non-commercial only.
- **NSC:** the IMDA licence covers the reference voices and the non-lexical clips cut from them.
