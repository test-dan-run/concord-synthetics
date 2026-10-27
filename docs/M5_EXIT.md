# M5 exit: phase C on the 10 M3 meetings (Oct 19–20, 2026)

The M5 exit is met. All 10 meetings of the M3 run passed stage 5 with the settings now in the defaults. The run is
`runs/m5-exit` in `apps/Synthetics`: a copy of `runs/m3-10` with its phase C settings brought up to date, so the
M3 record is untouched. The figures are in `runs/m5-exit/voice_report.json`, from `synthetics voice-report`.

The exit was cut from 50 meetings to these 10 on Oct 19, because 50 would have cost about 5 hours of scripts and a
day of phase C without moving the utterance rates. M9 reports the meeting rates again over its 20 meetings.

## Figures

| | Value |
|---|---|
| Meetings passed | 10 of 10 |
| Utterances | 3,783 |
| Passed without any re-synthesis | 96.1% |
| Re-synthesised for a failed check | 3.9% |
| Still failing after the last attempt, kept flagged | 27 (0.71%) |
| ASR-limited | 0 (MERaLiON is no longer in phase C) |
| Reference swaps | 2 clip swaps in 2 meetings, no speaker swaps |
| Mean SV to reference | WavLM 0.764, ReDimNet 0.821 |
| Mean UTMOS | 3.24 |

The bake-off predicted about 100% first pass and a swap in about one meeting in seven. First pass came out lower
because real scripts have many short turns (backchannels, cut-offs) that the bake-off's 5–15 s clips did not.

Phase C takes about 2.5 hours for the 10 meetings from scratch (about 2.4 s per turn, re-synthesis included). It peaked
well under the VRAM limit: 21.3 GB loaded without MERaLiON, against 31.27 GB peak with it in the bake-off.

## The 27 utterances kept flagged

| Reason | Count | What they are |
|---|---|---|
| SV only | 13 | Mostly near misses (ReDimNet 0.55–0.65 against 0.65), several on cut-off turns ending in "-" |
| CER on short turns | 10 | "ok (lah)", "ya", "Right.", "Can we-": particles and cut-offs are the fragile part |
| WER | 3 | Names ("Mei Ling") and a particle sentence ("safety first, (hor)") |
| MOS | 1 | One backchannel |

None of these points to a new systematic fault. They are listed in each meeting's `voice_validation.json` under
`kept_failed`.

## What the run found, and what changed

The first real meetings failed for reasons the bake-off could not show. Each fix is in stage 5, with tests.

1. **MERaLiON dropped from phase C** (`383f39d`). It left about 1 GB of VRAM headroom, and the bake-off never
   needed it. An utterance that fails Qwen3-ASR is re-synthesised. Adding `asr_singlish` back to phase C turns
   the fallback on again.
2. **Backchannels skip SV** (`3aca1af`). "mm" hums scored 0.07–0.54 on WavLM and pulled speakers' consistency
   down, which forced needless swaps. Backchannels and turns with no words now skip SV and stay out of the
   consistency and pair checks.
3. **ASR in English** (`3aca1af`). With language detection on, Qwen3-ASR heard one-word turns as Chinese or
   Japanese ("Right." → 对). `voice.asr_language` is now `en`. On 442 utterances it fixed one failure and broke none.
4. **ASR spellings** (`3aca1af`). The ASR writes "Yeah" and "Okay" where scripts say "ya" and "ok". The scorer maps
   them on both sides. The lexicon is unchanged, so scripts and audio stay cached.
5. **Pair check on the right scale** (`126d5fd`). `sv_pair_max` is a threshold on single clip pairs, but stage 5
   compared it with centroid cosines, which read about 1.25 times higher. Every meeting got about two needless
   swaps. The check now uses the mean cosine over all utterance pairs between two speakers.
6. **Exhausted utterances kept, not fatal** (`126d5fd`). `voice.on_exhausted` is now `keep_flagged`. One bad
   utterance no longer fails a 30-minute meeting.
7. **Consistency on long enough utterances** (`710b59a`). `sv_consistency_min` comes from 5–15 s clips, but it was
   applied to utterances down to 1 s. Speakers with mostly 2 s turns failed it with any voice, and three meetings
   failed with no swap left. Consistency now uses only utterances of 3 s or more, and is skipped for a speaker
   with fewer than 3 of them. Per-utterance SV still applies to every turn of 1 s or more.

Before fix 7, the same 10 meetings gave 7 passes, 16 clip swaps and 5 speaker swaps
(`runs/m5-exit/voice_report_before_consistency_fix.json`).

## Worth a look later

- **Short-turn speakers:** a speaker with fewer than three utterances of 3 s or more has no consistency check now.
  Per-utterance SV is the only guard for them. It still works (it flagged the weak takes of S18 in `m-000005`).
- **Particles in TTS input:** "(hor)" goes to the TTS as "haw", and the ASR hears "huh" or "Paul". Particles count
  in WER, so short sentences ending in a particle are fragile.
- **Cut-off turns:** the TTS often drops the last word before the dash ("Can we-" → "Can."). Their CER and SV
  failures are a large share of what is kept flagged.
- **The bake-off's pair figures** used 3-clip centroids, so "one swap in seven meetings" was on a slightly inflated
  scale. This run's figure (2 clip swaps in 10 meetings, from the consistency check) supersedes it.

## Stages 6 to 8 on the same meetings

Stages 6 to 8 had only run on fakes. Before M9, they ran on these 10 meetings (`synthetics post -r m5-exit`). The
first pass failed 1 meeting in stage 7 and 5 in stage 8; both faults are fixed in `08c7975`:

- **Stage 7, RT60:** absorption was calibrated on 4 talker positions, but the check takes the median over every
  talker. In a 14-talker room the probes said 0.52 s and all talkers 0.60 s, against a 0.50 s target.
  `simulation.calibration_probes` is now 20, so every talker is calibrated. It costs seconds.
- **Stage 8, alignment:** on clips of 0.12–0.40 s (11 "(mm)" backchannels and one "Ya.") the forced aligner put
  the word up to 0.31 s past the end of the clip. Aligned times are now clamped to the clip for clips under
  `thresholds.align_clamp_below_s` (1.0 s). Longer clips are not clamped, so a misplaced alignment still fails.

After the fixes, all 10 meetings pass stages 6 to 8: realised overlap on target, RT60 within 9% of target,
alignment coverage 100%. `manifest.jsonl` has 10 lines, and no NSC speaker appears in two splits
(`runs/m5-exit/run_report.json`). Stages 6 and 7 took about 6 minutes for the 10 meetings, and phase D about
10 minutes.
