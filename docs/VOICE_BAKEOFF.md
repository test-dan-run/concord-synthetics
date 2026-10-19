# Phase C bake-off (M4)

Oct 19, 2026. These measurements fixed the stage 4 and 5 models and thresholds. They were taken on the RTX 5090
after the M3 run released it. The raw output is in `runs/bakeoff/` (`report.md`, `report.json`, `clips.jsonl`, and
`listen/` for listening), and the SV fit is in `config/sv_thresholds.json`. To rerun:
`just bakeoff --modes ultimate,continuation,reference` and `just calibrate-sv`.

## What was chosen

| Role | Service | Model |
|---|---|---|
| TTS | `voxcpm2` | VoxCPM2 2B, mode `ultimate` (reference clip as both prompt and reference), cfg 2.0, 10 steps, torch.compile on |
| Primary ASR | `qwen3-asr-1-7b` | Qwen3-ASR 1.7B on vLLM |
| Singlish ASR | `meralion-3-3b-asr` | MERaLiON-3 3B (custom transformers server) |
| SV, model 1 | `wavlm-sv` | WavLM-Large + ECAPA-TDNN (the Seed-TTS-eval similarity model), 256-d |
| SV, model 2 | `redimnet2-sv` | ReDimNet2 b6 (VoxBlink2 + VoxCeleb2 + CN-Celeb2) |
| MOS | `utmos` | UTMOS22 strong (CPU) |
| Aligner (phase D) | `qwen3-forced-aligner` | Qwen3-ForcedAligner 0.6B |

`wavlm-sv` was first `microsoft/wavlm-base-plus-sv`. On NSC clips it scored different-speaker pairs up to 0.95,
above its lowest same-speaker pair (0.90), so it was replaced before the bake-off.

## Setup

- **Speakers and clips:** 30 NSC speakers, spread across gender × ethnicity. Each speaker's best clean clip is the
  reference. Three of their other clean clips are targets, so every clone has a real recording of the same words by
  the same person to compare with.
- **Volume:** 270 clones (90 per TTS mode).
- **Room colour:** 10 speakers were also cloned from the raw, uncleaned reference.

## Results by TTS mode

Numbers are means, with p10 in brackets. Thresholds are the calibrated ones (see SV thresholds below).

| Measure | `ultimate` | `continuation` | `reference` | Real recordings |
|---|---|---|---|---|
| WavLM similarity to the reference | **0.806** (0.737) | 0.794 (0.726) | 0.704 (0.562) | 0.772 (0.686) |
| ReDimNet similarity to the reference | **0.844** (0.802) | 0.837 (0.796) | 0.778 (0.691) | 0.850 (0.781) |
| Speaker ID among the 30, both models | 100% | 100% | 100% | |
| Qwen3-ASR error (WER, or CER if short) | 0.042 | 0.050 | 0.039 | 0.131 |
| MERaLiON error | 0.032 | 0.038 | 0.039 | 0.093 |
| Passes ASR through Qwen3 alone | 100% | 100% | 100% | 96.7% |
| UTMOS | 2.94 | 2.95 | 3.07 | 2.54 |
| Would pass stage 5 | **100%** | 100% | 92.2% | |

`ultimate` is the default. It has the best similarity at equal ASR error and MOS. `reference` mode (no prompt
continuation) loses about 0.1 similarity, and 8% of its clones fall below the WavLM threshold.

Clones are cleaner than the recordings they imitate: they have lower ASR error and higher MOS. They also score
closer to their reference than the speaker's own other recordings do, because they copy the reference's channel.

## SV thresholds

These come from `calibrate-sv` on 875 clean NSC clips of 175 speakers: 1,750 same-speaker pairs and 15,225
different-speaker pairs.

| Model | EER | EER, same gender and ethnicity | `sv` and `sv_pair_max` (EER threshold) | `sv_consistency_min` (p05) |
|---|---|---|---|---|
| WavLM | 0.53% | 0.98% | 0.5568 | 0.6872 |
| ReDimNet | 0.18% | 0.34% | 0.6515 | 0.7991 |

They are the defaults in `config/defaults/stages.yaml` and `config/schema.py`.

- **Clone vs reference:** every `ultimate` clone passes. The lowest scores are 0.630 (WavLM) and 0.731 (ReDimNet).
- **Consistency:** the lowest per-speaker clone consistency is 0.722 (WavLM) and 0.835 (ReDimNet), above both
  minimums.
- **Speaker pairs:** 4 of 435 clone pairs exceed the WavLM `sv_pair_max`, and 2 exceed the ReDimNet one. Real
  recordings of the same speakers give 7 and 2, so these pairs really do sound alike to the models. At about 15
  pairs per meeting, expect a voice swap in roughly one meeting in seven. M5 measures the actual rate.

## MOS: a floor relative to the reference

UTMOS rates the real NSC references from 1.33 to 3.29, and unevenly by group. A fixed 3.0 floor (the plan's) would
have rejected whole voices:

| Group | Mean UTMOS of the reference | `ultimate` clones at or above 3.0 |
|---|---|---|
| Female, Chinese | 2.32 | 7% |
| Female, Malay | 2.07 | 25% |
| Male, Chinese | 2.30 | 25% |
| Male, Others | 2.70 | 50% |
| Male, Malay | 2.70 | 67% |
| Female, Indian | 2.87 | 75% |
| Male, Indian | 2.88 | 75% |
| Female, Others | 2.78 | 100% |

A clone's MOS tracks its reference's (r = 0.86) and is on average 0.39 above it. The worst clone was 0.53 below.
So stage 5 now requires:

- MOS ≥ `mos_min` (1.0), an absolute floor for broken audio;
- MOS ≥ the reference clip's UTMOS − `mos_drop_max` (0.75).

Every bake-off clone passes. A garbled clone still falls well below its own reference. `SpeakerCheck.ref_mos`
records the reference's score. The `mos` stored in the clean reference index is DNSMOS from M2, on a different
scale, and is not used for this.

## ASR

`wer_max` stays at 0.35 and `cer_max_short` at 0.40.

- **Clone error:** the better ASR's error is 0 at the median, 0.156 at p99 and 0.261 at worst.
- **Worst cases are notation, not speech:** for example "S E S" in the NSC transcript vs "SCS". Re-synthesis cannot
  fix these, so a tighter limit would only exhaust attempts.
- **Limit:** a one-word slip is about WER 0.15 and passes. Example: VoxCPM2 said "next the second or not" for "next
  Thursday, can or not". Both ASRs heard it, but at that WER no threshold separates slips from notation.
- **MERaLiON was never needed:** the ASR-limited rate was 0% across 270 clones.
- **Particle notation:** MERaLiON writes particles in NSC notation, as `(lah)`. `asr_words` normalises these to the
  same words as Qwen3's `lah`.

## Room colour

Clones from the cleaned reference scored 0.28 MOS higher than clones of the same targets from the raw reference
(3.04 vs 2.76). The DeepFilterNet3 + WPE cleanup from M2 carries through to the clones. Pairs to listen to are in
`runs/bakeoff/listen/`.

## Speed

VoxCPM2 runs at RTF 1.17 without torch.compile and 0.22 with it. The compiled output had the same quality on a
check of 8 clones: MOS 3.25 → 3.31, WavLM 0.739 → 0.748, ReDimNet 0.809 → 0.808. Output is still bit-identical
for a given seed, but it differs from uncompiled output for the same seed. Compile costs about 80 s at start-up.

At RTF 0.22, a 150-turn meeting needs about 3 minutes of TTS, against about 15 before.

## VRAM (measured on the 5090)

Device totals, including each process's CUDA context. Idle is about 1.4 GB.

| Service | Loaded | Peak | Manifest |
|---|---|---|---|
| `voxcpm2` (compiled) | 6.6 GB | 7.4 GB | 8 GB |
| `qwen3-asr-1-7b` | 6.8 GB | 6.8 GB | util 0.22 |
| `meralion-3-3b-asr` | 7.2 GB | 8.1 GB on short clips | 9 GB |
| `wavlm-sv` | 1.9 GB | 2.5 GB at 20 s | 3 GB |
| `redimnet2-sv` | 0.75 GB | 2.6 GB at 20 s | 2 GB |
| `qwen3-forced-aligner` | 2.3 GB | 3.7 GB at 173 s | 4 GB |

**Phase C under load:** with all six phase C services running the bake-off, VRAM sat at a median of 30.6 GB and
peaked at 31.27 GB (0.959 of the card). That is above the 0.93 `util_limit`, and torch.compile has since added
about 0.5 GB. On paper the manifests sum to 0.907, but each caching allocator keeps its peak, and CUDA contexts
are not counted. Nothing ran out of memory in 47 speakers, but the margin is about 1 GB. See the open decision
below.

## Fixes found on the GPU

- **SV input length:** both SV services ran out of their memory cap on a 60 s clip. WavLM attention grows with
  the square of the length, and ReDimNet's activations grow with it. Both now cut input at 20 s. Clone turns and
  references are shorter than that, and the embedding of 60 s equals that of its first 20 s.
- **Aligner length:** a 292 s clip aligned only up to 245 s, with mean score 0.15. qwen-asr's own limit is 180 s
  (`MAX_FORCE_ALIGN_INPUT_SECONDS`), so the server now rejects longer requests. Scores fall with length: 0.69 at
  7 s, 0.61 at 119 s, 0.49 at 173 s. Align one utterance at a time.
- **vLLM audio log line:** Qwen3-ASR logs `Please install vllm[audio]` on every request. It is harmless: decoding
  falls back and returns correct text.

## Decision: phase C VRAM

Phase C had about 1 GB of headroom with MERaLiON loaded. A C2 sub-phase for MERaLiON, like B/B2, was considered
and dropped (Oct 19). Instead, phase C no longer loads the Singlish ASR at all:

- Phase C runs TTS, Qwen3-ASR, both SV models and UTMOS, about 23 GB.
- An utterance that fails Qwen3-ASR goes back to stage 4 for re-synthesis with a new seed, and Qwen3-ASR checks
  it again, like any other stage 5 failure.
- The bake-off's ASR-limited rate was 0% across 270 clones, so MERaLiON would have changed nothing.

The fallback still exists in stage 5. Adding `asr_singlish` back to phase C's roles in
`config/defaults/phases.yaml` turns it on.
