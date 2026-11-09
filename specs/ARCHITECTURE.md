# Synthetics architecture

The document describes how Synthetics is built, i.e., its parts, how data moves between them, the files it writes,
the viewer's API and screens, and the styling rules. `specs/REQUIREMENTS.md` says what the system must do, and the
code is the reference when the two disagree.

## Fitting a feature into the existing modules

Every new feature must fit the modules and patterns below, so before you write code, write a short feature map in
your plan or pull request description. The map names each part of the feature, the existing module it goes in, and
the existing code whose pattern it copies. Add a new pattern only when no row in the table fits, and say why in the
map.

| When you add | Put it in | Copy the pattern of |
|---|---|---|
| A setting | `config/schema.py` and a YAML file in `config/defaults/` | A schema default that reproduces the old behaviour, turned on in the YAML, and left out of the cache slice at its default (`s1_sample.py`, `s3_script_val.py`) |
| A change to one stage's output | The stage in `stages/` and its model in `artifacts/` | The `Stage` protocol in `stages/base.py`, with new fields given defaults |
| A new run file | A Pydantic model in `artifacts/`, a path on `MeetingPaths` and the stage's `writes` | `artifacts/io.py` for atomic writes |
| A model role | `backends/base.py` (protocol), `backends/aicore_impl.py` (client), `backends/fakes.py` (fake) and `config/defaults/backends.yaml` | An existing role, e.g., `sv_wavlm`, with record and replay through `backends/transport.py` |
| A check on a script or on audio | `validators/` | `validators/script_rules.py` (code checks) or `validators/voice_checks.py` |
| A CLI command | `orchestrator/cli.py`, and a recipe in `Justfile` when people run it often | An existing Typer command |
| A viewer read route | `viewer/api.py` | The whitelist of files and audio names in `api.py` |
| A viewer route that writes | `viewer/builder.py`, or a module it registers | `suggest_routes.py` and `export_routes.py`, which `builder.add_routes` registers, so `--read-only` drops them |
| A way for chat to change settings | `suggest/questions.yaml` | An existing question group with options and effects (ask first, because the decision model is trained on the questions) |
| A screen | `frontend/src/features/<area>/` and a route in `frontend/src/app/routes.tsx` | An existing screen built from `components/ui.tsx` and the hooks in `lib/api.ts` |
| A frontend helper or number derived from run files | `frontend/src/lib/` | `lib/geometry.ts` and `lib/stats.ts`, which compute in the browser |

## System overview

Synthetics has four parts. The pipeline makes meetings, the platform serves the models, the viewer API serves run
files and runs the builder, and the frontend is the web UI.

```
 frontend (React, Vite)  --HTTP-->  viewer API (FastAPI, backend/synthetics/viewer)
                                         |  reads run files, creates and starts runs
                                         v
 synthetics CLI (Typer)  ------->  pipeline: orchestrator -> stages 1 to 8 -> runs/<run_id>/
                                         |  aicore client
                                         v
                                   platform services in ../../platform (GPU and CPU)
```

The unit of work is a meeting. Each meeting has an id (`m-000001`) and a seed, is processed on its own, and can be
made again from its seed and the run config.

## Module map

All Python code is in `backend/synthetics/`.

| Module | What it holds |
|---|---|
| `config/` | `schema.py` (`RunConfig` and every setting), `distributions.py` (uniform, log-normal, Bernoulli, categorical and Zipf), `loader.py`, and `defaults/*.yaml`, which hold every default |
| `artifacts/` | One Pydantic model per run file, `paths.py` (`MeetingPaths`) and `io.py` (atomic writes) |
| `stages/` | `s1_sample.py` to `s8_labels.py`, `base.py` (the `Stage` protocol), `runner.py` (cache keys and status), `script_plan.py` (the stage 2 turn plan) and `prompts.py` |
| `orchestrator/` | `cli.py` (the `synthetics` command), `scheduler.py`, `lease.py` (GPU lease), `phase_c.py` and `phase_post.py`, `status.py`, `run_report.py` and `background.py` (runs started from the viewer) |
| `backends/` | Role protocols, `aicore` clients, fakes, and record and replay |
| `validators/` | Script rule checks, LLM judges, voice checks, the sampler report, the model comparison, and SV and judge calibration |
| `acoustics/` | Room, scene (poses per segment), noise, device chain and geometry |
| `export/` | RTTM, transcript and manifest writers, and `pack.py` (export packs) |
| `text/` | The Singlish convention, numbers as words, and PDPA stripping |
| `nsc/` | NSC pools and pool growth, reference extraction and cleanup, and non-speech clips |
| `voice/` | Audio operations, the non-speech sound library and stage 5's measurement cache |
| `suggest/` | `questions.yaml` (decision questions), the engine, effects, decision model clients and training data generation |
| `viewer/` | `api.py` (read routes and audio), `builder.py` (create, start and stop runs, pilots, review marks and voice swaps), `suggest_routes.py`, `export_routes.py`, `admin.py` (speaker pool, `--admin` only) and `contract.py` |

## Pipeline stages

Each meeting goes through eight stages, and the GPU stages run in phases because only one phase's models fit on the
GPU at a time.

| Stage | Name | Runs in | Writes |
|---|---|---|---|
| 1 | Meeting sampler | CPU | `meeting.json` |
| 2 | Script generation | GPU phase A | `script.json`, `prompts/<speaker_id>.txt` |
| 3 | Script validation | GPU phases B and B2 | `script_validation.json` |
| 4 | Voice synthesis | GPU phase C | `utterances/*.wav`, `synthesis.json` |
| 5 | Voice validation | GPU phase C | `voice_validation.json`, `voice_measures.npz` |
| 6 | Conversation assembly | CPU | `timeline.json`, `tracks/<speaker_id>.wav` |
| 7 | Acoustic simulation | CPU | `mixture.wav`, `mixture_mc.wav` for array mics, `stems/`, `noise.wav`, `render.json` |
| 8 | Labels and packaging | GPU phase D | `labels/meeting.rttm`, `labels/transcript.json`, `labels/manifest_entry.json` |

Stage 3 sends a failed script back to stage 2, and stage 5 sends a failed utterance back to stage 4.

### Stage contract, caching and revisions

A stage declares what it reads and writes, its model roles and its version. It reads earlier stages' files, writes
its own files and a status record, and does nothing else, and `stages/runner.py` enforces the contract.

A stage is skipped when `status.json` holds a pass with the same cache key. The key is a hash of the stage id and
version, the stage's slice of the config, the hashes of its inputs, any pending feedback and, from revision 2, the
run revision. As a result, a change to render settings runs only stages 7 and 8 again, and a change to a sampled
acoustic distribution runs stages 1, 7 and 8 again, never the LLM and TTS stages.

A run's config is frozen when the run is created. A change that would alter old runs' draws or cache keys is gated
on `RunConfig.revision`, which is 1 for a run made before revisions and 2 for new runs. Under revision 2, stage 1
draws the acoustic values (`ACOUSTIC_FIELDS` in `artifacts/meeting.py`) from a random stream of their own, and stages
2 to 6 hash `meeting.json` without those fields, so a change to acoustics never moves the other draws. The same seed
draws different meetings under the two revisions.

Stages 3, 5 and 7 can be switched off. A switched off stage is recorded as skipped, later stages go on without its
output, and the manifest records the switches in `stages_enabled`.

### Scheduling, waves and GPU phases

The scheduler runs each stage across all meetings of a wave, so only one phase's models use the 32 GB GPU at a time.
A wave is `phases.wave_size` meetings (10 for new runs, and one wave for a run without the setting), and each wave
finishes all eight stages before the next starts. Results don't depend on waves, because every draw is seeded per
meeting and stage. Execution is sequential, with one meeting and one request at a time, and `status.json` decides
what is left.

| Phase | Services loaded together | Retry handling |
|---|---|---|
| A, generation | `qwen-sea-lion-v4-5-27b-it` | Writes again the scripts that failed phase B, in the next round |
| B, judging | `gemma-sea-lion-v4-27b-it-nvfp4` | Borderline scripts are deferred to B2 |
| B2, tie break | `apertus-sea-lion-v4-8b-it` | Runs over deferred scripts only |
| C, voice | `voxcpm2`, `qwen3-asr-1-7b`, `wavlm-sv`, `redimnet2-sv`, `utmos` | Re-synthesis and reference swaps inside the phase |
| D, alignment | `qwen3-forced-aligner` | None |
| prep | `deepfilternet3`, `wpe-dereverb`, `dnsmos` | NSC reference cleanup only, not part of a run |

Phases A and B alternate for up to 3 rounds, because the generator and the judge don't fit on the GPU together. The
two judges don't fit together either, so the tie break is its own phase.

For each phase, `orchestrator/lease.py` runs the platform's `tools/stack.py up` with the phase's services, which
checks the VRAM budget and waits for health. The scheduler then runs the phase's stages over every pending meeting
and runs `stack.py down`. A phase with no pending meeting is never leased, fake and replay modes never lease, and
`NO_EVICT=1` stops `stack.py` from stopping other projects' services. The platform runs one GPU service at a time,
except a GPU pack. Phase C needs its four GPU models in one loop, so it is the pack `synthetics-phase-c`, which the
platform starts together with a share of the GPU each (8, 7, 3 and 2 GB minimums).

### Model roles and services

Every model is a platform service, reached through `aicore`. LLMs and ASR are served by vLLM on the OpenAI API, and
TTS, speaker embeddings, MOS and the aligner are custom services on the OIP v2 inference API. Weights live in
`$AI_ASSETS`, which every service mounts.

| Stage | Role | Service | Model |
|---|---|---|---|
| 2 | `generator` | `qwen-sea-lion-v4-5-27b-it` | Qwen-SEA-LION-v4.5-27B-IT, NVFP4A16 (weights in FP4) |
| 3 | `judge_primary` | `gemma-sea-lion-v4-27b-it-nvfp4` | Gemma-SEA-LION-v4-27B-IT-NVFP4 |
| 3 | `judge_tiebreak` | `apertus-sea-lion-v4-8b-it` | Apertus-SEA-LION-v4-8B-IT, borderline scripts only |
| 4 | `tts` | `voxcpm2` | VoxCPM2 2B in `ultimate` mode, cfg 2.0, 10 steps |
| 5 | `sv_wavlm`, `sv_redimnet` | `wavlm-sv`, `redimnet2-sv` | WavLM-Large with ECAPA-TDNN, and ReDimNet2 |
| 5 | `asr_primary` | `qwen3-asr-1-7b` | Qwen3-ASR-1.7B |
| 5 | `asr_singlish` | `meralion-3-3b-asr` | MERaLiON-3-3B-ASR, not leased by default |
| 5 | `quality` | `utmos` | UTMOS22 on the CPU |
| 8 | `aligner` | `qwen3-forced-aligner` | Qwen3-ForcedAligner 0.6B |
| prep | `enhancer`, `dereverb`, `ref_quality` | `deepfilternet3`, `wpe-dereverb`, `dnsmos` | Reference cleanup on the CPU |
| viewer | `decision` | `laya-synthetics` | Laya (421M) fine-tuned on `suggest/questions.yaml`, on the CPU |

The judges use a different model family from the generator. The generator is weight-only FP4 because AI Singapore
publishes no 4-bit Qwen-SEA-LION, and quantising activations too needs calibration on a loaded 52 GB model, which
doesn't fit in this machine's RAM.

## Meeting sampling

Stage 1 samples each meeting's room and people from the settings in `room_classes.yaml`, `seating.yaml`,
`acoustics.yaml`, `turn_taking.yaml` and `personas.yaml`, and it writes `meeting.json`, which every later stage
reads. The defaults come from general knowledge of meeting corpora, not from measurements of Singapore recordings,
so the fine-tuning team can replace them.

The class boundary is fixed at 7 speakers, so small rooms have 2 to 6 speakers on a laptop mic and large rooms have 7
to 20 on a table mic. Stage 1 samples in this order:

1. First, it picks the room class from a ratio (30% small and 70% large by default) and samples the speaker count
   from the class range.
2. Second, it samples the room and the table and checks the fit. The table seats about one person per 0.6 to
   0.75 m of edge with at least 1 m to each wall. Small rooms resample until everyone fits, and large rooms put the
   overflow on wall chairs.
3. Third, it places each person at a table seat, a wall seat (large rooms only) or a presenter spot by the front
   wall, standing with the mouth at 1.5 to 1.75 m. The chair is a person at the table, weighted toward seniority,
   and speaker ids follow turn share, so `S01` is the chair.
4. Fourth, it places the mic. A laptop sits 0.3 to 0.5 m in front of one seat. A table mic sits at the table centre,
   or with p=0.3 in front of the chair (`mic.type` "chair").
5. Fifth, it samples the acoustics, the meeting dynamics (turn shares, a presentation, hybrid callers and the
   topic) and the voices. Each person gets an NSC speaker from the meeting's split, and the persona's gender,
   ethnicity and languages follow that speaker. `speaker_pool.ethnicity_mix` can set target shares per ethnicity.

## Script generation and judging

Stage 2 plans the turns before it writes words. The plan in `stages/script_plan.py` fixes who speaks and each turn's
interaction (`normal`, `interruption`, `backchannel`, `overlap_start`, `side` or `presentation`), meets the turn
shares and turn-taking rates by construction, and stays the same across rounds, so a new round changes only the
words. The generator writes the planned turns as JSON in chunks of 40, with the last 30 turns as context, and each
speaker's persona prompt is saved word for word.

Stage 3 runs its checks in code first (`validators/script_rules.py`), e.g., speaker count, turn shares and
interaction rates. The LLM judge then scores the script with the rubric in `judge.rubric_version`, and it writes
its rationale before its scores.

- `rubric_v2` scores coherence, Singlish, realism, persona consistency per speaker and overall.
- `rubric_v3` scores responsiveness, consistency, progression, language fit (each speaker's English against their
  own background), realism, persona consistency, social dynamics and overall.

A script passes when its overall score is at or above `thresholds.judge_pass` (3.0 on a 1 to 5 scale). With
`thresholds.judge_criterion_floor` set, any criterion or persona score at or below the floor also fails the script,
and the feedback names the low scores. When the overall score is within 0.5 of the pass mark, the tie break judge
scores the script on its own in phase B2, and both judges must pass it. A failed script goes back to stage 2 with
the feedback, for up to 3 rounds, and the meeting is then dropped. `just judge-set` exports a rating sheet for people
to fill in, and `just fit-judge` fits the pass mark from it.

## Voice synthesis and checks

Stage 4 clones each speaker from one cleaned NSC reference clip of 5 to 15 s and keeps that clip for the whole
meeting. Long turns are split at sentence ends, non-speech tags become NSC laughter, coughs and breaths from the
sound library, and each utterance is one dry WAV with at most 50 ms of silence at each end.

Stage 5 checks every utterance, and the thresholds come from `just bakeoff` and `just calibrate-sv`, which writes
`config/sv_thresholds.yaml`:

- Speaker similarity passes only when both SV models score the utterance above their equal error rate thresholds.
  Utterances under 1 s, backchannels and turns without words skip the check.
- Consistency is the mean cosine between a speaker's own utterances of 3 s or more, and separability is the mean
  cosine across two speakers' utterances.
- The ASR round trip runs with the language fixed to English, and the error is scored after Singlish normalisation
  (`wer_max` 0.35, or `cer_max_short` 0.40 for 3 words or fewer).
- Quality needs a UTMOS of at least 1.0 and no more than 0.75 below the speaker's own reference, because UTMOS rates
  the real references from about 1.3 to 3.3.

A failing utterance is synthesised again with a new seed, up to 3 attempts. A speaker who fails consistency, or the
quieter speaker of a pair that is too close, gets a reference swap, first to another clip of the same NSC speaker and
then to a same-split speaker of the same gender and ethnicity. On a retry, only the new utterances reach the models,
because `voice_measures.npz` keeps every measurement by the audio's sha256 and service. The weakest cases are short
sentences ending in a particle, backchannels fused with the next word, local names in short sentences and cut-off
turns, and they make up most of the about 1% of utterances kept after failing every attempt.

## Turn-taking

Stage 6 places utterances on a timeline, and each main turn follows the floor with the transition its interaction
asks for, i.e., a sampled gap or long pause, an overlap lead, or an interruption that cuts off the floor speaker.
Backchannels go inside other speakers' turns, side conversations run alongside the main one, and nobody overlaps
their own speech. The defaults are in `turn_taking.yaml`:

| Setting | Default for new runs |
|---|---|
| Turn shares | Zipf, exponent 1.0, with the chair taking the top share |
| Gaps | Log-normal, median 0.5 s, capped at 2.5 s |
| Overlapped speaker changes | 35%, starting 0.3 to 2.0 s before the previous turn ends, at most half its length |
| Overlap target | 8 to 15% of speech time in small rooms, 12 to 20% in large rooms |
| Interruptions | 2 to 5% of speaker changes, plus 10% of the questions after a presentation |
| Backchannels | 1 to 3 per minute |

The meeting's overlap target is drawn once. Stage 6 then scales the overlap leads, up to 3 times and never more than
half of either turn, and when that isn't enough, it lets more normal speaker changes overlap. The overlap share and
the target move together, because a lower share alone only makes stage 6 add the missing overlap to speaker changes
the script didn't plan as overlaps. When a meeting comes out shorter than planned, stage 6 drops some backchannels to
keep the planned rate.

## Acoustics

Stage 7 builds the room in pyroomacoustics from `meeting.json`, with image sources in small rooms and image sources
plus ray tracing in large rooms. It scales absorption until the talkers' median mid-band RT60 matches the sampled
target, and it fails when the measured RT60 is more than 15% off. Talkers have directivity and turn between turns,
presenters often face the screen, and levels are set at the source and never normalised, so far talkers stay quiet.

Noise beds (HVAC and projector fan) are added at the sampled SNR, and keyboard typing and events are added relative
to the speech. The device chain then applies a band limit, AGC, clipping and the Opus codec, in that order. Remote
callers pass through their own band limit and codec first, and hybrid meetings get the side effects of an echo
canceller. With `simulation.write_stems`, which is on for new runs, stage 7 also writes each speaker as heard at the
mic and the noise alone. The mixture is the same with or without stems.

## Labels

Stage 8 writes the RTTM from the timeline and force-aligns each utterance on its dry audio, one utterance per
request, for word timestamps. It writes the transcript in the Singlish convention and the meeting's manifest entry.
It fails the meeting when any of these checks fails:

1. The RTTM's total speech differs from the timeline segments by more than millisecond rounding.
2. An aligned word lies outside its segment by more than `thresholds.align_slack_s` (50 ms).
3. An utterance's alignment coverage is below `thresholds.align_coverage_min` (90%).

## Speaker pools and splits

Every NSC speaker has one split for good in a pinned pool CSV, and each run copies the splits into
`speaker_splits.json`. A meeting clones voices only from its own split, so no speaker appears in two splits, within a
run or across runs, and people who recorded together always share a split.

| Pool | File | Speakers (train, dev, test) |
|---|---|---|
| p3-175-v1 | `data/nsc_part3_reference_speakers.csv` | 175 (50, 25, 100) |
| p3-406-v2, for new runs | `data/pools/p3-406-v2.csv` and its sidecar `p3-406-v2.json` | 406 (170, 58, 178) |

`synthetics pool build` grows a pool from a curated voice folder. It keeps every earlier speaker's split, puts a new
speaker whose partner is in the pool into the partner's split, and spreads the rest at 70, 15 and 15% with a seeded
draw, balanced by gender and ethnicity. A pool is never rewritten, the sidecar holds the counts that validation
checks, and `speaker_pool.refs_index` points runs at the folder's clip manifest.

## Decision model

The viewer turns a user's words into settings by asking the decision model the yes or no and multiple choice
questions in `suggest/questions.yaml`. The model only gives each option a probability, and `suggest/effects.py`
applies the chosen options' effects, clamped to the file's bounds. Confident answers apply, uncertain ones are offered
as choices, and stage switches and backends are always asked. The engine asks in two calls, first the global
questions and every group's gate, and then the scope and choices of the groups in play.

The service is `laya-synthetics` on the platform's `decision.systemone` API, on the CPU, so it answers while a GPU
phase runs. The viewer checks its health first and falls back to word matching when it is down, and
`SYNTHETICS_DECISION_SERVICE=0` forces word matching. Training rows come from `synthetics suggest-data`
(`suggest/datagen.py`), and the rows, training scripts and their README live in
`/home/user/code/datasets/synthetics-decisions/`.

| On the 50 hand-written requests in `tests/data/suggest_eval.jsonl` | Right option | Changes nobody asked for |
|---|---|---|
| Word matching | 57% | 20 |
| `laya-synthetics` | 62% | 1 |

## Run directory layout

A run lives in `runs/<run_id>/`, or under `SYNTHETICS_RUNS` when it is set.

```
runs/<run_id>/
  config.yaml                  frozen run config (every distribution and setting used)
  speaker_splits.json          NSC speaker -> train, dev or test (fixed by the pool CSV)
  review.json                  keep or leave out marks from the review screen (optional)
  manifest.jsonl               one line per finished meeting
  run_report.json              drops, failures, time per phase and split check (synthetics report)
  voice_report.json            phase C rates (synthetics voice-report)
  process.json, process.log    a run started from the viewer, and its output
  judge_sets/                  rating sheets from synthetics export-judge-set
  meetings/<meeting_id>/
    status.json                per stage state, attempts, times, detail line and metrics
    meeting.json               stage 1
    prompts/<speaker_id>.txt   stage 2, the persona prompt word for word
    script.json                stage 2
    script_validation.json     stage 3
    utterances/<turn_id>.wav   stage 4, dry
    synthesis.json             stage 4, the voice behind each speaker
    voice_validation.json      stage 5
    voice_measures.npz         stage 5's cache of measurements, safe to delete, not an output
    timeline.json              stage 6
    tracks/<speaker_id>.wav    stage 6, dry and time-aligned
    mixture.wav                stage 7, and mixture_mc.wav for array mics
    stems/<speaker_id>.wav     stage 7, one speaker as heard at the mic
    noise.wav                  stage 7, the noise alone
    render.json                stage 7, what the simulation measured and applied
    labels/meeting.rttm        stage 8
    labels/transcript.json     stage 8
    labels/manifest_entry.json stage 8
```

## Output contract (schema version 1)

The files below are what Synthetics hands to the fine-tuning application. Every meeting JSON file and every manifest
line carries `schema_version: "1"`. New fields are always added with defaults, and a rename, removal or change of
meaning raises the version, so a reader written for version 1 keeps working. The Pydantic models in
`backend/synthetics/artifacts/` define every field, and the sections below explain what the fields mean.

A meeting is complete when its `status.json` shows `pass` for stage 8, and only complete meetings appear in
`manifest.jsonl`. The manifest's `files` map gives the labelled outputs' paths relative to the run folder, and the
run report, voice report, process files, judge sets and `voice_measures.npz` are for the pipeline team and are not
part of the contract. `review.json` maps a meeting id to `{decision: keep | exclude, note, at}`, and the pipeline
never reads it.

### Time and audio

All times are seconds from the first sample of the meeting's audio. Every WAV of a meeting is 16-bit PCM at the run's
`sample_rate` (16 kHz by default), and every WAV has `timeline.stats.duration` times the sample rate samples.

- `mixture.wav` is mono. For an array mic it is the mean of the channels after the device chain, then passed through
  the codec when the meeting has one.
- `mixture_mc.wav` exists only when `mic.channels` is above 1. Channel `k` is the capsule at
  `mic.position + mic.array_geometry[k]`, with the device chain up to clipping but no codec.
- `tracks/<speaker_id>.wav` holds one speaker's utterances at their timeline positions, dry, with silence elsewhere.
  A truncated utterance stops at its cut point with a 10 ms fade.
- `stems/<speaker_id>.wav` is one speaker through the room at their mixture level, and `noise.wav` holds the noise.
  Both get the mixture's band limit, gains and AGC gain curve but no clipping or codec, so the stems plus the noise
  add up to the mixture before those two, within 16-bit rounding. A remote caller's stem has their far end path but
  not the echo canceller's ducking, and `render.json` lists the files as `stem:<speaker_id>` and `noise`.

### Timeline (`timeline.json`)

`segments` holds one entry per heard turn, sorted by start, and `end - start` is the audible length. When a turn was
cut off, `truncated_at` is the number of seconds into the utterance where it was cut. One speaker's segments never
overlap each other, and `kind` is the turn's script interaction, which says what the script asked for, not how the
turn was timed. `dropped_turns` lists the backchannels left out of a meeting that came out shorter than planned.

| `stats` field | Meaning |
|---|---|
| `duration` | Length of the meeting audio |
| `speech_time` | Sum of segment lengths, each speaker counted on their own, equal to the RTTM total |
| `voiced_time`, `overlap_time` | Time with at least one speaker active, and with two or more |
| `overlap_share`, `overlap_target` | `overlap_time / voiced_time`, and the meeting's sampled target for it |
| `backchannels`, `backchannels_per_min` | Heard backchannels and their rate |
| `speaker_changes` | Transitions between main turns of different speakers |
| `interruptions`, `interruption_share` | Turns cut off by the next speaker, and their share of speaker changes |
| `long_pauses` | Transitions with a long pause |

### Script checks (`script_validation.json`)

Stage 3 records the code checks in `rules`, each with its value and limit, and each judge's result in `primary`
and, when the score was near the pass mark, `tiebreak`. A judge's `scores` hold the criteria of its
`rubric_version`, each from 1 to 5, and a criterion the rubric doesn't ask for is left out.

| Rubric | Criteria in `scores` |
|---|---|
| `rubric_v1`, `rubric_v2` | `coherence`, `singlish`, `realism`, `persona_consistency` (per speaker), `overall` |
| `rubric_v3` | `responsiveness`, `consistency`, `progression`, `language_fit`, `realism`, `persona_consistency` (per speaker), `social_dynamics`, `overall` |

Every rubric has `realism`, `persona_consistency` and `overall`, so a reader that uses only those works with all of
them.

### Voice checks (`voice_validation.json`)

- `utterances` has one entry per checked turn. `sv` maps each SV model to the cosine with the reference, `asr` lists
  the ASR checks (primary first), `mos` is the predicted MOS, `passed` and `reasons` give the verdict, and `skipped`
  names checks that didn't apply.
- `speakers` has one entry per speaker with `consistency`, `consistency_utts` and `ref_mos`, and `pairs` has one
  entry per speaker pair with `cross_cos` and `too_close`.
- `kept_failed` lists turns that still failed after `retries.tts_attempts`. With `voice.on_exhausted: keep_flagged`,
  the default, they stay in the audio and the labels, and the manifest counts them.

### Speaker turns (`labels/meeting.rttm`)

The RTTM has one line per timeline segment, in start order, with millisecond precision.

```
SPEAKER <meeting_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

Speaker labels are local to the meeting. The NSC voice behind each label is in the manifest's `speakers` list, taken
from `synthesis.json`, which holds the voice after any reference swap, while `meeting.json` keeps the first voice
stage 1 assigned.

### Transcript (`labels/transcript.json`)

`utterances` holds one entry per timeline segment, in the same order.

- `text` follows the Singlish transcription convention below, and `text_norm` is lowercase with markup and
  punctuation removed, keeping particles and fillers as words. Use `text_norm` for WER.
- `words` lists every aligned word with absolute `start` and `end` times and the aligner's `score`. A word differs
  from its `text_norm` word only by case and by apostrophes at its edges.
- `truncated` is true when the speaker was cut off, and then the text keeps only the words heard before the cut.
  `align_coverage` is the share of the turn's words that the aligner matched.

The aligner gets `zh` for an utterance with Chinese characters and `en` otherwise, and `aligner` names the service.

### Manifest (`manifest.jsonl`)

The manifest has one JSON object per complete meeting, sorted by meeting id. It is rebuilt from the
`labels/manifest_entry.json` files after every stage 8 phase, and by `synthetics manifest -r <run>`.

| Field | Content |
|---|---|
| `run_id`, `meeting_id`, `seed` | The meeting can be made again from `seed` and `config.yaml` |
| `split`, `room_class`, `n_speakers`, `duration` | For slicing results |
| `sampled` | Every scalar of `meeting.json` with dotted keys, e.g., `room.rt60_target`, `mic.type` ("laptop", "centre" or "chair") and `acoustics.snr_db` |
| `stages_enabled` | Stage name to on or off |
| `position_counts` | People at the table, on wall seats, presenting and remote |
| `speakers` | Per speaker, the NSC voice after any swap, split, position type, wall reason, share of speech time and ASR-limited count |
| `judge` | Script rounds, decision, rules passed, borderline, primary and tie break scores and services |
| `voice` | TTS service, utterance and re-synthesis counts, library turns, reference swaps, check counts, pairs too close and `kept_failed` |
| `turn_taking` | The timeline `stats`, raw counts included |
| `acoustic` | Stage 7 measurements from `render.json`, flattened |
| `labels` | Aligner, utterance and word counts, truncated utterances, alignment coverage and dropped turns |
| `files` | Output paths relative to the run folder |

### Acoustic report (`render.json`)

`rt60_measured` is the median over the talkers of the mid-band T20 (the mean of the 500 Hz and 1 kHz octave bands) on
each talker's impulse response at the first mic. `snr_db` is speech power over the power of the noise beds on the
single channel before the device chain. Typing and events are listed in `noise` with their times and positions.

### Export packs

The Export screen and `export/pack.py` write a run's kept meetings to a folder in `$SYNTHETICS_EXPORT_ROOT`
(`/home/user/code/datasets/synthetics-exports` by default). Every WAV in a pack is 16-bit mono PCM at 16 kHz.

```
<pack>/README.md                                   what the pack holds
<pack>/manifest.jsonl                              the kept meetings' manifest lines
<pack>/<meeting_id>/mixture.wav                    the recording
<pack>/<meeting_id>/meeting.json, timeline.json, labels/meeting.rttm, labels/transcript.json
<pack>/<meeting_id>/speakers/clean/<speaker_id>.wav   full length, from tracks/ (optional)
<pack>/<meeting_id>/speakers/room/<speaker_id>.wav    full length, from stems/ (optional)
<pack>/<meeting_id>/clips/clean/<turn_id>.wav         one line, cut from tracks/ (optional)
<pack>/<meeting_id>/clips/room/<turn_id>.wav          one line, cut from stems/, with 0.3 s of echo tail (optional)
<pack>/<meeting_id>/clips/clips.csv                   clip, speaker_id, start, end, text, text_norm, truncated
<pack>/<meeting_id>/noise.wav                         the noise alone (optional)
```

Clips are cut at the timeline's bounds, so a line cut short ends where it was cut.

### Approximations

- Talker directivity is a cardioid family pattern, about 8 dB quieter behind the talker. In large rooms the ray
  traced late reverberation comes from an omnidirectional source scaled by the pattern's average power, because the
  library supports directivity only for early reflections.
- The echo canceller is modelled by its side effects only. Local talkers are turned down while a remote caller
  speaks, and the start of each remote turn is attenuated.
- Noise comes from `$SOUND_LIBRARY_DIR/noise/index.yaml`, built by `data/sound_library/build_noise_library.py`.
  Kinds without clips (projector fan, chair and paper) are generated, and `render.json` names every source.
- A meeting with few speaker changes can run out of transitions to overlap. It then stays under its target, and
  stage 6 says "out of reach" on its status line.

## Singlish transcription convention

Synthetics defines one transcription convention, and the fine-tuning application uses it for training and scoring.
It follows how MERaLiON-3-3B-ASR writes its output, stage 2 writes scripts in it, and stage 8 writes transcripts in
it. The lexicon of particles, fillers, loanword spellings and TTS hints is in `config/defaults/singlish_lexicon.yaml`.

| Element | Rule | Example in `text` |
|---|---|---|
| Discourse particles | Canonical spelling in parentheses, with variants mapped (la to lah, lo to lor) | ok (lah), we go first |
| Particle set | (lah) (leh) (lor) (meh) (hor) (ah) (sia) (mah) (liao) (wor) | can or not (meh)? |
| Fillers | (uh) (um) (mm) in parentheses, with erm mapped to um | (um) I think so |
| Mandarin, and full Hokkien or Cantonese clauses | Simplified Chinese characters | then 他会教 |
| Tamil | Tamil script | |
| Malay | Latin script, standard spelling | we go makan first |
| Singlish loanwords | Singapore spelling from the lexicon | very shiok, paiseh (ah) |
| Acronyms | Uppercase, no dots, one token | the HDB flat |
| Numbers | Written as spoken words | twenty twenty six |
| False starts and cut-offs | A trailing hyphen | we wan- we want, I think we should- |
| Non-speech | Square bracket tags, left out of WER | \[laugh\] true |

TTS input comes from `text` with the parentheses removed and non-speech tags replaced by sound library clips. NSC
writes particles in square brackets and names as `#name#`, so the reference loader converts NSC notation, and clips
with `#name#` are never used.

## Viewer API

`synthetics viewer` serves the API and the built frontend from `frontend/dist` on port 8610. The routes come in
groups, and the flags decide which groups exist.

| Group | Module | Present when | Routes |
|---|---|---|---|
| Read | `viewer/api.py` | Always | `GET /api/viewer`, `/api/runs`, `/api/runs/{run}/config`, `/manifest`, `/summary`, `/speaker_splits`, `/meetings`, `/meetings/{meeting}/{artifact}`, `/meetings/{meeting}/audio/{name}`, `/review`, `/jobs`, `/export` |
| Builder | `viewer/builder.py` | Not `--read-only` | `GET /api/defaults`, `POST /api/preview`, `POST /api/runs`, `GET /api/runs/{run}/process`, `POST /api/runs/{run}/start`, `/stop`, `DELETE /api/runs/{run}` (pilot runs only), `POST .../voice-swap`, `POST /api/runs/{run}/review` |
| Suggest | `viewer/suggest_routes.py` | Not `--read-only` | `GET /api/topics`, `POST /api/suggest`, `POST /api/suggest/apply` |
| Export | `viewer/export_routes.py` | Not `--read-only` | `GET /api/runs/{run}/exports/estimate`, `POST` and `GET /api/runs/{run}/exports`, `GET .../meetings/{meeting}/download` |
| Admin | `viewer/admin.py` | `--admin`, never with `--read-only` | `GET /api/admin/pool`, `POST /api/admin/pool/drafts`, `GET` and `DELETE /api/admin/pool/draft`, `PATCH .../draft/speakers/{speaker}`, `POST .../draft/publish`, `GET /api/admin/voices/{speaker}/{clip}` |

The read routes check run ids, meeting ids and file names against what is on disk and a whitelist. Audio routes
serve only `mixture.wav`, `utterances/<turn>.wav`, `tracks/S<n>.wav`, `stems/S<n>.wav` and `noise.wav`, with byte
ranges. A run started from the viewer runs in a background process (`orchestrator/background.py`), and only one run
can run at a time.

### Viewer data contract

The viewer reads these files through the API, and `viewer/contract.py` checks the fields it uses.
`synthetics viewer-contract -r <run>` runs that check on every meeting of a run, and
`tests/test_viewer_contract.py` runs it on a fake run.

| File | Read by | Fields used |
|---|---|---|
| `manifest.jsonl` | Browser, Distributions, Detail | Meeting id, room class, speaker count, `position_counts`, `sampled`, split, `speakers`, `turn_taking`, `acoustic` |
| `config.yaml` | Distributions, New dataset | Room class ratio, per class ranges, hybrid, presentation and wall seat probabilities, turn-taking targets |
| `meeting.json` | Browser cards, Detail, Distributions | Room, front wall and screen, table, positions and mouth heights, presenter spots, mic, surfaces, noise, device chain, personas, turn shares, `people[].facing_deg`, `fit_resamples` |
| `status.json` | Browser, Detail, Pipeline | Per stage status, timings, detail line and metrics |
| `script.json`, `prompts/<speaker_id>.txt` | Detail persona panel | Turns with speaker, interaction, `text` and `text_norm`, and the persona prompt word for word |
| `script_validation.json` | Detail persona panel | The judge's persona consistency score |
| `synthesis.json`, `voice_validation.json` | Detail speaker card and table | NSC voice after any swap, SV scores, ASR-limited flags |
| `timeline.json`, `render.json` | Detail | Turn start times, dropped turns and measured RT60 |
| `process.json`, `process.log` | Pipeline, jobs card | State of a run started from the viewer, and its log |

The browser computes the mouth to mic distance in 3D, facing angles, room volume and histogram bins. The viewer takes
a speaker's voice from the manifest or `synthesis.json`, because `meeting.json` keeps the first assignment.

## Frontend structure

The frontend is React 19 with React Router 7, TanStack Query for server data, Tailwind 4 and Vite, and its tests use
vitest. `frontend/src/main.tsx` mounts the app, and `app/routes.tsx` defines the routes inside `app/Shell.tsx`, which
holds the sidebar.

```
/                                   Home: the guided flow when the builder is on, or the latest run
/create                             features/guided/Describe
/create/plan                        features/guided/Plan
/create/pilot/:run                  features/guided/Pilot
/runs/:run/review                   features/guided/GenerateReview
/runs/:run/export                   features/guided/Export
/runs/:run/meetings                 features/browser/MeetingBrowser (with MiniPlan)
/runs/:run/meetings/:meeting        features/detail/MeetingDetail (FloorPlan, Elevation, SidePanels,
                                    PersonaPanel, SpeakersTable)
/runs/:run/distributions            features/distributions/Distributions (with Histogram)
/runs/:run/pipeline                 features/pipeline/Pipeline
/new                                features/builder/NewDataset (FieldInput, PreviewPanel)
/admin/speakers                     features/admin/SpeakerPool (--admin only)
```

The sidebar groups the screens as "Make a dataset" (the guided flow), "Details" (the engineer screens) and
"Advanced" (the settings form). Shared code lives in a few places:

- `components/ui.tsx` holds the base components, e.g., `Panel`, `PageHeader`, `Button`, `Toggle`, `Chip`, `Tag`,
  `Fact`, `StatusDot` and `InfoTip`, and `components/MetricTile.tsx` and `SortableTable.tsx` hold the rest.
- `components/guided.tsx` holds the guided flow's step bar, audio player, chat panel and change chips, and
  `components/plan.tsx` holds the floor plan drawing parts.
- `lib/api.ts` holds every API call and TanStack Query hook, e.g., `useRuns`, `useMeetings`, `usePreview`,
  `useCreateRun` and `useRunAction`.
- `lib/draft.ts` holds the settings form's fields with their info text, and `lib/guided.ts` holds the guided flow's
  state, which is kept in the browser until a run exists. `lib/geometry.ts`, `stats.ts`, `people.ts`, `meetings.ts`
  and `format.ts` compute derived values.

## Styling patterns

The viewer uses the AeroAudio design system, i.e., dark slate only, Inter for text, and JetBrains Mono with tabular
figures for every number, id and unit. Surfaces are flat with 1 px borders. The colour tokens are defined in the
`@theme` block of `frontend/src/index.css`, and components use them as Tailwind classes, e.g., `bg-panel` and
`text-muted`, never raw hex values.

| Meaning | Token | Used for |
|---|---|---|
| Small room | `color-cat-1` (blue) | Seats on small room plans, swatches and chart bars |
| Large room | `color-cat-2` (orange) | Seats on large room plans, swatches and chart bars |
| One series | `color-primary` | Charts over all meetings, running status |
| Selection | `color-focus` | Selected seat, mouth to mic path and its label |
| Done, failed | `color-good`, `color-bad` | Status dots, and a hybrid gap over 2 points |
| Overlap | `color-overlap` hatch | Overlap start turns and overlaps on speaker lanes |

The categorical palette, `color-cat-1` to `color-cat-6`, is assigned in fixed order and never cycled. Speakers are
never coloured, because meetings reach 20 speakers, so they are named by number and marked by selection.

Plans follow these rules:

- The front wall is always drawn on the left, with the screen as a `color-focus` bar on it. Glass walls are dashed
  and plasterboard walls are solid.
- Position types differ by shape, i.e., table and wall seats are circles, presenters are rounded squares and free
  seats are dashed circles. Seat size follows the square root of turn share, and the chair has a heavier outline.
- The detail plan is drawn at 90 px per metre and scrolls inside its panel instead of shrinking.

Accessibility and layout follow AeroAudio's rules:

- Every seat on a plan is a button with an accessible name and `aria-pressed`, and the focus ring is never removed.
- Colour is never the only signal, so status dots sit next to a status word, and each chart has a table view.
- From the `lg` breakpoint the sidebar is fixed at 216 px, and below it the sidebar becomes a top bar with a menu.
  Wide tables scroll inside their frame.

## Known limitations

| Limitation | Effect | How the design handles it |
|---|---|---|
| Synthetic turn-taking may be too clean | The data helps diarisation less on real meetings | Overlap, interruptions and backchannels are simulated, and every rate is a setting |
| Cloned voices may sound alike | Label noise in 20 speaker meetings | Separability checks with two SV models, and reference swaps |
| The SEA-LION judges may share biases | Both judges miss the same flaws | The pass mark can be fitted to human ratings |
| NSC references carry room colour | Double reverberation after simulation | References are denoised and dereverberated before cloning |
| Some planned interruptions can't be placed | A meeting can land a little under its interruption target | The rate over many meetings stays inside the configured range |
