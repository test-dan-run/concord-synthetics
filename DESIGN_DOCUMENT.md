# Synthetics pipeline design

Owner: @Dan

This document describes how the Synthetics pipeline works and why it is built that way. `docs/OUTPUT_CONTRACT.md`
is the reference for the output files, and the code is the reference for everything else.

## Background and goals

Synthetics generates synthetic Singapore meeting recordings with full labels, for meetings of 2 to 20 speakers.
Real Singapore meeting datasets are rare, and datasets with more than 15 speakers are rarer still. As a result,
current transcription and diarisation models perform poorly in this domain, and collecting real data at scale is
slow and costly.

Goals:

- Generate realistic meetings from 2 to 20 speakers, in small rooms recorded on a laptop mic and large rooms
  recorded on a table mic.
- Capture what makes real meetings hard: overlapping speech, short turns, backchannels, interruptions, and a long
  tail of rarely-speaking participants.
- Reflect Singapore speech: Singlish discourse particles, code-switching into Mandarin, Malay and Tamil, local
  accents, and local named entities and acronyms.
- Produce labels that match the audio exactly: speaker turns (RTTM), word-level timestamps and normalised
  transcripts.
- Run on a single RTX 5090 (32 GB VRAM), with every model served by the platform at `../../platform`.

## Scope and non-goals

Synthetics is a data generator only. It outputs labelled audio plus metadata, and a separate fine-tuning
application consumes it.

In scope:

- Meeting configuration sampling, script generation, voice synthesis, conversation assembly, acoustic simulation
  and label generation.
- Quality gates that guarantee audio and labels agree.
- Speaker-disjoint train, dev and test splits.

Out of scope, owned by the fine-tuning application:

- Collecting and labelling a real evaluation set.
- Model training, fine-tuning, evaluation and ablation studies.

Because the generator never measures real data itself, every distribution it samples from (turn-taking, overlap
rates, acoustics, noise) is a configurable input with defaults. The fine-tuning team can replace those defaults
with statistics measured on real recordings by editing YAML, without code changes.

Synthetics is built for research use. Every model in the stack is used under a licence that permits research, and
the National Speech Corpus (NSC) speaker subset used for voice references is already on disk.

## Pipeline overview

Each meeting passes through eight stages. GPU stages run in phases, because only one phase's models fit on the GPU
at a time.

| Stage | Name | Runs in | Writes |
| --- | --- | --- | --- |
| 1 | Meeting sampler | CPU | `meeting.json` |
| 2 | Script generation | GPU phase A | `script.json`, `prompts/<speaker_id>.txt` |
| 3 | Script validation | GPU phases B and B2 | `script_validation.json` |
| 4 | Voice synthesis | GPU phase C | `utterances/*.wav`, `synthesis.json` |
| 5 | Voice validation | GPU phase C | `voice_validation.json` |
| 6 | Conversation assembly | CPU | `timeline.json`, `tracks/<speaker_id>.wav` |
| 7 | Acoustic simulation | CPU | `mixture.wav` (and `mixture_mc.wav` for array mics), `render.json` |
| 8 | Labels and packaging | GPU phase D | `labels/meeting.rttm`, `labels/transcript.json`, `labels/manifest_entry.json` |

Stage 3 sends a failed script back to stage 2, and stage 5 sends a failed utterance back to stage 4. Execution is
sequential: stage 1 runs on every meeting before phase A, and stages 6 and 7 run after phase C, before phase D.

## Meeting config rules

Every meeting's room and setup is randomly sampled under two rules, with the class boundary fixed at 7 speakers:
small rooms use a laptop mic with fewer than 7 people, and large rooms use a table mic with up to 20 people. Large
rooms can also seat attendees on chairs along the walls, and either room class can have 1 or 2 presenters standing
at the front.

| Parameter | Small room | Large room |
| --- | --- | --- |
| Speakers | 2–6 | 7–20 |
| Mic | Laptop mic | Table mic (single mic or small array), at the centre or in front of the chair |
| Length × width | \~3–5 m × 2.5–4 m | \~6–12 m × 4–7 m |
| Ceiling height | \~2.6–3.0 m | \~2.7–3.5 m |
| Table | Small rectangular or round | Long rectangular or boardroom oval |

Sampling order:

1. Pick the room class from a configurable ratio (30% small, 70% large by default, weighted toward large
   meetings).
2. Sample the speaker count from the class range, using a configurable distribution.
3. Sample room dimensions and a table, and check the fit: the table seats about one person per 0.6–0.75 m of table
   edge, with at least 1 m of clearance to each wall, and the table stands at the centre of the room. Small rooms
   resample until everyone fits at the table. Large rooms may seat fewer people at the table than attend; the
   overflow goes to wall seats (step 4).
4. Place each person in one of three positions:
   - Table seats: around the table edge, mouths at about 1.1–1.2 m height with small jitter.
   - Wall seats (large rooms only): supplementary chairs in a row 0.3–0.6 m from a side or back wall, mouths at
     about 1.1–1.2 m height. These take the overflow once the table is full, and with a small probability a few
     attendees choose them even when table seats are free.
   - Presenter spots (both room classes, optional): 1 or 2 people standing near the front wall by the screen,
     mouths at about 1.5–1.75 m height, 0.5–1.5 m from the wall. Presenters count toward the speaker total.

   Seated positions stay fixed for the whole meeting. A presenter may drift within about 1 m of their spot between
   turns, and returns to a table or wall seat after presenting if the script says so.

   The chair is a table-seated person, weighted toward seniority. In large rooms the chair takes the head of the
   table with p=0.8: an end seat on the table's long axis, trading places with whoever sat there. Speaker ids
   follow turn share, so `S01` is the chair.
5. Place the mic:
   - Laptop: on the table edge, 0.3–0.5 m in front of one random seat, 0.1–0.25 m above the table.
   - Table mic (large rooms): at the table centre with a small offset, about 0.05 m above the surface (`mic.type`
     "centre"). With p=0.3 it sits instead 0.4–0.6 m (horizontally) in front of the chair's mouth, to capture the
     chair's decisions best (`mic.type` "chair").
6. Sample acoustics (wall materials, RT60, noise types and SNR) from per-class ranges.
7. Sample meeting dynamics: per-speaker turn shares (skewed, Zipf-like), whether the meeting has a presentation
   segment (and its presenters), whether the meeting is hybrid (remote participants through a speakerphone), and
   topic.
8. Assign each person an NSC reference speaker from the meeting's split, then sample the persona's gender,
   ethnicity, first language and code-switching languages to match that speaker, so the persona and the voice
   agree.

The result is written to `meeting.json`, which every later stage reads.

## Sampling statistics

Stages 1, 6 and 7 sample from the distributions below. They are defaults set from general knowledge of meeting
corpora, not measured on Singapore recordings. All live in `backend/synthetics/config/defaults/` and are recorded
per meeting in the manifest, so any run can be traced back to the values it used.

Turn-taking (stages 1 and 6):

| Parameter | Default |
| --- | --- |
| Turn shares | Zipf distribution, exponent 1.0; the chair takes the top share |
| Gap between turns | 50% of transitions: log-normal gap, median 0.5 s, sigma 0.5, capped at 2.5 s |
| Overlapping transitions | 50% of transitions: next turn starts 0.3–2.0 s before the previous ends, at most half its length |
| Overall overlap target | 10–20% of speech time (small rooms), 15–25% (large rooms) |
| Interruptions | 5–10% of turn changes; interrupted speaker cut off 0.3–1.5 s after the interruption starts |
| Backchannels | 1–3 per minute, from non-speakers, placed inside the current turn |
| Long pauses | 2% of transitions: 1–3 s |

Seating and presenters (stage 1):

| Parameter | Small room | Large room |
| --- | --- | --- |
| Wall seating | None | Overflow when table capacity is below the speaker count; otherwise p=0.15 that 1–3 attendees sit at the wall anyway |
| Wall-seat placement | None | 0.3–0.6 m from a side or back wall, chairs 0.5–0.7 m apart |
| Presentation segment | p=0.3 | p=0.4 |
| Presenters | 1 (p=0.8) or 2 (p=0.2) | 1 (p=0.7) or 2 (p=0.3) |
| Presenter position | Near the front wall, 0.5–1.5 m out, mouth height 1.5–1.75 m | Same |
| Presenter drift | Up to 1 m between turns, facing screen with p=0.3 | Same |
| After presenting | Return to a table seat with p=0.5, otherwise stay standing | Return to a table or wall seat with p=0.5 |

Acoustics (stages 1 and 7):

| Parameter | Small room | Large room |
| --- | --- | --- |
| RT60 | 0.3–0.6 s | 0.4–0.9 s |
| Walls | Mix of glass and plasterboard | Mix of glass and plasterboard |
| Floor / ceiling | Carpet p=0.7; acoustic tile p=0.6 | Carpet p=0.7; acoustic tile p=0.6 |
| SNR at mic | 10–30 dB | 5–25 dB |
| HVAC noise | Always on | Always on |
| Keyboard noise | p=0.6, near laptop | p=0.3 |
| Other events (doors, chairs, cups, paper) | 0–3 per 10 min | 1–5 per 10 min |
| Speaker orientation | Within ±60° of table centre, re-sampled per turn with p=0.3 | Same |
| Hybrid meeting | p=0.1 | p=0.3, 1–3 remote participants via speakerphone |
| AGC | p=0.7 | p=0.7 |
| Codec | p=0.5, Opus 16–32 kbps | p=0.5, Opus 16–32 kbps |
| Clipping | p=0.05 | p=0.05 |

## Stage designs

Each stage reads earlier stages' files, writes its own, and records a pass, fail, retry or defer status.

### Stage 2: Script generation

- Plan the turns first: who speaks and with which interaction type (normal, interruption, backchannel, overlap
  start, side conversation, presentation). The plan meets the turn shares from `meeting.json` and the overlap,
  interruption and backchannel counts by construction, places the presentation segment (monologue, then Q&A from
  the floor), and stays fixed across regeneration rounds, so a new round changes only the words.
- Assign a persona to each speaker: role, seniority, verbosity, accent and first language, code-switching
  tendency, and a stance in this meeting. Wall-seated attendees lean toward quieter personas with fewer turns.
- Save every persona prompt verbatim to `prompts/<speaker_id>.txt`.
- The generator writes the planned turns in chunks of structured JSON (40 turns per chunk, with the last 30 turns
  as context). Each turn carries its text and disfluency markers (false starts, repetitions, fillers), and is then
  brought into the transcription convention below.
- Include Singlish particles, code-switching and local entities and acronyms. Strip names and identifiers from any
  past-meeting context used in prompts, such as judge feedback (PDPA).

### Stage 3: Script validation

- Programmatic checks first: speaker count, turn shares, overlap and backchannel rates, persona IDs, schema
  validity.
- The primary LLM judge scores what code can't: persona consistency, topical coherence, natural Singlish. The
  rubric rewards realistic messiness, not polish. The judge writes its rationale before its scores.
- Tie-breaker: if the primary judge's overall score is within 0.5 of the pass mark (3.0 on a 1 to 5 scale), a
  second judge scores independently with the same rubric. Agreement decides; disagreement fails the script. The
  two judges do not fit on the GPU together, so a borderline script is deferred from phase B to phase B2, which
  reuses the primary judgement.
- A failed script goes back to stage 2 with the judge's feedback, for up to 3 rounds, and the meeting is then
  dropped.
- The pass mark is set in `thresholds.judge_pass`. `just judge-set` exports a rating sheet for human raters, and
  `just fit-judge` fits the pass mark and band from the filled sheet.

### Stage 4: Voice synthesis

- Clone each speaker from one clean reference clip of the NSC speaker that stage 1 assigned, and keep the same clip
  for every utterance of that speaker in the meeting. References are 5–15 s conversational NSC clips with their
  transcripts, denoised and dereverberated first so the TTS output is dry.
- VoxCPM2 runs in `ultimate` mode: the reference clip is both the prompt (with its transcript) and the reference.
- TTS input comes from `text`, with parentheses removed. Long turns are split at sentence ends and cloned piece by
  piece. Non-speech tags become real NSC laughter, coughs and breaths from the sound library, a cloned stand-in
  ("haha" for `[laugh]`), or a synthetic sound.
- Each utterance is one dry WAV, trimmed to at most 50 ms of silence at each end.
- Retries from stage 5 re-synthesise only the failed turns with a new seed, plus every turn of a speaker whose
  reference was swapped.

### Stage 5: Voice validation

- Speaker similarity: an utterance passes only if both speaker-verification models score it above their own
  thresholds against the speaker's reference clip. Utterances under 1 s, backchannels and turns without words skip
  this check, because a hum carries too little of the voice.
- Within-speaker consistency: the mean pairwise cosine of a speaker's utterances of at least 3 s, in both models.
  A speaker with fewer than 3 such utterances skips the check.
- Between-speaker separability: the mean cosine over every utterance pair across two speakers, in both models. A
  pair is too close when either model is above its threshold.
- ASR round trip: the primary ASR transcribes every utterance with the language fixed to English, and the error is
  computed after Singlish normalisation (WER, or CER for turns of 3 words or fewer). Fillers are dropped from both
  sides; particles count.
- Quality: UTMOS must be at least 1.0 and no more than 0.75 below the MOS of the speaker's own reference clip.
- A failing utterance is re-synthesised with a new seed, up to 3 attempts. A speaker that fails consistency, or
  the speaker with fewer turns in a too-close pair, gets a reference swap: another clip of the same NSC speaker
  first (at most one per speaker), then another speaker of the same split whose gender and ethnicity match (at
  most two per meeting). An utterance that still fails is kept in the audio and listed in `kept_failed`
  (`voice.on_exhausted: keep_flagged`), so one bad utterance does not fail a whole meeting.

### Stage 6: Conversation assembly

- Place utterances on a timeline rather than concatenating them.
- Each main turn follows the floor with the transition its interaction type asks for: a sampled gap or long pause,
  an overlap lead, or an interruption that cuts off the floor speaker (`truncated_at`).
- Backchannels go inside other speakers' turns, and side conversations run alongside the main conversation. Nobody
  overlaps their own speech.
- The meeting's overlap target is drawn once. Stage 6 then scales the overlap leads (up to 3 times, never more than
  half of either turn), and if that is not enough, lets more normal transitions overlap, until the realised overlap
  share meets the target.
- When the synthesised meeting is shorter than planned, backchannels are thinned so the realised rate stays at the
  planned one.
- Output `timeline.json` and one dry, time-aligned track per speaker.

### Stage 7: Acoustic simulation

- Build the room in pyroomacoustics from `meeting.json`, with image sources in small rooms and image sources plus
  ray tracing in large rooms.
- Calibrate absorption so the talkers' median mid-band RT60 matches the sampled target, over up to 20 talker
  positions. The stage fails when the measured RT60 is more than 15% off target.
- Use source directivity, and turn each speaker's facing direction between turns while keeping seat positions
  fixed. Presenters often face the screen or the audience rather than the mic, and their small position drift is
  applied per turn.
- Set levels at the source and never normalise them, so wall-seated speakers and presenters stay quieter and more
  reverberant at the table mic, and the laptop's near talker stays loud.
- For an array mic, simulate every capsule and derive the single channel as their mean.
- Add noise: HVAC and projector fan beds at the sampled SNR, plus keyboard typing and events (doors, chairs, cups,
  paper) relative to the speech level.
- Apply the device chain: band limit, AGC, clipping, then the Opus codec on the single channel. Remote participants
  first pass through their own band limit and codec, and hybrid meetings get echo-cancellation artifacts.

### Stage 8: Labels and packaging

- Generate RTTM from the timeline.
- Force-align each utterance on its dry audio, not the reverberant mix, for word-level timestamps.
- Write the transcript in the Singlish transcription convention below.
- Write the meeting's manifest entry with every sampled parameter, quality score and tag.
- Fail the meeting unless the RTTM speech time equals the timeline, every aligned word lies inside its segment
  (give or take 50 ms), and alignment covers at least 90% of each utterance's words. Aligned times on clips under
  1 s are clamped to the clip first, because the aligner can place a short "mm" past its end.

## Singlish transcription convention

Synthetics defines one fixed transcription convention, and the fine-tuning application adopts it for training and
evaluation. It follows how [MERaLiON-3-3B-ASR](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) writes its output.
Stage 2 writes scripts in it, and stage 8 writes transcripts in it.

Each utterance stores two fields: `text` (cased, punctuated, for reading) and `text_norm` (lowercase, no
punctuation, markup removed; used for WER and training).

| Element | Rule | Example (`text`) |
| --- | --- | --- |
| Discourse particles | Canonical spelling in parentheses; variants mapped (la → lah, lo → lor) | ok (lah), we go first |
| Particle set | (lah) (leh) (lor) (meh) (hor) (ah) (sia) (mah) (liao) (wor) | can or not (meh)? |
| Fillers | Closed set in parentheses: (uh) (um) (mm); erm → um | (um) I think so |
| Mandarin | Simplified Chinese characters | then 他会教 |
| Tamil | Tamil script |  |
| Malay | Latin script, standard spelling | we go makan first |
| Singlish loanwords | Romanised Singapore spelling from the lexicon | very shiok, paiseh (ah) |
| Full Hokkien or Cantonese clauses | Chinese characters |  |
| Acronyms | Uppercase, no dots, one token | the HDB flat |
| Numbers | Written as spoken words | twenty twenty six |
| False starts | Fragment with trailing hyphen | we wan- we want |
| Cut off by interruption | Trailing hyphen | I think we should- |
| Non-speech | Square-bracket tags, excluded from WER: \[laugh\] \[cough\] \[breath\] \[noise\] | \[laugh\] true |

Rules in practice:

- The canonical lexicon (particles, fillers, loanword spellings, TTS pronunciation hints) lives in
  `backend/synthetics/config/defaults/singlish_lexicon.yaml`.
- TTS input is derived from `text` by removing parentheses and replacing non-speech tags with sound-library clips.
- `text_norm` keeps particle and filler words but drops the parentheses, so they count in WER.
- NSC writes particles in square brackets and names as `#name#`. The reference loader converts NSC notation to
  this convention, and clips with `#name#` redactions are never used as references (PDPA).

## System architecture

The unit of work is a meeting: each has an ID (`m-000001`) and a seed, is processed independently, and is fully
reproducible from its seed and the run config.

Design principles:

- **Uniform stage contract.** A stage declares what it reads and writes, its backend roles and its version, reads
  earlier stages' files, writes its own files plus a status record, and does nothing else. The runner enforces the
  contract, and stages can be re-run or switched off independently.
- **Caching.** A stage is skipped when its status holds a pass with the same cache key: a hash of the stage id and
  version, the stage's slice of the config, the hashes of its inputs, any pending feedback, and the run revision.
  Changing render settings re-runs only stages 7 and 8, and changing a sampled acoustic distribution re-runs stages
  1, 7 and 8, never the LLM and TTS stages.
- **Run revisions.** A run's config is frozen when the run is created. A behaviour change that would alter old
  runs' draws or cache keys is gated on `RunConfig.revision`, so old runs keep their results and stay cache hits.
  New settings get schema defaults that reproduce the old behaviour, and the YAML defaults turn them on for new
  runs.
- **Switchable components.** Stages 3, 5 and 7 can be switched off, and the switches are recorded in the manifest
  (`stages_enabled`), so the fine-tuning team can run ablations. A switched-off stage is recorded as skipped and
  later stages go on without its output.
- **Fixed split assignment.** A pool of 175 NSC Part 3 speakers has fixed splits (50 train, 25 dev, 100 test) in
  `data/nsc_part3_reference_speakers.csv`. Each run copies them into `speaker_splits.json`, and a meeting only
  clones voices from its own split, so splits are speaker-disjoint within and across runs. Partners who recorded
  together are always in the same split.
- **Typed config and artifacts.** Pydantic models validate the run config and every file in the run directory.
  Every distribution is a small typed object (uniform, log-normal, Bernoulli, categorical, Zipf) set in YAML.

Storage layout:

```
runs/<run_id>/
  config.yaml                  # frozen run config
  speaker_splits.json          # NSC speaker -> train/dev/test
  meetings/<meeting_id>/
    meeting.json               # stage 1
    script.json                # stage 2
    prompts/<speaker_id>.txt   # stage 2 (persona prompt, verbatim)
    script_validation.json     # stage 3
    utterances/*.wav           # stage 4 (dry)
    synthesis.json             # stage 4 (voice per speaker, per-utterance record)
    voice_validation.json      # stage 5
    timeline.json              # stage 6
    tracks/<speaker_id>.wav    # stage 6 (dry, time-aligned)
    mixture.wav                # stage 7 (mixture_mc.wav too for array mics)
    render.json                # stage 7 (what the simulation measured and applied)
    labels/meeting.rttm        # stage 8
    labels/transcript.json     # stage 8
    labels/manifest_entry.json # stage 8
    status.json                # per-stage status and timings
  manifest.jsonl               # one line per completed meeting
```

Code structure (`backend/synthetics/`):

```
config/        # schemas, distributions, loader; defaults/*.yaml
artifacts/     # one Pydantic model per run file, path helpers, atomic writers
stages/        # one Stage class per stage, the runner (cache keys, status), the stage 2 turn plan and prompts
validators/    # rule checks, LLM judges, voice checks, bake-off, SV and judge calibration
backends/      # role protocols, aicore clients, fakes, record and replay
acoustics/     # room, scene (poses per segment), noise, device chain, geometry
orchestrator/  # CLI, scheduler, GPU lease, phase C and post loops, status, run report, background runs
export/        # RTTM, transcript and manifest writers
text/          # Singlish convention, numbers, PDPA stripping
nsc/           # NSC pool, reference extraction and cleanup, non-lexical clips
voice/         # audio operations and the non-lexical sound library
viewer/        # the viewer's API, run builder and data contract check
```

The `backends/` layer keeps every model behind a role interface that calls a platform service through `aicore`, so
swapping a model is a change to `config/defaults/backends.yaml`.

## Model stack

Stages that understand or produce language use Singapore-tuned models where possible, and the judges use a
different architecture from the generator.

| Stage | Model | Service | Role |
| --- | --- | --- | --- |
| 2 | [Qwen-SEA-LION-v4.5-27B-IT](https://huggingface.co/aisingapore/Qwen-SEA-LION-v4.5-27B-IT), NVFP4A16 quantised locally | `qwen-sea-lion-v4-5-27b-it` | Script generator |
| 3 | [Gemma-SEA-LION-v4-27B-IT-NVFP4](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4-27B-IT-NVFP4) | `gemma-sea-lion-v4-27b-it-nvfp4` | Primary judge |
| 3 | [Apertus-SEA-LION-v4-8B-IT](https://huggingface.co/collections/aisingapore/sea-lion-v4) | `apertus-sea-lion-v4-8b-it` | Tie-breaker judge, borderline scripts only |
| 4 | VoxCPM2 2B, `ultimate` mode, cfg 2.0, 10 steps | `voxcpm2` | Voice cloning TTS |
| 5 | WavLM-Large + ECAPA-TDNN | `wavlm-sv` | Speaker verification |
| 5 | ReDimNet2 | `redimnet2-sv` | Speaker verification |
| 5 | Qwen3-ASR-1.7B | `qwen3-asr-1-7b` | Primary ASR round trip |
| 5 | MERaLiON-3-3B-ASR | `meralion-3-3b-asr` | Singlish ASR fallback, not leased by default |
| 5 | UTMOS22 | `utmos` | Quality score (CPU) |
| 8 | Qwen3-ForcedAligner 0.6B | `qwen3-forced-aligner` | Word-level timestamps |
| prep | DeepFilterNet3, then WPE, ranked by DNSMOS | `deepfilternet3`, `wpe-dereverb`, `dnsmos` | Reference cleanup (CPU) |

The generator is NVFP4A16 (weight-only FP4) because AI Singapore publishes no 4-bit Qwen-SEA-LION, and the W4A4
path needs activation calibration on a loaded 52 GB model, which does not fit in this machine's RAM.

## Voice models and thresholds

The stage 4 and 5 models and thresholds come from a bake-off on 30 NSC speakers (`just bakeoff`) and an SV
calibration on 875 clean NSC clips of 175 speakers (`just calibrate-sv`), which writes `config/sv_thresholds.yaml`.

- **TTS mode.** `ultimate` gives the best speaker similarity at equal ASR error and MOS. `reference` mode, without
  prompt continuation, loses about 0.1 similarity.
- **SV thresholds.** Each model's pass mark (`thresholds.sv`) and pair limit (`thresholds.sv_pair_max`) is its
  equal error rate threshold on same-speaker against different-speaker pairs. The consistency minimum
  (`thresholds.sv_consistency_min`) is the 5th percentile of same-speaker pairs. These thresholds were fitted on
  single clips of 5 to 15 s, which is why consistency uses only utterances of 3 s or more, and why the pair check
  compares the mean cosine over utterance pairs rather than speaker centroids, which read higher.
- **MOS floor relative to the reference.** UTMOS rates the real NSC references from about 1.3 to 3.3, and unevenly
  by gender and ethnicity, so a fixed floor would reject whole voices. A clone's MOS tracks its reference's, so
  stage 5 requires a small absolute floor (`thresholds.mos_min`) plus a limit on the drop below the reference
  (`thresholds.mos_drop_max`).
- **ASR limits.** `wer_max` is 0.35 and `cer_max_short` 0.40. The worst clone errors are notation differences
  (for example "S E S" against "SCS"), which re-synthesis cannot fix, so a tighter limit would only exhaust
  attempts. The ASR runs with the language fixed to English (`voice.asr_language`), because with detection on it
  heard one-word turns as Chinese or Japanese. The scorer maps the ASR's spellings ("Yeah", "Okay") to the
  script's ("ya", "ok") on both sides.
- **No Singlish ASR in phase C.** With MERaLiON loaded, phase C had about 1 GB of VRAM headroom, and the bake-off
  never needed it. An utterance that fails the primary ASR is re-synthesised instead, so `asr_limited` is always
  zero. Adding `asr_singlish` to phase C's roles in `config/defaults/phases.yaml` turns the fallback on.
- **Length limits.** Both SV services cut input at 20 s, and the aligner rejects requests over 180 s, so stage 8
  aligns one utterance per request.

Known weak spots, from the reference runs: short sentences ending in a particle ("(hor)" reaches the TTS as
"haw"), backchannels such as "(mm) good." where the TTS fuses the hum into the next word, local names in short
sentences, and cut-off turns where the TTS drops the last word before the hyphen. These make up most of the
utterances kept after failing every attempt, about 1% of all utterances.

## GPU phases and orchestration

The pipeline runs stage by stage across all meetings of a run, so only one phase's models occupy the 32 GB GPU at
a time.

| Phase | Services loaded together | Retry handling |
| --- | --- | --- |
| A. Generation | Qwen-SEA-LION 27B | Regenerates scripts that failed phase B, in the next round |
| B. Judging | Gemma-SEA-LION 27B NVFP4 | Borderline scripts are deferred to B2 |
| B2. Tie-break | Apertus-SEA-LION 8B | Runs over deferred scripts only |
| C. Voice | VoxCPM2, Qwen3-ASR, WavLM SV, ReDimNet2 SV, UTMOS | Re-synthesis and reference swaps within the phase |
| D. Alignment | Qwen3-ForcedAligner | None |
| prep | DeepFilterNet3, WPE, DNSMOS | Only for NSC reference cleanup, not part of a run |

Phases A and B alternate in rounds, for up to 3 rounds, because the generator and the judge do not fit together.
The two judges do not fit together either, so the tie-breaker is its own sub-phase, B2.

Orchestration:

- **Uniform service API.** Every model is a platform service with one contract per kind of model: LLMs and ASR are
  served by vLLM on the OpenAI API; TTS, speaker embeddings, MOS and the aligner are custom services on the OIP v2
  inference contract. Synthetics calls them only through `aicore`, and never starts containers or loads a model in
  its own process.
- **GPU lease.** For each phase the orchestrator runs the platform's `tools/stack.py up` with the phase's services
  (it checks the VRAM budget and waits for health), runs the phase's stages over every pending meeting, then runs
  `stack.py down`. A phase with no pending meetings is never leased. `NO_EVICT=1` stops `stack.py` from stopping
  other projects' services. In fake and replay modes there is no lease.
- **Local weights.** Weights live in the platform's store, `$AI_ASSETS`, which every service mounts, so cold
  starts don't download and other apps can reuse the models.
- **Execution.** One meeting and one request at a time, with `status.json` deciding what is left. Correctness comes
  before speed, and the stage interfaces keep any future concurrency confined to the scheduler and the backend
  clients. The 20-meeting reference run `m9-e2e` took about 9 hours, the largest share in phase C, and `resume` makes
  stopping at any point safe.

## Output contract

The fine-tuning application is the customer of Synthetics, so the output formats are fixed and versioned.

| Output | Format | Notes |
| --- | --- | --- |
| Audio | WAV, 16 kHz (configurable) | Multi-channel for array mics, plus a derived single-channel version |
| Speaker turns | RTTM | From the assembled timeline |
| Transcript | JSON, word-level with start and end times | Forced-aligned on dry audio; Singlish convention applied |
| Splits | train, dev or test per meeting | Speaker-disjoint, assigned by the generator |
| Manifest | JSONL, one line per meeting | Every sampled parameter, enabled stages, quality scores, tags |

Manifest tags include room class, speaker count, mic type, hybrid or not, RT60 and SNR, the number of wall-seated
speakers and presenters, each speaker's position type (table, wall, presenter or remote), judge scores and rounds,
and the utterances kept after failing every check (`voice.kept_failed`).

Intermediate files (dry utterances, timeline, dry tracks) are kept, so acoustic settings can be re-run without
regenerating scripts or audio. `docs/OUTPUT_CONTRACT.md` describes every file and field.

## Known limitations

| Limitation | Effect | How the design handles it |
| --- | --- | --- |
| Synthetic turn-taking may be too clean | Data improves diarisation less on real meetings | Explicit overlap, interruption and backchannel simulation in stage 6; every statistic configurable from real data |
| Cloned voices may sound alike | Label noise in 20-speaker meetings | Between-speaker separability check with two embedders, and reference swaps |
| The SEA-LION judges may share biases | Judges miss the same flaws | Accepted; the judge pass mark can be fitted to human ratings |
| The validation embedder may match the fine-tuning team's diarisation embedder | Circular validation | Two SV models of different architectures, with equal standing |
| NSC references carry room colour | Double reverberation after simulation | References are denoised and dereverberated before cloning |
| Some planned interruptions cannot be placed | A meeting can land a little under its interruption target | The realised rate over many meetings stays inside the configured range |
