# Synthetics — Synthetic Singapore Meeting Audio Pipeline Design

Oct 18, 2026 · @Dan

## Background and goals

Synthetics is a pipeline that generates synthetic Singapore meeting recordings with full labels, targeting meetings of up to 20 speakers. Real Singapore meeting datasets are rare, and datasets with more than 15 speakers are rarer still. As a result, current transcription and diarisation models perform poorly in this domain, and collecting real data at scale is slow and costly.

Goals:

- Generate realistic meetings from 2 to 20 speakers, in small rooms recorded on a laptop mic and large rooms recorded on a centre-table mic.
- Capture what makes real meetings hard: overlapping speech, short turns, backchannels, interruptions, and a long tail of rarely-speaking participants.
- Reflect Singapore speech: Singlish discourse particles, code-switching, local accents, and local named entities and acronyms.
- Produce labels that match the audio exactly: speaker turns (RTTM), word-level timestamps and normalised transcripts.
- Run on a single RTX 5090 (32GB VRAM) with all services in Docker.

## Scope and non-goals

Synthetics is a data generator only. It outputs labelled audio plus metadata, and a separate fine-tuning application consumes it.

In scope:

- Meeting configuration sampling, script generation, voice synthesis, conversation assembly, acoustic simulation and label generation.
- Quality gates that guarantee audio and labels agree.
- Speaker-disjoint train, dev and test splits.

Out of scope, owned by the fine-tuning application:

- Collecting and labelling a real evaluation set.
- Model training, fine-tuning, evaluation and ablation studies.

Because the generator never measures real data itself, every distribution it samples from (turn-taking, overlap rates, acoustics, noise) is a configurable input with defaults. The fine-tuning team can later replace those defaults with statistics measured on real recordings, without code changes.

Synthetics is built for research use. Every model in the stack is used under a licence that permits research, and the National Speech Corpus (NSC) speaker subset used for voice references is already downloaded locally for research use, so no new data collection is needed to start.

## Pipeline overview

Each meeting passes through eight stages, grouped into four GPU phases plus CPU work.

&#91;embedded content: pipeline stages and the GPU phase each runs in\]

Stages 3 and 5 send failures back for regeneration; stages 1, 6 and 7 need no GPU and can run alongside any GPU phase.

## Meeting config rules

Every meeting's room and setup is randomly sampled under two rules, with the class boundary fixed at 7 speakers: small rooms use a laptop mic with fewer than 7 people, and large rooms use a centre-table mic with up to 20 people. Large rooms can also seat overflow attendees on chairs along the walls, and either room class can have 1 or 2 presenters standing at the front.

| Parameter | Small room | Large room |
| --- | --- | --- |
| Speakers | 2–6 | 7–20 |
| Mic | Laptop mic | Centre-table mic (single mic or small array) |
| Length × width | \~3–5 m × 2.5–4 m | \~6–12 m × 4–7 m |
| Ceiling height | \~2.6–3.0 m | \~2.7–3.5 m |
| Table | Small rectangular or round | Long rectangular or boardroom oval |

Sampling order:

1. Pick the room class from a configurable ratio (for example 30% small, 70% large, weighted toward the large-meeting gap).
2. Sample the speaker count from the class range, using a configurable distribution.
3. Sample room dimensions and a table, and check the fit: the table seats about one person per 0.6–0.75 m of table edge, with about 1 m of clearance to each wall. Small rooms resample until everyone fits at the table. Large rooms may seat fewer people at the table than attend; the overflow goes to wall seats (step 4).
4. Place each person in one of three positions:
   - Table seats: around the table edge, mouths at about 1.1–1.2 m height with small jitter.
   - Wall seats (large rooms only): supplementary chairs in a row 0.3–0.6 m from a side or back wall, mouths at about 1.1–1.2 m height. These take the overflow once the table is full, and with a small probability a few attendees choose them even when table seats are free.
   - Presenter spots (both room classes, optional): 1 or 2 people standing near the front wall by the screen, mouths at about 1.5–1.75 m height, 0.5–1.5 m from the wall. Presenters count toward the speaker total.

   Seated positions stay fixed for the whole meeting. A presenter may drift within about 1 m of their spot between turns, and returns to a table or wall seat after presenting if the script says so.
5. Place the mic:
   - Laptop: on the table edge, 0.3–0.5 m in front of one random seat, 0.1–0.25 m above the table.
   - Centre mic: at the table centre with a small offset, about 0.05 m above the surface.
6. Sample acoustics (wall materials, RT60, noise types and SNR) from per-class ranges.
7. Sample meeting dynamics: per-speaker turn shares (skewed, Zipf-like), whether the meeting has a presentation segment (and its presenters), whether the meeting is hybrid (remote participants through a speakerphone), and topic.

The result is written to `meeting.json`, which every later stage reads.

## Sampling statistics

Stages 1, 6 and 7 sample from the distributions below. They are the baseline for the first runs, set from general knowledge of meeting corpora rather than measured on Singapore recordings, and will be tuned through experiments. All live in `config/` and are recorded per meeting in the manifest, so any run can be traced back to the values it used.

Turn-taking (stages 1 and 6):

| Parameter | Default |
| --- | --- |
| Turn shares | Zipf distribution, exponent 1.0; the chair takes the top share |
| Gap between turns | 70% of transitions: log-normal gap, median 0.3 s, capped at 2 s |
| Overlapping transitions | 50% of transitions: next turn starts 0.3–2.0 s before the previous ends, at most half its length (was 30% and 0.1–1.0 s until Oct 21, 2026; stage 6 then had to triple most leads to reach the overlap target) |
| Overall overlap target | 10–20% of speech time (small rooms), 15–25% (large rooms) |
| Interruptions | 5–10% of turn changes; interrupted speaker cut off 0.3–1.5 s after the interruption starts |
| Backchannels | 1–3 per minute, from non-speakers, placed inside the current turn |
| Long pauses | 2% of transitions: 1–3 s |

Seating and presenters (stage 1):

| Parameter | Small room | Large room |
| --- | --- | --- |
| Wall seating | None | Overflow when table capacity is below the speaker count; otherwise p=0.15 that 1–3 attendees sit at the wall anyway |
| Wall-seat placement | — | 0.3–0.6 m from a side or back wall, chairs 0.5–0.7 m apart |
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

Each stage reads the previous stage's artifact, writes its own, and records a pass, fail or retry status.

### Stage 2: Script generation

- Assign a persona to each seat: role, seniority, verbosity, accent and L1 background, code-switching tendency. Personas stay fixed for the meeting. Presenters get a presentation segment: longer monologue turns with slide references, followed by Q&A with questions and interruptions from the floor. Wall-seated attendees lean toward quieter personas with fewer turns.
- Generate a structured script, not prose. Each turn has a speaker ID, text, interaction type (normal, interruption, backchannel, overlap start, side conversation) and disfluency markers (false starts, repetitions, fillers).
- Target the turn-share distribution from `meeting.json`, so a few speakers dominate and many speak rarely.
- Include Singlish particles, code-switching and local entities and acronyms. Strip names and identifiers from any past-meeting context used in prompts (PDPA).

### Stage 3: Script validation

- Programmatic checks first: speaker count, turn-share distribution, overlap and backchannel rates, persona IDs, schema validity.
- Primary LLM judge scores what code can't: persona consistency, topical coherence, natural Singlish. The rubric explicitly rewards realistic messiness, not polish. The judge writes its rationale before its scores.
- Tie-breaker: if the primary judge's overall score lands within a borderline band (for example ±0.5 of the pass threshold on a 1–5 scale), a second judge scores independently with the same rubric. Agreement decides; disagreement fails the script.
- Failed scripts are regenerated in rounds with judge feedback, up to N attempts, then dropped.
- Calibrate the judges against 50–100 human-rated scripts, ideally rated by Singaporeans.

### Stage 4: Voice synthesis

- Assign each speaker one NSC reference speaker, drawn only from the meeting's split.
- Use conversational NSC parts for references, 5–15 s clips with their transcripts, denoised and dereverberated first so TTS output is dry.
- Use continuation-style cloning with the reference transcript, keeping the same reference for every utterance of a speaker within a meeting.
- Synthesize one dry WAV per utterance. Use a small library of real non-lexical sounds (laughter, coughs, "mm-hmm") where TTS output is poor.

### Stage 5: Voice validation

- Speaker similarity: an utterance passes only if both speaker-verification models score it above their own thresholds, each calibrated on real NSC same-speaker vs. different-speaker pairs.
- Within-speaker consistency across all of a speaker's utterances.
- Between-speaker separability: flag any two speakers in a meeting whose clone centroids are too close in either model, and swap one reference.
- ASR round-trip, in two steps. The primary ASR transcribes every utterance and WER is computed after Singlish normalisation. Failures go to a Singlish-aware ASR. If it passes, the utterance passes and is tagged "ASR-limited". If both fail, it is a real TTS failure and is re-synthesized with a new seed, up to N attempts.
- Use a looser threshold or character error rate for very short utterances such as backchannels.
- Quality score (UTMOS or DNSMOS) drops artifact-heavy clips.

### Stage 6: Conversation assembly

- Place utterances on a timeline rather than concatenating them.
- Sample gaps, overlaps and interruption points from configurable turn-taking statistics.
- Insert backchannels inside other speakers' turns. Truncate the first speaker's utterance on interruptions.
- Output `timeline.json` and one dry, time-aligned track per speaker.

### Stage 7: Acoustic simulation

- Build the room in pyroomacoustics from `meeting.json`. Use the hybrid image-source plus ray-tracing mode for large rooms.
- Sample fairly reflective materials, since glass walls are common in Singapore meeting rooms.
- Use source directivity, and vary each speaker's facing direction between turns while keeping seat positions fixed. Presenters often face the screen or the audience rather than the mic, and their small position drift is applied per turn. Wall-seated speakers sit further from the centre mic, with more reverberant, lower-level pickup; keep that effect rather than normalising it away.
- Laptop mic: keep the near-far imbalance (one speaker very close), and add keyboard noise near the laptop. Centre mic: simulate an array and derive single-channel output.
- Add noise: HVAC by default, plus projector fans, paper, chairs, cups and doors, at sampled SNRs.
- Apply a device chain: AGC, band-limiting, Opus or VoIP codecs, occasional clipping, and echo-cancellation artifacts for hybrid meetings.

### Stage 8: Labels and packaging

- Generate RTTM from the timeline.
- Run forced alignment on each dry speaker track, not the reverberant mix, for word-level timestamps.
- Write transcripts in the Singlish transcription convention below.
- Write the meeting's manifest entry with every sampled parameter, quality score and tag.

## Singlish transcription convention

Synthetics defines one fixed transcription convention, and the fine-tuning application adopts it for training and evaluation. It follows how [MERaLiON-3-3B-ASR](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) writes its output, so the stage 5 check compares like with like. Stage 2 writes scripts in it, and stage 8 writes transcripts in it.

Each utterance stores two fields: `text` (cased, punctuated, for reading) and `text_norm` (lowercase, no punctuation, markup removed; used for WER and training).

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

- The canonical lexicon (particles, fillers, loanword spellings, TTS pronunciation hints) lives in `config/singlish_lexicon.yaml` and is versioned with each run.
- TTS input is derived from `text` by removing parentheses and replacing non-speech tags with clips from the sound library.
- `text_norm` keeps particle and filler words but drops the parentheses, so they count in WER.

## System architecture

The unit of work is a meeting: each has an ID and a seed, is processed independently, and is fully reproducible from its seed and the run config.

Design principles:

- **Uniform stage contract.** A stage reads the previous artifact, writes its own artifact plus a status record, and does nothing else. Stages can be re-run, swapped or disabled independently.
- **Caching.** A stage is skipped when its output already exists for the same config. Changing acoustic settings re-runs only stages 7–8, not the expensive LLM and TTS stages.
- **Switchable components.** Every stage has an on/off flag and parameter overrides, recorded in the meeting's metadata, so the fine-tuning team can run ablations.
- **Global split assignment.** NSC speakers are assigned to train, dev or test once per run, before generation, which guarantees speaker-disjoint splits.
- **Typed config.** Pydantic models loaded from YAML validate the global config and `meeting.json` at sampling time.

Storage layout:

```
runs/<run_id>/
  config.yaml                  # frozen global config for the run
  speaker_splits.json          # NSC speaker -> train/dev/test
  meetings/<meeting_id>/
    meeting.json               # stage 1
    script.json                # stage 2
    script_validation.json     # stage 3
    utterances/*.wav           # stage 4 (dry)
    voice_validation.json      # stage 5
    timeline.json              # stage 6
    tracks/<speaker_id>.wav    # stage 6 (dry, time-aligned)
    mixture.wav                # stage 7
    labels/meeting.rttm        # stage 8
    labels/transcript.json     # stage 8
    status.json                # per-stage status and timings
  manifest.jsonl               # one line per completed meeting
```

Code structure:

```
synthetics/
  config/        # schemas + default YAMLs (room classes, acoustics, timing stats)
  stages/        # one module per stage, each exposing run(meeting_dir, cfg)
  validators/    # rule checks, LLM judges, embedding / ASR / quality scorers
  backends/      # pluggable LLM, TTS, ASR, embedding, aligner clients
  acoustics/     # pyroomacoustics room builder, noise library, device chain
  orchestrator/  # phase scheduler, GPU lease, caching, retries, status
  export/        # RTTM / transcript / manifest writers
```

The `backends/` layer keeps every model behind a client interface that calls a Docker service, so swapping a model is a config change.

## Model stack

All stages that understand or produce language use Singapore-tuned models where possible, and the judges use a different architecture from the generator.

| Stage | Model | Role | Notes |
| --- | --- | --- | --- |
| 2 | [Qwen-SEA-LION-v4.5-27B-IT](https://huggingface.co/aisingapore/Qwen-SEA-LION-v4.5-27B-IT) (4-bit) | Script generator | Fine-tuned from Qwen3.6-27B. Test with thinking on and off. |
| 3 | [Gemma-SEA-LION-v4-27B-IT-NVFP4](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4-27B-IT-NVFP4) | Primary judge | Gemma 3 base, 128K context, reported <0.5% loss vs. full precision. Gemma Terms of Use. |
| 3 | [Apertus-SEA-LION-v4-8B-IT](https://huggingface.co/collections/aisingapore/sea-lion-v4) | Tie-breaker judge | Third architecture, used only in the borderline band. |
| 4 | VoxCPM2 | Voice cloning TTS | 2B, Apache-2.0, continuation-based cloning for high speaker similarity. |
| 5 | WavLM-based speaker verification | Speaker similarity, separability | Both models must pass. |
| 5 | ReDimNet2 | Speaker similarity, separability | Thresholds calibrated separately per model. |
| 5 | Qwen3-ASR-1.7B | Primary ASR round-trip | Same family as the forced aligner. |
| 5 | MERaLiON-3-3B-ASR | Singlish-aware ASR fallback | Open weights; served via its own vLLM-backed container. Outputs particles in parentheses and Mandarin in Chinese characters, matching our convention. |
| 5 | UTMOS / DNSMOS | Quality score | Can run on CPU. |
| 8 | Qwen3-ForcedAligner | Word-level timestamps | Runs on dry speaker tracks. |
| — | DeepFilterNet (or similar) | Reference cleanup | Denoise and dereverb NSC references; CPU. |

No published benchmark covers Singlish or Singapore accents for these TTS and ASR models. Before committing, run a bake-off on a few hundred NSC references measuring speaker similarity, between-speaker separability and Singlish WER.

## GPU phases and orchestration

The pipeline runs stage by stage across a batch of meetings, so only one phase's models occupy the 32GB GPU at a time and each model loads once per batch.

| Phase | Services loaded together | Approx. VRAM | Retry handling |
| --- | --- | --- | --- |
| A. Generation | Qwen-SEA-LION 27B | \~17GB + KV cache | Regenerates failures from phase B in the next round |
| B. Judging | Gemma-SEA-LION 27B NVFP4 + Apertus-SEA-LION 8B | Fits in 32GB | Tie-breaker runs inline |
| C. Voice | VoxCPM2 + Qwen3-ASR + WavLM SV + ReDimNet2 + quality model + MERaLiON-3-3B-ASR | \~12–20GB | Re-synthesis inline |
| D. Alignment | Qwen3-ForcedAligner | \~2–4GB | — |
| CPU | Stages 1, 6, 7 | 0 | Overlap any GPU phase |

MERaLiON-3-3B-ASR is small enough to co-load in phase C. If VRAM gets tight, run it as a sub-phase at the end of phase C over all flagged utterances, with re-synthesis in the next round. Phases A and B alternate in rounds, because two 27B models at 4-bit don't fit together.

Orchestration:

- **Uniform service API.** LLMs are served through vLLM's OpenAI-compatible server. TTS, ASR, embedders and the aligner run in thin FastAPI containers with consistent endpoints: `/synthesize`, `/transcribe`, `/embed`, `/align`.
- **GPU lease.** The orchestrator owns the GPU. For each phase it starts the containers, waits for health checks, drains the batch queue, then stops them before the next phase.
- **Batch size.** Model swaps cost roughly 30–90 s, so size batches for tens of minutes per phase, for example a few hundred meetings.
- **Local weights.** Mount model weights from a local volume so cold starts don't download.
- **Blackwell compatibility.** Build images on a CUDA 12.8+ base with recent PyTorch. Research TTS repos often pin older versions, so test every container on the 5090 early.
- **Execution.** Start with a single-machine runner that loops over meetings and checks `status.json`. Move to a job queue or Ray later without changing stage code.

## Output contract

The fine-tuning application is the customer of Synthetics, so these formats are fixed and versioned.

| Output | Format | Notes |
| --- | --- | --- |
| Audio | WAV, 16 kHz (configurable) | Multi-channel for array mics, plus a derived single-channel version |
| Speaker turns | RTTM | From the assembled timeline |
| Transcript | JSON, word-level with start/end times | Forced-aligned on dry tracks; Singlish normalisation convention applied |
| Splits | train / dev / test per meeting | Speaker-disjoint, assigned by the generator |
| Manifest | JSONL, one line per meeting | Every sampled parameter, enabled components, quality scores, tags |

Manifest tags include:

- Room class, speaker count, mic type, hybrid or not, RT60 and SNR, number of wall-seated speakers and presenters, and each speaker's position type (table, wall or presenter), for slicing results.
- Per-utterance "ASR-limited" flags, marking audio where the primary ASR failed but the Singlish-aware ASR passed. These are useful for weighting or building a hard evaluation subset.
- Judge scores and the number of regeneration rounds.

Intermediate artifacts (dry utterances, timeline, dry tracks) are kept, so acoustic settings can be re-run without regenerating scripts or audio.

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Synthetic turn-taking is too clean | Data doesn't improve diarisation on real meetings | Explicit overlap and backchannel simulation in stage 6; statistics configurable from real data |
| Cloned voices collapse toward each other | Label noise in 20-speaker meetings | Between-speaker separability check with two embedders; reference swapping |
| TTS mispronounces Singlish particles and local names | Unnatural speech, ASR check failures | Bake-off on NSC references; phonetic respelling; possible TTS fine-tuning on NSC |
| Shared biases across SEA-LION judges | Judges miss the same flaws | Accepted; human-rated calibration set for the judges |
| Validation embedder matches the diarisation model's embedder | Circular validation | Check the fine-tuning team's diarisation embedder; weight the other model more |
| Double reverb from NSC reference colouration | Unrealistic acoustics | Denoise and dereverb references before cloning |
| Containers incompatible with Blackwell | Setup delays | Test every image on the 5090 early, on CUDA 12.8+ |
