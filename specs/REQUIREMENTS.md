# Synthetics requirements

Synthetics generates synthetic Singapore meeting recordings with exact labels, so that a separate fine-tuning
application can train and test transcription and diarisation models on meetings of 2 to 20 speakers. Real Singapore
meeting recordings are rare, and recordings with more than 15 speakers are rarer still, so current models do poorly
on these meetings and collecting real data at scale is slow and costly.

The document lists who uses Synthetics, what it must do and how to tell when each part is done. A change that no
user story below asks for is out of scope until the user agrees to it and a story is added here.
`specs/ARCHITECTURE.md` describes how the system meets these requirements.

## Users

- The fine-tuning team trains and evaluates models on the output. They need fixed file formats, exact labels and
  speaker splits that never share a speaker.
- The pipeline team builds and tunes the generator. They need to inspect every meeting and check each run against
  its settings.
- A dataset requester knows which meetings they want but not how the pipeline works. They describe the meetings
  in plain English, listen to samples and export the result.
- A pool admin adds NSC voices to the speaker pool.

## Goals

- Generate realistic meetings of 2 to 20 speakers, in small rooms recorded on a laptop mic and large rooms
  recorded on a table mic.
- Include what makes real meetings hard for models, e.g., overlapping speech, short turns, backchannels,
  interruptions and a long tail of people who rarely speak.
- Reflect Singapore speech, with Singlish particles, code switching into Mandarin, Malay and Tamil, local accents,
  and local names and acronyms.
- Produce labels that match the audio exactly, and run on one RTX 5090 (32 GB) with every model served by the
  platform in `../../platform`.

## Out of scope

- Collecting or labelling real recordings, and any training, fine-tuning, evaluation or ablation study. The
  fine-tuning application owns those.
- Editing an existing run's settings, running single stages again from the viewer, and editing labels.
- Uploading new recordings of real people to the speaker pool. The admin page imports curated NSC voice folders
  only.

## Dataset generation

### G1. Generate a labelled run

As the pipeline team, I want to create a run of N meetings from a seed and settings, so that I can generate a
dataset with one command.

Acceptance criteria:

- `just init <run> <n> <seed>` writes a frozen `config.yaml`, the speaker splits and one folder per meeting, and
  `just run <run>` takes every meeting through all eight stages.
- A meeting that passes stage 8 has a recording, an RTTM file, a word level transcript and a manifest line, at the
  paths in "Output contract" in `specs/ARCHITECTURE.md`.
- A run on fake backends finishes with no GPU and the same files as a real run.

### G2. Reproduce and resume

As the pipeline team, I want a run to be reproducible and safe to stop, so that a crash or a stop costs no work.

Acceptance criteria:

- The same seed and `config.yaml` give the same meetings, because every random draw is seeded per run, meeting,
  stage and attempt.
- `synthetics resume -r <run>` continues from `status.json`, and a stage that already passed with the same cache
  key is skipped.
- Old runs keep their results when the code changes. A resumed old run with none of the new settings stays fully
  cached.

### G3. Sample meetings by rule

As the fine-tuning team, I want each meeting's room, people, mic and acoustics drawn from configurable
distributions, so that I can match them to statistics from real recordings by editing YAML.

Acceptance criteria:

- Small rooms have 2 to 6 speakers and a laptop mic, and large rooms have 7 to 20 speakers and a table mic.
- Large rooms can seat people on wall chairs, either class can have 1 or 2 presenters, and a meeting can be hybrid
  with remote callers.
- Every distribution is a setting in `backend/synthetics/config/defaults/`, and the manifest records the values
  each meeting drew.
- `just sampler-report 1000` compares every sampled rate with its target and fails on a miss.

### G4. Write and judge conversations

As the fine-tuning team, I want each meeting's conversation to sound like a real Singapore meeting, so that models
learn from realistic speech.

Acceptance criteria:

- Stage 2 plans the turns (who speaks, and with which interaction) before it writes the words, and the plan meets
  the meeting's turn shares and turn-taking rates.
- Stage 3 checks the script in code and with an LLM judge, and sends a failed script back to stage 2 with the
  judge's feedback for up to 3 rounds.
- Every persona prompt is saved word for word, and names and identifiers are stripped from any feedback that
  reaches a prompt (PDPA).

### G5. Clone and check voices

As the fine-tuning team, I want each speaker to keep one clear, distinct voice, so that the speaker labels hold.

Acceptance criteria:

- Each speaker is cloned from one cleaned reference clip of an NSC speaker from the meeting's own split.
- Stage 5 checks speaker similarity with two speaker verification models, the ASR round trip and the predicted
  quality, and it re-synthesises a failing utterance or swaps a reference.
- An utterance that fails every attempt stays in the audio and is listed in `kept_failed` and in the manifest.

### G6. Assemble and simulate the room

As the fine-tuning team, I want overlapping speech and real room conditions, so that the data is as hard as real
meetings.

Acceptance criteria:

- Stage 6 places turns on a timeline with gaps, overlaps, interruptions that cut the floor speaker off,
  backchannels and side conversations, and nobody overlaps their own speech.
- The realised overlap share of each meeting meets its sampled target, or the meeting's status says it was out of
  reach.
- Stage 7 simulates the room, mic, noise and device chain, matches the sampled RT60 within 15% and never normalises
  levels.
- New runs also write each speaker as heard at the mic and the noise alone, and those files add up to the mixture
  before clipping and the codec.

### G7. Exact labels

As the fine-tuning team, I want labels that agree with the audio, so that training and scoring are not noisy.

Acceptance criteria:

- Stage 8 fails a meeting unless the RTTM speech time equals the timeline, every aligned word lies inside its
  segment within 50 ms, and alignment covers at least 90% of each utterance's words.
- Transcripts follow the Singlish transcription convention in `specs/ARCHITECTURE.md`, with a readable `text` and
  a normalised `text_norm` per utterance.
- Every output file carries `schema_version`, and a rename, removal or change of meaning raises it.

### G8. Speaker splits with no shared speaker

As the fine-tuning team, I want train, dev and test to share no speaker, within a run and across runs, so that test
scores are honest.

Acceptance criteria:

- Every NSC speaker has one split for good in a pinned pool CSV, and people who recorded together share a split.
- A meeting clones voices only from its own split, and `synthetics report` checks that the splits are disjoint.
- NSC clips with `#name#` redactions are never used.

### G9. Switch stages off for ablations

As the fine-tuning team, I want to switch stages 3, 5 and 7 off, so that I can measure what each one adds.

Acceptance criteria:

- A switched off stage is recorded as skipped, and later stages go on without its output.
- The manifest records the switches in `stages_enabled`.

## Guided flow in the viewer

### V1. Describe the dataset in plain English

As a dataset requester, I want to describe the meetings in my own words, so that I don't need to learn the
settings.

Acceptance criteria:

- The Describe step takes a free text description, a list of topics, key terms with an optional pronunciation,
  and the amount of audio in hours.
- "Suggest topics for me" picks topics from `config/defaults/topic_bank.yaml`.
- "Make a plan" turns the description into settings and creates nothing.

### V2. Read and adjust the plan

As a dataset requester, I want to see what will be generated and change it by chatting, so that I can fix the plan
before any GPU time is spent.

Acceptance criteria:

- The plan shows the room mix, people, callers, presentations, length, topics, GPU time and three example
  meetings with floor plans.
- Each chat message changes settings only through the decision questions. Applied changes come back as chips with
  Undo, and options the model is unsure of come back as buttons.
- Stage switches and backends always need the user's OK, and the viewer says when it falls back to word matching.

### V3. Listen to sample meetings

As a dataset requester, I want to hear three sample meetings before the full run, so that I can catch a bad voice
or setting early.

Acceptance criteria:

- "Make sample meetings" creates a pilot run of 3 meetings and starts it, after a GPU confirmation for real
  backends.
- Each sample shows who is in the room, the recording and every line with a play button.
- A message that names a speaker's voice swaps that voice in that sample only. Any other change applies to every
  meeting and offers to remake the samples.
- "Looks good" creates and starts the full run from the same settings. When the settings did not change after the
  pilot, the pilot's meetings become the full run's first meetings, and only stage 8 runs again for them.

### V4. Review meetings as they finish

As a dataset requester, I want to review meetings while the run goes on, so that I don't wait hours for the first
result.

Acceptance criteria:

- New runs run in waves of 10 meetings, and each wave finishes all eight stages before the next starts.
- The review screen shows progress per wave, Pause and Resume, the recording with speaker lanes and the lines
  around the playhead.
- Keep or Leave out is saved to `review.json`, which the pipeline never reads.

### V5. Export a pack

As a dataset requester, I want the kept meetings in a folder of plain files, so that I can use them without the
pipeline.

Acceptance criteria:

- Every WAV in a pack is 16 bit mono PCM at 16 kHz, and every meeting has its recording, RTTM, transcript and
  meeting details.
- Each speaker's own audio is optional, as clips with a CSV or at full length, and clean or as heard in the room.
  The noise alone is optional too.
- The Export screen shows the size of every part before the export starts, and the export's progress while it
  runs.

## Engineer screens in the viewer

### E1. Browse a run's meetings

As the pipeline team, I want to filter a run's meetings and spot failures, so that I can find the meetings to
inspect.

Acceptance criteria:

- The browser shows a card per meeting with a floor plan drawn to scale, the speaker count, RT60, SNR and status.
- Filters for room class, split, minimum speakers, callers, presenters and wall seats apply immediately.

### E2. Inspect one meeting

As the pipeline team, I want to check a meeting's geometry and personas in under a minute, so that I can trace a
problem to its cause.

Acceptance criteria:

- The detail screen shows the floor plan and side view to scale, each stage's status, and a fact panel for the
  room, mic, surfaces, noise and device chain.
- Selecting a speaker shows their persona prompt word for word next to the turns it produced, and their position,
  distance to the mic and voice check scores.

### E3. Compare a run with its settings

As the pipeline team, I want each sampled distribution drawn against its configured range, so that I can see
sampler drift.

Acceptance criteria:

- The Distributions screen shows the room class mix, presentations, hybrid meetings, wall seats, splits, RT60, SNR,
  mic distances, overlap, backchannels and interruptions, each against `config.yaml`.

### E4. Create and run a dataset from a form

As the pipeline team, I want every setting in a form with a live preview, so that I can build a run without editing
YAML by hand.

Acceptance criteria:

- The New dataset screen starts from the defaults or from an existing run, and it accepts a YAML overlay for any
  other setting.
- The preview runs stage 1 in memory and shows the meetings the run will sample, and validation errors appear under
  their field.
- The Pipeline screen starts, stops and resumes a run and shows the stage counts and the log. Only one run can run
  at a time.
- `synthetics viewer --read-only` leaves out every screen and route that creates, starts or marks runs.

## Speaker pool admin

### A1. Grow the speaker pool

As a pool admin, I want to import a curated voice folder and publish a larger pool, so that runs have more voices.

Acceptance criteria:

- The admin page is available only with `synthetics viewer --admin`.
- An import checks each new speaker, i.e., at least 3 clips at quality 2.5 or above, sheet attributes present, no
  speaker already in the pool, and partners in the same split.
- The server assigns every new speaker a split for good, and admins never choose splits. Publishing writes a new
  pool file and a sidecar, and older runs keep their pool.

## Non-functional requirements

- Hardware
  - The pipeline runs on one 32 GB GPU, with one GPU phase's models loaded at a time, and the decision model runs on
    the CPU.
  - The GPU is shared with other projects, so a lease never stops another project's services.
- Time
  - A real meeting takes about 10 minutes of model work, plus about 12 minutes of model loading per wave.
  - A fake meeting takes about 10 seconds.
- Licences to carry into the dataset card
  - MERaLiON-3 requires an acknowledgement under the MERaLiON Public Licence, and the Gemma terms apply.
  - WavLM-Large ECAPA is CC BY-SA 3.0 and is used only to check Synthetics' own audio.
  - The ESC-50 noise clips are CC BY-NC (non-commercial use only), and the DEMAND room recordings are CC BY-SA 3.0.
  - The IMDA licence for the NSC covers the reference voices and the non-speech clips cut from them.
