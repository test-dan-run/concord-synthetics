# Synthetics

This file is the starting point for anyone, person or coding agent, who works on Synthetics. It states what the
application is for and what its design commits to, in one place. The detailed documents it summarises are listed
under "Where the detail lives", and they win when they disagree with this file.

Last updated Oct 21, 2026, after milestone M13.

## Purpose

Synthetics generates synthetic Singapore meeting recordings with full labels, for meetings of 2 to 20 speakers.
Real Singapore meeting recordings are rare, and recordings with more than 15 speakers are rarer still, so current
transcription and diarisation models do poorly on them. Collecting real data at that scale is slow and costly, so
Synthetics makes the training data instead.

A separate fine-tuning application consumes the output. Synthetics only generates data: it never trains or
evaluates models, and it never collects or labels real recordings. Because it never measures real data, every
distribution it samples from (turn-taking, overlap, acoustics, noise) is a configurable input with defaults, and
the fine-tuning team can replace the defaults with measured values by editing YAML.

The goals are:

- Realistic meetings in small rooms (2 to 6 speakers, laptop mic) and large rooms (7 to 20 speakers, centre-table
  mic), with the hard parts of real meetings: overlapping speech, short turns, backchannels, interruptions and a
  long tail of people who rarely speak.
- Singapore speech: Singlish discourse particles, code-switching into Mandarin, Malay and Tamil, local accents, and
  local names and acronyms.
- Labels that match the audio exactly: speaker turns (RTTM), word timestamps and normalised transcripts.
- Speaker-disjoint train, dev and test splits that stay the same across runs.
- Everything runs on one machine with one RTX 5090 (32 GB), using the models on the platform at
  `/home/user/code/dev/platform`.

The project is for research use. Every model is used under a licence that permits research, and the National
Speech Corpus (NSC) data used for reference voices is already on disk.

## The pipeline

Each meeting is one unit of work, with an id (`m-000001`) and a seed, and it can be reproduced from its seed and
the run config alone. Each meeting goes through eight stages. Each stage reads the files of earlier stages, writes
its own files and a status record, and does nothing else.

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

Stage 1 samples the room class, speaker count, room, table and seats, presenters, wall seats, mic, acoustics,
device chain and meeting dynamics. It also assigns each person an NSC reference speaker from the meeting's split,
and then samples the persona (gender, ethnicity, first language, code-switching) to match that speaker, so the
persona and the voice agree. People sit at the table, on wall chairs in large rooms (when the table is full, or by
choice), or stand at the front as presenters. Hybrid meetings add remote participants heard through a speakerphone.

Stage 2 first plans the turns (who speaks, and with which interaction type), so the turn shares and the overlap,
interruption and backchannel counts are met by construction. The generator LLM then writes the words in chunks of
structured JSON. Every persona prompt is saved verbatim. Stage 3 runs rule checks and then an LLM judge, which
writes its rationale before its scores. A score near the pass mark goes to a second judge, and a failed script goes
back to stage 2 with the judge's feedback, for up to 3 rounds before the meeting is dropped.

Stage 4 clones every utterance from one clean NSC clip of the assigned speaker, with real NSC laughter and coughs
for non-speech tags. Stage 5 checks every utterance with two speaker-verification models (both must pass), an ASR
round trip, and a quality score. It also checks each speaker's consistency and that no two speakers in a meeting
sound alike. A failed utterance is re-synthesised up to 3 times, and a close pair of voices gets a different clip or
speaker. An utterance that fails every attempt is kept and listed (about 1% of utterances).

Stage 6 places the utterances on a timeline with sampled gaps, overlaps, interruptions (the interrupted speaker is
cut off), backchannels inside other turns and side conversations, and it scales overlaps to reach the meeting's
overlap target. Stage 7 simulates the room in pyroomacoustics (image sources, plus ray tracing in large rooms),
calibrates absorption to the sampled RT60, turns talkers between turns, moves presenters, adds noise at the sampled
SNR, and applies the device chain (AGC, band limit, Opus codec, clipping, echo-cancellation artifacts). Levels are
set at the source and never normalised, so far talkers stay quiet and reverberant. Stage 8 writes the RTTM from the
timeline, force-aligns words on the dry speaker tracks, and writes the transcript and manifest entry.

## Design features

The design commits to the following, and code changes should keep them.

- Uniform stage contract. A stage declares what it reads and writes, its backend roles and its version, and the
  runner enforces it. Stages can be re-run or switched off independently.
- Caching. A stage is skipped when its status holds a pass with the same cache key: a hash of the stage id and
  version, the stage's slice of the config, the hashes of its inputs, any pending feedback, and the run revision.
  Changing render settings re-runs stages 7 and 8 only. In revision 2 runs, changing a sampled acoustic
  distribution re-runs stages 1, 7 and 8 only, never the LLM and TTS stages.
- Run revisions. A run's config is frozen when the run is created. Behaviour changes that would alter old runs'
  draws or cache keys are gated on `RunConfig.revision` (1 for runs made before Oct 21, 2026, and 2 for new runs),
  so old runs keep their results and stay cache hits.
- Ablations. Stages 3, 5 and 7 can be switched off. The switches are recorded in the manifest
  (`stages_enabled`), and later stages go on without the missing output.
- Quality gates. Stage 3 gates scripts and stage 5 gates voices. Stage 8 fails a meeting unless the RTTM speech
  time equals the timeline, every aligned word falls inside its segment, and alignment covers at least 90% of
  each utterance's words.
- Fixed splits. A pool of 175 NSC Part 3 speakers has fixed splits (50 train, 25 dev, 100 test) in
  `data/nsc_part3_reference_speakers.csv`. Each run copies them into `speaker_splits.json`, and a meeting only
  clones voices from its own split, so splits are speaker-disjoint within and across runs. Partners who recorded
  together are always in the same split.
- One transcription convention. Each utterance has `text` (cased and punctuated, particles and fillers in
  parentheses such as "ok (lah)", non-speech as tags such as "[laugh]") and `text_norm` (lowercase, no markup,
  used for WER and training). It follows how MERaLiON-3 ASR writes Singlish. The lexicon is
  `config/defaults/singlish_lexicon.yaml`.
- Typed config and artifacts. Pydantic models validate the run config and every file in the run directory. Every
  distribution is a small typed object (uniform, log-normal, Bernoulli, categorical, Zipf) set in YAML.
- Models as platform services. Every model is a service on the platform, called through the `aicore` client, and
  each model role maps to a service name in `config/defaults/backends.yaml`, so swapping a model is a config
  change. Synthetics never starts containers or loads a model in process.
- GPU phases. Only one phase's models fit on the GPU at a time, so the pipeline runs stage by stage across all
  meetings of a run. For each phase the orchestrator leases the services through the platform's `stack.py up` and
  releases them with `stack.py down`. Phases A (generator) and B (judge) alternate in rounds, and B2 (the
  tie-break judge) runs alone because the two judges do not fit together.
- Correctness before speed. Execution is sequential: one meeting and one request at a time. A 20-meeting run
  takes about 9 hours, and `resume` makes it safe to stop at any point. Speed-ups wait for the performance backlog
  in `IMPLEMENTATION_PLAN.md`, and they should only touch the scheduler and the backend clients.
- Fixed output contract. The fine-tuning application depends on the output files, so they carry
  `schema_version` and are described field by field in `docs/OUTPUT_CONTRACT.md`. Intermediate files (dry
  utterances, timeline, dry tracks) are kept, so the acoustics can be re-run without new scripts or audio.

## The Room Config Viewer

The viewer is a web UI over `runs/`, for the pipeline team and the fine-tuning team. Its room viewer has three
screens: a meeting browser with a mini floor plan per meeting, a meeting detail screen (floor plan and side
elevation to scale, pipeline status, the selected speaker's facts, the persona prompt beside the turns it
produced, and a speakers table), and the run's sampled distributions against the configured ranges. Its run
builder (M14) has two more: New dataset, a form over the common settings plus a YAML overlay
with a live stage 1 preview, and Pipeline, which starts, stops and follows a run.

- The room viewer never writes to a run. Its only action is exporting a slice of the manifest.
- The run builder writes only what `init-run` writes (the frozen config, splits and meeting folders) and starts
  the same `run` as the CLI in a background process, which records itself in `process.json` and `process.log`. It
  never edits a frozen config. One run at a time, and a run on real backends asks for confirmation first.
  `synthetics viewer --read-only` leaves the builder out.
- The API (`backend/synthetics/viewer/api.py`) lists runs and serves whitelisted files; the builder's routes are
  in `viewer/builder.py`. Every derived number (3D mic distances, facing, histogram bins) is computed in the
  browser.
- The UI (`frontend/`) is React, Vite and Tailwind on the AeroAudio design system
  (`/home/user/code/dev/DESIGN_SYSTEM.md`). Room classes use the categorical colours `cat-1` and `cat-2`, and
  speakers are never coloured, because meetings reach 20 speakers. Position types differ by shape.
- `synthetics viewer-contract -r <run>` checks that a run has every field the viewer reads.

## Repository layout

```
apps/Synthetics/
  AGENTS.md, README.md, DESIGN_DOCUMENT.md, UI_DESIGN_DOCUMENT.md, IMPLEMENTATION_PLAN.md
  docs/                 HANDOFF (current state), OUTPUT_CONTRACT, M5_EXIT, VOICE_BAKEOFF, PARALLEL_WORK (history)
  backend/synthetics/
    config/             schemas, distributions, loader; defaults/*.yaml
    artifacts/          one Pydantic model per run file, path helpers, atomic writers
    stages/             s1_sample.py to s8_labels.py, the runner, the stage 2 turn plan and prompts
    validators/         script rules, judges, voice checks, bake-off, SV and judge calibration, reports
    backends/           role protocols, aicore clients, fakes, record and replay
    acoustics/          room, scene (poses per segment), noise, device chain, geometry
    text/               Singlish convention, numbers, PDPA stripping
    nsc/                voice pool, reference extraction and cleanup, non-lexical clips
    voice/              audio operations and the non-lexical sound library
    orchestrator/       CLI, scheduler, GPU lease, phase C and post loops, status, run report, background runs
    export/             RTTM, transcript and manifest writers
    viewer/             data contract check, the API, and the run builder's routes
  frontend/             the viewer UI
  data/                 the pinned NSC pool CSV and the scripts that rebuild it
  assets/               icon and UI mock-ups
  tests/                pytest suite (no GPU needed)
  runs/                 run directories (not in git)
```

## Working on the code

Commands (from `apps/Synthetics`):

```bash
uv sync                          # Python 3.12 environment
just test                        # pytest, no GPU or platform needed
just lint                        # ruff
just viewer-test                 # frontend tests (vitest)
just init <run> <n> <seed>       # create runs/<run> with a frozen config
just run <run>                   # all phases; `synthetics resume -r <run>` continues after a stop
just status <run>                # state counts per stage
just viewer                      # the viewer on http://127.0.0.1:8610 (after viewer-setup and viewer-build)
```

Rules to follow:

- Keep `just test`, `just lint` and `just viewer-test` passing. Test new behaviour against the fake backends
  (`--set backends.mode=fake` on `init-run` gives a run with no GPU).
- Never edit a run's frozen `config.yaml`, and do not change old runs' results. Gate behaviour changes that would
  alter draws or cache keys on `RunConfig.revision`. Bumping a stage's `version` re-runs that stage on every old
  run that is resumed, so only do it when the outputs should change.
- Leave `runs/m3-10` untouched. `runs/m9-e2e` and `runs/m5-exit` are the reference real runs.
- Weights live in the platform's store, `$AI_ASSETS`, and datasets in `/home/user/code/datasets/`. Nothing large is
  downloaded into this app.
- GPU work needs the platform services. Ask before starting a GPU phase or a new real run, since they take hours
  and other projects share the machine. Watch RAM and `/tmp`.
- When the code departs from `DESIGN_DOCUMENT.md`, record the change in its "Changes since the first draft" table
  and in `IMPLEMENTATION_PLAN.md`. When an output file changes, update `docs/OUTPUT_CONTRACT.md`. When the
  project state changes, update `docs/HANDOFF.md`.
- Documents are written in plain English: simple words, complete sentences, no dashes.

## Where the detail lives

| Document | What it holds |
| --- | --- |
| `docs/HANDOFF.md` | The current state, decisions waiting for the owner, and open items |
| `DESIGN_DOCUMENT.md` | The pipeline design, sampling tables, model stack and risks, with every departure listed |
| `UI_DESIGN_DOCUMENT.md` | The viewer's screens, visual design and data contract |
| `IMPLEMENTATION_PLAN.md` | How it is built: interfaces, services, data, milestones with results, testing |
| `docs/OUTPUT_CONTRACT.md` | Every output file and field the fine-tuning application reads |
| `docs/VOICE_BAKEOFF.md`, `docs/M5_EXIT.md` | The phase C model choice, thresholds, and what the first real runs found |
| `README.md` | Setup, commands and milestone checks |
