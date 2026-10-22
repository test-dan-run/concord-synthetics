# Synthetics implementation plan

Oct 18, 2026, updated Oct 21, 2026 after milestone M13.

This document describes how Synthetics is built. It started as the build plan, and it now records the build as it
stands: the interfaces, the platform services, the data, and each milestone with its result. `AGENTS.md` has the
purpose and the design commitments in short form, `DESIGN_DOCUMENT.md` has the pipeline design, and
`UI_DESIGN_DOCUMENT.md` has the viewer design. `docs/HANDOFF.md` has the current state and the open decisions.

Status: milestones M0 to M13 are done. The pipeline runs all eight stages end to end on the platform's models, the
20-meeting M9 run passed every stage, and the Room Config Viewer reads any run. What is left is the performance
backlog at the end of §5 and the open items in `docs/HANDOFF.md`.

The build order was backend interfaces first, then the pipeline stages behind them, then the platform services,
and the viewer last. Correctness came before speed throughout. Execution is sequential, with no concurrency, no
throughput tuning and no profiling, and the stage and backend interfaces keep later speed-ups inside the scheduler
and the backend clients.

---

## 1. Platform integration

Every model is a platform service. Each model has a manifest in `platform/services/<name>.yaml`, and Synthetics
calls services only through `aicore.client(name)`. It never starts Docker containers and never loads a model in
its own process. The platform handles compose, health checks, VRAM budgets and contract tests, and other apps can
use the same services.

- Weights are in `$AI_ASSETS` (`/home/user/code/ai-assets`), the platform's weights store, which every service
  mounts.
  - vLLM engine services pull into `$AI_ASSETS/hf-cache`. That directory is owned by root, so engine weights only
    arrive through `just up` on the platform.
  - Custom services use `just fetch-weights <name>`, which writes to `$AI_ASSETS/custom-models/<name>/<version>`.
  - Checkpoints made locally go in `$AI_ASSETS/<dir>`, and the manifest uses `model: /models/<dir>`. The
    Qwen-SEA-LION NVFP4A16 checkpoint is one of these.
  - Nothing is downloaded into `apps/Synthetics`, and no weights go in the `platform/` git tree.
- The GPU lease is `stack.py`. For each GPU phase, `orchestrator/lease.py` runs `platform/tools/stack.py up
  <services>` and later `stack.py down <services>`, as Concord does. `stack.py up` validates, checks the VRAM
  budget, starts GPU services one at a time, waits for health, and stops services that do not fit (unless
  `NO_EVICT=1`). In fake and replay modes the lease does nothing.
- Service APIs follow the platform contracts. LLMs and ASR use the OpenAI API (`chat.completions` and
  `audio.transcriptions`). TTS, speaker embeddings, quality scores, alignment and reference cleanup use the OIP v2
  inference contract. These replace the custom `/synthesize`, `/transcribe`, `/embed` and `/align` routes of the
  first design.

---

## 2. Repository layout

Tooling matches Concord: uv, Python 3.12, Pydantic v2, Typer, ruff, pytest, a Justfile, and `aicore[oip]` as an
editable path dependency. The frontend uses npm, React 19, Vite 8, Tailwind v4, TanStack Query, react-router 7
and vitest.

```
apps/Synthetics/
  pyproject.toml               # aicore = { path = "../../platform/clients/aicore", editable = true }
  Justfile                     # setup, test, lint, init, run, stage, post, status, nsc-*, judge-set, fit-judge,
                               # bakeoff, calibrate-sv, sampler-report, viewer, viewer-* recipes
  .env.example                 # SYNTHETICS_RUNS, SYNTHETICS_PORT, PLATFORM_DIR, PLATFORM_VENV, NSC_DIR,
                               # NSC_REFS_DIR, SOUND_LIBRARY_DIR, AI_HOST, NO_EVICT
  data/
    nsc_part3_reference_speakers.csv   # the pinned NSC voice pool with fixed splits (§4.3)
    selection/                 # four scripts that rebuild the CSV
    sound_library/             # build_noise_library.py (noise beds and events, §4.3)
  backend/synthetics/
    config/                    # schema.py, distributions.py, loader.py
      defaults/                # room_classes, seating, acoustics, turn_taking, personas, stages, backends,
                               # phases, singlish_lexicon (.yaml)
    artifacts/                 # meeting.py, models.py, voice.py, labels.py, paths.py, io.py
    stages/                    # base.py, runner.py, s1_sample.py to s8_labels.py, script_plan.py, prompts.py
    validators/                # script_rules, judges, voice_checks, bakeoff, sv_calibration, calibration
                               # (judge set), sampler_report, coverage
    backends/                  # base.py (protocols), aicore_impl.py, fakes.py, fakes_audio.py, registry.py,
                               # transport.py (record, replay, retries)
    acoustics/                 # geometry, room, scene, noise, device
    text/                      # singlish.py, numbers.py, pdpa.py
    nsc/                       # pool, extract, cleanup, references, nonlexical, settings
    voice/                     # audio_ops.py, nonlexical.py (stage 4 sound library)
    orchestrator/              # cli, scheduler, lease, phase_c, phase_post, status, run, run_report
    export/                    # rttm.py, transcript.py, manifest.py
    viewer/                    # contract.py (M10), api.py (M12)
  frontend/                    # the viewer UI (M13)
    src/app/                   # Shell (sidebar, run select, jobs card), routes
    src/components/            # ui, MetricTile, SortableTable, icons, plan drawing parts
    src/features/browser/      # MeetingBrowser, MiniPlan
    src/features/detail/       # MeetingDetail, FloorPlan, Elevation, SidePanels, PersonaPanel, SpeakersTable
    src/features/distributions/  # Distributions, Histogram
    src/lib/                   # api (types and hooks), colors, format, geometry, meetings, people, stats
  assets/                      # icon.svg, mockups/
  tests/                       # pytest suite, all on fakes
  runs/                        # not in git; SYNTHETICS_RUNS can point elsewhere
```

---

## 3. Backend interfaces

All interfaces are typed Python (Pydantic v2 models and `typing.Protocol`). They were written and tested with fake
backends before any model service existed, so every stage can be developed and tested without a GPU.

### 3.1 Configuration (`config/`)

```python
class RunConfig(_M):                   # frozen to runs/<run_id>/config.yaml; every model is frozen, extra="forbid"
    schema_version: Literal["1"] = "1"
    revision: int = 1                  # sampling and cache scheme; stages.yaml sets 2 for new runs (see below)
    run_id: str
    seed: int
    n_meetings: int
    sample_rate: int = 16000
    room_classes: RoomClassesConfig    # ratio, per-class speaker range and distribution, dims, ceiling, table
                                       # shapes, seat pitch 0.6 to 0.75 m, wall clearance, max_fit_resamples
    seating: SeatingConfig             # wall seating, presenters (count, position, drift, facing screen, return
                                       # to a seat), screen
    acoustics: AcousticsConfig         # per-class RT60, materials, SNR, noise events, keyboard, hybrid and remote
                                       # count, mic, device chain probabilities, orientation
    turn_taking: TurnTakingConfig      # Zipf exponent, gaps, overlap share and lead, interruptions,
                                       # backchannels, long pauses, per-class overlap targets
    dynamics: DynamicsConfig           # meeting types, topics, durations, turns per minute
    personas: PersonasConfig           # roles, seniority, verbosity, code-switching
    stages: dict[StageId, StageToggle] # enabled flags (ablations)
    script: ScriptConfig               # stage 2: template version, chunk size, context, temperature, thinking
    judge: JudgeConfig                 # stage 3: rubric version, temperature, max judged turns
    voice: VoiceConfig                 # stages 4 and 5: TTS mode and settings, swaps, SV and MOS minimum
                                       # lengths, consistency rules, ASR language, on_exhausted
    assembly: AssemblyConfig           # stage 6: interruption and backchannel placement, overlap control
    simulation: SimulationConfig       # stage 7: ISM order, ray tracing, materials, calibration, levels, AGC,
                                       # noise library, hybrid far-end path
    labels: LabelsConfig               # stage 8: aligner language
    backends: BackendsConfig           # mode (aicore, fake, replay, record) and role -> service name
    phases: PhasesConfig               # platform dir, no_evict, lease timeout, phase -> roles, batch size
    retries: RetryConfig               # script_rounds 3, tts_attempts 3
    thresholds: ThresholdsConfig       # judge pass mark and band, SV per model, WER, CER, MOS, script rules,
                                       # stage 8 alignment gates
    lexicon: LexiconRef                # the versioned Singlish lexicon
    speaker_pool: SpeakerPoolConfig    # pool CSV path and version, meeting split mix (70 / 15 / 15)

def load_run_config(*yaml_paths, overrides=None) -> RunConfig: ...   # defaults, then overlays, then --set
def config_hash(model) -> str: ...                                   # canonical JSON -> sha256[:12]
```

Each distribution is a small typed union with a `.sample(rng)` method: `Uniform(lo, hi)`, `IntUniform`,
`LogNormal(median, sigma, cap)`, `Bernoulli(p)`, `Categorical(weights)`, `Zipf(exponent)` and `Fixed(value)`. The
fine-tuning team can replace any default with a measured value by editing YAML. The frozen config records every
value a run used.

The run revision decides how a run samples and caches. Revision 1 is every run made before Oct 21, 2026, which is
what a frozen config without the field loads as. In revision 2, stage 1 draws the acoustic values of
`meeting.json` (`ACOUSTIC_FIELDS` in `artifacts/meeting.py`) from their own random stream, and stages 2 to 6 hash
`meeting.json` without them. A change to a sampled acoustic distribution then re-runs only stages 1, 7 and 8.
Any later change that would move old runs' draws or cache keys should be gated on the revision in the same way.

The two speaker-verification models have equal standing in `ThresholdsConfig`. Each has its own calibrated
thresholds (`sv`, `sv_pair_max`, `sv_consistency_min`), and neither score is weighted above the other. The design
warns about circular validation if one of them matches the fine-tuning team's diarisation embedder, so the equal
weighting stays until that embedder is known.

### 3.2 Artifacts (`artifacts/`): the data contract

There is one Pydantic model per file in the run directory, and each carries `schema_version`. Writers are atomic
(a temp file in the same directory, then a rename). These models are the contract for both the fine-tuning
application and the viewer. `docs/OUTPUT_CONTRACT.md` describes every field.

| File | Model | Main fields |
|---|---|---|
| `meeting.json` | `MeetingSpec` | room (dims, front wall, screen, per-wall materials, floor, ceiling, RT60 target), table (shape, size, height, seat pitch, slots), wall chairs, people (position type, wall reason, seat, mouth xyz, facing, presenter spot, drift and return seat, chair flag, turn share, NSC speaker with sheet attributes, persona), mic (type, position, channels, array geometry), speakerphone, acoustics, device chain, dynamics (meeting type, topic, duration, turn target, Zipf, chair, presentation segment, hybrid, remote speakers), `fit_resamples` |
| `script.json` | `Script` | brief; `turns[]` with `turn_id`, `speaker_id`, `interaction` (normal, interruption, backchannel, overlap_start, side, presentation), `text`, `text_norm`, disfluencies, non-speech tags; `generator` (service, template version, round, thinking, temperature) |
| `prompts/<speaker_id>.txt` | plain text | the rendered persona prompt from stage 2, verbatim |
| `script_validation.json` | `ScriptValidation` | rule results; primary judgement (rationale, overall, persona consistency per speaker); tie-break judgement; decision; round; feedback |
| `utterances/<turn_id>.wav` | audio | one dry utterance per script turn, mono, at the run rate, trimmed to at most 0.05 s of silence at each end |
| `synthesis.json` | `Synthesis` | voice per speaker (NSC speaker, clip id), one record per utterance (duration, seed, attempt, pieces) |
| `voice_validation.json` | `VoiceValidation` | per utterance: both SV scores, WER or CER, `asr_limited`, MOS, skipped checks, passed; per speaker: consistency; per pair: mean cross-speaker cosine in both models; reference swaps; `kept_failed`; decision |
| `timeline.json` | `Timeline` | `segments[]` (`turn_id`, `speaker_id`, `kind`, `start`, `end`, `truncated_at`); `stats` (overlap share, backchannels per minute, interruption share, counts); dropped turns |
| `tracks/<speaker_id>.wav` | audio | the dry, time-aligned track of each speaker |
| `mixture.wav`, `mixture_mc.wav` | audio | the recording; multi-channel as well when the mic is an array |
| `render.json` | `AcousticRender` | RT60 target and measured, SNR, channels, noise used, device chain applied |
| `labels/meeting.rttm` | RTTM | speaker turns from the timeline |
| `labels/transcript.json` | `Transcript` | per utterance `text`, `text_norm`, `words[]` with start, end and alignment score |
| `labels/manifest_entry.json` | `ManifestEntry` | the meeting's line of `manifest.jsonl` |
| `status.json` | `MeetingStatus` | per stage: state (pending, running, pass, fail, retry, skipped, dropped), attempts, start and end times, cache key, detail line, metrics, pending feedback |
| `speaker_splits.json` | `SpeakerSplits` | NSC speaker -> split, pool version, sheet metadata, counts per split |
| `manifest.jsonl` | `ManifestEntry` | one line per meeting whose stage 8 passed: every sampled value, `stages_enabled`, position counts, per-speaker split, position type and turn share, judge scores and rounds, voice figures and `kept_failed`, turn-taking stats, acoustic measurements, label figures, file paths |
| `run_report.json`, `voice_report.json`, `phase_seconds.jsonl` | run level | stage counts, rounds, drops, phase time, phase C rates, split disjointness |

`artifacts/paths.py` has one helper per path (`MeetingPaths(run_dir, meeting_id).script`, and so on), so no code
builds paths by hand. `manifest.jsonl` is always rebuilt from the per-meeting entries and never edited in place.

### 3.3 Stage contract (`stages/`)

```python
class StageId(IntEnum): SAMPLE=1; SCRIPT=2; SCRIPT_VAL=3; VOICE=4; VOICE_VAL=5; ASSEMBLE=6; ACOUSTIC=7; LABELS=8

class Stage(Protocol):
    id: StageId
    name: str
    phase: Phase                         # A, B, B2, C, D, CPU or prep
    roles: tuple[Role, ...]              # backend roles it needs (empty for CPU stages)
    reads: tuple[str, ...]               # artifact keys (MeetingPaths attributes, plus speaker_splits)
    writes: tuple[str, ...]
    version: str                         # bump when the code changes outputs; part of the cache key
    def config_slice(self, cfg: RunConfig) -> BaseModel: ...
    def run(self, ctx: StageContext) -> StageResult: ...

@dataclass
class StageContext:
    paths: MeetingPaths
    cfg: RunConfig
    rng: np.random.Generator             # SeedSequence([run seed, meeting seed, stage, attempt])
    backends: Backends                   # only the roles this stage declared
    feedback: str | None                 # judge or voice-check feedback from the previous round
    log: logging.Logger
    attempt: int = 1
    seed: int = 0                        # integer seed for backends, same derivation as rng

@dataclass
class StageResult:
    state: Literal["pass", "fail", "retry", "defer"]   # defer: needs a role leased in a later phase (stage 3, B2)
    detail: str                          # one line, shown on the viewer's stage tile
    metrics: dict[str, float | int | str]
    feedback: str | None = None          # sent to the upstream stage on retry
    retry_stage: StageId | None = None   # stage 3 -> 2, stage 5 -> 4
```

The runner (`stages/runner.py`) enforces these rules:

- A stage runs only when its required inputs exist. A disabled stage is marked `skipped`. Outputs of stages that
  can be switched off (`ABLATABLE_OUTPUTS`: script validation, voice validation, render and mixture) are only
  required when their stage is enabled.
- Cache key = sha256 of the stage id, `version`, the canonical `config_slice`, the input hashes, the pending
  feedback, and the revision when it is 2 or more. In revision 2 the stages in `ACOUSTICS_BLIND` (2 to 6) hash
  `meeting.json` without its acoustic fields. A stage is skipped when `status.json` holds a pass with the same
  key and its outputs exist.
- Seeds come from `np.random.SeedSequence`, so a meeting can be reproduced from its seed and the run config.
- An exception inside a stage is recorded as a failure with the error in the detail line. A pass whose declared
  outputs are missing is turned into a failure.
- A `retry` result marks the upstream stage `retry` with the feedback, and the scheduler runs it again.

Stage versions in the current code: stage 1 v1, stage 2 v4, stage 3 v3, stage 4 v1, stage 5 v5, stage 6 v1, stage 7
v1, stage 8 v2. Bumping a version re-runs that stage on every old run that is resumed.

### 3.4 Backend roles (`backends/`)

Each model role is a small Protocol in `backends/base.py`. Each role has an `Aicore*` implementation that calls the
platform service, and a `Fake*` implementation that is deterministic and needs no GPU. `transport.py` wraps every
request with retries and with record and replay (the key is a hash of the service, the canonical request and the
array hashes), as in Concord. `backends.yaml` maps each role to a service name, so swapping a model is a config
change.

```python
@dataclass(frozen=True)
class Audio: samples: np.ndarray; sr: int        # float32 mono; resample (soxr), save, load, wav_bytes

class ChatLLM(Protocol):
    def chat(self, messages: list[Msg], *, json_schema: dict | None = None, temperature: float, seed: int,
             max_tokens: int, thinking: bool | None = None) -> ChatResult: ...
    # vLLM structured outputs; whitespace in the JSON is disabled; thinking via chat_template_kwargs

class VoiceCloner(Protocol):
    def clone(self, text: str, ref: VoiceRef, *, seed: int) -> Audio: ...   # VoiceRef = clean clip + transcript

class Transcriber(Protocol):
    def transcribe(self, audio: Audio, *, language: str | None = None) -> str: ...

class SpeakerEmbedder(Protocol):
    def embed(self, clips: Sequence[Audio]) -> np.ndarray: ...             # (n, d), L2-normalised

class QualityScorer(Protocol):
    def score(self, clips: Sequence[Audio]) -> np.ndarray: ...             # MOS-like, (n,)

class Aligner(Protocol):
    def align(self, audio: Audio, text: str, *, language: str) -> list[WordTiming]: ...

class Enhancer(Protocol):
    def clean(self, audio: Audio) -> Audio: ...                            # denoise or dereverb
```

Roles: `generator`, `judge_primary`, `judge_tiebreak`, `tts`, `asr_primary`, `asr_singlish`, `sv_wavlm`,
`sv_redimnet`, `quality` (UTMOS on TTS output), `ref_quality` (DNSMOS on cleaned references), `aligner`,
`enhancer` and `dereverb`. `Backends.for_roles(roles)` builds clients lazily, and the scheduler checks every
configured role's endpoint at startup, so a missing manifest fails early.

The fakes are built so a fake run exercises every check. `FakeChatLLM` answers by the JSON schema's title (brief,
script chunk, judgement). `FakeVoiceCloner` renders a tone whose hidden chords encode the utterance and the voice,
so `FakeTranscriber` returns the cloned text and `FakeSpeakerEmbedder` separates speakers. Fake runs can use the
real NSC references, because the fake voice is keyed by the reference's clip id.

The Singlish normaliser (`text` to `text_norm`) and its lexicon are not a backend role. They live in `text/`,
because the lexicon is versioned with each run, and the platform's `text-normalizer` service does a different job.

### 3.5 Orchestrator (`orchestrator/`)

```python
class GpuLease:                          # context manager over stack.py up / down; LeaseError on refusal or timeout
class StatusStore:                       # per-meeting status.json, atomic updates under a file lock
class PhaseScheduler:
    def run(self, until=StageId.LABELS, only=None) -> RunReport:
        # 1. copy the fixed splits from the pool CSV, then stage 1 for every meeting (CPU)
        # 2. rounds: lease A -> stage 2 for pending meetings; lease B -> stage 3 with the primary judge;
        #    lease B2 -> stage 3 for meetings waiting on the tie-breaker; failed scripts go back to A with
        #    feedback; after retries.script_rounds rounds an unresolved script is dropped
        # 3. lease C -> stages 4 and 5 per meeting, re-synthesis and reference swaps inline (phase_c.py)
        # 4. stages 6 and 7 per meeting (CPU), then lease D -> stage 8, then manifest.jsonl (phase_post.py)
```

- Phase membership is in `config/defaults/phases.yaml` as lists of roles. The scheduler resolves them to service
  names and checks the VRAM budget of each phase set before any GPU work starts. Grouping work by phase is
  required, because only one phase's models fit on the GPU at a time.
- Everything runs sequentially: one meeting at a time and one request at a time, with no thread or process pools.
  Stages never manage concurrency, so a pool can be added later in the scheduler alone.
- The whole run is one batch. `batch_size` is kept in the config for later.
- A phase whose meetings are all fresh (cache hits) is not leased at all.
- Time per phase goes to `phase_seconds.jsonl` after each meeting (`Stopwatch`), so a killed run keeps it.
  `run_report.json` is written after `run`, `resume` and `post`.

CLI (`synthetics`, Typer). Every command is idempotent and reads `status.json` to decide what is left.

| Command | What it does |
|---|---|
| `init-run <id> --n --seed --config --set` | create `runs/<id>` with a frozen config, speaker splits and meeting folders |
| `run -r <id> [--until N] [--only PHASE] [--keep-services]` | all phases in order |
| `resume -r <id>` | continue after any stop |
| `post -r <id>` | stages 6 to 8 and the manifest only |
| `stage N -r <id> -m <meeting> [--force]` | one stage on one meeting, with services already up |
| `status`, `report`, `manifest`, `splits` | state counts, run report, rebuild the manifest, copy the splits |
| `voice-report -r <id>` | phase C rates (pass, first pass, re-synthesis, swaps, kept failing) |
| `sampler-report --n` | stage 1 rates against every configured target (the M1 check) |
| `coverage-seed --n` | find a run seed whose meetings cover the M9 list |
| `export-judge-set`, `fit-judge-threshold` | judge calibration through a CSV rating sheet |
| `bakeoff`, `calibrate-sv` | the M4 bake-off and the SV threshold fit |
| `nsc validate`, `nsc extract`, `nsc clean`, `nsc nonlexical`, `nsc report` | NSC data preparation (M2) |
| `viewer-contract -r <id>`, `viewer` | the viewer's data contract check and the viewer server |

---

## 4. Platform services and data

### 4.1 Procedure for each service

Each service was registered with the platform's `register-model` skill, which covers the contract choice, VRAM
sizing, the manifest, tests and validation. The steps are:

1. Write the manifest.
2. Run `just validate` and `just vram <name>`.
3. For an engine service, run `just up <name>` (weights pull into `$AI_ASSETS/hf-cache`). For a custom service,
   write a `weights.yaml` with pinned revisions and sha256 hashes, run `just fetch-weights <name>`, then
   `just build <name>`.
4. Run `just conform <name>`.
5. Record the measured VRAM in the manifest comment.

Every image is built on the platform's CUDA 12.8 or later bases and was tested on the 5090.

### 4.2 Service list

| Phase | Service | Model | Tier | Contract | Port | VRAM (measured) |
|---|---|---|---|---|---|---|
| A | `qwen-sea-lion-v4-5-27b-it` | Qwen-SEA-LION-v4.5-27B-IT, NVFP4A16 quantised locally | vLLM | openai, chat | 8206 | util 0.75 (17.9 GiB weights, about 22.9 GB on the GPU) |
| B | `gemma-sea-lion-v4-27b-it-nvfp4` | Gemma-SEA-LION-v4-27B-IT-NVFP4 | vLLM | openai, chat | 8207 | util 0.80 (16.6 GiB weights, KV at 32k) |
| B2 | `apertus-sea-lion-v4-8b-it` | Apertus-SEA-LION-v4-8B-IT | vLLM | openai, chat | 8208 | util 0.68 (15.0 GiB weights, 4.0 GiB KV) |
| C | `voxcpm2` | VoxCPM2 2B, torch.compile | custom | oip-v2 | 8410 | 6.6 GB loaded, 7.4 GB peak |
| C | `qwen3-asr-1-7b` | Qwen3-ASR-1.7B | vLLM | openai, transcriptions | 8106 | util 0.22 (about 7 GB) |
| C | `wavlm-sv` | WavLM-Large with ECAPA-TDNN (the Seed-TTS-eval model) | custom | oip-v2 | 8404 | 1.9 GB loaded, 2.5 GB peak |
| C | `redimnet2-sv` | ReDimNet2 b6 | custom | oip-v2 | 8405 | 0.75 GB loaded, 2.6 GB peak |
| C | `utmos` | UTMOS22 strong | custom, CPU | oip-v2 | 8406 | CPU |
| not leased | `meralion-3-3b-asr` | MERaLiON-3-3B-ASR | custom (transformers) | openai, transcriptions | 8107 | 7.2 GB loaded, 8.1 GB peak |
| D | `qwen3-forced-aligner` | Qwen3-ForcedAligner 0.6B | custom | oip-v2 | 8407 | 2.3 GB loaded, 3.7 GB peak |
| prep | `deepfilternet3` | DeepFilterNet3 (denoise) | custom, CPU | oip-v2 | 8408 | CPU |
| prep | `wpe-dereverb` | nara_wpe (dereverb, not a neural model) | custom utility, CPU | oip-v2 | 8502 | CPU |
| prep | `dnsmos` | DNSMOS (reference ranking) | custom, CPU | oip-v2 | 8409 | CPU |

Phase budgets as measured:

- Phase A runs the generator alone.
- Phase B runs the Gemma judge alone. Its weights alone are 19.5 GB on disk, so it cannot share the GPU with
  Apertus under `util_limit` 0.93, and the tie-breaker became its own sub-phase, B2, over the borderline scripts
  only. Stage 3 defers those scripts in B and finishes them in B2, reusing the primary judgement.
- Phase C loads VoxCPM2, Qwen3-ASR, both SV models and UTMOS, at 21.3 GB. With MERaLiON co-loaded the bake-off
  peaked at 31.27 GB, about 1 GB of headroom, so MERaLiON was taken out of phase C. Adding `asr_singlish` back to
  the phase turns the Singlish fallback on again.
- Phase D runs the aligner alone.

The non-OpenAI contracts are needed because cloning takes a reference audio array plus its transcript, which the
OpenAI `audio.speech` route cannot carry, and because the embedding, scoring, alignment and cleanup services all
take audio arrays. The OIP signatures (`-1` is a variable dimension):

```yaml
voxcpm2:                          # request parameters: mode, cfg_value, inference_timesteps (from `voice`)
  inputs:  [TEXT BYTES [1], REF_AUDIO FP32 [-1], REF_SAMPLE_RATE INT32 [1], REF_TEXT BYTES [1], SEED INT64 [1]]
  outputs: [AUDIO FP32 [-1], SAMPLE_RATE INT32 [1]]
wavlm-sv / redimnet2-sv:          # batch-shaped; clips are zero-padded and LENGTHS gives the true sizes
  inputs:  [AUDIO FP32 [-1, -1], LENGTHS INT32 [-1], SAMPLE_RATE INT32 [1]]
  outputs: [EMBEDDINGS FP32 [-1, -1]]
utmos / dnsmos:                   # dnsmos also returns SIG and BAK
  inputs:  [AUDIO FP32 [-1, -1], LENGTHS INT32 [-1], SAMPLE_RATE INT32 [1]]
  outputs: [MOS FP32 [-1]]
qwen3-forced-aligner:
  inputs:  [AUDIO FP32 [-1], SAMPLE_RATE INT32 [1], TEXT BYTES [1], LANGUAGE BYTES [1]]
  outputs: [WORDS BYTES [-1], TIMES FP32 [-1, 2], SCORES FP32 [-1]]
deepfilternet3 / wpe-dereverb:
  inputs:  [AUDIO FP32 [-1], SAMPLE_RATE INT32 [1]]
  outputs: [AUDIO FP32 [-1], SAMPLE_RATE INT32 [1]]
```

Notes:

- The platform's OIP encodes tensors as JSON, so payloads stay short. Stage 8 aligns each utterance window of the
  dry track separately (the timeline gives each window) and offsets the words back into the track, so a long
  track is never one request. The SV and MOS signatures accept batches, but the clients send one clip per
  request for now. The SV services cut input at 20 s, and the aligner caps requests at 180 s.
- AI Singapore publishes no 4-bit Qwen-SEA-LION. W4A4 NVFP4 needs activation calibration on the loaded 52 GB
  model, which does not fit in the machine's 31 GB of RAM, so the generator is the data-free weight-only NVFP4A16
  variant (`experiments/quantize-qwen-sea-lion-nvfp4`), stored in `$AI_ASSETS`.
- MERaLiON runs on a custom transformers server. Its model class is not in the pinned vLLM (v0.31.0), and the
  authors' plugin needs vLLM 0.16, so a second vLLM pin would have been needed.
- The first WavLM choice, `microsoft/wavlm-base-plus-sv`, scored different-speaker NSC pairs up to 0.95, above
  its lowest same-speaker pair, so it was replaced before the bake-off.
- Reference cleanup is DeepFilterNet3 then WPE. The bake-off found WPE good enough, so no neural dereverb model
  was added.

### 4.3 Data assets (not models)

Datasets stay in `/home/user/code/datasets/`, next to the NSC data. Nothing is copied into the app.

#### NSC Part 3 source (checked Oct 18, 2026)

- `NSC_PART3/SDM/` holds 406 per-speaker zips (`SPEAKER<8-digit id>.zip`, 51 GB in total). Each zip has
  `SPEAKER<id>/manifest.json` (JSON lines: `conf`, `pair_id`, `source_start`, `source_end`, `text`),
  `WAVE/<pair_id without "_">.wav` (16 kHz mono 16-bit, one file per segment), and `TEXT/*.txt`. Every manifest
  line has a matching WAV.
- `SPEAKER_INFO_PART3.XLSX` has two sheets. All 406 SDM speakers are in the "Separate Room" sheet (columns:
  Conference ID, Speaker ID, Gender, Age band, Ethnicity, Education, Occupation, Language Proficiency, First
  Language, Partner ID, Relationship). The "Same Room" sheet uses different IDs and matches no SDM zip, so it is
  not used.
- The 406 speakers are 228 F and 178 M; 239 Chinese, 89 Malay, 73 Indian and 5 Others; ages 1X to 6X.
- Part 3 is conversational speech, as the design asks. IVR (telephone) audio is not used. The curated
  `nsc-spontaneous` and `nsc-scripted` sets are only for the bake-off and threshold calibration.

#### The 175 selected speakers and their splits (`data/nsc_part3_reference_speakers.csv`)

Only those 175 speaker zips are ever read. Only the test split needs voice variety, so it gets most of the pool,
and train and dev are kept at the smallest size that still fills a 20-speaker meeting with spares for swaps.

| Split | Speakers | Shared voices between two 20-speaker meetings (expected, k²/N) |
|---|---|---|
| train | 50 | 8 of 20 |
| dev | 25 | 16 of 20 |
| test | 100 | 4 of 20 |

Pool selection rules (`data/selection/2_select_pool.py`):

1. Quotas by ethnicity and gender: Chinese 43 F and 43 M, Malay 22 F and 22 M, Indian 20 F and 20 M, and Others
   3 F and 2 M (all 5 in the dataset). Malay and Indian speakers are oversampled compared with the full set, so
   local accents are well covered.
2. Age mix within each cell: 8% 1X, 25% 2X, 25% 3X, 22% 4X, 12% 5X and 8% 6X, which leans toward working ages.
   Where a cell runs short, the slots go to the next band.
3. Ranking within each cell and age band: speakers with the most clean reference material come first. Clean
   material means segments of 5 to 15 s with at least 8 words and no `#name#` redactions, `<...>` tags, `**` or
   `~` marks.

Pool result: 88 F and 87 M; 86 Chinese, 44 Malay, 40 Indian and 5 Others; ages 15, 50, 46, 42, 12 and 10 (1X to
6X); first language 97 English, 37 Chinese, 27 Malay, 13 Tamil and 1 Hindi. Every speaker has at least 43 clean
candidate clips (4.4 min or more). The 175 zips total 24.7 GB.

Split rules (`data/selection/3_assign_splits.py`, a seeded local search):

- Partner pairs (40 pairs, both in the pool) are always in the same split, because partners recorded the same
  conversation and one speaker's audio can carry faint traces of the other's voice.
- All 5 Others speakers go to test, where variety matters. Train and dev personas never draw "Others".
- Every other ethnicity and gender group, and every age band, is split in proportion to split size.

| Split | Chinese F/M | Malay F/M | Indian F/M | Others F/M | Ages 1X to 6X |
|---|---|---|---|---|---|
| train (50) | 13 / 12 | 7 / 6 | 6 / 6 | 0 / 0 | 5 / 14 / 13 / 12 / 3 / 3 |
| dev (25) | 6 / 7 | 3 / 3 | 3 / 3 | 0 / 0 | 2 / 7 / 7 / 6 / 2 / 1 |
| test (100) | 24 / 24 | 12 / 13 | 11 / 11 | 3 / 2 | 8 / 29 / 26 / 24 / 7 / 6 |

Splits are fixed per pool, not drawn per run. The CSV's `split` column is authoritative, and each run copies it
into `speaker_splits.json`, so the test voices stay the same across runs and the fine-tuning team can compare test
results from one run to the next. Meetings are assigned to splits by `speaker_pool.split_mix` (70% train, 15% dev
and 15% test by default).

The CSV is the source of truth, and the pipeline reads it directly. The four scripts in `data/selection/`
(statistics, pool, splits, CSV writer) rebuild it from the spreadsheet and the zips, in that order. To change the
pool, edit the scripts, rebuild the CSV, and bump `speaker_pool.version` (now `p3-175-v1`).

#### Rules for using the pool

- A meeting clones only speakers of its own split. Stage 4 swaps a reference first to another clip of the same
  speaker (at most one clip swap per speaker), then to another speaker of the same split whose sheet row matches
  (gender and ethnicity always, first language and age band when possible), at most two speaker swaps per meeting.
- The NSC speaker is assigned in stage 1, and each persona's gender, ethnicity, first language and code-switching
  languages are sampled to match that speaker's sheet row, so persona and voice agree.
- NSC writes particles in square brackets (`[lah]`), fillers in parentheses (`(uh)`) and names as `#name#`. Our
  convention puts particles in parentheses and keeps square brackets for non-speech, so the reference-text loader
  converts the notation through the lexicon before the transcript is used as the cloning prompt. Clips with
  `#name#` are never used as references (PDPA).

#### Derived datasets

- `nsc-part3-refs/raw/<speaker>/`: candidate segments read in place from the 175 zips (WAV and converted
  transcript), with no zip unpacked in full.
- `nsc-part3-refs/clean/<speaker>/`: references cleaned with DeepFilterNet3 then WPE and ranked by DNSMOS. Only the
  12 best-ranked candidates per speaker (by length and markup) are cleaned, because cleaning all of about 26,000
  candidates would take about 13 hours on CPU (`--max-candidates 0` cleans them all). Each speaker keeps 5 clips,
  875 in all, and `clean/index.jsonl` lists them with transcripts, duration, sheet metadata, cleanup version and
  MOS. The set is written once and shared by every run.
- `sound-library/nonlexical/`: real laughter, coughs, breaths and noises cut from pool speakers' NSC segments whose
  transcript is only a non-speech code (`synthetics nsc nonlexical`). The clips are dereverberated but not
  denoised, and `index.yaml` records each clip's speaker, split and gender, so stage 4 uses a speaker's own clip
  when there is one and never a clip of another split's speaker.
- `sound-library/noise/`: noise beds and events from DEMAND (recorded room noise, kept multi-channel and mixed
  without a simulated impulse response) and ESC-50 (events and keyboard), built by
  `data/sound_library/build_noise_library.py`, with source and licence per clip in `index.yaml`. Every noise kind
  also has a procedural generator, so stage 7 never depends on the library.

---

## 5. Milestones

Each milestone ended with tests passing and one concrete exit check. The table gives the status, and the sections
below give what was built and what the exit check found.

| Milestone | Status | Finished |
|---|---|---|
| M0 Foundations and interfaces | done | Oct 18 |
| M1 Stage 1, meeting sampler | done | Oct 18 |
| M2 NSC data prep and speaker splits | done | Oct 18 |
| M3 Phases A and B, stages 2 and 3 | done | Oct 18 |
| M4 Phase C bake-off | done | Oct 19 |
| M5 Stages 4 and 5 | done | Oct 20 |
| M6 to M8 Stages 6 to 8 | done | Oct 18 (code), Oct 20 (real audio) |
| M9 End-to-end run | done | Oct 21 |
| M10 Run statistics for the viewer | done | Oct 21 |
| M11 Hardening | done | Oct 21 |
| M12 Viewer API | done | Oct 21 |
| M13 Viewer UI | done | Oct 21 |
| M14 Run builder in the viewer | first version, branch `run-builder` | |

M4 to M8 were built in parallel by three agents in separate worktrees, under the rules in `docs/PARALLEL_WORK.md`.
That file is now history.

### M0: Foundations and interfaces

- Built: the scaffold, the §3.1 config schemas and default YAMLs, the §3.2 artifact models and path helpers, the
  §3.3 stage protocol, runner, cache key and status store, the §3.4 role protocols with fakes and record and
  replay, and the CLI with a lease that does nothing in fake mode.
- Exit: `synthetics run` on 5 meetings with fake backends produces the full storage layout, and a second run is a
  cache hit for every stage.

### M1: Stage 1, meeting sampler

- Built: room class, speaker count and the draws that do not depend on geometry (hybrid and remote count,
  presentation and presenters, who returns to a seat, wall seating by choice) come first, so resampling the room
  never biases their rates. Then the room, table and positions are redrawn until everyone fits, with
  `fit_resamples` counting the redraws. Then each person gets an NSC speaker from the meeting's split and a
  matching persona, then the chair and Zipf turn shares (speaker ids by share rank, so S01 is the chair), then the
  mic, speakerphone, acoustics, device chain and dynamics.
- Tests: geometry properties (every seat inside the room with the required clearance, seat pitch and mouth
  heights in range, no wall seats in small rooms), a byte-identical `meeting.json` for the same seed, and
  distribution tests on 10,000 meetings.
- Exit: `just sampler-report 1000` matches every configured rate within tolerance (z standard errors plus a
  small floor) and exits 1 on a miss.

### M2: NSC data prep and speaker splits

- Built: `nsc/pool.py` validates the CSV (175 unique speakers, every zip present, splits of 50, 25 and 100, no
  partner pair split). `nsc extract` writes the candidate 5 to 15 s segments. `deepfilternet3`, `wpe-dereverb` and
  `dnsmos` were registered on the platform, and `nsc clean` cleans and ranks the references. `splits` copies the
  fixed splits into a run.
- Exit: every pool speaker has 5 clean references, and each split can fill a 20-speaker meeting. None of the 231
  unused SDM speakers was needed.

### M3: Phases A and B, stages 2 and 3

- Built: the three LLM services (§4.2), with the generator quantised locally to NVFP4A16. `text/` holds the
  Singlish convention (`text`, `text_norm`, TTS input, NSC notation, numbers as words) with thorough tests,
  because stages 2, 5 and 8 depend on it. PDPA stripping removes identifiers and names outside the roster from any
  past-meeting context in a prompt.
- Stage 2 plans the turns before writing them (`script_plan.py`). The plan fixes the speaker of every turn
  (largest-remainder counts from the Zipf shares, with no speaker following themselves), the interaction types
  (exact overlap and interruption counts), the presentation segment (monologue, then questions from the floor),
  backchannels and side conversations. Interactions are counted on turn changes, since a speaker who keeps the
  floor has no transition to overlap. The generator then writes a brief and a stance per speaker, and then the
  words for the plan in chunks of 40 turns of structured JSON with the last 30 turns as context. Each persona
  prompt (template `persona_v2`) is saved verbatim. The plan stays fixed across regeneration rounds, so a round
  only changes the words.
- Stage 3 runs the rule checks (schema, speakers, shares, interaction rates, cut-offs, turn lengths, markup,
  presentation), then the primary judge with rubric `rubric_v2` (rationale before scores), then the tie-breaker
  when the overall score is within 0.5 of the pass mark of 3.0. Agreement decides, and disagreement fails the
  script. Long scripts are judged on an excerpt of 600 turns.
- Judge calibration is file-based. `export-judge-set` writes a rating sheet, the scripts and instructions, with the
  judges' scores kept apart so raters are not anchored. `fit-judge-threshold` fits the pass mark and band from a
  filled sheet into a YAML overlay for `init-run --config`. The ratings have not been collected yet, so the pass
  mark is provisional.
- Exit: the exit was 50 meetings. The run made was `m3-10` with 10 meetings, all of which passed phases A, B and
  B2. Two faults found there were fixed: two-person meetings with a 2:1 share left no transitions for overlaps, so
  the plan now counts interactions on turn changes, and the generator wrote stage directions into turns, which
  are now dropped. Gemma judges strictly and Apertus leniently.

### M4: Phase C bake-off

- Built: the phase C services, `synthetics bakeoff` (30 speakers, 270 clones over three TTS modes, each clone
  compared with a real recording of the same words by the same speaker), and `calibrate-sv` (875 clean clips,
  1,750 same-speaker and 15,225 different-speaker pairs).
- Exit: VoxCPM2 in `ultimate` mode was chosen (mean similarity 0.806 WavLM and 0.844 ReDimNet, 100% would pass
  stage 5). The SV thresholds are the EER thresholds (WavLM 0.5568, ReDimNet 0.6515) for single utterances and for
  speaker pairs, and the 5th percentile of same-speaker means for consistency (0.6872 and 0.7991). Real NSC
  references score only 1.3 to 3.3 on UTMOS, so the MOS check is speaker-relative: an utterance may score at most
  0.75 below its own reference clip, with an absolute floor of 1.0. Details are in `docs/VOICE_BAKEOFF.md`.

### M5: Stages 4 and 5

- Stage 4 clones each speaker from their best clean clip by MOS, kept for every utterance. Speech is split at
  sentence ends past 300 characters and cloned piece by piece. Each non-speech tag becomes a sound-library clip,
  a cloned stand-in ("haha" for a laugh), or a synthetic sound. The pieces are joined with 0.15 s gaps, resampled
  and trimmed. A retry from stage 5 re-makes only the listed turns, plus every turn of a swapped speaker, with a
  new seed.
- Stage 5 checks each utterance with both SV models against the reference clip, the ASR round trip (WER on
  `text_norm` words, CER for utterances of 3 words or fewer, fillers dropped from both sides, ASR spellings such as
  "yeah" mapped), and the MOS floor. Backchannels, turns without words, and utterances under 1 s skip SV.
  Consistency uses each speaker's utterances of 3 s or more and is skipped with fewer than 3 of them. The pair
  check compares the mean cosine over utterance pairs of two speakers with `sv_pair_max`. Failed utterances are
  re-synthesised up to 3 attempts in all, and an utterance still failing is kept and listed (`keep_flagged`).
- Exit: the exit was cut from 50 meetings to the 10 of the M3 run (`runs/m5-exit`), because 50 would have cost
  about 5 hours of scripts and a day of phase C without moving the utterance rates. All 10 meetings passed, with
  3,783 utterances, 96.1% passing on the first take, 27 kept failing (0.71%) and 2 clip swaps. Seven faults found
  on real meetings were fixed first, among them the SV check on hums, ASR language detection, the pair check scale
  and consistency on short utterances. `docs/M5_EXIT.md` has the details.

### M6: Stage 6, conversation assembly

- Built: each main turn follows the floor with the transition its interaction asks for. A normal turn gets a
  log-normal gap or, at 2%, a long pause. An overlap start begins 0.3 to 2.0 s before the floor turn ends, at most
  half of either turn. An interruption starts inside the floor turn, and the floor speaker is cut off 0.3 to 1.5 s
  later (`truncated_at`). Backchannels go inside the turn they follow and side conversations run alongside it.
  Nobody overlaps their own speech, because every segment goes in its speaker's first free slot.
- Overlap control: when the planned overlaps fall short of the meeting's overlap target, stage 6 scales the
  overlap leads (up to 3 times) and then lets more normal transitions overlap. Backchannels are thinned when the
  meeting comes out shorter than planned, to keep their rate.
- Tests: realised overlap, backchannel and interruption rates within their ranges over many meetings, no
  self-overlap, and truncation matching `truncated_at`.

### M7: Stage 7, acoustic simulation

- Built: a pyroomacoustics shoebox from `meeting.json`, with frequency-dependent materials from the
  pyroomacoustics database. One Eyring factor scales every surface's absorption until the talkers' median
  mid-band RT60 matches the target, calibrated on every talker position. Impulse responses are directive ISM up to
  order 3 plus ray tracing for the late tail in large rooms. Small rooms use ISM alone, at an order high enough
  for the RT60 fit (at most 30). Each
  ray-traced source position has its own seed, so the calibration, the check and the render see the same impulse
  responses.
- Each segment uses the talker's pose (seat with per-turn facing within 60 degrees of the table centre, presenter
  drift and screen facing, the speakerphone for remote participants). Facing and drift are quantised to bound the
  number of impulse responses. Levels are set at the source only.
- Noise beds (HVAC, projector fan) are mixed at the sampled SNR, with keyboard typing and events relative to
  speech. The device chain applies band limiting, AGC, clipping and Opus, and hybrid meetings add the far-end path
  for remote participants and echo-cancellation ducking and onset clipping. Opus runs through ffmpeg, and its
  output is realigned so the labels keep their timing.
- Exit: the measured RT60 must be within 15% of the target, or the stage fails. Changing only render settings
  re-runs only stages 7 and 8.

### M8: Stage 8, labels and packaging

- Built: the RTTM from the timeline, forced alignment per utterance window on the dry tracks (language `zh` for
  utterances with Chinese characters, `en` otherwise), the transcript in `text` and `text_norm`, and the manifest
  entry. `manifest.jsonl` is rebuilt from the entries.
- Quality gates: the RTTM speech total equals the timeline's, every aligned word lies inside its segment within
  0.05 s, and every utterance's alignment covers at least 90% of its words. On clips under 1 s, aligned times are
  clamped to the clip first, because the aligner placed "mm" past the end of short clips.
- `docs/OUTPUT_CONTRACT.md` documents the output for the fine-tuning team.
- Real audio (Oct 20): on the 10 `m5-exit` meetings the first pass failed 1 meeting in stage 7 (RT60 calibrated on
  4 probes, checked on all talkers) and 5 in stage 8 (short-clip alignment). After the two fixes all 10 passed,
  with RT60 within 9% of target and 100% alignment coverage.

### M9: End-to-end orchestration

- Built: the real lease through `stack.py`, phase rounds, `resume` after any stop, `coverage-seed`, and the run
  report (drops and why, time per phase, phase C rates, split disjointness).
- The run is `runs/m9-e2e`, 20 meetings, with a seed chosen by `coverage-seed` so the meetings cover all three
  splits, both room classes, a 20-speaker test meeting, presenters, wall seats (overflow and by choice) and remote
  participants.
- Exit: all 20 meetings passed every stage. Scripts passed in round 1 for 16 meetings, round 2 for 2 and round 3
  for 2 (regeneration rate 0.2), with none dropped. Phase C made 7,172 utterances in about 3.4 hours: 95.2% passed
  on the first take, 96 (1.3%) were kept failing, there were 4 clip swaps and no speaker swaps, and the mean SV
  scores were 0.774 (WavLM) and 0.834 (ReDimNet). Stages 6 and 7 took about 10 minutes and phase D about 10
  minutes. No NSC speaker appears in two splits. Meeting m-000020 first failed the RT60 check, and passed after the
  M11 fix (0.654 s against 0.65 s). Phase A takes about 5.5 minutes per script, and a 20-meeting run takes about 9
  hours in all.
- With M5, the stage 5 meeting rates now rest on 30 meetings.

### M10: Run statistics for the viewer

- Built: `viewer/contract.py` and `synthetics viewer-contract -r <run>`, which check every row of the UI design's
  data contract on every meeting. A meeting's file is only required once the stage that writes it has passed.
- Exit: passes on `m9-e2e` (52 rows), `m5-exit`, and a fake run that covers every position type
  (`tests/test_viewer_contract.py`). Speaker facing per turn is not in v1: the plan shows the one sampled direction
  per seat, and that decision is recorded in `UI_DESIGN_DOCUMENT.md`.

### M11: Hardening

- RT60 noise: ray tracing gave each probe a different random draw, so calibration and the check measured
  different impulse responses for the same talker. Each source position now gets its own seed (the run seed plus
  a CRC32 of the position key).
- Acoustics cache: run revision 2 (§3.1). Old runs were checked to stay cache hits.
- Ablations: switching off stage 3, 5 or 7 stalled the run before the manifest. Phase C now accepts a skipped
  stage 3 and skips stage 5 when it is off, and the manifest leaves out files of disabled stages.
  `tests/test_ablation.py` runs each case to the manifest.
- Turn-taking defaults: 50% of transitions now overlap, by 0.3 to 2.0 s (was 30% and 0.1 to 1.0 s). Stage 6 now
  stretches leads about 2 times on average instead of 3, and converts 3% of transitions instead of 26%.
- Gap between turns: log-normal, median 0.5 s, sigma 0.5, capped at 2.5 s (was median 0.3 s, sigma 0.6,
  capped at 2 s). Turn changes in the M9 mixtures sounded too quick. Re-assembling the `m9-e2e` meetings
  with both settings gave the same overlap share and stretching, and meetings 2% longer.
- Small faults: fake runs work on the real NSC references, phase time is written as it is spent, and
  `SOUND_LIBRARY_DIR` and `SYNTHETICS_PORT` are in `.env.example`.
- `DESIGN_DOCUMENT.md` has a table of every departure from the first draft, with the text corrected in line.

### M12: Viewer API

- Built: a read-only FastAPI app (`viewer/api.py`, `synthetics viewer`, port 8610) over `SYNTHETICS_RUNS`. It
  lists runs and serves their files, adds no fields of its own, and serves the built frontend.
- Routes:
  - `GET /api/runs`: id, meeting and done counts, seed, config hash, revision
  - `GET /api/runs/{id}/config`, `/manifest`, `/summary` (run and voice reports), `/speaker_splits`
  - `GET /api/runs/{id}/meetings`: one summary per meeting (status, the plan fields the grid needs, manifest)
  - `GET /api/runs/{id}/meetings/{mid}/{artifact}`: whitelisted JSON files and `prompts/*.txt` only, no audio
  - `GET /api/runs/{id}/jobs`: progress from the status files, for the sidebar jobs card
  - `GET /api/runs/{id}/export?...`: the filtered manifest slice, the viewer's one action
- Run ids, meeting ids and artifact names are checked against what is on disk and the whitelist before any path is
  built. `tests/test_viewer_api.py` covers every route, the whitelist and encoded traversal attempts, and checks
  that there are no write routes.

### M13: Room Config Viewer UI

- Built, in the order of the UI design's next steps:
  1. The categorical palette `cat-1` to `cat-6` and the Synthetics icon were added to
     `/home/user/code/dev/DESIGN_SYSTEM.md` (the earlier version is `DESIGN_SYSTEM.md.bak` beside it). The
     palette passed the dataviz validator on the panel surface.
  2. The frontend scaffold follows Concord and AeroAudio: tokens, `ui.tsx`, `MetricTile`, `SortableTable`,
     `format.ts`, and the sidebar shell with run select and jobs card.
  3. Meeting detail: stage tiles, the floor plan at 90 px per metre (walls by material pattern, front wall on the
     left, seats as buttons, presenters, wall chairs, layers for facing, mic paths and grid), the side elevation,
     the selected-speaker card and fact panels, the persona panel (verbatim prompt beside a script excerpt with a
     `text` and `text_norm` toggle), and the speakers table. One selection is shared by all of them, and the
     default is the speaker with the second-highest turn share.
  4. Meeting browser: tiles, filters that apply at once, sort, cards with mini plans, load more, empty, loading
     and error states, and Previous and Next in filter order.
  5. Sampler distributions: target tiles and grouped histograms with the configured range shaded, following the
     dataviz skill.
  6. A table view for every chart, keyboard access (every seat is a button with an accessible name and
     `aria-pressed`), and the responsive layout (the sidebar becomes a top bar below `lg`, and plans scroll
     instead of scaling).
- Exit: `just viewer-test` passes (14 tests), and the three screens were compared by screenshot with
  `assets/mockups/png/*` at 1440 px and 390 px wide.

### M14: Run builder in the viewer

- Goal: create a dataset from the viewer, choosing the parameters in a form, and run it without a terminal.
- Built (Oct 21, branch `run-builder`):
  - `viewer/builder.py`: `GET /api/defaults` (the complete default config), `POST /api/preview` (stage 1 in
    memory for a draft, up to 500 meetings, with the sampler checks; a meeting is a function of the seed, its id
    and the config, so these are the meetings the run will get), `POST /api/runs` (`init-run`), and
    `GET /api/runs/{id}/process`, `POST .../start` and `POST .../stop`. A draft is the run id, meeting count
    and seed, dotted overrides and a YAML overlay, applied as `--set` and `--config` apply them. Validation
    errors come back per field.
  - `orchestrator/background.py`: Start runs the CLI's `run` in its own session, logging to `process.log` and
    recording itself in `process.json` (`RunProcess`); the process writes how it ended. Stop sends SIGINT, so
    the GPU lease releases its services, and Start again resumes. One run at a time across all runs. A run on
    real backends needs `confirm`, which the UI asks for in a dialog listing each phase's services. The jobs
    route no longer reports a stopped run as active.
  - Frontend: New dataset (`/new`; curated fields in `lib/draft.ts`, starting from the defaults or from any run's
    frozen config, with a debounced preview) and Pipeline (`/runs/<run>/pipeline`), under a Generate group in
    the sidebar. `synthetics viewer --read-only` leaves the builder out.
- Tests: `tests/test_viewer_builder.py` (preview equals what stage 1 writes, validation, create, confirm, one
  at a time, stop, how the process ended, read-only mode) and `frontend/src/lib/draft.test.ts`. A fake run was
  also created, started and finished from the UI in a headless browser.
- Not done yet: a run cannot be deleted or renamed from the UI, a stop while `stack.py up` is still starting
  services can leave them up (the same as Ctrl-C in a terminal), and the time estimate is a single ratio from
  `m9-e2e`.

### After M13: performance backlog (not scheduled)

Start only where a run report shows a need. In the M9 run, phase C took about 3.4 hours and phase A about 5.5
minutes per script (about 2 hours with regeneration rounds), and the other phases took minutes.

- Concurrent requests per role within a phase (vLLM batches on the server side). Phases A and C would gain most.
- Running stages 6 and 7 on a CPU process pool, alongside the GPU phases.
- Batch sizing, to spread the 30 to 90 s cost of loading a phase's models.
- Batched SV and MOS calls (the signatures already accept batches), and the OIP binary tensor extension in
  `aisvc`.
- The phase A memory split between KV cache size and the number of concurrent requests.
- Paging on the viewer's meetings route, which sends about 12 KB per meeting and is fine for hundreds of meetings
  but not thousands.

Open quality items (decisions in `docs/HANDOFF.md`): "(mm) X." backchannels fused by the TTS (29 of the 96 kept
failing in M9), names in short sentences misheard by the ASR (15 of the 96), the judge calibration, and the
interruptions stage 6 cannot place (about a tenth of those planned).

---

## 6. Testing

- Unit tests (`just test`, 226 tests, no GPU or platform needed): config loading and sampling, stage 1 geometry
  and distributions, the Singlish convention, the turn plan and script rules, phase C on fake audio, timeline
  assembly, acoustics units, RTTM and transcript writers, cache keys and revisions, status transitions,
  orchestration and resume, ablations, the CLI, the output contract on fake audio, the viewer contract and the
  viewer API. `tests/conftest.py` builds one shared fake run for the viewer tests.
- Frontend tests (`just viewer-test`, 14 tests, vitest): the derived values in `src/lib` (geometry, facing, plan
  orientation, people rows, filters, sorting, histogram bins, formatting), and on the detail screen, seat
  selection by click and keyboard, wall labels and the persona excerpt, on two fixture meetings.
- Determinism: the same seed and config give a byte-identical `meeting.json`, and stage 7 renders the same
  mixture from the same inputs.
- Real services: the checks against real models are the real runs (`m3-10`, `m5-exit`, `m9-e2e`), the bake-off,
  and the platform's `just conform <name>` for each service. The record and replay transport is in place, but no
  recordings are checked in yet, and the `live` pytest marker (`just test-live`) has no tests yet.
- Platform side: every custom service has mock-model tests that run the contract checks in process, and passes
  `just conform` on the 5090.
- `ruff` is clean (`just lint`), and `tsc -b` type-checks the frontend as part of `npm run build`.

---

## 7. Implementation risks and how they turned out

| Risk | Outcome |
|---|---|
| A model has no vLLM support in the pinned vLLM (Qwen3-ASR, ForcedAligner, MERaLiON plugin) | Qwen3-ASR runs on vLLM. The aligner and MERaLiON run as custom servers on `aisvc`. The `Transcriber` and `Aligner` interfaces did not change. |
| The phase B pair does not fit under `util_limit` | It did not fit. The tie-breaker runs in its own sub-phase, B2, over borderline scripts only. |
| VoxCPM2 pins an old torch that does not run on sm_120 | It runs on the 5090 with torch.compile, at 7.4 GB peak. |
| Local NVFP4 quantisation hurts script quality | W4A4 was not possible in 31 GB of RAM, so the generator is weight-only NVFP4A16. Scripts pass the judges at a regeneration rate of 0.2. |
| Train and dev have few voices (50 and 25) | Accepted, since only test needs variety. If it shows in fine-tuning, grow train from the 231 unused SDM speakers and rebuild the CSV. |
| JSON tensor payloads are slow for audio | Accepted. Speed-ups are in the performance backlog. |
| Behaviour changes alter old runs | Gated on the run revision (§3.1), so old runs keep their draws and cache keys. |
| Phase C VRAM with the Singlish ASR | About 1 GB of headroom, so MERaLiON is out of phase C and `asr_limited` is always zero. |
