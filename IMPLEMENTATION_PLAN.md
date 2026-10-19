# Synthetics — Implementation Plan

Oct 18, 2026 · Draft for review

Sources: `DESIGN_DOCUMENT.md` (pipeline), `UI_DESIGN_DOCUMENT.md` (Room Config Viewer),
the AI service platform at `/home/user/code/dev/platform`, the AeroAudio design system at
`/home/user/code/dev/DESIGN_SYSTEM.md`, and the Concord app (`apps/Concord`) as the reference for how an
app uses the platform.

The order is backend interfaces first, then the pipeline stages behind them, then platform services, and
the viewer UI last.

**Correctness before speed.** The first goal is a working end-to-end pipeline. Until it exists there is
no performance work: execution is sequential, with no concurrency, no throughput tuning, no payload
optimisation and no profiling. The stage and backend interfaces keep later speed-ups confined to the
scheduler and the backend clients, so stage code does not change when they come. The backlog is at the
end of §5.

---

## 1. Platform integration

- **Every model is a platform service.** Each model in the stack gets a manifest in
  `platform/services/<name>.yaml`. Synthetics never starts Docker containers itself and never loads a
  model in process. It calls services only through `aicore.client(name)`. The platform already handles
  compose, health gating, VRAM budgets and contract tests, and other apps can reuse the same services.
- **Weights go in `$AI_ASSETS`.** This is the platform's weights store, `/home/user/code/ai-assets`
  (set in `platform/.env`). Every platform service mounts it, so any app can reuse the models.
  - Engine services (vLLM) pull into `$AI_ASSETS/hf-cache`.
  - Custom services use `just fetch-weights <name>`, which writes to
    `$AI_ASSETS/custom-models/<name>/<version>`.
  - Checkpoints made locally, such as quantised models, go in `$AI_ASSETS/<dir>`, and the manifest uses
    `model: /models/<dir>`.
  - Nothing is downloaded into `apps/Synthetics`. The platform rule is that weights never go in images or
    git, so they do not go inside the `platform/` git tree either.
- **The GPU lease is `stack.py`.** For each GPU phase, the orchestrator calls
  `platform/tools/stack.py up <services...>` and later `stack.py down <services...>`, as Concord does
  (`platform_dir`, `platform_venv` and `no_evict` settings). `stack.py up` already validates, checks the
  VRAM budget, starts GPU services one at a time, waits for health, and stops services that do not fit.
  Synthetics only has to say which services each phase needs.
- **Service APIs follow the platform contracts.** They replace the design doc's `/synthesize`,
  `/transcribe`, `/embed` and `/align` routes. The design doc's goal of one consistent API for each kind
  of service still holds. §4.2 gives the contract for each service.

---

## 2. Repository layout

The layout follows the design doc's code structure and adds the viewer. Tooling matches Concord: uv,
Python 3.12, ruff, pytest, a Justfile, and `aicore[oip]` as an editable path dependency.

```
apps/Synthetics/
  pyproject.toml               # aicore = { path = "../../platform/clients/aicore", editable = true }
  Justfile                     # setup, test, run, stage, splits, bakeoff, phases, viewer
  .env.example                 # SYNTHETICS_RUNS, PLATFORM_DIR, NSC_DIR, AI_HOST, NO_EVICT
  data/
    nsc_part3_reference_speakers.csv   # pinned NSC voice pool with fixed splits (§4.3)
    selection/                 # scripts that rebuild the CSV
  backend/synthetics/
    config/                    # Pydantic schemas + default YAMLs
      defaults/                # room_classes.yaml, acoustics.yaml, turn_taking.yaml, seating.yaml,
                               # backends.yaml, phases.yaml, singlish_lexicon.yaml, stages.yaml
    artifacts/                 # Pydantic models for every file in runs/<run_id>/ (the data contract)
    stages/                    # s1_sample.py ... s8_labels.py, each exposing a Stage
    validators/                # rule checks, LLM judges, SV / ASR / quality scorers
    backends/                  # role protocols + aicore implementations + fakes + record/replay
    acoustics/                 # room builder, directivity, noise library, device chain
    text/                      # Singlish lexicon, text -> text_norm, TTS input derivation
    nsc/                       # NSC pool loading, reference extraction, cleanup jobs, splits
    orchestrator/              # phase scheduler, GPU lease, cache, retries, status store, CLI
    export/                    # RTTM, transcript, manifest writers
    viewer/                    # read-only FastAPI over runs/ (milestone M12)
  frontend/                    # React + Vite + Tailwind v4, copied from Concord's base (milestone M13)
  tests/
  runs/                        # gitignored; SYNTHETICS_RUNS can point elsewhere
```

---

## 3. Backend interfaces (build these first)

All interfaces are typed Python (Pydantic v2 models and `typing.Protocol`). They are written and
unit-tested with fake backends before any model service exists, so stages 1 to 8 can be developed and
tested without a GPU.

### 3.1 Configuration (`config/`)

```python
class RunConfig(BaseModel):            # frozen to runs/<run_id>/config.yaml
    schema_version: Literal["1"]
    run_id: str
    seed: int
    n_meetings: int
    sample_rate: int = 16000
    room_classes: RoomClassesConfig    # ratio (0.3 / 0.7), per-class speaker range + distribution,
                                       # dims, ceiling, table shapes, seat pitch 0.6–0.75 m, wall clearance
    seating: SeatingConfig             # wall seating p, wall offset, chair pitch, presenter p + count dist,
                                       # presenter position, drift, facing-screen p, return-to-seat p
    acoustics: AcousticsConfig         # per-class RT60, materials, SNR, noise event rates, keyboard p,
                                       # hybrid p + remote count, AGC / codec / clipping p, orientation
    turn_taking: TurnTakingConfig      # Zipf exponent, gap log-normal, overlap fraction + range,
                                       # interruption rate + cutoff, backchannels/min, long pauses,
                                       # per-class overlap targets
    stages: dict[StageId, StageToggle] # enabled + per-stage overrides (ablations)
    backends: BackendsConfig           # role -> platform service name (see 3.4)
    phases: PhasesConfig               # phase -> list of roles, batch size, max rounds
    retries: RetryConfig               # N attempts for script regeneration and TTS re-synthesis
    thresholds: ThresholdsConfig       # judge pass mark + borderline band, SV thresholds per model,
                                       # WER / CER limits, short-utterance rule, MOS floor
    lexicon_path: Path                 # versioned Singlish lexicon
    speaker_pool: SpeakerPoolConfig    # path + version of the pinned NSC pool CSV (fixed splits);
                                       # meeting split mix (share of meetings per split)

def load_run_config(*yaml_paths: Path, overrides: dict | None = None) -> RunConfig: ...
def config_hash(model: BaseModel) -> str:   # canonical JSON -> sha256[:12]; the UI shows it
```

Each distribution is a small typed union, for example `Uniform(lo, hi)`, `LogNormal(median, sigma, cap)`,
`Bernoulli(p)`, `Categorical({k: w})` or `Zipf(exponent)`. Each one has a `.sample(rng)` method. The
fine-tuning team can then replace defaults with measured values by editing YAML, with no code change.
The config records every value used.

The two speaker-verification models have equal standing in `ThresholdsConfig`. Each has its own
calibrated threshold, and neither one's score is weighted above the other's. The design doc warns about
circular validation if one of them matches the fine-tuning team's diarisation embedder. The equal
weighting stays until that embedder is known.

### 3.2 Artifacts (`artifacts/`): the data contract

There is one Pydantic model per file in the storage layout, and each one carries `schema_version`.
Writers are atomic (write to a temp file, then rename). These models are the contract for both the
fine-tuning application and the viewer, so the fields the UI design asks for are included now.

| File | Model | Fields to include beyond the design doc |
|---|---|---|
| `meeting.json` | `MeetingSpec` | `fit_resamples` (UI tile); `front_wall` + `screen` segment; per-person `position_type` (`table` / `wall` / `presenter` / `remote`), `wall_reason` (`overflow` / `by_choice`), `seat_index`, mouth xyz, `facing_deg` (one per seat in v1); `nsc_speaker` and its sheet attributes; presenter `spot`, `drift_radius`, `return_seat`; `presentation_segment` (start/end turn index); mic type, position, channel count and array geometry; per-wall materials; floor and ceiling; noise sources; device chain; personas; turn shares; `chair_id` |
| `script.json` | `Script` | `turns[]`: `turn_id`, `speaker_id`, `interaction` (`normal` / `interruption` / `backchannel` / `overlap_start` / `side` / `presentation`), `text`, `text_norm`, `disfluencies[]`, `nonspeech[]`; `generator` (service, prompt template version, thinking on/off, seed) |
| `prompts/<speaker_id>.txt` | (plain text) | The exact rendered persona prompt from stage 2, stored verbatim (UI data contract) |
| `script_validation.json` | `ScriptValidation` | rule check results; primary judge rationale + scores (including persona consistency per speaker); tie-breaker result; round number; feedback text |
| `utterances/<turn_id>.wav` | (audio) | Dry audio, at the TTS sample rate, resampled to the run rate |
| `voice_validation.json` | `VoiceValidation` | per utterance: both SV scores, WER/CER for primary and fallback ASR, `asr_limited`, MOS, attempts, seed; per speaker: consistency; per pair: centroid distance in both models; reference swaps |
| `timeline.json` | `Timeline` | `segments[]`: `turn_id`, `speaker_id`, `start`, `end`, `truncated_at`, `kind`; `stats`: overlap share, backchannels/min, interruption share (UI asks for these in the manifest too) |
| `tracks/<speaker_id>.wav` | (audio) | Dry, time-aligned per-speaker track |
| `mixture.wav`, `mixture_mc.wav` | (audio) | Single channel; multi-channel when the mic is an array |
| `labels/meeting.rttm` | (RTTM) | — |
| `labels/transcript.json` | `Transcript` | per utterance `text`, `text_norm`, `words[]` with start/end and the alignment score |
| `status.json` | `MeetingStatus` | per stage: `state` (`pending` / `running` / `pass` / `fail` / `retry` / `skipped` / `dropped`), `attempts`, `started_at`, `ended_at`, `cache_key`, `detail` (the one-line detail shown on the UI stage tile) |
| `speaker_splits.json` | `SpeakerSplits` | NSC speaker → split (copied from the fixed pool CSV), pool version, sheet metadata, counts per split |
| `manifest.jsonl` | `ManifestEntry` | every sampled parameter; enabled components; judge scores; regeneration rounds; ASR-limited counts; turn-taking stats; position-type counts; per-speaker position type |

`artifacts/paths.py` holds one helper per path (`MeetingPaths(run_dir, meeting_id).script`, and so on),
so no code builds paths by hand.

### 3.3 Stage contract (`stages/`)

```python
class StageId(IntEnum): SAMPLE=1; SCRIPT=2; SCRIPT_VAL=3; VOICE=4; VOICE_VAL=5; ASSEMBLE=6; ACOUSTIC=7; LABELS=8

class Stage(Protocol):
    id: StageId
    name: str
    phase: Phase                         # A, B, C, D or CPU
    roles: tuple[Role, ...]              # backend roles it needs (empty for CPU stages)
    reads: tuple[str, ...]               # artifact keys
    writes: tuple[str, ...]
    version: str                         # bump when the code changes outputs; part of the cache key

    def config_slice(self, cfg: RunConfig) -> BaseModel: ...
    def run(self, ctx: StageContext) -> StageResult: ...

@dataclass
class StageContext:
    paths: MeetingPaths
    cfg: RunConfig
    rng: np.random.Generator             # seeded from (run seed, meeting seed, stage id, attempt)
    backends: Backends                   # only the roles this stage declared
    feedback: str | None                 # judge or voice-check feedback from the previous round
    log: logging.Logger

@dataclass
class StageResult:
    state: Literal["pass", "fail", "retry"]
    detail: str                          # one line, shown on the UI stage tile
    metrics: dict[str, float | int | str]
    feedback: str | None = None          # sent to the upstream stage on retry
    retry_stage: StageId | None = None   # stage 3 -> 2, stage 5 -> 4
```

Rules from the design doc, enforced by the runner:

- A stage reads only its declared inputs and writes only its declared outputs plus its status record.
- **Cache key** = sha256(stage id, `version`, `config_slice`, hashes of the input artifacts). The stage
  is skipped when `status.json` already holds a `pass` with the same key. Changing only acoustics
  invalidates stages 7 and 8 and nothing upstream.
- A disabled stage is recorded as `skipped` and its toggle goes into the manifest.
- Seeds are derived (`np.random.SeedSequence([run_seed, meeting_seed, stage, attempt])`), so a meeting
  can be reproduced from its seed and the run config alone.

### 3.4 Backend roles (`backends/`)

Each model role is a small Protocol. Each role has three implementations: `Aicore*` (calls the platform
service), `Fake*` (deterministic, for unit tests), and record/replay wrappers (as in Concord) for
integration tests without a GPU. `backends.yaml` maps each role to a platform service name or alias, so
swapping a model is a config change.

```python
@dataclass(frozen=True)
class Audio: samples: np.ndarray; sr: int        # float32 mono; helpers to resample (soxr) and save

class ChatLLM(Protocol):
    def chat(self, messages: list[Msg], *, json_schema: dict | None = None, temperature: float,
             seed: int, max_tokens: int, thinking: bool | None = None) -> ChatResult: ...
    # vLLM structured outputs via response_format={"type": "json_schema", ...}

class VoiceCloner(Protocol):
    def clone(self, text: str, ref: VoiceRef, *, seed: int) -> Audio: ...   # VoiceRef = clean audio + transcript

class Transcriber(Protocol):
    def transcribe(self, audio: Audio, *, language: str | None = None) -> str: ...

class SpeakerEmbedder(Protocol):
    name: str
    def embed(self, clips: Sequence[Audio]) -> np.ndarray: ...             # (n, d), L2-normalised

class QualityScorer(Protocol):
    def score(self, clips: Sequence[Audio]) -> np.ndarray: ...             # MOS-like, (n,)

class Aligner(Protocol):
    def align(self, audio: Audio, text: str, *, language: str) -> list[WordTiming]: ...

class Enhancer(Protocol):
    def clean(self, audio: Audio) -> Audio: ...                            # denoise or dereverb
```

Roles in `backends.yaml` (defaults): `generator`, `judge_primary`, `judge_tiebreak`, `tts`, `asr_primary`,
`asr_singlish`, `sv_wavlm`, `sv_redimnet`, `quality`, `aligner`, `enhancer`, `dereverb`.

`Backends.for_roles(roles)` builds clients lazily. The orchestrator checks `aicore.endpoint(name)` for
every configured role at startup, so a missing manifest fails early.

The Singlish normaliser (`text` to `text_norm`) and its lexicon are **not** a backend role. They live in
`text/` inside Synthetics. The lexicon is versioned with each run (`config/singlish_lexicon.yaml`), and a
shared platform service would have to version it per call. The platform's existing `text-normalizer`
service does a different job.

### 3.5 Orchestrator (`orchestrator/`)

```python
class GpuLease:                          # context manager
    def __init__(self, services: list[str], *, platform_dir: Path, no_evict: bool, timeout_s: int): ...
    def __enter__(self): ...             # stack.py up <services>; raises LeaseError on refusal or timeout
    def __exit__(self, *exc): ...        # stack.py down <services>

class StatusStore:                       # per-meeting status.json, atomic updates, file lock
    def get(self, meeting_id) -> MeetingStatus: ...
    def mark(self, meeting_id, stage, result: StageResult, cache_key: str): ...
    def pending(self, stage: StageId) -> list[str]: ...

class PhaseScheduler:
    def run(self, run: Run) -> RunReport:
        # 1. copy fixed splits from the pool CSV, then stage 1 for all meetings (CPU)
        # 2. rounds: lease A -> stage 2 for pending; lease B -> stage 3; failed -> back to A with
        #    feedback; stop after max rounds; unresolved scripts -> dropped
        # 3. lease C -> stages 4 and 5 with re-synthesis inline; reference swaps for close speakers
        #    (optional sub-phase for the Singlish ASR fallback if VRAM is tight)
        # 4. stages 6 and 7 for each meeting, one meeting at a time (CPU)
        # 5. lease D -> stage 8 alignment; then RTTM, transcript and manifest
```

- Phase membership lives in `config/defaults/phases.yaml` as lists of roles. The scheduler resolves
  them to service names and runs `just vram <services...>` once at startup. If a phase set is over
  `util_limit`, the run fails before any GPU work starts. Grouping work by phase is required, not an
  optimisation: only one phase's models fit on the GPU at a time.
- Everything runs **sequentially**: one meeting at a time and one request at a time within a phase, with
  no thread or process pools. Stages never manage concurrency themselves, so a pool can be added later in
  the scheduler alone.
- The whole run is one batch (20 meetings in M9). Batch size stays a config value for later.
- **CLI** (`synthetics`, Typer):
  - `init-run --config ... --n 20`
  - `splits`
  - `run [--until STAGE] [--only PHASE]`
  - `resume`
  - `stage N --meeting ID`, to debug one stage on one meeting
  - `status`
  - `bakeoff`
  - `calibrate-sv`
  - `export-judge-set`, `fit-judge-threshold`

  Every command is idempotent and reads `status.json` to decide what is left.

---

## 4. Platform services and data

### 4.1 Procedure for each service

Use the platform's `register-model` skill for each one. It covers the contract choice, VRAM sizing, the
manifest, tests and validation. Then work through the platform README steps:

1. Write the manifest.
2. Run `just validate` and `just vram <name>`.
3. Engine services: run `just up <name>` (weights pull into `$AI_ASSETS/hf-cache`).
   Custom services: write a `weights.yaml` with pinned revisions and sha256 hashes, run
   `just fetch-weights <name>` (weights go to `$AI_ASSETS/custom-models/<name>/<version>`), then run
   `just build <name>`.
4. Run `just conform <name>`.
5. Record the measured VRAM in the manifest comment.

Every image is built on the platform's CUDA 12.8+ or 13 bases and must be tested on the 5090 right away,
as the design doc's Blackwell risk asks. Tag each version claim marked "verify" below with `TODO(verify)`
until it has been checked.

### 4.2 Service list

Ports follow the platform's ranges. The ports currently in use are 8101–8104, 8201–8205, 8301, 8401–8403
and 8501.

| Phase | Service name | Model | Tier | Contract / API | Port | VRAM (estimate) |
|---|---|---|---|---|---|---|
| A | `qwen-sea-lion-v4-5-27b-it` | aisingapore/Qwen-SEA-LION-v4.5-27B-IT, NVFP4 | vLLM engine | openai / chat.completions | 8206 | util 0.80 (runs alone) |
| B | `gemma-sea-lion-v4-27b-it-nvfp4` | aisingapore/Gemma-SEA-LION-v4-27B-IT-NVFP4 | vLLM engine | openai / chat.completions | 8207 | util ~0.58 |
| B | `apertus-sea-lion-v4-8b-it` | aisingapore Apertus-SEA-LION-v4-8B-IT | vLLM engine, `--quantization fp8` | openai / chat.completions | 8208 | util ~0.33 |
| C | `voxcpm2` | VoxCPM2 (2B, Apache-2.0) | custom, py-cuda | **oip-v2 / infer** | 8105 | ~8 GB (measure) |
| C | `qwen3-asr-1-7b` | Qwen/Qwen3-ASR-1.7B | vLLM engine if supported, otherwise custom | openai / audio.transcriptions | 8106 | util ~0.12 |
| C | `meralion-3-3b-asr` | MERaLiON/MERaLiON-3-3B-ASR | vLLM engine + plugin (`vllm-meralion` in `engines.yaml`) | openai / audio.transcriptions | 8107 | util ~0.25 |
| C | `wavlm-sv` | microsoft/wavlm-base-plus-sv (or a larger WavLM SV head; decided in the bake-off) | custom, py-cuda | oip-v2 / infer | 8404 | ~2 GB |
| C | `redimnet2-sv` | ReDimNet2 | custom, py-cuda | oip-v2 / infer | 8405 | ~2 GB |
| C | `utmos` | UTMOS (DNSMOS as an alternative) | custom, py-cpu | oip-v2 / infer | 8406 | CPU |
| D | `qwen3-forced-aligner` | Qwen/Qwen3-ForcedAligner | custom, py-cuda (unless vLLM serves it) | oip-v2 / infer | 8407 | ~4 GB |
| prep | `deepfilternet3` | DeepFilterNet3 (denoise) | custom, py-cpu | oip-v2 / infer | 8408 | CPU |
| prep | `wpe-dereverb` | nara_wpe (dereverb, non-AI) | custom utility, py-cpu | oip-v2 / infer | 8502 | CPU |

Why each non-`openai` contract is needed (this text goes in the manifest's `contract_reason`):

- **TTS cloning (`voxcpm2`).** The OpenAI `audio.speech` route takes only a voice name. Cloning needs a
  reference audio array plus its transcript, which is non-text input. Pre-registering the 175 pool
  voices as named voices was considered and rejected, because references are swapped per meeting.
- **Speaker embeddings, quality score, alignment and enhancement.** Each takes an audio array, and no
  OpenAI route fits.

Proposed OIP signatures (`-1` means a variable dimension):

```yaml
voxcpm2:
  inputs:  [TEXT BYTES [1], REF_AUDIO FP32 [-1], REF_SAMPLE_RATE INT32 [1], REF_TEXT BYTES [1], SEED INT64 [1]]
  outputs: [AUDIO FP32 [-1], SAMPLE_RATE INT32 [1]]
wavlm-sv / redimnet2-sv:          # batch-shaped; clips are zero-padded and LENGTHS gives the true sizes
  inputs:  [AUDIO FP32 [-1, -1], LENGTHS INT32 [-1], SAMPLE_RATE INT32 [1]]
  outputs: [EMBEDDINGS FP32 [-1, -1]]
utmos:
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

- `aisvc` and `aicore.OIPClient` already support `BYTES` tensors. The platform OIP uses JSON tensor
  encoding only, so payloads must stay short: utterances and 5–15 s references are fine. Alignment
  therefore runs **per utterance window** of the dry track (the timeline gives each window's start and
  end), and the results are offset back into the track. A 30–60 min track is never sent as one request.
  This also avoids the aligner's maximum input length. The SV and MOS signatures accept batches, but the
  clients send one clip per request for now.
- MERaLiON is an audio LLM. If its plugin exposes only chat with `audio_url` and not
  `/v1/audio/transcriptions`, the manifest uses `api: chat.completions` and the `Transcriber` adapter
  sends the audio as a chat message. Concord's `openai_chat_audio` adapter is the precedent.
- **Qwen-SEA-LION runs as NVFP4**, which is native on Blackwell. If AI Singapore publishes an official
  4-bit variant, use it. Otherwise quantise it once locally with llm-compressor into
  `$AI_ASSETS/qwen-sea-lion-v4.5-27b-it-nvfp4` and set `model: /models/qwen-sea-lion-v4.5-27b-it-nvfp4`.
  The quantisation script goes in `platform/tools/` or `experiments/`, so the checkpoint is a platform
  asset and not an app asset.
- **Reference cleanup is denoise then dereverb.** DeepFilterNet3 removes noise, and WPE
  (`wpe-dereverb`) removes reverb. WPE is the first choice for dereverberation. A neural dereverb model
  would only be added if the bake-off shows WPE leaves too much room colour in the clones.
- Phase B budget: Gemma NVFP4 at about 0.58 plus Apertus 8B. In bf16 Apertus would be about 16 GB,
  which does not fit next to Gemma. With FP8 it is about 0.33, for a total of about 0.91, just under
  `util_limit` 0.93. If it does not fit in practice, run the tie-breaker as a sub-phase over the
  borderline scripts only.
- Phase C budget: about 0.25 + 0.12 + 0.25 + 0.06 + 0.06 = about 0.74. MERaLiON stays co-loaded. The
  design doc's sub-phase fallback is still available.

### 4.3 Data assets (not models)

These are datasets, so they stay in `/home/user/code/datasets/`, next to the NSC data that is already
there.

#### NSC Part 3 source (checked 2026-10-18)

- `NSC_PART3/SDM/` holds 406 per-speaker zips (`SPEAKER<8-digit id>.zip`, 51 GB in total). Each zip has
  `SPEAKER<id>/manifest.json` (JSON lines: `conf`, `pair_id`, `source_start`, `source_end`, `text`),
  `WAVE/<pair_id without "_">.wav` (16 kHz mono 16-bit, one file per segment), and `TEXT/*.txt`. Every
  manifest line has a matching WAV.
- `SPEAKER_INFO_PART3.XLSX` has two sheets. All 406 SDM speakers are in the **"Separate Room"** sheet
  (columns: Conference ID, Speaker ID, Gender, Age band, Ethnicity, Education, Occupation, Language
  Proficiency, First Language, Partner ID, Relationship). The "Same Room" sheet uses different IDs
  (`3014-2` and so on) and matches no SDM zip, so it is not used.
- The 406 speakers: 228 F / 178 M; 239 Chinese, 89 Malay, 73 Indian, 5 Others; ages 1X to 6X.
- Part 3 is conversational speech, as the design doc asks for. IVR (telephone) audio is not used. The
  curated `nsc-spontaneous` and `nsc-scripted` sets are used only for the bake-off and for threshold
  calibration.

#### The 175 selected speakers and their splits (`data/nsc_part3_reference_speakers.csv`)

Only those 175 speaker zips are ever read.

**Only test needs voice variety.** The test split gets most of the pool. Train and dev are kept at the
smallest size that still fills a 20-speaker meeting with spares left for reference swaps.

| Split | Speakers | Shared voices between two 20-speaker meetings (expected, k²/N) |
|---|---|---|
| train | 50 | 8 of 20 |
| dev | 25 | 16 of 20 |
| test | 100 | 4 of 20 |

**Pool selection rules** (`data/selection/2_select_pool.py`):

1. **Quotas by ethnicity and gender.** Chinese 43 F + 43 M, Malay 22 F + 22 M, Indian 20 F + 20 M, and
   Others 3 F + 2 M (all 5 in the dataset). Malay and Indian are oversampled compared with the full set,
   so local accents are well covered.
2. **Age mix within each cell.** Aim for 8% 1X, 25% 2X, 25% 3X, 22% 4X, 12% 5X and 8% 6X, which leans
   toward working ages for meetings. Where a cell runs short, the slots go to the next band.
3. **Ranking within each cell and age band.** Speakers with the most clean reference material come first.
   Clean material means segments of 5–15 s with at least 8 words and no `#name#` redactions, `<…>` tags,
   `**` or `~` marks.

Pool result: 88 F / 87 M; 86 Chinese, 44 Malay, 40 Indian, 5 Others; ages 15 / 50 / 46 / 42 / 12 / 10
(1X to 6X); first language 97 English, 37 Chinese, 27 Malay, 13 Tamil, 1 Hindi. Every speaker has at
least 43 clean candidate clips (4.4 min or more). The 175 zips total 24.7 GB, compared with 51 GB for
all of SDM.

**Split rules** (`data/selection/3_assign_splits.py`, seeded local search):

- Partner pairs (40 pairs, both in the pool) are always in the same split. Partners recorded the same
  conversation, so one speaker's audio can carry faint traces of the other's voice.
- All 5 Others speakers go to test, where variety matters. Train and dev personas never draw "Others".
- Every other ethnicity-and-gender group, and every age band, is split in proportion to split size.

| Split | Chinese F/M | Malay F/M | Indian F/M | Others F/M | Ages 1X–6X |
|---|---|---|---|---|---|
| train (50) | 13 / 12 | 7 / 6 | 6 / 6 | 0 / 0 | 5 / 14 / 13 / 12 / 3 / 3 |
| dev (25) | 6 / 7 | 3 / 3 | 3 / 3 | 0 / 0 | 2 / 7 / 7 / 6 / 2 / 1 |
| test (100) | 24 / 24 | 12 / 13 | 11 / 11 | 3 / 2 | 8 / 29 / 26 / 24 / 7 / 6 |

**Splits are fixed per pool, not drawn per run.** This changes the design doc's "global split
assignment" principle. The CSV's `split` column is authoritative, and each run copies it into
`speaker_splits.json`. The test voices then stay the same across runs, so the fine-tuning team's test
results can be compared from one run to the next. Splits stay speaker-disjoint as before.

**Provenance.** The CSV is the source of truth, and the pipeline reads it directly. The four scripts in
`data/selection/` (statistics, pool, splits, CSV writer) rebuild it from the spreadsheet and the SDM zips.
Run them in that order. Script 2 writes a first-pass split that script 3 replaces. To change the pool,
edit the scripts, rebuild the CSV, and bump the pool version recorded in each run's config.

#### Rules for using the pool

- **Cloning is limited to the meeting's split.** A meeting in a given split clones only that split's
  speakers. Stage 4 swaps a reference first to another clip of the same speaker, then to another speaker
  in the same split with matching sheet attributes.
- **The persona must match the voice.** The NSC speaker is assigned in **stage 1**, not stage 4. Each
  persona's gender, ethnicity, L1 and code-switching languages are then sampled to match that speaker's
  sheet row. The design doc assigns references in stage 4; this change keeps persona and voice
  consistent. Stage 4 still chooses which clip to use and handles swaps, within the same speaker set.
- **NSC transcript notation is different from ours.** NSC writes particles in square brackets
  (`[lah]`, `[ah]`), fillers in parentheses (`(uh)`), and names as `#name#`. Our convention puts
  particles in parentheses and keeps square brackets for non-speech. The reference-text loader converts
  the notation (`[lah]` → `(lah)`, and so on, through the lexicon) before the transcript is used as the
  cloning prompt. Clips with `#name#` are never used as references (PDPA).

#### Derived datasets

- `nsc-part3-refs/raw/<speaker>/`: candidate segments extracted only from the 175 selected zips (WAV +
  converted transcript).
- `nsc-part3-refs/clean/<speaker>/`: references cleaned with DeepFilterNet3 then WPE, written once and
  reused by every run. `index.jsonl` gives the clip path, transcript (raw and converted), duration, sheet
  metadata, cleanup version and MOS. Each speaker keeps its top few clips by MOS, so stage 4 can swap
  clips.
- `sound-library/`: real non-lexical clips (laughter, coughs, "mm-hmm", breaths) and noise beds (HVAC,
  projector fan, keyboard, door, chair, cup, paper), each with its source and licence recorded in
  `index.yaml`.

---

## 5. Milestones

Each milestone ends with tests passing and one concrete exit check. Interface work (M0) comes first, the
pipeline is built stage by stage behind fakes, and real services are added as each GPU phase is needed.
The UI comes last (M12–M13).

### M0: Foundations and interfaces (no models)
- Scaffold: `pyproject.toml`, `Justfile`, `.env.example`, ruff, pytest, `aicore` path dependency.
- §3.1 config schemas and default YAMLs, with all values from the design doc's sampling tables.
- §3.2 artifact models and path helpers. §3.3 stage protocol, runner, cache key and status store.
- §3.4 role protocols with `Fake*` implementations and record/replay.
- §3.5 CLI skeleton, plus a `GpuLease` that is a no-op in fake mode.
- **Exit:** `synthetics run` on 5 meetings with all fake backends produces the full storage layout, and
  a second run skips every stage (cache hit).

### M1: Stage 1, meeting sampler (CPU)
- Room class, speaker count, dimensions, table fit with resampling (`fit_resamples` counted), table and
  wall seats (overflow and by choice), presenters with drift and return seat, mic placement, acoustics,
  dynamics (Zipf shares, chair, presentation segment, hybrid, topic), NSC speaker per person from the
  meeting's split, and persona attributes sampled to match that speaker (§4.3).
- **Tests:** property tests (every seat inside the room with 1 m wall clearance where required, seat pitch
  within range, mouth heights within range, small rooms never use wall seats). A test that the same seed
  gives a byte-identical `meeting.json`. Distribution tests over 10,000 samples against the configured
  targets.
- **Exit:** a 1,000-meeting sample matches every configured rate within tolerance.

### M2: NSC data prep and speaker splits (CPU, plus prep services)
- `nsc/pool.py`: load and validate `data/nsc_part3_reference_speakers.csv`. Checks: 175 unique speakers,
  every zip present, split sizes 50 / 25 / 100, no partner pair split across splits.
- `synthetics nsc extract`: read the 175 selected zips in place and write only the candidate 5–15 s
  segments to `nsc-part3-refs/raw/`, with transcripts converted to our notation. No full unzip.
- Register `deepfilternet3` and `wpe-dereverb` on the platform (§4.1), then clean every candidate
  (denoise, then WPE dereverb) into `nsc-part3-refs/clean/` and keep the top clips per speaker by MOS.
- `synthetics splits`: copy the fixed splits from the CSV into the run's `speaker_splits.json`, with the
  sheet metadata per speaker and the pool version.
- **Exit:** every selected speaker has at least 3 clean references, and each split can still fill a
  20-speaker meeting after removing speakers that fail cleanup. Replacements, if needed, come from the
  231 unused SDM speakers in the same group.

### M3: Phase A and B services, and stages 2–3
- Platform: `qwen-sea-lion-v4-5-27b-it` (NVFP4, quantised locally if no official 4-bit variant exists,
  §4.2), `gemma-sea-lion-v4-27b-it-nvfp4`, `apertus-sea-lion-v4-8b-it`. Run `just vram` for the phase B
  set together.
- `text/`: Singlish lexicon, `text` to `text_norm`, TTS input derivation, all following the design doc's
  convention table. This gets thorough unit tests, because stages 2, 5 and 8 all depend on it.
- Stage 2: persona prompt templates (versioned) rendered per speaker and saved to `prompts/<speaker_id>.txt`.
  Structured JSON output with a schema. Turn-share targeting. PDPA stripping of past-meeting context.
  Test with thinking on and off.
- Stage 3: programmatic checks, then the primary judge (rationale before scores), then the borderline
  tie-breaker, with feedback sent back to stage 2.
- **Judge calibration is file-based, with no rating UI.** `export-judge-set` writes 50–100 scripts to a
  CSV rating sheet (one row per script, rubric columns to fill in). Raters fill it in a spreadsheet, and
  `fit-judge-threshold` reads it back and fits the pass threshold and borderline band. The raters
  (ideally Singaporean) still need to be named before this step.
- **Exit:** 50 meetings pass phases A and B in rounds, and the regeneration rate is reported.

### M4: Bake-off (gate before phase C is final)
- Register the phase C candidates (§4.2). `synthetics bakeoff` runs a few hundred NSC references and
  measures speaker similarity in both SV models, between-speaker separability, Singlish WER through
  both ASRs, and MOS. It writes a report. The report also checks whether WPE leaves audible room colour
  in the clones.
- `calibrate-sv`: fit per-model thresholds on real NSC same-speaker and different-speaker pairs.
- **Exit:** the TTS, SV and ASR choices are confirmed, or swapped by changing `backends.yaml`. The
  thresholds are written to config.

### M5: Stages 4–5 (phase C)
- Stage 4: pick one clean clip of the NSC speaker assigned in stage 1 and keep it for every utterance
  of that speaker; swaps go to another clip of the same speaker first, then to another same-split
  speaker with matching sheet attributes. Continuation cloning, one WAV per utterance, sound-library
  clips for non-lexical tags.
- Stage 5: both SV models must pass, each against its own threshold, with neither weighted above the
  other (§3.1). Within-speaker consistency. Between-speaker separability, where a pair is flagged if
  either model finds it too close, followed by reference swaps. Two-step ASR round-trip with the
  `asr_limited` tag. CER for short utterances. MOS floor. Re-synthesis with a new seed, up to N attempts.
- **Exit:** the 10 meetings of the M3 run through phase C (about 3,800 utterances), with pass,
  first-pass, re-synthesis, ASR-limited and swap rates reported (`synthetics voice-report`). The
  utterance rates are firm at this size. The meeting rates (pass, swaps) are not, so M9 reports them
  again over its 20 meetings. This was 50 meetings; it was cut to save about 5 hours of scripts and
  a day of phase C that would not have moved the utterance rates.

### M6: Stage 6, conversation assembly (CPU)
- Timeline placement driven by the turn-taking config: gaps, overlaps, interruptions with truncation,
  backchannels inside other speakers' turns, long pauses. Write per-speaker dry tracks and the stats
  block.
- **Tests:** realised overlap, backchannel and interruption rates fall within their configured ranges
  over many meetings. No speaker overlaps their own speech. Truncation matches `truncated_at`.

### M7: Stage 7, acoustic simulation (CPU)
- pyroomacoustics room from `meeting.json` (hybrid ISM plus ray tracing for large rooms), reflective
  materials, source directivity, per-turn facing and presenter drift (render per segment), wall-seat
  pickup kept without normalisation, laptop near-far imbalance and keyboard noise, centre-mic array
  with a derived single channel.
- Noise beds and events at the sampled SNR. Device chain: AGC, band-limiting, Opus 16–32 kbps, clipping,
  and AEC artifacts for hybrid meetings (speakerphone path for remote participants).
- **Exit:** RT60 measured on the simulated impulse responses falls within the sampled target. Changing
  only acoustics re-runs only stages 7–8.

### M8: Stage 8, labels and packaging (phase D)
- Platform: `qwen3-forced-aligner`.
- RTTM from the timeline. Forced alignment per utterance window on the dry tracks. Transcript in both
  `text` and `text_norm`. Manifest entry with every field in §3.2.
- **Quality gates** (the design doc goal that audio and labels agree): the RTTM total speech matches
  the sum of the timeline segments; every aligned word falls inside its segment; alignment coverage
  stays above a threshold; otherwise the stage fails.
- Version the output contract (`schema_version`) and write `docs/OUTPUT_CONTRACT.md` for the
  fine-tuning team.

### M9: End-to-end orchestration
- Real `GpuLease` through `stack.py`, phase rounds, `resume` after a crash, and a run report (drops
  and why, plus time per phase for reference only; nothing is tuned against it yet).
- A sequential run is expected to take many hours. That is accepted, and `resume` makes it safe to stop
  and restart.
- **Exit:** a 20-meeting run completes unattended. Speaker disjointness across splits is checked on
  the final manifest. With only 20 meetings, random sampling can miss rare cases, so the run's config
  makes sure of coverage: all three splits, both room classes, at least one 20-speaker test meeting,
  and at least one meeting each with presenters, wall seats (both overflow and by choice) and remote
  participants.
- The run report includes `synthetics voice-report` for its 20 meetings, so the M5 meeting rates
  (stage 5 pass rate, speaker swaps per meeting) rest on 30 meetings in all.

### M10: Run statistics for the viewer
- Turn-taking stats in `manifest.jsonl`, `fit_resamples`, per-speaker position types and per-stage
  detail strings. These are the "new fields needed from the pipeline" in the UI doc. Most are built
  into earlier milestones; this milestone checks they are complete against the UI data contract table.

### M11: Hardening
- Ablation runs (stage toggles recorded in the manifest).
- Update `DESIGN_DOCUMENT.md` where this plan departs from it: platform contracts in place of the
  custom routes (§1, §4.2), weights in `$AI_ASSETS` (§1), the fixed NSC pool and splits (§4.3), voice
  assignment in stage 1 (§4.3), and sequential execution (§3.5).

### M12: Viewer backend (read-only API)
- A browser cannot read the local disk, so the viewer gets a thin read-only FastAPI layer over
  `runs/<run_id>/`. It only lists runs and serves artifact files, and adds no fields of its own. All
  derived numbers are still computed in the browser, as the UI doc says.
- Routes:
  - `GET /api/runs` (id, meeting count, seed, config hash)
  - `GET /api/runs/{id}/config`
  - `GET /api/runs/{id}/manifest`
  - `GET /api/runs/{id}/meetings`: one summary per meeting (manifest plus status), for the browser grid
  - `GET /api/runs/{id}/meetings/{mid}/{artifact}`: whitelisted JSON files and `prompts/*.txt` only
  - `GET /api/runs/{id}/jobs`: progress from `status.json` files, for the sidebar jobs card
  - `GET /api/runs/{id}/export?filter=...`: the manifest slice, the viewer's one action
- No write routes. Path traversal is guarded. It serves the built frontend like Concord does.

### M13: Room Config Viewer UI (last)
The order follows the UI doc's next steps.
1. AeroAudio additions in `DESIGN_SYSTEM.md`: the categorical palette `color-cat-1…6` and the Synthetics
   icon (`assets/icon.svg`). This is a design-system change, so update the document in the same commit.
2. Frontend scaffold copied from Concord per DESIGN_SYSTEM §11: tokens, `ui.tsx`, `MetricTile`,
   `SortableTable`, `format.ts`, sidebar shell with run select and jobs card.
3. **Meeting detail** first: pipeline status tiles, floor plan (90 px/m, walls by material pattern, front
   wall left, seats as buttons, presenters, wall strip, layers), side elevation, selected-speaker card
   and fact panels, persona panel (verbatim prompt plus script excerpt with a text / text_norm toggle),
   speakers table. One selection state is shared by all of them.
4. **Meeting browser**: tiles, filters (no Apply button), sort, card grid with mini plans, load more,
   empty, loading and error states, Previous and Next in filter order.
5. **Sampler distributions**: target tiles and grouped histograms with the configured range shaded,
   following the `dataviz` skill.
6. Table view for every chart, keyboard and screen-reader pass, responsive checks (sidebar becomes a top
   bar below `lg`; plans scroll and do not scale).
- Check each screen against `assets/mockups/png/*` and `assets/mockups/html/Room Config Viewer.html`.

### After M13: performance backlog (not scheduled)
Start only after the end-to-end pipeline works, and only where the M9 run report shows a real need.
- Concurrent requests per role within a phase (vLLM batches on the server side).
- Running stages 6 and 7 on a CPU process pool, alongside the GPU phases.
- Batch sizing to spread the 30–90 s model-swap cost.
- Batched SV and MOS calls (the signatures already accept batches), plus the OIP binary tensor extension
  in `aisvc`.
- Phase A memory split (KV cache size versus number of concurrent requests).

---

## 6. Testing approach

- **Unit tests** (no GPU): config sampling, geometry, text normalisation, timeline assembly, RTTM and
  transcript writers, cache keys, status transitions, all against fake backends.
- **Recorded integration tests**: real service responses recorded once (`--record`), then replayed in
  CI-style runs, as Concord does.
- **Live tests** (`-m live`): one meeting end to end against running platform services.
- **Determinism tests**: same seed and config give identical CPU-stage artifacts.
- **Platform side**: every new custom service has mock-model tests that run `checks_for(service_meta(...))`
  in process, and passes `just conform <name>` on the 5090.

---

## 7. Implementation risks (in addition to the design doc's risks)

| Risk | Mitigation |
|---|---|
| A model has no vLLM support in the pinned v0.31.0 (Qwen3-ASR, ForcedAligner, MERaLiON plugin) | Use the platform's tiers in order: engine, then engine plugin, then custom server on `aisvc`. The `Transcriber` and `Aligner` interfaces do not change. |
| The phase B pair does not fit under `util_limit` | FP8 for Apertus; otherwise run the tie-breaker as a sub-phase over borderline scripts only. |
| VoxCPM2 pins an old torch that does not run on sm_120 | Test the build in M4 before writing any stage 4 code that depends on it. Fall back to the next bake-off candidate. |
| Local NVFP4 quantisation of Qwen-SEA-LION hurts script quality | Compare a sample of scripts against the bf16 model through the stage 3 judges before committing. |
| Train and dev have few voices (50 and 25), so models may overfit to them | Accepted, since only test needs variety. If it shows up in fine-tuning, grow train from the 231 unused SDM speakers and rebuild the CSV. |
| JSON tensor payloads are slow for audio | Accepted for now. Speed-ups are in the performance backlog. |
