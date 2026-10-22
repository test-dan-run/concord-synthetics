"""Typed run configuration (§3.1). Frozen into runs/<run_id>/config.yaml when a run is created.

Every sampled value lives here, so the fine-tuning team can replace defaults with measured values by
editing YAML. Per-class values use `PerClass` (`small` / `large`).
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .distributions import Bernoulli, Categorical, Count, Fixed, IntUniform, LogNormal, Real, Uniform, Zipf


class _M(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StageId(IntEnum):
    SAMPLE = 1
    SCRIPT = 2
    SCRIPT_VAL = 3
    VOICE = 4
    VOICE_VAL = 5
    ASSEMBLE = 6
    ACOUSTIC = 7
    LABELS = 8


class Phase(StrEnum):
    A = "A"
    B = "B"
    B2 = "B2"  # tie-break judge, when it does not fit beside the primary judge
    C = "C"
    D = "D"
    CPU = "CPU"
    PREP = "prep"


class Role(StrEnum):
    GENERATOR = "generator"
    JUDGE_PRIMARY = "judge_primary"
    JUDGE_TIEBREAK = "judge_tiebreak"
    TTS = "tts"
    ASR_PRIMARY = "asr_primary"
    ASR_SINGLISH = "asr_singlish"
    SV_WAVLM = "sv_wavlm"
    SV_REDIMNET = "sv_redimnet"
    QUALITY = "quality"
    REF_QUALITY = "ref_quality"  # ranks cleaned NSC references (prep); `quality` scores TTS output (phase C)
    ALIGNER = "aligner"
    ENHANCER = "enhancer"
    DEREVERB = "dereverb"


RoomClass = Literal["small", "large"]
Wall = Literal["N", "E", "S", "W"]


class PerClass[T](_M):
    small: T
    large: T

    def __getitem__(self, room_class: str) -> T:
        return getattr(self, room_class)


# ---------------------------------------------------------------- room classes


class TableShapeDims(_M):
    """Short side (rect width, oval minor axis, round diameter) and the long side as a fraction of the room
    length left after wall clearance. Round tables ignore `length_frac`."""

    width: Uniform
    length_frac: Uniform = Uniform(lo=0.5, hi=1.0)
    min_length: float = 1.2


class RoomClassSpec(_M):
    speakers: Count
    length: Uniform
    width: Uniform
    ceiling: Uniform
    table_shapes: Categorical  # rect | round | oval
    shapes: dict[Literal["rect", "round", "oval"], TableShapeDims]


class RoomClassesConfig(_M):
    ratio: Categorical  # {small: 0.3, large: 0.7}
    classes: PerClass[RoomClassSpec]
    seat_pitch: Uniform = Uniform(lo=0.6, hi=0.75)
    wall_clearance: float = 1.0  # table edge to every wall
    table_height: Uniform = Uniform(lo=0.72, hi=0.76)
    seat_offset: Uniform = Uniform(lo=0.2, hi=0.3)  # mouth, horizontally outward from the table edge
    min_person_gap: float = 0.5  # any two people in the room, 2D
    max_fit_resamples: int = 500
    # Added Oct 22, 2026 (new runs set it in room_classes.yaml). The default, a table anywhere that keeps the wall
    # clearance, is how earlier runs placed it, and a frozen config without the field reads it.
    table_at_centre: bool = False

    @model_validator(mode="after")
    def _ranges(self) -> RoomClassesConfig:
        sm, lg = self.classes.small.speakers, self.classes.large.speakers
        if not isinstance(sm, IntUniform) or not isinstance(lg, IntUniform):
            return self
        if sm.hi >= 7 or lg.lo < 7:
            raise ValueError("the class boundary is fixed at 7 speakers: small < 7, large >= 7")
        return self


# ---------------------------------------------------------------- seating


class WallSeatingConfig(_M):
    by_choice: Bernoulli = Bernoulli(p=0.15)  # large rooms with free table seats
    by_choice_count: IntUniform = IntUniform(lo=1, hi=3)
    offset: Uniform = Uniform(lo=0.3, hi=0.6)  # chair row to wall
    chair_pitch: Uniform = Uniform(lo=0.5, hi=0.7)
    walls: list[Wall] = ["N", "S", "E"]  # side or back walls; W is the front wall
    corner_margin: float = 0.4


class PresenterConfig(_M):
    presentation: PerClass[Bernoulli]
    count: PerClass[Categorical]  # {1: p, 2: p}
    distance_from_front: Uniform = Uniform(lo=0.5, hi=1.5)
    side_margin: float = 0.5
    mouth_height: Uniform = Uniform(lo=1.5, hi=1.75)
    drift_radius: Real = Fixed(value=1.0)
    facing_screen_p: float = 0.3
    return_to_seat: Bernoulli = Bernoulli(p=0.5)
    min_gap_between: float = 0.8
    start_frac: Uniform = Uniform(lo=0.05, hi=0.3)  # presentation start, as a share of the turn target
    length_frac: Uniform = Uniform(lo=0.15, hi=0.35)


class ScreenConfig(_M):
    width: Uniform = Uniform(lo=1.5, hi=2.5)
    bottom: float = 0.9
    top: float = 2.1


class SeatingConfig(_M):
    seated_mouth_height: Uniform = Uniform(lo=1.1, hi=1.2)
    wall: WallSeatingConfig = WallSeatingConfig()
    presenters: PresenterConfig
    screen: ScreenConfig = ScreenConfig()
    front_wall: Wall = "W"
    # Large rooms: the chance that the chair takes the head of the table, an end seat on its long axis. Added
    # Oct 22, 2026; the default 0 is how earlier runs seated the chair (any table seat).
    chair_at_head: Bernoulli = Bernoulli(p=0.0)


# ---------------------------------------------------------------- acoustics


class MicConfig(_M):
    laptop_distance: Uniform = Uniform(lo=0.3, hi=0.5)  # mouth to mic, horizontal
    laptop_height: Uniform = Uniform(lo=0.1, hi=0.25)  # above the table
    centre_offset: Uniform = Uniform(lo=-0.15, hi=0.15)  # per axis
    centre_height: float = 0.05
    centre_array: Bernoulli = Bernoulli(p=0.5)  # single mic or small array
    array_channels: int = 4
    array_radius: float = 0.035
    speakerphone_offset: Uniform = Uniform(lo=0.3, hi=0.8)  # from the table centre along the long axis
    # Large rooms: the chance that the table mic sits in front of the chair instead of at the centre, to capture
    # the chair's decisions best. Added Oct 22, 2026; the default 0 is how earlier runs placed it.
    chair_mic: Bernoulli = Bernoulli(p=0.0)
    chair_mic_distance: Uniform = Uniform(lo=0.4, hi=0.6)  # the chair's mouth to the mic, horizontal


class DeviceChainConfig(_M):
    agc: Bernoulli = Bernoulli(p=0.7)
    band_limit: Bernoulli = Bernoulli(p=0.5)
    band_limit_hz: Uniform = Uniform(lo=5000, hi=7500)
    codec: Bernoulli = Bernoulli(p=0.5)
    codec_kbps: IntUniform = IntUniform(lo=16, hi=32)
    clipping: Bernoulli = Bernoulli(p=0.05)
    clipping_level: Uniform = Uniform(lo=0.6, hi=0.9)  # of full scale
    aec_artifacts_if_hybrid: Bernoulli = Bernoulli(p=1.0)


class AcousticsConfig(_M):
    rt60: PerClass[Uniform]
    wall_material: Categorical  # per wall: {glass, plasterboard}
    carpet: Bernoulli = Bernoulli(p=0.7)  # else vinyl
    acoustic_tile: Bernoulli = Bernoulli(p=0.6)  # else plasterboard ceiling
    snr_db: PerClass[Uniform]
    hvac: Bernoulli = Bernoulli(p=1.0)
    projector_fan_if_presentation: Bernoulli = Bernoulli(p=0.6)
    keyboard: PerClass[Bernoulli]
    events_per_10min: PerClass[Uniform]
    event_types: Categorical
    orientation_max_deg: float = 60.0
    orientation_resample_p: float = 0.3
    hybrid: PerClass[Bernoulli]
    remote_count: PerClass[Count]
    mic: MicConfig = MicConfig()
    device_chain: DeviceChainConfig = DeviceChainConfig()


# ---------------------------------------------------------------- turn-taking and dynamics


class TurnTakingConfig(_M):
    zipf: Zipf = Zipf(exponent=1.0)
    gap: LogNormal = LogNormal(median=0.5, sigma=0.5, cap=2.5)
    overlap_share: float = 0.5  # of transitions
    overlap_lead: Uniform = Uniform(lo=0.3, hi=2.0)  # next turn starts this long before the previous ends
    overlap_max_frac: float = 0.5  # at most half the previous turn
    overlap_target: PerClass[Uniform]  # share of speech time
    interruption_rate: Uniform = Uniform(lo=0.05, hi=0.10)  # of turn changes
    interruption_cutoff: Uniform = Uniform(lo=0.3, hi=1.5)  # seconds after the interruption starts
    backchannels_per_min: Uniform = Uniform(lo=1.0, hi=3.0)
    long_pause_p: float = 0.02
    long_pause: Uniform = Uniform(lo=1.0, hi=3.0)
    side_conversations_per_10min: PerClass[Uniform]
    side_conversation_turns: IntUniform = IntUniform(lo=2, hi=4)


class DynamicsConfig(_M):
    duration_min: PerClass[Uniform]
    turns_per_min: Uniform = Uniform(lo=8, hi=12)  # non-backchannel turns
    presenter_share_boost: float = 3.0  # rank weights when ordering speakers by turn share
    wall_share_weight: float = 0.4
    remote_share_weight: float = 0.7
    topics: list[str]
    meeting_types: PerClass[Categorical]


# ---------------------------------------------------------------- personas


class PersonasConfig(_M):
    seniority_by_age: dict[str, Categorical]  # age band (1X..6X) -> {intern, junior, mid, senior, exec}
    roles: dict[str, list[str]]  # seniority -> role titles
    chair_seniority: dict[str, float]  # weights for choosing the chair among table-seated people
    code_switching: Categorical  # {low, medium, high}
    verbosity_by_tercile: list[Categorical]  # turn-share tercile (top, middle, bottom) -> {high, medium, low}
    l1_languages: dict[str, str]  # sheet first-language -> display name
    switch_languages: dict[str, list[str]]  # sheet ethnicity -> languages a speaker may code-switch into
    names: dict[str, dict[str, list[str]]]  # ethnicity -> gender -> given names (synthetic, never NSC)
    surnames: dict[str, list[str]]


# ---------------------------------------------------------------- stages, backends, phases


class StageToggle(_M):
    enabled: bool = True
    overrides: dict = Field(default_factory=dict)


class ScriptConfig(_M):
    template_version: str = "persona_v2"
    chunk_turns: int = 40
    context_turns: int = 30  # previous turns repeated in each chunk prompt
    temperature: float = 0.8
    max_tokens: int = 8192
    thinking: bool = False
    seed_offset: int = 0


class JudgeConfig(_M):
    rubric_version: str = "rubric_v2"
    temperature: float = 0.0
    max_tokens: int = 4096
    thinking: bool = False
    max_script_turns: int = 600  # longer scripts are judged on an excerpt of this many turns


class RetryConfig(_M):
    script_rounds: int = 3
    tts_attempts: int = 3


class ScriptRuleThresholds(_M):
    share_tolerance: float = 0.08  # max absolute gap between planned and realised share, per speaker
    min_words_normal: int = 2
    max_words_backchannel: int = 4
    max_unknown_tags: int = 0


class ThresholdsConfig(_M):
    judge_pass: float = 3.0  # 1-5 overall; rubric_v2 anchors 3 as "usable"
    judge_band: float = 0.5  # tie-breaker within +-band of the pass mark
    sv: dict[str, float] = Field(default_factory=lambda: {"sv_wavlm": 0.5568, "sv_redimnet": 0.6515})
    wer_max: float = 0.35
    cer_max_short: float = 0.40
    short_utterance_words: int = 3
    mos_min: float = 1.0  # absolute MOS floor (broken audio); the speaker-relative floor below does most of the work
    mos_drop_max: float = 0.75  # stage 5: an utterance may score at most this far below its reference clip's MOS
    script_rules: ScriptRuleThresholds = ScriptRuleThresholds()
    # stage 5; from `calibrate-sv` on 875 clean NSC clips (config/sv_thresholds.json)
    sv_consistency_min: dict[str, float] = Field(default_factory=lambda: {"sv_wavlm": 0.6872, "sv_redimnet": 0.7991})
    sv_pair_max: dict[str, float] = Field(default_factory=lambda: {"sv_wavlm": 0.5568, "sv_redimnet": 0.6515})
    # stage 8 quality gates
    align_coverage_min: float = 0.90  # share of text_norm words with a timing, per utterance
    align_slack_s: float = 0.05  # an aligned word may overrun its segment by this much
    align_clamp_below_s: float = 1.0  # clips shorter than this have aligned times clamped to the clip ("mm")


class BackendsConfig(_M):
    mode: Literal["aicore", "fake", "replay", "record"] = "aicore"
    recordings: Path = Path("tests/recorded")
    roles: dict[Role, str]


class PhaseSpec(_M):
    roles: list[Role]
    batch_size: int = 20
    max_rounds: int = 3


class PhasesConfig(_M):
    phases: dict[Phase, PhaseSpec]
    platform_dir: Path = Path("../../platform")
    no_evict: bool = False
    lease_timeout_s: int = 1800


class SpeakerPoolConfig(_M):
    path: Path = Path("data/nsc_part3_reference_speakers.csv")
    version: str = "p3-175-v1"
    split_mix: Categorical  # share of meetings per split


class LexiconRef(_M):
    path: Path = Path("backend/synthetics/config/defaults/singlish_lexicon.yaml")


class VoiceConfig(_M):
    """Stages 4-5 (M5)."""

    trim_pad_s: float = 0.05  # silence kept at each end of an utterance WAV (contract with stage 6)
    max_speaker_swaps: int = 2  # per meeting, after clip swaps run out
    max_clip_swaps: int = 1  # per speaker, before a speaker swap
    # stage 4: cloning
    tts_mode: Literal["ultimate", "continuation", "reference"] = "ultimate"  # voxcpm2 `mode` (clip as prompt and/or reference)
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    max_tts_chars: int = 300  # longer speech is split at sentence ends and cloned piece by piece
    piece_gap_s: float = 0.15  # silence between the pieces of one turn (speech chunks, non-lexical clips)
    trim_db: float = 40.0  # trimming treats audio this far below the peak as silence
    nonlexical_tts: dict[str, str] = Field(default_factory=lambda: {"laugh": "haha"})  # tag -> text to clone when the
    # sound library has no clip for it; other tags fall back to a synthetic sound
    # stage 5: checks
    asr_language: str | None = "en"  # None lets the ASR detect it, but it hears one-word turns as Chinese or
    # Japanese ("Right." -> "对。"); "en" keeps code-switched words romanised, as the scripts write them
    wer_ignore_fillers: bool = True  # (um) (uh) are dropped from both sides before WER; particles stay
    sv_min_s: float = 1.0  # shorter utterances skip the SV checks (embeddings are unreliable)
    mos_min_s: float = 1.0  # shorter utterances skip the MOS floor
    # Within-speaker consistency uses only utterances this long: thresholds.sv_consistency_min is a percentile of
    # 5-15 s NSC clip pairs, and 2 s clips embed so noisily that speakers with short turns failed it whatever voice
    # they got (WavLM to reference: 0.68 at 1-2 s, 0.77 at 3-5 s, 0.79 above 5 s)
    consistency_min_s: float = 3.0
    consistency_min_utts: int = 3  # fewer such utterances: the speaker check is skipped
    on_exhausted: Literal["fail", "keep_flagged"] = "keep_flagged"  # an utterance still failing after retries.tts_attempts


class AssemblyConfig(_M):
    """Stage 6 (M6). Turn-taking distributions live in `turn_taking`; these settle how they are realised."""

    lead_in_s: float = 1.0  # silence before the first segment
    tail_s: float = 1.0  # silence after the last segment
    min_self_gap_s: float = 0.15  # between two segments of one speaker (no one overlaps their own speech)
    interrupt_at: Uniform = Uniform(lo=0.35, hi=0.8)  # interruption start, as a share of the floor turn
    interrupt_min_before_s: float = 0.6  # the interrupted speaker says at least this much
    interrupt_min_tail_s: float = 0.2  # the cut lands at least this long before the utterance would end
    backchannel_at: Uniform = Uniform(lo=0.15, hi=0.85)  # position inside the host turn
    side_start_at: Uniform = Uniform(lo=0.1, hi=0.6)  # first side turn, as a share of the host turn
    side_gap: LogNormal = LogNormal(median=0.25, sigma=0.5, cap=1.0)  # between side-conversation turns
    # The meeting's overlap target (`turn_taking.overlap_target`) is met by scaling the overlap leads of
    # overlap_start transitions (up to `max_lead_scale`, still capped by `turn_taking.overlap_max_frac`), and,
    # when that is not enough, by letting more normal transitions overlap (`convert_transitions`).
    overlap_control: bool = True
    max_lead_scale: float = 3.0
    convert_transitions: bool = True
    fade_s: float = 0.01  # fade-out at a truncation point
    thin_backchannels: bool = True  # keep the planned backchannel rate when the meeting comes out shorter


class SurfaceMaterials(_M):
    """meeting.json surface material -> pyroomacoustics database entry (frequency-dependent absorption)."""

    walls: dict[str, str] = Field(default_factory=lambda: {"glass": "glass_window", "plasterboard": "plasterboard"})
    floor: dict[str, str] = Field(default_factory=lambda: {"carpet": "carpet_hairy", "vinyl": "linoleum_on_concrete"})
    ceiling: dict[str, str] = Field(default_factory=lambda: {"acoustic_tile": "ceiling_fissured_tile",
                                                             "plasterboard": "ceiling_plasterboard"})


class SimulationConfig(_M):
    """Stage 7 (M7). Room and device values are sampled in stage 1 (`acoustics`); these are render settings."""

    ism_max_order: int = 3  # with ray tracing for the late tail
    ray_tracing: list[RoomClass] = Field(default_factory=lambda: ["large"])
    ism_order_cap: int = 30  # rooms without ray tracing: ISM order reaches the T20 fit range, up to this cap
    materials: SurfaceMaterials = SurfaceMaterials()
    scattering: float = 0.25  # furniture and people; also makes the decay closer to exponential
    absorption_min: float = 0.01  # after scaling to the target RT60
    absorption_max: float = 0.95
    rt_n_rays: int | None = None  # pyroomacoustics default from the room size
    rt_receiver_radius: float = 0.5
    rir_tail_factor: float = 1.1  # impulse responses are cut at this many target RT60s
    directivity_p: float = 0.7  # talker directivity, cardioid family p (1 omni, 0.5 cardioid): about -8 dB behind
    calibration_iters: int = 5
    rt60_calibration_tol: float = 0.05  # relative; absorption is rescaled until the probe is this close
    rt60_tolerance: float = 0.15  # relative; stage 7 fails when the speakers' median measured RT60 is further off
    calibration_probes: int = 20  # talker positions whose median RT60 is calibrated. 20 (the largest meeting) means
    # every talker, the same set the RT60 check measures: with 4 of 14 talkers, m5-exit m-000008 calibrated to 0.52 s
    # on the probes but measured 0.60 s over all talkers, against a 0.50 s target
    facing_step_deg: float = 5.0  # facing and presenter drift are quantised, bounding the impulse responses
    drift_grid_m: float = 0.1
    # levels (dBFS of active speech at the source; the room then attenuates by distance, never normalised)
    speech_level_db: float = -20.0
    speaker_level_sd_db: float = 2.0  # per talker
    utterance_level_sd_db: float = 1.0
    backchannel_offset_db: float = -3.0
    presenter_offset_db: float = 2.0  # standing, projecting to the room
    output_level_db: float = -26.0  # speech level of the recording when the device has no AGC
    agc_target_db: float = -23.0
    agc_max_gain_db: float = 20.0
    agc_attack_s: float = 0.5
    agc_release_s: float = 3.0
    agc_gate_db: float = -45.0  # frames below this hold the gain (dBFS, after the fixed gain): quiet rooms do not pump
    # noise (the sampled SNR covers the beds; keyboard and events sit relative to speech)
    noise_library: Path | None = Path("noise")  # under $SOUND_LIBRARY_DIR (or absolute); None = procedural only
    bed_balance_db: Uniform = Uniform(lo=-6.0, hi=3.0)  # projector fan relative to HVAC
    event_level_db: Uniform = Uniform(lo=-12.0, hi=3.0)
    keyboard_level_db: Uniform = Uniform(lo=-25.0, hi=-12.0)
    # hybrid meetings
    speaker_mic_gap_m: float = 0.25  # loudspeaker to mic at least (a laptop's speakers sit by its keyboard)
    far_end_narrowband_p: float = 0.3
    far_end_kbps: IntUniform = IntUniform(lo=12, hi=24)
    aec_duck_db: Uniform = Uniform(lo=3.0, hi=10.0)
    aec_onset_s: Uniform = Uniform(lo=0.05, hi=0.2)


class LabelsConfig(_M):
    """Stage 8 (M8). Gates live in `thresholds`."""

    align_language: str = "en"  # the aligner's LANGUAGE for most utterances
    align_language_cjk: str = "zh"  # for utterances with Chinese characters (code-switched Mandarin)


# ---------------------------------------------------------------- the run


class RunConfig(_M):
    schema_version: Literal["1"] = "1"
    # How a run samples and caches. 1: runs made before Oct 21, 2026, which a frozen config without the field is.
    # 2 (the default for new runs, set in stages.yaml): stage 1 draws the acoustic values (`ACOUSTIC_FIELDS` of
    # meeting.json) from their own random stream, and stages 2-6 hash meeting.json without them, so a change to a
    # sampled acoustic distribution re-runs only stages 1, 7 and 8.
    revision: int = 1
    run_id: str
    seed: int
    n_meetings: int
    sample_rate: int = 16000
    room_classes: RoomClassesConfig
    seating: SeatingConfig
    acoustics: AcousticsConfig
    turn_taking: TurnTakingConfig
    dynamics: DynamicsConfig
    personas: PersonasConfig
    stages: dict[StageId, StageToggle] = Field(default_factory=lambda: {s: StageToggle() for s in StageId})
    script: ScriptConfig = ScriptConfig()
    judge: JudgeConfig = JudgeConfig()
    voice: VoiceConfig = VoiceConfig()
    assembly: AssemblyConfig = AssemblyConfig()
    simulation: SimulationConfig = SimulationConfig()
    labels: LabelsConfig = LabelsConfig()
    backends: BackendsConfig
    phases: PhasesConfig
    retries: RetryConfig = RetryConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    lexicon: LexiconRef = LexiconRef()
    speaker_pool: SpeakerPoolConfig

    def enabled(self, stage: StageId) -> bool:
        toggle = self.stages.get(stage)
        return toggle.enabled if toggle else True
