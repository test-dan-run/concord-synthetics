"""`meeting.json` (stage 1): the full sampled setup of one meeting, read by every later stage and the viewer.

Room frame: x runs along the room length from the front wall (W, x = 0) to the back wall (E, x = L);
y runs along the width from the S wall (y = 0) to the N wall (y = W); z is height above the floor.
Angles are degrees counter-clockwise from +x.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config.schema import RoomClass, Wall

# Drawn from stage 1's acoustic random stream (RunConfig.revision 2) and read only by stages 7 and 8: stages 2-6
# hash meeting.json without them
ACOUSTIC_FIELDS = (("room", "walls"), ("room", "floor"), ("room", "ceiling"), ("room", "rt60_target"),
                   ("acoustics",), ("device_chain",))

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
PositionType = Literal["table", "wall", "presenter", "remote"]
Split = Literal["train", "dev", "test"]


class _A(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Screen(_A):
    wall: Wall
    center: float  # along the wall (y for the W wall)
    width: float
    bottom: float
    top: float


class Room(_A):
    length: float
    width: float
    height: float
    front_wall: Wall
    walls: dict[Wall, Literal["glass", "plasterboard"]]
    floor: Literal["carpet", "vinyl"]
    ceiling: Literal["acoustic_tile", "plasterboard"]
    rt60_target: float
    screen: Screen | None = None


class TableSlot(_A):
    """A seat position around the table edge; `occupied_by` is None for a free seat."""

    index: int
    label: str  # "N3", "S1", "E", "W" ... side + number, for the UI
    side: str  # rect: N | S | E | W; round / oval: "arc"
    edge_s: float  # arc-length coordinate along the side (m), for pitch checks
    edge_xy: Vec2  # point on the table edge
    seat_xy: Vec2  # mouth position (x, y), offset outward from the edge
    occupied_by: str | None = None


class Table(_A):
    shape: Literal["rect", "round", "oval"]
    center: Vec2
    length: float  # along x (round: diameter)
    width: float  # along y (round: diameter)
    height: float
    seat_pitch: float
    capacity: int
    slots: list[TableSlot]


class WallChair(_A):
    index: int
    wall: Wall
    offset: float  # chair row to wall
    xy: Vec2
    occupied_by: str | None = None


class SeatRef(_A):
    kind: Literal["table", "wall"]
    index: int


class PresenterSpec(_A):
    spot: Vec2
    drift_radius: float
    facing_screen_p: float
    return_seat: SeatRef | None  # None: stays standing after presenting


class NscSpeaker(_A):
    """The NSC reference voice assigned in stage 1, with its sheet attributes (§4.3)."""

    speaker: str
    split: Split
    gender: Literal["F", "M"]
    age: str
    ethnicity: str
    first_language: str
    languages: list[str]


class Persona(_A):
    display_name: str  # synthetic, used in the script; never the NSC identity
    role: str
    seniority: str
    verbosity: Literal["low", "medium", "high"]
    gender: Literal["F", "M"]
    age_band: str
    ethnicity: str
    l1: str
    accent: str
    code_switching: Literal["low", "medium", "high"]
    switch_languages: list[str]


class Person(_A):
    speaker_id: str  # S01, S02, ...
    position_type: PositionType
    wall_reason: Literal["overflow", "by_choice"] | None = None
    seat: SeatRef | None = None  # table / wall seat; None for presenters (standing) and remote
    mouth: Vec3 | None  # None for remote participants (heard through the speakerphone)
    facing_deg: float | None  # one sampled direction per seat in v1
    presenter: PresenterSpec | None = None
    is_chair: bool = False
    turn_share: float
    nsc_speaker: NscSpeaker
    persona: Persona


class Mic(_A):
    type: Literal["laptop", "centre"]
    position: Vec3
    channels: int
    array_geometry: list[Vec3] = Field(default_factory=list)  # per-capsule offsets from `position`
    near_speaker: str | None = None  # laptop: the seat it sits in front of


class Speakerphone(_A):
    position: Vec3
    remote_speakers: list[str]


class NoiseSource(_A):
    kind: Literal["hvac", "projector_fan", "keyboard"]
    position: Vec3 | None = None  # None = diffuse


class Acoustics(_A):
    rt60: float
    snr_db: float
    noise_sources: list[NoiseSource]
    events: dict[str, int]  # event type -> count over the meeting
    events_per_10min: float
    orientation_max_deg: float
    orientation_resample_p: float


class DeviceChain(_A):
    agc: bool
    band_limit_hz: float | None
    codec_kbps: int | None  # Opus
    clipping_level: float | None
    aec_artifacts: bool


class PresentationSegment(_A):
    presenters: list[str]
    start_turn: int
    end_turn: int  # exclusive
    returns: dict[str, SeatRef | None]


class Dynamics(_A):
    meeting_type: str
    topic: str
    duration_min: float
    n_turns_target: int  # non-backchannel turns
    zipf_exponent: float
    turn_shares: dict[str, float]
    chair_id: str
    presentation: PresentationSegment | None
    hybrid: bool
    remote_speakers: list[str]


class MeetingSpec(_A):
    schema_version: Literal["1"] = "1"
    meeting_id: str
    seed: int
    split: Split
    room_class: RoomClass
    n_speakers: int
    fit_resamples: int  # room / table / positions redrawn before everyone fitted
    room: Room
    table: Table
    wall_chairs: list[WallChair]
    people: list[Person]
    mic: Mic
    speakerphone: Speakerphone | None
    acoustics: Acoustics
    device_chain: DeviceChain
    dynamics: Dynamics

    def person(self, speaker_id: str) -> Person:
        for p in self.people:
            if p.speaker_id == speaker_id:
                return p
        raise KeyError(speaker_id)

    def count(self, position_type: PositionType) -> int:
        return sum(p.position_type == position_type for p in self.people)
