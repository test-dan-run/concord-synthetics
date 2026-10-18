"""Where each talker is, and which way they face, for every segment of the timeline (stage 7).

- Table and wall seats stay fixed. Facing starts at the seat's `facing_deg` and is re-drawn per turn with
  `acoustics.orientation_resample_p`, within `orientation_max_deg` of the direction to the table centre.
- Presenters sit at their return seat before the presentation (when they have one), stand near their spot
  during it, drifting within `drift_radius` between turns and facing the screen with `facing_screen_p` (else the
  table), and go back to the return seat afterwards (or keep standing).
- Remote participants are heard through the speakerphone (an omnidirectional source). In small rooms stage 1
  puts it at the laptop mic itself; the laptop's loudspeaker is taken to sit `SimulationConfig.speaker_mic_gap_m`
  from the mic, toward its owner, so the direct path stays finite.

Positions and angles are quantised (`SimulationConfig.drift_grid_m`, `facing_step_deg`) so a meeting needs a
bounded number of impulse responses.
"""

from __future__ import annotations

import math

import numpy as np

from ..acoustics import geometry as geo
from ..artifacts.labels import Segment, Timeline
from ..artifacts.meeting import MeetingSpec, Person, SeatRef
from ..config.schema import SimulationConfig
from .room import Pose

FLEX = ("backchannel", "side")
SEATED_MOUTH_Z = 1.15  # a presenter back in a seat


def script_order(tl: Timeline) -> list[Segment]:
    """Segments in script order (turn ids are t0001, t0002, ... in script order)."""
    return sorted(tl.segments, key=lambda s: s.turn_id)


def seat_xy(m: MeetingSpec, ref: SeatRef) -> tuple[float, float]:
    return m.table.slots[ref.index].seat_xy if ref.kind == "table" else m.wall_chairs[ref.index].xy


def _q(x: float, step: float) -> float:
    return round(x / step) * step if step > 0 else x


def _toward_table(m: MeetingSpec, xy: tuple[float, float]) -> float:
    return geo.angle_deg(xy, m.table.center)


def speakerphone_xyz(m: MeetingSpec, gap: float) -> tuple[float, float, float]:
    """The remote talkers' source: the speakerphone, kept at least `gap` from the mic."""
    assert m.speakerphone is not None, "remote participant without a speakerphone"
    sp, mic = np.asarray(m.speakerphone.position, float), np.asarray(m.mic.position, float)
    if np.linalg.norm(sp - mic) >= gap:
        return tuple(float(c) for c in sp)
    toward = None
    if m.mic.near_speaker:
        owner = m.person(m.mic.near_speaker)
        if owner.mouth is not None:
            toward = np.array([owner.mouth[0] - mic[0], owner.mouth[1] - mic[1], 0.0])
    if toward is None or np.linalg.norm(toward) < 1e-6:
        toward = np.array([m.table.center[0] - mic[0], m.table.center[1] - mic[1], 0.0])
    if np.linalg.norm(toward) < 1e-6:
        toward = np.array([1.0, 0.0, 0.0])
    out = mic + gap * toward / np.linalg.norm(toward)
    out[2] = min(mic[2], m.table.height + 0.02)  # on the table, at the keyboard
    return tuple(float(round(c, 4)) for c in out)


def segment_poses(m: MeetingSpec, tl: Timeline, sim: SimulationConfig, rng: np.random.Generator) -> dict[str, Pose]:
    """turn_id -> pose of its talker during that segment."""
    ac = m.acoustics
    seg = m.dynamics.presentation
    main_k: dict[str, int] = {}
    k = 0
    for s in script_order(tl):
        if s.kind not in FLEX:
            main_k[s.turn_id] = k
            k += 1
    # each segment's main index: flex turns take the index of the main turn before them
    idx: dict[str, int] = {}
    last = 0
    for s in script_order(tl):
        last = main_k.get(s.turn_id, last)
        idx[s.turn_id] = last

    poses: dict[str, Pose] = {}
    facing: dict[str, float] = {}
    sp = speakerphone_xyz(m, sim.speaker_mic_gap_m) if m.speakerphone is not None else None
    for s in script_order(tl):
        p: Person = m.person(s.speaker_id)
        if p.position_type == "remote":
            assert sp is not None, "remote participant without a speakerphone"
            poses[s.turn_id] = Pose(sp, None)
            continue
        if p.position_type == "presenter":
            poses[s.turn_id] = _presenter_pose(m, p, idx[s.turn_id], seg, sim, rng)
            continue
        xy = (p.mouth[0], p.mouth[1])
        if s.speaker_id not in facing:
            facing[s.speaker_id] = p.facing_deg if p.facing_deg is not None else _toward_table(m, xy)
        elif rng.random() < ac.orientation_resample_p:
            dev = rng.uniform(-ac.orientation_max_deg, ac.orientation_max_deg)
            facing[s.speaker_id] = geo.wrap_deg(_toward_table(m, xy) + dev)
        poses[s.turn_id] = Pose(tuple(p.mouth), _q(facing[s.speaker_id], sim.facing_step_deg))
    return poses


def _presenter_pose(m: MeetingSpec, p: Person, k: int, seg, sim: SimulationConfig, rng: np.random.Generator) -> Pose:
    spec = p.presenter
    assert spec is not None and p.mouth is not None
    ret = spec.return_seat
    presenting = seg is not None and p.speaker_id in seg.presenters and seg.start_turn <= k < seg.end_turn
    before = seg is not None and k < seg.start_turn
    if not presenting and ret is not None:
        xy = seat_xy(m, ret)
        face = geo.wrap_deg(_toward_table(m, xy) + rng.uniform(-m.acoustics.orientation_max_deg,
                                                               m.acoustics.orientation_max_deg))
        return Pose((xy[0], xy[1], SEATED_MOUTH_Z), _q(face, sim.facing_step_deg))
    z = p.mouth[2]
    if presenting or not before:
        r = spec.drift_radius * math.sqrt(rng.random())
        a = rng.uniform(0, 2 * math.pi)
        x = spec.spot[0] + r * math.cos(a)
        y = spec.spot[1] + r * math.sin(a)
        margin = 0.3
        x = min(max(x, margin), m.room.length - margin)
        y = min(max(y, margin), m.room.width - margin)
        x, y = _q(x, sim.drift_grid_m), _q(y, sim.drift_grid_m)
    else:
        x, y = spec.spot
    if rng.random() < spec.facing_screen_p and m.room.screen is not None:
        face = geo.angle_deg((x, y), (0.0, m.room.screen.center))  # the screen is on the front wall (x = 0)
    else:
        face = _toward_table(m, (x, y)) + rng.uniform(-20.0, 20.0)
    return Pose((x, y, z), _q(geo.wrap_deg(face), sim.facing_step_deg))
