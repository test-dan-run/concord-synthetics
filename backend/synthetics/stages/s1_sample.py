"""Stage 1: meeting sampler (CPU). Writes meeting.json.

Sampling order (design doc "Meeting config rules"):
1. room class, speaker count; then the non-geometric draws (hybrid + remote count, presentation + presenter
   count + who returns to a seat, wall seating by choice), so resampling the room never biases their rates
2. room, table and positions, redrawn until everyone fits (`fit_resamples` counts the redraws)
3. NSC speaker per person from the meeting's split, persona attributes matched to that speaker's sheet row
4. chair and turn shares (Zipf; the chair takes the top share), speaker ids by share rank (S01 = chair); in large
   rooms the chair may then move to the head of the table
5. mic, speakerphone, acoustics, device chain, dynamics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from ..acoustics import geometry as geo
from ..artifacts.io import read_model, write_model
from ..artifacts.meeting import (
    Acoustics,
    DeviceChain,
    Dynamics,
    MeetingSpec,
    Mic,
    NoiseSource,
    NscSpeaker,
    Person,
    Persona,
    PresentationSegment,
    PresenterSpec,
    Room,
    Screen,
    SeatRef,
    Speakerphone,
    Table,
    TableSlot,
    WallChair,
)
from ..artifacts.models import SpeakerSplits
from ..config.distributions import Categorical
from ..config.schema import (
    AcousticsConfig,
    DynamicsConfig,
    PersonasConfig,
    Phase,
    RoomClassesConfig,
    RunConfig,
    SeatingConfig,
    StageId,
    TurnTakingConfig,
)
from .base import StageContext, StageResult

R = 4  # decimals kept in meeting.json
DIALECTS = ("Hokkien", "Cantonese", "Teochew")


def r(x: float) -> float:
    return round(float(x), R)


def r2(p: tuple[float, float]) -> tuple[float, float]:
    return (r(p[0]), r(p[1]))


def r3(p: tuple[float, float, float]) -> tuple[float, float, float]:
    return (r(p[0]), r(p[1]), r(p[2]))


class FitError(Exception):
    """This draw of room / table / positions cannot hold everyone; draw again."""


@dataclass
class Plan:
    """Counts decided before the geometry loop."""

    room_class: str
    shape: str
    n: int
    n_remote: int
    n_presenters: int
    presenter_returns: list[bool]
    by_choice: int  # wall seats by choice, applied only when the table has room for every seated person
    hybrid: bool
    presentation: bool

    @property
    def seated(self) -> int:
        return self.n - self.n_remote - self.n_presenters


@dataclass
class Layout:
    room_l: float
    room_w: float
    room_h: float
    shape: str
    table_center: tuple[float, float]
    table_l: float
    table_w: float
    table_h: float
    pitch: float
    slots: list[geo.EdgePoint]
    seat_offset: float
    table_seated: list[int]  # slot indices taken by seated people
    return_slots: dict[int, SeatRef]  # presenter index -> seat to return to
    wall_chairs: list[tuple[str, float, tuple[float, float]]]  # wall, offset, xy
    wall_reason: str | None
    n_wall: int
    presenter_spots: list[tuple[float, float]]
    screen_center: float
    screen_width: float
    seat_xy: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------- 1. plan


def draw_plan(cfg: RunConfig, rng: np.random.Generator) -> Plan:
    room_class = cfg.room_classes.ratio.sample(rng)
    spec = cfg.room_classes.classes[room_class]
    n = int(spec.speakers.sample(rng))
    shape = str(spec.table_shapes.sample(rng))  # drawn once, so refits never bias the shape mix
    ac, pres = cfg.acoustics, cfg.seating.presenters

    hybrid = ac.hybrid[room_class].sample(rng)
    n_remote = int(ac.remote_count[room_class].sample(rng)) if hybrid else 0
    n_remote = min(n_remote, n - 1)  # at least one person in the room
    hybrid = n_remote > 0

    presentation = pres.presentation[room_class].sample(rng)
    n_presenters = int(pres.count[room_class].sample(rng)) if presentation else 0
    n_presenters = min(n_presenters, n - n_remote - 1)  # at least one seated person (the laptop needs a seat)
    presentation = n_presenters > 0
    returns = [bool(pres.return_to_seat.sample(rng)) for _ in range(n_presenters)]

    by_choice = 0
    if room_class == "large" and cfg.seating.wall.by_choice.sample(rng):
        by_choice = cfg.seating.wall.by_choice_count.sample(rng)
    return Plan(room_class, shape, n, n_remote, n_presenters, returns, by_choice, hybrid, presentation)


# ---------------------------------------------------------------- 2. geometry


def draw_layout(cfg: RunConfig, plan: Plan, rng: np.random.Generator) -> Layout:
    rc = cfg.room_classes
    spec = rc.classes[plan.room_class]
    seating = cfg.seating
    clear, gap = rc.wall_clearance, rc.min_person_gap

    L, W, H = spec.length.sample(rng), spec.width.sample(rng), spec.ceiling.sample(rng)
    shape = plan.shape
    dims = spec.shapes[shape]
    avail_l, avail_w = L - 2 * clear, W - 2 * clear
    tw = dims.width.sample(rng)
    if shape == "round":
        tl = tw
        if tw > min(avail_l, avail_w):
            raise FitError("round table does not fit")
    else:
        tl = max(dims.min_length, dims.length_frac.sample(rng) * avail_l, tw)
        if tl > avail_l + 1e-9 or tw > avail_w + 1e-9:
            raise FitError("table does not fit with wall clearance")
    if rc.table_at_centre:
        cx, cy = L / 2, W / 2
    else:
        cx = rng.uniform(clear + tl / 2, L - clear - tl / 2)
        cy = rng.uniform(clear + tw / 2, W - clear - tw / 2)
    th = rc.table_height.sample(rng)
    pitch = rc.seat_pitch.sample(rng)
    if shape == "rect":
        slots = geo.rect_slots((cx, cy), tl, tw, pitch)
    else:
        round_ = shape == "round"
        perim = geo.ellipse_perimeter(tl / 2, tw / 2) if round_ else geo.stadium_perimeter(tl, tw)
        n_slots = geo.curved_slot_count(perim, pitch, rc.seat_pitch.lo, rc.seat_pitch.hi)
        if n_slots == 0:
            raise FitError("no seat spacing in range for this table")
        pitch = perim / n_slots
        slots = (geo.ellipse_slots if round_ else geo.stadium_slots)((cx, cy), tl, tw, n_slots)
    capacity = len(slots)
    seat_off = rc.seat_offset.sample(rng)
    seat_xy = [geo.offset(s.xy, s.normal, seat_off) for s in slots]
    if any(not geo.inside_room(p, L, W, gap / 2) for p in seat_xy):
        raise FitError("table seat outside the room")

    # who sits where at the table
    n_returning = sum(plan.presenter_returns)
    seated = plan.seated
    if plan.room_class == "small":
        if capacity < seated + n_returning:
            raise FitError("small room: not everyone fits at the table")
        n_table, n_wall, reason = seated, 0, None
    elif seated > capacity:
        n_table, n_wall, reason = capacity, seated - capacity, "overflow"
    elif plan.by_choice:
        k = min(plan.by_choice, seated - 1)
        n_table, n_wall, reason = seated - k, k, ("by_choice" if k > 0 else None)
    else:
        n_table, n_wall, reason = seated, 0, None
    order = [int(i) for i in rng.permutation(capacity)]
    table_seated = sorted(order[:n_table])
    free = sorted(order[n_table:])

    # presenters stand near the front wall, clear of the table and its seats
    spots: list[tuple[float, float]] = []
    pres = seating.presenters
    for _ in range(plan.n_presenters):
        for _attempt in range(30):
            p = (pres.distance_from_front.sample(rng), rng.uniform(pres.side_margin, W - pres.side_margin))
            if (geo.point_table_distance(p, shape, (cx, cy), tl, tw) >= gap
                    and all(geo.dist2(p, s) >= gap for s in seat_xy)
                    and all(geo.dist2(p, q) >= pres.min_gap_between for q in spots)):
                spots.append(p)
                break
        else:
            raise FitError("no presenter space at the front")

    # returning presenters: a free table seat first, else (large rooms) an extra wall chair
    return_slots: dict[int, SeatRef] = {}
    extra_wall = 0
    for i, back in enumerate(plan.presenter_returns):
        if not back:
            continue
        if free:
            return_slots[i] = SeatRef(kind="table", index=free.pop(0))
        else:
            return_slots[i] = SeatRef(kind="wall", index=n_wall + extra_wall)
            extra_wall += 1

    chairs: list[tuple[str, float, tuple[float, float]]] = []
    need = n_wall + extra_wall
    if need:
        ws = seating.wall
        chair_pitch = ws.chair_pitch.sample(rng)
        walls = [ws.walls[int(i)] for i in rng.permutation(len(ws.walls))]
        for wall in walls:
            off = ws.offset.sample(rng)
            a, b = geo.wall_line(wall, L, W, off, ws.corner_margin)
            length = geo.dist2(a, b)
            if length < 0:
                continue
            n_pos = int(math.floor(length / chair_pitch)) + 1
            start = (length - (n_pos - 1) * chair_pitch) / 2
            ux, uy = ((b[0] - a[0]) / length, (b[1] - a[1]) / length) if length > 0 else (0.0, 0.0)
            for k in range(n_pos):
                p = (a[0] + ux * (start + k * chair_pitch), a[1] + uy * (start + k * chair_pitch))
                if (geo.point_table_distance(p, shape, (cx, cy), tl, tw) >= gap
                        and all(geo.dist2(p, s) >= gap for s in seat_xy)
                        and all(geo.dist2(p, s) >= gap for s in spots)
                        and all(geo.dist2(p, c[2]) >= gap - 1e-9 for c in chairs)):
                    chairs.append((wall, off, p))
                    if len(chairs) == need:
                        break
            if len(chairs) == need:
                break
        if len(chairs) < need:
            raise FitError("not enough wall space for the wall seats")

    scr = seating.screen
    screen_w = min(scr.width.sample(rng), W - 0.4)
    screen_c = rng.uniform(screen_w / 2 + 0.2, W - screen_w / 2 - 0.2)
    return Layout(L, W, H, shape, (cx, cy), tl, tw, th, pitch, slots, seat_off, table_seated, return_slots, chairs,
                  reason, n_wall, spots, screen_c, screen_w, seat_xy)


# ---------------------------------------------------------------- 3-5. people, voices, personas, the rest


@dataclass
class _P:
    kind: str  # table | wall | presenter | remote
    index: int  # slot / chair / presenter / remote index
    nsc: str = ""
    seniority: str = ""
    sid: str = ""


def _speaker_attrs(splits: SpeakerSplits, sid: str) -> NscSpeaker:
    s = splits.speakers[sid]
    return NscSpeaker(speaker=sid, split=s.split, gender=s.gender, age=s.age, ethnicity=s.ethnicity,
                      first_language=s.first_language, languages=list(s.languages))


def _persona(cfg: RunConfig, nsc: NscSpeaker, seniority: str, tercile: int, used_names: set[str],
             rng: np.random.Generator) -> Persona:
    pc = cfg.personas
    role = pc.roles[seniority][int(rng.integers(len(pc.roles[seniority])))]
    eth = nsc.ethnicity if nsc.ethnicity in pc.names else "OTHERS"
    given = pc.names[eth][nsc.gender]
    sur = pc.surnames[eth]
    name = ""
    for _ in range(20):
        name = f"{given[int(rng.integers(len(given)))]} {sur[int(rng.integers(len(sur)))]}"
        if name not in used_names:
            break
    used_names.add(name)
    l1 = pc.l1_languages.get(nsc.first_language, nsc.first_language.title())
    sheet_langs = {pc.l1_languages.get(x, x.title()) for x in nsc.languages if x not in ("ENGLISH", "OTHER")}
    if l1 != "English":
        sheet_langs.add(l1)
    # sheet languages (fixed order), then occasionally a dialect the ethnicity commonly mixes in
    switch = sorted(sheet_langs)
    for extra in pc.switch_languages.get(nsc.ethnicity, []):
        if extra in DIALECTS and extra not in switch and switch and rng.random() < 0.3:
            switch.append(extra)
    level = pc.code_switching.sample(rng) if switch else "low"
    verbosity = pc.verbosity_by_tercile[min(tercile, len(pc.verbosity_by_tercile) - 1)].sample(rng)
    accent = "Singapore English" if l1 == "English" else f"Singapore English, {l1} L1"
    return Persona(display_name=name, role=role, seniority=seniority, verbosity=verbosity, gender=nsc.gender,
                   age_band=nsc.age, ethnicity=nsc.ethnicity, l1=l1, accent=accent, code_switching=level,
                   switch_languages=switch)


def sample_meeting(cfg: RunConfig, splits: SpeakerSplits, meeting_id: str, seed: int,
                   rng: np.random.Generator, acoustic_rng: np.random.Generator | None = None) -> MeetingSpec:
    """`acoustic_rng` draws the acoustic values (`ACOUSTIC_FIELDS`); without it they come from `rng`, in the order of
    revision 1 runs."""
    arng = acoustic_rng or rng
    plan = draw_plan(cfg, rng)
    resamples = 0
    while True:
        try:
            lay = draw_layout(cfg, plan, rng)
            break
        except FitError:
            resamples += 1
            if resamples > cfg.room_classes.max_fit_resamples:
                raise RuntimeError(f"{meeting_id}: no layout fits after {resamples} draws ({plan})") from None
    room_class = plan.room_class
    seating, ac, dyn, tt = cfg.seating, cfg.acoustics, cfg.dynamics, cfg.turn_taking

    # people by position
    people = [_P("table", i) for i in lay.table_seated]
    people += [_P("wall", i) for i in range(lay.n_wall)]
    people += [_P("presenter", i) for i in range(plan.n_presenters)]
    people += [_P("remote", i) for i in range(plan.n_remote)]
    assert len(people) == plan.n

    # NSC voices from the meeting's split; persona seniority from the speaker's age band
    split = cfg.speaker_pool.split_mix.sample(rng)
    candidates = splits.in_split(split)
    if len(candidates) < plan.n:
        raise RuntimeError(f"split {split} has {len(candidates)} speakers, meeting needs {plan.n}")
    chosen = [candidates[int(i)] for i in rng.choice(len(candidates), size=plan.n, replace=False)]
    for p, sid in zip(people, chosen, strict=True):
        p.nsc = sid
        age = splits.speakers[sid].age
        p.seniority = cfg.personas.seniority_by_age[age].sample(rng) if age in cfg.personas.seniority_by_age else "mid"

    # chair: a table-seated person, weighted toward seniority; then rank everyone else by weighted Gumbel keys
    table_people = [p for p in people if p.kind == "table"]
    w = np.array([cfg.personas.chair_seniority.get(p.seniority, 0.5) for p in table_people], float)
    chair = table_people[int(rng.choice(len(table_people), p=w / w.sum()))]
    kind_weight = {"table": 1.0, "wall": dyn.wall_share_weight, "presenter": dyn.presenter_share_boost,
                   "remote": dyn.remote_share_weight}
    others = [p for p in people if p is not chair]
    keys = [math.log(kind_weight[p.kind]) + float(rng.gumbel()) for p in others]
    ranked = [chair] + [p for _, p in sorted(zip(keys, others, strict=True), key=lambda t: -t[0])]
    shares = tt.zipf.shares(plan.n)
    for rank, p in enumerate(ranked, start=1):
        p.sid = f"S{rank:02d}"

    # large rooms: the chair usually takes the head of the table, trading seats with whoever sat there. Seats kept
    # for presenters who come back are left alone.
    head = seating.chair_at_head
    if room_class == "large" and head.p > 0 and head.sample(rng):
        kept = {ref.index for ref in lay.return_slots.values() if ref.kind == "table"}
        heads = [i for i in geo.head_slots([s.xy for s in lay.slots], lay.table_center, lay.shape) if i not in kept]
        if heads and chair.index not in heads:
            to = heads[int(rng.integers(len(heads)))]
            sitter = next((p for p in table_people if p.index == to), None)
            if sitter is not None:
                sitter.index = chair.index
            chair.index = to

    # mic
    mc = ac.mic
    th = lay.table_h
    if room_class == "small":
        laptop_owner = table_people[int(rng.integers(len(table_people)))]
        slot = lay.slots[laptop_owner.index]
        mouth = lay.seat_xy[laptop_owner.index]
        inward = (-slot.normal[0], -slot.normal[1])
        xy = geo.offset(mouth, inward, mc.laptop_distance.sample(rng))
        mic = Mic(type="laptop", position=r3((xy[0], xy[1], th + mc.laptop_height.sample(rng))), channels=1,
                  array_geometry=[(0.0, 0.0, 0.0)], near_speaker=laptop_owner.sid)
    else:
        # the table mic: at the centre, or (some meetings) in front of the chair, to capture the chair's decisions
        at_chair = mc.chair_mic.p > 0 and mc.chair_mic.sample(rng)
        if at_chair:
            slot = lay.slots[chair.index]
            xy = geo.offset(lay.seat_xy[chair.index], (-slot.normal[0], -slot.normal[1]),
                            mc.chair_mic_distance.sample(rng))
        else:
            cx, cy = lay.table_center
            xy = (cx + mc.centre_offset.sample(rng), cy + mc.centre_offset.sample(rng))
        if mc.centre_array.sample(rng):
            k = mc.array_channels
            geom = [r3((mc.array_radius * math.cos(2 * math.pi * i / k), mc.array_radius * math.sin(2 * math.pi * i / k),
                        0.0)) for i in range(k)]
        else:
            k, geom = 1, [(0.0, 0.0, 0.0)]
        mic = Mic(type="chair" if at_chair else "centre", position=r3((xy[0], xy[1], th + mc.centre_height)),
                  channels=k, array_geometry=geom, near_speaker=chair.sid if at_chair else None)

    # positions, facing, mouths
    cxy = lay.table_center
    max_dev = ac.orientation_max_deg
    by_sid: dict[str, Person] = {}
    used_names: set[str] = set()
    pres = seating.presenters
    presenter_sids = [p.sid for p in people if p.kind == "presenter"]
    returns: dict[str, SeatRef | None] = {}
    for p in sorted(people, key=lambda q: q.sid):
        rank = int(p.sid[1:])
        tercile = min(2, (rank - 1) * 3 // plan.n)
        nsc = _speaker_attrs(splits, p.nsc)
        persona = _persona(cfg, nsc, p.seniority, tercile, used_names, rng)
        common = {"speaker_id": p.sid, "is_chair": p is chair, "turn_share": r(shares[rank - 1]), "nsc_speaker": nsc,
                  "persona": persona}
        if p.kind in ("table", "wall"):
            xy = lay.seat_xy[p.index] if p.kind == "table" else lay.wall_chairs[p.index][2]
            z = seating.seated_mouth_height.sample(rng)
            facing = geo.wrap_deg(geo.angle_deg(xy, cxy) + rng.uniform(-max_dev, max_dev))
            person = Person(position_type=p.kind, wall_reason=(lay.wall_reason if p.kind == "wall" else None),
                            seat=SeatRef(kind=p.kind, index=p.index), mouth=r3((xy[0], xy[1], z)),
                            facing_deg=r(facing), **common)
        elif p.kind == "presenter":
            xy = lay.presenter_spots[p.index]
            z = pres.mouth_height.sample(rng)
            ret = lay.return_slots.get(p.index)
            returns[p.sid] = ret
            spec = PresenterSpec(spot=r2(xy), drift_radius=r(pres.drift_radius.sample(rng)),
                                 facing_screen_p=pres.facing_screen_p, return_seat=ret)
            person = Person(position_type="presenter", mouth=r3((xy[0], xy[1], z)),
                            facing_deg=r(geo.angle_deg(xy, cxy)), presenter=spec, **common)
        else:
            person = Person(position_type="remote", mouth=None, facing_deg=None, **common)
        by_sid[p.sid] = person
    people_out = [by_sid[s] for s in sorted(by_sid)]
    slot_owner = {p.index: p.sid for p in people if p.kind == "table"}
    chair_owner = {p.index: p.sid for p in people if p.kind == "wall"}

    labels = _slot_labels(lay.slots)
    table = Table(shape=lay.shape, center=r2(lay.table_center), length=r(lay.table_l), width=r(lay.table_w),
                  height=r(th), seat_pitch=r(lay.pitch), capacity=len(lay.slots),
                  slots=[TableSlot(index=i, label=labels[i], side=s.side, edge_s=r(s.s), edge_xy=r2(s.xy),
                                   seat_xy=r2(lay.seat_xy[i]), occupied_by=slot_owner.get(i))
                         for i, s in enumerate(lay.slots)])
    wall_chairs = [WallChair(index=i, wall=c[0], offset=r(c[1]), xy=r2(c[2]), occupied_by=chair_owner.get(i))
                   for i, c in enumerate(lay.wall_chairs)]

    # room surfaces and acoustics
    walls = {w_: ac.wall_material.sample(arng) for w_ in ("N", "E", "S", "W")}
    presentation_on = plan.presentation
    screen = (Screen(wall=seating.front_wall, center=r(lay.screen_center), width=r(lay.screen_width),
                     bottom=seating.screen.bottom, top=min(seating.screen.top, lay.room_h - 0.2))
              if presentation_on else None)
    room = Room(length=r(lay.room_l), width=r(lay.room_w), height=r(lay.room_h), front_wall=seating.front_wall,
                walls=walls, floor="carpet" if ac.carpet.sample(arng) else "vinyl",
                ceiling="acoustic_tile" if ac.acoustic_tile.sample(arng) else "plasterboard",
                rt60_target=r(ac.rt60[room_class].sample(arng)), screen=screen)

    duration = dyn.duration_min[room_class].sample(rng)
    n_turns = max(20, int(round(duration * dyn.turns_per_min.sample(rng))))
    noise = []
    if ac.hvac.sample(arng):
        noise.append(NoiseSource(kind="hvac", position=r3((lay.room_l / 2, lay.room_w / 2, lay.room_h - 0.05))))
    if presentation_on and ac.projector_fan_if_presentation.sample(arng):
        noise.append(NoiseSource(kind="projector_fan",
                                 position=r3((min(2.0, lay.room_l / 2), lay.room_w / 2, lay.room_h - 0.3))))
    if ac.keyboard[room_class].sample(arng):
        if room_class == "small":
            kp = (mic.position[0], mic.position[1], th + 0.02)
        else:
            tp = by_sid[table_people[int(arng.integers(len(table_people)))].sid]
            assert tp.seat is not None
            edge = lay.slots[tp.seat.index].xy
            kp = (edge[0], edge[1], th + 0.02)
        noise.append(NoiseSource(kind="keyboard", position=r3(kp)))
    rate = ac.events_per_10min[room_class].sample(arng)
    n_events = int(arng.poisson(rate * duration / 10))
    events: dict[str, int] = {str(k): 0 for k in ac.event_types.weights}
    for _ in range(n_events):
        events[str(ac.event_types.sample(arng))] += 1
    acoustics = Acoustics(rt60=room.rt60_target, snr_db=r(ac.snr_db[room_class].sample(arng)), noise_sources=noise,
                          events=events, events_per_10min=r(rate), orientation_max_deg=ac.orientation_max_deg,
                          orientation_resample_p=ac.orientation_resample_p)

    dc = ac.device_chain
    chain = DeviceChain(agc=dc.agc.sample(arng),
                        band_limit_hz=r(dc.band_limit_hz.sample(arng)) if dc.band_limit.sample(arng) else None,
                        codec_kbps=int(dc.codec_kbps.sample(arng)) if dc.codec.sample(arng) else None,
                        clipping_level=r(dc.clipping_level.sample(arng)) if dc.clipping.sample(arng) else None,
                        aec_artifacts=bool(plan.hybrid and dc.aec_artifacts_if_hybrid.sample(arng)))

    remote_sids = sorted(p.sid for p in people if p.kind == "remote")
    speakerphone = None
    if plan.hybrid:
        if room_class == "small":
            pos = mic.position  # the laptop's own speaker
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            dx = min(mc.speakerphone_offset.sample(rng), max(lay.table_l / 2 - 0.2, 0.0))
            pos = r3((lay.table_center[0] + sign * dx, lay.table_center[1], th + 0.05))
        speakerphone = Speakerphone(position=pos, remote_speakers=remote_sids)

    segment = None
    if presentation_on:
        start = int(round(pres.start_frac.sample(rng) * n_turns))
        length = max(6, int(round(pres.length_frac.sample(rng) * n_turns)))
        segment = PresentationSegment(presenters=sorted(presenter_sids), start_turn=start,
                                      end_turn=min(n_turns, start + length), returns=dict(sorted(returns.items())))
    meeting_type = str(dyn.meeting_types[room_class].sample(rng))
    topic = dyn.topics[int(rng.integers(len(dyn.topics)))]
    dynamics = Dynamics(meeting_type=meeting_type, topic=topic, duration_min=r(duration), n_turns_target=n_turns,
                        zipf_exponent=tt.zipf.exponent, turn_shares={p.speaker_id: p.turn_share for p in people_out},
                        chair_id=chair.sid, presentation=segment, hybrid=plan.hybrid, remote_speakers=remote_sids)
    return MeetingSpec(meeting_id=meeting_id, seed=seed, split=split, room_class=room_class, n_speakers=plan.n,
                       fit_resamples=resamples, room=room, table=table, wall_chairs=wall_chairs, people=people_out,
                       mic=mic, speakerphone=speakerphone, acoustics=acoustics, device_chain=chain, dynamics=dynamics)


def _slot_labels(slots: list[geo.EdgePoint]) -> list[str]:
    counts: dict[str, int] = {}
    for s in slots:
        counts[s.side] = counts.get(s.side, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for s in slots:
        seen[s.side] = seen.get(s.side, 0) + 1
        prefix = "A" if s.side == "arc" else s.side
        out.append(prefix if counts[s.side] == 1 and s.side != "arc" else f"{prefix}{seen[s.side]}")
    return out


# ---------------------------------------------------------------- the stage


def run_meeting(cfg: RunConfig, splits: SpeakerSplits, meeting_id: str) -> MeetingSpec:
    """Stage 1 for one meeting, exactly as a run draws it. The meeting is a function of the run seed and the meeting
    id only, so attempt numbers never change it. From revision 2 the acoustic values come from a stream of their own,
    so changing an acoustic distribution leaves every other draw where it was."""
    from .runner import meeting_seed

    seed = meeting_seed(cfg.seed, meeting_id)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, seed, int(StageId.SAMPLE)]))
    arng = (np.random.default_rng(np.random.SeedSequence([cfg.seed, seed, int(StageId.SAMPLE), 1]))
            if cfg.revision >= 2 else None)
    return sample_meeting(cfg, splits, meeting_id, seed, rng, arng)


# Settings added after runs were frozen. A frozen config without one reads its default, which reproduces those
# runs, so the cache key leaves a setting out while it holds that default (with the settings it governs) and the
# older runs' stage 1 stays a cache hit.
ADDED_SETTINGS: tuple[tuple[tuple[str, ...], object, tuple[tuple[str, ...], ...]], ...] = (
    (("room_classes", "table_at_centre"), False, ()),
    (("seating", "chair_at_head"), {"dist": "bernoulli", "p": 0.0}, ()),
    (("acoustics", "mic", "chair_mic"), {"dist": "bernoulli", "p": 0.0}, (("acoustics", "mic", "chair_mic_distance"),)),
)


class Stage1Config(BaseModel):
    room_classes: RoomClassesConfig
    seating: SeatingConfig
    acoustics: AcousticsConfig
    turn_taking: TurnTakingConfig
    dynamics: DynamicsConfig
    personas: PersonasConfig
    split_mix: Categorical

    @model_serializer(mode="wrap")
    def _without_unused_additions(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)

        def parent(path: tuple[str, ...]) -> dict[str, Any]:
            node = data
            for k in path[:-1]:
                node = node[k]
            return node

        for path, default, governed in ADDED_SETTINGS:
            if parent(path).get(path[-1]) == default:
                for drop in (path, *governed):
                    parent(drop).pop(drop[-1], None)
        return data


class SampleStage:
    id = StageId.SAMPLE
    name = "Config"
    phase = Phase.CPU
    roles = ()
    reads = ("speaker_splits",)
    writes = ("meeting",)
    version = "1"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        return Stage1Config(room_classes=cfg.room_classes, seating=cfg.seating, acoustics=cfg.acoustics,
                            turn_taking=cfg.turn_taking, dynamics=cfg.dynamics, personas=cfg.personas,
                            split_mix=cfg.speaker_pool.split_mix)

    def run(self, ctx: StageContext) -> StageResult:
        splits = read_model(ctx.paths.artifact("speaker_splits"), SpeakerSplits)
        spec = run_meeting(ctx.cfg, splits, ctx.paths.meeting_id)
        write_model(ctx.paths.meeting, spec)
        parts = [f"{spec.n_speakers} speakers"]
        for kind in ("presenter", "wall", "remote"):
            if spec.count(kind):
                parts.append(f"{spec.count(kind)} {kind}")
        parts.append(f"{spec.fit_resamples} resamples")
        return StageResult(state="pass", detail=" · ".join(parts),
                           metrics={"n_speakers": spec.n_speakers, "fit_resamples": spec.fit_resamples,
                                    "room_class": spec.room_class, "split": spec.split})
