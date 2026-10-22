"""M1: stage 1 meeting sampler — property, determinism and distribution tests."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from conftest import make_cfg
from synthetics.acoustics import geometry as geo
from synthetics.artifacts.io import write_model
from synthetics.artifacts.paths import RunPaths
from synthetics.backends.registry import Backends
from synthetics.config.loader import dump_run_config
from synthetics.stages.runner import meeting_id_for, run_stage
from synthetics.stages.s1_sample import SampleStage, sample_meeting
from synthetics.validators.sampler_report import render, sampler_checks

N_PROPERTY = 600


@pytest.fixture(scope="module")
def meetings(cfg, splits):
    return [sample_meeting(cfg, splits, meeting_id_for(i), i, np.random.default_rng(i)) for i in range(N_PROPERTY)]


def in_room_xy(m):
    return [p.mouth[:2] for p in m.people if p.mouth is not None]


def test_every_position_is_inside_the_room(meetings):
    for m in meetings:
        for xy in in_room_xy(m):
            assert geo.inside_room(xy, m.room.length, m.room.width, 0.2), (m.meeting_id, xy)
        for s in m.table.slots:
            assert geo.inside_room(s.seat_xy, m.room.length, m.room.width, 0.2)


def test_table_keeps_wall_clearance(cfg, meetings):
    c = cfg.room_classes.wall_clearance
    for m in meetings:
        t = m.table
        cx, cy = t.center
        gaps = [cx - t.length / 2, m.room.length - cx - t.length / 2, cy - t.width / 2, m.room.width - cy - t.width / 2]
        assert min(gaps) >= c - 1e-3, (m.meeting_id, gaps)


def test_seat_pitch_within_range(cfg, meetings):
    lo, hi = cfg.room_classes.seat_pitch.lo, cfg.room_classes.seat_pitch.hi
    for m in meetings:
        assert lo - 1e-3 <= m.table.seat_pitch <= hi + 1e-3
        by_side: dict[str, list[float]] = {}
        for s in m.table.slots:
            by_side.setdefault(s.side, []).append(s.edge_s)
        for side, ss in by_side.items():
            gaps = np.diff(sorted(ss))
            assert all(lo - 1e-3 <= g <= hi + 1e-3 for g in gaps), (m.meeting_id, side, gaps)


def test_mouth_heights_within_range(cfg, meetings):
    sz, pz = cfg.seating.seated_mouth_height, cfg.seating.presenters.mouth_height
    for m in meetings:
        for p in m.people:
            if p.position_type in ("table", "wall"):
                assert sz.lo - 1e-3 <= p.mouth[2] <= sz.hi + 1e-3
            elif p.position_type == "presenter":
                assert pz.lo - 1e-3 <= p.mouth[2] <= pz.hi + 1e-3
            else:
                assert p.mouth is None


def test_people_keep_a_minimum_gap(cfg, meetings):
    gap = cfg.room_classes.min_person_gap
    for m in meetings:
        for a, b in itertools.combinations(in_room_xy(m), 2):
            assert geo.dist2(a, b) >= gap - 1e-3, m.meeting_id


def test_small_rooms_never_use_wall_seats(meetings):
    for m in meetings:
        if m.room_class == "small":
            assert m.count("wall") == 0 and not m.wall_chairs
            assert m.n_speakers <= 6 and m.mic.type == "laptop"
        else:
            assert m.n_speakers >= 7 and m.mic.type in ("centre", "chair")


def test_wall_seats_and_reasons(cfg, meetings):
    off = cfg.seating.wall.offset
    for m in meetings:
        wall_people = [p for p in m.people if p.position_type == "wall"]
        for p in wall_people:
            assert p.wall_reason in ("overflow", "by_choice")
            chair = m.wall_chairs[p.seat.index]
            assert chair.occupied_by == p.speaker_id and chair.wall in cfg.seating.wall.walls
            assert off.lo - 1e-3 <= chair.offset <= off.hi + 1e-3
        if any(p.wall_reason == "overflow" for p in wall_people):
            assert all(s.occupied_by for s in m.table.slots), "overflow only when the table is full"
        if any(p.wall_reason == "by_choice" for p in wall_people):
            assert any(s.occupied_by is None for s in m.table.slots)


def test_presenters(cfg, meetings):
    d = cfg.seating.presenters.distance_from_front
    for m in meetings:
        pres = [p for p in m.people if p.position_type == "presenter"]
        seg = m.dynamics.presentation
        assert (seg is not None) == bool(pres)
        if not pres:
            assert m.room.screen is None
            continue
        assert m.room.screen is not None and m.room.screen.wall == "W"
        assert sorted(seg.presenters) == sorted(p.speaker_id for p in pres)
        assert 0 <= seg.start_turn < seg.end_turn <= m.dynamics.n_turns_target
        for p in pres:
            assert d.lo - 1e-3 <= p.presenter.spot[0] <= d.hi + 1e-3
            assert p.presenter.drift_radius <= 1.0 + 1e-9
            ret = p.presenter.return_seat
            if ret is not None and ret.kind == "table":
                assert m.table.slots[ret.index].occupied_by is None, "return seat must be free"
            if ret is not None and ret.kind == "wall":
                assert m.room_class == "large" and m.wall_chairs[ret.index].occupied_by is None


def test_voices_come_from_the_meeting_split_and_personas_match(splits, meetings):
    for m in meetings:
        ids = [p.nsc_speaker.speaker for p in m.people]
        assert len(set(ids)) == len(ids)
        for p in m.people:
            row = splits.speakers[p.nsc_speaker.speaker]
            assert row.split == m.split == p.nsc_speaker.split
            assert p.persona.gender == row.gender
            assert p.persona.ethnicity == row.ethnicity
            assert p.persona.age_band == row.age
            if row.first_language != "ENGLISH":
                assert p.persona.l1 != "English"
        names = [p.persona.display_name for p in m.people]
        assert len(set(names)) == len(names)


def test_turn_shares_and_chair(cfg, meetings):
    for m in meetings:
        shares = m.dynamics.turn_shares
        assert abs(sum(shares.values()) - 1) < 1e-3
        expected = cfg.turn_taking.zipf.shares(m.n_speakers)
        assert [round(shares[f"S{i + 1:02d}"], 4) for i in range(m.n_speakers)] == [round(x, 4) for x in expected]
        chair = m.person(m.dynamics.chair_id)
        assert chair.is_chair and chair.speaker_id == "S01" and chair.position_type == "table"


def test_mic_placement(cfg, meetings):
    mc = cfg.acoustics.mic
    for m in meetings:
        if m.mic.type == "laptop":
            owner = m.person(m.mic.near_speaker)
            d = geo.dist2(owner.mouth[:2], m.mic.position[:2])
            assert mc.laptop_distance.lo - 1e-3 <= d <= mc.laptop_distance.hi + 1e-3
            assert geo.point_table_distance(m.mic.position[:2], m.table.shape, m.table.center, m.table.length,
                                            m.table.width) == 0.0, "laptop sits on the table"
        else:
            assert m.room_class == "large"
            assert math.isclose(m.mic.position[2], m.table.height + mc.centre_height, abs_tol=1e-3)
            assert m.mic.channels in (1, mc.array_channels) and len(m.mic.array_geometry) == m.mic.channels
        if m.mic.type == "chair":
            assert m.mic.near_speaker == m.dynamics.chair_id
            d = geo.dist2(m.person(m.dynamics.chair_id).mouth[:2], m.mic.position[:2])
            assert mc.chair_mic_distance.lo - 1e-3 <= d <= mc.chair_mic_distance.hi + 1e-3
            assert geo.point_table_distance(m.mic.position[:2], m.table.shape, m.table.center, m.table.length,
                                            m.table.width) == 0.0, "the chair mic sits on the table"
        elif m.mic.type == "centre":
            assert m.mic.near_speaker is None
        if m.dynamics.hybrid:
            assert m.speakerphone is not None and m.speakerphone.remote_speakers == m.dynamics.remote_speakers
            assert m.count("remote") == len(m.dynamics.remote_speakers) > 0


def test_table_sits_at_the_room_centre(cfg, meetings):
    assert cfg.room_classes.table_at_centre
    for m in meetings:
        assert math.isclose(m.table.center[0], m.room.length / 2, abs_tol=1e-4), m.meeting_id
        assert math.isclose(m.table.center[1], m.room.width / 2, abs_tol=1e-4), m.meeting_id


def test_head_of_the_table():
    rect = geo.rect_slots((3.0, 2.0), 3.0, 1.4, 0.65)  # two seats at each end
    heads = geo.head_slots([s.xy for s in rect], (3.0, 2.0), "rect")
    assert {rect[i].side for i in heads} == {"E", "W"} and len(heads) == 4
    oval = geo.stadium_slots((3.0, 2.0), 3.0, 1.4, 12)
    heads = geo.head_slots([s.xy for s in oval], (3.0, 2.0), "oval")
    assert {oval[i].xy[0] < 3.0 for i in heads} == {True, False}  # both ends
    assert all(abs(oval[i].xy[0] - 3.0) > 0.8 for i in heads), "on the end arcs, past the straight sides"
    round_ = geo.ellipse_slots((2.0, 2.0), 1.2, 1.2, 6)
    assert geo.head_slots([s.xy for s in round_], (2.0, 2.0), "round") == []


def test_chair_takes_the_head_of_large_tables(cfg, splits):
    always = make_cfg(**{"seating.chair_at_head": {"dist": "bernoulli", "p": 1.0}})
    for i in range(200):
        m = sample_meeting(always, splits, meeting_id_for(i), i, np.random.default_rng(i))
        if m.room_class != "large":
            continue
        chair = m.person(m.dynamics.chair_id)
        heads = geo.head_slots([s.edge_xy for s in m.table.slots], m.table.center, m.table.shape)
        kept = {r.index for r in (m.dynamics.presentation.returns.values() if m.dynamics.presentation else [])
                if r is not None and r.kind == "table"}
        if set(heads) - kept:
            assert chair.seat.index in heads, m.meeting_id
        # still one person per seat, and the slot owners agree with the people
        owners = {s.index: s.occupied_by for s in m.table.slots if s.occupied_by}
        assert owners == {p.seat.index: p.speaker_id for p in m.people if p.position_type == "table"}


def test_settings_added_later_stay_out_of_older_runs_cache_keys(cfg):
    # an older frozen config reads the added settings at their defaults; its stage 1 slice must not change
    off = {"dist": "bernoulli", "p": 0.0}
    old = make_cfg(**{"room_classes.table_at_centre": False, "seating.chair_at_head": off,
                      "acoustics.mic.chair_mic": off})
    old_slice = SampleStage().config_slice(old).model_dump(mode="json")
    assert "table_at_centre" not in old_slice["room_classes"] and "chair_at_head" not in old_slice["seating"]
    assert not {"chair_mic", "chair_mic_distance"} & set(old_slice["acoustics"]["mic"])
    new_slice = SampleStage().config_slice(cfg).model_dump(mode="json")
    assert new_slice["room_classes"]["table_at_centre"] is True
    assert new_slice["acoustics"]["mic"]["chair_mic_distance"] == {"dist": "uniform", "lo": 0.4, "hi": 0.6}


def test_same_seed_gives_byte_identical_meeting_json(cfg, splits, tmp_path):
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    for run in (run_a, run_b):
        rp = RunPaths(run)
        dump_run_config(cfg, rp.config)
        write_model(rp.speaker_splits, splits)
        out = run_stage(SampleStage(), run, "m-000007", cfg, Backends(cfg.backends))
        assert out.state == "pass", out.detail
    a = (run_a / "meetings/m-000007/meeting.json").read_bytes()
    b = (run_b / "meetings/m-000007/meeting.json").read_bytes()
    assert a == b
    # a second run is a cache hit and leaves the file untouched
    out = run_stage(SampleStage(), run_a, "m-000007", cfg, Backends(cfg.backends))
    assert out.state == "cached"
    assert (run_a / "meetings/m-000007/meeting.json").read_bytes() == a


def test_different_meetings_differ(meetings):
    assert len({m.model_dump_json() for m in meetings[:50]}) == 50


@pytest.mark.parametrize("n", [10_000])
def test_distributions_match_targets(cfg, splits, n):
    ms = [sample_meeting(cfg, splits, meeting_id_for(i), i, np.random.default_rng(10_000 + i)) for i in range(n)]
    checks = sampler_checks(cfg, ms)
    failed = [c for c in checks if not c.ok]
    assert not failed, render(checks)
