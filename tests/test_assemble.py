"""Stage 6: timeline placement, realised turn-taking rates, truncation and the dry tracks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from helpers import fake_run
from post_helpers import run_to_phase_c, synthetic_utts
from synthetics.artifacts.io import read_model
from synthetics.artifacts.labels import Timeline
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.models import Script
from synthetics.artifacts.paths import MeetingPaths
from synthetics.backends.base import Audio
from synthetics.backends.registry import Backends
from synthetics.stages.runner import run_stage
from synthetics.stages.s1_sample import sample_meeting
from synthetics.stages.s6_assemble import AssembleStage, Utt, assemble


def _meetings(cfg, splits, n, seed0=0):
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        m = sample_meeting(cfg, splits, f"m-{i + 1:06d}", seed0 + i, rng)
        utts = synthetic_utts(m, cfg, rng)
        tl, b = assemble(utts, m.meeting_id, m.room_class, cfg.sample_rate, cfg.turn_taking, cfg.assembly, rng,
                         planned_min=m.dynamics.duration_min)
        yield m, utts, tl, b


@pytest.fixture(scope="module")
def many(cfg, splits):
    return list(_meetings(cfg, splits, 80))


def test_realised_rates_fall_within_their_configured_ranges(cfg, many):
    tt = cfg.turn_taking
    overlap, bc, intr, missed = [], [], [], []
    for m, _, tl, b in many:
        s = tl.stats
        lo, hi = tt.overlap_target[m.room_class].lo, tt.overlap_target[m.room_class].hi
        assert lo - 0.005 <= s.overlap_share <= hi + 0.005, (m.meeting_id, s.overlap_share)
        if abs(s.overlap_share - s.overlap_target) > 0.005:
            # out of reach for this turn plan (few speaker changes, e.g. two people and a presentation): every
            # lever is used up
            missed.append(m.meeting_id)
            assert s.overlap_share < s.overlap_target and b.scale == cfg.assembly.max_lead_scale
            assert b.converted == b.convertible, m.meeting_id
        # one interruption either way is the rounding of a count over a short meeting; and about a tenth of the
        # planned interruptions cannot be placed (a floor turn too short to cut, or the interrupter still talking),
        # so a meeting drawn near the bottom of the range can land a little under it (m9-e2e: 2 of 20)
        slack = 1.5 / max(s.speaker_changes, 1)
        assert 0.8 * tt.interruption_rate.lo - slack <= s.interruption_share <= tt.interruption_rate.hi + slack, \
            m.meeting_id
        # a meeting shorter than planned is thinned to the planned rate; one longer than planned keeps every
        # backchannel, so its rate can only fall below the planned one
        assert s.backchannels_per_min <= tt.backchannels_per_min.hi + 1.0 / (s.duration / 60), m.meeting_id
        overlap.append(s.overlap_share)
        bc.append(s.backchannels_per_min)
        intr.append(s.interruption_share)
    assert len(missed) <= 0.05 * len(many), missed
    assert tt.backchannels_per_min.lo <= np.mean(bc) <= tt.backchannels_per_min.hi
    assert tt.interruption_rate.lo <= np.mean(intr) <= tt.interruption_rate.hi
    small = [o for (m, *_), o in zip(many, overlap, strict=True) if m.room_class == "small"]
    large = [o for (m, *_), o in zip(many, overlap, strict=True) if m.room_class == "large"]
    assert small and large and np.mean(small) < np.mean(large)


def test_no_speaker_overlaps_their_own_speech(cfg, many):
    for _, _, tl, _ in many:
        by = defaultdict(list)
        for s in tl.segments:
            by[s.speaker_id].append((s.start, s.end))
        for spans in by.values():
            spans.sort()
            for (_, e), (s, _) in zip(spans, spans[1:], strict=False):
                assert s >= e + cfg.assembly.min_self_gap_s - 1e-3


def test_every_turn_is_placed_once_and_segments_are_sorted(many):
    for _, utts, tl, _ in many:
        ids = [s.turn_id for s in tl.segments] + tl.dropped_turns
        assert sorted(ids) == sorted(u.turn_id for u in utts)
        assert all(utts[int(t[1:]) - 1].kind == "backchannel" for t in tl.dropped_turns)
        assert [s.start for s in tl.segments] == sorted(s.start for s in tl.segments)
        assert tl.stats.duration >= max(s.end for s in tl.segments)


def test_interruptions_cut_the_floor_speaker_inside_the_interrupting_turn(cfg, many):
    seen = 0
    for _, utts, tl, _ in many:
        by_id = {s.turn_id: s for s in tl.segments}
        dur = {u.turn_id: u.dur for u in utts}
        main = [u for u in utts if u.kind not in ("backchannel", "side")]
        for a, nxt in zip(main, main[1:], strict=False):
            seg = by_id[a.turn_id]
            if seg.truncated_at is None:
                continue
            seen += 1
            ib = by_id[nxt.turn_id]
            assert nxt.kind == "interruption"
            assert seg.end - seg.start == pytest.approx(seg.truncated_at, abs=1e-9)
            assert seg.truncated_at < dur[a.turn_id] - cfg.assembly.interrupt_min_tail_s + 1e-3
            cut_after = seg.end - ib.start  # how long the floor speaker kept going
            assert 0.15 <= cut_after <= cfg.turn_taking.interruption_cutoff.hi + 1e-3
        assert tl.stats.interruptions == sum(s.truncated_at is not None for s in tl.segments)
    assert seen > 50


def test_backchannels_sit_inside_another_speakers_turn(many):
    inside = total = 0
    for _, _, tl, _ in many:
        main = [s for s in tl.segments if s.kind not in ("backchannel", "side")]
        for s in tl.segments:
            if s.kind != "backchannel":
                continue
            total += 1
            inside += any(h.speaker_id != s.speaker_id and h.start <= s.start and s.end <= h.end for h in main)
    assert inside / total > 0.9


def test_same_seed_gives_the_same_timeline(cfg, splits):
    a = [tl for *_, tl, _ in _meetings(cfg, splits, 3, seed0=7)]
    b = [tl for *_, tl, _ in _meetings(cfg, splits, 3, seed0=7)]
    assert [x.model_dump_json() for x in a] == [x.model_dump_json() for x in b]


def test_overlap_control_off_leaves_the_raw_statistics(cfg):
    from synthetics.config.schema import AssemblyConfig

    utts = [Utt(f"t{i:04d}", f"S0{1 + i % 2}", "overlap_start" if i else "normal", 3.0) for i in range(1, 21)]
    rng = np.random.default_rng(0)
    tl, b = assemble(utts, "m", "large", 16000, cfg.turn_taking, AssemblyConfig(overlap_control=False), rng)
    assert b.scale == 1.0 and b.converted == 0
    # every transition overlaps by its drawn lead, capped at half the 3 s turns
    lead = cfg.turn_taking.overlap_lead
    leads = [a.end - n.start for a, n in zip(tl.segments, tl.segments[1:], strict=False)]
    assert all(lead.lo - 1e-3 <= x <= min(lead.hi, 1.5) + 1e-3 for x in leads)


# ---------------------------------------------------------------- the stage on a fake run


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    rd, cfg = fake_run(tmp_path_factory.mktemp("s6"), n=2)
    ids = run_to_phase_c(rd, cfg)
    outs = [run_stage(AssembleStage(), rd, mid, cfg, Backends(cfg.backends)) for mid in ids]
    return rd, cfg, ids, outs


def test_stage6_writes_the_timeline_and_one_track_per_speaker(assembled):
    rd, cfg, ids, outs = assembled
    assert ids and [o.state for o in outs] == ["pass"] * len(ids)
    for mid in ids:
        p = MeetingPaths(rd, mid)
        m = read_model(p.meeting, MeetingSpec)
        tl = read_model(p.timeline, Timeline)
        assert sorted(f.stem for f in p.tracks_dir.glob("*.wav")) == sorted(x.speaker_id for x in m.people)
        n = int(round(tl.stats.duration * cfg.sample_rate))
        for spk in (x.speaker_id for x in m.people):
            a = Audio.load(p.track(spk))
            assert a.sr == cfg.sample_rate and len(a.samples) == n


def test_tracks_hold_each_utterance_at_its_segment_and_cut_at_truncated_at(assembled):
    rd, cfg, ids, _ = assembled
    sr = cfg.sample_rate
    checked_trunc = 0
    for mid in ids:
        p = MeetingPaths(rd, mid)
        tl = read_model(p.timeline, Timeline)
        tracks = {}
        for seg in tl.segments:
            if seg.speaker_id not in tracks:
                tracks[seg.speaker_id] = Audio.load(p.track(seg.speaker_id)).samples
            x = tracks[seg.speaker_id]
            u = Audio.load(p.utterance(seg.turn_id)).samples
            s0, n = int(round(seg.start * sr)), int(round((seg.end - seg.start) * sr))
            body = n - int(cfg.assembly.fade_s * sr) if seg.truncated_at is not None else n
            assert np.allclose(x[s0:s0 + body], u[:body], atol=1e-4)
            if seg.truncated_at is not None:
                checked_trunc += 1
                assert n < len(u)
                after = x[s0 + n: s0 + n + int(cfg.assembly.min_self_gap_s * sr)]
                assert np.all(after == 0), "nothing of the cut-off utterance is left after truncated_at"
    assert checked_trunc >= 1


def test_stage6_is_a_cache_hit_the_second_time(assembled):
    rd, cfg, ids, _ = assembled
    out = run_stage(AssembleStage(), rd, ids[0], cfg, Backends(cfg.backends))
    assert out.state == "cached"


def test_timeline_follows_the_script_turns(assembled):
    rd, _, ids, _ = assembled
    p = MeetingPaths(rd, ids[0])
    script = read_model(p.script, Script)
    tl = read_model(p.timeline, Timeline)
    kinds = {t.turn_id: (t.speaker_id, t.interaction) for t in script.turns}
    assert all(kinds[s.turn_id] == (s.speaker_id, s.kind) for s in tl.segments)
