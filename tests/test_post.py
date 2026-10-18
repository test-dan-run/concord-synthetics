"""Stages 7 and 8 on fake runs, the phase-post loop, the manifest and the cache behaviour of stages 6-8."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from helpers import fake_run
from post_helpers import run_to_phase_c
from synthetics.artifacts.io import file_sha256, read_model
from synthetics.artifacts.labels import AcousticRender, ManifestEntry, Timeline, Transcript
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.paths import MeetingPaths, RunPaths
from synthetics.backends.base import WordTiming
from synthetics.backends.fakes_audio import FakeAligner
from synthetics.backends.registry import Backends
from synthetics.config.schema import Phase, Role, StageId
from synthetics.export.rttm import read_rttm
from synthetics.export.transcript import cut_text
from synthetics.orchestrator.lease import GpuLease
from synthetics.orchestrator.phase_post import post_stages, run_post
from synthetics.orchestrator.scheduler import PhaseScheduler, RunReport
from synthetics.orchestrator.status import StatusStore
from synthetics.stages.runner import run_stage

# hybrid meetings with presenters and a centre array, so one small run covers the remote and array paths
RICH = {"acoustics.hybrid": {"small": {"dist": "bernoulli", "p": 1.0}, "large": {"dist": "bernoulli", "p": 1.0}},
        "seating.presenters.presentation": {"small": {"dist": "bernoulli", "p": 1.0},
                                            "large": {"dist": "bernoulli", "p": 1.0}},
        "acoustics.mic.centre_array": {"dist": "bernoulli", "p": 1.0}}


@pytest.fixture(scope="module")
def post(tmp_path_factory):
    rd, cfg = fake_run(tmp_path_factory.mktemp("post"), n=3, seed=5, **RICH)
    run_to_phase_c(rd, cfg)
    leases = []

    def factory(services, **kw):
        leases.append(tuple(services))
        return GpuLease(services, **{**kw, "enabled": False})

    sched = PhaseScheduler(rd, cfg, lease_factory=factory)
    report = RunReport()
    run_post(sched, RunPaths(rd).meeting_ids(), report)
    return rd, cfg, report, leases


def _paths(rd):
    return [RunPaths(rd).meeting(m) for m in RunPaths(rd).meeting_ids()]


def test_every_meeting_passes_stages_6_to_8(post):
    rd, _, report, leases = post
    states = {(o.meeting_id, int(o.stage)): o.state for o in report.outcomes}
    ids = RunPaths(rd).meeting_ids()
    assert all(states[(m, s)] == "pass" for m in ids for s in (6, 7, 8)), states
    assert leases == [("qwen3-forced-aligner",)], "one phase D lease, for the aligner only"


def test_rt60_measured_on_the_impulse_responses_is_within_the_target(post):
    rd, cfg, _, _ = post
    for p in _paths(rd):
        r = read_model(p.render, AcousticRender)
        assert abs(r.rt60_error) <= cfg.simulation.rt60_tolerance, (p.meeting_id, r.rt60_measured, r.rt60_target)
        assert r.rt60_measured == pytest.approx(np.median(list(r.rt60_per_speaker.values())), abs=1e-3)


def test_mixtures_have_the_timeline_length_and_the_mic_channels(post):
    rd, cfg, _, _ = post
    seen_array = seen_remote = False
    for p in _paths(rd):
        m = read_model(p.meeting, MeetingSpec)
        tl = read_model(p.timeline, Timeline)
        r = read_model(p.render, AcousticRender)
        info = sf.info(str(p.mixture))
        assert info.channels == 1 and info.samplerate == cfg.sample_rate
        assert info.frames == int(round(tl.stats.duration * cfg.sample_rate))
        assert r.channels == m.mic.channels
        if m.mic.channels > 1:
            seen_array = True
            mc, _ = sf.read(str(p.mixture_mc), dtype="float32")
            assert mc.shape == (info.frames, m.mic.channels)
            assert not np.allclose(mc[:, 0], mc[:, 1]), "capsules hear different signals"
        else:
            assert not p.mixture_mc.exists()
        if m.dynamics.remote_speakers:
            seen_remote = True
            assert set(r.remote) == set(m.dynamics.remote_speakers)
        assert r.snr_db == pytest.approx(m.acoustics.snr_db, abs=0.05)
    assert seen_array and seen_remote


def test_speech_in_the_mixture_follows_the_timeline(post):
    rd, cfg, _, _ = post
    sr = cfg.sample_rate
    for p in _paths(rd):
        tl = read_model(p.timeline, Timeline)
        x, _ = sf.read(str(p.mixture), dtype="float32")
        hop = sr // 10
        n = len(x) // hop
        e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
        voiced = np.zeros(n, bool)
        for s in tl.segments:
            voiced[int(s.start * 10): int(s.end * 10)] = True
        assert voiced.any() and (~voiced).any()
        assert np.median(e[voiced]) > np.median(e[~voiced]) + 3.0, p.meeting_id
        assert np.abs(x).max() <= 1.0
        # levels differ by distance and orientation, but no talker (the speakerphone included) drowns the rest
        talker = {}
        for s in tl.segments:
            if s.kind not in ("backchannel", "side") and s.end - s.start > 1.0:
                talker.setdefault(s.speaker_id, []).append(10 * np.log10(np.mean(x[int(s.start * sr):int(s.end * sr)] ** 2)))
        med = [np.median(v) for v in talker.values()]
        assert max(med) - min(med) < 20.0, (p.meeting_id, {k: round(float(np.median(v)), 1) for k, v in talker.items()})


def test_rttm_matches_the_timeline(post):
    rd, _, _, _ = post
    for p in _paths(rd):
        tl = read_model(p.timeline, Timeline)
        turns = read_rttm(p.rttm)
        assert len(turns) == len(tl.segments)
        assert {t.file_id for t in turns} == {tl.meeting_id}
        assert sum(t.duration for t in turns) == pytest.approx(sum(s.end - s.start for s in tl.segments), abs=0.002 * len(turns))
        assert [t.start for t in turns] == sorted(t.start for t in turns)


def test_transcript_words_sit_inside_their_segments_and_truncation_cuts_the_text(post):
    rd, cfg, _, _ = post
    slack = cfg.thresholds.align_slack_s
    truncated = 0
    for p in _paths(rd):
        tl = read_model(p.timeline, Timeline)
        tr = read_model(p.transcript, Transcript)
        segs = {s.turn_id: s for s in tl.segments}
        assert [u.turn_id for u in tr.utterances] == [s.turn_id for s in tl.segments]
        for u in tr.utterances:
            s = segs[u.turn_id]
            assert all(s.start - slack <= w.start <= w.end <= s.end + slack for w in u.words)
            assert u.align_coverage >= cfg.thresholds.align_coverage_min
            if u.truncated:
                truncated += 1
                assert u.text.endswith("-") and s.truncated_at is not None
                assert len(u.text_norm.split()) == len(u.words)
                assert u.words[-1].end <= s.start + s.truncated_at + 1e-4
    assert truncated > 0


def test_manifest_has_one_complete_line_per_meeting(post):
    rd, cfg, _, _ = post
    lines = RunPaths(rd).manifest.read_text().splitlines()
    assert len(lines) == len(RunPaths(rd).meeting_ids())
    for line in lines:
        e = ManifestEntry.model_validate(json.loads(line))
        p = RunPaths(rd).meeting(e.meeting_id)
        m = read_model(p.meeting, MeetingSpec)
        assert e.n_speakers == m.n_speakers == len(e.speakers)
        assert sum(s.turn_share for s in e.speakers) == pytest.approx(1.0, abs=0.01)
        assert {s.position_type for s in e.speakers} == {pp.position_type for pp in m.people}
        assert sum(e.position_counts.values()) == m.n_speakers
        assert e.stages_enabled == {s.name.lower(): True for s in StageId}
        for key in ("room.length", "room.rt60_target", "mic.type", "acoustics.snr_db", "device_chain.agc",
                    "dynamics.hybrid", "fit_resamples"):
            assert key in e.sampled
        assert e.acoustic["rt60_measured"] and e.labels["aligner"] == "qwen3-forced-aligner"
        assert e.judge["primary_overall"] is not None and e.turn_taking.overlap_share > 0
        for rel in e.files.values():
            assert (rd / rel).exists(), rel


def test_a_second_post_run_is_all_cache_hits_with_no_lease(post):
    rd, cfg, _, _ = post
    leases = []

    def factory(services, **kw):
        leases.append(tuple(services))
        return GpuLease(services, **{**kw, "enabled": False})

    report = RunReport()
    run_post(PhaseScheduler(rd, cfg, lease_factory=factory), RunPaths(rd).meeting_ids(), report)
    assert {o.state for o in report.outcomes} == {"cached"} and leases == []


def test_stage8_defers_without_the_aligner_and_gates_bad_alignments(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1, seed=3)
    run_to_phase_c(rd, cfg)
    mid = RunPaths(rd).meeting_ids()[0]
    s6, s7, s8 = (post_stages()[s] for s in (StageId.ASSEMBLE, StageId.ACOUSTIC, StageId.LABELS))
    b = Backends(cfg.backends)
    assert run_stage(s6, rd, mid, cfg, b).state == "pass"
    assert run_stage(s7, rd, mid, cfg, b).state == "pass"

    out = run_stage(s8, rd, mid, cfg, b.with_available([]))
    assert out.state == "defer" and StatusStore(rd).get(mid).get(StageId.LABELS).state == "pending"

    class Dropping(FakeAligner):
        def align(self, audio, text, *, language):
            return super().align(audio, text, language=language)[::2]

    class Late(FakeAligner):
        def align(self, audio, text, *, language):
            return [WordTiming(w.word, w.start + 2.0, w.end + 2.0, w.score)
                    for w in super().align(audio, text, language=language)]

    out = run_stage(s8, rd, mid, cfg, Backends(cfg.backends, overrides={Role.ALIGNER: Dropping()}), force=True)
    assert out.state == "fail" and "coverage" in out.detail
    out = run_stage(s8, rd, mid, cfg, Backends(cfg.backends, overrides={Role.ALIGNER: Late()}), force=True)
    assert out.state == "fail" and "outside their segment" in out.detail
    assert run_stage(s8, rd, mid, cfg, b, force=True).state == "pass"


def test_changing_render_settings_reruns_only_stages_7_and_8(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1, seed=3)
    run_to_phase_c(rd, cfg)
    sched = PhaseScheduler(rd, cfg)
    ids = RunPaths(rd).meeting_ids()
    run_post(sched, ids, RunReport())
    p = MeetingPaths(rd, ids[0])
    before = file_sha256(p.mixture)

    def rerun(**sim) -> dict[int, str]:
        c = cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update=sim)})
        report = RunReport()
        run_post(PhaseScheduler(rd, c), ids, report)
        upstream = PhaseScheduler(rd, c).run()  # stages 1-3 are untouched
        assert {o.state for o in upstream.outcomes} == {"cached"}
        return {int(o.stage): o.state for o in report.outcomes}

    # a room change shows in render.json, so the labels and manifest are rebuilt
    assert rerun(directivity_p=0.6) == {6: "cached", 7: "pass", 8: "pass"}
    assert file_sha256(p.mixture) != before
    # a global source level is normalised away: stage 7 runs, but nothing stage 8 reads or reports changes
    assert rerun(directivity_p=0.6, speech_level_db=-18.0) == {6: "cached", 7: "pass"}  # stage 8 fresh: not run
    assert StatusStore(rd).get(ids[0]).get(StageId.LABELS).state == "pass"


@pytest.mark.xfail(strict=True, reason="stage 1 samples acoustics into meeting.json, which stages 2-6 hash whole; "
                                        "needs the upstream change described in docs/OUTPUT_CONTRACT.md")
def test_changing_sampled_acoustics_reruns_only_stages_7_and_8(tmp_path):
    from synthetics.config.distributions import Uniform

    rd, cfg = fake_run(tmp_path, n=1, seed=3)
    PhaseScheduler(rd, cfg).run()
    snr = cfg.acoustics.snr_db.model_copy(update={"small": Uniform(lo=0, hi=5), "large": Uniform(lo=0, hi=5)})
    cfg2 = cfg.model_copy(update={"acoustics": cfg.acoustics.model_copy(update={"snr_db": snr})})
    report = PhaseScheduler(rd, cfg2).run()
    assert not any(o.stage == StageId.SCRIPT for o in report.outcomes), "stage 2 ran again"


def test_stage7_is_reproducible(tmp_path):
    rd, cfg = fake_run(tmp_path, n=1, seed=3, **RICH)
    run_to_phase_c(rd, cfg)
    mid = RunPaths(rd).meeting_ids()[0]
    b = Backends(cfg.backends)
    s6, s7 = post_stages()[StageId.ASSEMBLE], post_stages()[StageId.ACOUSTIC]
    run_stage(s6, rd, mid, cfg, b)
    p = MeetingPaths(rd, mid)
    run_stage(s7, rd, mid, cfg, b)
    first = (file_sha256(p.mixture), file_sha256(p.render))
    run_stage(s7, rd, mid, cfg, b, force=True)
    assert (file_sha256(p.mixture), file_sha256(p.render)) == first


def test_only_phase_d_runs_stage_8_alone(post):
    rd, cfg, _, _ = post
    report = RunReport()
    run_post(PhaseScheduler(rd, cfg), RunPaths(rd).meeting_ids(), report, only=Phase.D)
    assert {int(o.stage) for o in report.outcomes} <= {8}


def test_cut_text_keeps_the_heard_words_with_a_hyphen():
    t = "Ok so the vendor (lah), cannot make it [laugh] by Friday."
    assert cut_text(t, 5) == "Ok so the vendor (lah)-"
    assert cut_text(t, 1) == "Ok-"
    assert cut_text("We should-", 2) == "We should-"
    assert cut_text(t, 0) == ""
