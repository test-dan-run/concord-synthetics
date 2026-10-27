"""Run coverage: the in-memory stage 1 sample matches what a run writes, and the seed search."""

from __future__ import annotations

from helpers import fake_run
from synthetics.artifacts.io import read_model
from synthetics.artifacts.meeting import MeetingSpec
from synthetics.artifacts.paths import RunPaths
from synthetics.config.schema import StageId
from synthetics.orchestrator.scheduler import PhaseScheduler
from synthetics.validators.coverage import coverage, find_seed, sample_meetings


def test_sampled_meetings_match_the_run(tmp_path):
    rd, cfg = fake_run(tmp_path, n=4, seed=9)
    PhaseScheduler(rd, cfg).run(until=StageId.SAMPLE)
    rp = RunPaths(rd)
    written = [read_model(rp.meeting(m).meeting, MeetingSpec) for m in rp.meeting_ids()]
    assert [m.model_dump() for m in sample_meetings(cfg, 4, 9)] == [m.model_dump() for m in written]


def test_find_seed_covers_the_list(tmp_path):
    _, cfg = fake_run(tmp_path, n=20)
    seed, cov = find_seed(cfg, 20, tries=200)
    assert seed is not None and all(cov.values())
    assert coverage(sample_meetings(cfg, 20, seed)) == cov
