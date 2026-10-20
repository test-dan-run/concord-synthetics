"""M11 ablations: switching off stage 3, 5 or 7 skips it cleanly, the meeting still reaches the manifest, and the
manifest records the toggle. Stages 1, 2, 4, 6 and 8 write what every later stage needs, so they have no ablation."""

from __future__ import annotations

import json

import pytest

from helpers import fake_run
from synthetics.artifacts.labels import ManifestEntry
from synthetics.artifacts.paths import RunPaths
from synthetics.config.schema import StageId
from synthetics.orchestrator.lease import GpuLease
from synthetics.orchestrator.scheduler import PhaseScheduler
from synthetics.orchestrator.status import StatusStore
from synthetics.viewer.contract import check_run


def _scheduler(rd, cfg):
    return PhaseScheduler(rd, cfg, lease_factory=lambda s, **kw: GpuLease(s, **{**kw, "enabled": False}))


@pytest.mark.parametrize("off", [StageId.SCRIPT_VAL, StageId.VOICE_VAL, StageId.ACOUSTIC])
def test_a_switched_off_stage_is_skipped_and_the_meeting_still_finishes(tmp_path, monkeypatch, off):
    from voice_helpers import fake_refs

    monkeypatch.setenv("NSC_REFS_DIR", str(fake_refs(tmp_path / "refs")))
    monkeypatch.setenv("SOUND_LIBRARY_DIR", str(tmp_path / "no-library"))
    rd, cfg = fake_run(tmp_path / "run", n=2, seed=11, **{"simulation.noise_library": None,
                                                          f"stages.{int(off)}.enabled": False})
    _scheduler(rd, cfg).run(until=StageId.LABELS)

    rp, store = RunPaths(rd), StatusStore(rd)
    for mid in rp.meeting_ids():
        states = {int(k): v.state for k, v in store.get(mid).stages.items()}
        assert states == {int(s): "skipped" if s == off else "pass" for s in StageId}, (mid, states)
        assert store.get(mid).get(off).detail == "disabled in config"
    entries = [ManifestEntry.model_validate(json.loads(x)) for x in rp.manifest.read_text().splitlines()]
    assert [e.meeting_id for e in entries] == rp.meeting_ids()
    for e in entries:
        assert e.stages_enabled == {s.name.lower(): s != off for s in StageId}
        if off == StageId.ACOUSTIC:
            assert e.acoustic == {} and "mixture" not in e.files and "render" not in e.files
            assert not rp.meeting(e.meeting_id).mixture.exists()
        if off == StageId.SCRIPT_VAL:
            assert "decision" not in e.judge
        if off == StageId.VOICE_VAL:
            assert "decision" not in e.voice
    assert all(r.ok for r in check_run(rd)), [r.as_dict() for r in check_run(rd) if not r.ok]

    again = _scheduler(rd, cfg).run(until=StageId.LABELS)  # nothing left to do, and nothing waits on the skip
    assert {o.state for o in again.outcomes} <= {"cached", "skipped"}, again.outcomes
