"""The M9 run report, `run_report.json` in the run directory: stage counts, every meeting that dropped or failed
and why, time per phase (summed over every `run` / `resume` of the run; for reference only), the phase C
figures, and the speaker-disjointness check on the final manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..artifacts.io import atomic_write_text
from ..artifacts.paths import RunPaths
from ..config.schema import StageId
from .status import StatusStore

REPORT = "run_report.json"


def split_overlap(manifest: Path) -> dict[str, list[str]]:
    """NSC speakers that appear in more than one split across the manifest (empty = disjoint)."""
    splits: dict[str, set[str]] = {}
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                for s in e["speakers"]:
                    splits.setdefault(s["nsc_speaker"], set()).add(s["split"])
    return {sp: sorted(v) for sp, v in sorted(splits.items()) if len(v) > 1}


def problems(run_dir: Path) -> list[dict]:
    """Each meeting's first stage that dropped or failed, with its detail."""
    store = StatusStore(run_dir)
    out = []
    for mid in RunPaths(run_dir).meeting_ids():
        st = store.get(mid)
        for sid in StageId:
            s = st.get(sid)
            if s.state in ("dropped", "fail"):
                out.append({"meeting": mid, "stage": int(sid), "state": s.state, "detail": s.detail or ""})
                break
    return out


def write_run_report(run_dir: Path, phase_seconds: dict[str, float] | None = None) -> dict:
    """Rebuild run_report.json, adding `phase_seconds` (this invocation's) to the totals already recorded."""
    from .phase_c import summarise_c
    from .scheduler import summarise

    run_dir = Path(run_dir)
    path = run_dir / REPORT
    old = json.loads(path.read_text()) if path.is_file() else {}
    seconds = dict(old.get("phase_seconds", {}))
    for k, v in (phase_seconds or {}).items():
        seconds[k] = round(seconds.get(k, 0.0) + v, 1)
    rp = RunPaths(run_dir)
    overlap = split_overlap(rp.manifest)
    n_manifest = sum(1 for line in rp.manifest.read_text().splitlines() if line.strip()) if rp.manifest.is_file() else 0
    report = {"run": run_dir.name, "stages": summarise(run_dir, StageId.LABELS), "problems": problems(run_dir),
              "phase_seconds": seconds, "phase_c": summarise_c(run_dir),
              "manifest": {"meetings": n_manifest, "speakers_disjoint_across_splits": not overlap,
                           "speakers_in_several_splits": overlap}}
    atomic_write_text(path, json.dumps(report, indent=2) + "\n")
    return report
