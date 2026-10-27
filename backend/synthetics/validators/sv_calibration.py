"""`synthetics calibrate-sv`: per-model SV thresholds from real NSC same-speaker and different-speaker pairs.

Every clean reference clip (`nsc-part3-refs/clean/`, five per pool speaker) is embedded by both SV models. Same
pairs are all clip pairs within a speaker; different pairs are one clip per speaker against every other speaker
(the "hard" subset shares gender and ethnicity). Per model:

- `sv` (clone vs its reference) = the EER threshold: a clone must score like a real same-speaker pair;
- `sv_pair_max` (two speakers of a meeting) = the same EER threshold: a pair of clone centroids above it looks like
  one person to that model;
- `sv_consistency_min` = the 5th percentile, over speakers, of the mean pairwise cosine of a speaker's own clips.

The two models stay on equal footing: each gets its own thresholds from the same pairs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from ..config.schema import Role
from ..nsc.references import ReferenceIndex
from .voice_checks import consistency

SV_ROLES = (Role.SV_WAVLM, Role.SV_REDIMNET)


def eer(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    """(equal error rate, threshold): accept when score >= threshold."""
    same, diff = np.sort(np.asarray(same, float)), np.sort(np.asarray(diff, float))
    t = np.unique(np.concatenate([same, diff]))
    frr = np.searchsorted(same, t, side="left") / len(same)  # same pairs below t are rejected
    far = 1.0 - np.searchsorted(diff, t, side="left") / len(diff)  # different pairs at or above t are accepted
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(t[i])


def _stats(x: np.ndarray) -> dict:
    return {"n": int(len(x)), "mean": round(float(x.mean()), 4), "p05": round(float(np.quantile(x, 0.05)), 4),
            "p50": round(float(np.median(x)), 4), "p95": round(float(np.quantile(x, 0.95)), 4),
            "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}


def fit_model(emb: dict[str, np.ndarray], refs: ReferenceIndex, speakers: list[str]) -> dict:
    """Thresholds and score statistics for one model. `emb`: clip id -> unit vector."""
    same, cons = [], []
    for s in speakers:
        clips = [c.pair_id for c in refs.clips(s) if c.pair_id in emb]
        if len(clips) >= 2:
            m = np.stack([emb[c] for c in clips])
            same += [float(emb[a] @ emb[b]) for a, b in combinations(clips, 2)]
            cons.append(consistency(m))
    first = {s: refs.clips(s)[0] for s in speakers if refs.clips(s) and refs.clips(s)[0].pair_id in emb}
    diff, hard = [], []
    for a, b in combinations(sorted(first), 2):
        ca, cb = first[a], first[b]
        v = float(emb[ca.pair_id] @ emb[cb.pair_id])
        diff.append(v)
        if (ca.gender, ca.ethnicity) == (cb.gender, cb.ethnicity):
            hard.append(v)
    same_a, diff_a, hard_a, cons_a = map(np.asarray, (same, diff, hard, cons))
    e, t = eer(same_a, diff_a)
    e_hard, t_hard = eer(same_a, hard_a) if len(hard_a) else (None, None)
    return {"eer": round(e, 4), "eer_threshold": round(t, 4),
            "eer_hard": round(e_hard, 4) if e_hard is not None else None,
            "eer_hard_threshold": round(t_hard, 4) if t_hard is not None else None,
            "far_1pct_threshold": round(float(np.quantile(diff_a, 0.99)), 4),
            "consistency_p05": round(float(np.quantile(cons_a, 0.05)), 4),
            "same": _stats(same_a), "diff": _stats(diff_a), "hard_diff": _stats(hard_a) if len(hard_a) else None,
            "speaker_consistency": _stats(cons_a)}


def thresholds(fits: dict[str, dict]) -> dict:
    """The YAML overlay: thresholds.sv, sv_pair_max and sv_consistency_min per model."""
    return {"thresholds": {"sv": {m: f["eer_threshold"] for m, f in fits.items()},
                           "sv_pair_max": {m: f["eer_threshold"] for m, f in fits.items()},
                           "sv_consistency_min": {m: f["consistency_p05"] for m, f in fits.items()}}}


def calibrate(refs: ReferenceIndex, embedders: dict[str, object], *, speakers: list[str] | None = None,
              log: Callable[[str], None] = print) -> dict:
    speakers = speakers or refs.speakers()
    clips = [c for s in speakers for c in refs.clips(s)]
    emb: dict[str, dict[str, np.ndarray]] = {name: {} for name in embedders}
    for i, c in enumerate(clips, 1):
        audio = refs.voice_ref(c.pair_id).audio
        for name, e in embedders.items():
            emb[name][c.pair_id] = e.embed([audio])[0]
        if i % 100 == 0 or i == len(clips):
            log(f"embedded {i}/{len(clips)} clips")
    fits = {name: fit_model(emb[name], refs, speakers) for name in embedders}
    return {"speakers": len(speakers), "clips": len(clips), "refs_index": refs.sha256, "models": fits,
            "overlay": thresholds(fits)}


def run_calibration(*, mode: str = "aicore", out: Path | None = None, lease: bool = True,
                    speakers: list[str] | None = None, log: Callable[[str], None] = print) -> dict:
    from ..backends.registry import Backends
    from ..config.loader import load_run_config
    from ..nsc.settings import nsc_paths
    from ..orchestrator.lease import GpuLease
    from ..orchestrator.run import resolve

    cfg = resolve(load_run_config(overrides={"run_id": "calibrate-sv", "seed": 0, "n_meetings": 0,
                                             "backends.mode": mode}))
    backends = Backends(cfg.backends, SV_ROLES)
    services = sorted({cfg.backends.roles[r] for r in SV_ROLES})
    refs = ReferenceIndex.load(nsc_paths())
    with GpuLease(services, platform_dir=cfg.phases.platform_dir, no_evict=cfg.phases.no_evict,
                  timeout_s=cfg.phases.lease_timeout_s, enabled=lease and mode in ("aicore", "record")):
        res = calibrate(refs, {r.value: backends.get(r) for r in SV_ROLES}, speakers=speakers, log=log)
    res["services"] = {r.value: cfg.backends.roles[r] for r in SV_ROLES}
    if out is not None:
        from ..artifacts.io import atomic_write_text

        head = (f"# Written by `synthetics calibrate-sv` from {res['clips']} clean NSC clips of {res['speakers']} "
                f"speakers ({', '.join(f'{k}: {v}' for k, v in res['services'].items())}).\n"
                f"# EER: {', '.join(f'{m} {f['eer']:.2%}' for m, f in res['models'].items())}. "
                f"Pass it to init-run with --config.\n")
        atomic_write_text(out, head + yaml.safe_dump(res["overlay"], sort_keys=False))
        atomic_write_text(out.with_suffix(".json"), json.dumps(res, indent=2) + "\n")
    return res
