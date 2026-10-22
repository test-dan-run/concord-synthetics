"""Realised stage 1 rates against the configured targets (M1 exit check).

Each check compares an observed rate or mean with its target. The tolerance is `z` standard errors (binomial
for rates, sample-based for means) plus a small floor, so the check scales with the sample size.
Range checks (seat pitch, mouth heights, RT60, SNR) must hold for every meeting.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from ..acoustics import geometry as geo
from ..artifacts.meeting import MeetingSpec
from ..config.distributions import Categorical, IntUniform, Uniform
from ..config.schema import RunConfig

Z = 4.0
FLOOR = 0.005


@dataclass
class Check:
    name: str
    target: float | str
    observed: float | str
    tolerance: float | None
    n: int
    ok: bool
    note: str = ""


def _rate(name: str, hits: Iterable[bool], p: float, note: str = "") -> Check:
    xs = list(hits)
    n = len(xs)
    if n == 0:
        return Check(name, p, "n/a", None, 0, True, "no samples")
    obs = sum(xs) / n
    tol = Z * math.sqrt(max(p * (1 - p), 1e-12) / n) + FLOOR
    return Check(name, round(p, 4), round(obs, 4), round(tol, 4), n, abs(obs - p) <= tol, note)


def _mean(name: str, values: Sequence[float], mu: float, sd: float, note: str = "") -> Check:
    n = len(values)
    if n == 0:
        return Check(name, mu, "n/a", None, 0, True, "no samples")
    obs = float(np.mean(values))
    tol = Z * sd / math.sqrt(n) + FLOOR * max(abs(mu), 1)
    return Check(name, round(mu, 4), round(obs, 4), round(tol, 4), n, abs(obs - mu) <= tol, note)


def _uniform_sd(u: Uniform | IntUniform) -> float:
    if isinstance(u, IntUniform):
        k = u.hi - u.lo + 1
        return math.sqrt((k * k - 1) / 12)
    return (u.hi - u.lo) / math.sqrt(12)


def _at_least(name: str, hits: Iterable[bool], p: float, note: str = "") -> Check:
    """A rate that the setting `p` only sets a floor for (draws that miss can still land on the outcome)."""
    c = _rate(name, hits, p, note)
    if isinstance(c.observed, float):
        c.target, c.ok = f">= {p}", c.observed >= p - (c.tolerance or 0)
    return c


def _in_range(name: str, values: Sequence[float], lo: float, hi: float, eps: float = 1e-3) -> Check:
    bad = [v for v in values if v < lo - eps or v > hi + eps]
    return Check(name, f"[{lo}, {hi}]", f"[{min(values):.3f}, {max(values):.3f}]" if values else "n/a", None,
                 len(values), not bad, f"{len(bad)} out of range" if bad else "")


def sampler_checks(cfg: RunConfig, meetings: Sequence[MeetingSpec]) -> list[Check]:
    rc, seat, ac = cfg.room_classes, cfg.seating, cfg.acoustics
    checks: list[Check] = []
    by = {c: [m for m in meetings if m.room_class == c] for c in ("small", "large")}

    checks.append(_rate("room class: small share", (m.room_class == "small" for m in meetings),
                        rc.ratio.probs["small"]))
    for split, p in cfg.speaker_pool.split_mix.probs.items():
        checks.append(_rate(f"split: {split} share", (m.split == split for m in meetings), p))

    for c, ms in by.items():
        spk = rc.classes[c].speakers
        if isinstance(spk, IntUniform):
            checks.append(_mean(f"{c}: speakers mean", [m.n_speakers for m in ms], spk.mean(), _uniform_sd(spk)))
            checks.append(_in_range(f"{c}: speakers range", [m.n_speakers for m in ms], spk.lo, spk.hi))
        shapes = rc.classes[c].table_shapes
        for shape, p in shapes.probs.items():
            checks.append(_rate(f"{c}: table {shape} share", (m.table.shape == shape for m in ms), p))

        pres = seat.presenters
        checks.append(_rate(f"{c}: presentation rate", (m.dynamics.presentation is not None for m in ms),
                            pres.presentation[c].p, "slightly lower when the room holds a single attendee"))
        with_pres = [m for m in ms if m.dynamics.presentation is not None]
        two = pres.count[c].probs
        p2 = sum(v for k, v in two.items() if int(k) == 2)
        checks.append(_rate(f"{c}: two presenters | presentation", (m.count("presenter") == 2 for m in with_pres), p2,
                            "capped when too few people are in the room"))
        hyb = ac.hybrid[c]
        checks.append(_rate(f"{c}: hybrid rate", (m.dynamics.hybrid for m in ms), hyb.p))
        rem = ac.remote_count[c]
        hybrid_ms = [m for m in ms if m.dynamics.hybrid]
        if isinstance(rem, IntUniform):
            checks.append(_mean(f"{c}: remote participants | hybrid", [m.count("remote") for m in hybrid_ms],
                                rem.mean(), _uniform_sd(rem)))
        checks.append(_rate(f"{c}: keyboard noise", (any(n.kind == "keyboard" for n in m.acoustics.noise_sources)
                                                    for m in ms), ac.keyboard[c].p))
        checks.append(_mean(f"{c}: RT60 mean", [m.acoustics.rt60 for m in ms], ac.rt60[c].mean(), _uniform_sd(ac.rt60[c])))
        checks.append(_in_range(f"{c}: RT60 range", [m.acoustics.rt60 for m in ms], ac.rt60[c].lo, ac.rt60[c].hi))
        checks.append(_mean(f"{c}: SNR mean", [m.acoustics.snr_db for m in ms], ac.snr_db[c].mean(),
                            _uniform_sd(ac.snr_db[c])))
        checks.append(_in_range(f"{c}: SNR range", [m.acoustics.snr_db for m in ms], ac.snr_db[c].lo, ac.snr_db[c].hi))
        ev = ac.events_per_10min[c]
        checks.append(_mean(f"{c}: events per 10 min (rate)", [m.acoustics.events_per_10min for m in ms], ev.mean(),
                            _uniform_sd(ev)))
        checks.append(_in_range(f"{c}: room length", [m.room.length for m in ms], rc.classes[c].length.lo,
                                rc.classes[c].length.hi))
        checks.append(_in_range(f"{c}: room width", [m.room.width for m in ms], rc.classes[c].width.lo,
                                rc.classes[c].width.hi))
        checks.append(_in_range(f"{c}: ceiling", [m.room.height for m in ms], rc.classes[c].ceiling.lo,
                                rc.classes[c].ceiling.hi))

    large = by["large"]
    no_overflow = [m for m in large if not any(p.wall_reason == "overflow" for p in m.people)]
    checks.append(_rate("large: wall seats by choice | table has room",
                        (m.count("wall") > 0 for m in no_overflow), seat.wall.by_choice.p))
    checks.append(_rate("small: never wall seats", (m.count("wall") == 0 for m in by["small"]), 1.0))
    returns = [ref is not None for m in meetings if m.dynamics.presentation
               for ref in m.dynamics.presentation.returns.values()]
    checks.append(_rate("presenter returns to a seat", returns, seat.presenters.return_to_seat.p))

    walls = [mat == "glass" for m in meetings for mat in m.room.walls.values()]
    if isinstance(ac.wall_material, Categorical):
        checks.append(_rate("wall material: glass share", walls, ac.wall_material.probs.get("glass", 0.0)))
    checks.append(_rate("floor: carpet", (m.room.floor == "carpet" for m in meetings), ac.carpet.p))
    checks.append(_rate("ceiling: acoustic tile", (m.room.ceiling == "acoustic_tile" for m in meetings),
                        ac.acoustic_tile.p))
    dc = ac.device_chain
    checks.append(_rate("device: AGC", (m.device_chain.agc for m in meetings), dc.agc.p))
    checks.append(_rate("device: codec", (m.device_chain.codec_kbps is not None for m in meetings), dc.codec.p))
    checks.append(_rate("device: band-limit", (m.device_chain.band_limit_hz is not None for m in meetings),
                        dc.band_limit.p))
    checks.append(_rate("device: clipping", (m.device_chain.clipping_level is not None for m in meetings),
                        dc.clipping.p))
    checks.append(_rate("device: AEC artifacts | hybrid", (m.device_chain.aec_artifacts for m in meetings
                                                          if m.dynamics.hybrid), dc.aec_artifacts_if_hybrid.p))
    checks.append(_rate("table mic: array", (m.mic.channels > 1 for m in large), ac.mic.centre_array.p))
    checks.append(_rate("large: table mic in front of the chair", (m.mic.type == "chair" for m in large),
                        ac.mic.chair_mic.p))
    chair_mic = [geo.dist2(m.mic.position[:2], m.person(m.mic.near_speaker).mouth[:2]) for m in large
                 if m.mic.type == "chair" and m.mic.near_speaker]
    if chair_mic:
        checks.append(_in_range("chair mic to the chair (horizontal)", chair_mic, ac.mic.chair_mic_distance.lo,
                                ac.mic.chair_mic_distance.hi))
    if seat.chair_at_head.p > 0:
        checks.append(_at_least("large: chair at the head of the table", (_chair_at_head(m) for m in large),
                                seat.chair_at_head.p, "chairs drawn elsewhere can land at the head too"))

    # geometry ranges over every meeting
    pitches = [m.table.seat_pitch for m in meetings]
    checks.append(_in_range("table seat pitch", pitches, rc.seat_pitch.lo, rc.seat_pitch.hi))
    seated_z = [p.mouth[2] for m in meetings for p in m.people if p.position_type in ("table", "wall") and p.mouth]
    checks.append(_in_range("seated mouth height", seated_z, seat.seated_mouth_height.lo, seat.seated_mouth_height.hi))
    pres_z = [p.mouth[2] for m in meetings for p in m.people if p.position_type == "presenter" and p.mouth]
    if pres_z:
        checks.append(_in_range("presenter mouth height", pres_z, seat.presenters.mouth_height.lo,
                                seat.presenters.mouth_height.hi))
    pres_x = [p.presenter.spot[0] for m in meetings for p in m.people if p.presenter]
    if pres_x:
        d = seat.presenters.distance_from_front
        checks.append(_in_range("presenter distance from front wall", pres_x, d.lo, d.hi))
    wall_off = [c.offset for m in meetings for c in m.wall_chairs]
    if wall_off:
        checks.append(_in_range("wall chair offset", wall_off, seat.wall.offset.lo, seat.wall.offset.hi))
    laptop = [geo.dist2(m.mic.position[:2], m.person(m.mic.near_speaker).mouth[:2]) for m in by["small"]
              if m.mic.near_speaker]
    if laptop:
        checks.append(_in_range("laptop mic to its seat (horizontal)", laptop, ac.mic.laptop_distance.lo,
                                ac.mic.laptop_distance.hi))
    clear = [_table_clearance(m) for m in meetings]
    checks.append(_in_range("table clearance to walls", clear, rc.wall_clearance, 1e9))
    if rc.table_at_centre:
        off = [max(abs(m.table.center[0] - m.room.length / 2), abs(m.table.center[1] - m.room.width / 2))
               for m in meetings]
        checks.append(_in_range("table centre off the room centre", off, 0.0, 0.0))
    chair_top = [max(m.dynamics.turn_shares.values()) == m.dynamics.turn_shares[m.dynamics.chair_id]
                 for m in meetings]
    checks.append(_rate("chair takes the top turn share", chair_top, 1.0))
    checks.append(_mean("fit resamples per meeting", [m.fit_resamples for m in meetings],
                        float(np.mean([m.fit_resamples for m in meetings])) if meetings else 0.0, 1.0,
                        "informational"))
    return checks


def _chair_at_head(m: MeetingSpec) -> bool:
    chair = m.person(m.dynamics.chair_id)
    heads = geo.head_slots([s.edge_xy for s in m.table.slots], m.table.center, m.table.shape)
    return chair.seat is not None and chair.seat.index in heads


def _table_clearance(m: MeetingSpec) -> float:
    t, room = m.table, m.room
    cx, cy = t.center
    return min(cx - t.length / 2, room.length - cx - t.length / 2, cy - t.width / 2, room.width - cy - t.width / 2)


def render(checks: Sequence[Check]) -> str:
    w = max(len(c.name) for c in checks)
    lines = [f"{'check'.ljust(w)}  {'target':>14}  {'observed':>18}  {'tol':>7}  {'n':>6}  ok"]
    for c in checks:
        tol = "" if c.tolerance is None else f"{c.tolerance:.4f}"
        lines.append(f"{c.name.ljust(w)}  {str(c.target):>14}  {str(c.observed):>18}  {tol:>7}  {c.n:>6}  "
                     f"{'ok' if c.ok else 'FAIL'}{('  ' + c.note) if c.note else ''}")
    failed = sum(not c.ok for c in checks)
    lines.append(f"\n{len(checks) - failed}/{len(checks)} checks within tolerance")
    return "\n".join(lines)


def as_dicts(checks: Sequence[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
