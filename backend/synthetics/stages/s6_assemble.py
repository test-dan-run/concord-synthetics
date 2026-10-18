"""Stage 6: conversation assembly (CPU). Writes timeline.json and tracks/<speaker_id>.wav.

Utterances are placed on a timeline, not concatenated. Each main turn follows the floor with the transition its
script interaction asks for (design doc "Turn-taking"):
- normal / presentation: a log-normal gap, or with `long_pause_p` a long pause
- overlap_start: the turn starts `overlap_lead` before the floor turn ends (at most `overlap_max_frac` of
  either turn)
- interruption: the turn starts inside the floor turn, and the floor speaker is cut off `interruption_cutoff`
  seconds later (`truncated_at`)
Backchannels go inside the main turn they follow in the script. Side conversations start inside their host turn
and run alongside the main conversation. Nobody overlaps their own speech: every segment goes in the first free
slot of its speaker at or after its desired start.

Stage 2 plans its backchannel count from the sampled meeting length. When the synthesised meeting comes out
shorter than planned, backchannels are thinned (listed in `dropped_turns`) so the realised rate stays at the
planned one; a longer meeting keeps them all.

The meeting's overlap target (`turn_taking.overlap_target`) is drawn once. Every other random value is drawn
before the overlap search, which then only scales the overlap leads (and, if that is not enough, lets more normal
transitions overlap) until the realised overlap share meets the target.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from ..artifacts.io import read_model, write_model
from ..artifacts.labels import Segment, Timeline, TimelineStats
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import Script
from ..artifacts.paths import MeetingPaths
from ..artifacts.voice import Synthesis
from ..backends.base import Audio
from ..config.schema import AssemblyConfig, Phase, RunConfig, StageId, TurnTakingConfig
from .base import StageContext, StageResult

FLEX = ("backchannel", "side")  # placed inside other turns; not part of the main sequence


@dataclass(frozen=True)
class Utt:
    turn_id: str
    speaker_id: str
    kind: str
    dur: float  # seconds of audio in the utterance WAV


@dataclass
class Draws:
    """Every random value of one meeting's assembly, drawn before the overlap search."""

    target: float
    gap: list[float]  # per main index
    long: list[float | None]  # long pause instead of the gap
    lead: list[float]
    interrupt_at: list[float]
    cutoff: list[float]
    convert_order: list[int]  # main indices of normal transitions that may be overlapped, in order
    flex_at: dict[int, float]  # script index -> backchannel position or first side-turn position
    side_gap: dict[int, float]
    drop_order: list[int]  # backchannel script indices, in the order they are thinned


@dataclass
class Placed:
    start: float
    audible: float  # end - start
    truncated: bool = False


@dataclass
class Built:
    placed: dict[int, Placed]  # script index -> placement (dropped turns are absent)
    long_pauses: int
    interruptions: int
    scale: float = 1.0
    converted: int = 0
    convertible: int = 0  # normal transitions that could have been overlapped
    stats: dict = field(default_factory=dict)


def main_indices(utts: list[Utt]) -> list[int]:
    return [i for i, u in enumerate(utts) if u.kind not in FLEX]


def draw(utts: list[Utt], room_class: str, tt: TurnTakingConfig, ac: AssemblyConfig, rng: np.random.Generator) -> Draws:
    main = main_indices(utts)
    n = len(main)
    target = tt.overlap_target[room_class].sample(rng)
    gap = [tt.gap.sample(rng) for _ in range(n)]
    long = [tt.long_pause.sample(rng) if rng.random() < tt.long_pause_p else None for _ in range(n)]
    lead = [tt.overlap_lead.sample(rng) for _ in range(n)]
    interrupt_at = [ac.interrupt_at.sample(rng) for _ in range(n)]
    cutoff = [tt.interruption_cutoff.sample(rng) for _ in range(n)]
    normal = [k for k in range(1, n) if utts[main[k]].kind in ("normal", "presentation") and long[k] is None
              and utts[main[k]].speaker_id != utts[main[k - 1]].speaker_id]
    convert_order = [normal[int(j)] for j in rng.permutation(len(normal))]
    flex_at, side_gap = {}, {}
    for i, u in enumerate(utts):
        if u.kind == "backchannel":
            flex_at[i] = ac.backchannel_at.sample(rng)
        elif u.kind == "side":
            flex_at[i] = ac.side_start_at.sample(rng)
            side_gap[i] = ac.side_gap.sample(rng)
    bcs = [i for i, u in enumerate(utts) if u.kind == "backchannel"]
    drop_order = [bcs[int(j)] for j in rng.permutation(len(bcs))]
    return Draws(target, gap, long, lead, interrupt_at, cutoff, convert_order, flex_at, side_gap, drop_order)


class Occupancy:
    """Per-speaker sorted, disjoint intervals; `slot` finds the first free start at or after a desired time."""

    def __init__(self, gap: float):
        self.gap = gap
        self.iv: dict[str, list[tuple[float, float]]] = {}

    def slot(self, spk: str, t: float, dur: float) -> float:
        iv = self.iv.get(spk, [])
        j = max(0, bisect.bisect_left(iv, (t, t)) - 1)
        while j < len(iv):
            s, e = iv[j]
            if t + dur + self.gap <= s:
                break
            if t < e + self.gap:
                t = e + self.gap
            j += 1
        return t

    def add(self, spk: str, start: float, end: float) -> None:
        bisect.insort(self.iv.setdefault(spk, []), (start, end))


def build(utts: list[Utt], d: Draws, tt: TurnTakingConfig, ac: AssemblyConfig, scale: float, n_convert: int,
          skip: frozenset[int] = frozenset()) -> Built:
    main = main_indices(utts)
    pos = {i: k for k, i in enumerate(main)}
    convert = set(d.convert_order[:n_convert])
    occ = Occupancy(ac.min_self_gap_s)
    placed: list[Placed | None] = [None] * len(utts)
    floor_end = ac.lead_in_s
    host: int | None = None  # script index of the latest main turn
    prev_side: Placed | None = None
    long_pauses = interruptions = 0

    def put(i: int, start: float, audible: float, truncated: bool = False) -> Placed:
        p = Placed(start, audible, truncated)
        placed[i] = p
        occ.add(utts[i].speaker_id, start, start + audible)
        return p

    for i, u in enumerate(utts):
        if i in skip:
            continue
        if placed[i] is not None:
            host, prev_side = i, None
            floor_end = max(floor_end, placed[i].start + placed[i].audible)
            continue
        if u.kind in FLEX:
            if host is None:  # nothing to sit inside: treat as a main-floor continuation
                put(i, occ.slot(u.speaker_id, floor_end, u.dur), u.dur)
                continue
            h = placed[host]
            if u.kind == "backchannel":
                want = h.start + d.flex_at[i] * max(0.0, h.audible - u.dur)
            elif prev_side is None:
                want = h.start + d.flex_at[i] * h.audible
            else:
                want = prev_side.start + prev_side.audible + d.side_gap[i]
            p = put(i, occ.slot(u.speaker_id, want, u.dur), u.dur)
            if u.kind == "side":
                prev_side = p
            continue

        k = pos[i]
        if k == 0:
            want = ac.lead_in_s
        else:
            prev_i = main[k - 1]
            prev = placed[prev_i]
            same = utts[prev_i].speaker_id == u.speaker_id
            overlap = not same and (u.kind in ("overlap_start", "interruption") or k in convert)
            if overlap:  # an interruption reaching here could not be realised; it overlaps instead
                lead = min(scale * d.lead[k], tt.overlap_max_frac * prev.audible, tt.overlap_max_frac * u.dur)
                want = floor_end - lead
            elif d.long[k] is not None and not same:
                want = floor_end + d.long[k]
                long_pauses += 1
            else:
                want = floor_end + d.gap[k]
        start = occ.slot(u.speaker_id, want, u.dur)

        # the next main turn interrupts this one: place both now, so the cut is known before backchannels go in
        nxt = main[k + 1] if k + 1 < len(main) else None
        if nxt is not None and utts[nxt].kind == "interruption" and utts[nxt].speaker_id != u.speaker_id:
            b = utts[nxt]
            cutoff = min(d.cutoff[k + 1], u.dur - ac.interrupt_min_before_s - ac.interrupt_min_tail_s)
            if cutoff >= min(tt.interruption_cutoff.lo, 0.3):
                lo = start + ac.interrupt_min_before_s
                hi = start + u.dur - ac.interrupt_min_tail_s - cutoff
                ti = occ.slot(b.speaker_id, min(max(start + d.interrupt_at[k + 1] * u.dur, lo), hi), b.dur)
                if ti <= hi:
                    put(i, start, ti + cutoff - start, truncated=True)
                    put(nxt, ti, b.dur)
                    interruptions += 1
                    host, prev_side = i, None
                    floor_end = max(floor_end, start + placed[i].audible)
                    continue
        put(i, start, u.dur)
        host, prev_side = i, None
        floor_end = max(floor_end, start + u.dur)
    return Built({i: p for i, p in enumerate(placed) if p is not None}, long_pauses, interruptions)


def quantise(utts: list[Utt], b: Built, sr: int) -> list[Segment]:
    """Segments on the sample grid: start and audible length in whole samples."""
    segs = []
    for i, p in sorted(b.placed.items()):
        u = utts[i]
        s0 = int(round(p.start * sr))
        n_full = int(round(u.dur * sr))
        n = min(n_full, int(round(p.audible * sr))) if p.truncated else n_full
        segs.append(Segment(turn_id=u.turn_id, speaker_id=u.speaker_id, kind=u.kind, start=s0 / sr, end=(s0 + n) / sr,
                            truncated_at=(n / sr) if p.truncated else None))
    return segs


def coverage(segs: list[Segment]) -> tuple[float, float]:
    """(voiced time, time with two or more speakers active)."""
    edges = sorted([(s.start, 1) for s in segs] + [(s.end, -1) for s in segs])
    voiced = overlap = 0.0
    active, t_prev = 0, 0.0
    for t, delta in edges:
        if active >= 1:
            voiced += t - t_prev
        if active >= 2:
            overlap += t - t_prev
        active += delta
        t_prev = t
    return voiced, overlap


def stats(utts: list[Utt], segs: list[Segment], b: Built, d: Draws, ac: AssemblyConfig) -> TimelineStats:
    voiced, overlap = coverage(segs)
    duration = max(s.end for s in segs) + ac.tail_s
    main = main_indices(utts)
    changes = sum(utts[main[k]].speaker_id != utts[main[k - 1]].speaker_id for k in range(1, len(main)))
    n_bc = sum(s.kind == "backchannel" for s in segs)
    return TimelineStats(duration=round(duration, 4), speech_time=round(sum(s.end - s.start for s in segs), 4),
                         overlap_share=round(overlap / voiced, 4) if voiced else 0.0,
                         backchannels_per_min=round(n_bc / (duration / 60), 4),
                         interruption_share=round(b.interruptions / changes, 4) if changes else 0.0,
                         long_pauses=b.long_pauses, voiced_time=round(voiced, 4), overlap_time=round(overlap, 4),
                         overlap_target=round(d.target, 4), speaker_changes=changes, interruptions=b.interruptions,
                         backchannels=n_bc)


def control(utts: list[Utt], d: Draws, tt: TurnTakingConfig, ac: AssemblyConfig, sr: int,
            skip: frozenset[int]) -> tuple[Built, list[Segment]]:
    """The overlap search: lead scale first (up to `max_lead_scale`), then extra overlapped transitions. Returns
    the build whose overlap share is closest to the target."""
    tried: list[tuple[float, Built, list[Segment]]] = []

    def at(scale: float, n_convert: int) -> float:
        b = build(utts, d, tt, ac, scale, n_convert, skip)
        segs = quantise(utts, b, sr)
        voiced, overlap = coverage(segs)
        b.scale, b.converted = scale, n_convert
        tried.append((overlap / voiced if voiced else 0.0, b, segs))
        return tried[-1][0]

    def bisect_scale(lo: float, hi: float, n_convert: int, iters: int = 10) -> None:
        for _ in range(iters):
            mid = (lo + hi) / 2
            if at(mid, n_convert) < d.target:
                lo = mid
            else:
                hi = mid

    share = at(1.0, 0)
    if not ac.overlap_control:
        pass
    elif share > d.target:
        bisect_scale(0.0, 1.0, 0)
    elif at(ac.max_lead_scale, 0) >= d.target:
        bisect_scale(1.0, ac.max_lead_scale, 0)
    elif ac.convert_transitions and d.convert_order:
        lo, hi = 0, len(d.convert_order)
        if at(ac.max_lead_scale, hi) >= d.target:
            while hi - lo > 1:  # fewest extra overlaps that reach the target at full scale
                mid = (lo + hi) // 2
                if at(ac.max_lead_scale, mid) >= d.target:
                    hi = mid
                else:
                    lo = mid
            bisect_scale(0.0, ac.max_lead_scale, hi)
    _, best, segs = min(tried, key=lambda t: abs(t[0] - d.target))
    return best, segs


def assemble(utts: list[Utt], meeting_id: str, room_class: str, sr: int, tt: TurnTakingConfig, ac: AssemblyConfig,
             rng: np.random.Generator, planned_min: float | None = None) -> tuple[Timeline, Built]:
    """`planned_min`: the meeting length stage 2 planned its backchannels for (None: never thin)."""
    d = draw(utts, room_class, tt, ac, rng)
    skip: frozenset[int] = frozenset()
    best, segs = control(utts, d, tt, ac, sr, skip)
    n_bc = len(d.drop_order)
    if ac.thin_backchannels and planned_min and n_bc:
        rate = n_bc / planned_min
        for _ in range(3):  # thinning shortens nothing much, but overlap control does: settle both
            keep = min(n_bc, int(round(rate * (max(s.end for s in segs) + ac.tail_s) / 60)))
            new = frozenset(d.drop_order[:n_bc - keep])
            if new == skip:
                break
            skip = new
            best, segs = control(utts, d, tt, ac, sr, skip)
    best.convertible = len(d.convert_order)
    best.stats = stats(utts, segs, best, d, ac).model_dump()
    order = sorted(range(len(segs)), key=lambda j: (segs[j].start, segs[j].turn_id))
    tl = Timeline(meeting_id=meeting_id, sample_rate=sr, segments=[segs[j] for j in order],
                  stats=TimelineStats(**best.stats), dropped_turns=[utts[i].turn_id for i in sorted(skip)])
    return tl, best


def write_tracks(paths: MeetingPaths, tl: Timeline, speakers: list[str], fade_s: float) -> None:
    """One dry, time-aligned track per speaker, built and written one at a time (a 45 min track is ~170 MB)."""
    sr = tl.sample_rate
    n_total = int(round(tl.stats.duration * sr))
    fade = int(round(fade_s * sr))
    by_spk: dict[str, list[Segment]] = {s: [] for s in speakers}
    for seg in tl.segments:
        by_spk.setdefault(seg.speaker_id, []).append(seg)
    for spk, segs in by_spk.items():
        buf = np.zeros(n_total, np.float32)
        for seg in segs:
            x = Audio.load(paths.utterance(seg.turn_id))
            s0 = int(round(seg.start * sr))
            n = int(round((seg.end - seg.start) * sr))
            chunk = x.samples[:n].copy()
            if seg.truncated_at is not None and fade:
                k = min(fade, len(chunk))
                chunk[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
            buf[s0:s0 + len(chunk)] += chunk
        Audio(buf, sr).save(paths.track(spk))
        del buf


def load_utts(paths: MeetingPaths, script: Script, syn: Synthesis, sr: int) -> list[Utt]:
    if syn.sample_rate != sr:
        raise ValueError(f"synthesis.json is at {syn.sample_rate} Hz, the run is at {sr} Hz")
    by_turn = {u.turn_id: u for u in syn.utterances}
    missing = [t.turn_id for t in script.turns if t.turn_id not in by_turn]
    if missing:
        raise ValueError(f"no utterance for {len(missing)} script turns ({', '.join(missing[:5])})")
    utts = []
    for t in script.turns:
        info = sf.info(str(paths.utterance(t.turn_id)))
        if info.samplerate != sr:
            raise ValueError(f"{t.turn_id}.wav is at {info.samplerate} Hz, the run is at {sr} Hz")
        utts.append(Utt(t.turn_id, t.speaker_id, t.interaction, info.frames / sr))
    return utts


# ---------------------------------------------------------------- the stage


class Stage6Config(BaseModel):
    turn_taking: TurnTakingConfig
    assembly: AssemblyConfig
    sample_rate: int


class AssembleStage:
    id = StageId.ASSEMBLE
    name = "Assembly"
    phase = Phase.CPU
    roles = ()
    reads = ("meeting", "script", "synthesis", "utterances_dir")
    writes = ("timeline", "tracks_dir")
    version = "1"

    def config_slice(self, cfg: RunConfig) -> BaseModel:
        return Stage6Config(turn_taking=cfg.turn_taking, assembly=cfg.assembly, sample_rate=cfg.sample_rate)

    def run(self, ctx: StageContext) -> StageResult:
        from .runner import meeting_seed

        cfg = ctx.cfg
        m = read_model(ctx.paths.meeting, MeetingSpec)
        script = read_model(ctx.paths.script, Script)
        syn = read_model(ctx.paths.synthesis, Synthesis)
        utts = load_utts(ctx.paths, script, syn, cfg.sample_rate)
        # a function of (run seed, meeting seed) only, so a re-run after an upstream change stays reproducible
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, meeting_seed(cfg.seed, m.meeting_id), int(self.id)]))
        tl, b = assemble(utts, m.meeting_id, m.room_class, cfg.sample_rate, cfg.turn_taking, cfg.assembly, rng,
                         planned_min=m.dynamics.duration_min)
        write_tracks(ctx.paths, tl, [p.speaker_id for p in m.people], cfg.assembly.fade_s)
        write_model(ctx.paths.timeline, tl)
        s = tl.stats
        return StageResult(state="pass",
                           detail=(f"{s.duration / 60:.1f} min · overlap {s.overlap_share:.0%} "
                                   f"(target {s.overlap_target:.0%}"
                                   + (", out of reach" if abs(s.overlap_share - (s.overlap_target or 0)) > 0.005 else "")
                                   + f") · {s.interruptions} interruptions"
                                   + (f" · {len(tl.dropped_turns)} backchannels thinned" if tl.dropped_turns else "")),
                           metrics={"duration_s": s.duration, "overlap_share": s.overlap_share,
                                    "overlap_target": s.overlap_target or 0.0, "interruption_share": s.interruption_share,
                                    "backchannels_per_min": s.backchannels_per_min, "long_pauses": s.long_pauses,
                                    "lead_scale": round(b.scale, 3), "converted_transitions": b.converted,
                                    "dropped_turns": len(tl.dropped_turns)})
