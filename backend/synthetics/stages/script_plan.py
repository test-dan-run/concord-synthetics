"""Stage 2 turn plan: who speaks when, and how, before any text is written.

The plan hits the turn-share targets from meeting.json by construction (largest-remainder counts, shuffled
with no speaker following themselves), marks interaction types from the turn-taking config, places the
presentation segment (monologue, then Q&A from the floor), backchannels and side conversations. The LLM
then writes the words for each planned turn. Stage 6 realises the timing of each interaction type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..artifacts.meeting import MeetingSpec
from ..config.schema import TurnTakingConfig

QA_INTERRUPTION_P = 0.25  # questions from the floor that cut into the presenter's answer


@dataclass(frozen=True)
class PlannedTurn:
    speaker_id: str
    interaction: str  # normal | interruption | backchannel | overlap_start | side | presentation


@dataclass
class TurnPlan:
    turns: list[PlannedTurn]
    interruption_rate: float
    backchannels_per_min: float
    targets: dict[str, int]  # main turns per speaker (excludes backchannels and side turns)

    def main_turns(self) -> list[PlannedTurn]:
        return [t for t in self.turns if t.interaction not in ("backchannel", "side")]


def share_counts(shares: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder apportionment of `n` turns, at least one turn per speaker."""
    ids = sorted(shares)
    n = max(n, len(ids))
    raw = {s: shares[s] * (n - len(ids)) for s in ids}
    counts = {s: 1 + int(math.floor(raw[s])) for s in ids}
    left = n - sum(counts.values())
    for s in sorted(ids, key=lambda s: (-(raw[s] - math.floor(raw[s])), s))[:left]:
        counts[s] += 1
    return counts


def order_no_repeats(counts: dict[str, int], rng: np.random.Generator, prev: str | None = None) -> list[str]:
    """A random order of the multiset `counts` with no speaker twice in a row (when possible). At each step a
    speaker is drawn in proportion to remaining count; the largest pile is forced when it would otherwise be
    impossible to finish without a repeat."""
    remaining = dict(counts)
    out: list[str] = []
    total = sum(remaining.values())
    while total:
        cands = [s for s, c in remaining.items() if c > 0 and s != prev]
        if not cands:
            cands = [s for s, c in remaining.items() if c > 0]
        biggest = max(cands, key=lambda s: (remaining[s], s))
        if remaining[biggest] * 2 >= total + 1:  # it must take every other slot from here on
            pick = biggest
        else:
            w = np.array([remaining[s] for s in cands], float)
            pick = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        out.append(pick)
        remaining[pick] -= 1
        total -= 1
        prev = pick
    return out


def build_plan(m: MeetingSpec, tt: TurnTakingConfig, rng: np.random.Generator) -> TurnPlan:
    n = m.dynamics.n_turns_target
    shares = m.dynamics.turn_shares
    targets = share_counts(shares, n)
    seg = m.dynamics.presentation
    speakers = sorted(shares)

    # 1. presentation segment: monologue blocks by the presenters, then Q&A with the floor. Presenters' turns come
    #    out of their own budget, floor questions out of the askers' budgets, so the totals still match the targets.
    pres_turns: list[PlannedTurn] = []
    if seg is not None:
        length = seg.end_turn - seg.start_turn
        presenters = seg.presenters
        budget = sum(targets[p] for p in presenters)
        mono = min(budget, max(2, min(int(round(length * 0.6)), int(0.7 * budget))))
        # each presenter's monologue block is proportional to their own budget
        left = dict(targets)
        mono_turns: list[PlannedTurn] = []
        for k, p in enumerate(presenters):
            share = mono - len(mono_turns) if k == len(presenters) - 1 else int(round(mono * targets[p] / budget))
            share = min(share, left[p])
            mono_turns += [PlannedTurn(p, "presentation")] * share
            left[p] -= share
        floor = [s for s in speakers if s not in presenters]
        qa: list[PlannedTurn] = []
        for i in range(max(0, length - len(mono_turns))):
            if i % 2 == 0:  # a question from the floor, from whoever still has turns to spend
                cands = [s for s in floor if left[s] > 1] or [s for s in floor if left[s] > 0]
                if not cands:
                    break
                w = np.array([left[s] for s in cands], float)
                q = cands[int(rng.choice(len(cands), p=w / w.sum()))]
                qa.append(PlannedTurn(q, "interruption" if rng.random() < QA_INTERRUPTION_P else "normal"))
                left[q] -= 1
            else:  # a presenter answers, if any of them has budget left
                ans = [p for p in presenters if left[p] > 0]
                if ans:
                    a = ans[i // 2 % len(ans)]
                    qa.append(PlannedTurn(a, "normal"))
                    left[a] -= 1
        pres_turns = mono_turns + qa
        for t in pres_turns:
            targets[t.speaker_id] -= 1
        assert min(targets.values()) >= 0, targets

    # 2. the rest of the meeting, around the segment, with exact remaining counts; interruptions and
    #    overlap starts placed as exact counts too, so the realised rates match the drawn ones
    start = seg.start_turn if seg is not None else n
    free_order = order_no_repeats(targets, rng)
    interruption_rate = tt.interruption_rate.sample(rng)
    n_free = len(free_order)
    kinds = ["normal"] * n_free
    slots = [int(i) for i in rng.permutation(np.arange(1, n_free))] if n_free > 1 else []
    n_int = int(round(interruption_rate * (n_free - 1)))
    n_ovl = int(round(tt.overlap_share * (n_free - 1 - n_int)))
    for i in slots[:n_int]:
        kinds[i] = "interruption"
    for i in slots[n_int:n_int + n_ovl]:
        kinds[i] = "overlap_start"
    free_turns = [PlannedTurn(s, k) for s, k in zip(free_order, kinds, strict=True)]
    main: list[PlannedTurn] = free_turns[:start] + pres_turns + free_turns[start:]
    if main:
        main[0] = PlannedTurn(main[0].speaker_id, "normal")  # the first turn cannot overlap anything
    # a speaker following themselves (at the segment seams) continues normally
    for i in range(1, len(main)):
        if main[i].speaker_id == main[i - 1].speaker_id and main[i].interaction in ("interruption", "overlap_start"):
            main[i] = PlannedTurn(main[i].speaker_id, "normal")

    # 3. backchannels inside other speakers' turns, from listeners weighted toward the more active ones
    duration = m.dynamics.duration_min
    bc_rate = tt.backchannels_per_min.sample(rng)
    n_bc = int(round(bc_rate * duration))
    slots = sorted(int(i) for i in rng.choice(len(main), size=min(n_bc, len(main)), replace=False)) if main else []
    bc_after: dict[int, str] = {}
    for i in slots:
        host = main[i].speaker_id
        nxt = main[i + 1].speaker_id if i + 1 < len(main) else None
        others = [s for s in speakers if s not in (host, nxt)] or [s for s in speakers if s != host]
        w = np.sqrt(np.array([shares[s] for s in others], float))
        bc_after[i] = others[int(rng.choice(len(others), p=w / w.sum()))]

    # 4. side conversations between two in-room, non-presenting people, outside the presentation
    rate = tt.side_conversations_per_10min[m.room_class].sample(rng)
    n_side = int(rng.poisson(rate * duration / 10))
    in_room = [p.speaker_id for p in m.people if p.position_type in ("table", "wall") and not p.is_chair]
    side_after: dict[int, list[str]] = {}
    seg_range = range(start, start + len(pres_turns) + 1)
    for _ in range(n_side if len(in_room) >= 2 else 0):
        i = int(rng.integers(len(main)))
        if i in seg_range or i in side_after:
            continue
        a, b = (in_room[int(j)] for j in rng.choice(len(in_room), size=2, replace=False))
        k = tt.side_conversation_turns.sample(rng)
        side_after[i] = [a if j % 2 == 0 else b for j in range(k)]

    turns: list[PlannedTurn] = []
    for i, t in enumerate(main):
        turns.append(t)
        if i in bc_after:
            turns.append(PlannedTurn(bc_after[i], "backchannel"))
        for s in side_after.get(i, []):
            turns.append(PlannedTurn(s, "side"))
    final_targets = {s: sum(1 for t in main if t.speaker_id == s) for s in speakers}
    return TurnPlan(turns, interruption_rate, bc_rate, final_targets)
