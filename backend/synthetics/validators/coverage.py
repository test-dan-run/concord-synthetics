"""The M9 coverage list, and a search for a run seed whose meetings meet it.

With only 20 meetings, random sampling can miss rare cases, so the M9 run's seed is chosen to cover: all three
splits, both room classes, at least one 20-speaker test meeting, and at least one meeting each with presenters,
wall seats (overflow and by choice) and remote participants. Stage 1 is sampled in memory, exactly as a run does,
so the chosen seed reproduces the same meetings in `init-run`. The distributions themselves are untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..artifacts.meeting import MeetingSpec
from ..config.schema import RunConfig


def sample_meetings(cfg: RunConfig, n: int, seed: int) -> list[MeetingSpec]:
    """Stage 1 for meetings 1..n of a run with this seed, in memory."""
    from ..nsc.pool import speaker_splits
    from ..stages.runner import meeting_id_for, meeting_seed
    from ..stages.s1_sample import sample_meeting

    sp = speaker_splits(cfg.speaker_pool.path, cfg.speaker_pool.version)
    out = []
    for i in range(1, n + 1):
        mid = meeting_id_for(i)
        s = meeting_seed(seed, mid)
        out.append(sample_meeting(cfg, sp, mid, s, np.random.default_rng(np.random.SeedSequence([seed, s, 1]))))
    return out


def coverage(meetings: Sequence[MeetingSpec], max_speakers: int = 20) -> dict[str, bool]:
    people = [p for m in meetings for p in m.people]
    return {
        "splits train, dev, test": {m.split for m in meetings} >= {"train", "dev", "test"},
        "room classes small, large": {m.room_class for m in meetings} >= {"small", "large"},
        f"a {max_speakers}-speaker test meeting": any(m.split == "test" and m.n_speakers == max_speakers
                                                      for m in meetings),
        "presenters": any(p.position_type == "presenter" for p in people),
        "wall seats, overflow": any(p.position_type == "wall" and p.wall_reason == "overflow" for p in people),
        "wall seats, by choice": any(p.position_type == "wall" and p.wall_reason == "by_choice" for p in people),
        "remote participants": any(p.position_type == "remote" for p in people),
    }


def find_seed(cfg: RunConfig, n: int, start: int = 1, tries: int = 500) -> tuple[int | None, dict[str, bool]]:
    """The first seed from `start` whose n meetings meet the whole list (None after `tries`), and its checks."""
    best: tuple[int | None, dict[str, bool]] = (None, {})
    for seed in range(start, start + tries):
        cov = coverage(sample_meetings(cfg, n, seed))
        if all(cov.values()):
            return seed, cov
        if sum(cov.values()) > sum(best[1].values()):
            best = (None, cov)
    return best
