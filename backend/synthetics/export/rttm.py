"""RTTM speaker turns from the timeline: one SPEAKER line per segment, in start order.

    SPEAKER <meeting_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

Times have millisecond precision. Overlapping segments of different speakers stay as they are; one speaker's
segments never overlap (stage 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..artifacts.io import atomic_write_text
from ..artifacts.labels import Timeline


@dataclass(frozen=True)
class RttmTurn:
    file_id: str
    start: float
    duration: float
    speaker: str


def rttm_text(tl: Timeline) -> str:
    lines = [f"SPEAKER {tl.meeting_id} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {s.speaker_id} <NA> <NA>"
             for s in sorted(tl.segments, key=lambda s: (s.start, s.speaker_id))]
    return "\n".join(lines) + ("\n" if lines else "")


def write_rttm(path: Path, tl: Timeline) -> None:
    atomic_write_text(path, rttm_text(tl))


def read_rttm(path: Path) -> list[RttmTurn]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            out.append(RttmTurn(f[1], float(f[3]), float(f[4]), f[7]))
    return out
