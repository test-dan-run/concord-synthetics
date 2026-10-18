"""Room-frame geometry shared by stage 1 (placement) and stage 7 (simulation).

x: front wall (W, x = 0) to back wall (E, x = L); y: S wall (y = 0) to N wall (y = W). Angles in degrees,
counter-clockwise from +x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class EdgePoint:
    side: str
    s: float  # arc length along the side
    xy: Vec2
    normal: Vec2  # outward unit normal


def dist2(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def angle_deg(frm: Vec2, to: Vec2) -> float:
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0]))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def offset(p: Vec2, n: Vec2, d: float) -> Vec2:
    return (p[0] + n[0] * d, p[1] + n[1] * d)


# ---------------------------------------------------------------- table edges


def rect_slots(center: Vec2, length: float, width: float, pitch: float) -> list[EdgePoint]:
    """Seats on each straight side at exactly `pitch` spacing, centred on the side (end margins >= pitch / 2)."""
    cx, cy = center
    x0, x1, y0, y1 = cx - length / 2, cx + length / 2, cy - width / 2, cy + width / 2
    sides = [  # name, start, direction, side length, outward normal
        ("N", (x0, y1), (1.0, 0.0), length, (0.0, 1.0)),
        ("E", (x1, y1), (0.0, -1.0), width, (1.0, 0.0)),
        ("S", (x1, y0), (-1.0, 0.0), length, (0.0, -1.0)),
        ("W", (x0, y0), (0.0, 1.0), width, (-1.0, 0.0)),
    ]
    out = []
    for name, start, direction, side_len, normal in sides:
        n = int(math.floor(side_len / pitch + 1e-9))
        if n <= 0:
            continue
        margin = (side_len - (n - 1) * pitch) / 2
        for k in range(n):
            s = margin + k * pitch
            out.append(EdgePoint(name, s, (start[0] + direction[0] * s, start[1] + direction[1] * s), normal))
    return out


def _ellipse_arc(a: float, b: float, samples: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2 * math.pi, samples + 1)
    x, y = a * np.cos(t), b * np.sin(t)
    seg = np.hypot(np.diff(x), np.diff(y))
    return t, np.concatenate([[0.0], np.cumsum(seg)])


def ellipse_perimeter(a: float, b: float) -> float:
    return float(_ellipse_arc(a, b)[1][-1])


def curved_slot_count(perimeter: float, pitch: float, lo: float, hi: float) -> int:
    """Seat count whose even spacing perimeter / n lies in [lo, hi], closest to `pitch`; 0 if none."""
    best, best_err = 0, math.inf
    for n in range(max(1, int(perimeter / hi)), int(perimeter / lo) + 2):
        spacing = perimeter / n
        if lo - 1e-9 <= spacing <= hi + 1e-9 and abs(spacing - pitch) < best_err:
            best, best_err = n, abs(spacing - pitch)
    return best


def ellipse_slots(center: Vec2, length: float, width: float, n: int, phase: float = 0.5) -> list[EdgePoint]:
    """`n` seats evenly spaced by arc length around an ellipse (round tables: length == width)."""
    a, b = length / 2, width / 2
    t, cum = _ellipse_arc(a, b)
    perim = cum[-1]
    out = []
    for k in range(n):
        s = (k + phase) * perim / n
        tk = float(np.interp(s, cum, t))
        x, y = a * math.cos(tk), b * math.sin(tk)
        nx, ny = x / (a * a), y / (b * b)
        norm = math.hypot(nx, ny)
        out.append(EdgePoint("arc", float(s), (center[0] + x, center[1] + y), (nx / norm, ny / norm)))
    return out


def stadium_perimeter(length: float, width: float) -> float:
    return 2 * (length - width) + math.pi * width


def stadium_slots(center: Vec2, length: float, width: float, n: int, phase: float = 0.5) -> list[EdgePoint]:
    """`n` seats evenly spaced by arc length around a stadium (boardroom oval): straight N and S sides of
    length `length - width` joined by semicircular ends of radius `width / 2`."""
    cx, cy = center
    rad, straight = width / 2, length - width
    perim = stadium_perimeter(length, width)
    half_arc = math.pi * rad
    out = []
    for k in range(n):
        s = (k + phase) * perim / n
        if s < straight:  # N side, west to east
            xy, nrm = (cx - straight / 2 + s, cy + rad), (0.0, 1.0)
        elif s < straight + half_arc:  # E end, north to south
            a = math.pi / 2 - (s - straight) / rad
            nrm = (math.cos(a), math.sin(a))
            xy = (cx + straight / 2 + rad * nrm[0], cy + rad * nrm[1])
        elif s < 2 * straight + half_arc:  # S side, east to west
            xy, nrm = (cx + straight / 2 - (s - straight - half_arc), cy - rad), (0.0, -1.0)
        else:  # W end, south to north
            a = -math.pi / 2 - (s - 2 * straight - half_arc) / rad
            nrm = (math.cos(a), math.sin(a))
            xy = (cx - straight / 2 + rad * nrm[0], cy + rad * nrm[1])
        out.append(EdgePoint("arc", float(s), xy, nrm))
    return out


def point_table_distance(p: Vec2, shape: str, center: Vec2, length: float, width: float) -> float:
    """Distance from a point to the table outline (0 inside). Round = circle, oval = stadium."""
    dx, dy = p[0] - center[0], p[1] - center[1]
    if shape == "rect":
        ex, ey = max(abs(dx) - length / 2, 0.0), max(abs(dy) - width / 2, 0.0)
        return math.hypot(ex, ey)
    rad = width / 2
    half = max(length - width, 0.0) / 2
    ex = max(abs(dx) - half, 0.0)
    return max(math.hypot(ex, dy) - rad, 0.0)


def wall_line(wall: str, room_l: float, room_w: float, off: float, margin: float) -> tuple[Vec2, Vec2]:
    """Start and end of a chair row `off` metres from `wall`, leaving `margin` at each corner."""
    if wall == "N":
        return (margin, room_w - off), (room_l - margin, room_w - off)
    if wall == "S":
        return (margin, off), (room_l - margin, off)
    if wall == "E":
        return (room_l - off, margin), (room_l - off, room_w - margin)
    if wall == "W":
        return (off, margin), (off, room_w - margin)
    raise ValueError(wall)


def inside_room(p: Vec2, room_l: float, room_w: float, margin: float = 0.0) -> bool:
    return margin - 1e-9 <= p[0] <= room_l - margin + 1e-9 and margin - 1e-9 <= p[1] <= room_w - margin + 1e-9
