"""Score a decision model on labelled requests (`tests/data/suggest_eval.jsonl`).

Each line names the groups a request should change and the option each should get; every other group should stay
unchanged. A group counts as found when the engine applies or offers it, and as right when the applied option, or
the top offered one, is the expected one. Reported: option accuracy on expected groups, and false changes (a group
changed that the request never touched).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .engine import answer_groups
from .questions import load_questions


def load_eval(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(model, rows: list[dict]) -> dict:
    qs = load_questions()
    right, expected, false_changes = Counter(), Counter(), Counter()
    misses = []
    for row in rows:
        s = answer_groups(model, row["request"], qs)
        got = {a.group: a.option for a in s.applied}
        got.update({o.group: o.options[0]["id"] for o in s.offers if o.group not in got})
        got.update({o.group: o.options[0]["id"] for o in s.confirm if o.group not in got})
        for g, want in row["expect"].items():
            expected[g] += 1
            if got.get(g) == want:
                right[g] += 1
            else:
                misses.append({"request": row["request"], "group": g, "want": want, "got": got.get(g)})
        for g in set(got) - set(row["expect"]):
            false_changes[g] += 1
            misses.append({"request": row["request"], "group": g, "want": "unchanged", "got": got[g]})
    n = sum(expected.values())
    return {"requests": len(rows), "expected": n, "accuracy": round(sum(right.values()) / n, 3) if n else None,
            "false_changes": sum(false_changes.values()),
            "per_group": {g: f"{right[g]}/{expected[g]}" for g in sorted(expected)},
            "false_by_group": dict(sorted(false_changes.items())), "misses": misses}
