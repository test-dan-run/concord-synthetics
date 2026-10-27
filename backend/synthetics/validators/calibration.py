"""Judge calibration, file-based: no rating UI.

`export_judge_set` writes runs/<run>/judge_sets/<name>/:
    sheet.csv          one row per script; rubric columns left blank for raters (judge scores are NOT shown)
    judge_scores.csv   the judges' scores for the same scripts, kept apart so raters are not anchored
    scripts/<meeting>.txt  the script with its persona cards, readable in any editor
    README.txt         rating instructions

`fit_threshold` reads a filled sheet (several raters = several rows per script, averaged), joins the judge
scores, and fits the pass mark (best balanced accuracy against the raters' pass labels) and the borderline
band (covers most of the judge's disagreements with the raters). It writes a YAML overlay for `--config`.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..artifacts.io import atomic_write_text, read_model
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import Script, ScriptValidation
from ..artifacts.paths import RunPaths
from ..stages import prompts

RUBRIC = ["coherence", "singlish", "realism", "persona_consistency", "overall"]
SHEET_COLUMNS = ["run_id", "meeting_id", "round", "room_class", "n_speakers", "n_turns", "script_file",
                 *RUBRIC, "pass", "rater", "notes"]
HUMAN_PASS_MARK = 3.0  # overall >= this counts as a pass when the rater left `pass` empty

README = """\
Rating instructions (Synthetics judge calibration)

Open each file in scripts/ and fill one row of sheet.csv per script (add a row per extra rater, same
meeting_id, with your name in `rater`). Score 1 (poor) to 5 (excellent):

- coherence: stays on topic; turns respond to what came before; questions get answers.
- singlish: particles, code-switching and local references are natural and fit each speaker.
- realism: sounds like real people talking (fillers, restarts, overlaps, uneven participation), not a script.
  Reward realistic messiness; penalise polish.
- persona_consistency: speakers' turns fit their persona cards (role, seniority, verbosity, code-switching).
- overall: would you accept this as training data for Singapore meeting ASR / diarisation?
- pass: y or n (your accept / reject decision). If empty, overall >= 3 counts as a pass.

Raters should ideally be Singaporean, so Singlish naturalness is judged by people who speak it.
"""


def export_judge_set(run_dir: Path, *, n: int = 60, name: str = "judge_set_1", seed: int = 1) -> Path:
    rp = RunPaths(Path(run_dir))
    candidates = [m for m in rp.meeting_ids() if rp.meeting(m).script.is_file() and rp.meeting(m).meeting.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no scripts under {rp.meetings_dir}")
    rng = np.random.default_rng(seed)
    picked = sorted(candidates if len(candidates) <= n else
                    [candidates[int(i)] for i in rng.choice(len(candidates), size=n, replace=False)])
    out = rp.judge_sets / name
    (out / "scripts").mkdir(parents=True, exist_ok=True)
    rows, judge_rows = [], []
    for mid in picked:
        mp = rp.meeting(mid)
        m = read_model(mp.meeting, MeetingSpec)
        s = read_model(mp.script, Script)
        body = [f"{mid} · {m.dynamics.meeting_type} · {m.dynamics.topic}", "", "Personas:",
                prompts.judge_persona_lines(m), "", "Brief:", s.brief, "", "Transcript:"]
        body += [f"{t.turn_id} {t.speaker_id} [{t.interaction}]: {t.text}" for t in s.turns]
        atomic_write_text(out / "scripts" / f"{mid}.txt", "\n".join(body) + "\n")
        rows.append({"run_id": rp.run_dir.name, "meeting_id": mid, "round": s.generator.round,
                     "room_class": m.room_class, "n_speakers": m.n_speakers, "n_turns": len(s.turns),
                     "script_file": f"scripts/{mid}.txt"})
        jr = {"meeting_id": mid, "round": s.generator.round}
        if mp.script_validation.is_file():
            v = read_model(mp.script_validation, ScriptValidation)
            if v.primary:
                jr |= {"primary_service": v.primary.service, "primary_overall": v.primary.scores.overall}
            if v.tiebreak:
                jr |= {"tiebreak_service": v.tiebreak.service, "tiebreak_overall": v.tiebreak.scores.overall}
            jr["decision"] = v.decision
        judge_rows.append(jr)
    _write_csv(out / "sheet.csv", SHEET_COLUMNS, rows)
    _write_csv(out / "judge_scores.csv", ["meeting_id", "round", "primary_service", "primary_overall",
                                          "tiebreak_service", "tiebreak_overall", "decision"], judge_rows)
    atomic_write_text(out / "README.txt", README)
    return out / "sheet.csv"


def _write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def _human_labels(sheet: Path) -> dict[str, bool]:
    votes: dict[str, list[bool]] = defaultdict(list)
    with open(sheet, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mid = (r.get("meeting_id") or "").strip()
            p = (r.get("pass") or "").strip().lower()
            if p in ("y", "yes", "1", "true", "pass"):
                votes[mid].append(True)
            elif p in ("n", "no", "0", "false", "fail"):
                votes[mid].append(False)
            elif (r.get("overall") or "").strip():
                votes[mid].append(float(r["overall"]) >= HUMAN_PASS_MARK)
    return {m: (sum(v) * 2 >= len(v)) for m, v in votes.items() if v}  # majority; ties count as pass


def fit_threshold(sheet: Path, *, judge_scores: Path | None = None, out: Path | None = None,
                  grid: tuple[float, float, float] = (1.0, 5.0, 0.05), band_coverage: float = 0.8) -> dict:
    sheet = Path(sheet)
    labels = _human_labels(sheet)
    judge_scores = judge_scores or sheet.with_name("judge_scores.csv")
    scores: dict[str, float] = {}
    with open(judge_scores, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r.get("primary_overall") or "").strip():
                scores[r["meeting_id"]] = float(r["primary_overall"])
    common = sorted(set(labels) & set(scores))
    if len(common) < 10:
        raise ValueError(f"only {len(common)} rated scripts with judge scores; need at least 10 (50-100 advised)")
    y = np.array([labels[m] for m in common])
    s = np.array([scores[m] for m in common])
    if y.all() or not y.any():
        raise ValueError("raters passed every script or failed every script; the threshold cannot be fitted")

    lo, hi, step = grid
    results = []
    for t in np.arange(lo, hi + 1e-9, step):
        pred = s >= t
        tpr = (pred & y).sum() / y.sum()
        tnr = (~pred & ~y).sum() / (~y).sum()
        results.append((float(t), float((tpr + tnr) / 2)))
    best_bacc = max(b for _, b in results)
    plateau = [t for t, b in results if b >= best_bacc - 1e-12]
    best_t = float(np.median(plateau))  # the middle of the best range, not its edge
    wrong = np.abs(s[(s >= best_t) != y] - best_t)
    band = float(np.quantile(wrong, band_coverage)) if len(wrong) else 0.25
    band = round(min(1.0, max(0.25, math.ceil(band * 20) / 20)), 2)
    res = {"n_scripts": len(common), "human_pass_rate": round(float(y.mean()), 4), "judge_pass": round(best_t, 2),
           "judge_band": band, "balanced_accuracy": round(best_bacc, 4), "disagreements": int(len(wrong))}
    if out is not None:
        atomic_write_text(Path(out), yaml.safe_dump({"thresholds": {"judge_pass": res["judge_pass"],
                                                                    "judge_band": res["judge_band"]}}, sort_keys=False))
        res["written"] = str(out)
    return res
