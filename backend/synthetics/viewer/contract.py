"""The viewer's data contract (`UI_DESIGN_DOCUMENT.md`, "Data contract") as checks over a run directory (M10).

Each `Row` is one line of the UI doc's table: an artifact and the fields the viewer reads from it. A meeting's file
is only required once the stage that writes it has passed, so a meeting that failed stage 7 is still checked up to
its timeline, and only meetings whose stage 8 passed must be in the manifest. `synthetics viewer-contract` prints
the rows; `tests/test_viewer_contract.py` runs them on a fake run.

Not in the contract: speaker facing per turn. v1 of the viewer shows the one sampled direction per seat
(`people[].facing_deg`), which is checked here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..artifacts.io import read_model
from ..artifacts.labels import AcousticRender, ManifestEntry, Timeline, Transcript
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import MeetingStatus, Script, ScriptValidation, SpeakerSplits
from ..artifacts.paths import MeetingPaths, RunPaths
from ..artifacts.voice import Synthesis, VoiceValidation
from ..config.loader import read_run_config
from ..config.schema import RunConfig, StageId

POSITION_WORDS = {"table": "table seat", "wall": "wall seat", "presenter": "presenter", "remote": "remote"}
TABLE_SHAPES = ("rect", "round", "oval")


@dataclass
class Row:
    artifact: str
    fields: str
    checked: set[str] = field(default_factory=set)  # meetings (or "run") the row was checked on
    problems: list[str] = field(default_factory=list)  # "<where>: <what>"

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {"artifact": self.artifact, "fields": self.fields, "checked": len(self.checked), "ok": self.ok,
                "problems": self.problems}


class _Rows:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], Row] = {}

    def row(self, artifact: str, fields: str) -> Row:
        return self.rows.setdefault((artifact, fields), Row(artifact, fields))

    def expect(self, artifact: str, fields: str, where: str, cond: bool, what: str) -> bool:
        r = self.row(artifact, fields)
        r.checked.add(where)
        if not cond:
            r.problems.append(f"{where}: {what}")
        return cond

    def load[M: BaseModel](self, artifact: str, where: str, path: Path, cls: type[M]) -> M | None:
        """Read and validate a file; a missing or invalid one is a problem on the artifact's parse row."""
        try:
            model = read_model(path, cls)
        except (OSError, ValidationError, ValueError) as e:
            self.expect(artifact, "parses", where, False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
            return None
        self.expect(artifact, "parses", where, True, "")
        return model


def _num(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _range(d: object) -> bool:
    return _num(getattr(d, "lo", None)) and _num(getattr(d, "hi", None))


def _prob(d: object) -> bool:
    p = getattr(d, "p", None)
    return _num(p) and 0.0 <= p <= 1.0


def check_config(rows: _Rows, cfg: RunConfig) -> None:
    a, w = "config.yaml", "run"
    weights = getattr(cfg.room_classes.ratio, "weights", {})
    rows.expect(a, "room-class ratio", w, set(map(str, weights)) == {"small", "large"}, f"weights {weights}")
    for cls in ("small", "large"):
        rows.expect(a, "per-class ranges: speakers, RT60, SNR", w,
                    _range(getattr(cfg.room_classes.classes, cls).speakers)
                    and _range(getattr(cfg.acoustics.rt60, cls)) and _range(getattr(cfg.acoustics.snr_db, cls)),
                    f"{cls}: a range without lo / hi")
        rows.expect(a, "hybrid and presentation probabilities", w,
                    _prob(getattr(cfg.acoustics.hybrid, cls))
                    and _prob(getattr(cfg.seating.presenters.presentation, cls)), f"{cls}: not a Bernoulli")
        rows.expect(a, "turn-taking targets", w, _range(getattr(cfg.turn_taking.overlap_target, cls)),
                    f"{cls}: overlap_target without lo / hi")
    rows.expect(a, "wall-seating probability", w, _prob(cfg.seating.wall.by_choice), "seating.wall.by_choice")
    rows.expect(a, "turn-taking targets", w,
                _range(cfg.turn_taking.backchannels_per_min) and _range(cfg.turn_taking.interruption_rate),
                "backchannels_per_min or interruption_rate without lo / hi")


def check_meeting_spec(rows: _Rows, mid: str, m: MeetingSpec) -> None:
    a = "meeting.json"
    ids = [p.speaker_id for p in m.people]
    rows.expect(a, "room: dimensions, front wall, surfaces", mid,
                set(m.room.walls) == {"N", "E", "S", "W"} and m.room.front_wall in m.room.walls
                and min(m.room.length, m.room.width, m.room.height) > 0, "walls or front wall")
    presenters = [p.speaker_id for p in m.people if p.position_type == "presenter"]
    rows.expect(a, "screen on the front wall (meetings with presenters)", mid,
                not presenters or (m.room.screen is not None and m.room.screen.wall == m.room.front_wall),
                "presenters but no screen on the front wall")
    occupied = {s.occupied_by for s in m.table.slots if s.occupied_by}
    rows.expect(a, "table: shape, size, seats", mid,
                m.table.shape in TABLE_SHAPES and len(m.table.slots) == m.table.capacity
                and occupied == {p.speaker_id for p in m.people if p.position_type == "table"},
                "slots do not match the table-seated people")
    for p in m.people:
        where = f"{mid}/{p.speaker_id}"
        ok = {"table": p.seat is not None and p.seat.kind == "table" and p.mouth is not None,
              "wall": (p.seat is not None and p.seat.kind == "wall" and p.mouth is not None
                       and p.wall_reason in ("overflow", "by_choice")),
              "presenter": p.presenter is not None and p.mouth is not None,
              "remote": p.mouth is None and m.speakerphone is not None
              and p.speaker_id in m.speakerphone.remote_speakers}[p.position_type]
        rows.expect(a, f"{p.position_type} positions", where, ok, f"{p.position_type} position is incomplete")
        if p.position_type != "remote":
            rows.expect(a, "facing per seat (v1: one direction per seat)", where, p.facing_deg is not None,
                        "no facing_deg")
    seg = m.dynamics.presentation
    if presenters or seg is not None:
        rows.expect(a, "presentation segment and return seats", mid,
                    seg is not None and sorted(seg.presenters) == sorted(presenters)
                    and seg.start_turn < seg.end_turn and set(seg.returns) == set(presenters),
                    "presentation segment does not match the presenters")
    rows.expect(a, "mic, speakerphone, noise, device chain", mid,
                m.mic.channels >= 1 and (m.mic.channels == 1 or len(m.mic.array_geometry) == m.mic.channels)
                and (m.speakerphone is not None) == m.dynamics.hybrid,
                "mic channels or speakerphone do not match")
    rows.expect(a, "persona and turn share per person", mid,
                set(m.dynamics.turn_shares) == set(ids) and abs(sum(m.dynamics.turn_shares.values()) - 1) < 0.01,
                "turn_shares do not cover the people or do not sum to 1")
    rows.expect(a, "fit resample count", mid, m.fit_resamples >= 0, "negative fit_resamples")


def check_status(rows: _Rows, mid: str, st: MeetingStatus) -> None:
    a = "status.json"
    for n, s in st.stages.items():
        if s.state in ("pass", "fail"):
            rows.expect(a, "per-stage state, timings, attempts, detail", f"{mid}/{n}",
                        bool(s.started_at and s.ended_at and s.attempts >= 1 and s.detail),
                        f"stage {n} {s.state} without times, attempts or detail")
    if st.get(StageId.SAMPLE).state == "pass":
        rows.expect(a, "resample and retry counts", mid,
                    "fit_resamples" in st.get(StageId.SAMPLE).metrics
                    and (st.get(StageId.SCRIPT_VAL).state != "pass" or "round" in st.get(StageId.SCRIPT_VAL).metrics),
                    "stage 1 fit_resamples or stage 3 round missing from metrics")
    if st.get(StageId.LABELS).state == "pass":
        rows.expect(a, "all eight stages recorded", mid, all(int(s) in st.stages for s in StageId),
                    f"stages {sorted(st.stages)}")


def check_meeting(rows: _Rows, paths: MeetingPaths, cfg: RunConfig, splits: SpeakerSplits | None,
                  entry: ManifestEntry | None) -> None:
    mid = paths.meeting_id
    st = rows.load("status.json", mid, paths.status, MeetingStatus) if paths.status.is_file() else None
    if st is None:
        return  # not started
    check_status(rows, mid, st)

    def passed(stage: StageId) -> bool:
        return st.get(stage).state == "pass"

    if not passed(StageId.SAMPLE):
        return
    m = rows.load("meeting.json", mid, paths.meeting, MeetingSpec)
    if m is None:
        return
    check_meeting_spec(rows, mid, m)
    ids = [p.speaker_id for p in m.people]

    script = None
    if passed(StageId.SCRIPT):
        a = "prompts/<speaker_id>.txt"
        for p in m.people:
            where, f = f"{mid}/{p.speaker_id}", paths.prompt(p.speaker_id)
            first = f.read_text(encoding="utf-8").splitlines()[0] if f.is_file() and f.stat().st_size else ""
            rows.expect(a, "rendered persona prompt per speaker, naming the position", where,
                        first.startswith(f"# Persona {p.speaker_id} ") and POSITION_WORDS[p.position_type] in first,
                        f"first line {first[:60]!r}" if first else "missing or empty")
        script = rows.load("script.json", mid, paths.script, Script)
        if script is not None:
            rows.expect("script.json", "turns: speaker, interaction, text, text_norm", mid,
                        bool(script.turns) and {t.speaker_id for t in script.turns} <= set(ids)
                        and len({t.turn_id for t in script.turns}) == len(script.turns),
                        "unknown speakers or repeated turn ids")
    if passed(StageId.SCRIPT_VAL):
        sv = rows.load("script_validation.json", mid, paths.script_validation, ScriptValidation)
        if sv is not None:
            pc = sv.primary.scores.persona_consistency if sv.primary else {}
            rows.expect("script_validation.json", "judge persona-consistency score per speaker", mid,
                        set(ids) <= set(pc), f"missing {sorted(set(ids) - set(pc))}")

    voices: dict[str, str] = {p.speaker_id: p.nsc_speaker.speaker for p in m.people}
    if passed(StageId.VOICE):
        syn = rows.load("synthesis.json", mid, paths.synthesis, Synthesis)
        if syn is not None:
            rows.expect("synthesis.json", "voice per speaker (after swaps)", mid, set(ids) <= set(syn.voices),
                        f"missing {sorted(set(ids) - set(syn.voices))}")
            voices.update({k: v.nsc_speaker for k, v in syn.voices.items()})
    if splits is not None:
        bad = [f"{s}={v}" for s, v in voices.items()
               if v not in splits.speakers or splits.speakers[v].split != m.split]
        rows.expect("speaker_splits.json", "NSC reference speaker per persona, in the meeting's split", mid, not bad,
                    ", ".join(bad))
    if passed(StageId.VOICE_VAL):
        vv = rows.load("voice_validation.json", mid, paths.voice_validation, VoiceValidation)
        if vv is not None:
            roles = set(cfg.thresholds.sv)
            missing = [u.turn_id for u in vv.utterances if "sv" not in u.skipped and not roles <= set(u.sv)]
            rows.expect("voice_validation.json", "both speaker-verification scores per utterance", mid,
                        bool(vv.utterances) and not missing,
                        "no utterances" if not vv.utterances else f"turns without {sorted(roles)}: {missing[:5]}")
            rows.expect("voice_validation.json", "ASR-limited flag per utterance", mid,
                        {u.speaker_id for u in vv.utterances} <= set(ids), "unknown speakers")
    if passed(StageId.ASSEMBLE):
        tl = rows.load("timeline.json", mid, paths.timeline, Timeline)
        if tl is not None and script is not None:
            turns = {t.turn_id for t in script.turns}
            heard = [s.turn_id for s in tl.segments]
            starts = [s.start for s in tl.segments]
            rows.expect("timeline.json", "turn start times", mid,
                        set(heard) <= turns and set(heard) | set(tl.dropped_turns) == turns and starts == sorted(starts),
                        "segments do not cover the script turns in start order")
    if passed(StageId.ACOUSTIC):
        rows.load("render.json", mid, paths.render, AcousticRender)
    if passed(StageId.LABELS):
        rows.load("labels/transcript.json", mid, paths.transcript, Transcript)

    a = "manifest.jsonl"
    rows.expect(a, "one line per meeting that passed stage 8", mid, (entry is not None) == passed(StageId.LABELS),
                "in the manifest without passing stage 8" if entry else "passed stage 8 but not in the manifest")
    if entry is None:
        return
    s = entry.sampled
    rows.expect(a, "meeting id, room class, speaker count, split", mid,
                (entry.room_class, entry.n_speakers, entry.split) == (m.room_class, m.n_speakers, m.split),
                "differs from meeting.json")
    counts = {k: m.count(k) for k in POSITION_WORDS}
    rows.expect(a, "position counts: table, wall-seated, presenters, remote", mid, entry.position_counts == counts,
                f"{entry.position_counts} != {counts}")
    rows.expect(a, "per-speaker position type", mid,
                [(x.speaker_id, x.position_type, x.wall_reason) for x in entry.speakers]
                == [(p.speaker_id, p.position_type, p.wall_reason) for p in m.people], "differs from meeting.json")
    rows.expect(a, "NSC reference speaker per persona", mid,
                {x.speaker_id: x.nsc_speaker for x in entry.speakers} == voices, "differs from synthesis.json")
    rows.expect(a, "tags: mic type, table shape", mid,
                s.get("mic.type") in ("laptop", "centre") and s.get("table.shape") in TABLE_SHAPES,
                f"mic.type {s.get('mic.type')!r}, table.shape {s.get('table.shape')!r}")
    rows.expect(a, "hybrid", mid, s.get("dynamics.hybrid") is (counts["remote"] > 0), f"{s.get('dynamics.hybrid')!r}")
    rows.expect(a, "RT60: sampled and measured", mid,
                _num(s.get("acoustics.rt60")) and _num(s.get("room.rt60_target"))
                and _num(entry.acoustic.get("rt60_measured")), "missing")
    rows.expect(a, "SNR: sampled and measured", mid,
                _num(s.get("acoustics.snr_db")) and _num(entry.acoustic.get("snr_db")), "missing")
    rows.expect(a, "fit resample count", mid, s.get("fit_resamples") == m.fit_resamples, "differs from meeting.json")
    tt = entry.turn_taking
    rows.expect(a, "turn-taking: overlap share, backchannels per minute, interruption share", mid,
                0 <= tt.overlap_share <= 1 and tt.backchannels_per_min >= 0 and 0 <= tt.interruption_share <= 1
                and tt.duration > 0, "out of range")


def check_run(run_dir: Path) -> list[Row]:
    """Every row of the viewer's data contract, checked on every meeting of the run."""
    rp = RunPaths(Path(run_dir))
    rows = _Rows()
    try:
        cfg = read_run_config(rp.config)
    except (OSError, ValidationError, ValueError) as e:
        rows.expect("config.yaml", "parses", "run", False, str(e).splitlines()[0][:160])
        return list(rows.rows.values())
    rows.expect("config.yaml", "parses", "run", True, "")
    check_config(rows, cfg)
    splits = rows.load("speaker_splits.json", "run", rp.speaker_splits, SpeakerSplits)
    entries: dict[str, ManifestEntry] = {}
    lines = rp.manifest.read_text(encoding="utf-8").splitlines() if rp.manifest.is_file() else []
    for i, line in enumerate(x for x in lines if x.strip()):
        try:
            e = ManifestEntry.model_validate(json.loads(line))
        except (ValidationError, ValueError) as err:
            rows.expect("manifest.jsonl", "parses", f"line {i + 1}", False, str(err).splitlines()[0][:160])
            continue
        rows.expect("manifest.jsonl", "parses", e.meeting_id, True, "")
        entries[e.meeting_id] = e
    rows.expect("manifest.jsonl", "only meetings of this run", "run", set(entries) <= set(rp.meeting_ids()),
                f"unknown {sorted(set(entries) - set(rp.meeting_ids()))}")
    for mid in rp.meeting_ids():
        check_meeting(rows, rp.meeting(mid), cfg, splits, entries.get(mid))
    return sorted(rows.rows.values(), key=lambda r: (_ORDER.index(r.artifact) if r.artifact in _ORDER else 99))


_ORDER = ["manifest.jsonl", "config.yaml", "meeting.json", "status.json", "script.json", "script_validation.json",
          "voice_validation.json", "timeline.json", "speaker_splits.json", "prompts/<speaker_id>.txt",
          "synthesis.json", "render.json", "labels/transcript.json"]


def format_rows(rows: list[Row], *, max_problems: int = 3) -> str:
    out = []
    for r in rows:
        mark = "ok  " if r.ok else "FAIL"
        out.append(f"{mark}  {r.artifact:<26} {r.fields}  ({len(r.checked)} checked)")
        out.extend(f"        {p}" for p in r.problems[:max_problems])
        if len(r.problems) > max_problems:
            out.append(f"        ... and {len(r.problems) - max_problems} more")
    return "\n".join(out)

