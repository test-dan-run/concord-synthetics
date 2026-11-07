"""Export packs: the kept meetings of a run, as files the fine-tuning team can use without the pipeline.

Every meeting always gets its room recording, its labels and its details. Each speaker's own audio is optional, in
any mix of two cuts and two sounds:

- cut: `clips` (one WAV per line, listed in clips.csv with speaker, start, end, text and text_norm) or `full`
  (one WAV per speaker, as long as the recording and lined up with it to the sample);
- sound: `clean` (the voice alone, stage 6's tracks) or `room` (as heard at the mic, stage 7's stems).

`noise` adds the noise on its own (stage 7). With `room` and `noise`, the speakers plus the noise add up to the
recording before clipping and the codec. Every WAV is 16-bit mono PCM at 16 kHz, resampled when a run used another
rate. Clips are cut at the timeline's bounds, so a line cut short by an interruption ends where it was cut; a room
clip keeps a short tail for the echo.

    <out>/README.md
    <out>/manifest.jsonl                   the kept meetings' manifest lines
    <out>/<meeting_id>/mixture.wav
    <out>/<meeting_id>/meeting.json, timeline.json, labels/meeting.rttm, labels/transcript.json
    <out>/<meeting_id>/speakers/{clean,room}/<speaker_id>.wav
    <out>/<meeting_id>/clips/{clean,room}/<turn_id>.wav and clips.csv
    <out>/<meeting_id>/noise.wav
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..artifacts.io import atomic_write_text, read_model
from ..artifacts.labels import Timeline, Transcript
from ..artifacts.paths import MeetingPaths, RunPaths

SAMPLE_RATE = 16000
ROOM_TAIL_S = 0.3  # kept after a room clip's end, for the echo


@dataclass(frozen=True)
class PackOptions:
    clips: bool = True
    full: bool = False
    clean: bool = True
    room: bool = True
    noise: bool = False

    def sounds(self) -> list[str]:
        return [s for s, on in (("clean", self.clean), ("room", self.room)) if on]


def _read(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SAMPLE_RATE:
        import soxr

        x = soxr.resample(x, sr, SAMPLE_RATE).astype(np.float32)
    return x


def _write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16", format="WAV")


def _source(p: MeetingPaths, sound: str, speaker_id: str) -> Path:
    return p.track(speaker_id) if sound == "clean" else p.stem(speaker_id)


def missing(p: MeetingPaths, opts: PackOptions, speakers: Iterable[str]) -> list[str]:
    """Files the options need that the meeting does not have (room audio and noise need stems from stage 7)."""
    need = [p.mixture, p.rttm, p.transcript, p.timeline]
    if opts.clips or opts.full:
        need += [_source(p, s, spk) for s in opts.sounds() for spk in speakers]
    if opts.noise:
        need.append(p.noise)
    return [str(f.relative_to(p.dir)) for f in need if not f.is_file()]


def pack_meeting(p: MeetingPaths, out: Path, opts: PackOptions) -> dict:
    """One meeting's folder of the pack. Returns counts for the progress report."""
    tl = read_model(p.timeline, Timeline)
    tr = read_model(p.transcript, Transcript)
    speakers = sorted({s.speaker_id for s in tl.segments})
    gaps = missing(p, opts, speakers)
    if gaps:
        raise FileNotFoundError(f"{p.meeting_id} has no {', '.join(gaps)}")
    d = out / p.meeting_id
    d.mkdir(parents=True, exist_ok=True)
    _write(d / "mixture.wav", _read(p.mixture))
    for name in ("meeting.json", "timeline.json", "labels/meeting.rttm", "labels/transcript.json"):
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p.dir / name, d / name)
    n_files = 6
    text = {u.turn_id: u for u in tr.utterances}
    rows = []
    for sound in opts.sounds() if (opts.clips or opts.full) else []:
        for spk in speakers:
            x = _read(_source(p, sound, spk))
            if opts.full:
                _write(d / "speakers" / sound / f"{spk}.wav", x)
                n_files += 1
            if opts.clips:
                for s in (s for s in tl.segments if s.speaker_id == spk):
                    tail = ROOM_TAIL_S if sound == "room" else 0.0
                    a, b = int(round(s.start * SAMPLE_RATE)), int(round((s.end + tail) * SAMPLE_RATE))
                    _write(d / "clips" / sound / f"{s.turn_id}.wav", x[a:min(b, len(x))])
                    n_files += 1
                    if sound == opts.sounds()[0]:
                        u = text.get(s.turn_id)
                        rows.append({"clip": f"{s.turn_id}.wav", "speaker_id": spk, "start": round(s.start, 3),
                                     "end": round(s.end, 3), "text": u.text if u else "",
                                     "text_norm": u.text_norm if u else "", "truncated": s.truncated_at is not None})
            del x
    if rows:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["start"], r["clip"])))
        atomic_write_text(d / "clips" / "clips.csv", buf.getvalue())
    if opts.noise:
        _write(d / "noise.wav", _read(p.noise))
        n_files += 1
    return {"meeting_id": p.meeting_id, "files": n_files, "clips": len(rows)}


def export_run(run_dir: Path, out: Path, meeting_ids: list[str], opts: PackOptions,
               progress: Callable[[int, int, str], None] | None = None) -> dict:
    """The pack for `meeting_ids` (the kept meetings) of a run, written into `out`."""
    rp = RunPaths(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for i, mid in enumerate(meeting_ids):
        if progress:
            progress(i, len(meeting_ids), mid)
        done.append(pack_meeting(rp.meeting(mid), out, opts))
    lines = []
    if rp.manifest.is_file():
        keep = set(meeting_ids)
        lines = [ln for ln in rp.manifest.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and json.loads(ln)["meeting_id"] in keep]
    atomic_write_text(out / "manifest.jsonl", "".join(ln + "\n" for ln in lines))
    atomic_write_text(out / "README.md", readme(run_dir.name, meeting_ids, opts))
    if progress:
        progress(len(meeting_ids), len(meeting_ids), "")
    return {"run": run_dir.name, "out": str(out), "meetings": len(done), "options": asdict(opts),
            "files": sum(m["files"] for m in done), "clips": sum(m["clips"] for m in done)}


def readme(run_id: str, meeting_ids: list[str], opts: PackOptions) -> str:
    parts = [f"# Synthetics pack: {run_id}", "",
             f"{len(meeting_ids)} synthetic Singapore meetings with exact labels. Every WAV is 16-bit mono PCM at "
             "16 kHz. `docs/OUTPUT_CONTRACT.md` in the Synthetics app describes every field.", "",
             "Each meeting folder holds:", "",
             "- `mixture.wav`: the room recording.",
             "- `labels/meeting.rttm` (who spoke when) and `labels/transcript.json` (words with times).",
             "- `meeting.json` (room, seats, mic, speaker profiles) and `timeline.json` (every line's start and end)."]
    if opts.full:
        parts.append("- `speakers/<sound>/<speaker_id>.wav`: each speaker alone, as long as the recording and lined "
                     "up with it.")
    if opts.clips:
        parts.append("- `clips/<sound>/<turn_id>.wav`, listed in `clips/clips.csv` with speaker, start, end, text and "
                     "text_norm (one clip per line; a line cut short by an interruption ends where it was cut).")
    if opts.clips or opts.full:
        parts.append(f"- Sounds: {', '.join(opts.sounds())}. `clean` is the voice alone; `room` is the voice as heard "
                     "at the mic (room echo, distance, mic), at the recording's loudness, with no noise.")
    if opts.noise:
        parts.append("- `noise.wav`: the background noise alone, as heard at the mic. The room speakers plus the "
                     "noise add up to the recording before clipping and compression.")
    return "\n".join(parts) + "\n"
