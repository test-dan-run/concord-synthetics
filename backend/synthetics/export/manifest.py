"""Manifest entries (stage 8) and the run's manifest.jsonl.

Stage 8 writes one `labels/manifest_entry.json` per meeting. `write_run_manifest` collects the entries of every
meeting whose stage 8 passed into `manifest.jsonl` (one line per meeting, sorted by meeting id), so the run-level
file is always rebuilt from per-meeting artifacts and never edited in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..artifacts.io import atomic_write_text, read_model
from ..artifacts.labels import AcousticRender, ManifestEntry, ManifestSpeaker, Scalar, Timeline, Transcript
from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import Script, ScriptValidation
from ..artifacts.paths import MeetingPaths, RunPaths
from ..artifacts.voice import Synthesis, VoiceValidation
from ..config.schema import RunConfig, StageId

# meeting.json parts that are per person or per seat: the manifest keeps them in `speakers` or leaves them out
SKIP = {"people", "table.slots", "wall_chairs", "dynamics.turn_shares", "mic.array_geometry",
        "acoustics.noise_sources", "speakerphone.remote_speakers", "dynamics.remote_speakers",
        "dynamics.presentation.returns", "dynamics.presentation.presenters", "room.screen", "schema_version",
        "meeting_id", "seed", "split", "room_class", "n_speakers"}


def flatten(obj: object, prefix: str = "", skip: set[str] = SKIP) -> dict[str, Scalar]:
    """Nested dicts and lists -> dotted keys (`mic.position.0`). Keys in `skip` (dotted) are left out."""
    out: dict[str, Scalar] = {}
    if prefix in skip:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k), skip))
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}.{i}", skip))
    elif obj is None or isinstance(obj, bool | int | float | str):
        out[prefix] = obj
    else:
        out[prefix] = str(obj)
    return out


def sampled_params(m: MeetingSpec) -> dict[str, Scalar]:
    data = m.model_dump(mode="json")
    out = flatten(data)
    out["acoustics.noise_sources"] = ",".join(n.kind for n in m.acoustics.noise_sources)
    out["room.screen"] = m.room.screen is not None
    if m.dynamics.presentation is not None:
        out["dynamics.presentation.presenters"] = ",".join(m.dynamics.presentation.presenters)
    out["speakerphone"] = m.speakerphone is not None
    return dict(sorted(out.items()))


def acoustic_summary(r: AcousticRender) -> dict[str, Scalar]:
    out: dict[str, Scalar] = {"rt60_target": r.rt60_target, "rt60_measured": r.rt60_measured,
                              "rt60_error": r.rt60_error, "absorption_scale": r.absorption_scale,
                              "snr_target_db": r.snr_target_db, "snr_db": r.snr_db, "channels": r.channels,
                              "ism_order": r.ism_order, "ray_tracing": r.ray_tracing,
                              "impulse_responses": r.n_impulse_responses,
                              "noise": ",".join(sorted({n.kind for n in r.noise})),
                              "noise_sources": ",".join(sorted({n.source for n in r.noise})),
                              "events": sum(n.start is not None for n in r.noise)}
    out.update({f"device.{k}": v for k, v in sorted(r.device.items())})
    for spk, info in sorted(r.remote.items()):
        out.update({f"remote.{spk}.{k}": v for k, v in sorted(info.items())})
    return out


def build_entry(paths: MeetingPaths, cfg: RunConfig, *, aligner: str) -> ManifestEntry:
    m = read_model(paths.meeting, MeetingSpec)
    script = read_model(paths.script, Script)
    sv = read_model(paths.script_validation, ScriptValidation) if paths.script_validation.is_file() else None
    syn = read_model(paths.synthesis, Synthesis)
    vv = read_model(paths.voice_validation, VoiceValidation) if paths.voice_validation.is_file() else None
    tl = read_model(paths.timeline, Timeline)
    render = read_model(paths.render, AcousticRender)
    tr = read_model(paths.transcript, Transcript)

    speech = {p.speaker_id: 0.0 for p in m.people}
    for s in tl.segments:
        speech[s.speaker_id] = speech.get(s.speaker_id, 0.0) + s.end - s.start
    total = sum(speech.values()) or 1.0
    limited = {p.speaker_id: 0 for p in m.people}
    for u in (vv.utterances if vv else []):
        limited[u.speaker_id] = limited.get(u.speaker_id, 0) + int(u.asr_limited)
    speakers = [ManifestSpeaker(speaker_id=p.speaker_id,
                                nsc_speaker=(syn.voices[p.speaker_id].nsc_speaker if p.speaker_id in syn.voices
                                             else p.nsc_speaker.speaker),
                                split=m.split, position_type=p.position_type, wall_reason=p.wall_reason,
                                turn_share=round(speech[p.speaker_id] / total, 4), asr_limited=limited[p.speaker_id])
                for p in m.people]

    judge: dict[str, Scalar] = {"rounds": script.generator.round}
    if sv is not None:
        judge.update({"decision": sv.decision, "rules_passed": sv.rules_passed, "borderline": sv.borderline,
                      "primary_overall": sv.primary.scores.overall if sv.primary else None,
                      "primary_service": sv.primary.service if sv.primary else None,
                      "tiebreak_overall": sv.tiebreak.scores.overall if sv.tiebreak else None,
                      "tiebreak_service": sv.tiebreak.service if sv.tiebreak else None})
    voice: dict[str, Scalar] = {"service": syn.service, "utterances": len(syn.utterances),
                                "resynthesised": sum(u.attempt > 1 for u in syn.utterances),
                                "library_turns": sum(u.source != "tts" for u in syn.utterances),
                                "swaps": len(syn.swaps)}
    if vv is not None:
        voice.update({"decision": vv.decision, "checked": len(vv.utterances),
                      "passed": sum(u.passed for u in vv.utterances),
                      "asr_limited": sum(u.asr_limited for u in vv.utterances),
                      "pairs_too_close": sum(p.too_close for p in vv.pairs)})

    n_words = sum(len(u.words) for u in tr.utterances)
    covs = [u.align_coverage for u in tr.utterances if u.text_norm]
    labels: dict[str, Scalar] = {"aligner": aligner, "utterances": len(tr.utterances), "words": n_words,
                                 "truncated": sum(u.truncated for u in tr.utterances),
                                 "align_coverage_min": round(min(covs), 4) if covs else None,
                                 "align_coverage_mean": round(sum(covs) / len(covs), 4) if covs else None,
                                 "dropped_turns": len(tl.dropped_turns)}

    rel = paths.dir.relative_to(paths.run_dir)
    files = {"meeting": "meeting.json", "script": "script.json", "timeline": "timeline.json", "tracks": "tracks",
             "mixture": "mixture.wav", "rttm": "labels/meeting.rttm", "transcript": "labels/transcript.json",
             "render": "render.json"}
    if "mixture_mc" in render.files:
        files["mixture_mc"] = "mixture_mc.wav"
    from ..stages.runner import meeting_seed

    return ManifestEntry(run_id=cfg.run_id, meeting_id=m.meeting_id, seed=meeting_seed(cfg.seed, m.meeting_id),
                         split=m.split, room_class=m.room_class, n_speakers=m.n_speakers, duration=tl.stats.duration,
                         sampled=sampled_params(m),
                         stages_enabled={s.name.lower(): cfg.enabled(s) for s in StageId},
                         position_counts={k: m.count(k) for k in ("table", "wall", "presenter", "remote")},
                         speakers=speakers, judge=judge, voice=voice, turn_taking=tl.stats,
                         files={k: str(rel / v) for k, v in files.items()},
                         acoustic=acoustic_summary(render), labels=labels)


def write_run_manifest(run_dir: Path) -> Path:
    """manifest.jsonl from the entries of meetings whose stage 8 passed."""
    from ..orchestrator.status import StatusStore

    rp = RunPaths(Path(run_dir))
    store = StatusStore(run_dir)
    lines = []
    for mid in rp.meeting_ids():
        p = rp.meeting(mid).manifest_entry
        if store.get(mid).get(StageId.LABELS).state == "pass" and p.is_file():
            entry = read_model(p, ManifestEntry)
            lines.append(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")))
    atomic_write_text(rp.manifest, "\n".join(lines) + ("\n" if lines else ""))
    return rp.manifest
