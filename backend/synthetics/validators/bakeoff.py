"""`synthetics bakeoff`: the phase C models on real NSC speakers, to choose the stage 4 and 5 models and settings.

For each sampled speaker, the best clean clip is the reference and the speaker's other clean clips are targets:
their transcripts (natural Singlish from that speaker) are cloned from the reference, so every clone has a real
recording of the same words by the same person to compare with. Measured per clone, for each TTS mode tried:

- speaker similarity in both SV models, to the reference and to the real recording of the target;
- WER (CER for short texts) through both ASRs, on the clone and on the real recording (the ASR baseline), and
  whether the clone would pass stage 5 only through the Singlish ASR (asr_limited);
- MOS of the clone, of the real recording and of the reference (`ref_utmos`, the clone's speaker-relative floor in
  stage 5; `ref_mos` is the reference's DNSMOS from cleanup);
- speaker identification: whether the clone's nearest reference among all sampled speakers is its own.
Between speakers: centroid cosines of clones and of real clips, per model.

Room colour: for a few speakers the same targets are also cloned from the raw (uncleaned) reference, and both sets
are compared on MOS and similarity; the WAVs are written to `listen/` for a listening check.
Writes report.json, report.md and clips.jsonl to the output directory.

Resume: each finished speaker is appended to `progress.jsonl` (rows and embeddings), headed by the settings the
results depend on. A rerun into the same directory skips the speakers already there, so a crash costs only the
speaker in progress, and a larger run reuses a smaller one's speakers. If the settings changed, the old file is
moved to `progress.jsonl.old` and the run starts over. Seeds are per clip, so a resumed run equals an unbroken one.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import numpy as np

from ..artifacts.io import atomic_write_text
from ..backends.base import Audio, VoiceRef
from ..config.schema import Role, RunConfig
from ..nsc.references import ReferenceIndex
from ..text.singlish import Lexicon, text_norm, tts_input
from .voice_checks import asr_words, centroid, score_asr

ROLES = (Role.TTS, Role.ASR_PRIMARY, Role.ASR_SINGLISH, Role.SV_WAVLM, Role.SV_REDIMNET, Role.QUALITY)
SV = (Role.SV_WAVLM.value, Role.SV_REDIMNET.value)
PROGRESS_VERSION = 1


def select_speakers(refs: ReferenceIndex, n: int, seed: int) -> list[str]:
    """`n` speakers with at least two clean clips, spread over gender x ethnicity groups (round robin)."""
    by_group: dict[tuple[str, str], list[str]] = {}
    for s in refs.speakers():
        clips = refs.clips(s)
        if len(clips) >= 2:
            by_group.setdefault((clips[0].gender, clips[0].ethnicity), []).append(s)
    rng = np.random.default_rng(seed)
    queues = {g: list(rng.permutation(v)) for g, v in sorted(by_group.items())}
    out: list[str] = []
    while len(out) < n and any(queues.values()):
        for g in sorted(queues):
            if queues[g] and len(out) < n:
                out.append(str(queues[g].pop()))
    return sorted(out)


def _speakable(text: str, lex: Lexicon) -> str:
    """The TTS text for a reference transcript (non-speech tags dropped)."""
    return " ".join(s.value for s in tts_input(text, lex) if s.kind == "speech").strip()


def clip_seed(seed: int, *parts: str) -> int:
    """The TTS seed of one clone: fixed by the run seed and the clip, not by the order clips are made in."""
    return int(hashlib.sha256("|".join([str(seed), *parts]).encode()).hexdigest()[:8], 16)


def _vecs(d: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {n: [float(x) for x in v] for n, v in d.items()}


def _arrays(d: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {n: np.asarray(v, np.float32) for n, v in d.items()}


def _summary(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, float)
    return {"n": len(a), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "p10": round(float(np.quantile(a, 0.1)), 4), "p90": round(float(np.quantile(a, 0.9)), 4)}


class Bakeoff:
    def __init__(self, cfg: RunConfig, backends, refs: ReferenceIndex, out_dir: Path, *,
                 log: Callable[[str], None] = print):
        self.cfg, self.refs, self.out, self.log = cfg, refs, Path(out_dir), log
        self.lex = Lexicon.load(cfg.lexicon.path)
        self.tts = backends.get(Role.TTS)
        self.asr = {Role.ASR_PRIMARY.value: backends.get(Role.ASR_PRIMARY),
                    Role.ASR_SINGLISH.value: backends.get(Role.ASR_SINGLISH)}
        self.sv = {r: backends.get(Role(r)) for r in SV}
        self.q = backends.get(Role.QUALITY)
        self.services = {r.value: cfg.backends.roles.get(r, "") for r in ROLES}

    def _emb(self, a: Audio) -> dict[str, np.ndarray]:
        return {n: e.embed([a])[0] for n, e in self.sv.items()}

    def _asr(self, a: Audio, text: str) -> dict:
        th, vc = self.cfg.thresholds, self.cfg.voice
        words = asr_words(text_norm(text, self.lex), self.lex, ignore_fillers=vc.wer_ignore_fillers, normalised=True)
        out = {}
        for name, client in self.asr.items():
            hyp = client.transcribe(a, language=vc.asr_language)
            s = score_asr(words, hyp, self.lex, short_words=th.short_utterance_words,
                          ignore_fillers=vc.wer_ignore_fillers)
            out[name] = {"hyp": hyp, "error": round(s.error, 4), "short": s.short,
                         "pass": s.error <= (th.cer_max_short if s.short else th.wer_max)}
        return out

    def _clone(self, text: str, ref: VoiceRef, seed: int, mode: str) -> Audio:
        if hasattr(self.tts, "params"):
            v = self.cfg.voice
            self.tts.params = {"mode": mode, "cfg_value": v.cfg_value, "inference_timesteps": v.inference_timesteps}
        from ..voice.audio_ops import trim

        return trim(self.tts.clone(text, ref, seed=seed).resample(self.cfg.sample_rate), pad_s=self.cfg.voice.trim_pad_s,
                    below_db=self.cfg.voice.trim_db)

    def _settings(self, targets: int, modes: list[str], seed: int) -> dict:
        """What a speaker's stored rows depend on; a stored speaker is reused only under the same settings."""
        th, v = self.cfg.thresholds, self.cfg.voice
        return {"version": PROGRESS_VERSION, "targets": targets, "modes": modes, "seed": seed,
                "services": self.services, "refs_index": self.refs.sha256, "lexicon": self.lex.version,
                "sample_rate": self.cfg.sample_rate,
                "voice": {k: getattr(v, k) for k in ("cfg_value", "inference_timesteps", "trim_pad_s", "trim_db",
                                                      "asr_language", "wer_ignore_fillers")},
                "asr": {"wer_max": th.wer_max, "cer_max_short": th.cer_max_short,
                        "short_utterance_words": th.short_utterance_words}}

    def _load_progress(self, path: Path, settings: dict) -> dict[str, dict]:
        """Speaker -> stored record; an unreadable last line (a crash mid-write) is ignored."""
        if not path.is_file():
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
        try:
            head = json.loads(lines[0]) if lines else {}
        except json.JSONDecodeError:
            head = {}
        if head.get("settings") != json.loads(json.dumps(settings)):
            path.replace(path.with_name(path.name + ".old"))
            self.log(f"{path.name}: settings changed, starting over (old file kept as {path.name}.old)")
            return {}
        done = {}
        for line in lines[1:]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["speaker"]] = rec
        return done

    def run(self, speakers: list[str], *, targets: int, modes: list[str], raw_speakers: int, seed: int) -> dict:
        th = self.cfg.thresholds
        rows: list[dict] = []
        ref_emb, real_emb = {}, {}
        listen = self.out / "listen"
        raw_set = set(speakers[:raw_speakers])
        self.out.mkdir(parents=True, exist_ok=True)
        progress = self.out / "progress.jsonl"
        settings = self._settings(targets, modes, seed)
        done = self._load_progress(progress, settings)
        if not progress.is_file():
            atomic_write_text(progress, json.dumps({"settings": settings}) + "\n")
        reused = 0
        for i, s in enumerate(speakers, 1):
            if s in done:
                rec = done[s]
                ref_emb[s] = _arrays(rec["ref_emb"])
                real_emb[s] = [_arrays(e) for e in rec["real_emb"]]
                rows += [{**r, "_emb": _arrays(r["_emb"])} for r in rec["rows"]]
                self._fill_ref_utmos(s, rows, len(rows) - len(rec["rows"]))
                reused += 1
                continue
            first = len(rows)
            clips = self.refs.clips(s)
            ref_clip, tgts = clips[0], clips[1:1 + targets]
            ref = self.refs.voice_ref(ref_clip.pair_id)
            ref_emb[s] = self._emb(ref.audio)
            refs_used = [("clean", ref)]
            if s in raw_set and ref_clip.raw_path is not None and ref_clip.raw_path.is_file():
                refs_used.append(("raw", VoiceRef(Audio.load(ref_clip.raw_path), ref.text, ref.clip_id)))
            ref_utmos = {kind: round(float(self.q.score([r.audio])[0]), 3) for kind, r in refs_used}
            for t in tgts:
                real = self.refs.voice_ref(t.pair_id).audio
                real_e = self._emb(real)
                real_emb.setdefault(s, []).append(real_e)
                real_asr = self._asr(real, t.text)
                real_mos = float(self.q.score([real])[0])
                text = _speakable(t.text, self.lex)
                for mode in modes:
                    for ref_kind, r in refs_used:
                        if ref_kind == "raw" and mode != modes[0]:
                            continue
                        cs = clip_seed(seed, s, t.pair_id, mode, ref_kind)
                        clone = self._clone(text, r, cs, mode)
                        e = self._emb(clone)
                        asr = self._asr(clone, t.text)
                        row = {"speaker": s, "target": t.pair_id, "ref": ref_clip.pair_id, "ref_kind": ref_kind,
                               "mode": mode, "seed": cs, "text": text, "duration": round(clone.duration, 3),
                               "sv_ref": {n: round(float(e[n] @ ref_emb[s][n]), 4) for n in SV},
                               "sv_real": {n: round(float(e[n] @ real_e[n]), 4) for n in SV},
                               "real_sv_ref": {n: round(float(real_e[n] @ ref_emb[s][n]), 4) for n in SV},
                               "asr": asr, "real_asr": real_asr, "mos": round(float(self.q.score([clone])[0]), 3),
                               "real_mos": round(real_mos, 3), "ref_mos": ref_clip.mos,
                               "ref_utmos": ref_utmos[ref_kind], "_emb": e}
                        rows.append(row)
                        if s in raw_set:
                            listen.mkdir(parents=True, exist_ok=True)
                            clone.save(listen / f"{s}_{t.pair_id}_{mode}_{ref_kind}.wav")
                if s in raw_set:
                    real.save(listen / f"{s}_{t.pair_id}_real.wav")
            if s in raw_set:
                for kind, r in refs_used:
                    r.audio.save(listen / f"{s}_ref_{kind}.wav")
            rec = {"speaker": s, "ref_emb": _vecs(ref_emb[s]), "real_emb": [_vecs(e) for e in real_emb.get(s, [])],
                   "rows": [{**r, "_emb": _vecs(r["_emb"])} for r in rows[first:]]}
            with progress.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.log(f"[{i}/{len(speakers)}] {s}: {len(tgts)} targets x {len(modes)} mode(s)")
        if reused:
            self.log(f"reused {reused} speaker(s) from {progress}")

        # speaker identification: nearest reference among all sampled speakers
        names = sorted(ref_emb)
        for row in rows:
            for n in SV:
                sims = [float(row["_emb"][n] @ ref_emb[o][n]) for o in names]
                row.setdefault("id_correct", {})[n] = names[int(np.argmax(sims))] == row["speaker"]
        report = {"services": self.services, "speakers": len(speakers), "targets_per_speaker": targets,
                  "modes": modes, "thresholds": {"sv": th.sv, "wer_max": th.wer_max, "cer_max_short": th.cer_max_short,
                                                 "mos_min": th.mos_min, "mos_drop_max": th.mos_drop_max},
                  "by_mode": {m: self._summarise([r for r in rows if r["mode"] == m and r["ref_kind"] == "clean"])
                              for m in modes},
                  "real_baseline": self._baseline(rows, modes[0]),
                  "separability": self._separability(rows, real_emb, modes[0]),
                  "room_colour": self._room_colour(rows, modes[0])}
        atomic_write_text(self.out / "clips.jsonl", "".join(
            json.dumps({k: v for k, v in r.items() if k != "_emb"}, ensure_ascii=False) + "\n" for r in rows))
        atomic_write_text(self.out / "report.json", json.dumps(report, indent=2) + "\n")
        atomic_write_text(self.out / "report.md", render_markdown(report))
        return report

    def _mos_ok(self, r: dict) -> bool:
        """Stage 5's MOS rule: an absolute floor and a floor relative to the reference clip."""
        th = self.cfg.thresholds
        return r["mos"] >= max(th.mos_min, r["ref_utmos"] - th.mos_drop_max)

    def _fill_ref_utmos(self, speaker: str, rows: list[dict], first: int) -> None:
        """Stored rows from before `ref_utmos` was recorded: score the reference clips now."""
        missing = {r["ref_kind"] for r in rows[first:] if "ref_utmos" not in r}
        if not missing:
            return
        clip = self.refs.best(speaker)
        audio = {"clean": lambda: self.refs.voice_ref(clip.pair_id).audio, "raw": lambda: Audio.load(clip.raw_path)}
        score = {k: round(float(self.q.score([audio[k]()])[0]), 3) for k in missing}
        for r in rows[first:]:
            r.setdefault("ref_utmos", score.get(r["ref_kind"]))

    def _summarise(self, rows: list[dict]) -> dict:
        th = self.cfg.thresholds
        p, s = Role.ASR_PRIMARY.value, Role.ASR_SINGLISH.value
        out = {"clones": len(rows)}
        for n in SV:
            out[f"{n}_to_ref"] = _summary([r["sv_ref"][n] for r in rows])
            out[f"{n}_to_real"] = _summary([r["sv_real"][n] for r in rows])
            out[f"{n}_pass_rate"] = round(float(np.mean([r["sv_ref"][n] >= th.sv[n] for r in rows])), 4) if rows else None
            out[f"{n}_id_accuracy"] = round(float(np.mean([r["id_correct"][n] for r in rows])), 4) if rows else None
        out["both_sv_pass_rate"] = round(float(np.mean([all(r["sv_ref"][n] >= th.sv[n] for n in SV) for r in rows])), 4) if rows else None
        for name in (p, s):
            out[f"{name}_error"] = _summary([r["asr"][name]["error"] for r in rows])
            out[f"{name}_pass_rate"] = round(float(np.mean([r["asr"][name]["pass"] for r in rows])), 4) if rows else None
        asr_pass = [r["asr"][p]["pass"] or r["asr"][s]["pass"] for r in rows]
        out["asr_pass_rate"] = round(float(np.mean(asr_pass)), 4) if rows else None
        out["asr_limited_rate"] = round(float(np.mean([not r["asr"][p]["pass"] and r["asr"][s]["pass"] for r in rows])), 4) if rows else None
        out["mos"] = _summary([r["mos"] for r in rows])
        out["mos_pass_rate"] = round(float(np.mean([self._mos_ok(r) for r in rows])), 4) if rows else None
        out["stage5_pass_rate"] = round(float(np.mean([
            all(r["sv_ref"][n] >= th.sv[n] for n in SV) and a and self._mos_ok(r)
            for r, a in zip(rows, asr_pass, strict=True)])), 4) if rows else None
        return out

    def _baseline(self, rows: list[dict], mode: str) -> dict:
        seen = {(r["speaker"], r["target"]): r for r in rows if r["mode"] == mode and r["ref_kind"] == "clean"}
        rs = list(seen.values())
        out = {"recordings": len(rs), "mos": _summary([r["real_mos"] for r in rs])}
        for n in SV:
            out[f"{n}_real_to_ref"] = _summary([r["real_sv_ref"][n] for r in rs])
        for name in (Role.ASR_PRIMARY.value, Role.ASR_SINGLISH.value):
            out[f"{name}_error"] = _summary([r["real_asr"][name]["error"] for r in rs])
            out[f"{name}_pass_rate"] = round(float(np.mean([r["real_asr"][name]["pass"] for r in rs])), 4) if rs else None
        return out

    def _separability(self, rows: list[dict], real_emb: dict, mode: str) -> dict:
        out = {}
        for n in SV:
            clones: dict[str, list] = {}
            for r in rows:
                if r["mode"] == mode and r["ref_kind"] == "clean":
                    clones.setdefault(r["speaker"], []).append(r["_emb"][n])
            cc = {s: centroid(np.stack(v)) for s, v in clones.items()}
            rc = {s: centroid(np.stack([e[n] for e in v])) for s, v in real_emb.items()}
            pc = [float(cc[a] @ cc[b]) for a, b in combinations(sorted(cc), 2)]
            pr = [float(rc[a] @ rc[b]) for a, b in combinations(sorted(rc), 2)]
            out[n] = {"clone_centroids": _summary(pc), "real_centroids": _summary(pr),
                      "clone_pairs_over_pair_max": int(sum(v > self.cfg.thresholds.sv_pair_max[n] for v in pc)),
                      "real_pairs_over_pair_max": int(sum(v > self.cfg.thresholds.sv_pair_max[n] for v in pr)),
                      "pairs": len(pc)}
        return out

    def _room_colour(self, rows: list[dict], mode: str) -> dict:
        raw = {(r["speaker"], r["target"]): r for r in rows if r["ref_kind"] == "raw"}
        clean = {(r["speaker"], r["target"]): r for r in rows if r["ref_kind"] == "clean" and r["mode"] == mode}
        keys = sorted(set(raw) & set(clean))
        if not keys:
            return {"pairs": 0}
        out = {"pairs": len(keys), "mos_clean_ref": _summary([clean[k]["mos"] for k in keys]),
               "mos_raw_ref": _summary([raw[k]["mos"] for k in keys]),
               "mos_gain_from_cleanup": round(float(np.mean([clean[k]["mos"] - raw[k]["mos"] for k in keys])), 4),
               "listen_dir": "listen/"}
        for n in SV:
            out[f"{n}_to_real_clean_ref"] = _summary([clean[k]["sv_real"][n] for k in keys])
            out[f"{n}_to_real_raw_ref"] = _summary([raw[k]["sv_real"][n] for k in keys])
        return out


def render_markdown(r: dict) -> str:
    def fmt(s):
        return "-" if not s else f"{s['mean']:.3f} (median {s['median']:.3f}, p10 {s['p10']:.3f})"

    lines = ["# Phase C bake-off", "", f"Services: {', '.join(f'{k} = `{v}`' for k, v in r['services'].items())}.",
             f"{r['speakers']} NSC speakers, {r['targets_per_speaker']} target clips each.", ""]
    for mode, m in r["by_mode"].items():
        lines += [f"## TTS mode `{mode}` ({m['clones']} clones)", "",
                  "| Measure | Value |", "|---|---|"]
        for n in SV:
            lines += [f"| {n} to reference | {fmt(m[f'{n}_to_ref'])} |", f"| {n} to the real recording | {fmt(m[f'{n}_to_real'])} |",
                      f"| {n} pass rate (threshold {r['thresholds']['sv'][n]}) | {m[f'{n}_pass_rate']} |",
                      f"| {n} speaker-ID accuracy | {m[f'{n}_id_accuracy']} |"]
        for k in ("asr_primary", "asr_singlish"):
            lines += [f"| {k} error (WER, CER if short) | {fmt(m[f'{k}_error'])} |", f"| {k} pass rate | {m[f'{k}_pass_rate']} |"]
        lines += [f"| ASR-limited rate | {m['asr_limited_rate']} |", f"| MOS | {fmt(m['mos'])} |",
                  f"| MOS pass rate (at least {r['thresholds']['mos_min']} and reference - "
                  f"{r['thresholds']['mos_drop_max']}) | {m['mos_pass_rate']} |",
                  f"| would pass stage 5 | {m['stage5_pass_rate']} |", ""]
    b = r["real_baseline"]
    lines += ["## Real recordings (baseline)", "", "| Measure | Value |", "|---|---|", f"| MOS | {fmt(b['mos'])} |"]
    for n in SV:
        lines.append(f"| {n}, real target to reference | {fmt(b[f'{n}_real_to_ref'])} |")
    for k in ("asr_primary", "asr_singlish"):
        lines += [f"| {k} error | {fmt(b[f'{k}_error'])} |", f"| {k} pass rate | {b[f'{k}_pass_rate']} |"]
    lines += ["", "## Between-speaker separability (centroid cosine)", "", "| Model | Clones | Real | Clone pairs over pair max |",
              "|---|---|---|---|"]
    for n, s in r["separability"].items():
        lines.append(f"| {n} | {fmt(s['clone_centroids'])} | {fmt(s['real_centroids'])} | "
                     f"{s['clone_pairs_over_pair_max']} of {s['pairs']} (real: {s['real_pairs_over_pair_max']}) |")
    rc = r["room_colour"]
    lines += ["", "## Room colour (clean vs raw reference)", ""]
    if rc.get("pairs"):
        lines += [f"{rc['pairs']} targets cloned from both. MOS from the clean reference {fmt(rc['mos_clean_ref'])}, "
                  f"from the raw one {fmt(rc['mos_raw_ref'])} (mean gain {rc['mos_gain_from_cleanup']:+.3f}). "
                  f"Listening files are in `{rc['listen_dir']}`."]
    else:
        lines.append("No raw references were cloned.")
    return "\n".join(lines) + "\n"


def run_bakeoff(*, out_dir: Path, mode: str = "aicore", n_speakers: int = 30, targets: int = 3,
                modes: list[str] | None = None, raw_speakers: int = 10, seed: int = 1, lease: bool = True,
                keep_services: bool = False, log: Callable[[str], None] = print) -> dict:
    from ..backends.registry import Backends
    from ..config.loader import load_run_config
    from ..config.schema import Phase
    from ..nsc.settings import nsc_paths
    from ..orchestrator.lease import GpuLease
    from ..orchestrator.run import resolve

    cfg = resolve(load_run_config(overrides={"run_id": "bakeoff", "seed": seed, "n_meetings": 0, "backends.mode": mode}))
    backends = Backends(cfg.backends, ROLES)
    services = sorted({cfg.backends.roles[r] for r in cfg.phases.phases[Phase.C].roles})
    refs = ReferenceIndex.load(nsc_paths())
    speakers = select_speakers(refs, n_speakers, seed)
    with GpuLease(services, platform_dir=cfg.phases.platform_dir, no_evict=cfg.phases.no_evict,
                  timeout_s=cfg.phases.lease_timeout_s, enabled=lease and mode in ("aicore", "record"),
                  keep=keep_services):
        return Bakeoff(cfg, backends, refs, out_dir, log=log).run(
            speakers, targets=targets, modes=modes or [cfg.voice.tts_mode], raw_speakers=raw_speakers, seed=seed)

