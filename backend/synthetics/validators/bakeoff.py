"""`synthetics bakeoff` (M4): the phase C models on real NSC speakers, before stage 5's choices are final.

For each sampled speaker, the best clean clip is the reference and the speaker's other clean clips are targets:
their transcripts (natural Singlish from that speaker) are cloned from the reference, so every clone has a real
recording of the same words by the same person to compare with. Measured per clone, for each TTS mode tried:

- speaker similarity in both SV models, to the reference and to the real recording of the target;
- WER (CER for short texts) through both ASRs, on the clone and on the real recording (the ASR baseline), and
  whether the clone would pass stage 5 only through the Singlish ASR (asr_limited);
- MOS of the clone, of the real recording and of the reference;
- speaker identification: whether the clone's nearest reference among all sampled speakers is its own.
Between speakers: centroid cosines of clones and of real clips, per model.

Room colour: for a few speakers the same targets are also cloned from the raw (uncleaned) reference, and both sets
are compared on MOS and similarity; the WAVs are written to `listen/` for a listening check.
Writes report.json, report.md and clips.jsonl to the output directory.
"""

from __future__ import annotations

import json
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

    def run(self, speakers: list[str], *, targets: int, modes: list[str], raw_speakers: int, seed: int) -> dict:
        th = self.cfg.thresholds
        rows: list[dict] = []
        ref_emb, real_emb = {}, {}
        listen = self.out / "listen"
        raw_set = set(speakers[:raw_speakers])
        for i, s in enumerate(speakers, 1):
            clips = self.refs.clips(s)
            ref_clip, tgts = clips[0], clips[1:1 + targets]
            ref = self.refs.voice_ref(ref_clip.pair_id)
            ref_emb[s] = self._emb(ref.audio)
            refs_used = [("clean", ref)]
            if s in raw_set and ref_clip.raw_path is not None and ref_clip.raw_path.is_file():
                refs_used.append(("raw", VoiceRef(Audio.load(ref_clip.raw_path), ref.text, ref.clip_id)))
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
                        clone = self._clone(text, r, seed + len(rows), mode)
                        e = self._emb(clone)
                        asr = self._asr(clone, t.text)
                        row = {"speaker": s, "target": t.pair_id, "ref": ref_clip.pair_id, "ref_kind": ref_kind,
                               "mode": mode, "text": text, "duration": round(clone.duration, 3),
                               "sv_ref": {n: round(float(e[n] @ ref_emb[s][n]), 4) for n in SV},
                               "sv_real": {n: round(float(e[n] @ real_e[n]), 4) for n in SV},
                               "real_sv_ref": {n: round(float(real_e[n] @ ref_emb[s][n]), 4) for n in SV},
                               "asr": asr, "real_asr": real_asr, "mos": round(float(self.q.score([clone])[0]), 3),
                               "real_mos": round(real_mos, 3), "ref_mos": ref_clip.mos, "_emb": e}
                        rows.append(row)
                        if s in raw_set:
                            listen.mkdir(parents=True, exist_ok=True)
                            clone.save(listen / f"{s}_{t.pair_id}_{mode}_{ref_kind}.wav")
                if s in raw_set:
                    real.save(listen / f"{s}_{t.pair_id}_real.wav")
            if s in raw_set:
                for kind, r in refs_used:
                    r.audio.save(listen / f"{s}_ref_{kind}.wav")
            self.log(f"[{i}/{len(speakers)}] {s}: {len(tgts)} targets x {len(modes)} mode(s)")

        # speaker identification: nearest reference among all sampled speakers
        names = sorted(ref_emb)
        for row in rows:
            for n in SV:
                sims = [float(row["_emb"][n] @ ref_emb[o][n]) for o in names]
                row.setdefault("id_correct", {})[n] = names[int(np.argmax(sims))] == row["speaker"]
        report = {"services": self.services, "speakers": len(speakers), "targets_per_speaker": targets,
                  "modes": modes, "thresholds": {"sv": th.sv, "wer_max": th.wer_max, "cer_max_short": th.cer_max_short,
                                                 "mos_min": th.mos_min},
                  "by_mode": {m: self._summarise([r for r in rows if r["mode"] == m and r["ref_kind"] == "clean"])
                              for m in modes},
                  "real_baseline": self._baseline(rows, modes[0]),
                  "separability": self._separability(rows, real_emb, modes[0]),
                  "room_colour": self._room_colour(rows, modes[0])}
        self.out.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.out / "clips.jsonl", "".join(
            json.dumps({k: v for k, v in r.items() if k != "_emb"}, ensure_ascii=False) + "\n" for r in rows))
        atomic_write_text(self.out / "report.json", json.dumps(report, indent=2) + "\n")
        atomic_write_text(self.out / "report.md", render_markdown(report))
        return report

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
        out["mos_pass_rate"] = round(float(np.mean([r["mos"] >= th.mos_min for r in rows])), 4) if rows else None
        out["stage5_pass_rate"] = round(float(np.mean([
            all(r["sv_ref"][n] >= th.sv[n] for n in SV) and a and r["mos"] >= th.mos_min
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
                  f"| MOS pass rate (floor {r['thresholds']['mos_min']}) | {m['mos_pass_rate']} |",
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


def run_bakeoff(*, out_dir: Path, mode: str = "aicore", n_speakers: int = 60, targets: int = 3,
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

