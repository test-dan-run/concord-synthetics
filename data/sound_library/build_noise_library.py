"""Build `<SOUND_LIBRARY_DIR>/noise/` (clips + index.yaml) from downloaded DEMAND and ESC-50 archives.

    uv run python data/sound_library/build_noise_library.py \
        --demand <dir with OOFFICE/ch01.wav ...> --esc50 <dir with ESC-50-master/>

Sources (checked 2026-10-18):
- DEMAND, https://zenodo.org/records/1227121, OOFFICE_16k.zip (md5 7b61cc2d182d5a654cb9c3101ddd4041). CC BY-SA 3.0
  (DEMAND.pdf; the Zenodo metadata field says CC BY 4.0, the stricter one is recorded). OMEETING is not used: it
  was recorded "while the microphone array is discussed", so it holds speech. OOFFICE ("a small office with three
  people using computers") is used as room tone: 0-135 s and 155-260 s, leaving out the stretch around 140-150 s
  where the voicing check below found pitch, as a 4-channel file (DEMAND ch01, ch02, ch06, ch05).
- ESC-50, https://github.com/karolpiczak/ESC-50 (master.zip). CC BY-NC 3.0 as a whole; each clip's Freesound
  source and original licence come from its LICENSE file. Categories used: keyboard_typing, mouse_click (keyboard),
  door_wood_knock, door_wood_creaks (door), pouring_water, can_opening (cup).

Every clip is checked for speech: a clip whose share of loud 40 ms frames with a clear pitch (90-300 Hz) is above
`MAX_VOICED` is left out. Real NSC speech scores about 0.2, the OOFFICE room tone 0.0.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import yaml
from scipy import signal

SR = 16000
MAX_VOICED = 0.03
VERSION = "noise-1"
DEFAULT_ROOT = Path("/home/user/code/datasets/sound-library")

DEMAND = {"url": "https://zenodo.org/records/1227121", "licence": "CC BY-SA 3.0",
          "attribution": "DEMAND, J. Thiemann, N. Ito, E. Vincent (2013)"}
DEMAND_CHANNELS = ("ch01", "ch02", "ch06", "ch05")  # around a 5 cm cell of the array, like a 4-capsule ring
DEMAND_SPANS = (("a", 0.0, 135.0), ("b", 155.0, 260.0))
ESC50_KINDS = {"keyboard_typing": "keyboard", "mouse_click": "keyboard", "door_wood_knock": "door",
               "door_wood_creaks": "door", "pouring_water": "cup", "can_opening": "cup"}
LICENCE_LINE = re.compile(r"^- \[(?P<stem>[^\]]+)\.ogg\]: clip derived from .*\((?P<url>http[^)]+)\) by (?P<by>.+) "
                          r"\[(?P<lic>[^\]]+)\]\s*$")


def voiced_share(x: np.ndarray, sr: int) -> float:
    x = signal.sosfiltfilt(signal.butter(4, [200, 3400], "bandpass", fs=sr, output="sos"), x.astype(np.float64))
    n = int(0.04 * sr)
    if len(x) < 2 * n:
        return 0.0
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[:: n // 2]
    e = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-12)
    loud = e > np.median(e) + 6
    lo, hi = int(sr / 300), int(sr / 90)
    voiced = 0
    for f in fr[loud]:
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[n - 1:]
        voiced += ac[lo:hi].max() / (ac[0] + 1e-12) > 0.6
    return voiced / len(fr)


def trim(x: np.ndarray, sr: int, floor_db: float = 40.0, pad_s: float = 0.05) -> np.ndarray:
    a = np.abs(x)
    if not a.size or a.max() <= 0:
        return x[:0]
    idx = np.flatnonzero(a > a.max() * 10 ** (-floor_db / 20))
    pad = int(pad_s * sr)
    return x[max(0, idx[0] - pad): idx[-1] + 1 + pad]


def demand(src: Path, out: Path) -> list[dict]:
    chans = [sf.read(str(src / "OOFFICE" / f"{c}.wav"), dtype="float32") for c in DEMAND_CHANNELS]
    assert all(sr == SR for _, sr in chans), "use the 16 kHz DEMAND files"
    x = np.stack([c for c, _ in chans], axis=1)
    entries = []
    (out / "hvac").mkdir(parents=True, exist_ok=True)
    for tag, t0, t1 in DEMAND_SPANS:
        part = x[int(t0 * SR): int(t1 * SR)]
        v = max(voiced_share(part[i:i + 10 * SR, 0], SR) for i in range(0, len(part) - 10 * SR + 1, 10 * SR))
        assert v <= MAX_VOICED, f"OOFFICE {tag}: voiced share {v:.3f}"
        rel = f"hvac/demand_ooffice_{tag}.wav"
        sf.write(str(out / rel), part, SR, subtype="PCM_16")
        entries.append({"id": f"demand-ooffice-{tag}", "kind": "hvac", "file": rel, "channels": len(DEMAND_CHANNELS),
                        "recorded": True, "duration": round(len(part) / SR, 3),
                        "source": f"{DEMAND['url']} OOFFICE {'/'.join(DEMAND_CHANNELS)} {t0:.0f}-{t1:.0f} s",
                        "licence": DEMAND["licence"], "attribution": DEMAND["attribution"],
                        "max_voiced_share": round(float(v), 4)})
    return entries


def esc50(root: Path, out: Path) -> tuple[list[dict], list[str]]:
    lic = {}
    for line in (root / "LICENSE").read_text(encoding="utf-8").splitlines():
        m = LICENCE_LINE.match(line.strip())
        if m:
            lic[m["stem"]] = m
    entries, skipped = [], []
    with open(root / "meta" / "esc50.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["category"] in ESC50_KINDS]
    for r in sorted(rows, key=lambda r: r["filename"]):
        kind = ESC50_KINDS[r["category"]]
        x, sr = sf.read(str(root / "audio" / r["filename"]), dtype="float32", always_2d=True)
        x = soxr.resample(x.mean(axis=1), sr, SR).astype(np.float32)
        x = trim(x, SR)
        stem = r["filename"].rsplit("-", 1)[0]  # 1-101336-A-30.wav -> 1-101336-A
        v = voiced_share(x, SR)
        if len(x) < int(0.1 * SR) or v > MAX_VOICED or stem not in lic:
            skipped.append(f"{r['filename']} ({r['category']}): "
                           + ("too short" if len(x) < int(0.1 * SR) else "no licence line" if stem not in lic
                              else f"voiced share {v:.3f}"))
            continue
        m = lic[stem]
        rel = f"{kind}/esc50_{stem}.wav"
        (out / kind).mkdir(parents=True, exist_ok=True)
        sf.write(str(out / rel), x, SR, subtype="PCM_16")
        entries.append({"id": f"esc50-{stem}", "kind": kind, "file": rel, "category": r["category"],
                        "duration": round(len(x) / SR, 3), "source": m["url"],
                        "licence": f"CC BY-NC 3.0 (ESC-50); original {m['lic']}",
                        "attribution": f"{m['by']} (Freesound), via ESC-50, K. J. Piczak (2015)",
                        "max_voiced_share": round(float(v), 4)})
    return entries, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand", type=Path, required=True)
    ap.add_argument("--esc50", type=Path, required=True, help="the ESC-50-master directory")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("SOUND_LIBRARY_DIR") or DEFAULT_ROOT) / "noise")
    a = ap.parse_args()
    clips = demand(a.demand, a.out)
    esc, skipped = esc50(a.esc50, a.out)
    clips += esc
    index = {"version": VERSION,
             "sources": [{"name": "DEMAND", "url": DEMAND["url"], "licence": DEMAND["licence"]},
                         {"name": "ESC-50", "url": "https://github.com/karolpiczak/ESC-50",
                          "licence": "CC BY-NC 3.0 (non-commercial); per-clip originals listed"}],
             "notes": "Built by data/sound_library/build_noise_library.py. Beds with recorded: true already hold "
                      "their room's acoustics and are mixed without a simulated impulse response.",
             "skipped": skipped, "clips": clips}
    (a.out / "index.yaml").write_text(yaml.safe_dump(index, sort_keys=False, allow_unicode=True), encoding="utf-8")
    kinds: dict[str, int] = {}
    for c in clips:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {a.out / 'index.yaml'}: {len(clips)} clips {kinds}; skipped {len(skipped)}")
    for s in skipped:
        print("  skipped", s)


if __name__ == "__main__":
    main()
