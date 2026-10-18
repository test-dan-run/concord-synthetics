"""Where the NSC source data and the derived reference datasets live (§4.3). Datasets stay in
/home/user/code/datasets/, next to the NSC data; nothing is copied into the app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NSC_DIR = Path("/home/user/code/datasets/NSC_PART3")
DEFAULT_REFS_DIR = Path("/home/user/code/datasets/nsc-part3-refs")
POOL_CSV = "data/nsc_part3_reference_speakers.csv"


@dataclass(frozen=True)
class NscPaths:
    nsc_dir: Path  # holds SDM/ and SPEAKER_INFO_PART3.XLSX
    refs_dir: Path  # nsc-part3-refs/
    pool_csv: Path

    @property
    def raw(self) -> Path:
        return self.refs_dir / "raw"

    @property
    def clean(self) -> Path:
        return self.refs_dir / "clean"

    def raw_speaker(self, speaker: str) -> Path:
        return self.raw / speaker

    def clean_speaker(self, speaker: str) -> Path:
        return self.clean / speaker

    @property
    def clean_index(self) -> Path:
        return self.clean / "index.jsonl"


def nsc_paths(nsc_dir: Path | None = None, refs_dir: Path | None = None, pool_csv: Path | None = None) -> NscPaths:
    from ..orchestrator.run import app_path

    return NscPaths(
        nsc_dir=Path(nsc_dir or os.environ.get("NSC_DIR") or DEFAULT_NSC_DIR),
        refs_dir=Path(refs_dir or os.environ.get("NSC_REFS_DIR") or DEFAULT_REFS_DIR),
        pool_csv=Path(pool_csv) if pool_csv else app_path(POOL_CSV))
