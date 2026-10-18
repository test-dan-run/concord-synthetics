"""One helper per path in the storage layout, so no code builds paths by hand.

    runs/<run_id>/
      config.yaml, speaker_splits.json, manifest.jsonl
      meetings/<meeting_id>/
        meeting.json, script.json, prompts/<speaker_id>.txt, script_validation.json, status.json, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def config(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def speaker_splits(self) -> Path:
        return self.run_dir / "speaker_splits.json"

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.jsonl"

    @property
    def meetings_dir(self) -> Path:
        return self.run_dir / "meetings"

    @property
    def judge_sets(self) -> Path:
        return self.run_dir / "judge_sets"

    def meeting(self, meeting_id: str) -> MeetingPaths:
        return MeetingPaths(self.run_dir, meeting_id)

    def meeting_ids(self) -> list[str]:
        if not self.meetings_dir.is_dir():
            return []
        return sorted(p.name for p in self.meetings_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


@dataclass(frozen=True)
class MeetingPaths:
    run_dir: Path
    meeting_id: str

    @property
    def run(self) -> RunPaths:
        return RunPaths(self.run_dir)

    @property
    def dir(self) -> Path:
        return self.run_dir / "meetings" / self.meeting_id

    @property
    def meeting(self) -> Path:
        return self.dir / "meeting.json"

    @property
    def script(self) -> Path:
        return self.dir / "script.json"

    @property
    def prompts_dir(self) -> Path:
        return self.dir / "prompts"

    def prompt(self, speaker_id: str) -> Path:
        return self.prompts_dir / f"{speaker_id}.txt"

    @property
    def script_validation(self) -> Path:
        return self.dir / "script_validation.json"

    @property
    def utterances_dir(self) -> Path:
        return self.dir / "utterances"

    def utterance(self, turn_id: str) -> Path:
        return self.utterances_dir / f"{turn_id}.wav"

    @property
    def synthesis(self) -> Path:
        return self.dir / "synthesis.json"

    @property
    def voice_validation(self) -> Path:
        return self.dir / "voice_validation.json"

    @property
    def timeline(self) -> Path:
        return self.dir / "timeline.json"

    @property
    def tracks_dir(self) -> Path:
        return self.dir / "tracks"

    def track(self, speaker_id: str) -> Path:
        return self.tracks_dir / f"{speaker_id}.wav"

    @property
    def mixture(self) -> Path:
        return self.dir / "mixture.wav"

    @property
    def mixture_mc(self) -> Path:
        return self.dir / "mixture_mc.wav"

    @property
    def render(self) -> Path:
        return self.dir / "render.json"

    @property
    def labels_dir(self) -> Path:
        return self.dir / "labels"

    @property
    def rttm(self) -> Path:
        return self.labels_dir / "meeting.rttm"

    @property
    def transcript(self) -> Path:
        return self.labels_dir / "transcript.json"

    @property
    def manifest_entry(self) -> Path:
        return self.labels_dir / "manifest_entry.json"

    @property
    def status(self) -> Path:
        return self.dir / "status.json"

    def artifact(self, key: str) -> Path:
        """Artifact key (as stages declare them in `reads` / `writes`) -> path. Run-level keys included."""
        run_level = {"speaker_splits": self.run.speaker_splits, "config": self.run.config}
        if key in run_level:
            return run_level[key]
        resolved = getattr(self, key, None)
        if not isinstance(resolved, Path):
            raise KeyError(f"unknown artifact key {key!r}")
        return resolved
