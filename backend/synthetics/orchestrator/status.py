"""Per-meeting status.json: atomic updates under a file lock."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ..artifacts.io import read_model, write_model
from ..artifacts.models import MeetingStatus, StageState, StageStatus
from ..artifacts.paths import RunPaths


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StatusStore:
    def __init__(self, run_dir: Path):
        self.run = RunPaths(Path(run_dir))

    @contextmanager
    def _locked(self, meeting_id: str) -> Iterator[Path]:
        path = self.run.meeting(meeting_id).status
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_name(".status.lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield path
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def get(self, meeting_id: str) -> MeetingStatus:
        path = self.run.meeting(meeting_id).status
        if not path.is_file():
            return MeetingStatus(meeting_id=meeting_id)
        return read_model(path, MeetingStatus)

    def update(self, meeting_id: str, stage: int, **changes) -> StageStatus:
        with self._locked(meeting_id) as path:
            status = read_model(path, MeetingStatus) if path.is_file() else MeetingStatus(meeting_id=meeting_id)
            current = status.get(stage)
            new = current.model_copy(update=changes)
            status.stages[int(stage)] = StageStatus.model_validate(new.model_dump())
            status.stages = dict(sorted(status.stages.items()))
            write_model(path, status)
            return new

    def start(self, meeting_id: str, stage: int) -> StageStatus:
        cur = self.get(meeting_id).get(stage)
        return self.update(meeting_id, stage, state="running", attempts=cur.attempts + 1, started_at=now_iso(),
                           ended_at=None)

    def mark(self, meeting_id: str, stage: int, state: StageState, *, cache_key: str | None = None, detail: str = "",
             metrics: dict | None = None) -> StageStatus:
        changes: dict = {"state": state, "ended_at": now_iso(), "detail": detail}
        if cache_key is not None:
            changes["cache_key"] = cache_key
        if metrics is not None:
            changes["metrics"] = metrics
        return self.update(meeting_id, stage, **changes)

    def pending(self, stage: int, meeting_ids: list[str] | None = None) -> list[str]:
        """Meetings whose `stage` has not passed and was not dropped or skipped."""
        ids = meeting_ids if meeting_ids is not None else self.run.meeting_ids()
        return [m for m in ids if self.get(m).get(stage).state not in ("pass", "dropped", "skipped")]

    def dropped(self, meeting_id: str) -> bool:
        return any(s.state == "dropped" for s in self.get(meeting_id).stages.values())
