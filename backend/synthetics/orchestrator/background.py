"""A run started from the viewer: the same phase-by-phase `run` as the CLI, in a background process.

The viewer starts `python -m synthetics.orchestrator.background <run_dir>` in its own session, with stdout and
stderr appended to `process.log`, and writes `process.json` (pid, state). The process writes its final state there
when it ends. Stopping sends SIGINT, as Ctrl-C would: the GPU lease releases its services on the way out, and
`resume` (or Start again) continues from the status files, so stopping is safe at any point.

One run at a time: the GPU phases cannot share the card, and execution is sequential by design.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from ..artifacts.io import read_model, write_model
from ..artifacts.models import RunProcess
from ..artifacts.paths import RunPaths
from .status import now_iso

MODULE = "synthetics.orchestrator.background"


def _alive(pid: int) -> bool:
    """The pid is a live process running this module (not a zombie, and not a reused pid)."""
    proc = Path(f"/proc/{pid}")
    if proc.is_dir():
        try:
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, IndexError):
            return False
        return state not in ("Z", "X") and MODULE in cmdline
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@dataclass
class ProcessView:
    """process.json as the viewer shows it. `state` adds `idle` (never started here) and `lost` (the record says
    running but the process is gone, e.g. killed hard or the machine restarted)."""

    state: str
    record: RunProcess | None

    def as_dict(self) -> dict:
        rec = self.record.model_dump(mode="json") if self.record else {}
        return {**rec, "state": self.state}


def read_process(run_dir: Path) -> ProcessView:
    path = RunPaths(run_dir).process
    if not path.is_file():
        return ProcessView("idle", None)
    rec = read_model(path, RunProcess)
    if rec.state == "running" and not _alive(rec.pid):
        return ProcessView("lost", rec)
    return ProcessView(rec.state, rec)


def running(run_dir: Path) -> bool:
    return read_process(run_dir).state == "running"


def log_tail(run_dir: Path, lines: int = 80, max_bytes: int = 64_000) -> str:
    path = RunPaths(run_dir).process_log
    if not path.is_file():
        return ""
    with open(path, "rb") as fh:
        fh.seek(max(0, path.stat().st_size - max_bytes))
        text = fh.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def start(run_dir: Path, *, live: bool, argv: list[str] | None = None) -> RunProcess:
    """Start the run in the background and record it. `argv` replaces the command (tests)."""
    from .run import APP_ROOT

    rp = RunPaths(Path(run_dir))
    cmd = argv or [sys.executable, "-m", MODULE, str(rp.run_dir.resolve())]
    with open(rp.process_log, "a", encoding="utf-8") as log:
        log.write(f"\n==== started {now_iso()} from the viewer\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(APP_ROOT), stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    rec = RunProcess(pid=proc.pid, state="running", live=live, started_at=now_iso())
    write_model(rp.process, rec)
    # reap the child when it exits, so it never lingers as a zombie of the viewer
    threading.Thread(target=proc.wait, daemon=True, name=f"reap-{proc.pid}").start()
    return rec


def stop(run_dir: Path) -> bool:
    """SIGINT to the run process. False when nothing is running."""
    view = read_process(run_dir)
    if view.state != "running" or view.record is None:
        return False
    os.kill(view.record.pid, signal.SIGINT)
    return True


def _finish(rp: RunPaths, state: str, exit_code: int, error: str | None = None) -> None:
    rec = read_model(rp.process, RunProcess) if rp.process.is_file() else RunProcess(
        pid=os.getpid(), state="running", live=False, started_at=now_iso())
    write_model(rp.process, rec.model_copy(update={"state": state, "ended_at": now_iso(), "exit_code": exit_code,
                                                   "error": error}))


def main(run_dir: Path) -> int:
    """The CLI's `run -r <run_dir>`, recording how it ended in process.json."""
    import json
    import traceback

    from ..config.schema import StageId
    from .cli import _load_dotenv, _setup_logging
    from .run import load_run
    from .run_report import write_run_report
    from .scheduler import PhaseScheduler, summarise

    _load_dotenv()
    _setup_logging(False)
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # a terminate stops as cleanly as an interrupt
    rp = RunPaths(Path(run_dir))
    try:
        cfg, _ = load_run(rp.run_dir)
        rep = PhaseScheduler(rp.run_dir, cfg).run(until=StageId.LABELS)
        write_run_report(rp.run_dir, rep.phase_seconds)
        print(json.dumps(summarise(rp.run_dir, StageId.LABELS), indent=2), flush=True)
    except KeyboardInterrupt:
        print("stopped from the viewer; Start again (or `synthetics resume`) continues", flush=True)
        _finish(rp, "stopped", 130)
        return 130
    except Exception as exc:
        traceback.print_exc()
        _finish(rp, "failed", 1, f"{type(exc).__name__}: {exc}")
        return 1
    _finish(rp, "finished", 0)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
