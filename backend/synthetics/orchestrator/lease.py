"""GPU lease: `platform/tools/stack.py up <services>` on enter, `down <services>` on exit, as Concord does.

stack.py validates, checks the VRAM budget, starts GPU services one at a time, waits for health and stops
services that do not fit (unless no_evict). In fake / replay mode the lease does nothing.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

log = logging.getLogger("synthetics.lease")


class LeaseError(RuntimeError):
    pass


def platform_env(platform_dir: Path) -> dict[str, str]:
    """Our environment plus the platform's .env (as `just` would load it, without overriding what is set)."""
    env = dict(os.environ)
    dotenv = platform_dir / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            env.setdefault(key.strip(), value.strip().strip("'\""))
    if os.environ.get("PLATFORM_VENV"):
        env["UV_PROJECT_ENVIRONMENT"] = os.environ["PLATFORM_VENV"]
    env.pop("VIRTUAL_ENV", None)
    return env


class GpuLease:
    def __init__(self, services: Sequence[str], *, platform_dir: Path, no_evict: bool = False, timeout_s: int = 1800,
                 enabled: bool = True, keep: bool = False, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.services = sorted(set(services))
        self.platform_dir = Path(platform_dir)
        self.no_evict = no_evict
        self.timeout_s = timeout_s
        self.enabled = enabled and bool(self.services)
        self.keep = keep  # leave the services running on exit (debugging, or the next phase reuses them)
        self.runner = runner

    def _stack(self, *args: str) -> subprocess.CompletedProcess:
        cmd = ["uv", "run", "--quiet", "--project", str(self.platform_dir), "python", "tools/stack.py", *args]
        log.info("$ %s", " ".join(cmd[cmd.index("tools/stack.py"):]))
        try:
            return self.runner(cmd, cwd=str(self.platform_dir), env=platform_env(self.platform_dir),
                               capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise LeaseError(f"stack.py {args[0]} timed out after {self.timeout_s}s") from exc
        except FileNotFoundError as exc:
            raise LeaseError(f"cannot run the platform tool: {exc}") from exc

    def __enter__(self) -> GpuLease:
        if not self.enabled:
            return self
        args = ["up", *self.services] + (["--no-evict"] if self.no_evict else [])
        res = self._stack(*args)
        if res.returncode != 0:
            tail = [ln for ln in (res.stdout + res.stderr).splitlines() if ln.strip()][-6:]
            errs = [ln.removeprefix("error: ") for ln in tail if ln.startswith("error:")] or tail
            raise LeaseError(f"stack.py up {' '.join(self.services)} refused: {'; '.join(errs)}")
        log.info("lease acquired: %s", ", ".join(self.services))
        return self

    def __exit__(self, *exc: object) -> None:
        if not self.enabled or self.keep:
            return
        res = self._stack("down", *self.services)
        if res.returncode != 0:
            log.warning("stack.py down %s exited %d: %s", " ".join(self.services), res.returncode, res.stderr[-300:])
        else:
            log.info("lease released: %s", ", ".join(self.services))


def check_vram(services: Sequence[str], *, platform_dir: Path,
               runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> tuple[bool, str]:
    """`just vram <services>` (tools/vram_check.py): False when the set is over util_limit."""
    cmd = ["uv", "run", "--quiet", "--project", str(platform_dir), "python", "tools/vram_check.py", *services]
    res = runner(cmd, cwd=str(platform_dir), env=platform_env(platform_dir), capture_output=True, text=True)
    return res.returncode == 0, (res.stdout + res.stderr).strip()
