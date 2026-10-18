"""Record / replay / retries around every backend request, as in Concord.

A request is described by a JSON-able dict (`request`) plus optional arrays. The key is
sha256(service, canonical request, array hashes), and recordings live at `<dir>/<service>/<key>.json`
holding the raw response, so replayed tests still exercise every parser.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .base import BackendError

RETRY_DELAYS = (2, 4, 8, 16)
Mode = Literal["aicore", "record", "replay"]


class ReplayMiss(BackendError):
    pass


def array_digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def request_key(service: str, request: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> str:
    body = json.dumps({"service": service, "request": request,
                       "arrays": {k: array_digest(v) for k, v in sorted((arrays or {}).items())}},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:24]


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (502, 503, 504):
        return True
    return type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout",
                                  "RemoteProtocolError", "InternalServerError", "ConnectTimeout")


@dataclass
class Transport:
    mode: Mode = "aicore"
    recordings: Path | None = None
    sleep: Callable[[float], None] = time.sleep

    def call(self, service: str, request: dict[str, Any], fn: Callable[[], Any],
             arrays: dict[str, np.ndarray] | None = None) -> Any:
        """Run `fn()` (returns the raw JSON-able response) through replay / record / retries."""
        key = request_key(service, request, arrays)
        path = (self.recordings / service / f"{key}.json") if self.recordings else None
        if self.mode == "replay":
            if path is None or not path.is_file():
                raise ReplayMiss(f"no recorded response for {service} ({key}) under {self.recordings}")
            return json.loads(path.read_text(encoding="utf-8"))["raw"]
        last: Exception | None = None
        for attempt, delay in enumerate((0, *RETRY_DELAYS)):
            if delay:
                self.sleep(delay)
            try:
                raw = fn()
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                if _retryable(exc) and attempt < len(RETRY_DELAYS):
                    continue
                raise BackendError(f"{service}: {type(exc).__name__}: {exc}") from exc
            if self.mode == "record" and path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"service": service, "request": request, "raw": raw},
                                           ensure_ascii=False) + "\n", encoding="utf-8")
            return raw
        raise BackendError(f"{service}: gave up after retries: {last}")
