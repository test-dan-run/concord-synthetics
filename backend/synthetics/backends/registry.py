"""`Backends`: lazily built role clients for one stage, from `backends.yaml` (role -> service)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config.schema import BackendsConfig, Role
from .base import BackendError
from .transport import Transport

CHAT_ROLES = {Role.GENERATOR, Role.JUDGE_PRIMARY, Role.JUDGE_TIEBREAK}
ENHANCER_ROLES = {Role.ENHANCER, Role.DEREVERB}
ASR_ROLES = {Role.ASR_PRIMARY, Role.ASR_SINGLISH}
SV_ROLES = {Role.SV_WAVLM, Role.SV_REDIMNET}


class Backends:
    def __init__(self, cfg: BackendsConfig, roles: Iterable[Role] | None = None, *, base_dir: Path | None = None,
                 overrides: dict[Role, Any] | None = None, available: Iterable[Role] | None = None):
        self.cfg = cfg
        self.allowed = set(roles) if roles is not None else set(Role)
        self.available = set(available) if available is not None else None  # roles whose service is leased now
        recordings = cfg.recordings if cfg.recordings.is_absolute() or base_dir is None else base_dir / cfg.recordings
        self.transport = Transport(mode="aicore" if cfg.mode == "fake" else cfg.mode, recordings=recordings)
        self._clients: dict[Role, Any] = dict(overrides or {})

    def for_roles(self, roles: Iterable[Role]) -> Backends:
        """A view limited to `roles`, sharing built clients."""
        view = Backends(self.cfg, roles, available=self.available)
        view.transport = self.transport
        view._clients = self._clients
        return view

    def with_available(self, roles: Iterable[Role] | None) -> Backends:
        """A view whose leased roles are `roles` (None = every role), sharing built clients."""
        view = Backends(self.cfg, self.allowed, available=roles)
        view.transport = self.transport
        view._clients = self._clients
        return view

    def has(self, role: Role) -> bool:
        """Declared by the stage and leased in the current phase."""
        return role in self.allowed and (self.available is None or role in self.available)

    def service(self, role: Role) -> str:
        try:
            return self.cfg.roles[role]
        except KeyError:
            raise BackendError(f"no service configured for role {role.value!r} in backends.yaml") from None

    def get(self, role: Role) -> Any:
        role = Role(role)
        if role not in self.allowed:
            raise BackendError(f"role {role.value!r} was not declared by this stage")
        if role not in self._clients:
            self._clients[role] = self._build(role)
        return self._clients[role]

    def _build(self, role: Role) -> Any:
        service = self.service(role)
        if self.cfg.mode == "fake":
            from . import fakes, fakes_audio

            if role in CHAT_ROLES:
                return fakes.FakeChatLLM(service)
            if role in ENHANCER_ROLES:
                return fakes.FakeEnhancer()
            if role in (Role.QUALITY, Role.REF_QUALITY):
                return fakes.FakeQualityScorer()
            if role == Role.TTS:
                return fakes_audio.FakeVoiceCloner()
            if role in ASR_ROLES:
                return fakes_audio.FakeTranscriber(service)
            if role in SV_ROLES:
                return fakes_audio.FakeSpeakerEmbedder(role.value)
            if role == Role.ALIGNER:
                return fakes_audio.FakeAligner()
            raise BackendError(f"no fake for role {role.value!r} yet")
        from . import aicore_impl

        if role in CHAT_ROLES:
            return aicore_impl.AicoreChatLLM(service, self.transport)
        if role in ENHANCER_ROLES:
            return aicore_impl.AicoreEnhancer(service, self.transport)
        if role in (Role.QUALITY, Role.REF_QUALITY):
            return aicore_impl.AicoreQualityScorer(service, self.transport)
        if role == Role.TTS:
            return aicore_impl.AicoreVoiceCloner(service, self.transport)
        if role in ASR_ROLES:
            return aicore_impl.AicoreTranscriber(service, self.transport)
        if role in SV_ROLES:
            return aicore_impl.AicoreSpeakerEmbedder(service, self.transport, name=role.value)
        if role == Role.ALIGNER:
            return aicore_impl.AicoreAligner(service, self.transport)
        raise BackendError(f"role {role.value!r} has no aicore client")


def check_endpoints(cfg: BackendsConfig, roles: Iterable[Role]) -> list[str]:
    """Roles whose service is missing from the platform endpoint map (empty = all present)."""
    if cfg.mode in ("fake", "replay"):
        return []
    import aicore

    missing = []
    for role in roles:
        name = cfg.roles.get(role)
        try:
            aicore.endpoint(name) if name else None
        except Exception:  # noqa: BLE001 - unknown service or no endpoint map
            missing.append(f"{role.value} -> {name}")
        if not name:
            missing.append(f"{role.value} -> (not configured)")
    return missing
