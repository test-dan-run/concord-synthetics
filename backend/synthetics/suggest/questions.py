"""The decision questions (`questions.yaml`) as typed models.

The file is the contract between the decision model and the run builder: the model only ever picks among the
options listed here, and every option's effects are plain edits of builder settings, so nothing the model says can
reach the config unless a person wrote it into this file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

QUESTIONS = Path(__file__).with_name("questions.yaml")
Op = Literal["shift", "scale", "range", "p", "scale_p", "weights", "set"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Effect(_M):
    path: str
    op: Op
    by: float | None = None
    lo: float | None = None
    hi: float | None = None
    value: Any = None


class Option(_M):
    id: str
    label: str
    effects: list[Effect] = []
    overlay: list[Effect] = []  # settings outside the form: written to the YAML overlay
    bundle: dict[str, str] = {}  # group id -> option id
    confirm: bool = False  # always asked, never applied as a suggestion


class Group(_M):
    id: str
    gate: str
    question: str
    options: list[Option]
    scope: bool = False
    examples: list[str] = []
    note: str | None = None

    def option(self, oid: str) -> Option:
        return next(o for o in self.options if o.id == oid)


class GlobalQuestion(_M):
    id: str
    question: str
    options: list[Option] | None = None  # None: a yes/no question
    on_yes: str | None = None
    note: str | None = None


class Policy(_M):
    gate_min: float
    apply_min: float
    offer_min: float
    confirm_always: list[str] = []


class ScopeOption(_M):
    id: Literal["both", "small", "large"]
    label: str


class Scope(_M):
    question: str
    options: list[ScopeOption]
    default: Literal["both", "small", "large"] = "both"


class QuestionSet(_M):
    version: int
    policy: Policy
    scope: Scope
    bounds: dict[str, tuple[float, float]]
    global_: list[GlobalQuestion] = Field(alias="global")
    groups: list[Group]

    def group(self, gid: str) -> Group:
        return next(g for g in self.groups if g.id == gid)


@lru_cache(maxsize=4)
def load_questions(path: Path = QUESTIONS) -> QuestionSet:
    return QuestionSet.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
