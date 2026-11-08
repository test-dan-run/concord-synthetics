"""Labelled requests for training and scoring a decision model: labels first, words second.

Each row starts as a `Target` drawn from `suggest/questions.yaml` with a seeded RNG: a few groups with a chosen
option, sometimes a room scope, and the global questions' answers (an exact number, something the settings cannot
express, harder or easier overall, what the run is for). A chat model then writes one request a non-technical
user might type that asks for exactly that and nothing else, so the labels are known before the words exist.

`expect` is what the engine should end up changing (the groups asked for directly, plus the groups a global
answer bundles in), in the same shape as `tests/data/suggest_eval.jsonl`. `laya_rows` turns a row into the
fine-tuning format of Laya's `laya.train` (`{state, questions, expected}`), asking the questions exactly as the
engine and `SystemOneDecisionModel` ask them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.base import Msg
from .engine import YES_NO
from .models import as_question
from .questions import QuestionSet, load_questions

DATA_VERSION = 3  # 2: adds the terse style; 3: the statement style, strength cues and numbers that imply a setting
GROUP_COUNTS = {0: 0.06, 1: 0.40, 2: 0.32, 3: 0.16, 4: 0.06}  # how many groups a request asks for directly
RARE_GROUPS = {"stages": 0.3, "backends": 0.5}  # relative weight: power-user asks, kept rare
P_SCOPE = 0.2  # a request with a scoped group names small or large rooms only
P_GLOBAL = {"exact_numbers": 0.2, "out_of_scope": 0.25, "overall_difficulty": 0.07, "purpose": 0.10}
# a first description of the dataset, a later chat message, a few words saying what the meetings are like
# ("echoey glass rooms"), or plain statements of fact ("everyone sits at the table"): people write the last two
# as often as a request to change
STYLES = {"description": 0.25, "chat_edit": 0.25, "terse": 0.25, "statement": 0.25}
# Groups whose options form a scale around `unchanged`: one step from it is a mild change, two or more a strong
# one, and the writer is told which words fit, so "a bit noisier" and "much noisier" read differently
SCALES = {"meeting_size", "length", "overlap", "interruptions", "backchannels", "dominance", "pace",
          "code_switching", "presentations", "hybrid", "wall_seats", "reverb", "noise", "mic_placement"}
MILD = "a mild change: words like a bit, slightly, a little, sometimes, often"
STRONG = "a strong change: words like very, way more, all the time, always, nearly everyone, as many as possible"
# A concrete number that says the option by itself, for requests with an exact number
NUMBERS = {("meeting_size", "smaller"): "3 or 4 people", ("meeting_size", "larger"): "10 to 15 people",
           ("meeting_size", "largest"): "20 people", ("length", "much_shorter"): "about 5 minutes",
           ("length", "much_longer"): "an hour or more", ("room_mix", "mostly_large"): "8 in 10 meetings big",
           ("room_mix", "mostly_small"): "8 in 10 meetings small", ("hybrid", "more"): "about half the people remote",
           ("hybrid", "most"): "remote people in 9 of 10 meetings"}
P_SINGLISH = 0.25
PERSONAS = ["an HR manager", "a speech researcher", "a product manager", "a data engineer", "a call-centre lead",
            "a student on an internship", "a startup founder", "a government officer", "a QA tester",
            "a linguist", "a team lead who is in a hurry", "a sales director"]
SCOPE_WORDS = {"small": "only the small meetings, not the big ones", "large": "only the big meetings, not the small ones"}
# Option labels the writer misreads in a trial batch, said plainly
HINTS = {("splits", "training"): "mostly training data", ("splits", "evaluation"): "mostly test data",
         ("splits", "balanced"): "an even split between training, validation and test data, not mostly one of them"}
# Everyday names of what each group is about, so the writer can be told what to leave out
ASPECTS = {"room_mix": "the mix of small and big meetings or rooms", "meeting_size": "how many people attend",
           "length": "how long meetings last", "overlap": "people talking over each other",
           "interruptions": "people interrupting or cutting in", "backchannels": "short replies like mm, ok, ya",
           "dominance": "who does most of the talking", "pace": "the pace or turn length",
           "code_switching": "mixing languages or Singlish", "presentations": "presentations",
           "hybrid": "remote or video-call participants", "wall_seats": "where people sit",
           "chair_position": "where the chair sits", "reverb": "echo or room sound",
           "noise": "background noise or how clean the audio is", "mic_placement": "where the mic is placed",
           "mic_array": "multichannel or single-mic audio", "device": "the recording device or audio quality",
           "race": "the race or ethnicity of speakers", "splits": "training, validation or test data",
           "stages": "skipping checks or pipeline steps", "backends": "fake models or running without the GPU"}
OUT_OF_SCOPE = ["say what the meetings should be about (pick a workplace topic)",
                "ask for a particular mood or emotion in the meetings",
                "ask for one named speaker to have a particular accent",
                "ask for some specific words or product names to come up"]
SYSTEM = (
    "You write messages that people type into a chat box of a tool that makes synthetic recordings of Singapore "
    "workplace meetings for training speech models. The people are not engineers: they type quickly and casually, "
    "in everyday words, never in settings or numbers unless told to. Messages vary: some are fragments, some have "
    "no greeting, some are lowercase. You are given exactly what the message must ask for. Ask for every item and "
    "for nothing else. Never copy the wording or the numbers of the items: say it the way that person would. "
    "Return JSON with one field, request.")
SCHEMA = {"title": "decision_request", "type": "object", "additionalProperties": False,
          "required": ["request"], "properties": {"request": {"type": "string", "minLength": 3, "maxLength": 400}}}


@dataclass
class Target:
    groups: dict[str, str]  # group id -> option id, asked for directly
    scope: str = "both"  # small | large | both: where the scoped groups' changes apply
    exact_numbers: bool = False
    out_of_scope: str | None = None  # what the request also asks for that no setting covers
    overall_difficulty: str = "unchanged"
    purpose: str = "unchanged"
    style: str = "description"
    singlish: bool = False
    persona: str = ""


@dataclass
class Row:
    id: str
    request: str
    expect: dict[str, str]  # group -> option the engine should apply (direct, then bundled)
    target: dict[str, Any] = field(default_factory=dict)
    version: int = DATA_VERSION


def _pick(rng: np.random.Generator, weights: dict) -> Any:
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def sample_target(rng: np.random.Generator, qs: QuestionSet) -> Target:
    n = _pick(rng, GROUP_COUNTS)
    ids = [g.id for g in qs.groups]
    w = np.array([RARE_GROUPS.get(i, 1.0) for i in ids])
    chosen = [ids[i] for i in rng.choice(len(ids), size=n, replace=False, p=w / w.sum())] if n else []
    groups = {}
    for gid in chosen:
        opts = [o.id for o in qs.group(gid).options if o.id != "unchanged"]
        groups[gid] = opts[int(rng.integers(len(opts)))]
    t = Target(groups=groups, style=_pick(rng, STYLES), singlish=bool(rng.random() < P_SINGLISH),
               persona=PERSONAS[int(rng.integers(len(PERSONAS)))])
    if any(qs.group(g).scope for g in groups) and rng.random() < P_SCOPE:
        t.scope = "small" if rng.random() < 0.5 else "large"
    t.exact_numbers = bool(groups) and rng.random() < P_GLOBAL["exact_numbers"]
    if rng.random() < P_GLOBAL["out_of_scope"]:
        t.out_of_scope = OUT_OF_SCOPE[int(rng.integers(len(OUT_OF_SCOPE)))]
    gq = {q.id: q for q in qs.global_}
    for name in ("overall_difficulty", "purpose"):
        if rng.random() < P_GLOBAL[name]:
            opts = [o.id for o in gq[name].options if o.id != "unchanged"]
            setattr(t, name, opts[int(rng.integers(len(opts)))])
    return t


def expected(t: Target, qs: QuestionSet) -> dict[str, str]:
    """What the engine should change: direct answers win over a global question's bundle."""
    out = dict(t.groups)
    gq = {q.id: q for q in qs.global_}
    for name in ("overall_difficulty", "purpose"):
        oid = getattr(t, name)
        if oid != "unchanged":
            for gid, bid in next(o for o in gq[name].options if o.id == oid).bundle.items():
                out.setdefault(gid, bid)
    return out


def brief(t: Target, qs: QuestionSet) -> str:
    """The user message for the writer: the items, the voice and the shape of the message."""
    items = []
    numbered = next((k for k in t.groups.items() if k in NUMBERS), None) if t.exact_numbers else None
    for gid, oid in t.groups.items():
        g = qs.group(gid)
        what = g.gate.strip().removeprefix("Does the request ").rstrip("?")
        line = f"- On this ({what}): {HINTS.get((gid, oid), g.option(oid).label)}"
        if gid in SCALES:
            ids = [o.id for o in g.options]
            line += f" ({MILD if abs(ids.index(oid) - ids.index('unchanged')) == 1 else STRONG})"
        if (gid, oid) == numbered:
            line += f"; say it with this number: {NUMBERS[numbered]}"
        items.append(line)
    gq = {q.id: q for q in qs.global_}
    for name in ("overall_difficulty", "purpose"):
        oid = getattr(t, name)
        if oid != "unchanged":
            label = next(o for o in gq[name].options if o.id == oid).label
            items.append(f"- Overall: {label}, without naming any particular aspect"
                         if name == "overall_difficulty" else f"- Say what the data is for: {label}")
    if t.scope != "both":
        items.append(f"- The changes above are for {SCOPE_WORDS[t.scope]}")
    if t.exact_numbers and numbered is None:
        items.append("- Give one exact number (minutes, a percentage, or a count) for one of the items above, "
                     "consistent with it; no other number anywhere in the message")
    if t.out_of_scope:
        items.append(f"- Also {t.out_of_scope}")
    if not items:
        items.append("- Nothing about the meetings at all: a greeting, a thank-you, or a question about the tool")
    shape = {"description": "a first description of the dataset they want, one to three sentences",
             "chat_edit": "a short follow-up chat message changing a plan they have already seen, a few words to "
                          "one sentence",
             "terse": "a few words (2 to 12) saying what the meetings are like, as a noun phrase or a plain "
                      "statement, not a question and not a request to change something",
             "statement": "one or two plain sentences stating how things are in the meetings (how people behave, "
                          "sit, join or sound), as facts, not as a question or a request"}[t.style]
    voice = ("in Singlish, the way a Singaporean types in a chat (lah, can, got, abit, leh, pls)" if t.singlish
             else "in plain English")
    # bundled groups stay off limits too: "harder overall" must not name an aspect, and a direct mention would
    # make that group's gate a yes the labels do not have
    named = set(t.groups)
    avoid = [ASPECTS[g.id] for g in qs.groups if g.id not in named and g.id in ASPECTS]
    if "meeting_size" not in named and "room_mix" not in named:
        avoid.append("the size of the rooms")
    return (f"Write {shape}, as {t.persona}, {voice}.\nThe message asks for:\n" + "\n".join(items)
            + "\nSay nothing at all about: " + "; ".join(avoid) + ".")


def write_request(llm: Any, t: Target, qs: QuestionSet, seed: int) -> str:
    res = llm.chat([Msg("system", SYSTEM), Msg("user", brief(t, qs))], json_schema=SCHEMA, temperature=0.9,
                   seed=seed, max_tokens=200, thinking=False)
    return " ".join(str(res.json()["request"]).split())


def generate(llm: Any, n: int, seed: int, *, qs: QuestionSet | None = None, workers: int = 16,
             done: Iterable[str] = (), log: Callable[[str], None] = print) -> Iterable[Row]:
    """Rows in id order; `done` ids are skipped, so a stopped run resumes. Row i's target and writing seed come
    from `SeedSequence([seed, i])`, so a row is the same whatever else is generated."""
    qs = qs or load_questions()
    skip = set(done)
    todo = []
    for i in range(n):
        rid = f"r{seed}-{i:05d}"
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        t = sample_target(rng, qs)
        if rid not in skip:
            todo.append((rid, t, int(rng.integers(2**31))))

    def one(job: tuple[str, Target, int]) -> Row:
        rid, t, s = job
        return Row(rid, write_request(llm, t, qs, s), expected(t, qs), asdict(t))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, row in enumerate(pool.map(one, todo), 1):
            if k % 100 == 0 or k == len(todo):
                log(f"wrote {k}/{len(todo)} requests")
            yield row


def read_rows(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def laya_rows(row: dict, qs: QuestionSet | None = None, *, unmentioned_choices: int = 3, seed: int = 0) -> dict:
    """One `laya.train` row: every global question and gate, the choices of the groups in `expect`, and a few
    choices of other groups whose answer is `unchanged`, so a gate that passes by mistake still changes nothing."""
    qs = qs or load_questions()
    t = row["target"]
    questions: dict[str, dict] = {}
    want: dict[str, Any] = {}
    gq = {q.id: q for q in qs.global_}
    for q in qs.global_:
        key = f"global:{q.id}"
        if q.options is None:
            questions[key] = as_question(q.question, YES_NO)
            want[key] = bool(t["exact_numbers"]) if q.id == "exact_numbers" else bool(t.get("out_of_scope"))
        else:
            labels = [o.label for o in q.options]
            questions[key] = as_question(q.question, labels)
            want[key] = f"o{[o.id for o in gq[q.id].options].index(t[q.id])}"
    for g in qs.groups:
        questions[f"gate:{g.id}"] = as_question(g.gate, YES_NO)
        want[f"gate:{g.id}"] = g.id in t["groups"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(row["id"].rsplit("-", 1)[1])]))
    others = [g.id for g in qs.groups if g.id not in row["expect"]]
    extra = [others[i] for i in rng.choice(len(others), size=min(unmentioned_choices, len(others)), replace=False)]
    for gid in [*t["groups"], *extra]:
        g = qs.group(gid)
        questions[f"choice:{gid}"] = as_question(g.question, [o.label for o in g.options])
        want[f"choice:{gid}"] = f"o{[o.id for o in g.options].index(t['groups'].get(gid, 'unchanged'))}"
    if any(qs.group(gid).scope for gid in t["groups"]):
        questions["scope"] = as_question(qs.scope.question, [o.label for o in qs.scope.options])
        want["scope"] = f"o{[o.id for o in qs.scope.options].index(t['scope'])}"
    return {"state": row["request"], "questions": questions, "expected": want}
