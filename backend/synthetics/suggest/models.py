"""Decision models: something that gives a probability per option of a multiple-choice question about a request.

`SystemOneDecisionModel` asks the `decision` role's service, a decision model on the CPU (so it answers while a GPU
phase holds the card), through the platform's `decision.systemone` API: the request is the state, and every
question goes as a yes/no (`noul`) or a `choice` among the option labels. `choose_many` sends a batch of questions
in one call, which is how the engine asks them. `FakeDecisionModel` matches words of the request against each
question's wording and examples, so fake mode and the tests work with no service.
"""

from __future__ import annotations

import re
from typing import Any

from ..backends.base import BackendError
from ..backends.transport import Transport
from .questions import QuestionSet, load_questions

TIMEOUT_S = 30.0
YES_NO = ["Yes", "No"]
# Yes/no answers read better with each side described (the Julia-1 card: 80.5% against 65% with literal true/false).
GATE_CRITERIA = {"true": "yes, the request says this", "false": "no, the request does not say this"}

Item = tuple[str, str, list[str]]  # (key, question, option labels)


def as_question(question: str, options: list[str]) -> dict[str, Any]:
    """One decision.systemone question: a yes/no for ["Yes", "No"], else a choice keyed o0, o1, ..."""
    if options == YES_NO:
        return {"type": "noul", "instructions": question.strip(), "criteria": GATE_CRITERIA}
    if not 2 <= len(options) <= 20:
        raise ValueError(f"{len(options)} options; a question needs 2 to 20")
    return {"type": "choice", "instructions": question.strip(),
            "criteria": {f"o{i}": label for i, label in enumerate(options)}}


def answer_probs(answer: dict[str, Any], options: list[str]) -> list[float]:
    """Probabilities in option order from one decision.systemone answer."""
    try:
        if options == YES_NO:
            yes = min(max(float(answer["noul"]), 0.0), 1.0)
            return [yes, 1.0 - yes]
        probs = answer["probabilities"]
        scores = [max(float(probs.get(f"o{i}", 0.0)), 0.0) for i in range(len(options))]
    except (KeyError, TypeError, ValueError):
        raise BackendError(f"malformed decision answer: {str(answer)[:300]}") from None
    total = sum(scores)
    return [s / total for s in scores] if total > 0 else [1 / len(options)] * len(options)


class SystemOneDecisionModel:
    def __init__(self, service: str, transport: Transport, *, client: Any = None):
        self.service = service
        self.transport = transport
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            import aicore

            self._client = aicore.client(self.service, timeout=TIMEOUT_S)
        return self._client

    def choose_many(self, request: str, items: list[Item]) -> dict[str, list[float]]:
        if not items:
            return {}
        questions = {key: as_question(question, options) for key, question, options in items}
        body = {"state": request.strip(), "questions": questions}
        answers = self.transport.call(self.service, body,
                                      lambda: self.client().decide(body["state"], questions))
        missing = [key for key in questions if key not in answers]
        if missing:
            raise BackendError(f"decision service gave no answer for {missing}")
        return {key: answer_probs(answers[key], options) for key, _, options in items}

    def choose(self, request: str, question: str, options: list[str]) -> list[float]:
        return self.choose_many(request, [("q", question, options)])["q"]


_STOP = set(["a", "an", "and", "the", "or", "of", "to", "for", "in", "on", "at", "by", "with", "as", "is", "are", "be", "does", "do", "it", "its", "this", "that", "these", "those", "request", "ask", "asks", "asking", "say", "says", "said", "mention", "mentions", "such", "about", "how", "should", "change", "changes", "more", "less", "fewer", "many", "lot", "lots", "much", "bit", "than", "same", "people", "meeting", "meetings", "speaker", "speakers", "someone", "person", "table", "room", "rooms", "run", "data", "whether", "which", "what", "when", "where", "who", "would", "like", "want", "need"])


def _raw(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]+", text.lower())}


_STOP |= {"mostly", "only", "just", "even", "some", "all", "every", "everyone", "really", "very", "kind", "lah"}
# a few everyday words for the option labels' terms, so "big boardrooms" reads as large rooms
SYNONYMS = {"big": "large", "boardr": "large", "huge": "large", "tiny": "small", "dry": "fake", "models": "gpu",
            "zoom": "remote", "teams": "remote", "dialli": "remote", "aircon": "noise", "noisy": "noise",
            "echoey": "echo", "crossta": "overlap"}


def _words(text: str) -> set[str]:
    ws = {w[:6] for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}
    return ws | {SYNONYMS[w] for w in ws if w in SYNONYMS}


def _clauses(text: str) -> list[str]:
    return [c for c in re.split(r"[,.;!?\n]| and | but | with ", text) if c.strip()]


UP = _raw("more lot lots many heavy heavily frequent usually most mostly noisier noisy louder bigger larger "
          "longer faster plenty harder echoey echo crowded busy always")
DOWN = _raw("fewer less little rarely never none no without quieter cleaner clean smaller shorter slower easier "
            "dry drier polite calm quick")


class FakeDecisionModel:
    """Word overlap: a gate says yes when the request shares a content word with the gate or its examples; a
    choice prefers the option whose label shares the most words with the request, nudged by up and down words.
    Deterministic and rough, for fake mode and tests."""

    def __init__(self, qs: QuestionSet | None = None):
        self.qs = qs or load_questions()
        self.cues: dict[str, set[str]] = {}
        for g in self.qs.groups:
            self.cues[g.gate.strip()] = _words(g.gate + " " + " ".join(g.examples))
            self.cues[g.question.strip()] = self.cues[g.gate.strip()]  # a choice reads the clauses its gate matched
        for gq in self.qs.global_:
            self.cues[gq.question.strip()] = _words(gq.question)
        self.globals = {gq.question.strip() for gq in self.qs.global_}

    def choose(self, request: str, question: str, options: list[str]) -> list[float]:
        req = _words(request)
        if options == ["Yes", "No"]:
            if question.startswith("Does the request give an exact number"):
                yes = bool(re.search(r"\b\d+(\.\d+)?\s*(db|speakers|people|minutes|min|%|percent)\b", request.lower()))
            elif question.startswith("Does the request ask for something these settings cannot express"):
                yes = bool(req & _words("topic topics emotion emotions accent accents words term terms"))
            else:
                yes = bool(req & self.cues.get(question.strip(), set()))
            return [0.9, 0.1] if yes else [0.1, 0.9]
        whole = _raw(request)
        cues = self.cues.get(question.strip())
        if cues:  # only the parts of the request about this question set its direction
            about = [c for c in _clauses(request) if _words(c) & cues]
            if about:
                request = " ".join(about)
                req = _words(request)
        raw = _raw(request)
        if not raw & (UP | DOWN):  # the clause has no direction of its own: the request's
            raw = whole
        up, down = bool(raw & UP), bool(raw & DOWN)
        # direction words only steer a scale (options going both ways), never a global question
        scale = any(_raw(o) & UP for o in options) and any(_raw(o) & DOWN for o in options)
        if question.strip() in self.globals or not scale:
            up = down = False
        scores = []
        for o in options:
            w, wr = _words(o), _raw(o)
            s = 2.0 * len(req & w)
            if o.lower() in ("unchanged", "not said", "not asked", "none") and not (up or down):
                s += 0.5
            o_up, o_down = bool(wr & UP), bool(wr & DOWN)
            if up != down:  # one clear direction in the request: options going that way win, the others lose
                s += 1.0 if (o_up if up else o_down) else 0.0
                s -= 1.0 if (o_down if up else o_up) else 0.0
            scores.append(s)
        best = max(range(len(options)), key=lambda i: (scores[i], -i))
        if scores[best] <= 0.5:  # nothing in the request speaks to it: unchanged if there is one
            best = next((i for i, o in enumerate(options) if o.lower() in ("unchanged", "not said", "not asked")),
                        best)
            p = 0.9
        else:
            runner = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
            p = 0.9 if scores[best] - runner >= 1.0 else 0.55
        rest = (1 - p) / max(len(options) - 1, 1)
        return [p if i == best else rest for i in range(len(options))]
