"""Decision models: something that gives a probability per option of a multiple-choice question about a request.

`AicoreDecisionModel` asks the `decision` role's service (LFM2.5-350M on llama.cpp, CPU only, so it answers while
a GPU phase holds the card) for one letter, constrained by a grammar to the option letters, and reads the letters'
probabilities from the top logprobs. `FakeDecisionModel` matches words of the request against each question's
wording and examples, so fake mode and the tests work with no service.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any

from ..backends.base import BackendError
from ..backends.transport import Transport
from .questions import QuestionSet, load_questions

LETTERS = string.ascii_uppercase
TIMEOUT_S = 30.0
SYSTEM = ("You read a request for synthetic meeting recordings and answer one multiple-choice question about it. "
          "Answer with the letter of the best option only. When the request does not say, pick the option that "
          "means no change or no.")
FALLBACK_P = 0.8  # the chosen letter's probability when the service returns no logprobs


def prompt(request: str, question: str, options: list[str]) -> str:
    lines = [f"Request: {request.strip()}", f"Question: {question.strip()}", "Options:"]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines.append("Answer:")
    return "\n".join(lines)


def letter_probs(raw: dict[str, Any], n: int) -> list[float]:
    """Probabilities of the first n letters from a chat completion with logprobs, renormalised over the letters."""
    try:
        choice = raw["choices"][0]
        text = (choice["message"].get("content") or "").strip()[:1].upper()
        tops = (choice.get("logprobs") or {}).get("content") or []
    except (KeyError, IndexError, TypeError):
        raise BackendError(f"malformed decision response: {str(raw)[:300]}") from None
    scores = [0.0] * n
    if tops:
        for t in tops[0].get("top_logprobs") or [{"token": tops[0].get("token"), "logprob": tops[0].get("logprob")}]:
            tok = (t.get("token") or "").strip().upper()
            if len(tok) == 1 and tok in LETTERS[:n] and t.get("logprob") is not None:
                scores[LETTERS.index(tok)] += math.exp(t["logprob"])
    total = sum(scores)
    if total <= 0:
        if text in LETTERS[:n]:
            rest = (1 - FALLBACK_P) / max(n - 1, 1)
            return [FALLBACK_P if LETTERS[i] == text else rest for i in range(n)]
        return [1 / n] * n
    return [v / total for v in scores]


class AicoreDecisionModel:
    def __init__(self, service: str, transport: Transport, *, client: Any = None):
        self.service = service
        self.transport = transport
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            import aicore

            self._client = aicore.client(self.service, timeout=TIMEOUT_S, max_retries=0)
        return self._client

    def choose(self, request: str, question: str, options: list[str]) -> list[float]:
        n = len(options)
        if not 2 <= n <= len(LETTERS):
            raise ValueError(f"{n} options; a question needs 2 to {len(LETTERS)}")
        body = {"messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt(request, question, options)}],
                "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": min(20, n + 5), "seed": 0}
        # llama.cpp's server reads a GBNF grammar from the body: only an option letter can be sampled
        extra = {"grammar": "root ::= " + " | ".join(f'"{LETTERS[i]}"' for i in range(n))}

        def call() -> Any:
            import aicore

            resp = self.client().chat.completions.with_raw_response.create(
                model=aicore.model_name(self.service), extra_body=extra, **body)
            return resp.http_response.json()

        return letter_probs(self.transport.call(self.service, {**body, **extra}, call), n)


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
