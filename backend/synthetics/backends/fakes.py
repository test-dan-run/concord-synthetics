"""Deterministic fake backends for unit tests and `mode: fake` runs. No GPU, no network.

`FakeChatLLM` answers by the JSON schema's `title`, so it can stand in for the generator and both judges:
- `meeting_brief`: a brief and a stance per speaker (speaker ids read from the prompt's roster lines)
- `script_chunk`: one turn per planned line (`<i> | <speaker> | <interaction>`) in the prompt
- `script_judgement`: fixed scores (`judge_overall`), persona consistency for every roster speaker
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from .base import Audio, ChatResult, Msg

PLAN_LINE = re.compile(r"^(\d+) \| (S\d+) \| ([a-z_]+)\s*$", re.M)
ROSTER_LINE = re.compile(r"^- (S\d+)\b", re.M)

_NORMAL = [
    "Ok so the vendor said they can only deliver by end of month (lah), so we need a plan B.",
    "(um) I think we should check with finance first, then we decide.",
    "Can or not (meh)? Last time also they say like that, then delay again.",
    "Actually the HDB side already approved, we just waiting for the LTA reply.",
    "We wan- we want to push the go-live to next Monday, is that ok with everybody?",
    "Wah this one quite shiok, if it works we save two weeks.",
    "I also not sure (leh), maybe Priya can confirm the numbers.",
    "Then the KPI for Q3 how? We still need to report to the GM.",
    "Paiseh (ah), I missed the email, can you share again?",
    "(uh) the SIT environment down since yesterday, so testing a bit stuck.",
]
_BACKCHANNEL = ["(mm)", "Ya.", "Ok (lah).", "Right.", "(mm) ya.", "Can."]
_PRESENT = [
    "So on this slide you can see the timeline, phase one ends in March and phase two starts in April.",
    "Next slide, (um) these are the three risks we identified, the biggest one is the vendor capacity.",
    "As you can see from the chart, the numbers went up about twenty percent compared to last quarter.",
]


def _h(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class FakeChatLLM:
    def __init__(self, service: str = "fake-llm", *, judge_overall: float = 4.2, judge_overall_by_call: list[float] | None = None):
        self.service = service
        self.judge_overall = judge_overall
        self.judge_overall_by_call = list(judge_overall_by_call or [])
        self.calls: list[dict] = []

    def chat(self, messages: list[Msg], *, json_schema: dict | None = None, temperature: float, seed: int,
             max_tokens: int, thinking: bool | None = None) -> ChatResult:
        import json

        title = (json_schema or {}).get("title", "")
        prompt = "\n".join(m.content for m in messages)
        self.calls.append({"title": title, "seed": seed, "thinking": thinking, "messages": messages})
        if title == "meeting_brief":
            speakers = sorted(set(ROSTER_LINE.findall(prompt)))
            out = {"brief": "The team reviews progress, open risks and next steps before the deadline.",
                   "stances": [{"speaker_id": s, "stance": f"{s} wants a clear owner for each open risk."}
                               for s in speakers]}
        elif title == "script_chunk":
            turns = []
            for i, spk, kind in PLAN_LINE.findall(prompt):
                k = _h(spk, i, seed)
                if kind == "backchannel":
                    text = _BACKCHANNEL[k % len(_BACKCHANNEL)]
                elif kind == "presentation":
                    text = _PRESENT[k % len(_PRESENT)]
                else:
                    text = _NORMAL[k % len(_NORMAL)]
                turns.append({"i": int(i), "speaker_id": spk, "text": text})
            # an interrupted turn ends with a cut-off hyphen
            kinds = {int(i): kind for i, _, kind in PLAN_LINE.findall(prompt)}
            for t in turns:
                if kinds.get(t["i"] + 1) == "interruption":
                    t["text"] = t["text"].rstrip(".?!") .rsplit(" ", 1)[0] + "-"
            out = {"turns": turns}
        elif title == "script_judgement":
            speakers = sorted(set(re.findall(r"\b(S\d\d)\b", prompt)))
            overall = self.judge_overall_by_call.pop(0) if self.judge_overall_by_call else self.judge_overall
            out = {"rationale": "Turns stay on topic; particles and code-switching sound natural.",
                   "coherence": overall, "singlish": overall, "realism": overall,
                   "persona_consistency": [{"speaker_id": s, "score": 4.0} for s in speakers],
                   "overall": overall, "feedback": "" if overall >= 3.0 else "Make the quieter speakers sound less formal."}
        else:
            out = {"text": "ok"}
        return ChatResult(text=json.dumps(out, ensure_ascii=False), finish_reason="stop", model=self.service)


class FakeEnhancer:
    """Identity with a tiny deterministic gain, so cleaned output differs from input."""

    def __init__(self, gain: float = 0.98):
        self.gain = gain

    def clean(self, audio: Audio) -> Audio:
        return Audio(audio.samples * self.gain, audio.sr)


class FakeQualityScorer:
    def score(self, clips: Sequence[Audio]) -> np.ndarray:
        out = []
        for c in clips:
            k = _h(round(float(np.abs(c.samples).mean()), 6), len(c.samples))
            out.append(3.0 + (k % 1000) / 600)  # 3.0 .. 4.67
        return np.asarray(out, dtype=np.float32)
