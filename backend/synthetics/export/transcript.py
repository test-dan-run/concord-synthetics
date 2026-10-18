"""Word-level transcript from per-utterance forced alignment (stage 8).

Each timeline segment is aligned on its own dry utterance (the same audio its speaker's track holds in that
window), with the turn's `text_norm` as the transcript, so a request never carries more than one utterance.
Word times are offset by the segment start. For a segment cut off by an interruption, only the words heard before
`truncated_at` are kept (a word cut in the middle stays, ending at the cut), and `text` / `text_norm` are cut to
match, with the cut-off hyphen of the convention.
"""

from __future__ import annotations

import difflib
import re

from ..artifacts.labels import Segment, TranscriptUtterance, Word
from ..artifacts.models import Turn
from ..backends.base import WordTiming
from ..text.singlish import text_norm

TRAILING_PUNCT = re.compile(r"[\s.,!?;:…\"”’]+$")


CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def align_language(text_norm: str, default: str, cjk: str) -> str:
    """The aligner's LANGUAGE for one utterance: `cjk` when it holds Chinese characters, else `default`. The
    aligner takes en, zh and other codes, but not Japanese, which our convention never produces."""
    return cjk if CJK.search(text_norm) else default


def norm_word(w: str) -> str:
    """An aligner word as a text_norm word: the aligner drops punctuation but keeps apostrophes, and text_norm strips
    them at word edges (it keeps them inside, as in "don't")."""
    return w.lower().strip("-'’")


def matches(expected: list[str], aligned: list[str]) -> dict[int, int]:
    """Aligned word index -> expected word index, for the words matched in order."""
    sm = difflib.SequenceMatcher(a=expected, b=aligned, autojunk=False)
    return {blk.b + k: blk.a + k for blk in sm.get_matching_blocks() for k in range(blk.size)}


def coverage(expected: list[str], aligned: list[str]) -> float:
    """Share of `expected` words matched, in order, by the aligner's words."""
    return len(matches(expected, aligned)) / len(expected) if expected else 1.0


def cut_text(text: str, n_words: int) -> str:
    """The shortest prefix of `text` (whitespace tokens) holding `n_words` text_norm words, ending in a hyphen."""
    if n_words <= 0:
        return ""
    kept, count = [], 0
    for tok in text.split():
        kept.append(tok)
        count += len(text_norm(tok).split())
        if count >= n_words:
            break
    out = TRAILING_PUNCT.sub("", " ".join(kept))
    return out if out.endswith("-") else out + "-"


def utterance(seg: Segment, turn: Turn, timings: list[WordTiming]) -> TranscriptUtterance:
    expected = turn.text_norm.split()
    aligned = [norm_word(t.word) for t in timings]
    matched = matches(expected, aligned)
    cov = len(matched) / len(expected) if expected else 1.0
    words = [Word(word=t.word, start=round(seg.start + t.start, 4), end=round(seg.start + t.end, 4),
                  score=round(t.score, 4)) for t in timings]
    text, norm = turn.text, turn.text_norm
    truncated = seg.truncated_at is not None
    if truncated:
        cut = seg.start + seg.truncated_at
        words = [w for w in words if w.start < cut]
        if words and words[-1].end > cut:
            words[-1] = words[-1].model_copy(update={"end": round(cut, 4)})
        # expected words up to the last one the kept aligned words reach
        reached = [matched[i] for i in range(len(words)) if i in matched]
        heard = max(reached) + 1 if reached else min(len(words), len(expected))
        text = cut_text(turn.text, heard)
        norm = " ".join(expected[:heard])
    return TranscriptUtterance(turn_id=seg.turn_id, speaker_id=seg.speaker_id, start=seg.start, end=seg.end,
                               text=text, text_norm=norm, truncated=truncated, words=words,
                               align_coverage=round(cov, 4))
