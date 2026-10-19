"""Scoring helpers for stage 5 and the bake-off: ASR error after Singlish normalisation, and SV aggregates.

WER compares `text_norm` words (particles count; fillers are dropped from both sides when asked, since ASR models
routinely omit "um" and "uh"). Short utterances use CER over the same words, as the design doc asks for
backchannels. ASR spellings of the same word ("yeah" for the scripts' "ya") are mapped first, on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..text.numbers import spell_numbers
from ..text.singlish import Lexicon, text_norm

# Scoring only: these are the same word to a listener, and the lexicon (which also shapes scripts and TTS input)
# stays as it is.
ASR_SPELLINGS = {"yeah": "ya", "yah": "ya", "okay": "ok"}


def asr_words(text: str, lex: Lexicon, *, ignore_fillers: bool, normalised: bool = False) -> list[str]:
    """Words for scoring: `text_norm` of the text (unless already normalised; digits spelled out first, since
    ASR writes "20" where scripts say "twenty"), ASR spellings mapped, fillers optionally dropped."""
    words = [ASR_SPELLINGS.get(w, w) for w in (text if normalised else text_norm(spell_numbers(text), lex)).split()]
    if ignore_fillers:
        words = [w for w in words if lex.filler(w) is None]
    return words


@dataclass(frozen=True)
class AsrScore:
    wer: float | None
    cer: float | None
    error: float  # the one that decides: CER for short utterances, else WER
    short: bool


def score_asr(ref_words: list[str], hypothesis: str, lex: Lexicon, *, short_words: int, ignore_fillers: bool,
              ignore: frozenset[str] = frozenset()) -> AsrScore:
    """`ref_words` must be non-empty (callers skip the ASR check for turns with no words). Hypothesis words in
    `ignore` are dropped (e.g. "haha" cloned for a [laugh] tag, which the reference does not contain)."""
    import jiwer

    if not ref_words:
        raise ValueError("no reference words to score against")
    hyp = [w for w in asr_words(hypothesis, lex, ignore_fillers=ignore_fillers) if w not in ignore]
    ref_s, hyp_s = " ".join(ref_words), " ".join(hyp)
    short = len(ref_words) <= short_words
    if not hyp_s:
        return AsrScore(wer=None if short else 1.0, cer=1.0 if short else None, error=1.0, short=short)
    if short:
        cer = float(jiwer.cer(ref_s, hyp_s))
        return AsrScore(wer=None, cer=round(cer, 4), error=cer, short=True)
    wer = float(jiwer.wer(ref_s, hyp_s))
    return AsrScore(wer=round(wer, 4), cer=None, error=wer, short=False)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def consistency(embs: np.ndarray) -> float:
    """Mean pairwise cosine of L2-normalised rows (1.0 for fewer than two rows)."""
    n = len(embs)
    if n < 2:
        return 1.0
    sims = embs @ embs.T
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def cross_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine over every pair of rows, one from each set (L2-normalised rows): the single-pair scale the SV
    thresholds were fitted on."""
    return float((a @ b.T).mean())


def centroid(embs: np.ndarray) -> np.ndarray:
    return unit(np.asarray(embs, dtype=np.float64).mean(axis=0))
