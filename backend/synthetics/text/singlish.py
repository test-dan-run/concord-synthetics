"""The Singlish transcription convention: canonical `text`, `text_norm`, TTS input and NSC conversion.

`text`      cased, punctuated; particles and fillers in parentheses: "ok (lah), we go first"; non-speech as
            square-bracket tags: "[laugh] true"; numbers as words; acronyms uppercase without dots.
`text_norm` lowercase, no punctuation or markup; particle and filler words kept (they count in WER);
            non-speech tags dropped; cut-off hyphens dropped.
TTS input   `text` without parentheses, with non-speech tags split out as sound-library clips.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .numbers import ONES, TENS, spell_numbers

DEFAULT_LEXICON = Path(__file__).resolve().parents[1] / "config" / "defaults" / "singlish_lexicon.yaml"

# Stage directions a generator sometimes adds despite the prompt; they are not speech, so they are dropped.
STAGE_DIRECTIONS = frozenset({"whisper", "whispers", "whispering", "whispered", "softly", "quietly", "aside", "sotto voce",
                              "under breath", "under their breath", "muttering", "mutters", "to self", "to herself",
                              "to himself", "pause", "pauses", "long pause", "silence", "sarcastically", "jokingly",
                              "nodding", "nods", "shrugs", "smiling", "smiles", "sighing"})
NUMBER_WORDS = frozenset(ONES + TENS[2:] + ["hundred", "thousand", "million", "billion"])
_PAREN = re.compile(r"\(([^()\[\]]{1,20})\)")
_BRACKET = re.compile(r"\[([^\[\]()]{1,20})\]")
_STAR_ACTION = re.compile(r"\*([a-zA-Z ]{2,20})\*")
_DOTTED_ACRONYM = re.compile(r"\b(?:[A-Z]\.){2,}")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*-?|-", re.U)


@dataclass(frozen=True)
class Lexicon:
    version: str
    particles: frozenset[str]
    particle_variants: dict[str, str]
    clause_final_only: frozenset[str]
    fillers: frozenset[str]
    filler_variants: dict[str, str]
    nonspeech: frozenset[str]
    nonspeech_variants: dict[str, str]
    loanwords: dict[str, str]
    tts_hints: dict[str, str]
    nsc: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_LEXICON) -> Lexicon:
        return _load(str(path))

    def particle(self, token: str) -> str | None:
        t = token.lower()
        if t in self.particles:
            return t
        return self.particle_variants.get(t)

    def filler(self, token: str) -> str | None:
        t = token.lower()
        if t in self.fillers:
            return t
        return self.filler_variants.get(t)

    def nonspeech_tag(self, token: str) -> str | None:
        t = token.lower().strip()
        if t in self.nonspeech:
            return t
        return self.nonspeech_variants.get(t)


@lru_cache(maxsize=8)
def _load(path: str) -> Lexicon:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Lexicon(
        version=str(d["version"]), particles=frozenset(d["particles"]), particle_variants=dict(d["particle_variants"]),
        clause_final_only=frozenset(d.get("clause_final_only", [])), fillers=frozenset(d["fillers"]),
        filler_variants={str(k): v for k, v in d["filler_variants"].items()}, nonspeech=frozenset(d["nonspeech"]),
        nonspeech_variants=dict(d["nonspeech_variants"]), loanwords={str(k): str(v) for k, v in d["loanwords"].items()},
        tts_hints=dict(d.get("tts_hints", {})), nsc={k: {str(a): str(b) for a, b in v.items()} for k, v in d.get("nsc", {}).items()})


# ---------------------------------------------------------------- canonical text


def _fix_markup(text: str, lex: Lexicon) -> str:
    """Parenthesised and bracketed tokens to their canonical forms; unknown markup is left for the validator."""

    def paren(m: re.Match) -> str:
        inner = m.group(1).strip()
        if inner.lower() in STAGE_DIRECTIONS:
            return ""  # never spoken, never transcribed
        if (p := lex.particle(inner)) is not None:
            return f"({p})"
        if (f := lex.filler(inner)) is not None:
            return f"({f})"
        if (tag := lex.nonspeech_tag(inner)) is not None:
            return f"[{tag}]"
        return m.group(0)

    def bracket(m: re.Match) -> str:
        inner = m.group(1).strip()
        if (tag := lex.nonspeech_tag(inner)) is not None:
            return f"[{tag}]"
        return m.group(0)

    def star(m: re.Match) -> str:
        tag = lex.nonspeech_tag(m.group(1))
        return f"[{tag}]" if tag else m.group(0)

    text = _STAR_ACTION.sub(star, text)
    text = _PAREN.sub(paren, text)
    return _BRACKET.sub(bracket, text)


def _wrap_bare_tokens(text: str, lex: Lexicon) -> str:
    """Bare particles and fillers -> parenthesised canonical form. Ambiguous words only at clause end."""
    out: list[str] = []
    # split keeping markup groups intact
    parts = re.split(r"(\([^()]*\)|\[[^\[\]]*\])", text)
    for part in parts:
        if part.startswith(("(", "[")):
            out.append(part)
            continue
        tokens = re.split(r"(\s+)", part)
        for i, tok in enumerate(tokens):
            m = re.fullmatch(r"([\"'“‘]?)([A-Za-z]+)([,.!?;:\"'”’]*)", tok)
            if not m:
                continue
            lead, word, trail = m.groups()
            low = word.lower()
            rest = "".join(tokens[i + 1:]).strip()
            clause_end = bool(trail and trail[0] in ",.!?;:") or rest == "" or rest[0] in ",.!?;:"
            after_word = any(re.search(r"[A-Za-z]", t) for t in tokens[:i]) and not re.search(r"[.!?]\s*$", "".join(tokens[:i]))
            p = lex.particle(low)
            if p is not None and (low not in lex.clause_final_only or (clause_end and after_word)):
                tokens[i] = f"{lead}({p}){trail}"
                continue
            f = lex.filler(low)
            if f is not None and (low != "er" or clause_end):
                tokens[i] = f"{lead}({f}){trail}"
        out.append("".join(tokens))
    return "".join(out)


def _loanwords(text: str, lex: Lexicon) -> str:
    for variant in sorted(lex.loanwords, key=len, reverse=True):
        canon = lex.loanwords[variant]
        if variant == canon:
            continue
        text = re.sub(rf"(?i)\b{re.escape(variant)}\b", lambda m, c=canon: _match_case(m.group(0), c), text)
    return text


def _match_case(src: str, canon: str) -> str:
    return canon.capitalize() if src[:1].isupper() else canon


def canonical_text(raw: str, lex: Lexicon) -> str:
    """Bring generated text into the convention (idempotent)."""
    t = unicodedata.normalize("NFC", raw).replace("’", "'").replace("‘", "'")
    t = t.replace("—", ", ").replace("–", "-").replace("…", "...")
    t = _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", "").upper(), t)
    t = spell_numbers(t)
    t = _fix_markup(t, lex)
    t = _wrap_bare_tokens(t, lex)
    t = _loanwords(t, lex)
    t = _SPACES.sub(" ", t).strip()
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    return t


# ---------------------------------------------------------------- text_norm


_CJK = re.compile(r"([㐀-䶿一-鿿豈-﫿])")


def text_norm(text: str, lex: Lexicon | None = None) -> str:
    """`text` -> `text_norm`: lowercase, markup and punctuation removed, particles and fillers kept as words,
    non-speech dropped, cut-off / false-start hyphens dropped, Chinese characters split one per token."""
    t = unicodedata.normalize("NFC", text)
    t = _BRACKET.sub(" ", t)  # non-speech (and any other bracket tag) is excluded from WER
    t = re.sub(r"[()]", " ", t)
    t = t.lower().replace("’", "'")
    t = _CJK.sub(r" \1 ", t)
    words = []
    for m in _WORD.finditer(t):
        w = m.group(0).strip("-'")
        if w:
            words.append(w)
    return " ".join(words)


# ---------------------------------------------------------------- disfluencies, non-speech, markup checks


@dataclass
class Markup:
    disfluencies: list[tuple[str, str]]  # (type, token)
    nonspeech: list[str]
    unknown: list[str]  # markup outside the convention


def analyse(text: str, lex: Lexicon) -> Markup:
    dis: list[tuple[str, str]] = []
    unknown: list[str] = []
    for m in _PAREN.finditer(text):
        inner = m.group(1)
        if inner in lex.fillers:
            dis.append(("filler", inner))
        elif inner not in lex.particles:
            unknown.append(m.group(0))
    nonspeech = []
    for m in _BRACKET.finditer(text):
        if m.group(1) in lex.nonspeech:
            nonspeech.append(m.group(1))
        else:
            unknown.append(m.group(0))
    plain = _BRACKET.sub(" ", _PAREN.sub(" ", text))
    toks = [w for w in re.findall(r"[^\W_]+(?:'[^\W_]+)*-?", plain, re.U)]
    for i, tok in enumerate(toks):
        if tok.endswith("-"):
            dis.append(("cutoff" if i == len(toks) - 1 else "false_start", tok))
        elif i > 0 and tok.lower() == toks[i - 1].lower() and tok.isalpha() and tok.lower() not in NUMBER_WORDS:
            dis.append(("repetition", tok))
    return Markup(dis, nonspeech, unknown)


def ends_with_cutoff(text: str) -> bool:
    return bool(re.search(r"[^\W_]-\s*[\"”]?\s*$", text, re.U))


# ---------------------------------------------------------------- TTS input


@dataclass(frozen=True)
class TtsSegment:
    kind: str  # "speech" | "nonspeech"
    value: str  # text to synthesise, or the non-speech tag (sound-library clip)


def tts_input(text: str, lex: Lexicon, *, hints: bool = True) -> list[TtsSegment]:
    """`text` -> segments: parentheses removed (particles / fillers spoken), non-speech tags as clips."""
    segments: list[TtsSegment] = []
    pos = 0
    for m in _BRACKET.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            segments.append(TtsSegment("speech", _speakable(before, lex, hints)))
        tag = m.group(1)
        if tag in lex.nonspeech:
            segments.append(TtsSegment("nonspeech", tag))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        segments.append(TtsSegment("speech", _speakable(tail, lex, hints)))
    return [s for s in segments if s.kind == "nonspeech" or re.search(r"[^\W_]", s.value, re.U)]


def _speakable(s: str, lex: Lexicon, hints: bool) -> str:
    s = _PAREN.sub(lambda m: m.group(1), s)
    if hints:
        for word, hint in lex.tts_hints.items():
            s = re.sub(rf"(?i)\b{re.escape(word)}\b", hint, s)
    s = re.sub(r"\s+([,.!?;:])", r"\1", _SPACES.sub(" ", s)).strip()
    return s.lstrip(",;: ")


# ---------------------------------------------------------------- NSC notation -> ours


_NSC_LANG_TAG = re.compile(r"</?[a-zA-Z]+>")
_NSC_NAME = re.compile(r"#[^#]*#")


def nsc_to_ours(text: str, lex: Lexicon) -> str:
    """NSC Part 3 transcript -> our notation: [lah] -> (lah), (uh) stays, (ppl) -> [laugh], <malay>..</malay>
    tags dropped. Text with #name# redactions must not be used (PDPA); this raises on it."""
    if _NSC_NAME.search(text):
        raise ValueError("NSC text contains a #name# redaction; such clips are never used as references")
    nsc = lex.nsc

    def bracket(m: re.Match) -> str:
        inner = m.group(1).strip().lower()
        if inner in nsc.get("bracket_particles", {}):
            return f"({nsc['bracket_particles'][inner]})"
        if inner in nsc.get("bracket_words", {}):
            return nsc["bracket_words"][inner]
        return inner

    def paren(m: re.Match) -> str:
        inner = m.group(1).strip().lower()
        if inner in nsc.get("paren_codes", {}):
            return f"[{nsc['paren_codes'][inner]}]"
        if inner in nsc.get("paren_fillers", {}):
            return f"({nsc['paren_fillers'][inner]})"
        return inner

    t = _NSC_LANG_TAG.sub(" ", text)
    t = re.sub(r"\(([^()]{1,20})\)", paren, t)  # parentheses first: bracket particles become parentheses
    t = re.sub(r"\[([^\[\]]{1,20})\]", lambda m: m.group(0) if m.group(1) in lex.nonspeech else bracket(m), t)
    t = t.replace("_", " ")
    t = _loanwords(t, lex)
    return _SPACES.sub(" ", t).strip()
