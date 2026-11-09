"""Stage 3 programmatic checks, run before any LLM judge: schema, speakers, turn shares, overlap /
interruption / backchannel rates, cut-offs, turn lengths, markup and presentation consistency."""

from __future__ import annotations

import re
from collections import Counter

from ..artifacts.meeting import MeetingSpec
from ..artifacts.models import RuleCheck, Script
from ..config.schema import RunConfig
from ..text.singlish import Lexicon, analyse, ends_with_cutoff

MAIN = ("normal", "interruption", "overlap_start", "presentation")


def _rate_ok(name: str, value: float, lo: float, hi: float, detail: str = "") -> RuleCheck:
    return RuleCheck(name=name, passed=lo <= value <= hi, value=round(value, 4), limit=f"[{lo:.3f}, {hi:.3f}]",
                     detail=detail)


def check_script(script: Script, m: MeetingSpec, cfg: RunConfig, lex: Lexicon) -> list[RuleCheck]:
    th = cfg.thresholds.script_rules
    tt = cfg.turn_taking
    turns = script.turns
    ids = {p.speaker_id for p in m.people}
    checks: list[RuleCheck] = []

    unknown_ids = sorted({t.speaker_id for t in turns} - ids)
    checks.append(RuleCheck(name="persona ids", passed=not unknown_ids, value=len(unknown_ids), limit=0,
                            detail=f"unknown speakers: {', '.join(unknown_ids)}" if unknown_ids else ""))
    silent = sorted(ids - {t.speaker_id for t in turns})
    checks.append(RuleCheck(name="speaker count", passed=not silent, value=len(ids) - len(silent), limit=len(ids),
                            detail=f"never speak: {', '.join(silent)}" if silent else ""))
    empty = [t.turn_id for t in turns if not t.text_norm.strip() and not t.nonspeech]
    checks.append(RuleCheck(name="non-empty turns", passed=not empty, value=len(empty), limit=0,
                            detail=", ".join(empty[:5])))

    main = [t for t in turns if t.interaction in MAIN]
    n_main = max(len(main), 1)
    counts = Counter(t.speaker_id for t in main)
    gaps = {s: abs(counts.get(s, 0) / n_main - m.dynamics.turn_shares[s]) for s in ids}
    worst = max(gaps, key=lambda s: gaps[s]) if gaps else None
    checks.append(RuleCheck(name="turn shares", passed=(not gaps or gaps[worst] <= th.share_tolerance),
                            value=round(gaps[worst], 4) if worst else 0.0, limit=th.share_tolerance,
                            detail=f"largest gap {worst}" if worst else ""))

    # interaction rates over the free discussion; the presentation segment (monologue + Q&A) has its own shape
    seg = m.dynamics.presentation
    in_seg = set(range(seg.start_turn, seg.end_turn)) if seg else set()
    free = [t for i, t in enumerate(main) if i not in in_seg and t.interaction != "presentation"]
    changes = [b for a, b in zip(free, free[1:], strict=False) if a.speaker_id != b.speaker_id]  # turn changes
    transitions = max(len(changes), 1)
    inter = sum(t.interaction == "interruption" for t in changes)
    checks.append(_rate_ok("interruption rate", inter / transitions, 0.0, tt.interruption_rate.hi + 0.03))
    ovl = sum(t.interaction == "overlap_start" for t in changes)
    expected = tt.overlap_share * (1 - tt.interruption_rate.mean())
    checks.append(_rate_ok("overlap-start rate", ovl / transitions, max(0.0, expected - 0.1), expected + 0.1))
    bc = sum(t.interaction == "backchannel" for t in turns)
    per_min = bc / max(m.dynamics.duration_min, 1e-6)
    checks.append(_rate_ok("backchannels per minute", per_min, max(0.0, tt.backchannels_per_min.lo - 0.5),
                           tt.backchannels_per_min.hi + 0.5))

    # the turn before an interruption is cut off
    missing_cut = []
    for i, t in enumerate(turns):
        if t.interaction != "interruption":
            continue
        j = i - 1
        while j >= 0 and turns[j].interaction in ("backchannel", "side"):
            j -= 1
        lexical = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", turns[j].text) if j >= 0 else ""
        if j >= 0 and not ends_with_cutoff(turns[j].text) and re.search(r"[^\W_]", lexical):
            missing_cut.append(turns[j].turn_id)
    checks.append(RuleCheck(name="interrupted turns cut off", passed=not missing_cut, value=len(missing_cut), limit=0,
                            detail=", ".join(missing_cut[:5])))

    long_bc = [t.turn_id for t in turns if t.interaction == "backchannel"
               and len(t.text_norm.split()) > th.max_words_backchannel]
    checks.append(RuleCheck(name="backchannel length", passed=len(long_bc) <= max(1, bc // 10), value=len(long_bc),
                            limit=max(1, bc // 10), detail=", ".join(long_bc[:5])))
    short = [t.turn_id for t in main if len(t.text_norm.split()) < th.min_words_normal and t.interaction == "presentation"]
    checks.append(RuleCheck(name="presentation turns are substantial", passed=not short, value=len(short), limit=0,
                            detail=", ".join(short[:5])))

    unknown_markup = Counter()
    for t in turns:
        for u in analyse(t.text, lex).unknown:
            unknown_markup[u] += 1
    n_unknown = sum(unknown_markup.values())
    checks.append(RuleCheck(name="markup in convention", passed=n_unknown <= th.max_unknown_tags, value=n_unknown,
                            limit=th.max_unknown_tags, detail=", ".join(f"{k} x{v}" for k, v in unknown_markup.most_common(5))))

    bad_pres = [t.turn_id for t in turns if t.interaction == "presentation"
                and (seg is None or t.speaker_id not in seg.presenters)]
    checks.append(RuleCheck(name="presentation by presenters", passed=not bad_pres, value=len(bad_pres), limit=0,
                            detail=", ".join(bad_pres[:5])))

    # checks beside the judge, each on only when its limit is set
    if th.min_words_per_planned_min is not None:
        words = sum(len(t.text_norm.split()) for t in turns)
        per_min = words / max(m.dynamics.duration_min, 1e-6)
        checks.append(RuleCheck(name="words per planned minute", passed=per_min >= th.min_words_per_planned_min,
                                value=round(per_min, 1), limit=th.min_words_per_planned_min,
                                detail=f"{words} words for {m.dynamics.duration_min:.1f} planned minutes"))
    if th.max_repeated_phrase_share is not None:
        share, examples = repeated_phrase_share(script)
        checks.append(RuleCheck(name="repeated phrasing", passed=share <= th.max_repeated_phrase_share,
                                value=round(share, 4), limit=th.max_repeated_phrase_share,
                                detail="; ".join(f"'{p}'" for p in examples[:3])))
    if th.max_self_introductions is not None:
        named = self_introductions(script, m)
        checks.append(RuleCheck(name="self-introductions", passed=len(named) <= th.max_self_introductions,
                                value=len(named), limit=th.max_self_introductions, detail=", ".join(named[:5])))
    return checks


PHRASE_WORDS = 4


def repeated_phrase_share(script: Script) -> tuple[float, list[str]]:
    """Share of turns (of at least PHRASE_WORDS words) that share a phrase of PHRASE_WORDS words with another turn,
    and the most repeated phrases. A writer that falls back on stock lines repeats them across a meeting."""
    grams = []
    for t in script.turns:
        w = t.text_norm.split()
        grams.append({" ".join(w[i:i + PHRASE_WORDS]) for i in range(len(w) - PHRASE_WORDS + 1)})
    turns_with = Counter(g for gs in grams for g in gs)
    long = [gs for gs in grams if gs]
    repeated = sum(any(turns_with[g] > 1 for g in gs) for gs in long)
    top = [g for g, n in turns_with.most_common(3) if n > 1]
    return (repeated / len(long) if long else 0.0), top


def self_introductions(script: Script, m: MeetingSpec) -> list[str]:
    """Turns in which the speaker names themself ("Karthik here", "this is Karthik", "I'm Karthik"): people in a
    meeting know who is talking, so a script that announces speakers is written for a reader."""
    names = {p.speaker_id: p.persona.display_name.split()[0].lower() for p in m.people if p.persona.display_name}
    out = []
    for t in script.turns:
        name = names.get(t.speaker_id)
        if not name:
            continue
        n = re.escape(name)
        if re.search(rf"\b{n}\s+here\b|\bthis is {n}\b|\bi'?m {n}\b|\bi am {n}\b|\bmy name is {n}\b", t.text.lower()):
            out.append(t.turn_id)
    return out
