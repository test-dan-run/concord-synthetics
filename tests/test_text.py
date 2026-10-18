"""Singlish convention: canonical text, text_norm, TTS input, NSC conversion, numbers. Stages 2, 5, 8 rely on it."""

from __future__ import annotations

import pytest

from synthetics.text import Lexicon, analyse, canonical_text, nsc_to_ours, text_norm, tts_input
from synthetics.text.numbers import cardinal, ordinal, spell_numbers, year
from synthetics.text.singlish import ends_with_cutoff


@pytest.fixture(scope="module")
def lex() -> Lexicon:
    return Lexicon.load()


@pytest.mark.parametrize("raw, expected", [
    # particles: canonical spelling in parentheses, variants mapped
    ("ok la, we go first", "ok (lah), we go first"),
    ("ok lah we go first", "ok (lah) we go first"),
    ("ok (la), we go first", "ok (lah), we go first"),
    ("can or not meh?", "can or not (meh)?"),
    ("like that lo", "like that (lor)"),
    ("so expensive sia", "so expensive (sia)"),
    ("he go already liao", "he go already (liao)"),
    # ambiguous words only become particles at the end of a clause, after a word
    ("We go ah?", "We go (ah)?"),
    ("Ah, I see.", "Ah, I see."),
    ("ah ma come already", "ah ma come already"),
    # fillers: closed set, erm -> um, hmm -> mm
    ("um I think so", "(um) I think so"),
    ("erm, maybe", "(um), maybe"),
    ("(erm) maybe", "(um) maybe"),
    ("Hmm, ok.", "(mm), ok."),
    ("He said er the vendor", "He said er the vendor"),
    # non-speech tags
    ("*laughs* true", "[laugh] true"),
    ("(laughter) ok lor", "[laugh] ok (lor)"),
    ("[coughs] sorry", "[cough] sorry"),
    # loanwords, acronyms, numbers
    ("very shiok, paisey ah", "very shiok, paiseh (ah)"),
    ("Kaypo one", "Kaypoh one"),
    ("we da bao lunch", "we dabao lunch"),
    ("the H.D.B. flat", "the HDB flat"),
    ("e.g. this one", "e.g. this one"),
    ("in 2026", "in twenty twenty six"),
    ("S$1,500 for 3 pax", "one thousand five hundred dollars for three pax"),
    ("20% more", "twenty percent more"),
    # false starts and cut-offs survive
    ("we wan- we want", "we wan- we want"),
    ("I think we should-", "I think we should-"),
    # Mandarin in characters, untouched
    ("then 他会教 lah", "then 他会教 (lah)"),
    ("  extra   spaces , here ", "extra spaces, here"),
    # stage directions are not speech
    ("(whisper) eh, after this go makan?", "eh, after this go makan?"),
    ("ok (whispering) can (lah)", "ok can (lah)"),
])
def test_canonical_text(lex, raw, expected):
    assert canonical_text(raw, lex) == expected


def test_canonical_text_is_idempotent(lex):
    for raw in ["ok la, we go first", "um I think so (lah)", "[laugh] true lor", "in 2026 we wan- we want", "可以 lah"]:
        once = canonical_text(raw, lex)
        assert canonical_text(once, lex) == once


@pytest.mark.parametrize("text, norm", [
    ("ok (lah), we go first", "ok lah we go first"),
    ("(um) I think so.", "um i think so"),
    ("[laugh] true", "true"),
    ("we wan- we want", "we wan we want"),
    ("I think we should-", "i think we should"),
    ("the HDB flat", "the hdb flat"),
    ("Don't worry (lor)!", "don't worry lor"),
    ("then 他会教 (lah)", "then 他 会 教 lah"),
    ("[cough] [laugh]", ""),
    ("Can or not (meh)?", "can or not meh"),
])
def test_text_norm(lex, text, norm):
    assert text_norm(text, lex) == norm


def test_text_norm_keeps_particles_and_fillers_for_wer(lex):
    assert text_norm("(uh) ok (lor)", lex).split() == ["uh", "ok", "lor"]


def test_analyse_finds_disfluencies_nonspeech_and_unknown_markup(lex):
    m = analyse("(um) we wan- we want to to go [laugh] (lah) (sneeze) [music] I think-", lex)
    assert ("filler", "um") in m.disfluencies
    assert ("false_start", "wan-") in m.disfluencies
    assert ("repetition", "to") in m.disfluencies
    assert ("cutoff", "think-") in m.disfluencies
    assert m.nonspeech == ["laugh"]
    assert m.unknown == ["(sneeze)", "[music]"]


def test_number_words_are_not_repetitions(lex):
    assert analyse("twenty twenty six", lex).disfluencies == []


def test_ends_with_cutoff():
    assert ends_with_cutoff("I think we should-")
    assert ends_with_cutoff('so we "need-"')
    assert not ends_with_cutoff("a well-known plan.")
    assert not ends_with_cutoff("ok -")


def test_tts_input(lex):
    segs = tts_input("[laugh] ok (lor), we go (ah)", lex, hints=False)
    assert [(s.kind, s.value) for s in segs] == [("nonspeech", "laugh"), ("speech", "ok lor, we go ah")]
    hinted = tts_input("very shiok (lor)", lex)
    assert hinted[0].value == "very shee-ok law"
    assert tts_input("[cough]", lex)[0].kind == "nonspeech"


@pytest.mark.parametrize("nsc, ours", [
    ("so are we going to (um) wait for instructions [eh]", "so are we going to (um) wait for instructions eh"),
    ("ok [lah] can", "ok (lah) can"),
    ("(ppl) true [lor]", "[laugh] true (lor)"),
    ("(ppc) sorry (ppb) ok (ppo)", "[cough] sorry [breath] ok [noise]"),
    ("we go <malay>makan</malay> first", "we go makan first"),
    ("(mmhmm) (erm) right", "(mm) (um) right"),
    ("[seh] so fast", "(sia) so fast"),
    ("very paisey [ah]", "very paiseh (ah)"),
])
def test_nsc_to_ours(lex, nsc, ours):
    assert nsc_to_ours(nsc, lex) == ours


def test_nsc_names_are_refused(lex):
    with pytest.raises(ValueError):
        nsc_to_ours("we went to #Bishan# yesterday", lex)


def test_numbers():
    assert cardinal(0) == "zero"
    assert cardinal(42) == "forty two"
    assert cardinal(1_234_567) == "one million two hundred thirty four thousand five hundred sixty seven"
    assert year(2026) == "twenty twenty six"
    assert year(2005) == "two thousand and five"
    assert year(1990) == "nineteen ninety"
    assert year(1905) == "nineteen oh five"
    assert ordinal(21) == "twenty first" and ordinal(12) == "twelfth" and ordinal(30) == "thirtieth"
    assert spell_numbers("meet at 3pm on the 21st") == "meet at three p m on the twenty first"
    assert spell_numbers("version 2.5") == "version two point five"
    assert spell_numbers("call 0612") == "call zero six one two"
    assert spell_numbers("$5k") == "five thousand dollars"
    assert spell_numbers("no digits here") == "no digits here"


def test_lexicon_version_is_recorded(lex):
    assert lex.version == "lex-1"
    assert lex.particles == {"lah", "leh", "lor", "meh", "hor", "ah", "sia", "mah", "liao", "wor"}
    assert lex.fillers == {"uh", "um", "mm"}
