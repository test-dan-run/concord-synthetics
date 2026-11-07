"""Key terms: canonical spelling and TTS respelling through the run lexicon, the prompt line, and cache keys that
only move when a run has key terms."""

from __future__ import annotations

from conftest import make_cfg
from synthetics.config.schema import KeyTerm
from synthetics.stages.prompts import key_terms_line
from synthetics.stages.s1_sample import SampleStage
from synthetics.stages.s2_script import ScriptStage
from synthetics.stages.s4_voice import VoiceStage
from synthetics.text.singlish import Lexicon, canonical_text, run_lexicon, spoken_as, tts_input

TERMS = [{"term": "SingPass", "say": "sing pass"}, {"term": "PDPC", "say": "say each letter"}, {"term": "Workday"}]


def test_no_key_terms_is_the_plain_lexicon():
    cfg = make_cfg()
    assert cfg.key_terms == [] and run_lexicon(cfg) is Lexicon.load(cfg.lexicon.path)
    assert key_terms_line([]) == ""


def test_terms_set_spelling_and_pronunciation():
    lex = run_lexicon(make_cfg(key_terms=TERMS))
    assert lex.version.startswith(Lexicon.load(make_cfg().lexicon.path).version + "+kt-")
    assert canonical_text("can login with singpass already", lex).endswith("SingPass already")
    spoken = " ".join(s.value for s in tts_input("the PDPC deadline", lex))
    assert "P D P C" in spoken
    assert spoken_as("PDPC", "letters") == "P D P C" and spoken_as("SingPass", " sing pass ") == "sing pass"
    line = key_terms_line([KeyTerm(**t) for t in TERMS])
    assert "SingPass, PDPC, Workday" in line


def test_only_text_stages_see_key_terms_in_their_cache_slice():
    plain, terms = make_cfg(), make_cfg(key_terms=TERMS)
    assert SampleStage().config_slice(plain) == SampleStage().config_slice(terms)
    assert ScriptStage().config_slice(plain) != ScriptStage().config_slice(terms)
    assert VoiceStage().config_slice(plain) != VoiceStage().config_slice(terms)
