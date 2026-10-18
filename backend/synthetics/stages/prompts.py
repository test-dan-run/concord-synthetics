"""Versioned prompt templates for stages 2 and 3. Bump the version key when wording changes: it is part of
the stage config slice (so the cache key) and is recorded in script.json / script_validation.json."""

from __future__ import annotations

from ..artifacts.meeting import MeetingSpec, Person

# ---------------------------------------------------------------- stage 2: persona prompts

PERSONA = {
    "persona_v1": """\
# Persona {sid} · {position_line}
You are {sid}, {name}, the {role} in a Singapore workplace meeting ({meeting_type}).
Meeting: {topic}. {n} participants, chaired by {chair_id} ({chair_name}).{hybrid_line}{presentation_line}

Profile
- Seniority: {seniority}{chair_note}
- Verbosity: {verbosity} ({verbosity_hint})
- L1: {l1}; accent: {accent}
- Code-switching: {code_switching}; {switch_hint}
- Turn share target: {share_pct:.1f}% of turns (about {target_turns} of {total_turns})

Position
{position_details}

Stance in this meeting
{stance}

Rules
- Speak only in the turns the plan gives {sid}, in {sid}'s own voice.
- {verbosity_rule}
- Never narrate, describe actions or add stage directions; only what {sid} says aloud.
""",
}

VERBOSITY_HINT = {"high": "long turns, 2-4 sentences, explains and elaborates",
                  "medium": "one or two sentences per turn",
                  "low": "short turns, a few words to one sentence, speaks only when needed"}
VERBOSITY_RULE = {"high": "Turns can run 2-4 sentences; you tend to think aloud and elaborate.",
                  "medium": "Keep most turns to one or two sentences.",
                  "low": "Keep turns short; a few words is often enough."}
SWITCH_HINT = {"low": "mostly Singapore English, Singlish particles now and then",
               "medium": "Singlish particles in about a third of turns, occasional words or phrases in {langs}",
               "high": "frequent Singlish particles, regularly switches mid-sentence into {langs}"}


def position_line(m: MeetingSpec, p: Person) -> str:
    if p.position_type == "table":
        return f"table seat {m.table.slots[p.seat.index].label}"
    if p.position_type == "wall":
        return f"wall seat ({m.wall_chairs[p.seat.index].wall} wall)"
    if p.position_type == "presenter":
        return "presenter, standing at the front"
    return "remote, on the speakerphone"


def position_details(m: MeetingSpec, p: Person) -> str:
    if p.position_type == "table":
        return "- You sit at the table with the others."
    if p.position_type == "wall":
        why = ("the table was full when you arrived" if p.wall_reason == "overflow"
               else "you chose a chair by the wall although the table had free seats")
        return (f"- You sit on a chair along the wall, away from the table: {why}.\n"
                "- Wall-seated attendees speak less and more briefly; you rarely start a new topic.")
    if p.position_type == "presenter":
        seg = m.dynamics.presentation
        ret = seg.returns.get(p.speaker_id) if seg else None
        after = ("After presenting you return to a seat and join the discussion."
                 if ret is not None else "After presenting you stay standing at the front.")
        return ("- You stand at the front by the screen and present slides during the presentation segment:\n"
                "  longer monologue turns with slide references, then you answer questions from the floor.\n"
                f"- {after}")
    return ("- You join remotely through the speakerphone on the table; you cannot see the room, so you sometimes\n"
            "  ask people to repeat or say who is speaking, and there is a little delay before you respond.")


def render_persona(m: MeetingSpec, p: Person, *, version: str, stance: str, total_turns: int, target_turns: int) -> str:
    chair = m.person(m.dynamics.chair_id)
    langs = ", ".join(p.persona.switch_languages) or "Singlish"
    hybrid = ""
    if m.dynamics.hybrid:
        hybrid = f" Hybrid: {', '.join(m.dynamics.remote_speakers)} join by speakerphone."
    pres = ""
    if m.dynamics.presentation:
        seg = m.dynamics.presentation
        pres = (f" Presentation segment by {' and '.join(seg.presenters)} from turn {seg.start_turn + 1} "
                f"to {seg.end_turn}, then Q&A from the floor.")
    return PERSONA[version].format(
        sid=p.speaker_id, position_line=position_line(m, p), name=p.persona.display_name, role=p.persona.role,
        meeting_type=m.dynamics.meeting_type, topic=m.dynamics.topic, n=m.n_speakers, chair_id=chair.speaker_id,
        chair_name=chair.persona.display_name, hybrid_line=hybrid, presentation_line=pres,
        seniority=p.persona.seniority, chair_note=(" (you chair this meeting)" if p.is_chair else ""),
        verbosity=p.persona.verbosity, verbosity_hint=VERBOSITY_HINT[p.persona.verbosity], l1=p.persona.l1,
        accent=p.persona.accent, code_switching=p.persona.code_switching,
        switch_hint=SWITCH_HINT[p.persona.code_switching].format(langs=langs), share_pct=100 * p.turn_share,
        target_turns=target_turns, total_turns=total_turns, position_details=position_details(m, p),
        stance=stance.strip(), verbosity_rule=VERBOSITY_RULE[p.persona.verbosity])


# ---------------------------------------------------------------- stage 2: script system prompt and chunks

SCRIPT_SYSTEM = {
    "persona_v1": """\
You write realistic transcripts of Singapore workplace meetings for training speech recognition and speaker
diarisation. The transcript must sound like real people talking, not like a written script: people hedge,
repeat themselves, restart sentences, trail off, react briefly, talk over each other and drift off topic.

Transcription convention (follow exactly):
- Discourse particles in parentheses with canonical spelling: (lah) (leh) (lor) (meh) (hor) (ah) (sia) (mah) (liao) (wor).
- Fillers in parentheses, only these: (uh) (um) (mm).
- Non-speech as square-bracket tags, only these: [laugh] [cough] [breath] [noise].
- Mandarin in Simplified Chinese characters; Tamil in Tamil script; Malay in standard Latin spelling.
- Singlish loanwords in romanised Singapore spelling (shiok, paiseh, kiasu, dabao, makan, alamak, kaypoh).
- Acronyms uppercase without dots (HDB, MRT, CPF, KPI). Numbers written as spoken words (twenty twenty six).
- False starts as a fragment with a trailing hyphen: "we wan- we want to".
- A turn that gets interrupted ends cut off with a trailing hyphen: "I think we should-".

Interaction types in the plan:
- normal: an ordinary turn that follows the previous one.
- overlap_start: starts a little before the previous speaker finishes; often a quick reaction or agreement.
- interruption: cuts in while the previous speaker is mid-sentence, so the previous turn ends cut off with "-".
- backchannel: 1-3 words said while someone else is talking, e.g. "(mm)", "ya", "ok (lah)", "right"; the main
  speaker carries on in their next turn as if uninterrupted.
- side: a quiet aside between two neighbours, off the main thread (a joke, a quick question, logistics).
- presentation: the presenter's monologue, with slide references ("on this slide", "next slide").

Privacy: use only the names in the roster below; invent fictional companies, products and places; never
include NRIC numbers, phone numbers, emails or addresses.

Meeting brief:
{brief}

Participants (each block is that person's persona prompt):

{personas}
""",
}

CHUNK_USER = {
    "persona_v2": """\
{context}Write turns {first} to {last} of {total}.{phase_note}
Plan, one line per turn (index | speaker | interaction):
{plan}
{feedback}
Backchannel turns are one to three words only, e.g. (mm), ya, ok (lah), right. A turn followed by an
interruption breaks off mid-sentence with a trailing hyphen. Inside "text" use single quotes, never double quotes.
Return JSON {{"turns": [{{"i": <index>, "speaker_id": "<speaker>", "text": "<what they say>"}}]}} with exactly
one entry per plan line, in the same order, and the same speaker as the plan.""",
    "persona_v1": """\
{context}Write turns {first} to {last} of {total}.{phase_note}
Plan, one line per turn (index | speaker | interaction):
{plan}
{feedback}
Return JSON {{"turns": [{{"i": <index>, "speaker_id": "<speaker>", "text": "<what they say>"}}]}} with exactly
one entry per plan line, in the same order, and the same speaker as the plan.""",
}


def chunk_schema(n: int) -> dict:
    return {"title": "script_chunk", "type": "object", "additionalProperties": False, "required": ["turns"],
            "properties": {"turns": {"type": "array", "minItems": n, "maxItems": n, "items": {
                "type": "object", "additionalProperties": False, "required": ["i", "speaker_id", "text"],
                "properties": {"i": {"type": "integer"}, "speaker_id": {"type": "string"}, "text": {"type": "string"}}}}}}


# ---------------------------------------------------------------- stage 2: meeting brief

BRIEF = {
    "persona_v1": """\
Plan a fictional Singapore workplace meeting.

Meeting type: {meeting_type}
Topic: {topic}
Length: about {duration:.0f} minutes, {n} participants{hybrid}.{presentation}

Roster:
{roster}

Write a meeting brief (120-200 words): a fictional organisation and project, the concrete situation, three to
five agenda items, and specifics people can refer to (fictional names of vendors, systems, places, dates and
amounts written as words). Then give each participant a stance: what they want from this meeting, what worries
them, and how they relate to the others (one or two sentences each). Use only the roster names for people.""",
}

BRIEF_SCHEMA = {"title": "meeting_brief", "type": "object", "additionalProperties": False,
                "required": ["brief", "stances"], "properties": {
                    "brief": {"type": "string"},
                    "stances": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                           "required": ["speaker_id", "stance"],
                                                           "properties": {"speaker_id": {"type": "string"},
                                                                          "stance": {"type": "string"}}}}}}


def roster_lines(m: MeetingSpec) -> str:
    lines = []
    for p in m.people:
        chair = ", chair" if p.is_chair else ""
        lines.append(f"- {p.speaker_id}: {p.persona.display_name}, {p.persona.role} ({p.persona.seniority}{chair}), "
                     f"{position_line(m, p)}, L1 {p.persona.l1}, code-switching {p.persona.code_switching}")
    return "\n".join(lines)


# ---------------------------------------------------------------- stage 3: judge

JUDGE_SYSTEM = {
    "rubric_v2": """\
You are an expert on Singapore workplace conversation and Singlish, rating synthetic meeting transcripts that
will train speech recognition and diarisation models. Be a strict, calibrated rater: most synthetic transcripts
have visible flaws, and a 5 should be rare.

Real meetings are messy. Reward realistic messiness: fillers, false starts, repetitions, trailing off, overlaps,
backchannels, interruptions, uneven participation, small digressions. Penalise polish: essay-like turns, every
speaker using complete well-formed sentences, neat summaries, speakers restating the agenda, everyone agreeing
politely, forced or excessive Singlish, particles in implausible places, personas that all sound the same.

Score each criterion from 1 to 5 with these anchors:
5 = could pass for a transcript of a real Singapore meeting; 4 = natural with a few scripted moments;
3 = clearly scripted in places but usable; 2 = mostly stilted or off-persona; 1 = unusable.
- coherence: the discussion stays on its topic and agenda; turns respond to what came before; questions get answers.
- singlish: particles, code-switching and local references are natural and fit each speaker's L1 and level.
- realism: it sounds like real people talking, with realistic messiness, not a written script.
- persona_consistency: for each speaker, the turns fit their persona (role, seniority, verbosity, code-switching).
- overall: would you accept this transcript as training data (not necessarily an average).

The plan fixes who speaks and each turn's interaction type, including short backchannels ([backchannel] turns
such as '(mm)' or 'ya') and the share of turns per speaker; do not penalise those, only how well each is written.

In the rationale (at most 200 words) first name the three least realistic turns by turn id and why, then the
strengths. Then give the scores, then feedback: concrete instructions for rewriting the script (empty if none).
Use single quotes inside strings, never double quotes.""",
    "rubric_v1": """\
You are an expert on Singapore workplace conversation and Singlish, rating synthetic meeting transcripts that
will train speech recognition and diarisation models. Real meetings are messy. Reward realistic messiness:
fillers, false starts, repetitions, trailing off, overlaps, backchannels, interruptions, uneven participation,
small digressions. Penalise polish: essay-like turns, everyone speaking in complete well-formed paragraphs,
neat summaries, forced or excessive Singlish, particles in implausible places.

Score each criterion from 1 (poor) to 5 (excellent):
- coherence: the discussion stays on its topic and agenda; turns respond to what came before; questions get answers.
- singlish: particles, code-switching and local references are natural and fit each speaker's L1 and level.
- realism: it sounds like real people talking, with realistic messiness, not a written script.
- persona_consistency: for each speaker, the turns fit their persona (role, seniority, verbosity, code-switching).
- overall: your holistic judgement of whether this transcript is good training data (not necessarily an average).

Write the rationale first (at most 200 words, cite turn ids), then the scores, then feedback: concrete
instructions for rewriting the script if it should improve (empty if nothing important).""",
}

JUDGE_USER = {
    "rubric_v2": """\
Meeting: {meeting_type}; topic: {topic}; {n} participants.

Personas:
{personas}

Transcript ({shown} of {total} turns{excerpt_note}); each line is turn id, speaker, interaction type, text:
{lines}""",
    "rubric_v1": """\
Meeting: {meeting_type}; topic: {topic}; {n} participants.

Personas:
{personas}

Transcript ({shown} of {total} turns{excerpt_note}); each line is turn id, speaker, interaction type, text:
{lines}""",
}


def judge_schema() -> dict:
    score = {"type": "number", "minimum": 1, "maximum": 5}
    return {"title": "script_judgement", "type": "object", "additionalProperties": False,
            "required": ["rationale", "coherence", "singlish", "realism", "persona_consistency", "overall", "feedback"],
            "properties": {
                "rationale": {"type": "string", "maxLength": 1600},  # ~200 words; bounds runaway answers
                "coherence": score, "singlish": score, "realism": score,
                "persona_consistency": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["speaker_id", "score"],
                    "properties": {"speaker_id": {"type": "string"}, "score": score}}},
                "overall": score,
                "feedback": {"type": "string", "maxLength": 1200}}}


def judge_persona_lines(m: MeetingSpec) -> str:
    out = []
    for p in m.people:
        out.append(f"- {p.speaker_id} {p.persona.display_name}: {p.persona.role}, {p.persona.seniority}"
                   f"{' (chair)' if p.is_chair else ''}, {position_line(m, p)}, verbosity {p.persona.verbosity}, "
                   f"L1 {p.persona.l1}, code-switching {p.persona.code_switching}"
                   f"{' into ' + ', '.join(p.persona.switch_languages) if p.persona.switch_languages else ''}, "
                   f"target share {100 * p.turn_share:.1f}%")
    return "\n".join(out)


# persona_v2 changed only the chunk instructions (backchannel length, cut-offs)
PERSONA["persona_v2"] = PERSONA["persona_v1"]
SCRIPT_SYSTEM["persona_v2"] = SCRIPT_SYSTEM["persona_v1"]
BRIEF["persona_v2"] = BRIEF["persona_v1"]
