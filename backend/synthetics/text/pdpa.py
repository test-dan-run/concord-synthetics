"""PDPA stripping for any past-meeting context carried into a prompt (judge feedback, earlier rounds).

Removes identifiers (NRIC / FIN, phone numbers, emails, URLs, postal codes) and person names that are not
in the current meeting's roster. Roster names are synthetic, so the script may use them freely.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

NRIC = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+65[\s-]?)?[3689]\d{3}[\s-]?\d{4}(?!\d)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
POSTAL = re.compile(r"\bSingapore\s+\d{6}\b|\bS\(?\d{6}\)?(?!\d)", re.I)


def strip_identifiers(text: str, *, roster_names: Iterable[str] = (), known_names: Iterable[str] = ()) -> str:
    """Replace identifiers with neutral placeholders. `known_names` are names to redact unless they are in
    `roster_names` (e.g. every synthetic name the config can produce)."""
    t = NRIC.sub("[ID]", text)
    t = EMAIL.sub("[email]", t)
    t = URL.sub("[link]", t)
    t = POSTAL.sub("[address]", t)
    t = PHONE.sub("[phone]", t)
    roster = [n.lower() for n in roster_names]
    for name in sorted(set(known_names), key=len, reverse=True):
        if any(re.search(rf"\b{re.escape(name.lower())}\b", r) for r in roster):
            continue  # part of a name in this meeting's roster
        t = re.sub(rf"\b{re.escape(name)}\b", "[name]", t)
    return t
