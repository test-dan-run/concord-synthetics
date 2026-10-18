"""Numbers written as spoken words ("twenty twenty six"), the convention's rule for `text`."""

from __future__ import annotations

import re

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
           "twelve": "twelfth"}


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + ("" if o == 0 else " " + ONES[o])
    if n < 1000:
        h, rest = divmod(n, 100)
        return ONES[h] + " hundred" + ("" if rest == 0 else " " + cardinal(rest))
    for value, name in SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return cardinal(head) + " " + name + ("" if rest == 0 else " " + cardinal(rest))
    raise AssertionError(n)


def year(n: int) -> str:
    """1900-2099 read in pairs: 2026 -> twenty twenty six, 2005 -> two thousand and five, 1990 -> nineteen ninety."""
    hi, lo = divmod(n, 100)
    if n % 1000 < 10 and n >= 2000:
        return "two thousand" + ("" if lo == 0 else " and " + ONES[lo])
    if lo == 0:
        return cardinal(hi) + " hundred"
    return cardinal(hi) + " " + (("oh " + ONES[lo]) if lo < 10 else cardinal(lo))


def ordinal(n: int) -> str:
    words = cardinal(n).split(" ")
    last = words[-1]
    if last in ORDINAL:
        words[-1] = ORDINAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def digits(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s)


_NUM = re.compile(
    r"(?P<cur>S\$|\$)?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?(?P<suffix>%|(?:st|nd|rd|th|k|m|pm|am)(?![A-Za-z]))?(?![\d.,]\d)",
    re.I)


def _replace(m: re.Match) -> str:
    raw, dec, suffix, cur = m.group("num"), m.group("dec"), (m.group("suffix") or "").lower(), m.group("cur")
    n = int(raw.replace(",", ""))
    if suffix in ("st", "nd", "rd", "th") and not dec:
        return ordinal(n)
    if not dec and not cur and not suffix and len(raw) == 4 and 1900 <= n <= 2099:
        words = year(n)
    elif len(raw) > 1 and raw.startswith("0"):
        words = digits(raw)  # codes, phone numbers
    else:
        words = cardinal(n)
    if dec:
        words += " point " + digits(dec)
    if suffix == "k":
        words += " thousand"
    elif suffix == "m":
        words += " million"
    elif suffix == "%":
        words += " percent"
    elif suffix in ("am", "pm"):
        words += " " + " ".join(suffix)
    if cur:
        words += " dollars"
    return words


_GLUED = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Z]{1,3}\b)")


def spell_numbers(text: str) -> str:
    """Replace every digit group in `text` with words ("Q3" -> "Q three", "5G" -> "five G").
    Leaves text without digits unchanged."""
    if not any(c.isdigit() for c in text):
        return text
    text = _GLUED.sub(" ", text)
    return _NUM.sub(_replace, text)
