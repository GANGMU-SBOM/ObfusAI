"""Opt-in detectors for personal data and credentials, with checksum validation.

Each detector is a regex plus an optional validator that rejects look-alikes
(a 16-digit number that fails the Luhn check is not a card number).  They are
used through ``ObfusAIComponent(extra_detectors=[...])`` and :func:`obfusai.audit.audit`.

Also here: ``dates`` (ISO, numeric, month-name and Chinese dates with optional times),
which the default intensities use so a date becomes one alias instead of three numbers.

Names: ``phones`` (international and common national formats written with a ``+``
or separators), ``cn_phones`` (mainland-China mobile numbers), ``cards`` (Luhn),
``ssn`` (US), ``cn_ids`` (18-digit resident ID with checksum), ``ibans`` (mod-97),
``uuids`` and ``secrets`` (well-known API-key and token formats, JWTs, private-key
blocks).

Standard library only.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Callable, Dict, Pattern, Tuple

__all__ = ["PATTERNS", "PII_DETECTORS", "VALIDATORS"]


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def luhn_ok(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def ssn_ok(value: str) -> bool:
    area, group, serial = value.split("-")
    return area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"


_CN_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CN_CHECK = "10X98765432"


def cn_id_ok(value: str) -> bool:
    body, check = value[:17], value[17].upper()
    if not body.isdigit():
        return False
    try:
        date(int(body[6:10]), int(body[10:12]), int(body[12:14]))
    except ValueError:
        return False
    total = sum(int(d) * w for d, w in zip(body, _CN_WEIGHTS))
    return _CN_CHECK[total % 11] == check


def iban_ok(value: str) -> bool:
    compact = value.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def date_ok(value: str) -> bool:
    """Reject look-alikes such as ``1/2/3`` or ``2024-13-45``; times and month-name dates pass."""
    numeric = re.match(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", value)
    if numeric:
        year, month, day = (int(g) for g in numeric.groups())
    else:
        short = re.fullmatch(r"(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})", value)
        if not short:
            clock = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", value)
            if clock:  # bare time such as "25:99" is not a time
                hour, minute, second = clock.group(1), clock.group(2), clock.group(3) or "0"
                return int(hour) <= 24 and int(minute) <= 59 and int(second) <= 59
            return True  # month-name dates were fully matched by their pattern
        first, second, year = (int(g) for g in short.groups())
        month, day = (first, second) if first <= 12 else (second, first)
        if len(short.group(3)) == 2:
            year += 2000
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def phone_ok(value: str) -> bool:
    digits = _digits(value)
    if not 9 <= len(digits) <= 15:
        return False
    # bare digit groups separated only by spaces ("1234 5698 7654") are too ambiguous
    return value.startswith("+") or any(ch in value for ch in "()-.")


_MONTH_NAME = r"(?:" + "|".join(m + r"[a-z]*\.?" for m in _MONTHS) + r")"
_TIME = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s?[AaPp]\.?[Mm]\.?)?"

PATTERNS: Dict[str, Tuple[Tuple[Pattern[str], bool], ...]] = {
    "dates": tuple(
        (re.compile(p), False)
        for p in (
            # 2024-01-05, 2024-01-05T10:30:00Z, 2024/1/5 10:30
            r"(?<![\w.])\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[T ]" + _TIME + r"(?:Z|[+-]\d{2}:?\d{2})?)?(?![\w])",
            # 05/01/2024, 5.1.24
            r"(?<![\w.])\d{1,2}[/.]\d{1,2}[/.]\d{2,4}(?![\w/.])",
            # 2024年1月5日
            r"\d{4}年\d{1,2}月(?:\d{1,2}日?)?",
            # Jan 5, 2024 / March 5th / 5 January 2024
            r"\b" + _MONTH_NAME + r" \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?\b",
            r"\b\d{1,2}(?:st|nd|rd|th)? " + _MONTH_NAME + r"(?: \d{4})?\b",
            # 10:30, 10:30:15, 9:05 PM
            r"(?<![\w:.])" + _TIME + r"(?![\w:])",
        )
    ),
    "secrets": tuple(
        (re.compile(p), False)
        for p in (
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",  # JWT
            r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",  # AWS access key id
            r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",  # GitHub token
            r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",  # Slack token
            r"\bAIza[0-9A-Za-z_-]{35}\b",  # Google API key
            r"\bsk-[A-Za-z0-9_-]{20,}\b",  # OpenAI / Anthropic style secret keys
        )
    ),
    "uuids": ((re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"), False),),
    "cards": ((re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])"), False),),
    "ssn": ((re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"), False),),
    "cn_ids": ((re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])"), False),),
    "cn_phones": ((re.compile(r"(?<![\d])(?:\+?86[ -]?)?1[3-9]\d{9}(?!\d)"), False),),
    "ibans": ((re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"), False),),
    "phones": (
        (re.compile(r"(?<![\w.+-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}(?![\w-])"), False),
        (re.compile(r"(?<![\w.])\+\d{8,15}(?!\w)"), False),
    ),
}

VALIDATORS: Dict[str, Callable[[str], bool]] = {
    "dates": date_ok,
    "cards": luhn_ok,
    "ssn": ssn_ok,
    "cn_ids": cn_id_ok,
    "ibans": iban_ok,
    "phones": phone_ok,
}

#: The personal-data detectors (everything here except ``secrets`` and ``uuids``).
PII_DETECTORS: Tuple[str, ...] = ("phones", "cn_phones", "cards", "ssn", "cn_ids", "ibans")
