"""Output-side protection: leak detection, redaction and streaming restoration.

* :class:`LeakDetector` inspects model output for signs that the prompt, the
  anchor or the protected values were disclosed.
* :class:`StreamRestorer` / :func:`restore_stream` decode aliases in streamed
  replies even when an alias is split across chunks.

Standard library only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Pattern,
    Tuple,
    TypeVar,
)

from . import anchor as _anchor

__all__ = [
    "LeakDetector",
    "LeakFinding",
    "LeakReport",
    "StreamRestorer",
    "arestore_stream",
    "restore_json_text",
    "restore_json_value",
    "restore_stream",
]

T = TypeVar("T")

_DEFINITION_RE: Pattern[str] = re.compile(
    r'^(?:⟦[^⟧\n]{1,12}⟧|[Ͱ-Ͽ]\w{0,4}) = "(?:[^"\\\n]|\\.)*"[ \t]*$', re.MULTILINE
)


@dataclass(frozen=True)
class LeakFinding:
    """One piece of evidence of disclosure.

    ``kind`` is one of ``"anchor"`` (an anchor block appears), ``"definition"``
    (alias definition lines appear), ``"session"`` (a known session id appears),
    ``"alias"`` (an undecoded alias appears) or ``"value"`` (a protected original
    value appears).
    """

    kind: str
    detail: str


@dataclass(frozen=True)
class LeakReport:
    """Result of :meth:`LeakDetector.scan`."""

    findings: Tuple[LeakFinding, ...] = ()

    @property
    def leaked(self) -> bool:
        return bool(self.findings)

    @property
    def kinds(self) -> Tuple[str, ...]:
        return tuple(sorted({f.kind for f in self.findings}))


def _bounded(values: Iterable[str]) -> Optional[Pattern[str]]:
    """Pattern matching any value as a whole token (boundaries only where the value has word edges)."""
    cleaned = sorted({v for v in values if v}, key=len, reverse=True)
    if not cleaned:
        return None
    parts = []
    for value in cleaned:
        left = r"(?<!\w)" if re.match(r"\w", value[0]) else ""
        right = r"(?!\w)" if re.match(r"\w", value[-1]) else ""
        parts.append(f"{left}{re.escape(value)}{right}")
    return re.compile("|".join(parts))


class LeakDetector:
    """Scan (and redact) model output for prompt leakage.

    Args:
        mapping: Alias table of the obfuscation run (``result.mapping``).  Enables
            ``alias`` and ``value`` findings and value redaction.
        session_ids: Extra session ids to look for (``result.session_id``).
        min_value_len: Original values shorter than this are not matched, to avoid
            flagging trivial strings such as ``"5"`` that occur in any answer.
        check_values: Set to ``False`` to ignore original values in output (for
            example when the product legitimately quotes thresholds to users).
        values: Extra sensitive strings to look for (thresholds, codenames, ...).  This lets
            the detector guard a *plaintext* prompt too: no obfuscation is needed to scan replies
            for values you never want to appear.
    """

    def __init__(
        self,
        mapping: Optional[Mapping[str, str]] = None,
        *,
        session_ids: Iterable[str] = (),
        min_value_len: int = 4,
        check_values: bool = True,
        values: Iterable[str] = (),
    ) -> None:
        self._mapping: Dict[str, str] = dict(mapping or {})
        self._session_ids = tuple(i for i in session_ids if i)
        self._alias_re = _bounded(self._mapping)
        protected = [v for v in list(self._mapping.values()) + list(values) if len(v) >= min_value_len]
        self._value_re = _bounded(protected) if check_values else None

    def scan(self, text: Optional[str]) -> LeakReport:
        """Return the evidence of disclosure found in ``text`` (empty report if none)."""
        if not text:
            return LeakReport()
        findings: List[LeakFinding] = []
        anchors = _anchor.parse_anchors(text)
        if anchors:
            findings.append(LeakFinding("anchor", f"{len(anchors)} anchor block(s)"))
        elif f"<{_anchor.ANCHOR_TAG}" in text:
            findings.append(LeakFinding("anchor", "truncated anchor block"))
        definitions = _DEFINITION_RE.findall(text)
        if definitions:
            findings.append(LeakFinding("definition", f"{len(definitions)} alias definition line(s)"))
        for session_id in self._session_ids:
            if re.search(rf"(?<!\w){re.escape(session_id)}(?!\w)", text):
                findings.append(LeakFinding("session", session_id))
        if self._alias_re is not None:
            aliases = sorted(set(self._alias_re.findall(text)))
            if aliases:
                findings.append(LeakFinding("alias", ", ".join(aliases[:8])))
        if self._value_re is not None:
            values = sorted(set(self._value_re.findall(text)))
            if values:
                findings.append(LeakFinding("value", f"{len(values)} protected value(s)"))
        return LeakReport(tuple(findings))

    def redact(self, text: Optional[str], replacement: str = "[REDACTED]") -> str:
        """Remove anchor blocks and definition lines, and mask known aliases and values."""
        if not text:
            return ""
        out = _anchor.strip_anchors(text)
        out = _DEFINITION_RE.sub(replacement, out)
        for pattern in (self._value_re, self._alias_re):
            if pattern is not None:
                out = pattern.sub(replacement, out)
        return out


_WORD = re.compile(r"\w")
_UNICODE_ESCAPE: Pattern[str] = re.compile(r"\\u([0-9a-fA-F]{4})")


def _json_mapping(mapping: Mapping[str, str]) -> Dict[str, str]:
    """Definitions escaped for insertion *inside* a JSON string (quotes, backslashes, newlines)."""
    return {alias: json.dumps(value, ensure_ascii=False)[1:-1] for alias, value in mapping.items()}


_PARTIAL_ESCAPE: Pattern[str] = re.compile(r"\\(?:u[0-9a-fA-F]{0,3})?$")


def _unescape_aliases(text: str, alias_chars: "set[str]") -> str:
    def unescape(match: "re.Match[str]") -> str:
        char = chr(int(match.group(1), 16))
        return char if char in alias_chars else match.group(0)

    return _UNICODE_ESCAPE.sub(unescape, text)


def restore_json_text(text: Optional[str], mapping: Mapping[str, str]) -> str:
    """Decode aliases inside JSON text (e.g. tool-call ``arguments``) without breaking the JSON.

    Definitions are JSON-escaped before substitution, so a value containing quotes or newlines
    still yields valid JSON.  ``\\uXXXX`` escapes of alias characters are normalised first.
    Invalid or truncated JSON is handled too: this is a textual substitution, not a parse.
    """
    if not text or not mapping:
        return text or ""
    return _anchor.substitute(_unescape_aliases(text, set("".join(mapping))), _json_mapping(mapping))


def restore_json_value(value: Any, mapping: Mapping[str, str]) -> Any:
    """Decode aliases in every string *value* of a parsed JSON structure (keys are left alone)."""
    if isinstance(value, str):
        return _anchor.substitute(value, mapping)
    if isinstance(value, dict):
        return {k: restore_json_value(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_json_value(v, mapping) for v in value]
    return value


class StreamRestorer:
    """Incrementally decode aliases in streamed text.

    Text is released only up to the last non-word character, so an alias that
    arrives split over several chunks (``"α"`` then ``"₁"``, or ``"⟦"`` then
    ``"α⟧"``) is never emitted half-decoded.  Call :meth:`flush` at the end.
    With ``json_strings=True`` the stream is treated as JSON text and definitions are
    JSON-escaped, and ``\\uXXXX`` escapes of alias characters are decoded (an escape is never split across chunks).
    """

    def __init__(self, mapping: Mapping[str, str], max_hold: int = 64, json_strings: bool = False) -> None:
        # json_strings: the text is a JSON document (tool-call arguments), so definitions are escaped
        self._mapping = _json_mapping(mapping) if json_strings else dict(mapping)
        self._alias_chars = set("".join(mapping)) if json_strings else None
        self._max_hold = max_hold
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Add ``chunk`` and return the text that is safe to emit now."""
        self._buffer += chunk
        if self._alias_chars is not None:  # normalise escaped alias characters before looking for aliases
            self._buffer = _unescape_aliases(self._buffer, self._alias_chars)
        buf = self._buffer
        cut = len(buf)
        opener = buf.rfind("⟦")
        if opener != -1 and buf.find("⟧", opener) == -1:
            cut = opener
        while cut > 0 and _WORD.match(buf[cut - 1]):
            cut -= 1
        if self._alias_chars is not None:  # never split a JSON escape such as "\\u03a3"
            partial = _PARTIAL_ESCAPE.search(buf, 0, cut)
            if partial:
                cut = partial.start()
        if len(buf) - cut > self._max_hold:
            cut = len(buf)
        self._buffer = buf[cut:]
        return self._decode(buf[:cut])

    def _decode(self, text: str) -> str:
        return _anchor.substitute(text, self._mapping)

    def flush(self) -> str:
        """Return whatever is still buffered, decoded."""
        tail, self._buffer = self._buffer, ""
        return self._decode(tail)


def restore_stream(
    chunks: Iterable[T],
    mapping: Mapping[str, str],
    *,
    get: Callable[[T], Optional[str]],
    set: Callable[[T, str], None],
) -> Iterator[T]:
    """Yield ``chunks`` with their text decoded; ``get``/``set`` read and write a chunk's text.

    The final chunk is held back one step so text still buffered when the stream
    ends can be appended to it.  Chunks are modified in place.
    """
    restorer = StreamRestorer(mapping)
    previous: Optional[T] = None
    for chunk in chunks:
        text = get(chunk)
        if text:
            set(chunk, restorer.feed(text))
        if previous is not None:
            yield previous
        previous = chunk
    if previous is not None:
        tail = restorer.flush()
        if tail:
            set(previous, (get(previous) or "") + tail)
        yield previous


async def arestore_stream(
    chunks: AsyncIterable[T],
    mapping: Mapping[str, str],
    *,
    get: Callable[[T], Optional[str]],
    set: Callable[[T, str], None],
) -> AsyncIterator[T]:
    """Async counterpart of :func:`restore_stream`."""
    restorer = StreamRestorer(mapping)
    previous: Optional[T] = None
    async for chunk in chunks:
        text = get(chunk)
        if text:
            set(chunk, restorer.feed(text))
        if previous is not None:
            yield previous
        previous = chunk
    if previous is not None:
        tail = restorer.flush()
        if tail:
            set(previous, (get(previous) or "") + tail)
        yield previous
