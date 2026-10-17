"""Output-side protection: leak detection, redaction and streaming restoration.

* :class:`LeakDetector` inspects model output for signs that the prompt, the
  anchor or the protected values were disclosed.
* :class:`StreamRestorer` / :func:`restore_stream` decode aliases in streamed
  replies even when an alias is split across chunks.

Standard library only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import (
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
    """

    def __init__(
        self,
        mapping: Optional[Mapping[str, str]] = None,
        *,
        session_ids: Iterable[str] = (),
        min_value_len: int = 4,
        check_values: bool = True,
    ) -> None:
        self._mapping: Dict[str, str] = dict(mapping or {})
        self._session_ids = tuple(i for i in session_ids if i)
        self._alias_re = _bounded(self._mapping)
        values = (v for v in self._mapping.values() if len(v) >= min_value_len)
        self._value_re = _bounded(values) if check_values else None

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


class StreamRestorer:
    """Incrementally decode aliases in streamed text.

    Text is released only up to the last non-word character, so an alias that
    arrives split over several chunks (``"α"`` then ``"₁"``, or ``"⟦"`` then
    ``"α⟧"``) is never emitted half-decoded.  Call :meth:`flush` at the end.
    """

    def __init__(self, mapping: Mapping[str, str], max_hold: int = 64) -> None:
        self._mapping = dict(mapping)
        self._max_hold = max_hold
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Add ``chunk`` and return the text that is safe to emit now."""
        self._buffer += chunk
        buf = self._buffer
        cut = len(buf)
        opener = buf.rfind("⟦")
        if opener != -1 and buf.find("⟧", opener) == -1:
            cut = opener
        while cut > 0 and _WORD.match(buf[cut - 1]):
            cut -= 1
        if len(buf) - cut > self._max_hold:
            cut = len(buf)
        self._buffer = buf[cut:]
        return _anchor.substitute(buf[:cut], self._mapping)

    def flush(self) -> str:
        """Return whatever is still buffered, decoded."""
        tail, self._buffer = self._buffer, ""
        return _anchor.substitute(tail, self._mapping)


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
