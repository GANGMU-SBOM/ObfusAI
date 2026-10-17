"""Core engine of ObfusAI.

ObfusAI rewrites sensitive prompt text so that proprietary metrics, numbers,
keywords and programmatic identifiers are replaced with abstract Greek
symbols, then appends a *semantic anchor* (a meta-instruction block holding the
symbol table) so a capable LLM can resolve the symbols and follow the original
logic.

Security note: this is obfuscation, not encryption.  The anchor must travel
in the same context window as the obfuscated text, so anything that can make
the model dump its full context can recover both.  ObfusAI raises the effort
needed to read a leaked prompt and keeps casual prompt-extraction attacks from
yielding readable IP; it does not make a prompt confidential.

The module depends only on the Python standard library.
"""

from __future__ import annotations

import copy
import hashlib
import re
import secrets
import string
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Union,
)

from . import anchor as _anchor
from .anchor import DECOY_VALUE as _DECOY_VALUE

__all__ = [
    "DETECTORS",
    "Intensity",
    "ObfusAIComponent",
    "ObfuscatedPayload",
    "ObfuscationResult",
    "ObfuscationSession",
    "ObfuscationStats",
]

_SUBSCRIPT_DIGITS = str.maketrans(string.digits, "₀₁₂₃₄₅₆₇₈₉")

# Greek glyphs that do not look like Latin letters or digits.
_SYMBOL_POOL: Tuple[str, ...] = tuple("αβγδεζηθικλμνξπρστφχψω") + tuple("ΓΔΘΛΞΠΣΦΨΩ")

#: Spans that must survive untouched so templating and chat-format markers keep working.
_PROTECTED_RE: Pattern[str] = re.compile(
    r"\{\{[^{}\n]*\}\}"  # {{ jinja }}
    r"|\$\{[^{}\n]*\}"  # ${shell}
    r"|\{[A-Za-z_][\w.\[\]]*\}"  # {format_field}
    r"|%\([A-Za-z_]\w*\)[sdrf]"  # %(printf)s
    r"|<\|[^|<>\n]*\|>"  # <|special_token|>
    r"|\[/?INST\]|<</?SYS>>"  # llama-style markers
)

_NUMBER_RE: Pattern[str] = re.compile(
    r"(?<![\w.])"
    r"[$€£¥]?"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:%|\s?(?:bps|bp|ms|MB|GB|TB|USD|EUR|GBP|[kKMBx])(?![\w]))?"
    r"(?!\.\d)(?![\w])"  # not the head of "10.0.0.7" or "1.2.3"
)
_IDENTIFIER_RES: Tuple[Pattern[str], ...] = (
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),  # snake_case
    re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),  # UPPER_SNAKE
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),  # camelCase
    re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b"),  # PascalCase
    re.compile(r"(?<!@)\b[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)+\b"),  # dotted.path (not e-mail domains)
    re.compile(r"\b[A-Za-z_]\w*\(\)"),  # call()
)
_EMAIL_RE: Pattern[str] = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL_RE: Pattern[str] = re.compile(r"https?://[^\s<>\"'`)\]]*[^\s<>\"'`)\].,;:!?]")
_IP_RE: Pattern[str] = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?!\w|\.\d)")
_QUOTED_RE: Pattern[str] = re.compile(r'"([^"\n]{2,200})"|`([^`\n]{2,200})`')
_TITLE_PHRASE_RE: Pattern[str] = re.compile(r"\b[A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,})+\b")
_ACRONYM_RE: Pattern[str] = re.compile(r"\b[A-Z]{3,8}\b")
_LONG_WORD_RE: Pattern[str] = re.compile(r"\b[A-Za-z]{4,}\b")
#: Matches anything that could be mistaken for an alias already present in the input.
_RESERVED_SCAN_RE: Pattern[str] = re.compile(r"⟦[^⟧\n]{0,16}⟧|[\u0370-\u03ff]\w*")

#: Directive / format words that steer model behaviour and should stay readable.
_KEEP_UPPER = frozenset(
    "ALWAYS NEVER MUST NOT ONLY IMPORTANT NOTE WARNING CRITICAL REQUIRED OPTIONAL "
    "TRUE FALSE NONE NULL JSON YAML XML HTML CSV SQL API URL HTTP HTTPS UTF ASCII "
    "AND THE FOR YOU ARE WITH THEN ELSE ANY ALL OUTPUT INPUT USER SYSTEM".split()
)
_COMMON_LONG_WORDS = frozenset(
    "about above after again against always another because before being between "
    "cannot could different directly during either every example following however "
    "include including information instead instruction instructions important "
    "message messages never other output personal possible provide provided "
    "question response responses result results should simply something still "
    "their there these things think those through under until using value values "
    "were what when where whether which while without would write your".split()
)


class Intensity(str, Enum):
    """Obfuscation depth."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


#: name -> ((pattern, replace-capture-groups-instead-of-whole-match), ...)
_DETECTOR_PATTERNS: Dict[str, Tuple[Tuple[Pattern[str], bool], ...]] = {
    "identifiers": tuple((p, False) for p in _IDENTIFIER_RES),
    "quoted": ((_QUOTED_RE, True),),
    "emails": ((_EMAIL_RE, False),),
    "urls": ((_URL_RE, False),),
    "ips": ((_IP_RE, False),),
    "numbers": ((_NUMBER_RE, False),),
    "phrases": ((_TITLE_PHRASE_RE, False),),
    "acronyms": ((_ACRONYM_RE, False),),
    "words": ((_LONG_WORD_RE, False),),
}
#: Detector names accepted by ``detectors=`` / ``extra_detectors=`` (in matching order).
DETECTORS: Tuple[str, ...] = tuple(_DETECTOR_PATTERNS)


@dataclass(frozen=True)
class _Profile:
    """Feature switches for one intensity level."""

    detectors: FrozenSet[str]
    decoys: int
    reverse_values: bool
    wrap_symbols: bool
    shuffle_anchor: bool
    min_word_len: int = 9


_PROFILES: Dict[Intensity, _Profile] = {
    Intensity.LOW: _Profile(frozenset({"identifiers", "numbers"}), 0, False, False, False),
    Intensity.MEDIUM: _Profile(
        frozenset({"identifiers", "numbers", "quoted", "phrases", "acronyms"}),
        0, False, False, False,
    ),
    Intensity.HIGH: _Profile(
        frozenset({"identifiers", "numbers", "quoted", "phrases", "acronyms", "words"}),
        4, True, True, True,
    ),
}


@dataclass(frozen=True)
class ObfuscationStats:
    """Size accounting for an obfuscation run (characters, not tokens)."""

    aliases: int
    replacements: int
    source_chars: int
    text_chars: int
    anchor_chars: int

    @property
    def growth(self) -> float:
        """Output size (text + anchor) divided by source size; ``1.0`` when nothing changed."""
        if not self.source_chars:
            return 1.0
        return (self.text_chars + self.anchor_chars) / self.source_chars


@dataclass(frozen=True)
class ObfuscationResult:
    """Outcome of obfuscating a single text."""

    text: str
    anchor: str
    mapping: Mapping[str, str]
    intensity: Intensity
    session_id: str
    anchor_position: str = "append"
    warnings: Tuple[str, ...] = ()
    stats: Optional[ObfuscationStats] = None

    @property
    def prompt(self) -> str:
        """Obfuscated text combined with its anchor, ready to send to an LLM."""
        return _join_anchor(self.text, self.anchor, self.anchor_position)

    def restore(self, text: str) -> str:
        """Replace aliases found in ``text`` (e.g. a model reply) by their originals."""
        return ObfusAIComponent.deobfuscate(text, self.mapping)


@dataclass(frozen=True)
class ObfuscatedPayload:
    """Outcome of obfuscating a chat-completion payload."""

    payload: Dict[str, Any]
    mapping: Mapping[str, str]
    anchor: str
    session_id: str
    warnings: Tuple[str, ...] = ()
    stats: Optional[ObfuscationStats] = None

    def restore(self, text: str) -> str:
        """Replace aliases found in ``text`` by their originals."""
        return ObfusAIComponent.deobfuscate(text, self.mapping)


@dataclass
class _Span:
    start: int
    end: int
    value: str


def _join_anchor(text: str, anchor: str, position: str) -> str:
    if not anchor:
        return text
    if not text:
        return anchor
    if position == "prepend":
        return f"{anchor}\n\n{text}"
    return f"{text}\n\n{anchor}"


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "surrogatepass")).hexdigest()


class ObfuscationSession:
    """Shared alias table for one or more texts that travel in the same context.

    Create sessions through :meth:`ObfusAIComponent.session`.  Every text passed
    to :meth:`encode` reuses the same aliases, so a single anchor describes all
    of them (for instance a system prompt plus tool descriptions).
    """

    def __init__(self, component: "ObfusAIComponent", corpus: Sequence[str]) -> None:
        self._component = component
        self._profile = _PROFILES[component.intensity]
        joined = "\n".join(corpus)
        self._reserved = set(_RESERVED_SCAN_RE.findall(joined))
        self.session_id: str = _digest(component.key, joined)[:8]
        self._value_to_token: Dict[str, str] = {}
        self._mapping: Dict[str, str] = {}
        self._next_index = 0
        self._warnings: List[str] = []
        self._replacements = 0
        self._source_chars = 0
        self._text_chars = 0

    @property
    def mapping(self) -> Dict[str, str]:
        """Alias -> original value (copy)."""
        return dict(self._mapping)

    @property
    def warnings(self) -> Tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def anchor(self) -> str:
        """Meta-instruction block describing every alias issued so far."""
        return self._component._render_anchor(self)

    @property
    def stats(self) -> ObfuscationStats:
        """Size accounting over everything encoded so far."""
        return ObfuscationStats(
            aliases=len(self._mapping),
            replacements=self._replacements,
            source_chars=self._source_chars,
            text_chars=self._text_chars,
            anchor_chars=len(self.anchor),
        )

    def _wrap(self, bare: str) -> str:
        return f"⟦{bare}⟧" if self._profile.wrap_symbols else bare

    def _is_free(self, bare: str, wrapped: str) -> bool:
        return not (bare in self._reserved or wrapped in self._reserved or wrapped in self._mapping)

    def _new_token(self) -> str:
        order = self._component._symbol_order
        while True:
            idx = self._next_index
            self._next_index += 1
            base = order[idx % len(order)]
            generation = idx // len(order)
            suffix = str(generation).translate(_SUBSCRIPT_DIGITS) if generation else ""
            wrapped = self._wrap(base + suffix)
            if self._is_free(base + suffix, wrapped):
                return wrapped

    def _stable_token(self, value: str) -> str:
        """Alias derived from the value and the key only, so it survives across sessions."""
        order = self._component._symbol_order
        h = int(_digest(self._component.key, "alias", value), 16)
        for probe in range(4096):
            k = h + probe
            bare = order[k % len(order)] + str((k // len(order)) % 1000).translate(_SUBSCRIPT_DIGITS)
            wrapped = self._wrap(bare)
            if self._is_free(bare, wrapped):
                return wrapped
        return self._new_token()  # pragma: no cover - 4096 consecutive collisions

    def _token_for(self, value: str) -> str:
        token = self._value_to_token.get(value)
        if token is None:
            token = (
                self._stable_token(value)
                if self._component.alias_mode == "stable"
                else self._new_token()
            )
            self._value_to_token[value] = token
            self._mapping[token] = value
        return token

    def encode(self, text: Optional[str]) -> str:
        """Obfuscate ``text`` using (and extending) this session's alias table."""
        if text is None or text == "":
            return ""
        if not isinstance(text, str):
            raise TypeError(f"expected str, got {type(text).__name__}")
        self._source_chars += len(text)
        try:
            spans = self._component._extract(text, self._profile, self._warnings)
        except Exception as exc:  # pragma: no cover - defensive, extraction is regex-only
            self._warnings.append(f"extraction aborted: {exc.__class__.__name__}")
            spans = []
        out: List[str] = []
        cursor = 0
        for span in spans:
            out.append(text[cursor : span.start])
            out.append(self._token_for(span.value))
            cursor = span.end
        out.append(text[cursor:])
        encoded = "".join(out)
        self._replacements += len(spans)
        self._text_chars += len(encoded)
        return encoded

    def decorate(self, text: str, position: Optional[str] = None) -> str:
        """Attach this session's anchor to ``text``."""
        return _join_anchor(text, self.anchor, position or self._component.anchor_position)

    def restore(self, text: str) -> str:
        """Replace aliases in ``text`` by their originals."""
        return ObfusAIComponent.deobfuscate(text, self._mapping)


class ObfusAIComponent:
    """Prompt obfuscation engine.

    Args:
        intensity: ``"low"`` (numbers and code identifiers), ``"medium"`` (adds
            quoted strings, acronyms and proper-noun phrases) or ``"high"``
            (adds long domain words, wrapped ``⟦α⟧`` aliases, reversed anchor
            definitions, decoy aliases and a shuffled alias table).  Higher
            levels cost more tokens and slightly lower decoding reliability.
        seed: Secret that keys the alias alphabet order and the decoy layout.
            Reuse a seed for byte-identical output across calls (helps provider
            prompt caching).  Defaults to a fresh random value per instance.
        keywords: Extra terms that are always obfuscated (case-insensitive).
        protected: Terms that are never obfuscated.
        rules: Custom regular expressions (strings or compiled patterns) whose
            matches are always obfuscated.  When a pattern has capture groups,
            only the groups are replaced.  Invalid patterns raise ``ValueError``.
        detectors: Replace the detector set of the chosen intensity.  Names are
            listed in :data:`DETECTORS`.
        extra_detectors: Add detectors (e.g. ``("emails", "urls", "ips")``) to the
            intensity's set.
        alias_mode: ``"sequential"`` (default; aliases follow first occurrence) or
            ``"stable"`` (an alias depends only on the key and the value, so the
            same value gets the same alias in every prompt: better for prompt
            caching and for several prompts that share one conversation).
        anchor_position: ``"append"`` (default) or ``"prepend"``.
        anchor_language: Language of the decoding instructions, ``"en"`` or ``"zh"``.
        anchor_style: ``"full"`` (default) or ``"compact"`` (shorter instructions,
            fewer tokens, slightly less guidance for weaker models).
    """

    def __init__(
        self,
        intensity: Union[str, Intensity] = Intensity.MEDIUM,
        *,
        seed: Union[str, int, bytes, None] = None,
        keywords: Iterable[str] = (),
        protected: Iterable[str] = (),
        rules: Iterable[Union[str, Pattern[str]]] = (),
        detectors: Optional[Iterable[str]] = None,
        extra_detectors: Iterable[str] = (),
        alias_mode: str = "sequential",
        anchor_position: str = "append",
        anchor_language: str = "en",
        anchor_style: str = "full",
    ) -> None:
        try:
            self.intensity: Intensity = (
                intensity if isinstance(intensity, Intensity) else Intensity(str(intensity).lower())
            )
        except ValueError:
            raise ValueError(
                f"intensity must be one of 'low', 'medium', 'high'; got {intensity!r}"
            ) from None
        _check_choice("anchor_position", anchor_position, ("append", "prepend"))
        _check_choice("alias_mode", alias_mode, ("sequential", "stable"))
        _check_choice("anchor_language", anchor_language, _anchor.LANGUAGES)
        _check_choice("anchor_style", anchor_style, _anchor.STYLES)
        self.anchor_position = anchor_position
        self.alias_mode = alias_mode
        self.anchor_language = anchor_language
        self.anchor_style = anchor_style

        base = _PROFILES[self.intensity].detectors if detectors is None else frozenset(detectors)
        chosen = base | frozenset(extra_detectors)
        unknown = sorted(chosen - set(DETECTORS))
        if unknown:
            raise ValueError(f"unknown detectors {unknown}; choose from {list(DETECTORS)}")
        self.detectors: FrozenSet[str] = chosen

        if seed is None:
            seed = secrets.token_hex(16)
        elif isinstance(seed, bytes):
            seed = seed.hex()
        self.key: str = str(seed)
        self._symbol_order: Tuple[str, ...] = tuple(
            sorted(_SYMBOL_POOL, key=lambda glyph: _digest(self.key, "glyph", glyph))
        )
        self._keyword_re = self._compile_terms(keywords)
        self._protected_re = self._compile_terms(protected)
        self._rules = self._compile_rules(rules)

    # ------------------------------------------------------------------ public API

    def session(self, texts: Iterable[Optional[str]] = ()) -> ObfuscationSession:
        """Open a shared alias table.

        ``texts`` is the full set of strings that will be encoded; aliases are
        chosen so none of them collides with symbols already present there.
        """
        corpus = [t for t in texts if isinstance(t, str)]
        return ObfuscationSession(self, corpus)

    def obfuscate(self, text: Optional[str]) -> ObfuscationResult:
        """Obfuscate one text and build its anchor.

        Empty or ``None`` input yields an empty result.  Text with nothing to
        obfuscate is returned unchanged with an empty anchor.
        """
        source = text if isinstance(text, str) else ("" if text is None else None)
        if source is None:
            raise TypeError(f"expected str or None, got {type(text).__name__}")
        session = self.session([source])
        encoded = session.encode(source)
        return ObfuscationResult(
            text=encoded,
            anchor=session.anchor,
            mapping=session.mapping,
            intensity=self.intensity,
            session_id=session.session_id,
            anchor_position=self.anchor_position,
            warnings=session.warnings,
            stats=session.stats,
        )

    def protect(self, text: Optional[str]) -> str:
        """Shortcut returning the obfuscated text with its anchor attached."""
        return self.obfuscate(text).prompt

    @staticmethod
    def deobfuscate(text: Optional[str], mapping: Mapping[str, str]) -> str:
        """Replace every alias in ``text`` with the original value from ``mapping``."""
        return _anchor.substitute(text or "", mapping)

    def obfuscate_payload(
        self,
        payload: Mapping[str, Any],
        *,
        style: str = "auto",
        roles: Sequence[str] = ("system", "developer"),
        text_keys: Sequence[str] = ("description", "title"),
    ) -> ObfuscatedPayload:
        """Obfuscate a chat-completion request without changing its shape.

        System/developer messages, the Anthropic ``system`` field, and the
        ``description``/``title`` strings of tool schemas share one alias table;
        the anchor is attached to the system prompt (a system message is
        created when none exists).  Tool names, parameter names, enums and
        types are left untouched so tool calls still bind correctly.

        Args:
            payload: Request body (not mutated).
            style: ``"openai"`` (chat completions), ``"anthropic"``, ``"responses"``
                (OpenAI Responses API: ``instructions`` / ``input`` / flat ``tools``)
                or ``"auto"``.
            roles: Message roles whose content is obfuscated.
            text_keys: Schema keys whose string values are obfuscated.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        _check_choice("style", style, ("auto", "openai", "anthropic", "responses"))
        body: Dict[str, Any] = copy.deepcopy(dict(payload))
        if style == "auto":
            tools = body.get("tools")
            anthropic_tools = isinstance(tools, list) and any(
                isinstance(t, dict) and "input_schema" in t for t in tools
            )
            if "messages" not in body and ("instructions" in body or "input" in body):
                style = "responses"
            else:
                style = "anthropic" if ("system" in body or anthropic_tools) else "openai"

        slots: List[Tuple[Any, Any]] = []  # (container, key) of each string to encode
        messages = body.get("input") if style == "responses" else body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") in roles:
                    _collect_content(message, "content", slots)
        _collect_content(body, "instructions" if style == "responses" else "system", slots)
        _collect_schema_text(body.get("tools"), text_keys, slots)

        session = self.session(c[k] for c, k in slots)
        for container, key in slots:
            container[key] = session.encode(container[key])

        anchor = session.anchor
        if anchor:
            self._attach_anchor(body, anchor, style, roles)
        return ObfuscatedPayload(
            body, session.mapping, anchor, session.session_id, session.warnings, session.stats
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _compile_terms(terms: Iterable[str]) -> Optional[Pattern[str]]:
        cleaned = sorted({t.strip() for t in terms if isinstance(t, str) and t.strip()}, key=len, reverse=True)
        if not cleaned:
            return None
        return re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(t) for t in cleaned) + r")(?!\w)", re.IGNORECASE
        )

    @staticmethod
    def _compile_rules(rules: Iterable[Union[str, Pattern[str]]]) -> Tuple[Pattern[str], ...]:
        compiled: List[Pattern[str]] = []
        for rule in rules:
            if isinstance(rule, str):
                try:
                    compiled.append(re.compile(rule))
                except re.error as exc:
                    raise ValueError(f"invalid rule pattern {rule!r}: {exc}") from None
            elif hasattr(rule, "finditer"):
                compiled.append(rule)
            else:
                raise TypeError(f"rules must be str or compiled patterns, got {type(rule).__name__}")
        return tuple(compiled)

    def _extract(self, text: str, profile: _Profile, warnings: List[str]) -> List[_Span]:
        """Return non-overlapping spans to replace, in text order."""
        blocked: List[Tuple[int, int]] = [m.span() for m in _PROTECTED_RE.finditer(text)]
        if self._protected_re is not None:
            blocked.extend(m.span() for m in self._protected_re.finditer(text))

        # Insertion order breaks ties: when two spans start together the longer wins,
        # and between equals the earlier-added source (keywords, rules, detectors) wins.
        candidates: List[Tuple[int, int, str]] = []  # (start, end, detector)

        def add(pattern: Optional[Pattern[str]], detector: str, groups_only: bool = False) -> None:
            if pattern is None:
                return
            try:
                groups = range(1, pattern.groups + 1) if groups_only else (0,)
                for m in pattern.finditer(text):
                    for g in groups:
                        if m.group(g) is None:
                            continue
                        start, end = m.span(g)
                        if end > start:
                            candidates.append((start, end, detector))
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"pattern skipped: {exc.__class__.__name__}")

        add(self._keyword_re, "keywords")
        for rule in self._rules:
            add(rule, "rules", groups_only=rule.groups > 0)
        for name in DETECTORS:
            if name in self.detectors:
                for pattern, groups_only in _DETECTOR_PATTERNS[name]:
                    add(pattern, name, groups_only)

        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
        spans: List[_Span] = []
        last_end = 0
        for start, end, detector in candidates:
            if start < last_end:
                continue
            if any(start < b_end and end > b_start for b_start, b_end in blocked):
                continue
            value = text[start:end]
            if not self._worth_replacing(text, start, end, value, detector, profile):
                continue
            spans.append(_Span(start, end, value))
            last_end = end
        return spans

    @staticmethod
    def _worth_replacing(
        text: str, start: int, end: int, value: str, detector: str, profile: _Profile
    ) -> bool:
        if detector == "numbers":
            if len(value) == 1 and value.isdigit():
                return False
            line_start = text.rfind("\n", 0, start) + 1
            if not text[line_start:start].strip() and text[end : end + 1] in (".", ")"):
                return False  # list marker such as "1." or "2)"
        elif detector == "acronyms":
            if value in _KEEP_UPPER:
                return False
        elif detector == "words":
            if len(value) < profile.min_word_len or value.lower() in _COMMON_LONG_WORDS:
                return False
        return True

    def _render_anchor(self, session: ObfuscationSession) -> str:
        mapping = session._mapping
        if not mapping:
            return ""
        profile = session._profile
        entries: List[Tuple[str, str]] = list(mapping.items())
        if profile.decoys:
            reserved_before = session._next_index
            entries.extend((session._new_token(), _DECOY_VALUE) for _ in range(profile.decoys))
            session._next_index = reserved_before  # keep real-token numbering stable
        if profile.shuffle_anchor:
            entries.sort(key=lambda e: _digest(session.session_id, e[0]))
        return _anchor.render_anchor(
            session.session_id,
            entries,
            reverse=profile.reverse_values,
            has_decoys=bool(profile.decoys),
            language=self.anchor_language,
            style=self.anchor_style,
        )

    def _attach_anchor(
        self, body: Dict[str, Any], anchor: str, style: str, roles: Sequence[str]
    ) -> None:
        position = self.anchor_position
        if style == "responses":
            instructions = body.get("instructions")
            if isinstance(instructions, str) or instructions is None:
                items = body.get("input")
                target = None
                if instructions is None and isinstance(items, list):
                    target = next(
                        (i for i in items if isinstance(i, dict) and i.get("role") in roles), None
                    )
                if target is None:
                    body["instructions"] = _join_anchor(instructions or "", anchor, position)
                    return
                content = target.get("content")
                if isinstance(content, str) or content is None:
                    target["content"] = _join_anchor(content or "", anchor, position)
                elif isinstance(content, list):
                    part = {"type": "input_text", "text": anchor}
                    if position == "prepend":
                        content.insert(0, part)
                    else:
                        content.append(part)
            return
        if style == "anthropic":
            system = body.get("system")
            if isinstance(system, str) or system is None:
                body["system"] = _join_anchor(system or "", anchor, position)
            elif isinstance(system, list):
                block = {"type": "text", "text": anchor}
                if position == "prepend":
                    system.insert(0, block)
                else:
                    system.append(block)
            return
        messages = body.get("messages")
        if not isinstance(messages, list):
            return
        for message in messages:
            if not (isinstance(message, dict) and message.get("role") in roles):
                continue
            content = message.get("content")
            if isinstance(content, str) or content is None:
                message["content"] = _join_anchor(content or "", anchor, position)
                return
            if isinstance(content, list):
                part = {"type": "text", "text": anchor}
                if position == "prepend":
                    content.insert(0, part)
                else:
                    content.append(part)
                return
        messages.insert(0, {"role": "system", "content": anchor})


def _check_choice(name: str, value: str, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}; got {value!r}")


def _collect_content(container: Any, key: str, slots: List[Tuple[Any, Any]]) -> None:
    """Record every text slot of a message ``content`` / ``system`` value."""
    value = container.get(key) if isinstance(container, dict) else None
    if isinstance(value, str):
        slots.append((container, key))
    elif isinstance(value, list):
        for part in value:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                slots.append((part, "text"))


def _collect_schema_text(
    node: Any, text_keys: Sequence[str], slots: List[Tuple[Any, Any]]
) -> None:
    """Record string values stored under ``text_keys`` anywhere inside tool schemas."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                if key in text_keys:
                    slots.append((node, key))
            else:
                _collect_schema_text(value, text_keys, slots)
    elif isinstance(node, list):
        for item in node:
            _collect_schema_text(item, text_keys, slots)
