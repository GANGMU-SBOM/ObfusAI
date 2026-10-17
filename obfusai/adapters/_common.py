"""Helpers shared by the SDK adapters (duck-typed; no provider SDK is imported)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from ..core import ObfusAIComponent

#: Request fields that carry prompt text; everything else is forwarded untouched.
PROMPT_FIELDS = ("messages", "tools", "system")


def protect_kwargs(
    component: ObfusAIComponent, kwargs: Mapping[str, Any], style: str
) -> Tuple[Dict[str, Any], Mapping[str, str]]:
    """Obfuscate the prompt-bearing fields of ``kwargs``.

    Returns ``(new_kwargs, mapping)``.  Only :data:`PROMPT_FIELDS` are copied, so
    arguments such as ``response_format`` classes or callbacks are never deep-copied.
    """
    subset = {k: kwargs[k] for k in PROMPT_FIELDS if k in kwargs and kwargs[k] is not None}
    if not subset:
        return dict(kwargs), {}
    protected = component.obfuscate_payload(subset, style=style)
    merged = dict(kwargs)
    merged.update(protected.payload)
    return merged, protected.mapping


def restore_attr(obj: Any, name: str, mapping: Mapping[str, str]) -> None:
    """Decode aliases in ``obj.<name>`` in place when it is a string."""
    value: Optional[Any] = getattr(obj, name, None)
    if isinstance(value, str) and value:
        setattr(obj, name, ObfusAIComponent.deobfuscate(value, mapping))
