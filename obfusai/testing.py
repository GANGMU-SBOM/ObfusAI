"""Test doubles for code that sends ObfusAI-protected prompts to an LLM."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

from .anchor import decode_with_anchor

__all__ = ["DecodingMockLLM"]

Messages = Sequence[Mapping[str, Any]]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


class DecodingMockLLM:
    """Deterministic stand-in for a chat model that obeys the anchor.

    It does what a capable model is told to do: expand every alias using the
    anchor, then hand the *decoded* system prompt and the last user message to
    ``responder``.  Use it to unit-test pipelines and to show that an
    obfuscated prompt carries the same information as the plaintext one.

    Args:
        responder: ``(decoded_system_prompt, user_message) -> reply``.
    """

    def __init__(self, responder: Callable[[str, str], str]) -> None:
        self.responder = responder
        self.calls: List[List[Dict[str, Any]]] = []

    def __call__(self, messages: Messages) -> str:
        self.calls.append([dict(m) for m in messages])
        system = decode_with_anchor(
            "\n".join(
                _text(m.get("content")) for m in messages if m.get("role") in ("system", "developer")
            )
        )
        user = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
        return self.responder(system, user)
