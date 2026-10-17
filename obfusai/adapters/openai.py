"""Drop-in wrapper for the OpenAI Python SDK (sync and async clients).

Nothing from ``openai`` is imported, so the wrapper works with any client that
exposes ``chat.completions.create`` (including Azure and OpenAI-compatible
servers)::

    from openai import OpenAI
    from obfusai import ObfusAIComponent
    from obfusai.adapters.openai import ObfusAIOpenAI

    client = ObfusAIOpenAI(OpenAI(), ObfusAIComponent("medium", seed="..."))
    reply = client.chat.completions.create(model="gpt-4o", messages=[...], tools=[...])

System/developer messages and tool descriptions are obfuscated, the anchor is
attached to the system prompt, and aliases found in the reply are decoded.
Streaming replies (``stream=True``, ``n=1``) are decoded chunk by chunk.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from ..core import ObfusAIComponent
from ..guard import arestore_stream, restore_stream
from ._common import protect_kwargs, restore_attr

__all__ = ["ObfusAIOpenAI"]


def _get_delta(chunk: Any) -> Optional[str]:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    content = getattr(getattr(choices[0], "delta", None), "content", None)
    return content if isinstance(content, str) else None


def _set_delta(chunk: Any, text: str) -> None:
    chunk.choices[0].delta.content = text


class _Completions:
    def __init__(self, owner: "ObfusAIOpenAI") -> None:
        self._owner = owner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        owner = self._owner
        call_kwargs, mapping = protect_kwargs(owner.component, kwargs, "openai")
        result = owner.client.chat.completions.create(*args, **call_kwargs)
        if not owner.restore_responses or not mapping:
            return result
        stream = bool(kwargs.get("stream"))
        if inspect.isawaitable(result):
            return self._finish_async(result, mapping, stream)
        return self._finish(result, mapping, stream)

    async def _finish_async(self, awaited: Any, mapping: Mapping[str, str], stream: bool) -> Any:
        return self._finish(await awaited, mapping, stream)

    @staticmethod
    def _finish(result: Any, mapping: Mapping[str, str], stream: bool) -> Any:
        if stream:
            if hasattr(result, "__aiter__"):
                return arestore_stream(result, mapping, get=_get_delta, set=_set_delta)
            return restore_stream(result, mapping, get=_get_delta, set=_set_delta)
        for choice in getattr(result, "choices", None) or []:
            restore_attr(getattr(choice, "message", None), "content", mapping)
        return result


class _Chat:
    def __init__(self, owner: "ObfusAIOpenAI") -> None:
        self.completions = _Completions(owner)


class ObfusAIOpenAI:
    """Proxy around an OpenAI client that protects prompts on ``chat.completions.create``.

    All other attributes are forwarded to the wrapped client.

    Args:
        client: ``OpenAI`` / ``AsyncOpenAI`` (or compatible) instance.
        component: Engine to use (defaults to ``ObfusAIComponent()``).
        restore_responses: Decode aliases in replies (default ``True``).
    """

    def __init__(
        self,
        client: Any,
        component: Optional[ObfusAIComponent] = None,
        *,
        restore_responses: bool = True,
    ) -> None:
        self.client = client
        self.component = component or ObfusAIComponent()
        self.restore_responses = restore_responses
        self.chat = _Chat(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
