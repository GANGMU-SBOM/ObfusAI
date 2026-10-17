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
Streaming replies (``stream=True``, any ``n``) are decoded chunk by chunk.

The Responses API is supported the same way (``instructions``, ``input`` items with
``system``/``developer`` roles, and flat function tools; sync, async and streaming)::

    reply = client.responses.create(model="gpt-4o", instructions=SYSTEM, input="hi", tools=[...])
    reply.output_text          # decoded

Protected entry points are ``chat.completions.create`` and ``responses.create``;
``parse``/``stream`` helpers are not exposed, and other client attributes are forwarded
untouched and are **not** protected.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from ..core import ObfusAIComponent
from ._common import protect_kwargs, restore_attr
from ._streams import (
    ChatChunkRestorer,
    ResponsesEventRestorer,
    arestore_events,
    restore_events,
    restore_response_output,
)

__all__ = ["ObfusAIOpenAI"]


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
            restorer = ChatChunkRestorer(mapping)
            if hasattr(result, "__aiter__"):
                return arestore_events(result, restorer)
            return restore_events(result, restorer)
        for choice in getattr(result, "choices", None) or []:
            restore_attr(getattr(choice, "message", None), "content", mapping)
        return result


class _Responses:
    def __init__(self, owner: "ObfusAIOpenAI") -> None:
        self._owner = owner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        owner = self._owner
        call_kwargs, mapping = protect_kwargs(owner.component, kwargs, "responses")
        result = owner.client.responses.create(*args, **call_kwargs)
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
            restorer = ResponsesEventRestorer(mapping)
            if hasattr(result, "__aiter__"):
                return arestore_events(result, restorer)
            return restore_events(result, restorer)
        restore_response_output(result, mapping)
        return result


class _Chat:
    def __init__(self, owner: "ObfusAIOpenAI") -> None:
        self.completions = _Completions(owner)


class ObfusAIOpenAI:
    """Proxy around an OpenAI client that protects ``chat.completions.create`` and ``responses.create``.

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
        self.responses = _Responses(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
