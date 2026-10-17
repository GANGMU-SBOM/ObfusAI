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

``responses.parse`` (structured output) and the ``responses.stream`` helper are protected too::

    parsed = client.responses.parse(model="gpt-4o", instructions=SYSTEM, input="hi", text_format=Signal)
    parsed.output_text         # decoded (``output_parsed`` is parsed by the SDK from the model's raw text)

    with client.responses.stream(model="gpt-4o", instructions=SYSTEM, input="hi") as stream:
        for event in stream:               # delta / snapshot / done events are decoded
            ...
        final = stream.get_final_response()   # decoded text

Protected entry points are ``chat.completions.create``, ``responses.create``,
``responses.parse`` and ``responses.stream``; other client attributes are forwarded
untouched and are **not** protected.  Structured-output schemas given as dicts (``text``,
``response_format``) have their ``description``/``title`` text obfuscated; schemas given
as pydantic classes (``text_format``) are converted by the SDK and are sent as-is.
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Iterator, Mapping, Optional

from ..core import ObfusAIComponent
from ._common import protect_kwargs
from ._streams import (
    ChatChunkRestorer,
    ResponsesEventRestorer,
    arestore_events,
    restore_chat_completion,
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
        restore_chat_completion(result, mapping)
        return result


class _RestoredResponseStream:
    """Decoded wrapper around the SDK's ``ResponseStream`` / ``AsyncResponseStream``."""

    def __init__(self, inner: Any, mapping: Mapping[str, str]) -> None:
        self._inner = inner
        self._mapping = dict(mapping)

    def __iter__(self) -> Iterator[Any]:
        return restore_events(self._inner, ResponsesEventRestorer(self._mapping))

    def __aiter__(self) -> AsyncIterator[Any]:
        return arestore_events(self._inner, ResponsesEventRestorer(self._mapping))

    def get_final_response(self) -> Any:
        result = self._inner.get_final_response()
        if inspect.isawaitable(result):
            return self._await_final(result)
        restore_response_output(result, self._mapping)
        return result

    async def _await_final(self, awaited: Any) -> Any:
        result = await awaited
        restore_response_output(result, self._mapping)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ResponseStreamManager:
    """Wraps the object returned by ``responses.stream(...)`` (a sync or async context manager)."""

    def __init__(self, inner: Any, mapping: Mapping[str, str]) -> None:
        self._inner = inner
        self._mapping = mapping

    def __enter__(self) -> _RestoredResponseStream:
        return _RestoredResponseStream(self._inner.__enter__(), self._mapping)

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)

    async def __aenter__(self) -> _RestoredResponseStream:
        return _RestoredResponseStream(await self._inner.__aenter__(), self._mapping)

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Responses:
    def __init__(self, owner: "ObfusAIOpenAI") -> None:
        self._owner = owner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("create", args, kwargs)

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        """Structured output; the SDK parses ``output_parsed`` from the model's raw text."""
        return self._call("parse", args, kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        owner = self._owner
        call_kwargs, mapping = protect_kwargs(owner.component, kwargs, "responses")
        manager = owner.client.responses.stream(*args, **call_kwargs)
        if not owner.restore_responses or not mapping:
            return manager
        return _ResponseStreamManager(manager, mapping)

    def _call(self, method: str, args: Any, kwargs: Any) -> Any:
        owner = self._owner
        call_kwargs, mapping = protect_kwargs(owner.component, kwargs, "responses")
        result = getattr(owner.client.responses, method)(*args, **call_kwargs)
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
    """Proxy around an OpenAI client that protects ``chat.completions.create`` and ``responses.create``/``parse``/``stream``.

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
