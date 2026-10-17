"""Drop-in wrapper for the Anthropic Python SDK (sync and async clients).

Nothing from ``anthropic`` is imported, so the wrapper works with any client that
exposes ``messages.create``::

    from anthropic import Anthropic
    from obfusai import ObfusAIComponent
    from obfusai.adapters.anthropic import ObfusAIAnthropic

    client = ObfusAIAnthropic(Anthropic(), ObfusAIComponent("medium", seed="..."))
    reply = client.messages.create(model="claude-sonnet-5-5", system="...", messages=[...], max_tokens=512)

The ``system`` prompt and tool descriptions are obfuscated, the anchor is added to
the system prompt, and aliases found in text blocks of the reply are decoded.

Streaming is decoded too:

* ``messages.create(stream=True)`` returns the raw event stream with every
  ``text_delta`` decoded (aliases split across events included).
* ``with client.messages.stream(...) as stream`` returns a stream whose
  ``text_stream``, ``get_final_message()``, ``get_final_text()`` and event iteration
  (``for event in stream``: raw, ``text`` and ``content_block_stop`` events) are decoded.
  Use one consumer per stream: the SDK's ``text_stream`` and event iteration share a
  single underlying iterator.  ``current_message_snapshot`` is decoded per content
  block once that block has stopped; while a block is still streaming it holds aliases.

Protected entry points are ``messages.create`` and ``messages.stream``.  Other
``messages`` methods are blocked, except ``count_tokens``, so a prompt can never
leave through an unprotected endpoint by accident; other client attributes are
forwarded untouched and are **not** protected.
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Iterator, Mapping, Optional

from ..core import ObfusAIComponent
from ..guard import StreamRestorer
from ._common import protect_kwargs
from ._streams import (
    AnthropicEventRestorer,
    MessageStreamEventRestorer,
    arestore_events,
    restore_events,
    restore_message_content,
)

__all__ = ["ObfusAIAnthropic"]

_PASSTHROUGH = frozenset({"count_tokens"})


class _EventStream:
    """Decoded view of a raw event stream (sync or async), keeping ``close``/context-manager use."""

    def __init__(self, inner: Any, mapping: Mapping[str, str]) -> None:
        self._inner = inner
        restorer = AnthropicEventRestorer(mapping)
        self._is_async = hasattr(inner, "__aiter__")
        self._events: Any = (
            arestore_events(inner, restorer) if self._is_async else restore_events(inner, restorer)
        )

    def __iter__(self) -> Iterator[Any]:
        return iter(self._events)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._events.__aiter__()

    def __enter__(self) -> "_EventStream":
        if hasattr(self._inner, "__enter__"):
            self._inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc) if hasattr(self._inner, "__exit__") else None

    async def __aenter__(self) -> "_EventStream":
        if hasattr(self._inner, "__aenter__"):
            await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc) if hasattr(self._inner, "__aexit__") else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RestoredMessageStream:
    """Decoded wrapper around the SDK's ``MessageStream`` / ``AsyncMessageStream``."""

    def __init__(self, inner: Any, mapping: Mapping[str, str]) -> None:
        self._inner = inner
        self._mapping = dict(mapping)

    @property
    def text_stream(self) -> Any:
        source = self._inner.text_stream
        restorer = StreamRestorer(self._mapping)
        if hasattr(source, "__aiter__"):
            return self._atext(source, restorer)
        return self._text(source, restorer)

    @staticmethod
    def _text(source: Any, restorer: StreamRestorer) -> Iterator[str]:
        for chunk in source:
            out = restorer.feed(chunk)
            if out:
                yield out
        tail = restorer.flush()
        if tail:
            yield tail

    @staticmethod
    async def _atext(source: Any, restorer: StreamRestorer) -> AsyncIterator[str]:
        async for chunk in source:
            out = restorer.feed(chunk)
            if out:
                yield out
        tail = restorer.flush()
        if tail:
            yield tail

    def get_final_message(self) -> Any:
        result = self._inner.get_final_message()
        if inspect.isawaitable(result):
            return self._await_message(result)
        restore_message_content(getattr(result, "content", None), self._mapping)
        return result

    def get_final_text(self) -> Any:
        result = self._inner.get_final_text()
        if inspect.isawaitable(result):
            return self._await_text(result)
        return ObfusAIComponent.deobfuscate(result, self._mapping)

    async def _await_message(self, awaited: Any) -> Any:
        result = await awaited
        restore_message_content(getattr(result, "content", None), self._mapping)
        return result

    async def _await_text(self, awaited: Any) -> str:
        return ObfusAIComponent.deobfuscate(await awaited, self._mapping)

    def __iter__(self) -> Iterator[Any]:
        return restore_events(self._inner, MessageStreamEventRestorer(self._mapping))

    def __aiter__(self) -> AsyncIterator[Any]:
        return arestore_events(self._inner, MessageStreamEventRestorer(self._mapping))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _StreamManager:
    """Wraps the object returned by ``messages.stream(...)`` (a sync or async context manager)."""

    def __init__(self, inner: Any, mapping: Mapping[str, str]) -> None:
        self._inner = inner
        self._mapping = mapping

    def __enter__(self) -> _RestoredMessageStream:
        return _RestoredMessageStream(self._inner.__enter__(), self._mapping)

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)

    async def __aenter__(self) -> _RestoredMessageStream:
        return _RestoredMessageStream(await self._inner.__aenter__(), self._mapping)

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Messages:
    def __init__(self, owner: "ObfusAIAnthropic") -> None:
        self._owner = owner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        owner = self._owner
        call_kwargs, mapping = protect_kwargs(owner.component, kwargs, "anthropic")
        result = owner.client.messages.create(*args, **call_kwargs)
        if not owner.restore_responses or not mapping:
            return result
        stream = bool(kwargs.get("stream"))
        if inspect.isawaitable(result):
            return self._finish_async(result, mapping, stream)
        return self._finish(result, mapping, stream)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        owner = self._owner
        call_kwargs, mapping = protect_kwargs(owner.component, kwargs, "anthropic")
        manager = owner.client.messages.stream(*args, **call_kwargs)
        if not owner.restore_responses or not mapping:
            return manager
        return _StreamManager(manager, mapping)

    async def _finish_async(self, awaited: Any, mapping: Mapping[str, str], stream: bool) -> Any:
        return self._finish(await awaited, mapping, stream)

    @staticmethod
    def _finish(result: Any, mapping: Mapping[str, str], stream: bool) -> Any:
        if stream:
            return _EventStream(result, mapping)
        restore_message_content(getattr(result, "content", None), mapping)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in _PASSTHROUGH:
            return getattr(self._owner.client.messages, name)
        raise AttributeError(
            f"messages.{name} is not protected by ObfusAIAnthropic; call it on the wrapped client "
            "(`wrapper.client`) if sending plaintext prompts there is acceptable"
        )


class ObfusAIAnthropic:
    """Proxy around an Anthropic client that protects prompts on ``messages.create``/``stream``.

    All other client attributes are forwarded to the wrapped client unprotected.

    Args:
        client: ``Anthropic`` / ``AsyncAnthropic`` (or compatible) instance.
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
        self.messages = _Messages(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
