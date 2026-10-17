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

Streaming: ``messages.create(stream=True)`` is rejected while response restoration is
on, because aliases split across stream events cannot be decoded safely at this
layer; pass ``restore_responses=False`` and decode with
:class:`obfusai.guard.StreamRestorer` instead.  ``messages.stream(...)`` protects the
request and returns the SDK's stream manager unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from ..core import ObfusAIComponent
from ._common import protect_kwargs, restore_attr

__all__ = ["ObfusAIAnthropic"]


class _Messages:
    def __init__(self, owner: "ObfusAIAnthropic") -> None:
        self._owner = owner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        owner = self._owner
        if kwargs.get("stream") and owner.restore_responses:
            raise NotImplementedError(
                "ObfusAIAnthropic cannot restore streamed events; use restore_responses=False "
                "with obfusai.guard.StreamRestorer"
            )
        call_kwargs, mapping = protect_kwargs(owner.component, kwargs, "anthropic")
        result = owner.client.messages.create(*args, **call_kwargs)
        if not owner.restore_responses or not mapping:
            return result
        if inspect.isawaitable(result):
            return self._finish_async(result, mapping)
        return self._finish(result, mapping)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        call_kwargs, _ = protect_kwargs(self._owner.component, kwargs, "anthropic")
        return self._owner.client.messages.stream(*args, **call_kwargs)

    async def _finish_async(self, awaited: Any, mapping: Mapping[str, str]) -> Any:
        return self._finish(await awaited, mapping)

    @staticmethod
    def _finish(result: Any, mapping: Mapping[str, str]) -> Any:
        for block in getattr(result, "content", None) or []:
            if getattr(block, "type", None) == "text":
                restore_attr(block, "text", mapping)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner.client.messages, name)


class ObfusAIAnthropic:
    """Proxy around an Anthropic client that protects prompts on ``messages.create``.

    All other attributes are forwarded to the wrapped client.

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
