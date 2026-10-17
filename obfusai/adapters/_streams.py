"""Event-level alias restoration for provider streams (duck-typed, standard library only).

Both restorers decode text *deltas* as they pass, hold back a possibly-split
alias, and, just before the event that closes a text block, emit one synthetic
delta carrying the held-back tail, so the concatenated deltas always equal the
fully decoded text.
"""

from __future__ import annotations

import copy
from typing import Any, AsyncIterable, AsyncIterator, Dict, Hashable, Iterable, Iterator, List, Mapping, Tuple

from ..guard import StreamRestorer
from ._common import restore_attr

__all__ = [
    "AnthropicEventRestorer",
    "ResponsesEventRestorer",
    "arestore_events",
    "restore_events",
    "restore_message_content",
    "restore_response_output",
]


def _clone(obj: Any) -> Any:
    return obj.model_copy() if hasattr(obj, "model_copy") else copy.copy(obj)


def restore_message_content(blocks: Any, mapping: Mapping[str, str]) -> None:
    """Decode the ``text`` of every text block in an Anthropic message ``content`` list."""
    for block in blocks or []:
        if getattr(block, "type", None) == "text":
            restore_attr(block, "text", mapping)


def restore_response_output(response: Any, mapping: Mapping[str, str]) -> None:
    """Decode the ``text`` of every output part of a Responses API ``response`` object."""
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            restore_attr(part, "text", mapping)


class AnthropicEventRestorer:
    """Stateful decoder for ``messages.create(stream=True)`` events."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = dict(mapping)
        self._restorers: Dict[Any, StreamRestorer] = {}
        self._templates: Dict[Any, Any] = {}

    def _drain(self, index: Any) -> List[Any]:
        restorer = self._restorers.pop(index, None)
        template = self._templates.pop(index, None)
        if restorer is None or template is None:
            return []
        tail = restorer.flush()
        if not tail:
            return []
        event = _clone(template)
        event.delta = _clone(template.delta)
        event.delta.text = tail
        return [event]

    def process(self, event: Any) -> List[Any]:
        """Return the events to emit in place of ``event`` (in order)."""
        kind = getattr(event, "type", None)
        out: List[Any] = []
        if kind == "content_block_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", None) == "text_delta" and isinstance(delta.text, str):
                restorer = self._restorers.setdefault(event.index, StreamRestorer(self.mapping))
                delta.text = restorer.feed(delta.text)
                self._templates[event.index] = event
        elif kind == "content_block_stop":
            out.extend(self._drain(getattr(event, "index", None)))
        elif kind == "message_stop":
            for index in list(self._restorers):
                out.extend(self._drain(index))
        out.append(event)
        return out

    def finish(self) -> List[Any]:
        """Events for text still buffered when the stream ended without closing its blocks."""
        out: List[Any] = []
        for index in list(self._restorers):
            out.extend(self._drain(index))
        return out


class ResponsesEventRestorer:
    """Stateful decoder for OpenAI Responses API stream events."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = dict(mapping)
        self._restorers: Dict[Hashable, StreamRestorer] = {}
        self._templates: Dict[Hashable, Any] = {}

    @staticmethod
    def _key(event: Any) -> Tuple[Any, Any, Any]:
        return (
            getattr(event, "item_id", None),
            getattr(event, "output_index", 0),
            getattr(event, "content_index", 0),
        )

    def _drain(self, key: Hashable) -> List[Any]:
        restorer = self._restorers.pop(key, None)
        template = self._templates.pop(key, None)
        if restorer is None or template is None:
            return []
        tail = restorer.flush()
        if not tail:
            return []
        event = _clone(template)
        event.delta = tail
        return [event]

    def process(self, event: Any) -> List[Any]:
        kind = getattr(event, "type", None)
        out: List[Any] = []
        if kind == "response.output_text.delta" and isinstance(getattr(event, "delta", None), str):
            key = self._key(event)
            restorer = self._restorers.setdefault(key, StreamRestorer(self.mapping))
            event.delta = restorer.feed(event.delta)
            self._templates[key] = event
        elif kind == "response.output_text.done":
            out.extend(self._drain(self._key(event)))
            restore_attr(event, "text", self.mapping)
        elif kind == "response.content_part.done":
            restore_attr(getattr(event, "part", None), "text", self.mapping)
        elif kind == "response.output_item.done":
            for part in getattr(getattr(event, "item", None), "content", None) or []:
                restore_attr(part, "text", self.mapping)
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            for key in list(self._restorers):
                out.extend(self._drain(key))
            restore_response_output(getattr(event, "response", None), self.mapping)
        out.append(event)
        return out

    def finish(self) -> List[Any]:
        out: List[Any] = []
        for key in list(self._restorers):
            out.extend(self._drain(key))
        return out


def restore_events(events: Iterable[Any], restorer: Any) -> Iterator[Any]:
    """Yield ``events`` decoded by ``restorer`` (an event restorer from this module)."""
    for event in events:
        yield from restorer.process(event)
    yield from restorer.finish()


async def arestore_events(events: AsyncIterable[Any], restorer: Any) -> AsyncIterator[Any]:
    """Async counterpart of :func:`restore_events`."""
    async for event in events:
        for out in restorer.process(event):
            yield out
    for out in restorer.finish():
        yield out
