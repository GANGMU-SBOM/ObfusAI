"""Event-level alias restoration for provider streams (duck-typed, standard library only).

Both restorers decode text *deltas* as they pass, hold back a possibly-split
alias, and, just before the event that closes a text block, emit one synthetic
delta carrying the held-back tail, so the concatenated deltas always equal the
fully decoded text.
"""

from __future__ import annotations

import copy
from typing import Any, AsyncIterable, AsyncIterator, Dict, Hashable, Iterable, Iterator, List, Mapping, Tuple

from ..guard import StreamRestorer, restore_json_text, restore_json_value
from ._common import restore_attr, restore_json_attr

__all__ = [
    "AnthropicEventRestorer",
    "ChatChunkRestorer",
    "MessageStreamEventRestorer",
    "ResponsesEventRestorer",
    "arestore_events",
    "restore_events",
    "restore_chat_completion",
    "restore_message_content",
    "restore_response_output",
]


def _clone(obj: Any) -> Any:
    return obj.model_copy() if hasattr(obj, "model_copy") else copy.copy(obj)


def restore_tool_input(block: Any, mapping: Mapping[str, str]) -> None:
    """Decode aliases in the ``input`` of an Anthropic ``tool_use`` block (a parsed JSON object)."""
    value = getattr(block, "input", None)
    if isinstance(value, (dict, list)):
        block.input = restore_json_value(value, mapping)


def restore_message_content(blocks: Any, mapping: Mapping[str, str]) -> None:
    """Decode text blocks and ``tool_use`` inputs in an Anthropic message ``content`` list."""
    for block in blocks or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            restore_attr(block, "text", mapping)
        elif kind in ("tool_use", "server_tool_use"):
            restore_tool_input(block, mapping)


def restore_response_output(response: Any, mapping: Mapping[str, str]) -> None:
    """Decode output text and ``function_call`` arguments of a Responses API ``response`` object."""
    for item in getattr(response, "output", None) or []:
        restore_response_item(item, mapping)


def restore_chat_completion(result: Any, mapping: Mapping[str, str]) -> None:
    """Decode a Chat Completions response: message text and tool-call arguments."""
    for choice in getattr(result, "choices", None) or []:
        message = getattr(choice, "message", None)
        restore_attr(message, "content", mapping)
        for tool_call in getattr(message, "tool_calls", None) or []:
            restore_json_attr(getattr(tool_call, "function", None), "arguments", mapping)
        restore_json_attr(getattr(message, "function_call", None), "arguments", mapping)  # legacy


def restore_response_item(item: Any, mapping: Mapping[str, str]) -> None:
    """Decode one Responses API output item (message text parts, function-call arguments)."""
    for part in getattr(item, "content", None) or []:
        restore_attr(part, "text", mapping)
    restore_json_attr(item, "arguments", mapping)


class AnthropicEventRestorer:
    """Stateful decoder for ``messages.create(stream=True)`` events."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = dict(mapping)
        self._restorers: Dict[Any, StreamRestorer] = {}
        self._templates: Dict[Any, Any] = {}
        self._json_restorers: Dict[Any, StreamRestorer] = {}
        self._json_templates: Dict[Any, Any] = {}

    def _drain(self, index: Any) -> List[Any]:
        out: List[Any] = []
        for restorers, templates, attr in (
            (self._restorers, self._templates, "text"),
            (self._json_restorers, self._json_templates, "partial_json"),
        ):
            restorer = restorers.pop(index, None)
            template = templates.pop(index, None)
            if restorer is None or template is None:
                continue
            tail = restorer.flush()
            if tail:
                event = _clone(template)
                event.delta = _clone(template.delta)
                setattr(event.delta, attr, tail)
                out.append(event)
        return out

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
            elif getattr(delta, "type", None) == "input_json_delta" and isinstance(delta.partial_json, str):
                restorer = self._json_restorers.setdefault(
                    event.index, StreamRestorer(self.mapping, json_strings=True)
                )
                delta.partial_json = restorer.feed(delta.partial_json)
                self._json_templates[event.index] = event
        elif kind == "content_block_stop":
            out.extend(self._drain(getattr(event, "index", None)))
        elif kind == "message_stop":
            for index in sorted(set(self._restorers) | set(self._json_restorers), key=str):
                out.extend(self._drain(index))
        out.append(event)
        return out

    def finish(self) -> List[Any]:
        """Events for text still buffered when the stream ended without closing its blocks."""
        out: List[Any] = []
        for index in sorted(set(self._restorers) | set(self._json_restorers), key=str):
            out.extend(self._drain(index))
        return out


class MessageStreamEventRestorer:
    """Stateful decoder for the events yielded by the SDK's ``messages.stream()`` helper.

    That stream interleaves raw events with derived ones.  After a raw text
    ``content_block_delta`` the SDK emits a ``text`` event (``text`` = the delta,
    ``snapshot`` = the accumulated text); here both carry decoded text, and the
    ``snapshot`` is the decoded text emitted so far.  Just before a block's
    ``content_block_stop`` a synthetic raw delta plus ``text`` event carry any
    held-back tail.  ``content_block_stop.content_block`` and the ``message`` of
    ``message_stop`` are decoded in place (they alias the SDK's own snapshot, so
    ``get_final_message()`` agrees).  Other event types pass through unchanged.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = dict(mapping)
        self._restorers: Dict[Any, StreamRestorer] = {}
        self._decoded: Dict[Any, str] = {}
        self._delta_templates: Dict[Any, Any] = {}
        self._text_templates: Dict[Any, Any] = {}
        self._current: Any = None  # (index, decoded piece) of the raw delta just seen
        self._json_restorers: Dict[Any, StreamRestorer] = {}
        self._json_delta_templates: Dict[Any, Any] = {}
        self._json_event_templates: Dict[Any, Any] = {}
        self._json_current: Any = None

    def _drain(self, index: Any) -> List[Any]:
        out: List[Any] = []
        restorer = self._restorers.pop(index, None)
        delta_tpl = self._delta_templates.pop(index, None)
        text_tpl = self._text_templates.pop(index, None)
        decoded = self._decoded.pop(index, "")
        tail = restorer.flush() if restorer is not None and delta_tpl is not None else ""
        if tail:
            raw = _clone(delta_tpl)
            raw.delta = _clone(delta_tpl.delta)
            raw.delta.text = tail
            out.append(raw)
            if text_tpl is not None:
                derived = _clone(text_tpl)
                derived.text = tail
                derived.snapshot = decoded + tail
                out.append(derived)
        json_restorer = self._json_restorers.pop(index, None)
        json_delta_tpl = self._json_delta_templates.pop(index, None)
        json_event_tpl = self._json_event_templates.pop(index, None)
        json_tail = json_restorer.flush() if json_restorer is not None and json_delta_tpl is not None else ""
        if json_tail:
            raw = _clone(json_delta_tpl)
            raw.delta = _clone(json_delta_tpl.delta)
            raw.delta.partial_json = json_tail
            out.append(raw)
            if json_event_tpl is not None:
                derived = _clone(json_event_tpl)
                derived.partial_json = json_tail
                out.append(derived)
        return out

    def process(self, event: Any) -> List[Any]:
        kind = getattr(event, "type", None)
        out: List[Any] = []
        if kind == "content_block_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", None) == "text_delta" and isinstance(delta.text, str):
                restorer = self._restorers.setdefault(event.index, StreamRestorer(self.mapping))
                piece = restorer.feed(delta.text)
                self._decoded[event.index] = self._decoded.get(event.index, "") + piece
                delta.text = piece
                self._delta_templates[event.index] = event
                self._current = (event.index, piece)
            elif getattr(delta, "type", None) == "input_json_delta" and isinstance(delta.partial_json, str):
                restorer = self._json_restorers.setdefault(
                    event.index, StreamRestorer(self.mapping, json_strings=True)
                )
                piece = restorer.feed(delta.partial_json)
                delta.partial_json = piece
                self._json_delta_templates[event.index] = event
                self._json_current = (event.index, piece)
        elif kind == "input_json":
            if self._json_current is not None:
                index, piece = self._json_current
                event.partial_json = piece
                self._json_event_templates[index] = event
            else:
                restore_json_attr(event, "partial_json", self.mapping)
            event.snapshot = restore_json_value(getattr(event, "snapshot", None), self.mapping)
            self._json_current = None
        elif kind == "text":
            if self._current is not None:
                index, piece = self._current
                event.text = piece
                event.snapshot = self._decoded.get(index, "")
                self._text_templates[index] = event
            else:  # no raw delta in front of it: decode what we can
                restore_attr(event, "text", self.mapping)
                restore_attr(event, "snapshot", self.mapping)
            self._current = None
        elif kind == "content_block_stop":
            out.extend(self._drain(getattr(event, "index", None)))
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) == "text":
                restore_attr(block, "text", self.mapping)
            elif getattr(block, "type", None) in ("tool_use", "server_tool_use"):
                restore_tool_input(block, self.mapping)
        elif kind == "message_stop":
            for index in sorted(set(self._restorers) | set(self._json_restorers), key=str):
                out.extend(self._drain(index))
            restore_message_content(getattr(getattr(event, "message", None), "content", None), self.mapping)
        out.append(event)
        return out

    def finish(self) -> List[Any]:
        out: List[Any] = []
        for index in sorted(set(self._restorers) | set(self._json_restorers), key=str):
            out.extend(self._drain(index))
        return out


class ChatChunkRestorer:
    """Stateful decoder for Chat Completions stream chunks, one alias buffer per choice.

    Works for any ``n``: deltas are decoded per ``choice.index``.  Text still held
    back when a choice finishes (``finish_reason`` set) is emitted as a synthetic
    chunk carrying only that choice, placed right before the chunk that finishes it;
    leftovers at the end of the stream are emitted last.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = dict(mapping)
        self._restorers: Dict[int, StreamRestorer] = {}
        self._templates: Dict[int, Tuple[Any, Any]] = {}  # index -> (chunk, choice)
        # tool-call argument deltas: (choice index, tool index) -> restorer / (chunk, choice, tool call)
        self._tool_restorers: Dict[Tuple[int, int], StreamRestorer] = {}
        self._tool_templates: Dict[Tuple[int, int], Tuple[Any, Any, Any]] = {}

    @staticmethod
    def _synthetic(chunk: Any, choice: Any) -> Any:
        new_choice = _clone(choice)
        new_choice.delta = _clone(choice.delta)
        new_choice.finish_reason = None
        synthetic = _clone(chunk)
        synthetic.choices = [new_choice]
        if getattr(synthetic, "usage", None) is not None:
            synthetic.usage = None
        return synthetic

    def _drain(self, index: int) -> List[Any]:
        out: List[Any] = []
        restorer = self._restorers.pop(index, None)
        template = self._templates.pop(index, None)
        tail = restorer.flush() if restorer is not None and template is not None else ""
        if tail:
            chunk, choice = template
            synthetic = self._synthetic(chunk, choice)
            synthetic.choices[0].delta.content = tail
            out.append(synthetic)
        for key in sorted(k for k in self._tool_restorers if k[0] == index):
            restorer = self._tool_restorers.pop(key)
            chunk, choice, tool_call = self._tool_templates.pop(key)
            tool_tail = restorer.flush()
            if not tool_tail:
                continue
            synthetic = self._synthetic(chunk, choice)
            new_call = _clone(tool_call)
            new_call.function = _clone(tool_call.function)
            new_call.function.arguments = tool_tail
            synthetic.choices[0].delta.content = None
            synthetic.choices[0].delta.tool_calls = [new_call]
            out.append(synthetic)
        return out

    def _indexes(self) -> List[int]:
        return sorted(set(self._restorers) | {k[0] for k in self._tool_restorers})

    def process(self, chunk: Any) -> List[Any]:
        out: List[Any] = []
        for choice in getattr(chunk, "choices", None) or []:
            index = getattr(choice, "index", 0)
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                restorer = self._restorers.setdefault(index, StreamRestorer(self.mapping))
                delta.content = restorer.feed(content)
                self._templates[index] = (chunk, choice)
            for tool_call in getattr(delta, "tool_calls", None) or []:
                function = getattr(tool_call, "function", None)
                arguments = getattr(function, "arguments", None)
                if isinstance(arguments, str):
                    key = (index, getattr(tool_call, "index", 0) or 0)
                    restorer = self._tool_restorers.setdefault(
                        key, StreamRestorer(self.mapping, json_strings=True)
                    )
                    function.arguments = restorer.feed(arguments)
                    self._tool_templates[key] = (chunk, choice, tool_call)
            if getattr(choice, "finish_reason", None) is not None:
                out.extend(self._drain(index))
        out.append(chunk)
        return out

    def finish(self) -> List[Any]:
        out: List[Any] = []
        for index in self._indexes():
            out.extend(self._drain(index))
        return out


class ResponsesEventRestorer:
    """Stateful decoder for OpenAI Responses API stream events.

    Handles the raw events of ``responses.create(stream=True)`` and the events of the
    ``responses.stream()`` helper, whose text deltas also carry a cumulative
    ``snapshot`` (decoded here as the text emitted so far).
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = dict(mapping)
        self._restorers: Dict[Hashable, StreamRestorer] = {}
        self._templates: Dict[Hashable, Any] = {}
        self._decoded: Dict[Hashable, str] = {}
        self._arg_restorers: Dict[Hashable, StreamRestorer] = {}
        self._arg_templates: Dict[Hashable, Any] = {}
        self._arg_decoded: Dict[Hashable, str] = {}

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
        decoded = self._decoded.pop(key, "")
        if restorer is None or template is None:
            return []
        tail = restorer.flush()
        if not tail:
            return []
        event = _clone(template)
        event.delta = tail
        if hasattr(template, "snapshot"):
            event.snapshot = decoded + tail
        return [event]

    @staticmethod
    def _arg_key(event: Any) -> Tuple[Any, Any]:
        return (getattr(event, "item_id", None), getattr(event, "output_index", 0))

    def _drain_args(self, key: Hashable) -> List[Any]:
        restorer = self._arg_restorers.pop(key, None)
        template = self._arg_templates.pop(key, None)
        decoded = self._arg_decoded.pop(key, "")
        if restorer is None or template is None:
            return []
        tail = restorer.flush()
        if not tail:
            return []
        event = _clone(template)
        event.delta = tail
        if hasattr(template, "snapshot"):
            event.snapshot = decoded + tail
        return [event]

    def process(self, event: Any) -> List[Any]:
        kind = getattr(event, "type", None)
        out: List[Any] = []
        if kind == "response.function_call_arguments.delta" and isinstance(getattr(event, "delta", None), str):
            key = self._arg_key(event)
            restorer = self._arg_restorers.setdefault(key, StreamRestorer(self.mapping, json_strings=True))
            piece = restorer.feed(event.delta)
            self._arg_decoded[key] = self._arg_decoded.get(key, "") + piece
            event.delta = piece
            if hasattr(event, "snapshot"):
                event.snapshot = self._arg_decoded[key]
            self._arg_templates[key] = event
        elif kind == "response.function_call_arguments.done":
            out.extend(self._drain_args(self._arg_key(event)))
            restore_json_attr(event, "arguments", self.mapping)
        elif kind == "response.output_text.delta" and isinstance(getattr(event, "delta", None), str):
            key = self._key(event)
            restorer = self._restorers.setdefault(key, StreamRestorer(self.mapping))
            piece = restorer.feed(event.delta)
            self._decoded[key] = self._decoded.get(key, "") + piece
            event.delta = piece
            if hasattr(event, "snapshot"):
                event.snapshot = self._decoded[key]
            self._templates[key] = event
        elif kind == "response.output_text.done":
            out.extend(self._drain(self._key(event)))
            restore_attr(event, "text", self.mapping)
        elif kind == "response.content_part.done":
            restore_attr(getattr(event, "part", None), "text", self.mapping)
        elif kind == "response.output_item.done":
            restore_response_item(getattr(event, "item", None), self.mapping)
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            for key in list(self._restorers):
                out.extend(self._drain(key))
            for key in list(self._arg_restorers):
                out.extend(self._drain_args(key))
            restore_response_output(getattr(event, "response", None), self.mapping)
        out.append(event)
        return out

    def finish(self) -> List[Any]:
        out: List[Any] = []
        for key in list(self._restorers):
            out.extend(self._drain(key))
        for key in list(self._arg_restorers):
            out.extend(self._drain_args(key))
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
