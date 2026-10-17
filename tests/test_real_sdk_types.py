"""Run the stream restorers on the real SDK event models when the SDKs are installed."""

from __future__ import annotations

import pytest

from obfusai.adapters._streams import (
    AnthropicEventRestorer,
    ResponsesEventRestorer,
    restore_events,
)

MAPPING = {"α": "alpha_x"}


def test_anthropic_real_event_types() -> None:
    types = pytest.importorskip("anthropic.types")
    chunks = ["see ", "α"]
    events = [
        types.RawContentBlockDeltaEvent(type="content_block_delta", index=0, delta=types.TextDelta(type="text_delta", text=t))
        for t in chunks
    ] + [types.RawContentBlockStopEvent(type="content_block_stop", index=0)]
    out = list(restore_events(events, AnthropicEventRestorer(MAPPING)))
    assert "".join(e.delta.text for e in out if e.type == "content_block_delta") == "see alpha_x"
    assert out[-1].type == "content_block_stop"


def test_responses_real_event_types() -> None:
    types = pytest.importorskip("openai.types.responses")
    ids = dict(item_id="i", output_index=0, content_index=0, sequence_number=0, logprobs=[])
    events = [types.ResponseTextDeltaEvent(type="response.output_text.delta", delta=t, **ids) for t in ["see ", "α"]]
    events.append(types.ResponseTextDoneEvent(type="response.output_text.done", text="see α", **ids))
    out = list(restore_events(events, ResponsesEventRestorer(MAPPING)))
    assert "".join(e.delta for e in out if e.type == "response.output_text.delta") == "see alpha_x"
    assert out[-1].text == "see alpha_x"


def test_anthropic_message_stream_helper_iteration_is_decoded() -> None:
    """Drive the real SDK MessageStream (event building, snapshots) through our wrapper."""
    anthropic = pytest.importorskip("anthropic")
    from anthropic.lib.streaming import MessageStream
    from anthropic.types import (
        Message, MessageDeltaUsage, RawContentBlockDeltaEvent, RawContentBlockStartEvent,
        RawContentBlockStopEvent, RawMessageDeltaEvent, RawMessageStartEvent, RawMessageStopEvent,
        TextBlock, TextDelta, Usage,
    )
    from anthropic.types.raw_message_delta_event import Delta

    from obfusai.adapters.anthropic import _RestoredMessageStream

    message = Message(id="m", type="message", role="assistant", model="c", content=[], stop_reason=None,
                      stop_sequence=None, usage=Usage(input_tokens=1, output_tokens=1))
    events = [RawMessageStartEvent(type="message_start", message=message),
              RawContentBlockStartEvent(type="content_block_start", index=0, content_block=TextBlock(type="text", text="", citations=None))]
    events += [RawContentBlockDeltaEvent(type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text=t))
               for t in ["see ", "α", " ok ", "α"]]
    events += [RawContentBlockStopEvent(type="content_block_stop", index=0),
               RawMessageDeltaEvent(type="message_delta", delta=Delta(stop_reason="end_turn", stop_sequence=None),
                                    usage=MessageDeltaUsage(output_tokens=3)),
               RawMessageStopEvent(type="message_stop")]

    class Raw:
        def __iter__(self):
            return iter(events)

    stream = MessageStream(Raw(), output_format=anthropic.NOT_GIVEN)
    wrapped = _RestoredMessageStream(stream, MAPPING)
    out = list(wrapped)
    texts = [e.text for e in out if e.type == "text"]
    assert "".join(texts) == "see alpha_x ok alpha_x"
    assert [e for e in out if e.type == "text"][-1].snapshot == "see alpha_x ok alpha_x"
    assert "".join(e.delta.text for e in out if e.type == "content_block_delta") == "see alpha_x ok alpha_x"
    assert wrapped.get_final_text() == "see alpha_x ok alpha_x"
    assert stream.get_final_message().content[0].text == "see alpha_x ok alpha_x"


def test_chat_completion_chunks_real_types_two_choices() -> None:
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    from obfusai.adapters._streams import ChatChunkRestorer

    def chunk(index, content, finish=None):
        return ChatCompletionChunk(
            id="c", created=0, model="m", object="chat.completion.chunk",
            choices=[Choice(index=index, delta=ChoiceDelta(content=content), finish_reason=finish)],
        )

    events = [chunk(0, "see "), chunk(1, "also "), chunk(0, "α"), chunk(1, "α"), chunk(0, None, "stop"), chunk(1, None, "stop")]
    out = list(restore_events(events, ChatChunkRestorer(MAPPING)))
    text = {0: "", 1: ""}
    for c in out:
        text[c.choices[0].index] += c.choices[0].delta.content or ""
    assert text == {0: "see alpha_x", 1: "also alpha_x"}
    assert out[-1].choices[0].finish_reason == "stop"


def test_responses_stream_helper_events_with_snapshot_real_types() -> None:
    pytest.importorskip("openai")
    from openai.lib.streaming.responses._events import ResponseTextDeltaEvent, ResponseTextDoneEvent

    ids = dict(item_id="i", output_index=0, content_index=0, sequence_number=0, logprobs=[])
    acc = ""
    events = []
    for t in ["see ", "α", " ok"]:
        acc += t
        events.append(ResponseTextDeltaEvent(type="response.output_text.delta", delta=t, snapshot=acc, **ids))
    events.append(ResponseTextDoneEvent(type="response.output_text.done", text=acc, parsed=None, **ids))
    out = list(restore_events(events, ResponsesEventRestorer(MAPPING)))
    deltas = [e for e in out if e.type == "response.output_text.delta"]
    assert "".join(e.delta for e in deltas) == "see alpha_x ok"
    assert deltas[-1].snapshot == "see alpha_x ok"
    assert out[-1].text == "see alpha_x ok"
