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
