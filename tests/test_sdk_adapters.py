from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any, List

import pytest

from obfusai import ObfusAIComponent
from obfusai.adapters.anthropic import ObfusAIAnthropic
from obfusai.adapters.openai import ObfusAIOpenAI
from obfusai.anchor import decode_with_anchor

SYSTEM = "Rule: BUY when edge_bps > 35. Codename project_falcon."
TOOLS = [{"type": "function", "function": {"name": "get_signal", "description": "Reads edge_bps."}}]


def _echo_alias_reply(messages: List[dict]) -> str:
    """A reply that (wrongly) echoes an alias, to prove restoration."""
    system = messages[0]["content"]
    alias = system.split(" > ")[0].split()[-1]
    return f"The metric {alias} matters."


class FakeOpenAI:
    def __init__(self, stream_size: int = 0, async_mode: bool = False) -> None:
        self.calls: List[dict] = []
        self.stream_size = stream_size
        self.base_url = "https://fake"
        create = self._create
        if async_mode:
            sync_create = self._create

            async def create(*a: Any, **k: Any) -> Any:  # type: ignore[misc]
                return sync_create(*a, **k)

        self.chat = NS(completions=NS(create=create))

    def _create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = _echo_alias_reply(kwargs["messages"])
        if kwargs.get("stream"):
            n = self.stream_size
            parts = [text[i : i + n] for i in range(0, len(text), n)]
            chunks = [NS(choices=[NS(delta=NS(content=p))]) for p in parts] + [NS(choices=[NS(delta=NS(content=None))])]
            return iter(chunks)
        return NS(choices=[NS(message=NS(content=text, role="assistant"))])


def _kwargs() -> dict:
    return dict(
        model="gpt-4o",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}],
        tools=TOOLS,
        temperature=0.2,
    )


def test_openai_wrapper_protects_request_and_restores_reply() -> None:
    fake = FakeOpenAI()
    client = ObfusAIOpenAI(fake, ObfusAIComponent("low", seed="s"))
    kwargs = _kwargs()
    reply = client.chat.completions.create(**kwargs)
    sent = fake.calls[0]
    assert "edge_bps" not in sent["messages"][0]["content"].split("<obfusai-context")[0]
    assert decode_with_anchor(sent["messages"][0]["content"]) == SYSTEM
    assert "edge_bps" not in sent["tools"][0]["function"]["description"]
    assert sent["model"] == "gpt-4o" and sent["temperature"] == 0.2
    assert kwargs["messages"][0]["content"] == SYSTEM  # caller's data untouched
    assert reply.choices[0].message.content == "The metric edge_bps matters."
    assert client.base_url == "https://fake"  # attribute passthrough


def test_openai_wrapper_without_restoration_or_prompts() -> None:
    fake = FakeOpenAI()
    client = ObfusAIOpenAI(fake, ObfusAIComponent("low", seed="s"), restore_responses=False)
    assert "matters" in client.chat.completions.create(**_kwargs()).choices[0].message.content
    assert client.chat.completions.create(**_kwargs()).choices[0].message.content != "The metric edge_bps matters."
    plain = ObfusAIOpenAI(fake).chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert plain.choices[0].message.content


@pytest.mark.parametrize("size", [1, 3, 7])
def test_openai_streaming_restore(size: int) -> None:
    client = ObfusAIOpenAI(FakeOpenAI(stream_size=size), ObfusAIComponent("high", seed="s"))
    chunks = list(client.chat.completions.create(stream=True, **_kwargs()))
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == "The metric edge_bps matters."


def test_openai_async_client_and_stream() -> None:
    async def run() -> str:
        client = ObfusAIOpenAI(FakeOpenAI(async_mode=True), ObfusAIComponent("low", seed="s"))
        reply = await client.chat.completions.create(**_kwargs())
        return reply.choices[0].message.content

    assert asyncio.run(run()) == "The metric edge_bps matters."


# ----------------------------------------------------------------- Anthropic
REPLY = "The metric {alias} matters."


def _alias_of(system: str) -> str:
    return system.split(" > ")[0].split()[-1]


def _anthropic_events(text: str, size: int, blocks: int = 1) -> list:
    events = [NS(type="message_start")]
    for index in range(blocks):
        events.append(NS(type="content_block_start", index=index))
        for i in range(0, len(text), size):
            events.append(NS(type="content_block_delta", index=index, delta=NS(type="text_delta", text=text[i : i + size])))
        events.append(NS(type="content_block_delta", index=index, delta=NS(type="input_json_delta", partial_json="{}")))
        events.append(NS(type="content_block_stop", index=index))
    events.append(NS(type="message_stop"))
    return events


class FakeAnthropic:
    def __init__(self, chunk: int = 3, async_mode: bool = False) -> None:
        self.calls: List[dict] = []
        self.chunk = chunk
        self.messages = NS(create=self._create, stream=self._stream, count_tokens=lambda **k: 7, batches="batches")
        if async_mode:
            sync_create = self._create

            async def create(**k: Any) -> Any:
                return sync_create(**k)

            self.messages.create = create

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = REPLY.format(alias=_alias_of(kwargs["system"]))
        if kwargs.get("stream"):
            events = _anthropic_events(text, self.chunk, blocks=2)
            if kwargs.get("_async_stream"):
                return _AsyncIter(events)
            return iter(events)
        return NS(content=[NS(type="text", text=text), NS(type="tool_use", name="x")])

    def _stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = REPLY.format(alias=_alias_of(kwargs["system"]))
        return FakeStreamManager(text, self.chunk)


class _AsyncIter:
    def __init__(self, items: list) -> None:
        self.items = items

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for item in self.items:
                yield item
        return gen()


class FakeStreamManager:
    def __init__(self, text: str, chunk: int) -> None:
        self.text, self.chunk = text, chunk
        self.exited = False

    def __enter__(self) -> Any:
        text, chunk = self.text, self.chunk
        return NS(
            text_stream=iter([text[i : i + chunk] for i in range(0, len(text), chunk)]),
            get_final_message=lambda: NS(content=[NS(type="text", text=text)]),
            get_final_text=lambda: text,
            current_message_snapshot="snapshot",
        )

    def __exit__(self, *exc: Any) -> None:
        self.exited = True


def _anth(client: Any = None, **opts: Any) -> ObfusAIAnthropic:
    return ObfusAIAnthropic(client or FakeAnthropic(), ObfusAIComponent("low", seed="s"), **opts)


def _call(client: ObfusAIAnthropic, **extra: Any) -> Any:
    return client.messages.create(model="claude", max_tokens=10, system=SYSTEM,
                                  messages=[{"role": "user", "content": "hi"}], **extra)


def test_anthropic_wrapper() -> None:
    fake = FakeAnthropic()
    client = _anth(fake)
    reply = _call(client, tools=[{"name": "t", "description": "Reads edge_bps.", "input_schema": {"type": "object"}}])
    sent = fake.calls[0]
    assert decode_with_anchor(sent["system"]) == SYSTEM
    assert "edge_bps" not in sent["tools"][0]["description"]
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert reply.content[0].text == "The metric edge_bps matters."
    assert client.messages.count_tokens() == 7


def test_anthropic_blocks_unprotected_endpoints() -> None:
    with pytest.raises(AttributeError, match="not protected"):
        _anth().messages.batches


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 100])
def test_anthropic_create_stream_restores_each_block(chunk: int) -> None:
    client = _anth(FakeAnthropic(chunk=chunk))
    events = list(_call(client, stream=True))
    for index in (0, 1):
        text = "".join(
            e.delta.text for e in events
            if e.type == "content_block_delta" and e.index == index and e.delta.type == "text_delta"
        )
        assert text == "The metric edge_bps matters."
    assert events[0].type == "message_start" and events[-1].type == "message_stop"
    stops = [i for i, e in enumerate(events) if e.type == "content_block_stop"]
    assert len(stops) == 2  # tail is flushed *before* the closing event of its block
    assert all(e.delta.type != "input_json_delta" or e.delta.partial_json == "{}" for e in events if e.type == "content_block_delta")


def test_anthropic_stream_without_close_flushes_at_end() -> None:
    from obfusai.adapters._streams import AnthropicEventRestorer, restore_events

    mapping = {"α": "alpha_x"}
    events = [NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="see α"))]
    out = list(restore_events(events, AnthropicEventRestorer(mapping)))
    assert "".join(e.delta.text for e in out) == "see alpha_x"


def test_anthropic_message_stream_manager() -> None:
    fake = FakeAnthropic(chunk=2)
    client = _anth(fake)
    manager = client.messages.stream(model="claude", max_tokens=10, system=SYSTEM, messages=[])
    assert "<obfusai-context" in fake.calls[-1]["system"]
    with manager as stream:
        assert "".join(stream.text_stream) == "The metric edge_bps matters."
        assert stream.get_final_message().content[0].text == "The metric edge_bps matters."
        assert stream.get_final_text() == "The metric edge_bps matters."
        assert stream.current_message_snapshot == "snapshot"
        with pytest.raises(NotImplementedError):
            iter(stream)
    assert manager._inner.exited


def test_anthropic_without_restoration_passes_streams_through() -> None:
    client = _anth(restore_responses=False)
    events = list(_call(client, stream=True))
    text = "".join(e.delta.text for e in events if e.type == "content_block_delta" and e.delta.type == "text_delta")
    assert "edge_bps" not in text


def test_anthropic_async_create_and_stream() -> None:
    async def run() -> tuple:
        fake = FakeAnthropic(chunk=2, async_mode=True)
        client = _anth(fake)
        reply = await _call(client)
        streamed = await _call(client, stream=True, _async_stream=True)
        text = ""
        async for event in streamed:
            if event.type == "content_block_delta" and event.index == 0 and event.delta.type == "text_delta":
                text += event.delta.text
        return reply.content[0].text, text

    assert asyncio.run(run()) == ("The metric edge_bps matters.",) * 2


# ----------------------------------------------------------------- OpenAI Responses API
class FakeResponse:
    def __init__(self, text: str) -> None:
        self.output = [NS(type="message", content=[NS(type="output_text", text=text)])]

    @property
    def output_text(self) -> str:
        return "".join(p.text for item in self.output for p in item.content)


def _responses_events(text: str, size: int) -> list:
    ids = dict(item_id="msg_1", output_index=0, content_index=0)
    events = [NS(type="response.created")]
    for i in range(0, len(text), size):
        events.append(NS(type="response.output_text.delta", delta=text[i : i + size], **ids))
    events += [
        NS(type="response.output_text.done", text=text, **ids),
        NS(type="response.content_part.done", part=NS(text=text), **ids),
        NS(type="response.output_item.done", item=NS(content=[NS(text=text)])),
        NS(type="response.completed", response=FakeResponse(text)),
    ]
    return events


class FakeResponsesClient:
    def __init__(self, chunk: int = 3, async_mode: bool = False) -> None:
        self.calls: List[dict] = []
        self.chunk = chunk
        self.responses = NS(create=self._create)
        if async_mode:
            sync_create = self._create

            async def create(**k: Any) -> Any:
                return sync_create(**k)

            self.responses.create = create

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        sent = kwargs.get("instructions") or kwargs["input"][0]["content"]
        sent = sent if isinstance(sent, str) else sent[0]["text"]
        text = REPLY.format(alias=_alias_of(sent))
        if kwargs.get("stream"):
            events = _responses_events(text, self.chunk)
            return _AsyncIter(events) if kwargs.get("_async_stream") else iter(events)
        return FakeResponse(text)


def _resp(client: Any = None, **opts: Any) -> ObfusAIOpenAI:
    return ObfusAIOpenAI(client or FakeResponsesClient(), ObfusAIComponent("low", seed="s"), **opts)


def test_responses_instructions_and_tools() -> None:
    fake = FakeResponsesClient()
    kwargs = dict(model="gpt-4o", instructions=SYSTEM, input="hi", temperature=0.1,
                  tools=[{"type": "function", "name": "get_signal", "description": "Reads edge_bps.",
                          "parameters": {"type": "object", "properties": {"k": {"type": "string", "description": "edge_bps"}}}}])
    reply = _resp(fake).responses.create(**kwargs)
    sent = fake.calls[0]
    assert decode_with_anchor(sent["instructions"]) == SYSTEM
    assert sent["input"] == "hi" and sent["temperature"] == 0.1
    assert sent["tools"][0]["name"] == "get_signal"
    assert "edge_bps" not in sent["tools"][0]["description"]
    assert "edge_bps" not in sent["tools"][0]["parameters"]["properties"]["k"]["description"]
    assert kwargs["instructions"] == SYSTEM
    assert reply.output_text == "The metric edge_bps matters."


def test_responses_developer_item_in_input() -> None:
    fake = FakeResponsesClient()
    reply = _resp(fake).responses.create(
        model="m",
        input=[{"role": "developer", "content": [{"type": "input_text", "text": SYSTEM}]},
               {"role": "user", "content": "hi"}],
    )
    first = fake.calls[0]["input"][0]["content"]
    assert decode_with_anchor("\n".join(p["text"] for p in first)) == SYSTEM
    assert fake.calls[0]["input"][1] == {"role": "user", "content": "hi"}
    assert reply.output_text == "The metric edge_bps matters."


def test_responses_nothing_to_hide_and_blocked_helpers() -> None:
    fake = FakeResponsesClient()
    client = _resp(fake)
    assert client.responses.create(model="m", input="hello", instructions="be kind").output_text
    assert fake.calls[0]["instructions"] == "be kind"
    with pytest.raises(AttributeError):
        client.responses.parse


@pytest.mark.parametrize("chunk", [1, 2, 3, 100])
def test_responses_stream_restore(chunk: int) -> None:
    client = _resp(FakeResponsesClient(chunk=chunk))
    events = list(client.responses.create(model="m", instructions=SYSTEM, input="hi", stream=True))
    deltas = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert deltas == "The metric edge_bps matters."
    kinds = [e.type for e in events]
    assert kinds.index("response.output_text.done") > max(i for i, k in enumerate(kinds) if k == "response.output_text.delta")
    assert next(e for e in events if e.type == "response.output_text.done").text == "The metric edge_bps matters."
    assert next(e for e in events if e.type == "response.content_part.done").part.text == "The metric edge_bps matters."
    assert next(e for e in events if e.type == "response.completed").response.output_text == "The metric edge_bps matters."


def test_responses_async() -> None:
    async def run() -> tuple:
        fake = FakeResponsesClient(chunk=2, async_mode=True)
        client = _resp(fake)
        reply = await client.responses.create(model="m", instructions=SYSTEM, input="hi")
        stream = await client.responses.create(model="m", instructions=SYSTEM, input="hi", stream=True, _async_stream=True)
        text = "".join([e.delta async for e in stream if e.type == "response.output_text.delta"])
        return reply.output_text, text

    assert asyncio.run(run()) == ("The metric edge_bps matters.",) * 2


def test_core_responses_style_detection() -> None:
    engine = ObfusAIComponent("low", seed="s")
    out = engine.obfuscate_payload({"input": "x", "tools": [{"type": "function", "name": "f", "description": "limit_x = 42.5"}]})
    assert out.payload["instructions"].startswith("<obfusai-context")
    assert engine.obfuscate_payload({"input": "plain"}).anchor == ""
    chat = engine.obfuscate_payload({"messages": [{"role": "system", "content": "limit_x = 42.5"}]})
    assert "messages" in chat.payload and "instructions" not in chat.payload
