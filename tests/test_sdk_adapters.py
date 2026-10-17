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
class FakeAnthropic:
    def __init__(self) -> None:
        self.calls: List[dict] = []
        self.messages = NS(create=self._create, stream=self._stream, count_tokens=lambda **k: 7)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        system = kwargs["system"]
        alias = system.split(" > ")[0].split()[-1]
        return NS(content=[NS(type="text", text=f"The metric {alias} matters."), NS(type="tool_use", name="x")])

    def _stream(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "stream-manager"


def test_anthropic_wrapper() -> None:
    fake = FakeAnthropic()
    client = ObfusAIAnthropic(fake, ObfusAIComponent("low", seed="s"))
    reply = client.messages.create(
        model="claude", max_tokens=10, system=SYSTEM, messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "t", "description": "Reads edge_bps.", "input_schema": {"type": "object"}}],
    )
    sent = fake.calls[0]
    assert decode_with_anchor(sent["system"]) == SYSTEM
    assert "edge_bps" not in sent["tools"][0]["description"]
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert reply.content[0].text == "The metric edge_bps matters."
    assert client.messages.count_tokens() == 7
    assert client.messages.stream(model="claude", system=SYSTEM, messages=[]) == "stream-manager"
    assert "<obfusai-context" in fake.calls[-1]["system"]


def test_anthropic_stream_create_is_explicit() -> None:
    client = ObfusAIAnthropic(FakeAnthropic())
    with pytest.raises(NotImplementedError):
        client.messages.create(stream=True, system=SYSTEM, messages=[])
    ObfusAIAnthropic(FakeAnthropic(), restore_responses=False).messages.create(
        stream=True, system=SYSTEM, messages=[]
    )
