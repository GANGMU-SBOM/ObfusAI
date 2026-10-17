"""Tool-call argument decoding through the real openai / anthropic clients on a mock transport."""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")

from obfusai import ObfusAIComponent  # noqa: E402
from obfusai.anchor import decode_with_anchor  # noqa: E402

SYSTEM = "Rule: BUY when edge_bps > 35. Codename project_falcon."
TOOL_DESC = "Reads edge_bps for a ticker."


def _alias_of(text: str) -> str:
    return text.split(" > ")[0].split()[-1]


def _chunks(text: str, size: int) -> list:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _sse(events: list, with_names: bool = True, done: bool = False) -> bytes:
    out = ""
    for event in events:
        out += (f"event: {event['type']}\n" if with_names else "") + f"data: {json.dumps(event)}\n\n"
    return (out + ("data: [DONE]\n\n" if done else "")).encode()


def _anthropic_http():
    """The HTTP library the installed anthropic SDK demands (newer releases refuse plain httpx objects)."""
    try:
        import httpx2

        return httpx2
    except ImportError:
        return httpx


def _mock_client(handler):
    lib = _anthropic_http()
    return lib.Client(transport=lib.MockTransport(handler))


# ----------------------------------------------------------------- OpenAI Chat Completions
def _chat_client(requests: list, size: int):
    openai = pytest.importorskip("openai")
    from obfusai.adapters.openai import ObfusAIOpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        alias = _alias_of(body["messages"][0]["content"])
        arguments = json.dumps({"metric": alias, "note": "ok"})
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
        if body.get("stream"):
            call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_signal", "arguments": ""}}
            chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [call]}, "finish_reason": None}]}]
            for piece in _chunks(arguments, size):
                part = {"index": 0, "function": {"arguments": piece}}
                chunks.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [part]}, "finish_reason": None}]})
            chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(chunks, False, True))
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_signal", "arguments": arguments}}]}
        return httpx.Response(200, json={**base, "object": "chat.completion",
                                         "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]})

    sdk = openai.OpenAI(api_key="k", base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return ObfusAIOpenAI(sdk, ObfusAIComponent("low", seed="s"))


def _chat_kwargs() -> dict:
    return dict(
        model="gpt-4o",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "get_signal", "description": TOOL_DESC,
                                                 "parameters": {"type": "object", "properties": {"metric": {"type": "string"}}}}}],
    )


def test_chat_completion_tool_call_arguments_are_decoded() -> None:
    requests: list = []
    reply = _chat_client(requests, 3).chat.completions.create(**_chat_kwargs())
    assert "edge_bps" not in requests[0]["tools"][0]["function"]["description"]
    arguments = reply.choices[0].message.tool_calls[0].function.arguments
    assert json.loads(arguments) == {"metric": "edge_bps", "note": "ok"}


@pytest.mark.parametrize("size", [1, 2, 3, 100])
def test_chat_completion_stream_tool_call_arguments_are_decoded(size: int) -> None:
    chunks = list(_chat_client([], size).chat.completions.create(stream=True, **_chat_kwargs()))
    arguments = "".join(
        call.function.arguments or ""
        for chunk in chunks for choice in chunk.choices for call in (choice.delta.tool_calls or [])
    )
    assert json.loads(arguments) == {"metric": "edge_bps", "note": "ok"}
    finish = [i for i, c in enumerate(chunks) if c.choices and c.choices[0].finish_reason]
    last_args = max(i for i, c in enumerate(chunks) if c.choices and c.choices[0].delta.tool_calls)
    assert finish and last_args < finish[-1]  # nothing after the finishing chunk


# ----------------------------------------------------------------- OpenAI Responses
def _responses_client(requests: list, size: int):
    openai = pytest.importorskip("openai")
    from obfusai.adapters.openai import ObfusAIOpenAI

    def response(args: str, status: str = "completed", with_output: bool = True) -> dict:
        item = {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_signal",
                "arguments": args, "status": "completed"}
        return {"id": "r1", "object": "response", "created_at": 0, "model": "m", "status": status,
                "output": [item] if with_output else [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        arguments = json.dumps({"metric": _alias_of(body["instructions"]), "note": "ok"})
        if not body.get("stream"):
            return httpx.Response(200, json=response(arguments))
        item = response(arguments)["output"][0]
        seq = iter(range(1000))
        events = [
            {"type": "response.created", "response": response("", "in_progress", False)},
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": "", "status": "in_progress"}},
        ]
        events += [{"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": piece}
                   for piece in _chunks(arguments, size)]
        events += [
            {"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 0, "name": "get_signal", "arguments": arguments},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response(arguments)},
        ]
        for event in events:
            event["sequence_number"] = next(seq)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))

    sdk = openai.OpenAI(api_key="k", base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return ObfusAIOpenAI(sdk, ObfusAIComponent("low", seed="s"))


def _responses_kwargs() -> dict:
    return dict(model="m", instructions=SYSTEM, input="go",
                tools=[{"type": "function", "name": "get_signal", "description": TOOL_DESC,
                        "parameters": {"type": "object", "properties": {"metric": {"type": "string"}}}}])


def test_responses_function_call_arguments_are_decoded() -> None:
    requests: list = []
    reply = _responses_client(requests, 3).responses.create(**_responses_kwargs())
    assert "edge_bps" not in requests[0]["tools"][0]["description"]
    assert json.loads(reply.output[0].arguments) == {"metric": "edge_bps", "note": "ok"}


@pytest.mark.parametrize("size", [1, 2, 3, 100])
def test_responses_stream_function_call_arguments_are_decoded(size: int) -> None:
    events = list(_responses_client([], size).responses.create(stream=True, **_responses_kwargs()))
    deltas = [e for e in events if e.type == "response.function_call_arguments.delta"]
    assert json.loads("".join(e.delta for e in deltas)) == {"metric": "edge_bps", "note": "ok"}
    done = next(e for e in events if e.type == "response.function_call_arguments.done")
    assert json.loads(done.arguments)["metric"] == "edge_bps" and events.index(done) > events.index(deltas[-1])
    assert json.loads(next(e for e in events if e.type == "response.output_item.done").item.arguments)["metric"] == "edge_bps"
    assert json.loads(next(e for e in events if e.type == "response.completed").response.output[0].arguments)["metric"] == "edge_bps"


@pytest.mark.parametrize("size", [1, 3])
def test_responses_stream_helper_function_call_arguments_are_decoded(size: int) -> None:
    client = _responses_client([], size)
    with client.responses.stream(**_responses_kwargs()) as stream:
        events = list(stream)
        final = stream.get_final_response()
    deltas = [e for e in events if e.type == "response.function_call_arguments.delta"]
    assert json.loads("".join(e.delta for e in deltas)) == {"metric": "edge_bps", "note": "ok"}
    assert json.loads(deltas[-1].snapshot) == {"metric": "edge_bps", "note": "ok"}
    assert json.loads(final.output[0].arguments)["metric"] == "edge_bps"


# ----------------------------------------------------------------- Anthropic
def _anthropic_client(requests: list, size: int):
    anthropic = pytest.importorskip("anthropic")
    from obfusai.adapters.anthropic import ObfusAIAnthropic

    lib = _anthropic_http()
    usage = {"input_tokens": 1, "output_tokens": 1}

    def message(content: list, stop=None) -> dict:
        return {"id": "m1", "type": "message", "role": "assistant", "model": "c", "content": content,
                "stop_reason": stop, "stop_sequence": None, "usage": usage}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        alias = _alias_of(body["system"])
        arguments = json.dumps({"metric": alias, "note": "ok"})
        if not body.get("stream"):
            tool = {"type": "tool_use", "id": "tu_1", "name": "get_signal", "input": json.loads(arguments)}
            return lib.Response(200, json=message([tool], "tool_use"))
        events = [
            {"type": "message_start", "message": message([])},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_signal", "input": {}}},
        ]
        events += [{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": piece}}
                   for piece in _chunks(arguments, size)]
        events += [
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        return lib.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))

    sdk = anthropic.Anthropic(api_key="k", base_url="http://test", http_client=_mock_client(handler))
    return ObfusAIAnthropic(sdk, ObfusAIComponent("low", seed="s"))


def _anthropic_kwargs() -> dict:
    return dict(model="c", max_tokens=50, system=SYSTEM, messages=[{"role": "user", "content": "go"}],
                tools=[{"name": "get_signal", "description": TOOL_DESC,
                        "input_schema": {"type": "object", "properties": {"metric": {"type": "string"}}}}])


def test_anthropic_tool_use_input_is_decoded() -> None:
    requests: list = []
    reply = _anthropic_client(requests, 3).messages.create(**_anthropic_kwargs())
    assert decode_with_anchor(requests[0]["system"]) == SYSTEM
    assert "edge_bps" not in requests[0]["tools"][0]["description"]
    assert reply.content[0].input == {"metric": "edge_bps", "note": "ok"}


@pytest.mark.parametrize("size", [1, 2, 3, 100])
def test_anthropic_create_stream_input_json_is_decoded(size: int) -> None:
    events = list(_anthropic_client([], size).messages.create(stream=True, **_anthropic_kwargs()))
    deltas = [e for e in events if e.type == "content_block_delta"]
    assert json.loads("".join(e.delta.partial_json for e in deltas)) == {"metric": "edge_bps", "note": "ok"}
    kinds = [e.type for e in events]
    assert max(i for i, k in enumerate(kinds) if k == "content_block_delta") < kinds.index("content_block_stop")


@pytest.mark.parametrize("size", [1, 3])
def test_anthropic_stream_helper_tool_input_is_decoded(size: int) -> None:
    client = _anthropic_client([], size)
    with client.messages.stream(**_anthropic_kwargs()) as stream:
        events = list(stream)
        final = stream.get_final_message()
    pieces = "".join(e.partial_json for e in events if e.type == "input_json")
    assert json.loads(pieces) == {"metric": "edge_bps", "note": "ok"}
    last = [e for e in events if e.type == "input_json"][-1]
    assert last.snapshot == {"metric": "edge_bps", "note": "ok"}
    stop = next(e for e in events if e.type == "content_block_stop")
    assert stop.content_block.input == {"metric": "edge_bps", "note": "ok"}
    assert final.content[0].input == {"metric": "edge_bps", "note": "ok"}
