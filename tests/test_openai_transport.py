"""Drive the wrappers through the *real* openai client against a mock HTTP transport.

Checks request bodies on the wire and decoding of real SDK objects (create, parse, stream helper).
"""

from __future__ import annotations

import json

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from obfusai import ObfusAIComponent  # noqa: E402
from obfusai.adapters.openai import ObfusAIOpenAI  # noqa: E402
from obfusai.anchor import decode_with_anchor  # noqa: E402

SYSTEM = "Rule: BUY when edge_bps > 35. Codename project_falcon."
EXPECTED = "The metric edge_bps matters."


def _alias_of(text: str) -> str:
    return text.split(" > ")[0].split()[-1]


def _response_json(text: str) -> dict:
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-4o", "status": "completed",
        "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    }


def _sse(events: list) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _stream_events(text: str, size: int) -> list:
    base = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    seq = iter(range(1000))
    item = _response_json(text)["output"][0]
    events = [
        {"type": "response.created", "response": {**_response_json(""), "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "content": [], "status": "in_progress"}},
        {"type": "response.content_part.added", **base, "part": {"type": "output_text", "text": "", "annotations": []}},
    ]
    for i in range(0, len(text), size):
        events.append({"type": "response.output_text.delta", **base, "delta": text[i : i + size], "logprobs": []})
    events += [
        {"type": "response.output_text.done", **base, "text": text, "logprobs": []},
        {"type": "response.content_part.done", **base, "part": {"type": "output_text", "text": text, "annotations": []}},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": _response_json(text)},
    ]
    for event in events:
        event["sequence_number"] = next(seq)
    return events


def _client(requests: list, size: int = 3) -> ObfusAIOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        text = f"The metric {_alias_of(body['instructions'])} matters."
        if body.get("text", {}).get("format", {}).get("name") == "Signal":
            text = json.dumps({"side": "BUY"})  # structured output is JSON
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(_stream_events(text, size)))
        return httpx.Response(200, json=_response_json(text))

    sdk = openai.OpenAI(api_key="k", base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return ObfusAIOpenAI(sdk, ObfusAIComponent("low", seed="s"))


def test_create_over_the_wire() -> None:
    requests: list = []
    reply = _client(requests).responses.create(model="gpt-4o", instructions=SYSTEM, input="hi")
    assert decode_with_anchor(requests[0]["instructions"]) == SYSTEM
    assert "edge_bps" not in requests[0]["instructions"].split("<obfusai-context")[0]
    assert reply.output_text == EXPECTED


def test_parse_over_the_wire_with_dict_schema_and_pydantic_class() -> None:
    from pydantic import BaseModel

    class Signal(BaseModel):
        side: str

    requests: list = []
    client = _client(requests)
    reply = client.responses.parse(model="gpt-4o", instructions=SYSTEM, input="hi", text_format=Signal)
    assert reply.output_text == '{"side": "BUY"}'
    assert reply.output_parsed == Signal(side="BUY")
    assert requests[0]["text"]["format"]["name"] == "Signal"
    assert decode_with_anchor(requests[0]["instructions"]) == SYSTEM

    schema = {"type": "json_schema", "name": "s", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["side"],
        "properties": {"side": {"type": "string", "description": "BUY when edge_bps > 35"}}}}
    client.responses.create(model="gpt-4o", instructions=SYSTEM, input="hi", text={"format": schema})
    sent = requests[-1]["text"]["format"]["schema"]["properties"]["side"]
    assert "edge_bps" not in sent["description"]
    assert schema["schema"]["properties"]["side"]["description"].startswith("BUY when edge_bps")


@pytest.mark.parametrize("size", [1, 2, 3, 100])
def test_stream_helper_over_the_wire(size: int) -> None:
    requests: list = []
    client = _client(requests, size)
    with client.responses.stream(model="gpt-4o", instructions=SYSTEM, input="hi") as stream:
        events = list(stream)
        final = stream.get_final_response()
    deltas = [e for e in events if e.type == "response.output_text.delta"]
    assert "".join(e.delta for e in deltas) == EXPECTED
    assert deltas[-1].snapshot == EXPECTED
    assert next(e for e in events if e.type == "response.output_text.done").text == EXPECTED
    assert final.output_text == EXPECTED
    assert requests[0]["stream"] is True and "<obfusai-context" in requests[0]["instructions"]


@pytest.mark.parametrize("size", [1, 3])
def test_create_stream_over_the_wire(size: int) -> None:
    client = _client([], size)
    events = list(client.responses.create(model="gpt-4o", instructions=SYSTEM, input="hi", stream=True))
    assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == EXPECTED
