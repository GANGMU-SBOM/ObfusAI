"""End-to-end tests of the language-agnostic gateway: a fake upstream, the gateway, real SDKs and raw HTTP."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from obfusai import ObfusAIComponent
from obfusai.anchor import decode_with_anchor
from obfusai.proxy import make_server
from obfusai.testing import AliasEchoServer

SYSTEM = "Rule: BUY when edge_bps > 35. Codename project_falcon."
TOOL_DESC = "Reads edge_bps for a ticker."
EXPECTED = "The metric edge_bps matters."


@pytest.fixture
def stack():
    """(fake upstream, gateway base URL, gateway log lines); chunk size is set per test via .chunk."""
    upstream = AliasEchoServer()
    logs: List[str] = []
    gateway = make_server(
        ObfusAIComponent("low", seed="s"), "127.0.0.1", 0,
        upstream_openai=upstream.url, upstream_anthropic=upstream.url, log=logs.append,
    )
    thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{gateway.server_address[1]}"
    yield upstream, base, logs
    gateway.shutdown()
    gateway.server_close()
    upstream.close()


def _post(url: str, payload: Any, headers: Dict[str, str] = None, raw: bytes = None):
    request = urllib.request.Request(url, data=raw if raw is not None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
    return urllib.request.urlopen(request, timeout=10)


def _chat_body(**extra: Any) -> Dict[str, Any]:
    return {"model": "m", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "get_signal", "description": TOOL_DESC,
                                                        "parameters": {"type": "object", "properties": {"metric": {"type": "string"}}}}}],
            **extra}


def _visible(text: str) -> str:
    return text.split("<obfusai-context")[0]


# ----------------------------------------------------------------- raw HTTP: what any language's SDK sees
def test_raw_http_chat_completion_is_protected_and_decoded(stack) -> None:
    upstream, base, _ = stack
    with _post(base + "/v1/chat/completions", _chat_body(), {"Authorization": "Bearer k"}) as resp:
        data = json.loads(resp.read())
    sent = upstream.requests[0]
    assert sent["headers"]["Authorization"] == "Bearer k"  # the client's own key is forwarded
    assert "edge_bps" not in _visible(sent["body"]["messages"][0]["content"])
    assert decode_with_anchor(sent["body"]["messages"][0]["content"]) == SYSTEM
    assert "edge_bps" not in sent["body"]["tools"][0]["function"]["description"]
    message = data["choices"][0]["message"]
    assert message["content"] == EXPECTED
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"metric": "edge_bps"}


@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_raw_http_sse_streams_are_decoded(stack, chunk: int) -> None:
    upstream, base, _ = stack
    upstream.chunk = chunk
    with _post(base + "/v1/chat/completions", _chat_body(stream=True)) as resp:
        assert "text/event-stream" in resp.headers["Content-Type"]
        lines = resp.read().decode().split("\n")
    datas = [line[6:] for line in lines if line.startswith("data: ")]
    assert datas[-1] == "[DONE]"
    chunks = [json.loads(d) for d in datas[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    args = "".join(t["function"]["arguments"] or "" for c in chunks for t in (c["choices"][0]["delta"].get("tool_calls") or []))
    assert text == EXPECTED and json.loads(args) == {"metric": "edge_bps"}


# ----------------------------------------------------------------- real SDKs pointed at the gateway
@pytest.mark.parametrize("chunk", [1, 3])
def test_openai_sdk_through_gateway(stack, chunk: int) -> None:
    openai = pytest.importorskip("openai")
    upstream, base, _ = stack
    upstream.chunk = chunk
    client = openai.OpenAI(api_key="k", base_url=base + "/v1")
    reply = client.chat.completions.create(model="m", messages=_chat_body()["messages"], tools=_chat_body()["tools"])
    assert reply.choices[0].message.content == EXPECTED
    assert json.loads(reply.choices[0].message.tool_calls[0].function.arguments) == {"metric": "edge_bps"}
    stream = client.chat.completions.create(model="m", messages=_chat_body()["messages"], tools=_chat_body()["tools"], stream=True)
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text == EXPECTED
    assert "edge_bps" not in _visible(upstream.requests[0]["body"]["messages"][0]["content"])


@pytest.mark.parametrize("chunk", [1, 3])
def test_openai_responses_through_gateway(stack, chunk: int) -> None:
    openai = pytest.importorskip("openai")
    upstream, base, _ = stack
    upstream.chunk = chunk
    client = openai.OpenAI(api_key="k", base_url=base + "/v1")
    tools = [{"type": "function", "name": "get_signal", "description": TOOL_DESC, "parameters": {"type": "object", "properties": {}}}]
    reply = client.responses.create(model="m", instructions=SYSTEM, input="go", tools=tools)
    assert reply.output_text == EXPECTED
    assert json.loads(reply.output[1].arguments) == {"metric": "edge_bps"}
    assert "edge_bps" not in upstream.requests[0]["body"]["tools"][0]["description"]
    events = list(client.responses.create(model="m", instructions=SYSTEM, input="go", tools=tools, stream=True))
    assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == EXPECTED
    args = "".join(e.delta for e in events if e.type == "response.function_call_arguments.delta")
    assert json.loads(args) == {"metric": "edge_bps"}
    with client.responses.stream(model="m", instructions=SYSTEM, input="go", tools=tools) as helper:
        list(helper)
        assert helper.get_final_response().output_text == EXPECTED


@pytest.mark.parametrize("chunk", [1, 3])
def test_anthropic_sdk_through_gateway(stack, chunk: int) -> None:
    anthropic = pytest.importorskip("anthropic")
    upstream, base, _ = stack
    upstream.chunk = chunk
    client = anthropic.Anthropic(api_key="k", base_url=base)
    kwargs = dict(model="c", max_tokens=50, system=SYSTEM, messages=[{"role": "user", "content": "go"}],
                  tools=[{"name": "get_signal", "description": TOOL_DESC, "input_schema": {"type": "object", "properties": {}}}])
    reply = client.messages.create(**kwargs)
    assert reply.content[0].text == EXPECTED and reply.content[1].input == {"metric": "edge_bps"}
    assert decode_with_anchor(upstream.requests[0]["body"]["system"]) == SYSTEM
    assert {k.lower(): v for k, v in upstream.requests[0]["headers"].items()}["x-api-key"] == "k"
    events = list(client.messages.create(stream=True, **kwargs))
    text = "".join(e.delta.text for e in events if e.type == "content_block_delta" and e.delta.type == "text_delta")
    args = "".join(e.delta.partial_json for e in events if e.type == "content_block_delta" and e.delta.type == "input_json_delta")
    assert text == EXPECTED and json.loads(args) == {"metric": "edge_bps"}
    with client.messages.stream(**kwargs) as helper:
        assert "".join(helper.text_stream) == EXPECTED
        assert helper.get_final_message().content[1].input == {"metric": "edge_bps"}


# ----------------------------------------------------------------- behaviour and safety
def test_invalid_json_is_rejected_and_never_forwarded(stack) -> None:
    upstream, base, _ = stack
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base + "/v1/chat/completions", None, raw=b"{not json")
    assert excinfo.value.code == 400 and upstream.requests == []
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base + "/v1/chat/completions", [1, 2])
    assert excinfo.value.code == 400 and upstream.requests == []


def test_absolute_form_request_target_cannot_steer_the_proxy(stack) -> None:
    upstream, base, _ = stack
    port = int(base.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(b"GET http://evil.example/steal HTTP/1.0\r\nHost: x\r\n\r\n")
        reply = sock.recv(4096)
    assert b"400" in reply.split(b"\r\n")[0] and upstream.requests == []


def test_unprotected_endpoints_are_forwarded_with_a_warning_or_blocked_in_strict_mode(stack) -> None:
    upstream, base, logs = stack
    with _post(base + "/v1/completions", {"prompt": "plain"}) as resp:
        assert json.loads(resp.read()) == {"echo": {"prompt": "plain"}}
    assert any("unprotected POST /v1/completions" in line for line in logs)
    strict = make_server(ObfusAIComponent("low", seed="s"), "127.0.0.1", 0, upstream_openai=upstream.url,
                         upstream_anthropic=upstream.url, strict=True, log=lambda m: None)
    threading.Thread(target=strict.serve_forever, daemon=True).start()
    try:
        before = len(upstream.requests)
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"http://127.0.0.1:{strict.server_address[1]}/v1/completions", {"prompt": "plain"})
        assert excinfo.value.code == 403 and len(upstream.requests) == before
    finally:
        strict.shutdown()
        strict.server_close()


def test_upstream_errors_are_relayed_untouched_and_get_and_health_work(stack) -> None:
    upstream, base, _ = stack
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base + "/v1/chat/completions", _chat_body(model="rate-limited"))
    assert excinfo.value.code == 429 and json.loads(excinfo.value.read())["error"]["type"] == "rate_limit"
    with urllib.request.urlopen(base + "/healthz", timeout=5) as resp:
        assert json.loads(resp.read()) == {"status": "ok"}
    request = urllib.request.Request(base + "/v1/models", headers={"Authorization": "Bearer k"})
    with urllib.request.urlopen(request, timeout=5) as resp:
        assert json.loads(resp.read())["data"][0]["id"] == "m"
    assert upstream.requests[-1]["path"] == "/v1/models" and upstream.requests[-1]["headers"]["Authorization"] == "Bearer k"


def test_unreachable_upstream_gives_a_502() -> None:
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    port = dead.getsockname()[1]
    dead.close()  # nothing listens here any more
    gateway = make_server(ObfusAIComponent("low", seed="s"), "127.0.0.1", 0,
                          upstream_openai=f"http://127.0.0.1:{port}", upstream_anthropic=f"http://127.0.0.1:{port}", log=lambda m: None)
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"http://127.0.0.1:{gateway.server_address[1]}/v1/chat/completions", _chat_body())
        assert excinfo.value.code == 502
    finally:
        gateway.shutdown()
        gateway.server_close()


def test_nothing_to_hide_passes_through_unchanged(stack) -> None:
    upstream, base, _ = stack
    body = {"model": "m", "messages": [{"role": "system", "content": "Be polite and kind."}, {"role": "user", "content": "hi"}]}
    with _post(base + "/v1/chat/completions", body) as resp:
        assert resp.status == 200
    assert upstream.requests[0]["body"] == body  # nothing hidden: no anchor, no rewrite
