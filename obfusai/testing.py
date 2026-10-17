"""Test doubles for code that sends ObfusAI-protected prompts to an LLM."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .anchor import decode_with_anchor

__all__ = ["AliasEchoServer", "DecodingMockLLM"]

Messages = Sequence[Mapping[str, Any]]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


class DecodingMockLLM:
    """Deterministic stand-in for a chat model that obeys the anchor.

    It does what a capable model is told to do: expand every alias using the
    anchor, then hand the *decoded* system prompt and the last user message to
    ``responder``.  Use it to unit-test pipelines and to show that an
    obfuscated prompt carries the same information as the plaintext one.

    Args:
        responder: ``(decoded_system_prompt, user_message) -> reply``.
    """

    def __init__(self, responder: Callable[[str, str], str]) -> None:
        self.responder = responder
        self.calls: List[List[Dict[str, Any]]] = []

    def __call__(self, messages: Messages) -> str:
        self.calls.append([dict(m) for m in messages])
        system = decode_with_anchor(
            "\n".join(
                _text(m.get("content")) for m in messages if m.get("role") in ("system", "developer")
            )
        )
        user = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
        return self.responder(system, user)


def _alias_of(text: str) -> str:
    return text.split(" > ")[0].split()[-1]


def _pieces(text: str, size: int) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class AliasEchoServer:
    """A fake OpenAI/Anthropic upstream for integration tests in *any* language.

    Serves ``/v1/chat/completions``, ``/v1/responses`` and ``/v1/messages`` (JSON and SSE).  Its "model"
    quotes the first alias it finds in the system prompt, both in its text and in a tool call, exactly
    like a model that copies a symbol; so a client behind the gateway should see the *decoded* value
    (``edge_bps``) while ``requests`` shows what really reached the provider.  Any other POST is
    echoed back.  ``chunk`` is the size of streamed text pieces.  Use as a context manager.
    """

    def __init__(self, chunk: int = 3) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.chunk = chunk
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, status: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse(self, events: List[Dict[str, Any]], named: bool = True, done: bool = False) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    head = f"event: {event['type']}\n" if named else ""
                    self.wfile.write(f"{head}data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                if done:
                    self.wfile.write(b"data: [DONE]\n\n")

            def do_GET(self) -> None:
                outer.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
                self._json(200, {"object": "list", "data": [{"id": "m"}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"method": "POST", "path": self.path, "body": body, "headers": dict(self.headers)})
                if self.path == "/boom/v1/chat/completions" or body.get("model") == "rate-limited":
                    self._json(429, {"error": {"message": "slow down", "type": "rate_limit"}})
                    return
                path = self.path.split("?")[0]
                route = {"/v1/chat/completions": outer.chat, "/v1/responses": outer.responses,
                         "/v1/messages": outer.anthropic}.get(path)
                if route is None:
                    self._json(200, {"echo": body})
                    return
                route(self, body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    # ---- OpenAI chat
    def chat(self, h: Any, body: Dict[str, Any]) -> None:
        alias = _alias_of(body["messages"][0]["content"])
        text, args = f"The metric {alias} matters.", json.dumps({"metric": alias})
        base = {"id": "c1", "created": 0, "model": "m"}
        if body.get("stream"):
            chunks = [{**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            chunks += [{**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]} for p in _pieces(text, self.chunk)]
            call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_signal", "arguments": ""}}
            chunks.append({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]})
            for p in _pieces(args, self.chunk):
                part = {"index": 0, "function": {"arguments": p}}
                chunks.append({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [part]}, "finish_reason": None}]})
            chunks.append({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            h._sse(chunks, named=False, done=True)
            return
        message = {"role": "assistant", "content": text, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_signal", "arguments": args}}]}
        h._json(200, {**base, "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]})

    # ---- OpenAI responses
    def responses(self, h: Any, body: Dict[str, Any]) -> None:
        alias = _alias_of(body["instructions"])
        text, args = f"The metric {alias} matters.", json.dumps({"metric": alias})
        msg = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": text, "annotations": []}]}
        call = {"id": "fc_1", "type": "function_call", "call_id": "c1", "name": "get_signal", "arguments": args, "status": "completed"}

        def response(output: List[Any], status: str = "completed") -> Dict[str, Any]:
            return {"id": "r1", "object": "response", "created_at": 0, "model": "m", "status": status, "output": output,
                    "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}

        if not body.get("stream"):
            h._json(200, response([msg, call]))
            return
        ids = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
        events: List[Dict[str, Any]] = [
            {"type": "response.created", "response": response([], "in_progress")},
            {"type": "response.output_item.added", "output_index": 0, "item": {**msg, "content": [], "status": "in_progress"}},
            {"type": "response.content_part.added", **ids, "part": {"type": "output_text", "text": "", "annotations": []}},
        ]
        events += [{"type": "response.output_text.delta", **ids, "delta": p, "logprobs": []} for p in _pieces(text, self.chunk)]
        events += [
            {"type": "response.output_text.done", **ids, "text": text, "logprobs": []},
            {"type": "response.content_part.done", **ids, "part": {"type": "output_text", "text": text, "annotations": []}},
            {"type": "response.output_item.done", "output_index": 0, "item": msg},
            {"type": "response.output_item.added", "output_index": 1, "item": {**call, "arguments": "", "status": "in_progress"}},
        ]
        events += [{"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 1, "delta": p} for p in _pieces(args, self.chunk)]
        events += [
            {"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 1, "name": "get_signal", "arguments": args},
            {"type": "response.output_item.done", "output_index": 1, "item": call},
            {"type": "response.completed", "response": response([msg, call])},
        ]
        for n, event in enumerate(events):
            event["sequence_number"] = n
        h._sse(events)

    # ---- Anthropic
    def anthropic(self, h: Any, body: Dict[str, Any]) -> None:
        alias = _alias_of(body["system"])
        text, args = f"The metric {alias} matters.", json.dumps({"metric": alias})
        usage = {"input_tokens": 1, "output_tokens": 1}

        def message(content: List[Any], stop: Any = None) -> Dict[str, Any]:
            return {"id": "m1", "type": "message", "role": "assistant", "model": "c", "content": content,
                    "stop_reason": stop, "stop_sequence": None, "usage": usage}

        tool = {"type": "tool_use", "id": "tu_1", "name": "get_signal", "input": json.loads(args)}
        if not body.get("stream"):
            h._json(200, message([{"type": "text", "text": text}, tool], "tool_use"))
            return
        events: List[Dict[str, Any]] = [
            {"type": "message_start", "message": message([])},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ]
        events += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}} for p in _pieces(text, self.chunk)]
        events += [
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_signal", "input": {}}},
        ]
        events += [{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": p}} for p in _pieces(args, self.chunk)]
        events += [
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        h._sse(events)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self) -> "AliasEchoServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
