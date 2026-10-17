"""A language-agnostic gateway: protect prompts for *any* SDK by pointing it at a local proxy.

Native ports of the engine to every language would drift apart (and Go/Rust regexes lack the
look-around the detectors use).  The gateway keeps one implementation and works with every
language that can set a base URL (JavaScript/TypeScript, Go, Java, Kotlin, C#, Rust, Ruby, PHP,
Swift, curl ...)::

    obfusai serve --port 8787 --level medium --seed "$OBFUSAI_SEED"
    # then point the SDK at http://127.0.0.1:8787  (OpenAI: .../v1,  Anthropic: no /v1 suffix)

Handled endpoints (request obfuscated, response decoded, streaming included):

* ``POST /v1/chat/completions``      OpenAI Chat Completions (and compatible servers)
* ``POST /v1/responses``             OpenAI Responses API
* ``POST /v1/messages``              Anthropic Messages (and ``/v1/messages/count_tokens``)

Everything else is forwarded untouched, i.e. **unprotected**; ``strict=True`` rejects those POSTs
instead.  API keys are never stored: the client's own ``Authorization`` / ``x-api-key`` headers are
forwarded.  Each request gets its own alias table.  The server binds to localhost by default and has no
authentication of its own, so do not expose it to untrusted networks.

Standard library only (``http.server`` and ``http.client``); intended as a sidecar, not as an
internet-facing edge proxy.
"""

from __future__ import annotations

import http.client
import json
import ssl
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

from .adapters._streams import (
    AnthropicEventRestorer,
    ChatChunkRestorer,
    ResponsesEventRestorer,
    restore_chat_completion,
    restore_message_content,
    restore_response_output,
)
from .core import ObfusAIComponent

__all__ = ["ROUTES", "make_server", "serve"]

#: path -> (provider, payload style)
ROUTES: Dict[str, Tuple[str, str]] = {
    "/v1/chat/completions": ("openai", "openai"),
    "/v1/responses": ("openai", "responses"),
    "/v1/messages": ("anthropic", "anthropic"),
    "/v1/messages/count_tokens": ("anthropic", "anthropic"),
}

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding", "content-encoding",
}
_MAX_BODY = 64 * 1024 * 1024


_FREE_FORM = frozenset({"input"})  # tool_use.input is arbitrary user JSON: it must stay a plain dict


def _ns(value: Any) -> Any:
    """JSON -> attribute-style objects, so the SDK-object restorers can be reused on raw JSON."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: (v if k in _FREE_FORM and isinstance(v, (dict, list)) else _ns(v)) for k, v in value.items()})
    if isinstance(value, list):
        return [_ns(v) for v in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _restore_json_body(route_style: str, data: Any, mapping: Dict[str, str]) -> Any:
    obj = _ns(data)
    if route_style == "openai":
        restore_chat_completion(obj, mapping)
    elif route_style == "responses":
        restore_response_output(obj, mapping)
    elif route_style == "anthropic":
        restore_message_content(getattr(obj, "content", None), mapping)
    return _plain(obj)


def _make_stream_restorer(route_style: str, mapping: Dict[str, str]) -> Any:
    if route_style == "openai":
        return ChatChunkRestorer(mapping)
    if route_style == "responses":
        return ResponsesEventRestorer(mapping)
    return AnthropicEventRestorer(mapping)


def _dumps(value: Any) -> str:
    """Compact JSON, like the providers emit (keeps naive non-Python clients simple)."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _sse_block(event: Optional[str], data: str) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {data}\n\n".encode("utf-8")


class _Upstream:
    def __init__(self, base_url: str) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"invalid upstream URL {base_url!r}")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.prefix = parts.path.rstrip("/")

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=timeout, context=ssl.create_default_context())
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)


class _Config:
    def __init__(
        self, component: ObfusAIComponent, upstream_openai: str, upstream_anthropic: str,
        strict: bool, timeout: float, log: Callable[[str], None],
    ) -> None:
        self.component = component
        self.upstreams = {"openai": _Upstream(upstream_openai), "anthropic": _Upstream(upstream_anthropic)}
        self.strict = strict
        self.timeout = timeout
        self.log = log


def _handler_class(config: _Config) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ObfusAI-Gateway"

        # ---- plumbing
        def log_message(self, fmt: str, *args: Any) -> None:  # request lines only, never bodies
            config.log(f"{self.address_string()} {fmt % args}")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": {"message": message, "type": "obfusai_gateway_error"}})

        def _read_body(self) -> Optional[bytes]:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                self._error(411, "chunked request bodies are not supported; send Content-Length")
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY:
                self._error(413, "request body too large")
                return None
            return self.rfile.read(length) if length else b""

        def _forward_headers(self, body: Optional[bytes]) -> Dict[str, str]:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
            headers["Accept-Encoding"] = "identity"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            return headers

        def _open_upstream(self, provider: str, body: Optional[bytes]) -> Optional[Tuple[http.client.HTTPConnection, http.client.HTTPResponse]]:
            upstream = config.upstreams[provider]
            try:
                conn = upstream.connect(config.timeout)
                conn.request(self.command, upstream.prefix + self.path, body, self._forward_headers(body))
                return conn, conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                self._error(502, f"upstream request failed: {exc.__class__.__name__}: {exc}")
                return None

        def _provider_for_passthrough(self) -> str:
            return "anthropic" if self.headers.get("x-api-key") or self.path.startswith("/v1/messages") else "openai"

        # ---- relaying
        def _relay_verbatim(self, resp: http.client.HTTPResponse) -> None:
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            body = resp.read()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _relay_json(self, resp: http.client.HTTPResponse, style: str, mapping: Dict[str, str]) -> None:
            raw = resp.read()
            try:
                body = _dumps(_restore_json_body(style, json.loads(raw), mapping)).encode("utf-8")
            except ValueError:
                body = raw  # not JSON: nothing to decode
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse_events(self, resp: http.client.HTTPResponse) -> Iterator[Tuple[Optional[str], str]]:
            event: Optional[str] = None
            data: List[str] = []
            while True:
                line = resp.readline()
                if not line:
                    if data:
                        yield event, "\n".join(data)
                    return
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                if not text:
                    if data:
                        yield event, "\n".join(data)
                    event, data = None, []
                elif text.startswith("event:"):
                    event = text[6:].strip()
                elif text.startswith("data:"):
                    data.append(text[5:][1:] if text[5:].startswith(" ") else text[5:])

        def _relay_sse(self, resp: http.client.HTTPResponse, style: str, mapping: Dict[str, str]) -> None:
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            restorer = _make_stream_restorer(style, mapping)

            def emit(event: Optional[str], payload: Any) -> None:
                name = event and (payload.get("type") if isinstance(payload, dict) and payload.get("type") else event)
                self.wfile.write(_sse_block(name, _dumps(payload)))
                self.wfile.flush()

            def flush_tail() -> None:
                for tail in restorer.finish():
                    plain = _plain(tail)
                    emit(plain.get("type") if style != "openai" else None, plain)

            try:
                for event, data in self._sse_events(resp):
                    if data.strip() == "[DONE]":
                        flush_tail()
                        self.wfile.write(_sse_block(event, data))
                        self.wfile.flush()
                        continue
                    try:
                        parsed = json.loads(data)
                    except ValueError:
                        self.wfile.write(_sse_block(event, data))
                        self.wfile.flush()
                        continue
                    for out in restorer.process(_ns(parsed)):
                        emit(event, _plain(out))
                flush_tail()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client went away

        # ---- request handlers
        def _passthrough(self) -> None:
            body = self._read_body() if self.command in ("POST", "PUT", "PATCH", "DELETE") else None
            if body is None and self.command in ("POST", "PUT", "PATCH"):
                return
            opened = self._open_upstream(self._provider_for_passthrough(), body)
            if opened is None:
                return
            conn, resp = opened
            try:
                self._relay_verbatim(resp)
            finally:
                conn.close()

        def _guard(self) -> bool:
            """Reject absolute-form / odd request targets so the proxy cannot be steered elsewhere."""
            if not self.path.startswith("/") or self.path.startswith("//"):
                self._error(400, "request target must be an absolute path")
                return False
            return True

        def do_POST(self) -> None:
            if not self._guard():
                return
            route = ROUTES.get(self.path.split("?", 1)[0].rstrip("/"))
            if route is None:
                if config.strict:
                    self._error(403, f"POST {self.path} is not a protected endpoint (gateway runs in strict mode)")
                    return
                config.log(f"WARNING forwarding unprotected POST {self.path.split('?', 1)[0]}")
                self._passthrough()
                return
            provider, style = route
            raw = self._read_body()
            if raw is None:
                return
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("body must be a JSON object")
            except ValueError as exc:
                self._error(400, f"invalid JSON request body: {exc}")  # fail closed: never forward unparsed prompts
                return
            protected = config.component.obfuscate_payload(payload, style=style)
            body = _dumps(protected.payload).encode("utf-8")
            opened = self._open_upstream(provider, body)
            if opened is None:
                return
            conn, resp = opened
            try:
                mapping = dict(protected.mapping)
                content_type = resp.getheader("Content-Type", "")
                if resp.status >= 400 or not mapping:
                    self._relay_verbatim(resp)
                elif "text/event-stream" in content_type:
                    self._relay_sse(resp, style, mapping)
                else:
                    self._relay_json(resp, style, mapping)
            finally:
                conn.close()

        def do_GET(self) -> None:
            if not self._guard():
                return
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
                return
            self._passthrough()

        do_DELETE = do_PUT = do_PATCH = do_GET

    return Handler


def make_server(
    component: ObfusAIComponent,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    upstream_openai: str = "https://api.openai.com",
    upstream_anthropic: str = "https://api.anthropic.com",
    strict: bool = False,
    timeout: float = 600.0,
    log: Optional[Callable[[str], None]] = None,
) -> ThreadingHTTPServer:
    """Create (but do not start) the gateway server; ``port=0`` picks a free port."""
    logger = log or (lambda message: print(message, file=sys.stderr))
    config = _Config(component, upstream_openai, upstream_anthropic, strict, timeout, logger)
    server = ThreadingHTTPServer((host, port), _handler_class(config))
    server.daemon_threads = True
    return server


def serve(component: ObfusAIComponent, host: str = "127.0.0.1", port: int = 8787, **options: Any) -> None:
    """Run the gateway until interrupted."""
    server = make_server(component, host, port, **options)
    print(f"ObfusAI gateway listening on http://{host}:{server.server_address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
