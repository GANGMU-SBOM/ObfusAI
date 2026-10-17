"""Start the gateway in front of a fake provider, so you can try any language's client with no API key.

    python examples/gateway/run_demo_stack.py            # gateway on :8787
    node examples/gateway/client.mjs                     # (or go run / java / ruby / php ...)

The fake provider "answers" by quoting the first alias it finds in the system prompt, so a correct
client prints the *decoded* name (edge_bps), while the provider log below shows what really arrived
(symbols only).  Replace the fake provider with the real one by running ``obfusai serve`` instead.
"""

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from obfusai import ObfusAIComponent  # noqa: E402
from obfusai.proxy import make_server  # noqa: E402
from obfusai.testing import AliasEchoServer  # noqa: E402


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    upstream = AliasEchoServer(chunk=4)
    gateway = make_server(
        ObfusAIComponent("low", seed="demo"), "127.0.0.1", port,
        upstream_openai=upstream.url, upstream_anthropic=upstream.url,
        log=lambda line: print("gateway:", line, flush=True),
    )
    print(f"gateway   http://127.0.0.1:{gateway.server_address[1]}   (fake provider at {upstream.url})", flush=True)

    def show() -> None:
        seen = 0
        while True:
            threading.Event().wait(0.3)
            for request in upstream.requests[seen:]:
                body = request.get("body") or {}
                system = body.get("system") or body.get("instructions") or (body.get("messages") or [{}])[0].get("content", "")
                print("provider received:", json.dumps(str(system).split("<obfusai-context")[0].strip()), flush=True)
            seen = len(upstream.requests)

    threading.Thread(target=show, daemon=True).start()
    try:
        gateway.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
