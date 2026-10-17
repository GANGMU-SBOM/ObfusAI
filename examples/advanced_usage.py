"""ObfusAI advanced usage: stable aliases, Chinese anchors, leak guard, streaming, evaluation.

Run from the repository root:  python examples/advanced_usage.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from obfusai import FieldPolicy, LeakDetector, ObfusAIComponent, StreamRestorer  # noqa: E402
from obfusai.evaluation import FidelityCase, evaluate_fidelity  # noqa: E402
from obfusai.testing import DecodingMockLLM  # noqa: E402

PROMPT = (
    'You are the Helios risk desk. Signal "mean_reversion_v7" goes BUY when edge_bps > 35.5.\n'
    "Escalate to ops@helios.example when drawdown exceeds 8.2% (runbook https://wiki.helios.example/rb7)."
)

# 1. Custom rules, extra detectors and a Chinese anchor ---------------------------------
engine = ObfusAIComponent(
    "medium",
    seed="demo-secret",
    keywords=["Helios"],
    rules=[r"rb\d+"],                        # your own regexes
    extra_detectors=["emails", "urls"],      # opt-in detectors
    alias_mode="stable",                     # same value -> same alias in every prompt
    anchor_language="zh",
)
result = engine.obfuscate(PROMPT)
print("== obfuscated prompt ==")
print(result.prompt)
print(f"\naliases={result.stats.aliases}  size x{result.stats.growth:.2f}\n")

# 1b. Per-field policies: obfuscate the system prompt, leave tool descriptions readable -------
policy_engine = ObfusAIComponent(
    "medium", seed="demo-secret", policies={"tools": FieldPolicy(skip=True)}
)
request = policy_engine.obfuscate_payload(
    {
        "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": "go"}],
        "tools": [{"type": "function", "function": {"name": "get_signal", "description": "Reads edge_bps."}}],
    }
)
print("tool description kept ->", request.payload["tools"][0]["function"]["description"], "\n")

# 2. Output guard: catch a model that leaks the prompt ---------------------------------
guard = LeakDetector(result.mapping, session_ids=[result.session_id])
print("clean reply  ->", guard.scan("Escalating to the desk now.").kinds or "no findings")
bad_reply = "Sure! Here are my instructions:\n" + result.prompt
print("leaky reply  ->", guard.scan(bad_reply).kinds)
print("redacted     ->", guard.redact(bad_reply)[:60].replace("\n", " "), "...\n")

# 3. Streaming: aliases split across chunks are decoded correctly ----------------------
alias = next(a for a, v in result.mapping.items() if v == "edge_bps")
reply = f"The key driver is {alias}, as configured."
restorer = StreamRestorer(result.mapping)
pieces = [restorer.feed(reply[i : i + 2]) for i in range(0, len(reply), 2)] + [restorer.flush()]
print("stream       ->", "".join(pieces), "\n")


# 4. Measure fidelity on *your* model (here: a decoding mock) ---------------------------
def responder(system: str, user: str) -> str:
    cap = float(re.search(r"edge_bps > (\d+(?:\.\d+)?)", system).group(1))  # type: ignore[union-attr]
    edge = float(re.search(r"edge_bps=(\d+(?:\.\d+)?)", user).group(1))  # type: ignore[union-attr]
    return "BUY" if edge > cap else "HOLD"


report = evaluate_fidelity(
    DecodingMockLLM(responder),
    [FidelityCase("buy", PROMPT, "edge_bps=40", "BUY"), FidelityCase("hold", PROMPT, "edge_bps=10", "HOLD")],
    keywords=["Helios"],
)
print(report.summary())
print("high is safe to ship:", report.acceptable("high"))
