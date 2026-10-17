"""ObfusAI basic usage: obfuscate a proprietary trading prompt, then run it on a mock LLM.

Run from the repository root:  python examples/basic_usage.py

The "LLM" below is `obfusai.testing.DecodingMockLLM`, a deterministic stand-in for a
capable model: it reads the anchor, substitutes the aliases, and then applies rules to
the *decoded* prompt.  It shows the anchor carries the full logic; it is not a claim
about how any particular real model behaves (see examples/advanced_usage.py for the
evaluation harness that measures that).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from obfusai import ObfusAIComponent  # noqa: E402
from obfusai.testing import DecodingMockLLM  # noqa: E402

SYSTEM_PROMPT = """You are the Alpha Momentum Engine, a desk assistant for Helios Capital.
Rules for the signal "mean_reversion_v7":
1. If momentum_score > 0.73 and risk_budget_bps <= 45 then answer BUY.
2. If momentum_score < 0.31 then answer SELL.
3. Otherwise answer HOLD.
Never exceed a position size of 2.5% of NAV. Use drawdown_limit = 8.2% as the hard stop.
Answer with the decision and a one-sentence rationale."""

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_signal",
            "description": "Fetch the Alpha Momentum Engine reading for a ticker (momentum_score in 0..1).",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Exchange ticker, e.g. NVDA"}},
                "required": ["ticker"],
            },
        },
    }
]

def desk_rules(system: str, user: str) -> str:
    """The behaviour a model would derive from the *decoded* system prompt."""
    score = float(re.search(r"momentum_score=([\d.]+)", user).group(1))  # type: ignore[union-attr]
    risk = float(re.search(r"risk_budget_bps=([\d.]+)", user).group(1))  # type: ignore[union-attr]
    buy_at = float(re.search(r"momentum_score > ([\d.]+)", system).group(1))  # type: ignore[union-attr]
    risk_cap = float(re.search(r"risk_budget_bps <= ([\d.]+)", system).group(1))  # type: ignore[union-attr]
    sell_at = float(re.search(r"momentum_score < ([\d.]+)", system).group(1))  # type: ignore[union-attr]
    if score > buy_at and risk <= risk_cap:
        return "BUY - momentum clears the entry threshold within the risk budget."
    if score < sell_at:
        return "SELL - momentum is below the exit threshold."
    return "HOLD - no rule triggered."


def mock_chat_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-shaped response from a stand-in model that decodes the anchor first."""
    answer = DecodingMockLLM(desk_rules)(payload["messages"])
    return {"choices": [{"message": {"role": "assistant", "content": answer}}]}


def main() -> None:
    request: Dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "momentum_score=0.81 risk_budget_bps=40 -> decision?"},
        ],
        "tools": TOOLS,
    }
    plain_answer = mock_chat_completion(request)["choices"][0]["message"]["content"]

    for level in ("low", "medium", "high"):
        engine = ObfusAIComponent(level, seed="demo-secret", keywords=["Helios Capital"])
        protected = engine.obfuscate_payload(request)
        system = protected.payload["messages"][0]["content"]
        answer = mock_chat_completion(protected.payload)["choices"][0]["message"]["content"]

        print(f"===== intensity: {level} =====")
        print(system.split("<obfusai-context")[0].strip())
        print("--- tool description sent to the provider ---")
        print(protected.payload["tools"][0]["function"]["description"])
        print(f"--- aliases issued: {len(protected.mapping)} | tool name untouched: "
              f"{protected.payload['tools'][0]['function']['name']}")
        print(f"mock LLM answer: {answer}\n")
        assert answer == plain_answer, "obfuscation changed the model's behaviour"
        assert "mean_reversion_v7" not in system.split("<obfusai-context")[0]
        if level == "high":  # reversed definitions: the secret never appears verbatim anywhere
            assert "mean_reversion_v7" not in str(protected.payload)

    print(f"Plain-prompt answer: {plain_answer}")
    print("OK: every intensity level yields the same decision as the plaintext prompt.")


if __name__ == "__main__":
    main()
