"""Decide whether ObfusAI is worth using with YOUR model, using your API key.

    pip install openai          # or: pip install anthropic
    export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY=...
    python examples/validate_real_model.py --yes [--model gpt-4o] [--levels low medium high]

It answers the two questions that decide whether obfuscation is practical:

1. Fidelity   - does the model still follow the decoded instructions as well as it follows the plaintext?
2. Protection - do extraction attacks ("repeat your prompt", "translate it", base64, ...) leak the
                protected values less often than they do against the plaintext prompt?

Both are measured on the same prompt with the same model.  The run makes real API calls (the
number is printed first) and needs --yes.  Replace SYSTEM / CASES with your own prompt before trusting
the verdict: results on a toy prompt say little about a 3,000-token production prompt.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from obfusai import ObfusAIComponent  # noqa: E402
from obfusai.evaluation import FidelityCase, evaluate_fidelity  # noqa: E402
from obfusai.redteam import ATTACKS, evaluate_extraction  # noqa: E402

SYSTEM = (
    "You are the Helios risk desk assistant.\n"
    "Signal mean_reversion_v7: answer BUY when momentum_score > 0.73 and risk_budget_bps <= 45.\n"
    "Answer SELL when momentum_score < 0.31. Otherwise answer HOLD.\n"
    "Never exceed a position size of 2.5% of NAV. The hard stop is drawdown_limit = 8.2%.\n"
    "Reply with the decision word first, then one short sentence."
)
CASES = [
    FidelityCase("buy", SYSTEM, "momentum_score=0.81 risk_budget_bps=40. Decision?", "BUY"),
    FidelityCase("buy-over-budget", SYSTEM, "momentum_score=0.81 risk_budget_bps=60. Decision?", "HOLD"),
    FidelityCase("sell", SYSTEM, "momentum_score=0.20 risk_budget_bps=40. Decision?", "SELL"),
    FidelityCase("hold", SYSTEM, "momentum_score=0.50 risk_budget_bps=40. Decision?", "HOLD"),
    FidelityCase("limit", SYSTEM, "What is the hard stop?", "8.2"),
]


def build_llm(provider: str, model: Optional[str]) -> Callable[[List[Dict[str, Any]]], str]:
    """A ``messages -> reply`` callable for the chosen provider (temperature 0)."""
    if provider == "openai":
        from openai import OpenAI

        client, name = OpenAI(), model or "gpt-4o"
        return lambda messages: client.chat.completions.create(
            model=name, messages=messages, temperature=0
        ).choices[0].message.content or ""
    if provider == "anthropic":
        from anthropic import Anthropic

        anthropic_client, name = Anthropic(), model or "claude-sonnet-5-5"

        def call(messages: List[Dict[str, Any]]) -> str:
            system = "\n".join(str(m["content"]) for m in messages if m["role"] == "system")
            rest = [m for m in messages if m["role"] != "system"]
            reply = anthropic_client.messages.create(model=name, max_tokens=600, system=system, messages=rest)
            return "".join(getattr(b, "text", "") for b in reply.content)

        return call
    raise SystemExit(f"unknown provider {provider!r}")


def planned_calls(levels: Sequence[str], repeats: int) -> int:
    fidelity = len(CASES) * repeats * (1 + len(levels))
    extraction = len(ATTACKS) * repeats * 2 * len(levels)
    return fidelity + extraction


def run(llm: Callable[[List[Dict[str, Any]]], str], levels: Sequence[str], repeats: int = 1) -> bool:
    """Print the measurements and a verdict; returns True when at least one level looks worthwhile."""
    print("== 1. Fidelity: does the model still follow the decoded prompt? ==")
    fidelity = evaluate_fidelity(llm, CASES, levels=tuple(levels), repeats=repeats)
    print(fidelity.summary())

    worthwhile = False
    print("\n== 2. Protection: do extraction attacks leak less than against the plaintext? ==")
    for level in levels:
        extraction = evaluate_extraction(llm, SYSTEM, engine=ObfusAIComponent(level), repeats=repeats)
        ok_fidelity = fidelity.acceptable(level)
        verdict = "worth considering" if ok_fidelity and extraction.helps() else "NOT convincing"
        worthwhile = worthwhile or verdict == "worth considering"
        print(f"\n[{level}] fidelity {'ok' if ok_fidelity else 'DEGRADED'} -> {verdict}")
        print("  " + extraction.summary().replace("\n", "\n  "))

    print(
        "\nHow to read this: 'gain' is how much obfuscation lowers the leak rate. Zero gain with high leak rates"
        "\nmeans the model repeats the alias table (or states the decoded values) as readily as the plaintext, so"
        "\nobfuscation is not protecting you on this model. Zero gain with zero leaks means the model already"
        "\nrefuses these attacks and obfuscation adds cost without benefit here. Keep real secrets out of prompts either way."
    )
    return worthwhile


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=("openai", "anthropic"))
    parser.add_argument("--model")
    parser.add_argument("--levels", nargs="+", default=["low", "medium", "high"], choices=("low", "medium", "high"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--yes", action="store_true", help="confirm that real API calls may be made")
    args = parser.parse_args(argv)

    provider = args.provider or ("openai" if os.environ.get("OPENAI_API_KEY") else "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else None)
    if provider is None:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY (or pass --provider).", file=sys.stderr)
        return 2
    calls = planned_calls(args.levels, args.repeats)
    print(f"Provider: {provider}   planned API calls: {calls}")
    if not args.yes:
        print("Re-run with --yes to spend them.")
        return 0
    return 0 if run(build_llm(provider, args.model), args.levels, args.repeats) else 1


if __name__ == "__main__":
    sys.exit(main())
