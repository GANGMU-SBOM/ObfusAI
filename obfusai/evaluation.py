"""Measure whether obfuscation preserves a model's behaviour.

Obfuscation only helps if your model still follows the decoded instructions.
:func:`evaluate_fidelity` replays test cases against any chat callable, once
with the plaintext prompt (baseline) and once per intensity level, and reports
pass rates, leak rates and prompt growth.  Run it against your real model
before shipping a level.

Standard library only; the LLM is any ``Callable[[messages], str]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .core import ObfusAIComponent
from .guard import LeakDetector

__all__ = ["CaseResult", "FidelityCase", "FidelityReport", "LevelReport", "evaluate_fidelity"]

ChatCallable = Callable[[List[Dict[str, Any]]], str]
Check = Union[str, Sequence[str], Callable[[str], bool]]


@dataclass(frozen=True)
class FidelityCase:
    """One behavioural test.

    ``check`` is a substring that must appear in the reply, a sequence of such
    substrings (all required), or a predicate ``reply -> bool``.
    """

    name: str
    system: str
    user: str
    check: Check
    tools: Optional[Sequence[Mapping[str, Any]]] = None

    def passes(self, reply: str) -> bool:
        if callable(self.check):
            return bool(self.check(reply))
        needles = [self.check] if isinstance(self.check, str) else list(self.check)
        return all(n in reply for n in needles)


@dataclass(frozen=True)
class CaseResult:
    case: str
    passed: bool
    leaked: bool
    reply: str
    error: str = ""


@dataclass(frozen=True)
class LevelReport:
    """Aggregate for one intensity level (or the plaintext baseline)."""

    level: str
    results: Tuple[CaseResult, ...]
    growth: float = 1.0

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 1.0

    @property
    def errors(self) -> int:
        """Calls that raised instead of returning a reply."""
        return sum(bool(r.error) for r in self.results)

    @property
    def leak_rate(self) -> float:
        return sum(r.leaked for r in self.results) / len(self.results) if self.results else 0.0


@dataclass(frozen=True)
class FidelityReport:
    baseline: LevelReport
    levels: Tuple[LevelReport, ...] = field(default_factory=tuple)

    def acceptable(self, level: str, *, tolerance: float = 0.0) -> bool:
        """True when ``level`` passes at least as often as the baseline (minus ``tolerance``) and never leaks.

        A baseline that never passes means the test setup itself is broken, so nothing is acceptable.
        """
        report = next(lv for lv in self.levels if lv.level == level)
        if self.baseline.pass_rate == 0.0:
            return False
        return report.pass_rate >= self.baseline.pass_rate - tolerance and report.leak_rate == 0.0

    def summary(self) -> str:
        rows = [("level", "pass", "leak", "errors", "size x")]
        for lv in (self.baseline,) + self.levels:
            rows.append((lv.level, f"{lv.pass_rate:.0%}", f"{lv.leak_rate:.0%}", str(lv.errors), f"{lv.growth:.2f}"))
        widths = [max(len(r[i]) for r in rows) for i in range(5)]
        return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)


def evaluate_fidelity(
    llm: ChatCallable,
    cases: Sequence[FidelityCase],
    *,
    levels: Sequence[str] = ("low", "medium", "high"),
    seed: str = "obfusai-eval",
    repeats: int = 1,
    **component_options: Any,
) -> FidelityReport:
    """Run ``cases`` on the plaintext prompt and on each obfuscation level.

    Args:
        llm: ``messages -> reply text``.  Exceptions count as failures.
        cases: Behavioural tests.
        levels: Intensity levels to evaluate.
        seed: Seed for reproducible aliases.
        repeats: Calls per case and level (use >1 for non-deterministic models).
        **component_options: Forwarded to :class:`ObfusAIComponent` (keywords, rules, ...).
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    def run(level: str, build: Callable[[FidelityCase], Tuple[Dict[str, Any], LeakDetector]]) -> LevelReport:
        results: List[CaseResult] = []
        source_chars = out_chars = 0
        for case in cases:
            payload, detector = build(case)
            source_chars += len(case.system)
            out_chars += sum(len(str(m.get("content", ""))) for m in payload["messages"] if m["role"] == "system")
            for _ in range(repeats):
                try:
                    reply = llm(payload["messages"])
                    error = ""
                except Exception as exc:  # a failing model call is a failed case, not a crash
                    reply, error = "", f"{exc.__class__.__name__}: {exc}"
                leaked = bool(error == "" and level != "plaintext" and detector.scan(reply).kinds)
                results.append(CaseResult(case.name, error == "" and case.passes(reply), leaked, reply, error))
        growth = out_chars / source_chars if source_chars else 1.0
        return LevelReport(level, tuple(results), growth)

    def request(case: FidelityCase) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": case.system},
                {"role": "user", "content": case.user},
            ]
        }
        if case.tools:
            body["tools"] = [dict(t) for t in case.tools]
        return body

    baseline = run("plaintext", lambda case: (request(case), LeakDetector()))

    def protected(level: str) -> Callable[[FidelityCase], Tuple[Dict[str, Any], LeakDetector]]:
        engine = ObfusAIComponent(level, seed=seed, **component_options)

        def build(case: FidelityCase) -> Tuple[Dict[str, Any], LeakDetector]:
            result = engine.obfuscate_payload(request(case))
            # structural leaks only (anchor, definitions, aliases, session id): a *correct* answer
            # often contains a hidden value ("HOLD", "8.2%"), which is not a leak in a fidelity test
            return result.payload, LeakDetector(result.mapping, session_ids=[result.session_id], check_values=False)

        return build

    reports = tuple(run(level, protected(level)) for level in levels)
    return FidelityReport(baseline, reports)
