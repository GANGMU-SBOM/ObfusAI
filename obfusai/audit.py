"""Find personal data and credentials in a prompt *before* you ship it.

Obfuscation hides a value from a casual reader but the anchor still carries it, so
the right fix for an API key or a customer's card number is to remove it from the
prompt.  :func:`audit` locates such values (with checksum validation to cut false
positives) and reports masked previews, never the full value.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .core import _PROFILES, DETECTORS, ObfusAIComponent
from .detectors import PII_DETECTORS

__all__ = ["DEFAULT_AUDIT_DETECTORS", "AuditFinding", "AuditReport", "audit", "mask"]

#: Detectors used when none are given: e-mails, IPs, credentials and personal data.
DEFAULT_AUDIT_DETECTORS: Tuple[str, ...] = ("emails", "ips", "secrets") + PII_DETECTORS


def mask(value: str) -> str:
    """Short preview that does not disclose the value (first/last two characters only)."""
    if len(value) < 10:
        return "•" * len(value)
    return f"{value[:2]}…{value[-2:]} ({len(value)} chars)"


@dataclass(frozen=True)
class AuditFinding:
    """One sensitive value: which detector matched, where, and a masked preview."""

    detector: str
    start: int
    end: int
    preview: str


@dataclass(frozen=True)
class AuditReport:
    findings: Tuple[AuditFinding, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def counts(self) -> Dict[str, int]:
        """Findings per detector."""
        out: Dict[str, int] = {}
        for finding in self.findings:
            out[finding.detector] = out.get(finding.detector, 0) + 1
        return out


def audit(text: Optional[str], detectors: Optional[Iterable[str]] = None) -> AuditReport:
    """Scan ``text`` for sensitive values.

    Args:
        text: Prompt text (empty or ``None`` yields a clean report).
        detectors: Detector names (see :data:`obfusai.DETECTORS`); defaults to
            :data:`DEFAULT_AUDIT_DETECTORS`.
    """
    if not text:
        return AuditReport()
    chosen = tuple(detectors) if detectors is not None else DEFAULT_AUDIT_DETECTORS
    unknown = sorted(set(chosen) - set(DETECTORS))
    if unknown:
        raise ValueError(f"unknown detectors {unknown}; choose from {list(DETECTORS)}")
    engine = ObfusAIComponent("low", seed="audit", detectors=chosen)
    spans = engine._extract(text, _PROFILES[engine.intensity], [])
    findings: List[AuditFinding] = [
        AuditFinding(span.detector, span.start, span.end, mask(span.value)) for span in spans
    ]
    return AuditReport(tuple(findings))
