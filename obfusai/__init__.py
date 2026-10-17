"""ObfusAI: prompt hygiene for LLM apps: audit, leak detection, measurement and optional obfuscation."""

from .core import (
    DETECTORS,
    FieldPolicy,
    Intensity,
    ObfusAIComponent,
    ObfuscatedPayload,
    ObfuscationResult,
    ObfuscationSession,
    ObfuscationStats,
)
from .audit import AuditFinding, AuditReport, audit
from .guard import LeakDetector, LeakReport, StreamRestorer

__version__ = "0.10.0"

__all__ = [
    "ObfusAIComponent",
    "Intensity",
    "DETECTORS",
    "FieldPolicy",
    "ObfuscationResult",
    "ObfuscatedPayload",
    "ObfuscationSession",
    "ObfuscationStats",
    "audit",
    "AuditFinding",
    "AuditReport",
    "LeakDetector",
    "LeakReport",
    "StreamRestorer",
]
