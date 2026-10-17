"""ObfusAI: prompt, role and tool-schema obfuscation middleware for LLM applications."""

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
