"""ObfusAI: prompt, role and tool-schema obfuscation middleware for LLM applications."""

from .core import (
    DETECTORS,
    Intensity,
    ObfusAIComponent,
    ObfuscatedPayload,
    ObfuscationResult,
    ObfuscationSession,
    ObfuscationStats,
)
from .guard import LeakDetector, LeakReport, StreamRestorer

__version__ = "0.3.0"

__all__ = [
    "ObfusAIComponent",
    "Intensity",
    "DETECTORS",
    "ObfuscationResult",
    "ObfuscatedPayload",
    "ObfuscationSession",
    "ObfuscationStats",
    "LeakDetector",
    "LeakReport",
    "StreamRestorer",
]
