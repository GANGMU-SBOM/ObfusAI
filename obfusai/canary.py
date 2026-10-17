"""Per-recipient canary tokens: find out *which copy* of a prompt leaked.

Give each customer, deployment or environment its own prompt copy containing a unique, unguessable
reference token derived from a secret you keep.  When a prompt (or a suspect product's reply to an
extraction attempt) turns up somewhere, :func:`find_recipients` tells you which copies it came from,
without a database of issued tokens.

This is evidence, not prevention.  Anyone who reads the prompt can strip a token they notice, and
comparing two copies reveals the differences, so use several tokens per copy, keep the template
unremarkable, and combine it with real controls (sealed storage, access logs).  It also does nothing against
distillation, which learns from outputs, not from the prompt.

Standard library only.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Iterable, List

__all__ = ["DEFAULT_TEMPLATE", "canary_token", "embed_canaries", "find_recipients"]

DEFAULT_TEMPLATE = "(internal reference {token})"


def canary_token(recipient: str, secret: str, index: int = 0) -> str:
    """The token for ``recipient``: ``ref-`` plus 16 hex characters of an HMAC under ``secret``."""
    if not secret:
        raise ValueError("secret must not be empty")
    digest = hmac.new(secret.encode("utf-8"), f"{recipient}\x1f{index}".encode("utf-8"), hashlib.sha256)
    return "ref-" + digest.hexdigest()[:16]


def embed_canaries(
    text: str, recipient: str, secret: str, *, count: int = 3, template: str = DEFAULT_TEMPLATE
) -> str:
    """Insert ``count`` distinct tokens for ``recipient`` at spread-out paragraph boundaries.

    A short prompt with fewer paragraph breaks than ``count`` gets the surplus tokens appended.  The
    ``template`` must contain ``{token}``.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if "{token}" not in template:
        raise ValueError("template must contain {token}")
    paragraphs = text.split("\n\n")
    marks = [template.format(token=canary_token(recipient, secret, i)) for i in range(count)]
    # one mark per evenly spaced paragraph; with more marks than paragraphs they share the last ones
    for i, mark in enumerate(marks):
        target = min(i * len(paragraphs) // count, len(paragraphs) - 1)
        paragraphs[target] = f"{paragraphs[target]} {mark}" if paragraphs[target] else mark
    return "\n\n".join(paragraphs)


def find_recipients(text: str, recipients: Iterable[str], secret: str, *, count: int = 3) -> List[str]:
    """Recipients whose canary tokens (any of the first ``count``) appear in ``text``."""
    found = []
    for recipient in recipients:
        if any(canary_token(recipient, secret, i) in text for i in range(count)):
            found.append(recipient)
    return found
