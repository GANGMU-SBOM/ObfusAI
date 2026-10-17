from __future__ import annotations

import pytest

from obfusai.canary import canary_token, embed_canaries, find_recipients


def test_canary_tokens_are_deterministic_unique_and_secret_dependent() -> None:
    assert canary_token("acme", "s") == canary_token("acme", "s")
    assert len({canary_token("acme", "s", i) for i in range(5)} | {canary_token("globex", "s")}) == 6
    assert canary_token("acme", "s") != canary_token("acme", "other")
    with pytest.raises(ValueError):
        canary_token("acme", "")


def test_embed_and_find_recipients() -> None:
    text = "Intro.\n\nRules here.\n\nMore rules.\n\nClosing."
    marked = embed_canaries(text, "acme", "s3cret", count=3)
    assert marked.count("ref-") == 3 and marked.startswith("Intro.")
    assert find_recipients(marked, ["acme", "globex", "initech"], "s3cret") == ["acme"]
    assert find_recipients(marked, ["acme"], "wrong-secret") == []
    # a partial leak that kept only one paragraph still identifies the source
    leaked_fragment = marked.split("\n\n")[1]
    assert find_recipients(leaked_fragment, ["acme", "globex"], "s3cret") == ["acme"]
    # short prompts still get all tokens; templates are validated
    assert embed_canaries("one paragraph", "acme", "s", count=3).count("ref-") == 3
    with pytest.raises(ValueError):
        embed_canaries("x", "a", "s", template="no placeholder")
    with pytest.raises(ValueError):
        embed_canaries("x", "a", "s", count=0)
