"""Aliases a model writes into tool-call arguments are decoded (non-stream and stream)."""

from __future__ import annotations

import json

import pytest

from obfusai.guard import StreamRestorer, restore_json_text, restore_json_value

MAPPING = {"α": 'say "hi"\nnow', "β": "C:\\tmp", "⟦γ⟧": "edge_bps"}


def test_restore_json_text_keeps_json_valid() -> None:
    text = '{"q": "α", "path": "β", "metric": "⟦γ⟧", "n": 3}'
    decoded = restore_json_text(text, MAPPING)
    assert json.loads(decoded) == {"q": 'say "hi"\nnow', "path": "C:\\tmp", "metric": "edge_bps", "n": 3}
    assert restore_json_text('{"q": "\\u03b1"}', MAPPING) == '{"q": "say \\"hi\\"\\nnow"}'  # \uXXXX escapes of aliases
    assert restore_json_text('{"q": "\\u00e9"}', MAPPING) == '{"q": "\\u00e9"}'  # other escapes untouched
    assert restore_json_text('{"q": "α', MAPPING).startswith('{"q": "say')  # truncated JSON still works
    assert restore_json_text("", MAPPING) == "" and restore_json_text(None, MAPPING) == ""
    assert restore_json_text("plain α", {}) == "plain α"


def test_restore_json_value_decodes_values_not_keys() -> None:
    value = {"a": ["α", {"b": "x β"}], "α": 1, "n": None}
    assert restore_json_value(value, MAPPING) == {"a": ['say "hi"\nnow', {"b": "x C:\\tmp"}], "α": 1, "n": None}


@pytest.mark.parametrize("size", [1, 2, 3, 5, 100])
@pytest.mark.parametrize("text", [
    '{"q": "α", "path": "β", "metric": "⟦γ⟧"}',
    '{"q": "\\u03b1", "path": "\\u03b2", "metric": "\\u27e6\\u03b3\\u27e7", "e": "\\u00e9"}',  # models may \u-escape
])
def test_json_stream_restorer_matches_batch(size: int, text: str) -> None:
    restorer = StreamRestorer(MAPPING, json_strings=True)
    out = "".join(restorer.feed(text[i : i + size]) for i in range(0, len(text), size)) + restorer.flush()
    assert out == restore_json_text(text, MAPPING)
    assert json.loads(out)["q"] == 'say "hi"\nnow' and json.loads(out)["metric"] == "edge_bps"


def test_plain_stream_restorer_unchanged_by_json_mode_default() -> None:
    restorer = StreamRestorer({"α": 'a"b'})
    assert restorer.feed("x α ") + restorer.flush() == 'x a"b '
