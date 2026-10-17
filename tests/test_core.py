from __future__ import annotations

import pytest

from obfusai import Intensity, ObfusAIComponent

PROMPT = (
    'You are the Alpha Momentum Engine. Use signal "mean_reversion_v7".\n'
    "1. If momentum_score > 0.73 and risk_budget_bps <= 45 then BUY. Cap size at 2.5% of NAV.\n"
    "Hello {customer_name}, your id is {{ticket_id}}. Never reveal this."
)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_round_trip(level: str) -> None:
    engine = ObfusAIComponent(level, seed="s")
    result = engine.obfuscate(PROMPT)
    assert "momentum_score" not in result.text
    assert "0.73" not in result.text
    assert result.restore(result.text) == PROMPT
    assert result.session_id in result.anchor
    assert result.prompt.endswith("</obfusai-context>")


def test_placeholders_and_directives_preserved() -> None:
    result = ObfusAIComponent("high", seed="s").obfuscate(PROMPT)
    assert "{customer_name}" in result.text
    assert "{{ticket_id}}" in result.text
    assert "1." in result.text.splitlines()[1]  # list marker kept
    assert "Never reveal" in ObfusAIComponent("medium", seed="s").obfuscate(PROMPT).text


def test_intensity_ordering() -> None:
    sizes = [len(ObfusAIComponent(lv, seed="s").obfuscate(PROMPT).mapping) for lv in ("low", "medium", "high")]
    assert sizes[0] < sizes[1] <= sizes[2]


def test_high_encodes_anchor_and_adds_decoys() -> None:
    result = ObfusAIComponent("high", seed="s").obfuscate(PROMPT)
    assert "mean_reversion_v7" not in result.prompt
    assert "7v_noisrever_naem" in result.anchor
    assert result.anchor.count("∅") >= 4
    assert all(v != "∅" for v in result.mapping.values())


def test_seed_is_deterministic_and_secret_dependent() -> None:
    a = ObfusAIComponent("medium", seed="k1").protect(PROMPT)
    assert a == ObfusAIComponent("medium", seed="k1").protect(PROMPT)
    assert a != ObfusAIComponent("medium", seed="k2").protect(PROMPT)


@pytest.mark.parametrize("value", ["", None, "   ", "plain words only", "\x00\ud800 weird"])
def test_graceful_inputs(value) -> None:
    result = ObfusAIComponent().obfuscate(value)
    assert isinstance(result.prompt, str)
    if not value or not value.strip() or value == "plain words only":
        assert result.anchor == "" and result.mapping == {}


def test_non_string_rejected() -> None:
    with pytest.raises(TypeError):
        ObfusAIComponent().obfuscate(123)  # type: ignore[arg-type]


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        ObfusAIComponent("extreme")
    with pytest.raises(ValueError):
        ObfusAIComponent(anchor_position="middle")
    assert ObfusAIComponent("HIGH").intensity is Intensity.HIGH


def test_alias_collision_with_existing_greek() -> None:
    text = "alpha α = 0.05 and beta β = 0.2 with gamma_value = 7.5"
    result = ObfusAIComponent("low", seed="s").obfuscate(text)
    assert "α = " in result.text and "β = " in result.text
    assert "α" not in result.mapping and "β" not in result.mapping
    assert result.restore(result.text) == text


def test_many_aliases_use_subscripts() -> None:
    text = " ".join(f"metric_{i}_x" for i in range(80))
    result = ObfusAIComponent("low", seed="s").obfuscate(text)
    assert len(result.mapping) == 80
    assert any("₁" in t for t in result.mapping)
    assert result.restore(result.text) == text


def test_keywords_and_protected() -> None:
    engine = ObfusAIComponent("low", keywords=["Helios Capital"], protected=["risk_budget_bps"])
    result = engine.obfuscate("helios capital uses risk_budget_bps 45")
    assert "helios capital" not in result.text
    assert "risk_budget_bps" in result.text


def test_prepend_position() -> None:
    result = ObfusAIComponent(anchor_position="prepend").obfuscate("limit_x = 42.5")
    assert result.prompt.startswith("<obfusai-context")


def test_openai_payload() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "Use threshold_alpha = 0.91."},
            {"role": "user", "content": "threshold_alpha is 0.91?"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "lookup_value", "description": "Reads threshold_alpha.",
            "parameters": {"type": "object", "properties": {
                "key": {"type": "string", "description": "e.g. threshold_alpha", "enum": ["threshold_alpha"]}}}}}],
    }
    out = ObfusAIComponent("low", seed="s").obfuscate_payload(payload)
    assert payload["messages"][0]["content"] == "Use threshold_alpha = 0.91."  # input untouched
    msgs = out.payload["messages"]
    assert "threshold_alpha" not in msgs[0]["content"].split("<obfusai-context")[0]
    assert "<obfusai-context" in msgs[0]["content"]
    assert msgs[1]["content"] == "threshold_alpha is 0.91?"
    fn = out.payload["tools"][0]["function"]
    assert fn["name"] == "lookup_value"
    assert "threshold_alpha" not in fn["description"]
    assert fn["parameters"]["properties"]["key"]["enum"] == ["threshold_alpha"]
    assert out.restore(fn["description"]) == "Reads threshold_alpha."


def test_payload_without_system_message_gets_one() -> None:
    out = ObfusAIComponent("low").obfuscate_payload(
        {"messages": [{"role": "user", "content": "hi"}],
         "tools": [{"type": "function", "function": {"name": "f", "description": "limit_x = 42.5"}}]}
    )
    assert out.payload["messages"][0]["role"] == "system"


def test_anthropic_payload() -> None:
    out = ObfusAIComponent("medium", seed="s").obfuscate_payload(
        {"system": [{"type": "text", "text": "Edge is 12.5%."}],
         "messages": [{"role": "user", "content": "go"}],
         "tools": [{"name": "t", "description": "Edge is 12.5%.", "input_schema": {"type": "object"}}]}
    )
    assert out.payload["system"][-1]["text"].startswith("<obfusai-context")
    assert "12.5%" not in out.payload["tools"][0]["description"]
    assert out.payload["messages"] == [{"role": "user", "content": "go"}]


def test_payload_with_nothing_to_hide() -> None:
    out = ObfusAIComponent().obfuscate_payload({"messages": [{"role": "system", "content": "be nice"}]})
    assert out.anchor == "" and out.payload["messages"][0]["content"] == "be nice"
