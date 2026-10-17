from __future__ import annotations

import pytest

from obfusai import FieldPolicy, ObfusAIComponent
from obfusai.anchor import decode_with_anchor, strip_anchors

SYSTEM = "Rule: BUY when edge_bps > 35.5. Mail ops@desk.example about project_falcon."
TOOLS = [{"type": "function", "function": {"name": "get_signal", "description": "Reads edge_bps (limit 42.5), ops@desk.example"}}]


def _payload():
    return {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "edge_bps 40"}],
        "tools": TOOLS,
    }


def test_skip_tools_keeps_descriptions_readable_but_still_decodes() -> None:
    engine = ObfusAIComponent("low", seed="s", policies={"tools": FieldPolicy(skip=True)})
    out = engine.obfuscate_payload(_payload())
    assert out.payload["tools"][0]["function"]["description"] == TOOLS[0]["function"]["description"]
    system = out.payload["messages"][0]["content"]
    assert "edge_bps" not in strip_anchors(system) and decode_with_anchor(system) == SYSTEM
    assert out.stats.source_chars == out.stats.source_chars and out.stats.growth >= 1.0


def test_per_field_detectors() -> None:
    engine = ObfusAIComponent(
        "low",
        seed="s",
        policies={"system": FieldPolicy(extra_detectors=["emails"]), "tools": FieldPolicy(detectors=["numbers"])},
    )
    out = engine.obfuscate_payload(_payload())
    system = strip_anchors(out.payload["messages"][0]["content"])
    assert "ops@desk.example" not in system and "edge_bps" not in system
    tool = out.payload["tools"][0]["function"]["description"]
    assert "edge_bps" in tool and "ops@desk.example" in tool and "42.5" not in tool  # numbers only
    assert decode_with_anchor(out.payload["messages"][0]["content"]) == SYSTEM


def test_policy_keywords_protected_rules_add_to_component_settings() -> None:
    engine = ObfusAIComponent(
        "low",
        seed="s",
        keywords=["Falcon"],
        policies={
            "system": FieldPolicy(keywords=["Mail"], protected=["edge_bps"], rules=[r"ZX-\d+"]),
        },
    )
    result = engine.obfuscate("Mail Falcon edge_bps ZX-77", field="system")
    assert set(result.mapping.values()) == {"Mail", "Falcon", "ZX-77"}
    assert "edge_bps" in result.text
    plain = engine.obfuscate("Mail Falcon edge_bps ZX-77")  # no field: component settings only
    assert "Mail" in plain.text and "Falcon" not in plain.text


def test_wildcard_fallback_and_specific_wins() -> None:
    engine = ObfusAIComponent(
        "low", policies={"*": FieldPolicy(skip=True), "system": FieldPolicy()}
    )
    assert engine.obfuscate("limit_x = 42.5", field="system").mapping
    assert engine.obfuscate("limit_x = 42.5", field="user").mapping == {}
    assert engine.obfuscate("limit_x = 42.5").mapping == {}  # no field -> "*"


def test_roles_are_policy_fields_and_responses_fields() -> None:
    engine = ObfusAIComponent(
        "low", seed="s", policies={"developer": FieldPolicy(skip=True), "instructions": FieldPolicy(skip=True)}
    )
    out = engine.obfuscate_payload(
        {"messages": [{"role": "developer", "content": "limit_x = 42.5"}, {"role": "system", "content": "limit_y = 7.5"}]}
    )
    assert out.payload["messages"][0]["content"] == "limit_x = 42.5"
    assert "limit_y" not in strip_anchors(out.payload["messages"][1]["content"])
    resp = engine.obfuscate_payload({"instructions": "limit_x = 42.5", "input": "hi"})
    assert resp.anchor == "" and resp.payload["instructions"] == "limit_x = 42.5"


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        ObfusAIComponent(policies={"tools": FieldPolicy(detectors=["nope"])})
    with pytest.raises(ValueError):
        ObfusAIComponent(policies={"tools": FieldPolicy(rules=["(unclosed"])})
    with pytest.raises(TypeError):
        ObfusAIComponent(policies={"tools": {"skip": True}})  # type: ignore[dict-item]
    assert ObfusAIComponent(policies=None).obfuscate("limit_x = 42.5").mapping


def test_anchor_never_lands_in_skipped_message() -> None:
    engine = ObfusAIComponent("low", seed="s", policies={"system": FieldPolicy(skip=True)})
    out = engine.obfuscate_payload(
        {"messages": [{"role": "system", "content": "keep limit_x = 1.5"}, {"role": "user", "content": "q"}],
         "tools": [{"type": "function", "function": {"name": "f", "description": "limit_y = 7.5"}}]}
    )
    first, second = out.payload["messages"][:2]  # a dedicated anchor message is inserted in front
    assert first["role"] == "system" and first["content"].startswith("<obfusai-context")
    assert second == {"role": "system", "content": "keep limit_x = 1.5"}  # untouched
    assert engine.is_skipped("system") and not engine.is_skipped("tools")
    resp = engine.obfuscate_payload(
        {"input": [{"role": "system", "content": "x"}, {"role": "developer", "content": "limit_z = 3.5"}]}
    )
    assert resp.payload["input"][0]["content"] == "x" and "<obfusai-context" in resp.payload["input"][1]["content"]
