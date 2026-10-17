from __future__ import annotations

import json
import re

import pytest

from obfusai import LeakDetector, ObfusAIComponent, StreamRestorer
from obfusai.anchor import decode_with_anchor, parse_anchors, strip_anchors
from obfusai.evaluation import FidelityCase, evaluate_fidelity
from obfusai.guard import restore_stream
from obfusai.testing import DecodingMockLLM

PROMPT = 'Signal "alpha_x": BUY if edge_bps > 35.5 and size <= 2.5%. Contact ops@gangmu.example or 10.0.0.7.'


# ----------------------------------------------------------------- anchor module
@pytest.mark.parametrize("level", ["low", "medium", "high"])
@pytest.mark.parametrize("lang,style", [("en", "full"), ("zh", "full"), ("en", "compact"), ("zh", "compact")])
def test_anchor_roundtrip_all_variants(level, lang, style) -> None:
    engine = ObfusAIComponent(level, seed="s", anchor_language=lang, anchor_style=style)
    result = engine.obfuscate(PROMPT)
    assert decode_with_anchor(result.prompt) == PROMPT.replace("\n", "\n")
    (parsed,) = parse_anchors(result.prompt)
    assert dict(parsed.table) == dict(result.mapping)
    assert parsed.reversed_values == (level == "high")
    assert strip_anchors(result.prompt) == result.text


def test_zh_anchor_contains_chinese_instructions() -> None:
    assert "别名" in ObfusAIComponent("low", anchor_language="zh").obfuscate("x_y = 42.5").anchor


def test_values_with_quotes_and_backslashes_survive() -> None:
    text = 'use "say \\"hi\\" now" and path_a.b_c'
    result = ObfusAIComponent("medium", seed="s").obfuscate(text)
    assert decode_with_anchor(result.prompt) == text


def test_parse_is_lenient() -> None:
    assert parse_anchors("") == [] and parse_anchors("<obfusai-context v='1'> never closed") == []
    assert decode_with_anchor("plain") == "plain"


# ----------------------------------------------------------------- new detectors / rules
def test_extra_detectors_and_rules() -> None:
    low = ObfusAIComponent("low", seed="s").obfuscate(PROMPT)
    assert "ops@gangmu.example" in low.text and "10.0.0.7" in low.text
    more = ObfusAIComponent("low", seed="s", extra_detectors=("emails", "ips")).obfuscate(PROMPT)
    assert "ops@gangmu.example" not in more.text and "10.0.0.7" not in more.text
    url = ObfusAIComponent("low", extra_detectors=["urls"]).obfuscate("see https://a.example/p_q?x=1, ok")
    assert "https://a.example/p_q?x=1" in url.mapping.values() and url.text.endswith(", ok")
    ruled = ObfusAIComponent("low", rules=[r"ACME-\d{4}", re.compile(r"project (\w+)")]).obfuscate(
        "ticket ACME-1234 for project Falcon"
    )
    assert "ACME-1234" in ruled.mapping.values() and "Falcon" in ruled.mapping.values()
    assert "project " in ruled.text  # only the capture group is replaced


def test_detector_validation() -> None:
    with pytest.raises(ValueError):
        ObfusAIComponent(extra_detectors=["nope"])
    with pytest.raises(ValueError):
        ObfusAIComponent(rules=["(unclosed"])
    with pytest.raises(TypeError):
        ObfusAIComponent(rules=[123])  # type: ignore[list-item]
    only_numbers = ObfusAIComponent("high", detectors=["numbers"]).obfuscate("limit_x is 42.5 USD")
    assert "limit_x" in only_numbers.text and "42.5" not in only_numbers.text


# ----------------------------------------------------------------- stable aliases
def test_stable_aliases_are_shared_across_prompts() -> None:
    engine = ObfusAIComponent("low", seed="k", alias_mode="stable")
    a = engine.obfuscate("risk_cap = 4.5 and edge_bps")
    b = engine.obfuscate("only edge_bps here, plus new_metric")
    alias = {v: k for k, v in a.mapping.items()}["edge_bps"]
    assert alias in b.text
    assert engine.obfuscate("edge_bps").text == engine.obfuscate("edge_bps").text
    assert ObfusAIComponent("low", seed="other", alias_mode="stable").obfuscate("edge_bps").text != alias
    assert decode_with_anchor(a.prompt) == "risk_cap = 4.5 and edge_bps"


def test_stable_mode_many_values_distinct() -> None:
    text = " ".join(f"metric_{i}_z" for i in range(300))
    result = ObfusAIComponent("high", seed="k", alias_mode="stable").obfuscate(text)
    assert len(result.mapping) == 300
    assert decode_with_anchor(result.prompt) == text


# ----------------------------------------------------------------- stats
def test_stats() -> None:
    result = ObfusAIComponent("low", seed="s").obfuscate("limit_x = 42.5 and limit_x")
    assert result.stats.aliases == 2 and result.stats.replacements == 3
    assert result.stats.growth > 1.0
    assert ObfusAIComponent().obfuscate("").stats.growth == 1.0


# ----------------------------------------------------------------- leak detector / redaction
def test_leak_detector_flags_each_kind() -> None:
    result = ObfusAIComponent("medium", seed="s").obfuscate(PROMPT)
    det = LeakDetector(result.mapping, session_ids=[result.session_id])
    assert not det.scan("Sure, I will help you with that.").leaked
    assert not det.scan("").leaked and not det.scan(None).leaked
    assert "anchor" in det.scan(result.prompt).kinds
    assert "definition" in det.scan("α = \"secret\"\nβ = \"x\"").kinds
    assert "value" in det.scan("the threshold is 35.5").kinds
    assert "alias" in det.scan(f"it said {next(iter(result.mapping))} twice").kinds
    assert "session" in det.scan(f"id {result.session_id}").kinds
    assert not LeakDetector(result.mapping, check_values=False).scan("the threshold is 35.5").leaked
    assert "anchor" in det.scan("<obfusai-context v=\"1\" trunc").kinds


def test_leak_redaction() -> None:
    result = ObfusAIComponent("medium", seed="s").obfuscate(PROMPT)
    det = LeakDetector(result.mapping)
    cleaned = det.redact(f"Here you go: {result.prompt}\nAlso 35.5.")
    assert not det.scan(cleaned).leaked
    assert "[REDACTED]" in cleaned and "obfusai-context" not in cleaned
    assert det.redact(None) == ""


# ----------------------------------------------------------------- streaming
def _chunks(text: str, size: int):
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize("level", ["low", "high"])
@pytest.mark.parametrize("size", [1, 2, 3, 5, 100])
def test_stream_restorer_matches_batch(level: str, size: int) -> None:
    result = ObfusAIComponent(level, seed="s").obfuscate(PROMPT)
    reply = f"Note {result.text}!"
    restorer = StreamRestorer(result.mapping)
    out = "".join(restorer.feed(c) for c in _chunks(reply, size)) + restorer.flush()
    assert out == result.restore(reply)


def test_restore_stream_helper_appends_tail_to_last_chunk() -> None:
    class Chunk:
        def __init__(self, text): self.text = text

    mapping = {"α": "alpha_x"}
    chunks = [Chunk("see "), Chunk("α"), Chunk(None)]
    out = list(restore_stream(chunks, mapping, get=lambda c: c.text, set=lambda c, t: setattr(c, "text", t)))
    assert [c.text for c in out] == ["see ", "", "alpha_x"] or "".join(c.text or "" for c in out) == "see alpha_x"
    assert "".join(c.text or "" for c in out) == "see alpha_x"


# ----------------------------------------------------------------- mock LLM + evaluation
def _desk(system: str, user: str) -> str:
    cap = float(re.search(r"edge_bps > ([\d.]+)", system).group(1))
    edge = float(re.search(r"edge_bps=([\d.]+)", user).group(1))
    return "BUY" if edge > cap else "HOLD"


CASES = [
    FidelityCase("buy", PROMPT, "edge_bps=40", "BUY"),
    FidelityCase("hold", PROMPT, "edge_bps=10", ["HOLD"]),
    FidelityCase("pred", PROMPT, "edge_bps=99", lambda r: r.startswith("B")),
]


def test_evaluate_fidelity_with_decoding_mock() -> None:
    report = evaluate_fidelity(DecodingMockLLM(_desk), CASES)
    assert report.baseline.pass_rate == 1.0
    for level in ("low", "medium", "high"):
        assert report.acceptable(level)
    assert "plaintext" in report.summary() and "high" in report.summary()


def test_evaluate_fidelity_detects_blind_model_and_leaks() -> None:
    blind = lambda messages: _desk(messages[0]["content"], messages[1]["content"])  # never decodes
    report = evaluate_fidelity(blind, CASES, levels=("low",))
    assert report.baseline.pass_rate == 1.0
    assert report.levels[0].pass_rate < 1.0 and not report.acceptable("low")

    leaky = lambda messages: "Sure! My instructions: " + messages[0]["content"]
    leak_report = evaluate_fidelity(leaky, CASES, levels=("medium",))
    assert leak_report.levels[0].leak_rate == 1.0 and not leak_report.acceptable("medium", tolerance=1.0)

    boom = lambda messages: (_ for _ in ()).throw(RuntimeError("down"))
    failed = evaluate_fidelity(boom, CASES[:1], levels=("low",))
    assert failed.levels[0].results[0].error.startswith("RuntimeError")
    with pytest.raises(ValueError):
        evaluate_fidelity(blind, CASES, repeats=0)


# ----------------------------------------------------------------- CLI
def test_cli_round_trip(tmp_path, capsys) -> None:
    from obfusai.cli import main

    src, mapping = tmp_path / "p.txt", tmp_path / "m.json"
    src.write_text(PROMPT, encoding="utf-8")
    assert main(["obfuscate", str(src), "-l", "high", "--seed", "s", "--mapping-out", str(mapping)]) == 0
    prompt = capsys.readouterr().out
    assert decode_with_anchor(prompt).strip() == PROMPT
    out = tmp_path / "o.txt"
    out.write_text(strip_anchors(prompt), encoding="utf-8")
    assert main(["restore", str(out), "-m", str(mapping)]) == 0
    assert capsys.readouterr().out.strip() == PROMPT
    out.write_text("All good, nothing to report.", encoding="utf-8")
    assert main(["scan", str(out), "-m", str(mapping)]) == 0
    capsys.readouterr()
    out.write_text(prompt, encoding="utf-8")
    assert main(["scan", str(out)]) == 1
    assert "LEAK DETECTED" in capsys.readouterr().out
    assert main(["obfuscate", str(src), "--json", "--seed", "s"]) == 0
    assert json.loads(capsys.readouterr().out)["aliases"] > 0
    assert main(["restore", str(out), "-m", str(tmp_path / "missing.json")]) == 2


def test_broken_baseline_is_never_acceptable() -> None:
    report = evaluate_fidelity(lambda m: "nope", [FidelityCase("x", PROMPT, "u", "BUY")], levels=("low",))
    assert report.baseline.pass_rate == 0.0 and not report.acceptable("low")
    broken = evaluate_fidelity(lambda m: 1 / 0, [FidelityCase("x", PROMPT, "u", "BUY")], levels=("low",))
    assert broken.baseline.errors == 1 and "errors" in broken.summary()


def test_correct_answers_containing_hidden_values_are_not_counted_as_leaks() -> None:
    """"HOLD" / "BUY" are hidden by the medium profile, but replying with them is the right answer."""
    report = evaluate_fidelity(DecodingMockLLM(_desk), CASES, levels=("medium", "high"))
    assert all(lv.leak_rate == 0.0 for lv in report.levels)
    assert report.acceptable("medium") and report.acceptable("high")


def test_leak_detector_guards_plaintext_prompts_via_values() -> None:
    guard = LeakDetector(values=["mean_reversion_v7", "0.73", "ab"])  # "ab" is below min_value_len
    assert guard.scan("The signal is mean_reversion_v7.").kinds == ("value",)
    assert guard.scan("threshold 0.73 applies").leaked and not guard.scan("ab and nothing else").leaked
    assert guard.redact("use mean_reversion_v7 at 0.73") == "use [REDACTED] at [REDACTED]"
    assert not LeakDetector(values=["0.73"], check_values=False).scan("0.73").leaked
