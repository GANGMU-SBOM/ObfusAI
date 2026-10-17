from __future__ import annotations

import pytest

from obfusai import DETECTORS, ObfusAIComponent, audit
from obfusai.anchor import decode_with_anchor
from obfusai.audit import mask
from obfusai.cli import main
from obfusai.detectors import cn_id_ok, iban_ok, luhn_ok, phone_ok, ssn_ok

PII = ["phones", "cn_phones", "cards", "ssn", "cn_ids", "ibans", "uuids", "secrets"]


def hidden(text: str, detectors=PII, **kw):
    result = ObfusAIComponent("low", seed="s", detectors=list(detectors), **kw).obfuscate(text)
    assert decode_with_anchor(result.prompt) == text
    return set(result.mapping.values())


@pytest.mark.parametrize(
    "value",
    [
        "+1 (415) 555-2671", "415-555-2671", "+44 20 7946 0958", "+4915123456789", "(020) 7946-0958",
        "13800138000", "+86 13800138000",
        "4111 1111 1111 1111", "5500-0000-0000-0004", "378282246310005",
        "123-45-6789", "11010519491231002X",
        "GB82 WEST 1234 5698 7654 32", "DE89370400440532013000",
        "123e4567-e89b-12d3-a456-426614174000",
        "sk-ant-api03-abcdefghijklmnopqrstuvwx", "AKIAIOSFODNN7EXAMPLE", "ghp_" + "a1B2" * 9,
        "xoxb-123456789012-abcdefghij", "AIza" + "x" * 35,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
    ],
)
def test_detects_sensitive_values(value: str) -> None:
    assert value in hidden(f"see {value} now."), value


@pytest.mark.parametrize(
    "text",
    [
        "date 2024-01-05 and version 1.2.3 and ip 10.0.0.7",
        "order 1234567890123 total 4111 1111 1111 1112",  # fails Luhn
        "ssn 000-12-3456 and 666-12-3456 and 900-12-3456",
        "id 11010519491231002Y and 110105194913310021",  # bad checksum / bad date
        "GB82 WEST 1234 5698 7654 33",  # bad mod-97
        "call 12-345 or 1234567",  # too short for a phone
        "price 42, 1,234 and 0.73",
    ],
)
def test_validators_reject_lookalikes(text: str) -> None:
    assert hidden(text, detectors=PII) == set()


def test_validators_directly() -> None:
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112") and not luhn_ok("0000000000000000")
    assert ssn_ok("123-45-6789") and not ssn_ok("123-00-6789")
    assert cn_id_ok("11010519491231002X") and not cn_id_ok("11010519491231002Y")
    assert iban_ok("GB82 WEST 1234 5698 7654 32") and not iban_ok("GB00 WEST 1234 5698 7654 32")
    assert phone_ok("+4915123456789") and not phone_ok("1234567890")


def test_new_detectors_are_registered_and_ordered_before_numbers() -> None:
    assert set(PII) <= set(DETECTORS)
    assert DETECTORS.index("cards") < DETECTORS.index("numbers")
    # a card number is one alias, not four fragments, even with the generic number detector on
    result = ObfusAIComponent("low", seed="s", extra_detectors=["cards", "phones"]).obfuscate("pay 4111 1111 1111 1111 or +1 (415) 555-2671")
    assert {"4111 1111 1111 1111", "+1 (415) 555-2671"} <= set(result.mapping.values())


def test_number_regex_no_longer_swallows_trailing_commas() -> None:
    result = ObfusAIComponent("low", seed="s").obfuscate("limits 42, 1,234,567 and 0.5, then 12,34")
    assert set(result.mapping.values()) == {"42", "1,234,567", "0.5", "12", "34"}
    assert "Σ," in result.text or "," in result.text


def test_audit_reports_masked_findings() -> None:
    text = "mail ops@desk.example key sk-ant-api03-abcdefghijklmnopqrstuvwx card 4111 1111 1111 1111"
    report = audit(text)
    assert not report.clean and report.counts == {"emails": 1, "secrets": 1, "cards": 1}
    assert all("abcdefghijklmnopqrstuv" not in f.preview and "4111 1111" not in f.preview for f in report.findings)
    assert text[report.findings[0].start : report.findings[0].end] == "ops@desk.example"
    assert audit("").clean and audit(None).clean and audit("nothing here").clean
    assert audit(text, ["emails"]).counts == {"emails": 1}
    with pytest.raises(ValueError):
        audit(text, ["nope"])
    assert mask("short") == "•••••" and mask("abcdefghijkl").startswith("ab…kl")


def test_audit_cli(tmp_path, capsys) -> None:
    f = tmp_path / "p.txt"
    f.write_text("contact 13800138000 or ops@desk.example", encoding="utf-8")
    assert main(["audit", str(f)]) == 1
    out = capsys.readouterr().out
    assert "cn_phones" in out and "emails" in out and "13800138000" not in out and "SENSITIVE VALUES FOUND" in out
    assert main(["audit", str(f), "-d", "ssn"]) == 0
    assert "clean" in capsys.readouterr().out
    f.write_text("nothing sensitive", encoding="utf-8")
    assert main(["audit", str(f)]) == 0


DATES = [
    "2024-01-05", "2024-01-05T10:30:00Z", "2024-01-05 10:30", "2024/1/5", "05/01/2024", "13/01/2024", "5.1.24",
    "2024年1月5日", "2024年1月", "Jan 5, 2024", "March 5th", "5 January 2024", "5th March",
    "10:30", "10:30:15", "9:05 PM", "23:59",
]


@pytest.mark.parametrize("value", DATES)
def test_dates_become_one_alias(value: str) -> None:
    assert value in hidden(f"due {value} sharp", detectors=["dates"]), value


@pytest.mark.parametrize("text", ["2024-13-45", "1/2/3", "ratio 3:1", "v1.2.3", "ip 10.0.0.7", "at 25:99", "may be 5 or so", "Mar"])
def test_dates_reject_lookalikes(text: str) -> None:
    assert hidden(text, detectors=["dates"]) == set()


def test_dates_are_in_default_profiles_and_beat_numbers() -> None:
    for level in ("low", "medium", "high"):
        result = ObfusAIComponent(level, seed="s").obfuscate("Review on 2024-01-05 at 10:30 for limit_x = 42.5")
        values = set(result.mapping.values())
        assert {"2024-01-05", "10:30", "limit_x", "42.5"} <= values
        assert not ({"2024", "01", "05", "10", "30"} & values)
        assert decode_with_anchor(result.prompt) == "Review on 2024-01-05 at 10:30 for limit_x = 42.5"
    off = ObfusAIComponent("low", seed="s", detectors=["numbers"]).obfuscate("on 2024-01-05")
    assert "2024" in off.mapping.values()  # opting out restores the old behaviour
