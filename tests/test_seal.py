from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from obfusai import ObfusAIComponent
from obfusai.anchor import decode_with_anchor
from obfusai.cli import main

SYSTEM = "Rule: BUY when edge_bps > 35. Codename project_falcon."


# ----------------------------------------------------------------- seal
seal_mod = pytest.importorskip("cryptography")
from obfusai.seal import (  # noqa: E402
    MARKER, SealError, generate_key, is_sealed, seal, unseal, unseal_payload, unseal_text,
)


def test_seal_round_trip_and_blob_hides_the_plaintext() -> None:
    key = generate_key()
    blob = seal(SYSTEM, key=key, label="skill-desk")
    assert blob.startswith(MARKER) and "edge_bps" not in blob and "\n" not in blob
    assert unseal(blob, key=key, expect_label="skill-desk") == SYSTEM
    assert seal(SYSTEM, key=key) != seal(SYSTEM, key=key)  # fresh nonce every time
    assert unseal(seal("héllo 世界 🌀", key=key), key=key) == "héllo 世界 🌀"
    assert is_sealed(blob) and not is_sealed(SYSTEM) and not is_sealed(None)


def test_seal_passphrase_mode() -> None:
    blob = seal(SYSTEM, passphrase="correct horse battery staple")
    assert unseal(blob, passphrase="correct horse battery staple") == SYSTEM
    with pytest.raises(SealError):
        unseal(blob, passphrase="wrong")
    with pytest.raises(SealError, match="passphrase"):
        unseal(blob, key=generate_key())


def test_seal_rejects_wrong_key_label_tampering_and_misuse() -> None:
    key = generate_key()
    blob = seal(SYSTEM, key=key, label="a")
    with pytest.raises(SealError):
        unseal(blob, key=generate_key())
    with pytest.raises(SealError, match="different label"):
        unseal(blob, key=key, expect_label="b")
    tampered = blob[:-4] + ("AAAA" if not blob.endswith("AAAA") else "BBBB")
    with pytest.raises(SealError):
        unseal(tampered, key=key)
    relabelled = blob.replace("AQAB", "AQAC", 1)  # flipping a header byte breaks authentication
    with pytest.raises(SealError):
        unseal(relabelled, key=key)
    for bad in ("not sealed", MARKER, MARKER + "!!!", MARKER + "AAAA"):
        with pytest.raises(SealError):
            unseal(bad, key=key)
    with pytest.raises(SealError):
        seal("x")  # neither key nor passphrase
    with pytest.raises(SealError):
        seal("x", key=key, passphrase="p")  # both
    with pytest.raises(SealError):
        seal("x", key="short")
    with pytest.raises(SealError):
        seal("x", key=key, label="é" * 200)


def test_unseal_text_and_payload_walk_nested_structures() -> None:
    key = generate_key()
    a, b = seal("alpha", key=key), seal("beta", key=key)
    assert unseal_text(f"System: {a} and {b}!", key=key) == "System: alpha and beta!"
    payload = {"messages": [{"role": "system", "content": a}, {"role": "user", "content": "hi"}],
               "tools": [{"function": {"description": f"uses {b}"}}], "n": 3}
    out = unseal_payload(payload, key=key)
    assert out["messages"][0]["content"] == "alpha" and out["tools"][0]["function"]["description"] == "uses beta"
    assert out["n"] == 3 and payload["messages"][0]["content"] == a  # input untouched


# ----------------------------------------------------------------- CLI
def test_cli_keygen_seal_unseal_and_canary(tmp_path, capsys, monkeypatch) -> None:
    assert main(["keygen"]) == 0
    key = capsys.readouterr().out.strip()
    monkeypatch.setenv("OBFUSAI_SEAL_KEY", key)
    prompt, sealed = tmp_path / "skill.md", tmp_path / "skill.sealed"
    prompt.write_text(SYSTEM + "\n", encoding="utf-8")
    assert main(["seal", str(prompt), "-o", str(sealed), "--label", "skill-desk"]) == 0
    assert "edge_bps" not in sealed.read_text(encoding="utf-8")
    assert main(["unseal", str(sealed), "--expect-label", "skill-desk"]) == 0
    assert capsys.readouterr().out.strip() == SYSTEM
    assert main(["unseal", str(sealed), "--expect-label", "other"]) == 2  # error, no plaintext
    monkeypatch.setenv("OBFUSAI_SEAL_KEY", "A" * 43)
    assert main(["unseal", str(sealed)]) == 2
    monkeypatch.delenv("OBFUSAI_SEAL_KEY")
    assert main(["seal", str(prompt)]) == 2  # no key available
    capsys.readouterr()

    monkeypatch.setenv("OBFUSAI_CANARY_SECRET", "s3cret")
    marked = tmp_path / "marked.md"
    assert main(["canary", "embed", str(prompt), "--recipient", "acme", "-o", str(marked)]) == 0
    assert main(["canary", "find", str(marked), "--recipients", "acme", "globex"]) == 0
    assert capsys.readouterr().out.strip() == "acme"
    assert main(["canary", "find", str(prompt), "--recipients", "acme"]) == 1


# ----------------------------------------------------------------- gateway: clients only ever hold ciphertext
def _gateway(upstream, key):
    from obfusai.proxy import make_server

    server = make_server(ObfusAIComponent("low", seed="s"), "127.0.0.1", 0, upstream_openai=upstream.url,
                         upstream_anthropic=upstream.url, log=lambda m: None, seal_key=key)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=10)


def test_gateway_unseals_obfuscates_and_forwards() -> None:
    from obfusai.testing import AliasEchoServer

    key = generate_key()
    sealed = seal(SYSTEM, key=key, label="skill-desk")
    body = {"model": "m", "messages": [{"role": "system", "content": sealed}, {"role": "user", "content": "go"}]}
    with AliasEchoServer() as upstream:
        server, base = _gateway(upstream, key)
        try:
            with _post(base + "/v1/chat/completions", body) as resp:
                reply = json.loads(resp.read())
        finally:
            server.shutdown()
            server.server_close()
        sent = upstream.requests[0]["body"]["messages"][0]["content"]
        assert MARKER not in sent and "edge_bps" not in sent.split("<obfusai-context")[0]
        assert decode_with_anchor(sent) == SYSTEM  # decrypted, then obfuscated, then forwarded
        assert reply["choices"][0]["message"]["content"] == "The metric edge_bps matters."
    assert MARKER in json.dumps(body) and "edge_bps" not in json.dumps(body)  # what the client held


def test_gateway_fails_closed_without_the_right_key() -> None:
    from obfusai.testing import AliasEchoServer

    sealed = seal(SYSTEM, key=generate_key())
    body = {"model": "m", "messages": [{"role": "system", "content": sealed}, {"role": "user", "content": "go"}]}
    with AliasEchoServer() as upstream:
        for gateway_key in (None, generate_key()):
            server, base = _gateway(upstream, gateway_key)
            try:
                with pytest.raises(urllib.error.HTTPError) as excinfo:
                    _post(base + "/v1/chat/completions", body)
                assert excinfo.value.code == 400
                assert sealed not in excinfo.value.read().decode()  # the error never echoes the blob
            finally:
                server.shutdown()
                server.server_close()
        assert upstream.requests == []  # nothing was forwarded
