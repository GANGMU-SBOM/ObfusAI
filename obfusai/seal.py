"""Encryption at rest for skill prompts (AES-256-GCM): keep the plaintext out of repos, packages and storage.

What this protects, and what it cannot:

* **Protects** a prompt file, package or backup against anyone who gets the artifact but not the key
  (a leaked repository, a shared archive, a build cache, a public marketplace listing).
* **Cannot protect** against whoever *runs* the prompt.  A model needs the plaintext, so it exists in
  memory wherever you decrypt it and in the request to the model provider.  If the prompt runs on a machine
  you do not control, the person there can read it.  Keep the key, and the decryption, on infrastructure
  you control (for example behind ``obfusai serve``, which unseals placeholders server-side).

A sealed prompt is one URL-safe line, ``obfusai-sealed:v1:...``, that can sit anywhere a string can (a
file, a config value, a ``messages[].content``).  The label is authenticated, so a blob sealed for one
skill cannot be swapped for another.  Requires the ``cryptography`` package (``pip install obfusai[seal]``);
passphrase keys are derived with scrypt from the standard library.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from typing import Any, Optional, Pattern, Union

__all__ = [
    "MARKER",
    "SealError",
    "generate_key",
    "is_sealed",
    "seal",
    "unseal",
    "unseal_payload",
    "unseal_text",
]

MARKER = "obfusai-sealed:v1:"
_VERSION = 1
_KDF_RAW, _KDF_SCRYPT = 0, 1
_SCRYPT = dict(n=2**15, r=8, p=1, dklen=32, maxmem=2**26)
_BLOB_RE: Pattern[str] = re.compile(re.escape(MARKER) + r"[A-Za-z0-9_-]+")


class SealError(ValueError):
    """Wrong key or passphrase, wrong label, malformed or tampered blob."""


def _aesgcm() -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - exercised only without cryptography
        raise ImportError("obfusai.seal requires the 'cryptography' package: pip install obfusai[seal]") from exc
    return AESGCM


def generate_key() -> str:
    """A fresh random 256-bit key as URL-safe base64 (store it in a secret manager, never in the repo)."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def _raw_key(key: Union[str, bytes]) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(key if isinstance(key, bytes) else key.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        raise SealError("key is not valid URL-safe base64") from None
    if len(raw) != 32:
        raise SealError("key must decode to exactly 32 bytes (use generate_key())")
    return raw


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _check_secret(key: Optional[Union[str, bytes]], passphrase: Optional[str]) -> None:
    if (key is None) == (passphrase is None):
        raise SealError("pass exactly one of key= or passphrase=")


def seal(
    text: str,
    *,
    key: Optional[Union[str, bytes]] = None,
    passphrase: Optional[str] = None,
    label: str = "",
) -> str:
    """Encrypt ``text`` (authenticated).  ``label`` (e.g. the skill name) is bound to the ciphertext."""
    _check_secret(key, passphrase)
    label_bytes = label.encode("utf-8")
    if len(label_bytes) > 255:
        raise SealError("label is limited to 255 bytes")
    if passphrase is not None:
        salt = os.urandom(16)
        secret = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, **_SCRYPT)  # type: ignore[arg-type]
        header = bytes([_VERSION, _KDF_SCRYPT, len(label_bytes)]) + label_bytes + salt
    else:
        secret = _raw_key(key)  # type: ignore[arg-type]
        header = bytes([_VERSION, _KDF_RAW, len(label_bytes)]) + label_bytes
    nonce = os.urandom(12)
    ciphertext = _aesgcm()(secret).encrypt(nonce, text.encode("utf-8"), header)  # the header is authenticated
    return MARKER + _b64(header + nonce + ciphertext)


def unseal(
    blob: str,
    *,
    key: Optional[Union[str, bytes]] = None,
    passphrase: Optional[str] = None,
    expect_label: Optional[str] = None,
) -> str:
    """Decrypt a sealed blob.  With ``expect_label`` the blob must have been sealed for that label."""
    _check_secret(key, passphrase)
    if not blob.startswith(MARKER):
        raise SealError("not a sealed prompt")
    try:
        data = _unb64(blob[len(MARKER):])
        version, kdf, label_len = data[0], data[1], data[2]
        label = data[3 : 3 + label_len]
        position = 3 + label_len
        if version != _VERSION or kdf not in (_KDF_RAW, _KDF_SCRYPT):
            raise SealError("unsupported sealed-prompt version")
        if kdf == _KDF_SCRYPT:
            salt = data[position : position + 16]
            position += 16
        header, nonce, ciphertext = data[:position], data[position : position + 12], data[position + 12 :]
    except (IndexError, ValueError):
        raise SealError("malformed sealed prompt") from None
    if expect_label is not None and label != expect_label.encode("utf-8"):
        raise SealError("sealed for a different label")
    if kdf == _KDF_SCRYPT:
        if passphrase is None:
            raise SealError("this blob was sealed with a passphrase")
        secret = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, **_SCRYPT)  # type: ignore[arg-type]
    else:
        if key is None:
            raise SealError("this blob was sealed with a key")
        secret = _raw_key(key)
    try:
        return _aesgcm()(secret).decrypt(nonce, ciphertext, header).decode("utf-8")
    except Exception as exc:  # cryptography.exceptions.InvalidTag and friends
        if isinstance(exc, ImportError):  # pragma: no cover
            raise
        raise SealError("wrong key or passphrase, or the blob was modified") from None


def is_sealed(text: Any) -> bool:
    """True when ``text`` contains at least one sealed blob."""
    return isinstance(text, str) and MARKER in text


def unseal_text(text: str, **secret: Any) -> str:
    """Replace every sealed blob embedded in ``text`` by its plaintext (other text is left alone)."""
    return _BLOB_RE.sub(lambda m: unseal(m.group(0), **secret), text)


def unseal_payload(value: Any, **secret: Any) -> Any:
    """Copy of a JSON-like structure with every sealed blob in its strings decrypted."""
    if isinstance(value, str):
        return unseal_text(value, **secret) if MARKER in value else value
    if isinstance(value, dict):
        return {k: unseal_payload(v, **secret) for k, v in value.items()}
    if isinstance(value, list):
        return [unseal_payload(v, **secret) for v in value]
    return value
