"""Command-line interface: ``obfusai obfuscate | restore | scan | audit | serve | keygen | seal | unseal | canary``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, TextIO

from . import __version__
from .anchor import LANGUAGES, STYLES
from .audit import audit
from .core import DETECTORS, ObfusAIComponent
from .guard import LeakDetector


def _read(path: str, stdin: TextIO) -> str:
    if path == "-":
        return stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping = data.get("mapping", data) if isinstance(data, dict) else None
    if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
        raise ValueError(f"{path}: expected a JSON object mapping aliases to strings")
    return mapping


def _add_engine_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-l", "--level", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--seed", help="secret seed for reproducible output (serve: also read from $OBFUSAI_SEED)")
    parser.add_argument("-k", "--keyword", action="append", default=[], help="always obfuscate this term")
    parser.add_argument("-p", "--protect", action="append", default=[], help="never obfuscate this term")
    parser.add_argument("-r", "--rule", action="append", default=[], help="always obfuscate matches of this regex")
    parser.add_argument("-d", "--detector", action="append", default=[], choices=DETECTORS, help="add a detector")
    parser.add_argument("--alias-mode", choices=("sequential", "stable"), default="sequential")
    parser.add_argument("--lang", choices=LANGUAGES, default="en", help="anchor instruction language")
    parser.add_argument("--style", choices=STYLES, default="full", help="anchor verbosity")
    parser.add_argument("--position", choices=("append", "prepend"), default="append")


def _component(args: argparse.Namespace) -> ObfusAIComponent:
    return ObfusAIComponent(
        args.level,
        seed=args.seed,
        keywords=args.keyword,
        protected=args.protect,
        rules=args.rule,
        extra_detectors=args.detector,
        alias_mode=args.alias_mode,
        anchor_position=args.position,
        anchor_language=args.lang,
        anchor_style=args.style,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obfusai", description="Obfuscate LLM prompts while keeping them executable by LLMs."
    )
    parser.add_argument("--version", action="version", version=f"obfusai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    obf = sub.add_parser("obfuscate", help="obfuscate a prompt file (or '-' for stdin)")
    obf.add_argument("file", nargs="?", default="-")
    _add_engine_options(obf)
    obf.add_argument("--mapping-out", metavar="FILE", help="write the alias table (sensitive!) as JSON")
    obf.add_argument("--json", action="store_true", help="print a JSON document instead of the prompt")

    res = sub.add_parser("restore", help="decode aliases in text using a mapping file")
    res.add_argument("file", nargs="?", default="-")
    res.add_argument("-m", "--mapping", required=True, metavar="FILE")

    scan = sub.add_parser("scan", help="check model output for prompt leakage (exit 1 if found)")
    scan.add_argument("file", nargs="?", default="-")
    scan.add_argument("-m", "--mapping", metavar="FILE", help="alias table to also look for")
    scan.add_argument("--no-values", action="store_true", help="ignore original values in the output")

    aud = sub.add_parser("audit", help="find personal data and credentials in a prompt (exit 1 if found)")
    aud.add_argument("file", nargs="?", default="-")
    aud.add_argument("-d", "--detector", action="append", default=[], choices=DETECTORS,
                     help="detector to run (repeatable; default: emails, ips, secrets and personal-data detectors)")

    sub.add_parser("keygen", help="print a new random 256-bit key for `seal` (keep it in a secret manager)")

    for name, summary in (("seal", "encrypt a prompt file (AES-256-GCM) so no plaintext is stored"),
                          ("unseal", "decrypt a sealed prompt file")):
        sealer = sub.add_parser(name, help=summary)
        sealer.add_argument("file", nargs="?", default="-")
        sealer.add_argument("-o", "--output", metavar="FILE", help="write here instead of stdout")
        sealer.add_argument("--key-env", default="OBFUSAI_SEAL_KEY", help="environment variable holding the key")
        sealer.add_argument("--passphrase-env", help="use a passphrase from this environment variable instead of a key")
        if name == "seal":
            sealer.add_argument("--label", default="", help="bind the blob to a purpose, e.g. the skill name")
        else:
            sealer.add_argument("--expect-label", help="refuse blobs sealed for another label")

    can = sub.add_parser("canary", help="per-recipient canary tokens to trace which prompt copy leaked")
    can_sub = can.add_subparsers(dest="canary_command", required=True)
    embed = can_sub.add_parser("embed", help="embed tokens for one recipient")
    embed.add_argument("file", nargs="?", default="-")
    embed.add_argument("--recipient", required=True)
    embed.add_argument("--secret-env", default="OBFUSAI_CANARY_SECRET")
    embed.add_argument("--count", type=int, default=3)
    embed.add_argument("-o", "--output", metavar="FILE")
    find = can_sub.add_parser("find", help="which recipients' tokens appear in a text (exit 1 if none)")
    find.add_argument("file", nargs="?", default="-")
    find.add_argument("--recipients", nargs="+", required=True)
    find.add_argument("--secret-env", default="OBFUSAI_CANARY_SECRET")
    find.add_argument("--count", type=int, default=3)

    srv = sub.add_parser(
        "serve",
        help="run a local gateway so any language's SDK is protected by pointing its base URL at it",
    )
    _add_engine_options(srv)
    srv.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only; the gateway has no auth)")
    srv.add_argument("--port", type=int, default=8787)
    srv.add_argument("--upstream-openai", default="https://api.openai.com", help="OpenAI-compatible upstream base URL")
    srv.add_argument("--upstream-anthropic", default="https://api.anthropic.com", help="Anthropic upstream base URL")
    srv.add_argument("--strict", action="store_true", help="reject POSTs to endpoints the gateway cannot protect")
    return parser


def _env_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"environment variable {name} is not set")
    return value


def _seal_secret(args: argparse.Namespace) -> dict:
    if args.passphrase_env:
        return {"passphrase": _env_secret(args.passphrase_env)}
    return {"key": _env_secret(args.key_env)}


def _write(text: str, output: Optional[str], stdout: TextIO) -> None:
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        stdout.write(text if text.endswith("\n") else text + "\n")


def main(argv: Optional[Sequence[str]] = None, stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "obfuscate":
            result = _component(args).obfuscate(_read(args.file, stdin))
            if args.mapping_out:
                with open(args.mapping_out, "w", encoding="utf-8") as handle:
                    json.dump({"session_id": result.session_id, "mapping": dict(result.mapping)}, handle, ensure_ascii=False, indent=2)
            if args.json:
                doc = {
                    "prompt": result.prompt,
                    "session_id": result.session_id,
                    "aliases": len(result.mapping),
                    "growth": round(result.stats.growth, 3) if result.stats else 1.0,
                }
                stdout.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
            else:
                stdout.write(result.prompt + ("\n" if not result.prompt.endswith("\n") else ""))
            return 0
        if args.command == "keygen":
            from .seal import generate_key

            stdout.write(generate_key() + "\n")
            return 0
        if args.command in ("seal", "unseal"):
            from .seal import SealError, seal, unseal

            text = _read(args.file, stdin)
            try:
                if args.command == "seal":
                    _write(seal(text.rstrip("\n"), label=args.label, **_seal_secret(args)), args.output, stdout)
                else:
                    _write(unseal(text.strip(), expect_label=args.expect_label, **_seal_secret(args)), args.output, stdout)
            except SealError as exc:
                raise ValueError(str(exc)) from None
            return 0
        if args.command == "canary":
            from .canary import embed_canaries, find_recipients

            secret = _env_secret(args.secret_env)
            text = _read(args.file, stdin)
            if args.canary_command == "embed":
                _write(embed_canaries(text, args.recipient, secret, count=args.count), args.output, stdout)
                return 0
            hits = find_recipients(text, args.recipients, secret, count=args.count)
            stdout.write("\n".join(hits) + ("\n" if hits else "no match\n"))
            return 0 if hits else 1
        if args.command == "serve":
            from .proxy import serve

            args.seed = args.seed or os.environ.get("OBFUSAI_SEED")
            if args.seed is None:
                sys.stderr.write("obfusai: warning: no --seed / $OBFUSAI_SEED, aliases change on every restart\n")
            if args.host not in ("127.0.0.1", "localhost", "::1"):
                sys.stderr.write("obfusai: warning: the gateway has no authentication; bind to localhost unless the network is trusted\n")
            serve(
                _component(args), args.host, args.port,
                upstream_openai=args.upstream_openai, upstream_anthropic=args.upstream_anthropic, strict=args.strict,
                seal_key=os.environ.get("OBFUSAI_SEAL_KEY"), seal_passphrase=os.environ.get("OBFUSAI_SEAL_PASSPHRASE"),
            )
            return 0
        if args.command == "restore":
            stdout.write(ObfusAIComponent.deobfuscate(_read(args.file, stdin), _load_mapping(args.mapping)))
            return 0
        if args.command == "audit":
            found = audit(_read(args.file, stdin), args.detector or None)
            for finding in found.findings:
                stdout.write(f"{finding.detector}\t{finding.start}-{finding.end}\t{finding.preview}\n")
            stdout.write("clean\n" if found.clean else f"SENSITIVE VALUES FOUND: {found.counts}\n")
            return 0 if found.clean else 1
        mapping = _load_mapping(args.mapping) if args.mapping else {}
        report = LeakDetector(mapping, check_values=not args.no_values).scan(_read(args.file, stdin))
        for finding in report.findings:
            stdout.write(f"{finding.kind}: {finding.detail}\n")
        stdout.write("LEAK DETECTED\n" if report.leaked else "clean\n")
        return 1 if report.leaked else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"obfusai: error: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
