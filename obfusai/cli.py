"""Command-line interface: ``obfusai obfuscate | restore | scan`` (also ``python -m obfusai``)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence, TextIO

from . import __version__
from .anchor import LANGUAGES, STYLES
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obfusai", description="Obfuscate LLM prompts while keeping them executable by LLMs."
    )
    parser.add_argument("--version", action="version", version=f"obfusai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    obf = sub.add_parser("obfuscate", help="obfuscate a prompt file (or '-' for stdin)")
    obf.add_argument("file", nargs="?", default="-")
    obf.add_argument("-l", "--level", choices=("low", "medium", "high"), default="medium")
    obf.add_argument("--seed", help="secret seed for reproducible output")
    obf.add_argument("-k", "--keyword", action="append", default=[], help="always obfuscate this term")
    obf.add_argument("-p", "--protect", action="append", default=[], help="never obfuscate this term")
    obf.add_argument("-r", "--rule", action="append", default=[], help="always obfuscate matches of this regex")
    obf.add_argument("-d", "--detector", action="append", default=[], choices=DETECTORS, help="add a detector")
    obf.add_argument("--alias-mode", choices=("sequential", "stable"), default="sequential")
    obf.add_argument("--lang", choices=LANGUAGES, default="en", help="anchor instruction language")
    obf.add_argument("--style", choices=STYLES, default="full", help="anchor verbosity")
    obf.add_argument("--position", choices=("append", "prepend"), default="append")
    obf.add_argument("--mapping-out", metavar="FILE", help="write the alias table (sensitive!) as JSON")
    obf.add_argument("--json", action="store_true", help="print a JSON document instead of the prompt")

    res = sub.add_parser("restore", help="decode aliases in text using a mapping file")
    res.add_argument("file", nargs="?", default="-")
    res.add_argument("-m", "--mapping", required=True, metavar="FILE")

    scan = sub.add_parser("scan", help="check model output for prompt leakage (exit 1 if found)")
    scan.add_argument("file", nargs="?", default="-")
    scan.add_argument("-m", "--mapping", metavar="FILE", help="alias table to also look for")
    scan.add_argument("--no-values", action="store_true", help="ignore original values in the output")
    return parser


def main(argv: Optional[Sequence[str]] = None, stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "obfuscate":
            engine = ObfusAIComponent(
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
            result = engine.obfuscate(_read(args.file, stdin))
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
        if args.command == "restore":
            stdout.write(ObfusAIComponent.deobfuscate(_read(args.file, stdin), _load_mapping(args.mapping)))
            return 0
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
