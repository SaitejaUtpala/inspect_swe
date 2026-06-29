#!/usr/bin/env python
"""Peek into nested Inspect .eval logs without opening the full file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A .eval file or any parent log directory.")
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Only include .eval paths containing this text. Can be repeated.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching .eval files instead of reading one.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which matching .eval to read, newest first.",
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--messages", type=int, default=8)
    parser.add_argument("--chars", type=int, default=1200)
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--tools-only", action="store_true")
    args = parser.parse_args()

    files = _eval_files(Path(args.path), args.contains)
    if not files:
        raise SystemExit("No matching .eval files found.")

    if args.list:
        for index, path in enumerate(files):
            print(f"[{index}] {path}")
        return 0

    if args.index >= len(files):
        raise SystemExit(f"--index {args.index} out of range; found {len(files)} files.")

    _print_eval(files[args.index], args)
    return 0


def _eval_files(root: Path, contains: list[str]) -> list[Path]:
    if root.is_file():
        files = [root] if root.suffix == ".eval" else []
    else:
        files = sorted(root.rglob("*.eval"), key=lambda path: path.stat().st_mtime, reverse=True)
    for token in contains:
        files = [path for path in files if token in str(path)]
    return files


def _print_eval(path: Path, args: argparse.Namespace) -> None:
    log = read_eval_log(str(path))
    samples = list(log.samples or [])

    print(f"LOG: {path}")
    print(f"run_id: {log.eval.run_id}")
    print(f"task: {log.eval.task}")
    print(f"model: {log.eval.model}")
    print(f"samples: {len(samples)}")

    if log.results is not None:
        print("\nRESULTS")
        print(_short(log.results, args.chars))

    if args.metadata:
        print("\nEVAL METADATA")
        print(_short(log.eval.metadata, args.chars))

    for sample in samples[: args.samples]:
        _print_sample(sample, args)


def _print_sample(sample: Any, args: argparse.Namespace) -> None:
    print("\n" + "=" * 100)
    print(f"SAMPLE id={sample.id} uuid={sample.uuid}")
    print(f"scores={_short(getattr(sample, 'scores', None), args.chars)}")

    if args.metadata:
        print("\nSAMPLE METADATA")
        print(_short(getattr(sample, "metadata", None), args.chars))

    shown = 0
    for message in getattr(sample, "messages", []) or []:
        role = getattr(message, "role", type(message).__name__)
        function = getattr(message, "function", None)
        if args.tools_only and role != "tool":
            continue
        suffix = f" function={function}" if function else ""
        print(f"\n[{role}{suffix}]")
        print(_short(getattr(message, "text", ""), args.chars))
        shown += 1
        if shown >= args.messages:
            break


def _short(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [truncated {len(text) - limit} chars]"


if __name__ == "__main__":
    raise SystemExit(main())
