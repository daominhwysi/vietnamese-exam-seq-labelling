#!/usr/bin/env python3
"""Recursively replace a text fragment in files under a root directory.

This is useful for normalizing generated Markdown/HTML outputs, for example
replacing ``&nbsp;`` with regular spaces inside ``real_data_annotator/out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SUFFIXES = (".md", ".html", ".json", ".txt")


def iter_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and (not suffixes or path.suffix.lower() in suffixes):
            files.append(path)
    return files


def replace_in_file(path: Path, old: str, new: str) -> bool:
    original = path.read_bytes()
    text = original.decode("utf-8")
    updated = text.replace(old, new)
    if updated == text:
        return False
    path.write_bytes(updated.encode("utf-8"))
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively replace text in files under a directory."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="real_data_annotator/out",
        help="Root directory to scan (default: real_data_annotator/out).",
    )
    parser.add_argument(
        "--old",
        default="&nbsp;",
        help="Text fragment to replace (default: &nbsp;).",
    )
    parser.add_argument(
        "--new",
        default=" ",
        help="Replacement text (default: a single space).",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        help=(
            "File suffix to include, e.g. .md. Repeat to allow multiple suffixes. "
            "Defaults to .md, .html, .json, and .txt."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    suffixes = tuple(
        suffix if suffix.startswith(".") else f".{suffix}"
        for suffix in (args.suffixes or DEFAULT_SUFFIXES)
    )

    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    changed = 0
    for file_path in iter_files(root, suffixes):
        if replace_in_file(file_path, args.old, args.new):
            changed += 1

    print(f"Updated {changed} file(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
