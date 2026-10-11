#!/usr/bin/env python3
"""Scan a release tree for CJK characters in filenames and public text."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yml",
    ".yaml",
}
SKIPPED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def scan_tree(root: Path) -> list[dict[str, object]]:
    """Return one scan record for every candidate file under ``root``."""

    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIPPED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        filename_match = CJK.search(relative)
        content_match = None
        status = "pass"
        note = ""
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                text = ""
                note = "binary_or_non_utf8_skipped"
            content_match = CJK.search(text)
        else:
            note = "non_text_content_skipped"
        if filename_match or content_match:
            status = "fail"
            note = "CJK character detected"
        rows.append(
            {
                "path": relative,
                "filename_status": "fail" if filename_match else "pass",
                "content_status": "fail" if content_match else "pass",
                "overall_status": status,
                "notes": note,
            }
        )
    return rows


def main() -> None:
    """Run the scanner and optionally write a TSV report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    rows = scan_tree(arguments.root)
    failures = [row for row in rows if row["overall_status"] == "fail"]
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    print(f"scanned_files={len(rows)} failures={len(failures)}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
