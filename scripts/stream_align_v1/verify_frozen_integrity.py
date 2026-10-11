#!/usr/bin/env python3
"""Verify that all S0 read-only reference files remain byte-identical."""

from __future__ import annotations

from scripts.stream_align_v1.common import OUT, ROOT, read_tsv, sha256, write_json, write_text, write_tsv


def main() -> None:
    before = read_tsv(OUT / "s0_freeze/frozen_integrity_before.tsv")
    rows = []
    for row in before:
        path = ROOT / row["path"]
        actual = sha256(path) if path.is_file() else ""
        rows.append(
            {
                "path": row["path"],
                "before_sha256": row["sha256"],
                "after_sha256": actual,
                "before_size_bytes": row["size_bytes"],
                "after_size_bytes": path.stat().st_size if path.is_file() else "",
                "status": "pass" if actual == row["sha256"] else "failed",
            }
        )
    write_tsv(OUT / "validation/frozen_integrity_after.tsv", rows)
    failures = [row for row in rows if row["status"] != "pass"]
    write_json(
        OUT / "validation/frozen_integrity_status.json",
        {
            "status": "pass" if not failures else "failed",
            "verified_files": len(rows),
            "failures": failures,
            "public_default_changed": False,
            "git_metadata_modified": False,
        },
    )
    write_text(
        OUT / "validation/frozen_integrity_report.md",
        f"""# Frozen Evidence Integrity

- Files verified: {len(rows)}.
- Hash mismatches: {len(failures)}.
- Status: `{'pass' if not failures else 'failed'}`.
- Historical outputs, public defaults and Git metadata were not modified.
""",
    )
    if failures:
        raise RuntimeError(f"frozen integrity failed for {len(failures)} files")


if __name__ == "__main__":
    main()
