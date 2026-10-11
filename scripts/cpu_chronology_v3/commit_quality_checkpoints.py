#!/usr/bin/env python
"""Validate and commit completed V5 local checkpoints without recomputation."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_chronology_v3/06_validation"
CHECKPOINT = Path(tempfile.gettempdir()) / "mdcorealign_cpu_chronology_v3_checkpoints"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    expected_quality = {
        "fresh_mdcath_quality.tsv": 512 * 5,
        "fresh_long_quality.tsv": 16 * 5,
        "historical_stress_quality.tsv": 34 * 5,
    }
    for name, expected in expected_quality.items():
        rows = read_tsv(OUT / name)
        if len(rows) != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, found {len(rows)}")
        cases = {row["case_id"] for row in rows}
        if len(cases) * 5 != expected:
            raise RuntimeError(f"{name}: invalid case/method cardinality")

    sources = [OUT / "adaptive_refinement_paths.tsv"]
    sources.extend(sorted(CHECKPOINT.glob("adaptive_refinement_paths_*.tsv")))
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for source in sources:
        if not source.exists():
            continue
        for row in read_tsv(source):
            merged[(row["panel"], row["case_id"])] = row
    expected_by_panel = {"fresh_mdcath": 512, "fresh_long": 16, "historical": 34}
    observed = {
        panel: sum(key[0] == panel for key in merged)
        for panel in expected_by_panel
    }
    if observed != expected_by_panel:
        raise RuntimeError(f"adaptive path cardinality mismatch: {observed}")
    rows = [merged[key] for key in sorted(merged)]
    write_tsv(OUT / "adaptive_refinement_paths.tsv", rows)
    print({"quality_rows_validated": sum(expected_quality.values()), "adaptive_paths": len(rows), **observed})


if __name__ == "__main__":
    main()
