#!/usr/bin/env python
"""Prospectively lock a stratified 128-domain timing subset."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_chronology_v3"
SOURCE = OUT / "05_fresh_panels/fresh_mdcath_512.tsv"
TARGET = OUT / "05_fresh_panels/fresh_mdcath_performance_128.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def stable_score(case_id: str) -> str:
    return hashlib.sha256(f"cpu-chronology-v3-performance:{case_id}".encode()).hexdigest()


def main() -> None:
    if TARGET.exists():
        raise RuntimeError(f"locked subset already exists: {TARGET}")
    rows = read_tsv(SOURCE)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["stratum"], []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: stable_score(row["case_id"]))
    allocation = {key: 1 for key in groups}
    remaining = 128 - len(allocation)
    while remaining:
        eligible = [key for key, group in groups.items() if allocation[key] < len(group)]
        key = max(
            eligible,
            key=lambda item: (
                len(groups[item]) / allocation[item],
                hashlib.sha256(item.encode()).hexdigest(),
            ),
        )
        allocation[key] += 1
        remaining -= 1
    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][: allocation[key]])
    selected.sort(key=lambda row: stable_score(row["case_id"]))
    if len(selected) != 128 or len({row["case_id"] for row in selected}) != 128:
        raise RuntimeError("invalid performance subset cardinality")
    write_tsv(TARGET, selected)
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    payload = {
        "source_panel": str(SOURCE.relative_to(ROOT)),
        "source_count": len(rows),
        "selected_count": len(selected),
        "strata_represented": len(groups),
        "selection_rule": "one_per_stratum_then_deterministic_size_balanced_allocation",
        "quality_results_inspected": False,
        "sha256": digest,
    }
    (OUT / "05_fresh_panels/fresh_mdcath_performance_128_lock.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
