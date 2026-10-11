#!/usr/bin/env python
"""Select deterministic CATH-balanced mdCATH domain subsets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT, read_tsv, trajectory_id, write_tsv


TARGETS = (5, 10, 20)
SEED = 20260625


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain-manifest",
        type=Path,
        default=RESULT_ROOT / "manifest/mdcath_domain_hierarchy_manifest.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "subsets")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def complete(row: dict[str, str]) -> bool:
    """Return whether a domain row is eligible for balanced subset selection."""

    return row.get("include_available") in {"1", "1.0"} and all(
        row.get(key, "NA") not in {"", "NA"}
        for key in ("cath_class", "cath_architecture", "cath_topology", "cath_homologous_superfamily", "h5_path")
    )


def stable_shuffle(rows: list[dict[str, str]], seed: int, class_id: str, target: int) -> list[dict[str, str]]:
    """Shuffle deterministically while preserving reproducibility."""

    rng = random.Random(f"{seed}:{class_id}:{target}")
    rows = list(rows)
    rng.shuffle(rows)
    return rows


def greedy_select(rows: list[dict[str, str]], target: int, seed: int, class_id: str) -> list[tuple[dict[str, str], str]]:
    """Select domains that add hierarchy diversity before random fill."""

    selected: list[tuple[dict[str, str], str]] = []
    remaining = sorted(rows, key=lambda r: (r["cath_architecture"], r["cath_topology"], r["cath_homologous_superfamily"], r["domain_id"]))

    def best_candidate(candidates: list[dict[str, str]], reason: str) -> dict[str, str]:
        seen_arch = {row["cath_architecture"] for row, _ in selected}
        seen_top = {row["cath_topology"] for row, _ in selected}
        seen_hsf = {row["cath_homologous_superfamily"] for row, _ in selected}
        shuffled = stable_shuffle(candidates, seed, f"{class_id}:{reason}:{len(selected)}", target)
        return max(
            shuffled,
            key=lambda row: (
                row["cath_architecture"] not in seen_arch,
                row["cath_topology"] not in seen_top,
                row["cath_homologous_superfamily"] not in seen_hsf,
                row["domain_id"],
            ),
        )

    def add_new_value(field: str, reason: str) -> None:
        nonlocal remaining
        while len(selected) < target:
            seen = {row[field] for row, _ in selected}
            candidates = [row for row in remaining if row[field] not in seen]
            if not candidates:
                break
            candidate = best_candidate(candidates, reason)
            selected.append((candidate, reason))
            remaining = [row for row in remaining if row["domain_id"] != candidate["domain_id"]]

    add_new_value("cath_architecture", "adds_new_architecture")
    add_new_value("cath_topology", "adds_new_topology")
    add_new_value("cath_homologous_superfamily", "adds_new_homologous_superfamily")
    if len(selected) < target:
        for row in stable_shuffle(remaining, seed, class_id, target):
            if len(selected) >= target:
                break
            selected.append((row, "deterministic_random_fill"))
    return selected


def build_subset_rows(rows: list[dict[str, str]], target: int, seed: int) -> list[dict[str, str]]:
    """Build one subset table."""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if complete(row):
            grouped[row["cath_class"]].append(row)
    subset_rows: list[dict[str, str]] = []
    subset_name = f"mdcath_cath_balanced_{target}_per_class"
    for class_id in sorted(grouped, key=str):
        selected = greedy_select(grouped[class_id], target, seed, class_id)
        for rank, (row, reason) in enumerate(selected, start=1):
            domain_id = row["domain_id"]
            subset_rows.append(
                {
                    "subset_name": subset_name,
                    "domain_id": domain_id,
                    "trajectory_id": trajectory_id(domain_id, 320, 1),
                    "cath_class": row["cath_class"],
                    "cath_architecture": row["cath_architecture"],
                    "cath_topology": row["cath_topology"],
                    "cath_homologous_superfamily": row["cath_homologous_superfamily"],
                    "cath_code": row["cath_code"],
                    "h5_path": row["h5_path"],
                    "h5_size_bytes": row["h5_size_bytes"],
                    "selected_temperature_K": 320,
                    "selected_replica_index_zero_based": 0,
                    "selected_replica_label_one_based": 1,
                    "selection_rank_within_class": rank,
                    "selection_reason": reason,
                    "fallback_used": 0,
                    "fallback_reason": "NA",
                    "random_seed": seed,
                    "notes": "One canonical trajectory per selected domain: 320 K, R1.",
                }
            )
    return subset_rows


def main() -> None:
    """Write 5/10/20-per-class subset manifests."""

    args = parse_args()
    rows = read_tsv(args.domain_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "subset_name",
        "domain_id",
        "trajectory_id",
        "cath_class",
        "cath_architecture",
        "cath_topology",
        "cath_homologous_superfamily",
        "cath_code",
        "h5_path",
        "h5_size_bytes",
        "selected_temperature_K",
        "selected_replica_index_zero_based",
        "selected_replica_label_one_based",
        "selection_rank_within_class",
        "selection_reason",
        "fallback_used",
        "fallback_reason",
        "random_seed",
        "notes",
    ]
    for target in TARGETS:
        subset = build_subset_rows(rows, target, args.seed)
        output = args.output_dir / f"mdcath_cath_balanced_{target}_per_class.tsv"
        write_tsv(output, subset, columns)
        print(output)


if __name__ == "__main__":
    main()
