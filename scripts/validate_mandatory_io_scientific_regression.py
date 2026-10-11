#!/usr/bin/env python3
"""Compare frozen RF-v3 and mandatory-I/O v4 compact scientific outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _tsv_column(path: Path, column: str) -> np.ndarray:
    with path.open(newline="") as handle:
        values = [row[column] for row in csv.DictReader(handle, delimiter="\t")]
    try:
        return np.asarray(values, dtype=float)
    except ValueError:
        return np.asarray(values, dtype=str)


def _difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.dtype.kind in "US" or right.dtype.kind in "US":
        return float(np.count_nonzero(left != right))
    return float(np.max(np.abs(left - right), initial=0.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    specifications = {
        "exact": {
            "npz": ["rotations", "translations"],
            "frame_metrics.tsv": ["full_set_RMSD", "weighted_RMSD"],
            "residue_profile.tsv": [
                "stable_weight", "mobility_score", "robust_RMSF",
                "core_label", "flexible_segment_id",
            ],
        },
        "throughput": {
            "npz": ["rotations", "translations"],
            "frame_metrics.tsv": ["full_set_RMSD", "weighted_RMSD"],
            "residue_profile.tsv": [
                "stable_weight", "mobility_score", "robust_RMSF",
                "core_label", "flexible_segment_id",
            ],
        },
        "W5": {
            "local_deformation.tsv": [
                "local_deformation_index", "local_rigidity_score",
                "locally_deforming_segment_id",
            ],
        },
    }
    rows: list[dict[str, object]] = []
    for mode, files in specifications.items():
        baseline_dir = args.baseline / mode
        candidate_dir = args.candidate / mode
        for filename, columns in files.items():
            for column in columns:
                if filename == "npz":
                    with np.load(baseline_dir / "transforms.npz") as left_archive:
                        left = left_archive[column]
                    with np.load(candidate_dir / "transforms.npz") as right_archive:
                        right = right_archive[column]
                    output_name = column
                else:
                    left = _tsv_column(baseline_dir / filename, column)
                    right = _tsv_column(candidate_dir / filename, column)
                    output_name = f"{filename}:{column}"
                difference = _difference(left, right)
                rows.append(
                    {
                        "mode": mode,
                        "output": output_name,
                        "max_abs_difference": difference,
                        "identical_or_tolerance": difference == 0.0,
                        "status": "pass" if difference == 0.0 else "fail",
                    }
                )
    if len(rows) != 21:
        raise RuntimeError(f"expected 21 registered comparisons, observed {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["status"] != "pass"]
    print(f"scientific regression: {len(rows) - len(failures)}/{len(rows)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
