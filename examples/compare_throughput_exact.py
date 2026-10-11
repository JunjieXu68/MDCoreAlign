#!/usr/bin/env python3
"""Compare compact throughput and exact outputs without requiring SciPy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(values.size, dtype=float)
    ranked[order] = np.arange(values.size, dtype=float)
    return ranked


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def read_profile(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        "weight": np.asarray([float(row["stable_weight"]) for row in rows]),
        "core": np.asarray([row["core_label"].lower() in {"1", "true"} for row in rows]),
        "rmsf": np.asarray([float(row["robust_RMSF"]) for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("throughput", type=Path)
    parser.add_argument("exact", type=Path)
    args = parser.parse_args()
    left, right = read_profile(args.throughput / "residue_profile.tsv"), read_profile(args.exact / "residue_profile.tsv")
    union = np.count_nonzero(left["core"] | right["core"])
    with np.load(args.throughput / "transforms.npz") as a, np.load(args.exact / "transforms.npz") as b:
        transforms_valid = bool(np.isfinite(a["rotations"]).all() and np.isfinite(b["rotations"]).all() and a["rotations"].shape == b["rotations"].shape)
    def rmsd_values(root: Path) -> np.ndarray:
        with (root / "frame_metrics.tsv").open(newline="") as handle:
            return np.asarray([float(row["full_set_RMSD"]) for row in csv.DictReader(handle, delimiter="\t")])
    payload = {
        "stable_weight_spearman": spearman(left["weight"], right["weight"]),
        "core_jaccard": float(np.count_nonzero(left["core"] & right["core"]) / union),
        "rmsd_mae": float(np.mean(np.abs(rmsd_values(args.throughput) - rmsd_values(args.exact)))),
        "robust_rmsf_spearman": spearman(left["rmsf"], right["rmsf"]),
        "transform_arrays_valid": transforms_valid,
        "warnings": "review diagnostics.json in both outputs",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
