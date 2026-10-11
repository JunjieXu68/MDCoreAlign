#!/usr/bin/env python3
"""Measure fixed-weight all-frame fitting for one source snapshot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    root = Path(__file__).resolve().parents[2]
    from mdsuperpose.window_refine import exact_all_frame_fit

    with np.load(
        root
        / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/runs/3pio100/10000/transition_aligned_coords.npz",
        allow_pickle=True,
    ) as archive:
        all_coords = np.asarray(archive["coords"], dtype=np.float64)
    weights = np.linspace(1.0, 0.2, all_coords.shape[1], dtype=np.float64)
    rows = []
    for frames in (100, 1000, 10000):
        coords = all_coords[:frames]
        contracts = [("full", True)]
        if "compute_ordinary_rmsf" in exact_all_frame_fit.__code__.co_varnames:
            contracts.append(("compact", False))
        for contract, ordinary in contracts:
            def run():
                kwargs = {"chunk_size": 2000}
                if "compute_ordinary_rmsf" in exact_all_frame_fit.__code__.co_varnames:
                    kwargs["compute_ordinary_rmsf"] = ordinary
                return exact_all_frame_fit(coords, coords[0], weights, **kwargs)

            run()
            times = []
            result = None
            for _ in range(5):
                start = time.perf_counter()
                result = run()
                times.append(time.perf_counter() - start)
            assert result is not None
            rows.append(
                {
                    "label": args.label,
                    "n_frames": frames,
                    "n_residues": coords.shape[1],
                    "contract": contract,
                    "median_seconds": statistics.median(times),
                    "mean_rmsd": float(np.mean(result.rmsd)),
                    "mean_weighted_rmsd": float(np.mean(result.weighted_rmsd)),
                    "ordinary_rmsf_computed": getattr(result, "ordinary_rmsf", None) is not None,
                    "repeats": 5,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
