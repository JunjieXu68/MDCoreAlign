#!/usr/bin/env python3
"""Benchmark a source snapshot on locked 12-to-10k frame prefixes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import numpy as np


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def _median(function, repeats: int) -> tuple[float, object]:
    function()
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - start)
    return float(statistics.median(values)), result


def main() -> None:
    args = _parse()
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    root = Path(__file__).resolve().parents[2]
    from mdcorealign.experimental import solve_fast_global_alignment
    import mdsuperpose.sequential_weights as sequential
    from mdsuperpose.window_refine import exact_all_frame_fit, temporally_stratified_indices

    with np.load(
        root
        / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/runs/3pio100/10000/transition_aligned_coords.npz",
        allow_pickle=True,
    ) as archive:
        all_coords = np.asarray(archive["coords"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    frame_counts = (12, 100, 500, 1000, 5000, 10000)
    for frame_count in frame_counts:
        coords = all_coords[:frame_count]
        selected = temporally_stratified_indices(
            frame_count,
            min(2000, frame_count),
        )

        def exact_run(ordinary: bool = True):
            estimator = getattr(
                sequential,
                "estimate_representative_weights_sequential_mean",
                sequential.estimate_representative_weights_sequential_lean,
            )
            estimate = estimator(
                coords[selected],
                coords[0],
                rule="huber",
                delta=1.0,
                max_iterations=20,
                tolerance=1e-6,
                objective_tolerance=1e-5,
                objective_patience=2,
                sequence_continuity_strength=0.05,
                contact_consistency_strength=0.0,
                record_history=False,
            )
            kwargs = {"chunk_size": 2000}
            if "compute_ordinary_rmsf" in exact_all_frame_fit.__code__.co_varnames:
                kwargs["compute_ordinary_rmsf"] = ordinary
            return estimate, exact_all_frame_fit(
                coords,
                coords[0],
                estimate.mean_weights,
                **kwargs,
            )

        def throughput_run(ordinary: bool = True):
            kwargs = {
                "representative_count": min(256, frame_count),
                "update_rounds": 4,
                "chunk_size": 2000,
                "store_aligned": False,
                "compute_confidence": False,
            }
            if "compute_ordinary_rmsf" in solve_fast_global_alignment.__code__.co_varnames:
                kwargs["compute_ordinary_rmsf"] = ordinary
            return solve_fast_global_alignment(coords, **kwargs)

        contracts = [("exact_full", lambda: exact_run(True)), ("throughput_full", lambda: throughput_run(True))]
        if args.label != "baseline_checkpoint":
            contracts.extend(
                [
                    ("exact_compact", lambda: exact_run(False)),
                    ("throughput_compact", lambda: throughput_run(False)),
                ]
            )
        for contract, function in contracts:
            seconds, result = _median(function, args.repeats)
            tracemalloc.start()
            function()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            fit = result[1] if contract.startswith("exact") else result
            rows.append(
                {
                    "label": args.label,
                    "case": "mdcath_3pio100_transition_aligned_stress",
                    "n_frames": frame_count,
                    "n_residues": coords.shape[1],
                    "contract": contract,
                    "median_loaded_seconds": seconds,
                    "tracemalloc_peak_bytes": peak,
                    "mean_rmsd": float(np.mean(fit.rmsd)),
                    "mean_weighted_rmsd": float(np.mean(fit.weighted_rmsd)),
                    "ordinary_rmsf_computed": getattr(fit, "ordinary_rmsf", None) is not None,
                    "repeats": args.repeats,
                    "aligned_coordinates_retained": False,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
