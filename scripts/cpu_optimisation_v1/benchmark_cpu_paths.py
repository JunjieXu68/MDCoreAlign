#!/usr/bin/env python3
"""Run locked CPU kernel and compact end-to-end optimisation benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdcorealign.experimental import solve_fast_global_alignment  # noqa: E402
from mdsuperpose.fast_kabsch import (  # noqa: E402
    apply_transforms_batch,
    weighted_kabsch_batch,
    weighted_kabsch_variable_weights_batch,
)
from mdsuperpose.sequential_weights import (  # noqa: E402
    estimate_representative_weights_sequential_mean,
)
from mdsuperpose.trajectory import load_trajectory_from_multimodel_pdb  # noqa: E402
from mdsuperpose.window_refine import (  # noqa: E402
    exact_all_frame_fit,
    temporally_stratified_indices,
)


OUT = ROOT / "results/cpu_optimisation_v1"
KERNEL_RESIDUES = (75, 150, 300, 600)
KERNEL_FRAMES = (1, 32, 100, 500, 2000)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def median_seconds(function, *, repeats: int = 3) -> float:
    values = []
    function()
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(statistics.median(values))


def synthetic_coords(frames: int, residues: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reference = np.cumsum(rng.normal(scale=0.55, size=(residues, 3)), axis=0)
    noise = rng.normal(scale=0.08, size=(frames, residues, 3))
    coords = reference[None, :, :] + noise
    mobile_start = max(3, int(residues * 0.75))
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    coords[:, mobile_start:, 1] += 1.5 * np.sin(phase)[:, None]
    return np.asarray(coords, dtype=np.float64)


def load_panel() -> list[tuple[str, np.ndarray, str]]:
    panel: list[tuple[str, np.ndarray, str]] = []
    toy = load_trajectory_from_multimodel_pdb(
        ROOT / "examples/toy_md/toy_ca_trajectory.pdb"
    ).coords
    panel.append(("toy_md", toy, "bundled_toy"))
    real_small = load_trajectory_from_multimodel_pdb(
        ROOT / "examples/real_md_small/trajectory_small.pdb"
    ).coords
    panel.append(("real_md_small", real_small, "bundled_real_md"))
    with np.load(
        ROOT
        / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/runs/3pio100/10000/transition_aligned_coords.npz",
        allow_pickle=True,
    ) as archive:
        panel.append(
            (
                "mdcath_3pio100",
                np.asarray(archive["coords"][:100], dtype=np.float64),
                "warm_start_sensitive_locked_prefix",
            )
        )
    with np.load(
        ROOT
        / "results/scaling_optimization_v5/phase7b_unit_integrity/regenerated/1te7A00/transition_aligned_coords_10000.npz",
        allow_pickle=True,
    ) as archive:
        panel.append(
            (
                "mdcath_1te7A00",
                np.asarray(archive["coords"][:100], dtype=np.float64),
                "locked_medium_mdcath_prefix",
            )
        )
    panel.append(("synthetic_medium", synthetic_coords(100, 300, 301), "fixed_seed_synthetic"))
    panel.append(("synthetic_large", synthetic_coords(100, 600, 601), "fixed_seed_synthetic"))
    return panel


def write_locked_panel() -> None:
    rows = []
    for name, coords, source in load_panel():
        rows.append(
            {
                "case": name,
                "source_class": source,
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "selection": "CA coordinates as stored",
                "reference": "first_frame",
                "exact_representative_count": min(64, coords.shape[0]),
                "throughput_representative_count": min(64, coords.shape[0]),
                "throughput_update_rounds": 4,
                "locked_before_optimisation": True,
            }
        )
    write_tsv(OUT / "01_baseline/locked_panel.tsv", rows)


def kernel_benchmark(label: str) -> None:
    rows: list[dict[str, object]] = []
    for n_residues in KERNEL_RESIDUES:
        for n_frames in KERNEL_FRAMES:
            rng = np.random.default_rng(n_residues * 10000 + n_frames)
            reference = rng.normal(size=(n_residues, 3))
            coords = reference[None, :, :] + rng.normal(
                scale=0.2,
                size=(n_frames, n_residues, 3),
            )
            weights = rng.uniform(0.1, 1.0, size=n_residues)
            variable = rng.uniform(0.1, 1.0, size=(n_frames, n_residues))
            repeats = 5 if n_frames * n_residues <= 60_000 else 3

            fixed_seconds = median_seconds(
                lambda: weighted_kabsch_batch(
                    coords,
                    reference,
                    weights,
                    validate_inputs=False,
                    compute_metrics=True,
                ),
                repeats=repeats,
            )
            variable_seconds = median_seconds(
                lambda: weighted_kabsch_variable_weights_batch(
                    coords,
                    reference,
                    variable,
                    validate_inputs=False,
                    compute_metrics=True,
                ),
                repeats=repeats,
            )
            fit = weighted_kabsch_batch(
                coords,
                reference,
                weights,
                validate_inputs=False,
                compute_metrics=False,
            )
            apply_seconds = median_seconds(
                lambda: apply_transforms_batch(
                    coords,
                    fit.rotations,
                    fit.translations,
                ),
                repeats=repeats,
            )
            weight_sum = float(np.sum(weights))
            reference_center = (weights @ reference) / weight_sum
            mobile_centers = np.einsum("tni,n->ti", coords, weights) / weight_sum
            mobile_centered = coords - mobile_centers[:, None, :]
            reference_centered = reference - reference_center
            covariance_seconds = median_seconds(
                lambda: np.einsum(
                    "tni,n,nj->tij",
                    mobile_centered,
                    weights / weight_sum,
                    reference_centered,
                ),
                repeats=repeats,
            )
            covariances = np.einsum(
                "tni,n,nj->tij",
                mobile_centered,
                weights / weight_sum,
                reference_centered,
            )
            svd_seconds = median_seconds(
                lambda: np.linalg.svd(covariances, full_matrices=False),
                repeats=repeats,
            )
            aligned = apply_transforms_batch(coords, fit.rotations, fit.translations)
            metric_seconds = median_seconds(
                lambda: np.sum((aligned - reference[None, :, :]) ** 2, axis=2),
                repeats=repeats,
            )
            common = {
                "label": label,
                "n_frames": n_frames,
                "n_residues": n_residues,
                "residue_frames": n_frames * n_residues,
                "threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
            }
            for stage, seconds in (
                ("fixed_weight_full_wrapper", fixed_seconds),
                ("variable_weight_full_wrapper", variable_seconds),
                ("aligned_coordinate_generation", apply_seconds),
                ("covariance", covariance_seconds),
                ("stacked_svd", svd_seconds),
                ("metric_reduction", metric_seconds),
            ):
                rows.append({**common, "stage": stage, "median_seconds": seconds})
            rows.append(
                {
                    **common,
                    "stage": "transform_only",
                    "median_seconds": "",
                    "status": "unavailable_in_baseline" if label == "baseline" else "measured_separately",
                }
            )
    target = (
        OUT / "01_baseline/kernel_baseline.tsv"
        if label == "baseline"
        else OUT / "07_validation/kernel_optimised.tsv"
    )
    write_tsv(target, rows)


def _measure(function):
    tracemalloc.start()
    start = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, int(peak)


def exact_components(coords: np.ndarray):
    selected = temporally_stratified_indices(coords.shape[0], min(64, coords.shape[0]))
    estimate, weight_seconds, weight_peak = _measure(
        lambda: estimate_representative_weights_sequential_mean(
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
    )
    fit, fit_seconds, fit_peak = _measure(
        lambda: exact_all_frame_fit(
            coords,
            coords[0],
            estimate.mean_weights,
            chunk_size=min(2000, coords.shape[0]),
        )
    )
    return estimate, fit, weight_seconds, fit_seconds, max(weight_peak, fit_peak)


def end_to_end(label: str, threads: int) -> None:
    rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []
    for name, coords, source in load_panel():
        estimate, fit, weight_seconds, fit_seconds, exact_peak = exact_components(coords)
        exact_total = weight_seconds + fit_seconds
        rows.append(
            {
                "label": label,
                "threads": threads,
                "case": name,
                "source_class": source,
                "mode": "exact",
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "weight_estimation_seconds": weight_seconds,
                "all_frame_fit_seconds": fit_seconds,
                "result_construction_seconds": max(0.0, exact_total - weight_seconds - fit_seconds),
                "end_to_end_seconds": exact_total,
                "mean_rmsd": float(np.mean(fit.rmsd)),
                "mean_weighted_rmsd": float(np.mean(fit.weighted_rmsd)),
                "average_iterations": estimate.average_iterations_per_frame,
                "converged_fraction": estimate.converged_fraction,
                "ordinary_rmsf_computed": True,
            }
        )
        memory_rows.append(
            {
                "label": label,
                "threads": threads,
                "case": name,
                "mode": "exact",
                "tracemalloc_peak_bytes": exact_peak,
            }
        )
        fast, fast_seconds, fast_peak = _measure(
            lambda: solve_fast_global_alignment(
                coords,
                representative_count=min(64, coords.shape[0]),
                update_rounds=4,
                chunk_size=min(2000, coords.shape[0]),
                store_aligned=False,
                compute_confidence=False,
            )
        )
        rows.append(
            {
                "label": label,
                "threads": threads,
                "case": name,
                "source_class": source,
                "mode": "throughput",
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "weight_estimation_seconds": fast.timing.get("global_refinement", ""),
                "all_frame_fit_seconds": fast.timing.get("exact_all_frame_fit", ""),
                "result_construction_seconds": "",
                "end_to_end_seconds": fast_seconds,
                "mean_rmsd": fast.mean_rmsd,
                "mean_weighted_rmsd": fast.mean_weighted_rmsd,
                "average_iterations": fast.average_iterations_per_frame,
                "converged_fraction": fast.converged_fraction,
                "ordinary_rmsf_computed": True,
            }
        )
        memory_rows.append(
            {
                "label": label,
                "threads": threads,
                "case": name,
                "mode": "throughput",
                "tracemalloc_peak_bytes": fast_peak,
            }
        )
    base = OUT / ("01_baseline" if label == "baseline" else "07_validation")
    write_tsv(base / f"end_to_end_{label}_threads_{threads}.tsv", rows)
    write_tsv(base / f"memory_{label}_threads_{threads}.tsv", memory_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("baseline", "optimised"), required=True)
    parser.add_argument("--mode", choices=("lock", "kernel", "end-to-end"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "lock":
        write_locked_panel()
    elif args.mode == "kernel":
        kernel_benchmark(args.label)
    else:
        end_to_end(args.label, args.threads)


if __name__ == "__main__":
    main()
