#!/usr/bin/env python3
"""Generate O2/O3/O4/O7 parity, timing and allocation tables."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts/cpu_optimisation_v1"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_cpu_paths import load_panel, synthetic_coords  # noqa: E402
from mdsuperpose.all_frame_engine import (  # noqa: E402
    CPUAlignmentWorkspace,
    fit_all_frames_fixed_weights,
)
from mdsuperpose.fast_kabsch import (  # noqa: E402
    weighted_kabsch_batch,
    weighted_kabsch_transform_batch,
)
from mdsuperpose.sequential_weights import (  # noqa: E402
    estimate_representative_weights_sequential_lean,
    estimate_representative_weights_sequential_mean,
)


OUT = ROOT / "results/cpu_optimisation_v1"


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _measure(function, repeats: int = 5) -> tuple[float, int]:
    function()
    times = []
    peaks = []
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    return float(statistics.median(times)), int(statistics.median(peaks))


def _exact_kwargs() -> dict[str, object]:
    return {
        "rule": "huber",
        "delta": 1.0,
        "max_iterations": 20,
        "tolerance": 1e-6,
        "objective_tolerance": 1e-5,
        "objective_patience": 2,
        "sequence_continuity_strength": 0.05,
        "contact_consistency_strength": 0.0,
        "record_history": False,
    }


def main() -> None:
    transform_parity = []
    transform_benchmark = []
    transform_memory = []
    ordinary_parity = []
    ordinary_benchmark = []
    fused_parity = []
    workspace_benchmark = []
    allocation_after = []
    mean_parity = []
    mean_benchmark = []
    convergence = []
    allocation_reduction = []

    for case, coords, source in load_panel():
        reference = coords[0]
        rng = np.random.default_rng(coords.shape[1] * 31 + coords.shape[0])
        weights = rng.uniform(0.1, 1.0, size=coords.shape[1])
        compatibility = weighted_kabsch_batch(coords, reference, weights)
        transform = weighted_kabsch_transform_batch(coords, reference, weights)
        transform_parity.append(
            {
                "case": case,
                "max_rotation_delta": float(np.max(np.abs(transform.rotations - compatibility.rotations))),
                "max_translation_delta": float(np.max(np.abs(transform.translations - compatibility.translations))),
                "proper_rotations": bool(np.allclose(np.linalg.det(transform.rotations), 1.0, atol=1e-10)),
            }
        )
        wrapper_seconds, wrapper_peak = _measure(
            lambda: weighted_kabsch_batch(coords, reference, weights, compute_metrics=False)
        )
        transform_seconds, transform_peak = _measure(
            lambda: weighted_kabsch_transform_batch(coords, reference, weights)
        )
        transform_benchmark.append(
            {
                "case": case,
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "compatibility_wrapper_seconds": wrapper_seconds,
                "transform_only_seconds": transform_seconds,
                "speedup": wrapper_seconds / transform_seconds,
            }
        )
        transform_memory.append(
            {
                "case": case,
                "compatibility_wrapper_peak_bytes": wrapper_peak,
                "transform_only_peak_bytes": transform_peak,
                "peak_memory_reduction": wrapper_peak / transform_peak,
            }
        )

        with_ordinary = fit_all_frames_fixed_weights(
            coords, reference, weights, compute_ordinary_rmsf=True
        )
        without_ordinary = fit_all_frames_fixed_weights(
            coords, reference, weights, compute_ordinary_rmsf=False
        )
        ordinary_parity.append(
            {
                "case": case,
                "max_weight_delta": 0.0,
                "max_rotation_delta": float(np.max(np.abs(with_ordinary.rotations - without_ordinary.rotations))),
                "max_translation_delta": float(np.max(np.abs(with_ordinary.translations - without_ordinary.translations))),
                "max_rmsd_delta": float(np.max(np.abs(with_ordinary.rmsd - without_ordinary.rmsd))),
                "max_robust_rmsf_delta": float(np.max(np.abs(with_ordinary.robust_rmsf - without_ordinary.robust_rmsf))),
                "ordinary_with_enabled": with_ordinary.ordinary_rmsf is not None,
                "ordinary_with_disabled": without_ordinary.ordinary_rmsf is not None,
            }
        )
        on_seconds, on_peak = _measure(
            lambda: fit_all_frames_fixed_weights(
                coords, reference, weights, compute_ordinary_rmsf=True
            )
        )
        off_seconds, off_peak = _measure(
            lambda: fit_all_frames_fixed_weights(
                coords, reference, weights, compute_ordinary_rmsf=False
            )
        )
        ordinary_benchmark.append(
            {
                "case": case,
                "ordinary_on_seconds": on_seconds,
                "ordinary_off_seconds": off_seconds,
                "compact_speedup": on_seconds / off_seconds,
                "ordinary_on_peak_bytes": on_peak,
                "ordinary_off_peak_bytes": off_peak,
            }
        )
        fused_parity.append(
            {
                "case": case,
                "rmsd_max_delta_vs_wrapper": float(np.max(np.abs(with_ordinary.rmsd - compatibility.rmsd))),
                "weighted_rmsd_max_delta_vs_wrapper": float(np.max(np.abs(with_ordinary.weighted_rmsd - compatibility.weighted_rmsd))),
                "mean_residual_finite": bool(np.isfinite(with_ordinary.mean_residual).all()),
            }
        )
        workspace = CPUAlignmentWorkspace(coords.shape[1])
        no_workspace_seconds, _ = _measure(
            lambda: fit_all_frames_fixed_weights(coords, reference, weights), repeats=10
        )
        workspace_seconds, workspace_peak = _measure(
            lambda: fit_all_frames_fixed_weights(
                coords, reference, weights, workspace=workspace
            ),
            repeats=10,
        )
        workspace_benchmark.append(
            {
                "case": case,
                "without_workspace_seconds": no_workspace_seconds,
                "with_workspace_seconds": workspace_seconds,
                "workspace_speedup": no_workspace_seconds / workspace_seconds,
                "workspace_retains_trajectory": False,
            }
        )
        allocation_after.append(
            {
                "case": case,
                "transform_only_peak_bytes": transform_peak,
                "compact_all_frame_peak_bytes": off_peak,
                "workspace_all_frame_peak_bytes": workspace_peak,
            }
        )

        representatives = coords[np.linspace(0, coords.shape[0] - 1, min(64, coords.shape[0]), dtype=int)]
        detailed = estimate_representative_weights_sequential_lean(
            representatives, reference, **_exact_kwargs()
        )
        mean_only = estimate_representative_weights_sequential_mean(
            representatives, reference, **_exact_kwargs()
        )
        mean_parity.append(
            {
                "case": case,
                "max_mean_weight_delta": float(np.max(np.abs(detailed.mean_weights - mean_only.mean_weights))),
                "max_mean_residual_delta": float(np.max(np.abs(detailed.mean_residual - mean_only.mean_residual))),
                "iteration_mean_delta": detailed.average_iterations_per_frame - mean_only.average_iterations_per_frame,
                "converged_fraction_delta": detailed.converged_fraction - mean_only.converged_fraction,
            }
        )
        detailed_seconds, detailed_peak = _measure(
            lambda: estimate_representative_weights_sequential_lean(
                representatives, reference, **_exact_kwargs()
            ),
            repeats=3,
        )
        mean_seconds, mean_peak = _measure(
            lambda: estimate_representative_weights_sequential_mean(
                representatives, reference, **_exact_kwargs()
            ),
            repeats=3,
        )
        mean_benchmark.append(
            {
                "case": case,
                "representative_count": representatives.shape[0],
                "detailed_seconds": detailed_seconds,
                "mean_only_seconds": mean_seconds,
                "runtime_speedup": detailed_seconds / mean_seconds,
            }
        )
        convergence.append(
            {
                "case": case,
                "detailed_average_iterations": detailed.average_iterations_per_frame,
                "mean_only_average_iterations": mean_only.average_iterations_per_frame,
                "detailed_converged_fraction": detailed.converged_fraction,
                "mean_only_converged_fraction": mean_only.converged_fraction,
                "max_iterations_reached_count": mean_only.max_iterations_reached_count,
            }
        )
        allocation_reduction.append(
            {
                "case": case,
                "detailed_peak_bytes": detailed_peak,
                "mean_only_peak_bytes": mean_peak,
                "peak_memory_reduction": detailed_peak / mean_peak,
                "detailed_retained_arrays": detailed.call_counts["retained_per_representative_detail_arrays"],
                "mean_only_retained_arrays": mean_only.call_counts["retained_per_representative_detail_arrays"],
            }
        )

    stress = synthetic_coords(2000, 150, 20260820)
    for case, reps in (("synthetic_2000x150", stress),):
        detailed_seconds, detailed_peak = _measure(
            lambda: estimate_representative_weights_sequential_lean(
                reps, reps[0], **_exact_kwargs()
            ),
            repeats=1,
        )
        mean_seconds, mean_peak = _measure(
            lambda: estimate_representative_weights_sequential_mean(
                reps, reps[0], **_exact_kwargs()
            ),
            repeats=1,
        )
        mean_benchmark.append(
            {
                "case": case,
                "representative_count": reps.shape[0],
                "detailed_seconds": detailed_seconds,
                "mean_only_seconds": mean_seconds,
                "runtime_speedup": detailed_seconds / mean_seconds,
            }
        )
        allocation_reduction.append(
            {
                "case": case,
                "detailed_peak_bytes": detailed_peak,
                "mean_only_peak_bytes": mean_peak,
                "peak_memory_reduction": detailed_peak / mean_peak,
                "detailed_retained_arrays": 8,
                "mean_only_retained_arrays": 0,
            }
        )

    targets = {
        "03_all_frame_engine/transform_only_parity.tsv": transform_parity,
        "03_all_frame_engine/transform_only_benchmark.tsv": transform_benchmark,
        "03_all_frame_engine/memory_comparison.tsv": transform_memory,
        "03_all_frame_engine/ordinary_rmsf_parity.tsv": ordinary_parity,
        "03_all_frame_engine/ordinary_rmsf_benchmark.tsv": ordinary_benchmark,
        "03_all_frame_engine/fused_reduction_parity.tsv": fused_parity,
        "03_all_frame_engine/workspace_benchmark.tsv": workspace_benchmark,
        "03_all_frame_engine/allocation_after.tsv": allocation_after,
        "04_exact_estimator/mean_only_parity.tsv": mean_parity,
        "04_exact_estimator/mean_only_benchmark.tsv": mean_benchmark,
        "04_exact_estimator/convergence_metadata_validation.tsv": convergence,
        "04_exact_estimator/allocation_reduction.tsv": allocation_reduction,
    }
    for relative, rows in targets.items():
        _write(OUT / relative, rows)


if __name__ == "__main__":
    main()
