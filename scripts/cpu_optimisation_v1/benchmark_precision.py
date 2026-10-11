#!/usr/bin/env python3
"""Evaluate optional CPU throughput precision candidates on the locked panel."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts/cpu_optimisation_v1"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_cpu_paths import load_panel  # noqa: E402
from mdcorealign.experimental import solve_fast_global_alignment  # noqa: E402
from mdsuperpose.fast_kabsch import weighted_kabsch_transform_batch  # noqa: E402
from mdsuperpose.window_refine import (  # noqa: E402
    bounded_global_weight_refinement,
    temporally_stratified_indices,
)


OUT = ROOT / "results/cpu_optimisation_v1/05_precision_solver"


@dataclass(slots=True)
class Candidate:
    weights: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    rmsd: np.ndarray
    rmsf: np.ndarray


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 1.0 if np.allclose(a, b) else 0.0
    order_a = np.argsort(a, kind="mergesort")
    order_b = np.argsort(b, kind="mergesort")
    rank_a = np.empty(a.size, dtype=float)
    rank_b = np.empty(b.size, dtype=float)
    rank_a[order_a] = np.arange(a.size)
    rank_b[order_b] = np.arange(b.size)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _core(weights: np.ndarray) -> np.ndarray:
    count = max(1, int(np.floor(weights.size * 0.6 + 0.5)))
    order = np.lexsort((np.arange(weights.size), -weights))
    labels = np.zeros(weights.size, dtype=bool)
    labels[order[:count]] = True
    return labels


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def _rotation_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = a @ np.swapaxes(b, 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _normal_candidate(coords: np.ndarray, precision: str) -> Candidate:
    result = solve_fast_global_alignment(
        coords,
        representative_count=min(64, coords.shape[0]),
        update_rounds=4,
        chunk_size=min(2000, coords.shape[0]),
        store_aligned=False,
        compute_confidence=False,
        compute_ordinary_rmsf=False,
        representative_precision=precision,
    )
    return Candidate(
        weights=np.asarray(result.frozen_weights),
        rotations=np.asarray(result.rotations),
        translations=np.asarray(result.translations),
        rmsd=np.asarray(result.rmsd),
        rmsf=np.asarray(result.robust_rmsf),
    )


def _aggressive_float32(coords: np.ndarray) -> Candidate:
    values = np.asarray(coords, dtype=np.float32)
    selected = temporally_stratified_indices(values.shape[0], min(64, values.shape[0]))
    weights, _, _, _ = bounded_global_weight_refinement(
        values[selected],
        values[0],
        np.ones(values.shape[1]),
        update_rounds=4,
        precision="float32",
    )
    rotations = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    translations = np.empty((values.shape[0], 3), dtype=np.float64)
    rmsd = np.empty(values.shape[0], dtype=np.float64)
    aligned_for_rmsf = np.empty(values.shape, dtype=np.float64)
    weights32 = weights.astype(np.float32)
    for start in range(0, values.shape[0], min(2000, values.shape[0])):
        stop = min(start + min(2000, values.shape[0]), values.shape[0])
        transform = weighted_kabsch_transform_batch(
            values[start:stop],
            values[0],
            weights32,
            validate_inputs=False,
            dtype=np.float32,
        )
        aligned32 = (
            values[start:stop] @ np.swapaxes(transform.rotations, 1, 2)
            + transform.translations[:, None, :]
        )
        aligned = aligned32.astype(np.float64)
        rotations[start:stop] = transform.rotations
        translations[start:stop] = transform.translations
        aligned_for_rmsf[start:stop] = aligned
        rmsd[start:stop] = np.sqrt(
            np.mean((aligned - values[0].astype(np.float64)) ** 2, axis=(1, 2))
        )
    rmsf = np.sqrt(np.sum(np.var(aligned_for_rmsf, axis=0), axis=1))
    return Candidate(weights, rotations, translations, rmsd, rmsf)


def _median_runtime(function, repeats: int = 5) -> float:
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(statistics.median(values))


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


def main() -> None:
    quality_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    methods = {
        "throughput_float64": lambda coords: _normal_candidate(coords, "float64"),
        "throughput_mixed": lambda coords: _normal_candidate(coords, "mixed"),
        "throughput_float32_representatives": lambda coords: _normal_candidate(coords, "float32"),
        "throughput_float32_aggressive": _aggressive_float32,
    }
    for case, coords, source in load_panel():
        reference = methods["throughput_float64"](coords)
        reference_core = _core(reference.weights)
        for method, function in methods.items():
            try:
                candidate = reference if method == "throughput_float64" else function(coords)
                core_jaccard = _jaccard(reference_core, _core(candidate.weights))
                rmsf_spearman = _spearman(reference.rmsf, candidate.rmsf)
                quality_rows.append(
                    {
                        "case": case,
                        "source_class": source,
                        "method": method,
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "weight_max_abs_difference": float(np.max(np.abs(candidate.weights - reference.weights))),
                        "weight_mean_abs_difference": float(np.mean(np.abs(candidate.weights - reference.weights))),
                        "weight_spearman": _spearman(reference.weights, candidate.weights),
                        "core_jaccard": core_jaccard,
                        "rotation_geodesic_median_radians": float(np.median(_rotation_angles(candidate.rotations, reference.rotations))),
                        "rotation_geodesic_max_radians": float(np.max(_rotation_angles(candidate.rotations, reference.rotations))),
                        "translation_difference_median": float(np.median(np.linalg.norm(candidate.translations - reference.translations, axis=1))),
                        "rmsd_mae": float(np.mean(np.abs(candidate.rmsd - reference.rmsd))),
                        "rmsd_abs_difference_p95": float(np.quantile(np.abs(candidate.rmsd - reference.rmsd), 0.95)),
                        "rmsf_spearman": rmsf_spearman,
                        "rmsf_mae": float(np.mean(np.abs(candidate.rmsf - reference.rmsf))),
                        "finite": bool(all(np.isfinite(array).all() for array in (candidate.weights, candidate.rotations, candidate.translations, candidate.rmsd, candidate.rmsf))),
                        "quality_pass": bool(core_jaccard >= 0.95 and rmsf_spearman >= 0.999),
                    }
                )
                seconds = _median_runtime(lambda function=function: function(coords))
                performance_rows.append(
                    {
                        "case": case,
                        "method": method,
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "median_seconds": seconds,
                    }
                )
            except Exception as error:
                quality_rows.append(
                    {
                        "case": case,
                        "source_class": source,
                        "method": method,
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "finite": False,
                        "quality_pass": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    baseline = {
        row["case"]: float(row["median_seconds"])
        for row in performance_rows
        if row["method"] == "throughput_float64"
    }
    for row in performance_rows:
        row["speedup_vs_float64"] = baseline[row["case"]] / float(row["median_seconds"])
    _write(OUT / "precision_quality.tsv", quality_rows)
    _write(OUT / "precision_performance.tsv", performance_rows)
    candidate_rows = [row for row in quality_rows if row["method"] == "throughput_mixed"]
    candidate_perf = [row for row in performance_rows if row["method"] == "throughput_mixed"]
    median_core = statistics.median(float(row["core_jaccard"]) for row in candidate_rows)
    minimum_core = min(float(row["core_jaccard"]) for row in candidate_rows)
    median_speedup = statistics.median(float(row["speedup_vs_float64"]) for row in candidate_perf)
    promote = median_core >= 0.99 and minimum_core >= 0.95 and median_speedup >= 1.15
    (OUT / "precision_contract.md").write_text(
        "# CPU throughput precision contract\n\n"
        "Exact remains float64. Mixed precision limits float32 to representative-frame weight updates; frozen weights and final all-frame fitting/RMSF remain float64. The aggressive float32 all-frame candidate is benchmark-only.\n"
    )
    (OUT / "precision_decision.md").write_text(
        "# Precision decision\n\n"
        f"- Mixed median core Jaccard: {median_core:.6f}.\n"
        f"- Mixed minimum core Jaccard: {minimum_core:.6f}.\n"
        f"- Mixed median end-to-end speedup: {median_speedup:.3f}x.\n"
        f"- Promotion gate passed: {str(promote).lower()}.\n"
        "- Public throughput default remains float64; non-default candidates remain experimental.\n"
    )


if __name__ == "__main__":
    main()
