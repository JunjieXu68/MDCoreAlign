#!/usr/bin/env python3
"""Benchmark the O1 covariance rewrite against the frozen einsum path."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose.fast_kabsch import weighted_kabsch_transform_batch  # noqa: E402


OUT = ROOT / "results/cpu_optimisation_v1/02_kabsch_kernel"
RESIDUES = (75, 150, 300, 600)
FRAMES = (1, 32, 100, 500, 2000)


def _row_operators(covariances: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(covariances, full_matrices=False)
    determinants = np.linalg.det(u @ vt)
    corrected = u.copy()
    corrected[:, :, 2] *= np.where(determinants < 0.0, -1.0, 1.0)[:, None]
    return corrected @ vt


def _legacy_transform(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight_sum = float(np.sum(weights))
    reference_center = (weights @ reference) / weight_sum
    mobile_centers = np.einsum("tni,n->ti", coords, weights) / weight_sum
    mobile_centered = coords - mobile_centers[:, None, :]
    reference_centered = reference - reference_center
    covariances = np.einsum(
        "tni,n,nj->tij",
        mobile_centered,
        weights / weight_sum,
        reference_centered,
    )
    row_operators = _row_operators(covariances)
    translations = reference_center[None, :] - np.einsum(
        "ti,tij->tj",
        mobile_centers,
        row_operators,
    )
    return np.swapaxes(row_operators, 1, 2), translations, covariances


def _matmul_covariance(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    normalized = weights / float(np.sum(weights))
    reference_center = normalized @ reference
    mobile_centers = coords.swapaxes(1, 2) @ normalized
    weighted_centered = (coords - mobile_centers[:, None, :]) * normalized[None, :, None]
    return weighted_centered.swapaxes(1, 2) @ (reference - reference_center)


def _median(function, repeats: int) -> float:
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(statistics.median(values))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parity_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    for residues in RESIDUES:
        for frames in FRAMES:
            rng = np.random.default_rng(residues * 10000 + frames)
            reference = rng.normal(size=(residues, 3))
            coords = reference[None, :, :] + rng.normal(
                scale=0.2,
                size=(frames, residues, 3),
            )
            weights = rng.uniform(0.1, 1.0, size=residues)
            old_rotation, old_translation, old_covariance = _legacy_transform(
                coords,
                reference,
                weights,
            )
            new = weighted_kabsch_transform_batch(
                coords,
                reference,
                weights,
                validate_inputs=False,
            )
            new_covariance = _matmul_covariance(coords, reference, weights)
            old_aligned = coords @ np.swapaxes(old_rotation, 1, 2) + old_translation[:, None, :]
            new_aligned = coords @ np.swapaxes(new.rotations, 1, 2) + new.translations[:, None, :]
            parity_rows.append(
                {
                    "n_frames": frames,
                    "n_residues": residues,
                    "max_abs_covariance_delta": float(np.max(np.abs(new_covariance - old_covariance))),
                    "max_abs_rotation_delta": float(np.max(np.abs(new.rotations - old_rotation))),
                    "max_abs_translation_delta": float(np.max(np.abs(new.translations - old_translation))),
                    "max_abs_aligned_coordinate_delta": float(np.max(np.abs(new_aligned - old_aligned))),
                    "max_abs_rmsd_delta": float(
                        np.max(
                            np.abs(
                                np.sqrt(np.mean((new_aligned - reference) ** 2, axis=(1, 2)))
                                - np.sqrt(np.mean((old_aligned - reference) ** 2, axis=(1, 2)))
                            )
                        )
                    ),
                    "proper_rotation": bool(np.allclose(np.linalg.det(new.rotations), 1.0, atol=1e-10)),
                }
            )
            repeats = 7 if frames * residues <= 60_000 else 3
            old_cov_seconds = _median(
                lambda: np.einsum(
                    "tni,n,nj->tij",
                    coords - np.einsum("tni,n->ti", coords, weights)[:, None, :] / np.sum(weights),
                    weights / np.sum(weights),
                    reference - (weights @ reference) / np.sum(weights),
                ),
                repeats,
            )
            new_cov_seconds = _median(
                lambda: _matmul_covariance(coords, reference, weights),
                repeats,
            )
            old_full_seconds = _median(
                lambda: _legacy_transform(coords, reference, weights),
                repeats,
            )
            new_full_seconds = _median(
                lambda: weighted_kabsch_transform_batch(
                    coords,
                    reference,
                    weights,
                    validate_inputs=False,
                ),
                repeats,
            )
            benchmark_rows.append(
                {
                    "n_frames": frames,
                    "n_residues": residues,
                    "residue_frames": frames * residues,
                    "legacy_covariance_seconds": old_cov_seconds,
                    "matmul_covariance_seconds": new_cov_seconds,
                    "covariance_speedup": old_cov_seconds / new_cov_seconds,
                    "legacy_transform_seconds": old_full_seconds,
                    "matmul_transform_seconds": new_full_seconds,
                    "transform_speedup": old_full_seconds / new_full_seconds,
                }
            )
    _write(OUT / "covariance_parity.tsv", parity_rows)
    _write(OUT / "covariance_benchmark.tsv", benchmark_rows)
    median_cov = statistics.median(float(row["covariance_speedup"]) for row in benchmark_rows)
    median_full = statistics.median(float(row["transform_speedup"]) for row in benchmark_rows)
    max_coordinate = max(float(row["max_abs_aligned_coordinate_delta"]) for row in parity_rows)
    (OUT / "covariance_decision.md").write_text(
        "# O1 covariance decision\n\n"
        f"- Median covariance-kernel speedup: {median_cov:.3f}x.\n"
        f"- Median full transform speedup: {median_full:.3f}x.\n"
        f"- Maximum aligned-coordinate difference: {max_coordinate:.3e}.\n"
        "- Decision: retain batched matrix multiplication; compatibility wrappers remain unchanged.\n"
    )


if __name__ == "__main__":
    main()
