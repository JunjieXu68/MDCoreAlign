#!/usr/bin/env python3
"""Evaluate rotation solvers without changing the canonical exact backend."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.linalg import svd as scipy_svd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mdsuperpose.sequential_weights as sequential  # noqa: E402


OUT = ROOT / "results/cpu_optimisation_v1/05_precision_solver"


def _proper_svd(matrix: np.ndarray, backend: str) -> np.ndarray:
    if backend == "numpy_svd":
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    else:
        u, _, vt = scipy_svd(
            matrix,
            full_matrices=False,
            check_finite=False,
            overwrite_a=False,
            lapack_driver=backend.removeprefix("scipy_"),
        )
    operator = u @ vt
    if np.linalg.det(operator) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        operator = u @ vt
    return operator


def _horn(matrix: np.ndarray) -> np.ndarray:
    sxx, sxy, sxz = matrix[0]
    syx, syy, syz = matrix[1]
    szx, szy, szz = matrix[2]
    kernel = np.array(
        [
            [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
            [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
            [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
            [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
        ],
        dtype=np.float64,
    )
    _, vectors = np.linalg.eigh(kernel)
    w, x, y, z = vectors[:, -1]
    column_rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return column_rotation.T


def _matrices() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(41)
    rows: list[tuple[str, np.ndarray]] = []
    for index in range(32):
        rows.append(("full_rank", rng.normal(size=(3, 3))))
        u = rng.normal(size=(3, 1))
        v = rng.normal(size=(1, 3))
        rows.append(("rank_one", u @ v))
        rows.append(("near_rank_deficient", u @ v + 1e-10 * rng.normal(size=(3, 3))))
        reflection = rng.normal(size=(3, 3))
        reflection[:, 2] *= -1.0
        rows.append(("reflection", reflection))
        rows.append(("small_scale", rng.normal(size=(3, 3)) * 1e-10))
        rows.append(("large_scale", rng.normal(size=(3, 3)) * 1e10))
    for path in (
        ROOT / "examples/toy_md/toy_ca_trajectory.pdb",
        ROOT / "examples/real_md_small/trajectory_small.pdb",
    ):
        from mdsuperpose.trajectory import load_trajectory_from_multimodel_pdb

        coords = load_trajectory_from_multimodel_pdb(path).coords
        reference = coords[0]
        for frame in coords[: min(32, coords.shape[0])]:
            rows.append((f"locked_{path.parent.name}", (frame - frame.mean(0)).T @ (reference - reference.mean(0))))
    return rows


def _median(function, repeats: int) -> float:
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    matrices = _matrices()
    backends = ("numpy_svd", "scipy_gesvd", "scipy_gesdd", "horn")
    reference_operators = [_proper_svd(matrix, "scipy_gesvd") for _, matrix in matrices]
    parity: list[dict[str, object]] = []
    benchmark: list[dict[str, object]] = []
    for backend in backends:
        failures = 0
        differences = []
        objective_differences = []
        determinant_errors = []
        for (category, matrix), reference_operator in zip(
            matrices, reference_operators, strict=True
        ):
            try:
                operator = _horn(matrix) if backend == "horn" else _proper_svd(matrix, backend)
                differences.append(float(np.max(np.abs(operator - reference_operator))))
                objective_differences.append(
                    abs(float(np.trace(operator.T @ matrix)) - float(np.trace(reference_operator.T @ matrix)))
                    / max(1.0, float(np.linalg.norm(matrix)))
                )
                determinant_errors.append(abs(float(np.linalg.det(operator)) - 1.0))
                parity.append(
                    {
                        "backend": backend,
                        "category": category,
                        "max_abs_rotation_delta": differences[-1],
                        "relative_objective_delta": objective_differences[-1],
                        "determinant_error": determinant_errors[-1],
                        "finite": bool(np.isfinite(operator).all()),
                    }
                )
            except Exception as error:  # pragma: no cover - recorded diagnostic
                failures += 1
                parity.append(
                    {
                        "backend": backend,
                        "category": category,
                        "max_abs_rotation_delta": "",
                        "relative_objective_delta": "",
                        "determinant_error": "",
                        "finite": False,
                        "error": type(error).__name__,
                    }
                )
        solve = (
            (lambda: [_horn(matrix) for _, matrix in matrices])
            if backend == "horn"
            else (lambda backend=backend: [_proper_svd(matrix, backend) for _, matrix in matrices])
        )
        seconds = _median(solve, repeats=5)
        benchmark.append(
            {
                "backend": backend,
                "matrix_count": len(matrices),
                "total_seconds": seconds,
                "microseconds_per_matrix": seconds / len(matrices) * 1e6,
                "failure_count": failures,
                "max_abs_rotation_delta": max(differences, default=float("nan")),
                "max_relative_objective_delta": max(objective_differences, default=float("nan")),
                "max_determinant_error": max(determinant_errors, default=float("nan")),
            }
        )

    rng = np.random.default_rng(43)
    reference = rng.normal(size=(150, 3))
    coords = reference[None, :, :] + rng.normal(scale=0.15, size=(64, 150, 3))
    original = sequential._scipy_svd
    try:
        for backend in ("numpy_svd", "scipy_gesvd", "scipy_gesdd"):
            if backend == "numpy_svd":
                sequential._scipy_svd = None
            else:
                driver = backend.removeprefix("scipy_")

                def selected_svd(matrix, *, lapack_driver="gesvd", _driver=driver, **kwargs):
                    return scipy_svd(
                        matrix,
                        full_matrices=False,
                        check_finite=False,
                        overwrite_a=True,
                        lapack_driver=_driver,
                    )

                sequential._scipy_svd = selected_svd
            seconds = _median(
                lambda: sequential.estimate_representative_weights_sequential_mean(
                    coords,
                    reference,
                    max_iterations=20,
                    tolerance=1e-6,
                    objective_tolerance=1e-5,
                    objective_patience=2,
                    sequence_continuity_strength=0.05,
                ),
                repeats=3,
            )
            benchmark.append(
                {
                    "backend": backend,
                    "matrix_count": "",
                    "total_seconds": seconds,
                    "microseconds_per_matrix": "",
                    "failure_count": 0,
                    "benchmark_scope": "64x150_exact_mean_estimator",
                }
            )
    finally:
        sequential._scipy_svd = original
    _write(OUT / "svd_numerical_parity.tsv", parity)
    _write(OUT / "svd_backend_benchmark.tsv", benchmark)
    exact_rows = [row for row in benchmark if row.get("benchmark_scope")]
    best = min(exact_rows, key=lambda row: float(row["total_seconds"]))
    canonical = next(row for row in exact_rows if row["backend"] == "scipy_gesvd")
    improvement = float(canonical["total_seconds"]) / float(best["total_seconds"])
    (OUT / "rotation_backend_decision.md").write_text(
        "# Rotation backend decision\n\n"
        f"- Fastest tested exact estimator backend: `{best['backend']}` ({improvement:.3f}x versus SciPy gesvd).\n"
        "- Horn remains experimental because degenerate matrices can select a different equally optimal rotation.\n"
        "- Decision: retain SciPy gesvd for exact unless the measured end-to-end gain is both material and parity-safe.\n"
    )


if __name__ == "__main__":
    main()
