#!/usr/bin/env python
"""Run Phase 6A alignment-free local-window controls and timing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

from mdsuperpose.window_scan import scan_window_local


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
INPUT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"
    / "runs/4oelA02/10000/transition_aligned_coords.npz"
)
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v4"
WINDOW_ROOT = RESULT_ROOT / "window_local"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v4"
THREADS = 2


def environment() -> dict[str, str]:
    """Return the locked canonical execution environment."""

    values = os.environ.copy()
    values.update(
        {
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "OMP_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "NUMEXPR_NUM_THREADS": str(THREADS),
            "PYTHONPATH": "src:.",
        }
    )
    return values


def sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a union-schema TSV."""

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


def verify_hashes() -> dict[str, object]:
    """Verify the frozen Phase 0 and Phase 2 evidence hashes."""

    failures: list[str] = []
    project_manifest = (
        PROJECT_ROOT / "results/scaling_optimization_v1/baseline/key_output_hashes.tsv"
    )
    with project_manifest.open(newline="") as handle:
        project_rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in project_rows:
        path = PROJECT_ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            failures.append(row["path"])
    golden_manifest = (
        PROJECT_ROOT / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
    )
    with golden_manifest.open(newline="") as handle:
        golden_rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["status"] == "success"
        ]
    for row in golden_rows:
        path = PROJECT_ROOT / row["reference_npz"]
        if not path.is_file() or sha256(path) != row["reference_sha256"]:
            failures.append(row["reference_npz"])
    return {
        "status": "pass" if not failures else "failed",
        "project_completion_checked": len(project_rows),
        "golden_checked": len(golden_rows),
        "failures": failures,
    }


def load_prefix(n_frames: int) -> tuple[np.ndarray, float]:
    """Decode the fixed synthetic scaling input and return a prefix."""

    start = time.perf_counter()
    with np.load(INPUT, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"][:n_frames], dtype=np.float64)
    return coords, time.perf_counter() - start


def child_run(n_frames: int, statistic: str) -> int:
    """Execute one isolated scan and emit a JSON timing record."""

    process_start = time.perf_counter()
    coords, decode_seconds = load_prefix(n_frames)
    preprocessing_start = time.perf_counter()
    if coords.ndim != 3 or coords.shape[2] != 3 or not np.isfinite(coords).all():
        raise ValueError("invalid benchmark coordinates")
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    result = scan_window_local(
        coords,
        window_size=5,
        feature_set="minimal",
        statistic=statistic,
        mapping="rank",
        descriptor_chunk_size=2000,
        sample_count=512,
    )
    payload = {
        "status": "success",
        "n_frames": n_frames,
        "n_residues": coords.shape[1],
        "statistic": statistic,
        "decode_seconds": decode_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        **result.timing,
        **result.metadata,
        "child_full_workflow_seconds": time.perf_counter() - process_start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def benchmark_scan() -> list[dict[str, object]]:
    """Run warm-up and clean fresh-process timings."""

    rows: list[dict[str, object]] = []
    for n_frames in (500, 1000, 5000, 10000):
        repeats = 5 if n_frames == 10000 else 3
        for statistic in ("relative_mad", "winsorised_variance"):
            for repeat in range(repeats + 1):
                command = [
                    str(PYTHON),
                    str(Path(__file__).resolve()),
                    "--child",
                    "--n-frames",
                    str(n_frames),
                    "--statistic",
                    statistic,
                ]
                start = time.perf_counter()
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment(),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                fresh_seconds = time.perf_counter() - start
                row: dict[str, object] = {
                    "case_id": "synthetic_scaling_4oelA02",
                    "source": INPUT.relative_to(PROJECT_ROOT).as_posix(),
                    "output_contract": "scan_only",
                    "n_frames": n_frames,
                    "n_residues": 153,
                    "window_size": 5,
                    "feature_set": "minimal",
                    "statistic": statistic,
                    "mapping": "rank",
                    "random_seed": 42,
                    "repeat": repeat,
                    "warmup": int(repeat == 0),
                    "fresh_process_seconds": fresh_seconds,
                    "return_code": completed.returncode,
                }
                try:
                    row.update(json.loads(completed.stdout.strip().splitlines()[-1]))
                except (IndexError, json.JSONDecodeError):
                    row["status"] = "failed"
                    row["stderr"] = completed.stderr.strip()
                rows.append(row)
    for row in rows:
        selected = [
            candidate
            for candidate in rows
            if candidate["n_frames"] == row["n_frames"]
            and candidate["statistic"] == row["statistic"]
            and not candidate["warmup"]
            and candidate.get("status") == "success"
        ]
        if selected:
            row["median_scan_only_seconds"] = statistics.median(
                float(candidate["total_scan_only"]) for candidate in selected
            )
            row["median_fresh_process_seconds"] = statistics.median(
                float(candidate["fresh_process_seconds"]) for candidate in selected
            )
    return rows


def _toy_reference(n_residues: int = 30) -> np.ndarray:
    index = np.arange(n_residues, dtype=np.float64)
    return np.column_stack(
        (3.8 * index, 2.0 * np.sin(index / 3.0), 1.5 * np.cos(index / 4.0))
    )


def synthetic_controls() -> list[dict[str, object]]:
    """Run the ten deterministic scientific controls required by the plan."""

    rng = np.random.default_rng(42)
    reference = _toy_reference()
    identical = np.repeat(reference[None], 40, axis=0)
    translation = np.asarray(
        [reference + index * np.asarray((0.4, -0.2, 0.1)) for index in range(40)]
    )
    rotations = []
    for index in range(40):
        angle = index * 0.05
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        rotations.append(reference @ rotation.T + np.asarray((2.0, -1.0, 0.5)))
    local_loop = identical.copy()
    local_loop[:, 12:17] += rng.normal(scale=0.8, size=(40, 5, 3))
    outlier = identical.copy()
    outlier[:, 14] += rng.normal(scale=1.2, size=(40, 3))
    tail = identical.copy()
    tail[:, -5:] += rng.normal(scale=0.8, size=(40, 5, 3))
    chain_case = identical.copy()
    chain_case[:, 4:6] += rng.normal(scale=0.8, size=(40, 2, 3))
    gap_case = chain_case.copy()
    collinear_reference = np.column_stack(
        (np.arange(30, dtype=float), np.zeros((30, 2)))
    )
    collinear = np.repeat(collinear_reference[None], 40, axis=0)
    cases: list[tuple[str, np.ndarray, dict[str, object]]] = [
        ("identical_frames", identical, {}),
        ("whole_protein_translation", translation, {}),
        ("whole_protein_rotation", np.asarray(rotations), {}),
        ("gaussian_local_loop", local_loop, {}),
        ("single_residue_outlier", outlier, {}),
        ("flexible_terminal_tail", tail, {}),
        (
            "two_chains_no_crossing",
            chain_case,
            {"chain_ids": np.asarray(["A"] * 15 + ["B"] * 15)},
        ),
        (
            "missing_residue_gap",
            gap_case,
            {
                "residue_positions": np.asarray(
                    list(range(1, 16)) + list(range(20, 35))
                )
            },
        ),
        ("collinear_degeneracy", collinear, {}),
        ("fixed_seed_repeatability", local_loop, {}),
    ]
    rows: list[dict[str, object]] = []
    for case_id, coords, kwargs in cases:
        first = scan_window_local(coords, feature_set="full", **kwargs)
        second = (
            scan_window_local(coords, feature_set="full", **kwargs)
            if case_id == "fixed_seed_repeatability"
            else first
        )
        rigid = case_id in {
            "identical_frames",
            "whole_protein_translation",
            "whole_protein_rotation",
            "collinear_degeneracy",
        }
        localisation = float(np.mean(first.movement_percentile[10:19]))
        distant = float(
            np.mean(
                np.concatenate(
                    (first.movement_percentile[:6], first.movement_percentile[23:])
                )
            )
        )
        if rigid:
            passed = float(np.max(first.movement_score)) < 1e-12
        elif case_id == "gaussian_local_loop":
            passed = localisation > distant + 0.25
        elif case_id == "single_residue_outlier":
            passed = first.movement_percentile[14] >= 0.8
        elif case_id == "flexible_terminal_tail":
            passed = float(np.mean(first.movement_percentile[-5:])) > 0.6
        elif case_id == "two_chains_no_crossing":
            passed = float(np.max(first.movement_score[15:])) < 1e-12
        elif case_id == "missing_residue_gap":
            passed = float(np.max(first.movement_score[15:])) < 1e-12
        else:
            passed = np.array_equal(
                first.candidate_stable_label,
                second.candidate_stable_label,
            )
        rows.append(
            {
                "case_id": case_id,
                "status": "pass" if passed else "fail",
                "max_movement_index": float(np.max(first.movement_score)),
                "localisation_score": localisation,
                "distant_score": distant,
                "candidate_core_fraction": float(
                    np.mean(first.candidate_stable_label)
                ),
                "n_edges": first.edges.n_edges,
                "output_contract": "scan_only",
                "note": "movement index is not RMSD",
            }
        )
    return rows


def write_environment_and_integrity() -> None:
    """Freeze the exact Phase 6 starting environment and evidence status."""

    import scipy

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    environment_payload = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "thread_environment": {
            name: environment()[name]
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "random_seed": 42,
        "input_sha256": sha256(INPUT),
        "phase5_handoff_sha256": sha256(
            PROJECT_ROOT
            / "results/scaling_optimization_v3/handoff_phase5_20260728.tar.gz"
        ),
    }
    (RESULT_ROOT / "environment_phase6.json").write_text(
        json.dumps(environment_payload, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "integrity_before_phase6.json").write_text(
        json.dumps(verify_hashes(), indent=2, sort_keys=True) + "\n"
    )


def write_report(
    timings: list[dict[str, object]],
    controls: list[dict[str, object]],
) -> None:
    """Write the Phase 6A design and measured feasibility decision."""

    medians: dict[str, float] = {}
    for statistic in ("relative_mad", "winsorised_variance"):
        values = [
            float(row["total_scan_only"])
            for row in timings
            if row["n_frames"] == 10000
            and row["statistic"] == statistic
            and not row["warmup"]
            and row.get("status") == "success"
        ]
        medians[statistic] = statistics.median(values)
    controls_passed = sum(row["status"] == "pass" for row in controls)
    continue_gate = (
        controls_passed == len(controls)
        and medians["winsorised_variance"] < 1.0
    )
    report = f"""# Phase 6A direct local-window design and feasibility

## Output contract

`window_local_w5` is an alignment-free scan. It evaluates sparse squared
C-alpha distances and reports a movement index, soft stability weights and a
candidate core. The movement index is **not RMSD**. This stage produces no
rigid transform and no RMSF.

## Descriptor

The minimal W5 representation retains unique within-segment sequence
separations 2--4. Chain changes and explicit missing-residue gaps break the
window. The implementation creates an `(T, E)` sparse descriptor array and
never creates a dense `(T, N, N)` array.

Two pre-specified robust summaries were measured:

- relative MAD uses every frame and two exact medians;
- winsorised relative variance estimates its clipping centre from 512 uniform
  frames and streams clipped first/second moments across all frames.

Movement is mapped to stability by deterministic reverse rank, followed by one
chain-aware smoothing pass with strength 0.05. The candidate core is the fixed
top 60% of weights.

## Controls

- Passed: {controls_passed}/{len(controls)}.
- Rigid translation/rotation invariance: {"pass" if all(row["status"] == "pass" for row in controls[:3]) else "fail"}.
- Chain/gap isolation and fixed-seed repeatability: {"pass" if controls_passed == len(controls) else "fail"}.

## 10k feasibility timing

- Relative MAD median scan-only: {medians["relative_mad"]:.6f} s.
- Winsorised relative-variance median scan-only: {medians["winsorised_variance"]:.6f} s.
- Canonical CPU environment: `/home/jjxu/miniconda3/bin/python`, float64,
  MKL/OpenBLAS/OMP/NumExpr fixed to two threads.
- Input: 10,000-frame, 153-residue transition-aligned synthetic scaling
  trajectory. It is not real continuous MD.

The exact-MAD path {"passes" if medians["relative_mad"] < 1.0 else "misses"} the
1.0 s scan gate; full-matrix median/absolute-deviation aggregation nevertheless
dominates its runtime. The pre-specified streaming winsorised path
{"passes" if medians["winsorised_variance"] < 1.0 else "misses"} the gate.
Phase 6A continuation decision:
**{"continue to Phase 6B" if continue_gate else "stop"}**.

## Reproducible command

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. \\
/home/jjxu/miniconda3/bin/python \\
scripts/scaling_optimization_v4/run_phase6a.py
```
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "00_window_local_design.md").write_text(report)
    (WINDOW_ROOT / "phase6a_summary.json").write_text(
        json.dumps(
            {
                "controls_passed": controls_passed,
                "controls_total": len(controls),
                "relative_mad_10k_median_seconds": medians["relative_mad"],
                "winsorised_variance_10k_median_seconds": medians[
                    "winsorised_variance"
                ],
                "continue_to_phase6b": continue_gate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def parent_run() -> int:
    """Run and persist the complete Phase 6A package."""

    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use {PYTHON}; got {sys.executable}")
    WINDOW_ROOT.mkdir(parents=True, exist_ok=True)
    write_environment_and_integrity()
    controls = synthetic_controls()
    write_tsv(WINDOW_ROOT / "synthetic_controls.tsv", controls)
    timings = benchmark_scan()
    write_tsv(WINDOW_ROOT / "scan_timing.tsv", timings)
    write_report(timings, controls)
    summary = json.loads((WINDOW_ROOT / "phase6a_summary.json").read_text())
    return 0 if summary["continue_to_phase6b"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--n-frames", type=int)
    parser.add_argument(
        "--statistic",
        choices=("relative_mad", "winsorised_variance"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.child:
        if args.n_frames is None or args.statistic is None:
            raise ValueError("--n-frames and --statistic are required with --child")
        return child_run(args.n_frames, args.statistic)
    return parent_run()


if __name__ == "__main__":
    raise SystemExit(main())
