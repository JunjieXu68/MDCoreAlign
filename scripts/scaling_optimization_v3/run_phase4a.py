"""Run Phase 4A environment reconciliation and the minimal re-baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v3"
ENV_ROOT = RESULT_ROOT / "environment"
BASELINE_ROOT = RESULT_ROOT / "baseline"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v3"
CANONICAL_PYTHON = Path("/home/jjxu/miniconda3/bin/python")
INTERPRETERS = (
    Path("/home/jjxu/miniconda3/bin/python"),
    Path("/home/jjxu/miniconda3/envs/mdlovofit/bin/python"),
)
PRIMARY_INPUT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"
    / "runs/4oelA02/10000/transition_aligned_coords.npz"
)
REFERENCE_ROOT = PROJECT_ROOT / "results/scaling_optimization_v2/regression/reference_outputs"
THREAD_COUNTS = (1, 2, 4, 8)
ROBUST_KWARGS: dict[str, object] = {
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


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _thread_environment(thread_count: int) -> dict[str, str]:
    environment = os.environ.copy()
    value = str(thread_count)
    environment.update(
        {
            "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
            "OPENBLAS_NUM_THREADS": value,
            "OMP_NUM_THREADS": value,
            "MKL_NUM_THREADS": value,
            "NUMEXPR_NUM_THREADS": value,
            "VECLIB_MAXIMUM_THREADS": value,
        }
    )
    return environment


def _probe_payload() -> dict[str, object]:
    import importlib.util
    import multiprocessing

    import scipy
    import threadpoolctl

    try:
        import MDAnalysis
    except ImportError:
        mdanalysis_version = None
    else:
        mdanalysis_version = MDAnalysis.__version__

    packages: dict[str, object] = {}
    for name in ("mdcorealign", "mdsuperpose"):
        specification = importlib.util.find_spec(name)
        packages[name] = {
            "available": specification is not None,
            "origin": None if specification is None else specification.origin,
        }
    optional_backends = {
        name: importlib.util.find_spec(name) is not None
        for name in ("cupy", "torch", "jax", "numba")
    }
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = []
    return {
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "numpy_version": np.__version__,
        "numpy_file": np.__file__,
        "scipy_version": scipy.__version__,
        "mdanalysis_version": mdanalysis_version,
        "threadpoolctl_version": threadpoolctl.__version__,
        "threadpool_info": threadpoolctl.threadpool_info(),
        "packages": packages,
        "optional_backends": optional_backends,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "logical_cpu_count": os.cpu_count(),
        "multiprocessing_cpu_count": multiprocessing.cpu_count(),
        "cpu_affinity": affinity,
    }


def _child_environment(output: Path) -> int:
    output.write_text(json.dumps(_probe_payload(), indent=2, sort_keys=True) + "\n")
    return 0


def _load_primary(n_frames: int) -> np.ndarray:
    with np.load(PRIMARY_INPUT, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"][:n_frames], dtype=float)
    if coords.shape[0] != n_frames:
        raise ValueError(f"requested {n_frames} frames, found {coords.shape[0]}")
    return coords


def _child_timing(
    method: str,
    n_frames: int,
    output: Path,
    *,
    memory_sample_interval: float,
) -> int:
    scripts_root = PROJECT_ROOT / "scripts/scaling_optimization"
    sys.path.insert(0, str(scripts_root))
    from profile_support import MemorySampler
    from mdsuperpose.long_mode import solve_long_trajectory_alignment

    total_start = time.perf_counter()
    decode_start = time.perf_counter()
    coords = _load_primary(n_frames)
    decode_seconds = time.perf_counter() - decode_start
    if method == "legacy_object":
        estimator = "legacy"
        refinement_passes = 0
    elif method == "batched_independent":
        estimator = "batched"
        refinement_passes = 0
    elif method == "phase3_batched_refined":
        estimator = "batched"
        refinement_passes = 2
    else:
        raise ValueError(f"unsupported timing method: {method}")

    sampler = MemorySampler(interval_seconds=memory_sample_interval)
    sampler.start()
    try:
        solve_start = time.perf_counter()
        result = solve_long_trajectory_alignment(
            coords,
            mode="long",
            max_weight_fit_frames=min(2000, n_frames),
            frame_selection="uniform",
            random_seed=42,
            chunk_size=min(2000, n_frames),
            store_aligned=False,
            representative_weight_estimator=estimator,
            representative_batch_size=min(512, n_frames),
            representative_warm_refinement_passes=refinement_passes,
            **ROBUST_KWARGS,
        )
        solve_seconds = time.perf_counter() - solve_start
    finally:
        sampler.stop()
    representative = result.subset_result
    if method == "legacy_object":
        iterations = sum(frame.iteration_count for frame in representative.frame_results)
        call_counts = {
            "kabsch_calls": iterations + len(representative.frame_results),
            "svd_calls": iterations + len(representative.frame_results),
            "per_representative_result_objects": len(representative.frame_results),
        }
    else:
        call_counts = dict(representative.call_counts)
    payload = {
        "method": method,
        "n_frames": n_frames,
        "n_residues": coords.shape[1],
        "input_decode_seconds": decode_seconds,
        "solve_seconds": solve_seconds,
        "in_child_end_to_end_seconds": time.perf_counter() - total_start,
        "representative_weight_seconds": result.timing["iterative_weight_estimation"],
        "all_frame_fit_seconds": result.timing["weighted_fit"],
        "rmsf_seconds": result.timing["RMSF"],
        "peak_rss_mb": sampler.to_dict().get("peak_rss_mb"),
        "mean_rmsd": result.mean_rmsd,
        "mean_weighted_rmsd": result.mean_weighted_rmsd,
        "average_iterations": result.average_iterations_per_frame,
        "converged_fraction": result.converged_fraction,
        "refinement_passes": refinement_passes,
        **call_counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _run_child_timing(
    *,
    method: str,
    n_frames: int,
    thread_count: int,
    repeat: int,
    warmup: bool,
    output_root: Path,
    timeout: float = 240.0,
) -> dict[str, object]:
    output = output_root / (
        f"{method}__f{n_frames}__t{thread_count}__r{repeat}"
        f"{'__warmup' if warmup else ''}.json"
    )
    command = [
        str(CANONICAL_PYTHON),
        str(Path(__file__).resolve()),
        "--child-timing",
        "--method",
        method,
        "--n-frames",
        str(n_frames),
        "--output",
        str(output),
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_thread_environment(thread_count),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    row: dict[str, object] = {
        "method": method,
        "n_frames": n_frames,
        "thread_count": thread_count,
        "repeat": repeat,
        "warmup": int(warmup),
        "fresh_process_seconds": elapsed,
        "command": " ".join(command),
    }
    if completed.returncode != 0:
        row.update(
            {
                "status": "failed",
                "return_code": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
        )
        return row
    row.update(json.loads(output.read_text()))
    row["status"] = "success"
    return row


def _audit_interpreters() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    payloads: dict[str, dict[str, object]] = {}
    raw_root = ENV_ROOT / "raw_interpreters"
    raw_root.mkdir(parents=True, exist_ok=True)
    for interpreter in INTERPRETERS:
        label = "base" if interpreter == CANONICAL_PYTHON else "mdlovofit_env"
        output = raw_root / f"{label}.json"
        command = [
            str(interpreter),
            str(Path(__file__).resolve()),
            "--child-environment",
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_thread_environment(1),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            rows.append(
                {
                    "label": label,
                    "requested_interpreter": str(interpreter),
                    "status": "failed",
                    "stderr": completed.stderr.strip(),
                }
            )
            continue
        payload = json.loads(output.read_text())
        payloads[label] = payload
        pip_show = subprocess.run(
            [str(interpreter), "-m", "pip", "show", "mdcorealign"],
            cwd=PROJECT_ROOT,
            env=_thread_environment(1),
            capture_output=True,
            text=True,
            check=False,
        )
        packages = payload["packages"]
        rows.append(
            {
                "label": label,
                "requested_interpreter": str(interpreter),
                "resolved_interpreter": payload["sys_executable"],
                "python_version": payload["python_version"],
                "numpy_version": payload["numpy_version"],
                "numpy_file": payload["numpy_file"],
                "scipy_version": payload["scipy_version"],
                "mdanalysis_version": payload["mdanalysis_version"],
                "threadpoolctl_version": payload["threadpoolctl_version"],
                "mdcorealign_source": packages["mdcorealign"]["origin"],
                "mdsuperpose_source": packages["mdsuperpose"]["origin"],
                "editable_install": int("Editable project location:" in pip_show.stdout),
                "pip_show_status": "installed" if pip_show.returncode == 0 else "not_installed",
                "conda_default_env": payload["conda_default_env"],
                "conda_prefix": payload["conda_prefix"],
                "cupy_available": payload["optional_backends"]["cupy"],
                "torch_available": payload["optional_backends"]["torch"],
                "jax_available": payload["optional_backends"]["jax"],
                "numba_available": payload["optional_backends"]["numba"],
                "logical_cpu_count": payload["logical_cpu_count"],
                "cpu_affinity_count": len(payload["cpu_affinity"]),
                "status": "success",
            }
        )
    return rows, payloads


def _run_thread_calibration() -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    raw_root = ENV_ROOT / "thread_calibration_raw"
    for thread_count in THREAD_COUNTS:
        for n_frames in (500, 10000):
            for method in ("legacy_object", "batched_independent"):
                for repeat in range(4):
                    row = _run_child_timing(
                        method=method,
                        n_frames=n_frames,
                        thread_count=thread_count,
                        repeat=repeat,
                        warmup=repeat == 0,
                        output_root=raw_root,
                    )
                    rows.append(row)
                    print(
                        f"[thread] t={thread_count} f={n_frames} {method} "
                        f"r={repeat} {row['status']}",
                        flush=True,
                    )
    medians: dict[tuple[int, int, str], float] = {}
    for thread_count in THREAD_COUNTS:
        for n_frames in (500, 10000):
            for method in ("legacy_object", "batched_independent"):
                values = [
                    float(row["in_child_end_to_end_seconds"])
                    for row in rows
                    if row["thread_count"] == thread_count
                    and row["n_frames"] == n_frames
                    and row["method"] == method
                    and not row["warmup"]
                    and row["status"] == "success"
                ]
                if values:
                    medians[(thread_count, n_frames, method)] = statistics.median(values)
    scores: dict[int, float] = {}
    for thread_count in THREAD_COUNTS:
        ratios: list[float] = []
        for n_frames in (500, 10000):
            for method in ("legacy_object", "batched_independent"):
                best = min(
                    medians[(candidate, n_frames, method)]
                    for candidate in THREAD_COUNTS
                    if (candidate, n_frames, method) in medians
                )
                ratios.append(medians[(thread_count, n_frames, method)] / best)
        scores[thread_count] = statistics.geometric_mean(ratios)
    selected = min(scores, key=scores.get)
    for row in rows:
        key = (int(row["thread_count"]), int(row["n_frames"]), str(row["method"]))
        row["clean_median_in_child_seconds"] = medians.get(key, "")
        row["aggregate_thread_score"] = scores.get(int(row["thread_count"]), "")
        row["selected_thread_policy"] = int(row["thread_count"] == selected)
    return rows, selected


def _verify_golden_panel() -> dict[str, object]:
    scripts_root = PROJECT_ROOT / "scripts/scaling_optimization"
    sys.path.insert(0, str(scripts_root))
    from build_phase2_regression import build_cases
    from mdsuperpose.core_selection import select_core_by_target_fraction
    from mdsuperpose.long_mode import build_long_flexibility_profile, solve_long_trajectory_alignment

    failures: list[str] = []
    checked = 0
    max_differences = {
        "weight": 0.0,
        "rmsd": 0.0,
        "weighted_rmsd": 0.0,
        "rmsf": 0.0,
        "rotation": 0.0,
        "translation": 0.0,
    }
    for case in build_cases():
        coords, chain = case.loader()
        golden_path = REFERENCE_ROOT / f"{case.case_id}.npz"
        with np.load(golden_path, allow_pickle=False) as payload:
            golden = {
                key.removeprefix("legacy_long_"): np.asarray(payload[key])
                for key in payload.files
                if key.startswith("legacy_long_")
            }
        result = solve_long_trajectory_alignment(
            coords,
            mode="long",
            max_weight_fit_frames=min(2000, coords.shape[0]),
            frame_selection="uniform",
            random_seed=42,
            chunk_size=min(2000, coords.shape[0]),
            store_aligned=False,
            representative_weight_estimator="legacy",
            **ROBUST_KWARGS,
        )
        labels = (
            [f"{value}:{index + 1}:UNK" for index, value in enumerate(chain)]
            if chain is not None
            else None
        )
        profile = build_long_flexibility_profile(
            result,
            residue_labels=labels,
            min_segment_length=3,
            merge_gap=1,
        )
        candidate = {
            "frozen_weights": result.frozen_weights,
            "core_labels": select_core_by_target_fraction(
                result.frozen_weights, 0.6
            ).core_mask.astype(np.uint8),
            "rotations": result.rotations,
            "translations": result.translations,
            "rmsd": result.rmsd,
            "weighted_rmsd": result.weighted_rmsd,
            "robust_rmsf": result.robust_rmsf,
            "ordinary_rmsf": result.ordinary_rmsf,
            "flexible_labels": (profile.flexible_segment_id > 0).astype(np.uint8),
        }
        checked += 1
        case_failed = False
        for key, category in (
            ("frozen_weights", "weight"),
            ("rmsd", "rmsd"),
            ("weighted_rmsd", "weighted_rmsd"),
            ("robust_rmsf", "rmsf"),
            ("rotations", "rotation"),
            ("translations", "translation"),
        ):
            difference = float(np.max(np.abs(candidate[key] - golden[key])))
            max_differences[category] = max(max_differences[category], difference)
            case_failed |= difference > 1e-8
        case_failed |= not np.array_equal(candidate["core_labels"], golden["core_labels"])
        case_failed |= not np.array_equal(
            candidate["flexible_labels"], golden["flexible_labels"]
        )
        if case_failed:
            failures.append(case.case_id)
        print(f"[golden] {case.case_id} {'FAIL' if case_failed else 'pass'}", flush=True)
    return {
        "checked_cases": checked,
        "passed_cases": checked - len(failures),
        "failed_cases": failures,
        "status": "pass" if not failures else "failed",
        **{f"max_{key}_abs_difference": value for key, value in max_differences.items()},
    }


def _verify_frozen_hashes() -> dict[str, object]:
    manifest = PROJECT_ROOT / "results/scaling_optimization_v1/baseline/key_output_hashes.tsv"
    failures: list[dict[str, str]] = []
    checked = 0
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = PROJECT_ROOT / row["path"]
            actual = _sha256(path) if path.is_file() else "missing"
            checked += 1
            if actual != row["sha256"]:
                failures.append(
                    {"path": row["path"], "expected": row["sha256"], "actual": actual}
                )
    golden_manifest = (
        PROJECT_ROOT / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
    )
    golden_failures: list[dict[str, str]] = []
    golden_checked = 0
    with golden_manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") != "success":
                continue
            path = PROJECT_ROOT / row["reference_npz"]
            actual = _sha256(path) if path.is_file() else "missing"
            golden_checked += 1
            if actual != row["reference_sha256"]:
                golden_failures.append(
                    {
                        "path": row["reference_npz"],
                        "expected": row["reference_sha256"],
                        "actual": actual,
                    }
                )
    return {
        "project_completion_checked": checked,
        "project_completion_failures": failures,
        "golden_checked": golden_checked,
        "golden_failures": golden_failures,
        "status": "pass" if not failures and not golden_failures else "failed",
    }


def _run_tests(thread_count: int) -> dict[str, object]:
    output = BASELINE_ROOT / "full_test_results.txt"
    command = [str(CANONICAL_PYTHON), "-m", "pytest", "-q", "tests"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_thread_environment(thread_count),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(completed.stdout + completed.stderr)
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "output_file": output.relative_to(PROJECT_ROOT).as_posix(),
        "last_line": (completed.stdout.strip().splitlines() or [""])[-1],
    }


def _minimal_rebaseline(thread_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_root = BASELINE_ROOT / "raw_runs"
    for n_frames in (500, 10000):
        for method in (
            "legacy_object",
            "batched_independent",
            "phase3_batched_refined",
        ):
            for repeat in range(4):
                row = _run_child_timing(
                    method=method,
                    n_frames=n_frames,
                    thread_count=thread_count,
                    repeat=repeat,
                    warmup=repeat == 0,
                    output_root=raw_root,
                )
                rows.append(row)
                print(
                    f"[baseline] f={n_frames} {method} r={repeat} {row['status']}",
                    flush=True,
                )
    medians: dict[tuple[int, str], tuple[float, float, float]] = {}
    for n_frames in (500, 10000):
        for method in (
            "legacy_object",
            "batched_independent",
            "phase3_batched_refined",
        ):
            selected = [
                row
                for row in rows
                if row["n_frames"] == n_frames
                and row["method"] == method
                and not row["warmup"]
                and row["status"] == "success"
            ]
            if selected:
                medians[(n_frames, method)] = (
                    statistics.median(
                        float(row["in_child_end_to_end_seconds"]) for row in selected
                    ),
                    statistics.median(
                        float(row["fresh_process_seconds"]) for row in selected
                    ),
                    max(float(row["peak_rss_mb"]) for row in selected),
                )
    for row in rows:
        summary = medians.get((int(row["n_frames"]), str(row["method"])))
        if summary:
            row["median_in_child_seconds"] = summary[0]
            row["median_fresh_process_seconds"] = summary[1]
            row["max_peak_rss_mb"] = summary[2]
            legacy = medians.get((int(row["n_frames"]), "legacy_object"))
            if legacy:
                row["speedup_vs_legacy_in_child"] = legacy[0] / summary[0]
    return rows


def _run_mdlovofit_rebaseline(thread_count: int) -> list[dict[str, object]]:
    """Run the external 10k comparator through the frozen Phase 1 wrapper."""

    rows: list[dict[str, object]] = []
    runner = PROJECT_ROOT / "scripts/scaling_optimization/run_profile_matrix.py"
    raw_root = BASELINE_ROOT / "raw_runs"
    raw_root.mkdir(parents=True, exist_ok=True)
    for repeat in range(4):
        warmup = repeat == 0
        output = raw_root / f"external_mdlovofit__f10000__r{repeat}.json"
        command = [
            str(CANONICAL_PYTHON),
            str(runner),
            "--child",
            "--output-root",
            str(RESULT_ROOT),
            "--raw-output",
            str(output),
            "--run-id",
            f"phase4a_mdlovofit_r{repeat}",
            "--dataset-id",
            "4oelA02_transition_aligned",
            "--input-path",
            str(PRIMARY_INPUT),
            "--data-semantics",
            "transition_aligned_synthetic_scaling",
            "--expected-residues",
            "153",
            "--n-frames",
            "10000",
            "--config",
            "mdlovofit",
            "--repeat-id",
            str(repeat),
            "--external-timeout",
            "300",
            "--mdlovofit-executable",
            "/usr/local/bin/mdlovofit",
        ]
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_thread_environment(thread_count),
            capture_output=True,
            text=True,
            check=False,
            timeout=360,
        )
        fresh = time.perf_counter() - start
        row: dict[str, object] = {
            "method": "external_mdlovofit",
            "n_frames": 10000,
            "n_residues": 153,
            "thread_count": thread_count,
            "repeat": repeat,
            "warmup": int(warmup),
            "fresh_process_seconds": fresh,
            "command": " ".join(command),
        }
        if completed.returncode != 0 or not output.is_file():
            row.update(
                {
                    "status": "failed",
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                }
            )
        else:
            payload = json.loads(output.read_text())
            details = payload.get("details", {})
            memory = payload.get("memory", {})
            diagnostics = payload.get("diagnostics", {})
            row.update(
                {
                    "status": payload.get("status", "failed"),
                    "in_child_end_to_end_seconds": payload.get(
                        "stages_seconds", {}
                    ).get("total"),
                    "solve_seconds": details.get("algorithm_runtime_seconds"),
                    "representative_weight_seconds": "",
                    "all_frame_fit_seconds": details.get(
                        "algorithm_runtime_seconds"
                    ),
                    "rmsf_seconds": "",
                    "peak_rss_mb": memory.get("peak_rss_mb"),
                    "mean_rmsd": diagnostics.get("mean_rmsd"),
                    "mean_weighted_rmsd": "",
                    "native_all_atom_rmsd": diagnostics.get(
                        "native_all_atom_rmsd"
                    ),
                    "native_least_mobile_rmsd": diagnostics.get(
                        "native_least_mobile_rmsd"
                    ),
                    "failure": payload.get("failure_or_skip_reason", ""),
                }
            )
        rows.append(row)
        print(f"[baseline] f=10000 external_mdlovofit r={repeat} {row['status']}")
    successful = [
        row for row in rows if row["status"] == "success" and not row["warmup"]
    ]
    if successful:
        median_in_child = statistics.median(
            float(row["in_child_end_to_end_seconds"]) for row in successful
        )
        median_fresh = statistics.median(
            float(row["fresh_process_seconds"]) for row in successful
        )
        peak_rss = max(float(row["peak_rss_mb"]) for row in successful)
        for row in rows:
            row["median_in_child_seconds"] = median_in_child
            row["median_fresh_process_seconds"] = median_fresh
            row["max_peak_rss_mb"] = peak_rss
    return rows


def _capture_environment_files(
    payloads: dict[str, dict[str, object]],
    selected_threads: int,
) -> None:
    ENV_ROOT.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [str(CANONICAL_PYTHON), "-m", "pip", "freeze"],
        cwd=PROJECT_ROOT,
        env=_thread_environment(selected_threads),
        capture_output=True,
        text=True,
        check=True,
    )
    (ENV_ROOT / "canonical_pip_freeze.txt").write_text(freeze.stdout)
    show_config = subprocess.run(
        [
            str(CANONICAL_PYTHON),
            "-c",
            "import numpy as np; np.show_config()",
        ],
        cwd=PROJECT_ROOT,
        env=_thread_environment(selected_threads),
        capture_output=True,
        text=True,
        check=True,
    )
    (ENV_ROOT / "numpy_show_config.txt").write_text(show_config.stdout)
    canonical_payload = payloads["base"]
    (ENV_ROOT / "threadpool_info.json").write_text(
        json.dumps(canonical_payload["threadpool_info"], indent=2, sort_keys=True)
        + "\n"
    )
    canonical = {
        "canonical_python": str(CANONICAL_PYTHON),
        "reason": (
            "This absolute interpreter generated Phase 3, resolves the editable source "
            "tree, and uses the frozen NumPy 1.26.0 numerical environment."
        ),
        "selected_thread_count": selected_threads,
        "thread_environment": {
            key: str(selected_threads)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "phase3_interpreter_evidence": (
            "Phase 3 was invoked with bare python under CONDA_DEFAULT_ENV=base; "
            "the shell resolved it to /home/jjxu/miniconda3/bin/python."
        ),
        "environment_probe": canonical_payload,
    }
    (ENV_ROOT / "canonical_environment.json").write_text(
        json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    )


def _complete_interpreter_artifacts() -> int:
    """Capture per-interpreter package and NumPy details without rerunning benchmarks."""

    raw_root = ENV_ROOT / "raw_interpreters"
    raw_root.mkdir(parents=True, exist_ok=True)
    for interpreter in INTERPRETERS:
        label = "base" if interpreter == CANONICAL_PYTHON else "mdlovofit_env"
        freeze = subprocess.run(
            [str(interpreter), "-m", "pip", "freeze"],
            cwd=PROJECT_ROOT,
            env=_thread_environment(1),
            capture_output=True,
            text=True,
            check=False,
        )
        (raw_root / f"{label}_pip_freeze.txt").write_text(
            freeze.stdout + freeze.stderr
        )
        config = subprocess.run(
            [str(interpreter), "-c", "import numpy as np; np.show_config()"],
            cwd=PROJECT_ROOT,
            env=_thread_environment(1),
            capture_output=True,
            text=True,
            check=False,
        )
        (raw_root / f"{label}_numpy_show_config.txt").write_text(
            config.stdout + config.stderr
        )
        explicit = subprocess.run(
            [
                "/home/jjxu/miniconda3/bin/conda",
                "list",
                "--explicit",
                "-p",
                str(interpreter.parent.parent),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        (raw_root / f"{label}_conda_explicit.txt").write_text(
            explicit.stdout + explicit.stderr
        )
    cpu = subprocess.run(
        ["lscpu"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (ENV_ROOT / "cpu_topology.txt").write_text(cpu.stdout + cpu.stderr)
    return 0


def _write_report(
    *,
    selected_threads: int,
    test_result: dict[str, object],
    golden_result: dict[str, object],
    integrity: dict[str, object],
    baseline_rows: list[dict[str, object]],
) -> None:
    def median(n_frames: int, method: str, field: str) -> float:
        values = [
            float(row[field])
            for row in baseline_rows
            if int(row["n_frames"]) == n_frames
            and row["method"] == method
            and not int(row["warmup"])
            and row["status"] == "success"
        ]
        return statistics.median(values)

    legacy_10k = median(10000, "legacy_object", "in_child_end_to_end_seconds")
    independent_10k = median(
        10000, "batched_independent", "in_child_end_to_end_seconds"
    )
    refined_10k = median(
        10000, "phase3_batched_refined", "in_child_end_to_end_seconds"
    )
    mdlovofit_values = [
        float(row["in_child_end_to_end_seconds"])
        for row in baseline_rows
        if row["method"] == "external_mdlovofit"
        and not int(row["warmup"])
        and row["status"] == "success"
    ]
    mdlovofit_row = (
        f"| external MDLovoFit | {statistics.median(mdlovofit_values):.6f} |"
        if mdlovofit_values
        else "| external MDLovoFit | unavailable |"
    )
    phase3_change = (refined_10k - 3.029171471018344) / 3.029171471018344
    report = f"""# Phase 4A environment reconciliation

## Canonical environment

- Python: `{CANONICAL_PYTHON}`
- Fixed BLAS/thread policy: `{selected_threads}` thread(s)
- Input: `results/mdcath_benchmark/transition_aligned_long_mode_benchmark/runs/4oelA02/10000/transition_aligned_coords.npz`
- Data semantics: transition-aligned synthetic scaling; no biological-continuity claim.

The base interpreter was selected because it generated the Phase 3 evidence,
resolves the editable source tree, and uses Python 3.11.4 with NumPy 1.26.0.
The dedicated `mdlovofit` interpreter uses a different Python/NumPy stack and
does not have MDCoreAlign installed; its timing is therefore not mixed into
this comparison.

## Minimal re-baseline

| Method | 10k in-child median (s) |
|---|---:|
| legacy_object | {legacy_10k:.6f} |
| batched_independent (0 refinement passes) | {independent_10k:.6f} |
| Phase 3 batched refined candidate (2 passes) | {refined_10k:.6f} |
{mdlovofit_row}

The Phase 3 refined-candidate timing changed by {phase3_change * 100:.2f}% from
the previously reported 3.029171 s. This is
{"within" if abs(phase3_change) <= 0.20 else "outside"} the 20% stop threshold.
`batched_independent` and the Phase 3 refined candidate are reported separately
because they do not have identical initialisation semantics.

## Verification

- Full tests: `{test_result["status"]}`; `{test_result["last_line"]}`
- Golden legacy panel: `{golden_result["passed_cases"]}/{golden_result["checked_cases"]}` passed
- Frozen project-completion and golden hashes: `{integrity["status"]}`
- Git metadata: not initialised or modified.

## Exact commands

```bash
PY={CANONICAL_PYTHON}
OPENBLAS_NUM_THREADS={selected_threads} OMP_NUM_THREADS={selected_threads} \\
MKL_NUM_THREADS={selected_threads} NUMEXPR_NUM_THREADS={selected_threads} \\
PYTHONPATH=src:. "$PY" scripts/scaling_optimization_v3/run_phase4a.py
```

The calibration and baseline children were also launched with the same
absolute interpreter; exact child commands are retained in the TSV tables.
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "00_environment_reconciliation.md").write_text(report)


def _parent_main() -> int:
    if Path(sys.executable).resolve() != CANONICAL_PYTHON.resolve():
        raise RuntimeError(
            f"Phase 4A must run with {CANONICAL_PYTHON}, got {sys.executable}"
        )
    ENV_ROOT.mkdir(parents=True, exist_ok=True)
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    interpreter_rows, payloads = _audit_interpreters()
    _write_tsv(ENV_ROOT / "interpreter_comparison.tsv", interpreter_rows)

    thread_rows, selected_threads = _run_thread_calibration()
    _write_tsv(ENV_ROOT / "thread_calibration.tsv", thread_rows)
    _capture_environment_files(payloads, selected_threads)

    test_result = _run_tests(selected_threads)
    (BASELINE_ROOT / "test_status.json").write_text(
        json.dumps(test_result, indent=2, sort_keys=True) + "\n"
    )
    golden_result = _verify_golden_panel()
    (BASELINE_ROOT / "golden_regression_status.json").write_text(
        json.dumps(golden_result, indent=2, sort_keys=True) + "\n"
    )
    baseline_rows = _minimal_rebaseline(selected_threads)
    baseline_rows.extend(_run_mdlovofit_rebaseline(selected_threads))
    _write_tsv(BASELINE_ROOT / "minimal_rebaseline.tsv", baseline_rows)
    integrity = _verify_frozen_hashes()
    (BASELINE_ROOT / "integrity_after_phase4a.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    _write_report(
        selected_threads=selected_threads,
        test_result=test_result,
        golden_result=golden_result,
        integrity=integrity,
        baseline_rows=baseline_rows,
    )
    stop = (
        test_result["status"] != "pass"
        or golden_result["status"] != "pass"
        or integrity["status"] != "pass"
    )
    return 2 if stop else 0


def _resume_after_protection_stop() -> int:
    """Resume only the corrected test scope and omitted external comparator."""

    canonical = json.loads((ENV_ROOT / "canonical_environment.json").read_text())
    selected_threads = int(canonical["selected_thread_count"])
    test_result = _run_tests(selected_threads)
    (BASELINE_ROOT / "test_status.json").write_text(
        json.dumps(test_result, indent=2, sort_keys=True) + "\n"
    )
    with (BASELINE_ROOT / "minimal_rebaseline.tsv").open(newline="") as handle:
        baseline_rows = list(csv.DictReader(handle, delimiter="\t"))
    baseline_rows = [
        row for row in baseline_rows if row.get("method") != "external_mdlovofit"
    ]
    baseline_rows.extend(_run_mdlovofit_rebaseline(selected_threads))
    _write_tsv(BASELINE_ROOT / "minimal_rebaseline.tsv", baseline_rows)
    golden_result = json.loads(
        (BASELINE_ROOT / "golden_regression_status.json").read_text()
    )
    integrity = _verify_frozen_hashes()
    (BASELINE_ROOT / "integrity_after_phase4a.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    _write_report(
        selected_threads=selected_threads,
        test_result=test_result,
        golden_result=golden_result,
        integrity=integrity,
        baseline_rows=baseline_rows,
    )
    return 0 if test_result["status"] == "pass" and integrity["status"] == "pass" else 2


def _finalize_existing_phase4a() -> int:
    """Finalize reports from already completed Phase 4A raw measurements."""

    canonical = json.loads((ENV_ROOT / "canonical_environment.json").read_text())
    selected_threads = int(canonical["selected_thread_count"])
    test_result = json.loads((BASELINE_ROOT / "test_status.json").read_text())
    golden_result = json.loads(
        (BASELINE_ROOT / "golden_regression_status.json").read_text()
    )
    with (BASELINE_ROOT / "minimal_rebaseline.tsv").open(newline="") as handle:
        baseline_rows = list(csv.DictReader(handle, delimiter="\t"))
    integrity = _verify_frozen_hashes()
    (BASELINE_ROOT / "integrity_after_phase4a.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    _write_report(
        selected_threads=selected_threads,
        test_result=test_result,
        golden_result=golden_result,
        integrity=integrity,
        baseline_rows=baseline_rows,
    )
    return (
        0
        if test_result["status"] == "pass"
        and golden_result["status"] == "pass"
        and integrity["status"] == "pass"
        else 2
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-environment", action="store_true")
    parser.add_argument("--child-timing", action="store_true")
    parser.add_argument("--resume-after-protection-stop", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--complete-interpreter-artifacts", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--n-frames", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory-sample-interval", type=float, default=0.01)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.child_environment:
        if args.output is None:
            raise ValueError("--output is required")
        return _child_environment(args.output)
    if args.child_timing:
        if args.output is None or args.method is None or args.n_frames is None:
            raise ValueError("--method, --n-frames and --output are required")
        return _child_timing(
            args.method,
            args.n_frames,
            args.output,
            memory_sample_interval=args.memory_sample_interval,
        )
    if args.resume_after_protection_stop:
        return _resume_after_protection_stop()
    if args.finalize_existing:
        return _finalize_existing_phase4a()
    if args.complete_interpreter_artifacts:
        return _complete_interpreter_artifacts()
    return _parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
