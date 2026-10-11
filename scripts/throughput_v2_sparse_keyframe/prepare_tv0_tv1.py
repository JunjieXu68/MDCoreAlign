#!/usr/bin/env python3
"""Freeze the Throughput-v2 study baseline and lock its contracts and gates."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "throughput_v2_sparse_keyframe"

PHASE_DIRS = (
    "00_freeze",
    "01_baseline_contract",
    "02_current_throughput_audit",
    "03_sparse_landmarks",
    "04_keyframes",
    "05_profile_sampling",
    "06_adaptive_refinement",
    "07_fresh_validation",
    "08_performance",
    "09_decision",
    "handoff_to_chatgpt",
)

SOURCE_FILES = (
    "pyproject.toml",
    "src/mdcorealign/__init__.py",
    "src/mdcorealign/cli.py",
    "src/mdcorealign/contracts.py",
    "src/mdcorealign/experimental.py",
    "src/mdcorealign/experimental_gpu_batch.py",
    "src/mdcorealign/gpu_backend.py",
    "src/mdcorealign/pdb_writer.py",
    "src/mdsuperpose/all_frame_engine.py",
    "src/mdsuperpose/fast_kabsch.py",
    "src/mdsuperpose/representative_weights.py",
    "src/mdsuperpose/sequential_weights.py",
    "src/mdsuperpose/window_kinematics.py",
    "src/mdsuperpose/window_refine.py",
    "src/mdsuperpose/window_scan.py",
)

REFERENCE_FILES = (
    "results/cpu_chronology_v3/05_fresh_panels/fresh_mdcath_512.tsv",
    "results/cpu_chronology_v3/05_fresh_panels/fresh_mdcath_512_lock.json",
    "results/cpu_chronology_v3/05_fresh_panels/fresh_long_panel.tsv",
    "results/cpu_chronology_v3/05_fresh_panels/fresh_mdcath_performance_128.tsv",
    "results/cpu_chronology_v3/05_fresh_panels/historical_stress_panel.tsv",
    "results/cpu_chronology_v3/06_validation/fresh_mdcath_quality.tsv",
    "results/cpu_chronology_v3/06_validation/fresh_long_quality.tsv",
    "results/cpu_chronology_v3/07_performance/fresh_long_repeats.tsv",
    "results/cpu_chronology_v3/07_performance/fresh_mdcath_128_repeats.tsv",
    "results/cpu_chronology_v3/07_performance/stage_decomposition_v3.tsv",
    "results/cpu_chronology_v3/08_decision/tail_case_summary.tsv",
    "results/gpu_matched_validation_v1/u2_real_panel/real_continuous_panel.tsv",
    "results/gpu_matched_validation_v1/u2_real_panel/real_panel_internal_matched_timings.tsv",
    "results/gpu_matched_validation_v1/u5_quality/system_level_quality.tsv",
    "results/gpu_matched_validation_v1/u5_quality/quality_gate_summary.json",
    "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/speedup_vs_target_frames.tsv",
    "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/scaling_model_summary.tsv",
    "results/long_mode_optimization_v2/ultra_long_validation/rep_frame_selector_ablation.tsv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _command_output(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def _freeze_sources() -> None:
    freeze = OUTPUT / "00_freeze"
    rows: list[dict[str, object]] = []
    existing: list[Path] = []
    for relative in SOURCE_FILES:
        path = ROOT / relative
        rows.append(
            {
                "repository_relative_path": relative,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "sha256": _sha256(path) if path.is_file() else "",
                "role": "frozen_source",
            }
        )
        if path.is_file():
            existing.append(path)
    _write_tsv(
        freeze / "source_manifest.tsv",
        ["repository_relative_path", "exists", "size_bytes", "sha256", "role"],
        rows,
    )
    checkpoint = freeze / "source_checkpoint.tar.gz"
    with tarfile.open(checkpoint, "w:gz") as archive:
        for path in existing:
            archive.add(path, arcname=path.relative_to(ROOT))


def _freeze_references() -> None:
    rows: list[dict[str, object]] = []
    for relative in REFERENCE_FILES:
        path = ROOT / relative
        rows.append(
            {
                "repository_relative_path": relative,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "sha256": _sha256(path) if path.is_file() else "",
                "role": "frozen_reference_evidence",
                "use_policy": "read_only; reuse only when input and contract hashes match",
            }
        )
    _write_tsv(
        OUTPUT / "00_freeze" / "frozen_reference_manifest.tsv",
        [
            "repository_relative_path",
            "exists",
            "size_bytes",
            "sha256",
            "role",
            "use_policy",
        ],
        rows,
    )


def _environment() -> None:
    try:
        import scipy
    except ImportError:
        scipy_version = "unavailable"
    else:
        scipy_version = scipy.__version__
    try:
        import cupy
    except ImportError:
        cupy_version = "unavailable"
    else:
        cupy_version = cupy.__version__
    _, cpu_model = _command_output(
        ["bash", "-lc", "lscpu | awk -F: '/Model name/{gsub(/^ +/,\"\",$2); print $2; exit}'"]
    )
    gpu_code, gpu_model = _command_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    git_head_code, git_head = _command_output(["git", "rev-parse", "HEAD"])
    git_status_code, git_status = _command_output(["git", "status", "--short", "--branch"])
    rows = [
        {"key": "generated_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"key": "python_executable", "value": sys.executable},
        {"key": "python_version", "value": platform.python_version()},
        {"key": "numpy_version", "value": np.__version__},
        {"key": "scipy_version", "value": scipy_version},
        {"key": "cupy_version", "value": cupy_version},
        {"key": "cpu_model", "value": cpu_model},
        {"key": "physical_logical_cpu_count", "value": f"{os.cpu_count()} logical; see lscpu snapshot"},
        {"key": "gpu", "value": gpu_model if gpu_code == 0 else "unavailable"},
        {"key": "OMP_NUM_THREADS", "value": os.environ.get("OMP_NUM_THREADS", "unset")},
        {"key": "MKL_NUM_THREADS", "value": os.environ.get("MKL_NUM_THREADS", "unset")},
        {"key": "OPENBLAS_NUM_THREADS", "value": os.environ.get("OPENBLAS_NUM_THREADS", "unset")},
        {"key": "package_version", "value": "0.1.0rc1"},
        {"key": "git_valid_head", "value": str(git_head_code == 0)},
        {"key": "git_head", "value": git_head if git_head_code == 0 else "unavailable; repository has no commits"},
        {"key": "git_status_read_succeeded", "value": str(git_status_code == 0)},
        {"key": "git_status", "value": git_status.replace("\n", " | ")},
    ]
    _write_tsv(OUTPUT / "00_freeze" / "environment_manifest.tsv", ["key", "value"], rows)
    _, lscpu = _command_output(["lscpu"])
    _, numpy_config = _command_output(
        [sys.executable, "-c", "import numpy as n; n.show_config()"]
    )
    (OUTPUT / "00_freeze" / "environment_details.txt").write_text(
        f"lscpu\n=====\n{lscpu}\n\nNumPy configuration\n===================\n{numpy_config}\n",
        encoding="utf-8",
    )


def _write_scope() -> None:
    text = """# Throughput-v2 Sparse/Keyframe Study Scope

This study is isolated from the public exact and current throughput paths. The
current throughput implementation and its frozen results are the primary
baseline. The study tests whether sparse residue landmarks, temporal keyframes,
sampled profile statistics, and deterministic local refinement create a new
quality-speed Pareto point under matched compact-output contracts.

The study does not change public defaults, exact chronology, package version,
Git metadata, manuscript claims, or historical evidence. It does not rerun
MDLovoFit, THESEUS, W5 publication benchmarks, full aligned-coordinate output,
or the full 5,397-domain external comparison.

The repository has no valid Git HEAD. No commit hash is invented. Source parity
is instead recorded with a path-level SHA-256 manifest and a source checkpoint.

Current-throughput transforms are not universally present in frozen caches.
Where absent, transform comparison is limited to prospectively registered audit
frames. Whole-trajectory current-throughput evaluation is not allowed on fresh
validation systems.
"""
    (OUTPUT / "00_freeze" / "study_scope.md").write_text(text, encoding="utf-8")


def _write_contracts_and_gates() -> None:
    baseline = OUTPUT / "01_baseline_contract"
    contracts = [
        {
            "contract": "current_throughput_compact",
            "stable_weights": "current fixed-budget throughput estimator",
            "transforms": "explicit every frame",
            "rmsd": "full-selection per frame",
            "weighted_rmsd": "per frame",
            "stable_rmsf": "full-frame",
            "ordinary_rmsf": "no",
            "aligned_coordinates": "no",
            "serialisation_timed": "no",
        },
        {
            "contract": "v2_sparse_align",
            "stable_weights": "identical current throughput estimator",
            "transforms": "explicit every frame from sparse landmarks",
            "rmsd": "full-selection per frame",
            "weighted_rmsd": "per frame",
            "stable_rmsf": "full-frame",
            "ordinary_rmsf": "no",
            "aligned_coordinates": "no",
            "serialisation_timed": "no",
        },
        {
            "contract": "v2_keyframe_align",
            "stable_weights": "identical current throughput estimator",
            "transforms": "explicit keyframes plus interpolation",
            "rmsd": "full-selection per frame under interpolated transforms",
            "weighted_rmsd": "per frame",
            "stable_rmsf": "full-frame",
            "ordinary_rmsf": "no",
            "aligned_coordinates": "no",
            "serialisation_timed": "no",
        },
        {
            "contract": "v2_profile",
            "stable_weights": "identical current throughput estimator",
            "transforms": "sampled only; not required every frame",
            "rmsd": "approximate summary with uncertainty",
            "weighted_rmsd": "approximate summary with uncertainty",
            "stable_rmsf": "sampled weighted estimate with uncertainty",
            "ordinary_rmsf": "no",
            "aligned_coordinates": "no",
            "serialisation_timed": "no",
        },
    ]
    _write_tsv(baseline / "output_contracts.tsv", list(contracts[0]), contracts)

    gate_rows: list[dict[str, object]] = []
    balanced = {
        "catastrophic_mismatch_count": "=0",
        "median_core_jaccard": ">=0.95",
        "p5_core_jaccard": ">=0.85",
        "minimum_non_tail_core_jaccard": ">=0.75",
        "median_stable_weight_spearman": ">=0.995",
        "p5_stable_weight_spearman": ">=0.98",
        "median_per_frame_rmsd_abs_diff_A": "<=0.10",
        "p95_of_case_p95_rmsd_abs_diff_A": "<=0.50",
        "median_stable_rmsf_spearman": ">=0.995",
        "p5_stable_rmsf_spearman": ">=0.98",
        "median_rmsf_mae_A": "<=0.10",
        "unresolved_interval_rate": "<=0.02",
        "atlas_10k_speedup": ">=2.0",
        "control_50k_speedup": ">=3.0",
        "control_100k_speedup": ">=5.0",
    }
    fast = {
        "catastrophic_mismatch_count": "=0",
        "median_core_jaccard": ">=0.90",
        "p5_core_jaccard": ">=0.75",
        "minimum_core_jaccard": ">=0.60",
        "median_stable_weight_spearman": ">=0.98",
        "median_per_frame_rmsd_abs_diff_A": "<=0.25",
        "p95_of_case_p95_rmsd_abs_diff_A": "<=1.0",
        "median_stable_rmsf_spearman": ">=0.98",
        "p5_stable_rmsf_spearman": ">=0.95",
        "median_rmsf_mae_A": "<=0.25",
        "unresolved_interval_rate": "<=0.05",
        "uncertainty_output_calibrated": "required",
        "atlas_10k_speedup": ">=3.0",
        "control_50k_speedup": ">=5.0",
        "control_100k_speedup": ">=10.0",
    }
    for gate, values in (("balanced", balanced), ("fast", fast)):
        for metric, threshold in values.items():
            gate_rows.append({"gate": gate, "metric": metric, "threshold": threshold})
    _write_tsv(baseline / "quality_speed_gates.tsv", ["gate", "metric", "threshold"], gate_rows)

    catastrophic = [
        "non-finite transform or metric",
        "invalid/reflected rotation not repaired",
        "core Jaccard < 0.50 versus frozen throughput",
        "median aligned-coordinate audit difference > 2.0 Angstrom",
        "P95 per-frame RMSD difference > 2.0 Angstrom",
        "stable-core RMSF Spearman < 0.90",
        "mapping/frame-count failure",
    ]
    (baseline / "catastrophic_definition.md").write_text(
        "# Catastrophic Mismatch Definition\n\n"
        + "\n".join(f"- {item}" for item in catastrophic)
        + "\n",
        encoding="utf-8",
    )
    registry = {
        "study": "throughput_v2_sparse_keyframe",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_contract": "current_throughput_compact",
        "contracts": contracts,
        "gates": {"balanced": balanced, "fast": fast},
        "catastrophic_definition": catastrophic,
        "fresh_results_opened_at_registration": False,
        "modification_policy": "immutable after registration",
    }
    registry_path = baseline / "gate_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (baseline / "gate_registry.sha256").write_text(
        f"{_sha256(registry_path)}  gate_registry.json\n",
        encoding="utf-8",
    )


def main() -> None:
    for directory in PHASE_DIRS:
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)
    _freeze_sources()
    _freeze_references()
    _environment()
    _write_scope()
    _write_contracts_and_gates()
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
