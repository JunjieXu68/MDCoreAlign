#!/usr/bin/env python3
"""Create the read-only 2026-08-07 database benchmark review bundle."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/database_scale_benchmark_v2"
B3 = RESULT_ROOT / "b3_full_mdcath"
B2 = RESULT_ROOT / "b2_contracts"
B0 = RESULT_ROOT / "b0_freeze"
MDL = RESULT_ROOT / "mdlovofit_comparison"
BUNDLE_NAME = "benchmark_review_bundle_20260807"
TARGET = RESULT_ROOT / f"{BUNDLE_NAME}.tar.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, root: Path, destination: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = root / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(materialized)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def bool_value(value: object) -> bool:
    return str(value).lower() == "true"


def quantile_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered)

    def quantile(fraction: float) -> float:
        position = (count - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, count - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "median": statistics.median(ordered),
        "q1": quantile(0.25),
        "q3": quantile(0.75),
    }


def summarize_b3() -> dict[str, Any]:
    status = load_json(B3 / "b3_status.json")
    latest = load_json(B3 / "latest_full_run_status.json")
    cpu_times: list[float] = []
    gpu_times: list[float] = []
    exact_rows: list[dict[str, str]] = []
    auto_cpu = auto_gpu = 0
    with (B3 / "b3_all_trajectory_results.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] != "success":
                continue
            cpu_times.append(float(row["cpu_throughput_loaded_seconds"]))
            gpu_times.append(float(row["gpu_throughput_loaded_seconds"]))
            auto_cpu += row["backend_auto_selected"] == "cpu"
            auto_gpu += row["backend_auto_selected"] == "gpu"
            if row.get("exact_completed", "").lower() == "true":
                exact_rows.append(row)

    def exact_median(field: str) -> float:
        return statistics.median(float(row[field]) for row in exact_rows)

    cpu_catastrophic = sum(
        bool_value(row["cpu_exact_catastrophic_quality_failure"])
        for row in exact_rows
    )
    gpu_catastrophic = sum(
        bool_value(row["gpu_exact_catastrophic_quality_failure"])
        for row in exact_rows
    )
    catastrophic_union = sum(
        bool_value(row["cpu_exact_catastrophic_quality_failure"])
        or bool_value(row["gpu_exact_catastrophic_quality_failure"])
        for row in exact_rows
    )
    cpu_summary = quantile_summary(cpu_times)
    gpu_summary = quantile_summary(gpu_times)

    initial_b3_wall_seconds = 563.0
    resumed_wall_seconds = float(latest["wall_seconds"])
    reconstructed_active_wall_seconds = initial_b3_wall_seconds + resumed_wall_seconds
    completed_available = int(status["success"]) + int(status["failed"])
    return {
        **status,
        "registered_domains": 5_398,
        "available_domains": 5_397,
        "registered_trajectories": int(status["manifest_rows"]),
        "available_completed_trajectories": completed_available,
        "warnings": int(status["moderate_confidence"])
        + int(status["low_confidence"]),
        "backend_auto_cpu": auto_cpu,
        "backend_auto_gpu": auto_gpu,
        "cpu_runtime": cpu_summary,
        "gpu_runtime": gpu_summary,
        "gpu_speedup_ratio_of_medians": cpu_summary["median"]
        / gpu_summary["median"],
        "exact_subset_rows": len(exact_rows),
        "exact_quality": {
            "cpu_core_jaccard_median": exact_median(
                "cpu_exact_core_jaccard_vs_exact"
            ),
            "gpu_core_jaccard_median": exact_median(
                "gpu_exact_core_jaccard_vs_exact"
            ),
            "cpu_weight_spearman_median": exact_median(
                "cpu_exact_weight_spearman_vs_exact"
            ),
            "gpu_weight_spearman_median": exact_median(
                "gpu_exact_weight_spearman_vs_exact"
            ),
            "cpu_rmsf_spearman_median": exact_median(
                "cpu_exact_rmsf_spearman_vs_exact"
            ),
            "gpu_rmsf_spearman_median": exact_median(
                "gpu_exact_rmsf_spearman_vs_exact"
            ),
            "cpu_rmsf_mae_A_median": exact_median(
                "cpu_exact_rmsf_mae_vs_exact_A"
            ),
            "gpu_rmsf_mae_A_median": exact_median(
                "gpu_exact_rmsf_mae_vs_exact_A"
            ),
            "cpu_flexible_label_agreement_median": exact_median(
                "cpu_exact_flexible_label_agreement_vs_exact"
            ),
            "gpu_flexible_label_agreement_median": exact_median(
                "gpu_exact_flexible_label_agreement_vs_exact"
            ),
            "cpu_catastrophic": cpu_catastrophic,
            "gpu_catastrophic": gpu_catastrophic,
            "catastrophic_union": catastrophic_union,
        },
        "initial_service_wall_seconds_from_journal": initial_b3_wall_seconds,
        "resumed_full_wall_seconds": resumed_wall_seconds,
        "reconstructed_active_wall_seconds": reconstructed_active_wall_seconds,
        "reconstructed_trajectories_per_hour": completed_available
        / reconstructed_active_wall_seconds
        * 3600.0,
        "resumed_new_trajectories_per_hour": (
            int(latest["new_successes"]) + int(latest["new_failures"])
        )
        / resumed_wall_seconds
        * 3600.0,
    }


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)} h {int(minutes)} min {seconds:.1f} s"


def b3_report(summary: dict[str, Any]) -> str:
    quality = summary["exact_quality"]
    return "\n".join(
        [
            "# B3 full-mdCATH benchmark report",
            "",
            "This report was generated post hoc from frozen B3 result tables. No "
            "benchmark, algorithm, threshold, or trajectory was rerun.",
            "",
            "## Completion",
            "",
            "B3 completed with warnings under the prospectively registered stop rule.",
            "",
            f"- Registered trajectories: `{summary['registered_trajectories']:,}`",
            f"- Available and completed trajectories: `{summary['available_completed_trajectories']:,}`",
            f"- Successful trajectories: `{summary['success']:,}`",
            f"- Failed trajectories: `{summary['failed']:,}`",
            f"- Explicitly unavailable inputs: `{summary['explicitly_unavailable']:,}`",
            f"- Registered domains: `{summary['registered_domains']:,}`",
            f"- Available domains: `{summary['available_domains']:,}`",
            f"- High-confidence outputs: `{summary['high_confidence']:,}`",
            f"- Moderate-confidence outputs: `{summary['moderate_confidence']:,}`",
            f"- Low-confidence outputs: `{summary['low_confidence']:,}`",
            f"- Successful outputs carrying warnings: `{summary['warnings']:,}`",
            "",
            "Non-catastrophic warnings were retained rather than used as a reason to "
            "stop the deployment. No systematic unit, mapping, or numerical failure "
            "triggered the registered stop condition.",
            "",
            "## Backend execution",
            "",
            f"- CPU throughput outputs: `{summary['success']:,}`",
            f"- GPU throughput outputs: `{summary['success']:,}`",
            f"- Backend-auto selected CPU: `{summary['backend_auto_cpu']:,}`",
            f"- Backend-auto selected GPU: `{summary['backend_auto_gpu']:,}`",
            f"- CPU W5 outputs: `{summary['success']:,}`",
            f"- GPU W5 outputs: `{summary['success']:,}`",
            "",
            "GPU H2D, compute, D2H, loaded runtime, peak VRAM and transform hashes "
            "are populated. GPU event/device synchronization is implemented in the "
            "included `src/mdcorealign/gpu_backend.py`.",
            "",
            "## Exact-quality subset",
            "",
            f"The prospectively locked exact subset planned and completed `{summary['exact_subset_rows']:,}` trajectories.",
            "",
            "| Metric | CPU median | GPU median |",
            "|---|---:|---:|",
            f"| Core Jaccard | {quality['cpu_core_jaccard_median']:.6f} | {quality['gpu_core_jaccard_median']:.6f} |",
            f"| Weight Spearman | {quality['cpu_weight_spearman_median']:.6f} | {quality['gpu_weight_spearman_median']:.6f} |",
            f"| RMSF Spearman | {quality['cpu_rmsf_spearman_median']:.6f} | {quality['gpu_rmsf_spearman_median']:.6f} |",
            f"| RMSF MAE (A) | {quality['cpu_rmsf_mae_A_median']:.6f} | {quality['gpu_rmsf_mae_A_median']:.6f} |",
            f"| Flexible-label agreement | {quality['cpu_flexible_label_agreement_median']:.6f} | {quality['gpu_flexible_label_agreement_median']:.6f} |",
            "",
            f"The final subset contained `{quality['catastrophic_union']}` trajectories "
            "flagged catastrophic by either implementation. CPU and GPU each flagged "
            f"`{quality['cpu_catastrophic']}` trajectories.",
            "",
            "## Performance",
            "",
            f"- Median CPU loaded runtime: `{summary['cpu_runtime']['median']:.9f} s`",
            f"- Median GPU loaded runtime: `{summary['gpu_runtime']['median']:.9f} s`",
            f"- GPU/CPU speedup, ratio of medians: `{summary['gpu_speedup_ratio_of_medians']:.6f}x`",
            f"- Peak RSS: `{summary['peak_RSS_mb']:.3f} MiB`",
            f"- Peak VRAM: `{int(summary['peak_VRAM_bytes']):,} bytes`",
            f"- Initial B3 service wall time from systemd journal: `{summary['initial_service_wall_seconds_from_journal']:.1f} s`",
            f"- Resumed full-run wall time: `{summary['resumed_full_wall_seconds']:.6f} s`",
            f"- Reconstructed active wall time: `{summary['reconstructed_active_wall_seconds']:.6f} s` ({format_duration(summary['reconstructed_active_wall_seconds'])})",
            f"- Reconstructed active throughput: `{summary['reconstructed_trajectories_per_hour']:.3f} trajectories/hour`",
            "",
            "The reconstructed active wall time sums the initial service interval "
            "(`13:19:24` to `13:28:47` AEST) and the recorded resumed runner wall. "
            "It excludes the intentional pause while MDLovoFit ran. The original "
            "B3 status did not contain one consolidated all-session wall field.",
            "",
            "## Limitations",
            "",
            "- Exact sequential analysis was run on a locked 10,000-trajectory subset, not all trajectories.",
            "- The pilot did not pass every prospective approximation-quality gate; the full run continued because no registered systematic integrity stop condition occurred.",
            "- W5 is a local-deformation contract and is not an MDLovoFit-equivalent alignment workflow.",
            "- Backend-auto uses the prospectively registered workload threshold; it is not a learned scheduler.",
            "- This report is a review artifact and does not alter the frozen B3 evidence.",
        ]
    )


def checkpoint_timestamps() -> list[dict[str, Any]]:
    paths = list((B3 / "domain_checkpoints").glob("*.json"))
    ordered = sorted(paths, key=lambda path: (path.stat().st_mtime, path.name))
    times = [path.stat().st_mtime for path in ordered]
    gap_index = max(
        range(1, len(times)), key=lambda index: times[index] - times[index - 1]
    )
    rows = []
    for index, path in enumerate(ordered):
        stamp = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        rows.append(
            {
                "checkpoint_index": index + 1,
                "domain_id": path.stem,
                "checkpoint_path": str(path.relative_to(ROOT)),
                "mtime_local": stamp.isoformat(),
                "execution_block": "initial" if index < gap_index else "resumed",
            }
        )
    return rows


def stage_table() -> list[dict[str, str]]:
    def mtime(path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()

    def bounds(root: Path) -> tuple[str, str]:
        paths = list(root.glob("*.json"))
        timestamps = [path.stat().st_mtime for path in paths]
        return (
            datetime.fromtimestamp(min(timestamps)).astimezone().isoformat(),
            datetime.fromtimestamp(max(timestamps)).astimezone().isoformat(),
        )

    all_start, all_end = bounds(MDL / "all_domain_checkpoints")
    deployment_start, deployment_end = bounds(MDL / "deployment_domain_checkpoints")
    internal_start, internal_end = bounds(MDL / "real_long_internal_checkpoints")
    external_start, external_end = bounds(MDL / "real_long_prefix_checkpoints")
    final_pilot_start = mtime(MDL / "mdlovofit_full_database_pilot.tsv")
    final_pilot_end = mtime(MDL / "mdlovofit_deployment_execution_decision.json")
    all_finalize_end = mtime(MDL / "mdlovofit_all_domain_corrected.tsv")
    deployment_finalize_end = mtime(MDL / "mdlovofit_deployment_status.json")
    b3_run_end = mtime(B3 / "latest_full_run_status.json")
    b3_finalize_end = mtime(B3 / "b3_status.json")
    internal_finalize_end = mtime(MDL / "internal_real_long_status.json")
    external_finalize_end = mtime(MDL / "mdlovofit_real_long_status.json")
    report_end = mtime(MDL / "mdlovofit_addendum_completion_status.json")
    return [
        {"stage": "finalize_pilot", "status": "success", "start": final_pilot_start, "end": final_pilot_end, "timestamp_basis": "artifact mtimes"},
        {"stage": "all_domain", "status": "success", "start": all_start, "end": all_end, "timestamp_basis": "checkpoint bounds"},
        {"stage": "finalize_all_domain", "status": "success", "start": all_end, "end": all_finalize_end, "timestamp_basis": "serial boundary and artifact mtime"},
        {"stage": "deployment", "status": "success", "start": deployment_start, "end": deployment_end, "timestamp_basis": "checkpoint bounds"},
        {"stage": "finalize_deployment", "status": "success", "start": deployment_end, "end": deployment_finalize_end, "timestamp_basis": "serial boundary and artifact mtime"},
        {"stage": "resume_internal_B3", "status": "success", "start": deployment_finalize_end, "end": b3_run_end, "timestamp_basis": "serial boundary and recorded run-status mtime"},
        {"stage": "finalize_internal_B3", "status": "success", "start": b3_run_end, "end": b3_finalize_end, "timestamp_basis": "serial boundary and artifact mtime"},
        {"stage": "real_long_internal", "status": "success", "start": internal_start, "end": internal_end, "timestamp_basis": "checkpoint bounds"},
        {"stage": "finalize_real_long_internal", "status": "success", "start": internal_end, "end": internal_finalize_end, "timestamp_basis": "serial boundary and artifact mtime"},
        {"stage": "real_long", "status": "success", "start": external_start, "end": external_end, "timestamp_basis": "checkpoint bounds"},
        {"stage": "finalize_real_long", "status": "success", "start": external_end, "end": external_finalize_end, "timestamp_basis": "serial boundary and artifact mtime"},
        {"stage": "make_comparison_report", "status": "success", "start": external_finalize_end, "end": report_end, "timestamp_basis": "serial boundary and artifact mtime"},
    ]


EXPLICIT_DESCRIPTIONS = {
    "trajectory_id": "Unique trajectory identifier.",
    "domain_id": "CATH domain identifier.",
    "system_id": "System or domain identifier.",
    "system": "Human-readable benchmark system name.",
    "status": "Execution status for the row.",
    "method": "Method or backend represented by the row.",
    "contract": "Registered output/timing contract.",
    "confidence_category": "Prospectively registered high, moderate, low, or failure confidence category.",
    "warning_reason": "Semicolon-separated reason for a non-high-confidence result.",
    "failure_class": "Categorized failure type.",
    "failure_reason": "Recorded failure message.",
    "backend_auto_selected": "Backend selected by the registered auto policy.",
    "backend_auto_reason": "Registered rule that caused the auto-backend decision.",
    "temperature_K": "Simulation temperature in kelvin.",
    "replicate": "Replicate index or label.",
    "n_frames": "Number of trajectory frames analyzed.",
    "n_residues": "Number of selected CA residues.",
    "n_chains": "Number of selected chains.",
    "coordinate_path": "Dataset-internal coordinate path.",
    "source_file": "Repository-relative or recorded source dataset path.",
    "canonical_coordinate_unit": "Canonical coordinate unit used by the benchmark.",
    "unit_integrity": "Coordinate-unit integrity check result.",
    "mapping_integrity": "Residue/coordinate mapping integrity check result.",
    "selection": "Atom selection used for the benchmark.",
    "prefix_frames": "Number of leading frames in the matched prefix.",
    "prefix_label": "Stable label for a frame-prefix setting.",
    "is_warmup": "Whether the row is an unreported timing warmup.",
    "repeat_index": "Measured repeat index; negative values denote warmup where documented.",
}


def describe_column(column: str) -> tuple[str, str]:
    if column in EXPLICIT_DESCRIPTIONS:
        return EXPLICIT_DESCRIPTIONS[column], ""
    lower = column.lower()
    if "sha256" in lower:
        return "SHA-256 integrity digest for the named object.", "hex digest"
    if lower.endswith("_seconds") or "runtime_sec" in lower:
        return "Recorded wall-clock duration for the named stage or contract.", "seconds"
    if lower.endswith("_bytes"):
        return "Recorded size or memory usage for the named object.", "bytes"
    if "rss_mb" in lower:
        return "Resident-set memory measurement.", "MiB"
    if "rmsd" in lower:
        return "RMSD value or exact-relative RMSD comparison identified by the column name.", "angstrom unless noted"
    if "rmsf" in lower:
        if "spearman" in lower:
            return "Spearman rank correlation for the named RMSF comparison.", "dimensionless"
        if "mae" in lower:
            return "Mean absolute error for the named RMSF comparison.", "angstrom"
        return "RMSF value or RMSF-derived comparison identified by the column name.", "angstrom unless noted"
    if "spearman" in lower:
        return "Spearman rank correlation for the named comparison.", "dimensionless"
    if "jaccard" in lower:
        return "Jaccard overlap for the named label or core comparison.", "0 to 1"
    if "agreement" in lower:
        return "Agreement fraction for the named labels or implementations.", "0 to 1"
    if "fraction" in lower:
        return "Fraction identified by the column name.", "0 to 1"
    if "weight" in lower:
        return "Stable-core weight value or weight-derived diagnostic identified by the column name.", "dimensionless"
    if "core" in lower:
        return "Stable-core quantity or diagnostic identified by the column name.", "dimensionless unless noted"
    if "w5" in lower or "local_deformation" in lower or "local_rigidity" in lower:
        return "W5 alignment-free local-deformation diagnostic identified by the column name.", "dimensionless"
    if "transform" in lower:
        return "Rigid-transform output or integrity identifier identified by the column name.", "varies"
    if "gpu" in lower or "cpu" in lower:
        return "Backend-specific benchmark field identified by the column name.", "varies"
    if lower.startswith("cath_"):
        return "CATH classification field identified by the column name.", "categorical"
    if lower.endswith("_path") or lower.endswith("_file"):
        return "Recorded path for the named input or output.", "path"
    if lower.startswith("n_") or "count" in lower or lower.endswith("_rows"):
        return "Count identified by the column name.", "count"
    if lower.startswith("mean_") or lower.startswith("median_") or lower.startswith("p95_"):
        return "Summary statistic identified by the column name.", "see metric name"
    if lower.startswith("is_") or lower.startswith("has_") or lower.endswith("_used") or lower.endswith("_completed") or lower.endswith("_requested"):
        return "Boolean flag identified by the column name.", "boolean"
    return (
        "Recorded benchmark metadata field; semantics follow the included source contract and the literal column name.",
        "varies",
    )


def data_dictionary(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"))
        relative = str(path.relative_to(root))
        for index, column in enumerate(header, start=1):
            description, units = describe_column(column)
            rows.append(
                {
                    "file": relative,
                    "column_index": index,
                    "column_name": column,
                    "description": description,
                    "units_or_range": units,
                }
            )
    return rows


def journal_text() -> str:
    command = [
        "journalctl",
        "--user",
        "-u",
        "mdcorealign-mdlovofit-postpilot-v2.service",
        "-u",
        "mdcorealign-mdlovofit-pilot-v2.service",
        "-u",
        "mdcorealign-b3-v2.service",
        "--since",
        "2026-08-03 12:30:00",
        "--until",
        "2026-08-07 09:00:00",
        "-o",
        "short-iso",
        "--no-pager",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return (
        "$ "
        + " ".join(command)
        + "\n\n"
        + completed.stdout
        + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")
    )


def readme(
    summary: dict[str, Any],
    copied: list[tuple[Path, Path]],
    bundle_root: Path,
) -> str:
    environment = load_json(B0 / "environment.json")
    key_hashes = {
        str(source.relative_to(ROOT)): sha256(source)
        for source, _ in copied
        if source.name
        in {
            "b3_all_trajectory_results.tsv",
            "b3_all_trajectory_manifest_locked.tsv",
            "b3_exact_quality_subset.tsv",
            "mdlovofit_all_domain_corrected_matched.tsv",
            "mdlovofit_matched_contract_summary.tsv",
        }
    }
    inventory_lines = [
        f"- `{destination.relative_to(bundle_root)}` <- `{source.relative_to(ROOT)}`"
        for source, destination in copied
    ]
    hash_lines = [f"- `{path}`: `{digest}`" for path, digest in sorted(key_hashes.items())]
    return "\n".join(
        [
            "# MDCoreAlign database-scale benchmark review bundle",
            "",
            "This is a read-only review package assembled on 2026-08-07 from completed, frozen benchmark evidence. No benchmark or scientific algorithm was rerun.",
            "",
            "## Canonical environment",
            "",
            f"- Python: `{environment['python_executable']}` (`{environment['python_version']}`)",
            f"- NumPy: `{environment['numpy_version']}`",
            f"- SciPy: `{environment['scipy_version']}`",
            f"- CuPy: `{environment['cupy_version']}`",
            f"- CUDA runtime: `{environment['cuda_runtime_version']}`",
            f"- GPU: `{environment['gpu']['stdout'].strip()}`",
            "- CPU: `Intel Xeon W-2195, 18 cores / 36 threads`",
            "- Thread environment: `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`",
            "- Canonical coordinates: CA-only, angstrom, validated mapping and unit metadata.",
            "- Git HEAD was invalid/uninitialized in the frozen environment; no Git metadata was changed.",
            "",
            "The complete captured environment is in `configuration/environment.json`.",
            "",
            "## Benchmark contracts",
            "",
            "- Exact global analysis: sequential_lean stable-core weights followed by exact all-frame fitting.",
            "- Throughput global analysis: CPU and synchronized GPU throughput, transforms, RMSD, RMSF and stable/flexible summaries.",
            "- Backend auto: CPU below 150,000 residue-frames and GPU at or above that registered threshold.",
            "- W5: alignment-free local-deformation diagnostic, not an MDLovoFit-equivalent alignment workflow.",
            "- MDLovoFit: principal external comparator; mandatory multi-model PDB preparation and aligned-PDB writing are reported separately where possible.",
            "- Shared HDF5/cache decoding is upstream input preparation and is not charged repeatedly to one method.",
            "",
            "## Exact commands",
            "",
            "```bash",
            "export PYTHONPATH=src:.",
            "export MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2",
            "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/database_scale_benchmark_v2/run_b3_full_mdcath.py pilot --input-workers 2",
            "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/database_scale_benchmark_v2/run_b3_full_mdcath.py finalize-pilot",
            "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/database_scale_benchmark_v2/run_b3_full_mdcath.py full --input-workers 2",
            "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/database_scale_benchmark_v2/run_b3_full_mdcath.py finalize-full",
            "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/database_scale_benchmark_v2/run_mdlovofit_addendum.py pilot",
            "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/database_scale_benchmark_v2/orchestrate_mdlovofit_addendum.py",
            "```",
            "",
            "The observed transient systemd commands and timestamps are preserved in `provenance/systemd_journal_timing.txt`.",
            "",
            "## Key status",
            "",
            f"- B3 registered `{summary['registered_trajectories']:,}` rows; `{summary['available_completed_trajectories']:,}` available trajectories completed.",
            f"- B3 successes/failures/unavailable: `{summary['success']:,}` / `{summary['failed']:,}` / `{summary['explicitly_unavailable']:,}`.",
            f"- Exact-quality subset: `{summary['exact_subset_rows']:,}` completed.",
            f"- Median CPU/GPU loaded runtime: `{summary['cpu_runtime']['median']:.9f}` / `{summary['gpu_runtime']['median']:.9f}` seconds.",
            f"- GPU speedup as ratio of medians: `{summary['gpu_speedup_ratio_of_medians']:.6f}x`.",
            f"- Reconstructed B3 active wall: `{summary['reconstructed_active_wall_seconds']:.3f}` seconds.",
            "",
            "## Known limitations",
            "",
            "- The original B3 output did not include `b3_report.md`; `results/b3/b3_report.md` is a post-hoc lightweight summary derived only from frozen tables.",
            "- The original B3 status did not contain one consolidated all-session wall field. This bundle reconstructs active wall time from the initial systemd interval and resumed runner status, excluding the intentional pause.",
            "- Exact analysis covers a locked 10,000-trajectory subset rather than all 134,925 available trajectories.",
            "- Many successful throughput rows carry prospective moderate/low-confidence warnings; warnings were not hidden or treated as automatic failures.",
            "- THESEUS entries are validated historical evidence where identified and were not necessarily rerun in the addendum.",
            "- Real-long 10,000-frame prefixes were unavailable locally and were not synthesized.",
            "- Raw trajectories, HDF5, coordinates, NPZ/NPY caches, aligned PDB files, Python caches and secrets are intentionally excluded.",
            "",
            "## Key source hashes",
            "",
            *hash_lines,
            "",
            "All bundled-file hashes are listed in `SHA256SUMS`; `bundle_inventory.tsv` maps bundle paths to source paths and sizes.",
            "",
            "## File inventory",
            "",
            *inventory_lines,
        ]
    )


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {TARGET}")
    summary = summarize_b3()
    with tempfile.TemporaryDirectory(prefix=f"{BUNDLE_NAME}_", dir=RESULT_ROOT) as temporary:
        staging = Path(temporary) / BUNDLE_NAME
        staging.mkdir()
        copied: list[tuple[Path, Path]] = []

        copy_specs = [
            (B3 / "b3_all_trajectory_results.tsv", "results/b3/b3_all_trajectory_results.tsv"),
            (B3 / "b3_backend_auto_decisions.tsv", "results/b3/b3_backend_auto_decisions.tsv"),
            (B3 / "b3_exact_quality_subset.tsv", "results/b3/b3_exact_quality_subset.tsv"),
            (B3 / "b3_database_throughput_summary.tsv", "results/b3/b3_database_throughput_summary.tsv"),
            (B3 / "b3_failure_warning_atlas.tsv", "results/b3/b3_failure_warning_atlas.tsv"),
            (B2 / "b3_all_trajectory_manifest_locked.tsv", "results/b3/b3_all_trajectory_manifest.tsv"),
            (B3 / "b3_exact_quality_subset_locked.tsv", "results/b3/b3_exact_quality_subset_selection_manifest.tsv"),
            (B3 / "b3_status.json", "results/b3/b3_status.json"),
            (B3 / "latest_full_run_status.json", "results/b3/latest_full_run_status.json"),
            (B3 / "exact_execution_decision.json", "results/b3/exact_execution_decision.json"),
            (B3 / "b3_pilot_quality_gate_audit.json", "results/b3/b3_pilot_quality_gate_audit.json"),
            (MDL / "mdlovofit_all_domain_corrected.tsv", "results/mdlovofit/mdlovofit_all_domain_corrected.tsv"),
            (MDL / "mdlovofit_all_domain_corrected_matched.tsv", "results/mdlovofit/mdlovofit_all_domain_corrected_matched.tsv"),
            (MDL / "mdlovofit_deployment_status.json", "results/mdlovofit/mdlovofit_deployment_status.json"),
            (MDL / "mdlovofit_deployment_execution_decision.json", "results/mdlovofit/mdlovofit_deployment_execution_decision.json"),
            (MDL / "mdlovofit_matched_contract_summary.tsv", "results/mdlovofit/mdlovofit_matched_contract_summary.tsv"),
            (MDL / "mdlovofit_failures_timeouts.tsv", "results/mdlovofit/mdlovofit_failures_timeouts.tsv"),
            (MDL / "mdlovofit_real_long_prefixes.tsv", "results/mdlovofit/mdlovofit_real_long_prefixes.tsv"),
            (MDL / "mdlovofit_real_long_matched_contracts.tsv", "results/mdlovofit/mdlovofit_real_long_matched_contracts.tsv"),
            (MDL / "mdlovofit_comparison_report.md", "results/mdlovofit/mdlovofit_comparison_report.md"),
            (MDL / "internal_real_long_prefixes.tsv", "results/real_long/internal_real_long_prefixes.tsv"),
            (MDL / "internal_real_long_prefixes_raw_repeats.tsv", "results/real_long/internal_real_long_prefixes_raw_repeats.tsv"),
            (MDL / "mdlovofit_real_long_prefixes.tsv", "results/real_long/mdlovofit_real_long_prefixes.tsv"),
            (MDL / "mdlovofit_real_long_prefixes_raw_repeats.tsv", "results/real_long/mdlovofit_real_long_prefixes_raw_repeats.tsv"),
            (MDL / "mdlovofit_real_long_matched_contracts.tsv", "results/real_long/mdlovofit_real_long_matched_contracts.tsv"),
            (MDL / "manifests/mdlovofit_real_long_prefix_manifest_locked.tsv", "results/real_long/real_long_prefix_manifest_locked.tsv"),
            (MDL / "manifests/mdlovofit_real_long_prefix_registration.json", "results/real_long/real_long_prefix_registration.json"),
            (B0 / "environment.json", "configuration/environment.json"),
            (B2 / "backend_auto_policy.json", "configuration/backend_auto_policy.json"),
            (B2 / "prospective_quality_gates.json", "configuration/prospective_quality_gates.json"),
            (B2 / "registered_contracts.json", "configuration/registered_contracts.json"),
            (B2 / "b2_contract_registration.md", "configuration/b2_contract_registration.md"),
            (MDL / "manifests/mdlovofit_registered_policy.json", "configuration/mdlovofit_registered_policy.json"),
            (ROOT / "scripts/database_scale_benchmark_v2/run_b3_full_mdcath.py", "source/scripts/run_b3_full_mdcath.py"),
            (ROOT / "scripts/database_scale_benchmark_v2/register_b2_contracts.py", "source/scripts/register_b2_contracts.py"),
            (ROOT / "scripts/database_scale_benchmark_v2/run_b1_tail_adjudication.py", "source/scripts/run_b1_tail_adjudication.py"),
            (ROOT / "scripts/database_scale_benchmark_v2/audit_b3_pilot_gates.py", "source/scripts/audit_b3_pilot_gates.py"),
            (ROOT / "scripts/database_scale_benchmark_v2/run_mdlovofit_addendum.py", "source/scripts/run_mdlovofit_addendum.py"),
            (ROOT / "scripts/database_scale_benchmark_v2/run_real_long_internal_prefixes.py", "source/scripts/run_real_long_internal_prefixes.py"),
            (ROOT / "scripts/database_scale_benchmark_v2/run_real_long_mdlovofit_addendum.py", "source/scripts/run_real_long_mdlovofit_addendum.py"),
            (ROOT / "scripts/database_scale_benchmark_v2/orchestrate_mdlovofit_addendum.py", "source/scripts/orchestrate_mdlovofit_addendum.py"),
            (ROOT / "scripts/database_scale_gpu_v1/common.py", "source/scripts/database_scale_gpu_common.py"),
            (ROOT / "src/mdcorealign/gpu_backend.py", "source/src/mdcorealign/gpu_backend.py"),
            (ROOT / "src/mdcorealign/experimental.py", "source/src/mdcorealign/experimental.py"),
            (ROOT / "src/mdcorealign/contracts.py", "source/src/mdcorealign/contracts.py"),
            (ROOT / "src/mdsuperpose/sequential_weights.py", "source/src/mdsuperpose/sequential_weights.py"),
            (ROOT / "src/mdsuperpose/window_kinematics.py", "source/src/mdsuperpose/window_kinematics.py"),
            (ROOT / "src/mdsuperpose/window_refine.py", "source/src/mdsuperpose/window_refine.py"),
            (ROOT / "docs/CODEX_MDCoreAlign_large_scale_benchmark_no_hdf5_optimization.md", "source/plans/CODEX_MDCoreAlign_large_scale_benchmark_no_hdf5_optimization.md"),
            (ROOT / "docs/CODEX_MDCoreAlign_MDLovoFit_comparison_addendum.md", "source/plans/CODEX_MDCoreAlign_MDLovoFit_comparison_addendum.md"),
            (ROOT / "scripts/database_scale_benchmark_v2/create_benchmark_review_bundle.py", "source/scripts/create_benchmark_review_bundle.py"),
        ]
        for source, destination in copy_specs:
            copied.append((source, copy_file(source, staging, destination)))

        write_text(staging / "results/b3/b3_report.md", b3_report(summary))
        write_tsv(staging / "provenance/stage_execution_table.tsv", stage_table())
        write_tsv(
            staging / "provenance/b3_checkpoint_timestamps.tsv",
            checkpoint_timestamps(),
        )
        write_text(
            staging / "provenance/systemd_journal_timing.txt", journal_text()
        )
        copy_file(
            MDL / "orchestrator_status.json",
            staging,
            "provenance/orchestrator_status.json",
        )
        copy_file(
            MDL / "mdlovofit_orchestrator.log",
            staging,
            "provenance/mdlovofit_orchestrator.log",
        )

        major_tables = [
            staging / "results/b3/b3_all_trajectory_results.tsv",
            staging / "results/b3/b3_backend_auto_decisions.tsv",
            staging / "results/b3/b3_exact_quality_subset.tsv",
            staging / "results/b3/b3_database_throughput_summary.tsv",
            staging / "results/b3/b3_failure_warning_atlas.tsv",
            staging / "results/b3/b3_all_trajectory_manifest.tsv",
            staging / "results/b3/b3_exact_quality_subset_selection_manifest.tsv",
            staging / "results/mdlovofit/mdlovofit_all_domain_corrected.tsv",
            staging / "results/mdlovofit/mdlovofit_all_domain_corrected_matched.tsv",
            staging / "results/mdlovofit/mdlovofit_matched_contract_summary.tsv",
            staging / "results/mdlovofit/mdlovofit_failures_timeouts.tsv",
            staging / "results/mdlovofit/mdlovofit_real_long_prefixes.tsv",
            staging / "results/mdlovofit/mdlovofit_real_long_matched_contracts.tsv",
            staging / "results/real_long/internal_real_long_prefixes.tsv",
            staging / "results/real_long/internal_real_long_prefixes_raw_repeats.tsv",
            staging / "results/real_long/mdlovofit_real_long_prefixes.tsv",
            staging / "results/real_long/mdlovofit_real_long_prefixes_raw_repeats.tsv",
            staging / "results/real_long/mdlovofit_real_long_matched_contracts.tsv",
            staging / "results/real_long/real_long_prefix_manifest_locked.tsv",
        ]
        write_tsv(
            staging / "documentation/data_dictionary.tsv",
            data_dictionary(staging, major_tables),
        )
        write_text(staging / "00_README_FIRST.md", readme(summary, copied, staging))

        inventory_rows = []
        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path.name in {"SHA256SUMS", "bundle_inventory.tsv"}:
                continue
            matching_source = next(
                (
                    source
                    for source, destination in copied
                    if destination == path
                ),
                None,
            )
            inventory_rows.append(
                {
                    "bundle_path": str(path.relative_to(staging)),
                    "source_path": (
                        str(matching_source.relative_to(ROOT))
                        if matching_source is not None
                        else "generated_review_artifact"
                    ),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        write_tsv(staging / "bundle_inventory.tsv", inventory_rows)
        checksum_paths = [
            path
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        write_text(
            staging / "SHA256SUMS",
            "\n".join(
                f"{sha256(path)}  {path.relative_to(staging)}"
                for path in checksum_paths
            ),
        )

        forbidden_suffixes = {".h5", ".hdf5", ".npz", ".npy", ".pdb"}
        forbidden_names = {"__pycache__", ".env"}
        for path in staging.rglob("*"):
            if path.suffix.lower() in forbidden_suffixes or any(
                part in forbidden_names for part in path.parts
            ):
                raise RuntimeError(f"forbidden bundle content: {path}")
        secret_scan = subprocess.run(
            [
                "rg",
                "-i",
                "-n",
                "hf_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|password\\s*[:=]",
                str(staging),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if secret_scan.returncode == 0:
            raise RuntimeError("secret-like content detected:\n" + secret_scan.stdout)
        if secret_scan.returncode not in {0, 1}:
            raise RuntimeError("secret scan failed: " + secret_scan.stderr)

        with tarfile.open(TARGET, "w:gz", compresslevel=6) as archive:
            archive.add(staging, arcname=BUNDLE_NAME, recursive=True)

    with tarfile.open(TARGET, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if not any(name.endswith("/00_README_FIRST.md") for name in names):
            raise RuntimeError("bundle validation failed: README missing")
        if not any(name.endswith("/results/b3/b3_report.md") for name in names):
            raise RuntimeError("bundle validation failed: B3 report missing")
        if any(
            Path(name).suffix.lower() in {".h5", ".hdf5", ".npz", ".npy", ".pdb"}
            or "__pycache__" in Path(name).parts
            for name in names
        ):
            raise RuntimeError("bundle validation failed: forbidden member")
    print(
        json.dumps(
            {
                "bundle": str(TARGET.relative_to(ROOT)),
                "size_bytes": TARGET.stat().st_size,
                "sha256": sha256(TARGET),
                "member_count": len(members),
                "status": "complete",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
