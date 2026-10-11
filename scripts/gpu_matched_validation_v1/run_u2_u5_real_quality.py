#!/usr/bin/env python3
"""Benchmark the expanded real panel and evaluate locked quality criteria."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment
from mdcorealign.gpu_backend import (
    PersistentGPUWorker,
    scan_w5_gpu,
)
from mdsuperpose.window_kinematics import scan_local_kinematics
from scripts.gpu_matched_validation_v1.common import (
    RESULTS,
    load_coords,
    load_panel,
    quality_against_exact,
    read_tsv,
    repeat_call,
    spearman,
    timing_summary,
    write_json,
    write_tsv,
)


OUTPUT = RESULTS / "u2_real_panel"
QUALITY = RESULTS / "u5_quality"
CRITERIA = RESULTS / "u0_freeze" / "locked_quality_criteria.json"


def _timing_row(
    *,
    row: dict[str, str],
    contract: str,
    method: str,
    runtimes: list[float],
    result,
) -> dict[str, object]:
    summary = timing_summary(runtimes)
    n_frames = int(row["n_frames"])
    n_residues = int(row["n_residues"])
    timing = getattr(result, "timing", {})
    metadata = getattr(result, "metadata", {})
    median = summary["median_seconds"]
    return {
        "system": row["system"],
        "trajectory_id": row["trajectory_id"],
        "source_type": row["source_type"],
        "contract": contract,
        "method": method,
        "n_frames": n_frames,
        "n_residues": n_residues,
        **summary,
        "repeat_runtimes_seconds": ";".join(
            f"{value:.9g}" for value in runtimes
        ),
        "frames_per_second": n_frames / median,
        "residue_frames_per_second": n_frames * n_residues / median,
        "transfer_seconds": (
            timing.get("CPU_GPU_transfer_total", 0.0)
        ),
        "gpu_compute_seconds": timing.get("GPU_compute_total", 0.0),
        "peak_vram_bytes": metadata.get("peak_GPU_memory_bytes", 0),
        "aligned_coordinates_materialised": bool(
            getattr(result, "aligned_coords", None) is not None
        ),
        "warmups": 1,
        "measured_repeats": 5,
    }


def _save_result(path: Path, exact, cpu, gpu) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        exact_weights=exact.frozen_weights,
        exact_rmsd=exact.rmsd,
        exact_rmsf=exact.robust_rmsf,
        exact_rotations=exact.rotations,
        exact_translations=exact.translations,
        cpu_weights=cpu.frozen_weights,
        cpu_rmsd=cpu.rmsd,
        cpu_rmsf=cpu.robust_rmsf,
        gpu_weights=gpu.frozen_weights,
        gpu_rmsd=gpu.rmsd,
        gpu_rmsf=gpu.robust_rmsf,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        nargs="*",
        help="Optional system names for an incremental/resume run.",
    )
    args = parser.parse_args()
    selected = set(args.systems or ())
    criteria = json.loads(CRITERIA.read_text())["required"]
    timing_path = OUTPUT / "real_panel_internal_matched_timings.tsv"
    quality_path = QUALITY / "system_level_quality.tsv"
    w5_path = QUALITY / "w5_cpu_gpu_agreement.tsv"
    timing_rows: list[dict[str, object]] = (
        read_tsv(timing_path) if selected and timing_path.exists() else []
    )
    quality_rows: list[dict[str, object]] = (
        read_tsv(quality_path) if selected and quality_path.exists() else []
    )
    w5_rows: list[dict[str, object]] = (
        read_tsv(w5_path) if selected and w5_path.exists() else []
    )
    if selected:
        timing_rows = [
            row for row in timing_rows if row["system"] not in selected
        ]
        quality_rows = [
            row for row in quality_rows if row["system"] not in selected
        ]
        w5_rows = [row for row in w5_rows if row["system"] not in selected]

    for row in load_panel():
        if selected and row["system"] not in selected:
            continue
        coords, chains = load_coords(row)
        n_frames = coords.shape[0]
        print(f"[U2/U5] {row['system']} {coords.shape}", flush=True)

        exact, exact_times = repeat_call(
            lambda: solve_exact_contract(
                coords,
                chain_ids=chains,
                representative_count=min(2000, n_frames),
                chunk_size=1000,
                store_aligned=False,
            )
        )
        timing_rows.append(
            _timing_row(
                row=row,
                contract="A_analysis_metrics_transforms_rmsd_rmsf",
                method="CPU_exact_sequential_lean",
                runtimes=exact_times,
                result=exact,
            )
        )

        cpu, cpu_times = repeat_call(
            lambda: solve_fast_global_alignment(
                coords,
                chain_ids=chains,
                representative_count=min(256, n_frames),
                update_rounds=4,
                chunk_size=1000,
                store_aligned=False,
                compute_confidence=False,
            )
        )
        timing_rows.append(
            _timing_row(
                row=row,
                contract="A_analysis_metrics_transforms_rmsd_rmsf",
                method="CPU_throughput",
                runtimes=cpu_times,
                result=cpu,
            )
        )

        worker = PersistentGPUWorker.create(
            chain_ids=chains,
            representative_count=min(256, n_frames),
            update_rounds=4,
            chunk_size=1000,
            precision="float32",
            rotation_solver="horn",
        )
        gpu, gpu_times = repeat_call(lambda: worker.throughput(coords))
        timing_rows.append(
            _timing_row(
                row=row,
                contract="A_analysis_metrics_transforms_rmsd_rmsf",
                method="GPU_throughput_float32_Horn",
                runtimes=gpu_times,
                result=gpu,
            )
        )

        for method, candidate in (
            ("CPU_throughput", cpu),
            ("GPU_throughput_float32_Horn", gpu),
        ):
            metrics = quality_against_exact(
                coords,
                chains,
                candidate,
                exact,
            )
            case_pass = (
                metrics["finite_outputs"]
                and metrics["core_jaccard"]
                >= criteria["minimum_core_jaccard_min"]
                and metrics["weight_spearman"]
                >= criteria["minimum_weight_spearman_min"]
                and metrics["rmsf_spearman"]
                >= criteria["minimum_rmsf_spearman_min"]
                and metrics["median_abs_rmsd_difference_A"]
                <= criteria["median_absolute_rmsd_difference_angstrom_max"]
                and metrics["p95_abs_rmsd_difference_A"]
                <= criteria["p95_absolute_rmsd_difference_angstrom_max"]
            )
            quality_rows.append(
                {
                    "system": row["system"],
                    "source_type": row["source_type"],
                    "method": method,
                    "n_frames": n_frames,
                    "n_residues": coords.shape[1],
                    **metrics,
                    "case_quality_pass": case_pass,
                }
            )

        cpu_w5, cpu_w5_times = repeat_call(
            lambda: scan_local_kinematics(
                coords,
                chain_ids=chains,
                evaluate_blocks=False,
            )
        )
        gpu_w5, gpu_w5_times = repeat_call(
            lambda: scan_w5_gpu(
                coords,
                chain_ids=chains,
                precision="float32",
            )
        )
        timing_rows.extend(
            (
                _timing_row(
                    row=row,
                    contract="W5_local_deformation_not_alignment_equivalent",
                    method="CPU_W5",
                    runtimes=cpu_w5_times,
                    result=cpu_w5,
                ),
                _timing_row(
                    row=row,
                    contract="W5_local_deformation_not_alignment_equivalent",
                    method="GPU_W5",
                    runtimes=gpu_w5_times,
                    result=gpu_w5,
                ),
            )
        )
        w5_rows.append(
            {
                "system": row["system"],
                "n_frames": n_frames,
                "n_residues": coords.shape[1],
                "local_deformation_spearman_cpu_gpu": spearman(
                    cpu_w5.local_deformation_index,
                    gpu_w5.local_deformation_index,
                ),
                "local_rigidity_spearman_cpu_gpu": spearman(
                    cpu_w5.local_rigidity_score,
                    gpu_w5.local_rigidity_score,
                ),
                "cpu_median_seconds": np.median(cpu_w5_times),
                "gpu_median_seconds": np.median(gpu_w5_times),
                "gpu_speedup": (
                    np.median(cpu_w5_times) / np.median(gpu_w5_times)
                ),
                "W5_called_RMSD": False,
            }
        )
        _save_result(
            QUALITY / "result_cache" / f"{row['system']}.npz",
            exact,
            cpu,
            gpu,
        )

    write_tsv(timing_path, timing_rows)
    write_tsv(quality_path, quality_rows)
    write_tsv(w5_path, w5_rows)

    gpu_rows = [
        row
        for row in quality_rows
        if row["method"] == "GPU_throughput_float32_Horn"
    ]
    def values(field: str) -> list[float]:
        return [float(row[field]) for row in gpu_rows]

    def truth(value: object) -> bool:
        return str(value).strip().lower() == "true"

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria_file": CRITERIA.relative_to(RESULTS.parent.parent).as_posix(),
        "n_systems": len(gpu_rows),
        "n_case_quality_pass": sum(
            truth(row["case_quality_pass"]) for row in gpu_rows
        ),
        "median_core_jaccard": float(
            np.median(values("core_jaccard"))
        ),
        "minimum_core_jaccard": float(
            np.min(values("core_jaccard"))
        ),
        "median_weight_spearman": float(
            np.median(values("weight_spearman"))
        ),
        "minimum_weight_spearman": float(
            np.min(values("weight_spearman"))
        ),
        "median_rmsf_spearman": float(
            np.median(values("rmsf_spearman"))
        ),
        "minimum_rmsf_spearman": float(
            np.min(values("rmsf_spearman"))
        ),
        "median_flexible_label_agreement": float(
            np.median(values("flexible_label_agreement"))
        ),
        "median_absolute_rmsd_difference_A": float(
            np.median(values("median_abs_rmsd_difference_A"))
        ),
        "p95_system_p95_absolute_rmsd_difference_A": float(
            np.quantile(
                values("p95_abs_rmsd_difference_A"),
                0.95,
            )
        ),
        "median_heldout_pairwise_variance_relative_difference": float(
            np.median(
                values("heldout_pairwise_variance_relative_difference")
            )
        ),
        "median_contact_persistence_mae": float(
            np.median(values("contact_persistence_mae"))
        ),
        "numerical_failure_count": sum(
            not truth(row["finite_outputs"]) for row in gpu_rows
        ),
    }
    aggregate["locked_aggregate_gate_pass"] = bool(
        aggregate["n_case_quality_pass"] == aggregate["n_systems"]
        and aggregate["median_core_jaccard"]
        >= criteria["median_core_jaccard_min"]
        and aggregate["minimum_core_jaccard"]
        >= criteria["minimum_core_jaccard_min"]
        and aggregate["median_weight_spearman"]
        >= criteria["median_weight_spearman_min"]
        and aggregate["minimum_weight_spearman"]
        >= criteria["minimum_weight_spearman_min"]
        and aggregate["median_rmsf_spearman"]
        >= criteria["median_rmsf_spearman_min"]
        and aggregate["minimum_rmsf_spearman"]
        >= criteria["minimum_rmsf_spearman_min"]
        and aggregate["median_flexible_label_agreement"]
        >= criteria["median_flexible_label_agreement_min"]
        and aggregate["median_heldout_pairwise_variance_relative_difference"]
        <= criteria[
            "median_heldout_pairwise_variance_relative_difference_max"
        ]
        and aggregate["median_contact_persistence_mae"]
        <= criteria["median_contact_persistence_mae_max"]
        and aggregate["numerical_failure_count"] == 0
    )
    write_json(QUALITY / "quality_gate_summary.json", aggregate)
    (QUALITY / "quality_performance_envelope.md").write_text(
        "\n".join(
            [
                "# U5 quality-performance envelope",
                "",
                f"- Systems: `{aggregate['n_systems']}`",
                f"- Per-case pass: `{aggregate['n_case_quality_pass']}/{aggregate['n_systems']}`",
                f"- Median core Jaccard: `{aggregate['median_core_jaccard']:.6f}`",
                f"- Minimum core Jaccard: `{aggregate['minimum_core_jaccard']:.6f}`",
                f"- Median weight Spearman: `{aggregate['median_weight_spearman']:.6f}`",
                f"- Median RMSF Spearman: `{aggregate['median_rmsf_spearman']:.6f}`",
                f"- Median RMSF/RMSD quality gate passed: `{aggregate['locked_aggregate_gate_pass']}`",
                "- Quality was evaluated per system; frames were not pooled.",
                "- Transform-identical parity was not required.",
                "- W5 remains a non-equivalent local-deformation contract.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
