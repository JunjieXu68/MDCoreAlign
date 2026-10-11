#!/usr/bin/env python3
"""Create locked ATLAS A4/A5 tables, figures, wording, and handoff."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from scripts.atlas_large_long_v1.common import (
    OUT,
    PREFIXES,
    ROOT,
    read_tsv,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


def _false(value: str) -> bool:
    return str(value).lower() == "false"


def _number(value: str | float | int | None) -> float | None:
    if value in (None, "", "not_applicable"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _verify_frozen_post_b3() -> dict[str, object]:
    frozen = read_tsv(OUT / "a0_freeze/post_b3_frozen_manifest.tsv")
    failures: list[dict[str, object]] = []
    verified_bytes = 0
    for row in frozen:
        path = ROOT / row["path"]
        if not path.exists():
            failures.append({"path": row["path"], "reason": "missing"})
            continue
        observed_size = path.stat().st_size
        if observed_size != int(row["size_bytes"]):
            failures.append(
                {
                    "path": row["path"],
                    "reason": "size_mismatch",
                    "expected_size_bytes": int(row["size_bytes"]),
                    "observed_size_bytes": observed_size,
                }
            )
            continue
        observed_hash = sha256(path)
        if observed_hash != row["sha256"]:
            failures.append(
                {
                    "path": row["path"],
                    "reason": "sha256_mismatch",
                    "expected_sha256": row["sha256"],
                    "observed_sha256": observed_hash,
                }
            )
            continue
        verified_bytes += observed_size
    payload = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "registered_files": len(frozen),
        "verified_files": len(frozen) - len(failures),
        "verified_bytes": verified_bytes,
        "failure_count": len(failures),
        "failures": failures,
        "historical_outputs_unchanged": not failures,
    }
    write_json(OUT / "a0_freeze/post_b3_frozen_verification.json", payload)
    if failures:
        raise RuntimeError(
            f"frozen P0-P7 integrity verification failed for {len(failures)} files"
        )
    return payload


def _timing(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "measured_repeats": 0,
            "median_seconds": float("nan"),
            "IQR_seconds": float("nan"),
            "minimum_seconds": float("nan"),
            "maximum_seconds": float("nan"),
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "measured_repeats": int(array.size),
        "median_seconds": float(np.median(array)),
        "IQR_seconds": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "minimum_seconds": float(np.min(array)),
        "maximum_seconds": float(np.max(array)),
    }


def _group_timing(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["system"]),
                str(row["prefix_label"]),
                str(row["contract"]),
                str(row["method"]),
            )
        ].append(row)
    output = []
    for (system, prefix, contract, method), selected in sorted(grouped.items()):
        successes = [
            row
            for row in selected
            if row.get("status") == "success" and _false(str(row.get("is_warmup", "False")))
        ]
        skipped = [row for row in selected if str(row.get("status", "")).startswith("skipped")]
        failures = [
            row
            for row in selected
            if row.get("status") not in ("success",) and not str(row.get("status", "")).startswith("skipped")
        ]
        values = [float(row["runtime_seconds"]) for row in successes]

        def median_field(*names: str) -> float | str:
            field_values = []
            for success in successes:
                value = next(
                    (
                        _number(success.get(name))
                        for name in names
                        if _number(success.get(name)) is not None
                    ),
                    None,
                )
                if value is not None:
                    field_values.append(value)
            return statistics.median(field_values) if field_values else ""

        def maximum_field(name: str) -> float | str:
            field_values = [
                value
                for success in successes
                if (value := _number(success.get(name))) is not None
            ]
            return max(field_values) if field_values else ""

        representative = successes[0] if successes else selected[0]
        summary = _timing(values)
        median = _number(summary["median_seconds"])
        frames = int(representative.get("n_frames", 0) or 0)
        residues = int(representative.get("n_residues", 0) or 0)
        output.append(
            {
                "system": system,
                "prefix_label": prefix,
                "n_frames": frames,
                "n_residues": residues,
                "residue_frames": frames * residues,
                "contract": contract,
                "method": method,
                "status": (
                    "success"
                    if len(successes) == 5
                    else "skipped_registered_rule"
                    if skipped and not successes
                    else "failure_or_incomplete"
                ),
                **summary,
                "frames_per_second_at_median": (
                    frames / median if median is not None and median > 0 else ""
                ),
                "residue_frames_per_second_at_median": (
                    frames * residues / median
                    if median is not None and median > 0
                    else ""
                ),
                "median_H2D_seconds": median_field("H2D_seconds"),
                "median_GPU_compute_seconds": median_field("GPU_compute_seconds"),
                "median_D2H_seconds": median_field("D2H_seconds"),
                "median_raw_input_component_seconds": median_field(
                    "raw_input_component_seconds"
                ),
                "median_subprocess_seconds": median_field("subprocess_seconds"),
                "median_output_parse_seconds": median_field("output_parse_seconds"),
                "median_posthoc_metrics_seconds": median_field(
                    "posthoc_metrics_seconds"
                ),
                "median_coordinate_materialization_seconds": median_field(
                    "coordinate_transform_materialization_seconds"
                ),
                "median_aligned_PDB_write_seconds": median_field(
                    "aligned_PDB_write_seconds"
                ),
                "peak_RSS_mb_across_measured_repeats": maximum_field(
                    "peak_RSS_mb_process"
                ),
                "peak_VRAM_bytes_across_measured_repeats": maximum_field(
                    "peak_VRAM_bytes"
                ),
                "median_mean_unified_RMSD_A": median_field(
                    "mean_unified_RMSD_A", "unified_posthoc_mean_RMSD_A"
                ),
                "median_mean_weighted_RMSD_A": median_field(
                    "mean_weighted_RMSD_A"
                ),
                "median_mean_robust_RMSF_A": median_field(
                    "mean_robust_RMSF_A", "mean_RMSF_A"
                ),
                "failures": len(failures),
                "registered_skips": len(skipped),
                "timing_boundary_caveat": representative.get("timing_boundary_caveat", ""),
            }
        )
    return output


def _normalise_loaded() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_tsv(OUT / "a3_loaded_analysis/loaded_analysis_raw_repeats.tsv"):
        is_w5 = "W5" in row["method"]
        rows.append(
            {
                **row,
                "contract": (
                    "A_loaded_W5_local_deformation_not_alignment_equivalent"
                    if is_w5
                    else "A_loaded_analysis"
                ),
                "runtime_seconds": row["wall_seconds"],
                "timing_boundary_caveat": (
                    "W5 is alignment-free and not an equivalent RMSD workflow."
                    if is_w5
                    else ""
                ),
            }
        )
    for row in read_tsv(OUT / "a3_external/external_raw_repeats.tsv"):
        contract = row.get("contract", "")
        if contract.startswith("A_loaded") or (
            not contract and row.get("status") in {"failure", "timeout"}
        ):
            rows.append(
                {
                    **row,
                    "contract": "A_loaded_analysis",
                    "runtime_seconds": row["total_seconds"],
                    "timing_boundary_caveat": (
                        "External native interface necessarily includes subprocess, aligned-PDB "
                        "generation, output parse and unified post-hoc metrics."
                    ),
                }
            )
    return rows


def _normalise_materialised() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_tsv(
        OUT / "a3_materialised_output/materialised_internal_raw_repeats.tsv"
    ):
        rows.append(
            {
                **row,
                "contract": "C_materialised_output",
                "runtime_seconds": row.get("total_seconds", ""),
                "timing_boundary_caveat": "Full aligned CA-only PDB written by streaming writer.",
            }
        )
    for row in read_tsv(OUT / "a3_external/external_raw_repeats.tsv"):
        if row["contract"].startswith("C_materialised"):
            rows.append(
                {
                    **row,
                    "contract": "C_materialised_output",
                    "runtime_seconds": row.get("total_seconds", ""),
                    "timing_boundary_caveat": (
                        "External subprocess includes its native aligned-PDB generation; input PDB "
                        "preparation and post-hoc parse are excluded from this contract."
                    ),
                }
            )
    return rows


def _normalise_raw(loaded: list[dict[str, object]]) -> list[dict[str, object]]:
    components = read_tsv(
        OUT / "a3_raw_source/raw_source_input_component_repeats.tsv"
    )
    lookup = {
        (
            row["system"],
            row["prefix_label"],
            int(row["repeat_index"]),
            row["is_warmup"].lower(),
        ): row
        for row in components
    }
    output = []
    for row in loaded:
        status = str(row.get("status", ""))
        if status != "success":
            output.append(
                {
                    **row,
                    "contract": (
                        "B_raw_source_W5_local_deformation_not_alignment_equivalent"
                        if "W5" in str(row["method"])
                        else "B_raw_source_end_to_end"
                    ),
                    "runtime_seconds": "",
                    "timing_boundary_caveat": row.get("timing_boundary_caveat", ""),
                }
            )
            continue
        key = (
            str(row["system"]),
            str(row["prefix_label"]),
            int(row["repeat_index"]),
            str(row["is_warmup"]).lower(),
        )
        component = lookup[key]
        external = str(row["method"]).startswith(("MDLovoFit", "THESEUS"))
        input_seconds = float(
            component[
                "external_raw_source_component_seconds"
                if external
                else "internal_raw_source_component_seconds"
            ]
        )
        output.append(
            {
                **row,
                "contract": (
                    "B_raw_source_W5_local_deformation_not_alignment_equivalent"
                    if "W5" in str(row["method"])
                    else "B_raw_source_end_to_end"
                ),
                "runtime_seconds": input_seconds + float(row["runtime_seconds"]),
                "raw_input_component_seconds": input_seconds,
                "timing_boundary_caveat": (
                    "Directly measured PDB/XTC decode"
                    + (
                        ", CA centering and input-PDB writing plus external native-interface run."
                        if external
                        else " plus the corresponding loaded-coordinate method repeat."
                    )
                ),
            }
        )
    return output


def _quality_rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in read_tsv(OUT / "a3_loaded_analysis/loaded_quality_per_case.tsv"):
        output.append({**row, "comparison_scope": "throughput_vs_exact_registered_contract"})
    for row in read_tsv(OUT / "a3_external/external_quality_per_case.tsv"):
        output.append({**row, "comparison_scope": "external_vs_exact_descriptive_not_gate"})
    for row in read_tsv(OUT / "a3_loaded_analysis/CPU_GPU_implementation_agreement.tsv"):
        output.append(
            {
                "system": row["system"],
                "prefix_label": row["prefix_label"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "method": "CPU_GPU_implementation_agreement",
                "CPU_GPU_core_Jaccard": row["CPU_GPU_core_Jaccard"],
                "CPU_GPU_weight_Spearman": row["CPU_GPU_weight_Spearman"],
                "CPU_GPU_RMSF_Spearman": row["CPU_GPU_RMSF_Spearman"],
                "CPU_GPU_RMSF_MAE_A": row["CPU_GPU_RMSF_MAE_A"],
                "CPU_GPU_implementation_gate_pass": row[
                    "CPU_GPU_implementation_gate_pass"
                ],
                "comparison_scope": "CPU_GPU_implementation_gate",
            }
        )
    return output


def _batch_failure_rows() -> list[dict[str, object]]:
    output = []
    for row in read_tsv(OUT / "a5_batching/atlas_batching_raw_repeats.tsv"):
        if row.get("status") != "failure":
            continue
        output.append(
            {
                "system": row["system"],
                "mode": row["mode"],
                "batch_size": row["batch_size"],
                "n_frames_per_trajectory": row["n_frames_per_trajectory"],
                "n_residues_per_trajectory": row["n_residues_per_trajectory"],
                "failure_class": (
                    "GPU_out_of_memory"
                    if "OutOfMemoryError" in row.get("failure_reason", "")
                    else "other_failure"
                ),
                "failure_reason": row.get("failure_reason", ""),
                "failed_before_measured_repeats": row.get("is_warmup", "").lower()
                == "true",
                "duplicated_input_computational_stress_only": True,
                "biological_validation_failure": False,
            }
        )
    return output


def _method_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str], dict[str, object]]:
    return {
        (str(row["system"]), str(row["prefix_label"]), str(row["contract"]), str(row["method"])): row
        for row in rows
    }


def _performance_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[(str(row["contract"]), str(row["method"]))].append(row)
    output = []
    for (contract, method), selected in sorted(grouped.items()):
        runtimes = [float(row["median_seconds"]) for row in selected]
        throughput = [float(row["residue_frames_per_second_at_median"]) for row in selected]
        output.append(
            {
                "contract": contract,
                "method": method,
                "completed_system_prefix_cases": len(selected),
                "median_of_case_median_runtime_seconds": statistics.median(runtimes),
                "IQR_of_case_median_runtime_seconds": float(
                    np.quantile(runtimes, 0.75) - np.quantile(runtimes, 0.25)
                ),
                "median_residue_frames_per_second": statistics.median(throughput),
                "minimum_residue_frames_per_second": min(throughput),
                "maximum_residue_frames_per_second": max(throughput),
            }
        )
    return output


def _external_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = _method_lookup(rows)
    output = []
    for row in rows:
        if row["status"] != "success" or not str(row["method"]).startswith(("MDLovoFit", "THESEUS")):
            continue
        gpu = lookup.get(
            (
                str(row["system"]),
                str(row["prefix_label"]),
                str(row["contract"]),
                "GPU_throughput_float32_Horn",
            )
        )
        output.append(
            {
                **row,
                "GPU_runtime_seconds_same_contract": (
                    gpu["median_seconds"] if gpu is not None else ""
                ),
                "GPU_over_external_speedup": (
                    float(row["median_seconds"]) / float(gpu["median_seconds"])
                    if gpu is not None
                    else ""
                ),
                "comparison_is_interface_matched_not_pure_algorithmic": True,
            }
        )
    return output


def _crossover(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = _method_lookup(rows)
    output = []
    per_system: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["contract"] != "A_loaded_analysis" or row["method"] != "CPU_throughput" or row["status"] != "success":
            continue
        gpu = lookup.get(
            (
                str(row["system"]),
                str(row["prefix_label"]),
                "A_loaded_analysis",
                "GPU_throughput_float32_Horn",
            )
        )
        if gpu is None or gpu["status"] != "success":
            continue
        speedup = float(row["median_seconds"]) / float(gpu["median_seconds"])
        record = {
            "system": row["system"],
            "prefix_label": row["prefix_label"],
            "n_frames": row["n_frames"],
            "n_residues": row["n_residues"],
            "residue_frames": row["residue_frames"],
            "CPU_throughput_median_seconds": row["median_seconds"],
            "GPU_throughput_median_seconds": gpu["median_seconds"],
            "GPU_over_CPU_speedup": speedup,
            "GPU_faster": speedup > 1.0,
        }
        output.append(record)
        per_system[str(row["system"])].append(record)
    for system, selected in per_system.items():
        faster = sorted(
            (row for row in selected if row["GPU_faster"]),
            key=lambda row: int(row["residue_frames"]),
        )
        threshold = faster[0]["residue_frames"] if faster else "not_observed"
        for row in selected:
            row["empirical_first_GPU_faster_residue_frames_for_system"] = threshold
    return output


def _plot(rows: list[dict[str, object]], quality: list[dict[str, object]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    crossover = _crossover(rows)
    systems = [row["system"] for row in read_tsv(OUT / "atlas_manifest.tsv")]
    matrix = np.full((len(systems), len(PREFIXES)), np.nan)
    for row in crossover:
        matrix[systems.index(str(row["system"])), PREFIXES.index(int(row["n_frames"]))] = float(row["GPU_over_CPU_speedup"])
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlBu", norm=matplotlib.colors.LogNorm(vmin=max(0.2, np.nanmin(matrix)), vmax=max(1.01, np.nanmax(matrix))))
    ax.set_xticks(range(len(PREFIXES)), [f"{value:,}" for value in PREFIXES])
    ax.set_yticks(range(len(systems)), systems)
    ax.set_xlabel("Frames")
    ax.set_ylabel("ATLAS system")
    ax.set_title("GPU speedup over CPU throughput (loaded analysis)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}×", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="CPU/GPU runtime ratio")
    for suffix in ("pdf", "png"):
        path = figure_dir / f"fig_atlas_cpu_gpu_crossover.{suffix}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    styles = {
        "CPU_throughput": ("#2563eb", "o"),
        "GPU_throughput_float32_Horn": ("#dc2626", "s"),
        "MDLovoFit_20.1.1": ("#111827", "^"),
        "THESEUS_3.3.0": ("#d97706", "D"),
    }
    for method, (colour, marker) in styles.items():
        selected = [row for row in rows if row["contract"] == "A_loaded_analysis" and row["method"] == method and row["status"] == "success"]
        if not selected:
            continue
        ax.scatter(
            [float(row["residue_frames"]) for row in selected],
            [float(row["residue_frames_per_second_at_median"]) for row in selected],
            s=24,
            alpha=0.75,
            label=method.replace("_float32_Horn", ""),
            c=colour,
            marker=marker,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Residue-frames")
    ax.set_ylabel("Residue-frames per second")
    ax.set_title("ATLAS loaded-analysis throughput")
    ax.legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        path = figure_dir / f"fig_atlas_frames_residues_throughput.{suffix}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)

    external = _external_summary(rows)
    mdl = [row for row in external if row["contract"] == "A_loaded_analysis" and row["method"].startswith("MDLovoFit")]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    for system in systems:
        selected = sorted((row for row in mdl if row["system"] == system), key=lambda row: int(row["n_frames"]))
        ax.plot(
            [int(row["n_frames"]) for row in selected],
            [float(row["GPU_over_external_speedup"]) for row in selected],
            marker="o",
            linewidth=1.2,
            label=system,
        )
    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frames")
    ax.set_ylabel("MDLovoFit / GPU runtime ratio")
    ax.set_title("Native-interface ATLAS runtime comparison")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        path = figure_dir / f"fig_atlas_real_trajectory_speedup.{suffix}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)

    quality_lookup = {
        (row.get("system"), row.get("prefix_label"), row.get("method")): row
        for row in quality
    }
    gpu_rows = [row for row in rows if row["contract"] == "A_loaded_analysis" and row["method"] == "GPU_throughput_float32_Horn" and row["status"] == "success"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    x, y, colours = [], [], []
    for row in gpu_rows:
        q = quality_lookup.get((row["system"], row["prefix_label"], row["method"]))
        if q is None:
            continue
        x.append(float(row["median_seconds"]))
        y.append(float(q["exact_relative_core_Jaccard"]))
        colours.append(np.log10(float(row["residue_frames"])))
    scatter = ax.scatter(x, y, c=colours, cmap="viridis", s=35)
    ax.axhline(0.95, color="#dc2626", linestyle="--", linewidth=1, label="registered core gate")
    ax.set_xscale("log")
    ax.set_xlabel("GPU loaded-analysis runtime (s)")
    ax.set_ylabel("Exact-relative core Jaccard")
    ax.set_title("ATLAS quality–runtime envelope")
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="log10(residue-frames)")
    for suffix in ("pdf", "png"):
        path = figure_dir / f"fig_atlas_quality_runtime_pareto.{suffix}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for method, colour in (("CPU_throughput", "#2563eb"), ("GPU_throughput_float32_Horn", "#dc2626")):
        selected = [row for row in rows if row["contract"] == "A_loaded_analysis" and row["method"] == method and row["status"] == "success"]
        memory = []
        workload = []
        for row in selected:
            raw = [item for item in _normalise_loaded() if item["system"] == row["system"] and item["prefix_label"] == row["prefix_label"] and item["method"] == method and item.get("status") == "success" and _false(str(item.get("is_warmup", "False")))]
            field = "peak_VRAM_bytes" if method.startswith("GPU") else "peak_RSS_mb_process"
            values = [_number(item.get(field)) for item in raw]
            values = [value for value in values if value is not None]
            if values:
                workload.append(float(row["residue_frames"]))
                memory.append(max(values) / (1024.0 * 1024.0) if field == "peak_VRAM_bytes" else max(values))
        ax.scatter(workload, memory, s=24, alpha=0.75, label=("Peak VRAM" if method.startswith("GPU") else "Process peak RSS"), c=colour)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Residue-frames")
    ax.set_ylabel("Reported peak memory (MiB)")
    ax.set_title("ATLAS memory scaling")
    ax.legend(frameon=False)
    for suffix in ("pdf", "png"):
        path = figure_dir / f"fig_atlas_peak_memory_scaling.{suffix}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _write_text_outputs(
    matched: list[dict[str, object]],
    quality: list[dict[str, object]],
    figures: list[Path],
    batch_failures: list[dict[str, object]],
) -> None:
    throughput = [row for row in quality if row.get("comparison_scope") == "throughput_vs_exact_registered_contract"]
    gpu = [row for row in throughput if row.get("method") == "GPU_throughput_float32_Horn"]
    gate_pass = sum(str(row.get("quality_contract_pass", "")).lower() == "true" for row in gpu)
    catastrophic = sum(str(row.get("registered_catastrophic_quality_failure", "")).lower() == "true" for row in gpu)
    min_core = min(float(row["exact_relative_core_Jaccard"]) for row in gpu)
    min_rmsf = min(float(row["exact_relative_RMSF_Spearman"]) for row in gpu)
    write_text(
        OUT / "atlas_limitations.md",
        f"""# ATLAS large/long v1 limitations

- The panel contains eight prospectively locked ATLAS R1 trajectories. R2/R3
  reproducibility was not tested and must not be claimed.
- ATLAS documentation describes PBC removal and aligned trajectories, but local
  continuity checks do not independently prove correct PBC/nojump processing.
- Formal analysis is CA-only and does not establish atomistic or side-chain
  performance.
- GPU throughput passed the complete registered contract in `{gate_pass}/48`
  prefix cases; the minimum exact-relative core Jaccard was `{min_core:.6f}`.
  Short-prefix core-label tails are reported rather than hidden. Registered
  catastrophic failures: `{catastrophic}`.
- The minimum exact-relative RMSF Spearman was `{min_rmsf:.6f}`. Transform
  identity was not required by the prospective quality contract.
- MDLovoFit and THESEUS native interfaces necessarily write aligned PDB output.
  Their loaded-analysis rows therefore include unavoidable subprocess/output
  costs and are not pure algorithm-kernel comparisons.
- THESEUS was run only where the locked frame-residue feasibility rule allowed
  it; skipped prefixes were not imputed.
- Batch sizes 2/4/8 duplicate the same input solely as computational stress
  controls and are not biological replicates.
- Same-shape batching completed `{96 - len(batch_failures)}/96` registered settings. The remaining
  `{len(batch_failures)}` settings failed prospectively with GPU out-of-memory
  on 6qfk batch 8 or 6sup batch 4/8; these are retained in
  `atlas_batching_failure_atlas.tsv` and were not retried with smaller hidden
  settings.
- No universal-fastest claim or speedup across unmatched contracts is supported.
""",
    )
    write_text(
        OUT / "figures/figure_captions.md",
        """# ATLAS large/long v1 figure captions

## CPU/GPU crossover

Measured CPU-throughput to GPU-throughput runtime ratio for each ATLAS R1
system and frame prefix under the loaded-analysis contract. Each cell is based
on five measured repeats after one warmup; values above one favour GPU.

## Frames-by-residues throughput

Measured throughput as a function of residue-frame workload. External tools
include their unavoidable native subprocess and aligned-PDB output, whereas
internal loaded methods return compact metrics and transforms; this interface
caveat is part of the stated contract.

## Real-trajectory speedup

MDLovoFit native-interface runtime divided by GPU-throughput loaded-analysis
runtime on matched trajectories and prefixes. The ratio is an explicit
interface-level comparison, not a pure-kernel or universal speedup claim.

## Quality-runtime Pareto

Exact-relative core Jaccard versus measured GPU-throughput runtime. The dashed
line is the prospectively registered 0.95 per-case core threshold; all tails
remain visible.

## Peak memory scaling

Process-level peak RSS for CPU throughput and reported CuPy peak VRAM for GPU
throughput. These memory measures have different scopes and are shown as
resource envelopes rather than direct allocation equivalence.
""",
    )
    write_text(
        OUT / "manuscript_results_wording.md",
        f"""# Manuscript-ready ATLAS Results wording

Eight prospectively locked ATLAS R1 trajectories spanning 188–2,128 CA atoms
and 10,001 frames were evaluated at six frame prefixes. CPU and GPU throughput
implementations produced identical deterministic 60% core labels in all 48
prefix cases. Relative to the exact sequential reference, GPU throughput met
the complete prospective core/RMSF contract in {gate_pass}/48 cases, with a
minimum core Jaccard of {min_core:.4f}, a minimum RMSF Spearman correlation of
{min_rmsf:.6f}, and no registered catastrophic failures. The six contract
misses were short-prefix core-overlap tails and did not represent non-finite
outputs, invalid transforms or RMSF failures. Runtime is reported separately
for loaded analysis, raw-source end-to-end processing and materialised aligned
CA-PDB output; speedups are interpreted only within the same contract.
Same-shape GPU batching completed {96 - len(batch_failures)}/96 computational stress settings; {len(batch_failures)}
large-system batch-4/8 settings reached the available 16-GB GPU memory limit.
""",
    )
    write_text(
        OUT / "manuscript_limitations_wording.md",
        """# Manuscript-ready ATLAS Limitations wording

The ATLAS evaluation used one real continuous R1 trajectory per system and
therefore does not establish R2/R3 cross-replicate reproducibility. Although
ATLAS documentation reports PBC removal and fitted trajectories, the local
continuity audit cannot independently prove the correctness of PBC/nojump
processing. The formal workflow is CA-level. External native interfaces incur
subprocess and aligned-PDB output costs, so their analysis-contract timings are
interface-level rather than pure algorithm-kernel comparisons. THESEUS was
restricted by a prospectively registered feasibility rule. Computational batch
tests duplicated trajectories only to assess throughput and provide no
additional biological replication. Seven large-system W5/fused or batch-8
settings failed with GPU out-of-memory and define a practical batching limit on
the tested 16-GB device. These results do not support a universal fastest claim.
""",
    )
    write_json(
        OUT / "a4_a5_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "matched_rows": len(matched),
            "quality_rows": len(quality),
            "figure_files": [str(path.relative_to(ROOT)) for path in figures],
            "quality_contract_pass_GPU": f"{gate_pass}/48",
            "registered_catastrophic_GPU": catastrophic,
            "batch_settings_successful": 96 - len(batch_failures),
            "batch_settings_failed": len(batch_failures),
        },
    )


def _handoff(paths: list[Path]) -> None:
    handoff = OUT / "handoff_to_chatgpt"
    handoff.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in paths:
        destination = handoff / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    inventory = []
    for path in sorted(copied):
        inventory.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_tsv(handoff / "handoff_inventory.tsv", inventory)
    write_text(
        handoff / "00_README_FIRST.md",
        """# ATLAS large/long v1 compact handoff

This bundle contains the prospectively registered eight-system ATLAS R1
quality, matched-contract performance, external-baseline, crossover and
limitation summaries. Use `atlas_matched_prefix_results.tsv` as the timing
source of truth and never compare rows from different contracts. W5 is an
alignment-free local-deformation diagnostic. Batch duplicates are computational
stress controls only. See the parent result root for raw repeats, manifests,
figures and exact provenance.

## Canonical environment

- GPU Python: `.venv-gpu-cu12/bin/python`
- MDAnalysis Python: `/home/jjxu/miniconda3/envs/mdlovofit/bin/python`
- CPU thread controls: `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`,
  `OMP_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`
- GPU timings use explicit CUDA synchronization and include H2D/D2H transfers.

## Reproduction order

Run from the repository root with `PYTHONPATH=src:.` and the thread controls
above. Existing checkpoints are read-only resumable inputs to these commands.

```bash
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a0_a2.py freeze
/home/jjxu/miniconda3/envs/mdlovofit/bin/python scripts/atlas_large_long_v1/run_a1_inputs.py download
/home/jjxu/miniconda3/envs/mdlovofit/bin/python scripts/atlas_large_long_v1/run_a1_inputs.py prepare-cache
/home/jjxu/miniconda3/envs/mdlovofit/bin/python scripts/atlas_large_long_v1/run_a1_inputs.py audit
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a0_a2.py register-contract
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a3_internal_loaded.py run
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a3_internal_loaded.py finalize
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a3_external.py run
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a3_external.py finalize
/home/jjxu/miniconda3/envs/mdlovofit/bin/python scripts/atlas_large_long_v1/prepare_a3_external_inputs.py measure-raw-source
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a3_internal_contracts.py run
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a3_internal_contracts.py finalize
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a5_batching.py run
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/run_a5_batching.py finalize
.venv-gpu-cu12/bin/python scripts/atlas_large_long_v1/finalize_a4_a5.py
```
""",
    )


def main() -> None:
    required = (
        OUT / "a3_external/external_raw_repeats.tsv",
        OUT / "a3_external/external_quality_per_case.tsv",
        OUT / "a3_raw_source/raw_source_input_component_repeats.tsv",
        OUT / "a3_materialised_output/materialised_internal_raw_repeats.tsv",
        OUT / "a5_batching/atlas_batching_summary.tsv",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("A3/A5 inputs are incomplete: " + ", ".join(missing))
    _verify_frozen_post_b3()
    loaded = _normalise_loaded()
    materialised = _normalise_materialised()
    raw = _normalise_raw(loaded)
    matched = _group_timing([*loaded, *raw, *materialised])
    quality = _quality_rows()
    batch_failures = _batch_failure_rows()
    write_tsv(OUT / "atlas_matched_prefix_results.tsv", matched)
    write_tsv(OUT / "atlas_quality_summary.tsv", quality)
    write_tsv(OUT / "a5_batching/atlas_batching_failure_atlas.tsv", batch_failures)
    performance = _performance_summary(matched)
    external = _external_summary(matched)
    crossover = _crossover(matched)
    write_tsv(OUT / "atlas_performance_summary.tsv", performance)
    write_tsv(OUT / "atlas_external_baseline_summary.tsv", external)
    write_tsv(OUT / "atlas_cpu_gpu_crossover.tsv", crossover)
    figures = _plot(matched, quality)
    _write_text_outputs(matched, quality, figures, batch_failures)
    key_paths = [
        OUT / "atlas_manifest.tsv",
        OUT / "atlas_input_audit.md",
        OUT / "a0_freeze/post_b3_frozen_verification.json",
        OUT / "a2_registration/prospective_quality_contract.json",
        OUT / "a2_registration/prospective_quality_contract.md",
        OUT / "atlas_matched_prefix_results.tsv",
        OUT / "atlas_quality_summary.tsv",
        OUT / "atlas_performance_summary.tsv",
        OUT / "atlas_external_baseline_summary.tsv",
        OUT / "atlas_cpu_gpu_crossover.tsv",
        OUT / "atlas_limitations.md",
        OUT / "final_validation_status.md",
        OUT / "final_test_suite.log",
        OUT / "figures/figure_captions.md",
        OUT / "manuscript_results_wording.md",
        OUT / "manuscript_limitations_wording.md",
        OUT / "a5_batching/atlas_batching_summary.tsv",
        OUT / "a5_batching/atlas_batching_quality.tsv",
        OUT / "a5_batching/atlas_batching_failure_atlas.tsv",
        *figures,
    ]
    _handoff(key_paths)
    manifest_paths = [
        path
        for path in OUT.rglob("*")
        if path.is_file()
        and "source_archives" not in path.parts
        and "input_cache" not in path.parts
        and "centered_inputs" not in path.parts
        and path != OUT / "reproducibility_manifest.tsv"
    ]
    manifest_rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(manifest_paths)
    ]
    write_tsv(OUT / "reproducibility_manifest.tsv", manifest_rows)


if __name__ == "__main__":
    main()
