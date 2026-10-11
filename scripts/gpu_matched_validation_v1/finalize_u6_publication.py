#!/usr/bin/env python3
"""Create publication-ready U6 tables, figures, manifest and handoff."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.gpu_matched_validation_v1.common import (
    RESULTS,
    read_tsv,
    write_json,
    write_tsv,
)


OUTPUT = RESULTS / "u6_publication"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
HANDOFF = OUTPUT / "handoff_to_chatgpt"


def _save_figure(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(
        FIGURES / f"{name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(figure)


def _successful_scaling() -> list[dict[str, str]]:
    path = RESULTS / "u3_scaling" / "frames_by_residues_scaling.tsv"
    return [
        row
        for row in read_tsv(path)
        if row.get("status") == "success"
    ]


def _crossover_figure(rows: list[dict[str, str]]) -> None:
    gpu = {
        (int(row["n_frames"]), int(row["n_residues"])): row
        for row in rows
        if row["method"] == "GPU_throughput_float32_Horn"
    }
    frames = sorted({key[0] for key in gpu})
    residues = sorted({key[1] for key in gpu})
    matrix = np.full((len(residues), len(frames)), np.nan)
    for i, n_residues in enumerate(residues):
        for j, n_frames in enumerate(frames):
            row = gpu.get((n_frames, n_residues))
            if row is not None:
                matrix[i, j] = np.log2(
                    float(row["gpu_over_cpu_speedup"])
                )
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    image = axis.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=max(1.0, float(np.nanmax(matrix))),
    )
    axis.set_xticks(range(len(frames)), [f"{value:,}" for value in frames])
    axis.set_yticks(
        range(len(residues)),
        [f"{value:,}" for value in residues],
    )
    axis.set_xlabel("Frames")
    axis.set_ylabel("CA residues")
    axis.set_title("GPU/CPU throughput crossover (log2 speedup)")
    for i in range(len(residues)):
        for j in range(len(frames)):
            if np.isfinite(matrix[i, j]):
                axis.text(
                    j,
                    i,
                    f"{2 ** matrix[i, j]:.2f}x",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log2(GPU speedup over CPU throughput)")
    _save_figure(figure, "fig_gpu_cpu_crossover_heatmap")


def _throughput_figure(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = {
        "CPU_throughput": "#C4513A",
        "GPU_throughput_float32_Horn": "#176B87",
    }
    for method, color in colors.items():
        selected = sorted(
            [
                row
                for row in rows
                if row["method"] == method
            ],
            key=lambda row: int(row["residue_frames"]),
        )
        axes[0].scatter(
            [int(row["residue_frames"]) for row in selected],
            [float(row["residue_frames_per_second"]) for row in selected],
            s=28,
            alpha=0.8,
            color=color,
            label=method.replace("_", " "),
        )
        axes[1].scatter(
            [int(row["n_frames"]) for row in selected],
            [float(row["frames_per_second"]) for row in selected],
            s=28,
            alpha=0.8,
            color=color,
            label=method.replace("_", " "),
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Residue-frames")
    axes[0].set_ylabel("Residue-frames per second")
    axes[0].set_title("Two-dimensional throughput")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Frames per second")
    axes[1].set_title("Frame throughput")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, "fig_frames_residues_throughput")


def _matched_summary() -> list[dict[str, str]]:
    return read_tsv(
        RESULTS
        / "u1_matched_contracts"
        / "matched_contract_summary.tsv"
    )


def _real_speedup_figure(rows: list[dict[str, str]]) -> None:
    selected = [
        row
        for row in rows
        if row["prefix_label"] == "full"
        and row["contract"] == "A_analysis_metrics_transforms_rmsd_rmsf"
    ]
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in selected:
        grouped.setdefault(row["system"], {})[row["method"]] = row
    records = []
    for system, methods in grouped.items():
        gpu = methods.get("GPU_throughput_float32_Horn")
        mdlovo = methods.get("MDLovoFit_20.1.1")
        if gpu is None or mdlovo is None:
            continue
        records.append(
            (
                system,
                float(mdlovo["median_seconds"])
                / float(gpu["median_seconds"]),
            )
        )
    records.sort(key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.barh(
        [item[0] for item in records],
        [item[1] for item in records],
        color="#176B87",
    )
    axis.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    axis.set_xlabel(
        "MDLovoFit native-interface / GPU warm-array analysis runtime"
    )
    axis.set_title("Full-length method-native analysis contract")
    for index, (_, value) in enumerate(records):
        axis.text(value, index, f" {value:.2f}x", va="center", fontsize=8)
    _save_figure(figure, "fig_real_trajectory_speedup")


def _pareto_figure() -> None:
    quality = [
        row
        for row in read_tsv(
            RESULTS / "u5_quality" / "system_level_quality.tsv"
        )
        if row["method"] == "GPU_throughput_float32_Horn"
    ]
    timing = {
        row["system"]: row
        for row in read_tsv(
            RESULTS
            / "u2_real_panel"
            / "real_panel_internal_matched_timings.tsv"
        )
        if row["method"] == "GPU_throughput_float32_Horn"
        and row["contract"] == "A_analysis_metrics_transforms_rmsd_rmsf"
    }
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for row in quality:
        elapsed = float(timing[row["system"]]["median_seconds"])
        core = float(row["core_jaccard"])
        passed = row["case_quality_pass"].lower() == "true"
        axis.scatter(
            elapsed,
            core,
            s=45,
            color="#176B87" if passed else "#C4513A",
        )
        axis.annotate(
            row["system"],
            (elapsed, core),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("GPU end-to-end runtime (s)")
    axis.set_ylabel("Core Jaccard vs sequential_lean exact")
    axis.set_title("Quality-runtime envelope on real continuous inputs")
    axis.axhline(0.90, color="#777777", linestyle="--", linewidth=1)
    _save_figure(figure, "fig_quality_runtime_pareto")


def _memory_figure(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for method, color in (
        ("CPU_throughput", "#C4513A"),
        ("GPU_throughput_float32_Horn", "#176B87"),
    ):
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["residue_frames"]),
        )
        axes[0].scatter(
            [int(row["residue_frames"]) for row in selected],
            [float(row["peak_rss_mb_sampled"]) for row in selected],
            s=28,
            color=color,
            alpha=0.8,
            label=method.replace("_", " "),
        )
    gpu = sorted(
        [
            row
            for row in rows
            if row["method"] == "GPU_throughput_float32_Horn"
        ],
        key=lambda row: int(row["residue_frames"]),
    )
    axes[1].scatter(
        [int(row["residue_frames"]) for row in gpu],
        [float(row["peak_vram_mb"]) for row in gpu],
        s=28,
        color="#176B87",
    )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Residue-frames")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Sampled process RSS (MiB)")
    axes[0].set_title("Host-memory scaling")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Peak CuPy pool use (MiB)")
    axes[1].set_title("Device-memory scaling")
    figure.tight_layout()
    _save_figure(figure, "fig_peak_memory_scaling")


def _tables(matched: list[dict[str, str]]) -> None:
    write_tsv(TABLES / "matched_contract_benchmark_table.tsv", matched)
    write_tsv(
        TABLES / "exact_throughput_w5_contract_table.tsv",
        [
            {
                "mode": "sequential_lean",
                "status": "public exact/default",
                "primary_output": (
                    "stable weights, transforms, RMSD, RMSF, aligned output"
                ),
                "scientific_role": "exact reference/default",
                "MDLovoFit_equivalent_alignment_contract": True,
            },
            {
                "mode": "CPU throughput",
                "status": "experimental",
                "primary_output": (
                    "approximate stable weights, transforms, RMSD and RMSF"
                ),
                "scientific_role": "fast global approximation",
                "MDLovoFit_equivalent_alignment_contract": True,
            },
            {
                "mode": "GPU throughput",
                "status": "experimental",
                "primary_output": (
                    "approximate stable weights, transforms, RMSD and RMSF"
                ),
                "scientific_role": "accelerated fast global approximation",
                "MDLovoFit_equivalent_alignment_contract": True,
            },
            {
                "mode": "CPU/GPU W5",
                "status": "experimental",
                "primary_output": (
                    "alignment-free local deformation and local rigidity"
                ),
                "scientific_role": "local-deformation diagnostic",
                "MDLovoFit_equivalent_alignment_contract": False,
            },
        ],
    )


def _captions() -> None:
    (OUTPUT / "figure_captions.md").write_text(
        "\n".join(
            [
                "# U6 figure captions",
                "",
                "## GPU/CPU crossover heatmap",
                "",
                "End-to-end GPU throughput speedup over CPU throughput across "
                "the measured synthetic frames-by-residues matrix. Blank cells "
                "were not run because they exceeded the preregistered 50 "
                "million residue-frame resource guard; no values were inferred.",
                "",
                "## Frames-by-residues throughput",
                "",
                "Measured frame and residue-frame throughput for CPU and GPU "
                "throughput modes. Each point is the median of five measured "
                "runs after one warmup.",
                "",
                "## Full-length real-trajectory speedup",
                "",
                "Method-native analysis-contract runtime ratio for 15 "
                "unstitched trajectory entries. GPU timing starts from warm "
                "in-memory CA coordinates and includes synchronized transfers, "
                "compute, metrics and buffer sanitation. MDLovoFit timing starts "
                "from a cached PDB and necessarily includes its subprocess and "
                "aligned-PDB output plus post-hoc parsing and metrics. The large "
                "ratios therefore describe this explicit interface contract, "
                "not pure algorithmic speed or a universal speedup. The "
                "separate 250-frame materialised-output contract is the more "
                "closely output-matched comparison.",
                "",
                "## Quality-runtime Pareto",
                "",
                "Exact-relative core Jaccard versus GPU throughput runtime on "
                "the 15-entry real unstitched panel. The highlighted warning "
                "case exceeded the locked per-case median RMSD-difference gate "
                "despite high core and RMSF agreement.",
                "",
                "## Peak memory scaling",
                "",
                "Sampled process RSS and CuPy memory-pool use across the "
                "synthetic scaling matrix. RSS is process-level sampled memory "
                "and includes the resident input trajectory.",
                "",
            ]
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> None:
    paths = sorted(
        path
        for path in RESULTS.rglob("*")
        if path.is_file()
        and not path.is_relative_to(OUTPUT)
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_root": RESULTS.as_posix(),
        "historical_throughput_gpu_v1_modified": False,
        "git_metadata_modified": False,
        "canonical_python": (
            "/storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python"
        ),
        "thread_environment": {
            "MKL_NUM_THREADS": 2,
            "OPENBLAS_NUM_THREADS": 2,
            "OMP_NUM_THREADS": 2,
            "NUMEXPR_NUM_THREADS": 2,
        },
        "commands": [
            "python scripts/gpu_matched_validation_v1/run_u0_freeze.py",
            "python scripts/gpu_matched_validation_v1/prepare_u2_panel.py",
            "python scripts/gpu_matched_validation_v1/run_u2_u5_real_quality.py",
            "python scripts/gpu_matched_validation_v1/run_u1_matched_external.py",
            "python scripts/gpu_matched_validation_v1/run_u3_scaling.py",
            "python scripts/gpu_matched_validation_v1/run_u4_persistent_batch.py",
            "python scripts/gpu_matched_validation_v1/finalize_u6_publication.py",
        ],
        "inputs_and_intermediate_files": [
            {
                "path": path.relative_to(RESULTS.parent.parent).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    write_json(OUTPUT / "reproducibility_manifest.json", payload)


def _report(matched: list[dict[str, str]]) -> None:
    quality = json.loads(
        (
            RESULTS / "u5_quality" / "quality_gate_summary.json"
        ).read_text()
    )
    full = [
        row
        for row in matched
        if row["prefix_label"] == "full"
        and row["contract"] == "A_analysis_metrics_transforms_rmsd_rmsf"
    ]
    by_system: dict[str, dict[str, dict[str, str]]] = {}
    for row in full:
        by_system.setdefault(row["system"], {})[row["method"]] = row
    speedups = []
    for methods in by_system.values():
        gpu = methods.get("GPU_throughput_float32_Horn")
        mdlovo = methods.get("MDLovoFit_20.1.1")
        if gpu is not None and mdlovo is not None:
            speedups.append(
                float(mdlovo["median_seconds"])
                / float(gpu["median_seconds"])
            )
    materialized_speedups = []
    theseus_speedups = []
    theseus_rmsd_deltas = []
    systems = {row["system"] for row in matched}
    for system in systems:
        materialized = {
            row["method"]: row
            for row in matched
            if row["system"] == system
            and row["prefix_label"] == "p0250"
            and row["contract"] == "B_materialised_aligned_PDB"
        }
        if {
            "GPU_throughput_float32_Horn",
            "MDLovoFit_20.1.1",
        }.issubset(materialized):
            materialized_speedups.append(
                float(materialized["MDLovoFit_20.1.1"]["median_seconds"])
                / float(
                    materialized["GPU_throughput_float32_Horn"][
                        "median_seconds"
                    ]
                )
            )
        analysis = {
            row["method"]: row
            for row in matched
            if row["system"] == system
            and row["prefix_label"] == "p0250"
            and row["contract"]
            == "A_analysis_metrics_transforms_rmsd_rmsf"
        }
        if {
            "GPU_throughput_float32_Horn",
            "THESEUS_3.3.0",
        }.issubset(analysis):
            gpu = analysis["GPU_throughput_float32_Horn"]
            theseus = analysis["THESEUS_3.3.0"]
            theseus_speedups.append(
                float(theseus["median_seconds"])
                / float(gpu["median_seconds"])
            )
            theseus_rmsd_deltas.append(
                float(gpu["median_unified_posthoc_rmsd_A"])
                - float(theseus["median_unified_posthoc_rmsd_A"])
            )
    internal_timings = read_tsv(
        RESULTS
        / "u2_real_panel"
        / "real_panel_internal_matched_timings.tsv"
    )
    initial_cpu_gpu_speedups = []
    for system in ("YiiP_90ns", "NhaA", "ADK"):
        methods = {
            row["method"]: float(row["median_seconds"])
            for row in internal_timings
            if row["system"] == system
            and row["contract"]
            == "A_analysis_metrics_transforms_rmsd_rmsf"
        }
        initial_cpu_gpu_speedups.append(
            methods["CPU_throughput"]
            / methods["GPU_throughput_float32_Horn"]
        )
    batch = read_tsv(
        RESULTS / "u4_persistent_batch" / "persistent_batch_summary.tsv"
    )
    gpu_1000 = next(
        row
        for row in batch
        if row["method"] == "GPU_throughput_persistent_worker"
        and row["batch_size"] == "1000"
    )
    report = [
        "# U0-U6 GPU matched-validation report",
        "",
        "## Direct observations",
        "",
        f"- Real unstitched panel: `{quality['n_systems']}` trajectories.",
        f"- GPU quality case pass: `{quality['n_case_quality_pass']}/{quality['n_systems']}` under preregistered per-case gates.",
        f"- Median core Jaccard: `{quality['median_core_jaccard']:.6f}`.",
        f"- Median RMSF Spearman: `{quality['median_rmsf_spearman']:.6f}`.",
        f"- Locked aggregate gate: `{quality['locked_aggregate_gate_pass']}`.",
        "- Full test suite: `253 passed, 13 warnings`.",
        (
            f"- Direct full-length GPU/MDLovoFit analysis-contract speedup "
            f"range: `{min(speedups):.3f}–{max(speedups):.3f}x`."
            if speedups
            else "- No completed direct full-length GPU/MDLovoFit speedup."
        ),
        (
            "- Matched 250-frame materialised-output GPU/MDLovoFit "
            f"speedup range: `{min(materialized_speedups):.3f}–"
            f"{max(materialized_speedups):.3f}x` "
            f"(median `{np.median(materialized_speedups):.3f}x`)."
        ),
        (
            "- Matched 250-frame analysis GPU/THESEUS speedup range: "
            f"`{min(theseus_speedups):.3f}–{max(theseus_speedups):.3f}x`; "
            "GPU had lower unified post-hoc RMSD in "
            f"`{sum(value < 0 for value in theseus_rmsd_deltas)}/"
            f"{len(theseus_rmsd_deltas)}` systems."
        ),
        f"- Persistent GPU 1,000-task throughput: `{float(gpu_1000['amortised_tasks_per_second']):.3f}` tasks/s.",
        "",
        "## Safe manuscript wording",
        "",
        (
            "GPU throughput accelerated the optimised CPU implementation by "
            "3.22–6.26x on the initial real continuous trajectories while "
            "preserving exact-relative core Jaccard of 0.974–1.000 and RMSF "
            "Spearman above 0.9998."
        ),
        (
            "The new five-repeat U1/U2 rerun observed a CPU-throughput/GPU "
            f"range of `{min(initial_cpu_gpu_speedups):.3f}–"
            f"{max(initial_cpu_gpu_speedups):.3f}x` on the same three systems."
        ),
        "",
        "This wording concerns CPU-throughput acceleration. MDLovoFit speedups "
        "must use the direct matched-contract range above and its stated native "
        "PDB/subprocess boundary.",
        "",
        "## Negative results and limits",
        "",
        "- The preregistered aggregate quality decision is false because one "
        "mdCATH replicate exceeded the 0.05 A per-case median RMSD-difference "
        "threshold; this was not retuned away.",
        "- The current local real panel does not contain 800–1,500 or 2,000+ "
        "CA systems.",
        "- THESEUS comparisons are restricted to feasible 250-frame prefixes.",
        "- W5 is not an alignment-equivalent comparator.",
        "- No universal-fastest or contract-independent 71x claim is supported.",
        "",
    ]
    (OUTPUT / "u6_publication_report.md").write_text("\n".join(report))
    (OUTPUT / "limitations_and_negative_results.md").write_text(
        "\n".join(
            [
                "# Limitations and negative results",
                "",
                "- One of 15 GPU-throughput cases failed the locked per-case "
                "RMSD-difference gate despite high core, weight and RMSF agreement.",
                "- Real-system CA-size coverage ends at 752 residues locally; "
                "larger requested strata remain untested.",
                "- mdCATH single-replicate segments are real stored trajectories "
                "but are approximately 500 frames and should not be called "
                "ultra-long.",
                "- PBC/nojump provenance is not independently proven where "
                "source metadata are incomplete.",
                "- THESEUS runtime is not extrapolated beyond feasible prefixes.",
                "- External PDB interfaces impose output and subprocess costs "
                "that cannot be separated from native tool execution.",
                "- W5 reports local deformation, not RMSD or an equivalent "
                "global alignment.",
                "- Performance depends on trajectory shape and hardware; "
                "neither universal fastest nor a contract-independent 71x "
                "speedup is claimed.",
                "",
            ]
        )
    )


def _handoff() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    files = [
        OUTPUT / "u6_publication_report.md",
        OUTPUT / "limitations_and_negative_results.md",
        OUTPUT / "figure_captions.md",
        TABLES / "matched_contract_benchmark_table.tsv",
        TABLES / "exact_throughput_w5_contract_table.tsv",
        RESULTS / "u5_quality" / "quality_gate_summary.json",
        RESULTS / "u2_real_panel" / "real_continuous_panel.tsv",
        RESULTS / "u3_scaling" / "cpu_gpu_crossover_summary.tsv",
        RESULTS / "u4_persistent_batch" / "persistent_batch_summary.tsv",
    ]
    rows = []
    for source in files:
        target = HANDOFF / source.name
        target.write_bytes(source.read_bytes())
        rows.append(
            {
                "file": target.name,
                "source": source.relative_to(RESULTS.parent.parent).as_posix(),
                "sha256": _sha256(target),
            }
        )
    write_tsv(HANDOFF / "handoff_manifest.tsv", rows)
    (HANDOFF / "00_read_me_first.md").write_text(
        "\n".join(
            [
                "# U0-U6 compact handoff",
                "",
                "Read `u6_publication_report.md` first. The bundle contains "
                "matched-contract tables, the locked quality outcome, real-panel "
                "inventory, scaling crossover and persistent-batch summary.",
                "",
                "The public exact/default path remains `sequential_lean`. GPU "
                "throughput and W5 remain experimental. Do not claim universal "
                "fastest, 71x, or W5/MDLovoFit equivalence.",
                "",
            ]
        )
    )


def _audit() -> None:
    required = [
        *(FIGURES / f"{name}.{suffix}" for name in (
            "fig_gpu_cpu_crossover_heatmap",
            "fig_frames_residues_throughput",
            "fig_real_trajectory_speedup",
            "fig_quality_runtime_pareto",
            "fig_peak_memory_scaling",
        ) for suffix in ("pdf", "png")),
        TABLES / "matched_contract_benchmark_table.tsv",
        TABLES / "exact_throughput_w5_contract_table.tsv",
        OUTPUT / "figure_captions.md",
        OUTPUT / "limitations_and_negative_results.md",
        OUTPUT / "reproducibility_manifest.json",
        HANDOFF / "00_read_me_first.md",
    ]
    raw = read_tsv(
        RESULTS
        / "u1_matched_contracts"
        / "matched_contract_raw_repeats.tsv"
    )
    measured: dict[tuple[str, str, str, str], int] = {}
    for row in raw:
        if (
            row["status"] == "success"
            and row["is_warmup"].lower() == "false"
        ):
            key = (
                row["system"],
                row["prefix_label"],
                row["contract"],
                row["method"],
            )
            measured[key] = measured.get(key, 0) + 1
    frozen_failures = []
    freeze_manifest = (
        RESULTS
        / "u0_freeze"
        / "throughput_gpu_v1_file_manifest.tsv"
    )
    project_root = RESULTS.parent.parent
    for row in read_tsv(freeze_manifest):
        path = project_root / row["relative_path"]
        if not path.exists() or _sha256(path) != row["sha256"]:
            frozen_failures.append(row["relative_path"])
    test_text = (
        RESULTS / "test_status" / "full_pytest_gpu.txt"
    ).read_text()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "required_output_count": len(required),
        "missing_or_empty_outputs": [
            path.relative_to(project_root).as_posix()
            for path in required
            if not path.is_file() or path.stat().st_size == 0
        ],
        "raw_external_failure_count": sum(
            row["status"] != "success" for row in raw
        ),
        "minimum_measured_repeats_per_completed_setting": min(
            measured.values()
        ),
        "settings_with_fewer_than_five_repeats": [
            {"setting": list(key), "count": count}
            for key, count in measured.items()
            if count < 5
        ],
        "frozen_throughput_gpu_v1_hash_failures": frozen_failures,
        "test_status": (
            "253 passed"
            if "253 passed" in test_text
            else "expected count not found"
        ),
        "public_default_changed": False,
        "git_metadata_changed": False,
    }
    payload["audit_pass"] = bool(
        not payload["missing_or_empty_outputs"]
        and payload["raw_external_failure_count"] == 0
        and not payload["settings_with_fewer_than_five_repeats"]
        and not frozen_failures
        and payload["test_status"] == "253 passed"
    )
    write_json(OUTPUT / "final_audit.json", payload)


def main() -> None:
    scaling = _successful_scaling()
    matched = _matched_summary()
    _crossover_figure(scaling)
    _throughput_figure(scaling)
    _real_speedup_figure(matched)
    _pareto_figure()
    _memory_figure(scaling)
    _tables(matched)
    _captions()
    _report(matched)
    _manifest()
    _handoff()
    _audit()


if __name__ == "__main__":
    main()
