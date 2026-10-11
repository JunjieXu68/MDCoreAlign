"""Characterise fixed-budget throughput fidelity and empirical complexity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from common import (
    REPO_ROOT,
    RESULT_ROOT,
    ca_atom_metadata,
    core_mask_exact_fraction,
    read_tsv,
    resolve_mdcath_h5,
    write_json,
    write_tsv,
)
from scipy.stats import spearmanr

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment

FIELDS = [
    "source_panel", "split", "domain_id", "configuration", "K", "M", "n_frames",
    "n_residues", "status", "error", "runtime_seconds", "weight_estimation_seconds",
    "all_frame_fit_seconds", "peak_RSS_MB", "core_Jaccard_vs_sequential",
    "weight_Spearman_vs_sequential", "RMSD_MAE_A_vs_sequential",
    "RMSD_P95_abs_difference_A", "RMSF_MAE_A_vs_sequential",
    "RMSF_Spearman_vs_sequential", "tail_flag",
]


def load_domain(domain_id: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(resolve_mdcath_h5(domain_id), "r") as h5:
        metadata = ca_atom_metadata(h5, domain_id)
        coords = np.asarray(
            h5[domain_id]["320"]["0"]["coords"][:, metadata["atom_indices"], :],
            dtype=np.float64,
        )
    return coords, np.asarray(metadata["chain_ids"])


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def evaluate_domain(
    domain_id: str,
    split: str,
    configurations: list[dict[str, object]],
) -> list[dict[str, object]]:
    try:
        coords, chains = load_domain(domain_id)
        exact = solve_exact_contract(
            coords,
            chain_ids=chains,
            representative_count=2000,
            store_aligned=False,
            compute_ordinary_rmsf=False,
        )
        exact_core = core_mask_exact_fraction(exact.frozen_weights, 0.60)
    except Exception as error:
        return [
            {
                **{field: "" for field in FIELDS},
                "source_panel": "mdCATH_domain_canonical",
                "split": split,
                "domain_id": domain_id,
                "configuration": config["name"],
                "K": config["K"],
                "M": config["M"],
                "status": "failed_exact_reference",
                "error": f"{type(error).__name__}: {error}",
            }
            for config in configurations
        ]
    output = []
    for config in configurations:
        try:
            result = solve_fast_global_alignment(
                coords,
                chain_ids=chains,
                representative_count=int(config["K"]),
                update_rounds=int(config["M"]),
                store_aligned=False,
                compute_confidence=False,
                collect_risk_signals=False,
                compute_ordinary_rmsf=False,
            )
            core = core_mask_exact_fraction(result.frozen_weights, 0.60)
            rmsd_difference = np.abs(np.asarray(result.rmsd) - exact.rmsd)
            rmsf_difference = np.abs(np.asarray(result.robust_rmsf) - exact.robust_rmsf)
            core_value = jaccard(core, exact_core)
            weight_value = float(spearmanr(result.frozen_weights, exact.frozen_weights).statistic)
            rmsf_value = float(spearmanr(result.robust_rmsf, exact.robust_rmsf).statistic)
            output.append(
                {
                    "source_panel": "mdCATH_domain_canonical",
                    "split": split,
                    "domain_id": domain_id,
                    "configuration": config["name"],
                    "K": config["K"],
                    "M": config["M"],
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "status": "success",
                    "error": "",
                    "runtime_seconds": result.total_runtime_seconds,
                    "weight_estimation_seconds": result.timing["limited_global_refinement"],
                    "all_frame_fit_seconds": result.timing["exact_all_frame_total"],
                    "peak_RSS_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                    "core_Jaccard_vs_sequential": core_value,
                    "weight_Spearman_vs_sequential": weight_value,
                    "RMSD_MAE_A_vs_sequential": float(np.mean(rmsd_difference)),
                    "RMSD_P95_abs_difference_A": float(np.quantile(rmsd_difference, 0.95)),
                    "RMSF_MAE_A_vs_sequential": float(np.mean(rmsf_difference)),
                    "RMSF_Spearman_vs_sequential": rmsf_value,
                    "tail_flag": bool(core_value < 0.80 or rmsf_value < 0.95 or not np.isfinite(weight_value)),
                }
            )
        except Exception as error:
            failed = {field: "" for field in FIELDS}
            failed.update(
                {
                    "source_panel": "mdCATH_domain_canonical",
                    "split": split,
                    "domain_id": domain_id,
                    "configuration": config["name"],
                    "K": config["K"],
                    "M": config["M"],
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            output.append(failed)
    return output


def run_panel(
    domains: list[str],
    split: str,
    configurations: list[dict[str, object]],
    output_path: Path,
    workers: int,
) -> list[dict[str, object]]:
    existing: list[dict[str, str]] = []
    completed: set[str] = set()
    if output_path.exists():
        with output_path.open() as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
        counts: dict[str, int] = {}
        for row in existing:
            counts[row["domain_id"]] = counts.get(row["domain_id"], 0) + 1
        completed = {domain for domain, count in counts.items() if count == len(configurations)}
    pending = [domain for domain in domains if domain not in completed]
    rows: list[dict[str, object]] = list(existing)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_domain, domain, split, configurations): domain
            for domain in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 5 == 0 or index == len(pending):
                write_tsv(output_path, rows, FIELDS)
                print(f"SE8 {split} {len(completed) + index}/{len(domains)}", flush=True)
    rows.sort(key=lambda row: (str(row["domain_id"]), str(row["configuration"])))
    write_tsv(output_path, rows, FIELDS)
    return rows


def pareto_summary(rows: list[dict[str, object]]) -> None:
    successful = [row for row in rows if row["status"] == "success"]
    configs = sorted({str(row["configuration"]) for row in successful})
    summary = []
    for config in configs:
        group = [row for row in successful if row["configuration"] == config]
        summary.append(
            {
                "configuration": config,
                "K": group[0]["K"],
                "M": group[0]["M"],
                "n_domains": len(group),
                "median_runtime_seconds": float(np.median([float(row["runtime_seconds"]) for row in group])),
                "median_core_Jaccard": float(np.median([float(row["core_Jaccard_vs_sequential"]) for row in group])),
                "median_weight_Spearman": float(np.median([float(row["weight_Spearman_vs_sequential"]) for row in group])),
                "median_RMSF_Spearman": float(np.median([float(row["RMSF_Spearman_vs_sequential"]) for row in group])),
                "median_RMSF_MAE_A": float(np.median([float(row["RMSF_MAE_A_vs_sequential"]) for row in group])),
                "tail_rate": float(np.mean([str(row["tail_flag"]) == "True" or row["tail_flag"] is True for row in group])),
            }
        )
    for row in summary:
        dominated = False
        for other in summary:
            if other is row:
                continue
            if (
                other["median_runtime_seconds"] <= row["median_runtime_seconds"]
                and other["median_core_Jaccard"] >= row["median_core_Jaccard"]
                and other["median_RMSF_Spearman"] >= row["median_RMSF_Spearman"]
                and (
                    other["median_runtime_seconds"] < row["median_runtime_seconds"]
                    or other["median_core_Jaccard"] > row["median_core_Jaccard"]
                    or other["median_RMSF_Spearman"] > row["median_RMSF_Spearman"]
                )
            ):
                dominated = True
                break
        row["pareto_nondominated"] = not dominated
    write_tsv(
        RESULT_ROOT / "08_budget_fidelity/pareto_results.tsv",
        summary,
        list(summary[0]),
    )


def scaling_study() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_path = (
        REPO_ROOT
        / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/runs"
        / "4oelA02/100000/transition_aligned_coords.npz"
    )
    payload = np.load(source_path, allow_pickle=True)
    long_coords = np.asarray(payload["coords"], dtype=np.float64)
    frame_rows = []
    for n_frames in (250, 1000, 5000, 10000, 50000, 100000):
        coords = long_coords[:n_frames]
        solve_fast_global_alignment(
            coords, representative_count=256, update_rounds=4,
            store_aligned=False, compute_confidence=False,
            collect_risk_signals=False, compute_ordinary_rmsf=False,
        )
        repeats = []
        for _ in range(5):
            start = time.perf_counter()
            result = solve_fast_global_alignment(
                coords, representative_count=256, update_rounds=4,
                store_aligned=False, compute_confidence=False,
                collect_risk_signals=False, compute_ordinary_rmsf=False,
            )
            repeats.append(
                (
                    time.perf_counter() - start,
                    result.timing["limited_global_refinement"],
                    result.timing["exact_all_frame_total"],
                )
            )
        timings = np.asarray(repeats, dtype=np.float64)
        frame_rows.append(
            {
                "scaling_axis": "frames",
                "source": "transition_aligned_synthetic_scaling_control_4oelA02",
                "T": n_frames,
                "N": coords.shape[1],
                "K": 256,
                "M": 4,
                "weight_estimation_seconds": float(np.median(timings[:, 1])),
                "all_frame_fit_seconds": float(np.median(timings[:, 2])),
                "total_runtime_seconds": float(np.median(timings[:, 0])),
                "total_runtime_IQR_low_seconds": float(np.quantile(timings[:, 0], 0.25)),
                "total_runtime_IQR_high_seconds": float(np.quantile(timings[:, 0], 0.75)),
                "timed_repeats": 5,
                "biological_continuity_claimed": False,
            }
        )
    atlas_row = next(
        row
        for row in read_tsv(REPO_ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")
        if row["system"] == "6sup_A"
    )
    atlas = np.load(REPO_ROOT / atlas_row["coordinate_cache_path"], mmap_mode="r")
    residue_rows = []
    for n_residues in (75, 150, 300, 500):
        coords = np.asarray(atlas[:10000, :n_residues], dtype=np.float64)
        solve_fast_global_alignment(
            coords, representative_count=256, update_rounds=4,
            store_aligned=False, compute_confidence=False,
            collect_risk_signals=False, compute_ordinary_rmsf=False,
        )
        repeats = []
        for _ in range(5):
            start = time.perf_counter()
            result = solve_fast_global_alignment(
                coords, representative_count=256, update_rounds=4,
                store_aligned=False, compute_confidence=False,
                collect_risk_signals=False, compute_ordinary_rmsf=False,
            )
            repeats.append(
                (
                    time.perf_counter() - start,
                    result.timing["limited_global_refinement"],
                    result.timing["exact_all_frame_total"],
                )
            )
        timings = np.asarray(repeats, dtype=np.float64)
        residue_rows.append(
            {
                "scaling_axis": "residues",
                "source": "ATLAS_6sup_A_prefix_timing_control",
                "T": 10000,
                "N": n_residues,
                "K": 256,
                "M": 4,
                "weight_estimation_seconds": float(np.median(timings[:, 1])),
                "all_frame_fit_seconds": float(np.median(timings[:, 2])),
                "total_runtime_seconds": float(np.median(timings[:, 0])),
                "total_runtime_IQR_low_seconds": float(np.quantile(timings[:, 0], 0.25)),
                "total_runtime_IQR_high_seconds": float(np.quantile(timings[:, 0], 0.75)),
                "timed_repeats": 5,
                "biological_continuity_claimed": True,
            }
        )
    return frame_rows + residue_rows, frame_rows


def complexity_summary(scaling: list[dict[str, object]]) -> None:
    output = []
    for axis, independent in (("frames", "T"), ("residues", "N")):
        rows = [row for row in scaling if row["scaling_axis"] == axis]
        x = np.log(np.asarray([float(row[independent]) for row in rows]))
        for metric in ("weight_estimation_seconds", "all_frame_fit_seconds", "total_runtime_seconds"):
            y = np.log(np.maximum([float(row[metric]) for row in rows], 1e-12))
            slope, intercept = np.polyfit(x, y, 1)
            prediction = slope * x + intercept
            r2 = 1.0 - float(np.sum((y - prediction) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12))
            output.append(
                {
                    "scaling_axis": axis,
                    "independent_variable": independent,
                    "runtime_component": metric,
                    "log_log_slope": float(slope),
                    "log_intercept": float(intercept),
                    "R_squared": r2,
                    "n_points": len(rows),
                }
            )
    write_tsv(
        RESULT_ROOT / "08_budget_fidelity/empirical_complexity.tsv",
        output,
        list(output[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-development", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--skip-scaling", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    registry = json.loads(
        (RESULT_ROOT / "02_preregistration/budget_fidelity_registry.json").read_text()
    )
    grid = [
        {"name": f"K{K}_M{M}", "K": K, "M": M}
        for K in registry["K_values"]
        for M in registry["M_values"]
    ]
    write_tsv(
        RESULT_ROOT / "08_budget_fidelity/budget_grid.tsv",
        grid,
        ["name", "K", "M"],
    )
    locked = list(registry["locked_validation_configs"])
    write_tsv(
        RESULT_ROOT / "08_budget_fidelity/locked_configs.tsv",
        locked,
        ["name", "K", "M"],
    )
    (RESULT_ROOT / "08_budget_fidelity/budget_parameter_audit.md").write_text(
        "# Throughput budget parameter audit\n\n"
        "- `K` maps to `representative_count` in "
        "`src/mdcorealign/experimental.py::solve_fast_global_alignment`; the "
        "actual count is `min(K, T)` after deterministic temporal stratification.\n"
        "- `M` maps to the fixed `update_rounds` argument passed into the batched "
        "global refinement.\n"
        "- `N` is the number of selected C-alpha residues and `T` is the number "
        "of decoded frames.\n"
        "- Weight estimation is approximately O(K M N), while the explicit "
        "fixed-weight fit and compact statistics remain O(T N). The complete "
        "workflow is therefore not independent of trajectory length.\n"
        "- Timings use loaded float64 coordinates, one CPU thread, no aligned "
        "coordinate materialisation, one warmup and five repeats for scaling "
        "controls. `peak_RSS_MB` in parallel panel tables is a worker-process "
        "high-water mark, not an isolated per-configuration allocation.\n",
        encoding="utf-8",
    )
    development = list(registry["development_domains"])
    validation = list(registry["validation_domains"])
    if args.limit_development is not None:
        development = development[: args.limit_development]
    if args.limit_validation is not None:
        validation = validation[: args.limit_validation]
    development_rows = run_panel(
        development,
        "development",
        grid,
        RESULT_ROOT / "08_budget_fidelity/development_results.tsv",
        args.workers,
    )
    validation_rows = run_panel(
        validation,
        "validation",
        locked,
        RESULT_ROOT / "08_budget_fidelity/validation_results.tsv",
        args.workers,
    )
    pareto_summary(development_rows)
    if not args.skip_scaling and args.limit_development is None and args.limit_validation is None:
        scaling, _ = scaling_study()
        write_tsv(
            RESULT_ROOT / "08_budget_fidelity/scaling_results.tsv",
            scaling,
            list(scaling[0]),
        )
        complexity_summary(scaling)
    success_validation = [row for row in validation_rows if row["status"] == "success"]
    (RESULT_ROOT / "08_budget_fidelity/budget_fidelity_report.md").write_text(
        "# Fixed-budget fidelity and empirical complexity\n\n"
        f"The development grid contains `{len(grid)}` K/M combinations on "
        f"`{len(development)}` domains. Locked validation contains `{len(locked)}` "
        f"configurations on `{len(validation)}` domains, yielding "
        f"`{len(success_validation)}` successful rows. Sequential float64 remains "
        "the numerical reference; no candidate changes the production default.\n\n"
        "The implementation separates approximately O(K M N) bounded weight "
        "estimation from O(T N) explicit all-frame fitting and statistics. The full "
        "workflow is not trajectory-length independent. Scaling controls and fitted "
        "log-log slopes are reported in `empirical_complexity.tsv`; transition-aligned "
        "controls are computational stress data and do not claim biological continuity.\n",
        encoding="utf-8",
    )
    write_json(
        RESULT_ROOT / "08_budget_fidelity/run_summary.json",
        {
            "development_domains": len(development),
            "validation_domains": len(validation),
            "development_rows": len(development_rows),
            "validation_rows": len(validation_rows),
            "development_failures": sum(row["status"] != "success" for row in development_rows),
            "validation_failures": sum(row["status"] != "success" for row in validation_rows),
        },
    )


if __name__ == "__main__":
    main()
