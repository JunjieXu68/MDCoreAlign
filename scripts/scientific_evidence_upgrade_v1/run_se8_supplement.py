"""Complete the preregistered synthetic and ATLAS SE8 fidelity panels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from common import REPO_ROOT, RESULT_ROOT, core_mask_exact_fraction, read_tsv, write_json, write_tsv
from run_se3_se4 import build_settings, load_template
from scipy.stats import spearmanr
from synthetic_truth import (
    aligned_from_transform,
    binary_mask_metrics,
    binary_ranking_metrics,
    gauge_to_reference,
    generate_synthetic_trajectory,
    infer_transforms,
    rotation_geodesic_degrees,
    top_cardinality_mask,
)

from mdcorealign.contracts import solve_exact_contract
from mdcorealign.experimental import solve_fast_global_alignment

SYNTHETIC_FIELDS = [
    "split", "case_id", "template_id", "scenario", "seed", "local_amplitude_A",
    "hinge_amplitude_deg", "configuration", "K", "M", "n_frames", "n_residues",
    "status", "error", "runtime_seconds", "weight_estimation_seconds",
    "all_frame_fit_seconds", "worker_peak_RSS_MB", "core_Jaccard_vs_sequential",
    "weight_Spearman_vs_sequential", "RMSD_MAE_A_vs_sequential",
    "RMSD_P95_abs_difference_A", "RMSF_MAE_A_vs_sequential",
    "RMSF_Spearman_vs_sequential", "core_Jaccard_vs_truth", "core_AUPRC_vs_truth",
    "rotation_error_median_deg", "rotation_error_P95_deg",
    "core_coordinate_error_median_A", "core_coordinate_error_P95_A", "tail_flag",
]

ATLAS_FIELDS = [
    "split", "system_id", "configuration", "K", "M", "n_frames", "n_residues",
    "status", "error", "runtime_seconds", "weight_estimation_seconds",
    "all_frame_fit_seconds", "worker_peak_RSS_MB", "core_Jaccard_vs_sequential",
    "weight_Spearman_vs_sequential", "RMSD_MAE_A_vs_sequential",
    "RMSD_P95_abs_difference_A", "RMSF_MAE_A_vs_sequential",
    "RMSF_Spearman_vs_sequential", "tail_flag",
]


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 1.0 if np.allclose(left, right) else float("nan")
    return float(spearmanr(left, right).statistic)


def failure_row(fields: list[str], base: dict[str, object], error: Exception) -> dict[str, object]:
    row = {field: "" for field in fields}
    row.update(base)
    row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    return row


def synthetic_case(
    setting: dict[str, object], configurations: list[dict[str, object]],
) -> list[dict[str, object]]:
    base = {
        "split": setting["template"]["split"],
        "case_id": setting["case_id"],
        "template_id": setting["template"]["domain_id"],
        "scenario": setting["scenario"],
        "seed": setting["seed"],
        "local_amplitude_A": setting["local_amplitude_A"],
        "hinge_amplitude_deg": setting["hinge_amplitude_deg"],
    }
    try:
        reference, metadata, _ = load_template(setting["template"])
        trajectory = generate_synthetic_trajectory(
            reference,
            scenario=str(setting["scenario"]),
            n_frames=int(setting["n_frames"]),
            seed=int(setting["seed"]),
            local_amplitude_A=float(setting["local_amplitude_A"]),
            hinge_amplitude_deg=float(setting["hinge_amplitude_deg"]),
        )
        exact = solve_exact_contract(
            trajectory.observed,
            chain_ids=np.asarray(metadata["chain_ids"]),
            representative_count=2000,
            store_aligned=False,
            compute_ordinary_rmsf=False,
        )
        exact_core = core_mask_exact_fraction(exact.frozen_weights, 0.60)
    except Exception as error:
        return [
            failure_row(
                SYNTHETIC_FIELDS,
                {**base, "configuration": config["name"], "K": config["K"], "M": config["M"]},
                error,
            )
            for config in configurations
        ]

    rows = []
    for config in configurations:
        config_base = {**base, "configuration": config["name"], "K": config["K"], "M": config["M"]}
        try:
            result = solve_fast_global_alignment(
                trajectory.observed,
                chain_ids=np.asarray(metadata["chain_ids"]),
                representative_count=int(config["K"]),
                update_rounds=int(config["M"]),
                store_aligned=False,
                compute_confidence=False,
                collect_risk_signals=False,
                compute_ordinary_rmsf=False,
            )
            core = core_mask_exact_fraction(result.frozen_weights, 0.60)
            rmsd_delta = np.abs(np.asarray(result.rmsd) - np.asarray(exact.rmsd))
            rmsf_delta = np.abs(np.asarray(result.robust_rmsf) - np.asarray(exact.robust_rmsf))
            aligned = gauge_to_reference(
                aligned_from_transform(trajectory.observed, result.rotations, result.translations),
                trajectory.reference,
            )
            rotations, _ = infer_transforms(trajectory.observed, aligned)
            rotation_error = rotation_geodesic_degrees(rotations, trajectory.true_rotations)
            truth_mask = trajectory.true_stable_mask
            evaluation_mask = truth_mask if truth_mask is not None else np.ones(core.size, dtype=bool)
            coordinate_error = np.sqrt(
                np.mean(np.sum((aligned[:, evaluation_mask] - trajectory.internal[:, evaluation_mask]) ** 2, axis=2), axis=1)
            )
            truth_jaccard: float | str = ""
            truth_auprc: float | str = ""
            if truth_mask is not None and 0 < np.count_nonzero(truth_mask) < truth_mask.size:
                truth_count = int(np.count_nonzero(truth_mask))
                truth_rank_mask = top_cardinality_mask(result.frozen_weights, truth_count)
                truth_jaccard = binary_mask_metrics(truth_rank_mask, truth_mask)["Jaccard"]
                truth_auprc = binary_ranking_metrics(result.frozen_weights, truth_mask)["AUPRC"]
            core_value = jaccard(core, exact_core)
            rmsf_spearman = safe_spearman(result.robust_rmsf, exact.robust_rmsf)
            rows.append(
                {
                    **config_base,
                    "n_frames": trajectory.observed.shape[0],
                    "n_residues": trajectory.observed.shape[1],
                    "status": "success",
                    "error": "",
                    "runtime_seconds": result.total_runtime_seconds,
                    "weight_estimation_seconds": result.timing["limited_global_refinement"],
                    "all_frame_fit_seconds": result.timing["exact_all_frame_total"],
                    "worker_peak_RSS_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                    "core_Jaccard_vs_sequential": core_value,
                    "weight_Spearman_vs_sequential": safe_spearman(result.frozen_weights, exact.frozen_weights),
                    "RMSD_MAE_A_vs_sequential": float(np.mean(rmsd_delta)),
                    "RMSD_P95_abs_difference_A": float(np.quantile(rmsd_delta, 0.95)),
                    "RMSF_MAE_A_vs_sequential": float(np.mean(rmsf_delta)),
                    "RMSF_Spearman_vs_sequential": rmsf_spearman,
                    "core_Jaccard_vs_truth": truth_jaccard,
                    "core_AUPRC_vs_truth": truth_auprc,
                    "rotation_error_median_deg": float(np.median(rotation_error)),
                    "rotation_error_P95_deg": float(np.quantile(rotation_error, 0.95)),
                    "core_coordinate_error_median_A": float(np.median(coordinate_error)),
                    "core_coordinate_error_P95_A": float(np.quantile(coordinate_error, 0.95)),
                    "tail_flag": bool(core_value < 0.80 or rmsf_spearman < 0.95),
                }
            )
        except Exception as error:
            rows.append(failure_row(SYNTHETIC_FIELDS, config_base, error))
    return rows


def atlas_input(system_id: str) -> tuple[np.ndarray, np.ndarray]:
    manifest = read_tsv(REPO_ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")
    row = next(item for item in manifest if item["system"] == system_id)
    coords = np.asarray(np.load(REPO_ROOT / row["coordinate_cache_path"], mmap_mode="r"), dtype=np.float64)
    metadata = np.load(REPO_ROOT / row["residue_metadata_path"])
    return coords, np.asarray(metadata["chain_ids"])


def atlas_case(system_id: str, configurations: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        coords, chains = atlas_input(system_id)
        exact = solve_exact_contract(
            coords, chain_ids=chains, representative_count=2000,
            store_aligned=False, compute_ordinary_rmsf=False,
        )
        exact_core = core_mask_exact_fraction(exact.frozen_weights, 0.60)
    except Exception as error:
        return [
            failure_row(
                ATLAS_FIELDS,
                {"split": "locked_ATLAS_R1", "system_id": system_id,
                 "configuration": config["name"], "K": config["K"], "M": config["M"]},
                error,
            )
            for config in configurations
        ]
    rows = []
    for config in configurations:
        base = {
            "split": "locked_ATLAS_R1", "system_id": system_id,
            "configuration": config["name"], "K": config["K"], "M": config["M"],
        }
        try:
            result = solve_fast_global_alignment(
                coords, chain_ids=chains, representative_count=int(config["K"]),
                update_rounds=int(config["M"]), store_aligned=False,
                compute_confidence=False, collect_risk_signals=False,
                compute_ordinary_rmsf=False,
            )
            core = core_mask_exact_fraction(result.frozen_weights, 0.60)
            rmsd_delta = np.abs(np.asarray(result.rmsd) - np.asarray(exact.rmsd))
            rmsf_delta = np.abs(np.asarray(result.robust_rmsf) - np.asarray(exact.robust_rmsf))
            core_value = jaccard(core, exact_core)
            rmsf_spearman = safe_spearman(result.robust_rmsf, exact.robust_rmsf)
            rows.append(
                {
                    **base, "n_frames": coords.shape[0], "n_residues": coords.shape[1],
                    "status": "success", "error": "",
                    "runtime_seconds": result.total_runtime_seconds,
                    "weight_estimation_seconds": result.timing["limited_global_refinement"],
                    "all_frame_fit_seconds": result.timing["exact_all_frame_total"],
                    "worker_peak_RSS_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                    "core_Jaccard_vs_sequential": core_value,
                    "weight_Spearman_vs_sequential": safe_spearman(result.frozen_weights, exact.frozen_weights),
                    "RMSD_MAE_A_vs_sequential": float(np.mean(rmsd_delta)),
                    "RMSD_P95_abs_difference_A": float(np.quantile(rmsd_delta, 0.95)),
                    "RMSF_MAE_A_vs_sequential": float(np.mean(rmsf_delta)),
                    "RMSF_Spearman_vs_sequential": rmsf_spearman,
                    "tail_flag": bool(core_value < 0.80 or rmsf_spearman < 0.95),
                }
            )
        except Exception as error:
            rows.append(failure_row(ATLAS_FIELDS, base, error))
    return rows


def run_parallel(
    items: list[object], worker, configurations: list[dict[str, object]],
    output_path: Path, fields: list[str], workers: int, key_name: str,
) -> list[dict[str, object]]:
    existing: list[dict[str, str]] = []
    completed: set[str] = set()
    if output_path.exists():
        with output_path.open() as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
        counts: dict[str, int] = {}
        for row in existing:
            counts[row[key_name]] = counts.get(row[key_name], 0) + 1
        completed = {key for key, count in counts.items() if count == len(configurations)}
    pending = [item for item in items if str(item["case_id"] if key_name == "case_id" else item) not in completed]
    rows: list[dict[str, object]] = list(existing)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, item, configurations): item for item in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 5 == 0 or index == len(pending):
                write_tsv(output_path, rows, fields)
                print(f"SE8 supplement {output_path.stem} {len(completed) + index}/{len(items)}", flush=True)
    rows.sort(key=lambda row: (str(row[key_name]), str(row["configuration"])))
    write_tsv(output_path, rows, fields)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-synthetic-development", type=int)
    parser.add_argument("--limit-synthetic-validation", type=int)
    parser.add_argument("--limit-atlas", type=int)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    budget = json.loads((RESULT_ROOT / "02_preregistration/budget_fidelity_registry.json").read_text())
    synthetic_registry = json.loads((RESULT_ROOT / "02_preregistration/synthetic_truth_registry.json").read_text())
    grid = [{"name": f"K{K}_M{M}", "K": K, "M": M} for K in budget["K_values"] for M in budget["M_values"]]
    locked = list(budget["locked_validation_configs"])
    all_settings = build_settings(synthetic_registry)
    development_ids = set(budget["synthetic_development_templates"])
    validation_ids = set(budget["synthetic_validation_templates"])
    development = [item for item in all_settings if item["template"]["domain_id"] in development_ids]
    validation = [item for item in all_settings if item["template"]["domain_id"] in validation_ids]
    if args.limit_synthetic_development is not None:
        development = development[: args.limit_synthetic_development]
    if args.limit_synthetic_validation is not None:
        validation = validation[: args.limit_synthetic_validation]
    development_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    if not args.skip_synthetic:
        development_rows = run_parallel(
            development, synthetic_case, grid,
            RESULT_ROOT / "08_budget_fidelity/synthetic_development_results.tsv",
            SYNTHETIC_FIELDS, args.workers, "case_id",
        )
        validation_rows = run_parallel(
            validation, synthetic_case, locked,
            RESULT_ROOT / "08_budget_fidelity/synthetic_validation_results.tsv",
            SYNTHETIC_FIELDS, args.workers, "case_id",
        )
    atlas_rows: list[dict[str, object]] = []
    if not args.skip_atlas:
        atlas_systems = [row["system"] for row in read_tsv(REPO_ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")]
        if args.limit_atlas is not None:
            atlas_systems = atlas_systems[: args.limit_atlas]
        atlas_rows = run_parallel(
            atlas_systems, atlas_case, locked,
            RESULT_ROOT / "08_budget_fidelity/atlas_validation_results.tsv",
            ATLAS_FIELDS, 1, "system_id",
        )
    write_json(
        RESULT_ROOT / "08_budget_fidelity/supplement_summary.json",
        {
            "synthetic_development_settings": len(development),
            "synthetic_development_rows": len(development_rows),
            "synthetic_validation_settings": len(validation),
            "synthetic_validation_rows": len(validation_rows),
            "atlas_rows": len(atlas_rows),
            "failures": sum(
                row.get("status") != "success"
                for row in development_rows + validation_rows + atlas_rows
            ),
            "memory_scope": "worker process high-water mark; not isolated per configuration",
        },
    )


if __name__ == "__main__":
    main()
