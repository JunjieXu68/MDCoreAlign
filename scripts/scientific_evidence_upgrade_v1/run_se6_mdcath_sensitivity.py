"""Run the locked 5,397-domain mdCATH RMSF alignment-sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from common import (
    B3_MANIFEST,
    REPO_ROOT,
    RESULT_ROOT,
    ca_atom_metadata,
    core_mask_exact_fraction,
    read_tsv,
    resolve_mdcath_h5,
    stable_hash,
    write_json,
    write_tsv,
)
from scipy.stats import pearsonr, spearmanr

from mdcorealign.contracts import solve_contract

FIELDS = [
    "domain_id", "trajectory_id", "split", "cath_class", "cath_architecture",
    "cath_topology", "cath_homologous_superfamily", "temperature_K", "replicate",
    "n_frames", "n_residues", "chain_count", "status", "error",
    "signed_mean_difference_A", "MAE_A", "RMSE_A", "normalised_MAE", "Pearson",
    "Spearman", "top10_flexible_Jaccard", "top20_flexible_Jaccard",
    "bottom10_rigid_Jaccard", "flexible_quantile_change_fraction",
    "maximum_local_absolute_change_A", "RMSF_peak_index_shift",
    "rank_displacement_mean", "rank_displacement_median", "rank_displacement_p95",
    "alignment_sensitivity_index", "bundled_mean_RMSD_A", "bundled_p95_RMSD_A",
    "radius_gyration_range_A", "mean_conventional_RMSF_A", "mean_stable_RMSF_A",
    "stable_core_fragmentation", "effective_core_fraction", "weight_entropy",
    "large_motion_RMSD_p95_A", "source_h5",
]


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    output = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        output[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return output


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def quantile_mask(values: np.ndarray, fraction: float, high: bool) -> np.ndarray:
    count = max(1, int(np.floor(values.size * fraction)))
    scores = values if high else -values
    order = np.lexsort((np.arange(values.size), -scores))
    output = np.zeros(values.size, dtype=bool)
    output[order[:count]] = True
    return output


def weight_entropy(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    span = float(np.ptp(values))
    scaled = (values - np.min(values)) / span if span > 0.0 else np.full(values.size, 0.5)
    scaled = np.clip(scaled, 1e-12, 1.0 - 1e-12)
    entropy = -(scaled * np.log(scaled) + (1.0 - scaled) * np.log(1.0 - scaled))
    return float(np.mean(entropy) / math.log(2.0))


def process_domain(row: dict[str, str], split: str, save_profile: bool = False) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    domain_id = row["domain_id"]
    try:
        with h5py.File(resolve_mdcath_h5(domain_id), "r") as h5:
            metadata = ca_atom_metadata(h5, domain_id)
            group = h5[domain_id]["320"]["0"]
            coords = np.asarray(group["coords"][:, metadata["atom_indices"], :], dtype=np.float64)
            conventional = np.asarray(group["rmsf"], dtype=np.float64) * 10.0
            bundled_rmsd = np.asarray(group["rmsd"], dtype=np.float64) * 10.0
            gyration = np.asarray(group["gyrationRadius"], dtype=np.float64) * 10.0
        result = solve_contract(
            coords,
            contract="throughput",
            backend="cpu",
            chain_ids=metadata["chain_ids"],
            store_aligned=False,
            compute_ordinary_rmsf=False,
        ).alignment
        if result is None:
            raise RuntimeError("throughput returned no alignment")
        stable = np.asarray(result.robust_rmsf, dtype=np.float64)
        weights = np.asarray(result.frozen_weights, dtype=np.float64)
        if stable.shape != conventional.shape:
            raise ValueError("RMSF residue mapping mismatch")
        difference = stable - conventional
        top10_a = quantile_mask(stable, 0.10, True)
        top10_b = quantile_mask(conventional, 0.10, True)
        top20_a = quantile_mask(stable, 0.20, True)
        top20_b = quantile_mask(conventional, 0.20, True)
        bottom10_a = quantile_mask(stable, 0.10, False)
        bottom10_b = quantile_mask(conventional, 0.10, False)
        rank_difference = np.abs(rank_average(stable) - rank_average(conventional))
        core = core_mask_exact_fraction(weights, 0.60)
        transitions = core[1:] != core[:-1]
        chain_breaks = metadata["chain_ids"][1:] != metadata["chain_ids"][:-1]
        fragmentation = int(np.count_nonzero(transitions & ~chain_breaks))
        mean_conventional = float(np.mean(conventional))
        pearson = float(pearsonr(stable, conventional).statistic) if np.ptp(conventional) > 0 and np.ptp(stable) > 0 else 1.0
        spearman = float(spearmanr(stable, conventional).statistic) if np.ptp(conventional) > 0 and np.ptp(stable) > 0 else 1.0
        output = {
            "domain_id": domain_id,
            "trajectory_id": f"{domain_id}__T320__R1",
            "split": split,
            "cath_class": row["cath_class"],
            "cath_architecture": row["cath_architecture"],
            "cath_topology": row["cath_topology"],
            "cath_homologous_superfamily": row["cath_homologous_superfamily"],
            "temperature_K": 320,
            "replicate": 1,
            "n_frames": coords.shape[0],
            "n_residues": coords.shape[1],
            "chain_count": len(set(str(item) for item in metadata["chain_ids"])),
            "status": "success",
            "error": "",
            "signed_mean_difference_A": float(np.mean(difference)),
            "MAE_A": float(np.mean(np.abs(difference))),
            "RMSE_A": float(np.sqrt(np.mean(difference * difference))),
            "normalised_MAE": float(np.mean(np.abs(difference)) / max(mean_conventional, 1e-12)),
            "Pearson": pearson,
            "Spearman": spearman,
            "top10_flexible_Jaccard": jaccard(top10_a, top10_b),
            "top20_flexible_Jaccard": jaccard(top20_a, top20_b),
            "bottom10_rigid_Jaccard": jaccard(bottom10_a, bottom10_b),
            "flexible_quantile_change_fraction": float(np.mean(top20_a != top20_b)),
            "maximum_local_absolute_change_A": float(np.max(np.abs(difference))),
            "RMSF_peak_index_shift": abs(int(np.argmax(stable)) - int(np.argmax(conventional))),
            "rank_displacement_mean": float(np.mean(rank_difference)),
            "rank_displacement_median": float(np.median(rank_difference)),
            "rank_displacement_p95": float(np.quantile(rank_difference, 0.95)),
            "alignment_sensitivity_index": float(np.sum(np.abs(difference)) / max(np.sum(conventional), 1e-12)),
            "bundled_mean_RMSD_A": float(np.mean(bundled_rmsd)),
            "bundled_p95_RMSD_A": float(np.quantile(bundled_rmsd, 0.95)),
            "radius_gyration_range_A": float(np.ptp(gyration)),
            "mean_conventional_RMSF_A": mean_conventional,
            "mean_stable_RMSF_A": float(np.mean(stable)),
            "stable_core_fragmentation": fragmentation,
            "effective_core_fraction": float(np.sum(weights) / max(np.max(weights), 1e-12) / weights.size),
            "weight_entropy": weight_entropy(weights),
            "large_motion_RMSD_p95_A": float(np.quantile(bundled_rmsd, 0.95)),
            "source_h5": str(resolve_mdcath_h5(domain_id).relative_to(REPO_ROOT)),
        }
        profile = None
        if save_profile:
            profile = {
                "residue_ids": metadata["residue_ids"],
                "residue_names": metadata["residue_names"],
                "chain_ids": metadata["chain_ids"],
                "dataset_conventional_RMSF_A": conventional,
                "stable_reference_RMSF_A": stable,
                "stable_weight": weights,
                "core_label": core.astype(np.int8),
            }
        return output, profile
    except Exception as error:
        failed = {field: "" for field in FIELDS}
        failed.update(
            {
                "domain_id": domain_id,
                "trajectory_id": f"{domain_id}__T320__R1",
                "split": split,
                "cath_class": row.get("cath_class", ""),
                "cath_architecture": row.get("cath_architecture", ""),
                "cath_topology": row.get("cath_topology", ""),
                "cath_homologous_superfamily": row.get("cath_homologous_superfamily", ""),
                "temperature_K": 320,
                "replicate": 1,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return failed, None


def bootstrap_ci(values: np.ndarray, seed: int = 20260831) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(2000, dtype=np.float64)
    for index in range(estimates.size):
        draw = rng.integers(0, values.size, size=values.size)
        estimates[index] = np.median(values[draw])
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 1.0
    for reverse_index in range(pvalues.size - 1, -1, -1):
        index = order[reverse_index]
        value = pvalues[index] * pvalues.size / (reverse_index + 1)
        running = min(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def finalise(rows: list[dict[str, str]], domain_rows: dict[str, dict[str, str]]) -> None:
    successful = [row for row in rows if row["status"] == "success"]
    validation = [row for row in successful if row["split"] == "validation"]
    per_domain = sorted(successful, key=lambda row: row["domain_id"])
    write_tsv(
        RESULT_ROOT / "06_mdcath_rmsf_sensitivity/per_domain_metrics.tsv",
        per_domain,
        FIELDS,
    )
    metrics = [
        "signed_mean_difference_A", "MAE_A", "RMSE_A", "normalised_MAE",
        "Pearson", "Spearman", "top10_flexible_Jaccard",
        "top20_flexible_Jaccard", "bottom10_rigid_Jaccard",
        "flexible_quantile_change_fraction", "maximum_local_absolute_change_A",
        "alignment_sensitivity_index",
    ]
    summary_rows = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in validation])
        low, high = bootstrap_ci(values)
        summary_rows.append(
            {
                "split": "validation",
                "metric": metric,
                "n_domains": values.size,
                "median": float(np.median(values)),
                "IQR_low": float(np.quantile(values, 0.25)),
                "IQR_high": float(np.quantile(values, 0.75)),
                "bootstrap_95CI_low": low,
                "bootstrap_95CI_high": high,
            }
        )
    write_tsv(
        RESULT_ROOT / "06_mdcath_rmsf_sensitivity/temperature_summary.tsv",
        [
            {
                "temperature_K": 320,
                "n_domains": len(validation),
                "median_alignment_sensitivity_index": float(np.median([float(row["alignment_sensitivity_index"]) for row in validation])),
                "median_top20_flexible_Jaccard": float(np.median([float(row["top20_flexible_Jaccard"]) for row in validation])),
                "scope_note": "domain-canonical panel only; temperature effect is not estimable",
            }
        ],
        ["temperature_K", "n_domains", "median_alignment_sensitivity_index", "median_top20_flexible_Jaccard", "scope_note"],
    )
    write_tsv(
        RESULT_ROOT / "06_mdcath_rmsf_sensitivity/sensitivity_summary.tsv",
        summary_rows,
        list(summary_rows[0]),
    )
    predictors = [
        "n_residues", "bundled_mean_RMSD_A", "radius_gyration_range_A",
        "mean_conventional_RMSF_A", "stable_core_fragmentation",
        "effective_core_fraction", "weight_entropy", "chain_count",
    ]
    association = []
    pvalues = []
    target = np.asarray([float(row["alignment_sensitivity_index"]) for row in validation])
    for predictor in predictors:
        values = np.asarray([float(row[predictor]) for row in validation])
        if np.ptp(values) == 0.0:
            statistic, pvalue = float("nan"), 1.0
        else:
            result = spearmanr(values, target)
            statistic, pvalue = float(result.statistic), float(result.pvalue)
        association.append(
            {
                "outcome": "alignment_sensitivity_index",
                "predictor": predictor,
                "method": "Spearman_domain_level",
                "effect_size": statistic,
                "pvalue": pvalue,
            }
        )
        pvalues.append(pvalue)
    adjusted = bh_adjust(np.asarray(pvalues))
    for row, value in zip(association, adjusted, strict=True):
        row["BH_adjusted_pvalue"] = float(value)
    write_tsv(
        RESULT_ROOT / "06_mdcath_rmsf_sensitivity/association_models.tsv",
        association,
        list(association[0]),
    )
    ordered = sorted(validation, key=lambda row: float(row["alignment_sensitivity_index"]))
    selections: list[dict[str, object]] = []
    chosen = (
        [(row, "lowest") for row in ordered[:3]]
        + [(row, "nearest_validation_median") for row in sorted(ordered, key=lambda row: abs(float(row["alignment_sensitivity_index"]) - float(np.median(target))))[:3]]
        + [(row, "highest") for row in ordered[-3:]]
    )
    profile_dir = RESULT_ROOT / "06_mdcath_rmsf_sensitivity/case_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    for row, category in chosen:
        _, profile = process_domain(domain_rows[row["domain_id"]], row["split"], save_profile=True)
        if profile is None:
            continue
        profile_path = profile_dir / f"{row['domain_id']}.tsv"
        profile_rows = []
        for index in range(len(profile["residue_ids"])):
            profile_rows.append(
                {
                    "domain_id": row["domain_id"],
                    "residue_index": index,
                    "chain": profile["chain_ids"][index],
                    "residue_id": profile["residue_ids"][index],
                    "resname": profile["residue_names"][index],
                    "dataset_conventional_RMSF_A": profile["dataset_conventional_RMSF_A"][index],
                    "stable_reference_RMSF_A": profile["stable_reference_RMSF_A"][index],
                    "stable_weight": profile["stable_weight"][index],
                    "core_label": profile["core_label"][index],
                }
            )
        write_tsv(profile_path, profile_rows, list(profile_rows[0]))
        selections.append(
            {
                "domain_id": row["domain_id"],
                "category": category,
                "alignment_sensitivity_index": row["alignment_sensitivity_index"],
                "top20_flexible_Jaccard": row["top20_flexible_Jaccard"],
                "n_residues": row["n_residues"],
                "profile_path": str(profile_path.relative_to(REPO_ROOT)),
                "selection_note": "distribution_selected_explanatory_not_mechanistic",
            }
        )
    write_tsv(
        RESULT_ROOT / "06_mdcath_rmsf_sensitivity/case_selection.tsv",
        selections,
        list(selections[0]),
    )
    median_asi = float(np.median(target))
    median_top20 = float(np.median([float(row["top20_flexible_Jaccard"]) for row in validation]))
    prevalence = float(np.mean([
        float(row["alignment_sensitivity_index"]) >= 0.10
        or float(row["top20_flexible_Jaccard"]) <= 0.85
        for row in validation
    ]))
    if median_asi >= 0.20 and median_top20 <= 0.70 and prevalence >= 0.50:
        decision = "ALIGNMENT_SENSITIVITY_STRONG"
    elif (median_asi >= 0.10 or median_top20 <= 0.85) and prevalence >= 0.25:
        decision = "ALIGNMENT_SENSITIVITY_MODERATE"
    else:
        decision = "ALIGNMENT_SENSITIVITY_LIMITED"
    (RESULT_ROOT / "06_mdcath_rmsf_sensitivity/alignment_sensitivity_decision.md").write_text(
        f"# Alignment-sensitivity decision\n\n"
        f"Registered decision: **{decision}**. Validation-domain median ASI was "
        f"`{median_asi:.6g}`, median top-20% flexible-residue Jaccard was "
        f"`{median_top20:.6g}`, and registered material-change prevalence was "
        f"`{prevalence:.3%}` across `{len(validation)}` domains.\n\n"
        "This is a 5,397-domain, 320 K replicate-1 canonical analysis. It is "
        "database-scale domain coverage, not the complete 134,950 temperature/replicate "
        "trajectory deployment. Temperature dependence is therefore not inferred. "
        "Neither RMSF convention is called biological ground truth.\n",
        encoding="utf-8",
    )
    write_json(
        RESULT_ROOT / "06_mdcath_rmsf_sensitivity/run_summary.json",
        {
            "n_total": len(rows),
            "n_success": len(successful),
            "n_failed": len(rows) - len(successful),
            "n_validation": len(validation),
            "decision": decision,
            "median_ASI": median_asi,
            "median_top20_Jaccard": median_top20,
            "material_change_prevalence": prevalence,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    convention = json.loads(
        (RESULT_ROOT / "05_mdcath_rmsf_convention/audit_summary.json").read_text()
    )
    if convention["decision"] != "PASS_SE6_MAY_PROCEED":
        raise RuntimeError("SE5 reproduction gate did not pass")
    registry = json.loads(
        (RESULT_ROOT / "02_preregistration/rmsf_sensitivity_registry.json").read_text()
    )
    development = set(registry["domain_split"]["development_domains"])
    domain_rows: dict[str, dict[str, str]] = {}
    for row in read_tsv(B3_MANIFEST):
        if row["locally_available"] == "True":
            domain_rows.setdefault(row["domain_id"], row)
    selected = sorted(domain_rows, key=lambda item: stable_hash("se6-run:" + item))
    if args.limit is not None:
        selected = selected[: args.limit]
    pairing = [
        {
            "domain_id": domain_id,
            "trajectory_id": f"{domain_id}__T320__R1",
            "split": "development" if domain_id in development else "validation",
            "temperature_K": 320,
            "replicate": 1,
            "n_residues": domain_rows[domain_id]["expected_n_residues"],
            "dataset_RMSF_path": f"/{domain_id}/320/0/rmsf",
            "coordinate_path": f"/{domain_id}/320/0/coords",
            "mapping": "embedded_pdbProteinAtoms_CA_order",
        }
        for domain_id in selected
    ]
    write_tsv(
        RESULT_ROOT / "06_mdcath_rmsf_sensitivity/pairing_manifest.tsv",
        pairing,
        list(pairing[0]),
    )
    checkpoint = RESULT_ROOT / "06_mdcath_rmsf_sensitivity/per_trajectory_metrics.tsv"
    existing: list[dict[str, str]] = []
    completed: set[str] = set()
    if checkpoint.exists():
        with checkpoint.open() as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
        completed = {row["domain_id"] for row in existing}
    pending = [domain_id for domain_id in selected if domain_id not in completed]
    rows: list[dict[str, object]] = list(existing)
    environment = os.environ
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_domain,
                domain_rows[domain_id],
                "development" if domain_id in development else "validation",
            ): domain_id
            for domain_id in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row, _ = future.result()
            rows.append(row)
            if index % 25 == 0 or index == len(pending):
                write_tsv(checkpoint, rows, FIELDS)
                print(f"SE6 progress {len(existing) + index}/{len(selected)}", flush=True)
    rows.sort(key=lambda row: str(row["domain_id"]))
    write_tsv(checkpoint, rows, FIELDS)
    if args.limit is None:
        finalise([dict(row) for row in rows], domain_rows)
    print(json.dumps({"n_rows": len(rows), "n_failed": sum(row["status"] != "success" for row in rows)}))


if __name__ == "__main__":
    main()
