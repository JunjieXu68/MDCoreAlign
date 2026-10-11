"""Run SE3-SE4 controlled generative-truth alignment evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from common import (
    REPO_ROOT,
    RESULT_ROOT,
    ca_atom_metadata,
    relative,
    resolve_mdcath_h5,
    write_json,
    write_tsv,
)
from scipy.stats import pearsonr, spearmanr
from synthetic_truth import (
    SyntheticTrajectory,
    aligned_from_transform,
    binary_mask_metrics,
    binary_ranking_metrics,
    gauge_to_reference,
    generate_synthetic_trajectory,
    infer_transforms,
    rmsf,
    rotation_geodesic_degrees,
    top_cardinality_mask,
)

from mdcorealign.contracts import solve_contract
from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_with_occupancy,
    resolve_mdlovofit_executable,
    run_mdlovofit_on_coordinates,
)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

RESULT_FIELDS = [
    "case_id", "template_id", "split", "scenario", "seed", "local_amplitude_A",
    "hinge_amplitude_deg", "n_frames", "n_residues", "method", "status", "error",
    "true_core_count", "estimated_core_count", "core_AUROC", "core_AUPRC",
    "core_Jaccard", "core_Dice", "core_precision", "core_recall", "core_MCC",
    "stability_Spearman", "stability_Pearson", "stability_calibration_slope",
    "stability_calibration_intercept", "stability_MAE", "top_stability_decile_overlap",
    "bottom_stability_decile_overlap", "score_standard_deviation", "score_entropy",
    "rotation_error_median_deg", "rotation_error_p95_deg", "rotation_error_max_deg",
    "core_coordinate_error_median_A", "core_coordinate_error_p95_A",
    "core_coordinate_error_max_A", "translation_error_median_A",
    "RMSF_MAE_A", "RMSF_Spearman", "stable_core_RMSF_true_mean_A",
    "stable_core_RMSF_estimated_mean_A", "mobile_RMSF_true_mean_A",
    "mobile_RMSF_estimated_mean_A", "mobile_RMSF_MAE_A", "global_leakage_core_RMSF_A",
    "unified_RMSD_mean_A", "unified_RMSD_median_A", "weighted_objective_mean_A",
    "MDLovoFit_fraction_supplied", "MDLovoFit_native_least_mobile_RMSD",
    "output_path",
]


def load_template(entry: dict[str, object]) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    source_type = str(entry["source_type"])
    if source_type.startswith("mdCATH"):
        import h5py

        domain_id = str(entry["domain_id"])
        with h5py.File(resolve_mdcath_h5(domain_id), "r") as h5:
            metadata = ca_atom_metadata(h5, domain_id)
            indices = metadata["atom_indices"]
            coords = np.asarray(h5[domain_id]["320"]["0"]["coords"][0, indices], dtype=np.float64)
        sequence = "".join(AA3_TO_1.get(str(item), "X") for item in metadata["residue_names"])
        return coords, metadata, sequence
    path = REPO_ROOT / str(entry["source_file"])
    trajectory = np.load(path, mmap_mode="r")
    coords = np.asarray(trajectory[0], dtype=np.float64)
    manifest = list(csv.DictReader((REPO_ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv").open(), delimiter="\t"))
    row = next(item for item in manifest if item["system"] == str(entry["domain_id"]))
    metadata_npz = np.load(REPO_ROOT / row["residue_metadata_path"])
    metadata = {
        "atom_indices": np.arange(coords.shape[0], dtype=np.int64),
        "residue_ids": np.asarray(metadata_npz["residue_ids"]),
        "residue_names": np.asarray(metadata_npz["residue_names"]),
        "chain_ids": np.asarray(metadata_npz["chain_ids"]),
    }
    sequence = "".join(AA3_TO_1.get(str(item), "X") for item in metadata["residue_names"])
    return coords, metadata, sequence


def build_settings(registry: dict[str, object]) -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []
    local_levels = [float(value) for value in registry["local_amplitudes_A"]]
    hinge_levels = [float(value) for value in registry["hinge_amplitudes_deg"]]
    for template in registry["templates"]:
        for scenario in registry["scenarios"]:
            if scenario == "S0":
                levels = [(0.0, 0.0)]
            elif scenario in {"S1", "S2", "S5", "S6"}:
                levels = [(value, 0.0) for value in local_levels]
            elif scenario == "S3":
                levels = [(0.0, value) for value in hinge_levels]
            else:
                levels = list(zip(local_levels, hinge_levels, strict=True))
            for local_amplitude, hinge_amplitude in levels:
                for seed in registry["seeds"]:
                    case_id = (
                        f"{template['domain_id']}__{scenario}__L{local_amplitude:.2f}"
                        f"__H{hinge_amplitude:.1f}__S{int(seed)}"
                    ).replace(".", "p")
                    settings.append(
                        {
                            "case_id": case_id,
                            "template": template,
                            "scenario": scenario,
                            "local_amplitude_A": local_amplitude,
                            "hinge_amplitude_deg": hinge_amplitude,
                            "seed": int(seed),
                            "n_frames": int(registry["frames"]),
                        }
                    )
    return settings


def method_alignment(
    trajectory: SyntheticTrajectory,
    method: str,
    chain_ids: np.ndarray,
    workdir: Path,
    mdlovofit_executable: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, float | None, float | None]:
    values = trajectory.observed
    n_residues = values.shape[1]
    if method == "oracle_true_core_Kabsch":
        weights = trajectory.true_stability.copy()
        if trajectory.true_stable_mask is not None and np.ptp(weights) == 0.0:
            weights = trajectory.true_stable_mask.astype(np.float64)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones(n_residues, dtype=np.float64)
        fit = fit_all_frames_fixed_weights(
            values, values[0], weights, return_aligned=False,
            compute_rmsd=True, compute_weighted_rmsd=True,
            compute_mean_residual=False, compute_robust_rmsf=False,
            compute_ordinary_rmsf=False,
        )
        aligned = aligned_from_transform(values, fit.rotations, fit.translations)
        return aligned, weights, trajectory.true_stable_mask, fit.weighted_rmsd, None, None
    if method == "ordinary_full_CA_Kabsch":
        weights = np.ones(n_residues, dtype=np.float64)
        fit = fit_all_frames_fixed_weights(
            values, values[0], weights, return_aligned=False,
            compute_rmsd=True, compute_weighted_rmsd=True,
            compute_mean_residual=False, compute_robust_rmsf=False,
            compute_ordinary_rmsf=False,
        )
        aligned = aligned_from_transform(values, fit.rotations, fit.translations)
        return aligned, weights, None, fit.weighted_rmsd, None, None
    if method in {"MDCoreAlign_sequential_reference", "MDCoreAlign_throughput_CPU"}:
        contract = "exact" if method == "MDCoreAlign_sequential_reference" else "throughput"
        result = solve_contract(
            values,
            contract=contract,
            backend="cpu",
            chain_ids=chain_ids,
            store_aligned=False,
            compute_ordinary_rmsf=False,
        ).alignment
        if result is None:
            raise RuntimeError("MDCoreAlign contract returned no alignment")
        aligned = aligned_from_transform(values, result.rotations, result.translations)
        core_count = (
            int(np.count_nonzero(trajectory.true_stable_mask))
            if trajectory.true_stable_mask is not None
            else int(np.floor(0.60 * n_residues))
        )
        mask = top_cardinality_mask(result.frozen_weights, core_count)
        return aligned, np.asarray(result.frozen_weights), mask, np.asarray(result.weighted_rmsd), None, None
    if method == "MDLovoFit_true_fraction_untimed":
        if mdlovofit_executable is None:
            raise FileNotFoundError("MDLovoFit executable unavailable")
        true_count = (
            int(np.count_nonzero(trajectory.true_stable_mask))
            if trajectory.true_stable_mask is not None
            else int(np.floor(0.60 * n_residues))
        )
        fraction = true_count / n_residues
        run = run_mdlovofit_on_coordinates(
            values,
            fraction=fraction,
            workdir=workdir,
            executable=mdlovofit_executable,
            timeout_seconds=180.0,
        )
        if run.aligned_coords is None:
            raise RuntimeError("MDLovoFit did not return aligned coordinates")
        aligned = gauge_to_reference(run.aligned_coords, trajectory.reference)
        _, occupancies = read_multimodel_pdb_with_occupancy(run.output_path, n_residues)
        if not np.allclose(occupancies, occupancies[0]):
            raise ValueError("MDLovoFit output subset is not constant across frames")
        mask = occupancies[0] > 0.5
        parsed = run.parsed_output.average_rmsd_least_mobile
        return aligned, mask.astype(np.float64), mask, None, fraction, parsed
    raise ValueError(method)


def entropy01(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    minimum = float(np.min(values))
    span = float(np.ptp(values))
    scaled = (values - minimum) / span if span > 0.0 else np.full(values.size, 0.5)
    probability = np.clip(scaled, 1e-12, 1.0 - 1e-12)
    entropy = -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))
    return float(np.mean(entropy) / math.log(2.0))


def overlap_at_fraction(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.floor(left.size * fraction)))
    left_mask = top_cardinality_mask(left, count)
    right_mask = top_cardinality_mask(right, count)
    union = np.count_nonzero(left_mask | right_mask)
    return float(np.count_nonzero(left_mask & right_mask) / union) if union else 1.0


def evaluate_method(
    setting: dict[str, object],
    trajectory: SyntheticTrajectory,
    method: str,
    aligned: np.ndarray,
    scores: np.ndarray,
    estimated_mask: np.ndarray | None,
    weighted_rmsd: np.ndarray | None,
    fraction: float | None,
    native_least_rmsd: float | None,
    output_path: Path,
) -> dict[str, object]:
    aligned = gauge_to_reference(aligned, trajectory.reference)
    rotations, translations = infer_transforms(trajectory.observed, aligned)
    rotation_error = rotation_geodesic_degrees(rotations, trajectory.true_rotations)
    coordinate_difference = np.linalg.norm(aligned - trajectory.internal, axis=2)
    true_mask = trajectory.true_stable_mask
    evaluation_mask = true_mask if true_mask is not None else np.ones(scores.size, dtype=bool)
    core_coordinate = np.sqrt(np.mean(coordinate_difference[:, evaluation_mask] ** 2, axis=1))
    true_rmsf = rmsf(trajectory.internal)
    estimated_rmsf = rmsf(aligned)
    full_delta = aligned - trajectory.reference[None, :, :]
    frame_rmsd = np.sqrt(np.mean(np.sum(full_delta * full_delta, axis=2), axis=1))
    row: dict[str, object] = {
        "case_id": setting["case_id"],
        "template_id": setting["template"]["domain_id"],
        "split": setting["template"]["split"],
        "scenario": setting["scenario"],
        "seed": setting["seed"],
        "local_amplitude_A": setting["local_amplitude_A"],
        "hinge_amplitude_deg": setting["hinge_amplitude_deg"],
        "n_frames": setting["n_frames"],
        "n_residues": scores.size,
        "method": method,
        "status": "success",
        "error": "",
        "true_core_count": int(np.count_nonzero(true_mask)) if true_mask is not None else "",
        "estimated_core_count": int(np.count_nonzero(estimated_mask)) if estimated_mask is not None else "",
        "score_standard_deviation": float(np.std(scores)),
        "score_entropy": entropy01(scores),
        "rotation_error_median_deg": float(np.median(rotation_error)),
        "rotation_error_p95_deg": float(np.quantile(rotation_error, 0.95)),
        "rotation_error_max_deg": float(np.max(rotation_error)),
        "core_coordinate_error_median_A": float(np.median(core_coordinate)),
        "core_coordinate_error_p95_A": float(np.quantile(core_coordinate, 0.95)),
        "core_coordinate_error_max_A": float(np.max(core_coordinate)),
        "translation_error_median_A": float(np.median(np.linalg.norm(translations - trajectory.true_translations, axis=1))),
        "RMSF_MAE_A": float(np.mean(np.abs(estimated_rmsf - true_rmsf))),
        "RMSF_Spearman": float(spearmanr(estimated_rmsf, true_rmsf).statistic) if np.ptp(true_rmsf) > 0.0 else 1.0,
        "stable_core_RMSF_true_mean_A": float(np.mean(true_rmsf[evaluation_mask])),
        "stable_core_RMSF_estimated_mean_A": float(np.mean(estimated_rmsf[evaluation_mask])),
        "mobile_RMSF_true_mean_A": float(np.mean(true_rmsf[trajectory.mobile_mask])) if np.any(trajectory.mobile_mask) else "",
        "mobile_RMSF_estimated_mean_A": float(np.mean(estimated_rmsf[trajectory.mobile_mask])) if np.any(trajectory.mobile_mask) else "",
        "mobile_RMSF_MAE_A": float(np.mean(np.abs(estimated_rmsf[trajectory.mobile_mask] - true_rmsf[trajectory.mobile_mask]))) if np.any(trajectory.mobile_mask) else "",
        "global_leakage_core_RMSF_A": float(np.mean(estimated_rmsf[evaluation_mask] - true_rmsf[evaluation_mask])),
        "unified_RMSD_mean_A": float(np.mean(frame_rmsd)),
        "unified_RMSD_median_A": float(np.median(frame_rmsd)),
        "weighted_objective_mean_A": float(np.mean(weighted_rmsd)) if weighted_rmsd is not None else "",
        "MDLovoFit_fraction_supplied": fraction if fraction is not None else "",
        "MDLovoFit_native_least_mobile_RMSD": native_least_rmsd if native_least_rmsd is not None else "",
        "output_path": relative(output_path),
    }
    if true_mask is not None and estimated_mask is not None and 0 < np.count_nonzero(true_mask) < true_mask.size:
        ranking = binary_ranking_metrics(scores, true_mask)
        masks = binary_mask_metrics(estimated_mask, true_mask)
        row.update(
            {
                "core_AUROC": ranking["AUROC"],
                "core_AUPRC": ranking["AUPRC"],
                "core_Jaccard": masks["Jaccard"],
                "core_Dice": masks["Dice"],
                "core_precision": masks["precision"],
                "core_recall": masks["recall"],
                "core_MCC": masks["MCC"],
            }
        )
    if setting["scenario"] == "S5":
        score_span = float(np.ptp(scores))
        normalised = (scores - np.min(scores)) / score_span if score_span > 0 else np.full(scores.size, 0.5)
        slope, intercept = np.polyfit(trajectory.true_stability, normalised, 1)
        row.update(
            {
                "stability_Spearman": float(spearmanr(normalised, trajectory.true_stability).statistic),
                "stability_Pearson": float(pearsonr(normalised, trajectory.true_stability).statistic),
                "stability_calibration_slope": float(slope),
                "stability_calibration_intercept": float(intercept),
                "stability_MAE": float(np.mean(np.abs(normalised - trajectory.true_stability))),
                "top_stability_decile_overlap": overlap_at_fraction(normalised, trajectory.true_stability, 0.10),
                "bottom_stability_decile_overlap": overlap_at_fraction(-normalised, -trajectory.true_stability, 0.10),
            }
        )
    np.savez_compressed(
        output_path,
        scores=np.asarray(scores, dtype=np.float64),
        estimated_mask=np.asarray(estimated_mask, dtype=bool) if estimated_mask is not None else np.asarray([], dtype=bool),
        rotations=rotations,
        translations=translations,
        rmsf=estimated_rmsf,
        frame_rmsd=frame_rmsd,
    )
    return row


def failed_row(setting: dict[str, object], method: str, error: Exception) -> dict[str, object]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "case_id": setting["case_id"],
            "template_id": setting["template"]["domain_id"],
            "split": setting["template"]["split"],
            "scenario": setting["scenario"],
            "seed": setting["seed"],
            "local_amplitude_A": setting["local_amplitude_A"],
            "hinge_amplitude_deg": setting["hinge_amplitude_deg"],
            "n_frames": setting["n_frames"],
            "n_residues": setting["template"]["n_residues"],
            "method": method,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def write_template_manifest(registry: dict[str, object]) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    rows = []
    loaded: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for entry in registry["templates"]:
        coords, metadata, sequence = load_template(entry)
        template_id = str(entry["domain_id"])
        loaded[template_id] = (coords, metadata)
        source = Path(str(entry["source_file"]))
        source_path = source if source.is_absolute() else REPO_ROOT / source
        rows.append(
            {
                "template_id": template_id,
                "accession_chain": template_id,
                "split": entry["split"],
                "length": coords.shape[0],
                "sequence": sequence,
                "domain_annotation": entry["domain_annotation"],
                "secondary_structure": "not_used_for_outcome_selection; sequence-segment proxy registered",
                "source_type": entry["source_type"],
                "source_path": relative(source_path),
                "source_sha256": entry["source_sha256"],
                "source_hash_status": "reused_from_locked_source_manifest",
                "selection_basis": entry["selection_basis"],
            }
        )
    write_tsv(
        RESULT_ROOT / "03_synthetic_truth/template_manifest.tsv",
        rows,
        list(rows[0]),
    )
    return loaded


def generator_validation(trajectory: SyntheticTrajectory, case_id: str) -> dict[str, object]:
    mask = trajectory.true_stable_mask
    oracle_weights = (
        mask.astype(np.float64)
        if mask is not None and np.count_nonzero(mask) >= 3
        else np.ones(trajectory.reference.shape[0], dtype=np.float64)
    )
    fit = fit_all_frames_fixed_weights(
        trajectory.observed,
        trajectory.reference,
        oracle_weights,
        return_aligned=False,
        compute_rmsd=False,
        compute_weighted_rmsd=False,
        compute_mean_residual=False,
        compute_robust_rmsf=False,
        compute_ordinary_rmsf=False,
    )
    angle = rotation_geodesic_degrees(fit.rotations, trajectory.true_rotations)
    return {
        "case_id": case_id,
        "frame0_reference_max_abs_A": float(np.max(np.abs(trajectory.observed[0] - trajectory.reference))),
        "rotation_orthogonality_max": float(np.max(np.abs(trajectory.true_rotations @ np.swapaxes(trajectory.true_rotations, 1, 2) - np.eye(3)))),
        "rotation_determinant_max_abs_error": float(np.max(np.abs(np.linalg.det(trajectory.true_rotations) - 1.0))),
        "oracle_rotation_median_deg": float(np.median(angle)),
        "oracle_rotation_p95_deg": float(np.quantile(angle, 0.95)),
        "passed": bool(
            np.max(np.abs(trajectory.observed[0] - trajectory.reference)) < 1e-10
            and np.max(np.abs(np.linalg.det(trajectory.true_rotations) - 1.0)) < 1e-10
        ),
    }


def aggregate_metric(rows: list[dict[str, str]], metric: str, seed: int = 20260831) -> tuple[float, float, float, int]:
    valid = [row for row in rows if row["status"] == "success" and row.get(metric, "") not in {"", "nan", "NA"}]
    if not valid:
        return float("nan"), float("nan"), float("nan"), 0
    by_template: dict[str, list[float]] = {}
    for row in valid:
        by_template.setdefault(row["template_id"], []).append(float(row[metric]))
    template_values = np.asarray([np.median(values) for values in by_template.values()])
    estimate = float(np.median(template_values))
    rng = np.random.default_rng(seed)
    samples = np.empty(2000, dtype=np.float64)
    for index in range(samples.size):
        draw = rng.integers(0, template_values.size, size=template_values.size)
        samples[index] = np.median(template_values[draw])
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), len(by_template)


def summarize() -> None:
    path = RESULT_ROOT / "04_transform_metrics/synthetic_truth_results.tsv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    heldout = [row for row in rows if row["split"] == "heldout"]
    summary_specs = {
        "core_recovery_summary.tsv": ["core_AUPRC", "core_Jaccard", "core_MCC"],
        "rotation_error_summary.tsv": ["rotation_error_median_deg", "rotation_error_p95_deg", "core_coordinate_error_median_A", "core_coordinate_error_p95_A"],
        "internal_motion_summary.tsv": ["RMSF_MAE_A", "RMSF_Spearman", "mobile_RMSF_MAE_A", "global_leakage_core_RMSF_A"],
    }
    for filename, metrics in summary_specs.items():
        output = []
        groups = sorted({(row["scenario"], row["method"]) for row in heldout})
        for scenario, method in groups:
            group = [row for row in heldout if row["scenario"] == scenario and row["method"] == method]
            for metric in metrics:
                estimate, low, high, count = aggregate_metric(group, metric)
                output.append(
                    {
                        "split": "heldout",
                        "scenario": scenario,
                        "method": method,
                        "metric": metric,
                        "template_clustered_median": estimate,
                        "bootstrap_95CI_low": low,
                        "bootstrap_95CI_high": high,
                        "n_templates": count,
                    }
                )
        fields = [
            "split", "scenario", "method", "metric",
            "template_clustered_median", "bootstrap_95CI_low",
            "bootstrap_95CI_high", "n_templates",
        ]
        write_tsv(RESULT_ROOT / "04_transform_metrics" / filename, output, fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-mdlovofit", action="store_true")
    parser.add_argument("--template", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    registry_path = RESULT_ROOT / "02_preregistration/synthetic_truth_registry.json"
    registry = json.loads(registry_path.read_text())
    shutil.copy2(registry_path, RESULT_ROOT / "03_synthetic_truth/generator_config.json")
    all_settings = build_settings(registry)
    if args.merge_only:
        merged_rows: list[dict[str, str]] = []
        merged_validation: list[dict[str, str]] = []
        for entry in registry["templates"]:
            template_id = str(entry["domain_id"])
            result_path = RESULT_ROOT / f"04_transform_metrics/checkpoint_{template_id}.tsv"
            validation_path = RESULT_ROOT / f"03_synthetic_truth/generator_validation_{template_id}.tsv"
            if not result_path.exists() or not validation_path.exists():
                raise FileNotFoundError(f"missing worker output for {template_id}")
            with result_path.open() as handle:
                merged_rows.extend(csv.DictReader(handle, delimiter="\t"))
            with validation_path.open() as handle:
                merged_validation.extend(csv.DictReader(handle, delimiter="\t"))
        merged_rows.sort(key=lambda row: (row["case_id"], row["method"]))
        merged_validation.sort(key=lambda row: row["case_id"])
        write_tsv(
            RESULT_ROOT / "04_transform_metrics/synthetic_truth_results.tsv",
            merged_rows,
            RESULT_FIELDS,
        )
        write_tsv(
            RESULT_ROOT / "03_synthetic_truth/generator_validation.tsv",
            merged_validation,
            list(merged_validation[0]),
        )
        summarize()
        summary = {
            "settings": len(all_settings),
            "rows": len(merged_rows),
            "failures": sum(row["status"] != "success" for row in merged_rows),
            "generator_passed": all(row["passed"] == "True" for row in merged_validation),
            "worker_merge": True,
        }
        write_json(RESULT_ROOT / "03_synthetic_truth/run_summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return

    if args.template is None:
        templates = write_template_manifest(registry)
    else:
        entry = next(
            item for item in registry["templates"] if item["domain_id"] == args.template
        )
        coords, metadata, _ = load_template(entry)
        templates = {args.template: (coords, metadata)}
    settings = all_settings
    if args.template is not None:
        settings = [
            setting
            for setting in settings
            if setting["template"]["domain_id"] == args.template
        ]
    if args.limit is not None:
        settings = settings[: args.limit]
    trajectory_rows = []
    for setting in settings:
        case_id = str(setting["case_id"])
        trajectory_rows.append(
            {
                "case_id": case_id,
                "template_id": setting["template"]["domain_id"],
                "split": setting["template"]["split"],
                "scenario": setting["scenario"],
                "seed": setting["seed"],
                "local_amplitude_A": setting["local_amplitude_A"],
                "hinge_amplitude_deg": setting["hinge_amplitude_deg"],
                "n_frames": setting["n_frames"],
                "n_residues": setting["template"]["n_residues"],
                "ground_truth_mask": f"results/scientific_evidence_upgrade_v1/03_synthetic_truth/ground_truth_masks/{case_id}.npz",
                "ground_truth_transform": f"results/scientific_evidence_upgrade_v1/03_synthetic_truth/ground_truth_transforms/{case_id}.npz",
                "coordinate_storage": "deterministic_regeneration_from_template_registry_and_seed",
            }
        )
    trajectory_manifest_path = RESULT_ROOT / "03_synthetic_truth/trajectory_manifest.tsv"
    if args.template is None:
        write_tsv(trajectory_manifest_path, trajectory_rows, list(trajectory_rows[0]))
    if args.prepare_only:
        print(json.dumps({"status": "SE3_PREPARED", "settings": len(settings)}))
        return
    mask_dir = RESULT_ROOT / "03_synthetic_truth/ground_truth_masks"
    transform_dir = RESULT_ROOT / "03_synthetic_truth/ground_truth_transforms"
    method_dir = RESULT_ROOT / "03_synthetic_truth/method_outputs"
    for directory in (mask_dir, transform_dir, method_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULT_ROOT / "04_transform_metrics" / (
        f"checkpoint_{args.template}.tsv"
        if args.template is not None
        else "synthetic_truth_results.tsv"
    )
    existing_rows: list[dict[str, str]] = []
    completed: set[tuple[str, str]] = set()
    if checkpoint.exists():
        with checkpoint.open() as handle:
            existing_rows = list(csv.DictReader(handle, delimiter="\t"))
        completed = {
            (row["case_id"], row["method"])
            for row in existing_rows
            if row.get("status") == "success"
        }
    methods = [
        "oracle_true_core_Kabsch",
        "ordinary_full_CA_Kabsch",
        "MDCoreAlign_sequential_reference",
        "MDCoreAlign_throughput_CPU",
    ]
    if not args.skip_mdlovofit:
        methods.append("MDLovoFit_true_fraction_untimed")
    try:
        mdlovofit_executable = resolve_mdlovofit_executable() if not args.skip_mdlovofit else None
    except FileNotFoundError:
        mdlovofit_executable = None
    validation_rows = []
    all_rows: list[dict[str, object]] = list(existing_rows)
    for index, setting in enumerate(settings, start=1):
        template_id = str(setting["template"]["domain_id"])
        reference, metadata = templates[template_id]
        trajectory = generate_synthetic_trajectory(
            reference,
            scenario=str(setting["scenario"]),
            n_frames=int(setting["n_frames"]),
            seed=int(setting["seed"]),
            local_amplitude_A=float(setting["local_amplitude_A"]),
            hinge_amplitude_deg=float(setting["hinge_amplitude_deg"]),
        )
        case_id = str(setting["case_id"])
        validation_rows.append(generator_validation(trajectory, case_id))
        np.savez_compressed(
            mask_dir / f"{case_id}.npz",
            true_stable_mask=(trajectory.true_stable_mask if trajectory.true_stable_mask is not None else np.asarray([], dtype=bool)),
            true_stability=trajectory.true_stability,
            mobile_mask=trajectory.mobile_mask,
        )
        np.savez_compressed(
            transform_dir / f"{case_id}.npz",
            rotations=trajectory.true_rotations,
            translations=trajectory.true_translations,
            hinge_angles_deg=(trajectory.hinge_angles_deg if trajectory.hinge_angles_deg is not None else np.asarray([], dtype=np.float64)),
        )
        for method in methods:
            if (case_id, method) in completed:
                continue
            all_rows = [
                row
                for row in all_rows
                if not (row.get("case_id") == case_id and row.get("method") == method)
            ]
            output_path = method_dir / f"{case_id}__{method}.npz"
            try:
                with tempfile.TemporaryDirectory(
                    prefix="se3_mdlovofit_",
                    dir=RESULT_ROOT / "03_synthetic_truth",
                ) as temporary:
                    aligned, scores, mask, weighted, fraction, native = method_alignment(
                        trajectory,
                        method,
                        np.asarray(metadata["chain_ids"]),
                        Path(temporary),
                        mdlovofit_executable,
                    )
                row = evaluate_method(
                    setting,
                    trajectory,
                    method,
                    aligned,
                    scores,
                    mask,
                    weighted,
                    fraction,
                    native,
                    output_path,
                )
            except Exception as error:
                row = failed_row(setting, method, error)
            all_rows.append(row)
            write_tsv(checkpoint, all_rows, RESULT_FIELDS)
        if index % 10 == 0 or index == len(settings):
            print(f"SE3 progress {index}/{len(settings)} cases", flush=True)
    validation_path = RESULT_ROOT / "03_synthetic_truth" / (
        f"generator_validation_{args.template}.tsv"
        if args.template is not None
        else "generator_validation.tsv"
    )
    write_tsv(
        validation_path,
        validation_rows,
        list(validation_rows[0]),
    )
    (RESULT_ROOT / "04_transform_metrics/metric_implementation.md").write_text(
        """# Metric implementation\n\n"
        "All transforms are independently reconstructed from original and aligned "
        "coordinates after a frame-0 gauge correction. Rotation error is the SO(3) "
        "geodesic angle. Core coordinate error compares estimated and true mapped "
        "coordinates on the known generative core. Core rankings use deterministic "
        "index tie breaking at the true cardinality. Template-clustered bootstrap "
        "intervals use 2,000 resamples. PDB rounding affects MDLovoFit coordinate "
        "metrics and remains visible rather than corrected away.\n"
        """,
        encoding="utf-8",
    )
    if args.template is None:
        summarize()
    summary = {
        "settings": len(settings),
        "methods": methods,
        "rows": len(all_rows),
        "failures": sum(str(row.get("status")) != "success" for row in all_rows),
        "generator_passed": all(bool(row["passed"]) for row in validation_rows),
    }
    run_summary_path = RESULT_ROOT / "03_synthetic_truth" / (
        f"run_summary_{args.template}.json"
        if args.template is not None
        else "run_summary.json"
    )
    write_json(run_summary_path, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
