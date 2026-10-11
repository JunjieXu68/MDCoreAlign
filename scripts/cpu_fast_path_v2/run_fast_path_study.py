#!/usr/bin/env python
"""Run the prospectively registered MDCoreAlign CPU fast-path study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.mdcath_benchmark.extract_ca_trajectory_from_mdcath_h5 import (  # noqa: E402
    extract_ca_trajectory,
)
from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights  # noqa: E402
from mdsuperpose.fast_kabsch import (  # noqa: E402
    apply_transforms_batch,
    weighted_kabsch_transform_batch,
)
from mdsuperpose.parallel_weights import (  # noqa: E402
    ParallelWeightResult,
    estimate_batch_then_sequential,
    estimate_blockwise_warm_start,
    estimate_independent_batch,
    estimate_risk_triggered_refinement,
    estimate_shared_weight,
)
from mdsuperpose.sequential_weights import (  # noqa: E402
    estimate_representative_weights_sequential_mean,
)
from mdsuperpose.window_refine import temporally_stratified_indices  # noqa: E402


OUT = ROOT / "results/cpu_fast_path_v2"
ROBUST_KWARGS = {
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


@dataclass(slots=True)
class StudyResult:
    method: str
    weights: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    rmsd: np.ndarray
    weighted_rmsd: np.ndarray
    robust_rmsf: np.ndarray
    mean_residual: np.ndarray
    runtime_seconds: float
    timing: dict[str, float]
    refinement_fraction: float
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=["audit", "development", "lock", "heldout", "performance"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def append_tsv(
    path: Path,
    row: dict[str, object],
    *,
    sync: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        if sync:
            os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_coords(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=True) as archive:
        key = "coords" if "coords" in archive.files else archive.files[0]
        coords = np.asarray(archive[key], dtype=np.float64)
        chains = None
        if "chain_ids" in archive.files:
            chains = np.asarray(archive["chain_ids"]).astype(str)
        elif "residue_metadata" in archive.files:
            metadata = list(archive["residue_metadata"])
            values: list[str] = []
            for item in metadata:
                if isinstance(item, dict):
                    values.append(str(item.get("chain", item.get("chain_id", "A"))))
                else:
                    values.append("A")
            if len(values) == coords.shape[1]:
                chains = np.asarray(values, dtype=str)
    return coords, chains


def load_case(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / row["input_path"]
    if path.suffix == ".npy":
        coords = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64).copy()
        chains = None
    elif path.suffix == ".npz":
        coords, chains = _npz_coords(path)
    elif path.suffix in {".h5", ".hdf5"}:
        coords, _, residue_metadata, _ = extract_ca_trajectory(
            path,
            coordinate_path=row["input_key"],
            contiguous_full_atom_read=True,
        )
        coords = np.asarray(coords, dtype=np.float64)
        chains = np.asarray(
            [str(item.get("chain", item.get("chain_id", "A"))) for item in residue_metadata],
            dtype=str,
        )
    else:
        raise ValueError(f"unsupported study input: {path}")
    if chains is None or chains.shape != (coords.shape[1],):
        chains = np.asarray(["A"] * coords.shape[1], dtype=str)
    # Touch all pages before any loaded-coordinate timer starts.
    if not np.isfinite(float(np.sum(coords, dtype=np.float64))):
        raise ValueError(f"non-finite coordinates: {row['case_id']}")
    return coords, chains


def representative_indices(n_frames: int) -> np.ndarray:
    return temporally_stratified_indices(n_frames, min(2000, n_frames))


def _fit_float64(
    coords: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    start = time.perf_counter()
    fit = fit_all_frames_fixed_weights(
        coords,
        coords[0],
        weights,
        chunk_size=2000,
        return_aligned=False,
        compute_ordinary_rmsf=False,
    )
    elapsed = time.perf_counter() - start
    assert fit.rmsd is not None
    assert fit.weighted_rmsd is not None
    assert fit.robust_rmsf is not None
    assert fit.mean_residual is not None
    return {
        "rotations": fit.rotations,
        "translations": fit.translations,
        "rmsd": fit.rmsd,
        "weighted_rmsd": fit.weighted_rmsd,
        "robust_rmsf": fit.robust_rmsf,
        "mean_residual": fit.mean_residual,
    }, {"all_frame_fit": elapsed, **fit.timing}


def _fit_precision(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    transform_dtype: np.dtype,
    reduction_dtype: np.dtype,
    chunk_size: int = 2000,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Study-only compact fit used to adjudicate non-float64 candidates."""

    start_total = time.perf_counter()
    values = np.asarray(coords, dtype=transform_dtype)
    reference = np.asarray(coords[0], dtype=transform_dtype)
    fixed = np.asarray(weights, dtype=transform_dtype)
    n_frames, n_residues, _ = values.shape
    rotations = np.empty((n_frames, 3, 3), dtype=transform_dtype)
    translations = np.empty((n_frames, 3), dtype=transform_dtype)
    rmsd = np.empty(n_frames, dtype=reduction_dtype)
    weighted = np.empty(n_frames, dtype=reduction_dtype)
    coordinate_sum = np.zeros((n_residues, 3), dtype=reduction_dtype)
    coordinate_sumsq = np.zeros((n_residues, 3), dtype=reduction_dtype)
    residual_sum = np.zeros(n_residues, dtype=reduction_dtype)
    fit_seconds = 0.0
    reduction_seconds = 0.0
    weight_sum = np.sum(fixed, dtype=reduction_dtype)
    for first in range(0, n_frames, chunk_size):
        last = min(first + chunk_size, n_frames)
        chunk = values[first:last]
        fit_start = time.perf_counter()
        transform = weighted_kabsch_transform_batch(
            chunk,
            reference,
            fixed,
            validate_inputs=False,
            dtype=transform_dtype,
        )
        aligned = apply_transforms_batch(
            chunk,
            transform.rotations,
            transform.translations,
        ).astype(transform_dtype, copy=False)
        fit_seconds += time.perf_counter() - fit_start
        reduce_start = time.perf_counter()
        aligned_reduce = aligned.astype(reduction_dtype, copy=False)
        reference_reduce = reference.astype(reduction_dtype, copy=False)
        delta = aligned_reduce - reference_reduce[None, :, :]
        squared = np.sum(delta * delta, axis=2, dtype=reduction_dtype)
        rmsd[first:last] = np.sqrt(np.mean(squared, axis=1, dtype=reduction_dtype))
        weighted[first:last] = np.sqrt(
            np.sum(squared * fixed.astype(reduction_dtype)[None, :], axis=1, dtype=reduction_dtype)
            / weight_sum
        )
        coordinate_sum += np.sum(aligned_reduce, axis=0, dtype=reduction_dtype)
        coordinate_sumsq += np.sum(
            aligned_reduce * aligned_reduce, axis=0, dtype=reduction_dtype
        )
        residual_sum += np.sum(np.sqrt(squared), axis=0, dtype=reduction_dtype)
        rotations[first:last] = transform.rotations
        translations[first:last] = transform.translations
        reduction_seconds += time.perf_counter() - reduce_start
    mean = coordinate_sum / n_frames
    variance = np.maximum(coordinate_sumsq / n_frames - mean * mean, 0.0)
    robust_rmsf = np.sqrt(np.sum(variance, axis=1, dtype=reduction_dtype))
    total = time.perf_counter() - start_total
    return {
        "rotations": rotations,
        "translations": translations,
        "rmsd": rmsd,
        "weighted_rmsd": weighted,
        "robust_rmsf": robust_rmsf,
        "mean_residual": residual_sum / n_frames,
    }, {
        "all_frame_fit": total,
        "all_frame_transform": fit_seconds,
        "all_frame_reductions": reduction_seconds,
    }


def run_exact(coords: np.ndarray, chains: np.ndarray) -> StudyResult:
    selected = representative_indices(coords.shape[0])
    total_start = time.perf_counter()
    weight_start = time.perf_counter()
    estimate = estimate_representative_weights_sequential_mean(
        coords[selected],
        coords[0],
        sequence_chain_ids=chains,
        collect_timing=True,
        **ROBUST_KWARGS,
    )
    weight_seconds = time.perf_counter() - weight_start
    fit, fit_timing = _fit_float64(coords, estimate.mean_weights)
    total = time.perf_counter() - total_start
    return StudyResult(
        method="exact_sequential_f64_compact",
        weights=np.asarray(estimate.mean_weights, dtype=np.float64),
        rotations=fit["rotations"],
        translations=fit["translations"],
        rmsd=fit["rmsd"],
        weighted_rmsd=fit["weighted_rmsd"],
        robust_rmsf=fit["robust_rmsf"],
        mean_residual=fit["mean_residual"],
        runtime_seconds=total,
        timing={
            "representative_selection": 0.0,
            "weight_estimation": weight_seconds,
            "chronological_correction": weight_seconds,
            **fit_timing,
            "stable_core_RMSF": fit_timing.get("RMSF", 0.0),
            "result_construction": max(0.0, total - weight_seconds - fit_timing["all_frame_fit"]),
        },
        refinement_fraction=1.0,
        metadata={
            "representative_count": int(selected.size),
            "average_iterations": estimate.average_iterations_per_frame,
            "converged_fraction": estimate.converged_fraction,
            "warnings": ";".join(estimate.warnings),
        },
    )


def _candidate_estimate(
    config: str,
    representatives: np.ndarray,
    reference: np.ndarray,
    chains: np.ndarray,
) -> ParallelWeightResult:
    if config == "A_independent_batch":
        return estimate_independent_batch(representatives, reference, chain_ids=chains)
    if config == "B_shared_f64":
        return estimate_shared_weight(representatives, reference, chain_ids=chains)
    if config.startswith("C_correction_"):
        return estimate_batch_then_sequential(
            representatives,
            reference,
            chain_ids=chains,
            correction_iterations=int(config.rsplit("_", 1)[1]),
        )
    if config.startswith("D_block_"):
        return estimate_blockwise_warm_start(
            representatives,
            reference,
            chain_ids=chains,
            block_size=int(config.rsplit("_", 1)[1]),
        )
    if config.startswith("E_risk_"):
        _, _, quantile_text, padding_text = config.split("_")
        return estimate_risk_triggered_refinement(
            representatives,
            reference,
            chain_ids=chains,
            risk_quantile=float(quantile_text),
            interval_padding=int(padding_text),
            correction_iterations=2,
        )
    raise ValueError(f"unknown candidate config: {config}")


def run_candidate(
    coords: np.ndarray,
    chains: np.ndarray,
    config: str,
) -> StudyResult:
    selected = representative_indices(coords.shape[0])
    total_start = time.perf_counter()
    estimate = _candidate_estimate(config, coords[selected], coords[0], chains)
    fit, fit_timing = _fit_float64(coords, estimate.mean_weights)
    total = time.perf_counter() - total_start
    return StudyResult(
        method=config,
        weights=estimate.mean_weights,
        rotations=fit["rotations"],
        translations=fit["translations"],
        rmsd=fit["rmsd"],
        weighted_rmsd=fit["weighted_rmsd"],
        robust_rmsf=fit["robust_rmsf"],
        mean_residual=fit["mean_residual"],
        runtime_seconds=total,
        timing={
            "representative_selection": 0.0,
            "weight_estimation": estimate.total_runtime_seconds,
            "chronological_correction": sum(
                value for key, value in estimate.timing.items() if "chronological" in key
            ),
            **fit_timing,
            "stable_core_RMSF": fit_timing.get("RMSF", 0.0),
            "result_construction": max(
                0.0,
                total - estimate.total_runtime_seconds - fit_timing["all_frame_fit"],
            ),
        },
        refinement_fraction=estimate.refinement_fraction,
        metadata=estimate.metadata,
    )


PRECISION_CONFIGS = {
    "throughput_compact_float64": ("float64", np.float64, np.float64),
    "throughput_compact_float32": ("float32", np.float32, np.float32),
    "throughput_compact_mixed_A": ("float32", np.float64, np.float64),
    "throughput_compact_mixed_B": ("float32", np.float32, np.float64),
}


def run_precision(
    coords: np.ndarray,
    chains: np.ndarray,
    config: str,
) -> StudyResult:
    weight_precision, transform_dtype, reduction_dtype = PRECISION_CONFIGS[config]
    selected = representative_indices(coords.shape[0])
    total_start = time.perf_counter()
    estimate = estimate_shared_weight(
        coords[selected],
        coords[0],
        chain_ids=chains,
        precision=weight_precision,
    )
    if transform_dtype == np.float64 and reduction_dtype == np.float64:
        fit, fit_timing = _fit_float64(coords, estimate.mean_weights)
    else:
        fit, fit_timing = _fit_precision(
            coords,
            estimate.mean_weights,
            transform_dtype=np.dtype(transform_dtype),
            reduction_dtype=np.dtype(reduction_dtype),
        )
    total = time.perf_counter() - total_start
    return StudyResult(
        method=config,
        weights=estimate.mean_weights,
        rotations=fit["rotations"],
        translations=fit["translations"],
        rmsd=fit["rmsd"],
        weighted_rmsd=fit["weighted_rmsd"],
        robust_rmsf=fit["robust_rmsf"],
        mean_residual=fit["mean_residual"],
        runtime_seconds=total,
        timing={
            "weight_estimation": estimate.total_runtime_seconds,
            **fit_timing,
        },
        refinement_fraction=0.0,
        metadata={
            "representative_dtype": weight_precision,
            "transform_dtype": np.dtype(transform_dtype).name,
            "reduction_dtype": np.dtype(reduction_dtype).name,
        },
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    first = 0
    while first < values.size:
        last = first + 1
        while last < values.size and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = 0.5 * (first + last - 1)
        first = last
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(np.asarray(left).ravel())
    right_rank = average_ranks(np.asarray(right).ravel())
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def core_mask(weights: np.ndarray) -> np.ndarray:
    count = max(1, int(np.floor(0.60 * weights.size)))
    order = np.lexsort((np.arange(weights.size), -np.asarray(weights)))
    mask = np.zeros(weights.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def compare(candidate: StudyResult, reference: StudyResult) -> dict[str, object]:
    candidate_core = core_mask(candidate.weights)
    reference_core = core_mask(reference.weights)
    union = np.count_nonzero(candidate_core | reference_core)
    rmsd_difference = np.abs(candidate.rmsd - reference.rmsd)
    finite = bool(
        np.isfinite(candidate.rotations).all()
        and np.isfinite(candidate.translations).all()
        and np.isfinite(candidate.rmsd).all()
        and np.isfinite(candidate.robust_rmsf).all()
    )
    jaccard = float(np.count_nonzero(candidate_core & reference_core) / union)
    p95_rmsd = float(np.quantile(rmsd_difference, 0.95))
    return {
        "core_jaccard": jaccard,
        "weight_spearman": spearman(candidate.weights, reference.weights),
        "rmsd_absolute_difference_median_A": float(np.median(rmsd_difference)),
        "rmsd_absolute_difference_p95_A": p95_rmsd,
        "robust_rmsf_spearman": spearman(candidate.robust_rmsf, reference.robust_rmsf),
        "robust_rmsf_mae_A": float(np.mean(np.abs(candidate.robust_rmsf - reference.robust_rmsf))),
        "flexible_label_agreement": float(
            np.mean((candidate.weights <= np.quantile(candidate.weights, 0.2)) == (reference.weights <= np.quantile(reference.weights, 0.2)))
        ),
        "non_finite_output": not finite,
        "catastrophic_mismatch": bool((not finite) or jaccard < 0.5 or p95_rmsd > 1.0),
        "candidate_runtime_seconds": candidate.runtime_seconds,
        "reference_runtime_seconds": reference.runtime_seconds,
        "speedup_vs_reference": reference.runtime_seconds / candidate.runtime_seconds,
        "refinement_fraction": candidate.refinement_fraction,
    }


def result_row(
    panel_row: dict[str, str],
    candidate: StudyResult,
    reference: StudyResult,
    *,
    split: str,
) -> dict[str, object]:
    return {
        "case_id": panel_row["case_id"],
        "family_id": panel_row["family_id"],
        "split": split,
        "n_frames": int(panel_row["n_frames"]),
        "n_residues": int(panel_row["n_residues"]),
        "configuration": candidate.method,
        **compare(candidate, reference),
        "timing_json": json.dumps(candidate.timing, sort_keys=True),
        "metadata_json": json.dumps(candidate.metadata, sort_keys=True),
    }


CANDIDATE_CONFIGS = [
    "A_independent_batch",
    "B_shared_f64",
    "C_correction_1",
    "C_correction_2",
    "C_correction_4",
    "D_block_8",
    "D_block_16",
    "D_block_32",
    "D_block_64",
    "E_risk_0.80_0",
    "E_risk_0.90_1",
    "E_risk_0.95_1",
]


def phase_audit() -> None:
    rows = read_tsv(OUT / "02_real_panel/full_panel.tsv") + read_tsv(
        OUT / "02_real_panel/challenge_panel.tsv"
    )
    manifest: list[dict[str, object]] = []
    for row in rows:
        path = ROOT / row["input_path"]
        manifest.append(
            {
                "case_id": row["case_id"],
                "input_path": row["input_path"],
                "resolved_path": str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve()),
                "size_bytes": path.resolve().stat().st_size,
                "sha256": sha256_file(path.resolve()),
                "exists": path.exists(),
            }
        )
    write_tsv(OUT / "02_real_panel/input_integrity_manifest.tsv", manifest)


def phase_development() -> None:
    panel = read_tsv(OUT / "02_real_panel/development_panel.tsv")
    precision_path = OUT / "03_precision_compact/development_results.tsv"
    candidate_path = OUT / "04_parallel_weights/development_ablation.tsv"
    precision_path.unlink(missing_ok=True)
    candidate_path.unlink(missing_ok=True)
    for row in panel:
        coords, chains = load_case(row)
        exact = run_exact(coords, chains)
        precision_reference = run_precision(coords, chains, "throughput_compact_float64")
        for config in PRECISION_CONFIGS:
            result = precision_reference if config == "throughput_compact_float64" else run_precision(coords, chains, config)
            append_tsv(
                precision_path,
                result_row(row, result, precision_reference, split="development"),
            )
        for config in CANDIDATE_CONFIGS:
            candidate = run_candidate(coords, chains, config)
            append_tsv(
                candidate_path,
                result_row(row, candidate, exact, split="development"),
            )
        del coords, chains, exact, precision_reference


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _development_candidate_acceptable(rows: list[dict[str, str]]) -> bool:
    return (
        _percentile([float(row["core_jaccard"]) for row in rows], 0.5) >= 0.90
        and min(float(row["core_jaccard"]) for row in rows) >= 0.70
        and _percentile([float(row["weight_spearman"]) for row in rows], 0.5) >= 0.98
        and not any(row["catastrophic_mismatch"] == "True" for row in rows)
    )


def _precision_development_gate(rows: list[dict[str, str]]) -> bool:
    return (
        _percentile([float(row["core_jaccard"]) for row in rows], 0.5) >= 0.995
        and _percentile([float(row["core_jaccard"]) for row in rows], 0.05) >= 0.98
        and min(float(row["core_jaccard"]) for row in rows) >= 0.95
        and _percentile([float(row["weight_spearman"]) for row in rows], 0.5) >= 0.9995
        and _percentile([float(row["rmsd_absolute_difference_median_A"]) for row in rows], 0.5) <= 0.005
        and _percentile([float(row["rmsd_absolute_difference_p95_A"]) for row in rows], 0.95) <= 0.05
        and _percentile([float(row["robust_rmsf_spearman"]) for row in rows], 0.5) >= 0.9995
        and _percentile([float(row["speedup_vs_reference"]) for row in rows], 0.5) >= 1.15
        and not any(row["non_finite_output"] == "True" for row in rows)
    )


def phase_lock() -> None:
    precision = read_tsv(OUT / "03_precision_compact/development_results.tsv")
    candidates = read_tsv(OUT / "04_parallel_weights/development_ablation.tsv")
    selected_precision: list[dict[str, object]] = []
    precision_groups: dict[str, list[dict[str, str]]] = {}
    for row in precision:
        precision_groups.setdefault(row["configuration"], []).append(row)
    eligible_precision = [
        config
        for config, rows in precision_groups.items()
        if config != "throughput_compact_float64" and _precision_development_gate(rows)
    ]
    if eligible_precision:
        winner = max(
            eligible_precision,
            key=lambda config: _percentile(
                [float(row["speedup_vs_reference"]) for row in precision_groups[config]], 0.5
            ),
        )
        selected_precision.append({"family": "precision", "configuration": winner, "selection_status": "locked_before_heldout"})
    else:
        selected_precision.append({"family": "precision", "configuration": "none", "selection_status": "no_development_candidate_passed"})
    write_tsv(OUT / "03_precision_compact/locked_precision_candidate.tsv", selected_precision)

    groups: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        groups.setdefault(row["configuration"], []).append(row)
    family_configs = {
        "A": ["A_independent_batch"],
        "B": ["B_shared_f64"],
        "C": ["C_correction_1", "C_correction_2", "C_correction_4"],
        "D": ["D_block_8", "D_block_16", "D_block_32", "D_block_64"],
        "E": ["E_risk_0.80_0", "E_risk_0.90_1", "E_risk_0.95_1"],
    }
    locked: list[dict[str, object]] = []
    for family, configs in family_configs.items():
        eligible = [config for config in configs if _development_candidate_acceptable(groups[config])]
        if not eligible:
            locked.append({"family": family, "configuration": "none", "selection_status": "failed_development_screen"})
            continue
        winner = max(
            eligible,
            key=lambda config: _percentile(
                [float(row["speedup_vs_reference"]) for row in groups[config]], 0.5
            ),
        )
        locked.append({
            "family": family,
            "configuration": winner,
            "selection_status": "locked_before_heldout",
            "development_median_speedup": _percentile(
                [float(row["speedup_vs_reference"]) for row in groups[winner]], 0.5
            ),
        })
    write_tsv(OUT / "04_parallel_weights/locked_candidate_configs.tsv", locked)


def _locked_configs() -> tuple[list[str], str | None]:
    rows = read_tsv(OUT / "04_parallel_weights/locked_candidate_configs.tsv")
    configs = [row["configuration"] for row in rows if row["configuration"] != "none"]
    precision_rows = read_tsv(OUT / "03_precision_compact/locked_precision_candidate.tsv")
    precision = precision_rows[0]["configuration"]
    return configs, None if precision == "none" else precision


def phase_heldout() -> None:
    configs, precision_config = _locked_configs()
    heldout_path = OUT / "04_parallel_weights/heldout_results.tsv"
    challenge_path = OUT / "04_parallel_weights/challenge_results.tsv"
    precision_quality_path = OUT / "03_precision_compact/heldout_quality.tsv"
    precision_runtime_path = OUT / "03_precision_compact/heldout_runtime.tsv"
    precision_challenge_path = OUT / "03_precision_compact/challenge_cases.tsv"
    for path in (heldout_path, challenge_path, precision_quality_path, precision_runtime_path, precision_challenge_path):
        path.unlink(missing_ok=True)
    panels = (
        (read_tsv(OUT / "02_real_panel/heldout_panel.tsv"), "heldout", heldout_path),
        (read_tsv(OUT / "02_real_panel/challenge_panel.tsv"), "challenge", challenge_path),
    )
    for panel, split, output_path in panels:
        for row in panel:
            try:
                coords, chains = load_case(row)
                exact = run_exact(coords, chains)
                for config in configs:
                    result = run_candidate(coords, chains, config)
                    append_tsv(output_path, result_row(row, result, exact, split=split))
                if precision_config is not None:
                    precision_reference = run_precision(coords, chains, "throughput_compact_float64")
                    precision_result = run_precision(coords, chains, precision_config)
                    precision_row = result_row(row, precision_result, precision_reference, split=split)
                    append_tsv(
                        precision_challenge_path if split == "challenge" else precision_quality_path,
                        precision_row,
                    )
                    if split == "heldout":
                        append_tsv(
                            precision_runtime_path,
                            {
                                "case_id": row["case_id"],
                                "configuration": precision_config,
                                "runtime_seconds": precision_result.runtime_seconds,
                                "float64_runtime_seconds": precision_reference.runtime_seconds,
                                "speedup": precision_reference.runtime_seconds / precision_result.runtime_seconds,
                            },
                        )
            except Exception as error:
                append_tsv(
                    OUT / "06_performance/failures_warnings.tsv",
                    {
                        "phase": split,
                        "case_id": row["case_id"],
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                )
                raise


def _rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _performance_methods() -> list[str]:
    configs, precision = _locked_configs()
    methods = ["exact_sequential_f64_compact", "B_shared_f64"]
    for config in configs:
        if config not in methods:
            methods.append(config)
    if precision is not None:
        methods.append(precision)
    return methods


def _run_named(coords: np.ndarray, chains: np.ndarray, method: str) -> StudyResult:
    if method == "exact_sequential_f64_compact":
        return run_exact(coords, chains)
    if method in PRECISION_CONFIGS:
        return run_precision(coords, chains, method)
    return run_candidate(coords, chains, method)


def phase_performance(repeats: int, warmups: int) -> None:
    methods = _performance_methods()
    panel = read_tsv(OUT / "02_real_panel/full_panel.tsv")
    repeats_path = OUT / "06_performance/real_system_repeats.tsv"
    prefix_path = OUT / "06_performance/prefix_scaling.tsv"
    stage_path = OUT / "06_performance/stage_decomposition.tsv"
    memory_path = OUT / "06_performance/memory.tsv"
    for path in (repeats_path, prefix_path, stage_path, memory_path):
        path.unlink(missing_ok=True)
    for row_index, row in enumerate(panel):
        coords, chains = load_case(row)
        prefixes = sorted(set([min(2000, coords.shape[0]), min(5000, coords.shape[0]), coords.shape[0]]))
        for prefix in prefixes:
            values = np.ascontiguousarray(coords[:prefix])
            # Rotate method order deterministically by system and prefix.
            offset = (row_index + prefixes.index(prefix)) % len(methods)
            ordered_methods = methods[offset:] + methods[:offset]
            for method in ordered_methods:
                for _ in range(warmups):
                    _run_named(values, chains, method)
                for repeat in range(repeats):
                    rss_before = _rss_mib()
                    result = _run_named(values, chains, method)
                    rss_after = _rss_mib()
                    output_row = {
                        "case_id": row["case_id"],
                        "family_id": row["family_id"],
                        "n_frames": prefix,
                        "n_residues": values.shape[1],
                        "full_length": prefix == coords.shape[0],
                        "method": method,
                        "repeat": repeat + 1,
                        "loaded_runtime_seconds": result.runtime_seconds,
                        "fresh_process_runtime_seconds": "not_measured",
                        "method_order_position": ordered_methods.index(method),
                        "cpu_threads": 1,
                        "ordinary_rmsf": False,
                        "aligned_coordinates": False,
                        "refinement_fraction": result.refinement_fraction,
                    }
                    append_tsv(repeats_path, output_row)
                    append_tsv(prefix_path, output_row, sync=False)
                    append_tsv(memory_path, {
                        "case_id": row["case_id"],
                        "n_frames": prefix,
                        "n_residues": values.shape[1],
                        "method": method,
                        "repeat": repeat + 1,
                        "peak_rss_before_mib": rss_before,
                        "peak_rss_after_mib": rss_after,
                    }, sync=False)
                    for stage, seconds in result.timing.items():
                        append_tsv(stage_path, {
                            "case_id": row["case_id"],
                            "n_frames": prefix,
                            "n_residues": values.shape[1],
                            "method": method,
                            "repeat": repeat + 1,
                            "stage": stage,
                            "seconds": seconds,
                        }, sync=False)
        del coords, chains


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    dispatch: dict[str, Callable[[], None]] = {
        "audit": phase_audit,
        "development": phase_development,
        "lock": phase_lock,
        "heldout": phase_heldout,
        "performance": lambda: phase_performance(args.repeats, args.warmups),
    }
    dispatch[args.phase]()


if __name__ == "__main__":
    main()
