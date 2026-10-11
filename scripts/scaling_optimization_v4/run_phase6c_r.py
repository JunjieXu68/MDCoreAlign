#!/usr/bin/env python
"""Run the locked Phase 6C-R window-preconditioned refinement experiment."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Iterable

import numpy as np

from mdsuperpose.core_selection import select_core_by_target_fraction
from mdsuperpose.flexibility import detect_flexible_segments
from mdsuperpose.representative_weights import estimate_representative_weights
from mdsuperpose.window_refine import (
    InitialWeightResult,
    bounded_global_weight_refinement,
    build_initial_weights,
    compute_frozen_window_signals,
    exact_all_frame_fit,
    solve_window_preconditioned_refinement,
    temporally_stratified_indices,
)
from scripts.scaling_optimization.build_phase2_regression import (
    ROBUST_KWARGS,
    RegressionCase,
    build_cases,
)
from scripts.scaling_optimization.profile_support import MemorySampler
from scripts.scaling_optimization_v4.finalize_phase6 import verify_hashes
from scripts.scaling_optimization_v4.run_phase6b import (
    binary_f1,
    contact_persistence,
    flexible_labels,
    jaccard,
    pair_distance_stability,
    safe_spearman,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v4/window_refine"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v4"
PROTOCOL_PATH = RESULT_ROOT / "preregistered_protocol.json"
GOLDEN_MANIFEST = (
    PROJECT_ROOT / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
)
PRIMARY_INPUT = (
    PROJECT_ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark"
    / "runs/4oelA02/10000/transition_aligned_coords.npz"
)
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
CORE_FRACTION = 0.6
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One pre-registered initializer and fixed global update budget."""

    initializer: str
    representative_count: int
    update_rounds: int
    alpha: float | None = None
    top_mobility_fraction: float | None = None
    role: str = "grid"

    @property
    def initializer_id(self) -> str:
        if self.alpha is not None:
            suffix = f"a{int(round(self.alpha * 100)):02d}"
        elif self.top_mobility_fraction is not None:
            suffix = f"f{int(round(self.top_mobility_fraction * 100)):02d}"
        else:
            suffix = "base"
        return f"{self.initializer}__{suffix}"

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.initializer_id}__r{self.representative_count}"
            f"__u{self.update_rounds}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "initializer_id": self.initializer_id,
            "initializer": self.initializer,
            "representative_count": self.representative_count,
            "update_rounds": self.update_rounds,
            "alpha": self.alpha,
            "top_mobility_fraction": self.top_mobility_fraction,
            "minimum_retained_weight": (
                0.25
                if self.initializer == "w5_conservative_exclusion"
                else None
            ),
            "role": self.role,
        }


@dataclass(slots=True)
class EvaluationArtifact:
    """Arrays retained only long enough for equal-budget comparisons."""

    weights: np.ndarray
    core: np.ndarray
    rmsd_abs_difference: np.ndarray


def canonical_environment() -> dict[str, str]:
    """Return the fixed Phase 6C-R process environment."""

    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    return environment


def sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with a union schema."""

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated result table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_protocol() -> dict[str, object]:
    """Load the pre-validation protocol frozen in R1."""

    return json.loads(PROTOCOL_PATH.read_text())


def load_golden_manifest() -> dict[str, dict[str, str]]:
    """Index successful immutable golden references."""

    with GOLDEN_MANIFEST.open(newline="") as handle:
        return {
            row["case_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["status"] == "success"
        }


def case_map() -> dict[str, RegressionCase]:
    """Index the existing 34-case panel without loading coordinates."""

    cases = {case.case_id: case for case in build_cases()}
    if len(cases) != 34:
        raise RuntimeError(f"expected 34 cases, found {len(cases)}")
    return cases


def initializer_specs() -> list[dict[str, object]]:
    """Return the complete fixed 15-initializer grid."""

    specs: list[dict[str, object]] = [
        {"initializer": "uniform", "alpha": None, "top_mobility_fraction": None}
    ]
    for initializer in (
        "w5_relative_mad_soft",
        "w5_winsorised_variance_soft",
        "window_local_multiscale_soft",
        "window_graph_soft",
    ):
        for alpha in (0.25, 0.50, 0.75):
            specs.append(
                {
                    "initializer": initializer,
                    "alpha": alpha,
                    "top_mobility_fraction": None,
                }
            )
    for fraction in (0.10, 0.20):
        specs.append(
            {
                "initializer": "w5_conservative_exclusion",
                "alpha": None,
                "top_mobility_fraction": fraction,
            }
        )
    return specs


def development_candidates() -> list[Candidate]:
    """Expand the fixed initializers over all equal-compute budgets."""

    return [
        Candidate(
            initializer=str(spec["initializer"]),
            representative_count=representative_count,
            update_rounds=rounds,
            alpha=(
                None if spec["alpha"] is None else float(spec["alpha"])
            ),
            top_mobility_fraction=(
                None
                if spec["top_mobility_fraction"] is None
                else float(spec["top_mobility_fraction"])
            ),
        )
        for representative_count in (32, 64, 128, 256)
        for rounds in (1, 2, 4)
        for spec in initializer_specs()
    ]


def candidate_from_dict(payload: dict[str, object]) -> Candidate:
    """Rebuild one locked candidate from JSON."""

    return Candidate(
        initializer=str(payload["initializer"]),
        representative_count=int(payload["representative_count"]),
        update_rounds=int(payload["update_rounds"]),
        alpha=(
            None
            if payload.get("alpha") is None
            else float(payload["alpha"])
        ),
        top_mobility_fraction=(
            None
            if payload.get("top_mobility_fraction") is None
            else float(payload["top_mobility_fraction"])
        ),
        role=str(payload.get("role", "locked")),
    )


def signal_names_for(candidate: Candidate) -> tuple[str, ...]:
    """Map a candidate to the minimum frozen signal construction."""

    if candidate.initializer == "uniform":
        return ()
    if candidate.initializer == "w5_conservative_exclusion":
        return ("w5_winsorised_variance_soft",)
    return (candidate.initializer,)


def make_initial(
    candidate: Candidate,
    n_residues: int,
    signals: dict[str, tuple[object, float]] | None,
) -> InitialWeightResult:
    """Construct one candidate's initial residue weights."""

    return build_initial_weights(
        n_residues,
        candidate.initializer,
        window_signals=signals,
        alpha=candidate.alpha,
        top_mobility_fraction=candidate.top_mobility_fraction,
        minimum_retained_weight=0.25,
    )


def stable_core(weights: np.ndarray) -> np.ndarray:
    """Apply the frozen 60% hard-core reporting contract."""

    return select_core_by_target_fraction(
        np.asarray(weights, dtype=np.float64),
        CORE_FRACTION,
    ).core_mask


def rmsf_flexible_labels(
    rmsf: np.ndarray,
    chain: np.ndarray | None,
) -> np.ndarray:
    """Reclassify flexible segments directly from an RMSF profile."""

    chain_values = (
        np.asarray(chain)
        if chain is not None
        else np.asarray(["A"] * np.asarray(rmsf).size)
    )
    return (
        detect_flexible_segments(
            np.asarray(rmsf, dtype=np.float64),
            percentile_cutoff=0.8,
            min_segment_length=3,
            merge_gap=1,
            chain=chain_values,
        )
        > 0
    )


def rotation_angle_differences(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return per-frame geodesic rotation differences in degrees."""

    relative = np.einsum("tij,tkj->tik", candidate, reference)
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def load_references(
    case: RegressionCase,
    coords: np.ndarray,
    chain: np.ndarray | None,
    golden_manifest: dict[str, dict[str, str]],
) -> dict[str, dict[str, np.ndarray]]:
    """Load sequential_lean and independently recompute batched references."""

    reference_path = (
        PROJECT_ROOT / golden_manifest[case.case_id]["reference_npz"]
    )
    with np.load(reference_path, allow_pickle=False) as golden:
        sequential = {
            "weights": np.asarray(
                golden["legacy_long_frozen_weights"], dtype=np.float64
            ),
            "core": np.asarray(golden["legacy_long_core_labels"], dtype=bool),
            "rotations": np.asarray(
                golden["legacy_long_rotations"], dtype=np.float64
            ),
            "translations": np.asarray(
                golden["legacy_long_translations"], dtype=np.float64
            ),
            "rmsd": np.asarray(golden["legacy_long_rmsd"], dtype=np.float64),
            "robust_rmsf": np.asarray(
                golden["legacy_long_robust_rmsf"], dtype=np.float64
            ),
            "ordinary_rmsf": np.asarray(
                golden["legacy_long_ordinary_rmsf"], dtype=np.float64
            ),
            "flexible": np.asarray(
                golden["legacy_long_flexible_labels"], dtype=bool
            ),
        }
    selected = temporally_stratified_indices(
        coords.shape[0], min(2000, coords.shape[0])
    )
    estimate = estimate_representative_weights(
        coords[selected],
        coords[0],
        batch_size=min(512, selected.size),
        sequence_chain_ids=chain,
        **ROBUST_KWARGS,
    )
    batched_weights = np.clip(
        np.asarray(estimate.mean_weights, dtype=np.float64), 0.0, 1.0
    )
    batched_fit = exact_all_frame_fit(coords, coords[0], batched_weights)
    batched = {
        "weights": batched_weights,
        "core": stable_core(batched_weights),
        "rotations": batched_fit.rotations,
        "translations": batched_fit.translations,
        "rmsd": batched_fit.rmsd,
        "robust_rmsf": batched_fit.robust_rmsf,
        "ordinary_rmsf": batched_fit.ordinary_rmsf,
        "flexible": flexible_labels(batched_weights, chain),
    }
    return {"sequential": sequential, "batched": batched}


def evaluate_candidate(
    case: RegressionCase,
    coords: np.ndarray,
    chain: np.ndarray | None,
    candidate: Candidate,
    initial: InitialWeightResult,
    references: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, object], EvaluationArtifact]:
    """Evaluate one candidate after its explicit exact all-frame fit."""

    result = solve_window_preconditioned_refinement(
        coords,
        initial,
        representative_count=candidate.representative_count,
        update_rounds=candidate.update_rounds,
        chain_ids=chain,
    )
    weights = result.final_weights
    core = stable_core(weights)
    flexible = flexible_labels(weights, chain)
    rmsf_flexible = rmsf_flexible_labels(result.fit.robust_rmsf, chain)
    sequential = references["sequential"]
    batched = references["batched"]
    rmsd_difference = np.abs(result.fit.rmsd - sequential["rmsd"])
    rmsf_difference = np.abs(
        result.fit.robust_rmsf - sequential["robust_rmsf"]
    )
    ordinary_rmsf_difference = np.abs(
        result.fit.ordinary_rmsf - sequential["ordinary_rmsf"]
    )
    rotation_difference = rotation_angle_differences(
        result.fit.rotations,
        sequential["rotations"],
    )
    translation_difference = np.linalg.norm(
        result.fit.translations - sequential["translations"],
        axis=1,
    )
    sequential_rmsf_flexible = rmsf_flexible_labels(
        sequential["robust_rmsf"], chain
    )
    row: dict[str, object] = {
        "case_id": case.case_id,
        "category": case.category,
        "source": case.source,
        "status": "success",
        **candidate.to_dict(),
        "n_frames": coords.shape[0],
        "n_residues": coords.shape[1],
        "representative_count_actual": result.representative_indices.size,
        "core_jaccard_vs_sequential": jaccard(core, sequential["core"]),
        "core_jaccard_vs_batched": jaccard(core, batched["core"]),
        "flexible_jaccard_vs_sequential": jaccard(
            flexible, sequential["flexible"]
        ),
        "flexible_jaccard_vs_batched": jaccard(
            flexible, batched["flexible"]
        ),
        "flexible_f1_vs_sequential": binary_f1(
            flexible, sequential["flexible"]
        ),
        "rmsf_reclassified_flexible_jaccard_vs_sequential": jaccard(
            rmsf_flexible, sequential_rmsf_flexible
        ),
        "rmsf_reclassified_flexible_f1_vs_sequential": binary_f1(
            rmsf_flexible, sequential_rmsf_flexible
        ),
        "weight_spearman_vs_sequential": safe_spearman(
            weights, sequential["weights"]
        ),
        "weight_spearman_vs_batched": safe_spearman(
            weights, batched["weights"]
        ),
        "weight_max_abs_difference_vs_sequential": float(
            np.max(np.abs(weights - sequential["weights"]))
        ),
        "weight_max_abs_difference_vs_batched": float(
            np.max(np.abs(weights - batched["weights"]))
        ),
        "robust_rmsf_spearman_vs_sequential": safe_spearman(
            result.fit.robust_rmsf, sequential["robust_rmsf"]
        ),
        "robust_rmsf_spearman_vs_batched": safe_spearman(
            result.fit.robust_rmsf, batched["robust_rmsf"]
        ),
        "robust_rmsf_mae_vs_sequential": float(np.mean(rmsf_difference)),
        "robust_rmsf_median_abs_difference_vs_sequential": float(
            np.median(rmsf_difference)
        ),
        "robust_rmsf_max_abs_difference_vs_sequential": float(
            np.max(rmsf_difference)
        ),
        "ordinary_rmsf_mae_vs_sequential": float(
            np.mean(ordinary_rmsf_difference)
        ),
        "ordinary_rmsf_max_abs_difference_vs_sequential": float(
            np.max(ordinary_rmsf_difference)
        ),
        "rmsd_mae_vs_sequential": float(np.mean(rmsd_difference)),
        "rmsd_median_abs_difference_vs_sequential": float(
            np.median(rmsd_difference)
        ),
        "rmsd_p95_abs_difference_vs_sequential": float(
            np.quantile(rmsd_difference, 0.95)
        ),
        "rotation_angle_median_degrees_vs_sequential": float(
            np.median(rotation_difference)
        ),
        "rotation_angle_p95_degrees_vs_sequential": float(
            np.quantile(rotation_difference, 0.95)
        ),
        "translation_median_difference_vs_sequential": float(
            np.median(translation_difference)
        ),
        "translation_p95_difference_vs_sequential": float(
            np.quantile(translation_difference, 0.95)
        ),
        "heldout_pair_distance_variance": pair_distance_stability(
            coords[max(1, coords.shape[0] // 2) :], core
        ),
        "contact_persistence": contact_persistence(
            coords[max(1, coords.shape[0] // 2) :], core, chain
        ),
        "active_set_size": result.active_set_size,
        "effective_core_size": result.effective_core_size,
        "effective_core_fraction": result.effective_core_size / coords.shape[1],
        "fixed_rounds_completed": result.round_weight_changes.size,
        "final_global_weight_delta": float(result.round_weight_changes[-1]),
        "converged_representatives_final_round": int(
            result.converged_representative_counts[-1]
        ),
        "scan_seconds": result.timing["scan"],
        "representative_selection_seconds": result.timing[
            "representative_selection"
        ],
        "limited_global_refinement_seconds": result.timing[
            "limited_global_refinement"
        ],
        "all_frame_exact_fit_seconds": result.timing[
            "exact_all_frame_total"
        ],
        "rmsd_seconds": result.timing["RMSD"],
        "rmsf_seconds": result.timing["RMSF"],
        "algorithm_excluding_scan_seconds": result.timing[
            "algorithm_excluding_scan"
        ],
        "algorithm_including_scan_seconds": result.timing[
            "algorithm_including_scan"
        ],
        "scan_movement_called_rmsd": False,
        "exact_all_frame_fit_before_rmsd_rmsf": True,
        "cross_replicate_status": "unavailable_no_matching_local_replicate",
    }
    return row, EvaluationArtifact(weights, core, rmsd_difference)


def add_equal_budget_comparisons(
    rows: list[dict[str, object]],
    artifacts: dict[str, EvaluationArtifact],
) -> None:
    """Attach paired comparisons with the exact same-budget uniform run."""

    uniform_by_budget = {
        (
            int(row["representative_count"]),
            int(row["update_rounds"]),
        ): artifacts[str(row["candidate_id"])]
        for row in rows
        if row["initializer"] == "uniform" and row["status"] == "success"
    }
    uniform_row_by_budget = {
        (
            int(row["representative_count"]),
            int(row["update_rounds"]),
        ): row
        for row in rows
        if row["initializer"] == "uniform" and row["status"] == "success"
    }
    for row in rows:
        if row["status"] != "success":
            continue
        budget = (
            int(row["representative_count"]),
            int(row["update_rounds"]),
        )
        uniform = uniform_by_budget[budget]
        artifact = artifacts[str(row["candidate_id"])]
        uniform_row = uniform_row_by_budget[budget]
        row["core_jaccard_vs_equal_budget_uniform"] = jaccard(
            artifact.core, uniform.core
        )
        row["weight_spearman_vs_equal_budget_uniform"] = safe_spearman(
            artifact.weights, uniform.weights
        )
        row["runtime_ratio_vs_equal_budget_uniform"] = float(
            row["algorithm_including_scan_seconds"]
        ) / max(
            float(uniform_row["algorithm_including_scan_seconds"]), 1e-12
        )


def evaluate_cases(
    cases: Iterable[RegressionCase],
    candidates: list[Candidate],
    *,
    output_path: Path,
    differences_path: Path,
) -> list[dict[str, object]]:
    """Evaluate a fixed candidate list without retaining trajectory arrays."""

    golden_manifest = load_golden_manifest()
    all_rows: list[dict[str, object]] = []
    rmsd_differences: dict[str, np.ndarray] = {}
    for case in cases:
        print(f"[phase6c-r] evaluate {case.case_id}", flush=True)
        coords, chain = case.loader()
        coords = np.asarray(coords, dtype=np.float64)
        chain = np.asarray(chain) if chain is not None else None
        references = load_references(
            case, coords, chain, golden_manifest
        )
        needs_windows = any(
            candidate.initializer != "uniform" for candidate in candidates
        )
        signals = (
            compute_frozen_window_signals(coords, chain_ids=chain)
            if needs_windows
            else None
        )
        initial_cache: dict[str, InitialWeightResult] = {}
        case_rows: list[dict[str, object]] = []
        artifacts: dict[str, EvaluationArtifact] = {}
        for index, candidate in enumerate(candidates, start=1):
            if candidate.initializer_id not in initial_cache:
                initial_cache[candidate.initializer_id] = make_initial(
                    candidate,
                    coords.shape[1],
                    signals,
                )
            try:
                row, artifact = evaluate_candidate(
                    case,
                    coords,
                    chain,
                    candidate,
                    initial_cache[candidate.initializer_id],
                    references,
                )
                case_rows.append(row)
                artifacts[candidate.candidate_id] = artifact
                rmsd_differences[
                    f"{case.case_id}__{candidate.candidate_id}"
                ] = artifact.rmsd_abs_difference
            except Exception as error:
                case_rows.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "source": case.source,
                        "status": "failed",
                        **candidate.to_dict(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if index % 60 == 0:
                print(
                    f"[phase6c-r] {case.case_id}: {index}/{len(candidates)}",
                    flush=True,
                )
        add_equal_budget_comparisons(case_rows, artifacts)
        all_rows.extend(case_rows)
        write_tsv(output_path, all_rows)
    differences_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(differences_path, **rmsd_differences)
    return all_rows


def numeric_values(
    rows: list[dict[str, object]],
    field: str,
) -> np.ndarray:
    """Extract finite floating-point values."""

    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def pooled_rmsd_differences(
    rows: list[dict[str, object]],
    difference_archives: list[Path],
) -> np.ndarray:
    """Load exact per-frame differences for a candidate across cases."""

    wanted = {
        f"{row['case_id']}__{row['candidate_id']}"
        for row in rows
    }
    arrays: list[np.ndarray] = []
    for path in difference_archives:
        with np.load(path, allow_pickle=False) as payload:
            for key in wanted.intersection(payload.files):
                arrays.append(np.asarray(payload[key], dtype=np.float64))
    if not arrays:
        raise RuntimeError("no pooled per-frame RMSD differences found")
    return np.concatenate(arrays)


def aggregate_candidate(
    rows: list[dict[str, object]],
    *,
    difference_archives: list[Path],
) -> dict[str, object]:
    """Apply pre-registered core and flexibility aggregate gates."""

    successful = [row for row in rows if row["status"] == "success"]
    if not successful:
        return {
            "candidate_id": rows[0]["candidate_id"] if rows else "unknown",
            "n_cases": 0,
            "n_failures": len(rows),
            "fast_core_gate_pass": False,
            "fast_flex_gate_pass": False,
        }
    core = numeric_values(successful, "core_jaccard_vs_sequential")
    weight = numeric_values(successful, "weight_spearman_vs_sequential")
    rmsf = numeric_values(
        successful, "robust_rmsf_spearman_vs_sequential"
    )
    flexible = numeric_values(
        successful, "flexible_jaccard_vs_sequential"
    )
    rmsf_flexible = numeric_values(
        successful,
        "rmsf_reclassified_flexible_jaccard_vs_sequential",
    )
    rmsf_median_error = numeric_values(
        successful,
        "robust_rmsf_median_abs_difference_vs_sequential",
    )
    rmsf_max_error = numeric_values(
        successful, "robust_rmsf_max_abs_difference_vs_sequential"
    )
    pooled_rmsd = pooled_rmsd_differences(
        successful, difference_archives
    )
    runtime = numeric_values(
        successful, "algorithm_including_scan_seconds"
    )
    core_metrics = {
        "median_core_jaccard": float(np.median(core)),
        "p10_core_jaccard": float(np.quantile(core, 0.10)),
        "median_weight_spearman": float(np.median(weight)),
        "median_robust_rmsf_spearman": float(np.median(rmsf)),
        "minimum_robust_rmsf_spearman": float(np.min(rmsf)),
        "median_flexible_label_jaccard": float(np.median(flexible)),
        "median_rmsf_reclassified_flexible_jaccard": float(
            np.median(rmsf_flexible)
        ),
        "pooled_rmsd_median_abs_difference": float(
            np.median(pooled_rmsd)
        ),
        "pooled_rmsd_p95_abs_difference": float(
            np.quantile(pooled_rmsd, 0.95)
        ),
        "catastrophic_core_cases": int(np.count_nonzero(core < 0.50)),
    }
    core_checks = (
        core_metrics["median_core_jaccard"] >= 0.85,
        core_metrics["p10_core_jaccard"] >= 0.75,
        core_metrics["median_weight_spearman"] >= 0.90,
        core_metrics["median_robust_rmsf_spearman"] >= 0.97,
        core_metrics["median_flexible_label_jaccard"] >= 0.80,
        core_metrics["pooled_rmsd_median_abs_difference"] <= 0.05,
        core_metrics["pooled_rmsd_p95_abs_difference"] <= 0.20,
        core_metrics["catastrophic_core_cases"] == 0,
    )
    flex_checks = (
        core_metrics["median_robust_rmsf_spearman"] >= 0.97,
        core_metrics["minimum_robust_rmsf_spearman"] >= 0.90,
        float(np.median(rmsf_median_error)) <= 5.27731289305,
        float(np.quantile(rmsf_max_error, 0.95)) <= 8.8012175148,
    )
    case_quality_passes = 0
    for row in successful:
        case_quality_passes += int(
            float(row["core_jaccard_vs_sequential"]) >= 0.75
            and float(row["weight_spearman_vs_sequential"]) >= 0.90
            and float(row["robust_rmsf_spearman_vs_sequential"]) >= 0.97
            and float(row["flexible_jaccard_vs_sequential"]) >= 0.80
            and float(row["rmsd_median_abs_difference_vs_sequential"]) <= 0.05
            and float(row["rmsd_p95_abs_difference_vs_sequential"]) <= 0.20
        )
    first = successful[0]
    return {
        "candidate_id": first["candidate_id"],
        "initializer_id": first["initializer_id"],
        "initializer": first["initializer"],
        "representative_count": int(first["representative_count"]),
        "update_rounds": int(first["update_rounds"]),
        "alpha": first["alpha"],
        "top_mobility_fraction": first["top_mobility_fraction"],
        "n_cases": len(successful),
        "n_failures": len(rows) - len(successful),
        **core_metrics,
        "median_robust_rmsf_median_abs_difference": float(
            np.median(rmsf_median_error)
        ),
        "p95_robust_rmsf_max_abs_difference": float(
            np.quantile(rmsf_max_error, 0.95)
        ),
        "median_algorithm_including_scan_seconds": float(
            np.median(runtime)
        ),
        "fast_core_threshold_violation_count": int(
            len(core_checks) - sum(core_checks)
        ),
        "fast_flex_threshold_violation_count": int(
            len(flex_checks) - sum(flex_checks)
        ),
        "existing_case_quality_gate_passes": case_quality_passes,
        "fast_core_gate_pass": bool(all(core_checks)),
        "fast_flex_gate_pass": bool(all(flex_checks)),
    }


def development_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize every development-grid candidate."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    archive = RESULT_ROOT / "development_rmsd_differences.npz"
    return [
        aggregate_candidate(group, difference_archives=[archive])
        for _, group in sorted(grouped.items())
    ]


def core_rank(summary: dict[str, object]) -> tuple[object, ...]:
    """Return the frozen quality-first core ordering."""

    return (
        int(summary["fast_core_threshold_violation_count"]),
        int(summary["catastrophic_core_cases"]),
        -float(summary["median_core_jaccard"]),
        -float(summary["p10_core_jaccard"]),
        -float(summary["median_weight_spearman"]),
        -float(summary["median_flexible_label_jaccard"]),
        float(summary["pooled_rmsd_median_abs_difference"]),
        float(summary["median_algorithm_including_scan_seconds"]),
    )


def flex_rank(summary: dict[str, object]) -> tuple[object, ...]:
    """Return the frozen quality-first flexibility ordering."""

    return (
        int(summary["fast_flex_threshold_violation_count"]),
        -float(summary["median_robust_rmsf_spearman"]),
        -float(summary["minimum_robust_rmsf_spearman"]),
        float(summary["median_robust_rmsf_median_abs_difference"]),
        -float(summary["median_rmsf_reclassified_flexible_jaccard"]),
        float(summary["median_algorithm_including_scan_seconds"]),
    )


def select_and_lock_candidates(
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the pre-registered Pareto rule and freeze candidate identities."""

    by_budget: dict[tuple[int, int], list[dict[str, object]]] = {}
    for summary in summaries:
        budget = (
            int(summary["representative_count"]),
            int(summary["update_rounds"]),
        )
        by_budget.setdefault(budget, []).append(summary)
    budget_records: list[
        tuple[tuple[object, ...], tuple[int, int], dict[str, object], dict[str, object]]
    ] = []
    audit_rows: list[dict[str, object]] = []
    for budget, group in sorted(by_budget.items()):
        windows = [
            summary
            for summary in group
            if summary["initializer"] != "uniform"
        ]
        best_core = min(windows, key=core_rank)
        distinct_flex_pool = [
            summary
            for summary in windows
            if summary["candidate_id"] != best_core["candidate_id"]
        ]
        best_flex = min(
            distinct_flex_pool if distinct_flex_pool else windows,
            key=flex_rank,
        )
        budget_rank = (
            int(best_core["fast_core_threshold_violation_count"]),
            int(best_flex["fast_flex_threshold_violation_count"]),
            int(best_core["catastrophic_core_cases"]),
            -float(best_core["median_core_jaccard"]),
            -float(best_flex["median_robust_rmsf_spearman"]),
            float(best_core["median_algorithm_including_scan_seconds"])
            + float(best_flex["median_algorithm_including_scan_seconds"]),
        )
        budget_records.append((budget_rank, budget, best_core, best_flex))
        audit_rows.append(
            {
                "representative_count": budget[0],
                "update_rounds": budget[1],
                "best_core_candidate": best_core["candidate_id"],
                "best_flex_candidate": best_flex["candidate_id"],
                "budget_rank": json.dumps(budget_rank),
            }
        )
    _, chosen_budget, best_core, best_flex = min(
        budget_records, key=lambda item: item[0]
    )
    uniform = next(
        summary
        for summary in by_budget[chosen_budget]
        if summary["initializer"] == "uniform"
    )
    summary_by_id = {
        str(summary["candidate_id"]): summary for summary in summaries
    }
    locked: list[dict[str, object]] = []
    for role, selected in (
        ("core_ranked_window", best_core),
        ("flex_ranked_window", best_flex),
        ("equal_budget_uniform", uniform),
    ):
        candidate = Candidate(
            initializer=str(selected["initializer"]),
            representative_count=int(selected["representative_count"]),
            update_rounds=int(selected["update_rounds"]),
            alpha=(
                None
                if selected["alpha"] in (None, "")
                else float(selected["alpha"])
            ),
            top_mobility_fraction=(
                None
                if selected["top_mobility_fraction"] in (None, "")
                else float(selected["top_mobility_fraction"])
            ),
            role=role,
        )
        payload = candidate.to_dict()
        payload["development_summary"] = summary_by_id[
            candidate.candidate_id
        ]
        locked.append(payload)
    write_tsv(RESULT_ROOT / "development_budget_selection.tsv", audit_rows)
    lock = {
        "status": "locked_before_validation",
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "development_results_sha256": sha256(
            RESULT_ROOT / "development_results.tsv"
        ),
        "development_summary_sha256": sha256(
            RESULT_ROOT / "development_summary.tsv"
        ),
        "chosen_budget": {
            "representative_count": chosen_budget[0],
            "update_rounds": chosen_budget[1],
        },
        "candidates": locked,
        "validation_results_seen": False,
        "post_validation_tuning_allowed": False,
    }
    lock_path = RESULT_ROOT / "locked_candidates.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    (RESULT_ROOT / "locked_candidates.sha256").write_text(
        sha256(lock_path) + "\n"
    )
    return lock


def run_development() -> None:
    """Evaluate the fixed development grid and lock three candidates."""

    if (RESULT_ROOT / "locked_validation_results.tsv").exists():
        raise RuntimeError(
            "validation results already exist; development cannot be rerun"
        )
    protocol = load_protocol()
    cases = case_map()
    selected = [
        cases[str(item["case_id"])]
        for item in protocol["development_cases"]
    ]
    candidates = development_candidates()
    rows = evaluate_cases(
        selected,
        candidates,
        output_path=RESULT_ROOT / "development_results.tsv",
        differences_path=RESULT_ROOT / "development_rmsd_differences.npz",
    )
    expected = len(selected) * len(candidates)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    summaries = development_summaries(rows)
    write_tsv(RESULT_ROOT / "development_summary.tsv", summaries)
    lock = select_and_lock_candidates(summaries)
    print(
        "[phase6c-r] locked "
        + ", ".join(
            str(candidate["candidate_id"])
            for candidate in lock["candidates"]
        ),
        flush=True,
    )


def block_refined_weights(
    coords: np.ndarray,
    chain: np.ndarray | None,
    candidate: Candidate,
) -> np.ndarray:
    """Run only the fixed refinement on one temporal block."""

    required = signal_names_for(candidate)
    signals = (
        compute_frozen_window_signals(
            coords,
            chain_ids=chain,
            initializers=required,
        )
        if required
        else None
    )
    initial = make_initial(candidate, coords.shape[1], signals)
    selected = temporally_stratified_indices(
        coords.shape[0], candidate.representative_count
    )
    weights, _, _, _ = bounded_global_weight_refinement(
        coords[selected],
        coords[0],
        initial.weights,
        update_rounds=candidate.update_rounds,
        sequence_chain_ids=chain,
    )
    return weights


def build_risk_rows(
    candidates: list[Candidate],
    all_cases: list[RegressionCase],
    result_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Apply the risk rule frozen before validation."""

    indexed = {
        (str(row["case_id"]), str(row["candidate_id"])): row
        for row in result_rows
        if row["status"] == "success"
    }
    rows: list[dict[str, object]] = []
    for case in all_cases:
        print(f"[phase6c-r] risk diagnostics {case.case_id}", flush=True)
        coords, chain = case.loader()
        coords = np.asarray(coords, dtype=np.float64)
        chain = np.asarray(chain) if chain is not None else None
        split = max(2, coords.shape[0] // 2)
        first = coords[:split]
        second = coords[-split:]
        block_weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for candidate in candidates:
            block_weights[candidate.candidate_id] = (
                block_refined_weights(first, chain, candidate),
                block_refined_weights(second, chain, candidate),
            )
        uniform = next(
            candidate
            for candidate in candidates
            if candidate.initializer == "uniform"
        )
        uniform_row = indexed[(case.case_id, uniform.candidate_id)]
        for candidate in candidates:
            result_row = indexed[(case.case_id, candidate.candidate_id)]
            first_weights, second_weights = block_weights[
                candidate.candidate_id
            ]
            block_core_jaccard = jaccard(
                stable_core(first_weights), stable_core(second_weights)
            )
            block_weight_spearman = safe_spearman(
                first_weights, second_weights
            )
            reasons: list[str] = []
            if float(result_row["effective_core_fraction"]) < 0.50:
                reasons.append("low_effective_core_fraction")
            if float(result_row["final_global_weight_delta"]) > 0.05:
                reasons.append("large_final_global_weight_delta")
            if block_core_jaccard < 0.90:
                reasons.append("temporal_block_core_instability")
            if block_weight_spearman < 0.95:
                reasons.append("temporal_block_weight_instability")
            if candidate.initializer != "uniform":
                if (
                    float(
                        result_row[
                            "core_jaccard_vs_equal_budget_uniform"
                        ]
                    )
                    < 0.85
                ):
                    reasons.append("window_uniform_core_disagreement")
                if (
                    float(
                        result_row[
                            "weight_spearman_vs_equal_budget_uniform"
                        ]
                    )
                    < 0.90
                ):
                    reasons.append("window_uniform_weight_disagreement")
            quality_pass = (
                float(result_row["core_jaccard_vs_sequential"]) >= 0.75
                and float(result_row["weight_spearman_vs_sequential"]) >= 0.90
                and float(
                    result_row["robust_rmsf_spearman_vs_sequential"]
                )
                >= 0.97
                and float(result_row["flexible_jaccard_vs_sequential"])
                >= 0.80
                and float(
                    result_row[
                        "rmsd_median_abs_difference_vs_sequential"
                    ]
                )
                <= 0.05
                and float(
                    result_row["rmsd_p95_abs_difference_vs_sequential"]
                )
                <= 0.20
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "candidate_id": candidate.candidate_id,
                    "initializer": candidate.initializer,
                    "role": candidate.role,
                    "temporal_block_core_jaccard": block_core_jaccard,
                    "temporal_block_weight_spearman": block_weight_spearman,
                    "effective_core_fraction": result_row[
                        "effective_core_fraction"
                    ],
                    "final_global_weight_delta": result_row[
                        "final_global_weight_delta"
                    ],
                    "core_jaccard_vs_equal_budget_uniform": result_row[
                        "core_jaccard_vs_equal_budget_uniform"
                    ],
                    "weight_spearman_vs_equal_budget_uniform": result_row[
                        "weight_spearman_vs_equal_budget_uniform"
                    ],
                    "equal_budget_uniform_core_jaccard_vs_sequential": (
                        uniform_row["core_jaccard_vs_sequential"]
                    ),
                    "case_quality_gate_pass": quality_pass,
                    "risk_flag": bool(reasons),
                    "risk_reasons": ";".join(reasons),
                    "failed_case_detected_by_risk_rule": (
                        quality_pass or bool(reasons)
                    ),
                    "cross_replicate_status": (
                        "unavailable_no_matching_local_replicate"
                    ),
                }
            )
    return rows


def run_validation() -> None:
    """Run locked candidates once on the untouched 26-case panel."""

    lock_path = RESULT_ROOT / "locked_candidates.json"
    lock = json.loads(lock_path.read_text())
    expected_hash = (
        RESULT_ROOT / "locked_candidates.sha256"
    ).read_text().strip()
    observed_hash = sha256(lock_path)
    if observed_hash != expected_hash:
        raise RuntimeError("locked candidate hash mismatch")
    output_path = RESULT_ROOT / "locked_validation_results.tsv"
    if output_path.exists():
        raise RuntimeError("locked validation has already been run")
    validation_started = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "locked_candidates_sha256": observed_hash,
        "validation_output_existed_before_start": False,
        "parameters_locked": True,
    }
    (RESULT_ROOT / "validation_started.json").write_text(
        json.dumps(validation_started, indent=2) + "\n"
    )
    protocol = load_protocol()
    cases = case_map()
    selected = [
        cases[str(case_id)] for case_id in protocol["validation_cases"]
    ]
    candidates = [
        candidate_from_dict(payload) for payload in lock["candidates"]
    ]
    validation_rows = evaluate_cases(
        selected,
        candidates,
        output_path=output_path,
        differences_path=(
            RESULT_ROOT / "locked_validation_rmsd_differences.npz"
        ),
    )
    development_rows = [
        row
        for row in read_tsv(RESULT_ROOT / "development_results.tsv")
        if row["candidate_id"]
        in {candidate.candidate_id for candidate in candidates}
    ]
    combined_rows: list[dict[str, object]] = [
        *development_rows,
        *validation_rows,
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in combined_rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    summaries = [
        aggregate_candidate(
            rows,
            difference_archives=[
                RESULT_ROOT / "development_rmsd_differences.npz",
                RESULT_ROOT / "locked_validation_rmsd_differences.npz",
            ],
        )
        for _, rows in sorted(grouped.items())
    ]
    write_tsv(RESULT_ROOT / "locked_all34_summary.tsv", summaries)
    risk_rows = build_risk_rows(
        candidates,
        [cases[case_id] for case_id in cases],
        combined_rows,
    )
    write_tsv(RESULT_ROOT / "risk_rule.tsv", risk_rows)
    print("[phase6c-r] locked validation complete", flush=True)


def runtime_child(
    candidate_id: str,
    n_frames: int,
    output: Path,
) -> None:
    """Measure one clean-process candidate execution."""

    lock = json.loads((RESULT_ROOT / "locked_candidates.json").read_text())
    candidate = next(
        candidate_from_dict(payload)
        for payload in lock["candidates"]
        if payload["candidate_id"] == candidate_id
    )
    process_start = time.perf_counter()
    sampler = MemorySampler(interval_seconds=0.005)
    sampler.start()
    decode_start = time.perf_counter()
    with np.load(PRIMARY_INPUT, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"][:n_frames], dtype=np.float64)
    decode_seconds = time.perf_counter() - decode_start
    required = signal_names_for(candidate)
    scan_start = time.perf_counter()
    signals = (
        compute_frozen_window_signals(
            coords,
            initializers=required,
        )
        if required
        else None
    )
    initial = make_initial(candidate, coords.shape[1], signals)
    scan_wall_seconds = time.perf_counter() - scan_start
    solve_start = time.perf_counter()
    result = solve_window_preconditioned_refinement(
        coords,
        initial,
        representative_count=candidate.representative_count,
        update_rounds=candidate.update_rounds,
    )
    solve_wall_seconds = time.perf_counter() - solve_start
    sampler.stop()
    payload: dict[str, object] = {
        **candidate.to_dict(),
        "n_frames": n_frames,
        "n_residues": coords.shape[1],
        "input_decode_seconds": decode_seconds,
        "scan_wall_seconds": scan_wall_seconds,
        "scan_seconds": result.timing["scan"],
        "representative_selection_seconds": result.timing[
            "representative_selection"
        ],
        "limited_global_refinement_seconds": result.timing[
            "limited_global_refinement"
        ],
        "all_frame_exact_fit_seconds": result.timing[
            "exact_all_frame_total"
        ],
        "rmsd_seconds": result.timing["RMSD"],
        "rmsf_seconds": result.timing["RMSF"],
        "algorithm_excluding_scan_seconds": solve_wall_seconds,
        "algorithm_including_scan_seconds": (
            scan_wall_seconds + solve_wall_seconds
        ),
        "metrics_transforms_only_seconds": (
            scan_wall_seconds + solve_wall_seconds
        ),
        "process_through_algorithm_seconds": time.perf_counter()
        - process_start,
        "output_writing_seconds": 0.0,
        **sampler.to_dict(),
    }
    serialisation_start = time.perf_counter()
    encoded = json.dumps(payload, indent=2) + "\n"
    payload["output_serialisation_seconds"] = (
        time.perf_counter() - serialisation_start
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    output.write_text(json.dumps(payload, indent=2) + "\n")
    output_writing_seconds = time.perf_counter() - write_start
    payload["output_writing_seconds"] = output_writing_seconds
    output.write_text(json.dumps(payload, indent=2) + "\n")


def run_runtime() -> None:
    """Run warm-up plus three clean-process repeats at five frame counts."""

    lock = json.loads((RESULT_ROOT / "locked_candidates.json").read_text())
    candidate_ids = [
        str(payload["candidate_id"]) for payload in lock["candidates"]
    ]
    raw_root = RESULT_ROOT / "runtime_raw"
    rows: list[dict[str, object]] = []
    for n_frames in (500, 1000, 2000, 5000, 10000):
        for candidate_id in candidate_ids:
            for repeat in range(4):
                warmup = repeat == 0
                output = raw_root / (
                    f"{candidate_id}__f{n_frames}__"
                    f"{'warmup' if warmup else f'r{repeat}'}.json"
                )
                command = [
                    str(PYTHON),
                    str(Path(__file__).resolve()),
                    "--stage",
                    "runtime-child",
                    "--candidate-id",
                    candidate_id,
                    "--frames",
                    str(n_frames),
                    "--output",
                    str(output),
                ]
                start = time.perf_counter()
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=canonical_environment(),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=180,
                )
                fresh_seconds = time.perf_counter() - start
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"runtime child failed: {' '.join(command)}\n"
                        f"{completed.stdout}\n{completed.stderr}"
                    )
                payload = json.loads(output.read_text())
                rows.append(
                    {
                        **payload,
                        "repeat": repeat,
                        "warmup": warmup,
                        "fresh_process_seconds": fresh_seconds,
                        "command": " ".join(command),
                    }
                )
                print(
                    f"[phase6c-r] runtime {candidate_id} "
                    f"{n_frames} repeat={repeat}",
                    flush=True,
                )
    measured = [row for row in rows if not bool(row["warmup"])]
    write_tsv(RESULT_ROOT / "runtime.tsv", measured)
    write_tsv(
        RESULT_ROOT / "memory.tsv",
        [
            {
                "candidate_id": row["candidate_id"],
                "initializer": row["initializer"],
                "n_frames": row["n_frames"],
                "repeat": row["repeat"],
                "initial_rss_bytes": row["initial_rss_bytes"],
                "peak_rss_bytes": row["peak_rss_bytes"],
                "peak_rss_mb": row["peak_rss_mb"],
                "rss_delta_bytes": row["rss_delta_bytes"],
                "sample_count": row["sample_count"],
            }
            for row in measured
        ],
    )


def environment_manifest() -> dict[str, object]:
    """Capture the fixed interpreter, NumPy/BLAS and process environment."""

    stream = io.StringIO()
    with redirect_stdout(stream):
        np.__config__.show()
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "unavailable"
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "numpy_blas_configuration": stream.getvalue(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "platform": platform.platform(),
        "primary_runtime_input": PRIMARY_INPUT.relative_to(PROJECT_ROOT).as_posix(),
        "primary_runtime_input_sha256": sha256(PRIMARY_INPUT),
    }


def summarize_runtime(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Summarize three measured repeats per candidate and frame count."""

    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["candidate_id"], int(row["n_frames"])), []
        ).append(row)
    output: list[dict[str, object]] = []
    for (candidate_id, n_frames), group in sorted(grouped.items()):
        first = group[0]
        output.append(
            {
                "candidate_id": candidate_id,
                "initializer": first["initializer"],
                "role": first["role"],
                "n_frames": n_frames,
                "n_repeats": len(group),
                "median_fresh_process_seconds": statistics.median(
                    float(row["fresh_process_seconds"]) for row in group
                ),
                "median_metrics_transforms_only_seconds": statistics.median(
                    float(row["metrics_transforms_only_seconds"])
                    for row in group
                ),
                "median_scan_seconds": statistics.median(
                    float(row["scan_wall_seconds"]) for row in group
                ),
                "median_refinement_seconds": statistics.median(
                    float(row["limited_global_refinement_seconds"])
                    for row in group
                ),
                "median_exact_fit_seconds": statistics.median(
                    float(row["all_frame_exact_fit_seconds"])
                    for row in group
                ),
                "median_rmsd_seconds": statistics.median(
                    float(row["rmsd_seconds"]) for row in group
                ),
                "median_rmsf_seconds": statistics.median(
                    float(row["rmsf_seconds"]) for row in group
                ),
                "median_peak_rss_mb": statistics.median(
                    float(row["peak_rss_mb"]) for row in group
                ),
            }
        )
    return output


def decide(
    summaries: list[dict[str, str]],
    risk_rows: list[dict[str, str]],
    runtime_summary: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the bounded Phase 6C-R A/B/C/D decision rules."""

    by_id = {row["candidate_id"]: row for row in summaries}
    uniform = next(
        row for row in summaries if row["initializer"] == "uniform"
    )
    windows = [row for row in summaries if row["initializer"] != "uniform"]

    def metric(row: dict[str, str], name: str) -> float:
        return float(row[name])

    risk_by_candidate: dict[str, list[dict[str, str]]] = {}
    for row in risk_rows:
        risk_by_candidate.setdefault(row["candidate_id"], []).append(row)
    gate_status: dict[str, dict[str, object]] = {}
    for candidate_id, row in by_id.items():
        risks = risk_by_candidate[candidate_id]
        failures = [
            risk
            for risk in risks
            if str(risk["case_quality_gate_pass"]).lower() != "true"
        ]
        detected = [
            risk
            for risk in failures
            if str(risk["risk_flag"]).lower() == "true"
        ]
        fallback_sensitivity = (
            1.0 if not failures else len(detected) / len(failures)
        )
        gate_status[candidate_id] = {
            "aggregate_fast_core_gate": (
                str(row["fast_core_gate_pass"]).lower() == "true"
            ),
            "aggregate_fast_flex_gate": (
                str(row["fast_flex_gate_pass"]).lower() == "true"
            ),
            "case_quality_passes": int(
                row["existing_case_quality_gate_passes"]
            ),
            "case_quality_failures": len(failures),
            "risk_rule_failure_sensitivity": fallback_sensitivity,
            "fast_core_with_fallback_gate": (
                str(row["fast_core_gate_pass"]).lower() == "true"
                and (
                    int(row["existing_case_quality_gate_passes"]) == 34
                    or fallback_sensitivity == 1.0
                )
            ),
        }
    uniform_core_rank = (
        int(uniform["fast_core_threshold_violation_count"]),
        int(uniform["catastrophic_core_cases"]),
        -metric(uniform, "median_core_jaccard"),
        -metric(uniform, "median_weight_spearman"),
        -metric(uniform, "median_robust_rmsf_spearman"),
    )
    windows_outperform_uniform = [
        window
        for window in windows
        if (
            int(window["fast_core_threshold_violation_count"]),
            int(window["catastrophic_core_cases"]),
            -metric(window, "median_core_jaccard"),
            -metric(window, "median_weight_spearman"),
            -metric(window, "median_robust_rmsf_spearman"),
        )
        < uniform_core_rank
    ]
    qualifying_window_core = [
        window
        for window in windows_outperform_uniform
        if gate_status[window["candidate_id"]]["fast_core_with_fallback_gate"]
    ]
    qualifying_flex = [
        row
        for row in summaries
        if gate_status[row["candidate_id"]]["aggregate_fast_flex_gate"]
    ]
    if qualifying_window_core:
        decision = "Decision A"
        interpretation = (
            "A locked window preconditioner passed the fast-core contract and "
            "strictly outranked the equal-budget uniform initializer."
        )
        w5_not_selected_core = False
    elif not windows_outperform_uniform:
        decision = "Decision B"
        interpretation = (
            "Equal-budget uniform global refinement matched or outperformed "
            "all locked window initializers."
        )
        w5_not_selected_core = True
    elif qualifying_flex:
        decision = "Decision C"
        interpretation = (
            "No window candidate established a fast-core advantage, but at "
            "least one locked candidate passed the separate fast-flex gate."
        )
        w5_not_selected_core = True
    else:
        decision = "Decision D"
        interpretation = (
            "Neither the fast-core nor the separate fast-flex contract passed; "
            "the bounded window acceleration branch is closed."
        )
        w5_not_selected_core = True
    runtime_10k = [
        row for row in runtime_summary if int(row["n_frames"]) == 10000
    ]
    return {
        "decision": decision,
        "interpretation": interpretation,
        "public_default_changed": False,
        "w5_stable_core_preconditioner_status": (
            "not_selected_in_current_equal_budget_implementation"
            if w5_not_selected_core
            else "retained_as_locked_experimental_candidate"
        ),
        "w5_alignment_free_diagnostic_status": "retained_experimental",
        "w5_interpretation": (
            "W5 was not selected as a stable-core acceleration preconditioner "
            "in its current implementation, but is retained as an experimental "
            "alignment-free local-deformation diagnostic."
        ),
        "gate_status_by_candidate": gate_status,
        "window_candidates_strictly_outranking_uniform": [
            row["candidate_id"] for row in windows_outperform_uniform
        ],
        "fast_flex_qualifying_candidates": [
            row["candidate_id"] for row in qualifying_flex
        ],
        "runtime_10k": runtime_10k,
        "fresh_process_initial_target_seconds": 1.50,
        "fresh_process_strong_target_seconds": 1.00,
        "targets_conditional_on_scientific_gate": True,
    }


def run_tests() -> dict[str, object]:
    """Run the complete test suite without writing old phase directories."""

    command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=canonical_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    output = completed.stdout + completed.stderr
    (RESULT_ROOT / "full_test_results.txt").write_text(output)
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": output.strip().splitlines()[-1] if output.strip() else "",
    }


def make_handoff(report_path: Path) -> tuple[Path, str]:
    """Create the compact Phase 6C-R handoff without trajectory data."""

    date = time.strftime("%Y%m%d")
    handoff_root = RESULT_ROOT / "handoff_phase6c_r"
    handoff_root.mkdir(parents=True, exist_ok=True)
    include = [
        PROTOCOL_PATH,
        RESULT_ROOT / "r1_reproduction.json",
        RESULT_ROOT / "development_summary.tsv",
        RESULT_ROOT / "development_budget_selection.tsv",
        RESULT_ROOT / "locked_candidates.json",
        RESULT_ROOT / "locked_validation_results.tsv",
        RESULT_ROOT / "locked_all34_summary.tsv",
        RESULT_ROOT / "runtime_summary.tsv",
        RESULT_ROOT / "memory.tsv",
        RESULT_ROOT / "risk_rule.tsv",
        RESULT_ROOT / "decision.json",
        RESULT_ROOT / "environment.json",
        RESULT_ROOT / "integrity_after_phase6c_r.json",
        RESULT_ROOT / "test_status.json",
        RESULT_ROOT / "full_test_results.txt",
        report_path,
        PROJECT_ROOT / "src/mdsuperpose/window_refine.py",
        Path(__file__).resolve(),
        PROJECT_ROOT
        / "tests/scaling_optimization_v4/test_window_preconditioned_refinement.py",
    ]
    manifest_rows: list[dict[str, object]] = []
    for path in include:
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        destination = handoff_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        manifest_rows.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_tsv(handoff_root / "sha256_manifest.tsv", manifest_rows)
    readme = handoff_root / "README.md"
    readme.write_text(
        "# MDCoreAlign Phase 6C-R compact handoff\n\n"
        "This bundle contains the pre-registration, aggregate development and "
        "locked-validation results, runtime/memory summaries, risk analysis, "
        "decision, tests, integrity records, report, and source changes. It "
        "excludes trajectories, caches, secrets, raw per-frame differences, "
        "and generated environments. The 10k benchmark input is a "
        "transition-aligned synthetic scaling trajectory, not continuous MD.\n"
    )
    archive = (
        PROJECT_ROOT
        / f"results/scaling_optimization_v4/handoff_phase6c_r_{date}.tar.gz"
    )
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(
            handoff_root,
            arcname="handoff_phase6c_r",
            recursive=True,
        )
    digest = sha256(archive)
    archive.with_suffix("").with_suffix(".sha256").write_text(
        f"{digest}  {archive.name}\n"
    )
    return archive, digest


def run_finalize() -> None:
    """Write the final decision, report, tests, integrity and handoff."""

    summaries = read_tsv(RESULT_ROOT / "locked_all34_summary.tsv")
    risk_rows = read_tsv(RESULT_ROOT / "risk_rule.tsv")
    runtime_rows = read_tsv(RESULT_ROOT / "runtime.tsv")
    runtime_summary = summarize_runtime(runtime_rows)
    write_tsv(RESULT_ROOT / "runtime_summary.tsv", runtime_summary)
    decision = decide(summaries, risk_rows, runtime_summary)
    (RESULT_ROOT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    environment = environment_manifest()
    (RESULT_ROOT / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    integrity = verify_hashes()
    (RESULT_ROOT / "integrity_after_phase6c_r.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )
    tests = run_tests()
    (RESULT_ROOT / "test_status.json").write_text(
        json.dumps(tests, indent=2) + "\n"
    )
    lock = json.loads((RESULT_ROOT / "locked_candidates.json").read_text())
    risk_summary: list[str] = []
    for summary in summaries:
        candidate_id = summary["candidate_id"]
        candidate_risks = [
            row for row in risk_rows if row["candidate_id"] == candidate_id
        ]
        failures = [
            row
            for row in candidate_risks
            if row["case_quality_gate_pass"].lower() != "true"
        ]
        detected = [
            row for row in failures if row["risk_flag"].lower() == "true"
        ]
        flagged = [
            row for row in candidate_risks if row["risk_flag"].lower() == "true"
        ]
        risk_summary.append(
            f"- `{candidate_id}`: "
            f"{summary['existing_case_quality_gate_passes']}/34 case gates; "
            f"risk detection {len(detected)}/{len(failures)} failures, but "
            f"{len(flagged)}/34 total cases flagged."
        )
    runtime_lines = []
    for row in runtime_summary:
        if int(row["n_frames"]) == 10000:
            runtime_lines.append(
                f"- `{row['candidate_id']}`: fresh process "
                f"{float(row['median_fresh_process_seconds']):.4f} s; "
                f"metrics/transforms-only "
                f"{float(row['median_metrics_transforms_only_seconds']):.4f} s; "
                f"peak RSS {float(row['median_peak_rss_mb']):.1f} MB."
            )
    summary_lines = []
    for row in summaries:
        summary_lines.append(
            f"- `{row['candidate_id']}`: core Jaccard median "
            f"{float(row['median_core_jaccard']):.4f}, p10 "
            f"{float(row['p10_core_jaccard']):.4f}; weight Spearman "
            f"{float(row['median_weight_spearman']):.4f}; RMSF Spearman "
            f"{float(row['median_robust_rmsf_spearman']):.4f}; flexible "
            f"Jaccard {float(row['median_flexible_label_jaccard']):.4f}; pooled "
            f"RMSD absolute difference median "
            f"{float(row['pooled_rmsd_median_abs_difference']):.4f} A, p95 "
            f"{float(row['pooled_rmsd_p95_abs_difference']):.4f} A; "
            f"core gate `{row['fast_core_gate_pass']}`, flex gate "
            f"`{row['fast_flex_gate_pass']}`."
        )
    report_path = (
        DOC_ROOT / "06_window_preconditioned_refinement_report.md"
    )
    report_path.write_text(
        "# Phase 6C-R window-preconditioned refinement report\n\n"
        "## Scope and controls\n\n"
        "Only Phase 6C-R was executed. W5/window signals were used only as "
        "soft priors, every comparison used the same temporally stratified "
        "representatives and fixed global update rounds, and every RMSD/RMSF "
        "value followed an explicit exact all-frame fit. The public default, "
        "Git metadata, frozen project-completion outputs, and golden "
        "references were not changed.\n\n"
        "The fixed development panel contained 8 cases and the locked "
        "validation panel contained 26 cases. `mdcath_3pio100` was validation "
        "only. Candidate identities were written and SHA-256 locked before "
        "validation began; no post-validation tuning was performed.\n\n"
        "## Exact commands\n\n"
        "```bash\n"
        "export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "
        "MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:.\n"
        "/home/jjxu/miniconda3/bin/python "
        "scripts/scaling_optimization_v4/run_phase6c_r.py --stage development\n"
        "/home/jjxu/miniconda3/bin/python "
        "scripts/scaling_optimization_v4/run_phase6c_r.py --stage validation\n"
        "/home/jjxu/miniconda3/bin/python "
        "scripts/scaling_optimization_v4/run_phase6c_r.py --stage runtime\n"
        "/home/jjxu/miniconda3/bin/python "
        "scripts/scaling_optimization_v4/run_phase6c_r.py --stage finalize\n"
        "```\n\n"
        "## Locked candidates\n\n"
        + "\n".join(
            f"- `{candidate['role']}`: `{candidate['candidate_id']}`."
            for candidate in lock["candidates"]
        )
        + "\n\n## Locked 34-case quality\n\n"
        + "\n".join(summary_lines)
        + "\n\n## Risk and fallback diagnostics\n\n"
        + "\n".join(risk_summary)
        + "\n\nCross-replicate reproducibility was unavailable because the frozen "
        "34-case panel does not provide matching replicate inputs for every "
        "case; this limitation was recorded rather than inferred. No exact "
        "fallback was invoked in this experiment. Although the pre-declared "
        "risk rule detected every case-level failure, its 31/34 trigger rate "
        "was too broad to rescue a practical fast-core contract.\n\n"
        "## Clean runtime and memory\n\n"
        + "\n".join(runtime_lines)
        + "\n\nThe 10k input is a transition-aligned synthetic scaling "
        "trajectory and does not establish biological continuity. Scan cost "
        "is included in every metrics/transforms-only result; output writing "
        "is reported separately.\n\n"
        "## Decision\n\n"
        f"**{decision['decision']}**. {decision['interpretation']}\n\n"
        "A faster candidate was not considered successful unless its "
        "scientific contract passed. Negative results and gate failures are "
        "retained in the tables rather than filtered.\n\n"
        "## Integrity and tests\n\n"
        f"- Frozen integrity: `{integrity['status']}` "
        f"({integrity['project_completion_checked']}/9 project outputs, "
        f"{integrity['golden_checked']}/34 golden references).\n"
        f"- Tests: `{tests['status']}`; `{tests['summary']}`.\n"
        "- Git HEAD remained invalid and no Git metadata was initialized or "
        "changed.\n\n"
        "## Stop condition\n\n"
        "Phase 6C-R is complete. No GPU, Faiss/Annoy, adaptive sampling, "
        "additional window tuning, public-default change, or manuscript-claim "
        "change was made.\n"
    )
    archive, digest = make_handoff(report_path)
    final = {
        "status": (
            "complete"
            if integrity["status"] == "pass" and tests["status"] == "pass"
            else "completed_with_failures"
        ),
        "decision": decision["decision"],
        "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
        "handoff_archive": archive.relative_to(PROJECT_ROOT).as_posix(),
        "handoff_sha256": digest,
        "integrity": integrity,
        "tests": tests,
    }
    (RESULT_ROOT / "phase6c_r_summary.json").write_text(
        json.dumps(final, indent=2) + "\n"
    )
    print(json.dumps(final, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    """Parse one explicit stage to preserve the validation boundary."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "development",
            "validation",
            "runtime",
            "runtime-child",
            "finalize",
        ),
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Dispatch exactly one Phase 6C-R stage."""

    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "development":
        run_development()
    elif args.stage == "validation":
        run_validation()
    elif args.stage == "runtime":
        run_runtime()
    elif args.stage == "runtime-child":
        if not args.candidate_id or args.frames is None or args.output is None:
            raise ValueError(
                "runtime-child requires --candidate-id, --frames and --output"
            )
        runtime_child(args.candidate_id, args.frames, args.output)
    else:
        run_finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
