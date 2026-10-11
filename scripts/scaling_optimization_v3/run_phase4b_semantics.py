"""Adjudicate scientific effects of sequential representative warm-start."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v3/semantic"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v3"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts/scaling_optimization"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_phase2_regression import ROBUST_KWARGS, build_cases  # noqa: E402
from mdsuperpose.alignment import RigidTransform  # noqa: E402
from mdsuperpose.core_selection import select_core_by_target_fraction  # noqa: E402
from mdsuperpose.fast_kabsch import weighted_kabsch_batch  # noqa: E402
from mdsuperpose.flexibility import compute_rmsf, detect_flexible_segments  # noqa: E402
from mdsuperpose.representative_weights import estimate_representative_weights  # noqa: E402
from mdsuperpose.robust import solve_iterative_robust_alignment  # noqa: E402


CORE_FRACTION = 0.6
MAX_HOLDOUT_TRAIN = 500
MAX_HOLDOUT_EVAL = 500
STRICT_NONPARITY = {
    "mdcath_1ss3A00",
    "mdcath_5h1sc00",
    "mdcath_3pio100",
    "mdcath_2pbiA03",
    "mdcath_1b35D00",
    "mdcath_3d9xA01",
    "mdcath_1bctA00",
    "real_4a18O00",
}
REPLICATE_ROOT = PROJECT_ROOT / "results/mdcath_benchmark/replicate_stitching"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.sum(first | second))
    return float(np.sum(first & second) / union) if union else 1.0


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return 1.0 if np.allclose(first, second) else 0.0
    value = float(spearmanr(first, second).statistic)
    return value if np.isfinite(value) else 0.0


def representative_indices(n_frames: int, maximum: int = 2000) -> np.ndarray:
    count = min(maximum, n_frames)
    return np.unique(np.linspace(0, n_frames - 1, count, dtype=int))


def orderings(count: int) -> list[tuple[str, np.ndarray]]:
    base = np.arange(count, dtype=int)
    interleaved = np.empty(count, dtype=int)
    left = 0
    right = count - 1
    for index in range(count):
        if index % 2 == 0:
            interleaved[index] = left
            left += 1
        else:
            interleaved[index] = right
            right -= 1
    values = [
        ("chronological", base),
        ("reverse", base[::-1]),
        ("interleaved", interleaved),
    ]
    for seed in range(10):
        values.append(
            (
                f"permuted_seed_{seed}",
                np.random.default_rng(4100 + seed).permutation(count),
            )
        )
    return values


def _weight_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-15, 1.0 - 1e-15)
    return float(np.mean(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))))


def estimate_scalar(
    representatives: np.ndarray,
    reference: np.ndarray,
    order: np.ndarray,
    *,
    warm_start: bool,
    trace_rows: list[dict[str, object]] | None = None,
    trace_label: str = "",
) -> dict[str, object]:
    n_representatives, n_residues, _ = representatives.shape
    weights = np.empty((n_representatives, n_residues), dtype=float)
    rmsd = np.empty(n_representatives, dtype=float)
    weighted_rmsd = np.empty(n_representatives, dtype=float)
    iterations = np.empty(n_representatives, dtype=int)
    converged = np.empty(n_representatives, dtype=bool)
    previous_weights: np.ndarray | None = None
    previous_transform: RigidTransform | None = None
    start = time.perf_counter()
    for sequence_index, representative_index in enumerate(order):
        callback = None
        if trace_rows is not None:
            def callback(
                payload: dict[str, object],
                *,
                sequence_index: int = sequence_index,
                representative_index: int = int(representative_index),
            ) -> None:
                trace_rows.append(
                    {
                        "order": trace_label,
                        "sequence_index": sequence_index,
                        "representative_index": representative_index,
                        **payload,
                    }
                )

        result = solve_iterative_robust_alignment(
            representatives[int(representative_index)],
            reference,
            initial_weights=previous_weights if warm_start else None,
            initial_transform=previous_transform if warm_start else None,
            diagnostic_callback=callback,
            **ROBUST_KWARGS,
        )
        weights[int(representative_index)] = result.weights
        rmsd[int(representative_index)] = result.rmsd
        weighted_rmsd[int(representative_index)] = result.weighted_rmsd
        iterations[int(representative_index)] = result.iteration_count
        converged[int(representative_index)] = result.converged
        if warm_start:
            previous_weights = result.weights
            previous_transform = RigidTransform(
                rotation=result.rotation,
                translation=result.translation,
            )
    mean_weights = np.mean(weights, axis=0)
    return {
        "weights": weights,
        "mean_weights": mean_weights,
        "rmsd": rmsd,
        "weighted_rmsd": weighted_rmsd,
        "iterations": iterations,
        "converged": converged,
        "runtime_seconds": time.perf_counter() - start,
    }


def estimate_batched(
    representatives: np.ndarray,
    reference: np.ndarray,
) -> dict[str, object]:
    start = time.perf_counter()
    result = estimate_representative_weights(
        representatives,
        reference,
        batch_size=min(512, representatives.shape[0]),
        warm_start_refinement_passes=0,
        **ROBUST_KWARGS,
    )
    return {
        "weights": result.weights,
        "mean_weights": result.mean_weights,
        "rmsd": result.final_rmsd,
        "weighted_rmsd": result.final_weighted_rmsd,
        "iterations": result.iteration_counts,
        "converged": result.converged,
        "runtime_seconds": time.perf_counter() - start,
    }


def evaluate_weights(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    fit = weighted_kabsch_batch(
        coords,
        reference,
        weights,
        validate_inputs=False,
        compute_metrics=True,
    )
    ordinary_fit = weighted_kabsch_batch(
        coords,
        reference,
        np.ones(reference.shape[0], dtype=float),
        validate_inputs=False,
        compute_metrics=False,
    )
    core = select_core_by_target_fraction(weights, CORE_FRACTION).core_mask
    robust_rmsf = compute_rmsf(fit.aligned_coords)
    ordinary_rmsf = compute_rmsf(ordinary_fit.aligned_coords)
    flexible = detect_flexible_segments(
        np.clip(1.0 - weights, 0.0, 1.0),
        percentile_cutoff=0.8,
        min_segment_length=3,
        merge_gap=1,
    ) > 0
    core_pair_variance, contact_persistence = alignment_independent_core_metrics(
        coords, reference, core
    )
    return {
        "core": core,
        "flexible": flexible,
        "robust_rmsf": robust_rmsf,
        "ordinary_rmsf": ordinary_rmsf,
        "mean_full_rmsd": float(np.mean(fit.rmsd)),
        "mean_core_fit_rmsd": float(np.mean(fit.weighted_rmsd)),
        "pairwise_core_distance_variance": core_pair_variance,
        "core_contact_persistence": contact_persistence,
    }


def alignment_independent_core_metrics(
    coords: np.ndarray,
    reference: np.ndarray,
    core: np.ndarray,
) -> tuple[float, float]:
    indices = np.flatnonzero(core)
    if indices.shape[0] < 2:
        return float("nan"), float("nan")
    first, second = np.triu_indices(indices.shape[0], k=1)
    if first.shape[0] > 2000:
        chosen = np.linspace(0, first.shape[0] - 1, 2000, dtype=int)
        first = first[chosen]
        second = second[chosen]
    atom_first = indices[first]
    atom_second = indices[second]
    distances = np.linalg.norm(
        coords[:, atom_first] - coords[:, atom_second],
        axis=2,
    )
    pair_variance = float(np.mean(np.var(distances, axis=0)))
    sequence_separation = np.abs(atom_first - atom_second)
    reference_distances = np.linalg.norm(
        reference[atom_first] - reference[atom_second],
        axis=1,
    )
    contacts = (reference_distances <= 8.0) & (sequence_separation > 2)
    if not np.any(contacts):
        persistence = float("nan")
    else:
        persistence = float(np.mean(distances[:, contacts] <= 8.0))
    return pair_variance, persistence


def method_summary(
    estimate: dict[str, object],
    evaluation: dict[str, object],
) -> dict[str, object]:
    mean_weights = np.asarray(estimate["mean_weights"], dtype=float)
    return {
        "runtime_seconds": estimate["runtime_seconds"],
        "mean_representative_rmsd": float(np.mean(estimate["rmsd"])),
        "mean_representative_weighted_rmsd": float(
            np.mean(estimate["weighted_rmsd"])
        ),
        "converged_fraction": float(np.mean(estimate["converged"])),
        "median_iterations": float(np.median(estimate["iterations"])),
        "weight_bound_fraction": float(
            np.mean((estimate["weights"] <= 1e-12) | (estimate["weights"] >= 1 - 1e-12))
        ),
        "aggregate_weight_entropy": _weight_entropy(mean_weights),
        "effective_core_size": float(np.sum(mean_weights)),
        "hard_core_size": int(np.sum(evaluation["core"])),
        "core_segment_count": segment_count(evaluation["core"]),
        "flexible_segment_count": segment_count(evaluation["flexible"]),
        "mean_full_rmsd": evaluation["mean_full_rmsd"],
        "mean_core_fit_rmsd": evaluation["mean_core_fit_rmsd"],
        "mean_robust_rmsf": float(np.mean(evaluation["robust_rmsf"])),
        "mean_ordinary_rmsf": float(np.mean(evaluation["ordinary_rmsf"])),
        "robust_vs_ordinary_rmsf_spearman": safe_spearman(
            evaluation["robust_rmsf"], evaluation["ordinary_rmsf"]
        ),
        "pairwise_core_distance_variance": evaluation[
            "pairwise_core_distance_variance"
        ],
        "core_contact_persistence": evaluation["core_contact_persistence"],
    }


def segment_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    return int(np.sum(mask & ~np.r_[False, mask[:-1]]))


def order_sensitivity() -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    list[dict[str, object]],
]:
    rows: list[dict[str, object]] = []
    focused: dict[str, dict[str, object]] = {}
    trace_rows: list[dict[str, object]] = []
    for case in build_cases():
        print(f"[orders] {case.case_id}", flush=True)
        coords, chain = case.loader()
        selected = representative_indices(coords.shape[0])
        representatives = coords[selected]
        reference = coords[0]
        chronological: dict[str, object] | None = None
        chronological_eval: dict[str, object] | None = None
        legacy_estimates: dict[str, dict[str, object]] = {}
        for label, order in orderings(representatives.shape[0]):
            traces = trace_rows if case.case_id == "mdcath_3pio100" and label in {
                "chronological",
                "reverse",
            } else None
            estimate = estimate_scalar(
                representatives,
                reference,
                order,
                warm_start=True,
                trace_rows=traces,
                trace_label=label,
            )
            evaluation = evaluate_weights(coords, reference, estimate["mean_weights"])
            if label == "chronological":
                chronological = estimate
                chronological_eval = evaluation
            assert chronological is not None and chronological_eval is not None
            summary = method_summary(estimate, evaluation)
            row = {
                "case_id": case.case_id,
                "category": case.category,
                "strict_phase3_parity_case": int(
                    case.case_id not in STRICT_NONPARITY
                ),
                "n_frames": coords.shape[0],
                "n_residues": coords.shape[1],
                "n_representatives": representatives.shape[0],
                "method": "legacy_object",
                "representative_order": label,
                "weight_max_abs_difference_vs_legacy_chronological": float(
                    np.max(
                        np.abs(
                            estimate["mean_weights"]
                            - chronological["mean_weights"]
                        )
                    )
                ),
                "weight_spearman_vs_legacy_chronological": safe_spearman(
                    estimate["mean_weights"], chronological["mean_weights"]
                ),
                "core_jaccard_vs_legacy_chronological": jaccard(
                    evaluation["core"], chronological_eval["core"]
                ),
                "flexible_jaccard_vs_legacy_chronological": jaccard(
                    evaluation["flexible"], chronological_eval["flexible"]
                ),
                "terminal_fraction": min(1.0, 10.0 / coords.shape[1]),
                **summary,
            }
            rows.append(row)
            legacy_estimates[label] = {
                "estimate": estimate,
                "evaluation": evaluation,
            }

        scalar = estimate_scalar(
            representatives,
            reference,
            np.arange(representatives.shape[0]),
            warm_start=False,
        )
        batched = estimate_batched(representatives, reference)
        for method, estimate in (
            ("scalar_independent", scalar),
            ("batched_independent", batched),
        ):
            evaluation = evaluate_weights(coords, reference, estimate["mean_weights"])
            assert chronological is not None and chronological_eval is not None
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "strict_phase3_parity_case": int(
                        case.case_id not in STRICT_NONPARITY
                    ),
                    "n_frames": coords.shape[0],
                    "n_residues": coords.shape[1],
                    "n_representatives": representatives.shape[0],
                    "method": method,
                    "representative_order": "independent_initialisation",
                    "weight_max_abs_difference_vs_legacy_chronological": float(
                        np.max(
                            np.abs(
                                estimate["mean_weights"]
                                - chronological["mean_weights"]
                            )
                        )
                    ),
                    "weight_spearman_vs_legacy_chronological": safe_spearman(
                        estimate["mean_weights"], chronological["mean_weights"]
                    ),
                    "core_jaccard_vs_legacy_chronological": jaccard(
                        evaluation["core"], chronological_eval["core"]
                    ),
                    "flexible_jaccard_vs_legacy_chronological": jaccard(
                        evaluation["flexible"], chronological_eval["flexible"]
                    ),
                    "terminal_fraction": min(1.0, 10.0 / coords.shape[1]),
                    **method_summary(estimate, evaluation),
                }
            )
            if case.case_id == "mdcath_3pio100":
                focused[method] = {
                    "estimate": estimate,
                    "evaluation": evaluation,
                }
        if case.case_id == "mdcath_3pio100":
            focused["legacy_orders"] = legacy_estimates
            focused["selected_indices"] = selected
            focused["coords"] = coords
            focused["chain"] = chain
    return rows, focused, trace_rows


def infer_for_method(
    coords: np.ndarray,
    reference: np.ndarray,
    method: str,
) -> dict[str, object]:
    selected = representative_indices(coords.shape[0], MAX_HOLDOUT_TRAIN)
    representatives = coords[selected]
    if method == "legacy_object":
        return estimate_scalar(
            representatives,
            reference,
            np.arange(representatives.shape[0]),
            warm_start=True,
        )
    if method == "scalar_independent":
        return estimate_scalar(
            representatives,
            reference,
            np.arange(representatives.shape[0]),
            warm_start=False,
        )
    if method == "batched_independent":
        return estimate_batched(representatives, reference)
    raise ValueError(method)


def holdout_analysis() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in build_cases():
        coords, _ = case.loader()
        if coords.shape[0] < 20:
            continue
        midpoint = coords.shape[0] // 2
        halves = (("first_to_second", coords[:midpoint], coords[midpoint:]),
                  ("second_to_first", coords[midpoint:], coords[:midpoint]))
        method_outputs: dict[tuple[str, str], tuple[dict[str, object], dict[str, object]]] = {}
        for split, training, heldout in halves:
            training = training[
                representative_indices(training.shape[0], MAX_HOLDOUT_TRAIN)
            ]
            heldout = heldout[
                representative_indices(heldout.shape[0], MAX_HOLDOUT_EVAL)
            ]
            reference = training[0]
            for method in (
                "legacy_object",
                "scalar_independent",
                "batched_independent",
            ):
                estimate = infer_for_method(training, reference, method)
                evaluation = evaluate_weights(
                    heldout, reference, estimate["mean_weights"]
                )
                method_outputs[(split, method)] = (estimate, evaluation)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "split": split,
                        "method": method,
                        "training_frames": training.shape[0],
                        "heldout_frames": heldout.shape[0],
                        "n_residues": coords.shape[1],
                        **method_summary(estimate, evaluation),
                    }
                )
        for method in (
            "legacy_object",
            "scalar_independent",
            "batched_independent",
        ):
            first_estimate, first_eval = method_outputs[("first_to_second", method)]
            second_estimate, second_eval = method_outputs[("second_to_first", method)]
            for row in rows[-6:]:
                if row["method"] == method:
                    row["swap_half_weight_spearman"] = safe_spearman(
                        first_estimate["mean_weights"],
                        second_estimate["mean_weights"],
                    )
                    row["swap_half_core_jaccard"] = jaccard(
                        first_eval["core"], second_eval["core"]
                    )
                    row["swap_half_flexible_jaccard"] = jaccard(
                        first_eval["flexible"], second_eval["flexible"]
                    )
        print(f"[holdout] {case.case_id}", flush=True)
    return rows


def cross_replicate_analysis() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_id in ("2rdeA02", "1vq8L01", "1b4aA01"):
        root = REPLICATE_ROOT / domain_id / "replicates"
        replicates: dict[str, np.ndarray] = {}
        for label in ("R1", "R2", "R3"):
            path = root / f"{label}_ca_coords.npz"
            with np.load(path, allow_pickle=False) as payload:
                replicates[label] = np.asarray(payload["coords"], dtype=float)
        estimates: dict[tuple[str, str], dict[str, object]] = {}
        for source in ("R1", "R2"):
            for method in (
                "legacy_object",
                "scalar_independent",
                "batched_independent",
            ):
                estimates[(source, method)] = infer_for_method(
                    replicates[source],
                    replicates[source][0],
                    method,
                )
        for source in ("R1", "R2"):
            for method in (
                "legacy_object",
                "scalar_independent",
                "batched_independent",
            ):
                estimate = estimates[(source, method)]
                for target in ("R2", "R3"):
                    if target == source:
                        continue
                    evaluation = evaluate_weights(
                        replicates[target],
                        replicates[source][0],
                        estimate["mean_weights"],
                    )
                    other_source = "R2" if source == "R1" else "R1"
                    other = estimates[(other_source, method)]
                    other_evaluation = evaluate_weights(
                        replicates[target],
                        replicates[other_source][0],
                        other["mean_weights"],
                    )
                    rows.append(
                        {
                            "domain_id": domain_id,
                            "temperature_K": 320,
                            "source_replicate": source,
                            "target_replicate": target,
                            "method": method,
                            "source_frames": replicates[source].shape[0],
                            "target_frames": replicates[target].shape[0],
                            "n_residues": replicates[source].shape[1],
                            "cross_source_weight_spearman": safe_spearman(
                                estimate["mean_weights"],
                                other["mean_weights"],
                            ),
                            "cross_source_core_jaccard": jaccard(
                                evaluation["core"], other_evaluation["core"]
                            ),
                            "cross_source_rmsf_spearman": safe_spearman(
                                evaluation["robust_rmsf"],
                                other_evaluation["robust_rmsf"],
                            ),
                            "cross_source_flexible_jaccard": jaccard(
                                evaluation["flexible"],
                                other_evaluation["flexible"],
                            ),
                            **method_summary(estimate, evaluation),
                        }
                    )
        print(f"[replicate] {domain_id}", flush=True)
    return rows


def subset_analysis() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in build_cases():
        coords, _ = case.loader()
        count = min(500, max(10, coords.shape[0] // 2))
        count = min(count, coords.shape[0])
        reference = coords[0]
        baseline_indices = np.unique(
            np.linspace(0, coords.shape[0] - 1, count, dtype=int)
        )
        for method in (
            "legacy_object",
            "scalar_independent",
            "batched_independent",
        ):
            baseline_estimate = infer_subset(
                coords[baseline_indices], reference, method
            )
            baseline_eval = evaluate_weights(
                coords, reference, baseline_estimate["mean_weights"]
            )
            for seed in range(5):
                rng = np.random.default_rng(7300 + seed)
                indices = np.sort(
                    rng.choice(coords.shape[0], size=count, replace=False)
                )
                estimate = infer_subset(coords[indices], reference, method)
                evaluation = evaluate_weights(
                    coords, reference, estimate["mean_weights"]
                )
                rows.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "method": method,
                        "subset_seed": seed,
                        "subset_count": count,
                        "n_frames": coords.shape[0],
                        "n_residues": coords.shape[1],
                        "weight_spearman_vs_uniform_subset": safe_spearman(
                            estimate["mean_weights"],
                            baseline_estimate["mean_weights"],
                        ),
                        "core_jaccard_vs_uniform_subset": jaccard(
                            evaluation["core"], baseline_eval["core"]
                        ),
                        "flexible_jaccard_vs_uniform_subset": jaccard(
                            evaluation["flexible"], baseline_eval["flexible"]
                        ),
                        "mean_full_rmsd_delta_vs_uniform": float(
                            evaluation["mean_full_rmsd"]
                            - baseline_eval["mean_full_rmsd"]
                        ),
                        "mean_core_fit_rmsd_delta_vs_uniform": float(
                            evaluation["mean_core_fit_rmsd"]
                            - baseline_eval["mean_core_fit_rmsd"]
                        ),
                    }
                )
        print(f"[subset] {case.case_id}", flush=True)
    return rows


def infer_subset(
    representatives: np.ndarray,
    reference: np.ndarray,
    method: str,
) -> dict[str, object]:
    if method == "legacy_object":
        return estimate_scalar(
            representatives,
            reference,
            np.arange(representatives.shape[0]),
            warm_start=True,
        )
    if method == "scalar_independent":
        return estimate_scalar(
            representatives,
            reference,
            np.arange(representatives.shape[0]),
            warm_start=False,
        )
    return estimate_batched(representatives, reference)


def write_3pio_outputs(
    focused: dict[str, dict[str, object]],
    trace_rows: list[dict[str, object]],
) -> None:
    legacy_orders = focused["legacy_orders"]
    chronological = legacy_orders["chronological"]
    reverse = legacy_orders["reverse"]
    permuted = [
        value
        for label, value in legacy_orders.items()
        if label.startswith("permuted")
    ]
    permutation_weights = np.asarray(
        [value["estimate"]["mean_weights"] for value in permuted]
    )
    scalar = focused["scalar_independent"]
    batched = focused["batched_independent"]
    coords = focused["coords"]
    rows: list[dict[str, object]] = []
    for index in range(coords.shape[1]):
        rows.append(
            {
                "residue_index_zero_based": index,
                "residue_id": index + 1,
                "chain": "A",
                "annotation_status": "topology_annotations_unavailable",
                "terminal_residue": int(index < 5 or index >= coords.shape[1] - 5),
                "legacy_chronological_weight": chronological["estimate"][
                    "mean_weights"
                ][index],
                "legacy_reverse_weight": reverse["estimate"]["mean_weights"][index],
                "legacy_permutation_min_weight": float(
                    np.min(permutation_weights[:, index])
                ),
                "legacy_permutation_max_weight": float(
                    np.max(permutation_weights[:, index])
                ),
                "scalar_independent_weight": scalar["estimate"]["mean_weights"][
                    index
                ],
                "batched_independent_weight": batched["estimate"]["mean_weights"][
                    index
                ],
                "legacy_chronological_robust_rmsf": chronological["evaluation"][
                    "robust_rmsf"
                ][index],
                "legacy_reverse_robust_rmsf": reverse["evaluation"]["robust_rmsf"][
                    index
                ],
                "batched_independent_robust_rmsf": batched["evaluation"][
                    "robust_rmsf"
                ][index],
                "legacy_core_label": int(chronological["evaluation"]["core"][index]),
                "batched_core_label": int(batched["evaluation"]["core"][index]),
                "legacy_flexible_label": int(
                    chronological["evaluation"]["flexible"][index]
                ),
                "batched_flexible_label": int(
                    batched["evaluation"]["flexible"][index]
                ),
            }
        )
    write_tsv(
        RESULT_ROOT / "per_residue/mdcath_3pio100_per_residue.tsv",
        rows,
    )
    write_tsv(RESULT_ROOT / "mdcath_3pio100_iteration_traces.tsv", trace_rows)


def summarize_evidence(
    order_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> tuple[str, dict[str, float]]:
    legacy_order = [
        row
        for row in order_rows
        if row["method"] == "legacy_object"
        and row["representative_order"] != "chronological"
    ]
    order_core_min = min(
        float(row["core_jaccard_vs_legacy_chronological"]) for row in legacy_order
    )
    order_weight_min = min(
        float(row["weight_spearman_vs_legacy_chronological"]) for row in legacy_order
    )
    holdout_by_method: dict[str, list[float]] = {}
    for row in holdout_rows:
        holdout_by_method.setdefault(str(row["method"]), []).append(
            float(row["mean_core_fit_rmsd"])
        )
    medians = {
        method: statistics.median(values)
        for method, values in holdout_by_method.items()
    }
    independent_best = min(
        medians["scalar_independent"], medians["batched_independent"]
    )
    legacy_advantage = independent_best - medians["legacy_object"]
    scientifically_negligible = abs(legacy_advantage) <= 1e-3
    if order_core_min < 0.98 and (
        legacy_advantage <= 0 or scientifically_negligible
    ):
        outcome = "Outcome B"
    elif order_core_min >= 0.98 and legacy_advantage > 0:
        outcome = "Outcome A"
    elif abs(legacy_advantage) < 1e-3:
        outcome = "Outcome D"
    else:
        outcome = "Outcome C"
    return outcome, {
        "legacy_order_core_jaccard_min": order_core_min,
        "legacy_order_weight_spearman_min": order_weight_min,
        "legacy_holdout_core_fit_rmsd_median": medians["legacy_object"],
        "scalar_independent_holdout_core_fit_rmsd_median": medians[
            "scalar_independent"
        ],
        "batched_independent_holdout_core_fit_rmsd_median": medians[
            "batched_independent"
        ],
        "legacy_holdout_advantage_positive_is_better": legacy_advantage,
    }


def write_reports(
    outcome: str,
    summary: dict[str, float],
    focused: dict[str, dict[str, object]],
) -> None:
    report = f"""# Phase 4B warm-start semantic adjudication

## Provisional decision

**{outcome}**

The analysis changed representative processing order while retaining frame 0
as the fixed structural reference. Therefore observed differences isolate the
sequential transfer of weights/transforms and are not reference-choice effects.

## Main measurements

- Minimum legacy core Jaccard across non-chronological orders:
  {summary["legacy_order_core_jaccard_min"]:.6f}
- Minimum legacy weight Spearman across orders:
  {summary["legacy_order_weight_spearman_min"]:.6f}
- Median temporal-heldout core-fit RMSD, legacy:
  {summary["legacy_holdout_core_fit_rmsd_median"]:.6f} A
- Median temporal-heldout core-fit RMSD, scalar independent:
  {summary["scalar_independent_holdout_core_fit_rmsd_median"]:.6f} A
- Median temporal-heldout core-fit RMSD, batched independent:
  {summary["batched_independent_holdout_core_fit_rmsd_median"]:.6f} A

The implemented solver does not expose one formal global trajectory objective;
the tables therefore report the actual per-representative weighted residuals,
unweighted residuals, convergence, weight dispersion, core/contact stability
and held-out measurements rather than inventing an objective.

## Interpretation limits

Order dependence is an algorithmic property of `legacy_object`, not numerical
noise. Conversely, disagreement with legacy is not by itself evidence that the
independently initialised solution is superior. Cross-replicate evidence here
uses three cached 320 K mdCATH domains and remains a development diagnostic.
The transition-aligned 10k case is synthetic scaling evidence, not continuous
biological MD.

## Exact command

```bash
PY=/home/jjxu/miniconda3/bin/python
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. \\
"$PY" scripts/scaling_optimization_v3/run_phase4b_semantics.py
```
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "01_warmstart_semantic_adjudication.md").write_text(report)

    legacy = focused["legacy_orders"]["chronological"]
    reverse = focused["legacy_orders"]["reverse"]
    batched = focused["batched_independent"]
    case_report = f"""# Focused case: mdcath_3pio100

## Metadata

- Frames: {focused["coords"].shape[0]}
- Residues: {focused["coords"].shape[1]}
- Representatives: {len(focused["selected_indices"])}
- Source semantics: 100-frame deterministic subset of a transition-aligned
  synthetic scaling trajectory.
- Residue topology annotations: unavailable; termini are index-based and
  loops/domains/chain breaks are not guessed.

## Legacy order dependence

- Chronological vs reverse weight Spearman:
  {safe_spearman(legacy["estimate"]["mean_weights"], reverse["estimate"]["mean_weights"]):.6f}
- Chronological vs reverse core Jaccard:
  {jaccard(legacy["evaluation"]["core"], reverse["evaluation"]["core"]):.6f}
- Chronological vs reverse flexible-label Jaccard:
  {jaccard(legacy["evaluation"]["flexible"], reverse["evaluation"]["flexible"]):.6f}

## Independent comparison

- Legacy chronological held-in core-fit RMSD:
  {legacy["evaluation"]["mean_core_fit_rmsd"]:.6f} A
- Batched-independent held-in core-fit RMSD:
  {batched["evaluation"]["mean_core_fit_rmsd"]:.6f} A
- Legacy vs batched core Jaccard:
  {jaccard(legacy["evaluation"]["core"], batched["evaluation"]["core"]):.6f}
- Legacy vs batched flexible-label Jaccard:
  {jaccard(legacy["evaluation"]["flexible"], batched["evaluation"]["flexible"]):.6f}

The per-residue table and iteration traces localize the discrepancy. No claim
of a loop/domain mechanism is made because matching topology annotations are
not present in the compact input. Evidence of a local optimum or warm-start
lock-in is inferred only when order-dependent terminal states differ while
independent initialisation remains deterministic.
"""
    cases = DOC_ROOT / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    (cases / "mdcath_3pio100_case_report.md").write_text(case_report)


def main() -> int:
    if Path(sys.executable).resolve() != Path(
        "/home/jjxu/miniconda3/bin/python"
    ).resolve():
        raise RuntimeError("Use the canonical absolute Python interpreter.")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    order_rows, focused, trace_rows = order_sensitivity()
    write_tsv(RESULT_ROOT / "order_sensitivity.tsv", order_rows)
    write_3pio_outputs(focused, trace_rows)
    holdout_rows = holdout_analysis()
    write_tsv(RESULT_ROOT / "holdout_metrics.tsv", holdout_rows)
    cross_rows = cross_replicate_analysis()
    write_tsv(RESULT_ROOT / "cross_replicate_metrics.tsv", cross_rows)
    subset_rows = subset_analysis()
    write_tsv(RESULT_ROOT / "subset_sensitivity.tsv", subset_rows)
    outcome, summary = summarize_evidence(order_rows, holdout_rows)
    (RESULT_ROOT / "semantic_summary.json").write_text(
        json.dumps(
            {
                "outcome": outcome,
                "summary": summary,
                "order_rows": len(order_rows),
                "holdout_rows": len(holdout_rows),
                "cross_replicate_rows": len(cross_rows),
                "subset_rows": len(subset_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_reports(outcome, summary, focused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
