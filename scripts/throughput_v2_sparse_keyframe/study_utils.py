"""Shared utilities for the isolated Throughput-v2 study scripts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdcorealign.experimental_throughput_v2 import (  # noqa: E402
    V2AlignmentResult,
    deterministic_core_mask,
)
from mdsuperpose.alignment import apply_transform, solve_weighted_rigid_alignment  # noqa: E402
from mdsuperpose.all_frame_engine import AllFrameFitResult, fit_all_frames_fixed_weights  # noqa: E402
from mdsuperpose.chronology_study import representative_indices  # noqa: E402
from mdsuperpose.fast_kabsch import apply_transforms_batch  # noqa: E402
from mdsuperpose.parallel_weights import estimate_shared_weight  # noqa: E402
from scripts.cpu_chronology_v3.run_self_variability import load_case  # noqa: E402


OUT = ROOT / "results/throughput_v2_sparse_keyframe"
CONTROL_ROOT = (
    ROOT
    / "results/mdcath_benchmark/transition_aligned_long_mode_benchmark/"
    "domains/4oelA02/raw_replicates"
)


@dataclass(frozen=True, slots=True)
class FrozenCurrentResult:
    weights: np.ndarray
    fit: AllFrameFitResult
    weight_seconds: float
    fit_seconds: float
    total_seconds: float


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_transition_control(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the registered 4oelA02 control from each raw replicate copy."""

    raw: list[np.ndarray] = []
    for label in "ABCDE":
        with np.load(CONTROL_ROOT / f"{label}_ca_coords.npz", allow_pickle=True) as archive:
            raw.append(np.asarray(archive["coords"], dtype=np.float64))
    n_residues = raw[0].shape[1]
    output = np.empty((int(n_frames), n_residues, 3), dtype=np.float64)
    first = raw[0]
    copied = min(int(n_frames), first.shape[0])
    output[:copied] = first[:copied] - np.mean(first[-1], axis=0)[None, None, :]
    current_last = output[copied - 1]
    position = copied
    segment = 1
    unit_weights = np.ones(n_residues, dtype=np.float64)
    while position < n_frames:
        source = raw[segment % len(raw)]
        fit = solve_weighted_rigid_alignment(source[0], current_last, unit_weights)
        transformed = apply_transform(source, fit)
        stop = min(n_frames, position + transformed.shape[0])
        output[position:stop] = transformed[: stop - position]
        current_last = output[stop - 1]
        position = stop
        segment += 1
    return output, np.full(n_residues, "A", dtype="U1")


def load_study_case(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    case_id = row["case_id"]
    if case_id.startswith("transition_control_4oelA02_"):
        return build_transition_control(int(row["n_frames"]))
    return load_case(
        {
            "case_id": case_id,
            "input_path": row["input_path"],
            "input_key": row.get("input_key", ""),
            "n_frames": row["n_frames"],
            "n_residues": row["n_residues"],
        }
    )


def frozen_current_throughput(
    coords: np.ndarray,
    chains: np.ndarray,
    *,
    chunk_size: int = 2000,
) -> FrozenCurrentResult:
    """Run the frozen compact current-throughput contract on development data."""

    selected = representative_indices(coords.shape[0])
    reference = np.asarray(coords[0], dtype=np.float64)
    weight_start = time.perf_counter()
    estimate = estimate_shared_weight(
        np.asarray(coords[selected], dtype=np.float64),
        reference,
        chain_ids=chains,
    )
    weight_seconds = time.perf_counter() - weight_start
    fit_start = time.perf_counter()
    fit = fit_all_frames_fixed_weights(
        coords,
        reference,
        estimate.mean_weights,
        chunk_size=chunk_size,
        return_aligned=False,
        compute_ordinary_rmsf=False,
    )
    fit_seconds = time.perf_counter() - fit_start
    return FrozenCurrentResult(
        weights=np.asarray(estimate.mean_weights, dtype=np.float64),
        fit=fit,
        weight_seconds=weight_seconds,
        fit_seconds=fit_seconds,
        total_seconds=weight_seconds + fit_seconds,
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="stable")
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
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) <= 1e-15 or np.std(right_rank) <= 1e-15:
        return float(np.array_equal(left_rank, right_rank))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def compare_alignment(
    candidate: V2AlignmentResult,
    baseline: FrozenCurrentResult,
    coords: np.ndarray,
) -> dict[str, object]:
    """Compare a candidate with current throughput under the locked metrics."""

    if any(
        value is None
        for value in (
            baseline.fit.rmsd,
            baseline.fit.weighted_rmsd,
            baseline.fit.robust_rmsf,
        )
    ):
        raise RuntimeError("baseline compact outputs are incomplete")
    rmsd_difference = np.abs(candidate.rmsd - baseline.fit.rmsd)
    relative = candidate.rotations @ np.swapaxes(baseline.fit.rotations, 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation = np.degrees(np.arccos(cosine))
    audit_count = min(coords.shape[0], max(64, int(math.ceil(0.01 * coords.shape[0]))))
    audit = np.unique(
        np.rint(np.linspace(0, coords.shape[0] - 1, audit_count)).astype(np.int64)
    )
    candidate_aligned = apply_transforms_batch(
        np.asarray(coords[audit], dtype=np.float64),
        candidate.rotations[audit],
        candidate.translations[audit],
    )
    baseline_aligned = apply_transforms_batch(
        np.asarray(coords[audit], dtype=np.float64),
        baseline.fit.rotations[audit],
        baseline.fit.translations[audit],
    )
    coordinate = np.sqrt(
        np.mean(np.sum((candidate_aligned - baseline_aligned) ** 2, axis=2), axis=1)
    )
    rmsf_difference = np.abs(candidate.robust_rmsf - baseline.fit.robust_rmsf)
    candidate_core = deterministic_core_mask(candidate.weights)
    baseline_core = deterministic_core_mask(baseline.weights)
    determinant = np.linalg.det(candidate.rotations)
    nonfinite = not all(
        np.isfinite(value).all()
        for value in (
            candidate.rotations,
            candidate.translations,
            candidate.rmsd,
            candidate.robust_rmsf,
        )
    )
    reflected = bool(np.any(determinant <= 0.0))
    core_jaccard = jaccard(candidate_core, baseline_core)
    rmsf_spearman = spearman(candidate.robust_rmsf, baseline.fit.robust_rmsf)
    catastrophic = bool(
        nonfinite
        or reflected
        or core_jaccard < 0.50
        or float(np.median(coordinate)) > 2.0
        or float(np.quantile(rmsd_difference, 0.95)) > 2.0
        or rmsf_spearman < 0.90
    )
    return {
        "core_jaccard": core_jaccard,
        "stable_weight_spearman": spearman(candidate.weights, baseline.weights),
        "rotation_geodesic_median_degrees": float(np.median(rotation)),
        "rotation_geodesic_p95_degrees": float(np.quantile(rotation, 0.95)),
        "aligned_coordinate_audit_median_A": float(np.median(coordinate)),
        "aligned_coordinate_audit_p95_A": float(np.quantile(coordinate, 0.95)),
        "rmsd_abs_diff_median_A": float(np.median(rmsd_difference)),
        "rmsd_abs_diff_p95_A": float(np.quantile(rmsd_difference, 0.95)),
        "rmsd_abs_diff_max_A": float(np.max(rmsd_difference)),
        "stable_rmsf_spearman": rmsf_spearman,
        "stable_rmsf_mae_A": float(np.mean(rmsf_difference)),
        "non_finite": nonfinite,
        "reflected_rotation": reflected,
        "catastrophic_mismatch": catastrophic,
    }


def profile_quality(
    profile_rmsf: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    baseline_rmsf: np.ndarray,
) -> dict[str, object]:
    difference = np.abs(profile_rmsf - baseline_rmsf)
    return {
        "stable_rmsf_spearman": spearman(profile_rmsf, baseline_rmsf),
        "stable_rmsf_mae_A": float(np.mean(difference)),
        "interval_coverage": float(np.mean((baseline_rmsf >= lower) & (baseline_rmsf <= upper))),
        "calibration_slope": float(
            np.polyfit(profile_rmsf, baseline_rmsf, 1)[0]
            if np.std(profile_rmsf) > 1e-15
            else np.nan
        ),
        "catastrophic_mismatch": bool(
            not np.isfinite(profile_rmsf).all()
            or spearman(profile_rmsf, baseline_rmsf) < 0.90
        ),
    }
