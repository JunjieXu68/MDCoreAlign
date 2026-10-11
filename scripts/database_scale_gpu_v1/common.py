"""Shared utilities for the D0-D1 database-scale GPU benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from mdsuperpose.flexibility import detect_flexible_segments


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/database_scale_gpu_v1"
D0_ROOT = RESULT_ROOT / "d0_freeze"
D1_ROOT = RESULT_ROOT / "d1_all_domain"
REGISTRY_PATH = D0_ROOT / "benchmark_registry.tsv"
QUALITY_GATES_PATH = D0_ROOT / "locked_d1_quality_gates.json"
CORRECTED_ROOT = ROOT / "results/project_completion_v2_unit_corrected"
CORRECTED_TABLE = (
    CORRECTED_ROOT / "aggregate/all_domain_unit_corrected_per_domain.tsv"
)
HISTORICAL_RUNS = (
    ROOT
    / "results/mdcath_benchmark/all_domain_canonical/run_fullframes"
    / "per_trajectory"
)
MANIFEST_PATH = (
    ROOT
    / "results/mdcath_benchmark/all_domain_canonical"
    / "mdcath_all_domain_canonical_manifest.tsv"
)
SNAPSHOT_REVISION = "5e3ed8aec62b689e01751db16275fdcdbc39e47f"
SNAPSHOT_DATA = (
    ROOT
    / "results/mdcath_benchmark/cache"
    / "datasets--compsciencelab--mdCATH/snapshots"
    / SNAPSHOT_REVISION
    / "data"
)
MISSING_DOMAIN_ID = "2mh3A00"
EXPECTED_REGISTERED = 5_398
EXPECTED_ELIGIBLE = 5_397


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Atomically write a union-schema TSV."""

    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    """Atomically write formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    """Return a deterministic text hash."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def repository_path(path: Path) -> str:
    """Return a repository-relative path when possible."""

    try:
        return path.resolve(strict=False).relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def source_h5(domain_id: str) -> Path:
    """Return the canonical local mdCATH HDF5 path."""

    return SNAPSHOT_DATA / f"mdcath_dataset_{domain_id}.h5"


def cache_blob_sha256(path: Path) -> str:
    """Return the SHA-256 identifier used by the content-addressed HF cache."""

    resolved_name = path.resolve().name
    if not re.fullmatch(r"[0-9a-f]{64}", resolved_name):
        raise ValueError(f"cache target is not SHA-256 addressed: {path}")
    return resolved_name


def corrected_profile_path(trajectory_id: str) -> Path:
    """Resolve the unit-corrected exact profile without changing old outputs."""

    corrected = (
        CORRECTED_ROOT
        / "affected_reruns/per_trajectory"
        / trajectory_id
        / "residue_profile.tsv"
    )
    historical = HISTORICAL_RUNS / trajectory_id / "residue_profile.tsv"
    path = corrected if corrected.is_file() else historical
    if not path.is_file():
        raise FileNotFoundError(f"missing exact profile: {trajectory_id}")
    return path


def load_exact_profile(path: Path) -> dict[str, object]:
    """Load exact weights, labels, RMSF and residue mapping."""

    rows = read_tsv(path)
    if not rows:
        raise ValueError(f"empty exact profile: {path}")
    weights = np.asarray(
        [float(row["soft_stability_weight"]) for row in rows],
        dtype=np.float64,
    )
    rmsf = np.asarray(
        [float(row["robust_RMSF"]) for row in rows],
        dtype=np.float64,
    )
    core = np.asarray(
        [int(row["core_label"]) > 0 for row in rows],
        dtype=bool,
    )
    flexible = np.asarray(
        [int(row["flexible_segment_id"]) > 0 for row in rows],
        dtype=bool,
    )
    mapping = [
        (
            str(row.get("chain", "_")),
            str(row.get("residue_id", "")),
            str(row.get("resname", "UNK")),
        )
        for row in rows
    ]
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(rmsf).all()
        or np.any(weights < 0.0)
    ):
        raise ValueError(f"invalid exact profile values: {path}")
    return {
        "rows": rows,
        "weights": weights,
        "rmsf": rmsf,
        "core": core,
        "flexible": flexible,
        "mapping": mapping,
    }


def mapping_digest(mapping: Iterable[tuple[str, str, str]]) -> str:
    """Hash an explicit ordered chain/residue/name mapping."""

    text = "\n".join("\t".join(values) for values in mapping)
    return stable_hash(text)


def normalized_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative weights to a maximum of one."""

    values = np.asarray(weights, dtype=np.float64)
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return np.zeros_like(values)
    return values / maximum


def core_mask(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    """Select a deterministic fixed-fraction core."""

    values = np.asarray(weights, dtype=np.float64)
    count = max(1, int(np.floor(values.size * fraction + 0.5)))
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def flexible_mask(weights: np.ndarray, chains: np.ndarray) -> np.ndarray:
    """Apply the frozen profile-to-flexible-segment rule."""

    labels = detect_flexible_segments(
        np.clip(1.0 - normalized_weights(weights), 0.0, 1.0),
        percentile_cutoff=0.8,
        min_segment_length=3,
        merge_gap=1,
        chain=chains,
    )
    return labels > 0


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Return Boolean Jaccard overlap."""

    left_values = np.asarray(left, dtype=bool)
    right_values = np.asarray(right, dtype=bool)
    union = int(np.count_nonzero(left_values | right_values))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left_values & right_values) / union)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman correlation with a deterministic constant fallback."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size != right_values.size or left_values.size == 0:
        return float("nan")
    if np.all(left_values == left_values[0]) and np.all(
        right_values == right_values[0]
    ):
        return 1.0 if np.allclose(left_values, right_values) else 0.0
    value = spearmanr(left_values, right_values).statistic
    return float(value) if np.isfinite(value) else 0.0


def cutoff_margin(weights: np.ndarray, fraction: float = 0.6) -> float:
    """Return the weight gap at the hard-core boundary."""

    values = np.sort(np.asarray(weights, dtype=np.float64))[::-1]
    count = max(1, int(np.floor(values.size * fraction + 0.5)))
    if count >= values.size:
        return math.inf
    return float(values[count - 1] - values[count])


def result_finite(result: object) -> bool:
    """Return whether all scientific arrays on an alignment result are finite."""

    names = (
        "frozen_weights",
        "rotations",
        "translations",
        "robust_rmsf",
        "rmsd",
        "weighted_rmsd",
    )
    return all(
        np.isfinite(np.asarray(getattr(result, name))).all() for name in names
    )


def quality_against_exact(
    candidate: object,
    *,
    exact_weights: np.ndarray,
    exact_rmsd: np.ndarray,
    exact_rmsf: np.ndarray,
    exact_core: np.ndarray,
    exact_flexible: np.ndarray,
    chains: np.ndarray,
) -> dict[str, object]:
    """Evaluate one matched alignment result against the reused exact core."""

    candidate_weights = np.asarray(candidate.frozen_weights, dtype=np.float64)
    candidate_core = core_mask(candidate_weights)
    candidate_flexible = flexible_mask(candidate_weights, chains)
    rmsd_delta = np.abs(np.asarray(candidate.rmsd) - exact_rmsd)
    core_overlap = jaccard(candidate_core, exact_core)
    median_delta = float(np.median(rmsd_delta))
    return {
        "finite_outputs": result_finite(candidate),
        "core_jaccard_vs_exact": core_overlap,
        "weight_spearman_vs_exact": safe_spearman(
            normalized_weights(candidate_weights),
            normalized_weights(exact_weights),
        ),
        "rmsf_spearman_vs_exact": safe_spearman(
            np.asarray(candidate.robust_rmsf),
            exact_rmsf,
        ),
        "rmsf_mae_vs_exact_A": float(
            np.mean(np.abs(np.asarray(candidate.robust_rmsf) - exact_rmsf))
        ),
        "flexible_label_agreement_vs_exact": float(
            np.mean(candidate_flexible == exact_flexible)
        ),
        "flexible_label_jaccard_vs_exact": jaccard(
            candidate_flexible,
            exact_flexible,
        ),
        "median_absolute_rmsd_difference_vs_exact_A": median_delta,
        "p95_absolute_rmsd_difference_vs_exact_A": float(
            np.quantile(rmsd_delta, 0.95)
        ),
        "mean_unified_posthoc_rmsd_A": float(np.mean(candidate.rmsd)),
        "effective_core_fraction": float(
            np.sum(normalized_weights(candidate_weights))
            / candidate_weights.size
        ),
        "catastrophic_quality_failure": bool(
            not result_finite(candidate)
            or core_overlap < 0.5
            or median_delta > 1.0
        ),
    }


def cpu_gpu_agreement(
    cpu: object,
    gpu: object,
    *,
    chains: np.ndarray,
) -> dict[str, object]:
    """Compare CPU and GPU implementations under one output contract."""

    cpu_weights = np.asarray(cpu.frozen_weights, dtype=np.float64)
    gpu_weights = np.asarray(gpu.frozen_weights, dtype=np.float64)
    cpu_core = core_mask(cpu_weights)
    gpu_core = core_mask(gpu_weights)
    core_overlap = jaccard(cpu_core, gpu_core)
    maximum_weight_delta = float(np.max(np.abs(cpu_weights - gpu_weights)))
    minimum_margin = min(cutoff_margin(cpu_weights), cutoff_margin(gpu_weights))
    tie_explained = bool(
        core_overlap < 1.0
        and core_overlap >= 0.95
        and maximum_weight_delta <= 1e-4
        and minimum_margin <= 1e-4
    )
    return {
        "cpu_gpu_core_jaccard": core_overlap,
        "cpu_gpu_weight_spearman": safe_spearman(cpu_weights, gpu_weights),
        "cpu_gpu_rmsf_spearman": safe_spearman(
            np.asarray(cpu.robust_rmsf),
            np.asarray(gpu.robust_rmsf),
        ),
        "cpu_gpu_rmsf_mae_A": float(
            np.mean(
                np.abs(
                    np.asarray(cpu.robust_rmsf)
                    - np.asarray(gpu.robust_rmsf)
                )
            )
        ),
        "cpu_gpu_median_absolute_rmsd_difference_A": float(
            np.median(np.abs(np.asarray(cpu.rmsd) - np.asarray(gpu.rmsd)))
        ),
        "cpu_gpu_maximum_weight_difference": maximum_weight_delta,
        "cpu_gpu_core_cutoff_minimum_margin": minimum_margin,
        "cpu_gpu_core_difference_explained_by_numerical_tie": tie_explained,
        "cpu_gpu_flexible_label_agreement": float(
            np.mean(
                flexible_mask(cpu_weights, chains)
                == flexible_mask(gpu_weights, chains)
            )
        ),
    }


def percentile(values: list[float], quantile: float) -> float:
    """Return one finite quantile or NaN for an empty collection."""

    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.quantile(finite, quantile))


def finite_float(value: object) -> float | None:
    """Parse one finite number."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
