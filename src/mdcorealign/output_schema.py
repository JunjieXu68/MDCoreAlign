"""Versioned output contracts for the stable MDCoreAlign product interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "1.1.0"
"""Schema version for canonical JSON and tabular outputs."""

THROUGHPUT_METHOD = "fixed-budget continuous-weight alignment"
THROUGHPUT_MODE = "throughput"
THROUGHPUT_EXECUTION_CONTRACT = "throughput"

FRAME_METRIC_COLUMNS = (
    "frame_index",
    "full_set_RMSD",
    "weighted_RMSD",
)
"""Frozen v1 frame-metric columns; RMSD values are in Angstrom."""

RESIDUE_PROFILE_COLUMNS = (
    "residue_index",
    "chain",
    "residue_id",
    "resname",
    "stable_weight",
    "mobility_score",
    "core_label",
    "robust_RMSF",
    "mean_residual",
    "flexible_segment_id",
    "mobility_rank",
    "mobility_percentile",
)
"""Frozen v1 residue-profile columns before optional ordinary RMSF."""

FIELD_UNITS: dict[str, str] = {
    "full_set_RMSD": "angstrom",
    "weighted_RMSD": "angstrom",
    "unified_posthoc_RMSD": "angstrom",
    "robust_RMSF": "angstrom",
    "ordinary_RMSF": "angstrom",
    "mean_residual": "angstrom",
    "stable_weight": "dimensionless_0_to_1",
    "mobility_score": "dimensionless_0_to_1",
    "core_fraction": "fraction_0_to_1",
    "effective_core_fraction": "continuous_weight_fraction_0_to_1",
    "effective_core_size": "effective_residue_count",
}


def canonical_contract_metadata(
    execution_contract: str | None,
    *,
    result_mode: str,
) -> dict[str, object]:
    """Return canonical product metadata without leaking internal mode names."""

    if execution_contract == THROUGHPUT_EXECUTION_CONTRACT:
        return {
            "method": THROUGHPUT_METHOD,
            "mode": THROUGHPUT_MODE,
            "execution_contract": THROUGHPUT_EXECUTION_CONTRACT,
            "experimental_mode": False,
        }
    if execution_contract == "exact":
        return {
            "method": "sequential reference continuous-weight alignment",
            "mode": "exact",
            "execution_contract": "exact",
            "experimental_mode": False,
        }
    return {
        "method": "legacy compatibility alignment",
        "mode": result_mode,
        "execution_contract": execution_contract,
        "experimental_mode": True,
    }


def _sum_present(values: Mapping[str, Any], names: tuple[str, ...]) -> float:
    return float(sum(float(values[name]) for name in names if name in values))


def canonical_timing(raw_timing: Mapping[str, Any]) -> dict[str, float]:
    """Translate internal timing labels into the stable v1 timing schema."""

    raw = dict(raw_timing)
    weight_estimation = float(
        raw.get(
            "weight_estimation",
            raw.get(
                "limited_global_refinement",
                _sum_present(
                    raw,
                    (
                        "representative_selection",
                        "representative_weighted_fit",
                        "representative_residual",
                        "representative_weight_update",
                        "representative_sequence_regularization",
                    ),
                ),
            ),
        )
    )
    all_frame_fit = float(
        raw.get(
            "all_frame_fit_metrics_RMSF",
            raw.get(
                "exact_all_frame_total",
                _sum_present(
                    raw,
                    (
                        "all_frame_weighted_fit",
                        "all_frame_ordinary_fit",
                        "transform_application",
                    ),
                ),
            ),
        )
    )
    return {
        "weight_estimation_seconds": weight_estimation,
        "all_frame_fit_seconds": all_frame_fit,
        "rmsd_seconds": float(raw.get("RMSD", raw.get("rmsd", 0.0))),
        "rmsf_seconds": float(raw.get("RMSF", raw.get("rmsf", 0.0))),
        "diagnostics_seconds": float(
            raw.get("confidence_diagnostics", raw.get("diagnostics", 0.0))
        ),
        "coordinate_materialization_seconds": float(
            raw.get(
                "aligned_coordinate_materialization",
                raw.get("aligned_output_materialization", 0.0),
            )
        ),
        "end_to_end_seconds": float(
            raw.get(
                "total_runtime_seconds",
                raw.get("algorithm_including_scan", raw.get("total", 0.0)),
            )
        ),
    }


def schema_metadata(software_version: str) -> dict[str, object]:
    """Return common version and unit metadata for public JSON outputs."""

    return {
        "schema_version": SCHEMA_VERSION,
        "software_version": software_version,
        "field_units": dict(FIELD_UNITS),
    }
