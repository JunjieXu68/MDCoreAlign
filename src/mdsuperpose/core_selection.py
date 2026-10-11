"""Hard core extraction helpers.

This module turns soft per-atom weights into deterministic hard core
selections. The current implementation keeps hard selection outside the main
optimization loop so the weighting stage and the post-processing stage remain
decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import ensure_atom_vector


@dataclass(frozen=True)
class CoreSelectionResult:
    """Discrete core-selection summary derived from soft weights."""

    core_mask: np.ndarray
    core_indices: np.ndarray
    non_core_indices: np.ndarray
    core_size: int
    core_fraction: float


@dataclass(frozen=True)
class GroupAggregationResult:
    """Grouped weight summary for future residue-level aggregation."""

    group_ids: np.ndarray
    group_weights: np.ndarray
    group_sizes: np.ndarray


def _validate_weights(weights: np.ndarray) -> np.ndarray:
    """Validate a one-dimensional soft-weight vector."""

    weights = ensure_atom_vector(weights, len(weights), name="weights")
    if weights.shape[0] == 0:
        raise ValueError("weights must not be empty")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    return weights


def _validate_threshold(threshold: float) -> float:
    """Validate a numeric threshold."""

    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return threshold


def _stable_descending_order(weights: np.ndarray) -> np.ndarray:
    """Rank atoms by descending weight with index-based tie breaking."""

    indices = np.arange(weights.shape[0])
    return np.lexsort((indices, -weights))


def _build_selection_result(core_mask: np.ndarray) -> CoreSelectionResult:
    """Build a structured selection result from a boolean mask."""

    core_mask = np.asarray(core_mask, dtype=bool)
    if core_mask.ndim != 1 or core_mask.shape[0] == 0:
        raise ValueError("core_mask must be a non-empty one-dimensional array")

    core_indices = np.flatnonzero(core_mask)
    non_core_indices = np.flatnonzero(~core_mask)
    core_size = int(core_indices.shape[0])
    core_fraction = float(core_size / core_mask.shape[0])
    return CoreSelectionResult(
        core_mask=core_mask,
        core_indices=core_indices,
        non_core_indices=non_core_indices,
        core_size=core_size,
        core_fraction=core_fraction,
    )


def select_core_by_threshold(weights: np.ndarray, threshold: float) -> CoreSelectionResult:
    """Select atoms whose weights meet or exceed a threshold."""

    weights = _validate_weights(weights)
    threshold = _validate_threshold(threshold)
    return _build_selection_result(weights >= threshold)


def select_core_by_top_k(weights: np.ndarray, k: int) -> CoreSelectionResult:
    """Select the top-k atoms using deterministic tie breaking."""

    weights = _validate_weights(weights)
    k = int(k)
    if k < 0 or k > weights.shape[0]:
        raise ValueError(f"k must be between 0 and {weights.shape[0]}")

    order = _stable_descending_order(weights)
    selected = order[:k]
    mask = np.zeros(weights.shape[0], dtype=bool)
    mask[selected] = True
    return _build_selection_result(mask)


def select_core_by_target_fraction(weights: np.ndarray, fraction: float) -> CoreSelectionResult:
    """Select a target fraction of top-weighted atoms.

    The requested fraction is converted to an integer core size using
    round-half-up and at least one selected atom for any positive fraction.
    """

    weights = _validate_weights(weights)
    fraction = float(fraction)
    if not np.isfinite(fraction):
        raise ValueError("fraction must be finite")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    k = max(1, int(np.floor(fraction * weights.shape[0] + 0.5)))
    return select_core_by_top_k(weights, k=min(k, weights.shape[0]))


def aggregate_weights_by_group(
    weights: np.ndarray,
    group_ids: np.ndarray,
    reduction: str = "mean",
) -> GroupAggregationResult:
    """Aggregate atom weights by a later residue- or group-level label."""

    weights = _validate_weights(weights)
    group_ids = np.asarray(group_ids)
    if group_ids.shape != weights.shape:
        raise ValueError(f"group_ids must have shape {weights.shape}")

    unique_group_ids, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
    aggregated = np.zeros(unique_group_ids.shape[0], dtype=float)
    np.add.at(aggregated, inverse, weights)

    if reduction == "mean":
        aggregated = aggregated / counts
    elif reduction == "sum":
        aggregated = aggregated
    elif reduction == "max":
        aggregated = np.full(unique_group_ids.shape[0], -np.inf, dtype=float)
        np.maximum.at(aggregated, inverse, weights)
    else:
        raise ValueError("reduction must be one of: 'mean', 'sum', 'max'")

    return GroupAggregationResult(
        group_ids=unique_group_ids,
        group_weights=aggregated,
        group_sizes=counts,
    )


def format_selected_atoms(
    indices: np.ndarray,
    labels: list[Any] | tuple[Any, ...] | np.ndarray | None = None,
    max_items: int | None = None,
) -> list[Any]:
    """Format selected atom indices as indices or optional labels."""

    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be non-negative when provided")

    if labels is None:
        items = indices.tolist()
    else:
        labels = list(labels)
        if len(labels) == 0 and indices.shape[0] > 0:
            raise ValueError("labels must not be empty when indices are present")
        items = [labels[index] for index in indices]

    if max_items is None:
        return items
    return items[:max_items]


def summarize_core_selection(
    result: CoreSelectionResult,
    labels: list[Any] | tuple[Any, ...] | np.ndarray | None = None,
    max_items: int = 10,
) -> dict[str, Any]:
    """Return a compact summary of a hard core selection."""

    n_atoms = result.core_mask.shape[0]
    if labels is not None and len(labels) != n_atoms:
        raise ValueError(f"labels must have length {n_atoms}")

    return {
        "core_size": result.core_size,
        "core_fraction": result.core_fraction,
        "core_indices": result.core_indices.tolist(),
        "non_core_indices": result.non_core_indices.tolist(),
        "core_preview": format_selected_atoms(result.core_indices, labels=labels, max_items=max_items),
        "non_core_preview": format_selected_atoms(
            result.non_core_indices,
            labels=labels,
            max_items=max_items,
        ),
    }


def threshold_core_mask(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Backward-compatible threshold selection returning only the mask."""

    return select_core_by_threshold(weights, threshold).core_mask


def top_fraction_core_mask(weights: np.ndarray, fraction: float) -> np.ndarray:
    """Backward-compatible top-fraction selection returning only the mask."""

    return select_core_by_target_fraction(weights, fraction).core_mask
