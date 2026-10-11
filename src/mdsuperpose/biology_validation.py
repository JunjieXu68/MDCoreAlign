"""Lightweight helpers for external biological validation summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DomainRange:
    """Inclusive residue range assigned to one biological domain."""

    name: str
    start: int
    end: int


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign average ranks with stable tie handling."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=float)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation with simple input validation."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of equal shape")
    if x.shape[0] < 2:
        raise ValueError("at least two paired values are required")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    x_norm = float(np.linalg.norm(x_centered))
    y_norm = float(np.linalg.norm(y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / (x_norm * y_norm))


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without external dependencies."""

    return pearson_correlation(average_ranks(x), average_ranks(y))


def top_fraction_mask(values: np.ndarray, fraction: float, *, highest: bool = True) -> np.ndarray:
    """Select the top fraction of one score vector with stable ordering."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")

    count = max(1, int(round(values.shape[0] * fraction)))
    if highest:
        order = np.argsort(-values, kind="mergesort")
    else:
        order = np.argsort(values, kind="mergesort")
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[order[:count]] = True
    return mask


def overlap_fraction(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    """Compute overlap divided by the smaller selected set size."""

    first_mask = np.asarray(first_mask, dtype=bool)
    second_mask = np.asarray(second_mask, dtype=bool)
    if first_mask.shape != second_mask.shape or first_mask.ndim != 1:
        raise ValueError("masks must be one-dimensional arrays of equal shape")
    denominator = min(int(np.sum(first_mask)), int(np.sum(second_mask)))
    if denominator == 0:
        return 0.0
    return float(np.sum(first_mask & second_mask) / denominator)


def enrichment_ratio(selection_mask: np.ndarray, group_mask: np.ndarray) -> float:
    """Compute selected-group enrichment relative to the background fraction."""

    selection_mask = np.asarray(selection_mask, dtype=bool)
    group_mask = np.asarray(group_mask, dtype=bool)
    if selection_mask.shape != group_mask.shape or selection_mask.ndim != 1:
        raise ValueError("selection_mask and group_mask must have equal one-dimensional shape")
    selected_count = int(np.sum(selection_mask))
    background_count = int(np.sum(group_mask))
    if selected_count == 0 or background_count == 0:
        return 0.0
    selected_fraction = float(np.sum(selection_mask & group_mask) / selected_count)
    background_fraction = float(background_count / selection_mask.shape[0])
    if background_fraction == 0.0:
        return 0.0
    return float(selected_fraction / background_fraction)


def assign_domains_from_ranges(
    residue_numbers: np.ndarray,
    ranges: list[DomainRange],
    *,
    default_label: str = "unassigned",
) -> np.ndarray:
    """Assign one categorical domain label to each residue number."""

    residue_numbers = np.asarray(residue_numbers, dtype=int)
    if residue_numbers.ndim != 1:
        raise ValueError("residue_numbers must be one-dimensional")
    labels = np.full(residue_numbers.shape[0], default_label, dtype=object)
    for domain in ranges:
        mask = (residue_numbers >= domain.start) & (residue_numbers <= domain.end)
        labels[mask] = domain.name
    return labels


def infer_sequence_blocks(residue_numbers: np.ndarray) -> np.ndarray:
    """Infer contiguous sequence blocks from monotonic residue-number runs."""

    residue_numbers = np.asarray(residue_numbers, dtype=int)
    if residue_numbers.ndim != 1:
        raise ValueError("residue_numbers must be one-dimensional")
    if residue_numbers.shape[0] == 0:
        return np.zeros(0, dtype=int)

    block_ids = np.zeros(residue_numbers.shape[0], dtype=int)
    current_block = 0
    for index in range(1, residue_numbers.shape[0]):
        if residue_numbers[index] < residue_numbers[index - 1]:
            current_block += 1
        block_ids[index] = current_block
    return block_ids


def build_residue_labels(
    residue_names: np.ndarray,
    residue_numbers: np.ndarray,
    *,
    block_ids: np.ndarray | None = None,
) -> list[str]:
    """Build residue labels and disambiguate repeated numbering blocks when needed."""

    residue_names = np.asarray(residue_names, dtype=object)
    residue_numbers = np.asarray(residue_numbers, dtype=int)
    if residue_names.shape != residue_numbers.shape or residue_names.ndim != 1:
        raise ValueError("residue_names and residue_numbers must be one-dimensional with equal shape")

    if block_ids is None:
        block_ids = infer_sequence_blocks(residue_numbers)
    else:
        block_ids = np.asarray(block_ids, dtype=int)
        if block_ids.shape != residue_numbers.shape:
            raise ValueError("block_ids must match residue_numbers shape")

    labels = [f"{resname}{resid}" for resname, resid in zip(residue_names, residue_numbers, strict=True)]
    if len(set(labels)) == len(labels):
        return labels
    return [
        f"{resname}{resid}_block{block_id + 1}"
        for resname, resid, block_id in zip(residue_names, residue_numbers, block_ids, strict=True)
    ]


def residue_contact_density(
    coords: np.ndarray,
    *,
    residue_numbers: np.ndarray | None = None,
    block_ids: np.ndarray | None = None,
    cutoff: float = 10.0,
    min_sequence_separation: int = 2,
) -> np.ndarray:
    """Compute a simple per-residue CA contact-density proxy from one structure."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if coords.shape[0] == 0:
        raise ValueError("coords must contain at least one residue")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if min_sequence_separation < 0:
        raise ValueError("min_sequence_separation must be non-negative")

    n_residues = coords.shape[0]
    if residue_numbers is None:
        residue_numbers = np.arange(n_residues, dtype=int)
    else:
        residue_numbers = np.asarray(residue_numbers, dtype=int)
        if residue_numbers.shape != (n_residues,):
            raise ValueError("residue_numbers must have shape (N,)")

    if block_ids is None:
        block_ids = infer_sequence_blocks(residue_numbers)
    else:
        block_ids = np.asarray(block_ids, dtype=int)
        if block_ids.shape != (n_residues,):
            raise ValueError("block_ids must have shape (N,)")

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    same_block = block_ids[:, None] == block_ids[None, :]
    sequence_separation = np.abs(residue_numbers[:, None] - residue_numbers[None, :])
    nonlocal_mask = ~(same_block & (sequence_separation <= min_sequence_separation))
    contact_mask = (distances <= cutoff) & (distances > 0.0) & nonlocal_mask
    return np.sum(contact_mask, axis=1).astype(float)


def residue_interblock_contact_density(
    coords: np.ndarray,
    *,
    block_ids: np.ndarray,
    cutoff: float = 10.0,
) -> np.ndarray:
    """Compute per-residue contact density restricted to other inferred blocks."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if coords.shape[0] == 0:
        raise ValueError("coords must contain at least one residue")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")

    block_ids = np.asarray(block_ids, dtype=int)
    if block_ids.shape != (coords.shape[0],):
        raise ValueError("block_ids must have shape (N,)")

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    interblock_mask = block_ids[:, None] != block_ids[None, :]
    contact_mask = (distances <= cutoff) & (distances > 0.0) & interblock_mask
    return np.sum(contact_mask, axis=1).astype(float)
