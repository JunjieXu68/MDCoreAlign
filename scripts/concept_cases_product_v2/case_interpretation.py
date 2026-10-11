"""Shared, locked interpretation rules for the CP v2 case-study assets.

This module operates only on frozen per-residue decomposition tables. It does
not estimate weights, align coordinates, or change any scientific algorithm.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


HIGH_SCORE = 0.70
LOW_SCORE = 0.30
COHERENT_MIN_SELECTED = 8
COHERENT_MERGE_GAP = 2
LOCAL_MIN_SELECTED = 3
LOCAL_MAX_SELECTED = 20
LOCAL_MERGE_GAP = 1
BOUNDARY_DISTANCE = 5


@dataclass(frozen=True)
class Region:
    """A chain-aware geometric region derived from registered thresholds."""

    region_type: str
    chain: str
    indices: tuple[int, ...]
    start_residue_id: str
    end_residue_id: str
    start_residue_index: int
    end_residue_index: int
    span_length: int
    selected_residue_count: int
    mean_global_displacement_score: float
    mean_local_deformation_score: float
    mean_stable_weight: float
    qualifying_coherent_region: str = ""

    def as_dict(self, region_id: str) -> dict[str, Any]:
        """Return a table-ready representation."""

        return {
            "region_id": region_id,
            "region_type": self.region_type,
            "chain": self.chain,
            "start_residue_id": self.start_residue_id,
            "end_residue_id": self.end_residue_id,
            "start_residue_index": self.start_residue_index,
            "end_residue_index": self.end_residue_index,
            "span_length": self.span_length,
            "selected_residue_count": self.selected_residue_count,
            "mean_global_displacement_score": self.mean_global_displacement_score,
            "mean_local_deformation_score": self.mean_local_deformation_score,
            "mean_stable_weight": self.mean_stable_weight,
            "qualifying_coherent_region": self.qualifying_coherent_region,
            "interpretation_scope": "geometric_only_no_biological_domain_annotation",
        }


def _arrays(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extract explicit metadata and registered scores from frozen rows."""

    return {
        "chain": np.asarray([str(row["chain"]) for row in rows], dtype=object),
        "residue_id": np.asarray([str(row["residue_id"]) for row in rows], dtype=object),
        "residue_index": np.asarray([int(row["residue_index"]) for row in rows]),
        "global": np.asarray([float(row["global_displacement_score"]) for row in rows]),
        "local": np.asarray([float(row["local_deformation_score"]) for row in rows]),
        "weight": np.asarray([float(row["exact_stable_weight"]) for row in rows]),
    }


def classify_scores(global_score: float, local_score: float, stable_weight: float) -> str:
    """Assign one exclusive display class using the locked interpretation.

    High/high and low-global/high-local states take precedence over the stable
    anchor mask because the latter deliberately does not constrain W5. The
    overlap remains available through the separate boolean columns generated
    by :func:`classify_rows`.
    """

    if global_score >= HIGH_SCORE and local_score >= HIGH_SCORE:
        return "high_global_high_local"
    if global_score >= HIGH_SCORE and local_score <= LOW_SCORE:
        return "coherent_displacement_candidate"
    if local_score >= HIGH_SCORE and global_score < HIGH_SCORE:
        return "local_flexibility_candidate"
    if stable_weight >= HIGH_SCORE and global_score <= LOW_SCORE:
        return "stable_anchor_candidate"
    if global_score <= LOW_SCORE and local_score <= LOW_SCORE:
        return "low_global_low_local"
    return "intermediate"


def classify_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return frozen rows augmented with one shared interpretation contract."""

    classified: list[dict[str, Any]] = []
    for row in rows:
        global_score = float(row["global_displacement_score"])
        local_score = float(row["local_deformation_score"])
        stable_weight = float(row["exact_stable_weight"])
        updated = dict(row)
        updated.update(
            {
                "v2_primary_class": classify_scores(global_score, local_score, stable_weight),
                "is_stable_anchor_candidate": int(
                    stable_weight >= HIGH_SCORE and global_score <= LOW_SCORE
                ),
                "is_coherent_displacement_candidate": int(
                    global_score >= HIGH_SCORE and local_score <= LOW_SCORE
                ),
                "is_local_flexibility_candidate": int(
                    local_score >= HIGH_SCORE and global_score < HIGH_SCORE
                ),
                "is_high_global_high_local": int(
                    global_score >= HIGH_SCORE and local_score >= HIGH_SCORE
                ),
                "high_threshold": HIGH_SCORE,
                "low_threshold": LOW_SCORE,
            }
        )
        classified.append(updated)
    return classified


def _groups(
    mask: np.ndarray,
    chains: np.ndarray,
    residue_indices: np.ndarray,
    *,
    merge_gap: int,
) -> list[np.ndarray]:
    """Group selected residues without crossing chains or registered gaps."""

    groups: list[np.ndarray] = []
    for chain in dict.fromkeys(chains.tolist()):
        selected = np.flatnonzero((chains == chain) & mask)
        if selected.size == 0:
            continue
        current = [int(selected[0])]
        for index in selected[1:]:
            index = int(index)
            position_gap = int(residue_indices[index] - residue_indices[current[-1]])
            if position_gap <= merge_gap + 1:
                current.append(index)
            else:
                groups.append(np.asarray(current, dtype=np.int64))
                current = [index]
        groups.append(np.asarray(current, dtype=np.int64))
    return groups


def _region(
    kind: str,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    qualifying_coherent_region: str = "",
) -> Region:
    first = int(indices[0])
    last = int(indices[-1])
    start = int(arrays["residue_index"][first])
    end = int(arrays["residue_index"][last])
    return Region(
        region_type=kind,
        chain=str(arrays["chain"][first]),
        indices=tuple(int(value) for value in indices),
        start_residue_id=str(arrays["residue_id"][first]),
        end_residue_id=str(arrays["residue_id"][last]),
        start_residue_index=start,
        end_residue_index=end,
        span_length=end - start + 1,
        selected_residue_count=int(indices.size),
        mean_global_displacement_score=float(np.mean(arrays["global"][indices])),
        mean_local_deformation_score=float(np.mean(arrays["local"][indices])),
        mean_stable_weight=float(np.mean(arrays["weight"][indices])),
        qualifying_coherent_region=qualifying_coherent_region,
    )


def detect_regions(rows: Sequence[dict[str, Any]]) -> list[Region]:
    """Detect all registered v2 regions and chain-aware putative boundaries."""

    arrays = _arrays(rows)
    global_score = arrays["global"]
    local_score = arrays["local"]
    stable_weight = arrays["weight"]
    chains = arrays["chain"]
    positions = arrays["residue_index"]

    anchors = _groups(
        (stable_weight >= HIGH_SCORE) & (global_score <= LOW_SCORE),
        chains,
        positions,
        merge_gap=1,
    )
    coherent = [
        group
        for group in _groups(
            (global_score >= HIGH_SCORE) & (local_score <= LOW_SCORE),
            chains,
            positions,
            merge_gap=COHERENT_MERGE_GAP,
        )
        if group.size >= COHERENT_MIN_SELECTED
    ]
    local = [
        group
        for group in _groups(
            (local_score >= HIGH_SCORE) & (global_score < HIGH_SCORE),
            chains,
            positions,
            merge_gap=LOCAL_MERGE_GAP,
        )
        if LOCAL_MIN_SELECTED <= group.size <= LOCAL_MAX_SELECTED
    ]
    high_high = _groups(
        (global_score >= HIGH_SCORE) & (local_score >= HIGH_SCORE),
        chains,
        positions,
        merge_gap=1,
    )

    regions: list[Region] = []
    regions.extend(_region("stable_anchor", group, arrays) for group in anchors if group.size >= 3)
    regions.extend(_region("coherently_displaced", group, arrays) for group in coherent)
    regions.extend(_region("local_flexibility_peak", group, arrays) for group in local)
    regions.extend(_region("high_global_high_local", group, arrays) for group in high_high)

    boundary_members: dict[tuple[str, int], set[str]] = {}
    for coherent_number, group in enumerate(coherent, start=1):
        chain = str(chains[int(group[0])])
        start = int(positions[int(group[0])])
        end = int(positions[int(group[-1])])
        coherent_id = f"coherent_{coherent_number}"
        candidates = np.flatnonzero(
            (chains == chain)
            & (local_score >= HIGH_SCORE)
            & ((np.abs(positions - start) <= BOUNDARY_DISTANCE) | (np.abs(positions - end) <= BOUNDARY_DISTANCE))
        )
        for index in candidates:
            boundary_members.setdefault((chain, int(index)), set()).add(coherent_id)

    if boundary_members:
        mask = np.zeros(len(rows), dtype=bool)
        for _, index in boundary_members:
            mask[index] = True
        for group in _groups(mask, chains, positions, merge_gap=1):
            linked = sorted(
                {
                    coherent_id
                    for index in group
                    for coherent_id in boundary_members[(str(chains[int(index)]), int(index))]
                }
            )
            regions.append(
                _region(
                    "putative_boundary",
                    group,
                    arrays,
                    qualifying_coherent_region=",".join(linked),
                )
            )
    return regions


def region_rows(regions: Iterable[Region]) -> list[dict[str, Any]]:
    """Convert regions to stable table rows with per-type identifiers."""

    counts: dict[str, int] = {}
    output: list[dict[str, Any]] = []
    for region in regions:
        counts[region.region_type] = counts.get(region.region_type, 0) + 1
        output.append(region.as_dict(f"{region.region_type}_{counts[region.region_type]}") )
    return output


def region_mask(regions: Iterable[Region], size: int, region_type: str) -> np.ndarray:
    """Return a residue mask for one region type."""

    mask = np.zeros(size, dtype=bool)
    for region in regions:
        if region.region_type == region_type:
            mask[np.asarray(region.indices, dtype=np.int64)] = True
    return mask

