"""Sparse, alignment-free distance descriptors for experimental window scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SparseEdgeSet:
    """A deterministic sparse residue graph."""

    source: np.ndarray
    target: np.ndarray
    family: np.ndarray
    n_residues: int
    continuity_breaks: np.ndarray

    @property
    def n_edges(self) -> int:
        """Return the number of sparse edges."""

        return int(self.source.size)


def validate_coordinates(coords: np.ndarray) -> np.ndarray:
    """Validate a finite C-alpha trajectory with shape ``(T, N, 3)``."""

    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    if values.shape[0] < 1 or values.shape[1] < 3:
        raise ValueError("coords must contain at least one frame and three residues")
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    return values


def continuity_breaks(
    n_residues: int,
    *,
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Mark sequence breaks between adjacent coordinate rows.

    A break is introduced at a chain change. When integer residue positions are
    supplied, a non-unit position increment also creates a break.
    """

    n_residues = int(n_residues)
    if n_residues < 1:
        raise ValueError("n_residues must be positive")
    breaks = np.zeros(max(0, n_residues - 1), dtype=bool)
    if chain_ids is not None:
        chains = np.asarray(chain_ids)
        if chains.shape != (n_residues,):
            raise ValueError("chain_ids must have shape (n_residues,)")
        breaks |= chains[1:] != chains[:-1]
    if residue_positions is not None:
        positions = np.asarray(residue_positions)
        if positions.shape != (n_residues,):
            raise ValueError("residue_positions must have shape (n_residues,)")
        if not np.issubdtype(positions.dtype, np.number):
            raise TypeError("residue_positions must be numeric")
        breaks |= np.diff(positions.astype(np.float64)) != 1.0
    return breaks


def segment_ids_from_breaks(n_residues: int, breaks: np.ndarray) -> np.ndarray:
    """Convert adjacent-break flags into contiguous segment identifiers."""

    values = np.asarray(breaks, dtype=bool)
    if values.shape != (max(0, n_residues - 1),):
        raise ValueError("breaks has the wrong shape")
    segments = np.zeros(n_residues, dtype=np.int64)
    if n_residues > 1:
        segments[1:] = np.cumsum(values)
    return segments


def build_local_window_edges(
    n_residues: int,
    *,
    window_size: int = 5,
    feature_set: str = "minimal",
    chain_ids: np.ndarray | None = None,
    residue_positions: np.ndarray | None = None,
) -> SparseEdgeSet:
    """Build unique local edges without crossing chain or residue gaps.

    ``minimal`` omits adjacent C-alpha bonds and retains sequence separations
    2..``window_size - 1``. ``full`` additionally includes adjacent pairs.
    This unique-edge representation is equivalent to collecting the valid
    pair features from every centred window while avoiding duplicate work.
    """

    n_residues = int(n_residues)
    window_size = int(window_size)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if feature_set not in {"minimal", "full"}:
        raise ValueError("feature_set must be 'minimal' or 'full'")
    breaks = continuity_breaks(
        n_residues,
        chain_ids=chain_ids,
        residue_positions=residue_positions,
    )
    segments = segment_ids_from_breaks(n_residues, breaks)
    minimum_separation = 2 if feature_set == "minimal" else 1
    maximum_separation = window_size - 1
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for separation in range(minimum_separation, maximum_separation + 1):
        source = np.arange(0, n_residues - separation, dtype=np.int64)
        target = source + separation
        keep = segments[source] == segments[target]
        sources.append(source[keep])
        targets.append(target[keep])
    if not sources or sum(item.size for item in sources) == 0:
        raise ValueError("topology does not provide any valid local window edges")
    source = np.concatenate(sources)
    target = np.concatenate(targets)
    return SparseEdgeSet(
        source=source,
        target=target,
        family=np.full(source.size, f"local_w{window_size}", dtype="U24"),
        n_residues=n_residues,
        continuity_breaks=breaks,
    )


def combine_edge_sets(*edge_sets: SparseEdgeSet) -> SparseEdgeSet:
    """Combine sparse edge sets and remove exact family-aware duplicates."""

    if not edge_sets:
        raise ValueError("at least one edge set is required")
    n_residues = edge_sets[0].n_residues
    breaks = edge_sets[0].continuity_breaks
    rows: list[tuple[int, int, str]] = []
    for edges in edge_sets:
        if edges.n_residues != n_residues:
            raise ValueError("all edge sets must use the same residue count")
        for source, target, family in zip(
            edges.source,
            edges.target,
            edges.family,
            strict=True,
        ):
            left, right = sorted((int(source), int(target)))
            rows.append((left, right, str(family)))
    rows = sorted(set(rows), key=lambda row: (row[2], row[0], row[1]))
    return SparseEdgeSet(
        source=np.asarray([row[0] for row in rows], dtype=np.int64),
        target=np.asarray([row[1] for row in rows], dtype=np.int64),
        family=np.asarray([row[2] for row in rows], dtype="U24"),
        n_residues=n_residues,
        continuity_breaks=breaks.copy(),
    )


def edge_squared_distances(
    coords: np.ndarray,
    edges: SparseEdgeSet,
    *,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Evaluate squared distances for a sparse graph.

    The returned array has shape ``(n_frames, n_edges)``. No dense residue by
    residue distance matrix is constructed.
    """

    values = validate_coordinates(coords)
    if values.shape[1] != edges.n_residues:
        raise ValueError("coords and edges use different residue counts")
    n_frames = values.shape[0]
    if chunk_size is None:
        chunk_size = n_frames
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output = np.empty((n_frames, edges.n_edges), dtype=np.float64)
    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        delta = values[start:stop, edges.source] - values[start:stop, edges.target]
        output[start:stop] = np.einsum("tei,tei->te", delta, delta)
    return output


def scalar_edge_squared_distances(
    coords: np.ndarray,
    edges: SparseEdgeSet,
) -> np.ndarray:
    """Slow scalar reference used to test vectorised descriptor parity."""

    values = validate_coordinates(coords)
    output = np.empty((values.shape[0], edges.n_edges), dtype=np.float64)
    for frame_index in range(values.shape[0]):
        for edge_index, (source, target) in enumerate(
            zip(edges.source, edges.target, strict=True)
        ):
            delta = values[frame_index, source] - values[frame_index, target]
            output[frame_index, edge_index] = float(delta @ delta)
    return output
