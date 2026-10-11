"""RMSF and flexible-segment profiling after ordinary or robust alignment."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re

import numpy as np

from .alignment import solve_rigid_alignment
from .core_selection import select_core_by_target_fraction
from .trajectory import Trajectory
from .trajectory_mode import TrajectoryAlignmentResult, solve_trajectory_robust_alignment
from .utils import as_float_array, ensure_xyz_frames


@dataclass(frozen=True)
class FlexibilityProfile:
    """Per-residue flexibility table derived from aligned trajectory coordinates."""

    residue_id: tuple[str, ...]
    resname: tuple[str, ...]
    chain: tuple[str, ...]
    robust_aligned_rmsf: np.ndarray
    ordinary_aligned_rmsf: np.ndarray | None
    soft_stability_weight: np.ndarray
    mobility_score: np.ndarray
    flexible_segment_id: np.ndarray
    rank: np.ndarray
    percentile: np.ndarray

    @property
    def n_residues(self) -> int:
        """Return the number of profiled residues or atoms."""

        return len(self.residue_id)

    def to_rows(self) -> list[dict[str, object]]:
        """Return CSV-ready rows with manuscript-facing column names."""

        rows: list[dict[str, object]] = []
        for index in range(self.n_residues):
            row: dict[str, object] = {
                    "residue_id": self.residue_id[index],
                    "resname": self.resname[index],
                    "chain": self.chain[index],
                    "robust_aligned_RMSF": float(self.robust_aligned_rmsf[index]),
                }
            if self.ordinary_aligned_rmsf is not None:
                row["ordinary_aligned_RMSF"] = float(
                    self.ordinary_aligned_rmsf[index]
                )
            row.update(
                {
                    "soft_stability_weight": float(self.soft_stability_weight[index]),
                    "mobility_score": float(self.mobility_score[index]),
                    "flexible_segment_id": int(self.flexible_segment_id[index]),
                    "rank": int(self.rank[index]),
                    "percentile": float(self.percentile[index]),
                }
            )
            rows.append(row)
        return rows


def compute_rmsf(aligned_coords: np.ndarray) -> np.ndarray:
    """Compute per-atom RMSF from an already aligned trajectory.

    RMSF is defined as:

    `sqrt(mean_t ||x_i(t) - mean_t x_i(t)||^2)`

    The input must have shape `(n_frames, n_atoms, 3)`.
    """

    aligned_coords = ensure_xyz_frames(aligned_coords)
    mean_coords = np.mean(aligned_coords, axis=0)
    deltas = aligned_coords - mean_coords[None, :, :]
    squared_displacements = np.einsum("fij,fij->fi", deltas, deltas)
    return np.sqrt(np.mean(squared_displacements, axis=0))


def align_trajectory_ordinary(
    trajectory: Trajectory | np.ndarray,
    *,
    reference_frame_index: int = 0,
) -> np.ndarray:
    """Align every frame to one reference frame with ordinary Kabsch fitting."""

    trajectory = trajectory if isinstance(trajectory, Trajectory) else Trajectory(coords=trajectory)
    if not 0 <= reference_frame_index < trajectory.n_frames:
        raise ValueError("reference_frame_index is outside the trajectory")

    reference = trajectory.coords[reference_frame_index]
    aligned = np.empty_like(trajectory.coords, dtype=float)
    for frame_index, frame in enumerate(trajectory.coords):
        aligned[frame_index] = solve_rigid_alignment(frame, reference).aligned_coords
    return aligned


def robust_aligned_coords_from_result(result: TrajectoryAlignmentResult) -> np.ndarray:
    """Stack robust-aligned frame coordinates from a trajectory-mode result."""

    aligned_coords = getattr(result, "aligned_coords", None)
    if aligned_coords is not None:
        return np.asarray(aligned_coords, dtype=float)
    if not result.frame_results:
        raise ValueError("trajectory alignment result contains no frames")
    return np.asarray([frame.result.aligned_coords for frame in result.frame_results], dtype=float)


def mean_soft_weights_from_result(result: TrajectoryAlignmentResult) -> np.ndarray:
    """Average per-frame soft stability weights into one trajectory-level vector."""

    frozen_weights = getattr(result, "frozen_weights", None)
    if frozen_weights is not None:
        return np.asarray(frozen_weights, dtype=float)
    if not result.frame_results:
        raise ValueError("trajectory alignment result contains no frames")
    return np.mean(np.asarray([frame.result.weights for frame in result.frame_results], dtype=float), axis=0)


def percentile_scores(values: np.ndarray) -> np.ndarray:
    """Convert values to percentiles where larger values receive larger scores."""

    values = _validate_vector(values, name="values")
    if values.shape[0] == 1:
        return np.ones(1, dtype=float)

    order = np.argsort(values, kind="mergesort")
    percentiles = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and np.isclose(values[order[end]], values[order[start]]):
            end += 1
        mean_position = 0.5 * (start + end - 1)
        percentiles[order[start:end]] = mean_position / (values.shape[0] - 1)
        start = end
    return percentiles


def descending_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values in descending order with deterministic index tie-breaking."""

    values = _validate_vector(values, name="values")
    order = np.lexsort((np.arange(values.shape[0]), -values))
    ranks = np.empty(values.shape[0], dtype=int)
    ranks[order] = np.arange(1, values.shape[0] + 1)
    return ranks


def detect_flexible_segments(
    mobility_score: np.ndarray,
    *,
    percentile_cutoff: float = 0.8,
    min_segment_length: int = 3,
    merge_gap: int = 0,
    chain: list[str] | tuple[str, ...] | np.ndarray | None = None,
) -> np.ndarray:
    """Assign contiguous high-mobility atoms or residues to flexible segments.

    Segment IDs start at `1`; `0` means the atom or residue is not part of a
    selected flexible segment. Contiguity is defined by input order and, when
    provided, is broken at chain boundaries.
    """

    mobility_score = _validate_vector(mobility_score, name="mobility_score")
    percentile_cutoff = float(percentile_cutoff)
    if not 0.0 <= percentile_cutoff <= 1.0:
        raise ValueError("percentile_cutoff must be in [0, 1]")
    min_segment_length = int(min_segment_length)
    if min_segment_length <= 0:
        raise ValueError("min_segment_length must be positive")
    merge_gap = int(merge_gap)
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")

    if chain is None:
        chain_values = np.asarray([""] * mobility_score.shape[0], dtype=object)
    else:
        chain_values = np.asarray(chain, dtype=object)
        if chain_values.shape != mobility_score.shape:
            raise ValueError(f"chain must have shape {mobility_score.shape}")

    high_mobility = percentile_scores(mobility_score) >= percentile_cutoff
    if merge_gap > 0:
        high_mobility = _merge_short_gaps(high_mobility, chain_values, merge_gap=merge_gap)
    segment_ids = np.zeros(mobility_score.shape[0], dtype=int)
    next_segment_id = 1
    start = 0
    while start < mobility_score.shape[0]:
        if not high_mobility[start]:
            start += 1
            continue
        end = start + 1
        while (
            end < mobility_score.shape[0]
            and high_mobility[end]
            and chain_values[end] == chain_values[start]
        ):
            end += 1
        if end - start >= min_segment_length:
            segment_ids[start:end] = next_segment_id
            next_segment_id += 1
        start = end
    return segment_ids


def build_flexibility_profile(
    trajectory: Trajectory | np.ndarray,
    *,
    residue_labels: list[str] | tuple[str, ...] | None = None,
    trajectory_result: TrajectoryAlignmentResult | None = None,
    reference_mode: str = "first_frame",
    warm_start: bool = True,
    flexible_percentile_cutoff: float = 0.8,
    min_segment_length: int = 3,
    merge_gap: int = 0,
    mobility_score_mode: str = "soft_weight",
    compute_ordinary_rmsf: bool = True,
    **robust_kwargs: object,
) -> FlexibilityProfile:
    """Build a per-residue RMSF and flexible-segment profile.

    The robust RMSF is computed after continuous-weight alignment. The optional
    ordinary RMSF comparison uses ordinary Kabsch alignment to the first frame.
    Soft stability weights are averaged across frames.
    """

    trajectory = trajectory if isinstance(trajectory, Trajectory) else Trajectory(coords=trajectory)
    residue_id, resname, chain = normalize_residue_metadata(residue_labels, n_atoms=trajectory.n_atoms)

    ordinary_aligned = (
        align_trajectory_ordinary(trajectory, reference_frame_index=0)
        if compute_ordinary_rmsf
        else None
    )
    if trajectory_result is None:
        trajectory_result = solve_trajectory_robust_alignment(
            trajectory,
            reference_mode=reference_mode,
            warm_start=warm_start,
            **robust_kwargs,
        )
    robust_aligned = robust_aligned_coords_from_result(trajectory_result)
    robust_rmsf = compute_rmsf(robust_aligned)
    ordinary_rmsf = (
        compute_rmsf(ordinary_aligned)
        if ordinary_aligned is not None
        else None
    )
    stability = np.clip(mean_soft_weights_from_result(trajectory_result), 0.0, 1.0)
    if mobility_score_mode == "soft_weight":
        mobility_score = np.clip(1.0 - stability, 0.0, 1.0)
    elif mobility_score_mode == "robust_rmsf":
        mobility_score = _minmax_scale(robust_rmsf)
    else:
        raise ValueError("mobility_score_mode must be 'soft_weight' or 'robust_rmsf'")
    percentiles = percentile_scores(mobility_score)
    ranks = descending_ranks(mobility_score)
    segment_ids = detect_flexible_segments(
        mobility_score,
        percentile_cutoff=flexible_percentile_cutoff,
        min_segment_length=min_segment_length,
        merge_gap=merge_gap,
        chain=chain,
    )

    return FlexibilityProfile(
        residue_id=residue_id,
        resname=resname,
        chain=chain,
        robust_aligned_rmsf=robust_rmsf,
        ordinary_aligned_rmsf=ordinary_rmsf,
        soft_stability_weight=stability,
        mobility_score=mobility_score,
        flexible_segment_id=segment_ids,
        rank=ranks,
        percentile=percentiles,
    )


def write_flexibility_profile_csv(profile: FlexibilityProfile, path: str | Path) -> None:
    """Write a flexibility profile table to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = profile.to_rows()
    if not rows:
        raise ValueError("profile must contain at least one row")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_extended_flexibility_rows(
    profile: FlexibilityProfile,
    *,
    core_fraction: float = 0.6,
    terminal_window: int = 5,
) -> list[dict[str, object]]:
    """Build the formal per-residue flexibility table used by real benchmarks."""

    core = select_core_by_target_fraction(profile.soft_stability_weight, core_fraction)
    terminal_mask = terminal_residue_mask(profile.chain, terminal_window=terminal_window)
    rows: list[dict[str, object]] = []
    for index, base_row in enumerate(profile.to_rows()):
        row = dict(base_row)
        row.update(
            {
                "core_label": int(bool(core.core_mask[index])),
                "is_flexible": int(profile.flexible_segment_id[index] > 0),
                "is_terminal": int(bool(terminal_mask[index])),
            }
        )
        if profile.ordinary_aligned_rmsf is not None:
            delta = (
                profile.robust_aligned_rmsf[index]
                - profile.ordinary_aligned_rmsf[index]
            )
            row["delta_RMSF"] = float(delta)
            row["abs_delta_RMSF"] = float(abs(delta))
        rows.append(row)
    return rows


def summarize_flexible_segments(
    profile: FlexibilityProfile,
    *,
    system: str,
    core_fraction: float = 0.6,
    terminal_window: int = 5,
) -> list[dict[str, object]]:
    """Summarize contiguous flexible segments into one row per segment."""

    core = select_core_by_target_fraction(profile.soft_stability_weight, core_fraction)
    soft_mobile_mask = ~core.core_mask
    terminal_mask = terminal_residue_mask(profile.chain, terminal_window=terminal_window)
    rows: list[dict[str, object]] = []
    for segment_id in sorted(set(profile.flexible_segment_id.tolist())):
        if segment_id <= 0:
            continue
        indices = np.flatnonzero(profile.flexible_segment_id == segment_id)
        if indices.shape[0] == 0:
            continue
        row: dict[str, object] = {
                "system": system,
                "segment_id": int(segment_id),
                "chain": profile.chain[int(indices[0])],
                "start_residue": profile.residue_id[int(indices[0])],
                "end_residue": profile.residue_id[int(indices[-1])],
                "length": int(indices.shape[0]),
                "mean_robust_RMSF": float(np.mean(profile.robust_aligned_rmsf[indices])),
                "max_robust_RMSF": float(np.max(profile.robust_aligned_rmsf[indices])),
                "mean_mobility_score": float(np.mean(profile.mobility_score[indices])),
                "mean_stability_weight": float(np.mean(profile.soft_stability_weight[indices])),
                "overlaps_soft_mobile_region": int(bool(np.any(soft_mobile_mask[indices]))),
                "overlaps_terminal_region": int(bool(np.any(terminal_mask[indices]))),
                "annotation": "",
            }
        if profile.ordinary_aligned_rmsf is not None:
            row["mean_ordinary_RMSF"] = float(
                np.mean(profile.ordinary_aligned_rmsf[indices])
            )
            row["mean_delta_RMSF"] = float(
                np.mean(
                    profile.robust_aligned_rmsf[indices]
                    - profile.ordinary_aligned_rmsf[indices]
                )
            )
        rows.append(row)
    return rows


def summarize_flexibility_profile(
    profile: FlexibilityProfile,
    *,
    system: str,
    core_fraction: float = 0.6,
) -> dict[str, object]:
    """Build one system-level flexibility summary row."""

    core = select_core_by_target_fraction(profile.soft_stability_weight, core_fraction)
    core_mask = core.core_mask
    noncore_mask = ~core_mask
    top_index = int(np.argmax(profile.mobility_score))
    top_segment_id = int(profile.flexible_segment_id[top_index])
    mean_core = _mean_masked(profile.robust_aligned_rmsf, core_mask)
    mean_noncore = _mean_masked(profile.robust_aligned_rmsf, noncore_mask)
    return {
        "system": system,
        "n_residues": profile.n_residues,
        "n_flexible_segments": int(np.max(profile.flexible_segment_id)) if profile.n_residues else 0,
        "top_flexible_residue": profile.residue_id[top_index],
        "top_flexible_residue_RMSF": float(profile.robust_aligned_rmsf[top_index]),
        "top_flexible_segment": top_segment_id,
        "max_robust_RMSF": float(np.max(profile.robust_aligned_rmsf)),
        "mean_robust_RMSF": float(np.mean(profile.robust_aligned_rmsf)),
        "mean_ordinary_RMSF": (
            None
            if profile.ordinary_aligned_rmsf is None
            else float(np.mean(profile.ordinary_aligned_rmsf))
        ),
        "spearman_RMSF_vs_mobility": spearman_correlation(profile.robust_aligned_rmsf, profile.mobility_score),
        "spearman_RMSF_vs_1_minus_stability": spearman_correlation(
            profile.robust_aligned_rmsf,
            1.0 - profile.soft_stability_weight,
        ),
        "spearman_ordinary_RMSF_vs_mobility": (
            None
            if profile.ordinary_aligned_rmsf is None
            else spearman_correlation(
                profile.ordinary_aligned_rmsf,
                profile.mobility_score,
            )
        ),
        "mean_RMSF_core": mean_core,
        "mean_RMSF_noncore": mean_noncore,
        "delta_core_noncore_RMSF": None if mean_core is None or mean_noncore is None else float(mean_noncore - mean_core),
    }


def build_rmsf_long_rows(
    profile: FlexibilityProfile,
    *,
    system: str,
    core_fraction: float = 0.6,
) -> list[dict[str, object]]:
    """Build plot-ready long-format RMSF rows for ordinary and robust alignment."""

    core = select_core_by_target_fraction(profile.soft_stability_weight, core_fraction)
    rows: list[dict[str, object]] = []
    for index in range(profile.n_residues):
        values = [("robust", profile.robust_aligned_rmsf[index])]
        if profile.ordinary_aligned_rmsf is not None:
            values.insert(0, ("ordinary", profile.ordinary_aligned_rmsf[index]))
        for alignment, rmsf in values:
            rows.append(
                {
                    "system": system,
                    "chain": profile.chain[index],
                    "residue_id": profile.residue_id[index],
                    "resname": profile.resname[index],
                    "alignment": alignment,
                    "rmsf": float(rmsf),
                    "mobility_score": float(profile.mobility_score[index]),
                    "stability_weight": float(profile.soft_stability_weight[index]),
                    "flexible_segment_id": int(profile.flexible_segment_id[index]),
                    "core_label": int(bool(core.core_mask[index])),
                }
            )
    return rows


def write_table(path: str | Path, rows: list[dict[str, object]], *, delimiter: str = "\t") -> None:
    """Write a CSV/TSV table from dictionary rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    """Write formatted JSON metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def terminal_residue_mask(
    chain: list[str] | tuple[str, ...] | np.ndarray,
    *,
    terminal_window: int = 5,
) -> np.ndarray:
    """Return a mask for N- and C-terminal windows of each chain."""

    chain = np.asarray(chain, dtype=object)
    if chain.ndim != 1:
        raise ValueError("chain must be one-dimensional")
    terminal_window = int(terminal_window)
    if terminal_window < 0:
        raise ValueError("terminal_window must be non-negative")
    mask = np.zeros(chain.shape[0], dtype=bool)
    if terminal_window == 0:
        return mask
    for chain_id in dict.fromkeys(chain.tolist()):
        indices = np.flatnonzero(chain == chain_id)
        if indices.shape[0] == 0:
            continue
        mask[indices[:terminal_window]] = True
        mask[indices[-terminal_window:]] = True
    return mask


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman correlation with average ranks and flat-vector handling."""

    x = _validate_vector(x, name="x")
    y = _validate_vector(y, name="y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.shape[0] < 2:
        return None
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator == 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def load_ca_multimodel_pdb_trajectory(
    path: str | Path,
    *,
    chain_id: str | None = None,
) -> tuple[Trajectory, tuple[str, ...]]:
    """Load CA-only frames and residue labels from a multi-model PDB ensemble."""

    path = Path(path)
    frames: list[list[list[float]]] = []
    labels_by_frame: list[list[str]] = []
    current_coords: list[list[float]] = []
    current_labels: list[str] = []

    def flush_frame() -> None:
        if current_coords:
            frames.append(list(current_coords))
            labels_by_frame.append(list(current_labels))
            current_coords.clear()
            current_labels.clear()

    with path.open() as handle:
        for line in handle:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                altloc = line[16].strip()
                if altloc not in ("", "A"):
                    continue
                chain = line[21].strip() or "_"
                if chain_id is not None and chain != chain_id:
                    continue
                resname = line[17:20].strip() or "UNK"
                resseq = line[22:26].strip()
                insertion_code = line[26].strip()
                label = f"{chain}:{resname}{resseq}{insertion_code}"
                current_labels.append(label)
                current_coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            elif line.startswith("ENDMDL"):
                flush_frame()
    flush_frame()

    if not frames:
        raise ValueError(f"no CA trajectory frames could be parsed from {path}")
    reference_labels = tuple(labels_by_frame[0])
    if not reference_labels:
        raise ValueError(f"no CA atoms could be parsed from the first frame of {path}")
    for frame_index, labels in enumerate(labels_by_frame):
        if tuple(labels) != reference_labels:
            raise ValueError(
                f"CA atom correspondence changed at frame {frame_index}; "
                "all models must have identical CA order and labels"
            )
    return Trajectory(coords=np.asarray(frames, dtype=float)), reference_labels


def normalize_residue_metadata(
    residue_labels: list[str] | tuple[str, ...] | None,
    *,
    n_atoms: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Normalize residue labels into residue ID, residue name, and chain arrays."""

    if residue_labels is None:
        residue_labels = tuple(f"A:UNK{index + 1}" for index in range(n_atoms))
    residue_labels = tuple(str(label) for label in residue_labels)
    if len(residue_labels) != n_atoms:
        raise ValueError(f"residue_labels must have length {n_atoms}")

    residue_ids: list[str] = []
    resnames: list[str] = []
    chains: list[str] = []
    for index, label in enumerate(residue_labels, start=1):
        chain, resname, residue_id = _parse_residue_label(label, fallback_index=index)
        residue_ids.append(residue_id)
        resnames.append(resname)
        chains.append(chain)
    return tuple(residue_ids), tuple(resnames), tuple(chains)


def _validate_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    """Validate one non-empty finite one-dimensional vector."""

    values = as_float_array(values)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """Scale finite values to [0, 1], returning zeros for a flat profile."""

    values = _validate_vector(values, name="values")
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(high, low):
        return np.zeros(values.shape[0], dtype=float)
    return (values - low) / (high - low)


def _mean_masked(values: np.ndarray, mask: np.ndarray) -> float | None:
    """Return the mean of selected values or None for an empty mask."""

    values = _validate_vector(values, name="values")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError(f"mask must have shape {values.shape}")
    if not np.any(mask):
        return None
    return float(np.mean(values[mask]))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks for Spearman correlation."""

    values = _validate_vector(values, name="values")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and np.isclose(values[order[end]], values[order[start]]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _merge_short_gaps(high_mobility: np.ndarray, chain_values: np.ndarray, *, merge_gap: int) -> np.ndarray:
    """Fill short low-mobility gaps between high-mobility runs within each chain."""

    merged = high_mobility.copy()
    for chain_id in dict.fromkeys(chain_values.tolist()):
        chain_indices = np.flatnonzero(chain_values == chain_id)
        if chain_indices.shape[0] == 0:
            continue
        chain_high = merged[chain_indices]
        high_positions = np.flatnonzero(chain_high)
        if high_positions.shape[0] < 2:
            continue
        for left, right in zip(high_positions[:-1], high_positions[1:], strict=True):
            gap = right - left - 1
            if 0 < gap <= merge_gap:
                merged[chain_indices[left + 1 : right]] = True
    return merged


def _jsonable(value):
    """Convert NumPy values into JSON-friendly objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_residue_label(label: str, fallback_index: int) -> tuple[str, str, str]:
    """Parse lightweight labels such as `A:GLY42` into profile metadata."""

    chain = "A"
    residue = label
    if ":" in label:
        chain_part, residue = label.split(":", 1)
        if len(chain_part) == 1:
            chain = chain_part or "A"

    match = re.match(r"(?P<resname>[A-Za-z]{3,4})(?P<resid>-?\d+[A-Za-z]?)$", residue.strip())
    if match is None:
        return chain, "UNK", str(fallback_index)
    return chain, match.group("resname").upper()[:3], match.group("resid")
