"""Final PBC/nojump/centering verification addendum for formal MD inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.formal_flexibility_profiles import DATASETS  # noqa: E402
from final_quality_common import FINAL_ROOT, timestamp, write_json, write_manifest, write_table  # noqa: E402
from md_trajectory_focus_common import (  # noqa: E402
    FORMAL_SYSTEMS,
    MDSolveSummary,
    Trajectory,
    compare_to_reference,
    compute_rmsf,
    jaccard,
    load_formal_system,
    solve_default_soft,
    spearman_correlation,
    subset_loaded_system,
)


OUTPUT_DIR = FINAL_ROOT / "preprocessing_verification"
LARGE_DISPLACEMENT_THRESHOLD_A = 15.0
CENTER_DRIFT_THRESHOLD_A = 5.0

FIELDS = [
    "system",
    "trajectory_file",
    "topology_file",
    "selection",
    "n_frames_checked",
    "n_selected_atoms",
    "box_vectors_available",
    "center_of_geometry_drift_A",
    "max_frame_to_frame_CA_displacement_A",
    "large_jump_detected",
    "large_jump_threshold_A",
    "chain_continuity_status",
    "apparent_centering_status",
    "pbc_nojump_status",
    "recentered_rerun_performed",
    "core_jaccard_after_recenter",
    "mobility_spearman_after_recenter",
    "rmsf_spearman_after_recenter",
    "flexible_segment_overlap_after_recenter",
    "recommendation",
    "notes",
]

RECENTER_FIELDS = [
    "system",
    "recentered_rerun_performed",
    "runtime_seconds",
    "core_jaccard_vs_original",
    "mobility_spearman_vs_original",
    "rmsf_spearman_vs_original",
    "flexible_segment_overlap_vs_original",
    "original_unified_RMSD",
    "recentered_unified_RMSD",
    "delta_unified_RMSD",
    "original_weighted_RMSD",
    "recentered_weighted_RMSD",
    "delta_weighted_RMSD",
    "notes",
]

SYSTEM_LABELS = {
    "ifabp": "I-FABP",
    "adk_equilibrium": "ADK",
    "nhaa_equilibrium": "NhaA",
    "yiip_equilibrium_short": "YiiP short",
}

LIMITATION_SENTENCE = (
    "Formal inputs were analyzed as provided; PBC/nojump and centering status "
    "could not be independently verified from local input metadata."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    parser.add_argument("--skip-recentered-comparison", action="store_true")
    return parser.parse_args()


def load_selected_coords_and_boxes(system_key: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load formal selected coordinates and box vectors with MDAnalysis."""

    import MDAnalysis as mda

    config = DATASETS[system_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe = mda.Universe(str(config["topology"]), str(config["trajectory"]))
    atoms = universe.select_atoms(str(config["selection"]))
    if atoms.n_atoms == 0:
        raise ValueError(f"selection returned no atoms for {system_key}: {config['selection']}")
    coords: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    stride = int(config["stride"])
    max_frames = int(config["max_frames"])
    for frame_index, _ in enumerate(universe.trajectory[::stride]):
        coords.append(atoms.positions.astype(float).copy())
        dimensions = getattr(universe.trajectory.ts, "dimensions", None)
        boxes.append(np.asarray(dimensions, dtype=float).copy() if dimensions is not None else np.full(6, np.nan))
        if frame_index + 1 >= max_frames:
            break
    metadata = []
    for atom_index, atom in enumerate(atoms):
        metadata.append(
            {
                "atom_index": atom_index,
                "chain": str(getattr(atom, "segid", "") or getattr(atom, "chainID", "") or "A"),
                "resid": int(getattr(atom, "resid", atom_index + 1)),
                "resname": str(getattr(atom, "resname", "UNK")),
            }
        )
    return np.asarray(coords, dtype=float), np.asarray(boxes, dtype=float), metadata


def chain_continuity(metadata: list[dict[str, object]]) -> str:
    """Check numeric residue monotonicity within inferred chains."""

    by_chain: dict[str, list[int]] = {}
    for row in metadata:
        by_chain.setdefault(str(row["chain"]), []).append(int(row["resid"]))
    for values in by_chain.values():
        if any(next_value < value for value, next_value in zip(values, values[1:], strict=False)):
            return "warning_nonmonotonic_residue_ids"
    return "pass_numeric_residue_ids_nondecreasing_within_chain"


def recenter_coords(coords: np.ndarray) -> np.ndarray:
    """Remove each frame's center of geometry as a lightweight diagnostic."""

    centers = np.mean(coords, axis=1, keepdims=True)
    return coords - centers


def compare_recentered(system_key: str) -> tuple[dict[str, object], dict[str, object]]:
    """Run original and recentered diagnostic comparisons for one system."""

    start = time.perf_counter()
    original = load_formal_system(system_key)
    recentered_system = subset_loaded_system(
        original,
        recenter_coords(original.trajectory.coords),
        note_suffix="center_of_geometry_removed_per_frame_for_diagnostic_only",
    )
    original_solution = solve_default_soft(original, warm_start=False, reference_strategy="first_frame")
    recentered_solution = solve_default_soft(recentered_system, warm_start=False, reference_strategy="first_frame")
    comparison = compare_to_reference(recentered_solution, original_solution, suffix="original")
    original_rmsf = compute_rmsf(original_solution.aligned_coords)
    recentered_rmsf = compute_rmsf(recentered_solution.aligned_coords)
    rmsf_spearman = spearman_correlation(recentered_rmsf, original_rmsf)
    original_rmsd = float(original_solution.metrics["unified_posthoc_RMSD"])
    recentered_rmsd = float(recentered_solution.metrics["unified_posthoc_RMSD"])
    original_weighted = float(original_solution.metrics["weighted_RMSD"])
    recentered_weighted = float(recentered_solution.metrics["weighted_RMSD"])
    recentered_row = {
        "system": SYSTEM_LABELS[system_key],
        "recentered_rerun_performed": True,
        "runtime_seconds": time.perf_counter() - start,
        "core_jaccard_vs_original": comparison["jaccard_core_vs_original"],
        "mobility_spearman_vs_original": comparison["spearman_mobility_vs_original"],
        "rmsf_spearman_vs_original": rmsf_spearman,
        "flexible_segment_overlap_vs_original": comparison["flexible_segment_overlap_vs_original"],
        "original_unified_RMSD": original_rmsd,
        "recentered_unified_RMSD": recentered_rmsd,
        "delta_unified_RMSD": recentered_rmsd - original_rmsd,
        "original_weighted_RMSD": original_weighted,
        "recentered_weighted_RMSD": recentered_weighted,
        "delta_weighted_RMSD": recentered_weighted - original_weighted,
        "notes": "Diagnostic recentering removes only per-frame center of geometry; it is not a PBC/nojump repair.",
    }
    inline = {
        "recentered_rerun_performed": True,
        "core_jaccard_after_recenter": recentered_row["core_jaccard_vs_original"],
        "mobility_spearman_after_recenter": recentered_row["mobility_spearman_vs_original"],
        "rmsf_spearman_after_recenter": recentered_row["rmsf_spearman_vs_original"],
        "flexible_segment_overlap_after_recenter": recentered_row["flexible_segment_overlap_vs_original"],
    }
    return inline, recentered_row


def summarize_system(system_key: str, *, skip_recentered: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    """Summarize PBC/centering diagnostics for one system."""

    config = DATASETS[system_key]
    try:
        coords, boxes, metadata = load_selected_coords_and_boxes(system_key)
    except Exception as exc:
        row = {
            "system": SYSTEM_LABELS[system_key],
            "trajectory_file": str(config.get("trajectory", "")),
            "topology_file": str(config.get("topology", "")),
            "selection": str(config.get("selection", "")),
            "n_frames_checked": 0,
            "n_selected_atoms": 0,
            "box_vectors_available": "unavailable",
            "center_of_geometry_drift_A": "",
            "max_frame_to_frame_CA_displacement_A": "",
            "large_jump_detected": "unavailable",
            "large_jump_threshold_A": LARGE_DISPLACEMENT_THRESHOLD_A,
            "chain_continuity_status": "unavailable",
            "apparent_centering_status": "unavailable",
            "pbc_nojump_status": "not_verified",
            "recentered_rerun_performed": False,
            "recommendation": "inspect local input files before manuscript use",
            "notes": f"{LIMITATION_SENTENCE} Loading failed: {exc}",
        }
        return row, None

    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=2) if coords.shape[0] > 1 else np.zeros((0, coords.shape[1]))
    max_displacement = float(np.max(diffs)) if diffs.size else 0.0
    large_jump = bool(np.any(diffs > LARGE_DISPLACEMENT_THRESHOLD_A)) if diffs.size else False
    centers = np.mean(coords, axis=1)
    center_delta = np.linalg.norm(centers - centers[0], axis=1)
    center_drift = float(np.max(center_delta)) if center_delta.size else 0.0
    box_lengths = boxes[:, :3] if boxes.size else np.empty((0, 3))
    finite_box = box_lengths[np.isfinite(box_lengths) & (box_lengths > 0.0)]
    box_available = finite_box.size > 0
    continuity = chain_continuity(metadata)
    apparent_centering = "appears_centered_by_drift_heuristic" if center_drift <= CENTER_DRIFT_THRESHOLD_A else "large_center_drift_detected"
    pbc_status = "not_independently_verified_from_metadata"
    recommendation = "state preprocessing limitation explicitly"
    if large_jump:
        recommendation = "state limitation and inspect possible imaging/jump behavior"
    recentered_inline = {
        "recentered_rerun_performed": False,
        "core_jaccard_after_recenter": "",
        "mobility_spearman_after_recenter": "",
        "rmsf_spearman_after_recenter": "",
        "flexible_segment_overlap_after_recenter": "",
    }
    recentered_row = None
    if not skip_recentered:
        try:
            recentered_inline, recentered_row = compare_recentered(system_key)
        except Exception as exc:
            recentered_inline["recentered_rerun_performed"] = False
            recentered_inline["core_jaccard_after_recenter"] = ""
            recentered_inline["mobility_spearman_after_recenter"] = ""
            recentered_inline["rmsf_spearman_after_recenter"] = ""
            recentered_inline["flexible_segment_overlap_after_recenter"] = ""
            recommendation = f"{recommendation}; recentered diagnostic unavailable"
            recentered_row = {
                "system": SYSTEM_LABELS[system_key],
                "recentered_rerun_performed": False,
                "notes": f"Recentered diagnostic failed: {exc}",
            }

    row = {
        "system": SYSTEM_LABELS[system_key],
        "trajectory_file": str(config["trajectory"]),
        "topology_file": str(config["topology"]),
        "selection": str(config["selection"]),
        "n_frames_checked": int(coords.shape[0]),
        "n_selected_atoms": int(coords.shape[1]),
        "box_vectors_available": bool(box_available),
        "center_of_geometry_drift_A": center_drift,
        "max_frame_to_frame_CA_displacement_A": max_displacement,
        "large_jump_detected": large_jump,
        "large_jump_threshold_A": LARGE_DISPLACEMENT_THRESHOLD_A,
        "chain_continuity_status": continuity,
        "apparent_centering_status": apparent_centering,
        "pbc_nojump_status": pbc_status,
        "recommendation": recommendation,
        "notes": LIMITATION_SENTENCE,
        **recentered_inline,
    }
    return row, recentered_row


def build_summary(rows: list[dict[str, object]], recentered_rows: list[dict[str, object]]) -> str:
    """Build Markdown summary answering required questions."""

    jump_systems = [row["system"] for row in rows if str(row["large_jump_detected"]).lower() == "true"]
    continuity_warnings = [row["system"] for row in rows if not str(row["chain_continuity_status"]).startswith("pass")]
    recentered_done = [row for row in recentered_rows if str(row.get("recentered_rerun_performed")).lower() == "true"]
    stable_recenter = [
        row["system"]
        for row in recentered_done
        if float(row.get("core_jaccard_vs_original", 0.0)) >= 0.99
        and float(row.get("mobility_spearman_vs_original", 0.0)) >= 0.99
    ]
    return "\n".join(
        [
            "# PBC/Nojump/Centering Verification Addendum",
            "",
            f"- Generated at: `{timestamp()}`",
            f"- Required manuscript limitation: {LIMITATION_SENTENCE}",
            "",
            "## Required Answers",
            "",
            "1. Were PBC/nojump and centering independently verifiable? No. Box vectors and coordinate sanity checks were inspected when available, but upstream PBC/nojump provenance could not be independently verified from local input metadata.",
            f"2. Did any system show large jumps or chain-continuity problems? Large-jump systems: `{', '.join(jump_systems) if jump_systems else 'none'}`. Chain-continuity warnings: `{', '.join(continuity_warnings) if continuity_warnings else 'none'}`.",
            f"3. Were conclusions stable after recentering, if tested? Recentered diagnostics were performed for `{len(recentered_done)}` systems; core/mobility stability at >=0.99 was observed for `{', '.join(stable_recenter) if stable_recenter else 'none'}`.",
            f"4. What exact preprocessing limitation should be stated in the manuscript? `{LIMITATION_SENTENCE}`",
            "5. Should the preprocessing manifest be updated? Yes, update manuscript-facing text to include this addendum as a sanity check, while retaining the original limitation that true PBC/nojump status is not proven.",
            "",
            "## Notes",
            "",
            "- Diagnostic recentering removes only per-frame center of geometry and is not a nojump/PBC correction.",
            "- Missing metadata or unavailable checks are treated as explicit warnings rather than silent passes.",
            "",
        ]
    )


def main() -> int:
    """Run final preprocessing verification."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    recentered_rows: list[dict[str, object]] = []
    for system in args.systems:
        row, recentered = summarize_system(system, skip_recentered=args.skip_recentered_comparison)
        rows.append(row)
        if recentered is not None:
            recentered_rows.append(recentered)

    table = args.output_dir / "pbc_centering_verification.tsv"
    summary = args.output_dir / "pbc_centering_verification_summary.md"
    recentered_table = args.output_dir / "recentered_comparison.tsv"
    write_table(table, rows, FIELDS)
    if recentered_rows:
        write_table(recentered_table, recentered_rows, RECENTER_FIELDS)
    else:
        write_table(recentered_table, [], RECENTER_FIELDS)
    summary.write_text(build_summary(rows, recentered_rows))
    write_manifest(
        args.output_dir,
        script="scripts/verify_md_preprocessing_pbc_centering.py",
        inputs=[Path(DATASETS[system]["trajectory"]) for system in args.systems]
        + [Path(DATASETS[system]["topology"]) for system in args.systems],
        outputs=[table, summary, recentered_table],
        extra={
            "systems": args.systems,
            "large_displacement_threshold_A": LARGE_DISPLACEMENT_THRESHOLD_A,
            "center_drift_threshold_A": CENTER_DRIFT_THRESHOLD_A,
            "limitation_sentence": LIMITATION_SENTENCE,
            "random_seed": None,
        },
    )
    print(f"Wrote final preprocessing verification to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
