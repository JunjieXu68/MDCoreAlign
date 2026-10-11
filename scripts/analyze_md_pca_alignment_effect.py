"""Optional PCA / essential-dynamics comparison for MD alignment choices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from md_trajectory_focus_common import (  # noqa: E402
    MD_FOCUS_ROOT,
    align_trajectory_ordinary,
    common_manifest,
    load_formal_system,
    solve_default_soft,
    spearman_correlation,
    timestamp,
    write_table,
)


OUTPUT_DIR = MD_FOCUS_ROOT / "pca_alignment_effect"

FIELDS = [
    "alignment",
    "n_frames",
    "n_residues",
    "pc1_variance_fraction",
    "pc2_variance_fraction",
    "pc3_variance_fraction",
    "pc1_pc2_projection_file",
    "spearman_pc1_loading_vs_mobility",
    "spearman_pc2_loading_vs_mobility",
    "notes",
]

PROJECTION_FIELDS = ["alignment", "frame_index", "pc1", "pc2"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--system", choices=["adk_equilibrium"], default="adk_equilibrium")
    return parser.parse_args()


def pca_components(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projections, variance fractions, and per-residue PC loading norms."""

    frames, atoms, _ = aligned.shape
    matrix = aligned.reshape(frames, atoms * 3)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    eigenvalues = singular**2
    total = float(np.sum(eigenvalues))
    fractions = eigenvalues / total if total > 0.0 else np.zeros_like(eigenvalues)
    projections = matrix @ vt[:3].T
    loadings = np.linalg.norm(vt[:3].reshape(3, atoms, 3), axis=2)
    return projections, fractions[:3], loadings


def run_analysis(system_key: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run PCA alignment-effect analysis."""

    system = load_formal_system(system_key)
    robust = solve_default_soft(system)
    ordinary_aligned = align_trajectory_ordinary(system.trajectory)
    datasets = [
        ("ordinary", ordinary_aligned),
        ("robust", robust.aligned_coords),
    ]
    summary_rows = []
    projection_rows = []
    for alignment, coords in datasets:
        projections, fractions, loadings = pca_components(coords)
        for frame_index, projection in enumerate(projections):
            projection_rows.append(
                {
                    "alignment": alignment,
                    "frame_index": frame_index,
                    "pc1": float(projection[0]),
                    "pc2": float(projection[1]) if projections.shape[1] > 1 else "",
                }
            )
        summary_rows.append(
            {
                "alignment": alignment,
                "n_frames": system.trajectory.n_frames,
                "n_residues": system.trajectory.n_atoms,
                "pc1_variance_fraction": float(fractions[0]) if fractions.shape[0] > 0 else "",
                "pc2_variance_fraction": float(fractions[1]) if fractions.shape[0] > 1 else "",
                "pc3_variance_fraction": float(fractions[2]) if fractions.shape[0] > 2 else "",
                "pc1_pc2_projection_file": "pca_projection_data.tsv",
                "spearman_pc1_loading_vs_mobility": spearman_correlation(loadings[0], robust.mobility),
                "spearman_pc2_loading_vs_mobility": spearman_correlation(loadings[1], robust.mobility) if loadings.shape[0] > 1 else "",
                "notes": "ADK-only optional PCA; mobility vector comes from default robust alignment.",
            }
        )
    return summary_rows, projection_rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    lines = [
        "# MD PCA Alignment Effect Summary",
        "",
        f"- Generated at: `{timestamp()}`",
        "",
        "## Required Questions",
        "",
        "1. Does alignment choice affect essential-dynamics interpretation? Compare PC variance fractions between ordinary and robust rows.",
        "2. Are high-mobility residues associated with dominant PCA loadings? Inspect Spearman PC-loading vs mobility columns.",
        "3. Should this analysis be main text or supplementary? Treat as Supplementary unless a strong ADK-specific interpretation is developed.",
    ]
    for row in rows:
        lines.append(
            f"- {row['alignment']}: PC1={float(row['pc1_variance_fraction']):.4g}, "
            f"PC2={float(row['pc2_variance_fraction']):.4g}."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run PCA analysis."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, projections = run_analysis(args.system)
    write_table(args.output_dir / "pca_alignment_effect.tsv", rows, FIELDS)
    write_table(args.output_dir / "pca_projection_data.tsv", projections, PROJECTION_FIELDS)
    (args.output_dir / "pca_alignment_effect_summary.md").write_text(build_summary(rows))
    common_manifest(
        script="scripts/analyze_md_pca_alignment_effect.py",
        output_dir=args.output_dir,
        systems=[args.system],
        output_files=[
            args.output_dir / "pca_alignment_effect.tsv",
            args.output_dir / "pca_alignment_effect_summary.md",
            args.output_dir / "pca_projection_data.tsv",
        ],
    )
    print(f"Wrote PCA alignment-effect outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
