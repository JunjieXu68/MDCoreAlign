"""Optional CA-vs-backbone atom-selection sensitivity status report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_quality_common import FINAL_ROOT, read_table, timestamp, write_manifest, write_table  # noqa: E402


OUTPUT_DIR = FINAL_ROOT / "atom_selection_sensitivity"
FLEX_SUMMARY = SCRIPT_DIR.parent / "results" / "flexibility" / "flexibility_summary_four_systems.tsv"

FIELDS = [
    "system",
    "selection_label",
    "selection_query",
    "n_atoms",
    "n_residues",
    "runtime_algorithm_seconds",
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "effective_core_size",
    "n_core_residues",
    "n_core_segments",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "n_flexible_segments",
    "converged",
    "n_iterations",
    "notes",
]

COMPARISON_FIELDS = [
    "system",
    "core_jaccard_CA_vs_backbone",
    "mobility_spearman_CA_vs_backbone",
    "rmsf_spearman_CA_vs_backbone",
    "flexible_segment_overlap_CA_vs_backbone",
    "runtime_ratio_backbone_over_CA",
    "interpretation",
]

SYSTEMS = ["ADK equilibrium", "NhaA equilibrium"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--flex-summary", type=Path, default=FLEX_SUMMARY)
    return parser.parse_args()


def build_rows(flex_summary: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build conservative atom-selection sensitivity rows."""

    flex = {row["system"]: row for row in read_table(flex_summary)} if flex_summary.exists() else {}
    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for system in SYSTEMS:
        row = flex.get(system, {})
        rows.append(
            {
                "system": system,
                "selection_label": "CA",
                "selection_query": "protein and name CA",
                "n_atoms": row.get("n_residues", ""),
                "n_residues": row.get("n_residues", ""),
                "runtime_algorithm_seconds": row.get("runtime_seconds", ""),
                "unified_posthoc_RMSD": "",
                "weighted_RMSD": "",
                "effective_core_size": "",
                "n_core_residues": "",
                "n_core_segments": "",
                "mean_core_RMSF": row.get("mean_RMSF_core", ""),
                "mean_noncore_RMSF": row.get("mean_RMSF_noncore", ""),
                "noncore_core_RMSF_ratio": "",
                "spearman_robust_RMSF_vs_mobility": row.get("spearman_RMSF_vs_mobility", ""),
                "n_flexible_segments": row.get("n_flexible_segments", ""),
                "converged": "from_existing_formal_CA_profile",
                "n_iterations": "",
                "notes": "Existing formal CA-only flexibility profile reused; no new formal rerun performed.",
            }
        )
        rows.append(
            {
                "system": system,
                "selection_label": "backbone",
                "selection_query": "protein and backbone",
                "converged": "unavailable",
                "notes": "Backbone selection sensitivity not run in this release-candidate pass because residue-level backbone-to-CA aggregation/matching protocol is not yet frozen; no array-index matching was used.",
            }
        )
        comparisons.append(
            {
                "system": system,
                "core_jaccard_CA_vs_backbone": "",
                "mobility_spearman_CA_vs_backbone": "",
                "rmsf_spearman_CA_vs_backbone": "",
                "flexible_segment_overlap_CA_vs_backbone": "",
                "runtime_ratio_backbone_over_CA": "",
                "interpretation": "not_applicable: backbone mapping/aggregation requires a dedicated validated protocol; keep CA-only as primary formal representation for now.",
            }
        )
    return rows, comparisons


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    return "\n".join(
        [
            "# Optional Atom-Selection Sensitivity",
            "",
            f"- Generated at: `{timestamp()}`",
            "- Status: `not_applicable_for_backbone_comparison_in_release_candidate`",
            "- CA-only formal profiles are summarized for two representative systems.",
            "- Backbone sensitivity was not forced because no validated residue-level backbone-to-CA aggregation protocol is frozen in the release candidate.",
            "",
            "## Required Answers",
            "",
            "1. Are stable-core conclusions similar between CA and backbone selections? Not evaluated.",
            "2. How much runtime overhead does backbone selection add? Not evaluated.",
            "3. Should CA be recommended as default? Yes for the current formal release-candidate workflow.",
            "4. Should atom-selection sensitivity be main text or supplementary? Supplementary/future validation unless a dedicated backbone rerun is added.",
            "",
            "No backbone result was inferred from array index alone.",
            "",
        ]
    )


def main() -> int:
    """Run atom-selection sensitivity status analysis."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, comparisons = build_rows(args.flex_summary)
    table = args.output_dir / "atom_selection_sensitivity.tsv"
    comparison_table = args.output_dir / "atom_selection_comparison.tsv"
    summary = args.output_dir / "atom_selection_sensitivity_summary.md"
    write_table(table, rows, FIELDS)
    write_table(comparison_table, comparisons, COMPARISON_FIELDS)
    summary.write_text(build_summary(rows))
    write_manifest(
        args.output_dir,
        script="scripts/analyze_atom_selection_sensitivity.py",
        inputs=[args.flex_summary],
        outputs=[table, comparison_table, summary],
        extra={
            "status": "not_applicable_for_backbone_comparison_in_release_candidate",
            "reason": "backbone residue aggregation/matching protocol not frozen; no index-only mapping used",
            "random_seed": None,
        },
    )
    print(f"Wrote atom-selection sensitivity status to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
