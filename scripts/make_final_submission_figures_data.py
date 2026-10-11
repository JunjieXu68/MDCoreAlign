"""Create final submission figure-data and user-decision tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import FINAL_SUBMISSION_ROOT, read_table, timestamp, write_manifest, write_table  # noqa: E402


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "final_figures"
RECOMMENDATION_FIELDS = ["user_concern", "recommended_MDCoreAlign_check", "interpretation", "main_or_supplementary_output"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def copy_if_available(source: Path, target: Path) -> None:
    """Copy a table if available, otherwise write an empty placeholder."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, target)
    else:
        target.write_text("")


def write_workflow(path: Path) -> None:
    """Write workflow decision-tree figure note."""

    path.write_text(
        "\n".join(
            [
                "# Figure 1 Workflow / Decision Tree",
                "",
                "1. Start with a preprocessed conformational ensemble.",
                "2. Run ordinary full-set RMSD fitting as a familiar reference.",
                "3. Run robust stable-core alignment to estimate soft core membership.",
                "4. Audit preprocessing uncertainty with displacement and centering checks.",
                "5. Check temporal block stability for unstable core assignments.",
                "6. Check stride and trajectory-length dependence for sampling effects.",
                "7. Check reference sensitivity; do not claim reference independence.",
                "8. Interpret robust RMSF and flexible segments as alignment-frame-dependent profiles.",
                "",
                "This workflow is a decision tree for trustworthy use, not only an algorithm diagram.",
                "",
            ]
        )
    )


def write_recommendations(path: Path) -> None:
    """Write user recommendation table."""

    rows = [
        {
            "user_concern": "oversampled frames",
            "recommended_MDCoreAlign_check": "stride sensitivity",
            "interpretation": "similar summaries across strides support robustness to frame thinning",
            "main_or_supplementary_output": "supplementary",
        },
        {
            "user_concern": "insufficient trajectory length",
            "recommended_MDCoreAlign_check": "length dependence",
            "interpretation": "plateaus or large changes help judge whether the analyzed window is long enough",
            "main_or_supplementary_output": "main summary plus supplementary table",
        },
        {
            "user_concern": "reference choice effects",
            "recommended_MDCoreAlign_check": "reference sensitivity / consensus",
            "interpretation": "reference sensitivity is diagnostic and should not be hidden",
            "main_or_supplementary_output": "main caveat plus supplementary details",
        },
        {
            "user_concern": "unstable core assignment",
            "recommended_MDCoreAlign_check": "temporal block stability",
            "interpretation": "block agreement supports stable-core interpretation; disagreement marks caution",
            "main_or_supplementary_output": "supplementary",
        },
        {
            "user_concern": "mobile/flexible residues",
            "recommended_MDCoreAlign_check": "robust RMSF + mobility score",
            "interpretation": "RMSF is alignment-frame dependent and should be read with mobility weights",
            "main_or_supplementary_output": "main figure candidate",
        },
        {
            "user_concern": "PBC/preprocessing uncertainty",
            "recommended_MDCoreAlign_check": "preprocessing audit",
            "interpretation": "lightweight checks do not prove PBC/nojump status",
            "main_or_supplementary_output": "methods limitation plus supplementary table",
        },
        {
            "user_concern": "reproducibility",
            "recommended_MDCoreAlign_check": "clean install + toy quickstart",
            "interpretation": "verifies user-facing package path without external baselines",
            "main_or_supplementary_output": "GitHub docs plus supplementary",
        },
    ]
    write_table(path, rows, RECOMMENDATION_FIELDS)


def main() -> int:
    """Create final submission figure data package."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig1 = args.output_dir / "fig1_workflow_decision_tree.md"
    fig2 = args.output_dir / "fig2_md_benchmark_summary.tsv"
    fig3 = args.output_dir / "fig3_md_trajectory_diagnostics_summary.tsv"
    fig4 = args.output_dir / "fig4_flexibility_interpretation_summary.tsv"
    rec = args.output_dir / "table_user_recommendations.tsv"
    captions = args.output_dir / "main_figure_captions_final.md"
    supplement = args.output_dir / "supplementary_figure_allocation.md"
    write_workflow(fig1)
    copy_if_available(ROOT / "results" / "submission_polish" / "main_figures" / "fig2_benchmark_summary.tsv", fig2)
    copy_if_available(ROOT / "results" / "submission_polish" / "main_figures" / "fig3_md_diagnostics_summary.tsv", fig3)
    copy_if_available(ROOT / "results" / "submission_polish" / "main_figures" / "fig4_flexibility_profile_summary.tsv", fig4)
    write_recommendations(rec)
    captions.write_text(
        "\n".join(
            [
                "# Final Main Figure Captions",
                "",
                "## Figure 1. MDCoreAlign workflow for robust stable-core alignment and flexibility profiling.",
                "The workflow includes ordinary RMSD fitting, robust stable-core alignment, preprocessing audit, temporal/stride/length diagnostics, reference sensitivity, and RMSF/flexible segment interpretation.",
                "",
                "## Figure 2. Formal MD trajectory benchmark summary.",
                "Runtime and unified post-hoc RMSD are summarized from validated formal MD trajectory tables. External baselines are reported only when available and should not be described as rerun unless explicitly rerun in the current environment.",
                "",
                "## Figure 3. MD trajectory diagnostics.",
                "Temporal block, stride, trajectory-length, reference-sensitivity, and warm-start diagnostics are summarized. Reference sensitivity is diagnostic, not a failure, and no reference independence is claimed.",
                "",
                "## Figure 4. Flexibility interpretation.",
                "Robust RMSF, soft mobility score, and flexible segment summaries are shown. RMSF is alignment-frame dependent.",
                "",
            ]
        )
    )
    supplement.write_text(
        "\n".join(
            [
                "# Supplementary Figure Allocation",
                "",
                "- Full preprocessing audit tables.",
                "- Recentered diagnostic comparison.",
                "- Temporal convergence classifications.",
                "- Reference-consensus availability/status.",
                "- Optional CA-vs-backbone and external MD mini-validation status.",
                "- Clean-install, TestPyPI, and reproducibility capsule logs.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/make_final_submission_figures_data.py",
        inputs=[
            ROOT / "results" / "submission_polish" / "main_figures" / "fig2_benchmark_summary.tsv",
            ROOT / "results" / "submission_polish" / "main_figures" / "fig3_md_diagnostics_summary.tsv",
            ROOT / "results" / "submission_polish" / "main_figures" / "fig4_flexibility_profile_summary.tsv",
        ],
        outputs=[fig1, fig2, fig3, fig4, rec, captions, supplement],
        extra={
            "fig2_rows": len(read_table(fig2)),
            "fig3_rows": len(read_table(fig3)),
            "fig4_rows": len(read_table(fig4)),
            "generated_at": timestamp(),
        },
    )
    print(f"Wrote final submission figure data to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
