"""Create and summarize reviewer pre-mortem feedback."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import FINAL_SUBMISSION_ROOT, read_table, timestamp, write_manifest, write_table  # noqa: E402


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "reviewer_premortem"
FIELDS = [
    "reviewer_id",
    "main_contribution_one_sentence",
    "weakest_claim",
    "hardest_figure",
    "requested_analysis",
    "readme_install_easy",
    "reference_pbc_limitation_adequate",
    "journal_recommendation",
    "overall_score_1_to_5",
    "free_comments",
]
SUMMARY_FIELDS = [
    "feedback_status",
    "n_reviewers",
    "common_weakest_claim",
    "common_hardest_figure",
    "common_requested_analysis",
    "readme_install_easy_rate",
    "limitation_adequate_rate",
    "journal_recommendation_mode",
    "mean_overall_score",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--feedback-file", type=Path, default=OUTPUT_DIR / "reviewer_premortem_feedback.tsv")
    return parser.parse_args()


def common(rows: list[dict[str, str]], field: str) -> str:
    """Return most common non-empty value."""

    values = [row.get(field, "").strip() for row in rows if row.get(field, "").strip()]
    return Counter(values).most_common(1)[0][0] if values else ""


def yes_rate(rows: list[dict[str, str]], field: str) -> str:
    """Return yes-rate."""

    if not rows:
        return ""
    yes = sum(row.get(field, "").strip().lower() in {"yes", "true", "1"} for row in rows)
    return f"{yes / len(rows):.3f}"


def mean_score(rows: list[dict[str, str]]) -> str:
    """Return mean reviewer score."""

    values = []
    for row in rows:
        try:
            values.append(float(row.get("overall_score_1_to_5", "")))
        except ValueError:
            pass
    return f"{sum(values) / len(values):.3f}" if values else ""


def main() -> int:
    """Create reviewer pre-mortem template and summary."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = args.output_dir / "reviewer_premortem_template.tsv"
    rows = read_table(args.feedback_file) if args.feedback_file.exists() else []
    status = "completed" if rows else "pending"
    summary_rows = [
        {
            "feedback_status": status,
            "n_reviewers": len(rows),
            "common_weakest_claim": common(rows, "weakest_claim"),
            "common_hardest_figure": common(rows, "hardest_figure"),
            "common_requested_analysis": common(rows, "requested_analysis"),
            "readme_install_easy_rate": yes_rate(rows, "readme_install_easy"),
            "limitation_adequate_rate": yes_rate(rows, "reference_pbc_limitation_adequate"),
            "journal_recommendation_mode": common(rows, "journal_recommendation"),
            "mean_overall_score": mean_score(rows),
            "notes": "pending feedback is acceptable before internal pre-review begins" if not rows else "summarize honestly in internal planning",
        }
    ]
    summary_table = args.output_dir / "reviewer_premortem_summary.tsv"
    summary_md = args.output_dir / "reviewer_premortem_summary.md"
    write_table(template, [], FIELDS)
    write_table(summary_table, summary_rows, SUMMARY_FIELDS)
    summary_md.write_text(
        "\n".join(
            [
                "# Reviewer Pre-Mortem Summary",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Feedback status: `{status}`",
                f"- Reviewers summarized: `{len(rows)}`",
                f"- Weakest claim: `{summary_rows[0]['common_weakest_claim'] or 'pending'}`",
                f"- Hardest figure: `{summary_rows[0]['common_hardest_figure'] or 'pending'}`",
                f"- Requested analysis: `{summary_rows[0]['common_requested_analysis'] or 'pending'}`",
                "- Pending status is acceptable and should not be treated as fabricated reviewer support.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/summarize_reviewer_premortem.py",
        inputs=[ROOT / "docs" / "reviewer_premortem.md", args.feedback_file],
        outputs=[template, summary_table, summary_md],
        extra={"feedback_status": status, "n_reviewers": len(rows)},
    )
    print(f"Wrote reviewer pre-mortem outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
