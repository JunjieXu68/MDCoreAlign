"""Create and summarize external user-testing feedback templates."""

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


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "user_testing"
FEEDBACK_FIELDS = [
    "tester_id",
    "tester_background",
    "operating_system",
    "python_version",
    "install_method",
    "installation_success",
    "time_to_install_minutes",
    "quickstart_success",
    "time_to_first_result_minutes",
    "cli_help_clear",
    "output_files_created",
    "understood_mobility_score",
    "understood_robust_RMSF",
    "understood_flexible_segments",
    "main_friction",
    "documentation_gap",
    "bug_or_error_message",
    "overall_usability_1_to_5",
    "would_use_for_MD_trajectory",
    "notes",
]
SUMMARY_FIELDS = [
    "feedback_status",
    "n_testers",
    "installation_success_rate",
    "quickstart_success_rate",
    "median_time_to_first_result_minutes",
    "common_friction",
    "documentation_gaps",
    "bugs_reported",
    "overall_usability_mean",
    "supplementary_recommendation",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--feedback-file", type=Path, default=OUTPUT_DIR / "user_feedback.tsv")
    return parser.parse_args()


def _yes_rate(rows: list[dict[str, str]], field: str) -> str:
    """Return yes-rate as text."""

    if not rows:
        return ""
    yes = sum(row.get(field, "").strip().lower() in {"yes", "true", "1", "pass"} for row in rows)
    return f"{yes / len(rows):.3f}"


def _mean(rows: list[dict[str, str]], field: str) -> str:
    """Return numeric mean if available."""

    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, "")))
        except ValueError:
            pass
    return f"{sum(values) / len(values):.3f}" if values else ""


def _median(rows: list[dict[str, str]], field: str) -> str:
    """Return numeric median if available."""

    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, "")))
        except ValueError:
            pass
    if not values:
        return ""
    values.sort()
    return f"{values[len(values) // 2]:.3f}"


def _common(rows: list[dict[str, str]], field: str) -> str:
    """Return common non-empty text values."""

    values = [row.get(field, "").strip() for row in rows if row.get(field, "").strip()]
    return "; ".join(item for item, _ in Counter(values).most_common(3))


def main() -> int:
    """Write user-testing template and summary."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = args.output_dir / "user_feedback_template.tsv"
    write_table(template, [], FEEDBACK_FIELDS)
    rows = read_table(args.feedback_file) if args.feedback_file.exists() else []
    status = "completed" if rows else "pending"
    summary_rows = [
        {
            "feedback_status": status,
            "n_testers": len(rows),
            "installation_success_rate": _yes_rate(rows, "installation_success"),
            "quickstart_success_rate": _yes_rate(rows, "quickstart_success"),
            "median_time_to_first_result_minutes": _median(rows, "time_to_first_result_minutes"),
            "common_friction": _common(rows, "main_friction"),
            "documentation_gaps": _common(rows, "documentation_gap"),
            "bugs_reported": _common(rows, "bug_or_error_message"),
            "overall_usability_mean": _mean(rows, "overall_usability_1_to_5"),
            "supplementary_recommendation": "mention in Supplementary only after tester feedback exists" if not rows else "summarize in Supplementary user-testing note",
        }
    ]
    summary_table = args.output_dir / "user_testing_summary.tsv"
    summary_md = args.output_dir / "user_testing_summary.md"
    write_table(summary_table, summary_rows, SUMMARY_FIELDS)
    summary_md.write_text(
        "\n".join(
            [
                "# External User Testing Summary",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Feedback status: `{status}`",
                f"- External testers participated: `{len(rows)}`",
                f"- Installation success rate: `{summary_rows[0]['installation_success_rate'] or 'not_applicable'}`",
                f"- Toy quickstart success rate: `{summary_rows[0]['quickstart_success_rate'] or 'not_applicable'}`",
                f"- Main usability issues: `{summary_rows[0]['common_friction'] or 'pending'}`",
                f"- Documentation gaps improved: `{summary_rows[0]['documentation_gaps'] or 'pending'}`",
                "- Missing feedback is not a blocker before user testing begins.",
                "",
            ]
        )
    )
    write_manifest(
        args.output_dir,
        script="scripts/summarize_user_testing_feedback.py",
        inputs=[ROOT / "docs" / "user_testing_protocol.md", args.feedback_file],
        outputs=[template, summary_table, summary_md],
        extra={"feedback_status": status, "n_testers": len(rows)},
    )
    print(f"Wrote user-testing outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
