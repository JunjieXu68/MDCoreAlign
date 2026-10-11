"""Audit submission-critical analysis readiness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import read_table, sha256_file, write_json, write_table  # noqa: E402
from submission_critical_common import ROOT  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "submission_critical" / "readiness_audit"

FIELDS = ["component", "status", "evidence", "notes"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def exists(path: Path) -> bool:
    """Return whether a path exists and is non-empty when it is a file."""

    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def _row(component: str, status: str, evidence: Path | str, notes: str) -> dict[str, object]:
    """Build one audit row."""

    evidence_text = str(evidence.relative_to(ROOT)) if isinstance(evidence, Path) and evidence.is_absolute() else str(evidence)
    return {"component": component, "status": status, "evidence": evidence_text, "notes": notes}


def stale_test_count_warnings() -> list[str]:
    """Return stale test-count mentions in main docs."""

    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "current_main_results_summary.md",
        ROOT / "docs" / "current_main_results_summary.txt",
        ROOT / "docs" / "rmsf_flexible_segments.md",
        ROOT / "results" / "flexibility" / "figures" / "figure_captions_draft.md",
    ]
    warnings = []
    pattern = re.compile(r"(\d+)\s+tests?\s+passed", re.IGNORECASE)
    for path in paths:
        if not path.exists():
            continue
        for match in pattern.finditer(path.read_text(errors="ignore")):
            count = int(match.group(1))
            if count < 90:
                warnings.append(f"{path.relative_to(ROOT)} contains stale low test count {count}")
    return warnings


def audit_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build readiness audit rows and blockers."""

    checks: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []

    required = [
        ("main benchmark tables", ROOT / "results" / "final_multi_system_summary_20260603" / "final_unified_multi_system_main_table.csv"),
        ("final statistics tables", ROOT / "results" / "final_statistics" / "final_statistics_summary.md"),
        ("regularization formal rerun", ROOT / "results" / "submission_critical" / "regularization_rerun" / "regularization_rerun_four_systems.tsv"),
        ("full diagnostics rerun", ROOT / "results" / "submission_critical" / "diagnostics_rerun" / "alignment_traces_four_systems.tsv"),
        ("THESEUS reproducibility status", ROOT / "results" / "submission_critical" / "theseus_reproducibility" / "theseus_rerun_or_validation.tsv"),
        ("true lambda sensitivity rerun", ROOT / "results" / "submission_critical" / "lambda_sensitivity_rerun" / "lambda_sensitivity_rerun.tsv"),
        ("figures exist", ROOT / "results" / "flexibility" / "figures" / "fig_rmsf_profiles_four_systems.pdf"),
        ("figure data exist", ROOT / "results" / "flexibility" / "figure_data" / "fig_rmsf_profiles.tsv"),
        ("test report exists", ROOT / "results" / "paper_ready" / "test_report.txt"),
        ("project manifest exists", ROOT / "results" / "paper_ready" / "project_state_manifest.json"),
    ]
    for component, path in required:
        status = "pass" if exists(path) else "blocker"
        row = _row(component, status, path, "required submission-critical artifact")
        checks.append(row)
        if status == "blocker":
            blockers.append(row)

    theseus_path = ROOT / "results" / "submission_critical" / "theseus_reproducibility" / "theseus_rerun_or_validation.tsv"
    if theseus_path.exists():
        rows = read_table(theseus_path)
        if rows and all(row.get("status") == "not_rerun_executable_missing" for row in rows):
            checks.append(
                _row(
                    "THESEUS current-environment rerun",
                    "warning",
                    theseus_path,
                    "THESEUS results were validated from existing tables but not rerun because executable was missing.",
                )
            )

    warnings = stale_test_count_warnings()
    checks.append(
        _row(
            "stale test-count scan",
            "warning" if warnings else "pass",
            "; ".join(warnings) if warnings else "README/docs/current summaries",
            "No stale low test-count mentions found." if not warnings else "Update docs before submission.",
        )
    )

    availability_dir = ROOT / "results" / "submission_critical" / "paper_text_updates"
    checks.append(
        _row(
            "code/data availability placeholders",
            "pass" if exists(availability_dir / "availability_reproducibility_update.md") else "warning",
            availability_dir / "availability_reproducibility_update.md",
            "Availability text draft should be reviewed before submission.",
        )
    )
    return checks, blockers


def recommendation(rows: list[dict[str, object]], blockers: list[dict[str, object]]) -> str:
    """Return final readiness recommendation."""

    if blockers:
        return "D. Not ready; scientific reruns required"
    warning_count = sum(row["status"] == "warning" for row in rows)
    if warning_count == 0:
        return "B. Ready for Bioinformatics submission pending author review"
    if warning_count <= 2:
        return "C. Almost ready; only minor documentation fixes remain"
    return "C. Almost ready; resolve warnings before Bioinformatics/BIB submission"


def build_summary(rows: list[dict[str, object]], blockers: list[dict[str, object]]) -> str:
    """Build audit Markdown summary."""

    rec = recommendation(rows, blockers)
    status_counts = {status: sum(row["status"] == status for row in rows) for status in ("pass", "warning", "blocker", "not_applicable")}
    lines = [
        "# Submission Readiness Audit",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        f"- Recommendation: `{rec}`.",
        f"- Status counts: `{status_counts}`.",
        "",
        "## Journal-Specific Readiness",
        "",
        "- BMC Bioinformatics: ready if blockers are zero and availability text is reviewed.",
        "- Bioinformatics: ready pending author review if blockers are zero; THESEUS-not-rerun should be transparently stated if applicable.",
        "- BIB: still benefits from cautious biological interpretation and external annotation review, especially for YiiP.",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        for row in blockers:
            lines.append(f"- {row['component']}: {row['notes']}")
    else:
        lines.append("- None detected by this audit.")
    return "\n".join(lines) + "\n"


def write_manifest(output_dir: Path) -> None:
    """Write manifest."""

    files = [
        output_dir / "submission_readiness_checklist.tsv",
        output_dir / "submission_readiness_summary.md",
        output_dir / "blockers.json",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "script": "scripts/audit_submission_readiness.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.exists()},
        },
    )


def main() -> int:
    """Run readiness audit."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, blockers = audit_rows()
    write_table(args.output_dir / "submission_readiness_checklist.tsv", rows, FIELDS)
    (args.output_dir / "submission_readiness_summary.md").write_text(build_summary(rows, blockers))
    write_json(args.output_dir / "blockers.json", {"blockers": blockers, "recommendation": recommendation(rows, blockers)})
    write_manifest(args.output_dir)
    print(f"Wrote readiness audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
