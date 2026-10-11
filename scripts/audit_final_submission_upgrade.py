"""Audit final submission-upgrade outputs and readiness."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import (  # noqa: E402
    FINAL_SUBMISSION_ROOT,
    file_nonempty,
    has_local_absolute_path,
    read_table,
    timestamp,
    write_json,
    write_manifest,
    write_table,
)


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "final_audit"
FIELDS = ["task", "check", "status", "evidence", "notes"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def add(rows: list[dict[str, str]], task: str, check: str, status: str, evidence: str, notes: str) -> None:
    """Append one checklist row."""

    rows.append({"task": task, "check": check, "status": status, "evidence": evidence, "notes": notes})


def json_value(path: Path, key: str, default: object = None) -> object:
    """Read one JSON value if available."""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text()).get(key, default)
    except Exception:
        return default


def test_count() -> int | None:
    """Parse final test count."""

    path = FINAL_SUBMISSION_ROOT / "final_test_report.txt"
    if not path.exists():
        return None
    text = path.read_text()
    match = re.search(r"(\d+)\s+passed", text)
    return int(match.group(1)) if match else None


def no_stale_test_counts(expected: int | None) -> tuple[str, str]:
    """Check scanned docs for stale explicit test counts."""

    if expected is None:
        return "warning", "final test report missing"
    stale = []
    pattern = re.compile(r"(\d+)\s+tests\s+passed|tests\s+run:\s+`?(\d+)`?", re.IGNORECASE)
    for root in [ROOT / "README.md", ROOT / "docs", ROOT / "results" / "flexibility", ROOT / "results" / "paper_ready", FINAL_SUBMISSION_ROOT]:
        paths = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            for match in pattern.finditer(path.read_text(errors="ignore")):
                value = int(match.group(1) or match.group(2))
                if value != expected:
                    stale.append(f"{path.relative_to(ROOT)}:{value}")
                    break
    return ("blocker", ", ".join(stale[:12])) if stale else ("pass", f"all explicit test counts match {expected}")


def scan_unsupported_claims() -> tuple[str, str]:
    """Scan final text for unsupported overclaim patterns."""

    text = "\n".join(path.read_text(errors="ignore").lower() for path in (FINAL_SUBMISSION_ROOT / "paper_text_final").glob("*.md"))
    bad_patterns = [
        "reference independent",
        "pbc/nojump was proven",
        "theseus was rerun locally",
        "yiip is biologically validated",
    ]
    hits = [pattern for pattern in bad_patterns if pattern in text and f"do not claim {pattern}" not in text]
    return ("blocker", ", ".join(hits)) if hits else ("pass", "no unsupported final-text claim detected")


def build_rows() -> list[dict[str, str]]:
    """Build final submission-upgrade audit rows."""

    rows: list[dict[str, str]] = []
    release_meta = FINAL_SUBMISSION_ROOT / "release_metadata" / "release_metadata_audit.tsv"
    release_archive = FINAL_SUBMISSION_ROOT / "release_archive" / "blockers.json"
    testpypi = FINAL_SUBMISSION_ROOT / "testpypi_validation" / "testpypi_validation_report.json"
    user_testing = FINAL_SUBMISSION_ROOT / "user_testing" / "user_testing_summary.tsv"
    clean_install = FINAL_SUBMISSION_ROOT / "user_readiness" / "clean_install_report.json"
    repro = FINAL_SUBMISSION_ROOT / "reproducibility_capsule" / "reproduce_core_results_report.json"
    metadata_rows = read_table(release_meta)
    metadata_blockers = [row for row in metadata_rows if row.get("status") == "blocker"]
    metadata_warnings = [row for row in metadata_rows if row.get("status") == "warning"]
    metadata_status = "blocker" if metadata_blockers else ("warning" if metadata_warnings else "pass")
    add(rows, "P0", "release metadata audit", metadata_status, str(release_meta.relative_to(ROOT)), f"pending/warnings={[row.get('field') for row in metadata_warnings]}")
    archive_blockers = json_value(release_archive, "blockers", [])
    archive_warnings = json_value(release_archive, "warnings", [])
    archive_status = "blocker" if archive_blockers else ("warning" if archive_warnings else "pass")
    add(rows, "P0", "release archive build/twine", archive_status, str(release_archive.relative_to(ROOT)), f"warnings={archive_warnings}")
    testpypi_status = json_value(testpypi, "status", "missing")
    add(rows, "P0", "TestPyPI validation", "warning" if testpypi_status == "not_applicable_pending_upload" else ("pass" if testpypi_status == "pass" else "blocker"), str(testpypi.relative_to(ROOT)), f"status={testpypi_status}")
    clean_status = json_value(clean_install, "status", "missing")
    add(rows, "P0", "clean install", "pass" if clean_status == "pass" else "blocker", str(clean_install.relative_to(ROOT)), f"status={clean_status}")
    add(rows, "P0", "user testing protocol", "pass" if file_nonempty(ROOT / "docs" / "user_testing_protocol.md") and file_nonempty(user_testing) else "blocker", "docs/user_testing_protocol.md", "feedback may be pending")
    repro_status = json_value(repro, "status", "missing")
    add(rows, "P0", "reproducibility capsule", "pass" if repro_status == "pass" else "warning", str(repro.relative_to(ROOT)), f"status={repro_status}")

    p1_files = [
        ("residue-level reference consensus", FINAL_SUBMISSION_ROOT / "reference_consensus_residue_level" / "reference_consensus_system_summary.tsv"),
        ("atom-selection sensitivity", FINAL_SUBMISSION_ROOT / "atom_selection_sensitivity" / "atom_selection_sensitivity.tsv"),
        ("external MD mini-validation", FINAL_SUBMISSION_ROOT / "external_md_mini_validation" / "external_md_mini_validation.tsv"),
        ("final figure data", FINAL_SUBMISSION_ROOT / "final_figures" / "table_user_recommendations.tsv"),
    ]
    for check, path in p1_files:
        allow_empty = check == "external MD mini-validation"
        status = "pass" if file_nonempty(path) or (allow_empty and path.exists()) else "warning"
        add(rows, "P1", check, status, str(path.relative_to(ROOT)), "not_applicable is acceptable for optional analyses")

    p2_files = [
        ("reviewer pre-mortem", FINAL_SUBMISSION_ROOT / "reviewer_premortem" / "reviewer_premortem_template.tsv"),
        ("paper text snippets", FINAL_SUBMISSION_ROOT / "paper_text_final" / "limitations_final_submission.md"),
    ]
    for check, path in p2_files:
        add(rows, "P2", check, "pass" if file_nonempty(path) else "warning", str(path.relative_to(ROOT)), "pending feedback is acceptable")

    for rel in ["README.md", "LICENSE", "CITATION.cff"]:
        add(rows, "P0", rel, "pass" if file_nonempty(ROOT / rel) else "blocker", rel, "required release file")
    doc_paths = [ROOT / "README.md", ROOT / "docs" / "release.md", ROOT / "docs" / "installation.md", ROOT / "docs" / "quickstart.md", ROOT / "CITATION.cff", ROOT / "pyproject.toml"]
    bad_paths = [str(path.relative_to(ROOT)) for path in doc_paths if has_local_absolute_path(path.read_text())]
    add(rows, "P0", "no local absolute paths", "blocker" if bad_paths else "pass", ",".join(bad_paths), "public docs/metadata scan")
    stale_status, stale_note = no_stale_test_counts(test_count())
    add(rows, "P0", "no stale test counts", stale_status, "README/docs/results", stale_note)
    claim_status, claim_note = scan_unsupported_claims()
    add(rows, "P0", "no unsupported claims", claim_status, "paper_text_final", claim_note)
    return rows


def recommendation(rows: list[dict[str, str]]) -> str:
    """Return final recommendation."""

    blockers = [row for row in rows if row["status"] == "blocker"]
    warnings = [row for row in rows if row["status"] == "warning"]
    if blockers:
        return "D. Not ready; blockers remain"
    warning_names = {row["check"] for row in warnings}
    if warning_names <= {
        "release metadata audit",
        "release archive build/twine",
        "TestPyPI validation",
        "reproducibility capsule",
        "residue-level reference consensus",
        "atom-selection sensitivity",
        "external MD mini-validation",
    }:
        return "C. Almost ready; only release metadata or DOI pending"
    if warnings:
        return "B. Ready for BMC Bioinformatics submission after author review"
    return "A. Ready for Bioinformatics submission after author review"


def main() -> int:
    """Run final submission-upgrade audit."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    blockers = [row for row in rows if row["status"] == "blocker"]
    warnings = [row for row in rows if row["status"] == "warning"]
    rec = recommendation(rows)
    table = args.output_dir / "final_submission_upgrade_checklist.tsv"
    summary = args.output_dir / "final_submission_upgrade_summary.md"
    blockers_json = args.output_dir / "blockers.json"
    write_table(table, rows, FIELDS)
    write_json(
        blockers_json,
        {
            "generated_at": timestamp(),
            "blockers": [row["check"] for row in blockers],
            "warnings": [row["check"] for row in warnings],
            "recommendation": rec,
        },
    )
    summary.write_text(
        "\n".join(
            [
                "# Final Submission Upgrade Audit",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Recommendation: `{rec}`",
                f"- Blockers: `{[row['check'] for row in blockers]}`",
                f"- Warnings: `{[row['check'] for row in warnings]}`",
                "",
                "## Main vs Supplementary",
                "",
                "- Main text: workflow, formal MD benchmark summary, flexibility interpretation, preprocessing/reference caveats, user-ready package status.",
                "- Supplementary: release archive readiness, TestPyPI status, user testing protocol, reproducibility capsule, residue-consensus status, atom-selection status, external MD status, reviewer pre-mortem.",
                "",
            ]
        )
    )
    write_manifest(args.output_dir, script="scripts/audit_final_submission_upgrade.py", outputs=[table, summary, blockers_json], extra={"recommendation": rec})
    print(f"Wrote final submission upgrade audit to {args.output_dir}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
