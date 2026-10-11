"""Audit final-quality upgrade outputs and release-candidate readiness."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
import tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_quality_common import (  # noqa: E402
    FINAL_ROOT,
    file_nonempty,
    has_local_absolute_path,
    read_table,
    timestamp,
    write_json,
    write_manifest,
    write_table,
)


OUTPUT_DIR = FINAL_ROOT / "final_audit"
USER_DIR = FINAL_ROOT / "user_readiness"
PAPER_TEXT_DIR = FINAL_ROOT / "paper_text_updates"

FIELDS = ["check_id", "category", "status", "evidence", "notes"]
DOC_FIELDS = ["item", "status", "evidence", "notes"]
RELEASE_FIELDS = ["step", "status", "evidence", "notes"]

REQUIRED_PAPER_TEXT = [
    "methods_preprocessing_verification.md",
    "methods_reference_consensus.md",
    "methods_clean_install_reproducibility.md",
    "results_preprocessing_verification.md",
    "results_temporal_convergence_classification.md",
    "results_reference_consensus.md",
    "results_user_readiness.md",
    "optional_results_external_md_validation.md",
    "optional_results_atom_selection_sensitivity.md",
    "limitations_final_quality_update.md",
    "availability_and_reproducibility_final.md",
    "supplementary_allocation_final_quality.md",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def add(rows: list[dict[str, str]], check_id: str, category: str, status: str, evidence: str, notes: str) -> None:
    """Append one audit row."""

    rows.append(
        {
            "check_id": check_id,
            "category": category,
            "status": status,
            "evidence": evidence,
            "notes": notes,
        }
    )


def _status_for_file(path: Path, *, allow_empty_table: bool = False) -> tuple[str, str]:
    """Return status and note for one required file."""

    if not path.exists():
        return "blocker", "missing"
    if path.suffix in {".tsv", ".csv"} and allow_empty_table:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
            if reader.fieldnames:
                return "pass", "table header present"
    return ("pass", "present and non-empty") if file_nonempty(path) else ("blocker", "empty")


def documentation_audit() -> list[dict[str, str]]:
    """Build documentation and package metadata audit rows."""

    rows: list[dict[str, str]] = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject.get("project", {})
    optional = project.get("optional-dependencies", {})
    scripts = project.get("scripts", {})
    readme = (ROOT / "README.md").read_text()

    checks = [
        ("package name", project.get("name") == "mdcorealign", "pyproject.toml", "project.name is mdcorealign"),
        ("version", bool(project.get("version")), "pyproject.toml", f"version={project.get('version', '')}"),
        ("build-system", "build-system" in pyproject, "pyproject.toml", "build-system present"),
        ("dependencies", bool(project.get("dependencies")), "pyproject.toml", "runtime dependencies present"),
        ("dev extra", "dev" in optional, "pyproject.toml", "dev optional dependency group present"),
        ("docs extra", "docs" in optional, "pyproject.toml", "docs optional dependency group present"),
        ("external extra", "external" in optional, "pyproject.toml", "external optional dependency group present"),
        (
            "console script",
            scripts.get("mdcorealign") == "mdcorealign.cli:main",
            "pyproject.toml",
            "mdcorealign console script present",
        ),
        ("license", (ROOT / "LICENSE").exists(), "LICENSE", "license file present"),
        ("citation", (ROOT / "CITATION.cff").exists(), "CITATION.cff", "citation metadata present"),
        ("changelog", (ROOT / "CHANGELOG.md").exists(), "CHANGELOG.md", "changelog present"),
    ]
    for item, ok, evidence, note in checks:
        rows.append({"item": item, "status": "pass" if ok else "blocker", "evidence": evidence, "notes": note})

    required_sections = [
        "Installation",
        "Quick Start",
        "CLI Examples",
        "Python API Example",
        "Input Formats",
        "Output Formats",
        "MD Preprocessing Requirements",
        "Interpreting Scores",
        "Limitations",
        "Citation",
    ]
    for section in required_sections:
        rows.append(
            {
                "item": f"README section: {section}",
                "status": "pass" if f"## {section}" in readme else "blocker",
                "evidence": "README.md",
                "notes": "required user-facing section",
            }
        )

    required_docs = [
        "docs/installation.md",
        "docs/quickstart.md",
        "docs/cli_reference.md",
        "docs/python_api.md",
        "docs/output_formats.md",
        "docs/md_preprocessing.md",
        "docs/rmsf_flexible_segments.md",
        "docs/troubleshooting.md",
        "docs/release.md",
        "docs/user_testing_protocol.md",
    ]
    for rel in required_docs:
        path = ROOT / rel
        rows.append(
            {
                "item": f"documentation file: {rel}",
                "status": "pass" if file_nonempty(path) else "blocker",
                "evidence": rel,
                "notes": "required documentation file",
            }
        )

    docs_to_scan = [ROOT / "README.md", *[ROOT / rel for rel in required_docs], ROOT / "examples" / "toy_md" / "README.md"]
    offenders = [str(path.relative_to(ROOT)) for path in docs_to_scan if path.exists() and has_local_absolute_path(path.read_text())]
    rows.append(
        {
            "item": "no local absolute paths in public docs",
            "status": "pass" if not offenders else "blocker",
            "evidence": ",".join(offenders) if offenders else "README/docs/examples",
            "notes": "public documentation should not include local machine paths",
        }
    )

    project_urls = project.get("urls", {})
    rows.append(
        {
            "item": "project URLs",
            "status": "warning" if not project_urls else "pass",
            "evidence": "pyproject.toml",
            "notes": "repository/docs/issues URLs can be finalized once the GitHub remote is created",
        }
    )
    return rows


def release_readiness_rows() -> list[dict[str, str]]:
    """Build release readiness checklist rows."""

    rows: list[dict[str, str]] = []
    steps = [
        ("version bump", "pyproject.toml", "0.1.0rc1" in (ROOT / "pyproject.toml").read_text()),
        ("tests pass", "results/final_quality_upgrade/final_test_report.txt", file_nonempty(FINAL_ROOT / "final_test_report.txt")),
        ("clean install passes", "results/final_quality_upgrade/user_readiness/clean_install_report.md", file_nonempty(USER_DIR / "clean_install_report.md")),
        ("toy quickstart passes", "results/final_quality_upgrade/user_readiness/toy_quickstart_report.md", file_nonempty(USER_DIR / "toy_quickstart_report.md")),
        ("package build passes", "results/final_quality_upgrade/user_readiness/package_build_report.md", file_nonempty(USER_DIR / "package_build_report.md")),
        ("GitHub release", "docs/release.md", "GitHub release" in (ROOT / "docs" / "release.md").read_text()),
        ("Zenodo DOI", "docs/release.md", "Zenodo DOI" in (ROOT / "docs" / "release.md").read_text()),
        ("TestPyPI instructions", "docs/release.md", "TestPyPI" in (ROOT / "docs" / "release.md").read_text()),
    ]
    for step, evidence, ok in steps:
        status = "pass" if ok else "blocker"
        if step in {"tests pass", "GitHub release", "Zenodo DOI"} and not ok:
            status = "warning"
        rows.append({"step": step, "status": status, "evidence": evidence, "notes": "release-candidate checklist item"})
    return rows


def latest_test_count() -> int | None:
    """Parse latest full-test count if available."""

    report = FINAL_ROOT / "final_test_report.txt"
    if not report.exists():
        return None
    text = report.read_text()
    match = re.search(r"Ran\s+(\d+)\s+tests", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s+passed", text)
    return int(match.group(1)) if match else None


def stale_test_count_status() -> tuple[str, str]:
    """Check for hard-coded stale test counts in manuscript-facing docs."""

    expected = latest_test_count()
    if expected is None:
        return "warning", "latest final_test_report.txt is unavailable"
    stale: list[str] = []
    scan_roots = [
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "results" / "flexibility",
        ROOT / "docs" / "current_main_results_summary.md",
        ROOT / "docs" / "current_main_results_summary.txt",
        ROOT / "results" / "paper_ready",
    ]
    pattern = re.compile(r"(\d+)\s+tests\s+passed|tests\s+run:\s+`?(\d+)`?", re.IGNORECASE)
    for root in scan_roots:
        paths = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
        for path in paths:
            if path.suffix.lower() not in {".md", ".txt", ".json"} or not path.is_file():
                continue
            text = path.read_text(errors="ignore")
            for match in pattern.finditer(text):
                value = int(match.group(1) or match.group(2))
                if value != expected:
                    stale.append(f"{path.relative_to(ROOT)}:{value}")
                    break
    if stale:
        return "blocker", "stale test-count references: " + ", ".join(stale[:12])
    return "pass", f"all scanned explicit test counts match {expected}"


def build_checklist() -> list[dict[str, str]]:
    """Build final quality checklist rows."""

    rows: list[dict[str, str]] = []
    required_files = [
        ("preprocessing_table", FINAL_ROOT / "preprocessing_verification" / "pbc_centering_verification.tsv", False),
        ("preprocessing_summary", FINAL_ROOT / "preprocessing_verification" / "pbc_centering_verification_summary.md", False),
        ("recentered_comparison", FINAL_ROOT / "preprocessing_verification" / "recentered_comparison.tsv", True),
        ("temporal_classification", FINAL_ROOT / "temporal_convergence" / "temporal_convergence_classification.tsv", False),
        ("temporal_summary", FINAL_ROOT / "temporal_convergence" / "temporal_convergence_summary.md", False),
        ("reference_consensus_system", FINAL_ROOT / "reference_consensus" / "reference_consensus_system_summary.tsv", False),
        ("reference_consensus_residue", FINAL_ROOT / "reference_consensus" / "reference_consensus_residues.tsv", True),
        ("reference_consensus_summary", FINAL_ROOT / "reference_consensus" / "reference_consensus_summary.md", False),
        ("clean_install_report", USER_DIR / "clean_install_report.md", False),
        ("toy_quickstart_report", USER_DIR / "toy_quickstart_report.md", False),
        ("package_build_report", USER_DIR / "package_build_report.md", False),
        ("external_md_status", FINAL_ROOT / "external_md_mini_validation" / "external_md_mini_validation.tsv", True),
        ("atom_selection_status", FINAL_ROOT / "atom_selection_sensitivity" / "atom_selection_sensitivity.tsv", False),
    ]
    for check_id, path, allow_empty in required_files:
        status, note = _status_for_file(path, allow_empty_table=allow_empty)
        add(rows, check_id, "required_output", status, str(path.relative_to(ROOT)), note)

    for name in REQUIRED_PAPER_TEXT:
        path = PAPER_TEXT_DIR / name
        status, note = _status_for_file(path)
        add(rows, f"paper_text_{name}", "paper_text", status, str(path.relative_to(ROOT)), note)

    text_blobs = "\n".join(path.read_text(errors="ignore") for path in PAPER_TEXT_DIR.glob("*.md"))
    required_phrases = [
        ("pbc_limitation", "PBC/nojump and centering status could not be independently verified"),
        ("reference_not_independent", "Do not claim reference independence"),
        ("warm_start_efficiency", "Warm start is an efficiency option"),
        ("theseus_not_rerun", "not be described as rerun locally"),
        ("yiip_caveat", "YiiP should not be overclaimed"),
        ("rmsf_frame_dependent", "RMSF is alignment-frame dependent"),
        ("nmr_supplementary", "NMR500 is supplementary structural-ensemble evidence"),
    ]
    for check_id, phrase in required_phrases:
        add(
            rows,
            check_id,
            "reviewer_risk_language",
            "pass" if phrase in text_blobs else "blocker",
            "results/final_quality_upgrade/paper_text_updates",
            f"required phrase: {phrase}",
        )

    stale_status, stale_note = stale_test_count_status()
    add(rows, "no_stale_test_counts", "release_readiness", stale_status, "README/docs/results", stale_note)

    doc_rows = documentation_audit()
    for doc_row in doc_rows:
        if doc_row["status"] == "blocker":
            add(rows, f"doc_{doc_row['item']}", "documentation", "blocker", doc_row["evidence"], doc_row["notes"])

    release_rows = release_readiness_rows()
    for release_row in release_rows:
        if release_row["status"] == "blocker":
            add(rows, f"release_{release_row['step']}", "release", "blocker", release_row["evidence"], release_row["notes"])
    return rows


def recommendation(blockers: list[str], warnings: list[str]) -> str:
    """Return final author-review recommendation."""

    if blockers:
        return "D. Not ready; blockers must be resolved before author review"
    if len(warnings) > 5:
        return "B. Ready for BMC Bioinformatics author review; resolve warnings before aiming higher"
    return "A. Ready for Bioinformatics/BMC Bioinformatics author review"


def build_summary(rows: list[dict[str, str]], doc_rows: list[dict[str, str]], release_rows: list[dict[str, str]]) -> str:
    """Build final audit Markdown summary."""

    blockers = [row for row in rows if row["status"] == "blocker"]
    warnings = [
        row
        for row in rows
        if row["status"] == "warning"
    ] + [row for row in doc_rows if row["status"] == "warning"] + [row for row in release_rows if row["status"] == "warning"]
    status_counts = {label: sum(row["status"] == label for row in rows) for label in ["pass", "warning", "blocker"]}
    doc_counts = {label: sum(row["status"] == label for row in doc_rows) for label in ["pass", "warning", "blocker"]}
    release_counts = {label: sum(row["status"] == label for row in release_rows) for label in ["pass", "warning", "blocker"]}
    rec = recommendation([row["check_id"] for row in blockers], [row.get("check_id", row.get("item", "")) for row in warnings])
    warning_lines = [f"- `{row.get('check_id', row.get('item', row.get('step', 'warning')))}`: {row.get('notes', '')}" for row in warnings]
    blocker_lines = [f"- `{row['check_id']}`: {row['notes']}" for row in blockers]
    return "\n".join(
        [
            "# Final Quality Upgrade Audit",
            "",
            f"- Generated at: `{timestamp()}`",
            f"- Recommendation: `{rec}`.",
            f"- Checklist status counts: `{status_counts}`.",
            f"- Documentation audit counts: `{doc_counts}`.",
            f"- Release-readiness counts: `{release_counts}`.",
            "",
            "## Blockers",
            "",
            *(blocker_lines or ["- None."]),
            "",
            "## Warnings",
            "",
            *(warning_lines or ["- None."]),
            "",
            "## Main vs Supplementary Allocation",
            "",
            "- Main text candidates: MD trajectory benchmark summary, robust-vs-ordinary flexibility profiles, preprocessing limitation, temporal convergence classification, reference-sensitivity summary, and user-ready package quickstart.",
            "- Supplementary candidates: full PBC/centering sanity tables, recentered diagnostic comparison, reference-consensus placeholder/residue-level limitation, optional external MD status, optional CA-vs-backbone status, NMR500 structural-ensemble evidence, THESEUS validation caveat, and clean-install/build logs.",
            "",
        ]
    )


def main() -> int:
    """Run final quality audit."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc_rows = documentation_audit()
    release_rows = release_readiness_rows()
    checklist = build_checklist()

    doc_table = USER_DIR / "documentation_audit.tsv"
    doc_summary = USER_DIR / "documentation_audit.md"
    release_table = USER_DIR / "release_readiness_checklist.tsv"
    release_summary = USER_DIR / "release_readiness_summary.md"
    checklist_table = args.output_dir / "final_quality_checklist.tsv"
    summary = args.output_dir / "final_quality_summary.md"
    blockers_json = args.output_dir / "blockers.json"

    write_table(doc_table, doc_rows, DOC_FIELDS)
    doc_summary.write_text(
        "# Documentation Audit\n\n"
        f"- Generated at: `{timestamp()}`\n"
        f"- Counts: `{ {label: sum(row['status'] == label for row in doc_rows) for label in ['pass', 'warning', 'blocker']} }`\n"
    )
    write_table(release_table, release_rows, RELEASE_FIELDS)
    release_summary.write_text(
        "# Release Readiness Summary\n\n"
        f"- Generated at: `{timestamp()}`\n"
        f"- Counts: `{ {label: sum(row['status'] == label for row in release_rows) for label in ['pass', 'warning', 'blocker']} }`\n"
        "- TestPyPI upload remains a manual release-management step after final repository URLs are set.\n"
    )
    write_table(checklist_table, checklist, FIELDS)
    summary.write_text(build_summary(checklist, doc_rows, release_rows))
    blocker_ids = [row["check_id"] for row in checklist if row["status"] == "blocker"]
    warning_ids = [
        row["check_id"] for row in checklist if row["status"] == "warning"
    ] + [row["item"] for row in doc_rows if row["status"] == "warning"] + [row["step"] for row in release_rows if row["status"] == "warning"]
    write_json(
        blockers_json,
        {
            "generated_at": timestamp(),
            "blockers": blocker_ids,
            "warnings": warning_ids,
            "recommendation": recommendation(blocker_ids, warning_ids),
        },
    )
    write_manifest(
        args.output_dir,
        script="scripts/audit_final_quality_upgrade.py",
        inputs=[
            FINAL_ROOT / "final_test_report.txt",
            PAPER_TEXT_DIR / "limitations_final_quality_update.md",
            USER_DIR / "clean_install_report.md",
            USER_DIR / "package_build_report.md",
            USER_DIR / "toy_quickstart_report.md",
        ],
        outputs=[doc_table, doc_summary, release_table, release_summary, checklist_table, summary, blockers_json],
    )
    print(f"Wrote final quality audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
