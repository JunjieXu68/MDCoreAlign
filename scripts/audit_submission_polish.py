"""Audit submission-polish outputs for reviewer-risk reduction."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import read_table, row_count, to_float, write_json, write_table  # noqa: E402
from md_trajectory_focus_common import timestamp  # noqa: E402


OUT_DIR = ROOT / "results" / "submission_polish"
CHECKLIST = OUT_DIR / "final_submission_polish_checklist.tsv"
SUMMARY = OUT_DIR / "final_submission_polish_summary.md"
BLOCKERS = OUT_DIR / "blockers.json"

FIELDS = ["check_id", "requirement", "status", "evidence", "notes"]

REQUIRED_FILES = [
    OUT_DIR / "preprocessing_verification" / "pbc_centering_verification.tsv",
    OUT_DIR / "preprocessing_verification" / "pbc_centering_verification_summary.md",
    OUT_DIR / "preprocessing_verification" / "manifest.json",
    OUT_DIR / "temporal_convergence" / "temporal_convergence_classification.tsv",
    OUT_DIR / "temporal_convergence" / "temporal_convergence_summary.md",
    OUT_DIR / "main_figures" / "fig1_workflow_notes.md",
    OUT_DIR / "main_figures" / "fig2_benchmark_summary.tsv",
    OUT_DIR / "main_figures" / "fig3_md_diagnostics_summary.tsv",
    OUT_DIR / "main_figures" / "fig4_flexibility_profile_summary.tsv",
    OUT_DIR / "main_figures" / "main_figure_captions_draft.md",
    OUT_DIR / "paper_text" / "methods_md_preprocessing_final.md",
    OUT_DIR / "paper_text" / "results_md_trajectory_diagnostics_final.md",
    OUT_DIR / "paper_text" / "results_reference_sensitivity_final.md",
    OUT_DIR / "paper_text" / "results_warm_start_final.md",
    OUT_DIR / "paper_text" / "limitations_final.md",
    OUT_DIR / "paper_text" / "supplementary_allocation_final.md",
]

TEXT_COUNT_PATTERNS = [
    re.compile(r"(\d+)\s+tests\s+passed", re.IGNORECASE),
    re.compile(r"Ran\s+(\d+)\s+tests", re.IGNORECASE),
    re.compile(r"Tests\s+passed:\s+(\d+)", re.IGNORECASE),
    re.compile(r"Total\s+tests\s+run:\s+(\d+)", re.IGNORECASE),
]


def _read_text(path: Path) -> str:
    """Read text or return an empty string for missing files."""

    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _status(condition: bool, *, warning: bool = False) -> str:
    """Return pass/warning/blocker status."""

    if condition:
        return "pass"
    return "warning" if warning else "blocker"


def _add(rows: list[dict[str, object]], check_id: str, requirement: str, status: str, evidence: str, notes: str = "") -> None:
    """Append one checklist row."""

    rows.append(
        {
            "check_id": check_id,
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "notes": notes,
        }
    )


def _detect_current_test_count() -> int | None:
    """Detect the current documented test count from the paper-ready report."""

    path = ROOT / "results" / "paper_ready" / "test_report.txt"
    text = _read_text(path)
    matches: list[int] = []
    for pattern in TEXT_COUNT_PATTERNS:
        matches.extend(int(match.group(1)) for match in pattern.finditer(text))
    return max(matches) if matches else None


def _stale_test_count_hits(current_count: int | None) -> list[str]:
    """Find documented test-count mentions that disagree with the current count."""

    search_roots = [ROOT / "README.md", ROOT / "docs", ROOT / "results" / "flexibility", ROOT / "results" / "paper_ready"]
    hits: list[str] = []
    for root in search_roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in paths:
            if path.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            text = _read_text(path)
            for pattern in TEXT_COUNT_PATTERNS:
                for match in pattern.finditer(text):
                    count = int(match.group(1))
                    if current_count is not None and count != current_count:
                        hits.append(f"{path.relative_to(ROOT)}:{match.group(0)}")
    return hits


def build_checklist() -> list[dict[str, object]]:
    """Build all submission-polish audit checks."""

    rows: list[dict[str, object]] = []
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.exists()]
    _add(
        rows,
        "required_files",
        "all submission-polish output files exist",
        _status(not missing),
        "all required files found" if not missing else "; ".join(missing),
    )
    nonempty_tables = []
    empty_tables = []
    for path in REQUIRED_FILES:
        if path.suffix.lower() == ".tsv" and path.exists():
            count = row_count(path)
            if count and count > 0:
                nonempty_tables.append(f"{path.relative_to(ROOT)}:{count}")
            else:
                empty_tables.append(str(path.relative_to(ROOT)))
    _add(
        rows,
        "nonempty_tables",
        "all required TSV tables are non-empty",
        _status(not empty_tables),
        "; ".join(nonempty_tables) if not empty_tables else "; ".join(empty_tables),
    )

    prep_summary = _read_text(OUT_DIR / "preprocessing_verification" / "pbc_centering_verification_summary.md")
    pbc_phrase = "PBC/nojump and centering status could not be independently verified"
    _add(
        rows,
        "pbc_status",
        "PBC/preprocessing status is explicitly described",
        _status(pbc_phrase in prep_summary),
        pbc_phrase if pbc_phrase in prep_summary else "phrase missing",
        "Do not invent upstream preprocessing provenance.",
    )

    temporal_path = OUT_DIR / "temporal_convergence" / "temporal_convergence_classification.tsv"
    temporal_rows = read_table(temporal_path) if temporal_path.exists() else []
    true_failures = [
        row
        for row in temporal_rows
        if row.get("classification") in {"true_failure", "weight_collapse"}
    ]
    temporal_summary = _read_text(OUT_DIR / "temporal_convergence" / "temporal_convergence_summary.md")
    temporal_limited = "per-block diagnostic traces were not generated" in temporal_summary.lower()
    _add(
        rows,
        "temporal_nonconvergence",
        "temporal non-convergence is classified or explicitly limited",
        _status(not true_failures and bool(temporal_rows)),
        f"rows={len(temporal_rows)}; true_failure_or_weight_collapse={len(true_failures)}",
        "Summary explicitly limits oscillation calls from summary-only diagnostics." if temporal_limited else "Trace limitation phrase not detected.",
    )

    fig3_path = OUT_DIR / "main_figures" / "fig3_md_diagnostics_summary.tsv"
    fig3_rows = read_table(fig3_path) if fig3_path.exists() else []
    reference_rows = [row for row in fig3_rows if row.get("diagnostic") == "reference_sensitivity"]
    reference_text = _read_text(OUT_DIR / "paper_text" / "results_reference_sensitivity_final.md")
    reference_ok = bool(reference_rows) and "not be described as reference independent" in reference_text.lower()
    _add(
        rows,
        "reference_sensitivity",
        "reference sensitivity is not hidden",
        _status(reference_ok),
        f"reference rows={len(reference_rows)}",
        "No reference-independence claim is allowed.",
    )

    warm_text = _read_text(OUT_DIR / "paper_text" / "results_warm_start_final.md")
    warm_ok = "efficiency option" in warm_text.lower() and "not be described as scientifically necessary" in warm_text.lower()
    _add(
        rows,
        "warm_start",
        "warm start is described as an efficiency option",
        _status(warm_ok),
        "efficiency wording present" if warm_ok else "required wording missing",
    )

    limitations = _read_text(OUT_DIR / "paper_text" / "limitations_final.md")
    theseus_ok = "do not imply theseus was rerun locally" in limitations.lower()
    _add(
        rows,
        "theseus_caveat",
        "THESEUS rerun caveat is present",
        _status(theseus_ok),
        "THESEUS caveat present" if theseus_ok else "THESEUS caveat missing",
    )

    yiip_ok = "limited external biological-proxy support" in limitations.lower()
    _add(
        rows,
        "yiip_caveat",
        "YiiP biology caveat is present",
        _status(yiip_ok),
        "YiiP caveat present" if yiip_ok else "YiiP caveat missing",
    )

    rmsf_ok = "rmsf is alignment-frame dependent" in limitations.lower()
    _add(
        rows,
        "rmsf_caveat",
        "RMSF alignment-frame dependence is explicit",
        _status(rmsf_ok),
        "RMSF caveat present" if rmsf_ok else "RMSF caveat missing",
    )

    nmr_ok = "nmr500 should be treated as supplementary structural-ensemble evidence" in limitations.lower()
    _add(
        rows,
        "nmr500_caveat",
        "NMR500 is supplementary structural-ensemble evidence for MD-focused manuscript",
        _status(nmr_ok),
        "NMR500 caveat present" if nmr_ok else "NMR500 caveat missing",
    )

    supplement = _read_text(OUT_DIR / "paper_text" / "supplementary_allocation_final.md")
    main_figures_exist = all(
        (OUT_DIR / "main_figures" / name).exists()
        for name in (
            "fig1_workflow_notes.md",
            "fig2_benchmark_summary.tsv",
            "fig3_md_diagnostics_summary.tsv",
            "fig4_flexibility_profile_summary.tsv",
            "main_figure_captions_draft.md",
        )
    )
    supplement_ok = "nmr500 structural-ensemble benchmark" in supplement.lower()
    _add(
        rows,
        "figures_and_supplement",
        "main figures and supplement allocation exist",
        _status(main_figures_exist and supplement_ok),
        f"main_figure_files_exist={main_figures_exist}; supplement_allocation={supplement_ok}",
    )

    current_count = _detect_current_test_count()
    stale_hits = _stale_test_count_hits(current_count)
    _add(
        rows,
        "stale_test_counts",
        "no stale test counts remain",
        _status(not stale_hits),
        f"current_count={current_count}; stale_hits={len(stale_hits)}",
        "; ".join(stale_hits[:8]),
    )
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    """Write summary markdown and blockers JSON."""

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    blockers = [row for row in rows if row["status"] == "blocker"]
    warnings = [row for row in rows if row["status"] == "warning"]
    recommendation = (
        "Ready for Bioinformatics/BMC Bioinformatics author review after final human read-through."
        if not blockers
        else "Not ready for author review until blockers are resolved."
    )
    SUMMARY.write_text(
        "\n".join(
            [
                "# Final Submission-Polish Summary",
                "",
                f"- Generated at: `{timestamp()}`",
                f"- Status counts: `{status_counts}`",
                f"- Recommendation: `{recommendation}`",
                "",
                "## Blockers",
                "",
                *(f"- {row['check_id']}: {row['evidence']}" for row in blockers),
                *([] if blockers else ["- None."]),
                "",
                "## Warnings",
                "",
                *(f"- {row['check_id']}: {row['notes'] or row['evidence']}" for row in warnings),
                *([] if warnings else ["- None."]),
                "",
            ]
        )
    )
    write_json(
        BLOCKERS,
        {
            "generated_at": timestamp(),
            "blockers": [
                {
                    "check_id": row["check_id"],
                    "requirement": row["requirement"],
                    "evidence": row["evidence"],
                    "notes": row["notes"],
                }
                for row in blockers
            ],
            "warnings": [
                {
                    "check_id": row["check_id"],
                    "requirement": row["requirement"],
                    "evidence": row["evidence"],
                    "notes": row["notes"],
                }
                for row in warnings
            ],
            "recommendation": recommendation,
        },
    )


def main() -> None:
    """Run the submission-polish audit."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_checklist()
    write_table(CHECKLIST, rows, FIELDS)
    write_summary(rows)
    print(f"Wrote submission-polish audit to {OUT_DIR}")


if __name__ == "__main__":
    main()
