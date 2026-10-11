"""Write final submission-upgrade manuscript text snippets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_submission_common import FINAL_SUBMISSION_ROOT, read_table, timestamp, write_manifest  # noqa: E402


OUTPUT_DIR = FINAL_SUBMISSION_ROOT / "paper_text_final"
FILES = [
    "methods_software_availability_final.md",
    "methods_reproducibility_capsule_final.md",
    "methods_user_readiness_final.md",
    "results_user_readiness_final.md",
    "results_reference_consensus_final.md",
    "results_atom_selection_sensitivity_final.md",
    "results_external_md_validation_final.md",
    "limitations_final_submission.md",
    "availability_statement_final.md",
    "data_availability_statement_final.md",
    "ai_disclosure_draft.md",
    "supplementary_allocation_final.md",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def status_from_json(path: Path, key: str = "status") -> str:
    """Read status from a JSON file."""

    if not path.exists():
        return "missing"
    try:
        return str(json.loads(path.read_text()).get(key, "missing"))
    except Exception:
        return "unavailable"


def build_texts() -> dict[str, str]:
    """Build final text snippets."""

    generated = timestamp()
    release_status = read_table(FINAL_SUBMISSION_ROOT / "release_metadata" / "release_metadata_audit.tsv")
    release_values = {row["field"]: row["value"] for row in release_status}
    testpypi_status = status_from_json(FINAL_SUBMISSION_ROOT / "testpypi_validation" / "testpypi_validation_report.json")
    clean_status = status_from_json(FINAL_SUBMISSION_ROOT / "release_archive" / "blockers.json", "blockers")
    repo = release_values.get("project.urls", "pending")
    doi = release_values.get("doi", "pending")
    return {
        "methods_software_availability_final.md": "\n".join(
            [
                "# Methods: Software Availability",
                "",
                f"Generated: `{generated}`.",
                "",
                "MDCoreAlign is distributed as a Python package with the import namespace `mdcorealign`, module entry point `python -m mdcorealign`, and console script `mdcorealign`. The user-facing CLI includes `align`, `flexibility`, `diagnostics`, and `audit-preprocessing` commands.",
                "",
                f"Repository URL status: `{repo}`. Zenodo DOI status: `{doi}`. TestPyPI/PyPI status: `{testpypi_status}`.",
                "",
                "Exact release metadata are audited in `results/final_submission_upgrade/release_metadata/release_metadata_audit.tsv`.",
            ]
        ),
        "methods_reproducibility_capsule_final.md": "\n".join(
            [
                "# Methods: Reproducibility Capsule",
                "",
                f"Generated: `{generated}`.",
                "",
                "The lightweight reproducibility capsule reruns the toy quickstart, package import, CLI help, and release-readiness tests. It checksum-validates cached formal result tables. It does not rerun full MD trajectory benchmarks, NMR500, MDLovoFit, or THESEUS.",
                "",
                "Primary file: `results/final_submission_upgrade/reproducibility_capsule/reproduce_core_results_report.md`.",
            ]
        ),
        "methods_user_readiness_final.md": "\n".join(
            [
                "# Methods: User Readiness",
                "",
                f"Generated: `{generated}`.",
                "",
                "User readiness was assessed by clean-install checks, CLI help checks, a bundled toy quickstart, package build validation, TestPyPI status validation, and a prepared external user-testing protocol.",
                "",
                "External MDLovoFit and THESEUS executables are optional baselines and are not required for core MDCoreAlign commands.",
            ]
        ),
        "results_user_readiness_final.md": "\n".join(
            [
                "# Results: User Readiness",
                "",
                f"Generated: `{generated}`.",
                "",
                f"Release archive blockers status: `{clean_status}`. TestPyPI validation status: `{testpypi_status}`. External user-testing feedback remains pending unless feedback files are added.",
                "",
                "Primary files: `results/final_submission_upgrade/release_archive/`, `results/final_submission_upgrade/testpypi_validation/`, and `results/final_submission_upgrade/user_testing/`.",
            ]
        ),
        "results_reference_consensus_final.md": "\n".join(
            [
                "# Results: Reference Consensus",
                "",
                f"Generated: `{generated}`.",
                "",
                "System-level reference sensitivity is summarized, but residue-level reference consensus is unavailable in this release-candidate pass because per-reference residue-level profiles were not present. No reference independence is claimed.",
                "",
                "Primary files: `results/final_submission_upgrade/reference_consensus_residue_level/reference_consensus_system_summary.tsv` and `results/final_quality_upgrade/reference_consensus/reference_consensus_system_summary.tsv`.",
            ]
        ),
        "results_atom_selection_sensitivity_final.md": "\n".join(
            [
                "# Results: Atom-Selection Sensitivity",
                "",
                f"Generated: `{generated}`.",
                "",
                "CA-level selection remains the formal default. Backbone sensitivity is marked unavailable/not_applicable unless a validated residue-level backbone-to-CA aggregation protocol is run. No array-index-only residue mapping was used.",
                "",
                "Primary file: `results/final_submission_upgrade/atom_selection_sensitivity/atom_selection_sensitivity.tsv`.",
            ]
        ),
        "results_external_md_validation_final.md": "\n".join(
            [
                "# Results: External MD Mini-Validation",
                "",
                f"Generated: `{generated}`.",
                "",
                "No uncontrolled external trajectory downloads or web scraping were performed. Missing config is not_applicable and should remain supplementary unless the author explicitly provides and promotes controlled external MD cases.",
                "",
                "Primary file: `results/final_submission_upgrade/external_md_mini_validation/external_md_mini_validation.tsv`.",
            ]
        ),
        "limitations_final_submission.md": "\n".join(
            [
                "# Limitations: Final Submission",
                "",
                f"Generated: `{generated}`.",
                "",
                "- Do not claim THESEUS was rerun locally if only existing tables were validated.",
                "- Do not claim PBC/nojump was proven if only lightweight checks were done.",
                "- Do not claim reference independence.",
                "- Do not overclaim YiiP biological validation.",
                "- RMSF is alignment-frame dependent.",
                "- CA-level selection is the formal default unless atom-selection analysis supports broader claims.",
                "- GitHub URL, Zenodo DOI, and TestPyPI/PyPI status must be reported exactly as audited.",
            ]
        ),
        "availability_statement_final.md": "\n".join(
            [
                "# Availability Statement",
                "",
                f"Generated: `{generated}`.",
                "",
                f"Software repository: `{repo}`. Release tag: `v0.1.0rc1`. Zenodo DOI: `{doi}`. If any value is pending, replace it only after the real external record exists.",
            ]
        ),
        "data_availability_statement_final.md": "\n".join(
            [
                "# Data Availability Statement",
                "",
                f"Generated: `{generated}`.",
                "",
                "Small toy data required for quickstart are included in the repository. Large MD trajectories and external benchmark data are not required for the lightweight reproducibility capsule and should not be committed to GitHub. Cached official result tables are checksum-validated where full reruns are not performed.",
            ]
        ),
        "ai_disclosure_draft.md": "\n".join(
            [
                "# AI Disclosure Draft",
                "",
                f"Generated: `{generated}`.",
                "",
                "Draft wording for journals that request AI disclosure: AI-assisted coding and editorial tooling was used to help organize software, tests, documentation, and audit summaries. The authors remain responsible for all scientific claims, code, analyses, and manuscript text.",
            ]
        ),
        "supplementary_allocation_final.md": "\n".join(
            [
                "# Supplementary Allocation",
                "",
                f"Generated: `{generated}`.",
                "",
                "Main text candidates: MDCoreAlign workflow decision tree, formal MD benchmark summary, flexibility interpretation summary, preprocessing limitation, reference-sensitivity caveat, and user-facing installation/quickstart status.",
                "",
                "Supplementary candidates: release metadata audit, release archive readiness, TestPyPI pending/validation report, user-testing protocol, reproducibility capsule checksums, residue-level reference-consensus availability, CA-vs-backbone status, external MD mini-validation status, reviewer pre-mortem template, THESEUS validation caveat, and NMR500 structural-ensemble evidence.",
            ]
        ),
    }


def main() -> int:
    """Write final paper text snippets."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts = build_texts()
    outputs = []
    for name in FILES:
        path = args.output_dir / name
        path.write_text(texts[name] + "\n")
        outputs.append(path)
    write_manifest(
        args.output_dir,
        script="scripts/write_final_submission_paper_text.py",
        inputs=[
            FINAL_SUBMISSION_ROOT / "release_metadata" / "release_metadata_audit.tsv",
            FINAL_SUBMISSION_ROOT / "release_archive" / "blockers.json",
            FINAL_SUBMISSION_ROOT / "testpypi_validation" / "testpypi_validation_report.json",
        ],
        outputs=outputs,
        extra={"text_files": FILES},
    )
    print(f"Wrote final submission paper text snippets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
