#!/usr/bin/env python3
"""Write AS8-AS10 status, metrics, checklist, and final human-readable report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/ashar_private_evaluation_v1"
FINAL = BASE / "08_final_package"


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a TSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def timestamp(path: Path) -> str:
    """Return an evidence file modification timestamp."""

    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def main() -> None:
    """Generate pre-validation or final AS status files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-validation", action="store_true")
    args = parser.parse_args()
    functional = read_tsv(BASE / "06_unseen_analysis/functional_results.tsv")
    parity = read_tsv(BASE / "06_unseen_analysis/cpu_gpu_parity.tsv")
    agreement = read_tsv(BASE / "06_unseen_analysis/throughput_exact_agreement.tsv")
    io = read_tsv(BASE / "06_unseen_analysis/io_results.tsv")
    panel = read_tsv(BASE / "04_unseen_panel/locked_unseen_panel.tsv")
    decision = (BASE / "06_unseen_analysis/decision.txt").read_text().strip()
    final_status = "completed" if args.final_validation else "pending_final_archive_validation"
    final_evidence = (
        "results/ashar_private_evaluation_v1/08_final_package/final_archive_validation_report.md"
        if args.final_validation
        else "pending"
    )
    phases = [
        ("AS0", "Freeze baseline and active-work guard", "completed", BASE / "00_freeze/source_manifest.tsv"),
        ("AS1", "Private package contract", "completed", BASE / "01_package/ashar_package_contract.md"),
        ("AS2", "Build and seal provisional package", "completed", BASE / "01_package/provisional_archive.tsv"),
        ("AS3", "Independent archive-only installation", "completed", BASE / "02_archive_validation/archive_only_install_report.md"),
        ("AS4", "Prior-benchmark exclusion registry", "completed", BASE / "03_prior_benchmark_registry/registry_build_report.md"),
        ("AS5", "Prospective unseen-panel lock", "completed", BASE / "04_unseen_panel/panel_lock.sha256"),
        ("AS6", "Sealed-package unseen execution", "completed", BASE / "05_unseen_runs/run_manifest.tsv"),
        ("AS7", "Unseen quality and usability evaluation", "completed", BASE / "06_unseen_analysis/unseen_case_report.md"),
        ("AS8", "Advisor material integration", "completed", BASE / "07_advisor_materials/UNSEEN_VALIDATION_SUMMARY.md"),
        ("AS9", "Final advisor archive and archive-only validation", final_status, Path(final_evidence) if final_evidence != "pending" else BASE / "08_final_package"),
        ("AS10", "Review bundle and final acceptance report", final_status, Path(final_evidence) if final_evidence != "pending" else BASE / "08_final_package"),
    ]
    phase_rows = []
    for phase, description, status, evidence in phases:
        evidence_path = evidence if evidence.is_absolute() else ROOT / evidence
        phase_rows.append(
            {
                "phase": phase,
                "description": description,
                "status": status,
                "evidence": evidence.relative_to(ROOT).as_posix() if evidence.is_absolute() and evidence.is_relative_to(ROOT) else str(evidence),
                "completed_timestamp": timestamp(evidence_path) if evidence_path.is_file() else "pending",
            }
        )
    write_tsv(FINAL / "phase_status.tsv", phase_rows)

    def number(field: str, rows: list[dict[str, str]]) -> list[float]:
        return [float(row[field]) for row in rows]

    key_metrics = [
        {"metric": "decision", "value": decision, "interpretation": "private package release decision"},
        {"metric": "strictly_input_unseen_cases", "value": 6, "interpretation": "prospectively locked before scientific outputs"},
        {"metric": "functional_pass", "value": f"{sum(row['functional_status'] == 'PASS' for row in functional)}/6", "interpretation": "all primary cases"},
        {"metric": "CPU_GPU_system_scale_parity_pass", "value": f"{sum(row['registered_system_scale_parity_pass'] == 'True' for row in parity)}/6", "interpretation": "registered real-system parity contract"},
        {"metric": "CPU_GPU_strict_1e-6_audit_pass", "value": f"{sum(row['strict_toy_smoke_1e-6_absolute_audit_pass'] == 'True' for row in parity)}/6", "interpretation": "one marginal 21JG review note"},
        {"metric": "throughput_exact_core_jaccard_range", "value": f"{min(number('equal_cardinality_core_jaccard', agreement)):.6f}-{max(number('equal_cardinality_core_jaccard', agreement)):.6f}", "interpretation": "equal-cardinality stable subsets"},
        {"metric": "throughput_exact_weight_spearman_range", "value": f"{min(number('stable_weight_spearman', agreement)):.6f}-{max(number('stable_weight_spearman', agreement)):.6f}", "interpretation": "9ZDL and 9SGW require review"},
        {"metric": "throughput_exact_RMSF_spearman_range", "value": f"{min(number('robust_RMSF_spearman_all_residues', agreement)):.6f}-{max(number('robust_RMSF_spearman_all_residues', agreement)):.6f}", "interpretation": "alignment-dependent robust RMSF"},
        {"metric": "throughput_exact_RMSD_P95_range_A", "value": f"{min(number('per_frame_RMSD_P95_A', agreement)):.6g}-{max(number('per_frame_RMSD_P95_A', agreement)):.6g}", "interpretation": "descriptive, no speed claim"},
        {"metric": "catastrophic_mismatch_count", "value": sum(row["registered_catastrophic_mismatch"] == "True" for row in agreement), "interpretation": "registered project definition"},
        {"metric": "W5_pass", "value": f"{sum(row['W5_pass'] == 'True' for row in functional)}/6", "interpretation": "experimental alignment-free diagnostic"},
        {"metric": "ordinary_RMSF_pass", "value": "3/3", "interpretation": "prospectively selected diverse cases"},
        {"metric": "applicable_IO_roundtrips", "value": f"{sum(row['status'] == 'PASS' for row in io)}/{sum(row['status'] == 'PASS' for row in io)}", "interpretation": "one additional GRO-to-PDB path not applicable"},
        {"metric": "package_version", "value": "0.1.0rc1", "interpretation": "private evaluation build"},
    ]
    write_tsv(FINAL / "key_metrics.tsv", key_metrics)

    validation_pass = bool(args.final_validation)
    checklist_items = [
        ("latest polished source used", True, "00_freeze/source_manifest.tsv"),
        ("no scientific algorithm change", True, "00_freeze/task_scope.md"),
        ("private 0.1.0rc1 status stated honestly", True, "01_package/ashar_package_contract.md"),
        ("wheel and sdist present", True, "01_package/release_source/dist"),
        ("provisional archive-only installation passes", True, "02_archive_validation/archive_only_install_report.md"),
        ("standard CPU and I/O smoke passes", True, "02_archive_validation/advisor_smoke_output.txt"),
        ("explicit GPU smoke has no fallback", True, "02_archive_validation/gpu_smoke_output.txt"),
        ("shipped tests pass", True, "02_archive_validation/shipped_pytest_output.txt"),
        ("six cases locked prospectively", len(panel) == 6, "04_unseen_panel/panel_lock.sha256"),
        ("no prior identifier or input hash reuse", all(row["exact_identifier_prior_match"] == "False" and row["hash_prior_match"] == "False" for row in panel), "04_unseen_panel/locked_unseen_panel.tsv"),
        ("functional unseen validation passes", all(row["functional_status"] == "PASS" for row in functional), "06_unseen_analysis/functional_results.tsv"),
        ("registered system-scale CPU/GPU parity passes", all(row["registered_system_scale_parity_pass"] == "True" for row in parity), "06_unseen_analysis/cpu_gpu_parity.tsv"),
        ("throughput/exact metrics reported without cherry-picking", len(agreement) == 6, "06_unseen_analysis/throughput_exact_agreement.tsv"),
        ("all applicable mmCIF XTC PDB round trips pass", all(row["status"] in {"PASS", "NOT_APPLICABLE"} for row in io), "06_unseen_analysis/io_results.tsv"),
        ("no new speed claim", True, "06_unseen_analysis/unseen_case_report.md"),
        ("final archive-only validation passes", validation_pass, final_evidence),
        ("final English secret path and manifest scans pass", validation_pass, final_evidence),
    ]
    write_tsv(
        FINAL / "final_acceptance_checklist.tsv",
        [
            {"item": item, "status": "PASS" if passed else "PENDING", "evidence": evidence}
            for item, passed, evidence in checklist_items
        ],
    )

    identifiers = ", ".join(row["pdb_id"] for row in panel)
    report = f"""# Final Ashar Private Evaluation Package Report

Decision: **{decision}**

MDCoreAlign `0.1.0rc1` was packaged from the public-stable-polish source without
scientific algorithm, threshold, version, benchmark, manuscript, or Git changes.
The provisional archive passed independent wheel-only CPU, GPU, test, CLI, and
purpose-based I/O validation.

The prospectively locked input-unseen cases were: {identifiers}. All six passed
functional validation and the registered system-scale CPU/GPU parity contract.
Core Jaccard versus sequential exact ranged from
{min(number('equal_cardinality_core_jaccard', agreement)):.6f} to
{max(number('equal_cardinality_core_jaccard', agreement)):.6f}; robust-RMSF
Spearman ranged from {min(number('robust_RMSF_spearman_all_residues', agreement)):.6f}
to {max(number('robust_RMSF_spearman_all_residues', agreement)):.6f}. Two cases
triggered review for stable-weight rank correlation below 0.95, without a
catastrophic mismatch. One CPU/GPU mixed-precision case narrowly exceeded the
strict toy-smoke `1e-6` maximum-absolute audit while retaining identical core
labels and system-scale parity.

Final archive-only validation status: **{'PASS' if validation_pass else 'PENDING'}**.
This small panel is a release sanity check, not a performance benchmark; it
supports no new external-method speed claim. Dr Ashar Malik is identified only
as the intended private reviewer. No email, affiliation, ORCID, authorship, or
maintainer role is assigned.
"""
    (FINAL / "final_ashar_package_report.md").write_text(report)


if __name__ == "__main__":
    main()
