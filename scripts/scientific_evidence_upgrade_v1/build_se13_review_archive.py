"""Build and independently validate the mandatory SE13 review archive."""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from common import REPO_ROOT, RESULT_ROOT, read_tsv, sha256_file, write_json, write_tsv

DATE = "20260901"
ARCHIVE_ROOT = RESULT_ROOT / "13_review_archive"
STAGING = ARCHIVE_ROOT / "handoff_to_chatgpt"
ARCHIVE = RESULT_ROOT / f"scientific_evidence_upgrade_v1_review_bundle_{DATE}.tar.gz"


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def sanitize_private_paths(root: Path) -> None:
    """Replace host-private paths in copied text records only."""

    replacements = {
        "/home/" + "jjxu": "<HOME>",
        "/storage/" + "junjie/projects/mdlovofit": "<REPO_ROOT>",
        "/storage/" + "junjie": "<STORAGE_ROOT>",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".md", ".tsv", ".json", ".py", ".txt", ".xml", ".sha256",
        }:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for source, target in replacements.items():
            content = content.replace(source, target)
        path.write_text(content, encoding="utf-8")


def paired_value(source: str, comparison: str, metric: str) -> dict[str, str]:
    rows = read_tsv(RESULT_ROOT / "07_heldout_geometry/paired_method_summary.tsv")
    return next(
        row for row in rows
        if row["source_type"] == source
        and row["fold_type"] == "contiguous"
        and row["comparison"] == comparison
        and row["metric"] == metric
    )


def key_metrics() -> list[dict[str, object]]:
    synthetic = read_tsv(
        RESULT_ROOT / "10_integrated_results/tables/synthetic_benchmark_summary.tsv"
    )
    budget = read_tsv(RESULT_ROOT / "10_integrated_results/tables/budget_fidelity_table.tsv")
    complexity = read_tsv(RESULT_ROOT / "08_budget_fidelity/empirical_complexity.tsv")
    convention = json.loads(
        (RESULT_ROOT / "05_mdcath_rmsf_convention/audit_summary.json").read_text()
    )

    def synthetic_value(scenario: str, method: str, metric: str) -> str:
        return next(
            row[metric] for row in synthetic
            if row["scenario"] == scenario and row["method"] == method
        )

    production = next(
        row for row in budget
        if row["source"] == "mdCATH" and row["configuration"] == "production"
    )
    atlas = next(
        row for row in budget
        if row["source"] == "ATLAS_R1" and row["configuration"] == "production"
    )
    random_sd = paired_value(
        "mdCATH", "MDCoreAlign_throughput_minus_random_equal_cardinality",
        "median_pair_distance_SD_A",
    )
    mdl_sd = paired_value(
        "mdCATH", "MDCoreAlign_throughput_minus_MDLovoFit",
        "median_pair_distance_SD_A",
    )
    frame_fit = next(
        row for row in complexity
        if row["scaling_axis"] == "frames"
        and row["runtime_component"] == "all_frame_fit_seconds"
    )
    weight = next(
        row for row in complexity
        if row["scaling_axis"] == "frames"
        and row["runtime_component"] == "weight_estimation_seconds"
    )
    rows = [
        ("synthetic_S1_throughput_core_AUPRC", synthetic_value("S1", "MDCoreAlign_throughput_CPU", "core_AUPRC"), "heldout_template_clustered_median", "04_transform_metrics/core_recovery_summary.tsv"),
        ("synthetic_S2_throughput_core_AUPRC", synthetic_value("S2", "MDCoreAlign_throughput_CPU", "core_AUPRC"), "heldout_template_clustered_median", "04_transform_metrics/core_recovery_summary.tsv"),
        ("synthetic_S5_throughput_core_AUPRC", synthetic_value("S5", "MDCoreAlign_throughput_CPU", "core_AUPRC"), "heldout_template_clustered_median", "04_transform_metrics/core_recovery_summary.tsv"),
        ("synthetic_S3_throughput_core_Jaccard", synthetic_value("S3", "MDCoreAlign_throughput_CPU", "core_Jaccard"), "heldout_template_clustered_median", "04_transform_metrics/core_recovery_summary.tsv"),
        ("synthetic_S4_throughput_core_Jaccard", synthetic_value("S4", "MDCoreAlign_throughput_CPU", "core_Jaccard"), "heldout_template_clustered_median", "04_transform_metrics/core_recovery_summary.tsv"),
        ("synthetic_S3_throughput_rotation_error_deg", synthetic_value("S3", "MDCoreAlign_throughput_CPU", "rotation_error_median_deg"), "heldout_template_clustered_median", "04_transform_metrics/rotation_error_summary.tsv"),
        ("synthetic_S4_throughput_rotation_error_deg", synthetic_value("S4", "MDCoreAlign_throughput_CPU", "rotation_error_median_deg"), "heldout_template_clustered_median", "04_transform_metrics/rotation_error_summary.tsv"),
        ("SE7_systems", 520, "512 mdCATH plus 8 ATLAS R1", "07_heldout_geometry/panel_lock.tsv"),
        ("SE7_method_fold_rows", 20800, "all successful", "07_heldout_geometry/pair_distance_results.tsv"),
        ("mdCATH_MDCA_minus_random_pair_SD_A", random_sd["median"], f"95% CI {random_sd['bootstrap_95CI_low']} to {random_sd['bootstrap_95CI_high']}", "07_heldout_geometry/paired_method_summary.tsv"),
        ("mdCATH_MDCA_minus_MDLovoFit_pair_SD_A", mdl_sd["median"], f"95% CI {mdl_sd['bootstrap_95CI_low']} to {mdl_sd['bootstrap_95CI_high']}", "07_heldout_geometry/paired_method_summary.tsv"),
        ("mdCATH_RMSF_audit_passed", convention["n_passed"], f"of {convention['n_cases']}; SE6 stopped", "05_mdcath_rmsf_convention/audit_summary.json"),
        ("mdCATH_RMSF_audit_max_abs_difference_A", convention["maximum_absolute_difference_A"], "locked maximum 0.0001 A", "05_mdcath_rmsf_convention/audit_summary.json"),
        ("production_mdcath_core_Jaccard_vs_exact", production["median_core_Jaccard_vs_sequential"], "384 heldout domains", "08_budget_fidelity/validation_results.tsv"),
        ("production_mdcath_RMSF_Spearman_vs_exact", production["median_RMSF_Spearman_vs_sequential"], "384 heldout domains", "08_budget_fidelity/validation_results.tsv"),
        ("production_mdcath_tail_rate", production["tail_rate"], "registered tail definition", "08_budget_fidelity/validation_results.tsv"),
        ("production_ATLAS_core_Jaccard_vs_exact", atlas["median_core_Jaccard_vs_sequential"], "8 R1 systems", "08_budget_fidelity/atlas_validation_results.tsv"),
        ("frame_scaling_all_frame_fit_log_slope", frame_fit["log_log_slope"], f"R2={frame_fit['R_squared']}", "08_budget_fidelity/empirical_complexity.tsv"),
        ("frame_scaling_weight_estimation_log_slope", weight["log_log_slope"], f"R2={weight['R_squared']}", "08_budget_fidelity/empirical_complexity.tsv"),
        ("full_tests_passed", 407, "11 skipped; 2 subtests passed", "12_validation/full_test_output_rerun.txt"),
        ("focused_tests_passed", 11, "scientific evidence regression", "12_validation/focused_test_output.txt"),
        ("narrative_decision", "C", "do not force a high-impact alignment-sensitivity narrative", "10_integrated_results/narrative_decision.md"),
    ]
    return [
        {"metric": metric, "value": value, "context": context, "source": source}
        for metric, value, context, source in rows
    ]


def phase_status() -> list[dict[str, str]]:
    rows = [
        ("SE0", "complete", "Source/evidence/environment baseline frozen; Git HEAD unavailable."),
        ("SE1", "complete_with_limitation", "Historical CPU evidence is two-thread; single-thread external contract unresolved."),
        ("SE2", "complete", "New analyses preregistered and hashed before held-out inspection."),
        ("SE3", "complete", "684 controlled settings; 3,420 method rows; zero failures."),
        ("SE4", "complete", "Rotation geodesic, coordinate, core-ranking, RMSD and RMSF truth metrics."),
        ("SE5", "complete_gate_failed", "96/100 passed; four cases exceeded locked tolerance."),
        ("SE6", "skipped_by_registered_rule", "No full-library alignment-sensitivity result was generated."),
        ("SE7", "complete", "520 systems; 20,800 method-fold rows; zero failures."),
        ("SE8", "complete_with_registry_error", "Supported K/M panels and scaling complete; M=8 retained as unsupported."),
        ("SE9", "complete", "Primary-source method and RMSF-label provenance audit."),
        ("SE10", "complete", "Four-layer integration; Narrative C selected."),
        ("SE11", "complete", "Five PDF/PNG figure pairs and six publication tables."),
        ("SE12", "complete", "Eleven manuscript-ready draft assets without overwriting the manuscript."),
        ("SE13", "complete", "407 full tests passed; archive independently validated."),
    ]
    return [{"phase": phase, "status": status, "notes": notes} for phase, status, notes in rows]


def write_entry_files() -> None:
    metrics = key_metrics()
    phases = phase_status()
    write_tsv(STAGING / "key_metrics.tsv", metrics, list(metrics[0]))
    write_tsv(STAGING / "phase_status.tsv", phases, list(phases[0]))
    (STAGING / "00_README_FIRST.md").write_text(
        "# MDCoreAlign scientific evidence upgrade v1 review handoff\n\n"
        "This self-contained bundle contains the preregistered scientific evidence "
        "upgrade, compact supporting tables, figures, manuscript drafts, study code, "
        "and validation logs. It does not contain raw mdCATH/ATLAS trajectories or "
        "generated coordinate arrays.\n\n"
        "## Canonical commands\n\n"
        "```bash\n"
        "python -m pytest -q tests/scientific_evidence_upgrade_v1/test_scientific_evidence_upgrade.py\n"
        "python -m pytest -q\n"
        "python -m ruff check scripts/scientific_evidence_upgrade_v1 tests/scientific_evidence_upgrade_v1\n"
        "python scripts/scientific_evidence_upgrade_v1/finalize_se10_se12.py\n"
        "```\n\n"
        "## Read first\n\n"
        "- `final_scientific_evidence_report.md`: integrated findings.\n"
        "- `narrative_decision.md`: registered Narrative C decision.\n"
        "- `claim_evidence_matrix.tsv`: safe and unsafe claims.\n"
        "- `figures/`: five candidate figures in PDF and PNG.\n"
        "- `tests/`: canonical, focused, Ruff, compile and figure-build records.\n"
        "- `sha256_manifest.tsv`: hashes for every included file except itself.\n",
        encoding="utf-8",
    )
    (STAGING / "final_scientific_evidence_report.md").write_text(
        "# Final scientific evidence report\n\n"
        "## Decision\n\n"
        "Narrative C is selected. The evidence supports a bounded computational-method "
        "and tool-paper framing, not a universal stable-core solution or a database-wide "
        "alignment-sensitive RMSF claim.\n\n"
        "## Controlled generative truth\n\n"
        "Pure rigid motion was recovered to numerical precision. Throughput stable-core "
        "ranking was strong for one-loop, two-loop and continuous heteroscedastic motion. "
        "Hard-core recovery was materially weaker for hinge and hinge-plus-loop motion, "
        "and the no-unique-core scenario intentionally has no binary truth target.\n\n"
        "## Held-out real geometry\n\n"
        "Across 512 mdCATH domains and eight ATLAS R1 systems, MDCoreAlign and MDLovoFit "
        "cores were substantially more stable than random equal-cardinality controls. "
        "MDLovoFit had a small advantage in pair-distance fluctuation; contact-persistence "
        "differences were small or mixed. These are alignment-independent geometric "
        "criteria, not biological truth.\n\n"
        "## RMSF convention\n\n"
        "The mdCATH source audit identified a first-frame-anchor full-C-alpha Kabsch "
        "convention with mean-position fluctuation and nanometre storage. The independent "
        "audit reproduced ordering, units and correlations but exceeded the locked "
        "sub-millangstrom tolerance in four of 100 cases. SE6 was therefore skipped.\n\n"
        "## Budget and complexity\n\n"
        "Production K=256/M=4 retained high exact-relative core and RMSF fidelity on 384 "
        "held-out mdCATH domains and eight ATLAS R1 systems. Empirical scaling supports "
        "approximately O(KMN) fixed-budget weight estimation plus O(TN) explicit "
        "all-frame fitting/statistics. M=8 was preregistered in error and retained as an "
        "unsupported configuration.\n\n"
        "## Validation\n\n"
        "The final canonical suite passed 407 tests with 11 optional skips, one documented "
        "MDAnalysis dt warning and two passing subtests. Eleven focused study tests, Ruff, "
        "compileall and figure regeneration also passed.\n",
        encoding="utf-8",
    )


def copy_required_evidence() -> None:
    copy(
        RESULT_ROOT / "00_freeze/source_manifest.tsv",
        STAGING / "source_manifest.tsv",
    )
    copy(
        RESULT_ROOT / "00_freeze/evidence_manifest.tsv",
        STAGING / "evidence_manifest.tsv",
    )
    sanitized = (RESULT_ROOT / "00_freeze/environment_manifest.tsv").read_text()
    sanitized = sanitized.replace("/home/" + "jjxu", "<HOME>")
    (STAGING / "environment_manifest_sanitized.tsv").write_text(sanitized, encoding="utf-8")
    copy_tree(RESULT_ROOT / "02_preregistration", STAGING / "all_preregistrations")
    mappings = [
        ("01_existing_evidence_audit/cpu_timing_contract.tsv", "cpu_timing_contract.tsv"),
        ("03_synthetic_truth/template_manifest.tsv", "synthetic_template_manifest.tsv"),
        ("04_transform_metrics/synthetic_truth_results.tsv", "synthetic_truth_results.tsv"),
        ("04_transform_metrics/core_recovery_summary.tsv", "core_recovery_summary.tsv"),
        ("04_transform_metrics/rotation_error_summary.tsv", "rotation_error_summary.tsv"),
        ("05_mdcath_rmsf_convention/convention_decision.md", "mdcath_convention_decision.md"),
        ("05_mdcath_rmsf_convention/reproduction_results.tsv", "mdcath_reproduction_results.tsv"),
        ("07_heldout_geometry/paired_method_summary.tsv", "heldout_geometry_summary.tsv"),
        ("10_integrated_results/tables/budget_fidelity_table.tsv", "budget_fidelity_summary.tsv"),
        ("08_budget_fidelity/empirical_complexity.tsv", "empirical_complexity.tsv"),
        ("09_literature_audit/rmsf_label_provenance.tsv", "literature_label_provenance.tsv"),
        ("10_integrated_results/claim_evidence_matrix.tsv", "claim_evidence_matrix.tsv"),
        ("10_integrated_results/narrative_decision.md", "narrative_decision.md"),
        ("10_integrated_results/limitations.md", "known_limitations.md"),
    ]
    for source, destination in mappings:
        copy(RESULT_ROOT / source, STAGING / destination)
    write_tsv(
        STAGING / "mdcath_per_domain_metrics.tsv",
        [{
            "status": "not_generated",
            "reason": "SE5 reproduction gate failed; SE6 skipped by preregistered stop rule",
            "n_domain_rows": 0,
        }],
        ["status", "reason", "n_domain_rows"],
    )
    (STAGING / "alignment_sensitivity_decision.md").write_text(
        "# Alignment-sensitivity decision\n\n"
        "No database-scale decision was estimated. SE6 was skipped because the "
        "independent mdCATH RMSF convention audit failed the prospectively locked "
        "numerical tolerance in four of 100 cases.\n",
        encoding="utf-8",
    )


def copy_supporting_material() -> None:
    copy_tree(RESULT_ROOT / "11_manuscript_materials", STAGING / "manuscript_materials")
    copy_tree(RESULT_ROOT / "10_integrated_results/figures", STAGING / "figures")
    copy_tree(RESULT_ROOT / "12_validation", STAGING / "tests")
    copy_tree(RESULT_ROOT / "10_integrated_results/tables", STAGING / "publication_tables")
    support = STAGING / "supporting_evidence"
    support.mkdir(parents=True, exist_ok=True)
    for relative in (
        "03_synthetic_truth/generator_config.json",
        "03_synthetic_truth/generator_validation.tsv",
        "03_synthetic_truth/trajectory_manifest.tsv",
        "04_transform_metrics/internal_motion_summary.tsv",
        "04_transform_metrics/metric_implementation.md",
        "05_mdcath_rmsf_convention/audit_panel.tsv",
        "05_mdcath_rmsf_convention/audit_summary.json",
        "05_mdcath_rmsf_convention/reproduction_failure_adjudication.md",
        "05_mdcath_rmsf_convention/source_audit.md",
        "05_mdcath_rmsf_convention/unit_mapping.tsv",
        "06_mdcath_rmsf_sensitivity/SKIPPED.md",
        "07_heldout_geometry/panel_lock.tsv",
        "07_heldout_geometry/fold_registry.tsv",
        "07_heldout_geometry/pair_distance_results.tsv",
        "08_budget_fidelity/budget_grid.tsv",
        "08_budget_fidelity/development_results.tsv",
        "08_budget_fidelity/validation_results.tsv",
        "08_budget_fidelity/synthetic_validation_results.tsv",
        "08_budget_fidelity/atlas_validation_results.tsv",
        "08_budget_fidelity/pareto_results.tsv",
        "08_budget_fidelity/scaling_results.tsv",
        "08_budget_fidelity/unsupported_budget_adjudication.md",
        "09_literature_audit/claim_citation_audit.md",
        "09_literature_audit/literature_search_log.tsv",
        "09_literature_audit/related_methods_matrix.tsv",
        "10_integrated_results/evidence_layer_matrix.tsv",
        "10_integrated_results/figure_validation.tsv",
    ):
        copy(RESULT_ROOT / relative, support / relative)
    code = STAGING / "changed_study_code"
    copy_tree(REPO_ROOT / "scripts/scientific_evidence_upgrade_v1", code / "scripts")
    copy_tree(
        REPO_ROOT / "tests/scientific_evidence_upgrade_v1",
        code / "tests/scientific_evidence_upgrade_v1",
    )
    for cache in code.rglob("__pycache__"):
        shutil.rmtree(cache)
    sanitize_private_paths(STAGING / "tests")
    sanitize_private_paths(STAGING / "changed_study_code")


def text_files(root: Path) -> list[Path]:
    suffixes = {".md", ".tsv", ".json", ".py", ".txt", ".xml", ".sha256"}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def scan_staging() -> dict[str, object]:
    secret_patterns = [
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"(?i)(api[_-]?key|password|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    ]
    secret_hits: list[str] = []
    private_path_hits: list[str] = []
    for path in text_files(STAGING):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if any(pattern.search(line) for pattern in secret_patterns):
                secret_hits.append(f"{path.relative_to(STAGING)}:{line_number}")
            private_home = "/home/" + "jjxu"
            private_storage = "/storage/" + "junjie"
            if private_home in line or private_storage in line:
                private_path_hits.append(f"{path.relative_to(STAGING)}:{line_number}")
    largest = max((path.stat().st_size, path) for path in STAGING.rglob("*") if path.is_file())
    return {
        "secret_hits": secret_hits,
        "private_absolute_path_hits": private_path_hits,
        "largest_file": str(largest[1].relative_to(STAGING)),
        "largest_file_bytes": largest[0],
        "files_over_100_MB": [
            str(path.relative_to(STAGING))
            for path in STAGING.rglob("*")
            if path.is_file() and path.stat().st_size > 100 * 1024 * 1024
        ],
    }


def write_hash_manifest() -> None:
    files = [
        path for path in sorted(STAGING.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.tsv"
    ]
    rows = [
        {
            "path": str(path.relative_to(STAGING)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    write_tsv(STAGING / "sha256_manifest.tsv", rows, list(rows[0]))


def create_archive() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(STAGING, arcname="handoff_to_chatgpt")


def validate_archive(scan: dict[str, object]) -> dict[str, object]:
    required = {
        "00_README_FIRST.md", "final_scientific_evidence_report.md",
        "phase_status.tsv", "key_metrics.tsv", "source_manifest.tsv",
        "evidence_manifest.tsv", "all_preregistrations", "cpu_timing_contract.tsv",
        "synthetic_template_manifest.tsv", "synthetic_truth_results.tsv",
        "core_recovery_summary.tsv", "rotation_error_summary.tsv",
        "mdcath_convention_decision.md", "mdcath_reproduction_results.tsv",
        "mdcath_per_domain_metrics.tsv", "alignment_sensitivity_decision.md",
        "heldout_geometry_summary.tsv", "budget_fidelity_summary.tsv",
        "empirical_complexity.tsv", "literature_label_provenance.tsv",
        "claim_evidence_matrix.tsv", "narrative_decision.md",
        "manuscript_materials", "figures", "tests", "changed_study_code",
        "sha256_manifest.tsv",
    }
    present = {path.name for path in STAGING.iterdir()}
    missing = sorted(required - present)
    with tempfile.TemporaryDirectory(prefix="sev1-review-verify-") as temporary:
        destination = Path(temporary)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(destination, filter="data")
        extracted = destination / "handoff_to_chatgpt"
        manifest = read_tsv(extracted / "sha256_manifest.tsv")
        hash_failures = []
        for row in manifest:
            path = extracted / row["path"]
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                hash_failures.append(row["path"])
        extracted_count = sum(path.is_file() for path in extracted.rglob("*"))
    return {
        "archive": str(ARCHIVE.relative_to(REPO_ROOT)),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256_file(ARCHIVE),
        "included_file_count": sum(path.is_file() for path in STAGING.rglob("*")),
        "extracted_file_count": extracted_count,
        "missing_required": missing,
        "manifest_hash_failures": hash_failures,
        "secret_scan_passed": not scan["secret_hits"],
        "private_absolute_path_scan_passed": not scan["private_absolute_path_hits"],
        "files_over_100_MB": scan["files_over_100_MB"],
        "largest_file": scan["largest_file"],
        "largest_file_bytes": scan["largest_file_bytes"],
        "extraction_and_manifest_verification_passed": (
            not missing and not hash_failures and extracted_count == sum(path.is_file() for path in STAGING.rglob("*"))
        ),
    }


def main() -> None:
    if ARCHIVE_ROOT.exists():
        shutil.rmtree(ARCHIVE_ROOT)
    STAGING.mkdir(parents=True)
    write_entry_files()
    copy_required_evidence()
    copy_supporting_material()
    scan = scan_staging()
    if scan["secret_hits"] or scan["private_absolute_path_hits"] or scan["files_over_100_MB"]:
        raise RuntimeError(json.dumps(scan, indent=2))
    write_hash_manifest()
    create_archive()
    validation = validate_archive(scan)
    write_json(RESULT_ROOT / "scientific_evidence_upgrade_v1_review_bundle_validation.json", validation)
    if validation["missing_required"] or validation["manifest_hash_failures"]:
        raise RuntimeError(json.dumps(validation, indent=2))
    if not validation["extraction_and_manifest_verification_passed"]:
        raise RuntimeError("Archive extraction or manifest verification failed")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
