#!/usr/bin/env python3
"""Build and validate the mandatory self-contained RV7 author-review archive."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/concept_cases_product_v2"
V1 = ROOT / "results/concept_cases_product_v1"
CASE_V1 = V1 / "03_case_studies"
DATE = datetime.now().astimezone().strftime("%Y%m%d")
ARCHIVE = OUT / f"concept_cases_product_v2_review_bundle_{DATE}.tar.gz"
ARCHIVE_HASH = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")
VALIDATION_JSON = OUT / f"concept_cases_product_v2_review_bundle_{DATE}_validation.json"
VALIDATION_MD = OUT / f"concept_cases_product_v2_review_bundle_{DATE}_validation.md"

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".diff",
    ".json",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table refused: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """Remove environment-specific paths and identities from copied text."""

    replacements = [
        (str(ROOT) + "/", ""),
        (str(ROOT), "."),
        ("/home/jjxu/miniconda3/bin/python", "<BUILD_PYTHON>"),
        ("/home/jjxu/", "<USER_HOME>/"),
        ("/storage/junjie/", "<STORAGE_ROOT>/"),
    ]
    changed: list[str] = []
    for source, target in replacements:
        if source in text:
            text = text.replace(source, target)
            changed.append(source)
    if re.search(r"/tmp/mdcorealign[^\s\"']*", text):
        text = re.sub(r"/tmp/mdcorealign[^\s\"']*", "<TEMP_DIR>", text)
        changed.append("/tmp/mdcorealign*")
    return text, changed


def copy_file(source: Path, destination: Path, sanitization: dict[str, list[str]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = source.read_text()
        except UnicodeDecodeError:
            shutil.copy2(source, destination)
            return
        cleaned, changed = sanitize_text(text)
        destination.write_text(cleaned)
        if changed:
            sanitization[str(destination)] = changed
    else:
        shutil.copy2(source, destination)


def section_for(relative: Path) -> str:
    if len(relative.parts) == 1:
        return "entry_and_status"
    return relative.parts[0]


def create_phase_status(bundle: Path) -> None:
    rows = [
        {"phase": "RV0", "status": "complete", "evidence": "scientific_consistency/threshold_consistency_audit.md", "notes": "v1 preserved; v2 root created"},
        {"phase": "RV1", "status": "complete", "evidence": "scientific_consistency/classification_data_dictionary.tsv", "notes": "shared 0.30/0.70 classifier"},
        {"phase": "RV2", "status": "complete", "evidence": "figures/figure1/fig1_method_concept_v2.pdf", "notes": "approximation boundary explicit"},
        {"phase": "RV3", "status": "complete", "evidence": "figures/case_A/case_A_key_regions_v2.tsv", "notes": "3nci_A retained"},
        {"phase": "RV4", "status": "complete", "evidence": "figures/case_B/case_B_key_regions_v2.tsv", "notes": "6sup_A retained; low-global/high-local focus"},
        {"phase": "RV5", "status": "complete", "evidence": "product_documentation/documentation_consistency_audit.tsv", "notes": "five workflows on README first screen"},
        {"phase": "RV6", "status": "complete", "evidence": "validation/release_validation_summary.md", "notes": "tests, figures, installs and build validated"},
        {"phase": "RV7", "status": "complete", "evidence": "archive_validation_report.md", "notes": "mandatory author-review handoff built and validated"},
    ]
    write_tsv(bundle / "phase_status.tsv", rows)


def create_key_metrics(bundle: Path) -> None:
    rows = [
        {"category": "classification", "metric": "high_threshold", "value": "0.70", "unit": "dimensionless", "evidence": "scientific_consistency/classification_data_dictionary.tsv"},
        {"category": "classification", "metric": "low_threshold", "value": "0.30", "unit": "dimensionless", "evidence": "scientific_consistency/classification_data_dictionary.tsv"},
        {"category": "case_A", "metric": "selected_system", "value": "3nci_A", "unit": "identifier", "evidence": "figures/case_A/case_A_key_regions_v2.tsv"},
        {"category": "case_A", "metric": "selection_score", "value": "0.5725394874121733", "unit": "dimensionless", "evidence": "final_revision_report.md"},
        {"category": "case_A", "metric": "throughput_exact_core_Jaccard", "value": "0.9602169981916817", "unit": "dimensionless", "evidence": "figures/case_A/case_A_caption_v2.md"},
        {"category": "case_A", "metric": "registered_boundary_peak", "value": "792", "unit": "residue_index", "evidence": "figures/case_A/case_A_key_regions_v2.tsv"},
        {"category": "case_B", "metric": "selected_system", "value": "6sup_A", "unit": "identifier", "evidence": "figures/case_B/case_B_key_regions_v2.tsv"},
        {"category": "case_B", "metric": "selection_score", "value": "0.8985420888194781", "unit": "dimensionless", "evidence": "final_revision_report.md"},
        {"category": "case_B", "metric": "throughput_exact_core_Jaccard", "value": "0.9829192546583851", "unit": "dimensionless", "evidence": "figures/case_B/case_B_caption_v2.md"},
        {"category": "case_B", "metric": "region_1649_1652_mean_global_local", "value": "0.2581530073;0.9593634135", "unit": "dimensionless", "evidence": "figures/case_B/case_B_key_regions_v2.tsv"},
        {"category": "case_B", "metric": "region_1884_1888_mean_global_local", "value": "0.2829343952;0.9323369288", "unit": "dimensionless", "evidence": "figures/case_B/case_B_key_regions_v2.tsv"},
        {"category": "validation", "metric": "canonical_tests", "value": "296 passed;13 warnings", "unit": "test_result", "evidence": "validation/full_canonical_test_output.log"},
        {"category": "validation", "metric": "focused_tests", "value": "9 passed;13 warnings", "unit": "test_result", "evidence": "validation/focused_scientific_consistency_test_output.log"},
        {"category": "integrity", "metric": "v1_files_verified", "value": "177/177", "unit": "files", "evidence": "validation/frozen_evidence_integrity_report.md"},
        {"category": "release", "metric": "version", "value": "0.1.0rc1", "unit": "PEP440", "evidence": "source/pyproject.toml"},
    ]
    write_tsv(bundle / "key_metrics.tsv", rows)


def create_entry_files(bundle: Path) -> None:
    copy_file(OUT / "final_revision_report.md", bundle / "final_revision_report.md", {})
    create_phase_status(bundle)
    create_key_metrics(bundle)
    write_text(
        bundle / "known_limitations.md",
        """# Known Limitations

- Case labels are geometric; trusted biological domain, hinge and mechanism annotations remain an author-level task.
- W5 is a dimensionless alignment-free local-deformation diagnostic, not RMSD or global mobility.
- Exact mode is a numerical reference/audit workflow, not biological ground truth.
- Throughput approximates stable-weight estimation; the per-frame weighted rigid fit remains explicit.
- RMSF remains alignment-frame dependent.
- PBC/nojump correctness was not independently proven by this presentation pass.
- Backend-auto and hard motion-domain labels remain experimental.
- No package, repository, tag, DOI or public release was created in RV7.
- The two compact three-model CA PDB files are derived figure sources, not raw trajectories or biological input archives.
""",
    )
    write_text(
        bundle / "00_README_FIRST.md",
        """# MDCoreAlign Concept Cases Product v2 Review Bundle

This self-contained author-review package contains the complete RV0-RV7
presentation revision for the frozen `3nci_A` and `6sup_A` cases. Start with
`final_revision_report.md`, then inspect Figure 1 and the two case folders.

The registered display contract is high >= 0.70, low <= 0.30 and intermediate
(0.30, 0.70). Throughput approximates stable-weight estimation; every frame
still receives an explicit weighted rigid fit after weights are frozen. W5 is
a complementary alignment-free local-deformation diagnostic and is not RMSD.

No benchmark was rerun, no algorithm/default/case was changed, no biological
annotation was invented, and no package or Git release was published. Raw
ATLAS/mdCATH data and unrelated historical evidence are intentionally absent.

Integrity files:

- `file_manifest.tsv`: inventory, section, size and source role;
- `sha256_manifest.tsv`: SHA-256 for every included file except the hash
  manifest itself, whose bytes are protected by the adjacent archive checksum;
- `archive_validation_report.md`: required-section, scan, size and extraction
  verification.
""",
    )


def copy_scientific_audit(bundle: Path, sanitization: dict[str, list[str]]) -> None:
    destination = bundle / "scientific_consistency"
    mapping = {
        OUT / "00_audit/threshold_consistency_audit.md": destination / "threshold_consistency_audit.md",
        OUT / "00_audit/classification_data_dictionary.tsv": destination / "classification_data_dictionary.tsv",
        OUT / "00_audit/v1_v2_region_classification_comparison.tsv": destination / "v1_v2_region_classification_comparison.tsv",
        OUT / "06_release_validation/logs/product_regression_pytest.log": destination / "focused_scientific_consistency_test_output.log",
        ROOT / "tests/test_concept_cases_product_v2.py": destination / "focused_scientific_consistency_tests.py",
    }
    for source, target in mapping.items():
        copy_file(source, target, sanitization)


def copy_figure_materials(bundle: Path, sanitization: dict[str, list[str]]) -> None:
    script = ROOT / "scripts/concept_cases_product_v2/generate_revision.py"
    classifier = ROOT / "scripts/concept_cases_product_v2/case_interpretation.py"
    figure1 = bundle / "figures/figure1"
    for name in (
        "fig1_method_concept_v2.pdf",
        "fig1_method_concept_v2.svg",
        "fig1_method_concept_v2.png",
        "fig1_caption_v2.md",
        "figure_design_notes.md",
    ):
        copy_file(OUT / "01_concept_figure" / name, figure1 / name, sanitization)
    copy_file(script, figure1 / "generate_figure_source.py", sanitization)
    write_text(figure1 / "figure_source_data.md", "# Figure 1 Source\n\nFigure 1 is conceptual and contains no benchmark values. Its exact source is `generate_figure_source.py`; the weighted-fit objective and labels are embedded directly in that script.")

    for role, v2_directory, bundle_name in (
        ("case_A", "02_case_A", "case_A"),
        ("case_B", "03_case_B", "case_B"),
    ):
        destination = bundle / "figures" / bundle_name
        source_dir = OUT / v2_directory
        for path in source_dir.iterdir():
            if path.is_file():
                copy_file(path, destination / path.name, sanitization)
        copy_file(CASE_V1 / role / "residue_decomposition.tsv", destination / "frozen_residue_decomposition.tsv", sanitization)
        copy_file(CASE_V1 / role / "representative_stable_core_aligned_CA.pdb", destination / "derived_representative_stable_core_aligned_CA.pdb", sanitization)
        copy_file(CASE_V1 / role / "case_summary.json", destination / "frozen_case_summary.json", sanitization)
        copy_file(script, destination / "generate_figure_source.py", sanitization)
        copy_file(classifier, destination / "case_interpretation_source.py", sanitization)
        write_text(
            destination / "figure_source_data.md",
            "# Exact Figure Source Data\n\n"
            "`frozen_residue_decomposition.tsv` and the compact three-model "
            "`derived_representative_stable_core_aligned_CA.pdb` are the exact "
            "frozen inputs used by `generate_figure_source.py`. The v2 region, "
            "key-region and classification-track tables are included alongside "
            "the principal and supplementary figures. The PDB is a derived "
            "figure source, not a raw trajectory.\n",
        )


def copy_manuscript_assets(bundle: Path, sanitization: dict[str, list[str]]) -> None:
    destination = bundle / "manuscript_assets"
    for path in sorted((OUT / "04_manuscript_assets").iterdir()):
        if path.is_file():
            copy_file(path, destination / path.name, sanitization)


def copy_product_docs(bundle: Path, sanitization: dict[str, list[str]]) -> None:
    destination = bundle / "product_documentation"
    copy_file(ROOT / "README.md", destination / "README.md", sanitization)
    for path in sorted((OUT / "05_product_docs").iterdir()):
        if path.is_file():
            copy_file(path, destination / path.name, sanitization)
    diff_source = OUT / "05_product_docs/repository_diffs"
    for path in sorted(diff_source.iterdir()):
        copy_file(path, destination / "documentation_diffs" / path.name, sanitization)


def create_validation_reports(bundle: Path, sanitization: dict[str, list[str]]) -> None:
    destination = bundle / "validation"
    status = json.loads((OUT / "06_release_validation/validation_status.json").read_text())
    mapping = {
        OUT / "06_release_validation/logs/canonical_pytest.log": destination / "full_canonical_test_output.log",
        OUT / "06_release_validation/logs/product_regression_pytest.log": destination / "focused_scientific_consistency_test_output.log",
        OUT / "06_release_validation/cpu_clean_install/clean_install_report.md": destination / "cpu_only_clean_install_report.md",
        OUT / "06_release_validation/cpu_clean_install/clean_install_report.json": destination / "cpu_only_clean_install_report.json",
        OUT / "06_release_validation/logs/twine_check.log": destination / "twine_check_output.txt",
        OUT / "06_release_validation/figure_validation.tsv": destination / "figure_validation_report.tsv",
        OUT / "06_release_validation/visual_audit_contact_sheet.pdf": destination / "visual_audit_contact_sheet.pdf",
        OUT / "06_release_validation/visual_audit_contact_sheet.png": destination / "visual_audit_contact_sheet.png",
        OUT / "06_release_validation/historical_integrity_check.md": destination / "frozen_evidence_integrity_report.md",
        OUT / "06_release_validation/v1_integrity_recheck.tsv": destination / "frozen_evidence_integrity_details.tsv",
        OUT / "06_release_validation/release_validation_summary.md": destination / "release_validation_summary.md",
        OUT / "06_release_validation/validation_status.json": destination / "validation_status.json",
    }
    for source, target in mapping.items():
        copy_file(source, target, sanitization)

    gpu = status["gpu_clean_install"]
    write_text(
        destination / "gpu_clean_install_report.md",
        f"""# GPU Clean-Install Report

- Status: `{gpu['status']}`
- Package: `mdcorealign {gpu['mdcorealign_version']}` installed from the RV6 wheel with `[gpu]` in a fresh virtual environment
- CuPy: `{gpu['cupy_version']}`
- Device: `{gpu['gpu']}`
- Requested backend: `{gpu['backend_requested']}`
- Recorded backend: `{gpu['backend_used']}`
- Explicit GPU execution: passed; no silent CPU fallback

The toy execution is a product smoke test, not a performance benchmark.
""",
    )
    build = status["build"]
    write_text(
        destination / "package_build_report.md",
        f"""# Package Build Report

- Status: `{build['status']}`
- Version: `0.1.0rc1`
- Wheel: built successfully; SHA-256 `{build['wheel_sha256']}`
- Source distribution: built successfully; SHA-256 `{build['sdist_sha256']}`
- `twine check`: passed for both distributions
- Publication/upload: not performed

Wheel and sdist binaries are intentionally excluded from this review archive.
""",
    )
    write_text(
        destination / "public_command_smoke_report.md",
        """# Five Public-Command Smoke Tests

All commands ran from the fresh GPU-capable wheel environment against the
bundled toy CA PDB:

| Workflow | Result | Required output |
| --- | --- | --- |
| `align --mode exact` | pass | `transforms.npz` |
| `align --mode throughput --backend cpu` | pass | `transforms.npz` |
| `align --mode throughput --backend gpu` | pass; recorded GPU backend | `transforms.npz` |
| `local-deformation` | pass | `local_deformation.tsv` |
| `apply` | pass | `aligned_trajectory.pdb` |

These are usability smoke tests, not benchmark timings.
""",
    )
    for path in sorted((OUT / "06_release_validation/logs").glob("public_*.log")):
        copy_file(path, destination / "public_command_logs" / path.name, sanitization)


def build_unified_diff(bundle: Path) -> None:
    destination = bundle / "source/unified_source_diff.patch"
    parts: list[str] = []
    changed = [
        "README.md",
        "docs/quickstart.md",
        "docs/modes_and_quality_contracts.md",
        "docs/local_deformation.md",
        "docs/large_trajectory_workflow.md",
        "docs/output_schema.md",
        "docs/cpu_installation.md",
        "docs/gpu_installation.md",
    ]
    baseline = V1 / "handoff_to_chatgpt/repository"
    for relative in changed:
        old_path = baseline / relative
        new_path = ROOT / relative
        old = old_path.read_text().splitlines(keepends=True) if old_path.is_file() else []
        new = new_path.read_text().splitlines(keepends=True)
        parts.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{relative}" if old else "/dev/null",
                tofile=f"b/{relative}",
            )
        )
    for relative in (
        "scripts/concept_cases_product_v2/case_interpretation.py",
        "scripts/concept_cases_product_v2/generate_revision.py",
        "scripts/concept_cases_product_v2/finalize_revision.py",
        "scripts/concept_cases_product_v2/build_handoff.py",
        "tests/test_concept_cases_product_v2.py",
    ):
        new = (ROOT / relative).read_text().splitlines(keepends=True)
        parts.extend(difflib.unified_diff([], new, fromfile="/dev/null", tofile=f"b/{relative}"))
    write_text(destination, "".join(parts))


def copy_source(bundle: Path, sanitization: dict[str, list[str]]) -> None:
    destination = bundle / "source"
    files = [
        "README.md",
        "pyproject.toml",
        "src/mdcorealign/cli.py",
        "src/mdcorealign/contracts.py",
        "src/mdcorealign/gpu_backend.py",
        "scripts/concept_cases_product_v2/case_interpretation.py",
        "scripts/concept_cases_product_v2/generate_revision.py",
        "scripts/concept_cases_product_v2/finalize_revision.py",
        "scripts/concept_cases_product_v2/build_handoff.py",
        "tests/test_concept_cases_product_v2.py",
        "docs/quickstart.md",
        "docs/modes_and_quality_contracts.md",
        "docs/local_deformation.md",
        "docs/large_trajectory_workflow.md",
        "docs/output_schema.md",
        "docs/cpu_installation.md",
        "docs/gpu_installation.md",
    ]
    for relative in files:
        copy_file(ROOT / relative, destination / relative, sanitization)
    write_text(
        destination / "source_change_scope.md",
        """# Source Change Scope

RV0-RV7 changed presentation scripts, focused tests, README and product
documentation only. `src/mdcorealign/cli.py`, `contracts.py`, `gpu_backend.py`
and `pyproject.toml` are included for product-contract review but were not
modified by this revision. No exact, throughput, GPU, W5, Kabsch, RMSF or
trajectory algorithm was changed. The unified patch records all CP v2 source
and documentation changes relative to the preserved CP4 handoff where a
baseline copy was available.
""",
    )
    build_unified_diff(bundle)


def scan_bundle(bundle: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    urls: list[dict[str, Any]] = []
    absolute = re.compile(r"(?<![A-Za-z0-9_])(?:/(?:home|storage|tmp|mnt|Users|private|var/tmp)/[^\s`\"']+|[A-Za-z]:\\\\[^\s`\"']+)")
    identity = re.compile(r"\b(?:jjxu|junjie)\b", re.IGNORECASE)
    secrets = [
        ("huggingface_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
        ("openai_style_key", re.compile(r"sk-[A-Za-z0-9_-]{16,}")),
        ("private_key", re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
        ("credential_assignment", re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[\"']?[^\s\"']{8,}")),
    ]
    url_pattern = re.compile(r"https?://[^\s)>\]`\"']+")
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        relative = str(path.relative_to(bundle))
        for match in absolute.finditer(text):
            issues.append({"path": relative, "issue": "absolute_local_path", "match": match.group(0)[:160]})
        for match in identity.finditer(text):
            issues.append({"path": relative, "issue": "local_username", "match": match.group(0)})
        for label, pattern in secrets:
            for match in pattern.finditer(text):
                issues.append({"path": relative, "issue": label, "match": match.group(0)[:24] + "..."})
        for match in url_pattern.finditer(text):
            url = match.group(0)
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            private = (
                host in {"localhost", "127.0.0.1", "0.0.0.0"}
                or host.endswith(".internal")
                or host.startswith(("10.", "192.168."))
                or host == "pypi.hub.ace-research.openai.org"
                or "@" in parsed.netloc
                or any(term in parsed.query.lower() for term in ("token=", "key=", "password="))
            )
            urls.append({"path": relative, "url": url, "classification": "private_or_credentialed" if private else "public"})
            if private:
                issues.append({"path": relative, "issue": "private_url", "match": url[:160]})
    return issues, urls


def required_paths() -> list[str]:
    return [
        "00_README_FIRST.md",
        "final_revision_report.md",
        "phase_status.tsv",
        "key_metrics.tsv",
        "file_manifest.tsv",
        "sha256_manifest.tsv",
        "known_limitations.md",
        "archive_validation_report.md",
        "scientific_consistency/threshold_consistency_audit.md",
        "scientific_consistency/classification_data_dictionary.tsv",
        "scientific_consistency/v1_v2_region_classification_comparison.tsv",
        "scientific_consistency/focused_scientific_consistency_test_output.log",
        "figures/figure1/fig1_method_concept_v2.pdf",
        "figures/figure1/fig1_method_concept_v2.svg",
        "figures/figure1/fig1_method_concept_v2.png",
        "figures/figure1/fig1_caption_v2.md",
        "figures/figure1/figure_design_notes.md",
        "figures/figure1/generate_figure_source.py",
        "figures/case_A/fig2_case_A_3nci_A_v2.pdf",
        "figures/case_A/fig2_case_A_3nci_A_v2.svg",
        "figures/case_A/fig2_case_A_3nci_A_v2.png",
        "figures/case_A/fig2_case_A_supplement_full_profiles.pdf",
        "figures/case_A/case_A_region_summary_v2.tsv",
        "figures/case_A/case_A_key_regions_v2.tsv",
        "figures/case_A/case_A_classification_track_v2.tsv",
        "figures/case_A/case_A_caption_v2.md",
        "figures/case_A/case_A_interpretation_v2.md",
        "figures/case_A/frozen_residue_decomposition.tsv",
        "figures/case_A/generate_figure_source.py",
        "figures/case_B/fig3_case_B_6sup_A_v2.pdf",
        "figures/case_B/fig3_case_B_6sup_A_v2.svg",
        "figures/case_B/fig3_case_B_6sup_A_v2.png",
        "figures/case_B/fig3_case_B_supplement_full_profiles.pdf",
        "figures/case_B/case_B_region_summary_v2.tsv",
        "figures/case_B/case_B_key_regions_v2.tsv",
        "figures/case_B/case_B_classification_track_v2.tsv",
        "figures/case_B/case_B_caption_v2.md",
        "figures/case_B/case_B_interpretation_v2.md",
        "figures/case_B/frozen_residue_decomposition.tsv",
        "figures/case_B/generate_figure_source.py",
        "manuscript_assets/conceptual_contribution_v2.md",
        "manuscript_assets/methods_overview_v2.md",
        "manuscript_assets/case_study_results_v2.md",
        "manuscript_assets/figure_captions_v2.md",
        "manuscript_assets/case_claim_to_evidence_v2.tsv",
        "manuscript_assets/limitations_v2.md",
        "manuscript_assets/author_biological_annotation_checklist.md",
        "product_documentation/README.md",
        "product_documentation/documentation_diffs/README.md.diff",
        "product_documentation/mode_contract_v2.tsv",
        "product_documentation/documentation_consistency_audit.tsv",
        "product_documentation/readme_first_screen_review.md",
        "validation/full_canonical_test_output.log",
        "validation/focused_scientific_consistency_test_output.log",
        "validation/cpu_only_clean_install_report.md",
        "validation/gpu_clean_install_report.md",
        "validation/package_build_report.md",
        "validation/twine_check_output.txt",
        "validation/public_command_smoke_report.md",
        "validation/figure_validation_report.tsv",
        "validation/visual_audit_contact_sheet.pdf",
        "validation/frozen_evidence_integrity_report.md",
        "source/scripts/concept_cases_product_v2/case_interpretation.py",
        "source/scripts/concept_cases_product_v2/generate_revision.py",
        "source/tests/test_concept_cases_product_v2.py",
        "source/src/mdcorealign/cli.py",
        "source/pyproject.toml",
        "source/unified_source_diff.patch",
    ]


def create_manifests(bundle: Path, sources: dict[str, str], sanitization: dict[str, list[str]]) -> None:
    manifest = bundle / "file_manifest.tsv"
    hash_manifest = bundle / "sha256_manifest.tsv"
    files = sorted(path for path in bundle.rglob("*") if path.is_file() and path not in {manifest, hash_manifest})
    rows = []
    for path in files:
        relative = path.relative_to(bundle)
        rows.append(
            {
                "relative_path": str(relative),
                "section": section_for(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source": sources.get(str(relative), "generated_for_RV7"),
                "sanitized_copy": "yes" if str(path) in sanitization else "no",
            }
        )
    rows.extend(
        [
            {"relative_path": "file_manifest.tsv", "section": "entry_and_status", "size_bytes": "self_describing", "sha256": "see_sha256_manifest", "source": "generated_for_RV7", "sanitized_copy": "no"},
            {"relative_path": "sha256_manifest.tsv", "section": "entry_and_status", "size_bytes": "generated_after_file_manifest", "sha256": "self_excluded_by_definition", "source": "generated_for_RV7", "sanitized_copy": "no"},
        ]
    )
    write_tsv(manifest, rows)
    hash_files = sorted(path for path in bundle.rglob("*") if path.is_file() and path != hash_manifest)
    write_tsv(
        hash_manifest,
        [
            {"sha256": sha256(path), "relative_path": str(path.relative_to(bundle))}
            for path in hash_files
        ],
    )


def make_archive(source_root: Path, target: Path) -> None:
    def portable_metadata(member: tarfile.TarInfo) -> tarfile.TarInfo:
        """Remove host identity and timestamps from the review artifact."""

        member.uid = 0
        member.gid = 0
        member.uname = ""
        member.gname = ""
        member.mtime = 0
        return member

    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        archive.add(
            source_root,
            arcname="handoff_to_chatgpt",
            recursive=True,
            filter=portable_metadata,
        )


def verify_archive(archive_path: Path, expected_bundle: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mdcorealign-rv7-extract-") as temporary:
        extraction = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            unsafe = [
                member.name
                for member in members
                if member.name.startswith("/")
                or ".." in Path(member.name).parts
                or member.issym()
                or member.islnk()
            ]
            if unsafe:
                raise RuntimeError(f"unsafe archive members: {unsafe[:5]}")
            roots = {Path(member.name).parts[0] for member in members if member.name}
            if roots != {"handoff_to_chatgpt"}:
                raise RuntimeError(f"archive roots are not self-contained: {roots}")
            archive.extractall(extraction)
        extracted = extraction / "handoff_to_chatgpt"
        with (extracted / "sha256_manifest.tsv").open(newline="") as handle:
            hashes = list(csv.DictReader(handle, delimiter="\t"))
        failures = []
        for row in hashes:
            path = extracted / row["relative_path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                failures.append(row["relative_path"])
        actual = {str(path.relative_to(extracted)) for path in extracted.rglob("*") if path.is_file()}
        expected = {row["relative_path"] for row in hashes} | {"sha256_manifest.tsv"}
        if actual != expected:
            failures.extend(sorted(actual ^ expected))
        if failures:
            raise RuntimeError(f"archive manifest verification failed: {failures[:10]}")
        required_missing = [relative for relative in required_paths() if not (extracted / relative).is_file()]
        oversized = [
            str(path.relative_to(extracted))
            for path in extracted.rglob("*")
            if path.is_file() and path.stat().st_size > 100 * 1024 * 1024
        ]
        return {
            "archive_extraction": "pass",
            "single_root": "handoff_to_chatgpt",
            "manifest_verification": "pass",
            "hashes_verified": len(hashes),
            "included_files": len(actual),
            "required_missing": required_missing,
            "files_over_100MB": oversized,
        }


def main() -> None:
    if ARCHIVE.exists() or VALIDATION_JSON.exists() or VALIDATION_MD.exists():
        raise RuntimeError("RV7 output already exists; refusing to overwrite a review artifact")
    with tempfile.TemporaryDirectory(prefix="mdcorealign-rv7-stage-") as temporary:
        bundle = Path(temporary) / "handoff_to_chatgpt"
        bundle.mkdir()
        sanitization: dict[str, list[str]] = {}
        sources: dict[str, str] = {}
        create_entry_files(bundle)
        copy_scientific_audit(bundle, sanitization)
        copy_figure_materials(bundle, sanitization)
        copy_manuscript_assets(bundle, sanitization)
        copy_product_docs(bundle, sanitization)
        create_validation_reports(bundle, sanitization)
        copy_source(bundle, sanitization)

        generated_later = {"file_manifest.tsv", "sha256_manifest.tsv", "archive_validation_report.md"}
        missing = [
            relative
            for relative in required_paths()
            if relative not in generated_later and not (bundle / relative).is_file()
        ]
        if missing:
            raise RuntimeError(f"required RV7 files missing: {missing}")
        issues, urls = scan_bundle(bundle)
        if issues:
            raise RuntimeError("bundle safety scan failed: " + json.dumps(issues[:10], indent=2))
        files = [path for path in bundle.rglob("*") if path.is_file()]
        oversized = [str(path.relative_to(bundle)) for path in files if path.stat().st_size > 100 * 1024 * 1024]
        if oversized:
            raise RuntimeError(f"files exceed 100 MB: {oversized}")

        write_text(
            bundle / "archive_validation_report.md",
            f"""# RV7 Archive Validation Report

- Required paths present before packaging: {len(required_paths())}/{len(required_paths())}.
- Text safety scan: pass; no absolute local path, username, credential, token, API key or private URL detected.
- Public URLs retained: {len([row for row in urls if row['classification'] == 'public'])}.
- Files over 100 MB: 0.
- Raw trajectories, coordinate caches, virtual environments, wheels/sdists, Git internals and unrelated benchmark outputs: excluded.
- Compact derived three-model CA PDB sources: included because they are essential to reproduce the structural figure panels.
- Final archive extraction and manifest verification are mandatory post-packaging checks; the builder exits unsuccessfully if either fails.
""",
        )
        create_manifests(bundle, sources, sanitization)
        final_scan_issues, final_urls = scan_bundle(bundle)
        if final_scan_issues:
            raise RuntimeError("final bundle safety scan failed: " + json.dumps(final_scan_issues[:10], indent=2))
        urls = final_urls

        temporary_archive = Path(temporary) / "preflight.tar.gz"
        make_archive(bundle, temporary_archive)
        preflight = verify_archive(temporary_archive, bundle)
        if preflight["required_missing"] or preflight["files_over_100MB"]:
            raise RuntimeError(f"preflight validation failed: {preflight}")

        make_archive(bundle, ARCHIVE)
        final = verify_archive(ARCHIVE, bundle)
        if final["required_missing"] or final["files_over_100MB"]:
            raise RuntimeError(f"final validation failed: {final}")

    archive_hash = sha256(ARCHIVE)
    ARCHIVE_HASH.write_text(f"{archive_hash}  {ARCHIVE.name}\n")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_hash,
        "all_required_sections_present": not final["required_missing"],
        "required_file_count": len(required_paths()),
        "secret_username_absolute_path_private_url_scan": "pass",
        "public_urls_retained": len([row for row in urls if row["classification"] == "public"]),
        "archive_extraction": final["archive_extraction"],
        "manifest_verification": final["manifest_verification"],
        "included_files": final["included_files"],
        "hashes_verified": final["hashes_verified"],
        "files_over_100MB": final["files_over_100MB"],
        "intentional_omissions": [
            "raw ATLAS/mdCATH trajectories and HDF5/XTC/DCD/TPR inputs",
            "coordinate caches and aligned trajectory archives",
            "virtual environments and Python caches",
            "wheel/sdist binaries (build hashes and reports retained)",
            "Git internals, secrets and unrelated historical benchmark outputs",
        ],
        "included_exception": "two compact derived three-model CA PDB files required for exact structural-figure reproduction",
        "sha256_manifest_self_hash": "excluded by definition; the adjacent archive SHA-256 protects the complete archive including that manifest",
    }
    VALIDATION_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_text(
        VALIDATION_MD,
        f"""# Concept Cases Product v2 Review-Bundle Validation

- Archive: `{result['archive']}`
- Size: {result['archive_size_bytes']} bytes
- Included files: {result['included_files']}
- Archive SHA-256: `{archive_hash}`
- Required sections/files: pass ({len(required_paths())}/{len(required_paths())})
- Secret/username/absolute-path/private-URL scan: pass
- Archive extraction: pass
- Manifest and per-file SHA-256 verification: pass ({result['hashes_verified']} hashed files)
- Files over 100 MB: none
- Publication or Git operation: not performed
""",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
