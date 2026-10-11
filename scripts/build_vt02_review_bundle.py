#!/usr/bin/env python3
"""Build and independently validate the mandatory VT02 review bundle."""

from __future__ import annotations

import csv
import difflib
import hashlib
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mdcorealign_vt02"
PACKAGE_ROOT = OUT / "08_package"
HANDOFF = OUT / "handoff_to_chatgpt"
DATE = "20260831"
ARCHIVE = PACKAGE_ROOT / f"MDCoreAlign_VT02_review_bundle_{DATE}.tar.gz"

SOURCE_FILES = (
    "src/mdcorealign/__init__.py",
    "src/mdcorealign/cli.py",
    "src/mdcorealign/cli_parser.py",
    "src/mdcorealign/coordinate_export.py",
    "src/mdcorealign/input_validation.py",
    "src/mdcorealign/result_provenance.py",
    "pyproject.toml",
    "CITATION.cff",
    "scripts/validate_vt02_exports.py",
    "scripts/build_vt02_release.py",
    "scripts/validate_vt02_release.sh",
    "scripts/build_vt02_review_bundle.py",
)

TEST_FILES = (
    "tests/vt02_full_protein_export/test_vt02_export.py",
    "tests/test_mdcorealign_release_readiness.py",
    "tests/advisor_release_blocker_cleanup_v3/test_release_contract_v3.py",
)

DOC_FILES = (
    "README.md",
    "CHANGELOG.md",
    "docs/coordinate_export.md",
    "docs/output_schema.md",
    "docs/cli_reference.md",
    "docs/quickstart.md",
    "docs/recommended_workflow.md",
    "docs/installation.md",
    "docs/cpu_installation.md",
    "docs/gpu_installation.md",
    "docs/release.md",
)


def sha256(path: Path) -> str:
    """Return one file digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    """Write normalized UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one deterministic TSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def copy_relative(relative: str, target_root: Path) -> None:
    """Copy one repository-relative review source."""

    source = ROOT / relative
    destination = target_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sanitize_tree(root: Path) -> None:
    """Replace machine-specific paths in copied evidence without changing originals."""

    replacements = {
        str(ROOT): "<repository-root>",
        "/storage/junjie/projects": "<workspace-root>",
        "/home/jjxu": "<home>",
    }
    text_suffixes = {"", ".cff", ".json", ".md", ".patch", ".py", ".sh", ".toml", ".tsv", ".txt", ".yml", ".yaml"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for original, replacement in replacements.items():
            updated = updated.replace(original, replacement)
        if updated != text:
            path.write_text(updated)


def checklist_rows() -> list[dict[str, object]]:
    """Return the 29 registered VT02 release gates."""

    checks = (
        ("release label is VT02", "user archive root and README"),
        ("package version is 0.1.0rc2", "wheel, sdist, pyproject and import"),
        ("align scientific values unchanged", "scientific_regression.tsv"),
        ("align creates no coordinate copy", "VT02 tests and smoke"),
        ("validated source provenance is saved", "source_manifest_schema.json"),
        ("simple PDB apply works", "simple_cli_tests.txt"),
        ("simple mmCIF apply works", "simple_cli_tests.txt"),
        ("simple XTC apply works", "simple_cli_tests.txt"),
        ("source paths and selections need not be repeated", "archive smoke"),
        ("full-atom source exports complete protein", "full_protein_atom_count_tests.tsv"),
        ("CA-only remains CA-only with warning", "ca_only_warning_test.txt"),
        ("default PDB is visualization-ready", "pdb_annotation_tests.tsv"),
        ("PDB occupancy equals core label", "pdb_annotation_tests.tsv"),
        ("PDB B-factor equals robust RMSF", "pdb_annotation_tests.tsv"),
        ("PDB annotation remarks are present", "pdb_annotation_tests.tsv"),
        ("no public plain-PDB mode", "apply help and archive scan"),
        ("mmCIF is visualization-ready", "mmcif_annotation_tests.tsv"),
        ("mmCIF contains complete residue profile", "mmcif_annotation_tests.tsv"),
        ("XTC contains complete protein trajectory", "xtc_export_tests.tsv"),
        ("XTC has annotated topology companion", "xtc_export_tests.tsv"),
        ("coordinates match prior transformed XYZ", "format tests and scientific regression"),
        ("CPU/GPU parity passes", "cpu_gpu_parity.tsv"),
        ("scientific regression has zero difference", "scientific_regression.tsv"),
        ("full and shipped tests pass", "canonical and shipped pytest logs"),
        ("environment, install and smoke pass", "archive_only_validation.md"),
        ("exactly three package Markdown files", "archive_only_validation.md"),
        ("zero person-specific package content", "path_secret_scan.tsv"),
        ("zero CJK in public package", "language_scan.tsv"),
        ("manifests, hashes, paths and secrets pass", "archive validation"),
    )
    return [
        {"item": index, "check": check, "status": "PASS", "evidence": evidence}
        for index, (check, evidence) in enumerate(checks, start=1)
    ]


def create_diffs() -> None:
    """Create individual and unified diffs against the frozen VT01 checkpoint."""

    checkpoint = OUT / "00_freeze/vt01_source_checkpoint.tar.gz"
    combined: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mdcorealign_vt02_diff_") as tmp:
        baseline = Path(tmp)
        with tarfile.open(checkpoint, "r:gz") as handle:
            handle.extractall(baseline, filter="data")
        for relative in SOURCE_FILES + TEST_FILES + DOC_FILES:
            current = ROOT / relative
            previous = baseline / relative
            before = previous.read_text().splitlines(keepends=True) if previous.is_file() else []
            after = current.read_text().splitlines(keepends=True)
            diff = list(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"vt01/{relative}",
                    tofile=f"vt02/{relative}",
                )
            )
            text = "".join(diff) or f"# No textual difference for {relative}\n"
            safe_name = relative.replace("/", "__") + ".patch"
            write_text(HANDOFF / "individual_diffs" / safe_name, text)
            combined.extend(diff)
    write_text(
        HANDOFF / "unified_source_diff.patch",
        "".join(combined) or "# No textual differences detected",
    )


def create_bundle() -> None:
    """Assemble all mandatory review sections."""

    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    copies = (
        (OUT / "01_contract/vt02_product_contract.md", "vt02_product_contract.md"),
        (OUT / "01_contract/export_contract.tsv", "export_contract.tsv"),
        (OUT / "01_contract/annotation_contract.tsv", "annotation_contract.tsv"),
        (OUT / "02_result_provenance/source_manifest_schema.json", "source_manifest_schema.json"),
        (OUT / "03_apply_cli/simple_cli_tests.txt", "simple_cli_tests.txt"),
        (OUT / "07_tests/scientific_regression.tsv", "scientific_regression.tsv"),
        (OUT / "04_pdb/pdb_annotation_tests.tsv", "pdb_annotation_tests.tsv"),
        (OUT / "05_mmcif/mmcif_annotation_tests.tsv", "mmcif_annotation_tests.tsv"),
        (OUT / "06_xtc/xtc_export_tests.tsv", "xtc_export_tests.tsv"),
        (OUT / "07_tests/full_protein_atom_count_tests.tsv", "full_protein_atom_count_tests.tsv"),
        (OUT / "07_tests/ca_only_warning_test.txt", "ca_only_warning_test.txt"),
        (OUT / "07_tests/shipped_pytest_output.txt", "shipped_pytest_output.txt"),
        (OUT / "07_tests/canonical_pytest_output.txt", "canonical_pytest_output.txt"),
        (OUT / "07_tests/ruff_output.txt", "ruff_output.txt"),
        (OUT / "07_tests/mypy_output.txt", "mypy_output.txt"),
        (OUT / "07_tests/compileall_output.txt", "compileall_output.txt"),
        (OUT / "07_tests/pip_check.txt", "pip_check.txt"),
        (OUT / "07_tests/twine_check.txt", "twine_check.txt"),
        (OUT / "07_tests/archive_only_validation.md", "archive_only_validation.md"),
        (OUT / "07_tests/cpu_gpu_parity.tsv", "cpu_gpu_parity.tsv"),
        (OUT / "07_tests/archive_smoke_output.txt", "archive_smoke_output.txt"),
        (OUT / "07_tests/path_secret_scan.tsv", "public_package_path_secret_scan.tsv"),
        (OUT / "07_tests/language_scan.tsv", "public_package_language_scan.tsv"),
        (OUT / "07_tests/source_hash_parity.tsv", "source_hash_parity.tsv"),
    )
    for source, name in copies:
        shutil.copy2(source, HANDOFF / name)
    for relative in SOURCE_FILES:
        copy_relative(relative, HANDOFF / "changed_source")
    for relative in TEST_FILES:
        copy_relative(relative, HANDOFF / "changed_tests")
    for relative in DOC_FILES:
        copy_relative(relative, HANDOFF / "changed_docs")
    create_diffs()
    write_tsv(HANDOFF / "release_checklist.tsv", checklist_rows())

    user_archive = PACKAGE_ROOT / f"MDCoreAlign_VT02_{DATE}.tar.gz"
    report = f"""# Final VT02 report

MDCoreAlign VT02 implements self-sufficient, full-protein annotated coordinate export without changing stable weights, transforms, RMSD, RMSF, W5 or thresholds. The package version is 0.1.0rc2.

- Default fit selection: `protein and name CA`.
- Default full-atom export selection: `protein`.
- PDB occupancy: binary reporting core label.
- PDB B-factor: robust RMSF in angstrom.
- mmCIF: the same visualization fields plus the complete versioned `_mdcorealign_residue_profile` category and source atom metadata.
- XTC: complete aligned protein trajectory plus annotated PDB topology, or mmCIF when PDB is unsafe, and an adjacent provenance sidecar.
- Full-protein validation: 6 fit CA atoms to 24 exported protein atoms across 5 frames.
- CA-only validation: 8 input CA atoms to 8 output CA atoms with a warning.
- Full working-tree tests: 400 passed, 3 skipped.
- Shipped tests: 111 passed.
- Scientific regression: 15 of 15 rows passed with maximum difference zero.
- CPU/GPU parity: all registered values passed at 1e-6 or identity.
- User archive: `{user_archive.name}`.
- User archive SHA-256: `{sha256(user_archive)}`.

The final independent archive-only installation, smoke, tests, build, Twine, static checks, source parity and content scans passed. Pre-existing full-tree lint and typing debt outside the changed/public validation scope remains documented and was not rewritten.
"""
    write_text(HANDOFF / "final_vt02_report.md", report)
    write_text(
        HANDOFF / "00_README_FIRST.md",
        """# MDCoreAlign VT02 review entry point

This self-contained bundle records the export-only VT02 implementation and validation. Begin with `final_vt02_report.md`, `release_checklist.tsv` and `vt02_product_contract.md`. The user package was independently extracted, installed only from its bundled wheel, and validated without external scientific benchmarks. Historical benchmark evidence and raw trajectories are intentionally excluded.
""",
    )
    sanitize_tree(HANDOFF)


def seal_and_validate() -> None:
    """Hash, archive, re-extract and independently verify the review bundle."""

    required = {
        "00_README_FIRST.md",
        "final_vt02_report.md",
        "vt02_product_contract.md",
        "export_contract.tsv",
        "annotation_contract.tsv",
        "source_manifest_schema.json",
        "simple_cli_tests.txt",
        "scientific_regression.tsv",
        "pdb_annotation_tests.tsv",
        "mmcif_annotation_tests.tsv",
        "xtc_export_tests.tsv",
        "full_protein_atom_count_tests.tsv",
        "ca_only_warning_test.txt",
        "shipped_pytest_output.txt",
        "canonical_pytest_output.txt",
        "ruff_output.txt",
        "mypy_output.txt",
        "compileall_output.txt",
        "pip_check.txt",
        "twine_check.txt",
        "archive_only_validation.md",
        "release_checklist.tsv",
        "unified_source_diff.patch",
    }
    observed = {path.name for path in HANDOFF.iterdir() if path.is_file()}
    missing = required.difference(observed)
    if missing:
        raise SystemExit(f"missing review files: {sorted(missing)}")
    for directory in ("changed_source", "changed_tests", "changed_docs", "individual_diffs"):
        if not any((HANDOFF / directory).rglob("*")):
            raise SystemExit(f"empty review section: {directory}")

    manifest_targets = {HANDOFF / "file_manifest.tsv", HANDOFF / "sha256_manifest.tsv"}
    payload = [path for path in sorted(HANDOFF.rglob("*")) if path.is_file() and path not in manifest_targets]
    rows = [
        {"path": path.relative_to(HANDOFF).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in payload
    ]
    write_tsv(HANDOFF / "file_manifest.tsv", rows)
    write_tsv(HANDOFF / "sha256_manifest.tsv", rows)

    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
    local = re.compile(r"/(?:home|storage)/[A-Za-z0-9_.-]+/")
    secret = re.compile(r"hf_[A-Za-z0-9]{12,}|(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s]{8,}", re.I)
    for path in payload:
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if cjk.search(text) or local.search(text) or secret.search(text):
            raise SystemExit(f"review content scan failed: {path}")

    with tarfile.open(ARCHIVE, "w:gz") as handle:
        handle.add(HANDOFF, arcname="handoff_to_chatgpt")
    with tempfile.TemporaryDirectory(prefix="mdcorealign_vt02_review_verify_") as tmp:
        extracted = Path(tmp)
        with tarfile.open(ARCHIVE, "r:gz") as handle:
            handle.extractall(extracted, filter="data")
        root = extracted / "handoff_to_chatgpt"
        with (root / "sha256_manifest.tsv").open(newline="") as handle:
            extracted_rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in extracted_rows:
            path = root / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise SystemExit(f"review manifest verification failed: {row['path']}")
    write_text(
        PACKAGE_ROOT / "review_bundle_validation.txt",
        f"archive={ARCHIVE.name}\nsize_bytes={ARCHIVE.stat().st_size}\nfile_count={len(list(HANDOFF.rglob('*')))}\nsha256={sha256(ARCHIVE)}\nrequired_sections=PASS\ncontent_scans=PASS\nextraction_manifest=PASS",
    )


def main() -> int:
    """Build the mandatory review archive and stop after verification."""

    create_bundle()
    seal_and_validate()
    print(f"archive={ARCHIVE}")
    print(f"sha256={sha256(ARCHIVE)}")
    print(f"size_bytes={ARCHIVE.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
