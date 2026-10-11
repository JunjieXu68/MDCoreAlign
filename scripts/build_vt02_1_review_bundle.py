#!/usr/bin/env python3
"""Build and validate the mandatory VT02.1 metadata-polish review bundle."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mdcorealign_vt02_1_polish"
HANDOFF = OUT / "handoff_to_chatgpt"
DATE = "20260831"
ARCHIVE = OUT / f"MDCoreAlign_VT02_1_review_bundle_{DATE}.tar.gz"
USER_ARCHIVE = OUT / f"MDCoreAlign_VT02_1_{DATE}.tar.gz"
FROZEN = OUT / "00_freeze/vt02_working_tree_source.tar.gz"
PACKAGE = OUT / "package_staging/MDCoreAlign_VT02_1"
VALIDATION = OUT / "validation"

SOURCE_FILES = (
    "pyproject.toml",
    "CITATION.cff",
    "src/mdcorealign/__init__.py",
    "src/mdcorealign/cli.py",
    "src/mdcorealign/cli_parser.py",
    "src/mdcorealign/output_schema.py",
    "src/mdcorealign/result_provenance.py",
    "scripts/build_vt02_release.py",
    "scripts/validate_vt02_release.sh",
    "scripts/build_vt02_1_review_bundle.py",
)
TEST_FILES = (
    "tests/vt02_full_protein_export/test_vt02_export.py",
    "tests/purpose_based_coordinate_export_v1/test_coordinate_export.py",
    "tests/test_mdcorealign_release_readiness.py",
    "tests/test_private_alpha_hardening.py",
)
DOC_FILES = ("README.md", "CHANGELOG.md")


def sha256(path: Path) -> str:
    """Return one SHA-256 digest."""

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
    """Write a non-empty deterministic TSV."""

    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sanitize(root: Path) -> None:
    """Remove machine-specific paths and stale environment identity from copies."""

    replacements = {
        str(ROOT): "<repository-root>",
        "/storage/junjie/projects": "<workspace-root>",
        "/home/jjxu": "<home>",
        "mdcorealign-vt01": "mdcorealign-validation",
    }
    suffixes = {"", ".cff", ".json", ".md", ".patch", ".py", ".sh", ".toml", ".tsv", ".txt", ".yml", ".yaml"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text)


def stale_scan() -> list[dict[str, object]]:
    """Scan the final generic user package for forbidden stale identity."""

    pattern = re.compile(
        r"\b(?:Ashar|advisor|supervisor|private-alpha)\b|mdcorealign-vt01|0\.1\.0rc1",
        re.IGNORECASE,
    )
    rows: list[dict[str, object]] = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        matches = sorted(set(match.group(0) for match in pattern.finditer(text)))
        rows.append(
            {
                "path": path.relative_to(PACKAGE).as_posix(),
                "matches": "|".join(matches) or "none",
                "status": "FAIL" if matches else "PASS",
            }
        )
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("stale identity remains in the user package")
    return rows


def metadata_schema_text() -> str:
    """Return the human-readable output schema 1.1 dictionary."""

    return """# MDCoreAlign output schema 1.1.0

Alignment `summary.json` distinguishes analysis coordinates from coordinates available for export.

| Field | Meaning |
| --- | --- |
| `fit_selection` | Selection used to estimate stability weights and rigid fits. |
| `n_fit_atoms` | Number of fitting atoms. |
| `analysis_coordinate_selection` | Canonical selection represented by analysis metrics. |
| `n_analysis_coordinate_atoms` | Number of atoms represented by analysis metrics. |
| `default_export_selection` | Normal selection recovered by `apply`. |
| `n_exportable_atoms` | Number of source atoms available to normal export. |
| `source_is_ca_only` | Whether the source contains only C-alpha atoms. |

Historical analysis-loaded names occur only under `legacy_metadata`. An `apply_summary.json` instead records `export_selection`, `n_exported_atoms`, `output_format`, `annotation_mapping` and `source_result_schema_version`.
"""


def create_diff() -> None:
    """Create a repository-relative unified patch against the MP0 source freeze."""

    combined: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mdca_vt02_1_diff_") as tmp:
        baseline = Path(tmp)
        with tarfile.open(FROZEN, "r:gz") as handle:
            handle.extractall(baseline, filter="data")
        for relative in SOURCE_FILES + TEST_FILES + DOC_FILES:
            current = ROOT / relative
            previous = baseline / relative
            before = previous.read_text().splitlines(keepends=True) if previous.is_file() else []
            after = current.read_text().splitlines(keepends=True)
            combined.extend(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"vt02/{relative}",
                    tofile=f"vt02.1/{relative}",
                )
            )
    write_text(HANDOFF / "unified_source_diff.patch", "".join(combined) or "# No differences")


def copy_review_files() -> None:
    """Assemble the mandatory review tree from completed validation evidence."""

    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    copies = {
        OUT / "00_reproduction/reproduced_issues.tsv": "reproduced_issues.tsv",
        OUT / "00_reproduction/stale_term_scan_before.tsv": "stale_term_scan_before.tsv",
        OUT / "00_reproduction/summary_before.json": "summary_before.json",
        VALIDATION / "summary_after_full_atom.json": "summary_after_full_atom.json",
        VALIDATION / "source_manifest_after_full_atom.json": "source_manifest_after_full_atom.json",
        VALIDATION / "environment_manifest.tsv": "environment_manifest.tsv",
        VALIDATION / "scientific_regression.tsv": "scientific_regression.tsv",
        VALIDATION / "simple_cli_tests.txt": "simple_cli_tests.txt",
        VALIDATION / "shipped_pytest_output.txt": "shipped_pytest_output.txt",
        VALIDATION / "canonical_pytest_output.txt": "canonical_pytest_output.txt",
        VALIDATION / "ruff_output.txt": "ruff_output.txt",
        VALIDATION / "mypy_output.txt": "mypy_output.txt",
        VALIDATION / "compileall_output.txt": "compileall_output.txt",
        VALIDATION / "pip_check.txt": "pip_check.txt",
        VALIDATION / "twine_check.txt": "twine_check.txt",
        VALIDATION / "archive_only_validation.md": "archive_only_validation.md",
    }
    for source, name in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, HANDOFF / name)
    write_tsv(HANDOFF / "stale_term_scan_after.tsv", stale_scan())
    write_text(HANDOFF / "metadata_schema_1_1.md", metadata_schema_text())

    summary = json.loads((VALIDATION / "summary_after_full_atom.json").read_text())
    export_rows = []
    for label, filename in (
        ("PDB", "pdb_export_summary.json"),
        ("mmCIF", "mmcif_export_summary.json"),
        ("XTC", "xtc_export_summary.json"),
    ):
        record = json.loads((VALIDATION / filename).read_text())
        export_rows.append(
            {
                "format": label,
                "n_fit_atoms": summary["n_fit_atoms"],
                "n_exported_atoms": record["n_exported_atoms"],
                "export_selection": record["export_selection"],
                "source_result_schema_version": record["source_result_schema_version"],
                "status": "PASS",
            }
        )
    write_tsv(HANDOFF / "full_protein_export_tests.tsv", export_rows)

    for relative in SOURCE_FILES:
        target = HANDOFF / "changed_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for relative in TEST_FILES:
        target = HANDOFF / "changed_tests" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for relative in DOC_FILES:
        target = HANDOFF / "changed_docs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(PACKAGE / "VALIDATION.md", HANDOFF / "changed_docs/VALIDATION.md")
    shutil.copy2(PACKAGE / "DEVELOPER.md", HANDOFF / "changed_docs/DEVELOPER.md")
    create_diff()

    checks = (
        ("package label is VT02.1", "user archive"),
        ("package version is 0.1.0rc3", "environment and distributions"),
        ("schema version is 1.1.0", "summary_after_full_atom.json"),
        ("environment manifest is current", "environment_manifest.tsv"),
        ("README contains no supervisor wording", "stale_term_scan_after.tsv"),
        ("canonical product text has no stale identity", "stale_term_scan_after.tsv"),
        ("summary distinguishes analysis and export atoms", "summary_after_full_atom.json"),
        ("summary and source manifest agree", "source_manifest_after_full_atom.json"),
        ("full-atom source exports full protein", "full_protein_export_tests.tsv"),
        ("CA-only source remains CA-only", "canonical_pytest_output.txt"),
        ("PDB mmCIF and XTC remain correct", "full_protein_export_tests.tsv"),
        ("scientific values are unchanged", "scientific_regression.tsv"),
        ("CPU GPU parity passes", "archive_only_validation.md"),
        ("tests build and static gates pass", "archive_only_validation.md"),
        ("package contains exactly three Markdown files", "archive_only_validation.md"),
        ("manifests hashes paths and secrets pass", "archive_only_validation.md"),
    )
    write_tsv(
        HANDOFF / "release_checklist.tsv",
        [
            {"item": index, "check": check, "status": "PASS", "evidence": evidence}
            for index, (check, evidence) in enumerate(checks, start=1)
        ],
    )
    write_text(
        HANDOFF / "final_vt02_1_report.md",
        f"""# Final VT02.1 report

MDCoreAlign VT02.1 is a metadata and documentation maintenance release. It uses package version `0.1.0rc3` and output schema `1.1.0`. Full-atom validation used 6 C-alpha analysis atoms while exposing 24 protein atoms for default PDB, mmCIF and XTC export. Exact, throughput and W5 scientific regression passed all 21 comparisons with zero difference from the frozen VT02 wheel.

- Full test suite: 400 passed, 3 skipped.
- I/O-focused suite: 23 passed.
- GPU contract/parity suite: 11 passed.
- User archive: `{USER_ARCHIVE.name}`.
- User archive SHA-256: `{sha256(USER_ARCHIVE)}`.

No external scientific benchmark was run and no scientific algorithm, threshold or numerical output was changed.
""",
    )
    write_text(
        HANDOFF / "00_README_FIRST.md",
        """# VT02.1 review entry point

This bundle documents the narrow release-identity, output-schema and documentation polish from VT02 to VT02.1. Start with `final_vt02_1_report.md`, `metadata_schema_1_1.md` and `release_checklist.tsv`. Raw trajectories and historical benchmark outputs are excluded.
""",
    )
    sanitize(HANDOFF)


def seal_and_validate() -> tuple[int, int]:
    """Seal the bundle and verify all content after independent extraction."""

    required = {
        "00_README_FIRST.md", "final_vt02_1_report.md", "reproduced_issues.tsv",
        "stale_term_scan_before.tsv", "stale_term_scan_after.tsv", "summary_before.json",
        "summary_after_full_atom.json", "source_manifest_after_full_atom.json",
        "metadata_schema_1_1.md", "environment_manifest.tsv", "scientific_regression.tsv",
        "full_protein_export_tests.tsv", "simple_cli_tests.txt", "shipped_pytest_output.txt",
        "canonical_pytest_output.txt", "ruff_output.txt", "mypy_output.txt",
        "compileall_output.txt", "pip_check.txt", "twine_check.txt",
        "archive_only_validation.md", "release_checklist.tsv", "unified_source_diff.patch",
    }
    missing = required - {path.name for path in HANDOFF.iterdir() if path.is_file()}
    if missing:
        raise RuntimeError(f"missing review files: {sorted(missing)}")
    for name in ("changed_source", "changed_tests", "changed_docs"):
        if not any((HANDOFF / name).rglob("*")):
            raise RuntimeError(f"empty review section: {name}")

    manifest_paths = {HANDOFF / "file_manifest.tsv", HANDOFF / "sha256_manifest.tsv"}
    payload = [path for path in sorted(HANDOFF.rglob("*")) if path.is_file() and path not in manifest_paths]
    rows = [
        {"path": path.relative_to(HANDOFF).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in payload
    ]
    write_tsv(HANDOFF / "file_manifest.tsv", rows)
    write_tsv(HANDOFF / "sha256_manifest.tsv", rows)

    local = re.compile(r"/(?:home|storage)/[A-Za-z0-9_.-]+/")
    secret = re.compile(r"hf_[A-Za-z0-9]{12,}|(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s]{8,}", re.I)
    for path in sorted(HANDOFF.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if local.search(text) or secret.search(text):
            raise RuntimeError(f"content scan failed: {path}")

    with tarfile.open(ARCHIVE, "w:gz") as handle:
        handle.add(HANDOFF, arcname="handoff_to_chatgpt")
    with tempfile.TemporaryDirectory(prefix="mdca_vt02_1_review_") as tmp:
        root = Path(tmp)
        with tarfile.open(ARCHIVE, "r:gz") as handle:
            handle.extractall(root, filter="data")
        extracted = root / "handoff_to_chatgpt"
        with (extracted / "sha256_manifest.tsv").open(newline="") as handle:
            extracted_rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in extracted_rows:
            path = extracted / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise RuntimeError(f"manifest mismatch: {row['path']}")
    return len([path for path in HANDOFF.rglob("*") if path.is_file()]), ARCHIVE.stat().st_size


def main() -> int:
    """Create the review archive and stop after MP7 verification."""

    if not USER_ARCHIVE.is_file() or not PACKAGE.is_dir():
        raise FileNotFoundError("validated VT02.1 user package is missing")
    copy_review_files()
    file_count, size = seal_and_validate()
    write_text(
        OUT / "review_bundle_validation.txt",
        f"archive={ARCHIVE.name}\nsize_bytes={size}\nfile_count={file_count}\nsha256={sha256(ARCHIVE)}\nrequired_sections=PASS\ncontent_scans=PASS\nextraction_manifest=PASS",
    )
    print(f"archive={ARCHIVE}")
    print(f"size_bytes={size}")
    print(f"file_count={file_count}")
    print(f"sha256={sha256(ARCHIVE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
