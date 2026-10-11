#!/usr/bin/env python3
"""Build and independently validate the VT02.2 pip-first review bundle."""

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
OUT = ROOT / "results/mdcorealign_vt02_2_pip_first"
HANDOFF = OUT / "handoff_to_chatgpt"
DATE = "20260901"
ARCHIVE = OUT / f"MDCoreAlign_VT02_2_review_bundle_{DATE}.tar.gz"
USER_ARCHIVE = OUT / "06_release" / f"MDCoreAlign_VT02_2_{DATE}.tar.gz"
FROZEN = OUT / "00_freeze/vt02_1_working_tree_source.tar.gz"

SOURCE_FILES = (
    "pyproject.toml",
    "CITATION.cff",
    "CHANGELOG.md",
    "src/mdcorealign/__init__.py",
    "src/mdcorealign/cli.py",
    "src/mdcorealign/cli_parser.py",
    "scripts/build_vt02_release.py",
    "scripts/validate_vt02_2_release.sh",
    "scripts/build_vt02_2_review_bundle.py",
)
TEST_FILES = (
    "tests/vt02_2_pip_first/test_dependency_metadata.py",
    "tests/vt02_full_protein_export/test_vt02_export.py",
    "tests/single_full_product_v5/test_single_product_contract.py",
    "tests/test_mdcorealign_release_readiness.py",
)
DOC_FILES = ("README.md",)


def sha256(path: Path) -> str:
    """Return one SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    """Write normalized text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a deterministic non-empty TSV."""

    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source: Path, destination: Path) -> None:
    """Copy one required file."""

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sanitize(root: Path) -> None:
    """Remove workstation-specific absolute paths from copied evidence."""

    replacements = {
        str(ROOT): "<repository-root>",
        "/storage/junjie/projects": "<workspace-root>",
        "/home/jjxu": "<home>",
    }
    temporary = re.compile(r"/tmp/(?:mdca|mdcorealign)[A-Za-z0-9_.\-/]*")
    suffixes = {
        "", ".cff", ".json", ".md", ".patch", ".py", ".sh", ".toml",
        ".tsv", ".txt", ".yml", ".yaml",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        text = temporary.sub("<temporary-path>", text)
        path.write_text(text)


def create_diff() -> None:
    """Create a repository-relative patch against the frozen VT02.1 snapshot."""

    combined: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mdca_vt02_2_diff_") as tmp:
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
                    fromfile=f"VT02.1/{relative}",
                    tofile=f"VT02.2/{relative}",
                )
            )
    write_text(HANDOFF / "unified_source_diff.patch", "".join(combined) or "# No differences")


def assemble() -> None:
    """Assemble the required PI10 handoff tree."""

    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    copies = {
        OUT / "05_docs/final_pip_first_release_report.md": "final_pip_first_release_report.md",
        OUT / "02_packaging/installation_contract.md": "installation_contract.md",
        OUT / "02_packaging/dependency_matrix.tsv": "dependency_matrix.tsv",
        OUT / "02_packaging/support_matrix.tsv": "support_matrix.tsv",
        OUT / "02_packaging/wheel_metadata.txt": "wheel_metadata.txt",
        OUT / "01_reproduction/reproduced_installation_issues.tsv": "reproduced_installation_issues.tsv",
        OUT / "03_pip_validation/python311_clean_install.txt": "python311_clean_install.txt",
        OUT / "03_pip_validation/python312_clean_install.txt": "python312_clean_install.txt",
        OUT / "03_pip_validation/no_conda_install_proof.txt": "no_conda_install_proof.txt",
        OUT / "03_pip_validation/sdist_clean_install.txt": "sdist_clean_install.txt",
        OUT / "03_pip_validation/pip_gpu_install.txt": "pip_gpu_install.txt",
        OUT / "03_pip_validation/cupy_inventory.tsv": "cupy_inventory.tsv",
        OUT / "03_pip_validation/gpu_kernel_probe.txt": "gpu_kernel_probe.txt",
        OUT / "03_pip_validation/mdcorealign_gpu_smoke.txt": "mdcorealign_gpu_smoke.txt",
        OUT / "03_pip_validation/cpu_gpu_parity.tsv": "cpu_gpu_parity.tsv",
        OUT / "04_conda_validation/environment.yml": "final_environment.yml",
        OUT / "04_conda_validation/conda_create_output.txt": "conda_create_output.txt",
        OUT / "04_conda_validation/conda_lock_recreation.txt": "conda_lock_recreation.txt",
        OUT / "04_conda_validation/final_archive_conda_validation.txt": "final_archive_conda_validation.txt",
        OUT / "03_pip_validation/install_script_tests.tsv": "install_script_tests.tsv",
        OUT / "03_pip_validation/scientific_regression.tsv": "scientific_regression.tsv",
        OUT / "03_pip_validation/shipped_pytest_output.txt": "shipped_pytest_output.txt",
        OUT / "03_pip_validation/canonical_pytest_output.txt": "canonical_pytest_output.txt",
        OUT / "05_docs/documented_command_audit.tsv": "documented_command_audit.tsv",
        OUT / "05_docs/release_checklist.tsv": "release_checklist.tsv",
        OUT / "03_pip_validation/archive_only_validation.md": "archive_only_validation.md",
        OUT / "03_pip_validation/io_focused_pytest_output.txt": "io_focused_pytest_output.txt",
        OUT / "03_pip_validation/ruff_output.txt": "ruff_output.txt",
        OUT / "03_pip_validation/mypy_output.txt": "mypy_output.txt",
        OUT / "03_pip_validation/compileall_output.txt": "compileall_output.txt",
        OUT / "03_pip_validation/twine_check.txt": "twine_check.txt",
        OUT / "03_pip_validation/pip_check.txt": "pip_check.txt",
        OUT / "03_pip_validation/cpu_io_smoke.txt": "cpu_io_smoke.txt",
        OUT / "03_pip_validation/path_secret_scan.tsv": "path_secret_scan.tsv",
        OUT / "03_pip_validation/language_scan.tsv": "language_scan.tsv",
        OUT / "03_pip_validation/source_hash_parity.tsv": "source_hash_parity.tsv",
        OUT / "05_docs/ruff_scope_note.md": "ruff_scope_note.md",
    }
    for source, name in copies.items():
        copy_file(source, HANDOFF / name)

    for relative in SOURCE_FILES:
        copy_file(ROOT / relative, HANDOFF / "changed_source" / relative)
    for relative in TEST_FILES:
        copy_file(ROOT / relative, HANDOFF / "changed_tests" / relative)
    for relative in DOC_FILES:
        copy_file(ROOT / relative, HANDOFF / "changed_docs" / relative)
    package = OUT / "02_packaging/package_staging/MDCoreAlign_VT02_2"
    for name in ("README.md", "VALIDATION.md", "DEVELOPER.md", "environment.yml", "install.sh"):
        copy_file(package / name, HANDOFF / "changed_docs" / f"sealed_{name}")
    create_diff()

    write_text(
        HANDOFF / "00_README_FIRST.md",
        """# MDCoreAlign VT02.2 review entry point

This bundle records the PI0-PI10 pip-first release hardening. Start with
`final_pip_first_release_report.md`, `installation_contract.md`,
`release_checklist.tsv` and `archive_only_validation.md`. The final decision is
`USER_INSTALL_READY`. No scientific benchmark was run and no scientific algorithm,
threshold or numerical output changed.
""",
    )
    sanitize(HANDOFF)


def seal_and_validate() -> tuple[int, int]:
    """Seal, scan, extract and verify the review archive."""

    required = {
        "00_README_FIRST.md", "final_pip_first_release_report.md",
        "installation_contract.md", "dependency_matrix.tsv", "support_matrix.tsv",
        "wheel_metadata.txt", "reproduced_installation_issues.tsv",
        "python311_clean_install.txt", "python312_clean_install.txt",
        "no_conda_install_proof.txt", "sdist_clean_install.txt", "pip_gpu_install.txt",
        "cupy_inventory.tsv", "gpu_kernel_probe.txt", "mdcorealign_gpu_smoke.txt",
        "cpu_gpu_parity.tsv", "final_environment.yml", "conda_create_output.txt",
        "conda_lock_recreation.txt", "install_script_tests.tsv",
        "scientific_regression.tsv", "shipped_pytest_output.txt",
        "canonical_pytest_output.txt", "documented_command_audit.tsv",
        "release_checklist.tsv", "archive_only_validation.md", "unified_source_diff.patch",
    }
    observed = {path.name for path in HANDOFF.iterdir() if path.is_file()}
    missing = required - observed
    if missing:
        raise RuntimeError(f"missing review files: {sorted(missing)}")
    for name in ("changed_source", "changed_tests", "changed_docs"):
        if not any(path.is_file() for path in (HANDOFF / name).rglob("*")):
            raise RuntimeError(f"empty review section: {name}")

    checklist = list(csv.DictReader((HANDOFF / "release_checklist.tsv").open(), delimiter="\t"))
    if len(checklist) != 22 or any(row["status"] != "PASS" for row in checklist):
        raise RuntimeError("the 22-item release checklist is not fully passing")

    manifests = {HANDOFF / "file_manifest.tsv", HANDOFF / "sha256_manifest.tsv"}
    payload = [path for path in sorted(HANDOFF.rglob("*")) if path.is_file() and path not in manifests]
    rows = [
        {
            "path": path.relative_to(HANDOFF).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payload
    ]
    write_tsv(HANDOFF / "file_manifest.tsv", rows)
    write_tsv(HANDOFF / "sha256_manifest.tsv", rows)

    local = re.compile(r"/(?:home|storage)/[A-Za-z0-9_.-]+/|/tmp/(?:mdca|mdcorealign)")
    secret = re.compile(
        r"hf_[A-Za-z0-9]{12,}|(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s]{8,}",
        re.I,
    )
    for path in sorted(HANDOFF.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if local.search(text) or secret.search(text):
            raise RuntimeError(f"review content scan failed: {path}")
        if path.stat().st_size > 100 * 1024 * 1024:
            raise RuntimeError(f"oversized review file: {path}")

    with tarfile.open(ARCHIVE, "w:gz") as handle:
        handle.add(HANDOFF, arcname="handoff_to_chatgpt")
    with tempfile.TemporaryDirectory(prefix="mdca_vt02_2_review_") as tmp:
        root = Path(tmp)
        with tarfile.open(ARCHIVE, "r:gz") as handle:
            handle.extractall(root, filter="data")
        extracted = root / "handoff_to_chatgpt"
        with (extracted / "sha256_manifest.tsv").open(newline="") as handle:
            extracted_rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in extracted_rows:
            path = extracted / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise RuntimeError(f"review manifest mismatch: {row['path']}")
    return len([path for path in HANDOFF.rglob("*") if path.is_file()]), ARCHIVE.stat().st_size


def main() -> int:
    """Build the review archive and stop after PI10."""

    if not USER_ARCHIVE.is_file() or not FROZEN.is_file():
        raise FileNotFoundError("final user archive or frozen baseline is missing")
    assemble()
    count, size = seal_and_validate()
    write_text(
        OUT / "review_bundle_validation.txt",
        f"""archive={ARCHIVE.name}
size_bytes={size}
file_count={count}
sha256={sha256(ARCHIVE)}
required_sections=PASS
content_scans=PASS
extraction_manifest=PASS
release_checklist=22/22 PASS
decision=USER_INSTALL_READY
""",
    )
    print(f"archive={ARCHIVE}")
    print(f"size_bytes={size}")
    print(f"file_count={count}")
    print(f"sha256={sha256(ARCHIVE)}")
    print("decision=USER_INSTALL_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
