"""Create private-alpha advisor bundle artifacts.

The script keeps the earlier private-alpha checklist outputs for compatibility
and also creates a distributable advisor bundle under ``dist_advisor/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "private_alpha_hardening"
BUNDLE_DIR = OUT / "private_alpha_bundle"
ADVISOR_RESULTS = ROOT / "results" / "advisor_bundle"
DIST_ADVISOR = ROOT / "dist_advisor"
PRIVATE_ALPHA_LABEL = "v0.1.0-alpha-advisor"
ARCHIVE_BASENAME = "MDCoreAlign_private_alpha_v0.1.0_advisor"
MAX_FILE_MB = float(os.environ.get("MDCOREALIGN_BUNDLE_MAX_FILE_MB", "25"))

BUNDLE_FIELDS = ["category", "path", "required", "status", "notes"]
ADVISOR_MANIFEST_FIELDS = ["path", "size_bytes", "category", "included", "notes"]
CHECKSUM_FIELDS = ["path", "sha256", "size_bytes"]
FEEDBACK_FIELDS = [
    "tester_id",
    "date",
    "platform",
    "python_version",
    "install_status",
    "toy_quickstart_status",
    "real_md_small_status",
    "unclear_error_messages",
    "unclear_outputs",
    "documentation_gaps",
    "time_to_first_success_minutes",
    "overall_usability_score_1_to_5",
    "blocking_issue",
    "free_text_feedback",
]


def _write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _status(path: str) -> str:
    return "present" if (ROOT / path).exists() else "missing"


def _run(command: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    return completed.returncode == 0, (completed.stdout + completed.stderr).strip()


def _current_version() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import mdcorealign

        return str(mdcorealign.__version__)
    finally:
        if str(ROOT / "src") in sys.path:
            sys.path.remove(str(ROOT / "src"))


def _git_info() -> tuple[str, bool, list[str]]:
    ok, commit = _run(["git", "rev-parse", "--short", "HEAD"])
    if not ok:
        # An invalid HEAD makes the parent workspace appear wholly untracked;
        # avoid recursively scanning large benchmark/data trees.
        return ("unavailable", False, [])
    ok_status, status = _run(
        [
            "git",
            "status",
            "--short",
            "--",
            "README.md",
            "docs",
            "dist_advisor",
            "examples",
            "pyproject.toml",
            "scripts",
            "src",
            "tests",
            "results/advisor_bundle",
            "results/input_format_support_upgrade",
            "results/private_alpha_hardening",
        ]
    )
    files = [line.strip() for line in status.splitlines() if line.strip()] if ok_status else []
    return (commit.strip() if ok else "unavailable", bool(files), files[:300])


def build_rows() -> list[dict[str, object]]:
    """Return the files expected in the legacy private-alpha tester handoff."""

    entries = [
        ("core", "README.md", True, "user-facing overview"),
        ("core", "pyproject.toml", True, "package metadata"),
        ("docs", "docs/installation.md", True, "install instructions"),
        ("docs", "docs/quickstart.md", True, "quick start"),
        ("docs", "docs/cli_reference.md", True, "CLI reference"),
        ("docs", "docs/output_formats.md", True, "output definitions"),
        ("docs", "docs/md_preprocessing.md", True, "preprocessing caveats"),
        ("docs", "docs/support_matrix.md", True, "supported and unsupported inputs"),
        ("docs", "docs/recommended_workflow.md", True, "new trajectory workflow"),
        ("docs", "docs/troubleshooting.md", True, "actionable errors"),
        ("docs", "docs/private_alpha_advisor_test.md", True, "advisor checklist"),
        ("example", "examples/toy_md/run_quickstart.sh", True, "toy workflow"),
        ("example", "examples/toy_md/toy_ca_trajectory.pdb", True, "toy input"),
        ("example", "examples/real_md_small/run_recommended_workflow.sh", True, "small real-MD workflow"),
        ("example", "examples/real_md_small/trajectory_small.pdb", True, "small bundled trajectory subset"),
        ("script", "scripts/audit_private_alpha_readiness.py", True, "readiness audit"),
        ("script", "scripts/package_private_alpha_bundle.py", True, "this bundle manifest script"),
        ("script", "scripts/check_clean_install.sh", True, "clean install validation"),
        ("script", "scripts/build_package.sh", True, "wheel/sdist build validation"),
    ]
    rows = []
    for category, path, required, notes in entries:
        rows.append(
            {
                "category": category,
                "path": path,
                "required": str(required).lower(),
                "status": _status(path),
                "notes": notes,
            }
        )
    return rows


def _legacy_private_alpha_outputs() -> None:
    """Write the older private-alpha checklist outputs used by existing tests."""

    OUT.mkdir(parents=True, exist_ok=True)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    _write_tsv(BUNDLE_DIR / "private_alpha_bundle_manifest.tsv", rows, BUNDLE_FIELDS)
    _write_tsv(OUT / "advisor_testing_template.tsv", [], FEEDBACK_FIELDS)
    missing = [row["path"] for row in rows if row["required"] == "true" and row["status"] != "present"]
    checklist = [
        "# Private Alpha Bundle Checklist",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Required files missing: `{len(missing)}`",
        "- Large benchmark trajectories are intentionally not included.",
        "- MDLovoFit/THESEUS are optional for core commands.",
        "",
        "## Advisor Test Command",
        "",
        "```bash",
        "pip install .",
        "bash examples/toy_md/run_quickstart.sh /tmp/mdcorealign_toy_test",
        "bash examples/real_md_small/run_recommended_workflow.sh /tmp/mdcorealign_real_small_test",
        "python -m build",
        "```",
    ]
    if missing:
        checklist.extend(["", "## Missing Required Files", ""])
        checklist.extend(f"- `{path}`" for path in missing)
    (BUNDLE_DIR / "private_alpha_bundle_checklist.md").write_text("\n".join(checklist) + "\n")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bundle_manifest": str((BUNDLE_DIR / "private_alpha_bundle_manifest.tsv").relative_to(ROOT)),
                "advisor_testing_template": str((OUT / "advisor_testing_template.tsv").relative_to(ROOT)),
                "missing_required_files": missing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _copy_file(src: Path, dst: Path, manifest: list[dict[str, object]], category: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "path": str(dst.relative_to(STAGING_ROOT)),
            "size_bytes": dst.stat().st_size,
            "category": category,
            "included": "true",
            "notes": "",
        }
    )


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name
    if name in {".git", ".pytest_cache", "__pycache__", ".mypy_cache"}:
        return True
    if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in path.parts):
        return True
    if name.endswith((".pyc", ".pyo")):
        return True
    if path.suffix.lower() in {".zip", ".tar", ".gz", ".whl"}:
        return True
    if "output" in parts and "examples" in parts:
        return True
    return False


def _copy_tree(src_dir: Path, dst_dir: Path, manifest: list[dict[str, object]], category: str) -> None:
    for src in src_dir.rglob("*"):
        if src.is_dir() or _should_skip(src):
            continue
        rel = src.relative_to(src_dir)
        _copy_file(src, dst_dir / rel, manifest, category)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_version_manifest(*, build_status: str, test_result: str) -> None:
    commit, dirty, dirty_files = _git_info()
    payload = {
        "package_name": "mdcorealign",
        "import_name": "mdcorealign",
        "cli_name": "mdcorealign",
        "current_version": _current_version(),
        "private_alpha_label": PRIVATE_ALPHA_LABEL,
        "git_commit_hash": commit,
        "git_dirty_status": dirty,
        "git_dirty_files": dirty_files,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "date_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_command": "PYTHONPATH=src python -m pytest -q tests",
        "test_result": test_result,
        "build_status": build_status,
    }
    ADVISOR_RESULTS.mkdir(parents=True, exist_ok=True)
    (ADVISOR_RESULTS / "version_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    dirty_list = "\n".join(f"- `{item}`" for item in dirty_files) if dirty_files else "- none"
    md = [
        "# Version Manifest",
        "",
        f"- Package name: `mdcorealign`",
        f"- Import name: `mdcorealign`",
        f"- CLI name: `mdcorealign`",
        f"- Current version: `{payload['current_version']}`",
        f"- Private alpha label: `{PRIVATE_ALPHA_LABEL}`",
        f"- Git commit hash: `{commit}`",
        f"- Git dirty status: `{dirty}`",
        f"- Python version: `{payload['python_version']}`",
        f"- Platform: `{payload['platform']}`",
        f"- Date/time: `{payload['date_time']}`",
        f"- Test command: `{payload['test_command']}`",
        f"- Test result: `{test_result}`",
        f"- Build status: `{build_status}`",
        "",
        "## Dirty Files",
        "",
        dirty_list,
    ]
    (ADVISOR_RESULTS / "version_manifest.md").write_text("\n".join(md) + "\n")


STAGING_ROOT = DIST_ADVISOR / ARCHIVE_BASENAME


def _create_advisor_staging() -> tuple[list[dict[str, object]], list[str]]:
    """Create a clean staging directory for the advisor bundle."""

    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    warnings: list[str] = []

    root_files = [
        "README.md",
        "pyproject.toml",
        "LICENSE",
        "CITATION.cff",
    ]
    for rel in root_files:
        src = ROOT / rel
        if src.exists():
            _copy_file(src, STAGING_ROOT / rel, manifest, "root")
        else:
            warnings.append(f"missing {rel}")

    for rel in [
        "README_FOR_ADVISOR.md",
        "INSTALLATION.md",
        "QUICKSTART.md",
        "ADVISOR_TESTING_GUIDE.md",
        "FEEDBACK_FORM.tsv",
        "FEEDBACK_FORM.md",
        "OUTPUT_INTERPRETATION.md",
        "SUPPORT_MATRIX.md",
        "ADVISOR_MANUSCRIPT_BRIEF.md",
    ]:
        src = DIST_ADVISOR / rel
        if src.exists():
            _copy_file(src, STAGING_ROOT / rel, manifest, "advisor_docs")
        else:
            warnings.append(f"missing dist_advisor/{rel}")

    for rel, category in [
        ("src", "source"),
        ("tests", "tests"),
        ("docs", "docs"),
        ("examples/toy_md", "example"),
        ("examples/real_md_small", "example"),
    ]:
        src = ROOT / rel
        if src.exists():
            _copy_tree(src, STAGING_ROOT / rel, manifest, category)
        else:
            warnings.append(f"missing {rel}")

    for rel in [
        "scripts/check_clean_install.sh",
        "scripts/build_package.sh",
        "scripts/run_toy_quickstart.sh",
        "scripts/package_private_alpha_bundle.py",
    ]:
        src = ROOT / rel
        if src.exists():
            _copy_file(src, STAGING_ROOT / rel, manifest, "script")
        else:
            warnings.append(f"missing {rel}")

    optional_results = [
        "results/private_alpha_hardening/private_alpha_readiness_summary.md",
        "results/input_format_support_upgrade/final_audit/input_format_support_summary.md",
        "results/advisor_bundle/test_report.txt",
        "results/input_format_support_upgrade/test_report.txt",
    ]
    for rel in optional_results:
        src = ROOT / rel
        if src.exists():
            _copy_file(src, STAGING_ROOT / rel, manifest, "result_summary")

    return manifest, warnings


def _archive_staging() -> tuple[Path, Path]:
    zip_path = DIST_ADVISOR / f"{ARCHIVE_BASENAME}.zip"
    tar_path = DIST_ADVISOR / f"{ARCHIVE_BASENAME}.tar.gz"
    for path in [zip_path, tar_path]:
        if path.exists():
            path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in STAGING_ROOT.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(DIST_ADVISOR))
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(STAGING_ROOT, arcname=ARCHIVE_BASENAME)
    return zip_path, tar_path


def _write_advisor_bundle_outputs(manifest: list[dict[str, object]], warnings: list[str], zip_path: Path, tar_path: Path) -> None:
    ADVISOR_RESULTS.mkdir(parents=True, exist_ok=True)
    _write_tsv(ADVISOR_RESULTS / "bundle_manifest.tsv", manifest, ADVISOR_MANIFEST_FIELDS)
    manifest_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "private_alpha_label": PRIVATE_ALPHA_LABEL,
        "archive_zip": str(zip_path.relative_to(ROOT)),
        "archive_tar_gz": str(tar_path.relative_to(ROOT)),
        "bundle_size_bytes": {
            str(zip_path.relative_to(ROOT)): zip_path.stat().st_size,
            str(tar_path.relative_to(ROOT)): tar_path.stat().st_size,
        },
        "max_file_mb": MAX_FILE_MB,
        "warnings": warnings,
        "n_files": len(manifest),
    }
    (ADVISOR_RESULTS / "bundle_manifest.json").write_text(json.dumps(manifest_json, indent=2, sort_keys=True) + "\n")

    checksum_rows: list[dict[str, object]] = []
    for path in [zip_path, tar_path, *[p for p in STAGING_ROOT.rglob("*") if p.is_file()]]:
        checksum_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_tsv(ADVISOR_RESULTS / "checksums.tsv", checksum_rows, CHECKSUM_FIELDS)

    large_files = [
        row for row in manifest if float(row["size_bytes"]) / (1024 * 1024) > MAX_FILE_MB
    ]
    summary = [
        "# Advisor Bundle Summary",
        "",
        f"- Private alpha label: `{PRIVATE_ALPHA_LABEL}`",
        f"- Zip archive: `{zip_path.relative_to(ROOT)}` ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)",
        f"- Tar archive: `{tar_path.relative_to(ROOT)}` ({tar_path.stat().st_size / (1024 * 1024):.2f} MB)",
        f"- Files included: `{len(manifest)}`",
        f"- Large-file threshold: `{MAX_FILE_MB:.1f} MB`",
        f"- Large files above threshold: `{len(large_files)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "This advisor bundle is for internal private-alpha testing only and is not a public release.",
    ]
    if warnings:
        summary.extend(["", "## Warnings", ""])
        summary.extend(f"- {warning}" for warning in warnings)
    if large_files:
        summary.extend(["", "## Large Files", ""])
        summary.extend(f"- `{row['path']}` ({int(row['size_bytes'])} bytes)" for row in large_files)
    (ADVISOR_RESULTS / "bundle_summary.md").write_text("\n".join(summary) + "\n")


def create_advisor_bundle() -> None:
    """Create zip/tar advisor bundle plus manifests and checksums."""

    DIST_ADVISOR.mkdir(parents=True, exist_ok=True)
    ADVISOR_RESULTS.mkdir(parents=True, exist_ok=True)
    test_report = ROOT / "results" / "advisor_bundle" / "test_report.txt"
    test_result = "not run"
    if test_report.exists():
        text = test_report.read_text(errors="ignore")
        test_result = text.strip().splitlines()[-1] if text.strip() else "empty test report"
    _write_version_manifest(build_status="bundle_created", test_result=test_result)
    manifest, warnings = _create_advisor_staging()
    zip_path, tar_path = _archive_staging()
    _write_advisor_bundle_outputs(manifest, warnings, zip_path, tar_path)


def main() -> None:
    """Write legacy private-alpha outputs and create advisor bundle archives."""

    _legacy_private_alpha_outputs()
    create_advisor_bundle()


if __name__ == "__main__":
    main()
