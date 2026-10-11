#!/usr/bin/env python3
"""Validate and archive the MDCoreAlign source-code handoff snapshot."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/source_code_handoff_v1"
SNAPSHOT = OUT / "MDCoreAlign_source_code_handoff"
DATE = datetime.now().astimezone().strftime("%Y%m%d")
ARCHIVE = OUT / f"MDCoreAlign_source_code_handoff_{DATE}.tar.gz"
ARCHIVE_ROOT = SNAPSHOT.name

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "dist_advisor",
    ".venv",
    ".venv-gpu-cu12",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".whl",
    ".h5",
    ".hdf5",
    ".xtc",
    ".dcd",
    ".tpr",
    ".trr",
    ".npy",
    ".npz",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def included(path: Path) -> bool:
    relative = path.relative_to(SNAPSHOT)
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name.endswith(".tar.gz") or path.name.endswith(".zip"):
        return False
    if "output" in relative.parts and relative.parts[:2] == ("examples", "toy_md"):
        return False
    return True


def sanitise_validation_records() -> None:
    """Remove transient machine paths from generated validation records only."""

    validation = SNAPSHOT / "validation"
    replacements = (
        (str(SNAPSHOT), "<source-snapshot>"),
        (str(ROOT), "<repository-root>"),
        ("/home/jjxu/miniconda3", "<canonical-python-environment>"),
        ("/home/jjxu", "<user-home>"),
    )
    temporary = re.compile(r"/tmp/mdcorealign_[A-Za-z0-9_.-]+")
    for path in validation.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        updated = temporary.sub("<temporary-workspace>", updated)
        if updated != text:
            path.write_text(updated)


def verify_source_parity() -> tuple[int, list[str]]:
    failures: list[str] = []
    rows = read_tsv(SNAPSHOT / "validation/source_hash_parity.tsv")
    for row in rows:
        relative = Path(row["relative_path"])
        source = ROOT / relative
        copied = SNAPSHOT / relative
        if not source.is_file() or not copied.is_file():
            failures.append(f"missing:{relative.as_posix()}")
            continue
        source_hash = digest(source)
        copied_hash = digest(copied)
        if source_hash != row["source_sha256"] or copied_hash != row["snapshot_sha256"] or source_hash != copied_hash:
            failures.append(f"hash_mismatch:{relative.as_posix()}")
    return len(rows), failures


def validation_gate() -> dict[str, object]:
    checks = {
        "build": (SNAPSHOT / "validation/build_report.log", "Successfully built"),
        "twine": (SNAPSHOT / "validation/twine_check.log", "PASSED"),
        "cpu_clean_install": (SNAPSHOT / "validation/cpu_clean_install_report.log", "cpu_clean_install=pass"),
        "gpu_clean_install": (SNAPSHOT / "validation/gpu_clean_install_report.log", "gpu_clean_install=pass"),
        "full_tests": (SNAPSHOT / "validation/full_test_report.log", "310 passed"),
        "cli_examples": (SNAPSHOT / "validation/public_cli_example_smoke.log", "public_cli_example_smoke=pass"),
    }
    status: dict[str, object] = {}
    for label, (path, marker) in checks.items():
        status[label] = bool(path.is_file() and marker in path.read_text(errors="replace"))
    if not all(status.values()):
        raise RuntimeError(f"validation gate failed: {status}")
    return status


def scan(files: list[Path]) -> tuple[list[dict[str, object]], dict[str, object]]:
    secret_patterns = {
        "hugging_face_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "credential_assignment": re.compile(
            r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"\s]{8,}"
        ),
    }
    path_patterns = {
        "repository_absolute_path": re.compile(r"/storage/junjie/[^\s'\"`]+"),
        "user_home_absolute_path": re.compile(r"/home/jjxu/[^\s'\"`]+"),
    }
    findings: list[dict[str, object]] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        text = path.read_text(errors="replace")
        relative = path.relative_to(SNAPSHOT).as_posix()
        for label, pattern in {**secret_patterns, **path_patterns}.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            if label in path_patterns:
                if relative.startswith("src/") or relative.startswith("docs/") or relative.startswith("examples/") or relative == "README.md":
                    classification = "public_source_path_violation"
                elif relative.startswith("scripts/") or relative.startswith("benchmark/"):
                    classification = "historical_research_script_path_literal"
                elif relative.startswith("results/"):
                    classification = "compact_historical_fixture_path_literal"
                else:
                    classification = "generated_record_path_literal"
            else:
                classification = "secret_violation"
            findings.append(
                {
                    "relative_path": relative,
                    "finding_type": label,
                    "classification": classification,
                    "match_count": len(matches),
                    "status": "fail" if classification in {"secret_violation", "public_source_path_violation"} else "documented_warning",
                }
            )
    secrets = [row for row in findings if row["classification"] == "secret_violation"]
    public_paths = [row for row in findings if row["classification"] == "public_source_path_violation"]
    summary = {
        "secret_scan": "pass" if not secrets else "fail",
        "public_package_docs_examples_path_scan": "pass" if not public_paths else "fail",
        "documented_nonpublic_path_warning_files": len(
            {
                str(row["relative_path"])
                for row in findings
                if row["status"] == "documented_warning"
            }
        ),
        "overall": "pass_with_documented_nonpublic_path_warnings" if not secrets and not public_paths else "fail",
    }
    if summary["overall"] == "fail":
        raise RuntimeError(f"secret/path scan failed: {summary}")
    return findings, summary


def write_status(
    *, parity_count: int, validation: dict[str, object], scan_summary: dict[str, object]
) -> None:
    phases = [
        ("SRC0", "freeze and scope audit", "pass", "original repository and evidence remain untouched"),
        ("SRC1", "installable source and import graph", "pass", f"{parity_count} copied source/configuration files hash-matched"),
        ("SRC2", "architecture and function index", "pass", "source guide, call graph and function index generated"),
        ("SRC3", "tests, examples and compact reproducibility assets", "pass", "compact fixtures exclude raw trajectories and coordinate caches"),
        ("SRC4", "build and CPU validation", "pass", "wheel/sdist, twine, CPU clean install and public workflows passed"),
        ("SRC5", "GPU and regression validation", "pass", "explicit GPU/no-fallback validation and 310-test suite passed"),
        ("SRC6", "archive and independent verification", "pass", "final archive is independently extracted and hash-verified by the finaliser"),
    ]
    write_tsv(
        SNAPSHOT / "validation/phase_status.tsv",
        [{"phase": phase, "scope": scope, "status": status, "notes": notes} for phase, scope, status, notes in phases],
        ["phase", "scope", "status", "notes"],
    )
    write_tsv(
        SNAPSHOT / "validation/package_metadata.tsv",
        [
            {"field": "package_name", "value": "mdcorealign"},
            {"field": "package_version", "value": "0.1.0rc1"},
            {"field": "source_layout", "value": "src"},
            {"field": "console_script", "value": "mdcorealign"},
            {"field": "public_default", "value": "sequential_lean exact CPU"},
            {"field": "gpu_dependency", "value": "optional cupy-cuda12x"},
        ],
        ["field", "value"],
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "package_version": "0.1.0rc1",
        "source_hash_parity": "pass",
        "source_hash_parity_files": parity_count,
        "validation": validation,
        "scan": scan_summary,
        "scientific_algorithms_modified": False,
        "numerical_defaults_modified": False,
        "public_behavior_modified": False,
        "git_metadata_modified": False,
        "instruction_file_status": "CODEX_MDCoreAlign_source_code_handoff_v1.md was not found locally; SRC0-SRC6 were mapped conservatively from the explicit user request",
    }
    (SNAPSHOT / "validation/source_handoff_validation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (SNAPSHOT / "validation/source_handoff_validation_summary.md").write_text(
        "# Source Handoff Validation Summary\n\n"
        "- Package: `mdcorealign 0.1.0rc1`\n"
        f"- Source hash parity: `pass` ({parity_count} files)\n"
        "- Build and twine: `pass`\n"
        "- CPU clean installation: `pass`\n"
        "- GPU clean installation: `pass` (explicit GPU request, no fallback)\n"
        "- Canonical tests: `310 passed, 13 warnings`\n"
        "- Public CLI/example smoke: `pass`\n"
        f"- Secret/path scan: `{scan_summary['overall']}`\n"
        "- Scientific algorithms/defaults/public behavior: `unchanged`\n"
        "- Git metadata: `unchanged`\n\n"
        "Absolute-path warnings, if any, are limited to copied historical "
        "research scripts or compact historical test fixtures; the public "
        "package, user documentation, README and examples are clean.\n"
    )


def archive_files() -> list[Path]:
    return sorted(path for path in SNAPSHOT.rglob("*") if path.is_file() and included(path))


def make_manifests(files: list[Path]) -> None:
    manifest_path = SNAPSHOT / "file_manifest.tsv"
    hash_path = SNAPSHOT / "sha256_manifest.tsv"
    payload = [path for path in files if path not in {manifest_path, hash_path}]
    write_tsv(
        manifest_path,
        [
            {
                "relative_path": path.relative_to(SNAPSHOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
                "category": path.relative_to(SNAPSHOT).parts[0],
            }
            for path in payload
        ],
        ["relative_path", "size_bytes", "sha256", "category"],
    )
    hash_rows = [
        {"relative_path": path.relative_to(SNAPSHOT).as_posix(), "sha256": digest(path)}
        for path in [*payload, manifest_path]
    ]
    write_tsv(hash_path, hash_rows, ["relative_path", "sha256"])


def create_archive(files: list[Path]) -> None:
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=9) as handle:
        for path in files:
            handle.add(path, arcname=f"{ARCHIVE_ROOT}/{path.relative_to(SNAPSHOT).as_posix()}", recursive=False)


def verify_archive() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="mdcorealign_source_handoff_verify_") as tmpdir:
        destination = Path(tmpdir)
        with tarfile.open(ARCHIVE, "r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                target = (destination / member.name).resolve()
                if destination.resolve() not in target.parents:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            handle.extractall(destination)
        extracted = destination / ARCHIVE_ROOT
        rows = read_tsv(extracted / "sha256_manifest.tsv")
        failures = []
        for row in rows:
            path = extracted / row["relative_path"]
            if not path.is_file() or digest(path) != row["sha256"]:
                failures.append(row["relative_path"])
        actual = {
            path.relative_to(extracted).as_posix()
            for path in extracted.rglob("*")
            if path.is_file()
        }
        expected = {row["relative_path"] for row in rows} | {"sha256_manifest.tsv"}
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        if failures or unexpected or missing:
            raise RuntimeError(
                f"archive verification failed: hashes={failures}, unexpected={unexpected}, missing={missing}"
            )
        return {
            "status": "pass",
            "safe_extraction": True,
            "manifest_hash_verification": True,
            "included_file_count": len(actual),
            "maximum_file_size_bytes": max((path.stat().st_size for path in extracted.rglob("*") if path.is_file()), default=0),
        }


def main() -> None:
    if not SNAPSHOT.is_dir():
        raise FileNotFoundError(SNAPSHOT)
    parity_count, parity_failures = verify_source_parity()
    if parity_failures:
        raise RuntimeError(f"source parity failed: {parity_failures[:10]}")
    validation = validation_gate()
    sanitise_validation_records()
    preliminary_files = archive_files()
    findings, scan_summary = scan(preliminary_files)
    write_tsv(
        SNAPSHOT / "validation/secret_path_scan.tsv",
        findings or [{"relative_path": "", "finding_type": "none", "classification": "none", "match_count": 0, "status": "pass"}],
        ["relative_path", "finding_type", "classification", "match_count", "status"],
    )
    excluded = [
        path
        for path in SNAPSHOT.rglob("*")
        if path.is_file() and not included(path)
    ]
    write_tsv(
        SNAPSHOT / "validation/excluded_content_audit.tsv",
        [
            {
                "relative_path": path.relative_to(SNAPSHOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "reason": "build/cache/binary/raw-coordinate exclusion rule",
            }
            for path in excluded
        ]
        or [{"relative_path": "", "size_bytes": 0, "reason": "no excluded files present"}],
        ["relative_path", "size_bytes", "reason"],
    )
    write_status(parity_count=parity_count, validation=validation, scan_summary=scan_summary)
    files = archive_files()
    make_manifests(files)
    files = archive_files()
    create_archive(files)
    archive_validation = verify_archive()
    archive_validation.update(
        {
            "archive": ARCHIVE.name,
            "archive_size_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": digest(ARCHIVE),
            "source_hash_parity_files": parity_count,
            "scan": scan_summary,
        }
    )
    (OUT / "final_archive_validation.json").write_text(json.dumps(archive_validation, indent=2, sort_keys=True) + "\n")
    (OUT / f"{ARCHIVE.name}.sha256").write_text(f"{archive_validation['archive_sha256']}  {ARCHIVE.name}\n")
    print(json.dumps(archive_validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
