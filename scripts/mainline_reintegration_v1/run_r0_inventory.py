#!/usr/bin/env python
"""Freeze and inventory the pre-reintegration MDCoreAlign source state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from contextlib import redirect_stdout

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
RESULT_ROOT = PROJECT_ROOT / "results/mainline_reintegration_v1"
BASELINE_ROOT = RESULT_ROOT / "baseline"
ARCHIVE = BASELINE_ROOT / "00_source_checkpoint.tar.gz"
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}
CHECKPOINT_ROOTS = (
    Path("src"),
    Path("tests"),
    Path("scripts/mainline_reintegration_v1"),
)
CHECKPOINT_FILES = (
    Path("pyproject.toml"),
    Path("README.md"),
    Path("docs/CODEX_MDCoreAlign_reintegrate_W5_and_return_to_mainline.md"),
)
W5_FILES = (
    "src/mdsuperpose/window_descriptors.py",
    "src/mdsuperpose/window_scan.py",
    "src/mdsuperpose/window_kinematics.py",
    "src/mdsuperpose/stability_graph.py",
    "src/mdsuperpose/window_refine.py",
)
GLOBAL_FILES = (
    "src/mdsuperpose/sequential_weights.py",
    "src/mdsuperpose/representative_weights.py",
    "src/mdsuperpose/long_mode.py",
    "src/mdsuperpose/fast_kabsch.py",
    "src/mdcorealign/cli.py",
    "src/mdcorealign/__init__.py",
)


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one TSV file."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a TSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def frozen_integrity() -> dict[str, object]:
    """Verify immutable project-completion and golden-reference artifacts."""

    failures: list[dict[str, str]] = []
    project_rows = read_tsv(
        PROJECT_ROOT
        / "results/scaling_optimization_v1/baseline/key_output_hashes.tsv"
    )
    for row in project_rows:
        path = PROJECT_ROOT / row["path"]
        observed = sha256(path) if path.is_file() else "missing"
        if observed != row["sha256"]:
            failures.append(
                {
                    "path": row["path"],
                    "expected": row["sha256"],
                    "observed": observed,
                }
            )
    golden_rows = [
        row
        for row in read_tsv(
            PROJECT_ROOT
            / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
        )
        if row["status"] == "success"
    ]
    for row in golden_rows:
        path = PROJECT_ROOT / row["reference_npz"]
        observed = sha256(path) if path.is_file() else "missing"
        if observed != row["reference_sha256"]:
            failures.append(
                {
                    "path": row["reference_npz"],
                    "expected": row["reference_sha256"],
                    "observed": observed,
                }
            )
    return {
        "status": "pass" if not failures else "failed",
        "project_completion_checked": len(project_rows),
        "golden_checked": len(golden_rows),
        "failures": failures,
    }


def package_version(name: str) -> str:
    """Return an installed package version or an explicit unavailable value."""

    try:
        module = __import__(name)
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    return str(getattr(module, "__version__", "unknown"))


def environment_record() -> dict[str, object]:
    """Capture the canonical interpreter and numerical runtime."""

    config_buffer = io.StringIO()
    with redirect_stdout(config_buffer):
        np.show_config()
    return {
        "python_executable": str(PYTHON),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": package_version("scipy"),
        "MDAnalysis_version": package_version("MDAnalysis"),
        "h5py_version": package_version("h5py"),
        "blas_numpy_show_config": config_buffer.getvalue(),
        "thread_environment": THREAD_ENVIRONMENT,
    }


def source_paths() -> list[Path]:
    """Return deterministic checkpoint inputs without bytecode or caches."""

    paths: list[Path] = []
    for root in CHECKPOINT_ROOTS:
        absolute = PROJECT_ROOT / root
        if not absolute.is_dir():
            continue
        paths.extend(
            path
            for path in absolute.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    paths.extend(
        PROJECT_ROOT / path
        for path in CHECKPOINT_FILES
        if (PROJECT_ROOT / path).is_file()
    )
    return sorted(set(paths))


def build_checkpoint(paths: list[Path]) -> list[dict[str, object]]:
    """Create a recoverable source archive and source hash manifest."""

    rows = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    write_tsv(BASELINE_ROOT / "00_source_manifest.tsv", rows)
    return rows


def run_tests() -> dict[str, object]:
    """Run the complete pre-integration test suite."""

    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    output = completed.stdout + completed.stderr
    (BASELINE_ROOT / "00_full_test_results.txt").write_text(output)
    return {
        "command": (
            "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
            "NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. "
            "/home/jjxu/miniconda3/bin/python -m pytest -q tests"
        ),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": (completed.stdout.strip().splitlines() or [""])[-1],
    }


def parser_contract() -> dict[str, object]:
    """Read the current CLI commands and defaults without invoking a shell."""

    from mdcorealign.cli import build_parser

    parser = build_parser()
    commands: dict[str, dict[str, object]] = {}
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not isinstance(choices, dict):
            continue
        for command, subparser in choices.items():
            defaults = {
                item.dest: item.default
                for item in subparser._actions
                if item.dest in {"mode", "warm_start", "max_iterations"}
            }
            commands[str(command)] = {
                "defaults": defaults,
                "help": subparser.description,
            }
    return {
        "console_entry_point": "mdcorealign = mdcorealign.cli:main",
        "commands": commands,
        "exact_public_default": {
            "cli_mode": "default",
            "solver": "solve_trajectory_robust_alignment",
            "long_mode_representative_estimator": "sequential_lean",
        },
        "public_python_namespace": "mdcorealign",
    }


def write_inventory(
    *,
    environment: dict[str, object],
    integrity: dict[str, object],
    tests: dict[str, object],
    checkpoint_rows: list[dict[str, object]],
    cli: dict[str, object],
) -> None:
    """Write the human-readable R0 inventory."""

    git = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = f"""# R0 reintegration inventory

## Scope

This is the pre-mainline-reintegration checkpoint. It predates all R2/R3
public interface edits. Raw trajectories, benchmark caches and secrets are
excluded.

## Canonical environment

- Python: `{environment['python_executable']}`
- Python version: `{str(environment['python_version']).splitlines()[0]}`
- NumPy: `{environment['numpy_version']}`
- SciPy: `{environment['scipy_version']}`
- MDAnalysis: `{environment['MDAnalysis_version']}`
- h5py: `{environment['h5py_version']}`
- Thread controls: `OPENBLAS/OMP/MKL/NUMEXPR=2`

Full NumPy/BLAS configuration is recorded in `00_environment.json`.

## Source checkpoint

- Archive: `baseline/00_source_checkpoint.tar.gz`
- Files: {len(checkpoint_rows)}
- Archive SHA256: `{sha256(ARCHIVE)}`
- Source paths and hashes: `baseline/00_source_manifest.tsv`

W5/window files:
{os.linesep.join(f'- `{path}`' for path in W5_FILES)}

Exact/fast-global files:
{os.linesep.join(f'- `{path}`' for path in GLOBAL_FILES)}

## Public interface before reintegration

- Console entry point: `{cli['console_entry_point']}`
- Commands: {', '.join(sorted(cli['commands']))}
- Alignment default: `--mode default`
- Exact/default solver: `solve_trajectory_robust_alignment`
- Existing long exact estimator: `sequential_lean`
- W5 and uniform low-budget modes are not yet mainline CLI commands/modes.

## Integrity

- Tests: **{tests['status'].upper()}**, `{tests['summary']}`
- Frozen artifacts: **{integrity['status'].upper()}**
- Project-completion hashes: {integrity['project_completion_checked']}/
  {integrity['project_completion_checked']}
- Golden references: {integrity['golden_checked']}/
  {integrity['golden_checked']}
- Valid Git HEAD: `{git.returncode == 0}`. Git metadata was not modified.
"""
    (BASELINE_ROOT / "00_reintegration_inventory.md").write_text(report)


def main() -> int:
    """Execute R0 exactly once."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-existing-empty-root",
        action="store_true",
        help="Reserved for controlled reruns before any R0 artifact exists.",
    )
    parser.parse_args()
    if BASELINE_ROOT.exists() and any(BASELINE_ROOT.iterdir()):
        raise RuntimeError("R0 baseline already exists; refusing to overwrite")
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    environment = environment_record()
    (BASELINE_ROOT / "00_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    cli = parser_contract()
    (BASELINE_ROOT / "00_public_interface_before.json").write_text(
        json.dumps(cli, indent=2) + "\n"
    )
    integrity = frozen_integrity()
    tests = run_tests()
    paths = source_paths()
    rows = build_checkpoint(paths)
    payload = {
        "phase": "R0",
        "status": (
            "pass"
            if integrity["status"] == "pass" and tests["status"] == "pass"
            else "failed"
        ),
        "integrity": integrity,
        "tests": tests,
        "source_checkpoint": {
            "path": str(ARCHIVE.relative_to(PROJECT_ROOT)),
            "sha256": sha256(ARCHIVE),
            "file_count": len(rows),
        },
        "public_default_changed": False,
        "git_metadata_modified": False,
    }
    (BASELINE_ROOT / "00_integrity_before.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_inventory(
        environment=environment,
        integrity=integrity,
        tests=tests,
        checkpoint_rows=rows,
        cli=cli,
    )
    if payload["status"] != "pass":
        raise RuntimeError("R0 tests or frozen integrity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
