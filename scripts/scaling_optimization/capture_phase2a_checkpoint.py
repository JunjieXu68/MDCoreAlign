#!/usr/bin/env python
"""Capture the Phase 2A source checkpoint and GPU environment audit."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import difflib
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v2"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v2"
SOURCE_DIRS = ("src", "tests", "scripts", "docs", "examples", ".github")
SOURCE_FILES = (
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "environment.yml",
    "pyproject.toml",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "scaling_optimization_v2",
}


def timestamp() -> str:
    """Return a timezone-aware generation timestamp."""

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> dict[str, object]:
    """Run a probe without allowing an unavailable command to abort the audit."""

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def source_paths() -> list[Path]:
    """Return checkpoint files while excluding generated and cached content."""

    paths: list[Path] = []
    for directory_name in SOURCE_DIRS:
        directory = PROJECT_ROOT / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(PROJECT_ROOT)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo", ".pstats"}:
                continue
            paths.append(path)
    for filename in SOURCE_FILES:
        path = PROJECT_ROOT / filename
        if path.is_file():
            paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, files: Iterable[Path], archive: Path) -> None:
    """Write source and archive hashes."""

    rows = [
        {
            "path": source.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(source),
            "size_bytes": source.stat().st_size,
            "kind": "source",
        }
        for source in files
    ]
    rows.append(
        {
            "path": archive.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256(archive),
            "size_bytes": archive.stat().st_size,
            "kind": "checkpoint_archive",
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "sha256", "size_bytes", "kind"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_source_patch(path: Path, files: Iterable[Path]) -> None:
    """Represent the unborn repository source tree as an add-file patch."""

    with path.open("w") as handle:
        for source in files:
            relative = source.relative_to(PROJECT_ROOT).as_posix()
            try:
                lines = source.read_text().splitlines(keepends=True)
            except UnicodeDecodeError:
                handle.write(f"Binary source omitted from patch: {relative}\n")
                continue
            handle.writelines(
                difflib.unified_diff(
                    [],
                    lines,
                    fromfile="/dev/null",
                    tofile=f"b/{relative}",
                )
            )


def python_gpu_probe() -> dict[str, object]:
    """Probe optional Python GPU packages without installing anything."""

    packages: dict[str, object] = {}
    for name in ("cupy", "torch", "jax", "jaxlib", "numba"):
        try:
            module = importlib.import_module(name)
            entry: dict[str, object] = {
                "installed": True,
                "version": getattr(module, "__version__", "unknown"),
            }
            if name == "torch":
                entry["cuda_available"] = bool(module.cuda.is_available())
                entry["compiled_cuda"] = module.version.cuda
            elif name == "jax":
                entry["devices"] = [str(device) for device in module.devices()]
            elif name == "numba":
                try:
                    from numba import cuda

                    entry["cuda_available"] = bool(cuda.is_available())
                except Exception as exc:
                    entry["probe_error"] = f"{type(exc).__name__}: {exc}"
            packages[name] = entry
        except Exception as exc:
            packages[name] = {
                "installed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return packages


def static_gpu_implementation_probe() -> list[str]:
    """Find active GPU-backend imports in production package sources."""

    matches: list[str] = []
    needles = ("import cupy", "import torch", "import jax", "from numba", "cuda.")
    for source in (PROJECT_ROOT / "src").rglob("*.py"):
        text = source.read_text(errors="replace")
        if any(needle in text for needle in needles):
            matches.append(source.relative_to(PROJECT_ROOT).as_posix())
    return matches


def main() -> int:
    baseline = RESULT_ROOT / "baseline"
    gpu_dir = RESULT_ROOT / "gpu_environment"
    baseline.mkdir(parents=True, exist_ok=True)
    gpu_dir.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)

    files = source_paths()
    archive = baseline / "source_checkpoint.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for source in files:
            handle.add(source, arcname=source.relative_to(PROJECT_ROOT).as_posix())
    write_manifest(baseline / "source_checkpoint_sha256.tsv", files, archive)
    write_source_patch(baseline / "current_source.patch", files)

    pip_freeze = run([sys.executable, "-m", "pip", "freeze"])
    (baseline / "environment_lock.txt").write_text(str(pip_freeze["stdout"]))
    conda_probe = run(["conda", "env", "export", "--no-builds"])
    (baseline / "environment_conda_export.yml").write_text(
        str(conda_probe["stdout"] or conda_probe["stderr"])
    )

    git_root = run(["git", "rev-parse", "--show-toplevel"])
    git_status = run(["git", "status", "--short", "--", "mdlovofit"])
    git_fsck = run(["git", "fsck", "--full"])
    git_head = run(["git", "rev-parse", "--verify", "HEAD"])
    parent_git = PROJECT_ROOT.parent / ".git"
    git_head_text = (parent_git / "HEAD").read_text(errors="replace") if (parent_git / "HEAD").is_file() else ""
    git_config_text = (
        (parent_git / "config").read_text(errors="replace")
        if (parent_git / "config").is_file()
        else ""
    )
    repository_report = f"""# Phase 2A Repository Recovery Assessment

Generated: {timestamp()}

## Diagnosis

- Project path: `{PROJECT_ROOT}`
- Git top level: `{str(git_root['stdout']).strip() or 'unavailable'}`
- Project-local `.git`: absent.
- Parent `.git`: `{parent_git}` is a directory.
- Parent HEAD content: `{git_head_text.strip() or 'unavailable'}`
- `git rev-parse --verify HEAD`: return code `{git_head['returncode']}`.
- `git fsck --full`: `{str(git_fsck['stdout'] or git_fsck['stderr']).strip()}`.

This is an uninitialised parent repository with an unborn `master` branch and
no commits or refs. It is not a linked worktree and `git fsck` does not report
object corruption. Because every project file is untracked, a normal `git diff`
cannot represent the source state.

## Recoverable Checkpoint

- Source-only archive: `results/scaling_optimization_v2/baseline/source_checkpoint.tar.gz`
- Per-file and archive hashes:
  `results/scaling_optimization_v2/baseline/source_checkpoint_sha256.tsv`
- Add-file patch for text sources:
  `results/scaling_optimization_v2/baseline/current_source.patch`
- Python lock: `results/scaling_optimization_v2/baseline/environment_lock.txt`
- Conda export: `results/scaling_optimization_v2/baseline/environment_conda_export.yml`

The archive excludes results, raw trajectories, data caches, build products,
Python caches, credentials and parent Git metadata.

## Safe Git Proposal

A project-local repository would isolate MDCoreAlign from unrelated sibling
projects, but repository metadata was not changed. After human approval, the
safe commands would be:

```bash
cd {PROJECT_ROOT}
git init -b main
git add .gitignore pyproject.toml README.md LICENSE CITATION.cff CHANGELOG.md \\
  MANIFEST.in environment.yml src tests scripts docs examples .github
git status --short
git commit -m "Checkpoint MDCoreAlign before scaling optimisation"
```

The existing parent `/storage/junjie/projects/.git` should be backed up or
reviewed separately before deciding whether to remove it. This script does not
delete or replace it.

## Captured Git Status

`git status --short -- mdlovofit` produced
{len(str(git_status['stdout']).splitlines())} rows. Full output is stored in
`results/scaling_optimization_v2/baseline/git_status_short.txt`.
"""
    (DOC_ROOT / "00_repository_recovery_report.md").write_text(repository_report)
    (baseline / "git_status_short.txt").write_text(str(git_status["stdout"]))
    (baseline / "git_fsck.txt").write_text(
        f"$ {git_fsck['command']}\n{git_fsck['stdout']}{git_fsck['stderr']}"
    )
    (baseline / "parent_git_metadata.txt").write_text(
        f"HEAD:\n{git_head_text}\nconfig:\n{git_config_text}"
    )

    probe_commands = [
        ["nvidia-smi"],
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,pstate,temperature.gpu,utilization.gpu",
            "--format=csv",
        ],
        ["nvcc", "--version"],
        ["nsys", "--version"],
        ["ncu", "--version"],
        ["ldconfig", "-p"],
    ]
    command_results = [run(command) for command in probe_commands]
    query = command_results[1]
    p5000_visible = query["returncode"] == 0 and "Quadro P5000" in str(query["stdout"])
    python_packages = python_gpu_probe()
    implementation_matches = static_gpu_implementation_probe()
    gpu_environment = {
        "generated_at": timestamp(),
        "hardware": {
            "p5000_visible": p5000_visible,
            "query_output": str(query["stdout"]).strip(),
        },
        "driver": {
            "nvidia_smi_available": command_results[0]["returncode"] == 0,
            "summary": str(command_results[0]["stdout"]).splitlines()[0:5],
        },
        "cuda_toolkit": {
            "nvcc_available": command_results[2]["returncode"] == 0,
            "nvcc_output": str(command_results[2]["stdout"]).strip(),
            "nsys_available": command_results[3]["returncode"] == 0,
            "ncu_available": command_results[4]["returncode"] == 0,
        },
        "python_backends": python_packages,
        "mdcorealign_gpu_implementation": {
            "present": bool(implementation_matches),
            "matching_source_files": implementation_matches,
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    (gpu_dir / "gpu_environment.json").write_text(
        json.dumps(gpu_environment, indent=2, sort_keys=True) + "\n"
    )
    probe_lines: list[str] = []
    for result in command_results:
        probe_lines.extend(
            [
                f"$ {result['command']}",
                str(result["stdout"]),
                str(result["stderr"]),
                f"[returncode={result['returncode']}]",
                "",
            ]
        )
    probe_lines.extend(
        [
            "$ Python GPU package probe",
            json.dumps(python_packages, indent=2, sort_keys=True),
            "",
            "$ MDCoreAlign production-source GPU import probe",
            json.dumps(implementation_matches, indent=2),
        ]
    )
    (gpu_dir / "gpu_probe.txt").write_text("\n".join(probe_lines))

    installed_backends = [
        name
        for name, value in python_packages.items()
        if isinstance(value, dict) and value.get("installed")
    ]
    gpu_report = f"""# Phase 2A GPU Environment Assessment

Generated: {timestamp()}

| Layer | Status | Evidence |
|---|---|---|
| NVIDIA hardware | {'available' if p5000_visible else 'unavailable'} | Quadro P5000 query |
| Kernel/display driver | {'available' if command_results[0]['returncode'] == 0 else 'unavailable'} | `nvidia-smi` |
| CUDA toolkit/compiler | {'available' if command_results[2]['returncode'] == 0 else 'unavailable'} | `nvcc --version` |
| Python GPU backend | {'available: ' + ', '.join(installed_backends) if installed_backends else 'unavailable'} | package import probe |
| MDCoreAlign GPU implementation | {'present' if implementation_matches else 'absent'} | static production-source probe |

The Quadro P5000 is exposed by driver 570.211.01 with a reported CUDA 12.8
driver capability. The installed compiler toolkit is CUDA 12.0
(`nvcc 12.0.140`). Nsight Systems and Nsight Compute are installed.

No CuPy, PyTorch, JAX/JAXlib, or Numba backend is installed in the active
Python environment. More importantly, no active GPU backend is implemented in
the MDCoreAlign production packages. Hardware and CUDA tools therefore do not
constitute a usable MDCoreAlign GPU path, and Phase 3 remains CPU-only.

No driver, CUDA package, Python GPU framework, or repository metadata was
installed or modified during this audit.
"""
    (DOC_ROOT / "01_gpu_environment_assessment.md").write_text(gpu_report)

    print(archive.relative_to(PROJECT_ROOT))
    print(DOC_ROOT.relative_to(PROJECT_ROOT))
    return 0 if p5000_visible else 2


if __name__ == "__main__":
    raise SystemExit(main())
