#!/usr/bin/env python
"""Capture the immutable Phase 0 repository and environment baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/scaling_optimization_v1/baseline"
HASH_TARGETS = (
    "docs/project_completion/fast_and_good_final_decision_report.md",
    "docs/project_completion/supervisor_summary.md",
    "results/project_completion/manuscript/mdcorealign_revised_main.md",
    "results/project_completion/manuscript/mdcorealign_revised_main.docx",
    "results/project_completion/tables/all_domain_mdcorealign_vs_mdlovofit_main_claim.tsv",
    "results/project_completion/figures/fig_all_domain_runtime_rmsd.pdf",
    "results/project_completion/figures/fig_all_domain_runtime_rmsd.png",
    "results/project_completion/figures/fig_long_mode_scaling.pdf",
    "results/project_completion/figures/fig_long_mode_scaling.png",
)


def command(args: list[str], *, timeout: float = 30.0) -> tuple[int, str]:
    """Run a read-only inventory command and capture combined output."""

    try:
        run = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = run.stdout
        if run.stderr:
            output += ("\n" if output else "") + run.stderr
        return run.returncode, output.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def sha256(path: Path) -> str:
    """Hash one immutable evidence file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_git_state(output: Path) -> dict[str, object]:
    """Capture Git metadata, including an explicit unavailable state."""

    root_rc, root = command(["git", "rev-parse", "--show-toplevel"])
    commit_rc, commit = command(["git", "rev-parse", "--verify", "HEAD"])
    branch_rc, branch = command(["git", "branch", "--show-current"])
    status_rc, status = command(["git", "status", "--short"], timeout=60.0)
    payload = {
        "repository_root": root if root_rc == 0 else "unavailable",
        "commit": commit if commit_rc == 0 else "unavailable",
        "branch": branch if branch_rc == 0 and branch else "unavailable",
        "status_return_code": status_rc,
        "status_short": status,
        "note": (
            "The project resides below an initialized parent Git directory with no valid HEAD; "
            "commit and branch are therefore unavailable."
            if commit_rc != 0
            else ""
        ),
    }
    lines = [
        "command: git rev-parse --show-toplevel",
        f"repository_root: {payload['repository_root']}",
        "command: git rev-parse --verify HEAD",
        f"commit: {payload['commit']}",
        "command: git branch --show-current",
        f"branch: {payload['branch']}",
        "command: git status --short",
        str(payload["status_short"]),
        f"note: {payload['note']}",
    ]
    (output / "git_state.txt").write_text("\n".join(lines).rstrip() + "\n")
    return payload


def capture_environment(output: Path) -> dict[str, object]:
    """Capture versioned software details without dumping environment secrets."""

    mdlovo_rc, mdlovo = command(["/usr/local/bin/mdlovofit"], timeout=10.0)
    theseus_path = Path("/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus")
    theseus_rc, theseus = command([str(theseus_path), "-v"], timeout=10.0)
    nvidia_rc, nvidia = command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    package_version = importlib.metadata.version("mdcorealign")
    relevant_env = {
        key: os.environ.get(key, "unset")
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    lines = [
        f"captured_at: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"mdcorealign: {package_version}",
        f"platform: {platform.platform()}",
        f"MDLovoFit executable: {shutil.which('mdlovofit') or 'unavailable'}",
        f"MDLovoFit invocation: mdlovofit -f 0.6 -t OUTPUT.pdb INPUT.pdb",
        f"MDLovoFit version output return code: {mdlovo_rc}",
        *mdlovofit_version_lines(mdlovo),
        f"THESEUS executable: {theseus_path if theseus_path.exists() else 'unavailable'}",
        f"THESEUS invocation: theseus -a0 INPUT.pdb",
        f"THESEUS version output return code: {theseus_rc}",
        *theseus_version_lines(theseus),
        f"GPU query return code: {nvidia_rc}",
        f"GPU: {nvidia or 'unavailable'}",
        "Relevant thread/backend environment:",
        *[f"{key}={value}" for key, value in relevant_env.items()],
    ]
    (output / "environment.txt").write_text("\n".join(lines) + "\n")
    pip_rc, pip_freeze = command([sys.executable, "-m", "pip", "freeze"], timeout=120.0)
    (output / "dependencies.txt").write_text(
        f"# command: python -m pip freeze\n# return_code: {pip_rc}\n{pip_freeze}\n"
    )
    return {
        "python_version": platform.python_version(),
        "package_version": package_version,
        "mdlovofit_available": mdlovo_rc in {0, 1},
        "theseus_available": theseus_path.exists() and theseus_rc in {0, 1},
        "gpu_query_available": nvidia_rc == 0,
        "thread_environment": relevant_env,
    }


def mdlovofit_version_lines(output: str) -> list[str]:
    """Extract a compact MDLovoFit version line."""

    lines = [line.strip() for line in output.splitlines() if "Version" in line]
    return [f"MDLovoFit version: {lines[0] if lines else 'unavailable'}"]


def theseus_version_lines(output: str) -> list[str]:
    """Extract a compact THESEUS version line."""

    lines = [line.strip() for line in output.splitlines() if "THESEUS" in line or "theseus" in line]
    return [f"THESEUS version/header: {lines[0] if lines else 'unavailable'}"]


def capture_hardware(output: Path) -> dict[str, object]:
    """Capture CPU, memory, GPU and storage metadata."""

    _, lscpu = command(["lscpu"])
    _, free = command(["free", "-b"])
    _, lsblk = command(["lsblk", "-J", "-o", "NAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS"])
    _, storage = command(["df", "-T", str(PROJECT_ROOT)])
    gpu_rc, gpu = command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    payload = {
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "cpu": lscpu,
        "memory": free,
        "block_devices": json.loads(lsblk) if lsblk.startswith("{") else lsblk,
        "project_storage": storage,
        "gpu": gpu if gpu_rc == 0 else "unavailable",
        "storage_note": "Project root is on /storage, an ext4 filesystem backed by rotational RAID5 disks.",
    }
    (output / "hardware.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def capture_hashes(output: Path) -> dict[str, str]:
    """Write SHA-256 hashes for immutable source-of-truth evidence."""

    hashes: dict[str, str] = {}
    rows = []
    for relative in HASH_TARGETS:
        path = PROJECT_ROOT / relative
        if path.exists():
            value = sha256(path)
            hashes[relative] = value
            rows.append({"path": relative, "sha256": value, "size_bytes": path.stat().st_size})
        else:
            rows.append({"path": relative, "sha256": "MISSING", "size_bytes": ""})
    with (output / "key_output_hashes.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "size_bytes"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return hashes


def main() -> int:
    """Capture Phase 0 files without modifying completed evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    git_state = capture_git_state(output)
    environment = capture_environment(output)
    _ = capture_hardware(output)
    hashes = capture_hashes(output)
    test_path = output / "test_results.txt"
    smoke_path = output / "smoke_benchmark.tsv"
    payload = {
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "phase": "Phase 0",
        "git": git_state,
        "environment": environment,
        "immutable_hash_count": len(hashes),
        "test_results_present": test_path.exists(),
        "smoke_benchmark_present": smoke_path.exists(),
        "completed_evidence_modified": False,
        "integrity_status": (
            "pass"
            if hashes and test_path.exists() and smoke_path.exists()
            else "incomplete"
        ),
    }
    (output / "baseline_integrity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
