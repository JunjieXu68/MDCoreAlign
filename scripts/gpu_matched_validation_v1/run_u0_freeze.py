#!/usr/bin/env python3
"""Freeze the validated GPU V1 evidence and preregister U1-U6 criteria."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "throughput_gpu_v1"
OUTPUT = ROOT / "results" / "gpu_matched_validation_v1" / "u0_freeze"
CANONICAL_PYTHON = ROOT / ".venv-gpu-cu12" / "bin" / "python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_output(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
    return {
        "status": "available" if completed.returncode in {0, 1} else "error",
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"missing frozen source: {SOURCE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in SOURCE.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        checksum = _sha256(path)
        size = path.stat().st_size
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": checksum,
            }
        )
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(checksum.encode("ascii"))
        aggregate.update(b"\n")

    manifest_path = OUTPUT / "throughput_gpu_v1_file_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    criteria = {
        "locked_before_new_heldout_evaluation": True,
        "locked_at": generated_at,
        "comparison": "GPU throughput relative to sequential_lean exact",
        "aggregation": "system-level metrics; no frame pooling",
        "required": {
            "finite_outputs_all_cases": True,
            "numerical_failure_count": 0,
            "median_core_jaccard_min": 0.95,
            "minimum_core_jaccard_min": 0.90,
            "median_weight_spearman_min": 0.99,
            "minimum_weight_spearman_min": 0.97,
            "median_rmsf_spearman_min": 0.999,
            "minimum_rmsf_spearman_min": 0.98,
            "median_flexible_label_agreement_min": 0.95,
            "median_absolute_rmsd_difference_angstrom_max": 0.05,
            "p95_absolute_rmsd_difference_angstrom_max": 0.20,
            "median_heldout_pairwise_variance_relative_difference_max": 0.10,
            "median_contact_persistence_mae_max": 0.05,
        },
        "interpretation": (
            "These are practical output-quality criteria, not requirements for "
            "transform-identical parity. Failure is reported without retuning."
        ),
    }
    (OUTPUT / "locked_quality_criteria.json").write_text(
        json.dumps(criteria, indent=2, sort_keys=True) + "\n"
    )

    git = _command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if git.get("return_code") != 0:
        git = {
            **git,
            "status": "no_valid_repository_commit",
            "note": "No Git metadata was initialized or modified.",
        }
    environment = {
        "generated_at": generated_at,
        "canonical_python": str(CANONICAL_PYTHON),
        "executing_python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "git": git,
        "mdlovofit": _command_output(
            [shutil.which("mdlovofit") or "mdlovofit"]
        ),
        "theseus": _command_output(
            [
                "/home/jjxu/miniconda3/envs/mdcorealign-theseus/bin/theseus",
                "-v",
            ]
        ),
    }
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        environment["gpu"] = {
            "cupy_version": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "device_name": properties["name"].decode(),
            "total_memory_bytes": int(properties["totalGlobalMem"]),
            "compute_capability": (
                f"{properties['major']}.{properties['minor']}"
            ),
        }
    except Exception as exc:
        environment["gpu"] = {"status": "unavailable", "error": str(exc)}
    (OUTPUT / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    freeze = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_root": SOURCE.relative_to(ROOT).as_posix(),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "aggregate_sha256": aggregate.hexdigest(),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "historical_outputs_modified": False,
        "git_metadata_modified": False,
        "public_default": "sequential_lean",
        "preserved_experimental_paths": [
            "CPU throughput",
            "GPU throughput",
            "W5 local-deformation",
        ],
    }
    (OUTPUT / "freeze_summary.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT / "u0_freeze_report.md").write_text(
        "\n".join(
            [
                "# U0 GPU V1 freeze",
                "",
                f"- Frozen source: `{freeze['source_root']}`",
                f"- Files recorded: `{freeze['file_count']}`",
                f"- Total bytes: `{freeze['total_bytes']}`",
                f"- Aggregate SHA-256: `{freeze['aggregate_sha256']}`",
                "- Historical result files were read and hashed, not modified.",
                "- Git metadata was not initialized or changed.",
                "- `sequential_lean` remains the exact public default.",
                "- GPU/CPU throughput and W5 remain explicitly experimental.",
                "- U5 quality criteria were locked before new held-out results.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
