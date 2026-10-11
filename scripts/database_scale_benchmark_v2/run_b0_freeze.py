#!/usr/bin/env python3
"""Freeze D1/GPU V1 evidence and register benchmark v2 provenance."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from scripts.database_scale_benchmark_v2.common import (
    B0_ROOT,
    GPU_PYTHON,
    RESULT_ROOT,
    ROOT,
    THREAD_ENV,
    aggregate_manifest_hash,
    read_tsv,
    repository_path,
    sha256,
    write_json,
    write_tsv,
)


FROZEN_RESULT_FILES = (
    ROOT / "results/database_scale_gpu_v1/d0_freeze/d0_status.json",
    ROOT / "results/database_scale_gpu_v1/d0_freeze/benchmark_registry.tsv",
    ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_matched.tsv",
    ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_gate_summary.json",
    ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_quality_timing_distributions.tsv",
    ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_stratified_quality_timing.tsv",
    ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_failure_warning_atlas.tsv",
    ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_report.md",
    ROOT / "results/database_scale_gpu_v1/d1_compact_handoff.md",
    ROOT / "results/gpu_matched_validation_v1/u6_publication/final_audit.json",
    ROOT / "results/gpu_matched_validation_v1/u6_publication/reproducibility_manifest.json",
    ROOT / "results/throughput_gpu_v1/t0_freeze/t0_status.json",
)

FROZEN_SOURCE_FILES = (
    ROOT / "src/mdsuperpose/sequential_weights.py",
    ROOT / "src/mdsuperpose/window_refine.py",
    ROOT / "src/mdsuperpose/window_kinematics.py",
    ROOT / "src/mdcorealign/experimental.py",
    ROOT / "src/mdcorealign/gpu_backend.py",
    ROOT / "src/mdsuperpose/pdb_writer.py",
    ROOT / "scripts/mdcath_benchmark/extract_ca_trajectory_from_mdcath_h5.py",
    ROOT / "scripts/database_scale_gpu_v1/run_d1_all_domain_matched.py",
)


def command(args: list[str]) -> dict[str, object]:
    """Run a metadata command without changing repository state."""

    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    return {
        "command": args,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def collect_environment() -> dict[str, object]:
    """Collect the canonical software, hardware, and storage boundary."""

    try:
        import cupy
        cupy_version = cupy.__version__
        cuda_runtime = cupy.cuda.runtime.runtimeGetVersion()
    except Exception as exc:
        cupy_version = f"unavailable: {type(exc).__name__}: {exc}"
        cuda_runtime = None
    blas = io.StringIO()
    with redirect_stdout(blas):
        np.show_config()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "expected_python_executable": str(GPU_PYTHON),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cupy_version": cupy_version,
        "cuda_runtime_version": cuda_runtime,
        "blas_configuration": blas.getvalue(),
        "thread_environment": {
            key: os.environ.get(key, "unset") for key in THREAD_ENV
        },
        "cpu": command(["lscpu"]),
        "memory": command(["free", "-b"]),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        ),
        "storage": command(["findmnt", "-T", str(ROOT), "-o", "SOURCE,TARGET,FSTYPE,OPTIONS"]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short", "--branch"]),
        "git_metadata_modified": False,
    }


def main() -> None:
    """Write the immutable B0 registration package."""

    if Path(sys.executable).resolve() != GPU_PYTHON.resolve():
        raise RuntimeError(
            f"B0 requires {GPU_PYTHON}; observed {sys.executable}"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for category, paths in (
        ("frozen_result", FROZEN_RESULT_FILES),
        ("frozen_source", FROZEN_SOURCE_FILES),
    ):
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest_rows.append(
                {
                    "category": category,
                    "path": repository_path(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    # Reuse the already verified U6 file hashes rather than altering that tree.
    u6_manifest = read_tsv(
        ROOT / "results/database_scale_gpu_v1/d0_freeze/frozen_evidence_manifest.tsv"
    )
    for row in u6_manifest:
        manifest_rows.append(
            {
                "category": "frozen_U6_file",
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
        )
    write_tsv(B0_ROOT / "frozen_evidence_manifest.tsv", manifest_rows)

    registry = read_tsv(
        ROOT / "results/database_scale_gpu_v1/d0_freeze/benchmark_registry.tsv"
    )
    source_rows = [
        {
            "domain_id": row["system_id"],
            "dataset_release": row["dataset_release"],
            "source_file": row["source_file"],
            "source_sha256": row["source_sha256"],
            "topology_sha256": row["topology_sha256"],
            "chain_residue_mapping_sha256": row["chain_residue_mapping_sha256"],
            "canonical_coordinate_unit": row["canonical_internal_unit"],
            "unit_status": row["unit_integrity"],
            "local_availability": row["local_available"],
        }
        for row in registry
    ]
    write_tsv(B0_ROOT / "mdcath_source_registry.tsv", source_rows)
    environment = collect_environment()
    write_json(B0_ROOT / "environment.json", environment)

    d1_gate = json.loads(
        (ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_gate_summary.json").read_text()
    )
    status = {
        "phase": "B0",
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_file_count": len(manifest_rows),
        "frozen_aggregate_sha256": aggregate_manifest_hash(manifest_rows),
        "registered_domains": len(registry),
        "available_domains": sum(
            row["local_available"].lower() == "true" for row in registry
        ),
        "d1_historical_gate_pass": d1_gate["gate_pass"],
        "d1_historical_gate_rewritten": False,
        "public_default": "sequential_lean",
        "public_default_changed": False,
        "historical_outputs_modified": False,
        "git_metadata_modified": False,
        "atom_selection": "ordered C-alpha atoms parsed from embedded PDB metadata",
        "canonical_coordinate_unit": "angstrom",
        "loader": repository_path(FROZEN_SOURCE_FILES[-2]),
        "loader_sha256": sha256(FROZEN_SOURCE_FILES[-2]),
        "plan_path": "docs/CODEX_MDCoreAlign_large_scale_benchmark_no_hdf5_optimization.md",
        "dedicated_HDF5_optimization_executed": False,
        "B1_authorized": True,
        "B3_authorized": False,
    }
    write_json(B0_ROOT / "b0_status.json", status)
    (B0_ROOT / "b0_freeze_report.md").write_text(
        "# B0 freeze and registration\n\n"
        "- Status: **PASS**.\n"
        f"- Frozen manifest entries: **{len(manifest_rows)}**.\n"
        f"- Registered mdCATH domains: **{len(registry):,}**.\n"
        f"- Available domains: **{status['available_domains']:,}**.\n"
        "- Canonical selection: ordered C-alpha residues from embedded PDB metadata.\n"
        "- Canonical coordinate unit: Angstrom.\n"
        "- D1 historical failed gate retained: **yes**.\n"
        "- Public exact/default changed: **no**.\n"
        "- Historical evidence or Git metadata modified: **no**.\n"
        "- Dedicated HDF5 optimisation executed: **no**.\n"
        "- B1 authorized: **yes**; B3 remains unauthorized until B1/B2 complete.\n"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
