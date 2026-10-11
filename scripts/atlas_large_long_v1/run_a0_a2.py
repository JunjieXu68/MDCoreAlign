#!/usr/bin/env python3
"""Freeze post-B3 evidence and register the ATLAS v1 quality contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import platform
import subprocess

from scripts.atlas_large_long_v1.common import (
    GPU_PYTHON,
    LOCKED,
    MDA_PYTHON,
    MDLOVOFIT,
    OUT,
    PREFIXES,
    P7,
    REPEATS,
    ROOT,
    THESEUS,
    THREAD_ENV,
    WARMUPS,
    read_tsv,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


def freeze() -> None:
    if OUT.exists():
        raise FileExistsError(
            f"A0 output root already exists; refusing to overwrite: {OUT}"
        )
    target = ROOT / "results/post_b3_optimisation_v1"
    files = sorted(path for path in target.rglob("*") if path.is_file())
    rows = []
    for index, path in enumerate(files, start=1):
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "freeze_scope": "completed_P0_P7",
            }
        )
        if index % 2000 == 0 or index == len(files):
            print(f"[A0 freeze] {index}/{len(files)}", flush=True)
    write_tsv(OUT / "a0_freeze/post_b3_frozen_manifest.tsv", rows)
    p0_manifest = target / "p0_freeze/frozen_source_evidence_manifest.tsv"
    git_status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    write_text(OUT / "a0_freeze/git_status_read_only.txt", git_status.stdout)
    write_json(
        OUT / "a0_freeze/a0_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "frozen_files": len(rows),
            "frozen_bytes": sum(int(row["size_bytes"]) for row in rows),
            "post_b3_root": str(target.relative_to(ROOT)),
            "p0_historical_freeze_manifest": str(p0_manifest.relative_to(ROOT)),
            "p0_historical_freeze_manifest_sha256": sha256(p0_manifest),
            "project_completion_and_golden_references_covered_by_P0_manifest": True,
            "public_default": "sequential_lean_exact_CPU",
            "principal_large_long_backend": "GPU_throughput_float32_Horn",
            "backend_auto_v2_status": "experimental",
            "W5_status": "experimental_alignment_free_local_deformation",
            "MDLovoFit_preprocessing": "per_frame_PDB_safe_COG_centering",
            "git_metadata_modified": False,
            "historical_outputs_modified": False,
            "python": str(GPU_PYTHON),
            "platform": platform.platform(),
            "thread_environment": THREAD_ENV,
        },
    )


def register_contract() -> None:
    manifest = OUT / "atlas_manifest.tsv"
    if not manifest.exists():
        raise FileNotFoundError("A1 atlas_manifest.tsv must exist before A2")
    rows = read_tsv(manifest)
    if len(rows) != 8 or any(row["input_audit_status"] != "pass" for row in rows):
        raise RuntimeError("A1 must pass for all eight systems before A2")
    forbidden = list(OUT.glob("a3_*")) + list(OUT.glob("atlas_*results.tsv"))
    if forbidden:
        raise RuntimeError(
            "benchmark outputs already exist; prospective registration refused: "
            + ", ".join(str(path) for path in forbidden)
        )
    contract = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "registration_is_pre_output": True,
        "locked_panel": {
            "systems": [row["system"] for row in rows],
            "replicate": "R1_only",
            "prefixes_frames": list(PREFIXES),
            "R2_R3_claim_permitted": False,
        },
        "methods": [
            "CPU_exact_sequential_lean",
            "CPU_throughput",
            "GPU_throughput_float32_Horn",
            "backend_auto_v2_threshold_25000",
            "CPU_W5",
            "GPU_W5",
            "MDLovoFit_20.1.1",
            "THESEUS_3.3.0_where_feasible",
        ],
        "algorithm_parameters": {
            "exact_representative_count": 2000,
            "exact_chunk_size": 1000,
            "throughput_representative_count": 256,
            "throughput_update_rounds": 4,
            "throughput_chunk_size": 1000,
            "GPU_precision": "float32",
            "GPU_rotation_solver": "horn",
            "core_fraction": 0.6,
            "flexible_fraction": 0.2,
            "backend_auto_v2_threshold_residue_frames": 25000,
            "CPU_threads": THREAD_ENV,
        },
        "timing": {
            "warmups": WARMUPS,
            "measured_repeats": REPEATS,
            "persistent_warmed_GPU_process": True,
            "explicit_CUDA_synchronization": True,
            "contracts": {
                "A_loaded_analysis": "decoded CA coordinates through compact analysis outputs",
                "B_raw_source_end_to_end": "PDB/XTC read through compact analysis outputs",
                "C_materialised_output": "analysis plus transform application and aligned CA PDB writing",
            },
            "contracts_must_not_be_mixed": True,
        },
        "external_policy": {
            "MDLovoFit_timeout_seconds_per_run": 1800,
            "MDLovoFit_fraction": 0.6,
            "MDLovoFit_centering": "subtract_each_frame_CA_COG_before_PDB_write",
            "THESEUS_timeout_seconds_per_run": 600,
            "THESEUS_feasible_max_frame_residue_points": 2000000,
            "external_timeout_is_recorded_not_retried_with_changed_threshold": True,
        },
        "quality_gates": {
            "CPU_GPU_implementation_core_Jaccard": 1.0,
            "threshold_tie_rule": "deterministic_top_k_by_descending_weight_then_residue_index; if a boundary tie causes disagreement, run and report a float64 GPU label-only adjudication without changing timed transforms",
            "exact_relative_core_Jaccard_min_per_case": 0.95,
            "exact_relative_RMSF_Spearman_min_per_case": 0.995,
            "RMSF_MAE_A": "report_per_system_no_fixed_gate",
            "RMSD_difference_A": "report_median_and_P95_per_system_no_transform_identity_gate",
            "non_finite_outputs_allowed": False,
            "invalid_transforms_allowed": False,
            "mapping_failures_allowed": False,
            "empty_cores_allowed": False,
        },
        "failure_classification": {
            "registered_catastrophic": "nonfinite_or_invalid_transform_or_mapping_failure_or_empty_core_or_core_Jaccard_lt_0.5_or_RMSF_Spearman_lt_0.9",
            "large_geometric_tail": "median_absolute_unified_RMSD_difference_gt_1A",
            "categories_are_separate": True,
        },
        "batching": {
            "batch_sizes": [1, 2, 4, 8],
            "contracts": ["throughput_only", "W5_only", "fused_throughput_plus_W5"],
            "duplicate_inputs_allowed_only_as_computational_stress_controls": True,
            "duplicate_inputs_are_not_biological_replicates": True,
        },
        "executables": {
            "GPU_python": str(GPU_PYTHON),
            "MDAnalysis_python": str(MDA_PYTHON),
            "MDLovoFit": str(MDLOVOFIT),
            "THESEUS": str(THESEUS),
        },
    }
    write_json(OUT / "a2_registration/prospective_quality_contract.json", contract)
    write_text(
        OUT / "a2_registration/prospective_quality_contract.md",
        """# Prospective ATLAS quality and timing contract

This contract was written after input audit and before method outputs. The locked
panel contains eight ATLAS R1 trajectories and six frame prefixes. Thresholds
must not be tuned after inspecting these results.

- CPU/GPU throughput core Jaccard must equal 1.0; deterministic boundary ties
  may be adjudicated by a reported float64 label-only step.
- Each throughput case must have exact-relative core Jaccard >=0.95 and RMSF
  Spearman >=0.995.
- RMSF MAE and median/P95 RMSD differences are reported without requiring
  transform identity.
- Non-finite output, invalid transforms, mapping failure and empty cores fail.
- Registered catastrophic failures and median RMSD tails >1 A are separate.
- Loaded analysis, raw-source end-to-end and materialised-output timings are
  separate contracts.
- MDLovoFit uses per-frame CA COG centering. THESEUS runs only when the locked
  frame-residue point count is <=2,000,000.
- R2/R3 reproducibility and independently verified PBC/nojump are out of scope.
""",
    )
    write_json(
        OUT / "a2_registration/registration_status.json",
        {
            "registered_at": contract["registered_at"],
            "systems": 8,
            "prefixes": len(PREFIXES),
            "benchmark_outputs_present_at_registration": False,
            "contract_sha256": sha256(
                OUT / "a2_registration/prospective_quality_contract.json"
            ),
            "thresholds_locked": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "register-contract"))
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    else:
        register_contract()


if __name__ == "__main__":
    main()
