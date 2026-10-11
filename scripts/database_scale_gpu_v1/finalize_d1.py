#!/usr/bin/env python3
"""Aggregate D1 checkpoints, run cold probes, and enforce locked gates."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from scripts.database_scale_gpu_v1.common import (
    D0_ROOT,
    D1_ROOT,
    EXPECTED_ELIGIBLE,
    EXPECTED_REGISTERED,
    QUALITY_GATES_PATH,
    REGISTRY_PATH,
    ROOT,
    finite_float,
    percentile,
    read_tsv,
    repository_path,
    sha256,
    write_json,
    write_tsv,
)
from scripts.scaling_optimization_v4.finalize_phase6 import verify_hashes


CHECKPOINT_ROOT = D1_ROOT / "checkpoints"
RUNNER = ROOT / "scripts/database_scale_gpu_v1/run_d1_all_domain_matched.py"
PYTHON = ROOT / ".venv-gpu-cu12/bin/python"


def truth(value: object) -> bool:
    """Parse a serialized Boolean."""

    return str(value).strip().lower() == "true"


def values(rows: list[dict[str, object]], field: str) -> list[float]:
    """Collect finite values from rows."""

    output: list[float] = []
    for row in rows:
        value = finite_float(row.get(field))
        if value is not None:
            output.append(value)
    return output


def summary_stats(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    """Return compact distribution statistics."""

    observed = values(rows, field)
    return {
        "field": field,
        "n": len(observed),
        "minimum": min(observed) if observed else "",
        "p05": percentile(observed, 0.05) if observed else "",
        "median": statistics.median(observed) if observed else "",
        "p95": percentile(observed, 0.95) if observed else "",
        "maximum": max(observed) if observed else "",
    }


def residue_bin(row: dict[str, object]) -> str:
    """Return a locked residue-count bin."""

    value = int(row["n_residues"])
    if value <= 100:
        return "50-100"
    if value <= 200:
        return "101-200"
    if value <= 300:
        return "201-300"
    return "301-500"


def frame_bin(row: dict[str, object]) -> str:
    """Return a locked frame-count bin."""

    value = int(row["n_frames"])
    if value <= 100:
        return "<=100"
    if value <= 250:
        return "101-250"
    if value <= 500:
        return "251-500"
    return ">500"


def core_bin(row: dict[str, object]) -> str:
    """Return an effective exact-core-weight bin."""

    value = float(row["effective_core_fraction_exact"])
    if value < 0.40:
        return "<0.40"
    if value < 0.60:
        return "0.40-0.60"
    return ">=0.60"


def motion_bin(row: dict[str, object]) -> str:
    """Return a historical RMSF-ratio motion stratum."""

    value = float(row["motion_proxy_noncore_core_RMSF_ratio"])
    if value < 2.0:
        return "low_<2"
    if value < 4.0:
        return "medium_2-4"
    return "high_>=4"


def workload_bin(row: dict[str, object]) -> str:
    """Return a residue-frame workload bin."""

    value = int(row["n_frames"]) * int(row["n_residues"])
    if value <= 50_000:
        return "<=50k"
    if value <= 100_000:
        return "50k-100k"
    if value <= 200_000:
        return "100k-200k"
    return ">200k"


def stratified_rows(success: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize D1 quality and performance over locked strata."""

    definitions = {
        "CATH_class": lambda row: str(row["cath_class"]),
        "residue_count": residue_bin,
        "frame_count": frame_bin,
        "chain_count": lambda row: str(row["observed_chain_count"]),
        "effective_core_fraction": core_bin,
        "motion_flexibility": motion_bin,
        "CPU_GPU_workload": workload_bin,
    }
    output: list[dict[str, object]] = []
    for dimension, function in definitions.items():
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in success:
            groups[function(row)].append(row)
        for label, group in sorted(groups.items()):
            for method in ("cpu", "gpu"):
                output.append(
                    {
                        "stratification": dimension,
                        "stratum": label,
                        "method": (
                            "CPU_throughput"
                            if method == "cpu"
                            else "GPU_throughput"
                        ),
                        "n_systems": len(group),
                        "core_jaccard_min": min(
                            float(row[f"{method}_core_jaccard_vs_exact"])
                            for row in group
                        ),
                        "core_jaccard_median": statistics.median(
                            float(row[f"{method}_core_jaccard_vs_exact"])
                            for row in group
                        ),
                        "weight_spearman_median": statistics.median(
                            float(
                                row[f"{method}_weight_spearman_vs_exact"]
                            )
                            for row in group
                        ),
                        "RMSF_spearman_median": statistics.median(
                            float(
                                row[f"{method}_rmsf_spearman_vs_exact"]
                            )
                            for row in group
                        ),
                        "RMSD_delta_median_A": statistics.median(
                            float(
                                row[
                                    f"{method}_median_absolute_rmsd_"
                                    "difference_vs_exact_A"
                                ]
                            )
                            for row in group
                        ),
                        "loaded_compute_median_seconds": statistics.median(
                            float(
                                row[
                                    f"{method}_throughput_loaded_compute_seconds"
                                ]
                            )
                            for row in group
                        ),
                        "catastrophic_count": sum(
                            truth(
                                row[
                                    f"{method}_catastrophic_quality_failure"
                                ]
                            )
                            for row in group
                        ),
                    }
                )
    return output


def load_checkpoints() -> list[dict[str, object]]:
    """Load exactly one checkpoint for every registered domain."""

    registry = read_tsv(REGISTRY_PATH)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for item in registry:
        path = CHECKPOINT_ROOT / f"{item['trajectory_id']}.json"
        if not path.is_file():
            missing.append(item["trajectory_id"])
            continue
        rows.append(json.loads(path.read_text()))
    if missing:
        raise RuntimeError(
            f"D1 has {len(missing)} missing checkpoints; sample={missing[:10]}"
        )
    if len(rows) != EXPECTED_REGISTERED:
        raise RuntimeError(
            f"D1 loaded {len(rows)}/{EXPECTED_REGISTERED} checkpoints"
        )
    return rows


def frozen_evidence_check() -> dict[str, object]:
    """Revalidate D0's immutable U6/GPU V1 manifest."""

    rows = read_tsv(D0_ROOT / "frozen_evidence_manifest.tsv")
    failures: list[dict[str, str]] = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
        elif path.stat().st_size != int(row["size_bytes"]):
            failures.append({"path": row["path"], "reason": "size_changed"})
        elif sha256(path) != row["sha256"]:
            failures.append({"path": row["path"], "reason": "hash_changed"})
    return {
        "status": "pass" if not failures else "fail",
        "checked": len(rows),
        "failures": failures,
    }


def run_cold_probes() -> list[dict[str, object]]:
    """Run or reuse the locked 12-case fresh-process GPU subset."""

    output_path = D1_ROOT / "cold_start_subset.tsv"
    existing = (
        {
            row["trajectory_id"]: row
            for row in read_tsv(output_path)
        }
        if output_path.is_file()
        else {}
    )
    registry = [
        row for row in read_tsv(REGISTRY_PATH) if truth(row["cold_start_subset"])
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": "src:.",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    for row in registry:
        trajectory_id = row["trajectory_id"]
        if trajectory_id in existing:
            continue
        start = time.perf_counter()
        process = subprocess.run(
            [
                str(PYTHON),
                str(RUNNER),
                "--cold-probe",
                trajectory_id,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        wall = time.perf_counter() - start
        marker = "D1_COLD_JSON="
        payload = None
        for line in process.stdout.splitlines():
            if line.startswith(marker):
                payload = json.loads(line[len(marker) :])
        if process.returncode == 0 and payload is not None:
            payload["fresh_process_parent_wall_seconds"] = wall
            payload["return_code"] = process.returncode
            payload["stderr"] = process.stderr.strip()
            existing[trajectory_id] = payload
        else:
            existing[trajectory_id] = {
                "trajectory_id": trajectory_id,
                "status": "failed",
                "fresh_process_parent_wall_seconds": wall,
                "return_code": process.returncode,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
            }
        write_tsv(output_path, sorted(existing.values(), key=lambda x: x["trajectory_id"]))
    return sorted(existing.values(), key=lambda row: row["trajectory_id"])


def gate_status(
    rows: list[dict[str, object]],
    frozen: dict[str, object],
    legacy: dict[str, object],
) -> dict[str, object]:
    """Apply only the criteria locked in D0."""

    gates = json.loads(QUALITY_GATES_PATH.read_text())["required"]
    success = [row for row in rows if row["status"] == "success"]
    unavailable = [
        row for row in rows if row["status"] == "explicitly_unavailable"
    ]
    failed = [row for row in rows if row["status"] == "failed"]
    unit_failures = [
        row for row in failed if row["failure_class"] == "unit_failure"
    ]
    mapping_failures = [
        row for row in failed if row["failure_class"] == "mapping_failure"
    ]
    cpu_catastrophic = [
        row
        for row in success
        if truth(row["cpu_catastrophic_quality_failure"])
    ]
    gpu_catastrophic = [
        row
        for row in success
        if truth(row["gpu_catastrophic_quality_failure"])
    ]
    low_core = [
        row
        for row in success
        if min(
            float(row["cpu_core_jaccard_vs_exact"]),
            float(row["gpu_core_jaccard_vs_exact"]),
        )
        < float(gates["minimum_core_jaccard_vs_exact"])
    ]
    implementation_failures = []
    for row in success:
        core_ok = (
            float(row["cpu_gpu_core_jaccard"]) == 1.0
            or truth(
                row[
                    "cpu_gpu_core_difference_explained_by_numerical_tie"
                ]
            )
        )
        if not (
            core_ok
            and float(row["cpu_gpu_weight_spearman"])
            >= float(gates["cpu_gpu_minimum_weight_spearman"])
            and float(row["cpu_gpu_rmsf_spearman"])
            >= float(gates["cpu_gpu_minimum_RMSF_spearman"])
            and float(
                row["cpu_gpu_median_absolute_rmsd_difference_A"]
            )
            <= float(
                gates["cpu_gpu_maximum_median_RMSD_difference_A"]
            )
        ):
            implementation_failures.append(row)
    w5_failures = [
        row
        for row in success
        if float(row["w5_cpu_gpu_local_deformation_spearman"])
        < float(gates["w5_cpu_gpu_minimum_local_deformation_spearman"])
    ]
    checks = {
        "registered_count": len(rows) == EXPECTED_REGISTERED,
        "eligible_success_count": len(success) == EXPECTED_ELIGIBLE,
        "explicitly_unavailable_count": len(unavailable) == 1,
        "eligible_processing_failures": len(failed) == 0,
        "unit_integrity": len(unit_failures) == 0,
        "mapping_integrity": len(mapping_failures) == 0,
        "catastrophic_quality": not cpu_catastrophic
        and not gpu_catastrophic,
        "minimum_core_jaccard": not low_core,
        "CPU_GPU_implementation_agreement": not implementation_failures,
        "W5_CPU_GPU_agreement": not w5_failures,
        "U6_GPU_V1_frozen_integrity": frozen["status"] == "pass",
        "legacy_frozen_integrity": legacy["status"] == "pass",
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "D1",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "registered": len(rows),
            "success": len(success),
            "explicitly_unavailable": len(unavailable),
            "failed": len(failed),
            "unit_failures": len(unit_failures),
            "mapping_failures": len(mapping_failures),
            "cpu_catastrophic": len(cpu_catastrophic),
            "gpu_catastrophic": len(gpu_catastrophic),
            "below_core_jaccard_0_5": len(low_core),
            "CPU_GPU_implementation_failures": len(
                implementation_failures
            ),
            "W5_CPU_GPU_failures": len(w5_failures),
        },
        "failure_trajectory_ids": {
            "processing": [row["trajectory_id"] for row in failed],
            "CPU_catastrophic": [
                row["trajectory_id"] for row in cpu_catastrophic
            ],
            "GPU_catastrophic": [
                row["trajectory_id"] for row in gpu_catastrophic
            ],
            "CPU_GPU_implementation": [
                row["trajectory_id"] for row in implementation_failures
            ],
            "W5_CPU_GPU": [
                row["trajectory_id"] for row in w5_failures
            ],
        },
        "D2_authorized": all(checks.values()),
        "D2_executed": False,
    }


def main() -> None:
    """Finalize D1 and stop before D2."""

    rows = load_checkpoints()
    write_tsv(D1_ROOT / "d1_all_domain_matched.tsv", rows)
    success = [row for row in rows if row["status"] == "success"]
    cold = run_cold_probes()
    frozen = frozen_evidence_check()
    legacy = verify_hashes()
    gate = gate_status(rows, frozen, legacy)
    write_json(D1_ROOT / "d1_gate_summary.json", gate)
    write_json(D1_ROOT / "frozen_integrity_after_D1.json", frozen)
    write_json(D1_ROOT / "legacy_frozen_integrity_after_D1.json", legacy)
    write_json(
        D1_ROOT / "d2_authorization_status.json",
        {
            "authorized": gate["D2_authorized"],
            "D1_gate_pass": gate["gate_pass"],
            "D2_executed": False,
            "instruction": (
                "D2 may begin only after human review of the D1 report."
                if gate["D2_authorized"]
                else "D2 is blocked because D1 quality/integrity did not pass."
            ),
        },
    )

    distribution_fields = (
        "cpu_core_jaccard_vs_exact",
        "gpu_core_jaccard_vs_exact",
        "cpu_weight_spearman_vs_exact",
        "gpu_weight_spearman_vs_exact",
        "cpu_rmsf_spearman_vs_exact",
        "gpu_rmsf_spearman_vs_exact",
        "cpu_rmsf_mae_vs_exact_A",
        "gpu_rmsf_mae_vs_exact_A",
        "cpu_flexible_label_agreement_vs_exact",
        "gpu_flexible_label_agreement_vs_exact",
        "cpu_median_absolute_rmsd_difference_vs_exact_A",
        "gpu_median_absolute_rmsd_difference_vs_exact_A",
        "cpu_p95_absolute_rmsd_difference_vs_exact_A",
        "gpu_p95_absolute_rmsd_difference_vs_exact_A",
        "cpu_gpu_core_jaccard",
        "cpu_gpu_weight_spearman",
        "cpu_gpu_rmsf_spearman",
        "w5_cpu_gpu_local_deformation_spearman",
        "cpu_throughput_loaded_compute_seconds",
        "gpu_throughput_loaded_compute_seconds",
        "cpu_w5_loaded_compute_seconds",
        "gpu_w5_loaded_compute_seconds",
        "input_decode_preprocessing_seconds",
        "task_end_to_end_seconds",
        "peak_RSS_mb",
        "gpu_throughput_peak_VRAM_bytes",
        "gpu_w5_peak_VRAM_bytes",
    )
    distributions = [
        summary_stats(success, field) for field in distribution_fields
    ]
    write_tsv(D1_ROOT / "d1_quality_timing_distributions.tsv", distributions)
    write_tsv(
        D1_ROOT / "d1_stratified_quality_timing.tsv",
        stratified_rows(success),
    )
    warning_rows = [
        row
        for row in rows
        if row["status"] != "success" or truth(row.get("quality_warning", False))
    ]
    if warning_rows:
        write_tsv(D1_ROOT / "d1_failure_warning_atlas.tsv", warning_rows)
    else:
        write_tsv(
            D1_ROOT / "d1_failure_warning_atlas.tsv",
            [
                {
                    "status": "none",
                    "trajectory_id": "",
                    "interpretation": "No failures or quality warnings.",
                }
            ],
        )

    med = {
        row["field"]: row["median"]
        for row in distributions
    }
    report = f"""# D1 corrected mdCATH all-domain matched panel

## Gate outcome

- D1 gate: **{'PASS' if gate['gate_pass'] else 'FAIL'}**
- Registered domains: **{len(rows):,}**
- Successful eligible domains: **{len(success):,}/{EXPECTED_ELIGIBLE:,}**
- Explicitly unavailable retained input: **{gate['counts']['explicitly_unavailable']}**
- Processing/unit/mapping failures: **{gate['counts']['failed']} / {gate['counts']['unit_failures']} / {gate['counts']['mapping_failures']}**
- CPU/GPU implementation-agreement failures: **{gate['counts']['CPU_GPU_implementation_failures']}**
- CPU/GPU W5 agreement failures: **{gate['counts']['W5_CPU_GPU_failures']}**
- Catastrophic CPU/GPU quality failures: **{gate['counts']['cpu_catastrophic']} / {gate['counts']['gpu_catastrophic']}**
- D2 full 134,950-trajectory run authorized: **{gate['D2_authorized']}**
- D2 executed in this phase: **no**

## Matched quality

The exact core weights were reused from the immutable unit-corrected
`sequential_lean` profiles. A common fixed-weight all-frame fit recomputed exact
transforms, RMSD, and RMSF for matched comparison; exact core identification
was not rerun.

- Median CPU/GPU throughput core Jaccard versus exact:
  **{float(med['cpu_core_jaccard_vs_exact']):.6f} /
  {float(med['gpu_core_jaccard_vs_exact']):.6f}**
- Median CPU/GPU weight Spearman versus exact:
  **{float(med['cpu_weight_spearman_vs_exact']):.6f} /
  {float(med['gpu_weight_spearman_vs_exact']):.6f}**
- Median CPU/GPU RMSF Spearman versus exact:
  **{float(med['cpu_rmsf_spearman_vs_exact']):.6f} /
  {float(med['gpu_rmsf_spearman_vs_exact']):.6f}**
- Median CPU/GPU per-system RMSD difference versus exact:
  **{float(med['cpu_median_absolute_rmsd_difference_vs_exact_A']):.6f} /
  {float(med['gpu_median_absolute_rmsd_difference_vs_exact_A']):.6f} A**
- Median CPU/GPU implementation core Jaccard:
  **{float(med['cpu_gpu_core_jaccard']):.6f}**
- Median CPU/GPU W5 local-deformation Spearman:
  **{float(med['w5_cpu_gpu_local_deformation_spearman']):.6f}**

## Performance boundaries

- Median input/decode/preprocessing:
  **{float(med['input_decode_preprocessing_seconds']):.6f} s**
- Median already-loaded CPU/GPU throughput:
  **{float(med['cpu_throughput_loaded_compute_seconds']):.6f} /
  {float(med['gpu_throughput_loaded_compute_seconds']):.6f} s**
- Median already-loaded CPU/GPU W5:
  **{float(med['cpu_w5_loaded_compute_seconds']):.6f} /
  {float(med['gpu_w5_loaded_compute_seconds']):.6f} s**
- Forced fresh-process GPU subset: **{sum(row['status'] == 'success' for row in cold)}/{len(cold)} successful**

Input decode, loaded compute, host/device transfer, GPU compute, and task
end-to-end time are reported separately. GPU throughput used one persistent
CUDA worker. W5 remains an alignment-free local-deformation diagnostic and is
not treated as an MDLovoFit-equivalent alignment workflow.

## Scope and interpretation

This phase establishes broad all-domain robustness on one canonical 320 K/R1
trajectory per mdCATH domain. It does not establish full-database throughput,
real approximately 10k-frame continuous performance, or scaling above 800
residues. Those remain separate D2-D4 questions. No external-baseline
database-wide run was performed, and no universal-fastest or 71x claim is made.
"""
    (D1_ROOT / "d1_all_domain_report.md").write_text(report)
    print(json.dumps(gate, indent=2))
    if not gate["gate_pass"]:
        raise SystemExit("D1 gate failed; D2 remains blocked")


if __name__ == "__main__":
    main()
