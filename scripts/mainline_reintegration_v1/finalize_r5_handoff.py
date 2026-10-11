#!/usr/bin/env python3
"""Finalize the compact R0-R5 mainline-reintegration handoff."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile

from scripts.mainline_reintegration_v1.run_r0_inventory import (
    frozen_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/mainline_reintegration_v1"
R4 = ROOT / "results/project_completion_v2_unit_corrected"
FINAL = RESULT / "final"
HANDOFF = RESULT / "handoff"
ARCHIVE = RESULT / "mainline_reintegration_r0_r5_handoff_20260729.tar.gz"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_status() -> dict[str, object]:
    path = FINAL / "full_test_results.txt"
    text = path.read_text() if path.is_file() else ""
    match = re.search(
        r"(\d+) passed, 2 subtests passed in ([0-9.]+)s",
        text,
    )
    return {
        "command": (
            "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
            "NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. "
            "/home/jjxu/miniconda3/bin/python -m pytest -q"
        ),
        "status": "pass" if match else "failed_or_unavailable",
        "passed": int(match.group(1)) if match else None,
        "subtests_passed": 2 if match else None,
        "elapsed_seconds": float(match.group(2)) if match else None,
        "output_path": str(path.relative_to(ROOT)),
    }


def selected_files() -> list[Path]:
    source = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "docs/cli_reference.md",
        ROOT / "docs/CODEX_MDCoreAlign_reintegrate_W5_and_return_to_mainline.md",
        ROOT / "src/mdcorealign/__init__.py",
        ROOT / "src/mdcorealign/cli.py",
        ROOT / "src/mdcorealign/experimental.py",
        ROOT / "src/mdsuperpose/benchmark.py",
        ROOT / "src/mdsuperpose/window_kinematics.py",
        ROOT / "scripts/mdcath_benchmark/extract_ca_trajectory_from_mdcath_h5.py",
        ROOT / "scripts/mdcath_benchmark/run_mdcath_subset_benchmark.py",
        ROOT / "scripts/package_private_alpha_bundle.py",
        ROOT / "tests/mainline_reintegration_v1/test_experimental_modes.py",
    ]
    phase_scripts = sorted(
        (ROOT / "scripts/mainline_reintegration_v1").glob("*.py")
    )
    result_files = [
        *sorted((RESULT / "baseline").glob("*.json")),
        *sorted((RESULT / "baseline").glob("*.md")),
        *sorted((RESULT / "unit_provenance").glob("*")),
        *sorted((RESULT / "local_deformation_integration").glob("*")),
        *sorted((RESULT / "fast_global_integration").glob("*")),
        RESULT / "02_03_status.json",
        RESULT / "02_03_integrity_after.json",
        *sorted((RESULT / "matched_benchmarks").glob("*")),
        R4 / "04_r4_status.json",
        R4 / "04_unit_corrected_evidence_report.md",
        R4 / "aggregate/all_domain_unit_corrected_overall.tsv",
        R4 / "aggregate/all_domain_unit_corrected_by_cath_class.tsv",
        R4 / "clean_parity/clean_parity_summary.json",
        R4 / "fast_global/fast_global_distribution_summary.tsv",
        R4 / "fast_global/fast_global_catastrophic_cases.tsv",
        *sorted(FINAL.glob("*")),
    ]
    return sorted(
        {
            path
            for path in [*source, *phase_scripts, *result_files]
            if path.is_file() and path != ARCHIVE
        }
    )


def write_final_reports(
    integrity: dict[str, object],
    tests: dict[str, object],
) -> None:
    r4 = json.loads((R4 / "04_r4_status.json").read_text())
    r5 = json.loads(
        (RESULT / "matched_benchmarks/05_r5_status.json").read_text()
    )
    speed = {
        row["comparison"]: row
        for row in read_tsv(
            RESULT / "matched_benchmarks/matched_speed_claims.tsv"
        )
    }
    contract1 = read_tsv(
        RESULT / "matched_benchmarks/contract1_runtime_summary.tsv"
    )
    contract2 = {
        row["method"]: row
        for row in read_tsv(
            RESULT / "matched_benchmarks/contract2_runtime_summary.tsv"
        )
    }

    def c1(method: str, boundary: str, chunk: str = "2000") -> float:
        row = next(
            item
            for item in contract1
            if item["method"] == method
            and item["timing_boundary"] == boundary
            and item["descriptor_chunk_size"] == chunk
        )
        return float(row["median_runtime_seconds"])

    corrected = r4["corrected_all_domain"]
    fast = r4["fast_global"]
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "mainline_reintegration_decision.md").write_text(
        f"""# Mainline reintegration decision after R5

## Decision

R0-R5 are complete. The exact `sequential_lean` implementation remains the
public default. Uniform low-budget fast-global alignment, W5 local-deformation
analysis, and diagnostic-rich composition are available only through explicit
experimental interfaces.

W5 is retained as an alignment-free, dimensionless local-deformation
diagnostic. Its score is not RMSD or global mobility. The sparse motion graph
is exposed only as soft co-motion/hinge evidence; hard biological-domain labels
are not presented as mature output.

## Evidence status

- Formal unit provenance is resolved from source HDF5 evidence.
- Corrected all-domain root:
  `results/project_completion_v2_unit_corrected/`.
- Clean-source parity: {r4['clean_parity']['n_pass']}/
  {r4['clean_parity']['n_cases']} passed.
- Corrected MDCoreAlign/MDLovoFit paired successes:
  {corrected['n_success_both_mdcorealign_mdlovofit']}.
- Corrected all-three successes: {corrected['n_success_all_three']}.
- Frozen integrity: {integrity['status']} (
  {integrity['project_completion_checked']}/9 project outputs,
  {integrity['golden_checked']}/34 golden references).
- Tests: {tests['passed']} passed and {tests['subtests_passed']} subtests passed.

## Warnings

- Fast-global remains experimental: {fast['catastrophic_failure_count']} R4
  all-domain tail cases met the predefined catastrophic threshold, and
  {fast['fixed_budget_not_converged_count']} runs reached the fixed budget
  without the strict convergence label.
- One corrected affected case (`1kmdA00__T320__R1`) reproducibly failed THESEUS
  GSL Jacobi SVD; it was not imputed.
- R6-R8 were not executed.
- No GPU backend was implemented or benchmarked.
- No Git metadata was initialized or modified.
"""
    )
    (FINAL / "unit_correction_and_evidence_status.md").write_text(
        f"""# Unit correction and evidence status

The provenance audit classified 162 formal records as accidentally scaled by
ten, 10 derived stitching records as affected descendants, and three cases as
deliberate scale controls. Direct source audits confirmed formal mdCATH HDF5
coordinates in Angstrom. Deliberate scale controls remain synthetic
arbitrary-scale tests and are excluded from absolute-Angstrom biological
summaries.

The corrected immutable evidence root contains {corrected['n_target_domains']}
target domains. MDCoreAlign and MDLovoFit each succeeded on
{corrected['n_success_both_mdcorealign_mdlovofit']} paired cases; all three
methods succeeded on {corrected['n_success_all_three']} cases. Median unified
RMSD was {float(corrected['median_mdcorealign_unified_RMSD_A']):.4f} A for
MDCoreAlign, {float(corrected['median_mdlovofit_unified_RMSD_A']):.4f} A for
MDLovoFit, and {float(corrected['median_theseus_unified_RMSD_A']):.4f} A for
THESEUS on their documented success sets.

The 100-case clean parity panel passed 100/100. Maximum exact RMSD and RMSF
differences were below 5e-9 A. Frozen historical outputs were not rewritten.
"""
    )
    (FINAL / "local_deformation_contract_decision.md").write_text(
        f"""# Local-deformation contract decision

On the unit-clean 10k/153-residue transition-aligned synthetic stress case:

- loaded W5 scan-only median: {c1('w5_scan_only', 'already_loaded_compute'):.4f} s;
- persistent-worker W5 median:
  {c1('w5_scan_only', 'warm_persistent_worker_compute'):.4f} s;
- fresh-process W5 median:
  {c1('w5_scan_only', 'fresh_process_decode_compute'):.4f} s;
- loaded W5 plus sparse graph median:
  {c1('w5_sparse_graph', 'already_loaded_compute'):.4f} s.

These outputs do not include transforms, RMSD, or RMSF and are not compared as
MDLovoFit-equivalent work. W5 remains experimental but scientifically
supported as a rigid-transform- and scale-invariant local-deformation
diagnostic. The sparse graph remains a soft exploratory co-motion/hinge layer.
"""
    )
    (FINAL / "fast_global_contract_decision.md").write_text(
        f"""# Fast-global contract decision

The all-domain corrected panel completed fast-global for
{fast['n_success']} cases. Median core Jaccard was
{fast['median_core_jaccard']:.4f}, median weight Spearman was
{fast['median_weight_spearman']:.4f}, median RMSF Spearman was
{fast['median_RMSF_spearman']:.4f}, and median absolute unified-RMSD difference
from exact was {fast['median_absolute_RMSD_difference_A']:.4f} A.

Promotion is not justified. The panel contained
{fast['catastrophic_failure_count']} predefined tail failures and a 95th
percentile absolute RMSD difference of
{fast['p95_absolute_RMSD_difference_A']:.4f} A. Fast-global therefore remains
an explicit experimental fixed-budget mode with confidence warnings; no
automatic fallback or silent default substitution is used.

On the clean 10k matched transforms/RMSD/RMSF contract, exact median runtime
was {float(contract2['sequential_lean_exact']['median_runtime_seconds']):.4f} s
and fast-global was
{float(contract2['uniform_fast_global']['median_runtime_seconds']):.4f} s.
"""
    )
    exact_mdlo = speed["MDLovoFit_over_exact_MDCoreAlign"]
    fast_mdlo = speed["MDLovoFit_over_fast_global"]
    exact_theseus = speed["THESEUS_over_exact_MDCoreAlign"]
    (FINAL / "matched_speed_claims.md").write_text(
        f"""# Matched speed claims

The formal materialized-output comparison uses 12 paired domains, three
measured repeats per method, and one common contract: loaded CA coordinates to
an aligned CA multi-model PDB.

- Median paired MDLovoFit/exact-MDCoreAlign runtime ratio:
  {float(exact_mdlo['median_paired_runtime_ratio']):.3f}x; MDLovoFit was slower
  on {float(exact_mdlo['fraction_numerator_slower']) * 12:.0f}/12 domains.
- Median paired MDLovoFit/experimental-fast-global ratio:
  {float(fast_mdlo['median_paired_runtime_ratio']):.3f}x; MDLovoFit was slower
  on {float(fast_mdlo['fraction_numerator_slower']) * 12:.0f}/12 domains.
- Median paired THESEUS/exact-MDCoreAlign runtime ratio:
  {float(exact_theseus['median_paired_runtime_ratio']):.3f}x; THESEUS was
  slower on {float(exact_theseus['fraction_numerator_slower']) * 12:.0f}/12
  domains.

Ratios above one favour the MDCoreAlign denominator. Exact MDCoreAlign was not
uniformly faster than either external method. External subprocess timings
include native fitting/search and mandatory native output generation; those
kernel and output components cannot be isolated without modifying the tools.

No 71x headline claim is supported or made. Historical cross-run arithmetic
with unmatched boundaries must remain exploratory context only.
"""
    )
    write_json(
        FINAL / "final_status.json",
        {
            "scope_completed": ["R0", "R1", "R2", "R3", "R4", "R5"],
            "scope_not_executed": ["R6", "R7", "R8"],
            "status": "pass_with_documented_warnings",
            "public_default": "sequential_lean_exact",
            "fast_global_status": "experimental",
            "w5_status": "experimental_local_deformation_diagnostic",
            "motion_graph_status": "experimental_soft_diagnostics",
            "r4_status": r4["status"],
            "r5_status": r5["status"],
            "integrity": integrity,
            "tests": tests,
            "git_metadata_modified": False,
            "gpu_implemented": False,
            "speedup_71x_claimed": False,
        },
    )


def build_handoff() -> tuple[str, int]:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    files = selected_files()
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    write_tsv(HANDOFF / "file_manifest.tsv", rows)
    (HANDOFF / "reproducible_commands.md").write_text(
        """# Reproducible R5 commands

```bash
cd /path/to/mdlovofit
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:.
PY=/home/jjxu/miniconda3/bin/python

$PY scripts/mainline_reintegration_v1/run_r5_matched_contracts.py contract1
$PY scripts/mainline_reintegration_v1/run_r5_matched_contracts.py contract2
$PY scripts/mainline_reintegration_v1/run_r5_matched_contracts.py contract3
$PY scripts/mainline_reintegration_v1/run_r5_matched_contracts.py persistent-batch
$PY scripts/mainline_reintegration_v1/run_r5_matched_contracts.py finalize
$PY -m pytest -q
```

The 10k input is transition-aligned synthetic scaling data, not a biologically
continuous trajectory. Contract 3 requires locally installed MDLovoFit and
THESEUS executables.
"""
    )
    files = selected_files()
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        archive.add(
            HANDOFF / "file_manifest.tsv",
            arcname=str((HANDOFF / "file_manifest.tsv").relative_to(ROOT)),
        )
        archive.add(
            HANDOFF / "reproducible_commands.md",
            arcname=str(
                (HANDOFF / "reproducible_commands.md").relative_to(ROOT)
            ),
        )
    digest = sha256(ARCHIVE)
    write_json(
        HANDOFF / "archive_status.json",
        {
            "archive": str(ARCHIVE.relative_to(ROOT)),
            "sha256": digest,
            "included_file_count": len(files) + 2,
            "raw_trajectories_included": False,
            "large_npz_or_hdf5_included": False,
            "cache_included": False,
            "secrets_included": False,
        },
    )
    return digest, len(files) + 2


def main() -> None:
    integrity = frozen_integrity()
    if integrity["status"] != "pass":
        raise RuntimeError("frozen reference integrity failed")
    tests = load_test_status()
    if tests["status"] != "pass":
        raise RuntimeError("captured full test result is missing or failed")
    write_final_reports(integrity, tests)
    status = json.loads((FINAL / "final_status.json").read_text())
    status["handoff_archive"] = {
        "path": str(ARCHIVE.relative_to(ROOT)),
        "sha256_record": str(
            (HANDOFF / "archive_status.json").relative_to(ROOT)
        ),
    }
    write_json(FINAL / "final_status.json", status)
    build_handoff()


if __name__ == "__main__":
    main()
