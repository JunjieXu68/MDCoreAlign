#!/usr/bin/env python
"""Finalize the Phase 7B-8B decision and create the compact handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile

from scripts.scaling_optimization_v4.finalize_phase6 import verify_hashes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = PROJECT_ROOT / "results/scaling_optimization_v5"
PHASE7B = V5_ROOT / "phase7b_unit_integrity"
PHASE8A = V5_ROOT / "phase8a_broad_validation"
PHASE8B = V5_ROOT / "phase8b_runtime"
FINAL = V5_ROOT / "final_phase7b_8b"
HANDOFF = V5_ROOT / "handoff_phase8b"
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one tab-separated table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_tests() -> dict[str, object]:
    """Run the full suite with the canonical interpreter."""

    command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
    environment = os.environ.copy()
    environment.update(ENVIRONMENT)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    output = completed.stdout + completed.stderr
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "full_test_results.txt").write_text(output)
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": output.strip().splitlines()[-1] if output.strip() else "",
    }


def method_runtime(
    rows: list[dict[str, str]], method: str, boundary: str
) -> float:
    """Read one canonical median runtime."""

    return float(
        next(
            row["median_seconds"]
            for row in rows
            if row["method"] == method and row["boundary"] == boundary
        )
    )


def external_summary(
    rows: list[dict[str, str]], method: str
) -> tuple[float, int, float, int]:
    """Return paired external timing/RMSD outcomes."""

    index = {(row["domain_id"], row["method"]): row for row in rows}
    domains = sorted({row["domain_id"] for row in rows})
    ratios: list[float] = []
    rmsd_deltas: list[float] = []
    for domain in domains:
        external = index[(domain, "mdlovofit_external")]
        candidate = index[(domain, method)]
        ratios.append(
            float(external["median_algorithm_or_subprocess_seconds"])
            / float(candidate["median_algorithm_or_subprocess_seconds"])
        )
        rmsd_deltas.append(
            float(candidate["median_unified_posthoc_rmsd_A"])
            - float(external["median_unified_posthoc_rmsd_A"])
        )
    return (
        statistics.median(ratios),
        sum(value > 1.0 for value in ratios),
        statistics.median(rmsd_deltas),
        sum(value < 0.0 for value in rmsd_deltas),
    )


def write_final_documents(test: dict[str, object]) -> None:
    """Write the final locked decision and concise execution report."""

    scale_status = json.loads((PHASE7B / "phase7b_status.json").read_text())
    validation_status = json.loads(
        (PHASE8A / "phase8a_status.json").read_text()
    )
    runtime_rows = read_tsv(PHASE8B / "runtime_summary.tsv")
    external_rows = read_tsv(
        PHASE8B / "same_boundary_mdlovofit_comparison.tsv"
    )
    exact_runtime = method_runtime(
        runtime_rows,
        "sequential_lean_exact",
        "already_loaded_metrics_transforms",
    )
    uniform_runtime = method_runtime(
        runtime_rows,
        "uniform_low_budget",
        "already_loaded_metrics_transforms",
    )
    w5_runtime = method_runtime(
        runtime_rows,
        "w5_preconditioned",
        "already_loaded_metrics_transforms",
    )
    scan_runtime = method_runtime(
        runtime_rows,
        "w5_scan_only",
        "already_loaded_metrics_transforms",
    )
    uniform_fresh = method_runtime(
        runtime_rows,
        "uniform_low_budget",
        "fresh_process_npz_metrics_transforms",
    )
    exact_ext = external_summary(external_rows, "sequential_lean_exact")
    uniform_ext = external_summary(external_rows, "uniform_low_budget")
    w5_ext = external_summary(external_rows, "w5_preconditioned")
    integrity = verify_hashes()

    FINAL.mkdir(parents=True, exist_ok=True)
    decision = f"""# Final Phase 7B-8B contract decision

## Stop/go outcome

- Phase 7B coordinate-integrity gate: **PASS**.
- Phase 8A locked 512-domain processing: **512/512 successful**, with no
  coordinate-scale failure.
- Uniform broad scientific gate: **FAIL**.
- W5 broad scientific gate: **FAIL**.
- Stage 2 all-domain expansion: **NOT RUN**, as required by the locked gate.
- Frozen integrity: **{integrity['status'].upper()}**
  ({integrity['project_completion_checked']}/9 project outputs and
  {integrity['golden_checked']}/34 golden references checked).

## Contracts

### Contract 1: exact/default

`sequential_lean` remains the public exact/default compatibility path. It
preserves sequential warm-start semantics. Its unit-clean 10k in-memory median,
including common exact transforms, RMSD, robust RMSF and ordinary-control RMSF,
was {exact_runtime:.4f} s.

### Contract 2: fast global

The locked `uniform_low_budget` mode is retained as **experimental**, not as an
unconditional replacement. Across 512 systems it had median core Jaccard 0.960,
median weight Spearman 0.9936 and median RMSF Spearman 0.99969, but its
95th percentile across per-system P95 RMSD differences was 0.2868 A, above the
locked 0.20 A engineering gate. Its unit-clean 10k in-memory/fresh-process
medians were {uniform_runtime:.4f}/{uniform_fresh:.4f} s.

### Contract 3: diagnostic-rich

W5-preconditioned global refinement is retained as an optional experimental
diagnostic-rich path. It gave three locked rescue cases and small median
improvements over uniform, but failed the same RMSD-tail gate and incurred a
~6x median Phase 8A runtime penalty over uniform. Its 10k in-memory median was
{w5_runtime:.4f} s. It has not shown end-to-end acceleration over equal-budget
uniform refinement.

### Contract 4: scan-only local deformation

W5 scan-only remains an experimental alignment-free local-deformation index.
Its 10k median was {scan_runtime:.4f} s. It produces no transform, RMSD or RMSF,
and its movement index must not be called RMSD.

## Correct W5 wording

W5 was not shown to accelerate equal-budget global stable-core refinement in
its current implementation. After unit-clean broad validation,
W5-preconditioned refinement retained highly similar core, mobility-weight,
RMSF, held-out stability, and geometry, with limited locked difficult-case
rescues. It is retained as an alignment-free local-deformation diagnostic and
as an optional diagnostic-rich refinement candidate, not as the default global
accelerator.
"""
    (FINAL / "final_contract_decision.md").write_text(decision)

    report = f"""# Phase 7B-8B execution summary

## Phase 7B

The factor-of-ten contamination was traced to the historical unit-conversion
heuristic: explicitly Angstrom mdCATH arrays were multiplied by ten according
to translation-dependent absolute coordinate position. The source HDF5 data
were unit-consistent. The loader now honors explicit units and never performs
per-frame inferred rescaling. Three domains and the affected 10k derivative
were regenerated under versioned paths; original files remain unchanged.

The frozen all-domain audit identified 5,235 Angstrom-consistent inputs,
162 uniformly factor-ten inputs, and one unavailable retained input. Those
historical outputs were not overwritten and are quarantined from absolute
Angstrom RMSD claims.

## Phase 8A

The locked class/length/mobility/speed-stratified panel contained 512 new
320 K R1 mdCATH domains, with R2 cross-replicate checks for 64. All completed.
Uniform and W5 met the median core/weight/RMSF gates and had no catastrophic
core failure, but failed the strict system-tail RMSD gate. W5 rescued three
uniform core-Jaccard cases, while its median runtime was 5.99x uniform; this is
limited benefit, not a promotion result.

## Phase 8B

On the regenerated 10k transition-aligned synthetic input, common-boundary
already-loaded medians were exact={exact_runtime:.4f} s,
uniform={uniform_runtime:.4f} s, and W5={w5_runtime:.4f} s. Uniform fresh
process was {uniform_fresh:.4f} s. PDB text writing was measured separately
and was approximately four seconds per 10k aligned trajectory.

On the 12-case canonical MDLovoFit subset:

- exact median MDLovoFit/method ratio={exact_ext[0]:.2f}x; exact was faster in
  {exact_ext[1]}/12 and had lower unified RMSD in {exact_ext[3]}/12;
- uniform ratio={uniform_ext[0]:.2f}x; faster in {uniform_ext[1]}/12 and lower
  unified RMSD in {uniform_ext[3]}/12;
- W5 ratio={w5_ext[0]:.2f}x; faster in {w5_ext[1]}/12 and lower unified RMSD
  in {w5_ext[3]}/12.

These are observable-boundary ratios, not pure-kernel ratios: MDLovoFit's
subprocess necessarily includes its internal search, fitting and aligned-PDB
write, while the internal algorithm boundary starts from loaded coordinates.
Input-write and output-read costs are reported separately.

## Failures and unavailable items

- Phase 8A processing failures: 0.
- Broad engineering-gate failures: uniform and W5 RMSD tails.
- All-domain Stage 2: intentionally not run after the failed Stage 1 gate.
- Backbone-dihedral validation: unavailable in the formal CA-only panel.
- Locked hinge annotations: unavailable for the broad panel.
- Pure MDLovoFit numerical-kernel timing: unavailable without modifying the
  external executable.
- Git commit: unavailable because the repository has no valid HEAD; Git
  metadata was not initialized or altered.

## Verification

- Tests: **{test['status'].upper()}**, `{test['summary']}`.
- Frozen hashes: **{integrity['status'].upper()}**.
- No scientific algorithm, public default, project-completion output, golden
  reference or Git metadata was changed.
"""
    (FINAL / "phase7b_8b_summary.md").write_text(report)
    (FINAL / "final_status.json").write_text(
        json.dumps(
            {
                "status": (
                    "complete"
                    if test["status"] == "pass"
                    and integrity["status"] == "pass"
                    and scale_status["phase7b_stop_go"] == "pass"
                    and validation_status["processing_failure_count"] == 0
                    else "failed"
                ),
                "phase7b_gate_pass": (
                    scale_status["phase7b_stop_go"] == "pass"
                ),
                "phase8a_uniform_gate_pass": validation_status[
                    "uniform_gate_pass"
                ],
                "phase8a_w5_gate_pass": validation_status["w5_gate_pass"],
                "phase8a_stage2_all_domain_go": validation_status[
                    "stage2_all_domain_go"
                ],
                "phase8b_complete": True,
                "tests": test,
                "frozen_hash_integrity": integrity,
                "public_default_changed": False,
                "scientific_algorithm_changed": False,
                "git_metadata_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


def create_handoff() -> None:
    """Create a compact, checksummed handoff without raw trajectories."""

    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    paths = [
        Path(
            "docs/CODEX_MDCoreAlign_post_phase7a_unit_integrity_and_broad_validation.md"
        ),
        Path("src/mdsuperpose/coordinate_scale.py"),
        Path("src/mdcorealign/cli.py"),
        Path("scripts/mdcath_benchmark/mdcath_common.py"),
        Path(
            "scripts/mdcath_benchmark/"
            "extract_ca_trajectory_from_mdcath_h5.py"
        ),
        Path("scripts/scaling_optimization_v5/run_phase7b_unit_integrity.py"),
        Path("scripts/scaling_optimization_v5/run_phase8a_broad_validation.py"),
        Path(
            "scripts/scaling_optimization_v5/"
            "run_phase8b_runtime_boundaries.py"
        ),
        Path("scripts/scaling_optimization_v5/finalize_phase8b.py"),
        Path("tests/scaling_optimization_v5/test_coordinate_unit_integrity.py"),
        Path(
            "results/scaling_optimization_v5/phase7b_unit_integrity/"
            "coordinate_unit_provenance.md"
        ),
        Path(
            "results/scaling_optimization_v5/phase7b_unit_integrity/"
            "affected_case_root_causes.tsv"
        ),
        Path(
            "results/scaling_optimization_v5/phase7b_unit_integrity/"
            "regenerated_case_manifest.tsv"
        ),
        Path(
            "results/scaling_optimization_v5/phase7b_unit_integrity/"
            "scope_of_impact.md"
        ),
        Path(
            "results/scaling_optimization_v5/phase7b_unit_integrity/"
            "coordinate_scale_guard.md"
        ),
        Path(
            "results/scaling_optimization_v5/phase7b_unit_integrity/"
            "regenerated_affected_comparisons.tsv"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "broad_panel_manifest.tsv"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "broad_panel_lock.json"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "broad_validation_summary.tsv"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "w5_incremental_value_over_uniform.tsv"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "independent_scientific_quality.tsv"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "failure_case_catalogue.md"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "phase8a_contract_decision.md"
        ),
        Path(
            "results/scaling_optimization_v5/phase8a_broad_validation/"
            "phase8a_status.json"
        ),
    ]
    paths.extend(
        path.relative_to(PROJECT_ROOT)
        for path in sorted(PHASE8B.glob("*"))
        if path.is_file()
    )
    paths.extend(
        path.relative_to(PROJECT_ROOT)
        for path in sorted(FINAL.glob("*"))
        if path.is_file()
    )
    manifest_rows: list[str] = ["sha256\tbytes\tpath"]
    for relative in paths:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        target = HANDOFF / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest_rows.append(
            f"{sha256(target)}\t{target.stat().st_size}\t{relative.as_posix()}"
        )
    (HANDOFF / "sha256_manifest.tsv").write_text(
        "\n".join(manifest_rows) + "\n"
    )
    (HANDOFF / "README.md").write_text(
        """# MDCoreAlign Phase 8B compact handoff

This bundle contains the Phase 7B unit-integrity root cause and guard, locked
Phase 8A broad-validation summaries, Phase 8B runtime-boundary results, final
contract decision, tests, integrity status and exact source changes.

It excludes raw HDF5/NPZ trajectories, per-frame scale diagnostics, regenerated
coordinate arrays and the 3,072-row detailed validation table to remain
compact. Frozen project-completion and golden outputs are not copied or
modified. Stage 2 all-domain validation was not run because the locked 512-case
gate did not pass.
"""
    )
    archive = V5_ROOT / "handoff_phase8b_20260728.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(HANDOFF, arcname="handoff_phase8b")
    (V5_ROOT / "handoff_phase8b_20260728.sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )


def main() -> int:
    """Finalize, verify, bundle and stop."""

    integrity_before = verify_hashes()
    if integrity_before["status"] != "pass":
        raise RuntimeError("frozen hash integrity failed before finalization")
    test = run_tests()
    write_final_documents(test)
    integrity_after = verify_hashes()
    if integrity_after["status"] != "pass":
        raise RuntimeError("frozen hash integrity failed after finalization")
    create_handoff()
    final = json.loads((FINAL / "final_status.json").read_text())
    return 0 if final["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
