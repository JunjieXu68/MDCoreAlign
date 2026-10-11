#!/usr/bin/env python
"""Finalize the W0-W5 pilot without modifying frozen evidence or Git metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
RESULT_ROOT = PROJECT_ROOT / "results/window_motion_domains"
W0_ROOT = RESULT_ROOT / "w0_scale_provenance"
PILOT_ROOT = RESULT_ROOT / "pilot_validation"
FINAL_ROOT = RESULT_ROOT / "final"
HANDOFF_ROOT = RESULT_ROOT / "handoff_window_motion_domain_pilot"
ARCHIVE = RESULT_ROOT / "handoff_window_motion_domain_pilot_20260729.tar.gz"
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONPATH": "src:.",
}
HANDOFF_FILES = (
    "src/mdsuperpose/window_kinematics.py",
    "scripts/window_motion_domains/run_w0_scale_provenance.py",
    "scripts/window_motion_domains/run_window_motion_domain_pilot.py",
    "scripts/window_motion_domains/finalize_window_motion_domain_pilot.py",
    "tests/window_motion_domains/test_window_kinematics.py",
    "results/window_motion_domains/w0_scale_provenance/"
    "scale_case_provenance.tsv",
    "results/window_motion_domains/w0_scale_provenance/"
    "source_hdf5_unit_evidence.tsv",
    "results/window_motion_domains/w0_scale_provenance/"
    "w5_scale_invariance.tsv",
    "results/window_motion_domains/w0_scale_provenance/"
    "scale_provenance_adjudication.md",
    "results/window_motion_domains/w0_scale_provenance/w0_status.json",
    "results/window_motion_domains/pilot_validation/synthetic_recovery.tsv",
    "results/window_motion_domains/pilot_validation/"
    "real_panel_reproducibility.tsv",
    "results/window_motion_domains/pilot_validation/"
    "heldout_independent_quality.tsv",
    "results/window_motion_domains/pilot_validation/"
    "secondary_sequential_comparison.tsv",
    "results/window_motion_domains/pilot_validation/performance_summary.tsv",
    "results/window_motion_domains/pilot_validation/"
    "window_motion_domain_pilot_report.md",
    "results/window_motion_domains/pilot_validation/pilot_status.json",
    "results/window_motion_domains/final/w0_w5_summary.md",
    "results/window_motion_domains/final/"
    "window_motion_domain_contract_decision.md",
    "results/window_motion_domains/final/frozen_hash_integrity.json",
    "results/window_motion_domains/final/full_test_results.txt",
    "results/window_motion_domains/final/final_status.json",
)


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one TSV."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a TSV."""

    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def verify_hashes() -> dict[str, object]:
    """Verify frozen project-completion and golden-reference files."""

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
            / "results/scaling_optimization_v2/regression/"
            "golden_manifest.tsv"
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


def run_tests() -> dict[str, object]:
    """Run the complete test suite in the canonical environment."""

    command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
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
    (FINAL_ROOT / "full_test_results.txt").write_text(output)
    summary = (completed.stdout.strip().splitlines() or [""])[-1]
    match = re.search(
        r"(?P<tests>\d+) passed(?:, (?P<subtests>\d+) subtests passed)?",
        summary,
    )
    return {
        "command": (
            "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
            "NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. "
            "/home/jjxu/miniconda3/bin/python -m pytest -q tests"
        ),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": summary,
        "tests_passed": int(match.group("tests")) if match else None,
        "subtests_passed": (
            int(match.group("subtests"))
            if match and match.group("subtests")
            else 0
        ),
    }


def performance_index() -> dict[tuple[str, str], dict[str, str]]:
    """Index pilot performance summaries."""

    return {
        (row["method"], row["boundary"]): row
        for row in read_tsv(PILOT_ROOT / "performance_summary.tsv")
    }


def write_final_documents(
    tests: dict[str, object],
    integrity: dict[str, object],
) -> None:
    """Write the final conservative summary and contract decision."""

    w0 = json.loads((W0_ROOT / "w0_status.json").read_text())
    pilot = json.loads((PILOT_ROOT / "pilot_status.json").read_text())
    performance = performance_index()
    loaded = {
        method: float(
            performance[(method, "already_loaded")][
                "median_runtime_seconds"
            ]
        )
        for method in (
            "w5_scan_only",
            "w5_motion_domain_graph",
            "optional_domain_alignment",
            "sequential_lean",
            "uniform_fast_global",
        )
    }
    fresh = {
        method: float(
            performance[
                (method, "fresh_process_including_npz_decode")
            ]["median_runtime_seconds"]
        )
        for method in loaded
    }
    report = (PILOT_ROOT / "window_motion_domain_pilot_report.md").read_text()
    metric = re.compile(r"^- median (?P<name>.+?): (?P<value>[0-9.]+);?$", re.M)
    reported = {
        match.group("name"): match.group("value")
        for match in metric.finditer(report)
    }
    summary = f"""# W0-W5 window-motion-domain pilot summary

## W0: scale provenance

- Historical scale-x10 rows adjudicated: {w0['historical_affected_rows']}.
- Formal biological rows with accidental conversion: {w0['formal_biological_rows']}.
- Frame-count/stitching stress rows with accidental conversion:
  {w0['frame_stitching_stress_rows']}.
- Deliberate arbitrary-scale controls: {w0['deliberate_scale_test_controls']}.
- W5 scale/rigid-transform invariance: **PASS**.
- Formal all-domain rerun: **not performed**.

The factor-ten values were produced by an accidental historical conversion,
not by deliberate scaling of formal mdCATH inputs. Deliberate 0.1x/1x/10x
cases are synthetic arbitrary-scale invariance controls only and are excluded
from absolute-Angstrom biological summaries.

## W1-W4: scientific pilot

- Required synthetic controls: {pilot['synthetic_pass_count']}/
  {pilot['synthetic_case_count']} passed.
- Real CA-only validation panel: {pilot['real_domain_count']} mdCATH domains,
  each with R1/R2 cross-replicate analysis.
- Median temporal local-index Spearman:
  {reported.get('temporal-block local-index Spearman', 'see source table')}.
- Median cross-replicate local-index Spearman:
  {reported.get('cross-replicate local-index Spearman', 'see source table')}.
- Median temporal boundary Jaccard:
  {pilot['median_temporal_boundary_jaccard']:.4f}.
- Median cross-replicate boundary Jaccard:
  {pilot['median_cross_replicate_boundary_jaccard']:.4f}.
- Backbone-dihedral validation: unavailable for CA-only inputs.

Continuous local-deformation signals showed moderate reproducibility, while
discrete motion-domain boundaries were weakly reproducible. The graph output is
therefore an experimental diagnostic/hypothesis, not a validated biological
domain annotation.

## W5: performance

Canonical already-loaded medians on the unit-clean 10k/153-residue
transition-aligned synthetic stress case:

| Method | Loaded median (s) | Fresh-process median (s) |
|---|---:|---:|
| W5 scan only | {loaded['w5_scan_only']:.4f} | {fresh['w5_scan_only']:.4f} |
| W5 + motion-domain graph | {loaded['w5_motion_domain_graph']:.4f} | {fresh['w5_motion_domain_graph']:.4f} |
| Graph + optional domain alignment | {loaded['optional_domain_alignment']:.4f} | {fresh['optional_domain_alignment']:.4f} |
| sequential_lean | {loaded['sequential_lean']:.4f} | {fresh['sequential_lean']:.4f} |
| uniform fast-global | {loaded['uniform_fast_global']:.4f} | {fresh['uniform_fast_global']:.4f} |

The W5+graph loaded target below 0.75 s passed, but uniform fast-global remained
faster. The stress case is not biologically continuous MD.

## Integrity

- Tests: **{tests['status'].upper()}**, `{tests['summary']}`.
- Frozen hashes: **{integrity['status'].upper()}**,
  {integrity['project_completion_checked']}/
  {integrity['project_completion_checked']} project outputs and
  {integrity['golden_checked']}/{integrity['golden_checked']} golden
  references checked.
- Public default: unchanged (`sequential_lean`).
- Git metadata: unchanged; the repository still has no valid `HEAD`.
"""
    (FINAL_ROOT / "w0_w5_summary.md").write_text(summary)

    decision = """# Experimental window-motion-domain contract decision

## Decision

The W0 provenance gate passed. The W1-W5 pilot is **partially successful**:
synthetic recovery/invariance and the loaded CPU performance target passed,
but real-system discrete motion-domain boundaries were not sufficiently
reproducible for promotion.

## Public contract

`sequential_lean` remains the exact/public default. No public import, CLI
default, manuscript evidence, project-completion output, golden reference, or
Git metadata was changed.

## Experimental contracts

1. **Local-deformation diagnostic:** W5 scan-only may be retained
   experimentally as a translation/rotation-invariant, scale-normalised,
   robust local-deformation index. It is not RMSD or global mobility.
2. **Motion-domain hypothesis:** the sparse local/contact/anchor graph may
   generate candidate coherent domains and hinge boundaries, but its current
   real-panel boundary reproducibility is weak. Domain labels require further
   validation before biological interpretation.
3. **Optional anchor alignment:** exact transforms, RMSD, and RMSF may be
   computed only after explicit largest-stable, least-deforming, or
   user-specified domain selection. These are downstream geometric audits.

W5 was not selected as a stable-core acceleration preconditioner in its current
implementation, but is retained as an experimental alignment-free
local-deformation diagnostic. The present primary-algorithm branch adds a
separate candidate motion-domain interpretation; it does not establish a
validated replacement for global stable-core alignment.

## Stop condition

No all-domain rerun, GPU backend, Faiss/Annoy path, further W5 tuning, or
global-alignment optimisation was performed. Review is required before any
conditional extension beyond this pilot.
"""
    (FINAL_ROOT / "window_motion_domain_contract_decision.md").write_text(
        decision
    )

    git_check = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = {
        "work_package": "scale_provenance_and_window_motion_domain_pilot",
        "overall_status": "partial",
        "w0_status": w0["status"],
        "synthetic_gate_pass": pilot["synthetic_gate_pass"],
        "performance_gate_pass": pilot["performance_target_pass"],
        "real_boundary_reproducibility_warning": (
            pilot["real_boundary_reproducibility_warning"]
        ),
        "tests": tests,
        "frozen_hash_integrity": integrity["status"],
        "git_head_available": git_check.returncode == 0,
        "formal_all_domain_rerun_performed": False,
        "public_default_changed": False,
        "frozen_evidence_modified": False,
        "conditional_phase_executed": False,
    }
    (FINAL_ROOT / "final_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


def build_handoff() -> None:
    """Build a compact, checksummed handoff directory and archive."""

    if HANDOFF_ROOT.exists() or ARCHIVE.exists():
        raise RuntimeError(
            "handoff output already exists; refusing to overwrite it"
        )
    HANDOFF_ROOT.mkdir(parents=True)
    for relative in HANDOFF_FILES:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        target = HANDOFF_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    readme = """# Window-motion-domain pilot handoff

This compact bundle contains the W0 provenance adjudication, isolated
experimental W1-W3 implementation, locked W4 validation summaries, W5 timing,
tests, integrity evidence and final decision. It intentionally excludes the
large residue-level profile and per-repeat timing table, which remain at their
repository-relative paths.

Reproduce from the repository root:

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:.
/home/jjxu/miniconda3/bin/python \
  scripts/window_motion_domains/run_w0_scale_provenance.py
/home/jjxu/miniconda3/bin/python \
  scripts/window_motion_domains/run_window_motion_domain_pilot.py
/home/jjxu/miniconda3/bin/python -m pytest -q tests
```

Do not interpret transition-aligned stitching controls as biologically
continuous MD. Do not include historical accidental scale-x10 results in
absolute-Angstrom biological summaries.
"""
    (HANDOFF_ROOT / "README.md").write_text(readme)
    manifest_rows = [
        {
            "path": str(path.relative_to(HANDOFF_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(HANDOFF_ROOT.rglob("*"))
        if path.is_file()
    ]
    write_tsv(HANDOFF_ROOT / "sha256_manifest.tsv", manifest_rows)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(HANDOFF_ROOT, arcname=HANDOFF_ROOT.name)
    (ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")).write_text(
        f"{sha256(ARCHIVE)}  {ARCHIVE.name}\n"
    )


def main() -> int:
    """Run final checks and build the compact handoff."""

    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    tests = run_tests()
    integrity = verify_hashes()
    (FINAL_ROOT / "frozen_hash_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )
    write_final_documents(tests, integrity)
    if tests["status"] != "pass" or integrity["status"] != "pass":
        raise RuntimeError("tests or frozen-hash integrity failed")
    build_handoff()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
