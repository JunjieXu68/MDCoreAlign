"""Create the compact Phase 5 handoff without raw trajectories."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "results/scaling_optimization_v3"
STAGING = V3_ROOT / "handoff_phase5"
ARCHIVE = V3_ROOT / "handoff_phase5_20260728.tar.gz"


INCLUDE = (
    "docs/scaling_optimization_v3/00_environment_reconciliation.md",
    "docs/scaling_optimization_v3/01_warmstart_semantic_adjudication.md",
    "docs/scaling_optimization_v3/02_sequential_lean_report.md",
    "docs/scaling_optimization_v3/cases/mdcath_3pio100_case_report.md",
    "results/scaling_optimization_v3/environment/interpreter_comparison.tsv",
    "results/scaling_optimization_v3/environment/canonical_environment.json",
    "results/scaling_optimization_v3/environment/canonical_pip_freeze.txt",
    "results/scaling_optimization_v3/environment/numpy_show_config.txt",
    "results/scaling_optimization_v3/environment/threadpool_info.json",
    "results/scaling_optimization_v3/environment/thread_calibration.tsv",
    "results/scaling_optimization_v3/environment/cpu_topology.txt",
    "results/scaling_optimization_v3/environment/raw_interpreters/base.json",
    "results/scaling_optimization_v3/environment/raw_interpreters/base_pip_freeze.txt",
    "results/scaling_optimization_v3/environment/raw_interpreters/base_numpy_show_config.txt",
    "results/scaling_optimization_v3/environment/raw_interpreters/base_conda_explicit.txt",
    "results/scaling_optimization_v3/environment/raw_interpreters/mdlovofit_env.json",
    "results/scaling_optimization_v3/environment/raw_interpreters/mdlovofit_env_pip_freeze.txt",
    "results/scaling_optimization_v3/environment/raw_interpreters/mdlovofit_env_numpy_show_config.txt",
    "results/scaling_optimization_v3/environment/raw_interpreters/mdlovofit_env_conda_explicit.txt",
    "results/scaling_optimization_v3/baseline/minimal_rebaseline.tsv",
    "results/scaling_optimization_v3/baseline/test_status.json",
    "results/scaling_optimization_v3/baseline/full_test_results.txt",
    "results/scaling_optimization_v3/baseline/golden_regression_status.json",
    "results/scaling_optimization_v3/baseline/integrity_after_phase4a.json",
    "results/scaling_optimization_v3/semantic/order_sensitivity.tsv",
    "results/scaling_optimization_v3/semantic/holdout_metrics.tsv",
    "results/scaling_optimization_v3/semantic/cross_replicate_metrics.tsv",
    "results/scaling_optimization_v3/semantic/subset_sensitivity.tsv",
    "results/scaling_optimization_v3/semantic/semantic_summary.json",
    "results/scaling_optimization_v3/semantic/mdcath_3pio100_iteration_traces.tsv",
    "results/scaling_optimization_v3/semantic/per_residue/mdcath_3pio100_per_residue.tsv",
    "results/scaling_optimization_v3/sequential_lean/parity.tsv",
    "results/scaling_optimization_v3/sequential_lean/performance.tsv",
    "results/scaling_optimization_v3/sequential_lean/memory.tsv",
    "results/scaling_optimization_v3/sequential_lean/call_counts.tsv",
    "results/scaling_optimization_v3/sequential_lean/test_results.txt",
    "results/scaling_optimization_v3/sequential_lean/test_status.json",
    "results/scaling_optimization_v3/sequential_lean/integrity_after_phase5.json",
    "results/scaling_optimization_v3/sequential_lean/phase5_summary.json",
    "results/scaling_optimization_v2/regression/golden_manifest.tsv",
    "results/scaling_optimization_v1/baseline/key_output_hashes.tsv",
    "src/mdsuperpose/sequential_weights.py",
    "src/mdsuperpose/long_mode.py",
    "scripts/scaling_optimization_v3/run_phase4a.py",
    "scripts/scaling_optimization_v3/run_phase4b_semantics.py",
    "scripts/scaling_optimization_v3/run_phase5.py",
    "scripts/scaling_optimization_v3/finalize_phase5_handoff.py",
    "tests/scaling_optimization/test_sequential_lean_equivalence.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    missing: list[str] = []
    for relative in INCLUDE:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            missing.append(relative)
            continue
        target = STAGING / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if missing:
        raise FileNotFoundError("Missing handoff inputs: " + ", ".join(missing))

    readme = """# MDCoreAlign Phase 5 compact handoff

This bundle contains the canonical environment record, minimal re-baseline,
warm-start semantic adjudication, the focused `mdcath_3pio100` analysis,
`sequential_lean` parity/performance/memory/call-count tables, source extracts,
tests and integrity manifests.

It intentionally excludes raw trajectories and fresh-process raw-run JSON.
The 10k input is a transition-aligned synthetic scaling trajectory, not real
continuous MD. The public default was not changed. No guarded hybrid, GPU,
FastScan, Faiss or Annoy implementation is included.
"""
    (STAGING / "README.md").write_text(readme)

    rows: list[dict[str, object]] = []
    for path in sorted(STAGING.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.tsv":
            rows.append(
                {
                    "path": path.relative_to(STAGING).as_posix(),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = STAGING / "sha256_manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "sha256", "size_bytes"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(STAGING, arcname="handoff_phase5")
    checksum = sha256(ARCHIVE)
    (V3_ROOT / "handoff_phase5_20260728.sha256").write_text(
        f"{checksum}  {ARCHIVE.name}\n"
    )
    print(f"created {ARCHIVE.relative_to(PROJECT_ROOT)}")
    print(f"sha256 {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
