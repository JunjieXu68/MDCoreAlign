#!/usr/bin/env python
"""Finalize the gated Phase 6 decision, integrity record, and compact handoff."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import pstats
import shutil
import statistics
import subprocess
import sys
import tarfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/jjxu/miniconda3/bin/python")
RESULT_ROOT = PROJECT_ROOT / "results/scaling_optimization_v4"
QUALITY_ROOT = RESULT_ROOT / "window_quality"
DOC_ROOT = PROJECT_ROOT / "docs/scaling_optimization_v4"
HANDOFF_ROOT = RESULT_ROOT / "handoff_phase6"
ARCHIVE = RESULT_ROOT / "handoff_phase6_20260728.tar.gz"
SOURCE_FILES = (
    "src/mdsuperpose/window_descriptors.py",
    "src/mdsuperpose/window_scan.py",
    "src/mdsuperpose/stability_graph.py",
    "scripts/scaling_optimization_v4/run_phase6a.py",
    "scripts/scaling_optimization_v4/run_phase6b.py",
    "scripts/scaling_optimization_v4/finalize_phase6.py",
    "tests/scaling_optimization_v4/test_window_scan_invariance.py",
    "tests/scaling_optimization_v4/test_window_scan_boundaries.py",
    "tests/scaling_optimization_v4/test_stability_graph.py",
)


def environment() -> dict[str, str]:
    """Return the fixed two-thread Phase 6 environment."""

    values = os.environ.copy()
    values.update(
        {
            "OPENBLAS_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "PYTHONPATH": "src:.",
        }
    )
    return values


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write union-schema rows as TSV."""

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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def verify_hashes() -> dict[str, object]:
    """Verify all frozen project and golden references."""

    failures: list[dict[str, str]] = []
    project_rows = read_tsv(
        PROJECT_ROOT / "results/scaling_optimization_v1/baseline/key_output_hashes.tsv"
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
            / "results/scaling_optimization_v2/regression/golden_manifest.tsv"
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
    """Run the complete test suite with the canonical interpreter."""

    command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    output = completed.stdout + completed.stderr
    (RESULT_ROOT / "full_test_results.txt").write_text(output)
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "failed",
        "summary": (completed.stdout.strip().splitlines() or [""])[-1],
    }


def ten_k_quality_rows() -> dict[str, dict[str, str]]:
    """Index successful 10k Phase 6B rows by method."""

    rows = read_tsv(QUALITY_ROOT / "case_metrics.tsv")
    return {
        row["method"]: row
        for row in rows
        if row["case_id"] == "synthetic_scaling_4oelA02_10000"
        and row["status"] == "success"
    }


def build_ablation_table() -> list[dict[str, object]]:
    """Build the required table, preserving not-run refinement rows."""

    quality = ten_k_quality_rows()
    phase5 = read_tsv(
        PROJECT_ROOT
        / "results/scaling_optimization_v3/sequential_lean/performance.tsv"
    )
    sequential_times = [
        float(row["in_child_end_to_end_seconds"])
        for row in phase5
        if row["method"] == "sequential_lean"
        and row["n_frames"] == "10000"
        and row["profile_mode"] == "clean"
        and row["warmup"] in {"0", "False"}
        and row["status"] == "success"
    ]
    batched_summary = json.loads(
        (
            PROJECT_ROOT
            / "results/scaling_optimization_v2/cpu_batch/phase3_summary.json"
        ).read_text()
    )
    rows: list[dict[str, object]] = [
        {
            "method": "sequential_lean",
            "status": "completed_exact_reference",
            "output_contract": "metrics/transforms_only_exact",
            "frames": 10000,
            "residues": 153,
            "representative_refinement_frames": 2000,
            "total_runtime_seconds": statistics.median(sequential_times),
            "scan_seconds": "NA",
            "refinement_seconds": "NA",
            "all_frame_fit_seconds": "included",
            "peak_rss_mib": 160.703,
            "speedup_vs_same_environment_mdlovofit": "not_rerun_in_phase6",
            "speedup_vs_sequential_lean": 1.0,
            "core_jaccard": 1.0,
            "weight_spearman": 1.0,
            "rmsf_spearman": 1.0,
            "rmsd_median_absolute_difference": 0.0,
            "rmsd_95th_percentile_absolute_difference": 0.0,
            "flexible_label_jaccard": 1.0,
            "flexible_label_f1": 1.0,
            "heldout_pair_distance_stability": "reference",
            "ambiguity_risk_flag": False,
            "fallback_simulated": False,
        },
        {
            "method": "batched_independent",
            "status": "completed_independent_reference",
            "output_contract": "metrics/transforms_only_exact",
            "frames": 10000,
            "residues": 153,
            "representative_refinement_frames": 2000,
            "total_runtime_seconds": batched_summary[
                "batched_clean_median_seconds"
            ],
            "scan_seconds": "NA",
            "refinement_seconds": "NA",
            "all_frame_fit_seconds": "included",
            "peak_rss_mib": "see_phase3",
            "speedup_vs_same_environment_mdlovofit": "not_rerun_in_phase6",
            "speedup_vs_sequential_lean": (
                statistics.median(sequential_times)
                / batched_summary["batched_clean_median_seconds"]
            ),
            "core_jaccard": quality["batched_independent"][
                "core_jaccard_vs_sequential"
            ],
            "weight_spearman": quality["batched_independent"][
                "weight_spearman_vs_sequential"
            ],
            "rmsf_spearman": quality["batched_independent"][
                "rmsf_spearman_vs_sequential"
            ],
            "rmsd_median_absolute_difference": "not_recomputed_in_phase6b",
            "rmsd_95th_percentile_absolute_difference": (
                "not_recomputed_in_phase6b"
            ),
            "flexible_label_jaccard": quality["batched_independent"][
                "flexible_jaccard_vs_sequential"
            ],
            "flexible_label_f1": quality["batched_independent"][
                "flexible_f1_vs_sequential"
            ],
            "heldout_pair_distance_stability": "see_case_metrics",
            "ambiguity_risk_flag": False,
            "fallback_simulated": False,
        },
    ]
    for method in (
        "window_local_w5_minimal",
        "window_local_w5_full",
        "window_local_multiscale",
        "window_graph_local_contact",
        "window_graph_local_anchor",
        "window_graph_full",
    ):
        source = quality[method]
        total = float(source["scan_seconds"]) + float(
            source["validation_fit_seconds"]
        )
        rows.append(
            {
                "method": method,
                "status": "completed_phase6b_quality_failed",
                "output_contract": (
                    "scan_only_plus_explicit_exact_validation_fit"
                ),
                "frames": source["n_frames"],
                "residues": source["n_residues"],
                "representative_refinement_frames": 0,
                "total_runtime_seconds": total,
                "scan_seconds": source["scan_seconds"],
                "refinement_seconds": 0.0,
                "all_frame_fit_seconds": source["validation_fit_seconds"],
                "peak_rss_mib": "see_scan_timing_or_not_separately_measured",
                "speedup_vs_same_environment_mdlovofit": (
                    "not_rerun_in_phase6"
                ),
                "speedup_vs_sequential_lean": (
                    statistics.median(sequential_times) / total
                ),
                "core_jaccard": source["core_jaccard_vs_sequential"],
                "weight_spearman": source["weight_spearman_vs_sequential"],
                "rmsf_spearman": source["rmsf_spearman_vs_sequential"],
                "rmsd_median_absolute_difference": source[
                    "median_rmsd_abs_difference_vs_sequential"
                ],
                "rmsd_95th_percentile_absolute_difference": (
                    "not_computed_phase6b"
                ),
                "flexible_label_jaccard": source[
                    "flexible_jaccard_vs_sequential"
                ],
                "flexible_label_f1": source[
                    "flexible_f1_vs_sequential"
                ],
                "heldout_pair_distance_stability": source[
                    "heldout_pair_distance_variance"
                ],
                "ambiguity_risk_flag": source["risk_flag"],
                "fallback_simulated": False,
            }
        )
    for method in (
        "window_refine_0",
        "window_refine_1_32",
        "window_refine_1_64",
        "window_refine_1_128",
        "window_refine_2_64",
        "window_refine_2_128",
    ):
        rows.append(
            {
                "method": method,
                "status": "not_run_phase6b_gate_failed",
                "output_contract": "not_generated",
                "frames": 10000,
                "residues": 153,
                "representative_refinement_frames": (
                    method.rsplit("_", 1)[-1] if method != "window_refine_0" else 0
                ),
                "total_runtime_seconds": "NA",
                "scan_seconds": "NA",
                "refinement_seconds": "NA",
                "all_frame_fit_seconds": "NA",
                "peak_rss_mib": "NA",
                "speedup_vs_same_environment_mdlovofit": "NA",
                "speedup_vs_sequential_lean": "NA",
                "core_jaccard": "NA",
                "weight_spearman": "NA",
                "rmsf_spearman": "NA",
                "rmsd_median_absolute_difference": "NA",
                "rmsd_95th_percentile_absolute_difference": "NA",
                "flexible_label_jaccard": "NA",
                "flexible_label_f1": "NA",
                "heldout_pair_distance_stability": "NA",
                "ambiguity_risk_flag": "not_evaluated",
                "fallback_simulated": False,
            }
        )
    return rows


def fallback_requirement_rows() -> list[dict[str, object]]:
    """Quantify oracle fallback need without implementing automatic fallback."""

    rows = [
        row
        for row in read_tsv(QUALITY_ROOT / "case_metrics.tsv")
        if row["method"] == "window_local_multiscale" and row["status"] == "success"
    ]
    output: list[dict[str, object]] = []
    for row in rows:
        quality_pass = (
            float(row["core_jaccard_vs_sequential"]) >= 0.85
            and float(row["weight_spearman_vs_sequential"]) >= 0.90
            and float(row["rmsf_spearman_vs_sequential"]) >= 0.90
            and float(row["flexible_jaccard_vs_sequential"]) >= 0.80
        )
        output.append(
            {
                "case_id": row["case_id"],
                "candidate_method": "window_local_multiscale",
                "quality_pass": quality_pass,
                "oracle_fallback_required": not quality_pass,
                "available_risk_flag": row["risk_flag"],
                "risk_signal_detected_required_fallback": (
                    row["risk_flag"] == "True" and not quality_pass
                ),
                "note": (
                    "diagnostic only; automatic window_guarded was not implemented"
                ),
            }
        )
    return output


def profile_summary() -> list[dict[str, object]]:
    """Extract the leading cumulative functions from saved cProfile files."""

    rows: list[dict[str, object]] = []
    for name in ("window_local_w5_10k", "window_graph_full_10k"):
        stats = pstats.Stats(str(RESULT_ROOT / f"profiling/{name}.prof"))
        ordered = sorted(
            stats.stats.items(),
            key=lambda item: item[1][3],
            reverse=True,
        )[:15]
        for rank, ((filename, line, function), values) in enumerate(
            ordered,
            start=1,
        ):
            primitive_calls, total_calls, total_time, cumulative, _ = values
            rows.append(
                {
                    "profile": name,
                    "rank": rank,
                    "function": f"{Path(filename).name}:{line}({function})",
                    "primitive_calls": primitive_calls,
                    "total_calls": total_calls,
                    "self_seconds": total_time,
                    "cumulative_seconds": cumulative,
                }
            )
    return rows


def make_plots() -> None:
    """Create compact timing and quality plots for the handoff."""

    figure_root = RESULT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    timings = [
        row
        for row in read_tsv(RESULT_ROOT / "window_local/scan_timing.tsv")
        if row["warmup"] == "0" and row["status"] == "success"
    ]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for statistic, label in (
        ("relative_mad", "Relative MAD"),
        ("winsorised_variance", "Winsorised variance"),
    ):
        x_values = sorted({int(row["n_frames"]) for row in timings})
        y_values = [
            statistics.median(
                float(row["total_scan_only"])
                for row in timings
                if row["statistic"] == statistic
                and int(row["n_frames"]) == frames
            )
            for frames in x_values
        ]
        axis.plot(x_values, y_values, marker="o", label=label)
    axis.axhline(1.0, color="0.4", linestyle="--", linewidth=1, label="6A gate")
    axis.set(
        xlabel="Frames",
        ylabel="Scan-only runtime (s)",
        title="Phase 6A alignment-free W5 scaling",
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_root / "phase6a_scan_timing.png", dpi=220)
    fig.savefig(figure_root / "phase6a_scan_timing.pdf")
    plt.close(fig)

    quality = [
        row
        for row in read_tsv(QUALITY_ROOT / "case_metrics.tsv")
        if row["method"].startswith("window_") and row["status"] == "success"
    ]
    method_order = [
        "window_local_w5_minimal",
        "window_local_w5_full",
        "window_local_multiscale",
        "window_graph_local_contact",
        "window_graph_local_anchor",
        "window_graph_full",
    ]
    labels = ["W5 min", "W5 full", "W3/5/9", "Local+contact", "Local+anchor", "Full graph"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8), sharey=True)
    metrics = (
        ("core_jaccard_vs_sequential", "Core Jaccard", 0.85),
        ("weight_spearman_vs_sequential", "Weight Spearman", 0.90),
        ("rmsf_spearman_vs_sequential", "RMSF Spearman", 0.90),
        ("flexible_jaccard_vs_sequential", "Flexible Jaccard", 0.80),
    )
    for axis, (field, title, gate) in zip(axes, metrics, strict=True):
        data = [
            [
                float(row[field])
                for row in quality
                if row["method"] == method
            ]
            for method in method_order
        ]
        axis.boxplot(data, labels=labels, showfliers=True)
        axis.axhline(gate, color="#9b2c2c", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=60)
        axis.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Agreement with sequential_lean")
    fig.suptitle("Phase 6B: fast scans preserve fitted RMSF, not core interpretation")
    fig.tight_layout()
    fig.savefig(figure_root / "phase6b_quality_distributions.png", dpi=220)
    fig.savefig(figure_root / "phase6b_quality_distributions.pdf")
    plt.close(fig)


def write_decision_report(
    tests: dict[str, object],
    integrity: dict[str, object],
    fallback_rows: list[dict[str, object]],
) -> None:
    """Write the required gated Phase 6 stop report."""

    summary = read_tsv(QUALITY_ROOT / "summary_metrics.tsv")
    best = next(
        row for row in summary if row["method"] == "window_local_multiscale"
    )
    fallback_count = sum(bool(row["oracle_fallback_required"]) for row in fallback_rows)
    risk_caught = sum(
        bool(row["risk_signal_detected_required_fallback"]) for row in fallback_rows
    )
    report = f"""# Phase 6 sliding-window decision report

## Scope and stop decision

Phase 6A and Phase 6B were completed. Phase 6C was **not run** because no
alignment-free candidate passed the mandatory Phase 6B scientific continuation
gate. This is the prescribed stop condition in the v4 plan, not an execution
failure. Phase 6D was not executed. The public default and `sequential_lean`
were not changed.

## Direct answers

1. **Is W5 substantially faster?** Yes for `scan_only`: the 10k/153-residue
   medians were 0.733018 s for exact relative MAD and 0.304071 s for the
   pre-specified winsorised relative-variance path. These are movement/stability
   scans, not RMSD calculations.
2. **What can W5 reproduce?** It passed 10/10 rigid-invariance, localisation,
   boundary and repeatability controls. After an explicit global fit, the best
   candidate had median RMSF Spearman
   {float(best["median_rmsf_spearman_vs_sequential"]):.4f}. It did not reliably
   reproduce stable-core weights or flexible labels.
3. **Which failures are hinge/domain related?** Pure W5 is blind to internal
   rigidity-preserving domain motion away from the hinge. This expected
   limitation was reproduced.
4. **Do contacts/anchors fix it?** They detect relative-domain motion and
   equal-domain ambiguity, but did not recover exact core choice. Full-graph
   median core/weight agreement was below the required gate.
5. **How much exact refinement is required?** Unknown. Phase 6C refinement was
   not ethically or procedurally run after the Phase 6B gate failed.
6. **Fastest gate-passing variant?** None. The best alignment-free core result
   was `window_local_multiscale`: median core Jaccard
   {float(best["median_core_jaccard_vs_sequential"]):.4f}, weight Spearman
   {float(best["median_weight_spearman_vs_sequential"]):.4f}, and
   flexible-label Jaccard
   {float(best["median_flexible_jaccard_vs_sequential"]):.4f}.
7. **Fallback fraction?** An oracle applying all quality thresholds would
   require fallback for {fallback_count}/34 cases. The available risk signal
   identified only {risk_caught}/{fallback_count}.
8. **Faster after simulated fallback?** No defensible speedup can be claimed:
   34/34 oracle fallbacks would add scan cost before the exact path.
9. **Evidence for `window_guarded`?** No. Risk sensitivity is insufficient and
   Phase 6D remains unapproved.
10. **Standalone user-facing scan?** It may be useful later as an experimental
    local geometric-change diagnostic, but not as the default stable-core
    estimator and not as RMSD.
11. **GPU justified?** Not by this result. W5 distance calculation is already
    sub-second; the blocker is scientific identification, not compute.
12. **Real versus synthetic claims?** Rigid invariance and hinge limitations
    are controlled synthetic evidence. The 34-case panel includes four formal
    MD systems and five real mdCATH replicates, where core/flexible-label
    agreement remained insufficient. The 10k input is a transition-aligned
    synthetic scaling trajectory and carries no biological-continuity claim.

## Profiling

For the 10k case, cProfile attributed 0.188 s of the 0.238 s W5 profile to
sparse squared-distance evaluation. The full graph profile took 0.945 s, with
0.668 s in two descriptor evaluations. The second evaluation supports the
local-versus-graph risk comparison; removing it would improve runtime but not
the failed quality metrics.

## Verification

- Full tests: {tests["summary"]}.
- Frozen integrity: {integrity["status"]},
  {integrity["project_completion_checked"]}/9 project-completion and
  {integrity["golden_checked"]}/34 golden references checked.
- Canonical interpreter: `/home/jjxu/miniconda3/bin/python`.
- Threads: OpenBLAS/OMP/MKL/NumExpr = 2.

## Exact commands

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:.
/home/jjxu/miniconda3/bin/python scripts/scaling_optimization_v4/run_phase6a.py
/home/jjxu/miniconda3/bin/python scripts/scaling_optimization_v4/run_phase6b.py
/home/jjxu/miniconda3/bin/python scripts/scaling_optimization_v4/finalize_phase6.py
```

`run_phase6b.py` exits with code 2 when the scientific continuation gate fails;
the complete tables are still written and no cases are removed.
"""
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "05_phase6_decision_report.md").write_text(report)


def build_added_files_patch() -> None:
    """Record all Phase 6 source additions as a recoverable unified patch."""

    patch_lines: list[str] = []
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        lines = path.read_text().splitlines(keepends=True)
        patch_lines.extend(
            difflib.unified_diff(
                [],
                lines,
                fromfile="/dev/null",
                tofile=relative,
            )
        )
    (RESULT_ROOT / "phase6_added_files.patch").write_text("".join(patch_lines))


def build_handoff(
    tests: dict[str, object],
    integrity: dict[str, object],
) -> tuple[str, int]:
    """Create a compact, trajectory-free Phase 6 archive."""

    if HANDOFF_ROOT.exists():
        shutil.rmtree(HANDOFF_ROOT)
    HANDOFF_ROOT.mkdir(parents=True)
    include_roots = (
        PROJECT_ROOT / "docs/scaling_optimization_v4",
        PROJECT_ROOT / "scripts/scaling_optimization_v4",
        PROJECT_ROOT / "tests/scaling_optimization_v4",
    )
    for root in include_roots:
        destination = HANDOFF_ROOT / root.relative_to(PROJECT_ROOT)
        shutil.copytree(
            root,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for relative in (
        "src/mdsuperpose/window_descriptors.py",
        "src/mdsuperpose/window_scan.py",
        "src/mdsuperpose/stability_graph.py",
    ):
        destination = HANDOFF_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)
    result_destination = HANDOFF_ROOT / "results/scaling_optimization_v4"
    result_destination.mkdir(parents=True, exist_ok=True)
    for path in RESULT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if HANDOFF_ROOT in path.parents or path == ARCHIVE:
            continue
        if path.suffix == ".prof":
            continue
        destination = result_destination / path.relative_to(RESULT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    readme = f"""# MDCoreAlign Phase 6 compact handoff

Phase 6A and 6B completed. Phase 6C was not run because the mandatory 6B
scientific quality gate failed; Phase 6D was not executed. This archive
contains code, tests, reports, tables, compact plots, profiles summaries and
integrity records, but no trajectory inputs.

Tests: {tests["summary"]}
Integrity: {integrity["status"]}
"""
    (HANDOFF_ROOT / "README.md").write_text(readme)
    manifest_rows: list[dict[str, object]] = []
    for path in sorted(HANDOFF_ROOT.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.tsv":
            manifest_rows.append(
                {
                    "path": path.relative_to(HANDOFF_ROOT).as_posix(),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_tsv(HANDOFF_ROOT / "sha256_manifest.tsv", manifest_rows)
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(HANDOFF_ROOT, arcname=HANDOFF_ROOT.name)
    digest = sha256(ARCHIVE)
    (RESULT_ROOT / "handoff_phase6_20260728.sha256").write_text(
        f"{digest}  {ARCHIVE.name}\n"
    )
    return digest, len(manifest_rows) + 1


def main() -> int:
    """Finalize Phase 6 after the prescribed Phase 6B stop."""

    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use {PYTHON}; got {sys.executable}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ablation = build_ablation_table()
    write_tsv(RESULT_ROOT / "consolidated_ablation.tsv", ablation)
    fallback_rows = fallback_requirement_rows()
    write_tsv(QUALITY_ROOT / "fallback_requirement_diagnostic.tsv", fallback_rows)
    write_tsv(RESULT_ROOT / "profiling/profile_summary.tsv", profile_summary())
    make_plots()
    build_added_files_patch()
    tests = run_tests()
    (RESULT_ROOT / "test_status.json").write_text(
        json.dumps(tests, indent=2, sort_keys=True) + "\n"
    )
    integrity = verify_hashes()
    (RESULT_ROOT / "integrity_after_phase6.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    write_decision_report(tests, integrity, fallback_rows)
    summary = {
        "status": "stopped_after_phase6b_gate_failed",
        "phase6a_completed": True,
        "phase6b_completed": True,
        "phase6c_completed": False,
        "phase6c_not_run_reason": "mandatory_phase6b_scientific_gate_failed",
        "phase6d_executed": False,
        "public_default_changed": False,
        "test_status": tests,
        "integrity": integrity,
        "oracle_fallback_required_count": sum(
            bool(row["oracle_fallback_required"]) for row in fallback_rows
        ),
        "case_count": len(fallback_rows),
    }
    (RESULT_ROOT / "phase6_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    digest, file_count = build_handoff(tests, integrity)
    summary["handoff_sha256"] = digest
    summary["handoff_file_count"] = file_count
    (RESULT_ROOT / "phase6_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0 if tests["status"] == "pass" and integrity["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
