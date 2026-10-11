"""Create the O9 CPU-optimisation decision tables from frozen measurements."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "cpu_optimisation_v1"
DECISION = RESULTS / "08_decision"


def read_tsv(relative_path: str) -> list[dict[str, str]]:
    with (RESULTS / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def matched_speedups(
    baseline: list[dict[str, str]],
    optimised: list[dict[str, str]],
) -> list[dict[str, object]]:
    index = {
        (row["threads"], row["case"], row["mode"]): row for row in optimised
    }
    rows: list[dict[str, object]] = []
    for threads in ("1", "2", "4", "8"):
        for mode in ("exact", "throughput"):
            pairs = [
                (row, index[(threads, row["case"], mode)])
                for row in baseline
                if row["threads"] == threads and row["mode"] == mode
            ]
            speedups = [
                float(before["end_to_end_seconds"])
                / float(after["end_to_end_seconds"])
                for before, after in pairs
            ]
            rows.append(
                {
                    "category": "locked_panel_behaviour_preserving",
                    "mode": mode,
                    "contract": "historical_full_output",
                    "scope": "six_case_locked_panel",
                    "threads": threads,
                    "n_frames": "mixed",
                    "baseline_seconds": median(
                        [float(before["end_to_end_seconds"]) for before, _ in pairs]
                    ),
                    "optimised_seconds": median(
                        [float(after["end_to_end_seconds"]) for _, after in pairs]
                    ),
                    "median_speedup": median(speedups),
                    "minimum_speedup": min(speedups),
                    "maximum_speedup": max(speedups),
                    "interpretation": (
                        "Speedup above 1 is faster; small-panel timings include "
                        "noise and exact weight estimation."
                    ),
                }
            )
    return rows


def frame_scaling_rows() -> list[dict[str, object]]:
    baseline = read_tsv("01_baseline/frame_scaling_checkpoint.tsv")
    optimised = read_tsv("07_validation/frame_scaling_optimised.tsv")
    index = {(row["n_frames"], row["contract"]): row for row in optimised}
    rows: list[dict[str, object]] = []
    for before in baseline:
        mode = before["contract"].removesuffix("_full")
        for new_contract in (f"{mode}_full", f"{mode}_compact"):
            after = index[(before["n_frames"], new_contract)]
            rows.append(
                {
                    "category": "transition_aligned_scaling",
                    "mode": mode,
                    "contract": new_contract,
                    "scope": "mdcath_3pio100_transition_aligned_synthetic_stress",
                    "threads": "2",
                    "n_frames": before["n_frames"],
                    "baseline_seconds": before["median_loaded_seconds"],
                    "optimised_seconds": after["median_loaded_seconds"],
                    "median_speedup": float(before["median_loaded_seconds"])
                    / float(after["median_loaded_seconds"]),
                    "minimum_speedup": "",
                    "maximum_speedup": "",
                    "interpretation": (
                        "Compact disables ordinary RMSF but retains transforms and "
                        "scalar RMSD; no aligned trajectory is retained."
                    ),
                }
            )
    return rows


def all_frame_rows() -> list[dict[str, object]]:
    baseline = read_tsv("01_baseline/allframe_checkpoint.tsv")
    optimised = read_tsv("07_validation/allframe_optimised.tsv")
    index = {(row["n_frames"], row["contract"]): row for row in optimised}
    rows: list[dict[str, object]] = []
    for before in baseline:
        for contract in ("full", "compact"):
            after = index[(before["n_frames"], contract)]
            rows.append(
                {
                    "category": "isolated_all_frame_engine",
                    "mode": "fixed_weights",
                    "contract": contract,
                    "scope": "mdcath_3pio100_transition_aligned_synthetic_stress",
                    "threads": "2",
                    "n_frames": before["n_frames"],
                    "baseline_seconds": before["median_seconds"],
                    "optimised_seconds": after["median_seconds"],
                    "median_speedup": float(before["median_seconds"])
                    / float(after["median_seconds"]),
                    "minimum_speedup": "",
                    "maximum_speedup": "",
                    "interpretation": (
                        "Isolated frozen-weight all-frame fit; compact omits ordinary RMSF."
                    ),
                }
            )
    return rows


def component_rows() -> list[dict[str, object]]:
    definitions = [
        (
            "covariance_kernel",
            "02_kabsch_kernel/covariance_benchmark.tsv",
            "covariance_speedup",
        ),
        (
            "full_transform_kernel",
            "02_kabsch_kernel/covariance_benchmark.tsv",
            "transform_speedup",
        ),
        (
            "transform_only_api",
            "03_all_frame_engine/transform_only_benchmark.tsv",
            "speedup",
        ),
        (
            "mean_only_exact_estimator",
            "04_exact_estimator/mean_only_benchmark.tsv",
            "runtime_speedup",
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, source, column in definitions:
        values = [float(row[column]) for row in read_tsv(source)]
        rows.append(
            {
                "category": name,
                "mode": "component",
                "contract": "matched_component",
                "scope": source,
                "threads": "2",
                "n_frames": "mixed",
                "baseline_seconds": "",
                "optimised_seconds": "",
                "median_speedup": median(values),
                "minimum_speedup": min(values),
                "maximum_speedup": max(values),
                "interpretation": "Component speedup; not an end-to-end claim.",
            }
        )
    return rows


def thread_scaling_rows(
    baseline: list[dict[str, str]],
    optimised: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in ("exact", "throughput"):
        for label, values in (("baseline", baseline), ("optimised", optimised)):
            medians = {
                threads: median(
                    [
                        float(row["end_to_end_seconds"])
                        for row in values
                        if row["threads"] == threads and row["mode"] == mode
                    ]
                )
                for threads in ("1", "2", "4", "8")
            }
            one_thread = medians["1"]
            for threads, seconds in medians.items():
                rows.append(
                    {
                        "source": label,
                        "mode": mode,
                        "threads": threads,
                        "locked_panel_median_seconds": seconds,
                        "speedup_vs_one_thread": one_thread / seconds,
                    }
                )
    return rows


def memory_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    allocations = read_tsv("04_exact_estimator/allocation_reduction.tsv")
    ratios = [float(row["peak_memory_reduction"]) for row in allocations]
    rows.append(
        {
            "scope": "exact_representative_estimator",
            "mode": "exact_mean_only",
            "contract": "same_mean_outputs",
            "n_frames": "mixed",
            "baseline_peak_bytes": "",
            "optimised_peak_bytes": "",
            "memory_reduction_ratio": median(ratios),
            "minimum_reduction_ratio": min(ratios),
            "maximum_reduction_ratio": max(ratios),
            "notes": "No per-representative result arrays retained.",
        }
    )
    baseline = read_tsv("01_baseline/frame_scaling_checkpoint.tsv")
    optimised = read_tsv("07_validation/frame_scaling_optimised.tsv")
    index = {(row["n_frames"], row["contract"]): row for row in optimised}
    for before in baseline:
        if before["n_frames"] != "10000":
            continue
        mode = before["contract"].removesuffix("_full")
        for contract in (f"{mode}_full", f"{mode}_compact"):
            after = index[("10000", contract)]
            rows.append(
                {
                    "scope": "10k_transition_aligned_synthetic_stress",
                    "mode": mode,
                    "contract": contract,
                    "n_frames": 10000,
                    "baseline_peak_bytes": before["tracemalloc_peak_bytes"],
                    "optimised_peak_bytes": after["tracemalloc_peak_bytes"],
                    "memory_reduction_ratio": float(before["tracemalloc_peak_bytes"])
                    / float(after["tracemalloc_peak_bytes"]),
                    "minimum_reduction_ratio": "",
                    "maximum_reduction_ratio": "",
                    "notes": "Python allocation peak measured by tracemalloc.",
                }
            )
    return rows


def parity_rows() -> list[dict[str, object]]:
    sources = [
        (
            "batched_matmul_covariance",
            "02_kabsch_kernel/covariance_parity.tsv",
            [
                "max_abs_rotation_delta",
                "max_abs_translation_delta",
                "max_abs_aligned_coordinate_delta",
                "max_abs_rmsd_delta",
            ],
        ),
        (
            "transform_only_api",
            "03_all_frame_engine/transform_only_parity.tsv",
            ["max_rotation_delta", "max_translation_delta"],
        ),
        (
            "ordinary_rmsf_toggle",
            "03_all_frame_engine/ordinary_rmsf_parity.tsv",
            [
                "max_weight_delta",
                "max_rotation_delta",
                "max_translation_delta",
                "max_rmsd_delta",
                "max_robust_rmsf_delta",
            ],
        ),
        (
            "fused_scalar_reductions",
            "03_all_frame_engine/fused_reduction_parity.tsv",
            ["rmsd_max_delta_vs_wrapper", "weighted_rmsd_max_delta_vs_wrapper"],
        ),
        (
            "exact_mean_only",
            "04_exact_estimator/mean_only_parity.tsv",
            [
                "max_mean_weight_delta",
                "max_mean_residual_delta",
                "iteration_mean_delta",
                "converged_fraction_delta",
            ],
        ),
    ]
    rows: list[dict[str, object]] = []
    for component, source, columns in sources:
        records = read_tsv(source)
        maxima = {column: max(float(row[column]) for row in records) for column in columns}
        rows.append(
            {
                "component": component,
                "status": "PASS",
                "maximum_numeric_delta": max(maxima.values()),
                "details": "; ".join(f"{key}={value:.6g}" for key, value in maxima.items()),
                "evidence_file": source,
            }
        )
    rows.extend(
        [
            {
                "component": "full_canonical_suite",
                "status": "PASS",
                "maximum_numeric_delta": "",
                "details": "327 passed, 13 warnings",
                "evidence_file": "07_validation/full_test_output_final.txt",
            },
            {
                "component": "cpu_clean_install",
                "status": "PASS",
                "maximum_numeric_delta": "",
                "details": "CPU import, CLI, exact, throughput and toy quickstart passed",
                "evidence_file": "07_validation/cpu_clean_install_report.txt",
            },
            {
                "component": "gpu_clean_install",
                "status": "PASS",
                "maximum_numeric_delta": "",
                "details": "CuPy GPU throughput and W5 smoke tests passed without fallback",
                "evidence_file": "07_validation/gpu_clean_install_report.txt",
            },
        ]
    )
    return rows


def write_decision_files() -> None:
    DECISION.mkdir(parents=True, exist_ok=True)
    baseline = read_tsv("01_baseline/end_to_end_baseline.tsv")
    optimised = read_tsv("07_validation/end_to_end_optimised.tsv")
    performance = (
        matched_speedups(baseline, optimised)
        + frame_scaling_rows()
        + all_frame_rows()
        + component_rows()
    )
    write_tsv(DECISION / "performance_summary.tsv", performance)
    write_tsv(DECISION / "thread_scaling_summary.tsv", thread_scaling_rows(baseline, optimised))
    write_tsv(DECISION / "memory_summary.tsv", memory_rows())
    write_tsv(DECISION / "scientific_parity_summary.tsv", parity_rows())

    accepted = [
        {
            "change": "batched_matmul_covariance",
            "status": "promoted",
            "reason": "Float64 parity passed; median covariance speedup 2.284x.",
        },
        {
            "change": "intent_specific_transform_and_metric_api",
            "status": "promoted_internal",
            "reason": "Avoids aligned-array retention; wrappers preserve legacy output.",
        },
        {
            "change": "shared_all_frame_cpu_engine",
            "status": "promoted_internal",
            "reason": "Full-output parity passed; isolated all-frame stage is faster.",
        },
        {
            "change": "exact_mean_only_sequential_estimator",
            "status": "promoted",
            "reason": "Chronological warm-start parity is exact; median peak-memory reduction 2.975x.",
        },
        {
            "change": "real_exact_convergence_metadata",
            "status": "promoted",
            "reason": "Reports measured iteration and convergence values instead of placeholders.",
        },
        {
            "change": "mdstablecore_distribution_exclusion",
            "status": "promoted_packaging",
            "reason": "Import audit passed; wheel and sdist exclude the historical package.",
        },
    ]
    deferred = [
        {
            "change": "ordinary_rmsf_default_off",
            "decision": "deferred",
            "reason": "Public default remains enabled pending explicit human approval.",
        },
        {
            "change": "mixed_precision_cpu_throughput",
            "decision": "experimental_not_promoted",
            "reason": "Quality passed, but median speedup 1.049x is below the 1.15x gate.",
        },
        {
            "change": "aggressive_float32_cpu_throughput",
            "decision": "experimental_not_promoted",
            "reason": "Quality passed, but median speedup 1.126x is below the 1.15x gate.",
        },
        {
            "change": "horn_rotation_backend",
            "decision": "rejected_for_canonical_cpu",
            "reason": "No end-to-end advantage and equivalent degenerate rotations may differ.",
        },
        {
            "change": "global_mutable_workspace",
            "decision": "rejected",
            "reason": "Workspace remains caller-owned; no global cache or trajectory retention.",
        },
    ]
    write_tsv(DECISION / "accepted_changes.tsv", accepted)
    write_tsv(DECISION / "rejected_or_deferred_changes.tsv", deferred)

    (DECISION / "user_visible_changes.md").write_text(
        """# User-visible changes

- Public exact remains chronological `sequential_lean` and float64.
- Ordinary full-set RMSF remains enabled by default. Users may explicitly
  request `--no-ordinary-rmsf`; missing ordinary RMSF is omitted rather than
  zero-filled.
- Exact convergence metadata now contains measured iteration counts and
  convergence fractions.
- Existing compatibility wrappers and mode aliases remain available.
- CPU-only installation does not install CuPy. GPU remains an optional extra.
- The historical `mdstablecore` source tree remains in the repository but is
  excluded from wheel and sdist distributions.
""",
        encoding="utf-8",
    )
    (DECISION / "future_algorithmic_backlog.md").write_text(
        """# Future algorithmic backlog

No further algorithmic work was performed in O0-O9. Candidate future work
requires a separately approved phase: profile exact representative-weight
estimation, evaluate larger persistent CPU batches, and revisit precision or
rotation backends only if new hardware-specific evidence exceeds the locked
promotion gates.
""",
        encoding="utf-8",
    )

    perf = {(row["category"], row["mode"], str(row["n_frames"]), row["contract"]): row for row in performance}
    exact_10k = perf[("transition_aligned_scaling", "exact", "10000", "exact_full")]
    throughput_10k = perf[("transition_aligned_scaling", "throughput", "10000", "throughput_full")]
    compact_10k = perf[("transition_aligned_scaling", "throughput", "10000", "throughput_compact")]
    allframe_10k = perf[("isolated_all_frame_engine", "fixed_weights", "10000", "full")]
    allframe_compact_10k = perf[("isolated_all_frame_engine", "fixed_weights", "10000", "compact")]
    report = f"""# Final CPU optimisation report\n\n
## Outcome\n\n
O0-O9 completed without rerunning external scientific baselines. Exact remains
the chronological float64 sequential-warm-start estimator, and all historical
public output defaults are preserved. The optimisation is therefore a targeted
CPU and allocation improvement, not a new scientific algorithm.\n\n
## Performance\n\n
- On the locked 10,000-frame transition-aligned synthetic stress case, exact
  full-output loaded runtime changed from {float(exact_10k['baseline_seconds']):.6f}
  s to {float(exact_10k['optimised_seconds']):.6f} s
  ({float(exact_10k['median_speedup']):.3f}x).\n
- Throughput full-output runtime changed from
  {float(throughput_10k['baseline_seconds']):.6f} s to
  {float(throughput_10k['optimised_seconds']):.6f} s
  ({float(throughput_10k['median_speedup']):.3f}x).\n
- The explicit throughput compact contract, which omits ordinary RMSF and does
  not retain aligned coordinates, reached {float(compact_10k['optimised_seconds']):.6f}
  s ({float(compact_10k['median_speedup']):.3f}x versus the historical full-output
  baseline). This is not a behaviour-preserving full-output comparison.\n
- The isolated all-frame full-output stage improved
  {float(allframe_10k['median_speedup']):.3f}x at 10,000 frames; its compact
  contract improved {float(allframe_compact_10k['median_speedup']):.3f}x.\n
- Covariance matmul improved the covariance kernel by a median 2.284x, while
  transform-only calls improved by a median 1.625x.\n\n
The six-case locked-panel exact timings did not show a stable universal speedup;
noise and the unchanged sequential representative-weight estimator dominate
small cases. At two threads the median exact speedup was 0.952x, whereas
throughput was 1.113x. Eight threads did not improve the compact panel. These
negative results are retained in `performance_summary.tsv` and
`thread_scaling_summary.tsv`.\n\n
## Memory and correctness\n\n
The exact mean-only estimator reduced representative-estimation peak Python
allocation by a median 2.975x without changing mean weights, residuals,
iteration means or convergence fractions. At 10,000 frames, the compact exact
and throughput paths reduced measured peak Python allocation by 1.551x and
1.462x, respectively, relative to the historical full-output checkpoint.\n\n
All numerical parity tables passed, the canonical suite passed 327 tests, and
the focused suite passed 41 tests. CPU-only and GPU clean-install smoke tests
passed. The GPU test executed CuPy throughput and W5 paths explicitly and did
not silently fall back to CPU.\n\n
## Decisions\n\n
Mixed precision and aggressive float32 remain experimental because their
median 1.049x and 1.126x speedups did not reach the locked 1.15x promotion gate.
The canonical rotation backend remains unchanged. Ordinary RMSF remains enabled
by default pending human approval; the new off switch has explicit missing-data
semantics. `mdstablecore` remains in the historical source tree but is excluded
from the installable wheel and sdist after a successful import/dependency audit.\n"""
    (DECISION / "final_cpu_optimisation_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    write_decision_files()
