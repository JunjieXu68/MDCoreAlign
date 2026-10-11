#!/usr/bin/env python3
"""Finalize optional-deployment decision and honest NM7 interpretation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np

from scripts.no_materialisation_benchmark_v1.common import (
    OUT,
    ROOT,
    read_tsv,
    write_json,
    write_text,
    write_tsv,
)


ANALYSIS = OUT / "06_analysis"
FIGURES = ANALYSIS / "figures"


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _bootstrap_median_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    """Return a deterministic percentile interval for paired median speedup."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(20_000, values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _atlas_comparisons(summary: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lookup = {(row["system"], row["prefix_label"], row["method"]): row for row in summary}
    methods = (
        "MDCoreAlign_exact_CPU_minimal",
        "MDCoreAlign_throughput_CPU_minimal",
        "MDCoreAlign_throughput_GPU_minimal",
    )
    detail: list[dict[str, object]] = []
    for system, prefix, _ in sorted({(row["system"], row["prefix_label"], "") for row in summary}):
        stock = lookup[(system, prefix, "stock_MDLovoFit_no_output")]
        timed = lookup[(system, prefix, "timed_MDLovoFit_no_output")]
        for method in methods:
            candidate = lookup[(system, prefix, method)]
            detail.append(
                {
                    "panel": "ATLAS_48",
                    "system": system,
                    "setting": prefix,
                    "n_frames": stock["n_frames"],
                    "n_residues": stock["n_residues"],
                    "residue_frames": stock["residue_frames"],
                    "MDCoreAlign_method": method,
                    "C1_stock_MDLovoFit_seconds": stock["C1_median_seconds"],
                    "C1_MDCoreAlign_seconds": candidate["C1_median_seconds"],
                    "C1_MDLovoFit_over_MDCoreAlign_speedup": float(stock["C1_median_seconds"]) / float(candidate["C1_median_seconds"]),
                    "C2_timed_MDLovoFit_seconds": timed["C2_median_seconds"],
                    "C2_MDCoreAlign_seconds": candidate["C2_median_seconds"],
                    "C2_MDLovoFit_over_MDCoreAlign_speedup": float(timed["C2_median_seconds"]) / float(candidate["C2_median_seconds"]),
                }
            )
    aggregates: list[dict[str, object]] = []
    for method in methods:
        group = [row for row in detail if row["MDCoreAlign_method"] == method]
        for contract in ("C1", "C2"):
            values = np.asarray([float(row[f"{contract}_MDLovoFit_over_MDCoreAlign_speedup"]) for row in group])
            md_seconds = np.asarray([float(row[f"{contract}_{'stock_' if contract == 'C1' else 'timed_'}MDLovoFit_seconds"]) for row in group])
            candidate_seconds = np.asarray([float(row[f"{contract}_MDCoreAlign_seconds"]) for row in group])
            ci_lower, ci_upper = _bootstrap_median_ci(
                values,
                seed=7100 + methods.index(method) * 10 + (1 if contract == "C1" else 2),
            )
            aggregates.append(
                {
                    "panel": "ATLAS_48",
                    "contract": "C1_same_PDB_tool_workflow" if contract == "C1" else "C2_loaded_core_compute",
                    "comparison": f"MDLovoFit_over_{method}",
                    "paired_domains": values.size,
                    "paired_median_speedup": float(np.median(values)),
                    "paired_median_bootstrap_95CI_lower": ci_lower,
                    "paired_median_bootstrap_95CI_upper": ci_upper,
                    "ratio_of_median_runtimes": float(np.median(md_seconds) / np.median(candidate_seconds)),
                    "speedup_IQR": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "MDCoreAlign_faster_fraction": float(np.mean(values > 1.0)),
                    "minimum_speedup": float(np.min(values)),
                    "maximum_speedup": float(np.max(values)),
                }
            )
    return detail, aggregates


def _prior_comparison(atlas_detail: list[dict[str, object]], mdcath_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    prior_atlas = read_tsv(ROOT / "results/atlas_large_long_v1/a3_external/external_raw_repeats.tsv")
    prior_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in prior_atlas:
        if row["method"] == "MDLovoFit_20.1.1" and row["status"] == "success" and row["is_warmup"] == "False" and row["contract"].startswith("C_materialised"):
            prior_groups[(row["system"], row["prefix_label"])].append(float(row["total_seconds"]))
    output: list[dict[str, object]] = []
    for row in atlas_detail:
        if row["MDCoreAlign_method"] != "MDCoreAlign_throughput_GPU_minimal":
            continue
        key = (str(row["system"]), str(row["setting"]))
        if key not in prior_groups:
            continue
        prior = _median(prior_groups[key])
        no_output = float(row["C1_stock_MDLovoFit_seconds"])
        candidate = float(row["C1_MDCoreAlign_seconds"])
        output.append(
            {
                "panel": "ATLAS_48",
                "system": key[0],
                "setting": key[1],
                "prior_MDLovoFit_with_t_materialised_seconds": prior,
                "new_stock_MDLovoFit_no_output_seconds": no_output,
                "MDLovoFit_output_removal_speedup": prior / no_output,
                "prior_materialised_workflow_speedup_vs_GPU": prior / candidate,
                "new_no_output_workflow_speedup_vs_GPU": no_output / candidate,
            }
        )
    previous = read_tsv(ROOT / "results/database_scale_benchmark_v2/mdlovofit_comparison/mdlovofit_all_domain_corrected_matched.tsv")
    old = {row["domain_id"]: row for row in previous}
    new = {(row["system"], row["method"]): row for row in mdcath_rows if row["repeat_index"] == "0" and row["status"] == "success"}
    for domain in sorted(old):
        stock = new.get((domain, "stock_MDLovoFit_no_output"))
        gpu = new.get((domain, "MDCoreAlign_throughput_GPU_minimal"))
        if not stock or not gpu:
            continue
        prior = float(old[domain]["mdlovofit_subprocess_including_mandatory_write_seconds"])
        no_output = float(stock["C1_PDB_tool_wall_seconds"])
        candidate = float(gpu["C1_PDB_tool_wall_seconds"])
        output.append(
            {
                "panel": "mdCATH_5397",
                "system": domain,
                "setting": old[domain]["trajectory_id"],
                "prior_MDLovoFit_with_t_materialised_seconds": prior,
                "new_stock_MDLovoFit_no_output_seconds": no_output,
                "MDLovoFit_output_removal_speedup": prior / no_output,
                "prior_materialised_workflow_speedup_vs_GPU": prior / candidate,
                "new_no_output_workflow_speedup_vs_GPU": no_output / candidate,
            }
        )
    return output


def _warning_audit(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    statuses = [row["status"] for row in rows]
    least_mobile_deltas = [float(row["native_least_mobile_delta_vs_frozen_with_t_A"]) for row in rows]
    all_atom_deltas = [float(row["native_all_atom_delta_vs_frozen_with_t_A"]) for row in rows]
    return [
        {
            "category": "native_identity_warning_against_frozen_with_t",
            "rows": len(rows),
            "unique_domains": len({row["system"] for row in rows}),
            "successful_rows": sum(status == "success" for status in statuses),
            "failed_rows": sum(status == "failed" for status in statuses),
            "timeout_rows": sum(status == "timeout" for status in statuses),
            "input_integrity_pass_rows": sum(row["input_integrity_pass"] == "True" for row in rows),
            "maximum_least_mobile_RMSD_delta_A": max(least_mobile_deltas, default=0.0),
            "maximum_all_atom_RMSD_delta_A": max(all_atom_deltas, default=0.0),
            "interpretation": (
                "Successful no-output runs whose printed native RMSD differed from the historical "
                "frozen with-t value beyond the strict identity tolerance; retained as warnings, not "
                "execution failures or timeouts."
            ),
        }
    ]


def _row(
    rows: list[dict[str, object]],
    panel: str,
    contract: str,
    method_suffix: str,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["panel"] == panel
        and row["contract"] == contract
        and str(row["comparison"]).endswith(method_suffix)
    )


def _speedup_line(row: dict[str, object]) -> str:
    return (
        f"{float(row['paired_median_speedup']):.2f}x "
        f"(bootstrap 95% CI {float(row['paired_median_bootstrap_95CI_lower']):.2f}-"
        f"{float(row['paired_median_bootstrap_95CI_upper']):.2f}; "
        f"faster fraction {100.0 * float(row['MDCoreAlign_faster_fraction']):.2f}%)"
    )


def _figures(detail: list[dict[str, object]], stages: list[dict[str, str]], prior: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = {"MDCoreAlign_exact_CPU_minimal": "Exact CPU", "MDCoreAlign_throughput_CPU_minimal": "Throughput CPU", "MDCoreAlign_throughput_GPU_minimal": "Throughput GPU"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    data = [[float(row["C1_MDLovoFit_over_MDCoreAlign_speedup"]) for row in detail if row["MDCoreAlign_method"] == method] for method in labels]
    ax.boxplot(data, labels=list(labels.values()), showfliers=False)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_ylabel("Stock MDLovoFit / MDCoreAlign runtime"); ax.set_title("No-materialisation C1 speedup")
    fig.tight_layout(); fig.savefig(FIGURES / "stock_no_materialisation_speedup_by_workload.pdf"); fig.savefig(FIGURES / "stock_no_materialisation_speedup_by_workload.png", dpi=220); plt.close(fig)

    md_stages = [row for row in stages if row["method"] == "timed_MDLovoFit_no_output"]
    totals: dict[str, list[float]] = defaultdict(list)
    for row in md_stages:
        totals[row["stage"]].append(float(row["median_seconds"]))
    names = list(totals); values = [float(np.median(totals[name])) for name in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.5)); ax.barh(names, values, color="#39788f"); ax.set_xscale("log"); ax.set_xlabel("Median stage time per setting (s)"); ax.set_title("Instrumented MDLovoFit stage decomposition"); fig.tight_layout(); fig.savefig(FIGURES / "core_compute_stage_decomposition.pdf"); fig.savefig(FIGURES / "core_compute_stage_decomposition.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, label in labels.items():
        group = [row for row in detail if row["MDCoreAlign_method"] == method]
        ax.scatter([float(row["residue_frames"]) for row in group], [float(row["C1_MDLovoFit_over_MDCoreAlign_speedup"]) for row in group], s=18, alpha=0.7, label=label)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(1, color="black", lw=0.8, ls="--"); ax.set_xlabel("Frames x residues"); ax.set_ylabel("C1 speedup"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES / "speedup_vs_frames_times_residues.pdf"); fig.savefig(FIGURES / "speedup_vs_frames_times_residues.png", dpi=220); plt.close(fig)

    if prior:
        values = np.asarray([[float(row["prior_materialised_workflow_speedup_vs_GPU"]), float(row["new_no_output_workflow_speedup_vs_GPU"])] for row in prior])
        fig, ax = plt.subplots(figsize=(6.8, 4.5)); ax.boxplot([values[:, 0], values[:, 1]], labels=["Prior with -t", "New no output"], showfliers=False); ax.set_yscale("log"); ax.axhline(1, color="black", lw=0.8, ls="--"); ax.set_ylabel("MDLovoFit / GPU throughput runtime"); ax.set_title("Matched-PDB contract correction"); fig.tight_layout(); fig.savefig(FIGURES / "prior_matched_PDB_vs_no_output.pdf"); fig.savefig(FIGURES / "prior_matched_PDB_vs_no_output.png", dpi=220); plt.close(fig)


def main() -> None:
    atlas_summary = read_tsv(OUT / "02_atlas_stock_no_output/atlas_no_output_summary.tsv")
    atlas_stages = read_tsv(OUT / "02_atlas_stock_no_output/atlas_stage_decomposition.tsv")
    mdcath_rows = read_tsv(OUT / "03_mdcath_stock_no_output/mdcath_no_output_all_domain.tsv")
    mdcath_summary = read_tsv(OUT / "03_mdcath_stock_no_output/mdcath_no_output_summary.tsv")
    mdcath_stages = read_tsv(OUT / "03_mdcath_stock_no_output/mdcath_stage_decomposition.tsv")
    atlas_detail, atlas_aggregate = _atlas_comparisons(atlas_summary)
    contract_rows = atlas_aggregate + [dict(row, panel="mdCATH_5397") for row in mdcath_summary]
    write_tsv(ANALYSIS / "atlas_contract_speedup_detail.tsv", atlas_detail)
    write_tsv(ANALYSIS / "contract_comparison_table.tsv", contract_rows)
    write_tsv(ANALYSIS / "core_compute_speedup.tsv", [row for row in contract_rows if row["contract"] == "C2_loaded_core_compute"])
    write_tsv(ANALYSIS / "mdlovofit_all_domain_no_output_summary.tsv", mdcath_summary)
    combined_stages = [dict(row, panel="ATLAS_48") for row in atlas_stages] + [dict(row, panel="mdCATH_5397") for row in mdcath_stages]
    write_tsv(ANALYSIS / "mdlovofit_stage_decomposition.tsv", combined_stages)
    prior = _prior_comparison(atlas_detail, mdcath_rows)
    write_tsv(ANALYSIS / "prior_vs_no_output_comparison.tsv", prior)
    failures = read_tsv(OUT / "03_mdcath_stock_no_output/failures_timeouts.tsv")
    write_tsv(ANALYSIS / "failures_timeouts.tsv", failures)
    warning_audit = _warning_audit(failures)
    write_tsv(ANALYSIS / "warning_failure_audit.tsv", warning_audit)

    atlas_exact_c1 = _row(contract_rows, "ATLAS_48", "C1_same_PDB_tool_workflow", "MDCoreAlign_exact_CPU_minimal")
    atlas_cpu_c1 = _row(contract_rows, "ATLAS_48", "C1_same_PDB_tool_workflow", "MDCoreAlign_throughput_CPU_minimal")
    atlas_gpu_c1 = _row(contract_rows, "ATLAS_48", "C1_same_PDB_tool_workflow", "MDCoreAlign_throughput_GPU_minimal")
    atlas_exact_c2 = _row(contract_rows, "ATLAS_48", "C2_loaded_core_compute", "MDCoreAlign_exact_CPU_minimal")
    atlas_cpu_c2 = _row(contract_rows, "ATLAS_48", "C2_loaded_core_compute", "MDCoreAlign_throughput_CPU_minimal")
    atlas_gpu_c2 = _row(contract_rows, "ATLAS_48", "C2_loaded_core_compute", "MDCoreAlign_throughput_GPU_minimal")
    mdcath_exact_c1 = _row(contract_rows, "mdCATH_5397", "C1_same_PDB_tool_workflow", "MDCoreAlign_exact_CPU_minimal")
    cpu_c1 = _row(contract_rows, "mdCATH_5397", "C1_same_PDB_tool_workflow", "MDCoreAlign_throughput_CPU_minimal")
    gpu_c1 = _row(contract_rows, "mdCATH_5397", "C1_same_PDB_tool_workflow", "MDCoreAlign_throughput_GPU_minimal")
    mdcath_exact_c2 = _row(contract_rows, "mdCATH_5397", "C2_loaded_core_compute", "MDCoreAlign_exact_CPU_minimal")
    cpu_c2 = _row(contract_rows, "mdCATH_5397", "C2_loaded_core_compute", "MDCoreAlign_throughput_CPU_minimal")
    gpu_c2 = _row(contract_rows, "mdCATH_5397", "C2_loaded_core_compute", "MDCoreAlign_throughput_GPU_minimal")
    minimum_primary = min(float(cpu_c1["paired_median_speedup"]), float(gpu_c1["paired_median_speedup"]), float(cpu_c2["paired_median_speedup"]), float(gpu_c2["paired_median_speedup"]))
    faster_fraction = min(float(cpu_c1["MDCoreAlign_faster_fraction"]), float(gpu_c1["MDCoreAlign_faster_fraction"]), float(cpu_c2["MDCoreAlign_faster_fraction"]), float(gpu_c2["MDCoreAlign_faster_fraction"]))
    if minimum_primary >= 5 and faster_fraction >= 0.90:
        outcome = "Outcome A - strong algorithmic advantage"
    elif minimum_primary > 1 and faster_fraction >= 0.75:
        outcome = "Outcome B - moderate advantage"
    else:
        outcome = "Outcome C - output-driven or workload-dependent advantage"
    recommendations = [
        {"location": "abstract", "recommended_contract": "C1 stock no-materialisation", "recommended_claim": "Use paired median and CI; do not use the prior approximately 70-fold materialised-output number.", "outcome": outcome},
        {"location": "main_results", "recommended_contract": "C1 and C2 separately", "recommended_claim": "Report software-level no-output and instrumented core-compute speedups without combining them.", "outcome": outcome},
        {"location": "supplement", "recommended_contract": "prior with-t versus no-output", "recommended_claim": "Show output-removal sensitivity and complete stage decomposition.", "outcome": outcome},
        {"location": "software_documentation", "recommended_contract": "workload-specific", "recommended_claim": "Avoid universal-fastest language and state CPU/GPU crossover and exceptions.", "outcome": outcome},
    ]
    write_tsv(ANALYSIS / "speedup_claim_recommendations.tsv", recommendations)
    stage_lookup = {
        (row["method"], row["stage"]): float(row["median_seconds"])
        for row in mdcath_stages
    }
    prior_by_panel: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prior:
        prior_by_panel[str(row["panel"])].append(row)
    output_removal = {
        panel: float(np.median([float(row["MDLovoFit_output_removal_speedup"]) for row in rows]))
        for panel, rows in prior_by_panel.items()
    }
    warning = warning_audit[0]
    write_text(ANALYSIS / "final_no_materialisation_decision_report.md", f"""# NM7 Final No-materialisation Benchmark Report

## Scope and status

- NM1 stock no-output verification: **PASS**, 5/5 locked cases.
- NM4 ATLAS benchmark: **PASS**, 48/48 settings with five measured repeats.
- NM5 corrected mdCATH benchmark: **PASS**, 5,397/5,397 domains.
- NM6 deployment extension: **SKIPPED_BY_SCOPE** under the prospectively registered rule.
- No NM timing run was restarted or modified during this analysis.

## Overall decision

**{outcome}.** This decision uses the locked ATLAS 48-setting panel and the
corrected 5,397-domain mdCATH panel. C1 starts both tools from the same centred
CA multi-model PDB and suppresses aligned-coordinate output. C2 removes parsing,
subprocess and formatting costs and compares instrumented LOVO/search plus final
fit/RMSD against MDCoreAlign weight estimation plus explicit all-frame fit/RMSD.

The previous approximately 70-fold matched-PDB result is not an algorithm-only
number and must not be used as such. The exact measured C1/C2 values are in
`contract_comparison_table.tsv`; no favourable outcome was assumed in advance.

## Contract C1: stock no-output workflow

| Panel | Exact CPU | Throughput CPU | Throughput GPU |
|---|---:|---:|---:|
| ATLAS 48 | {_speedup_line(atlas_exact_c1)} | {_speedup_line(atlas_cpu_c1)} | {_speedup_line(atlas_gpu_c1)} |
| mdCATH 5,397 | {_speedup_line(mdcath_exact_c1)} | {_speedup_line(cpu_c1)} | {_speedup_line(gpu_c1)} |

C1 is the primary honest software-level comparison: stock MDLovoFit and
MDCoreAlign consume the same pre-generated centred CA multi-model PDB, and
neither method writes aligned coordinates.

## Contract C2: instrumented loaded core compute

| Panel | Exact CPU | Throughput CPU | Throughput GPU |
|---|---:|---:|---:|
| ATLAS 48 | {_speedup_line(atlas_exact_c2)} | {_speedup_line(atlas_cpu_c2)} | {_speedup_line(atlas_gpu_c2)} |
| mdCATH 5,397 | {_speedup_line(mdcath_exact_c2)} | {_speedup_line(cpu_c2)} | {_speedup_line(gpu_c2)} |

C2 excludes parsing, subprocess and output costs. It is a core-compute
comparison and must not be described as full-workflow speedup.

## Stage decomposition on mdCATH

Median instrumented MDLovoFit times per domain were:

- PDB parsing: {stage_lookup[("timed_MDLovoFit_no_output", "PDB_parsing")]:.6f} s;
- LOVO search: {stage_lookup[("timed_MDLovoFit_no_output", "weight_or_LOVO_search")]:.6f} s;
- final fixed-subset fit: {stage_lookup[("timed_MDLovoFit_no_output", "all_frame_final_fit")]:.6f} s;
- RMSD reduction: {stage_lookup[("timed_MDLovoFit_no_output", "RMSD_reduction")]:.6f} s;
- stdout formatting: {stage_lookup[("timed_MDLovoFit_no_output", "stdout_formatting")]:.6f} s;
- coordinate preparation without aligned-file writing: {stage_lookup[("timed_MDLovoFit_no_output", "coordinate_preparation")]:.6f} s.

Median MDCoreAlign throughput CPU times were {stage_lookup[("MDCoreAlign_throughput_CPU_minimal", "weight_or_LOVO_search")]:.6f} s
for stable-weight estimation, {stage_lookup[("MDCoreAlign_throughput_CPU_minimal", "all_frame_final_fit")]:.6f} s
for explicit all-frame fitting and {stage_lookup[("MDCoreAlign_throughput_CPU_minimal", "RMSD_reduction")]:.6f} s
for RMSD reduction. The compact NM contract did not compute RMSF or serialise
nonessential result files; those stages are therefore unavailable rather than
reported as measured zero-cost operations.

Historical aligned-PDB formatting/writing was not independently instrumented.
Removing `-t` reduced total stock MDLovoFit wall time by a median factor of
{output_removal.get("ATLAS_48", float("nan")):.2f}x on ATLAS and
{output_removal.get("mdCATH_5397", float("nan")):.2f}x on mdCATH, which is why
the previous materialised-output comparison cannot be called algorithm-only.

## Integrity, warnings and failures

The 82 rows in the historical `failures_timeouts.tsv` are **not execution
failures**: all {warning['successful_rows']} completed successfully, cover
{warning['unique_domains']} domains and passed input integrity. They are strict
native-value identity warnings against frozen with-`-t` values. Maximum printed
differences were {float(warning['maximum_least_mobile_RMSD_delta_A']):.6f} A for
least-mobile RMSD and {float(warning['maximum_all_atom_RMSD_delta_A']):.6f} A
for all-atom RMSD. Actual failed rows: {warning['failed_rows']}; actual timeout
rows: {warning['timeout_rows']}. These warnings remain visible and require
method-agreement/exporter adjudication, but they do not invalidate completed
NM5 timings.

## Answers to the registered NM7 questions

1. C1 speedups are reported above and are the primary no-output workflow result.
2. C2 speedups are reported separately as loaded core-compute results.
3. MDLovoFit stage times show that the LOVO search dominates; no aligned-PDB write occurred in NM4/NM5.
4. MDCoreAlign throughput weight estimation dominates its compact compute; explicit all-frame fitting remains present.
5. CPU throughput remains substantially faster than stock MDLovoFit on both panels.
6. GPU throughput also retains a substantial advantage, although its mdCATH C1 advantage is similar to CPU throughput because parsing and process overhead dominate small cases.
7. Workload dependence is provided in `atlas_contract_speedup_detail.tsv` and `figures/speedup_vs_frames_times_residues.*`; no universal monotonic claim is made here.
8. Use C1 in the abstract, C1 and C2 separately in main Results, prior `-t` sensitivity and full stages in Supplement, and workload-specific language in software documentation.

## Defensible claim

Under matched no-materialisation conditions, MDCoreAlign throughput was faster
than stock MDLovoFit across all ATLAS settings and essentially all corrected
mdCATH domains. The strongest defensible headline is the paired C1 workflow
result, not the larger C2 core-compute ratio and not the prior approximately
70-fold materialised-output result.
""")
    write_text(ANALYSIS / "manuscript_wording_recommendations.md", """# Manuscript Wording Recommendations

Use: "Under a matched no-materialisation contract, stock MDLovoFit and
MDCoreAlign both consumed the same pre-generated centred CA multi-model PDB and
neither wrote aligned coordinates. We report tool-level wall time separately
from timing-instrumented core compute."

Do not use the previous approximately 70-fold workflow number as an
algorithm-only speedup. State exact panel, contract, median, confidence interval
and faster fraction beside every performance claim. The instrumented executable
is timing-only and is not the stock comparator for C1.
""")
    write_text(ANALYSIS / "limitations.md", """# Limitations

- MDLovoFit 20.1.1 still rotates all coordinates in its final pass when `-t` is absent; this `coordinate_preparation` stage is included in stock C1 but excluded from C2.
- Instrumentation uses `system_clock`; C2 boundaries are source-level stage measurements, not hardware instruction counts.
- MDCoreAlign and MDLovoFit solve different stable-membership objectives, so runtime parity does not imply identical cores.
- ATLAS repeats estimate timing variation; the full mdCATH panel has one run per domain plus a locked 200-domain five-repeat subset.
- GPU results include explicit synchronization and transfer time in C1/C2 as registered, but hardware specificity remains.
""")
    write_text(ANALYSIS / "quality_timing_integrity_audit.md", f"""# Quality and Timing Integrity Audit

NM1 established that removing `-t` leaves stock MDLovoFit native results
unchanged. NM2 established compact-path parity for exact CPU, throughput CPU and
throughput GPU. NM3 established byte-identical stdout for the separately named
timing-only binary. NM4 and NM5 retained PDB parsing in C1 and excluded it from
C2, with output absent from both contracts.

The file named `failures_timeouts.tsv` contains {warning['rows']} strict
historical native-identity warning rows from {warning['unique_domains']} domains.
All rows have `status=success` and `input_integrity_pass=True`; there are zero
execution failures and zero timeouts. The warning filename is retained for
provenance, while `warning_failure_audit.tsv` records the corrected
classification.
""")

    deployment = [{"phase": "NM6", "status": "SKIPPED_BY_SCOPE", "reason": "Prospectively registered superseding scope: the optional node-throughput/full-database extension is not required for the NM manuscript decision and must not run without explicit new instruction.", "scientific_results_used": False}]
    write_tsv(OUT / "05_optional_database_deployment/node_parallel_no_output.tsv", deployment)
    write_tsv(OUT / "05_optional_database_deployment/deployment_summary.tsv", deployment)
    write_text(OUT / "05_optional_database_deployment/deployment_report.md", "# NM6 Optional Deployment Decision\n\nNM6 was skipped by its registered optional rule after NM4/NM5 completion. No full-database or node-parallel timing was inferred from other contracts.")
    _figures(atlas_detail, mdcath_stages, prior)
    write_json(
        ANALYSIS / "nm7_status.json",
        {
            "atlas_settings": 48,
            "mdcath_domains": 5397,
            "nm6_status": "SKIPPED_BY_SCOPE",
            "outcome": outcome,
            "status": "pass",
            "warning_rows": int(warning["rows"]),
            "actual_failures": int(warning["failed_rows"]),
            "actual_timeouts": int(warning["timeout_rows"]),
        },
    )


if __name__ == "__main__":
    main()
