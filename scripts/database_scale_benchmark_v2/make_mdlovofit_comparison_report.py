#!/usr/bin/env python3
"""Assemble the M4 MDLovoFit comparison tables after all registered runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.database_scale_benchmark_v2.common import (
    M_ROOT,
    ROOT,
    read_tsv,
    write_json,
    write_tsv,
)


D1 = ROOT / "results/database_scale_gpu_v1/d1_all_domain/d1_all_domain_matched.tsv"
CORRECTED_HISTORICAL = (
    ROOT
    / "results/project_completion_v2_unit_corrected/aggregate/"
    "all_domain_unit_corrected_per_domain.tsv"
)
U1_SUMMARY = (
    ROOT
    / "results/gpu_matched_validation_v1/u1_matched_contracts/"
    "matched_contract_summary.tsv"
)


def _float(row: dict[str, Any], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _residue_bin(n_residues: int) -> str:
    if n_residues < 100:
        return "lt100"
    if n_residues < 200:
        return "100-199"
    if n_residues < 400:
        return "200-399"
    return "ge400"


def join_all_domain() -> list[dict[str, Any]]:
    mdlovo = {
        row["trajectory_id"]: row
        for row in read_tsv(M_ROOT / "mdlovofit_all_domain_corrected.tsv")
    }
    d1 = {row["trajectory_id"]: row for row in read_tsv(D1)}
    historical = {
        row["trajectory_id"]: row for row in read_tsv(CORRECTED_HISTORICAL)
    }
    if not set(mdlovo).issubset(d1):
        raise RuntimeError(
            f"all-domain D1 evidence is missing {len(set(mdlovo) - set(d1))} "
            "locked MDLovoFit trajectories"
        )
    rows: list[dict[str, Any]] = []
    for trajectory_id in sorted(mdlovo):
        external = mdlovo[trajectory_id]
        internal = d1[trajectory_id]
        old = historical.get(trajectory_id, {})
        n_residues = int(internal["n_residues"])
        exact_seconds = _float(internal, "exact_profile_load_seconds") + _float(
            internal, "exact_fixed_weight_fit_seconds"
        )
        mdlovo_seconds = _float(external, "loaded_analysis_seconds")
        mdlovo_rmsd = _float(external, "unified_posthoc_RMSD_A")
        row: dict[str, Any] = {
            "domain_id": internal["system_id"],
            "trajectory_id": trajectory_id,
            "cath_class": internal["cath_class"],
            "n_frames": int(internal["n_frames"]),
            "n_residues": n_residues,
            "residue_count_bin": _residue_bin(n_residues),
            "mdlovofit_status": external["status"],
            "shared_HDF5_decode_seconds": external.get(
                "shared_HDF5_decode_seconds", ""
            ),
            "mdlovofit_input_preparation_seconds": external.get(
                "method_specific_input_preparation_seconds", ""
            ),
            "mdlovofit_subprocess_including_mandatory_write_seconds": external.get(
                "mdlovofit_subprocess_including_mandatory_output_write_seconds", ""
            ),
            "mdlovofit_output_parse_seconds": external.get(
                "output_parse_seconds", ""
            ),
            "mdlovofit_posthoc_evaluation_seconds": external.get(
                "posthoc_evaluation_seconds", ""
            ),
            "mdlovofit_loaded_analysis_seconds": mdlovo_seconds,
            "mdlovofit_complete_end_to_end_seconds": external.get(
                "complete_end_to_end_seconds", ""
            ),
            "mdlovofit_unified_posthoc_RMSD_A": mdlovo_rmsd,
            "mdlovofit_native_all_atom_RMSD_A": external.get(
                "native_all_atom_RMSD_A", ""
            ),
            "mdlovofit_native_least_mobile_RMSD_A": external.get(
                "native_least_mobile_RMSD_A", ""
            ),
            "exact_profile_reuse_plus_fixed_fit_seconds": exact_seconds,
            "exact_unified_posthoc_RMSD_A": _float(
                internal, "exact_mean_unified_posthoc_rmsd_A"
            ),
            "cpu_throughput_loaded_seconds": _float(
                internal, "cpu_throughput_loaded_compute_seconds"
            ),
            "cpu_throughput_unified_posthoc_RMSD_A": _float(
                internal, "cpu_mean_unified_posthoc_rmsd_A"
            ),
            "gpu_throughput_loaded_seconds": _float(
                internal, "gpu_throughput_loaded_compute_seconds"
            ),
            "gpu_throughput_unified_posthoc_RMSD_A": _float(
                internal, "gpu_mean_unified_posthoc_rmsd_A"
            ),
            "theseus_status_validated_historical": old.get("theseus_status", ""),
            "theseus_runtime_seconds_validated_historical": old.get(
                "theseus_runtime_sec", ""
            ),
            "theseus_unified_posthoc_RMSD_A_validated_historical": old.get(
                "theseus_unified_RMSD", ""
            ),
            "theseus_current_addendum_rerun": False,
            "mdlovofit_mask_semantics_used": False,
            "exact_timing_caveat": (
                "Frozen exact profile load plus matched fixed-weight all-frame fit; "
                "not a fresh sequential core-identification rerun."
            ),
        }
        for method in ("exact", "cpu_throughput", "gpu_throughput"):
            runtime_field = {
                "exact": "exact_profile_reuse_plus_fixed_fit_seconds",
                "cpu_throughput": "cpu_throughput_loaded_seconds",
                "gpu_throughput": "gpu_throughput_loaded_seconds",
            }[method]
            rmsd_field = {
                "exact": "exact_unified_posthoc_RMSD_A",
                "cpu_throughput": "cpu_throughput_unified_posthoc_RMSD_A",
                "gpu_throughput": "gpu_throughput_unified_posthoc_RMSD_A",
            }[method]
            row[f"{method}_runtime_ratio_vs_mdlovofit"] = (
                float(row[runtime_field]) / mdlovo_seconds
                if mdlovo_seconds > 0
                else float("nan")
            )
            row[f"{method}_faster_than_mdlovofit"] = (
                float(row[runtime_field]) < mdlovo_seconds
            )
            row[f"{method}_RMSD_delta_vs_mdlovofit_A"] = (
                float(row[rmsd_field]) - mdlovo_rmsd
            )
            row[f"{method}_lower_RMSD_than_mdlovofit"] = (
                float(row[rmsd_field]) < mdlovo_rmsd
            )
        rows.append(row)
    write_tsv(M_ROOT / "mdlovofit_all_domain_corrected_matched.tsv", rows)
    return rows


def _summaries(values: Iterable[float], name: str) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return {
        f"median_{name}": float(np.median(data)),
        f"q1_{name}": float(np.quantile(data, 0.25)),
        f"q3_{name}": float(np.quantile(data, 0.75)),
        f"iqr_{name}": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
    }


def summarize_all_domain(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strata: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", rows)]
    for field in ("cath_class", "residue_count_bin"):
        for value in sorted({str(row[field]) for row in rows}):
            strata.append((field, value, [row for row in rows if str(row[field]) == value]))
    methods = {
        "exact_profile_reuse_plus_fixed_fit": (
            "exact_profile_reuse_plus_fixed_fit_seconds",
            "exact_unified_posthoc_RMSD_A",
        ),
        "CPU_throughput": (
            "cpu_throughput_loaded_seconds",
            "cpu_throughput_unified_posthoc_RMSD_A",
        ),
        "GPU_throughput": (
            "gpu_throughput_loaded_seconds",
            "gpu_throughput_unified_posthoc_RMSD_A",
        ),
        "MDLovoFit_20.1.1": (
            "mdlovofit_loaded_analysis_seconds",
            "mdlovofit_unified_posthoc_RMSD_A",
        ),
        "THESEUS_validated_historical": (
            "theseus_runtime_seconds_validated_historical",
            "theseus_unified_posthoc_RMSD_A_validated_historical",
        ),
    }
    summaries: list[dict[str, Any]] = []
    for stratum, value, group in strata:
        for method, (runtime_field, rmsd_field) in methods.items():
            valid = [
                row
                for row in group
                if np.isfinite(_float(row, runtime_field))
                and np.isfinite(_float(row, rmsd_field))
            ]
            runtimes = np.asarray([_float(row, runtime_field) for row in valid])
            rmsds = np.asarray([_float(row, rmsd_field) for row in valid])
            paired_external_runtime = np.asarray(
                [float(row["mdlovofit_loaded_analysis_seconds"]) for row in valid]
            )
            paired_external_rmsd = np.asarray(
                [float(row["mdlovofit_unified_posthoc_RMSD_A"]) for row in valid]
            )
            summaries.append(
                {
                    "panel": "corrected_all_domain_5397",
                    "stratum": stratum,
                    "stratum_value": value,
                    "method": method,
                    "n": len(valid),
                    **_summaries(runtimes, "runtime_seconds"),
                    **_summaries(rmsds, "unified_posthoc_RMSD_A"),
                    "median_runtime_ratio_vs_MDLovoFit": float(
                        np.median(runtimes / paired_external_runtime)
                    ),
                    "fraction_faster_than_MDLovoFit": float(
                        np.mean(runtimes < paired_external_runtime)
                    ),
                    "median_RMSD_delta_vs_MDLovoFit_A": float(
                        np.median(rmsds - paired_external_rmsd)
                    ),
                    "fraction_lower_RMSD_than_MDLovoFit": float(
                        np.mean(rmsds < paired_external_rmsd)
                    ),
                    "timing_boundary": (
                        "MDLovoFit loaded analysis includes CA-PDB generation, "
                        "subprocess with mandatory aligned-PDB writing, parse and post-hoc evaluation; "
                        "CPU/GPU rows are loaded-coordinate analysis."
                    ),
                }
            )
    return summaries


def real_long_matched_rows() -> list[dict[str, Any]]:
    internal = read_tsv(M_ROOT / "internal_real_long_prefixes.tsv")
    external = read_tsv(M_ROOT / "mdlovofit_real_long_prefixes.tsv")
    rows: list[dict[str, Any]] = [dict(row) for row in internal]
    for row in external:
        common = {
            **row,
            "method": "MDLovoFit_20.1.1",
            "median_unified_posthoc_RMSD_A": row.get(
                "median_unified_posthoc_RMSD_A", ""
            ),
            "native_all_atom_RMSD_A": row.get(
                "median_native_all_atom_RMSD_A", ""
            ),
            "native_least_mobile_RMSD_A": row.get(
                "median_native_least_mobile_RMSD_A", ""
            ),
        }
        rows.append(
            {
                **common,
                "contract": "A_analysis_metrics_transforms_rmsd_rmsf",
                "median_loaded_contract_seconds": row.get(
                    "median_loaded_analysis_seconds", ""
                ),
                "timing_boundary": (
                    "External input preparation, subprocess with mandatory aligned-PDB "
                    "write, parse and post-hoc evaluation included."
                ),
            }
        )
        rows.append(
            {
                **common,
                "contract": "B_materialised_aligned_PDB",
                "median_loaded_contract_seconds": row.get(
                    "median_materialised_output_contract_seconds", ""
                ),
                "timing_boundary": (
                    "External input preparation, subprocess and mandatory materialised "
                    "aligned-PDB output plus verification included."
                ),
            }
        )
    if U1_SUMMARY.is_file():
        for row in read_tsv(U1_SUMMARY):
            if row["method"] != "THESEUS_3.3.0" or row["prefix_label"] != "p0250":
                continue
            rows.append(
                {
                    **row,
                    "prefix_label": "p00250",
                    "prefix_frames": 250,
                    "validated_source": str(U1_SUMMARY.relative_to(ROOT)),
                    "current_addendum_rerun": False,
                    "timing_boundary": row.get("timing_boundary", ""),
                }
            )
    write_tsv(M_ROOT / "mdlovofit_real_long_matched_contracts.tsv", rows)
    return rows


def collect_failures() -> list[dict[str, Any]]:
    sources = (
        M_ROOT / "mdlovofit_all_domain_corrected.tsv",
        M_ROOT / "mdlovofit_external_panel.tsv",
        M_ROOT / "mdlovofit_real_long_prefixes_raw_repeats.tsv",
    )
    failures: list[dict[str, Any]] = []
    for source in sources:
        for row in read_tsv(source):
            if row["status"] != "success":
                failures.append({"source_table": source.name, **row})
    if not failures:
        failures = [
            {
                "source_table": "all_registered_MDLovoFit_runs",
                "status": "none",
                "failure_class": "",
                "failure_reason": "No failure or timeout was observed.",
            }
        ]
    write_tsv(M_ROOT / "mdlovofit_failures_timeouts.tsv", failures)
    return failures


def main() -> None:
    joined = join_all_domain()
    all_domain_summary = summarize_all_domain(joined)
    real_long = real_long_matched_rows()
    write_tsv(
        M_ROOT / "mdlovofit_matched_contract_summary.tsv",
        [*all_domain_summary, *real_long],
    )
    failures = collect_failures()
    pilot = json.loads(
        (M_ROOT / "mdlovofit_deployment_execution_decision.json").read_text()
    )
    deployment = json.loads((M_ROOT / "mdlovofit_deployment_status.json").read_text())
    real_status = json.loads((M_ROOT / "mdlovofit_real_long_status.json").read_text())
    overall = next(
        row
        for row in all_domain_summary
        if row["stratum"] == "all" and row["method"] == "GPU_throughput"
    )
    report = [
        "# MDLovoFit comparison addendum report",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Registered execution",
        "",
        f"- Corrected all-domain panel: `{len(joined):,}` paired domains.",
        f"- Deployment pilot: `{pilot['pilot_rows']:,}` trajectories; decision: `{pilot['decision']}`.",
        f"- Deployment panel completed: `{deployment['rows']:,}` trajectories.",
        f"- Real-long prefix settings: `{real_status['settings']}` across `{real_status['systems']}` systems.",
        "- No local real-continuous input reached 10,000 frames; this requested prefix is explicitly unavailable rather than synthesized.",
        "",
        "## Primary external comparison",
        "",
        f"- GPU-throughput median runtime ratio versus MDLovoFit on the corrected all-domain panel: `{overall['median_runtime_ratio_vs_MDLovoFit']:.6g}`.",
        f"- GPU-throughput fraction faster than MDLovoFit: `{overall['fraction_faster_than_MDLovoFit']:.2%}`.",
        f"- GPU-throughput median unified RMSD delta versus MDLovoFit: `{overall['median_RMSD_delta_vs_MDLovoFit_A']:.6g} A`.",
        f"- GPU-throughput fraction with lower unified post-hoc RMSD: `{overall['fraction_lower_RMSD_than_MDLovoFit']:.2%}`.",
        "",
        "## Timing and interpretation safeguards",
        "",
        "- Shared HDF5/cache decode is reported separately and is not charged repeatedly to one method.",
        "- MDLovoFit requires an aligned-PDB output. Its subprocess therefore contains both alignment and unavoidable output writing; these components cannot be separated without modifying the external executable.",
        "- The all-domain CPU/GPU timings are loaded-coordinate analysis timings, while the real-long table provides explicit analysis and materialised-output contracts.",
        "- The exact all-domain audit timing reuses frozen exact profiles and adds a matched fixed-weight fit; it is not presented as a fresh sequential core-identification runtime.",
        "- THESEUS rows are validated corrected historical evidence and were not rerun in this addendum.",
        "- MDLovoFit occupancy/core-mask semantics were not used for full-scale mask comparison.",
        "- No universal-fastest or 71-fold claim is made.",
        "",
        "## Failures and timeouts",
        "",
        f"- Recorded failure/timeout rows: `{sum(row['status'] != 'none' for row in failures):,}`.",
        "",
    ]
    (M_ROOT / "mdlovofit_comparison_report.md").write_text("\n".join(report))
    write_json(
        M_ROOT / "mdlovofit_addendum_completion_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "all_domain_rows": len(joined),
            "deployment_rows": deployment["rows"],
            "real_long_settings": real_status["settings"],
            "failure_timeout_rows": sum(row["status"] != "none" for row in failures),
            "primary_external_comparator": "MDLovoFit_20.1.1",
            "universal_fastest_claim": False,
            "claim_71x": False,
        },
    )


if __name__ == "__main__":
    main()
