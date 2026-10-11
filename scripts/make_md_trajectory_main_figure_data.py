"""Build compact MD-trajectory main-figure data tables for submission polish."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import fmt, read_table, to_float, write_table  # noqa: E402
from md_trajectory_focus_common import timestamp  # noqa: E402


OUT_DIR = ROOT / "results" / "submission_polish" / "main_figures"
SOURCE_DIR = ROOT / "results" / "md_trajectory_focus"
FINAL_SUMMARY = ROOT / "results" / "final_multi_system_summary_20260603" / "final_real_trajectory_summary.csv"
FLEX_SUMMARY = ROOT / "results" / "flexibility" / "flexibility_summary_four_systems.tsv"
FLEX_STATS = ROOT / "results" / "flexibility" / "flexibility_statistics_four_systems.tsv"
CONVERGENCE_CLASSIFICATION = (
    ROOT
    / "results"
    / "submission_polish"
    / "temporal_convergence"
    / "temporal_convergence_classification.tsv"
)

FIG2_FIELDS = [
    "system",
    "method",
    "runtime_seconds",
    "algorithm_runtime_seconds",
    "unified_posthoc_rmsd",
    "weighted_rmsd",
    "mdlovofit_native_all_atom_rmsd",
    "mdlovofit_native_least_mobile_rmsd",
    "iteration_count",
    "core_fraction",
    "runtime_ratio_vs_soft",
    "rmsd_delta_vs_soft",
    "source_file",
    "notes",
]

FIG3_FIELDS = [
    "system",
    "diagnostic",
    "n_rows",
    "core_jaccard_min",
    "core_jaccard_max",
    "mobility_spearman_min",
    "mobility_spearman_max",
    "flexible_overlap_min",
    "flexible_overlap_max",
    "length_threshold_fraction_at_core_jaccard_ge_0_80",
    "warm_start_runtime_ratio_warm_over_cold",
    "warm_start_iteration_delta_warm_minus_cold",
    "warm_start_output_preserved",
    "nonconverged_rows",
    "classification_counts",
    "interpretation",
    "source_file",
]

FIG4_FIELDS = [
    "system",
    "n_residues",
    "n_flexible_segments",
    "spearman_robust_rmsf_vs_mobility",
    "spearman_ordinary_rmsf_vs_mobility",
    "mean_robust_rmsf",
    "mean_ordinary_rmsf",
    "mean_robust_rmsf_core",
    "mean_robust_rmsf_noncore",
    "delta_mean_noncore_minus_core",
    "top_flexible_residue",
    "top_flexible_residue_rmsf",
    "top_flexible_segment",
    "notes",
    "source_file",
]


def _finite(values: list[float | None]) -> list[float]:
    """Return finite values."""

    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _min(values: list[float | None]) -> object:
    """Return compact minimum or blank."""

    items = _finite(values)
    return fmt(min(items)) if items else ""


def _max(values: list[float | None]) -> object:
    """Return compact maximum or blank."""

    items = _finite(values)
    return fmt(max(items)) if items else ""


def _system_rows(rows: list[dict[str, str]], system: str) -> list[dict[str, str]]:
    """Filter rows for one system."""

    return [row for row in rows if row.get("system") == system]


def _classification_counts() -> dict[str, str]:
    """Return per-system temporal non-convergence classification counts."""

    if not CONVERGENCE_CLASSIFICATION.exists():
        return {}
    counts: dict[str, dict[str, int]] = {}
    for row in read_table(CONVERGENCE_CLASSIFICATION):
        system = row["system"]
        label = row["classification"]
        counts.setdefault(system, {})
        counts[system][label] = counts[system].get(label, 0) + 1
    return {
        system: ", ".join(f"{label}:{count}" for label, count in sorted(system_counts.items()))
        for system, system_counts in counts.items()
    }


def build_fig2_rows() -> list[dict[str, object]]:
    """Build the benchmark summary table for main figure 2."""

    rows: list[dict[str, object]] = []
    for row in read_table(FINAL_SUMMARY):
        system = row["system"]
        soft_runtime = to_float(row.get("soft_runtime_s"))
        soft_rmsd = to_float(row.get("soft_unified_rmsd"))
        methods = [
            (
                "MDCoreAlign soft robust",
                soft_runtime,
                soft_rmsd,
                to_float(row.get("soft_weighted_rmsd")),
                "",
                "",
                row.get("soft_iterations", ""),
                row.get("soft_core_fraction", ""),
                "primary method",
            ),
            (
                "External MDLovoFit",
                to_float(row.get("mdlovofit_runtime_s")),
                to_float(row.get("mdlovofit_unified_rmsd")),
                "",
                row.get("mdlovofit_native_all_atom_rmsd", ""),
                row.get("mdlovofit_native_least_mobile_rmsd", ""),
                "",
                "",
                "external least-mobile-subset baseline",
            ),
            (
                "THESEUS",
                to_float(row.get("theseus_runtime_s")),
                to_float(row.get("theseus_unified_rmsd")),
                "",
                "",
                "",
                "",
                "",
                "independent probabilistic-superposition baseline; rerun caveat documented separately",
            ),
        ]
        for method, runtime, rmsd, weighted, native_all, native_core, iterations, core_fraction, notes in methods:
            runtime_ratio = runtime / soft_runtime if runtime is not None and soft_runtime else None
            rmsd_delta = rmsd - soft_rmsd if rmsd is not None and soft_rmsd is not None else None
            rows.append(
                {
                    "system": system,
                    "method": method,
                    "runtime_seconds": fmt(runtime),
                    "algorithm_runtime_seconds": fmt(runtime),
                    "unified_posthoc_rmsd": fmt(rmsd),
                    "weighted_rmsd": fmt(weighted),
                    "mdlovofit_native_all_atom_rmsd": native_all,
                    "mdlovofit_native_least_mobile_rmsd": native_core,
                    "iteration_count": iterations,
                    "core_fraction": core_fraction,
                    "runtime_ratio_vs_soft": fmt(runtime_ratio),
                    "rmsd_delta_vs_soft": fmt(rmsd_delta),
                    "source_file": row.get("source_file", str(FINAL_SUMMARY)),
                    "notes": notes,
                }
            )
    return rows


def _diagnostic_summary_row(
    *,
    system: str,
    diagnostic: str,
    rows: list[dict[str, str]],
    core_field: str,
    mobility_field: str,
    flexible_field: str,
    source_file: Path,
    interpretation: str,
    classification: str = "",
) -> dict[str, object]:
    """Build a generic diagnostic summary row."""

    nonconverged = sum(str(row.get("converged", "")).lower() == "false" for row in rows)
    return {
        "system": system,
        "diagnostic": diagnostic,
        "n_rows": len(rows),
        "core_jaccard_min": _min([to_float(row.get(core_field)) for row in rows]),
        "core_jaccard_max": _max([to_float(row.get(core_field)) for row in rows]),
        "mobility_spearman_min": _min([to_float(row.get(mobility_field)) for row in rows]),
        "mobility_spearman_max": _max([to_float(row.get(mobility_field)) for row in rows]),
        "flexible_overlap_min": _min([to_float(row.get(flexible_field)) for row in rows]),
        "flexible_overlap_max": _max([to_float(row.get(flexible_field)) for row in rows]),
        "length_threshold_fraction_at_core_jaccard_ge_0_80": "",
        "warm_start_runtime_ratio_warm_over_cold": "",
        "warm_start_iteration_delta_warm_minus_cold": "",
        "warm_start_output_preserved": "",
        "nonconverged_rows": nonconverged,
        "classification_counts": classification,
        "interpretation": interpretation,
        "source_file": str(source_file),
    }


def _length_threshold(rows: list[dict[str, str]], threshold: float = 0.80) -> str:
    """Return the smallest prefix fraction at which core Jaccard reaches threshold."""

    candidates: list[tuple[float, float]] = []
    for row in rows:
        fraction = to_float(row.get("prefix_fraction"))
        jaccard = to_float(row.get("jaccard_core_vs_full"))
        if fraction is not None and jaccard is not None:
            candidates.append((fraction, jaccard))
    for fraction, jaccard in sorted(candidates):
        if jaccard >= threshold:
            return str(fmt(fraction))
    return "not_reached"


def _warm_start_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Build warm-start diagnostic rows."""

    output: list[dict[str, object]] = []
    systems = sorted({row["system"] for row in rows})
    for system in systems:
        system_rows = _system_rows(rows, system)
        warm = next((row for row in system_rows if row.get("warm_start") == "True"), None)
        cold = next((row for row in system_rows if row.get("warm_start") == "False"), None)
        if warm is None or cold is None:
            continue
        warm_runtime = to_float(warm.get("runtime_algorithm_seconds"))
        cold_runtime = to_float(cold.get("runtime_algorithm_seconds"))
        warm_iter = to_float(warm.get("n_iterations"))
        cold_iter = to_float(cold.get("n_iterations"))
        core_jaccard = to_float(cold.get("jaccard_core_vs_warm_start_default"))
        mobility_spearman = to_float(cold.get("spearman_mobility_vs_warm_start_default"))
        preserved = bool(
            core_jaccard is not None
            and mobility_spearman is not None
            and core_jaccard >= 0.99
            and mobility_spearman >= 0.99
        )
        output.append(
            {
                "system": system,
                "diagnostic": "warm_start_ablation",
                "n_rows": len(system_rows),
                "core_jaccard_min": fmt(core_jaccard),
                "core_jaccard_max": fmt(core_jaccard),
                "mobility_spearman_min": fmt(mobility_spearman),
                "mobility_spearman_max": fmt(mobility_spearman),
                "flexible_overlap_min": "",
                "flexible_overlap_max": "",
                "length_threshold_fraction_at_core_jaccard_ge_0_80": "",
                "warm_start_runtime_ratio_warm_over_cold": fmt(warm_runtime / cold_runtime if warm_runtime and cold_runtime else None),
                "warm_start_iteration_delta_warm_minus_cold": fmt(warm_iter - cold_iter if warm_iter is not None and cold_iter is not None else None),
                "warm_start_output_preserved": "yes" if preserved else "no",
                "nonconverged_rows": sum(str(row.get("converged", "")).lower() == "false" for row in system_rows),
                "classification_counts": "",
                "interpretation": "warm start is an efficiency option; outputs are preserved when Jaccard and mobility Spearman remain >= 0.99",
                "source_file": str(SOURCE_DIR / "warm_start_ablation" / "warm_start_ablation.tsv"),
            }
        )
    return output


def build_fig3_rows() -> list[dict[str, object]]:
    """Build compact MD diagnostic summaries for main figure 3."""

    temporal_path = SOURCE_DIR / "temporal_block_stability" / "temporal_block_stability.tsv"
    stride_path = SOURCE_DIR / "stride_sensitivity" / "stride_sensitivity.tsv"
    length_path = SOURCE_DIR / "trajectory_length_dependence" / "trajectory_length_dependence.tsv"
    reference_path = SOURCE_DIR / "reference_sensitivity" / "reference_sensitivity.tsv"
    warm_path = SOURCE_DIR / "warm_start_ablation" / "warm_start_ablation.tsv"
    temporal = read_table(temporal_path)
    stride = read_table(stride_path)
    length = read_table(length_path)
    reference = read_table(reference_path)
    warm = read_table(warm_path)
    systems = sorted({row["system"] for row in temporal})
    classifications = _classification_counts()
    output: list[dict[str, object]] = []
    for system in systems:
        temporal_rows = [row for row in _system_rows(temporal, system) if row.get("block_id") != "full"]
        output.append(
            _diagnostic_summary_row(
                system=system,
                diagnostic="temporal_block_stability",
                rows=temporal_rows,
                core_field="jaccard_core_vs_full",
                mobility_field="spearman_mobility_vs_full",
                flexible_field="flexible_segment_overlap_vs_full",
                source_file=temporal_path,
                interpretation="non-full blocks compared with full-trajectory soft core assignment",
                classification=classifications.get(system, ""),
            )
        )
        output.append(
            _diagnostic_summary_row(
                system=system,
                diagnostic="stride_sensitivity",
                rows=_system_rows(stride, system),
                core_field="jaccard_core_vs_stride1",
                mobility_field="spearman_mobility_vs_stride1",
                flexible_field="flexible_segment_overlap_vs_stride1",
                source_file=stride_path,
                interpretation="subsampled trajectories compared with stride-1 assignment",
            )
        )
        length_rows = _system_rows(length, system)
        length_row = _diagnostic_summary_row(
            system=system,
            diagnostic="trajectory_length_dependence",
            rows=length_rows,
            core_field="jaccard_core_vs_full",
            mobility_field="spearman_mobility_vs_full",
            flexible_field="flexible_segment_overlap_vs_full",
            source_file=length_path,
            interpretation="smallest prefix fraction reaching core Jaccard >= 0.80 is reported as the length threshold",
        )
        length_row["length_threshold_fraction_at_core_jaccard_ge_0_80"] = _length_threshold(length_rows)
        output.append(length_row)
        output.append(
            _diagnostic_summary_row(
                system=system,
                diagnostic="reference_sensitivity",
                rows=[row for row in _system_rows(reference, system) if row.get("reference_strategy") != "first_frame"],
                core_field="jaccard_core_vs_default_reference",
                mobility_field="spearman_mobility_vs_default_reference",
                flexible_field="flexible_segment_overlap_vs_default_reference",
                source_file=reference_path,
                interpretation="reference sensitivity is explicitly reported; no reference-independence claim is made",
            )
        )
    output.extend(_warm_start_rows(warm))
    return output


def build_fig4_rows() -> list[dict[str, object]]:
    """Build the flexibility-profile summary table for main figure 4."""

    stats_by_system = {row["system"]: row for row in read_table(FLEX_STATS)}
    rows: list[dict[str, object]] = []
    for row in read_table(FLEX_SUMMARY):
        system = row["system"]
        stats = stats_by_system.get(system, {})
        rows.append(
            {
                "system": system,
                "n_residues": row.get("n_residues", ""),
                "n_flexible_segments": row.get("n_flexible_segments", ""),
                "spearman_robust_rmsf_vs_mobility": row.get("spearman_RMSF_vs_mobility", ""),
                "spearman_ordinary_rmsf_vs_mobility": row.get("spearman_ordinary_RMSF_vs_mobility", ""),
                "mean_robust_rmsf": row.get("mean_robust_RMSF", ""),
                "mean_ordinary_rmsf": row.get("mean_ordinary_RMSF", ""),
                "mean_robust_rmsf_core": stats.get("mean_robust_RMSF_core", row.get("mean_RMSF_core", "")),
                "mean_robust_rmsf_noncore": stats.get("mean_robust_RMSF_noncore", row.get("mean_RMSF_noncore", "")),
                "delta_mean_noncore_minus_core": stats.get("delta_mean_noncore_minus_core", row.get("delta_core_noncore_RMSF", "")),
                "top_flexible_residue": row.get("top_flexible_residue", ""),
                "top_flexible_residue_rmsf": row.get("top_flexible_residue_RMSF", ""),
                "top_flexible_segment": row.get("top_flexible_segment", ""),
                "notes": "RMSF is alignment-frame dependent; YiiP external biology proxy remains limited.",
                "source_file": str(FLEX_SUMMARY),
            }
        )
    return rows


def write_notes_and_captions() -> None:
    """Write main-figure notes and caption draft."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fig1_workflow_notes.md").write_text(
        "\n".join(
            [
                "# Figure 1 Workflow Notes",
                "",
                f"- Generated at: `{timestamp()}`",
                "- The workflow figure should summarize input MD/NMR conformational ensembles, ordinary rigid fitting, soft-core robust fitting, stable-core weight estimation, RMSF/flexible-segment profiling, and visualization/export.",
                "- Figure 1 is conceptual and does not require new numerical benchmarking.",
                "- Runtime claims should remain restricted to alignment/core-identification algorithm runtime, excluding export or rendering overhead.",
                "- RMSF is alignment-frame dependent and should be interpreted as a downstream profile conditioned on the chosen fitting frame.",
                "",
            ]
        )
    )
    (OUT_DIR / "main_figure_captions_draft.md").write_text(
        "\n".join(
            [
                "# Main Figure Captions Draft",
                "",
                f"- Generated at: `{timestamp()}`",
                "",
                "## Figure 1. MDCoreAlign workflow",
                "Soft-membership robust alignment estimates stable-core weights while fitting each conformation to a reference frame. The resulting weights support stable/mobile residue profiling, RMSF comparison, flexible-segment detection, and visualization export. RMSF is explicitly alignment-frame dependent.",
                "",
                "## Figure 2. Formal MD trajectory benchmark summary",
                "Algorithm-runtime and unified post-hoc RMSD are compared for MDCoreAlign soft robust alignment, external MDLovoFit, and THESEUS across I-FABP, ADK, NhaA, and YiiP short. MDLovoFit-native all-atom and least-mobile RMSD fields are reported separately when available.",
                "",
                "## Figure 3. MD trajectory diagnostics",
                "Temporal block, stride, trajectory-length, reference-sensitivity, and warm-start diagnostics are summarized by system. Reference sensitivity is reported directly; the method is not claimed to be reference independent. Warm start is treated as an optional efficiency feature rather than a scientific requirement.",
                "",
                "## Figure 4. Flexibility profile summary",
                "Per-system RMSF and soft mobility summaries quantify how robust-core alignment changes local flexibility interpretation. Core and non-core RMSF are compared using robust-aligned RMSF; YiiP remains a challenging membrane-protein case with limited external biological-proxy support.",
                "",
            ]
        )
    )


def main() -> None:
    """Build all main-figure data outputs."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_table(OUT_DIR / "fig2_benchmark_summary.tsv", build_fig2_rows(), FIG2_FIELDS)
    write_table(OUT_DIR / "fig3_md_diagnostics_summary.tsv", build_fig3_rows(), FIG3_FIELDS)
    write_table(OUT_DIR / "fig4_flexibility_profile_summary.tsv", build_fig4_rows(), FIG4_FIELDS)
    write_notes_and_captions()
    print(f"Wrote MD trajectory main-figure data to {OUT_DIR}")


if __name__ == "__main__":
    main()
