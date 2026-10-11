"""Assemble a formal regularization-ablation table from existing outputs.

This script does not rerun or retune the core algorithm. It records available
evidence and explicitly marks missing ablation cells that still require a true
formal rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_ready_common import ROOT, fmt, read_table, to_float, to_int, write_table  # noqa: E402


FIELDS = [
    "system",
    "method",
    "is_default_soft_sequence_contact_result",
    "n_residues",
    "n_frames_or_models",
    "runtime_seconds",
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "effective_core_size",
    "number_of_core_residues",
    "number_of_mobile_residues",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "Spearman_robust_RMSF_vs_mobility",
    "number_of_flexible_segments",
    "core_fragmentation_score",
    "number_of_core_segments",
    "convergence_status",
    "input_status",
    "metric_definition",
    "notes",
]

METHODS = [
    "ordinary_alignment",
    "hard_trim_control",
    "soft_no_sequence_no_contact",
    "soft_sequence_only",
    "soft_contact_only",
    "soft_sequence_contact",
]

SYSTEM_MAP = {
    "I-FABP": "I-FABP",
    "ADK equilibrium": "ADK equilibrium",
    "NhaA equilibrium": "NhaA equilibrium",
    "YiiP equilibrium short": "YiiP equilibrium short",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ablation" / "regularization")
    return parser.parse_args()


def metadata_by_system() -> dict[str, dict[str, object]]:
    """Load formal flexibility metadata by display label."""

    summary = json.loads((ROOT / "results" / "flexibility" / "summary.json").read_text())
    return {item["label"]: item for item in summary["datasets"]}


def trajectory_rows_by_system_method() -> dict[tuple[str, str], dict[str, str]]:
    """Map existing formal trajectory rows into display system/method keys."""

    rows = read_table(ROOT / "results" / "final_multi_system_summary_20260603" / "final_real_trajectory_summary.csv")
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        system = row["system"]
        mapping[(system, "soft_no_sequence_no_contact")] = {
            "runtime_seconds": row["soft_runtime_s"],
            "unified_posthoc_RMSD": row["soft_unified_rmsd"],
            "weighted_RMSD": row["soft_weighted_rmsd"],
            "effective_core_size": "",
            "number_of_core_residues": "",
            "convergence_status": "available_existing_formal_summary",
        }
        mapping[(system, "ordinary_alignment")] = {
            "runtime_seconds": "",
            "unified_posthoc_RMSD": "",
            "weighted_RMSD": "",
            "effective_core_size": "",
            "number_of_core_residues": "",
            "convergence_status": "ordinary_reference_available_in_raw_tables_not_final_summary",
        }
    return mapping


def hard_trim_by_system() -> dict[str, dict[str, str]]:
    """Return hard-trim control rows where available."""

    rows = read_table(ROOT / "results" / "final_multi_system_summary_20260603" / "final_baseline_breadth_hardtrim_summary.csv")
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        output[row["system"]] = {
            "runtime_seconds": row["hard_trim_runtime_s"],
            "unified_posthoc_RMSD": row["hard_trim_unified_rmsd"],
            "weighted_RMSD": "",
            "convergence_status": "available_existing_hard_trim_control",
        }
    return output


def flexibility_by_system() -> dict[str, dict[str, str]]:
    """Load RMSF and flexible-segment summary values."""

    return {row["system"]: row for row in read_table(ROOT / "results" / "flexibility" / "flexibility_summary_four_systems.tsv")}


def biology_regularized_available() -> set[str]:
    """Return systems with sequence+contact biological-proxy rows."""

    rows = read_table(ROOT / "results" / "higher_tier_external_biology" / "external_biology_combined_summary.csv")
    names = {
        "ifabp": "I-FABP",
        "adk_equilibrium": "ADK equilibrium",
        "nhaa_equilibrium": "NhaA equilibrium",
        "yiip_equilibrium_short": "YiiP equilibrium short",
    }
    return {names[row["dataset_name"]] for row in rows if row["method_name"] == "soft_weight_seq_contact_regularized"}


def build_rows() -> list[dict[str, object]]:
    """Build the formal ablation table from available evidence."""

    metadata = metadata_by_system()
    trajectory_map = trajectory_rows_by_system_method()
    hard_trim = hard_trim_by_system()
    flexibility = flexibility_by_system()
    seq_contact_biology = biology_regularized_available()
    rows: list[dict[str, object]] = []
    for system in SYSTEM_MAP:
        meta = metadata.get(system, {})
        flex = flexibility.get(system, {})
        n_res = meta.get("n_residues", flex.get("n_residues", ""))
        n_frames = meta.get("n_frames", "")
        for method in METHODS:
            base: dict[str, object] = {
                "system": system,
                "method": method,
                "is_default_soft_sequence_contact_result": "yes" if method == "soft_sequence_contact" else "no",
                "n_residues": n_res,
                "n_frames_or_models": n_frames,
                "runtime_seconds": "",
                "unified_posthoc_RMSD": "",
                "weighted_RMSD": "",
                "effective_core_size": "",
                "number_of_core_residues": "",
                "number_of_mobile_residues": "",
                "mean_core_RMSF": "",
                "mean_noncore_RMSF": "",
                "noncore_core_RMSF_ratio": "",
                "Spearman_robust_RMSF_vs_mobility": "",
                "number_of_flexible_segments": "",
                "core_fragmentation_score": "",
                "number_of_core_segments": "",
                "convergence_status": "",
                "input_status": "not_available_existing_results",
                "metric_definition": "Rows are assembled from existing formal outputs; missing cells require a dedicated formal rerun.",
                "notes": "",
            }
            if method == "hard_trim_control" and system in hard_trim:
                base.update(hard_trim[system])
                base["input_status"] = "available_existing_summary"
            elif (system, method) in trajectory_map:
                base.update(trajectory_map[(system, method)])
                base["input_status"] = "available_existing_summary"
            elif method == "soft_sequence_contact" and system in seq_contact_biology:
                base["input_status"] = "available_biology_proxy_only_geometry_missing"
                base["convergence_status"] = "geometry_not_available_in_existing_outputs"
            if method == "soft_no_sequence_no_contact":
                mean_core = to_float(flex.get("mean_RMSF_core"))
                mean_noncore = to_float(flex.get("mean_RMSF_noncore"))
                base.update(
                    {
                        "mean_core_RMSF": fmt(mean_core),
                        "mean_noncore_RMSF": fmt(mean_noncore),
                        "noncore_core_RMSF_ratio": fmt(mean_noncore / mean_core if mean_core else None),
                        "Spearman_robust_RMSF_vs_mobility": fmt(to_float(flex.get("spearman_RMSF_vs_mobility"))),
                        "number_of_flexible_segments": flex.get("n_flexible_segments", ""),
                    }
                )
                core_size = to_int(base.get("number_of_core_residues")) or to_int(base.get("effective_core_size"))
                if not core_size:
                    fraction = to_float(flex.get("core_fraction"), 0.6) or 0.6
                    core_size = int(round(float(n_res) * fraction)) if n_res else None
                if core_size is not None:
                    base["number_of_core_residues"] = core_size
                    base["effective_core_size"] = core_size
                    base["number_of_mobile_residues"] = int(n_res) - core_size if n_res else ""
            if method == "soft_sequence_only":
                base["notes"] = "Formal sequence-only trajectory output is not present yet."
            elif method == "soft_contact_only":
                base["notes"] = "Formal contact-only trajectory output is not present yet."
            elif method == "soft_sequence_contact":
                base["notes"] = "Biological proxy rows exist, but matching geometry/RMSF formal rerun is still needed."
            elif method == "hard_trim_control" and system not in hard_trim:
                base["notes"] = "Hard-trim control missing for this system in existing final summaries."
            rows.append(base)
    return rows


def summary_markdown(rows: list[dict[str, object]]) -> str:
    """Build a conservative ablation summary."""

    available = [row for row in rows if str(row["input_status"]).startswith("available_existing")]
    missing = [row for row in rows if row["input_status"] == "not_available_existing_results"]
    hard_rows = [row for row in available if row["method"] == "hard_trim_control"]
    soft_rows = {row["system"]: row for row in available if row["method"] == "soft_no_sequence_no_contact"}
    soft_better_hard = 0
    for hard in hard_rows:
        soft = soft_rows.get(hard["system"])
        if soft and to_float(soft["unified_posthoc_RMSD"]) is not None and to_float(hard["unified_posthoc_RMSD"]) is not None:
            soft_better_hard += int(to_float(soft["unified_posthoc_RMSD"]) < to_float(hard["unified_posthoc_RMSD"]))
    lines = [
        "# Regularization Ablation Summary",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        "- This table is an assembly of existing formal outputs, not a new hyperparameter-tuned rerun.",
        f"- Available existing rows: `{len(available)}`.",
        f"- Missing formal ablation cells: `{len(missing)}`.",
        "",
        "## Questions",
        "",
        "- Does sequence regularization improve core continuity? Current existing formal outputs are insufficient; sequence-only formal rerun is required.",
        "- Does contact regularization improve biological or structural plausibility? Existing biological-proxy rows support testing this, but geometry/RMSF-matched formal reruns are still needed.",
        "- Does sequence+contact preserve or improve RMSD? Not answerable from existing four-system formal geometry tables.",
        "- Is regularization mainly helping geometry, interpretability, or both? Current evidence mainly supports interpretability/biology proxies; geometry needs formal rerun.",
        "- Are there systems where regularization hurts a metric? YiiP external biology proxy remains weak/negative and should be treated cautiously.",
        f"- Does soft membership outperform hard trimming in geometric quality? In available hard-trim systems, soft has lower unified RMSD in `{soft_better_hard}/{len(hard_rows)}` cases.",
        "",
        "## Required Follow-up",
        "",
        "- Run a true formal 4-system regularization ablation with fixed defaults for sequence-only, contact-only, and sequence+contact settings.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write formal regularization-ablation analysis outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_table(args.output_dir / "regularization_ablation_four_systems.tsv", rows, FIELDS)
    plot_fields = [
        "system",
        "method",
        "unified_posthoc_RMSD",
        "runtime_seconds",
        "mean_core_RMSF",
        "mean_noncore_RMSF",
        "Spearman_robust_RMSF_vs_mobility",
        "input_status",
    ]
    write_table(args.output_dir / "regularization_ablation_plot_data.tsv", rows, plot_fields)
    (args.output_dir / "regularization_ablation_summary.md").write_text(summary_markdown(rows))
    print(f"Wrote regularization ablation analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
