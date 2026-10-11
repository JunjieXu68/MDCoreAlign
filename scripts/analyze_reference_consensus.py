"""Analyze reference-consensus and reference-sensitive behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_quality_common import FINAL_ROOT, read_table, timestamp, to_float, write_manifest, write_table  # noqa: E402


OUTPUT_DIR = FINAL_ROOT / "reference_consensus"
REFERENCE_TABLE = SCRIPT_DIR.parent / "results" / "md_trajectory_focus" / "reference_sensitivity" / "reference_sensitivity.tsv"
FLEX_SUMMARY = SCRIPT_DIR.parent / "results" / "flexibility" / "flexibility_summary_four_systems.tsv"

RESIDUE_FIELDS = [
    "system",
    "chain",
    "residue_id",
    "resname",
    "n_reference_strategies",
    "core_frequency",
    "consensus_core_label",
    "consensus_mobile_label",
    "mean_mobility_score",
    "sd_mobility_score",
    "min_mobility_score",
    "max_mobility_score",
    "reference_sensitive",
    "reference_sensitive_reason",
    "default_reference_core_label",
    "default_reference_mobility_score",
]

SYSTEM_FIELDS = [
    "system",
    "n_residues",
    "n_consensus_core",
    "n_consensus_mobile",
    "n_reference_sensitive_residues",
    "fraction_reference_sensitive",
    "mean_core_frequency",
    "median_core_frequency",
    "mean_mobility_sd",
    "reference_strategy_count",
    "lowest_pairwise_core_jaccard",
    "highest_pairwise_core_jaccard",
    "lowest_pairwise_mobility_spearman",
    "highest_pairwise_mobility_spearman",
]

REFERENCE_STRATEGIES = [
    "first_frame",
    "middle_frame",
    "last_frame",
    "ordinary_mean_structure",
    "robust_mean_structure",
]

FLEX_SYSTEM_ALIASES = {
    "ADK": "ADK equilibrium",
    "I-FABP": "I-FABP",
    "NhaA": "NhaA equilibrium",
    "YiiP short": "YiiP equilibrium short",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-table", type=Path, default=REFERENCE_TABLE)
    parser.add_argument("--flex-summary", type=Path, default=FLEX_SUMMARY)
    return parser.parse_args()


def _range(values: list[float | None]) -> tuple[object, object]:
    """Return min/max for finite values."""

    items = [float(value) for value in values if value is not None]
    if not items:
        return "", ""
    return min(items), max(items)


def build_system_rows(reference_table: Path, flex_summary: Path) -> list[dict[str, object]]:
    """Build system-level reference-consensus summaries from available tables."""

    reference_rows = read_table(reference_table)
    flex_rows = {row["system"]: row for row in read_table(flex_summary)} if flex_summary.exists() else {}
    systems = sorted({row["system"] for row in reference_rows})
    output: list[dict[str, object]] = []
    for system in systems:
        rows = [row for row in reference_rows if row["system"] == system]
        jaccard_min, jaccard_max = _range([to_float(row.get("jaccard_core_vs_default_reference")) for row in rows])
        mobility_min, mobility_max = _range([to_float(row.get("spearman_mobility_vs_default_reference")) for row in rows])
        flex_key = FLEX_SYSTEM_ALIASES.get(system, system)
        n_residues = flex_rows.get(flex_key, {}).get("n_residues", "")
        output.append(
            {
                "system": system,
                "n_residues": n_residues,
                "n_consensus_core": "",
                "n_consensus_mobile": "",
                "n_reference_sensitive_residues": "",
                "fraction_reference_sensitive": "",
                "mean_core_frequency": "",
                "median_core_frequency": "",
                "mean_mobility_sd": "",
                "reference_strategy_count": len({row["reference_strategy"] for row in rows}),
                "lowest_pairwise_core_jaccard": jaccard_min,
                "highest_pairwise_core_jaccard": jaccard_max,
                "lowest_pairwise_mobility_spearman": mobility_min,
                "highest_pairwise_mobility_spearman": mobility_max,
            }
        )
    return output


def build_summary(system_rows: list[dict[str, object]]) -> str:
    """Build Markdown summary."""

    lowest = sorted(
        (
            (to_float(row["lowest_pairwise_core_jaccard"], 1.0) or 1.0, row["system"])
            for row in system_rows
        ),
        key=lambda item: item[0],
    )
    most_sensitive = lowest[0][1] if lowest else "unavailable"
    return "\n".join(
        [
            "# Reference-Consensus and Reference-Sensitivity Summary",
            "",
            f"- Generated at: `{timestamp()}`",
            "- Residue-level per-reference mobility profiles were not available in the existing formal outputs; residue-level consensus was therefore not inferred.",
            "- The system-level table summarizes available reference-sensitivity rows relative to the first-frame default.",
            "",
            "## Required Answers",
            "",
            "1. Which residues are stable across reference choices? Unavailable from existing outputs because per-reference residue profiles were not generated.",
            f"2. Which systems are most reference-sensitive? `{most_sensitive}` has the lowest available core Jaccard relative to the first-frame default.",
            "3. Does reference sensitivity affect all residues or a limited subset? This cannot be resolved without residue-level per-reference profiles.",
            "4. Should the manuscript recommend reporting reference-consensus cores? Yes, but as a recommended diagnostic/future output unless residue-level consensus is generated in a dedicated formal rerun.",
            "5. How should reference choice be described in Methods and Limitations? State that stable-core interpretation is reference-aware and that no reference-independence claim is made.",
            "",
            "## Definitions Reserved For Future Residue-Level Consensus",
            "",
            "- consensus_core if core frequency >= 0.8.",
            "- consensus_mobile if core frequency <= 0.2.",
            "- reference_sensitive if mobility SD exceeds the system-specific 75th percentile or core label changes across references.",
            "",
        ]
    )


def main() -> int:
    """Run reference-consensus analysis."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    residue_table = args.output_dir / "reference_consensus_residues.tsv"
    system_table = args.output_dir / "reference_consensus_system_summary.tsv"
    summary = args.output_dir / "reference_consensus_summary.md"
    residue_rows: list[dict[str, object]] = []
    system_rows = build_system_rows(args.reference_table, args.flex_summary)
    write_table(residue_table, residue_rows, RESIDUE_FIELDS)
    write_table(system_table, system_rows, SYSTEM_FIELDS)
    summary.write_text(build_summary(system_rows))
    write_manifest(
        args.output_dir,
        script="scripts/analyze_reference_consensus.py",
        inputs=[args.reference_table, args.flex_summary],
        outputs=[residue_table, system_table, summary],
        extra={
            "residue_level_per_reference_profiles_available": False,
            "consensus_core_threshold": 0.8,
            "consensus_mobile_threshold": 0.2,
            "reference_sensitive_mobility_sd_percentile": 75,
            "reference_strategies_expected": REFERENCE_STRATEGIES,
            "random_seed": None,
        },
    )
    print(f"Wrote reference consensus outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
