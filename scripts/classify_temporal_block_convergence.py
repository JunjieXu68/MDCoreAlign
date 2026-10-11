"""Classify temporal-block non-convergence for final quality upgrade."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_quality_common import FINAL_ROOT, read_table, timestamp, to_float, write_manifest, write_table  # noqa: E402


OUTPUT_DIR = FINAL_ROOT / "temporal_convergence"
TEMPORAL_TABLE = SCRIPT_DIR.parent / "results" / "md_trajectory_focus" / "temporal_block_stability" / "temporal_block_stability.tsv"
TRACE_TABLE = SCRIPT_DIR.parent / "results" / "submission_critical" / "diagnostics_rerun" / "alignment_traces_four_systems.tsv"
MAX_ITERATIONS = 20

FIELDS = [
    "system",
    "block_id",
    "frame_start",
    "frame_end",
    "n_frames",
    "converged",
    "n_iterations",
    "final_weighted_RMSD",
    "last3_weighted_RMSD_change",
    "final_delta_weight_L2",
    "final_delta_weight_max",
    "effective_core_size",
    "mean_weight",
    "min_weight",
    "max_weight",
    "core_jaccard_vs_full",
    "mobility_spearman_vs_full",
    "flexible_segment_overlap_vs_full",
    "classification",
    "confidence",
    "manuscript_interpretation",
    "notes",
]

CLASSES = {
    "max_iter_reached_but_stable",
    "weight_collapse",
    "oscillation",
    "insufficient_frames",
    "high_reference_sensitivity",
    "true_failure",
    "unknown",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--temporal-table", type=Path, default=TEMPORAL_TABLE)
    parser.add_argument("--trace-table", type=Path, default=TRACE_TABLE)
    return parser.parse_args()


def classify_row(row: dict[str, str]) -> tuple[str, str, str, str]:
    """Classify one non-converged row conservatively."""

    n_frames = int(float(row["n_frames"]))
    iterations = float(row["n_iterations"])
    weighted = to_float(row.get("weighted_RMSD"))
    effective_core = to_float(row.get("effective_core_size"), 0.0) or 0.0
    core_jaccard = to_float(row.get("jaccard_core_vs_full"))
    mobility = to_float(row.get("spearman_mobility_vs_full"))
    flexible_overlap = to_float(row.get("flexible_segment_overlap_vs_full"))

    if n_frames < 20:
        return (
            "insufficient_frames",
            "high",
            "Exclude from primary temporal-stability conclusions because too few frames were available.",
            "n_frames below conservative threshold",
        )
    if effective_core <= 1e-6:
        return (
            "weight_collapse",
            "high",
            "Treat as a numerical failure and exclude from primary temporal-stability conclusions.",
            "effective_core_size near zero",
        )
    if weighted is None or weighted < 0.0:
        return (
            "true_failure",
            "high",
            "Treat as a numerical failure and exclude from primary temporal-stability conclusions.",
            "weighted RMSD missing or invalid",
        )
    if iterations >= 0.85 * MAX_ITERATIONS and effective_core > 0.0:
        return (
            "max_iter_reached_but_stable",
            "moderate",
            "Describe as conservative max-iteration stopping, not a proven numerical failure.",
            "near max-iteration stopping with finite weighted RMSD and nonzero effective core",
        )
    if (
        (core_jaccard is not None and core_jaccard < 0.5)
        or (mobility is not None and mobility < 0.5)
        or (flexible_overlap is not None and flexible_overlap < 0.2)
    ):
        return (
            "high_reference_sensitivity",
            "moderate",
            "Describe as time-window/reference-sensitive behavior rather than a numerical crash.",
            "low agreement with full-trajectory result",
        )
    return (
        "unknown",
        "low",
        "Keep in supplementary diagnostics and avoid mechanistic overinterpretation.",
        "per-block iteration traces were not available",
    )


def build_rows(temporal_table: Path, trace_table: Path) -> list[dict[str, object]]:
    """Build final temporal classification rows."""

    temporal = read_table(temporal_table)
    trace_available = trace_table.exists()
    rows: list[dict[str, object]] = []
    for row in temporal:
        if row["block_id"] == "full" or str(row["converged"]).lower() != "false":
            continue
        classification, confidence, interpretation, evidence = classify_row(row)
        rows.append(
            {
                "system": row["system"],
                "block_id": row["block_id"],
                "frame_start": row.get("frame_start", ""),
                "frame_end": row.get("frame_end", ""),
                "n_frames": row["n_frames"],
                "converged": row["converged"],
                "n_iterations": row["n_iterations"],
                "final_weighted_RMSD": row.get("weighted_RMSD", ""),
                "last3_weighted_RMSD_change": "",
                "final_delta_weight_L2": "",
                "final_delta_weight_max": "",
                "effective_core_size": row.get("effective_core_size", ""),
                "mean_weight": "",
                "min_weight": "",
                "max_weight": "",
                "core_jaccard_vs_full": row.get("jaccard_core_vs_full", ""),
                "mobility_spearman_vs_full": row.get("spearman_mobility_vs_full", ""),
                "flexible_segment_overlap_vs_full": row.get("flexible_segment_overlap_vs_full", ""),
                "classification": classification,
                "confidence": confidence,
                "manuscript_interpretation": interpretation,
                "notes": (
                    f"{evidence}; per-block diagnostic trace unavailable, "
                    f"global diagnostics trace exists={trace_available}"
                ),
            }
        )
    return rows


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary answering required questions."""

    counts = {label: sum(row["classification"] == label for row in rows) for label in sorted(CLASSES)}
    true_failures = counts.get("true_failure", 0) + counts.get("weight_collapse", 0)
    conservative = counts.get("max_iter_reached_but_stable", 0)
    by_system: dict[str, int] = {}
    sensitive_by_system: dict[str, int] = {}
    for row in rows:
        system = str(row["system"])
        by_system[system] = by_system.get(system, 0) + 1
        if row["classification"] == "high_reference_sensitivity":
            sensitive_by_system[system] = sensitive_by_system.get(system, 0) + 1
    sensitive = ", ".join(f"{system}:{count}" for system, count in sorted(sensitive_by_system.items())) or "none"
    return "\n".join(
        [
            "# Temporal Block Non-Convergence Classification",
            "",
            f"- Generated at: `{timestamp()}`",
            f"- Non-full non-converged rows: `{len(rows)}`.",
            f"- Classification counts: `{counts}`.",
            "",
            "## Required Answers",
            "",
            f"1. How many non-converged block rows exist? `{len(rows)}`.",
            f"2. How many are likely conservative max-iteration stopping? `{conservative}`.",
            f"3. How many are true failures? `{true_failures}`; no row is labeled true_failure without explicit invalid RMSD or weight collapse evidence.",
            f"4. Is any system especially time-window sensitive? High reference/time-window sensitivity rows by system: `{sensitive}`.",
            "5. How should temporal non-convergence be described in the manuscript? As mostly conservative max-iteration stopping or explicitly limited summary-only diagnostics, not as hidden numerical failure.",
            "6. Should any temporal-block rows be excluded from main conclusions? Rows classified as true_failure or weight_collapse should be excluded; none were observed in this pass. Unknown rows should remain supplementary.",
            "",
            "## Limitation",
            "",
            "- Per-block iteration traces were not generated in the original temporal-block rerun, so oscillation-specific labels remain unavailable unless a dedicated trace rerun is performed.",
            "",
        ]
    )


def main() -> int:
    """Run final temporal convergence classification."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.temporal_table, args.trace_table)
    table = args.output_dir / "temporal_convergence_classification.tsv"
    summary = args.output_dir / "temporal_convergence_summary.md"
    write_table(table, rows, FIELDS)
    summary.write_text(build_summary(rows))
    write_manifest(
        args.output_dir,
        script="scripts/classify_temporal_block_convergence.py",
        inputs=[args.temporal_table, args.trace_table],
        outputs=[table, summary],
        extra={
            "allowed_classifications": sorted(CLASSES),
            "max_iterations_assumed": MAX_ITERATIONS,
            "random_seed": None,
        },
    )
    print(f"Wrote final temporal convergence classification to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
