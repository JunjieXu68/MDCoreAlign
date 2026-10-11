"""Rerun formal regularization ablations on the four promoted trajectory systems.

This script intentionally writes into results/submission_critical/ and does
not overwrite the existing official benchmark tables.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_ready_common import fmt, sha256_file, write_json  # noqa: E402
from submission_critical_common import (  # noqa: E402
    DEFAULT_CONTACT_LAMBDA,
    DEFAULT_CORE_THRESHOLD,
    DEFAULT_FLEXIBLE_PERCENTILE,
    DEFAULT_SEQUENCE_LAMBDA,
    FORMAL_SYSTEMS,
    ROOT,
    base_output_row,
    load_formal_system,
    run_hard_trim,
    run_ordinary,
    run_soft,
    write_table,
)


OUTPUT_DIR = ROOT / "results" / "submission_critical" / "regularization_rerun"

FIELDS = [
    "system",
    "method",
    "input_coordinate_file",
    "topology_file",
    "selection",
    "reference_strategy",
    "n_residues",
    "n_frames_or_models",
    "lambda_sequence",
    "lambda_contact",
    "core_threshold",
    "flexible_percentile",
    "runtime_algorithm_seconds",
    "runtime_total_seconds",
    "unified_posthoc_RMSD",
    "weighted_RMSD",
    "mean_residual",
    "median_residual",
    "effective_core_size",
    "n_core_residues",
    "n_mobile_residues",
    "n_core_segments",
    "core_fragmentation_score",
    "mean_core_RMSF",
    "mean_noncore_RMSF",
    "noncore_core_RMSF_ratio",
    "spearman_robust_RMSF_vs_mobility",
    "n_flexible_segments",
    "converged",
    "n_iterations",
    "notes",
]

METHODS = [
    ("ordinary_alignment", 0.0, 0.0),
    ("hard_trim_control", 0.0, 0.0),
    ("soft_no_regularization", 0.0, 0.0),
    ("soft_sequence_only", DEFAULT_SEQUENCE_LAMBDA, 0.0),
    ("soft_contact_only", 0.0, DEFAULT_CONTACT_LAMBDA),
    ("soft_sequence_plus_contact_default", DEFAULT_SEQUENCE_LAMBDA, DEFAULT_CONTACT_LAMBDA),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--systems", nargs="+", choices=FORMAL_SYSTEMS, default=FORMAL_SYSTEMS)
    return parser.parse_args()


def _method_runner(method: str, lambda_sequence: float, lambda_contact: float) -> Callable:
    """Return the concrete runner for one method."""

    if method == "ordinary_alignment":
        return lambda system: run_ordinary(system)
    if method == "hard_trim_control":
        return lambda system: run_hard_trim(system, core_fraction=DEFAULT_CORE_THRESHOLD)
    return lambda system: run_soft(system, lambda_sequence=lambda_sequence, lambda_contact=lambda_contact)


def _row(system, method: str, lambda_sequence: float, lambda_contact: float, metrics: dict[str, object], notes: str) -> dict[str, object]:
    """Build one output row."""

    row = base_output_row(system, method, lambda_sequence, lambda_contact)
    row.update(metrics)
    row["notes"] = notes
    return row


def run_rerun(systems: list[str], output_dir: Path) -> list[dict[str, object]]:
    """Execute all formal reruns and write per-run raw logs."""

    raw_dir = output_dir / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for system_key in systems:
        system = load_formal_system(system_key)
        for method, lambda_sequence, lambda_contact in METHODS:
            start = time.perf_counter()
            notes = "formal_rerun_completed"
            try:
                runner = _method_runner(method, lambda_sequence, lambda_contact)
                metrics, *_ = runner(system)
            except Exception as exc:
                metrics = {
                    "runtime_algorithm_seconds": "",
                    "runtime_total_seconds": time.perf_counter() - start,
                    "unified_posthoc_RMSD": "",
                    "weighted_RMSD": "",
                    "mean_residual": "",
                    "median_residual": "",
                    "effective_core_size": "",
                    "n_core_residues": "",
                    "n_mobile_residues": "",
                    "n_core_segments": "",
                    "core_fragmentation_score": "",
                    "mean_core_RMSF": "",
                    "mean_noncore_RMSF": "",
                    "noncore_core_RMSF_ratio": "",
                    "spearman_robust_RMSF_vs_mobility": "",
                    "n_flexible_segments": "",
                    "converged": "unavailable",
                    "n_iterations": "",
                }
                notes = f"unavailable: {type(exc).__name__}: {exc}"
            row = _row(system, method, lambda_sequence, lambda_contact, metrics, notes)
            rows.append(row)
            write_json(
                raw_dir / f"{system_key}_{method}.json",
                {
                    "system_key": system_key,
                    "method": method,
                    "lambda_sequence": lambda_sequence,
                    "lambda_contact": lambda_contact,
                    "core_threshold": DEFAULT_CORE_THRESHOLD,
                    "flexible_percentile": DEFAULT_FLEXIBLE_PERCENTILE,
                    "row": row,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"[regularization] {system.label} {method}: {notes}")
    return rows


def _numeric(row: dict[str, object], field: str) -> float | None:
    """Convert a row value to float."""

    try:
        text = str(row.get(field, "")).strip()
        return float(text) if text else None
    except ValueError:
        return None


def _by_system_method(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    """Index rows by system and method."""

    return {(str(row["system"]), str(row["method"])): row for row in rows}


def build_summary(rows: list[dict[str, object]]) -> str:
    """Build the regularization rerun Markdown summary."""

    indexed = _by_system_method(rows)
    systems = list(dict.fromkeys(str(row["system"]) for row in rows))
    seq_better_continuity = 0
    contact_better_continuity = 0
    default_geometry_not_worse = 0
    hurt_notes: list[str] = []
    for system in systems:
        no_reg = indexed.get((system, "soft_no_regularization"), {})
        seq = indexed.get((system, "soft_sequence_only"), {})
        contact = indexed.get((system, "soft_contact_only"), {})
        default = indexed.get((system, "soft_sequence_plus_contact_default"), {})
        no_seg = _numeric(no_reg, "n_core_segments")
        seq_seg = _numeric(seq, "n_core_segments")
        contact_seg = _numeric(contact, "n_core_segments")
        if no_seg is not None and seq_seg is not None and seq_seg <= no_seg:
            seq_better_continuity += 1
        if no_seg is not None and contact_seg is not None and contact_seg <= no_seg:
            contact_better_continuity += 1
        no_rmsd = _numeric(no_reg, "unified_posthoc_RMSD")
        default_rmsd = _numeric(default, "unified_posthoc_RMSD")
        if no_rmsd is not None and default_rmsd is not None:
            if default_rmsd <= no_rmsd * 1.01:
                default_geometry_not_worse += 1
            elif default_rmsd > no_rmsd:
                hurt_notes.append(f"{system}: default RMSD {fmt(default_rmsd)} vs no-reg {fmt(no_rmsd)}")
    lines = [
        "# Formal Regularization Rerun Summary",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        "- Scope: I-FABP, ADK, NhaA, and YiiP short using the formal trajectory inputs and fixed defaults.",
        "- This is a true rerun of alignment procedures, not a post-hoc relabeling of old outputs.",
        "",
        "## Required Questions",
        "",
        f"1. Does sequence regularization improve core continuity? It reduced or preserved core segment count in `{seq_better_continuity}/{len(systems)}` systems relative to no regularization.",
        f"2. Does contact regularization improve structural/biological plausibility? It reduced or preserved the core segment count in `{contact_better_continuity}/{len(systems)}` systems; this is structural-continuity evidence, not an external biological annotation claim.",
        f"3. Does sequence+contact preserve or improve geometry relative to no regularization? It preserved unified post-hoc RMSD within 1% in `{default_geometry_not_worse}/{len(systems)}` systems.",
        "4. Does regularization mainly affect RMSD, core continuity, RMSF interpretation, or runtime? Inspect `regularization_rerun_plot_data.tsv`; the primary intended effects are core continuity and RMSF/core interpretation, with RMSD monitored as a guardrail.",
        "5. Are there systems where regularization hurts a metric? " + ("; ".join(hurt_notes) if hurt_notes else "No >1% unified RMSD degradation was observed for the default setting."),
        "6. Is the default sequence+contact model still justified? Keep the default only if the table shows geometry is preserved while continuity/interpretable-core metrics are not worse; do not claim geometry improvement unless supported per system.",
        "",
        "## Files",
        "",
        "- `regularization_rerun_four_systems.tsv`: complete rerun table.",
        "- `regularization_rerun_plot_data.tsv`: plot-ready subset of the same rerun results.",
        "- `raw_logs/`: one JSON record per system-method rerun.",
    ]
    return "\n".join(lines) + "\n"


def plot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return compact rows for regularization plots."""

    fields = [
        "system",
        "method",
        "lambda_sequence",
        "lambda_contact",
        "unified_posthoc_RMSD",
        "weighted_RMSD",
        "runtime_algorithm_seconds",
        "n_core_segments",
        "core_fragmentation_score",
        "mean_core_RMSF",
        "mean_noncore_RMSF",
        "noncore_core_RMSF_ratio",
        "spearman_robust_RMSF_vs_mobility",
        "n_flexible_segments",
        "notes",
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def write_manifest(output_dir: Path, rows: list[dict[str, object]], systems: list[str]) -> None:
    """Write a manifest with checksums."""

    outputs = [
        output_dir / "regularization_rerun_four_systems.tsv",
        output_dir / "regularization_rerun_summary.md",
        output_dir / "regularization_rerun_plot_data.tsv",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "script": "scripts/rerun_regularization_formal_systems.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "systems": systems,
            "methods": [method for method, _, _ in METHODS],
            "default_parameters": {
                "core_threshold": DEFAULT_CORE_THRESHOLD,
                "flexible_percentile": DEFAULT_FLEXIBLE_PERCENTILE,
                "lambda_sequence_default": DEFAULT_SEQUENCE_LAMBDA,
                "lambda_contact_default": DEFAULT_CONTACT_LAMBDA,
            },
            "row_count": len(rows),
            "output_files": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in outputs
                if path.exists()
            },
        },
    )


def main() -> int:
    """Run formal regularization reruns."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_rerun(args.systems, args.output_dir)
    write_table(args.output_dir / "regularization_rerun_four_systems.tsv", rows, FIELDS)
    (args.output_dir / "regularization_rerun_summary.md").write_text(build_summary(rows))
    plot_data = plot_rows(rows)
    write_table(args.output_dir / "regularization_rerun_plot_data.tsv", plot_data, list(plot_data[0]))
    write_manifest(args.output_dir, rows, args.systems)
    print(f"Wrote formal regularization rerun outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
