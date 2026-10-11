"""Audit paper-ready figures and tables for completeness and stale wording."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_ready_common import ROOT, column_count, read_table, row_count, sha256_file, write_table  # noqa: E402


FIELDS = [
    "category",
    "path",
    "required",
    "exists",
    "size_bytes",
    "row_count",
    "column_count",
    "required_columns_present",
    "modified_time",
    "sha256",
    "warning_messages",
]

REQUIRED_FILES = [
    ("flexibility_figure", "results/flexibility/figures/fig_rmsf_profiles_four_systems.pdf", []),
    ("flexibility_figure", "results/flexibility/figures/fig_rmsf_profiles_four_systems.png", []),
    ("flexibility_figure", "results/flexibility/figures/fig_mobility_vs_robust_rmsf.pdf", []),
    ("flexibility_figure", "results/flexibility/figures/fig_core_vs_noncore_rmsf.pdf", []),
    ("flexibility_figure", "results/flexibility/figures/figure_captions_draft.md", []),
    ("flexibility_figure_data", "results/flexibility/figure_data/fig_rmsf_profiles.tsv", ["system", "alignment", "rmsf"]),
    ("flexibility_figure_data", "results/flexibility/figure_data/fig_mobility_vs_rmsf.tsv", ["system", "mobility_score", "robust_aligned_RMSF"]),
    ("flexibility_figure_data", "results/flexibility/figure_data/fig_core_noncore_rmsf.tsv", ["system", "core_group", "robust_aligned_RMSF"]),
    ("trajectory_benchmark_table", "results/final_statistics/trajectory_benchmark_statistics.tsv", ["system", "baseline_method", "runtime_ratio_soft_over_baseline"]),
    ("pair_benchmark_table", "results/final_statistics/pair_benchmark_statistics.tsv", ["system", "baseline_method", "runtime_ratio_soft_over_baseline"]),
    ("nmr500_benchmark_table", "results/final_statistics/nmr500_benchmark_statistics.tsv", ["comparison", "n_paired_cases", "bootstrap_seed"]),
    ("theseus_comparison_table", "results/higher_tier_external_baseline_theseus_formal/combined_theseus_formal_summary.csv", ["dataset_name", "method_name", "unified_posthoc_rmsd_mean"]),
    ("biological_support_table", "results/final_statistics/biology_support_statistics.tsv", ["system", "support_status", "metric_definition"]),
    ("hard_trim_control_table", "results/final_multi_system_summary_20260603/final_baseline_breadth_hardtrim_summary.csv", ["system", "hard_trim_unified_rmsd"]),
    ("regularization_ablation_table", "results/ablation/regularization/regularization_ablation_four_systems.tsv", ["system", "method", "input_status"]),
    ("parameter_sensitivity_table", "results/sensitivity/sensitivity_summary.tsv", ["system", "core_threshold", "flexible_percentile"]),
    ("diagnostics_table", "results/diagnostics/alignment_traces_four_systems.tsv", ["system", "iteration", "diagnostic_source"]),
]

MARKDOWN_TO_SCAN = [
    "README.md",
    "docs/current_main_results_summary.md",
    "docs/current_main_results_summary.txt",
    "docs/rmsf_flexible_segments.md",
    "results/flexibility/figures/figure_captions_draft.md",
    "results/final_statistics/final_statistics_summary.md",
    "results/ablation/regularization/regularization_ablation_summary.md",
    "results/diagnostics/diagnostics_summary.md",
    "results/sensitivity/default_parameter_justification.md",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "paper_ready" / "audit")
    return parser.parse_args()


def audit_rows() -> tuple[list[dict[str, object]], bool]:
    """Return audit rows and failure status."""

    rows: list[dict[str, object]] = []
    failed = False
    stale_warnings = stale_markdown_warnings()
    for category, relative, required_columns in REQUIRED_FILES:
        path = ROOT / relative
        warnings: list[str] = []
        exists = path.exists()
        if not exists:
            failed = True
            warnings.append("missing required file")
        size = path.stat().st_size if exists else 0
        if exists and size == 0:
            failed = True
            warnings.append("file is empty")
        present = required_columns_present(path, required_columns) if exists and required_columns else True
        if not present:
            failed = True
            warnings.append("required columns missing")
        if relative in stale_warnings:
            warnings.extend(stale_warnings[relative])
        rows.append(
            {
                "category": category,
                "path": relative,
                "required": "yes",
                "exists": exists,
                "size_bytes": size if exists else "",
                "row_count": row_count(path) if exists else "",
                "column_count": column_count(path) if exists else "",
                "required_columns_present": present,
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if exists else "",
                "sha256": sha256_file(path) if exists else "",
                "warning_messages": " | ".join(warnings),
            }
        )
    return rows, failed


def required_columns_present(path: Path, required_columns: list[str]) -> bool:
    """Check whether required columns are present in a CSV/TSV file."""

    if path.suffix.lower() not in {".csv", ".tsv"}:
        return True
    rows = read_table(path)
    if not rows:
        return False
    return set(required_columns).issubset(rows[0])


def stale_markdown_warnings() -> dict[str, list[str]]:
    """Return stale wording warnings by relative path."""

    warnings: dict[str, list[str]] = {}
    stale_tests = re.compile(r"\b(70|76|80)\s+(?:unit\s+)?tests?\b|`(70|76|80)`\s+(?:unit\s+)?tests?", re.I)
    obsolete_names = re.compile(r"\bMDStableCore\b")
    for relative in MARKDOWN_TO_SCAN:
        path = ROOT / relative
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        for pattern, label in ((stale_tests, "stale test count"), (obsolete_names, "obsolete method name")):
            if pattern.search(text):
                warnings.setdefault(relative, []).append(label)
    return warnings


def markdown_summary(rows: list[dict[str, object]], failed: bool) -> str:
    """Build human-readable audit summary."""

    warning_rows = [row for row in rows if row["warning_messages"]]
    lines = [
        "# Paper Outputs Audit",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        f"- Status: `{'failed' if failed else 'passed'}`",
        f"- Required files checked: `{len(rows)}`",
        f"- Rows with warnings: `{len(warning_rows)}`",
        "",
        "| Category | Path | Exists | Rows | Columns | Warnings |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['path']} | {row['exists']} | {row['row_count']} | {row['column_count']} | {row['warning_messages']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run paper-output audit."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, failed = audit_rows()
    write_table(args.output_dir / "paper_outputs_audit.tsv", rows, FIELDS)
    (args.output_dir / "paper_outputs_audit.md").write_text(markdown_summary(rows, failed))
    print(f"Wrote paper output audit to {args.output_dir}; status={'failed' if failed else 'passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
