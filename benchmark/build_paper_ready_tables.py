"""Build paper-ready benchmark tables from structured raw benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


METHOD_LABELS = {
    "ordinary_alignment": "Ordinary alignment",
    "soft_weight_robust_alignment": "Soft robust",
    "soft_weight_no_regularization": "Soft robust",
    "external_mdlovofit": "MDLovoFit",
    "ordinary_alignment_pair": "Ordinary alignment",
    "soft_weight_robust_alignment_pair": "Soft robust",
    "external_mdlovofit_pair": "MDLovoFit",
    "soft_weight_sequence_regularized": "Soft + sequence prior",
    "soft_weight_contact_regularized": "Soft + contact prior",
    "soft_weight_seq_contact_regularized": "Soft + seq + contact",
    "ablation_base_no_regularization": "Soft robust",
    "ablation_sequence_only": "Soft + sequence prior",
    "ablation_contact_only": "Soft + contact prior",
    "ablation_sequence_plus_contact": "Soft + seq + contact",
}

SCENARIO_LABELS = {
    "clean_rigid_transform_only": "Clean rigid",
    "rigid_transform_moderate_outliers": "Moderate outliers",
    "rigid_transform_many_outliers": "Many outliers",
    "two_quasi_rigid_domains": "Two quasi-rigid domains",
}

DATASET_LABELS = {
    "ifabp": "I-FABP",
    "adk_equilibrium": "ADK equilibrium",
    "ifabp_first_last_ca": "I-FABP first/last CA pair",
}

TABLE_COLUMNS = [
    "case_id",
    "case_label",
    "method_name",
    "method_label",
    "runtime_seconds_mean",
    "runtime_seconds_std",
    "algorithm_runtime_seconds_mean",
    "algorithm_runtime_seconds_std",
    "overhead_seconds_mean",
    "overhead_seconds_std",
    "unified_posthoc_rmsd_mean",
    "unified_posthoc_rmsd_std",
    "mdlovofit_native_rmsd_mean",
    "mdlovofit_native_rmsd_std",
    "weighted_rmsd_mean",
    "weighted_rmsd_std",
    "iteration_count_mean",
    "iteration_count_std",
    "core_fraction_mean",
    "core_fraction_std",
    "parsing_success_rate",
]

BIO_EXTRA_COLUMNS = [
    "core_transition_count_mean",
    "core_transition_count_std",
    "weight_total_variation_mean",
    "weight_total_variation_std",
    "contact_graph_disagreement_mean",
    "contact_graph_disagreement_std",
    "converged_fraction_mean",
    "converged_fraction_std",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for paper-ready table generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=ROOT / "results" / "paper_ready_20260326",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read one CSV file into a list of dictionaries."""

    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write structured rows with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> None:
    """Write a compact markdown table with selected columns."""

    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "")) if row.get(key, "") is not None else "" for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def to_float(value: str | object | None) -> float | None:
    """Convert a CSV scalar into float when present."""

    if value in (None, ""):
        return None
    return float(value)


def mean_std_text(mean: object, std: object, digits: int = 3) -> str:
    """Format mean/std values for a paper-style markdown table."""

    mean_value = to_float(mean)
    std_value = to_float(std)
    if mean_value is None:
        return ""
    if std_value is None or abs(std_value) < 1e-15:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def rate_text(value: object, digits: int = 2) -> str:
    """Format a rate-like value when present."""

    numeric = to_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def labeled_case(case_id: str) -> str:
    """Map a raw case identifier to a human-readable label."""

    return SCENARIO_LABELS.get(case_id, DATASET_LABELS.get(case_id, case_id))


def labeled_method(method_name: str) -> str:
    """Map a raw method identifier to a human-readable label."""

    return METHOD_LABELS.get(method_name, method_name)


def build_base_row(summary_row: dict[str, str], *, case_id: str) -> dict[str, object]:
    """Normalize one summary row into the common paper-table schema."""

    row = {
        "case_id": case_id,
        "case_label": labeled_case(case_id),
        "method_name": summary_row["method_name"],
        "method_label": labeled_method(summary_row["method_name"]),
    }
    for column in TABLE_COLUMNS[4:]:
        row[column] = to_float(summary_row.get(column))
    return row


def aggregate_metric_rows(
    rows: list[dict[str, str]],
    *,
    method_field: str = "method_name",
    numeric_fields: list[str],
) -> dict[str, dict[str, float | None]]:
    """Aggregate repeated metric rows into mean/std dictionaries by method."""

    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        method = row[method_field]
        bucket = grouped.setdefault(method, {field: [] for field in numeric_fields})
        for field in numeric_fields:
            value = to_float(row.get(field))
            if value is not None:
                bucket[field].append(value)

    aggregated: dict[str, dict[str, float | None]] = {}
    for method, field_map in grouped.items():
        aggregated[method] = {}
        for field, values in field_map.items():
            if not values:
                aggregated[method][f"{field}_mean"] = None
                aggregated[method][f"{field}_std"] = None
                continue
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            aggregated[method][f"{field}_mean"] = mean_value
            aggregated[method][f"{field}_std"] = variance ** 0.5
    return aggregated


def aggregate_extra_metadata_from_runs(
    rows: list[dict[str, str]],
    *,
    numeric_fields: list[str],
) -> dict[str, dict[str, float | None]]:
    """Aggregate numeric metadata stored inside the extra_metadata JSON field."""

    expanded_rows: list[dict[str, str]] = []
    for row in rows:
        metadata = json.loads(row["extra_metadata"])
        expanded = {"method_name": row["method_name"]}
        for field in numeric_fields:
            value = metadata.get(field)
            expanded[field] = "" if value is None else str(value)
        expanded_rows.append(expanded)
    return aggregate_metric_rows(expanded_rows, numeric_fields=numeric_fields)


def load_structure_pair_provenance(payload_path: Path) -> dict[str, str]:
    """Extract provenance fields for the named real structure pair."""

    payload = json.loads(payload_path.read_text())
    steps = payload.get("preprocessing_steps", [])
    parsed: dict[str, str] = {}
    for step in steps:
        if "=" not in step:
            continue
        key, value = step.split("=", 1)
        parsed[key] = value
    return parsed


def write_paper_table(
    table_name: str,
    rows: list[dict[str, object]],
    *,
    markdown_rows: list[dict[str, object]],
    output_dir: Path,
    markdown_columns: list[tuple[str, str]],
) -> None:
    """Write both CSV and markdown versions of one paper-ready table."""

    csv_path = output_dir / f"{table_name}.csv"
    md_path = output_dir / f"{table_name}.md"
    write_csv_rows(csv_path, rows)
    write_markdown_table(md_path, markdown_rows, markdown_columns)


def build_synthetic_table(raw_dir: Path, output_dir: Path) -> None:
    """Build the paper-ready synthetic benchmark table."""

    summary_rows = read_csv_rows(raw_dir / "synthetic_summary.csv")
    rows = [build_base_row(summary_row, case_id=summary_row["dataset_name"]) for summary_row in summary_rows]
    rows.sort(key=lambda row: (row["case_label"], row["method_label"]))

    pretty_rows: list[dict[str, object]] = []
    for row in rows:
        pretty_rows.append(
            {
                "Case": row["case_label"],
                "Method": row["method_label"],
                "Runtime (s)": mean_std_text(row["runtime_seconds_mean"], row["runtime_seconds_std"]),
                "Algorithm (s)": mean_std_text(
                    row["algorithm_runtime_seconds_mean"],
                    row["algorithm_runtime_seconds_std"],
                ),
                "Overhead (s)": mean_std_text(row["overhead_seconds_mean"], row["overhead_seconds_std"]),
                "Unified post-hoc RMSD": mean_std_text(
                    row["unified_posthoc_rmsd_mean"],
                    row["unified_posthoc_rmsd_std"],
                    digits=4,
                ),
                "MDLovoFit-native RMSD": mean_std_text(
                    row["mdlovofit_native_rmsd_mean"],
                    row["mdlovofit_native_rmsd_std"],
                    digits=4,
                ),
                "Weighted RMSD": mean_std_text(row["weighted_rmsd_mean"], row["weighted_rmsd_std"], digits=4),
                "Iterations": mean_std_text(row["iteration_count_mean"], row["iteration_count_std"], digits=2),
                "Core fraction": mean_std_text(row["core_fraction_mean"], row["core_fraction_std"], digits=3),
                "Parsing success": rate_text(row["parsing_success_rate"]),
            }
        )

    write_paper_table(
        "synthetic_benchmark_summary",
        rows,
        markdown_rows=pretty_rows,
        output_dir=output_dir,
        markdown_columns=[
            ("Case", "Case"),
            ("Method", "Method"),
            ("Runtime (s)", "Runtime (s)"),
            ("Unified post-hoc RMSD", "Unified post-hoc RMSD"),
            ("MDLovoFit-native RMSD", "MDLovoFit-native RMSD"),
            ("Weighted RMSD", "Weighted RMSD"),
            ("Iterations", "Iterations"),
            ("Core fraction", "Core fraction"),
        ],
    )
    write_csv_rows(output_dir / "synthetic_benchmark_summary_readable.csv", pretty_rows)


def build_real_summary_table(
    *,
    summary_path: Path,
    output_dir: Path,
    table_name: str,
) -> list[dict[str, object]]:
    """Build one clean real-trajectory summary table."""

    summary_rows = read_csv_rows(summary_path)
    case_id = summary_rows[0]["dataset_name"]
    rows = [build_base_row(summary_row, case_id=case_id) for summary_row in summary_rows]
    rows.sort(key=lambda row: row["method_label"])
    pretty_rows: list[dict[str, object]] = []
    for row in rows:
        pretty_rows.append(
            {
                "Case": row["case_label"],
                "Method": row["method_label"],
                "Runtime (s)": mean_std_text(row["runtime_seconds_mean"], row["runtime_seconds_std"]),
                "Algorithm (s)": mean_std_text(
                    row["algorithm_runtime_seconds_mean"],
                    row["algorithm_runtime_seconds_std"],
                ),
                "Overhead (s)": mean_std_text(row["overhead_seconds_mean"], row["overhead_seconds_std"]),
                "Unified post-hoc RMSD": mean_std_text(
                    row["unified_posthoc_rmsd_mean"],
                    row["unified_posthoc_rmsd_std"],
                    digits=4,
                ),
                "MDLovoFit-native RMSD": mean_std_text(
                    row["mdlovofit_native_rmsd_mean"],
                    row["mdlovofit_native_rmsd_std"],
                    digits=4,
                ),
                "Weighted RMSD": mean_std_text(row["weighted_rmsd_mean"], row["weighted_rmsd_std"], digits=4),
                "Iterations": mean_std_text(row["iteration_count_mean"], row["iteration_count_std"], digits=2),
                "Core fraction": mean_std_text(row["core_fraction_mean"], row["core_fraction_std"], digits=3),
                "Parsing success": rate_text(row["parsing_success_rate"]),
            }
        )
    write_paper_table(
        table_name,
        rows,
        markdown_rows=pretty_rows,
        output_dir=output_dir,
        markdown_columns=[
            ("Case", "Case"),
            ("Method", "Method"),
            ("Runtime (s)", "Runtime (s)"),
            ("Algorithm (s)", "Algorithm (s)"),
            ("Overhead (s)", "Overhead (s)"),
            ("Unified post-hoc RMSD", "Unified post-hoc RMSD"),
            ("MDLovoFit-native RMSD", "MDLovoFit-native RMSD"),
            ("Weighted RMSD", "Weighted RMSD"),
            ("Iterations", "Iterations"),
            ("Core fraction", "Core fraction"),
        ],
    )
    write_csv_rows(output_dir / f"{table_name}_readable.csv", pretty_rows)
    return rows


def build_structure_pair_table(raw_dir: Path, output_dir: Path) -> None:
    """Build the real structure-pair paper table with explicit provenance."""

    summary_rows = read_csv_rows(raw_dir / "ifabp_first_last_ca_structure_pair_summary.csv")
    provenance = load_structure_pair_provenance(raw_dir / "ifabp_first_last_ca_structure_pair_runs.json")
    rows = [build_base_row(summary_row, case_id=summary_row["dataset_name"]) for summary_row in summary_rows]
    for row in rows:
        row["source_topology"] = provenance.get("topology_path")
        row["source_trajectory"] = provenance.get("trajectory_path")
        row["selection"] = provenance.get("selection")
        row["reference_frame_index"] = provenance.get("reference_frame_index")
        row["mobile_frame_index"] = provenance.get("mobile_frame_index")
        row["correspondence"] = provenance.get("correspondence")
    rows.sort(key=lambda row: row["method_label"])

    pretty_rows: list[dict[str, object]] = []
    for row in rows:
        pretty_rows.append(
            {
                "Case": row["case_label"],
                "Method": row["method_label"],
                "Runtime (s)": mean_std_text(row["runtime_seconds_mean"], row["runtime_seconds_std"]),
                "Algorithm (s)": mean_std_text(
                    row["algorithm_runtime_seconds_mean"],
                    row["algorithm_runtime_seconds_std"],
                ),
                "Overhead (s)": mean_std_text(row["overhead_seconds_mean"], row["overhead_seconds_std"]),
                "Unified post-hoc RMSD": mean_std_text(
                    row["unified_posthoc_rmsd_mean"],
                    row["unified_posthoc_rmsd_std"],
                    digits=4,
                ),
                "MDLovoFit-native RMSD": mean_std_text(
                    row["mdlovofit_native_rmsd_mean"],
                    row["mdlovofit_native_rmsd_std"],
                    digits=4,
                ),
                "Weighted RMSD": mean_std_text(row["weighted_rmsd_mean"], row["weighted_rmsd_std"], digits=4),
                "Iterations": mean_std_text(row["iteration_count_mean"], row["iteration_count_std"], digits=2),
                "Core fraction": mean_std_text(row["core_fraction_mean"], row["core_fraction_std"], digits=3),
                "Topology": row["source_topology"],
                "Trajectory": row["source_trajectory"],
                "Selection": row["selection"],
                "Reference frame": row["reference_frame_index"],
                "Mobile frame": row["mobile_frame_index"],
                "Correspondence": row["correspondence"],
            }
        )

    write_paper_table(
        "real_structure_pair_summary",
        rows,
        markdown_rows=pretty_rows,
        output_dir=output_dir,
        markdown_columns=[
            ("Case", "Case"),
            ("Method", "Method"),
            ("Runtime (s)", "Runtime (s)"),
            ("Unified post-hoc RMSD", "Unified post-hoc RMSD"),
            ("MDLovoFit-native RMSD", "MDLovoFit-native RMSD"),
            ("Weighted RMSD", "Weighted RMSD"),
            ("Iterations", "Iterations"),
            ("Core fraction", "Core fraction"),
        ],
    )
    write_csv_rows(output_dir / "real_structure_pair_summary_readable.csv", pretty_rows)
    write_markdown_table(
        output_dir / "real_structure_pair_provenance.md",
        [
            {
                "Case": DATASET_LABELS.get("ifabp_first_last_ca", "ifabp_first_last_ca"),
                "Topology": provenance.get("topology_path", ""),
                "Trajectory": provenance.get("trajectory_path", ""),
                "Selection": provenance.get("selection", ""),
                "Reference frame": provenance.get("reference_frame_index", ""),
                "Mobile frame": provenance.get("mobile_frame_index", ""),
                "Correspondence": provenance.get("correspondence", ""),
            }
        ],
        [
            ("Case", "Case"),
            ("Topology", "Topology"),
            ("Trajectory", "Trajectory"),
            ("Selection", "Selection"),
            ("Reference frame", "Reference frame"),
            ("Mobile frame", "Mobile frame"),
            ("Correspondence", "Correspondence"),
        ],
    )


def build_biological_table(raw_dir: Path, output_dir: Path) -> None:
    """Build the biological plausibility summary table."""

    summary_rows = read_csv_rows(raw_dir / "ifabp_biological_summary.csv")
    metric_rows = read_csv_rows(raw_dir / "ifabp_biological_metrics.csv")
    extra = aggregate_metric_rows(
        metric_rows,
        numeric_fields=[
            "core_transition_count",
            "weight_total_variation",
            "contact_graph_disagreement",
            "converged_fraction",
        ],
    )

    rows = [build_base_row(summary_row, case_id=summary_row["dataset_name"]) for summary_row in summary_rows]
    for row in rows:
        values = extra.get(row["method_name"], {})
        for key, value in values.items():
            row[key] = value
    rows.sort(key=lambda row: row["method_label"])

    pretty_rows: list[dict[str, object]] = []
    for row in rows:
        pretty_rows.append(
            {
                "Case": row["case_label"],
                "Method": row["method_label"],
                "Runtime (s)": mean_std_text(row["runtime_seconds_mean"], row["runtime_seconds_std"]),
                "Unified post-hoc RMSD": mean_std_text(
                    row["unified_posthoc_rmsd_mean"],
                    row["unified_posthoc_rmsd_std"],
                    digits=4,
                ),
                "Weighted RMSD": mean_std_text(row["weighted_rmsd_mean"], row["weighted_rmsd_std"], digits=4),
                "Core transitions": mean_std_text(
                    row.get("core_transition_count_mean"),
                    row.get("core_transition_count_std"),
                    digits=2,
                ),
                "Weight total variation": mean_std_text(
                    row.get("weight_total_variation_mean"),
                    row.get("weight_total_variation_std"),
                    digits=3,
                ),
                "Contact disagreement": mean_std_text(
                    row.get("contact_graph_disagreement_mean"),
                    row.get("contact_graph_disagreement_std"),
                    digits=4,
                ),
                "Converged fraction": mean_std_text(
                    row.get("converged_fraction_mean"),
                    row.get("converged_fraction_std"),
                    digits=3,
                ),
            }
        )

    write_paper_table(
        "biological_plausibility_summary",
        rows,
        markdown_rows=pretty_rows,
        output_dir=output_dir,
        markdown_columns=[
            ("Case", "Case"),
            ("Method", "Method"),
            ("Runtime (s)", "Runtime (s)"),
            ("Unified post-hoc RMSD", "Unified post-hoc RMSD"),
            ("Weighted RMSD", "Weighted RMSD"),
            ("Core transitions", "Core transitions"),
            ("Weight total variation", "Weight total variation"),
            ("Contact disagreement", "Contact disagreement"),
        ],
    )
    write_csv_rows(output_dir / "biological_plausibility_summary_readable.csv", pretty_rows)


def build_ablation_table(raw_dir: Path, output_dir: Path) -> None:
    """Build the ablation summary table, including core-smoothness metrics."""

    summary_rows = read_csv_rows(raw_dir / "ifabp_ablation_summary.csv")
    run_rows = read_csv_rows(raw_dir / "ifabp_ablation_runs.csv")
    extra = aggregate_extra_metadata_from_runs(
        run_rows,
        numeric_fields=[
            "core_transition_count",
            "weight_total_variation",
            "contact_graph_disagreement",
            "converged_fraction",
        ],
    )

    rows = [build_base_row(summary_row, case_id=summary_row["dataset_name"]) for summary_row in summary_rows]
    for row in rows:
        values = extra.get(row["method_name"], {})
        for key, value in values.items():
            row[key] = value
    rows.sort(key=lambda row: row["method_label"])

    pretty_rows: list[dict[str, object]] = []
    for row in rows:
        pretty_rows.append(
            {
                "Case": row["case_label"],
                "Method": row["method_label"],
                "Runtime (s)": mean_std_text(row["runtime_seconds_mean"], row["runtime_seconds_std"]),
                "Unified post-hoc RMSD": mean_std_text(
                    row["unified_posthoc_rmsd_mean"],
                    row["unified_posthoc_rmsd_std"],
                    digits=4,
                ),
                "Weighted RMSD": mean_std_text(row["weighted_rmsd_mean"], row["weighted_rmsd_std"], digits=4),
                "Iterations": mean_std_text(row["iteration_count_mean"], row["iteration_count_std"], digits=2),
                "Core fraction": mean_std_text(row["core_fraction_mean"], row["core_fraction_std"], digits=3),
                "Core transitions": mean_std_text(
                    row.get("core_transition_count_mean"),
                    row.get("core_transition_count_std"),
                    digits=2,
                ),
                "Weight total variation": mean_std_text(
                    row.get("weight_total_variation_mean"),
                    row.get("weight_total_variation_std"),
                    digits=3,
                ),
                "Contact disagreement": mean_std_text(
                    row.get("contact_graph_disagreement_mean"),
                    row.get("contact_graph_disagreement_std"),
                    digits=4,
                ),
            }
        )

    write_paper_table(
        "ablation_summary",
        rows,
        markdown_rows=pretty_rows,
        output_dir=output_dir,
        markdown_columns=[
            ("Case", "Case"),
            ("Method", "Method"),
            ("Runtime (s)", "Runtime (s)"),
            ("Unified post-hoc RMSD", "Unified post-hoc RMSD"),
            ("Weighted RMSD", "Weighted RMSD"),
            ("Iterations", "Iterations"),
            ("Core transitions", "Core transitions"),
        ],
    )
    write_csv_rows(output_dir / "ablation_summary_readable.csv", pretty_rows)


def main() -> None:
    """Build all paper-ready summary tables from the current raw benchmark outputs."""

    args = parse_args()
    raw_dir = args.paper_root / "raw"
    output_dir = args.paper_root / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_synthetic_table(raw_dir / "synthetic", output_dir)
    build_real_summary_table(
        summary_path=raw_dir / "trajectory_ifabp" / "ifabp_real_summary.csv",
        output_dir=output_dir,
        table_name="ifabp_trajectory_summary",
    )
    build_real_summary_table(
        summary_path=raw_dir / "trajectory_adk" / "adk_equilibrium_real_summary.csv",
        output_dir=output_dir,
        table_name="real_case_adk_trajectory_summary",
    )
    build_structure_pair_table(raw_dir / "structure_pair", output_dir)
    build_biological_table(raw_dir / "biology", output_dir)
    build_ablation_table(raw_dir / "ablation", output_dir)

    print("Paper-ready tables complete")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
