"""Batch smoke benchmark for 100 diverse CA-only NMR ensembles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import MDAnalysis as mda
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdsuperpose import (  # noqa: E402
    Trajectory,
    benchmark_result_to_record,
    benchmark_results_to_rows,
    compare_methods_with_soft_configurations,
    summarize_benchmark_rows,
    write_benchmark_csv,
    write_benchmark_json,
)


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the batch NMR benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-models-used", type=int, default=20)
    parser.add_argument("--min-models", type=int, default=3)
    parser.add_argument("--min-ca-atoms", type=int, default=20)
    parser.add_argument("--max-ca-atoms", type=int, default=450)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--robust-rule", type=str, default="huber")
    parser.add_argument("--robust-delta", type=float, default=0.25)
    parser.add_argument("--robust-max-iterations", type=int, default=20)
    parser.add_argument("--robust-tolerance", type=float, default=1e-6)
    parser.add_argument("--soft-warm-start", action="store_true")
    parser.add_argument(
        "--soft-frame-order",
        choices=("input", "reference_rmsd", "reference_centered_rmsd"),
        default="input",
    )
    parser.add_argument("--objective-tolerance", type=float, default=None)
    parser.add_argument("--objective-patience", type=int, default=0)
    parser.add_argument("--pilot-count", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nmr_ensemble_batch_20260326",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=ROOT / "_tmp_nmr_ensemble_batch",
    )
    parser.add_argument("--mdlovofit-executable", type=Path, default=None)
    parser.add_argument("--mdlovofit-config", type=Path, default=None)
    return parser.parse_args()


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    """POST one JSON payload and decode the JSON response."""

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _query_nmr_entry_ids(candidate_limit: int, page_size: int) -> list[str]:
    """Query RCSB for solution NMR protein entries with multiple models."""

    identifiers: list[str] = []
    start = 0
    while len(identifiers) < candidate_limit:
        payload = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "SOLUTION NMR",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "greater",
                            "value": 0,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.deposited_model_count",
                            "operator": "greater",
                            "value": 1,
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": page_size},
                "results_content_type": ["experimental"],
            },
        }
        response = _post_json(RCSB_SEARCH_URL, payload)
        result_set = response.get("result_set", [])
        if not result_set:
            break
        identifiers.extend([str(item["identifier"]) for item in result_set])
        start += page_size
    # preserve order while removing duplicates
    return list(dict.fromkeys(identifiers))[:candidate_limit]


def _download_pdb(pdb_id: str, cache_dir: Path) -> Path:
    """Download one PDB entry to the local cache if needed."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cache_dir / f"{pdb_id}.pdb"
    if pdb_path.exists():
        return pdb_path
    url = RCSB_PDB_DOWNLOAD.format(pdb_id=pdb_id)
    with urlopen(url, timeout=60) as response:
        pdb_path.write_bytes(response.read())
    return pdb_path


def _chain_ids_from_ca(ca_atoms) -> list[str]:
    """Return stable chain-like identifiers from one CA selection."""

    chain_ids: list[str] = []
    for atom in ca_atoms:
        chain_id = getattr(atom, "chainID", "") or atom.segid or "_"
        chain_ids.append(str(chain_id))
    return sorted(set(chain_ids))


def _length_bin(n_ca_atoms: int) -> str:
    """Assign one coarse length bin for diversity reporting."""

    if n_ca_atoms < 60:
        return "small"
    if n_ca_atoms < 110:
        return "medium"
    if n_ca_atoms < 180:
        return "large"
    if n_ca_atoms < 280:
        return "xlarge"
    return "very_large"


def _feature_tags(n_models: int, n_ca_atoms: int, chain_count: int) -> str:
    """Create one compact feature-tag string for reporting."""

    tags: list[str] = []
    tags.append(_length_bin(n_ca_atoms))
    tags.append("multichain" if chain_count > 1 else "single_chain")
    if n_models >= 50:
        tags.append("high_model_count")
    elif n_models >= 20:
        tags.append("medium_model_count")
    else:
        tags.append("low_model_count")
    return ",".join(tags)


def _inspect_candidate(
    pdb_id: str,
    *,
    cache_dir: Path,
    min_models: int,
    min_ca_atoms: int,
    max_ca_atoms: int,
    max_models_used: int,
) -> dict[str, object]:
    """Inspect one candidate NMR ensemble for first-pass feasibility."""

    row: dict[str, object] = {
        "pdb_id": pdb_id,
        "source_pdb_path": None,
        "n_models_original": None,
        "n_models_used": None,
        "n_atoms_total": None,
        "n_ca_atoms": None,
        "chain_count": None,
        "chain_ids": None,
        "atom_order_consistent": None,
        "all_models_same_atom_count": None,
        "ca_selection_clean": None,
        "length_bin": None,
        "feature_tags": None,
        "feasible": False,
        "rejection_reason": None,
    }

    try:
        pdb_path = _download_pdb(pdb_id, cache_dir)
        row["source_pdb_path"] = str(pdb_path)
        universe = mda.Universe(str(pdb_path))
        ca = universe.select_atoms("protein and name CA")
        n_models_original = int(len(universe.trajectory))
        row["n_models_original"] = n_models_original
        row["n_atoms_total"] = int(universe.atoms.n_atoms)
        row["n_ca_atoms"] = int(ca.n_atoms)
        chain_ids = _chain_ids_from_ca(ca)
        row["chain_ids"] = ",".join(chain_ids)
        row["chain_count"] = len(chain_ids)

        if n_models_original < min_models:
            row["rejection_reason"] = "too_few_models"
            return row
        if ca.n_atoms < min_ca_atoms:
            row["rejection_reason"] = "too_few_ca_atoms"
            return row
        if ca.n_atoms > max_ca_atoms:
            row["rejection_reason"] = "too_many_ca_atoms_for_smoke"
            return row

        first_signature = [
            (atom.segid, int(atom.resid), atom.resname, atom.name)
            for atom in universe.atoms
        ]
        atom_order_consistent = True
        model_atom_counts: list[int] = []
        ca_coords = []
        for _ts in universe.trajectory:
            current_signature = [
                (atom.segid, int(atom.resid), atom.resname, atom.name)
                for atom in universe.atoms
            ]
            if current_signature != first_signature:
                atom_order_consistent = False
            model_atom_counts.append(universe.atoms.n_atoms)
            ca_coords.append(ca.positions.copy())

        ca_coords = np.asarray(ca_coords, dtype=float)
        all_models_same_atom_count = bool(len(set(model_atom_counts)) == 1)
        ca_selection_clean = bool(ca.n_atoms > 0 and np.isfinite(ca_coords).all())
        row["atom_order_consistent"] = atom_order_consistent
        row["all_models_same_atom_count"] = all_models_same_atom_count
        row["ca_selection_clean"] = ca_selection_clean

        if not atom_order_consistent:
            row["rejection_reason"] = "atom_order_inconsistent"
            return row
        if not all_models_same_atom_count:
            row["rejection_reason"] = "atom_count_varies_across_models"
            return row
        if not ca_selection_clean:
            row["rejection_reason"] = "ca_selection_not_clean"
            return row

        row["n_models_used"] = min(n_models_original, max_models_used)
        row["length_bin"] = _length_bin(int(ca.n_atoms))
        row["feature_tags"] = _feature_tags(n_models_original, int(ca.n_atoms), len(chain_ids))
        row["feasible"] = True
        return row
    except (HTTPError, URLError, TimeoutError) as exc:
        row["rejection_reason"] = f"download_error:{type(exc).__name__}"
        return row
    except Exception as exc:  # noqa: BLE001
        row["rejection_reason"] = f"inspection_error:{type(exc).__name__}"
        return row


def _take_evenly_spaced(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Select roughly evenly spaced rows after sorting by CA length."""

    if count <= 0 or not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: (int(row["n_ca_atoms"]), int(row["n_models_original"]), str(row["pdb_id"])))
    if count >= len(sorted_rows):
        return sorted_rows
    indices = np.linspace(0, len(sorted_rows) - 1, count)
    chosen: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in indices:
        index = int(round(float(value)))
        row = sorted_rows[index]
        if str(row["pdb_id"]) not in seen:
            chosen.append(row)
            seen.add(str(row["pdb_id"]))
    if len(chosen) < count:
        for row in sorted_rows:
            if str(row["pdb_id"]) in seen:
                continue
            chosen.append(row)
            seen.add(str(row["pdb_id"]))
            if len(chosen) == count:
                break
    return chosen


def _select_diverse_entries(feasible_rows: list[dict[str, object]], target_count: int) -> list[dict[str, object]]:
    """Select a diverse set of feasible entries for the 100-case smoke pass."""

    if len(feasible_rows) < target_count:
        raise ValueError(f"only {len(feasible_rows)} feasible NMR ensembles were found; {target_count} are required")

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    def add_rows(rows: list[dict[str, object]]) -> None:
        for row in rows:
            pdb_id = str(row["pdb_id"])
            if pdb_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(pdb_id)

    multichain = [row for row in feasible_rows if int(row["chain_count"]) > 1]
    high_model = [row for row in feasible_rows if int(row["n_models_original"]) >= 50]
    very_large = [row for row in feasible_rows if int(row["n_ca_atoms"]) >= 220]

    add_rows(_take_evenly_spaced(multichain, min(10, len(multichain), target_count)))
    add_rows(_take_evenly_spaced(high_model, min(10, len(high_model), max(0, target_count - len(selected)))))
    add_rows(_take_evenly_spaced(very_large, min(10, len(very_large), max(0, target_count - len(selected)))))

    remaining = [row for row in feasible_rows if str(row["pdb_id"]) not in selected_ids]
    add_rows(_take_evenly_spaced(remaining, target_count - len(selected)))

    return selected[:target_count]


def _extract_ca_coords(pdb_path: Path, max_models_used: int) -> np.ndarray:
    """Extract CA coordinates from up to `max_models_used` models."""

    universe = mda.Universe(str(pdb_path))
    ca = universe.select_atoms("protein and name CA")
    coords = []
    for frame_index, _ts in enumerate(universe.trajectory):
        if frame_index >= max_models_used:
            break
        coords.append(ca.positions.copy())
    return np.asarray(coords, dtype=float)


def _aggregate_method_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate one overall summary by method across all selected datasets."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method_name"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    numeric_fields = [
        "runtime_seconds",
        "algorithm_runtime_seconds",
        "overhead_seconds",
        "unified_posthoc_rmsd",
        "mdlovofit_native_all_atom_rmsd",
        "mdlovofit_native_least_mobile_rmsd",
        "weighted_rmsd",
        "iteration_count",
        "core_fraction",
    ]
    for method_name, group in grouped.items():
        summary: dict[str, object] = {
            "method_name": method_name,
            "n_cases": len(group),
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in group if row.get(field) not in ("", None)]
            summary[f"{field}_mean"] = float(np.mean(values)) if values else None
            summary[f"{field}_median"] = float(np.median(values)) if values else None
        parsing_values = [bool(row["parsing_success"]) for row in group if row.get("parsing_success") not in ("", None)]
        summary["parsing_success_rate"] = float(np.mean(parsing_values)) if parsing_values else None
        summary_rows.append(summary)
    return sorted(summary_rows, key=lambda row: str(row["method_name"]))


def _aggregate_length_bin_summary(selected_rows: list[dict[str, object]], benchmark_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate soft-vs-external summary by length bin."""

    meta_by_pdb = {str(row["pdb_id"]): row for row in selected_rows}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in benchmark_rows:
        dataset_name = str(row["dataset_name"])
        if dataset_name not in meta_by_pdb:
            continue
        key = (str(meta_by_pdb[dataset_name]["length_bin"]), str(row["method_name"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (length_bin, method_name), group in sorted(grouped.items()):
        numeric = lambda field: [float(item[field]) for item in group if item.get(field) not in ("", None)]  # noqa: E731
        summary_rows.append(
            {
                "length_bin": length_bin,
                "method_name": method_name,
                "n_cases": len(group),
                "runtime_seconds_mean": float(np.mean(numeric("runtime_seconds"))) if numeric("runtime_seconds") else None,
                "unified_posthoc_rmsd_mean": float(np.mean(numeric("unified_posthoc_rmsd"))) if numeric("unified_posthoc_rmsd") else None,
                "weighted_rmsd_mean": float(np.mean(numeric("weighted_rmsd"))) if numeric("weighted_rmsd") else None,
                "mdlovofit_native_all_atom_rmsd_mean": float(np.mean(numeric("mdlovofit_native_all_atom_rmsd"))) if numeric("mdlovofit_native_all_atom_rmsd") else None,
            }
        )
    return summary_rows


def _build_summary_note(
    *,
    target_count: int,
    candidate_count: int,
    feasible_count: int,
    selected_rows: list[dict[str, object]],
    benchmark_summary_rows: list[dict[str, object]],
) -> str:
    """Write one concise markdown summary of the 100-case batch benchmark."""

    selected_n_ca = [int(row["n_ca_atoms"]) for row in selected_rows]
    selected_models = [int(row["n_models_original"]) for row in selected_rows]
    chain_counts = [int(row["chain_count"]) for row in selected_rows]
    multichain_count = sum(count > 1 for count in chain_counts)

    method_map = {str(row["method_name"]): row for row in benchmark_summary_rows}
    soft_row = method_map.get("soft_robust_nmr_ensemble")
    external_row = method_map.get("external_mdlovofit")
    ordinary_row = method_map.get("ordinary_alignment")

    lines = [
        "# NMR Ensemble Batch Benchmark Summary",
        "",
        "This batch should be interpreted as an `NMR ensemble benchmark`, not an MD trajectory benchmark.",
        "",
        "## Batch scope",
        "",
        f"- candidate_entries_inspected: `{candidate_count}`",
        f"- feasible_entries_found: `{feasible_count}`",
        f"- selected_entries_benchmarked: `{target_count}`",
        f"- CA atom count range: `{min(selected_n_ca)} - {max(selected_n_ca)}`",
        f"- original model count range: `{min(selected_models)} - {max(selected_models)}`",
        f"- multichain entries included: `{multichain_count}`",
        "",
        "## Overall method summary",
        "",
        "| method_name | n_cases | runtime_seconds_mean | unified_posthoc_rmsd_mean | weighted_rmsd_mean | mdlovofit_native_all_atom_rmsd_mean | mdlovofit_native_least_mobile_rmsd_mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in benchmark_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method_name", "")),
                    str(row.get("n_cases", "")),
                    str(row.get("runtime_seconds_mean", "")),
                    str(row.get("unified_posthoc_rmsd_mean", "")),
                    str(row.get("weighted_rmsd_mean", "")),
                    str(row.get("mdlovofit_native_all_atom_rmsd_mean", "")),
                    str(row.get("mdlovofit_native_least_mobile_rmsd_mean", "")),
                ]
            )
            + " |"
        )

    if soft_row is not None and external_row is not None:
        lines.extend(
            [
                "",
                "## Quick comparison",
                "",
                f"- soft_vs_external_runtime_mean: `{soft_row.get('runtime_seconds_mean')}` vs `{external_row.get('runtime_seconds_mean')}`",
                f"- soft_vs_external_unified_posthoc_rmsd_mean: `{soft_row.get('unified_posthoc_rmsd_mean')}` vs `{external_row.get('unified_posthoc_rmsd_mean')}`",
            ]
        )
    if ordinary_row is not None:
        lines.append(
            f"- ordinary_alignment_unified_posthoc_rmsd_mean: `{ordinary_row.get('unified_posthoc_rmsd_mean')}`"
        )

    lines.extend(
        [
            "",
            "## Incorporation recommendation",
            "",
            "- keep this as a separate `NMR ensemble benchmark` block",
            "- keep CA-only preprocessing for the first batch benchmark",
            "- do not merge these cases into the MD trajectory benchmark group",
            "- the current 100-case batch is appropriate for broader validation, but any paper-facing use should still keep the benchmark class explicitly separate",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the 100-case NMR ensemble smoke benchmark."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    original_pdb_dir = args.output_dir / "original_pdbs"
    candidate_ids = _query_nmr_entry_ids(args.candidate_limit, args.page_size)
    print(f"candidate_ids_fetched={len(candidate_ids)}")

    inspected_rows: list[dict[str, object]] = []
    feasible_rows: list[dict[str, object]] = []
    for index, pdb_id in enumerate(candidate_ids, start=1):
        row = _inspect_candidate(
            pdb_id,
            cache_dir=original_pdb_dir,
            min_models=args.min_models,
            min_ca_atoms=args.min_ca_atoms,
            max_ca_atoms=args.max_ca_atoms,
            max_models_used=args.max_models_used,
        )
        inspected_rows.append(row)
        if row["feasible"]:
            feasible_rows.append(row)
        if index % 25 == 0:
            print(f"inspected={index} feasible={len(feasible_rows)}")

    requested_count = args.pilot_count if args.pilot_count > 0 else args.target_count
    selected_rows = _select_diverse_entries(feasible_rows, requested_count)
    print(f"selected_for_benchmark={len(selected_rows)}")

    benchmark_rows: list[dict[str, object]] = []
    benchmark_records: list[dict[str, object]] = []
    selected_dir = args.output_dir / "selected_ca_inputs"
    selected_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for index, row in enumerate(selected_rows, start=1):
        pdb_id = str(row["pdb_id"])
        pdb_path = Path(str(row["source_pdb_path"]))
        coords = _extract_ca_coords(pdb_path, int(row["n_models_used"]))
        trajectory = Trajectory(coords=coords)
        results = compare_methods_with_soft_configurations(
            trajectory,
            mdlovofit_fraction=args.fraction,
            workdir=args.workdir / pdb_id,
            soft_configurations=[
                {
                    "method_name": "soft_robust_nmr_ensemble",
                    "core_fraction": args.fraction,
                    "rule": args.robust_rule,
                    "delta": args.robust_delta,
                    "max_iterations": args.robust_max_iterations,
                    "tolerance": args.robust_tolerance,
                    "warm_start": args.soft_warm_start,
                    "frame_order": args.soft_frame_order,
                    "objective_tolerance": args.objective_tolerance,
                    "objective_patience": args.objective_patience,
                }
            ],
            mdlovofit_executable=args.mdlovofit_executable,
            mdlovofit_config=args.mdlovofit_config,
        )
        preprocessing_steps = [
            "benchmark_class=nmr_ensemble",
            "interpretation=multi_conformation_ensemble_not_md_trajectory",
            f"source_pdb={pdb_path}",
            "selection=protein and name CA",
            "input_format=multi_model_pdb",
            f"n_models_original={row['n_models_original']}",
            f"n_models_used={row['n_models_used']}",
            f"n_ca_atoms={row['n_ca_atoms']}",
            f"soft_warm_start={args.soft_warm_start}",
            f"soft_frame_order={args.soft_frame_order}",
            f"objective_tolerance={args.objective_tolerance}",
            f"objective_patience={args.objective_patience}",
            f"chain_count={row['chain_count']}",
            f"feature_tags={row['feature_tags']}",
            "reference_mode=first_model",
        ]
        benchmark_rows.extend(
            benchmark_results_to_rows(
                results,
                benchmark_mode="nmr_ensemble",
                dataset_name=pdb_id,
                repeat_index=0,
                preprocessing_steps=preprocessing_steps,
            )
        )
        benchmark_records.append(
            {
                "pdb_id": pdb_id,
                "inspection": row,
                "results": [benchmark_result_to_record(result) for result in results],
            }
        )
        if index % 10 == 0 or index == len(selected_rows):
            elapsed = time.perf_counter() - start
            print(f"benchmarked={index}/{len(selected_rows)} elapsed_seconds={elapsed:.2f}")

    inspected_csv_path = args.output_dir / "candidate_inspection.csv"
    selected_csv_path = args.output_dir / "selected_nmr_ensembles.csv"
    runs_csv_path = args.output_dir / "nmr_ensemble_batch_runs.csv"
    runs_json_path = args.output_dir / "nmr_ensemble_batch_runs.json"
    per_dataset_summary_csv_path = args.output_dir / "nmr_ensemble_batch_summary_by_dataset.csv"
    overall_summary_csv_path = args.output_dir / "nmr_ensemble_batch_overall_method_summary.csv"
    length_bin_summary_csv_path = args.output_dir / "nmr_ensemble_batch_length_bin_summary.csv"
    summary_note_path = args.output_dir / "nmr_ensemble_batch_summary.md"
    config_json_path = args.output_dir / "nmr_ensemble_batch_config.json"

    write_benchmark_csv(inspected_csv_path, inspected_rows)
    write_benchmark_csv(selected_csv_path, selected_rows)
    write_benchmark_csv(runs_csv_path, benchmark_rows)
    per_dataset_summary_rows = summarize_benchmark_rows(benchmark_rows)
    write_benchmark_csv(per_dataset_summary_csv_path, per_dataset_summary_rows)
    overall_summary_rows = _aggregate_method_summary(benchmark_rows)
    write_benchmark_csv(overall_summary_csv_path, overall_summary_rows)
    length_bin_summary_rows = _aggregate_length_bin_summary(selected_rows, benchmark_rows)
    write_benchmark_csv(length_bin_summary_csv_path, length_bin_summary_rows)
    write_benchmark_json(
        runs_json_path,
        {
            "candidate_ids_fetched": candidate_ids,
            "candidate_rows": inspected_rows,
            "selected_rows": selected_rows,
            "benchmark_records": benchmark_records,
        },
    )
    summary_note_path.write_text(
        _build_summary_note(
            target_count=len(selected_rows),
            candidate_count=len(inspected_rows),
            feasible_count=len(feasible_rows),
            selected_rows=selected_rows,
            benchmark_summary_rows=overall_summary_rows,
        )
    )
    write_benchmark_json(
        config_json_path,
        {
            "target_count": args.target_count,
            "candidate_limit": args.candidate_limit,
            "page_size": args.page_size,
            "max_models_used": args.max_models_used,
            "min_models": args.min_models,
            "min_ca_atoms": args.min_ca_atoms,
            "max_ca_atoms": args.max_ca_atoms,
            "fraction": args.fraction,
            "robust_rule": args.robust_rule,
            "robust_delta": args.robust_delta,
            "robust_max_iterations": args.robust_max_iterations,
            "robust_tolerance": args.robust_tolerance,
            "pilot_count": args.pilot_count,
        },
    )

    print(f"candidate_inspection_csv={inspected_csv_path}")
    print(f"selected_nmr_ensembles_csv={selected_csv_path}")
    print(f"nmr_batch_runs_csv={runs_csv_path}")
    print(f"nmr_batch_summary_by_dataset_csv={per_dataset_summary_csv_path}")
    print(f"nmr_batch_overall_method_summary_csv={overall_summary_csv_path}")
    print(f"nmr_batch_length_bin_summary_csv={length_bin_summary_csv_path}")
    print(f"nmr_batch_summary_md={summary_note_path}")
    print(f"nmr_batch_config_json={config_json_path}")


if __name__ == "__main__":
    main()
