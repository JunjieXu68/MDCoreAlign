"""Run blocked, alignment-independent held-out geometric validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from common import (
    REPO_ROOT,
    RESULT_ROOT,
    ca_atom_metadata,
    read_tsv,
    resolve_mdcath_h5,
    sha256_file,
    stable_hash,
    write_json,
    write_tsv,
)

from mdcorealign.contracts import solve_contract
from mdsuperpose.mdlovofit import (
    read_multimodel_pdb_with_occupancy,
    resolve_mdlovofit_executable,
    run_mdlovofit_on_coordinates,
)

METHODS = [
    "MDCoreAlign_throughput",
    "MDCoreAlign_sequential_reference",
    "MDLovoFit",
    "random_equal_cardinality",
    "full_selection_noncore_baseline",
]
RESULT_FIELDS = [
    "source_type", "system_id", "fold_type", "fold_index", "method", "status", "error",
    "n_frames_total", "n_train_frames", "n_test_frames", "n_residues", "core_size",
    "core_fraction", "core_indices", "median_pair_distance_SD_A",
    "median_pair_distance_MAD_A", "median_pair_distance_CV", "P95_distance_fluctuation_A",
    "weighted_pair_distance_SD_A", "training_contact_count", "contact_persistence",
    "contact_state_entropy", "contact_break_form_rate", "intra_chain_contact_persistence",
    "inter_chain_contact_persistence", "pair_sample_count", "contact_threshold_A",
]
ATLAS_SYSTEMS = ["1jnr_C", "1r6w_A", "1tg7_A", "1u55_A", "3nci_A", "5vz0_A", "6qfk_A", "6sup_A"]


def write_registry_amendment() -> None:
    path = RESULT_ROOT / "02_preregistration/heldout_geometry_registry_amendment.json"
    payload = {
        "amendment_version": "SE7-pre-output-availability-correction-v1",
        "created_before_SE7_outcomes": True,
        "original_registry_preserved": True,
        "reason": (
            "Six ATLAS identifiers in the original registry do not exist in the locked "
            "local atlas_manifest.tsv. They are replaced solely from that availability "
            "manifest; no method result was inspected."
        ),
        "actual_locked_atlas_families": ATLAS_SYSTEMS,
        "atlas_frame_policy": (
            "use 1000 chronological temporally stratified frames from each locked "
            "10001-frame R1 trajectory before four-fold blocking"
        ),
        "mdcath_policy": "retain the originally hashed 512-domain 320K R1 panel",
        "external_method_timing": "untimed scientific core selection only",
    }
    write_json(path, payload)
    (path.with_suffix(".sha256")).write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )


def load_mdcath(domain_id: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(resolve_mdcath_h5(domain_id), "r") as h5:
        metadata = ca_atom_metadata(h5, domain_id)
        coords = np.asarray(
            h5[domain_id]["320"]["0"]["coords"][:, metadata["atom_indices"], :],
            dtype=np.float64,
        )
    return coords, np.asarray(metadata["chain_ids"])


def atlas_manifest() -> dict[str, dict[str, str]]:
    rows = read_tsv(REPO_ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")
    return {row["system"]: row for row in rows}


def load_atlas(system_id: str) -> tuple[np.ndarray, np.ndarray]:
    row = atlas_manifest()[system_id]
    source = np.load(REPO_ROOT / row["coordinate_cache_path"], mmap_mode="r")
    indices = np.unique(np.linspace(0, source.shape[0] - 1, 1000).astype(np.int64))
    coords = np.asarray(source[indices], dtype=np.float64)
    metadata = np.load(REPO_ROOT / row["residue_metadata_path"])
    return coords, np.asarray(metadata["chain_ids"])


def folds(n_frames: int, fold_type: str) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_frames, dtype=np.int64)
    output = []
    if fold_type == "contiguous":
        blocks = np.array_split(indices, 4)
    elif fold_type == "interleaved":
        blocks = [indices[index::4] for index in range(4)]
    else:
        raise ValueError(fold_type)
    for block in blocks:
        test = np.asarray(block, dtype=np.int64)
        train = np.setdiff1d(indices, test, assume_unique=True)
        output.append((train, test))
    return output


def select_core(
    method: str,
    train_coords: np.ndarray,
    chain_ids: np.ndarray,
    system_id: str,
    fold_type: str,
    fold_index: int,
    mdlovofit_executable: Path | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    n_residues = train_coords.shape[1]
    core_size = max(1, int(np.floor(0.60 * n_residues)))
    if method in {"MDCoreAlign_throughput", "MDCoreAlign_sequential_reference"}:
        contract = "throughput" if method.endswith("throughput") else "exact"
        result = solve_contract(
            train_coords,
            contract=contract,
            backend="cpu",
            chain_ids=chain_ids,
            store_aligned=False,
            compute_ordinary_rmsf=False,
        ).alignment
        if result is None:
            raise RuntimeError("MDCoreAlign returned no alignment")
        weights = np.asarray(result.frozen_weights, dtype=np.float64)
        order = np.lexsort((np.arange(n_residues), -weights))
        mask = np.zeros(n_residues, dtype=bool)
        mask[order[:core_size]] = True
        return mask, weights
    if method == "MDLovoFit":
        if mdlovofit_executable is None:
            raise FileNotFoundError("MDLovoFit executable unavailable")
        temporary_root = RESULT_ROOT / "07_heldout_geometry/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="se7_mdl_", dir=temporary_root) as temporary:
            run = run_mdlovofit_on_coordinates(
                train_coords,
                fraction=core_size / n_residues,
                workdir=temporary,
                executable=mdlovofit_executable,
                timeout_seconds=600.0,
            )
            _, occupancy = read_multimodel_pdb_with_occupancy(run.output_path, n_residues)
        if not np.allclose(occupancy, occupancy[0]):
            raise ValueError("MDLovoFit did not produce one fixed subset")
        return occupancy[0] > 0.5, None
    if method == "random_equal_cardinality":
        seed = int(stable_hash(f"random:{system_id}:{fold_type}:{fold_index}")[:16], 16)
        rng = np.random.default_rng(seed)
        mask = np.zeros(n_residues, dtype=bool)
        mask[rng.choice(n_residues, size=core_size, replace=False)] = True
        return mask, None
    if method == "full_selection_noncore_baseline":
        return np.ones(n_residues, dtype=bool), None
    raise ValueError(method)


def deterministic_pairs(indices: np.ndarray, key: str, maximum: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.triu_indices(indices.size, k=1)
    if first.size > maximum:
        rng = np.random.default_rng(int(stable_hash(key)[:16], 16))
        chosen = np.sort(rng.choice(first.size, size=maximum, replace=False))
        first = first[chosen]
        second = second[chosen]
    return indices[first], indices[second]


def pair_metrics(
    train: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    chains: np.ndarray,
    weights: np.ndarray | None,
    key: str,
) -> dict[str, object]:
    selected = np.flatnonzero(mask)
    left, right = deterministic_pairs(selected, key)
    if left.size == 0:
        raise ValueError("core contains fewer than two residues")
    pair_sd = np.empty(left.size, dtype=np.float64)
    pair_mad = np.empty(left.size, dtype=np.float64)
    pair_cv = np.empty(left.size, dtype=np.float64)
    p95_fluctuation = np.empty(left.size, dtype=np.float64)
    persistence = np.full(left.size, np.nan, dtype=np.float64)
    contact_entropy = np.full(left.size, np.nan, dtype=np.float64)
    contact_transition = np.full(left.size, np.nan, dtype=np.float64)
    reference_distance = np.linalg.norm(train[0, left] - train[0, right], axis=1)
    training_contacts = reference_distance <= 8.0
    for start in range(0, left.size, 2000):
        stop = min(start + 2000, left.size)
        distances = np.linalg.norm(
            test[:, left[start:stop], :] - test[:, right[start:stop], :],
            axis=2,
        )
        means = np.mean(distances, axis=0)
        pair_sd[start:stop] = np.std(distances, axis=0)
        pair_mad[start:stop] = np.median(np.abs(distances - np.median(distances, axis=0)), axis=0)
        pair_cv[start:stop] = pair_sd[start:stop] / np.maximum(means, 1e-12)
        p95_fluctuation[start:stop] = np.quantile(np.abs(distances - means), 0.95, axis=0)
        contacts = distances <= 8.0
        local_training = training_contacts[start:stop]
        if np.any(local_training):
            probabilities = np.mean(contacts[:, local_training], axis=0)
            target_indices = np.flatnonzero(local_training) + start
            persistence[target_indices] = probabilities
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            contact_entropy[target_indices] = -(
                clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped)
            )
            contact_transition[target_indices] = np.mean(
                contacts[1:, local_training] != contacts[:-1, local_training],
                axis=0,
            ) if contacts.shape[0] > 1 else 0.0
    pair_weights = None
    weighted_sd: float | str = ""
    if weights is not None:
        pair_weights = np.asarray(weights[left] * weights[right], dtype=np.float64)
        if float(np.sum(pair_weights)) > 0:
            weighted_sd = float(np.sum(pair_weights * pair_sd) / np.sum(pair_weights))
    contact_indices = np.flatnonzero(training_contacts)
    intra = chains[left] == chains[right]
    inter = ~intra
    def contact_mean(selection: np.ndarray) -> float | str:
        valid = selection & training_contacts & np.isfinite(persistence)
        return float(np.mean(persistence[valid])) if np.any(valid) else ""
    return {
        "median_pair_distance_SD_A": float(np.median(pair_sd)),
        "median_pair_distance_MAD_A": float(np.median(pair_mad)),
        "median_pair_distance_CV": float(np.median(pair_cv)),
        "P95_distance_fluctuation_A": float(np.quantile(p95_fluctuation, 0.95)),
        "weighted_pair_distance_SD_A": weighted_sd,
        "training_contact_count": int(contact_indices.size),
        "contact_persistence": float(np.nanmean(persistence)) if contact_indices.size else "",
        "contact_state_entropy": float(np.nanmean(contact_entropy)) if contact_indices.size else "",
        "contact_break_form_rate": float(np.nanmean(contact_transition)) if contact_indices.size else "",
        "intra_chain_contact_persistence": contact_mean(intra),
        "inter_chain_contact_persistence": contact_mean(inter),
        "pair_sample_count": int(left.size),
        "contact_threshold_A": 8.0,
    }


def process_system(source_type: str, system_id: str, include_interleaved: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        executable = resolve_mdlovofit_executable()
    except FileNotFoundError:
        executable = None
    coords, chains = load_mdcath(system_id) if source_type == "mdCATH" else load_atlas(system_id)
    result_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    fold_types = ["contiguous", "interleaved"] if include_interleaved else ["contiguous"]
    for fold_type in fold_types:
        for fold_index, (train_indices, test_indices) in enumerate(folds(coords.shape[0], fold_type)):
            train = coords[train_indices]
            test = coords[test_indices]
            for method in METHODS:
                base = {
                    "source_type": source_type,
                    "system_id": system_id,
                    "fold_type": fold_type,
                    "fold_index": fold_index,
                    "method": method,
                    "n_frames_total": coords.shape[0],
                    "n_train_frames": train.shape[0],
                    "n_test_frames": test.shape[0],
                    "n_residues": coords.shape[1],
                }
                try:
                    mask, weights = select_core(
                        method, train, chains, system_id, fold_type, fold_index, executable
                    )
                    metrics = pair_metrics(
                        train, test, mask, chains, weights,
                        f"{source_type}:{system_id}:{fold_type}:{fold_index}:{method}",
                    )
                    row = {
                        **base,
                        "status": "success",
                        "error": "",
                        "core_size": int(np.count_nonzero(mask)),
                        "core_fraction": float(np.mean(mask)),
                        "core_indices": ";".join(str(index) for index in np.flatnonzero(mask)),
                        **metrics,
                    }
                    selection_rows.append(
                        {
                            **base,
                            "status": "success",
                            "core_size": int(np.count_nonzero(mask)),
                            "core_indices": row["core_indices"],
                            "continuous_weights_available": weights is not None,
                        }
                    )
                except Exception as error:
                    row = {field: "" for field in RESULT_FIELDS}
                    row.update(base)
                    row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                    selection_rows.append(
                        {
                            **base,
                            "status": "failed",
                            "core_size": "",
                            "core_indices": "",
                            "continuous_weights_available": False,
                        }
                    )
                result_rows.append(row)
    return result_rows, selection_rows


def bootstrap_median(values: list[float], seed_key: str) -> tuple[float | str, float | str]:
    """Return a domain-level bootstrap interval for a median."""

    if not values:
        return "", ""
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(stable_hash(seed_key)[:16], 16))
    sampled = rng.choice(array, size=(2000, array.size), replace=True)
    medians = np.median(sampled, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def paired_summary(rows: list[dict[str, str]]) -> None:
    successful = [row for row in rows if row["status"] == "success"]
    metrics = ["median_pair_distance_SD_A", "contact_persistence", "contact_state_entropy", "contact_break_form_rate"]
    output = []
    for fold_type in ("contiguous", "interleaved"):
        for source_type in ("mdCATH", "ATLAS"):
            source_rows = [
                row
                for row in successful
                if row["source_type"] == source_type and row["fold_type"] == fold_type
            ]
            if not source_rows:
                continue
            for method in METHODS:
                method_rows = [row for row in source_rows if row["method"] == method]
                for metric in metrics:
                    by_system: dict[str, list[float]] = {}
                    for row in method_rows:
                        if row[metric] in {"", "nan"}:
                            continue
                        by_system.setdefault(row["system_id"], []).append(float(row[metric]))
                    values = [float(np.median(group)) for group in by_system.values()]
                    low, high = bootstrap_median(
                        values,
                        f"absolute:{source_type}:{fold_type}:{method}:{metric}",
                    )
                    output.append(
                        {
                            "source_type": source_type,
                            "fold_type": fold_type,
                            "comparison": method,
                            "metric": metric,
                            "n_units": len(values),
                            "median": float(np.median(values)) if values else "",
                            "IQR_low": float(np.quantile(values, 0.25)) if values else "",
                            "IQR_high": float(np.quantile(values, 0.75)) if values else "",
                            "bootstrap_95CI_low": low,
                            "bootstrap_95CI_high": high,
                            "bootstrap_unit": "domain_or_family",
                        }
                    )
            keyed = {
                (row["system_id"], int(row["fold_index"]), row["method"]): row
                for row in source_rows
            }
            systems = sorted({row["system_id"] for row in source_rows})
            for comparator in ("MDLovoFit", "random_equal_cardinality"):
                for method in (
                    "MDCoreAlign_throughput",
                    "MDCoreAlign_sequential_reference",
                    "MDLovoFit",
                ):
                    if method == comparator:
                        continue
                    for metric in metrics:
                        differences = []
                        for system in systems:
                            fold_values = []
                            for fold_index in range(4):
                                left = keyed.get((system, fold_index, method))
                                right = keyed.get((system, fold_index, comparator))
                                if (
                                    left
                                    and right
                                    and left[metric] not in {"", "nan"}
                                    and right[metric] not in {"", "nan"}
                                ):
                                    fold_values.append(float(left[metric]) - float(right[metric]))
                            if fold_values:
                                differences.append(float(np.median(fold_values)))
                        low, high = bootstrap_median(
                            differences,
                            f"paired:{source_type}:{fold_type}:{method}:{comparator}:{metric}",
                        )
                        output.append(
                            {
                                "source_type": source_type,
                                "fold_type": fold_type,
                                "comparison": f"{method}_minus_{comparator}",
                                "metric": metric,
                                "n_units": len(differences),
                                "median": float(np.median(differences)) if differences else "",
                                "IQR_low": float(np.quantile(differences, 0.25)) if differences else "",
                                "IQR_high": float(np.quantile(differences, 0.75)) if differences else "",
                                "bootstrap_95CI_low": low,
                                "bootstrap_95CI_high": high,
                                "bootstrap_unit": "domain_or_family",
                            }
                        )
    write_tsv(
        RESULT_ROOT / "07_heldout_geometry/paired_method_summary.tsv",
        output,
        list(output[0]),
    )
    mdca = [
        row
        for row in output
        if row["comparison"] == "MDCoreAlign_throughput_minus_MDLovoFit"
    ]
    failures = [row for row in rows if row["status"] != "success"]
    (RESULT_ROOT / "07_heldout_geometry/heldout_geometry_report.md").write_text(
        "# Held-out alignment-independent geometry report\n\n"
        f"The locked panel produced `{len(successful)}` successful method-fold "
        f"rows across contiguous and interleaved splits and `{len(failures)}` failed "
        "rows. Pair-distance and contact "
        "statistics were evaluated on held-out raw coordinates without a rigid fit. "
        "The equal-cardinality operating point was floor(0.60*N); full selection was "
        "retained only as a non-core baseline.\n\n"
        "MDCoreAlign-minus-MDLovoFit paired domain/family medians are listed in "
        "`paired_method_summary.tsv`; negative pair-distance differences favour "
        "MDCoreAlign, whereas positive contact-persistence differences favour "
        "MDCoreAlign. These metrics are independent geometric adjudicators, not "
        "biological truth. ATLAS uses 1,000 prospectively stratified R1 frames and "
        "does not establish R2/R3 reproducibility.\n\n"
        + "\n".join(
            f"- {row['source_type']} {row['fold_type']} {row['metric']}: "
            f"median paired difference `{row['median']}` "
            f"(95% CI `{row['bootstrap_95CI_low']}` to `{row['bootstrap_95CI_high']}`)"
            for row in mdca
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-mdcath", type=int, default=None)
    parser.add_argument("--include-interleaved", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    write_registry_amendment()
    registry = json.loads(
        (RESULT_ROOT / "02_preregistration/heldout_geometry_registry.json").read_text()
    )
    mdcath = list(registry["mdcath_domains"])
    if args.limit_mdcath is not None:
        mdcath = mdcath[: args.limit_mdcath]
    panel = [
        {"source_type": "mdCATH", "system_id": domain_id, "frame_policy": "all_320K_R1_frames", "selection_status": "original_locked_registry"}
        for domain_id in mdcath
    ] + [
        {"source_type": "ATLAS", "system_id": system_id, "frame_policy": "1000_temporally_stratified_R1_frames", "selection_status": "pre_output_availability_amendment"}
        for system_id in ATLAS_SYSTEMS
    ]
    write_tsv(RESULT_ROOT / "07_heldout_geometry/panel_lock.tsv", panel, list(panel[0]))
    fold_rows = []
    for row in panel:
        for fold_type in (["contiguous", "interleaved"] if args.include_interleaved else ["contiguous"]):
            for fold_index in range(4):
                fold_rows.append(
                    {
                        **row,
                        "fold_type": fold_type,
                        "fold_index": fold_index,
                        "training_blocks": "all_except_test_block",
                        "test_block": fold_index,
                    }
                )
    write_tsv(RESULT_ROOT / "07_heldout_geometry/fold_registry.tsv", fold_rows, list(fold_rows[0]))
    result_path = RESULT_ROOT / "07_heldout_geometry/pair_distance_results.tsv"
    existing: list[dict[str, str]] = []
    complete_systems: set[tuple[str, str]] = set()
    if result_path.exists():
        with result_path.open() as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
        counts: dict[tuple[str, str], int] = {}
        for row in existing:
            key = (row["source_type"], row["system_id"])
            counts[key] = counts.get(key, 0) + 1
        expected = len(METHODS) * 4 * (2 if args.include_interleaved else 1)
        complete_systems = {key for key, count in counts.items() if count == expected}
    pending = [row for row in panel if (row["source_type"], row["system_id"]) not in complete_systems]
    results: list[dict[str, object]] = list(existing)
    selections: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_system,
                row["source_type"],
                row["system_id"],
                args.include_interleaved,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            system_rows, selection_rows = future.result()
            results.extend(system_rows)
            selections.extend(selection_rows)
            if index % 5 == 0 or index == len(pending):
                write_tsv(result_path, results, RESULT_FIELDS)
                print(f"SE7 progress {len(complete_systems) + index}/{len(panel)} systems", flush=True)
    results.sort(key=lambda row: (str(row["source_type"]), str(row["system_id"]), str(row["fold_type"]), int(row["fold_index"]), str(row["method"])))
    write_tsv(result_path, results, RESULT_FIELDS)
    write_tsv(
        RESULT_ROOT / "07_heldout_geometry/contact_persistence_results.tsv",
        results,
        RESULT_FIELDS,
    )
    selection_rows = [
        {
            "source_type": row["source_type"],
            "system_id": row["system_id"],
            "fold_type": row["fold_type"],
            "fold_index": row["fold_index"],
            "method": row["method"],
            "status": row["status"],
            "n_frames_total": row["n_frames_total"],
            "n_train_frames": row["n_train_frames"],
            "n_test_frames": row["n_test_frames"],
            "n_residues": row["n_residues"],
            "core_size": row["core_size"],
            "core_indices": row["core_indices"],
            "continuous_weights_available": row["method"] in {
                "MDCoreAlign_throughput",
                "MDCoreAlign_sequential_reference",
            },
        }
        for row in results
    ]
    write_tsv(
        RESULT_ROOT / "07_heldout_geometry/core_selections.tsv",
        selection_rows,
        list(selection_rows[0]),
    )
    if args.limit_mdcath is None:
        paired_summary([dict(row) for row in results])
    print(json.dumps({"rows": len(results), "failures": sum(row["status"] != "success" for row in results)}))


if __name__ == "__main__":
    main()
