"""Analysis-only replay diagnostics and five fixed representative-frame perturbations.

Calls the existing production estimator with explicitly selected representatives;
does not run exact, MDLovoFit, all-frame fitting, or performance measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_info, threadpool_limits

from mdcorealign.benchmark_minimal import load_ca_multimodel_pdb_minimal
from mdsuperpose import window_refine

SEEDS = (20260915, 20260916, 20260917, 20260918, 20260919)
SAMPLING_RULE = (
    "Keep local training indices 0 and n-1. Partition the n-2 interior indices "
    "into count-2 disjoint consecutive strata using integer edges "
    "1+floor(b*(n-2)/(count-2)); sample one index uniformly from each half-open "
    "stratum with numpy.default_rng(seed). Keep temporal order. The fixed "
    "reference is always the original first frame, passed separately."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_tsv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_ranks(weights: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(weights)), -weights))
    ranks = np.empty(len(weights), dtype=np.int64)
    ranks[order] = np.arange(1, len(weights) + 1)
    return ranks


def stratified_sample(n_frames: int, count: int, seed: int) -> np.ndarray:
    """Anchor both endpoints and draw once per disjoint interior time stratum."""
    if not 2 <= count <= n_frames:
        raise ValueError("Require 2 <= representative count <= training frames")
    if count == 2:
        return np.asarray([0, n_frames - 1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    edges = 1 + np.arange(count - 1, dtype=np.int64) * (n_frames - 2) // (count - 2)
    middle = [int(rng.integers(int(a), int(b))) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    result = np.asarray([0, *middle, n_frames - 1], dtype=np.int64)
    if len(result) != count or np.any(np.diff(result) <= 0):
        raise AssertionError("Representative count/order/uniqueness invariant failed")
    return result


def estimate_with_indices(
    coords: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
    chains: np.ndarray,
    *,
    diagnostics: bool = False,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Use the unchanged bounded estimator; optionally observe its real arrays.

    The wrapper delegates every continuity call to the original implementation.
    No update or regularization formula is duplicated here. Caller-local reads
    are guarded by code identity and restored in finally, including on failure.
    """
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or not len(selected) or np.any(np.diff(selected) <= 0):
        raise ValueError("Representative indices must be nonempty, sorted and unique")
    if selected[0] < 0 or selected[-1] >= len(coords):
        raise ValueError("Representative index outside training source")
    trace: dict = {}
    rounds: list[dict] = []
    original = window_refine.apply_sequence_continuity_regularization_batch
    code = window_refine.bounded_global_weight_refinement.__code__

    def capture(weights: np.ndarray, **kwargs: object) -> np.ndarray:
        frame = sys._getframe(1)
        if frame.f_code is not code:
            raise RuntimeError("Unexpected caller: refusing ambiguous diagnostic capture")
        local = frame.f_locals
        record = {
            "round": int(local["round_index"]) + 1,
            "weights_before_update": local["weights"].copy(),
            "reference_distances_A": local["distances"].copy(),
            "robust_frame_weights": weights.copy(),
        }
        output = original(weights, **kwargs)
        record["continuity_frame_weights"] = output.copy()
        rounds.append(record)
        return output

    try:
        if diagnostics:
            window_refine.apply_sequence_continuity_regularization_batch = capture
        weights, _, _, _ = window_refine.bounded_global_weight_refinement(
            coords[selected],
            reference,
            np.ones(coords.shape[1], dtype=np.float64),
            update_rounds=4,
            delta=1.0,
            sequence_continuity_strength=0.05,
            sequence_chain_ids=chains,
            precision="float64",
            trace=trace,
        )
    finally:
        window_refine.apply_sequence_continuity_regularization_batch = original
    if diagnostics:
        if [r["round"] for r in rounds] != [1, 2, 3, 4]:
            raise AssertionError("Expected exactly four original continuity calls")
        for i, record in enumerate(rounds):
            np.testing.assert_array_equal(
                record["weights_before_update"], trace["weight_history"][i]
            )
            record["normalized_weights_after_update"] = trace["weight_history"][i + 1].copy()
    np.testing.assert_array_equal(weights, trace["weight_history"][-1])
    return weights, trace, rounds


def prepare_selection_cache(case: dict, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    source = Path(case["selection_input_pdb"])
    cache = Path(case["selection_coords_cache"])
    metadata_path = cache.with_suffix(".metadata.json")
    summary = json.loads((Path(case["source_case_directory"]) / "case_summary.json").read_text())
    expected_shape = (summary["n_frames"], len(case["residues"]), 3)
    if cache.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_sha256"] != sha256(source) or metadata["cache_sha256"] != sha256(cache):
            raise ValueError("Selection cache or source hash mismatch")
        values = np.load(cache, mmap_mode="r")
        chains = np.asarray(metadata["chain_ids"])
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        values, chains = load_ca_multimodel_pdb_minimal(
            source,
            expected_frames=expected_shape[0],
            expected_residues=expected_shape[1],
        )
        temporary = cache.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values)
        temporary.replace(cache)
        metadata = {
            "schema_version": 1,
            "case": case["case_id"],
            "run_id": config["run_id"],
            "source_path": str(source),
            "source_sha256": sha256(source),
            "cache_path": str(cache),
            "cache_sha256": sha256(cache),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "chain_ids": chains.tolist(),
            "loader": "mdcorealign.benchmark_minimal.load_ca_multimodel_pdb_minimal",
            "scope": "all common_input.pdb frames; use frozen train_indices for estimation",
        }
        write_json(metadata_path, metadata)
    if values.shape != expected_shape or values.dtype != np.float64:
        raise ValueError("Selection cache shape or precision mismatch")
    np.testing.assert_array_equal(chains, [r["chain"] for r in case["residues"]])
    reference, _ = load_ca_multimodel_pdb_minimal(
        case["reference_source"],
        expected_frames=1,
        expected_residues=values.shape[1],
    )
    np.testing.assert_array_equal(values[0], reference[0])
    print(f"{case['case_id']}: selection cache verified {values.shape}", flush=True)
    return values, chains, metadata


def load_saved(case: dict) -> dict:
    source = Path(case["source_case_directory"])
    with (source / "residue_results.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    saved = {}
    for scope in ("full", "train"):
        subset = sorted(
            (r for r in rows if r["scope"] == scope), key=lambda r: int(r["atom_index"])
        )
        if [int(r["atom_index"]) for r in subset] != list(range(len(case["residues"]))):
            raise ValueError("Saved residue index coverage mismatch")
        weights = np.asarray([float(r["MDCoreAlign_weight"]) for r in subset])
        mlf = np.asarray([bool(int(r["MDLovoFit_core"])) for r in subset])
        mca = np.asarray([bool(int(r["MDCoreAlign_matched_k"])) for r in subset])
        k = int(mlf.sum())
        np.testing.assert_array_equal(mca, stable_ranks(weights) <= k)
        export = source / (
            "mdlovofit_train"
            if scope == "train" and (source / "mdlovofit_train").exists()
            else "mdlovofit_full"
        )
        with np.load(export / "membership.npz") as masks:
            if not np.all(masks["masks"] == masks["masks"][0]):
                raise ValueError("The saved MDLovoFit masks are not a fixed core")
            np.testing.assert_array_equal(mlf, masks["masks"][0])
        group = np.where(
            mca & mlf, "shared", np.where(mca, "MCA_only", np.where(mlf, "MLF_only", "neither"))
        )
        saved[scope] = {
            "weights": weights,
            "mca": mca,
            "mlf": mlf,
            "rank": stable_ranks(weights),
            "k": k,
            "group": group,
        }
    return saved


def provenance(case: dict, config: dict, scope: str, k: int) -> dict:
    index_file = f"cases/{case['case_id']}/sampling/frame_indices.json"
    descriptive = scope == "full" and case["case_id"] == "ADK"
    return {
        "case": case["case_id"],
        "selection_source": case["selection_input_pdb"],
        "selection_scope": scope,
        "selection_frame_indices": f"{index_file}#{scope}_selection_source_indices",
        "evaluation_source": case["selection_input_pdb"]
        if descriptive
        else case["evaluation_input_pdb"],
        "evaluation_frame_indices": f"{index_file}#{scope}_evaluation_source_indices",
        "core_definition": "matched-k; descending frozen weight, ascending input-index ties",
        "k": k,
        "reference_source": case["reference_source"],
        "run_id": config["run_id"],
    }


def identity(residue: dict, index: int, case: dict, saved: dict) -> dict:
    return {
        "atom_index": index,
        "chain": residue["chain"],
        "source_segid": residue.get("source_segid", ""),
        "source_resid": residue["source_resid"],
        "icode": residue.get("icode", ""),
        "resname": residue["resname"],
        "literature_resid": residue.get("literature_resid", ""),
        "mapping_status": residue.get("mapping_status", ""),
        "literature_regions": ";".join(
            name for name, mask in case["regions"].items() if mask[index]
        ),
        "old_full_group": saved["full"]["group"][index],
        "old_train_group": saved["train"]["group"][index],
        "tracked_old_full_disagreement": bool(
            saved["full"]["mca"][index] != saved["full"]["mlf"][index]
        ),
        "tracked_old_train_disagreement": bool(
            saved["train"]["mca"][index] != saved["train"]["mlf"][index]
        ),
    }


def geometry(mask: np.ndarray, pairs: dict) -> tuple[float | None, int]:
    valid = mask[pairs["left"]] & mask[pairs["right"]]
    return (
        float(np.sqrt(np.mean(pairs["variance_A2"][valid]))) if valid.any() else None,
        int(valid.sum()),
    )


def run_case(case: dict, config: dict) -> tuple[list, list, list, dict]:
    full, chains, cache_meta = prepare_selection_cache(case, config)
    source = Path(case["source_case_directory"])
    destination = Path(config["run_directory"]) / "cases" / case["case_id"] / "sampling"
    destination.mkdir(parents=True, exist_ok=True)
    train_indices = np.asarray(case["train_indices"], dtype=np.int64)
    if train_indices[0] != 0 or np.any(np.diff(train_indices) <= 0):
        raise ValueError("Unexpected training reference/order")
    if case["case_id"] == "ADK" and np.intersect1d(train_indices, case["test_indices"]).size:
        raise ValueError("ADK training/evaluation frames overlap")
    train = full[train_indices]
    reference = full[0]
    saved = load_saved(case)
    old_summary = json.loads((source / "case_summary.json").read_text())
    count = min(len(train), int(old_summary["production_metadata"]["representative_count"]))
    if count != 256 or old_summary["production_metadata"]["update_rounds"] != 4:
        raise ValueError("Unexpected saved production budget/round count")
    indices_record = {
        "case": case["case_id"],
        "run_id": config["run_id"],
        "rule": SAMPLING_RULE,
        "full_selection_source_indices": list(range(len(full))),
        "train_selection_source_indices": train_indices.tolist(),
        "full_evaluation_source_indices": list(range(len(full)))
        if case["case_id"] == "ADK"
        else case["test_indices"],
        "train_evaluation_source_indices": case["test_indices"],
        "reference_source": case["reference_source"],
        "reference_source_index": 0,
        "replicate_indices": [],
        "original_representative_indices": {},
    }
    diagnostics_rows: list[dict] = []
    replay_results = {}
    # ADK original full-trajectory differences and train differences need distinct
    # replay diagnostics. For polymerase both saved scopes are the same R1 set.
    for scope in ("full", "train"):
        if scope == "train" and len(train) == len(full):
            np.testing.assert_array_equal(saved["full"]["weights"], saved["train"]["weights"])
            np.testing.assert_array_equal(saved["full"]["mca"], saved["train"]["mca"])
            np.testing.assert_array_equal(saved["full"]["mlf"], saved["train"]["mlf"])
            replay_results[scope] = dict(replay_results["full"], reused_scope="full")
            indices_record["original_representative_indices"][scope] = indices_record[
                "original_representative_indices"
            ]["full"]
            continue
        values = full if scope == "full" else train
        original_indices = window_refine.temporally_stratified_indices(len(values), count)
        indices_record["original_representative_indices"][scope] = original_indices.tolist()
        weights, trace, rounds = estimate_with_indices(
            values, reference, original_indices, chains, diagnostics=True
        )
        error = float(np.max(np.abs(weights - saved[scope]["weights"])))
        # Decimal TSV round-trips float64 exactly; require exact reproduction.
        np.testing.assert_array_equal(weights, saved[scope]["weights"])
        replay_results[scope] = {
            "exact_float64_reproduction": True,
            "max_abs_error": error,
            "representative_count": len(original_indices),
        }
        arrays = {
            "final_frozen_weights": weights,
            "original_representative_indices": original_indices,
            **trace,
        }
        for record in rounds:
            r = record["round"]
            arrays.update(
                {f"round_{r}_{key}": value for key, value in record.items() if key != "round"}
            )
            distances = record["reference_distances_A"]
            for i, residue in enumerate(case["residues"]):
                diagnostics_rows.append(
                    {
                        **provenance(case, config, scope, saved[scope]["k"]),
                        **identity(residue, i, case, saved),
                        "round": r,
                        "representative_count": count,
                        "weight_before_update": float(record["weights_before_update"][i]),
                        "pre_update_reference_distance_mean_A": float(distances[:, i].mean()),
                        "pre_update_reference_RMS_displacement_A": float(
                            np.sqrt(np.mean(distances[:, i] ** 2))
                        ),
                        "pre_update_reference_distance_median_A": float(np.median(distances[:, i])),
                        "robust_frame_weight_mean_before_continuity": float(
                            record["robust_frame_weights"][:, i].mean()
                        ),
                        "frame_weight_mean_after_continuity_before_global_normalization": float(
                            record["continuity_frame_weights"][:, i].mean()
                        ),
                        "normalized_shared_weight_after_update": float(
                            record["normalized_weights_after_update"][i]
                        ),
                        "final_frozen_weight": float(weights[i]),
                        "replay_exact": True,
                        "intermediate_semantics": "Distances from fits using weights BEFORE this update to fixed reference; not RMSF and not residuals refitted with final weights.",
                        "capture_source": "original bounded_global_weight_refinement trace and unchanged continuity-call wrapper",
                    }
                )
        np.savez_compressed(destination / f"original_{scope}_diagnostic_arrays.npz", **arrays)
        print(
            f"{case['case_id']}: {scope} original representatives reproduce saved weights exactly",
            flush=True,
        )
    baseline = saved["train"]
    baseline_mask, baseline_weights = baseline["mca"], baseline["weights"]
    k = baseline["k"]
    replicate_weights, replicate_ranks, replicate_masks = [], [], []
    summary_rows = []
    with np.load(source / "heldout_pair_variances.npz") as archive:
        pairs = {key: archive[key] for key in ("left", "right", "variance_A2")}
    for seed in SEEDS:
        indices = stratified_sample(len(train), count, seed)
        indices_record["replicate_indices"].append(
            {
                "seed": seed,
                "local_training_indices": indices.tolist(),
                "source_frame_indices": train_indices[indices].tolist(),
            }
        )
        weights, _, _ = estimate_with_indices(train, reference, indices, chains)
        ranks = stable_ranks(weights)
        mask = ranks <= k
        shared = int(np.count_nonzero(mask & baseline_mask))
        union = int(np.count_nonzero(mask | baseline_mask))
        score, pair_count = geometry(mask, pairs)
        summary_rows.append(
            {
                **provenance(case, config, "train", k),
                "seed": seed,
                "representative_count": count,
                "representative_indices": f"cases/{case['case_id']}/sampling/frame_indices.json#replicate_indices/seed={seed}",
                "jaccard": shared / union,
                "dice": shared / k,
                "shared_with_training_baseline": shared,
                "changed_membership_count": int(np.count_nonzero(mask ^ baseline_mask)),
                "rank_spearman": float(spearmanr(baseline["rank"], ranks).statistic),
                "unit_mass_weight_L1": float(
                    np.sum(
                        np.abs(weights / weights.sum() - baseline_weights / baseline_weights.sum())
                    )
                ),
                "mean_abs_weight_difference": float(np.mean(np.abs(weights - baseline_weights))),
                "max_abs_weight_difference": float(np.max(np.abs(weights - baseline_weights))),
                "G_A": score,
                "G_pair_count": pair_count,
                "baseline_G_A": geometry(baseline_mask, pairs)[0],
                "original_replay_exact": True,
                "interpretation": "Five fixed representative draws; empirical sensitivity, not calibrated probabilities or biological replicates. No MDLovoFit robustness comparison.",
            }
        )
        replicate_weights.append(weights)
        replicate_ranks.append(ranks)
        replicate_masks.append(mask)
        print(f"{case['case_id']}: seed {seed} complete, J={shared / union:.6f}", flush=True)
    all_weights = np.stack(replicate_weights)
    all_ranks = np.stack(replicate_ranks)
    all_masks = np.stack(replicate_masks)
    inclusion_counts = all_masks.sum(axis=0)
    residue_rows = []
    for replicate, seed in enumerate(SEEDS):
        for i, residue in enumerate(case["residues"]):
            residue_rows.append(
                {
                    **provenance(case, config, "train", k),
                    **identity(residue, i, case, saved),
                    "seed": seed,
                    "baseline_weight": float(baseline_weights[i]),
                    "baseline_rank": int(baseline["rank"][i]),
                    "baseline_included": bool(baseline_mask[i]),
                    "weight": float(all_weights[replicate, i]),
                    "rank": int(all_ranks[replicate, i]),
                    "included": bool(all_masks[replicate, i]),
                    "inclusion_count_5": int(inclusion_counts[i]),
                    "rank_min_5": int(all_ranks[:, i].min()),
                    "rank_max_5": int(all_ranks[:, i].max()),
                    "weight_min_5": float(all_weights[:, i].min()),
                    "weight_max_5": float(all_weights[:, i].max()),
                }
            )
    np.savez_compressed(
        destination / "five_seed_weights_ranks_masks.npz",
        seeds=np.asarray(SEEDS),
        weights=all_weights,
        ranks=all_ranks,
        masks=all_masks,
        baseline_weights=baseline_weights,
        baseline_mask=baseline_mask,
        inclusion_counts=inclusion_counts,
    )
    write_json(destination / "frame_indices.json", indices_record)
    audit = {
        "case": case["case_id"],
        "cache": cache_meta,
        "replay": replay_results,
        "extra_estimate_count": 5,
        "fixed_seeds": list(SEEDS),
        "budget": count,
        "rounds": 4,
        "delta_A": 1.0,
        "sequence_continuity_strength": 0.05,
        "precision": "float64",
        "reference_fixed": True,
        "original_input_reference_equal": True,
        "full_train_identical": bool(np.array_equal(saved["full"]["weights"], baseline_weights)),
        "interpretation": "Only representative indices change; the original estimator sees the same reference, unit initial weights, scientific configuration and chain order. No all-frame fit is needed for frozen-weight sensitivity.",
    }
    write_json(destination / "replay_validation.json", audit)
    return summary_rows, residue_rows, diagnostics_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if tuple(config["seeds"]) != SEEDS or [c["case_id"] for c in config["cases"]] != [
        "ADK",
        "3nci_A",
    ]:
        raise ValueError("Frozen case/seed contract mismatch")
    if config["max_extra_estimates_per_case"] != 5 or config["production_mode"] != "throughput":
        raise ValueError("Frozen estimator contract mismatch")
    run = Path(config["run_directory"])
    protocol = {
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "sampling_rule": SAMPLING_RULE,
        "seeds": list(SEEDS),
        "representative_count": 256,
        "update_rounds": 4,
        "estimator_source": str(
            Path(inspect.getfile(window_refine.bounded_global_weight_refinement)).resolve()
        ),
        "estimator_source_sha256": sha256(
            Path(inspect.getfile(window_refine.bounded_global_weight_refinement))
        ),
        "estimator_source_line": inspect.getsourcelines(
            window_refine.bounded_global_weight_refinement
        )[1],
        "intermediate_capture": "Read-only arrays copied at original continuity function call; original called unchanged, wrapper restored in finally. Existing estimator trace supplies shared-weight history.",
        "residual_semantics": "Representative-to-fixed-reference displacement under pre-update weights, not RMSF; final-round residuals precede final weights.",
        "weight_L1_definition": "sum_i abs(w_seed_i/sum(w_seed) - w_baseline_i/sum(w_baseline))",
        "rank_spearman_definition": "Spearman correlation of deterministic unique ranks; weight ties use ascending input index.",
        "timing_role": "none; no formal timing, speedup, exact, or MDLovoFit rerun",
    }
    write_json(run / "sampling_protocol.json", protocol)
    summary, residues, diagnostics, audits = [], [], [], []
    with threadpool_limits(limits=1):
        protocol["numerical_threadpools"] = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in protocol["numerical_threadpools"]):
            raise RuntimeError("Numerical thread contract failed")
        write_json(run / "sampling_protocol.json", protocol)
        for case in config["cases"]:
            if args.prepare_only:
                prepare_selection_cache(case, config)
                continue
            s, r, d, audit = run_case(case, config)
            summary.extend(s)
            residues.extend(r)
            diagnostics.extend(d)
            audits.append(audit)
    if not args.prepare_only:
        if len(summary) != 10 or len(residues) != 5 * (214 + 903):
            raise AssertionError("Five-seed/all-residue output coverage failed")
        write_tsv(run / "sensitivity_summary.tsv", summary)
        write_tsv(run / "sensitivity_by_residue.tsv", residues)
        write_tsv(run / "diagnostics.tsv", diagnostics)
        write_json(
            run / "sampling_validation.json",
            {
                "status": "analysis_completed",
                "audits": audits,
                "extra_estimates_total": 10,
                "scientific_success_threshold": None,
            },
        )


if __name__ == "__main__":
    main()
