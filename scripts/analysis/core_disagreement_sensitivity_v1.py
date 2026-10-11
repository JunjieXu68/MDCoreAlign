"""Residue-level set differences and geometry from frozen biological outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.analysis.biological_core_validation_v1 import geometry_score, pair_variances
from scripts.atlas_strict_single_thread_cpu_v1.common import sha256, write_json, write_tsv
from scripts.atlas_strict_single_thread_cpu_v1.extract_cores import overlap, top_k

GROUPS = ("shared", "MCA_only", "MLF_only", "neither")


def read_tsv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def stable_ranks(weights: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(weights)), -weights))
    ranks = np.empty(len(weights), dtype=int)
    ranks[order] = np.arange(1, len(weights) + 1)
    return ranks


def classify(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a & b, "shared", np.where(a, "MCA_only", np.where(b, "MLF_only", "neither")))


def contexts(case: dict) -> list[tuple[str, str, str, str]]:
    cid = case["case_id"]
    if cid == "ADK":
        return [
            ("ADK_full_descriptive", "full", "ADK_full", "ADK_full"),
            ("ADK_train_heldout", "train", "ADK_train", "ADK_test"),
        ]
    return [("3nci_A_R1_to_R2", "full", "3nci_A_R1", "3nci_A_R2")]


def provenance(
    config: dict, case: dict, context: str, selection_key: str, evaluation_key: str, k: int
) -> dict:
    return {
        "case": case["case_id"],
        "context": context,
        "selection_source": case["selection_input_pdb"],
        "selection_frame_indices": f"frame_index_sets.json#/{selection_key}",
        "evaluation_source": case["evaluation_input_pdb"]
        if evaluation_key != "ADK_full"
        else case["selection_input_pdb"],
        "evaluation_frame_indices": f"frame_index_sets.json#/{evaluation_key}",
        "core_definition": config["primary_core_definition"],
        "k": k,
        "reference_source": case["reference_source"],
        "run_id": config["run_id"],
    }


def variance_matrix(pairs: dict, residues: list[dict]) -> np.ndarray:
    """Symmetric eligible-pair matrix; invalid/self pairs remain NaN."""
    n = len(residues)
    left, right = pairs["left"], pairs["right"]
    assert len(set(zip(left.tolist(), right.tolist(), strict=True))) == len(left)
    assert np.all(left < right) and np.all(left >= 0) and np.all(right < n)
    chain = np.array([r["chain"] for r in residues])
    sequence = np.array([r["sequence_position"] for r in residues])
    assert not np.any(
        (chain[left] == chain[right]) & (np.abs(sequence[left] - sequence[right]) <= 3)
    )
    assert np.all(np.isfinite(pairs["variance_A2"])) and np.all(pairs["variance_A2"] >= 0)
    values = np.full((n, n), np.nan)
    values[left, right] = pairs["variance_A2"]
    values[right, left] = pairs["variance_A2"]
    return values


def relative_geometry(matrix: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = matrix[:, reference]
    counts = np.sum(np.isfinite(selected), axis=1)
    means = np.divide(
        np.nansum(selected, axis=1), counts, out=np.full(len(matrix), np.nan), where=counts > 0
    )
    return np.sqrt(means), counts


def decompose(pairs: dict, common: np.ndarray, unique: np.ndarray) -> list[dict]:
    left, right = pairs["left"], pairs["right"]
    selected = common | unique
    masks = {
        "C-C": common[left] & common[right],
        "C-U": (common[left] & unique[right]) | (unique[left] & common[right]),
        "U-U": unique[left] & unique[right],
    }
    all_pairs = selected[left] & selected[right]
    assert np.array_equal(sum(mask.astype(int) for mask in masks.values()), all_pairs.astype(int))
    total_n = int(all_pairs.sum())
    total_q = float(pairs["variance_A2"][all_pairs].sum())
    result = []
    for name, mask in {**masks, "total": all_pairs}.items():
        n = int(mask.sum())
        q = float(pairs["variance_A2"][mask].sum())
        result.append(
            {
                "pair_class": name,
                "pair_count": n,
                "variance_sum_A2": q,
                "mean_variance_A2": q / n if n else None,
                "G2_contribution_A2": q / total_n if total_n else None,
                "fraction_of_total_variance_sum": q / total_q if total_q else None,
                "total_pair_count": total_n,
                "reconstructed_G_A": float(np.sqrt(total_q / total_n)) if total_n else None,
            }
        )
    assert sum(r["pair_count"] for r in result[:3]) == total_n
    assert np.isclose(
        sum(r["variance_sum_A2"] for r in result[:3]), total_q, rtol=1e-14, atol=1e-12
    )
    return result


def freeze_sets(run: Path) -> None:
    config = json.loads((run / "analysis_config.json").read_text())
    frame_sets = {}
    memberships = []
    summaries = []
    cutoffs = []
    rounding = []
    segments = []
    for case in config["cases"]:
        cid = case["case_id"]
        p = Path(case["source_case_directory"])
        old_summary = json.loads((p / "case_summary.json").read_text())
        assert old_summary["configuration_sha256"] == config["source_configuration_sha256"]
        full_n = old_summary["n_frames"]
        n = len(case["residues"])
        if cid == "ADK":
            frame_sets.update(
                {
                    "ADK_full": list(range(full_n)),
                    "ADK_train": case["train_indices"],
                    "ADK_test": case["test_indices"],
                }
            )
            assert not set(case["train_indices"]) & set(case["test_indices"])
        else:
            frame_sets.update(
                {"3nci_A_R1": case["train_indices"], "3nci_A_R2": case["test_indices"]}
            )
            assert Path(case["coords_path"]).resolve() != Path(case["test_coords_path"]).resolve()
        source = read_tsv(p / "residue_results.tsv")
        records = {scope: [r for r in source if r["scope"] == scope] for scope in ["full", "train"]}
        if cid == "3nci_A":
            for key in [
                "MDCoreAlign_weight",
                "MDCoreAlign_native60",
                "MDCoreAlign_matched_k",
                "MDLovoFit_core",
                "MDLovoFit_frequency",
            ]:
                assert [r[key] for r in records["full"]] == [r[key] for r in records["train"]], key
        for context, scope, selection_key, evaluation_key in contexts(case):
            rows = records[scope]
            assert len(rows) == n and [int(r["atom_index"]) for r in rows] == list(range(n))
            weights = np.array([float(r["MDCoreAlign_weight"]) for r in rows])
            native = np.array([int(r["MDCoreAlign_native60"]) for r in rows], dtype=bool)
            ext_dir = "mdlovofit_train" if scope == "train" and cid == "ADK" else "mdlovofit_full"
            ext = np.load(p / ext_dir / "membership.npz")
            masks = ext["masks"]
            assert masks.shape == (len(frame_sets[selection_key]), n)
            assert np.all(masks == masks[0])
            assert np.array_equal(ext["core"], masks[0]) and np.array_equal(
                ext["frequency"], masks.mean(axis=0)
            )
            mlf = masks[0]
            k = int(mlf.sum())
            mca = top_k(weights, k)
            rank = stable_ranks(weights)
            assert np.array_equal(
                mca, np.array([int(r["MDCoreAlign_matched_k"]) for r in rows], dtype=bool)
            )
            assert np.array_equal(
                mlf, np.array([int(r["MDLovoFit_core"]) for r in rows], dtype=bool)
            )
            assert np.array_equal(native, top_k(weights, int(np.floor(0.6 * n + 0.5))))
            group = classify(mca, mlf)
            common = mca & mlf
            cutoff = float(weights[rank == k][0])
            next_weight = float(weights[rank == k + 1][0])
            gap = cutoff - next_weight
            info = provenance(config, case, context, selection_key, evaluation_key, k)
            counted = {f"n_{g}": int(np.sum(group == g)) for g in GROUPS}
            agreement = overlap(mca, mlf)
            assert sum(counted.values()) == n and counted["n_MCA_only"] == counted["n_MLF_only"]
            assert np.isclose(agreement["dice"], int(common.sum()) / k)
            summary = {
                **info,
                "source_scope": scope,
                "N": n,
                "native_MCA_k": int(native.sum()),
                **counted,
                **agreement,
                "nonshared_fraction_each_core": int((mca & ~mlf).sum()) / k,
                "cutoff_weight": cutoff,
                "next_weight": next_weight,
                "cutoff_gap": gap,
                "MLF_only_min_weight": float(weights[group == "MLF_only"].min()),
                "MLF_only_max_weight": float(weights[group == "MLF_only"].max()),
                "MLF_only_zero_weight_count": int(np.sum(weights[group == "MLF_only"] == 0)),
                "MLF_only_weight_share": float(weights[group == "MLF_only"].sum() / weights.sum()),
                "MLF_constant_membership_verified_all_frames": True,
                "R1_full_train_identical": True if cid == "3nci_A" else "not_applicable",
            }
            summaries.append(summary)
            regions = {a: np.asarray(b, dtype=bool) for a, b in case["regions"].items()}
            if cid == "ADK":
                labels = np.array([r["literature_regions"] for r in rows])
                edges = np.where(labels[1:] != labels[:-1])[0]
            else:
                assert int((regions["framework_A"] & regions["framework_B"]).sum()) == 29
                edges = np.unique(
                    np.concatenate(
                        [np.where(mask[1:] != mask[:-1])[0] for mask in regions.values()]
                    )
                )
            boundary_indices = np.unique(np.r_[edges, edges + 1])
            m = max(1, int(np.floor(0.02 * n + 0.5)))
            deltas = list(dict.fromkeys([0, -1, 1, -m, m]))
            new_ks = [k + d for d in deltas if 1 <= k + d <= n]
            per_context = []
            for i, r in enumerate(rows):
                atom = case["residues"][i]
                fa = bool(regions.get("framework_A", np.zeros(n, bool))[i])
                fb = bool(regions.get("framework_B", np.zeros(n, bool))[i])
                category = (
                    (
                        "both_frameworks"
                        if fa and fb
                        else "framework_A_only"
                        if fa
                        else "framework_B_only"
                        if fb
                        else "neither_framework"
                    )
                    if cid == "3nci_A"
                    else "not_applicable"
                )
                boundary_distance = (
                    int(np.min(np.abs(boundary_indices - i))) if len(boundary_indices) else None
                )
                row = {
                    **info,
                    "source_scope": scope,
                    **atom,
                    "group": str(group[i]),
                    "frozen_weight": float(weights[i]),
                    "weight_is_zero": bool(weights[i] == 0),
                    "rank": int(rank[i]),
                    "rank_minus_k": int(rank[i] - k),
                    "cutoff_weight": cutoff,
                    "weight_minus_cutoff": float(weights[i] - cutoff),
                    "cutoff_gap": gap,
                    "exact_weight_tie_size": int(np.sum(weights == weights[i])),
                    "exact_tie_at_cutoff": bool(
                        weights[i] == cutoff and np.sum(weights == cutoff) > 1
                    ),
                    "tie_rule": "descending exact frozen_weight; ascending original input atom_index",
                    "MLF_frequency": float(ext["frequency"][i]),
                    "MCA_matched": bool(mca[i]),
                    "MCA_native60": bool(native[i]),
                    "MLF_core": bool(mlf[i]),
                    "literature_regions": r["literature_regions"],
                    "framework_A": fa,
                    "framework_B": fb,
                    "framework_category": category,
                    "sequence_distance_to_annotation_boundary": boundary_distance,
                    "annotation_boundary_kind": "domain boundary"
                    if cid == "ADK"
                    else "framework segment membership boundary",
                    "within_3_residues_of_annotation_boundary": boundary_distance <= 3
                    if boundary_distance is not None
                    else None,
                    "within_tested_cutoff_band": bool(min(new_ks) < rank[i] <= max(new_ks)),
                    "rank_improvement_needed_for_inclusion": max(0, int(rank[i] - k)),
                    "cutoff_delta_for_membership_toggle": int(
                        rank[i] - 1 - k if mca[i] else rank[i] - k
                    ),
                }
                memberships.append(row)
                per_context.append(row)
                if group[i] in ["MCA_only", "MLF_only"]:
                    for delta in deltas:
                        new_k = k + delta
                        if 1 <= new_k <= n:
                            cutoffs.append(
                                {
                                    **row,
                                    "delta_k": delta,
                                    "new_k": new_k,
                                    "included_at_new_k": bool(rank[i] <= new_k),
                                    "comparison_definition": "matched-k only at delta=0; fixed MLF reference for other k",
                                    "fit_weights_unchanged_by_reporting": True,
                                }
                            )
                if cid == "3nci_A" and rank[i] == 542:
                    rounding.append(
                        {
                            **row,
                            "native_k": int(native.sum()),
                            "matched_k": k,
                            "only_native_not_matched": bool(native[i] and not mca[i]),
                        }
                    )
            for g in ["MCA_only", "MLF_only", "all_disagreement"]:
                chosen = [
                    r
                    for r in per_context
                    if r["group"] == g
                    or (g == "all_disagreement" and r["group"] in ["MCA_only", "MLF_only"])
                ]
                blocks = []
                for r in chosen:
                    if (
                        not blocks
                        or r["chain"] != blocks[-1][-1]["chain"]
                        or r["sequence_position"] != blocks[-1][-1]["sequence_position"] + 1
                    ):
                        blocks.append([])
                    blocks[-1].append(r)
                for block in blocks:
                    segments.append(
                        {
                            **info,
                            "group": g,
                            "start_resid": block[0]["source_resid"],
                            "end_resid": block[-1]["source_resid"],
                            "n_residues": len(block),
                            "members": ";".join(
                                f"{r['chain']}:{r['resname']}{r['source_resid']}{r['icode']}"
                                for r in block
                            ),
                            "literature_labels": ";".join(
                                sorted(set(r["literature_regions"] for r in block))
                            ),
                        }
                    )
            np.savez_compressed(
                run / "cases" / cid / f"{context}_sets.npz",
                weights=weights,
                rank=rank,
                MCA=mca,
                MLF=mlf,
                shared=common,
                group=group,
                native=native,
            )
    write_json(run / "frame_index_sets.json", frame_sets)
    for name, data in [
        ("residue_membership.tsv", memberships),
        (
            "disagreement_residues.tsv",
            [r for r in memberships if r["group"] in ["MCA_only", "MLF_only"]],
        ),
        ("core_overlap_summary.tsv", summaries),
        ("cutoff_sensitivity.tsv", cutoffs),
        ("native_rounding.tsv", rounding),
        ("disagreement_segments.tsv", segments),
    ]:
        write_tsv(run / name, data)
    write_json(
        run / "validation/set_reconstruction.json",
        {
            "status": "integer sets and all-frame native occupancies verified",
            "source_manifest_sha256": sha256(Path(config["source_run"]) / "run_manifest.json"),
            "contexts": [r["context"] for r in summaries],
        },
    )
    print("Set reconstruction complete", flush=True)


def analyze_geometry(run: Path) -> None:
    config = json.loads((run / "analysis_config.json").read_text())
    memberships = read_tsv(run / "residue_membership.tsv")
    geometries = []
    groups = []
    decompositions = []
    checks = []
    for case in config["cases"]:
        cid = case["case_id"]
        p = Path(case["source_case_directory"])
        for context, _scope, selection_key, evaluation_key in contexts(case):
            rows = [r for r in memberships if r["context"] == context]
            sets = np.load(run / "cases" / cid / f"{context}_sets.npz")
            if context == "ADK_full_descriptive":
                coords = np.load(case["selection_coords_cache"], mmap_mode="r")
                pairs = pair_variances(coords, case["residues"])
                np.savez_compressed(
                    run / "cases" / cid / "full_descriptive_pair_variances.npz", **pairs
                )
                cache_path = run / "cases" / cid / "full_descriptive_pair_variances.npz"
            else:
                cache_path = p / "heldout_pair_variances.npz"
                pairs = dict(np.load(cache_path))
                assert int(pairs["n_frames"]) == len(case["test_indices"])
            matrix = variance_matrix(pairs, case["residues"])
            references = {"common": sets["shared"]}
            if cid == "3nci_A":
                references.update(
                    {
                        name: np.asarray(case["regions"][name], dtype=bool)
                        for name in ["framework_A", "framework_B"]
                    }
                )
            h_values = {name: relative_geometry(matrix, mask) for name, mask in references.items()}
            info = provenance(
                config, case, context, selection_key, evaluation_key, int(sets["MCA"].sum())
            )
            for i, r in enumerate(rows):
                extra = {}
                for name, (h, partners) in h_values.items():
                    extra[f"h_{name}_A"] = float(h[i]) if np.isfinite(h[i]) else None
                    extra[f"{name}_partner_count"] = int(partners[i])
                geometries.append(
                    {
                        **r,
                        **extra,
                        "variance_cache": str(cache_path),
                        "geometry_role": "full-source descriptive; not heldout"
                        if context == "ADK_full_descriptive"
                        else "frozen training sets evaluated only on heldout source",
                        "variance_ddof": 0,
                    }
                )
            for reference, (h, _partners) in h_values.items():
                for group in GROUPS:
                    values = h[sets["group"] == group]
                    valid = values[np.isfinite(values)]
                    groups.append(
                        {
                            **info,
                            "group": group,
                            "geometry_reference": reference,
                            "n_residues": len(values),
                            "n_evaluable": len(valid),
                            "median_h_A": float(np.median(valid)) if len(valid) else None,
                            "min_h_A": float(valid.min()) if len(valid) else None,
                            "max_h_A": float(valid.max()) if len(valid) else None,
                            "mean_h_A": float(valid.mean()) if len(valid) else None,
                        }
                    )
            for method in ["MCA", "MLF"]:
                parts = decompose(pairs, sets["shared"], sets[method] & ~sets["shared"])
                for row in parts:
                    decompositions.append(
                        {
                            **info,
                            "method": method,
                            **row,
                            "variance_cache": str(cache_path),
                            "variance_ddof": 0,
                        }
                    )
                direct, count = geometry_score(sets[method], pairs)
                assert count == parts[-1]["pair_count"]
                assert np.isclose(direct, parts[-1]["reconstructed_G_A"], atol=1e-12, rtol=1e-12)
                prior = None
                if context != "ADK_full_descriptive":
                    lookup = "MDCoreAlign_matched_k" if method == "MCA" else "MDLovoFit"
                    prior = float(
                        next(
                            r for r in read_tsv(p / "geometry_metrics.tsv") if r["method"] == lookup
                        )["G_A"]
                    )
                    assert np.isclose(prior, direct, atol=1e-12, rtol=1e-12)
                checks.append(
                    {
                        **info,
                        "method": method,
                        "G_A": direct,
                        "prior_G_A": prior,
                        "G_absolute_difference_A": abs(direct - prior)
                        if prior is not None
                        else None,
                        "pair_count": count,
                    }
                )
    for name, data in [
        ("geometry_by_residue.tsv", geometries),
        ("geometry_group_summary.tsv", groups),
        ("geometry_pair_decomposition.tsv", decompositions),
        ("geometry_reconstruction_checks.tsv", checks),
    ]:
        write_tsv(run / name, data)
    print("Geometry analysis complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["sets", "geometry", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ["sets", "all"]:
        freeze_sets(args.run.resolve())
    if args.stage in ["geometry", "all"]:
        analyze_geometry(args.run.resolve())
