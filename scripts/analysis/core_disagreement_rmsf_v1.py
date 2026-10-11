"""Native-profile differences and controlled held-out binary-set RMSF diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mdcorealign.benchmark_minimal import load_ca_multimodel_pdb_minimal
from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights
from scripts.atlas_strict_single_thread_cpu_v1.common import sha256, write_json, write_tsv


def sha256_file(path):
    return sha256(Path(path))


def index_ranges(indices):
    """Compact, lossless zero-based half-open ranges for provenance columns."""
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return "[]"
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 1) + 1]
    stops = np.r_[starts[1:], len(values)]
    return json.dumps(
        [[int(values[a]), int(values[b - 1]) + 1] for a, b in zip(starts, stops, strict=True)]
    )


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def source_sets(case, scope):
    rows = [
        r
        for r in read_rows(Path(case["source_case_directory"]) / "residue_results.tsv")
        if r["scope"] == scope
    ]
    if [int(r["atom_index"]) for r in rows] != list(range(len(case["residues"]))):
        raise ValueError("source residue rows must preserve complete input order")
    for row, residue in zip(rows, case["residues"], strict=True):
        if any(
            str(row[key]) != str(residue[key])
            for key in ["chain", "source_resid", "icode", "resname"]
        ):
            raise ValueError("source residue identities disagree with frozen configuration")
    for key in ["MDCoreAlign_matched_k", "MDLovoFit_core"]:
        if any(r[key] not in {"0", "1"} for r in rows):
            raise ValueError("source core columns must contain saved binary 0/1 values")
    mca = np.array([r["MDCoreAlign_matched_k"] == "1" for r in rows])
    mlf = np.array([r["MDLovoFit_core"] == "1" for r in rows])
    if mca.sum() != mlf.sum() or mca.sum() == 0:
        raise ValueError("source matched-k masks do not have the same positive cardinality")
    return {
        "C": mca & mlf,
        "U_MCA": mca & ~mlf,
        "U_MLF": mlf & ~mca,
        "C_plus_U_MCA": mca,
        "C_plus_U_MLF": mlf,
    }


def build_evaluation_cache(case):
    """Parse R2 once; ADK reuses the separately owned full selection cache."""
    target = Path(case["evaluation_coords_cache"])
    sidecar = target.with_suffix(".json")
    n = len(case["residues"])
    indices = np.asarray(case["test_indices"], dtype=np.int64)
    expected = (len(indices), n, 3)
    if target.exists():
        audit = json.loads(sidecar.read_text())
        coords = np.load(target, mmap_mode="r")
        if coords.shape != expected or coords.dtype != np.float64:
            raise ValueError("evaluation cache shape or dtype mismatch")
        if (
            audit["sha256"] != sha256_file(target)
            or audit["evaluation_frame_indices"] != index_ranges(indices)
            or audit["evaluation_source"] != case["evaluation_input_pdb"]
            or audit["case"] != case["case_id"]
        ):
            raise ValueError("evaluation cache provenance mismatch")
        return coords
    target.parent.mkdir(parents=True, exist_ok=True)
    if case["case_id"] == "ADK":
        source = Path(case["selection_coords_cache"])
        full = np.load(source, mmap_mode="r")
        if full.shape != (len(case["train_indices"]) + len(indices), n, 3):
            raise ValueError("ADK selection cache must contain the full original trajectory")
        coords = full[indices]
        source_kind = "exact slice of shared full PDB-parsed selection cache"
    else:
        source = Path(case["evaluation_input_pdb"])
        coords, chains = load_ca_multimodel_pdb_minimal(
            source, expected_frames=len(indices), expected_residues=n
        )
        if list(chains) != [r["chain"] for r in case["residues"]]:
            raise ValueError("R2 chain order differs from frozen residue mapping")
        source_kind = "existing minimal CA PDB loader; original 0.001 Angstrom precision"
    if coords.shape != expected or not np.isfinite(coords).all():
        raise ValueError("invalid evaluation coordinates")
    temporary = target.with_suffix(".partial.npy")
    np.save(temporary, coords)
    temporary.replace(target)
    write_json(
        sidecar,
        {
            "case": case["case_id"],
            "source": str(source),
            "source_sha256": sha256_file(source),
            "source_kind": source_kind,
            "evaluation_source": case["evaluation_input_pdb"],
            "evaluation_frame_indices": index_ranges(indices),
            "frame_index_encoding": "zero-based half-open ranges",
            "shape": list(expected),
            "dtype": "float64",
            "sha256": sha256_file(target),
        },
    )
    return np.load(target, mmap_mode="r")


def binary_fit_rmsf(coords, reference, mask, *, chunk_size=256, rank_rtol=1e-10):
    """Existing streamed fitting engine, preceded by a numerical rank audit."""
    coords = np.asarray(coords)
    reference = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if coords.ndim != 3 or coords.shape[2] != 3 or len(coords) < 2:
        raise ValueError("RMSF requires at least two coordinate frames")
    if reference.shape != coords.shape[1:] or mask.shape != (coords.shape[1],):
        raise ValueError("reference or mask shape mismatch")
    if not np.isfinite(reference).all() or int(mask.sum()) < 3:
        raise ValueError("degenerate fitting set: fewer than three finite reference points")
    target = reference[mask] - reference[mask].mean(axis=0)
    singular = np.linalg.svd(target, compute_uv=False)
    if singular[0] <= 0 or singular[1] <= rank_rtol * singular[0]:
        raise ValueError("degenerate fitting set: collinear reference")
    minimum_ratio = float("inf")
    for start in range(0, len(coords), chunk_size):
        block = np.asarray(coords[start : start + chunk_size], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("evaluation coordinates contain nonfinite values")
        points = block[:, mask]
        centered = points - points.mean(axis=1, keepdims=True)
        covariance = centered.swapaxes(1, 2) @ target
        sv = np.linalg.svd(covariance, compute_uv=False)
        ratio = np.divide(sv[:, 1], sv[:, 0], out=np.zeros(len(sv)), where=sv[:, 0] > 0)
        minimum_ratio = min(minimum_ratio, float(ratio.min()))
        if np.any(ratio <= rank_rtol):
            bad = start + int(np.flatnonzero(ratio <= rank_rtol)[0])
            raise ValueError(f"degenerate cross covariance at evaluation-local frame {bad}")
    fit = fit_all_frames_fixed_weights(
        coords,
        reference,
        mask.astype(np.float64),
        chunk_size=chunk_size,
        return_aligned=False,
        compute_rmsd=False,
        compute_weighted_rmsd=False,
        compute_mean_residual=False,
        compute_robust_rmsf=True,
        compute_ordinary_rmsf=False,
        validate_inputs=False,
    )
    if fit.robust_rmsf is None or not np.isfinite(fit.robust_rmsf).all():
        raise ValueError("nonfinite streamed RMSF result")
    return fit.robust_rmsf, {
        "status": "evaluable",
        "fit_set_size": int(mask.sum()),
        "rank_rtol": rank_rtol,
        "reference_second_to_first_singular_ratio": float(singular[1] / singular[0]),
        "minimum_cross_covariance_second_to_first_singular_ratio": minimum_ratio,
        "evaluated_frames": len(coords),
        "chunks_processed": fit.chunks_processed,
        "degenerate_frame_count": 0,
        "aligned_trajectory_retained": False,
    }


def neighborhood_mask(residues, atom_index, radius=3):
    center = residues[atom_index]
    return np.array(
        [
            r["chain"] == center["chain"]
            and abs(r["sequence_position"] - center["sequence_position"]) <= radius
            for r in residues
        ],
        dtype=bool,
    )


def comparison_regions(case, sets, original_sets):
    regions = {"all_residues": np.ones(len(case["residues"]), dtype=bool)}
    regions.update(
        {"literature_" + name: np.array(mask, dtype=bool) for name, mask in case["regions"].items()}
    )
    regions.update(
        {
            name: sets[key]
            for name, key in [("shared", "C"), ("MCA_only", "U_MCA"), ("MLF_only", "U_MLF")]
        }
    )
    original_difference = original_sets["U_MCA"] | original_sets["U_MLF"]
    neighborhoods = np.zeros(len(case["residues"]), dtype=bool)
    for i in np.flatnonzero(original_difference):
        neighborhood = neighborhood_mask(case["residues"], i)
        regions[f"original_difference_atom_{i}_plus_minus_3"] = neighborhood
        neighborhoods |= neighborhood
    regions["original_difference_all"] = original_difference
    regions["original_difference_neighborhood_union"] = neighborhoods
    return regions


def provenance(config, case, analysis_kind):
    native = analysis_kind == "native_full_method_outputs"
    full_count = len(case["train_indices"]) + (
        len(case["test_indices"]) if case["case_id"] == "ADK" else 0
    )
    selection_indices = range(full_count) if native else case["train_indices"]
    evaluation_indices = range(full_count) if native else case["test_indices"]
    return {
        "case": case["case_id"],
        "analysis_kind": analysis_kind,
        "selection_source": case["selection_input_pdb"],
        "selection_scope": "full" if native else "train",
        "selection_frame_indices": index_ranges(selection_indices),
        "evaluation_source": case["selection_input_pdb"]
        if native
        else case["evaluation_input_pdb"],
        "evaluation_frame_indices": index_ranges(evaluation_indices),
        "frame_index_encoding": "zero-based half-open ranges",
        "core_definition": (
            "native complete method outputs; matched-k masks only define summary groups"
            if native
            else "training-frozen binary sets; identical equal-weight Kabsch; not either native method"
        ),
        "reference_source": case["reference_source"]
        + "#model1 (original selection-source frame 0)",
        "run_id": config["run_id"],
        "source_run": config["source_run"],
        "rmsf_definition": "sqrt(sum_xyz population variance over aligned evaluation frames about their temporal mean)",
        "difference_sign": "method_a minus method_b",
    }


def compare_profiles(case, profiles, pairs, regions, context, sizes):
    summaries, per_residue = [], []
    for a, b in pairs:
        base = {
            **context,
            "method_a": a,
            "method_b": b,
            "k": f"{sizes[a]}/{sizes[b]}",
            "k_a": sizes[a],
            "k_b": sizes[b],
        }
        if a not in profiles or b not in profiles:
            summaries.append(
                {**base, "region": "all_residues", "status": "unevaluable_fit_see_audit"}
            )
            continue
        difference = profiles[a] - profiles[b]
        for name, mask in regions.items():
            indices = np.flatnonzero(mask)
            if not len(indices):
                summaries.append(
                    {**base, "region": name, "status": "unevaluable_empty_region", "n_residues": 0}
                )
                continue
            i = int(indices[np.argmax(np.abs(difference[indices]))])
            residue = case["residues"][i]
            summaries.append(
                {
                    **base,
                    "region": name,
                    "status": "evaluable",
                    "n_residues": len(indices),
                    "mean_RMSF_a_A": float(profiles[a][mask].mean()),
                    "mean_RMSF_b_A": float(profiles[b][mask].mean()),
                    "mean_signed_difference_A": float(difference[mask].mean()),
                    "profile_RMSE_A": float(np.sqrt(np.mean(difference[mask] ** 2))),
                    "max_abs_difference_A": float(np.max(np.abs(difference[mask]))),
                    "max_atom_index": i,
                    "max_chain": residue["chain"],
                    "max_source_resid": residue["source_resid"],
                    "max_icode": residue["icode"],
                    "max_resname": residue["resname"],
                    "max_literature_resid": residue["literature_resid"],
                }
            )
        for i, residue in enumerate(case["residues"]):
            per_residue.append(
                {
                    **base,
                    **residue,
                    "RMSF_a_A": float(profiles[a][i]),
                    "RMSF_b_A": float(profiles[b][i]),
                    "signed_difference_A": float(difference[i]),
                    "absolute_difference_A": float(abs(difference[i])),
                    "original_difference": bool(regions["original_difference_all"][i]),
                    "original_difference_plus_minus_3": bool(
                        regions["original_difference_neighborhood_union"][i]
                    ),
                    "summary_group": next(
                        (g for g in ["shared", "MCA_only", "MLF_only"] if regions[g][i]), "neither"
                    ),
                }
            )
    return summaries, per_residue


def run_case(config, case):
    run = Path(config["run_directory"])
    destination = run / "cases" / case["case_id"] / "rmsf"
    destination.mkdir(parents=True, exist_ok=True)
    original = source_sets(case, "full")
    training = source_sets(case, "train")
    source = Path(case["source_case_directory"]) / "alignment_profiles.npz"
    native = dict(np.load(source))
    native = {"native_" + k: native[k] for k in ["MDCoreAlign", "MDLovoFit"]}
    k = int(original["C_plus_U_MLF"].sum())
    native_context = {
        **provenance(config, case, "native_full_method_outputs"),
        "profile_source": str(source),
        "profile_source_sha256": sha256_file(source),
        "k_interpretation": "matched report size; native MDCoreAlign fit retains continuous weights across all residues",
    }
    summary, residues = compare_profiles(
        case,
        native,
        [("native_MDCoreAlign", "native_MDLovoFit")],
        comparison_regions(case, original, original),
        native_context,
        {name: k for name in native},
    )
    print(case["case_id"], "native profile differences ready; preparing held-out cache", flush=True)
    coords = build_evaluation_cache(case)
    print(case["case_id"], "evaluation cache ready", coords.shape, flush=True)
    reference, _ = load_ca_multimodel_pdb_minimal(
        case["reference_source"], expected_frames=1, expected_residues=len(case["residues"])
    )
    diagnostic, audits, sizes = {}, {}, {}
    for name in ["C", "C_plus_U_MCA", "C_plus_U_MLF"]:
        key = "diagnostic_" + name
        sizes[key] = int(training[name].sum())
        try:
            diagnostic[key], audits[key] = binary_fit_rmsf(coords, reference[0], training[name])
        except ValueError as error:
            audits[key] = {
                "status": "unevaluable",
                "reason": str(error),
                "fit_set_size": sizes[key],
            }
        print(case["case_id"], key, audits[key]["status"], flush=True)
    pairs = [
        ("diagnostic_C_plus_U_MCA", "diagnostic_C_plus_U_MLF"),
        ("diagnostic_C_plus_U_MCA", "diagnostic_C"),
        ("diagnostic_C_plus_U_MLF", "diagnostic_C"),
    ]
    controlled_context = {
        **provenance(config, case, "controlled_binary_Kabsch_heldout"),
        "profile_source": str(destination / "profiles.npz"),
        "profile_source_sha256": "recorded in rmsf_audit.json after export",
        "k_interpretation": "actual equal-weight binary fitting-set sizes",
    }
    controlled_summary, controlled_residues = compare_profiles(
        case,
        diagnostic,
        pairs,
        comparison_regions(case, training, original),
        controlled_context,
        sizes,
    )
    summary.extend(controlled_summary)
    residues.extend(controlled_residues)
    np.savez_compressed(destination / "profiles.npz", **native, **diagnostic)
    write_tsv(destination / "rmsf_impact.tsv", summary)
    write_tsv(destination / "rmsf_by_residue.tsv", residues)
    write_json(
        destination / "rmsf_audit.json",
        {
            "case": case["case_id"],
            "native_provenance": native_context,
            "diagnostic_provenance": controlled_context,
            "fits": audits,
            "diagnostic_set_atom_indices": {
                k: np.flatnonzero(training[k]).tolist()
                for k in ["C", "C_plus_U_MCA", "C_plus_U_MLF"]
            },
            "original_difference_atom_indices": np.flatnonzero(
                original["U_MCA"] | original["U_MLF"]
            ).tolist(),
            "training_difference_atom_indices": np.flatnonzero(
                training["U_MCA"] | training["U_MLF"]
            ).tolist(),
            "profile_sha256": sha256_file(destination / "profiles.npz"),
            "source_residue_results_sha256": sha256_file(
                Path(case["source_case_directory"]) / "residue_results.tsv"
            ),
            "reference_sha256": sha256_file(case["reference_source"]),
            "engine": "mdsuperpose.all_frame_engine.fit_all_frames_fixed_weights",
            "moments": "existing engine blockwise Chan population mean/M2; no complete aligned trajectory output",
            "interpretation": "Native full-source method differences cannot be attributed solely to core membership. Controlled held-out binary fits isolate set choice under one fixed fitter; neither native algorithm is rerun.",
        },
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", choices=["ADK", "3nci_A"])
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.run / "analysis_config.json").read_text())
    for case in config["cases"]:
        if args.case and args.case != case["case_id"]:
            continue
        if args.cache_only:
            coords = build_evaluation_cache(case)
            print(case["case_id"], "evaluation cache ready", coords.shape, flush=True)
        else:
            run_case(config, case)
    if not args.cache_only:
        for filename in ["rmsf_impact.tsv", "rmsf_by_residue.tsv"]:
            combined = []
            for case in config["cases"]:
                p = args.run / "cases" / case["case_id"] / "rmsf" / filename
                if p.exists():
                    combined.extend(read_rows(p))
            write_tsv(args.run / filename, combined)


if __name__ == "__main__":
    main()
