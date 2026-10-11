"""Bounded biological core validation using the unchanged AT5 throughput estimator.

All writes are confined to the explicitly supplied new run directory. No timing
benchmark or production algorithm is modified or run by this analysis.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from mdcorealign.benchmark_minimal import load_ca_multimodel_pdb_minimal, solve_minimal_cpu
from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights
from mdsuperpose.fast_kabsch import apply_transforms_batch
from mdsuperpose.pdb_writer import StreamingCAPDBWriter
from scripts.atlas_strict_single_thread_cpu_v1.common import (
    STOCK_MDLOVOFIT,
    sha256,
    thread_environment,
    write_json,
    write_tsv,
)
from scripts.atlas_strict_single_thread_cpu_v1.extract_cores import (
    overlap,
    parse_mdlovofit_mask,
    residue_metadata,
    top_k,
)

SEED = 20260914


def write_ca(coords: np.ndarray, path: Path, residues: list[dict]) -> None:
    """Reuse the streaming writer with explicitly mapped residue identities."""
    writer = StreamingCAPDBWriter(len(residues))
    for i, atom in enumerate(residues):
        for start, stop, value in (
            (17, 20, f"{atom['resname']:>3s}"),
            (21, 22, atom["chain"]),
            (22, 26, f"{int(atom['source_resid']):4d}"),
            (26, 27, atom.get("icode", "") or " "),
        ):
            if len(value) != stop - start:
                raise ValueError(f"PDB identity field overflow: {atom}")
            writer._line_buffer[i, start:stop] = np.frombuffer(
                value.encode("ascii"), dtype=np.uint8
            )
    writer.write(coords, path)


def common_input(
    coords: np.ndarray, path: Path, residues: list[dict], center: bool = True
) -> np.ndarray:
    values = np.asarray(coords, dtype=np.float64)
    if center:
        values = values - values.mean(axis=1, keepdims=True)
    write_ca(values, path, residues)
    parsed, _ = load_ca_multimodel_pdb_minimal(
        path, expected_frames=len(values), expected_residues=len(residues)
    )
    return parsed


def read_export(
    path: Path, n_frames: int, n_atoms: int, expected_labels: list[tuple] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read stock aligned coordinates and binary occupancies without assuming fixed membership."""
    aligned = np.empty((n_frames, n_atoms, 3))
    masks = np.empty((n_frames, n_atoms), dtype=bool)
    frame = atom = 0
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        for line in handle:
            kind = line[:6].strip()
            if kind == b"ATOM":
                if frame >= n_frames or atom >= n_atoms or line[12:16].strip() != b"CA":
                    raise ValueError("invalid stock export CA/frame cardinality")
                if expected_labels is not None:
                    identity = (
                        line[21:22].decode().strip(),
                        line[22:27].decode().strip(),
                        line[17:20].decode().strip(),
                    )
                    if identity != expected_labels[atom]:
                        raise ValueError("stock export residue identity mismatch")
                occupancy = float(line[54:60])
                if occupancy not in (0.0, 1.0):
                    raise ValueError("non-binary MDLovoFit occupancy")
                masks[frame, atom] = occupancy == 1.0
                aligned[frame, atom] = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
                atom += 1
            elif kind in (b"END", b"ENDMDL") and atom:
                if atom != n_atoms:
                    raise ValueError("incomplete stock output frame")
                frame += 1
                atom = 0
    if atom or frame != n_frames:
        raise ValueError("stock export frame count mismatch")
    return aligned, masks


def mdlovo(coords: np.ndarray, input_path: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "aligned.pdb"
    command = [str(STOCK_MDLOVOFIT), "-f", "0.600000", "-t", str(output), str(input_path)]
    with (
        (destination / "stdout.txt").open("w") as stdout,
        (destination / "stderr.txt").open("w") as stderr,
    ):
        result = subprocess.run(
            command,
            env=thread_environment(),
            stdout=stdout,
            stderr=stderr,
            timeout=3600,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"MDLovoFit exit {result.returncode}; evidence retained at {destination}"
        )
    expected = [
        (r["chain"], r["residue_id"], r["resname"])
        for r in residue_metadata(input_path, coords.shape[1])
    ]
    aligned, masks = read_export(output, len(coords), coords.shape[1], expected)
    frequency = masks.mean(axis=0)
    sizes = masks.sum(axis=1)
    constant = bool(np.all(masks == masks[0]))
    if len(set(sizes.tolist())) != 1:
        raise ValueError(
            "MDLovoFit varies per-frame cardinality; no new consensus-size rule invented"
        )
    k = int(sizes[0])
    if constant:
        mask, count, labels = parse_mdlovofit_mask(output, coords.shape[1])
        if count != len(coords) or labels != expected:
            raise ValueError("stock export identity mismatch")
    else:
        mask = top_k(frequency, k)
        # Explicit derived consensus; raw frame masks are retained below.
    np.savez_compressed(destination / "membership.npz", frequency=frequency, masks=masks, core=mask)
    metadata = {
        "command": command,
        "input_sha256": sha256(input_path),
        "executable_sha256": sha256(STOCK_MDLOVOFIT),
        "output_sha256": sha256(output),
        "returncode": result.returncode,
        "n_frames": len(coords),
        "core_size": k,
        "constant_across_frames": constant,
        "core_kind": "native_fixed_core" if constant else "derived_frequency_consensus_core",
        "consensus_rule": "descending selection frequency, ascending input-index ties; observed common frame cardinality",
        "timing_role": "none; scientific analysis only",
    }
    write_json(destination / "export_manifest.json", metadata)
    # Align the first exported frame once to the common reference; this common
    # rigid correction changes neither per-residue RMSF nor internal geometry.
    anchor = fit_all_frames_fixed_weights(
        aligned[:1],
        coords[0],
        np.ones(coords.shape[1]),
        return_aligned=False,
        compute_robust_rmsf=False,
        compute_ordinary_rmsf=False,
    )
    aligned = apply_transforms_batch(
        aligned,
        np.repeat(anchor.rotations, len(aligned), axis=0),
        np.repeat(anchor.translations, len(aligned), axis=0),
    )
    metadata["first_frame_anchor_rmsd_A"] = float(
        np.sqrt(np.mean(np.sum((aligned[0] - coords[0]) ** 2, axis=1)))
    )
    write_json(destination / "export_manifest.json", metadata)
    output.unlink()  # Validated occupancy arrays, hashes, transforms/figures survive; failure PDBs survive.
    return {"aligned": aligned, "mask": mask, "frequency": frequency, "metadata": metadata}


def rmsf(aligned: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.sum((aligned - aligned.mean(axis=0)) ** 2, axis=2), axis=0))


def pair_variances(coords: np.ndarray, residues: list[dict], frame_block: int = 16) -> dict:
    """Chan-merged population variances of all eligible pair distances in Angstrom."""
    left, right = np.triu_indices(len(residues), 1)
    chains = np.asarray([r["chain"] for r in residues])
    sequence = np.asarray([r["sequence_position"] for r in residues], dtype=int)
    valid = ~((chains[left] == chains[right]) & (np.abs(sequence[left] - sequence[right]) <= 3))
    mean = np.zeros(int(valid.sum()))
    m2 = mean.copy()
    count = 0
    for start in range(0, len(coords), frame_block):
        block = np.stack([pdist(frame)[valid] for frame in coords[start : start + frame_block]])
        b = len(block)
        block_mean = block.mean(axis=0)
        block_m2 = np.sum((block - block_mean) ** 2, axis=0)
        total = count + b
        delta = block_mean - mean
        mean += delta * b / total
        m2 += block_m2 + delta**2 * count * b / total
        count = total
    if count < 2:
        raise ValueError("heldout geometry needs at least two frames")
    return {
        "left": left[valid],
        "right": right[valid],
        "mean_A": mean,
        "variance_A2": m2 / count,
        "n_frames": np.asarray(count),
    }


def geometry_score(mask: np.ndarray, pairs: dict) -> tuple[float | None, int]:
    selected = mask[pairs["left"]] & mask[pairs["right"]]
    n = int(selected.sum())
    return (float(np.sqrt(np.mean(pairs["variance_A2"][selected]))) if n else None, n)


def random_scores(n_residues: int, k: int, pairs: dict, count: int = 1000) -> list[float | None]:
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(count):
        mask = np.zeros(n_residues, dtype=bool)
        mask[rng.choice(n_residues, k, replace=False)] = True
        value, _ = geometry_score(mask, pairs)
        scores.append(value)
    return scores


def region_statistics(mask: np.ndarray, region: np.ndarray, evaluable: np.ndarray) -> dict:
    valid_core = mask & evaluable
    denominator = int(valid_core.sum())
    positives = int((region & evaluable).sum())
    n = int(evaluable.sum())
    intersection = int((valid_core & region).sum())
    purity = intersection / denominator if denominator else None
    return {
        "overlap": intersection,
        "core_size": int(mask.sum()),
        "evaluated_core_size": denominator,
        "unannotated_core_fraction": 1 - denominator / int(mask.sum()) if np.any(mask) else None,
        "region_size": positives,
        "evaluable_residues": n,
        "purity": purity,
        "coverage": intersection / positives if positives else None,
        "enrichment": purity / (positives / n) if purity is not None and positives and n else None,
    }


def orientation_check(coords: np.ndarray, residues: list[dict], destination: Path) -> dict:
    destination.mkdir()
    indices = np.unique(np.linspace(0, len(coords) - 1, min(64, len(coords))).astype(int))
    sample = coords[indices].copy()
    rng = np.random.default_rng(SEED)
    rotations = Rotation.random(len(sample) - 1, random_state=rng).as_matrix()
    translation = rng.uniform(-25, 25, (len(sample) - 1, 3))
    perturbed = sample.copy()
    perturbed[1:] = sample[1:] @ rotations + translation[:, None, :]
    prequant_error = max(
        float(np.max(np.abs(pdist(a) - pdist(b)))) for a, b in zip(sample, perturbed, strict=True)
    )
    a = common_input(sample, destination / "original.pdb", residues, center=False)
    b = common_input(perturbed, destination / "perturbed.pdb", residues, center=False)
    assert np.array_equal(a[0], b[0])
    chains = np.asarray([r["chain"] for r in residues])
    ma = solve_minimal_cpu(a, mode="throughput", chain_ids=chains)
    mb = solve_minimal_cpu(b, mode="throughput", chain_ids=chains)
    la = mdlovo(a, destination / "original.pdb", destination / "mdlovofit_original")
    lb = mdlovo(b, destination / "perturbed.pdb", destination / "mdlovofit_perturbed")
    pairs_a = pair_variances(a, residues)
    pairs_b = pair_variances(b, residues)
    frozen = ma.core_labels
    ga, _ = geometry_score(frozen, pairs_a)
    gb, _ = geometry_score(frozen, pairs_b)
    record = {
        "seed": SEED,
        "sample_frame_indices": indices.tolist(),
        "reference_unchanged": True,
        "max_pair_distance_error_before_PDB_A": prequant_error,
        "max_pair_distance_error_after_PDB_A": max(
            float(np.max(np.abs(pdist(x) - pdist(y)))) for x, y in zip(a, b, strict=True)
        ),
        "max_MDCoreAlign_weight_difference": float(
            np.max(np.abs(ma.frozen_weights - mb.frozen_weights))
        ),
        "mean_MDCoreAlign_weight_difference": float(
            np.mean(np.abs(ma.frozen_weights - mb.frozen_weights))
        ),
        "MDCoreAlign_core_jaccard": overlap(ma.core_labels, mb.core_labels)["jaccard"],
        "MDCoreAlign_core_changed_residues": int(np.count_nonzero(ma.core_labels ^ mb.core_labels)),
        "MDLovoFit_core_jaccard": overlap(la["mask"], lb["mask"])["jaccard"],
        "MDLovoFit_frequency_max_difference": float(
            np.max(np.abs(la["frequency"] - lb["frequency"]))
        ),
        "same_frozen_set_geometry_original_A": ga,
        "same_frozen_set_geometry_perturbed_A": gb,
        "geometry_absolute_difference_A": abs(ga - gb),
        "interpretation": "Numerical diagnostic on the same fixed sample, including 0.001 A PDB quantisation; no fit-quality tuning.",
    }
    np.savez_compressed(
        destination / "weight_core_comparison.npz",
        mdca_original=ma.frozen_weights,
        mdca_perturbed=mb.frozen_weights,
        mdlovo_original=la["frequency"],
        mdlovo_perturbed=lb["frequency"],
    )
    write_json(destination / "orientation_check.json", record)
    return record


def domain_motion(
    coords: np.ndarray, residues: list[dict], regions: dict
) -> tuple[list[dict], list[dict]]:
    """COM distances and a three-domain angle without any alignment."""
    centers = {name: coords[:, mask].mean(axis=1) for name, mask in regions.items() if np.any(mask)}
    rows = []
    summary = []
    for a, b in (("CORE", "LID"), ("CORE", "NMP"), ("LID", "NMP")):
        value = np.linalg.norm(centers[a] - centers[b], axis=1)
        name = f"{a}_{b}_centroid_distance_A"
        summary.append(
            {
                "metric": name,
                "mean": float(value.mean()),
                "SD": float(value.std()),
                "min": float(value.min()),
                "max": float(value.max()),
                "p5": float(np.quantile(value, 0.05)),
                "p95": float(np.quantile(value, 0.95)),
            }
        )
        rows.extend({"frame_index": i, "metric": name, "value": x} for i, x in enumerate(value))
    u, v = centers["LID"] - centers["CORE"], centers["NMP"] - centers["CORE"]
    angle = np.degrees(
        np.arccos(
            np.clip(
                np.sum(u * v, axis=1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1
            )
        )
    )
    name = "LID_CORE_NMP_centroid_angle_degrees"
    summary.append(
        {
            "metric": name,
            "mean": float(angle.mean()),
            "SD": float(angle.std()),
            "min": float(angle.min()),
            "max": float(angle.max()),
            "p5": float(np.quantile(angle, 0.05)),
            "p95": float(np.quantile(angle, 0.95)),
        }
    )
    rows.extend({"frame_index": i, "metric": name, "value": x} for i, x in enumerate(angle))
    return rows, summary


def validate_split(case: dict, n_frames: int, n_test_frames: int | None = None) -> None:
    """Reject leakage, reordered/duplicate frames and out-of-range frozen splits."""
    independent = bool(case.get("test_coords_path"))
    for key, bound in [
        ("train_indices", n_frames),
        ("test_indices", n_test_frames if independent else n_frames),
    ]:
        values = np.asarray(case[key])
        if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
            raise ValueError(f"{key} must be nonempty integer indices")
        if (
            bound is None
            or np.any(values < 0)
            or np.any(values >= bound)
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(f"{key} must be unique ordered indices within source bounds")
    if independent:
        if Path(case["coords_path"]).resolve() == Path(case["test_coords_path"]).resolve():
            raise ValueError("independent replicate paths must differ")
    elif np.intersect1d(case["train_indices"], case["test_indices"]).size:
        raise ValueError("training and heldout frames overlap")
    if len(case["test_indices"]) < 2:
        raise ValueError("heldout geometry needs at least two frames")


def analyze_case(config_path: Path, case_id: str) -> None:
    config = json.loads(config_path.read_text())
    run = Path(config["run_directory"])
    case = next(c for c in config["cases"] if c["case_id"] == case_id)
    destination = run / "cases" / case_id / "analysis"
    destination.mkdir(parents=True, exist_ok=False)
    residues = case["residues"]
    raw = np.load(case["coords_path"], mmap_mode="r")
    test_raw = (
        np.load(case["test_coords_path"], mmap_mode="r") if case.get("test_coords_path") else None
    )
    validate_split(case, len(raw), len(test_raw) if test_raw is not None else None)
    coords = common_input(raw, destination / "common_input.pdb", residues)
    train_indices = np.asarray(case["train_indices"], dtype=int)
    if case.get("test_coords_path"):
        test = common_input(
            test_raw[np.asarray(case["test_indices"], dtype=int)],
            destination / "heldout_input.pdb",
            residues,
        )
    else:
        test = coords[np.asarray(case["test_indices"], dtype=int)]
    train = coords[train_indices]
    chains = np.asarray([r["chain"] for r in residues])
    print(
        f"{case_id}: input ready {coords.shape}, train={len(train)}, heldout={len(test)}",
        flush=True,
    )
    full = solve_minimal_cpu(coords, mode="throughput", chain_ids=chains)
    fitted = apply_transforms_batch(coords, full.rotations, full.translations)
    external = mdlovo(coords, destination / "common_input.pdb", destination / "mdlovofit_full")
    if len(train) == len(coords) and np.array_equal(train_indices, np.arange(len(coords))):
        training, train_external = full, external
    else:
        training = solve_minimal_cpu(train, mode="throughput", chain_ids=chains)
        write_ca(train, destination / "train_input.pdb", residues)
        train_external = mdlovo(
            train, destination / "train_input.pdb", destination / "mdlovofit_train"
        )
    regions = {name: np.asarray(mask, dtype=bool) for name, mask in case["regions"].items()}
    evaluable = np.asarray(case["literature_evaluable"], dtype=bool)
    full_matched = top_k(full.frozen_weights, int(external["mask"].sum()))
    train_matched = top_k(training.frozen_weights, int(train_external["mask"].sum()))
    selected_frames = np.unique(np.linspace(0, len(coords) - 1, 7).astype(int))
    alignments = {"MDCoreAlign": fitted, "MDLovoFit": external["aligned"]}
    for name, weights in [
        ("all_CA", np.ones(len(residues))),
        *([("literature_CORE", regions["CORE"].astype(float))] if "CORE" in regions else []),
    ]:
        fit = fit_all_frames_fixed_weights(
            coords,
            coords[0],
            weights,
            return_aligned=True,
            compute_robust_rmsf=False,
            compute_ordinary_rmsf=False,
        )
        alignments[name] = fit.aligned_coords
    rmsf_profiles = {name: rmsf(values) for name, values in alignments.items()}
    for name, values in alignments.items():
        write_ca(values[selected_frames], destination / f"superposition_{name}.pdb", residues)
    np.savez_compressed(destination / "alignment_profiles.npz", **rmsf_profiles)
    write_ca(coords[:1], destination / "reference.pdb", residues)
    residue_rows = []
    region_rows = []
    for scope, result, ext, matched in [
        ("full", full, external, full_matched),
        ("train", training, train_external, train_matched),
    ]:
        for i, residue in enumerate(residues):
            residue_rows.append(
                {
                    "case_id": case_id,
                    "scope": scope,
                    **residue,
                    "literature_regions": ";".join(
                        name for name, mask in regions.items() if mask[i]
                    ),
                    "MDCoreAlign_weight": float(result.frozen_weights[i]),
                    "MDCoreAlign_native60": int(result.core_labels[i]),
                    "MDCoreAlign_matched_k": int(matched[i]),
                    "MDLovoFit_core": int(ext["mask"][i]),
                    "MDLovoFit_frequency": float(ext["frequency"][i]),
                    **(
                        {
                            f"RMSF_{name}_A": float(profile[i])
                            for name, profile in rmsf_profiles.items()
                        }
                        if scope == "full"
                        else {}
                    ),
                }
            )
        for name, region in regions.items():
            for method, mask in [
                ("MDCoreAlign_native60", result.core_labels),
                ("MDCoreAlign_matched_k", matched),
                ("MDLovoFit", ext["mask"]),
            ]:
                stats = region_statistics(mask, region, evaluable)
                region_rows.append(
                    {
                        "case_id": case_id,
                        "scope": scope,
                        "region": name,
                        "method": method,
                        **stats,
                        "region_mean_weight": float(result.frozen_weights[region].mean()),
                        "region_median_weight": float(np.median(result.frozen_weights[region])),
                        "region_weight_p25": float(
                            np.quantile(result.frozen_weights[region], 0.25)
                        ),
                        "region_weight_p75": float(
                            np.quantile(result.frozen_weights[region], 0.75)
                        ),
                        "region_core_inclusion_fraction": float(mask[region].mean()),
                        "annotation_role": case["region_roles"][name],
                    }
                )
    write_tsv(destination / "residue_results.tsv", residue_rows)
    write_tsv(destination / "region_metrics.tsv", region_rows)
    print(f"{case_id}: full and training cores frozen; starting heldout distances", flush=True)
    pairs = pair_variances(test, residues)
    np.savez_compressed(destination / "heldout_pair_variances.npz", **pairs)
    selected_sets = {
        "MDCoreAlign_native60": training.core_labels,
        "MDCoreAlign_matched_k": train_matched,
        "MDLovoFit": train_external["mask"],
        **{f"literature_{name}": mask for name, mask in regions.items()},
    }
    random_by_k = {
        k: random_scores(len(residues), k, pairs)
        for k in sorted({int(mask.sum()) for mask in selected_sets.values() if np.any(mask)})
    }
    random_rows = [
        {"case_id": case_id, "cardinality": k, "draw": i, "G_A": value, "seed": SEED}
        for k, scores in random_by_k.items()
        for i, value in enumerate(scores)
    ]
    geometry_rows = []
    for name, mask in selected_sets.items():
        score, pair_count = geometry_score(mask, pairs)
        draws = random_by_k[int(mask.sum())]
        control = np.asarray([value for value in draws if value is not None])
        geometry_rows.append(
            {
                "case_id": case_id,
                "method": name,
                "core_size": int(mask.sum()),
                "pair_count": pair_count,
                "G_A": score,
                "variance_ddof": 0,
                "validation_kind": case["validation_kind"],
                "n_train_frames": len(train),
                "n_test_frames": len(test),
                "random_count": len(draws),
                "random_evaluable_count": len(control),
                "random_unevaluable_count": len(draws) - len(control),
                "random_median_G_A": float(np.median(control)),
                "random_p5_G_A": float(np.quantile(control, 0.05)),
                "random_p95_G_A": float(np.quantile(control, 0.95)),
                "G_over_random_median": score / float(np.median(control))
                if score is not None
                else None,
                "random_fraction_at_or_below_selected": float(np.mean(control <= score))
                if score is not None
                else None,
                "interpretation": "equal-size reference, not a biological p value; different-size literature sets are not direct wins/losses",
            }
        )
    write_tsv(destination / "geometry_metrics.tsv", geometry_rows)
    write_tsv(destination / "random_geometry.tsv", random_rows)
    motion = []
    if case_id == "ADK":
        motion_rows, motion = domain_motion(coords, residues, regions)
        write_tsv(destination / "domain_motion_timeseries.tsv", motion_rows)
        write_tsv(destination / "domain_motion_summary.tsv", motion)
    check = orientation_check(coords, residues, destination / "orientation")
    summary = {
        "case_id": case_id,
        "status": "complete",
        "mode": "throughput",
        "backend": "cpu",
        "production_metadata": full.metadata,
        "configuration_sha256": sha256(config_path),
        "n_frames": len(coords),
        "n_residues": len(residues),
        "train_indices": train_indices.tolist(),
        "test_indices": case["test_indices"],
        "validation_kind": case["validation_kind"],
        "superposition_frame_indices": selected_frames.tolist(),
        "full_core_agreement": overlap(full_matched, external["mask"]),
        "MDLovoFit_core_kind": external["metadata"]["core_kind"],
        "orientation": check,
        "domain_motion": motion,
        "RMSF_definition": "sqrt(mean_t ||aligned_x_i(t) - temporal_mean(aligned_x_i)||^2); Angstrom; population normalization",
        "literature_used_in_automatic_estimation": False,
    }
    write_json(destination / "case_summary.json", summary)
    print(f"{case_id}: COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    analyze_case(args.config, args.case)


if __name__ == "__main__":
    main()
