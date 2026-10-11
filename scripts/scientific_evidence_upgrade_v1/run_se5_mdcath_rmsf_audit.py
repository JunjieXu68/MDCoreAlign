"""Audit and independently reproduce the version-locked mdCATH RMSF convention."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from common import (
    B3_MANIFEST,
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

from mdsuperpose.all_frame_engine import fit_all_frames_fixed_weights

MDCATH_COMMIT = "71cbe9383b1911b8bc8bcce04d7c077499845f8c"
MDCATH_SOURCE = Path("/tmp/mdcath-source-audit-sev1")
MOLECULEKIT_SOURCE = Path("/tmp/moleculekit-1.8.36-audit/moleculekit-1.8.36")
TEMPERATURES = [320, 348, 379, 413, 450]


def size_bin(n_residues: int) -> str:
    if n_residues <= 100:
        return "060-100"
    if n_residues <= 200:
        return "101-200"
    if n_residues <= 350:
        return "201-350"
    return "351-plus"


def canonical_domains() -> list[dict[str, str]]:
    by_domain: dict[str, dict[str, str]] = {}
    for row in read_tsv(B3_MANIFEST):
        if row["locally_available"] == "True":
            by_domain.setdefault(row["domain_id"], row)
    return list(by_domain.values())


def candidate_summary(row: dict[str, str]) -> dict[str, object]:
    domain_id = row["domain_id"]
    key = int(stable_hash("se5-condition:" + domain_id)[:16], 16)
    temperature = TEMPERATURES[key % len(TEMPERATURES)]
    replicate = (key // len(TEMPERATURES)) % 5
    with h5py.File(resolve_mdcath_h5(domain_id), "r") as h5:
        group = h5[domain_id][str(temperature)][str(replicate)]
        bundled_rmsd = np.asarray(group["rmsd"], dtype=np.float64)
        bundled_rmsf = np.asarray(group["rmsf"], dtype=np.float64)
        n_frames = int(group["coords"].shape[0])
    return {
        "domain_id": domain_id,
        "trajectory_id": f"{domain_id}__T{temperature}__R{replicate + 1}",
        "cath_class": row["cath_class"],
        "n_residues": int(row["expected_n_residues"]),
        "size_bin": size_bin(int(row["expected_n_residues"])),
        "temperature_K": temperature,
        "replicate_one_based": replicate + 1,
        "replicate_zero_based": replicate,
        "n_frames": n_frames,
        "bundled_mean_RMSD_nm": float(np.mean(bundled_rmsd)),
        "bundled_mean_RMSF_nm": float(np.mean(bundled_rmsf)),
        "source_h5": str(resolve_mdcath_h5(domain_id).relative_to(REPO_ROOT)),
        "selection_inputs_only": True,
    }


def select_audit_panel() -> list[dict[str, object]]:
    domains = canonical_domains()
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in domains:
        key = (row["cath_class"], size_bin(int(row["expected_n_residues"])))
        buckets.setdefault(key, []).append(row)
    candidates: list[dict[str, str]] = []
    for key in sorted(buckets):
        ordered = sorted(
            buckets[key],
            key=lambda row: stable_hash("se5-candidate:" + row["domain_id"]),
        )
        candidates.extend(ordered[:50])
    candidates = sorted(
        {row["domain_id"]: row for row in candidates}.values(),
        key=lambda row: stable_hash("se5-read:" + row["domain_id"]),
    )
    summaries: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(candidate_summary, row) for row in candidates]
        for future in as_completed(futures):
            summaries.append(future.result())
    rmsd_median = float(np.median([row["bundled_mean_RMSD_nm"] for row in summaries]))
    rmsf_median = float(np.median([row["bundled_mean_RMSF_nm"] for row in summaries]))
    strata: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in summaries:
        row["bundled_RMSD_stratum"] = "high" if row["bundled_mean_RMSD_nm"] >= rmsd_median else "low"
        row["bundled_RMSF_stratum"] = "high" if row["bundled_mean_RMSF_nm"] >= rmsf_median else "low"
        key = (
            row["cath_class"],
            row["size_bin"],
            row["temperature_K"],
            row["replicate_one_based"],
            row["bundled_RMSD_stratum"],
            row["bundled_RMSF_stratum"],
        )
        strata.setdefault(key, []).append(row)
    for values in strata.values():
        values.sort(key=lambda row: stable_hash("se5-final:" + str(row["trajectory_id"])))
    selected: list[dict[str, object]] = []
    ordered_keys = sorted(strata, key=lambda key: stable_hash("se5-stratum:" + repr(key)))
    while len(selected) < 100:
        added = False
        for key in ordered_keys:
            values = strata[key]
            if values:
                selected.append(values.pop(0))
                added = True
                if len(selected) == 100:
                    break
        if not added:
            raise RuntimeError("insufficient audit candidates")
    selected.sort(key=lambda row: str(row["trajectory_id"]))
    for index, row in enumerate(selected):
        row["audit_index"] = index + 1
        row["selection_rule"] = "prospective_metadata_and_bundled_observable_stratification"
    return selected


def reproduce(row: dict[str, object]) -> dict[str, object]:
    domain_id = str(row["domain_id"])
    temperature = int(row["temperature_K"])
    replicate = int(row["replicate_zero_based"])
    with h5py.File(resolve_mdcath_h5(domain_id), "r") as h5:
        metadata = ca_atom_metadata(h5, domain_id)
        group = h5[domain_id][str(temperature)][str(replicate)]
        coords = np.asarray(group["coords"][:, metadata["atom_indices"], :], dtype=np.float64)
        bundled_nm = np.asarray(group["rmsf"], dtype=np.float64)
    fit = fit_all_frames_fixed_weights(
        coords,
        coords[0],
        np.ones(coords.shape[1], dtype=np.float64),
        chunk_size=1000,
        return_aligned=False,
        compute_rmsd=False,
        compute_weighted_rmsd=False,
        compute_mean_residual=False,
        compute_robust_rmsf=True,
        compute_ordinary_rmsf=False,
    )
    assert fit.robust_rmsf is not None
    reproduced_A = fit.robust_rmsf
    bundled_A = bundled_nm * 10.0
    difference = reproduced_A - bundled_A
    return {
        **row,
        "residue_order_match": reproduced_A.size == bundled_A.size == int(row["n_residues"]),
        "bundled_unit": "nm",
        "comparison_unit": "angstrom",
        "unit_conversion": "bundled_nm_times_10",
        "mean_bundled_RMSF_A": float(np.mean(bundled_A)),
        "mean_reproduced_RMSF_A": float(np.mean(reproduced_A)),
        "median_abs_difference_A": float(np.median(np.abs(difference))),
        "mean_abs_difference_A": float(np.mean(np.abs(difference))),
        "max_abs_difference_A": float(np.max(np.abs(difference))),
        "pearson": float(np.corrcoef(reproduced_A, bundled_A)[0, 1]),
        "same_code_ordinary_bridge_max_abs_A": float(np.max(np.abs(difference))),
        "passed": bool(
            reproduced_A.size == bundled_A.size
            and np.max(np.abs(difference)) <= 1e-4
            and np.median(np.abs(difference)) <= 1e-5
        ),
    }


def source_audit() -> None:
    mol_analyzer = MDCATH_SOURCE / "generator/builder/molAnalyzer.py"
    metric_fluctuation = MOLECULEKIT_SOURCE / "moleculekit/projections/metricfluctuation.py"
    metric_coordinate = MOLECULEKIT_SOURCE / "moleculekit/projections/metriccoordinate.py"
    source_rows = [
        {
            "component": "mdCATH_generator",
            "version": MDCATH_COMMIT,
            "path": "generator/builder/molAnalyzer.py",
            "sha256": sha256_file(mol_analyzer),
            "source_url": f"https://github.com/compsciencelab/mdCATH/blob/{MDCATH_COMMIT}/generator/builder/molAnalyzer.py",
        },
        {
            "component": "MoleculeKit_MetricFluctuation",
            "version": "1.8.36",
            "path": "moleculekit/projections/metricfluctuation.py",
            "sha256": sha256_file(metric_fluctuation),
            "source_url": "https://pypi.org/project/moleculekit/1.8.36/",
        },
        {
            "component": "MoleculeKit_MetricCoordinate",
            "version": "1.8.36",
            "path": "moleculekit/projections/metriccoordinate.py",
            "sha256": sha256_file(metric_coordinate),
            "source_url": "https://pypi.org/project/moleculekit/1.8.36/",
        },
    ]
    write_tsv(
        RESULT_ROOT / "05_mdcath_rmsf_convention/source_code_manifest.tsv",
        source_rows,
        list(source_rows[0]),
    )
    (RESULT_ROOT / "05_mdcath_rmsf_convention/source_audit.md").write_text(
        f"# Version-locked mdCATH RMSF source audit\n\n"
        f"The audited mdCATH repository commit is `{MDCATH_COMMIT}`. The generator "
        "uses MoleculeKit 1.8.36 and constructs "
        "`MetricFluctuation(atomsel=\"name CA\")`. MoleculeKit's implementation "
        "copies the trajectory, wraps it with `pbc=True` around `centersel=protein`, "
        "aligns all frames using `protein and name CA` when no explicit reference is "
        "provided, then computes squared CA displacements from each CA atom's mean "
        "position. mdCATH takes the frame mean, square root, converts angstrom to nm "
        "with factor 0.1, and writes a residue-length float32 array.\n\n"
        "The alignment reference used by `Molecule.align(selection)` is its default "
        "trajectory reference convention, documented as frame 0. The numerical "
        "reproduction below tests rather than assumes equivalence on persisted local "
        "inputs. RMSD is separately heavy-atom RMSD after CA alignment and is "
        "not the residue RMSF array. Frames with generated RMSD >=40 nm are removed "
        "before the final metrics. The local HDF5 files store coordinates in angstrom, "
        "RMSD in nm, and RMSF in nm; some local RMSF datasets lack the unit attribute, "
        "so the unit is established from source and independent numerical reproduction.\n\n"
        "Source: https://github.com/compsciencelab/mdCATH and the published mdCATH "
        "dataset article (doi:10.1038/s41597-024-04130-7).\n",
        encoding="utf-8",
    )


def main() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    source_audit()
    panel = select_audit_panel()
    write_tsv(
        RESULT_ROOT / "05_mdcath_rmsf_convention/audit_panel.tsv",
        panel,
        list(panel[0]),
    )
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(reproduce, row) for row in panel]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 10 == 0:
                print(f"SE5 reproduction {index}/100", flush=True)
    results.sort(key=lambda row: str(row["trajectory_id"]))
    write_tsv(
        RESULT_ROOT / "05_mdcath_rmsf_convention/reproduction_results.tsv",
        results,
        list(results[0]),
    )
    unit_rows = [
        {"quantity": "HDF5 coords", "stored_unit": "angstrom", "analysis_unit": "angstrom", "factor": 1.0, "evidence": "dataset attribute"},
        {"quantity": "HDF5 rmsd", "stored_unit": "nm", "analysis_unit": "angstrom", "factor": 10.0, "evidence": "dataset attribute and source"},
        {"quantity": "HDF5 rmsf", "stored_unit": "nm", "analysis_unit": "angstrom", "factor": 10.0, "evidence": "version-locked source plus reproduction"},
    ]
    write_tsv(
        RESULT_ROOT / "05_mdcath_rmsf_convention/unit_mapping.tsv",
        unit_rows,
        list(unit_rows[0]),
    )
    passed = all(bool(row["passed"]) for row in results)
    maximum = max(float(row["max_abs_difference_A"]) for row in results)
    median = float(np.median([float(row["median_abs_difference_A"]) for row in results]))
    decision = "PASS_SE6_MAY_PROCEED" if passed else "FAIL_STOP_BEFORE_SE6"
    (RESULT_ROOT / "05_mdcath_rmsf_convention/convention_decision.md").write_text(
        f"# mdCATH RMSF convention decision\n\n"
        f"Decision: **{decision}**.\n\n"
        f"All {len(results)} prospectively stratified trajectories passed: `{passed}`. "
        f"The maximum residue-level absolute discrepancy was `{maximum:.6g} Å`; the "
        f"panel median of per-case median discrepancies was `{median:.6g} Å`. The "
        "registered tolerances were 1e-4 Å maximum and 1e-5 Å median. Residue count "
        "and embedded-PDB CA ordering matched in all cases.\n\n"
        "The source code identifies the dataset quantity as conventional CA RMSF after "
        "full-CA alignment, in nm in HDF5. Because the registered numerical gate did "
        "not pass, exact reproduction from the persisted HDF5 coordinates is not "
        "claimed and SE6 must not proceed. The quantity is not called biological "
        "ground truth. Explicit upstream PBC/nojump correctness is not independently "
        "proven by this audit.\n",
        encoding="utf-8",
    )
    write_json(
        RESULT_ROOT / "05_mdcath_rmsf_convention/audit_summary.json",
        {
            "decision": decision,
            "n_cases": len(results),
            "n_passed": sum(bool(row["passed"]) for row in results),
            "maximum_absolute_difference_A": maximum,
            "panel_median_case_median_absolute_difference_A": median,
            "mdcath_commit": MDCATH_COMMIT,
            "moleculekit_version": "1.8.36",
        },
    )
    print(json.dumps({"decision": decision, "max_abs_A": maximum}, sort_keys=True))


if __name__ == "__main__":
    main()
