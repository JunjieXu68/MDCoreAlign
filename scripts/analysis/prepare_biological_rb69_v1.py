#!/usr/bin/env python3
"""Inventory existing RB69 inputs and materialise the R2 CA evaluation cache.

Reads historical files without changing them. No alignment method is run here.
The locally available ATLAS protein archive contains the R2 trajectory already.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_A = [
    (34, 41),
    (49, 62),
    (361, 367),
    (397, 431),
    (436, 464),
    (581, 609),
    (614, 633),
    (641, 715),
    (749, 755),
    (883, 889),
]
FRAMEWORK_B = [(2, 42), (49, 72), (80, 163), (180, 250), (262, 281), (290, 300), (318, 384)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def ranges_to_resids(ranges):
    return [r for a, b in ranges for r in range(a, b + 1)]


def topology_summary(path):
    u = mda.Universe(str(path))
    ca = u.select_atoms("protein and name CA")
    return (
        u,
        ca,
        {
            "path": str(path),
            "sha256": digest(path),
            "n_atoms": len(u.atoms),
            "n_residues": len(u.residues),
            "residue_counts": dict(collections.Counter(u.residues.resnames)),
            "segments": [
                {"segid": s.segid, "n_atoms": len(s.atoms), "n_residues": len(s.residues)}
                for s in u.segments
            ],
            "protein_sequence": "".join(seq1(r.resname) for r in ca.residues),
            "protein_n_CA": len(ca),
            "protein_resids": ca.resids.tolist(),
            "protein_resnames": ca.resnames.tolist(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/biological_core_validation_v1"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / "cases/3nci_A/source_R2"
    source.mkdir(parents=True, exist_ok=True)
    evidence = output / "sources"
    evidence.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "results/atlas_large_long_v1/source_archives/3nci_A_protein.zip"
    required = ["3nci_A_prod_R2.tpr", "3nci_A_prod_R2_fit.xtc", "3nci_A.pdb", "README.txt"]
    with zipfile.ZipFile(archive) as z:
        members = {i.filename: i.file_size for i in z.infolist()}
        for name in required:
            target = source / name
            if target.exists():
                if target.stat().st_size != members[name]:
                    raise ValueError(f"existing output has unexpected size: {target}")
                continue
            with z.open(name) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            print(f"extracted {name}", flush=True)

    snapshots = {}
    for name, url in {
        "3nci_rcsb_entity": "https://data.rcsb.org/rest/v1/core/polymer_entity/3NCI/1",
        "3nci_rcsb_instance": "https://data.rcsb.org/rest/v1/core/polymer_entity_instance/3NCI/A",
    }.items():
        snapshot = evidence / f"{name}.json"
        if not snapshot.exists():
            with urllib.request.urlopen(url, timeout=30) as handle:
                payload = json.load(handle)
            write_json(snapshot, payload)
        snapshots[name] = json.loads(snapshot.read_text())

    r1_dir = ROOT / "results/post_b3_optimisation_v1/p7_timing_atlas/atlas_readiness_inputs/3nci_A"
    u1, ca1, t1 = topology_summary(r1_dir / "3nci_A_prod_R1.tpr")
    u2, ca2, t2 = topology_summary(source / "3nci_A_prod_R2.tpr")
    entity = snapshots["3nci_rcsb_entity"]
    instance = snapshots["3nci_rcsb_instance"]
    rcsb_seq = entity["entity_poly"]["pdbx_seq_one_letter_code_can"]
    auth_map = instance["rcsb_polymer_entity_instance_container_identifiers"][
        "auth_to_entity_poly_seq_mapping"
    ]
    assert t1["protein_sequence"] == t2["protein_sequence"] == rcsb_seq
    assert auth_map == [str(i) for i in range(1, len(rcsb_seq) + 1)]
    assert t1["protein_resids"] == t2["protein_resids"] == list(range(1, 904))
    assert t1["residue_counts"] == t2["residue_counts"]

    cache1 = ROOT / "results/atlas_large_long_v1/input_cache/3nci_A_R1_CA_A_float64.npy"
    meta1 = ROOT / "results/atlas_large_long_v1/input_cache/3nci_A_R1_residue_metadata.npz"
    metadata = np.load(meta1, allow_pickle=False)
    assert metadata["residue_ids"].tolist() == [str(r) for r in ca1.resids]
    assert metadata["residue_names"].tolist() == ca1.resnames.tolist()
    assert metadata["chain_ids"].tolist() == ["A"] * 903
    assert metadata["atom_names"].tolist() == ["CA"] * 903
    npy1 = np.load(cache1, mmap_mode="r")
    assert npy1.shape == (10001, 903, 3) and npy1.dtype == np.float64

    # Protein-only XTC requires protein-only PDB topology; full TPR inventories
    # the production solvent/ions separately. Never pair its 132132 atoms with
    # the protein-only 14642-atom XTC.
    v2 = mda.Universe(str(source / "3nci_A.pdb"), str(source / "3nci_A_prod_R2_fit.xtc"))
    pca2 = v2.select_atoms("protein and name CA")
    assert pca2.resids.tolist() == ca2.resids.tolist()
    assert pca2.resnames.tolist() == ca2.resnames.tolist()
    assert len(v2.trajectory) == 10001
    cache2 = source / "3nci_A_R2_CA_A_float64.npy"
    times = np.empty(len(v2.trajectory), dtype=np.float64)
    # Only new-run derived outputs are writable.
    values = np.lib.format.open_memmap(
        cache2, mode="w+", dtype=np.float64, shape=(len(v2.trajectory), len(pca2), 3)
    )
    for i, ts in enumerate(v2.trajectory):
        values[i] = pca2.positions
        times[i] = ts.time
    values.flush()
    del values
    np.save(source / "frame_times_ps.npy", times)
    np.save(source / "frame_indices.npy", np.arange(len(times), dtype=np.int64))
    assert np.allclose(np.diff(times), 10.0)
    print(f"R2 cache ready: {cache2}", flush=True)

    # Avoid generating index-cache files in the historical directory: reuse the
    # existing immutable index explicitly through a normal reader; its existing
    # cache is present and valid. Sample checks validate existing R1 cache values.
    v1 = mda.Universe(str(r1_dir / "3nci_A.pdb"), str(r1_dir / "3nci_A_prod_R1_fit.xtc"))
    pca1 = v1.select_atoms("protein and name CA")
    sampled = np.linspace(0, len(v1.trajectory) - 1, 11, dtype=int)
    sample_errors = []
    for idx in sampled:
        v1.trajectory[int(idx)]
        sample_errors.append(float(np.max(np.abs(npy1[idx] - pca1.positions))))
    assert max(sample_errors) == 0.0
    r1_times = [float(v1.trajectory[i].time) for i in [0, 1, len(v1.trajectory) - 1]]
    assert len(v1.trajectory) == 10001 and r1_times == [0.0, 10.0, 100000.0]

    rows = []
    a, b = set(ranges_to_resids(FRAMEWORK_A)), set(ranges_to_resids(FRAMEWORK_B))
    for idx, atom in enumerate(pca2):
        rows.append(
            {
                "case_id": "3nci_A",
                "replicate": "R2",
                "ca_index_0based": idx,
                "source_PDB_atom_index_0based": int(atom.index),
                "source_chain_raw": atom.chainID,
                "chain": "A",
                "resid": int(atom.resid),
                "insertion_code": atom.icode,
                "resname": atom.resname,
                "oneletter": seq1(atom.resname),
                "literature_resid": int(auth_map[idx]),
                "mapping_status": "mapped_sequence_and_author_number_verified",
                "chain_mapping_basis": "TPR_Protein_chain_A_and_exact_903aa_RCSB_chain_A_sequence",
                "framework_A": int(atom.resid in a),
                "framework_B": int(atom.resid in b),
            }
        )
    mapping = source / "residue_mapping_R2.tsv"
    with mapping.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (source / "protein_sequence.fasta").write_text(
        ">ATLAS_3nci_A_R2|RCSB_3NCI_A|author_resid_1-903|D222A_D327A\n" + rcsb_seq + "\n"
    )
    np.savez_compressed(
        source / "3nci_A_R2_residue_metadata.npz",
        chain_ids=np.array(["A"] * 903),
        residue_ids=ca2.resids.astype(str),
        residue_names=ca2.resnames,
        atom_names=np.array(["CA"] * 903),
        insertion_codes=pca2.icodes,
        oneletter=np.array(list(rcsb_seq)),
    )

    source_urls = {
        "original_structure": "https://www.rcsb.org/structure/3NCI",
        "original_structure_primary_paper": "https://doi.org/10.1021/bi101192f",
        "atlas_paper": "https://doi.org/10.1093/nar/gkad1084",
        "atlas_full_text": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10767941/",
        "atlas_entry": "https://www.dsimb.inserm.fr/ATLAS/protein/3nci_A",
        "atlas_archive": "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/3nci_A",
        "framework_paper": "https://academic.oup.com/nar/article/44/15/7457/2457750",
        "framework_paper_PMC": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5009745/",
        "atlas_primary_fulltext_XML": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10767941/fullTextXML",
        "framework_primary_fulltext_XML": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5009745/fullTextXML",
    }
    inventory = {
        "case_id": "3nci_A",
        "source_urls": source_urls,
        "protein": "RB69 DNA polymerase gp43",
        "organism": "Escherichia phage RB69",
        "ncbi_taxonomy_id": 12353,
        "uniprot": "Q38087",
        "sequence": rcsb_seq,
        "chain": "A",
        "residue_numbering": "RCSB author 1-903; exactly matched to local TPR and PDB residue identifiers",
        "mutations": ["D222A", "D327A"],
        "mutation_source": entity["rcsb_polymer_entity"]["pdbx_mutation"],
        "missing_or_added_protein_residues_vs_deposited_3NCI": [],
        "terminal_tags_vs_deposited_3NCI": "none; 903-aa sequences equal",
        "original_crystal_state": "ternary complex: DNA template (author T, 18 nt), primer (author P, 13 nt; terminal DOC), dCTP (DCP), 3 calcium ions",
        "simulated_state": "isolated mutant protein in water with sodium and chloride; DNA, dCTP and calcium absent from both full production TPR topologies",
        "R1_full_topology": t1,
        "R2_full_topology": t2,
        "R1_cache": str(cache1),
        "R1_cache_sha256": digest(cache1),
        "R1_residue_metadata": str(meta1),
        "R1_metadata_matches_TPR": True,
        "R1_coordinate_sample_indices": sampled.tolist(),
        "R1_coordinate_sample_max_abs_errors_A": sample_errors,
        "R1_n_frames": len(v1.trajectory),
        "R1_first_second_last_times_ps": r1_times,
        "R2_cache": str(cache2),
        "R2_cache_sha256": digest(cache2),
        "R2_mapping": str(mapping),
        "R2_topology_PDB": str(source / "3nci_A.pdb"),
        "R2_trajectory": str(source / "3nci_A_prod_R2_fit.xtc"),
        "R2_trajectory_sha256": digest(source / "3nci_A_prod_R2_fit.xtc"),
        "R2_n_frames": len(times),
        "R2_first_last_times_ps": [float(times[0]), float(times[-1])],
        "frame_interval_ps": 10.0,
        "duration_ns_per_replicate": 100.0,
        "subset": "complete local R1/R2 protein trajectories, no temporal truncation or subsampling",
        "coordinate_unit": "Angstrom (MDAnalysis XTC conversion from nm)",
        "PBC": "ATLAS archive README says PBC removed; archive protein trajectories are pre-aligned",
        "prealignment": "ATLAS fitted protein trajectory; fitting selection and reference not independently recovered here",
        "replication": {
            "training": "R1",
            "geometry_evaluation": "R2",
            "basis": "ATLAS primary paper Methods specifies three 100-ns production replicates with different seeds for initial Boltzmann-distributed velocities; existing archive supplies named R1/R2/R3 production TPR and fitted trajectories; R1/R2 share identical construction but have distinct TPR and XTC hashes",
            "primary_paper_local_snapshot": str(evidence / "atlas_gkad1084.xml"),
            "independence_limit": "same source crystal/protocol; separate simulations are not separate biological experiments; exact random-seed values not extracted",
            "R1_R2_TPR_hash_distinct": t1["sha256"] != t2["sha256"],
        },
        "archive": str(archive),
        "archive_members": members,
        "extracted_members": required,
        "source_files_modified": False,
        "limitations": [
            "Ren frameworks describe relative invariance across crystal structures; they are external structural annotations, not MD-specific unique truth.",
            "Both frameworks must be retained separately and may overlap; their union is not an asserted correct core.",
            "Local ligand-free mutant simulations cannot establish DNA translocation or proofreading dynamics.",
        ],
    }
    write_json(evidence / "3nci_inventory.json", inventory)
    write_json(
        evidence / "3nci_literature.json",
        {
            "case_id": "3nci_A",
            "author": "Zhong Ren",
            "year": 2016,
            "title": "Molecular events during translocation and proofreading extracted from 200 static structures of DNA polymerase",
            "doi": "10.1093/nar/gkw555",
            "source_url": source_urls["framework_paper"],
            "source_location": "Figure 3 caption, panels A and B; directly verified from accessible primary article HTML",
            "numbering": "RB69 author residue numbering; mapped to local structure by full sequence and residue IDs",
            "evidence_type": "external crystallographic structure meta-analysis",
            "frameworks": [
                {
                    "name": "framework_A",
                    "ranges_inclusive": FRAMEWORK_A,
                    "residue_ids": ranges_to_resids(FRAMEWORK_A),
                    "n_residues": len(a),
                    "semantics": "palm-dominated relative rigid framework excluding fingers, with smaller N-terminal and thumb components",
                    "reported_internal_rmsd_matrix_average_A": 0.22,
                },
                {
                    "name": "framework_B",
                    "ranges_inclusive": FRAMEWORK_B,
                    "residue_ids": ranges_to_resids(FRAMEWORK_B),
                    "n_residues": len(b),
                    "semantics": "relative rigid framework spanning most N-terminal and exonuclease regions",
                    "reported_internal_rmsd_matrix_average_A": 0.23,
                },
            ],
            "shared_residue_ids": sorted(a & b),
            "n_shared_residues": len(a & b),
            "interpretation": "two frameworks can move relative to one another; do not choose whichever overlaps best, and do not treat their union as a single correct core",
        },
    )
    print(
        json.dumps(
            {
                "inventory": str(evidence / "3nci_inventory.json"),
                "framework_A_size": len(a),
                "framework_B_size": len(b),
                "shared": len(a & b),
                "R2_cache": str(cache2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
