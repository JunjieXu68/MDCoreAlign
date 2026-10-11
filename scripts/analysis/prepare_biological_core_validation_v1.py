"""Freeze mapped literature, full local inputs and splits before case analysis."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import yaml

from scripts.atlas_strict_single_thread_cpu_v1.common import sha256, write_json, write_tsv

ROOT = Path(__file__).resolve().parents[2]


def prepare(run: Path) -> None:
    if (run / "analysis_config.json").exists():
        raise FileExistsError("configuration already frozen; use a fresh run directory")
    local = json.loads((run / "sources/local_inventory.json").read_text())
    lit = json.loads((run / "sources/adk_ifabp_literature.json").read_text())
    rb = json.loads((run / "sources/3nci_inventory.json").read_text())
    rb_lit = json.loads((run / "sources/3nci_literature.json").read_text())
    mappings = list(
        csv.DictReader((run / "sources/residue_mapping_ADK_IFABP.tsv").open(), delimiter="\t")
    )
    config = {
        "run_directory": str(run.resolve()),
        "production_mode": "throughput",
        "random_seed": 20260914,
        "random_sets_per_cardinality": 1000,
        "variance_ddof": 0,
        "exclude_same_chain_sequence_separation_le": 3,
        "annotation_input_to_automatic_methods": False,
        "cases": [],
    }
    annotations, all_residues, inventories, inputs = [], [], [], []
    for source_id, case_id in [("ADK", "ADK"), ("IFABP", "I-FABP"), ("3nci_A", "3nci_A")]:
        dest = run / "cases" / case_id
        dest.mkdir(parents=True, exist_ok=True)
        if source_id != "3nci_A":
            info = local["cases"][source_id]
            topology = info["inputs"]["topology"]["path"]
            trajectory = info["inputs"]["trajectory"]["path"]
            universe = mda.Universe(topology, trajectory)
            ca = universe.select_atoms("protein and name CA")
            coords = np.stack([ca.positions.copy() for _ in universe.trajectory]).astype(np.float64)
            coords_path = dest / "source_CA_A_float64.npy"
            np.save(coords_path, coords)
            rows = [r for r in mappings if r["case_id"] == source_id]
            residues = [
                {
                    "atom_index": int(r["ca_index_zero_based"]),
                    "source_atom_index": int(r["source_atom_index_zero_based"]),
                    "chain": "A",
                    "source_chain": "",
                    "source_segid": r["source_segid"],
                    "chain_mapping": "single source protein segid explicitly represented as output chain A",
                    "source_resid": int(r["source_resid"]),
                    "resname": r["source_resname"],
                    "icode": "",
                    "sequence_position": int(r["ca_index_zero_based"]) + 1,
                    "literature_resid": int(r["source_resid"]),
                    "one_letter": r["one_letter_residue"],
                    "uniprot": r["uniprot_accession"],
                    "uniprot_position": int(r["uniprot_position_one_based"]),
                    "mapping_status": "sequence_and_number_verified",
                }
                for r in rows
            ]
            assert list(ca.resids) == [r["source_resid"] for r in residues]
            assert list(ca.resnames) == [r["resname"] for r in residues]
            n = len(coords)
            dt = info["timing_and_split"]["documented_frame_interval_ps"]
            intervals = (
                lit["adk_domain_annotation"]["domains"]
                if source_id == "ADK"
                else {
                    f"portal_{r['start']}_{r['end']}": [[r["start"], r["end"]]]
                    for r in lit["ifabp_experimental_annotation"]["apo_dynamic_regions_inclusive"]
                }
            )
            if source_id == "IFABP":
                intervals["portal_combined"] = [ranges[0] for ranges in intervals.values()]
            roles = {
                k: (
                    "literature_relative_stable_domain"
                    if k == "CORE"
                    else "literature_mobile_domain"
                )
                if source_id == "ADK"
                else "experimental_apo_dynamic_region_state_match_limited"
                for k in intervals
            }
            case = {
                "train_indices": list(range(n // 2)),
                "test_indices": list(range(n // 2, n)),
                "validation_kind": "chronological_time_block_first_half_to_second_half",
            }
            inv = {
                "case_id": case_id,
                "topology": topology,
                "trajectory": trajectory,
                "n_frames": n,
                "n_CA": len(ca),
                "frame_interval_ps": dt,
                "stored_span_ns": (n - 1) * dt / 1000,
                "source_doi": info["dataset_url"],
                "state": info["ligand_and_state"]["interpretation"],
                "prealignment": info["preprocessing"]["source_alignment"],
                "PBC": info["preprocessing"]["pbc_unwrap_nojump_status"],
                "unknowns": "; ".join(info["unknowns"]),
                "metadata_detail": str(run / "sources/local_inventory.json"),
            }
            inputs.extend(info["inputs"].values())
            doi = (
                lit["adk_domain_annotation"]["doi"]
                if source_id == "ADK"
                else lit["ifabp_experimental_annotation"]["doi"]
            )
        else:
            topology = str(Path(rb["R1_full_topology"]["path"]).parent / "3nci_A.pdb")
            trajectory = str(Path(topology).parent / "3nci_A_prod_R1_fit.xtc")
            universe = mda.Universe(topology, trajectory)
            ca = universe.select_atoms("protein and name CA")
            coords_path = Path(rb["R1_cache"])
            rows = list(csv.DictReader(Path(rb["R2_mapping"]).open(), delimiter="\t"))
            residues = [
                {
                    "atom_index": int(r["ca_index_0based"]),
                    "source_atom_index": int(r["source_PDB_atom_index_0based"]),
                    "chain": r["chain"],
                    "source_chain": r["source_chain_raw"],
                    "source_segid": "seg_0_Protein_chain_A",
                    "chain_mapping": r["chain_mapping_basis"],
                    "source_resid": int(r["resid"]),
                    "resname": r["resname"],
                    "icode": r["insertion_code"],
                    "sequence_position": int(r["ca_index_0based"]) + 1,
                    "literature_resid": int(r["literature_resid"]),
                    "one_letter": r["oneletter"],
                    "uniprot": rb["uniprot"],
                    "uniprot_position": int(r["resid"]),
                    "mapping_status": r["mapping_status"],
                }
                for r in rows
            ]
            assert list(ca.resids) == [r["source_resid"] for r in residues]
            assert list(ca.resnames) == [r["resname"] for r in residues]
            n, dt = rb["R1_n_frames"], rb["frame_interval_ps"]
            intervals = {r["name"]: r["ranges_inclusive"] for r in rb_lit["frameworks"]}
            roles = {k: "external_crystallographic_relative_framework" for k in intervals}
            case = {
                "train_indices": list(range(n)),
                "test_indices": list(range(rb["R2_n_frames"])),
                "test_coords_path": rb["R2_cache"],
                "validation_kind": "independent_simulation_replica_R1_to_R2",
            }
            inv = {
                "case_id": case_id,
                "topology": topology,
                "trajectory": trajectory,
                "heldout_topology": rb["R2_full_topology"]["path"],
                "heldout_trajectory": rb["R2_trajectory"],
                "n_frames": n,
                "n_CA": len(ca),
                "frame_interval_ps": dt,
                "stored_span_ns": 100,
                "state": json.dumps(rb["simulated_state"]),
                "mutations": json.dumps(rb["mutations"]),
                "prealignment": rb["prealignment"],
                "PBC": rb["PBC"],
                "unknowns": "; ".join(rb["limitations"]),
                "metadata_detail": str(run / "sources/3nci_inventory.json"),
            }
            inputs.extend(
                [
                    {"path": rb[k], "sha256": rb[k + "_sha256"]}
                    for k in ["R1_cache", "R2_cache", "R2_trajectory"]
                ]
            )
            doi = rb_lit["doi"]
        regions = {
            name: [any(a <= r["literature_resid"] <= b for a, b in ranges) for r in residues]
            for name, ranges in intervals.items()
        }
        evaluable = regions["portal_combined"] if case_id == "I-FABP" else [True] * len(residues)
        case.update(
            {
                "case_id": case_id,
                "coords_path": str(coords_path.resolve()),
                "frame_interval_ps": dt,
                "residues": residues,
                "regions": regions,
                "region_roles": roles,
                "literature_evaluable": evaluable,
            }
        )
        config["cases"].append(case)
        inv.update(
            {
                "n_train_frames": len(case["train_indices"]),
                "n_test_frames": len(case["test_indices"]),
                "validation_kind": case["validation_kind"],
                "new_frame_selection": "all available chronological frames",
            }
        )
        inventories.append(inv)
        for i, r in enumerate(residues):
            all_residues.append({"case_id": case_id, **r})
            for region, mask in regions.items():
                annotations.append(
                    {
                        "case_id": case_id,
                        **r,
                        "region": region,
                        "member": int(mask[i]),
                        "evaluable_for_metrics": int(evaluable[i]),
                        "annotation_role": roles[region],
                        "source_doi": doi,
                        "unknown_policy": "unannotated residues are not experimental stable negatives"
                        if case_id == "I-FABP"
                        else "domain/framework membership is not absolute rigidity truth",
                    }
                )
        universe.trajectory[0]
        protein = universe.select_atoms("protein")
        if not hasattr(protein, "chainIDs"):
            universe.add_TopologyAttr("chainIDs")
        protein.chainIDs = "A"
        protein.write(str(dest / "reference_full_protein.pdb"))
        write_json(
            dest / "reference_structure_manifest.json",
            {
                "topology": topology,
                "trajectory": trajectory,
                "frame_index": 0,
                "selection": "protein",
                "output_chain_mapping": "A explicitly mapped from source segid/PDB sequence",
                "coordinates": "actual first source frame; no replacement by crystal background",
            },
        )
    write_tsv(run / "case_inventory.tsv", inventories)
    write_tsv(run / "residue_mapping.tsv", all_residues)
    write_tsv(run / "literature_annotations.tsv", annotations)
    config["annotation_sha256"] = sha256(run / "literature_annotations.tsv")
    write_json(run / "analysis_config.json", config)
    (run / "analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    scope = json.loads((run / "task_scope.json").read_text())
    scope["case_frame_policy"] = (
        "ADK/I-FABP chronological first half training, second half heldout; 3nci_A existing independent R1 training/R2 heldout; all available frames"
    )
    scope["split_revision_basis"] = (
        "existing R2 and independent initial velocities confirmed before algorithm analyses"
    )
    write_json(run / "task_scope.json", scope)
    write_json(
        run / "run_manifest.json",
        {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "configuration_sha256": sha256(run / "analysis_config.json"),
            "annotation_sha256": config["annotation_sha256"],
            "inputs": inputs,
            "production_entrypoint": "mdcorealign.benchmark_minimal.solve_minimal_cpu(mode=throughput)",
            "formal_timing_rerun": False,
            "prior_timing_report": "results/atlas_strict_single_thread_cpu_v1/09_report/final_atlas_single_thread_report.md",
        },
    )
    print("Frozen", run / "analysis_config.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    prepare(parser.parse_args().run.resolve())
