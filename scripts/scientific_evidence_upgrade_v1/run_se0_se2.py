"""Freeze evidence, audit reusable assets, and prospectively register SE0-SE2."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from common import (
    B3_MANIFEST,
    REPO_ROOT,
    RESULT_ROOT,
    read_tsv,
    relative,
    sha256_file,
    stable_hash,
    write_json,
    write_tsv,
)

PHASE_DIRS = [
    "00_freeze",
    "01_existing_evidence_audit",
    "02_preregistration",
    "03_synthetic_truth",
    "04_transform_metrics",
    "05_mdcath_rmsf_convention",
    "06_mdcath_rmsf_sensitivity",
    "07_heldout_geometry",
    "08_budget_fidelity",
    "09_literature_audit",
    "10_integrated_results",
    "11_manuscript_materials",
    "handoff_to_chatgpt",
]


def run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def source_manifest() -> None:
    files: list[Path] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    for name in ("pyproject.toml", "README.md"):
        path = REPO_ROOT / name
        if path.exists():
            files.append(path)
    for path in sorted((REPO_ROOT / "dist").glob("*")):
        if path.is_file():
            files.append(path)
    rows = [
        {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "category": (
                "distribution_artifact" if path.parent.name == "dist" else "source"
            ),
        }
        for path in sorted(set(files))
    ]
    write_tsv(
        RESULT_ROOT / "00_freeze/source_manifest.tsv",
        rows,
        ["path", "size_bytes", "sha256", "category"],
    )


def evidence_manifest() -> None:
    candidates = [
        "results/no_materialisation_benchmark_v1/00_freeze/environment_manifest.tsv",
        "results/no_materialisation_benchmark_v1/00_freeze/contract_registry.md",
        "results/no_materialisation_benchmark_v1/06_analysis/final_no_materialisation_decision_report.md",
        "results/no_materialisation_benchmark_v1/06_analysis/mdlovofit_all_domain_no_output_summary.tsv",
        "results/no_materialisation_benchmark_v1/method_agreement_v2/08_reporting/final_method_agreement_report.md",
        "results/database_scale_benchmark_v2/b2_contracts/b3_all_trajectory_manifest_locked.tsv",
        "results/database_scale_benchmark_v2/b3_full_mdcath/b3_database_throughput_summary.tsv",
        "results/database_scale_benchmark_v2/b3_full_mdcath/b3_exact_quality_subset.tsv",
        "results/project_completion_v2_unit_corrected/aggregate/all_domain_unit_corrected_per_domain.tsv",
        "results/atlas_large_long_v1/atlas_manifest.tsv",
        "results/atlas_large_long_v1/atlas_matched_prefix_results.tsv",
        "results/mdcath_benchmark/manifest/mdcath_domain_hierarchy_manifest.tsv",
        "results/mdcath_benchmark/manifest/mdcath_trajectory_manifest.tsv",
        "results/manuscript_writing_bundle/manuscript.md",
    ]
    rows = []
    for name in candidates:
        path = REPO_ROOT / name
        rows.append(
            {
                "path": name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() else "",
                "frozen_use": "read_only_reuse",
            }
        )
    write_tsv(
        RESULT_ROOT / "00_freeze/evidence_manifest.tsv",
        rows,
        ["path", "exists", "size_bytes", "sha256", "frozen_use"],
    )


def environment_manifest() -> None:
    git_code, git_head = run(["git", "rev-parse", "--verify", "HEAD"])
    _, git_status = run(["git", "status", "--short"])
    _, cpu = run(["bash", "-lc", "lscpu | sed -n '1,24p'"])
    gpu_code, gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    blas_code, blas = run(
        [sys.executable, "-c", "import numpy as n; n.show_config()"]
    )
    values = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository_root": ".",
        "package_version_pyproject": "0.1.0rc4",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "MDAnalysis": package_version("MDAnalysis"),
        "gemmi": package_version("gemmi"),
        "cupy": package_version("cupy-cuda12x"),
        "h5py": package_version("h5py"),
        "git_head_status": "valid" if git_code == 0 else "invalid_or_unavailable",
        "git_head": git_head if git_code == 0 else "",
        "git_status_read_only": git_status,
        "cpu_info": cpu,
        "gpu_info": gpu if gpu_code == 0 else "unavailable",
        "blas_lapack": blas if blas_code == 0 else "unavailable",
    }
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
    ):
        values[variable] = os.environ.get(variable, "unset")
    write_tsv(
        RESULT_ROOT / "00_freeze/environment_manifest.tsv",
        ({"key": key, "value": value} for key, value in values.items()),
        ["key", "value"],
    )


def study_scope() -> None:
    (RESULT_ROOT / "00_freeze/study_scope.md").write_text(
        """# Scientific evidence upgrade v1 scope\n\n"
        "This study evaluates the frozen MDCoreAlign 0.1.0rc4 implementation. "
        "No production algorithm, threshold, public default, historical result, "
        "or Git metadata may be changed. New code remains study-only under "
        "`scripts/scientific_evidence_upgrade_v1/`.\n\n"
        "Synthetic trajectories provide **generative truth**, not biological "
        "truth. Real-data RMSF conventions are compared as alternative alignment "
        "frames; neither quantity is called ground truth. MDLovoFit is permitted "
        "only as an untimed scientific comparator on newly registered synthetic "
        "and held-out-geometric panels. Existing external timings are read-only.\n\n"
        "The frozen no-materialisation CPU contract used two BLAS/OpenMP threads. "
        "A single-thread headline is therefore not asserted in this study unless "
        "separately supported by frozen evidence.\n"
        """,
        encoding="utf-8",
    )


def audit_cpu_contract() -> None:
    rows = [
        {
            "panel": "mdCATH_5397_domains",
            "mdcorealign_method": "throughput_cpu",
            "comparator": "stock_MDLovoFit_20.1.1_no_t_no_rmsf",
            "median_speedup": "17.65036055",
            "thread_contract": "2_threads",
            "single_thread_claim": "THREAD_CONTRACT_UNRESOLVED",
            "input_contract": "same_pre_generated_centered_CA_multimodel_PDB",
            "timing_contract": "C1_stock_no_materialisation_workflow",
            "output_suppression": "no_aligned_PDB_no_RMSF",
            "warmup_repeats": "frozen_records; see source evidence",
            "cpu_node": "b004971-dla",
            "process_affinity": "not_recorded",
            "status": "usable_as_frozen_two_thread_paired_evidence",
        },
        {
            "panel": "ATLAS_48_settings",
            "mdcorealign_method": "throughput_cpu",
            "comparator": "stock_MDLovoFit_20.1.1_no_t_no_rmsf",
            "median_speedup": "21.94",
            "thread_contract": "2_threads",
            "single_thread_claim": "THREAD_CONTRACT_UNRESOLVED",
            "input_contract": "same_pre_generated_centered_CA_multimodel_PDB",
            "timing_contract": "C1_stock_no_materialisation_workflow",
            "output_suppression": "no_aligned_PDB_no_RMSF",
            "warmup_repeats": "frozen_records; see source evidence",
            "cpu_node": "b004971-dla",
            "process_affinity": "not_recorded",
            "status": "usable_as_frozen_two_thread_paired_evidence",
        },
    ]
    write_tsv(
        RESULT_ROOT / "01_existing_evidence_audit/cpu_timing_contract.tsv",
        rows,
        list(rows[0]),
    )


def inventory_assets() -> None:
    transform_patterns = [
        "results/no_materialisation_benchmark_v1/method_agreement_v2/**/*.npz",
        "results/no_materialisation_benchmark_v1/method_agreement_v2/**/*.npy",
        "results/atlas_large_long_v1/**/*.npz",
        "results/atlas_large_long_v1/**/*.npy",
        "results/project_completion_v2_unit_corrected/**/*.tsv",
    ]
    transform_rows = []
    for pattern in transform_patterns:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if not path.is_file():
                continue
            name = path.name.lower()
            retained = any(token in name for token in ("transform", "rotation", "translation"))
            transform_rows.append(
                {
                    "path": relative(path),
                    "format": path.suffix,
                    "size_bytes": path.stat().st_size,
                    "likely_contains_transform": retained,
                    "source_coordinates_available": "requires_source_manifest_resolution",
                    "reuse_decision": "inspect_and_reuse" if retained else "summary_only",
                }
            )
    if not transform_rows:
        transform_rows.append(
            {
                "path": "",
                "format": "",
                "size_bytes": "",
                "likely_contains_transform": False,
                "source_coordinates_available": "unknown",
                "reuse_decision": "no_direct_transform_array_identified",
            }
        )
    write_tsv(
        RESULT_ROOT / "01_existing_evidence_audit/transform_inventory.tsv",
        transform_rows,
        list(transform_rows[0]),
    )

    rmsf_rows = [
        {
            "asset": "mdCATH_bundled_RMSF",
            "path_pattern": "<H5>/<domain>/<temperature>/<replicate>/rmsf",
            "residue_mapping": "embedded_pdbProteinAtoms_CA_order",
            "stored_unit_attribute": "absent",
            "working_unit_status": "requires_SE5_source_and_reproduction_audit",
            "reuse_decision": "pair_after_SE5_gate",
        },
        {
            "asset": "mdCATH_bundled_RMSD",
            "path_pattern": "<H5>/<domain>/<temperature>/<replicate>/rmsd",
            "residue_mapping": "trajectory_level",
            "stored_unit_attribute": "nm",
            "working_unit_status": "explicit_nm",
            "reuse_decision": "selection_covariate_only",
        },
        {
            "asset": "historical_MDCoreAlign_profiles",
            "path_pattern": "results/mdcath_benchmark/**/residue_profile.tsv",
            "residue_mapping": "per-output manifest",
            "stored_unit_attribute": "historical_tables_A_after_unit_audit",
            "working_unit_status": "reuse_only_when_hash_and_mapping_match",
            "reuse_decision": "prefer_recompute_from_same_source_for_SE6",
        },
        {
            "asset": "ATLAS_CA_coordinate_cache",
            "path_pattern": "results/atlas_large_long_v1/input_cache/*_CA_A_float64.npy",
            "residue_mapping": "paired_residue_metadata_npz",
            "stored_unit_attribute": "angstrom_by_locked_manifest",
            "working_unit_status": "available_for_SE7_SE8",
            "reuse_decision": "read_only_reuse",
        },
    ]
    write_tsv(
        RESULT_ROOT / "01_existing_evidence_audit/rmsf_pairing_inventory.tsv",
        rmsf_rows,
        list(rmsf_rows[0]),
    )

    (RESULT_ROOT / "01_existing_evidence_audit/reuse_decision.md").write_text(
        """# Reuse decision\n\n"
        "Frozen C1 timing values are reused without rerunning external tools. "
        "They support a paired **two-thread** CPU workflow comparison; the "
        "single-thread contract is unresolved. Existing method-agreement and "
        "ATLAS coordinate assets are reused when hashes and mappings are explicit. "
        "mdCATH RMSF arrays are not interpreted until the source-code and numerical "
        "reproduction gate in SE5 passes. New results are written only below the "
        "scientific-evidence-upgrade root.\n"
        """,
        encoding="utf-8",
    )


def select_synthetic_templates(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    excluded = {"3nci_A", "6sup_A", "3nciA00", "6supA00"}
    by_domain = {
        row["domain_id"]: row
        for row in rows
        if row["locally_available"] == "True"
        and 60 <= int(row["expected_n_residues"]) <= 500
        and row["domain_id"] not in excluded
    }
    eligible = list(by_domain.values())
    bins = [(60, 100), (101, 180), (181, 300), (301, 500)]
    targets = {
        "development": [("1", 0), ("2", 1), ("3", 2), ("4", 0), ("1", 3)],
        "heldout": [("2", 0), ("3", 1), ("1", 2), ("6", 0), ("2", 3)],
    }
    selected: list[dict[str, str]] = []
    used_superfamilies: set[str] = set()
    output: list[dict[str, object]] = []
    for split, split_targets in targets.items():
        for cath_class, bin_index in split_targets:
            low, high = bins[bin_index]
            candidates = sorted(
                (
                    row
                    for row in eligible
                    if row["cath_class"] == cath_class
                    and low <= int(row["expected_n_residues"]) <= high
                ),
                key=lambda row: stable_hash(f"synthetic:{split}:{row['domain_id']}"),
            )
            row = next(
                (
                    candidate
                    for candidate in candidates
                    if ".".join(
                        [
                            candidate["cath_class"],
                            candidate["cath_architecture"],
                            candidate["cath_topology"],
                            candidate["cath_homologous_superfamily"],
                        ]
                    )
                    not in used_superfamilies
                ),
                None,
            )
            if row is None:
                raise RuntimeError(f"could not satisfy synthetic stratum {split} {cath_class}/{bin_index}")
            family = ".".join(
                [
                    row["cath_class"],
                    row["cath_architecture"],
                    row["cath_topology"],
                    row["cath_homologous_superfamily"],
                ]
            )
            used_superfamilies.add(family)
            selected.append(row)
            output.append(
                {
                    "domain_id": row["domain_id"],
                    "trajectory_id": row["trajectory_id"],
                    "split": split,
                    "n_residues": int(row["expected_n_residues"]),
                    "cath_code": family,
                    "domain_annotation": "single_CATH_domain",
                    "source_type": "mdCATH_embedded_experimental_template",
                    "source_file": row["source_file"],
                    "source_sha256": row["source_sha256"],
                    "selection_basis": "metadata_only_CATH_class_length_stratum_hash_order",
                }
            )

    atlas_manifest = read_tsv(REPO_ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv")
    atlas_by_system = {row["system"]: row for row in atlas_manifest}
    atlas_specs = [
        (
            "1jnr_C",
            "development",
            "three_CATH_domains_on_chain_C; 1.20.58.100, 3.50.50.60, 3.90.700.10",
            "https://www.rcsb.org/annotations/1JNR",
        ),
        (
            "1r6w_A",
            "heldout",
            "two_CATH_domains_on_chain_A; 1r6wA01 and 1r6wA02",
            "https://www.cathdb.info/pdb/1r6w",
        ),
    ]
    for system, split, annotation, annotation_source in atlas_specs:
        row = atlas_by_system[system]
        output.append(
            {
                "domain_id": system,
                "trajectory_id": row["trajectory_id"],
                "split": split,
                "n_residues": int(row["n_CA"]),
                "cath_code": "multi_domain_chain",
                "domain_annotation": annotation,
                "domain_annotation_source": annotation_source,
                "source_type": "ATLAS_R1_first_frame_experimental_template",
                "source_file": row["coordinate_cache_path"],
                "source_sha256": row["coordinate_cache_sha256"],
                "selection_basis": "metadata_only_locked_local_ATLAS_multidomain_chain",
            }
        )
    output.sort(key=lambda row: (str(row["split"]), stable_hash(str(row["domain_id"]))))
    if sum(row["split"] == "development" for row in output) != 6:
        raise RuntimeError("development split must contain six templates")
    if sum(row["split"] == "heldout" for row in output) != 6:
        raise RuntimeError("heldout split must contain six templates")
    return output


def register_analyses() -> None:
    b3_rows = read_tsv(B3_MANIFEST)
    canonical = [row for row in b3_rows if row["locally_available"] == "True"]
    templates = select_synthetic_templates(canonical)
    common = {
        "registry_version": "scientific-evidence-upgrade-v1",
        "locked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "modification_policy": "immutable_after_hash; exploratory work must use a separate root",
        "core_fraction": 0.60,
        "core_cardinality_rule": "floor(0.60*N), deterministic index tie break",
        "bootstrap": {
            "unit": "structural_template_or_domain_family",
            "replicates": 2000,
            "seed": 20260831,
            "confidence_interval": 0.95,
        },
        "missing_data": "retain and report failures; no outcome-dependent deletion",
    }
    synthetic = {
        **common,
        "templates": templates,
        "frames": 500,
        "seeds": [1701, 1702, 1703],
        "global_rotation": "uniform_random_unit_quaternion_with_frame0_identity",
        "global_translation_A": [0.0, 30.0],
        "local_amplitudes_A": [0.25, 0.75, 1.5],
        "hinge_amplitudes_deg": [5.0, 15.0, 30.0],
        "scenarios": ["S0", "S1", "S2", "S3", "S4", "S5", "S6"],
        "methods": [
            "oracle_true_core_Kabsch",
            "ordinary_full_CA_Kabsch",
            "MDCoreAlign_sequential_reference",
            "MDCoreAlign_throughput_CPU",
            "MDLovoFit_true_fraction_untimed",
        ],
        "gauge": "row-vector input-to-reference; frame0 identity and zero translation",
        "failure": "nonfinite output, invalid transform, empty core, process failure, or mapping mismatch",
        "figure_rule": "all heldout template medians plus scenario distributions; no best-case selection",
    }
    write_json(
        RESULT_ROOT / "02_preregistration/synthetic_truth_registry.json",
        synthetic,
    )

    domain_ids = sorted({row["domain_id"] for row in canonical})
    domain_order = sorted(domain_ids, key=lambda item: stable_hash("rmsf:" + item))
    development_count = max(1, int(np.floor(0.10 * len(domain_order))))
    rmsf_registry = {
        **common,
        "source_panel": "corrected_5397_domain_canonical_mdCATH_panel",
        "canonical_trajectory": "320K replicate 1 for database-scale analysis",
        "domain_split": {
            "development_domains": domain_order[:development_count],
            "validation_domains": domain_order[development_count:],
            "split_key": "SHA256('rmsf:' + domain_id)",
        },
        "audit_panel_rule": (
            "100 trajectories selected before comparison by deterministic strata of "
            "domain length, CATH class, temperature, replicate, bundled mean RMSD, "
            "and bundled mean RMSF"
        ),
        "audit_temperatures_K": [320, 348, 379, 413, 450],
        "audit_replicates_one_based": [1, 2, 3, 4, 5],
        "reproduction_tolerance_A": {"max_abs": 1e-4, "median_abs": 1e-5},
        "full_analysis_stop_rule": "stop SE6 if residue order/unit or tolerance gate fails",
        "sensitivity_metrics": [
            "signed_mean_difference_A",
            "MAE_A",
            "RMSE_A",
            "normalised_MAE",
            "Pearson",
            "Spearman",
            "top10_flexible_Jaccard",
            "top20_flexible_Jaccard",
            "bottom10_rigid_Jaccard",
            "flexible_quantile_change_fraction",
            "maximum_local_absolute_change_A",
            "peak_index_shift",
            "alignment_sensitivity_index",
        ],
        "narrative_thresholds": {
            "strong": "validation domain median ASI>=0.20 and top20 Jaccard<=0.70 with prevalence>=0.50",
            "moderate": "median ASI>=0.10 or top20 Jaccard<=0.85 with prevalence>=0.25",
            "limited": "otherwise",
        },
        "case_rule": "three highest, three nearest median, three lowest validation-domain sensitivity",
    }
    write_json(
        RESULT_ROOT / "02_preregistration/rmsf_sensitivity_registry.json",
        rmsf_registry,
    )

    geometry_order = sorted(domain_ids, key=lambda item: stable_hash("geometry:" + item))
    geometry_registry = {
        **common,
        "mdcath_domains": geometry_order[:512],
        "mdcath_trajectory": "320K replicate 1",
        "atlas_families": [
            "1jnr_C",
            "1r6w_A",
            "1y57_B",
            "2c7b_A",
            "2j5e_A",
            "3d3e_A",
            "4b2a_A",
            "5g53_A",
        ],
        "folds": "four contiguous temporal folds; train on three, test on one",
        "secondary_split": "deterministic interleaved four-fold",
        "methods": [
            "MDCoreAlign_throughput",
            "MDCoreAlign_sequential_reference",
            "MDLovoFit_untimed",
            "random_equal_cardinality",
            "full_selection_noncore_baseline",
        ],
        "pair_sampling": "all pairs if <=10000; otherwise SHA256-ordered deterministic 10000 pairs",
        "contact_threshold_CA_A": 8.0,
        "random_control_seed": 3407,
        "failure": "input/mapping/process/nonfinite/empty-core failure retained in atlas",
    }
    write_json(
        RESULT_ROOT / "02_preregistration/heldout_geometry_registry.json",
        geometry_registry,
    )

    budget_dev = sorted(domain_ids, key=lambda item: stable_hash("budget-dev:" + item))[:128]
    budget_remaining = [item for item in domain_ids if item not in set(budget_dev)]
    budget_validation = sorted(
        budget_remaining,
        key=lambda item: stable_hash("budget-val:" + item),
    )[:384]
    budget_registry = {
        **common,
        "development_domains": budget_dev,
        "validation_domains": budget_validation,
        "synthetic_development_templates": [
            row["domain_id"] for row in templates if row["split"] == "development"
        ],
        "synthetic_validation_templates": [
            row["domain_id"] for row in templates if row["split"] == "heldout"
        ],
        "K_values": [16, 32, 64, 128, 256],
        "M_values": [1, 2, 4, 8],
        "locked_validation_configs": [
            {"name": "lower_1", "K": 32, "M": 2},
            {"name": "lower_2", "K": 64, "M": 2},
            {"name": "production", "K": 256, "M": 4},
            {"name": "higher", "K": 256, "M": 8},
        ],
        "frame_scaling_T": [250, 1000, 5000, 10000, 50000, 100000],
        "residue_strata_N": [75, 150, 300, 500],
        "timing_contract": "loaded_coordinates_CPU_one_thread_no_materialisation",
        "thread_environment": {
            "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
        "fidelity_metrics": [
            "core_Jaccard",
            "AUPRC_synthetic",
            "weight_Spearman",
            "rotation_geodesic_deg_synthetic",
            "core_coordinate_error_A_synthetic",
            "RMSD_error_A",
            "RMSF_MAE_A",
            "RMSF_Spearman",
            "tail_rate",
        ],
        "pareto_rule": "non-dominated median runtime and fidelity on development only",
    }
    write_json(
        RESULT_ROOT / "02_preregistration/budget_fidelity_registry.json",
        budget_registry,
    )

    (RESULT_ROOT / "02_preregistration/metric_definitions.md").write_text(
        """# Registered metric definitions\n\n"
        "All coordinates and geometric errors are reported in angstrom. Rotation "
        "error is the SO(3) geodesic angle from `acos(clip((trace(Rhat Rtrue^T)-1)/2))`. "
        "Core-coordinate error is the RMS coordinate difference between estimated "
        "and true mapped coordinates on the known synthetic core. Binary masks use "
        "exactly `floor(0.60*N)` residues unless a synthetic scenario supplies its "
        "known cardinality. RMSF is `sqrt(sum_xyz mean_t((x_t-mean_t x)^2))`. "
        "Alignment-sensitivity index is the residue sum of absolute RMSF differences "
        "divided by the conventional RMSF sum plus `1e-12`. Held-out pair-distance "
        "metrics require no rigid fit. Bootstrap resampling is clustered by template, "
        "domain, or ATLAS family, never by frame.\n\n"
        "Synthetic truth is generative truth only. Dataset-provided conventional "
        "RMSF and stable-reference RMSF are alternative observables, not biological "
        "ground truth. Sequential output is a numerical reference, not biological "
        "truth.\n"
        """,
        encoding="utf-8",
    )

    registry_paths = sorted((RESULT_ROOT / "02_preregistration").glob("*"))
    digest_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in registry_paths
        if path.name != "analysis_plan.sha256" and path.is_file()
    ]
    combined = "\n".join(digest_lines) + "\n"
    aggregate = stable_hash(combined)
    (RESULT_ROOT / "02_preregistration/analysis_plan.sha256").write_text(
        combined + f"{aggregate}  COMBINED_REGISTRATION\n",
        encoding="utf-8",
    )


def main() -> None:
    for directory in PHASE_DIRS:
        (RESULT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    source_manifest()
    evidence_manifest()
    environment_manifest()
    study_scope()
    audit_cpu_contract()
    inventory_assets()
    register_analyses()
    print(json.dumps({"status": "SE0_SE2_COMPLETE", "root": relative(RESULT_ROOT)}))


if __name__ == "__main__":
    main()
