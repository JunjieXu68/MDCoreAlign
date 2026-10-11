#!/usr/bin/env python3
"""Freeze U6/GPU V1 and build the locked D1 mdCATH registry."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from scripts.database_scale_gpu_v1.common import (
    CORRECTED_TABLE,
    D0_ROOT,
    EXPECTED_ELIGIBLE,
    EXPECTED_REGISTERED,
    MANIFEST_PATH,
    MISSING_DOMAIN_ID,
    QUALITY_GATES_PATH,
    REGISTRY_PATH,
    RESULT_ROOT,
    ROOT,
    SNAPSHOT_REVISION,
    cache_blob_sha256,
    corrected_profile_path,
    load_exact_profile,
    mapping_digest,
    read_tsv,
    repository_path,
    sha256,
    source_h5,
    stable_hash,
    write_json,
    write_tsv,
)
from scripts.scaling_optimization_v4.finalize_phase6 import verify_hashes


FROZEN_ROOTS = (
    ROOT / "results/throughput_gpu_v1",
    ROOT / "results/gpu_matched_validation_v1",
)
SOURCE_FILES = (
    ROOT / "src/mdcorealign/gpu_backend.py",
    ROOT / "src/mdcorealign/contracts.py",
    ROOT / "src/mdcorealign/experimental.py",
)


def tree_manifest(root: Path) -> list[dict[str, object]]:
    """Hash every regular file under one frozen root."""

    rows: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "frozen_root": repository_path(root),
                "path": repository_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def aggregate_manifest(rows: list[dict[str, object]]) -> str:
    """Hash a deterministic frozen-file manifest."""

    payload = "\n".join(
        f"{row['path']}\t{row['size_bytes']}\t{row['sha256']}" for row in rows
    )
    return stable_hash(payload)


def command_output(command: list[str]) -> dict[str, object]:
    """Capture a short environment command without failing D0."""

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "return_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def mapping_from_profile(
    trajectory_id: str,
) -> tuple[Path, dict[str, object], str | None]:
    """Load and validate one exact profile for registry construction."""

    try:
        profile_path = corrected_profile_path(trajectory_id)
        profile = load_exact_profile(profile_path)
        return profile_path, profile, None
    except Exception as exc:
        return (
            Path(""),
            {},
            f"{type(exc).__name__}: {exc}",
        )


def build_registry() -> list[dict[str, object]]:
    """Build the immutable 5,398-domain canonical registry."""

    corrected_rows = read_tsv(CORRECTED_TABLE)
    manifest_rows = read_tsv(MANIFEST_PATH)
    if len(corrected_rows) != EXPECTED_REGISTERED:
        raise RuntimeError(
            f"corrected table has {len(corrected_rows)}/{EXPECTED_REGISTERED}"
        )
    if len(manifest_rows) != EXPECTED_REGISTERED:
        raise RuntimeError(
            f"canonical manifest has {len(manifest_rows)}/{EXPECTED_REGISTERED}"
        )
    corrected = {row["domain_id"]: row for row in corrected_rows}
    manifest = {row["domain_id"]: row for row in manifest_rows}
    if len(corrected) != EXPECTED_REGISTERED or corrected.keys() != manifest.keys():
        raise RuntimeError("corrected table and canonical manifest membership differ")

    preliminary: list[dict[str, object]] = []
    for domain_id in sorted(corrected):
        row = corrected[domain_id]
        manifest_row = manifest[domain_id]
        trajectory_id = f"{domain_id}__T320__R1"
        h5_path = source_h5(domain_id)
        available = h5_path.is_file()
        profile_path = Path("")
        profile: dict[str, object] = {}
        mapping_error: str | None = None
        if available and row["mdcorealign_status"] == "success":
            profile_path, profile, mapping_error = mapping_from_profile(
                trajectory_id
            )
        expected_missing = domain_id == MISSING_DOMAIN_ID
        source_hash = ""
        source_hash_status = "unavailable"
        source_size = 0
        if available:
            try:
                source_hash = cache_blob_sha256(h5_path)
                source_hash_status = (
                    "hugging_face_content_addressed_blob_sha256"
                )
                source_size = h5_path.stat().st_size
            except Exception as exc:
                source_hash_status = f"failed:{type(exc).__name__}:{exc}"

        mapping = profile.get("mapping", [])
        n_profile = len(mapping)
        n_residues = (
            int(row["n_residues"])
            if row["n_residues"] not in {"", "NA"}
            else None
        )
        n_frames = (
            int(row["n_frames"])
            if row["n_frames"] not in {"", "NA"}
            else None
        )
        mapping_ok = bool(
            available
            and mapping_error is None
            and n_residues is not None
            and n_profile == n_residues
        )
        unit_ok = bool(
            available
            and row.get("source_unit", "").lower() == "angstrom"
            and row.get("conversion_factor_to_angstrom") == "1.0"
            and row.get("canonical_internal_unit", "").lower() == "angstrom"
        )
        eligible = bool(
            available
            and row["mdcorealign_status"] == "success"
            and source_hash_status
            == "hugging_face_content_addressed_blob_sha256"
            and mapping_ok
            and unit_ok
        )
        if eligible:
            registry_status = "eligible"
        elif expected_missing and not available:
            registry_status = "explicitly_unavailable"
        else:
            registry_status = "blocked_integrity"

        chains = [item[0] for item in mapping]
        effective_core = (
            float(
                np.sum(
                    np.asarray(profile["weights"], dtype=np.float64)
                    / max(float(np.max(profile["weights"])), 1e-12)
                )
                / n_profile
            )
            if n_profile
            else None
        )
        preliminary.append(
            {
                "dataset": "mdCATH",
                "dataset_release": (
                    f"HuggingFace compsciencelab/mdCATH@{SNAPSHOT_REVISION}"
                ),
                "system_id": domain_id,
                "trajectory_id": trajectory_id,
                "source_file": repository_path(h5_path),
                "source_sha256": source_hash,
                "source_sha256_status": source_hash_status,
                "topology_file": (
                    f"{repository_path(h5_path)}::/{domain_id}/pdb"
                ),
                "topology_sha256": source_hash,
                "topology_storage": "embedded_in_same_content_addressed_HDF5",
                "coordinate_path": f"/{domain_id}/320/0/coords",
                "source_coordinate_unit": row.get("source_unit", ""),
                "conversion_factor_to_angstrom": row.get(
                    "conversion_factor_to_angstrom", ""
                ),
                "canonical_internal_unit": row.get(
                    "canonical_internal_unit", ""
                ),
                "unit_provenance": row.get("unit_correction_status", ""),
                "chain_residue_mapping_source": (
                    repository_path(profile_path) if n_profile else ""
                ),
                "chain_residue_mapping_sha256": (
                    mapping_digest(mapping) if n_profile else ""
                ),
                "exact_profile_sha256": (
                    sha256(profile_path) if n_profile else ""
                ),
                "chain_count": len(set(chains)) if chains else "",
                "n_residues": n_residues if n_residues is not None else "",
                "n_frames": n_frames if n_frames is not None else "",
                "frame_interval": "unavailable_not_encoded_in_local_HDF5_metadata",
                "replicate": "R1",
                "replica_index_zero_based": 0,
                "temperature_K": 320,
                "trajectory_continuity": (
                    "single_mdCATH_simulation_replicate_not_stitched"
                ),
                "synthetic_status": "biological_simulation_not_synthetic_stitch",
                "expected_output_contract": (
                    "analysis_metrics_transforms_RMSD_RMSF_plus_W5_diagnostic"
                ),
                "cath_class": row.get("cath_class", ""),
                "cath_architecture": row.get("cath_architecture", ""),
                "cath_topology": row.get("cath_topology", ""),
                "cath_homologous_superfamily": row.get(
                    "cath_homologous_superfamily", ""
                ),
                "fixed_core_fraction": row.get("stable_core_fraction", ""),
                "effective_core_fraction": (
                    effective_core if effective_core is not None else ""
                ),
                "motion_proxy_noncore_core_RMSF_ratio": row.get(
                    "noncore_core_RMSF_ratio", ""
                ),
                "local_available": available,
                "source_storage_bytes": source_size,
                "mapping_integrity": (
                    "pass" if mapping_ok else mapping_error or "unavailable"
                ),
                "unit_integrity": "pass" if unit_ok else "unresolved",
                "registry_status": registry_status,
                "unavailable_reason": (
                    row.get("failure_reason", "")
                    if registry_status == "explicitly_unavailable"
                    else ""
                ),
                "canonical_selection_rule": (
                    "prefer_320K_R1_fixed_before_D1_per_historical_formal_panel"
                ),
                "selection_rule_inspected_performance": False,
                "processing_sort_key": (
                    f"{n_residues or 999999:06d}:{trajectory_id}"
                ),
                "cold_start_subset": False,
            }
        )

    eligible_rows = [
        row for row in preliminary if row["registry_status"] == "eligible"
    ]
    cold_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in eligible_rows:
        n_residues = int(row["n_residues"])
        size_bin = (
            "small"
            if n_residues <= 100
            else "medium"
            if n_residues <= 200
            else "large"
        )
        key = (str(row["cath_class"]), size_bin)
        cold_groups.setdefault(key, []).append(row)
    selected: list[dict[str, object]] = []
    for key in sorted(cold_groups):
        selected.append(
            min(
                cold_groups[key],
                key=lambda item: stable_hash(
                    f"d1-cold-v1::{item['trajectory_id']}"
                ),
            )
        )
    selected = sorted(
        selected,
        key=lambda item: stable_hash(f"d1-cold-cap::{item['trajectory_id']}"),
    )[:12]
    selected_ids = {row["trajectory_id"] for row in selected}
    for row in preliminary:
        row["cold_start_subset"] = row["trajectory_id"] in selected_ids
    return preliminary


def environment_payload() -> dict[str, object]:
    """Record the exact D0-D1 execution environment."""

    import cupy
    import h5py
    import numpy
    import scipy

    git_status = command_output(["git", "status", "--short", "--branch"])
    git_head = command_output(["git", "rev-parse", "HEAD"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "h5py_version": h5py.__version__,
        "cupy_version": cupy.__version__,
        "cuda_device_count": cupy.cuda.runtime.getDeviceCount(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONPATH",
            )
        },
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
        "git_status": git_status,
        "git_head": git_head,
        "git_metadata_modified": False,
        "note": (
            "h5py 3.16.0 was added to the isolated GPU environment for D1 "
            "HDF5 decoding; U6 result trees were not modified."
        ),
    }


def locked_quality_gates() -> dict[str, object]:
    """Return the D1 gates locked before any all-domain GPU evaluation."""

    return {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_D1_final_run": True,
        "post_hoc_threshold_tuning_allowed": False,
        "scope": "system-level corrected 320K R1 mdCATH domains",
        "exact_reference": (
            "unit-corrected sequential_lean mean weights reused from frozen "
            "profiles; matched fixed-weight all-frame fit recomputed"
        ),
        "required": {
            "registered_domain_count": EXPECTED_REGISTERED,
            "eligible_domain_count": EXPECTED_ELIGIBLE,
            "explicitly_unavailable_domain_count": 1,
            "eligible_processing_failure_count_max": 0,
            "unit_integrity_failure_count_max": 0,
            "mapping_integrity_failure_count_max": 0,
            "numerical_failure_count_max": 0,
            "catastrophic_quality_failure_count_max": 0,
            "minimum_core_jaccard_vs_exact": 0.5,
            "cpu_gpu_minimum_weight_spearman": 0.9999,
            "cpu_gpu_minimum_RMSF_spearman": 0.9999,
            "cpu_gpu_maximum_median_RMSD_difference_A": 0.001,
            "cpu_gpu_core_rule": (
                "exact mask or Jaccard>=0.95 with max weight delta<=1e-4 "
                "and cutoff margin<=1e-4"
            ),
            "w5_cpu_gpu_minimum_local_deformation_spearman": 0.999,
            "report_system_level_distributions": True,
        },
        "catastrophic_definition": (
            "non-finite scientific output, core Jaccard below 0.5, or median "
            "absolute per-frame RMSD difference above 1.0 A"
        ),
        "interpretation": (
            "These gates test integrity and catastrophic failure, not "
            "transform-identical parity with sequential_lean."
        ),
    }


def main() -> None:
    """Run D0 without changing any frozen evidence."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    frozen_rows: list[dict[str, object]] = []
    for root in FROZEN_ROOTS:
        frozen_rows.extend(tree_manifest(root))
    write_tsv(D0_ROOT / "frozen_evidence_manifest.tsv", frozen_rows)
    source_rows = [
        {
            "path": repository_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in SOURCE_FILES
    ]
    write_tsv(D0_ROOT / "gpu_v1_source_manifest.tsv", source_rows)

    registry = build_registry()
    write_tsv(REGISTRY_PATH, registry)
    write_json(QUALITY_GATES_PATH, locked_quality_gates())
    write_json(D0_ROOT / "environment.json", environment_payload())
    frozen_integrity = verify_hashes()
    write_json(D0_ROOT / "legacy_frozen_integrity.json", frozen_integrity)

    counts = {
        status: sum(row["registry_status"] == status for row in registry)
        for status in (
            "eligible",
            "explicitly_unavailable",
            "blocked_integrity",
        )
    }
    status = "pass" if (
        len(registry) == EXPECTED_REGISTERED
        and counts["eligible"] == EXPECTED_ELIGIBLE
        and counts["explicitly_unavailable"] == 1
        and counts["blocked_integrity"] == 0
        and frozen_integrity["status"] == "pass"
    ) else "fail"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "D0",
        "status": status,
        "registered_domains": len(registry),
        "registry_counts": counts,
        "missing_domain": MISSING_DOMAIN_ID,
        "missing_domain_interpretation": (
            "retained explicit unavailable input after historical "
            "Hugging Face/Xet download stall; not an algorithm failure"
        ),
        "frozen_file_count": len(frozen_rows),
        "frozen_aggregate_sha256": aggregate_manifest(frozen_rows),
        "frozen_roots": [repository_path(path) for path in FROZEN_ROOTS],
        "legacy_frozen_integrity": frozen_integrity,
        "public_default": "sequential_lean",
        "public_default_changed": False,
        "historical_outputs_modified": False,
        "git_metadata_modified": False,
        "D1_authorized": status == "pass",
        "D2_authorized": False,
    }
    write_json(D0_ROOT / "d0_status.json", summary)
    report = f"""# D0 database-scale freeze and registry

- Status: **{status.upper()}**
- Registered mdCATH domains: **{len(registry):,}**
- Locally eligible 320 K/R1 domains: **{counts['eligible']:,}**
- Explicitly unavailable domains: **{counts['explicitly_unavailable']}**
- Integrity-blocked available domains: **{counts['blocked_integrity']}**
- Frozen U6/GPU V1 files: **{len(frozen_rows)}**
- Legacy project/golden integrity: **{frozen_integrity['status'].upper()}**
- Public exact/default path: `sequential_lean` (unchanged)

`{MISSING_DOMAIN_ID}` is retained in the registry as explicitly unavailable
because its Hugging Face/Xet download stalled in the historical production
run. It is not counted as an algorithm failure.

The canonical biological selection was fixed before D1 as 320 K, replicate R1,
one non-stitched mdCATH simulation trajectory per domain. HDF5 files are
identified by their content-addressed Hugging Face SHA-256 blob names. Topology
and residue mapping are embedded in the same HDF5 and cross-referenced to the
locked exact residue profile.

The isolated GPU environment required one environment-only addition:
`h5py==3.16.0`. No algorithm, historical result, public default, golden
reference, or Git metadata was changed.
"""
    (D0_ROOT / "d0_freeze_report.md").write_text(report)
    if status != "pass":
        raise SystemExit("D0 integrity gate failed; D1 is not authorized")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
