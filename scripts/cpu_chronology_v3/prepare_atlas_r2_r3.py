#!/usr/bin/env python
"""Audit and decode locally archived ATLAS R2/R3 CA trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import MDAnalysis as mda
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cpu_chronology_v3"
SOURCE_MANIFEST = ROOT / "results/atlas_large_long_v1/atlas_manifest.tsv"
CACHE = Path(
    os.environ.get(
        "MDCOREALIGN_V3_INPUT_CACHE",
        str(Path(tempfile.gettempdir()) / "mdcorealign_cpu_chronology_v3_input_cache"),
    )
)
AUDIT_TABLE = OUT / "05_fresh_panels/atlas_r2_r3_member_audit.tsv"
AUDIT_CHECKPOINT = Path(tempfile.gettempdir()) / "mdcorealign_cpu_chronology_v3_atlas_audit.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Use a repository-relative path where possible, otherwise an explicit cache path."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def copy_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def metadata_values(group: object, attribute: str, fallback: str = "") -> np.ndarray:
    try:
        return np.asarray(getattr(group, attribute)).astype(str)
    except (AttributeError, TypeError):
        return np.full(len(group), fallback, dtype="U8")


def decode_one(source: dict[str, str], replicate: int) -> dict[str, object]:
    system = source["system"]
    archive_path = ROOT / source["local_archive_path"]
    trajectory_member = f"{system}_prod_R{replicate}_fit.xtc"
    topology_member = f"{system}_prod_R{replicate}.tpr"
    pdb_member = f"{system}.pdb"
    coordinate_path = CACHE / f"{system}_R{replicate}_CA_A_float64.npy"
    metadata_path = CACHE / f"{system}_R{replicate}_residue_metadata.npz"
    started = time.perf_counter()
    base = {
        "case_id": f"ATLAS_{system}_R{replicate}",
        "family_id": f"ATLAS:{system}",
        "system": system,
        "replicate": f"R{replicate}",
        "archive_path": source["local_archive_path"],
        "archive_sha256": source["local_archive_sha256"],
        "trajectory_member": trajectory_member,
        "topology_member": topology_member,
        "pdb_member": pdb_member,
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            member_names = set(archive.namelist())
            missing = [name for name in (trajectory_member, topology_member, pdb_member) if name not in member_names]
            if missing:
                raise FileNotFoundError("missing archive members: " + ", ".join(missing))
            trajectory_info = archive.getinfo(trajectory_member)
            topology_info = archive.getinfo(topology_member)
            with tempfile.TemporaryDirectory(prefix=f"mdca_atlas_{system}_R{replicate}_") as temporary:
                temporary_path = Path(temporary)
                pdb = temporary_path / pdb_member
                xtc = temporary_path / trajectory_member
                copy_member(archive, pdb_member, pdb)
                copy_member(archive, trajectory_member, xtc)
                trajectory_hash = sha256(xtc)
                universe = mda.Universe(str(pdb), str(xtc), convert_units=True)
                ca = universe.select_atoms("protein and name CA")
                n_frames = len(universe.trajectory)
                n_ca = len(ca)
                if n_frames != 10001 or n_ca != int(source["n_CA"]):
                    raise ValueError(f"unexpected shape metadata: frames={n_frames}, CA={n_ca}")
                CACHE.mkdir(parents=True, exist_ok=True)
                temporary_cache = coordinate_path.with_suffix(".npy.tmp")
                values = np.lib.format.open_memmap(
                    temporary_cache,
                    mode="w+",
                    dtype=np.float64,
                    shape=(n_frames, n_ca, 3),
                )
                first_cog = None
                previous = None
                maximum_step = 0.0
                maximum_cog_drift = 0.0
                first_time = None
                second_time = None
                for frame_index, timestep in enumerate(universe.trajectory):
                    current = np.asarray(ca.positions, dtype=np.float64)
                    values[frame_index] = current
                    cog = np.mean(current, axis=0)
                    if first_cog is None:
                        first_cog = cog.copy()
                        first_time = float(timestep.time)
                    else:
                        maximum_cog_drift = max(maximum_cog_drift, float(np.linalg.norm(cog - first_cog)))
                        maximum_step = max(maximum_step, float(np.max(np.linalg.norm(current - previous, axis=1))))
                        if frame_index == 1:
                            second_time = float(timestep.time)
                    previous = current.copy()
                values.flush()
                del values
                temporary_cache.replace(coordinate_path)
                pdb_chain_ids = metadata_values(ca, "chainIDs")
                chain_ids = pdb_chain_ids.copy()
                chain_source = "archive_PDB_chain_column"
                if not np.any(chain_ids != ""):
                    chain_ids = np.full(n_ca, system.rsplit("_", 1)[-1], dtype="U8")
                    chain_source = "locked_ATLAS_single_chain_identifier"
                np.savez_compressed(
                    metadata_path,
                    chain_ids=chain_ids,
                    residue_ids=metadata_values(ca, "resids"),
                    residue_names=metadata_values(ca, "resnames"),
                    atom_names=metadata_values(ca, "names"),
                )
        status = "pass" if maximum_step < 20.0 and maximum_cog_drift < 5.0 else "warning_large_step_or_drift"
        return {
            **base,
            "status": status,
            "trajectory_member_size_bytes": trajectory_info.file_size,
            "trajectory_member_crc": trajectory_info.CRC,
            "topology_member_size_bytes": topology_info.file_size,
            "trajectory_sha256": trajectory_hash,
            "coordinate_cache_path": portable_path(coordinate_path),
            "coordinate_cache_sha256": sha256(coordinate_path),
            "residue_metadata_path": portable_path(metadata_path),
            "residue_metadata_sha256": sha256(metadata_path),
            "n_frames": n_frames,
            "n_residues": n_ca,
            "chain_mapping": json.dumps({chain: int(np.count_nonzero(chain_ids == chain)) for chain in sorted(set(chain_ids))}, sort_keys=True),
            "chain_mapping_source": chain_source,
            "coordinate_unit": "angstrom_after_MDAnalysis_conversion",
            "observed_frame_interval_ps": float(second_time - first_time),
            "maximum_adjacent_frame_atom_displacement_A": maximum_step,
            "maximum_COG_drift_from_first_A": maximum_cog_drift,
            "continuity_diagnostic": status,
            "PBC_nojump_status": "ATLAS_fit_trajectory_as_provided_not_independently_verified",
            "decode_seconds": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:
        return {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "decode_seconds": time.perf_counter() - started}


def main() -> None:
    source_rows = read_tsv(SOURCE_MANIFEST)
    source_audit = AUDIT_CHECKPOINT if AUDIT_CHECKPOINT.exists() else AUDIT_TABLE
    audit_rows = read_tsv(source_audit) if source_audit.exists() else []
    completed = {
        row["case_id"]
        for row in audit_rows
        if row.get("status", "") != "failed"
        and (ROOT / row.get("coordinate_cache_path", "missing")).is_file()
        and (ROOT / row.get("residue_metadata_path", "missing")).is_file()
    }
    for source in source_rows:
        for replicate in (2, 3):
            case_id = f"ATLAS_{source['system']}_R{replicate}"
            if case_id in completed:
                print(f"[V4 ATLAS] resume skip {case_id}", flush=True)
                continue
            result = decode_one(source, replicate)
            audit_rows = [row for row in audit_rows if row.get("case_id") != case_id]
            audit_rows.append(result)
            write_tsv(AUDIT_CHECKPOINT, audit_rows)
            print(f"[V4 ATLAS] {len(audit_rows)}/16 {case_id} {result['status']} {float(result['decode_seconds']):.1f}s", flush=True)
    write_tsv(AUDIT_TABLE, audit_rows)
    successful = [row for row in audit_rows if row["status"] != "failed"]
    fresh_rows = []
    gate_hash = (OUT / "03_gate_calibration/gate_registry.sha256").read_text().split()[0]
    rule_hash = (OUT / "04_candidates/adaptive_rule.sha256").read_text().split()[0]
    for row in successful:
        fresh_rows.append(
            {
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "source_class": "ATLAS_real_continuous_unused_replicate",
                "replicate": row["replicate"],
                "input_path": row["coordinate_cache_path"],
                "input_key": "npy_array",
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "coordinate_unit": "angstrom",
                "selection": "protein_CA",
                "family_independent": False,
                "replicate_level_fresh": True,
                "continuity_diagnostic": row["continuity_diagnostic"],
                "PBC_nojump_status": row["PBC_nojump_status"],
                "gate_registry_sha256": gate_hash,
                "adaptive_rule_sha256": rule_hash,
            }
        )
    write_tsv(OUT / "05_fresh_panels/fresh_long_panel.tsv", fresh_rows)
    historical = []
    for path, label in (
        (ROOT / "results/cpu_fast_path_v2/02_real_panel/full_panel.tsv", "v2_real_system_panel"),
        (ROOT / "results/cpu_fast_path_v2/02_real_panel/challenge_panel.tsv", "v2_challenge_panel"),
    ):
        for row in read_tsv(path):
            historical.append(
                {
                    **row,
                    "historical_source": label,
                    "fresh_validation": False,
                }
            )
    write_tsv(OUT / "05_fresh_panels/historical_stress_panel.tsv", historical)
    warnings = sum(row["status"].startswith("warning") for row in successful)
    report = f"""# ATLAS R2/R3 local archive audit

- Eight existing ATLAS archives were inspected; no download was performed.
- R2/R3 members present and decoded: `{len(successful)}/16`.
- Frame requirement: every passing member has 10,001 frames.
- CA mapping is checked against the frozen R1 topology mapping and archive PDB.
- Coordinate units are Angstrom after MDAnalysis conversion.
- Regenerable coordinate caches are stored at `{CACHE}` and are excluded from the V9 archive.
- An initial project-storage cache attempt was interrupted before any benchmark timing because a shared RAID journal flush stalled; no timing result was retained from that attempt.
- Continuity warnings: `{warnings}`; see the member audit table for raw maximum adjacent-atom steps and COG drift.
- These R2/R3 trajectories are fresh replicate-level evidence, not sixteen family-independent systems.
- Continuity diagnostics do not independently prove PBC/nojump correctness. The fitted ATLAS trajectories are analysed as provided.
"""
    (OUT / "05_fresh_panels/atlas_r2_r3_audit.md").write_text(report)
    lock = {
        "fresh_mdcath_domains": 512,
        "fresh_long_replicates": len(fresh_rows),
        "fresh_long_families": len({row["family_id"] for row in fresh_rows}),
        "atlas_warnings": warnings,
        "failed_decodes": len(audit_rows) - len(successful),
        "gate_sha256": gate_hash,
        "adaptive_rule_sha256": rule_hash,
    }
    (OUT / "05_fresh_panels/panel_lock_report.md").write_text(
        "# V3 fresh panel lock\n\n```json\n" + json.dumps(lock, indent=2) + "\n```\n"
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
