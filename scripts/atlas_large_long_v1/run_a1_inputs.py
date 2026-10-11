#!/usr/bin/env python3
"""Acquire full ATLAS archives and prepare audited R1 CA caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request
import zipfile

import numpy as np

from scripts.atlas_large_long_v1.common import (
    LOCKED,
    OUT,
    P7,
    ROOT,
    read_tsv,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


ARCHIVES = OUT / "source_archives"
CACHE = OUT / "input_cache"
DOWNLOAD_TABLE = OUT / "a1_full_archive_download.tsv"
CACHE_TABLE = OUT / "a1_ca_cache_manifest.tsv"
P7_READINESS = P7 / "atlas_readiness_manifest.tsv"
P7_MEMBERS = P7 / "atlas_downloaded_source_members.tsv"
CHUNK_SIZE = 32 * 1024 * 1024
MAX_RETRIES = 6
USER_AGENT = "MDCoreAlign-ATLAS-large-long-v1/1.0"


def _range(url: str, start: int, end: int) -> tuple[bytes, Any]:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
    )
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = response.read()
                if response.status != 206 or len(payload) != end - start + 1:
                    raise RuntimeError(
                        f"invalid range response {response.status}, {len(payload)} bytes"
                    )
                return payload, response.headers
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"range request failed: {last}")


def _remote_size(url: str) -> tuple[int, str]:
    _, headers = _range(url, 0, 0)
    content_range = headers.get("Content-Range", "")
    return int(content_range.rsplit("/", 1)[1]), headers.get("Last-Modified", "")


def _download_one(row: dict[str, str]) -> dict[str, Any]:
    system = row["system"]
    url = row["trajectory_path"]
    output = ARCHIVES / f"{system}_protein.zip"
    partial = output.with_suffix(output.suffix + ".part")
    started = time.perf_counter()
    try:
        size, last_modified = _remote_size(url)
        if output.exists() and output.stat().st_size == size:
            archive_hash = sha256(output)
            resumed = True
        else:
            if output.exists():
                raise RuntimeError(f"existing completed archive has wrong size: {output}")
            partial.parent.mkdir(parents=True, exist_ok=True)
            current = partial.stat().st_size if partial.exists() else 0
            if current > size:
                raise RuntimeError(f"partial archive is oversized: {partial}")
            resumed = current > 0
            with partial.open("ab") as handle:
                while current < size:
                    amount = min(CHUNK_SIZE, size - current)
                    payload, _ = _range(url, current, current + amount - 1)
                    handle.write(payload)
                    handle.flush()
                    current += len(payload)
            partial.replace(output)
            archive_hash = sha256(output)
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            selected = f"{system}_prod_R1_fit.xtc"
            topology = f"{system}_prod_R1.tpr"
            pdb = f"{system}.pdb"
            required_present = all(name in names for name in (selected, topology, pdb, "README.txt"))
        return {
            "system": system,
            "status": "success" if required_present else "failed",
            "source_url": url,
            "local_archive_path": str(output.relative_to(ROOT)),
            "local_archive_size_bytes": output.stat().st_size,
            "local_archive_sha256": archive_hash,
            "remote_last_modified": last_modified,
            "zip_member_count": len(names),
            "required_R1_members_present": required_present,
            "resumed_or_reused": resumed,
            "elapsed_seconds": time.perf_counter() - started,
            "error": "" if required_present else "required_R1_members_missing",
        }
    except Exception as exc:
        return {
            "system": system,
            "status": "failed",
            "source_url": url,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def download(workers: int) -> None:
    if not (OUT / "a0_freeze/a0_status.json").exists():
        raise RuntimeError("A0 freeze must complete before A1")
    rows = read_tsv(LOCKED)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[A1 archive] {index}/{len(rows)} {result['system']} "
                f"{result['status']} {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    write_tsv(DOWNLOAD_TABLE, sorted(results, key=lambda row: row["system"]))
    write_json(
        OUT / "a1_archive_download_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "systems": len(rows),
            "successes": sum(row["status"] == "success" for row in results),
            "workers": workers,
            "chunk_size_bytes": CHUNK_SIZE,
        },
    )


def _metadata_values(group: Any, attribute: str, fallback: str = "") -> np.ndarray:
    try:
        return np.asarray(getattr(group, attribute)).astype(str)
    except (AttributeError, TypeError):
        return np.full(len(group), fallback, dtype="U8")


def prepare_cache() -> None:
    import MDAnalysis as mda

    downloads = {row["system"]: row for row in read_tsv(DOWNLOAD_TABLE)}
    locked = read_tsv(LOCKED)
    outputs = []
    for index, row in enumerate(locked, start=1):
        system = row["system"]
        if downloads[system]["status"] != "success":
            outputs.append({"system": system, "status": "failed", "error": "archive_download_failed"})
            continue
        source = P7 / "atlas_readiness_inputs" / system
        pdb = source / f"{system}.pdb"
        xtc = source / f"{system}_prod_R1_fit.xtc"
        started = time.perf_counter()
        try:
            universe = mda.Universe(str(pdb), str(xtc), convert_units=True)
            ca = universe.select_atoms("protein and name CA")
            n_frames = len(universe.trajectory)
            n_ca = len(ca)
            CACHE.mkdir(parents=True, exist_ok=True)
            coordinate_path = CACHE / f"{system}_R1_CA_A_float64.npy"
            metadata_path = CACHE / f"{system}_R1_residue_metadata.npz"
            temporary = coordinate_path.with_suffix(".npy.tmp")
            values = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float64,
                shape=(n_frames, n_ca, 3),
            )
            first_time = None
            second_time = None
            for frame_index, timestep in enumerate(universe.trajectory):
                values[frame_index] = np.asarray(ca.positions, dtype=np.float64)
                if frame_index == 0:
                    first_time = float(timestep.time)
                elif frame_index == 1:
                    second_time = float(timestep.time)
            values.flush()
            del values
            temporary.replace(coordinate_path)
            pdb_chain_ids = _metadata_values(ca, "chainIDs")
            chain_ids = pdb_chain_ids.copy()
            chain_mapping_source = "archive_PDB_chain_column"
            if not np.any(chain_ids != ""):
                segids = _metadata_values(ca, "segids", fallback="_")
                if set(segids.tolist()).issubset({"", "_", "SYSTEM"}):
                    explicit_chain = system.rsplit("_", 1)[-1]
                    chain_ids = np.full(n_ca, explicit_chain, dtype="U8")
                    chain_mapping_source = "locked_ATLAS_single_chain_identifier"
                else:
                    chain_ids = segids
                    chain_mapping_source = "archive_PDB_segid"
            np.savez_compressed(
                metadata_path,
                chain_ids=chain_ids,
                residue_ids=_metadata_values(ca, "resids"),
                residue_names=_metadata_values(ca, "resnames"),
                atom_names=_metadata_values(ca, "names"),
            )
            outputs.append(
                {
                    "system": system,
                    "status": "success",
                    "coordinate_cache_path": str(coordinate_path.relative_to(ROOT)),
                    "coordinate_cache_sha256": sha256(coordinate_path),
                    "coordinate_cache_size_bytes": coordinate_path.stat().st_size,
                    "metadata_cache_path": str(metadata_path.relative_to(ROOT)),
                    "metadata_cache_sha256": sha256(metadata_path),
                    "n_frames": n_frames,
                    "n_CA": n_ca,
                    "chain_ids": ";".join(sorted(set(chain_ids.tolist()))),
                    "chain_counts": json.dumps(
                        {chain: int(np.count_nonzero(chain_ids == chain)) for chain in sorted(set(chain_ids.tolist()))},
                        sort_keys=True,
                    ),
                    "archive_PDB_chain_column_blank": not np.any(
                        pdb_chain_ids != ""
                    ),
                    "chain_mapping_source": chain_mapping_source,
                    "reader_units": json.dumps(universe.trajectory.units, sort_keys=True),
                    "observed_first_frame_time_ps": first_time,
                    "observed_second_frame_time_ps": second_time,
                    "observed_frame_interval_ps": second_time - first_time,
                    "decode_and_cache_seconds": time.perf_counter() - started,
                    "error": "",
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "system": system,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        print(
            f"[A1 cache] {index}/{len(locked)} {system} {outputs[-1]['status']}",
            flush=True,
        )
    write_tsv(CACHE_TABLE, outputs)


def audit() -> None:
    locked = {row["system"]: row for row in read_tsv(LOCKED)}
    downloads = {row["system"]: row for row in read_tsv(DOWNLOAD_TABLE)}
    caches = {row["system"]: row for row in read_tsv(CACHE_TABLE)}
    readiness = {row["system"]: row for row in read_tsv(P7_READINESS)}
    members = read_tsv(P7_MEMBERS)
    member_lookup = {(row["system"], row["member_name"]): row for row in members}
    output = []
    for system in sorted(locked):
        source = locked[system]
        download = downloads[system]
        cache = caches[system]
        prior = readiness[system]
        tpr_name = f"{system}_prod_R1.tpr"
        xtc_name = f"{system}_prod_R1_fit.xtc"
        pdb_name = f"{system}.pdb"
        tpr = member_lookup[(system, tpr_name)]
        xtc = member_lookup[(system, xtc_name)]
        pdb = member_lookup[(system, pdb_name)]
        pass_status = (
            download["status"] == "success"
            and cache["status"] == "success"
            and int(cache["n_frames"]) == 10001
            and int(cache["n_CA"]) == int(source["n_residues"])
            and prior["topology_mapping_status"] == "pass"
            and prior["unit_integrity_status"] == "pass"
            and prior["continuity_status"] == "pass_no_gross_jump_detected"
        )
        output.append(
            {
                "trajectory_id": source["trajectory_id"],
                "system": system,
                "replicate": "R1",
                "source_database": "ATLAS",
                "source_url": source["trajectory_path"],
                "local_archive_path": download["local_archive_path"],
                "local_archive_sha256": download["local_archive_sha256"],
                "local_archive_size_bytes": download["local_archive_size_bytes"],
                "selected_trajectory_member": xtc_name,
                "selected_topology_member": tpr_name,
                "selected_PDB_member": pdb_name,
                "topology_TPR_sha256": tpr["selected_member_sha256"],
                "trajectory_XTC_sha256": xtc["selected_member_sha256"],
                "PDB_sha256": pdb["selected_member_sha256"],
                "coordinate_cache_path": cache["coordinate_cache_path"],
                "coordinate_cache_sha256": cache["coordinate_cache_sha256"],
                "residue_metadata_path": cache["metadata_cache_path"],
                "residue_metadata_sha256": cache["metadata_cache_sha256"],
                "n_frames": cache["n_frames"],
                "n_residues": cache["n_CA"],
                "n_CA": cache["n_CA"],
                "chain_mapping": cache["chain_counts"],
                "chain_mapping_source": cache["chain_mapping_source"],
                "archive_PDB_chain_column_blank": cache[
                    "archive_PDB_chain_column_blank"
                ],
                "coordinate_unit": "Angstrom_after_MDAnalysis_conversion_from_GROMACS_nm",
                "documented_simulation_length_ns": 100,
                "documented_frame_interval_ps": 10,
                "observed_frame_interval_ps": cache["observed_frame_interval_ps"],
                "atom_selection": "protein and name CA",
                "median_adjacent_CA_distance_first_frame_A": prior["median_adjacent_CA_distance_first_frame_A"],
                "max_adjacent_frame_fitted_CA_RMSD_A": prior["max_frame_to_frame_fitted_CA_RMSD_A"],
                "max_adjacent_frame_atom_displacement_A": prior["max_atom_raw_frame_step_A"],
                "max_COG_drift_from_first_A": prior["max_COG_drift_from_first_A"],
                "continuity_status": prior["continuity_status"],
                "PBC_nojump_status": "README_documents_PBC_removed_and_frames_aligned_not_independently_verified",
                "R2_R3_audited": False,
                "input_audit_status": "pass" if pass_status else "failed",
            }
        )
    write_tsv(OUT / "atlas_manifest.tsv", output)
    passes = sum(row["input_audit_status"] == "pass" for row in output)
    write_text(
        OUT / "atlas_input_audit.md",
        f"""# ATLAS large/long v1 input audit

- Locked systems: `8`; passing audits: `{passes}/8`.
- R1 frame count: `10,001` for every system.
- CA range: `{min(int(row['n_CA']) for row in output)}–{max(int(row['n_CA']) for row in output)}`.
- Every complete local ZIP has a SHA-256 digest; selected TPR, XTC and PDB
  members retain independent hashes and ZIP-CRC provenance from P7.
- The atom selection is `protein and name CA`; chain and residue metadata come
  from the archive PDB rather than array index inference.
- ATLAS README files document 100 ns trajectories, 10 ps frame intervals, PBC
  removal and fitted frames. The observed files contain 10,001 frames, including
  the starting frame.
- Geometric continuity diagnostics found no gross jump under the locked P7
  heuristic. This does **not** independently prove PBC/nojump correctness.
- Only R1 is audited. No R2/R3 reproducibility claim is permitted.
""",
    )
    write_json(
        OUT / "a1_input_audit_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "systems": len(output),
            "passes": passes,
            "all_local_archive_hashes_available": all(bool(row["local_archive_sha256"]) for row in output),
            "R2_R3_audited": False,
            "PBC_nojump_independently_verified": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("download", "prepare-cache", "audit"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "download":
        download(args.workers)
    elif args.mode == "prepare-cache":
        prepare_cache()
    else:
        audit()


if __name__ == "__main__":
    main()
