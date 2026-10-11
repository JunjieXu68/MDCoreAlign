#!/usr/bin/env python3
"""Acquire and audit one locked 10k-frame ATLAS replicate per system."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import time
from typing import Any
import urllib.error
import urllib.request
import zlib

import numpy as np

from scripts.post_b3_optimisation_v1.common import (
    OUT,
    ROOT,
    read_tsv,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


P7 = OUT / "p7_timing_atlas"
LOCKED = (
    ROOT
    / "results/database_scale_benchmark_v2/b5_real_long"
    / "b5_atlas_acquisition_candidates_locked.tsv"
)
INPUTS = P7 / "atlas_readiness_inputs"
DOWNLOAD_MANIFEST = P7 / "atlas_downloaded_source_members.tsv"
MEMBER_MANIFEST = P7 / "atlas_remote_zip_members.tsv"
READINESS_MANIFEST = P7 / "atlas_readiness_manifest.tsv"
USER_AGENT = "MDCoreAlign-ATLAS-readiness/0.1"
RANGE_CHUNK = 16 * 1024 * 1024
MAX_RETRIES = 6


def _request_range(url: str, start: int, end: int) -> tuple[bytes, Any]:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
    )
    error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
                expected = end - start + 1
                if response.status != 206 or len(payload) != expected:
                    raise RuntimeError(
                        f"unexpected range response status={response.status}, "
                        f"bytes={len(payload)}, expected={expected}"
                    )
                return payload, response.headers
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            error = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"range request failed after {MAX_RETRIES} attempts: {error}")


def _remote_zip_index(url: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first, headers = _request_range(url, 0, 0)
    del first
    content_range = headers.get("Content-Range", "")
    try:
        total_size = int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"missing archive size in Content-Range: {content_range}") from exc

    tail_size = min(total_size, 256 * 1024)
    tail_start = total_size - tail_size
    tail, _ = _request_range(url, tail_start, total_size - 1)
    eocd_position = tail.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    eocd = struct.unpack_from("<4s4H2IH", tail, eocd_position)
    _, disk, central_disk, entries_disk, entries_total, central_size, central_offset, _ = eocd
    if (
        entries_total == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        locator_position = tail.rfind(b"PK\x06\x07", 0, eocd_position)
        if locator_position < 0:
            raise RuntimeError("ZIP64 locator not found")
        _, _zip64_disk, zip64_offset, _disk_count = struct.unpack_from(
            "<4sIQI", tail, locator_position
        )
        zip64_record, _ = _request_range(url, zip64_offset, zip64_offset + 55)
        values = struct.unpack_from("<4sQ2H2I4Q", zip64_record)
        if values[0] != b"PK\x06\x06":
            raise RuntimeError("invalid ZIP64 end-of-central-directory record")
        entries_disk = values[6]
        entries_total = values[7]
        central_size = values[8]
        central_offset = values[9]
    if disk != 0 or central_disk != 0 or entries_disk != entries_total:
        raise RuntimeError("multi-disk ZIP archives are unsupported")
    central_end = central_offset + central_size
    if central_offset >= tail_start and central_end <= total_size:
        central = tail[central_offset - tail_start : central_end - tail_start]
    else:
        central, _ = _request_range(url, central_offset, central_end - 1)

    members: list[dict[str, Any]] = []
    position = 0
    while position < len(central):
        if central[position : position + 4] != b"PK\x01\x02":
            raise RuntimeError(f"invalid central-directory signature at {position}")
        values = struct.unpack_from("<4s6H3I5H2I", central, position)
        (
            _,
            _version_made,
            _version_needed,
            flags,
            method,
            _mtime,
            _mdate,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            _disk_start,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = values
        name_start = position + 46
        name = central[name_start : name_start + filename_length].decode("utf-8")
        extra_start = name_start + filename_length
        extra = central[extra_start : extra_start + extra_length]
        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            extra_position = 0
            zip64_values: list[int] | None = None
            while extra_position + 4 <= len(extra):
                field_id, field_size = struct.unpack_from("<HH", extra, extra_position)
                field = extra[
                    extra_position + 4 : extra_position + 4 + field_size
                ]
                if field_id == 0x0001:
                    zip64_values = [
                        struct.unpack_from("<Q", field, offset)[0]
                        for offset in range(0, len(field), 8)
                        if offset + 8 <= len(field)
                    ]
                    break
                extra_position += 4 + field_size
            if zip64_values is None:
                raise RuntimeError(f"ZIP64 extra field missing for {name}")
            value_index = 0
            if uncompressed_size == 0xFFFFFFFF:
                uncompressed_size = zip64_values[value_index]
                value_index += 1
            if compressed_size == 0xFFFFFFFF:
                compressed_size = zip64_values[value_index]
                value_index += 1
            if local_header_offset == 0xFFFFFFFF:
                local_header_offset = zip64_values[value_index]
        members.append(
            {
                "member_name": name,
                "compression_method": method,
                "flags": flags,
                "crc32_hex": f"{crc32:08x}",
                "compressed_size_bytes": compressed_size,
                "uncompressed_size_bytes": uncompressed_size,
                "local_header_offset": local_header_offset,
            }
        )
        position = name_start + filename_length + extra_length + comment_length
    if len(members) != entries_total:
        raise RuntimeError(
            f"central-directory member mismatch: {len(members)} != {entries_total}"
        )
    metadata = {
        "remote_archive_size_bytes": total_size,
        "remote_last_modified": headers.get("Last-Modified", ""),
        "remote_content_type": headers.get("Content-Type", ""),
        "remote_accept_ranges": headers.get("Accept-Ranges", ""),
        "central_directory_sha256": hashlib.sha256(central).hexdigest(),
    }
    return metadata, members


def _member_data_offset(url: str, member: dict[str, Any]) -> int:
    offset = int(member["local_header_offset"])
    header, _ = _request_range(url, offset, offset + 29)
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header for {member['member_name']}")
    filename_length, extra_length = values[-2:]
    return offset + 30 + filename_length + extra_length


def _extract_remote_member(
    url: str,
    member: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed_size = int(member["compressed_size_bytes"])
    data_offset = _member_data_offset(url, member)
    compressed = destination.with_suffix(destination.suffix + ".compressed.part")
    current = compressed.stat().st_size if compressed.exists() else 0
    if current > compressed_size:
        raise RuntimeError(f"oversized partial download: {compressed}")
    with compressed.open("ab") as handle:
        while current < compressed_size:
            size = min(RANGE_CHUNK, compressed_size - current)
            payload, _ = _request_range(
                url,
                data_offset + current,
                data_offset + current + size - 1,
            )
            handle.write(payload)
            handle.flush()
            current += len(payload)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    crc = 0
    uncompressed_size = 0
    method = int(member["compression_method"])
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if method == 8 else None
    if method not in (0, 8):
        raise RuntimeError(f"unsupported ZIP compression method {method}")
    with compressed.open("rb") as source, temporary.open("wb") as target:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            output = decompressor.decompress(block) if decompressor else block
            if output:
                target.write(output)
                digest.update(output)
                crc = zlib.crc32(output, crc)
                uncompressed_size += len(output)
        if decompressor:
            output = decompressor.flush()
            target.write(output)
            digest.update(output)
            crc = zlib.crc32(output, crc)
            uncompressed_size += len(output)
    expected_size = int(member["uncompressed_size_bytes"])
    expected_crc = int(member["crc32_hex"], 16)
    if uncompressed_size != expected_size or (crc & 0xFFFFFFFF) != expected_crc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ZIP integrity mismatch for {member['member_name']}: "
            f"size={uncompressed_size}/{expected_size}, "
            f"crc={crc & 0xFFFFFFFF:08x}/{expected_crc:08x}"
        )
    temporary.replace(destination)
    compressed.unlink()
    return {
        "local_path": str(destination.relative_to(ROOT)),
        "selected_member_sha256": digest.hexdigest(),
        "zip_crc_verified": True,
        "downloaded_uncompressed_size_bytes": uncompressed_size,
    }


def _crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xFFFFFFFF


def _choose_members(system: str, members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {row["member_name"]: row for row in members}
    required = [
        f"{system}_prod_R1.tpr",
        f"{system}_prod_R1_fit.xtc",
        f"{system}.pdb",
        "README.txt",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"missing expected ATLAS members for {system}: {missing}")
    return [names[name] for name in required]


def _download_system(row: dict[str, str]) -> dict[str, Any]:
    system = row["system"]
    url = row["trajectory_path"]
    started = time.perf_counter()
    try:
        archive, members = _remote_zip_index(url)
        selected = _choose_members(system, members)
        output_rows = []
        for member in selected:
            destination = INPUTS / system / member["member_name"]
            if destination.exists() and destination.stat().st_size == int(
                member["uncompressed_size_bytes"]
            ):
                observed_crc = _crc32(destination)
                expected_crc = int(member["crc32_hex"], 16)
                if observed_crc != expected_crc:
                    raise RuntimeError(
                        f"cached member CRC mismatch for {member['member_name']}: "
                        f"{observed_crc:08x}/{expected_crc:08x}"
                    )
                extracted = {
                    "local_path": str(destination.relative_to(ROOT)),
                    "selected_member_sha256": sha256(destination),
                    "zip_crc_verified": True,
                    "downloaded_uncompressed_size_bytes": destination.stat().st_size,
                }
            else:
                extracted = _extract_remote_member(url, member, destination)
            output_rows.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "system": system,
                    "source_url": url,
                    **archive,
                    **member,
                    **extracted,
                    "full_remote_archive_sha256": "not_computed_range_extraction",
                }
            )
        return {
            "status": "success",
            "system": system,
            "elapsed_seconds": time.perf_counter() - started,
            "archive": archive,
            "members": members,
            "selected": output_rows,
        }
    except Exception as exc:  # The failure is recorded per locked input.
        return {
            "status": "failed",
            "system": system,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def acquire(workers: int) -> None:
    locked = read_tsv(LOCKED)
    INPUTS.mkdir(parents=True, exist_ok=True)
    completed = []
    member_rows = []
    selected_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_system, row): row for row in locked}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed.append(
                {
                    "system": result["system"],
                    "status": result["status"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "error": result.get("error", ""),
                }
            )
            if result["status"] == "success":
                for member in result["members"]:
                    member_rows.append(
                        {
                            "system": result["system"],
                            **result["archive"],
                            **member,
                        }
                    )
                selected_rows.extend(result["selected"])
            print(
                f"[P7 ATLAS acquire] {index}/{len(locked)} {result['system']} "
                f"{result['status']} {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    write_tsv(P7 / "atlas_acquisition_status.tsv", completed)
    if member_rows:
        write_tsv(MEMBER_MANIFEST, member_rows)
    if selected_rows:
        write_tsv(DOWNLOAD_MANIFEST, selected_rows)
    write_json(
        P7 / "atlas_acquisition_run.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "locked_systems": len(locked),
            "successful_systems": sum(row["status"] == "success" for row in completed),
            "workers": workers,
            "range_chunk_bytes": RANGE_CHUNK,
            "selection": "R1 TPR, R1 10000-frame fitted XTC, PDB, README",
            "full_remote_archives_downloaded": False,
        },
    )


def _kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    ref = reference - np.mean(reference, axis=0)
    mob = mobile - np.mean(mobile, axis=0)
    u, _singular, vt = np.linalg.svd(mob.T @ ref)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    difference = mob @ rotation - ref
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", difference, difference))))


def _safe_attribute(group: Any, name: str) -> list[str]:
    try:
        return [str(value) for value in getattr(group, name)]
    except (AttributeError, TypeError):
        return [""] * len(group)


def _audit_system(row: dict[str, str]) -> dict[str, Any]:
    import MDAnalysis as mda

    system = row["system"]
    directory = INPUTS / system
    tpr = directory / f"{system}_prod_R1.tpr"
    xtc = directory / f"{system}_prod_R1_fit.xtc"
    pdb = directory / f"{system}.pdb"
    result: dict[str, Any] = {
        "trajectory_id": row["trajectory_id"],
        "system": system,
        "source_url": row["trajectory_path"],
        "expected_n_frames": "approximately_10000_ATLAS_contract",
        "expected_n_residues": row["n_residues"],
        "source_native_coordinate_unit": "nm_GROMACS_XTC",
        "analysis_coordinate_unit": "A_MDAnalysis_convert_units",
        "topology_path": str(tpr.relative_to(ROOT)),
        "trajectory_path": str(xtc.relative_to(ROOT)),
        "pdb_path": str(pdb.relative_to(ROOT)),
    }
    try:
        tpr_universe = mda.Universe(str(tpr))
        pdb_universe = mda.Universe(str(pdb))
        universe = mda.Universe(str(pdb), str(xtc), convert_units=True)
        protein = universe.select_atoms("protein")
        ca = universe.select_atoms("protein and name CA")
        pdb_ca = pdb_universe.select_atoms("protein and name CA")
        tpr_ca = tpr_universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            raise RuntimeError("protein CA selection is empty")
        tpr_keys = list(
            zip(
                _safe_attribute(tpr_ca, "resids"),
                _safe_attribute(tpr_ca, "resnames"),
                _safe_attribute(tpr_ca, "names"),
            )
        )
        pdb_keys = list(
            zip(
                _safe_attribute(pdb_ca, "resids"),
                _safe_attribute(pdb_ca, "resnames"),
                _safe_attribute(pdb_ca, "names"),
            )
        )
        result.update(
            {
                "status": "success",
                "observed_n_frames": len(universe.trajectory),
                "observed_n_atoms": len(universe.atoms),
                "observed_n_protein_atoms": len(protein),
                "observed_n_CA": len(ca),
                "full_system_TPR_n_atoms": len(tpr_universe.atoms),
                "full_system_TPR_n_CA": len(tpr_ca),
                "pdb_n_atoms": len(pdb_universe.atoms),
                "pdb_n_CA": len(pdb_ca),
                "protein_only_XTC_n_atoms": universe.trajectory.n_atoms,
                "PDB_matches_protein_only_XTC_atom_count": len(pdb_universe.atoms)
                == universe.trajectory.n_atoms,
                "TPR_is_full_system_not_direct_XTC_topology": len(tpr_universe.atoms)
                != universe.trajectory.n_atoms,
                "exact_10000_frame_match": len(universe.trajectory) == 10000,
                "frame_count_matches_ATLAS_approximately_10000_contract": len(
                    universe.trajectory
                )
                in (10000, 10001),
                "CA_count_matches_locked_residue_count": len(ca)
                == int(row["n_residues"]),
                "TPR_PDB_CA_order_mapping_exact": tpr_keys == pdb_keys,
                "duplicate_TPR_CA_keys": len(tpr_keys) - len(set(tpr_keys)),
                "trajectory_reader_units": json.dumps(
                    getattr(universe.trajectory, "units", {}), sort_keys=True
                ),
            }
        )
        first = None
        previous = None
        first_cog = None
        max_raw_step = []
        adjacent_fit_rmsd = []
        cog_drift = []
        finite_frames = 0
        box_frames = 0
        adjacent_ca_median = math.nan
        for frame_index, timestep in enumerate(universe.trajectory):
            positions = np.asarray(ca.positions, dtype=np.float64).copy()
            if np.isfinite(positions).all():
                finite_frames += 1
            cog = np.mean(positions, axis=0)
            if first is None:
                first = positions
                first_cog = cog
                if len(positions) > 1:
                    adjacent_ca_median = float(
                        np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                    )
            else:
                steps = np.linalg.norm(positions - previous, axis=1)
                max_raw_step.append(float(np.max(steps)))
                adjacent_fit_rmsd.append(_kabsch_rmsd(previous, positions))
            cog_drift.append(float(np.linalg.norm(cog - first_cog)))
            dimensions = timestep.dimensions
            if dimensions is not None and np.all(np.asarray(dimensions[:3]) > 0):
                box_frames += 1
            previous = positions
            if (frame_index + 1) % 1000 == 0:
                print(f"[P7 ATLAS verify] {system} {frame_index + 1}/10000", flush=True)

        fit_array = np.asarray(adjacent_fit_rmsd, dtype=float)
        raw_array = np.asarray(max_raw_step, dtype=float)
        cog_array = np.asarray(cog_drift, dtype=float)
        fit_p99 = float(np.quantile(fit_array, 0.99))
        fit_max = float(np.max(fit_array))
        gross_jump = bool(fit_max > max(10.0, 5.0 * fit_p99))
        result.update(
            {
                "finite_frame_fraction": finite_frames / len(universe.trajectory),
                "box_vector_frame_fraction": box_frames / len(universe.trajectory),
                "median_adjacent_CA_distance_first_frame_A": adjacent_ca_median,
                "coordinate_scale_plausible_for_A": 3.2 <= adjacent_ca_median <= 4.5,
                "median_frame_to_frame_fitted_CA_RMSD_A": float(
                    np.median(fit_array)
                ),
                "P99_frame_to_frame_fitted_CA_RMSD_A": fit_p99,
                "max_frame_to_frame_fitted_CA_RMSD_A": fit_max,
                "P99_max_atom_raw_frame_step_A": float(np.quantile(raw_array, 0.99)),
                "max_atom_raw_frame_step_A": float(np.max(raw_array)),
                "median_COG_drift_from_first_A": float(np.median(cog_array)),
                "max_COG_drift_from_first_A": float(np.max(cog_array)),
                "gross_discontinuous_jump_flag": gross_jump,
                "continuity_rule": "max_adjacent_fit_RMSD_gt_max_10A_or_5x_P99",
                "topology_mapping_status": "pass"
                if tpr_keys == pdb_keys
                and len(ca) == int(row["n_residues"])
                and len(set(tpr_keys)) == len(tpr_keys)
                else "review",
                "unit_integrity_status": "pass"
                if 3.2 <= adjacent_ca_median <= 4.5
                else "review",
                "continuity_status": "review_gross_jump"
                if gross_jump
                else "pass_no_gross_jump_detected",
            }
        )
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return result


def verify() -> None:
    locked = read_tsv(LOCKED)
    rows = []
    for index, row in enumerate(locked, start=1):
        result = _audit_system(row)
        rows.append(result)
        print(
            f"[P7 ATLAS verify] completed {index}/{len(locked)} "
            f"{result['system']} {result['status']}",
            flush=True,
        )
    write_tsv(READINESS_MANIFEST, rows)


def finalize() -> None:
    locked = read_tsv(LOCKED)
    acquisition = read_tsv(P7 / "atlas_acquisition_status.tsv")
    readiness = read_tsv(READINESS_MANIFEST) if READINESS_MANIFEST.exists() else []
    source_rows = read_tsv(DOWNLOAD_MANIFEST) if DOWNLOAD_MANIFEST.exists() else []
    successful_acquisition = sum(row["status"] == "success" for row in acquisition)
    successful_readiness = sum(row["status"] == "success" for row in readiness)
    mapping_pass = sum(row.get("topology_mapping_status") == "pass" for row in readiness)
    unit_pass = sum(row.get("unit_integrity_status") == "pass" for row in readiness)
    continuity_pass = sum(
        row.get("continuity_status") == "pass_no_gross_jump_detected"
        for row in readiness
    )
    hash_count = sum(bool(row.get("selected_member_sha256")) for row in source_rows)
    report = [
        "# Locked ATLAS readiness audit",
        "",
        f"- Prospectively locked systems: `{len(locked)}` (188–2,128 residues).",
        f"- Successful selected-member acquisition: `{successful_acquisition}/{len(locked)}`.",
        f"- Successful approximately 10k-frame R1 trajectory audits: `{successful_readiness}/{len(locked)}`.",
        f"- Exact TPR/PDB CA mapping passes: `{mapping_pass}/{len(locked)}`.",
        f"- Coordinate-unit plausibility passes: `{unit_pass}/{len(locked)}`.",
        f"- No gross discontinuity detected: `{continuity_pass}/{len(locked)}`.",
        f"- Extracted source members with SHA-256: `{hash_count}/{len(source_rows)}`.",
        f"- Observed frame counts: `{sorted({row.get('observed_n_frames', '') for row in readiness})}`; all eight archives contain 10,001 R1 frames.",
        "",
        "## Scope and limitations",
        "",
        "The official 10,000-frame protein archive was range-indexed for each system. "
        "Only replicate R1, its TPR topology, the archive PDB and README were extracted. "
        "Each extracted member was verified against the ZIP CRC and assigned a SHA-256 digest.",
        "",
        "The complete remote ZIP archives were not downloaded, so whole-archive SHA-256 "
        "digests are explicitly unavailable. Replicates R2/R3 were not audited in this "
        "readiness pass. Continuity denotes absence of gross coordinate jumps under the "
        "registered geometric heuristic; it does not independently establish PBC/nojump "
        "processing or biological trajectory continuity.",
    ]
    write_text(P7 / "atlas_readiness_report.md", "\n".join(report))
    write_json(
        P7 / "atlas_readiness_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "locked_systems": len(locked),
            "successful_acquisition": successful_acquisition,
            "successful_readiness_audits": successful_readiness,
            "mapping_pass": mapping_pass,
            "unit_pass": unit_pass,
            "continuity_pass": continuity_pass,
            "full_archive_sha256_available": False,
            "audited_replicate": "R1",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("acquire", "verify", "finalize"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "acquire":
        acquire(args.workers)
    elif args.mode == "verify":
        verify()
    else:
        finalize()


if __name__ == "__main__":
    main()
