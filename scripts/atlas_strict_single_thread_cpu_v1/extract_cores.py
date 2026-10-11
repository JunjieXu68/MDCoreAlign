#!/usr/bin/env python3
"""Untimed stable-core extraction and method-agreement analysis."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mdcorealign.benchmark_minimal import solve_minimal_from_pdb
from scripts.atlas_strict_single_thread_cpu_v1.common import (
    FRACTION,
    OUT,
    ROOT,
    STOCK_MDLOVOFIT,
    THREAD_ENV_HASH,
    absolute_input,
    read_tsv,
    setting_id,
    sha256,
    thread_environment,
    write_json,
    write_text,
    write_tsv,
)
from scripts.atlas_strict_single_thread_cpu_v1.prepare_study import LOGICAL_CPU


def residue_metadata(path: Path, expected: int) -> list[dict[str, Any]]:
    """Read residue identity from the first PDB model only."""

    rows = []
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"ENDMDL"):
                break
            if not line.startswith(b"ATOM"):
                continue
            rows.append(
                {
                    "atom_index": len(rows),
                    "chain": line[21:22].decode("ascii").strip() or "_",
                    "residue_id": line[22:26].decode("ascii").strip()
                    + line[26:27].decode("ascii").strip(),
                    "resname": line[17:20].decode("ascii").strip(),
                    "atom_name": line[12:16].decode("ascii").strip(),
                }
            )
    if len(rows) != expected or any(row["atom_name"] != "CA" for row in rows):
        raise ValueError(f"invalid first-model CA mapping for {path}: {len(rows)} != {expected}")
    return rows


def parse_mdlovofit_mask(
    path: Path,
    expected_atoms: int,
) -> tuple[np.ndarray, int, list[tuple[str, str, str]]]:
    """Validate a CA core across stock END-delimited or standard ENDMDL models.

    Stock MDLovoFit 20.1.1 writes ``REMARK FRAME:`` / ATOM / ``END``
    blocks. Its END records terminate individual frames, not the trajectory.
    """

    masks: list[np.ndarray] = []
    current: list[bool] = []
    current_labels: list[tuple[str, str, str]] = []
    first_labels: list[tuple[str, str, str]] | None = None
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        for line in handle:
            if line.startswith(b"ATOM"):
                occupancy = float(line[54:60])
                if not (abs(occupancy) <= 1e-6 or abs(occupancy - 1.0) <= 1e-6):
                    raise ValueError("MDLovoFit core occupancy is not binary")
                if line[12:16].strip() != b"CA":
                    raise ValueError("MDLovoFit core output contains a non-CA atom")
                current.append(abs(occupancy - 1.0) <= 1e-6)
                current_labels.append(
                    (
                        line[21:22].decode("ascii").strip() or "_",
                        line[22:26].decode("ascii").strip() + line[26:27].decode("ascii").strip(),
                        line[17:20].decode("ascii").strip(),
                    )
                )
            elif line[:6].strip() in (b"ENDMDL", b"END"):
                if not current and line[:6].strip() == b"END":
                    # Standard PDB may append a terminal END after ENDMDL.
                    continue
                if len(current) != expected_atoms:
                    raise ValueError(
                        f"MDLovoFit model has {len(current)} atoms; expected {expected_atoms}"
                    )
                masks.append(np.asarray(current, dtype=bool))
                if first_labels is None:
                    first_labels = current_labels
                elif current_labels != first_labels:
                    raise ValueError("MDLovoFit residue mapping differs across output models")
                current = []
                current_labels = []
    if current:
        if len(current) != expected_atoms:
            raise ValueError("truncated final MDLovoFit output model")
        masks.append(np.asarray(current, dtype=bool))
        if first_labels is None:
            first_labels = current_labels
        elif current_labels != first_labels:
            raise ValueError("MDLovoFit residue mapping differs across output models")
    if not masks:
        raise ValueError("MDLovoFit output contains no atom models")
    first = masks[0]
    if any(not np.array_equal(first, mask) for mask in masks[1:]):
        raise ValueError("MDLovoFit occupancy core differs across output models")
    assert first_labels is not None
    return first, len(masks), first_labels


def run_mdlovofit_export(
    row: dict[str, str],
    extraction_id: str,
    *,
    logical_cpu: int = LOGICAL_CPU,
) -> dict[str, Any]:
    """Run or reuse one validated, untimed export without discarding failed evidence."""

    sid = setting_id(row)
    if not extraction_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in extraction_id
    ):
        raise ValueError(f"invalid extraction identifier: {extraction_id!r}")
    input_path = absolute_input(row)
    panel_path = OUT / "01_panel/atlas_48_setting_manifest.tsv"
    panel_hash = sha256(panel_path)
    frozen_panel_hash = (panel_path.with_suffix(".sha256")).read_text().split()[0]
    if panel_hash != frozen_panel_hash:
        raise ValueError("AT6 panel differs from its frozen SHA-256")
    registered = [entry for entry in read_tsv(panel_path) if setting_id(entry) == sid]
    if len(registered) != 1 or registered[0] != row:
        raise ValueError(f"AT6 export row differs from the frozen panel: {sid}")
    if input_path.stat().st_size != int(row["input_size_bytes"]):
        raise ValueError(f"AT6 input size differs from the frozen panel: {sid}")
    input_hash = sha256(input_path)
    if input_hash != row["input_sha256"]:
        raise ValueError(f"AT6 input SHA-256 differs from the frozen panel: {sid}")
    executable_hash = sha256(STOCK_MDLOVOFIT)
    executable_rows = read_tsv(OUT / "00_freeze/executable_hashes.tsv")
    frozen_executable = [
        entry for entry in executable_rows if entry["path"] == str(STOCK_MDLOVOFIT)
    ]
    if len(frozen_executable) != 1 or frozen_executable[0]["sha256"] != executable_hash:
        raise ValueError("AT6 stock MDLovoFit differs from the frozen executable")
    if FRACTION != 0.60:
        raise ValueError("AT6 requires the registered fraction 0.60")
    expected_atoms = int(row["n_ca"])
    expected_models = int(row["n_frames"])
    expected_labels = [
        (atom["chain"], atom["residue_id"], atom["resname"])
        for atom in residue_metadata(input_path, expected_atoms)
    ]
    identity = {
        "schema_version": 1,
        "setting_id": sid,
        "extraction_id": extraction_id,
        "timing_scope": "untimed_core_extraction_not_AT5",
        "panel_manifest_sha256": panel_hash,
        "input_path": str(input_path),
        "input_sha256": input_hash,
        "input_size_bytes": int(row["input_size_bytes"]),
        "executable_path": str(STOCK_MDLOVOFIT),
        "executable_sha256": executable_hash,
        "fraction": FRACTION,
        "n_ca": expected_atoms,
        "thread_environment_sha256": THREAD_ENV_HASH,
    }
    records = OUT / "06_core_extraction/export_records"
    checkpoint = records / f"{sid}__{extraction_id}.json"

    def validated(record: dict[str, Any]) -> dict[str, Any]:
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError(f"AT6 checkpoint identity mismatch: {checkpoint}")
        raw_mask = record["mask"]
        if len(raw_mask) != expected_atoms or any(type(value) is not bool for value in raw_mask):
            raise ValueError(f"invalid AT6 checkpoint mask: {checkpoint}")
        mask = np.asarray(raw_mask, dtype=bool)
        labels = [tuple(label) for label in record["residue_labels"]]
        if (
            record["status"] != "validated"
            or record["returncode"] != 0
            or record["model_count"] != expected_models
            or labels != expected_labels
            or record["core_size"] != int(mask.sum())
            or not 0 < int(mask.sum()) < expected_atoms
            or record["mask_sha256"] != hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        ):
            raise ValueError(f"AT6 checkpoint mapping/mask mismatch: {checkpoint}")
        for stream in ("stdout", "stderr"):
            log_path = ROOT / record[f"{stream}_path"]
            if not log_path.is_file() or sha256(log_path) != record[f"{stream}_sha256"]:
                raise ValueError(f"AT6 checkpoint log hash mismatch: {log_path}")
        return {**record, "mask": mask, "residue_labels": labels}

    if checkpoint.exists():
        return {**validated(json.loads(checkpoint.read_text())), "checkpoint_reused": True}
    if logical_cpu not in os.sched_getaffinity(0):
        raise ValueError(f"AT6 logical CPU {logical_cpu} is outside the available affinity")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = OUT / "06_core_extraction/export_attempts" / f"{sid}__{extraction_id}__{stamp}"
    attempt.mkdir(parents=True)
    stdout_path = attempt / "stdout.txt"
    stderr_path = attempt / "stderr.txt"
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="mdca_at6_") as temporary:
        output = Path(temporary) / "aligned_core.pdb"
        command = [
            "taskset", "-c", str(logical_cpu), str(STOCK_MDLOVOFIT),
            "-f", f"{FRACTION:.6f}", "-t", str(output), str(input_path),
        ]
        attempt_identity = {
            **identity,
            "command": command,
            "logical_cpu": logical_cpu,
            "started_utc": started,
            "stdout_path": str(stdout_path.relative_to(ROOT)),
            "stderr_path": str(stderr_path.relative_to(ROOT)),
        }
        write_json(attempt / "command.json", attempt_identity)
        try:
            with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=thread_environment(),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=7200,
                    check=False,
                )
            if result.returncode != 0 or not output.is_file():
                raise RuntimeError(f"MDLovoFit core export failed for {sid}: exit={result.returncode}")
            mask, model_count, labels = parse_mdlovofit_mask(output, expected_atoms)
            record = {
                **attempt_identity,
                "status": "validated",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": result.returncode,
                "mask": mask.tolist(),
                "core_size": int(mask.sum()),
                "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                "model_count": model_count,
                "residue_labels": labels,
                "stdout_sha256": sha256(stdout_path),
                "stderr_sha256": sha256(stderr_path),
                "export_size_bytes": output.stat().st_size,
                "export_sha256": sha256(output),
            }
            answer = validated(record)
            records.mkdir(parents=True, exist_ok=True)
            validated_path = attempt / "validated_record.json"
            write_json(validated_path, record)
            os.link(validated_path, checkpoint)
            return {**answer, "checkpoint_reused": False}
        except BaseException as error:
            preserved = attempt / "failed_aligned_core.pdb"
            if output.is_file():
                shutil.move(str(output), preserved)
            write_json(
                attempt / "failure.json",
                {
                    **attempt_identity,
                    "status": "failed",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "preserved_export_path": (
                        str(preserved.relative_to(ROOT)) if preserved.is_file() else None
                    ),
                    "stdout_sha256": sha256(stdout_path) if stdout_path.is_file() else None,
                    "stderr_sha256": sha256(stderr_path) if stderr_path.is_file() else None,
                },
            )
            raise


def top_k(weights: np.ndarray, k: int) -> np.ndarray:
    """Select exactly k weights using deterministic input-index tie breaking."""

    order = np.lexsort((np.arange(weights.size), -weights))
    labels = np.zeros(weights.size, dtype=bool)
    labels[order[:k]] = True
    return labels


def overlap(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Compute binary set agreement without assigning ground truth."""

    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    total = int(a.size)
    return {
        "intersection_size": intersection,
        "union_size": union,
        "jaccard": intersection / union if union else 1.0,
        "dice": 2.0 * intersection / (int(a.sum()) + int(b.sum())),
        "overlap_fraction": intersection / min(int(a.sum()), int(b.sum())),
        "exact_match": bool(np.array_equal(a, b)),
        "n_atoms": total,
    }


def disagreement_segments(
    a: np.ndarray,
    b: np.ndarray,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return contiguous chain-aware XOR segments."""

    differing = a ^ b
    segments = []
    start = None
    for index in range(differing.size + 1):
        is_difference = index < differing.size and bool(differing[index])
        same_chain = (
            start is not None
            and index < differing.size
            and metadata[index]["chain"] == metadata[index - 1]["chain"]
        )
        if is_difference and start is None:
            start = index
        elif start is not None and (not is_difference or not same_chain):
            end = index - 1
            segments.append(
                {
                    "chain": metadata[start]["chain"],
                    "start_atom_index": start,
                    "end_atom_index": end,
                    "start_residue_id": metadata[start]["residue_id"],
                    "end_residue_id": metadata[end]["residue_id"],
                    "length": end - start + 1,
                }
            )
            start = index if is_difference else None
    return segments


def main() -> None:
    destination = OUT / "06_core_extraction"
    panel = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    full = [row for row in panel if row["prefix_id"] == "f10001"]

    audit_runs: dict[str, list[dict[str, Any]]] = {}
    self_rows = []
    for row in full:
        runs = [
            run_mdlovofit_export(row, "self_audit_1"),
            run_mdlovofit_export(row, "self_audit_2"),
        ]
        audit_runs[setting_id(row)] = runs
        metrics = overlap(runs[0]["mask"], runs[1]["mask"])
        self_rows.append(
            {
                "setting_id": setting_id(row),
                "run_1_sha256": runs[0]["mask_sha256"],
                "run_2_sha256": runs[1]["mask_sha256"],
                "self_jaccard": metrics["jaccard"],
                "exact_match": metrics["exact_match"],
                "core_size_run_1": runs[0]["core_size"],
                "core_size_run_2": runs[1]["core_size"],
            }
        )
        write_tsv(destination / "mdlovofit_self_consistency.tsv", self_rows)
    stochastic = any(not row["exact_match"] for row in self_rows)

    mdca_weight_rows = []
    mdca_core_rows = []
    mdlovo_core_rows = []
    mapping_rows = []
    agreement_rows = []
    for index, row in enumerate(panel, start=1):
        sid = setting_id(row)
        source = absolute_input(row)
        metadata = residue_metadata(source, int(row["n_ca"]))
        result = solve_minimal_from_pdb(
            source,
            mode="throughput",
            backend="cpu",
            expected_frames=int(row["n_frames"]),
            expected_residues=int(row["n_ca"]),
        )
        weights = np.asarray(result.frozen_weights, dtype=np.float64)
        native = np.asarray(result.core_labels, dtype=bool)
        if stochastic and sid in audit_runs:
            mdlovo_runs = audit_runs[sid] + [run_mdlovofit_export(row, "stochastic_run_3")]
        else:
            extraction_count = 3 if stochastic else 1
            mdlovo_runs = [
                run_mdlovofit_export(row, f"registered_{run_index + 1}")
                for run_index in range(extraction_count)
            ]
        primary_run = mdlovo_runs[0]
        mdlovo_mask = np.asarray(primary_run["mask"], dtype=bool)
        expected_labels = [
            (atom["chain"], atom["residue_id"], atom["resname"]) for atom in metadata
        ]
        atom_order_match = all(
            mdlovo_run["residue_labels"] == expected_labels for mdlovo_run in mdlovo_runs
        )
        model_count_match = all(
            mdlovo_run["model_count"] == int(row["n_frames"]) for mdlovo_run in mdlovo_runs
        )
        if not atom_order_match or not model_count_match:
            raise ValueError(f"MDLovoFit mapping/model-count mismatch for {sid}")
        matched = top_k(weights, int(mdlovo_mask.sum()))

        for atom, weight, native_label, matched_label in zip(
            metadata,
            weights,
            native,
            matched,
            strict=True,
        ):
            base = {
                "setting_id": sid,
                "system_id": row["system_id"],
                "prefix_id": row["prefix_id"],
                **atom,
            }
            mdca_weight_rows.append({**base, "stable_weight": weight})
            mdca_core_rows.append(
                {
                    **base,
                    "native_60_core": int(native_label),
                    "matched_k_core": int(matched_label),
                }
            )
        for mdlovo_run in mdlovo_runs:
            for atom, mdlovo_label in zip(
                metadata,
                np.asarray(mdlovo_run["mask"], dtype=bool),
                strict=True,
            ):
                mdlovo_core_rows.append(
                    {
                        "setting_id": sid,
                        "system_id": row["system_id"],
                        "prefix_id": row["prefix_id"],
                        **atom,
                        "extraction_id": mdlovo_run["extraction_id"],
                        "native_60_core": int(mdlovo_label),
                    }
                )
        native_metrics = overlap(native, mdlovo_mask)
        matched_metrics = overlap(matched, mdlovo_mask)
        replicate_jaccards = []
        replicate_dice = []
        for mdlovo_run in mdlovo_runs:
            run_mask = np.asarray(mdlovo_run["mask"], dtype=bool)
            run_matched = top_k(weights, int(run_mask.sum()))
            run_metrics = overlap(run_matched, run_mask)
            replicate_jaccards.append(run_metrics["jaccard"])
            replicate_dice.append(run_metrics["dice"])
        segments = disagreement_segments(matched, mdlovo_mask, metadata)
        agreement_rows.append(
            {
                "setting_id": sid,
                "system_id": row["system_id"],
                "prefix_id": row["prefix_id"],
                "n_frames": row["n_frames"],
                "n_ca": row["n_ca"],
                "mdcorealign_native_core_size": int(native.sum()),
                "mdlovofit_native_core_size": int(mdlovo_mask.sum()),
                "matched_k": int(mdlovo_mask.sum()),
                **{f"matched_k_{key}": value for key, value in matched_metrics.items()},
                "matched_k_jaccard_across_extractions_median": float(np.median(replicate_jaccards)),
                "matched_k_jaccard_across_extractions_min": min(replicate_jaccards),
                "matched_k_jaccard_across_extractions_max": max(replicate_jaccards),
                "matched_k_dice_across_extractions_median": float(np.median(replicate_dice)),
                "native_jaccard": native_metrics["jaccard"],
                "native_dice": native_metrics["dice"],
                "disagreement_segment_count": len(segments),
                "disagreement_segment_sizes": ",".join(
                    str(segment["length"]) for segment in segments
                ),
                "disagreement_segments_json": json.dumps(segments, separators=(",", ":")),
            }
        )
        mapping_rows.append(
            {
                "setting_id": sid,
                "expected_atoms": row["n_ca"],
                "mapped_atoms": len(metadata),
                "mdlovofit_models": primary_run["model_count"],
                "expected_models": row["n_frames"],
                "atom_order_match": atom_order_match,
                "model_count_match": model_count_match,
                "core_count_valid": 0 < int(mdlovo_mask.sum()) < int(row["n_ca"]),
                "status": "pass",
            }
        )
        write_tsv(destination / "mdcorealign_weights.tsv", mdca_weight_rows)
        write_tsv(destination / "mdcorealign_cores.tsv", mdca_core_rows)
        write_tsv(destination / "mdlovofit_cores.tsv", mdlovo_core_rows)
        write_tsv(destination / "core_mapping_audit.tsv", mapping_rows)
        write_tsv(destination / "core_agreement_per_setting.tsv", agreement_rows)
        print(f"AT6_COMPLETE {index}/48 {sid}", flush=True)

    write_text(
        destination / "core_extraction_report.md",
        "\n".join(
            [
                "# Untimed core extraction report",
                "",
                f"- Settings completed: `{len(agreement_rows)}/48`",
                f"- MDLovoFit stochasticity detected: `{'yes' if stochastic else 'no'}`",
                "- MDLovoFit membership source: `occupancy == 1.00` in stock `-t` output.",
                "- Successful routine aligned exports were temporary, untimed, parsed, and deleted.",
                "- The persistent parser-recovery export and any failed-export PDB/log evidence are retained for audit.",
                "- Independent untimed exports may be prefetched on distinct logical CPUs; scheduling is recorded separately in prefetch_audit/ and contributes no AT5 timing records.",
                "- Validated export checkpoints are reused only after checking frozen input/executable hashes, extraction identity, model mapping, core mask, and log hashes.",
                "- Primary comparison: exact top-k MDCoreAlign weights where k is the MDLovoFit core size.",
                "- Interpretation: method agreement only; neither core is biological ground truth.",
            ]
        ),
    )


if __name__ == "__main__":
    main()
