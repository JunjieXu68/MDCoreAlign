#!/usr/bin/env python3
"""Validate the benchmark-only compact MDCoreAlign execution paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mdcorealign.benchmark_minimal import (
    load_ca_multimodel_pdb_minimal,
    solve_minimal_cpu,
    solve_minimal_gpu,
)
from mdcorealign.contracts import solve_contract

from scripts.no_materialisation_benchmark_v1.common import OUT, write_json, write_text, write_tsv


DEST = OUT / "02_atlas_stock_no_output"
INPUT = (
    Path("results/atlas_large_long_v1/a3_external/centered_inputs")
    / "1u55_A/f00250_centered_CA.pdb"
)


def _labels(weights: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    count = max(1, int(np.floor(weights.size * fraction + 0.5)))
    order = np.lexsort((np.arange(weights.size), -weights))
    labels = np.zeros(weights.size, dtype=bool)
    labels[order[:count]] = True
    return labels


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _compare(method: str, standard: object, compact: object, tolerance: float) -> dict[str, object]:
    alignment = standard.alignment
    metrics = {
        "weight_max_abs_difference": _max_abs(alignment.frozen_weights, compact.frozen_weights),
        "core_label_mismatch_count": int(np.count_nonzero(_labels(alignment.frozen_weights) != compact.core_labels)),
        "rotation_max_abs_difference": _max_abs(alignment.rotations, compact.rotations),
        "translation_max_abs_difference": _max_abs(alignment.translations, compact.translations),
        "rmsd_max_abs_difference": _max_abs(alignment.rmsd, compact.rmsd),
        "weighted_rmsd_max_abs_difference": _max_abs(alignment.weighted_rmsd, compact.weighted_rmsd),
    }
    passed = (
        metrics["core_label_mismatch_count"] == 0
        and all(
            value <= tolerance
            for key, value in metrics.items()
            if key != "core_label_mismatch_count"
        )
    )
    return {
        "method": method,
        "n_frames": compact.rmsd.size,
        "n_residues": compact.frozen_weights.size,
        "tolerance": tolerance,
        **metrics,
        "parity_pass": passed,
    }


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    coords, chains = load_ca_multimodel_pdb_minimal(
        INPUT,
        expected_frames=250,
        expected_residues=188,
    )
    rows: list[dict[str, object]] = []

    standard_exact = solve_contract(
        coords,
        contract="exact",
        backend="cpu",
        chain_ids=chains,
        representative_count=2000,
        chunk_size=1000,
        store_aligned=False,
    )
    compact_exact = solve_minimal_cpu(
        coords,
        mode="exact",
        chain_ids=chains,
        representative_count=2000,
        chunk_size=1000,
    )
    rows.append(_compare("exact_CPU", standard_exact, compact_exact, 1e-10))

    standard_cpu = solve_contract(
        coords,
        contract="throughput",
        backend="cpu",
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        store_aligned=False,
    )
    compact_cpu = solve_minimal_cpu(
        coords,
        mode="throughput",
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
    )
    rows.append(_compare("throughput_CPU", standard_cpu, compact_cpu, 1e-10))

    standard_gpu = solve_contract(
        coords,
        contract="throughput",
        backend="gpu",
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
        precision="float32",
        rotation_solver="horn",
        store_aligned=False,
    )
    compact_gpu = solve_minimal_gpu(
        coords,
        chain_ids=chains,
        representative_count=256,
        update_rounds=4,
        chunk_size=1000,
    )
    rows.append(_compare("throughput_GPU", standard_gpu, compact_gpu, 2e-4))

    write_tsv(DEST / "mdcorealign_minimal_path_parity.tsv", rows)
    passed = all(bool(row["parity_pass"]) for row in rows)
    write_text(
        DEST / "mdcorealign_minimal_path_report.md",
        "# MDCoreAlign Minimal-path Parity\n\n"
        f"Status: **{'PASS' if passed else 'FAIL'}** for exact CPU, throughput "
        "CPU and throughput GPU on the locked 1u55_A/250-frame input. The "
        "benchmark-only path suppresses RMSF, confidence diagnostics, aligned "
        "coordinate retention and serialisation, while preserving frozen weights, "
        "hard-core labels, transforms and RMSD outputs within the registered "
        "backend-specific tolerances. It is not exposed as a public mode.\n",
    )
    write_json(
        DEST / "nm2_status.json",
        {"status": "pass" if passed else "fail", "rows": len(rows), "all_parity": passed},
    )
    if not passed:
        raise SystemExit("NM2 parity failed; inspect mdcorealign_minimal_path_parity.tsv")
    print(json.dumps({"status": "pass", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
