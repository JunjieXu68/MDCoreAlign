#!/usr/bin/env python
"""Calibrate and hash the deterministic adaptive chronology rule."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdsuperpose.chronology_adaptive import adaptive_risk_features  # noqa: E402
from mdsuperpose.chronology_study import (  # noqa: E402
    compare_results,
    representative_indices,
    run_candidate,
    run_exact,
)
from scripts.cpu_chronology_v3.run_self_variability import load_case, read_tsv  # noqa: E402


OUT = ROOT / "results/cpu_chronology_v3"
CALIBRATION_OUTPUT = OUT / "04_candidates/adaptive_rule_calibration.tsv"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def calibration_panel() -> list[dict[str, str]]:
    rows = read_tsv(OUT / "02_self_variability/calibration_panel.tsv")
    known = {row["case_id"] for row in rows}
    for row in read_tsv(ROOT / "results/cpu_fast_path_v2/02_real_panel/development_panel.tsv"):
        if row["case_id"] in known:
            continue
        converted = dict(row)
        converted["panel_role"] = "v2_development_family"
        converted["domain_id"] = row["family_id"].split(":", 1)[-1]
        rows.append(converted)
        known.add(row["case_id"])
    return rows


def prefix(metrics: dict[str, object], label: str) -> dict[str, object]:
    return {f"{label}_{key}": value for key, value in metrics.items()}


def run_calibration() -> None:
    rows = calibration_panel()
    completed = set()
    if CALIBRATION_OUTPUT.exists() and CALIBRATION_OUTPUT.stat().st_size:
        completed = {row["case_id"] for row in read_tsv(CALIBRATION_OUTPUT)}
    start = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        if row["case_id"] in completed:
            continue
        coords, chains = load_case(row)
        exact = run_exact(coords, chains)
        throughput = run_candidate(coords, chains, "throughput_compact_f64")
        c1 = run_candidate(coords, chains, "C_correction_1_calibration")
        c2 = run_candidate(coords, chains, "C_correction_2_v3")
        d64 = run_candidate(coords, chains, "D_block_64_v3")
        selected = representative_indices(coords.shape[0])
        representatives = np.asarray(coords[selected], dtype=np.float64)
        reference = np.asarray(coords[0], dtype=np.float64)
        c1_features = adaptive_risk_features(
            representatives, reference, throughput.weights, c1.weights
        )
        c2_features = adaptive_risk_features(
            representatives, reference, throughput.weights, c2.weights
        )
        output = {
            "case_id": row["case_id"],
            "family_id": row["family_id"],
            "panel_role": row["panel_role"],
            "n_frames": int(row["n_frames"]),
            "n_residues": int(row["n_residues"]),
            **{f"c1_feature_{key}": value for key, value in c1_features.items()},
            **{f"c2_feature_{key}": value for key, value in c2_features.items()},
            **prefix(compare_results(throughput, exact, coords), "throughput"),
            **prefix(compare_results(c1, exact, coords), "c1"),
            **prefix(compare_results(c2, exact, coords), "c2"),
            **prefix(compare_results(d64, exact, coords), "d64"),
        }
        exists = CALIBRATION_OUTPUT.exists() and CALIBRATION_OUTPUT.stat().st_size > 0
        with CALIBRATION_OUTPUT.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output), delimiter="\t")
            if not exists:
                writer.writeheader()
            writer.writerow(output)
            handle.flush()
        print(f"[V3 calibration] {index}/{len(rows)} {row['case_id']} elapsed={time.perf_counter()-start:.1f}s", flush=True)


def unsafe(row: dict[str, str], prefix_name: str) -> bool:
    gates = json.loads((OUT / "03_gate_calibration/v3_gate_registry.json").read_text())["balanced"]
    return bool(
        row[f"{prefix_name}_catastrophic_mismatch"] == "True"
        or row[f"{prefix_name}_non_finite_transform"] == "True"
        or float(row[f"{prefix_name}_core_jaccard"]) < float(gates["minimum_core_jaccard_min"])
        or float(row[f"{prefix_name}_weight_spearman"]) < float(gates["p5_weight_spearman_min"])
        or float(row[f"{prefix_name}_rmsd_absolute_difference_p95_A"]) > float(gates["p95_of_case_p95_rmsd_difference_A_max"])
        or float(row[f"{prefix_name}_stable_core_rmsf_spearman"]) < float(gates["p5_stable_core_rmsf_spearman_min"])
    )


def choose_rule(rows: list[dict[str, str]], prefix_name: str, eligible: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    keys = ("core_jaccard_vs_initial", "weight_spearman_vs_initial", "cutoff_margin", "residual_tail_p95")
    arrays = {key: np.asarray([float(row[f"{prefix_name}_feature_{key}"]) for row in rows]) for key in keys}
    unsafe_mask = np.asarray([unsafe(row, prefix_name) for row in rows]) & eligible
    quantiles = (0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 0.95, 0.99, 1.0)
    lower_grids = [sorted(set(float(np.quantile(arrays[key][eligible], q)) for q in quantiles)) for key in keys[:3]]
    upper_grid = sorted(set(float(np.quantile(arrays[keys[3]][eligible], q)) for q in quantiles))
    lower_grids = [[values[0] - 1e-12, *values, values[-1] + 1e-12] for values in lower_grids]
    upper_grid = [upper_grid[0] - 1e-12, *upper_grid, upper_grid[-1] + 1e-12]
    best: tuple[int, float, dict[str, float], np.ndarray] | None = None
    for core, spearman, margin, residual in itertools.product(*lower_grids, upper_grid):
        flagged = eligible & (
            (arrays[keys[0]] < core)
            | (arrays[keys[1]] < spearman)
            | (arrays[keys[2]] < margin)
            | (arrays[keys[3]] > residual)
        )
        false_negatives = int(np.count_nonzero(unsafe_mask & ~flagged))
        if false_negatives:
            continue
        flag_count = int(np.count_nonzero(flagged))
        distance = core + spearman - margin - residual * 1e-6
        rule = {
            "core_jaccard_min": core,
            "weight_spearman_min": spearman,
            "cutoff_margin_min": margin,
            "residual_tail_max": residual,
        }
        candidate = (flag_count, -distance, rule, flagged)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"no monotonic {prefix_name} risk rule catches every unsafe calibration case")
    return best[2], best[3]


def lock_rule() -> None:
    rule_path = OUT / "04_candidates/adaptive_rule.json"
    if rule_path.exists():
        raise RuntimeError("adaptive rule is already locked")
    rows = read_tsv(CALIBRATION_OUTPUT)
    eligible = np.ones(len(rows), dtype=bool)
    c1_rule, c1_flagged = choose_rule(rows, "c1", eligible)
    c2_rule, c2_flagged = choose_rule(rows, "c2", c1_flagged)
    accepted = np.where(~c1_flagged, "C1", np.where(~c2_flagged, "C2", "D64"))
    calibration_rows = []
    for index, row in enumerate(rows):
        final_prefix = accepted[index].lower()
        final_prefix = "d64" if final_prefix == "d64" else final_prefix
        calibration_rows.append(
            {
                "case_id": row["case_id"],
                "accepted_level": accepted[index],
                "c1_unsafe_vs_exact": unsafe(row, "c1"),
                "c2_unsafe_vs_exact": unsafe(row, "c2"),
                "final_catastrophic_mismatch": row[f"{final_prefix}_catastrophic_mismatch"],
                "final_core_jaccard": row[f"{final_prefix}_core_jaccard"],
                "final_rmsd_p95_A": row[f"{final_prefix}_rmsd_absolute_difference_p95_A"],
            }
        )
    write_tsv(OUT / "04_candidates/adaptive_rule_routes.tsv", calibration_rows)
    payload = {
        "rule_version": "adaptive_C1_C2_D64_v3_locked_1",
        "training_sources": ["V1 calibration panel", "v2 development families", "v2 registered challenges"],
        "c1": c1_rule,
        "c2": c2_rule,
        "calibration_cases": len(rows),
        "c1_acceptance_rate": float(np.mean(accepted == "C1")),
        "c2_acceptance_rate": float(np.mean(accepted == "C2")),
        "d64_refinement_rate": float(np.mean(accepted == "D64")),
        "calibration_final_catastrophic_count": sum(row["final_catastrophic_mismatch"] == "True" for row in calibration_rows),
    }
    rule_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(rule_path.read_bytes()).hexdigest()
    (OUT / "04_candidates/adaptive_rule.sha256").write_text(f"{digest}  adaptive_rule.json\n")
    (OUT / "04_candidates/adaptive_rule.md").write_text(
        f"""# Locked adaptive C1 to C2 to D64 rule

The rule uses four monotonic internal signals available during the fast run:
matched-core agreement with batch initialisation, weight-rank agreement,
the deterministic 60% cutoff margin and representative residual-tail P95.

Low-risk C1 is accepted first. Flagged cases run C2; cases still flagged run
frozen D64. Whole-trajectory sequential exact is never used as production
fallback. The thresholds were selected only on calibration/development/tail
data to catch every calibration case failing the per-case balanced safeguards
while minimising escalation count.

- C1 thresholds: `{json.dumps(c1_rule, sort_keys=True)}`
- C2 thresholds: `{json.dumps(c2_rule, sort_keys=True)}`
- Calibration C1 acceptance: `{payload['c1_acceptance_rate']:.3f}`
- Calibration C2 acceptance: `{payload['c2_acceptance_rate']:.3f}`
- Calibration D64 rate: `{payload['d64_refinement_rate']:.3f}`
- Final calibration catastrophic mismatches: `{payload['calibration_final_catastrophic_count']}`

This shallow rule is frozen before fresh-panel selection and may not be retuned
after held-out results are inspected.
"""
    )
    registry = [
        {"configuration": "exact_sequential_f64_compact", "role": "unchanged reference", "public_mode": False, "parameters": "chronological float64; 2000 representatives"},
        {"configuration": "throughput_compact_f64", "role": "historical fast comparator", "public_mode": False, "parameters": "shared float64; 4 global rounds"},
        {"configuration": "D_block_64_v3", "role": "exact-like candidate", "public_mode": False, "parameters": "frozen block size 64"},
        {"configuration": "C_correction_2_v3", "role": "balanced candidate", "public_mode": False, "parameters": "shared initialisation; two chronological corrections"},
        {"configuration": "adaptive_C1_C2_D64_v3", "role": "adaptive candidate", "public_mode": False, "parameters": f"rule_sha256={digest}"},
    ]
    write_tsv(OUT / "04_candidates/candidate_registry.tsv", registry)
    write_tsv(OUT / "04_candidates/locked_candidate_configs.tsv", registry)
    (OUT / "04_candidates/C2_definition.md").write_text("# Frozen C2 definition\n\nShared float64 four-round batch initialisation followed by two bounded chronological correction iterations per representative and unchanged float64 all-frame fitting.\n")
    feature_rows = [
        {"feature": "core_jaccard_vs_initial", "direction": "lower_is_risk", "available_without_exact": True},
        {"feature": "weight_spearman_vs_initial", "direction": "lower_is_risk", "available_without_exact": True},
        {"feature": "cutoff_margin", "direction": "lower_is_risk", "available_without_exact": True},
        {"feature": "residual_tail_p95", "direction": "higher_is_risk", "available_without_exact": True},
    ]
    write_tsv(OUT / "04_candidates/adaptive_risk_features.tsv", feature_rows)
    print(json.dumps({"adaptive_rule_sha256": digest, **payload}, indent=2))


def main() -> None:
    if not CALIBRATION_OUTPUT.exists():
        run_calibration()
    lock_rule()


if __name__ == "__main__":
    main()
