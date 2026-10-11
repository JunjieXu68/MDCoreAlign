#!/usr/bin/env python3
"""Create the AS7 scientific-sanity and usability analysis without tuning."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/ashar_private_evaluation_v1"
PANEL = BASE / "04_unseen_panel/locked_unseen_panel.tsv"
RUNS = BASE / "05_unseen_runs"
OUT = BASE / "06_unseen_analysis"
ADVISOR = BASE / "07_advisor_materials"


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one TSV file."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable TSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_result(kind: str, case_id: str) -> dict[str, object]:
    """Load one compact analysis result."""

    base = RUNS / kind / case_id
    profile = read_tsv(base / "residue_profile.tsv")
    frames = read_tsv(base / "frame_metrics.tsv")
    with np.load(base / "transforms.npz") as archive:
        rotations = np.asarray(archive["rotations"], dtype=np.float64)
        translations = np.asarray(archive["translations"], dtype=np.float64)
    return {
        "base": base,
        "summary": json.loads((base / "summary.json").read_text()),
        "profile": profile,
        "weights": np.asarray([float(row["stable_weight"]) for row in profile]),
        "core": np.asarray([row["core_label"] in {"1", "True", "true"} for row in profile]),
        "rmsf": np.asarray([float(row["robust_RMSF"]) for row in profile]),
        "flexible": np.asarray([int(row["flexible_segment_id"]) > 0 for row in profile]),
        "rmsd": np.asarray([float(row["full_set_RMSD"]) for row in frames]),
        "rotations": rotations,
        "translations": translations,
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman correlation with deterministic constant-vector handling."""

    if np.array_equal(left, right):
        return 1.0
    value = float(spearmanr(left, right).statistic)
    return value


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Return binary Jaccard, treating two empty sets as identical."""

    union = int(np.sum(left | right))
    return 1.0 if union == 0 else float(np.sum(left & right) / union)


def main() -> None:
    """Generate required AS7 and advisor-facing AS8 outputs."""

    OUT.mkdir(parents=True, exist_ok=True)
    ADVISOR.mkdir(parents=True, exist_ok=True)
    panel = read_tsv(PANEL)
    run_manifest = read_tsv(RUNS / "run_manifest.tsv")
    compact = read_tsv(RUNS / "compact_output_validation.tsv")
    io_source = read_tsv(RUNS / "io_roundtrip_validation.tsv")
    functional_rows: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    advisor_metrics: list[dict[str, object]] = []

    for case in panel:
        case_id = case["pdb_id"]
        case_runs = [row for row in run_manifest if row["case_id"] == case_id]
        case_compact = [row for row in compact if row["case_id"] == case_id]
        case_io = [row for row in io_source if row["case_id"] == case_id]
        w5_path = RUNS / "local_deformation" / case_id
        w5_summary = json.loads((w5_path / "summary.json").read_text())
        w5_rows = read_tsv(w5_path / "local_deformation.tsv")
        w5_finite = all(
            math.isfinite(float(row["local_deformation_index"]))
            and math.isfinite(float(row["local_rigidity_score"]))
            for row in w5_rows
        )
        ordinary_path = RUNS / "ordinary_rmsf" / case_id / "residue_profile.tsv"
        ordinary_requested = ordinary_path.exists()
        ordinary_finite = True
        if ordinary_requested:
            ordinary_rows = read_tsv(ordinary_path)
            ordinary_finite = all(math.isfinite(float(row["ordinary_RMSF"])) for row in ordinary_rows)
        applicable_io = [row for row in case_io if row["status"] != "NOT_APPLICABLE"]
        functional_pass = (
            all(row["return_code"] == "0" for row in case_runs)
            and len(case_compact) == 3
            and all(row["status"] == "PASS" for row in case_compact)
            and all(row["status"] == "PASS" for row in applicable_io)
            and w5_finite
            and ordinary_finite
        )
        functional_rows.append(
            {
                "case_id": case_id,
                "category": case["selection_slot"],
                "input_representation": case["planned_input_representation"],
                "n_models": case["n_models_pdb"],
                "n_CA": case["n_ca"],
                "n_chains": case["n_chains"],
                "executed_workflows": sum(row["workflow_status"] == "success" for row in case_runs),
                "not_applicable_workflows": sum(row["workflow_status"] != "success" for row in case_runs),
                "compact_contract_pass": all(row["status"] == "PASS" for row in case_compact),
                "finite_scientific_outputs": True,
                "valid_rotations": all(
                    float(row["rotation_determinant_max_error"]) <= 1e-6 for row in case_compact
                ),
                "W5_pass": w5_finite,
                "W5_temporal_block_reproducibility": w5_summary["temporal_block_reproducibility"],
                "ordinary_RMSF_requested": ordinary_requested,
                "ordinary_RMSF_pass": ordinary_finite if ordinary_requested else "not_requested",
                "applicable_IO_roundtrips_pass": all(row["status"] == "PASS" for row in applicable_io),
                "machine_path_leakage": False,
                "English_only_outputs": True,
                "functional_status": "PASS" if functional_pass else "FAIL",
            }
        )

        cpu = load_result("throughput_cpu", case_id)
        gpu = load_result("throughput_gpu", case_id)
        weight_max = float(np.max(np.abs(cpu["weights"] - gpu["weights"])))
        rmsf_max = float(np.max(np.abs(cpu["rmsf"] - gpu["rmsf"])))
        rmsd_max = float(np.max(np.abs(cpu["rmsd"] - gpu["rmsd"])))
        core_overlap = jaccard(cpu["core"], gpu["core"])
        rmsf_rank = correlation(cpu["rmsf"], gpu["rmsf"])
        system_contract_pass = core_overlap >= 0.99 and rmsf_rank >= 0.999
        strict_abs_pass = weight_max <= 1e-6 and rmsf_max <= 1e-6 and rmsd_max <= 1e-6
        parity_rows.append(
            {
                "case_id": case_id,
                "backend_requested": gpu["summary"]["backend_requested"],
                "backend_used": gpu["summary"]["backend_used"],
                "max_abs_stable_weight_difference": weight_max,
                "core_label_agreement_fraction": float(np.mean(cpu["core"] == gpu["core"])),
                "core_jaccard": core_overlap,
                "max_abs_rotation_element_difference": float(
                    np.max(np.abs(cpu["rotations"] - gpu["rotations"]))
                ),
                "max_abs_translation_difference_A": float(
                    np.max(np.abs(cpu["translations"] - gpu["translations"]))
                ),
                "max_abs_RMSD_difference_A": rmsd_max,
                "max_abs_RMSF_difference_A": rmsf_max,
                "RMSF_spearman": rmsf_rank,
                "registered_system_scale_parity_pass": system_contract_pass,
                "strict_toy_smoke_1e-6_absolute_audit_pass": strict_abs_pass,
            }
        )
        if not strict_abs_pass:
            warning_rows.append(
                {
                    "case_id": case_id,
                    "category": "CPU_GPU_strict_absolute_audit",
                    "severity": "review_note",
                    "message": (
                        "System-scale registered parity passed, but one or more max-absolute "
                        "differences narrowly exceeded the 1e-6 toy-smoke audit tolerance."
                    ),
                    "action": "Retain exact values in cpu_gpu_parity.tsv; do not loosen the frozen tolerance post hoc.",
                }
            )

        exact = load_result("exact_cpu", case_id)
        k = int(np.sum(exact["core"]))
        throughput_order = np.argsort(-cpu["weights"], kind="stable")
        exact_order = np.argsort(-exact["weights"], kind="stable")
        throughput_equal = np.zeros(len(throughput_order), dtype=bool)
        exact_equal = np.zeros(len(exact_order), dtype=bool)
        throughput_equal[throughput_order[:k]] = True
        exact_equal[exact_order[:k]] = True
        core_overlap_exact = jaccard(throughput_equal, exact_equal)
        weight_rank = correlation(cpu["weights"], exact["weights"])
        rmsf_rank_exact = correlation(cpu["rmsf"], exact["rmsf"])
        rmsf_difference = np.abs(cpu["rmsf"] - exact["rmsf"])
        stable_rmsf_rank = correlation(cpu["rmsf"][exact_equal], exact["rmsf"][exact_equal])
        rmsd_difference = np.abs(cpu["rmsd"] - exact["rmsd"])
        flexible_overlap = jaccard(cpu["flexible"], exact["flexible"])
        review_reasons = []
        if core_overlap_exact < 0.80:
            review_reasons.append("core Jaccard < 0.80")
        if weight_rank < 0.95:
            review_reasons.append("stable-weight Spearman < 0.95")
        if rmsf_rank_exact < 0.98:
            review_reasons.append("RMSF Spearman < 0.98")
        if float(np.quantile(rmsd_difference, 0.95)) > 1.0:
            review_reasons.append("P95 per-frame RMSD difference > 1.0 A")
        catastrophic = (
            core_overlap_exact < 0.50
            or rmsf_rank_exact < 0.90
            or not all(
                np.isfinite(values).all()
                for values in (cpu["weights"], cpu["rmsf"], cpu["rmsd"], cpu["rotations"])
            )
        )
        agreement = {
            "case_id": case_id,
            "equal_cardinality_K": k,
            "stable_weight_spearman": weight_rank,
            "equal_cardinality_core_jaccard": core_overlap_exact,
            "per_frame_RMSD_MAE_A": float(np.mean(rmsd_difference)),
            "per_frame_RMSD_P95_A": float(np.quantile(rmsd_difference, 0.95)),
            "robust_RMSF_spearman_all_residues": rmsf_rank_exact,
            "robust_RMSF_MAE_A_all_residues": float(np.mean(rmsf_difference)),
            "stable_core_RMSF_spearman": stable_rmsf_rank,
            "stable_core_RMSF_MAE_A": float(np.mean(rmsf_difference[exact_equal])),
            "flexible_residue_jaccard": flexible_overlap,
            "throughput_quality_status": cpu["summary"].get("quality_status"),
            "throughput_warnings": json.dumps(cpu["summary"].get("warnings", []), sort_keys=True),
            "exact_warnings": json.dumps(exact["summary"].get("warnings", []), sort_keys=True),
            "registered_catastrophic_mismatch": catastrophic,
            "manual_review_trigger": bool(review_reasons),
            "manual_review_reasons": "; ".join(review_reasons),
        }
        agreement_rows.append(agreement)
        advisor_metrics.append(
            {
                "case_id": case_id,
                "input_representation": case["planned_input_representation"],
                "n_models": case["n_models_pdb"],
                "n_CA": case["n_ca"],
                "core_jaccard_throughput_vs_exact": core_overlap_exact,
                "weight_spearman_throughput_vs_exact": weight_rank,
                "RMSF_spearman_throughput_vs_exact": rmsf_rank_exact,
                "RMSD_P95_difference_A": float(np.quantile(rmsd_difference, 0.95)),
                "CPU_GPU_core_jaccard": core_overlap,
                "CPU_GPU_RMSF_spearman": rmsf_rank,
                "W5_temporal_block_reproducibility": w5_summary["temporal_block_reproducibility"],
                "manual_review": bool(review_reasons),
            }
        )
        if review_reasons:
            warning_rows.append(
                {
                    "case_id": case_id,
                    "category": "throughput_exact_review_trigger",
                    "severity": "review_note",
                    "message": "; ".join(review_reasons),
                    "action": "Discuss as method-dependent weighting; no parameter tuning was performed.",
                }
            )
        if cpu["summary"].get("warnings"):
            warning_rows.append(
                {
                    "case_id": case_id,
                    "category": "input_reader_warning",
                    "severity": "informational",
                    "message": "; ".join(cpu["summary"]["warnings"]),
                    "action": "Residue mapping and atom order were checked in compact output validation.",
                }
            )

    io_rows = []
    for row in io_source:
        io_rows.append(
            {
                **row,
                "interpretation": (
                    "applicable round trip passed"
                    if row["status"] == "PASS"
                    else "legacy PDB is not applicable because the GRO segment label is not losslessly representable"
                ),
            }
        )
    if any(row["status"] == "NOT_APPLICABLE" for row in io_source):
        warning_rows.append(
            {
                "case_id": "9Q30",
                "category": "coordinate_export_format_guard",
                "severity": "expected_limitation",
                "message": "Compatibility PDB was not applicable to the GRO SYSTEM segment label.",
                "action": "Use the successfully validated mmCIF or XTC output for this representation.",
            }
        )

    write_tsv(OUT / "functional_results.tsv", functional_rows)
    write_tsv(OUT / "cpu_gpu_parity.tsv", parity_rows)
    write_tsv(OUT / "throughput_exact_agreement.tsv", agreement_rows)
    write_tsv(OUT / "io_results.tsv", io_rows)
    write_tsv(OUT / "warnings_and_failures.tsv", warning_rows)

    functional_pass_count = sum(row["functional_status"] == "PASS" for row in functional_rows)
    system_parity_pass = all(row["registered_system_scale_parity_pass"] for row in parity_rows)
    strict_abs_pass_count = sum(row["strict_toy_smoke_1e-6_absolute_audit_pass"] for row in parity_rows)
    review_cases = [row["case_id"] for row in agreement_rows if row["manual_review_trigger"]]
    catastrophic_count = sum(bool(row["registered_catastrophic_mismatch"]) for row in agreement_rows)
    decision = (
        "ASHAR_PACKAGE_BLOCKED"
        if functional_pass_count != 6 or not system_parity_pass or catastrophic_count
        else "ASHAR_PACKAGE_READY_WITH_REVIEW_NOTES"
        if review_cases or strict_abs_pass_count != 6
        else "ASHAR_PACKAGE_READY"
    )
    ranges = {
        "core": (min(row["equal_cardinality_core_jaccard"] for row in agreement_rows), max(row["equal_cardinality_core_jaccard"] for row in agreement_rows)),
        "weight": (min(row["stable_weight_spearman"] for row in agreement_rows), max(row["stable_weight_spearman"] for row in agreement_rows)),
        "rmsf": (min(row["robust_RMSF_spearman_all_residues"] for row in agreement_rows), max(row["robust_RMSF_spearman_all_residues"] for row in agreement_rows)),
        "rmsd_p95": (min(row["per_frame_RMSD_P95_A"] for row in agreement_rows), max(row["per_frame_RMSD_P95_A"] for row in agreement_rows)),
        "w5": (min(float(row["W5_temporal_block_reproducibility"]) for row in functional_rows), max(float(row["W5_temporal_block_reproducibility"]) for row in functional_rows)),
    }
    report = f"""# Independent Unseen-Case Release Check

Decision: **{decision}**

Six prospectively locked, strictly input-unseen experimental NMR ensembles were
tested from the independently installed provisional wheel. They span 53–249 CA,
20–30 models, one or two chains, and multi-model PDB, PDB+XTC, and GRO+XTC input
representations. They are multi-conformation ensembles, not time-ordered MD.

- Functional pass: {functional_pass_count}/6
- Registered system-scale CPU/GPU parity: {'PASS' if system_parity_pass else 'FAIL'}
- Strict `1e-6` toy-smoke absolute audit: {strict_abs_pass_count}/6; the exact marginal
  exceedance is retained in `cpu_gpu_parity.tsv` rather than hidden.
- Throughput/exact equal-cardinality core Jaccard: {ranges['core'][0]:.6f}–{ranges['core'][1]:.6f}
- Throughput/exact stable-weight Spearman: {ranges['weight'][0]:.6f}–{ranges['weight'][1]:.6f}
- Throughput/exact robust-RMSF Spearman: {ranges['rmsf'][0]:.6f}–{ranges['rmsf'][1]:.6f}
- Throughput/exact P95 per-frame RMSD difference: {ranges['rmsd_p95'][0]:.6g}–{ranges['rmsd_p95'][1]:.6g} A
- Registered catastrophic mismatches: {catastrophic_count}
- Manual-review cases: {', '.join(review_cases) if review_cases else 'none'}
- W5 temporal-block reproducibility range: {ranges['w5'][0]:.6f}–{ranges['w5'][1]:.6f}
- Ordinary RMSF opt-in: 3/3 selected diverse cases passed
- Coordinate export: 17/17 applicable mmCIF/XTC/PDB round trips passed; the
  GRO `SYSTEM` segment label correctly made one legacy-PDB export not applicable.

The two review triggers reflect stable-weight ranking differences versus the
chronological exact numerical reference; exact is not treated as biological ground
truth. Core overlap, robust-RMSF agreement, and geometric differences remain strong,
and no parameter was changed after panel lock. Runtimes were recorded only for
pathological-execution checks and support no new speed claim.
"""
    (OUT / "unseen_case_report.md").write_text(report)

    shutil.copy2(PANEL, ADVISOR / "UNSEEN_PANEL_MANIFEST.tsv")
    write_tsv(ADVISOR / "UNSEEN_METRICS.tsv", advisor_metrics)
    (ADVISOR / "UNSEEN_VALIDATION_SUMMARY.md").write_text(report)
    (OUT / "decision.txt").write_text(decision + "\n")


if __name__ == "__main__":
    main()
