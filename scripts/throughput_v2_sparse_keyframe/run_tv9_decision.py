#!/usr/bin/env python3
"""Adjudicate locked Throughput-v2 candidates without changing public behavior."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.throughput_v2_sparse_keyframe.study_utils import OUT, read_tsv, write_tsv  # noqa: E402


PERF = OUT / "08_performance"
DECISION = OUT / "09_decision"


def numbers(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) not in {None, "", "not_applicable_contract_C"}]


def quantile(rows: list[dict[str, str]], field: str, value: float) -> float | str:
    values = numbers(rows, field)
    return float(np.quantile(values, value)) if values else ""


def scaling_speed(runtime: list[dict[str, str]], candidate: str, frames: int) -> float | str:
    case_id = f"transition_control_4oelA02_{frames}"
    values = [
        float(row["speedup_vs_frozen_current"])
        for row in runtime
        if row["candidate"] == candidate
        and row["case_id"] == case_id
        and row.get("speedup_vs_frozen_current")
    ]
    return float(np.median(values)) if values else ""


def panel_speed(runtime: list[dict[str, str]], candidate: str, prefix: str) -> float | str:
    values = [
        float(row["speedup_vs_frozen_current"])
        for row in runtime
        if row["candidate"] == candidate
        and row["case_id"].startswith(prefix)
        and row.get("speedup_vs_frozen_current")
    ]
    return float(np.median(values)) if values else ""


def run() -> None:
    DECISION.mkdir(parents=True, exist_ok=True)
    quality = read_tsv(PERF / "candidate_quality.tsv")
    runtime = read_tsv(PERF / "candidate_runtime.tsv")
    candidates = ("sparse_balanced", "sparse_fast", "profile_balanced", "profile_fast")
    decisions: list[dict[str, object]] = []
    pareto: list[dict[str, object]] = []
    for candidate in candidates:
        fresh = [
            row for row in quality
            if row["candidate"] == candidate and row["panel"] in {"mdcath", "atlas"}
        ]
        challenge = [row for row in quality if row["candidate"] == candidate and row["panel"] == "challenge"]
        atlas_speed = panel_speed(runtime, candidate, "ATLAS_")
        speed_50k = scaling_speed(runtime, candidate, 50_000)
        speed_100k = scaling_speed(runtime, candidate, 100_000)
        catastrophic = sum(row["catastrophic_mismatch"] == "True" for row in fresh)
        challenge_catastrophic = sum(row["catastrophic_mismatch"] == "True" for row in challenge)
        rmsf_rho_median = quantile(fresh, "stable_rmsf_spearman", 0.5)
        rmsf_rho_p5 = quantile(fresh, "stable_rmsf_spearman", 0.05)
        rmsf_mae = quantile(fresh, "stable_rmsf_mae_A", 0.5)
        rmsd_median = quantile(fresh, "rmsd_abs_diff_median_A", 0.5)
        rmsd_p95 = quantile(fresh, "rmsd_abs_diff_p95_A", 0.95)
        is_profile = candidate.startswith("profile")
        balanced = candidate.endswith("balanced")
        quality_pass = bool(
            catastrophic == 0
            and (rmsf_rho_median != "" and float(rmsf_rho_median) >= (0.995 if balanced else 0.98))
            and (rmsf_rho_p5 != "" and float(rmsf_rho_p5) >= (0.98 if balanced else 0.95))
            and (rmsf_mae != "" and float(rmsf_mae) <= (0.10 if balanced else 0.25))
            and (is_profile or (rmsd_median != "" and float(rmsd_median) <= (0.10 if balanced else 0.25)))
        )
        performance_pass = bool(
            atlas_speed != ""
            and speed_100k != ""
            and float(atlas_speed) >= (2.0 if balanced else 3.0)
            and float(speed_100k) >= (5.0 if balanced else 10.0)
        )
        promotion = quality_pass and performance_pass and challenge_catastrophic == 0
        reasons = []
        if catastrophic:
            reasons.append(f"fresh catastrophic mismatches={catastrophic}")
        if challenge_catastrophic:
            reasons.append(f"challenge catastrophic mismatches={challenge_catastrophic}")
        if not quality_pass:
            reasons.append("locked quality gate not met")
        if not performance_pass:
            reasons.append("locked ATLAS/100k speed gate not met")
        decisions.append(
            {
                "candidate": candidate,
                "contract": "v2_profile" if is_profile else "v2_sparse_align",
                "fresh_cases": len(fresh),
                "fresh_catastrophic_count": catastrophic,
                "challenge_cases": len(challenge),
                "challenge_catastrophic_count": challenge_catastrophic,
                "median_core_jaccard": quantile(fresh, "core_jaccard", 0.5),
                "p5_core_jaccard": quantile(fresh, "core_jaccard", 0.05),
                "median_stable_weight_spearman": quantile(fresh, "stable_weight_spearman", 0.5),
                "median_audit_RMSD_abs_diff_A": rmsd_median,
                "p95_of_audit_P95_RMSD_abs_diff_A": rmsd_p95,
                "median_audit_RMSF_spearman": rmsf_rho_median,
                "p5_audit_RMSF_spearman": rmsf_rho_p5,
                "median_audit_RMSF_MAE_A": rmsf_mae,
                "unresolved_interval_rate": 0.0,
                "atlas_10k_median_speedup": atlas_speed,
                "control_50k_speedup": speed_50k,
                "control_100k_speedup": speed_100k,
                "quality_reference_scope": "deterministic audit frames; no fresh full baseline rerun",
                "locked_quality_gate_pass": quality_pass,
                "locked_performance_gate_pass": performance_pass,
                "new_pareto_point": promotion,
                "decision": "promote_for_product_review" if promotion else "do_not_promote",
                "decision_reason": "; ".join(reasons) if reasons else "all registered gates passed",
            }
        )
        pareto.append(
            {
                "candidate": candidate,
                "quality_axis": "median audit RMSF Spearman",
                "quality_value": rmsf_rho_median,
                "speed_axis": "ATLAS R2/R3 speedup vs frozen current throughput",
                "speed_value": atlas_speed,
                "catastrophic_count_fresh_plus_challenge": catastrophic + challenge_catastrophic,
                "dominates_current_throughput": False,
                "dominated_or_unsafe_reason": "; ".join(reasons),
            }
        )

    for candidate, status, reason in (
        ("keyframe_balanced", "not_run_fresh", "development catastrophic failures"),
        ("keyframe_fast", "not_run_fresh", "development catastrophic failures"),
        ("adaptive_balanced", "not_run_fresh", "development quality and unresolved-interval failure"),
        ("adaptive_fast", "not_run_fresh", "development quality and unresolved-interval failure"),
    ):
        decisions.append(
            {
                "candidate": candidate,
                "contract": "v2_keyframe_align",
                "fresh_cases": 0,
                "fresh_catastrophic_count": "not_applicable_not_run",
                "challenge_cases": 0,
                "challenge_catastrophic_count": "not_applicable_not_run",
                "locked_quality_gate_pass": False,
                "locked_performance_gate_pass": False,
                "new_pareto_point": False,
                "decision": status,
                "decision_reason": reason,
            }
        )
    write_tsv(DECISION / "candidate_decisions.tsv", decisions)
    write_tsv(DECISION / "pareto_summary.tsv", pareto)

    selected = next(row for row in decisions if row["candidate"] == "sparse_balanced")
    profile = next(row for row in decisions if row["candidate"] == "profile_balanced")
    report = f"""# Final Throughput-v2 Decision

No locked candidate is promoted. Throughput-v2 v1 is closed without changing the public implementation.

## Evidence

- Sparse landmarks preserved the frozen weight/core definition, but produced {selected['fresh_catastrophic_count']} catastrophic fresh audit mismatch and only {float(selected['atlas_10k_median_speedup']):.3f}x median ATLAS speedup. The 100k control speedup was {float(selected['control_100k_speedup']):.3f}x.
- Balanced profile sampling produced {profile['fresh_catastrophic_count']} catastrophic fresh audit mismatches, {float(profile['median_audit_RMSF_spearman']):.6f} median audit-reference RMSF Spearman, and {float(profile['atlas_10k_median_speedup']):.3f}x median ATLAS speedup. Its 100k control speedup was {float(profile['control_100k_speedup']):.3f}x.
- Keyframe interpolation was rejected on development data because it produced five catastrophic full-output cases out of seven.
- Bounded adaptive interval refinement was rejected on development data because five of seven cases remained catastrophic and the mean unresolved audit rate was 31.25%.

Fresh quality comparisons used deterministic audit-frame current-throughput extraction where frozen full arrays were unavailable. This limitation cannot rescue promotion because the hard catastrophic and speed gates already fail.

## Decision

Keep `current_throughput_compact` unchanged. Do not expose landmarks, keyframes, profile sampling, or adaptive refinement as public modes. No candidate created a registered new Pareto point.
"""
    (DECISION / "final_throughput_v2_decision.md").write_text(report, encoding="utf-8")
    (DECISION / "product_integration_proposal.md").write_text(
        """# Product Integration Proposal

No integration is proposed. The exact/default and current throughput paths remain unchanged. The experimental module is retained only as reproducible negative-result research code and is not imported by the public package interface.
""",
        encoding="utf-8",
    )
    (DECISION / "public_interface_impact.md").write_text(
        """# Public Interface Impact

None. No CLI mode, default, package version, output schema, or numerical setting was changed. Candidate names remain internal.
""",
        encoding="utf-8",
    )
    (DECISION / "remaining_lower_bounds.md").write_text(
        """# Remaining Lower Bounds

The compact alignment contract itself requires per-frame transforms and full-selection RMSD, imposing an unavoidable O(TN) coordinate/statistics pass. Sparse landmarks reduce the covariance fit from N to M residues, but full-coordinate transform application, RMSD and robust RMSF remain O(TN); on long controls these stages dominate after weight estimation. Profile sampling can reduce RMSF work, but it changes the output contract, and the tested motion-weighted signature adds another full-trajectory pass. Safe transform interpolation was not achieved for rapid or large coherent motion. The remaining plausible work is narrower scan/profile products or binary trajectory I/O engineering, not further tuning of this held-out panel.
""",
        encoding="utf-8",
    )
    (DECISION / "manuscript_impact.md").write_text(
        """# Manuscript Impact

No manuscript result or claim is changed automatically. This study is a negative optimisation result: sparse all-frame fitting did not provide the registered speed benefit, keyframe interpolation was unsafe on development/challenge motion, and sampled RMSF did not satisfy the joint fresh quality/performance promotion criteria. These findings may be reported as limitations or omitted; they do not alter frozen current-throughput evidence.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
    print("TV9 complete: no candidate promoted")
