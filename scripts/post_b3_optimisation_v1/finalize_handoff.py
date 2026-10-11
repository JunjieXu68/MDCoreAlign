#!/usr/bin/env python3
"""Verify frozen evidence and build the compact post-B3 handoff."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from scripts.post_b3_optimisation_v1.common import (
    OUT,
    ROOT,
    read_tsv,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


HANDOFF = OUT / "handoff_to_chatgpt"
FROZEN = OUT / "p0_freeze/frozen_source_evidence_manifest.tsv"


REPORTS = {
    "p1_corrected_benchmark_interpretation.md": OUT
    / "p1_corrected_reporting/corrected_benchmark_interpretation_report.md",
    "p1_corrected_benchmark_summary.tsv": OUT
    / "p1_corrected_reporting/corrected_benchmark_summary.tsv",
    "p2_confidence_v2_report.md": OUT
    / "p2_confidence_v2/confidence_v2_report.md",
    "p2_confidence_v2_calibration.tsv": OUT
    / "p2_confidence_v2/confidence_v2_calibration.tsv",
    "p3_backend_auto_v2_report.md": OUT
    / "p3_backend_auto_v2/backend_auto_v2_report.md",
    "p3_backend_auto_v2_policy.tsv": OUT
    / "p3_backend_auto_v2/backend_auto_v2_policy_comparison.tsv",
    "p3_cold_warm_timing.json": OUT
    / "p3_backend_auto_v2/cold_warm_timing.json",
    "p4_GPU_batching_report.md": OUT / "p4_gpu_batching/GPU_batching_report.md",
    "p4_gpu_batching_timing.tsv": OUT / "p4_gpu_batching/gpu_batching_timing.tsv",
    "p4_gpu_batching_quality.tsv": OUT / "p4_gpu_batching/gpu_batching_quality.tsv",
    "p5_production_mode_report.md": OUT
    / "p5_production_mode/production_mode_report.md",
    "p5_production_mode_summary.tsv": OUT
    / "p5_production_mode/production_mode_summary.tsv",
    "p6_parallel_MDLovoFit_report.md": OUT
    / "p6_mdlovofit_fairness/parallel_MDLovoFit_report.md",
    "p6_parallel_mdlovofit_summary.tsv": OUT
    / "p6_mdlovofit_fairness/parallel_mdlovofit_summary.tsv",
    "p6_framewise_centering_remediation.tsv": OUT
    / "p6_mdlovofit_fairness/pdb_range_failure_framewise_centered_reruns.tsv",
    "p7_timing_outlier_report.md": OUT
    / "p7_timing_atlas/timing_outlier_report.md",
    "p7_controlled_gpu_timing_summary.tsv": OUT
    / "p7_timing_atlas/controlled_gpu_timing_summary.tsv",
    "p7_atlas_readiness_report.md": OUT
    / "p7_timing_atlas/atlas_readiness_report.md",
    "p7_atlas_readiness_manifest.tsv": OUT
    / "p7_timing_atlas/atlas_readiness_manifest.tsv",
    "p7_atlas_source_members.tsv": OUT
    / "p7_timing_atlas/atlas_downloaded_source_members.tsv",
}


def verify_frozen() -> dict[str, object]:
    rows = read_tsv(FROZEN)
    failures = []
    checked_bytes = 0
    for index, row in enumerate(rows, start=1):
        path = ROOT / row["path"]
        reason = ""
        observed_size = path.stat().st_size if path.exists() else -1
        observed_hash = ""
        if not path.exists():
            reason = "missing"
        elif observed_size != int(row["size_bytes"]):
            reason = "size_mismatch"
        else:
            observed_hash = sha256(path)
            checked_bytes += observed_size
            if observed_hash != row["sha256"]:
                reason = "sha256_mismatch"
        if reason:
            failures.append(
                {
                    "path": row["path"],
                    "reason": reason,
                    "expected_size_bytes": row["size_bytes"],
                    "observed_size_bytes": observed_size,
                    "expected_sha256": row["sha256"],
                    "observed_sha256": observed_hash,
                }
            )
        if index % 2000 == 0 or index == len(rows):
            print(f"[freeze verification] {index}/{len(rows)}", flush=True)
    verification_rows = failures or [
        {
            "path": "ALL_FROZEN_FILES",
            "reason": "pass",
            "expected_size_bytes": sum(int(row["size_bytes"]) for row in rows),
            "observed_size_bytes": checked_bytes,
            "expected_sha256": "see_frozen_source_evidence_manifest.tsv",
            "observed_sha256": "all_individual_hashes_match",
        }
    ]
    write_tsv(OUT / "p0_freeze/frozen_integrity_recheck.tsv", verification_rows)
    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checked_files": len(rows),
        "checked_bytes": checked_bytes,
        "failure_count": len(failures),
        "all_frozen_hashes_match": not failures,
    }
    write_json(OUT / "p0_freeze/frozen_integrity_recheck.json", status)
    return status


def copy_compact_evidence() -> list[dict[str, object]]:
    evidence = HANDOFF / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name, source in REPORTS.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination = evidence / name
        shutil.copy2(source, destination)
        inventory.append(
            {
                "handoff_path": str(destination.relative_to(ROOT)),
                "source_path": str(source.relative_to(ROOT)),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    return inventory


def build_handoff(frozen_status: dict[str, object]) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    phase_rows = [
        {"phase": "P0", "status": "complete", "primary_result": "24,080 frozen source/evidence files", "caveat": "repository has no valid commit history; Git metadata was not modified"},
        {"phase": "P1", "status": "complete", "primary_result": "8 registered catastrophes; 54 separate large geometric tails", "caveat": "derived reporting only; frozen TSVs unchanged"},
        {"phase": "P2", "status": "complete", "primary_result": "134,950 confidence-v2 rows across 5,398 domains", "caveat": "no black-box classifier trained on eight tails"},
        {"phase": "P3", "status": "complete", "primary_result": "threshold_25000 selected on development; 4.181941 s validation compute", "caveat": "9/200 backend choices were >5% slower; public default unchanged"},
        {"phase": "P4", "status": "complete", "primary_result": "6.866x batch-25 gain with Jaccard/Spearman 1.0", "caveat": "same-shape engineering path; large shapes show saturation"},
        {"phase": "P5", "status": "complete", "primary_result": "10,000 trajectories; 14,937 trajectories/hour backend-auto-v2", "caveat": "serial mode order was not page-cache balanced"},
        {"phase": "P6", "status": "complete", "primary_result": "45/45 remediated; 11,838 trajectories/hour at 16 workers", "caveat": "framewise centering required; latency rises at 16 workers"},
        {"phase": "P7", "status": "complete", "primary_result": "0/109 persistent GPU outliers; ATLAS readiness 8/8", "caveat": "ATLAS R1 only; whole-archive hashes and R2/R3 unavailable"},
    ]
    write_tsv(HANDOFF / "phase_status.tsv", phase_rows)

    metrics = [
        {"topic": "B3 registered catastrophic quality", "value": "8/10000", "contract": "prospectively registered exact-relative criteria", "evidence": "evidence/p1_corrected_benchmark_interpretation.md"},
        {"topic": "B3 large geometric tail", "value": "54/10000", "contract": "median absolute unified-RMSD difference >1 A", "evidence": "evidence/p1_corrected_benchmark_summary.tsv"},
        {"topic": "confidence-v2 coverage", "value": "134950 trajectories; 5398 domains", "contract": "derived metadata only", "evidence": "evidence/p2_confidence_v2_report.md"},
        {"topic": "backend-auto-v2 validation", "value": "4.181941 s; 9/200 choices >5% slower", "contract": "loaded compute; locked 200-case validation", "evidence": "evidence/p3_backend_auto_v2_report.md"},
        {"topic": "GPU same-domain batching", "value": "6.866x batch25/batch1", "contract": "same shape/mapping, transfers and synchronization included", "evidence": "evidence/p4_GPU_batching_report.md"},
        {"topic": "P5 GPU throughput only", "value": "13999.7 trajectories/hour", "contract": "raw-source observed production run", "evidence": "evidence/p5_production_mode_summary.tsv"},
        {"topic": "P5 GPU throughput + W5", "value": "14907.9 trajectories/hour", "contract": "raw-source observed production run", "evidence": "evidence/p5_production_mode_summary.tsv"},
        {"topic": "P5 backend-auto-v2", "value": "14937.0 trajectories/hour", "contract": "raw-source observed production run", "evidence": "evidence/p5_production_mode_summary.tsv"},
        {"topic": "MDLovoFit centering remediation", "value": "45/45 success; max Kabsch RMSD delta 2.946e-10 A", "contract": "framewise rigid COG translation", "evidence": "evidence/p6_framewise_centering_remediation.tsv"},
        {"topic": "MDLovoFit parallel throughput", "value": "1048.8, 2099.5, 4109.8, 7461.1, 11838.1 trajectories/hour", "contract": "1/2/4/8/16 workers; locked 2000-panel", "evidence": "evidence/p6_parallel_mdlovofit_summary.tsv"},
        {"topic": "controlled GPU outliers", "value": "0/109 remain above 0.1 s", "contract": "one warmup + five synchronized persistent repeats", "evidence": "evidence/p7_timing_outlier_report.md"},
        {"topic": "ATLAS readiness", "value": "8/8 systems; 10001 frames each; 188-2128 CA", "contract": "R1 selected-member audit", "evidence": "evidence/p7_atlas_readiness_report.md"},
        {"topic": "test suite", "value": "279 passed; 13 warnings", "contract": "canonical GPU environment", "evidence": "reproducible_commands.md"},
        {"topic": "frozen evidence integrity", "value": f"{frozen_status['checked_files']} files; {frozen_status['failure_count']} failures", "contract": "individual SHA-256 recheck", "evidence": "../p0_freeze/frozen_integrity_recheck.json"},
    ]
    write_tsv(HANDOFF / "key_metrics.tsv", metrics)

    write_text(
        HANDOFF / "negative_results_and_limitations.md",
        """# Negative results and limitations

- The frozen helper incorrectly mixed 54 post-hoc geometric tails with the 8 prospectively registered catastrophic failures. Historical files remain unchanged; only derived reporting is corrected.
- Backend-auto v2 made nine validation choices more than 5% slower than the best measured backend. It remains experimental and does not replace the public default.
- GPU batch scaling is shape dependent. The 482-residue case showed weaker gains at large batches, and batch 50 repeats the same 25 biological trajectories.
- P5 raw-source runs were not order balanced. Later modes benefited from a warmer page cache, so loaded compute is the method-comparison boundary.
- A common single COG translation did not repair the 45 MDLovoFit PDB-range failures. Per-frame rigid COG translation was required; both attempts are preserved.
- MDLovoFit 16-worker node throughput improved, but median single-task latency rose from 2.63 to 3.58 seconds and parallel efficiency was sublinear.
- The original 109 D1 GPU timing outliers did not persist after controlled warm timing, indicating cold-start or timing noise rather than stable slow cases.
- ATLAS archives contain 10,001, not exactly 10,000, R1 frames. Only R1 and selected members were audited. Full remote ZIP SHA-256 values and R2/R3 audits are unavailable.
- ATLAS fitted-trajectory geometry showed no gross jumps, but this does not independently prove PBC/nojump processing or biological continuity.
- No universal-fastest claim is supported. Loaded analysis, matched materialised output, raw-source throughput and node-parallel throughput are separate contracts.
""",
    )

    write_text(
        HANDOFF / "reproducible_commands.md",
        """# Reproducible commands

Run from the repository root with the absolute interpreters shown below.

```bash
env MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. /storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python -m pytest -q
env PYTHONPATH=src:. /storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/post_b3_optimisation_v1/run_p7_timing_outliers.py register
env MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. /storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/post_b3_optimisation_v1/run_p7_timing_outliers.py run
env PYTHONPATH=src:. /storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/post_b3_optimisation_v1/run_p7_timing_outliers.py finalize
env PYTHONPATH=src:. /storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/post_b3_optimisation_v1/run_p7_atlas_readiness.py acquire --workers 4
env MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. /home/jjxu/miniconda3/envs/mdlovofit/bin/python scripts/post_b3_optimisation_v1/run_p7_atlas_readiness.py verify
env PYTHONPATH=src:. /storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python scripts/post_b3_optimisation_v1/run_p7_atlas_readiness.py finalize
```

P3-P6 exact commands and parameters are encoded in their scripts, locked manifests and copied reports. P0-P7 were run serially; no timed benchmark stages were overlapped.
""",
    )

    source_rows = []
    for path in sorted(
        list((ROOT / "scripts/post_b3_optimisation_v1").glob("*.py"))
        + [ROOT / "src/mdcorealign/experimental_gpu_batch.py"]
        + list((ROOT / "tests/post_b3_optimisation_v1").glob("*.py"))
    ):
        source_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_tsv(HANDOFF / "source_files.tsv", source_rows)

    copied = copy_compact_evidence()
    own_files = [
        HANDOFF / "phase_status.tsv",
        HANDOFF / "key_metrics.tsv",
        HANDOFF / "negative_results_and_limitations.md",
        HANDOFF / "reproducible_commands.md",
        HANDOFF / "source_files.tsv",
    ]
    inventory = copied + [
        {
            "handoff_path": str(path.relative_to(ROOT)),
            "source_path": "generated_handoff",
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in own_files
    ]
    write_tsv(HANDOFF / "handoff_file_inventory.tsv", inventory)

    readme = f"""# Post-B3 optimisation compact handoff

Generated: `{datetime.now(timezone.utc).isoformat()}`

## Completion

P0-P7 are complete. The canonical frozen source/evidence baseline contains
`{frozen_status['checked_files']}` files and was reverified with
`{frozen_status['failure_count']}` SHA-256 failures. The full test suite passed:
`279 passed, 13 warnings`.

Start with `phase_status.tsv`, `key_metrics.tsv` and
`negative_results_and_limitations.md`. Compact source reports and tables are in
`evidence/`; large raw tables, checkpoints and ATLAS coordinate files are not
duplicated into this handoff.

## Contract safety

Sequential/exact behavior and public defaults were not changed. P3 backend-auto,
P4 batching and the P5 production paths remain experimental. Do not claim
universal fastest. Keep loaded analysis, matched materialised output,
raw-source database throughput and node-parallel throughput separate.

## Repository state

The parent Git repository reports `No commits yet on master`. No Git metadata was
initialized or modified during P0-P7, so a valid commit identifier is unavailable.
"""
    write_text(HANDOFF / "00_README_FIRST.md", readme)

    final_files = sorted(path for path in HANDOFF.rglob("*") if path.is_file())
    write_tsv(
        HANDOFF / "final_sha256_manifest.tsv",
        [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in final_files
        ],
    )
    write_json(
        HANDOFF / "handoff_status.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phases_complete": [f"P{index}" for index in range(8)],
            "test_command": "env MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:. /storage/junjie/projects/mdlovofit/.venv-gpu-cu12/bin/python -m pytest -q",
            "tests_passed": 279,
            "test_warnings": 13,
            "frozen_integrity_pass": frozen_status["all_frozen_hashes_match"],
            "public_defaults_changed": False,
            "historical_evidence_changed": False,
            "git_metadata_changed": False,
        },
    )


def main() -> None:
    frozen_status = verify_frozen()
    if not frozen_status["all_frozen_hashes_match"]:
        raise RuntimeError("frozen evidence integrity check failed")
    build_handoff(frozen_status)


if __name__ == "__main__":
    main()
