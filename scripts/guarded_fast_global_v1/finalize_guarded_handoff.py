#!/usr/bin/env python3
"""Build the compact G0-G4 guarded fast-global handoff bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from scripts.mainline_reintegration_v1.run_r0_inventory import (
    frozen_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/guarded_fast_global_v1"
HANDOFF = RESULT / "handoff_to_chatgpt"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status() -> dict[str, object]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "head_available": completed.returncode == 0,
        "head": completed.stdout.strip() if completed.returncode == 0 else None,
        "error": completed.stderr.strip() if completed.returncode else "",
        "git_metadata_modified_by_this_run": False,
    }


def test_status() -> dict[str, object]:
    junit = HANDOFF / "full_test_junit.xml"
    if not junit.is_file():
        return {
            "status": "unavailable",
            "reason": "full_test_junit.xml not found",
        }
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    output_text = (HANDOFF / "full_test_output.txt").read_text()
    match = re.search(
        r"(\d+) passed, (\d+) subtests passed",
        output_text,
    )
    pytest_tests = int(match.group(1)) if match else tests
    pytest_subtests = int(match.group(2)) if match else 0
    return {
        "status": "pass" if failures == 0 and errors == 0 else "fail",
        "pytest_tests_passed": pytest_tests,
        "pytest_subtests_passed": pytest_subtests,
        "junit_testcases": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "command": (
            "PYTHONPATH=src:. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "
            "MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 "
            "/home/jjxu/miniconda3/bin/python -m pytest -q "
            "--junitxml=results/guarded_fast_global_v1/"
            "handoff_to_chatgpt/full_test_junit.xml"
        ),
    }


def paired_contract_summary() -> list[dict[str, object]]:
    rows = read_tsv(
        RESULT / "g4_matched_contracts/matched_output_runtime.tsv"
    )
    output: list[dict[str, object]] = []
    for contract in (
        "A_metrics_transforms",
        "B_materialized_aligned_PDB",
    ):
        medians: dict[tuple[str, str], float] = {}
        domains = sorted({row["domain_id"] for row in rows})
        methods = sorted({row["method"] for row in rows})
        for domain in domains:
            for method in methods:
                values = [
                    float(row["runtime_seconds"])
                    for row in rows
                    if row["contract"] == contract
                    and row["domain_id"] == domain
                    and row["method"] == method
                    and row["status"] == "success"
                ]
                if values:
                    medians[(domain, method)] = statistics.median(values)
        for numerator, denominator in (
            ("sequential_lean_exact", "guarded_fast_global"),
            ("external_mdlovofit", "guarded_fast_global"),
            ("external_mdlovofit", "sequential_lean_exact"),
        ):
            ratios = [
                medians[(domain, numerator)]
                / medians[(domain, denominator)]
                for domain in domains
                if (domain, numerator) in medians
                and (domain, denominator) in medians
            ]
            output.append(
                {
                    "contract": contract,
                    "numerator": numerator,
                    "denominator": denominator,
                    "n_paired_domains": len(ratios),
                    "median_paired_runtime_ratio": statistics.median(ratios),
                    "fraction_numerator_slower": float(
                        np.mean(np.asarray(ratios) > 1.0)
                    ),
                    "minimum_ratio": min(ratios),
                    "maximum_ratio": max(ratios),
                }
            )
    return output


def main() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    g1_rows = read_tsv(
        RESULT / "g1_adjudication/classification_summary.tsv"
    )
    g1 = {
        row["scientific_classification"]: int(row["count"])
        for row in g1_rows
    }
    g2 = json.loads(
        (
            RESULT
            / "g2_risk_certificate/final_certificate_decision.json"
        ).read_text()
    )
    w5 = json.loads(
        (
            RESULT
            / "g2_risk_certificate/w5_validation_decision.json"
        ).read_text()
    )
    g3 = json.loads(
        (RESULT / "g3_guarded/g3_summary.json").read_text()
    )
    stress = {
        row["method"]: row
        for row in read_tsv(
            RESULT / "g3_guarded/stress_10k_runtime_summary.tsv"
        )
    }
    g4 = json.loads(
        (RESULT / "g4_matched_contracts/g4_status.json").read_text()
    )
    batch = {
        (row["batch_size"], row["method"]): row
        for row in read_tsv(
            RESULT
            / "g4_matched_contracts/persistent_batch_detailed_summary.tsv"
        )
    }
    key_metrics = {
        "g0": {
            "corrected_paired_domains": 5397,
            "development_domains": 1623,
            "validation_domains": 3774,
            "predefined_tail_failures": 13,
        },
        "g1": {
            "strict_nonconvergence_count": 4544,
            **g1,
            "strict_criterion_interpretation": (
                "too conservative as a standalone scientific-failure label"
            ),
        },
        "g2": {
            **g2,
            "conditional_W5_validation_tail_recall": w5["tail_recall"],
            "conditional_W5_normal_flag_reduction_fraction": w5[
                "normal_flag_reduction_fraction"
            ],
            "conditional_W5_amortized_seconds": w5[
                "amortized_W5_seconds_per_all_trajectory"
            ],
        },
        "g3": g3,
        "g3_10k_stress": {
            "exact_median_seconds": float(
                stress["sequential_lean_exact"]["median_runtime_seconds"]
            ),
            "guarded_median_seconds": float(
                stress["guarded_fast_global"]["median_runtime_seconds"]
            ),
            "guarded_accepted_level": stress[
                "guarded_fast_global"
            ]["accepted_levels"],
            "guarded_target_0_75_seconds_pass": (
                float(
                    stress["guarded_fast_global"][
                        "median_runtime_seconds"
                    ]
                )
                < 0.75
            ),
        },
        "g4": {
            **g4,
            "paired_contracts": paired_contract_summary(),
            "batch_1000": {
                method: {
                    "median_total_seconds": float(
                        batch[("1000", method)][
                            "median_total_runtime_seconds"
                        ]
                    ),
                    "trajectories_per_second": float(
                        batch[("1000", method)][
                            "median_trajectories_per_second"
                        ]
                    ),
                    "fallback_rate": float(
                        batch[("1000", method)][
                            "median_fallback_rate"
                        ]
                    ),
                    "quality": batch[("1000", method)][
                        "final_quality_status"
                    ],
                }
                for method in (
                    "sequential_lean_exact",
                    "guarded_fast_global",
                    "external_mdlovofit",
                )
            },
        },
    }
    write_json(HANDOFF / "02_key_metrics.json", key_metrics)
    write_tsv(
        HANDOFF / "01_phase_status.tsv",
        [
            {
                "phase": "G0",
                "status": "pass",
                "decision": "5397 corrected paired domains frozen",
            },
            {
                "phase": "G1",
                "status": "complete",
                "decision": (
                    "strict non-convergence is conservative; 13 tails retained"
                ),
            },
            {
                "phase": "G2",
                "status": "primary_gate_failed",
                "decision": (
                    "cheap certificate missed 1/8 validation tails; "
                    "preexisting temporal safety route required"
                ),
            },
            {
                "phase": "G3",
                "status": "partial_engineering_success",
                "decision": (
                    "13/13 predefined tails corrected, but strict held-out "
                    "parity and fallback-rate targets failed"
                ),
            },
            {
                "phase": "G4",
                "status": "complete",
                "decision": (
                    "matched contracts and batches complete; no universal "
                    "speed claim"
                ),
            },
        ],
    )
    integrity = {
        "frozen_evidence": frozen_integrity(),
        "git": git_status(),
        "g0_split_lock": json.loads(
            (RESULT / "g0_freeze/split_lock.json").read_text()
        ),
        "g2_thresholds_changed_after_validation": False,
        "g3_thresholds_changed_after_validation": False,
        "public_default_changed": False,
        "historical_frozen_evidence_modified": False,
    }
    write_json(HANDOFF / "integrity_after_g4.json", integrity)
    write_json(HANDOFF / "test_status.json", test_status())
    write_tsv(
        HANDOFF / "03_code_changes.tsv",
        [
            {
                "path": "src/mdsuperpose/window_refine.py",
                "change": (
                    "optional trace and caller-supplied representative indices"
                ),
                "default_numerics_changed": False,
            },
            {
                "path": "src/mdcorealign/experimental.py",
                "change": (
                    "optional online risk signals and representative-index reuse"
                ),
                "default_numerics_changed": False,
            },
            {
                "path": "src/mdcorealign/guarded.py",
                "change": "experimental guarded-v1 cascade and batch worker",
                "default_numerics_changed": False,
            },
            {
                "path": "src/mdcorealign/pdb_writer.py",
                "change": "byte-identical streaming CA PDB writer",
                "default_numerics_changed": False,
            },
            {
                "path": (
                    "scripts/guarded_fast_global_v1/"
                    "run_guarded_fast_global.py"
                ),
                "change": "G0-G3 staged audit and validation runner",
                "default_numerics_changed": False,
            },
            {
                "path": (
                    "scripts/guarded_fast_global_v1/"
                    "run_g4_matched_contracts.py"
                ),
                "change": "G4 matched contracts and batch benchmark",
                "default_numerics_changed": False,
            },
        ],
    )
    (HANDOFF / "04_reproducible_commands.md").write_text(
        """# Reproducible commands

Run from the repository root with:

```bash
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PY=/home/jjxu/miniconda3/bin/python
```

G0-G3 stages:

```bash
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g0
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g1 --workers 12
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g1-finalize
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g2-fit
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g2-validate
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g2-w5 --workers 12
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g2-freeze-w5
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g2-w5-validate --workers 12
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g2-finalize
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g3-freeze-route
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g3-level2 --split development --workers 12
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g3-freeze-level2-policy
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g3-level2 --split validation --workers 12
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g3-finalize
```

For clean G3 stress/G4 timing, set all four thread variables to 2:

```bash
$PY scripts/guarded_fast_global_v1/run_guarded_fast_global.py g3-stress
$PY scripts/guarded_fast_global_v1/run_g4_matched_contracts.py matched-contracts
$PY scripts/guarded_fast_global_v1/run_g4_matched_contracts.py writer
$PY scripts/guarded_fast_global_v1/run_g4_matched_contracts.py persistent-batch
$PY scripts/guarded_fast_global_v1/run_g4_matched_contracts.py cold-start
$PY scripts/guarded_fast_global_v1/run_g4_matched_contracts.py finalize
```
"""
    )
    (HANDOFF / "05_negative_results_and_decision.md").write_text(
        f"""# Negative results and decision

1. The development-only cheap certificate missed one of eight held-out
   predefined tails. Conditional W5 retained the seven tails already found,
   but did not recover the missed tail.
2. Level 2 failed the strict quality contract for 133/631 development and
   357/1557 validation routes. It corrected 0/5 development and 0/8 validation
   predefined tails, so guarded-v1 does not return Level-2 output.
3. The frozen temporal safety route sent all 13 predefined tails to exact
   fallback. This removed those catastrophic failures, but direct Level-1
   acceptance was only {g3['direct_level1_acceptance_fraction']:.3%}, exact
   fallback was {g3['exact_fallback_fraction']:.3%}, and 284 accepted Level-1
   outputs still failed the strict descriptive parity contract.
4. On the 10k/153-residue transition-aligned synthetic stress case, guarded-v1
   routed to Level 3 and took
   {float(stress['guarded_fast_global']['median_runtime_seconds']):.3f} s versus
   {float(stress['sequential_lean_exact']['median_runtime_seconds']):.3f} s for
   exact alone. The 0.75 s target failed.
5. On 1,000 short 12-frame toy trajectories, persistent guarded took
   {float(batch[('1000', 'guarded_fast_global')]['median_total_runtime_seconds']):.3f}
   s versus
   {float(batch[('1000', 'sequential_lean_exact')]['median_total_runtime_seconds']):.3f}
   s for exact. Certificate overhead is not beneficial at this small frame
   count.

## Decision

`sequential_lean` remains the exact public default. Guarded-v1 is retained as
an experimental auditable cascade, not promoted as a safe unconditional
replacement. W5 remains conditional and experimental. The evidence does not
support "universally fastest", "complete domination", or an unconditional
10.9x guarded speed claim.
"""
    )
    inventory_rows = []
    for path in sorted(RESULT.rglob("*")):
        if not path.is_file() or HANDOFF in path.parents:
            continue
        inventory_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_tsv(HANDOFF / "06_output_inventory.tsv", inventory_rows)
    readme = f"""# Guarded fast-global G0-G4 handoff

## Bottom line

G0-G4 are complete. The corrected 5,397-domain panel was frozen and all 13
predefined tail failures were routed to exact fallback. However, guarded-v1
did not pass the strict held-out parity contract, the 70% direct-acceptance
target, the below-10% fallback target, or the 10k below-0.75 s target.
Therefore it remains experimental and `sequential_lean` remains the public
exact default.

## Key measurements

- G1: 4,544 strict non-convergence labels, but 3,822 were scientifically
  acceptable under the locked comparison; 779 were borderline and 13 were
  predefined tails.
- G2 cheap validation recall: {g2['cheap_tail_recall']:.3%} (7/8).
- G3 direct Level-1 acceptance:
  {g3['direct_level1_acceptance_fraction']:.3%}.
- G3 exact fallback: {g3['exact_fallback_fraction']:.3%}.
- G3 remaining predefined catastrophic tails: 0/13.
- G3 strict parity failures after routing: 284/5,397.
- G3 10k guarded/exact:
  {float(stress['guarded_fast_global']['median_runtime_seconds']):.3f} /
  {float(stress['sequential_lean_exact']['median_runtime_seconds']):.3f} s.
- Streaming PDB writer speedup: {g4['writer_speedup']:.3f}x, byte-identical.
- Frozen integrity: {integrity['frozen_evidence']['status']}
  ({integrity['frozen_evidence']['project_completion_checked']} project
  hashes and {integrity['frozen_evidence']['golden_checked']} golden hashes).
- Tests: see `test_status.json`.

## Start here

Read `02_key_metrics.json`, `05_negative_results_and_decision.md`, and the G3
and G4 reports linked by `06_output_inventory.tsv`.
"""
    (HANDOFF / "00_read_me_first.md").write_text(readme)
    manifest_files = [
        path
        for path in sorted(HANDOFF.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    write_json(
        HANDOFF / "manifest.json",
        {
            "phase": "G0-G4",
            "status": "complete_with_failed_engineering_gates",
            "files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in manifest_files
            ],
            "public_default": "sequential_lean",
            "guarded_status": "experimental_not_promoted",
            "universal_fastest_claimed": False,
        },
    )


if __name__ == "__main__":
    main()
