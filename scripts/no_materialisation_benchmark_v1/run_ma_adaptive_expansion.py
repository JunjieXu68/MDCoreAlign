#!/usr/bin/env python3
"""Run the registered 512-to-1,024-domain MA4 adaptive expansion."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

from scripts.no_materialisation_benchmark_v1.common import OUT, ROOT, read_tsv, write_json, write_text, write_tsv
from scripts.no_materialisation_benchmark_v1.lock_ma3_ma7_panels import stratified_select
from scripts.no_materialisation_benchmark_v1.run_ma3_ma7_agreement import MANIFESTS, process_task


MA = OUT / "method_agreement_v2"
REPORTING = MA / "08_reporting"
PROGRESS = REPORTING / "ma_adaptive_expansion_progress.log"


def lock_expansion() -> list[dict[str, str]]:
    all_rows = read_tsv(OUT / "03_mdcath_stock_no_output/mdcath_no_output_all_domain.tsv")
    domains = []
    for row in all_rows:
        if row["method"] != "stock_MDLovoFit_no_output" or row["repeat_index"] != "0":
            continue
        domains.append(
            {
                "domain_id": row["system"],
                "trajectory_id": row["prefix_label"],
                "cath_class": row["cath_class"],
                "n_frames": row["n_frames"],
                "n_residues": row["n_residues"],
                "residue_count_bin": row["residue_count_bin"],
                "frame_count_bin": row["frame_count_bin"],
            }
        )
    tails = {row["domain_id"] for row in read_tsv(MANIFESTS / "mdcath_registered_tails.tsv")}
    selected = stratified_select(domains, 1024, tails)
    initial = {row["domain_id"] for row in read_tsv(MANIFESTS / "mdcath_primary_512.tsv")}
    if {row["domain_id"] for row in selected[:512]} != initial:
        raise RuntimeError("the deterministic 1,024 panel does not preserve the locked initial 512-domain set")
    rows = [
        {
            "selection_index": index,
            **row,
            "initial_512": row["domain_id"] in initial,
            "adaptive_addition": row["domain_id"] not in initial,
            "selection_rule": "registered_512_to_1024_adaptive_expansion_after_fewer_than_20_Jaccard_lt_0.60_cases",
        }
        for index, row in enumerate(selected)
    ]
    write_tsv(MANIFESTS / "mdcath_adaptive_1024.tsv", rows)
    additions = [row for row in rows if row["adaptive_addition"]]
    if len(additions) != 512:
        raise RuntimeError(f"expected 512 adaptive additions, found {len(additions)}")
    write_tsv(MANIFESTS / "mdcath_adaptive_additions_512.tsv", additions)
    write_text(
        MANIFESTS / "mdcath_adaptive_expansion_lock_report.md",
        "# Registered mdCATH Adaptive Expansion\n\n"
        "The initial 512-domain panel produced a narrow bootstrap confidence interval and consistent major strata, but only eight exact-method cases with core Jaccard below 0.60. This activated the prospectively registered `fewer than 20 divergent cases` trigger. The primary 60% panel was therefore expanded deterministically to 1,024 domains. No new detailed, fraction-sensitivity, timing, W5 or THESEUS analysis was added. A full 5,397-domain agreement run still requires explicit human approval.\n",
    )
    return additions


def main() -> None:
    decision = json.loads((REPORTING / "adaptive_expansion_decision.json").read_text())
    if not decision.get("adaptive_expansion_triggered"):
        raise RuntimeError("the registered 512-domain adaptive-expansion trigger is not active")
    additions = lock_expansion()
    tasks = [
        {
            "panel": "mdCATH_adaptive_expansion_512",
            "kind": "mdcath",
            "case_id": row["domain_id"],
            "domain_id": row["domain_id"],
            "n_frames": int(row["n_frames"]),
            "n_residues": int(row["n_residues"]),
            "detailed": False,
        }
        for row in additions
    ]
    results = []
    started = time.time()
    with PROGRESS.open("w", buffering=1) as progress:
        progress.write(f"START\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\ttasks={len(tasks)}\n")
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    progress.write(f"FAILED\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{task['case_id']}\t{type(exc).__name__}:{exc}\n")
                    raise
                results.append(result)
                elapsed = time.time() - started
                progress.write(
                    f"COMPLETE\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{index}/{len(tasks)}\t{task['case_id']}\t"
                    f"case_seconds={result['elapsed_untimed_seconds']:.3f}\trate_per_hour={index/elapsed*3600.0:.2f}\n"
                )
    results.sort(key=lambda result: str(result["task"]["case_id"]))
    core = [row for result in results for row in result["core"]]
    continuous = [row for result in results for row in result["continuous"]]
    heldout = [row for result in results for row in result["heldout"]]
    write_tsv(REPORTING / "adaptive_expansion_core_agreement.tsv", core)
    write_tsv(REPORTING / "adaptive_expansion_continuous_agreement.tsv", continuous)
    write_tsv(REPORTING / "adaptive_expansion_heldout_geometry.tsv", heldout)
    write_json(
        REPORTING / "ma_adaptive_expansion_raw_status.json",
        {
            "status": "pass",
            "cases": len(results),
            "core_rows": len(core),
            "continuous_rows": len(continuous),
            "heldout_rows": len(heldout),
            "hours": (time.time() - started) / 3600.0,
        },
    )
    print(json.dumps({"status": "pass", "cases": len(results), "hours": (time.time() - started) / 3600.0}, indent=2))


if __name__ == "__main__":
    main()
