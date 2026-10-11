#!/usr/bin/env python3
"""Run AT4 preflight on the smallest and largest locked settings."""

from __future__ import annotations

import json
import shlex

from mdsuperpose.mdlovofit import parse_mdlovofit_stdout
from scripts.atlas_strict_single_thread_cpu_v1.common import (
    METHOD_MDCA,
    METHOD_MDLOVOFIT,
    OUT,
    absolute_input,
    command_for,
    read_tsv,
    sha256,
    warm_page_cache,
    write_text,
    write_tsv,
)
from scripts.atlas_strict_single_thread_cpu_v1.prepare_study import LOGICAL_CPU
from scripts.atlas_strict_single_thread_cpu_v1.run_timing import (
    run_command,
    timeout_seconds,
)


def main() -> None:
    decision = (OUT / "03_thread_affinity_audit/thread_contract_decision.md").read_text()
    if "SINGLE_THREAD_CONTRACT_PASS" not in decision:
        raise SystemExit("AT4 prohibited: strict thread contract has not passed")
    panel = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    selected = [
        min(panel, key=lambda row: int(row["frames_times_ca"])),
        max(panel, key=lambda row: int(row["frames_times_ca"])),
    ]
    destination = OUT / "04_preflight"
    logs = destination / "preflight_logs"
    rows = []
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for setting in selected:
        source = absolute_input(setting)
        observed_hash = sha256(source)
        if observed_hash != setting["input_sha256"]:
            raise ValueError(f"preflight input hash mismatch: {source}")
        for method in (METHOD_MDCA, METHOD_MDLOVOFIT):
            warm_page_cache(source)
            command = command_for(method, setting, LOGICAL_CPU)
            shell_lines.append(shlex.join(command))
            stem = f"{setting['system_id']}__{setting['prefix_id']}__{method}"
            stdout = logs / f"{stem}.stdout"
            stderr = logs / f"{stem}.stderr"
            result = run_command(
                command,
                stdout_path=stdout,
                stderr_path=stderr,
                timeout_seconds=timeout_seconds(),
            )
            output_valid = False
            if method == METHOD_MDCA and result["exit_code"] == 0:
                payload = json.loads(stdout.read_text().strip())
                output_valid = (
                    payload["status"] == "success"
                    and not payload["aligned_coordinates_materialised"]
                    and not payload["rmsf_computed"]
                )
            elif method == METHOD_MDLOVOFIT and result["exit_code"] == 0:
                parsed = parse_mdlovofit_stdout(stdout.read_text())
                output_valid = parsed.average_rmsd_all_atoms is not None
            rows.append(
                {
                    "setting_id": f"{setting['system_id']}__{setting['prefix_id']}",
                    "size_class": "small" if setting is selected[0] else "largest",
                    "method": method,
                    "input_hash_match": True,
                    "logical_cpu": LOGICAL_CPU,
                    "command": shlex.join(command),
                    "exit_code": result["exit_code"],
                    "timeout": result["timeout"],
                    "analysis_output_valid": output_valid,
                    "aligned_coordinate_output_created": False,
                    "operational_wall_seconds_not_for_analysis": result["wall_seconds"],
                    "status": "pass"
                    if result["exit_code"] == 0 and not result["timeout"] and output_valid
                    else "fail",
                }
            )
            write_tsv(destination / "preflight_results.tsv", rows)
    write_text(destination / "preflight_commands.sh", "\n".join(shell_lines))
    passed = len(rows) == 4 and all(row["status"] == "pass" for row in rows)
    write_text(
        destination / "preflight_decision.md",
        "\n".join(
            [
                "# Preflight decision",
                "",
                f"`{'PREFLIGHT_PASS' if passed else 'PREFLIGHT_FAILED'}`",
                "",
                "Preflight operational wall times are excluded from all speedup analyses.",
                "Both methods used the same hash-verified PDB, CPU affinity, thread environment,",
                "and no-coordinate-output command contract. Resume/checkpoint behaviour is",
                "covered by focused runner tests before AT5.",
            ]
        ),
    )
    if not passed:
        raise SystemExit("PREFLIGHT_FAILED")
    print("PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
