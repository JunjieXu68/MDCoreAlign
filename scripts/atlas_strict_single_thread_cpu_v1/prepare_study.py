#!/usr/bin/env python3
"""Freeze AT0/AT1 evidence and prospectively register AT2."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.atlas_strict_single_thread_cpu_v1.common import (
    ATLAS_MANIFEST,
    HISTORICAL_TIMING,
    METHOD_MDCA,
    METHOD_MDLOVOFIT,
    OUT,
    PREFIXES,
    PYTHON,
    ROOT,
    SOURCE_MANIFEST,
    STOCK_MDLOVOFIT,
    THREAD_ENV,
    THREAD_ENV_HASH,
    absolute_input,
    command_for,
    read_tsv,
    setting_id,
    sha256,
    stable_hash_int,
    thread_environment,
    write_text,
    write_tsv,
)

UTC = timezone.utc
LOGICAL_CPU = 17
IDLE_LOAD1_MAX = 18.0
IDLE_SELECTED_CPU_BUSY_MAX_PERCENT = 10.0
CONFLICT_PATTERNS = (
    "build_exact_temporal_panel",
    "run_benchmark.py temporal-pilot",
    "mdlovofit -f",
    "atlas_strict_single_thread_cpu_v1.run_timing",
    "pytest",
    "python -m build",
)


def run_command(
    command: list[str],
    *,
    timeout: float = 30.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a short identity command without altering host state."""

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return {
            "return_code": result.returncode,
            "stdout": result.stdout.rstrip(),
            "stderr": result.stderr.rstrip(),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "return_code": "timeout",
            "stdout": (error.stdout or "").rstrip(),
            "stderr": (error.stderr or "").rstrip(),
        }


def process_snapshot() -> tuple[str, bool]:
    """Capture active processes and detect registered timing conflicts."""

    result = run_command(
        [
            "ps",
            "-eo",
            "pid,ppid,stat,etimes,pcpu,pmem,rss,comm,args",
            "--sort=-pcpu",
        ]
    )
    text = str(result["stdout"])
    return text, snapshot_has_active_conflict(text)


def snapshot_has_active_conflict(
    snapshot: str,
    *,
    ignored_pids: set[int] | None = None,
) -> bool:
    """Treat matching runnable/sleeping work as conflict but ignore stopped tasks."""

    ignored = ignored_pids or set()
    for line in snapshot.splitlines()[1:]:
        fields = line.split(maxsplit=8)
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        if int(fields[0]) in ignored or fields[2].startswith("T"):
            continue
        if any(pattern in line for pattern in CONFLICT_PATTERNS):
            return True
    return False


def package_identity() -> dict[str, Any]:
    """Query the exact Python environment selected for the benchmark."""

    code = (
        "import json,mdcorealign,numpy,scipy;"
        "print(json.dumps({'mdcorealign_path':mdcorealign.__file__,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__}))"
    )
    result = run_command([str(PYTHON), "-c", code], env=thread_environment())
    if result["return_code"] != 0:
        return {"error": result["stderr"], "python": str(PYTHON)}
    return json.loads(str(result["stdout"]))


def mdlovofit_identity() -> dict[str, str]:
    """Capture stock MDLovoFit identity without running a benchmark."""

    result = run_command([str(STOCK_MDLOVOFIT), "--help"])
    output = "\n".join(part for part in (str(result["stdout"]), str(result["stderr"])) if part)
    version = next(
        (line.strip("# ") for line in output.splitlines() if "Version" in line),
        "unavailable",
    )
    return {"version": version, "help_output": output}


def pyproject_version() -> str:
    """Read the frozen package version without importing build tooling."""

    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if line.strip().startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"')
    return "unavailable"


def freeze_at0() -> bool:
    """Write AT0 identity and active-process evidence."""

    destination = OUT / "00_freeze"
    snapshot, conflict = process_snapshot()
    load = os.getloadavg()
    write_text(
        destination / "active_process_check.md",
        "\n".join(
            [
                "# Active-process check",
                "",
                f"- Snapshot UTC: `{datetime.now(UTC).isoformat()}`",
                f"- Load average: `{load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}`",
                f"- Concurrent CPU/storage-heavy task detected: `{'yes' if conflict else 'no'}`",
                f"- Formal timing permitted at this snapshot: `{'no' if conflict else 'subject to AT3'}`",
                "",
                "The runner never terminates unrelated work. Raw process snapshot:",
                "",
                "```text",
                snapshot,
                "```",
            ]
        ),
    )
    identity = package_identity()
    mdlovo = mdlovofit_identity()
    software = [
        {"item": "MDCoreAlign_version", "value": pyproject_version(), "status": "recorded"},
        {
            "item": "MDCoreAlign_import_path",
            "value": identity.get("mdcorealign_path", ""),
            "status": "recorded",
        },
        {"item": "Python_executable", "value": str(PYTHON), "status": "recorded"},
        {"item": "Python_version", "value": platform.python_version(), "status": "recorded"},
        {"item": "NumPy_version", "value": identity.get("numpy", ""), "status": "recorded"},
        {"item": "SciPy_version", "value": identity.get("scipy", ""), "status": "recorded"},
        {"item": "MDLovoFit_executable", "value": str(STOCK_MDLOVOFIT), "status": "present"},
        {"item": "MDLovoFit_version", "value": mdlovo["version"], "status": "recorded"},
        {
            "item": "Git_HEAD",
            "value": "unavailable_invalid_or_unborn_HEAD",
            "status": "not_invented",
        },
        {
            "item": "Git_status",
            "value": "deferred_while_unrelated_git_processes_active",
            "status": "not_collected",
        },
    ]
    write_tsv(destination / "software_manifest.tsv", software)
    hash_paths = [
        STOCK_MDLOVOFIT,
        PYTHON,
        ROOT / "pyproject.toml",
        ROOT / "src/mdcorealign/benchmark_minimal.py",
        ROOT / "src/mdsuperpose/window_refine.py",
        ROOT / "src/mdsuperpose/all_frame_engine.py",
        Path(__file__),
    ]
    hashes = []
    for path in hash_paths:
        hashes.append(
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "sha256": sha256(path) if path.is_file() else "",
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "status": "present" if path.is_file() else "missing",
            }
        )
    write_tsv(destination / "executable_hashes.tsv", hashes)
    hardware = []
    for item, command in (
        ("lscpu", ["lscpu"]),
        ("lscpu_extended", ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE"]),
        ("memory", ["free", "-b"]),
        ("filesystem", ["df", "-T", str(SOURCE_MANIFEST)]),
        ("uname", ["uname", "-a"]),
    ):
        result = run_command(command)
        hardware.append(
            {
                "item": item,
                "value": str(result["stdout"]).replace("\n", " | "),
                "status": "recorded" if result["return_code"] == 0 else "failed",
            }
        )
    write_tsv(destination / "hardware_manifest.tsv", hardware)
    environment = [
        {"variable": key, "value": value, "role": "registered_thread_limit"}
        for key, value in THREAD_ENV.items()
    ]
    environment.extend(
        [
            {
                "variable": "THREAD_ENV_SHA256",
                "value": THREAD_ENV_HASH,
                "role": "contract_identity",
            },
            {"variable": "platform", "value": platform.platform(), "role": "host_identity"},
            {"variable": "python_prefix", "value": sys.prefix, "role": "environment_identity"},
        ]
    )
    write_tsv(destination / "environment_manifest.tsv", environment)
    write_text(destination / "mdlovofit_help.txt", mdlovo["help_output"])
    numpy_config = run_command(
        [str(PYTHON), "-c", "import numpy as np; np.show_config()"],
        env=thread_environment(),
    )
    write_text(
        destination / "numpy_blas_config.txt",
        str(numpy_config["stdout"]) + "\n" + str(numpy_config["stderr"]),
    )
    packages = run_command(
        [str(PYTHON), "-m", "pip", "freeze"],
        timeout=60,
        env=thread_environment(),
    )
    write_text(destination / "pip_freeze.txt", str(packages["stdout"]))
    return conflict


def lock_panel(*, verify_hashes: bool) -> list[dict[str, Any]]:
    """Recover the exact frozen 48-setting ATLAS panel."""

    source_rows = [row for row in read_tsv(SOURCE_MANIFEST) if row["status"] == "success"]
    atlas = {row["system"]: row for row in read_tsv(ATLAS_MANIFEST)}
    panel: list[dict[str, Any]] = []
    for source in source_rows:
        path = ROOT / source["centered_input_path"]
        observed_hash = sha256(path) if verify_hashes else source["centered_input_sha256"]
        observed_size = path.stat().st_size
        panel.append(
            {
                "system_id": source["system"],
                "family_id": f"ATLAS_{source['system']}",
                "replicate": atlas[source["system"]]["replicate"],
                "prefix_id": source["prefix_label"],
                "n_frames": int(source["n_frames"]),
                "n_ca": int(source["n_residues"]),
                "frames_times_ca": int(source["n_frames"]) * int(source["n_residues"]),
                "input_path": source["centered_input_path"],
                "input_sha256": observed_hash,
                "input_size_bytes": observed_size,
                "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
                "hash_verified": verify_hashes and observed_hash == source["centered_input_sha256"],
                "size_verified": observed_size == int(source["centered_input_size_bytes"]),
            }
        )
    panel.sort(key=lambda row: (row["system_id"], row["n_frames"]))
    systems = {row["system_id"] for row in panel}
    if len(panel) != 48 or len(systems) != 8:
        raise ValueError("locked panel must contain 48 settings and 8 systems")
    for system in systems:
        prefixes = tuple(row["prefix_id"] for row in panel if row["system_id"] == system)
        if prefixes != PREFIXES:
            raise ValueError(f"unexpected prefix set for {system}: {prefixes}")
    if not all(row["size_verified"] for row in panel):
        raise ValueError("one or more input sizes differ from the frozen manifest")
    if verify_hashes and not all(row["hash_verified"] for row in panel):
        raise ValueError("one or more input hashes differ from the frozen manifest")
    path = OUT / "01_panel/atlas_48_setting_manifest.tsv"
    write_tsv(path, panel)
    write_text(path.with_suffix(".sha256"), f"{sha256(path)}  {path.name}")
    return panel


def derive_timeout() -> tuple[int, float]:
    """Derive the common timeout without inspecting new benchmark results."""

    rows = read_tsv(HISTORICAL_TIMING)
    values = [
        float(row["C1_PDB_tool_wall_seconds"])
        for row in rows
        if row["method"] in {"stock_MDLovoFit_no_output", "MDCoreAlign_throughput_CPU_minimal"}
        and row["is_warmup"] == "False"
        and row["status"] == "success"
    ]
    historical_max = max(values)
    timeout = int(math.ceil((historical_max * 1.5 + 60.0) / 60.0) * 60)
    return timeout, historical_max


def preregister(panel: list[dict[str, Any]]) -> None:
    """Write and hash the complete AT2 contract before timing."""

    destination = OUT / "02_preregistration"
    timeout, historical_max = derive_timeout()
    benchmark_contract = f"""# Strict single-thread ATLAS benchmark contract

This prospectively locked study compares only production MDCoreAlign throughput
on CPU with stock MDLovoFit on the existing 48-setting ATLAS R1 panel.

- Input: identical pre-generated, per-frame-centred, CA-only multi-model PDB.
- Boundary: PDB parse through stable-reference estimation and normal all-frame
  fit/RMSD completion; no aligned-coordinate output, RMSF, W5, GPU, or raw-XTC decode.
- Core fraction: 0.60, fixed before timing.
- CPU: logical CPU `{LOGICAL_CPU}` for both methods; physical mapping must pass AT3.
- Numerical pools: one thread using the registered environment.
- Execution: sequential; one unmeasured warm-up and five measured repeats.
- Cache: full-file sequential read outside the timer before every measured command.
- Idle gate: load1 <= {IDLE_LOAD1_MAX:.1f} and selected-CPU busy <= {IDLE_SELECTED_CPU_BUSY_MAX_PERCENT:.1f}%.
- Timeout: {timeout} seconds for either method, derived before timing from the
  historical maximum ({historical_max:.6f} s) with a 1.5x plus 60 s margin.
- Primary runtime: per-setting median wall time and panel median paired speedup.
- Primary CI: system-clustered bootstrap retaining all six prefixes per system.

Formal timing is prohibited unless AT3 demonstrates the thread and affinity contract.
"""
    write_text(destination / "benchmark_contract.md", benchmark_contract)
    commands = []
    for method in (METHOD_MDCA, METHOD_MDLOVOFIT):
        example = command_for(method, panel[0], LOGICAL_CPU)
        commands.append(
            {
                "method": method,
                "logical_cpu": LOGICAL_CPU,
                "thread_environment_sha256": THREAD_ENV_HASH,
                "command_template": " ".join(example).replace(
                    str(absolute_input(panel[0])), "{input_pdb}"
                ),
                "input_parsed_inside_timing": True,
                "aligned_coordinates_materialised": False,
                "rmsf_computed": False,
                "w5_computed": False,
                "timeout_seconds": timeout,
            }
        )
    write_tsv(destination / "command_contract.tsv", commands)

    ordered = sorted(
        panel,
        key=lambda row: (stable_hash_int(setting_id(row)), setting_id(row)),
    )
    fifth_first = {
        setting_id(row): METHOD_MDCA if index % 2 == 0 else METHOD_MDLOVOFIT
        for index, row in enumerate(ordered)
    }
    run_order = []
    for row in panel:
        sid = setting_id(row)
        warm_first = METHOD_MDCA if stable_hash_int(sid + ":warmup") % 2 == 0 else METHOD_MDLOVOFIT
        warm_second = METHOD_MDLOVOFIT if warm_first == METHOD_MDCA else METHOD_MDCA
        for position, method in enumerate((warm_first, warm_second), start=1):
            run_order.append(
                {
                    "setting_id": sid,
                    "phase": "warmup",
                    "repeat": 0,
                    "order_position": position,
                    "method": method,
                }
            )
        for repeat in range(1, 6):
            first = METHOD_MDCA if repeat in (1, 3) else METHOD_MDLOVOFIT
            if repeat == 5:
                first = fifth_first[sid]
            second = METHOD_MDLOVOFIT if first == METHOD_MDCA else METHOD_MDCA
            for position, method in enumerate((first, second), start=1):
                run_order.append(
                    {
                        "setting_id": sid,
                        "phase": "measured",
                        "repeat": repeat,
                        "order_position": position,
                        "method": method,
                    }
                )
    write_tsv(destination / "run_order.tsv", run_order)
    measured = [row for row in run_order if row["phase"] == "measured"]
    first_counts = {
        method: sum(row["order_position"] == 1 and row["method"] == method for row in measured)
        for method in (METHOD_MDCA, METHOD_MDLOVOFIT)
    }
    if first_counts[METHOD_MDCA] != first_counts[METHOD_MDLOVOFIT]:
        raise ValueError(f"measured order is not balanced: {first_counts}")

    write_text(
        destination / "statistical_plan.md",
        """# Statistical plan

The five measured repeats are technical replicates. The primary setting value is
the median wall time. Setting speedup is the median stock-MDLovoFit wall time
divided by median MDCoreAlign wall time. The panel statistic is the median of 48
setting speedups. A deterministic 10,000-resample system-clustered bootstrap
samples eight systems with replacement and retains all six prefixes per sampled
system. A setting-level bootstrap is sensitivity analysis only. Report absolute
times, IQR, CV, per-system and full-length summaries, order effects, invalidations,
failures, and timeouts. No noisy result is discarded.
""",
    )
    write_text(
        destination / "core_agreement_contract.md",
        """# Core-agreement contract

Core extraction is untimed. Fraction 0.60 is fixed for continuity and
equal-cardinality comparison, not asserted as biologically optimal. MDLovoFit
membership is parsed only from validated occupancy=1.00 CA output. MDCoreAlign
native labels use production output. Primary matched-k labels take the top k
continuous MDCoreAlign weights, where k is actual MDLovoFit core size, using
descending weight and ascending input index for deterministic ties. Report native
and matched-k overlap metrics plus chain-aware disagreement segments. The random
equal-cardinality Jaccard reference is 0.428571. Neither method is ground truth.
""",
    )
    targets = [
        destination / "benchmark_contract.md",
        destination / "command_contract.tsv",
        destination / "statistical_plan.md",
        destination / "core_agreement_contract.md",
        destination / "run_order.tsv",
    ]
    write_text(
        destination / "preregistration.sha256",
        "\n".join(f"{sha256(path)}  {path.name}" for path in targets),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-input-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conflict = freeze_at0()
    panel = lock_panel(verify_hashes=args.verify_input_hashes)
    preregister(panel)
    print(
        json.dumps(
            {
                "active_conflict": conflict,
                "hashes_verified": args.verify_input_hashes,
                "panel_settings": len(panel),
                "preregistration": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
