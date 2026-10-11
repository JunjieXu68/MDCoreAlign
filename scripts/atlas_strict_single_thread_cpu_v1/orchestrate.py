#!/usr/bin/env python3
"""Run AT0-AT10 sequentially after registered host-idle gates pass."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timezone

from scripts.atlas_strict_single_thread_cpu_v1.common import (
    OUT,
    PYTHON,
    ROOT,
    thread_environment,
    write_json,
)
from scripts.atlas_strict_single_thread_cpu_v1.prepare_study import (
    IDLE_LOAD1_MAX,
    process_snapshot,
    snapshot_has_active_conflict,
)

UTC = timezone.utc
LOG = OUT / "orchestrator.log"


def log(message: str) -> None:
    """Append one durable orchestration event."""

    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(UTC).isoformat()} {message}"
    with LOG.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(line, flush=True)


def active_conflict() -> bool:
    """Detect unrelated registered heavy work while ignoring this orchestrator."""

    snapshot, _ = process_snapshot()
    ignored = {os.getpid(), os.getppid()}
    return snapshot_has_active_conflict(snapshot, ignored_pids=ignored)


def wait_for_idle() -> None:
    """Wait without changing host scheduling or unrelated processes."""

    while active_conflict() or os.getloadavg()[0] > IDLE_LOAD1_MAX:
        log(
            f"WAITING active_conflict={active_conflict()} "
            f"load1={os.getloadavg()[0]:.3f} threshold={IDLE_LOAD1_MAX:.1f}"
        )
        time.sleep(60)
    log(f"HOST_GATE_PASS load1={os.getloadavg()[0]:.3f}")


def run_phase(name: str, module: str, *arguments: str) -> None:
    """Run one phase synchronously and stop on any failure."""

    command = [str(PYTHON), "-m", module, *arguments]
    log(f"START {name} command={' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT, check=False)
    log(f"END {name} exit={result.returncode}")
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit {result.returncode}")


def run_validation() -> None:
    """Run and retain the required AT10 validation evidence."""

    destination = OUT / "09_report/validation"
    destination.mkdir(parents=True, exist_ok=True)
    commands = {
        "focused_pytest.txt": [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            "tests/test_atlas_strict_single_thread_cpu_v1.py",
            "tests/test_atlas_untimed_export_recovery.py",
        ],
        "full_pytest.txt": [str(PYTHON), "-m", "pytest", "-q", "tests"],
        "ruff.txt": [
            str(PYTHON),
            "-m",
            "ruff",
            "check",
            "scripts/atlas_strict_single_thread_cpu_v1",
            "tests/test_atlas_strict_single_thread_cpu_v1.py",
            "tests/test_atlas_untimed_export_recovery.py",
        ],
        "compileall.txt": [
            str(PYTHON),
            "-m",
            "compileall",
            "-q",
            "scripts/atlas_strict_single_thread_cpu_v1",
            "tests/test_atlas_strict_single_thread_cpu_v1.py",
            "tests/test_atlas_untimed_export_recovery.py",
        ],
    }
    statuses = []
    for filename, command in commands.items():
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=thread_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        (destination / filename).write_text(
            f"$ {' '.join(command)}\n\n{result.stdout}\n{result.stderr}"
        )
        statuses.append((filename, result.returncode))
        write_json(
            OUT / "09_report/validation_results.json",
            {
                "generated_utc": datetime.now(UTC).isoformat(),
                "checks": [{"log": name, "returncode": code} for name, code in statuses],
            },
        )
        if result.returncode != 0:
            raise SystemExit(f"validation failed: {filename}")
    (OUT / "09_report/validation_summary.md").write_text(
        "# AT10 validation summary\n\n"
        + "\n".join(f"- `{name}`: `PASS`" for name, _ in statuses)
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    """Parse a restart point without repeating completed phases."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-at",
        choices=("AT0", "AT3", "AT4", "AT5", "AT6", "AT7", "AT10"),
        default="AT0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phases = {"AT0": 0, "AT3": 3, "AT4": 4, "AT5": 5, "AT6": 6, "AT7": 7, "AT10": 10}
    start = phases[args.start_at]
    log(f"ORCHESTRATOR_START start_at={args.start_at}")
    wait_for_idle()
    if start <= 0:
        run_phase(
            "AT0_AT2_FINAL_FREEZE",
            "scripts.atlas_strict_single_thread_cpu_v1.prepare_study",
            "--verify-input-hashes",
        )
    if start <= 3:
        run_phase("AT3", "scripts.atlas_strict_single_thread_cpu_v1.audit_threads")
    if start <= 4:
        run_phase("AT4", "scripts.atlas_strict_single_thread_cpu_v1.run_preflight")
    if start <= 5:
        run_phase("AT5", "scripts.atlas_strict_single_thread_cpu_v1.run_timing")
    if start <= 6:
        run_phase("AT6", "scripts.atlas_strict_single_thread_cpu_v1.extract_cores")
    if start <= 7:
        run_phase("AT7_AT9", "scripts.atlas_strict_single_thread_cpu_v1.analyze_results")
    if start <= 10:
        log("START AT10_VALIDATION")
        run_validation()
        log("END AT10_VALIDATION exit=0")
        run_phase("AT10_ARCHIVE", "scripts.atlas_strict_single_thread_cpu_v1.package_review")
    log("ORCHESTRATOR_COMPLETE")


if __name__ == "__main__":
    main()
