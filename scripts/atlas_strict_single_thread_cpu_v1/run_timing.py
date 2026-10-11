#!/usr/bin/env python3
"""Run the preregistered strict single-thread ATLAS timing panel."""

from __future__ import annotations

import csv
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.atlas_strict_single_thread_cpu_v1.common import (
    OUT,
    THREAD_ENV_HASH,
    absolute_input,
    command_for,
    read_tsv,
    sha256,
    thread_environment,
    warm_page_cache,
    write_json,
    write_text,
    write_tsv,
)
from scripts.atlas_strict_single_thread_cpu_v1.prepare_study import (
    IDLE_LOAD1_MAX,
    IDLE_SELECTED_CPU_BUSY_MAX_PERCENT,
    LOGICAL_CPU,
    process_snapshot,
    snapshot_has_active_conflict,
)

UTC = timezone.utc
RAW = OUT / "05_timing/raw_timing_records.tsv"
CHECKPOINT = OUT / "05_timing/checkpoint.json"
LOGS = OUT / "05_timing/run_logs"
FIELDS = [
    "setting_id",
    "system_id",
    "prefix_id",
    "method",
    "phase",
    "repeat",
    "pair_attempt",
    "order_position",
    "command",
    "input_sha256",
    "logical_cpu",
    "affinity_mask",
    "thread_environment_hash",
    "start_time_utc",
    "end_time_utc",
    "wall_seconds",
    "user_cpu_seconds",
    "system_cpu_seconds",
    "max_rss_mb",
    "exit_code",
    "timeout",
    "invalidated",
    "invalidation_reason",
    "stdout_sha256",
    "stderr_sha256",
]


def cpu_sample(cpu: int) -> tuple[int, int]:
    """Return total and idle ticks for one logical CPU."""

    prefix = f"cpu{cpu} "
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith(prefix):
            fields = [int(value) for value in line.split()[1:]]
            idle = fields[3] + fields[4]
            return sum(fields), idle
    raise ValueError(f"CPU {cpu} is not present in /proc/stat")


def selected_cpu_busy_percent(interval: float = 1.0) -> float:
    """Measure selected-CPU utilisation before a pair."""

    total0, idle0 = cpu_sample(LOGICAL_CPU)
    time.sleep(interval)
    total1, idle1 = cpu_sample(LOGICAL_CPU)
    delta = total1 - total0
    return 0.0 if delta <= 0 else 100.0 * (1.0 - (idle1 - idle0) / delta)


def machine_idle() -> tuple[bool, str]:
    """Apply the prospectively registered machine-idle gate."""

    snapshot, _ = process_snapshot()
    ignored = {os.getpid(), os.getppid()}
    conflict = snapshot_has_active_conflict(snapshot, ignored_pids=ignored)
    load1 = os.getloadavg()[0]
    busy = selected_cpu_busy_percent()
    okay = not conflict and load1 <= IDLE_LOAD1_MAX and busy <= IDLE_SELECTED_CPU_BUSY_MAX_PERCENT
    reason = f"conflict={conflict};load1={load1:.3f};cpu{LOGICAL_CPU}_busy={busy:.3f}%"
    if conflict:
        reason += ";registered_process_conflict"
    if not okay:
        reason += f";snapshot_sha256={__import__('hashlib').sha256(snapshot.encode()).hexdigest()}"
    return okay, reason


def wait_until_idle() -> str:
    """Wait without modifying host state until the registered idle gate passes."""

    while True:
        okay, reason = machine_idle()
        if okay:
            return reason
        print(f"WAITING_FOR_IDLE {datetime.now(UTC).isoformat()} {reason}", flush=True)
        time.sleep(30)


def append_row(row: dict[str, Any]) -> None:
    """Append and fsync one immutable raw command record."""

    RAW.parent.mkdir(parents=True, exist_ok=True)
    exists = RAW.exists()
    with RAW.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def reconcile_and_completed_pairs() -> set[tuple[str, str, int]]:
    """Invalidate interrupted partial pairs and return complete pair keys."""

    if not RAW.exists():
        return set()
    rows = read_tsv(RAW)
    attempts: dict[tuple[str, str, int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["setting_id"],
            row["phase"],
            int(row["repeat"]),
            int(row["pair_attempt"]),
        )
        attempts.setdefault(key, []).append(row)
    complete = set()
    changed = False
    for key, attempt_rows in attempts.items():
        valid_methods = {
            row["method"]
            for row in attempt_rows
            if row["invalidated"] == "False"
            and row["exit_code"] == "0"
            and row["timeout"] == "False"
        }
        if valid_methods == {"MDCoreAlign_throughput_CPU", "stock_MDLovoFit"}:
            complete.add(key[:3])
            continue
        for row in attempt_rows:
            if row["invalidated"] == "False":
                row["invalidated"] = "True"
                row["invalidation_reason"] = "incomplete_or_failed_paired_attempt"
                changed = True
    if changed:
        write_tsv(RAW, rows, fields=FIELDS)
    return complete


def next_pair_attempt(setting: str, phase: str, repeat: int) -> int:
    """Return a monotonically increasing attempt number for one pair."""

    if not RAW.exists():
        return 1
    attempts = [
        int(row["pair_attempt"])
        for row in read_tsv(RAW)
        if row["setting_id"] == setting and row["phase"] == phase and int(row["repeat"]) == repeat
    ]
    return max(attempts, default=0) + 1


def run_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute one command and obtain per-child wait4 resource usage."""

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    start_utc = datetime.now(UTC).isoformat()
    start = time.perf_counter_ns()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=OUT.parent.parent,
            env=thread_environment(),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        usage = None
        status = None
        while status is None:
            waited_pid, waited_status, waited_usage = os.wait4(process.pid, os.WNOHANG)
            if waited_pid == process.pid:
                status = waited_status
                usage = waited_usage
                break
            if time.monotonic() >= deadline:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(1.0)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(process.pid, 0)
                break
            time.sleep(0.01)
        process.returncode = os.waitstatus_to_exitcode(status)
    end = time.perf_counter_ns()
    assert usage is not None
    return {
        "start_time_utc": start_utc,
        "end_time_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": (end - start) / 1e9,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_mb": usage.ru_maxrss / 1024.0,
        "exit_code": process.returncode,
        "timeout": timed_out,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }


def timeout_seconds() -> int:
    """Read the hash-locked common timeout from the command contract."""

    rows = read_tsv(OUT / "02_preregistration/command_contract.tsv")
    values = {int(row["timeout_seconds"]) for row in rows}
    if len(values) != 1:
        raise ValueError("command contract does not contain one common timeout")
    return values.pop()


def update_completion_report() -> None:
    """Summarise current progress from immutable raw rows."""

    rows = read_tsv(RAW) if RAW.exists() else []
    valid = [row for row in rows if row["invalidated"] == "False" and row["exit_code"] == "0"]
    measured = [row for row in valid if row["phase"] == "measured"]
    failures = [row for row in rows if row["exit_code"] != "0" or row["timeout"] == "True"]
    write_tsv(
        OUT / "05_timing/failures_timeouts.tsv",
        failures
        or [
            {
                "setting_id": "none",
                "method": "none",
                "failure_type": "none_at_current_checkpoint",
            }
        ],
    )
    write_text(
        OUT / "05_timing/timing_completion_report.md",
        "\n".join(
            [
                "# Timing completion report",
                "",
                f"- Last checkpoint UTC: `{datetime.now(UTC).isoformat()}`",
                f"- Raw command rows: `{len(rows)}`",
                f"- Valid measured commands: `{len(measured)}/480`",
                f"- Failures/timeouts: `{len(failures)}`",
                f"- Complete: `{'yes' if len(measured) == 480 else 'no'}`",
            ]
        ),
    )


def main() -> None:
    decision = (OUT / "03_thread_affinity_audit/thread_contract_decision.md").read_text()
    if "SINGLE_THREAD_CONTRACT_PASS" not in decision:
        raise SystemExit("formal timing prohibited: AT3 contract has not passed")
    panel_rows = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    panel = {f"{row['system_id']}__{row['prefix_id']}": row for row in panel_rows}
    order = read_tsv(OUT / "02_preregistration/run_order.tsv")
    timeout = timeout_seconds()
    done_pairs = reconcile_and_completed_pairs()
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for entry in order:
        key = (entry["setting_id"], entry["phase"], int(entry["repeat"]))
        grouped.setdefault(key, []).append(entry)
    for pair_key, pair_entries in grouped.items():
        sid, phase, repeat = pair_key
        if (sid, phase, repeat) in done_pairs:
            continue
        idle_evidence = wait_until_idle()
        row = panel[sid]
        pair_attempt = next_pair_attempt(sid, phase, repeat)
        for entry in sorted(pair_entries, key=lambda item: int(item["order_position"])):
            method = entry["method"]
            input_path = absolute_input(row)
            warm_page_cache(input_path)
            log_stem = (
                f"{sid}__{phase}__r{repeat}__a{pair_attempt}__p{entry['order_position']}__{method}"
            )
            stdout_path = LOGS / f"{log_stem}.stdout"
            stderr_path = LOGS / f"{log_stem}.stderr"
            command = command_for(method, row, LOGICAL_CPU)
            result = run_command(
                command,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_seconds=timeout,
            )
            raw = {
                "setting_id": sid,
                "system_id": row["system_id"],
                "prefix_id": row["prefix_id"],
                "method": method,
                "phase": phase,
                "repeat": repeat,
                "pair_attempt": pair_attempt,
                "order_position": entry["order_position"],
                "command": shlex.join(command),
                "input_sha256": row["input_sha256"],
                "logical_cpu": LOGICAL_CPU,
                "affinity_mask": str(LOGICAL_CPU),
                "thread_environment_hash": THREAD_ENV_HASH,
                **result,
                "invalidated": False,
                "invalidation_reason": "",
            }
            append_row(raw)
            write_json(
                CHECKPOINT,
                {
                    "last_key": [sid, phase, repeat, pair_attempt, method],
                    "last_idle_evidence": idle_evidence,
                    "raw_command_rows": len(read_tsv(RAW)),
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )
            update_completion_report()
            print(
                f"COMPLETE {len(read_tsv(RAW))}/576 {sid} {phase} {repeat} {method} "
                f"wall={result['wall_seconds']:.6f}s exit={result['exit_code']}",
                flush=True,
            )
            if result["exit_code"] != 0:
                raise SystemExit(
                    "command failed; retained in raw records: "
                    f"{sid, phase, repeat, pair_attempt, method}"
                )
        done_pairs.add((sid, phase, repeat))
    update_completion_report()
    print("AT5_TIMING_COMPLETE")


if __name__ == "__main__":
    main()
