#!/usr/bin/env python3
"""Verify the registered one-CPU and one-compute-thread contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.atlas_strict_single_thread_cpu_v1.common import (
    METHOD_MDCA,
    METHOD_MDLOVOFIT,
    OUT,
    PYTHON,
    ROOT,
    THREAD_ENV,
    command_for,
    read_tsv,
    setting_id,
    thread_environment,
    write_text,
    write_tsv,
)
from scripts.atlas_strict_single_thread_cpu_v1.prepare_study import (
    LOGICAL_CPU,
    process_snapshot,
)

UTC = timezone.utc


def cpu_topology() -> list[dict[str, str]]:
    """Return the host logical-to-physical CPU mapping."""

    result = subprocess.run(
        ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    header = [value.lower() for value in lines[0]]
    return [dict(zip(header, values, strict=True)) for values in lines[1:]]


def proc_affinity(pid: int) -> str:
    """Read the allowed CPU list from procfs."""

    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def process_identity(pid: int) -> dict[str, str]:
    """Capture a best-effort process identity before a short-lived child exits."""

    base = Path(f"/proc/{pid}")
    try:
        command = base.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        return {
            "pid": str(pid),
            "comm": base.joinpath("comm").read_text().strip(),
            "command": command,
            "affinity": proc_affinity(pid),
        }
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": str(pid), "comm": "exited_before_capture", "command": "", "affinity": ""}


def descendants(pid: int) -> set[int]:
    """Return living recursive child processes from procfs."""

    found: set[int] = set()
    pending = [pid]
    while pending:
        current = pending.pop()
        child_file = Path(f"/proc/{current}/task/{current}/children")
        if not child_file.exists():
            continue
        children = {int(value) for value in child_file.read_text().split()}
        new = children - found
        found.update(new)
        pending.extend(new)
    return found


def task_ticks(pid: int) -> dict[int, int]:
    """Read per-task user plus system CPU ticks."""

    ticks: dict[int, int] = {}
    for stat in Path(f"/proc/{pid}/task").glob("*/stat"):
        try:
            fields = stat.read_text().split()
            ticks[int(stat.parent.name)] = int(fields[13]) + int(fields[14])
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
    return ticks


def trace_command(
    method: str,
    panel_row: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    """Run one untimed command and trace its process/thread behaviour."""

    command = command_for(method, panel_row, LOGICAL_CPU)
    environment = thread_environment()
    events: list[str] = []
    active_tasks: set[int] = set()
    seen_children: set[int] = set()
    affinities: set[str] = set()
    launcher_affinities: set[str] = set()
    max_tasks = 0
    previous: dict[int, int] = {}
    compute_sampling_started = False
    with tempfile.TemporaryDirectory(prefix="mdca_at3_") as temporary:
        stdout_path = Path(temporary) / "stdout.txt"
        stderr_path = Path(temporary) / "stderr.txt"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            while process.poll() is None:
                try:
                    affinity = proc_affinity(process.pid)
                    if not compute_sampling_started:
                        launcher_affinities.add(affinity)
                        compute_sampling_started = affinity == str(LOGICAL_CPU)
                    if not compute_sampling_started:
                        time.sleep(0.005)
                        continue
                    affinities.add(affinity)
                    current = task_ticks(process.pid)
                    max_tasks = max(max_tasks, len(current))
                    active_tasks.update(
                        task for task, ticks in current.items() if ticks > previous.get(task, ticks)
                    )
                    previous = current
                    children = descendants(process.pid)
                    new_children = children - seen_children
                    if new_children:
                        identities = [process_identity(pid) for pid in sorted(new_children)]
                        events.append(f"{datetime.now(UTC).isoformat()} new_children={identities}")
                    seen_children.update(children)
                except FileNotFoundError:
                    pass
                time.sleep(0.02)
            elapsed = time.perf_counter() - started
        stdout_text = stdout_path.read_text(errors="replace")
        stderr_text = stderr_path.read_text(errors="replace")
    return (
        {
            "setting_id": setting_id(panel_row),
            "method": method,
            "exit_code": process.returncode,
            "wall_seconds_untimed_audit": elapsed,
            "max_task_count": max_tasks,
            "active_task_count": len(active_tasks),
            "active_task_ids": ",".join(str(value) for value in sorted(active_tasks)),
            "child_process_count": len(seen_children),
            "child_process_ids": ",".join(str(value) for value in sorted(seen_children)),
            "observed_affinity_masks": ",".join(sorted(affinities)),
            "launcher_affinity_masks": ",".join(sorted(launcher_affinities)),
            "compute_sampling_started": compute_sampling_started,
            "stdout_tail": " | ".join(stdout_text.splitlines()[-3:]),
            "stderr_tail": " | ".join(stderr_text.splitlines()[-3:]),
        },
        events,
    )


def audit_threadpools(destination: Path) -> dict[str, Any]:
    """Inspect loaded numerical pools separately from the production command."""

    code = """
import json
import os
import numpy
from mdcorealign.benchmark_minimal import solve_minimal_from_pdb
from threadpoolctl import threadpool_info
print(json.dumps({
    "pid": os.getpid(),
    "affinity": sorted(os.sched_getaffinity(0)),
    "threadpools": threadpool_info(),
}, sort_keys=True))
"""
    command = ["taskset", "-c", str(LOGICAL_CPU), str(PYTHON), "-c", code]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=thread_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"threadpool audit failed: {result.stderr}")
    payload = json.loads(result.stdout)
    write_text(destination, json.dumps(payload, indent=2, sort_keys=True))
    return payload


def machine_state() -> list[dict[str, Any]]:
    """Capture state without changing governors, boost, or scheduling."""

    load = os.getloadavg()
    memory = subprocess.run(["free", "-b"], capture_output=True, text=True, check=True)
    governor = Path(f"/sys/devices/system/cpu/cpu{LOGICAL_CPU}/cpufreq/scaling_governor")
    boost = Path("/sys/devices/system/cpu/cpufreq/boost")
    return [
        {"item": "snapshot_utc", "value": datetime.now(UTC).isoformat()},
        {"item": "load_average_1m", "value": load[0]},
        {"item": "load_average_5m", "value": load[1]},
        {"item": "load_average_15m", "value": load[2]},
        {"item": "selected_logical_cpu", "value": LOGICAL_CPU},
        {
            "item": "governor",
            "value": governor.read_text().strip() if governor.exists() else "unavailable",
        },
        {"item": "boost", "value": boost.read_text().strip() if boost.exists() else "unavailable"},
        {"item": "memory", "value": memory.stdout.replace("\n", " | ")},
        {
            "item": "input_filesystem",
            "value": subprocess.run(
                ["df", "-T", str(OUT)], capture_output=True, text=True, check=True
            ).stdout.replace("\n", " | "),
        },
    ]


def main() -> None:
    destination = OUT / "03_thread_affinity_audit"
    snapshot, conflict = process_snapshot()
    if conflict:
        write_text(
            destination / "thread_contract_decision.md",
            "# Thread contract decision\n\n`DEFERRED_ACTIVE_CONFLICT`\n\nAT3 was not run because a registered CPU/storage-heavy process was active.",
        )
        raise SystemExit("AT3 deferred: active CPU/storage-heavy task detected")
    topology = cpu_topology()
    write_tsv(destination / "cpu_topology.tsv", topology)
    selected = next(row for row in topology if int(row["cpu"]) == LOGICAL_CPU)
    siblings = [
        row["cpu"]
        for row in topology
        if row["core"] == selected["core"] and row["socket"] == selected["socket"]
    ]
    write_tsv(
        destination / "affinity_contract.tsv",
        [
            {
                "logical_cpu": LOGICAL_CPU,
                "physical_core": selected["core"],
                "socket": selected["socket"],
                "numa_node": selected["node"],
                "hyperthread_siblings": ",".join(siblings),
                "required_affinity_mask": str(LOGICAL_CPU),
                "selection_rule": "fixed topology-based preregistration before timing",
            }
        ],
    )
    write_tsv(
        destination / "thread_environment.tsv",
        [
            {
                "variable": key,
                "required_value": value,
                "observed_parent_value": thread_environment().get(key, ""),
            }
            for key, value in THREAD_ENV.items()
        ],
    )
    write_tsv(destination / "machine_state.tsv", machine_state())

    panel = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    small = min(panel, key=lambda row: int(row["frames_times_ca"]))
    large = max(panel, key=lambda row: int(row["frames_times_ca"]))
    traces = []
    events = [f"pre_audit_process_snapshot\n{snapshot}"]
    audit_json = destination / "threadpoolctl.json"
    pool_audit = audit_threadpools(audit_json)
    for row in (small, large):
        for method in (METHOD_MDCA, METHOD_MDLOVOFIT):
            trace, process_events = trace_command(method, row)
            traces.append(trace)
            events.extend(process_events)
    write_tsv(destination / "mdlovofit_thread_trace.tsv", traces)
    write_text(destination / "process_tree.txt", "\n".join(events))

    pools = pool_audit["threadpools"]
    pools_pass = all(int(pool.get("num_threads", 0)) == 1 for pool in pools)
    pool_affinity_pass = pool_audit["affinity"] == [LOGICAL_CPU]
    traces_pass = all(
        row["exit_code"] == 0
        and row["child_process_count"] == 0
        and row["active_task_count"] <= 1
        and row["compute_sampling_started"]
        and row["observed_affinity_masks"] == str(LOGICAL_CPU)
        for row in traces
    )
    passed = pools_pass and pool_affinity_pass and traces_pass
    decision = "SINGLE_THREAD_CONTRACT_PASS" if passed else "SINGLE_THREAD_CONTRACT_FAILED"
    write_text(
        destination / "thread_contract_decision.md",
        "\n".join(
            [
                "# Thread contract decision",
                "",
                f"`{decision}`",
                "",
                f"- Logical CPU: `{LOGICAL_CPU}`",
                f"- Physical core: `{selected['core']}`",
                f"- Hyperthread siblings: `{','.join(siblings)}`",
                f"- Numerical pools report one thread: `{'yes' if pools_pass else 'no'}`",
                f"- Pool probe affinity is CPU {LOGICAL_CPU}: `{'yes' if pool_affinity_pass else 'no'}`",
                f"- No worker processes and <=1 active compute task: `{'yes' if traces_pass else 'no'}`",
            ]
        ),
    )
    if not passed:
        raise SystemExit(decision)
    print(decision)


if __name__ == "__main__":
    main()
