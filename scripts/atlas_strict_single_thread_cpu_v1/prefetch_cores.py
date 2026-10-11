#!/usr/bin/env python3
"""Prefetch independent AT6 exports; this scheduling is outside formal timing."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from queue import Queue
from typing import Any

from scripts.atlas_strict_single_thread_cpu_v1.common import (
    FRACTION,
    OUT,
    THREAD_ENV,
    read_tsv,
    setting_id,
    write_json,
    write_tsv,
)
from scripts.atlas_strict_single_thread_cpu_v1.extract_cores import run_mdlovofit_export

CPUS = tuple(range(17, 25))


def export_schedule(panel: list[dict[str, str]]) -> list[tuple[dict[str, str], str]]:
    """Retain the original export identities and put the largest inputs first."""

    full = [row for row in panel if row["prefix_id"] == "f10001"]
    if len(panel) != 48 or len(full) != 8 or len({setting_id(row) for row in panel}) != 48:
        raise ValueError("AT6 prefetch requires the locked 48-setting, eight-system panel")
    tasks = [
        (row, extraction_id)
        for row in full
        for extraction_id in ("self_audit_1", "self_audit_2")
    ]
    tasks.extend((row, "registered_1") for row in panel)
    return sorted(
        tasks,
        key=lambda task: (
            -int(task[0]["n_frames"]) * int(task[0]["n_ca"]),
            setting_id(task[0]),
            task[1],
        ),
    )


def main() -> None:
    """Assign one distinct CPU to each active export and retain scheduling evidence."""

    available = set(os.sched_getaffinity(0))
    unavailable = set(CPUS) - available
    if unavailable:
        raise RuntimeError(f"AT6 prefetch CPUs unavailable in current affinity: {sorted(unavailable)}")
    panel = read_tsv(OUT / "01_panel/atlas_48_setting_manifest.tsv")
    tasks = export_schedule(panel)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = OUT / "06_core_extraction/prefetch_audit" / stamp
    destination.mkdir(parents=True)
    write_json(
        destination / "contract.json",
        {
            "timing_scope": "untimed_core_extraction_not_AT5",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "logical_cpus": list(CPUS),
            "available_affinity": sorted(available),
            "max_workers": 8,
            "numerical_threads_per_export": 1,
            "thread_environment": THREAD_ENV,
            "fraction": FRACTION,
            "planned_exports": len(tasks),
            "self_audit_exports": 16,
            "registered_1_exports": 48,
            "priority": "descending n_frames times n_ca",
            "purpose": "Prefetch unchanged stock -f 0.60 -t exports for the original AT6 identities.",
            "stochastic_followup": "Any conditional extra exports remain in the original AT6 main.",
        },
    )
    write_tsv(
        destination / "schedule.tsv",
        [
            {
                "priority": index,
                "setting_id": setting_id(row),
                "extraction_id": extraction_id,
                "n_frames": row["n_frames"],
                "n_ca": row["n_ca"],
                "frames_times_ca": int(row["n_frames"]) * int(row["n_ca"]),
            }
            for index, (row, extraction_id) in enumerate(tasks, start=1)
        ],
    )
    cpus: Queue[int] = Queue()
    for cpu in CPUS:
        cpus.put(cpu)
    log_lock = threading.Lock()

    def event(payload: dict[str, Any]) -> None:
        line = json.dumps(
            {"utc": datetime.now(timezone.utc).isoformat(), **payload}, sort_keys=True
        )
        with log_lock:
            with (destination / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(line, flush=True)

    def run_one(row: dict[str, str], extraction_id: str) -> dict[str, Any]:
        cpu = cpus.get()
        base = {"setting_id": setting_id(row), "extraction_id": extraction_id, "logical_cpu": cpu}
        try:
            event({**base, "event": "start"})
            result = run_mdlovofit_export(row, extraction_id, logical_cpu=cpu)
            summary = {
                **base,
                "status": "validated",
                "checkpoint_reused": result["checkpoint_reused"],
                "export_logical_cpu": result["logical_cpu"],
                "mask_sha256": result["mask_sha256"],
                "model_count": result["model_count"],
                "core_size": result["core_size"],
            }
            event({**summary, "event": "complete"})
            return summary
        except Exception as error:
            event({**base, "event": "failed", "error": repr(error)})
            raise
        finally:
            cpus.put(cpu)

    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="at6_export") as executor:
        futures = {
            executor.submit(run_one, row, extraction_id): (row, extraction_id)
            for row, extraction_id in tasks
        }
        for future in as_completed(futures):
            row, extraction_id = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "setting_id": setting_id(row),
                        "extraction_id": extraction_id,
                        "error": repr(error),
                    }
                )
    write_json(
        destination / "completion.json",
        {
            "timing_scope": "untimed_core_extraction_not_AT5",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "planned_exports": len(tasks),
            "validated_exports": len(completed),
            "failures": failures,
            "exports": completed,
            "status": "pass" if not failures and len(completed) == len(tasks) else "failed",
        },
    )
    if failures:
        raise SystemExit(f"AT6 prefetch failed for {len(failures)} exports; see {destination}")


if __name__ == "__main__":
    main()
