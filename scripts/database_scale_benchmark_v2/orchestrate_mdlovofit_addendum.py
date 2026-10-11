#!/usr/bin/env python3
"""Continue M2 deployment phases after the locked feasibility pilot finishes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

from scripts.database_scale_benchmark_v2.common import M_ROOT, ROOT, write_json


RUNNER = ROOT / "scripts/database_scale_benchmark_v2/run_mdlovofit_addendum.py"
REAL_LONG_RUNNER = (
    ROOT
    / "scripts/database_scale_benchmark_v2/run_real_long_mdlovofit_addendum.py"
)
REAL_LONG_INTERNAL_RUNNER = (
    ROOT
    / "scripts/database_scale_benchmark_v2/run_real_long_internal_prefixes.py"
)
B3_RUNNER = ROOT / "scripts/database_scale_benchmark_v2/run_b3_full_mdcath.py"
REPORT_RUNNER = (
    ROOT
    / "scripts/database_scale_benchmark_v2/make_mdlovofit_comparison_report.py"
)


def phase_commands(python: Path) -> list[tuple[str, list[str]]]:
    """Return the immutable post-pilot execution order."""

    return [
        ("finalize_pilot", [str(python), str(RUNNER), "finalize-pilot"]),
        ("all_domain", [str(python), str(RUNNER), "all-domain"]),
        ("finalize_all_domain", [str(python), str(RUNNER), "finalize-all-domain"]),
        ("deployment", [str(python), str(RUNNER), "deployment"]),
        ("finalize_deployment", [str(python), str(RUNNER), "finalize-deployment"]),
        (
            "resume_internal_B3",
            [str(python), str(B3_RUNNER), "full", "--input-workers", "2"],
        ),
        ("finalize_internal_B3", [str(python), str(B3_RUNNER), "finalize-full"]),
        (
            "real_long_internal",
            [str(python), str(REAL_LONG_INTERNAL_RUNNER), "run"],
        ),
        (
            "finalize_real_long_internal",
            [str(python), str(REAL_LONG_INTERNAL_RUNNER), "finalize"],
        ),
        ("real_long", [str(python), str(REAL_LONG_RUNNER), "run"]),
        ("finalize_real_long", [str(python), str(REAL_LONG_RUNNER), "finalize"]),
        ("make_comparison_report", [str(python), str(REPORT_RUNNER)]),
    ]


def main() -> None:
    status_path = M_ROOT / "orchestrator_status.json"
    completed: list[str] = []
    try:
        for phase, command in phase_commands(Path(sys.executable)):
            print(f"[MDLovoFit orchestrator] starting {phase}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            completed.append(phase)
            write_json(
                status_path,
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "running",
                    "completed_phases": completed,
                    "current_or_next_phase": phase,
                },
            )
        write_json(
            status_path,
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "completed_phases": completed,
            },
        )
    except Exception as exc:
        write_json(
            status_path,
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "completed_phases": completed,
                "failure": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
