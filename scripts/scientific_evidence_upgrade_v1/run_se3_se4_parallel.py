"""Run the locked synthetic panel with independent template workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import REPO_ROOT, RESULT_ROOT


def execute(command: list[str], log_path: Path, environment: dict[str, str]) -> tuple[str, int]:
    template_id = command[-1]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return template_id, completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    script = REPO_ROOT / "scripts/scientific_evidence_upgrade_v1/run_se3_se4.py"
    registry = json.loads(
        (RESULT_ROOT / "02_preregistration/synthetic_truth_registry.json").read_text()
    )
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    subprocess.run(
        [sys.executable, str(script), "--prepare-only"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    log_dir = RESULT_ROOT / "03_synthetic_truth/worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, str(script), "--template", str(entry["domain_id"])]
        for entry in registry["templates"]
    ]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                execute,
                command,
                log_dir / f"{command[-1]}.log",
                environment,
            ): command[-1]
            for command in commands
        }
        for future in as_completed(futures):
            template_id, return_code = future.result()
            print(f"template {template_id}: return_code={return_code}", flush=True)
            if return_code != 0:
                failures.append(template_id)
    if failures:
        raise RuntimeError(f"synthetic workers failed: {failures}")
    subprocess.run(
        [sys.executable, str(script), "--merge-only"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
