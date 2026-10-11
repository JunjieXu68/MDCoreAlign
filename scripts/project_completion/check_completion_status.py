#!/usr/bin/env python3
"""Print the current MDCoreAlign project-completion status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATUS_JSON = Path("results/project_completion/status/current_status.json")
MASTER_LOG = Path("results/project_completion/logs/master_completion.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=int, default=25, help="Number of master-log lines to print.")
    args = parser.parse_args()
    if not STATUS_JSON.exists():
        print(f"Status file not found: {STATUS_JSON}")
        return
    data = json.loads(STATUS_JSON.read_text())
    print(f"Updated: {data.get('updated_at')}")
    print(f"Working directory: {data.get('working_directory')}")
    print(f"Git commit: {data.get('git_commit')}")
    print()
    print(f"{'phase':45s} {'status':18s} message")
    print("-" * 90)
    for phase, entry in data.get("phases", {}).items():
        print(f"{phase:45s} {entry.get('status',''):18s} {entry.get('message','')}")
    if MASTER_LOG.exists() and args.tail > 0:
        print("\nMaster log tail:")
        lines = MASTER_LOG.read_text(errors="replace").splitlines()
        for line in lines[-args.tail :]:
            print(line)


if __name__ == "__main__":
    main()
