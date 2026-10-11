"""Run or print the staged mid-level experimental suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "midlevel_suite_manifest.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the staged suite runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help="Module id to run. Repeatable. Defaults to all modules.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute commands. By default this script only prints them.",
    )
    return parser.parse_args()


def load_manifest() -> dict[str, object]:
    """Load the mid-level experiment manifest."""

    return json.loads(MANIFEST_PATH.read_text())


def main() -> None:
    """Print or execute the requested staged experiment commands."""

    args = parse_args()
    manifest = load_manifest()
    modules = manifest["modules"]
    requested = set(args.module) if args.module else None

    print(f"suite={manifest['suite_name']}")
    print(f"goal={manifest['goal']}")
    for module in modules:
        if requested is not None and module["id"] not in requested:
            continue
        print()
        print(f"[{module['id']}] {module['title']}")
        print(f"command: {module['default_command']}")
        if args.execute:
            completed = subprocess.run(
                module["default_command"],
                cwd=ROOT,
                shell=True,
                check=False,
            )
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
