#!/usr/bin/env python
"""Write a resource preflight report before mdCATH production launch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mdcath_benchmark.mdcath_common import RESULT_ROOT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=RESULT_ROOT / "cache")
    parser.add_argument("--output", type=Path, default=Path("docs/mdcath_resource_preflight.md"))
    return parser.parse_args()


def run(command: list[str]) -> str:
    """Run a small diagnostic command."""

    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=15)
        return (completed.stdout + completed.stderr).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    """Run preflight diagnostics."""

    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.cache_dir)
    sbatch = shutil.which("sbatch")
    squeue = shutil.which("squeue")
    hf_token_set = bool(os.environ.get("HF_TOKEN"))
    mdcath_cache_env = os.environ.get("MDCATH_CACHE", "")
    recommended = "SLURM array" if sbatch and squeue else "local multiprocessing"
    if usage.free < 500 * 1024**3:
        recommended = "abort due to insufficient cache disk"
    lines = [
        "# mdCATH Resource Preflight",
        "",
        f"- hostname: `{run(['hostname'])}`",
        f"- date: `{run(['date'])}`",
        f"- cache path: `{args.cache_dir}`",
        f"- available disk at cache path: `{usage.free / 1024**3:.1f} GiB`",
        f"- HF_TOKEN set: `{'yes' if hf_token_set else 'no'}`",
        f"- MDCATH_CACHE env: `{mdcath_cache_env or 'not set'}`",
        f"- sbatch available: `{sbatch or 'no'}`",
        f"- squeue available: `{squeue or 'no'}`",
        f"- recommended mode: `{recommended}`",
        "",
        "## df -h",
        "",
        "```text",
        run(["df", "-h", str(args.cache_dir), "/storage/junjie", "/home/jjxu"]),
        "```",
        "",
        "## free -h",
        "",
        "```text",
        run(["free", "-h"]),
        "```",
        "",
        "## lscpu",
        "",
        "```text",
        "\n".join(run(["lscpu"]).splitlines()[:40]),
        "```",
        "",
        "## Notes",
        "",
        "- MDCoreAlign is CPU-oriented; `nvidia-smi` is not required for core runs.",
        "- HF_TOKEN is recommended for production-scale Hugging Face downloads.",
        "- If SLURM is unavailable, production can be launched as a resumable local background process.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

