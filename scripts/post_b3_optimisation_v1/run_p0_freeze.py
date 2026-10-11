#!/usr/bin/env python3
"""Freeze the post-B3 source evidence without touching historical outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform

from scripts.post_b3_optimisation_v1.common import (
    OUT,
    ROOT,
    sha256,
    write_json,
    write_text,
    write_tsv,
)


FREEZE_ROOTS = (
    ROOT / "results/database_scale_benchmark_v2/b2_contracts",
    ROOT / "results/database_scale_benchmark_v2/b3_full_mdcath",
    ROOT / "results/database_scale_benchmark_v2/mdlovofit_comparison",
    ROOT / "results/gpu_matched_validation_v1",
    ROOT / "results/throughput_gpu_v1",
    ROOT / "results/project_completion",
    ROOT / "results/project_completion_v2_unit_corrected",
    ROOT / "results/scaling_optimization_v2/regression",
    ROOT / "results/scaling_optimization_v2/handoff_phase3",
)


def main() -> None:
    target = OUT / "p0_freeze/frozen_source_evidence_manifest.tsv"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite P0 freeze: {target}")
    rows = []
    for root in FREEZE_ROOTS:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                        "sha256": sha256(path),
                        "freeze_group": root.relative_to(ROOT).as_posix(),
                    }
                )
    write_tsv(target, rows)
    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "thread_environment": {
            key: os.environ.get(key, "")
            for key in (
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "source_files": len(rows),
        "source_bytes": sum(int(row["size_bytes"]) for row in rows),
        "historical_outputs_modified": False,
        "git_metadata_modified": False,
        "public_defaults_modified": False,
    }
    write_json(OUT / "p0_freeze/p0_status.json", environment)
    write_text(
        OUT / "p0_freeze/p0_freeze_report.md",
        "\n".join(
            [
                "# P0 post-B3 evidence freeze",
                "",
                f"- Files hashed: `{len(rows):,}`",
                f"- Bytes covered: `{environment['source_bytes']:,}`",
                "- Historical evidence, Git metadata and public defaults were not modified.",
                "- All subsequent outputs are written under `results/post_b3_optimisation_v1/`.",
            ]
        ),
    )


if __name__ == "__main__":
    main()
