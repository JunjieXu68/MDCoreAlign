#!/usr/bin/env python3
"""Create the compact T0-T7 handoff without changing frozen evidence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mainline_reintegration_v1.run_r0_inventory import (  # noqa: E402
    frozen_integrity,
)


RESULTS = ROOT / "results" / "throughput_gpu_v1"
HANDOFF = RESULTS / "handoff_to_chatgpt"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _json(path: Path):
    return json.loads(path.read_text())


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _method(
    rows: list[dict[str, str]],
    system: str,
    contract: str,
    method: str,
) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["system"] == system
        and row["contract"] == contract
        and row["method"] == method
    )


def main() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    integrity = frozen_integrity()
    if integrity["status"] != "pass":
        raise RuntimeError(f"frozen integrity failed: {integrity}")
    t3 = _rows(RESULTS / "t3_balanced_lite" / "balanced_lite_summary.tsv")
    t5 = _json(RESULTS / "t5_gpu_design" / "t5_summary.json")
    t6 = _json(RESULTS / "t6_scaling" / "manifest.json")
    t6_rows = _rows(RESULTS / "t6_scaling" / "cpu_gpu_scaling_matrix.tsv")
    t7 = _json(RESULTS / "t7_real_long" / "manifest.json")
    t7_rows = _rows(
        RESULTS / "t7_real_long" / "real_long_matched_contracts.tsv"
    )
    t7_quality = _rows(
        RESULTS / "t7_real_long" / "real_long_quality_vs_exact.tsv"
    )

    key_rows: list[dict[str, object]] = []
    for row in t3:
        key_rows.append(
            {
                "stage": "T3",
                "scope": row["split"],
                "metric": "balanced_lite_high_confidence_fraction",
                "value": row["high_confidence_fraction"],
                "interpretation": (
                    "Fixed-cost audit; warning is not an automatic exact "
                    "fallback."
                ),
            }
        )
        key_rows.append(
            {
                "stage": "T3",
                "scope": row["split"],
                "metric": "predefined_tail_warning_fraction",
                "value": row["tail_warning_recall"],
                "interpretation": (
                    "Below 1.0, so balanced-lite is not a safety guarantee."
                ),
            }
        )
    key_rows.extend(
        [
            {
                "stage": "T5",
                "scope": "10k_frames_153_residues",
                "metric": "best_GPU_end_to_end_speedup_vs_CPU_throughput",
                "value": t5["best_GPU_setting"][
                    "end_to_end_speedup_vs_CPU"
                ],
                "interpretation": (
                    "Warm loaded synthetic scaling input; Horn/float32."
                ),
            },
            {
                "stage": "T5",
                "scope": "10k_frames_153_residues",
                "metric": "GPU_implementation_parity_pass",
                "value": (
                    f"{t5['implementation_parity_pass_count']}/"
                    f"{t5['implementation_parity_total']}"
                ),
                "interpretation": (
                    "SVD/cuSOLVER alias/Horn across float64, float32 and mixed."
                ),
            },
            {
                "stage": "T6",
                "scope": "synthetic_scaling_matrix",
                "metric": "GPU_end_to_end_target_pass",
                "value": (
                    f"{t6['GPU_end_to_end_target_pass_cases']}/"
                    f"{t6['GPU_case_count']}"
                ),
                "interpretation": (
                    "Target was >=1.5x; inputs are not biological trajectories."
                ),
            },
            {
                "stage": "T6",
                "scope": "synthetic_scaling_matrix",
                "metric": "GPU_compute_target_pass",
                "value": (
                    f"{t6['GPU_compute_target_pass_cases']}/"
                    f"{t6['GPU_case_count']}"
                ),
                "interpretation": "Target was >=2x GPU compute speedup.",
            },
        ]
    )
    for system in ("YiiP_90ns", "NhaA", "ADK"):
        cpu = _method(
            t7_rows,
            system,
            "metrics_transforms",
            "CPU_throughput",
        )
        gpu = _method(
            t7_rows,
            system,
            "metrics_transforms",
            "GPU_throughput_float32_Horn",
        )
        material_cpu = _method(
            t7_rows,
            system,
            "aligned_trajectory_materialisation",
            "CPU_throughput",
        )
        material_gpu = _method(
            t7_rows,
            system,
            "aligned_trajectory_materialisation",
            "GPU_throughput_float32_Horn",
        )
        quality = next(
            row
            for row in t7_quality
            if row["system"] == system
            and row["method"] == "GPU_throughput_float32_Horn"
        )
        key_rows.extend(
            [
                {
                    "stage": "T7",
                    "scope": system,
                    "metric": "GPU_vs_CPU_throughput_speedup",
                    "value": (
                        float(cpu["runtime_seconds"])
                        / float(gpu["runtime_seconds"])
                    ),
                    "interpretation": "Warm metrics/transforms-only contract.",
                },
                {
                    "stage": "T7",
                    "scope": system,
                    "metric": "GPU_vs_CPU_materialisation_speedup",
                    "value": (
                        float(material_cpu["runtime_seconds"])
                        / float(material_gpu["runtime_seconds"])
                    ),
                    "interpretation": (
                        "Aligned coordinates materialised; PDB writing excluded."
                    ),
                },
                {
                    "stage": "T7",
                    "scope": system,
                    "metric": "GPU_core_jaccard_vs_exact",
                    "value": quality["core_jaccard_vs_exact"],
                    "interpretation": "Practical scientific quality metric.",
                },
                {
                    "stage": "T7",
                    "scope": system,
                    "metric": "GPU_RMSF_spearman_vs_exact",
                    "value": quality["RMSF_spearman_vs_exact"],
                    "interpretation": "RMSF remains alignment-frame dependent.",
                },
            ]
        )
    key_rows.extend(
        [
            {
                "stage": "validation",
                "scope": "canonical_CPU",
                "metric": "tests",
                "value": "244 passed; 6 optional-GPU skipped; 2 subtests passed",
                "interpretation": "Full suite in canonical CPU environment.",
            },
            {
                "stage": "validation",
                "scope": "isolated_CUDA12",
                "metric": "tests",
                "value": "250 passed",
                "interpretation": "Full suite with CuPy GPU tests enabled.",
            },
            {
                "stage": "validation",
                "scope": "frozen_evidence",
                "metric": "integrity",
                "value": (
                    f"{integrity['project_completion_checked']}/9 project; "
                    f"{integrity['golden_checked']}/34 golden"
                ),
                "interpretation": "No frozen hash failure.",
            },
        ]
    )
    _write_table(HANDOFF / "02_key_metrics.tsv", key_rows)

    mode_text = """# Practical mode contracts

| Contract | Primary algorithm | Automatic exact fallback | Status |
|---|---|---:|---|
| `exact` | CPU `sequential_lean` | not applicable | public exact/default reference |
| `throughput` | fixed-budget fast-global | no | experimental opt-in |
| `balanced-lite` | fast-global + fixed sampled audit + at most one refinement | no | experimental opt-in |
| `local-deformation` | alignment-free chain-aware W5 | no | experimental diagnostic |

`--backend cpu|gpu|auto` is implemented for explicit experimental contracts.
Exact remains CPU-only. The public default is unchanged. Low-confidence
balanced-lite output carries a warning and recommends explicit exact
verification; it never silently launches `sequential_lean`.

W5 outputs local deformation/rigidity signals. They are not RMSD, global
mobility, or mature hard biological-domain labels. The optional sparse graph
remains CPU-only and experimental.
"""
    (HANDOFF / "01_mode_contracts.md").write_text(mode_text)

    commands = f"""# Reproducible commands

Repository root: `.`

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
NUMEXPR_NUM_THREADS=2 PYTHONPATH=src \\
.venv-gpu-cu12/bin/python scripts/throughput_gpu_v1/run_t4_t5.py all

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
NUMEXPR_NUM_THREADS=2 PYTHONPATH=src \\
.venv-gpu-cu12/bin/python scripts/throughput_gpu_v1/run_t6_scaling.py

PYTHONPATH=src /home/jjxu/miniconda3/bin/python \\
scripts/throughput_gpu_v1/prepare_t7_real_long.py

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
NUMEXPR_NUM_THREADS=2 PYTHONPATH=src \\
.venv-gpu-cu12/bin/python scripts/throughput_gpu_v1/run_t7_real_long.py

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
NUMEXPR_NUM_THREADS=2 PYTHONPATH=src \\
/home/jjxu/miniconda3/bin/python -m pytest -q

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
NUMEXPR_NUM_THREADS=2 PYTHONPATH=src \\
.venv-gpu-cu12/bin/python -m pytest -q
```

The isolated CUDA environment used `{sys.executable}` for finalization and
`.venv-gpu-cu12/bin/python` for GPU runs. Input decode/extraction is excluded
from T7 warm loaded timings.
"""
    (HANDOFF / "03_reproducible_commands.md").write_text(commands)

    limitations = """# Negative results and limits

1. Guarded-v1 remains preserved as a negative/experimental result. It was too
   conservative and expensive and is not the new throughput architecture.
2. Balanced-lite warned on only 8/13 predefined tails overall and 6/8 locked
   validation tails. It is a bounded quality hint, not an all-failure detector
   or safety guarantee.
3. GPU cold start is material. On the canonical 10k/150 probe, fresh-process
   GPU latency exceeded loaded CPU throughput latency. Persistent workloads are
   the intended GPU-advantage region.
4. Pinned staging and two-buffer H2D prefetch are active per call when
   allocation succeeds, but pinned host buffers are not yet reused across
   persistent-worker calls.
5. T6 is deterministic synthetic scaling evidence and has no biological
   continuity claim.
6. T7 uses single source trajectory files (4,187-9,001 frames), not stitched
   trajectories. Available metadata does not independently prove PBC/nojump or
   uninterrupted physical sampling provenance.
7. MDLovoFit was not assigned a warm in-memory T7 time because its
   subprocess/PDB I/O boundary is unmatched. No 71x or universal-fastest claim
   is supported.
8. GPU validation currently covers one Quadro P5000/CUDA 12/CuPy environment.
   Portability and crossover on newer GPUs are unmeasured.
9. `cusolver_svd` currently names the CuPy batched SVD/cuSOLVER path; it is not
   a separate custom kernel.
10. Hard motion-domain segmentation and sequential_lean were not ported to GPU.
11. W5 scan-only output is not an equivalent alignment workflow and must not be
   directly benchmarked against MDLovoFit as though it were.
"""
    (HANDOFF / "04_negative_results_and_limits.md").write_text(limitations)

    inventory_roots = (
        RESULTS / "t0_freeze",
        RESULTS / "t2_quality_metadata",
        RESULTS / "t3_balanced_lite",
        RESULTS / "t4_gpu_environment",
        RESULTS / "t5_gpu_design",
        RESULTS / "t6_scaling",
        RESULTS / "t7_real_long",
    )
    inventory_rows = []
    for directory in inventory_roots:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "input_cache" in path.parts:
                continue
            inventory_rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_table(HANDOFF / "05_output_inventory.tsv", inventory_rows)

    readme = f"""# T0-T7 throughput/GPU handoff

## Bottom line

T0-T7 are complete. `sequential_lean` remains the public exact/default CPU
path. Pure fast-global is now an explicit throughput contract; balanced-lite
adds only a fixed sampled audit and at most one bounded refinement, with no
automatic exact fallback. W5 remains an experimental alignment-free local
deformation diagnostic.

The isolated CuPy backend passed numerical implementation checks and the full
GPU-enabled test suite. On the synthetic scaling matrix, all
`{t6['GPU_case_count']}` GPU throughput cases met both stated engineering
targets. On three real single-file long trajectories, warm GPU throughput was
faster than CPU throughput while retaining exact-relative core Jaccard
0.974-1.000 and RMSF Spearman above 0.9998.

These results do not support a universal-fastest claim. GPU cold start matters,
balanced-lite is not a perfect tail detector, and no unmatched in-memory
MDLovoFit timing is reported.

## Validation

- Canonical CPU: `244 passed, 6 skipped, 2 subtests passed`.
- Isolated CUDA 12: `250 passed`.
- Frozen integrity: `{integrity['status']}`
  (`{integrity['project_completion_checked']}/9` project and
  `{integrity['golden_checked']}/34` golden hashes).
- Git metadata changed: **no**.
- Public default changed: **no**.

## Read next

1. `01_mode_contracts.md`
2. `02_key_metrics.tsv`
3. `04_negative_results_and_limits.md`
4. `05_output_inventory.tsv`
"""
    (HANDOFF / "00_read_me_first.md").write_text(readme)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stages": ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"],
        "public_default_changed": False,
        "git_metadata_modified": False,
        "guarded_v1_preserved": True,
        "frozen_integrity": integrity,
        "tests": {
            "canonical_CPU": (
                "244 passed, 6 skipped, 2 subtests passed"
            ),
            "isolated_CUDA12": "250 passed",
        },
        "T7": t7,
        "handoff_files": [
            str(path.relative_to(ROOT))
            for path in sorted(HANDOFF.iterdir())
            if path.is_file()
        ],
    }
    (HANDOFF / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
