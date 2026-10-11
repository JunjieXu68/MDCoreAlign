"""Evaluate THESEUS-aligned outputs with the MDCoreAlign unified RMSD protocol."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mdsuperpose import solve_rigid_alignment  # noqa: E402
from mdsuperpose.mdlovofit import read_multimodel_pdb_coords  # noqa: E402
from submission_critical_common import FORMAL_SYSTEMS, load_formal_system, write_table  # noqa: E402


SYSTEM_TO_KEY = {
    "I-FABP": "ifabp",
    "ADK": "adk_equilibrium",
    "ADK equilibrium": "adk_equilibrium",
    "NhaA": "nhaa_equilibrium",
    "NhaA equilibrium": "nhaa_equilibrium",
    "YiiP short": "yiip_equilibrium_short",
    "YiiP equilibrium short": "yiip_equilibrium_short",
}

STEMS = {
    "ifabp": "ifabp",
    "adk_equilibrium": "adk",
    "nhaa_equilibrium": "nhaa",
    "yiip_equilibrium_short": "yiip_short",
}


def frame_rmsd(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return per-frame RMSD to a fixed reference."""

    squared = np.sum((coords - reference[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.mean(squared, axis=1))


def orient_aligned_to_reference(aligned: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove arbitrary THESEUS global frame by fitting the first aligned model."""

    transform = solve_rigid_alignment(aligned[0], reference)
    rotation = transform.rotation
    translation = transform.translation
    return aligned @ rotation.T + translation


def evaluate_system(system_key: str, aligned_pdb: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate one aligned THESEUS PDB."""

    system = load_formal_system(system_key)
    original = system.trajectory.coords
    aligned = read_multimodel_pdb_coords(aligned_pdb, n_atoms=original.shape[1])
    if aligned.shape != original.shape:
        raise ValueError(f"shape mismatch for {system.label}: aligned={aligned.shape}, original={original.shape}")

    oriented = orient_aligned_to_reference(aligned, original[0])
    rmsd = frame_rmsd(oriented, original[0])
    summary = {
        "system": system.label,
        "aligned_pdb": aligned_pdb,
        "n_frames": int(aligned.shape[0]),
        "n_residues": int(aligned.shape[1]),
        "mean_unified_posthoc_rmsd": float(np.mean(rmsd)),
        "median_unified_posthoc_rmsd": float(np.median(rmsd)),
        "std_unified_posthoc_rmsd": float(np.std(rmsd, ddof=0)),
        "min_unified_posthoc_rmsd": float(np.min(rmsd)),
        "max_unified_posthoc_rmsd": float(np.max(rmsd)),
        "reference_protocol": "THESEUS output globally reoriented by fitting aligned model 1 to original frame 1; RMSD then computed to original frame 1",
        "status": "metric_evaluation_success",
    }
    per_frame = [
        {
            "system": system.label,
            "frame_index": index,
            "unified_posthoc_rmsd": float(value),
        }
        for index, value in enumerate(rmsd)
    ]
    return summary, per_frame


def find_default_aligned_pdb(runs_dir: Path, system_key: str) -> Path | None:
    """Find a likely THESEUS superposed PDB for one system."""

    run_dir = runs_dir / STEMS[system_key]
    candidates = list(run_dir.glob("*_sup.pdb")) + list(run_dir.glob("*sup*.pdb"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (len(path.name), path.name))[0]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEM_TO_KEY), default=None)
    parser.add_argument("--aligned-pdb", type=Path, default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "results" / "reviewer2_hardening" / "theseus_rerun" / "runs",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=ROOT / "results" / "reviewer2_hardening" / "theseus_rerun" / "tables",
    )
    return parser.parse_args()


def main() -> int:
    """Evaluate one or all systems."""

    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    systems = [SYSTEM_TO_KEY[args.system]] if args.system else list(FORMAL_SYSTEMS)
    summary_rows: list[dict[str, object]] = []
    per_frame_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for key in systems:
        aligned_pdb = args.aligned_pdb if args.aligned_pdb else find_default_aligned_pdb(args.runs_dir, key)
        if aligned_pdb is None or not aligned_pdb.exists():
            failures.append(f"{key}: missing aligned THESEUS PDB")
            continue
        try:
            summary, per_frame = evaluate_system(key, aligned_pdb)
        except Exception as exc:  # noqa: BLE001 - report exact parse/evaluation failures
            failures.append(f"{key}: {exc}")
            continue
        summary_rows.append(summary)
        per_frame_rows.extend(per_frame)

    if summary_rows:
        write_table(
            args.tables_dir / "theseus_unified_rmsd_summary.tsv",
            summary_rows,
            list(summary_rows[0].keys()),
        )
    if per_frame_rows:
        write_table(
            args.tables_dir / "theseus_per_frame_rmsd.tsv",
            per_frame_rows,
            list(per_frame_rows[0].keys()),
        )
    if failures:
        failure_path = args.tables_dir / "theseus_unified_rmsd_failures.txt"
        failure_path.write_text("\n".join(failures) + "\n")
        print(f"Evaluation completed with failures. See {failure_path}")
    else:
        print("THESEUS unified RMSD evaluation complete")
    return 0 if summary_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
