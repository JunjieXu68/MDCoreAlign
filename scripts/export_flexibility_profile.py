"""Export RMSF and flexible-segment profiles from a trajectory-like input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdsuperpose import (  # noqa: E402
    build_flexibility_profile,
    load_ca_multimodel_pdb_trajectory,
    load_trajectory_from_array_file,
    write_flexibility_profile_csv,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Export ordinary-vs-robust RMSF and flexible segment calls."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input trajectory: .npy/.npz coordinate array or multi-model PDB.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path for the flexibility profile.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "array", "ca-pdb"),
        default="auto",
        help="Input format. auto uses file extension.",
    )
    parser.add_argument(
        "--npz-key",
        default=None,
        help="Optional coordinate key for .npz inputs.",
    )
    parser.add_argument(
        "--chain-id",
        default=None,
        help="Optional chain ID filter for CA-only multi-model PDB input.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("first_frame", "previous_frame"),
        default="first_frame",
        help="Reference mode for robust alignment.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable warm start across frames.",
    )
    parser.add_argument(
        "--rule",
        choices=("gaussian", "huber", "tukey"),
        default="huber",
        help="Robust weighting rule.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help="Huber delta parameter.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma parameter.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=4.685,
        help="Tukey cutoff parameter.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum robust-alignment iterations per frame.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance.",
    )
    parser.add_argument(
        "--flexible-percentile-cutoff",
        type=float,
        default=0.8,
        help="Percentile cutoff for flexible-segment candidates.",
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=3,
        help="Minimum contiguous residues for a flexible segment.",
    )
    return parser


def _resolve_input_format(path: Path, requested: str) -> str:
    """Resolve auto input format."""

    if requested != "auto":
        return requested
    if path.suffix.lower() in {".npy", ".npz"}:
        return "array"
    if path.suffix.lower() in {".pdb", ".ent"}:
        return "ca-pdb"
    raise ValueError("could not infer input format; pass --input-format explicitly")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    input_format = _resolve_input_format(input_path, args.input_format)

    if input_format == "array":
        trajectory = load_trajectory_from_array_file(input_path, key=args.npz_key)
        residue_labels = None
    else:
        trajectory, residue_labels = load_ca_multimodel_pdb_trajectory(
            input_path,
            chain_id=args.chain_id,
        )

    robust_parameters: dict[str, float | int] = {
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
    }
    if args.rule == "huber":
        robust_parameters["delta"] = args.delta
    elif args.rule == "gaussian":
        robust_parameters["sigma"] = args.sigma
    elif args.rule == "tukey":
        robust_parameters["c"] = args.c

    profile = build_flexibility_profile(
        trajectory,
        residue_labels=residue_labels,
        reference_mode=args.reference_mode,
        warm_start=not args.no_warm_start,
        rule=args.rule,
        flexible_percentile_cutoff=args.flexible_percentile_cutoff,
        min_segment_length=args.min_segment_length,
        **robust_parameters,
    )
    write_flexibility_profile_csv(profile, args.output_csv)
    print(f"Wrote flexibility profile: {args.output_csv}")
    print(f"Residues/atoms: {profile.n_residues}")
    print(f"Flexible segments: {max(profile.flexible_segment_id.tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
