"""Create a small private-alpha real-MD CA-only PDB example from local data.

This helper intentionally requires explicit local topology/trajectory paths.
It does not download data and does not make benchmark claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("examples/real_md_small"), type=Path)
    parser.add_argument("--selection", default="protein and name CA")
    parser.add_argument("--n-frames", default=20, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--system-name", default="ADK small CA subset")
    return parser.parse_args()


def write_pdb_model(handle, atoms, model_index: int) -> None:
    """Write one CA-only PDB model."""

    handle.write(f"MODEL     {model_index:4d}\n")
    for serial, atom in enumerate(atoms, start=1):
        chain = (getattr(atom, "segid", "") or getattr(atom, "chainID", "") or "A")[:1]
        resid = int(getattr(atom, "resid", serial))
        resname = str(getattr(atom, "resname", "UNK"))[:3]
        name = str(getattr(atom, "name", "CA"))[:4]
        x, y, z = atom.position
        handle.write(
            f"ATOM  {serial:5d} {name:>4} {resname:>3} {chain:1s}{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}           C\n"
        )
    handle.write("TER\nENDMDL\n")


def main() -> int:
    """Generate the small example."""

    import MDAnalysis as mda

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = mda.Universe(str(args.topology), str(args.trajectory))
    atoms = universe.select_atoms(args.selection)
    if atoms.n_atoms == 0:
        raise ValueError(f"selection returned zero atoms: {args.selection}")
    trajectory_path = args.output_dir / "trajectory_small.pdb"
    topology_path = args.output_dir / "topology.pdb"
    with trajectory_path.open("w") as handle:
        count = 0
        for _ in universe.trajectory[:: args.stride]:
            write_pdb_model(handle, atoms, count + 1)
            count += 1
            if count >= args.n_frames:
                break
        handle.write("END\n")
    universe.trajectory[0]
    with topology_path.open("w") as handle:
        write_pdb_model(handle, atoms, 1)
        handle.write("END\n")
    metadata_path = args.output_dir / "subset_metadata.json"
    metadata_path.write_text(
        "{\n"
        f'  "system_name": "{args.system_name}",\n'
        f'  "selection": "{args.selection}",\n'
        f'  "n_frames": {count},\n'
        f'  "n_atoms": {atoms.n_atoms},\n'
        '  "purpose": "small private-alpha usability example, not a benchmark"\n'
        "}\n"
    )
    print(f"Wrote {trajectory_path}")
    print(f"Wrote {topology_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
