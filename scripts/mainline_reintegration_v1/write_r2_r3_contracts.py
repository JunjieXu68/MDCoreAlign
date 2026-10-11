#!/usr/bin/env python3
"""Write the locked R2/R3 experimental contracts after implementation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "mainline_reintegration_v1"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    local_root = RESULTS / "local_deformation_integration"
    fast_root = RESULTS / "fast_global_integration"
    local_root.mkdir(parents=True, exist_ok=True)
    fast_root.mkdir(parents=True, exist_ok=True)

    (local_root / "02_local_deformation_contract.md").write_text(
        """# R2 local-deformation contract

Status: experimental, opt-in, alignment-free.

## Command

```bash
mdcorealign local-deformation \\
  --topology topology.pdb \\
  --trajectory trajectory.xtc \\
  --fit-selection "protein and name CA" \\
  --window-size 5 \\
  --output-dir output
```

The fixed W5 descriptor uses robust relative median absolute deviations of
squared CA pair distances for chain-aware local edges. Pair distances make the
descriptor translation and rotation invariant. Division by each edge's robust
distance scale makes the result dimensionless and invariant under global
coordinate scaling, within floating-point tolerance. Descriptor evaluation is
frame-chunked.

`local_deformation_index` is not RMSD and is not coherent global mobility.
`local_rigidity_score` is the complementary rank-mapped local score. Neither is
a biological domain label. Temporal-block agreement and residue confidence are
diagnostics, not guarantees.

Array units are never inferred from geometry. `--source-unit nanometer` applies
one explicit factor of 10; `auto` preserves array values and marks their
canonical unit unknown. PDB and backend coordinates carry explicit provenance.

The public exact/default alignment path is unchanged.
"""
    )
    local_schema = [
        ("atom_index", "integer", "one-based output row index"),
        ("residue_id", "string", "identifier parsed from topology metadata"),
        ("resname", "string", "residue name parsed from topology metadata"),
        ("chain", "string", "chain or segment identifier"),
        ("residue_position", "integer", "numeric topology residue position"),
        (
            "local_deformation_index",
            "float",
            "dimensionless robust local pair-distance instability; not RMSD",
        ),
        (
            "local_rigidity_score",
            "float",
            "rank-mapped local rigidity score in [0,1]",
        ),
        ("percentile", "float", "deformation percentile in [0,1]"),
        ("rank", "integer", "descending deformation rank"),
        (
            "locally_deforming_segment_id",
            "integer",
            "chain-aware contiguous segment ID; zero is unselected",
        ),
        ("is_terminal", "integer", "one at a chain terminal"),
        (
            "chain_break_before",
            "integer",
            "one when chain or consecutive residue continuity breaks",
        ),
        (
            "temporal_block_reproducibility",
            "float",
            "mean pairwise block Spearman for the complete profile",
        ),
        ("confidence", "float", "support and temporal-rank confidence"),
        ("uncertainty", "float", "one minus confidence"),
        ("source_unit", "string", "declared or backend source-unit status"),
        (
            "conversion_factor_to_angstrom",
            "float",
            "explicit factor applied before analysis",
        ),
        (
            "canonical_internal_unit",
            "string",
            "angstrom when resolved, otherwise unknown",
        ),
    ]
    write_tsv(
        local_root / "02_local_deformation_output_schema.tsv",
        [
            {"field": field, "type": dtype, "meaning": meaning}
            for field, dtype, meaning in local_schema
        ],
    )
    (local_root / "02_motion_graph_experimental_contract.md").write_text(
        """# R2 sparse motion-graph experimental contract

The graph is disabled unless `--experimental-motion-graph` is supplied.

Primary outputs are soft edge co-motion affinity, edge instability,
residue-level hinge-boundary probability and local consensus confidence.
An exploratory partition ID may be written for audit purposes only. It is not
promoted as a reproducible biological domain annotation because real-trajectory
hard-boundary reproducibility was weak in the W0-W5 pilot.

No public default, stable-core definition or exact alignment path depends on
the sparse graph.
"""
    )

    (fast_root / "03_fast_global_contract.md").write_text(
        """# R3 fast-global contract

`mdcorealign align --mode fast-global` is an explicit experimental mode.
The default remains the exact sequential robust trajectory solver.

Fast-global uses deterministic uniform temporal stratification (default 256
representatives), a fixed four-round shared-weight robust update, and then an
explicit exact weighted Kabsch fit for every frame. Reported transforms,
post-fit RMSD and robust/ordinary RMSF are therefore computed on all frames for
the estimated fixed weights. Weight estimation is approximate; transform
identity with the exact default is not claimed.

No automatic fallback is implemented. Users receive temporal-block agreement,
representative-subset agreement, effective core size, final weight change,
representative tolerance fraction and warning flags. These diagnostics have
not yet been calibrated for fallback sensitivity/specificity.

Timing includes confidence diagnostics and aligned-coordinate materialisation
when the CLI writes an aligned trajectory. Comparisons must use matched output
contracts.
"""
    )
    confidence_schema = [
        (
            "temporal_block_weight_spearman",
            "float|null",
            "agreement between independently estimated first/second-half weights",
        ),
        (
            "temporal_block_core_jaccard",
            "float|null",
            "top-60% core overlap between temporal halves",
        ),
        (
            "representative_subset_weight_spearman",
            "float|null",
            "agreement between odd/even representative subsets",
        ),
        (
            "representative_subset_core_jaccard",
            "float|null",
            "top-60% core overlap between representative subsets",
        ),
        (
            "effective_core_size",
            "float",
            "sum(weights)/max(weights), not a hard core count",
        ),
        (
            "final_round_weight_change",
            "float",
            "maximum shared-weight change in the final fixed round",
        ),
        (
            "representative_converged_fraction",
            "float",
            "representatives within the numerical tolerance in the final round",
        ),
        (
            "convergence_status",
            "string",
            "fixed-budget stability classification",
        ),
        ("warning_flags", "list[string]", "diagnostic warnings; no auto fallback"),
    ]
    write_tsv(
        fast_root / "03_fast_global_confidence_schema.tsv",
        [
            {"field": field, "type": dtype, "meaning": meaning}
            for field, dtype, meaning in confidence_schema
        ],
    )
    (fast_root / "03_diagnostic_rich_contract.md").write_text(
        """# R3 diagnostic-rich contract

`--mode diagnostic-rich` composes experimental fast-global alignment with the
alignment-free W5 local-deformation profile. The sparse graph remains optional
behind `--experimental-motion-graph`.

This is a diagnostic-rich mode, not the fastest mode. Global RMSD/RMSF and
transforms come from the explicit all-frame fast-global fit. W5 retains its
separate dimensionless local-deformation semantics and is never relabelled as
RMSD or global mobility.
"""
    )
    status = {
        "r2": {
            "status": "implemented_pending_full_suite",
            "public_default_changed": False,
            "command": "mdcorealign local-deformation",
            "motion_graph_default": False,
        },
        "r3": {
            "status": "implemented_pending_full_suite",
            "public_default_changed": False,
            "fast_global_default": False,
            "automatic_fallback": False,
        },
    }
    (RESULTS / "02_03_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
