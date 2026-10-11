# Conversion Guidance

This document gives conservative input-preparation guidance. It does not
replace tested force-field or MD-engine conversion workflows.

## GROMACS

Recommended pair: `GRO`, `PDB`, or `TPR` topology with `XTC` or `TRR`
trajectory.

Preprocess nojump/unwrap/center before MDCoreAlign. Poor conversion can remove
chain or residue metadata, so inspect `input_audit.tsv` and
`chain_mapping_report.tsv`.

## AMBER

Recommended pair: `PRMTOP`/`PARM7` topology with `NC`/NetCDF or `MDCRD` where
supported.

Residue metadata must be preserved and atom order must match the topology.

## CHARMM/NAMD

Recommended pair: `PSF` or `PDB` topology with `DCD` trajectory.

Use a fixed protein selection and inspect chain/residue mapping before
biological interpretation.

## MDLovoFit-Style PDB Trajectory

Multi-model PDB can be used for small examples or structural ensembles if atom
order is consistent across frames/models. It is not recommended for large
production trajectories.

## Analysis-Only Files

`XVG`, `DAT`, and table-only `CSV` files cannot be used as MDCoreAlign
trajectory inputs because coordinates are required.

## DNA/RNA

RNA/DNA workflows are not formally validated. If used experimentally, define an
explicit atom selection, such as phosphate or sugar atoms, and do not interpret
results as validated protein stable-core analysis.

## Protein-Ligand Or Membrane Systems

Fit on protein C-alpha atoms. Apply the transform to protein or all atoms when
needed for visualization. Ligands, lipids, solvent, and ions are transformed
but are not part of stable-core inference by default.
