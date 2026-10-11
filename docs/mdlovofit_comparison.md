# MDLovoFit Input Comparison

This comparison is intended to clarify input handling, not to criticize
MDLovoFit. MDLovoFit remains the direct least-mobile-subset baseline for this
project.

## MDLovoFit Official Input Style

MDLovoFit is primarily documented around PDB trajectory input:

- input files are PDB trajectory file(s);
- each PDB file may contain multiple frames;
- frames are separated by `END` or `ENDMDL`;
- input PDB files appear at the end of the command line;
- users commonly export PDB trajectories from VMD or another converter.

## MDLovoFit Atom Selection

- The default atom selection is C-alpha atoms.
- Alternative atom names can be supplied with `-atoms`.
- Complex selections can be supplied with `-atomsfile`, where occupancy 0/1
  marks excluded/included atoms.

## MDLovoFit RNA/DNA Status

RNA/DNA is not documented as a formal MDLovoFit validation mode. MDLovoFit may
technically accept other atom selections if the input is a consistent PDB point
set, but this should be described as unvalidated rather than officially
supported.

## MDCoreAlign Contrast

MDCoreAlign aims to provide a more modern user-facing workflow while keeping
support claims conservative:

- accepts topology + trajectory style inputs where the Python backend can read
  valid metadata;
- separates `--fit-selection` from `--apply-selection`;
- provides input audit and dry-run modes;
- writes CSV/TSV/JSON outputs;
- supports robust RMSF and flexible segment outputs;
- provides chain mapping and preprocessing warnings;
- does not claim formal RNA/DNA support.

MDCoreAlign's formal validation remains protein C-alpha MD trajectory analysis
with consistent topology and residue mapping.

## Feature Table

| feature | MDLovoFit | MDCoreAlign | caveat |
| --- | --- | --- | --- |
| input style | PDB trajectory file(s) | topology+trajectory where readable; PDB/array examples | MDCoreAlign formal workflow still requires metadata validity |
| trajectory formats | PDB trajectory after conversion | PDB/array directly; common topology+trajectory pairs through optional backend | not all backend-readable formats are formally validated |
| default atom selection | C-alpha atoms | `protein and name CA` | both tools default to protein CA-style use |
| custom selection | `-atoms`, `-atomsfile` occupancy mask | `--fit-selection`, `--apply-selection` | MDCoreAlign selection support is deliberately conservative |
| topology metadata | primarily PDB metadata | topology metadata when backend-readable | metadata must be checked |
| protein-ligand transform | possible through PDB point sets/selections | fit protein CA, apply transform to protein/all atoms | ligand mobility is not formal output |
| RNA/DNA support | not documented as formal mode | not formally supported | future/experimental only with explicit selection |
| preprocessing audit | not primary focus | `audit-preprocessing`, `align --dry-run` | audit cannot prove PBC/nojump |
| output files | MDLovoFit-native outputs and visualization fields | CSV/TSV/JSON/PDB outputs | semantics are documented separately |
| diagnostics | fraction/fit behavior from MDLovoFit outputs | convergence/runtime/audit diagnostics | comparison should label native vs unified metrics |
| RMSF/flexible segments | not the primary interface | robust/ordinary RMSF and flexible segments | RMSF is alignment-frame dependent |
