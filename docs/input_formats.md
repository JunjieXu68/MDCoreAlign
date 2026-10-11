# Input Formats

MDCoreAlign is primarily validated for protein MD trajectories with consistent
topology and residue mapping, using C-alpha fitting as the formal default.

Other MDAnalysis-readable formats may work if topology, atom order and residue
metadata are valid, but they are not yet formally validated in the MDCoreAlign
manuscript.

## Recommended And Formally Supported Inputs

| input | status | notes |
| --- | --- | --- |
| Topology + trajectory pairs with reliable residue metadata | Recommended | best route for real MD data |
| PDB + XTC | Validated for purpose-based apply I/O | streamed through MDAnalysis; metadata and atom order must remain valid |
| GRO + XTC | Validated for purpose-based apply I/O | streamed through MDAnalysis; common GROMACS workflow |
| PSF + DCD | Expected, not validated in the IO v1 matrix | backend-readable CHARMM/NAMD workflow |
| PRMTOP/PARM7 + NetCDF/NC | Expected, not validated in the IO v1 matrix | backend-readable AMBER workflow |
| Multi-model PDB | Validated for small examples or structural ensembles | built-in reader; atom order must be identical across models |
| Protein MD with consistent atom order | Formal target | missing residues or atom order changes must be fixed upstream |
| Protein C-alpha fitting | Formal default | reproducible and easiest to compare across tools |

## Backend-Readable But Experimental Inputs

The following formats may be readable through the standard MDAnalysis adapter
when topology and residue metadata are valid, but they are not yet formal
MDCoreAlign manuscript claims:

- TRR/TNG
- H5MD
- GSD
- LAMMPS dump / lammpstrj
- DMS
- MOL2
- XYZ
- other MDAnalysis-readable formats with valid topology and residue metadata

## Unsupported Or Not Recommended Inputs

- trajectory-only inputs without reliable topology;
- topology/trajectory atom count mismatch;
- variable atom-set trajectories;
- missing or inconsistent residue correspondence;
- raw wrapped or uncentered trajectories without preprocessing;
- RNA/DNA-only trajectories as a formal validated mode;
- protein-nucleic acid complexes as full-complex stable-core models;
- IDP/highly disordered trajectories where a stable core may not exist;
- subunit exchange or symmetry-swapping trajectories without fixed chain correspondence.

## Recommended CLI Style

Prefer explicit topology/trajectory arguments for real MD inputs:

```bash
mdcorealign audit-preprocessing \
  --topology topology.pdb \
  --trajectory trajectory.xtc \
  --fit-selection "protein and name CA"
```

For small examples and structural ensembles, a CA-only multi-model PDB remains
the simplest route:

```bash
mdcorealign flexibility \
  --input trajectory_ca.pdb \
  --input-format ca-pdb \
  --fit-selection "protein and name CA"
```

If format detection fails, use `--topology-format` and `--trajectory-format`.
If the backend still cannot read the files, convert to a conservative
PDB/XTC, GRO/XTC, PSF/DCD, PRMTOP/NC, CA-only PDB, or NumPy workflow.

## Validated Coordinate-Export Input Matrix

This matrix is specific to `mdcorealign apply` and was exercised with compact
local fixtures. It does not broaden the scientific benchmark claims.

| input pair | reader | metadata | coordinate export validation |
| --- | --- | --- | --- |
| PDB + XTC | MDAnalysis | atom/residue/chain from PDB; time/box from XTC | PDBx/mmCIF, XTC and compatibility PDB paths tested |
| GRO + XTC | MDAnalysis | atom/residue from GRO; time/box from XTC | XTC streaming and round-trip tested |
| CA multi-model PDB | built-in | PDB atom/residue/chain records | structure mmCIF and compatibility PDB tested |
| NumPy array | built-in | explicit synthetic CA/UNK rows | normal and compatibility PDB tested; XTC can use a synthetic topology companion, not a recovered real topology |

Other MDAnalysis-readable pairs remain available-but-unvalidated ecosystem
support until a dedicated fixture validates their metadata and round trip.

## Read Configuration and Replay

Source manifest 2.2 records the actual array key, original unit request and
conversion, topology and trajectory formats independently, fit selection,
chain restriction and original frame order. `apply` restores that configuration
before applying the saved transforms. Conflicting explicit overrides fail;
matching file hashes alone are insufficient to establish the same input read.
`--frames`, `--frame-range` and `--stride` on `apply` select export frames without
changing the saved transform-to-source-frame correspondence.

`--chain A` restricts fitting to chain A; a missing chain fails. The default
export still contains all available protein atoms, including other chains.
Protein selection uses the same documented residue-name policy in the built-in
PDB and MDAnalysis readers, including MSE in either ATOM or HETATM records.
Nucleic acids and arbitrary hetero residues do not become protein merely from
the record type. Ambiguous or unsupported selections are rejected.

Array rows have explicitly synthetic CA/UNK metadata, with no inferred residue
identity. Both ordinary and compatibility export retain that convention.
Nanometre arrays are converted once to the internal Angstrom unit; saved read
configuration ensures the same conversion during replay. Multi-array NPZ input
records the resolved key, not just a hash of the container.

For old manifest 2.1, array replay is refused when original units/key cannot be
proven; rerun `align` to obtain manifest 2.2. Standard PDB/MDAnalysis results can
retain the original default export only when source hashes, fit labels and
counts agree. A different export subset requires a new manifest with ordered
source/fit atom identities. This restriction preserves source validation.
