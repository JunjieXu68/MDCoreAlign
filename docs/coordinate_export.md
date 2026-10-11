# VT02 Annotated Coordinate Export

The normal interface is suffix driven and reuses validated provenance from the
alignment result:

```bash
mdcorealign apply RESULT_DIR -o aligned.pdb
mdcorealign apply RESULT_DIR -o aligned.cif
mdcorealign apply RESULT_DIR -o aligned.xtc
```

`align` writes `diagnostics/source_manifest.json` and does not materialize
coordinates. `apply` verifies source hashes, frame count, the saved fit
selection and fit-atom order before writing. Moved sources can be supplied with
the advanced `--topology`, `--trajectory`, or `--input` overrides; a hash
mismatch is rejected.

The default fit selection is protein C-alpha. The default export selection is
the complete protein when those atoms exist. The C-alpha-derived transform is
applied to every exported protein atom. CA-only input remains CA-only and emits
an explicit warning; absent atoms are never reconstructed.

Annotated PDB and mmCIF use occupancy for the binary reporting-core label and
B-factor for stable-core-aligned robust RMSF in angstrom. The fields are
analysis annotations, not experimental quantities. If exported residues lack
analysis values after a chain-restricted fit, source occupancy/B-factor values
are preserved throughout and the export warns; it does not invent RMSF. mmCIF also includes the
complete `_mdcorealign_residue_profile` category and preserves original
occupancy/B values under `_mdcorealign_source_atom_metadata`.

XTC writes all frames by default and automatically creates a matching
annotated topology PDB. When PDB metadata limits would lose information, the
companion is mmCIF. An adjacent `.mdcorealign.json` sidecar records hashes,
selections, units, frame/atom counts and the companion path.

MDCoreAlign separates compact scientific analysis from coordinate
materialization. `align` estimates stability weights and writes explicit per-frame
rigid transforms. `apply` later consumes those transforms through one shared
chunked engine and writes coordinates for a declared purpose.

## Decision Table

| question | choice | default output |
| --- | --- | --- |
| Do you need one annotated structure? | `-o aligned.pdb` or `-o aligned.cif` | frame 0 |
| Do you need selected annotated structural models? | add `--frames` or `--all-frames` | PDB/mmCIF ensemble |
| Do you need a full MD trajectory? | `-o aligned.xtc` | XTC, all frames, annotated topology |
| Does a workflow require multi-model compatibility PDB? | advanced legacy compatibility controls | multi-model PDB |

The output format may be overridden only for a supported combination.
Structure export supports mmCIF and explicit PDB. Full trajectory export uses
XTC. Compatibility export uses PDB. It preserves coordinate/model layout but does
not promise old bytes where they encoded fabricated ALA residue identities;
pure arrays now use synthetic UNK consistently. This avoids silently generating enormous
text trajectories.

## Streaming Contract

The source reader yields at most `--chunk-size` selected frames. The shared
engine applies `x_aligned = x R^T + t` for each frame and immediately passes
the aligned chunk to the writer. The complete aligned trajectory is never
constructed in memory before writing. For topology+trajectory input,
MDAnalysis also decodes only the selected chunks. The built-in multi-model PDB
and NumPy readers retain their compact input array but still avoid a second
full aligned array.

The transform archive must match the original frame count, fit atom count, fit
labels, topology, selection, and frame order. The exporter validates finite,
right-handed orthogonal rotations before writing.

## Units, Time, And Unit Cell

MDCoreAlign uses Angstrom internally. PDB and mmCIF coordinates are written in
Angstrom. XTC stores nanometres by format convention; the writer performs an
explicit Angstrom-to-nanometre conversion before calling the MDAnalysis XDR
backend, and MDAnalysis exposes Angstrom through its high-level reader.

For XTC, available time, step, and unit-cell vectors are preserved. A rigid
rotation is also applied to the three Cartesian box vectors. Translation does
not affect the box. Use `--drop-unit-cell` only when omission is intentional.
If the source has no valid unit cell, the output records that absence rather
than inventing a box.

The raw XDR file retains the consistently rotated 3x3 box matrix. MDAnalysis
high-level `Timestep.dimensions` may canonicalize the box orientation.
`--drop-unit-cell` remains available. Coordinate export does not independently
prove upstream PBC/nojump correctness.

mmCIF records selected-frame time/step and per-model Cartesian box vectors in
explicit `_mdcorealign_*` provenance categories. Compatibility PDB does not
preserve trajectory time or box metadata; this limitation is recorded in the
JSON provenance.

## Metadata And Provenance

PDBx/mmCIF preserves atom, residue, insertion-code, chain, segment and element
metadata while using occupancy and B-factor for the registered visualization
annotations. XTC is automatically paired with a matching annotated topology.
Every run writes:

- `apply_summary.json`;
- `coordinate_export_provenance.json` with input, transform, and output hashes;
- `selected_frames.tsv`;
- `export_atom_metadata.tsv`;
- `run_metadata.json`.

## Known Limits

PDB is a visualization and compatibility format with fixed-width limits. It cannot safely preserve
multi-character chain IDs, large residue numbers, more than 99,999 atoms, more
than 9,999 model identifiers, or coordinates outside its fixed 8.3 fields.
Large text output also requires an explicit override. Use mmCIF for large or
metadata-rich structures and XTC for trajectories.

PBC/nojump correctness remains an upstream preprocessing responsibility.
Preserving or rotating a recorded unit cell does not prove that the input was
properly unwrapped, centred, or made whole.

## Source Identity and Subset Export

Manifest 2.2 separates the source topology from the requested export subset.
Default full-protein export and a valid `--apply-selection "protein and name CA"`
use the same saved transforms. Source hashes, ordered atom identities and fit
identity remain checked; empty selections, changed source coordinates and
inconsistent replay options fail before writing coordinates.

A chain-restricted fit can leave other exported residues without RMSF in the
analysis profile. In that case, export preserves the source occupancy/B-factor
values for the entire structure and emits an explicit warning. It does not
invent RMSF for unanalysed residues. The analysis profile remains the record of
computed values. Pure arrays instead carry explicit synthetic CA/UNK metadata.
See [input_formats.md](input_formats.md) for unit/key replay and old-manifest
compatibility limits.
