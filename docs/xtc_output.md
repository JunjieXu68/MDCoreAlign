# XTC Trajectory Output

Use `mdcorealign apply --purpose trajectory` to write all aligned frames to
XTC. Frame subsets and strides remain available, but all frames are the
trajectory-purpose default.

XTC export uses the standard-installation MDAnalysis dependency and the shared streaming transform engine.
Only one chunk of selected coordinates is aligned at a time. MDCoreAlign uses
Angstrom internally; the writer explicitly converts to format-native
nanometres before calling the MDAnalysis XDR backend. MDAnalysis converts back
to Angstrom through its high-level reader.

Available time, step, and unit-cell metadata are preserved. Each Cartesian box
vector is rotated by the frame rotation; translation is not applied to the
box. Missing unit cells remain missing. `--drop-unit-cell` records deliberate
omission in provenance.

The raw XTC 3x3 matrix retains this rotated orientation. MDAnalysis' high-level
`Timestep.dimensions` exposes box lengths/angles and canonicalizes global box
orientation on read. Users who need the exact rotated matrix should use the
MDAnalysis low-level XDR record or an MD tool that preserves the full matrix;
otherwise use `--drop-unit-cell` and document downstream PBC handling.

XTC is not self-describing. Keep the exact topology and atom ordering matching
the export selection. `export_atom_metadata.tsv` documents the selected order
but does not replace a topology.

The compact validation matrix covers PDB+XTC and GRO+XTC through MDAnalysis.
Other readable pairs are available in the MDAnalysis ecosystem but remain
outside the formally validated matrix until independently round-trip tested.
No extra installation step is required after installing the standard wheel.
