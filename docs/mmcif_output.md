# PDBx/mmCIF Structure Output

Use `mdcorealign apply --purpose structure` for one structure or an explicitly
selected structural ensemble. Frame 0 is the default. Select multiple models
with `--frames`, `--frame-range`, `--stride`, or
`--all-selected-frames`.

The writer uses the standard-installation Gemmi dependency and emits a standard `_atom_site` loop. Atom and
residue names, auth and label identifiers, insertion codes, alternate
locations, elements, ATOM/HETATM records, coordinates, and deterministic model
numbers are retained where available. Source frame indices are stored
separately in `_mdcorealign_frame`, so model numbers are not presented as MD
frame numbers.

The `_mdcorealign_export` category records source and transform provenance.
Occupancy is `1.00` and B factor is `0.00`; neither field is overloaded with
stability, RMSF, or W5 values. Those values remain in analysis tables.

mmCIF coordinates are Angstrom. Per-model Cartesian box vectors, when present,
are stored in `_mdcorealign_box_vector`. Large text output is refused unless
`--allow-large-text-output` is explicit.

No extra installation step is required after installing the standard wheel.
