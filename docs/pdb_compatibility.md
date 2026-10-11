# Multi-Model PDB Compatibility Output

Use `mdcorealign apply --purpose compatibility` for legacy consumers, compact
smoke tests, and historical external-tool reproduction. Omitting both purpose
and format keeps this established PDB behavior for old valid commands.

The compatibility writer now consumes the shared streaming transform engine
without constructing a complete aligned trajectory. Existing CA-only array
formatting is byte-identical in the focused regression test, and PDB metadata
from built-in PDB input is preserved.

PDB remains constrained by fixed fields: one-character chain IDs, four-column
residue numbers, five-column atom serials, 8.3 coordinates, and four-column
model identifiers. MDCoreAlign refuses unsafe overflow instead of silently
truncating or wrapping identifiers. Large text output is guarded.

Use PDBx/mmCIF for large or metadata-rich structures and XTC for full
trajectories. Compatibility PDB does not preserve trajectory time or unit-cell
metadata; this is documented in `coordinate_export_provenance.json`.
