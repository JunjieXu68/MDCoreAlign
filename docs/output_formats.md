# Output Formats

## Compact Analysis Layout

The default `align` result root is:

```text
summary.json
residue_profile.tsv
frame_metrics.tsv
transforms.npz
flexible_segments.tsv
diagnostics/
```

`diagnostics/` contains only generated audit/provenance files, normally
`run_metadata.json`, `timing.json`, `input_audit.tsv`,
`chain_mapping_report.tsv`, `selected_frame_indices.tsv`, and
`diagnostics.json`. Optional W5 files are also placed there.

`summary.json` lists `primary_outputs` and `diagnostic_outputs` with paths
relative to the result root. `residue_profile.tsv` contains stable weights,
mobility scores, fixed-fraction core labels, robust RMSF, residuals and segment
labels. `ordinary_RMSF` is present only when `--ordinary-rmsf` is requested.

`frame_metrics.tsv` contains full-set and weighted RMSD after the explicit
per-frame fit. `transforms.npz` contains one rotation and translation per
frame. RMSF is alignment-frame dependent; mobility score is model-derived and
is not identical to RMSF.

Deprecated `weights.csv` and duplicate flat audit files are available only
with `--legacy-artifacts`.

## Purpose-Based Coordinate Outputs

| purpose | default format | intended use |
| --- | --- | --- |
| `structure` | PDBx/mmCIF | structure or selected structural ensemble |
| `trajectory` | XTC | full aligned MD trajectory |
| `compatibility` | multi-model PDB | legacy interoperability |

All coordinate writers use the shared bounded-memory transform-application
engine. Coordinate export time is downstream materialization overhead, not
compact analysis time.
